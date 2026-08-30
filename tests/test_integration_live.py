"""Интеграционные тесты Фаз 3–8 — живые Ollama (@pytest.mark.integration).

Маркер `integration` (pyproject). Сервер векторизации берётся из env
`LIVE_OLLAMA_URL` (дефолт — рабочий адрес REQUIREMENTS §4,
qwen3-embedding:8b, dim 4096); суммаризатор — из `LIVE_SUMMARY_URL`
(дефолт 192.168.3.112, ornith-1.5:35b). При недоступности — SKIP, а не
падение (ARCH §7). Проверяется: форматы живых векторов, качество на русских
перефразах, дедуп-порог, догон pending фоновым воркером; суммаризация:
реальная длина summary, язык, латентность, timeout, погонка фонового воркера
(режим «Б»). Фаза 8: save мгновенный — вектора/дедуп догоняет фоновый
воркер, поэтому перед проверками поиска и кандидатов гоняется
notes-очередь (`process_pending`).
"""

from __future__ import annotations

import asyncio
import httpx
import os
import socket
import time
from types import SimpleNamespace
from urllib.parse import urlparse

import pytest
from fakes import FailingEmbedder, cosine

from app.config import Settings
from app.services.dedup import DeduplicationService
from app.services.embedding import EmbeddingError, EmbeddingService
from app.services.notes import NoteService
from app.services.search import SearchService
from app.services.summary import SummaryError, SummaryService
from app.services.worker import BackgroundWorker
from app.storage import chunks, vectors
from app.storage.db import init_db, session

# Дефолт из REQUIREMENTS §4 (операторский адрес); перебить env при другом.
LIVE_URL_DEFAULT = "http://192.168.3.113:11434"
DIM = 4096  # qwen3-embedding:8b — REQUIREMENTS §8

# Живой суммаризатор — Фаза 4 (REQUIREMENTS §4/§5.5).
LIVE_SUMMARY_URL_DEFAULT = "http://192.168.3.112:11434"
LIVE_SUMMARY_MODEL_DEFAULT = "ornith-1.5:35b"

pytestmark = pytest.mark.integration


def _reachable(url: str, timeout: float = 2.0) -> bool:
    parsed = urlparse(url)
    try:
        with socket.create_connection(
            (parsed.hostname, parsed.port or 11434), timeout=timeout
        ):
            return True
    except OSError:
        return False


@pytest.fixture(scope="session")
def live(tmp_path_factory) -> SimpleNamespace:
    """Живой набор сервисов на доступной Ollama; иначе — SKIP."""
    url = os.environ.get("LIVE_OLLAMA_URL", LIVE_URL_DEFAULT)
    if not _reachable(url):
        pytest.skip(f"живая Ollama недоступна: {url}")
    db_path = tmp_path_factory.mktemp("live") / "notes.db"
    settings = Settings(
        ollama_base_url=url,
        summary_ollama_base_url=url,  # в Фазе 3 суммаризация не вызывается
        summary_model="unused-in-phase-3",
        mcp_auth_token="live-integration-token",
        db_path=str(db_path),
        embedding_dim=DIM,
    )
    init_db(settings)
    embedding = EmbeddingService(settings)
    return SimpleNamespace(
        settings=settings,
        embedding=embedding,
        notes=NoteService(settings, embedding),
        search=SearchService(settings, embedding),
    )


def test_embedding_shape_and_determinism(live) -> None:
    """/api/embed живой: размерность 4096; один текст — тот же вектор."""
    text = "интеграционная проверка формата живых векторов"
    vector = live.embedding.embed(text)
    assert len(vector) == DIM
    similarity = sum(a * b for a, b in zip(vector, live.embedding.embed(text)))
    assert similarity > 0.999  # модель детерминирована на одинаковых входах


def test_hybrid_search_finds_russian_paraphrase(live) -> None:
    """Ключевое свойство гибрида: перефраз найден вектором, чужое — не в топе."""
    unique = f"live-{os.getpid()}-"  # устойчивость к повторным прогонам
    saved = []
    for text in (
        f"{unique}ставка НДС выросла до двадцати процентов с двадцать четвёртого года",
        f"{unique}ретрит команды запланирован на третью неделю июня в Тбилиси",
        f"{unique}продакшен база PostgreSQL переехала на кластер pg15-prod",
    ):
        saved.append(live.notes.save(text)["id"])
    # Фаза 8: save мгновенный — вектора создаёт фоновый воркер; перед
    # поиском догоняем notes-очередь, иначе векторной половины гибрида нет.
    BackgroundWorker(live.settings, live.embedding).process_pending()
    result = live.search.search(
        "команда собирается в июне на выездное мероприятие в Грузии"
    )
    assert result["results"], result
    first = result["results"][0]
    assert first["id"] == saved[1]  # смысловая пара — заметка про ретрит
    # векторный hit валиден: не-null и не отрезан порогом (FR-1)
    assert first["cosine"] is not None
    assert first["cosine"] >= live.settings.score_threshold


def test_dedup_catches_paraphrase(live) -> None:
    """Почти дословный перефраз ловится порогом 0.92 (REQUIREMENTS FR-4).

    Фаза 8: косинус-дедуп фоновый (save ловит только дословные повторы),
    поэтому проверяем живую близость на входе дедупа: замена «в» на «—»
    (пунктуация) должна дать косинус-кандидата с cosine ≥ порога 0.92 —
    такого кандидата судья (или фоллбек Этапа 2.2) сводит в фоне.
    """
    first = live.notes.save("Интеграция-дедуп: ретроспектива продукта в пятницу 14:00 в переговорной Браво")
    second = live.notes.save("Интеграция-дедуп: ретроспектива продукта — в пятницу 14:00 в переговорной Браво")
    # Фаза 8: довекторизация обеих заметок notes-очередью воркера.
    BackgroundWorker(live.settings, live.embedding).process_pending()
    with session(live.settings) as conn:
        vector = vectors.get_vector(conn, second["id"])
    assert vector is not None
    candidates = DeduplicationService(live.settings).find_candidates(
        vector, exclude_id=second["id"]
    )
    pair = {candidate_id: cosine_value for candidate_id, cosine_value in candidates}
    assert pair[first["id"]] >= live.settings.dedup_similarity  # вход сведения
    # NFR-3: без суммаризатора слияния нет — обе заметки целы (trash пуст).
    with session(live.settings) as conn:
        for note_id in (first["id"], second["id"]):
            assert conn.execute(
                "SELECT deleted_at FROM notes WHERE id = ?", (note_id,)
            ).fetchone()[0] is None


def test_worker_catches_pending_queue(live, tmp_path_factory) -> None:
    """Воркер на живой машине дотягивает pending до ok (партия + вектор)."""
    # Свежая БД: Фаза 8 делает save мгновенным, в общей сессионной БД
    # копятся чужие pending — точность «== 1» держим на чистой базе.
    db = tmp_path_factory.mktemp("live-worker") / "notes.db"
    settings = live.settings.model_copy(update={"db_path": str(db)})
    init_db(settings)
    with session(settings) as conn:
        conn.execute(
            "INSERT INTO notes (text, author, vector_status, summary_status) "
            "VALUES ('Интеграция: отложенная заметка для живого воркера', "
            "'test', 'pending', 'pending')"
        )
        note_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    worker = BackgroundWorker(settings, live.embedding)
    assert worker.process_pending() == 1
    with session(settings) as conn:
        row = conn.execute(
            "SELECT vector_status FROM notes WHERE id = ?", (note_id,)
        ).fetchone()
        assert row["vector_status"] == "ok"
        assert vectors.get_vector(conn, note_id) is not None


def test_offline_save_then_worker_repairs(live) -> None:
    """Запись мгновенная (Фаза 8): заметка сразу с vector_status=pending;
    живой воркер доводит до ok."""
    offline_settings = live.settings.model_copy(
        update={"ollama_base_url": "http://127.0.0.1:1"}
    )
    notes_broken = NoteService(offline_settings, FailingEmbedder())
    saved = notes_broken.save(
        "Деградационная заметка интеграции: сначала pending, потом ok"
    )
    assert saved["stored"] is True
    with session(live.settings) as conn:
        assert conn.execute(
            "SELECT vector_status FROM notes WHERE id = ?", (saved["id"],)
        ).fetchone()[0] == "pending"
    worker = BackgroundWorker(live.settings, live.embedding)
    assert worker.process_pending() >= 1
    with session(live.settings) as conn:
        assert conn.execute(
            "SELECT vector_status FROM notes WHERE id = ?", (saved["id"],)
        ).fetchone()[0] == "ok"

# --- Фаза 4: живой суммаризатор (режим «Б») ----------------------------------


RU_NOTE = (
    "Интеграция-суммари: продакшен база PostgreSQL переехала на кластер "
    "pg15-prod (IP 192.168.3.50) 12 сентября 2026, downtime составил 90 "
    "секунд, владельцем миграции назначен Артём, откат не потребовался."
)
EN_NOTE = (
    "Deploy note: scoring service v2.3 was released to production on "
    "Friday, March 14, 2026, deploy window 22:30 UTC, owner Marina Klein, "
    "rollback plan kept in runbook #47."
)


def _live_summary_url() -> str:
    return os.environ.get("LIVE_SUMMARY_URL", LIVE_SUMMARY_URL_DEFAULT)


def _live_summary_model() -> str:
    return os.environ.get("LIVE_SUMMARY_MODEL", LIVE_SUMMARY_MODEL_DEFAULT)


@pytest.fixture(scope="session")
def live_summary(tmp_path_factory) -> SimpleNamespace:
    """Живой суммаризатор + прогрев (холодный старт может превышать 60 с).

    Пять попыток прогрева; если модель так и не ответила (например, удалённый
    хост недостижим по модели) — SKIP, чтобы не маскировать сбои под падения.
    """
    url = _live_summary_url()
    model = _live_summary_model()
    if not _reachable(url):
        pytest.skip(f"живая Ollama суммаризации недоступна: {url}")
    settings = Settings(
        ollama_base_url=os.environ.get("LIVE_OLLAMA_URL", LIVE_URL_DEFAULT),
        summary_ollama_base_url=url,
        summary_model=model,
        mcp_auth_token="live-summary-token",
        db_path=str(tmp_path_factory.mktemp("live-summary") / "notes.db"),
    )
    init_db(settings)
    service = SummaryService(settings)
    text = "Прогрев живого суммаризатора перед интеграционными проверками Фазы 4."
    for attempt in range(1, 4):
        t0 = time.monotonic()
        try:
            service.summarize(text)
            print(
                f"\n[live-summary] прогрев: попытка {attempt}, "
                f"{time.monotonic() - t0:.1f} с (модель остаётся в памяти 15 м)"
            )
            break
        except SummaryError:
            print(f"\n[live-summary] прогрев попытка {attempt} не удалась")
            time.sleep(10)
    else:
        pytest.skip("суммаризатор не прогрелся за 5 попыток (вероятно, холодный старт)")
    return SimpleNamespace(settings=settings, summary=service)


def test_live_summary_quality_and_language(live_summary) -> None:
    """Живая генерация: непустое, ≤ MAX_SUMMARY_CHARS, язык заметки сохранён."""
    for text, cyrillic_expected in ((RU_NOTE, True), (EN_NOTE, False)):
        summary = live_summary.summary.summarize(text)
        assert 0 < len(summary) <= live_summary.settings.max_summary_chars
        assert summary.strip() == summary  # без обёрточных пробелов
        has_cyrillic = any("\u0400" <= ch <= "\u04FF" for ch in summary)
        assert has_cyrillic == cyrillic_expected, summary


def test_live_summary_think_disabled(live_summary) -> None:
    """SUMMARY_THINK=false: "think": false — генерация работает, content полон."""
    settings_no_think = live_summary.settings.model_copy(
        update={"summary_think": False}
    )
    service_no_think = SummaryService(settings_no_think)
    try:
        summary = service_no_think.summarize(RU_NOTE)
        assert 0 < len(summary) <= settings_no_think.max_summary_chars
    finally:
        service_no_think.close()


def test_live_summary_latency_report(live_summary) -> None:
    """Латентность фоновой генерации: замер печатается (бриф Ф4 п.6)."""
    t0 = time.monotonic()
    summary = live_summary.summary.summarize(RU_NOTE)
    elapsed = time.monotonic() - t0
    print(f"\n[live-summary] латентность generate: {elapsed:.2f} с ({len(summary)} симв)")
    assert elapsed > 0
    assert len(summary) <= live_summary.settings.max_summary_chars


def test_live_summary_timeout_fails_fast(tmp_path_factory) -> None:
    """Клиентский таймаут SUMMARY_TIMEOUT_SEC: отказ влезает в бюджет."""
    url = _live_summary_url()
    assert _reachable(url)  # скипнут на уровне fixture, если сервер «вон»
    short = Settings(
        ollama_base_url=os.environ.get("LIVE_OLLAMA_URL", LIVE_URL_DEFAULT),
        summary_ollama_base_url=url,
        summary_model=_live_summary_model(),
        mcp_auth_token="live-summary-token",
        summary_timeout_sec=1,  # read меньше любой генерации 35B-модели
        db_path=str(tmp_path_factory.mktemp("live-timeout") / "notes.db"),
    )
    init_db(short)
    service = SummaryService(short)
    t0 = time.monotonic()
    with pytest.raises(SummaryError) as exc_info:
        service.summarize(RU_NOTE)
    print(f"\n[live-summary] отказ по таймауту за {time.monotonic() - t0:.2f} с")
    assert "недоступен" in str(exc_info.value) or "HTTP" in str(exc_info.value)
    service.close()


def test_live_worker_backfills_summary_mode_b(live_summary, tmp_path_factory) -> None:
    """Полный путь режима «Б» на живых серверах: save → воркер → pending → ok.

    Векторизация живая (вектор по полному тексту — retrieval не ждёт суммари),
    суммаризация — только из воркера; замер «up to ok» — бриф Ф4 п.6.
    """
    db = tmp_path_factory.mktemp("live-mode-b") / "notes.db"
    settings = live_summary.settings.model_copy(update={"db_path": str(db)})
    init_db(settings)
    embedding = EmbeddingService(settings)  # живой векторизатор (§4)
    summary = SummaryService(settings)      # живой суммаризатор (§5.5)
    notes = NoteService(settings, embedding)
    worker = BackgroundWorker(settings, embedding, summary)
    t0 = time.monotonic()
    saved = notes.save(RU_NOTE)
    assert saved["summary_pending"] is True
    # Фаза 8: векторизация — отдельная notes-очередь воркера (вектор по
    # полному тексту, retrieval не ждёт суммари).
    assert worker.process_pending() == 1
    assert worker.process_summary_pending() == 1
    elapsed = time.monotonic() - t0
    with session(settings) as conn:
        row = conn.execute(
            "SELECT summary, summary_status, vector_status FROM notes WHERE id = ?",
            (saved["id"],),
        ).fetchone()
        assert vectors.get_vector(conn, saved["id"]) is not None
    assert row["summary_status"] == "ok"
    assert row["vector_status"] == "ok"  # вектор по полному тексту, не по summary
    assert 0 < len(row["summary"]) <= settings.max_summary_chars
    print(
        f"\n[live-summary] режим «Б» до ok: {elapsed:.2f} с "
        f"(save+embedding+дедуп ~0.5–1.5 с, воркер догнал суммари)"
    )
    print(f"[live-summary] суммари воркера: {row['summary']}")
    embedding.close()
    summary.close()


# --- Фаза 7: чанк-индексация длинной заметки (живая Ollama) -------------------

CHUNKS_FACT = (
    "Контрольный факт Фазы 7: счётчик подшипника конвейера B-12 заменён "
    "17 марта 2027 по наряду 84031, исполнитель Пётр Хомяков, склад №4."
)
CHUNKS_QUERY = "наряд 84031 счётчик конвейера B-12"


def _chunks_note_text() -> str:
    """Заметка ~15k символов: контрольный факт — точно в середине текста
    (не в первом и не в последнем чанке)."""
    para = (
        "Логбук эксплуатации: резервные копии БД сходятся по расписанию, "
        "проверка дисковой подсистемы пройдена без замечаний. "
        "Мониторинг отдаёт штатные значения, отчёты выгружены в архив. "
        "Запланированные окна обслуживания не пересекаются с дежурством. "
    )
    half = para * 30
    text = half + CHUNKS_FACT + half
    assert 14000 <= len(text) <= 20000
    return text


@pytest.fixture(scope="session")
def live_chunks(tmp_path_factory) -> SimpleNamespace:
    """Живой векторизатор + дефолты §8, MAX_NOTE_CHARS=20000.

    Отдельная БД (tmp) — живые заметки фаз 3–4 не мешают ранжированию
    контрольного факта. Ollama недоступна — SKIP (skip расставит fixture).

    Прогрев обязателен: 8B-модель на CPU 113 грузится минуты; с
    read-таймаутом 720 с (решение О. 2026-08-30) дефолты §8 (32×3)
    влезают, подъёмки идут по канону как в проде.
    """
    url = os.environ.get("LIVE_OLLAMA_URL", LIVE_URL_DEFAULT)
    if not _reachable(url):
        pytest.skip(f"живая Ollama недоступна: {url}")
    settings = Settings(
        ollama_base_url=url,
        summary_ollama_base_url=url,
        summary_model="unused-in-phase-7",
        mcp_auth_token="live-chunk-token",
        db_path=str(tmp_path_factory.mktemp("live-chunks") / "notes.db"),
        embedding_dim=DIM,
        max_note_chars=20000,
    )
    init_db(settings)
    embedding = EmbeddingService(settings)
    for attempt in range(1, 3):  # модель 8B на CPU 113 грузится минуты
        try:
            embedding.embed("прогрев живого эмбеддинга перед чанковым тестом")
            break
        except EmbeddingError:
            print(f"\n[live-chunks] прогрев {attempt}: модель ещё не готова")
            time.sleep(5)
    return SimpleNamespace(
        settings=settings,
        embedding=embedding,
        notes=NoteService(settings, embedding),
        search=SearchService(settings, embedding),
    )


def _embed_full_slow(settings: Settings, text: str) -> list[float]:
    """Живой embed ПОЛНОГО текста с расширенным read-таймаутом.

    Ретро-справка для сравнения с Фазой 3: EmbeddingService держит
    read-таймаут 20 c, а 3.7k токенов одним запросом на стенде 113
    кодируются минуты — полный вектор в notes_vec может остаться pending.
    Для замера это не важно: тот же текст → тот же вектор (модель
    детерминирована), есть httpx-вызов напрямую, ollama /api/embed.
    """
    response = httpx.post(
        str(settings.ollama_base_url).rstrip("/") + "/api/embed",
        json={"model": settings.embedding_model, "input": text},
        timeout=httpx.Timeout(2.0, read=300.0),
    )
    response.raise_for_status()
    embeddings = response.json()["embeddings"]
    assert len(embeddings) == 1 and len(embeddings[0]) == DIM
    return embeddings[0]


def test_long_note_chunk_relevance(live_chunks) -> None:
    """Бриф §7: релевантность на длинной 15k-заметке, сравнение с Фазой 3.

    save → заметка сразу с чанками; чанки лежат в notes_chunks БЕЗ
    векторов (pending по анти-джойну, векторной строки нет). Этот момент —
    поведение Фазы 3: векторная сторона — вектор ПОЛНОГО текста (fallback),
    snippet от начала текста; его cosine печатаем как ретро-справку. Затем
    воркер довекторизует чанки живой Ollama, и тот же запрос по
    специфичному фрагменту из середины находит заметку через ЛУЧШИЙ чанк:
    cosine и snippet — из него.

    Замер стенда: 8B-модель на CPU кодирует полный 15k-текст ~2 мин — по
    решению О. (2026-08-30) read-таймаут EmbeddingService поднят до 720 с,
    поэтому полный вектор успевает прямо в sync-пути save (важно для
    дедупа перефразов); вектора чанков добирает воркер, поиск по факту
    из середины идёт через ЛУЧШИЙ чанк — косинус и snippet из него.
    """
    settings = live_chunks.settings
    text = _chunks_note_text()
    saved = live_chunks.notes.save(text)  # полный embed ~2 мин на CPU-113
    note_id = saved["id"]
    assert saved["stored"] is True
    assert saved.get("warning") is None  # read 720 с — полный текст успевает

    # Нейтральные заметки — чтобы «top-1» был осмысленным ранжированием.
    live_chunks.notes.save("Штатная заметка А: ретрит команды в мае, Тбилиси")
    live_chunks.notes.save("Штатная заметка Б: отчётный период Q3 закрывается")

    with session(settings) as conn:
        chunk_rows = chunks.get_note_chunks(conn, note_id)
        assert len(chunk_rows) >= 2  # заметка многочанковая
        # Все чанки длинной заметки pending (анти-джойн, статус-колонки
        # нет); счётчик по ВСЕЙ БД не вяжем — соседние мелкие save'ы на
        # медленном стенде могут эпизодически не успеть в read-таймаут
        # (их добьёт тот же воркер) и не мешают сценарию.
        assert all(
            chunks.get_vector(conn, chunk_id) is None
            for chunk_id, _idx, _text, _tokens in chunk_rows
        )
        pending_before = chunks.count_pending(conn)

    query = CHUNKS_QUERY
    query_vector = live_chunks.embedding.embed(query)

    # Ретро-справка Фазы 3: cosine запроса к вектору ПОЛНОГО текста,
    # записанному при save в notes_vec; если БД его не получила — прямой
    # живой вызов (_embed_full_slow), замер не зависит от состояний БД.
    with session(settings) as conn:
        full_vector = vectors.get_vector(conn, note_id)
    if full_vector is None:
        full_vector = _embed_full_slow(settings, text)
    retro_cosine = cosine(query_vector, full_vector)

    # Поведение Фазы 3 до векторизации чанков: если полный вектор успел
    # в sync-путь (быстрый стенд) — поиск даёт fallback-хит против него
    # (snippet от начала текста); на медленном стенде заметка вектора не
    # имеет и живёт в поиске только FTS-плечом до догонки воркером.
    retro = live_chunks.search.search(query)["results"]
    retro_hit = next((r for r in retro if r["id"] == note_id), None)
    retro_rank = (
        [r["id"] for r in retro].index(note_id) + 1 if retro_hit else None
    )
    if retro_hit and retro_hit["cosine"] is not None:
        assert retro_hit["cosine"] == pytest.approx(retro_cosine, abs=1e-3)
        assert retro_hit["snippet"] == text[: settings.snippet_chars]

    # Воркер довекторизует pending-чанки живой Ollama; вычитывающая
    # партия мала (batch×concurrency), выгребаем очередь по кругу.
    worker = BackgroundWorker(settings, live_chunks.embedding)
    vectorized = 0
    for _ in range(30):  # защита от вечного цикла
        processed = asyncio.run(worker.process_pending_chunks())
        vectorized += processed
        if not processed:
            break
    assert vectorized == pending_before
    with session(settings) as conn:
        assert chunks.count_pending(conn) == 0
        # вектора у всех чанков БД (длинной заметки + shorts после reuse):
        assert chunks.count_vectors(conn) == chunks.count_chunks(conn)

    # Тот же запрос: заметка найдена через ЛУЧШИЙ чанк.
    results = live_chunks.search.search(query)["results"]
    assert results and results[0]["id"] == note_id
    hit = results[0]
    scored = [
        (cosine(query_vector, live_chunks.embedding.embed(chunk_text)), chunk_id)
        for chunk_id, _idx, chunk_text, _tokens in chunk_rows
    ]
    best_cosine, best_chunk_id = max(scored)
    best_chunk_text = next(
        chunk_text
        for chunk_id, _idx, chunk_text, _tokens in chunk_rows
        if chunk_id == best_chunk_id
    )
    assert "наряду 84031" in best_chunk_text  # факт целиком в лучшем чанке
    assert hit["cosine"] == pytest.approx(best_cosine, abs=1e-3)
    assert hit["snippet"] == best_chunk_text[: settings.snippet_chars]
    assert hit["snippet"] != text[: settings.snippet_chars]
    print(
        f"\n[live-chunks] Фаза 3 (полный вектор): cosine={retro_cosine:.3f}, "
        f"rank={retro_rank if retro_rank else 'нет (ниже порога)'}; "
        f"Фаза 7 (лучший чанк): cosine={best_cosine:.3f}, top-1; "
        f"чанков={len(chunk_rows)}, чанк-векторов добито={vectorized}/{pending_before}"
    )
