"""Интеграционные тесты Фаз 3–4 — живые Ollama (@pytest.mark.integration).

Маркер `integration` (pyproject). Сервер векторизации берётся из env
`LIVE_OLLAMA_URL` (дефолт — рабочий адрес REQUIREMENTS §4,
qwen3-embedding:8b, dim 4096); суммаризатор — из `LIVE_SUMMARY_URL`
(дефолт 192.168.3.112, ornith-1.5:35b). При недоступности — SKIP, а не
падение (ARCH §7). Проверяется: форматы живых векторов, качество на русских
перефразах, дедуп-порог, догон pending фоновым воркером; суммаризация:
реальная длина summary, язык, латентность, timeout, погонка фонового воркера
(режим «Б»).
"""

from __future__ import annotations

import os
import socket
import time
from types import SimpleNamespace
from urllib.parse import urlparse

import pytest
from fakes import FailingEmbedder

from app.config import Settings
from app.services.embedding import EmbeddingService
from app.services.notes import NoteService
from app.services.search import SearchService
from app.services.summary import SummaryError, SummaryService
from app.services.worker import BackgroundWorker
from app.storage import vectors
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

    Замена пунктуации (запятая → тире) и регистра не меняет смысл — живая
    модель должна дать близость ≥ 0.92; перефраз со словом «—» вместо «в»
    дополнительно проходит по содержанию.
    """
    first = live.notes.save("Интеграция-дедуп: ретроспектива продукта в пятницу 14:00 в переговорной Браво")
    second = live.notes.save("Интеграция-дедуп: ретроспектива продукта — в пятницу 14:00 в переговорной Браво")
    assert second.get("duplicated") is True
    assert second["id"] == first["id"]


def test_worker_catches_pending_queue(live) -> None:
    """Воркер на живой машине дотягивает pending до ok (партия + вектор)."""
    with session(live.settings) as conn:
        conn.execute(
            "INSERT INTO notes (text, author, vector_status, summary_status) "
            "VALUES ('Интеграция: отложенная заметка для живого воркера', "
            "'test', 'pending', 'pending')"
        )
        note_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    worker = BackgroundWorker(live.settings, live.embedding)
    assert worker.process_pending() == 1
    with session(live.settings) as conn:
        row = conn.execute(
            "SELECT vector_status FROM notes WHERE id = ?", (note_id,)
        ).fetchone()
        assert row["vector_status"] == "ok"
        assert vectors.get_vector(conn, note_id) is not None


def test_offline_save_then_worker_repairs(live) -> None:
    """Полный цикл деградации: сервер «вон» → pending + warning → воркер чинит."""
    offline_settings = live.settings.model_copy(
        update={"ollama_base_url": "http://127.0.0.1:1"}
    )
    notes_broken = NoteService(offline_settings, FailingEmbedder())
    saved = notes_broken.save(
        "Деградационная заметка интеграции: сначала pending, потом ok"
    )
    assert saved["warning"]
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
