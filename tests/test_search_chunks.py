"""SearchService и чанки (Фаза 7, шаг 4): агрегация, fallback, snippet.

Проверяются (brief §6): KNN по чанкам с агрегацией до заметок (лучший чанк
задаёт cosine и snippet), fallback на вектор полного текста для заметок без
готовых чанков, единственность заметки в выдаче (мелкая с reuse не
дублируется), фильтрация trash, порог сравнивается с cosine ЛУЧШЕГО чанка.
FTS-сторона — test_search_service, RRF-механика — test_search_hybrid.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fakes import HashEmbedder, cosine

from app.config import get_settings
from app.services.notes import NoteService
from app.services.search import SearchService
from app.services.splitter import count_tokens, encoding, token_windows
from app.storage import chunks, vectors
from app.storage.db import init_db, session, transaction

PROD_MAX_NOTE_CHARS = "20000"

DEFS = {"chunk_size": 1024, "chunk_overlap": 180, "chunk_min_target": 200}

RU_SENTENCES = (
    "LLM Second Brain — self-hosted MCP-сервер долговременной памяти. "
    "Сервис хранит заметки в SQLite и отдаёт их моделям через инструменты. "
    "Векторизация идёт через внешнюю Ollama с моделью qwen3-embedding:8b. "
    "Полнотекстовый поиск держит русские словоформы на токенизаторе trigram. "
    "Слияние источников делает Reciprocal Rank Fusion с константой 60. "
    "Суммаризация работает в фоне и не блокирует запись заметки. "
)

_FACT = (
    "Зарплатный реестр за август 2026 хранится на сервере appsrv "
    "в каталоге /srv/payroll-reports. "
)

_SOURCE_TOKENS: list[int] | None = None


def _source_tokens() -> list[int]:
    """Кэш токенов фикстуры (tiktoken-токенизация — не на каждый тест)."""
    global _SOURCE_TOKENS
    if _SOURCE_TOKENS is None:
        _SOURCE_TOKENS = encoding().encode(RU_SENTENCES * 400)
    return _SOURCE_TOKENS


def text_with_tokens(n_tokens: int) -> str:
    """Текст ровно из n_tokens токенов повторённой русской фикстуры."""
    return encoding().decode(_source_tokens()[:n_tokens])


def long_note_with_fact() -> str:
    """Многочанковая заметка (~2700 токенов) с уникальным фактом на
    токен-позиции ~1500 — внутрь окна chunk 2 (в chunk 1/3 не попадает)."""
    return text_with_tokens(1500) + _FACT + text_with_tokens(1200)


@pytest.fixture(autouse=True)
def _prod_limits(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Боевые MAX_NOTE_CHARS (20000) — многочанковые заметки в тестах."""
    monkeypatch.setenv("MAX_NOTE_CHARS", PROD_MAX_NOTE_CHARS)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _make_services(embedder: HashEmbedder) -> tuple[NoteService, SearchService]:
    settings = get_settings()
    return NoteService(settings, embedder), SearchService(settings, embedder)


@pytest.fixture
def services() -> tuple[NoteService, SearchService, HashEmbedder]:
    init_db(get_settings())
    embedder = HashEmbedder(get_settings().embedding_dim)
    notes, searcher = _make_services(embedder)
    return notes, searcher, embedder


def _vectorize_pending_chunks(embedder: HashEmbedder) -> int:
    """Довекторизовать pending-чанки (как это сделает воркер, шаг 5)."""
    with session(get_settings()) as conn, transaction(conn):
        pending = chunks.pending_chunks(conn, 10_000)
        for chunk_id, chunk_text in pending:
            chunks.upsert_vector(conn, chunk_id, embedder.embed(chunk_text))
    return len(pending)


def _insert_legacy_note(text: str) -> int:
    """Заметка эпохи до Фазы 7: notes_vec есть, строк в notes_chunks нет."""
    embedder = HashEmbedder(get_settings().embedding_dim)
    with session(get_settings()) as conn, transaction(conn):
        cursor = conn.execute(
            "INSERT INTO notes (text, author, vector_status) VALUES (?, 'x', 'ok')",
            (text,),
        )
        note_id = int(cursor.lastrowid or 0)
        vectors.upsert(conn, note_id, embedder.embed(text))
    return note_id


class TestAggregation:
    def test_fact_in_middle_found_via_best_chunk(
        self, monkeypatch
    ) -> None:
        """Сценарий брифа §1: запрос по факту из середины длинной заметки
        находит её; косинус — от лучшего чанка, snippet — из него же.
        Порог занижен (HashEmbedder на подстроке не даёт 0.35 — это зона
        живой Ollama, шаг 7); механика агрегации — та же."""
        monkeypatch.setenv("MAX_NOTE_CHARS", PROD_MAX_NOTE_CHARS)
        monkeypatch.setenv("SCORE_THRESHOLD", "0.0")
        get_settings.cache_clear()
        init_db(get_settings())
        embedder = HashEmbedder(get_settings().embedding_dim)
        notes, searcher = _make_services(embedder)
        text = long_note_with_fact()
        note_id = notes.save(text)["id"]
        expected_chunks = len(token_windows(count_tokens(text), **DEFS))
        assert _vectorize_pending_chunks(embedder) == expected_chunks
        results = searcher.search("зарплатный реестр appsrv")["results"]
        assert [r["id"] for r in results][:1] == [note_id]
        hit = results[0]
        with session(get_settings()) as conn:
            rows = chunks.get_note_chunks(conn, note_id)
        query_vector = embedder.embed("зарплатный реестр appsrv")
        scored = sorted(
            (
                (
                    cosine(query_vector, embedder.embed(row_text)),
                    row_text,
                )
                for _row_id, _idx, row_text, _tokens in rows
            ),
            reverse=True,
        )
        best_cosine, best_chunk_text = scored[0]
        assert "appsrv" in best_chunk_text  # факт целиком в лучшем чанке
        assert hit["cosine"] == pytest.approx(best_cosine, abs=1e-4)  # float32 в vec0
        assert hit["snippet"] == best_chunk_text[: get_settings().snippet_chars]
        assert hit["snippet"] != text[: get_settings().snippet_chars]

    def test_rank_by_best_chunk_cosine_dim8(self, monkeypatch) -> None:
        """Векторный источник ранжируется по cosine лучшего чанка: A (0.6)
        в выдаче, B (максимум 0.3 < порога) — векторного hit не имеет."""
        monkeypatch.setenv("EMBEDDING_DIM", "8")
        get_settings.cache_clear()
        init_db(get_settings())
        note_a, note_b = _notes_with_chunk_vectors_dim8(monkeypatch)
        query = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        searcher = SearchService(get_settings(), ScriptedEmbedder(8, [query]))
        results = searcher.search("Fundamentalt neuprofilieren", top_k=5)["results"]
        assert [r["id"] for r in results] == [note_a]
        assert results[0]["cosine"] == pytest.approx(0.6, abs=1e-6)
        assert results[0]["snippet"] == "а-два: лучший чанк с высокой близостью"[
            : get_settings().snippet_chars
        ]
        assert note_b not in [r["id"] for r in results]


class TestFallback:
    def test_legacy_note_without_chunks_found_by_full_vector(
        self, services
    ) -> None:
        """Заметка до Фазы 7 (нет notes_chunks) — fallback на notes_vec:
        найдена, cosine полного текста (1.0 на тождественном запросе),
        snippet от начала полного текста."""
        notes, searcher, _embedder = services
        text = "Легаси-заметка про ТаскФлоу: деплой прошёл 2026-08-29 без инцидентов"
        note_id = _insert_legacy_note(text)
        results = searcher.search(text)["results"]
        assert results[0]["id"] == note_id
        assert results[0]["cosine"] == pytest.approx(1.0, abs=1e-4)
        assert results[0]["snippet"] == text[: get_settings().snippet_chars]

    def test_pending_chunks_fall_back_to_full_vector(
        self, monkeypatch
    ) -> None:
        """Многочанковая заметка с pending-чанками ищется по полному вектору:
        cosine равен близости запроса к ПОЛНОМУ вектору, snippet — от начала
        текста (не чанка). Порог занижаем, т.к. подстрока короче заметки."""
        monkeypatch.setenv("MAX_NOTE_CHARS", PROD_MAX_NOTE_CHARS)
        monkeypatch.setenv("SCORE_THRESHOLD", "0.0")
        get_settings.cache_clear()
        init_db(get_settings())
        embedder = HashEmbedder(get_settings().embedding_dim)
        notes, searcher = _make_services(embedder)
        text = text_with_tokens(3000)
        note_id = notes.save(text)["id"]  # чанки pending, полный вектор ok
        with session(get_settings()) as conn:
            assert chunks.count_pending(conn) == len(
                token_windows(count_tokens(text), **DEFS)
            )
        query = "Полнотекстовый поиск держит русские словоформы"
        results = searcher.search(query)["results"]
        hit = next(r for r in results if r["id"] == note_id)
        # векторные кандидаты: заметка без готовых чанков → cosine ПОЛНОГО вектора
        assert hit["cosine"] == pytest.approx(
            cosine(embedder.embed(query), embedder.embed(text)), abs=1e-4
        )
        assert hit["snippet"] == text[: get_settings().snippet_chars]

    def test_small_note_reuse_not_duplicated(self, services) -> None:
        """Мелкая заметка (reuse: чанк-вектор == полный) входит в выдачу
        ОДИН раз — оба источника не дублируют её."""
        notes, searcher, _embedder = services
        text = "Единственная маленькая заметка про чанки"
        note_id = notes.save(text)["id"]
        results = searcher.search(text, top_k=5)["results"]
        assert [r["id"] for r in results].count(note_id) == 1
        assert results[0]["cosine"] == pytest.approx(1.0, abs=1e-4)
        assert results[0]["snippet"] == text[: get_settings().snippet_chars]

    def test_trash_note_chunks_filtered(self, services) -> None:
        """Soft delete прячет заметку и из чанк-поиска (вектора живы)."""
        notes, searcher, embedder = services
        note_id = notes.save(long_note_with_fact())["id"]
        _vectorize_pending_chunks(embedder)
        notes.delete(note_id)
        results = searcher.search("зарплатный реестр appsrv")["results"]
        assert [r["id"] for r in results] == []


# --- хелперы размера 8 (сценарные вектора, как в test_search_hybrid) --------


def _notes_with_chunk_vectors_dim8(monkeypatch: pytest.MonkeyPatch) -> tuple[int, int]:
    """Детерминированный векторный ландшафт: A — лучший чанк 0.6 по запросу E1,
    B — максимум 0.3 (ниже порога 0.35) — векторного источника не имеет."""
    monkeypatch.setenv("EMBEDDING_DIM", "8")
    monkeypatch.setenv("SCORE_THRESHOLD", "0.35")
    get_settings.cache_clear()
    settings = get_settings()

    e1 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    orthogonal = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # cosine(E1) = 0
    high = [0.6, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # 0.6 ≥ 0.35
    low = [0.3, 0.95, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # 0.3 < 0.35

    def add_note(text: str, chunk_rows: list[tuple[str, list[float]]]) -> int:
        with session(settings) as conn, transaction(conn):
            cursor = conn.execute(
                "INSERT INTO notes (text, vector_status) VALUES (?, 'ok')", (text,)
            )
            note_id = int(cursor.lastrowid or 0)
            chunk_ids = chunks.replace_note_chunks(
                conn, note_id, [(chunk_text, 5) for chunk_text, _v in chunk_rows]
            )
            for chunk_id, (_chunk_text, vector) in zip(
                chunk_ids, chunk_rows, strict=True
            ):
                chunks.upsert_vector(conn, chunk_id, vector)
        return note_id

    note_a = add_note(
        "Первая заметка: два чанка рукописных",
        [
            ("а-один: ортогональный чанк", orthogonal),
            ("а-два: лучший чанк с высокой близостью", high),
        ],
    )
    note_b = add_note(
        "Вторая заметка: слабые чанки рукописные",
        [
            ("б-один: слабый чанк", low),
            ("б-два: слабее чанк", orthogonal),
        ],
    )
    return note_a, note_b


class ScriptedEmbedder:
    """Сценарный эмбеддер: готовые вектора по порядку (запрос dim=8)."""

    def __init__(self, dim: int, queue: list[list[float]]) -> None:
        self.dim = dim
        self._queue = list(queue)

    def embed(self, _text: str) -> list[float]:
        if not self._queue:
            return [0.0] * self.dim
        return self._queue.pop(0) if len(self._queue) > 1 else self._queue[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]

    def close(self) -> None:
        return None