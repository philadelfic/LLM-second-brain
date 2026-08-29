"""SearchService — гибридный поиск (Фаза 3, ARCHITECTURE §4.2).

Два источника кандидатов, слияние Reciprocal Rank Fusion, единый top_k:
- вектор: vec0 топ-50 по косинусу **по полным текстам** (не summary) — ловит
  перефразы; отсечение: cosine ≥ SCORE_THRESHOLD (мусор не проходит в fusion);
- полнотекст: FTS5/BM25 топ-50 (trigram: русские словоформы, точные токены) —
  ловит идентификаторы/даты; порогу не подлежит (REQUIREMENTS FR-1).
- RRF устойчив к несопоставимым шкалам (косинус vs BM25): score(d) =
  Σ_sources 1/(RRF_K + rank_source), rank с 1.

Деградация (NFR-3): отказ кодирования запроса → поиск FTS-only + `warning`
(обучающий текст §5.3); НЕ жёсткая ошибка — поиск не ломается от внешней
зависимости. Кандидаты с векторным hit ниже порога вылетают до слияния —
их `cosine` в выдаче null (векторный hit валиден только ≥ порога).

Выдача FR-1: summary/snippet, без полного текста (memory_get адресно).
Ties (равный rrf_score) — по updated_at DESC, id DESC: свежее полезнее
(в духе FR-2), детерминированно без случайности.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from app.config import Settings
from app.services.embedding import Embedder, EmbeddingError, EmbeddingService
from app.services.emit import snippet, summary_of
from app.storage import vectors
from app.storage.db import session

# Верхняя граница контракта (REQUIREMENTS FR-1: top_k 1..20); env задаёт
# только умолчание — DEFAULT_TOP_K, поэтому MAX_TOP_K не настраивается.
MAX_TOP_K = 20

# Кандидатов из каждого источника до слияния (ARCH §4.2, REQUIREMENTS §5.4).
CANDIDATE_LIMIT = 50

# Отказ кодирования запроса: поиск деградирует к FTS-only + warning (§5.3).
WARNING_FTS_ONLY = (
    "поиск без семантики: кодирование запроса не удалось (векторизация "
    "недоступна), выдача только полнотекстовая — перефразы могут быть пропущены"
)

HINT_NO_RESULTS = (
    "по запросу ничего не найдено; переформулируй шире "
    "(ищется по подстрокам от 3 символов) или сделай обзор через memory_list"
)

HINT_SHORT_QUERY = (
    "каждое слово запроса короче 3 символов — trigram по ним не ищет; "
    "добавь осмысленные слова"
)


class SearchValidationError(ValueError):
    """Нарушение доменных ограничений запроса (длина, top_k)."""


class SearchService:
    """Гибрид vec0 + FTS5 → RRF; дефолт-эмбеддер — EmbeddingService."""

    def __init__(
        self, settings: Settings, embedding: Embedder | None = None
    ) -> None:
        self._settings = settings
        # DI для тестов: детерминированный HashEmbedder вместо сети.
        self._embedding: Embedder = embedding if embedding is not None else EmbeddingService(settings)

    def search(self, query: str, top_k: int | None = None) -> dict[str, Any]:
        """Гибридный поиск (ARCH §4.2); выдача FR-1 без полного текста."""
        query = self._validate_query(query)
        top_k = self._default_top_k() if top_k is None else top_k
        if not 1 <= top_k <= MAX_TOP_K:
            raise SearchValidationError(
                f"top_k: ожидается 1..{MAX_TOP_K}, получено {top_k}"
            )
        query_vector = self._query_vector(query)
        expression = self._match_expression(query)

        with session(self._settings) as conn:
            vector_hits = (
                self._vector_candidates(conn, query_vector)
                if query_vector is not None
                else []
            )
            fts_rows = self._fts_candidates(conn, expression) if expression else []

            # --- слияние RRF: score(d) = Σ 1/(RRF_K + rank) -----------------
            scores: dict[int, float] = {}
            cosine_by_id: dict[int, float] = {}
            for rank, (note_id, cosine) in enumerate(vector_hits, start=1):
                scores[note_id] = scores.get(note_id, 0.0) + 1.0 / (
                    self._settings.rrf_k + rank
                )
                cosine_by_id[note_id] = cosine
            for rank, row in enumerate(fts_rows, start=1):
                scores[row["id"]] = scores.get(row["id"], 0.0) + 1.0 / (
                    self._settings.rrf_k + rank
                )

            rows = self._fetch_rows(conn, list(scores))
        results = self._merge(scores, cosine_by_id, rows)[:top_k]
        warning = None if query_vector is not None else WARNING_FTS_ONLY
        if not results:
            hint = (
                HINT_SHORT_QUERY
                if query_vector is None and expression is None
                else HINT_NO_RESULTS
            )
            return {"results": [], "warning": warning, "hint": hint}
        return {"results": results, "warning": warning}

    # --- источники кандидатов ------------------------------------------------

    def _vector_candidates(
        self, conn: sqlite3.Connection, query_vector: list[float]
    ) -> list[tuple[int, float]]:
        """Топ-50 активных по косинусу + гейт порога (FR-1).

        KNN у vec0 идёт по всем векторам (trash вектора физически живы —
        ARCH §3.3), поэтому окно расширяется на число удалённых с векторами,
        а пост-фильтр оставляет только активные.
        """
        trash = conn.execute(
            "SELECT COUNT(*) FROM notes_vec WHERE note_id IN "
            "(SELECT id FROM notes WHERE deleted_at IS NOT NULL)"
        ).fetchone()[0]
        hits = vectors.knn(conn, query_vector, CANDIDATE_LIMIT + trash)
        if not hits:
            return []
        placeholders = ",".join("?" * len(hits))
        active = {
            row[0]
            for row in conn.execute(
                f"SELECT id FROM notes WHERE deleted_at IS NULL "
                f"AND id IN ({placeholders})",
                [note_id for note_id, _ in hits],
            )
        }
        threshold = self._settings.score_threshold
        return [
            (note_id, cosine)
            for note_id, cosine in hits[:CANDIDATE_LIMIT]
            if note_id in active and cosine >= threshold
        ]

    def _fts_candidates(
        self, conn: sqlite3.Connection, expression: str
    ) -> list[sqlite3.Row]:
        """Топ-50 FTS5/BM25 по активным заметкам; rank с 1 — для RRF."""
        return conn.execute(
            "SELECT n.id, n.text, n.summary, n.summary_status, n.author, "
            "       n.created_at, n.updated_at, "
            "       bm25(notes_fts) AS badness "
            "FROM notes_fts JOIN notes n ON n.id = notes_fts.rowid "
            "WHERE notes_fts MATCH ? AND n.deleted_at IS NULL "
            "ORDER BY badness, n.updated_at DESC, n.id DESC LIMIT ?",
            (expression, CANDIDATE_LIMIT),
        ).fetchall()

    # --- слияние и выдача ------------------------------------------------------

    def _fetch_rows(
        self, conn: sqlite3.Connection, ids: list[int]
    ) -> dict[int, sqlite3.Row]:
        """Метаданные кандидатов (мягкое чтение: удалённые исчезают)."""
        if not ids:
            return {}
        placeholders = ",".join("?" * len(ids))
        rows = conn.execute(
            f"SELECT id, text, summary, summary_status, author, "
            f"       created_at, updated_at "
            f"FROM notes WHERE deleted_at IS NULL AND id IN ({placeholders})",
            ids,
        ).fetchall()
        return {row["id"]: row for row in rows}

    def _merge(
        self,
        scores: dict[int, float],
        cosine_by_id: dict[int, float],
        rows: dict[int, sqlite3.Row],
    ) -> list[dict[str, Any]]:
        """Сортировка: rrf_score DESC → updated_at DESC → id DESC; сборка FR-1."""
        candidates = [
            (scores[note_id], row)
            for note_id, row in rows.items()
            if note_id in scores
        ]
        candidates.sort(
            key=lambda pair: (pair[0], pair[1]["updated_at"], pair[1]["id"]),
            reverse=True,
        )
        # Совместимость: rrf_score по итоговому слиянию; cosine — только у
        # кандидатов с валидным векторным hit (≥ SCORE_THRESHOLD).
        return [
            self._result(row, score, cosine_by_id.get(row["id"]))
            for score, row in candidates
        ]

    def _result(
        self, row: sqlite3.Row, rrf_score: float, cosine: float | None
    ) -> dict[str, Any]:
        """Формат элемента FR-1: без текста заметки (memory_get адресно)."""
        return {
            "id": row["id"],
            "summary": summary_of(row, self._settings),
            "snippet": snippet(row["text"], self._settings),
            "summary_status": row["summary_status"],
            "rrf_score": rrf_score,
            "cosine": cosine,
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "author": row["author"],
        }

    # --- внутреннее ---------------------------------------------------------

    def _query_vector(self, query: str) -> list[float] | None:
        """Кодирование запроса; отказ — деградация к FTS, не исключение."""
        try:
            return self._embedding.embed(query)
        except EmbeddingError:
            return None

    def _validate_query(self, query: str) -> str:
        """1..MAX_QUERY_CHARS — доменное правило FR-1 (бекстоп схемы)."""
        if not 1 <= len(query) <= self._settings.max_query_chars:
            raise SearchValidationError(
                f"query: длина должна быть 1..{self._settings.max_query_chars} "
                f"символов, получено {len(query)}"
            )
        return query

    def _default_top_k(self) -> int:
        return self._settings.default_top_k

    @staticmethod
    def _match_expression(query: str) -> str | None:
        """Слова ≥3 символов как цитированные подстроки через AND.

        None — нет ни одного слова, по которому trigram вообще может искать.
        """
        words = (word for word in query.split() if len(word) >= 3)
        unique = dict.fromkeys(words)
        if not unique:
            return None
        return " AND ".join(f'"{word.replace(chr(34), chr(34) * 2)}"' for word in unique)