"""SearchService — полнотекстовый поиск FTS5 (Фаза 2: FTS-only).

Гибрид RRF появится в Фазе 3 (ARCH §4.2): контракт выдачи уже по FR-1 —
`rrf_score` считается по одному источнику (1/(RRF_K + rank), rank с 1 —
эквивалент RRF одного источника), `cosine` = null. `warning` помечает
отсутствие семантики для моделей (канал §5.3); в Фазе 3 он станет
условным (только при отказе кодирования запроса).

Trigram-семантика (REQUIREMENTS §5.4): подстроки от 3 символов, русские
словоформы ловятся по общей подстроке. Запрос разбирается на слова:
каждое слово ищется как подстрока, слова соединяются AND. Слова короче
3 символов отбрасываются (trigram их не видит вообще). Каждое слово
«цитируется» — специальные последовательности FTS5 (AND/OR/*/( …) в
пользовательском запросе не превращаются в синтаксис, кавычки удваиваются.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from app.config import Settings
from app.services.emit import snippet, summary_of
from app.storage.db import session

# Верхняя граница контракта (REQUIREMENTS FR-1: top_k 1..20); env задаёт
# только умолчание — DEFAULT_TOP_K, поэтому MAX_TOP_K не настраивается.
MAX_TOP_K = 20

# ARCH §4.2: FTS-only — «поиск без семантики»; объяснение моделям (§5.3).
WARNING_FTS_ONLY = (
    "поиск только полнотекстовый (FTS5), без семантики: "
    "векторизация появится после Фазы 3"
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
    """FTS-поиск по активным заметкам (BM25); второй источник — Фаза 3."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self.degraded = True  # Фаза 2: семантики нет; Фаза 3 — при отказе Ollama

    def search(self, query: str, top_k: int | None = None) -> dict[str, Any]:
        """BM25 по trigram; выдача FR-1 без полного текста заметки."""
        query = self._validate_query(query)
        top_k = self._default_top_k() if top_k is None else top_k
        if not 1 <= top_k <= MAX_TOP_K:
            raise SearchValidationError(
                f"top_k: ожидается 1..{MAX_TOP_K}, получено {top_k}"
            )
        expression = self._match_expression(query)
        if expression is None:
            return {"results": [], "warning": WARNING_FTS_ONLY, "hint": HINT_SHORT_QUERY}
        with session(self._settings) as conn:
            rows = conn.execute(
                "SELECT n.id, n.text, n.summary, n.summary_status, n.author, "
                "       n.created_at, n.updated_at, "
                "       bm25(notes_fts) AS badness "
                "FROM notes_fts JOIN notes n ON n.id = notes_fts.rowid "
                "WHERE notes_fts MATCH ? AND n.deleted_at IS NULL "
                "ORDER BY badness, n.updated_at DESC, n.id DESC LIMIT ?",
                (expression, top_k),
            ).fetchall()
        results = [
            self._result(row, rank + 1)  # rank с 1 — для формулы RRF
            for rank, row in enumerate(rows)
        ]
        if not results:
            return {
                "results": [],
                "warning": self._warning(),
                "hint": HINT_NO_RESULTS,
            }
        return {"results": results, "warning": self._warning()}

    # --- внутреннее ---------------------------------------------------------

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

    def _warning(self) -> str | None:
        """warning только в деградации; в Фазе 2 это норма всех поисков."""
        return WARNING_FTS_ONLY if self.degraded else None

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

    def _result(self, row: sqlite3.Row, rank: int) -> dict[str, Any]:
        """Формат элемента FR-1: без текста заметки (memory_get адресно)."""
        return {
            "id": row["id"],
            "summary": summary_of(row, self._settings),
            "snippet": snippet(row["text"], self._settings),
            "summary_status": row["summary_status"],
            "rrf_score": 1 / (self._settings.rrf_k + rank),
            "cosine": None,  # векторов ещё нет — Фаза 3
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "author": row["author"],
        }