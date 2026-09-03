"""SearchService — гибридный поиск (Фаза 3, ARCHITECTURE §4.2; Фаза 7 — вектора
строятся по чанкам).

Два источника кандидатов, слияние Reciprocal Rank Fusion, единый top_k:
- вектор (Фаза 7): KNN по чанкам (топ-50) → агрегация до заметок: лучший чанк
  заметки задаёт близость (cosine = лучшего чанка, сравним с SCORE_THRESHOLD);
  заметки без ГОТОВЫХ чанков (унаследованные до Фазы 7 и pending у воркера) —
  fallback на вектор полного текста (notes_vec) как в Фазе 3. Полный текст и
  summary на качество поиска не влияли раньше — не влияют и сейчас: векторам
  всё равно, а summary только в выдаче;
- полнотекст: FTS5/BM25 топ-50 (trigram: русские словоформы, точные токены) —
  ловит идентификаторы/даты; порогу не подлежит (REQUIREMENTS FR-1). Слова
  запроса — OR подстрок (BUG-001: AND терял заметку без одного из слов).
- RRF устойчив к несопоставимым шкалам (косинус vs BM25): score(d) =
  Σ_sources 1/(RRF_K + rank_source), rank с 1. Векторная сторона после
  агрегации — один список, ранжирование как в Фазе 3.

Деградация (NFR-3): отказ кодирования запроса → поиск FTS-only + `warning`
(обучающий текст §5.3); НЕ жёсткая ошибка — поиск не ломается от внешней
зависимости. Кандидаты с лучшим векторным hit ниже порога вылетают до слияния.

Выдача FR-1: summary/snippet, без полного текста (memory_get адресно).
Snippet — из ЛУЧШЕГО чанка (первые SNIPPET_CHARS символов чанка): модель видит
релевантный фрагмент длинной заметки, а не её начало; у fallback-кандидатов —
из полного текста, как в Фазе 3. Ties (равный rrf_score) — по updated_at DESC,
id DESC: свежее полезнее (в духе FR-2), детерминированно без случайности.
Выдача — полный контракт сервис-слоя (REST отдаёт как есть);
MCP-слой срезает служебные поля — см. Фаза 9.
"""

from __future__ import annotations

import re
import sqlite3
from typing import Any

from app.config import Settings
from app.services.embedding import Embedder, EmbeddingError, EmbeddingService
from app.services.emit import snippet, summary_of
from app.services.namespaces import NamespaceError, NamespaceService
from app.storage import chunks, vectors
from app.storage.db import session

# Верхняя граница контракта (REQUIREMENTS FR-1: top_k 1..20); env задаёт
# только умолчание — DEFAULT_TOP_K, поэтому MAX_TOP_K не настраивается.
MAX_TOP_K = 20

# Кандидатов из каждого источника до слияния (ARCH §4.2, REQUIREMENTS §5.4).
CANDIDATE_LIMIT = 50

# Разбивка составных токенов запроса (BUG-001): «open-webui» → «open» +
# «webui» — FTS ловит тексты, где написание отличается («Open WebUI»).
_TOKEN_SPLIT_RE = re.compile(r"[^0-9A-Za-zА-Яа-яЁё]+")

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
    """Гибрид vec0+chunk-vec0 (агрегация) + FTS5 → RRF; эмбеддер — DI."""

    def __init__(
        self, settings: Settings, embedding: Embedder | None = None
    ) -> None:
        self._settings = settings
        # DI для тестов: детерминированный HashEmbedder вместо сети.
        self._embedding: Embedder = embedding if embedding is not None else EmbeddingService(settings)
        # Фаза 10: реестр неймспейсов (валидация узла, поддеревья).
        self._namespaces = NamespaceService(settings)

    def search(
        self,
        query: str,
        top_k: int | None = None,
        namespace: str | None = None,
        namespace_exact: bool = False,
    ) -> dict[str, Any]:
        """Гибридный поиск (ARCH §4.2 + Фаза 7); выдача FR-1 без полного текста.

        Фаза 10: `namespace` — узел иерархии; фильтр по его поддереву
        (узел + зарегистрированные листья; `namespace_exact` — только сам
        узел). None — глобальный поиск (RRF как раньше). Незарегистрированный
        узел — NamespaceError (транспорт Шага 3 обернёт в fail + hint).
        Каждый результат несёт свой `namespace` — модель видит, где нашлось
        (слой ориентирования № 3).
        """
        query = self._validate_query(query)
        top_k = self._default_top_k() if top_k is None else top_k
        if not 1 <= top_k <= MAX_TOP_K:
            raise SearchValidationError(
                f"top_k: ожидается 1..{MAX_TOP_K}, получено {top_k}"
            )
        query_vector = self._query_vector(query)
        expression = self._match_expression(query)
        ns_nodes = self._namespaces.filter_nodes(namespace, namespace_exact)

        with session(self._settings) as conn:
            vector_hits = (
                self._vector_candidates(conn, query_vector, ns_nodes)
                if query_vector is not None
                else []
            )
            fts_rows = (
                self._fts_candidates(conn, expression, ns_nodes) if expression else []
            )

            # --- слияние RRF: score(d) = Σ 1/(RRF_K + rank) -----------------
            # Векторный источник — уже агрегированный список заметок
            # (лучший чанк задал cosine и snippet), Фаза 3 — полный вектор.
            scores: dict[int, float] = {}
            cosine_by_id: dict[int, float] = {}
            snippet_source: dict[int, str | None] = {}
            for rank, (note_id, cosine, chunk_text) in enumerate(vector_hits, start=1):
                scores[note_id] = scores.get(note_id, 0.0) + 1.0 / (
                    self._settings.rrf_k + rank
                )
                cosine_by_id[note_id] = cosine
                snippet_source[note_id] = chunk_text
            for rank, row in enumerate(fts_rows, start=1):
                scores[row["id"]] = scores.get(row["id"], 0.0) + 1.0 / (
                    self._settings.rrf_k + rank
                )

            rows = self._fetch_rows(conn, list(scores))
        results = self._merge(scores, cosine_by_id, snippet_source, rows)[:top_k]
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

    def _namespace_filter(
        self, namespace: str | None, namespace_exact: bool
    ) -> list[str] | None:
        """Узлы-партиции для фильтра — общая логика в реестре (filter_nodes)."""
        return self._namespaces.filter_nodes(namespace, namespace_exact)

    def _vector_candidates(
        self,
        conn: sqlite3.Connection,
        query_vector: list[float],
        ns_nodes: list[str] | None,
    ) -> list[tuple[int, float, str | None]]:
        """Векторные кандидаты: чанки → агрегация до заметок + fallback.

        Каждая заметка входит в источник ОДИН раз: близость задаёт её лучший
        чанк; заметки без готовых чанков (нет строк notes_chunks вообще или
        векторов у чанков) получают cosine полного вектора (notes_vec).
        Фаза 10: поиск в партициях узлов поддерева (ns_nodes) — или глобально.
        Возврат уже отсортирован по cosine (для ранга в RRF) и отфильтрован
        по активности и порогу.
        """
        by_chunk_source = self._chunk_candidates(conn, query_vector, ns_nodes)
        # Fallback Фазы 7: заметки, у которых нет готовых чанк-векторов
        # (legacy/пending), ищутся по полному тексту, как в Фазе 3.
        for note_id, cosine in self._full_text_candidates(conn, query_vector, ns_nodes):
            if note_id not in by_chunk_source:
                by_chunk_source[note_id] = (cosine, None)
        ranked = sorted(
            by_chunk_source.items(), key=lambda pair: pair[1][0], reverse=True
        )
        return [
            (note_id, cosine, chunk_text)
            for note_id, (cosine, chunk_text) in ranked
        ]

    def _chunk_candidates(
        self,
        conn: sqlite3.Connection,
        query_vector: list[float],
        ns_nodes: list[str] | None,
    ) -> dict[int, tuple[float, str]]:
        """KNN по notes_chunks_vec (топ-50) → одна запись на заметку.

        Первый чанк заметки в отсортированной по близости выдаче KNN — её
        лучший чанк: он задаёт cosine и текст для snippet. Порог сравнивается
        с cosine лучшего чанка (brief §6). Чанки trash-заметок отбрасываются
        пост-фильтром (vec0 не знает о notes). Фаза 10: KNN сканирует партиции
        узлов поддерева (ns_nodes) — или глобально; trash-окно считается
        в тех же партициях.
        """
        if ns_nodes:
            ns_ph = ",".join("?" * len(ns_nodes))
            ns_clause = f" AND n.namespace IN ({ns_ph})"
            ns_params: list[object] = list(ns_nodes)
        else:
            ns_clause = ""
            ns_params = []
        trash = conn.execute(
            "SELECT COUNT(*) FROM notes_chunks_vec WHERE chunk_id IN "
            "(SELECT c.id FROM notes_chunks c JOIN notes n ON n.id = c.note_id "
            f" WHERE n.deleted_at IS NOT NULL{ns_clause})",
            ns_params,
        ).fetchone()[0]
        hits = chunks.knn(conn, query_vector, CANDIDATE_LIMIT + trash, ns_filter=ns_nodes)
        hits = hits[:CANDIDATE_LIMIT]
        if not hits:
            return {}
        placeholders = ",".join("?" * len(hits))
        # один заход: note_id и text чанка + активность заметки (в узлах)
        chunk_meta = {
            row["id"]: (row["note_id"], row["text"])
            for row in conn.execute(
                "SELECT c.id, c.note_id, c.text FROM notes_chunks c "
                "JOIN notes n ON n.id = c.note_id "
                f"WHERE n.deleted_at IS NULL AND c.id IN ({placeholders})"
                f"{ns_clause}",
                [chunk_id for chunk_id, _ in hits] + ns_params,
            )
        }
        by_note: dict[int, tuple[float, str]] = {}
        threshold = self._settings.score_threshold
        for chunk_id, cosine_value in hits:  # порядок KNN: лучший чанк — первым
            meta = chunk_meta.get(chunk_id)
            if meta is None:
                continue  # чанк trash-заметки
            note_id, text = meta
            if note_id in by_note:
                continue  # лучший чанк заметки уже учтён
            if cosine_value < threshold:
                continue  # лучший чанк ниже порога — остальные тем более
            by_note[note_id] = (cosine_value, text)
        return by_note

    def _full_text_candidates(
        self,
        conn: sqlite3.Connection,
        query_vector: list[float],
        ns_nodes: list[str] | None,
    ) -> list[tuple[int, float]]:
        """Топ-50 активных по косинусу полного текста + гейт порога (FR-1).

        Окно KNN расширяется на число trash-векторов, пост-фильтр оставляет
        только активные. Заметки, уже попавшие в чанк-агрегацию, здесь
        отфильтруются наверху (они не fallback). Фаза 10: KNN — в партициях
        узлов поддерева, активные — только из тех же узлов.
        """
        if ns_nodes:
            ns_ph = ",".join("?" * len(ns_nodes))
            ns_clause = f" AND namespace IN ({ns_ph})"
            ns_params: list[object] = list(ns_nodes)
        else:
            ns_clause = ""
            ns_params = []
        trash = conn.execute(
            "SELECT COUNT(*) FROM notes_vec WHERE note_id IN "
            "(SELECT id FROM notes WHERE deleted_at IS NOT NULL"
            f"{ns_clause})",
            ns_params,
        ).fetchone()[0]
        hits = vectors.knn(conn, query_vector, CANDIDATE_LIMIT + trash, ns_filter=ns_nodes)
        if not hits:
            return []
        placeholders = ",".join("?" * len(hits))
        active = {
            row[0]
            for row in conn.execute(
                f"SELECT id FROM notes WHERE deleted_at IS NULL "
                f"{ns_clause} AND id IN ({placeholders})",
                ns_params + [note_id for note_id, _ in hits],
            )
        }
        threshold = self._settings.score_threshold
        return [
            (note_id, cosine)
            for note_id, cosine in hits[:CANDIDATE_LIMIT]
            if note_id in active and cosine >= threshold
        ]

    def _fts_candidates(
        self,
        conn: sqlite3.Connection,
        expression: str,
        ns_nodes: list[str] | None,
    ) -> list[sqlite3.Row]:
        """Топ-50 FTS5/BM25 по активным заметкам в узлах поддерева; rank с 1 —
        для RRF. Фаза 10: фильтр namespace — JOIN'ом на notes (FTS-индекс
        не пересоздаётся, бриф Шаг 1)."""
        if ns_nodes:
            ns_ph = ",".join("?" * len(ns_nodes))
            ns_clause = f" AND n.namespace IN ({ns_ph})"
            params: list[object] = [expression, *ns_nodes]
        else:
            ns_clause = ""
            params = [expression]
        return conn.execute(
            "SELECT n.id, n.text, n.summary, n.summary_status, n.author, "
            "       n.namespace, n.created_at, n.updated_at, "
            "       bm25(notes_fts) AS badness "
            "FROM notes_fts JOIN notes n ON n.id = notes_fts.rowid "
            f"WHERE notes_fts MATCH ?{ns_clause} AND n.deleted_at IS NULL "
            "ORDER BY badness, n.updated_at DESC, n.id DESC LIMIT ?",
            (*params, CANDIDATE_LIMIT),
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
            f"SELECT id, namespace, text, summary, summary_status, author, "
            f"       created_at, updated_at "
            f"FROM notes WHERE deleted_at IS NULL AND id IN ({placeholders})",
            ids,
        ).fetchall()
        return {row["id"]: row for row in rows}

    def _merge(
        self,
        scores: dict[int, float],
        cosine_by_id: dict[int, float],
        snippet_source: dict[int, str | None],
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
        # кандидатов с валидным векторным hit (≥ SCORE_THRESHOLD); snippet —
        # из лучшего чанка, если он есть, иначе из полного текста (fallback).
        return [
            self._result(
                row,
                score,
                cosine_by_id.get(row["id"]),
                snippet_source.get(row["id"]),
            )
            for score, row in candidates
        ]

    def _result(
        self,
        row: sqlite3.Row,
        rrf_score: float,
        cosine: float | None,
        snippet_source: str | None,
    ) -> dict[str, Any]:
        """Формат элемента FR-1: без текста заметки (memory_get адресно)."""
        return {
            "id": row["id"],
            "summary": summary_of(row, self._settings),
            "snippet": snippet(snippet_source or row["text"], self._settings),
            "summary_status": row["summary_status"],
            "rrf_score": rrf_score,
            "cosine": cosine,
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "author": row["author"],
            "namespace": row["namespace"],
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
        """Слова ≥3 символов (плюс ≥3-символьные части составных токенов)
        как цитированные подстроки через OR.

        OR, а не AND (BUG-001): AND выкидывал заметку целиком, если хотя бы
        одно слово запроса не встречалось в её тексте («openwebui chat_id»
        не находило заметку про chat_id без «openwebui»). При OR BM25
        ранжирует по числу/редкости совпавших слов — заметка со всеми
        словами выше; шум отсекается RRF-слиянием и top_k.
        None — нет ни одного слова, по которому trigram вообще может искать.
        """
        tokens: list[str] = []
        for word in query.split():
            if len(word) >= 3:
                tokens.append(word)
            for part in _TOKEN_SPLIT_RE.split(word):
                if len(part) >= 3 and part != word:
                    tokens.append(part)
        unique = dict.fromkeys(tokens)
        if not unique:
            return None
        return " OR ".join(
            f'"{token.replace(chr(34), chr(34) * 2)}"' for token in unique
        )