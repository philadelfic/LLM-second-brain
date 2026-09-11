"""Обвязка областей (субстрат 3.0.0): нормализация, сходство, гибридный поиск.

Один переиспользуемый помощник на три области (skills / terms / user):
каждая область описывается `AreaSpec` — своими таблицами, полями индексации
и проекцией выдачи; `AreaSearch` выполняет гибрид
vec0-KNN + FTS5(BM25) → RRF (тот же `settings.rrf_k`, примитив
`app.services.ranking.fuse_rrf`), фильтрует `deleted_at IS NULL` и
дедуплицирует id.

Деградация (NFR-3): отказ кодирования запроса → поиск FTS-only + `warning`
(прецедент `WARNING_FTS_ONLY` поиска заметок) — поиск не ломается от внешней
зависимости. Изоляция — архитектурный инвариант: SQL помощника адресует
только таблицы своей области, ни `notes*`, ни другие области.

`normalize_key` (ключ терминов и сравнение контекстов/фактов) и
`trigram_similarity` (близость контекстов и дедуп-подсказки без эмбеддинга)
— общие для сервисов областей (lsb-0008-01, lsb-0009-01).
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from typing import Any

from app.config import Settings
from app.services.embedding import Embedder, EmbeddingError, EmbeddingService
from app.services.ranking import fuse_rrf, match_expression
from app.services.search import (
    CANDIDATE_LIMIT,
    HINT_SHORT_QUERY,
    MAX_TOP_K,
    WARNING_FTS_ONLY,
)
from app.storage import area_vectors
from app.storage.db import (
    SKILLS_FTS_COLUMNS,
    SKILLS_FTS_TABLE,
    SKILLS_TABLE,
    TERMS_FTS_COLUMNS,
    TERMS_FTS_TABLE,
    TERMS_TABLE,
    USER_FACTS_FTS_COLUMNS,
    USER_FACTS_FTS_TABLE,
    USER_FACTS_TABLE,
    session,
)

# Пустой результат области — мягкий ответ с hint (архитектура субстрата §3.5:
# «не нашлось» — валидный исход, модель не лезет в область без причины).
HINT_NO_RESULTS = (
    "nothing found in this area; rephrase more broadly "
    "(searches by substrings from 3 characters)"
)

# Размер триграммы (символьная): как у FTS5 trigram и HashEmbedder тестов.
TRIGRAM_SIZE = 3

_WHITESPACE_RE = re.compile(r"\s+")


# --- нормализация и триграммное сходство ----------------------------------


def normalize_key(value: str) -> str:
    """Нормализовать строку ключа: lower → trim → пробелы схлопнуты → ё→е.

    Канон областей (lsb-0008 §3.2, lsb-0009 §3.4): одна механика на ключ
    термина (`term_norm` / `context_norm`) и на сравнение фактов. Регистр,
    хвостовые пробелы и разнобой внутренних пробелов не создают дублей,
    «ё»/«е» — одна буква.
    """
    lowered = value.lower().strip()
    collapsed = _WHITESPACE_RE.sub(" ", lowered)
    return collapsed.replace("ё", "е")


def _trigrams(text: str) -> set[str]:
    """Множество символьных триграмм текста (скользящее окно по 3)."""
    if len(text) < TRIGRAM_SIZE:
        return set()
    return {text[index : index + TRIGRAM_SIZE] for index in range(len(text) - 2)}


def trigram_similarity(left: str, right: str) -> float:
    """Триграммное сходство нормализованных строк — 0..1, симметрично.

    Коэффициент перекрытия (Szymkiewicz–Simpson): |A ∩ B| / min(|A|, |B|) по
    множествам триграмм нормализованных строк. Выбран осознанно: контексты
    терминов короткие («МГУ» vs «студенты МГУ» — близкие контексты, решение
    lsb-0008 §3.5), а Dice/Jaccard такие пары недооценивают (1/(1+N)).
    Границы: 1.0 — строки совпали или триграммы одной содержатся в другой;
    0.0 — нормализация пуста, триграмм нет или множества не пересекаются.
    Сходство считается вычислительно, без эмбеддинга (запись мгновенная).
    """
    normalized_left = normalize_key(left)
    normalized_right = normalize_key(right)
    if not normalized_left or not normalized_right:
        return 0.0
    if normalized_left == normalized_right:
        return 1.0
    left_trigrams = _trigrams(normalized_left)
    right_trigrams = _trigrams(normalized_right)
    if not left_trigrams or not right_trigrams:
        return 0.0
    return len(left_trigrams & right_trigrams) / min(
        len(left_trigrams), len(right_trigrams)
    )


# --- регистрация областей ---------------------------------------------------


@dataclass(frozen=True)
class AreaSpec:
    """Описание области для `AreaSearch`: свои таблицы, поля, проекция выдачи.

    `fts_columns` — колонки FTS5-индекса (для BM25); `select_columns` — поля
    проекции выдачи БЕЗ `id` (id добавляется всегда); `embed_fields` — поля,
    склеиваемые в текст векторизации записи (порядок = порядок склейки);
    `fts_weights` — веса колонок bm25 (пусто — все 1.0).
    """

    name: str
    table: str
    fts_table: str
    vec_table: str
    vec_id_column: str
    fts_columns: tuple[str, ...]
    select_columns: tuple[str, ...]
    embed_fields: tuple[str, ...]
    fts_weights: tuple[float, ...] = ()

    def embed_text(self, row: sqlite3.Row) -> str:
        """Текст векторизации записи: непустые поля, склеенные переводом строки.

        Форматы (архитектура субстрата §3.3): skills — `name + description`;
        terms — `term + context + definition`; user — `name + body`.
        """
        return "\n".join(
            str(row[column]) for column in self.embed_fields if row[column]
        )


SKILLS_AREA = AreaSpec(
    name="skills",
    table=SKILLS_TABLE,
    fts_table=SKILLS_FTS_TABLE,
    vec_table=area_vectors.SKILLS_VEC_TABLE,
    vec_id_column=area_vectors.SKILLS_VEC_ID_COLUMN,
    fts_columns=SKILLS_FTS_COLUMNS,
    select_columns=("name", "description"),
    embed_fields=("name", "description"),
)
TERMS_AREA = AreaSpec(
    name="terms",
    table=TERMS_TABLE,
    fts_table=TERMS_FTS_TABLE,
    vec_table=area_vectors.TERMS_VEC_TABLE,
    vec_id_column=area_vectors.TERMS_VEC_ID_COLUMN,
    fts_columns=TERMS_FTS_COLUMNS,
    select_columns=("term", "context", "definition"),
    embed_fields=("term", "context", "definition"),
)
USER_FACTS_AREA = AreaSpec(
    name="user",
    table=USER_FACTS_TABLE,
    fts_table=USER_FACTS_FTS_TABLE,
    vec_table=area_vectors.USER_FACTS_VEC_TABLE,
    vec_id_column=area_vectors.USER_FACTS_VEC_ID_COLUMN,
    fts_columns=USER_FACTS_FTS_COLUMNS,
    select_columns=("name", "body"),
    embed_fields=("name", "body"),
)

# Реестр областей (петля `areas` воркера и тесты изоляции ходят по нему).
ALL_AREAS: tuple[AreaSpec, ...] = (SKILLS_AREA, TERMS_AREA, USER_FACTS_AREA)


# --- гибридный поиск внутри области ----------------------------------------


class AreaSearchValidationError(ValueError):
    """Нарушение доменных ограничений запроса области (длина, top_k)."""


class AreaSearch:
    """Гибрид vec0-KNN + FTS5(BM25) → RRF внутри одной области; эмбеддер — DI.

    Область задаётся `AreaSpec`: помощник один на все три области, SQL каждой
    области адресует только её таблицы. Выдача — проекция области
    (`select_columns`) + `id` и `score` (RRF-слияние) — ровно те поля, что
    описаны контрактами фич. Пустой результат — мягкий ответ с hint; отказ
    эмбеддера — FTS-only + warning (WARNING_FTS_ONLY).
    """

    def __init__(
        self,
        settings: Settings,
        spec: AreaSpec,
        embedding: Embedder | None = None,
    ) -> None:
        self._settings = settings
        self._spec = spec
        # DI для тестов: детерминированный HashEmbedder вместо сети.
        self._embedding: Embedder = (
            embedding if embedding is not None else EmbeddingService(settings)
        )

    @property
    def spec(self) -> AreaSpec:
        """Описание области, под которое собран помощник (диагностика, тесты)."""
        return self._spec

    def search(self, query: str, top_k: int | None = None) -> dict[str, Any]:
        """Гибридный поиск внутри области (архитектура субстрата §3.5).

        Векторная сторона — KNN по vec0 области; FTS-сторона — BM25 по
        FTS5-индексу области; слияние — RRF (тот же `settings.rrf_k`),
        фильтр `deleted_at IS NULL` и дедупликация id. Отказ кодирования
        запроса не ломает поиск: FTS-only + `warning` (NFR-3).
        """
        query = self._validate_query(query)
        top_k = self._settings.default_top_k if top_k is None else top_k
        if not 1 <= top_k <= MAX_TOP_K:
            raise AreaSearchValidationError(
                f"top_k: expected 1..{MAX_TOP_K}, got {top_k}"
            )
        query_vector = self._query_vector(query)
        expression = match_expression(query)
        with session(self._settings) as conn:
            rankings: list[list[int]] = []
            if query_vector is not None:
                hits = area_vectors.knn(
                    conn,
                    self._spec.vec_table,
                    self._spec.vec_id_column,
                    query_vector,
                    CANDIDATE_LIMIT,
                )
                # Гейт порога косинуса (прецедент заметок, SCORE_THRESHOLD):
                # vec0-KNN ВСЕГДА отдаёт k ближайших — без порога «проба»
                # области («пусто = такой записи нет») стала бы недостижимой
                # при любой непустой таблице, и мягкий hint пустого поиска
                # (субстрат §3.5) не выдавался бы никогда. Порог — общая
                # калибровка кодировщика, не отдельная для областей.
                hits = [
                    (row_id, cosine)
                    for row_id, cosine in hits
                    if cosine >= self._settings.score_threshold
                ]
                rankings.append([row_id for row_id, _cosine in hits])
            if expression:
                rankings.append(
                    [
                        int(row["id"])
                        for row in self._fts_candidates(conn, expression)
                    ]
                )
            scores = fuse_rrf(rankings, self._settings.rrf_k)
            # Мягкое чтение: удалённые (trash) исчезают из выдачи, их id в
            # FTS/vec физически остаются (архитектура субстрата §3.2).
            rows = self._fetch_rows(conn, list(scores))
        results = self._merge(scores, rows)[:top_k]
        warning = None if query_vector is not None else WARNING_FTS_ONLY
        if not results:
            hint = (
                HINT_SHORT_QUERY
                if query_vector is None and expression is None
                else HINT_NO_RESULTS
            )
            return {"results": [], "warning": warning, "hint": hint}
        return {"results": results, "warning": warning}

    # --- источники кандидатов -----------------------------------------------

    def _fts_candidates(
        self, conn: sqlite3.Connection, expression: str
    ) -> list[sqlite3.Row]:
        """Топ-50 FTS5/BM25 по активным записям области; rank с 1 — для RRF."""
        spec = self._spec
        weights = spec.fts_weights or tuple(1.0 for _ in spec.fts_columns)
        weight_placeholders = ", ".join("?" * len(weights))
        return conn.execute(
            f"SELECT t.id, t.updated_at, "
            f"       bm25({spec.fts_table}, {weight_placeholders}) AS badness "
            f"FROM {spec.fts_table} JOIN {spec.table} t "
            f"ON t.id = {spec.fts_table}.rowid "
            f"WHERE {spec.fts_table} MATCH ? AND t.deleted_at IS NULL "
            "ORDER BY badness, t.updated_at DESC, t.id DESC LIMIT ?",
            (*weights, expression, CANDIDATE_LIMIT),
        ).fetchall()

    def _fetch_rows(
        self, conn: sqlite3.Connection, ids: list[int]
    ) -> dict[int, sqlite3.Row]:
        """Активные записи кандидатов с полями проекции (мягкое чтение)."""
        if not ids:
            return {}
        spec = self._spec
        placeholders = ", ".join("?" * len(ids))
        columns = ", ".join(("id", *spec.select_columns, "updated_at"))
        rows = conn.execute(
            f"SELECT {columns} FROM {spec.table} "
            f"WHERE deleted_at IS NULL AND id IN ({placeholders})",
            ids,
        ).fetchall()
        return {int(row["id"]): row for row in rows}

    # --- слияние и выдача ---------------------------------------------------

    def _merge(
        self,
        scores: dict[int, float],
        rows: dict[int, sqlite3.Row],
    ) -> list[dict[str, Any]]:
        """Сортировка: score DESC → updated_at DESC → id DESC; сборка проекции."""
        spec = self._spec
        candidates = [
            (scores[row_id], row)
            for row_id, row in rows.items()
            if row_id in scores
        ]
        candidates.sort(
            key=lambda pair: (pair[0], pair[1]["updated_at"], pair[1]["id"]),
            reverse=True,
        )
        return [
            {
                "id": int(row["id"]),
                **{column: row[column] for column in spec.select_columns},
                "score": score,
            }
            for score, row in candidates
        ]

    # --- внутреннее ---------------------------------------------------------

    def _query_vector(self, query: str) -> list[float] | None:
        """Кодирование запроса; отказ — деградация к FTS, не исключение."""
        try:
            return self._embedding.embed(query)
        except EmbeddingError:
            return None

    def _validate_query(self, query: str) -> str:
        """1..MAX_QUERY_CHARS — доменное правило запроса области."""
        if not 1 <= len(query) <= self._settings.max_query_chars:
            raise AreaSearchValidationError(
                f"query: length must be 1..{self._settings.max_query_chars} "
                f"characters, got {len(query)}"
            )
        return query
