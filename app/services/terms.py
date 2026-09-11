"""TermsService — сервисный слой области «terms» (lsb-0008, релиз 3.0.0).

Форма записи и лимиты — arch lsb-0008 §3.1; ключ и нормализация — §3.2;
данные — §3.3; поиск «все смыслы с контекстами» — §3.4; механика записи и
подсказок — §3.5; канонические тексты (hint'ы мягких отказов) — §3.7.

Ключевые свойства
-----------------
* **Ключ области** — `(term_norm, context_norm)` (§3.2): нормализация через
  `normalize_key` субстрата (lower → trim → пробелы схлопнуты → ё→е),
  хранятся и оригинальные формы. Уникальность — частичный UNIQUE по АКТИВНЫМ
  записям (`WHERE deleted_at IS NULL`): soft-deleted строку ключ не держит, и
  тот же ключ создаётся заново (§3.3).
* **Контекст обязателен ВСЕГДА** (§3.1) — пустой → отказ с дословным hint
  канона §3.7. Лимиты: `term` ≤ `term_max_chars` (100), `context` ≤
  `term_context_max_chars` (40), `definition` ≤ `term_definition_max_chars`
  (350); нарушение → `TermValidationError`, запись НЕ идёт.
* **Многозначность — норма** (§3.5): ключ совпал → UPDATE (определение и
  оригинальные формы); ключ не совпал и контекст слишком близок к уже
  использованному контексту ЭТОГО ЖЕ термина (триграммное сходство ≥
  `term_context_similarity` = 0.75) → мягкий отказ ДО записи (её нет) + hint с
  существующим контекстом и его id; иначе → INSERT нового смысла. Существующая
  запись не затирается: «МГУ» у студентов не перезаписывается «МГУ» в
  бухгалтерии.
* **Подсказки в успешном ответе** (FR-3.2): `senses` — смыслы этого термина
  `[{id, context}]` (только активные), `contexts` — глобальный список
  использованных контекстов области (топ `term_contexts_hint_limit` = 30 по
  частоте, без дублей по нормализации).
* **Эмбеддинг в момент записи НЕ вызывается** (FR-3.5, субстрат §3.3):
  близость контекста — триграммная (вычислительная); вектора записи догоняет
  петля `areas` воркера по `vector_status='pending'`, а после успешных
  записи/правки/удаления сервис её будит (`set_areas_notifier`).
* **Поиск двухшаговый** (§3.4): шаг 1 — точный нормализованный lookup по
  `term_norm` → ВСЕ смыслы термина с контекстами (`exact: true`); шаг 2 —
  точного нет → гибрид области (`AreaSearch` по `TERMS_AREA`, вектор =
  `term + context + definition`) → ближайшие смыслы `exact: false` + hint «это
  не точное совпадение»; шаг 3 — ничего → пусто + hint канона §3.7.
* **Удаление мягкое**: повторный/несуществующий id → мягкий ответ с hint
  (восстановление — оператором, прецедент заметок).
* **Листинга нет** (§3.1 требований): единственная дорога к определению —
  поиск; конвейер заметок сервис не затрагивает (изоляция: SQL адресует
  только таблицы области).

Контракты ответов (MCP-слой срезает служебные поля белым списком — lsb-0008-02,
REST-зеркало — lsb-0008-03):
- save (новый смысл) → {created: True, id, senses, contexts}
- save (ключ совпал) → {updated: True, id, senses, contexts}
- save (близкий контекст) → {created: False, hint, existing: {id, context}}
- search (точный)  → {senses: [{id, context, definition}], exact: True}
- search (близкие) → {senses: [{id, term, context, definition, score}],
                       exact: False, hint}
- search (пусто)   → {senses: [], exact: False, hint}
- update           → {id, updated: True} | {id, updated: False, hint};
                       занятый ключ → TermKeyConflictError (транспорт → 409)
- get              → {id, term, context, definition}; не найдена → {id, hint}
- delete           → {id, deleted: True} | {id, deleted: False, hint}
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from typing import Any

from app.config import Settings
from app.services.areas import (
    TERMS_AREA,
    AreaSearch,
    AreaSearchValidationError,
    normalize_key,
    trigram_similarity,
)
from app.services.embedding import Embedder, EmbeddingService
from app.services.search import MAX_TOP_K
from app.storage.db import session, transaction

# --- Hint'ы (канон §3.7: «дословные константы в коде»; тексты английские) ----

# Точного термина нет: ближайшие смыслы выдаются, но НЕ как точное совпадение
# (§3.4) — иначе модель молча подставит неверное определение.
HINT_NO_EXACT = (
    "no exact term — the senses above are the closest by meaning, not an "
    "exact match: check them against the context of the conversation"
)

# Термина нет вовсе (шаг 3 поиска, `get`, `delete`) — мягкий ответ без fail.
HINT_NOT_FOUND = (
    "term not found — there is no such term in this memory; save it via "
    "terms_save(term, context, definition) if it is worth keeping"
)

# Близкий контекст (мягкий отказ ДО записи, §3.5): шаблон подставляет данный
# контекст, существующий и его id — модель переиспользует формулировку.
HINT_CONTEXT_CLOSE = (
    "not saved: context '{given}' is too close to the existing one "
    "'{existing}' (id={id}) — reuse that context wording, or make the new "
    "context clearly different"
)

HINT_TERM_LIMIT = "not saved: term limit is 100 characters — shorten it"
HINT_CONTEXT_LIMIT = "not saved: context limit is 40 characters — shorten it"
HINT_DEFINITION_LIMIT = (
    "not saved: definition limit is 350 characters — shorten it"
)
HINT_CONTEXT_REQUIRED = (
    "not saved: context is required — specify the context in which the term "
    "is used (≤40 characters)"
)

# Hint, которого таблица канона §3.7 не задаёт: пустой `term` (канон описывает
# только лимиты, обязательность контекста и близкий контекст). Пустой термин
# ломает ключ области — отказ; стиль и язык — как у канонических.
HINT_TERM_REQUIRED = "not saved: term is required"

# Конфликт ключа при правке ПО ID (PUT /terms/{id}, субстрат §3.6: 409). Канон
# §3.7 задаёт тексты мягких отказов записи, но не конфликт ключа: транспорт
# обязан отличить его от валидации, поэтому текст собран по стилю канонических
# (английский, ведёт к существующей записи и к безопасному выходу).
HINT_KEY_CONFLICT = (
    "not saved: the key (term + context) is already used by another active "
    "record (id={id}: '{term}' / '{context}') — edit that record instead, or "
    "choose a different context or term"
)


class TermValidationError(ValueError):
    """Нарушение доменных ограничений термина (лимиты формы, запрос поиска).

    Текст исключения — hint для модели (транспорт MCP/REST отдаёт его клиенту
    как fail + hint), запись при этом НЕ идёт: прецедент `TitleValidationError`/
    `TITLE_HINT` и `SkillValidationError`/`UserFactValidationError`.
    """


class TermKeyConflictError(ValueError):
    """Новый ключ правки занят ДРУГОЙ активной записью области (субстрат
    §3.6: конфликт ключа → 409; отдельный тип, чтобы транспорт не путал его
    с нарушениями формы и мягкими отказами, которые отдаются как 422).

    Текст исключения — дословная hint-константа `HINT_KEY_CONFLICT` с id
    существующей записи: запись НЕ идёт, оператор правит её или меняет ключ.
    """


class TermsService:
    """CRUD области «terms»: ключ (term + context), смыслы, подсказки, поиск.

    DI: `settings` (лимиты формы, порог близости контекста, лимит списка
    контекстов), `embedding` (кодирование запроса гибридного поиска). Путь
    записи кодировщик НЕ зовёт (§3.5, субстрат §3.3): близость контекста
    триграммная, вектора догоняет петля `areas` воркера. Листинга у сервиса нет
    — публичные ручки только `save`/`update`/`search`/`get`/`delete`.
    """

    def __init__(
        self, settings: Settings, embedding: Embedder | None = None
    ) -> None:
        self._settings = settings
        # DI для тестов: HashEmbedder/фейк с журналом вызовов вместо сети.
        self._embedding: Embedder = (
            embedding if embedding is not None else EmbeddingService(settings)
        )
        # Гибридный поиск области (субстрат §3.5): vec0-KNN по `terms_vec`
        # (вектор = `term + context + definition`) + FTS5/BM25 по `terms_fts`
        # → RRF. SQL помощника адресует только таблицы области (изоляция).
        self._area_search = AreaSearch(settings, TERMS_AREA, self._embedding)
        # Сигнал воркеру (main.py): будить петлю areas сразу при появлении
        # pending-записи области, а не ждать выросший back-off.
        self._areas_notifier: Callable[[], None] | None = None

    def set_areas_notifier(self, notifier: Callable[[], None] | None) -> None:
        """Подключить сигнал пробуждения петли `areas` воркера (main.py).

        По образцу `SkillsService.set_areas_notifier`/`UserFactsService`:
        сервис собирается раньше воркера, нотификатор приходит после.
        """
        self._areas_notifier = notifier

    # --- запись и правка по ключу (arch §3.5) -------------------------------

    def save(self, term: str, context: str, definition: str) -> dict[str, Any]:
        """Сохранить термин с ключом (`term` + `context`) — три исхода §3.5.

        Порядок обязателен: форма проверяется ДО любых обращений к БД
        (нарушение лимита → `TermValidationError` с дословным hint §3.7).

        * ключ совпал (нормализация `term_norm` + `context_norm`) → UPDATE
          определения и оригинальных форм → `{updated: True, id, ...}`;
        * ключ не совпал, но контекст слишком близок (триграммное сходство
          нормализованных строк ≥ `term_context_similarity`) к контексту ЭТОГО
          ЖЕ термина → мягкий отказ, записи нет, hint указывает существующий
          контекст и его id;
        * иначе → INSERT нового смысла → `{created: True, id, ...}`.

        В обоих успешных исходах ответ несёт `senses` (смыслы этого термина
        после записи) и `contexts` (использованные контексты области). Эмбеддинг
        не вызывается: сходство контекста вычислительное (§3.5).
        """
        checked_term = self._checked_term(term)
        checked_context = self._checked_context(context)
        checked_definition = self._checked_definition(definition)
        term_norm = normalize_key(checked_term)
        context_norm = normalize_key(checked_context)
        with session(self._settings) as conn, transaction(conn):
            key_row = conn.execute(
                "SELECT id FROM terms WHERE term_norm = ? "
                "AND context_norm = ? AND deleted_at IS NULL",
                (term_norm, context_norm),
            ).fetchone()
            if key_row is not None:
                # Ключ совпал: это правка существующего смысла, не дубль
                # (оригинальные формы могут измениться, ключ — нет).
                term_id = int(key_row["id"])
                conn.execute(
                    "UPDATE terms SET term = ?, term_norm = ?, context = ?, "
                    "context_norm = ?, definition = ?, "
                    "vector_status = 'pending', "
                    "updated_at = strftime('%Y-%m-%dT%H:%M:%SZ','now') "
                    "WHERE id = ? AND deleted_at IS NULL",
                    (
                        checked_term,
                        term_norm,
                        checked_context,
                        context_norm,
                        checked_definition,
                        term_id,
                    ),
                )
                answer: dict[str, Any] = {"updated": True, "id": term_id}
            else:
                near = self._nearest_context(
                    conn, term_norm, checked_context
                )
                if (
                    near is not None
                    and near[2] >= self._settings.term_context_similarity
                ):
                    # Мягкий отказ ДО записи (решение О. 2026-09-11): близкая
                    # формулировка контекста не создаёт новый смысл — hint
                    # ведёт к существующему контексту и его id.
                    return {
                        "created": False,
                        "hint": HINT_CONTEXT_CLOSE.format(
                            given=checked_context,
                            existing=near[1],
                            id=near[0],
                        ),
                        "existing": {"id": near[0], "context": near[1]},
                    }
                cursor = conn.execute(
                    "INSERT INTO terms (term, term_norm, context, "
                    "context_norm, definition, vector_status) "
                    "VALUES (?, ?, ?, ?, ?, 'pending')",
                    (
                        checked_term,
                        term_norm,
                        checked_context,
                        context_norm,
                        checked_definition,
                    ),
                )
                answer = {"created": True, "id": int(cursor.lastrowid or 0)}
            # Справочная часть ответа (FR-3.2): смыслы термина после записи и
            # использованные контексты области — обе ветки успеха.
            answer["senses"] = self._senses(conn, term_norm)
            answer["contexts"] = self._contexts(conn)
        self._notify_areas_pending()
        return answer

    def update(
        self, id: int, term: str, context: str, definition: str
    ) -> dict[str, Any]:
        """Править запись ПО ID (PUT /terms/{id}; arch §3.6, субстрат §3.6).

        Отличается от `save` тем, что адресует строку по id, а не по ключу:
        цель — редкая чистка/правка терминологии оператором, включая переезд
        на другой ключ (`term`/`context`). Форма проверяется ДО БД (нарушение
        лимита или пустой контекст → `TermValidationError` → 422 с дословным
        hint §3.7); ключ пересчитывается (`term_norm`/`context_norm`), вектор
        уходит в `pending`, `updated_at` обновляется (догоняет петля `areas`).

        Исходы:

        * новый ключ занят ДРУГОЙ активной записью → `TermKeyConflictError`
          (`HINT_KEY_CONFLICT`, транспорт → 409): ключ области уникален, и
          переезд не должен затирать чужой смысл;
        * тот же ключ (сама запись) — не конфликт: правка идёт штатно;
        * записи нет/она soft-deleted → мягкий ответ `{id, updated: False,
          hint}` канона §3.7 (транспорт → 404).

        Подсказка «близкий контекст» здесь НЕ применяется: это защита записи
        НОВОГО смысла (§3.5), а правка адресует существующую строку осознанно.
        """
        checked_term = self._checked_term(term)
        checked_context = self._checked_context(context)
        checked_definition = self._checked_definition(definition)
        term_norm = normalize_key(checked_term)
        context_norm = normalize_key(checked_context)
        with session(self._settings) as conn, transaction(conn):
            row = conn.execute(
                "SELECT id FROM terms WHERE id = ? AND deleted_at IS NULL",
                (id,),
            ).fetchone()
            if row is None:
                return {"id": id, "updated": False, "hint": HINT_NOT_FOUND}
            occupied = conn.execute(
                "SELECT id, term, context FROM terms WHERE term_norm = ? "
                "AND context_norm = ? AND deleted_at IS NULL AND id != ?",
                (term_norm, context_norm, id),
            ).fetchone()
            if occupied is not None:
                raise TermKeyConflictError(
                    HINT_KEY_CONFLICT.format(
                        id=int(occupied["id"]),
                        term=str(occupied["term"]),
                        context=str(occupied["context"]),
                    )
                )
            conn.execute(
                "UPDATE terms SET term = ?, term_norm = ?, context = ?, "
                "context_norm = ?, definition = ?, "
                "vector_status = 'pending', "
                "updated_at = strftime('%Y-%m-%dT%H:%M:%SZ','now') "
                "WHERE id = ? AND deleted_at IS NULL",
                (
                    checked_term,
                    term_norm,
                    checked_context,
                    context_norm,
                    checked_definition,
                    id,
                ),
            )
        self._notify_areas_pending()
        return {"id": id, "updated": True}

    # --- чтение (arch §3.4, §3.5) -------------------------------------------

    def search(self, query: str, top_k: int | None = None) -> dict[str, Any]:
        """Найти термин: точный нормализованный lookup → гибрид области → пусто.

        Шаг 1: точное совпадение `term_norm` (нормализация снимает регистр, «ё»,
        лишние пробелы) → ВСЕ активные смыслы термина `[{id, context,
        definition}]`, `exact: true`; один смысл не выбирается за модель (§3.4).
        Шаг 2: точного нет → гибрид `TERMS_AREA` (vec0-KNN по `term + context +
        definition` + FTS5/BM25 → RRF) → ближайшие смыслы `[{id, term, context,
        definition, score}]`, `exact: false` + дословный hint «не точное
        совпадение». Шаг 3: ничего → `{senses: [], exact: false, hint}`.

        `top_k` — границы как у поиска заметок (`1..MAX_TOP_K`), нарушение —
        `TermValidationError` (мягкий отказ транспорта). Отказ эмбеддера поиск не
        ломает: FTS-only + `warning` (NFR-3). Заметки, навыки и факты
        пользователя в выдачу не попадают: SQL помощника адресует только
        таблицы области.
        """
        if top_k is not None and not 1 <= top_k <= MAX_TOP_K:
            raise TermValidationError(
                f"top_k: expected 1..{MAX_TOP_K}, got {top_k}"
            )
        query_norm = normalize_key(query)
        if query_norm:
            with session(self._settings) as conn:
                exact_rows = conn.execute(
                    "SELECT id, context, definition FROM terms "
                    "WHERE term_norm = ? AND deleted_at IS NULL ORDER BY id",
                    (query_norm,),
                ).fetchall()
            if exact_rows:
                return {
                    "senses": [
                        {
                            "id": int(row["id"]),
                            "context": row["context"],
                            "definition": row["definition"],
                        }
                        for row in exact_rows
                    ],
                    "exact": True,
                }
        try:
            found = self._area_search.search(query, top_k)
        except AreaSearchValidationError as exc:  # запрос вне домена области
            raise TermValidationError(str(exc)) from exc
        answer: dict[str, Any] = {
            "senses": [
                {
                    "id": int(item["id"]),
                    "term": item["term"],
                    "context": item["context"],
                    "definition": item["definition"],
                    "score": item["score"],
                }
                for item in found["results"]
            ],
            "exact": False,
            "warning": found.get("warning"),
        }
        # Hint канона §3.7: точного нет — ближайшие; ничего нет — термина нет.
        answer["hint"] = HINT_NO_EXACT if answer["senses"] else HINT_NOT_FOUND
        return answer

    def get(self, id: int) -> dict[str, Any]:
        """Прочитать запись по id: `{id, term, context, definition}`; нет — hint."""
        with session(self._settings) as conn:
            row = conn.execute(
                "SELECT id, term, context, definition FROM terms "
                "WHERE id = ? AND deleted_at IS NULL",
                (id,),
            ).fetchone()
        if row is None:
            return {"id": id, "hint": HINT_NOT_FOUND}
        return {
            "id": int(row["id"]),
            "term": row["term"],
            "context": row["context"],
            "definition": row["definition"],
        }

    # --- удаление (soft delete, arch §3.3) ----------------------------------

    def delete(self, id: int) -> dict[str, Any]:
        """Мягко удалить запись: `deleted_at` = now, строка/индексы живы.

        Идемпотентно по смыслу: повторный/несуществующий id → мягкий ответ
        `deleted: False` + hint канона §3.7 (восстановление — оператором).
        Удалённый ключ освобождается: частичный UNIQUE — только по активным
        записям, тот же ключ можно создать заново.
        """
        with session(self._settings) as conn, transaction(conn):
            cursor = conn.execute(
                "UPDATE terms SET deleted_at = "
                "strftime('%Y-%m-%dT%H:%M:%SZ','now') "
                "WHERE id = ? AND deleted_at IS NULL",
                (id,),
            )
            deleted = bool(cursor.rowcount)
        if not deleted:
            return {"id": id, "deleted": False, "hint": HINT_NOT_FOUND}
        # Сигнал петле areas един по всем успешным изменениям области: у
        # удаления pending не появляется, но сигнал безвреден и держит одну
        # точку пробуждения воркера (постановка lsb-0008-01).
        self._notify_areas_pending()
        return {"id": id, "deleted": True}

    # --- внутреннее: подсказки и близость контекста -------------------------

    def _senses(
        self, conn: sqlite3.Connection, term_norm: str
    ) -> list[dict[str, Any]]:
        """Активные смыслы этого термина — `[{id, context}]` по возрастанию id."""
        rows = conn.execute(
            "SELECT id, context FROM terms WHERE term_norm = ? "
            "AND deleted_at IS NULL ORDER BY id",
            (term_norm,),
        ).fetchall()
        return [
            {"id": int(row["id"]), "context": row["context"]} for row in rows
        ]

    def _contexts(self, conn: sqlite3.Connection) -> list[str]:
        """Использованные контексты области: топ по частоте, без дублей.

        Частота — число активных записей области с этим `context_norm` (GROUP BY
        `context_norm`), лимит — `term_contexts_hint_limit` (30). Дублей по
        нормализации нет. Какая оригинальная форма возвращается при совпадении
        нормализованных, решено так: форма САМОЙ ПОЗДНЕЙ записи этого контекста
        (MAX(id)) — «актуальная формулировка»; при равной частоте такие контексты
        идут первыми (порядок: частота DESC, `last_id` DESC). Выбор
        детерминирован и не зависит от порядка строк в таблице.
        """
        rows = conn.execute(
            "SELECT t.context AS context FROM ("
            "  SELECT context_norm, MAX(id) AS last_id, COUNT(*) AS uses "
            "  FROM terms WHERE deleted_at IS NULL GROUP BY context_norm"
            ") c JOIN terms t ON t.id = c.last_id "
            "ORDER BY c.uses DESC, c.last_id DESC LIMIT ?",
            (self._settings.term_contexts_hint_limit,),
        ).fetchall()
        return [str(row["context"]) for row in rows]

    def _nearest_context(
        self, conn: sqlite3.Connection, term_norm: str, context: str
    ) -> tuple[int, str, float] | None:
        """Ближайший использованный контекст этого термина — `(id, форма, сходство)`.

        Сходство — триграммное (`trigram_similarity` субстрата, коэффициент
        перекрытия по нормализованным строкам); эмбеддинг не вызывается (§3.5).
        При равенстве сходства побеждает меньший id — результат детерминирован.
        Нет активных смыслов термина → `None` (запись идёт без подсказки).
        """
        rows = conn.execute(
            "SELECT id, context FROM terms WHERE term_norm = ? "
            "AND deleted_at IS NULL ORDER BY id",
            (term_norm,),
        ).fetchall()
        nearest: tuple[int, str, float] | None = None
        for row in rows:
            similarity = trigram_similarity(context, str(row["context"]))
            if nearest is None or similarity > nearest[2]:
                nearest = (int(row["id"]), str(row["context"]), similarity)
        return nearest

    # --- внутреннее: валидация формы ---------------------------------------

    def _checked_term(self, term: str | None) -> str:
        """Обязательный `term`: непустой и ≤ `term_max_chars` символов.

        Пустой термин ломает ключ области → отказ с hint (канон §3.7 задаёт
        только лимит); превышение лимита → дословный hint канона §3.7.
        Возврат — исходная форма без краевых пробелов (оригинальные формы
        хранятся рядом с нормализованными, §3.2).
        """
        normalized = term.strip() if isinstance(term, str) else ""
        if not normalized:
            raise TermValidationError(HINT_TERM_REQUIRED)
        if len(normalized) > self._settings.term_max_chars:
            raise TermValidationError(HINT_TERM_LIMIT)
        return normalized

    def _checked_context(self, context: str | None) -> str:
        """Обязательный `context`: непустой и ≤ `term_context_max_chars`.

        Контекст обязателен ВСЕГДА, даже при единственном смысле (§3.1): пустой
        → отказ с дословным hint канона §3.7; превышение лимита — тоже.
        """
        normalized = context.strip() if isinstance(context, str) else ""
        if not normalized:
            raise TermValidationError(HINT_CONTEXT_REQUIRED)
        if len(normalized) > self._settings.term_context_max_chars:
            raise TermValidationError(HINT_CONTEXT_LIMIT)
        return normalized

    def _checked_definition(self, definition: str | None) -> str:
        """`definition` ≤ `term_definition_max_chars` символов.

        Канон §3.1 помечает обязательным только контекст (пустое определение —
        не нарушение лимита), поэтому проверяется лишь верхняя граница:
        превышение → дословный hint канона §3.7.
        """
        normalized = definition.strip() if isinstance(definition, str) else ""
        if len(normalized) > self._settings.term_definition_max_chars:
            raise TermValidationError(HINT_DEFINITION_LIMIT)
        return normalized

    # --- внутреннее: сигнал -------------------------------------------------

    def _notify_areas_pending(self) -> None:
        """Сигнал воркеру: область изменилась (будить петлю `areas` сразу)."""
        if self._areas_notifier is not None:
            self._areas_notifier()
