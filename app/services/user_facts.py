"""UserFactsService — сервисный слой области «user» (lsb-0009, релиз 3.0.0).

Форма факта и лимиты — arch lsb-0009 §3.1; данные — §3.2; ручки и контракты —
§3.3; дедуп «есть похожее» без эмбеддинга в момент записи — §3.4; канонические
тексты (постоянный hint атомарности, hint'ы мягких отказов) — §3.7.

Ключевые свойства
-----------------
* **Атомарность** — один факт = одна запись: форма `name` (≤5 слов, контракт
  `title` заметок) + `body` (≤1200 символов); нарушение →
  `UserFactValidationError` с дословным hint канона §3.7 (факт НЕ сохраняется).
* **Дедуп-подсказка ДО записи** (§3.4): кандидаты — FTS5 `trigram` по
  `name + body` (top-3, только активные), сходство — триграммное
  (`trigram_similarity` субстрата) по нормализованному тексту. Сильное
  совпадение (≥ `USER_SIMILAR_STRONG`) → мягкий отказ + `similar` (записи нет);
  средняя зона (≥ `USER_SIMILAR_WEAK`) → запись идёт + справочный список
  `related`; ниже порогов — запись без подсказки. Эмбеддинг в момент записи НЕ
  вызывается (FR-3.5): кандидаты и сходство вычислительны.
* **Постоянный hint атомарности** (FR-7.2) — в КАЖДОМ успешном ответе `save`.
* **Сентинелы «не передано» = оставить** (`_UNSET_NAME`/`_UNSET_BODY`,
  JSON-серизуемые строки — прецедент lsb-0004/`notes.py`); `null` в обязательном
  поле → мягкий отказ (обязательные поля не сбрасываются).
* **Правка инвалидирует вектор** (`vector_status='pending'`, `updated_at`
  обновляется) — вектора догоняет петля `areas` воркера (субстрат §3.3);
  после успешных записи/правки/удаления сервис её будит
  (`set_areas_notifier`).
* **Удаление мягкое** (`deleted_at`): исчезает из всех выдач области;
  повторный/несуществующий id → мягкий ответ с hint (восстановление —
  оператором, прецедент заметок).
* **Поиск — проба области** (§3.3, субстрат §3.5): гибрид `AreaSearch` по
  `USER_FACTS_AREA` (`user_facts_vec` KNN, вектор = `name + body` +
  `user_facts_fts` BM25 → RRF); выдача без тел — `excerpt` ≤
  `USER_SEARCH_EXCERPT_CHARS` (полное тело — `get`); пусто → мягкий ответ с
  hint канона §3.7. SQL помощника адресует только таблицы области (изоляция).

Контракты ответов (полные; MCP-слой срезает служебные поля белым списком —
lsb-0009-02, REST-зеркало — lsb-0009-03):
- save (запись)    → {id, stored: True, hint: <постоянный hint атомарности>}
                     (+ related/hint средней зоны)
- save (сильное совпадение) → {stored: False, hint, similar: {id, name}}
- update           → {id, changed: True} | {id, changed: False, hint}
- get              → {id, name, body}; не найден → {id, hint}
- search           → {results: [{id, name, excerpt}], warning?};
                     пусто → {results: [], warning?, hint}
- delete           → {id, deleted: True} | {id, deleted: False, hint}
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from typing import Any

from app.config import Settings
from app.services.areas import (
    USER_FACTS_AREA,
    AreaSearch,
    AreaSearchValidationError,
    trigram_similarity,
)
from app.services.embedding import Embedder, EmbeddingService
from app.services.ranking import match_expression
from app.services.search import MAX_TOP_K
from app.storage.db import session, transaction

# Кандидаты дедупа — top-3 активных записей области (arch §3.4): проверяем
# только ближайшие по FTS, полного скана реестра нет.
DEDUP_CANDIDATES = 3

# --- Hint'ы (канон §3.7: «дословные константы в коде»; тексты английские) ----
# Постоянный hint успешного save (FR-7.2): правило атомарности закрепляется в
# момент действия — несколько фактов = несколько отдельных вызовов.
HINT_ATOMIC = "one fact = one record — several facts mean several separate calls"

# Дедуп, сильное совпадение (FR-7.3): запись НЕ идёт, hint ведёт к правке
# существующего факта; шаблон подставляет id и name ближайшей записи.
HINT_SIMILAR_FACT = (
    "similar fact already exists: {id} — {name}; the same fact? update it via "
    "user_update(id={id}); a new fact — save it as a separate record"
)

# Дедуп, средняя зона: запись идёт, справочно перечисляются похожие факты.
HINT_RELATED_FACTS = (
    "possibly related facts: {related}; if this is the same fact, refine it via "
    "user_update instead of creating a duplicate"
)

# Мягкий отказ при `body` > 1200 (FR-7.4): «похоже на несколько фактов».
HINT_BODY_LIMIT = (
    "not saved: looks like several facts in one record — split them and save "
    "one fact per call (body limit is 1200 characters)"
)

# Лимит `name` (канон §3.7): контракт названия заметки (≤5 слов).
HINT_NAME_LIMIT = "not saved: name must be ≤5 words (like a note title)"

# Не найден (`get`/`update`/`delete`) — мягкий ответ с путём к поиску области.
HINT_NOT_FOUND = (
    "fact not found (possibly deleted); search the user area via user_search"
)

# Пустой поиск области: «факта нет» — валидный исход (проба), модель не лезет
# в область без причины.
HINT_SEARCH_EMPTY = (
    "nothing found in the user area — no fact matching this request is stored"
)

# Hint'ы, которых таблица канона §3.7 не задаёт (канон описывает только лимиты
# и мягкие отказы выше): пустое обязательное поле формы и `null` в обязательном
# поле правки. Стиль и язык — как у канонических.
HINT_NAME_REQUIRED = "not saved: name is required"
HINT_BODY_REQUIRED = "not saved: body is required"
HINT_REQUIRED_UNSET = (
    "not updated: {field} is mandatory — omitting the argument keeps the "
    "current value, null cannot reset it"
)

# Сентинелы «не передано» (прецедент lsb-0004): JSON-серизуемые строки —
# дефолты параметров MCP-инструмента правки, pydantic не ругается на
# несеризуемый дефолт (NFR-4). Сравнение в сервисе — по идентичности (`is`),
# поэтому коллизия с реальным содержимым факта невозможна даже при совпадении
# строки (как `_UNSET_SUMMARY` заметок).
_UNSET_NAME: str = "__LSB_UNSET_USER_NAME__"
_UNSET_BODY: str = "__LSB_UNSET_USER_BODY__"


class UserFactValidationError(ValueError):
    """Нарушение доменных ограничений факта (лимиты формы, запрос поиска).

    Текст исключения — hint для модели (транспорт MCP/REST отдаёт его клиенту
    как fail + hint), факт при этом НЕ сохраняется: прецедент
    `TitleValidationError`/`TITLE_HINT` и `SkillValidationError`.
    """


def _fact_text(name: str, body: str) -> str:
    """Текст факта для сходства и векторизации — `name + body`.

    Тот же формат, что у вектора области (`AreaSpec.embed_text`, субстрат §3.3:
    user — `name + body`, непустые поля через перевод строки).
    """
    return "\n".join(part for part in (name, body) if part)


def _related_text(related: list[dict[str, Any]]) -> str:
    """Справочный список похожих фактов для hint'а средней зоны — `id — name`."""
    return "; ".join(f"{item['id']} — {item['name']}" for item in related)


class UserFactsService:
    """CRUD области «user»: атомарные факты, дедуп-подсказка, изолированный поиск.

    DI: `settings` (лимиты формы, пороги сходства, обрезка выдачи), `embedding`
    (кодирование запроса гибридного поиска). Путь записи кодировщик НЕ зовёт
    (FR-3.5, субстрат §3.3): вектора записи догоняет петля `areas` воркера по
    `vector_status='pending'`. Сид skill-создателя (lsb-0007-04) живёт в таблице
    `skills` и к этой области отношения не имеет — сервис его не читает и не
    меняет.
    """

    def __init__(
        self, settings: Settings, embedding: Embedder | None = None
    ) -> None:
        self._settings = settings
        # DI для тестов: HashEmbedder/фейк с журналом вызовов вместо сети.
        # Запись факта кодировщик не зовёт вовсе (дедуп — FTS + триграммы);
        # кодировщик нужен только гибридному поиску области.
        self._embedding: Embedder = (
            embedding if embedding is not None else EmbeddingService(settings)
        )
        # Гибридный поиск области (субстрат §3.5): vec0-KNN по `user_facts_vec`
        # (вектор = `name + body`) + FTS5/BM25 по `user_facts_fts` → RRF.
        # SQL помощника адресует только таблицы области (изоляция).
        self._area_search = AreaSearch(settings, USER_FACTS_AREA, self._embedding)
        # Сигнал воркеру (main.py): будить петлю areas сразу при появлении
        # pending-записи области, а не ждать выросший back-off.
        self._areas_notifier: Callable[[], None] | None = None

    def set_areas_notifier(self, notifier: Callable[[], None] | None) -> None:
        """Подключить сигнал пробуждения петли `areas` воркера (main.py).

        По образцу `SkillsService.set_areas_notifier`: сервис собирается раньше
        воркера, нотификатор приходит после его создания.
        """
        self._areas_notifier = notifier

    # --- запись: создание (arch §3.4) ---------------------------------------

    def save(self, *, name: str, body: str) -> dict[str, Any]:
        """Сохранить ОДИН атомарный факт (валидация → дедуп → запись).

        Порядок обязателен: форма проверяется ДО дедупа (нарушение лимита →
        `UserFactValidationError` с дословным hint §3.7), дедуп-скан — ДО
        записи и без эмбеддинга (§3.4). Сильное совпадение → мягкий отказ
        `{stored: False, hint, similar}` без записи; средняя зона → запись +
        `related` + справочный hint; ниже порогов → запись. В любом успешном
        ответе — постоянный hint атомарности (FR-7.2): несколько фактов =
        несколько отдельных вызовов.
        """
        checked_name = self._checked_name(name)
        checked_body = self._checked_body(body)
        with session(self._settings) as conn:
            candidates = self._dedup_candidates(conn, checked_name, checked_body)
        if candidates:
            nearest, similarity = candidates[0]
            if similarity >= self._settings.user_similar_strong:
                # Мягкий отказ (решение О. 2026-09-11): запись НЕ идёт, hint
                # ведёт к правке существующего факта либо к отдельной записи.
                return {
                    "stored": False,
                    "hint": HINT_SIMILAR_FACT.format(
                        id=nearest["id"], name=nearest["name"]
                    ),
                    "similar": nearest,
                }
        related = [
            item
            for item, similarity in candidates
            if similarity >= self._settings.user_similar_weak
        ]
        fact_id = self._create(checked_name, checked_body)
        answer: dict[str, Any] = {
            "id": fact_id,
            "stored": True,
            "hint": HINT_ATOMIC,
        }
        if related:
            # Средняя зона: запись состоялась, похожие перечислены справочно.
            # Оба канонических текста остаются в ответе дословно: постоянный
            # hint атомарности — всегда, hint средней зоны — только здесь.
            answer["related"] = related
            answer["hint"] = (
                f"{HINT_ATOMIC}; "
                + HINT_RELATED_FACTS.format(related=_related_text(related))
            )
        self._notify_areas_pending()
        return answer

    def _create(self, name: str, body: str) -> int:
        """INSERT факта с `vector_status='pending'` (вектора — петля `areas`)."""
        with session(self._settings) as conn, transaction(conn):
            cursor = conn.execute(
                "INSERT INTO user_facts (name, body, vector_status) "
                "VALUES (?, ?, 'pending')",
                (name, body),
            )
            return int(cursor.lastrowid or 0)

    # --- правка (arch §3.3: «не передано» = оставить) -----------------------

    def update(
        self,
        id: int,
        *,
        name: str | None = _UNSET_NAME,
        body: str | None = _UNSET_BODY,
    ) -> dict[str, Any]:
        """Отредактировать факт по id: `name` и/или `body`.

        «Не передано» (сентинел, дефолт параметра) = оставить текущее значение;
        `null` в обязательном поле — мягкий отказ (обязательные поля не
        сбрасываются, hint со семантикой «не передано = оставить»). Переданное
        значение валидируется как при создании. Правка возвращает вектор записи
        в `pending` и обновляет `updated_at` (вектора догоняет петля `areas`);
        не найден/удалён → мягкий ответ с hint канона §3.7. Оба параметра не
        переданы — правки нет (ни `updated_at`, ни вектор не трогаем).
        """
        name_passed = name is not _UNSET_NAME
        body_passed = body is not _UNSET_BODY
        if name_passed and name is None:
            raise UserFactValidationError(
                HINT_REQUIRED_UNSET.format(field="name")
            )
        if body_passed and body is None:
            raise UserFactValidationError(
                HINT_REQUIRED_UNSET.format(field="body")
            )
        checked_name = self._checked_name(name) if name_passed else None
        checked_body = self._checked_body(body) if body_passed else None
        if checked_name is None and checked_body is None:
            # Оба параметра «не передано»: менять нечего (прецедент lsb-0004 —
            # «не передано» не сбрасывает и не переписывает запись).
            return {"id": id, "changed": False}
        with session(self._settings) as conn, transaction(conn):
            row = conn.execute(
                "SELECT id, name, body FROM user_facts "
                "WHERE id = ? AND deleted_at IS NULL",
                (id,),
            ).fetchone()
            if row is None:
                return {"id": id, "changed": False, "hint": HINT_NOT_FOUND}
            conn.execute(
                "UPDATE user_facts SET name = ?, body = ?, "
                "vector_status = 'pending', "
                "updated_at = strftime('%Y-%m-%dT%H:%M:%SZ','now') "
                "WHERE id = ? AND deleted_at IS NULL",
                (
                    checked_name if checked_name is not None else row["name"],
                    checked_body if checked_body is not None else row["body"],
                    id,
                ),
            )
        self._notify_areas_pending()
        return {"id": id, "changed": True}

    # --- чтение (arch §3.3) -------------------------------------------------

    def get(self, id: int) -> dict[str, Any]:
        """Прочитать факт: `{id, name, body}`; не найден → мягкий ответ с hint.

        Полное тело отдаёт только эта ручка (поиск — обрезанный `excerpt`).
        """
        with session(self._settings) as conn:
            row = conn.execute(
                "SELECT id, name, body FROM user_facts "
                "WHERE id = ? AND deleted_at IS NULL",
                (id,),
            ).fetchone()
        if row is None:
            return {"id": id, "hint": HINT_NOT_FOUND}
        return {"id": int(row["id"]), "name": row["name"], "body": row["body"]}

    def search(self, query: str, top_k: int | None = None) -> dict[str, Any]:
        """Гибридный поиск по области — «проба» (arch §3.3, субстрат §3.5).

        Сборка `AreaSearch` под область user: vec0-KNN по `user_facts_vec`
        (вектор = `name + body`) + FTS5/BM25 по `user_facts_fts` (`name`/`body`)
        → RRF, фильтр `deleted_at IS NULL`. Выдача без тел: `{results: [{id,
        name, excerpt}], warning?}`, `excerpt` обрезан до
        `user_search_excerpt_chars` (полное тело — `get`). Пусто — мягкий ответ
        с дословным hint канона §3.7 (факта нет — валидный исход). Отказ
        эмбеддера поиск не ломает: FTS-only + `warning` (NFR-3). Изоляция: SQL
        помощника адресует только таблицы области — заметки, навыки и термины
        не читаются. `top_k` — границы как у поиска заметок (`1..MAX_TOP_K`),
        нарушение — `UserFactValidationError` (мягкий отказ транспорта).
        """
        if top_k is not None and not 1 <= top_k <= MAX_TOP_K:
            raise UserFactValidationError(
                f"top_k: expected 1..{MAX_TOP_K}, got {top_k}"
            )
        try:
            found = self._area_search.search(query, top_k)
        except AreaSearchValidationError as exc:  # запрос вне домена области
            raise UserFactValidationError(str(exc)) from exc
        answer: dict[str, Any] = {
            "results": [
                {
                    "id": int(item["id"]),
                    "name": item["name"],
                    "excerpt": self._excerpt(item["body"]),
                }
                for item in found["results"]
            ],
            "warning": found.get("warning"),
        }
        if not answer["results"]:
            answer["hint"] = HINT_SEARCH_EMPTY
        return answer

    # --- удаление (soft delete, arch §3.4) ---------------------------------

    def delete(self, id: int) -> dict[str, Any]:
        """Мягкое удаление факта: `deleted_at` = now, строка/индексы живы.

        Идемпотентно по смыслу: повторный/несуществующий id → мягкий ответ
        `deleted: False` + hint канона §3.7 (восстановление — оператором,
        прецедент заметок). Удалённый факт исчезает из всех выдач области.
        """
        with session(self._settings) as conn, transaction(conn):
            cursor = conn.execute(
                "UPDATE user_facts SET deleted_at = "
                "strftime('%Y-%m-%dT%H:%M:%SZ','now') "
                "WHERE id = ? AND deleted_at IS NULL",
                (id,),
            )
            deleted = bool(cursor.rowcount)
        if not deleted:
            return {"id": id, "deleted": False, "hint": HINT_NOT_FOUND}
        # Сигнал петле areas един по всем успешным изменениям области
        # (постановка lsb-0009-01): у удаления pending не появляется, но
        # сигнал безвреден и держит одну точку пробуждения воркера.
        self._notify_areas_pending()
        return {"id": id, "deleted": True}

    # --- внутреннее: дедуп --------------------------------------------------

    def _dedup_candidates(
        self, conn: sqlite3.Connection, name: str, body: str
    ) -> list[tuple[dict[str, Any], float]]:
        """Кандидаты дедупа с триграммным сходством, по убыванию близости.

        Кандидаты — FTS5 `trigram` по `name + body` (top-3 активных записей
        ИЗОЛИРОВАННОЙ области, arch §3.4): SQL адресует только `user_facts`/
        `user_facts_fts`. Сходство считается вычислительно по нормализованному
        тексту (`trigram_similarity` субстрата: lower → trim → пробелы → ё→е) —
        эмбеддинг в момент записи не вызывается (FR-3.5). Удалённые записи
        кандидатами не бывают. Нет слов ≥3 символов — триграммный индекс искать
        не может: кандидатов нет (запись идёт без подсказки).
        """
        expression = match_expression(f"{name} {body}")
        if not expression:
            return []
        weights = ", ".join("1.0" for _ in USER_FACTS_AREA.fts_columns)
        rows = conn.execute(
            f"SELECT t.id, t.name, t.body "
            f"FROM {USER_FACTS_AREA.fts_table} JOIN {USER_FACTS_AREA.table} t "
            f"ON t.id = {USER_FACTS_AREA.fts_table}.rowid "
            f"WHERE {USER_FACTS_AREA.fts_table} MATCH ? AND t.deleted_at IS NULL "
            f"ORDER BY bm25({USER_FACTS_AREA.fts_table}, {weights}), "
            "t.updated_at DESC, t.id DESC LIMIT ?",
            (expression, DEDUP_CANDIDATES),
        ).fetchall()
        target = _fact_text(name, body)
        scored = [
            (
                {"id": int(row["id"]), "name": row["name"]},
                trigram_similarity(target, _fact_text(row["name"], row["body"])),
            )
            for row in rows
        ]
        scored.sort(key=lambda pair: (pair[1], pair[0]["id"]), reverse=True)
        return scored

    # --- внутреннее: валидация формы ---------------------------------------

    def _checked_name(self, name: str | None) -> str:
        """Обязательное `name`: непустое и ≤ `user_name_max_words` слов.

        Механизм тот же, что у `is_valid_title` заметок (контракт `title`):
        слова = `len(name.split())`; лимит — из настроек (канон —
        `TITLE_MAX_WORDS=5`). Нарушение → `UserFactValidationError` с дословным
        hint канона §3.7; возврат — нормализованное `name` без краевых пробелов.
        """
        normalized = name.strip() if isinstance(name, str) else ""
        if not normalized:
            raise UserFactValidationError(HINT_NAME_REQUIRED)
        if len(normalized.split()) > self._settings.user_name_max_words:
            raise UserFactValidationError(HINT_NAME_LIMIT)
        return normalized

    def _checked_body(self, body: str | None) -> str:
        """Обязательное `body`: непустое и ≤ `user_body_max_chars` символов.

        Превышение лимита — дословный hint канона §3.7 («похоже на несколько
        фактов в одной записи — разбей»): запись НЕ сохраняется.
        """
        normalized = body.strip() if isinstance(body, str) else ""
        if not normalized:
            raise UserFactValidationError(HINT_BODY_REQUIRED)
        if len(normalized) > self._settings.user_body_max_chars:
            raise UserFactValidationError(HINT_BODY_LIMIT)
        return normalized

    # --- внутреннее: выдача и сигнал ----------------------------------------

    def _excerpt(self, body: str) -> str:
        """Обрезка тела для выдачи поиска (`excerpt` ≤ лимита настроек).

        Полное тело поиск не отдаёт никогда (arch §3.3): детали — через `get`.
        """
        limit = self._settings.user_search_excerpt_chars
        return body if len(body) <= limit else body[:limit]

    def _notify_areas_pending(self) -> None:
        """Сигнал воркеру: область изменилась (будить петлю `areas` сразу)."""
        if self._areas_notifier is not None:
            self._areas_notifier()
