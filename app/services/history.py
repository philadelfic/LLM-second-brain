"""Слой истории версий (№41, релиз 3.2.0): общая таблица `record_versions`.

Актуальная запись живёт в прежних таблицах и читается как раньше; рядом
хранится история: `version=1` — baseline (содержимое целиком), дальше —
цепочка дельт (FR-1.1/FR-2.1/FR-2.2). Лист над хранилищем: доменные сервисы
не импортирует, `Settings` берёт из `app.config`, соединение — у вызывающего.

Ключевые решения (ARCH substrate-versioning §5, §6.1, §13):
- Формат дельты — пофайловый JSON-объект «поле → новое значение»; длинное
  текстовое поле (≥ `HISTORY_DELTA_INLINE_MAX`) кодируется построчной заменой
  (`{"__lines__": [[i, j, [строки]], …]}`) — история растёт по числу правок,
  а не по объёму записи (§5.2, §13.1).
- Запись — ТОЛЬКО на открытом соединении вызывающего, внутри его
  `transaction()`: нет состояния «правка прошла, истории нет» (FR-2.4, §13.5).
  Модуль на пути записи не открывает ни соединений, ни транзакций.
- Источник и сессия версии (`source`/`session`) приходят из contextvar
  (`bind_origin`/`current_origin`): сигнатуры сервисов не меняются (§13.4).
- История никогда не переписывается: откат — новая версия (§13.6); политика
  хранения (`HISTORY_MAX_VERSIONS`/`HISTORY_RETENTION_DAYS`) вычищает
  старейшие версии и ПЕРЕБАЗИРУЕТ самую раннюю сохраняемую в полный snapshot,
  номера не переиспользуются (§13.8).
"""

from __future__ import annotations

import difflib
import json
import sqlite3
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from app.config import Settings, get_settings
from app.services.listing import page_fields
from app.storage.db import session

# --- сущности и версионируемая проекция (§5.3) -----------------------------

ENTITY_NOTE = "note"
ENTITY_SKILL = "skill"
ENTITY_SKILL_TEMPLATE = "skill_template"
ENTITY_TERM = "term"
ENTITY_USER_FACT = "user_fact"

# Реестр проекции §5.3: какие поля сущности создают версию. Служебные
# (`vector_status`, `summary_status`, `links_at`, `node_order_at`,
# `classified_at`, `hint_path`, `confidence`) в проекцию не входят и версий
# не создают (FR-4.2). Производная (авто) `summary` заметки тоже вне проекции:
# в версию попадает только явно переданная выжимка (§13.3).
CONTENT_FIELDS: dict[str, tuple[str, ...]] = {
    ENTITY_NOTE: ("text", "title", "namespace", "expires_at"),
    ENTITY_SKILL: ("name", "description", "example", "steps", "text", "extra"),
    ENTITY_SKILL_TEMPLATE: ("instruction_template",),
    ENTITY_TERM: ("term", "context", "definition"),
    ENTITY_USER_FACT: ("name", "body"),
}


def note_projection(
    row: sqlite3.Row | Mapping[str, Any], *, explicit_summary: bool = False
) -> dict[str, Any]:
    """Проекция заметки по §5.3 — одна точка правды для save/update/merge.

    Контент: `text`/`title`/`namespace`/`expires_at`. `summary` входит только
    при ЯВНОЙ передаче (`explicit_summary=True`) — авто-перегенерация воркером
    версий не создаёт (§13.3). Служебные поля (`vector_status`,
    `summary_status`, `links_at`, `node_order_at`, `classified_at`) в проекцию
    не входят и версий не создают (FR-4.2).
    """
    fields: dict[str, Any] = {
        "text": row["text"],
        "title": row["title"],
        "namespace": row["namespace"],
        "expires_at": row["expires_at"],
    }
    if explicit_summary:
        fields["summary"] = row["summary"]
    return fields


# Мягкий ответ чтения, когда версии/записи нет: транспорт превращает его в
# 404 (никакого исключения — «версии нет» валидный исход).
HINT_VERSION_NOT_FOUND = (
    "version not found for this record — history_list shows the available versions"
)

# Источник версии, если запись вызвана вне `bind_origin` (прямой вызов слоя
# из тестов/утилит): настоящий источник неизвестен.
DEFAULT_SOURCE = "unknown"


class HistoryValidationError(ValueError):
    """Ошибка слоя истории: неизвестная сущность или версия вне диапазона."""


def _check_entity(entity: str) -> None:
    """Сущность обязана быть в реестре проекции (§5.3)."""
    if entity not in CONTENT_FIELDS:
        raise HistoryValidationError(
            "unknown entity: " + entity + " (known: " + ", ".join(CONTENT_FIELDS) + ")"
        )


# --- кодирование и применение дельты (§5.2) --------------------------------

def encode_delta(
    previous: Mapping[str, Any], current: Mapping[str, Any], *, inline_max: int
) -> tuple[dict[str, Any], list[str]]:
    """Дельта «поле → представление нового значения» + список изменившихся полей.

    Поле кодируется построчно, если и прежнее, и новое значения — строки и
    новое длиннее/равно `inline_max`; иначе — значением целиком (короткие поля,
    объекты, `null` при снятии). Представление выбирается по каждому полю
    в каждой версии (§5.2). Порядок полей в результате стабилен: сначала поля
    прежнего значения, затем новые (появление поля — в конец).
    """
    delta: dict[str, Any] = {}
    changed: list[str] = []
    for field in _ordered_fields(previous, current):
        old = previous.get(field)
        new = current.get(field)
        if field in previous and field in current and old == new:
            continue
        changed.append(field)
        delta[field] = _encode_value(old, new, inline_max=inline_max)
    return delta, changed


def _ordered_fields(
    previous: Mapping[str, Any], current: Mapping[str, Any]
) -> list[str]:
    """Поля обоих значений без повторов: порядок прежнего, затем новые."""
    fields = list(previous)
    fields.extend(field for field in current if field not in previous)
    return fields


def _encode_value(old: Any, new: Any, *, inline_max: int) -> Any:
    """Представление значения поля: построчная замена или значение целиком."""
    if isinstance(old, str) and isinstance(new, str) and len(new) >= inline_max:
        return _encode_lines(old, new)
    return new


def _encode_lines(previous: str, current: str) -> dict[str, list[list[Any]]]:
    """Построчная замена: только не-`equal` опкоды SequenceMatcher (§5.2).

    `[i, j, [строки]]` — в прежнем значении позиции `[i, j)` заменяются на
    перечисленные строки (insert — `i == j`, delete — пустой список).
    """
    previous_lines = previous.split("\n")
    current_lines = current.split("\n")
    matcher = difflib.SequenceMatcher(None, previous_lines, current_lines, autojunk=False)
    opcodes: list[list[Any]] = []
    for tag, start, stop, start_new, stop_new in matcher.get_opcodes():
        if tag == "equal":
            continue
        opcodes.append([start, stop, current_lines[start_new:stop_new]])
    return {"__lines__": opcodes}


def apply_delta(fields: dict[str, Any], delta: Mapping[str, Any]) -> dict[str, Any]:
    """Применить дельту к словарю полей (тот же объект; §5.2).

    Один алгоритм на все представления: `{"__lines__": …}` — построчная замена
    в текущем значении поля (не строки → пустая строка), любое другое значение
    (в том числе `null`) — присваивание целиком.
    """
    for field, value in delta.items():
        if isinstance(value, dict) and "__lines__" in value:
            lines = (fields.get(field) or "").split("\n")
            for start, stop, replacement in value["__lines__"]:
                lines[start:stop] = replacement
            fields[field] = "\n".join(lines)
        else:
            fields[field] = value
    return fields


# --- происхождение версии (contextvar, §13.4) ------------------------------

# Источник и сессия вызова живут в contextvar: сигнатуры сервисов не меняются,
# `asyncio.to_thread` контекст пробрасывает; jobs и миграция передают источник
# явно (без bind_origin).
_ORIGIN: ContextVar[tuple[str, str | None]] = ContextVar(
    "history_origin", default=(DEFAULT_SOURCE, None)
)
_ROLLBACK_TO: ContextVar[int | None] = ContextVar("history_rollback_to", default=None)


@contextmanager
def bind_origin(
    source: str, session: str | None = None, *, rollback_to: int | None = None
) -> Iterator[None]:
    """Обвязать участок работы источником версий: `source` + `session`.

    `rollback_to` отмечает участок как откат: версии получают
    `event='rollback'` и `restore_from` (FR-5.4). Вложенный `bind_origin`
    перекрывает внешний, выход возвращает прежнее значение (contextvar).
    """
    origin_token = _ORIGIN.set((source, session))
    rollback_token = _ROLLBACK_TO.set(rollback_to)
    try:
        yield
    finally:
        _ORIGIN.reset(origin_token)
        _ROLLBACK_TO.reset(rollback_token)


def current_origin() -> tuple[str, str | None]:
    """Текущий источник версии (`source`, `session`) участка работы."""
    return _ORIGIN.get()


def current_rollback_to() -> int | None:
    """Номер версии-цели, если участок работы — откат (иначе `None`)."""
    return _ROLLBACK_TO.get()


# --- запись: только на соединении вызывающего (§13.5) ----------------------

def _dump(value: Any) -> str:
    """JSON-строка записи (кириллица — как есть: читаемость для оператора)."""
    return json.dumps(value, ensure_ascii=False)


def _origin(source: str | None, session_id: str | None) -> tuple[str, str | None]:
    """Источник/сессия версии: явные аргументы сильнее текущего contextvar."""
    current_source, current_session = current_origin()
    resolved_source = current_source if source is None else source
    resolved_session = current_session if session_id is None else session_id
    return resolved_source, resolved_session


def _head(conn: sqlite3.Connection, entity: str, entity_id: int) -> int:
    """Головная версия записи (`0` — истории ещё нет)."""
    row = conn.execute(
        "SELECT MAX(version) FROM record_versions WHERE entity = ? AND entity_id = ?",
        (entity, entity_id),
    ).fetchone()
    return 0 if row is None or row[0] is None else int(row[0])


def record_initial(
    conn: sqlite3.Connection,
    entity: str,
    entity_id: int,
    fields: Mapping[str, Any],
    *,
    source: str | None = None,
    session: str | None = None,
) -> int:
    """Первая версия записи: `version=1, event='baseline'`, содержимое целиком.

    Baseline — единственная версия со `snapshot`; все дальнейшие — дельты
    (FR-2.1/FR-2.2). Писать baseline для записи, у которой он уже есть, —
    ошибка вызывающей стороны (история не переписывается).
    """
    _check_entity(entity)
    resolved_source, resolved_session = _origin(source, session)
    conn.execute(
        "INSERT INTO record_versions (entity, entity_id, version, event, source, "
        "session, changed, snapshot) VALUES (?, ?, 1, 'baseline', ?, ?, '[]', ?)",
        (entity, int(entity_id), resolved_source, resolved_session, _dump(fields)),
    )
    return 1


def record_change(
    conn: sqlite3.Connection,
    entity: str,
    entity_id: int,
    fields: Mapping[str, Any],
    *,
    source: str | None = None,
    session: str | None = None,
    event: str = "content",
) -> int | None:
    """Версия-правка: дельта от головной версии, `None` — если не изменилось.

    Пустая дельта версии не создаёт (не плодит шум). Под `current_rollback_to()`
    версия помечается `event='rollback'` с `restore_from` (FR-5.4). Для навыка
    `skills.version` поднимается до головной версии — в ТОЙ ЖЕ транзакции
    (§13.7). После записи применяется политика хранения (`_prune`).
    """
    _check_entity(entity)
    settings = get_settings()
    inline_max = settings.history_delta_inline_max
    entity_id = int(entity_id)
    head = _head(conn, entity, entity_id)
    if head == 0:
        # Запись появилась после baseline-миграции и истории ещё не имеет:
        # пишем baseline (иначе дельта не от чего отсчитывать).
        return record_initial(
            conn, entity, entity_id, fields, source=source, session=session
        )
    previous = reconstruct(conn, entity, entity_id, head, inline_max=inline_max) or {}
    delta, changed = encode_delta(previous, fields, inline_max=inline_max)
    if not changed:
        return None
    version = head + 1
    rollback_to = current_rollback_to()
    if rollback_to is not None:
        event = "rollback"
    resolved_source, resolved_session = _origin(source, session)
    conn.execute(
        "INSERT INTO record_versions (entity, entity_id, version, event, source, "
        "session, changed, delta, restore_from) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            entity,
            entity_id,
            version,
            event,
            resolved_source,
            resolved_session,
            _dump(changed),
            _dump(delta),
            rollback_to if event == "rollback" else None,
        ),
    )
    if entity == ENTITY_SKILL:
        conn.execute(
            "UPDATE skills SET version = ? WHERE id = ?", (version, entity_id)
        )
    _prune(conn, entity, entity_id, settings)
    return version


def record_event(
    conn: sqlite3.Connection,
    entity: str,
    entity_id: int,
    event: str,
    *,
    source: str | None = None,
    session: str | None = None,
) -> int:
    """Версия-событие без контента (`changed='[]'`, дельта NULL): FR-4.3/FR-6.

    Мягкое удаление/восстановление — факт в истории, а не правка полей; для
    навыка зеркало `skills.version` поднимается так же, как у правки.
    """
    _check_entity(entity)
    settings = get_settings()
    entity_id = int(entity_id)
    version = _head(conn, entity, entity_id) + 1
    resolved_source, resolved_session = _origin(source, session)
    conn.execute(
        "INSERT INTO record_versions (entity, entity_id, version, event, source, "
        "session, changed) VALUES (?, ?, ?, ?, ?, ?, '[]')",
        (entity, entity_id, version, event, resolved_source, resolved_session),
    )
    if entity == ENTITY_SKILL:
        conn.execute(
            "UPDATE skills SET version = ? WHERE id = ?", (version, entity_id)
        )
    _prune(conn, entity, entity_id, settings)
    return version


# --- чтение и восстановление (FR-5.1/FR-5.2) --------------------------------

def reconstruct(
    conn: sqlite3.Connection,
    entity: str,
    entity_id: int,
    version: int,
    *,
    inline_max: int,
) -> dict[str, Any] | None:
    """Содержимое версии: от baseline применять дельты до `version` включительно.

    `snapshot` в любой версии перекрывает накопленное состояние — это якорь
    re-base после prune (§13.8). `None` — записи/версии нет.
    """
    _check_entity(entity)
    entity_id = int(entity_id)
    version = int(version)
    rows = conn.execute(
        "SELECT version, snapshot, delta FROM record_versions WHERE entity = ? "
        "AND entity_id = ? AND version <= ? ORDER BY version",
        (entity, entity_id, version),
    ).fetchall()
    if not rows or int(rows[-1]["version"]) != version:
        return None
    fields: dict[str, Any] = {}
    for row in rows:
        if row["snapshot"] is not None:
            fields = json.loads(row["snapshot"])
        elif row["delta"] is not None:
            fields = apply_delta(fields, json.loads(row["delta"]))
    return fields


def read_versions(
    entity: str,
    entity_id: int,
    settings: Settings,
    *,
    limit: int | None = None,
    offset: int = 0,
) -> dict[str, Any]:
    """Список версий записи без тел: метаданные + «что изменилось» (FR-5.1).

    Порядок — от новых к старым (как у операторского архива навыков);
    страница — по контракту lsb-0013 (`total`/`has_more`/`next_offset`).
    """
    _check_entity(entity)
    entity_id = int(entity_id)
    page_size = settings.default_list_limit if limit is None else int(limit)
    offset = int(offset)
    with session(settings) as conn:
        total = int(
            conn.execute(
                "SELECT COUNT(*) FROM record_versions WHERE entity = ? "
                "AND entity_id = ?",
                (entity, entity_id),
            ).fetchone()[0]
        )
        rows = conn.execute(
            "SELECT version, created_at, event, source, session, changed FROM "
            "record_versions WHERE entity = ? AND entity_id = ? "
            "ORDER BY version DESC LIMIT ? OFFSET ?",
            (entity, entity_id, page_size, offset),
        ).fetchall()
    versions = [
        {
            "version": int(row["version"]),
            "created_at": row["created_at"],
            "event": row["event"],
            "source": row["source"],
            "session": row["session"],
            "changed": json.loads(row["changed"]),
        }
        for row in rows
    ]
    return {
        "entity": entity,
        "id": entity_id,
        "versions": versions,
        **page_fields(total, offset, len(versions)),
    }


def read_version(
    entity: str, entity_id: int, version: int, settings: Settings
) -> dict[str, Any]:
    """Содержимое конкретной версии: метаданные + реконструированные поля.

    Нет такой версии/записи — мягкий ответ с `hint` (транспорт превращает его
    в 404); исключения не бросаем: «версии нет» — валидный исход (§6.1).
    """
    _check_entity(entity)
    entity_id = int(entity_id)
    with session(settings) as conn:
        row = conn.execute(
            "SELECT version, created_at, event, source, session, changed FROM "
            "record_versions WHERE entity = ? AND entity_id = ? AND version = ?",
            (entity, entity_id, int(version)),
        ).fetchone()
        if row is None:
            return {
                "entity": entity,
                "id": entity_id,
                "version": int(version),
                "hint": HINT_VERSION_NOT_FOUND,
            }
        fields = reconstruct(
            conn,
            entity,
            entity_id,
            int(row["version"]),
            inline_max=settings.history_delta_inline_max,
        )
    return {
        "entity": entity,
        "id": entity_id,
        "version": int(row["version"]),
        "created_at": row["created_at"],
        "event": row["event"],
        "source": row["source"],
        "session": row["session"],
        "changed": json.loads(row["changed"]),
        "fields": fields,
    }


def history_stats(settings: Settings) -> dict[str, Any]:
    """Размер истории: счётчики версий по сущностям + общий (FR-10.3).

    Форма стабильна: все известные сущности присутствуют, даже с нулём
    (оператору видно, какая область истории ещё пуста).
    """
    with session(settings) as conn:
        counts = {
            str(row["entity"]): int(row["count"])
            for row in conn.execute(
                "SELECT entity, COUNT(*) AS count FROM record_versions GROUP BY entity"
            )
        }
    entities: dict[str, int] = dict.fromkeys(CONTENT_FIELDS, 0)
    entities.update(counts)
    return {"entities": entities, "total": sum(entities.values())}


# --- политика хранения: prune с re-base (§13.8) -----------------------------

def _prune(
    conn: sqlite3.Connection, entity: str, entity_id: int, settings: Settings
) -> None:
    """Почистить старейшие версии записи и перебазировать самую раннюю.

    Потолок `HISTORY_MAX_VERSIONS` и срок `HISTORY_RETENTION_DAYS` (0 — без
    ограничения) сохраняют `version=1` (якорь цепочки) и новейшие версии.
    После удаления самая ранняя сохраняемая версия переписывается в полный
    `snapshot` (`delta=NULL`): без этого её дельта ссылалась бы на удалённую
    версию и цепочка не восстановилась бы. Номера не переиспользуются.
    """
    max_versions = settings.history_max_versions
    retention_days = settings.history_retention_days
    if max_versions <= 0 and retention_days <= 0:
        return
    rows = conn.execute(
        "SELECT version, delta, created_at FROM record_versions WHERE entity = ? "
        "AND entity_id = ? ORDER BY version",
        (entity, entity_id),
    ).fetchall()
    if not rows:
        return
    versions = [int(row["version"]) for row in rows]
    base = versions[0]
    doomed: set[int] = set()
    if max_versions > 0 and len(versions) > max_versions:
        keep_count = max_versions - 1  # version=1 остаётся всегда
        newest = versions[-keep_count:] if keep_count > 0 else []
        keep = {base, *newest}
        doomed |= {version for version in versions if version not in keep}
    if retention_days > 0:
        cutoff = conn.execute(
            "SELECT strftime('%Y-%m-%dT%H:%M:%SZ','now', ?)",
            (f"-{int(retention_days)} days",),
        ).fetchone()[0]
        doomed |= {
            int(row["version"])
            for row in rows
            if int(row["version"]) != base and str(row["created_at"]) < cutoff
        }
    if not doomed:
        return
    survivors = [version for version in versions if version not in doomed]
    # Перебазируем самую раннюю сохраняемую версию, только если под ней версии
    # удаляются (иначе её дельта ссылается на удалённую и цепочка рвётся).
    anchor = survivors[0] if survivors[0] != base else next(iter(survivors[1:]), None)
    snapshot = None
    if anchor is not None and any(version < anchor for version in doomed):
        fields = reconstruct(
            conn, entity, entity_id, anchor, inline_max=settings.history_delta_inline_max
        )
        if fields is not None:
            snapshot = _dump(fields)
    placeholders = ", ".join("?" for _ in doomed)
    conn.execute(
        f"DELETE FROM record_versions WHERE entity = ? AND entity_id = ? "
        f"AND version IN ({placeholders})",
        (entity, entity_id, *sorted(doomed)),
    )
    if snapshot is not None and anchor is not None:
        conn.execute(
            "UPDATE record_versions SET snapshot = ?, delta = NULL WHERE entity = ? "
            "AND entity_id = ? AND version = ?",
            (snapshot, entity, entity_id, anchor),
        )
