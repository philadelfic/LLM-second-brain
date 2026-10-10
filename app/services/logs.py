"""LogsService — сервис области «Логи ассистента» (LSBN-F-10, релиз 3.2.0).

Форма записи, словарь объекта и связки id/версия/ссылка — arch §7.3; проекция и
версионирование — §5.4/§11; контракт сервиса — §6.1.

Ключевые свойства
-----------------
* **Запись лога — не знание** (FR-2.4/§4): сервис пишет только по явному вызову
  агента (`logs_save`) или оператора (REST, 03); автологирования нет, слоты
  своих вызовов в область не пишут.
* **Обязательные поля — `action` + `reason`** (§7.3 п.1): «что сделано» и
  «почему»; объект изменения, результат и «что проверено» необязательны, но
  структурированы. Форма проверяется **до** БД: каждый отказ — свой дословный
  hint (`LogValidationError`, прецедент `TermValidationError`), записи нет.
* **Словарь объекта закрыт** (§5.4): версионные сущности (`note`, `skill`,
  `term`, `user_fact`, `assistant_log`) требуют `entity_id` (и допускают
  `entity_version`), внешние (`file`, `commit`, `tracker_task`, `config`,
  `contour`) — непустой `entity_ref` (путь, хэш коммита, ключ задачи, имя
  конфига).
* **`source`/`session` не выдумывает модель** (§7.1/§13.3): по умолчанию они
  берутся из `history.current_origin()` (`bind_origin` миддлвара), явные
  значения вызывающего сильнее. В ТАБЛИЦЕ лога `source`/`session` — контентные
  поля («какой канал/сессия породили действие»), отдельно от
  `record_versions.source/session` (происхождение самой версии).
* **Версии — тем же слоем истории** (№41, §11/§13.5): `record_initial` при
  создании, `record_change` при фактической правке, `record_event('delete')` при
  мягком удалении — тем же соединением и в той же транзакции. Своих соединений и
  транзакций для истории сервис не открывает: атомарность держит вызывающий.
* **Эмбеддинг на пути записи НЕ зовётся**: вектор (непустые `action` + `reason` +
  `outcome`) догоняет петля `areas` воркера по `vector_status='pending'`, а после
  успешной записи/правки/удаления сервис её будит (`set_areas_notifier`).
* **Изоляция области** (§13.2/§13.4): SQL сервиса адресует только
  `assistant_logs`; состояние объекта изменения (`entity_state`) считает слой
  истории (04), а не сервис логов.
* Поиск/листинг/срезы — 03; `entity_state`/`resolve_entity` — 04.

Контракты ответов (MCP-слой срезает служебные поля белым списком — f10-02):
- save   → {created: True, id, version: 1} | {created: False, hint}
- get    → {id, action, reason, entity, entity_id, entity_version, entity_ref,
            outcome, verified, source, session, created_at, updated_at};
           нет/удалена → {id, hint}
- update → {id, updated: True} | {id, updated: False, hint}
- delete → {id, deleted: True} | {id, deleted: False, hint}
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from typing import Any

from app.config import Settings
from app.services import history
from app.services.embedding import Embedder
# Соединение области: `session` импортируется под псевдонимом — имя `session`
# занято контентным параметром лога (`save`/`update`, arch §5.4/§7.1).
from app.storage.db import session as db_session
from app.storage.db import transaction

# --- Словарь объекта изменения (arch §5.4: «константа в сервисе») ------------

# Версионные сущности памяти — адресуются id (и, если известна, версией).
VERSIONED_ENTITY_KINDS = frozenset(
    {"note", "skill", "term", "user_fact", "assistant_log"}
)
# Объекты вне памяти — id у них нет, ссылка только текстовая.
EXTERNAL_ENTITY_KINDS = frozenset(
    {"file", "commit", "tracker_task", "config", "contour"}
)
# Полный закрытый словарь `entity`.
ENTITY_KINDS = VERSIONED_ENTITY_KINDS | EXTERNAL_ENTITY_KINDS

# Источники записи (§5.1): канал, которым запись пришла в область. Фоновые
# джобы пишут `job:<имя>` (набор открыт — префикс, а не элемент словаря).
LOG_SOURCES = frozenset({"mcp", "rest", "merge", "migration"})
JOB_SOURCE_PREFIX = "job:"

# --- Hint'ы мягких отказов (arch §7.3: «дословные константы в коде») --------

HINT_ACTION_REQUIRED = (
    "not saved: action is required — say what was done (≤500 characters)"
)
HINT_REASON_REQUIRED = (
    "not saved: reason is required — say why it was done (≤500 characters)"
)
HINT_ACTION_LIMIT = "not saved: action limit is 500 characters — shorten it"
HINT_REASON_LIMIT = "not saved: reason limit is 500 characters — shorten it"
HINT_ENTITY_UNKNOWN = (
    "not saved: unknown entity '{given}' — one of: note, skill, term, user_fact, "
    "assistant_log, file, commit, tracker_task, config, contour"
)
HINT_ENTITY_ID_REQUIRED = (
    "not saved: entity '{given}' is versioned — pass entity_id, and "
    "entity_version when the change was made to a known version"
)
HINT_ENTITY_REF_REQUIRED = (
    "not saved: entity '{given}' has no id in this memory — pass entity_ref (a "
    "path, commit hash, task key or config name)"
)
HINT_ENTITY_VERSION_NEEDS_ID = "not saved: entity_version requires entity_id"
HINT_UPDATE_EMPTY = (
    "not updated: nothing passed — pass at least one field to change"
)
HINT_NOT_FOUND = (
    "log entry not found — there is no such entry in this area; save it via "
    "logs_save(action, reason) if it is worth keeping"
)

# Лимиты текстовых полей §7.3 задаёт только для `action`/`reason`; для
# `entity_ref`/`outcome`/`verified` (контракт §6.1: «(5) лимиты …») тексты
# собраны по каноническому шаблону «not saved: <поле> limit is N characters —
# shorten it». Отклонение зафиксировано в отчёте f10-02.
HINT_ENTITY_REF_LIMIT = (
    "not saved: entity_ref limit is 200 characters — shorten it"
)
HINT_OUTCOME_LIMIT = "not saved: outcome limit is 500 characters — shorten it"
HINT_VERIFIED_LIMIT = "not saved: verified limit is 200 characters — shorten it"

# Незнакомый источник записи: таблица §7.3 текста не задаёт (словарь источников
# описан в §5.1), поэтому текст собран тем же шаблоном «not saved: …
# — one of: …» и ведёт к допустимым значениям. Отклонение в отчёте f10-02.
HINT_SOURCE_UNKNOWN = (
    "not saved: unknown source '{given}' — one of: mcp, rest, merge, migration, "
    "or job:<name>"
)

# Сентинел «не передано = оставить» (прецедент `memory_update`/`user_update`):
# отличает «поле не тронуто» от «поле очищено» (пустая строка).
_UNSET: Any = object()


class LogValidationError(ValueError):
    """Нарушение формы записи лога (словарь, связки, лимиты, источник).

    Текст исключения — дословный hint (§7.3): транспорт MCP отдаёт его как
    fail + hint, REST (03) — как 422. Запись при этом НЕ идёт: прецеденты
    `TermValidationError`, `SkillValidationError`, `UserFactValidationError`.
    """


class LogsService:
    """CRUD области «Логи ассистента»: запись, чтение, правка, мягкое удаление.

    DI: `settings` (лимиты формы, флаг области), `embedding` (кодировщик для
    поиска — f10-03; путь записи его НЕ зовёт). Собственного `AreaSearch` сервис
    не держит (§13.2: поиск области — собственный SQL, f10-03), листинга и
    срезов в этой волне нет.
    """

    def __init__(
        self, settings: Settings, embedding: Embedder | None = None
    ) -> None:
        self._settings = settings
        # DI эмбеддера: поиск (03) кодирует запрос тем же слотом, что и заметки.
        self._embedding: Embedder | None = embedding
        # Сигнал воркеру (main.py): будить петлю areas сразу при появлении
        # pending-записи области, а не ждать выросший back-off.
        self._areas_notifier: Callable[[], None] | None = None

    def set_areas_notifier(self, notifier: Callable[[], None] | None) -> None:
        """Подключить сигнал пробуждения петли `areas` воркера (main.py).

        По образцу `SkillsService`/`TermsService`/`UserFactsService`: сервис
        собирается раньше воркера, нотификатор приходит после.
        """
        self._areas_notifier = notifier

    # --- запись (arch §6.1, §7.3) -------------------------------------------

    def save(
        self,
        action: str,
        reason: str,
        *,
        entity: str | None = None,
        entity_id: int | None = None,
        entity_version: int | None = None,
        entity_ref: str | None = None,
        outcome: str | None = None,
        verified: str | None = None,
        source: str | None = None,
        session: str | None = None,
    ) -> dict[str, Any]:
        """Записать лог: «что сделано, почему, объект, результат, что проверено».

        Порядок обязателен: форма проверяется ДО любых обращений к БД (§7.3).
        Нарушение — `LogValidationError` с дословным hint, записи нет. Текстовые
        поля берутся у вызывающего как есть, но с `strip()`.

        `source`/`session` не переданы → берутся из `history.current_origin()`
        (`bind_origin` миддлвара, §7.1): модель источник не выдумывает.

        Запись атомарна: INSERT (`vector_status='pending'`) и baseline-версия
        (`record_initial`) — в одной транзакции; после коммита сервис будит петлю
        `areas` (вектор догоняет воркер). Эмбеддинг не вызывается.
        """
        form = self._checked_form(
            action=action,
            reason=reason,
            entity=entity,
            entity_id=entity_id,
            entity_version=entity_version,
            entity_ref=entity_ref,
            outcome=outcome,
            verified=verified,
        )
        resolved_source, resolved_session = self._resolved_origin(source, session)
        with db_session(self._settings) as conn, transaction(conn):
            cursor = conn.execute(
                "INSERT INTO assistant_logs (action, reason, entity, entity_id, "
                "entity_version, entity_ref, outcome, verified, source, session, "
                "vector_status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')",
                (
                    form["action"],
                    form["reason"],
                    form["entity"],
                    form["entity_id"],
                    form["entity_version"],
                    form["entity_ref"],
                    form["outcome"],
                    form["verified"],
                    resolved_source,
                    resolved_session,
                ),
            )
            log_id = int(cursor.lastrowid or 0)
            # №41 (§13.5): baseline записи — той же транзакцией, проекция §5.4.
            live = self._projection_row(conn, log_id)
            history.record_initial(
                conn,
                history.ENTITY_ASSISTANT_LOG,
                log_id,
                history.assistant_log_projection(live),
            )
        self._notify_areas_pending()
        return {"created": True, "id": log_id, "version": 1}

    def update(
        self,
        id: int,
        *,
        action: str | None = _UNSET,
        reason: str | None = _UNSET,
        entity: str | None = _UNSET,
        entity_id: int | None = _UNSET,
        entity_version: int | None = _UNSET,
        entity_ref: str | None = _UNSET,
        outcome: str | None = _UNSET,
        verified: str | None = _UNSET,
        source: str | None = _UNSET,
        session: str | None = _UNSET,
    ) -> dict[str, Any]:
        """Частично править запись по id: «не передано — оставить» (сентинелы).

        Исходы:

        * ни одно поле не передано → `{id, updated: False, hint}` с
          `HINT_UPDATE_EMPTY`, версий не прибавляется, БД не трогается;
        * записи нет/она мягко удалена → `HINT_NOT_FOUND` (§7.3: удалённая
          запись — не «чужая», а отсутствующая для агента);
        * иначе — UPDATE переданных полей (пустая строка у объекта = очистка),
          `vector_status='pending'`, `updated_at` = now и версия-дельта
          (`record_change`) той же транзакцией; связки объекта проверяются по
          эффективным (текущим + переданным) значениям.

        После успешной правки сервис будит петлю `areas`.
        """
        if all(
            value is _UNSET
            for value in (
                action,
                reason,
                entity,
                entity_id,
                entity_version,
                entity_ref,
                outcome,
                verified,
                source,
                session,
            )
        ):
            return {"id": id, "updated": False, "hint": HINT_UPDATE_EMPTY}
        changes: dict[str, Any] = {}
        if action is not _UNSET:
            changes["action"] = self._checked_action(action)
        if reason is not _UNSET:
            changes["reason"] = self._checked_reason(reason)
        if entity is not _UNSET:
            changes["entity"] = self._checked_entity(entity)
        if entity_ref is not _UNSET:
            changes["entity_ref"] = self._checked_entity_ref(entity_ref)
        if outcome is not _UNSET:
            changes["outcome"] = self._checked_outcome(outcome)
        if verified is not _UNSET:
            changes["verified"] = self._checked_verified(verified)
        if entity_id is not _UNSET:
            changes["entity_id"] = entity_id
        if entity_version is not _UNSET:
            changes["entity_version"] = entity_version
        if source is not _UNSET:
            # Пустой `source` = «не передано»: колонка NOT NULL, источник
            # контекста вызова уже записан при создании.
            checked_source = self._checked_source(source)
            if checked_source is not None:
                changes["source"] = checked_source
        if session is not _UNSET:
            changes["session"] = self._text(session)
        if not changes:
            # Все переданные поля оказались пустыми (= «не передано»).
            return {"id": id, "updated": False, "hint": HINT_UPDATE_EMPTY}
        with db_session(self._settings) as conn, transaction(conn):
            row = conn.execute(
                "SELECT entity, entity_id, entity_version, entity_ref "
                "FROM assistant_logs WHERE id = ? AND deleted_at IS NULL",
                (id,),
            ).fetchone()
            if row is None:
                return {"id": id, "updated": False, "hint": HINT_NOT_FOUND}
            if any(
                field is not _UNSET
                for field in (entity, entity_id, entity_version, entity_ref)
            ):
                # Связки объекта — по эффективным значениям: правка одного поля
                # не должна оставить строку в противоречивом состоянии.
                self._checked_entity_links(
                    changes.get("entity", row["entity"]),
                    changes.get("entity_id", row["entity_id"]),
                    changes.get("entity_version", row["entity_version"]),
                    changes.get("entity_ref", row["entity_ref"]),
                )
            assignments = ", ".join(f"{column} = ?" for column in changes)
            conn.execute(
                f"UPDATE assistant_logs SET {assignments}, vector_status = "
                "'pending', updated_at = strftime('%Y-%m-%dT%H:%M:%SZ','now') "
                "WHERE id = ? AND deleted_at IS NULL",
                (*changes.values(), id),
            )
            # №41 (§13.5): правка — версия-дельта той же транзакцией.
            live = self._projection_row(conn, id)
            history.record_change(
                conn,
                history.ENTITY_ASSISTANT_LOG,
                id,
                history.assistant_log_projection(live),
            )
        self._notify_areas_pending()
        return {"id": id, "updated": True}

    # --- чтение (arch §6.1) -------------------------------------------------

    def get(self, id: int) -> dict[str, Any]:
        """Прочитать запись по id — полный контракт §7.1 (тело не режется).

        Нет записи или она мягко удалена → `{id, hint}` дословного канона §7.3.
        `entity_state` здесь НЕ считается: состояние объекта — слой истории
        (04, §13.4), сервис области в чужие таблицы не ходит.
        """
        with db_session(self._settings) as conn:
            row = conn.execute(
                "SELECT id, action, reason, entity, entity_id, entity_version, "
                "entity_ref, outcome, verified, source, session, created_at, "
                "updated_at FROM assistant_logs "
                "WHERE id = ? AND deleted_at IS NULL",
                (id,),
            ).fetchone()
        if row is None:
            return {"id": id, "hint": HINT_NOT_FOUND}
        return {
            "id": int(row["id"]),
            "action": row["action"],
            "reason": row["reason"],
            "entity": row["entity"],
            "entity_id": row["entity_id"],
            "entity_version": row["entity_version"],
            "entity_ref": row["entity_ref"],
            "outcome": row["outcome"],
            "verified": row["verified"],
            "source": row["source"],
            "session": row["session"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    # --- удаление (soft delete, arch §11) -----------------------------------

    def delete(self, id: int) -> dict[str, Any]:
        """Мягко удалить запись: `deleted_at` = now, строка и индексы живы.

        Идемпотентно по смыслу: повторный/несуществующий id → мягкий ответ
        `deleted: False` + hint канона §7.3 (восстановление — оператором).
        Удаление — факт истории (`record_event('delete')`) в той же транзакции:
        запись лога не исчезает, читается оператором и историей (§11).
        """
        with db_session(self._settings) as conn, transaction(conn):
            cursor = conn.execute(
                "UPDATE assistant_logs SET deleted_at = "
                "strftime('%Y-%m-%dT%H:%M:%SZ','now') "
                "WHERE id = ? AND deleted_at IS NULL",
                (id,),
            )
            deleted = bool(cursor.rowcount)
            if deleted:
                history.record_event(
                    conn, history.ENTITY_ASSISTANT_LOG, id, "delete"
                )
        if not deleted:
            return {"id": id, "deleted": False, "hint": HINT_NOT_FOUND}
        # Сигнал петле areas един по всем изменениям области: у удаления pending
        # не появляется, но сигнал безвреден и держит одну точку пробуждения.
        self._notify_areas_pending()
        return {"id": id, "deleted": True}

    # --- внутреннее: форма (§7.3, порядок обязателен) -----------------------

    def _checked_form(
        self,
        *,
        action: Any,
        reason: Any,
        entity: Any,
        entity_id: int | None,
        entity_version: int | None,
        entity_ref: Any,
        outcome: Any,
        verified: Any,
    ) -> dict[str, Any]:
        """Проверить форму новой записи ДО БД и вернуть её в нормальном виде.

        Порядок §7.3: (1) `action`/`reason` — непустые и в лимитах; (2) `entity`
        из словаря (пусто — допустимо); (3) связки объекта (id у версионных,
        текстовая ссылка у внешних, `entity_version` без id — отказ);
        (4) лимиты `entity_ref`/`outcome`/`verified`. Каждый отказ — свой hint.
        """
        checked_action = self._checked_action(action)
        checked_reason = self._checked_reason(reason)
        checked_entity = self._checked_entity(entity)
        self._checked_entity_links(
            checked_entity, entity_id, entity_version, self._text(entity_ref)
        )
        return {
            "action": checked_action,
            "reason": checked_reason,
            "entity": checked_entity,
            "entity_id": None if entity_id is None else int(entity_id),
            "entity_version": (
                None if entity_version is None else int(entity_version)
            ),
            "entity_ref": self._checked_entity_ref(entity_ref),
            "outcome": self._checked_outcome(outcome),
            "verified": self._checked_verified(verified),
        }

    def _checked_entity_links(
        self,
        entity: str | None,
        entity_id: int | None,
        entity_version: int | None,
        entity_ref: str | None,
    ) -> None:
        """Связки объекта изменения: id / версия / текстовая ссылка (§7.3 п.3)."""
        if entity in VERSIONED_ENTITY_KINDS and entity_id is None:
            raise LogValidationError(
                HINT_ENTITY_ID_REQUIRED.format(given=entity)
            )
        if entity_version is not None and entity_id is None:
            raise LogValidationError(HINT_ENTITY_VERSION_NEEDS_ID)
        if entity in EXTERNAL_ENTITY_KINDS and not entity_ref:
            raise LogValidationError(
                HINT_ENTITY_REF_REQUIRED.format(given=entity)
            )

    def _checked_action(self, action: str | None) -> str:
        """Обязательный `action`: непустой и ≤ `logs_action_max_chars`."""
        value = self._text(action)
        if not value:
            raise LogValidationError(HINT_ACTION_REQUIRED)
        if len(value) > self._settings.logs_action_max_chars:
            raise LogValidationError(HINT_ACTION_LIMIT)
        return value

    def _checked_reason(self, reason: str | None) -> str:
        """Обязательный `reason`: непустой и ≤ `logs_reason_max_chars`."""
        value = self._text(reason)
        if not value:
            raise LogValidationError(HINT_REASON_REQUIRED)
        if len(value) > self._settings.logs_reason_max_chars:
            raise LogValidationError(HINT_REASON_LIMIT)
        return value

    def _checked_entity(self, entity: str | None) -> str | None:
        """`entity` — тип объекта изменения: пусто или из закрытого словаря.

        Незнакомый тип → мягкий отказ с перечислением словаря (модель не
        додумывает тип объекта: он идёт в обратный запрос «почему изменилось»).
        """
        value = self._text(entity)
        if value is None:
            return None
        if value not in ENTITY_KINDS:
            raise LogValidationError(HINT_ENTITY_UNKNOWN.format(given=value))
        return value

    def _checked_entity_ref(self, entity_ref: str | None) -> str | None:
        """`entity_ref` — текстовая ссылка на объект вне памяти (≤ 200)."""
        value = self._text(entity_ref)
        if value is not None and (
            len(value) > self._settings.logs_entity_ref_max_chars
        ):
            raise LogValidationError(HINT_ENTITY_REF_LIMIT)
        return value

    def _checked_outcome(self, outcome: str | None) -> str | None:
        """`outcome` — результат работы (≤ `logs_outcome_max_chars`)."""
        value = self._text(outcome)
        if value is not None and len(value) > self._settings.logs_outcome_max_chars:
            raise LogValidationError(HINT_OUTCOME_LIMIT)
        return value

    def _checked_verified(self, verified: str | None) -> str | None:
        """`verified` — что проверено (≤ `logs_verified_max_chars`)."""
        value = self._text(verified)
        if value is not None and (
            len(value) > self._settings.logs_verified_max_chars
        ):
            raise LogValidationError(HINT_VERIFIED_LIMIT)
        return value

    def _checked_source(self, source: str | None) -> str | None:
        """Явно переданный `source` — из словаря §5.1 или `job:<имя>`.

        Источник не выдумывается моделью (§7.1): незнакомое значение — мягкий
        отказ тем же стилем hint. Пустая строка = «не передано».
        """
        value = self._text(source)
        if value is None:
            return None
        if value not in LOG_SOURCES and not value.startswith(JOB_SOURCE_PREFIX):
            raise LogValidationError(HINT_SOURCE_UNKNOWN.format(given=value))
        return value

    def _resolved_origin(
        self, source: str | None, session: str | None
    ) -> tuple[str | None, str | None]:
        """Источник/сессия записи: явные аргументы сильнее текущего `bind_origin`.

        Оба поля независимы: не переданное берётся из `history.current_origin()`
        (миддлварь вызова), переданное — проверяется (источник) и чистится
        (`strip`).
        """
        current_source, current_session = history.current_origin()
        if source is None:
            resolved_source: str | None = current_source
        else:
            # Пустая строка = «не передано»: берём источник контекста вызова.
            resolved_source = self._checked_source(source) or current_source
        resolved_session = (
            current_session if session is None else self._text(session)
        )
        return resolved_source, resolved_session

    # --- внутреннее: утилиты ------------------------------------------------

    @staticmethod
    def _text(value: Any) -> str | None:
        """Текст поля «как есть», но без краевых пробелов; пусто → `None`."""
        if not isinstance(value, str):
            return None
        stripped = value.strip()
        return stripped or None

    @staticmethod
    def _projection_row(conn: sqlite3.Connection, log_id: int) -> sqlite3.Row:
        """Строка проекции лога после записи — одна форма baseline/дельты."""
        return conn.execute(
            "SELECT action, reason, entity, entity_id, entity_version, "
            "entity_ref, outcome, verified, source, session "
            "FROM assistant_logs WHERE id = ?",
            (log_id,),
        ).fetchone()

    def _notify_areas_pending(self) -> None:
        """Сигнал воркеру: область изменилась (будить петлю `areas` сразу)."""
        if self._areas_notifier is not None:
            self._areas_notifier()
