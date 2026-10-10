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
* **Поиск и листинг — собственный SQL области** (§13.2, f10-03): гибрид
  vec0-KNN + FTS5/BM25 → RRF со срезами по объекту/сессии/источнику/абсолютным
  датам; `entity_state`/`resolve_entity` — 04.

Контракты ответов (MCP-слой срезает служебные поля белым списком — f10-02):
- save   → {created: True, id, version: 1} | {created: False, hint}
- get    → {id, action, reason, entity, entity_id, entity_version, entity_ref,
            outcome, verified, source, session, created_at, updated_at};
           нет/удалена → {id, hint}
- update → {id, updated: True} | {id, updated: False, hint}
- delete → {id, deleted: True} | {id, deleted: False, hint}
- search → {results: [{id, action, entity, entity_id, created_at, score}],
            warning?, hint?} — `action` обрезан до excerpt (f10-03)
- list   → {items: [{id, action, entity, entity_id, created_at}], total,
            has_more, next_offset, next_cursor} — `action` целиком (f10-03);
           обрезает только MCP-выдача, REST оператора — полный текст
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from typing import Any

from app.config import Settings
from app.services import history
# Пустые исходы области и деградация FTS-only — общие константы (arch §7.3/§8:
# «переиспользуются как у других областей»), свои тексты не заводим.
from app.services.areas import HINT_NO_RESULTS, HINT_SHORT_QUERY
from app.services.embedding import Embedder, EmbeddingError
# Абсолютные даты срезов — парсер блока 4 (№29, arch §8/§13.12): свой не пишем.
from app.services.freshness import (
    FreshnessValidationError,
    parse_updated_after,
)
from app.services.listing import page_fields
from app.services.ranking import fuse_rrf, match_expression
from app.services.search import CANDIDATE_LIMIT, MAX_TOP_K, WARNING_FTS_ONLY
from app.storage import area_vectors
from app.storage.db import (
    ASSISTANT_LOGS_FTS_COLUMNS,
    ASSISTANT_LOGS_FTS_TABLE,
    ASSISTANT_LOGS_TABLE,
)
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
# Срезы по дате — только абсолютные (§7.3, §8/§13.12): относительная
# формулировка отвергается тем же стилем мягкого отказа.
HINT_DATE_ABSOLUTE = (
    "not filtered: dates are absolute only (YYYY-MM-DD); relative dates "
    "are not supported"
)
# Обратный диапазон канон §7.3 не описывает (там только HINT_DATE_ABSOLUTE):
# текст собран по стилю «not filtered: …» и ведёт к исправлению границ.
# Отклонение зафиксировано в отчёте f10-03.
HINT_DATE_RANGE = (
    "not filtered: date_from is later than date_to — swap the bounds"
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
    не держит (§13.2: поиск области — собственный SQL); поиск, листинг и
    срезы (f10-03) стоят на тех же примитивах (`match_expression`, `fuse_rrf`,
    `area_vectors.knn`, порог, `WARNING_FTS_ONLY`), но адресуют только свои
    таблицы (§6.1/§8).
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

    # --- поиск и срезы (arch §6.1/§8, f10-03) -------------------------------

    def search(
        self,
        query: str,
        top_k: int | None = None,
        *,
        entity: str | None = None,
        entity_id: int | None = None,
        entity_version: int | None = None,
        session: str | None = None,
        source: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> dict[str, Any]:
        """Гибридный поиск по записям области со срезами (arch §6.1/§8).

        Векторная сторона — KNN по `assistant_logs_vec` (top `CANDIDATE_LIMIT`,
        гейт `settings.score_threshold`); FTS-сторона — BM25 по
        `assistant_logs_fts`; слияние — `fuse_rrf` (общий `settings.rrf_k`),
        дедупликация id. **Собственный SQL области** (§13.2): `AreaSearch`
        не умеет срезы, а расширять общий помощник — трогать чужие области.

        Срезы (§8) — по объекту (`entity`/`entity_id`/`entity_version`),
        сессии, источнику и абсолютным датам; одни и те же предикаты идут в
        оба `WHERE` (FTS — прямо в SQL, вектор — постовым отсечением в
        `_filter_rows`). Пустой запрос — мягкий `HINT_SHORT_QUERY`, пустой
        результат — `HINT_NO_RESULTS`; отказ кодировщика не ломает поиск:
        FTS-only + `WARNING_FTS_ONLY` (NFR-3). `action` в выдаче обрезается
        до `logs_search_excerpt_chars` (FR-2.3): тело читается по id.
        """
        top_k = self._settings.default_top_k if top_k is None else top_k
        if not 1 <= top_k <= MAX_TOP_K:
            raise LogValidationError(
                f"top_k: expected 1..{MAX_TOP_K}, got {top_k}"
            )
        where_sql, params = self._slice_where(
            entity=entity,
            entity_id=entity_id,
            entity_version=entity_version,
            session=session,
            source=source,
            date_from=date_from,
            date_to=date_to,
        )
        stripped = query.strip() if isinstance(query, str) else ""
        if not stripped:
            # Пустой запрос: искать нечего — мягкий ответ, без эмбеддера.
            return {"results": [], "warning": None, "hint": HINT_SHORT_QUERY}
        query_vector = self._query_vector(stripped)
        expression = match_expression(stripped)
        with db_session(self._settings) as conn:
            rankings: list[list[int]] = []
            if query_vector is not None:
                hits = area_vectors.knn(
                    conn,
                    area_vectors.ASSISTANT_LOGS_VEC_TABLE,
                    area_vectors.ASSISTANT_LOGS_VEC_ID_COLUMN,
                    query_vector,
                    CANDIDATE_LIMIT,
                )
                # Гейт порога косинуса (прецедент заметок/областей): без него
                # «пусто» стало бы недостижимым при любой непустой таблице.
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
                        for row in self._fts_candidates(
                            conn, expression, where_sql, params
                        )
                    ]
                )
            scores = fuse_rrf(rankings, self._settings.rrf_k)
            # Мягкое чтение + тот же срез: удалённые исчезают, чужие срезы
            # отсекаются и у векторной стороны (vec0 фильтров не знает).
            rows = self._filter_rows(conn, list(scores), where_sql, params)
        results = self._merged_search_results(scores, rows)[:top_k]
        warning = None if query_vector is not None else WARNING_FTS_ONLY
        if not results:
            hint = (
                HINT_SHORT_QUERY
                if query_vector is None and expression is None
                else HINT_NO_RESULTS
            )
            return {"results": [], "warning": warning, "hint": hint}
        return {"results": results, "warning": warning}

    def list(
        self,
        *,
        entity: str | None = None,
        entity_id: int | None = None,
        entity_version: int | None = None,
        session: str | None = None,
        source: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int | None = None,
        offset: int = 0,
        max_limit: int | None = None,
    ) -> dict[str, Any]:
        """Листинг со срезами: новое первым, страница по контракту lsb-0013.

        Порядок — `created_at DESC, id DESC` (индекс
        `idx_assistant_logs_entity_active`); `total` — число активных записей
        с ТЕМ ЖЕ срезом; page-поля — `listing.page_fields` (`next_cursor`
        зарезервирован под будущий keyset). Потолок `limit` — `max_limit`
        (None → `settings.list_max_limit_rest`: MCP передаёт свой, меньший
        потолок); вне `1..max_limit` и `offset < 0` — мягкий отказ
        `LogValidationError` (прецедент `SkillsService.list`). Тело `action`
        не режется: полный текст нужен оператору, MCP режет сам.
        """
        ceiling = (
            self._settings.list_max_limit_rest if max_limit is None
            else max_limit
        )
        limit = self._settings.default_list_limit if limit is None else limit
        if not 1 <= limit <= ceiling:
            raise LogValidationError(
                f"limit: expected 1..{ceiling}, got {limit}"
            )
        if offset < 0:
            raise LogValidationError(
                f"offset: expected 0 or greater, got {offset}"
            )
        where_sql, params = self._slice_where(
            entity=entity,
            entity_id=entity_id,
            entity_version=entity_version,
            session=session,
            source=source,
            date_from=date_from,
            date_to=date_to,
        )
        with db_session(self._settings) as conn:
            total = int(
                conn.execute(
                    f"SELECT COUNT(*) FROM assistant_logs t WHERE {where_sql}",
                    params,
                ).fetchone()[0]
            )
            rows = conn.execute(
                f"SELECT t.id, t.action, t.entity, t.entity_id, t.created_at "
                f"FROM assistant_logs t WHERE {where_sql} "
                "ORDER BY t.created_at DESC, t.id DESC LIMIT ? OFFSET ?",
                (*params, limit, offset),
            ).fetchall()
        items = [
            {
                "id": int(row["id"]),
                "action": row["action"],
                "entity": row["entity"],
                "entity_id": row["entity_id"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]
        return {"items": items, **page_fields(total, offset, len(items))}

    # --- внутреннее: срезы и источники кандидатов (arch §8) ----------------

    def _slice_where(
        self,
        *,
        entity: str | None,
        entity_id: int | None,
        entity_version: int | None,
        session: str | None,
        source: str | None,
        date_from: str | None,
        date_to: str | None,
    ) -> tuple[str, list[Any]]:
        """Один SQL-предикат срезов для `search` и `list` (arch §8).

        Возвращает `("t.… AND t.…", [параметры])` — готовую часть `WHERE` для
        алиаса `t` таблицы области: мягкое чтение (`deleted_at IS NULL`) плюс
        переданные срезы. Фильтры валидируются ДО БД (словарь `entity`, связка
        `entity_version`↔`entity_id`, абсолютные даты): нарушение —
        `LogValidationError` с hint'ом канона.
        """
        checked_entity = self._checked_entity(entity)
        if entity_version is not None and entity_id is None:
            raise LogValidationError(HINT_ENTITY_VERSION_NEEDS_ID)
        start = self._checked_date(date_from)
        end = self._checked_date(date_to)
        if start is not None and end is not None and start > end:
            raise LogValidationError(HINT_DATE_RANGE)
        clauses = ["t.deleted_at IS NULL"]
        params: list[Any] = []
        if checked_entity is not None:
            clauses.append("t.entity = ?")
            params.append(checked_entity)
        if entity_id is not None:
            clauses.append("t.entity_id = ?")
            params.append(int(entity_id))
        if entity_version is not None:
            clauses.append("t.entity_version = ?")
            params.append(int(entity_version))
        session_value = self._text(session)
        if session_value is not None:
            clauses.append("t.session = ?")
            params.append(session_value)
        source_value = self._text(source)
        if source_value is not None:
            clauses.append("t.source = ?")
            params.append(source_value)
        if start is not None:
            # Абсолютный день UTC, включительно с обеих сторон (§8/§13.12).
            clauses.append("substr(t.created_at, 1, 10) >= ?")
            params.append(start)
        if end is not None:
            clauses.append("substr(t.created_at, 1, 10) <= ?")
            params.append(end)
        return " AND ".join(clauses), params

    def _checked_date(self, value: str | None) -> str | None:
        """`date_from`/`date_to`: только абсолютная дата `YYYY-MM-DD` (§8).

        Разбор — парсером блока 4 (`freshness.parse_updated_after`, №29):
        принимается ISO-8601 UTC, но срез держит только день; относительная
        формулировка → `HINT_DATE_ABSOLUTE` (мягкий отказ, §7.3/§13.12).
        """
        text = self._text(value)
        if text is None:
            return None
        try:
            return parse_updated_after(text)[:10]
        except FreshnessValidationError as exc:
            raise LogValidationError(HINT_DATE_ABSOLUTE) from exc

    def _fts_candidates(
        self,
        conn: sqlite3.Connection,
        expression: str,
        where_sql: str,
        params: list[Any],
    ) -> list[sqlite3.Row]:
        """Топ-`CANDIDATE_LIMIT` FTS5/BM25 по активным записям со срезом.

        Зеркалит `AreaSearch._fts_candidates`, но по своим таблицам области и
        с общим предикатом срезов; rank с 1 для RRF.
        """
        weights = tuple(1.0 for _ in ASSISTANT_LOGS_FTS_COLUMNS)
        weight_placeholders = ", ".join("?" * len(weights))
        return conn.execute(
            f"SELECT t.id, t.created_at, "
            f"       bm25({ASSISTANT_LOGS_FTS_TABLE}, {weight_placeholders}) "
            "AS badness "
            f"FROM {ASSISTANT_LOGS_FTS_TABLE} JOIN {ASSISTANT_LOGS_TABLE} t "
            f"ON t.id = {ASSISTANT_LOGS_FTS_TABLE}.rowid "
            f"WHERE {ASSISTANT_LOGS_FTS_TABLE} MATCH ? AND {where_sql} "
            "ORDER BY badness, t.created_at DESC, t.id DESC LIMIT ?",
            (*weights, expression, *params, CANDIDATE_LIMIT),
        ).fetchall()

    def _filter_rows(
        self,
        conn: sqlite3.Connection,
        ids: list[int],
        where_sql: str,
        params: list[Any],
    ) -> dict[int, sqlite3.Row]:
        """Активные кандидаты, прошедшие ТОТ ЖЕ срез, что и FTS-сторона.

        Векторная сторона vec0 фильтров не знает — срезы применяются здесь,
        одним предикатом (§8: «фильтры применяются к обеим сторонам»).
        """
        if not ids:
            return {}
        placeholders = ", ".join("?" * len(ids))
        rows = conn.execute(
            "SELECT t.id, t.action, t.entity, t.entity_id, t.created_at "
            f"FROM assistant_logs t WHERE {where_sql} "
            f"AND t.id IN ({placeholders})",
            (*params, *ids),
        ).fetchall()
        return {int(row["id"]): row for row in rows}

    def _merged_search_results(
        self, scores: dict[int, float], rows: dict[int, sqlite3.Row]
    ) -> list[dict[str, Any]]:
        """Сортировка score DESC → created_at DESC → id DESC и обрезка action."""
        excerpt = self._settings.logs_search_excerpt_chars
        candidates = [
            (scores[row_id], row)
            for row_id, row in rows.items()
            if row_id in scores
        ]
        candidates.sort(
            key=lambda pair: (pair[0], pair[1]["created_at"], pair[1]["id"]),
            reverse=True,
        )
        return [
            {
                "id": int(row["id"]),
                "action": self._excerpt(row["action"], excerpt),
                "entity": row["entity"],
                "entity_id": row["entity_id"],
                "created_at": row["created_at"],
                "score": score,
            }
            for score, row in candidates
        ]

    def _query_vector(self, query: str) -> list[float] | None:
        """Кодирование запроса; отказ эмбеддера — деградация к FTS (§8, NFR-3).

        Эмбеддер не сконфигурирован (DI пустого сервиса) — тоже FTS-only:
        поиск остаётся живым без внешней зависимости.
        """
        if self._embedding is None:
            return None
        try:
            return self._embedding.embed(query)
        except EmbeddingError:
            return None

    @staticmethod
    def _excerpt(text: str, limit: int) -> str:
        """Обрезать тело для выдачи поиска (полный текст — `logs_get`)."""
        return text if len(text) <= limit else text[:limit]

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
