"""Джоба `warnings` и её состояние (№12, релиз 3.2.0; arch §5.1/§6.5/§13.5).

Фоновая джоба `warnings` проходит по **размещённым** заметкам (название +
готовая выжимка + структура), сверяет фактический `namespace` с разметкой
классификатора и **подсвечивает** расхождение — БЕЗ автоматического переноса
(решение О. 2026-10-08, FR-2.2/§13.5). Результат — строка общей поверхности
`warnings` (`app/storage/db.py`), читаемая без вызовов модели (surface — no12-02).

Состояние (таблица `warnings`) обслуживает `WarningsService`:

* `open` — upsert по открытому ключу `(subject_type, subject_id, kind)`: не
  более одного ОТКРЫТОГО предупреждения (partial unique index, §5.1),
  повторный прогон двигает `message`/`payload`/`updated_at`, а не плодит строки
  (идемпотентность, №12 FR-4.4/§13.7);
* `resolve` — причина ушла (авто): `resolved_at=now`, `status='open'` остаётся;
* `resolve_for_subject` — снятие всех предупреждений субъекта при правке
  заметки (№12 FR-4.4); вызывается в ТОЙ ЖЕ транзакции через узкую инъекцию
  `NoteService.set_warnings_resolver`;
* `dismiss` — снятие человеком (`status='dismissed'`, REST — no12-02);
* `list`/`counts` — чтение только SQL, без моделей (FR-3.2, §13.4);
* `check_note`/`run` — прогон джобы с бюджетом вызовов классификатора.

Джоба регистрируется на каркасе lsb-0014 (сборщик + строка в `JOB_BUILDERS`,
импорт односторонний — warnings → jobs, круга нет) и объявляет слот `summary`
(та же модель, что у джобы `nodes`). Свежий `default` оставлен джобе `nodes`
(§13.5): две джобы не спорят за один узел.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from app.config import Settings
from app.services import jobs
from app.services.classifier import ClassificationError
from app.services.jobs import JobSpec, queue_snapshot
from app.services.listing import page_fields
from app.services.slot_outcome import outcome_of
from app.storage.db import session, transaction

if TYPE_CHECKING:  # рантайм-зависимостей у сервиса нет (DI)
    from app.services.namespaces import NamespaceService
    from app.services.worker import BackgroundWorker

# Имя джобы в журнале и очередь для `/health.queues`.
WARNINGS_JOB = "warnings"

# Словарь видов предупреждений — единый для всех поверхностей (№12 + №32).
# Две последние заводит №32, но значения фиксируются здесь.
KIND_NOTE_MISPLACED = "note_misplaced"
KIND_NOTE_NO_NODE = "note_no_node"
KIND_TERM_CONTEXTS_EXCESS = "term_contexts_excess"
KIND_TERM_CONTEXT_CLUSTER = "term_context_cluster"

# Тексты сообщений — обезличенные, английские (без имён людей и слова
# «оператор»): собираются при открытии, читаются без join'а к текстам.
MESSAGE_MISPLACED = (
    "note {id} is in '{namespace}' but the classifier places it in "
    "'{hint_path}' (confidence {confidence})"
)
MESSAGE_NO_NODE = (
    "note {id}: the classifier suggests node '{hint_path}', which does not "
    "exist yet"
)

# Отбор кандидатов (SQL, без модели, arch §6.5): активные заметки с готовой
# непустой выжимкой, РАЗМЕЩЁННЫЕ (`namespace != 'default'` ИЛИ уже разобранные
# `nodes`), ещё не проверенные после последней правки, свежие первыми.
_CANDIDATE_WHERE = (
    "deleted_at IS NULL "
    "AND summary_status = 'ok' AND summary IS NOT NULL AND summary != '' "
    "AND (namespace != 'default' OR node_order_at IS NOT NULL) "
    "AND (warnings_at IS NULL OR warnings_at < updated_at)"
)
_CANDIDATE_SELECT = (
    "SELECT id, updated_at FROM notes WHERE " + _CANDIDATE_WHERE
    + " ORDER BY updated_at DESC, id DESC LIMIT ?"
)
_CANDIDATE_STAT_SELECT = (
    "SELECT COUNT(*) AS pending, "
    "CAST(strftime('%s','now') - strftime('%s', MIN(updated_at)) AS INTEGER) "
    "AS oldest_pending_sec FROM notes WHERE " + _CANDIDATE_WHERE
)

# Столбцы листинга (`GET /warnings`, §7.1) — форма ответа и пагинация в no12-02.
_LIST_COLUMNS = (
    "id, kind, subject_type, subject_id, message, payload, status, "
    "created_at, updated_at, resolved_at"
)


class WarningsValidationError(ValueError):
    """Форма листинга предупреждений нарушена (`limit`/`offset`).

    Мягкий отказ: транспорт отвечает `hint` (MCP) либо 422 с текстом (REST),
    прецедент lsb-0013 — общий с листингами памяти.
    """


class WarningsService:
    """Состояние предупреждений и прогон джобы `warnings` (arch §6.5).

    Зависимости — узкая инъекция (`classifier`/`namespaces`): в юнит-тестах
    классификатор — фейк, в прогоне — та же модель слота `summary`, что у
    джобы `nodes`. `conn_factory` — точка сборки соединений (по умолчанию
    `session`); при передаче готового `conn` сервис работает в чужой
    транзакции (правка заметки, arch §6.5).
    """

    def __init__(
        self,
        settings: Settings,
        *,
        classifier: Any = None,
        namespaces: "NamespaceService | None" = None,
        conn_factory: Callable[[Settings], Any] = session,
    ) -> None:
        self._settings = settings
        self._classifier = classifier
        self._namespaces = namespaces
        self._conn = conn_factory
        # Итог последнего `check_note`: none | opened | updated | resolved |
        # failed. Нужен прогону, чтобы посчитать `warnings_run`, не меняя
        # контракт возврата `check_note` (str | None).
        self._last_check = "none"
        # Была ли последняя `open` вставкой (а не обновлением) — счётчик.
        self._last_open_created = False

    # --- состояние предупреждений -----------------------------------------

    @contextmanager
    def _connection(self, conn: sqlite3.Connection | None) -> Iterator[Any]:
        """Своё соединение с транзакцией или уже открытое (чужая транзакция)."""
        if conn is not None:
            yield conn
            return
        with self._conn(self._settings) as own, transaction(own):
            yield own

    def open(
        self,
        kind: str,
        subject_type: str,
        subject_id: int,
        message: str,
        payload: dict,
        *,
        conn: sqlite3.Connection | None = None,
    ) -> int:
        """Открыть/обновить предупреждение (upsert по открытому ключу).

        Событие `warning_opened` пишется ТОЛЬКО при вставке: повторный прогон
        обновляет строку и журнал не дублирует (идемпотентность, §13.7).
        """
        payload_json = json.dumps(payload or {}, ensure_ascii=False, sort_keys=True)
        with self._connection(conn) as active:
            row = active.execute(
                "SELECT id FROM warnings WHERE subject_type = ? "
                "AND subject_id = ? AND kind = ? AND status = 'open' "
                "AND resolved_at IS NULL",
                (subject_type, subject_id, kind),
            ).fetchone()
            if row is not None:
                warning_id = int(row["id"])
                active.execute(
                    "UPDATE warnings SET message = ?, payload = ?, "
                    "updated_at = strftime('%Y-%m-%dT%H:%M:%SZ','now') "
                    "WHERE id = ?",
                    (message, payload_json, warning_id),
                )
                self._last_open_created = False
                return warning_id
            cursor = active.execute(
                "INSERT INTO warnings "
                "(kind, subject_type, subject_id, message, payload) "
                "VALUES (?, ?, ?, ?, ?)",
                (kind, subject_type, subject_id, message, payload_json),
            )
        warning_id = int(cursor.lastrowid)
        self._last_open_created = True
        self._log(
            "warnings: opened warning",
            "warning_opened",
            warning_id=warning_id,
            kind=kind,
            subject_type=subject_type,
            subject_id=subject_id,
        )
        return warning_id

    def resolve(self, kind: str, subject_type: str, subject_id: int) -> int:
        """Снять предупреждение — причина ушла (авто); число снятых строк.

        `status='open'` остаётся, `resolved_at=now`: снятое не возвращается в
        открытую выборку и не удаляется (аудит, прецедент `promotions`).
        """
        with self._connection(None) as conn:
            count = self._resolve_in(conn, subject_type, subject_id, kind=kind)
        if count:
            self._log(
                "warnings: resolved warning",
                "warning_resolved",
                kind=kind,
                subject_type=subject_type,
                subject_id=subject_id,
                resolved=count,
            )
        return count

    def resolve_for_subject(
        self,
        subject_type: str,
        subject_id: int,
        *,
        conn: sqlite3.Connection | None = None,
    ) -> int:
        """Снять ВСЕ открытые предупреждения субъекта (правка заметки, FR-4.4).

        Вызывается в ТОЙ ЖЕ транзакции, что и правка (готовый `conn`).
        """
        with self._connection(conn) as active:
            count = self._resolve_in(active, subject_type, subject_id)
        if count:
            self._log(
                "warnings: resolved warnings on subject edit",
                "warning_resolved",
                subject_type=subject_type,
                subject_id=subject_id,
                resolved=count,
            )
        return count

    def _resolve_in(
        self,
        conn: sqlite3.Connection,
        subject_type: str,
        subject_id: int,
        *,
        kind: str | None = None,
    ) -> int:
        sql = (
            "UPDATE warnings SET resolved_at = "
            "strftime('%Y-%m-%dT%H:%M:%SZ','now'), updated_at = "
            "strftime('%Y-%m-%dT%H:%M:%SZ','now') WHERE subject_type = ? "
            "AND subject_id = ? AND status = 'open' AND resolved_at IS NULL"
        )
        params: list[object] = [subject_type, subject_id]
        if kind is not None:
            sql += " AND kind = ?"
            params.append(kind)
        return int(conn.execute(sql, params).rowcount)

    def dismiss(self, warning_id: int) -> dict:
        """Снять предупреждение человеком; нет/уже снято — мягкий ответ."""
        with self._connection(None) as conn:
            cursor = conn.execute(
                "UPDATE warnings SET status = 'dismissed', resolved_at = "
                "strftime('%Y-%m-%dT%H:%M:%SZ','now'), updated_at = "
                "strftime('%Y-%m-%dT%H:%M:%SZ','now') WHERE id = ? "
                "AND status = 'open' AND resolved_at IS NULL",
                (warning_id,),
            )
            if cursor.rowcount == 0:
                return {
                    "id": warning_id,
                    "dismissed": False,
                    "hint": "warning not found or already resolved/dismissed",
                }
            row = conn.execute(
                "SELECT resolved_at FROM warnings WHERE id = ?", (warning_id,)
            ).fetchone()
        self._log(
            "warnings: dismissed warning",
            "warning_dismissed",
            warning_id=warning_id,
        )
        return {
            "id": warning_id,
            "dismissed": True,
            "status": "dismissed",
            "resolved_at": row["resolved_at"],
        }

    def list(
        self,
        *,
        kind: str | None = None,
        subject_type: str | None = None,
        subject_id: int | None = None,
        status: str = "open",
        limit: int = 20,
        offset: int = 0,
        max_limit: int = 50,
    ) -> dict:
        """Листинг предупреждений — только SQL, без моделей (FR-3.2, §13.4).

        Форма общая с листингами памяти: поля страницы считает общий хелпер
        `page_fields` (прецедент `NoteService.list`). `limit` вне
        1..`max_limit` (и отрицательный `offset`) — `WarningsValidationError`:
        транспорт отвечает мягко (прецедент lsb-0013), а не молча режет.
        """
        limit = int(limit)
        offset = int(offset)
        if not 1 <= limit <= max_limit:
            raise WarningsValidationError(
                f"limit: expected 1..{max_limit}, got {limit}"
            )
        if offset < 0:
            raise WarningsValidationError(f"offset: expected ≥ 0, got {offset}")
        where: list[str] = []
        params: list[object] = []
        for column, value in (
            ("kind", kind),
            ("subject_type", subject_type),
            ("subject_id", subject_id),
            ("status", status),
        ):
            if value is not None:
                where.append(f"{column} = ?")
                params.append(value)
        clause = (" WHERE " + " AND ".join(where)) if where else ""
        with self._connection(None) as conn:
            total = int(
                conn.execute(
                    "SELECT COUNT(*) AS n FROM warnings" + clause, params
                ).fetchone()["n"]
            )
            rows = conn.execute(
                f"SELECT {_LIST_COLUMNS} FROM warnings" + clause
                + " ORDER BY id DESC LIMIT ? OFFSET ?",
                [*params, limit, offset],
            ).fetchall()
        warnings = [self._list_dict(row) for row in rows]
        return {"warnings": warnings, **page_fields(total, offset, len(warnings))}

    def counts(self) -> dict:
        """Агрегаты для `/health.warnings`: открытые по видам + total (§7.1)."""
        with self._connection(None) as conn:
            rows = conn.execute(
                "SELECT kind, COUNT(*) AS n FROM warnings WHERE status = 'open' "
                "AND resolved_at IS NULL GROUP BY kind"
            ).fetchall()
        by_kind = {row["kind"]: int(row["n"]) for row in rows}
        return {"by_kind": by_kind, "total": sum(by_kind.values())}

    @staticmethod
    def _list_dict(row: sqlite3.Row) -> dict:
        """Строка `warnings` — в форму ответа (payload разбирается из JSON)."""
        data = {column: row[column] for column in _LIST_COLUMNS.split(", ")}
        try:
            data["payload"] = json.loads(data["payload"] or "{}")
        except (TypeError, ValueError):
            data["payload"] = {}
        return data

    # --- прогон джобы ------------------------------------------------------

    def check_note(self, note_id: int) -> str | None:
        """Один прогон проверки заметки; kind предупреждения либо None.

        Читает `title`, `summary`, `namespace`; зовёт классификатор
        (`title` + `summary`, известные узлы реестра) и принимает решение по
        §6.5. Отказ классификатора (`ClassificationError`) заметку НЕ помечает
        (маркер не ставится, повтор следующим прогоном — контракт №39):
        пишется событие `warnings_classifier_failed` с `outcome`.
        """
        self._last_check = "none"
        with self._conn(self._settings) as conn:
            row = conn.execute(
                "SELECT title, summary, namespace, summary_status FROM notes "
                "WHERE id = ? AND deleted_at IS NULL",
                (note_id,),
            ).fetchone()
        if row is None:
            return None
        summary = row["summary"] or ""
        if row["summary_status"] != "ok" or not summary.strip():
            return None
        if self._classifier is None:
            return None
        known = self._namespaces.list_all()["namespaces"]
        title = row["title"] or ""
        text = f"{title}\n{summary}" if title else summary
        try:
            result = self._classifier.classify(text, known)
        except ClassificationError as exc:
            self._last_check = "failed"
            self._log(
                "warnings: classifier failed — note stays unchecked, retry later",
                "warnings_classifier_failed",
                outcome=outcome_of(exc).value,
                note_id=note_id,
                detail=str(exc)[:160],
            )
            return None
        return self._decide(note_id, row["namespace"], result, known)

    def _decide(self, note_id: int, namespace: str, result: Any, known: list) -> str | None:
        """Решение по §6.5: подсветка «не на месте»/«нет узла» без переноса."""
        hint = result.hint_path
        confidence = result.confidence
        if confidence < self._settings.warnings_min_confidence:
            return None  # низкая уверенность — мелкий шум не подсвечиваем
        if not hint:
            return None  # «общая» разметка не спорит с конкретным размещением
        if namespace == hint:
            # Разметка совпала с узлом — причина ушла: снимаем открытое.
            resolved = self.resolve(KIND_NOTE_MISPLACED, "note", note_id)
            resolved += self.resolve(KIND_NOTE_NO_NODE, "note", note_id)
            self._last_check = "resolved" if resolved else "none"
            return KIND_NOTE_MISPLACED if resolved else None
        existing = {node["path"] for node in known}
        if hint not in existing:
            self.open(
                KIND_NOTE_NO_NODE,
                "note",
                note_id,
                MESSAGE_NO_NODE.format(id=note_id, hint_path=hint),
                {"hint_path": hint, "confidence": confidence},
            )
            self._last_check = "opened" if self._last_open_created else "updated"
            return KIND_NOTE_NO_NODE
        if hint.split("/")[0] != namespace.split("/")[0]:
            self.open(
                KIND_NOTE_MISPLACED,
                "note",
                note_id,
                MESSAGE_MISPLACED.format(
                    id=note_id,
                    namespace=namespace,
                    hint_path=hint,
                    confidence=confidence,
                ),
                {
                    "namespace": namespace,
                    "hint_path": hint,
                    "confidence": confidence,
                },
            )
            self._last_check = "opened" if self._last_open_created else "updated"
            return KIND_NOTE_MISPLACED
        # Общий корень: расхождение «родитель/ребёнок» — предупреждение только
        # на строгом уровне (FR-3.3, дефолт — мелкий шум).
        if self._settings.warnings_level_strict:
            self.open(
                KIND_NOTE_MISPLACED,
                "note",
                note_id,
                MESSAGE_MISPLACED.format(
                    id=note_id,
                    namespace=namespace,
                    hint_path=hint,
                    confidence=confidence,
                ),
                {
                    "namespace": namespace,
                    "hint_path": hint,
                    "confidence": confidence,
                },
            )
            self._last_check = "opened" if self._last_open_created else "updated"
            return KIND_NOTE_MISPLACED
        return None

    def run(self, batch: int | None = None) -> int:
        """Прогон джобы: партия кандидатов в бюджете классификатора; их число.

        Бюджет `JOB_WARNINGS_CLASSIFIER_BUDGET` — вызовов модели за прогон:
        заметки сверх бюджета маркера НЕ получают и попадают в следующий прогон
        (FR-4.2). Разобранная заметка помечается `notes.warnings_at=now`
        (анти-зацикливание). Переноса заметок нет (§13.5).
        """
        limit = self._settings.job_warnings_batch if batch is None else int(batch)
        budget = int(self._settings.job_warnings_classifier_budget)
        with self._conn(self._settings) as conn:
            rows = conn.execute(_CANDIDATE_SELECT, (limit,)).fetchall()
        checked = 0
        opened = 0
        resolved = 0
        used = 0
        for row in rows:
            if used >= budget:
                break
            kind = self.check_note(int(row["id"]))
            used += 1
            if self._last_check == "failed":
                continue  # маркера нет — заметка снова кандидат
            self._mark_checked(int(row["id"]))
            checked += 1
            if self._last_check == "opened":
                opened += 1
            elif self._last_check == "resolved":
                resolved += 1
        self._log(
            "warnings: run finished",
            "warnings_run",
            checked=checked,
            opened=opened,
            resolved=resolved,
            budget_used=used,
        )
        return checked

    def _mark_checked(self, note_id: int) -> None:
        """Поставить маркер разбора: заметка не выедает бюджет повторно."""
        with self._conn(self._settings) as conn:
            conn.execute(
                "UPDATE notes SET warnings_at = "
                "strftime('%Y-%m-%dT%H:%M:%SZ','now') WHERE id = ?",
                (note_id,),
            )

    def queue_stat(self) -> dict:
        """Снимок очереди `warnings` для `/health.queues` — только SQL, без модели."""
        with self._conn(self._settings) as conn:
            row = conn.execute(_CANDIDATE_STAT_SELECT).fetchone()
        return queue_snapshot(row["pending"], row["oldest_pending_sec"])

    def queue_empty(self) -> bool:
        """Пуста ли очередь `warnings` (дешёвая перепроверка очереди, пул 6)."""
        return int(self.queue_stat()["pending"] or 0) == 0

    @staticmethod
    def _log(message: str, event: str, **fields: Any) -> None:
        """Событие джобы в общий журнал: `event` + обязательное поле `job`."""
        logging.getLogger("app").info(
            message, extra={"event": event, "job": WARNINGS_JOB, **fields}
        )


def build_warnings_job(
    worker: "BackgroundWorker", settings: Settings
) -> JobSpec | None:
    """Сборщик джобы `warnings` (FR-1.1): без классификатора — неприменима.

    Форма «по интервалу» (без события и без перепроверки очереди): прогон
    уходит в `to_thread`, очередь — производная от заметок. Слот — `summary`
    (та же модель, что у `nodes`): разрыв слота действует на неё как на прочих.
    """
    if worker._classifier is None:  # тестовый режим Фазы 3 — петли не было
        return None
    service = WarningsService(
        settings, classifier=worker._classifier, namespaces=worker._namespaces
    )
    batch = settings.job_warnings_batch

    async def process() -> int:
        return await asyncio.to_thread(service.run, batch)

    return JobSpec(
        name=WARNINGS_JOB,
        queue=WARNINGS_JOB,
        interval_sec=settings.job_warnings_interval_sec,
        batch=batch,
        enabled=settings.job_warnings_enabled,
        process=process,
        queue_empty=service.queue_empty,
        wait_event=None,
        idle_hook=None,
        queue_stat=service.queue_stat,
        slot="summary",
    )


# Регистрация джобы в реестре каркаса (FR-1.1/FR-1.6): своя джоба — в своём
# модуле (одна строка в `JOB_BUILDERS`); импорт односторонний — warnings → jobs.
jobs.JOB_BUILDERS += (build_warnings_job,)
