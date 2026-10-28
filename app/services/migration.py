"""Самомиграция схемы, фаза 4 — джоба нарезки (LSBN-F-22, M2) и M3:
union-контекст чтения, прерывание, сверка для /health.

Невлезающие заметки (migration_notes 'ожидает', M1) нарезаются моделью
на самодостаточные части по мере обработки — «обработалось, стало
доступно» (постановка F-22, п. 1). Тексты исходников живут в ПРЕЖНЕЙ
базе (union-чтение — M3): джоба читает их по legacy-пути из
`migration.state.prev_file`, проверяет TTL и зовёт `SummaryService.split`
(операция `split` — SPLIT_* env, слот нарезки отдельный, дефолт локальный)
и `NoteService.save_migration_parts` — одна транзакция на заметку:
части + история source="migration_split" + статус 'нарезана'+new_ids.

M3 (design §4): процессный `MigrationContext` (legacy_path|None, holder
здесь, дефолт None) — search/notes/dedup читают вторым соединением
прежнюю базу, пока phase ∈ {pour, split}; `abort_migration` — стоп джобы
(phase=aborted в активной базе), reverse-pour дельты в прежнюю (части
исключаются), swap назад — сервис остаётся на прежней базе; фаза 7:
обнуление очереди нарезки → phase=done; `migration_health` — счётчики
сверки (old_total/poured — ЗАМЕТКИ, решение пилота) для /health.

Статусы (design F-22 §3/§4):
- TTL-истекла (или исходник исчез из прежней базы) → 'expired' — модель
  не вызывается;
- части сохранены → 'нарезана' (ставит save_migration_parts);
- `MergeIncompleteError` (модель теряет содержание; повтор даст те же два
  раунда) → 'карантин' + запись в warnings, повтор по back-off;
- транзиент (`SummaryError`) — статус тот же, партия снимается: модель
  недоступна и для остальных, повтор по back-off петли (NFR-3).

Джоба регистрируется на каркасе lsb-0014 (сборщик + строка в
`jobs.JOB_BUILDERS`, прецедент warnings/terms_normalize): форма «по
интервалу», слот summary — разрыв слота нарезки виден в /health как у
прочих. Нет суммаризатора (DI-сборки тестов) — джоба неприменима.
Миграция не идёт (нет 'ожидает'/'карантин') — прогон пуст и дешев
(только SQL), очередь видна в /health.queues.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.config import Settings
from app.services import jobs
from app.services.summary import (
    MERGE_MAX_ROUNDS,
    MergeIncompleteError,
    SummaryError,
)
from app.storage.db import StorageError, note_chars_limit, session, transaction
from app.storage.migration import (
    MIGRATION_NOTES_TABLE,
    SOURCE_MIGRATION_SPLIT,
    MigrationError,
    read_state,
    reverse_pour,
    set_state_phase,
    sibling_db_path,
    swap,
)

#: Фазы, при которых union-чтение включено (design §4: фазы 3–7; после
#: обнуления — done, после прерывания — aborted — обе выключают union).
_UNION_PHASES = ("pour", "split")

if TYPE_CHECKING:  # только для аннотаций — модуль остаётся чистым
    from app.services.notes import NoteService
    from app.services.summary import MergeResult, Summarizer
    from app.services.warnings import WarningsService
    from app.services.worker import BackgroundWorker

MIGRATION_JOB = "migration"

#: Карантин нарезки (design F-22 §4): открытая запись kind=`split_deferred`
#: на исходнике (upsert — повторное откладывание двигает updated_at, не
#: плодит строки; прецедент merge_deferred F-23). Видна оператору в
#: warnings; модель нарезки недоступна — сервис работает (постановка п. 4).
SPLIT_DEFERRED_WARNING = "split_deferred"

#: Статусы, которые джоба берёт в работу: 'ожидает' — первично, 'карантин'
#: — повтор по back-off (design §4: «карантин … повтор по back-off»).
_TAKEN_STATUSES = ("ожидает", "карантин")

logger = logging.getLogger("app")


# --- union-контекст чтения (M3, design §4) -----------------------------------


@dataclass(frozen=True)
class MigrationContext:
    """Процессный union-контекст: где лежит прежняя база и фаза миграции.

    Holder — переменная `_CONTEXT` ниже, дефолт None: без миграции
    потребители (search/notes/dedup) работают ровно по прежним путям —
    нулевой оверхед (ни второго соединения, ни лишнего SQL). Контекст
    выставляет `refresh_migration_context` (lifespan после фаз 0–3, джоба
    при phase=done, abort после swap) — потребители только читают.
    """

    legacy_path: str  # абсолютный путь прежней базы (migration.state.prev_file)
    phase: str  # pour|split — читающие поверхности по ней не различаются
    # Резерв диапазона id (design §8.1): нижняя граница НОВЫХ id
    # (= MAX(id) прежней базы + 1, migration.state.id_floor). NoteService._insert
    # выдаёт новым заметкам/частям id не ниже — без коллизий с old_id
    # ждущих исходников (затенение в union, потери при abort). 0 — состояние
    # без ключа (миграция, начатая до появления id_floor): max+1 как раньше.
    id_floor: int = 0


_CONTEXT: MigrationContext | None = None


def migration_context() -> MigrationContext | None:
    """Текущий union-контекст процесса; None — миграции нет/выключена."""
    return _CONTEXT


def refresh_migration_context(settings: Settings) -> None:
    """Синхронизировать контекст с состоянием миграции активной базы.

    Включает union (phase ∈ {pour, split} и prev_file существует) или
    выключает (нет состояния, done/aborted, файл исчез). Вызывается из
    lifespan после фаз 0–3 (в т.ч. при возобновлении — рестарт середины
    миграции поднимает union без нового перелива), джобой при phase=done
    и abort'ом после swap. Идемпотентно; сбой чтения — union выключен.
    Файла базы нет — union выключен БЕЗ открытия соединения: sqlite3.connect
    создаёт пустой файл, а это побочный эффект, которого у чтения быть не
    должно (свежая база до init_db).
    """
    global _CONTEXT
    _CONTEXT = None
    if not Path(settings.db_path).is_file():
        return
    try:
        with session(settings) as conn:
            state = read_state(conn)
    except (sqlite3.Error, StorageError):
        return
    if not state or str(state.get("phase")) not in _UNION_PHASES:
        return
    prev_file = str(state.get("prev_file") or "")
    if not prev_file:
        return
    candidate = Path(settings.db_path).parent / prev_file
    if not candidate.is_file():
        return
    _CONTEXT = MigrationContext(
        legacy_path=str(candidate),
        phase=str(state["phase"]),
        id_floor=int(state.get("id_floor") or 0),
    )


def legacy_visible_ids(settings: Settings) -> set[int]:
    """id заметок, чья единственная живая копия — прежняя база.

    Статусы 'ожидает'/'карантин' (текст ещё в прежней базе); 'нарезана' и
    'expired' НЕ входят — исходник заменён частями либо закрыт, union его
    не показывает (design §4: «исключая id 'нарезана'/'expired'»).
    Затенённые id исключаются: активная база выдаёт новым заметкам
    max(id)+1, что может совпасть с id ждущего исходника — такая заметка
    в активной базе СВОЯ, а не исходник (иначе union склеил бы разные
    заметки по одному id). Таблицы migration_notes нет (миграции не было)
    — множество пусто.
    """
    try:
        with session(settings) as conn:
            rows = conn.execute(
                f"SELECT old_id FROM {MIGRATION_NOTES_TABLE} "
                "WHERE status IN ('ожидает', 'карантин') "
                "AND old_id NOT IN (SELECT id FROM notes)"
            ).fetchall()
    except (sqlite3.Error, StorageError):
        return set()
    return {int(row[0]) for row in rows}


def _now() -> str:
    """Текущее время в формате DDL ('%Y-%m-%dT%H:%M:%SZ') — сравнение с TTL.

    Формат совпадает с notes.expires_at (lsb-0004-02) — лексикографическое
    сравнение строк корректно для одного формата.
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class MigrationService:
    """Фаза 4 самомиграции: нарезка 'ожидает' частями ≤ целевого лимита."""

    def __init__(
        self,
        settings: Settings,
        *,
        summarizer: "Summarizer",
        notes: "NoteService",
        warnings: "WarningsService | None" = None,
    ) -> None:
        self._settings = settings
        self._summarizer = summarizer
        self._notes = notes
        # Поверхность предупреждений (карантин виден оператору, постановка
        # п. 4); None — DI-сборки тестов: остаётся только событие журнала
        # (прецедент _defer_merge_pair).
        self._warnings = warnings

    # --- очередь (для /health.queues и петли) --------------------------------

    def queue_stat(self) -> dict[str, int | None]:
        """Снимок очереди нарезки (только SQL): pending + возраст старейшей.

        Возраст — по `updated_at` взятого статуса (штамп двигают и нарезка,
        и карантин) — та же семантика «возраст по pending+waiting» (3.3.1).
        Таблицы migration_notes нет (миграции не было) — очередь пуста.
        """
        try:
            with session(self._settings) as conn:
                row = conn.execute(
                    f"SELECT COUNT(*) AS pending, "
                    f"CAST(strftime('%s','now') AS INTEGER) - "
                    f"CAST(strftime('%s', MIN(updated_at)) AS INTEGER) AS age "
                    f"FROM {MIGRATION_NOTES_TABLE} WHERE status IN (?, ?)",
                    _TAKEN_STATUSES,
                ).fetchone()
        except (sqlite3.Error, StorageError):
            return {"pending": 0, "oldest_pending_sec": None}
        return {"pending": int(row["pending"]), "oldest_pending_sec": row["age"]}

    def queue_empty(self) -> bool:
        """Пуста ли очередь нарезки (дешёвая перепроверка очереди, пул 6)."""
        return int(self.queue_stat()["pending"] or 0) == 0

    # --- фаза 4: партия нарезки ----------------------------------------------

    def process_batch(self, limit: int | None = None) -> int:
        """Нарезать партию 'ожидает' (минимальный id вперёд); число обработанных.

        Обработанное = изменённый статус ('нарезана'/'expired'/'карантин');
        транзиентный отказ статуса не меняет и партию снимает (возврат уже
        сделанного; пустой прогон растит back-off петли — модель недоступна,
        постановка п. 4: сервис работает, нарезка ждёт).
        """
        # 1) Активна ли миграция и где прежняя база (design §3: state).
        # phase=done/aborted — джоба стоит (M3): после прерывания и после
        # обнуления нарезать нечего и нельзя (сервис уже на прежней базе).
        state = self._read_state()
        if state is None or str(state.get("phase")) not in _UNION_PHASES:
            return 0
        legacy_path = self._legacy_path(state)
        if legacy_path is None:
            return 0
        rows = self._taken_rows(limit)
        if not rows:
            # Фаза 7 (M3): очередь нарезки пуста — сверка обнулена,
            # phase=done (union выключается; state остаётся — migration_notes
            # карта отката). Идемпотентно: следующий прогон — холостой.
            self._close_if_drained(state)
            return 0
        # DEF-7a: нарезка началась — фаза видна в state и /health (design §3:
        # pour — только перелив старта, split — нарезка; done/aborted — M3).
        self._mark_split_phase(state)
        done = 0
        legacy_settings = self._settings.model_copy(update={"db_path": legacy_path})
        with session(self._settings) as active, session(legacy_settings) as legacy:
            for row in rows:
                old_id = int(row["old_id"])
                note = legacy.execute(
                    "SELECT id, text, namespace, expires_at FROM notes "
                    "WHERE id = ?",
                    (old_id,),
                ).fetchone()
                if note is None or note["text"] is None:
                    # Исходника нет (оператор удалил в прежней базе): нарезать
                    # нечего — 'expired' закрывает заметку без модели.
                    self._mark(
                        active, old_id, "expired", reason="исходника нет в прежней базе"
                    )
                    done += 1
                    continue
                if note["expires_at"] is not None and str(
                    note["expires_at"]
                ) <= _now():
                    # TTL-истекла (design §4): 'expired' — физическое удаление
                    # делает штатная джоба зачистки в прежней базе (FR-6).
                    self._mark(active, old_id, "expired", reason="истёк TTL")
                    done += 1
                    continue
                outcome = self._split_one(
                    active,
                    old_id,
                    str(note["namespace"] or "default"),
                    str(note["text"]),
                    attempts=int(row["attempts"]),
                )
                if outcome == "retry-later":
                    break  # транзиент: модель лежит — партию снимаем
                done += 1
        return done

    # --- внутреннее ----------------------------------------------------------

    def _close_if_drained(self, state: dict) -> None:
        """Фаза 7: очередь пуста → phase=done и union выключен (M3).

        Вызывается на пустом прогоне при активной фазе: 'ожидает' и
        'карантин' кончились — каждая заметка либо перелита 1:1, либо
        нарезана, либо expired. State не снимается (design §3/§7):
        migration_notes остаётся картой отката, счётчики — в /health.
        """
        try:
            with session(self._settings) as conn, transaction(conn):
                set_state_phase(conn, "done")
        except (sqlite3.Error, StorageError):
            return  # сверка переживает рестарт — done поставит следующий прогон
        refresh_migration_context(self._settings)
        logger.info(
            "migration: split queue drained — counters reconciled, phase=done",
            extra={
                "event": "migration_done",
                "job": MIGRATION_JOB,
                "target": state.get("target"),
                "counters": state.get("counters"),
            },
        )

    def _mark_split_phase(self, state: dict) -> None:
        """Старт фазы нарезки: phase pour → split в state (DEF-7a, design §3).

        Пишется один раз — дальнейшие прогоны фазу не переписывают; done
        (обнуление очереди) и aborted (прерывание, M3) — как было. Сбой
        записи не ломает партию: фазу поставит следующий прогон (прецедент
        _close_if_drained).
        """
        if str(state.get("phase")) == "split":
            return
        try:
            with session(self._settings) as conn, transaction(conn):
                set_state_phase(conn, "split")
        except (sqlite3.Error, StorageError):
            return
        refresh_migration_context(self._settings)

    def _split_one(
        self,
        active: sqlite3.Connection,
        old_id: int,
        namespace: str,
        text: str,
        *,
        attempts: int,
    ) -> str:
        """Нарезать одну заметку: split → save_migration_parts → статус.

        Возврат: "done" (статус изменён), "quarantined", "retry-later"
        (транзиент — статус тот же, партию снимать).
        """
        try:
            result: MergeResult = self._summarizer.split(text)
        except MergeIncompleteError as exc:
            # F-22 design §4: за MERGE_MAX_ROUNDS нарезка не сошлась —
            # 'карантин' + warnings; повтор по back-off (минимальный id
            # карантин берёт вместе с 'ожидает').
            self._mark(
                active,
                old_id,
                "карантин",
                reason=str(exc)[:500],
                bump_attempts=True,
            )
            self._log(
                "migration: split incomplete — note quarantined",
                "migration_split_deferred",
                old_id=old_id,
                rounds=int(getattr(exc, "rounds", 0)) or None,
                detail=str(exc)[:300],
                meta=getattr(exc, "meta", None),
            )
            self._open_warning(old_id, str(exc))
            return "quarantined"
        except SummaryError as exc:
            # Транзиент (модель недоступна / ответ вне контракта): статус
            # тот же, попытка посчитана, партия снимается — back-off петли.
            self._bump_attempt(active, old_id)
            self._log(
                "migration: split failed — note stays pending, retry by back-off",
                "migration_split_failed",
                old_id=old_id,
                detail=str(exc)[:300],
                meta=getattr(exc, "meta", None),
            )
            return "retry-later"
        parts = [(part.title, part.text) for part in result.parts]
        # Наблюдаемость (DEF-6): slot_call на КАЖДЫЙ вызов нарезки — как у
        # merge/dedup (worker), дозапрос 2-го раунда виден отдельной строкой.
        self._log_slot_call(old_id, result)
        outcome = self._notes.save_migration_parts(old_id, namespace, parts)
        self._log(
            "migration: note split into self-contained parts",
            "migration_split",
            old_id=old_id,
            rounds=result.rounds,
            model=result.meta.model,
            parts=len(outcome["parts"]),
            part_ids=[int(part["id"]) for part in outcome["parts"]],
        )
        return "done"

    def _read_state(self) -> dict | None:
        """Состояние миграции активной базы; None — нарезать нечего."""
        try:
            with session(self._settings) as conn:
                return read_state(conn)
        except (sqlite3.Error, StorageError):
            return None

    def _legacy_path(self, state: dict) -> str | None:
        """Путь прежней базы из state (prev_file); нет — warning и холостой ход.

        После снятия состояния (фаза 7, M3) или у немигрировавшей базы
        prev_file отсутствует — это штатно (джоба просто не нужна).
        """
        prev_file = state.get("prev_file")
        if not prev_file:
            return None
        candidate = Path(self._settings.db_path).parent / str(prev_file)
        if not candidate.is_file():
            logger.warning(
                "migration: legacy file from state is missing — split paused",
                extra={
                    "event": "migration_legacy_missing",
                    "job": MIGRATION_JOB,
                    "file": str(prev_file),
                },
            )
            return None
        return str(candidate)

    def _taken_rows(self, limit: int | None) -> list[Any]:
        """Минимальные id взятых статусов ('ожидает' + карантин по back-off)."""
        query = (
            f"SELECT old_id, status, attempts FROM {MIGRATION_NOTES_TABLE} "
            f"WHERE status IN (?, ?) ORDER BY old_id"
        )
        if limit is not None:
            query += f" LIMIT {int(limit)}"
        try:
            with session(self._settings) as conn:
                return conn.execute(query, _TAKEN_STATUSES).fetchall()
        except (sqlite3.Error, StorageError):
            return []  # таблицы migration_notes нет — миграции не было

    @staticmethod
    def _mark(
        conn: sqlite3.Connection,
        old_id: int,
        status: str,
        *,
        reason: str | None = None,
        bump_attempts: bool = False,
    ) -> None:
        """Сменить статус записи миграции (внутри сессии вызывающего)."""
        conn.execute(
            f"UPDATE {MIGRATION_NOTES_TABLE} "
            f"SET status = ?, reason = ?, attempts = attempts + ?, "
            f"updated_at = strftime('%Y-%m-%dT%H:%M:%SZ','now') "
            f"WHERE old_id = ?",
            (status, reason, 1 if bump_attempts else 0, old_id),
        )

    @staticmethod
    def _bump_attempt(conn: sqlite3.Connection, old_id: int) -> None:
        """Посчитать транзиентную попытку (статус не меняется — design §4)."""
        conn.execute(
            f"UPDATE {MIGRATION_NOTES_TABLE} SET attempts = attempts + 1, "
            f"updated_at = strftime('%Y-%m-%dT%H:%M:%SZ','now') "
            f"WHERE old_id = ?",
            (old_id,),
        )

    def _open_warning(self, old_id: int, reason: str) -> None:
        """Карантин виден оператору (постановка п. 4); None — только журнал."""
        if self._warnings is None:
            return
        try:
            self._warnings.open(
                SPLIT_DEFERRED_WARNING,
                "note",
                old_id,
                "migration split incomplete after "
                f"{MERGE_MAX_ROUNDS} rounds — note quarantined, kept in "
                "the legacy base",
                {"reason": reason[:500]},
            )
        except Exception:  # noqa: BLE001 — предупреждение не ломает партию
            logger.warning(
                "migration: failed to record split_deferred warning",
                extra={
                    "event": "warning_open_failed",
                    "job": MIGRATION_JOB,
                    "old_id": old_id,
                },
            )

    def _log(self, message: str, event: str, **fields: Any) -> None:
        """Событие джобы в общий журнал: обязательное поле `job` (FR-1.4).

        Мета вызова (№49/№50) идёт плоскими полями `meta.model`/`meta.provider`
        — как у slot_call воркера; rounds/parts/part_ids — наблюдаемость
        нарезки (постановка: «какой моделью нарезано» — в журнале и отчёте).
        """
        meta = fields.pop("meta", None)
        payload: dict[str, Any] = {"event": event, "job": MIGRATION_JOB}
        for key, value in fields.items():
            if value is not None:
                payload[key] = value
        if meta is not None:
            for key in ("model", "provider", "source"):
                value = getattr(meta, key, None)
                if value is not None:
                    payload[key] = value
        logger.info(message, extra=payload)

    def _log_slot_call(self, old_id: int, result: MergeResult) -> None:
        """slot_call на каждый вызов нарезки (DEF-6; прецедент merge — worker).

        Те же поля успешного вызова, что у воркера (`_call_log_fields`):
        операция/модель/бюджет/длительность — «сколько времени ушло» видно
        по строке на вызов; без метаданных (тестовый двойник) — молча.
        Импорт по месту: воркер тяжёлый и тянет полсистемы — миграция не
        должна импортировать его наверх (прецедент lazy-импортов notes).
        """
        from app.services.worker import SLOT_SUMMARY, _call_log_fields

        for meta in result.metas or [result.meta]:
            if meta is None:
                continue
            payload: dict[str, Any] = {
                "event": "slot_call",
                "slot": SLOT_SUMMARY,
                "job": MIGRATION_JOB,
                "note_id": old_id,
                **_call_log_fields(meta),
            }
            logger.info("slot_call", extra=payload)


# --- прерывание (M3, design §4 «Прерывание») ---------------------------------


#: Мягкие отказы POST /migration/abort: прерывать нечего / уже прервано.
ABORT_HINT_NONE = (
    "no migration to abort: the base is not mid-migration "
    "(phase is done/aborted or there is no migration state)"
)
ABORT_HINT_ALREADY = (
    "migration is already aborted — the service is on the previous base"
)
ABORT_HINT_NOTHING_TO_SPLIT = (
    "nothing to abort: the split queue is empty (up-migrations complete "
    "at startup, nothing waits for the model)"
)


def abort_migration(settings: Settings) -> dict:
    """Прервать миграцию без потерь: сервис остаётся на прежней базе.

    Порядок (design §4): 1) «стоп джобы» — phase=aborted в активной базе,
    следующая партия нарезки берёт ноль заметок (идущая партия успевает
    только дописать части в уходящую базу — они исключаются из перелива,
    дизайн риска 5: короткие сессии + снапшот фазы 1); 2) reverse-pour
    дельты активной базы в прежнюю — апсерт UPDATE-else-INSERT (REPLACE
    ломает FTS внешнего контента), части (record_versions
    source='migration_split') и их зависимости исключаются — «id вне
    отображения частей»; 3) swap: активная (новая) база уходит под имя
    своего CHECK, прежняя возвращается на db_path — НЕ удаляется;
    4) phase=aborted в восстановленной базе (ставит reverse_pour) и
    union-контекст выключен. Повторный вызов после сбоя между (1) и (3)
    возобновляет abort: reverse_pour идемпотентен (апсерт).

    Возврат: {"aborted": True, ...} — прервано этим вызовом; мягкий отказ
    {"aborted": False, "hint": ...} — прерывать нечего/уже прервано.
    Прерывание применимо к миграции «вниз» (есть очередь нарезки):
    возврат «вверх» завершается при старте синхронно — прерывать нечего.
    """
    with session(settings) as conn:
        state = read_state(conn)
        live = note_chars_limit(conn)
    if state is None or str(state.get("phase")) not in (*_UNION_PHASES, "aborted"):
        return {"aborted": False, "hint": ABORT_HINT_NONE}
    prev_name = str(state.get("prev_file") or "")
    if not prev_name or prev_name == Path(settings.db_path).name:
        # Уже на прежней базе (swap после abort состоялся): prev_file
        # указывает на уходящую базу, т.е. на сам db_path.
        return {"aborted": False, "hint": ABORT_HINT_ALREADY}
    legacy_path = str(Path(settings.db_path).parent / prev_name)
    if not Path(legacy_path).is_file():
        raise MigrationError(f"файл прежней базы не найден: {prev_name}")
    if str(state.get("phase")) in _UNION_PHASES:
        pending = _pending_count(settings)
        if pending == 0:
            # Возврат «вверх»/обнулённая очередь: нарезка не нужна, обратно
            # переливать нечего (всё и так в обеих базах) — не трогаем.
            return {"aborted": False, "hint": ABORT_HINT_NOTHING_TO_SPLIT}
        # 1) Стоп джобы: phase=aborted ДО перелива (джоба видит на следующей
        # партии; юнион-чтение ещё работает — прежняя база целa до swap).
        with session(settings) as conn, transaction(conn):
            set_state_phase(conn, "aborted")
    # 2) Дельта активной базы → прежняя (части исключаются); target = CHECK
    # приёмника (prev_limit): штамп schema.note_chars остаётся её собственным.
    # exclude — все old_id миграции: новый заметке активная база может
    # выдать id исходника (max+1), апсерт затер бы оригинал прежней базы;
    # такая заметка остаётся только в уходящем файле (он не удаляется).
    counters = reverse_pour(
        settings,
        source_path=legacy_path,
        away_file=str(settings.db_path),
        target=int(state.get("prev_limit") or live or 0),
        reason="миграция прервана оператором",
        phase="aborted",
        exclude_note_ids=_migration_note_ids(settings),
    )
    # 3) Swap назад: новая база уходит под имя своего CHECK (не удаляется),
    # прежняя возвращается на db_path — следующие session() открывают её.
    swap(
        settings,
        away_file=sibling_db_path(settings.db_path, int(live or 0)),
        incoming_file=legacy_path,
    )
    # 4) Состояние восстановленной базы — phase=aborted (reverse_pour);
    # union-чтение выключается.
    refresh_migration_context(settings)
    logger.info(
        "migration aborted — service stays on the previous base",
        extra={
            "event": "migration_aborted",
            "job": MIGRATION_JOB,
            "prev_file": prev_name,
            **counters,
        },
    )
    return {
        "aborted": True,
        "phase": "aborted",
        "prev_file": prev_name,
        "counters": counters,
    }


def _pending_count(settings: Settings) -> int:
    """Число заметок в очереди нарезки ('ожидает' + 'карантин')."""
    try:
        with session(settings) as conn:
            row = conn.execute(
                f"SELECT COUNT(*) FROM {MIGRATION_NOTES_TABLE} "
                "WHERE status IN ('ожидает', 'карантин')"
            ).fetchone()
    except (sqlite3.Error, StorageError):
        return 0
    return int(row[0])


def _migration_note_ids(settings: Settings) -> set[int]:
    """Все old_id миграции (любой статус) — запрет апсерта в прежнюю базу.

    Для reverse-pour прерывания: id исходника в прежней базе занят его
    собственной строкой, а активная база могла выдать его новой заметке
    (sqlite: max(id)+1) — апсерт по такому id затёр бы оригинал.
    """
    try:
        with session(settings) as conn:
            rows = conn.execute(
                f"SELECT old_id FROM {MIGRATION_NOTES_TABLE}"
            ).fetchall()
    except (sqlite3.Error, StorageError):
        return set()
    return {int(row[0]) for row in rows}


# --- фаза 7: сверка для /health (M3, design §4/§7) ---------------------------


def migration_health(settings: Settings) -> dict:
    """Блок `migration` для /health: фаза, причина, прогресс, инвариант.

    Счётчики — ЗАМЕТКИ (решение пилота): old_total — заметки прежней базы
    (живой COUNT — сверковая база), poured — перелитые 1:1 при старте
    (зафиксирован в state.counters переливом; после abort'а — всё
    вернувшееся целым), split_notes/parts/quarantine/expired/waiting —
    живые статусы migration_notes и история частей. Инвариант сверки:
    old_total = poured + split_notes + waiting + quarantine + expired
    (расхождение = прежнюю базу правили в обход миграции — удаление
    исходника оператором); после abort'а parts=0 и очередь пуста.
    Без состояния миграции — {"phase": None}.
    """
    try:
        with session(settings) as conn:
            state = read_state(conn)
            statuses = _status_counts(conn)
            parts = _part_count(conn)
    except (sqlite3.Error, StorageError):
        return {"phase": None}
    if state is None:
        return {"phase": None}
    phase = str(state.get("phase"))
    waiting = statuses.get("ожидает", 0)
    split_notes = statuses.get("нарезана", 0)
    quarantine = statuses.get("карантин", 0)
    expired = statuses.get("expired", 0)
    old_total = _legacy_total(settings, state, phase)
    if old_total is None:
        poured = None
        invariant_ok = None
    elif phase == "aborted":
        # После прерывания всё вернулось целым: прежняя база снова полная,
        # state.counters от reverse_pour (строки перелива) — не про заметки.
        poured = old_total
        invariant_ok = old_total == poured + split_notes + waiting + quarantine + expired
    else:
        poured = int(state.get("counters", {}).get("poured", 0))
        invariant_ok = (
            old_total == poured + split_notes + waiting + quarantine + expired
        )
    return {
        "phase": phase,
        "target": state.get("target"),
        "reason": state.get("reason"),
        "counters": {
            "old_total": old_total,
            "poured": poured,
            "split_notes": split_notes,
            "parts": parts,
            "quarantine": quarantine,
            "expired": expired,
            "waiting": waiting,
        },
        "invariant_ok": invariant_ok,
    }


def _status_counts(conn: sqlite3.Connection) -> dict[str, int]:
    """Статусы migration_notes → {статус: число}; таблицы нет — пусто."""
    try:
        rows = conn.execute(
            f"SELECT status, COUNT(*) AS n FROM {MIGRATION_NOTES_TABLE} "
            "GROUP BY status"
        ).fetchall()
    except sqlite3.Error:
        return {}
    return {str(row["status"]): int(row["n"]) for row in rows}


def _part_count(conn: sqlite3.Connection) -> int:
    """Число частей нарезки в активной базе (источник record_versions)."""
    try:
        row = conn.execute(
            "SELECT COUNT(DISTINCT entity_id) FROM record_versions "
            "WHERE entity = 'note' AND source = ?",
            (SOURCE_MIGRATION_SPLIT,),
        ).fetchone()
    except sqlite3.Error:
        return 0
    return int(row[0])


def _legacy_total(settings: Settings, state: dict, phase: str) -> int | None:
    """Живое число заметок базы-источника сверки; нет файла — None.

    Во время миграции (pour/split/done) прежняя база — рядом под именем из
    state.prev_file; после abort'а она и есть активная (prev_file указывает
    на уходящую базу) — считаем db_path.
    """
    prev_file = str(state.get("prev_file") or "")
    if phase == "aborted":
        target_path = settings.db_path
    elif prev_file and prev_file != Path(settings.db_path).name:
        candidate = Path(settings.db_path).parent / prev_file
        if not candidate.is_file():
            return None
        target_path = str(candidate)
    else:
        return None
    try:
        with session(
            settings.model_copy(update={"db_path": target_path})
        ) as conn:
            return int(conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0])
    except (sqlite3.Error, StorageError):
        return None


def build_migration_job(
    worker: "BackgroundWorker", settings: Settings
) -> jobs.JobSpec | None:
    """Сборщик джобы `migration`: без суммаризатора — неприменима.

    Форма «по интервалу» (без события: 'ожидает' появляются только при
    самом старте миграции — M1); прогон уходит в `to_thread`. Слот —
    `summary` (нарезка наследует его слот при незаданных SPLIT_*; разрыв
    слота виден в /health и гейтит джобу как у прочих).
    """
    if worker._summarizer is None:  # тестовый режим без суммаризатора
        return None
    service = MigrationService(
        settings,
        summarizer=worker._summarizer,
        notes=worker._notes,
        warnings=worker._warnings,
    )

    async def process() -> int:
        return await asyncio.to_thread(service.process_batch)

    return jobs.JobSpec(
        name=MIGRATION_JOB,
        queue=MIGRATION_JOB,
        interval_sec=settings.pending_retry_sec,
        batch=settings.embedding_batch_size,
        enabled=True,
        process=process,
        queue_empty=service.queue_empty,
        wait_event=None,
        idle_hook=None,
        queue_stat=service.queue_stat,
        slot="summary",
    )


# Регистрация джобы в реестре каркаса (FR-1.1/FR-1.6): своя джоба — в своём
# модуле (одна строка в `JOB_BUILDERS`); импорт односторонний — migration → jobs.
jobs.JOB_BUILDERS += (build_migration_job,)
