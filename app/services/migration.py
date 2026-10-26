"""Самомиграция схемы, фаза 4 — джоба нарезки (LSBN-F-22, M2).

Невлезающие заметки (migration_notes 'ожидает', M1) нарезаются моделью
на самодостаточные части по мере обработки — «обработалось, стало
доступно» (постановка F-22, п. 1). Тексты исходников живут в ПРЕЖНЕЙ
базе (union-чтение — M3): джоба читает их по legacy-пути из
`migration.state.prev_file`, проверяет TTL и зовёт `SummaryService.split`
(операция `split` — SPLIT_* env, слот нарезки отдельный, дефолт локальный)
и `NoteService.save_migration_parts` — одна транзакция на заметку:
части + история source="migration_split" + статус 'нарезана'+new_ids.

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
from app.storage.db import session
from app.storage.migration import MIGRATION_NOTES_TABLE, read_state

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
        """
        with session(self._settings) as conn:
            row = conn.execute(
                f"SELECT COUNT(*) AS pending, "
                f"CAST(strftime('%s','now') AS INTEGER) - "
                f"CAST(strftime('%s', MIN(updated_at)) AS INTEGER) AS age "
                f"FROM {MIGRATION_NOTES_TABLE} WHERE status IN (?, ?)",
                _TAKEN_STATUSES,
            ).fetchone()
        age = row["age"]
        return {"pending": int(row["pending"]), "oldest_pending_sec": age}

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
        state = self._read_state()
        if state is None:
            return 0
        legacy_path = self._legacy_path(state)
        if legacy_path is None:
            return 0
        rows = self._taken_rows(limit)
        if not rows:
            return 0
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
        except sqlite3.Error:
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
        except sqlite3.Error:
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
