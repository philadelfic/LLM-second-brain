"""Разделитель слота нарезки (3.4.1, LSBN-T-27).

Во время раскатки джоба миграции берёт партию `MIGRATION_BATCH` заметок,
отрабатывает и уступает слот генерации на `MIGRATION_PAUSE_SEC` (пауза
ПОСЛЕ прогона, `pause_after_run` в каркасе джоб), затем берёт следующую.
Очередь однопоточная. Числа — СВОИ: партия больше
не берётся из `embedding_batch_size` (чанки /api/embed векторизации),
интервал — не из `pending_retry_sec` (ретрай до-обработки); поведение
векторизации и до-обработки не менялось — их тесты зелёные в сьюте,
здесь фиксируется отвязка.

Фейки (FixedSummarizer), без сети; сцена — спуск 20000 → 1500 как в
test_migration_split.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

import pytest

from app.config import Settings, get_settings
from app.services import jobs
from app.services.migration import MIGRATION_JOB, build_migration_job
from app.services.notes import NoteService
from app.storage.db import init_db, session, transaction
from app.storage.migration import (
    MIGRATION_NOTES_TABLE,
    run_startup_migration,
    startup_plan,
)
from fakes import FixedSummarizer

_LONG = "длинная заметка про окно раскатки нарезки " * 150  # ~5100 знаков


def _seed_legacy(settings: Settings, waiting: int) -> None:
    """Прежняя база (CHECK 20000): 1 короткая + `waiting` длинных 'ожидает'."""
    init_db(settings, note_chars=20000)
    with session(settings) as conn, transaction(conn):
        conn.execute("INSERT INTO notes(id, text) VALUES (1, 'короткая заметка')")
        for old_id in range(2, 2 + waiting):
            conn.execute(
                "INSERT INTO notes(id, text) VALUES (?, ?)", (old_id, _LONG)
            )


def _migrate_down(monkeypatch: pytest.MonkeyPatch, waiting: int) -> None:
    """Полный спуск 20000 → 1500: активная база + `waiting` 'ожидает'."""
    settings = get_settings()
    _seed_legacy(settings, waiting)
    monkeypatch.setenv("MAX_NOTE_CHARS", "1500")
    get_settings.cache_clear()
    run_startup_migration(get_settings(), startup_plan(get_settings()))


class _FakeWorker:
    """Утка под build_migration_job: суммаризатор, notes, без warnings."""

    def __init__(self, settings: Settings) -> None:
        self._summarizer = FixedSummarizer()
        self._notes = NoteService(settings)
        self._warnings = None


def _waiting_ids() -> list[int]:
    with session(get_settings()) as conn:
        return [
            int(r["old_id"])
            for r in conn.execute(
                f"SELECT old_id FROM {MIGRATION_NOTES_TABLE} "
                "WHERE status IN ('ожидает', 'карантин') ORDER BY old_id"
            )
        ]


# --- партия разделителя ------------------------------------------------------


class TestBatch:
    def test_job_takes_exactly_migration_batch(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Прогон джобы берёт ровно MIGRATION_BATCH 'ожидает' (не всех разом):
        6 ждущих, партия 4 → прогоны 4 → 2 → 0 (очередь выгребена)."""
        monkeypatch.setenv("DB_PATH", str(tmp_path / "notes.db"))
        monkeypatch.setenv("EMBEDDING_DIM", "8")
        monkeypatch.setenv("MIGRATION_BATCH", "4")
        get_settings.cache_clear()
        _migrate_down(monkeypatch, waiting=6)
        assert len(_waiting_ids()) == 6
        settings = get_settings()
        spec = build_migration_job(_FakeWorker(settings), settings)
        assert spec is not None and spec.batch == 4

        async def scenario() -> list[int]:
            return [await spec.process(), await spec.process(), await spec.process()]

        runs = asyncio.run(scenario())
        assert runs == [4, 2, 0]  # ровно партия, остаток, пустой (drained)
        assert _waiting_ids() == []

    def test_foreign_batch_settings_do_not_leak(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """embedding_batch_size/pending_retry_sec больше НЕ задают партию и
        интервал нарезки (LSBN-T-27): при чужих числах 32/7 партия 2, пауза 45."""
        monkeypatch.setenv("DB_PATH", str(tmp_path / "notes.db"))
        monkeypatch.setenv("EMBEDDING_DIM", "8")
        monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "32")
        monkeypatch.setenv("PENDING_RETRY_SEC", "7")
        monkeypatch.setenv("MIGRATION_BATCH", "2")
        monkeypatch.setenv("MIGRATION_PAUSE_SEC", "45")
        get_settings.cache_clear()
        _migrate_down(monkeypatch, waiting=5)
        settings = get_settings()
        assert settings.embedding_batch_size == 32  # чужое число на месте
        assert settings.pending_retry_sec == 7
        spec = build_migration_job(_FakeWorker(settings), settings)
        assert spec is not None
        assert spec.batch == 2  # НЕ embedding_batch_size
        assert spec.interval_sec == 45  # НЕ pending_retry_sec

        async def scenario() -> list[int]:
            return [await spec.process(), await spec.process()]

        assert asyncio.run(scenario()) == [2, 2]  # партия — MIGRATION_BATCH


# --- пауза ПОСЛЕ прогона в каркасе джоб --------------------------------------


def _spec(
    processed_results: list[int],
    *,
    pause_after_run: bool,
    queue_empty_results: list[bool] | None = None,
    on_run: Callable[[int], None] | None = None,
) -> jobs.JobSpec:
    """JobSpec с запрограммированными прогонами (и очередью, если задана)."""
    queue = list(processed_results)
    empty = list(queue_empty_results or [])

    async def process() -> int:
        result = queue.pop(0) if queue else 0
        if on_run is not None:
            on_run(result)
        return result

    async def queue_empty() -> bool:
        return empty.pop(0) if empty else True

    return jobs.JobSpec(
        name="migration",
        queue=MIGRATION_JOB,
        interval_sec=42,
        batch=3,
        enabled=True,
        process=process,
        queue_empty=queue_empty,
        wait_event=None,
        idle_hook=None,
        queue_stat=None,
        slot="summary",
        pause_after_run=pause_after_run,
    )


def _run_loop_timeline(
    processed_results: list[int],
    *,
    pause_after_run: bool,
    queue_empty_results: list[bool] | None = None,
    stop_after: int,
) -> list[tuple[str, float | int]]:
    """Петля с фейковым сном; таймлайн событий («run», n) и («sleep», delay)."""
    timeline: list[tuple[str, float | int]] = []

    def on_run(result: int) -> None:
        timeline.append(("run", result))

    spec = _spec(
        processed_results,
        pause_after_run=pause_after_run,
        queue_empty_results=queue_empty_results,
        on_run=on_run,
    )
    real_sleep = asyncio.sleep

    async def fake_sleep(delay: float, *args, **kwargs) -> None:
        timeline.append(("sleep", delay))
        await real_sleep(0)

    async def scenario() -> None:
        state = jobs.BackoffState(spec.interval_sec)
        iterations = {"n": 0}

        def stopping() -> bool:
            iterations["n"] += 1
            return iterations["n"] > stop_after

        jobs.asyncio.sleep = fake_sleep
        try:
            await jobs.run_loop(spec, stopping, state)
        finally:
            jobs.asyncio.sleep = real_sleep

    asyncio.run(scenario())
    return timeline


class TestPauseAfterRun:
    def test_productive_runs_paused_by_interval(self) -> None:
        """После КАЖДОГО продуктивного прогона — пауза interval_sec, пока
        очередь не пуста (LSBN-T-27): слот уступлен перед следующей партией.
        Пауза — не «период от старта»: сон стоит МЕЖДУ прогонами."""
        timeline = _run_loop_timeline(
            [1, 1, 0],
            pause_after_run=True,
            queue_empty_results=[False, False],
            stop_after=3,
        )
        assert timeline == [
            ("run", 1),
            ("sleep", 42.0),  # пауза ПОСЛЕ прогона (разделитель слота)
            ("run", 1),
            ("sleep", 42.0),  # снова пауза — очередь ещё не пуста
            ("run", 0),
            ("sleep", 42.0),  # пустой прогон — обычное ожидание петли
        ]

    def test_drained_queue_skips_pause(self) -> None:
        """Очередь опустела после прогона — паузы нет: петля сразу делает
        холостой прогон (фаза done не ждёт паузу разделителя)."""
        timeline = _run_loop_timeline(
            [1, 0],
            pause_after_run=True,
            queue_empty_results=[True],
            stop_after=2,
        )
        assert timeline == [
            ("run", 1),
            # паузы после продуктивного прогона нет — очередь пуста
            ("run", 0),
            ("sleep", 42.0),  # пустой прогон — обычное ожидание петли
        ]

    def test_default_flag_keeps_immediate_drain(self) -> None:
        """Без флага (прочие джобы: векторизация, до-обработка, связи)
        поведение прежнее — продуктивные прогоны идут подряд без паузы."""
        timeline = _run_loop_timeline(
            [2, 2, 0], pause_after_run=False, stop_after=3
        )
        assert timeline == [
            ("run", 2),
            ("run", 2),  # выгребание без паузы — как до 3.4.1
            ("run", 0),
            ("sleep", 42.0),
        ]
