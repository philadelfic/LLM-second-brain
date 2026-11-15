"""Разрез сводного summary-цикла на три джобы (релиз 3.9.0, LSBN-S-6).

Сводный цикл воркера `title → summarize → merge` жил в одной джобе `summary`
с одним событием и одним breaker'ом слота. В режиме `operation`
(`SLOT_MODEL=operation`) цикл разрезан на три джобы — `title`, `summarize`,
`merge` — у каждой своя очередь, своё событие, свой счётчик и свой
breaker (дизайн B3, стадия 2).

Проверяемые контракты (критерии §4 постановки):
1. в режиме `operation` реестр джоб содержит три джобы с разными событиями
   и разными очередями;
2. в режиме `legacy` реестр НЕ изменился (снимок снят с базы `dev` до
   правок);
3. отказ title не блокирует summarize (пауза — только у своей операции);
4. merge-сигнал доходит до merge-петли (ставится при создании задачи);
5. работа не теряется: заметку берут обе петли, merge-задача не пропадает;
6. в `/health.queues` в режиме `operation` видны три записи;
7. в режиме `operation` воркер не падает на ссылках к breaker'у слота
   `summary` (находка проверки S-5, M1).

Юнит без сети: модели — фейки (`fakes.FixedSummarizer` и его наследник с
отказом title), кодировщик — `HashEmbedder`.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time

import pytest
from fakes import FixedSummarizer, HashEmbedder, clear_seeded_skills

from app.config import get_settings
from app.services.jobs import build_job_specs
from app.services.notes import NoteService
from app.services.summary import SummaryError
from app.services.worker import BackgroundWorker
from app.storage.db import init_db, session, transaction

DIM = 8

#: Снимок реестра джоб в режиме `legacy`, снятый с базы `dev` (LSBN-S-5,
#: коммит 4950d09) ДО этой правки: (имя джобы, её очередь, её слот).
#: Критерий 2: реестр легаси-режима не изменился.
LEGACY_REGISTRY: tuple[tuple[str, str | None, str | None], ...] = (
    ("links", "links", None),
    ("migration", "migration", "summary"),
    ("embedding", "vector", "embedding"),
    ("summary", "summary", "summary"),
    ("judge", "judge", "judge"),
    ("areas", "areas", "embedding"),
    ("expiration", None, None),
    ("nodes", "nodes", "summary"),
)

#: Три джобы разрезанного цикла.
SPLIT_JOBS = ("title", "summarize", "merge")


class TitleFailingSummarizer(FixedSummarizer):
    """Суммаризатор: заголовок всегда отказывает, выжимка работает.

    Отказ заголовка — `SummaryError` без исхода: исход читается как
    `off_contract` (не-транзиентный) — пауза операции открывается на первом
    таком отказе (при `SLOT_PAUSE_AFTER=1`).
    """

    def title(self, text: str):  # noqa: ANN201 — сигнатура фейка
        self.calls.append(text)
        self.last_attempt_ok = False
        raise SummaryError("модель вернула не-заголовок")


def _make_settings(tmp_path, monkeypatch: pytest.MonkeyPatch, *, mode: str):
    """Настройки теста: своя БД, режим слоёв и порог паузы в одно отказ."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "notes.db"))
    monkeypatch.setenv("EMBEDDING_DIM", str(DIM))
    # LSBN-T-37 (3.6.3): правило короткого текста выключено — тест проверяет
    # модельный путь выжимки.
    monkeypatch.setenv("SUMMARY_SHORT_TEXT_CHARS", "0")
    # Порог паузы в один отказ: не-транзиентный отказ открывает паузу
    # операции сразу — иначе паузу открыл бы только третий отказ.
    monkeypatch.setenv("SLOT_PAUSE_AFTER", "1")
    if mode == "operation":
        monkeypatch.setenv("SLOT_MODEL", "operation")
    else:
        monkeypatch.delenv("SLOT_MODEL", raising=False)
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    clear_seeded_skills(settings)
    return settings


@pytest.fixture
def legacy_settings(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Режим по умолчанию (`legacy`): реестр джоб — как до разреза."""
    return _make_settings(tmp_path, monkeypatch, mode="legacy")


@pytest.fixture
def operation_settings(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Режим `operation`: цикл разрезан на три джобы."""
    return _make_settings(tmp_path, monkeypatch, mode="operation")


def make_worker(settings, summarizer=None) -> BackgroundWorker:
    """Воркер с суммаризатором (иначе джобы сводного цикла неприменимы)."""
    return BackgroundWorker(
        settings,
        HashEmbedder(DIM),
        summarizer if summarizer is not None else FixedSummarizer("Фикс-суммари."),
    )


def _registry(settings) -> list[tuple[str, str | None, str | None]]:
    """Реестр джоб: (имя, очередь, слот) — в порядке регистрации."""
    worker = make_worker(settings)
    specs = build_job_specs(worker, settings)
    return [(spec.name, spec.queue, spec.slot) for spec in specs]


def _specs(settings) -> dict:
    """Реестр джоб по имени (для проверок гейта/события/очереди)."""
    worker = make_worker(settings)
    return {spec.name: spec for spec in build_job_specs(worker, settings)}


def _pending_summaries(settings) -> int:
    """Сколько заметок ждёт выжимки (`summary_status='pending'`)."""
    with session(settings) as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM notes WHERE summary_status = 'pending' "
            "AND deleted_at IS NULL"
        ).fetchone()
    return int(row[0])


def _pending_merge_jobs(settings) -> int:
    """Сколько merge-заданий ждёт в очереди."""
    with session(settings) as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM worker_jobs WHERE slot = 'summary' "
            "AND kind = 'merge' AND status = 'pending'"
        ).fetchone()
    return int(row[0])


async def wait_until(predicate, timeout: float = 3.0) -> None:
    """Дождаться условия живой петли (опрос с yield'ами)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("условие не наступило за отведённое время")


# --- критерий 1: в режиме operation — три джобы со своим событием и очередью


def test_operation_registry_has_three_split_jobs(operation_settings) -> None:
    """Реестр `operation`: три джобы разрезанного цикла, сводной нет."""
    names = [name for name, _queue, _slot in _registry(operation_settings)]
    for name in SPLIT_JOBS:
        assert name in names
    assert "summary" not in names  # сводной джобы в этом режиме нет


def test_split_jobs_have_own_event_and_queue(operation_settings) -> None:
    """У каждой джобы — СВОЁ событие и своя очередь (общий Event делить нельзя).

    `jobs.py` делает `spec.wait_event.clear()` перед проверкой очереди —
    общий `asyncio.Event` стирал бы пробуждение соседней петли.
    """
    specs = _specs(operation_settings)
    events = [specs[name].wait_event for name in SPLIT_JOBS]
    queues = [specs[name].queue for name in SPLIT_JOBS]
    slots = [specs[name].slot for name in SPLIT_JOBS]

    assert all(event is not None for event in events)
    assert len({id(event) for event in events}) == 3  # разные объекты
    assert len(set(queues)) == 3
    assert len(set(slots)) == 3


def test_split_jobs_have_own_process_and_gate(operation_settings) -> None:
    """У каждой джобы — свой обработчик и свой гейт (breaker операции)."""
    specs = _specs(operation_settings)
    processes = [specs[name].process for name in SPLIT_JOBS]
    gates = [specs[name].gate for name in SPLIT_JOBS]

    assert len({id(process) for process in processes}) == 3
    assert len({id(gate) for gate in gates}) == 3
    assert specs["title"].slot == "title"
    assert specs["summarize"].slot == "summarize"
    assert specs["merge"].slot == "merge"


# --- критерий 2: в режиме legacy реестр не изменился


def test_legacy_registry_unchanged(legacy_settings) -> None:
    """Легаси-реестр — ровно тот, что снят с базы до правок."""
    assert tuple(_registry(legacy_settings)) == LEGACY_REGISTRY


def test_legacy_registry_has_no_split_jobs(legacy_settings) -> None:
    """В легаси-режиме разрезанных джоб в реестре нет."""
    names = {name for name, _queue, _slot in _registry(legacy_settings)}
    for name in SPLIT_JOBS:
        assert name not in names
    assert "summary" in names


# --- критерий 3: отказ title не блокирует summarize


def test_title_failure_does_not_block_summarize(operation_settings) -> None:
    """Отказ заголовка: пауза — только у title, выжимка обрабатывается.

    Заголовок падает (`off_contract`), выжимка работает: очередь summarize
    разгружается, а пауза (`paused_until`) — только у операции `title`.
    """
    worker = make_worker(operation_settings, TitleFailingSummarizer())
    NoteService(operation_settings, HashEmbedder(DIM)).save(
        "заметка без заголовка: ждёт и заголовок, и выжимку"
    )
    assert _pending_summaries(operation_settings) == 1

    assert worker.process_title_pending() == 0  # заголовок не записан
    assert worker._breakers["title"].snapshot()["paused_until"] is not None
    assert worker._breakers["summarize"].snapshot()["paused_until"] is None

    # Гейт джобы summarize открыт, гейт title — закрыт (пауза операции).
    # Реестр — того же воркера: пауза живёт в его памяти.
    specs = {spec.name: spec for spec in build_job_specs(worker, operation_settings)}
    assert specs["summarize"].gate() is True
    assert specs["title"].gate() is False

    # Выжимка обрабатывается: очередь summarize разгружается.
    assert worker.process_summary_pending() == 1
    assert _pending_summaries(operation_settings) == 0


def test_title_failure_leaves_title_work_pending(operation_settings) -> None:
    """Отказ заголовка не теряет работу: заметка ждёт следующего прогона."""
    worker = make_worker(operation_settings, TitleFailingSummarizer())
    NoteService(operation_settings, HashEmbedder(DIM)).save("заметка без заголовка")
    worker.process_title_pending()
    with session(operation_settings) as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM notes WHERE title IS NULL "
            "AND deleted_at IS NULL"
        ).fetchone()
    assert int(row[0]) == 1


# --- критерий 4: merge-сигнал доходит до merge-петли


def test_merge_job_creation_wakes_merge_loop(operation_settings) -> None:
    """Создание merge-задачи будит merge-петлю (сигнал — при создании)."""
    worker = make_worker(operation_settings)
    assert worker._merge_event.is_set() is False
    worker._create_job("summary", "merge", 1, payload=json.dumps({"older_id": 1}))
    assert worker._merge_event.is_set() is True


@pytest.mark.asyncio
async def test_merge_loop_takes_the_job(operation_settings) -> None:
    """Merge-петля берёт задание по своему сигналу (без общего back-off)."""
    worker = make_worker(operation_settings)
    worker._create_job("summary", "merge", 1, payload=json.dumps({"older_id": 1}))
    assert _pending_merge_jobs(operation_settings) == 1

    task = asyncio.create_task(worker._run_merge())
    try:
        await wait_until(lambda: _pending_merge_jobs(operation_settings) == 0)
    finally:
        worker.stop()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


# --- критерий 5: работа не теряется


def test_note_is_taken_by_both_loops(operation_settings) -> None:
    """Заметка без заголовка и без выжимки берут обе петли (предикаты
    независимы: `title IS NULL` и `summary_status='pending'`)."""
    worker = make_worker(operation_settings)
    NoteService(operation_settings, HashEmbedder(DIM)).save(
        "заметка без заголовка и без выжимки"
    )

    assert worker.process_title_pending() == 1
    assert worker.process_summary_pending() == 1
    with session(operation_settings) as conn:
        row = conn.execute(
            "SELECT title, summary, summary_status FROM notes WHERE id = 1"
        ).fetchone()
    assert row["title"] is not None
    assert row["summary_status"] == "ok"


def test_merge_job_is_not_lost_by_the_split(operation_settings) -> None:
    """Merge-задача не пропадает при разделении: петля merge её берёт."""
    worker = make_worker(operation_settings)
    notes = NoteService(operation_settings, HashEmbedder(DIM))
    notes.save("первая заметка про ежедневный бэкап", title="Бэкап")
    notes.save("вторая заметка про ежедневный бэкап", title="Бэкап вечером")
    worker._create_job(
        "summary", "merge", 2, payload=json.dumps({"older_id": 1, "cosine": 0.9})
    )

    assert worker._merge_queue_empty() is False
    assert worker.process_merge_pending() == 1
    assert worker._merge_queue_empty() is True
    assert _pending_merge_jobs(operation_settings) == 0


# --- критерий 6: /health.queues в режиме operation


def test_health_queues_show_three_split_queues(operation_settings) -> None:
    """В снимке очередей — три записи разрезанного цикла."""
    worker = make_worker(operation_settings)
    queues = worker.queues_health()
    for name in SPLIT_JOBS:
        assert name in queues
        assert queues[name]["pending"] == 0
    assert "summary" not in queues  # сводной очереди в этом режиме нет


def test_health_queues_split_queues_count_their_own_work(operation_settings) -> None:
    """У каждой очереди — своя работа: заголовок и выжимка считаются
    раздельно."""
    worker = make_worker(operation_settings)
    NoteService(operation_settings, HashEmbedder(DIM)).save(
        "заметка без заголовка: ждёт и заголовок, и выжимку"
    )
    queues = worker.queues_health()
    assert queues["title"]["pending"] == 1
    assert queues["summarize"]["pending"] == 1
    assert queues["merge"]["pending"] == 0


def test_legacy_health_queues_unchanged(legacy_settings) -> None:
    """В легаси-режиме очередь сводного цикла — одна, как до правок."""
    worker = make_worker(legacy_settings)
    queues = worker.queues_health()
    assert "summary" in queues
    for name in SPLIT_JOBS:
        assert name not in queues


# --- критерий 7: в режиме operation нет KeyError на breaker'е слота summary


def test_operation_mode_breaker_lookups_do_not_raise(operation_settings) -> None:
    """Ссылки на breaker слота `summary`/`embedding` не роняют воркер.

    Находка проверки S-5 (M1): в режиме `operation` словарь `_breakers`
    keyed по операциям — прямой lookup по имени слота давал `KeyError`.
    """
    worker = make_worker(operation_settings)
    NoteService(operation_settings, HashEmbedder(DIM)).save(
        "заметка без заголовка: очередь не пуста"
    )
    assert worker._summary_queue_empty() is False  # сводный предикат (легаси)
    assert worker._title_queue_empty() is False
    assert worker._summarize_queue_empty() is False
    assert worker._areas_queue_empty() is True


@pytest.mark.asyncio
async def test_operation_mode_worker_serves_queues(operation_settings, caplog) -> None:
    """В режиме `operation` воркер поднимается и разгружает очередь.

    Проверяется и падение: исключение в итерации петли пишется событием
    `loop_iteration_failed` (итерация не убивает петлю) — в журнале его
    быть не должно.
    """
    worker = make_worker(operation_settings)
    NoteService(operation_settings, HashEmbedder(DIM)).save(
        "заметка без заголовка: ждёт и заголовок, и выжимку"
    )

    with caplog.at_level(logging.WARNING, logger="app"):
        task = asyncio.create_task(worker.run())
        try:
            await wait_until(lambda: _pending_summaries(operation_settings) == 0)
        finally:
            worker.stop()
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    failures = [
        record
        for record in caplog.records
        if getattr(record, "event", None) == "loop_iteration_failed"
    ]
    assert failures == []
    with session(operation_settings) as conn:
        row = conn.execute("SELECT title FROM notes WHERE id = 1").fetchone()
    assert row["title"] is not None
