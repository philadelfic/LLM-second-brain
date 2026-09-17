"""Каркас фоновых джоб (lsb-0014, релиз 3.1.0): реестр, единый цикл, журнал.

Юниты на **синтетической** джобе: копии цикла в тестах нет — джоба описывается
`JobSpec`-ом (регистрация через реестр) и обслуживается `run_loop`. Проверяется
ровно то, что нельзя увидеть на существующих петлях: сброс интервала при
прогрессе, back-off пустых прогонов до потолка 15 мин, супервизор итерации,
выключенная джоба, пробуждение по событию, перепроверка очереди после `clear()`
(пул 6, lost wakeup) и отложенное задание (0 обработанных — петля уходит в сон,
а не крутится вхолостую). Поведение существующих петель этой постановкой не
меняется — их регресс живёт в tests/test_worker*.py.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time

import pytest

import app.services.jobs as jobs
from app.services import worker as worker_module
from app.services.jobs import (
    MAX_INTERVAL_SEC,
    BackoffState,
    JobSpec,
    build_job_specs,
    log_job,
    next_interval,
    run_loop,
)


async def _noop_process() -> int:
    """Прогон-заглушка: ничего не обрабатывает."""
    return 0


def make_spec(
    process,
    *,
    interval_sec: int = 30,
    enabled: bool = True,
    queue: str | None = None,
    batch: int | None = None,
    queue_empty=None,
    wait_event: asyncio.Event | None = None,
    idle_hook=None,
    queue_stat=None,
) -> JobSpec:
    """Синтетическая джоба: прогон задаёт тест, остальное — по умолчанию."""
    return JobSpec(
        name="synthetic",
        queue=queue,
        interval_sec=interval_sec,
        batch=batch,
        enabled=enabled,
        process=process,
        queue_empty=queue_empty,
        wait_event=wait_event,
        idle_hook=idle_hook,
        queue_stat=queue_stat,
    )


async def wait_until(predicate, timeout: float = 1.0) -> None:
    """Дождаться условия живого цикла (опрос с yield'ами)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("условие не наступило за отведённое время")


def patch_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Мгновенный asyncio.sleep с записью запрошенных пауз.

    Back-off у каркаса настоящий (30 с → 15 мин), но тест не ждёт его: пауза
    записывается и заменяется yield'ом. Тесты, где петля должна именно спать,
    этот хелпер не используют.
    """
    real_sleep = asyncio.sleep
    delays: list[float] = []

    async def spy(delay: float, *args: object, **kwargs: object) -> None:
        delays.append(delay)
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", spy)
    return delays


# --- реестр: регистрация описанием -------------------------------------------


@pytest.mark.asyncio
async def test_registered_job_is_served_by_unified_loop(monkeypatch) -> None:
    """Джоба из реестра обслуживается единым циклом; прогресс не даёт спать.

    Копии цикла в тесте нет: реестр собирает `build_job_specs`, обслуживает
    `run_loop`; прогон вернул > 0 — следующая партия идёт сразу, пауз ноль.
    """
    delays = patch_sleep(monkeypatch)
    runs: list[int] = []

    async def process() -> int:
        runs.append(1)
        return 1

    spec = make_spec(process, interval_sec=300)
    # Неприменимая джоба (сборщик вернул None) в реестр не попадает.
    monkeypatch.setattr(
        jobs, "JOB_BUILDERS", (lambda worker, settings: None, lambda w, s: spec)
    )
    specs = build_job_specs(object(), object())
    assert specs == [spec]
    await asyncio.wait_for(
        run_loop(specs[0], lambda: len(runs) >= 3, BackoffState(300)), timeout=1.0
    )
    assert len(runs) >= 3
    assert delays == []  # прогресс: очередь выгребается без сна


@pytest.mark.asyncio
async def test_progress_resets_interval(monkeypatch) -> None:
    """Прогресс сбрасывает выросший back-off в стартовый интервал джобы."""
    patch_sleep(monkeypatch)
    calls = 0

    async def process() -> int:
        nonlocal calls
        calls += 1
        return 1

    state = BackoffState(30)
    state.interval = float(MAX_INTERVAL_SEC)  # back-off успел дорасти до потолка
    spec = make_spec(process, interval_sec=30)
    await asyncio.wait_for(run_loop(spec, lambda: calls >= 2, state), timeout=1.0)
    assert state.interval == 30.0


@pytest.mark.asyncio
async def test_empty_runs_grow_backoff_to_cap(monkeypatch) -> None:
    """Пустой прогон растит back-off: 30 с → ×2 → потолок 15 минут."""
    delays = patch_sleep(monkeypatch)
    calls = 0

    async def process() -> int:
        nonlocal calls
        calls += 1
        return 0

    state = BackoffState(30)
    spec = make_spec(process, interval_sec=30)
    await asyncio.wait_for(run_loop(spec, lambda: calls >= 8, state), timeout=1.0)
    assert delays[:6] == [30.0, 60.0, 120.0, 240.0, 480.0, float(MAX_INTERVAL_SEC)]
    assert delays[6] == float(MAX_INTERVAL_SEC)  # выше потолка не растёт
    assert state.interval == float(MAX_INTERVAL_SEC)
    assert next_interval(MAX_INTERVAL_SEC, 30) == MAX_INTERVAL_SEC


# --- контракт надёжности ------------------------------------------------------


@pytest.mark.asyncio
async def test_iteration_failure_does_not_kill_loop(caplog) -> None:
    """Сбой итерации не убивает цикл: `loop_iteration_failed` с job, работа идёт."""
    calls = 0

    async def process() -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("boom")
        return 1

    spec = make_spec(process, interval_sec=30)
    with caplog.at_level(logging.WARNING, logger="app"):
        await asyncio.wait_for(
            run_loop(spec, lambda: calls >= 2, BackoffState(0)), timeout=1.0
        )
    assert calls >= 2  # следующая итерация выполнена
    failed = [
        record
        for record in caplog.records
        if getattr(record, "event", None) == "loop_iteration_failed"
    ]
    assert len(failed) == 1
    assert failed[0].job == "synthetic"  # обязательное поле журнала джоб
    assert failed[0].exc_info is not None  # супервизор пишет traceback


@pytest.mark.asyncio
async def test_disabled_job_does_not_start(monkeypatch) -> None:
    """Выключенная джоба не запускается, но из реестра не исчезает (очередь видна)."""
    calls = 0

    async def process() -> int:
        nonlocal calls
        calls += 1
        return 0

    spec = make_spec(process, enabled=False, interval_sec=3600)
    monkeypatch.setattr(jobs, "JOB_BUILDERS", (lambda worker, settings: spec,))
    assert build_job_specs(object(), object()) == [spec]
    await asyncio.wait_for(run_loop(spec, lambda: False, BackoffState(3600)), timeout=0.5)
    assert calls == 0


@pytest.mark.asyncio
async def test_wait_event_wakes_loop_immediately() -> None:
    """«По требованию»: событие будит петлю немедленно, не дожидаясь интервала."""
    event = asyncio.Event()
    calls = 0

    async def process() -> int:
        nonlocal calls
        calls += 1
        return 0

    spec = make_spec(
        process, interval_sec=3600, wait_event=event, queue_empty=lambda: True
    )
    task = asyncio.create_task(run_loop(spec, lambda: False, BackoffState(3600)))
    try:
        await wait_until(lambda: calls >= 1)
        await asyncio.sleep(0.05)
        assert calls == 1  # интервал 3600 с: сама петля не проснётся
        event.set()  # появилась работа — будим петлю
        await wait_until(lambda: calls >= 2, timeout=0.5)
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_queue_recheck_after_clear_keeps_work() -> None:
    """Отсутствие события не теряет работу: непустая очередь не даёт петле спать.

    Событие не выставлено ни разу — сигнала нет, но перепроверка очереди после
    `clear()` видит незабранное задание и продолжает прогоны без ожидания
    (lost wakeup закрыт: стирается только пустая очередь).
    """
    calls = 0

    async def process() -> int:
        nonlocal calls
        calls += 1
        return 0

    spec = make_spec(
        process,
        interval_sec=3600,
        wait_event=asyncio.Event(),
        queue_empty=lambda: False,
    )
    task = asyncio.create_task(run_loop(spec, lambda: False, BackoffState(3600)))
    try:
        await wait_until(lambda: calls >= 5, timeout=1.0)
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_deferred_job_waits_instead_of_spinning() -> None:
    """Отложенное задание (0 обработанных, pending остался) уводит петлю в сон.

    Иначе «модель недоступна» превращалось бы в busy-loop без back-off.
    """
    calls = 0
    idle: list[int] = []

    async def process() -> int:
        nonlocal calls
        calls += 1
        return 0

    spec = make_spec(process, interval_sec=3600, idle_hook=lambda: idle.append(1))
    task = asyncio.create_task(run_loop(spec, lambda: False, BackoffState(3600)))
    try:
        await wait_until(lambda: calls >= 1)
        await asyncio.sleep(0.1)  # интервал 3600 с: петля обязана спать
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
    assert calls == 1  # вхолостую не крутится
    assert idle == [1]  # idle-ветка (гигиена) отработала перед сном


# --- журнал -------------------------------------------------------------------


def test_log_job_carries_job_and_common_fields(caplog) -> None:
    """События `log_job` несут обязательное поле job и общие поля формата (FR-1.4)."""
    spec = make_spec(_noop_process)
    with caplog.at_level(logging.INFO, logger="app"):
        log_job(
            spec, "links_sweep", note_id=7, outcome="ok", target="default", count=3
        )
        log_job(spec, "links_idle", reason="empty")
    first, second = caplog.records
    assert first.levelno == logging.INFO
    assert (first.event, first.job) == ("links_sweep", "synthetic")
    assert (first.note_id, first.outcome, first.target, first.count) == (
        7,
        "ok",
        "default",
        3,
    )
    assert (second.event, second.job, second.reason) == (
        "links_idle",
        "synthetic",
        "empty",
    )
    assert not hasattr(second, "outcome")  # неприменимые поля не пишутся


def test_backoff_helpers_shared_with_worker() -> None:
    """Back-off живёт в каркасе: воркер переиспользует те же имена (поведение то же)."""
    assert worker_module.next_interval is next_interval
    assert worker_module.MAX_INTERVAL_SEC == MAX_INTERVAL_SEC
