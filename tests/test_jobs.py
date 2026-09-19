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
from fakes import FixedSummarizer, HashEmbedder

import app.services.jobs as jobs
from app.config import get_settings
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
from app.services.worker import (
    EXPIRATION_CLEANUP_INTERVAL_SEC,
    BackgroundWorker,
)
from app.storage.db import init_db


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


@pytest.fixture
def settings(test_env):
    """Настройки на тестовой БД (окружение выставляет autouse-фикстура)."""
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    return settings


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


# --- перевод петель воркера на каркас (lsb-0014-02) ---------------------------


def test_registry_contains_worker_jobs_and_links(settings) -> None:
    """Реестр собирает джобы воркера: имена, очереди, интервалы и формы.

    Пять петель воркера + джоба `links` из своего модуля (`links.py`
    регистрирует себя сама, постановка lsb-0010-03) и джоба `nodes`
    (lsb-0011-01, обход `default`) — порядок в реестре задаётся импортом
    (`app.services` → links, затем worker).
    """
    worker = BackgroundWorker(settings, HashEmbedder(8), FixedSummarizer("С."))
    specs = {spec.name: spec for spec in build_job_specs(worker, settings)}
    assert list(specs) == [
        "links",
        "embedding",
        "summary",
        "judge",
        "areas",
        "expiration",
        "nodes",
    ]
    retry = settings.pending_retry_sec
    assert specs["embedding"].queue == "vector"
    assert specs["summary"].queue == "summary"
    assert specs["judge"].queue == "judge"
    assert specs["areas"].queue == "areas"
    assert specs["expiration"].queue is None  # очередь не наблюдаемая
    # Джоба связей (lsb-0010-03): своя очередь и расписание из env.
    assert specs["links"].queue == "links"
    assert specs["links"].interval_sec == settings.job_links_interval_sec
    assert specs["links"].batch == settings.job_links_batch
    assert specs["links"].idle_hook is not None  # гигиена purge_orphans
    assert specs["links"].queue_stat is not None
    # Форма «по интервалу + событие `links`» (решение гейта 1c): готовый вектор
    # заметки будит петлю — событие берётся у воркера.
    assert specs["links"].wait_event is not None
    # Джоба обхода default (lsb-0011): своя очередь и расписание из env;
    # форма «по интервалу + событие» — сигнал `nodes` будит петлю сразу после
    # сшивания (lsb-0012).
    assert specs["nodes"].queue == "nodes"
    assert specs["nodes"].interval_sec == settings.job_nodes_interval_sec
    assert specs["nodes"].batch == settings.job_nodes_batch
    assert specs["nodes"].wait_event is not None
    assert specs["nodes"].queue_stat is not None
    for name in ("embedding", "summary", "judge", "areas"):
        assert specs[name].interval_sec == retry  # как было (FR-1.5)
    assert specs["expiration"].interval_sec == EXPIRATION_CLEANUP_INTERVAL_SEC
    # Форма «по интервалу + событие `embedding`» (gate 2026-09-19): свежая
    # pending-заметка будит петлю сразу — событие берётся у воркера (владельца
    # очереди векторизации). Перепроверки очереди нет: очередь векторизации и
    # есть ожидание модели — с ней петля крутилась бы вхолостую (busy-loop).
    assert specs["embedding"].wait_event is worker._embedding_event
    assert specs["embedding"].queue_empty is None
    assert specs["expiration"].wait_event is None
    assert specs["summary"].wait_event is not None
    assert specs["judge"].wait_event is not None
    assert specs["areas"].wait_event is not None
    # Гигиена worker_jobs — idle-ветка embedding-джобы (пул 5).
    assert specs["embedding"].idle_hook == worker._purge_done_jobs


def test_summary_job_is_not_registered_without_summarizer(settings) -> None:
    """Без суммаризатора summary-джоба неприменима — в реестр не попадает."""
    worker = BackgroundWorker(settings, HashEmbedder(8))
    names = [spec.name for spec in build_job_specs(worker, settings)]
    assert "summary" not in names
    assert names == ["links", "embedding", "judge", "areas", "expiration", "nodes"]


@pytest.mark.asyncio
async def test_fixed_schedule_ignores_progress_and_does_not_grow(
    monkeypatch,
) -> None:
    """Фиксированное расписание (expiration): пауза 300 с, прогресс не сдвигает.

    Состояние back-off с `fixed=True`: прогон, отчитавшийся о выполненной
    уборке, расписание не сдвигает — каркас спит ровно `interval_sec` и не
    растит интервал (300 с «как сейчас», FR-1.5).
    """
    delays = patch_sleep(monkeypatch)
    calls = 0

    async def process() -> int:
        nonlocal calls
        calls += 1
        return 1  # «уборка выполнена» — прогресс для очереди, не для расписания

    state = BackoffState(300, fixed=True)
    spec = make_spec(process, interval_sec=300)
    await asyncio.wait_for(run_loop(spec, lambda: calls >= 4, state), timeout=1.0)
    assert calls == 4
    assert delays == [300.0, 300.0, 300.0, 300.0]
    assert state.interval == 300.0  # back-off не вырос


@pytest.mark.asyncio
async def test_fixed_schedule_does_not_grow_on_failure(caplog, monkeypatch) -> None:
    """Сбой итерации фиксированной джобы: warning, пауза 300 с, без роста."""
    delays = patch_sleep(monkeypatch)
    calls = 0

    async def process() -> int:
        nonlocal calls
        calls += 1
        raise RuntimeError("boom")

    state = BackoffState(300, fixed=True)
    spec = make_spec(process, interval_sec=300)
    with caplog.at_level(logging.WARNING, logger="app"):
        await asyncio.wait_for(
            run_loop(spec, lambda: calls >= 3, state), timeout=1.0
        )
    assert calls == 3  # петля жива после сбоев
    assert delays == [300.0, 300.0, 300.0]
    assert state.interval == 300.0


@pytest.mark.asyncio
async def test_run_serves_registry_jobs(settings, monkeypatch) -> None:
    """`run()` обслуживает джобы реестра: своих циклов у воркера больше нет."""
    called = asyncio.Event()

    async def process() -> int:
        called.set()
        return 0  # пусто: джоба уходит в паузу (цикл живёт, yield есть)

    spec = make_spec(process, interval_sec=3600)
    monkeypatch.setattr(jobs, "JOB_BUILDERS", (lambda worker, settings: spec,))
    worker = BackgroundWorker(settings, HashEmbedder(8))
    task = asyncio.create_task(worker.run())
    try:
        await asyncio.wait_for(called.wait(), timeout=1.0)
    finally:
        worker.stop()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


# --- наблюдаемость очередей (lsb-0014-03) ------------------------------------


def _queue_waiting_records(caplog) -> list:
    """События `queue_waiting` из журнала (FR-2.3)."""
    return [
        record
        for record in caplog.records
        if getattr(record, "event", None) == "queue_waiting"
    ]


@pytest.mark.asyncio
async def test_queue_waiting_logged_when_waiting_with_pending_queue(
    monkeypatch, caplog
) -> None:
    """FR-2.3: уход в ожидание с непустой очередью — событие `queue_waiting`.

    Джоба формы «по интервалу» (без события, как `expiration`): прогон пуст,
    своя очередь
    не пуста (`queue_stat`) — «работа есть, но она не выполняется» видно в
    журнале, а не только в `/health`.
    """
    patch_sleep(monkeypatch)
    calls = 0

    async def process() -> int:
        nonlocal calls
        calls += 1
        return 0

    spec = make_spec(
        process,
        interval_sec=300,
        queue="vector",
        queue_stat=lambda: {"pending": 3, "oldest_pending_sec": 42},
    )
    with caplog.at_level(logging.INFO, logger="app"):
        await asyncio.wait_for(
            run_loop(spec, lambda: calls >= 2, BackoffState(300)), timeout=1.0
        )
    waited = _queue_waiting_records(caplog)
    assert len(waited) == 2  # по одному событию на каждый уход в ожидание
    first = waited[0]
    assert (first.job, first.queue) == ("synthetic", "vector")
    assert (first.pending, first.oldest_pending_sec) == (3, 42)


@pytest.mark.asyncio
async def test_queue_waiting_logged_before_event_wait(monkeypatch, caplog) -> None:
    """Форма «по требованию»: перед ожиданием события — тот же сигнал.

    Событие выставляется чуть позже (петля сама зовёт `clear()` после
    прогона — «по требованию» перепроверяет очередь), путь кода тот же:
    `queue_stat` перед `wait_for`.
    """
    patch_sleep(monkeypatch)
    event = asyncio.Event()
    calls = 0

    async def process() -> int:
        nonlocal calls
        calls += 1
        asyncio.get_running_loop().call_later(0.01, event.set)
        return 0

    spec = make_spec(
        process,
        interval_sec=300,
        queue="judge",
        wait_event=event,
        queue_empty=lambda: True,
        queue_stat=lambda: {"pending": 1, "oldest_pending_sec": 0},
    )
    with caplog.at_level(logging.INFO, logger="app"):
        await asyncio.wait_for(
            run_loop(spec, lambda: calls >= 2, BackoffState(300)), timeout=1.0
        )
    waited = _queue_waiting_records(caplog)
    assert len(waited) == 2
    assert (waited[0].job, waited[0].queue) == ("synthetic", "judge")
    assert (waited[0].pending, waited[0].oldest_pending_sec) == (1, 0)


@pytest.mark.asyncio
async def test_queue_waiting_silent_on_empty_or_absent_queue(
    monkeypatch, caplog
) -> None:
    """Пустая очередь и джоба без очереди — `queue_waiting` не пишется.

    Пустая очередь — ждать нечего; `expiration` очереди не имеет вовсе
    (`queue=None`) — снимок не берётся, даже если бы он был непустым.
    """
    patch_sleep(monkeypatch)
    calls = 0

    async def process() -> int:
        nonlocal calls
        calls += 1
        return 0

    empty = make_spec(
        process,
        interval_sec=300,
        queue="vector",
        queue_stat=lambda: {"pending": 0, "oldest_pending_sec": None},
    )
    invisible = make_spec(
        process,
        interval_sec=300,
        queue=None,
        queue_stat=lambda: {"pending": 5, "oldest_pending_sec": 7},
    )
    with caplog.at_level(logging.INFO, logger="app"):
        await asyncio.wait_for(
            run_loop(empty, lambda: calls >= 1, BackoffState(300)), timeout=1.0
        )
        await asyncio.wait_for(
            run_loop(invisible, lambda: calls >= 2, BackoffState(300)), timeout=1.0
        )
    assert _queue_waiting_records(caplog) == []
