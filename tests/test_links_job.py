"""lsb-0010-03: джоба расчёта связей и backfill.

Джоба `links` на каркасе lsb-0014: очередь — служебный маркер `notes.links_at`,
единое правило выборки для разового backfill и инкремента (свежие первыми,
батч за прогон), `queue_stat` для `/health.queues`, гигиена `purge_orphans` в
idle-ветке. Юниты на хосте: `vector_status` задаётся напрямую, вектора в
notes_vec может и не быть — расчёт связей моделей не зовёт (FR-2.2), поэтому
пометка маркера и порядок разбора проверяются без внешних сервисов.

Выдача связей (уровень 1 в транспорте) — постановка 10: здесь только расчёт.

Форма джобы — «по интервалу + событие `links`» (решение гейта 1c): заметка,
получившая готовый вектор (`BackgroundWorker.process_pending`), будит петлю
событием `_links_event` (`notify_links_pending`) — уровень 1 считается сразу, а
не через интервал. Здесь же проверяется, что при недоступных моделях
(`EmbeddingError`) сигнала нет и петля уходит в обычное ожидание с back-off.

Перепроверка очереди (`queue_empty`, пул 6, lost wakeup): перед ожиданием петля
смотрит ту же выборку, что `recompute_batch`, — сигнал, пришедший в окне до
`clear()`, работу не теряет, а заметка с `vector_status='pending'` очередь
непустой не делает (петля спит, а не крутится вхолостую).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import sqlite3
import time
from datetime import datetime, timedelta, timezone

import pytest
from fakes import FailingEmbedder, FixedSummarizer, HashEmbedder

from app.config import Settings, get_settings
from app.services.jobs import MAX_INTERVAL_SEC, BackoffState, build_job_specs, run_loop
from app.services.links import LINKS_JOB, LinksService
from app.services.worker import BackgroundWorker
from app.storage.db import init_db, session

DIM = 8


@pytest.fixture
def settings(test_env: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Схема в тестовой БД; размерность 8 — вектора не нужны (расчёт без LLM)."""
    monkeypatch.setenv("EMBEDDING_DIM", str(DIM))
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    return settings


def make_worker(settings: Settings) -> BackgroundWorker:
    """Воркер с суммаризатором: реестр содержит все джобы (включая `links`)."""
    return BackgroundWorker(settings, HashEmbedder(DIM), FixedSummarizer("Фикс."))


def links_spec(worker: BackgroundWorker, settings: Settings):
    """Описание джобы `links` из реестра каркаса."""
    specs = {spec.name: spec for spec in build_job_specs(worker, settings)}
    return specs[LINKS_JOB]


def _ts(seconds_ago: int) -> str:
    """ISO-8601 UTC-метка «N секунд назад» — как пишут её таблицы."""
    moment = datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def _seed(settings: Settings, rows: list[tuple]) -> None:
    """Заметки прямым SQL: (id, title, text, vector_status, updated_at)."""
    with session(settings) as conn:
        for note_id, title, text, vector_status, updated_at in rows:
            conn.execute(
                "INSERT INTO notes "
                "(id, title, text, namespace, vector_status, updated_at) "
                "VALUES (?, ?, ?, 'default', ?, ?)",
                (note_id, title, text, vector_status, updated_at),
            )


def _links_at(settings: Settings, note_id: int) -> str | None:
    with session(settings) as conn:
        return conn.execute(
            "SELECT links_at FROM notes WHERE id = ?", (note_id,)
        ).fetchone()[0]


def _sql_links_stat(settings: Settings) -> dict:
    """Прямой SQL по очереди `links` — эталон для снимка джобы."""
    with session(settings) as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS pending, "
            "MAX(CAST(strftime('%s','now') AS INTEGER) - "
            "CAST(strftime('%s', updated_at) AS INTEGER)) AS oldest_pending_sec "
            "FROM notes WHERE deleted_at IS NULL AND vector_status = 'ok' "
            "AND links_at IS NULL"
        ).fetchone()
    pending = int(row["pending"])
    oldest = row["oldest_pending_sec"]
    return {
        "pending": pending,
        "oldest_pending_sec": None if pending == 0 or oldest is None else int(oldest),
    }


def _run(spec) -> int:
    """Один прогон джобы (её `process` — async: синхронный SQL в потоке)."""
    return asyncio.run(spec.process())


def patch_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Мгновенный asyncio.sleep с записью запрошенных пауз (как в test_jobs)."""
    real_sleep = asyncio.sleep
    delays: list[float] = []

    async def spy(delay: float, *args: object, **kwargs: object) -> None:
        delays.append(delay)
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", spy)
    return delays


def patch_wait_event_timeout(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Мгновенный таймаут ожидания события с записью запрошенных пауз.

    Форма «по интервалу + событие» ждёт `wait_for(event.wait(), timeout)`:
    реальный таймаут (300 с) тест не переживёт — ожидание подменяется
    мгновенным `TimeoutError`, пауза записана, петля растит back-off.
    """
    delays: list[float] = []

    async def spy(awaitable, timeout=None, *args: object, **kwargs: object):
        awaitable.close()  # корутину event.wait() не ждём — как при таймауте
        delays.append(float(timeout))
        raise asyncio.TimeoutError

    monkeypatch.setattr(asyncio, "wait_for", spy)
    return delays


async def _wait_until(predicate, timeout: float = 1.0) -> None:
    """Дождаться условия живого цикла (опрос с yield'ами)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("условие не наступило за отведённое время")


# --- регистрация и расписание из окружения -----------------------------------


def test_links_job_is_registered(settings: Settings) -> None:
    """Джоба зарегистрирована описанием: очередь, интервал, батч, форма, хуки."""
    worker = make_worker(settings)
    spec = links_spec(worker, settings)
    assert spec.queue == LINKS_JOB
    assert spec.interval_sec == 300 == settings.job_links_interval_sec
    assert spec.batch == 100 == settings.job_links_batch
    assert spec.enabled is True
    # Форма «по интервалу + событие `links`» (гейт 1c): готовый вектор заметки
    # будит петлю — событие берётся у воркера (владельца embedding-очереди).
    assert spec.wait_event is worker._links_event
    # Перепроверка очереди задана (пул 6, lost wakeup) — форма с `queue_empty`:
    # работа, попавшая в окно до `clear()`, не теряется.
    assert spec.queue_empty is not None
    assert spec.idle_hook is not None  # гигиена purge_orphans
    assert spec.queue_stat is not None  # снимок очереди для /health


def test_job_env_overrides_schedule(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Три env джобы (FR-1.2): интервал/батч/выключение из окружения."""
    monkeypatch.setenv("JOB_LINKS_INTERVAL_SEC", "45")
    monkeypatch.setenv("JOB_LINKS_BATCH", "7")
    monkeypatch.setenv("JOB_LINKS_ENABLED", "false")
    get_settings.cache_clear()
    overridden = get_settings()
    spec = links_spec(make_worker(overridden), overridden)
    assert spec.interval_sec == 45
    assert spec.batch == 7
    assert spec.enabled is False


# --- батч, backfill и инкремент ----------------------------------------------


def test_run_processes_at_most_batch(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Прогон обрабатывает не более JOB_LINKS_BATCH заметок."""
    monkeypatch.setenv("JOB_LINKS_BATCH", "2")
    get_settings.cache_clear()
    overridden = get_settings()
    _seed(
        overridden,
        [
            (1, "Первая", "текст", "ok", _ts(30)),
            (2, "Вторая", "текст", "ok", _ts(20)),
            (3, "Третья", "текст", "ok", _ts(10)),
        ],
    )
    spec = links_spec(make_worker(overridden), overridden)
    assert spec.batch == 2

    assert _run(spec) == 2  # три заметки в очереди — потолок батча держит
    assert sum(_links_at(overridden, i) is not None for i in (1, 2, 3)) == 2
    assert _run(spec) == 1  # остаток разобран следующим прогоном
    assert all(_links_at(overridden, i) is not None for i in (1, 2, 3))


def test_backfill_newest_first_in_batches(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Backfill: N заметок разбираются за несколько прогонов, свежие первыми."""
    monkeypatch.setenv("JOB_LINKS_BATCH", "2")
    get_settings.cache_clear()
    overridden = get_settings()
    _seed(
        overridden,
        [
            (1, "Самая старая", "текст", "ok", _ts(400)),
            (2, "Вторая", "текст", "ok", _ts(300)),
            (3, "Третья", "текст", "ok", _ts(200)),
            (4, "Самая свежая", "текст", "ok", _ts(100)),
        ],
    )
    spec = links_spec(make_worker(overridden), overridden)
    assert spec.batch == 2

    assert _run(spec) == 2
    assert _links_at(overridden, 4) is not None
    assert _links_at(overridden, 3) is not None
    assert _links_at(overridden, 2) is None and _links_at(overridden, 1) is None

    assert _run(spec) == 2
    assert _links_at(overridden, 2) is not None and _links_at(overridden, 1) is not None

    assert _run(spec) == 0  # backlog опустел — джоба спит


def test_pending_vector_note_waits_for_vector(settings: Settings) -> None:
    """Заметка без вектора не выбирается и не помечается; после доготовки — да."""
    _seed(
        settings,
        [
            (1, "Готовая", "текст", "ok", _ts(0)),
            (2, "Ждёт вектора", "текст", "pending", _ts(0)),
        ],
    )
    spec = links_spec(make_worker(settings), settings)

    assert _run(spec) == 1  # pending-заметка в выборку не попала
    assert _links_at(settings, 1) is not None
    assert _links_at(settings, 2) is None  # задание не потеряно — ждёт вектора
    assert spec.queue_stat()["pending"] == 0  # pending-вектор в очереди links не виден

    with session(settings) as conn:
        conn.execute("UPDATE notes SET vector_status = 'ok' WHERE id = 2")

    assert spec.queue_stat()["pending"] == 1  # до-векторизованная попала в очередь
    assert _run(spec) == 1  # до-векторизованная заметка попала в выборку
    assert _links_at(settings, 2) is not None


def test_edit_returns_note_to_queue(settings: Settings) -> None:
    """Инкремент: правка текста/названия (сброс маркера) возвращает в очередь."""
    _seed(settings, [(1, "Заметка", "текст", "ok", _ts(0))])
    spec = links_spec(make_worker(settings), settings)

    assert spec.queue_stat()["pending"] == 1
    assert _run(spec) == 1
    assert spec.queue_stat()["pending"] == 0

    # NoteService.update при правке text/title сбрасывает links_at = NULL.
    with session(settings) as conn:
        conn.execute("UPDATE notes SET links_at = NULL WHERE id = 1")

    assert spec.queue_stat()["pending"] == 1
    assert _run(spec) == 1
    assert spec.queue_stat()["pending"] == 0


# --- событие `links`: готовый вектор будит петлю (решение гейта 1c) ----------


def test_process_pending_wakes_links_event(settings: Settings) -> None:
    """Готовый вектор заметки будит джобу `links`: уровень 1 считается сразу."""
    _seed(settings, [(1, "Заметка", "текст про встречу", "pending", _ts(0))])
    worker = make_worker(settings)
    assert worker._links_event.is_set() is False  # до довекторизации сигнала нет

    assert worker.process_pending() == 1  # заметка реально получила вектор
    assert worker._links_event.is_set()  # петля links проснётся немедленно

    spec = links_spec(worker, settings)
    assert _run(spec) == 1  # уровень 1 рассчитан, маркер проставлен
    assert _links_at(settings, 1) is not None


def test_embedding_error_does_not_wake_links(settings: Settings) -> None:
    """Отказ кодирования: события нет, задание ждёт готового вектора (back-off)."""
    _seed(settings, [(1, "Заметка", "текст про встречу", "pending", _ts(0))])
    worker = BackgroundWorker(settings, FailingEmbedder(), FixedSummarizer("Ф."))

    assert worker.process_pending() == 0  # модели недоступны — статусы не тронуты
    assert worker._links_event.is_set() is False  # сигнала нет (контракт сохранён)
    spec = links_spec(worker, settings)
    assert spec.queue_stat()["pending"] == 0  # очередь links пуста: вектора нет
    assert _run(spec) == 0  # задание не потеряно — ждёт готового вектора
    assert _links_at(settings, 1) is None


@pytest.mark.asyncio
async def test_links_event_wakes_loop_immediately(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Сигнал `links` будит петлю немедленно, не дожидаясь интервала (гейт 1c)."""
    worker = make_worker(settings)
    spec = links_spec(worker, settings)
    calls: list[int] = []

    def counting(self, limit: int) -> int:
        calls.append(limit)
        return 0

    monkeypatch.setattr(LinksService, "recompute_batch", counting)
    task = asyncio.create_task(
        run_loop(spec, lambda: False, BackoffState(spec.interval_sec))
    )
    try:
        await _wait_until(lambda: len(calls) >= 1)
        await asyncio.sleep(0.05)
        assert calls == [spec.batch]  # интервал 300 с: сама петля не проснётся
        worker.notify_links_pending()  # заметка получила готовый вектор
        await _wait_until(lambda: len(calls) >= 2, timeout=0.5)
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


# --- пул 6: потерянный будильник (lost wakeup) -------------------------------


def test_queue_empty_matches_recompute_selection(settings: Settings) -> None:
    """`queue_empty` смотрит ровно выборку `recompute_batch` (та же логика).

    Очередь непуста только из-за активной заметки с готовым вектором и пустым
    маркером; посчитанная, удалённая и ждущая вектора — не очередь.
    """
    _seed(
        settings,
        [
            (1, "Ждёт расчёта", "текст", "ok", _ts(0)),
            (2, "Уже посчитана", "текст", "ok", _ts(0)),
            (3, "Ждёт вектора", "текст", "pending", _ts(0)),
            (4, "Удалена", "текст", "ok", _ts(0)),
        ],
    )
    with session(settings) as conn:
        conn.execute("UPDATE notes SET links_at = ? WHERE id = 2", (_ts(0),))
        conn.execute("UPDATE notes SET deleted_at = ? WHERE id = 4", (_ts(0),))
    spec = links_spec(make_worker(settings), settings)

    assert spec.queue_empty() is False  # заметка 1 ждёт расчёта
    assert _run(spec) == 1  # прогон выгребает ровно её
    assert spec.queue_empty() is True  # очередь опустела вместе с выборкой
    assert spec.queue_stat()["pending"] == 0


@pytest.mark.asyncio
async def test_lost_wakeup_window_closed(settings: Settings, monkeypatch) -> None:
    """Работа, появившаяся до `clear()`, не теряется: следующий прогон сразу.

    Окно гонки — `idle_hook` перед `clear()`: заметка получила готовый вектор и
    `notify_links_pending` уже выставлен (без перепроверки `clear()` стёр бы
    сигнал, и заметка ждала бы интервал/back-off). Перепроверка очереди после
    `clear()` видит непустую выборку — петля НЕ уходит в сон, а считает сразу.
    """
    _seed(settings, [(1, "Заметка", "текст про встречу", "pending", _ts(0))])
    worker = make_worker(settings)

    def deliver(self) -> int:
        # Гигиена idle-ветки: ровно в этом окне приходит сигнал — ДО `clear()`.
        with session(settings) as conn:
            conn.execute("UPDATE notes SET vector_status = 'ok' WHERE id = 1")
        worker.notify_links_pending()
        return 0

    # Хук подменяется ДО сборки джобы: `JobSpec` хранит связанный метод.
    monkeypatch.setattr(LinksService, "purge_orphans", deliver)
    spec = links_spec(worker, settings)
    slept_early: list[bool] = []  # уснула ли петля до разбора очереди

    async def spy_wait_for(awaitable, timeout=None, *args, **kwargs):
        awaitable.close()  # корутину event.wait() не ждём — как при таймауте
        slept_early.append(_links_at(settings, 1) is None)
        raise asyncio.TimeoutError

    real_wait_for = asyncio.wait_for
    monkeypatch.setattr(asyncio, "wait_for", spy_wait_for)
    await real_wait_for(
        run_loop(
            spec, lambda: _links_at(settings, 1) is not None, BackoffState(300)
        ),
        timeout=2.0,
    )
    assert slept_early == []  # ни одного сна: работа разобрана сразу же
    assert _links_at(settings, 1) is not None  # посчитана сразу, не через интервал
    assert spec.queue_stat()["pending"] == 0


@pytest.mark.asyncio
async def test_pending_vector_note_does_not_spin_loop(
    settings: Settings, monkeypatch
) -> None:
    """Заметка, ждущая вектор, не крутит петлю вхолостую — она уходит в сон.

    `queue_empty` — та же выборка, что `recompute_batch` (`vector_status='ok'`):
    pending-заметка очередь непустой не делает, поэтому вместо busy-loop петля
    спит с back-off, а задание придёт событием от векторизации.
    """
    _seed(settings, [(1, "Ждёт вектора", "текст", "pending", _ts(0))])
    spec = links_spec(make_worker(settings), settings)
    calls: list[int] = []
    monkeypatch.setattr(
        LinksService,
        "recompute_batch",
        lambda self, limit: calls.append(limit) or 0,
    )
    real_wait_for = asyncio.wait_for  # ожидание теста — до подмены атрибута
    waits = patch_wait_event_timeout(monkeypatch)  # ожидание мгновенное, пауза записана
    state = BackoffState(300)

    await real_wait_for(
        run_loop(spec, lambda: len(waits) >= 2, state), timeout=2.0
    )
    assert spec.queue_empty() is True  # pending-вектор в очередь links не входит
    assert calls == [spec.batch, spec.batch]  # ровно по прогону на каждое ожидание
    assert waits[:2] == [300.0, 600.0]  # ушла в сон и растит back-off


# --- back-off, сон и idle-ветка ----------------------------------------------


@pytest.mark.asyncio
async def test_progress_resets_backoff(settings: Settings, monkeypatch) -> None:
    """Прогресс (заметка разобрана) сбрасывает выросший back-off."""
    patch_sleep(monkeypatch)
    _seed(settings, [(1, "Заметка", "текст", "ok", _ts(0))])
    spec = links_spec(make_worker(settings), settings)
    state = BackoffState(30)
    state.interval = float(MAX_INTERVAL_SEC)  # back-off успел дорасти до потолка

    await asyncio.wait_for(
        run_loop(spec, lambda: _links_at(settings, 1) is not None, state),
        timeout=2.0,
    )
    assert state.interval == float(spec.interval_sec)


@pytest.mark.asyncio
async def test_empty_queue_waits_event_and_grows_backoff(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Пустая выборка: гигиена idle-ветки, затем ожидание события с back-off.

    Форма «по интервалу + событие»: без сигнала петля не крутится вхолостую —
    ждёт `wait_for(event, timeout)` и по таймауту растит интервал (300 → 600).
    """
    idle: list[int] = []
    monkeypatch.setattr(
        LinksService, "purge_orphans", lambda self: idle.append(1) or 0
    )
    real_wait_for = asyncio.wait_for
    delays = patch_wait_event_timeout(monkeypatch)
    spec = links_spec(make_worker(settings), settings)
    state = BackoffState(300)

    task = asyncio.create_task(run_loop(spec, lambda: len(delays) >= 2, state))
    await real_wait_for(task, timeout=2.0)
    # Пустые прогоны: гигиена отработала перед каждым ожиданием, таймаут
    # ожидания растит back-off (300 → 600 → потолок 900) — вхолостую петля не
    # крутится.
    assert delays[:2] == [300.0, 600.0]
    assert state.interval == 900.0
    assert idle == [1, 1]


@pytest.mark.asyncio
async def test_idle_hook_db_error_does_not_kill_loop(
    settings: Settings, monkeypatch, caplog
) -> None:
    """Сбой гигиены (ошибка БД) не роняет джобу: warning и петля продолжает."""
    def boom(self) -> int:
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(LinksService, "purge_orphans", boom)
    delays = patch_sleep(monkeypatch)
    spec = links_spec(make_worker(settings), settings)

    with caplog.at_level(logging.WARNING, logger="app"):
        await asyncio.wait_for(
            run_loop(spec, lambda: len(delays) >= 1, BackoffState(300)), timeout=2.0
        )
    failed = [
        record
        for record in caplog.records
        if getattr(record, "event", None) == "loop_iteration_failed"
    ]
    assert failed and failed[0].job == LINKS_JOB  # петля жива, сбой виден в журнале


@pytest.mark.asyncio
async def test_disabled_job_not_started_but_queue_visible(
    settings: Settings, monkeypatch
) -> None:
    """JOB_LINKS_ENABLED=false: джоба не запускается, но очередь видна в /health."""
    monkeypatch.setenv("JOB_LINKS_ENABLED", "false")
    get_settings.cache_clear()
    disabled = get_settings()
    _seed(disabled, [(1, "Заметка", "текст", "ok", _ts(0))])
    called: list[int] = []
    monkeypatch.setattr(
        LinksService, "recompute_batch", lambda self, limit: called.append(limit) or 0
    )
    worker = make_worker(disabled)
    spec = links_spec(worker, disabled)

    assert spec.enabled is False
    queues = worker.queues_health()  # реестр джобу сохраняет
    assert "links" in queues
    assert queues["links"]["pending"] == 1  # накопление видно даже выключенной

    await asyncio.wait_for(
        run_loop(spec, lambda: False, BackoffState(300)), timeout=0.5
    )
    assert called == []  # выключенная джоба не запускается


# --- снимок очереди (FR-2.2) -------------------------------------------------


def test_queue_stat_matches_direct_sql(settings: Settings) -> None:
    """Снимок совпадает с прямым SQL: число и возраст ждущих заметок."""
    _seed(
        settings,
        [
            (1, "Ждёт", "текст", "ok", _ts(120)),
            (2, "Ждёт", "текст", "ok", _ts(5)),
            (3, "Ждёт вектора", "текст", "pending", _ts(0)),
            (4, "Посчитана", "текст", "ok", _ts(0)),
        ],
    )
    with session(settings) as conn:  # id=4 уже рассчитана — не в очереди links
        conn.execute("UPDATE notes SET links_at = '2026-01-01T00:00:00Z' WHERE id = 4")
    spec = links_spec(make_worker(settings), settings)

    stat = spec.queue_stat()
    assert stat["pending"] == 2  # pending-вектор и рассчитанная не считаются
    assert stat["oldest_pending_sec"] == pytest.approx(
        _sql_links_stat(settings)["oldest_pending_sec"], abs=1
    )
    assert stat["oldest_pending_sec"] >= 119


def test_queue_stat_empty_is_null(settings: Settings) -> None:
    """Пустая очередь: pending 0, oldest_pending_sec — null."""
    spec = links_spec(make_worker(settings), settings)
    assert spec.queue_stat() == {"pending": 0, "oldest_pending_sec": None}
