"""Ворота слота у внешних джоб и areas (3.4.1, LSBN-T-29; arch §6.2/§6.7).

До 3.4.1 migration/warnings заявляли слот `summary`, но при разрыве
продолжали дёргать мёртвый слот (гейта у них не было), а areas кодировала
слотом embedding, не заявляя его. Теперь ворота — в каркасе: `JobSpec.gate`
(для `run_loop`, с пробным стуком) и `JobSpec.is_gated` (для снимков
очередей — только память/SQL, никогда не зовёт модели). Юнит без сети:
сервис джоб подменяется фейком со счётчиком вызовов, разрыв открывается
вручную.

links и terms_normalize гейта не имеют намеренно: слот None — расчёт связей
моделей не зовёт (косинус по готовому вектору + FTS), нормализация терминов
работает без LLM (§13.10) — разрыв им не грозит.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import ClassVar

import pytest
from fakes import FixedClassifier, FixedSummarizer, HashEmbedder

from app.config import get_settings
from app.services import jobs, migration
from app.services import warnings as warnings_module
from app.services.jobs import BackoffState, run_loop
from app.services.llm_client import CHECK_AUTH_FAILED
from app.services.migration import build_migration_job
from app.services.slot_outcome import SlotOutcome
from app.services.warnings import build_warnings_job
from app.services.worker import (
    SLOT_EMBEDDING,
    BackgroundWorker,
    build_areas_job,
    build_summary_job,
)
from app.storage.db import init_db, session, transaction


@pytest.fixture
def settings(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "notes.db"))
    monkeypatch.setenv("EMBEDDING_DIM", "8")
    get_settings.cache_clear()
    resolved = get_settings()
    init_db(resolved)
    return resolved


def _freeze(monkeypatch, start: float = 1000.0) -> dict:
    """Заморозить часы модуля разрыва (детерминированные паузы)."""
    clock = {"t": start}
    monkeypatch.setattr("app.services.slot_breaker.time.time", lambda: clock["t"])
    return clock


def patch_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Мгновенный `asyncio.sleep` с записью запрошенных пауз (паттерн test_jobs)."""
    real_sleep = asyncio.sleep
    delays: list[float] = []

    async def spy(delay: float, *args: object, **kwargs: object) -> None:
        delays.append(delay)
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", spy)
    return delays


def _open_summary_break(worker: BackgroundWorker) -> None:
    """Открыть разрыв слота summary вручную (auth — разрыв сразу, arch §13.2)."""
    worker._breakers["summary"].note_failure(SlotOutcome.AUTH)


# --- (а) разрыв summary останавливает migration и warnings ---------------------


class _CountingSplitService:
    """Фейк `MigrationService` со счётчиком прогонов (подмена класса сборщика)."""

    instances: ClassVar[list[_CountingSplitService]] = []

    def __init__(self, settings, *, summarizer, notes, warnings=None) -> None:
        self.process_calls = 0
        type(self).instances.append(self)

    def process_batch(self, limit: int | None = None) -> int:
        self.process_calls += 1
        return 0

    def queue_empty(self) -> bool:
        return False

    def queue_stat(self) -> dict:
        return {"pending": 3, "oldest_pending_sec": 5}


class _CountingWarningsService:
    """Фейк `WarningsService` со счётчиком прогонов (подмена класса сборщика)."""

    instances: ClassVar[list[_CountingWarningsService]] = []

    def __init__(self, settings, classifier=None, namespaces=None) -> None:
        self.process_calls = 0
        type(self).instances.append(self)

    def run(self, batch: int | None = None) -> int:
        self.process_calls += 1
        return 0

    def queue_empty(self) -> bool:
        return False

    def queue_stat(self) -> dict:
        return {"pending": 2, "oldest_pending_sec": 8}


def test_summary_break_stops_migration(settings, monkeypatch, caplog) -> None:
    """Разрыв summary: джоба migration партию не берёт, очередь ждёт слот.

    Прогон сервиса не зовётся вовсе (`process_batch` — счётчик на нуле),
    гейт-0 — не прогресс петли (no39-05): один `queue_waiting` (снимок через
    `is_gated`: pending=0, waiting>0) и сон по back-off. Очередь джобы в
    `/health.queues` (`queues_health`) переезжает pending → waiting, как у
    прочих слотовых джоб.
    """
    monkeypatch.setattr(migration, "MigrationService", _CountingSplitService)
    worker = BackgroundWorker(settings, HashEmbedder(8), FixedSummarizer("Ф."))
    spec = build_migration_job(worker, settings)
    service = _CountingSplitService.instances[-1]
    _open_summary_break(worker)
    assert spec.gate is not None and spec.gate() is False

    delays = patch_sleep(monkeypatch)
    with caplog.at_level(logging.INFO, logger="app"):
        asyncio.run(
            run_loop(
                spec, lambda: len(delays) >= 1, BackoffState(spec.interval_sec)
            )
        )
    assert service.process_calls == 0  # сервис не дёргает мёртвый слот
    assert delays == [float(spec.interval_sec)]  # 0 — сон по back-off, не шторм

    stat = worker.queues_health()["migration"]
    assert stat == {
        "pending": 0, "oldest_pending_sec": 5, "waiting": 3, "attention": 0,
    }
    waited = [
        record for record in caplog.records
        if getattr(record, "event", None) == "queue_waiting"
        and getattr(record, "job", None) == "migration"
    ]
    assert len(waited) == 1
    assert waited[0].pending == 0
    assert waited[0].waiting == 3  # событие видит «работа ждёт слот»


def test_summary_break_stops_warnings(settings, monkeypatch) -> None:
    """Разрыв summary: джоба warnings партию не берёт, очередь ждёт слот."""
    monkeypatch.setattr(warnings_module, "WarningsService", _CountingWarningsService)
    worker = BackgroundWorker(
        settings, HashEmbedder(8), FixedSummarizer("Ф."),
        classifier=FixedClassifier(),
    )
    spec = build_warnings_job(worker, settings)
    service = _CountingWarningsService.instances[-1]
    _open_summary_break(worker)
    assert spec.gate is not None and spec.gate() is False

    delays = patch_sleep(monkeypatch)
    asyncio.run(
        run_loop(spec, lambda: len(delays) >= 1, BackoffState(spec.interval_sec))
    )
    assert service.process_calls == 0
    assert delays == [float(spec.interval_sec)]
    assert worker.queues_health()["warnings"] == {
        "pending": 0, "oldest_pending_sec": 8, "waiting": 2, "attention": 0,
    }


# --- (б) разрыв embedding останавливает areas ---------------------------------


@pytest.mark.asyncio
async def test_embedding_break_stops_areas(settings, monkeypatch, caplog) -> None:
    """Разрыв embedding: джоба areas партию не берёт, работа копится в waiting.

    areas кодирует embedding-слотом (№39, arch §6.9) — теперь слот заявлен
    (`JobSpec.slot`), ворота держат партию, а `/health.queues` показывает
    ожидание слота (прежде джоба не гейтилась вовсе).
    """
    worker = BackgroundWorker(settings, HashEmbedder(8))
    with session(settings) as conn, transaction(conn):
        conn.execute(
            "INSERT INTO assistant_logs (action, reason) VALUES ('deploy', 'why')"
        )
    calls: list[int] = []

    async def counting_process() -> int:
        calls.append(1)
        return 0

    monkeypatch.setattr(worker, "_process_areas", counting_process)
    spec = build_areas_job(worker, settings)  # захватывает подменённый прогон
    assert spec.slot == SLOT_EMBEDDING
    worker._breakers["embedding"].note_failure(SlotOutcome.AUTH)
    assert spec.gate is not None and spec.gate() is False
    # Обёртка каркаса (ровно та, что в обоих вызовах `run_loop`):
    assert await jobs._gated_process(spec) == 0
    assert calls == []

    with (
        caplog.at_level(logging.INFO, logger="app"),
        contextlib.suppress(asyncio.TimeoutError),
    ):
        # Реальный прогон петли 0.2 с: ворота держат всё время.
        await asyncio.wait_for(
            run_loop(spec, lambda: False, BackoffState(spec.interval_sec)),
            timeout=0.2,
        )
    assert calls == []  # партия не бралась, кодировщик не звался

    raw_pending = worker._areas_queue_stat()["pending"]
    assert raw_pending >= 1  # накопление на месте
    stat = worker.queues_health()["areas"]
    assert stat["pending"] == 0
    assert stat["waiting"] == raw_pending  # работа ждёт embedding-слот
    waited = [
        record for record in caplog.records
        if getattr(record, "event", None) == "queue_waiting"
        and getattr(record, "job", None) == "areas"
    ]
    assert waited and waited[0].pending == 0 and waited[0].waiting == raw_pending


# --- (в) снимок очереди без побочных эффектов ---------------------------------


class _ProbeSummarizer:
    """Суммаризатор-фейк со счётчиком probe: `summarize` зовётся нельзя."""

    def __init__(self) -> None:
        self.probe_calls = 0

    def probe(self) -> str:
        self.probe_calls += 1
        return CHECK_AUTH_FAILED

    def summarize(self, text: str) -> None:
        raise AssertionError("снимок очереди не зовёт модель")

    def close(self) -> None:  # интерфейс-совместимость с SummaryService
        return None


def test_is_gated_has_no_side_effects(settings, monkeypatch) -> None:
    """`is_gated` — только память: probe не зовётся и пауза не переставляется.

    Контракт `/health.queues`: наблюдаемость не жжёт вызовы. Даже при
    ИСТЁКШЕЙ паузе (стук «пора») снимок отвечает «не заблокирован», не делая
    пробного стука и не переставляя паузу; стук — забота `gate` (путь
    `run_loop`): он зовёт probe и по auth_failed переставляет паузу.
    """
    clock = _freeze(monkeypatch)
    summarizer = _ProbeSummarizer()
    worker = BackgroundWorker(settings, HashEmbedder(8), summarizer)
    spec = build_summary_job(worker, settings)
    _open_summary_break(worker)

    assert spec.is_gated is not None and spec.is_gated() is True
    assert summarizer.probe_calls == 0  # снимок молчит

    clock["t"] += settings.slot_pause_sec + 1  # пауза истекла — стук «пора»
    assert spec.is_gated() is False
    assert summarizer.probe_calls == 0  # probe НЕ звался
    assert worker._breakers["summary"].probe_due() is True  # пауза не переставлена

    assert spec.gate() is False  # путь run_loop: один стук
    assert summarizer.probe_calls == 1
    assert worker._breakers["summary"].is_paused() is True  # auth переставил
    assert spec.is_gated() is True
