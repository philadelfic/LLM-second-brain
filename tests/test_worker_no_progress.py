"""Отказ джобы не считается прогрессом петли (№39, no39-05; arch §6.9/§13.9).

Живой случай 12.10.2026: режим «decision-слот недоступен + `DECISION_FALLBACK=false`»
— задание остаётся `pending`, но петля считала его «берущимся сейчас» и била в
слот без back-off (2687 событий `decision_fallback` за 4 минуты, classify 1797 /
judge 890). Класс дефекта — тот же, что lsbdef-0011 (холостая петля без
фактической правки — не прогресс).

Здесь проверяем контракт: работа, которой слот отказывает, для петли — НЕ работа
(`_judge_queue_empty`/`_summary_queue_empty`/`_nodes_queue_empty` возвращают
истину), в `/health.queues` она видна как `waiting` (`pending == 0`), петля уходит
в ожидание по back-off, после возврата слота очередь разгружается без рестарта,
а замороженная пара (no39-04, `attention_at`) не берётся, как и раньше.

Юнит без сети: decision-клиент — на `httpx.MockTransport`; в режиме отказа он не
дёргается вовсе (каждый вызов — один `decision_fallback`, счётчик честный).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time

import httpx
import pytest
from fakes import (
    FailingEmbedder,
    FixedSummarizer,
    HashEmbedder,
    RecordingDedup,
    vectorize_notes,
)

from app.config import get_settings
from app.services.decision import DecisionService
from app.services.judge import JudgeService
from app.services.llm_client import LLMClient, SlotSpec
from app.services.notes import NoteService
from app.services.prompts import PromptRegistry
from app.services.worker import BackgroundWorker
from app.storage.db import init_db, session, transaction

DIM = 8
JUDGE_QUESTION = "is_text_duplicate"
TEXT_NEW = "по дороге домой зашёл в магазин: молоко и хлеб"
TEXT_OLD = "купил молоко и хлеб по дороге домой"


@pytest.fixture
def settings(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """БД во временной папке + decision-слот включён, переключение выключено."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "notes.db"))
    monkeypatch.setenv("EMBEDDING_DIM", str(DIM))
    monkeypatch.setenv("DECISION_ENABLED", "true")
    monkeypatch.setenv("DECISION_BASE_URL", "http://127.0.0.1:11434")
    monkeypatch.setenv("DECISION_MODEL", "decision-test")
    monkeypatch.setenv("DECISION_FALLBACK", "false")
    get_settings.cache_clear()
    resolved = get_settings()
    init_db(resolved)
    return resolved


class DecisionTransport:
    """Транспорт decision-слота: «не дубль» по умолчанию, журнал запросов.

    В режиме «слот недоступен» запросов быть не должно — журнал это показывает
    (`requests == []`), а не только счётчик сервиса.
    """

    def __init__(self, noul: float = 0.02) -> None:
        self.noul = noul
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(
            200, json={"answers": {JUDGE_QUESTION: {"type": "noul", "noul": self.noul}}}
        )


def _no_network(request: httpx.Request) -> httpx.Response:  # pragma: no cover
    """Генеративный путь судьи в этих тестах зван быть не должен."""
    raise AssertionError("генеративный путь судьи не должен вызываться")


def make_decision(settings, *, ok: bool) -> tuple[DecisionService, DecisionTransport]:
    """Decision-слот: `ok=False` — «недоступен» (`can_decide()` ложно, как при `disable()`)."""
    transport = DecisionTransport()
    service = DecisionService(
        settings,
        llm=LLMClient(SlotSpec.for_decision(settings), transport=transport.handler),
        registry=PromptRegistry(None),
    )
    if not ok:
        service.disable()
    return service, transport


def make_worker(
    settings, decision: DecisionService, *, candidates: tuple = ()
) -> BackgroundWorker:
    """Воркер с суммаризатором и судьёй на том же decision-слоте (как в проде).

    `candidates` — заранее известные косинус-кандидаты для пары-«почти-дубля».
    """
    judge = JudgeService(
        settings,
        httpx.MockTransport(_no_network),
        registry=PromptRegistry(None),
        decider=decision,
    )
    worker = BackgroundWorker(
        settings,
        FailingEmbedder(),
        FixedSummarizer("Фикс-суммари."),
        RecordingDedup(candidates=list(candidates)),
        judge=judge,
        decision=decision,
    )
    worker._ensure_job_table()
    return worker


def seed_judge_job(settings, worker: BackgroundWorker) -> int:
    """Ровно одна пара-«почти-дубль» в очереди judge: две заметки с векторами.

    Заметка берётся в работу только с готовым вектором — вектора гоняем фейковым
    кодировщиком, как это делает воркер (он же ставит judge-работу на каждую
    свежую заметку: их снимаем, очередь наполняем сами — одна пара, один
    вердикт за прогон). Косинус-кандидат задан фейком дедупа при сборке воркера.
    """
    notes = NoteService(settings, HashEmbedder(DIM))
    notes.save(TEXT_OLD, title="Покупки")
    notes.save(TEXT_NEW, title="Магазин")
    vectorize_notes(settings, HashEmbedder(DIM))
    with session(settings) as conn, transaction(conn):
        conn.execute("UPDATE worker_jobs SET status = 'done'")
    worker._ensure_job("judge", "dedup", 2)
    return 2


def _events(caplog, name: str) -> list:
    return [
        record
        for record in caplog.records
        if getattr(record, "event", None) == name
    ]


def _pending_judge_jobs(settings) -> int:
    with session(settings) as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM worker_jobs WHERE slot = 'judge' "
            "AND status = 'pending'"
        ).fetchone()
    return int(row[0])


def _pending_judge_id(settings) -> int:
    """Id единственной ожидающей judge-работы (очередь наполняет сам тест)."""
    with session(settings) as conn:
        row = conn.execute(
            "SELECT id FROM worker_jobs WHERE slot = 'judge' AND status = 'pending'"
        ).fetchone()
    assert row is not None
    return int(row["id"])


def _freeze(settings, job_id: int) -> None:
    """Заморозить пару (`attention_at`) — «ждёт вмешательства человека» (no39-04)."""
    with session(settings) as conn, transaction(conn):
        conn.execute(
            "UPDATE worker_jobs SET attention_at = datetime('now'), attempts = 8 "
            "WHERE id = ?",
            (job_id,),
        )


async def wait_until(predicate, timeout: float = 2.0) -> None:
    """Дождаться условия живой петли (опрос с yield'ами)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("условие не наступило за отведённое время")


async def _run(job, *, seconds: float):
    """Прогнать петлю джобы окно времени и мягко остановить её."""
    task = asyncio.create_task(job)
    try:
        await asyncio.sleep(seconds)
    finally:
        with contextlib.suppress(asyncio.CancelledError):
            task.cancel()
            await task
    return task


# --- «работа не берётся»: очередь для петли пуста ----------------------------


@pytest.mark.asyncio
async def test_unreachable_decision_no_fallback_is_waiting(settings, caplog) -> None:
    """Слот недоступен + переключение выключено: работа ждёт, петля спит по back-off.

    Пара-«почти-дубль» лежит в очереди judge, но вердикта ей взять негде:
    очередь для петли ПУСТА (`_judge_queue_empty()`), в `/health.queues` она
    видна как `waiting` (`pending == 0`), а петля уходит в ожидание — счётчик
    `decision_fallback` за окно не превышает одной честной попытки, шквала
    `dedup_judge_failed` нет, задание не теряется (остаётся `pending`).
    """
    decision, transport = make_decision(settings, ok=False)
    worker = make_worker(settings, decision, candidates=[(1, 0.9)])
    seed_judge_job(settings, worker)

    assert worker._judge_queue_empty() is True
    queues = worker.queues_health()
    assert queues["judge"]["pending"] == 0
    assert queues["judge"]["waiting"] >= 1

    with caplog.at_level(logging.INFO, logger="app"):
        await _run(worker._run_judge(), seconds=0.2)

    assert len(_events(caplog, "decision_fallback")) <= 1
    assert len(_events(caplog, "dedup_judge_failed")) <= 1
    assert transport.requests == []  # слот погашен — вызова не было
    assert _pending_judge_jobs(settings) == 1  # задание не потеряно


def test_summary_and_nodes_wait_when_decision_unreachable(settings) -> None:
    """Классификатору нужен decision: очереди `summary` и `nodes` тоже пусты для петли.

    Заметка без выжимки ждёт слот summary, кандидат классификаторного пула — слот
    summary с classification через decision: при недоступном решении и
    выключенном переключении ни та, ни другая работа не берётся, но обе видны в
    `/health.queues` как `waiting` (`pending == 0`).
    """
    decision, _ = make_decision(settings, ok=False)
    worker = make_worker(settings, decision)
    NoteService(settings, HashEmbedder(DIM)).save("заметка без названия, ждёт выжимку")
    NoteService(settings, HashEmbedder(DIM)).save("уже выжатая заметка", title="Выжимка")
    with session(settings) as conn, transaction(conn):
        conn.execute(
            "UPDATE notes SET summary = ?, summary_status = 'ok' WHERE id = 2",
            ("Готовая выжимка заметки.",),
        )

    assert worker._summary_queue_empty() is True
    assert worker._nodes_queue_empty() is True

    summary = worker.queues_health()["summary"]
    nodes = worker.queues_health()["nodes"]
    assert summary["pending"] == 0
    assert summary["waiting"] == 1  # заметка без выжимки ждёт слот
    assert nodes["pending"] == 0
    assert nodes["waiting"] == 1  # кандидат классификаторного пула ждёт слот


@pytest.mark.asyncio
async def test_live_scenario_no_fallback_storm(settings, caplog) -> None:
    """Живой сценарий 12.10.2026: за окно 2 с — не больше одной попытки на back-off.

    Было ≈11 вызовов/с (2687 событий за 4 минуты); здесь judge-петля гоняется на
    фейковом транспорте 2 с при недоступном decision и `DECISION_FALLBACK=false`:
    счётчик переключений растёт не быстрее back-off (≤ 2 за окно), шквала
    `dedup_judge_failed` в журнале нет.
    """
    decision, _ = make_decision(settings, ok=False)
    worker = make_worker(settings, decision, candidates=[(1, 0.9)])
    seed_judge_job(settings, worker)

    with caplog.at_level(logging.INFO, logger="app"):
        await _run(worker._run_judge(), seconds=2.0)

    assert len(_events(caplog, "decision_fallback")) <= 2
    assert len(_events(caplog, "dedup_judge_failed")) <= 2
    assert decision.snapshot()["fallbacks"] <= 2


@pytest.mark.asyncio
async def test_recovery_drains_queue(settings, monkeypatch) -> None:
    """Слот «починили» (`can_decide()` истинно) → очередь разгружается без рестарта.

    Пока решения нет, петля спит; сигнал `notify_judge_pending` будит её, и тот же
    прогон с живым слотом снимает работу (`waiting` падает до 0) — перезапуска
    воркера для возврата слота не требуется.
    """
    decision, transport = make_decision(settings, ok=False)
    worker = make_worker(settings, decision, candidates=[(1, 0.9)])
    seed_judge_job(settings, worker)
    assert worker._judge_queue_empty() is True

    task = asyncio.create_task(worker._run_judge())
    try:
        await asyncio.sleep(0.05)
        monkeypatch.setattr(decision, "can_decide", lambda: True)  # слот вернулся
        assert worker._judge_queue_empty() is False
        worker.notify_judge_pending()  # сигнал будит петлю, рестарта нет
        await wait_until(
            lambda: worker.queues_health()["judge"]["pending"] == 0
            and worker.queues_health()["judge"]["waiting"] == 0
        )
    finally:
        worker.stop()
        with contextlib.suppress(asyncio.CancelledError):
            task.cancel()
            await task

    assert len(transport.requests) == 1  # вердикт взят у живого слота
    assert _pending_judge_jobs(settings) == 0


def test_attention_pair_still_waits(settings) -> None:
    """Регресс no39-04: замороженная пара (`attention_at`) не берётся, как и раньше.

    Слот decision при этом жив — очередь пуста именно из-за внимания: работа
    остаётся `pending` в таблице, но `attention`/`waiting` — 1, `pending` — 0.
    """
    decision, _ = make_decision(settings, ok=True)
    worker = make_worker(settings, decision)
    seed_judge_job(settings, worker)
    _freeze(settings, _pending_judge_id(settings))

    assert worker._judge_queue_empty() is True
    stat = worker.queues_health()["judge"]
    assert stat["pending"] == 0
    assert stat["waiting"] == 1
    assert stat["attention"] == 1
