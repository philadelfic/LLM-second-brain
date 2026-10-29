"""Снятие партии при wait-класс отказах слота (LSBN-T-28; прецедент — миграция).

Живой дефект: партии `process_summary_pending`/`process_title_pending`/
`process_judge_pending` не прерывались при транзиентных отказах — каждый отказ
ловился per-item и делал `continue`, поэтому «висящий» провайдер (read-таймаут
900 с, config.py `summary_timeout_sec`/`judge_timeout_sec`) держал петлю и
поток до 8 часов на партию из 32 позиций: петля не в back-off, событие
`queue_waiting` не пишется, деградация невидима в журнале.

Контракт после LSBN-T-28 (здесь и проверяем):

* wait-класс исходов (`_BATCH_ABORT_OUTCOMES`: UNAVAILABLE/TIMEOUT/AUTH) на
  любой позиции снимает партию (`break`): модель вызвана только до отказа,
  возврат — число обработанных ДО отказа (сделанное засчитывается, как у
  джобы миграции `if outcome == "retry-later": break`), хвост остаётся
  pending — работа НЕ теряется: back-off петли, пробный стук разрыва и будка
  нотификатором добирают её после возврата сервиса;
* EMPTY/TRUNCATED/OFF_CONTRACT остаются per-item (`continue`): «провайдер
  лежит» они не означают — партия ПРОДОЛЖАЕТСЯ (проверяется на EMPTY);
* при AUTH разрыв слота открывается сразу — гейт каркаса (LSBN-T-29) не
  пустит следующую партию до снятия разрыва;
* отказ не на первой позиции: обработанные до отказа позиции записаны и
  засчитаны в возврат, повторным прогоном (сервис «починили») хвост
  догоняется без потерь.

Юнит без сети: суммаризатор/судья — тестовые двойники, отказы доменных ошибок
(`SummaryError`/`JudgeError`) несут исход `outcome=` нужного класса — тот же
путь `outcome_of`, что и у живых сервисов; векторизация — `HashEmbedder`.
"""

from __future__ import annotations

import pytest
from fakes import FailingEmbedder, HashEmbedder, RecordingDedup, fake_meta

from app.config import get_settings
from app.services.judge import JudgeError, JudgeResult
from app.services.notes import NoteService
from app.services.slot_outcome import SlotOutcome
from app.services.summary import SummaryError, SummaryResult
from app.services.worker import BackgroundWorker
from app.storage.db import init_db, session, transaction

BATCH = 3  # размер партии джоб (`EMBEDDING_BATCH_SIZE`) во всех сценариях
TEXTS = [
    "заметка партии на отказ слота №1",
    "заметка партии на отказ слота №2",
    "заметка партии на отказ слота №3",
]
WAIT_OUTCOMES = [
    SlotOutcome.TIMEOUT,
    SlotOutcome.UNAVAILABLE,
    SlotOutcome.AUTH,
]


@pytest.fixture
def settings(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "notes.db"))
    monkeypatch.setenv("EMBEDDING_DIM", "8")
    monkeypatch.setenv("EMBEDDING_BATCH_SIZE", str(BATCH))
    get_settings.cache_clear()
    resolved = get_settings()
    init_db(resolved)
    return resolved


class OutcomeSummarizer:
    """Суммаризатор-фейк: отказ нужного исхода на выбранных по счёту вызовах.

    `fail_outcome` — исход `SummaryError`; `fail_at` — номера вызовов
    (с единицы, в порядке партии), на которых фейк отказывает; остальные
    вызовы успешны. Журнал `calls` — запрошенные тексты: тесты проверяют,
    что партия снята ровно на отказавшей позиции (последующие не звались).
    """

    def __init__(
        self,
        fail_outcome: SlotOutcome | None = None,
        *,
        fail_at: frozenset[int] = frozenset(),
    ) -> None:
        self.fail_outcome = fail_outcome
        self.fail_at = fail_at
        self.calls: list[str] = []
        self.last_attempt_ok: bool | None = None

    def _fail_or_result(self, text: str, operation: str) -> SummaryResult:
        position = len(self.calls)  # вызов уже залогирован — счёт с единицы
        if self.fail_outcome is not None and position in self.fail_at:
            self.last_attempt_ok = False
            raise SummaryError("слот недоступен", outcome=self.fail_outcome)
        self.last_attempt_ok = True
        return SummaryResult(
            text="Фикс-суммари одной строкой.", meta=fake_meta("summary", operation)
        )

    def summarize(self, text: str) -> SummaryResult:
        self.calls.append(text)
        return self._fail_or_result(text, "summary")

    def title(self, text: str) -> SummaryResult:
        self.calls.append(text)
        return self._fail_or_result(text, "title")

    def close(self) -> None:  # интерфейс-совместимость с SummaryService
        return None


class OutcomeJudge:
    """Судья-фейк: отказ нужного исхода на выбранных по счёту вызовах.

    Интерфейс Judge (как `FlakyJudge` из test_worker_queue_unavailable_models,
    но с точкой отказа по номеру вызова): живой ответ — «НЕ ДУБЛЬ» (слияний
    нет, judge-работа снимается), отказ — `JudgeError` с исходом `fail_outcome`.
    """

    def __init__(
        self,
        fail_outcome: SlotOutcome | None = None,
        *,
        fail_at: frozenset[int] = frozenset(),
    ) -> None:
        self.fail_outcome = fail_outcome
        self.fail_at = fail_at
        self.calls = 0

    def judge(self, text_new: str, text_candidate: str) -> JudgeResult:
        self.calls += 1
        if self.fail_outcome is not None and self.calls in self.fail_at:
            raise JudgeError("судья недоступен", outcome=self.fail_outcome)
        return JudgeResult(verdict=False, meta=fake_meta("judge", "judge"))

    def probe(self) -> str:  # интерфейс-совместимость с JudgeService
        return "ok"

    def close(self) -> None:  # интерфейс-совместимость с JudgeService
        return None


def _make_worker(settings, summarizer, judge=None) -> BackgroundWorker:
    worker = BackgroundWorker(
        settings,
        HashEmbedder(8),
        summarizer,
        RecordingDedup(),
        judge=judge,
    )
    worker._ensure_job_table()
    return worker


def _save_notes(settings, count: int = BATCH) -> None:
    """Партия заметок без выжимки (summary_status='pending') и без названия."""
    notes = NoteService(settings, FailingEmbedder())
    for text in TEXTS[:count]:
        notes.save(text)  # без title — легаси-путь: title IS NULL


def _summary_statuses(settings) -> list[str]:
    with session(settings) as conn:
        rows = conn.execute(
            "SELECT summary_status FROM notes ORDER BY id"
        ).fetchall()
    return [row["summary_status"] for row in rows]


def _titles(settings) -> list[str | None]:
    with session(settings) as conn:
        rows = conn.execute("SELECT title FROM notes ORDER BY id").fetchall()
    return [row["title"] for row in rows]


# --- (а)/(б): wait-класс на первой позиции — партия снята -------------------


@pytest.mark.parametrize("outcome", WAIT_OUTCOMES, ids=lambda o: o.value)
def test_summary_abort_on_first_position(settings, outcome) -> None:
    """TIMEOUT/UNAVAILABLE/AUTH на первой позиции: партия снята, хвост цел.

    Модель (фейк-суммаризатор со счётчиком) вызвана ровно 1 раз — вторая и
    третья позиции партии не жгутся; все позиции остались pending (работа не
    потеряна), функция вернула 0 (ни одной обработанной).
    """
    summarizer = OutcomeSummarizer(outcome, fail_at=frozenset({1}))
    worker = _make_worker(settings, summarizer)
    _save_notes(settings)

    assert worker.process_summary_pending() == 0
    assert len(summarizer.calls) == 1  # партия снята на первой же позиции
    assert _summary_statuses(settings) == ["pending"] * BATCH  # ничего не потеряно


@pytest.mark.parametrize("outcome", WAIT_OUTCOMES, ids=lambda o: o.value)
def test_judge_abort_on_first_position(settings, outcome) -> None:
    """Judge, wait-класс на первой позиции партии: снятие, работы целы.

    Пять judge-работ (кандидат — заметка 1), отказ на первом вызове судьи:
    остальные позиции не звались, возврат 0, все работы остались pending.
    AUTH вдобавок открывает разрыв слота сразу — гейт каркаса (LSBN-T-29)
    не пустит следующую партию, пока разрыв не снят (стук/возврат сервиса).
    """
    notes = NoteService(settings, FailingEmbedder())
    for index in range(1, 7):
        notes.save(f"заметка очереди отказавшего судьи №{index}")
    judge = OutcomeJudge(outcome, fail_at=frozenset({1}))
    worker = BackgroundWorker(
        settings,
        HashEmbedder(8),
        OutcomeSummarizer(),
        RecordingDedup(candidates=[(1, 0.94)]),
        judge=judge,
    )
    vectorized = 0
    while (portion := worker.process_pending()) > 0:  # батч 3: две партии
        vectorized += portion
    assert vectorized == 6
    with session(settings) as conn, transaction(conn):
        conn.execute("DELETE FROM worker_jobs WHERE note_id = 1")
    jobs = worker._pending_jobs("judge", "dedup", 10)
    assert len(jobs) == 5

    assert worker.process_judge_pending() == 0
    assert judge.calls == 1  # партия снята на первой же отказавшей паре
    remaining = worker._pending_jobs("judge", "dedup", 10)
    assert [job["id"] for job in remaining] == [job["id"] for job in jobs]
    if outcome is SlotOutcome.AUTH:
        assert worker._breakers["judge"].is_paused() is True  # гейт закроет партию


@pytest.mark.parametrize("outcome", WAIT_OUTCOMES, ids=lambda o: o.value)
def test_title_abort_on_first_position(settings, outcome) -> None:
    """Title-доген, wait-класс на первой позиции: снятие, названия целы (NULL)."""
    summarizer = OutcomeSummarizer(outcome, fail_at=frozenset({1}))
    worker = _make_worker(settings, summarizer)
    _save_notes(settings)

    assert worker.process_title_pending() == 0
    assert len(summarizer.calls) == 1  # партия снята на первой же позиции
    assert _titles(settings) == [None] * BATCH  # названия не потеряны — догонит back-off


# --- (в): EMPTY — партия ПРОДОЛЖАЕТСЯ ----------------------------------------


def test_summary_empty_outcome_continues_batch(settings) -> None:
    """EMPTY не снимает партию (не wait-класс): позиция пропускается, партия идёт.

    Фейк отказывает EMPTY на всех позициях: вызовов — по числу партии (все
    позиции отработаны), каждая осталась pending по семантике EMPTY
    (per-item `continue`; покрытие — разрыв слота после трёх отказов),
    возврат 0.
    """
    summarizer = OutcomeSummarizer(SlotOutcome.EMPTY, fail_at=frozenset({1, 2, 3}))
    worker = _make_worker(settings, summarizer)
    _save_notes(settings)

    assert worker.process_summary_pending() == 0
    assert len(summarizer.calls) == BATCH  # партия НЕ снята — все позиции звались
    assert _summary_statuses(settings) == ["pending"] * BATCH  # семантика EMPTY: pending


# --- (г): отказ не на первой позиции — сделанное засчитывается ---------------


def test_summary_abort_mid_batch_counts_done_then_drains(settings) -> None:
    """TIMEOUT на третьей позиции: первые две записаны, возврат 2, хвост pending.

    Сделанное засчитывается (как у миграции): возврат 2 — прогресс петли,
    интервал сбрасывается. Хвост остаётся pending; «починили» сервис —
    следующий прогон догоняет его без потерь (работа не теряется).
    """
    summarizer = OutcomeSummarizer(SlotOutcome.TIMEOUT, fail_at=frozenset({3}))
    worker = _make_worker(settings, summarizer)
    _save_notes(settings)

    assert worker.process_summary_pending() == 2  # обработанные до отказа
    assert len(summarizer.calls) == 3  # отказ — на последней позиции партии
    assert _summary_statuses(settings) == ["ok", "ok", "pending"]

    summarizer.fail_outcome = None  # сервис вернулся
    summarizer.fail_at = frozenset()
    assert worker.process_summary_pending() == 1  # хвост добран
    assert _summary_statuses(settings) == ["ok"] * BATCH  # работа не потеряна


def test_judge_abort_mid_batch_counts_done(settings) -> None:
    """Judge, TIMEOUT на второй позиции партии: первая снята (done), возврат 1.

    Верdict «НЕ ДУБЛЬ» по первой паре — работа снята и засчитана; вторая
    пара отказала TIMEOUT — партия снята, хвост (в партии и за ней)
    остался pending.
    """
    notes = NoteService(settings, FailingEmbedder())
    for index in range(1, 7):
        notes.save(f"заметка очереди судьи с отказом в середине №{index}")
    judge = OutcomeJudge(SlotOutcome.TIMEOUT, fail_at=frozenset({2}))
    worker = BackgroundWorker(
        settings,
        HashEmbedder(8),
        OutcomeSummarizer(),
        RecordingDedup(candidates=[(1, 0.94)]),
        judge=judge,
    )
    vectorized = 0
    while (portion := worker.process_pending()) > 0:  # батч 3: две партии
        vectorized += portion
    assert vectorized == 6
    with session(settings) as conn, transaction(conn):
        conn.execute("DELETE FROM worker_jobs WHERE note_id = 1")

    assert worker.process_judge_pending() == 1  # первая работа снята, вторая отказала
    assert judge.calls == 2
    with session(settings) as conn:
        rows = conn.execute(
            "SELECT note_id, status FROM worker_jobs WHERE slot = 'judge' "
            "AND kind = 'dedup' ORDER BY id"
        ).fetchall()
    assert [(row["note_id"], row["status"]) for row in rows] == [
        (2, "done"),  # вердикт получен до отказа
        (3, "pending"),  # TIMEOUT — партия снята
        (4, "pending"),  # хвост партии не звался
        (5, "pending"),  # за партией — следующий прогон
        (6, "pending"),  # за партией — следующий прогон
    ]
