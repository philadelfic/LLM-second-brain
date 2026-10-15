"""Очередь при недоступных моделях → доступны (Backlog №39, релиз 3.3.0).

Сквозной сценарий по judge-очереди (N=3 задания): пока слот судьи недоступен,
очередь ЖДЁТ — задания остаются `pending`, не теряются и не замораживаются,
`/health.queues` показывает их честно; когда слот оживает — очередь
выгребается до нуля. Два варианта недоступности:

* **транзиентный отказ** (`UNAVAILABLE` — сеть/5xx/429, живой путь
  `classify_llm_error`): разрыв слота НЕ открывается
  (`SlotBreaker._TRANSIENT_OUTCOMES`), партия берётся, судья отказывает по
  каждой паре, работа остаётся pending, счёт пары (`attempts`) и внимание
  (`attention_at`) не растут — ждать модели-недоступность нормально
  (arch §6.2/§13.2);
* **разрыв слота** (`AUTH` — 401/403): разрыв открывается сразу, закрытые
  ворота партию не берут (вызовы модели не жгутся), работа видна в
  `waiting`; после истечения паузы один успешный `probe()` снимает разрыв —
  очередь выгребается (arch §6.2, §13.2).

Числа наблюдаемости (`queues_health`/`queue_stat`) сверяются с фактом по
строкам `worker_jobs` — «числа совпадают с фактом» из приёмки брифа 3.3.0.

Прогон — прямыми вызовами `_process_judge` (тело петли каркаса), без снов:
асинхронное ожидание `run_loop` (back-off, `queue_waiting`) покрыто
соседними тестами (`test_jobs.py`, `test_worker_queue_waiting_event.py`),
здесь важен контракт очереди, а не форма расписания; драйвер петли
`_judge_queue_empty` проверяется напрямую. Сети нет: судья — тестовый двойник
с отказами `JudgeError` нужного исхода, векторизация — `HashEmbedder`.
"""

from __future__ import annotations

import pytest
from fakes import FailingEmbedder, FixedSummarizer, HashEmbedder, RecordingDedup, fake_meta

from app.config import get_settings
from app.services.judge import JudgeError, JudgeResult
from app.services.notes import NoteService
from app.services.slot_outcome import SlotOutcome
from app.services.worker import BackgroundWorker
from app.storage.db import init_db, session, transaction


@pytest.fixture
def settings(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "notes.db"))
    monkeypatch.setenv("EMBEDDING_DIM", "8")
    get_settings.cache_clear()
    resolved = get_settings()
    init_db(resolved)
    return resolved


class FlakyJudge:
    """Судья-фейк: фаза «недоступен» — JudgeError с заданным исходом.

    `fail_outcome` — исход отказа (`UNAVAILABLE`/`AUTH`); None — слот
    отвечает (вердикт «не дубль», слияний нет — очередь judge чистая).
    Журналы `judge_calls`/`probe_calls` — тест проверяет, что при закрытых
    воротах вызовы модели не жгутся.
    """

    def __init__(self, fail_outcome: SlotOutcome | None = None) -> None:
        self.fail_outcome = fail_outcome
        self.verdict = False
        self.judge_calls = 0
        self.probe_calls = 0

    def judge(self, text_new: str, text_candidate: str) -> JudgeResult:
        self.judge_calls += 1
        if self.fail_outcome is not None:
            raise JudgeError("judge slot unavailable", outcome=self.fail_outcome)
        return JudgeResult(verdict=self.verdict, meta=fake_meta("judge", "judge"))

    def probe(self) -> str:
        self.probe_calls += 1
        return "ok"

    def close(self) -> None:  # интерфейс-совместимость с JudgeService
        return None


def _freeze(monkeypatch, start: float = 1000.0) -> dict:
    """Заморозить часы модуля разрыва (детерминированные паузы)."""
    clock = {"t": start}
    monkeypatch.setattr("app.services.slot_breaker.time.time", lambda: clock["t"])
    return clock


def _worker_with_three_judge_jobs(settings, judge: FlakyJudge) -> BackgroundWorker:
    """Четыре векторизованные заметки; 3 judge-работы, каждая идёт в судью.

    Кандидат один — ранняя заметка 1 (`RecordingDedup`), поэтому у заметок
    2–4 работа доходит до вызова судьи, а работа заметки 1 снимается сразу
    (раньше неё кандидатов нет — она и её работа убраны настроечным DELETE).
    """
    notes = NoteService(settings, FailingEmbedder())
    for index in range(1, 5):
        notes.save(f"заметка очереди недоступного судьи №{index}")
    worker = BackgroundWorker(
        settings,
        HashEmbedder(8),
        FixedSummarizer("Фикс-суммари."),
        RecordingDedup(candidates=[(1, 0.94)]),
        judge=judge,
    )
    assert worker.process_pending() == 4  # довекторизация создаёт dedup-работы
    with session(settings) as conn, transaction(conn):
        conn.execute("DELETE FROM worker_jobs WHERE note_id = 1")
    assert len(worker._pending_jobs("judge", "dedup", 10)) == 3
    return worker


def _job_rows(settings) -> list[dict]:
    with session(settings) as conn:
        rows = conn.execute(
            "SELECT id, note_id, status, attempts, attention_at FROM worker_jobs "
            "WHERE slot = 'judge' AND kind = 'dedup' ORDER BY id"
        ).fetchall()
    return [dict(row) for row in rows]


@pytest.mark.asyncio
async def test_transient_outage_waits_then_drains(settings, monkeypatch) -> None:
    """Сценарий 1–3: UNAVAILABLE → очередь ждёт; ожил → выгребается до нуля."""
    _freeze(monkeypatch)
    judge = FlakyJudge(fail_outcome=SlotOutcome.UNAVAILABLE)
    worker = _worker_with_three_judge_jobs(settings, judge)

    # --- фаза «недоступны»: транзиентный отказ, разрыв не открывается -------
    assert await worker._process_judge() == 0  # ни одно задание не обработано
    assert judge.judge_calls == 3  # судья опрошен по каждой паре и отказал
    rows = _job_rows(settings)
    assert len(rows) == 3
    assert all(row["status"] == "pending" for row in rows)  # задания живы
    assert all(row["attempts"] == 0 for row in rows)  # транзиентный счёт не растит
    assert all(row["attention_at"] is None for row in rows)  # внимания нет
    assert worker._breakers["judge"].is_paused() is False  # разрыв НЕ открыт
    assert worker._breakers["judge"].probe_due() is False  # и стука не требуется
    assert judge.probe_calls == 0
    # Драйвер петли: очередь для петли НЕ пуста — петля вернётся за работой.
    assert worker._judge_queue_empty() is False

    # --- наблюдаемость в фазе ожидания: очередь честна ----------------------
    stat = worker.queues_health()["judge"]
    assert stat["pending"] == 3  # совпадает с фактом по строкам worker_jobs
    assert isinstance(stat["oldest_pending_sec"], int)  # возраст старейшего — число
    assert stat["waiting"] == 0
    assert stat["attention"] == 0

    # --- фаза «доступны»: слот отвечает — прогон выгребает всё --------------
    judge.fail_outcome = None
    assert await worker._process_judge() == 3
    rows = _job_rows(settings)
    assert all(row["status"] == "done" for row in rows)  # ни одно не потеряно
    assert not worker._pending_jobs("judge", "dedup", 10)
    stat = worker.queues_health()["judge"]
    assert stat["pending"] == 0
    assert stat["oldest_pending_sec"] is None  # очередь пуста — «старейшего» нет
    assert stat["waiting"] == 0
    assert stat["attention"] == 0


@pytest.mark.asyncio
async def test_auth_outage_breaker_waits_then_probe_drains(
    settings, monkeypatch
) -> None:
    """Сценарий 4: AUTH-отказ — закрытый гейт и waiting; стук → выгребание."""
    clock = _freeze(monkeypatch)
    judge = FlakyJudge(fail_outcome=SlotOutcome.AUTH)
    worker = _worker_with_three_judge_jobs(settings, judge)

    # --- отказ доступа: разрыв открывается сразу, партия не берётся ---------
    assert await worker._process_judge() == 0
    assert worker._breakers["judge"].is_paused() is True
    rows = _job_rows(settings)
    assert all(row["status"] == "pending" for row in rows)
    assert all(row["attempts"] == 0 for row in rows)  # auth счёт пары не растит
    assert all(row["attention_at"] is None for row in rows)  # это не внимание

    # --- закрытый гейт: партия не берётся, вызовы модели не жгутся ----------
    calls_before = judge.judge_calls
    assert await worker._process_judge() == 0
    assert judge.judge_calls == calls_before  # модель не вызывалась
    assert judge.probe_calls == 0  # стук ещё не наступил
    # Очередь для петли «пуста» (иначе back-off-петля крутилась бы вхолостую).
    assert worker._judge_queue_empty() is True
    # Наблюдаемость: работа не потеряна — видна как waiting (не attention).
    stat = worker.queues_health()["judge"]
    assert stat["pending"] == 0
    assert stat["waiting"] == 3
    assert stat["attention"] == 0
    # Оговорка №1 отчёта T2-no39: в состоянии разрыва возраст ожидания НЕ
    # виден — `queue_snapshot` гасит `oldest_pending_sec` при pending=0
    # (docstring `_gated_queue_stat` обещает «сохраняются»). Фиксируем
    # фактическое поведение: null при непустом waiting.
    assert stat["oldest_pending_sec"] is None
    assert worker.slots_health()["judge"]["state"] == "paused"

    # --- пауза истекла: успешный стук снимает разрыв — выгребается до нуля --
    clock["t"] += settings.slot_pause_sec + 1
    judge.fail_outcome = None
    assert await worker._process_judge() == 3
    assert judge.probe_calls == 1  # один дешёвый стук перед партией
    assert worker._breakers["judge"].is_paused() is False
    assert worker.slots_health()["judge"]["state"] == "closed"
    assert all(row["status"] == "done" for row in _job_rows(settings))
    stat = worker.queues_health()["judge"]
    assert stat["pending"] == 0
    assert stat["waiting"] == 0
    assert stat["oldest_pending_sec"] is None
