"""Очередь при недоступных моделях → доступны (Backlog №39, релиз 3.3.0).

Сквозной сценарий по judge-очереди (N=5 заданий, батч петли
`EMBEDDING_BATCH_SIZE=2`): пока слот судьи недоступен, очередь ЖДЁТ — задания
остаются `pending`, не теряются и не замораживаются, `/health.queues`
показывает их честно; когда слот оживает — очередь выгребается до нуля, причём
дренаж 5 заданий занимает ≥3 итерации петли (2+2+1 по батчу), а контрольный
прогон после дренажа работы не находит. Два варианта недоступности:

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
`_judge_queue_empty` проверяется напрямую. Фаза «недоступны» транзиентного
сценария опрашивает всю очередь одним прогоном `process_judge_pending(5)` —
с явным лимитом: транзиентный отказ не двигает очередь (работы остаются
`pending`, вычитка по порядку id), батч-прогоны брали бы всё те же две первые
пары, а контракт фазы — отказ судьи по каждой из 5 пар. Сети нет: судья —
тестовый двойник с отказами `JudgeError` нужного исхода, векторизация —
`HashEmbedder`.
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
    monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "2")  # батч judge-петли и векторизации
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


def _worker_with_five_judge_jobs(settings, judge: FlakyJudge) -> BackgroundWorker:
    """Шесть векторизованных заметок; 5 judge-работ, каждая идёт в судью.

    Кандидат один — ранняя заметка 1 (`RecordingDedup`), поэтому у заметок
    2–6 работа доходит до вызова судьи, а работа заметки 1 снимается сразу
    (раньше неё кандидатов нет — она и её работа убраны настроечным DELETE).
    Довекторизация режется тем же батчем (`EMBEDDING_BATCH_SIZE=2`), поэтому
    `process_pending` гонится циклом до нуля (3 партии: 2+2+2), а не одним
    прогоном.
    """
    notes = NoteService(settings, FailingEmbedder())
    for index in range(1, 7):
        notes.save(f"заметка очереди недоступного судьи №{index}")
    worker = BackgroundWorker(
        settings,
        HashEmbedder(8),
        FixedSummarizer("Фикс-суммари."),
        RecordingDedup(candidates=[(1, 0.94)]),
        judge=judge,
    )
    vectorized = 0
    while (portion := worker.process_pending()) > 0:  # батч 2: три партии
        vectorized += portion
    assert vectorized == 6  # довекторизация создаёт dedup-работы
    with session(settings) as conn, transaction(conn):
        conn.execute("DELETE FROM worker_jobs WHERE note_id = 1")
    assert len(worker._pending_jobs("judge", "dedup", 10)) == 5
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
    """Сценарии 1–3: UNAVAILABLE → очередь ждёт; ожил → дренаж партиями."""
    _freeze(monkeypatch)
    judge = FlakyJudge(fail_outcome=SlotOutcome.UNAVAILABLE)
    worker = _worker_with_five_judge_jobs(settings, judge)

    # --- фаза «недоступны»: транзиентный отказ, разрыв не открывается -------
    # Вся очередь (5 пар) — одним прогоном с явным лимитом (см. докстринг
    # модуля): батч-прогон при отказе не двигает очередь.
    assert worker.process_judge_pending(5) == 0  # ни одно задание не обработано
    assert judge.judge_calls == 5  # судья опрошен по каждой паре и отказал
    rows = _job_rows(settings)
    assert len(rows) == 5
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
    assert stat["pending"] == 5  # совпадает с фактом по строкам worker_jobs
    assert isinstance(stat["oldest_pending_sec"], int)  # возраст старейшего — число
    assert stat["waiting"] == 0
    assert stat["attention"] == 0

    # --- фаза «доступны»: слот отвечает — дренаж партиями по батчу 2 --------
    judge.fail_outcome = None
    # Итерация 1: первые 2 работы (заметки 2–3).
    assert await worker._process_judge() == 2
    rows = _job_rows(settings)
    assert sum(row["status"] == "done" for row in rows) == 2
    assert sum(row["status"] == "pending" for row in rows) == 3
    # Итерация 2: следующие 2 (заметки 4–5).
    assert await worker._process_judge() == 2
    rows = _job_rows(settings)
    assert sum(row["status"] == "done" for row in rows) == 4
    assert sum(row["status"] == "pending" for row in rows) == 1
    # Итерация 3: хвост (заметка 6) — дренаж занял ТРИ прогона петли (2+2+1).
    assert await worker._process_judge() == 1
    rows = _job_rows(settings)
    assert all(row["status"] == "done" for row in rows)  # ни одно не потеряно
    # Контрольный прогон: очередь пуста — busy-loop-остатка нет.
    assert await worker._process_judge() == 0
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
    worker = _worker_with_five_judge_jobs(settings, judge)

    # --- отказ доступа: разрыв открывается сразу, партия не берётся ---------
    assert await worker._process_judge() == 0
    assert worker._breakers["judge"].is_paused() is True
    rows = _job_rows(settings)
    assert len(rows) == 5
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
    assert stat["waiting"] == 5
    assert stat["attention"] == 0
    # Оговорка №1 отчёта T2-no39: в состоянии разрыва возраст ожидания НЕ
    # виден — `queue_snapshot` гасит `oldest_pending_sec` при pending=0
    # (docstring `_gated_queue_stat` обещает «сохраняются»). Фиксируем
    # фактическое поведение: null при непустом waiting.
    assert stat["oldest_pending_sec"] is None
    assert worker.slots_health()["judge"]["state"] == "paused"

    # --- пауза истекла: стук снимает разрыв — дренаж партиями по батчу 2 ----
    clock["t"] += settings.slot_pause_sec + 1
    judge.fail_outcome = None
    # Итерация 1: один дешёвый стук перед партией, затем первые 2 работы.
    assert await worker._process_judge() == 2
    assert judge.probe_calls == 1  # стук — один, перед первой партией
    assert worker._breakers["judge"].is_paused() is False
    rows = _job_rows(settings)
    assert sum(row["status"] == "done" for row in rows) == 2
    assert sum(row["status"] == "pending" for row in rows) == 3
    # Итерация 2: следующие 2 (заметки 4–5).
    assert await worker._process_judge() == 2
    rows = _job_rows(settings)
    assert sum(row["status"] == "done" for row in rows) == 4
    assert sum(row["status"] == "pending" for row in rows) == 1
    # Итерация 3: хвост (заметка 6) — дренаж занял ТРИ прогона петли (2+2+1).
    assert await worker._process_judge() == 1
    assert worker.slots_health()["judge"]["state"] == "closed"
    assert all(row["status"] == "done" for row in _job_rows(settings))
    # Контрольный прогон: очередь пуста — busy-loop-остатка нет.
    assert await worker._process_judge() == 0
    assert judge.probe_calls == 1  # повторных стуков дренаж не требует
    stat = worker.queues_health()["judge"]
    assert stat["pending"] == 0
    assert stat["waiting"] == 0
    assert stat["oldest_pending_sec"] is None
