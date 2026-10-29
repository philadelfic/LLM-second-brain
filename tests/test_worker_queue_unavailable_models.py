"""Очередь при недоступных моделях → доступны (Backlog №39, релиз 3.3.0).

Сквозной сценарий по judge-очереди (N=5 заданий, батч петли
`EMBEDDING_BATCH_SIZE=2`): пока слот судьи недоступен, очередь ЖДЁТ — задания
остаются `pending`, не теряются и не замораживаются, `/health.queues`
показывает их честно; когда слот оживает — очередь выгребается до нуля, причём
дренаж 5 заданий занимает ≥3 итерации петли (2+2+1 по батчу), а контрольный
прогон после дренажа работы не находит. Два варианта недоступности:

* **транзиентный отказ** (`UNAVAILABLE` — сеть/5xx/429, живой путь
  `classify_llm_error`): разрыв слота НЕ открывается
  (`SlotBreaker._TRANSIENT_OUTCOMES`), партия снимается первым же отказом
  (LSBN-T-28, `_BATCH_ABORT_OUTCOMES`: «висящий» провайдер не жгут вызовами
  по хвосту партии), работа остаётся pending, счёт пары (`attempts`) и
  внимание (`attention_at`) не растут — ждать модели-недоступность нормально
  (arch §6.2/§13.2);
* **разрыв слота** (`AUTH` — 401/403): разрыв открывается сразу, закрытые
  ворота партию не берут (вызовы модели не жгутся), работа видна в
  `waiting`; после истечения паузы один успешный `probe()` снимает разрыв —
  очередь выгребается (arch §6.2, §13.2).

Числа наблюдаемости (`queues_health`/`queue_stat`) сверяются с фактом по
строкам `worker_jobs` — «числа совпадают с фактом» из приёмки брифа 3.3.0.

Прогон — обёрткой каркаса `jobs._gated_process(spec)` (та же, что в обоих
вызовах `run_loop`; с 3.4.1, LSBN-T-29, ворота слота — в каркасе, не в теле
джобы), без снов: асинхронное ожидание `run_loop` (back-off, `queue_waiting`)
покрыто соседними тестами (`test_jobs.py`, `test_worker_queue_waiting_event.py`),
здесь важен контракт очереди, а не форма расписания; драйвер петли
`_judge_queue_empty` проверяется напрямую. Фаза «недоступны» транзиентного
сценария — один прогон `process_judge_pending(5)` с явным лимитом:
транзиентный отказ не двигает очередь (работы остаются `pending`, вычитка по
порядку id), батч-прогоны брали бы всё те же две первые пары; с LSBN-T-28
отказ судьи снимает партию на первой же паре (`judge_calls == 1`). Сети нет:
судья — тестовый двойник с отказами `JudgeError` нужного исхода, векторизация —
`HashEmbedder`.

LSBN-T-30 (3.4.1) добавляет сюда embedding-джобу — та же тема «недоступные
модели + непустая очередь»: после введения перепроверки очереди
(`_embedding_queue_empty`) петля даёт на интервал ровно ДВА прогона (обычный
и один немедленный lost-wakeup, пул 6) и дальше спит с выросшим back-off —
холостого шторма нет (arch §3.2: отказ — не прогресс, no39-05).
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest
from fakes import FailingEmbedder, FixedSummarizer, HashEmbedder, RecordingDedup, fake_meta

from app.config import get_settings
from app.services import jobs
from app.services.embedding import EmbeddingError
from app.services.judge import JudgeError, JudgeResult
from app.services.notes import NoteService
from app.services.slot_outcome import SlotOutcome
from app.services.worker import BackgroundWorker, build_embedding_job, build_judge_job
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


def _backdate_judge_jobs(settings, seconds_ago: int) -> None:
    """Удревнить judge-работы (паттерн `_backdate` из test_worker_queues).

    Работы создаются только что — их возраст в снимке очереди 0 с; проверке
    «возраст старейшей ждущей работы > 0» (3.3.1, LSBN-S-3) нужна
    детерминированная ненулевая старость.
    """
    moment = (datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    with session(settings) as conn, transaction(conn):
        conn.execute(
            "UPDATE worker_jobs SET created_at = ?, updated_at = ? "
            "WHERE slot = 'judge' AND kind = 'dedup'",
            (moment, moment),
        )


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
    spec = build_judge_job(worker, settings)

    # --- фаза «недоступны»: транзиентный отказ, разрыв не открывается -------
    # Один прогон с явным лимитом (см. докстринг модуля): с LSBN-T-28 первый
    # отказ судьи снимает партию (`break`, wait-класс) — опрос только до
    # первой пары, очередь не двигается.
    assert worker.process_judge_pending(5) == 0  # ни одно задание не обработано
    assert judge.judge_calls == 1  # партия снята первой же отказавшей парой
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
    assert await jobs._gated_process(spec) == 2
    rows = _job_rows(settings)
    assert sum(row["status"] == "done" for row in rows) == 2
    assert sum(row["status"] == "pending" for row in rows) == 3
    # Итерация 2: следующие 2 (заметки 4–5).
    assert await jobs._gated_process(spec) == 2
    rows = _job_rows(settings)
    assert sum(row["status"] == "done" for row in rows) == 4
    assert sum(row["status"] == "pending" for row in rows) == 1
    # Итерация 3: хвост (заметка 6) — дренаж занял ТРИ прогона петли (2+2+1).
    assert await jobs._gated_process(spec) == 1
    rows = _job_rows(settings)
    assert all(row["status"] == "done" for row in rows)  # ни одно не потеряно
    # Контрольный прогон: очередь пуста — busy-loop-остатка нет.
    assert await jobs._gated_process(spec) == 0
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
    spec = build_judge_job(worker, settings)
    _backdate_judge_jobs(settings, 120)  # старейшая работа — 120 с (детерминизм возраста)

    # --- отказ доступа: разрыв открывается сразу, работа не двигается -------
    # Первый прогон ещё при живых воротах (разрыв откроют ИСХОДЫ отказов
    # внутри прогона) — партия обработанных не даёт, работы остаются pending.
    assert await jobs._gated_process(spec) == 0
    assert worker._breakers["judge"].is_paused() is True
    rows = _job_rows(settings)
    assert len(rows) == 5
    assert all(row["status"] == "pending" for row in rows)
    assert all(row["attempts"] == 0 for row in rows)  # auth счёт пары не растит
    assert all(row["attention_at"] is None for row in rows)  # это не внимание

    # --- закрытый гейт: партия не берётся, вызовы модели не жгутся ----------
    calls_before = judge.judge_calls
    assert await jobs._gated_process(spec) == 0
    assert judge.judge_calls == calls_before  # модель не вызывалась
    assert judge.probe_calls == 0  # стук ещё не наступил
    # Очередь для петли «пуста» (иначе back-off-петля крутилась бы вхолостую).
    assert worker._judge_queue_empty() is True
    # Наблюдаемость: работа не потеряна — видна как waiting (не attention).
    stat = worker.queues_health()["judge"]
    assert stat["pending"] == 0
    assert stat["waiting"] == 5
    assert stat["attention"] == 0
    # Возраст ожидания в разрыве слота ВИДЕН (3.3.1, LSBN-S-3): оговорка №1
    # отчёта T2-no39 («в состоянии разрыва возраст НЕ виден») снята релизом
    # 3.3.1 — `queue_snapshot` гасит `oldest_pending_sec` только при пустых
    # pending+waiting, а работа в разрыве живёт в waiting (возраст тот же).
    assert isinstance(stat["oldest_pending_sec"], int)
    assert stat["oldest_pending_sec"] > 0
    assert worker.slots_health()["judge"]["state"] == "paused"

    # --- пауза истекла: стук снимает разрыв — дренаж партиями по батчу 2 ----
    clock["t"] += settings.slot_pause_sec + 1
    judge.fail_outcome = None
    # Итерация 1: один дешёвый стук в воротах, затем первые 2 работы.
    assert await jobs._gated_process(spec) == 2
    assert judge.probe_calls == 1  # стук — один, перед первой партией
    assert worker._breakers["judge"].is_paused() is False
    rows = _job_rows(settings)
    assert sum(row["status"] == "done" for row in rows) == 2
    assert sum(row["status"] == "pending" for row in rows) == 3
    # Итерация 2: следующие 2 (заметки 4–5).
    assert await jobs._gated_process(spec) == 2
    rows = _job_rows(settings)
    assert sum(row["status"] == "done" for row in rows) == 4
    assert sum(row["status"] == "pending" for row in rows) == 1
    # Итерация 3: хвост (заметка 6) — дренаж занял ТРИ прогона петли (2+2+1).
    assert await jobs._gated_process(spec) == 1
    assert worker.slots_health()["judge"]["state"] == "closed"
    assert all(row["status"] == "done" for row in _job_rows(settings))
    # Контрольный прогон: очередь пуста — busy-loop-остатка нет.
    assert await jobs._gated_process(spec) == 0
    assert judge.probe_calls == 1  # повторных стуков дренаж не требует
    stat = worker.queues_health()["judge"]
    assert stat["pending"] == 0
    assert stat["waiting"] == 0
    assert stat["oldest_pending_sec"] is None


class UnavailableEmbedder:
    """Кодировщик-фейк: `EmbeddingError` с транзиентным исходом UNAVAILABLE.

    Транзиентный отказ разрыв слота embedding НЕ открывает
    (`SlotBreaker._TRANSIENT_OUTCOMES`) — ворота партию берут, модель честно
    вызывается, а ждать недоступность — работа back-off петли (arch §13.2).
    `calls` — размеры запрошенных батчей: тест считает вызовы модели и
    проверяет, что их не больше двух на интервал (LSBN-T-30).
    """

    def __init__(self) -> None:
        self.calls: list[int] = []

    def embed(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(len(texts))
        raise EmbeddingError(
            "embedding slot unavailable", outcome=SlotOutcome.UNAVAILABLE
        )

    def close(self) -> None:  # интерфейс-совместимость с EmbeddingService
        return None


@pytest.mark.asyncio
async def test_embedding_outage_no_idle_storm(settings, monkeypatch) -> None:
    """LSBN-T-30, embedding: недоступные модели + непустая очередь — шторма нет.

    Перепроверка очереди (`_embedding_queue_empty`) закрывает lost wakeup,
    но не превращает недоступность в холостой шторм: на интервал — ровно два
    прогона (обычный + один немедленный lost-wakeup), затем петля спит
    с выросшим back-off и модель не трогает до истечения интервала (arch
    §3.2: работа, которую петля взять не может, — не прогресс, no39-05).
    Заметки сеются прямым SQL (прецедент `_seed_pending`) без чанков —
    один прогон ровно один вызов модели, арифметика вызовов прозрачна.
    """
    with session(settings) as conn, transaction(conn):
        for index in range(1, 4):
            conn.execute(
                "INSERT INTO notes (id, title, text, namespace, vector_status) "
                "VALUES (?, ?, ?, 'default', 'pending')",
                (index, f"Заметка №{index}", f"текст заметки вектора №{index}"),
            )
    embedder = UnavailableEmbedder()
    worker = BackgroundWorker(settings, embedder)
    spec = build_embedding_job(worker, settings)

    # --- фаза «недоступны»: транзиентный отказ, разрыв не открывается -------
    assert await jobs._gated_process(spec) == 0  # один прогон вернул 0
    assert embedder.calls == [2]  # модель позвана один раз (батч 2 из 3)
    assert worker._breakers["embedding"].is_paused() is False  # разрыва нет
    assert worker._embedding_queue_empty() is False  # берущаяся работа есть
    # Очередь честна: ни одна заметка не потеряна и не «обработана».
    stat = worker.queues_health()["vector"]
    assert stat["pending"] == 3
    assert stat["waiting"] == 0

    # --- петля: два прогона на интервал, дальше сон с выросшим back-off -----
    # Реальный back-off тест не ждёт: `asyncio.sleep` — мгновенный шпион
    # с записью запрошенных пауз (паттерн `patch_sleep` из test_jobs).
    real_sleep = asyncio.sleep
    delays: list[float] = []

    async def spy_sleep(delay: float, *args: object, **kwargs: object) -> None:
        delays.append(delay)
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", spy_sleep)
    await jobs.run_loop(
        spec, lambda: len(delays) >= 2, jobs.BackoffState(spec.interval_sec)
    )
    # Две итерации по два прогона, по одному вызову embed_texts на прогон,
    # плюс прогон фазы «недоступны» выше: вызовов модели ровно 5, а не
    # десятки — повторных вызовов до истечения интервала нет.
    assert embedder.calls == [2, 2, 2, 2, 2]
    assert delays == [30.0, 60.0]  # интервал вырос: петля спит, а не крутится
    assert worker._embedding_event.is_set() is False  # отказ сигнала не ставит
