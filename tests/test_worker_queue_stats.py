"""Очереди при недоступных моделях: pending/waiting/attention (№39, 3.2.0).

Архитектура §6.9/§13.9: `pending` — только работа, которую петля может взять
СЕЙЧАС; `waiting` — отложено слотом/готовностью; `attention` — ждёт человека.
При открытом разрыве слота вся текущая работа уезжает в `waiting`
(`pending=0`), а очередь для петли становится пустой — иначе отложенное
задание гонит back-off-петлю в busy-loop (lsbdef-0008). Юнит без сети.

3.3.1 (LSBN-S-3): `oldest_pending_sec` — возраст старейшей работы, ждущей
исполнения (pending ИЛИ waiting): при разрыве слота возраст ожидания виден,
null — только когда обе категории пусты.

3.4.1 (LSBN-T-29): снимки двусоставные — `queue_stat` джобы отдаёт СЫРОЙ
снимок (свои вычисления waiting/attention не тронуты), а перенос
pending → waiting при заблокированном слоте делает каркас: `is_gated` →
`gated_queue_stat` в `queues_health` (и в событии `queue_waiting`). Поэтому
гейт-состояния проверяются здесь через `worker.queues_health()`.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fakes import FailingEmbedder, FixedSummarizer, HashEmbedder, clear_seeded_skills

from app.config import get_settings
from app.services.notes import NoteService
from app.services.slot_outcome import SlotOutcome
from app.services.worker import BackgroundWorker
from app.storage.db import init_db, session, transaction

DIM = 8


@pytest.fixture
def settings(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "notes.db"))
    monkeypatch.setenv("EMBEDDING_DIM", str(DIM))
    get_settings.cache_clear()
    resolved = get_settings()
    init_db(resolved)
    return resolved


def _ts(seconds_ago: int) -> str:
    moment = datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def _backdate(settings, table: str, row_id: int, seconds_ago: int) -> None:
    """Удревнить строку таблицы (паттерн test_worker_queues): детерминизм возраста."""
    with session(settings) as conn, transaction(conn):
        conn.execute(
            f"UPDATE {table} SET updated_at = ?, created_at = ? WHERE id = ?",
            (_ts(seconds_ago), _ts(seconds_ago), row_id),
        )


def make_worker(settings, embedding=None) -> BackgroundWorker:
    return BackgroundWorker(
        settings,
        embedding if embedding is not None else HashEmbedder(DIM),
        FixedSummarizer("Фикс-суммари."),
    )


def _judge_job_ids(settings) -> list[int]:
    with session(settings) as conn:
        return [int(row["id"]) for row in conn.execute(
            "SELECT id FROM worker_jobs WHERE slot = 'judge' ORDER BY id"
        ).fetchall()]


def _freeze(monkeypatch, start: float = 1000.0) -> dict:
    clock = {"t": start}
    monkeypatch.setattr("app.services.slot_breaker.time.time", lambda: clock["t"])
    return clock


def test_judge_waiting_and_attention(settings) -> None:
    """pending-работа без внимания — в pending; с вниманием — в attention/waiting."""
    worker = make_worker(settings)
    worker._create_job("judge", "dedup", 1)
    worker._create_job("judge", "dedup", 2)
    attention_id, pending_id = _judge_job_ids(settings)
    with session(settings) as conn, transaction(conn):
        conn.execute(
            "UPDATE worker_jobs SET attention_at = ? WHERE id = ?",
            (_ts(0), attention_id),
        )

    stat = worker.queues_health()["judge"]
    assert stat["pending"] == 1          # берётся только работа без внимания
    assert stat["attention"] == 1        # замороженная ждёт человека
    assert stat["waiting"] == 1          # у судьи отложено именно замороженное
    assert stat["oldest_pending_sec"] is not None
    assert pending_id != attention_id


def test_queue_empty_ignores_attention(settings) -> None:
    """Единственное attention-задание: очередь судьи для петли пуста (нет busy-loop)."""
    worker = make_worker(settings)
    worker._create_job("judge", "dedup", 1)
    (job_id,) = _judge_job_ids(settings)
    with session(settings) as conn, transaction(conn):
        conn.execute(
            "UPDATE worker_jobs SET attention_at = ? WHERE id = ?",
            (_ts(0), job_id),
        )

    assert worker._judge_queue_stat()["pending"] == 0
    assert worker._judge_queue_empty() is True


def test_paused_slot_moves_pending_to_waiting(settings, monkeypatch) -> None:
    """Разрыв открыт: сырой снимок как есть, каркас едет pending → waiting."""
    _freeze(monkeypatch)
    worker = make_worker(settings)
    worker._create_job("judge", "dedup", 1)
    worker._create_job("judge", "dedup", 2)
    assert worker._judge_queue_stat()["pending"] == 2  # сырое — джоба не гейтит

    worker._breakers["judge"].note_failure(SlotOutcome.AUTH)  # разрыв открыт
    assert worker._judge_queue_stat()["pending"] == 2  # сырое не менялось
    stat = worker.queues_health()["judge"]  # каркас применяет gated_queue_stat
    assert stat["pending"] == 0
    assert stat["waiting"] == 2  # вся текущая работа — в ожидании слота
    assert worker._judge_queue_empty() is True  # петля не крутится вхолостую


def test_vector_queue_waits_on_embedding_break(settings, monkeypatch) -> None:
    """Разрыв embedding: векторная очередь ждёт слот (pending=0, waiting=1)."""
    _freeze(monkeypatch)
    NoteService(settings, FailingEmbedder()).save("заметка ждёт кодировщик")
    worker = make_worker(settings, FailingEmbedder())

    assert worker.queues_health()["vector"]["pending"] == 1
    worker._breakers["embedding"].note_failure(SlotOutcome.AUTH)
    stat = worker.queues_health()["vector"]
    assert stat["pending"] == 0
    assert stat["waiting"] == 1


# --- 3.3.1 (LSBN-S-3): возраст ожидания виден и при разрыве слота -------------


def test_vector_age_survives_embedding_break(settings, monkeypatch) -> None:
    """Приёмка LSBN-S-3 п.1: в разрыве embedding возраст старейшей виден.

    При открытом разрыве и непустом waiting `oldest_pending_sec` — int ≥
    удревнённого возраста (допуск ±1 с, как `_assert_stat`), а не null; после
    закрытия разрыва работа возвращается в pending с возрастом того же порядка.
    """
    _freeze(monkeypatch)
    notes = NoteService(settings, FailingEmbedder())
    notes.save("первая заметка в разрыве слота")
    notes.save("вторая заметка в разрыве слота")
    _backdate(settings, "notes", 1, 120)  # старейшая ждёт 120 с
    worker = make_worker(settings, FailingEmbedder())

    worker._breakers["embedding"].note_failure(SlotOutcome.AUTH)  # разрыв открыт
    stat = worker.queues_health()["vector"]
    assert stat["pending"] == 0
    assert stat["waiting"] == 2
    assert isinstance(stat["oldest_pending_sec"], int)
    assert stat["oldest_pending_sec"] >= 119

    worker._breakers["embedding"].note_success()  # разрыв закрыт
    stat = worker.queues_health()["vector"]
    assert stat["pending"] == 2
    assert stat["waiting"] == 0
    assert isinstance(stat["oldest_pending_sec"], int)
    assert stat["oldest_pending_sec"] > 0  # работа та же — порядок возраста тот же


def test_queues_health_empty_db_age_is_null(settings) -> None:
    """Приёмка LSBN-S-3 п.2 (регресс): пустая очередь — возраст null.

    На чистой БД (сид-навык снят — он pending области areas) все очереди
    нулевые, `oldest_pending_sec is None`: null ⇔ нет работы ни в pending,
    ни в waiting (эталон — `test_queues_health_empty_queues_are_null`).
    Очередь `migration` (M2 3.4.0) — своя форма снимка (2 ключа, без
    waiting/attention), возраст тоже null.
    """
    clear_seeded_skills(settings)
    queues = make_worker(settings).queues_health()
    assert set(queues) == {
        "vector", "summary", "judge", "areas", "links", "nodes", "migration",
    }
    for name, stat in queues.items():
        assert stat == (
            {"pending": 0, "oldest_pending_sec": None}
            if name == "migration"
            else {
                "pending": 0, "oldest_pending_sec": None,
                "waiting": 0, "attention": 0,
            }
        )


def test_judge_attention_only_age_visible(settings, monkeypatch) -> None:
    """Приёмка LSBN-S-3 п.4: attention-only при живом слоте — возраст виден.

    Возраст судьи считается по ВСЕМ pending джобы (attention-работа тоже ждёт
    исполнения): без снятия attention-фильтра состояние `pending=0,
    waiting>0` при живом слоте давало бы null вопреки определению. Разрыв
    judge возраст сохраняет сквозь гейт (та же работа в waiting).
    """
    _freeze(monkeypatch)
    worker = make_worker(settings)
    worker._create_job("judge", "dedup", 1)
    (job_id,) = _judge_job_ids(settings)
    _backdate(settings, "worker_jobs", job_id, 90)  # работа создана 90 с назад
    with session(settings) as conn, transaction(conn):
        conn.execute(
            "UPDATE worker_jobs SET attention_at = ? WHERE id = ?",
            (_ts(0), job_id),
        )

    stat = worker.queues_health()["judge"]
    assert stat["pending"] == 0
    assert stat["waiting"] == 1
    assert stat["attention"] == 1
    assert isinstance(stat["oldest_pending_sec"], int)
    assert stat["oldest_pending_sec"] >= 89

    worker._breakers["judge"].note_failure(SlotOutcome.AUTH)  # разрыв judge
    stat = worker.queues_health()["judge"]
    assert stat["pending"] == 0
    assert stat["waiting"] == 1
    assert stat["attention"] == 1
    assert isinstance(stat["oldest_pending_sec"], int)
    assert stat["oldest_pending_sec"] >= 89  # возраст тот же — работа та же
