"""Наблюдаемость очередей и идемпотентная постановка заданий (lsb-0014-03).

FR-2.2: `/health.queues` — по каждой очереди число pending и возраст
старейшего (`null` — очередь пуста). Источники у очередей разные (pending-
статусы заметок, сумма по `ALL_AREAS`, `worker_jobs`), поэтому очередь
описывает сама джоба (`JobSpec.queue_stat`), а `queues_health` собирает
объект по реестру — только SQL, без обращений к моделям. FR-2.5:
`_ensure_job` ставит задание идемпотентно (pending с теми же
slot/kind/note_id не дублируется), существующий `_create_job` (judge/merge)
работает как раньше.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fakes import (
    FailingEmbedder,
    FixedSummarizer,
    HashEmbedder,
    clear_seeded_skills,
)

from app.config import get_settings
from app.services.notes import NoteService
from app.services.worker import BackgroundWorker
from app.storage.db import init_db, session, transaction

DIM = 8


@pytest.fixture
def settings(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """БД без сид-записей: очереди наполняет только сам тест."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "notes.db"))
    monkeypatch.setenv("EMBEDDING_DIM", str(DIM))
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    clear_seeded_skills(settings)
    return settings


def make_worker(settings, embedding=None) -> BackgroundWorker:
    """Воркер с суммаризатором: в реестре все пять джоб (summary включена)."""
    return BackgroundWorker(
        settings,
        embedding if embedding is not None else HashEmbedder(DIM),
        FixedSummarizer("Фикс-суммари."),
    )


def _ts(seconds_ago: int) -> str:
    """ISO-8601 UTC-метка «N секунд назад» — как пишут её таблицы."""
    moment = datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def _backdate(settings, table: str, row_id: int, seconds_ago: int) -> None:
    """Удревнить строку таблицы: возраст старейшего задания проверяем явно."""
    with session(settings) as conn, transaction(conn):
        conn.execute(
            f"UPDATE {table} SET updated_at = ?, created_at = ? WHERE id = ?",
            (_ts(seconds_ago), _ts(seconds_ago), row_id),
        )


def _seed_pending_areas() -> None:
    """По одной pending-записи в каждой области (как их пишут сервисы областей)."""
    with session(get_settings()) as conn, transaction(conn):
        conn.execute(
            "INSERT INTO skills (name, description, steps, text) "
            "VALUES ('Деплой релиза', 'как катить релиз', 'шаги', 'текст')"
        )
        conn.execute(
            "INSERT INTO terms (term, term_norm, context, context_norm, definition) "
            "VALUES ('ГЗ', 'гз', 'студенты МГУ', 'студенты мгу', 'госэкзамен')"
        )
        conn.execute(
            "INSERT INTO user_facts (name, body) VALUES ('Часовой пояс', 'Москва')"
        )


def _assert_stat(actual: dict, expected: dict) -> None:
    """Снимок очереди = прямому SQL: pending точно, возраст — до секунды точности.

    Снимок и прямой SQL берутся в разные моменты, поэтому возраст сверяется с
    допуском в 1 секунду (граница секунды).
    """
    assert actual["pending"] == expected["pending"]
    if expected["oldest_pending_sec"] is None:
        assert actual["oldest_pending_sec"] is None
    else:
        assert actual["oldest_pending_sec"] == pytest.approx(
            expected["oldest_pending_sec"], abs=1
        )


def _sql_note_stat(settings, status_column: str) -> dict:
    """Прямой SQL по очереди заметок — эталон для снимка."""
    with session(settings) as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS pending, "
            "MAX(CAST(strftime('%s','now') AS INTEGER) - "
            "CAST(strftime('%s', updated_at) AS INTEGER)) AS oldest_pending_sec "
            "FROM notes WHERE deleted_at IS NULL "
            f"AND {status_column} = 'pending'"
        ).fetchone()
    pending = int(row["pending"])
    oldest = row["oldest_pending_sec"]
    return {
        "pending": pending,
        "oldest_pending_sec": None if pending == 0 or oldest is None else int(oldest),
    }


def _sql_judge_stat(settings) -> dict:
    """Прямой SQL по judge-очереди в worker_jobs — эталон для снимка."""
    with session(settings) as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS pending, "
            "MAX(CAST(strftime('%s','now') AS INTEGER) - "
            "CAST(strftime('%s', created_at) AS INTEGER)) AS oldest_pending_sec "
            "FROM worker_jobs WHERE slot = 'judge' AND status = 'pending'"
        ).fetchone()
    pending = int(row["pending"])
    oldest = row["oldest_pending_sec"]
    return {
        "pending": pending,
        "oldest_pending_sec": None if pending == 0 or oldest is None else int(oldest),
    }


def _sql_links_stat(settings) -> dict:
    """Прямой SQL по очереди расчёта связей (маркер links_at) — эталон."""
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


def _sql_areas_stat(settings) -> dict:
    """Прямой SQL по областям (сумма pending, максимум возраста) — эталон."""
    pending = 0
    oldest: int | None = None
    with session(settings) as conn:
        for table in ("skills", "terms", "user_facts"):
            row = conn.execute(
                "SELECT COUNT(*) AS pending, "
                "MAX(CAST(strftime('%s','now') AS INTEGER) - "
                "CAST(strftime('%s', updated_at) AS INTEGER)) AS oldest_pending_sec "
                f"FROM {table} WHERE vector_status = 'pending' "
                "AND deleted_at IS NULL"
            ).fetchone()
            pending += int(row["pending"])
            value = row["oldest_pending_sec"]
            if value is not None:
                oldest = int(value) if oldest is None else max(oldest, int(value))
    return {
        "pending": pending,
        "oldest_pending_sec": None if pending == 0 else oldest,
    }


def _job_rows(settings) -> list:
    """Строки worker_jobs в порядке id (для проверок идемпотентности)."""
    with session(settings) as conn:
        return conn.execute(
            "SELECT id, slot, kind, note_id, payload, status "
            "FROM worker_jobs ORDER BY id"
        ).fetchall()


# --- снимки очередей (/health.queues, FR-2.2) --------------------------------


def test_queues_health_empty_queues_are_null(settings) -> None:
    """Пустые очереди: pending 0, возраст старейшего — null.

    В объект попадают все наблюдаемые очереди (vector/summary/judge/areas);
    `expiration` очереди не имеет и не наблюдаема.
    """
    queues = make_worker(settings).queues_health()
    assert set(queues) == {"vector", "summary", "judge", "areas", "links"}
    for stat in queues.values():
        assert stat == {"pending": 0, "oldest_pending_sec": None}


def test_queues_health_matches_direct_sql(settings) -> None:
    """Снимок каждой очереди совпадает с прямым SQL по тестовой БД."""
    notes = NoteService(settings, FailingEmbedder())
    notes.save("первая заметка очереди вектора")
    notes.save("вторая заметка очереди вектора")
    _seed_pending_areas()
    worker = make_worker(settings)
    worker._create_job("judge", "dedup", 1)
    worker._create_job("judge", "dedup", 2)
    # Старейшие задания: заметка id=1, область skills id=1, judge-работа id=1.
    _backdate(settings, "notes", 1, 120)
    _backdate(settings, "skills", 1, 45)
    _backdate(settings, "worker_jobs", 1, 30)

    queues = worker.queues_health()

    assert set(queues) == {"vector", "summary", "judge", "areas", "links"}
    _assert_stat(queues["vector"], _sql_note_stat(settings, "vector_status"))
    _assert_stat(queues["summary"], _sql_note_stat(settings, "summary_status"))
    _assert_stat(queues["judge"], _sql_judge_stat(settings))
    _assert_stat(queues["areas"], _sql_areas_stat(settings))
    _assert_stat(queues["links"], _sql_links_stat(settings))
    # Явные ожидания — снимок считает именно своё состояние.
    assert queues["vector"]["pending"] == 2
    assert queues["summary"]["pending"] == 2
    assert queues["judge"]["pending"] == 2
    assert queues["areas"]["pending"] == 3
    assert queues["vector"]["oldest_pending_sec"] >= 119
    assert queues["areas"]["oldest_pending_sec"] >= 44
    assert queues["judge"]["oldest_pending_sec"] >= 29


def test_pending_grows_and_oldest_increases_when_models_unavailable(
    settings,
) -> None:
    """Модели недоступны: pending растёт, возраст старейшего увеличивается.

    Легаси-поля `pending_vector`/`pending_summary` дают те же числа, что
    очереди `vector`/`summary` (FR-2.2 + обратная совместимость /health).
    """
    notes = NoteService(settings, FailingEmbedder())
    notes.save("первая отложенная заметка")
    _backdate(settings, "notes", 1, 120)
    worker = make_worker(settings, FailingEmbedder())

    first = worker.queues_health()
    assert first["vector"]["pending"] == 1
    assert first["vector"]["oldest_pending_sec"] >= 119
    # Прогон без моделей — 0 обработанных: задание остаётся pending (не прогресс).
    assert worker.process_pending() == 0

    notes.save("вторая отложенная заметка")
    notes.save("третья отложенная заметка")
    _backdate(settings, "notes", 1, 300)  # старейшее ждёт дольше
    second = worker.queues_health()

    assert second["vector"]["pending"] == 3
    assert second["summary"]["pending"] == 3
    assert second["vector"]["oldest_pending_sec"] > first["vector"]["oldest_pending_sec"]
    legacy = notes.health_counts()
    assert legacy["pending_vector"] == second["vector"]["pending"]
    assert legacy["pending_summary"] == second["summary"]["pending"]


def test_health_returns_queues_with_pending_work(client, token) -> None:
    """/health отдаёт очереди: сохранённая заметка ждёт в vector и summary.

    Модели в тестовом окружении недоступны — заметка остаётся pending, и это
    видно и в легаси-полях, и в снимках очередей (FR-2.2).
    """
    client.post(
        "/notes",
        json={"text": "заметка для наблюдаемости очередей", "title": "Заметка очереди"},
        headers={"Authorization": f"Bearer {token}"},
    )
    body = client.get("/health").json()
    assert body["pending_vector"] == 1
    assert body["pending_summary"] == 1
    # Снимки очередей дают те же числа, что легаси-поля (плюс видно, как долго ждёт).
    assert body["queues"]["vector"]["pending"] == body["pending_vector"]
    assert body["queues"]["summary"]["pending"] == body["pending_summary"]
    assert body["queues"]["vector"]["oldest_pending_sec"] is not None
    assert body["queues"]["summary"]["oldest_pending_sec"] is not None


# --- идемпотентная постановка задания (_ensure_job, FR-2.5) -------------------


def test_ensure_job_does_not_duplicate_pending(settings) -> None:
    """Два вызова на одну заметку → одна pending-строка (ключ slot/kind/note_id)."""
    worker = make_worker(settings)
    worker._ensure_job("summary", "reclaim", 7, payload='{"from": 1}')
    worker._ensure_job("summary", "reclaim", 7)
    worker._ensure_job("summary", "reclaim", 8)  # другая заметка — своё задание
    worker._ensure_job("judge", "reclaim", 7)  # другой слот — своё задание

    rows = _job_rows(settings)
    assert [(row["slot"], row["kind"], row["note_id"]) for row in rows] == [
        ("summary", "reclaim", 7),
        ("summary", "reclaim", 8),
        ("judge", "reclaim", 7),
    ]
    assert rows[0]["payload"] == '{"from": 1}'  # существующее задание не перезаписано
    assert {row["status"] for row in rows} == {"pending"}


def test_ensure_job_creates_new_after_done(settings) -> None:
    """После `_mark_job_done` новый вызов создаёт задание заново."""
    worker = make_worker(settings)
    worker._ensure_job("summary", "reclaim", 7)
    worker._mark_job_done(_job_rows(settings)[0]["id"])
    worker._ensure_job("summary", "reclaim", 7)

    assert [row["status"] for row in _job_rows(settings)] == ["done", "pending"]


def test_create_job_still_inserts_each_call(settings) -> None:
    """Регресс дедупа: judge/merge-путь (`_create_job`) не стал идемпотентным."""
    worker = make_worker(settings)
    worker._create_job("judge", "dedup", 1)
    worker._create_job("judge", "dedup", 1)
    assert len(_job_rows(settings)) == 2
