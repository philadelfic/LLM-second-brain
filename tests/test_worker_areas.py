"""Петля areas воркера (субстрат 3.0.0): pending записи областей → вектора.

Постановка 00, пул 3: pending записей skills/terms/user → `embed_texts` по
тексту области → upsert в vec0 области → `vector_status='ok'`; отказ
эмбеддера — записи остаются pending и событие `area_embed_failed`; своя
петля со своим back-off в `run()`.
"""

from __future__ import annotations

import asyncio
import contextlib
import time

import pytest
from fakes import FailingEmbedder, HashEmbedder, clear_seeded_skills

from app.config import get_settings
from app.services.areas import SKILLS_AREA, TERMS_AREA, USER_FACTS_AREA
from app.services.worker import BackgroundWorker
from app.storage import area_vectors
from app.storage.db import CREATOR_SKILL_NAME, init_db, session, transaction

DIM = 8


@pytest.fixture
def settings(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """БД области без записей: сид skill-создателя (lsb-0007-04) снят."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "notes.db"))
    monkeypatch.setenv("EMBEDDING_DIM", str(DIM))
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    clear_seeded_skills(settings)
    return settings


@pytest.fixture
def fast(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Как settings, но с PENDING_RETRY_SEC=0 — цикл без пауз."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "notes.db"))
    monkeypatch.setenv("EMBEDDING_DIM", str(DIM))
    monkeypatch.setenv("PENDING_RETRY_SEC", "0")
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    clear_seeded_skills(settings)
    return settings


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


def _statuses() -> dict[str, str]:
    with session(get_settings()) as conn:
        return {
            table: conn.execute(
                f"SELECT vector_status FROM {table} ORDER BY id LIMIT 1"
            ).fetchone()[0]
            for table in ("skills", "terms", "user_facts")
        }


# --- партии ------------------------------------------------------------------


def test_process_pending_areas_vectorizes_all_areas(settings) -> None:
    """Петля закрывает pending на всех трёх областях и пишет их вектора."""
    _seed_pending_areas()
    worker = BackgroundWorker(settings, HashEmbedder(DIM))
    assert worker.process_pending_areas() == 3
    assert _statuses() == {"skills": "ok", "terms": "ok", "user_facts": "ok"}
    with session(settings) as conn:
        for spec, row_id, text in (
            (SKILLS_AREA, 1, "Деплой релиза\nкак катить релиз"),
            (TERMS_AREA, 1, "ГЗ\nстуденты МГУ\nгосэкзамен"),
            (USER_FACTS_AREA, 1, "Часовой пояс\nМосква"),
        ):
            assert area_vectors.get_vector(
                conn, spec.vec_table, spec.vec_id_column, row_id
            ) == pytest.approx(HashEmbedder(DIM).embed(text), abs=1e-6)


def test_process_pending_areas_empty_queue(settings) -> None:
    assert BackgroundWorker(settings, HashEmbedder(DIM)).process_pending_areas() == 0


def test_process_pending_areas_failure_keeps_pending_and_logs(
    settings, caplog
) -> None:
    """Отказ эмбеддера: записи остаются pending, событие area_embed_failed."""
    _seed_pending_areas()
    worker = BackgroundWorker(settings, FailingEmbedder())
    with caplog.at_level("WARNING", logger="app"):
        assert worker.process_pending_areas() == 0
    assert _statuses() == {"skills": "pending", "terms": "pending", "user_facts": "pending"}
    with session(settings) as conn:
        for spec in (SKILLS_AREA, TERMS_AREA, USER_FACTS_AREA):
            assert area_vectors.count(conn, spec.vec_table) == 0
    events = [
        record
        for record in caplog.records
        if getattr(record, "event", None) == "area_embed_failed"
    ]
    assert {record.area for record in events} == {"skills", "terms", "user"}


def test_process_pending_areas_skips_trash(settings) -> None:
    """Trash не до-векторизуется: очередь — только активные записи."""
    with session(settings) as conn, transaction(conn):
        conn.execute(
            "INSERT INTO skills (name, description, steps, text, deleted_at) "
            "VALUES ('Удалённый', 'навык', 'шаги', 'текст', "
            "strftime('%Y-%m-%dT%H:%M:%SZ','now'))"
        )
    worker = BackgroundWorker(settings, HashEmbedder(DIM))
    assert worker.process_pending_areas() == 0
    with session(settings) as conn:
        assert conn.execute(
            "SELECT vector_status FROM skills WHERE id = 1"
        ).fetchone()[0] == "pending"
    assert worker._areas_queue_empty() is True


def test_process_pending_areas_respects_batch_size(settings, monkeypatch) -> None:
    """Батч режется по EMBEDDING_BATCH_SIZE (как notes-очередь)."""
    monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "1")
    get_settings.cache_clear()
    batched = get_settings()
    with session(batched) as conn, transaction(conn):
        for index in range(3):
            conn.execute(
                "INSERT INTO skills (name, description, steps, text) "
                "VALUES (?, 'описание', 'шаги', 'текст')",
                (f"Навык {index}",),
            )
    worker = BackgroundWorker(batched, HashEmbedder(DIM))
    assert worker.process_pending_areas() == 1
    assert worker.process_pending_areas() == 1
    assert worker.process_pending_areas() == 1
    assert worker.process_pending_areas() == 0


class MidFlightAreaEmbedder(HashEmbedder):
    """Кодировщик, правящий запись области в момент кодирования (гонка update)."""

    def __init__(self, dim: int, settings) -> None:
        super().__init__(dim)
        self._settings = settings
        self.mutated = False

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not self.mutated:
            self.mutated = True
            with session(self._settings) as conn, transaction(conn):
                conn.execute(
                    "UPDATE skills SET description = 'правка в полёте' WHERE id = 1"
                )
        return super().embed_texts(texts)


def test_process_pending_areas_race_with_update(settings) -> None:
    """Guard: запись изменилась в полёте — вектор не пишется, статус pending."""
    with session(settings) as conn, transaction(conn):
        conn.execute(
            "INSERT INTO skills (name, description, steps, text) "
            "VALUES ('Навык', 'старое описание', 'шаги', 'текст')"
        )
    worker = BackgroundWorker(settings, MidFlightAreaEmbedder(DIM, settings))
    assert worker.process_pending_areas() == 0
    with session(settings) as conn:
        row = conn.execute(
            "SELECT description, vector_status FROM skills WHERE id = 1"
        ).fetchone()
        assert row["description"] == "правка в полёте"
        assert row["vector_status"] == "pending"
        assert area_vectors.count(conn, SKILLS_AREA.vec_table) == 0


# --- петля (asyncio) ---------------------------------------------------------


def test_areas_interval_starts_at_configured(settings) -> None:
    """Своя петля = свой счётчик back-off (независим от заметок)."""
    worker = BackgroundWorker(settings, HashEmbedder(DIM))
    assert worker.areas_interval == float(settings.pending_retry_sec)


@pytest.mark.asyncio
async def test_areas_loop_closes_pending_in_all_areas(fast) -> None:
    """Петля areas в run(): pending на всех трёх областях → ok, интервал сброшен."""
    _seed_pending_areas()
    worker = BackgroundWorker(fast, HashEmbedder(DIM))
    task = asyncio.create_task(worker.run())
    statuses: dict[str, str] = {}
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        statuses = _statuses()
        if all(status == "ok" for status in statuses.values()):
            break
        await asyncio.sleep(0.01)
    worker.stop()
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    assert statuses == {"skills": "ok", "terms": "ok", "user_facts": "ok"}
    assert worker.areas_interval == float(fast.pending_retry_sec)


def test_seeded_creator_skill_is_vectorized(tmp_path, monkeypatch) -> None:
    """Сид skill-создателя (lsb-0007-04) — обычная pending-запись области.

    Свежая БД несёт навык «Create skills» с `vector_status='pending'`; петля
    `areas` кодирует его наравне с остальными записями (вектор по
    `name + description`) — проверка стыка сида и субстрата.
    """
    monkeypatch.setenv("DB_PATH", str(tmp_path / "notes.db"))
    monkeypatch.setenv("EMBEDDING_DIM", str(DIM))
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)  # сид НЕ снимаем — проверяем его штатную векторизацию
    worker = BackgroundWorker(settings, HashEmbedder(DIM))
    assert worker.process_pending_areas() == 1
    with session(settings) as conn:
        row = conn.execute(
            "SELECT id, vector_status FROM skills WHERE name = ?",
            (CREATOR_SKILL_NAME,),
        ).fetchone()
        assert row["vector_status"] == "ok"
        assert area_vectors.count(conn, area_vectors.SKILLS_VEC_TABLE) == 1
        assert (
            area_vectors.get_vector(
                conn,
                area_vectors.SKILLS_VEC_TABLE,
                area_vectors.SKILLS_VEC_ID_COLUMN,
                int(row["id"]),
            )
            is not None
        )
