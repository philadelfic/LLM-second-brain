"""Цикл миграции forward → reverse → forward и наблюдаемость нарезки
(DEF-5/DEF-6/DEF-7a — QA-дополнение 27.10.2026).

tmp_path (test_env), без модели: повторный forward НЕ стирает базу цепочки
(копия частей первого прогона живёт в notes.<limit>.db); джоба нарезки
пишет slot_call на каждый вызов модели (как merge/dedup в worker);
/health.migration phase=split во время нарезки, done — после обнуления.
"""

from __future__ import annotations

import json
import logging
import pathlib
import sqlite3

import pytest
from fakes import FixedSummarizer
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import create_app
from app.services.migration import MigrationService
from app.services.notes import NoteService
from app.storage.db import init_db, note_chars_limit, session, transaction
from app.storage.migration import (
    MIGRATION_NOTES_TABLE,
    read_state,
    run_startup_migration,
    sibling_db_path,
    startup_plan,
)

_LONG_TEXT = "длинная заметка про цепочку миграций " * 150  # ~5100 знаков


def _env(limit: int, monkeypatch: pytest.MonkeyPatch) -> None:
    """Переместить env-лимит и сбросить кеш настроек (новый «запуск сервиса»)."""
    monkeypatch.setenv("MAX_NOTE_CHARS", str(limit))
    get_settings.cache_clear()


def _seed_source(settings, long_ids: tuple[int, ...] = (2,)) -> None:
    """База под спуск (CHECK 20000): короткая id=1 + длинные → 'ожидает'."""
    init_db(settings, note_chars=20000)
    with session(settings) as conn, transaction(conn):
        conn.execute("INSERT INTO notes(id, text) VALUES (1, 'короткая заметка')")
        for old_id in long_ids:
            conn.execute(
                "INSERT INTO notes(id, text) VALUES (?, ?)", (old_id, _LONG_TEXT)
            )


def _run_plan() -> str:
    """Выполнить стартовый план миграции; режим — назад (шаг сценария)."""
    plan = startup_plan(get_settings())
    run_startup_migration(get_settings(), plan)
    return str(plan["mode"])


def _simulate_split_part(title: str, text: str) -> int:
    """Симуляция M2: часть нарезки в активной базе (паттерн test_migration_storage).

    Часть + история source='migration_split', исходник id=2 → 'нарезана':
    при обратном переливе часть исключается из дельты и остаётся ТОЛЬКО в
    уходящей базе — как в кластерном прогоне QA (215 частей в notes.1500.db).
    """
    settings = get_settings()
    with session(settings) as conn, transaction(conn):
        conn.execute("INSERT INTO notes(text, title) VALUES (?, ?)", (text, title))
        part_id = int(
            conn.execute("SELECT id FROM notes WHERE title = ?", (title,)).fetchone()[0]
        )
        conn.execute(
            "INSERT INTO record_versions(entity, entity_id, version, source) "
            "VALUES ('note', ?, 1, 'migration_split')",
            (part_id,),
        )
        conn.execute(
            f"UPDATE {MIGRATION_NOTES_TABLE} SET status = 'нарезана', new_ids = ? "
            "WHERE old_id = 2",
            (json.dumps([part_id]),),
        )
    return part_id


def _migration_phase() -> str | None:
    """Фаза миграции из state активной базы (None — миграции нет)."""
    with session(get_settings()) as conn:
        state = read_state(conn)
    return None if state is None else str(state.get("phase"))


def _service(summarizer: FixedSummarizer) -> MigrationService:
    """MigrationService на текущих настройках (notes — настоящий, без модели)."""
    settings = get_settings()
    return MigrationService(
        settings, summarizer=summarizer, notes=NoteService(settings)
    )


# --- DEF-5: цикл forward → reverse → forward ---------------------------------


class TestForwardReverseForward:
    def test_second_forward_spares_chain_base(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Повторный forward не стирает notes.1500.db: обе прежние базы живы,
        chain описывает обе, часть первого прогона остаётся в цепочной базе."""
        settings = get_settings()
        _seed_source(settings)
        # forward 20000 → 1500: длинная 2 — 'ожидает', нарезка даёт часть.
        _env(1500, monkeypatch)
        assert _run_plan() == "down"
        part_id = _simulate_split_part("Часть первая", "самодостаточная часть текста")
        # reverse 1500 → 20000: часть исключается из дельты и остаётся только
        # в уходящей notes.1500.db (QA D9: «части остаются в уходящей базе»).
        _env(20000, monkeypatch)
        assert _run_plan() == "up_chain"
        low_path = sibling_db_path(settings.db_path, 1500)
        assert pathlib.Path(low_path).is_file()
        # Повторный forward 20000 → 1500 (QA D10): целевое имя notes.1500.db
        # занято живой базой цепочки — стирать её нельзя.
        _env(1500, monkeypatch)
        assert startup_plan(get_settings())["mode"] == "down"
        assert _run_plan() == "down"
        # ВСЕ файлы цепочки существуют: notes.20000.db (уходящая) и
        # notes.1500.db (цепочная, часть первого прогона) не удалены.
        assert pathlib.Path(sibling_db_path(settings.db_path, 20000)).is_file()
        assert pathlib.Path(low_path).is_file()
        assert not pathlib.Path(low_path + ".incoming").exists()  # swap переименовал
        with session(get_settings()) as conn:
            # Активная — новая база (CHECK 1500), чужих частей в ней нет.
            assert note_chars_limit(conn) == 1500
            assert part_id not in {
                int(r[0]) for r in conn.execute("SELECT id FROM notes")
            }
            meta = dict(
                conn.execute("SELECT key, value FROM meta").fetchall()
            )
        state = json.loads(meta["migration.state"])
        chain = json.loads(meta["migration.chain"])
        assert state["phase"] == "pour"
        assert {entry["file"] for entry in chain} == {
            "notes.20000.db",
            "notes.1500.db",
        }
        assert all(
            (pathlib.Path(settings.db_path).parent / entry["file"]).is_file()
            for entry in chain
        )
        # Единственная копия части первого прогона — в цепочной базе.
        with session(
            settings.model_copy(update={"db_path": low_path})
        ) as conn:
            assert part_id in {
                int(r[0]) for r in conn.execute("SELECT id FROM notes")
            }

    def test_crash_leftover_without_chain_is_still_rebuilt(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Файл под целевым именем БЕЗ записи в chain (недолитый, обрыв до
        swap) — стирается и переливается заново (design §6, как было)."""
        settings = get_settings()
        _seed_source(settings)
        garbage = sibling_db_path(settings.db_path, 1500)
        conn = sqlite3.connect(garbage)
        conn.execute("CREATE TABLE half_poured(x)")
        conn.commit()
        conn.close()
        _env(1500, monkeypatch)
        assert _run_plan() == "down"
        with session(get_settings()) as conn:
            assert note_chars_limit(conn) == 1500
            assert conn.execute(
                "SELECT 1 FROM sqlite_master WHERE name = 'half_poured'"
            ).fetchone() is None


# --- DEF-6: slot_call на каждый вызов нарезки ---------------------------------


class TestSplitSlotCall:
    def test_split_logs_slot_call_per_model_call(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Успешная нарезка пишет slot_call (slot=summary, job=migration) с
        operation/model/длительностью вызова — как merge в worker."""
        _env(1500, monkeypatch)
        settings = get_settings()
        _seed_source(settings)
        assert _run_plan() == "down"
        with caplog.at_level(logging.INFO, logger="app"):
            assert _service(FixedSummarizer()).process_batch() == 1
        calls = [
            record
            for record in caplog.records
            if getattr(record, "event", None) == "slot_call"
        ]
        assert len(calls) == 1  # один вызов модели — одна строка slot_call
        record = calls[0]
        assert record.slot == "summary"
        assert record.job == "migration"
        assert record.note_id == 2
        assert record.operation == "split"
        assert record.model
        assert record.duration_ms >= 0


# --- DEF-7a: phase=split во время нарезки -------------------------------------


class TestSplitPhaseHealth:
    def test_health_phase_split_during_slicing_done_after_drain(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """pour после перелива → split с первой партией нарезки (state и
        /health) → done после обнуления очереди (пустой прогон, как было)."""
        _env(1500, monkeypatch)
        settings = get_settings()
        _seed_source(settings, long_ids=(2, 3))
        assert _run_plan() == "down"
        # 'pour' сразу после перелива (state; живой воркер под TestClient
        # мог бы успеть взять партию — фазу 'pour' проверяем без приложения).
        assert _migration_phase() == "pour"
        service = _service(FixedSummarizer())
        # Первая партия: фаза нарезки видна в state и /health (вторая длинная
        # остаётся 'ожидает' — нарезка не кончилась, done не придёт).
        assert service.process_batch(limit=1) == 1
        assert _migration_phase() == "split"
        with TestClient(create_app()) as client:
            assert client.get("/health").json()["migration"]["phase"] == "split"
        # Вторая партия донарезает; done ставит следующий ПУСТОЙ прогон.
        assert service.process_batch() == 1
        assert _migration_phase() == "split"
        assert service.process_batch() == 0
        assert _migration_phase() == "done"
        with TestClient(create_app()) as client:
            assert client.get("/health").json()["migration"]["phase"] == "done"
