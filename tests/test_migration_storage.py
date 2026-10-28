"""Самомиграция схемы — слой хранилища и старт (LSBN-F-22, M1).

tmp_path (test_env), без модели: обнаружение вниз/вверх/свежая база;
перелив id+FTS+вектора+области; swap; рестарт до/после swap; обратный
перелив без частей (и с частями — исключение по record_versions);
приёмка через lifespan: env поверх чужого CHECK — сервис поднимается.
"""

from __future__ import annotations

import json
import logging
import pathlib
import sqlite3

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import create_app
from app.storage import vectors
from app.storage.db import init_db, note_chars_limit, session, transaction
from app.storage.migration import (
    MIGRATION_NOTES_TABLE,
    create_new_base,
    run_startup_migration,
    sibling_db_path,
    startup_plan,
    swap,
)

_LONG_TEXT = "очень длинная заметка для миграции " * 150  # ~5100 знаков


def _env(limit: int, monkeypatch: pytest.MonkeyPatch) -> None:
    """Переместить env-лимит и сбросить кеш настроек (новый «запуск сервиса»)."""
    monkeypatch.setenv("MAX_NOTE_CHARS", str(limit))
    get_settings.cache_clear()


def _seed_source(settings) -> None:
    """Наполнить базу под перелив: заметки, связи, TTL, версии, вектора, области.

    Заметки 1 и 3 — короткие (1 — с вектором и чанком), 2 — длинная со всей
    обвязкой (при переливе вниз остаётся в прежней базе). Связи (1,2) и
    (1,3); предупреждение о заметке 2 и одно не про заметку; в области —
    term с вектором. Свежая база: id 1..3 и chunk id 1 детерминированы.
    """
    init_db(settings, note_chars=20000)
    with session(settings) as conn, transaction(conn):
        conn.execute(
            "INSERT INTO notes(id, text) VALUES (1, 'привет мир поиск заметка')"
        )
        conn.execute("INSERT INTO notes(id, text) VALUES (2, ?)", (_LONG_TEXT,))
        conn.execute(
            "INSERT INTO notes(id, text) VALUES (3, 'короткая вторая')"
        )
        conn.execute(
            "INSERT INTO links(note_a, note_b, kind) VALUES (1, 2, 'cosine')"
        )
        conn.execute(
            "INSERT INTO links(note_a, note_b, kind, score) VALUES (1, 3, 'mention', NULL)"
        )
        conn.execute(
            "INSERT INTO note_expirations(note_id, expires_at) "
            "VALUES (2, '2030-01-01T00:00:00Z')"
        )
        conn.execute(
            "INSERT INTO warnings(kind, subject_type, subject_id, message) "
            "VALUES ('dedup_stuck', 'note', 2, 'дубль не разрешён')"
        )
        conn.execute(
            "INSERT INTO warnings(kind, subject_type, subject_id, message) "
            "VALUES ('area_stuck', 'term', 1, 'вектор термина не кодируется')"
        )
        conn.execute(
            "INSERT INTO notes_chunks(note_id, idx, text, tokens) "
            "VALUES (1, 0, 'привет мир поиск', 4)"
        )
        conn.execute(
            "INSERT INTO notes_chunks(note_id, idx, text, tokens) "
            "VALUES (2, 0, ?, 100)",
            (_LONG_TEXT,),
        )
        conn.execute(
            "INSERT INTO record_versions(entity, entity_id, version, source) "
            "VALUES ('note', 1, 1, 'mcp')"
        )
        conn.execute(
            "INSERT INTO record_versions(entity, entity_id, version, source) "
            "VALUES ('note', 2, 1, 'mcp')"
        )
        conn.execute(
            "INSERT INTO terms(term, term_norm, context, context_norm, definition) "
            "VALUES ('Migr', 'migr', 'контекст', 'контекст', 'тестовый термин')"
        )
        dim = vectors.existing_vec_dim(conn)
        conn.execute(
            "INSERT INTO notes_vec(note_id, ns, embedding) VALUES (1, 'default', ?)",
            (vectors.pack([0.5] * dim),),
        )
        conn.execute(
            "INSERT INTO notes_vec(note_id, ns, embedding) VALUES (2, 'default', ?)",
            (vectors.pack([0.25] * dim),),
        )
        conn.execute(
            "INSERT INTO notes_chunks_vec(chunk_id, ns, embedding) "
            "VALUES (1, 'default', ?)",
            (vectors.pack([0.75] * dim),),
        )
        conn.execute(
            "INSERT INTO terms_vec(term_id, embedding) VALUES (1, ?)",
            (vectors.pack([0.125] * dim),),
        )


def _table_ids(conn: sqlite3.Connection, table: str, column: str = "id") -> set[int]:
    return {int(row[0]) for row in conn.execute(f"SELECT {column} FROM {table}")}


# --- фаза 0: обнаружение ----------------------------------------------------


class TestStartupPlan:
    def test_fresh_base_no_migration(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Свежая база (файла нет / env совпал) — миграция не нужна."""
        _env(1500, monkeypatch)
        settings = get_settings()
        assert startup_plan(settings)["mode"] == "none"  # файла ещё нет
        init_db(settings)
        assert startup_plan(settings)["mode"] == "none"  # CHECK == env

    def test_down_plan_counts_long_notes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Вниз (20000 → 1500): план down, prev_file = notes.20000.db, pending по длине."""
        settings = get_settings()
        _seed_source(settings)
        _env(1500, monkeypatch)
        plan = startup_plan(get_settings())
        assert plan["mode"] == "down"
        assert plan["schema_limit"] == 20000
        assert plan["target"] == 1500
        assert plan["pending"] == 1
        assert plan["prev_file"].endswith("notes.20000.db")

    def test_up_plan_reuses_chain_base(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Вверх после спуска: источник — база цепочки с CHECK = цели."""
        settings = get_settings()
        _seed_source(settings)
        prev_file = sibling_db_path(settings.db_path, 20000)
        create_new_base(
            get_settings(),
            source_path=settings.db_path,
            target=1500,
            prev_file=prev_file,
            reason="тест",
        )
        swap(
            get_settings(),
            away_file=prev_file,
            incoming_file=sibling_db_path(settings.db_path, 1500),
        )
        _env(20000, monkeypatch)
        plan = startup_plan(get_settings())
        assert plan["mode"] == "up_chain"
        assert plan["source_path"].endswith("notes.20000.db")
        assert plan["prev_file"].endswith("notes.1500.db")

    def test_up_plan_new_base_without_chain(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Вверх без родственника в цепочке — новая база, тоже без модели."""
        settings = get_settings()
        init_db(settings, note_chars=1500)
        _env(20000, monkeypatch)
        plan = startup_plan(get_settings())
        assert plan["mode"] == "up_new"
        assert plan["schema_limit"] == 1500


# --- фазы 2–3: перелив и swap -----------------------------------------------


class TestCreateNewBase:
    def test_pour_keeps_ids_fts_vectors_areas(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Перелив вниз: id сохранены, FTS построен триггерами, вектора и области на месте."""
        settings = get_settings()
        _seed_source(settings)
        counters = create_new_base(
            settings,
            source_path=settings.db_path,
            target=1500,
            prev_file=sibling_db_path(settings.db_path, 20000),
            reason="тест",
        )
        assert counters == {"old_total": 3, "poured": 2, "pending": 1}
        new_path = sibling_db_path(settings.db_path, 1500)
        assert pathlib.Path(new_path).is_file()
        new_settings = settings.model_copy(update={"db_path": new_path})
        with session(new_settings) as conn:
            assert _table_ids(conn, "notes") == {1, 3}
            # FTS построен триггерами INSERT — rebuild не нужен (design §3).
            rows = conn.execute(
                "SELECT rowid FROM notes_fts WHERE notes_fts MATCH '\"привет\"'"
            ).fetchall()
            assert [int(r[0]) for r in rows] == [1]
            # Зависимости влезающих: связь 1↔3 перелита, 1↔2 осталась в прежней.
            assert {
                (int(r["note_a"]), int(r["note_b"]))
                for r in conn.execute("SELECT note_a, note_b FROM links")
            } == {(1, 3)}
            assert _table_ids(conn, "notes_chunks", "note_id") == {1}
            # История заметок: сид skill_template (entity_id=0) не считается.
            note_versions = {
                int(r[0])
                for r in conn.execute(
                    "SELECT entity_id FROM record_versions WHERE entity = 'note'"
                )
            }
            assert note_versions == {1}
            assert conn.execute("SELECT COUNT(*) FROM note_expirations").fetchone()[0] == 0
            # Предупреждение о невлезающей заметке — в прежней базе.
            warnings = {
                (r["subject_type"], int(r["subject_id"]))
                for r in conn.execute("SELECT subject_type, subject_id FROM warnings")
            }
            assert warnings == {("term", 1)}
            # Вектора — построчно, байт в байт; области — целиком.
            dim = vectors.existing_vec_dim(conn)
            src_embedding = vectors.pack([0.5] * dim)
            assert conn.execute(
                "SELECT embedding FROM notes_vec WHERE note_id = 1"
            ).fetchone()[0] == src_embedding
            assert conn.execute("SELECT COUNT(*) FROM notes_vec").fetchone()[0] == 1
            assert conn.execute("SELECT COUNT(*) FROM notes_chunks_vec").fetchone()[0] == 1
            assert conn.execute("SELECT COUNT(*) FROM terms_vec").fetchone()[0] == 1
            # Мета миграции (design §3): штамп, цепочка, состояние, 'ожидает'.
            assert note_chars_limit(conn) == 1500
            state_rows = conn.execute(
                "SELECT value FROM meta WHERE key = 'schema.note_chars'"
            ).fetchall()
            assert [str(r[0]) for r in state_rows] == ["1500"]
            migration_notes = {
                int(r["old_id"]): r["status"]
                for r in conn.execute(f"SELECT old_id, status FROM {MIGRATION_NOTES_TABLE}")
            }
            assert migration_notes == {2: "ожидает"}
            # История заметок: сид skill_template (entity_id=0) не считается.
            note_versions = {
                int(r[0])
                for r in conn.execute(
                    "SELECT entity_id FROM record_versions WHERE entity = 'note'"
                )
            }
            assert note_versions == {1}
        with session(new_settings) as conn:
            chain = json.loads(
                conn.execute(
                    "SELECT value FROM meta WHERE key = 'migration.chain'"
                ).fetchone()[0]
            )
            state = json.loads(
                conn.execute(
                    "SELECT value FROM meta WHERE key = 'migration.state'"
                ).fetchone()[0]
            )
        assert chain == [{"file": "notes.20000.db", "note_chars": 20000}]
        assert state["phase"] == "pour"
        assert state["target"] == 1500
        assert state["prev_limit"] == 20000
        assert state["counters"] == counters

    def test_restart_before_swap_rebuilds_from_scratch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Рестарт до swap (design §6): недолитый файл стирается, перелив заново."""
        settings = get_settings()
        _seed_source(settings)
        temp_path = sibling_db_path(settings.db_path, 1500)
        garbage = sqlite3.connect(temp_path)  # «недолитый» файл прошлой попытки
        garbage.execute("CREATE TABLE half_poured(x)")
        garbage.commit()
        garbage.close()
        _env(1500, monkeypatch)
        plan = startup_plan(get_settings())
        run_startup_migration(get_settings(), plan)
        with session(get_settings()) as conn:
            assert note_chars_limit(conn) == 1500
            assert conn.execute(
                "SELECT 1 FROM sqlite_master WHERE name = 'half_poured'"
            ).fetchone() is None
            assert conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0] == 2
            # История заметок: сид skill_template (entity_id=0) не считается.
            note_versions = {
                int(r[0])
                for r in conn.execute(
                    "SELECT entity_id FROM record_versions WHERE entity = 'note'"
                )
            }
            assert note_versions == {1}
        # Прежняя база цела и переехала под именем своего CHECK.
        old_settings = settings.model_copy(
            update={"db_path": sibling_db_path(settings.db_path, 20000)}
        )
        with session(old_settings) as conn:
            assert note_chars_limit(conn) == 20000
            assert conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0] == 3

    def test_restart_after_swap_is_noop(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Рестарт после swap: CHECK активной базы равен env — миграция не нужна."""
        settings = get_settings()
        _seed_source(settings)
        _env(1500, monkeypatch)
        run_startup_migration(get_settings(), startup_plan(get_settings()))
        assert startup_plan(get_settings())["mode"] == "none"
        assert not pathlib.Path(
            sibling_db_path(settings.db_path, 1500)
        ).exists()  # новая база СТАЛА активной, файл-родственник не плодится


class TestSwap:
    def test_swap_moves_files(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Swap: прежняя база → notes.<L>.db, входящая → db_path, WAL сайдкаров нет."""
        settings = get_settings()
        init_db(settings, note_chars=20000)
        with session(settings) as conn, transaction(conn):
            conn.execute("INSERT INTO notes(text) VALUES ('старая база')")
        incoming = settings.db_path + ".incoming"
        new_settings = settings.model_copy(update={"db_path": incoming})
        init_db(new_settings, note_chars=1500)
        with session(new_settings) as conn, transaction(conn):
            conn.execute("INSERT INTO notes(text) VALUES ('новая база')")
        away_file = sibling_db_path(settings.db_path, 20000)
        swap_files = swap

        swap_files(settings, away_file=away_file, incoming_file=incoming)
        with session(get_settings()) as conn:
            assert note_chars_limit(conn) == 1500
            assert conn.execute("SELECT text FROM notes").fetchone()[0] == "новая база"
        with session(settings.model_copy(update={"db_path": away_file})) as conn:
            assert conn.execute("SELECT text FROM notes").fetchone()[0] == "старая база"
        assert not pathlib.Path(settings.db_path + "-wal").exists()


# --- обратное направление (env вверх / откат) -------------------------------


def _migrate_down(monkeypatch: pytest.MonkeyPatch) -> None:
    """Полный спуск 20000 → 1500 (перелив + swap) на тестовой базе."""
    settings = get_settings()
    _seed_source(settings)
    _env(1500, monkeypatch)
    run_startup_migration(get_settings(), startup_plan(get_settings()))


class TestReversePour:
    def test_reverse_pour_without_parts_restores_long_notes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Откат 1:1 без модели: дельта возвращается, длинные — целыми (design §4)."""
        _migrate_down(monkeypatch)
        settings = get_settings()
        # Дельта активной базы (CHECK 1500): правка влезающей + новая заметка.
        with session(settings) as conn, transaction(conn):
            conn.execute(
                "UPDATE notes SET text = 'поправленная заметка' WHERE id = 1"
            )
            conn.execute("INSERT INTO notes(text) VALUES ('написано после миграции')")
        _env(20000, monkeypatch)
        plan = startup_plan(get_settings())
        assert plan["mode"] == "up_chain"
        run_startup_migration(get_settings(), plan)
        with session(get_settings()) as conn:
            assert note_chars_limit(conn) == 20000
            rows = {
                int(r["id"]): str(r["text"])
                for r in conn.execute("SELECT id, text FROM notes")
            }
            assert rows[1] == "поправленная заметка"
            assert rows[4] == "написано после миграции"
            assert len(rows[2]) == len(_LONG_TEXT)  # длинная вернулась целой
            assert {1, 2, 3, 4} <= set(rows)
        with session(get_settings()) as conn:
            state = json.loads(
                conn.execute(
                    "SELECT value FROM meta WHERE key = 'migration.state'"
                ).fetchone()[0]
            )
            chain = json.loads(
                conn.execute(
                    "SELECT value FROM meta WHERE key = 'migration.chain'"
                ).fetchone()[0]
            )
        assert state["phase"] == "done"
        assert [entry["note_chars"] for entry in chain] == [20000, 1500]
        # Повторный старт с тем же env — миграция не нужна.
        assert startup_plan(get_settings())["mode"] == "none"

    def test_reverse_pour_excludes_split_parts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Части (record_versions source='migration_split') из дельты исключены."""
        _migrate_down(monkeypatch)
        settings = get_settings()
        with session(settings) as conn, transaction(conn):
            # Симуляция M2: часть нарезки заметки 2 + её обвязка.
            conn.execute(
                "INSERT INTO notes(text, title) VALUES ('часть про даты', 'Часть даты')"
            )
            part_id = int(
                conn.execute(
                    "SELECT id FROM notes WHERE title = 'Часть даты'"
                ).fetchone()[0]
            )
            conn.execute(
                "INSERT INTO links(note_a, note_b, kind) VALUES (1, ?, 'mention')",
                (part_id,),
            )
            conn.execute(
                "INSERT INTO notes_chunks(note_id, idx, text, tokens) "
                "VALUES (?, 0, 'часть про даты', 3)",
                (part_id,),
            )
            conn.execute(
                "INSERT INTO record_versions(entity, entity_id, version, source) "
                "VALUES ('note', ?, 1, 'migration_split')",
                (part_id,),
            )
            dim = vectors.existing_vec_dim(conn)
            conn.execute(
                "INSERT INTO notes_vec(note_id, ns, embedding) VALUES (?, 'default', ?)",
                (part_id, vectors.pack([0.1] * dim)),
            )
        _env(20000, monkeypatch)
        plan = startup_plan(get_settings())
        counters = run_startup_migration(get_settings(), plan)
        assert counters["kept_parts"] == 1
        # После swap вернувшаяся база — активная (db_path), sibling gone.
        with session(get_settings()) as conn:
            assert note_chars_limit(conn) == 20000
            assert part_id not in _table_ids(conn, "notes")
            assert part_id not in _table_ids(conn, "notes_chunks", "note_id")
            assert part_id not in _table_ids(
                conn, "record_versions", "entity_id"
            )  # entity_id 0 — сид шаблона, не часть
            assert part_id not in _table_ids(conn, "notes_vec", "note_id")
            pair_links = {
                (int(r["note_a"]), int(r["note_b"]))
                for r in conn.execute("SELECT note_a, note_b FROM links")
            }
            assert all(part_id not in pair for pair in pair_links)
        assert not pathlib.Path(
            sibling_db_path(get_settings().db_path, 20000)
        ).exists()


# --- приёмка M1: старт сервиса поверх чужого CHECK ---------------------------


class TestLifespanAcceptance:
    def test_env_1500_over_check_20000_service_boots(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """env=1500 поверх CHECK=20000: сервис поднят, влезающее — в новом файле."""
        settings = get_settings()
        _seed_source(settings)
        _env(1500, monkeypatch)
        with caplog.at_level(logging.INFO, logger="app"):
            with TestClient(create_app()) as client:
                response = client.get("/health")
                assert response.status_code == 200
        # Сервис на новой базе: CHECK 1500, влезающие заметки в ней.
        with session(get_settings()) as conn:
            assert note_chars_limit(conn) == 1500
            texts = {str(r[0]) for r in conn.execute("SELECT text FROM notes")}
            assert texts == {"привет мир поиск заметка", "короткая вторая"}
            status = {
                int(r["old_id"]): r["status"]
                for r in conn.execute(
                    f"SELECT old_id, status FROM {MIGRATION_NOTES_TABLE}"
                )
            }
            assert status == {2: "ожидает"}
        assert pathlib.Path(sibling_db_path(settings.db_path, 20000)).is_file()
        # Первая строка журнала джоб о миграции — с причиной и объёмом.
        events = [
            record for record in caplog.records
            if getattr(record, "event", None) == "migration_started"
        ]
        assert len(events) == 1
        assert events[0].job == "migration"
        assert events[0].schema_limit == 20000
        assert events[0].target == 1500
        assert events[0].pending == 1
        swapped = [
            record for record in caplog.records
            if getattr(record, "event", None) == "migration_swapped"
        ]
        assert len(swapped) == 1

    def test_env_20000_over_check_1500_rollback_without_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """env=20000 поверх CHECK=1500: обратная миграция 1:1, без модели."""
        _migrate_down(monkeypatch)
        _env(20000, monkeypatch)
        with TestClient(create_app()) as client:
            assert client.get("/health").status_code == 200
        with session(get_settings()) as conn:
            assert note_chars_limit(conn) == 20000
            assert conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0] == 3
        assert startup_plan(get_settings())["mode"] == "none"
