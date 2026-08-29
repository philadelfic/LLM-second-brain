"""Тесты слоя хранилища (Фаза 2, Шаг 2.1): схема, FTS-синхронизация, pragmas.

ARCHITECTURE §3.3: notes + notes_fts (external content, trigram) + триггеры;
WAL + busy_timeout; CHECK-лимит; идемпотентная инициализация при старте.
"""

from __future__ import annotations

import sqlite3

import pytest

from app.config import Settings, get_settings
from app.storage.db import BUSY_TIMEOUT_MS, StorageError, init_db, session


@pytest.fixture
def initialized_db() -> None:
    """Инициализировать схему в тестовой БД (DB_PATH — из test_env)."""
    init_db(get_settings())


def _match(conn: sqlite3.Connection, needle: str) -> list[int]:
    """rowid-ы заметок, содержащих подстроку needle (trigram ≥ 3 симв.)."""
    rows = conn.execute(
        "SELECT rowid FROM notes_fts WHERE notes_fts MATCH ?",
        (f'"{needle}"',),
    ).fetchall()
    return [row[0] for row in rows]


class TestSchemaCreated:
    def test_notes_columns(self) -> None:
        """Таблица notes — ровно 9 колонок из ARCHITECTURE §3.3."""
        init_db(get_settings())
        with session(get_settings()) as conn:
            columns = {
                row["name"]: row for row in conn.execute("PRAGMA table_info(notes)")
            }
        assert set(columns) == {
            "id", "text", "summary", "author", "vector_status",
            "summary_status", "created_at", "updated_at", "deleted_at",
        }
        # DDL-умолчания: summary пуст, статусы pending, author unknown.
        assert columns["summary"]["dflt_value"] == "''"
        assert columns["vector_status"]["dflt_value"] == "'pending'"
        assert columns["summary_status"]["dflt_value"] == "'pending'"
        assert columns["author"]["dflt_value"] == "'unknown'"
        assert columns["deleted_at"]["notnull"] == 0  # NULL = активна

    def test_fts_external_content_trigram(self) -> None:
        """notes_fts — FTS5 внешний контент с trigram-токенизатором."""
        init_db(get_settings())
        with session(get_settings()) as conn:
            sql = conn.execute(
                "SELECT sql FROM sqlite_master WHERE name='notes_fts'"
            ).fetchone()[0]
        assert "content='notes'" in sql
        assert "content_rowid='id'" in sql
        assert "tokenize='trigram'" in sql

    def test_sync_triggers_installed(self) -> None:
        init_db(get_settings())
        with session(get_settings()) as conn:
            names = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='trigger'"
                )
            }
        assert {"notes_fts_ai", "notes_fts_au"} <= names  # DELETE-триггер не нужен

    def test_check_backstop_rejects_out_of_range(self) -> None:
        """CHECK — последний рубеж: пустой и слишком длинный текст отвергнут."""
        init_db(get_settings())
        with session(get_settings()) as conn:
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute("INSERT INTO notes(text) VALUES ('')")
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute("INSERT INTO notes(text) VALUES (?)", ("x" * 2001,))

    def test_env_limit_substituted_into_check(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Лимит CHECK берётся из env при первой инициализации (ARCH §3.3)."""
        monkeypatch.setenv("MAX_NOTE_CHARS", "10")
        get_settings.cache_clear()
        settings = get_settings()
        init_db(settings)
        with session(settings) as conn:
            conn.execute("INSERT INTO notes(text) VALUES (?)", ("x" * 10,))
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute("INSERT INTO notes(text) VALUES (?)", ("x" * 11,))

    def test_start_fails_when_max_note_chars_diverged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Смена MAX_NOTE_CHARS поверх готовой БД — фатальный отказ старта.

        Лимит «запечён» в CHECK при первом создании; запуск сервиса с другим
        значением невозможен — крупные заметки падали бы в рантайме с
        невнятным IntegrityError (Фаза 5: аналог гейта EMBEDDING_DIM).
        """
        monkeypatch.setenv("MAX_NOTE_CHARS", "10")
        get_settings.cache_clear()
        init_db(get_settings())  # БД создана с лимитом 10
        monkeypatch.setenv("MAX_NOTE_CHARS", "500")
        get_settings.cache_clear()
        with pytest.raises(StorageError, match="MAX_NOTE_CHARS") as exc_info:
            init_db(get_settings())
        # Сообщение называет оба значения — оператор сразу видит конфликт.
        assert "10" in str(exc_info.value)
        assert "500" in str(exc_info.value)

    def test_start_ok_with_same_limit(self) -> None:
        """Перезапуск с тем же лимитом — сверка не мешает (идемпотентность)."""
        init_db(get_settings())
        init_db(get_settings())

    def test_idempotent_and_data_preserved(self) -> None:
        settings = get_settings()
        init_db(settings)
        with session(settings) as conn:
            conn.execute("INSERT INTO notes(text) VALUES ('заметка')")
        with session(settings) as conn:
            assert conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0] == 1
        init_db(settings)  # второй прогон без ошибок, данные целы
        with session(settings) as conn:
            assert conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0] == 1

    def test_defaults_on_insert(self, initialized_db: None) -> None:
        """Служебные поля заполняются DDL-умолчаниями."""
        with session(get_settings()) as conn:
            conn.execute("INSERT INTO notes(text) VALUES ('заметка')")
            row = conn.execute("SELECT * FROM notes WHERE id = 1").fetchone()
        assert row["summary"] == ""
        assert row["author"] == "unknown"
        assert row["vector_status"] == "pending"
        assert row["summary_status"] == "pending"
        assert row["deleted_at"] is None
        # UTC ISO-8601, формат DDL: 2026-08-29T12:34:56Z
        assert row["created_at"] and row["created_at"].endswith("Z")
        assert len(row["created_at"]) == 20
        assert row["updated_at"] == row["created_at"]  # одна секунда, одно изменение


class TestPragmas:
    def test_wal_mode_persisted(self, initialized_db: None) -> None:
        with session(get_settings()) as conn:
            assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"

    def test_busy_timeout(self) -> None:
        with session(get_settings()) as conn:
            assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == BUSY_TIMEOUT_MS


class TestInit:
    def test_unwritable_path_is_storage_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Путь, где родитель — файл, даёт понятную StorageError."""
        settings = get_settings()
        blocked = settings.db_path + "-file"  # ФАЙЛ на месте каталога
        with open(blocked, "w", encoding="utf-8") as handle:
            handle.write("занято")

        monkeypatch.setenv("DB_PATH", f"{blocked}/notes.db")
        get_settings.cache_clear()
        broken: Settings = get_settings()
        with pytest.raises(StorageError):
            init_db(broken)
        assert broken is not settings  # синглтон не тронут (кэш чистился)

    def test_schema_on_startup_of_app(self, client) -> None:
        """Критерий приёмки: схема создаётся при старте приложения."""
        with session(get_settings()) as conn:
            objects = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type IN ('table','trigger')"
                )
            }
        assert {"notes", "notes_fts", "notes_fts_ai", "notes_fts_au"} <= objects


class TestFtsSync:
    def test_insert_indexes_text(self, initialized_db: None) -> None:
        with session(get_settings()) as conn:
            conn.execute(
                "INSERT INTO notes(text) VALUES (?)",
                ("Сервис TaskFlow развёрнут на 192.168.1.50",),
            )
            assert _match(conn, "TaskFlow") == [1]

    def test_update_resyncs_fts(self, initialized_db: None) -> None:
        with session(get_settings()) as conn:
            conn.execute(
                "INSERT INTO notes(text) VALUES ('Сервис TaskFlow запущен')"
            )
            assert _match(conn, "TaskFlow") == [1]
            conn.execute("UPDATE notes SET text='Сервис TestFlow запущен' WHERE id=1")
            assert _match(conn, "TaskFlow") == []  # старое слово ушло из индекса
            assert _match(conn, "TestFlow") == [1]  # новое — попало

    def test_non_text_update_does_not_touch_fts(self, initialized_db: None) -> None:
        """UPDATE прочих колонок (summary) не дублирует записи в индексе."""
        with session(get_settings()) as conn:
            conn.execute(
                "INSERT INTO notes(text) VALUES ('Сервис TaskFlow запущен')"
            )
            conn.execute(
                "UPDATE notes SET summary = 'краткое содержание' WHERE id = 1"
            )
            assert _match(conn, "TaskFlow") == [1]  # ровно одна запись

    def test_soft_delete_keeps_row_and_index(self, initialized_db: None) -> None:
        """Soft delete не трогает ни строку, ни FTS-индекс (trash, ARCH §4.6)."""
        with session(get_settings()) as conn:
            conn.execute(
                "INSERT INTO notes(text) VALUES ('Сервис TaskFlow запущен')"
            )
            conn.execute("UPDATE notes SET deleted_at = CURRENT_TIMESTAMP WHERE id=1")
            row = conn.execute(
                "SELECT text, deleted_at FROM notes WHERE id = 1"
            ).fetchone()
            assert row["text"] == "Сервис TaskFlow запущен"  # физически жива
            assert row["deleted_at"] is not None
            assert _match(conn, "TaskFlow") == [1]  # индекс сохранён

    def test_trigram_finds_wordforms_by_substring(
        self, initialized_db: None
    ) -> None:
        """Trigram-семантика: подстроки ≥3 симв., включая русские формы.

        «развёртыван…» — общий кусок слова. Полное слово запроса, которого
        нет в тексте (другой суффикс), НЕ ищется — таков контракт trigram
        (REQUIREMENTS §5.4: подстроки, без внешнего стеммера).
        """
        with session(get_settings()) as conn:
            conn.execute(
                "INSERT INTO notes(text) VALUES ('Развёртывание сервера выполнено')"
            )
            assert conn.execute(
                "SELECT rowid FROM notes_fts WHERE notes_fts MATCH ?",
                ('"развёртыван"',),
            ).fetchall()
            # точный токен/адрес в тексте находится:
            conn.execute("UPDATE notes SET text='IP: 192.168.1.50' WHERE id=1")
            assert _match(conn, "192.168.1.50") == [1]
            # два и менее символов — гарантированно пусто:
            assert _match(conn, "ab") == []

    def test_fts_matches_only_indexed_text_not_summary(
        self, initialized_db: None
    ) -> None:
        """summary не индексируется (ARCH §3.3: «не индексируется FTS»)."""
        with session(get_settings()) as conn:
            conn.execute("INSERT INTO notes(text) VALUES ('заметка про задачи')")
            conn.execute(
                "UPDATE notes SET summary='Об АРХИВНОМ хранилище' WHERE id = 1"
            )
            assert _match(conn, "АРХИВНОМ") == []  # в индексе только text

    def test_rebuild_heals_direct_operator_edit(self) -> None:
        """Оператор правил notes.text мимо триггеров → старт самолечится."""
        settings = get_settings()
        init_db(settings)
        with session(settings) as conn:  # заметка попала в FTS штатным триггером
            conn.execute("INSERT INTO notes(text) VALUES ('Было в индексе')")
        raw = sqlite3.connect(settings.db_path)
        raw.execute("UPDATE notes SET text='Прямая правка мимо триггера'")
        raw.commit()
        raw.close()
        init_db(settings)  # integrity-check ловит, rebuild чинит — без ошибки
        with session(settings) as conn:
            assert _match(conn, "правка") == [1]
            assert _match(conn, "триггера") == [1]
            assert _match(conn, "Было") == []  # устаревшие токены вычищены