"""№12 (3.2.0): схема предупреждений — таблица `warnings`, индексы, маркер.

Общая поверхность предупреждений памяти и терминов (№12 + №32, arch §5.1):
таблица `warnings`, частичный уникальный индекс открытого ключа и рабочий
индекс листинга, плюс служебный маркер `notes.warnings_at` (§5.3/§5.4).
Запись состояния идемпотентна: повторный `init_db` (рестарт) — no-op.
"""

from __future__ import annotations

import pytest

from app.config import Settings, get_settings
from app.services.warnings import KIND_NOTE_MISPLACED, WarningsService
from app.storage.db import init_db, session


@pytest.fixture
def settings(test_env: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Схема в тестовой БД; размерность 8 — модели не участвуют."""
    monkeypatch.setenv("EMBEDDING_DIM", "8")
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    return settings


def _objects(settings: Settings) -> set[str]:
    with session(settings) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','index')"
        ).fetchall()
    return {row["name"] for row in rows}


def _notes_columns(settings: Settings) -> set[str]:
    with session(settings) as conn:
        return {row["name"] for row in conn.execute("PRAGMA table_info(notes)")}


def _warning_count(settings: Settings) -> int:
    with session(settings) as conn:
        return int(conn.execute("SELECT COUNT(*) AS n FROM warnings").fetchone()["n"])


def test_schema_and_marker_idempotent(settings: Settings) -> None:
    """Маркер и таблица/индексы есть; повторный `init_db` число строк не растит."""
    assert "warnings_at" in _notes_columns(settings)
    objects = _objects(settings)
    assert "warnings" in objects
    assert "idx_warnings_open_unique" in objects
    assert "idx_warnings_status_kind" in objects

    WarningsService(settings, classifier=None, namespaces=None).open(
        KIND_NOTE_MISPLACED, "note", 1, "m", {"v": 1}
    )
    before = _warning_count(settings)
    init_db(settings)  # рестарт сервиса — no-op
    assert _warning_count(settings) == before == 1
    assert "warnings_at" in _notes_columns(settings)


def test_open_unique_partial(settings: Settings) -> None:
    """Второй `open` того же ключа обновляет строку; после `resolve` — новая."""
    service = WarningsService(settings, classifier=None, namespaces=None)
    first = service.open(KIND_NOTE_MISPLACED, "note", 1, "m1", {"v": 1})
    second = service.open(KIND_NOTE_MISPLACED, "note", 1, "m2", {"v": 2})
    assert first == second  # upsert, а не вторая строка
    with session(settings) as conn:
        rows = conn.execute(
            "SELECT message FROM warnings WHERE id = ?", (first,)
        ).fetchall()
    assert len(rows) == 1 and rows[0]["message"] == "m2"

    assert service.resolve(KIND_NOTE_MISPLACED, "note", 1) == 1
    third = service.open(KIND_NOTE_MISPLACED, "note", 1, "m3", {})
    assert third != first  # причина вернулась — открывается новая строка
    assert _warning_count(settings) == 2
