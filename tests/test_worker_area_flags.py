"""The `areas` loop and its observability by the area flags (#40, FR-2.5/FR-4.3).

    A disabled area is not drained by the background: its pending records stay
    pending; the `areas` queue snapshot counts only enabled areas; meanwhile
    `areas_health()` serves `{name: {enabled, pending}}` over ALL areas — a
    disabled area's accumulation is visible (the JOB_*_ENABLED precedent); no
    model calls.
"""

from __future__ import annotations

import pytest
from fakes import HashEmbedder, clear_seeded_skills

from app.config import get_settings
from app.services.logs import LogsService
from app.services.worker import BackgroundWorker
from app.storage.db import init_db, session, transaction

DIM = 8


def _settings(monkeypatch: pytest.MonkeyPatch, **flags: str):
    monkeypatch.setenv("EMBEDDING_DIM", str(DIM))
    for name in (
        "AREA_SKILLS_ENABLED",
        "AREA_TERMS_ENABLED",
        "AREA_USER_FACTS_ENABLED",
        "AREA_LOGS_ENABLED",
    ):
        monkeypatch.delenv(name, raising=False)
    for name, value in flags.items():
        monkeypatch.setenv(name, value)
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    clear_seeded_skills(settings)  # the skill-creator seed does not disturb the count
    return settings


def _seed_pending_areas() -> None:
    """One pending record per area (as the area services write them)."""
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


def _status(table: str) -> str:
    with session(get_settings()) as conn:
        return conn.execute(
            f"SELECT vector_status FROM {table} ORDER BY id LIMIT 1"
        ).fetchone()[0]


def test_loop_skips_disabled_area(monkeypatch: pytest.MonkeyPatch) -> None:
    """AREA_TERMS_ENABLED=false: terms stays pending; the others are processed."""
    settings = _settings(monkeypatch, AREA_TERMS_ENABLED="false")
    _seed_pending_areas()
    worker = BackgroundWorker(settings, HashEmbedder(DIM))
    assert worker.process_pending_areas() == 2
    assert _status("skills") == "ok"
    assert _status("user_facts") == "ok"
    assert _status("terms") == "pending"


def test_areas_queue_stat_enabled_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """The `areas` queue snapshot counts pending of enabled areas only."""
    settings = _settings(monkeypatch, AREA_TERMS_ENABLED="false")
    _seed_pending_areas()
    worker = BackgroundWorker(settings, HashEmbedder(DIM))
    stat = worker._areas_queue_stat()
    assert stat["pending"] == 2  # skills + user; terms excluded (FR-2.5)


def test_areas_queue_stat_skips_disabled_logs(monkeypatch: pytest.MonkeyPatch) -> None:
    """f10-04 (§10): the `areas` queue snapshot does not count an off-area's pending logs."""
    settings = _settings(monkeypatch, AREA_LOGS_ENABLED="false")
    saved = LogsService(settings, embedding=HashEmbedder(DIM)).save("Замер", "повод")
    assert saved["created"] is True
    worker = BackgroundWorker(settings, HashEmbedder(DIM))
    assert worker._areas_queue_stat()["pending"] == 0

    monkeypatch.setenv("AREA_LOGS_ENABLED", "true")
    get_settings.cache_clear()
    on = get_settings()
    assert BackgroundWorker(on, HashEmbedder(DIM))._areas_queue_stat()["pending"] == 1


def test_areas_health_covers_all_areas(monkeypatch: pytest.MonkeyPatch) -> None:
    """`areas_health()` — over all areas; a disabled one's accumulation is visible."""
    settings = _settings(monkeypatch, AREA_TERMS_ENABLED="false")
    _seed_pending_areas()
    worker = BackgroundWorker(settings, HashEmbedder(DIM))
    health = worker.areas_health()
    assert set(health) == {"skills", "terms", "user", "logs"}
    assert health["skills"] == {"enabled": True, "pending": 1}
    assert health["user"] == {"enabled": True, "pending": 1}
    assert health["terms"] == {"enabled": False, "pending": 1}
    assert health["logs"] == {"enabled": True, "pending": 0}
    assert health["terms"] == {"enabled": False, "pending": 1}
