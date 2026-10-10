"""Петля `areas` и её наблюдаемость по флагам областей (№40, FR-2.5/FR-4.3).

Выключенная область не забирается фоном: её pending-записи остаются pending,
снимок очереди `areas` считает только включённые области; при этом
`areas_health()` отдаёт `{name: {enabled, pending}}` по ВСЕМ областям — накопление
у выключенной области видно (прецедент JOB_*_ENABLED), обращений к моделям нет.
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
    clear_seeded_skills(settings)  # сид skill-создателя не мешает счёту
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


def _status(table: str) -> str:
    with session(get_settings()) as conn:
        return conn.execute(
            f"SELECT vector_status FROM {table} ORDER BY id LIMIT 1"
        ).fetchone()[0]


def test_loop_skips_disabled_area(monkeypatch: pytest.MonkeyPatch) -> None:
    """AREA_TERMS_ENABLED=false: terms остаётся pending, прочие обрабатываются."""
    settings = _settings(monkeypatch, AREA_TERMS_ENABLED="false")
    _seed_pending_areas()
    worker = BackgroundWorker(settings, HashEmbedder(DIM))
    assert worker.process_pending_areas() == 2
    assert _status("skills") == "ok"
    assert _status("user_facts") == "ok"
    assert _status("terms") == "pending"


def test_areas_queue_stat_enabled_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Снимок очереди `areas` считает pending только включённых областей."""
    settings = _settings(monkeypatch, AREA_TERMS_ENABLED="false")
    _seed_pending_areas()
    worker = BackgroundWorker(settings, HashEmbedder(DIM))
    stat = worker._areas_queue_stat()
    assert stat["pending"] == 2  # skills + user; terms исключён (FR-2.5)


def test_areas_queue_stat_skips_disabled_logs(monkeypatch: pytest.MonkeyPatch) -> None:
    """f10-04 (§10): снимок очереди `areas` pending логов у off-области не считает."""
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
    """`areas_health()` — по всем областям, накопление у выключенной видно."""
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
