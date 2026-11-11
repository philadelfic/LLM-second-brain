"""The "Assistant logs" area's registry (LSBN-F-10, f10-01): AreaSpec, the flag, the queue.

    `LOGS_AREA` is registered in `ALL_AREAS` and picked up by `enabled_areas`
    (#40) with no new code; the `area_logs_enabled` flag gates the background and
    the surface, not the data — a disabled area stays in `ALL_AREAS`. The
    worker's `areas` loop sees pending logs by the same mechanics as the other
    areas (arch §5.3/§6.2).
"""

from __future__ import annotations

import pytest
from fakes import HashEmbedder, clear_seeded_skills

from app.config import Settings, get_settings
from app.services.areas import ALL_AREAS, LOGS_AREA, enabled_areas
from app.services.worker import BackgroundWorker
from app.storage.db import init_db, session, transaction

DIM = 8

_LOGS_FLAG = "AREA_LOGS_ENABLED"


def _settings(monkeypatch: pytest.MonkeyPatch, **flags: str) -> Settings:
    """The logs area's settings (the flag off → the true default); the schema created."""
    monkeypatch.setenv("EMBEDDING_DIM", str(DIM))
    monkeypatch.delenv(_LOGS_FLAG, raising=False)
    for name, value in flags.items():
        monkeypatch.setenv(name, value)
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    clear_seeded_skills(settings)  # the skill-creator seed does not disturb the count
    return settings


def test_logs_area_registered() -> None:
    """`LOGS_AREA` in `ALL_AREAS`; its spec matches the §6.2 contract."""
    assert LOGS_AREA in ALL_AREAS
    assert LOGS_AREA.name == "logs"
    assert LOGS_AREA.settings_flag == "area_logs_enabled"
    assert LOGS_AREA.table == "assistant_logs"
    assert LOGS_AREA.fts_table == "assistant_logs_fts"
    assert LOGS_AREA.vec_table == "assistant_logs_vec"
    assert LOGS_AREA.vec_id_column == "log_id"
    assert LOGS_AREA.embed_fields == ("action", "reason", "outcome")


def test_logs_embed_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """`embed_text` = non-empty `action` + `reason` + `outcome` joined by newline."""
    settings = _settings(monkeypatch)
    with session(settings) as conn, transaction(conn):
        conn.execute(
            "INSERT INTO assistant_logs (action, reason, outcome) "
            "VALUES ('deploy', '', 'rolled out')"
        )
        row = conn.execute("SELECT * FROM assistant_logs WHERE id = 1").fetchone()
    assert LOGS_AREA.embed_text(row) == "deploy\nrolled out"


def test_enabled_areas_toggle(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default: `logs` in `enabled_areas`; `AREA_LOGS_ENABLED=false` — out of the registry."""
    settings = _settings(monkeypatch)
    assert "logs" in [spec.name for spec in enabled_areas(settings)]

    disabled = _settings(monkeypatch, AREA_LOGS_ENABLED="false")
    assert LOGS_AREA in ALL_AREAS  # the data does not vanish
    assert "logs" not in [spec.name for spec in enabled_areas(disabled)]


def test_area_queue_counts_logs(monkeypatch: pytest.MonkeyPatch) -> None:
    """A pending log is visible to the `areas` loop: the queue counter and `_areas_queue_empty`."""
    settings = _settings(monkeypatch)
    worker = BackgroundWorker(settings, HashEmbedder(DIM))
    assert worker._areas_queue_stat()["pending"] == 0

    with session(settings) as conn, transaction(conn):
        conn.execute(
            "INSERT INTO assistant_logs (action, reason) VALUES ('deploy', 'why')"
        )

    assert worker._areas_queue_stat()["pending"] == 1
    assert worker._areas_queue_empty() is False
