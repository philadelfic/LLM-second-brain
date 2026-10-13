"""Реестр области «Логи ассистента» (LSBN-F-10, f10-01): AreaSpec, флаг, очередь.

`LOGS_AREA` зарегистрирована в `ALL_AREAS` и подхватывается `enabled_areas`
(№40) без нового кода; флаг `area_logs_enabled` гейтит фон и поверхность, но не
данные — выключенная область остаётся в `ALL_AREAS`. Петля `areas` воркера
видит pending-логи той же механикой, что и остальные области (arch §5.3/§6.2).
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
    """Настройки области логов (флаг снят → дефолт true), схема создана."""
    monkeypatch.setenv("EMBEDDING_DIM", str(DIM))
    monkeypatch.delenv(_LOGS_FLAG, raising=False)
    for name, value in flags.items():
        monkeypatch.setenv(name, value)
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    clear_seeded_skills(settings)  # сид skill-создателя не мешает счёту
    return settings


def test_logs_area_registered() -> None:
    """`LOGS_AREA` в `ALL_AREAS` и её спецификация совпадает с контрактом §6.2."""
    assert LOGS_AREA in ALL_AREAS
    assert LOGS_AREA.name == "logs"
    assert LOGS_AREA.settings_flag == "area_logs_enabled"
    assert LOGS_AREA.table == "assistant_logs"
    assert LOGS_AREA.fts_table == "assistant_logs_fts"
    assert LOGS_AREA.vec_table == "assistant_logs_vec"
    assert LOGS_AREA.vec_id_column == "log_id"
    assert LOGS_AREA.embed_fields == ("action", "reason", "outcome")


def test_logs_embed_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """`embed_text` = непустые `action` + `reason` + `outcome` через перевод строки."""
    settings = _settings(monkeypatch)
    with session(settings) as conn, transaction(conn):
        conn.execute(
            "INSERT INTO assistant_logs (action, reason, outcome) "
            "VALUES ('deploy', '', 'rolled out')"
        )
        row = conn.execute("SELECT * FROM assistant_logs WHERE id = 1").fetchone()
    assert LOGS_AREA.embed_text(row) == "deploy\nrolled out"


def test_enabled_areas_toggle(monkeypatch: pytest.MonkeyPatch) -> None:
    """Дефолт: `logs` в `enabled_areas`; `AREA_LOGS_ENABLED=false` — только из реестра."""
    settings = _settings(monkeypatch)
    assert "logs" in [spec.name for spec in enabled_areas(settings)]

    disabled = _settings(monkeypatch, AREA_LOGS_ENABLED="false")
    assert LOGS_AREA in ALL_AREAS  # данные не исчезают
    assert "logs" not in [spec.name for spec in enabled_areas(disabled)]


def test_area_queue_counts_logs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pending-лог виден петле `areas`: счётчик очереди и `_areas_queue_empty`."""
    settings = _settings(monkeypatch)
    worker = BackgroundWorker(settings, HashEmbedder(DIM))
    assert worker._areas_queue_stat()["pending"] == 0

    with session(settings) as conn, transaction(conn):
        conn.execute(
            "INSERT INTO assistant_logs (action, reason) VALUES ('deploy', 'why')"
        )

    assert worker._areas_queue_stat()["pending"] == 1
    assert worker._areas_queue_empty() is False
