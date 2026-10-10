"""REST оператора при выключенной области (№40, постановка no40-02, §13.10).

Выключение области скрывает её ТОЛЬКО от агента (§7.1: MCP-ручки, инструкция,
карта, анонс, фон). Оператор — тот, кто включает и выключает: при любом
значении флага его REST-ручки области работают, а история сущностей области
остаётся доступной (решение владельца 2026-10-08: «не видна совсем» относится
к агенту; гейт `AREA_*_ENABLED` на операторскую поверхность не распространяется).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import create_app
from app.storage.db import init_db


def _headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_operator_rest_kept(monkeypatch: pytest.MonkeyPatch, token: str) -> None:
    """AREA_TERMS_ENABLED=false: ручки `terms` и история термина у оператора живы."""
    monkeypatch.setenv("AREA_TERMS_ENABLED", "false")
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    assert settings.area_terms_enabled is False  # флаг области действительно снят

    with TestClient(create_app(settings)) as client:
        created = client.post(
            "/terms",
            json={
                "term": "ГЗ",
                "context": "студенты МГУ",
                "definition": "государственный экзамен",
            },
            headers=_headers(token),
        )
        assert created.status_code == 201, created.text
        term_id = created.json()["id"]

        # Ручка области отвечает оператору и отдаёт данные (не 404/403).
        read = client.get(f"/terms/{term_id}", headers=_headers(token))
        assert read.status_code == 200, read.text
        assert read.json()["term"] == "ГЗ"

        # История сущности выключенной области тоже доступна оператору (§13.10).
        listing = client.get(f"/history/term/{term_id}", headers=_headers(token))
        assert listing.status_code == 200, listing.text
        body = listing.json()
        assert body["entity"] == "term" and body["id"] == term_id
        assert body["total"] >= 1 and body["versions"]
