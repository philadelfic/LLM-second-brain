"""The operator's REST with an area disabled (#40, brief no40-02, §13.10).

    Disabling an area hides it from the AGENT only (§7.1: MCP handles, the
    instruction, the map, the announce, the background). The operator is the one
    who enables and disables: at any flag value the area's REST handles work for
    them, and the area's entity history stays reachable (the owner's 2026-10-08
    decision: "invisible entirely" refers to the agent; the `AREA_*_ENABLED` gate
    does not extend to the operator surface).
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
    """AREA_TERMS_ENABLED=false: the `terms` handles and the term history stay alive for the operator."""
    monkeypatch.setenv("AREA_TERMS_ENABLED", "false")
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    assert settings.area_terms_enabled is False  # the area flag is indeed off

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

        # The area's handle answers the operator and serves the data (not 404/403).
        read = client.get(f"/terms/{term_id}", headers=_headers(token))
        assert read.status_code == 200, read.text
        assert read.json()["term"] == "ГЗ"

        # A disabled area's entity history is available to the operator too (§13.10).
        listing = client.get(f"/history/term/{term_id}", headers=_headers(token))
        assert listing.status_code == 200, listing.text
        body = listing.json()
        assert body["entity"] == "term" and body["id"] == term_id
        assert body["total"] >= 1 and body["versions"]
