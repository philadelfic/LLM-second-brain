"""#12 (no12-02, 3.2.0): the warnings REST surface (arch §7.1/§13.4).

    `GET /warnings` serves the FULL contract (no whitelist trim — like all
    operator handles) with the `WARNINGS_MAX_LIMIT_REST` ceiling; `POST
    /warnings/{id}/dismiss` clears a warning by a human; a repeat/without the
    record — 404. Reads — SQL only, no model call.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.config import get_settings
from app.services.warnings import (
    KIND_NOTE_MISPLACED,
    KIND_NOTE_NO_NODE,
    KIND_TERM_CONTEXTS_EXCESS,
)

AUTH = "Bearer test-secret-token"


def _headers() -> dict[str, str]:
    return {"Authorization": AUTH}


def _seed(
    client: TestClient,
    *,
    kind: str = KIND_NOTE_MISPLACED,
    subject_type: str = "note",
    subject_id: int = 1,
) -> int:
    """Open a warning by the service — SQL, no model (a seed for reading)."""
    service = client.app.state.services.warnings  # type: ignore[attr-defined]
    return service.open(kind, subject_type, subject_id, "misplaced", {"a": 1})


def test_get_warnings_full_contract(client: TestClient) -> None:
    """`GET /warnings` — the row's full contract + the page fields."""
    _seed(client, subject_id=1)
    _seed(client, kind=KIND_NOTE_NO_NODE, subject_id=2)
    body = client.get("/warnings", headers=_headers()).json()
    assert body["total"] == 2 and len(body["warnings"]) == 2
    assert {"total", "has_more", "next_offset"} <= set(body)
    assert body["has_more"] is False and body["next_offset"] is None
    item = body["warnings"][0]
    assert set(item) == {
        "id", "kind", "subject_type", "subject_id", "message", "payload",
        "status", "created_at", "updated_at", "resolved_at",
    }
    assert item["subject_type"] == "note"
    assert item["status"] == "open"
    assert item["payload"] == {"a": 1}


def test_rest_ceiling(client: TestClient) -> None:
    """The REST surface's ceiling (`WARNINGS_MAX_LIMIT_REST`): above — 422."""
    settings = get_settings()
    _seed(client, subject_id=1)
    full = client.get(
        "/warnings",
        params={"limit": settings.warnings_max_limit_rest},
        headers=_headers(),
    )
    assert full.status_code == 200 and full.json()["total"] == 1
    over = client.get(
        "/warnings",
        params={"limit": settings.warnings_max_limit_rest + 1},
        headers=_headers(),
    )
    assert over.status_code == 422


def test_dismiss(client: TestClient) -> None:
    """`POST /warnings/{id}/dismiss` → dismissed; a repeat — 404."""
    warning_id = _seed(client, subject_id=1)
    first = client.post(f"/warnings/{warning_id}/dismiss", headers=_headers())
    assert first.status_code == 200
    body = first.json()
    assert body["id"] == warning_id
    assert body["status"] == "dismissed"
    assert body["resolved_at"] is not None

    again = client.post(f"/warnings/{warning_id}/dismiss", headers=_headers())
    assert again.status_code == 404

    missing = client.post("/warnings/9999/dismiss", headers=_headers())
    assert missing.status_code == 404


def test_dismissed_absent_from_open(client: TestClient) -> None:
    """A cleared one does not come at `status=open` but shows at `status=dismissed`."""
    warning_id = _seed(
        client,
        kind=KIND_TERM_CONTEXTS_EXCESS,
        subject_type="term",
        subject_id=5,
    )
    client.post(f"/warnings/{warning_id}/dismiss", headers=_headers())

    opened = client.get("/warnings", headers=_headers()).json()
    assert opened["total"] == 0 and opened["warnings"] == []

    dismissed = client.get(
        "/warnings", params={"status": "dismissed"}, headers=_headers()
    ).json()
    assert dismissed["total"] == 1
    assert dismissed["warnings"][0]["id"] == warning_id
