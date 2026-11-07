"""The backups operator handles (#14, 3.2.0, arch §7.2/§7.3).

    `GET /backups` — the snapshot list `{items, total}` (newest first; the item
    shape agreed); `POST /backups/verify` — on-demand verification (the default =
    the newest, an explicit `file`, no such file → 404 "no such snapshot"); the
    `backup_verify_ok`/`backup_verify_failed` event is written with the agreed
    fields.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.services.backup import BackupService, VERIFY_MARKER_SUFFIX


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _backup(client: TestClient) -> BackupService:
    return client.app.state.services.backup  # type: ignore[no-any-return]


def _with_event(caplog: pytest.LogCaptureFixture, event: str) -> list[Any]:
    return [
        record for record in caplog.records if getattr(record, "event", None) == event
    ]


def test_backups_list(client: TestClient, token: str) -> None:
    """GET /backups: items with the agreed shape, newest first, total."""
    backup = _backup(client)
    backup_dir = Path(backup._settings.backup_dir)
    older = backup_dir / "notes-20200101T000000Z.db"
    newer = backup_dir / "notes-20200202T000000Z.db"
    older.write_bytes(b"x" * 10)
    newer.write_bytes(b"x" * 20)
    response = client.get("/backups", headers=_auth(token))
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == len(body["items"])
    items = {item["file"]: item for item in body["items"]}
    assert set(items["notes-20200202T000000Z.db"]) == {
        "file",
        "size_bytes",
        "created_at",
        "verified",
        "bad",
        "shipped",
    }
    assert items["notes-20200101T000000Z.db"]["size_bytes"] == 10
    assert items["notes-20200202T000000Z.db"]["created_at"] == "2020-02-02T00:00:00Z"
    assert items["notes-20200202T000000Z.db"]["verified"] == "unknown"
    assert items["notes-20200202T000000Z.db"]["bad"] is False
    assert items["notes-20200202T000000Z.db"]["shipped"] is False
    names = [item["file"] for item in body["items"]]
    assert names.index("notes-20200202T000000Z.db") < names.index(
        "notes-20200101T000000Z.db"
    )


def test_backups_list_marks_bad(client: TestClient, token: str) -> None:
    """A marked snapshot shows as verified=failed, bad=True."""
    backup = _backup(client)
    backup_dir = Path(backup._settings.backup_dir)
    marked = backup_dir / "notes-20200303T000000Z.db"
    marked.write_bytes(b"x" * 5)
    marked.with_name(marked.name + VERIFY_MARKER_SUFFIX).write_text(
        "{}", encoding="utf-8"
    )
    body = client.get("/backups", headers=_auth(token)).json()
    item = next(i for i in body["items"] if i["file"] == marked.name)
    assert item["verified"] == "failed"
    assert item["bad"] is True


def test_backups_verify_on_demand(client: TestClient, token: str) -> None:
    """POST /backups/verify: an explicit file and the default (the newest); no file → 404."""
    backup = _backup(client)
    target = backup.snapshot()
    explicit = client.post(
        "/backups/verify", json={"file": target.name}, headers=_auth(token)
    )
    assert explicit.status_code == 200
    body = explicit.json()
    assert body["ok"] is True
    assert body["file"] == target.name
    assert body["checks"]["integrity"] == "ok"
    # The default (no/empty file) — the directory's newest snapshot.
    default = client.post("/backups/verify", json={}, headers=_auth(token))
    assert default.status_code == 200
    assert default.json()["ok"] is True
    # A nonexistent file — 404 with the agreed text.
    missing = client.post(
        "/backups/verify",
        json={"file": "notes-19700101T000000Z.db"},
        headers=_auth(token),
    )
    assert missing.status_code == 404
    assert missing.json()["detail"] == "no such snapshot"


def test_backup_verify_event(
    client: TestClient, token: str, caplog: pytest.LogCaptureFixture
) -> None:
    """The backup_verify_failed event carries file/reason/checks/latency_ms."""
    backup = _backup(client)
    backup_dir = Path(backup._settings.backup_dir)
    broken = backup_dir / "notes-20190101T000000Z.db"
    broken.write_bytes(b"garbage-not-a-database" * 4)
    with caplog.at_level(logging.WARNING, logger="app"):
        response = client.post(
            "/backups/verify", json={"file": broken.name}, headers=_auth(token)
        )
    assert response.status_code == 200
    assert response.json()["ok"] is False
    records = _with_event(caplog, "backup_verify_failed")
    assert records
    record = records[-1]
    assert record.file == broken.name  # type: ignore[attr-defined]
    assert record.reason in {"not_openable", "integrity_failed"}  # type: ignore[attr-defined]
    assert isinstance(record.checks, dict)  # type: ignore[attr-defined]
    assert record.latency_ms >= 0  # type: ignore[attr-defined]


def test_backup_verify_ok_event(
    client: TestClient, token: str, caplog: pytest.LogCaptureFixture
) -> None:
    """The backup_verify_ok event is written for a good snapshot."""
    backup = _backup(client)
    target = backup.snapshot()
    with caplog.at_level(logging.INFO, logger="app"):
        client.post("/backups/verify", json={"file": target.name}, headers=_auth(token))
    records = _with_event(caplog, "backup_verify_ok")
    assert records
    assert records[-1].file == target.name  # type: ignore[attr-defined]
