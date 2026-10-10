"""REST оператора: состояние «требует вмешательства» (№39, no39-02; arch §7.4).

`GET /jobs/attention` отдаёт замороженные задания (только SQL, без моделей),
`POST /jobs/attention/{job_id}/retry` возвращает задание в работу и сбрасывает
счётчик пары; вне внимания/без задания — 404. Внимание заданий — зона воркера:
в поверхность предупреждений (`warnings`, no12) ручка не пишет.
"""

from __future__ import annotations

import sqlite3
import time

from fastapi.testclient import TestClient

from app.config import get_settings
from app.services.worker import SLOT_CONFIG_META_KEY
from app.storage.db import session, transaction

AUTH = "Bearer test-secret-token"


def _headers() -> dict[str, str]:
    return {"Authorization": AUTH}


def _await_startup_reset(settings, timeout: float = 5.0) -> None:
    """Дождаться стартового сброса внимания (воркер пишет отпечаток в meta).

    Иначе задание, посеянное до `run()`, снимет фоновый стартовый сброс
    (`SLOT_ATTENTION_RESET_ON_START`) — гонка теста и жизни приложения.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        with session(settings) as conn:
            row = conn.execute(
                "SELECT 1 FROM meta WHERE key = ?", (SLOT_CONFIG_META_KEY,)
            ).fetchone()
        if row is not None:
            return
        time.sleep(0.02)
    raise AssertionError("worker startup did not record the slot fingerprint")


def _seed_attention(client: TestClient, *, note_id: int = 1) -> int:
    """Прямой INSERT: замороженная dedup-работа в очереди судьи."""
    settings = get_settings()
    _await_startup_reset(settings)
    client.app.state.worker._ensure_job_table()
    with session(settings) as conn, transaction(conn):
        cursor = conn.execute(
            "INSERT INTO worker_jobs (slot, kind, note_id, payload, attempts, "
            "last_outcome, attention_at) VALUES ('judge', 'dedup', ?, ?, 8, "
            "'off_contract', '2026-01-01T00:00:00Z')",
            (note_id, '{"candidate_id": 7}'),
        )
        return int(cursor.lastrowid or 0)


def test_list_and_retry(client: TestClient) -> None:
    job_id = _seed_attention(client)

    listing = client.get("/jobs/attention", headers=_headers())
    assert listing.status_code == 200
    body = listing.json()
    assert body["total"] == 1
    job = body["jobs"][0]
    assert job["job_id"] == job_id
    assert job["job"] == "judge"
    assert job["slot"] == "judge"
    assert job["outcome"] == "off_contract"
    assert job["attempts"] == 8
    assert job["since"] == "2026-01-01T00:00:00Z"

    retry = client.post(f"/jobs/attention/{job_id}/retry", headers=_headers())
    assert retry.status_code == 200
    assert retry.json() == {"job_id": job_id, "released": True, "attempts": 0}

    again = client.post(f"/jobs/attention/{job_id}/retry", headers=_headers())
    assert again.status_code == 404
    assert client.get("/jobs/attention", headers=_headers()).json()["total"] == 0


def test_retry_does_not_touch_warnings(client: TestClient) -> None:
    """Внимание заданий — не контент памяти: в `warnings` ручка не пишет."""
    settings = get_settings()
    job_id = _seed_attention(client)

    response = client.post(f"/jobs/attention/{job_id}/retry", headers=_headers())
    assert response.status_code == 200
    assert set(response.json()) == {"job_id", "released", "attempts"}
    with sqlite3.connect(settings.db_path) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        if "warnings" in tables:
            assert (
                conn.execute("SELECT COUNT(*) FROM warnings").fetchone()[0] == 0
            )
