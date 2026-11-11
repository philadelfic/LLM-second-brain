"""`/health.queues` carries the wait and the attention (#39, 3.2.0; arch §7.2).

    A queue serves `pending`/`oldest_pending_sec` and the new `waiting` counters
    (deferred by the slot/readiness) and `attention` (waiting for a human).
    Assembled by SQL only — no models called.
"""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_queues_have_waiting_attention(client: TestClient, token: str) -> None:
    """Every /health queue carries waiting/attention; pending is visible per note."""
    client.post(
        "/notes",
        json={"text": "заметка для наблюдаемости очередей", "title": "Очередь №39"},
        headers={"Authorization": f"Bearer {token}"},
    )
    body = client.get("/health").json()
    for stat in body["queues"].values():
        assert "waiting" in stat and "attention" in stat
        assert stat["waiting"] >= 0 and stat["attention"] >= 0
    # The note waits for the encoder: the work is visible in its queue's pending.
    assert body["queues"]["vector"]["pending"] == body["pending_vector"]
    assert body["queues"]["vector"]["waiting"] >= 0
