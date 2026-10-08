"""`/health.queues` несёт ожидание и внимание (№39, 3.2.0; arch §7.2).

Очередь отдаёт `pending`/`oldest_pending_sec` и новые счётчики `waiting`
(отложено слотом/готовностью) и `attention` (ждёт человека). Собирается
только SQL — моделей не зовём.
"""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_queues_have_waiting_attention(client: TestClient, token: str) -> None:
    """Каждая очередь /health несёт waiting/attention; pending виден по заметке."""
    client.post(
        "/notes",
        json={"text": "заметка для наблюдаемости очередей", "title": "Очередь №39"},
        headers={"Authorization": f"Bearer {token}"},
    )
    body = client.get("/health").json()
    for stat in body["queues"].values():
        assert "waiting" in stat and "attention" in stat
        assert stat["waiting"] >= 0 and stat["attention"] >= 0
    # Заметка ждёт кодировщик: работа видна в pending своей очереди.
    assert body["queues"]["vector"]["pending"] == body["pending_vector"]
    assert body["queues"]["vector"]["waiting"] >= 0
