"""`/health.slots` — причина и пауза слота (№39, 3.2.0; arch §7.2).

По каждому слоту (`embedding`/`summary`/`judge`) `/health` отдаёт
`{ok, last_outcome, paused_until, attention}`: исход последней попытки, исход
последнего отказа, паузу разрыва и число заданий в внимании. Сбор — только
SQL/память: `/health` не вызывает модели (иначе опрос здоровья будил бы слоты).
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.services.slot_outcome import SlotOutcome

SLOTS = ("embedding", "summary", "judge")


def test_slots_block(client: TestClient) -> None:
    """/health.slots несёт все слоты с полным набором полей."""
    body = client.get("/health").json()
    assert set(body["slots"]) == set(SLOTS)
    for stat in body["slots"].values():
        assert set(stat) == {"ok", "last_outcome", "paused_until", "attention"}
        assert stat["paused_until"] is None  # разрыва нет
        assert stat["attention"] == 0


def test_slots_block_paused_until_when_break_open(client: TestClient) -> None:
    """Открытый разрыв слота: last_outcome и paused_until заполнены."""
    worker = client.app.state.worker  # type: ignore[attr-defined]
    worker._breakers["embedding"].note_failure(SlotOutcome.AUTH)

    stat = client.get("/health").json()["slots"]["embedding"]
    assert stat["last_outcome"] == "auth"
    assert stat["paused_until"] is not None
    # Остальные слоты не затронуты — недоступность одного слота не залипает.
    assert client.get("/health").json()["slots"]["judge"]["paused_until"] is None


def test_health_no_model_calls(client: TestClient, monkeypatch) -> None:
    """Сбор /health не вызывает модели: счётчик заглушек остаётся нулевым."""
    services = client.app.state.services  # type: ignore[attr-defined]
    calls = {"n": 0}

    def spy(*args, **kwargs):
        calls["n"] += 1
        raise AssertionError("модель вызвана при сборе /health")

    for owner, name in (
        (services.embedding, "embed"),
        (services.embedding, "embed_texts"),
        (services.summary, "summarize"),
        (services.summary, "title"),
        (services.judge, "judge"),
        (services.judge, "probe"),
    ):
        monkeypatch.setattr(owner, name, spy, raising=False)

    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert set(body["slots"]) == set(SLOTS)
    assert calls["n"] == 0
