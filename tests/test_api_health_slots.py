"""`/health.slots` — причина и пауза слота (№39, 3.2.0; arch §7.2).

По каждому слоту (`embedding`/`summary`/`judge`) `/health` отдаёт
`{ok, last_outcome, paused_until, attention}`: исход последней попытки, исход
последнего отказа, паузу разрыва и число заданий в внимании. Сбор — только
SQL/память: `/health` не вызывает модели (иначе опрос здоровья будил бы слоты).
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.services.llm_client import ChatMeta, ChatResult
from app.services.slot_outcome import SlotOutcome

SLOTS = ("embedding", "summary", "judge", "decision")


def make_meta(**over: object) -> ChatMeta:
    """Метаданные одного вызова чата для проверки диагностики /health.slots."""
    base: dict[str, object] = {
        "slot": "summary",
        "provider": "ollama",
        "model": "m",
        "operation": "summary",
        "model_type": "generation",
        "budget": 4321,
        "budget_source": "context",
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "reasoning_tokens": None,
        "had_thinking": True,
        "thinking_chars": 12,
        "finish_reason": "length",
        "truncated": True,
        "duration_ms": 1,
    }
    base.update(over)
    return ChatMeta(**base)  # type: ignore[arg-type]


def test_slots_block(client: TestClient) -> None:
    """/health.slots несёт все слоты с полным набором полей (№39 + №49)."""
    body = client.get("/health").json()
    assert set(body["slots"]) == set(SLOTS)
    for stat in body["slots"].values():
        assert set(stat) == {
            "ok",
            "last_outcome",
            "paused_until",
            "attention",
            "last_finish_reason",
            "last_budget",
            "last_budget_source",
            "last_had_thinking",
        }
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


def test_last_call_diagnostics(client: TestClient, monkeypatch) -> None:
    """/health.slots несёт диагностику последнего вызова слота (lsbdef-0008)."""
    services = client.app.state.services  # type: ignore[attr-defined]
    meta = make_meta()
    monkeypatch.setattr(
        services.summary._llm,
        "chat_result",
        lambda *a, **k: ChatResult(content="кратко", meta=meta),
    )
    monkeypatch.setattr(
        services.judge._llm,
        "chat_result",
        lambda *a, **k: ChatResult(content="**DUPLICATE**", meta=meta),
    )
    services.summary.summarize("текст заметки для диагностики слота")
    assert services.judge.judge("новый текст", "кандидат").verdict is True

    slots = client.get("/health").json()["slots"]
    for name in ("summary", "judge"):
        stat = slots[name]
        assert stat["last_finish_reason"] == "length"
        assert stat["last_budget"] == 4321
        assert stat["last_budget_source"] == "context"
        assert stat["last_had_thinking"] is True


def test_diagnostics_none_before_first_call(client: TestClient) -> None:
    """До первого вызова слотов поля диагностики None — health не врёт."""
    for stat in client.get("/health").json()["slots"].values():
        assert stat["last_finish_reason"] is None
        assert stat["last_budget"] is None
        assert stat["last_budget_source"] is None
        assert stat["last_had_thinking"] is None


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
