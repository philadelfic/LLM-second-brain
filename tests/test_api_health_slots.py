"""`/health.slots` — the slot's cause and pause (#39, 3.2.0; arch §7.2).

    Per slot (`embedding`/`summary`/`judge`) `/health` serves
    `{ok, last_outcome, paused_until, attention}`: the last attempt's outcome, the
    last failure's outcome, the breaker's pause and the tasks in attention. The
    assembly — SQL/memory only: `/health` calls no models (otherwise a health
    poll would wake the slots).
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.services.llm_client import ChatMeta, ChatResult
from app.services.slot_outcome import SlotOutcome

SLOTS = ("embedding", "summary", "judge", "decision")


def make_meta(**over: object) -> ChatMeta:
    """One chat call's metadata for checking the /health.slots diagnostics."""
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
    """/health.slots carries all slots with the full field set (#39 + #49)."""
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
            # LSBN-T-33: the last call's separate budgets.
            "last_answer_budget",
            "last_reasoning_budget",
            "last_sent_budget",
        }
        assert stat["paused_until"] is None  # no breaker
        assert stat["attention"] == 0


def test_slots_block_paused_until_when_break_open(client: TestClient) -> None:
    """An open slot breaker: last_outcome and paused_until filled."""
    worker = client.app.state.worker  # type: ignore[attr-defined]
    worker._breakers["embedding"].note_failure(SlotOutcome.AUTH)

    stat = client.get("/health").json()["slots"]["embedding"]
    assert stat["last_outcome"] == "auth"
    assert stat["paused_until"] is not None
    # The other slots untouched — one slot's unavailability does not stick.
    assert client.get("/health").json()["slots"]["judge"]["paused_until"] is None


def test_last_call_diagnostics(client: TestClient, monkeypatch) -> None:
    """/health.slots carries the slot's last-call diagnostics (lsbdef-0008)."""
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
    """Before a slot's first call the diagnostic fields are None — health does not lie."""
    for stat in client.get("/health").json()["slots"].values():
        assert stat["last_finish_reason"] is None
        assert stat["last_budget"] is None
        assert stat["last_budget_source"] is None
        assert stat["last_had_thinking"] is None


def test_health_no_model_calls(client: TestClient, monkeypatch) -> None:
    """Assembling /health calls no models: the stubs' counter stays zero."""
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


OPERATIONS = (
    "summarize",
    "title",
    "merge",
    "split",
    "classify",
    "describe",
    "judge",
    "structure_judge",
    "embed",
)


def test_operations_block(client: TestClient) -> None:
    """/health.operations (LSBN-F-34): the model/provider/source per operation."""
    body = client.get("/health").json()
    assert set(body["operations"]) == set(OPERATIONS)
    for snapshot in body["operations"].values():
        assert set(snapshot) == {
            "model",
            "provider",
            "source",
            # LSBN-T-33: the operation's separate budgets.
            "answer_budget",
            "reasoning_budget",
            "sent_budget",
            "answer_source",
            "reasoning_source",
        }
        assert snapshot["source"] in ("op", "slot", "default")
        assert snapshot["model"]  # every operation has a model
    # Without overrides summarize/title/merge inherit the summary slot.
    assert body["operations"]["summarize"]["model"] == "test-summary-model"
    assert body["operations"]["summarize"]["source"] == "slot"
    assert body["operations"]["judge"]["model"] == "test-judge-model"


def test_operations_block_has_no_addresses(
    client: TestClient, monkeypatch, tmp_path
) -> None:
    """The brief's decision #4: /health prints no slot or key addresses —
            even with an operation override (the base_url is known to the journal
            only).
    """
    from app.config import get_settings
    from app.main import create_app

    monkeypatch.setenv("MERGE_MODEL", "merge-x")
    monkeypatch.setenv("MERGE_BASE_URL", "http://merge-private:11434")
    monkeypatch.setenv("MERGE_API_KEY", "sk-merge-test")
    monkeypatch.setenv("DB_PATH", str(tmp_path / "notes-operations.db"))
    get_settings.cache_clear()
    try:
        with TestClient(create_app()) as fresh:
            fresh.app.state.services.notes.set_summary_notifier(None)
            fresh.app.state.services.notes.set_vector_notifier(None)
            raw = fresh.get("/health").text
            body = fresh.get("/health").json()
    finally:
        get_settings.cache_clear()
    assert body["operations"]["merge"]["model"] == "merge-x"
    assert body["operations"]["merge"]["source"] == "op"
    assert "merge-private" not in raw
    assert "sk-merge-test" not in raw
    assert "base_url" not in raw
