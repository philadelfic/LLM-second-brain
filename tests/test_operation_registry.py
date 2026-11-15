"""The operation registry and the SLOT_MODEL flag (release 3.9.0, LSBN-S-5).

    The registry declares the ten operations with their contract (a fact about
    the code, not an operator switch); the `SLOT_MODEL` flag switches the env
    chain: `legacy` keeps "operation -> slot -> default" (and the /health
    operations block byte-for-byte), `operation` cuts the slot layer out
    ("operation -> default"). The per-operation thresholds are "base +
    override": unset -> the global base, set -> the override.

    The reference snapshot of /health.operations below was taken BEFORE the
    registry landed (the `legacy` mode must not change it).
"""

from __future__ import annotations

import json

import pytest
from fakes import FixedSummarizer, HashEmbedder, RecordingDedup

from app.config import Settings, get_settings
from app.services.llm_client import SlotSpec
from app.services.operations import (
    CONTRACTS,
    OPERATION_REGISTRY,
    breaker_key,
    operation_enabled,
    operation_mode,
    operations_health,
    pause_after,
    pause_sec,
    resolve_operation,
    settings_env_names,
    unknown_operation_warnings,
)
from app.services.worker import BackgroundWorker
from app.storage.db import init_db

REQUIRED_ENV: dict[str, str] = {
    "EMBEDDING_BASE_URL": "http://127.0.0.1:1",
    "SUMMARY_BASE_URL": "http://127.0.0.1:1",
    "SUMMARY_MODEL": "slot-summary-model",
    "JUDGE_BASE_URL": "http://127.0.0.1:1",
    "JUDGE_MODEL": "slot-judge-model",
    "MCP_AUTH_TOKEN": "test-secret-token",
}

# The reference /health.operations block (the `legacy` mode) - taken before the
# registry landed; the block must stay byte-for-byte the same.
LEGACY_OPERATIONS_HEALTH = """{
  "summarize": {
    "model": "slot-summary-model",
    "provider": "ollama",
    "source": "slot",
    "answer_budget": 35000,
    "reasoning_budget": 0,
    "sent_budget": 35000,
    "answer_source": "legacy",
    "reasoning_source": "legacy"
  },
  "title": {
    "model": "slot-summary-model",
    "provider": "ollama",
    "source": "slot",
    "answer_budget": 35000,
    "reasoning_budget": 0,
    "sent_budget": 35000,
    "answer_source": "legacy",
    "reasoning_source": "legacy"
  },
  "merge": {
    "model": "merge-only-model",
    "provider": "ollama",
    "source": "op",
    "answer_budget": 35000,
    "reasoning_budget": 0,
    "sent_budget": 35000,
    "answer_source": "legacy",
    "reasoning_source": "legacy"
  },
  "split": {
    "model": "slot-summary-model",
    "provider": "ollama",
    "source": "slot",
    "answer_budget": 35000,
    "reasoning_budget": 0,
    "sent_budget": 35000,
    "answer_source": "legacy",
    "reasoning_source": "legacy"
  },
  "classify": {
    "model": "slot-summary-model",
    "provider": "ollama",
    "source": "slot",
    "answer_budget": 768,
    "reasoning_budget": 0,
    "sent_budget": 768,
    "answer_source": "legacy",
    "reasoning_source": "legacy"
  },
  "describe": {
    "model": "slot-summary-model",
    "provider": "ollama",
    "source": "slot",
    "answer_budget": 64,
    "reasoning_budget": 0,
    "sent_budget": 64,
    "answer_source": "explicit",
    "reasoning_source": "legacy"
  },
  "judge": {
    "model": "slot-judge-model",
    "provider": "ollama",
    "source": "slot",
    "answer_budget": 256,
    "reasoning_budget": 0,
    "sent_budget": 256,
    "answer_source": "legacy",
    "reasoning_source": "legacy"
  },
  "structure_judge": {
    "model": "structure-x",
    "provider": "ollama",
    "source": "op",
    "answer_budget": 256,
    "reasoning_budget": 0,
    "sent_budget": 256,
    "answer_source": "legacy",
    "reasoning_source": "legacy"
  },
  "embed": {
    "model": "qwen3-embedding:8b",
    "provider": "ollama",
    "source": "default",
    "answer_budget": 35000,
    "reasoning_budget": 0,
    "sent_budget": 35000,
    "answer_source": "legacy",
    "reasoning_source": "legacy"
  }
}"""


def load(monkeypatch: pytest.MonkeyPatch, **overrides: str) -> Settings:
    """Build Settings from the required env + the overrides (a clean env)."""
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)
    env = {name.lower(): value for name, value in REQUIRED_ENV.items()}
    env.update({name.lower(): value for name, value in overrides.items()})
    for name, value in env.items():
        monkeypatch.setenv(name.upper(), value)
    get_settings.cache_clear()
    try:
        return get_settings()
    finally:
        get_settings.cache_clear()


# --- criterion 1: the registry covers the ten operations ---------------------


def test_registry_covers_ten_operations() -> None:
    """The registry: ten operations, none missing."""
    assert len(OPERATION_REGISTRY) == 10
    assert tuple(OPERATION_REGISTRY) == (
        "summarize",
        "title",
        "merge",
        "split",
        "classify",
        "describe",
        "judge",
        "structure_judge",
        "embed",
        "decision",
    )


def test_every_operation_has_a_contract_and_a_legacy_slot() -> None:
    """Each operation declares a contract of the closed set (B7) and the slot
    group it lives in today."""
    for name, spec in OPERATION_REGISTRY.items():
        assert spec.name == name
        assert spec.contract in CONTRACTS, name
        assert spec.legacy_slot, name
        assert spec.fields, name


def test_contracts_are_the_closed_set_of_the_design() -> None:
    """The contract set and the contract of every operation (the design, B7)."""
    assert CONTRACTS == {
        "text",
        "marker",
        "label",
        "probability",
        "parts-by-delimiter",
    }
    assert {
        name: spec.contract for name, spec in OPERATION_REGISTRY.items()
    } == {
        "summarize": "text",
        "title": "text",
        "merge": "parts-by-delimiter",
        "split": "parts-by-delimiter",
        "classify": "label",
        "describe": "text",
        "judge": "marker",
        "structure_judge": "marker",
        "embed": "text",
        "decision": "probability",
    }


def test_legacy_slot_of_every_operation() -> None:
    """The slot group of every operation (the `legacy` mode)."""
    assert {
        name: spec.legacy_slot for name, spec in OPERATION_REGISTRY.items()
    } == {
        "summarize": "summary",
        "title": "summary",
        "merge": "summary",
        "split": "summary",
        "classify": "summary",
        "describe": "summary",
        "judge": "judge",
        "structure_judge": "judge",
        "embed": "embedding",
        "decision": "decision",
    }


# --- criterion 2: the legacy /health.operations block is unchanged ------------


def test_legacy_health_snapshot_is_byte_identical(monkeypatch) -> None:
    """The /health.operations block in the `legacy` mode is byte-for-byte the
    reference (taken before the registry landed)."""
    settings = load(
        monkeypatch,
        MERGE_MODEL="merge-only-model",
        SUMMARY_TIMEOUT_SEC="850",
        DESCRIBE_NUM_PREDICT="64",
        STRUCTURE_JUDGE_MODEL="structure-x",
        EMBED_TIMEOUT_SEC="600",
    )
    assert settings.slot_model == "legacy"
    block = operations_health(settings)
    assert json.dumps(block, indent=2, ensure_ascii=False) == LEGACY_OPERATIONS_HEALTH


# --- criterion 3: the `operation` mode cuts the slot layer out ---------------


def test_operation_mode_reads_the_operation_env(monkeypatch) -> None:
    """In the `operation` mode the value comes from `<OP>_*`, not from the
    slot group."""
    settings = load(
        monkeypatch,
        SLOT_MODEL="operation",
        MERGE_MODEL="merge-only-model",
        SUMMARY_MODEL="slot-summary-model",
    )
    assert operation_mode(settings) is True
    values, sources = resolve_operation("merge", settings)
    assert values["model"] == "merge-only-model"
    assert sources["model"] == "op"
    spec, sources = SlotSpec.for_operation("merge", settings)
    assert spec.model == "merge-only-model"
    assert spec.model_source == "op"
    assert spec.name == "merge"


def test_operation_mode_does_not_inherit_the_slot(monkeypatch) -> None:
    """In the `operation` mode an unset field is NOT taken from the slot group:
    its source is `default` and the slot value is not consulted."""
    settings = load(
        monkeypatch,
        SLOT_MODEL="operation",
        SUMMARY_MODEL="slot-summary-model",
        SUMMARY_TIMEOUT_SEC="850",
    )
    _values, sources = resolve_operation("merge", settings)
    assert sources["model"] == "default"
    assert sources["timeout_sec"] == "default"
    assert "slot" not in set(sources.values())
    spec, _sources = SlotSpec.for_operation("merge", settings)
    assert spec.model != "slot-summary-model"
    assert spec.read_timeout != 850.0


def test_legacy_mode_keeps_the_slot_layer(monkeypatch) -> None:
    """The `legacy` mode is unchanged: "operation -> slot -> default"."""
    settings = load(
        monkeypatch,
        SUMMARY_MODEL="slot-summary-model",
        SUMMARY_TIMEOUT_SEC="850",
    )
    assert operation_mode(settings) is False
    _values, sources = resolve_operation("merge", settings)
    assert sources["model"] == "slot"
    assert sources["timeout_sec"] == "slot"
    spec, _sources = SlotSpec.for_operation("merge", settings)
    assert spec.model == "slot-summary-model"
    assert spec.read_timeout == 850.0


def test_breaker_key_follows_the_mode(monkeypatch) -> None:
    """The breaker key: the slot group in the `legacy` mode, the operation
    name in the `operation` mode."""
    legacy = load(monkeypatch)
    assert breaker_key("merge", legacy) == "summary"
    assert breaker_key("judge", legacy) == "judge"
    assert breaker_key("embed", legacy) == "embedding"
    operation = load(monkeypatch, SLOT_MODEL="operation")
    assert breaker_key("merge", operation) == "merge"
    assert breaker_key("judge", operation) == "judge"
    assert breaker_key("decision", operation) == "decision"


# --- criterion 4: the per-operation thresholds (base + override) --------------


def test_pause_thresholds_fall_back_to_the_global_base(monkeypatch) -> None:
    """Unset per-operation thresholds -> the global base."""
    settings = load(monkeypatch, SLOT_PAUSE_AFTER="5", SLOT_PAUSE_SEC="120")
    assert pause_after("merge", settings) == 5
    assert pause_sec("merge", settings) == 120
    assert settings.merge_pause_after is None
    assert settings.merge_pause_sec is None


def test_pause_thresholds_override_the_base(monkeypatch) -> None:
    """A set per-operation threshold beats the base - for that operation
    only."""
    settings = load(
        monkeypatch,
        SLOT_PAUSE_AFTER="5",
        SLOT_PAUSE_SEC="120",
        MERGE_PAUSE_AFTER="2",
        MERGE_PAUSE_SEC="60",
    )
    assert pause_after("merge", settings) == 2
    assert pause_sec("merge", settings) == 60
    assert pause_after("title", settings) == 5
    assert pause_sec("title", settings) == 120


def test_operation_enabled_switch(monkeypatch) -> None:
    """The enable switch of an operation: unset -> on; false -> off."""
    settings = load(monkeypatch)
    assert operation_enabled("summarize", settings) is True
    off = load(monkeypatch, MERGE_ENABLED="false")
    assert operation_enabled("merge", off) is False
    # The decision operation reuses the F-12 switch (off by default).
    assert operation_enabled("decision", settings) is False
    on = load(
        monkeypatch,
        DECISION_ENABLED="true",
        DECISION_BASE_URL="http://127.0.0.1:1",
        DECISION_MODEL="decision-model",
    )
    assert operation_enabled("decision", on) is True


# --- criterion 5: no false WARN about an unknown operation -------------------


def _new_operation_env() -> dict[str, str]:
    """The env of every new operation field (`<OP>_*`)."""
    env: dict[str, str] = {}
    for name in OPERATION_REGISTRY:
        env[f"{name.upper()}_MODEL"] = "operation-model"
        env[f"{name.upper()}_PAUSE_AFTER"] = "2"
        env[f"{name.upper()}_PAUSE_SEC"] = "60"
        if name != "decision":  # decision reuses the F-12 switch
            env[f"{name.upper()}_ENABLED"] = "true"
    return env


def test_new_operation_env_does_not_warn(monkeypatch) -> None:
    """No new env name produces a false WARN `unknown_operation`."""
    env = _new_operation_env()
    settings = load(monkeypatch, **env)
    warnings = unknown_operation_warnings(settings)
    assert [
        text for _code, text in warnings if any(name in text for name in env)
    ] == [], warnings


def test_typo_in_a_new_operation_env_still_warns(monkeypatch) -> None:
    """A typo of a new field is still a typo candidate."""
    load(monkeypatch, MERG_PAUSE_AFTER="2")
    warnings = unknown_operation_warnings()
    assert [text for _code, text in warnings if "MERG_PAUSE_AFTER" in text], warnings


def test_settings_env_names_contain_the_new_fields() -> None:
    """The Settings fields are the authoritative list of env names (3.8.1) -
    the new fields are in it."""
    names = settings_env_names()
    assert "SLOT_MODEL" in names
    for name in OPERATION_REGISTRY:
        assert f"{name.upper()}_PAUSE_AFTER" in names
        assert f"{name.upper()}_PAUSE_SEC" in names
        if name != "decision":
            assert f"{name.upper()}_ENABLED" in names


# --- the worker: the breaker key follows the mode ----------------------------


def test_worker_breakers_are_keyed_by_the_mode(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`_breakers`: one breaker per slot group (`legacy`) or one per operation
    (`operation`)."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "notes.db"))
    monkeypatch.setenv("SLOT_MODEL", "legacy")
    get_settings.cache_clear()
    legacy = get_settings()
    init_db(legacy)
    worker = BackgroundWorker(
        legacy,
        HashEmbedder(8),
        FixedSummarizer("F."),
        RecordingDedup(),
    )
    assert set(worker._breakers) == {"embedding", "summary", "judge"}

    monkeypatch.setenv("SLOT_MODEL", "operation")
    monkeypatch.setenv("MERGE_PAUSE_AFTER", "2")
    get_settings.cache_clear()
    operation = get_settings()
    worker = BackgroundWorker(
        operation,
        HashEmbedder(8),
        FixedSummarizer("F."),
        RecordingDedup(),
    )
    assert set(worker._breakers) == set(OPERATION_REGISTRY) - {"decision"}
    # The `decision` operation's breaker lives in the SERVICE (arch §13.8) —
    # the worker keeps no second breaker for it (release 3.9.0, LSBN-S-13).
    assert "decision" not in worker._breakers
    # The per-operation threshold reached the breaker of the operation; the
    # unset per-operation pause falls back to the global base.
    assert worker._breakers["merge"]._open_after == 2
    assert worker._breakers["merge"]._pause_sec == operation.slot_pause_sec
    assert worker._breakers["title"]._open_after == operation.slot_pause_after
    get_settings.cache_clear()
