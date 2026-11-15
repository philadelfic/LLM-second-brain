"""The env migration to the `<OP>_*` names (release 3.9.0, LSBN-S-16).

    The old slot-group names (`SUMMARY_*`, `EMBEDDING_*`) do not act in the
    `operation` mode: the slot layer is out of the chain there, so the startup
    WARNs and names the equivalent operation env names - a warning, NOT a
    fatal error (the service starts). In the `legacy` mode the old names act as
    before and nothing is warned about.

    The boundaries of the step:
    - `judge` and `decision` are NOT old group names: their prefixes ARE
      operation env names (`JUDGE_*` - the `judge` operation, `DECISION_*` -
      the `decision` operation, the owner's decision Q2), so they never warn;
    - the global `SLOT_*` knobs are the safeguards and the base of the
      per-operation thresholds (the owner's decision Q4) - they never warn;
    - no new env names are introduced: the app reads exactly the env names of
      the base commit (the criterion of the step).

    The last test freezes the `legacy` snapshot of the base commit (e08a622) -
    the env surface, the effective slot values, the resolved chain of every
    operation and the startup warnings.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import logging
import os
from pathlib import Path

import httpx
import pytest

from app.config import Settings, get_settings
from app.main import _operations_config_warnings, _provider_startup_check
from app.services import Services, build_services
from app.services.llm_client import LLMClient, SlotSpec
from app.services.operations import (
    LEGACY_SLOT_GROUP_OPERATIONS,
    OPERATION_FIELDS,
    OPERATION_REGISTRY,
    OPERATIONS,
    env_name,
    legacy_slot_env_plan,
    legacy_slot_env_warnings,
    resolve_operation,
)

DEV_ROOT = Path(__file__).resolve().parents[1]

# The env prefixes the step is about: the old slot groups, the global slot
# knobs and the per-operation names.
_GROUP_PREFIXES = ("SUMMARY_", "JUDGE_", "EMBEDDING_", "DECISION_", "SLOT_")
_OPERATION_PREFIXES = (
    "SUMMARIZE_",
    "TITLE_",
    "MERGE_",
    "SPLIT_",
    "CLASSIFY_",
    "DESCRIBE_",
    "STRUCTURE_JUDGE_",
    "EMBED_",
)

# The required env of the service (the validator needs them in every mode).
REQUIRED_ENV: dict[str, str] = {
    "MCP_AUTH_TOKEN": "test-secret-token",
    "EMBEDDING_BASE_URL": "http://127.0.0.1:1",
    "SUMMARY_BASE_URL": "http://127.0.0.1:1",
    "SUMMARY_MODEL": "snapshot-summary-model",
    "JUDGE_BASE_URL": "http://127.0.0.1:1",
    "JUDGE_MODEL": "snapshot-judge-model",
}

# The env of the `legacy` snapshot (the base commit's behaviour).
SNAPSHOT_ENV: dict[str, str] = {
    **REQUIRED_ENV,
    "EMBEDDING_MODEL": "snapshot-embedding-model",
    "SUMMARY_PROVIDER": "ollama",
    "SUMMARY_THINK": "true",
    "SUMMARY_NUM_PREDICT": "512",
    "SUMMARY_TIMEOUT_SEC": "750",
    "JUDGE_THINK": "true",
    "JUDGE_NUM_PREDICT": "4096",
    "SLOT_PAUSE_AFTER": "4",
    "SLOT_PAUSE_SEC": "600",
    "SLOT_PROMPT_RESERVE_TOKENS": "256",
}

# The `legacy` snapshot of the BASE commit (e08a622) - the env surface (the
# sha of the sorted `Settings` fields), the effective slot values, the resolved
# chain of every operation and the codes of the startup warnings. Taken with
# the script /tmp/lsb-s16-legacy-snapshot.py on the base worktree; the same
# script on the branch prints an identical JSON (the diff is empty).
LEGACY_ENV_SNAPSHOT: dict[str, object] = {
    "env_names_sha256": (
        "e34fbc53e7bd9a65dba36d098c19e857c9d4cfbb4433d97e43b9d1b896e569ee"
    ),
    "slot_values": {
        "embedding_model": "snapshot-embedding-model",
        "embedding_provider": "ollama",
        "judge_model": "snapshot-judge-model",
        "judge_num_predict": 4096,
        "judge_think": True,
        "slot_model": "legacy",
        "slot_pause_after": 4,
        "slot_pause_sec": 600,
        "slot_prompt_reserve_tokens": 256,
        "summary_model": "snapshot-summary-model",
        "summary_num_predict": 512,
        "summary_provider": "ollama",
        "summary_think": True,
        "summary_timeout_sec": 750,
    },
    "operations": {
        "classify": {
            "sources": {
                "api_key": "default",
                "base_url": "slot",
                "context_tokens": "default",
                "model": "slot",
                "model_type": "default",
                "num_predict": "slot",
                "provider": "default",
                "reasoning_num_predict": "default",
                "think": "default",
                "timeout_sec": "slot",
            },
            "values": {},
        },
        "describe": {
            "sources": {
                "api_key": "default",
                "base_url": "slot",
                "context_tokens": "default",
                "model": "slot",
                "model_type": "default",
                "num_predict": "slot",
                "provider": "default",
                "reasoning_num_predict": "default",
                "think": "default",
                "timeout_sec": "slot",
            },
            "values": {},
        },
        "embed": {
            "sources": {
                "api_key": "default",
                "base_url": "slot",
                "context_tokens": "default",
                "model": "slot",
                "model_type": "default",
                "num_predict": "default",
                "provider": "default",
                "reasoning_num_predict": "default",
                "think": "default",
                "timeout_sec": "default",
            },
            "values": {},
        },
        "judge": {
            "sources": {
                "api_key": "default",
                "base_url": "slot",
                "context_tokens": "default",
                "model": "slot",
                "model_type": "default",
                "num_predict": "slot",
                "provider": "default",
                "reasoning_num_predict": "default",
                "think": "slot",
                "timeout_sec": "default",
            },
            "values": {},
        },
        "merge": {
            "sources": {
                "api_key": "default",
                "base_url": "slot",
                "context_tokens": "default",
                "model": "slot",
                "model_type": "default",
                "num_predict": "op",
                "provider": "default",
                "reasoning_num_predict": "default",
                "think": "default",
                "timeout_sec": "slot",
            },
            "values": {"num_predict": "0"},
        },
        "split": {
            "sources": {
                "api_key": "default",
                "base_url": "slot",
                "context_tokens": "default",
                "model": "slot",
                "model_type": "default",
                "num_predict": "slot",
                "provider": "default",
                "reasoning_num_predict": "default",
                "think": "default",
                "timeout_sec": "slot",
            },
            "values": {},
        },
        "structure_judge": {
            "sources": {
                "api_key": "default",
                "base_url": "slot",
                "context_tokens": "default",
                "model": "slot",
                "model_type": "default",
                "num_predict": "slot",
                "provider": "default",
                "reasoning_num_predict": "default",
                "think": "slot",
                "timeout_sec": "default",
            },
            "values": {},
        },
        "summarize": {
            "sources": {
                "api_key": "default",
                "base_url": "slot",
                "context_tokens": "default",
                "model": "slot",
                "model_type": "default",
                "num_predict": "slot",
                "provider": "default",
                "reasoning_num_predict": "default",
                "think": "default",
                "timeout_sec": "slot",
            },
            "values": {},
        },
        "title": {
            "sources": {
                "api_key": "default",
                "base_url": "slot",
                "context_tokens": "default",
                "model": "slot",
                "model_type": "default",
                "num_predict": "slot",
                "provider": "default",
                "reasoning_num_predict": "default",
                "think": "default",
                "timeout_sec": "slot",
            },
            "values": {},
        },
    },
    "warning_codes": ["context_unknown"],
}


def _is_step_env(name: str) -> bool:
    return name.startswith(_GROUP_PREFIXES + _OPERATION_PREFIXES)


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """A clean env: no old group names, no operation names, no SLOT_* knobs.

    The ambient env of the developer machine must not leak into the tests -
    the old names are exactly what the step is about. The required names are
    set by the test itself (they are old group names too).
    """
    for name in list(os.environ):
        if _is_step_env(name):
            monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()
    return monkeypatch


def make_settings(monkeypatch: pytest.MonkeyPatch, **env: str) -> Settings:
    """The settings of the given env (the cache is cleared on both sides).

    The required names are added to the env (they are old group names too);
    a value of the test overrides the required one.
    """
    for name, value in {**REQUIRED_ENV, **env}.items():
        monkeypatch.setenv(name, value)
    get_settings.cache_clear()
    settings = get_settings()
    get_settings.cache_clear()
    return settings


def settings_without_group_env(**env: str) -> Settings:
    """The settings of an env with NO old group name.

    The required slot values are passed as constructor arguments (not as env
    names): the scenario is "the environment holds the new names only".
    """
    values = {
        "mcp_auth_token": REQUIRED_ENV["MCP_AUTH_TOKEN"],
        "embedding_base_url": REQUIRED_ENV["EMBEDDING_BASE_URL"],
        "summary_base_url": REQUIRED_ENV["SUMMARY_BASE_URL"],
        "summary_model": REQUIRED_ENV["SUMMARY_MODEL"],
        "judge_base_url": REQUIRED_ENV["JUDGE_BASE_URL"],
        "judge_model": REQUIRED_ENV["JUDGE_MODEL"],
    }
    values.update(env)
    return Settings(**values)


def _ok_handler(model: str):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": [{"name": model}]})

    return handler


def _with_mock_slots(settings: Settings) -> Services:
    """build_services with MockTransport clients of the three slots (no network)."""
    services = build_services(settings)
    ok = lambda model: _ok_handler(model)  # noqa: E731
    return dataclasses.replace(
        services,
        llm_embedding=LLMClient(
            SlotSpec.for_embedding(settings), transport=ok(settings.embedding_model)
        ),
        llm_summary=LLMClient(
            SlotSpec.for_summary(settings), transport=ok(settings.summary_model)
        ),
        llm_judge=LLMClient(
            SlotSpec.for_judge(settings), transport=ok(settings.judge_model)
        ),
    )


def _warning_records(caplog) -> list:
    return [
        record
        for record in caplog.records
        if getattr(record, "event", None) in {"legacy_slot_env_warning"}
    ]


# --- the criterion 1: the operation mode warns about the old group names ---


def test_the_operation_mode_warns_about_the_old_group_name(
    clean_env: pytest.MonkeyPatch,
) -> None:
    """SLOT_MODEL=operation + SUMMARY_MODEL -> a WARN that names the new names."""
    settings = make_settings(
        clean_env, SLOT_MODEL="operation", SUMMARY_MODEL="old-model"
    )
    warnings = legacy_slot_env_warnings(settings)
    codes = {code for code, _text in warnings}
    assert "legacy_slot_env" in codes
    texts = [text for code, text in warnings if code == "legacy_slot_env"]
    assert any("SUMMARY_MODEL" in text for text in texts)


def test_the_warning_names_the_equivalent_operation_env_names(
    clean_env: pytest.MonkeyPatch,
) -> None:
    """The WARN lists the operation env names that replace the old one."""
    settings = make_settings(
        clean_env,
        SLOT_MODEL="operation",
        SUMMARY_MODEL="old-model",
        EMBEDDING_MODEL="old-embed",
    )
    plan = dict(legacy_slot_env_plan(settings))
    assert plan["SUMMARY_MODEL"] == (
        "SUMMARIZE_MODEL",
        "TITLE_MODEL",
        "MERGE_MODEL",
        "SPLIT_MODEL",
        "CLASSIFY_MODEL",
        "DESCRIBE_MODEL",
    )
    assert plan["EMBEDDING_MODEL"] == ("EMBED_MODEL",)
    text = " ".join(text for _code, text in legacy_slot_env_warnings(settings))
    assert "SUMMARIZE_MODEL" in text
    assert "EMBED_MODEL" in text


def test_the_startup_with_the_old_name_is_not_fatal(
    clean_env: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The startup check warns and continues - the old name is not a fatal error."""
    settings = make_settings(
        clean_env, SLOT_MODEL="operation", SUMMARY_MODEL="old-model"
    )
    services = _with_mock_slots(settings)
    with caplog.at_level(logging.WARNING, logger="app"):
        asyncio.run(_provider_startup_check(services, settings))  # does not raise
    records = _warning_records(caplog)
    assert records
    assert all(record.levelno == logging.WARNING for record in records)
    assert all(getattr(record, "code", None) == "legacy_slot_env" for record in records)


# --- the criterion 2: a clean start with the new names ---


def test_a_clean_start_with_the_new_names_warns_about_nothing(
    clean_env: pytest.MonkeyPatch,
) -> None:
    """Only the `<OP>_*` names in the env -> no warning about the old names."""
    clean_env.setenv("SUMMARIZE_MODEL", "new-model")
    clean_env.setenv("TITLE_MODEL", "new-model")
    clean_env.setenv("MERGE_MODEL", "new-model")
    clean_env.setenv("EMBED_MODEL", "new-model")
    settings = settings_without_group_env(slot_model="operation")
    assert legacy_slot_env_warnings(settings) == []
    assert legacy_slot_env_plan(settings) == []


def test_the_new_names_do_not_produce_an_unknown_operation_warning(
    clean_env: pytest.MonkeyPatch,
) -> None:
    """The new names are known to the typo scanner - no `unknown_operation`."""
    from app.services.operations import unknown_operation_warnings

    clean_env.setenv("SUMMARIZE_MODEL", "new-model")
    clean_env.setenv("STRUCTURE_JUDGE_MODEL", "new-model")
    clean_env.setenv("EMBED_MODEL", "new-model")
    settings = settings_without_group_env(slot_model="operation")
    codes = {code for code, _text in unknown_operation_warnings(settings)}
    assert "unknown_operation" not in codes


# --- the criterion 3: the legacy mode is silent and the old names act ---


def test_in_the_legacy_mode_the_old_name_still_acts(
    clean_env: pytest.MonkeyPatch,
) -> None:
    """legacy + SUMMARY_MODEL -> the value reaches the chain (the slot layer)."""
    settings = make_settings(clean_env, SUMMARY_MODEL="old-model")
    assert settings.slot_model == "legacy"
    values, sources = resolve_operation("summarize", settings)
    assert values == {}
    assert sources["model"] == "slot"
    assert settings.summary_model == "old-model"


def test_in_the_legacy_mode_nothing_is_warned_about_the_old_names(
    clean_env: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """legacy -> the startup says nothing about the old group names."""
    settings = make_settings(
        clean_env, SUMMARY_MODEL="old-model", EMBEDDING_MODEL="old-embed"
    )
    assert legacy_slot_env_warnings(settings) == []
    with caplog.at_level(logging.WARNING, logger="app"):
        _operations_config_warnings(logging.getLogger("app"), settings)
    assert _warning_records(caplog) == []


# --- the criterion 4: the global SLOT_* knobs are not touched ---


def test_the_global_slot_knobs_are_not_warned_about(
    clean_env: pytest.MonkeyPatch,
) -> None:
    """SLOT_PAUSE_AFTER / SLOT_PAUSE_SEC / ... -> no error, no old-name warning."""
    clean_env.setenv("SLOT_PAUSE_AFTER", "5")
    clean_env.setenv("SLOT_PAUSE_SEC", "300")
    clean_env.setenv("SLOT_PROMPT_RESERVE_TOKENS", "128")
    clean_env.setenv("SLOT_MAX_OUTPUT_TOKENS", "4096")
    clean_env.setenv("SLOT_MODEL", "operation")
    settings = settings_without_group_env(slot_model="operation")
    assert legacy_slot_env_plan(settings) == []
    assert legacy_slot_env_warnings(settings) == []
    # the knobs are read and keep their meaning (the base of the per-op thresholds)
    assert settings.slot_pause_after == 5
    assert settings.slot_pause_sec == 300
    assert settings.slot_prompt_reserve_tokens == 128


# --- the criterion 5: the decision prefix is not an old name ---


def test_the_decision_prefix_is_never_an_old_name(
    clean_env: pytest.MonkeyPatch,
) -> None:
    """DECISION_* is the env of the `decision` operation - it never warns."""
    for mode in ("legacy", "operation"):
        settings = make_settings(
            clean_env,
                SLOT_MODEL=mode,
            DECISION_MODEL="decision-model",
            DECISION_BASE_URL="http://127.0.0.1:1",
            DECISION_TIMEOUT_SEC="300",
        )
        plan = dict(legacy_slot_env_plan(settings))
        assert not [name for name in plan if name.startswith("DECISION_")]
        assert not [
            text
            for _code, text in legacy_slot_env_warnings(settings)
            if "DECISION_" in text
        ]


def test_the_decision_env_name_stays_the_decision_prefix() -> None:
    """The `decision` operation keeps the DECISION_* prefix (the owner's Q2)."""
    assert env_name("decision", "model") == "DECISION_MODEL"
    assert "decision" in OPERATION_REGISTRY


# --- the criterion 6: the docs and the compose ---


def test_the_config_docs_carry_the_migration_table() -> None:
    """dev/docs/CONFIG.md: the old -> new table and the removal date."""
    text = (DEV_ROOT / "docs" / "CONFIG.md").read_text(encoding="utf-8")
    assert "SUMMARY_MODEL" in text
    assert "SUMMARIZE_MODEL" in text
    assert "EMBEDDING_MODEL" in text
    assert "EMBED_MODEL" in text
    assert "3.10.0" in text


def test_the_compose_carries_the_new_names() -> None:
    """dev/docker-compose.yml: the per-operation names are in the contour."""
    text = (DEV_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    for name in (
        "SUMMARIZE_MODEL",
        "TITLE_MODEL",
        "MERGE_MODEL",
        "SPLIT_MODEL",
        "CLASSIFY_MODEL",
        "DESCRIBE_MODEL",
        "STRUCTURE_JUDGE_MODEL",
        "EMBED_MODEL",
    ):
        assert name in text
    # the mode switch stays OFF on the contour (the acceptance of the release)
    assert 'SLOT_MODEL: "legacy"' in text


# --- the criterion 7: no new env names ---


def test_the_env_surface_did_not_grow() -> None:
    """The app reads exactly the env names of the base commit."""
    digest = hashlib.sha256(
        "\n".join(sorted(Settings.model_fields)).encode("utf-8")
    ).hexdigest()
    assert digest == LEGACY_ENV_SNAPSHOT["env_names_sha256"]


def test_no_new_env_prefix_was_introduced() -> None:
    """No OPERATION_* / SLOT_MODEL_* env name: the names are `<OP>_*` only."""
    for field in Settings.model_fields:
        assert not field.upper().startswith("OPERATION_")
        assert not field.upper().startswith("SLOT_MODEL_")


def test_the_old_group_prefixes_are_not_operation_names() -> None:
    """The old group names are the prefixes no operation claims."""
    for group in LEGACY_SLOT_GROUP_OPERATIONS:
        assert group not in OPERATION_REGISTRY
        assert group not in OPERATIONS


def test_the_suggested_names_are_registry_operation_names(
    clean_env: pytest.MonkeyPatch,
) -> None:
    """Every name the WARN suggests is an env name of the registry."""
    settings = make_settings(
        clean_env,
        SLOT_MODEL="operation",
        SUMMARY_MODEL="old-model",
        EMBEDDING_MODEL="old-embed",
    )
    known = {
        env_name(operation, field)
        for operation in OPERATION_REGISTRY
        for field in OPERATION_FIELDS
    }
    for _name, equivalents in legacy_slot_env_plan(settings):
        for equivalent in equivalents:
            assert equivalent in known


# --- the proof that the legacy layer did not change ---


def test_the_legacy_layer_matches_the_base_snapshot(
    clean_env: pytest.MonkeyPatch,
) -> None:
    """The `legacy` behaviour is the snapshot of the base commit (e08a622)."""
    for name, value in SNAPSHOT_ENV.items():
        clean_env.setenv(name, value)
    get_settings.cache_clear()
    settings = get_settings()
    get_settings.cache_clear()

    slot_values = {
        name: getattr(settings, name)
        for name in LEGACY_ENV_SNAPSHOT["slot_values"]  # type: ignore[union-attr]
    }
    operations: dict[str, object] = {}
    for operation in OPERATIONS:
        values, sources = resolve_operation(operation, settings)
        operations[operation] = {
            "values": {key: str(value) for key, value in sorted(values.items())},
            "sources": dict(sorted(sources.items())),
        }
    logger = logging.getLogger("app")
    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _Capture()
    logger.addHandler(handler)
    try:
        _operations_config_warnings(logger, settings)
    finally:
        logger.removeHandler(handler)

    assert slot_values == LEGACY_ENV_SNAPSHOT["slot_values"]
    assert operations == LEGACY_ENV_SNAPSHOT["operations"]
    assert [getattr(record, "code", None) for record in records] == LEGACY_ENV_SNAPSHOT[
        "warning_codes"
    ]
