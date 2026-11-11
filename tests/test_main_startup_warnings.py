"""Slot configuration startup warnings (#49, release 3.2.0; arch §7.2, §13.5).

    A bad slot configuration is nameable and not fatal: the startup check writes a
    WARNING-level `slot_config_warning` event and keeps going; the only fatal
    failure remains an authorization one (auth_failed → SystemExit(2)).
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging

import httpx
import pytest

from app.config import get_settings
from app.main import _provider_startup_check
from app.services import Services, build_services
from app.services.llm_client import (
    CHECK_AUTH_FAILED,
    LLMClient,
    SlotSpec,
)


def make_settings(monkeypatch: pytest.MonkeyPatch, **env: str):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    get_settings.cache_clear()
    settings = get_settings()
    get_settings.cache_clear()
    return settings


def _ok_handler(model: str):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": [{"name": model}]})

    return handler


def _with_mock_slots(settings) -> Services:
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
        if getattr(record, "event", None) == "slot_config_warning"
    ]


def test_warn_not_fatal(monkeypatch, caplog: pytest.LogCaptureFixture) -> None:
    """reasoning + an explicit 256 budget → a WARN at startup; the startup continues."""
    settings = make_settings(
        monkeypatch,
        SUMMARY_MODEL_TYPE="reasoning",
        SUMMARY_NUM_PREDICT="256",
    )
    services = _with_mock_slots(settings)
    with caplog.at_level(logging.WARNING, logger="app"):
        asyncio.run(_provider_startup_check(services))  # does not raise
    warnings = _warning_records(caplog)
    assert any(
        getattr(record, "code", None) == "reasoning_tight_budget"
        for record in warnings
    )
    assert all(record.levelno == logging.WARNING for record in warnings)
    assert all(getattr(record, "slot", None) for record in warnings)


def test_auth_still_fatal(monkeypatch) -> None:
    """Regression: an authorization failure still crashes the startup (SystemExit 2)."""
    monkeypatch.setattr(LLMClient, "check", lambda self: CHECK_AUTH_FAILED)
    services = _with_mock_slots(make_settings(monkeypatch))
    with pytest.raises(SystemExit) as excinfo:
        asyncio.run(_provider_startup_check(services))
    assert excinfo.value.code == 2
