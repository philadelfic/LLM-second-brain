"""Стартовая проверка decision-слота (LSBN-F-12, релиз 3.2.0; arch §6.5/§13.6).

Модель не умеет decision → WARN, событие `decision_check`, слот погашен,
приложение поднимается, `/health` отвечает. Осознанное отступление от
фатальности auth генеративных слотов: у decision есть штатный генеративный
путь, ронять обязательный слот из-за необязательного нельзя.
"""

from __future__ import annotations

import logging

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import create_app
from app.services.decision import DecisionQuestion, DecisionUnavailable
from app.services.llm_client import (
    DECISION_CHECK_AUTH_FAILED,
    DECISION_CHECK_UNREACHABLE,
    DECISION_CHECK_UNSUPPORTED,
    LLMClient,
)


def make_settings(monkeypatch, **env: str):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    get_settings.cache_clear()
    settings = get_settings()
    get_settings.cache_clear()
    return settings


def test_unsupported_warns_not_fatal(monkeypatch, caplog) -> None:
    monkeypatch.setattr(
        LLMClient, "check_decision", lambda self: DECISION_CHECK_UNSUPPORTED
    )
    settings = make_settings(
        monkeypatch,
        DECISION_ENABLED="true",
        DECISION_BASE_URL="http://127.0.0.1:1",
        DECISION_MODEL="decision-test",
    )
    app = create_app(settings)
    with caplog.at_level(logging.WARNING, logger="app"):
        with TestClient(app) as client:
            assert client.get("/health").status_code == 200
    records = [
        record
        for record in caplog.records
        if getattr(record, "event", None) == "decision_check"
    ]
    assert records, "ожидалось событие decision_check"
    assert records[0].result == "unsupported"
    assert records[0].slot == "decision"
    assert records[0].levelno == logging.WARNING
    assert app.state.services.decision.can_decide() is False


def test_ok_keeps_slot_alive(monkeypatch, caplog) -> None:
    monkeypatch.setattr(LLMClient, "check_decision", lambda self: "ok")
    settings = make_settings(
        monkeypatch,
        DECISION_ENABLED="true",
        DECISION_BASE_URL="http://127.0.0.1:1",
        DECISION_MODEL="decision-test",
    )
    app = create_app(settings)
    with caplog.at_level(logging.INFO, logger="app"):
        with TestClient(app):
            pass
    assert app.state.services.decision.can_decide() is True


def _startup_app(monkeypatch, check: str):
    """Приложение с включённым decision-слотом, стартовая проверка вернула `check`."""
    monkeypatch.setattr(LLMClient, "check_decision", lambda self: check)
    settings = make_settings(
        monkeypatch,
        DECISION_ENABLED="true",
        DECISION_BASE_URL="http://127.0.0.1:1",
        DECISION_MODEL="decision-test",
    )
    return create_app(settings)


def _switch_reason(app) -> str:
    """Причина переключения погашенного слота (без HTTP-вызова)."""
    question = DecisionQuestion(name="q", text="Is Q true?", kind="noul")
    with pytest.raises(DecisionUnavailable) as excinfo:
        app.state.services.decision.decide("state", [question], operation="judge")
    return excinfo.value.reason


def test_unsupported_check_switches_as_unsupported(monkeypatch) -> None:
    """Проверка вернула `unsupported` → причина переключения `unsupported` («модель не умеет»).

    Исход стартовой проверки известен `main.py`; именно он определяет причину,
    а не общий «слота нет» (f12-04, приёмка 12.10.2026).
    """
    app = _startup_app(monkeypatch, DECISION_CHECK_UNSUPPORTED)
    with TestClient(app):
        assert _switch_reason(app) == "unsupported"
        decision = app.state.services.decision.snapshot()
    assert decision["by_reason"]["unsupported"] == 1
    assert decision["by_reason"]["unavailable"] == 0


@pytest.mark.parametrize(
    "check", (DECISION_CHECK_UNREACHABLE, DECISION_CHECK_AUTH_FAILED)
)
def test_transport_check_switches_as_unavailable(monkeypatch, check: str) -> None:
    """Проверка вернула `unreachable`/`auth_failed` → причина `unavailable` («слота сейчас нет»)."""
    app = _startup_app(monkeypatch, check)
    with TestClient(app):
        assert _switch_reason(app) == "unavailable"
        decision = app.state.services.decision.snapshot()
    assert decision["by_reason"]["unavailable"] == 1
    assert decision["by_reason"]["unsupported"] == 0
