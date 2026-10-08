"""`/health` decision-слот: `decision_ok`, блок `decision` и ключ в `slots` (LSBN-F-12).

Релиз 3.2.0, arch §7.1/§7.3, FR-6.1…FR-6.3. Оператор видит, умеет ли слот,
сколько решений принято, сколько переключений и по каким причинам, паузу и
последний исход разрыва. Сбор `/health` — только память/SQL: модели не зовутся.

Хост-тест: приложение собирается на хосте как в `test_main_decision_check`.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import create_app
from app.services.decision import DecisionQuestion, DecisionUnavailable
from app.services.llm_client import LLMClient
from app.services.slot_outcome import SlotOutcome

DISABLED_REASONS = {"unavailable": 0, "unsupported": 0, "low_confidence": 0}


def make_settings(monkeypatch: pytest.MonkeyPatch, **env: str):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    get_settings.cache_clear()
    settings = get_settings()
    get_settings.cache_clear()
    return settings


def decision_app(monkeypatch: pytest.MonkeyPatch, check: str = "ok"):
    """Приложение с включённым decision-слотом (стартовая проверка — `check`)."""
    monkeypatch.setattr(LLMClient, "check_decision", lambda self: check)
    settings = make_settings(
        monkeypatch,
        DECISION_ENABLED="true",
        DECISION_BASE_URL="http://127.0.0.1:1",
        DECISION_MODEL="decision-test",
    )
    return create_app(settings)


def answer(noul: float) -> dict:
    """Канонический ответ вопроса (форма после `_parse_systemone`)."""
    return {
        "kind": "noul",
        "label": "true" if noul > 0.5 else "false",
        "probability": noul,
        "confidence": None,
        "probabilities": {},
        "score": None,
        "raw": {},
    }


def test_decision_disabled_shape(client: TestClient) -> None:
    """`decision_enabled=false` → `decision_ok is None`, блок пуст, счётчики нулевые."""
    body = client.get("/health").json()
    assert body["decision_ok"] is None
    assert body["decision"] == {
        "enabled": False,
        "ok": None,
        "answers": 0,
        "fallbacks": 0,
        "by_reason": dict(DISABLED_REASONS),
    }


def test_decision_ok_after_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """`ok` → слот жив (`decision_ok is True`); `unsupported` → слот погашен (False)."""
    app_ok = decision_app(monkeypatch, check="ok")
    with TestClient(app_ok) as client:
        body = client.get("/health").json()
        assert body["decision_ok"] is True
        assert body["decision"]["ok"] is True
        assert body["decision"]["enabled"] is True

    app_bad = decision_app(monkeypatch, check="unsupported")
    with TestClient(app_bad) as client:
        body = client.get("/health").json()
        assert body["decision_ok"] is False
        assert body["decision"]["ok"] is False
        assert app_bad.state.services.decision.can_decide() is False


def test_decision_aggregates(monkeypatch: pytest.MonkeyPatch) -> None:
    """N принятых решений и M переключений → `answers=N`, `fallbacks=M`, `by_reason`."""
    app = decision_app(monkeypatch)
    with TestClient(app) as client:
        service = app.state.services.decision
        question = DecisionQuestion(name="q", text="Is Q true?", kind="noul")
        monkeypatch.setattr(
            service._llm, "decide", lambda state, payload: {"q": answer(0.95)}
        )
        for _ in range(3):
            service.decide("state", [question], operation="judge")
        monkeypatch.setattr(
            service._llm, "decide", lambda state, payload: {"q": answer(0.5)}
        )
        for _ in range(2):
            with pytest.raises(DecisionUnavailable) as excinfo:
                service.decide("state", [question], operation="judge")
            assert excinfo.value.reason == "low_confidence"

        decision = client.get("/health").json()["decision"]
        assert decision["answers"] == 3
        assert decision["fallbacks"] == 2
        assert decision["by_reason"] == {
            "unavailable": 0,
            "unsupported": 0,
            "low_confidence": 2,
        }


def test_slots_contains_decision(client: TestClient, monkeypatch) -> None:
    """`/health.slots.decision` есть всегда; при разрыве — пауза и исход."""
    assert "decision" in client.get("/health").json()["slots"]

    app = decision_app(monkeypatch)
    with TestClient(app) as client:
        service = app.state.services.decision
        service._breaker.note_failure(SlotOutcome.AUTH)
        stat = client.get("/health").json()["slots"]["decision"]
        assert stat["last_outcome"] == "auth"
        assert stat["paused_until"] is not None


def test_health_no_model_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """Сбор `/health` не вызывает decision-модель (счётчик заглушки = 0)."""
    app = decision_app(monkeypatch)
    calls = {"n": 0}

    def spy(*args, **kwargs):
        calls["n"] += 1
        raise AssertionError("decision-модель вызвана при сборе /health")

    with TestClient(app) as client:
        llm = app.state.services.decision._llm
        monkeypatch.setattr(llm, "decide", spy)
        monkeypatch.setattr(llm, "check_decision", spy)
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert "decision" in body["slots"]
        assert calls["n"] == 0
