"""Decision-слот: сервис решений (LSBN-F-12, релиз 3.2.0; arch §6.4/§13.6).

Юнит без сети: клиент на httpx.MockTransport. Покрывается: выключенный слот —
noop без вызовов; порог и зона дают три причины переключения
(`low_confidence`/принято); свой разрыв слота экономит вызовы и
восстанавливается пробным стуком; потолок меток — `unsupported` без вызова.
"""

from __future__ import annotations

import httpx
import pytest

from app.config import get_settings
from app.services import build_services
from app.services.decision import (
    DecisionQuestion,
    DecisionService,
    DecisionUnavailable,
)
from app.services.llm_client import LLMClient, SlotSpec
from app.services.prompts import PromptRegistry

STATE = "state text"


def make_settings(monkeypatch: pytest.MonkeyPatch, **env: str):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    get_settings.cache_clear()
    settings = get_settings()
    get_settings.cache_clear()
    return settings


def decision_settings(monkeypatch: pytest.MonkeyPatch, **env: str):
    return make_settings(
        monkeypatch,
        DECISION_ENABLED="true",
        DECISION_BASE_URL="http://localhost:11434",
        DECISION_MODEL="decision-test",
        **env,
    )


class Recorder:
    def __init__(self, actions: list) -> None:
        self.actions = actions
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        action = self.actions.pop(0) if len(self.actions) > 1 else self.actions[0]
        if isinstance(action, Exception):
            raise action
        return action

    @property
    def calls(self) -> int:
        return len(self.requests)


def make_service(settings, actions: list) -> tuple[DecisionService, Recorder]:
    recorder = Recorder(actions)
    llm = LLMClient(SlotSpec.for_decision(settings), transport=recorder.handler)
    service = DecisionService(settings, llm=llm, registry=PromptRegistry(None))
    return service, recorder


def noul_question(**kwargs) -> DecisionQuestion:
    return DecisionQuestion(name="q", text="Is Q true?", kind="noul", **kwargs)


def answers(payload: dict) -> httpx.Response:
    return httpx.Response(200, json={"answers": payload})


# --- выключенный слот --------------------------------------------------------


def test_disabled_is_noop(monkeypatch) -> None:
    settings = make_settings(
        monkeypatch,
        DECISION_BASE_URL="http://localhost:11434",
        DECISION_MODEL="decision-test",
    )
    services = build_services(settings)
    assert services.decision is None
    assert services.llm_decision is None

    service, recorder = make_service(settings, [answers({"q": {"noul": 0.9}})])
    assert service.can_decide() is False
    with pytest.raises(DecisionUnavailable):
        service.decide(STATE, [noul_question()], operation="judge")
    assert recorder.calls == 0


# --- порог и зона ------------------------------------------------------------


def test_threshold_zone_low_confidence(monkeypatch) -> None:
    service, _ = make_service(
        decision_settings(monkeypatch), [answers({"q": {"noul": 0.52}})]
    )
    with pytest.raises(DecisionUnavailable) as excinfo:
        service.decide(STATE, [noul_question()], operation="judge")
    assert excinfo.value.reason == "low_confidence"
    assert excinfo.value.answer is not None
    assert excinfo.value.answer.probability == pytest.approx(0.52)


def test_threshold_decision_above_zone(monkeypatch) -> None:
    service, _ = make_service(
        decision_settings(monkeypatch), [answers({"q": {"noul": 0.9}})]
    )
    result = service.decide(STATE, [noul_question()], operation="judge")
    assert result.answers["q"].label == "true"
    assert result.meta.slot == "decision"
    assert result.meta.model == "decision-test"


def test_choice_below_threshold_low_confidence(monkeypatch) -> None:
    service, _ = make_service(
        decision_settings(monkeypatch),
        [answers({"q": {"choice": "task", "probabilities": {"task": 0.52, "generic": 0.1}}})],
    )
    question = DecisionQuestion(
        name="q", text="Where?", kind="choice", labels=("task", "generic"), criteria={"task": "x", "generic": "y"}
    )
    with pytest.raises(DecisionUnavailable) as excinfo:
        service.decide(STATE, [question], operation="classify")
    assert excinfo.value.reason == "low_confidence"
    assert excinfo.value.answer.label == "task"


# --- свой разрыв -------------------------------------------------------------


def test_breaker_skips_call_when_paused(monkeypatch) -> None:
    service, recorder = make_service(
        decision_settings(monkeypatch), [httpx.Response(401, text="nope")]
    )
    with pytest.raises(DecisionUnavailable) as first:
        service.decide(STATE, [noul_question()], operation="judge")
    assert first.value.reason == "unavailable"
    assert recorder.calls == 1
    # Разрыв открыт (401 = auth): второй вызов без HTTP-обращения.
    with pytest.raises(DecisionUnavailable) as second:
        service.decide(STATE, [noul_question()], operation="judge")
    assert second.value.reason == "unavailable"
    assert recorder.calls == 1


def test_breaker_auth_opens_and_probe_recovers(monkeypatch) -> None:
    service, _ = make_service(
        decision_settings(monkeypatch),
        [httpx.Response(401, text="nope"), answers({"probe": {"noul": 0.9}})],
    )
    with pytest.raises(DecisionUnavailable):
        service.decide(STATE, [noul_question()], operation="judge")
    assert service.snapshot()["breaker"]["state"] == "paused"
    assert service.probe() == "ok"
    assert service.snapshot()["breaker"]["state"] == "closed"


# --- потолок меток -----------------------------------------------------------


def test_too_many_labels_unsupported(monkeypatch) -> None:
    service, recorder = make_service(
        decision_settings(monkeypatch, DECISION_MAX_LABELS="2"), [httpx.Response(500, text="x")]
    )
    question = DecisionQuestion(
        name="q",
        text="Where?",
        kind="choice",
        labels=("a", "b", "c"),
        criteria={"a": "1", "b": "2", "c": "3"},
    )
    with pytest.raises(DecisionUnavailable) as excinfo:
        service.decide(STATE, [question], operation="classify")
    assert excinfo.value.reason == "unsupported"
    assert recorder.calls == 0


# --- счётчики и реестр -------------------------------------------------------


def test_questions_for_reads_registry(monkeypatch) -> None:
    service, _ = make_service(decision_settings(monkeypatch), [answers({})])
    judge = service.questions_for("judge")
    assert [question.name for question in judge] == ["is_text_duplicate"]
    assert judge[0].kind == "noul"
    assert service.questions_for("missing") == []


def test_snapshot_counts_fallbacks(monkeypatch) -> None:
    service, _ = make_service(
        decision_settings(monkeypatch), [answers({"q": {"noul": 0.52}})]
    )
    with pytest.raises(DecisionUnavailable):
        service.decide(STATE, [noul_question()], operation="judge")
    snapshot = service.snapshot()
    assert snapshot["enabled"] is True
    assert snapshot["ok"] is True
    assert snapshot["fallbacks"] == 1
    assert snapshot["by_reason"]["low_confidence"] == 1
    assert snapshot["answers"] == 0
