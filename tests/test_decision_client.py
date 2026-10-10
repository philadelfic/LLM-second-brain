"""Decision-слот: клиент /v1/systemone (LSBN-F-12, релиз 3.2.0; arch §6.2/§13.6).

Юнит через httpx.MockTransport — без сети. Покрывается одна точка адаптации
формата (`_systemone_payload` / `_parse_systemone`): один state и все вопросы
в одном теле (FR-2.3); три формы ответа пилота LSBN-T-13 разбираются в
`DecisionAnswer`; незнакомый формат — `LLMError` (не «пустой ответ»:
генерации нет); стартовая проверка `check_decision()` — все исходы.
"""

from __future__ import annotations

import json

import httpx
import pytest

from app.config import get_settings
from app.services.decision import answer_from_parsed
from app.services.llm_client import (
    DECISION_CHECK_AUTH_FAILED,
    DECISION_CHECK_OK,
    DECISION_CHECK_UNSUPPORTED,
    DECISION_CHECK_UNREACHABLE,
    LLMClient,
    LLMError,
    SlotSpec,
)


# --- вспомогательное ---------------------------------------------------------


def make_settings(monkeypatch: pytest.MonkeyPatch, **env: str):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    get_settings.cache_clear()
    settings = get_settings()
    get_settings.cache_clear()
    return settings


class Recorder:
    """Транспорт-лог: считает запросы, раздаёт scripted-ответы по очереди."""

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


def make_client(settings, actions: list) -> tuple[LLMClient, Recorder]:
    recorder = Recorder(actions)
    client = LLMClient(SlotSpec.for_decision(settings), transport=recorder.handler)
    return client, recorder


def decision_settings(monkeypatch: pytest.MonkeyPatch, **env: str):
    return make_settings(
        monkeypatch,
        DECISION_ENABLED="true",
        DECISION_BASE_URL="http://localhost:11434",
        DECISION_MODEL="decision-test",
        **env,
    )


def ok_answers(payload: dict) -> httpx.Response:
    return httpx.Response(200, json={"answers": payload})


# --- контракт клиента --------------------------------------------------------


def test_systemone_payload_shape(monkeypatch) -> None:
    """Один state и все вопросы группы в одном теле (FR-2.3)."""
    state = "node descriptions"
    questions = [
        {"name": "is_text_duplicate", "kind": "noul", "text": "Dup?", "labels": (), "criteria": None},
        {"name": "note_section", "kind": "choice", "text": "Where?", "labels": ("generic",), "criteria": {"generic": "elsewhere"}},
    ]
    payload = LLMClient._systemone_payload(state, questions)
    assert set(payload) == {"state", "questions"}
    assert payload["state"] == state
    assert [item["name"] for item in payload["questions"]] == [
        "is_text_duplicate",
        "note_section",
    ]
    assert payload["questions"][0]["type"] == "noul"
    assert payload["questions"][1]["labels"] == ["generic"]
    assert payload["questions"][1]["criteria"] == {"generic": "elsewhere"}

    client, recorder = make_client(
        decision_settings(monkeypatch),
        [ok_answers({"is_text_duplicate": {"noul": 0.9}, "note_section": {"choice": "generic", "probabilities": {"generic": 0.9}}})],
    )
    client.decide(state, questions)
    body = json.loads(recorder.requests[0].content)
    assert recorder.calls == 1
    assert recorder.requests[0].url.path == "/v1/systemone"
    assert body["model"] == "decision-test"
    assert body["state"] == state
    assert len(body["questions"]) == 2


def test_systemone_parse_noul(monkeypatch) -> None:
    """Форма noul пилота → DecisionAnswer(label=true, probability=noul)."""
    parsed = LLMClient._parse_systemone({"answers": {"q": {"type": "noul", "noul": 0.95}}})
    answer = answer_from_parsed("q", parsed["q"])
    assert answer.kind == "noul"
    assert answer.label == "true"
    assert answer.probability == pytest.approx(0.95)


def test_systemone_parse_choice(monkeypatch) -> None:
    """Форма choice пилота → DecisionAnswer(label, probabilities, confidence)."""
    parsed = LLMClient._parse_systemone(
        {"answers": {"q": {"choice": "task", "probabilities": {"task": 0.36, "generic": 0.1}, "confidence": 0.36}}}
    )
    answer = answer_from_parsed("q", parsed["q"])
    assert answer.kind == "choice"
    assert answer.label == "task"
    assert answer.probability == pytest.approx(0.36)
    assert answer.confidence == pytest.approx(0.36)
    assert answer.probabilities == {"task": 0.36, "generic": 0.1}


def test_systemone_parse_score(monkeypatch) -> None:
    """Форма score пилота → DecisionAnswer(score, legend в raw)."""
    parsed = LLMClient._parse_systemone(
        {"answers": {"q": {"score": 0.657, "legend": {"a": 1}, "probabilities": {"a": 0.6}, "confidence": 0.5}}}
    )
    answer = answer_from_parsed("q", parsed["q"])
    assert answer.kind == "score"
    assert answer.score == pytest.approx(0.657)
    assert answer.raw["legend"] == {"a": 1}


def test_systemone_unknown_shape_raises(monkeypatch) -> None:
    """Ответ без answers/незнакомая форма → LLMError (не «пустой ответ»)."""
    with pytest.raises(LLMError):
        LLMClient._parse_systemone({"nope": []})
    with pytest.raises(LLMError):
        LLMClient._parse_systemone({"answers": {"q": {"weird": 1}}})
    with pytest.raises(LLMError):
        LLMClient._parse_systemone("not an object")


# --- стартовая проверка «умеет decision» -------------------------------------


def test_check_decision_unsupported(monkeypatch) -> None:
    client, _ = make_client(
        decision_settings(monkeypatch),
        [httpx.Response(400, text="type must be choice, noul, or score")],
    )
    assert client.check_decision() == DECISION_CHECK_UNSUPPORTED


def test_check_decision_auth(monkeypatch) -> None:
    client, _ = make_client(decision_settings(monkeypatch), [httpx.Response(401, text="nope")])
    assert client.check_decision() == DECISION_CHECK_AUTH_FAILED


def test_check_decision_unreachable(monkeypatch) -> None:
    client, _ = make_client(
        decision_settings(monkeypatch), [httpx.ConnectError("no route")]
    )
    assert client.check_decision() == DECISION_CHECK_UNREACHABLE


def test_check_decision_ok(monkeypatch) -> None:
    client, _ = make_client(
        decision_settings(monkeypatch), [ok_answers({"probe": {"noul": 0.9}})]
    )
    assert client.check_decision() == DECISION_CHECK_OK
