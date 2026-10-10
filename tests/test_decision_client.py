"""Decision-слот: клиент /v1/systemone (LSBN-F-12, релиз 3.2.0; arch §6.2/§13.6).

Юнит через httpx.MockTransport — без сети. Покрывается одна точка адаптации
формата (`_systemone_payload` / `_parse_systemone`) в фактической форме
сервера (ollama ≥ 0.35.1; сверено живым запросом 2026-10-10): запрос —
`state` + `questions` ОБЪЕКТОМ «имя → вопрос» (`type` + `instructions` +
`criteria`); три формы ответа разбираются в `DecisionAnswer`; незнакомый
формат — `LLMError` (не «пустой ответ»: генерации нет); стартовая проверка
`check_decision()` — все исходы.
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
    """Один state и все вопросы группы в одном теле (FR-2.3).

    Тело — в фактической форме сервера: `questions` — ОБЪЕКТ «имя → вопрос»,
    формулировка едет в `instructions` (не `text`), у `choice` варианты —
    ключи `criteria` (метка без описания → null), у `noul` — необязательные
    описания двух исходов.
    """
    state = "node descriptions"
    questions = [
        {"name": "is_text_duplicate", "kind": "noul", "text": "Dup?", "labels": (), "criteria": None},
        {"name": "note_section", "kind": "choice", "text": "Where?", "labels": ("generic",), "criteria": {"generic": "elsewhere"}},
    ]
    payload = LLMClient._systemone_payload(state, questions)
    assert set(payload) == {"state", "questions"}
    assert payload["state"] == state
    assert isinstance(payload["questions"], dict)
    assert list(payload["questions"]) == ["is_text_duplicate", "note_section"]
    assert payload["questions"]["is_text_duplicate"] == {
        "type": "noul",
        "instructions": "Dup?",
    }
    assert payload["questions"]["note_section"] == {
        "type": "choice",
        "instructions": "Where?",
        "criteria": {"generic": "elsewhere"},
    }

    client, recorder = make_client(
        decision_settings(monkeypatch),
        [ok_answers({"is_text_duplicate": {"type": "noul", "noul": 0.9}, "note_section": {"type": "choice", "choice": "generic", "probabilities": {"generic": 0.9}}})],
    )
    client.decide(state, questions)
    body = json.loads(recorder.requests[0].content)
    assert recorder.calls == 1
    assert recorder.requests[0].url.path == "/v1/systemone"
    assert body["model"] == "decision-test"
    assert body["state"] == state
    assert len(body["questions"]) == 2


def test_systemone_payload_choice_labels_and_score() -> None:
    """Метка без описания → `null`; `score` — массив описаний уровней."""
    payload = LLMClient._systemone_payload(
        "node descriptions",
        [
            {"name": "note_section", "kind": "choice", "text": "Where?", "labels": ("work/sbos2020", "life", "generic"), "criteria": {"generic": "elsewhere"}},
            {"name": "urgency", "kind": "score", "text": "How urgent?", "labels": (), "criteria": ["later", "today", "now"]},
        ],
    )
    assert payload["questions"]["note_section"]["criteria"] == {
        "work/sbos2020": None,
        "life": None,
        "generic": "elsewhere",
    }
    assert payload["questions"]["urgency"]["criteria"] == ["later", "today", "now"]


def test_systemone_parse_noul(monkeypatch) -> None:
    """Ответ `noul` сервера → DecisionAnswer(label=true, probability=noul).

    Тело — фактический ответ ollama (сверено 2026-10-10): `model`/`answers`/
    `usage` верхнего уровня, у ответа вопроса — `type` и `noul`, уверенности нет.
    """
    parsed = LLMClient._parse_systemone(
        {
            "model": "clef-flash:latest",
            "answers": {"q": {"type": "noul", "noul": 0.95}},
            "usage": {"input_tokens": 144, "output_tokens": 0},
        }
    )
    answer = answer_from_parsed("q", parsed["q"])
    assert answer.kind == "noul"
    assert answer.label == "true"
    assert answer.probability == pytest.approx(0.95)
    assert answer.confidence is None
    assert answer.raw["type"] == "noul"


def test_systemone_parse_choice(monkeypatch) -> None:
    """Ответ `choice` сервера → DecisionAnswer(label, probabilities, confidence)."""
    parsed = LLMClient._parse_systemone(
        {
            "model": "clef-flash:latest",
            "answers": {
                "q": {
                    "type": "choice",
                    "choice": "bug",
                    "probabilities": {"bug": 0.8473, "billing": 0.1421},
                    "confidence": 0.5763,
                }
            },
            "usage": {"input_tokens": 161, "output_tokens": 0},
        }
    )
    answer = answer_from_parsed("q", parsed["q"])
    assert answer.kind == "choice"
    assert answer.label == "bug"
    assert answer.probability == pytest.approx(0.8473)
    assert answer.confidence == pytest.approx(0.5763)
    assert answer.probabilities == {"bug": 0.8473, "billing": 0.1421}


def test_systemone_parse_score(monkeypatch) -> None:
    """Ответ `score` сервера → DecisionAnswer(score, legend в raw)."""
    parsed = LLMClient._parse_systemone(
        {
            "model": "clef-flash:latest",
            "answers": {
                "q": {
                    "type": "score",
                    "score": 0.657,
                    "legend": {"0": "low", "1": "high"},
                    "probabilities": {"0": 0.4, "1": 0.6},
                    "confidence": 0.5,
                }
            },
            "usage": {"input_tokens": 150, "output_tokens": 0},
        }
    )
    answer = answer_from_parsed("q", parsed["q"])
    assert answer.kind == "score"
    assert answer.score == pytest.approx(0.657)
    assert answer.raw["legend"] == {"0": "low", "1": "high"}


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
