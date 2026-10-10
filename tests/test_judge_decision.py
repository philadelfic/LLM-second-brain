"""Судья дедупа на decision-слоте (LSBN-F-12, релиз 3.2.0; arch §6.3/§7.3/§13.6).

Судья отдаёт ТОТ ЖЕ внешний контракт (`JudgeResult.verdict: bool`), но решение
может прийти от decision-слота: state — боевой user-шаблон `judge_user`, один
`noul`-вопрос реестра, `label == "true"` → вердикт «ДУБЛЬ». Переключение на
генеративный путь — по трём причинам (`unavailable`/`unsupported`/
`low_confidence`) с записью `decision_fallback` в журнал; при
`DECISION_FALLBACK=false` генеративный вердикт НЕ берётся (контракт №39 —
задание ждёт). Выключенный decision (`decider=None`) — ровно прежнее поведение.

Юнит без сети: decision-клиент и генеративный слот — на httpx.MockTransport.
"""

from __future__ import annotations

import json
import logging

import httpx
import pytest

from app.config import get_settings
from app.services.decision import DecisionService
from app.services.judge import JudgeError, JudgeResult, JudgeService
from app.services.llm_client import LLMClient, SlotSpec
from app.services.prompts import PromptRegistry
from app.services.slot_outcome import SlotOutcome

TEXT_NEW = "По дороге домой зашёл в магазин: взял молоко и хлеб."
TEXT_CANDIDATE = "Купил молоко и хлеб по дороге домой."
JUDGE_QUESTION = "is_text_duplicate"


def make_settings(monkeypatch: pytest.MonkeyPatch, **env: str):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    get_settings.cache_clear()
    settings = get_settings()
    get_settings.cache_clear()
    return settings


def decision_settings(monkeypatch: pytest.MonkeyPatch, **env: str):
    """Настройки с включённым decision-слотом (генеративный судья — из conftest)."""
    return make_settings(
        monkeypatch,
        DECISION_ENABLED="true",
        DECISION_BASE_URL="http://127.0.0.1:11434",
        DECISION_MODEL="decision-test",
        **env,
    )


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


def systemone(answers: dict) -> httpx.Response:
    """Ответ `/v1/systemone` (фактическая форма сервера, сверено 2026-10-10)."""
    return httpx.Response(200, json={"answers": answers})


def noul(probability: float) -> httpx.Response:
    return systemone({JUDGE_QUESTION: {"type": "noul", "noul": probability}})


def chat_body(content: str = "**DUPLICATE**\n\nBoth notes say the same.") -> httpx.Response:
    """Ответ генеративного слота judge (`/api/chat`)."""
    return httpx.Response(
        200,
        json={
            "model": "test-judge-model",
            "message": {"role": "assistant", "content": content},
            "done": True,
        },
    )


def make_judge(
    settings,
    decision_actions: list,
    generative_actions: list,
    registry: PromptRegistry | None = None,
):
    """Судья с decision-слотом на MockTransport и генеративным путём-дублёром."""
    registry = registry if registry is not None else PromptRegistry(None)
    decision_recorder = Recorder(decision_actions)
    decider = DecisionService(
        settings,
        llm=LLMClient(
            SlotSpec.for_decision(settings), transport=decision_recorder.handler
        ),
        registry=registry,
    )
    generative = Recorder(generative_actions)
    judge = JudgeService(
        settings,
        httpx.MockTransport(generative.handler),
        registry=registry,
        decider=decider,
    )
    return judge, decider, decision_recorder, generative


def payloads(recorder: Recorder) -> list[dict]:
    return [json.loads(request.read().decode()) for request in recorder.requests]


def events(caplog, name: str) -> list:
    return [record for record in caplog.records if getattr(record, "event", None) == name]


# --- вердикт из решения -------------------------------------------------------


def test_verdict_from_noul_true(monkeypatch, caplog) -> None:
    """`noul=0.95` → дубль; `noul=0.02` → не дубль; генеративный путь не зван."""
    settings = decision_settings(monkeypatch)
    with caplog.at_level(logging.INFO, logger="app"):
        judge, decider, decision, generative = make_judge(
            settings, [noul(0.95)], [chat_body()]
        )
        result = judge.judge(TEXT_NEW, TEXT_CANDIDATE)
        assert isinstance(result, JudgeResult)
        assert result.verdict is True
        assert result.meta.model_type == "decision"
        assert generative.calls == 0

        judge, decider, decision, generative = make_judge(
            settings, [noul(0.02)], [chat_body()]
        )
        assert judge.judge(TEXT_NEW, TEXT_CANDIDATE).verdict is False
    assert decision.calls == 1
    assert events(caplog, "decision_fallback") == []


def test_decision_call_uses_pilot_state(monkeypatch, caplog) -> None:
    """`state` — боевой `judge_user`; вопрос — реестровый `noul`, формулировка в `instructions`."""
    settings = decision_settings(monkeypatch)
    registry = PromptRegistry(None)
    judge, decider, decision, generative = make_judge(
        settings, [noul(0.95)], [chat_body()], registry
    )
    judge.judge(TEXT_NEW, TEXT_CANDIDATE)
    assert decision.calls == 1
    request = decision.requests[0]
    assert request.url.path == "/v1/systemone"
    body = json.loads(request.read().decode())
    assert body["model"] == "decision-test"
    assert body["state"] == registry.judge_user.format(
        text_new=TEXT_NEW, text_candidate=TEXT_CANDIDATE
    )
    assert len(body["questions"]) == 1
    assert list(body["questions"]) == [JUDGE_QUESTION]
    question = body["questions"][JUDGE_QUESTION]
    assert question["type"] == "noul"
    assert question["instructions"] == registry.decision_questions["judge"][0].text
    assert question["criteria"] == {
        "true": "both texts carry the same fact with matching details",
        "false": "the texts differ in facts, details or intent",
    }


# --- переключение на генеративный слот ----------------------------------------


def test_fallback_on_low_confidence(monkeypatch, caplog) -> None:
    """Вероятность в зоне → генеративный путь; `low_confidence` в журнале."""
    settings = decision_settings(monkeypatch)
    with caplog.at_level(logging.INFO, logger="app"):
        judge, decider, decision, generative = make_judge(
            settings, [noul(0.5)], [chat_body("**DUPLICATE**")]
        )
        result = judge.judge(TEXT_NEW, TEXT_CANDIDATE)
    assert result.verdict is True  # вердикт дал генеративный слот
    assert generative.calls == 1
    records = events(caplog, "decision_fallback")
    assert records, "ожидалось событие decision_fallback"
    assert records[0].slot == "decision"
    assert records[0].operation == "judge"
    assert records[0].reason == "low_confidence"
    assert records[0].probability == 0.5


def test_fallback_on_unavailable(monkeypatch, caplog) -> None:
    """Слот decision недоступен → генеративный путь; при открытом разрыве — без HTTP."""
    settings = decision_settings(monkeypatch)
    with caplog.at_level(logging.INFO, logger="app"):
        judge, decider, decision, generative = make_judge(
            settings, [httpx.Response(401, json={"error": "unauthorized"})], [chat_body()]
        )
        assert judge.judge(TEXT_NEW, TEXT_CANDIDATE).verdict is True
        assert generative.calls == 1
        assert decision.calls == 1
        # Разрыв открыт (auth) — решения ждут, а не бьют в слот снова.
        assert judge.judge(TEXT_NEW, TEXT_CANDIDATE).verdict is True
    assert decision.calls == 1
    records = [
        record for record in events(caplog, "decision_fallback")
        if record.reason == "unavailable"
    ]
    assert len(records) >= 1
    assert records[-1].operation == "judge"


def test_fallback_on_unsupported(monkeypatch, caplog) -> None:
    """Слот погашен (`can_decide()` ложно) → генеративный путь, `unsupported`."""
    settings = decision_settings(monkeypatch)
    with caplog.at_level(logging.INFO, logger="app"):
        judge, decider, decision, generative = make_judge(
            settings, [noul(0.95)], [chat_body()]
        )
        decider.disable()
        assert decider.can_decide() is False
        assert judge.judge(TEXT_NEW, TEXT_CANDIDATE).verdict is True
    assert generative.calls == 1
    assert decision.calls == 0
    records = events(caplog, "decision_fallback")
    assert records and records[0].reason == "unsupported"


def test_fallback_disabled_waits(monkeypatch, caplog) -> None:
    """`DECISION_FALLBACK=false` и слот недоступен → JudgeError UNAVAILABLE, без генерации."""
    settings = decision_settings(monkeypatch, DECISION_FALLBACK="false")
    with caplog.at_level(logging.INFO, logger="app"):
        judge, decider, decision, generative = make_judge(
            settings, [httpx.Response(401, json={"error": "unauthorized"})], [chat_body()]
        )
        with pytest.raises(JudgeError) as excinfo:
            judge.judge(TEXT_NEW, TEXT_CANDIDATE)
    assert excinfo.value.outcome is SlotOutcome.UNAVAILABLE
    assert generative.calls == 0  # вердикт генеративного пути не взят — задание ждёт


# --- регресс ------------------------------------------------------------------


def test_contract_unchanged() -> None:
    """`JudgeResult.verdict` — bool, парсер маркеров не изменён."""
    assert JudgeService._verdict("**DUPLICATE**") is True
    assert JudgeService._verdict("**NOT DUPLICATE**") is False
    with pytest.raises(JudgeError):
        JudgeService._verdict("I am not sure")


def test_decision_disabled_same_as_before(monkeypatch, caplog) -> None:
    """`decider=None` → ровно прежний генеративный вызов, поля журнала прежние."""
    settings = decision_settings(monkeypatch)
    generative = Recorder([chat_body("**NOT DUPLICATE**")])
    judge = JudgeService(
        settings, httpx.MockTransport(generative.handler), registry=PromptRegistry(None)
    )
    with caplog.at_level(logging.INFO, logger="app"):
        result = judge.judge(TEXT_NEW, TEXT_CANDIDATE)
    assert result.verdict is False
    assert generative.calls == 1
    body = payloads(generative)[0]
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][1]["content"] == f"TEXT 1:\n{TEXT_NEW}\n\nTEXT 2:\n{TEXT_CANDIDATE}"
    assert events(caplog, "decision_answer") == []
    assert events(caplog, "decision_fallback") == []
