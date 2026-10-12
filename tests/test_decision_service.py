"""Decision-слот: сервис решений (LSBN-F-12, релиз 3.2.0; arch §6.4/§13.6).

Юнит без сети: клиент на httpx.MockTransport. Покрывается: выключенный слот —
noop без вызовов (и без счётчиков — это не переключение, f12-04); порог и зона
дают три причины переключения (`low_confidence`/принято); свой разрыв слота
экономит вызовы и восстанавливается пробным стуком; потолок меток —
`unsupported` без вызова; словарь причин закрыт (ровно три ключа `by_reason`,
`fallbacks == sum`, чужое имя — `ValueError`).
"""

from __future__ import annotations

import logging

import httpx
import pytest

from app.config import get_settings
from app.services import build_services
from app.services.decision import (
    DecisionQuestion,
    DecisionService,
    DecisionUnavailable,
)
from app.services.llm_client import (
    DECISION_CHECK_AUTH_FAILED,
    DECISION_CHECK_UNREACHABLE,
    DECISION_CHECK_UNSUPPORTED,
    LLMClient,
    SlotSpec,
)
from app.services.prompts import PromptRegistry
from app.services.slot_outcome import SlotOutcome

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


def events_of(caplog, name: str) -> list:
    """Записи журнала с полем `event` = name (события приложения)."""
    return [
        record for record in caplog.records if getattr(record, "event", None) == name
    ]


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


# --- словарь причин (f12-04): что переключение, а что нет --------------------

ZERO_REASONS = {"unavailable": 0, "unsupported": 0, "low_confidence": 0}


def test_disabled_by_flag_moves_no_counters(monkeypatch, caplog) -> None:
    """`DECISION_ENABLED=false` — не переключение: счётчики, разрыв и журнал целы (FR-4.4).

    Причина отказа остаётся `unavailable` («слота сейчас нет»), но именно
    выключенный флагом слот переключением НЕ считается: ни `fallbacks`, ни
    `by_reason`, ни событие `decision_fallback` не двигаются, HTTP-вызова нет.
    """
    settings = make_settings(
        monkeypatch,
        DECISION_BASE_URL="http://localhost:11434",
        DECISION_MODEL="decision-test",
    )
    service, recorder = make_service(settings, [answers({"q": {"noul": 0.9}})])
    assert service.can_decide() is False
    with caplog.at_level(logging.INFO, logger="app"):
        with pytest.raises(DecisionUnavailable) as excinfo:
            service.decide(STATE, [noul_question()], operation="judge")
    assert excinfo.value.reason == "unavailable"
    snapshot = service.snapshot()
    assert snapshot["enabled"] is False
    assert snapshot["fallbacks"] == 0
    assert snapshot["by_reason"] == dict(ZERO_REASONS)
    assert snapshot["breaker"]["state"] == "closed"
    assert recorder.calls == 0
    assert not events_of(caplog, "decision_fallback")


def test_disabled_by_startup_check_is_unavailable(monkeypatch) -> None:
    """Слот ВКЛЮЧЁН, но погашен стартовой проверкой → `unavailable`, а не `unsupported` (§13.6).

    Дефект приёмки 12.10.2026: живой недоступный слот читался оператором как
    «модель не умеет» — а это «слота сейчас нет». Вызова по-прежнему нет,
    счётчик переключения растёт.
    """
    settings = decision_settings(monkeypatch)
    service, recorder = make_service(settings, [answers({"q": {"noul": 0.9}})])
    service.disable()
    assert service.can_decide() is False
    with pytest.raises(DecisionUnavailable) as excinfo:
        service.decide(STATE, [noul_question()], operation="judge")
    assert excinfo.value.reason == "unavailable"
    snapshot = service.snapshot()
    assert snapshot["enabled"] is True
    assert snapshot["ok"] is False
    assert snapshot["by_reason"]["unavailable"] == 1
    assert snapshot["by_reason"]["unsupported"] == 0
    assert recorder.calls == 0


def test_disabled_by_unsupported_check_is_unsupported(monkeypatch) -> None:
    """Погашен проверкой с исходом `unsupported` → причина переключения `unsupported` (§13.6).

    «Модель не умеет decision» — про контракт сервера, и оператор обязан видеть
    её отдельно от «слота сейчас нет» (дефект приёмки 12.10.2026). Вызова
    по-прежнему нет, растёт ровно счётчик `unsupported`, словарь — те же три
    ключа.
    """
    service, recorder = make_service(
        decision_settings(monkeypatch), [answers({"q": {"noul": 0.9}})]
    )
    service.disable(outcome=DECISION_CHECK_UNSUPPORTED)
    assert service.can_decide() is False
    with pytest.raises(DecisionUnavailable) as excinfo:
        service.decide(STATE, [noul_question()], operation="judge")
    assert excinfo.value.reason == "unsupported"
    snapshot = service.snapshot()
    assert snapshot["enabled"] is True
    assert snapshot["ok"] is False
    assert set(snapshot["by_reason"]) == {
        "unavailable",
        "unsupported",
        "low_confidence",
    }
    assert snapshot["by_reason"]["unsupported"] == 1
    assert snapshot["by_reason"]["unavailable"] == 0
    assert snapshot["fallbacks"] == 1
    assert recorder.calls == 0


@pytest.mark.parametrize(
    "outcome", (DECISION_CHECK_UNREACHABLE, DECISION_CHECK_AUTH_FAILED)
)
def test_disabled_by_transport_check_is_unavailable(monkeypatch, outcome) -> None:
    """Погашен недоступностью (`unreachable`/`auth_failed`) → `unavailable` («слота сейчас нет»).

    Сеть/таймаут/5xx и отказ авторизации не говорят ничего о контракте: причина
    переключения — доступность, а не «модель не умеет decision» (f12-04).
    """
    service, recorder = make_service(
        decision_settings(monkeypatch), [answers({"q": {"noul": 0.9}})]
    )
    service.disable(outcome=outcome)
    with pytest.raises(DecisionUnavailable) as excinfo:
        service.decide(STATE, [noul_question()], operation="judge")
    assert excinfo.value.reason == "unavailable"
    snapshot = service.snapshot()
    assert snapshot["by_reason"]["unavailable"] == 1
    assert snapshot["by_reason"]["unsupported"] == 0
    assert recorder.calls == 0


def test_paused_alive_slot_is_unavailable(monkeypatch) -> None:
    """Разрыв ЖИВОГО слота → `unavailable`: пауза — не контракт и не «не умеет» (§13.6).

    Слот включён и не погашен (`ok is True`), открыт только разрыв: причина
    переключения остаётся `unavailable`, счётчик `unsupported` не трогается.
    """
    service, recorder = make_service(
        decision_settings(monkeypatch), [answers({"q": {"noul": 0.9}})]
    )
    service._breaker.note_failure(SlotOutcome.AUTH)  # разрыв открыт, слот жив
    with pytest.raises(DecisionUnavailable) as excinfo:
        service.decide(STATE, [noul_question()], operation="judge")
    assert excinfo.value.reason == "unavailable"
    snapshot = service.snapshot()
    assert snapshot["ok"] is True
    assert snapshot["by_reason"]["unavailable"] == 1
    assert snapshot["by_reason"]["unsupported"] == 0
    assert recorder.calls == 0


def test_unsupported_is_contract_only(monkeypatch) -> None:
    """`unsupported` — строго контракт: маршрут, тип вопроса, потолок меток, нет решения (§13.6).

    Ничего из «доступности» сюда не попадает: ни сеть, ни разрыв слота, ни
    пауза — всё это `unavailable`.
    """
    settings = decision_settings(monkeypatch)
    contract_cases = (
        ("не знает маршрут /v1/systemone", httpx.Response(404, text="unknown route")),
        ("не принимает тип вопроса", httpx.Response(400, text="unsupported type")),
    )
    for title, action in contract_cases:
        service, recorder = make_service(settings, [action])
        with pytest.raises(DecisionUnavailable) as excinfo:
            service.decide(STATE, [noul_question()], operation="judge")
        assert excinfo.value.reason == "unsupported", title
        assert service.snapshot()["by_reason"]["unsupported"] == 1, title
        assert recorder.calls == 1, title
    # Вариантов в choice-вопросе больше потолка — отказ без вызова.
    labels_settings = decision_settings(monkeypatch, DECISION_MAX_LABELS="2")
    service, recorder = make_service(labels_settings, [httpx.Response(500, text="x")])
    question = DecisionQuestion(
        name="q",
        text="Where?",
        kind="choice",
        labels=("a", "b", "c"),
        criteria={"a": "1", "b": "2", "c": "3"},
    )
    with pytest.raises(DecisionUnavailable) as too_many:
        service.decide(STATE, [question], operation="classify")
    assert too_many.value.reason == "unsupported"
    assert service.snapshot()["by_reason"]["unsupported"] == 1
    assert recorder.calls == 0
    # В ответе нет решения этого вопроса.
    service, recorder = make_service(settings, [answers({})])
    with pytest.raises(DecisionUnavailable) as no_answer:
        service.decide(STATE, [noul_question()], operation="judge")
    assert no_answer.value.reason == "unsupported"
    assert service.snapshot()["by_reason"]["unsupported"] == 1


def test_unavailable_for_transport_and_pause(monkeypatch) -> None:
    """Доступность — только `unavailable`: сеть/5xx/таймаут/разрыв слота (§13.6)."""
    settings = decision_settings(monkeypatch)
    transport_cases = (
        ("сеть", httpx.ConnectError("refused")),
        ("5xx", httpx.Response(503, text="bad gateway")),
        ("таймаут", httpx.ReadTimeout("медленно", request=None)),
    )
    for title, action in transport_cases:
        service, recorder = make_service(settings, [action])
        with pytest.raises(DecisionUnavailable) as excinfo:
            service.decide(STATE, [noul_question()], operation="judge")
        assert excinfo.value.reason == "unavailable", title
        snapshot = service.snapshot()
        assert snapshot["by_reason"]["unavailable"] == 1, title
        assert snapshot["by_reason"]["unsupported"] == 0, title
        assert recorder.calls == 1, title
    # Разрыв открыт (401 = auth) и стук не наступил — отказ без HTTP-вызова.
    service, recorder = make_service(settings, [httpx.Response(401, text="nope")])
    with pytest.raises(DecisionUnavailable):
        service.decide(STATE, [noul_question()], operation="judge")
    with pytest.raises(DecisionUnavailable) as paused:
        service.decide(STATE, [noul_question()], operation="judge")
    assert paused.value.reason == "unavailable"
    assert recorder.calls == 1
    assert service.snapshot()["by_reason"]["unavailable"] == 2


def test_by_reason_keys_are_exactly_three(monkeypatch) -> None:
    """`by_reason` — всегда ровно три ключа, `fallbacks == sum(by_reason)` (§7.3/FR-6.3)."""
    expected = {"unavailable", "unsupported", "low_confidence"}
    service, _ = make_service(decision_settings(monkeypatch), [answers({})])
    fresh = service.snapshot()
    assert set(fresh["by_reason"]) == expected
    assert fresh["by_reason"] == dict(ZERO_REASONS)
    assert fresh["fallbacks"] == sum(fresh["by_reason"].values())
    with pytest.raises(DecisionUnavailable):
        service.decide(STATE, [noul_question()], operation="judge")
    snapshot = service.snapshot()
    assert set(snapshot["by_reason"]) == expected
    assert snapshot["fallbacks"] == sum(snapshot["by_reason"].values())


def test_unknown_reason_is_a_development_error(monkeypatch) -> None:
    """Причина вне словаря трёх — ошибка разработки (`ValueError`), а не новый счётчик (§13.6)."""
    service, _ = make_service(decision_settings(monkeypatch), [answers({})])
    with pytest.raises(ValueError):
        service._fallback("timeout", operation="judge")
    assert service.snapshot()["fallbacks"] == 0
    assert set(service.snapshot()["by_reason"]) == {
        "unavailable",
        "unsupported",
        "low_confidence",
    }
