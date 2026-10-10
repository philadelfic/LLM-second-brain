"""Разметка узла на decision-слоте (LSBN-F-12, релиз 3.2.0; arch §6.4/§7.3/§13.6).

Классификатор отдаёт ТОТ ЖЕ внешний контракт (`hint_path` 1..3 слага или
`null`, `confidence` 0..1), но решение может прийти от decision-слота: состояние
— заметка + известные узлы (как сегодня), один `choice`-вопрос реестра с метками
= пути узлов + резервная `generic`, метка → `hint_path`, вероятность метки →
`confidence`, путь по-прежнему валидируется `NamespaceService.validate_path`.
Переключение на генеративный путь — при недоступности/неподдержке слота, при
потолке меток (`DECISION_MAX_LABELS`) и при невалидной метке; вероятности живут
только в журнале `DecisionService`.

Юнит без сети: decision-клиент и генеративный слот — на httpx.MockTransport.
"""

from __future__ import annotations

import json
import logging

import httpx
import pytest

from app.config import get_settings
from app.services.classifier import Classification, ClassificationError, ClassificationService
from app.services.decision import DecisionService
from app.services.llm_client import LLMClient, SlotSpec
from app.services.prompts import PromptRegistry
from app.services.warnings import WarningsService

TEXT = "Заметка про рабочий проект SBOS-2020."
CLASSIFY_QUESTION = "note_section"
GENERIC_LABEL = "generic"

KNOWN = [
    {"path": "work/sbos2020", "description": "рабочий проект"},
    {"path": "life", "description": "быт"},
]


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
    return httpx.Response(200, json={"answers": answers})


def choice(label: str, probabilities: dict | None = None) -> httpx.Response:
    body: dict = {"choice": label}
    if probabilities is not None:
        body["probabilities"] = probabilities
    return systemone({CLASSIFY_QUESTION: body})


def chat_body(hint_path: str | None = "life", confidence: float = 0.42) -> httpx.Response:
    payload = json.dumps({"hint_path": hint_path, "confidence": confidence})
    return httpx.Response(
        200,
        json={
            "model": "test-summary-model",
            "message": {"role": "assistant", "content": payload},
            "done": True,
        },
    )


def make_classifier(settings, decision_actions: list, generative_actions: list):
    registry = PromptRegistry(None)
    decision_recorder = Recorder(decision_actions)
    decider = DecisionService(
        settings,
        llm=LLMClient(
            SlotSpec.for_decision(settings), transport=decision_recorder.handler
        ),
        registry=registry,
    )
    generative = Recorder(generative_actions)
    classifier = ClassificationService(
        settings,
        httpx.MockTransport(generative.handler),
        registry=registry,
        decider=decider,
    )
    return classifier, decider, decision_recorder, generative


def payloads(recorder: Recorder) -> list[dict]:
    return [json.loads(request.read().decode()) for request in recorder.requests]


def events(caplog, name: str) -> list:
    return [record for record in caplog.records if getattr(record, "event", None) == name]


# --- разметка из решения ------------------------------------------------------


def test_hint_path_from_choice(monkeypatch, caplog) -> None:
    """Метка `work/sbos2020` → `hint_path`; вероятность метки → `confidence`."""
    settings = decision_settings(monkeypatch)
    classifier, decider, decision, generative = make_classifier(
        settings,
        [choice("work/sbos2020", {"work/sbos2020": 0.9, "life": 0.06, "generic": 0.04})],
        [chat_body()],
    )
    result = classifier.classify(TEXT, KNOWN)
    assert result.hint_path == "work/sbos2020"
    assert result.confidence == pytest.approx(0.9)
    assert result.meta is not None and result.meta.model_type == "decision"
    assert generative.calls == 0
    body = payloads(decision)[0]
    assert body["questions"][0]["name"] == CLASSIFY_QUESTION
    assert body["questions"][0]["type"] == "choice"
    assert body["questions"][0]["labels"] == ["work/sbos2020", "life", GENERIC_LABEL]
    assert body["state"] == ClassificationService._user_message(TEXT, KNOWN)


def test_generic_label_maps_to_null(monkeypatch, caplog) -> None:
    """Резервная метка `generic` → `hint_path is None`."""
    settings = decision_settings(monkeypatch)
    classifier, decider, decision, generative = make_classifier(
        settings,
        [choice("generic", {"generic": 0.8, "life": 0.1, "work/sbos2020": 0.1})],
        [chat_body()],
    )
    result = classifier.classify(TEXT, KNOWN)
    assert result.hint_path is None
    assert result.confidence == pytest.approx(0.8)
    assert generative.calls == 0


# --- переключение на генеративный слот ----------------------------------------


def test_invalid_label_falls_back(monkeypatch, caplog) -> None:
    """Метка не проходит `validate_path` → переключение, генеративный разбор."""
    settings = decision_settings(monkeypatch)
    with caplog.at_level(logging.INFO, logger="app"):
        classifier, decider, decision, generative = make_classifier(
            settings,
            [choice("default/secret", {"default/secret": 0.9})],
            [chat_body("life", 0.42)],
        )
        result = classifier.classify(TEXT, KNOWN)
    assert result.hint_path == "life"
    assert result.confidence == pytest.approx(0.42)
    assert generative.calls == 1
    assert decision.calls == 1


def test_too_many_labels_falls_back(monkeypatch, caplog) -> None:
    """Узлов больше `DECISION_MAX_LABELS` → переключение, decision-вызова нет."""
    settings = decision_settings(monkeypatch, DECISION_MAX_LABELS="3")
    many = KNOWN + [
        {"path": "work/x", "description": "x"},
        {"path": "work/y", "description": "y"},
        {"path": "life/z", "description": "z"},
    ]
    with caplog.at_level(logging.INFO, logger="app"):
        classifier, decider, decision, generative = make_classifier(
            settings, [choice("life", {"life": 0.9})], [chat_body("life", 0.5)]
        )
        result = classifier.classify(TEXT, many)
    assert result.hint_path == "life"
    assert decision.calls == 0
    assert generative.calls == 1
    records = events(caplog, "decision_fallback")
    assert records and records[0].reason == "unsupported"


# --- регресс ------------------------------------------------------------------


def test_classification_contract_unchanged(monkeypatch) -> None:
    """`Classification` несёт `hint_path` + `confidence` 0..1; разбор прежний."""
    settings = decision_settings(monkeypatch)
    service = ClassificationService(
        settings, httpx.MockTransport(lambda request: chat_body())
    )
    assert set(Classification.__dataclass_fields__) == {"hint_path", "confidence", "meta"}
    parsed = service._parse('{"hint_path": "work/sbos2020", "confidence": 0.7}')
    assert parsed.hint_path == "work/sbos2020"
    assert parsed.confidence == pytest.approx(0.7)
    assert service._parse('{"hint_path": null, "confidence": 0}').hint_path is None
    with pytest.raises(ClassificationError):
        service._parse('{"hint_path": "work/sbos2020", "confidence": 1.5}')
    with pytest.raises(ClassificationError):
        service._parse('{"hint_path": "default/secret", "confidence": 0.5}')


def test_warnings_and_nodes_share_decision(monkeypatch, caplog) -> None:
    """И `nodes`, и `warnings` ходят через один `classify` — вопрос разметки один."""
    settings = decision_settings(monkeypatch)
    classifier, decider, decision, generative = make_classifier(
        settings,
        [
            choice("generic", {"generic": 0.9}),
            choice("generic", {"generic": 0.9}),
        ],
        [chat_body()],
    )

    class FakeNamespaces:
        def list_all(self) -> dict:
            return {"namespaces": KNOWN}

    class FakeConn:
        def __init__(self, row: dict) -> None:
            self._row = row

        def execute(self, *args, **kwargs) -> "FakeConn":
            return self

        def fetchone(self) -> dict:
            return self._row

        def __enter__(self) -> "FakeConn":
            return self

        def __exit__(self, *exc) -> bool:
            return False

    row = {
        "title": "заметка",
        "summary": "краткое содержание",
        "namespace": "life",
        "summary_status": "ok",
    }
    warnings = WarningsService(
        settings,
        classifier=classifier,
        namespaces=FakeNamespaces(),
        conn_factory=lambda _settings: FakeConn(row),
    )
    # Джоба `warnings` («заметка не на своём месте»).
    assert warnings.check_note(1) is None
    # Джоба `nodes` (обход default) — тот же сервис и тот же вопрос разметки.
    assert classifier.classify(TEXT, KNOWN).hint_path is None
    assert decision.calls == 2
    assert [body["questions"][0]["name"] for body in payloads(decision)] == [
        CLASSIFY_QUESTION,
        CLASSIFY_QUESTION,
    ]
