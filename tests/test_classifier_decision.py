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
import pathlib

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
    body: dict = {"type": "choice", "choice": label}
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


def make_classifier(
    settings,
    decision_actions: list,
    generative_actions: list,
    registry: PromptRegistry | None = None,
):
    """Сервис разметки на decision-слоте; `registry` — чтобы подсунуть свой файл вопросов."""
    registry = registry or PromptRegistry(None)
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
    assert list(body["questions"]) == [CLASSIFY_QUESTION]
    question = body["questions"][CLASSIFY_QUESTION]
    assert question["type"] == "choice"
    # Варианты choice — ключи `criteria`: путь узла -> его описание (LSBN-F-37),
    # плюс резервная метка `generic` со своей формулировкой.
    assert list(question["criteria"]) == ["work/sbos2020", "life", GENERIC_LABEL]
    assert question["criteria"]["work/sbos2020"] == "рабочий проект"
    assert question["criteria"]["life"] == "быт"
    assert question["criteria"][GENERIC_LABEL] == "the note is generic and not bound to a known section"
    # state — только заметка (обрезанная по DECISION_NOTE_CHARS): карта узлов
    # ушла в criteria, поэтому state не содержит списка известных узлов.
    assert body["state"] == TEXT


def test_state_is_note_only_and_cut(monkeypatch, caplog) -> None:
    """LSBN-F-37: state — заметка без списка узлов, обрезанная по DECISION_NOTE_CHARS.

    Пустое описание узла уезжает в criteria как `null` (голая метка): сервер
    такой вариант принимает, а карта узлов не раздувает state.
    """
    settings = decision_settings(monkeypatch, DECISION_NOTE_CHARS="100")
    long_text = "я" * 250
    nodes = [
        {"path": "work/sbos2020", "description": ""},
        {"path": "life", "description": "быт"},
    ]
    classifier, decider, decision, generative = make_classifier(
        settings,
        [choice("life", {"life": 0.9, "work/sbos2020": 0.05, GENERIC_LABEL: 0.05})],
        [chat_body()],
    )
    result = classifier.classify(long_text, nodes)
    assert result.hint_path == "life"
    body = payloads(decision)[0]
    assert body["state"] == long_text[: settings.decision_note_chars]
    criteria = body["questions"][CLASSIFY_QUESTION]["criteria"]
    assert criteria["work/sbos2020"] is None
    assert criteria["life"] == "быт"
    assert criteria[GENERIC_LABEL] == "the note is generic and not bound to a known section"


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


def test_registry_wordings_survive_the_node_map(monkeypatch, tmp_path) -> None:
    """Формулировки decision_questions.json не теряются в criteria (ревью LSBN-F-37).

    Слияние: сначала реестр, потом живая карта узлов. Описание узла перекрывает
    критерий своего пути, а формулировки файла для остальных меток остаются —
    иначе тюнинг вопроса в файле умирал бы молча.
    """
    path = tmp_path / "decision_questions.json"
    path.write_text(
        json.dumps(
            {
                "classify": [
                    {
                        "name": CLASSIFY_QUESTION,
                        "text": "Which section does the note belong to?",
                        "kind": "choice",
                        "labels": [GENERIC_LABEL],
                        "criteria": {
                            GENERIC_LABEL: "формулировка из файла реестра",
                            "life": "быт из файла",
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    settings = decision_settings(monkeypatch)
    classifier, decider, decision, generative = make_classifier(
        settings,
        [choice("work/sbos2020", {"work/sbos2020": 0.9, GENERIC_LABEL: 0.1})],
        [chat_body()],
        registry=PromptRegistry(tmp_path),
    )
    assert classifier.classify(TEXT, KNOWN).hint_path == "work/sbos2020"
    criteria = payloads(decision)[0]["questions"][CLASSIFY_QUESTION]["criteria"]
    # Формулировка файла для `generic` жива: её не перетирает GENERIC_CRITERION.
    assert criteria[GENERIC_LABEL] == "формулировка из файла реестра"
    # Живая карта сильнее файла по тому же пути.
    assert criteria["life"] == "быт"
    # Узел, которого в файле нет, тоже на месте.
    assert criteria["work/sbos2020"] == "рабочий проект"


# --- decision не применим: выбирать не из чего --------------------------------


def test_single_candidate_skips_decision(monkeypatch, caplog) -> None:
    """Один кандидат (пустая карта узлов) → вызова решения НЕТ, идём генеративно.

    `choice`-вопрос с единственной меткой сервер отвергает (400 «criteria must
    contain 2–26 candidates»), поэтому при кандидатах меньше двух решение не
    спрашивается вовсе: событие `decision_skipped`, счётчики переключений не
    растут, разрыв decision-слота не двигается.
    """
    settings = decision_settings(monkeypatch)
    with caplog.at_level(logging.INFO, logger="app"):
        classifier, decider, decision, generative = make_classifier(
            settings,
            [choice(GENERIC_LABEL, {GENERIC_LABEL: 0.9})],
            [chat_body("life", 0.61)],
        )
        result = classifier.classify(TEXT, [])
    assert result.hint_path == "life"
    assert result.confidence == pytest.approx(0.61)
    assert decision.calls == 0
    assert generative.calls == 1
    snapshot = decider.snapshot()
    assert snapshot["fallbacks"] == 0
    assert snapshot["by_reason"] == {
        "unavailable": 0,
        "unsupported": 0,
        "low_confidence": 0,
    }
    assert snapshot["breaker"]["state"] == "closed"
    records = events(caplog, "decision_skipped")
    assert records and records[0].reason == "single_candidate"
    assert records[0].candidates == 1
    assert records[0].slot == "decision"
    assert records[0].operation == "classify"
    # Отказа не было — штатных переключений (и их событий) тоже нет.
    assert not events(caplog, "decision_fallback")
    assert not events(caplog, "slot_paused")


def test_single_candidate_is_not_a_fallback_switch(monkeypatch) -> None:
    """«Не применим» — не переключение: и при `DECISION_FALLBACK=false` идём генеративно."""
    settings = decision_settings(monkeypatch, DECISION_FALLBACK="false")
    classifier, decider, decision, generative = make_classifier(
        settings, [choice(GENERIC_LABEL, {GENERIC_LABEL: 0.9})], [chat_body("life", 0.5)]
    )
    result = classifier.classify(TEXT, [])
    assert result.hint_path == "life"
    assert result.confidence == pytest.approx(0.5)
    assert decision.calls == 0
    assert decider.snapshot()["fallbacks"] == 0


def test_two_candidates_call_decision(monkeypatch, caplog) -> None:
    """Два кандидата (узел + `generic`) — решение по-прежнему вызывается (регресс)."""
    settings = decision_settings(monkeypatch)
    one_node = [{"path": "work/sbos2020", "description": "рабочий проект"}]
    with caplog.at_level(logging.INFO, logger="app"):
        classifier, decider, decision, generative = make_classifier(
            settings,
            [choice("work/sbos2020", {"work/sbos2020": 0.9, GENERIC_LABEL: 0.1})],
            [chat_body()],
        )
        result = classifier.classify(TEXT, one_node)
    assert result.hint_path == "work/sbos2020"
    assert result.confidence == pytest.approx(0.9)
    assert result.meta is not None and result.meta.model_type == "decision"
    assert decision.calls == 1
    assert generative.calls == 0
    question = payloads(decision)[0]["questions"][CLASSIFY_QUESTION]
    assert list(question["criteria"]) == ["work/sbos2020", GENERIC_LABEL]
    assert not events(caplog, "decision_skipped")


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
    assert [list(body["questions"]) for body in payloads(decision)] == [
        [CLASSIFY_QUESTION],
        [CLASSIFY_QUESTION],
    ]
