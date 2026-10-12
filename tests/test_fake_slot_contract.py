"""Заглушка-провайдер контура: контракт decision-слота (приёмка 3.2.0, сценарий 9).

Зачем этот тест. Ветви сценария 9 приёмки (`scripts/slot_fake.sh on decision-*`)
наводят слот `decision` на `scripts/fake_slot.py`. Заглушка знала только СПИСОК
вопросов с полем `name` (форма пилота LSBN-T-13), а релиз шлёт `questions`
ОБЪЕКТОМ «имя → вопрос» (`type`/`instructions`/`criteria`) — как ollama ≥ 0.35.1
(`llm_client._systemone_payload`). Заглушка отвечала 422, а `check_decision()`
читает 422 как «модель не умеет decision» → слот гасится
(`/health.decision` = `enabled=true, ok=false, answers=0`), и ни одного решения
не спрашивается — весь блок decision в приёмке краснеет на живом, работающем
продукте (прогон 12.10.2026).

Тест поднимает НАСТОЯЩУЮ заглушку в потоке и прогоняет через неё реальный клиент
слота: стартовую проверку, решение судьи и две причины переключения. Это ровно
та связка «обвязка ↔ контракт», которой в юнитах не было.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Iterator

import httpx
import pytest

from app.config import get_settings
from app.services.decision import DecisionService, DecisionUnavailable
from app.services.llm_client import (
    DECISION_CHECK_AUTH_FAILED,
    DECISION_CHECK_OK,
    DECISION_CHECK_UNSUPPORTED,
    LLMClient,
    SlotSpec,
)
from app.services.prompts import PromptRegistry
from scripts.fake_slot import DEFAULT_MODELS, FakeSlotServer

MODE_OK = "decision-ok"
MODE_LOW = "decision-low-confidence"
MODE_UNSUPPORTED = "decision-unsupported"

# Порог и зона контура (docker-compose.test.yml — дефолты релиза).
THRESHOLD = "0.5"
ZONE = "0.05"


@contextlib.contextmanager
def running_fake(mode: str) -> Iterator[str]:
    """Поднять настоящую заглушку контура в потоке; yield — её базовый URL."""
    server = FakeSlotServer(("127.0.0.1", 0), mode, DEFAULT_MODELS, False)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://{server.server_address[0]}:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5.0)


def decision_slot(monkeypatch: pytest.MonkeyPatch, url: str) -> tuple[object, LLMClient]:
    """Настройки и клиент слота decision, наведённые на заглушку."""
    monkeypatch.setenv("DECISION_ENABLED", "true")
    monkeypatch.setenv("DECISION_BASE_URL", url)
    monkeypatch.setenv("DECISION_MODEL", "fake-decision")
    monkeypatch.setenv("DECISION_THRESHOLD", THRESHOLD)
    monkeypatch.setenv("DECISION_UNCERTAINTY_ZONE", ZONE)
    get_settings.cache_clear()
    settings = get_settings()
    get_settings.cache_clear()
    return settings, LLMClient(SlotSpec.for_decision(settings))


@pytest.mark.parametrize(
    ("mode", "expected"),
    (
        (MODE_OK, DECISION_CHECK_OK),
        (MODE_LOW, DECISION_CHECK_OK),
        (MODE_UNSUPPORTED, DECISION_CHECK_UNSUPPORTED),
        ("auth", DECISION_CHECK_AUTH_FAILED),
    ),
)
def test_startup_check_reads_the_contour_fake_honestly(
    monkeypatch: pytest.MonkeyPatch, mode: str, expected: str
) -> None:
    """`check_decision()` против заглушки: ok / unsupported / auth — по делу."""
    with running_fake(mode) as url:
        _settings, client = decision_slot(monkeypatch, url)
        try:
            assert client.check_decision() == expected
        finally:
            client.close()


def test_fake_accepts_both_question_shapes() -> None:
    """Контракт сервера — объект вопросов; форма пилота (список) тоже принята."""
    object_form = {
        "model": "fake-decision",
        "state": "probe",
        "questions": {
            "probe": {"type": "noul", "instructions": "Is it a probe?"},
        },
    }
    legacy_form = {
        "model": "fake-decision",
        "state": "probe",
        "questions": [{"name": "probe", "type": "noul", "text": "Is it a probe?"}],
    }
    choice_form = {
        "model": "fake-decision",
        "state": "note",
        "questions": {
            "note_section": {
                "type": "choice",
                "instructions": "Which section does the note belong to?",
                "criteria": {"work": None, "generic": None},
            },
        },
    }
    with running_fake(MODE_OK) as url:
        for payload in (object_form, legacy_form, choice_form):
            answer = httpx.post(f"{url}/v1/systemone", json=payload, timeout=5.0)
            assert answer.status_code == 200, payload
            assert answer.json()["answers"], payload
        # Метки `choice` — ключи `criteria` (контракт сервера), а не поле `labels`.
        labels = httpx.post(f"{url}/v1/systemone", json=choice_form, timeout=5.0)
        assert labels.json()["answers"]["note_section"]["choice"] == "work"


def test_judge_decides_through_the_fake_slot(monkeypatch: pytest.MonkeyPatch) -> None:
    """Судья получает решение от заглушки: вероятность, метка, счётчик решений."""
    with running_fake(MODE_OK) as url:
        settings, client = decision_slot(monkeypatch, url)
        try:
            service = DecisionService(
                settings, llm=client, registry=PromptRegistry()
            )
            questions = service.questions_for("judge")
            assert [question.name for question in questions] == ["is_text_duplicate"]
            result = service.decide(
                "TEXT 1: the note\nTEXT 2: the same note",
                questions,
                operation="judge",
            )
            answer = result.answers["is_text_duplicate"]
            assert answer.probability == pytest.approx(0.1)
            assert answer.label == "false"  # уверенное «нет» — слияния нет
            snapshot = service.snapshot()
            assert snapshot["answers"] == 1
            assert snapshot["fallbacks"] == 0
            assert snapshot["by_reason"] == {
                "unavailable": 0,
                "unsupported": 0,
                "low_confidence": 0,
            }
        finally:
            client.close()


def test_low_confidence_and_unsupported_are_counted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Вероятность в зоне → `low_confidence`; погашенный слот → `unsupported`."""
    with running_fake(MODE_LOW) as url:
        settings, client = decision_slot(monkeypatch, url)
        try:
            service = DecisionService(
                settings, llm=client, registry=PromptRegistry()
            )
            questions = service.questions_for("judge")
            with pytest.raises(DecisionUnavailable) as low:
                service.decide("TEXT 1: a\nTEXT 2: b", questions, operation="judge")
            assert low.value.reason == "low_confidence"
            snapshot = service.snapshot()
            assert snapshot["answers"] == 0
            assert snapshot["fallbacks"] == 1
            assert snapshot["by_reason"]["low_confidence"] == 1
        finally:
            client.close()

    with running_fake(MODE_UNSUPPORTED) as url:
        settings, client = decision_slot(monkeypatch, url)
        try:
            assert client.check_decision() == DECISION_CHECK_UNSUPPORTED
            service = DecisionService(
                settings, llm=client, registry=PromptRegistry()
            )
            service.disable()  # так делает main.py, прочитав исход проверки
            questions = service.questions_for("judge")
            with pytest.raises(DecisionUnavailable) as unsupported:
                service.decide("TEXT 1: a\nTEXT 2: b", questions, operation="judge")
            assert unsupported.value.reason == "unsupported"
            snapshot = service.snapshot()
            assert snapshot["ok"] is False
            assert snapshot["by_reason"]["unsupported"] == 1
        finally:
            client.close()
