"""Совместимость после волны decision (LSBN-F-12, релиз 3.2.0; arch §13.7).

Регресс: внешние контракты судьи (`DUPLICATE`/`NOT DUPLICATE`) и классификатора
(`hint_path`+`confidence`) после перевода на decision-слот НЕ изменились; набор
«дубль / не дубль» на фиксированном фейковом транспорте даёт те же вердикты;
журнал джоб обратно совместим — обязательные поля блока 2 (`event`/`job`/`slot`/
`outcome`) на месте, новые поля только добавлены, старые не удалены.
"""

from __future__ import annotations

import logging

import httpx

from app.services.judge import JudgeService
from app.services.prompts import PromptRegistry
from app.services.slot_outcome import SlotOutcome
from app.services.worker import BackgroundWorker
from fakes import FailingEmbedder

from test_classifier_decision import (
    GENERIC_LABEL,
    KNOWN,
    TEXT,
    chat_body as classify_body,
    choice,
    make_classifier,
)
from test_judge_decision import (
    TEXT_CANDIDATE,
    TEXT_NEW,
    chat_body,
    decision_settings,
    make_judge,
    noul,
)


def test_judge_and_classifier_contracts(monkeypatch) -> None:
    """Внешние контракты судьи и классификатора после волны не изменились."""
    settings = decision_settings(monkeypatch)
    # Судья: решение приходит от decision, но вердикт — прежний bool.
    judge, _, _, _ = make_judge(settings, [noul(0.95)], [chat_body()])
    assert judge.judge(TEXT_NEW, TEXT_CANDIDATE).verdict is True
    assert JudgeService._verdict("**DUPLICATE**") is True
    assert JudgeService._verdict("**NOT DUPLICATE**") is False
    # Классификатор: метка → `hint_path`, вероятность метки → `confidence` 0..1.
    classifier, _, _, _ = make_classifier(
        settings, [choice("life", {"life": 0.9})], [classify_body()]
    )
    result = classifier.classify(TEXT, KNOWN)
    assert result.hint_path == "life"
    assert 0.0 <= result.confidence <= 1.0


def test_dedup_behaviour_unchanged(monkeypatch) -> None:
    """Набор «дубль / не дубль» на фиксированном транспорте даёт те же вердикты."""
    settings = decision_settings(monkeypatch)
    for content, expected in (
        ("**DUPLICATE**", True),
        ("**NOT DUPLICATE**", False),
    ):
        judge = JudgeService(
            settings,
            httpx.MockTransport(lambda request, body=content: chat_body(body)),
            registry=PromptRegistry(None),
        )
        assert judge.judge(TEXT_NEW, TEXT_CANDIDATE).verdict is expected


def test_decision_skipped_is_not_fallback(monkeypatch, caplog) -> None:
    """«Решение не применимо» — не переключение: `decision_skipped`, счётчики и разрыв целы (FR-4.5).

    У `choice`-вопроса один кандидат (выбирать не из чего): вызова решения нет,
    событие `decision_skipped` (не `decision_fallback`), `fallbacks`/`by_reason`
    и разрыв decision-слота не двигаются — сразу генеративный путь.
    """
    settings = decision_settings(monkeypatch)
    with caplog.at_level(logging.INFO, logger="app"):
        classifier, decider, decision, generative = make_classifier(
            settings,
            [choice(GENERIC_LABEL, {GENERIC_LABEL: 0.9})],
            [classify_body()],
        )
        classifier.classify(TEXT, [])
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
    skipped = [
        record
        for record in caplog.records
        if getattr(record, "event", None) == "decision_skipped"
    ]
    assert skipped and skipped[0].reason == "single_candidate"
    assert not [
        record
        for record in caplog.records
        if getattr(record, "event", None) == "decision_fallback"
    ]


def test_journal_backward_compatible(monkeypatch, caplog) -> None:
    """Журнал джоб: обязательные поля блока 2 на месте, новые — только добавлены."""
    settings = decision_settings(monkeypatch)
    worker = BackgroundWorker(settings, FailingEmbedder())
    with caplog.at_level(logging.INFO, logger="app"):
        worker._log_slot_failure(
            "judge", "judge", SlotOutcome.AUTH, event="slot_call_failed", note_id=7
        )
    records = [
        record
        for record in caplog.records
        if getattr(record, "event", None) == "slot_call_failed"
    ]
    assert records, "ожидалось событие отказа слота"
    record = records[0]
    assert record.event == "slot_call_failed"
    assert record.job == "judge"
    assert record.slot == "judge"
    assert record.outcome == "auth"
    assert record.note_id == 7
