"""Compatibility after the decision wave (LSBN-F-12, release 3.2.0; arch §13.7).

    Regression: the judge's external contracts (`DUPLICATE`/`NOT DUPLICATE`) and
    the classifier's (`hint_path`+`confidence`) did NOT change after the move to
    the decision slot; the "duplicate / not duplicate" set on a fixed fake
    transport gives the same verdicts; the jobs journal is backward compatible —
    the block-2 mandatory fields (`event`/`job`/`slot`/`outcome`) in place, new
    fields only added, old ones not removed.
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
    """The judge's and classifier's external contracts unchanged after the wave."""
    settings = decision_settings(monkeypatch)
    # The judge: the decision comes from decision, but the verdict — the same bool.
    judge, _, _, _ = make_judge(settings, [noul(0.95)], [chat_body()])
    assert judge.judge(TEXT_NEW, TEXT_CANDIDATE).verdict is True
    assert JudgeService._verdict("**DUPLICATE**") is True
    assert JudgeService._verdict("**NOT DUPLICATE**") is False
    # The classifier: the label → `hint_path`; the label's probability → `confidence` 0..1.
    classifier, _, _, _ = make_classifier(
        settings, [choice("life", {"life": 0.9})], [classify_body()]
    )
    result = classifier.classify(TEXT, KNOWN)
    assert result.hint_path == "life"
    assert 0.0 <= result.confidence <= 1.0


def test_dedup_behaviour_unchanged(monkeypatch) -> None:
    """The "duplicate / not duplicate" set on a fixed transport gives the same verdicts."""
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
    """"The decision is not applicable" — not a switch: `decision_skipped`; the counters and the breaker intact (FR-4.5).

                The `choice` question has one candidate (nothing to choose from): no
                decision call; a `decision_skipped` event (not `decision_fallback`);
                `fallbacks`/`by_reason` and the decision slot's breaker do not move —
                straight to the generative path.
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
    """The jobs journal: the block-2 mandatory fields in place; new ones only added."""
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
