"""The decision decisions journal (LSBN-F-12, release 3.2.0; arch §6.4/§7.2/§13.6).

    The decision's probabilities and duration are visible in the journal
    (`decision_answer`); the switch reason — in `decision_fallback`; the
    probabilities are NOT written to the DB (FR-3.3): no decisions table, no
    history fields, no migrations. The journal — the application's JSON events
    (logging), not a table.

    A unit without the network: a decision client on httpx.MockTransport; the DB
    — temporary (conftest).
"""

from __future__ import annotations

import logging

import httpx
import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.services.decision import SLOT_DECISION
from app.storage.db import init_db, session

from test_classifier_decision import (
    CLASSIFY_QUESTION,
    KNOWN,
    TEXT,
    choice,
    make_classifier,
)
from test_judge_decision import (
    JUDGE_QUESTION,
    TEXT_CANDIDATE,
    TEXT_NEW,
    chat_body,
    decision_settings,
    make_judge,
    noul,
    systemone,
)


def events(caplog, name: str) -> list:
    return [record for record in caplog.records if getattr(record, "event", None) == name]


def schema() -> dict[str, set[str]]:
    """The DB tables and their columns — a schema snapshot before/after the decisions."""
    with session(get_settings()) as conn:
        tables = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        ]
        return {
            table: {
                row["name"] for row in conn.execute(f"PRAGMA table_info({table})")
            }
            for table in tables
        }


def run_judge_decision(settings, caplog) -> None:
    judge, decider, decision, generative = make_judge(
        settings, [noul(0.95)], []
    )
    assert judge.judge(TEXT_NEW, TEXT_CANDIDATE).verdict is True
    assert decision.calls == 1


def run_classify_decision(settings, caplog) -> None:
    classifier, decider, decision, generative = make_classifier(
        settings,
        [choice("work/sbos2020", {"work/sbos2020": 0.9, "generic": 0.1})],
        [],
    )
    assert classifier.classify(TEXT, KNOWN).hint_path == "work/sbos2020"
    assert decision.calls == 1


def test_decision_answer_probabilities_logged(monkeypatch, caplog) -> None:
    """`decision_answer` carries the probability, the threshold, the model and `duration_ms`."""
    from test_judge_decision import decision_settings

    settings = decision_settings(monkeypatch)
    with caplog.at_level(logging.INFO, logger="app"):
        run_judge_decision(settings, caplog)
        run_classify_decision(settings, caplog)
    records = events(caplog, "decision_answer")
    assert len(records) == 2
    judge_event, classify_event = records
    assert judge_event.slot == SLOT_DECISION
    assert judge_event.operation == "judge"
    assert judge_event.questions == [JUDGE_QUESTION]
    answer = judge_event.answers[JUDGE_QUESTION]
    assert answer["kind"] == "noul"
    assert answer["label"] == "true"
    assert answer["probability"] == pytest.approx(0.95)
    # LSBN-F-37 (ревью): одного `threshold` на всё событие нет — фактически
    # применённый порог и его шкала пишутся на каждый ответ.
    assert not hasattr(judge_event, "threshold")
    assert answer["applied_threshold"] == pytest.approx(0.5)
    assert answer["applied_scale"] == "probability"
    assert judge_event.model == "decision-test"
    assert isinstance(judge_event.duration_ms, int)
    assert judge_event.duration_ms >= 0
    assert classify_event.operation == "classify"
    assert classify_event.questions == [CLASSIFY_QUESTION]
    assert classify_event.answers[CLASSIFY_QUESTION]["label"] == "work/sbos2020"
    assert classify_event.answers[CLASSIFY_QUESTION]["probability"] == pytest.approx(0.9)
    # Метки без confidence: сравнение по голой probability и DECISION_THRESHOLD.
    classify_answer = classify_event.answers[CLASSIFY_QUESTION]
    assert classify_answer["applied_scale"] == "probability"
    assert classify_answer["applied_threshold"] == pytest.approx(0.5)


def test_no_probabilities_in_db(monkeypatch, caplog) -> None:
    """After the decisions the DB schema did not change: no decisions tables/fields."""
    from test_judge_decision import decision_settings

    settings = decision_settings(monkeypatch)
    init_db(get_settings())
    before = schema()
    with caplog.at_level(logging.INFO, logger="app"):
        run_judge_decision(settings, caplog)
        run_classify_decision(settings, caplog)
    after = schema()
    assert after == before
    names = {f"{table}.{column}" for table, columns in after.items() for column in columns}
    assert not [name for name in names if "decision" in name.lower()]
    assert not [name for name in names if "probab" in name.lower()]


# --- the events contract (f12-03: observability and tests) ------------------


def test_decision_check_fields(monkeypatch, caplog) -> None:
    """The startup check's `decision_check` event carries `model` and `result`."""
    from app.main import create_app
    from app.services.llm_client import LLMClient

    monkeypatch.setattr(LLMClient, "check_decision", lambda self: "unsupported")
    settings = decision_settings(monkeypatch)
    app = create_app(settings)
    with caplog.at_level(logging.WARNING, logger="app"):
        with TestClient(app):
            pass
    records = events(caplog, "decision_check")
    assert records, "ожидалось событие decision_check"
    assert records[0].slot == SLOT_DECISION
    assert records[0].model == "decision-test"
    assert records[0].result == "unsupported"


def test_decision_answer_fields(monkeypatch, caplog) -> None:
    """`decision_answer`: operation/applied_threshold/applied_scale/duration_ms/answers.

    LSBN-F-37 (ревью): порог и шкала — на каждый ответ; вероятности — только в журнале.
    """
    settings = decision_settings(monkeypatch)
    with caplog.at_level(logging.INFO, logger="app"):
        run_judge_decision(settings, caplog)
    (record,) = events(caplog, "decision_answer")
    assert record.operation == "judge"
    assert not hasattr(record, "threshold")
    assert isinstance(record.duration_ms, int)
    answer = record.answers[JUDGE_QUESTION]
    assert answer["probability"] == pytest.approx(0.95)
    assert answer["applied_threshold"] == pytest.approx(0.5)
    assert answer["applied_scale"] == "probability"


def test_decision_answer_applied_confidence_scale(monkeypatch, caplog) -> None:
    """choice по confidence: applied_scale=confidence и порог choice (LSBN-F-37)."""
    settings = decision_settings(monkeypatch)
    classifier, decider, decision, generative = make_classifier(
        settings,
        [
            systemone(
                {CLASSIFY_QUESTION: {"choice": "work/sbos2020", "confidence": 0.8}}
            )
        ],
        [],
    )
    with caplog.at_level(logging.INFO, logger="app"):
        assert classifier.classify(TEXT, KNOWN).hint_path == "work/sbos2020"
    (record,) = events(caplog, "decision_answer")
    answer = record.answers[CLASSIFY_QUESTION]
    assert answer["applied_scale"] == "confidence"
    assert answer["applied_threshold"] == pytest.approx(
        settings.decision_choice_threshold
    )
    assert answer["probability"] is None
    assert answer["confidence"] == pytest.approx(0.8)


def test_decision_fallback_reason_enum(monkeypatch, caplog) -> None:
    """The switch's `reason` — exactly three values; no others (§13.6/FR-4.2).

                The whole dictionary is checked: the uncertainty zone
                (`low_confidence`), the breaker and the network (`unavailable`), the
                contract with the server (`unsupported`). A foreign name is never
                accepted as a reason — in development that is an error
                (`ValueError`), not a fourth counter.
    """
    settings = decision_settings(monkeypatch)
    allowed = {"unavailable", "unsupported", "low_confidence"}
    with caplog.at_level(logging.INFO, logger="app"):
        judge, decider, decision, generative = make_judge(
            settings, [noul(0.5)], [chat_body("**DUPLICATE**")]
        )
        judge.judge(TEXT_NEW, TEXT_CANDIDATE)  # low_confidence
        judge_bad, _, _, _ = make_judge(
            settings,
            [httpx.Response(401, json={"error": "unauthorized"})],
            [chat_body()],
        )
        judge_bad.judge(TEXT_NEW, TEXT_CANDIDATE)  # unavailable
        judge_contract, _, _, _ = make_judge(
            settings, [systemone({})], [chat_body()]
        )
        judge_contract.judge(TEXT_NEW, TEXT_CANDIDATE)  # unsupported (no decision)
        reasons = {record.reason for record in events(caplog, "decision_fallback")}
        assert reasons == allowed
        assert set(decider.snapshot()["by_reason"]) == allowed
        with pytest.raises(ValueError):
            decider._fallback("timeout", operation="judge")
    assert set(decider.snapshot()["by_reason"]) == allowed
    assert decider.snapshot()["fallbacks"] == sum(
        decider.snapshot()["by_reason"].values()
    )


def test_decision_paused_events(monkeypatch, caplog) -> None:
    """The decision breaker writes `slot_paused`/`slot_probe`/`slot_resumed` (block 2)."""
    settings = decision_settings(monkeypatch)
    clock = {"t": 1000.0}
    monkeypatch.setattr("app.services.slot_breaker.time.time", lambda: clock["t"])
    judge, decider, decision, generative = make_judge(
        settings, [httpx.Response(401, text="nope"), noul(0.95)], [chat_body()]
    )
    with caplog.at_level(logging.INFO, logger="app"):
        judge.judge(TEXT_NEW, TEXT_CANDIDATE)  # 401 → auth → the breaker open
        paused = events(caplog, "slot_paused")
        assert paused and paused[0].slot == SLOT_DECISION
        assert paused[0].job == "judge"
        assert paused[0].outcome == "auth"
        # The pause expired: the knock goes through; the breaker lifts.
        clock["t"] += settings.slot_pause_sec + 1
        judge.judge(TEXT_NEW, TEXT_CANDIDATE)
    probes = events(caplog, "slot_probe")
    resumed = events(caplog, "slot_resumed")
    assert probes and probes[0].slot == SLOT_DECISION
    assert probes[0].result == "ok"
    assert resumed and resumed[0].slot == SLOT_DECISION
