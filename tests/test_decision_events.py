"""Журнал decision-решений (LSBN-F-12, релиз 3.2.0; arch §6.4/§7.2/§13.6).

Вероятности и длительность решения видны в журнале (`decision_answer`), причина
переключения — в `decision_fallback`; в БД вероятности НЕ пишутся (FR-3.3):
ни таблиц решений, ни полей-истории, ни миграций. Журнал — JSON-события
приложения (logging), а не таблица.

Юнит без сети: decision-клиент на httpx.MockTransport, БД — временная (conftest).
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
)


def events(caplog, name: str) -> list:
    return [record for record in caplog.records if getattr(record, "event", None) == name]


def schema() -> dict[str, set[str]]:
    """Таблицы БД и их колонки — снимок схемы до/после решений."""
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
    """`decision_answer` несёт вероятность, порог, модель и `duration_ms`."""
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
    assert judge_event.threshold == pytest.approx(0.5)
    assert judge_event.model == "decision-test"
    assert isinstance(judge_event.duration_ms, int)
    assert judge_event.duration_ms >= 0
    assert classify_event.operation == "classify"
    assert classify_event.questions == [CLASSIFY_QUESTION]
    assert classify_event.answers[CLASSIFY_QUESTION]["label"] == "work/sbos2020"
    assert classify_event.answers[CLASSIFY_QUESTION]["probability"] == pytest.approx(0.9)


def test_no_probabilities_in_db(monkeypatch, caplog) -> None:
    """После решений схема БД не менялась: таблиц/полей решений нет."""
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


# --- контракт событий (f12-03: наблюдаемость и тесты) -------------------------


def test_decision_check_fields(monkeypatch, caplog) -> None:
    """Событие стартовой проверки `decision_check` несёт `model` и `result`."""
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
    """`decision_answer`: operation/threshold/duration_ms/answers; вероятности — там."""
    settings = decision_settings(monkeypatch)
    with caplog.at_level(logging.INFO, logger="app"):
        run_judge_decision(settings, caplog)
    (record,) = events(caplog, "decision_answer")
    assert record.operation == "judge"
    assert record.threshold == pytest.approx(0.5)
    assert isinstance(record.duration_ms, int)
    assert record.answers[JUDGE_QUESTION]["probability"] == pytest.approx(0.95)


def test_decision_fallback_reason_enum(monkeypatch, caplog) -> None:
    """`reason` переключения — ровно три значения; иных нет (§13.6/FR-4.2)."""
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
    reasons = {record.reason for record in events(caplog, "decision_fallback")}
    assert reasons and reasons <= allowed
    assert set(decider.snapshot()["by_reason"]) == allowed


def test_decision_paused_events(monkeypatch, caplog) -> None:
    """Разрыв decision пишет `slot_paused`/`slot_probe`/`slot_resumed` (блок 2)."""
    settings = decision_settings(monkeypatch)
    clock = {"t": 1000.0}
    monkeypatch.setattr("app.services.slot_breaker.time.time", lambda: clock["t"])
    judge, decider, decision, generative = make_judge(
        settings, [httpx.Response(401, text="nope"), noul(0.95)], [chat_body()]
    )
    with caplog.at_level(logging.INFO, logger="app"):
        judge.judge(TEXT_NEW, TEXT_CANDIDATE)  # 401 → auth → разрыв открыт
        paused = events(caplog, "slot_paused")
        assert paused and paused[0].slot == SLOT_DECISION
        assert paused[0].job == "judge"
        assert paused[0].outcome == "auth"
        # Пауза истекла: стук проходит, разрыв снимается.
        clock["t"] += settings.slot_pause_sec + 1
        judge.judge(TEXT_NEW, TEXT_CANDIDATE)
    probes = events(caplog, "slot_probe")
    resumed = events(caplog, "slot_resumed")
    assert probes and probes[0].slot == SLOT_DECISION
    assert probes[0].result == "ok"
    assert resumed and resumed[0].slot == SLOT_DECISION
