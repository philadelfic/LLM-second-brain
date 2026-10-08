"""Журнал decision-решений (LSBN-F-12, релиз 3.2.0; arch §6.4/§7.2/§13.6).

Вероятности и длительность решения видны в журнале (`decision_answer`), причина
переключения — в `decision_fallback`; в БД вероятности НЕ пишутся (FR-3.3):
ни таблиц решений, ни полей-истории, ни миграций. Журнал — JSON-события
приложения (logging), а не таблица.

Юнит без сети: decision-клиент на httpx.MockTransport, БД — временная (conftest).
"""

from __future__ import annotations

import logging

import pytest

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
