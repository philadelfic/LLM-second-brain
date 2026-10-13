"""Реестр decision-вопросов (LSBN-F-12, релиз 3.2.0; arch §6.4/§13.6).

Встроенные группы проходят валидацию (тексты EN, обезличенные); файл
`decision_questions.json` переопределяет группу целиком; нарушения
(неизвестный kind, choice без меток/критериев, дубль имени, вопросов больше
DECISION_MAX_QUESTIONS) — фатальный `ConfigError` при старте.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from app.config import ConfigError
from app.services.prompts import PromptRegistry


def write_questions(tmp_path: pathlib.Path, data: dict) -> pathlib.Path:
    path = tmp_path / "decision_questions.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_builtin_questions_valid() -> None:
    registry = PromptRegistry(None)
    groups = registry.decision_questions
    assert set(groups) == {"judge", "classify"}
    assert all(question.kind in ("choice", "noul", "score") for items in groups.values() for question in items)
    judge = groups["judge"][0]
    assert judge.name == "is_text_duplicate"
    assert judge.kind == "noul"
    assert judge.text.strip()
    # Тексты EN и обезличенные (без имён людей).
    for items in groups.values():
        for question in items:
            assert not any(name in question.text for name in ("Олег", "Oleg", "Phil"))


def test_file_override_wins(tmp_path: pathlib.Path) -> None:
    write_questions(
        tmp_path,
        {"judge": [{"name": "custom_dup", "text": "Is it the same?", "kind": "noul"}]},
    )
    registry = PromptRegistry(tmp_path)
    assert [question.name for question in registry.decision_questions["judge"]] == [
        "custom_dup"
    ]


def test_invalid_kind_fatal(tmp_path: pathlib.Path) -> None:
    write_questions(
        tmp_path,
        {"judge": [{"name": "q", "text": "?", "kind": "magic"}]},
    )
    with pytest.raises(ConfigError, match="unknown kind"):
        PromptRegistry(tmp_path)


def test_choice_without_criteria_fatal(tmp_path: pathlib.Path) -> None:
    write_questions(
        tmp_path,
        {"classify": [{"name": "q", "text": "?", "kind": "choice", "labels": ["generic"]}]},
    )
    with pytest.raises(ConfigError, match="without criteria"):
        PromptRegistry(tmp_path)


def test_duplicate_name_fatal(tmp_path: pathlib.Path) -> None:
    write_questions(
        tmp_path,
        {
            "judge": [
                {"name": "q", "text": "?", "kind": "noul"},
                {"name": "q", "text": "?", "kind": "noul"},
            ]
        },
    )
    with pytest.raises(ConfigError, match="duplicate question name"):
        PromptRegistry(tmp_path)


def test_too_many_questions_fatal(tmp_path: pathlib.Path) -> None:
    write_questions(
        tmp_path,
        {
            "judge": [
                {"name": "q1", "text": "?", "kind": "noul"},
                {"name": "q2", "text": "?", "kind": "noul"},
            ]
        },
    )
    with pytest.raises(ConfigError, match="DECISION_MAX_QUESTIONS"):
        PromptRegistry(tmp_path, decision_max_questions=1)
