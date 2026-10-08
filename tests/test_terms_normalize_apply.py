"""№32 (3.2.0): применение плана — авто-переименование, слияние, история (§6.6).

Авто — только при ТОЧНОМ равенстве нормализованных определений (FR-3.1/FR-3.3);
конфликт ключа разрешается без потери данных (выживает меньший id, проигравший
в мягкое удаление, FR-4.1/FR-4.2); история (№41) фиксирует переименование
(`source='job:terms-normalize'`) и удаление.
"""

from __future__ import annotations

import pytest

from app.services.areas import normalize_key
from terms_normalize_common import (
    TERMS_NORMALIZE_JOB,
    active_rows,
    make_parts,
    make_settings,
    row_by_id,
    seed,
    versions,
)

ORIGIN = f"job:{TERMS_NORMALIZE_JOB}"


@pytest.fixture
def parts(test_env, monkeypatch):
    settings = make_settings(test_env, monkeypatch)
    return settings, *make_parts(settings)


def test_auto_only_when_definitions_equal(parts) -> None:
    """Одинаковые определения → авто; различающиеся → предупреждение без правок."""
    settings, _terms, _warnings, service = parts
    seed(
        settings,
        [
            {"id": 1, "term": "равные", "context": "проект", "definition": "одно и то же"},
            {"id": 2, "term": "равные", "context": "проекте", "definition": "одно и то же"},
        ],
    )
    result = service.normalize("равные")
    assert result["renamed"] + result["merged"] == 1
    rows = active_rows(settings, "равные")
    assert len(rows) == 1
    assert rows[0]["context_norm"] == normalize_key("проекте")
    assert rows[0]["definition"] == "одно и то же"

    # Разные определения: авто запрещено, данные не тронуты.
    seed(
        settings,
        [
            {"id": 3, "term": "разные", "context": "отчет", "definition": "первый смысл"},
            {"id": 4, "term": "разные", "context": "отчете", "definition": "второй смысл"},
        ],
    )
    service.normalize("разные")
    untouched = active_rows(settings, "разные")
    assert {row["id"] for row in untouched} == {3, 4}
    assert {row["context_norm"] for row in untouched} == {
        normalize_key("отчет"),
        normalize_key("отчете"),
    }


def test_rename_recomputes_key_and_vector(parts) -> None:
    """После переименования ключ каноничен, вектор — pending (догонит areas)."""
    settings, _terms, _warnings, service = parts
    seed(
        settings,
        [
            {"id": 1, "term": "вектор", "context": "проект", "definition": "д",
             "vector_status": "ok"},
            {"id": 2, "term": "вектор", "context": "проекте", "definition": "д"},
        ],
    )
    service.normalize("вектор")
    rows = active_rows(settings, "вектор")
    assert len(rows) == 1
    assert rows[0]["context_norm"] == normalize_key("проекте")
    assert rows[0]["vector_status"] == "pending"


def test_key_conflict_soft_deletes_loser(parts) -> None:
    """Занятый каноничный ключ: выживает меньший id, проигравший — в trash."""
    settings, _terms, _warnings, service = parts
    seed(
        settings,
        [
            {"id": 1, "term": "конфликт", "context": "проект", "definition": "д"},
            {"id": 2, "term": "конфликт", "context": "проекте", "definition": "д"},
            {"id": 3, "term": "конфликт", "context": "проекта", "definition": "д"},
        ],
    )
    service.normalize("конфликт")
    survivors = active_rows(settings, "конфликт")
    assert [row["id"] for row in survivors] == [1]
    assert survivors[0]["context_norm"] == normalize_key("проекта")
    for loser_id in (2, 3):
        loser = row_by_id(settings, loser_id)
        assert loser["deleted_at"] is not None
        assert loser["definition"] == "д"  # данные живы в trash


def test_history_records_rename_and_delete(parts) -> None:
    """История фиксирует переименование (джоба) и удаление (событие `delete`)."""
    settings, _terms, _warnings, service = parts
    seed(
        settings,
        [
            {"id": 1, "term": "история", "context": "проект", "definition": "д"},
            {"id": 2, "term": "история", "context": "проекте", "definition": "д"},
            {"id": 3, "term": "история", "context": "проекта", "definition": "д"},
        ],
    )
    service.normalize("история")
    survivor_versions = versions(settings, 1)
    assert any(row["source"] == ORIGIN for row in survivor_versions)
    deleted_versions = versions(settings, 3)
    assert any(row["event"] == "delete" for row in deleted_versions)


def test_hint_shrinks(parts) -> None:
    """После нормализации hint `terms_save` показывает короткий список контекстов."""
    settings, terms, _warnings, service = parts
    originals = {
        normalize_key(context)
        for context in (
            "проект разработки",
            "проекте разработки",
            "проекта разработки",
            "проектов разработки",
        )
    }
    seed(
        settings,
        [
            {"id": index, "term": "подсказка", "context": context, "definition": "д"}
            for index, context in enumerate(sorted(originals), start=1)
        ],
    )
    before = terms.save("якорь-один", "якорь-контекст-один", "def")["contexts"]
    assert len([c for c in before if normalize_key(c) in originals]) == 4
    service.normalize("подсказка")
    after = terms.save("якорь-два", "якорь-контекст-два", "def")["contexts"]
    assert len([c for c in after if normalize_key(c) in originals]) == 1
