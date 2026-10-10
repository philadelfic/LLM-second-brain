"""№32 (3.2.0): разные определения → предупреждение, переименования нет (FR-3.2)."""

from __future__ import annotations

import json

import pytest

from terms_normalize_common import (
    active_rows,
    make_parts,
    make_settings,
    seed,
    warnings_rows,
)


@pytest.fixture
def parts(test_env, monkeypatch):
    settings = make_settings(test_env, monkeypatch)
    return settings, *make_parts(settings)


def test_cluster_flagged_not_renamed(parts) -> None:
    """Разные определения → `term_context_cluster` в `warnings`, terms не тронуты."""
    settings, _terms, _warnings, service = parts
    seed(
        settings,
        [
            {"id": 1, "term": "флаг", "context": "проект", "definition": "первый смысл"},
            {"id": 2, "term": "флаг", "context": "проекте", "definition": "второй смысл"},
        ],
    )
    result = service.normalize("флаг")
    assert result["flagged"] == 1
    assert result["renamed"] == 0 and result["merged"] == 0
    flagged = warnings_rows(settings, "term_context_cluster")
    assert len(flagged) == 1
    payload = json.loads(flagged[0]["payload"])
    assert payload["reason"] == "definitions_differ"
    assert len(payload["contexts"]) == 2
    rows = active_rows(settings, "флаг")
    assert {row["id"] for row in rows} == {1, 2}
    assert all(row["vector_status"] != "pending" for row in rows)
