"""#32 (3.2.0): the context counter and the over-abundance flag (FR-5.1/FR-5.2)."""

from __future__ import annotations

import json

import pytest

from app.services.areas import normalize_key
from terms_normalize_common import (
    make_parts,
    make_settings,
    seed,
    warnings_rows,
)

CONTEXTS = ("бухгалтерия", "студенты", "погода", "музыка", "железо")


@pytest.fixture
def parts(test_env, monkeypatch):
    settings = make_settings(test_env, monkeypatch)
    return settings, *make_parts(settings)


def test_excess_threshold_and_resolve(parts) -> None:
    """≥ the threshold of distinct contexts → a warning with the counter; below — cleared."""
    settings, terms, _warnings, service = parts
    seed(
        settings,
        [
            {"id": index, "term": "много", "context": context, "definition": "д"}
            for index, context in enumerate(CONTEXTS, start=1)
        ],
    )
    # A warning is not a data edit: for the job's loop it is NOT progress
    # (otherwise a group with only a warning would spin the loop idle,
    # lsbdef-0011). The parsed groups are visible in the event as `terms`.
    assert service.run() == 0
    opened = warnings_rows(settings, "term_contexts_excess")
    assert len(opened) == 1
    assert json.loads(opened[0]["payload"])["contexts"] == 5
    assert opened[0]["resolved_at"] is None

    # We drop one (not the minimal) context → the count falls below the threshold.
    assert terms.delete(5)["deleted"] is True
    service.run()
    resolved = warnings_rows(settings, "term_contexts_excess")
    assert len(resolved) == 1  # rows are not multiplied (upsert/clearing)
    assert resolved[0]["resolved_at"] is not None
    assert normalize_key(CONTEXTS[4]) == "железо"  # the context is really gone
