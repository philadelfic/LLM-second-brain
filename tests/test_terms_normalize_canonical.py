"""#32 (3.2.0): choosing a cluster's canonical wording (§6.6, FR-2.3).

    The canon is data-driven and deterministic: the maximum `uses` (the frequency
    of `context_norm` over the area's active rows, as in `_contexts`), then the
    latest `MAX(id)`, then the shorter one, then the lexicographically smaller.
"""

from __future__ import annotations

import pytest

from terms_normalize_common import make_parts, make_settings, seed


@pytest.fixture
def service(test_env, monkeypatch):
    settings = make_settings(test_env, monkeypatch)
    return settings, make_parts(settings)[2]


def test_canonical_by_frequency_then_id(service) -> None:
    """Frequency beats id; at equal frequency — the latest id (stable)."""
    settings, normalizer = service
    # The term "alpha": form A (id 5) vs form B (id 2). A's frequency
    # is raised by another term's record → the canon = id 5 despite the later id 2.
    seed(
        settings,
        [
            {"id": 5, "term": "альфа", "context": "проект", "definition": "д"},
            {"id": 2, "term": "альфа", "context": "проекте", "definition": "д"},
            {"id": 9, "term": "бета", "context": "проект", "definition": "д"},
        ],
    )
    plan = normalizer.plan("альфа")
    assert len(plan["clusters"]) == 1
    assert plan["clusters"][0]["canonical_id"] == 5
    assert plan["clusters"][0]["canonical"] == "проект"

    # The term "gamma": form C (id 10) vs form D (id 20), equal frequencies →
    # the canon = the latest id 20; a repeated run makes the same choice.
    seed(
        settings,
        [
            {"id": 10, "term": "гамма", "context": "отчет", "definition": "д"},
            {"id": 20, "term": "гамма", "context": "отчете", "definition": "д"},
        ],
    )
    first = normalizer.plan("гамма")
    second = normalizer.plan("гамма")
    assert first["clusters"][0]["canonical_id"] == 20
    assert first == second
