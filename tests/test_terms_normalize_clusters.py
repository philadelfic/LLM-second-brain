"""№32 (3.2.0): кластеризация контекстов (single-linkage, §6.6)."""

from __future__ import annotations

import pytest

from terms_normalize_common import make_parts, make_settings, seed

TERM = "проект"


@pytest.fixture
def parts(test_env, monkeypatch):
    settings = make_settings(test_env, monkeypatch)
    return settings, *make_parts(settings)


def test_single_linkage_cluster(parts) -> None:
    """Цепочка близких контекстов собирается в один кластер."""
    settings, _terms, _warnings, service = parts
    seed(
        settings,
        [
            {"id": 1, "term": TERM, "context": "проект разработки", "definition": "д"},
            {"id": 2, "term": TERM, "context": "проекте разработки", "definition": "д"},
            {"id": 3, "term": TERM, "context": "проекты разработки", "definition": "д"},
        ],
    )
    plan = service.plan("проект")
    assert len(plan["clusters"]) == 1
    assert {member["id"] for member in plan["clusters"][0]["members"]} == {1, 2, 3}
    assert plan["clusters"][0]["verdict"] == "auto"


def test_distinct_contexts_not_clustered(parts) -> None:
    """Далёкие контексты остаются разными (кластеры размера 1)."""
    settings, _terms, _warnings, service = parts
    seed(
        settings,
        [
            {"id": 1, "term": TERM, "context": "бухгалтерия", "definition": "д"},
            {"id": 2, "term": TERM, "context": "студенты мгу", "definition": "д"},
            {"id": 3, "term": TERM, "context": "железо и звук", "definition": "д"},
        ],
    )
    plan = service.plan("проект")
    assert len(plan["clusters"]) == 3
    assert all(len(cluster["members"]) == 1 for cluster in plan["clusters"])
