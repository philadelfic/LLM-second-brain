"""№32 (3.2.0): выбор каноничной формулировки кластера (§6.6, FR-2.3).

Канон — по данным, детерминированно: максимум `uses` (частота `context_norm`
по активным строкам области, как в `_contexts`), затем позднейший `MAX(id)`,
затем короче, затем лексикографически меньше.
"""

from __future__ import annotations

import pytest

from terms_normalize_common import make_parts, make_settings, seed


@pytest.fixture
def service(test_env, monkeypatch):
    settings = make_settings(test_env, monkeypatch)
    return settings, make_parts(settings)[2]


def test_canonical_by_frequency_then_id(service) -> None:
    """Частота важнее id; при равенстве частоты — позднейший id (устойчиво)."""
    settings, normalizer = service
    # Термин «альфа»: «проект» (id 5) vs «проекте» (id 2). Частота у «проект»
    # поднята записью другого термина → канон = id 5, несмотря на поздний id 2.
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

    # Термин «гамма»: «отчет» (id 10) vs «отчете» (id 20), частоты равны →
    # канон = позднейший id 20; повторный прогон даёт тот же выбор.
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
