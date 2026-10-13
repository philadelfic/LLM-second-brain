"""№32 (3.2.0): счётчик контекстов и флаг переизбытка (FR-5.1/FR-5.2)."""

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
    """≥ порога разных контекстов → предупреждение со счётчиком; ниже — снято."""
    settings, terms, _warnings, service = parts
    seed(
        settings,
        [
            {"id": index, "term": "много", "context": context, "definition": "д"}
            for index, context in enumerate(CONTEXTS, start=1)
        ],
    )
    # Предупреждение — не правка данных: для петли джобы это НЕ прогресс
    # (иначе группа с одним лишь предупреждением крутила бы петлю вхолостую,
    # lsbdef-0011). Разобранные группы видны в событии как `terms`.
    assert service.run() == 0
    opened = warnings_rows(settings, "term_contexts_excess")
    assert len(opened) == 1
    assert json.loads(opened[0]["payload"])["contexts"] == 5
    assert opened[0]["resolved_at"] is None

    # Убираем один (не минимальный) контекст → число падает ниже порога.
    assert terms.delete(5)["deleted"] is True
    service.run()
    resolved = warnings_rows(settings, "term_contexts_excess")
    assert len(resolved) == 1  # строки не плодятся (upsert/снятие)
    assert resolved[0]["resolved_at"] is not None
    assert normalize_key(CONTEXTS[4]) == "железо"  # контекст действительно ушёл
