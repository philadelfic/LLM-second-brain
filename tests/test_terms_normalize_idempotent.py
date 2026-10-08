"""№32 (3.2.0): идемпотентность джобы — состояние в самих данных (§13.11)."""

from __future__ import annotations

import pytest

from terms_normalize_common import (
    active_rows,
    make_parts,
    make_settings,
    seed,
    versions,
    warnings_rows,
)
from app.storage.db import session


@pytest.fixture
def parts(test_env, monkeypatch):
    settings = make_settings(test_env, monkeypatch)
    return settings, *make_parts(settings)


def _warnings_total(settings) -> int:
    with session(settings) as conn:
        return int(conn.execute("SELECT COUNT(*) AS n FROM warnings").fetchone()["n"])


def test_second_run_noop(parts) -> None:
    """Второй прогон не меняет terms, не плодит warnings и не пишет событий."""
    settings, _terms, _warnings, service = parts
    seed(
        settings,
        [
            {"id": 1, "term": "идемп", "context": "проект", "definition": "д"},
            {"id": 2, "term": "идемп", "context": "проекте", "definition": "д"},
            {"id": 3, "term": "идемп", "context": "проекта", "definition": "д"},
        ],
    )
    assert service.run() == 1
    assert len(active_rows(settings, "идемп")) == 1
    terms_after = active_rows(settings, "идемп")
    warnings_after = _warnings_total(settings)
    versions_after = sum(len(versions(settings, term_id)) for term_id in (1, 2, 3))

    assert service.run() == 0  # после нормализации группа не берётся отбором
    assert active_rows(settings, "идемп") == terms_after
    assert _warnings_total(settings) == warnings_after
    assert (
        sum(len(versions(settings, term_id)) for term_id in (1, 2, 3))
        == versions_after
    )
