"""The "Assistant logs" area's listing tests (LSBN-F-10, f10-03).

    Arch assistant-logs §6.1/§7.1/§7.2/§8/§13.16: `LogsService.list` serves a page
    of active records per the lsb-0013 contract (`total`/`has_more`/`next_offset`/
    `next_cursor`); the order — `created_at DESC, id DESC`; the slices — the same
    as the search's (one SQL predicate). The `limit` ceiling is parameterized: MCP
    passes `list_max_limit_mcp`, REST defaults to `list_max_limit_rest` — the
    failure texts differ (FR-5.4/NFR-2). A relative date is rejected by the same
    `HINT_DATE_ABSOLUTE`; `offset < 0` — a soft failure (the `SkillsService.list`
    precedent).
"""

from __future__ import annotations

import pytest

from app.config import Settings, get_settings
from app.services.logs import (
    HINT_DATE_ABSOLUTE,
    LogsService,
    LogValidationError,
)
from app.storage.db import init_db, session

DIM = 8

REASON = "общий повод для замера скорости"
ACTION = "перенёс слот эмбеддингов на mainsrv"


@pytest.fixture
def settings(test_env: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> Settings:
    """The schema in a test DB (the dimension does not matter: the listing calls no model)."""
    monkeypatch.setenv("EMBEDDING_DIM", str(DIM))
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    return settings


@pytest.fixture
def logs(settings: Settings) -> LogsService:
    """The area service without an embedder: the listing never calls it."""
    return LogsService(settings)


def _save(logs: LogsService, action: str = ACTION, **extra: object) -> int:
    """Save a log and return its id."""
    result = logs.save(action, REASON, **extra)
    assert result["created"] is True
    return int(result["id"])


def _set_created(settings: Settings, log_id: int, moment: str) -> None:
    """Set the record's `created_at` manually (checking the order and the slices)."""
    with session(settings) as conn:
        conn.execute(
            "UPDATE assistant_logs SET created_at = ? WHERE id = ?",
            (moment, log_id),
        )


def _ids(page: dict) -> list[int]:
    return [item["id"] for item in page["items"]]


class TestListing:
    def test_list_pagination(self, logs: LogsService) -> None:
        """`total`/`has_more`/`next_offset` per the page_fields contract; the cursor None."""
        for index in range(3):
            _save(logs, action=f"{ACTION} #{index}")
        first = logs.list(limit=2)
        assert set(first) == {
            "items",
            "total",
            "has_more",
            "next_offset",
            "next_cursor",
        }
        assert first["total"] == 3 and first["has_more"] is True
        assert first["next_offset"] == 2 and first["next_cursor"] is None
        assert len(first["items"]) == 2
        tail = logs.list(limit=2, offset=first["next_offset"])
        assert len(tail["items"]) == 1 and tail["has_more"] is False
        assert tail["next_offset"] is None and tail["total"] == 3

    def test_list_limit_ceiling(self, logs: LogsService) -> None:
        """The MCP and REST ceilings give a DIFFERENT `limit: expected 1..…` failure text."""
        with pytest.raises(LogValidationError) as exc:
            logs.list(limit=50, max_limit=20)
        assert str(exc.value) == "limit: expected 1..20, got 50"
        with pytest.raises(LogValidationError) as exc:
            logs.list(limit=51)
        assert str(exc.value) == "limit: expected 1..50, got 51"
        with pytest.raises(LogValidationError) as exc:
            logs.list(limit=0, max_limit=20)
        assert str(exc.value) == "limit: expected 1..20, got 0"
        with pytest.raises(LogValidationError) as exc:
            logs.list(offset=-1)
        assert str(exc.value) == "offset: expected 0 or greater, got -1"

    def test_list_order(self, settings: Settings, logs: LogsService) -> None:
        """The order `created_at DESC, id DESC`; deleted ones are not visible."""
        old = _save(logs, action="старое")
        newest = _save(logs, action="новое")
        middle = _save(logs, action="среднее")
        _set_created(settings, old, "2026-01-01T10:00:00Z")
        _set_created(settings, middle, "2026-01-02T10:00:00Z")
        _set_created(settings, newest, "2026-01-03T10:00:00Z")
        assert _ids(logs.list()) == [newest, middle, old]
        logs.delete(middle)
        page = logs.list()
        assert _ids(page) == [newest, old] and page["total"] == 2
        # The same moment — the id DESC tie-break: the higher id first.
        _set_created(settings, old, "2026-01-03T10:00:00Z")
        assert _ids(logs.list()) == [newest, old]

    def test_list_slices(self, logs: LogsService) -> None:
        """The slices are the same as the search's (one SQL predicate, §8)."""
        first = _save(logs, entity="skill", entity_id=5, entity_version=1)
        other = _save(logs, entity="skill", entity_id=6, session="agent-2")
        assert _ids(logs.list(entity="skill", entity_id=5)) == [first]
        assert _ids(logs.list(entity="skill", entity_id=5, entity_version=1)) == [
            first
        ]
        assert _ids(logs.list(entity="skill", entity_id=5, entity_version=2)) == []
        assert _ids(logs.list(session="agent-2")) == [other]
        assert _ids(logs.list(source="rest")) == []
        assert _ids(logs.list(date_from="2099-01-01")) == []
        assert _ids(logs.list(date_to="2000-01-01")) == []

    def test_list_relative_date_refused(self, logs: LogsService) -> None:
        """A relative date in the listing — the same HINT_DATE_ABSOLUTE (§13.12)."""
        with pytest.raises(LogValidationError) as exc:
            logs.list(date_to="вчера")
        assert str(exc.value) == HINT_DATE_ABSOLUTE

    def test_list_empty_is_valid(self, logs: LogsService) -> None:
        """An empty listing — a valid outcome without a hint (not the same as search)."""
        page = logs.list()
        assert page["items"] == [] and page["total"] == 0
        assert page["has_more"] is False and page["next_offset"] is None
        assert "hint" not in page
