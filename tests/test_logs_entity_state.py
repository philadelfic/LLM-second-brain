"""The log reference object's state tests (LSBN-F-10, f10-04).

    Arch assistant-logs §11/§13.4: reading a log answers "what happened to the
    object" — the state is computed by the HISTORY LAYER (`history.resolve_entity`);
    the area service does no raw SQL over foreign tables (area isolation).
    `LogsService` adds `entity_state` to `get` (only when `entity` is set) and to
    every `list`/`search` item: `active` / `deleted` / `unavailable` (the object
    is missing or its area is disabled — we do not reveal the area exists) /
    `external` (a non-versioned object). `HISTORY_ENABLED=false` — the state is
    not completed; the output stays alive.
"""

from __future__ import annotations

import pytest

from app.config import Settings, get_settings
from app.services import history
from app.services.logs import LogsService
from app.services.notes import NoteService
from app.storage.db import init_db
from fakes import HashEmbedder

DIM = 8

ACTION = "перенёс слот эмбеддингов на mainsrv"
REASON = "замер показал вдвое быстрее"


@pytest.fixture
def settings(test_env: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setenv("EMBEDDING_DIM", str(DIM))
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    return settings


@pytest.fixture
def embedding() -> HashEmbedder:
    """A deterministic embedder: the area's search works without the network."""
    return HashEmbedder(DIM)


@pytest.fixture
def logs(settings: Settings, embedding: HashEmbedder) -> LogsService:
    return LogsService(settings, embedding=embedding)


def _save(logs: LogsService, **extra: object) -> int:
    result = logs.save(ACTION, REASON, **extra)
    assert result["created"] is True, result
    return int(result["id"])


def _note(settings: Settings, text: str = "Заметка про слот") -> int:
    """A live note by the real service: it has version 1 (#41)."""
    result = NoteService(settings, HashEmbedder(DIM)).save(text, title="Слот")
    return int(result["id"])


class TestResolveEntity:
    def test_state_active(self, settings: Settings, logs: LogsService) -> None:
        """An active note → `active`; `current_version` — the head version."""
        note_id = _note(settings)
        log_id = _save(logs, entity="note", entity_id=note_id)
        assert history.resolve_entity("note", note_id, settings) == {
            "state": "active",
            "current_version": 1,
        }
        assert logs.get(log_id)["entity_state"] == "active"

    def test_state_deleted(self, settings: Settings, logs: LogsService) -> None:
        """A soft-deleted object → `deleted`; the log record alive (FR-4.3)."""
        note_id = _note(settings)
        NoteService(settings, HashEmbedder(DIM)).delete(note_id)
        log_id = _save(logs, entity="note", entity_id=note_id)
        record = logs.get(log_id)
        assert record["entity_state"] == "deleted"
        assert record["action"] == ACTION

    def test_state_unavailable_missing(
        self, settings: Settings, logs: LogsService
    ) -> None:
        """No object → `unavailable`."""
        log_id = _save(logs, entity="note", entity_id=999_999)
        assert logs.get(log_id)["entity_state"] == "unavailable"

    def test_state_unavailable_area_disabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The object's area disabled → `unavailable`: the area is not revealed."""
        monkeypatch.setenv("EMBEDDING_DIM", str(DIM))
        monkeypatch.setenv("AREA_TERMS_ENABLED", "false")
        get_settings.cache_clear()
        settings = get_settings()
        init_db(settings)
        logs = LogsService(settings, embedding=HashEmbedder(DIM))
        log_id = _save(logs, entity="term", entity_id=1)
        resolved = history.resolve_entity("term", 1, settings)
        assert resolved == {"state": "unavailable", "current_version": None}
        assert logs.get(log_id)["entity_state"] == "unavailable"

    def test_state_external(self, settings: Settings, logs: LogsService) -> None:
        """A non-versioned object (`commit`) → `external` (a text-only reference)."""
        log_id = _save(logs, entity="commit", entity_ref="deadbeef")
        assert history.resolve_entity("commit", 0, settings) is None
        assert logs.get(log_id)["entity_state"] == "external"

    def test_state_without_entity(self, settings: Settings, logs: LogsService) -> None:
        """`entity` not set → no `entity_state` field in the record."""
        log_id = _save(logs)
        record = logs.get(log_id)
        assert record["action"] == ACTION
        assert "entity_state" not in record

    def test_resolve_history_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """`HISTORY_ENABLED=false` → the state is not completed; the output alive."""
        monkeypatch.setenv("EMBEDDING_DIM", str(DIM))
        monkeypatch.setenv("HISTORY_ENABLED", "false")
        get_settings.cache_clear()
        settings = get_settings()
        init_db(settings)
        logs = LogsService(settings, embedding=HashEmbedder(DIM))
        log_id = _save(logs, entity="note", entity_id=1)
        assert history.resolve_entity("note", 1, settings) is None
        record = logs.get(log_id)
        assert record["action"] == ACTION and "hint" not in record
        assert "entity_state" not in record


class TestListAndSearchStates:
    def test_items_annotated(
        self, settings: Settings, embedding: HashEmbedder, logs: LogsService
    ) -> None:
        """Every list/search item carries its own object's state (§11)."""
        note_id = _note(settings)
        active = _save(logs, entity="note", entity_id=note_id)
        external = _save(logs, entity="commit", entity_ref="deadbeef")
        bare = _save(logs)
        expected = {active: "active", external: "external", bare: "external"}

        page = logs.list()
        assert {item["id"]: item["entity_state"] for item in page["items"]} == expected

        found = logs.search("эмбеддингов mainsrv")
        assert found["results"], found
        assert {
            hit["id"]: hit["entity_state"] for hit in found["results"]
        } == expected
