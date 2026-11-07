"""The separate comment search (no24-02, release 3.2.0): the FTS `notes_comment_fts`.

    Comments are searched ONLY by their own handle (`SearchService.comment_search`
    / MCP `memory_comment_search` / REST `GET /notes/comments`) — the notes'
    semantic search (`notes_fts` over title/text) does not see comments
    (arch §13.6). The comment index is synced by the `notes_comment_fts_au`
    trigger at set/change/clear of `comment`; the migration is idempotent; the
    filters (`namespace`, `updated_after`) and the `top_k` limit work.
"""

from __future__ import annotations

import pytest
from fakes import HashEmbedder

from app.config import get_settings
from app.services.namespaces import NamespaceError, NamespaceService
from app.services.notes import NoteService
from app.services.search import HINT_NO_COMMENTS, SearchService
from app.storage.db import init_db, session

DIM = 8


def _searcher(settings) -> SearchService:
    """A SearchService with a deterministic HashEmbedder (no network)."""
    return SearchService(settings, HashEmbedder(settings.embedding_dim))


@pytest.fixture
def settings():
    """A fresh DB with the comment migration + the services (no network)."""
    init_db(get_settings())
    return get_settings()


def test_fts_migration_idempotent(settings) -> None:
    """After init_db the notes_comment_fts table and both triggers exist; a repeat — a no-op."""
    with session(settings) as conn:
        tables = {
            row["name"]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        triggers = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'trigger'"
            )
        }
    assert "notes_comment_fts" in tables
    assert {"notes_comment_fts_ai", "notes_comment_fts_au"} <= triggers
    assert "notes_fts" in tables  # the semantic index untouched

    init_db(settings)  # a repeated start — idempotently
    with session(settings) as conn:
        again = {
            row["name"]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        triggers_again = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'trigger'"
            )
        }
    assert "notes_comment_fts" in again
    assert {"notes_comment_fts_ai", "notes_comment_fts_au"} <= triggers_again


def test_comment_indexed_on_update(settings) -> None:
    """notes.update(comment=...) → the trigger indexes the comment; the search finds it."""
    notes = NoteService(settings)
    saved = notes.save("Заметка про деплой контейнера", title="Деплой")
    notes.update(saved["id"], comment="пригодится к задаче X")

    found = _searcher(settings).comment_search("пригодится")
    assert [r["id"] for r in found["results"]] == [saved["id"]]
    assert found["results"][0]["comment"] == "пригодится к задаче X"


def test_comment_cleared_removed_from_index(settings) -> None:
    """Clearing the comment (comment=None) → the note vanishes from the comment search."""
    notes = NoteService(settings)
    searcher = _searcher(settings)
    saved = notes.save("Заметка для сброса пометки", title="Сброс")
    notes.update(saved["id"], comment="уникальнаяпометка раз")
    assert searcher.comment_search("уникальнаяпометка")["results"]

    notes.update(saved["id"], comment=None)  # clearing the comment
    cleared = searcher.comment_search("уникальнаяпометка")
    assert cleared["results"] == []
    assert cleared["hint"] == HINT_NO_COMMENTS


def test_comment_search_only_by_comment(settings) -> None:
    """A word from the text but not from the comment — the separate search does not find it."""
    notes = NoteService(settings)
    saved = notes.save("секретслово встречается только в тексте", title="Текст")
    assert saved["id"]  # a note without a comment

    found = _searcher(settings).comment_search("секретслово")
    assert found["results"] == []
    assert found["hint"] == HINT_NO_COMMENTS


def test_comment_search_filters(settings) -> None:
    """namespace/namespace_exact, updated_after, top_k and an invalid node."""
    notes = NoteService(settings)
    namespaces = NamespaceService(settings)
    namespaces.create("work", "Рабочие заметки. Подпроекты — в листьях.")
    namespaces.create("work/sbos2020", "СУБО 2020: сервисы HR.")
    a = notes.save("первая", title="Первая", namespace="work/sbos2020")
    b = notes.save("вторая", title="Вторая", namespace="default")
    notes.update(a["id"], comment="общаяпометка раз")
    notes.update(b["id"], comment="общаяпометка два")
    searcher = _searcher(settings)

    # globally — both notes
    everywhere = searcher.comment_search("общаяпометка", top_k=20)
    assert {r["id"] for r in everywhere["results"]} == {a["id"], b["id"]}

    # namespace=work — the subtree (the node + the leaf): only a
    work = searcher.comment_search("общаяпометка", namespace="work")
    assert [r["id"] for r in work["results"]] == [a["id"]]

    # namespace_exact at the leaf — the node itself only
    exact = searcher.comment_search(
        "общаяпометка", namespace="work/sbos2020", namespace_exact=True
    )
    assert [r["id"] for r in exact["results"]] == [a["id"]]

    # top_k respected
    assert len(searcher.comment_search("общаяпометка", top_k=1)["results"]) == 1

    # updated_after in the future — empty + a hint
    empty = searcher.comment_search("общаяпометка", updated_after="2999-01-01")
    assert empty["results"] == []
    assert empty["hint"] == HINT_NO_COMMENTS

    # an unregistered node — NamespaceError (the transport will give a hint)
    with pytest.raises(NamespaceError):
        searcher.comment_search("общаяпометка", namespace="nope")
