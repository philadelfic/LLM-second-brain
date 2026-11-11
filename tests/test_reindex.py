"""The reindexing script (Phase 3): a vec0 reset, a new dimension, the queue.

    Unit tests without the network: live vectorization is unavailable in the test
    environment — the script stops at the "everything to pending" step (the
    NFR-3-safe path); catching up with live vectorization — in integration
    (test_integration_live.py, when Ollama is available).
"""

from __future__ import annotations

import pytest
from fakes import HashEmbedder, vectorize_notes

from app.config import get_settings
from app.services.notes import NoteService
from app.storage import vectors
from app.storage.db import init_db, session
from scripts.reindex import main


@pytest.fixture
def dim4(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """A DB under the old dimension 4, with two vectorized notes
            (the worker queue caught the vectors up — the Phase-8 save writes only
            pending).
    """
    monkeypatch.setenv("DB_PATH", str(tmp_path / "notes.db"))
    monkeypatch.setenv("EMBEDDING_DIM", "4")
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    notes = NoteService(settings, HashEmbedder(4))
    notes.save("заметка при старой размерности")
    notes.save("вторая заметка при старой размерности")
    # Phase 8: the worker builds the vectors — we bring the queue to "2 vectorized"
    assert vectorize_notes(settings, HashEmbedder(4)) == 2
    return settings


def test_reindex_rebuilds_table_at_new_dim(dim4, monkeypatch) -> None:
    """A DB of 4-dimensional vectors + EMBEDDING_DIM=8: reindex fixes the startup gate.

                After the script: notes_vec float[8], all notes pending (the vectors
                reset), init_db with the new dim passes without a failure, the notes
                preserved.
    """
    assert NoteService(dim4).get([1, 2])["notes"]  # the data before reindexing

    monkeypatch.setenv("EMBEDDING_DIM", "8")
    get_settings.cache_clear()
    settings = get_settings()  # now 8
    assert main(["--yes"]) == 0

    with session(settings) as conn:
        ddl = conn.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'notes_vec'"
        ).fetchone()[0]
        assert "float[8]" in ddl  # the table recreated under the new dimension
        statuses = conn.execute("SELECT vector_status FROM notes").fetchall()
        assert all(row[0] == "pending" for row in statuses)
        assert vectors.count(conn) == 0  # the old vectors deleted
    init_db(settings)  # the startup gate passes now
    assert NoteService(settings).get([1, 2])["notes"]  # no data lost


def test_reindex_same_dim_rebuilds_index(dim4) -> None:
    """A repeated call at an unchanged dimension — safe too."""
    assert main(["--yes"]) == 0
    init_db(dim4)  # consistent, no refusals


def test_reindex_missing_db(tmp_path, monkeypatch) -> None:
    """No DB file — a clear failure (code 2), not a traceback."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "notexist.db"))
    get_settings.cache_clear()
    assert main(["--yes"]) == 2


def test_reindex_confirmation_aborts(dim4) -> None:
    """Without --yes and with a declining prompt answer — the DB fully untouched."""
    import builtins

    builtins_input = builtins.input
    builtins.input = lambda _prompt: "n"
    try:
        assert main([]) == 1  # cancelled by the operator
    finally:
        builtins.input = builtins_input
    with session(dim4) as conn:
        # the index intact: the table in place, the vectors not reset
        assert vectors.count(conn) == 2
        assert NoteService(dim4).get([1])["notes"]


def test_reindex_offline_leaves_pending_queue(dim4, monkeypatch) -> None:
    """Live Ollama unavailable: the notes stay pending, code 0 (NFR-3)."""
    monkeypatch.setenv("EMBEDDING_DIM", "8")
    get_settings.cache_clear()
    settings = get_settings()
    assert main(["--yes"]) == 0
    with session(settings) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM notes WHERE vector_status = 'pending'"
        ).fetchone()[0] == 2