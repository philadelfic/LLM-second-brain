"""The foreign-node dedup hint at save (Phase 10, Step 5, US-8).

    A close (verbatim) duplicate in another node is legitimate: the write is not
    blocked, but the answer carries the "similar exists in <ns>" hint (the §5.7
    orientation signal). A duplicate in one's own node still blocks the write
    (duplicated).
"""

from __future__ import annotations

import pytest

from app.config import get_settings
from app.services.namespaces import NamespaceService
from app.services.notes import NoteService
from app.storage.db import init_db, session


@pytest.fixture
def settings(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "notes.db"))
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    return settings


TEXT = "СУБО 2020: реестр зарплат, чек-лист деплоя и контакты поддержки."


def _id_of(settings, namespace: str, text: str) -> int:
    with session(settings) as conn:
        return int(
            conn.execute(
                "SELECT id FROM notes WHERE namespace = ? AND text = ?",
                (namespace, text),
            ).fetchone()[0]
        )


class TestForeignDuplicateHint:
    def test_exact_duplicate_in_other_node_gives_hint(self, settings) -> None:
        """A verbatim duplicate in a foreign node: the note is created; the hint names the node."""
        NamespaceService(settings).create("work", "Рабочие заметки.")
        notes = NoteService(settings)
        first = notes.save(TEXT, namespace="work")
        assert first["stored"] is True
        second = notes.save(TEXT)  # in default
        assert second["stored"] is True
        assert second["id"] != first["id"]
        assert "work" in second["hint"]
        assert "blocked" in second["hint"]

    def test_normalized_duplicate_gives_hint(self, settings) -> None:
        """A "near-verbatim" duplicate (case/spaces) — a hint too."""
        NamespaceService(settings).create("work", "Рабочие заметки.")
        notes = NoteService(settings)
        notes.save(TEXT, namespace="work")
        tweaked = " ".join(("  " + TEXT.upper() + " ").split())
        second = notes.save(tweaked)
        assert second["stored"] is True
        assert "work" in second["hint"]

    def test_same_node_duplicate_still_blocks(self, settings) -> None:
        """A verbatim duplicate in one's own node — the old contract (duplicated)."""
        NamespaceService(settings).create("work", "Рабочие заметки.")
        notes = NoteService(settings)
        notes.save(TEXT, namespace="work")
        second = notes.save(TEXT, namespace="work")
        assert second["duplicated"] is True
        assert second["id"] == _id_of(settings, "work", TEXT)

    def test_no_hint_without_duplicates(self, settings) -> None:
        """A regular write — without a hint (the answer contract does not grow)."""
        notes = NoteService(settings)
        result = notes.save("уникальный текст заметки")
        assert "hint" not in result
        assert result["stored"] is True

    def test_no_hint_for_distinct_texts_in_other_node(self, settings) -> None:
        """A similar-by-words but different note in a foreign node — without a hint."""
        NamespaceService(settings).create("work", "Рабочие заметки.")
        notes = NoteService(settings)
        notes.save(TEXT, namespace="work")
        result = notes.save("Совсем другой текст про другой проект")
        assert "hint" not in result