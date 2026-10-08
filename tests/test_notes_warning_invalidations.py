"""№12 (3.2.0): правка заметки инвалидирует её предупреждение (FR-4.4).

`NoteService.set_warnings_resolver` — узкая инъекция (`WarningsService.
resolve_for_subject`), без циклических импортов. Правка `text`/`title`/`namespace`
и сшивание пары снимают открытые предупреждения заметки В ТОЙ ЖЕ транзакции.
"""

from __future__ import annotations

import pytest
from fakes import HashEmbedder

from app.config import Settings, get_settings
from app.services.notes import NoteService
from app.services.warnings import (
    KIND_NOTE_MISPLACED,
    KIND_NOTE_NO_NODE,
    WarningsService,
)
from app.storage.db import init_db, session

DIM = 8


@pytest.fixture
def settings(test_env: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setenv("EMBEDDING_DIM", str(DIM))
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    return settings


def _open_count(settings: Settings, note_id: int) -> int:
    with session(settings) as conn:
        return int(
            conn.execute(
                "SELECT COUNT(*) AS n FROM warnings WHERE subject_type = 'note' "
                "AND subject_id = ? AND status = 'open' AND resolved_at IS NULL",
                (note_id,),
            ).fetchone()["n"]
        )


def test_update_resolves_warning(settings: Settings) -> None:
    """Правка снимает предупреждение заметки; `merge_pair` — у обеих."""
    notes = NoteService(settings, embedding=HashEmbedder(DIM))
    warnings = WarningsService(settings, classifier=None, namespaces=None)
    notes.set_warnings_resolver(warnings.resolve_for_subject)

    older = notes.save("ранняя заметка про работу", author="t")["id"]
    newer = notes.save("поздняя заметка про дом", author="t")["id"]
    warnings.open(KIND_NOTE_MISPLACED, "note", older, "m", {})
    warnings.open(KIND_NOTE_NO_NODE, "note", newer, "m", {})

    notes.update(older, text="исправленный текст ранней заметки")
    assert _open_count(settings, older) == 0  # правка инвалидировала
    assert _open_count(settings, newer) == 1  # чужая заметка не тронута

    # Сшивание пары снимает предупреждения ОБЕИХ заметок (в той же транзакции).
    warnings.open(KIND_NOTE_MISPLACED, "note", older, "m2", {})
    notes.merge_pair(older, "слитый текст заметки", newer)
    assert _open_count(settings, older) == 0
    assert _open_count(settings, newer) == 0
