"""NoteService.merge_into_parts (LSBN-F-23, p. 5): the trace and the parts' links.

    The merge replaces a duplicate pair with a list of self-sufficient parts in
    ONE transaction: the parts are regular notes (the DB schema does not change —
    that is F-22); they inherit the links of BOTH originals and reference them by
    mention links; both originals go to trash by the regular soft delete and stay
    in history (a delete version-event, source="merge"). No LLM is called.
"""

from __future__ import annotations

import pytest
from fakes import HashEmbedder

from app.config import get_settings
from app.services.notes import NoteValidationError, NoteService, TitleValidationError
from app.storage.db import init_db, session


@pytest.fixture
def notes(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "notes.db"))
    monkeypatch.setenv("EMBEDDING_DIM", "8")
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    return NoteService(settings, HashEmbedder(8))


def _links(settings, note_id: int) -> list[tuple[int, int, str]]:
    with session(settings) as conn:
        rows = conn.execute(
            "SELECT note_a, note_b, kind FROM links "
            "WHERE note_a = ? OR note_b = ? ORDER BY note_a, note_b",
            (note_id, note_id),
        ).fetchall()
    return [(r["note_a"], r["note_b"], r["kind"]) for r in rows]


def test_parts_inherit_links_of_both_sources_and_mention_them(
    notes, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The parts inherit BOTH originals' links (kind/score preserved) and carry
            an explicit mention link to each original (p. 5).
    """
    older_id = notes.save("ранняя заметка пары", title="Ранняя пара")["id"]
    newer_id = notes.save("поздняя заметка пары", title="Поздняя пара")["id"]
    third_id = notes.save("третья заметка рядом", title="Третья рядом")["id"]
    settings = get_settings()
    with session(settings) as conn, conn:
        # Both originals' outward links: the earlier one→the third (entities), the later one→the third
        # (cosine); the pair itself has no row (the judge decided by cosine).
        conn.execute(
            "INSERT INTO links (note_a, note_b, kind, score) VALUES (?, ?, 'entities', 0.5)",
            (min(older_id, third_id), max(older_id, third_id)),
        )
        conn.execute(
            "INSERT INTO links (note_a, note_b, kind, score) VALUES (?, ?, 'cosine', 0.75)",
            (min(newer_id, third_id), max(newer_id, third_id)),
        )
    outcome = notes.merge_into_parts(
        older_id,
        newer_id,
        [("Встреча", "Встреча 12 сентября в 14:00, Браво."), ("Роадстран", "Роадстран обновлён до осени.")],
    )
    assert outcome["merged"] is True
    assert [p["title"] for p in outcome["parts"]] == ["Встреча", "Роадстран"]
    part_a, part_b = outcome["parts"][0]["id"], outcome["parts"][1]["id"]
    # Each part: a link to the third (both originals' inheritance goes onto ONE
    # "part↔third" pair — the one-kind-per-pair canon: entities beats
    # cosine) + a mention to BOTH originals (in trash).
    for part_id in (part_a, part_b):
        got = {(a, b, k) for a, b, k in _links(settings, part_id)}
        assert (min(part_id, third_id), max(part_id, third_id), "entities") in got
        assert (min(part_id, older_id), max(part_id, older_id), "mention") in got
        assert (min(part_id, newer_id), max(part_id, newer_id), "mention") in got
        assert len(got) == 3


def test_sources_go_to_trash_and_stay_in_history(
    notes, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both originals — in trash by the regular soft delete; in history — a delete
            version-event with source="merge" on each and the parts' content versions.
    """
    older_id = notes.save("текст первой", title="Первая")["id"]
    newer_id = notes.save("текст второй", title="Вторая")["id"]
    settings = get_settings()
    outcome = notes.merge_into_parts(
        older_id, newer_id, [("Слитая", "Обе заметки слились в одну часть.")]
    )
    part_id = outcome["parts"][0]["id"]
    with session(settings) as conn:
        rows = conn.execute(
            "SELECT id, deleted_at FROM notes ORDER BY id"
        ).fetchall()
        events = conn.execute(
            "SELECT entity_id, event, source FROM record_versions "
            "WHERE entity = 'note' AND event = 'delete' ORDER BY entity_id"
        ).fetchall()
        part_versions = conn.execute(
            "SELECT source FROM record_versions "
            "WHERE entity = 'note' AND entity_id = ? AND event != 'delete'",
            (part_id,),
        ).fetchall()
    assert [(r["id"], r["deleted_at"] is not None) for r in rows] == [
        (older_id, True),
        (newer_id, True),
        (part_id, False),
    ]
    assert [(e["entity_id"], e["source"]) for e in events] == [
        (older_id, "merge"),
        (newer_id, "merge"),
    ]
    assert [v["source"] for v in part_versions] == ["merge"]


def test_part_validation_and_guards(notes) -> None:
    """A part longer than the soft limit and a titleless part — a failure BEFORE the
            write; an empty parts list — a failure; no active earlier note — merged=False.
    """
    older_id = notes.save("текст ранней", title="Ранняя")["id"]
    newer_id = notes.save("текст поздней", title="Поздняя")["id"]
    limit = get_settings().max_note_chars
    long_text = "д" * (limit + 1)
    with pytest.raises(NoteValidationError, match=str(limit)):
        notes.merge_into_parts(older_id, newer_id, [("Слитая", long_text)])
    with pytest.raises(TitleValidationError):
        notes.merge_into_parts(older_id, newer_id, [(None, "текст без названия")])
    with pytest.raises(Exception, match="пустой список частей"):
        notes.merge_into_parts(older_id, newer_id, [])
    with session(get_settings()) as conn:
        assert conn.execute("SELECT COUNT(*) c FROM notes").fetchone()["c"] == 2
    gone = notes.merge_into_parts(
        10_000_001, newer_id, [("Слитая", "текст части")]
    )
    assert gone["merged"] is False and gone["parts"] == []
