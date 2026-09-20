"""Listing contract (lsb-0013, release 3.1.0): surface-aware ceiling + page fields.

Arch lsb-0013 §3.1–3.2: both listings (`NoteService.list`, `SkillsService.list`)
accept the ceiling as a parameter (`max_limit`; None → the REST ceiling), reject
an out-of-range limit with the same message, and compute
`total` / `has_more` / `next_offset` / `next_cursor` through `page_fields`.
Existing hints and ordering stay as they were; `total` counts active records
only and follows the same `namespace` filter as the page.
"""

from __future__ import annotations

import uuid

import pytest
from fakes import FailingEmbedder, HashEmbedder, clear_seeded_skills

from app.config import get_settings
from app.services.listing import page_fields
from app.services.namespaces import NamespaceService
from app.services.notes import NoteService, NoteValidationError
from app.services.skills import SkillValidationError, SkillsService
from app.storage.db import init_db

MCP_LIMIT = 20  # LIST_MAX_LIMIT_MCP default
REST_LIMIT = 50  # LIST_MAX_LIMIT_REST default


def unique(text: str) -> str:
    """Unique text: the literal dedup must not fold seeded notes together."""
    return f"{text} [{uuid.uuid4().hex[:8]}]"


def _skill_form(name: str) -> dict[str, str]:
    """Valid skill form; `name` varies per skill (antiseonymy is skipped)."""
    return {
        "name": name,
        "description": "How to deploy this service",
        "steps": "1) build; 2) ship; 3) verify",
        "text": "Run make deploy, then check /health.",
    }


@pytest.fixture
def notes() -> NoteService:
    """NoteService over a fresh DB with a deterministic embedder (no network)."""
    settings = get_settings()
    init_db(settings)
    return NoteService(settings, HashEmbedder(settings.embedding_dim))


@pytest.fixture
def skills() -> SkillsService:
    """SkillsService over a fresh DB; the failing embedder skips antiseonymy."""
    settings = get_settings()
    init_db(settings)
    clear_seeded_skills(settings)  # seeding the creator skill off: empty registry
    return SkillsService(settings, FailingEmbedder())


class TestPageFields:
    """The helper is the single place where the page fields are computed."""

    def test_contract_keys(self) -> None:
        assert set(page_fields(total=5, offset=0, count=5)) == {
            "total",
            "has_more",
            "next_offset",
            "next_cursor",
        }

    @pytest.mark.parametrize(
        ("total", "offset", "count", "has_more", "next_offset"),
        [
            (0, 0, 0, False, None),  # empty memory
            (19, 0, 19, False, None),  # below the ceiling
            (20, 0, 20, False, None),  # exactly the ceiling
            (21, 0, 20, True, 20),  # one record more — next page exists
            (21, 20, 1, False, None),  # last page
            (20, 20, 0, False, None),  # page beyond the memory
        ],
    )
    def test_borders(
        self,
        total: int,
        offset: int,
        count: int,
        has_more: bool,
        next_offset: int | None,
    ) -> None:
        fields = page_fields(total, offset, count)
        assert fields["total"] == total
        assert fields["has_more"] is has_more
        assert fields["next_offset"] == next_offset
        assert fields["next_cursor"] is None  # reserved for a future keyset

    def test_next_offset_uses_real_count_not_limit(self) -> None:
        """`next_offset = offset + count`: the last page may be shorter."""
        assert page_fields(total=30, offset=0, count=5)["next_offset"] == 5


class TestNotesListLimit:
    """Surface ceilings of `NoteService.list` (arch lsb-0013 §3.1–3.2)."""

    def test_mcp_ceiling_accepts_borders(self, notes: NoteService) -> None:
        assert notes.list(limit=1, max_limit=MCP_LIMIT)["items"] == []
        assert notes.list(limit=MCP_LIMIT, max_limit=MCP_LIMIT)["items"] == []

    @pytest.mark.parametrize("limit", [0, -1, MCP_LIMIT + 1, REST_LIMIT])
    def test_mcp_ceiling_rejects_outside(
        self, notes: NoteService, limit: int
    ) -> None:
        with pytest.raises(NoteValidationError) as exc:
            notes.list(limit=limit, max_limit=MCP_LIMIT)
        assert str(exc.value) == f"limit: expected 1..{MCP_LIMIT}, got {limit}"

    def test_rest_ceiling_accepts_borders(self, notes: NoteService) -> None:
        assert notes.list(limit=1, max_limit=REST_LIMIT)["items"] == []
        assert notes.list(limit=REST_LIMIT, max_limit=REST_LIMIT)["items"] == []

    def test_rest_ceiling_rejects_above(self, notes: NoteService) -> None:
        with pytest.raises(NoteValidationError) as exc:
            notes.list(limit=REST_LIMIT + 1, max_limit=REST_LIMIT)
        assert str(exc.value) == (
            f"limit: expected 1..{REST_LIMIT}, got {REST_LIMIT + 1}"
        )

    def test_default_ceiling_is_rest(self, notes: NoteService) -> None:
        """`max_limit=None` → REST ceiling: existing callers keep working."""
        assert notes.list(limit=REST_LIMIT)["items"] == []


class TestNotesListFields:
    """Pagination fields in every branch; hints and totals unchanged."""

    def test_empty_memory_keeps_hint_and_fields(self, notes: NoteService) -> None:
        result = notes.list()
        assert result["items"] == []
        assert result["hint"] == "memory is empty"
        assert result["total"] == 0
        assert result["has_more"] is False
        assert result["next_offset"] is None
        assert result["next_cursor"] is None

    def test_page_beyond_keeps_hint_and_fields(self, notes: NoteService) -> None:
        for i in range(3):
            notes.save(unique(f"note {i}"))
        result = notes.list(limit=MCP_LIMIT, offset=10)
        assert result["items"] == []
        assert result["hint"] == (
            "page beyond the memory: offset ≥ total; reduce offset"
        )
        assert result["total"] == 3
        assert result["has_more"] is False
        assert result["next_offset"] is None
        assert result["next_cursor"] is None

    def test_page_borders(self, notes: NoteService) -> None:
        """21 records, ceiling 20: first page has more, second page is last."""
        for i in range(21):
            notes.save(unique(f"note {i}"))
        first = notes.list(limit=MCP_LIMIT, offset=0, max_limit=MCP_LIMIT)
        assert len(first["items"]) == MCP_LIMIT
        assert first["total"] == 21
        assert first["has_more"] is True
        assert first["next_offset"] == MCP_LIMIT
        assert first["next_cursor"] is None
        second = notes.list(limit=MCP_LIMIT, offset=20, max_limit=MCP_LIMIT)
        assert len(second["items"]) == 1
        assert second["total"] == 21
        assert second["has_more"] is False
        assert second["next_offset"] is None

    def test_total_excludes_soft_deleted_and_follows_namespace_filter(
        self, notes: NoteService
    ) -> None:
        settings = get_settings()
        namespaces = NamespaceService(settings)
        namespaces.create("work", "Work notes.")
        namespaces.create("work/sub", "Work subproject notes.")
        notes.save(unique("root work note"), namespace="work")
        for i in range(3):
            notes.save(unique(f"sub note {i}"), namespace="work/sub")
        notes.delete(notes.list(limit=10, namespace="work/sub")["items"][0]["id"])

        subtree = notes.list(limit=REST_LIMIT, namespace="work")
        assert subtree["total"] == 3  # soft-deleted is not counted
        assert len(subtree["items"]) == 3
        assert subtree["has_more"] is False
        exact = notes.list(limit=REST_LIMIT, namespace="work", namespace_exact=True)
        assert exact["total"] == 1
        assert [item["namespace"] for item in exact["items"]] == ["work"]
        assert exact["has_more"] is False


class TestSkillsList:
    """`skills.list` behaves exactly like `notes.list` (same signature/fields)."""

    def test_empty_registry_has_page_fields(self, skills: SkillsService) -> None:
        result = skills.list()
        assert result["items"] == []
        assert result["total"] == 0
        assert result["has_more"] is False
        assert result["next_offset"] is None
        assert result["next_cursor"] is None

    def test_mcp_ceiling_borders_and_rejections(
        self, skills: SkillsService
    ) -> None:
        assert skills.list(limit=MCP_LIMIT, max_limit=MCP_LIMIT)["items"] == []
        for limit in (0, MCP_LIMIT + 1):
            with pytest.raises(SkillValidationError) as exc:
                skills.list(limit=limit, max_limit=MCP_LIMIT)
            assert str(exc.value) == f"limit: expected 1..{MCP_LIMIT}, got {limit}"

    def test_rest_ceiling_and_default(self, skills: SkillsService) -> None:
        assert skills.list(limit=REST_LIMIT)["items"] == []  # None → REST ceiling
        with pytest.raises(SkillValidationError) as exc:
            skills.list(limit=REST_LIMIT + 1, max_limit=REST_LIMIT)
        assert str(exc.value) == (
            f"limit: expected 1..{REST_LIMIT}, got {REST_LIMIT + 1}"
        )

    def test_page_borders_and_deleted_total(self, skills: SkillsService) -> None:
        for i in range(21):
            skills.save(**_skill_form(f"Deploy skill {i}"))
        first = skills.list(limit=MCP_LIMIT, max_limit=MCP_LIMIT)
        assert len(first["items"]) == MCP_LIMIT
        assert first["total"] == 21
        assert first["has_more"] is True
        assert first["next_offset"] == MCP_LIMIT
        second = skills.list(limit=MCP_LIMIT, offset=MCP_LIMIT, max_limit=MCP_LIMIT)
        assert len(second["items"]) == 1
        assert second["has_more"] is False
        assert second["next_offset"] is None

        skills.delete(first["items"][0]["id"])
        after_delete = skills.list(limit=MCP_LIMIT, max_limit=MCP_LIMIT)
        assert after_delete["total"] == 20  # soft-deleted is not counted
        assert after_delete["has_more"] is False
        assert after_delete["next_offset"] is None
