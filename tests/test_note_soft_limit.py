"""The note text's soft limit — from the config (LSBN-S-1; 3.4.1, LSBN-T-26).

    The decision: the limit stops being a code constant (`NOTE_TEXT_LIMIT`
    removed) and lives in Settings.max_note_chars
    (env MAX_NOTE_CHARS, default 2000, range 500..20000). One number feeds the
    save/edit validation, the merge/split parts, the hints, the handle
    descriptions, the instructions and the prompts — and at the same time the
    CHECK of new databases + the F-22 self-migration target (env = the active
    database's live CHECK, held by the reconciliation and the self-migration).

    The boundary is taken FROM THE CONFIG (not a literal): the value — saved;
    the value+1 — a soft failure with a hint (the limit + the actual length + the
    splitting rule). An env change simultaneously changes the validation, the
    texts (the instructions, the handle descriptions, the prompts),
    /health.note_chars_limit and the migration target — the
    `TestEnvChangePropagates` test. The prompt seeds: untouched ones are
    rewritten to the new canon; operator-edited ones are not touched
    (tests/test_prompts_registry.py::TestSeedValueRewrite).
"""

from __future__ import annotations

import pytest
from fakes import HashEmbedder

from app.config import get_settings
from app.services.notes import (
    HINT_NOTE_TOO_LONG,
    NoteService,
    NoteValidationError,
)
from app.services.prompts import PromptRegistry
from app.storage.db import init_db, session
from app.storage.migration import startup_plan
from app.transport.mcp import server_instructions, tool_descriptions

# The env values for the config-limit tests: not the active limit's literals
# but arbitrary points of the range — the boundary is always read from Settings.
LIMIT_A = "1700"
LIMIT_B = "2300"


def _note_service() -> NoteService:
    """A NoteService over the current settings with a deterministic encoder."""
    settings = get_settings()
    init_db(settings)
    return NoteService(settings, HashEmbedder(settings.embedding_dim))


@pytest.fixture
def service(test_env: dict[str, str]) -> NoteService:
    return _note_service()


def _text_of(note_id: int) -> str:
    settings = get_settings()
    with session(settings) as conn:
        row = conn.execute("SELECT text FROM notes WHERE id = ?", (note_id,)).fetchone()
    assert row is not None
    return row["text"]


class TestSaveBoundary:
    """The soft limit's boundary at save — the value from the config."""

    def test_save_at_limit_is_stored(self, service: NoteService) -> None:
        limit = get_settings().max_note_chars
        result = service.save("а" * limit, title="Заметка на границе")
        assert result["stored"] is True

    def test_save_over_limit_rejected_with_hint(self, service: NoteService) -> None:
        limit = get_settings().max_note_chars
        with pytest.raises(NoteValidationError) as exc_info:
            service.save("б" * (limit + 1), title="Слишком длинная")
        hint = str(exc_info.value)
        assert hint == HINT_NOTE_TOO_LONG.format(actual=limit + 1, limit=limit)
        assert f"{limit}-character note limit" in hint
        assert f"{limit + 1} characters" in hint
        assert "self-contained notes" in hint

    def test_over_limit_note_is_not_written(self, service: NoteService) -> None:
        limit = get_settings().max_note_chars
        with pytest.raises(NoteValidationError):
            service.save("в" * (limit + 1), title="Не записывается")
        settings = get_settings()
        with session(settings) as conn:
            count = conn.execute("SELECT count(*) AS n FROM notes").fetchone()["n"]
        assert count == 0


class TestUpdateBoundary:
    """The soft limit's boundary at edit."""

    def test_update_over_limit_rejected_text_kept(
        self, service: NoteService
    ) -> None:
        limit = get_settings().max_note_chars
        note_id = service.save("исходный текст заметки", title="Правка")["id"]
        with pytest.raises(NoteValidationError) as exc_info:
            service.update(note_id=note_id, text="г" * (limit + 1))
        assert f"{limit}-character note limit" in str(exc_info.value)
        assert _text_of(note_id) == "исходный текст заметки"

    def test_update_at_limit_is_applied(self, service: NoteService) -> None:
        limit = get_settings().max_note_chars
        note_id = service.save("исходный текст заметки", title="Правка")["id"]
        result = service.update(note_id=note_id, text="д" * limit)
        assert result["updated"] is True
        assert _text_of(note_id) == "д" * limit


class TestEnvDrivesLimitAndSchema:
    """env sets both the validation and the new database's CHECK — one number (LSBN-T-26)."""

    @pytest.mark.parametrize("env_limit", [LIMIT_A, LIMIT_B])
    def test_env_value_becomes_working_limit(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch, env_limit: str
    ) -> None:
        monkeypatch.setenv("DB_PATH", str(tmp_path / "notes.db"))
        monkeypatch.setenv("MAX_NOTE_CHARS", env_limit)
        get_settings.cache_clear()
        try:
            settings = get_settings()
            assert settings.max_note_chars == int(env_limit)
            init_db(settings)  # the new database's CHECK — the same number
            svc = _note_service()
            assert svc.save("е" * int(env_limit), title="Граница")["stored"] is True
            with pytest.raises(NoteValidationError):
                svc.save("ж" * (int(env_limit) + 1), title="Шаг влево — отказ")
        finally:
            get_settings.cache_clear()


class TestEnvChangePropagates:
    """An env change simultaneously changes the validation, the texts, /health and the
            migration target (LSBN-T-26, acceptance): one number — new everywhere.
    """

    @staticmethod
    def _snapshot(monkeypatch: pytest.MonkeyPatch, tmp_path, env_limit: str) -> dict:
        """Assemble the service and all texts at a given env; return the observations."""
        monkeypatch.setenv("DB_PATH", str(tmp_path / f"notes-{env_limit}.db"))
        monkeypatch.setenv("MAX_NOTE_CHARS", env_limit)
        get_settings.cache_clear()
        try:
            settings = get_settings()
            svc = _note_service()
            validation = {
                "stored": svc.save("з" * settings.max_note_chars, title="В границе"),
            }
            with pytest.raises(NoteValidationError) as exc_info:
                svc.save("и" * (settings.max_note_chars + 1), title="За границей")
            validation["hint"] = str(exc_info.value)
            return {
                "limit": settings.max_note_chars,
                "validation": validation,
                "instructions": server_instructions(settings.max_note_chars),
                "descriptions": tool_descriptions(settings.max_note_chars),
                "merge_prompt": PromptRegistry(
                    max_note_chars=settings.max_note_chars
                ).summary_merge_system,
                "migration_target": startup_plan(settings)["target"],
            }
        finally:
            get_settings.cache_clear()

    def test_all_surfaces_move_together(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        first = self._snapshot(monkeypatch, tmp_path, LIMIT_A)
        second = self._snapshot(monkeypatch, tmp_path, LIMIT_B)
        assert first["limit"] == int(LIMIT_A)
        assert second["limit"] == int(LIMIT_B)
        # Validation: each snapshot's boundary is its own value; the hint names it.
        assert first["validation"]["stored"]["stored"] is True
        assert f"{second['limit']}-character note limit" in second["validation"]["hint"]
        assert f"{first['limit']}-character note limit" not in second["validation"][
            "hint"
        ]
        # The texts: the instructions, the handle descriptions and the prompt carry the NEW number.
        for text in (
            second["instructions"],
            second["descriptions"]["memory_save"],
            second["descriptions"]["memory_update"],
            second["merge_prompt"],
        ):
            assert str(second["limit"]) in text
            assert str(first["limit"]) not in text
        # The migration target = the same env value (the pour/split aims at it).
        assert first["migration_target"] == int(LIMIT_A)
        assert second["migration_target"] == int(LIMIT_B)

    def test_health_shows_env_value(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The number in /health — from env (checked on the live application)."""
        from fastapi.testclient import TestClient

        from app.main import create_app

        monkeypatch.setenv("DB_PATH", str(tmp_path / "notes.db"))
        monkeypatch.setenv("MAX_NOTE_CHARS", LIMIT_B)
        get_settings.cache_clear()
        try:
            init_db(get_settings())
            with TestClient(create_app()) as test_client:
                response = test_client.get("/health")
            assert response.status_code == 200
            assert response.json()["note_chars_limit"] == int(LIMIT_B)
        finally:
            get_settings.cache_clear()


class TestVisibility:
    """The settings' limit is visible in the handle/instruction texts."""

    def test_limit_named_in_tool_descriptions(self) -> None:
        limit = get_settings().max_note_chars
        descriptions = tool_descriptions(limit)
        for name in ("memory_save", "memory_update"):
            text = descriptions[name]
            assert str(limit) in text, name
            assert "split it" in text, name

    def test_rule_in_server_instructions(self) -> None:
        limit = get_settings().max_note_chars
        instructions = server_instructions(limit)
        assert f"{limit} characters" in instructions
        assert "several self-contained notes" in instructions

    def test_merge_truncation_marker_gone(self) -> None:
        """F-23 replaced S-1's temporary truncation marker with a list of parts:
                    a truncated merge no longer exists — an incomplete merge is not saved
                    at all (a follow-up call → deferring the pair, F-23 pp. 2/6).
        """
        import app.services.summary as summary_module

        assert not hasattr(summary_module, "MERGE_TRUNCATION_MARKER")
        assert summary_module.MERGE_MAX_ROUNDS == 2
