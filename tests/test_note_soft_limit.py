"""Мягкий лимит текста заметки 1500 символов (LSBN-S-1, релиз 3.4.0).

Решение О. 15.10.2026: целевая длина заметки — 1500 символов; больше — это
несколько самодостаточных заметок. Лимит — число-дефолт кода
(`NOTE_TEXT_LIMIT` в app.services.notes), а НЕ env: MAX_NOTE_CHARS (35000)
и DDL держат паритет env ↔ живой CHECK (`_assert_note_chars_limit`) и не
трогаются — жёсткий CHECK на 1500 ставит только самомиграция (F-22).

Граница: 1500 — сохраняется, 1501 — мягкий отказ с хинтом (лимит +
фактическая длина + правило разбиения). REST отдаёт хинт как 422 detail.
Слияние (SummaryService.merge) держит результат в лимите с ЯВНЫМ маркером
усечения (юнит — tests/test_summary_service.py). Лимит виден оператору
в /health.note_chars_limit и в текстах ручек/инструкций.
"""

from __future__ import annotations

import pytest
from fakes import HashEmbedder

from app.config import get_settings
from app.services.notes import (
    HINT_NOTE_TOO_LONG,
    NOTE_TEXT_LIMIT,
    NoteService,
    NoteValidationError,
)
from app.services.summary import MERGE_TRUNCATION_MARKER
from app.storage.db import init_db, session
from app.transport.mcp import SERVER_INSTRUCTIONS, TOOL_DESCRIPTIONS


@pytest.fixture
def service(test_env: dict[str, str]) -> NoteService:
    """NoteService над тестовой БД с детерминированным кодировщиком."""
    settings = get_settings()
    init_db(settings)
    return NoteService(settings, HashEmbedder(settings.embedding_dim))


def _text_of(note_id: int) -> str:
    settings = get_settings()
    with session(settings) as conn:
        row = conn.execute("SELECT text FROM notes WHERE id = ?", (note_id,)).fetchone()
    assert row is not None
    return row["text"]


class TestSaveBoundary:
    """Граница мягкого лимита при сохранении."""

    def test_save_at_limit_is_stored(self, service: NoteService) -> None:
        result = service.save("а" * NOTE_TEXT_LIMIT, title="Заметка на границе")
        assert result["stored"] is True

    def test_save_over_limit_rejected_with_hint(self, service: NoteService) -> None:
        with pytest.raises(NoteValidationError) as exc_info:
            service.save("б" * (NOTE_TEXT_LIMIT + 1), title="Слишком длинная")
        hint = str(exc_info.value)
        assert hint == HINT_NOTE_TOO_LONG.format(
            actual=NOTE_TEXT_LIMIT + 1, limit=NOTE_TEXT_LIMIT
        )
        assert f"{NOTE_TEXT_LIMIT}-character note limit" in hint
        assert f"{NOTE_TEXT_LIMIT + 1} characters" in hint
        assert "self-contained notes" in hint

    def test_over_limit_note_is_not_written(self, service: NoteService) -> None:
        with pytest.raises(NoteValidationError):
            service.save("в" * (NOTE_TEXT_LIMIT + 1), title="Не записывается")
        settings = get_settings()
        with session(settings) as conn:
            count = conn.execute("SELECT count(*) AS n FROM notes").fetchone()["n"]
        assert count == 0


class TestUpdateBoundary:
    """Граница мягкого лимита при правке."""

    def test_update_over_limit_rejected_text_kept(
        self, service: NoteService
    ) -> None:
        note_id = service.save("исходный текст заметки", title="Правка")["id"]
        with pytest.raises(NoteValidationError) as exc_info:
            service.update(note_id=note_id, text="г" * (NOTE_TEXT_LIMIT + 1))
        assert f"{NOTE_TEXT_LIMIT}-character note limit" in str(exc_info.value)
        assert _text_of(note_id) == "исходный текст заметки"

    def test_update_at_limit_is_applied(self, service: NoteService) -> None:
        note_id = service.save("исходный текст заметки", title="Правка")["id"]
        result = service.update(note_id=note_id, text="д" * NOTE_TEXT_LIMIT)
        assert result["updated"] is True
        assert _text_of(note_id) == "д" * NOTE_TEXT_LIMIT


class TestStartupParity:
    """Env MAX_NOTE_CHARS и DDL не тронуты: старт с 35000 жив, лимит — 1500."""

    def test_startup_with_env_35000_and_soft_limit_active(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DB_PATH", str(tmp_path / "notes.db"))
        monkeypatch.setenv("MAX_NOTE_CHARS", "35000")
        get_settings.cache_clear()
        try:
            settings = get_settings()
            init_db(settings)  # старт не падает: env = живому CHECK
            assert settings.max_note_chars == 35000
            svc = NoteService(settings, HashEmbedder(settings.embedding_dim))
            with pytest.raises(NoteValidationError):
                svc.save("е" * 1501, title="Мягкий лимит на месте")
            assert svc.save("ж" * 1500, title="Граница")["stored"] is True
        finally:
            get_settings.cache_clear()


class TestVisibility:
    """Лимит виден в /health и в текстах ручек/инструкций."""

    def test_health_shows_note_chars_limit(self, client, token: str) -> None:
        response = client.get(
            "/health", headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 200
        assert response.json()["note_chars_limit"] == NOTE_TEXT_LIMIT

    def test_limit_named_in_tool_descriptions(self) -> None:
        for name in ("memory_save", "memory_update"):
            text = TOOL_DESCRIPTIONS[name]
            assert str(NOTE_TEXT_LIMIT) in text, name
            assert "split it" in text, name

    def test_rule_in_server_instructions(self) -> None:
        assert f"{NOTE_TEXT_LIMIT} characters" in SERVER_INSTRUCTIONS
        assert "several self-contained notes" in SERVER_INSTRUCTIONS

    def test_merge_marker_declares_limit(self) -> None:
        assert MERGE_TRUNCATION_MARKER.format(limit=NOTE_TEXT_LIMIT).startswith(
            "\n\n[truncated:"
        )
