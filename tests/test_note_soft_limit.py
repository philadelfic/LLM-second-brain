"""Мягкий лимит текста заметки — из конфига (LSBN-S-1; 3.4.1, LSBN-T-26).

Решение О. 28.10.2026: лимит перестаёт быть константой кода
(`NOTE_TEXT_LIMIT` убрана) и живёт в Settings.max_note_chars
(env MAX_NOTE_CHARS, дефолт 2000, диапазон 500..20000). Одно число кормит
валидацию записи/правки, части merge/split, хинты, описания ручек,
инструкции и промпты — и одновременно CHECK новых баз + цель самомиграции
F-22 (env = живой CHECK активной базы держат сверка и самомиграция).

Граница берётся ИЗ КОНФИГА (не литерал): значение — сохраняется,
значение+1 — мягкий отказ с хинтом (лимит + фактическая длина + правило
разбиения). Смена env меняет ОДНОВРЕМЕННО валидацию, тексты (инструкции,
описания ручек, промпты), /health.note_chars_limit и цель миграции —
тест `TestEnvChangePropagates`. Сиды промптов: нетронутые перезаписываются
под новый канон, правленые оператором не трогаются
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

# Значения env для тестов конфиг-лимита: не литералы действующего лимита,
# а произвольные точки диапазона — граница всегда читается из Settings.
LIMIT_A = "1700"
LIMIT_B = "2300"


def _note_service() -> NoteService:
    """NoteService над текущими настройками с детерминированным кодировщиком."""
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
    """Граница мягкого лимита при сохранении — значение из конфига."""

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
    """Граница мягкого лимита при правке."""

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
    """env задаёт и валидацию, и CHECK новой базы — одно число (LSBN-T-26)."""

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
            init_db(settings)  # CHECK новой базы — то же число
            svc = _note_service()
            assert svc.save("е" * int(env_limit), title="Граница")["stored"] is True
            with pytest.raises(NoteValidationError):
                svc.save("ж" * (int(env_limit) + 1), title="Шаг влево — отказ")
        finally:
            get_settings.cache_clear()


class TestEnvChangePropagates:
    """Смена env меняет ОДНОВРЕМЕННО валидацию, тексты, /health и цель
    миграции (LSBN-T-26, приёмка): одно число — везде новое."""

    @staticmethod
    def _snapshot(monkeypatch: pytest.MonkeyPatch, tmp_path, env_limit: str) -> dict:
        """Собрать сервис и все тексты при заданном env; вернуть наблюдения."""
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
        # Валидация: граница каждого снимка — своё значение, хинт называет его.
        assert first["validation"]["stored"]["stored"] is True
        assert f"{second['limit']}-character note limit" in second["validation"]["hint"]
        assert f"{first['limit']}-character note limit" not in second["validation"][
            "hint"
        ]
        # Тексты: инструкции, описания ручек и промпт несут НОВОЕ число.
        for text in (
            second["instructions"],
            second["descriptions"]["memory_save"],
            second["descriptions"]["memory_update"],
            second["merge_prompt"],
        ):
            assert str(second["limit"]) in text
            assert str(first["limit"]) not in text
        # Цель миграции = то же env-значение (перелив/нарезка под него).
        assert first["migration_target"] == int(LIMIT_A)
        assert second["migration_target"] == int(LIMIT_B)

    def test_health_shows_env_value(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Число в /health — из env (проверка на живом приложении)."""
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
    """Лимит из настроек виден в текстах ручек/инструкций."""

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
        """F-23 заменил временный маркер усечения S-1 списком частей:
        усечённого слияния больше не существует — неполное слияние не
        сохраняется вовсе (дозапрос → откладывание пары, пп. 2/6 F-23)."""
        import app.services.summary as summary_module

        assert not hasattr(summary_module, "MERGE_TRUNCATION_MARKER")
        assert summary_module.MERGE_MAX_ROUNDS == 2
