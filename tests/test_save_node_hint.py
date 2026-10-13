"""Подсказка узла при `memory_save` без `namespace` (№28, релиз 3.2.0).

Заметка без `namespace` пишется в `default` и получает в ответе короткую
подсказку `namespace_hint`: 1..2 узла карты, похожих на слова заголовка и
начала текста, либо явное «подходящего узла нет» (FR-1.1…FR-1.6, arch
§13.1/§13.2). Подсказка считается по карте узлов и словам заметки — БЕЗ
вызова модели, не двигает заметку и не блокирует запись.
"""

from __future__ import annotations

import logging

import pytest

from app.config import get_settings
from app.services.embedding import EmbeddingService
from app.services.llm_client import LLMClient
from app.services.namespaces import NamespaceService
from app.services.notes import NoteService
from app.storage.db import init_db, session


def _settings(tmp_path, monkeypatch: pytest.MonkeyPatch, **env: str):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "notes.db"))
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    return settings


@pytest.fixture
def settings(tmp_path, monkeypatch: pytest.MonkeyPatch):
    return _settings(tmp_path, monkeypatch)


class TestSuggest:
    """`NamespaceService.suggest` — сравнение слов с путями и описаниями узлов."""

    def test_suggest_matches_description_word(self, settings) -> None:
        """Одного совпавшего значимого слова описания достаточно (вес 2.0)."""
        NamespaceService(settings).create(
            "work", "Рабочие заметки про СУБО и реестр зарплат"
        )
        nodes = NamespaceService(settings).suggest("СУБО реестр", limit=2, min_score=2.0)
        assert [node["path"] for node in nodes] == ["work"]
        assert nodes[0]["score"] >= 2.0

    def test_suggest_matches_path_segment(self, settings) -> None:
        """Сегмент пути весит больше слова описания (3.0)."""
        service = NamespaceService(settings)
        service.create("work", "Рабочие заметки")
        service.create("work/subo", "Подпроект по учёту")
        nodes = service.suggest("subo отчеты", limit=2, min_score=2.0)
        assert any(
            node["path"] == "work/subo" and node["score"] >= 3.0 for node in nodes
        )

    def test_suggest_prefix_matches_inflection(self, settings) -> None:
        """«серверы» ловит «сервера»: общий префикс ≥4 символов (морфология)."""
        NamespaceService(settings).create("infra", "Заметки про сервера и сеть")
        nodes = NamespaceService(settings).suggest("серверы", limit=2, min_score=2.0)
        assert [node["path"] for node in nodes] == ["infra"]

    def test_suggest_no_candidates_is_empty(self, settings) -> None:
        """Нет совпадений — пустой список: узел не выдумывается."""
        NamespaceService(settings).create("work", "Рабочие заметки")
        assert NamespaceService(settings).suggest("погода", limit=2, min_score=2.0) == []

    def test_suggest_returns_at_most_limit(self, settings) -> None:
        """Отбор: не больше `limit`, порядок по score убыв., тай-брейк по пути."""
        service = NamespaceService(settings)
        service.create("alpha", "Отчёты по серверам")
        service.create("beta", "Отчёты и планы")
        service.create("gamma", "Серверы и сеть")
        nodes = service.suggest("отчёт сервер", limit=2, min_score=2.0)
        assert [node["path"] for node in nodes] == ["alpha", "beta"]
        assert nodes[0]["score"] == 4.0
        assert nodes[1]["score"] == 2.0


class TestSaveNamespaceHint:
    """Поле `namespace_hint` ответа `save` при записи без явного узла."""

    def test_save_without_namespace_gets_namespace_hint(self, settings) -> None:
        """Запись без узла: stored, лежит в default, подсказка указывает узел."""
        NamespaceService(settings).create(
            "work", "Рабочие заметки про СУБО и реестр зарплат"
        )
        result = NoteService(settings).save(
            "Реестр зарплат СУБО: доступы и контакты", title="СУБО реестр зарплат"
        )
        assert result["stored"] is True
        assert "work" in result["namespace_hint"]
        with session(settings) as conn:
            stored = conn.execute(
                "SELECT namespace FROM notes WHERE id = ?", (result["id"],)
            ).fetchone()[0]
        assert stored == "default"

    def test_save_hint_when_no_candidate(self, settings) -> None:
        """FR-1.4: нет кандидатов — явный текст «no node matched»."""
        result = NoteService(settings).save(
            "Погода в Москве на выходные", title="Погода Москва"
        )
        assert result["stored"] is True
        assert "no node matched" in result["namespace_hint"]

    def test_save_with_explicit_namespace_has_no_hint(self, settings) -> None:
        """Явный узел — подсказки нет (просьбу не переспрашиваем)."""
        NamespaceService(settings).create("work", "Рабочие заметки про СУБО")
        result = NoteService(settings).save(
            "Заметка про СУБО", title="Про СУБО", namespace="work"
        )
        assert result["stored"] is True
        assert "namespace_hint" not in result

    def test_save_no_model_on_write(self, settings, monkeypatch: pytest.MonkeyPatch) -> None:
        """Подсказка — чтение карты, а не классификация: модели не зовём."""
        def _forbid(*args, **kwargs):
            raise AssertionError("model must not be called on write")

        monkeypatch.setattr(EmbeddingService, "embed", _forbid)
        monkeypatch.setattr(EmbeddingService, "embed_texts", _forbid)
        monkeypatch.setattr(LLMClient, "embed", _forbid)
        monkeypatch.setattr(LLMClient, "chat_result", _forbid)
        NamespaceService(settings).create("work", "Рабочие заметки про СУБО")
        result = NoteService(settings).save("Заметка про СУБО", title="Про СУБО")
        assert result["stored"] is True
        assert "namespace_hint" in result

    def test_save_hint_failure_does_not_break_save(
        self, settings, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Сбой подсказки: запись живёт, поля нет, событие в логе."""
        def _boom(*args, **kwargs):
            raise RuntimeError("карта недоступна")

        monkeypatch.setattr(NamespaceService, "suggest", _boom)
        caplog.set_level(logging.WARNING, logger="app")
        result = NoteService(settings).save("Заметка без узла", title="Без узла")
        assert result["stored"] is True
        assert "namespace_hint" not in result
        assert any(
            getattr(record, "event", None) == "save_hint_failed"
            for record in caplog.records
        )

    def test_save_hint_disabled(self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Выключатель гасит поле целиком (MEMORY_NODE_HINT_ENABLED=false)."""
        settings = _settings(tmp_path, monkeypatch, MEMORY_NODE_HINT_ENABLED="false")
        result = NoteService(settings).save("Заметка без узла", title="Без узла")
        assert result["stored"] is True
        assert "namespace_hint" not in result

    def test_save_positional_namespace_still_works(self, settings) -> None:
        """Позиционный вызов save(text, author, ns) сохраняет смысл."""
        NamespaceService(settings).create("work", "Рабочие заметки")
        result = NoteService(settings).save("Заметка в узле", "author-x", "work")
        assert result["stored"] is True
        assert "namespace_hint" not in result
        with session(settings) as conn:
            stored = conn.execute(
                "SELECT namespace FROM notes WHERE id = ?", (result["id"],)
            ).fetchone()[0]
        assert stored == "work"
