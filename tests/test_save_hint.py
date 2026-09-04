"""Дедуп-хинт чужого узла при save (Фаза 10, Шаг 5, US-8).

Близкий (дословный) дубль в другом узле — легитимен: запись не блокирует,
но в ответе — hint «похожее есть в <ns>» (сигнал ориентирования §5.7).
Дубль в своём узле по-прежнему блокирует запись (duplicated).
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
        """Дословный дубль в чужом узле: запись создаётся, hint указывает узел."""
        NamespaceService(settings).create("work", "Рабочие заметки.")
        notes = NoteService(settings)
        first = notes.save(TEXT, namespace="work")
        assert first["stored"] is True
        second = notes.save(TEXT)  # в default
        assert second["stored"] is True
        assert second["id"] != first["id"]
        assert "work" in second["hint"]
        assert "блокирует" in second["hint"]

    def test_normalized_duplicate_gives_hint(self, settings) -> None:
        """«Почти дословный» дубль (регистр/пробелы) — тоже hint."""
        NamespaceService(settings).create("work", "Рабочие заметки.")
        notes = NoteService(settings)
        notes.save(TEXT, namespace="work")
        tweaked = " ".join(("  " + TEXT.upper() + " ").split())
        second = notes.save(tweaked)
        assert second["stored"] is True
        assert "work" in second["hint"]

    def test_same_node_duplicate_still_blocks(self, settings) -> None:
        """Дословный дубль в своём узле — прежний контракт (duplicated)."""
        NamespaceService(settings).create("work", "Рабочие заметки.")
        notes = NoteService(settings)
        notes.save(TEXT, namespace="work")
        second = notes.save(TEXT, namespace="work")
        assert second["duplicated"] is True
        assert second["id"] == _id_of(settings, "work", TEXT)

    def test_no_hint_without_duplicates(self, settings) -> None:
        """Обычная запись — без hint (контракт ответа не растёт)."""
        notes = NoteService(settings)
        result = notes.save("уникальный текст заметки")
        assert "hint" not in result
        assert result["stored"] is True

    def test_no_hint_for_distinct_texts_in_other_node(self, settings) -> None:
        """Похожая по словам, но другая заметка в чужом узле — без hint."""
        NamespaceService(settings).create("work", "Рабочие заметки.")
        notes = NoteService(settings)
        notes.save(TEXT, namespace="work")
        result = notes.save("Совсем другой текст про другой проект")
        assert "hint" not in result