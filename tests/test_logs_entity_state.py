"""Тесты состояния объекта ссылки лога (LSBN-F-10, f10-04).

Arch assistant-logs §11/§13.4: чтение лога отвечает на вопрос «что стало с
объектом» — состояние считает СЛОЙ ИСТОРИИ (`history.resolve_entity`), сервис
области в чужие таблицы сырым SQL не ходит (изоляция области). `LogsService`
добавляет `entity_state` в `get` (только когда `entity` задан) и в каждый
элемент `list`/`search`: `active` / `deleted` / `unavailable` (объекта нет либо
его область выключена — не раскрываем, что область существует) / `external`
(неверсионный объект). `HISTORY_ENABLED=false` — состояние не достраивается,
выдача остаётся живой.
"""

from __future__ import annotations

import pytest

from app.config import Settings, get_settings
from app.services import history
from app.services.logs import LogsService
from app.services.notes import NoteService
from app.storage.db import init_db
from fakes import HashEmbedder

DIM = 8

ACTION = "перенёс слот эмбеддингов на mainsrv"
REASON = "замер показал вдвое быстрее"


@pytest.fixture
def settings(test_env: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setenv("EMBEDDING_DIM", str(DIM))
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    return settings


@pytest.fixture
def embedding() -> HashEmbedder:
    """Детерминированный эмбеддер: поиск области работает без сети."""
    return HashEmbedder(DIM)


@pytest.fixture
def logs(settings: Settings, embedding: HashEmbedder) -> LogsService:
    return LogsService(settings, embedding=embedding)


def _save(logs: LogsService, **extra: object) -> int:
    result = logs.save(ACTION, REASON, **extra)
    assert result["created"] is True, result
    return int(result["id"])


def _note(settings: Settings, text: str = "Заметка про слот") -> int:
    """Живая заметка настоящим сервисом: у неё есть версия 1 (№41)."""
    result = NoteService(settings, HashEmbedder(DIM)).save(text, title="Слот")
    return int(result["id"])


class TestResolveEntity:
    def test_state_active(self, settings: Settings, logs: LogsService) -> None:
        """Активная заметка → `active`, `current_version` — головная версия."""
        note_id = _note(settings)
        log_id = _save(logs, entity="note", entity_id=note_id)
        assert history.resolve_entity("note", note_id, settings) == {
            "state": "active",
            "current_version": 1,
        }
        assert logs.get(log_id)["entity_state"] == "active"

    def test_state_deleted(self, settings: Settings, logs: LogsService) -> None:
        """Мягко удалённый объект → `deleted`; запись лога жива (FR-4.3)."""
        note_id = _note(settings)
        NoteService(settings, HashEmbedder(DIM)).delete(note_id)
        log_id = _save(logs, entity="note", entity_id=note_id)
        record = logs.get(log_id)
        assert record["entity_state"] == "deleted"
        assert record["action"] == ACTION

    def test_state_unavailable_missing(
        self, settings: Settings, logs: LogsService
    ) -> None:
        """Объекта нет → `unavailable`."""
        log_id = _save(logs, entity="note", entity_id=999_999)
        assert logs.get(log_id)["entity_state"] == "unavailable"

    def test_state_unavailable_area_disabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Область объекта выключена → `unavailable`: область не раскрывается."""
        monkeypatch.setenv("EMBEDDING_DIM", str(DIM))
        monkeypatch.setenv("AREA_TERMS_ENABLED", "false")
        get_settings.cache_clear()
        settings = get_settings()
        init_db(settings)
        logs = LogsService(settings, embedding=HashEmbedder(DIM))
        log_id = _save(logs, entity="term", entity_id=1)
        resolved = history.resolve_entity("term", 1, settings)
        assert resolved == {"state": "unavailable", "current_version": None}
        assert logs.get(log_id)["entity_state"] == "unavailable"

    def test_state_external(self, settings: Settings, logs: LogsService) -> None:
        """Неверсионный объект (`commit`) → `external` (ссылка только текстом)."""
        log_id = _save(logs, entity="commit", entity_ref="deadbeef")
        assert history.resolve_entity("commit", 0, settings) is None
        assert logs.get(log_id)["entity_state"] == "external"

    def test_state_without_entity(self, settings: Settings, logs: LogsService) -> None:
        """`entity` не задан → поля `entity_state` в записи нет."""
        log_id = _save(logs)
        record = logs.get(log_id)
        assert record["action"] == ACTION
        assert "entity_state" not in record

    def test_resolve_history_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """`HISTORY_ENABLED=false` → состояние не достраивается, выдача жива."""
        monkeypatch.setenv("EMBEDDING_DIM", str(DIM))
        monkeypatch.setenv("HISTORY_ENABLED", "false")
        get_settings.cache_clear()
        settings = get_settings()
        init_db(settings)
        logs = LogsService(settings, embedding=HashEmbedder(DIM))
        log_id = _save(logs, entity="note", entity_id=1)
        assert history.resolve_entity("note", 1, settings) is None
        record = logs.get(log_id)
        assert record["action"] == ACTION and "hint" not in record
        assert "entity_state" not in record


class TestListAndSearchStates:
    def test_items_annotated(
        self, settings: Settings, embedding: HashEmbedder, logs: LogsService
    ) -> None:
        """Каждый элемент list/search несёт состояние своего объекта (§11)."""
        note_id = _note(settings)
        active = _save(logs, entity="note", entity_id=note_id)
        external = _save(logs, entity="commit", entity_ref="deadbeef")
        bare = _save(logs)
        expected = {active: "active", external: "external", bare: "external"}

        page = logs.list()
        assert {item["id"]: item["entity_state"] for item in page["items"]} == expected

        found = logs.search("эмбеддингов mainsrv")
        assert found["results"], found
        assert {
            hit["id"]: hit["entity_state"] for hit in found["results"]
        } == expected
