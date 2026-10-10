"""Тесты листинга области «Логи ассистента» (LSBN-F-10, f10-03).

Arch assistant-logs §6.1/§7.1/§7.2/§8/§13.16: `LogsService.list` отдаёт страницу
активных записей по контракту lsb-0013 (`total`/`has_more`/`next_offset`/
`next_cursor`), порядок — `created_at DESC, id DESC`, срезы — те же, что у
поиска (один SQL-предикат). Потолок `limit` параметризуется: MCP передаёт
`list_max_limit_mcp`, REST по умолчанию берёт `list_max_limit_rest` — тексты
отказа различаются (FR-5.4/NFR-2). Относительная дата отвергается тем же
`HINT_DATE_ABSOLUTE`, `offset < 0` — мягкий отказ (прецедент `SkillsService.list`).
"""

from __future__ import annotations

import pytest

from app.config import Settings, get_settings
from app.services.logs import (
    HINT_DATE_ABSOLUTE,
    LogsService,
    LogValidationError,
)
from app.storage.db import init_db, session

DIM = 8

REASON = "общий повод для замера скорости"
ACTION = "перенёс слот эмбеддингов на mainsrv"


@pytest.fixture
def settings(test_env: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Схема в тестовой БД (размерность не важна: листинг модель не зовёт)."""
    monkeypatch.setenv("EMBEDDING_DIM", str(DIM))
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    return settings


@pytest.fixture
def logs(settings: Settings) -> LogsService:
    """Сервис области без эмбеддера: листинг его не зовёт."""
    return LogsService(settings)


def _save(logs: LogsService, action: str = ACTION, **extra: object) -> int:
    """Записать лог и вернуть id."""
    result = logs.save(action, REASON, **extra)
    assert result["created"] is True
    return int(result["id"])


def _set_created(settings: Settings, log_id: int, moment: str) -> None:
    """Выставить `created_at` записи вручную (проверка порядка и срезов)."""
    with session(settings) as conn:
        conn.execute(
            "UPDATE assistant_logs SET created_at = ? WHERE id = ?",
            (moment, log_id),
        )


def _ids(page: dict) -> list[int]:
    return [item["id"] for item in page["items"]]


class TestListing:
    def test_list_pagination(self, logs: LogsService) -> None:
        """`total`/`has_more`/`next_offset` по контракту page_fields, cursor None."""
        for index in range(3):
            _save(logs, action=f"{ACTION} #{index}")
        first = logs.list(limit=2)
        assert set(first) == {
            "items",
            "total",
            "has_more",
            "next_offset",
            "next_cursor",
        }
        assert first["total"] == 3 and first["has_more"] is True
        assert first["next_offset"] == 2 and first["next_cursor"] is None
        assert len(first["items"]) == 2
        tail = logs.list(limit=2, offset=first["next_offset"])
        assert len(tail["items"]) == 1 and tail["has_more"] is False
        assert tail["next_offset"] is None and tail["total"] == 3

    def test_list_limit_ceiling(self, logs: LogsService) -> None:
        """MCP- и REST-потолки дают РАЗНЫЙ текст отказа `limit: expected 1..…`."""
        with pytest.raises(LogValidationError) as exc:
            logs.list(limit=50, max_limit=20)
        assert str(exc.value) == "limit: expected 1..20, got 50"
        with pytest.raises(LogValidationError) as exc:
            logs.list(limit=51)
        assert str(exc.value) == "limit: expected 1..50, got 51"
        with pytest.raises(LogValidationError) as exc:
            logs.list(limit=0, max_limit=20)
        assert str(exc.value) == "limit: expected 1..20, got 0"
        with pytest.raises(LogValidationError) as exc:
            logs.list(offset=-1)
        assert str(exc.value) == "offset: expected 0 or greater, got -1"

    def test_list_order(self, settings: Settings, logs: LogsService) -> None:
        """Порядок `created_at DESC, id DESC`; удалённые не видны."""
        old = _save(logs, action="старое")
        newest = _save(logs, action="новое")
        middle = _save(logs, action="среднее")
        _set_created(settings, old, "2026-01-01T10:00:00Z")
        _set_created(settings, middle, "2026-01-02T10:00:00Z")
        _set_created(settings, newest, "2026-01-03T10:00:00Z")
        assert _ids(logs.list()) == [newest, middle, old]
        logs.delete(middle)
        page = logs.list()
        assert _ids(page) == [newest, old] and page["total"] == 2
        # Одинаковый момент времени — тай-брейк по id DESC: выше id — первый.
        _set_created(settings, old, "2026-01-03T10:00:00Z")
        assert _ids(logs.list()) == [newest, old]

    def test_list_slices(self, logs: LogsService) -> None:
        """Срезы те же, что у поиска (один SQL-предикат §8)."""
        first = _save(logs, entity="skill", entity_id=5, entity_version=1)
        other = _save(logs, entity="skill", entity_id=6, session="agent-2")
        assert _ids(logs.list(entity="skill", entity_id=5)) == [first]
        assert _ids(logs.list(entity="skill", entity_id=5, entity_version=1)) == [
            first
        ]
        assert _ids(logs.list(entity="skill", entity_id=5, entity_version=2)) == []
        assert _ids(logs.list(session="agent-2")) == [other]
        assert _ids(logs.list(source="rest")) == []
        assert _ids(logs.list(date_from="2099-01-01")) == []
        assert _ids(logs.list(date_to="2000-01-01")) == []

    def test_list_relative_date_refused(self, logs: LogsService) -> None:
        """Относительная дата в листинге — тот же HINT_DATE_ABSOLUTE (§13.12)."""
        with pytest.raises(LogValidationError) as exc:
            logs.list(date_to="вчера")
        assert str(exc.value) == HINT_DATE_ABSOLUTE

    def test_list_empty_is_valid(self, logs: LogsService) -> None:
        """Пустой листинг — валидный исход без hint (не тот же, что поиск)."""
        page = logs.list()
        assert page["items"] == [] and page["total"] == 0
        assert page["has_more"] is False and page["next_offset"] is None
        assert "hint" not in page
