"""Тесты сервиса области «Логи ассистента» (LSBN-F-10, f10-02).

Arch assistant-logs §6.1/§7.3/§5.4/§11: форма записи проверяется ДО БД, каждый
отказ — дословный hint (тексты сверяем литералами канона); запись атомарна
(INSERT + baseline-версия в одной транзакции), правка версионируется тем же
слоем истории (№41), удаление мягкое и тоже оставляет событие; «не передано» в
правке = «оставить»; `source`/`session` подставляет контекст вызова
(`bind_origin`), а не модель; SQL области адресует только свои таблицы.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from app.config import Settings, get_settings
from app.services import history
from app.services.logs import (
    ENTITY_KINDS,
    EXTERNAL_ENTITY_KINDS,
    HINT_ACTION_LIMIT,
    HINT_ACTION_REQUIRED,
    HINT_ENTITY_ID_REQUIRED,
    HINT_ENTITY_REF_LIMIT,
    HINT_ENTITY_REF_REQUIRED,
    HINT_ENTITY_UNKNOWN,
    HINT_ENTITY_VERSION_NEEDS_ID,
    HINT_NOT_FOUND,
    HINT_OUTCOME_LIMIT,
    HINT_REASON_LIMIT,
    HINT_REASON_REQUIRED,
    HINT_SOURCE_UNKNOWN,
    HINT_UPDATE_EMPTY,
    HINT_VERIFIED_LIMIT,
    VERSIONED_ENTITY_KINDS,
    LogsService,
    LogValidationError,
)
from app.storage.db import init_db, session

DIM = 8

# Дословные литералы канона arch §7.3 (править только в арх-доке).
CANON_HINTS = {
    HINT_ACTION_REQUIRED: (
        "not saved: action is required — say what was done (≤500 characters)"
    ),
    HINT_REASON_REQUIRED: (
        "not saved: reason is required — say why it was done (≤500 characters)"
    ),
    HINT_ACTION_LIMIT: "not saved: action limit is 500 characters — shorten it",
    HINT_REASON_LIMIT: "not saved: reason limit is 500 characters — shorten it",
    HINT_ENTITY_UNKNOWN: (
        "not saved: unknown entity '{given}' — one of: note, skill, term, "
        "user_fact, assistant_log, file, commit, tracker_task, config, contour"
    ),
    HINT_ENTITY_ID_REQUIRED: (
        "not saved: entity '{given}' is versioned — pass entity_id, and "
        "entity_version when the change was made to a known version"
    ),
    HINT_ENTITY_REF_REQUIRED: (
        "not saved: entity '{given}' has no id in this memory — pass entity_ref "
        "(a path, commit hash, task key or config name)"
    ),
    HINT_ENTITY_VERSION_NEEDS_ID: "not saved: entity_version requires entity_id",
    HINT_UPDATE_EMPTY: (
        "not updated: nothing passed — pass at least one field to change"
    ),
    HINT_NOT_FOUND: (
        "log entry not found — there is no such entry in this area; save it via "
        "logs_save(action, reason) if it is worth keeping"
    ),
}

ACTION = "перенёс слот эмбеддингов на mainsrv"
REASON = "замер показал вдвое большую скорость"


@pytest.fixture
def settings(test_env: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Схема в тестовой БД; размерность 8 — модель не участвует."""
    monkeypatch.setenv("EMBEDDING_DIM", str(DIM))
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    return settings


@pytest.fixture
def logs(settings: Settings) -> LogsService:
    """Сервис области без эмбеддера: путь записи его не зовёт (§13.2, поиск 03)."""
    return LogsService(settings)


def _rows(settings: Settings) -> list[dict]:
    """Все строки области (включая мягко удалённые) — сырая проверка факта."""
    with session(settings) as conn:
        return [
            dict(row)
            for row in conn.execute("SELECT * FROM assistant_logs ORDER BY id")
        ]


def _versions(settings: Settings, log_id: int) -> list[dict]:
    """Версии записи в общей истории (№41) — по возрастанию номера."""
    with session(settings) as conn:
        rows = conn.execute(
            "SELECT version, event, source, changed, snapshot, delta "
            "FROM record_versions WHERE entity = ? AND entity_id = ? "
            "ORDER BY version",
            (history.ENTITY_ASSISTANT_LOG, log_id),
        ).fetchall()
    return [dict(row) for row in rows]


def _fails(logs: LogsService, hint: str, **kwargs) -> None:
    """Попытка записи обязана упасть дословным hint'ом канона §7.3."""
    kwargs.setdefault("action", "что-то")
    kwargs.setdefault("reason", "зачем-то")
    with pytest.raises(LogValidationError) as exc:
        logs.save(**kwargs)
    assert str(exc.value) == hint


class TestDictionary:
    def test_entity_kinds_are_canon(self) -> None:
        """Словарь объекта §5.4: версионные / внешние / полный набор."""
        assert VERSIONED_ENTITY_KINDS == {
            "note",
            "skill",
            "term",
            "user_fact",
            "assistant_log",
        }
        assert EXTERNAL_ENTITY_KINDS == {
            "file",
            "commit",
            "tracker_task",
            "config",
            "contour",
        }
        assert ENTITY_KINDS == VERSIONED_ENTITY_KINDS | EXTERNAL_ENTITY_KINDS

    def test_hints_are_canon_verbatim(self) -> None:
        """Тексты мягких отказов — дословные константы §7.3."""
        for constant, canon in CANON_HINTS.items():
            assert constant == canon


class TestSave:
    def test_save_baseline_version(self, settings: Settings, logs: LogsService) -> None:
        """Запись → {created, id, version: 1}; baseline и pending-вектор."""
        answer = logs.save(ACTION, REASON)
        assert answer == {"created": True, "id": answer["id"], "version": 1}
        rows = _rows(settings)
        assert len(rows) == 1
        assert rows[0]["vector_status"] == "pending"
        assert rows[0]["deleted_at"] is None
        versions = _versions(settings, answer["id"])
        assert len(versions) == 1
        assert versions[0]["version"] == 1
        assert versions[0]["event"] == "baseline"
        # Проекция §5.4 — ровно контентные поля, без служебных (§5.4, FR-4.2).
        assert json.loads(versions[0]["snapshot"]) == (
            history.assistant_log_projection(rows[0])
        )

    def test_save_requires_action(self, settings: Settings, logs: LogsService) -> None:
        """Пустой `action` → отказ канона, записи нет."""
        _fails(logs, HINT_ACTION_REQUIRED, action="   ")
        assert _rows(settings) == []

    def test_save_requires_reason(self, settings: Settings, logs: LogsService) -> None:
        """Пустой `reason` → отказ канона, записи нет."""
        _fails(logs, HINT_REASON_REQUIRED, reason="")
        assert _rows(settings) == []

    def test_save_limits(self, settings: Settings, logs: LogsService) -> None:
        """Каждый текстовый лимит — свой hint; ни одна попытка не пишет."""
        _fails(
            logs,
            HINT_ACTION_LIMIT,
            action="a" * (settings.logs_action_max_chars + 1),
        )
        _fails(
            logs,
            HINT_REASON_LIMIT,
            reason="r" * (settings.logs_reason_max_chars + 1),
        )
        _fails(
            logs,
            HINT_ENTITY_REF_LIMIT,
            entity="commit",
            entity_ref="c" * (settings.logs_entity_ref_max_chars + 1),
        )
        _fails(
            logs,
            HINT_OUTCOME_LIMIT,
            outcome="o" * (settings.logs_outcome_max_chars + 1),
        )
        _fails(
            logs,
            HINT_VERIFIED_LIMIT,
            verified="v" * (settings.logs_verified_max_chars + 1),
        )
        assert _rows(settings) == []

    def test_save_entity_unknown(self, settings: Settings, logs: LogsService) -> None:
        """Тип объекта вне словаря §5.4 → мягкий отказ с перечислением."""
        _fails(logs, HINT_ENTITY_UNKNOWN.format(given="widget"), entity="widget")
        assert _rows(settings) == []

    def test_save_versioned_needs_id(
        self, settings: Settings, logs: LogsService
    ) -> None:
        """Версионная сущность без `entity_id` → отказ канона §7.3 п.3."""
        _fails(
            logs,
            HINT_ENTITY_ID_REQUIRED.format(given="note"),
            entity="note",
        )
        assert _rows(settings) == []

    def test_save_version_needs_id(
        self, settings: Settings, logs: LogsService
    ) -> None:
        """`entity_version` без `entity_id` → отказ канона §7.3 п.3."""
        _fails(logs, HINT_ENTITY_VERSION_NEEDS_ID, entity_version=3)
        assert _rows(settings) == []

    def test_save_external_needs_ref(
        self, settings: Settings, logs: LogsService
    ) -> None:
        """Неверсионный объект без текстовой ссылки — отказ; со ссылкой — запись."""
        _fails(
            logs,
            HINT_ENTITY_REF_REQUIRED.format(given="commit"),
            entity="commit",
        )
        assert _rows(settings) == []
        answer = logs.save(
            ACTION, REASON, entity="commit", entity_ref=" 1b999d7 "
        )
        assert answer["created"] is True
        row = _rows(settings)[0]
        assert (row["entity"], row["entity_ref"]) == ("commit", "1b999d7")

    def test_save_versioned_with_id_and_version(
        self, settings: Settings, logs: LogsService
    ) -> None:
        """Версионный объект с id и версией сохраняется как есть (§7.3 п.3)."""
        answer = logs.save(
            ACTION, REASON, entity="note", entity_id=41, entity_version=3
        )
        row = _rows(settings)[0]
        assert answer["created"] is True
        assert (row["entity"], row["entity_id"], row["entity_version"]) == (
            "note",
            41,
            3,
        )

    def test_save_defaults_from_origin(
        self, settings: Settings, logs: LogsService
    ) -> None:
        """`source`/`session` — из `bind_origin`, если не переданы (§7.1)."""
        with history.bind_origin("mcp", session="s-1"):
            logs.save(ACTION, REASON)
        row = _rows(settings)[0]
        assert (row["source"], row["session"]) == ("mcp", "s-1")
        # Явные значения вызывающего сильнее контекста вызова.
        with history.bind_origin("mcp", session="s-1"):
            logs.save(ACTION, REASON, source="job:areas-queue", session="j-2")
        row = _rows(settings)[1]
        assert (row["source"], row["session"]) == ("job:areas-queue", "j-2")

    def test_save_source_unknown(
        self, settings: Settings, logs: LogsService
    ) -> None:
        """Источник не из словаря §5.1 и без префикса `job:` — мягкий отказ."""
        _fails(
            logs,
            HINT_SOURCE_UNKNOWN.format(given="telepathy"),
            source="telepathy",
        )
        assert _rows(settings) == []

    def test_save_notifies_areas(self, settings: Settings, logs: LogsService) -> None:
        """Успешная запись будит петлю `areas` ровно один раз; отказ — не будит."""
        calls: list[int] = []
        logs.set_areas_notifier(lambda: calls.append(1))
        logs.save(ACTION, REASON)
        assert len(calls) == 1
        with pytest.raises(LogValidationError):
            logs.save("", REASON)
        assert len(calls) == 1


class TestRead:
    def test_get_full_record(self, settings: Settings, logs: LogsService) -> None:
        """`get` — полная запись §7.1, тело не режется (тела режет 03)."""
        log_id = logs.save(
            ACTION,
            REASON,
            entity="note",
            entity_id=41,
            entity_version=3,
            outcome="переключено",
            verified="замер изнутри контейнера",
        )
        record = logs.get(log_id["id"])
        assert record["id"] == log_id["id"]
        assert record["action"] == ACTION
        assert record["reason"] == REASON
        assert record["outcome"] == "переключено"
        assert record["verified"] == "замер изнутри контейнера"
        assert record["entity_id"] == 41 and record["entity_version"] == 3
        assert record["created_at"] and record["updated_at"]
        assert "hint" not in record

    def test_get_not_found_hint(self, settings: Settings, logs: LogsService) -> None:
        """Нет записи → мягкий ответ `{id, hint}` канона §7.3."""
        assert logs.get(404) == {"id": 404, "hint": HINT_NOT_FOUND}
        log_id = logs.save(ACTION, REASON)
        logs.delete(log_id["id"])
        assert logs.get(log_id["id"]) == {
            "id": log_id["id"],
            "hint": HINT_NOT_FOUND,
        }


class TestUpdate:
    def test_update_partial_keeps(self, settings: Settings, logs: LogsService) -> None:
        """Передано одно поле — остальные не тронуты (сентинелы)."""
        log_id = logs.save(ACTION, REASON, entity="note", entity_id=41)
        answer = logs.update(log_id["id"], outcome="вектор догнан")
        assert answer == {"id": log_id["id"], "updated": True}
        row = _rows(settings)[0]
        assert row["action"] == ACTION
        assert row["reason"] == REASON
        assert (row["entity"], row["entity_id"]) == ("note", 41)
        assert row["outcome"] == "вектор догнан"

    def test_update_creates_version(
        self, settings: Settings, logs: LogsService
    ) -> None:
        """Фактическая правка → версия-дельта + `vector_status='pending'`."""
        log_id = logs.save(ACTION, REASON)
        assert logs.update(log_id["id"], reason="передумал") == {
            "id": log_id["id"],
            "updated": True,
        }
        row = _rows(settings)[0]
        assert row["reason"] == "передумал"
        assert row["vector_status"] == "pending"
        versions = _versions(settings, log_id["id"])
        assert [version["version"] for version in versions] == [1, 2]
        assert versions[1]["event"] == "content"
        assert json.loads(versions[1]["changed"]) == ["reason"]
        assert json.loads(versions[1]["delta"]) == {"reason": "передумал"}

    def test_update_empty_hint(self, settings: Settings, logs: LogsService) -> None:
        """Ни одно поле не передано → `HINT_UPDATE_EMPTY`, версий не прибавилось."""
        log_id = logs.save(ACTION, REASON)
        before = _versions(settings, log_id["id"])
        assert logs.update(log_id["id"]) == {
            "id": log_id["id"],
            "updated": False,
            "hint": HINT_UPDATE_EMPTY,
        }
        assert _versions(settings, log_id["id"]) == before

    def test_update_not_found(self, settings: Settings, logs: LogsService) -> None:
        """Правка удалённой/несуществующей записи → `HINT_NOT_FOUND`."""
        log_id = logs.save(ACTION, REASON)
        logs.delete(log_id["id"])
        assert logs.update(log_id["id"], outcome="поздно") == {
            "id": log_id["id"],
            "updated": False,
            "hint": HINT_NOT_FOUND,
        }
        assert logs.update(404, outcome="мимо") == {
            "id": 404,
            "updated": False,
            "hint": HINT_NOT_FOUND,
        }

    def test_update_validates_entity_links(
        self, settings: Settings, logs: LogsService
    ) -> None:
        """Связки объекта проверяются и на правке (§7.3 п.3)."""
        log_id = logs.save(ACTION, REASON)
        with pytest.raises(LogValidationError) as exc:
            logs.update(log_id["id"], entity="note")
        assert str(exc.value) == HINT_ENTITY_ID_REQUIRED.format(given="note")
        with pytest.raises(LogValidationError) as exc:
            logs.update(log_id["id"], entity_version=2)
        assert str(exc.value) == HINT_ENTITY_VERSION_NEEDS_ID
        row = _rows(settings)[0]
        assert row["entity"] is None and row["entity_version"] is None


class TestDelete:
    def test_delete_soft_and_event(self, settings: Settings, logs: LogsService) -> None:
        """Удаление мягкое + событие истории; повторное — мягкий ответ."""
        log_id = logs.save(ACTION, REASON)
        assert logs.delete(log_id["id"]) == {"id": log_id["id"], "deleted": True}
        row = _rows(settings)[0]
        assert row["deleted_at"] is not None
        events = [version["event"] for version in _versions(settings, log_id["id"])]
        assert events == ["baseline", "delete"]
        assert logs.delete(log_id["id"]) == {
            "id": log_id["id"],
            "deleted": False,
            "hint": HINT_NOT_FOUND,
        }
        assert logs.delete(404) == {
            "id": 404,
            "deleted": False,
            "hint": HINT_NOT_FOUND,
        }
        # Строка жива: данные области не теряются (§11, FR-4.3).
        assert len(_rows(settings)) == 1


class TestIsolation:
    def test_logs_sql_isolation(self) -> None:
        """SQL сервиса адресует только таблицы области (§13.2/§13.4)."""
        source = Path("app/services/logs.py").read_text(encoding="utf-8")
        tables = set(
            re.findall(r"(?:FROM|INTO|UPDATE|JOIN)\s+([a-z_][a-z0-9_]*)", source)
        )
        assert tables == {"assistant_logs"}
        for foreign in (
            "notes",
            "skills",
            "terms",
            "user_facts",
            "record_versions",
            "warnings",
        ):
            assert not re.search(
                rf"(?:FROM|INTO|UPDATE|JOIN)\s+{foreign}", source
            ), foreign
