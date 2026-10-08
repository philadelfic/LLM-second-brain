"""История заметок на путях правки (№41, постановка no41-02).

Версия пишется В ТОЙ ЖЕ транзакции, что и правка (FR-2.4, §13.5): `save` —
baseline (`version=1`), `update` — дельта на контентное изменение, `delete` —
версия-событие, `merge_pair` — версия у выжившей (`source='merge'`) и событие
удаления у поглощённой. Служебные поля (`vector_status`/`summary_status`/
`links_at`/`node_order_at`/`classified_at`) и авто-перегенерация `summary`
воркером версий НЕ создают (§13.3); явная передача `summary` — создаёт.
Источник и сессия версии приходят из contextvar (`bind_origin`, §13.4).
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import uuid
from typing import Any

import pytest
from fakes import FixedSummarizer, HashEmbedder

from app.config import Settings, get_settings
from app.services import history
from app.services.history import ENTITY_NOTE, bind_origin, reconstruct
from app.services.namespaces import NamespaceService
from app.services.notes import NoteService
from app.services.worker import BackgroundWorker
from app.storage.db import init_db, session


def unique(text: str) -> str:
    """Уникальный текст: дословный дедуп не должен блокировать запись."""
    return f"{text} [{uuid.uuid4().hex[:8]}]"


@pytest.fixture
def service() -> NoteService:
    settings = get_settings()
    init_db(settings)
    return NoteService(settings, HashEmbedder(settings.embedding_dim))


def _versions(entity: str, entity_id: int) -> list[sqlite3.Row]:
    with session(get_settings()) as conn:
        return list(
            conn.execute(
                "SELECT * FROM record_versions WHERE entity = ? AND entity_id = ? "
                "ORDER BY version",
                (entity, entity_id),
            )
        )


def _fields(entity: str, entity_id: int, version: int) -> dict[str, Any] | None:
    with session(get_settings()) as conn:
        return reconstruct(
            conn,
            entity,
            entity_id,
            version,
            inline_max=get_settings().history_delta_inline_max,
        )


def _live_projection(note_id: int) -> dict[str, Any]:
    with session(get_settings()) as conn:
        row = conn.execute(
            "SELECT text, title, namespace, expires_at, comment FROM notes "
            "WHERE id = ?",
            (note_id,),
        ).fetchone()
    return history.note_projection(row)


def _save(service: NoteService, text: str, title: str = "Заметка") -> int:
    return int(service.save(text, title=title)["id"])


def test_save_writes_baseline(service: NoteService) -> None:
    """Создание заметки: ровно одна версия `baseline` с полной проекцией."""
    note_id = _save(service, unique("История: базовый текст"), title="Базис")
    rows = _versions(ENTITY_NOTE, note_id)
    assert len(rows) == 1
    row = rows[0]
    assert int(row["version"]) == 1
    assert row["event"] == "baseline"
    assert json.loads(row["changed"]) == []
    snapshot = json.loads(row["snapshot"])
    assert snapshot["text"].startswith("История: базовый текст")
    assert snapshot["title"] == "Базис"
    assert snapshot["namespace"] == "default"
    assert snapshot["expires_at"] is None
    assert "summary" not in snapshot  # авто-выжимка вне проекции (§13.3)


def test_update_text_writes_delta(service: NoteService) -> None:
    """Правка `text` → v2 с `changed == ['text']`, версия восстановима."""
    note_id = _save(service, unique("История: первый текст"))
    service.update(note_id, text=unique("История: второй текст"))
    rows = _versions(ENTITY_NOTE, note_id)
    assert [int(row["version"]) for row in rows] == [1, 2]
    assert rows[1]["event"] == "content"
    assert json.loads(rows[1]["changed"]) == ["text"]
    assert rows[1]["snapshot"] is None and rows[1]["delta"] is not None
    assert _fields(ENTITY_NOTE, note_id, 2) == _live_projection(note_id)
    assert "второй текст" in _fields(ENTITY_NOTE, note_id, 2)["text"]


def test_update_title_only_writes_version(service: NoteService) -> None:
    """Правка только `title` → версия есть; служебные поля не тронуты."""
    note_id = _save(service, unique("История: заголовок"))
    with session(get_settings()) as conn:
        conn.execute(
            "UPDATE notes SET vector_status = 'ok', summary_status = 'ok' "
            "WHERE id = ?",
            (note_id,),
        )
    service.update(note_id, title="Новое название")
    rows = _versions(ENTITY_NOTE, note_id)
    assert len(rows) == 2
    changed = json.loads(rows[1]["changed"])
    assert changed == ["title"]
    assert "vector_status" not in changed and "summary_status" not in changed
    with session(get_settings()) as conn:
        row = conn.execute(
            "SELECT vector_status, summary_status FROM notes WHERE id = ?",
            (note_id,),
        ).fetchone()
    assert row["vector_status"] == "ok" and row["summary_status"] == "ok"


def test_update_namespace_writes_version(service: NoteService) -> None:
    """Переезд узла → версия с `changed == ['namespace']`."""
    NamespaceService(get_settings()).create("work", "Рабочие заметки.")
    note_id = _save(service, unique("История: переезд"))
    service.update(note_id, namespace="work")
    rows = _versions(ENTITY_NOTE, note_id)
    assert len(rows) == 2
    assert json.loads(rows[1]["changed"]) == ["namespace"]
    assert _fields(ENTITY_NOTE, note_id, 2) == _live_projection(note_id)


def test_service_fields_do_not_create_versions(service: NoteService) -> None:
    """Служебные сбросы после правки `text` отдельных версий не создают."""
    note_id = _save(service, unique("История: служебные"))
    service.update(note_id, text=unique("История: служебные после правки"))
    assert len(_versions(ENTITY_NOTE, note_id)) == 2


def test_summary_regen_does_not_create_version() -> None:
    """Прогон авто-суммаризации воркером новых версий не добавляет."""
    settings = get_settings()
    init_db(settings)
    service = NoteService(settings, HashEmbedder(settings.embedding_dim))
    note_id = _save(service, unique("История: авто-выжимка"))
    worker = BackgroundWorker(settings, HashEmbedder(settings.embedding_dim),
                              FixedSummarizer("Авто-выжимка воркера."))
    assert worker.process_summary_pending() == 1
    with session(settings) as conn:
        row = conn.execute(
            "SELECT summary, summary_status FROM notes WHERE id = ?", (note_id,)
        ).fetchone()
    assert row["summary_status"] == "ok"  # выжимка перегенерирована
    assert len(_versions(ENTITY_NOTE, note_id)) == 1  # но версии нет


def test_explicit_summary_in_delta(service: NoteService) -> None:
    """Явный `summary` в `update` попадает в проекцию и в `changed` (§13.3)."""
    note_id = _save(service, unique("История: явная выжимка"))
    service.update(note_id, summary="Явная выжимка строкой.")
    rows = _versions(ENTITY_NOTE, note_id)
    assert len(rows) == 2
    assert json.loads(rows[1]["changed"]) == ["summary"]
    assert _fields(ENTITY_NOTE, note_id, 2)["summary"] == "Явная выжимка строкой."


def test_delete_event_version(service: NoteService) -> None:
    """Soft delete → версия-событие `delete`; повторного удаления версии нет."""
    note_id = _save(service, unique("История: удаление"))
    assert service.delete(note_id)["deleted"] is True
    rows = _versions(ENTITY_NOTE, note_id)
    assert len(rows) == 2
    assert rows[1]["event"] == "delete"
    assert rows[1]["delta"] is None
    assert json.loads(rows[1]["changed"]) == []
    assert service.delete(note_id)["deleted"] is False  # повторно — нет версии
    assert len(_versions(ENTITY_NOTE, note_id)) == 2


def test_merge_pair_versions_both_sides(service: NoteService) -> None:
    """Слияние: выжившая — контент-версия `merge`, поглощённая — `delete`."""
    older_id = _save(service, unique("История: ранняя"))
    newer_id = _save(service, unique("История: поздняя"))
    result = service.merge_pair(older_id, unique("История: объединённый"), newer_id)
    assert result["merged"] is True and result["deleted"] is True
    older = _versions(ENTITY_NOTE, older_id)
    assert len(older) == 2
    assert older[1]["event"] == "content" and older[1]["source"] == "merge"
    assert json.loads(older[1]["changed"]) == ["text"]
    newer = _versions(ENTITY_NOTE, newer_id)
    assert len(newer) == 2
    assert newer[1]["event"] == "delete" and newer[1]["source"] == "merge"
    # История поглощённой читается и после мягкого удаления (FR-7.2).
    assert _fields(ENTITY_NOTE, newer_id, 1) is not None


def test_chain_equals_live_after_three_edits(service: NoteService) -> None:
    """Инвариант §13.3: `reconstruct(head)` == проекция живой строки (≥3 правки)."""
    note_id = _save(service, unique("История: цепочка"))
    for attempt in range(3):
        service.update(note_id, text=unique(f"История: цепочка {attempt}"))
    rows = _versions(ENTITY_NOTE, note_id)
    assert [int(row["version"]) for row in rows] == [1, 2, 3, 4]
    head = int(rows[-1]["version"])
    assert _fields(ENTITY_NOTE, note_id, head) == _live_projection(note_id)


def test_history_write_atomic(service: NoteService, monkeypatch: pytest.MonkeyPatch) -> None:
    """Сбой записи истории откатывает и саму правку (FR-2.4, §13.5)."""
    note_id = _save(service, "История: атомарность")
    before = _live_projection(note_id)

    def boom(*args: object, **kwargs: object) -> int:
        raise RuntimeError("history write failed")

    monkeypatch.setattr(history, "record_change", boom)
    with pytest.raises(RuntimeError):
        service.update(note_id, text="История: не должна примениться")
    assert _live_projection(note_id) == before  # текст прежний
    assert len(_versions(ENTITY_NOTE, note_id)) == 1  # новой версии нет


def test_origin_from_contextvar(service: NoteService) -> None:
    """`bind_origin` задаёт источник и сессию версии (§13.4)."""
    with bind_origin("mcp", "agent-7"):
        note_id = _save(service, unique("История: источник"))
    row = _versions(ENTITY_NOTE, note_id)[0]
    assert row["source"] == "mcp"
    assert row["session"] == "agent-7"


def test_rest_request_binds_origin(client, token: str) -> None:
    """Правка через REST несёт `source='rest'` и сессию из `X-LSB-Session`."""
    response = client.post(
        "/notes",
        json={"text": unique("История: REST-запрос"), "title": "REST-история"},
        headers={
            "Authorization": f"Bearer {token}",
            "X-LSB-Session": "agent-rest",
        },
    )
    assert response.status_code == 201, response.text
    note_id = int(response.json()["id"])
    row = _versions(ENTITY_NOTE, note_id)[0]
    assert row["source"] == "rest"
    assert row["session"] == "agent-rest"


def test_mcp_middleware_binds_origin() -> None:
    """MCP-middleware оборачивает запрос источником `mcp` (§13.4)."""
    from app.transport.mcp import InstructionsRefresher, _mcp_session

    class _Ctx:
        def __init__(self, params: dict | None, request_id: str | None) -> None:
            self.method = "tools/call"
            self.params = params
            self.request_id = request_id

    seen: dict[str, Any] = {}

    async def call_next(ctx: Any) -> str:
        seen["origin"] = history.current_origin()
        return "ok"

    refresher = InstructionsRefresher(None, None)  # type: ignore[arg-type]
    # Имя клиента из initialize (clientInfo.name) — сессия = идентификатор агента.
    assert asyncio.run(
        refresher(_Ctx({"clientInfo": {"name": "owui"}}, "req-1"), call_next)
    ) == "ok"
    assert seen["origin"] == ("mcp", "owui")
    # Нет clientInfo — сессия = id запроса (не имя человека).
    asyncio.run(refresher(_Ctx(None, "req-42"), call_next))
    assert seen["origin"] == ("mcp", "req-42")
    assert _mcp_session(_Ctx(None, None)) is None


def test_comment_edit_writes_one_content_version(service: NoteService) -> None:
    """no24-01 (3.2.0): правка `comment` — ровно ОДНА контент-версия.

    `changed == ['comment']`, `source`/`session` — из `bind_origin` (§13.4),
    text-версии не плодятся (повторное значение версии не создаёт).
    """
    note_id = _save(service, unique("История: комментарий"))
    baseline = len(_versions(ENTITY_NOTE, note_id))

    with bind_origin("mcp", session="owui"):
        assert service.update(
            note_id, comment="пригодится к задаче X"
        )["updated"] is True

    rows = _versions(ENTITY_NOTE, note_id)
    assert len(rows) == baseline + 1  # text-версии не плодятся
    version = rows[-1]
    assert version["event"] == "content"
    assert json.loads(version["changed"]) == ["comment"]
    assert version["source"] == "mcp" and version["session"] == "owui"
    fields = _fields(ENTITY_NOTE, note_id, int(version["version"]))
    assert fields["comment"] == "пригодится к задаче X"
    assert fields["text"].startswith("История: комментарий")  # текст не тронут

    with bind_origin("mcp", session="owui"):
        service.update(note_id, comment="пригодится к задаче X")  # без изменения
    assert len(_versions(ENTITY_NOTE, note_id)) == baseline + 1
