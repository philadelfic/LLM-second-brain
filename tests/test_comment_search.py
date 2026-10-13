"""Отдельный поиск пометок (no24-02, релиз 3.2.0): FTS `notes_comment_fts`.

Пометки ищутся ТОЛЬКО своей ручкой (`SearchService.comment_search` /
MCP `memory_comment_search` / REST `GET /notes/comments`) — смысловой поиск
заметок (`notes_fts` по title/text) пометок не видит (arch §13.6). Индекс
пометок синхронизируется триггером `notes_comment_fts_au` при установке/смене/
сбросе `comment`; миграция идемпотентна; фильтры (`namespace`, `updated_after`)
и лимит `top_k` работают.
"""

from __future__ import annotations

import pytest
from fakes import HashEmbedder

from app.config import get_settings
from app.services.namespaces import NamespaceError, NamespaceService
from app.services.notes import NoteService
from app.services.search import HINT_NO_COMMENTS, SearchService
from app.storage.db import init_db, session

DIM = 8


def _searcher(settings) -> SearchService:
    """SearchService с детерминированным HashEmbedder (без сети)."""
    return SearchService(settings, HashEmbedder(settings.embedding_dim))


@pytest.fixture
def settings():
    """Свежая БД с миграцией пометок + сервисы (без сети)."""
    init_db(get_settings())
    return get_settings()


def test_fts_migration_idempotent(settings) -> None:
    """После init_db таблица notes_comment_fts и оба триггера есть; повтор — no-op."""
    with session(settings) as conn:
        tables = {
            row["name"]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        triggers = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'trigger'"
            )
        }
    assert "notes_comment_fts" in tables
    assert {"notes_comment_fts_ai", "notes_comment_fts_au"} <= triggers
    assert "notes_fts" in tables  # смысловой индекс не тронут

    init_db(settings)  # повторный старт — идемпотентно
    with session(settings) as conn:
        again = {
            row["name"]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        triggers_again = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'trigger'"
            )
        }
    assert "notes_comment_fts" in again
    assert {"notes_comment_fts_ai", "notes_comment_fts_au"} <= triggers_again


def test_comment_indexed_on_update(settings) -> None:
    """notes.update(comment=...) → триггер индексирует пометку; поиск находит."""
    notes = NoteService(settings)
    saved = notes.save("Заметка про деплой контейнера", title="Деплой")
    notes.update(saved["id"], comment="пригодится к задаче X")

    found = _searcher(settings).comment_search("пригодится")
    assert [r["id"] for r in found["results"]] == [saved["id"]]
    assert found["results"][0]["comment"] == "пригодится к задаче X"


def test_comment_cleared_removed_from_index(settings) -> None:
    """Сброс пометки (comment=None) → заметка исчезает из поиска по пометкам."""
    notes = NoteService(settings)
    searcher = _searcher(settings)
    saved = notes.save("Заметка для сброса пометки", title="Сброс")
    notes.update(saved["id"], comment="уникальнаяпометка раз")
    assert searcher.comment_search("уникальнаяпометка")["results"]

    notes.update(saved["id"], comment=None)  # сброс пометки
    cleared = searcher.comment_search("уникальнаяпометка")
    assert cleared["results"] == []
    assert cleared["hint"] == HINT_NO_COMMENTS


def test_comment_search_only_by_comment(settings) -> None:
    """Слово из текста, но не из пометки, отдельный поиск не находит."""
    notes = NoteService(settings)
    saved = notes.save("секретслово встречается только в тексте", title="Текст")
    assert saved["id"]  # заметка без пометки

    found = _searcher(settings).comment_search("секретслово")
    assert found["results"] == []
    assert found["hint"] == HINT_NO_COMMENTS


def test_comment_search_filters(settings) -> None:
    """namespace/namespace_exact, updated_after, top_k и невалидный узел."""
    notes = NoteService(settings)
    namespaces = NamespaceService(settings)
    namespaces.create("work", "Рабочие заметки. Подпроекты — в листьях.")
    namespaces.create("work/sbos2020", "СУБО 2020: сервисы HR.")
    a = notes.save("первая", title="Первая", namespace="work/sbos2020")
    b = notes.save("вторая", title="Вторая", namespace="default")
    notes.update(a["id"], comment="общаяпометка раз")
    notes.update(b["id"], comment="общаяпометка два")
    searcher = _searcher(settings)

    # глобально — обе заметки
    everywhere = searcher.comment_search("общаяпометка", top_k=20)
    assert {r["id"] for r in everywhere["results"]} == {a["id"], b["id"]}

    # namespace=work — поддерево (сам узел + лист), только a
    work = searcher.comment_search("общаяпометка", namespace="work")
    assert [r["id"] for r in work["results"]] == [a["id"]]

    # namespace_exact у листа — только сам узел
    exact = searcher.comment_search(
        "общаяпометка", namespace="work/sbos2020", namespace_exact=True
    )
    assert [r["id"] for r in exact["results"]] == [a["id"]]

    # top_k соблюдён
    assert len(searcher.comment_search("общаяпометка", top_k=1)["results"]) == 1

    # updated_after в будущем — пусто + hint
    empty = searcher.comment_search("общаяпометка", updated_after="2999-01-01")
    assert empty["results"] == []
    assert empty["hint"] == HINT_NO_COMMENTS

    # незарегистрированный узел — NamespaceError (транспорт даст hint)
    with pytest.raises(NamespaceError):
        searcher.comment_search("общаяпометка", namespace="nope")
