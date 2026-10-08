"""Источник версий фоновых джоб (№41, постановка no41-02).

Переезд `namespace` джобой `nodes` — контентное изменение (FR-3.1/FR-4.4):
версия несёт пометку `source='job:nodes'` из `bind_origin` тела джобы (§13.4),
а не «правил оператор».
"""

from __future__ import annotations

import asyncio
import json
import sqlite3

import pytest
from fakes import FixedSummarizer, HashEmbedder

from app.config import Settings, get_settings
from app.services.history import ENTITY_NOTE
from app.services.jobs import build_job_specs
from app.services.namespaces import NamespaceService
from app.services.notes import NoteService
from app.services.worker import NODES_JOB, BackgroundWorker
from app.storage.db import init_db, session

DIM = 8


@pytest.fixture
def settings(test_env: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setenv("EMBEDDING_DIM", str(DIM))
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    return settings


def _versions(entity_id: int) -> list[sqlite3.Row]:
    with session(get_settings()) as conn:
        return list(
            conn.execute(
                "SELECT * FROM record_versions WHERE entity = ? AND entity_id = ? "
                "ORDER BY version",
                (ENTITY_NOTE, entity_id),
            )
        )


def _nodes_spec(worker: BackgroundWorker, settings: Settings):
    specs = {spec.name: spec for spec in build_job_specs(worker, settings)}
    return specs[NODES_JOB]


def test_nodes_job_source(settings: Settings) -> None:
    """Переклассификация из джобы `nodes` даёт версию `source='job:nodes'`."""
    NamespaceService(settings).create("work", "Рабочие заметки.")
    service = NoteService(settings, HashEmbedder(DIM))
    note_id = int(service.save("История: переезд джобой", title="Джоба")["id"])
    # Готовая разметка — быстрый пул обхода (переезд без вызова модели).
    with session(settings) as conn:
        conn.execute(
            "UPDATE notes SET hint_path = 'work', confidence = 0.95 WHERE id = ?",
            (note_id,),
        )
    worker = BackgroundWorker(settings, HashEmbedder(DIM), FixedSummarizer("Фикс."))
    assert asyncio.run(_nodes_spec(worker, settings).process()) == 1
    with session(settings) as conn:
        row = conn.execute(
            "SELECT namespace FROM notes WHERE id = ?", (note_id,)
        ).fetchone()
    assert row["namespace"] == "work"
    rows = _versions(note_id)
    assert len(rows) == 2
    assert rows[1]["event"] == "content"
    assert rows[1]["source"] == "job:nodes"
    assert json.loads(rows[1]["changed"]) == ["namespace"]


def test_summary_job_does_not_write_versions(settings: Settings) -> None:
    """Джоба `summary` (авто-выжимка) новых версий не пишет (§13.3)."""
    service = NoteService(settings, HashEmbedder(DIM))
    note_id = int(service.save("История: фоновая выжимка", title="Фон")["id"])
    worker = BackgroundWorker(
        settings, HashEmbedder(DIM), FixedSummarizer("Готовая выжимка."),
        classifier=None,
    )
    specs = {spec.name: spec for spec in build_job_specs(worker, settings)}
    assert asyncio.run(specs["summary"].process()) >= 1
    assert len(_versions(note_id)) == 1
