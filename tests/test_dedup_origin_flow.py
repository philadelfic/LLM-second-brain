"""Приёмка LSBN-D-14 (3.6.2): части одного исходника не дубли.

Нарезка длинной заметки миграции рождает N самодостаточных частей с общим
`origin_note_id` (`NoteService.save_migration_parts`). Их векторы близки по
построению, поэтому фоновый дедуп сводил части обратно и терял содержимое.
Поток целиком, без моделей (HashEmbedder + ScriptedJudge): довекторизация
частей → judge-петля с РЕАЛЬНЫМ DeduplicationService → предфильтр отбрасывает
кандидатов своей origin-группы → судья не опрошен ни разу, merge-работ нет,
обе части живы.
"""

from __future__ import annotations

import logging

import pytest
from fakes import (
    FailingEmbedder,
    FixedSummarizer,
    HashEmbedder,
    ScriptedJudge,
    cosine,
)

from app.config import get_settings
from app.services.notes import NoteService
from app.services.worker import BackgroundWorker
from app.storage.db import init_db, session

NEWLINE = chr(10)  # склейка «заголовок + текст» как в воркере (_embed_input)

PART_ONE = (
    "Часть 1: ежедневный бэкап кластера",
    (
        "Ежедневный бэкап кластера: проверяем снимок и журнал репликации утром, "
        "чиним расхождения."
    ),
)
PART_TWO = (
    "Часть 2: ежедневный бэкап кластера",
    (
        "Ежедневный бэкап кластера: проверяем снимок и журнал репликации вечером, "
        "чиним расхождения."
    ),
)


@pytest.fixture
def dim8(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "notes.db"))
    monkeypatch.setenv("EMBEDDING_DIM", "8")
    monkeypatch.setenv("PENDING_RETRY_SEC", "0")
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    return settings


def test_parts_of_one_source_never_reach_the_judge(dim8, caplog) -> None:
    """Нарезка → вектор → судья: ни одного вопроса, ни одной merge-работы."""
    embedder = HashEmbedder(8)
    # Предпосылка: векторы частей близки настолько, что без правила они были бы
    # кандидатами друг другу (иначе тест зелёный по неверной причине).
    first_input = NEWLINE.join(PART_ONE)
    second_input = NEWLINE.join(PART_TWO)
    assert (
        cosine(embedder.embed(first_input), embedder.embed(second_input))
        >= dim8.dedup_candidate_similarity
    )
    notes = NoteService(dim8, FailingEmbedder())
    saved = notes.save_migration_parts(999, "default", [PART_ONE, PART_TWO])
    part_ids = [int(part["id"]) for part in saved["parts"]]
    judge = ScriptedJudge(default=True)  # любой вопрос — «ДУБЛЬ»
    worker = BackgroundWorker(dim8, embedder, FixedSummarizer("С."), judge=judge)
    with caplog.at_level(logging.INFO, logger="app"):
        assert worker.process_pending() == 2  # обе части довекторизованы
        assert worker.process_judge_pending() == 2  # кандидатов нет — вопросов нет
    assert judge.judge_calls == []  # модель не спрошена ни разу
    assert [
        record
        for record in caplog.records
        if getattr(record, "event", None) == "dedup_candidates"
    ] == []
    with session(dim8) as conn:
        merge_jobs = conn.execute(
            "SELECT COUNT(*) FROM worker_jobs WHERE kind = 'merge'"
        ).fetchone()[0]
        rows = conn.execute(
            "SELECT id, origin_note_id, vector_status, deleted_at "
            "FROM notes ORDER BY id"
        ).fetchall()
    assert merge_jobs == 0  # merge-работ не создано
    assert [row["id"] for row in rows] == part_ids
    assert all(row["origin_note_id"] == 999 for row in rows)  # общий исходник
    assert all(row["vector_status"] == "ok" for row in rows)
    assert all(row["deleted_at"] is None for row in rows)  # обе части живы
