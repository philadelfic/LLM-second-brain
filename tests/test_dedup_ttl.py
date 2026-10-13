"""Заметка с таймером (TTL) никогда не сливается (no46-03, релиз 3.2.0).

Правило О. (2026-10-10): заметка с `expires_at IS NOT NULL`

1. не участвует в дедупе ни как новая, ни как кандидат — судья для неё не
   вызывается, `merge_pair` по ней не проходит;
2. при физическом удалении её связи не остаются (см. test_expiration_cleanup).

Здесь проверяются четыре точки:

* `DeduplicationService.find_candidates`/`find_by_text` исключают TTL-заметку
  SQL-стороной (и KNN-ветка, и текстовый фоллбек);
* свежая заметка с TTL не запускает проверку кандидатов: ранний выход
  `process_judge_pending` до судьи, событие `dedup_skipped` с причиной
  `reason="note has a lifetime"`, модели не вызываются;
* второй барьер `process_merge_pending`: если у любой заметки пары есть TTL —
  слияния нет (обе заметки целы, суммаризатор не вызван);
* регресс: пара без TTL сливается как раньше.

Связи (`links`) TTL-заметке разрешены — это отдельный тест прочистки при
физическом удалении (tests/test_expiration_cleanup.py).
"""

from __future__ import annotations

import json
import logging

import pytest
from fakes import (
    FailingEmbedder,
    FixedSummarizer,
    HashEmbedder,
    RecordingDedup,
    ScriptedJudge,
)

from app.config import get_settings
from app.services.dedup import DeduplicationService
from app.services.notes import NoteService
from app.services.worker import BackgroundWorker
from app.storage import vectors
from app.storage.db import init_db, session, transaction

E1 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


@pytest.fixture
def dim8(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "notes.db"))
    monkeypatch.setenv("EMBEDDING_DIM", "8")
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    return settings


def _set_vector(settings, note_id: int, vector: list[float]) -> None:
    """Записать готовый полный вектор заметки (юнит слоя дедупа)."""
    with session(settings) as conn, transaction(conn):
        vectors.upsert(conn, note_id, vector)


def test_ttl_note_is_not_a_candidate(dim8) -> None:
    """TTL-заметка не кандидат: ни KNN-ветка, ни текстовый фоллбек её не видят."""
    notes = NoteService(dim8, HashEmbedder(8), DeduplicationService(dim8))
    ttl_text = "заметка с таймером и своим уникальным текстом"
    ttl_id = notes.save(ttl_text, expires_at="1d")["id"]
    _set_vector(dim8, ttl_id, E1)
    dedup = DeduplicationService(dim8)
    assert dedup.find_candidates(E1) == []       # SQL-сторона: TTL не кандидат
    assert dedup.find_by_text(ttl_text) is None  # и в дословном фоллбеке
    # Заметка без таймера кандидатом быть не перестала (регресс).
    plain_text = "обычная заметка без таймера"
    plain_id = notes.save(plain_text)["id"]
    _set_vector(dim8, plain_id, E1)
    assert [note_id for note_id, _ in dedup.find_candidates(E1)] == [plain_id]
    found = dedup.find_by_text(plain_text)
    assert found is not None and found["id"] == plain_id


def test_ttl_note_skips_judge(dim8, caplog) -> None:
    """Свежая заметка с TTL: ранний выход до судьи, событие dedup_skipped.

    Судья (ScriptedJudge) намеренно отвечает «ДУБЛЬ» — без раннего выхода
    он был бы вызван и создал бы merge-работу. Проверяем, что модели не
    дёргались и кандидаты для TTL-заметки не искались.
    """
    notes = NoteService(dim8, FailingEmbedder())  # save → pending, без векторов
    notes.save("ранняя заметка без таймера")
    ttl = notes.save("поздняя заметка с таймером", expires_at="1d")
    judge = ScriptedJudge([True])
    dedup = RecordingDedup(candidates=[(1, 0.95)])
    worker = BackgroundWorker(dim8, HashEmbedder(8), dedup=dedup, judge=judge)
    assert worker.process_pending() == 2          # векторизация → judge-работы
    with caplog.at_level(logging.INFO, logger="app"):
        assert worker.process_judge_pending() == 2  # обе работы обработаны
    assert judge.judge_calls == []                # судью не спросили ни разу
    # Кандидатов искали только для заметки без таймера: у TTL — ранний выход.
    assert [call[0] for call in dedup.calls] == [1]
    with session(dim8) as conn:
        merge_jobs = conn.execute(
            "SELECT COUNT(*) FROM worker_jobs "
            "WHERE slot = 'summary' AND kind = 'merge'"
        ).fetchone()[0]
    assert merge_jobs == 0                        # слияние не запускалось
    events = [
        record
        for record in caplog.records
        if getattr(record, "event", None) == "dedup_skipped"
    ]
    assert [(event.note_id, event.reason) for event in events] == [
        (ttl["id"], "note has a lifetime")
    ]


@pytest.mark.parametrize("ttl_on", ["older", "newer"])
def test_merge_refused_for_ttl_note(dim8, caplog, ttl_on) -> None:
    """Второй барьер: у любой заметки пары TTL → merge_pair не выполняется.

    Merge-работа ставится напрямую (как это делала бы judge-петля): проверяем
    именно барьер пути слияния, а не ранний выход judge-петли.
    """
    notes = NoteService(dim8, FailingEmbedder())
    older = notes.save("ранняя заметка пары")
    newer = notes.save("поздняя заметка пары")
    target = older["id"] if ttl_on == "older" else newer["id"]
    with session(dim8) as conn, transaction(conn):
        # Сентинел TTL на одной из сторон пары (абсолютный ISO-8601 UTC).
        conn.execute(
            "UPDATE notes SET expires_at = '2999-01-01T00:00:00Z' WHERE id = ?",
            (target,),
        )
    summarizer = FixedSummarizer(merged="слитый текст")
    worker = BackgroundWorker(dim8, HashEmbedder(8), summarizer)
    worker._create_job(
        "summary",
        "merge",
        newer["id"],
        payload=json.dumps({"older_id": older["id"], "cosine": 0.95}),
    )
    with caplog.at_level(logging.INFO, logger="app"):
        assert worker.process_merge_pending() == 1  # работа снята
    assert summarizer.merge_calls == []             # суммаризатор не звали
    with session(dim8) as conn:
        rows = conn.execute(
            "SELECT id, text, deleted_at FROM notes ORDER BY id"
        ).fetchall()
    assert rows[0]["text"] == "ранняя заметка пары"   # ранняя не изменена
    assert rows[1]["text"] == "поздняя заметка пары"  # поздняя не удалена
    assert rows[0]["deleted_at"] is None and rows[1]["deleted_at"] is None
    events = [
        record
        for record in caplog.records
        if getattr(record, "event", None) == "dedup_skipped"
    ]
    assert len(events) == 1
    assert events[0].reason == "note has a lifetime"
    assert (events[0].older_id, events[0].note_id) == (older["id"], newer["id"])


def test_plain_notes_still_merge(dim8) -> None:
    """Регресс: пара без таймера сливается как раньше (барьер не срабатывает)."""
    notes = NoteService(dim8, FailingEmbedder())
    older = notes.save("ранняя заметка про бэкапы")
    notes.save("поздняя заметка про деплои")
    summarizer = FixedSummarizer(merged="слитый текст")
    worker = BackgroundWorker(
        dim8,
        HashEmbedder(8),
        summarizer,
        RecordingDedup(candidates=[(older["id"], 0.95)]),
    )
    assert worker.process_pending() == 2
    assert worker.process_judge_pending() == 2
    assert worker.process_merge_pending() == 1
    with session(dim8) as conn:
        rows = conn.execute(
            "SELECT id, text, deleted_at FROM notes ORDER BY id"
        ).fetchall()
    assert rows[0]["text"] == "слитый текст"       # ранняя обновлена
    assert rows[0]["deleted_at"] is None
    assert rows[1]["deleted_at"] is not None       # поздняя в trash
