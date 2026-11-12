"""Регресс «шторм» на нарезанном источнике (LSBN-T-35, G1; LSBN-D-14).

Производство 3.6.1: одна длинная заметка нарезана на N частей с общим
`origin_note_id`. Части близки по вектору ПО ПОСТРОЕНИЮ, поэтому фоновый дедуп
считал их дублями друг друга: судья подтверждал «дубль», merge-путь шёл в
модель, ответы модели не ложились в JSON-контракт (`off_contract`), слот summary
уходил в паузу, пары возвращались в очередь — шторм merge-попыток.

Здесь проверяется весь конвейер фона ТРИ цикла подряд («пачка → пауза →
следующая пачка») на 12 частях одного исходника при худшем судье (отвечает
«ДУБЛЬ» на любой вопрос): ни одного вопроса судье, ни одного вызова merge, ни
одной merge-работы, ни одной паузы слота, attention = 0, все части живы и их
текст не изменён. Без правила own-origin (3.6.1) тест красный: части становятся
кандидатами, судья говорит «дубль», merge-путь переписывает части.
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
from app.services.worker import BackgroundWorker, _embed_input
from app.storage.db import init_db, session

ORIGIN_ID = 999  # исходник нарезки: у всех частей один и тот же
PARTS = 12  # размер группы: в бою встречались до 181 части
CYCLES = 3  # «пачка → пауза → следующая пачка»

#: Общее тело частей: чем длиннее общий текст, тем ближе векторы частей
#: (именно так и возникает шторм на боевом объёме).
BODY = (
    "Ежедневный бэкап кластера: проверяем снимок и журнал репликации, сверяем "
    "размеры с предыдущим днём, чиним расхождения, заносим результат в журнал "
    "дежурств и уведомляем ответственного инженера. Проверка идёт утром и "
    "вечером, расхождения фиксируются отдельной записью."
)


def _part(index: int) -> tuple[str, str]:
    """Часть нарезки: самодостаточный текст с общим контекстом источника."""
    return (
        f"Бэкап кластера, часть {index + 1}",
        f"Часть {index + 1}. Порядок действий для этой части: {BODY}",
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


def _events(caplog, event: str) -> list:
    return [
        record
        for record in caplog.records
        if getattr(record, "event", None) == event
    ]


def _off_contract_rows(settings) -> int:
    """Работы, последний исход которых — off_contract (счётчик шторма)."""
    with session(settings) as conn:
        return int(
            conn.execute(
                "SELECT COUNT(*) FROM worker_jobs "
                "WHERE last_outcome = 'off_contract'"
            ).fetchone()[0]
        )


def _merge_jobs(settings) -> int:
    with session(settings) as conn:
        return int(
            conn.execute(
                "SELECT COUNT(*) FROM worker_jobs WHERE kind = 'merge'"
            ).fetchone()[0]
        )


def _notes_state(settings) -> list[dict]:
    with session(settings) as conn:
        rows = conn.execute(
            "SELECT id, text, origin_note_id, vector_status, deleted_at "
            "FROM notes ORDER BY id"
        ).fetchall()
    return [dict(row) for row in rows]


def test_storm_on_a_split_source_is_not_reproduced(dim8, caplog) -> None:
    """12 частей одного исходника, три цикла фона, судья всегда «ДУБЛЬ»."""
    embedder = HashEmbedder(8)
    parts = [_part(index) for index in range(PARTS)]
    # Предпосылка шторма: части близки настолько, что без правила own-origin
    # каждая была бы кандидатом каждой (иначе тест зелёный по неверной причине).
    vectors = [embedder.embed(_embed_input(title, text)) for title, text in parts]
    worst = min(
        cosine(first, second)
        for number, first in enumerate(vectors)
        for second in vectors[number + 1 :]
    )
    assert worst >= dim8.dedup_candidate_similarity

    notes = NoteService(dim8, FailingEmbedder())
    saved = notes.save_migration_parts(ORIGIN_ID, "default", parts)
    part_ids = [int(part["id"]) for part in saved["parts"]]
    assert len(part_ids) == PARTS

    judge = ScriptedJudge(default=True)  # любой вопрос — «ДУБЛЬ» (худший случай)
    summarizer = FixedSummarizer("Фикс-суммари.", merged="слитый текст")
    worker = BackgroundWorker(dim8, embedder, summarizer, judge=judge)

    with caplog.at_level(logging.INFO, logger="app"):
        for cycle in range(CYCLES):
            # «пачка»: векторы → судья → merge-путь; следующая итерация — новая
            # пачка (так фон и живёт: одна пачка на итерацию, между ними пауза).
            worker.process_pending()
            worker.process_judge_pending()
            worker.process_merge_pending()
            assert worker.attention_count() == 0, f"attention после цикла {cycle}"

    # Модель не спрошена ни разу: ни судья, ни merge.
    assert judge.judge_calls == []
    assert summarizer.merge_calls == []
    # Merge-работ не создано — очередь summary не растёт из этой группы.
    assert _merge_jobs(dim8) == 0
    # Счётчик «не по контракту» и паузы слота summary не растут.
    assert _off_contract_rows(dim8) == 0
    assert _events(caplog, "slot_paused") == []
    assert _events(caplog, "dedup_judge_failed") == []
    assert _events(caplog, "dedup_merge_failed") == []
    health = worker.slots_health()["summary"]
    assert health["paused_until"] is None
    assert health["attention"] == 0
    assert health["ok"] is None  # merge-путь ни разу не трогал слот
    queues = worker.queues_health()
    assert queues["judge"]["pending"] == 0  # judge-очередь разобрана
    assert queues["judge"]["attention"] == 0
    assert queues["summary"]["attention"] == 0

    # Все части живы, текст не изменён, группа сохранена, векторы готовы.
    state = _notes_state(dim8)
    assert [row["id"] for row in state] == part_ids
    assert [row["text"] for row in state] == [text for _, text in parts]
    assert {row["origin_note_id"] for row in state} == {ORIGIN_ID}
    assert all(row["deleted_at"] is None for row in state)
    assert all(row["vector_status"] == "ok" for row in state)
