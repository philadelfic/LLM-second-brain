"""Фоновый дедуп в воркере (Фаза 8, Этап 2.1): кандидаты после векторизации.

process_pending после довекторизации каждой заметки ищет косинус-кандидатов
(find_candidates) против ранних активных заметок — точка подключения сведения
(Этап 2.2) и судьи-LLM (Этап 3). Здесь проверяем: факт вызова для каждой
довекторизованной (RecordingDedup), фильтр «только id < note_id» (пара
обрабатывается единожды — защита от зацикливания), пропуск дедупа при отказе
векторизации и связку с реальным DeduplicationService (критерий приёмки
Этапа 2.1: после довекторизации перефраз дал кандидата).

Этап 2.2 (фоллбек без судьи): приговор «дубль» по косинусу
DEDUP_SIMILARITY — merge-промпт суммаризатором (FixedSummarizer/
MergeFailingSummarizer), ранний дубликат обновляется объединённым текстом
(ре-векторизация/ре-суммаризация штатно), поздний — soft delete; отказ
слияния оставляет обе заметки, свежая возвращается в pending_vector
(NFR-3).

Этап 3.2 (судья): косинус — лишь предфильтр, финальный вердикт — за
JudgeService (ScriptedJudge): каждого живого кандидата спрашивают про пару
текстов (text_new = свежая, text_candidate = ранняя), сводится первый
признанный «ДУБЛЬ»; «НЕ ДУБЛЬ» всем — обе живы; отказ судьи (JudgeError)
— requeue свежей в pending_vector, повтор по back-off (NFR-3).
"""

from __future__ import annotations

import logging

import pytest
from fakes import (
    FailingEmbedder,
    FixedSummarizer,
    MergeFailingSummarizer,
    HashEmbedder,
    RecordingDedup,
    ScriptedJudge,
    cosine,
)

from app.config import get_settings
from app.services.dedup import DeduplicationService
from app.services.notes import NoteService
from app.services.worker import BackgroundWorker
from app.storage.db import init_db, session


@pytest.fixture
def dim8(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "notes.db"))
    monkeypatch.setenv("EMBEDDING_DIM", "8")
    monkeypatch.setenv("PENDING_RETRY_SEC", "0")
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    return settings


def _dedup_records(caplog) -> list:
    """Лог-записи с event=dedup_candidates (поиск кандидатов в process_pending)."""
    return [
        record
        for record in caplog.records
        if getattr(record, "event", None) == "dedup_candidates"
    ]


def test_process_pending_searches_candidates_for_each_note(dim8) -> None:
    """Каждая довекторизованная заметка ищет кандидатов с exclude_id = её id."""
    notes = NoteService(dim8, FailingEmbedder())  # save → pending, без векторов
    texts = ["первая отложенная заметка", "вторая отложенная заметка"]
    for text in texts:
        notes.save(text)
    dedup = RecordingDedup()  # кандидатов не даёт — проверяем только вызовы
    worker = BackgroundWorker(dim8, HashEmbedder(8), dedup=dedup)
    assert worker.process_pending() == 2
    assert [call[0] for call in dedup.calls] == [1, 2]
    for exclude_id, vector, _namespace in dedup.calls:
        expected = HashEmbedder(8).embed(texts[exclude_id - 1])
        assert vector == pytest.approx(expected, abs=1e-6)


def test_process_pending_keeps_only_older_candidates(dim8, caplog) -> None:
    """Пара «поздняя ↔ ранняя» обрабатывается единожды: id ≥ note_id отрезаны.

    Фейк возвращает обе заметки-«кандидата» для каждого вызова: при обработке
    id=1 остаётся пусто (оба не раньше), при id=2 — только ранняя (id=1).
    Найденный кандидат логируется (event dedup_candidates) — наблюдаемость
    предфильтра и точка подключения сведения (Этап 2.2).
    """
    notes = NoteService(dim8, FailingEmbedder())
    notes.save("ранняя заметка про бэкапы")
    notes.save("поздняя заметка про деплои")
    dedup = RecordingDedup(candidates=[(1, 0.95), (2, 0.95)])
    worker = BackgroundWorker(dim8, HashEmbedder(8), dedup=dedup)
    with caplog.at_level(logging.INFO, logger="app"):
        assert worker.process_pending() == 2
    records = _dedup_records(caplog)
    assert len(records) == 1
    assert records[0].note_id == 2
    assert records[0].candidates == [(1, 0.95)]


def test_embedding_failure_skips_dedup(dim8) -> None:
    """Отказ кодирования — 0 векторов, до дедупа не доходим (нечего сравнивать)."""
    notes = NoteService(dim8, FailingEmbedder())
    notes.save("заметка без векторизатора")
    dedup = RecordingDedup(candidates=[(0, 0.95)])
    worker = BackgroundWorker(dim8, FailingEmbedder(), dedup=dedup)
    assert worker.process_pending() == 0
    assert dedup.calls == []


def test_worker_builds_dedup_by_default(dim8) -> None:
    """Без DI воркер сам строит реальный DeduplicationService (связка жива)."""
    worker = BackgroundWorker(dim8, HashEmbedder(8))
    assert isinstance(worker._dedup, DeduplicationService)


def test_vectorized_paraphrase_yields_candidates(dim8, caplog) -> None:
    """Критерий приёмки 2.1: после довекторизации перефраз дал кандидата.

    Обе заметки pending → воркер довекторизует партией → у поздней кандидатом
    числится ранняя (cos ≈ 0.94 > DEDUP_CANDIDATE_SIMILARITY 0.80; что обе
    сохранились, а не отсечены, — следствие того, что косинус-дедуп в save
    исчез: решает фоновый конвейер).
    """
    embedder = HashEmbedder(8)
    texts = ["первая отложенная заметка", "вторая отложенная заметка"]
    notes = NoteService(dim8, FailingEmbedder())  # save мгновенный
    first = notes.save(texts[0])
    second = notes.save(texts[1])
    assert "duplicated" not in first and "duplicated" not in second
    # Предпосылка: перефраз выше кандидата-порога.
    pair_cos = cosine(embedder.embed(texts[0]), embedder.embed(texts[1]))
    assert pair_cos >= dim8.dedup_candidate_similarity
    worker = BackgroundWorker(dim8, embedder)
    with caplog.at_level(logging.INFO, logger="app"):
        assert worker.process_pending() == 2
    records = _dedup_records(caplog)
    assert len(records) == 1
    assert records[0].note_id == second["id"]
    assert records[0].candidates == [
        (first["id"], pytest.approx(pair_cos, abs=1e-4))
    ]


# --- сведение дублей (Этап 2.2): merge-промпт, ранний обновлён, поздний в trash


def test_merge_updates_earlier_and_soft_deletes_later(dim8, caplog) -> None:
    """Критерий приёмки 2.2: пара-перефраз (косинус реального дедупа
    0.94 > DEDUP_SIMILARITY 0.92) сводится: merge(sum(ранняя, поздняя))
    уходит в раннюю (pending на ре-векторизацию/ре-суммаризацию),
    поздняя — в trash; повторный прогон не сливает пару второй раз."""
    texts = ["первая отложенная заметка", "вторая отложенная заметка"]
    notes = NoteService(dim8, FailingEmbedder())  # save мгновенный
    notes.save(texts[0])
    notes.save(texts[1])
    merged_text = "Объединённая встреча: 12 сентября, 14:00, Браво."
    summarizer = FixedSummarizer("Фикс-суммари.", merged=merged_text)
    worker = BackgroundWorker(dim8, HashEmbedder(8), summarizer)
    with caplog.at_level(logging.INFO, logger="app"):
        assert worker.process_pending() == 2  # обе довекторизованы, сведение
    # merge вызван ровно один раз, порядок аргументов: (ранняя, поздняя)
    assert summarizer.merge_calls == [(texts[0], texts[1])]
    merged_log = [
        record
        for record in caplog.records
        if getattr(record, "event", None) == "dedup_merged"
    ]
    assert len(merged_log) == 1
    assert merged_log[0].older_id == 1 and merged_log[0].note_id == 2
    with session(dim8) as conn:
        rows = conn.execute(
            "SELECT id, text, vector_status, summary_status, deleted_at "
            "FROM notes ORDER BY id"
        ).fetchall()
    assert rows[0]["text"] == merged_text  # ранний дубликат обновлён
    assert rows[0]["vector_status"] == "pending"  # ре-векторизация — фон
    assert rows[0]["summary_status"] == "pending"  # ре-суммаризация — фон
    assert rows[0]["deleted_at"] is None
    assert rows[1]["deleted_at"] is not None  # поздняя — soft delete
    # воркер догоняет обновлённую раннюю, второй пары нет (нет зацикливания)
    assert worker.process_pending() == 1
    assert worker.process_summary_pending() == 1
    with session(dim8) as conn:
        row = conn.execute(
            "SELECT text, vector_status, summary, summary_status, deleted_at "
            "FROM notes WHERE id = 1"
        ).fetchone()
    assert row["text"] == merged_text
    assert row["vector_status"] == "ok" and row["summary_status"] == "ok"
    assert row["summary"] == "Фикс-суммари." and row["deleted_at"] is None
    assert len(summarizer.merge_calls) == 1  # сведение не повторилось


def test_merge_failure_keeps_both_and_retries(dim8, caplog) -> None:
    """Отказ слияния (NFR-3): обе заметки живы, свежая вернулась в
    pending_vector (не «processed» — back-off очереди держится); после
    восстановления суммаризатора повтор завершает сведение."""
    texts = ["первая отложенная заметка", "вторая отложенная заметка"]
    notes = NoteService(dim8, FailingEmbedder())
    notes.save(texts[0])
    notes.save(texts[1])
    summarizer = MergeFailingSummarizer()  # merge отказывает, пока fail
    worker = BackgroundWorker(dim8, HashEmbedder(8), summarizer)
    with caplog.at_level(logging.WARNING, logger="app"):
        assert worker.process_pending() == 1  # id=2 довекторизована без учёта
    assert len(summarizer.merge_calls) == 1  # попытка слияния была
    warnings = [
        record
        for record in caplog.records
        if getattr(record, "event", None) == "dedup_merge_failed"
    ]
    assert len(warnings) == 1 and warnings[0].older_id == 1
    with session(dim8) as conn:
        rows = conn.execute(
            "SELECT id, text, vector_status, deleted_at FROM notes ORDER BY id"
        ).fetchall()
    assert rows[0]["vector_status"] == "ok"  # ранняя не тронута
    assert rows[1]["vector_status"] == "pending"  # свежая — на повтор
    assert all(row["deleted_at"] is None for row in rows)  # обе живы (NFR-3)
    # «чиним» суммаризатор — повтор из очереди доводит сведение до конца
    summarizer.fail = False
    assert worker.process_pending() == 1
    with session(dim8) as conn:
        rows = conn.execute(
            "SELECT id, text, vector_status, deleted_at FROM notes ORDER BY id"
        ).fetchall()
    assert rows[0]["text"] == summarizer.merge_result  # ранняя слита
    assert rows[0]["vector_status"] == "pending"  # и в очереди на догонку
    assert rows[1]["deleted_at"] is not None  # поздняя — в trash
    assert len(summarizer.merge_calls) == 2  # отказ + успешный повтор


def test_candidate_below_fallback_threshold_keeps_pair(dim8) -> None:
    """Фоллбек без судьи (DI None — тестовый режим): кандидат в зоне
    [DEDUP_CANDIDATE_SIMILARITY, DEDUP_SIMILARITY) — не приговор: слияния
    нет, обе заметки остаются со статусом ok (с судьёй этот кандидат
    разбирается по вердикту — тесты Этапа 3.2 ниже)."""
    notes = NoteService(dim8, FailingEmbedder())
    notes.save("ранняя заметка про бэкапы")
    notes.save("поздняя заметка про деплои")
    summarizer = FixedSummarizer("Фикс-суммари.", merged="слитый текст")
    worker = BackgroundWorker(
        dim8, HashEmbedder(8), summarizer, RecordingDedup(candidates=[(1, 0.85)])
    )
    assert worker.process_pending() == 2
    assert summarizer.merge_calls == []  # 0.80 ≤ 0.85 < 0.92 — не дубль
    with session(dim8) as conn:
        rows = conn.execute(
            "SELECT id, vector_status, deleted_at FROM notes ORDER BY id"
        ).fetchall()
    assert [(row["vector_status"], row["deleted_at"]) for row in rows] == [
        ("ok", None),
        ("ok", None),
    ]  # без requeue: векторизованный статус устоял


def test_merge_verdict_uses_dedup_similarity_env(tmp_path, monkeypatch) -> None:
    """Порог вердикта — DEDUP_SIMILARITY (env): 0.99 прижимает кандидата
    0.95 в зону ожидания судьи — слияния нет."""
    for name, value in {
        "DB_PATH": str(tmp_path / "notes.db"),
        "EMBEDDING_DIM": "8",
        "PENDING_RETRY_SEC": "0",
        "DEDUP_SIMILARITY": "0.99",
    }.items():
        monkeypatch.setenv(name, value)
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    notes = NoteService(settings, FailingEmbedder())
    notes.save("ранняя заметка про бэкапы")
    notes.save("поздняя заметка про деплои")
    summarizer = FixedSummarizer("Фикс-суммари.", merged="слитый текст")
    worker = BackgroundWorker(
        settings, HashEmbedder(8), summarizer, RecordingDedup(candidates=[(1, 0.95)])
    )
    assert worker.process_pending() == 2
    assert summarizer.merge_calls == []  # 0.92 < 0.95 < 0.99 — зона суда
    get_settings.cache_clear()


def test_merge_skips_when_candidate_vanishes(dim8) -> None:
    """Кандидат ушёл в trash до слияния — пара протухла, сведение отменяется
    без потери данных (свежая остаётся обработанной)."""
    notes = NoteService(dim8, FailingEmbedder())
    notes.save("ранняя заметка про бэкапы")
    notes.save("поздняя заметка про деплои")
    notes.delete(1)  # ранняя стала trash — до векторизации партии
    summarizer = FixedSummarizer("Фикс-суммари.", merged="слитый текст")
    worker = BackgroundWorker(
        dim8, HashEmbedder(8), summarizer, RecordingDedup(candidates=[(1, 0.95)])
    )
    assert worker.process_pending() == 1  # trash не векторизуется
    assert summarizer.merge_calls == []  # сведение отменено
    with session(dim8) as conn:
        row = conn.execute(
            "SELECT vector_status, deleted_at FROM notes WHERE id = 2"
        ).fetchone()
    assert row["vector_status"] == "ok" and row["deleted_at"] is None


def test_candidates_without_summarizer_stay_unmerged(dim8) -> None:
    """Без суммаризатора (тестовый режим Фазы 3) сведение невозможно:
    пара остаётся, заметка считается обработанной — без холостого requeue
    (иначе цикл пере-кодировок без суммаризатора не завершится)."""
    texts = ["первая отложенная заметка", "вторая отложенная заметка"]
    notes = NoteService(dim8, FailingEmbedder())
    notes.save(texts[0])
    notes.save(texts[1])
    worker = BackgroundWorker(
        dim8, HashEmbedder(8), None, RecordingDedup(candidates=[(1, 0.95)])
    )
    assert worker.process_pending() == 2  # processed не режется: requeue нет
    with session(dim8) as conn:
        rows = conn.execute(
            "SELECT id, vector_status, deleted_at FROM notes ORDER BY id"
        ).fetchall()
    assert [(row["vector_status"], row["deleted_at"]) for row in rows] == [
        ("ok", None),
        ("ok", None),
    ]


# --- судья дедупа (Этап 3.2): вердикт LLM поверх косинус-предфильтра --------


def test_judge_merges_paraphrase_below_cosine_threshold(dim8, caplog) -> None:
    """Критерий приёмки 3.2: перефраз, которого порог «дубля» 0.92 не ловит
    (кандидат с cosine 0.85 в зоне [0.80, 0.92)), судья признаёт дублем —
    сведение: merge в раннюю (ре-векторизация/ре-суммаризация штатно),
    поздняя — trash; логи вердикта и сведения содержат cosine кандидата."""
    texts = ["первая отложенная заметка", "вторая отложенная заметка"]
    notes = NoteService(dim8, FailingEmbedder())  # save мгновенный
    notes.save(texts[0])
    notes.save(texts[1])
    summarizer = FixedSummarizer("Фикс-суммари.", merged="слитый текст")
    judge = ScriptedJudge([True])
    worker = BackgroundWorker(
        dim8,
        HashEmbedder(8),
        summarizer,
        RecordingDedup(candidates=[(1, 0.85)]),
        judge=judge,
    )
    with caplog.at_level(logging.INFO, logger="app"):
        assert worker.process_pending() == 2
    # Судья опрошен парой текстов: text_new = свежая, text_candidate = ранняя.
    assert judge.judge_calls == [(texts[1], texts[0])]
    verdict_log = [
        record
        for record in caplog.records
        if getattr(record, "event", None) == "dedup_judge"
    ]
    assert len(verdict_log) == 1
    assert verdict_log[0].candidate_id == 1
    assert verdict_log[0].cosine == pytest.approx(0.85)
    assert verdict_log[0].verdict is True
    # merge вызвался с (ранняя, поздняя), сведение в логе с cosine 0.85.
    assert summarizer.merge_calls == [(texts[0], texts[1])]
    merged_log = [
        record
        for record in caplog.records
        if getattr(record, "event", None) == "dedup_merged"
    ]
    assert len(merged_log) == 1
    assert merged_log[0].older_id == 1 and merged_log[0].note_id == 2
    assert merged_log[0].cosine == pytest.approx(0.85)
    with session(dim8) as conn:
        rows = conn.execute(
            "SELECT id, text, vector_status, deleted_at FROM notes ORDER BY id"
        ).fetchall()
    assert rows[0]["text"] == "слитый текст"  # ранняя — объединённый текст
    assert rows[0]["vector_status"] == "pending"  # ре-векторизация — фон
    assert rows[0]["deleted_at"] is None
    assert rows[1]["deleted_at"] is not None  # поздняя — soft delete


def test_judge_verdict_keeps_distinct_notes(dim8) -> None:
    """Критерий приёмки 3.2: разные заметки судьёй не сводятся — «НЕ ДУБЛЬ»:
    merge не зовётся, обе живы с 'ok'. Случай проверяет «финальное решение —
    судья, а не косинус»: 0.94 ≥ DEDUP_SIMILARITY — косинус-фоллбек слил бы,
    судья удержал."""
    texts = ["ранняя заметка про бэкапы", "поздняя заметка про деплои"]
    notes = NoteService(dim8, FailingEmbedder())
    notes.save(texts[0])
    notes.save(texts[1])
    summarizer = FixedSummarizer("Фикс-суммари.", merged="слитый текст")
    judge = ScriptedJudge([False])
    worker = BackgroundWorker(
        dim8,
        HashEmbedder(8),
        summarizer,
        RecordingDedup(candidates=[(1, 0.94)]),
        judge=judge,
    )
    assert worker.process_pending() == 2
    assert judge.judge_calls == [(texts[1], texts[0])]
    assert summarizer.merge_calls == []  # вердикт «НЕ ДУБЛЬ» — сведение нет
    with session(dim8) as conn:
        rows = conn.execute(
            "SELECT id, vector_status, deleted_at FROM notes ORDER BY id"
        ).fetchall()
    assert [(row["vector_status"], row["deleted_at"]) for row in rows] == [
        ("ok", None),
        ("ok", None),
    ]  # статусы устояли — без requeue (текст не менялся)


def test_judge_failure_keeps_both_and_retries(dim8, caplog) -> None:
    """Отказ судьи (JudgeError, NFR-3): обе заметки живы, свежая вернулась
    в pending_vector (не «processed» — back-off очереди держится); после
    восстановления судьи повтор из очереди завершает сведение."""
    texts = ["первая отложенная заметка", "вторая отложенная заметка"]
    notes = NoteService(dim8, FailingEmbedder())
    notes.save(texts[0])
    notes.save(texts[1])
    summarizer = FixedSummarizer("Фикс-суммари.", merged="слитый текст")
    judge = ScriptedJudge([True], fail=True)
    worker = BackgroundWorker(
        dim8,
        HashEmbedder(8),
        summarizer,
        RecordingDedup(candidates=[(1, 0.94)]),
        judge=judge,
    )
    with caplog.at_level(logging.WARNING, logger="app"):
        assert worker.process_pending() == 1  # id=2 довекторизована, но requeue
    assert len(judge.judge_calls) == 1  # вопрос был задан до отказа
    warnings = [
        record
        for record in caplog.records
        if getattr(record, "event", None) == "dedup_judge_failed"
    ]
    assert len(warnings) == 1 and warnings[0].candidates == [1]
    assert summarizer.merge_calls == []  # до слияния судья не довёл
    with session(dim8) as conn:
        rows = conn.execute(
            "SELECT id, text, vector_status, deleted_at FROM notes ORDER BY id"
        ).fetchall()
    assert rows[0]["vector_status"] == "ok"  # ранняя не тронута
    assert rows[1]["vector_status"] == "pending"  # свежая — на повтор
    assert all(row["deleted_at"] is None for row in rows)  # обе живы (NFR-3)
    # «Чиним» судью — повтор из очереди доводит дедуп до сведения.
    judge.fail = False
    assert worker.process_pending() == 1
    with session(dim8) as conn:
        rows = conn.execute(
            "SELECT id, text, vector_status, deleted_at FROM notes ORDER BY id"
        ).fetchall()
    assert rows[0]["text"] == summarizer.merge_result  # ранняя слита
    assert rows[0]["vector_status"] == "pending"  # и в очереди на догонку
    assert rows[1]["deleted_at"] is not None  # поздняя — в trash
    assert len(judge.judge_calls) == 2  # отказ + вердикт на повторе


def test_judge_picks_second_candidate_after_first_rejects(dim8) -> None:
    """Кандидаты опрашиваются по убыванию косинуса; «НЕ ДУБЛЬ» первому не
    останавливает опрос — сводится первый признанный (тут — второй)."""
    texts = [
        "ранняя заметка про бэкапы",
        "близкая заметка про бэкапы",
        "свежая заметка про бэкапы снова",
    ]
    notes = NoteService(dim8, FailingEmbedder())
    for text in texts:
        notes.save(text)
    summarizer = FixedSummarizer("Фикс-суммари.", merged="слитый текст")
    # Вызовы идут в порядке обработки: id=2 (против 1), затем id=3
    # (против 1, потом против 2) — очередь точно им соответствует.
    judge = ScriptedJudge([False, False, True])
    worker = BackgroundWorker(
        dim8,
        HashEmbedder(8),
        summarizer,
        RecordingDedup(candidates=[(1, 0.87), (2, 0.81)]),
        judge=judge,
    )
    assert worker.process_pending() == 3
    assert judge.judge_calls == [
        (texts[1], texts[0]),  # id=2 против id=1: «НЕ ДУБЛЬ»
        (texts[2], texts[0]),  # id=3 против id=1: «НЕ ДУБЛЬ»
        (texts[2], texts[1]),  # id=3 против id=2: «ДУБЛЬ»
    ]
    # Сведение с первым признанным (id=2): merge(ранняя, поздняя).
    assert summarizer.merge_calls == [(texts[1], texts[2])]
    with session(dim8) as conn:
        rows = conn.execute(
            "SELECT id, text, vector_status, deleted_at FROM notes ORDER BY id"
        ).fetchall()
    assert rows[0]["text"] == texts[0] and rows[0]["deleted_at"] is None
    assert rows[1]["text"] == "слитый текст"  # ранняя (id=2) обновлена
    assert rows[1]["vector_status"] == "pending"  # ре-векторизация — фон
    assert rows[2]["deleted_at"] is not None  # свежая (id=3) — trash
