"""LSBN-T-37 (3.6.3): короткая заметка — выжимка из своего текста, без модели.

Заметка в 100 знаков не выигрывает от пересказа: выжимка почти дословно
равна самой заметке, а модельный вызов на флинте — 40-75 секунд за штуку.
Правило: текст не длиннее SUMMARY_SHORT_TEXT_CHARS → выжимка берётся из
самого текста тем же `cap_summary` (граница слова, лимит MAX_SUMMARY_CHARS),
summary_status='ok', слот не зовётся, в журнал — событие `summary_shortcircuit`
(note_id, text_chars, summary_chars). 0 — правило выключено, поведение
прежнее: все заметки идут модельным путём.

Проверяется здесь: короткая заметка (модели нет), короткая длиннее лимита
выжимки (тот же cap_summary), длинная (модельный путь на том же фейке),
граница «ровно порог» / «порог+1» и выключенное правило.
"""

from __future__ import annotations

import logging

import pytest
from fakes import FailingEmbedder, FixedSummarizer, HashEmbedder

from app.config import get_settings
from app.services.notes import NoteService
from app.services.summary import cap_summary
from app.services.worker import BackgroundWorker
from app.storage.db import init_db, session

MODEL_ANSWER = "Выжимка модели одним предложением."


def _settings(tmp_path, monkeypatch, **env) -> object:
    """Настройки на tmp-базе; env — переопределения сверх тестовых дефолтов."""
    monkeypatch.setenv("EMBEDDING_DIM", "8")
    monkeypatch.setenv("PENDING_RETRY_SEC", "0")
    for name, value in env.items():
        monkeypatch.setenv(name, str(value))
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    return settings


@pytest.fixture
def settings(tmp_path, monkeypatch):
    """Настройки с дефолтным порогом короткого текста (400)."""
    return _settings(tmp_path, monkeypatch)


def _save(settings, text: str):
    """Заметка в очереди summary (save → summary_status='pending')."""
    notes = NoteService(settings, FailingEmbedder())
    return notes.save(text)["id"], notes


def _row(settings, note_id: int):
    with session(settings) as conn:
        return conn.execute(
            "SELECT summary, summary_status FROM notes WHERE id = ?", (note_id,)
        ).fetchone()


def test_short_note_is_summarized_without_a_model_call(settings, caplog) -> None:
    """Текст не длиннее порога: выжимка = cap_summary(текста), слот не зван."""
    text = (
        "Интеграция офиса: ретроспектива продукта назначена на 12 сентября "
        "2026 года в 14:00, переговорная Браво."
    )
    assert len(text) <= settings.summary_short_text_chars
    note_id, notes = _save(settings, text)
    fake = FixedSummarizer(MODEL_ANSWER)
    worker = BackgroundWorker(settings, HashEmbedder(8), fake)
    with caplog.at_level(logging.INFO, logger="app"):
        assert worker.process_summary_pending() == 1
    assert fake.calls == []  # счётчик вызовов слотов — ноль
    row = _row(settings, note_id)
    assert row["summary_status"] == "ok"
    assert row["summary"] == cap_summary(text, settings.max_summary_chars)
    assert row["summary"] == text  # короче MAX_SUMMARY_CHARS — дословно
    assert row["summary"] != MODEL_ANSWER
    # В выдаче — готовая выжимка, а не fallback-обрезка текста.
    assert notes.get([note_id])["notes"][0]["summary"] == row["summary"]
    # Журнал: по событию на заметку, с id и длиной текста.
    events = [
        rec
        for rec in caplog.records
        if getattr(rec, "event", None) == "summary_shortcircuit"
    ]
    assert len(events) == 1
    assert events[0].note_id == note_id
    assert events[0].text_chars == len(text)
    assert events[0].summary_chars == len(row["summary"])


def test_short_note_longer_than_summary_limit_is_capped(settings) -> None:
    """Короткая (не длиннее порога), но длиннее лимита выжимки — режет тот же cap_summary."""
    text = ("Длинная короткая заметка: " * 8).strip()
    assert settings.max_summary_chars < len(text) <= settings.summary_short_text_chars
    note_id, _ = _save(settings, text)
    fake = FixedSummarizer(MODEL_ANSWER)
    worker = BackgroundWorker(settings, HashEmbedder(8), fake)
    assert worker.process_summary_pending() == 1
    assert fake.calls == []
    row = _row(settings, note_id)
    assert row["summary"] == cap_summary(text, settings.max_summary_chars)
    assert len(row["summary"]) <= settings.max_summary_chars
    assert row["summary"].endswith("…")
    assert row["summary_status"] == "ok"


def test_long_note_takes_the_model_path(settings) -> None:
    """Длиннее порога — обычный модельный путь (тот же фейк, тот же cap)."""
    text = (
        "Заметка длиннее порога: "
        + "подробности встречи, решения и сроки по каждому пункту. " * 12
    ).strip()
    assert len(text) > settings.summary_short_text_chars
    note_id, _ = _save(settings, text)
    fake = FixedSummarizer(MODEL_ANSWER)
    worker = BackgroundWorker(settings, HashEmbedder(8), fake)
    assert worker.process_summary_pending() == 1
    assert fake.calls == [text]  # модель позвана ровно один раз, с этим текстом
    row = _row(settings, note_id)
    assert row["summary"] == MODEL_ANSWER  # ответ короче лимита — cap не режет
    assert row["summary_status"] == "ok"


@pytest.mark.parametrize(("extra", "model_called"), [(0, False), (1, True)])
def test_threshold_boundary(tmp_path, monkeypatch, extra, model_called) -> None:
    """Граница: ровно порог — из текста; порог+1 — модель."""
    settings = _settings(tmp_path, monkeypatch, SUMMARY_SHORT_TEXT_CHARS="60")
    text = ("слово " * 30)[: 60 + extra]
    assert len(text) == 60 + extra
    note_id, _ = _save(settings, text)
    fake = FixedSummarizer(MODEL_ANSWER)
    worker = BackgroundWorker(settings, HashEmbedder(8), fake)
    assert worker.process_summary_pending() == 1
    row = _row(settings, note_id)
    assert row["summary_status"] == "ok"
    if model_called:
        assert fake.calls == [text]
        assert row["summary"] == MODEL_ANSWER
    else:
        assert fake.calls == []
        assert row["summary"] == cap_summary(text, settings.max_summary_chars)


def test_zero_disables_the_rule(tmp_path, monkeypatch) -> None:
    """SUMMARY_SHORT_TEXT_CHARS=0 — правило выключено: всё идёт модельным путём."""
    settings = _settings(tmp_path, monkeypatch, SUMMARY_SHORT_TEXT_CHARS="0")
    assert settings.summary_short_text_chars == 0
    text = "Короткая заметка, которая при включённом правиле не дошла бы до модели."
    note_id, _ = _save(settings, text)
    fake = FixedSummarizer(MODEL_ANSWER)
    worker = BackgroundWorker(settings, HashEmbedder(8), fake)
    assert worker.process_summary_pending() == 1
    assert fake.calls == [text]
    row = _row(settings, note_id)
    assert row["summary"] == MODEL_ANSWER
    assert row["summary_status"] == "ok"
