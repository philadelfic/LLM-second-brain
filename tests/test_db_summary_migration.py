"""Миграция лимита саммари (решение гейта 3.1.0, 2026-09-19): 150 символов.

Апгрейд живой БД на 3.1.0: активные заметки с ГОТОВЫМ саммари длиннее
`MAX_SUMMARY_CHARS` помечаются на перегенерацию (`summary_status='pending'`),
текст саммари НЕ портится грубым усечением — новую выжимку кладёт фоновая
джоба `summary` (её ответ режет `summary.cap_summary`). Заметки, которые и так
в очереди (`summary_status != 'ok'`), не трогаются; trash (джоба его не
обслуживает) остаётся как есть. Идемпотентность: после перегенерации саммари
≤ лимита — повторный старт сервиса ничего не находит (no-op, без события в
логе); отдельного штампа в `meta` механика не требует.
"""

from __future__ import annotations

import logging

import pytest

from app.config import Settings, get_settings
from app.storage.db import init_db, session, transaction

# Заведомо длиннее любого разумного лимита: "д" — простые символы без пробелов,
# важно лишь то, что длина саммари больше лимита.
LONG_SUMMARY = "д" * 300
SHORT_SUMMARY = "Короткое саммари заметки."


def _seed(settings: Settings) -> None:
    """Живая БД перед апгрейдом: четыре заметки с разными состояниями саммари.

    1 — готовое длинное (миграция обязана поставить pending, текст не трогать);
    2 — готовое короткое (в лимите — не трогаем);
    3 — уже pending с длинным текстом (и так в очереди — статус не переписываем);
    4 — trash с готовым длинным (джоба `summary` trash не обслуживает).
    """
    with session(settings) as conn, transaction(conn):
        seeded = (
            (1, LONG_SUMMARY, "ok", None),
            (2, SHORT_SUMMARY, "ok", None),
            (3, LONG_SUMMARY, "pending", None),
            (4, LONG_SUMMARY, "ok", "2026-09-19T00:00:00Z"),
        )
        for note_id, summary, summary_status, deleted_at in seeded:
            conn.execute(
                "INSERT INTO notes (id, text, title, summary, summary_status, "
                "deleted_at) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    note_id,
                    f"текст заметки номер {note_id} про релиз и деплой",
                    f"Заметка {note_id}",
                    summary,
                    summary_status,
                    deleted_at,
                ),
            )


@pytest.fixture
def upgraded(test_env: dict[str, str]) -> Settings:
    """Свежая БД (схема + сиды) и данные до миграции лимита саммари."""
    settings = get_settings()
    init_db(settings)
    _seed(settings)
    return settings


def _summaries(settings: Settings) -> dict[int, tuple[str, str]]:
    """Снимок полей саммари всех заметок: id → (текст, статус)."""
    with session(settings) as conn:
        return {
            row["id"]: (row["summary"], row["summary_status"])
            for row in conn.execute(
                "SELECT id, summary, summary_status FROM notes ORDER BY id"
            )
        }


def _regen_events(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [
        record
        for record in caplog.records
        if getattr(record, "event", None) == "summary_regen_queued"
    ]


def test_upgrade_queues_only_long_ready_summaries(
    upgraded: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    """Миграция при старте: pending — только готовые саммари длиннее лимита."""
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="app"):
        init_db(get_settings())

    summaries = _summaries(get_settings())
    # 1 — длинное готовое → перегенерация; ТЕКСТ саммари не испорчен усечением.
    assert summaries[1] == (LONG_SUMMARY, "pending")
    # 2 — короткое готовое в лимите → не трогаем.
    assert summaries[2] == (SHORT_SUMMARY, "ok")
    # 3 — уже pending: и так в очереди, статус и текст как были.
    assert summaries[3] == (LONG_SUMMARY, "pending")
    # 4 — trash: джоба его не обслуживает, вечно pending не создаём.
    assert summaries[4] == (LONG_SUMMARY, "ok")

    events = _regen_events(caplog)
    assert len(events) == 1
    assert events[0].levelno == logging.WARNING
    assert events[0].notes == 1  # ровно одна заметка ушла на перегенерацию
    assert events[0].max_summary_chars == get_settings().max_summary_chars


def test_migration_is_idempotent(
    upgraded: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    """Перегенерация выполнена → повторный старт — no-op (без события в логе)."""
    init_db(get_settings())  # миграция: заметка 1 → pending
    # Фоновая джоба `summary` переделала саммари (cap_summary держит лимит).
    with session(get_settings()) as conn, transaction(conn):
        conn.execute(
            "UPDATE notes SET summary = ?, summary_status = 'ok' WHERE id = 1",
            (SHORT_SUMMARY,),
        )

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="app"):
        init_db(get_settings())  # повторный старт сервиса
        init_db(get_settings())  # и ещё один — миграция не воскресает

    assert _regen_events(caplog) == []  # мигрировать больше нечего
    assert _summaries(get_settings()) == {
        1: (SHORT_SUMMARY, "ok"),
        2: (SHORT_SUMMARY, "ok"),
        3: (LONG_SUMMARY, "pending"),  # всё ещё ждёт джобу (свой сценарий)
        4: (LONG_SUMMARY, "ok"),  # trash не мигрировался ни разу
    }


def test_limit_comes_from_settings(
    upgraded: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Порог миграции — именно `settings.max_summary_chars`, а не зашитые 150."""
    monkeypatch.setenv("MAX_SUMMARY_CHARS", "50")
    get_settings.cache_clear()
    try:
        init_db(get_settings())
        summaries = _summaries(get_settings())
    finally:
        get_settings.cache_clear()
    # Короткое саммари (< 50 симв.) остаётся 'ok', длинное (300) — в очередь.
    assert summaries[2] == (SHORT_SUMMARY, "ok")
    assert summaries[1] == (LONG_SUMMARY, "pending")
