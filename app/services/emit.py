"""Формирование выдач (бриф Фазы 2, п. 3): summary/snippet.

- `summary` — fallback-усечение: при `summary_status=pending` (Фаза 2 — всегда;
  далее — отказ суммаризатора, Фаза 4+) выдаются первые MAX_SUMMARY_CHARS
  символов текста (REQUIREMENTS §5.5). Готовое суммари отдаётся как есть.
- `snippet` — первые SNIPPET_CHARS символов текста, всегда (ARCH §4.2) —
  чтобы модель оценила релевантность, не открывая полный текст.

Усечение — по символам строки (не байтам): кириллица меряется честно.
"""

from __future__ import annotations

import sqlite3

from app.config import Settings


def summary_of(row: sqlite3.Row, settings: Settings) -> str:
    """Краткое содержание к выдаче: готовое или fallback-усечение (§5.5)."""
    if row["summary_status"] == "ok" and row["summary"]:
        return row["summary"]
    return row["text"][: settings.max_summary_chars]


def snippet(text: str, settings: Settings) -> str:
    """Фрагмент текста для memory_search: первые SNIPPET_CHARS символов."""
    return text[: settings.snippet_chars]