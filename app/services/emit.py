"""The output shaping (the Phase 2 brief, p. 3): summary/snippet.

    - `summary` - the fallback truncation: with `summary_status=pending` (Phase 2
      - always; later - a summarizer failure, Phase 4+) the first
      MAX_SUMMARY_CHARS characters of the text are served (REQUIREMENTS §5.5). A
      ready summary is served as is.
    - `snippet` - the first SNIPPET_CHARS characters of the text, always
      (ARCH §4.2) - so the model can judge relevance without opening the full
      text.

    The truncation counts CHARACTERS of the string (not bytes): Cyrillic is
    measured honestly.

    The summary limit is system-wide (the 3.1.0 gate decision: 150 characters):
    the fallback is cut by characters (this is the NOTE text, not a model answer -
    a crude cut is safe here, no ellipsis is added); the model summary is held to
    the same constant at the save point (worker, summary.cap_summary - a
    word-boundary truncation).
"""

from __future__ import annotations

import sqlite3

from app.config import Settings


def summary_of(row: sqlite3.Row, settings: Settings) -> str:
    """The summary for the output: the ready one or the fallback truncation (§5.5)."""
    if row["summary_status"] == "ok" and row["summary"]:
        return row["summary"]
    return row["text"][: settings.max_summary_chars]


def snippet(text: str, settings: Settings) -> str:
    """The text fragment for memory_search: the first SNIPPET_CHARS characters."""
    return text[: settings.snippet_chars]