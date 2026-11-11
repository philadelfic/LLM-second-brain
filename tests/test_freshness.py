"""The `updated_after` freshness (no29-01, release 3.2.0) — parsing the input value.

    Arch §6.1/§6.6/§13.3: only an absolute ISO-8601 UTC is accepted — a date
    (`YYYY-MM-DD`, midnight) or a moment with seconds (`YYYY-MM-DDTHH:MM:SSZ`).
    Any other form (an empty string, garbage, a time without a date, a `+03:00`
    offset, relative wording) is rejected with FreshnessValidationError — the
    transports catch it as a soft failure (fail + hint), not a crash.
"""

from __future__ import annotations

import pytest

from app.services.freshness import FreshnessValidationError, parse_updated_after


def test_parse_accepts_date_and_datetime() -> None:
    """A date → UTC midnight; a moment — as is (normalized to `...Z`)."""
    assert parse_updated_after("2026-10-08") == "2026-10-08T00:00:00Z"
    assert parse_updated_after("2026-10-08T12:34:56Z") == "2026-10-08T12:34:56Z"


@pytest.mark.parametrize(
    "value",
    ["", "за неделю", "2026-10-08T12:34", "2026-10-08+03:00"],
)
def test_parse_rejects_garbage(value: str) -> None:
    """Non-ISO-8601-UTC forms — one failure (a transport's soft failure)."""
    with pytest.raises(FreshnessValidationError):
        parse_updated_after(value)
