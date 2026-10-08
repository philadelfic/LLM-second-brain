"""Свежесть `updated_after` (no29-01, релиз 3.2.0) — разбор входного значения.

Arch §6.1/§6.6/§13.3: принимается только абсолютная ISO-8601 UTC — дата
(`YYYY-MM-DD`, полночь) или момент с секундами (`YYYY-MM-DDTHH:MM:SSZ`).
Любая иная форма (пустая строка, мусор, время без даты, смещение `+03:00`,
относительные выражения) отклоняется FreshnessValidationError — транспорты
ловят её как мягкий отказ (fail + hint), а не как падение.
"""

from __future__ import annotations

import pytest

from app.services.freshness import FreshnessValidationError, parse_updated_after


def test_parse_accepts_date_and_datetime() -> None:
    """Дата → полночь UTC; момент — как есть (нормализация к `...Z`)."""
    assert parse_updated_after("2026-10-08") == "2026-10-08T00:00:00Z"
    assert parse_updated_after("2026-10-08T12:34:56Z") == "2026-10-08T12:34:56Z"


@pytest.mark.parametrize(
    "value",
    ["", "за неделю", "2026-10-08T12:34", "2026-10-08+03:00"],
)
def test_parse_rejects_garbage(value: str) -> None:
    """Не-ISO-8601-UTC формы — единый отказ (мягкий отказ транспорта)."""
    with pytest.raises(FreshnessValidationError):
        parse_updated_after(value)
