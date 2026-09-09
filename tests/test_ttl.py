"""Тесты TTL-парсера (lsb-0004-02, этап 2): относительный TTL → expires_at.

Валидные форматы: 1d, 2h, 30m, 45s, 2w. Невалидные: пусто, 0d, -1h, 1x,
abc, 1.5d. Невалидный формат — мягкий отказ (TTLValidationError, подкласс
NoteValidationError — транспорт обернёт в fail+hint).
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

import pytest

from app.services.notes import NoteValidationError
from app.services.ttl import TTL_HINT, TTLValidationError, parse_ttl

# Формат ISO-8601 UTC, как в БД (strftime('%Y-%m-%dT%H:%M:%SZ','now')).
_ISO_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

# Секунд в единице TTL (зеркало _UNIT_SECONDS в app/services/ttl.py).
_UNIT_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}


def _parse_iso(value: str) -> datetime:
    """Распарсить ISO-8601 UTC (без Z-суффикса) в aware datetime."""
    return datetime.strptime(value, _ISO_FORMAT).replace(tzinfo=timezone.utc)


def _floor_seconds(dt: datetime) -> datetime:
    """Округлить вниз до целой секунды: expires_at имеет точность до секунды."""
    return dt.replace(microsecond=0)


class TestParseTtlValid:
    """Валидные форматы: 1d, 2h, 30m, 45s, 2w."""

    @pytest.mark.parametrize(
        ("ttl", "unit"),
        [
            ("1d", "d"),
            ("2h", "h"),
            ("30m", "m"),
            ("45s", "s"),
            ("2w", "w"),
        ],
    )
    def test_valid_formats(self, ttl: str, unit: str) -> None:
        """Валидный TTL → абсолютный expires_at в будущем, ровно на TTL вперёд."""
        value = int(ttl[:-1])  # число из «2w» → 2
        expected_seconds = value * _UNIT_SECONDS[unit]
        before = _floor_seconds(datetime.now(timezone.utc))
        expires_at = parse_ttl(ttl)
        after = _floor_seconds(datetime.now(timezone.utc))
        parsed = _parse_iso(expires_at)
        # Формат — ISO-8601 UTC, как в БД.
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", expires_at)
        # expires_at ≈ now + TTL: parsed - TTL попадает в окно вызова [before, after].
        assert before <= parsed - timedelta(seconds=expected_seconds) <= after

    def test_seconds(self) -> None:
        """45s — ровно 45 секунд вперёд."""
        before = _floor_seconds(datetime.now(timezone.utc))
        parsed = _parse_iso(parse_ttl("45s"))
        after = _floor_seconds(datetime.now(timezone.utc))
        assert before <= parsed - timedelta(seconds=45) <= after

    def test_weeks(self) -> None:
        """2w — ровно 2 недели (14 дней) вперёд."""
        before = _floor_seconds(datetime.now(timezone.utc))
        parsed = _parse_iso(parse_ttl("2w"))
        after = _floor_seconds(datetime.now(timezone.utc))
        assert before <= parsed - timedelta(weeks=2) <= after

    def test_large_value(self) -> None:
        """Большое целое число — валидно (например, 365d)."""
        before = _floor_seconds(datetime.now(timezone.utc))
        parsed = _parse_iso(parse_ttl("365d"))
        after = _floor_seconds(datetime.now(timezone.utc))
        assert before <= parsed - timedelta(days=365) <= after


class TestParseTtlInvalid:
    """Невалидные форматы: пусто, 0d, -1h, 1x, abc, 1.5d."""

    @pytest.mark.parametrize(
        "ttl",
        [
            "",       # пустая строка
            "   ",    # пробельная строка
            "0d",     # нулевое число
            "0s",     # нулевое число (секунды)
            "-1h",    # отрицательное число
            "1x",     # неизвестная единица
            "abc",    # мусор
            "1.5d",   # дробное число
            "1d2h",   # несколько единиц
            "d",      # без числа
            "1",      # без единицы
            "1D",     # заглавная единица (регистр важен)
        ],
    )
    def test_invalid_raises(self, ttl: str) -> None:
        """Невалидный TTL → TTLValidationError (мягкий отказ)."""
        with pytest.raises(TTLValidationError):
            parse_ttl(ttl)

    def test_is_note_validation_error(self) -> None:
        """TTLValidationError — подкласс NoteValidationError (транспорт ловит)."""
        assert issubclass(TTLValidationError, NoteValidationError)

    def test_hint_in_message(self) -> None:
        """Сообщение исключения несёт TTL_HINT (fail+hint для клиента)."""
        with pytest.raises(TTLValidationError) as excinfo:
            parse_ttl("1x")
        assert TTL_HINT in str(excinfo.value)

    def test_empty_hint(self) -> None:
        """Пустая строка → TTL_HINT (без мусора в hint)."""
        with pytest.raises(TTLValidationError) as excinfo:
            parse_ttl("")
        assert str(excinfo.value) == TTL_HINT

    def test_non_string(self) -> None:
        """Не-строка (None) → TTLValidationError."""
        with pytest.raises(TTLValidationError):
            parse_ttl(None)  # type: ignore[arg-type]
