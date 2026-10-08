"""Кодеки дельты истории (№41, постановка no41-01): формат §5.2, восстановление.

Дельта — один JSON-объект «поле → представление нового значения»: короткое
поле кодируется значением целиком, длинное (≥ `HISTORY_DELTA_INLINE_MAX`) —
построчной заменой `{"__lines__": [[i, j, [строки]], …]}` по не-`equal` опкодам
`difflib.SequenceMatcher`; `null` — поле снято. Восстановление — один алгоритм
`apply_delta`, возвращающий значение байт-в-байт (FR-2.2/FR-2.3).
"""

from __future__ import annotations

import json

from app.config import get_settings
from app.services.history import apply_delta, encode_delta

LONG_TEXT_LINES = 400  # заведомо длиннее порога: одна строка ≈ 10 символов


def _long_text() -> str:
    """Текст длиннее порога дельта-кодирования (одна строка ≈ 10 символов)."""
    return "\n".join(f"строка {index}" for index in range(LONG_TEXT_LINES))


def test_delta_roundtrip_long_text() -> None:
    """Правка одного слова в длинном тексте: дельта без полного текста, байт-в-байт."""
    inline_max = get_settings().history_delta_inline_max
    previous = {"text": _long_text()}
    current = {"text": previous["text"].replace("строка 7", "строка семь", 1)}
    assert len(current["text"]) >= inline_max

    delta, changed = encode_delta(previous, current, inline_max=inline_max)

    assert changed == ["text"]
    assert "__lines__" in delta["text"]
    # Дельта не хранит полный текст: и в самой записи, и в JSON целиком.
    assert current["text"] not in json.dumps(delta, ensure_ascii=False)
    assert len(json.dumps(delta, ensure_ascii=False)) < len(current["text"])
    # Восстановление — байт-в-байт.
    assert apply_delta(dict(previous), delta) == current


def test_delta_short_field_verbatim() -> None:
    """Короткое поле кодируется значением целиком (без построчной формы)."""
    delta, changed = encode_delta(
        {"title": "Старое название", "namespace": "default"},
        {"title": "Новое название", "namespace": "default"},
        inline_max=2048,
    )

    assert changed == ["title"]
    assert delta == {"title": "Новое название"}


def test_delta_repeat_same_field() -> None:
    """Одно поле правится дважды: baseline + две дельты восстанавливают обе версии."""
    baseline = {"text": "первая\nвторая\nтретья"}
    second = {"text": "первая\nВТОРАЯ\nтретья"}
    third = {"text": "первая\nВТОРАЯ\nТРЕТЬЯ"}

    delta_2, _ = encode_delta(baseline, second, inline_max=1)
    delta_3, _ = encode_delta(second, third, inline_max=1)

    # `apply_delta` меняет словарь на месте (алгоритм §5.2) — копии для сверки.
    restored_2 = apply_delta(dict(baseline), delta_2)
    restored_3 = apply_delta(dict(restored_2), delta_3)
    assert restored_2 == second
    assert restored_3 == third


def test_delta_field_removed_is_null() -> None:
    """Снятие поля кодируется `null`, восстановление даёт `None`."""
    delta, changed = encode_delta(
        {"example": "был текст", "name": "имя"},
        {"example": None, "name": "имя"},
        inline_max=2048,
    )

    assert changed == ["example"]
    assert delta == {"example": None}
    assert apply_delta({"example": "был текст"}, delta) == {"example": None}


def test_changed_fields_list() -> None:
    """`changed` — список изменившихся полей, порядок стабилен (порядок полей)."""
    previous = {"name": "a", "description": "b", "steps": "c"}
    current = {"name": "a", "description": "B", "steps": "C"}

    delta, changed = encode_delta(previous, current, inline_max=2048)

    assert changed == ["description", "steps"]
    assert list(delta) == ["description", "steps"]
