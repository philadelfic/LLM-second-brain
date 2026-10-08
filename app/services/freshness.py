"""Свежесть памяти — параметр `updated_after` (no29-01, релиз 3.2.0).

Чистые функции разбора/нормализации входного значения `updated_after` для
`memory_search`/`memory_list` и REST-поверхности оператора (arch §6.1/§6.4/§6.6,
§8, §13.3). Принимается только абсолютная дата в ISO-8601 UTC: `YYYY-MM-DD`
(полночь) или `YYYY-MM-DDTHH:MM:SSZ`. Граница включающая и одинаковая в поиске
и листинге; относительные выражения («за неделю», `1d`) не вводятся — решение
О. 2026-10-08. Синтаксис `updated_at` не меняется: служебные обновления
(векторизация, суммаризация) признак контентной свежести не двигают, отдельной
колонки «контентно обновлено» нет (§13.3).

Модуль чистый: без импорта домена и транспорта — его используют и сервисы, и
оба транспорта. Невалидное значение поднимает `FreshnessValidationError`,
которую транспорты ловят как мягкий отказ (fail + hint).
"""

from __future__ import annotations

from datetime import datetime

# Единственные допустимые формы параметра (arch §6.1): секундная точность,
# только UTC (`Z`) — без смещений и без долей секунды.
ISO_SECONDS = "%Y-%m-%dT%H:%M:%SZ"
ISO_DATE = "%Y-%m-%d"

# Текст отказа — единый для сервисов и обоих транспортов (мягкий отказ+хинт).
INVALID_UPDATED_AFTER = (
    "unknown updated_after; expected ISO-8601 UTC: "
    "YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ"
)


class FreshnessValidationError(ValueError):
    """Невалидное значение свежести — мягкий отказ транспорта (fail + hint)."""


def parse_updated_after(value: str) -> str:
    """Нормализовать `updated_after` к `YYYY-MM-DDTHH:MM:SSZ` (UTC).

    Принимаются ровно две формы: дата (`YYYY-MM-DD` → `00:00:00Z`) и момент с
    секундами (`YYYY-MM-DDTHH:MM:SSZ`). Любая другая форма (пустая строка,
    мусор, время без даты, смещение `+03:00`, относительные выражения) —
    `FreshnessValidationError`.
    """
    for fmt in (ISO_SECONDS, ISO_DATE):
        try:
            moment = datetime.strptime(value, fmt)
        except (TypeError, ValueError):
            continue
        return moment.strftime(ISO_SECONDS)
    raise FreshnessValidationError(INVALID_UPDATED_AFTER)
