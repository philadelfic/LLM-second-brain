"""TTL-парсер (lsb-0004-02, этап 2): относительный TTL → абсолютный expires_at.

Формат: `\\d+(s|m|h|d|w)` — целое положительное число + единица (например,
`1d`, `2h`, `30m`, `45s`, `2w`). Парсится в абсолютный `expires_at`
(ISO-8601 UTC, тот же формат, что created_at/updated_at в БД —
`strftime('%Y-%m-%dT%H:%M:%SZ','now')`).

Единицы: s=секунды, m=минуты, h=часы, d=дни, w=недели. Число — целое
положительное (≥1).

Невалидный формат (пустая строка, нулевое/отрицательное число, неизвестная
единица, мусор) — мягкий отказ: `TTLValidationError` (подкласс
`NoteValidationError`), который транспорт обернёт в fail+hint (по образцу
`TitleValidationError` в app/services/notes.py).
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

from app.services.notes import NoteValidationError

# Формат `\d+(s|m|h|d|w)`: целое положительное число + единица. `\d+` уже
# гарантирует ≥1 цифру; нулевое значение («0d») ловится отдельной проверкой.
_TTL_RE = re.compile(r"^(\d+)([smhdw])$")

# Секунд в единице TTL.
_UNIT_SECONDS: dict[str, int] = {
    "s": 1,
    "m": 60,
    "h": 3600,
    "d": 86400,
    "w": 604800,
}

# Формат ISO-8601 UTC, как в БД (strftime('%Y-%m-%dT%H:%M:%SZ','now')).
_ISO_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

# Подсказка для fail+hint (клиент-модель учится по hint, §5.3).
TTL_HINT = "expires_at: ожидается TTL вида «1d», «2h», «30m», «45s», «2w»"


class TTLValidationError(NoteValidationError):
    """Невалидный TTL (lsb-0004-02): пустая строка, нулевое/отрицательное
    число, неизвестная единица, мусор.

    Мягкий отказ: транспорт (MCP/REST) ловит родительский
    NoteValidationError → fail + hint (как TitleValidationError).
    """


def parse_ttl(ttl: str) -> str:
    """Относительный TTL → абсолютный `expires_at` (ISO-8601 UTC).

    Формат `\\d+(s|m|h|d|w)` (например, `1d`, `2h`, `30m`, `45s`, `2w`);
    число — целое положительное (≥1). Невалидный формат → `TTLValidationError`
    (мягкий отказ, транспорт обернёт в fail+hint).

    Args:
        ttl: относительный срок жизни, например `"1d"`.

    Returns:
        Абсолютный `expires_at` в формате `%Y-%m-%dT%H:%M:%SZ` (UTC) — как
        created_at/updated_at в БД.

    Raises:
        TTLValidationError: невалидный формат (пусто, 0/отрицательное число,
            неизвестная единица, мусор).
    """
    if not isinstance(ttl, str) or not ttl.strip():
        raise TTLValidationError(TTL_HINT)
    match = _TTL_RE.match(ttl)
    if match is None:
        raise TTLValidationError(
            f"expires_at: невалидный TTL «{ttl}» — {TTL_HINT}"
        )
    value = int(match.group(1))
    unit = match.group(2)
    if value <= 0:
        raise TTLValidationError(
            f"expires_at: TTL «{ttl}» — число должно быть положительным"
        )
    seconds = value * _UNIT_SECONDS[unit]
    expires = datetime.now(timezone.utc) + timedelta(seconds=seconds)
    return expires.strftime(_ISO_FORMAT)
