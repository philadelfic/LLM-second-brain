"""The TTL parser (lsb-0004-02, step 2): a relative TTL -> an absolute expires_at.

    The format: `\d+(s|m|h|d|w)` - a positive integer + a unit (for example,
    `1d`, `2h`, `30m`, `45s`, `2w`). Parsed into an absolute `expires_at`
    (ISO-8601 UTC, the same format as created_at/updated_at in the DB -
    `strftime('%Y-%m-%dT%H:%M:%SZ','now')`).

    Units: s=seconds, m=minutes, h=hours, d=days, w=weeks. The number is a
    positive integer (>= 1).

    An invalid format (an empty string, zero/negative, an unknown unit, garbage) -
    a soft refusal: `TTLValidationError` (a subclass of `NoteValidationError`),
    which the transport wraps into fail+hint (modeled after `TitleValidationError`
    in app/services/notes.py).
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

from app.services.notes import NoteValidationError

# The `\d+(s|m|h|d|w)` format: a positive integer + a unit. `\d+` already
# guarantees >= 1 digit; a zero value ("0d") is caught by a separate check.
_TTL_RE = re.compile(r"^(\d+)([smhdw])$")

# Seconds per TTL unit.
_UNIT_SECONDS: dict[str, int] = {
    "s": 1,
    "m": 60,
    "h": 3600,
    "d": 86400,
    "w": 604800,
}

# The ISO-8601 UTC format, as in the DB (strftime('%Y-%m-%dT%H:%M:%SZ','now')).
_ISO_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

# The hint for fail+hint (the model client learns from hints, §5.3).
TTL_HINT = "expires_at: ожидается TTL вида «1d», «2h», «30m», «45s», «2w»"


class TTLValidationError(NoteValidationError):
    """An invalid TTL (lsb-0004-02): an empty string, zero/negative,
            an unknown unit, garbage.

            A soft refusal: the transport (MCP/REST) catches the parent
            NoteValidationError -> fail + hint (like TitleValidationError).

    """


def parse_ttl(ttl: str) -> str:
    """A relative TTL -> an absolute `expires_at` (ISO-8601 UTC).

            The format `\d+(s|m|h|d|w)` (for example, `1d`, `2h`, `30m`, `45s`,
            `2w`); the number is a positive integer (>= 1). An invalid format ->
            `TTLValidationError` (a soft refusal; the transport wraps it into
            fail+hint).

            Args:
                ttl: the relative lifetime, for example `"1d"`.

            Returns:
                The absolute `expires_at` in the `%Y-%m-%dT%H:%M:%SZ` (UTC) format -
                like created_at/updated_at in the DB.

            Raises:
                TTLValidationError: an invalid format (empty, 0/negative, an unknown
                    unit, garbage).

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
