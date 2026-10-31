"""Memory freshness - the `updated_after` parameter (no29-01, 3.2.0).

    Pure functions parsing/normalizing the input `updated_after` for
    `memory_search`/`memory_list` and the operator's REST surface
    (arch SS6.1/SS6.4/SS6.6, SS8, SS13.3). Only an absolute ISO-8601 UTC date is
    accepted: `YYYY-MM-DD` (midnight) or `YYYY-MM-DDTHH:MM:SSZ`. The bound is
    inclusive and identical in search and listing; relative expressions ("last
    week", `1d`) are not introduced - the owner's decision. The `updated_at`
    syntax does not change: service updates (vectorization, summarization) do not
    move the content-freshness marker, and there is no separate "content-updated"
    column (SS13.3).

    The module is pure: it imports neither the domain nor the transport - both the
    services and both transports use it. An invalid value raises
    `FreshnessValidationError`, which the transports catch as a soft refusal
    (fail + hint).
"""

from __future__ import annotations

from datetime import datetime

# The only accepted parameter shapes (arch §6.1): second precision,
# UTC only (`Z`) - no offsets and no fractions of a second.
ISO_SECONDS = "%Y-%m-%dT%H:%M:%SZ"
ISO_DATE = "%Y-%m-%d"

# The refusal text is shared by the services and both transports (a soft refusal + a hint).
INVALID_UPDATED_AFTER = (
    "unknown updated_after; expected ISO-8601 UTC: "
    "YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ"
)


class FreshnessValidationError(ValueError):
    """An invalid freshness value - the transport's soft refusal (fail + hint)."""


def parse_updated_after(value: str) -> str:
    """Normalize `updated_after` to `YYYY-MM-DDTHH:MM:SSZ` (UTC).

            Exactly two forms are accepted: a date (`YYYY-MM-DD` -> `00:00:00Z`) and a
            moment with seconds (`YYYY-MM-DDTHH:MM:SSZ`). Any other form (an empty
            string, garbage, a time without a date, a `+03:00` offset, relative
            expressions) - `FreshnessValidationError`.

    """
    for fmt in (ISO_SECONDS, ISO_DATE):
        try:
            moment = datetime.strptime(value, fmt)
        except (TypeError, ValueError):
            continue
        return moment.strftime(ISO_SECONDS)
    raise FreshnessValidationError(INVALID_UPDATED_AFTER)
