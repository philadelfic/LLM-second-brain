"""The MCP handle output budget in characters (LSBN-F-13, 3.4.0).

    The owner's decision: output reduction is the DEFAULT BEHAVIOR, not an
    option - a handle serves headers within the budget, and the full text is
    read point-wise by id (`memory_get` of a single id). The unit is CHARACTERS,
    not tokens or bytes: the parameter is called `budget_chars`, and an item's
    measure is the length of its JSON serialization (`len(json.dumps(...))`).

    The default numbers live IN CODE, not env (following the `injection_budget`
    precedent, FR-4.4): raising the budget is a deliberate code change with the
    reason recorded.

    The module is pure functions without domain or I/O: the transport (`mcp.py`)
    serializes the items itself and calls `fit_within_budget`. The parameter range
    is checked by the transport (the soft refusal `HINT_INVALID_BUDGET`).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

#: The `budget_chars` parameter range (characters per answer).
BUDGET_CHARS_MIN = 100
BUDGET_CHARS_MAX = 100_000

#: The per-surface defaults (characters per answer). Search: a header ~60-100 chars,
#: top_k=5 fits with margin, top_k=20 is cut honestly. List: a titles page
#: (20 x ~60) fits whole; summaries is cut to ~6 items.
#: Get: the batch headers (20 x ~70) - with margin.
SEARCH_BUDGET_CHARS_DEFAULT = 1200
LIST_BUDGET_CHARS_DEFAULT = 2000
GET_BUDGET_CHARS_DEFAULT = 2000

#: The note environment map (LSBN-F-16): its own character ceiling for the
#: `neighbors` rows at `memory_get(map=true)`. The map lives on a single read
#: (where `budget_chars` does not apply), so the ceiling is its own - a code
#: constant, not env and not a handle parameter; a row that does not fit
#: degrades to a header with the truncation flag, as in F-13.
MAP_BUDGET_CHARS_DEFAULT = 1500

#: An item's header projection on degradation (brief p.4: an item that does
#: not fit whole is served as a header + the truncation flag).
HEADER_FIELDS: tuple[str, ...] = ("id", "title", "namespace")

#: The parameter's soft refusal (fail + hint, like mode/detail).
HINT_INVALID_BUDGET = (
    f"budget_chars out of range; expected {BUDGET_CHARS_MIN}.."
    f"{BUDGET_CHARS_MAX} characters"
)

#: The truncation flag is the ONLY hint of a successful answer (an extension
#: of the §4.2.6 listing exception, the 3.4.0 draft): how many items did not
#: fit, their ids (up to 8), and how to read on.
HINT_TRUNCATED = (
    "truncated: +{n} did not fit the {chars}-char budget ({ids}) — "
    "read point-wise by id"
)

#: How many ids of the unfitted items the hint shows (the rest - "+k").
HINT_IDS_MAX = 8


def item_chars(item: dict[str, Any]) -> int:
    """An item's size in characters - as it will land in the JSON output."""
    return len(json.dumps(item, ensure_ascii=False, separators=(",", ":")))


def header_of(item: dict[str, Any]) -> dict[str, Any]:
    """An item's header projection: identity only (id, title, namespace)."""
    return {name: item[name] for name in HEADER_FIELDS if name in item}


def fit_within_budget(
    items: Sequence[dict[str, Any]], budget_chars: int
) -> tuple[list[dict[str, Any]], list[Any], bool]:
    """Lay the items out under a character budget (two-level reading, p.2/p.4).

            An item that fits is served as is. One that does not degrades to a header
            with `"truncated": true` (the header fits - serve the header, the agent
            knows the id; even the header does not fit - skip the item). Returns
            `(fitted, cut_ids, any_truncated)`: the served items, the ids of everyone
            not received whole, and whether any degradation happened.

    """
    fitted: list[dict[str, Any]] = []
    cut_ids: list[Any] = []
    used = 0
    any_truncated = False
    for item in items:
        cost = item_chars(item)
        if used + cost <= budget_chars:
            fitted.append(dict(item))
            used += cost
            continue
        cut_ids.append(item.get("id"))
        header = header_of(item) | {"truncated": True}
        header_cost = item_chars(header)
        if used + header_cost <= budget_chars:
            fitted.append(header)
            used += header_cost
            any_truncated = True
    return fitted, cut_ids, any_truncated


def truncation_hint(cut_ids: Sequence[Any], budget_chars: int) -> str:
    """The truncation hint: how many did not fit and which ids (up to `HINT_IDS_MAX`)."""
    shown = [str(note_id) for note_id in cut_ids[:HINT_IDS_MAX]]
    hidden = len(cut_ids) - len(shown)
    if hidden > 0:
        shown.append(f"+{hidden}")
    return HINT_TRUNCATED.format(
        n=len(cut_ids), chars=budget_chars, ids=", ".join(shown)
    )
