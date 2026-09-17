"""Listing page fields — one place for every listing (lsb-0013, release 3.1.0).

Both listings of the service layer (`NoteService.list`, `SkillsService.list`)
answer with the same pagination contract: `total`, `has_more`, `next_offset`
and `next_cursor`. The arithmetic lives here so the two services cannot drift
apart (arch lsb-0013 §3.2); the transports only render the result.

`next_cursor` is always None for now: paging reuses `offset`, and the field is
reserved for a future keyset over `(updated_at, id)` without breaking the
contract (requirements FR-3.2/FR-3.3).
"""

from __future__ import annotations

from typing import Any


def page_fields(total: int, offset: int, count: int) -> dict[str, Any]:
    """Pagination fields of one page: total / has_more / next_offset / next_cursor.

    `total` — number of active records matching the same filter as the page;
    `count` — items actually returned for this page (smaller than the requested
    limit on the last page, so `next_offset` is derived from the real items, not
    from the limit). `has_more` is true while the page does not reach the end of
    the filtered set; `next_offset` is the ready-to-use offset of the following
    request, otherwise None.
    """
    has_more = offset + count < total
    return {
        "total": total,
        "has_more": has_more,
        "next_offset": offset + count if has_more else None,
        "next_cursor": None,
    }
