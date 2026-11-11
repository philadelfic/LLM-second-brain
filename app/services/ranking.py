"""The shared search ranking primitives (3.0.0): RRF and the FTS expression.

    Extracted from `app.services.search` when the area substrate arrived (3.0.0):
    the hybrid note search and the in-area hybrid search merge their sources the
    same way - `score(d) = sum over sources of 1/(RRF_K + rank_source)` with ranks
    from 1 - and build the same FTS query expression (words >= 3 characters as
    quoted substrings joined by OR; the trigram contract, BUG-001).

    The behavior of the existing note search does not change: `SearchService`
    calls these very functions (the RRF merge and `_match_expression`), and the
    areas use `AreaSearch` (app.services.areas).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import datetime, timezone

# The query's compound-token split (BUG-001): "open-webui" -> "open" +
#"webui" - FTS catches texts whose spelling differs ("Open WebUI").
_TOKEN_SPLIT_RE = re.compile(r"[^0-9A-Za-zА-Яа-яЁё]+")

#--- ranking boosts after RRF (3.4.0, LSBN-F-14) ------------------
# Deterministic multipliers over the merge (`apply_boosts`), no models and
# no VRAM. The weights are CODE constants (not env; pinned in the canon via
# changes.md); the operator switch is the env RANKING_BOOSTS_ENABLED
# (app.config.Settings.ranking_boosts_enabled, default true). The boosts shape
# the OUTPUT ORDER ONLY - an item's rrf_score stays the raw RRF
# (the §4.2.6 output contract / the F-13 headers do not change).

TITLE_EXACT_BOOST = 0.25  # the title matched the query's significant words exactly
TITLE_PREFIX_BOOST = 0.15  # the title starts with the query string (casefold)
FRESHNESS_BOOST_MAX = 0.10  # the note was updated just now ("today")
FRESHNESS_WINDOW_DAYS = 30  # the linear freshness decay window (days)
LINKED_BOOST = 0.05  # the note has a links row (either side)

# The note timestamp format (DDL: strftime('%Y-%m-%dT%H:%M:%SZ','now')).
_TS_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
_SECONDS_PER_DAY = 86400.0


def significant_tokens(text: str) -> list[str]:
    """The significant tokens of a text: words >= 3 characters plus the >= 3-char
            parts of compound tokens, in text order, without duplicates (BUG-001).

            "open-webui" -> "open-webui", "open", "webui" - FTS catches texts whose
            spelling differs ("Open WebUI"); "report" -> only "report". The text order
            is preserved (the first occurrence), a repeat is not added.

    """
    tokens: list[str] = []
    for word in text.split():
        if len(word) >= 3:
            tokens.append(word)
        for part in _TOKEN_SPLIT_RE.split(word):
            if len(part) >= 3 and part != word:
                tokens.append(part)
    return list(dict.fromkeys(tokens))


def match_expression(query: str) -> str | None:
    """The significant query words as quoted substrings joined by OR (FTS).

            OR, not AND (BUG-001): AND dropped a note entirely when even one query
            word was absent from its text. With OR, BM25 ranks by the number/rarity of
            the matched words - a record with all the words ranks higher; the noise is
            cut by the RRF merge and top_k. None - there is not a single word the
            trigram could search by (all the words are shorter than 3 characters).

    """
    tokens = significant_tokens(query)
    if not tokens:
        return None
    return " OR ".join(
        f'"{token.replace(chr(34), chr(34) * 2)}"' for token in tokens
    )


def fuse_rrf(rankings: Sequence[Sequence[int]], rrf_k: int) -> dict[int, float]:
    """Merge ranked id lists: `score(d) = sum of 1/(RRF_K + rank)`.

            RRF is robust to incomparable scales (cosine vs BM25); a source's
            contribution is the record's position in its output (rank from 1). A
            repeated id within one list adds up (the routine outputs have no
            duplicates - deduplication happens above); ids from different sources
            add up - that is exactly the merge's deduplication.

    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (rrf_k + rank)
    return scores


#--- the ranking boosts (3.4.0, LSBN-F-14) -----------------------------


def utc_now_iso() -> str:
    """The current UTC moment in the note-stamp format - the `now` input of the boosts."""
    return datetime.now(timezone.utc).strftime(_TS_FORMAT)


def title_boost(query: str, title: str | None) -> float:
    """The title boost: an exact match outranks a prefix one.

            "Exact" (TITLE_EXACT_BOOST) - the significant query words
            (`significant_tokens`, in text order) casefold-match the WHOLE title;
            "prefix" (TITLE_PREFIX_BOOST) - the title starts with the query string
            (casefold; the query's edge spaces are dropped). The stages are mutually
            exclusive: the exact one is checked first and the prefix one is not added.
            None/empty title - no boost (legacy notes without a title are neither
            penalized nor lifted).

    """
    if not title:
        return 0.0
    title_folded = " ".join(title.casefold().split())
    tokens = significant_tokens(query)
    if tokens and " ".join(tokens).casefold() == title_folded:
        return TITLE_EXACT_BOOST
    needle = query.strip().casefold()
    if needle and title_folded.startswith(needle):
        return TITLE_PREFIX_BOOST
    return 0.0


def freshness_boost(updated_at: str | None, now: str) -> float:
    """The freshness boost: a linear decay over updated_at across FRESHNESS_WINDOW_DAYS.

            "Today" (age <= 0 - the stamps share the UTC timezone) - the maximum
            FRESHNESS_BOOST_MAX; then a decay to zero at the window border; 30+ days -
            0. An unparseable/absent stamp - 0 (a deterministic degradation: the boost
            simply does not apply).

    """
    if not updated_at:
        return 0.0
    try:
        updated = datetime.strptime(updated_at, _TS_FORMAT)
        moment = datetime.strptime(now, _TS_FORMAT)
    except ValueError:
        return 0.0
    age_days = (moment - updated).total_seconds() / _SECONDS_PER_DAY
    if age_days <= 0.0:
        return FRESHNESS_BOOST_MAX
    if age_days >= FRESHNESS_WINDOW_DAYS:
        return 0.0
    return FRESHNESS_BOOST_MAX * (1.0 - age_days / FRESHNESS_WINDOW_DAYS)


def apply_boosts(
    scores: dict[int, float],
    rows: dict[int, dict],
    *,
    now: str,
    enabled: bool,
) -> dict[int, float]:
    """The multipliers after the RRF merge: `score x (1 + sum of boosts)` (LSBN-F-14).

            `rows[note_id]` is the row-dict contract of the boosts (absent/incomplete -
            a boost of 0, a deterministic degradation):
              - "query": str - the query itself (the same for all rows; the title
                boosts' input);
              - "title": str | None - the note's title;
              - "updated_at": str | None - the freshness stamp (the DDL default format);
              - "has_links": bool - there is a links row (either side).
            Returns a NEW dict; `fuse_rrf` and its callers (including the area search
            in app.services.areas) are untouched. `enabled=False` - the input unchanged
            (the values and the key order) - the "golden" pre-boost order; the rows are
            not even read in that case. The application order is by the sum:
            title + freshness + links, then one multiplier (the 3.4.0 brief).

    """
    if not enabled:
        return dict(scores)
    boosted: dict[int, float] = {}
    for note_id, score in scores.items():
        row = rows.get(note_id) or {}
        total = (
            title_boost(row.get("query", ""), row.get("title"))
            + freshness_boost(row.get("updated_at"), now)
            + (LINKED_BOOST if row.get("has_links") else 0.0)
        )
        boosted[note_id] = score * (1.0 + total)
    return boosted
