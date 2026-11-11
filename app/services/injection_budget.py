"""The injected-text budget (LSBN-F-11, 3.2.0).

    The SINGLE token metric of the whole release lives here: the measurement
    script `scripts/measure_injected_text.py` (f11-01), the test guard
    `tests/test_injection_budget.py` (f11-03) and the budget constants
    (f11-02/f11-03) all use it.

    The encoder is NOT created a second time: `cl100k_base` is taken from
    `app.services.splitter` - the same BPE vocabulary as the note token splitter,
    one load per process. A second encoder in the project would mean two different
    token numbers for one text; this way that never happens.
"""

from __future__ import annotations

from app.services.splitter import encoding

#: The metric's BPE vocabulary name - the same as the splitter's (pinned, not from env).
ENCODING_NAME = "cl100k_base"

#--- Injected-text budgets (LSBN-F-11, arch §13.17; canon §6) -------------
# The numbers live IN CODE (not env): raising any budget is a deliberate code
# change with the reason recorded (FR-4.4), not a silent env edit. Consumers:
# the texts of `app/transport/mcp.py` (f11-02) and the test guard (f11-03).

#: The whole instructions at connect - `build_instructions` (the core + the map etc.).
TOKENS_INSTRUCTIONS_MAX = 1300
#: The short `SERVER_INSTRUCTIONS` core (the role, the rules, the pointers).
TOKENS_CORE_MAX = 300
#: One tool description (`TOOL_DESCRIPTIONS`).
TOKENS_TOOL_DESCRIPTION_MAX = 120
#: All the surface descriptions (32) - the sum.
TOKENS_TOOL_SURFACE_MAX = 2600

#: Bytes per `memory_search` output element (the 3.1.0 call-weight metric).
BYTES_SEARCH_ITEM_MAX = 450
#: Bytes per `memory_list` listing element (the summaries detail).
BYTES_LIST_ITEM_MAX = 500
#: Bytes per `titles` listing element (the registry-scan mode).
BYTES_LIST_TITLE_ITEM_MAX = 100
#: Bytes for the related-notes list (`links` of a single read).
BYTES_LINKS_MAX = 500
#: The `memory_save` response bytes with the new fields (`namespace_hint`/`similar`).
BYTES_SAVE_RESPONSE_MAX = 900

#: The `skills_search` hint characters ("the body - `skills_get` only").
CHARS_SKILLS_HINT_MAX = 120
#: The version-update hint characters (`update_hint`).
CHARS_UPDATE_HINT_MAX = 240

__all__ = [
    "ENCODING_NAME",
    "count_tokens",
    "TOKENS_INSTRUCTIONS_MAX",
    "TOKENS_CORE_MAX",
    "TOKENS_TOOL_DESCRIPTION_MAX",
    "TOKENS_TOOL_SURFACE_MAX",
    "BYTES_SEARCH_ITEM_MAX",
    "BYTES_LIST_ITEM_MAX",
    "BYTES_LIST_TITLE_ITEM_MAX",
    "BYTES_LINKS_MAX",
    "BYTES_SAVE_RESPONSE_MAX",
    "CHARS_SKILLS_HINT_MAX",
    "CHARS_UPDATE_HINT_MAX",
]


def count_tokens(text: str) -> int:
    """The `cl100k_base` token count; an empty string -> 0 (no encoder call).

            The encoder cache lives in `splitter.encoding()` (`lru_cache`): the BPE
            vocabulary is parsed once per process; the counter only counts.

    """
    if not text:
        return 0
    return len(encoding().encode(text))
