"""The token splitter of long notes (Phase 7, brief PHASE7 SS4/SS6).

    A note is stored and served WHOLE (nobody cuts it) - the splitter exists only
    to BUILD vectors: a single vector of a long note "smears" the semantics,
    while per-chunk vectors keep the local semantics, so a search for a fact
    inside a long note finds it (step 3: NoteService splits a note into chunks;
    step 4: SearchService aggregates the chunk hits).

    Decisions (brief SS4/SS5, fixed with the owner):
    - Splitting - a fixed sliding window over tiktoken `cl100k_base` tokens. No
      Markdown Header Splitter: model notes are not markdown documents and may
      have no headers at all.
    - The window is `CHUNK_SIZE` tokens with `CHUNK_OVERLAP` overlap (a window
      seam does not cut a fact crossing the border); a tail chunk shorter than
      `CHUNK_MIN_TARGET` merges into the previous one (a meaningless "scrap" is
      not indexed). The merged tail may exceed CHUNK_SIZE by up to
      chunk_size + chunk_min_target - 1 tokens (~1223 at the defaults) - accepted:
      it is the price of having no junk tail; it does not affect search quality.
    - The encoding is pinned (no env variants): the BPE vocabulary is pre-built
      into the image at build time (Dockerfile) - otherwise a cold container
      without a network fails on the very first split.
    - A token cut may split a multi-byte UTF-8 symbol (byte-fallback tokens) - a
      U+FFFD is possible at a chunk border; for prose this is a rare, acceptable
      defect; decoding by tokens is standard practice (the full note text is not
      distorted: it is never reassembled from chunks).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import tiktoken

# A pinned encoding (brief §4: "a fixed encoding cl100k_base; env is
# not required"): the vocabulary size does not affect note search quality,
# and one encoding simplifies pre-building the cache into the image.
ENCODING_NAME = "cl100k_base"


@dataclass(frozen=True)
class Chunk:
    """A note chunk: the decoded text and the size in tokens.

            `tokens` goes into `notes_chunks.tokens` (step 2) - diagnostics and
            metrics; the text - for the chunk's vectorization (the worker, steps 3/5)
            and the snippet (step 4).

    """

    text: str
    tokens: int


@lru_cache(maxsize=1)
def encoding() -> tiktoken.Encoding:
    """Load the encoding once per process (the BPE parse takes ~0.1-0.5 s)."""
    return tiktoken.get_encoding(ENCODING_NAME)


def count_tokens(text: str) -> int:
    """The `cl100k_base` token count; an empty string -> 0 (no encoder call).

    The encoder cache lives in `splitter.encoding()` (`lru_cache`): the BPE
    vocabulary is parsed once per process; the counter only counts."""
    return len(encoding().encode(text))


def token_windows(
    total: int,
    *,
    chunk_size: int,
    chunk_overlap: int,
    chunk_min_target: int,
) -> list[tuple[int, int]]:
    """Sliding windows over tokens [0, total) (pure arithmetic - unit-testable).

            The rules (brief SS4/SS6):
            - a window = `chunk_size` tokens, the shift = chunk_size - chunk_overlap;
            - the last start is where the remainder already fits one window;
            - a tail window shorter than `chunk_min_target` (and not the only window)
              merges into the previous one: the previous end extends to the shared
              end; such a merged chunk is longer than chunk_size by at most
              chunk_min_target - chunk_overlap - 1 tokens.

            Raises:
                ValueError: invalid parameters - overlap >= size (the windows never
                    shift), min_target > size or < 1.

    """
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) обязан быть меньше "
            f"chunk_size ({chunk_size}), иначе окна не сдвигаются"
        )
    if chunk_min_target < 1:
        raise ValueError(f"chunk_min_target должен быть ≥ 1, получено {chunk_min_target}")
    if chunk_min_target > chunk_size:
        raise ValueError(
            f"chunk_min_target ({chunk_min_target}) не может превышать "
            f"chunk_size ({chunk_size}): тогда хвостовой чанк всегда сливается"
        )
    if total <= 0:
        return []
    if total <= chunk_size:
        return [(0, total)]  # the typical case: <= 3000 chars = 1 chunk (brief §4)

    starts: list[int] = []
    start = 0
    while True:
        starts.append(start)
        if start + chunk_size >= total:
            break
        start += chunk_size - chunk_overlap

    tail_len = total - starts[-1]
    tail_merged = len(starts) > 1 and tail_len < chunk_min_target
    if tail_merged:
        starts.pop()
        # A merged tail: the previous window extends to the end of the text (below).
    ends = [min(s + chunk_size, total) for s in starts]
    if tail_merged:
        ends[-1] = total
    return list(zip(starts, ends, strict=True))


def split_text(
    text: str,
    *,
    chunk_size: int,
    chunk_overlap: int,
    chunk_min_target: int,
) -> list[Chunk]:
    """Split a text into chunks by tokens: window/overlap/minimum tail.

            An empty text -> an empty list (the note is validated higher up the stack,
            but the pure function returns an honest result). Neighboring chunks
            overlap by chunk_overlap tokens - the seams never cut a fact crossing the
            border.

    """
    tokens = encoding().encode(text)
    windows = token_windows(
        len(tokens),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        chunk_min_target=chunk_min_target,
    )
    return [
        Chunk(text=encoding().decode(tokens[start:end]), tokens=end - start)
        for start, end in windows
    ]