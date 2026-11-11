"""#32 (3.2.0): the Russian word-form stemmer and the context closeness metric.

    ARCH §6.6/§13.6: context clusters are merged arithmetically — normalization +
    one's own light stemmer of Russian endings (`token_stem`) + the area's
    trigram metric. No external morphology (`pymorphy2`) is introduced; the
    stemmer is deterministic and covers "project/project-in/project-of" forms.
"""

from __future__ import annotations

from app.services.terms_normalize import (
    STEM_SUFFIXES,
    context_similarity,
    stem_overlap,
    token_stem,
)


def test_stem_russian_inflections() -> None:
    """The inflected forms → one stem; a short word is not cut."""
    assert token_stem("проект") == token_stem("проекте") == token_stem("проекта")
    assert token_stem("проект") == "проект"
    # The 4-char form — the remainder after cutting the ending is shorter than 4 → the word as is.
    assert token_stem("дома") == "дома"
    assert token_stem("ДОМА") == "дома"  # the key's normalization (lower)
    # yo normalizes to ye before cutting the ending.
    assert token_stem("ружьё") == token_stem("ружье")
    # The endings' order — by length descending (the longest match first).
    lengths = [len(suffix) for suffix in STEM_SUFFIXES]
    assert lengths == sorted(lengths, reverse=True)


def test_similarity_trigram_and_stem() -> None:
    """Close-in-wording contexts ≥ the threshold; different ones — below."""
    assert context_similarity("проект разработки", "проекте разработки") >= 0.75
    # The stem match: the forms' trigrams are lower; the stemmer — 1.0.
    assert context_similarity("проекта разработки", "проектов разработки") >= 0.75
    assert context_similarity("бухгалтерия", "проект") < 0.75
    assert stem_overlap("", "проект") == 0.0
