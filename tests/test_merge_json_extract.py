"""LSBN-D-13: the tolerant extraction of merge/split parts (live flint answers)."""

import pytest

from app.services.summary import SummaryError, SummaryService


def _parts(answer):
    return SummaryService._extract_json(answer)


def test_plain_object():
    got = _parts('{"parts": [{"title": "t", "text": "x"}]}')
    assert got == {"parts": [{"title": "t", "text": "x"}]}


def test_code_fence():
    answer = '```json\n{"parts": [{"title": "t", "text": "x"}]}\n```'
    assert _parts(answer)["parts"][0]["text"] == "x"


def test_trailing_prose_with_braces():
    # the live case: the object is followed by reasoning that contains braces
    answer = (
        '{"parts": [{"title": "t", "text": "x"}]}\n'
        "Note: I merged {both} texts and kept {all} facts."
    )
    assert _parts(answer)["parts"][0]["title"] == "t"


def test_two_objects_in_a_row():
    answer = (
        '{"parts": [{"title": "first", "text": "1"}]}\n'
        '{"parts": [{"title": "second", "text": "2"}]}'
    )
    assert _parts(answer)["parts"][0]["title"] == "first"


def test_bare_array_is_wrapped():
    answer = '[{"title": "t", "text": "x"}]'
    assert _parts(answer) == {"parts": [{"title": "t", "text": "x"}]}


def test_raw_quotes_inside_value():
    answer = '{"parts": [{"title": "t", "text": "the .113 "primary" slot"}]}'
    assert "primary" in _parts(answer)["parts"][0]["text"]


def test_brace_inside_string_does_not_break_scan():
    answer = '{"parts": [{"title": "t", "text": "cfg {a: 1} stays"}]}'
    assert _parts(answer)["parts"][0]["text"] == "cfg {a: 1} stays"


def test_garbage_still_refused():
    with pytest.raises(SummaryError):
        _parts("I cannot do that.")


class _Settings:
    max_note_chars = 5000


def test_size_hint_counts_parts():
    service = SummaryService.__new__(SummaryService)
    service._settings = _Settings()
    hint = service._size_hint(19_329)
    assert "19329" in hint
    assert "at least 4 part" in hint  # ceil(19329 / 5000) = 4
    assert "5000" in hint
