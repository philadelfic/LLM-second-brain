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


# --- LSBN-D-13, решение О. 11.11.2026: одна заметка = один факт --------

def _facts(text, parts):
    from app.services.summary import facts_coverage

    return facts_coverage(text, parts)


def test_facts_coverage_keeps_numbers_and_paths():
    source = "Слот на 192.168.3.112, лимит 5000, таймаут 3600 с, версия 3.5.1"
    parts = ["Слот на 192.168.3.112", "Лимит 5000 символов", "Таймаут 3600 с, версия 3.5.1"]
    assert _facts(source, parts) == 1.0


def test_facts_coverage_allows_rephrasing_words():
    source = "Заметку нарезает модель, лимит 5000 символов"
    parts = ["Нарезка выполняется моделью. Предел — 5000 символов"]
    assert _facts(source, parts) == 1.0


def test_facts_coverage_catches_lost_number():
    source = "Лимит 5000 символов, таймаут 3600 с"
    parts = ["Лимит 5000 символов"]
    assert _facts(source, parts) < 1.0


def test_split_uses_fact_check_not_word_check():
    from app.services.prompts import summary_split_system_prompt

    prompt = summary_split_system_prompt(5000)
    assert "ATOMIC FACTS" in prompt
    assert "one fact per" in prompt
    assert "5000" in prompt


# --- вердикт судьи по нарезке (решение О. 11.11.2026) -------------------

def _judge_with(answer):
    import httpx

    from app.services.judge import JudgeService
    from tests.test_summary_service import make_settings

    class _MP:  # минимальная замена monkeypatch: настройки по умолчанию
        def setenv(self, *_a, **_k):
            pass

    settings = make_settings(_MP())
    body = {"message": {"content": answer, "role": "assistant"}, "done": True}

    def handler(request):
        return httpx.Response(200, json=body)

    return JudgeService(settings, transport=httpx.MockTransport(handler))


def test_verify_split_accepts_complete():
    judge = _judge_with("COMPLETE")
    assert judge.verify_split("исходник", ["факт 1", "факт 2"]) is None


def test_verify_split_reports_missing():
    judge = _judge_with("MISSING: потерян таймаут 3600")
    verdict = judge.verify_split("исходник", ["факт 1"])
    assert verdict is not None and "MISSING" in verdict


def test_verify_split_rejects_garbage():
    import pytest

    from app.services.judge import JudgeError

    judge = _judge_with("ну не знаю")
    with pytest.raises(JudgeError):
        judge.verify_split("исходник", ["факт"])


def test_facts_coverage_folds_number_separators():
    """«5 000» и «5000» — один и тот же факт (живой случай покрытия 0.72)."""
    source = "Лимит 5000 символов, таймаут 3600 с"
    parts = ["Лимит 5 000 символов", "таймаут 3 600 с"]
    assert _facts(source, parts) == 1.0


def test_split_prompt_forbids_fragments():
    from app.services.prompts import summary_split_system_prompt

    prompt = summary_split_system_prompt(5000)
    assert "never a bare heading" in prompt
    assert "shorter than 100 characters is a fragment" in prompt


def test_split_verdict_prompt_checks_self_containment():
    from app.services.judge import SPLIT_VERDICT_SYSTEM

    assert "self-contained" in SPLIT_VERDICT_SYSTEM
    assert "FRAGMENTS" in SPLIT_VERDICT_SYSTEM
