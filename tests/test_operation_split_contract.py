"""The `split` operation (release 3.9.0, LSBN-S-11) — the answer contract
`parts-by-delimiter` and the defect LSBN-D-17 (the micro-parts).

The split of a long note is the operation with the NEW answer contract: in the
`operation` mode the model answers with free text, the parts are separated by
a delimiter line (a code constant), and the code makes the structure — no
JSON, no titles in the answer (the titles are a separate `title` call, the
owner's decision Q1 of 15.11.2026). The `legacy` mode keeps the historical
JSON contract unchanged: the rollback of the release is the flag
(`SLOT_MODEL=legacy`).

The criteria of the step:
  1. the `operation` mode accepts the free text with the delimiter;
  2. the delimiter is recognized only as a separate line, the empty segments
     are ignored, an answer without parts is off-contract;
  3. the `legacy` contract did not change (the snapshot below was taken on the
     base commit `1ce7303`, before this step);
  4. the title of a part comes from the `title` operation;
  5. D-17 is closed: a scrap (a part under the code's threshold) is not
     written — the follow-up call, and then `MergeIncompleteError`;
  6. the completeness is not weakened: the lost facts are caught by the
     coverage check and lead to a follow-up, not to a silent write;
  7. the `migration` job is gated by the `split` operation and its refusals
     go to the breaker of the `split` operation; the `legacy` mode keeps the
     `summary` slot.

A unit without a network: the models are fakes, the split service is checked
on the real `SummaryService` over `httpx.MockTransport`.
"""

from __future__ import annotations

import asyncio
import json
import statistics

import httpx
import pytest

from app.config import get_settings
from app.services.migration import MIGRATION_JOB, build_migration_job
from app.services.notes import NoteService
from app.services.operations import OPERATION_REGISTRY
from app.services.slot_outcome import SlotOutcome
from app.services.summary import (
    SPLIT_FACTS_FLOOR,
    SPLIT_FACTS_MIN,
    SPLIT_PART_MIN_CHARS,
    SPLIT_CONTRACT_DELIMITER,
    SPLIT_CONTRACT_JSON,
    SPLIT_PARTS_DELIMITER,
    MergeIncompleteError,
    SummaryError,
    SummaryService,
    facts_coverage,
    split_contract,
)
from app.services.worker import BackgroundWorker
from app.storage.db import init_db, session, transaction
from app.storage.migration import MIGRATION_NOTES_TABLE
from fakes import FixedSummarizer, HashEmbedder

#: The note limit of the test build (the parts must fit it; the source needs
#: two parts at this limit).
LIMIT = 600

#: The source of the split: a long note with numbers and dates (the hard
#: tokens of the fact check).
SOURCE = (
    "Ретроспектива продукта назначена на 12 сентября 2026, обсуждали 3 релиза "
    "и бюджет на 4 квартала. В фолоумапе обновление роадстрана до осени: версия "
    "2.1 выходит 15 октября 2026, миграция базы занимает 40 минут. Решили "
    "собрать 5 интервью и перенести демо на 20 ноября 2026, ответственные — "
    "Мария и Павел, сроки согласовать до 3 октября 2026. Ещё риски: лимит 10 "
    "пользователей на демо и 2 недели на миграцию данных, бюджет 500 тысяч "
    "рублей на инфраструктуру, релиз 1.0 закрываем 30 ноября 2026. Отдельно "
    "обсудили безопасность: доступ к 6 окружениям, журнал изменений за 90 дней "
    "и обязательный отчёт по 2 инцидентам закрываем к 8 октября 2026."
)

#: The two parts of the happy path: every number of the source is kept (the
#: fact coverage is 1.0), every part is over the minimum length of a part.
PART_ONE = (
    "Ретроспектива продукта назначена на 12 сентября 2026: обсуждали релизы и "
    "бюджет на кварталы, в фолоумапе обновление роадстрана до осени. Версия 2.1 "
    "выходит 15 октября 2026, миграция базы занимает 40 минут и идёт по шагам."
)
PART_TWO = (
    "Решили собрать 5 интервью и перенести демо на 20 ноября 2026, "
    "ответственные — Мария и Павел, сроки согласовать до 3 октября 2026. Риски: "
    "лимит 10 пользователей на демо и 2 недели на миграцию данных, бюджет 500 "
    "тысяч рублей, релиз 1.0 закрываем 30 ноября 2026. Отдельно обсудили "
    "безопасность: доступ к 6 окружениям, журнал изменений за 90 дней."
)

#: A short line that keeps the hard tokens of the source (the fact coverage
#: of a fixture must stay above the completeness threshold).
FACTS_LINE = (
    "Ретроспектива 12 сентября 2026, версия 2.1 выходит 15 октября 2026, "
    "миграция 40 минут, 5 интервью, демо 20 ноября 2026, сроки до 3 октября "
    "2026, лимит 10, 2 недели, бюджет 500 тысяч, релиз 1.0 30 ноября 2026, "
    "6 окружений и журнал изменений за 90 дней."
)

#: The answer of the new contract: the parts separated by the delimiter line.
FREE_TEXT_ANSWER = f"{PART_ONE}\n{SPLIT_PARTS_DELIMITER}\n{PART_TWO}"

#: The answer of the legacy contract: the JSON object with the parts.
JSON_ANSWER = json.dumps(
    {
        "parts": [
            {"title": "Ретроспектива и релиз", "text": PART_ONE},
            {"title": "Интервью и риски", "text": PART_TWO},
        ]
    },
    ensure_ascii=False,
)

#: A short JSON answer (one part) — for the parser's factual behaviour in the
#: `operation` mode: the JSON is not parsed as JSON any more.
JSON_ANSWER_SHORT = json.dumps(
    {"parts": [{"title": "Короткая", "text": FACTS_LINE}]},
    ensure_ascii=False,
)

#: The `legacy` contour snapshot — taken on the base commit `1ce7303` BEFORE
#: this step (the `legacy` mode must not change in any respect): the contract
#: of the operation, the parsed parts, the calls of the model, the system
#: prompt, the follow-up payload and the refusal of a part without a title.
LEGACY_SPLIT_CONTOUR: dict = {
    'parts': [['Ретроспектива и релиз', 'Ретроспектива продукта назначена на 12 сентября 2026: обсуждали релизы и бюджет на кварталы, в фолоумапе обновление роадстрана до осени. Версия 2.1 выходит 15 октября 2026, миграция базы занимает 40 минут и идёт по шагам.'], ['Интервью и риски', 'Решили собрать 5 интервью и перенести демо на 20 ноября 2026, ответственные — Мария и Павел, сроки согласовать до 3 октября 2026. Риски: лимит 10 пользователей на демо и 2 недели на миграцию данных, бюджет 500 тысяч рублей, релиз 1.0 закрываем 30 ноября 2026. Отдельно обсудили безопасность: доступ к 6 окружениям, журнал изменений за 90 дней.']],
    'calls': 1,
    'system': 'You are given ONE long note. Break it into ATOMIC FACTS: one fact per note (part). A part is a complete self-contained STATEMENT of one fact — one to three sentences, never a bare heading, never a fragment, never a list of items, never a phrase cut in the middle. A part shorter than 100 characters is a fragment and is wrong.  Each part states exactly one self-contained fact and carries ITS OWN names, numbers, dates, statuses, configs and paths - readable on its own, without references to the source or to other parts. Do not copy whole paragraphs and do not put several facts into one part; do not add anything of your own; do not lose a single fact of the source. Every part must stay well under the 600-character note limit. Copy numbers, names, dates, versions, paths and commands EXACTLY as they are written in the source — do not round, do not reformat, do not split a number with a space, do not translate them; only the connecting prose may be tightened. Give each part its own `title`: a meaningful name of at most 5 words. Write coherently, no headings, intros, quotes, or explanations. Respond in the language of the note. Answer with exactly one JSON object — no markdown fences, no text before or after it, no trailing commas. Escape every double quote inside a value in JSON style. Schema: {"parts": [{"title": "...", "text": "..."}, ...]}',
    'user': 'TEXT:\nРетроспектива продукта назначена на 12 сентября 2026, обсуждали 3 релиза и бюджет на 4 квартала. В фолоумапе обновление роадстрана до осени: версия 2.1 выходит 15 октября 2026, миграция базы занимает 40 минут. Решили собрать 5 интервью и перенести демо на 20 ноября 2026, ответственные — Мария и Павел, сроки согласовать до 3 октября 2026. Ещё риски: лимит 10 пользователей на демо и 2 недели на миграцию данных, бюджет 500 тысяч рублей на инфраструктуру, релиз 1.0 закрываем 30 ноября 2026. Отдельно обсудили безопасность: доступ к 6 окружениям, журнал изменений за 90 дней и обязательный отчёт по 2 инцидентам закрываем к 8 октября 2026.\n\nSIZE: the source is 639 characters; the note limit is 600. Return at least 2 part(s), each within 600 characters. Pack the facts DENSELY: a part well under the limit means wasted notes — merge related facts into one part unless a fact stands completely on its own. Keep every number, date, version, path and command verbatim; only the connecting prose may be tightened. Dropping, shortening or summarizing the content is a failure.',
    'followup_user': 'TEXT:\nРетроспектива продукта назначена на 12 сентября 2026, обсуждали 3 релиза и бюджет на 4 квартала. В фолоумапе обновление роадстрана до осени: версия 2.1 выходит 15 октября 2026, миграция базы занимает 40 минут. Решили собрать 5 интервью и перенести демо на 20 ноября 2026, ответственные — Мария и Павел, сроки согласовать до 3 октября 2026. Ещё риски: лимит 10 пользователей на демо и 2 недели на миграцию данных, бюджет 500 тысяч рублей на инфраструктуру, релиз 1.0 закрываем 30 ноября 2026. Отдельно обсудили безопасность: доступ к 6 окружениям, журнал изменений за 90 дней и обязательный отчёт по 2 инцидентам закрываем к 8 октября 2026.\n\nSIZE: the source is 639 characters; the note limit is 600. Return at least 2 part(s), each within 600 characters. Pack the facts DENSELY: a part well under the limit means wasted notes — merge related facts into one part unless a fact stands completely on its own. Keep every number, date, version, path and command verbatim; only the connecting prose may be tightened. Dropping, shortening or summarizing the content is a failure.\n\nYOUR ANSWER:\n{"parts": [{"title": "T", "text": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"}]}\n\npart 1 exceeds the 600-character note limit — split it into several self-contained parts\nReturn the FULL corrected set of parts as one JSON object.',
    'refusal': 'split: у части нет title или текста',
}


def _settings(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    mode: str = "operation",
    limit: int = LIMIT,
):
    """The test settings: own database, the slot model, the note limit."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "notes.db"))
    monkeypatch.setenv("MAX_NOTE_CHARS", str(limit))
    # The pause opens at the first failure (the breaker of the operation).
    monkeypatch.setenv("SLOT_PAUSE_AFTER", "1")
    if mode == "operation":
        monkeypatch.setenv("SLOT_MODEL", "operation")
        # The `operation` mode: the slot layer does not take part in the chain
        # (the design, B1) — the operations have their own env.
        monkeypatch.setenv("SPLIT_MODEL", "test-split-model")
        monkeypatch.setenv("SPLIT_BASE_URL", "http://127.0.0.1:1")
        monkeypatch.setenv("TITLE_MODEL", "test-title-model")
        monkeypatch.setenv("TITLE_BASE_URL", "http://127.0.0.1:1")
    else:
        monkeypatch.delenv("SLOT_MODEL", raising=False)
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    return settings


def _ok_body(content: str) -> dict:
    """The answer of the model (`/api/chat`)."""
    return {
        "model": "test-summary-model",
        "created_at": "2026-11-15T12:00:00Z",
        "message": {"role": "assistant", "content": content},
        "done": True,
    }


class _Script:
    """The scripted answers of the transport: one per call (the last repeats)."""

    def __init__(self, answers: list) -> None:
        self.answers = list(answers)
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        answer = (
            self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        )
        if isinstance(answer, Exception):
            raise answer
        if isinstance(answer, httpx.Response):
            return answer
        return httpx.Response(200, json=_ok_body(answer))

    @property
    def calls(self) -> int:
        return len(self.requests)


def _service(settings, answers: list) -> tuple[SummaryService, _Script]:
    """The split service over a scripted transport."""
    script = _Script(answers)
    service = SummaryService(
        settings, transport=httpx.MockTransport(script.handler)
    )
    return service, script


def _payload(request: httpx.Request) -> dict:
    """The body of the request."""
    return json.loads(request.read().decode())


def _system(request: httpx.Request) -> str:
    """The system prompt of the request."""
    for message in _payload(request)["messages"]:
        if message["role"] == "system":
            return str(message["content"])
    return ""


def _user(request: httpx.Request) -> str:
    """The user message of the request."""
    for message in _payload(request)["messages"]:
        if message["role"] == "user":
            return str(message["content"])
    return ""


# --- criterion 1: the free text with the delimiter ---------------------------


def test_the_operation_mode_split_accepts_the_free_text(tmp_path, monkeypatch):
    """The answer of the new contract: the free text split by the delimiter."""
    settings = _settings(tmp_path, monkeypatch, mode="operation")
    service, script = _service(
        settings, [FREE_TEXT_ANSWER, "Первый заголовок", "Второй заголовок"]
    )
    result = service.split(SOURCE)
    assert [part.text for part in result.parts] == [PART_ONE, PART_TWO]
    assert script.calls == 3  # the split + a title call per part
    assert [part.title for part in result.parts] == [
        "Первый заголовок",
        "Второй заголовок",
    ]


def test_the_operation_mode_prompt_names_the_delimiter(tmp_path, monkeypatch):
    """The prompt of the new contract: the delimiter verbatim, no JSON."""
    settings = _settings(tmp_path, monkeypatch, mode="operation")
    service, script = _service(settings, [FREE_TEXT_ANSWER])
    service.split(SOURCE)
    system = _system(script.requests[0])
    assert SPLIT_PARTS_DELIMITER in system
    assert "JSON" not in system
    assert '"parts"' not in system
    # The dense packing instead of "one fact per part" (the root of D-17).
    assert "DENSELY" in system
    assert "ATOMIC FACTS" not in system


def test_the_json_answer_is_not_required_in_the_operation_mode(tmp_path, monkeypatch):
    """The factual behaviour of the parser: the JSON answer is plain text."""
    settings = _settings(tmp_path, monkeypatch, mode="operation")
    service, _script = _service(settings, [JSON_ANSWER_SHORT])
    result = service.split(SOURCE)
    # The JSON is not parsed as JSON any more: it is one part of free text.
    assert len(result.parts) == 1
    assert result.parts[0].text == JSON_ANSWER_SHORT


def test_the_contract_of_the_operation_comes_from_the_registry(tmp_path, monkeypatch):
    """The contract of the `operation` mode — the registry's one."""
    settings = _settings(tmp_path, monkeypatch, mode="operation")
    assert OPERATION_REGISTRY["split"].contract == SPLIT_CONTRACT_DELIMITER
    assert split_contract(settings) == SPLIT_CONTRACT_DELIMITER
    legacy = _settings(tmp_path, monkeypatch, mode="legacy")
    assert split_contract(legacy) == SPLIT_CONTRACT_JSON


# --- criterion 2: the delimiter is a line of its own --------------------------


def test_the_delimiter_inside_a_line_is_not_a_separator(tmp_path, monkeypatch):
    """The delimiter as a part of a line does not split the answer."""
    settings = _settings(tmp_path, monkeypatch, mode="operation")
    text = FACTS_LINE.replace(
        "миграция 40 минут",
        f"миграция 40 минут {SPLIT_PARTS_DELIMITER} внутри строки",
    )
    service, _script = _service(settings, [text])
    result = service.split(SOURCE)
    assert len(result.parts) == 1
    assert result.parts[0].text == text


def test_the_empty_segments_are_ignored(tmp_path, monkeypatch):
    """The empty segments between the delimiters are skipped."""
    settings = _settings(tmp_path, monkeypatch, mode="operation")
    answer = (
        f"{SPLIT_PARTS_DELIMITER}\n{PART_ONE}\n{SPLIT_PARTS_DELIMITER}\n"
        f"{SPLIT_PARTS_DELIMITER}\n{PART_TWO}\n{SPLIT_PARTS_DELIMITER}"
    )
    service, _script = _service(settings, [answer])
    result = service.split(SOURCE)
    assert [part.text for part in result.parts] == [PART_ONE, PART_TWO]


def test_the_answer_without_parts_is_off_contract(tmp_path, monkeypatch):
    """No part at all — the refusal of the contract (off-contract)."""
    settings = _settings(tmp_path, monkeypatch, mode="operation")
    service, _script = _service(settings, [SPLIT_PARTS_DELIMITER])
    with pytest.raises(SummaryError) as excinfo:
        service.split(SOURCE)
    assert excinfo.value.outcome is SlotOutcome.OFF_CONTRACT


# --- criterion 3: the legacy contract did not change --------------------------


def test_the_legacy_split_parses_the_json_and_requires_a_title(tmp_path, monkeypatch):
    """The `legacy` mode: the JSON answer, the title of every part."""
    settings = _settings(tmp_path, monkeypatch, mode="legacy")
    service, script = _service(settings, [JSON_ANSWER])
    result = service.split(SOURCE)
    assert [part.text for part in result.parts] == [PART_ONE, PART_TWO]
    assert [part.title for part in result.parts] == [
        "Ретроспектива и релиз",
        "Интервью и риски",
    ]
    assert script.calls == 1  # no title call: the title is in the answer


def test_the_legacy_split_refuses_a_part_without_a_title(tmp_path, monkeypatch):
    """The `legacy` mode: a part without a title — the refusal."""
    settings = _settings(tmp_path, monkeypatch, mode="legacy")
    answer = json.dumps({"parts": [{"text": PART_ONE}]}, ensure_ascii=False)
    service, _script = _service(settings, [answer])
    with pytest.raises(SummaryError) as excinfo:
        service.split(SOURCE)
    assert excinfo.value.outcome is SlotOutcome.OFF_CONTRACT


def _legacy_contour(settings, monkeypatch) -> dict:
    """The `legacy` contour: the snapshot of the behaviour (the base commit)."""
    service, script = _service(settings, [JSON_ANSWER])
    result = service.split(SOURCE)
    contour = {
        "parts": [[part.title, part.text] for part in result.parts],
        "calls": script.calls,
        "system": _system(script.requests[0]),
        "user": _user(script.requests[0]),
    }
    # The follow-up of the `legacy` contract: the payload of the second call.
    long_part = "x" * (LIMIT + 10)
    long_answer = json.dumps(
        {"parts": [{"title": "T", "text": long_part}]}, ensure_ascii=False
    )
    followup, followup_script = _service(settings, [long_answer, long_answer])
    with pytest.raises(MergeIncompleteError):
        followup.split(SOURCE)
    contour["followup_user"] = _user(followup_script.requests[1])
    # The refusal of a part without a title.
    broken, _broken_script = _service(
        settings, [json.dumps({"parts": [{"text": PART_ONE}]}, ensure_ascii=False)]
    )
    try:
        broken.split(SOURCE)
        refusal = "no refusal"
    except SummaryError as exc:
        refusal = str(exc)
    contour["refusal"] = refusal
    return contour


def test_the_legacy_split_contour_snapshot_is_unchanged(tmp_path, monkeypatch):
    """The `legacy` contour is exactly the snapshot of the base commit."""
    settings = _settings(tmp_path, monkeypatch, mode="legacy")
    assert _legacy_contour(settings, monkeypatch) == LEGACY_SPLIT_CONTOUR


# --- criterion 4: the title of a part is a `title` call -----------------------


def test_the_title_of_every_part_comes_from_the_title_operation(tmp_path, monkeypatch):
    """The new contract: one `title` call per part."""
    settings = _settings(tmp_path, monkeypatch, mode="operation")
    service, script = _service(
        settings,
        [
            FREE_TEXT_ANSWER,
            "Ретроспектива продукта назначена на 12 сентября",
            "Интервью и риски",
        ],
    )
    result = service.split(SOURCE)
    assert [part.title for part in result.parts] == [
        "Ретроспектива продукта назначена на 12",  # trimmed to TITLE_MAX_WORDS
        "Интервью и риски",
    ]
    assert script.calls == 3  # the split + two titles
    # The title call carries the text of its part.
    assert PART_ONE in _user(script.requests[1])
    assert PART_TWO in _user(script.requests[2])


def test_the_legacy_title_comes_from_the_model_answer(tmp_path, monkeypatch):
    """The `legacy` mode: the title is in the answer, no extra call."""
    settings = _settings(tmp_path, monkeypatch, mode="legacy")
    service, script = _service(settings, [JSON_ANSWER])
    result = service.split(SOURCE)
    assert result.parts[0].title == "Ретроспектива и релиз"
    assert script.calls == 1


def test_the_title_failure_is_a_failure_of_the_split(tmp_path, monkeypatch):
    """The refusal of the title generator — the refusal of the split."""
    settings = _settings(tmp_path, monkeypatch, mode="operation")
    service, _script = _service(
        settings, [FREE_TEXT_ANSWER, httpx.Response(500, json={"error": "down"})]
    )
    with pytest.raises(SummaryError):
        service.split(SOURCE)


def test_the_part_without_a_title_goes_to_the_title_queue(tmp_path, monkeypatch):
    """The empty title of a part — NULL: the note waits for the title job."""
    settings = _settings(tmp_path, monkeypatch, mode="operation")
    notes = NoteService(settings)
    saved = notes.save_migration_parts(999, "default", [("", PART_ONE)])
    with session(settings) as conn:
        row = conn.execute(
            "SELECT title FROM notes WHERE id = ?",
            (saved["parts"][0]["id"],),
        ).fetchone()
    assert row["title"] is None


# --- criterion 5: D-17 — the micro-parts are not written ---------------------


def test_the_short_part_triggers_the_followup(tmp_path, monkeypatch):
    """A part under the code's threshold — the follow-up call."""
    settings = _settings(tmp_path, monkeypatch, mode="operation")
    scrap = "Короткий обрывок."
    service, script = _service(settings, [f"{scrap}\n{SPLIT_PARTS_DELIMITER}\n{PART_TWO}"])
    with pytest.raises(MergeIncompleteError):
        service.split(SOURCE)
    assert script.calls == 2  # the main call + the follow-up


def test_the_scrap_is_not_written(tmp_path, monkeypatch):
    """The scrap never reaches the notes: the split does not converge."""
    settings = _settings(tmp_path, monkeypatch, mode="operation")
    scrap = "Короткий обрывок."
    service, _script = _service(
        settings,
        [
            f"{scrap}\n{SPLIT_PARTS_DELIMITER}\n{PART_TWO}",
            f"{scrap}\n{SPLIT_PARTS_DELIMITER}\n{PART_TWO}",
        ],
    )
    with pytest.raises(MergeIncompleteError):
        service.split(SOURCE)


def test_the_median_length_of_the_parts_is_not_below_the_threshold():
    """The fixture of the step: the median length of a part ≥ the threshold."""
    lengths = sorted(len(part) for part in (PART_ONE, PART_TWO))
    assert statistics.median(lengths) >= SPLIT_PART_MIN_CHARS
    assert facts_coverage(SOURCE, [PART_ONE, PART_TWO]) == 1.0


# --- criterion 6: the completeness is not weakened ---------------------------


def test_the_lost_facts_lead_to_the_followup(tmp_path, monkeypatch):
    """The loss of the facts — the follow-up call, not a silent write."""
    settings = _settings(tmp_path, monkeypatch, mode="operation")
    thin = (
        "Ретроспектива продукта назначена на 12 сентября 2026, обсуждали релизы "
        "и бюджет на кварталы, в фолоумапе обновление роадстрана до осени, а "
        "безопасность и риски обсудили отдельно в следующей части заметки."
    )
    assert facts_coverage(SOURCE, [thin]) < SPLIT_FACTS_FLOOR
    service, script = _service(settings, [thin, thin])
    with pytest.raises(MergeIncompleteError):
        service.split(SOURCE)
    assert script.calls == 2  # the coverage check — the follow-up


def test_the_coverage_thresholds_are_not_lowered():
    """The thresholds of the completeness checks are as they were."""
    assert SPLIT_FACTS_MIN == 0.0
    assert SPLIT_FACTS_FLOOR == 0.3


# --- criterion 7: the `migration` job works the `split` operation -------------


def _migration_state(settings, tmp_path) -> None:
    """The migration state: a previous base with one note waiting for a split."""
    previous = tmp_path / "notes.2000.db"
    import sqlite3

    conn = sqlite3.connect(previous)
    conn.execute(
        "CREATE TABLE notes (id INTEGER PRIMARY KEY, text TEXT, namespace TEXT, "
        "expires_at TEXT, deleted_at TEXT)"
    )
    conn.execute("INSERT INTO notes (id, text) VALUES (1, ?)", (SOURCE,))
    conn.commit()
    conn.close()
    with session(settings) as conn, transaction(conn):
        conn.execute(
            "INSERT INTO meta (key, value) VALUES ('migration.state', ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (json.dumps({"phase": "split", "prev_file": previous.name, "id_floor": 2}),),
        )
        conn.execute(
            f"CREATE TABLE IF NOT EXISTS {MIGRATION_NOTES_TABLE} ("
            "old_id INTEGER PRIMARY KEY, status TEXT NOT NULL, new_ids TEXT, "
            "attempts INTEGER NOT NULL DEFAULT 0, reason TEXT, "
            "parts_json TEXT, attention INTEGER NOT NULL DEFAULT 0, "
            "updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')))"
        )
        conn.execute(
            f"INSERT INTO {MIGRATION_NOTES_TABLE} (old_id, status) VALUES (1, 'ожидает')"
        )


def _worker(settings, summarizer=None) -> BackgroundWorker:
    """A worker with the summarizer of the operation."""
    return BackgroundWorker(
        settings, HashEmbedder(8), summarizer or FixedSummarizer()
    )


def _failing_summarizer() -> FixedSummarizer:
    """The summarizer whose splitting is off-contract (the refusal)."""
    summarizer = FixedSummarizer()
    summarizer.split_mode = "off_contract"
    return summarizer


def test_the_migration_job_is_gated_by_the_split_operation(tmp_path, monkeypatch):
    """The `operation` mode: the job's key is the `split` operation."""
    settings = _settings(tmp_path, monkeypatch, mode="operation")
    worker = _worker(settings)
    spec = build_migration_job(worker, settings)
    assert spec is not None
    assert spec.slot == "split"
    assert spec.is_gated() is False
    # The pause of the breaker of the `split` operation stops the job.
    worker._breakers["split"].note_failure(SlotOutcome.OFF_CONTRACT)
    assert worker._breakers["split"].is_paused() is True
    assert spec.is_gated() is True
    assert spec.gate() is False


def test_the_refusal_of_the_migration_job_goes_to_the_split_breaker(tmp_path, monkeypatch):
    """The `operation` mode: the refusal of the splitting — the `split` breaker."""
    settings = _settings(tmp_path, monkeypatch, mode="operation")
    _migration_state(settings, tmp_path)
    worker = _worker(settings, _failing_summarizer())
    spec = build_migration_job(worker, settings)
    assert spec is not None
    asyncio.run(spec.process())
    breaker = worker._breakers["split"]
    assert breaker.snapshot()["last_outcome"] == "off_contract"
    assert breaker.is_paused() is True
    # The neighbouring operations are not touched.
    for name in ("summarize", "title", "merge", "classify"):
        assert worker._breakers[name].snapshot()["last_outcome"] is None


def test_the_legacy_migration_job_keeps_the_summary_slot(tmp_path, monkeypatch):
    """The `legacy` mode: the `summary` slot and the previous behaviour."""
    settings = _settings(tmp_path, monkeypatch, mode="legacy")
    _migration_state(settings, tmp_path)
    worker = _worker(settings, _failing_summarizer())
    spec = build_migration_job(worker, settings)
    assert spec is not None
    assert spec.slot == "summary"
    asyncio.run(spec.process())
    # The `legacy` mode: the job never touched the breaker.
    for breaker in worker._breakers.values():
        assert breaker.snapshot()["last_outcome"] is None
