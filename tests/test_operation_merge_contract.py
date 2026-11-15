"""The `merge` operation (release 3.9.0, LSBN-S-12) — the answer contract
`parts-by-delimiter` and the defect LSBN-D-13 (the merge parser demanded JSON
— on the live data the answer was `off_contract`).

The merge of a duplicate pair is the operation with the NEW answer contract: in
the `operation` mode the model answers with free text, the parts are separated
by a delimiter line (a code constant, the same one the split uses), and the
code makes the structure — no JSON, no titles in the answer (the code names the
parts: the first one keeps the title of the earlier note, the others wait for
the `title` queue — the owner's decision Q1 of 15.11.2026). The `legacy` mode
keeps the historical JSON contract unchanged: the rollback of the release is
the flag (`SLOT_MODEL=legacy`).

The criteria of the step:
  1. the `operation` mode accepts the free text with the delimiter;
  2. the JSON parser is not used in the `operation` mode (the `legacy` mode
     keeps the JSON repairs — the rollback must work);
  3. the completeness stays a CODE check: a lost fact leads to the follow-up,
     not to a silent write; the threshold of the coverage is not lowered;
  4. the title of the first part is the title of the earlier note — no model
     call is made;
  5. the other parts get their titles from the `title` queue;
  6. the `legacy` mode did not change (the snapshot below was taken on the base
     commit `86c35a8`, before this step);
  7. D-13 is closed: the live answer (free text) is parsed without
     `off_contract`.

A unit without a network: the models are fakes, the merge service is checked
on the real `SummaryService` over `httpx.MockTransport`.
"""

from __future__ import annotations

import json

import httpx
import pytest

from app.config import get_settings
from app.services.notes import NoteService
from app.services.operations import OPERATION_REGISTRY
from app.services.prompts import PromptRegistry
from app.services.slot_outcome import SlotOutcome
from app.services.summary import (
    CONTRACT_DELIMITER,
    CONTRACT_JSON,
    MERGE_COVERAGE_MIN,
    SPLIT_PARTS_DELIMITER,
    MergeIncompleteError,
    SummaryError,
    SummaryService,
    merge_coverage,
    operation_contract,
)
from app.services.worker import BackgroundWorker
from app.storage.db import init_db, session
from fakes import HashEmbedder, MergeFailingSummarizer

#: The note limit of the test build (the parts must fit it).
LIMIT = 600

#: The pair of the merge: two versions of the same note.
TEXT_A = (
    "Ретроспектива продукта назначена на 12 сентября 2026, обсуждали 3 релиза "
    "и бюджет на 4 квартала. В фолоумапе обновление роадстрана до осени: версия "
    "2.1 выходит 15 октября 2026, миграция базы занимает 40 минут."
)
TEXT_B = (
    "Решили собрать 5 интервью и перенести демо на 20 ноября 2026, "
    "ответственные — Мария и Павел, сроки согласовать до 3 октября 2026. Ещё "
    "риски: лимит 10 пользователей на демо и 2 недели на миграцию данных."
)

#: The two parts of the happy path: every significant token of the sources is
#: kept (the coverage check of the merge must pass).
PART_ONE = (
    TEXT_A
    + " Ретроспектива продукта и релиз 2.1 закрыты 15 октября 2026, миграция "
    "базы занимает 40 минут и идёт по шагам."
)
PART_TWO = (
    TEXT_B
    + " Демо перенесли на 20 ноября 2026, бюджет 500 тысяч рублей, риски "
    "зафиксированы."
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

#: The live pair of the prod merge (LSBN-D-13): two versions of one note.
LIVE_A = "Ежедневный бэкап кластера: проверяем снимок и журнал репликации утром."
LIVE_B = "Ежедневный бэкап кластера: проверяем снимок и журнал репликации вечером."

#: The live answer of the prod merge (LSBN-D-13): the prose of the model, not
#: a JSON object — the old parser refused it (`off_contract`).
LIVE_ANSWER = (
    "Ежедневный бэкап кластера: проверяем снимок и журнал репликации утром, "
    "а вечером — только журнал репликации."
)

#: The `legacy` contour snapshot — taken on the base commit `86c35a8` BEFORE
#: this step (the `legacy` mode must not change in any respect): the parsed
#: parts, the calls of the model, the system prompt, the user prompt, the
#: follow-up payload, the refusal and the written titles.
LEGACY_MERGE_CONTOUR: dict = {
    'parts': [
        [
            'Ретроспектива и релиз',
            'Ретроспектива продукта назначена на 12 сентября 2026, обсуждали 3 релиза и бюджет на 4 квартала. В фолоумапе обновление роадстрана до осени: версия 2.1 выходит 15 октября 2026, миграция базы занимает 40 минут. Ретроспектива продукта и релиз 2.1 закрыты 15 октября 2026, миграция базы занимает 40 минут и идёт по шагам.',
        ],
        [
            'Интервью и риски',
            'Решили собрать 5 интервью и перенести демо на 20 ноября 2026, ответственные — Мария и Павел, сроки согласовать до 3 октября 2026. Ещё риски: лимит 10 пользователей на демо и 2 недели на миграцию данных. Демо перенесли на 20 ноября 2026, бюджет 500 тысяч рублей, риски зафиксированы.',
        ],
    ],
    'calls': 1,
    'system': 'You have two versions of the same note. Merge them into one or several SELF-CONTAINED notes (parts): combine all information from both — facts, names, numbers, dates, statuses, configs and paths; state each fact once, omit repetitions; do not drop anything and do not add anything of your own — a shortened retelling is a failure, not a merge. Every part must be readable on its own, without references to other parts. Use ONE part only if the merged text fits the 600-character note limit; otherwise split the content into several parts, each within the same limit. Give each part its own `title`: a meaningful name of at most 5 words. Write coherently, no headings, intros, quotes, or explanations. Respond in the language of the notes. Answer with exactly one JSON object — no markdown fences, no text before or after it, no trailing commas. Escape every double quote inside a value in JSON style. Schema: {"parts": [{"title": "...", "text": "..."}, ...]}',
    'user': 'TEXT 1:\nРетроспектива продукта назначена на 12 сентября 2026, обсуждали 3 релиза и бюджет на 4 квартала. В фолоумапе обновление роадстрана до осени: версия 2.1 выходит 15 октября 2026, миграция базы занимает 40 минут.\n\nTEXT 2:\nРешили собрать 5 интервью и перенести демо на 20 ноября 2026, ответственные — Мария и Павел, сроки согласовать до 3 октября 2026. Ещё риски: лимит 10 пользователей на демо и 2 недели на миграцию данных.\n\nSIZE: the source is 411 characters; the note limit is 600. Return at least 1 part(s), each within 600 characters. Dropping, shortening or summarizing the content is a failure: every fact, name, number and date must be present in the parts.',
    'followup_user': 'TEXT 1:\nРетроспектива продукта назначена на 12 сентября 2026, обсуждали 3 релиза и бюджет на 4 квартала. В фолоумапе обновление роадстрана до осени: версия 2.1 выходит 15 октября 2026, миграция базы занимает 40 минут.\n\nTEXT 2:\nРешили собрать 5 интервью и перенести демо на 20 ноября 2026, ответственные — Мария и Павел, сроки согласовать до 3 октября 2026. Ещё риски: лимит 10 пользователей на демо и 2 недели на миграцию данных.\n\nSIZE: the source is 411 characters; the note limit is 600. Return at least 1 part(s), each within 600 characters. Dropping, shortening or summarizing the content is a failure: every fact, name, number and date must be present in the parts.\n\nYOUR ANSWER:\n{"parts": [{"title": "T", "text": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"}]}\n\npart 1 exceeds the 600-character note limit — split it into several self-contained parts\nReturn the FULL corrected set of parts as one JSON object.',
    'refusal': 'merge: у части нет title или текста',
    'written': [
        [
            3,
            'Ретроспектива и релиз',
        ],
        [
            4,
            'Интервью и риски',
        ],
    ],
    'merged': True,
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
        monkeypatch.setenv("MERGE_MODEL", "test-merge-model")
        monkeypatch.setenv("MERGE_BASE_URL", "http://127.0.0.1:1")
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


def _service(settings, answers: list, registry=None) -> tuple[SummaryService, _Script]:
    """The merge service over a scripted transport."""
    script = _Script(answers)
    service = SummaryService(
        settings, transport=httpx.MockTransport(script.handler), registry=registry
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


def test_the_operation_mode_merge_accepts_the_free_text(tmp_path, monkeypatch):
    """The answer of the new contract: the free text split by the delimiter."""
    settings = _settings(tmp_path, monkeypatch, mode="operation")
    service, script = _service(settings, [FREE_TEXT_ANSWER])
    result = service.merge(TEXT_A, TEXT_B, title="Ранняя заметка")
    assert [part.text for part in result.parts] == [PART_ONE, PART_TWO]
    assert script.calls == 1  # the merge call only — no title call
    assert [part.title for part in result.parts] == ["Ранняя заметка", ""]


def test_the_prompt_of_the_new_contract_names_the_delimiter(tmp_path, monkeypatch):
    """The prompt of the new contract: the delimiter verbatim, no JSON."""
    settings = _settings(tmp_path, monkeypatch, mode="operation")
    service, script = _service(settings, [FREE_TEXT_ANSWER])
    service.merge(TEXT_A, TEXT_B)
    system = _system(script.requests[0])
    assert SPLIT_PARTS_DELIMITER in system
    assert "JSON" not in system
    assert '"parts"' not in system
    # The requirements of self-sufficiency and completeness stay.
    assert "SELF-CONTAINED" in system
    assert "do not drop anything" in system


def test_the_json_answer_is_not_required_in_the_operation_mode(tmp_path, monkeypatch):
    """The factual behaviour of the parser: the JSON answer is plain text."""
    # The note limit is bigger here: the JSON answer is one part of free text.
    settings = _settings(tmp_path, monkeypatch, mode="operation", limit=2000)
    service, _script = _service(settings, [JSON_ANSWER])
    result = service.merge(TEXT_A, TEXT_B)
    # The JSON is not parsed as JSON any more: it is one part of free text.
    assert len(result.parts) == 1
    assert result.parts[0].text == JSON_ANSWER


def test_the_contract_of_the_merge_comes_from_the_registry(tmp_path, monkeypatch):
    """The contract of the `operation` mode — the registry's one."""
    settings = _settings(tmp_path, monkeypatch, mode="operation")
    assert OPERATION_REGISTRY["merge"].contract == CONTRACT_DELIMITER
    assert operation_contract(settings, "merge") == CONTRACT_DELIMITER
    legacy = _settings(tmp_path, monkeypatch, mode="legacy")
    assert operation_contract(legacy, "merge") == CONTRACT_JSON


def test_the_operator_file_wins_over_the_builtin_prompt(tmp_path, monkeypatch):
    """The file of the new key has priority over the built-in text."""
    settings = _settings(tmp_path, monkeypatch, mode="operation")
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    custom = "Сведи заметки в части и раздели их строкой === PART ===."
    (prompts / "summary_merge_system_delimiter.txt").write_text(
        custom, encoding="utf-8"
    )
    registry = PromptRegistry(prompts_dir=prompts, max_note_chars=LIMIT)
    service, script = _service(settings, [FREE_TEXT_ANSWER], registry=registry)
    service.merge(TEXT_A, TEXT_B)
    assert _system(script.requests[0]) == custom


def test_the_legacy_prompt_file_still_wins_and_the_new_key_does_not_leak(
    tmp_path, monkeypatch
):
    """The `legacy` mode: the file `summary_merge_system` still wins; the new
    key does not touch the legacy contract."""
    settings = _settings(tmp_path, monkeypatch, mode="legacy")
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    legacy_custom = "Сведи две заметки в одну."
    (prompts / "summary_merge_system.txt").write_text(legacy_custom, encoding="utf-8")
    (prompts / "summary_merge_system_delimiter.txt").write_text(
        "ПРОМПТ НОВОГО КОНТРАКТА", encoding="utf-8"
    )
    registry = PromptRegistry(prompts_dir=prompts, max_note_chars=LIMIT)
    service, script = _service(settings, [JSON_ANSWER], registry=registry)
    service.merge(TEXT_A, TEXT_B)
    assert _system(script.requests[0]) == legacy_custom


# --- criterion 2: the JSON parser is not used in the `operation` mode ---------


def test_the_json_repairs_are_not_used_in_the_operation_mode(
    tmp_path, monkeypatch
):
    """The `operation` mode: the JSON parser and its repairs are not called."""
    import app.services.summary as summary_module

    calls: list[str] = []
    original_escape = summary_module._escape_raw_quotes
    original_extract = summary_module.SummaryService._extract_json

    def spy_escape(text: str) -> str:
        calls.append("escape")
        return original_escape(text)

    def spy_extract(content: str):
        calls.append("extract")
        return original_extract(content)

    monkeypatch.setattr(summary_module, "_escape_raw_quotes", spy_escape)
    monkeypatch.setattr(summary_module.SummaryService, "_extract_json", spy_extract)
    settings = _settings(tmp_path, monkeypatch, mode="operation")
    service, _script = _service(settings, [FREE_TEXT_ANSWER])
    service.merge(TEXT_A, TEXT_B)
    assert calls == []  # the free text is not parsed and not repaired as JSON


def test_the_json_repairs_work_in_the_legacy_mode(tmp_path, monkeypatch):
    """The `legacy` mode: the JSON repairs are in place (the rollback)."""
    import app.services.summary as summary_module

    calls: list[str] = []
    original_escape = summary_module._escape_raw_quotes
    original_extract = summary_module.SummaryService._extract_json

    def spy_escape(text: str) -> str:
        calls.append("escape")
        return original_escape(text)

    def spy_extract(content: str):
        calls.append("extract")
        return original_extract(content)

    monkeypatch.setattr(summary_module, "_escape_raw_quotes", spy_escape)
    monkeypatch.setattr(summary_module.SummaryService, "_extract_json", spy_extract)
    settings = _settings(tmp_path, monkeypatch, mode="legacy")
    service, _script = _service(settings, [JSON_ANSWER])
    result = service.merge(TEXT_A, TEXT_B)
    assert len(result.parts) == 2
    assert "extract" in calls  # the JSON parser is on (the rollback must work)


# --- criterion 3: the completeness stays a code check ------------------------


def test_the_lost_fact_leads_to_the_followup(tmp_path, monkeypatch):
    """The loss of a fact — the follow-up call, not a silent write."""
    settings = _settings(tmp_path, monkeypatch, mode="operation")
    thin = (
        "Ретроспектива продукта назначена на 12 сентября 2026, обсуждали релизы "
        "и бюджет на кварталы, риски обсудили отдельно в следующей части заметки."
    )
    assert merge_coverage(f"{TEXT_A}\n{TEXT_B}", [thin]) < MERGE_COVERAGE_MIN
    service, script = _service(settings, [thin, thin])
    with pytest.raises(MergeIncompleteError):
        service.merge(TEXT_A, TEXT_B)
    assert script.calls == 2  # the coverage check — the follow-up


def test_the_followup_asks_for_the_same_format(tmp_path, monkeypatch):
    """The follow-up call names the SAME format as the main prompt."""
    settings = _settings(tmp_path, monkeypatch, mode="operation")
    thin = (
        "Ретроспектива продукта назначена на 12 сентября 2026, обсуждали релизы "
        "и бюджет на кварталы, риски обсудили отдельно в следующей части заметки."
    )
    service, script = _service(settings, [thin, thin])
    with pytest.raises(MergeIncompleteError):
        service.merge(TEXT_A, TEXT_B)
    followup = _user(script.requests[1])
    assert SPLIT_PARTS_DELIMITER in followup
    assert "one JSON object" not in followup


def test_the_coverage_threshold_is_not_lowered():
    """The threshold of the completeness check is the strict one."""
    assert MERGE_COVERAGE_MIN == 1.0


# --- criterion 4: the title of the first part is the earlier note's ----------


def test_the_first_part_keeps_the_title_of_the_earlier_note(tmp_path, monkeypatch):
    """The first part — the title of the earlier note, without a model call."""
    settings = _settings(tmp_path, monkeypatch, mode="operation")
    service, script = _service(settings, [FREE_TEXT_ANSWER])
    result = service.merge(TEXT_A, TEXT_B, title="Ранняя заметка")
    assert result.parts[0].title == "Ранняя заметка"
    assert result.parts[1].title == ""  # no title — the `title` queue
    assert script.calls == 1  # no title call in the merge path


def test_the_title_of_the_earlier_note_is_trimmed_to_the_limit(tmp_path, monkeypatch):
    """A long title of the earlier note — trimmed to TITLE_MAX_WORDS."""
    settings = _settings(tmp_path, monkeypatch, mode="operation")
    service, _script = _service(settings, [FREE_TEXT_ANSWER])
    result = service.merge(
        TEXT_A, TEXT_B, title="Ретроспектива продукта назначена на сентябрь"
    )
    assert len(result.parts[0].title.split()) == 5


def test_the_earlier_note_without_a_title_leaves_the_first_part_titleless(
    tmp_path, monkeypatch
):
    """The earlier note has no title — the first part waits for the queue too."""
    settings = _settings(tmp_path, monkeypatch, mode="operation")
    service, _script = _service(settings, [FREE_TEXT_ANSWER])
    result = service.merge(TEXT_A, TEXT_B, title=None)
    assert [part.title for part in result.parts] == ["", ""]


# --- criterion 5: the other parts get the title from the `title` queue -------


def _merge_job(worker: BackgroundWorker, settings, note_id: int, older_id: int):
    """A merge job in the queue — as the judge loop places it."""
    worker._create_job(
        "summary", "merge", note_id, payload=json.dumps({"older_id": older_id})
    )


def _worker(settings, summarizer) -> BackgroundWorker:
    """A worker with the given summarizer."""
    return BackgroundWorker(settings, HashEmbedder(8), summarizer)


def _merge_pair(settings) -> tuple[int, int]:
    """The pair of the merge: an earlier note with a title and a later one."""
    notes = NoteService(settings)
    older_id = notes.save(TEXT_A, title="Ранняя заметка")["id"]
    newer_id = notes.save(TEXT_B, title="Поздняя заметка")["id"]
    return older_id, newer_id


def _titles(settings) -> list[tuple[int, str | None]]:
    """The alive notes: their ids and titles."""
    with session(settings) as conn:
        rows = conn.execute(
            "SELECT id, title FROM notes WHERE deleted_at IS NULL ORDER BY id"
        ).fetchall()
    return [(int(row["id"]), row["title"]) for row in rows]


def test_the_part_without_a_title_is_written_with_null(tmp_path, monkeypatch):
    """The merge path: the first part keeps the title, the other one is NULL."""
    settings = _settings(tmp_path, monkeypatch, mode="operation")
    service, _script = _service(settings, [FREE_TEXT_ANSWER])
    worker = _worker(settings, service)
    older_id, newer_id = _merge_pair(settings)
    _merge_job(worker, settings, newer_id, older_id)
    assert worker.process_merge_pending() == 1
    # The sources are in the trash, the parts are the alive notes.
    with session(settings) as conn:
        alive = conn.execute(
            "SELECT COUNT(*) c FROM notes WHERE deleted_at IS NULL"
        ).fetchone()["c"]
        trashed = conn.execute(
            "SELECT COUNT(*) c FROM notes WHERE deleted_at IS NOT NULL"
        ).fetchone()["c"]
    assert alive == 2 and trashed == 2
    titles = [title for _note_id, title in _titles(settings)]
    assert sorted(title for title in titles if title) == ["Ранняя заметка"]
    assert titles.count(None) == 1  # the second part — NULL (the title queue)


def test_the_title_job_names_the_part_later(tmp_path, monkeypatch):
    """The part without a title is picked up by the `title` job."""
    settings = _settings(tmp_path, monkeypatch, mode="operation")
    service, script = _service(settings, [FREE_TEXT_ANSWER, "Заголовок части"])
    worker = _worker(settings, service)
    older_id, newer_id = _merge_pair(settings)
    _merge_job(worker, settings, newer_id, older_id)
    worker.process_merge_pending()
    # The queue of the title job: the note with a NULL title.
    with session(settings) as conn:
        queued = conn.execute(
            "SELECT COUNT(*) c FROM notes WHERE title IS NULL AND deleted_at IS NULL"
        ).fetchone()["c"]
    assert queued == 1
    assert worker.process_title_pending() == 1
    titles = [title for _note_id, title in _titles(settings)]
    assert "Заголовок части" in titles
    assert None not in titles  # the titleless part is named


# --- criterion 6: the legacy contract did not change --------------------------


def test_the_legacy_merge_parses_the_json_and_requires_a_title(tmp_path, monkeypatch):
    """The `legacy` mode: the JSON answer, the title of every part."""
    settings = _settings(tmp_path, monkeypatch, mode="legacy")
    service, script = _service(settings, [JSON_ANSWER])
    result = service.merge(TEXT_A, TEXT_B)
    assert [part.text for part in result.parts] == [PART_ONE, PART_TWO]
    assert [part.title for part in result.parts] == [
        "Ретроспектива и релиз",
        "Интервью и риски",
    ]
    assert script.calls == 1  # no title call: the title is in the answer


def test_the_legacy_merge_refuses_a_part_without_a_title(tmp_path, monkeypatch):
    """The `legacy` mode: a part without a title — the refusal."""
    settings = _settings(tmp_path, monkeypatch, mode="legacy")
    answer = json.dumps({"parts": [{"text": PART_ONE}]}, ensure_ascii=False)
    service, _script = _service(settings, [answer])
    with pytest.raises(SummaryError) as excinfo:
        service.merge(TEXT_A, TEXT_B)
    assert excinfo.value.outcome is SlotOutcome.OFF_CONTRACT


def _legacy_contour(settings) -> dict:
    """The `legacy` contour: the snapshot of the behaviour (the base commit)."""
    service, script = _service(settings, [JSON_ANSWER])
    result = service.merge(TEXT_A, TEXT_B)
    contour = {
        "parts": [[part.title, part.text] for part in result.parts],
        "calls": script.calls,
        "system": _system(script.requests[0]),
        "user": _user(script.requests[0]),
    }
    # The follow-up of the `legacy` contract: the payload of the second call.
    long_answer = json.dumps(
        {"parts": [{"title": "T", "text": "x" * (LIMIT + 10)}]}, ensure_ascii=False
    )
    followup, followup_script = _service(settings, [long_answer, long_answer])
    with pytest.raises(MergeIncompleteError):
        followup.merge(TEXT_A, TEXT_B)
    contour["followup_user"] = _user(followup_script.requests[1])
    # The refusal of a part without a title.
    broken, _broken_script = _service(
        settings, [json.dumps({"parts": [{"text": PART_ONE}]}, ensure_ascii=False)]
    )
    try:
        broken.merge(TEXT_A, TEXT_B)
        refusal = "no refusal"
    except SummaryError as exc:
        refusal = str(exc)
    contour["refusal"] = refusal
    # The write path: the parts are written with their titles.
    notes = NoteService(settings)
    older_id = notes.save("текст ранней заметки", title="Ранняя заметка")["id"]
    newer_id = notes.save("текст поздней заметки", title="Поздняя заметка")["id"]
    outcome = notes.merge_into_parts(
        older_id, newer_id, [(part.title, part.text) for part in result.parts]
    )
    with session(settings) as conn:
        rows = conn.execute(
            "SELECT id, title FROM notes WHERE deleted_at IS NULL ORDER BY id"
        ).fetchall()
    contour["written"] = [[int(row["id"]), row["title"]] for row in rows]
    contour["merged"] = outcome["merged"]
    return contour


def test_the_legacy_merge_contour_snapshot_is_unchanged(tmp_path, monkeypatch):
    """The `legacy` contour is exactly the snapshot of the base commit."""
    settings = _settings(tmp_path, monkeypatch, mode="legacy")
    assert _legacy_contour(settings) == LEGACY_MERGE_CONTOUR


# --- criterion 7: D-13 is closed ----------------------------------------------


def test_the_live_answer_is_parsed_without_off_contract(tmp_path, monkeypatch):
    """The live answer (free text, as on the prod) — no `off_contract`."""
    settings = _settings(tmp_path, monkeypatch, mode="operation")
    service, _script = _service(settings, [LIVE_ANSWER])
    result = service.merge(LIVE_A, LIVE_B)
    assert len(result.parts) == 1
    assert result.parts[0].text == LIVE_ANSWER


def test_the_live_answer_was_off_contract_in_the_legacy_mode(tmp_path, monkeypatch):
    """The `legacy` mode: the same answer is refused (the defect D-13)."""
    settings = _settings(tmp_path, monkeypatch, mode="legacy")
    service, _script = _service(settings, [LIVE_ANSWER])
    with pytest.raises(SummaryError) as excinfo:
        service.merge(LIVE_A, LIVE_B)
    assert excinfo.value.outcome is SlotOutcome.OFF_CONTRACT


# --- the refusal of the merge goes to the breaker of the operation -----------


def test_the_refusal_of_the_merge_goes_to_the_merge_breaker(tmp_path, monkeypatch):
    """The `operation` mode: the refusal of the merge — the `merge` breaker."""
    settings = _settings(tmp_path, monkeypatch, mode="operation")
    worker = _worker(settings, MergeFailingSummarizer())
    older_id, newer_id = _merge_pair(settings)
    _merge_job(worker, settings, newer_id, older_id)
    worker.process_merge_pending()
    breaker = worker._breakers["merge"]
    assert breaker.snapshot()["last_outcome"] == "off_contract"
    # The neighbouring operations are not touched.
    for name in ("summarize", "title", "split", "classify"):
        assert worker._breakers[name].snapshot()["last_outcome"] is None


def test_the_legacy_merge_keeps_the_previous_breaker(tmp_path, monkeypatch):
    """The `legacy` mode: the refusal stays in the journal, the breaker of the
    `summary` slot group is touched (as before)."""
    settings = _settings(tmp_path, monkeypatch, mode="legacy")
    worker = _worker(settings, MergeFailingSummarizer())
    older_id, newer_id = _merge_pair(settings)
    _merge_job(worker, settings, newer_id, older_id)
    worker.process_merge_pending()
    # The `legacy` mode has no breaker per operation.
    assert "merge" not in worker._breakers
    assert worker._breakers["summary"].snapshot()["last_outcome"] == "off_contract"
