"""The breaker contour on operations (release 3.9.0, LSBN-S-7).

    The `operation` mode (`SLOT_MODEL=operation`) keys the breakers by the
    OPERATION, not by the slot group. The contour around the breakers must
    follow the same key: the probe knock (the service behind the operation),
    the failure record (the breaker of the operation), the attention state
    (the pairs of the operation) and the decision rule (only `judge` and
    `classify` wait for the decision slot).

    Checkable contracts (the criteria of the step):
    1. the probe finds the service of every operation; an unknown operation
       answers `CHECK_OK`;
    2. `/health.slots` in the `operation` mode does not fall over and does
       not lose an operation;
    3. a failure of an operation pauses the breaker of that operation only;
    4. the attention is counted and reset per operation;
    5. an unavailable decision slot blocks `judge` and `classify` only;
    6. the `legacy` mode is unchanged (the snapshot below was taken on the
       base commit `be02293`, before this step).

    A unit without a network: the models are fakes, the encoder — `HashEmbedder`.
"""

from __future__ import annotations

import pytest
from fakes import (
    FailingEmbedder,
    FixedClassifier,
    FixedSummarizer,
    HashEmbedder,
    ScriptedJudge,
    clear_seeded_skills,
)

from app.config import get_settings
from app.services.areas import USER_FACTS_AREA
from app.services.llm_client import CHECK_OK
from app.services.notes import NoteService
from app.services.operations import OPERATION_REGISTRY
from app.services.slot_outcome import SlotOutcome
from app.services.summary import SummaryError
from app.services.worker import BackgroundWorker
from app.storage.db import init_db, session, transaction

DIM = 8

#: The `legacy` contour snapshot — taken on the base commit `be02293` BEFORE
#: this step (the `legacy` mode must not change in any respect). The script
#: that produced it: the breaker keys, the blocked slots with an unavailable
#: decision slot, the `/health.slots` block (the keys of the entries) and the
#: breaker that a failure of the `summary` slot pauses.
LEGACY_CONTOUR: dict = {
    "breakers": ["embedding", "judge", "summary"],
    "slot_blocked": {"embedding": False, "judge": True, "summary": True},
    "health_keys": ["decision", "embedding", "judge", "summary"],
    "health_entry_keys": {
        "decision": [
            "attention",
            "enabled",
            "last_budget",
            "last_budget_source",
            "last_finish_reason",
            "last_had_thinking",
            "last_outcome",
            "ok",
            "paused_until",
        ],
        "embedding": [
            "attention",
            "failures",
            "last_answer_budget",
            "last_budget",
            "last_budget_source",
            "last_finish_reason",
            "last_had_thinking",
            "last_outcome",
            "last_reasoning_budget",
            "last_sent_budget",
            "ok",
            "paused_until",
            "slot",
            "state",
        ],
        "judge": [
            "attention",
            "failures",
            "last_answer_budget",
            "last_budget",
            "last_budget_source",
            "last_finish_reason",
            "last_had_thinking",
            "last_outcome",
            "last_reasoning_budget",
            "last_sent_budget",
            "ok",
            "paused_until",
            "slot",
            "state",
        ],
        "summary": [
            "attention",
            "failures",
            "last_answer_budget",
            "last_budget",
            "last_budget_source",
            "last_finish_reason",
            "last_had_thinking",
            "last_outcome",
            "last_reasoning_budget",
            "last_sent_budget",
            "ok",
            "paused_until",
            "slot",
            "state",
        ],
    },
    "paused_after_summary_failure": ["summary"],
}


class ProbingSummarizer(FixedSummarizer):
    """The summarizer with a probe knock (the name of the service is the answer)."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.probes = 0

    def probe(self) -> str:
        self.probes += 1
        return "probe:summarizer"


class ProbingEmbedder(HashEmbedder):
    """The encoder with a probe knock."""

    def __init__(self, dim: int = DIM) -> None:
        super().__init__(dim)
        self.probes = 0

    def probe(self) -> str:
        self.probes += 1
        return "probe:embedding"


class ProbingJudge(ScriptedJudge):
    """The dedup judge with a probe knock."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.probes = 0

    def probe(self) -> str:
        self.probes += 1
        return "probe:judge"


class ProbingClassifier(FixedClassifier):
    """The classifier with a probe knock."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.probes = 0

    def probe(self) -> str:
        self.probes += 1
        return "probe:classifier"


class ProbingPromoter:
    """The promotion service (the `describe` operation) with a probe knock."""

    def __init__(self) -> None:
        self.probes = 0

    def probe(self) -> str:
        self.probes += 1
        return "probe:promoter"

    def run(self) -> None:
        return None


class ProbingDecision:
    """The decision slot: there, but unable to decide (the F-12 reserve)."""

    def __init__(self, *, can_decide: bool = True) -> None:
        self._can_decide = can_decide
        self.probes = 0

    def can_decide(self) -> bool:
        return self._can_decide

    def probe(self) -> str:
        self.probes += 1
        return "probe:decision"

    def snapshot(self) -> dict:
        return {
            "enabled": True,
            "ok": self._can_decide,
            "breaker": {"last_outcome": None, "paused_until": None},
            "answers": 0,
            "fallbacks": 0,
            "by_reason": {},
        }


class TitleFailingSummarizer(ProbingSummarizer):
    """The summarizer: the title always fails, the summary works.

    A `SummaryError` without an outcome is read as `off_contract` (a
    non-transient failure) — the pause of the operation opens at once (with
    `SLOT_PAUSE_AFTER=1`).
    """

    def title(self, text: str):  # noqa: ANN201 — the fake's signature
        self.calls.append(text)
        self.last_attempt_ok = False
        raise SummaryError("the model returned not-a-title")


def _settings(tmp_path, monkeypatch: pytest.MonkeyPatch, *, mode: str):
    """The test settings: own database, the slot model, a one-failure pause."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "notes.db"))
    monkeypatch.setenv("EMBEDDING_DIM", str(DIM))
    monkeypatch.setenv("SUMMARY_SHORT_TEXT_CHARS", "0")
    # The pause opens at the first failure — otherwise the third failure
    # would open it.
    monkeypatch.setenv("SLOT_PAUSE_AFTER", "1")
    # The decision slot: there, but unable to decide, and the fallback is
    # off — the state the decision rule is about (no39-05).
    monkeypatch.setenv("DECISION_FALLBACK", "false")
    if mode == "operation":
        monkeypatch.setenv("SLOT_MODEL", "operation")
    else:
        monkeypatch.delenv("SLOT_MODEL", raising=False)
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    clear_seeded_skills(settings)
    return settings


@pytest.fixture
def legacy_settings(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """The `legacy` mode (the default): the contour is as before."""
    return _settings(tmp_path, monkeypatch, mode="legacy")


@pytest.fixture
def operation_settings(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """The `operation` mode: the breakers are keyed by the operation."""
    return _settings(tmp_path, monkeypatch, mode="operation")


def make_worker(
    settings, *, summarizer=None, decision=None, classifier=None, embedder=None
) -> BackgroundWorker:
    """A worker with every service behind an operation (the probe map)."""
    return BackgroundWorker(
        settings,
        embedder if embedder is not None else ProbingEmbedder(DIM),
        summarizer if summarizer is not None else ProbingSummarizer(),
        judge=ProbingJudge(),
        classifier=(
            classifier if classifier is not None else ProbingClassifier()
        ),
        promoter=ProbingPromoter(),
        decision=decision,
    )


def _paused(worker: BackgroundWorker, key: str) -> bool:
    """Is the breaker under the key paused right now?"""
    breaker = worker._breakers.get(key)
    return breaker is not None and breaker.is_paused()


# --- criterion 1: the probe knock finds the service of the operation ---------


def test_operation_probe_hits_the_service_of_the_operation(operation_settings) -> None:
    """The knock goes to the service behind the operation."""
    worker = make_worker(operation_settings, decision=ProbingDecision())
    assert worker._probe_slot("summarize") == "probe:summarizer"
    assert worker._probe_slot("title") == "probe:summarizer"
    assert worker._probe_slot("merge") == "probe:summarizer"
    assert worker._probe_slot("judge") == "probe:judge"
    assert worker._probe_slot("structure_judge") == "probe:judge"
    assert worker._probe_slot("classify") == "probe:classifier"
    assert worker._probe_slot("describe") == "probe:promoter"
    assert worker._probe_slot("embed") == "probe:embedding"
    assert worker._probe_slot("decision") == "probe:decision"


def test_operation_probe_does_not_knock_on_a_neighbour(operation_settings) -> None:
    """The knock of one operation does not touch the service of another."""
    worker = make_worker(operation_settings)
    worker._probe_slot("title")
    assert worker._summarizer.probes == 1
    assert worker._embedding.probes == 0
    assert worker._judge.probes == 0
    assert worker._classifier.probes == 0


def test_unknown_operation_probe_is_check_ok(operation_settings) -> None:
    """An unknown operation — no service: the `CHECK_OK` stub."""
    worker = make_worker(operation_settings)
    assert worker._probe_slot("no_such_operation") == CHECK_OK
    assert worker._probe_slot("split") == "probe:summarizer"


def test_legacy_probe_still_knocks_on_the_slot(legacy_settings) -> None:
    """The `legacy` mode keeps the slot map: the slot's own service."""
    worker = make_worker(legacy_settings)
    assert worker._probe_slot("summary") == "probe:summarizer"
    assert worker._probe_slot("judge") == "probe:judge"
    assert worker._probe_slot("embedding") == "probe:embedding"
    assert worker._probe_slot("title") == CHECK_OK  # not a slot name


# --- criterion 2: /health.slots does not fall over in the operation mode ----


def test_slots_health_in_operation_mode_does_not_raise(operation_settings) -> None:
    """`/health.slots` in the `operation` mode: no exception, no lost operation."""
    worker = make_worker(operation_settings)
    health = worker.slots_health()
    assert len(health) >= len(worker._breakers)
    for key in worker._breakers:
        assert key in health
    assert set(OPERATION_REGISTRY) <= set(health)


def test_slots_health_in_operation_mode_has_no_legacy_slot_keys(
    operation_settings,
) -> None:
    """The legacy slot keys are NOT added in this form (the step S-15)."""
    worker = make_worker(operation_settings)
    health = worker.slots_health()
    assert "embedding" not in health
    assert "summary" not in health


def test_slots_health_entries_carry_the_breaker_state(operation_settings) -> None:
    """The entry of an operation carries its breaker snapshot."""
    worker = make_worker(operation_settings)
    health = worker.slots_health()
    for key, breaker in worker._breakers.items():
        assert health[key]["paused_until"] == breaker.snapshot()["paused_until"]
        assert health[key]["attention"] == 0


def test_slots_health_counts_the_attention_of_an_operation(operation_settings) -> None:
    """The attention counter of an operation is over its own pairs."""
    worker = make_worker(operation_settings)
    worker._create_job("summary", "merge", 1)
    worker._create_job("judge", "dedup", 2)
    _set_attention(operation_settings)
    health = worker.slots_health()
    assert health["merge"]["attention"] == 1
    assert health["judge"]["attention"] == 1
    assert health["summarize"]["attention"] == 0


# --- criterion 3: a failure pauses the breaker of the operation --------------


def test_title_failure_pauses_the_title_breaker_only(operation_settings) -> None:
    """A failure of the title operation: the pause is only at `title`."""
    worker = make_worker(operation_settings, summarizer=TitleFailingSummarizer())
    NoteService(operation_settings, ProbingEmbedder(DIM)).save(
        "a note without a title: it waits for the title and for the summary"
    )
    assert worker.process_title_pending() == 0
    assert _paused(worker, "title") is True
    assert _paused(worker, "summarize") is False
    assert _paused(worker, "merge") is False


def test_embed_failure_pauses_the_embed_breaker(operation_settings) -> None:
    """A failure of the embed operation (the areas loop) pauses `embed`.

    The real path: the areas loop encodes the pending area records and the
    failure goes through the single point of a slot failure
    (`_log_slot_failure`) — the breaker key must be the operation.
    """
    worker = make_worker(operation_settings, embedder=FailingEmbedder())
    with session(operation_settings) as conn, transaction(conn):
        conn.execute(
            "INSERT INTO user_facts (name, body, vector_status) "
            "VALUES ('факт', 'тело факта', 'pending')"
        )
    assert worker._process_area_batch(USER_FACTS_AREA, 10) == 0
    assert _paused(worker, "embed") is True
    assert _paused(worker, "summarize") is False
    assert _paused(worker, "title") is False


def test_classify_failure_pauses_the_classify_breaker(operation_settings) -> None:
    """A failure of the classify operation pauses `classify` only.

    The real path: the classifier of a default note fails and the failure
    goes through the single point of a slot failure.
    """
    worker = make_worker(
        operation_settings, classifier=FixedClassifier(fail=True)
    )
    worker._classify_default_note(1, "текст заметки")
    assert _paused(worker, "classify") is True
    assert _paused(worker, "summarize") is False
    assert _paused(worker, "merge") is False


def test_failure_of_an_operation_does_not_pause_a_neighbour(operation_settings) -> None:
    """The breaker of the neighbour operation stays closed."""
    worker = make_worker(operation_settings)
    worker._log_slot_failure(
        "summary",
        worker._operation_slot("summarize"),
        SlotOutcome.OFF_CONTRACT,
        event="summary_failed",
    )
    assert _paused(worker, "summarize") is True
    assert _paused(worker, "title") is False
    assert _paused(worker, "merge") is False


# --- criterion 4: the attention is per operation ----------------------------


def _set_attention(settings) -> None:
    """Put every task row into attention."""
    with session(settings) as conn, transaction(conn):
        conn.execute(
            "UPDATE worker_jobs SET attention_at = "
            "strftime('%Y-%m-%dT%H:%M:%SZ','now')"
        )


def _rows(settings) -> dict[tuple[str, str], dict]:
    """The task rows by (slot, kind): the attempts and the attention state."""
    with session(settings) as conn:
        rows = conn.execute(
            "SELECT id, slot, kind, attempts, attention_at FROM worker_jobs"
        ).fetchall()
    return {
        (row["slot"], row["kind"]): {
            "id": int(row["id"]),
            "attempts": int(row["attempts"] or 0),
            "attention": row["attention_at"] is not None,
        }
        for row in rows
    }


def test_attention_count_is_per_operation(operation_settings) -> None:
    """A failure of one pair counts only that pair of the operation."""
    worker = make_worker(operation_settings)
    worker._create_job("summary", "title", 1)
    worker._create_job("summary", "merge", 2)
    rows = _rows(operation_settings)
    worker._note_attempt(rows[("summary", "title")]["id"], SlotOutcome.OFF_CONTRACT)
    after = _rows(operation_settings)
    assert after[("summary", "title")]["attempts"] == 1
    assert after[("summary", "merge")]["attempts"] == 0
    # The breaker's state is not read by the pair counter (the invariant).
    assert _paused(worker, "summarize") is False
    assert _paused(worker, "merge") is False


def test_attention_reset_by_operation_keeps_the_neighbour(operation_settings) -> None:
    """The reset of one operation does not clear the attention of another."""
    worker = make_worker(operation_settings)
    worker._create_job("summary", "title", 1)
    worker._create_job("summary", "merge", 2)
    worker._create_job("judge", "dedup", 3)
    _set_attention(operation_settings)

    released = worker._reset_attention_for_slot("title", "startup")
    assert released == 1
    rows = _rows(operation_settings)
    assert rows[("summary", "title")]["attention"] is False
    assert rows[("summary", "merge")]["attention"] is True
    assert rows[("judge", "dedup")]["attention"] is True


def test_attention_reset_of_the_summary_family_is_per_operation(
    operation_settings,
) -> None:
    """The reset of `summarize` does not clear the pairs of `merge`/`title`."""
    worker = make_worker(operation_settings)
    worker._create_job("summary", "title", 1)
    worker._create_job("summary", "merge", 2)
    worker._create_job("summary", "summarize", 3)
    _set_attention(operation_settings)

    assert worker._reset_attention_for_slot("summarize", "startup") == 1
    rows = _rows(operation_settings)
    assert rows[("summary", "summarize")]["attention"] is False
    assert rows[("summary", "title")]["attention"] is True
    assert rows[("summary", "merge")]["attention"] is True


def test_legacy_attention_reset_is_per_slot(legacy_settings) -> None:
    """The `legacy` mode: the reset is over the slot group (as before)."""
    worker = make_worker(legacy_settings)
    worker._create_job("summary", "title", 1)
    worker._create_job("summary", "merge", 2)
    worker._create_job("judge", "dedup", 3)
    _set_attention(legacy_settings)

    released = worker._reset_attention_for_slot("summary", "startup")
    assert released == 2
    rows = _rows(legacy_settings)
    assert rows[("summary", "title")]["attention"] is False
    assert rows[("summary", "merge")]["attention"] is False
    assert rows[("judge", "dedup")]["attention"] is True


# --- criterion 5: the decision rule is per operation ------------------------


def test_unavailable_decision_blocks_judge_and_classify(operation_settings) -> None:
    """The decision slot is down and the fallback is off: `judge`/`classify` wait."""
    worker = make_worker(operation_settings, decision=ProbingDecision(can_decide=False))
    assert worker._operation_blocked("judge") is True
    assert worker._operation_blocked("classify") is True
    assert worker._slot_blocked("judge") is True
    assert worker._slot_blocked("classify") is True
    for operation in ("summarize", "title", "merge", "describe", "embed"):
        assert worker._operation_blocked(operation) is False, operation
        assert worker._slot_blocked(operation) is False, operation


def test_legacy_decision_rule_keeps_the_summary_family(legacy_settings) -> None:
    """The `legacy` mode: the whole summary family waits (as before)."""
    worker = make_worker(legacy_settings, decision=ProbingDecision(can_decide=False))
    assert worker._slot_blocked("summary") is True
    assert worker._slot_blocked("judge") is True
    assert worker._slot_blocked("embedding") is False


def test_available_decision_blocks_nothing(operation_settings) -> None:
    """The decision slot is there: nothing is blocked by the decision."""
    worker = make_worker(operation_settings, decision=ProbingDecision(can_decide=True))
    assert worker._operation_blocked("judge") is False
    assert worker._operation_blocked("classify") is False
    assert worker._operation_blocked("summarize") is False


# --- criterion 6: the legacy mode did not change ----------------------------


def test_legacy_contour_snapshot_is_unchanged(legacy_settings) -> None:
    """The `legacy` contour is exactly the snapshot of the base commit."""
    worker = make_worker(
        legacy_settings,
        decision=ProbingDecision(can_decide=False),
    )
    health = worker.slots_health()
    snapshot = {
        "breakers": sorted(worker._breakers),
        "slot_blocked": {
            name: worker._slot_blocked(name)
            for name in ("embedding", "summary", "judge")
        },
        "health_keys": sorted(health),
        "health_entry_keys": {key: sorted(value) for key, value in health.items()},
    }
    worker._log_slot_failure(
        "summary", "summary", SlotOutcome.OFF_CONTRACT, event="legacy_probe"
    )
    snapshot["paused_after_summary_failure"] = sorted(
        name
        for name, breaker in worker._breakers.items()
        if breaker.is_paused()
    )
    assert snapshot == LEGACY_CONTOUR
