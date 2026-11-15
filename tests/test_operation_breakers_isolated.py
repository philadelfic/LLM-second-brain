"""The breakers of the operations are isolated (release 3.9.0, LSBN-S-13).

    The defect LSBN-D-18: a failure of the `merge` operation paused the whole
    `summary` slot for 900 s — the summarization stood still. In the `operation`
    mode every operation owns its breaker: its own failure counter, its own
    pause and its own back-off. The thresholds are base + override (the
    per-operation value, else the global base `SLOT_PAUSE_AFTER`/`SLOT_PAUSE_SEC`
    — the owner's decision of 15.11.2026).

    The criteria of the step:
      1. D-18 is closed: a failure of `merge` does not mute the other operations;
      2. the pause length is the operation's threshold (base + override);
      3. no breaker is shared;
      4. the back-off is independent;
      5. the non-transient outcomes count the pair's attempts even under a pause;
      6. the `decision` operation has its own breaker (its own thresholds, the
         env prefix stays `DECISION_*`);
      7. the `legacy` mode did not change (the snapshot below was taken on the
         base commit `cc89d64`, before this step — the script is described in
         the comment over the snapshot).

    A unit without a network: the models are fakes, the encoder — `HashEmbedder`.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from fakes import (
    FailingEmbedder,
    FixedClassifier,
    FixedSummarizer,
    HashEmbedder,
    MergeFailingSummarizer,
    ScriptedJudge,
)

from app.config import get_settings
from app.services.decision import DecisionService
from app.services.notes import NoteService
from app.services.operations import OPERATION_REGISTRY, breaker_key, env_name
from app.services.slot_outcome import SlotOutcome
from app.services.worker import (
    BackgroundWorker,
    build_merge_job,
    build_summarize_job,
)
from app.storage.db import init_db, session

DIM = 8

#: The pair of the merge: two versions of the same note.
TEXT_A = "Ретроспектива продукта назначена на 12 сентября 2026, обсуждали 3 релиза."
TEXT_B = "Решили собрать 5 интервью и перенести демо на 20 ноября 2026."
TEXT_C = "Заметка без заголовка: она ждёт своей очереди title."

#: The `legacy` contour snapshot — taken on the base commit `cc89d64` BEFORE
#: this step (the `legacy` mode must not change in any respect). The script
#: that produced it: the `legacy` mode's worker (the fakes of `tests/fakes.py`),
#: the keys of the breakers, the `/health.slots` block (the keys of the entries
#: and the state of every entry), the states of the breakers before and after a
#: failure of the `summary` slot group, the blocked slots and the pair's
#: attempt counter (two non-transient failures under an open pause).
LEGACY_BREAKER_CONTOUR: dict = {
    "attempts_are_counted": {
        "attempts": 2,
        "attention": False,
        "last_outcome": "empty",
    },
    "blocked": {
        "embedding": False,
        "judge": False,
        "summary": False,
    },
    "breaker_keys": [
        "embedding",
        "judge",
        "summary",
    ],
    "health_after": {
        "decision": {
            "attention": 0,
            "failures": None,
            "last_outcome": None,
            "paused": False,
            "state": None,
        },
        "embedding": {
            "attention": 0,
            "failures": 0,
            "last_outcome": None,
            "paused": False,
            "state": "closed",
        },
        "judge": {
            "attention": 0,
            "failures": 0,
            "last_outcome": None,
            "paused": False,
            "state": "closed",
        },
        "summary": {
            "attention": 0,
            "failures": 1,
            "last_outcome": "off_contract",
            "paused": True,
            "state": "paused",
        },
    },
    "health_before": {
        "decision": {
            "attention": 0,
            "failures": None,
            "last_outcome": None,
            "paused": False,
            "state": None,
        },
        "embedding": {
            "attention": 0,
            "failures": 0,
            "last_outcome": None,
            "paused": False,
            "state": "closed",
        },
        "judge": {
            "attention": 0,
            "failures": 0,
            "last_outcome": None,
            "paused": False,
            "state": "closed",
        },
        "summary": {
            "attention": 0,
            "failures": 0,
            "last_outcome": None,
            "paused": False,
            "state": "closed",
        },
    },
    "health_entry_keys": {
        "decision": [
            "attention",
            "enabled",
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
    "health_keys": [
        "decision",
        "embedding",
        "judge",
        "summary",
    ],
    "paused_after": [
        "summary",
    ],
    "slot_states": {
        "embedding": {
            "failures": 0,
            "last_outcome": None,
            "slot": "embedding",
            "state": "closed",
        },
        "judge": {
            "failures": 0,
            "last_outcome": None,
            "slot": "judge",
            "state": "closed",
        },
        "summary": {
            "failures": 0,
            "last_outcome": None,
            "slot": "summary",
            "state": "closed",
        },
    },
    "slot_states_after": {
        "embedding": {
            "failures": 0,
            "last_outcome": None,
            "slot": "embedding",
            "state": "closed",
        },
        "judge": {
            "failures": 0,
            "last_outcome": None,
            "slot": "judge",
            "state": "closed",
        },
        "summary": {
            "failures": 1,
            "last_outcome": "off_contract",
            "slot": "summary",
            "state": "paused",
        },
    },
}


def _settings(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    mode: str = "operation",
    **env: str,
):
    """The test settings: own database, the slot model, the pause thresholds."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "notes.db"))
    monkeypatch.setenv("EMBEDDING_DIM", str(DIM))
    # The short-note short-circuit is off — every note needs a model call.
    monkeypatch.setenv("SUMMARY_SHORT_TEXT_CHARS", "0")
    # The pause opens at the first failure — otherwise the third failure
    # would open it.
    monkeypatch.setenv("SLOT_PAUSE_AFTER", "1")
    monkeypatch.setenv("SLOT_PAUSE_SEC", "900")
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    if mode == "operation":
        monkeypatch.setenv("SLOT_MODEL", "operation")
    else:
        monkeypatch.delenv("SLOT_MODEL", raising=False)
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    return settings


def _worker(settings, *, summarizer=None, embedder=None, decision=None):
    """A worker with the fakes behind the operations."""
    return BackgroundWorker(
        settings,
        embedder if embedder is not None else HashEmbedder(DIM),
        summarizer if summarizer is not None else FixedSummarizer(),
        judge=ScriptedJudge(),
        classifier=FixedClassifier(),
        decision=decision,
    )


def _notes(settings) -> NoteService:
    """The note service of the test database."""
    return NoteService(settings)


def _merge_job(worker: BackgroundWorker, newer_id: int, older_id: int) -> None:
    """A merge job in the queue — as the judge loop places it."""
    worker._create_job(
        "summary", "merge", newer_id, payload=json.dumps({"older_id": older_id})
    )


def _paused(worker: BackgroundWorker, key: str) -> bool:
    """Is the breaker under the key paused right now?"""
    breaker = worker._breakers.get(key)
    return breaker is not None and breaker.is_paused()


# --- criterion 1: D-18 is closed ---------------------------------------------


def test_the_failure_of_the_merge_pauses_only_the_merge_breaker(tmp_path, monkeypatch):
    """A failure of the merge operation: the pause is only at `merge`.

    The real path: the merge loop takes a merge job and the summarizer's merge
    fails — the failure goes through the single point of a slot failure and the
    breaker key must be the operation.
    """
    settings = _settings(tmp_path, monkeypatch)
    worker = _worker(settings, summarizer=MergeFailingSummarizer())
    notes = _notes(settings)
    older_id = notes.save(TEXT_A, title="Ранняя заметка")["id"]
    newer_id = notes.save(TEXT_B, title="Поздняя заметка")["id"]
    _merge_job(worker, newer_id, older_id)

    worker.process_merge_pending()

    assert _paused(worker, "merge") is True
    assert worker._breakers["merge"].snapshot()["last_outcome"] == "off_contract"
    for operation in ("summarize", "title", "describe", "classify", "embed"):
        assert _paused(worker, operation) is False, operation
        assert worker._breakers[operation].snapshot()["last_outcome"] is None


def test_the_summary_and_the_title_loops_work_after_a_merge_failure(
    tmp_path, monkeypatch
):
    """The merge is down — the summarization and the titles go on.

    The defect D-18: the pause of the merge muted the whole summary slot.
    """
    settings = _settings(tmp_path, monkeypatch)
    worker = _worker(settings, summarizer=MergeFailingSummarizer())
    notes = _notes(settings)
    older_id = notes.save(TEXT_A, title="Ранняя заметка")["id"]
    newer_id = notes.save(TEXT_B, title="Поздняя заметка")["id"]
    notes.save(TEXT_C)  # without a title — the `title` queue
    _merge_job(worker, newer_id, older_id)

    worker.process_merge_pending()
    assert _paused(worker, "merge") is True

    assert worker.process_summary_pending() == 3  # the summaries are written
    assert worker.process_title_pending() == 1  # the title is written
    assert _paused(worker, "merge") is True  # the merge is still down


def test_the_merge_failure_does_not_pause_the_neighbour_operations(
    tmp_path, monkeypatch
):
    """The counters of the neighbour operations stay at zero."""
    settings = _settings(tmp_path, monkeypatch)
    worker = _worker(settings, summarizer=MergeFailingSummarizer())
    notes = _notes(settings)
    older_id = notes.save(TEXT_A, title="Ранняя заметка")["id"]
    newer_id = notes.save(TEXT_B, title="Поздняя заметка")["id"]
    _merge_job(worker, newer_id, older_id)

    worker.process_merge_pending()

    assert worker._breakers["merge"].snapshot()["failures"] == 1
    for operation in ("summarize", "title", "split", "classify", "describe"):
        assert worker._breakers[operation].snapshot()["failures"] == 0, operation


# --- criterion 2: the pause length is the operation's threshold ---------------


def test_the_pause_of_the_merge_is_the_per_operation_value(tmp_path, monkeypatch):
    """`MERGE_PAUSE_SEC`/`MERGE_PAUSE_AFTER` are the thresholds of `merge`."""
    settings = _settings(
        tmp_path,
        monkeypatch,
        MERGE_PAUSE_SEC="120",
        MERGE_PAUSE_AFTER="5",
    )
    worker = _worker(settings)
    assert worker._breakers["merge"].pause_sec == 120
    assert worker._breakers["merge"].open_after == 5


def test_the_base_applies_to_an_operation_without_its_own_value(
    tmp_path, monkeypatch
):
    """An unset per-operation field: the global base is in force."""
    settings = _settings(tmp_path, monkeypatch)
    worker = _worker(settings)
    for operation in ("summarize", "title", "merge", "judge", "embed"):
        assert worker._breakers[operation].pause_sec == 900, operation
        assert worker._breakers[operation].open_after == 1, operation


def test_the_per_operation_value_wins_over_the_base(tmp_path, monkeypatch):
    """The per-operation value beats the base; the neighbour keeps the base."""
    settings = _settings(tmp_path, monkeypatch, MERGE_PAUSE_SEC="120")
    worker = _worker(settings)
    assert worker._breakers["merge"].pause_sec == 120
    assert worker._breakers["summarize"].pause_sec == 900


def test_the_pause_of_the_merge_is_its_own_value_on_the_real_path(
    tmp_path, monkeypatch
):
    """The pause of the merge operation lasts its own time (not the base's)."""
    settings = _settings(tmp_path, monkeypatch, MERGE_PAUSE_SEC="120")
    worker = _worker(settings, summarizer=MergeFailingSummarizer())
    notes = _notes(settings)
    older_id = notes.save(TEXT_A, title="Ранняя заметка")["id"]
    newer_id = notes.save(TEXT_B, title="Поздняя заметка")["id"]
    _merge_job(worker, newer_id, older_id)

    worker.process_merge_pending()

    paused_until = worker._breakers["merge"].snapshot()["paused_until"]
    assert paused_until is not None
    # The pause is 120 s long — not the base's 900 s.
    until = datetime.strptime(paused_until, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc
    )
    left = (until - datetime.now(timezone.utc)).total_seconds()
    assert 100 <= left <= 120
    assert worker._breakers["merge"].pause_sec == 120
    assert worker._breakers["summarize"].pause_sec == 900


# --- criterion 3: no breaker is shared ---------------------------------------


def test_every_operation_has_its_own_breaker_key(tmp_path, monkeypatch):
    """The breaker key of every operation of the registry is unique."""
    settings = _settings(tmp_path, monkeypatch)
    keys = [breaker_key(operation, settings) for operation in OPERATION_REGISTRY]
    assert len(set(keys)) == len(OPERATION_REGISTRY)
    assert set(keys) == set(OPERATION_REGISTRY)


def test_no_two_operations_share_a_breaker(tmp_path, monkeypatch):
    """The breakers of the operations are different objects."""
    settings = _settings(tmp_path, monkeypatch)
    worker = _worker(settings)
    breakers = [
        worker._breakers[operation]
        for operation in OPERATION_REGISTRY
        if operation in worker._breakers
    ]
    assert len({id(breaker) for breaker in breakers}) == len(breakers)
    # The `decision` operation's breaker lives in the service (arch §13.8).
    assert "decision" not in worker._breakers


def test_the_failure_of_one_operation_does_not_move_the_neighbour(
    tmp_path, monkeypatch
):
    """The neighbour keeps its counter, its pause and its outcome."""
    settings = _settings(tmp_path, monkeypatch)
    worker = _worker(settings)
    worker._note_slot_outcome(
        "merge", worker._operation_slot("merge"), SlotOutcome.OFF_CONTRACT
    )
    neighbour = worker._breakers["summarize"].snapshot()
    assert neighbour["failures"] == 0
    assert neighbour["last_outcome"] is None
    assert neighbour["paused_until"] is None


def test_the_failure_of_the_vectorization_goes_to_the_embed_breaker(
    tmp_path, monkeypatch
):
    """The `embed` operation's own breaker: the failure of the notes queue.

    The vectorization path of the notes (the areas path writes there too).
    """
    settings = _settings(tmp_path, monkeypatch)
    worker = _worker(settings, embedder=FailingEmbedder())
    _notes(settings).save(TEXT_C)

    assert worker.process_pending() == 0

    assert _paused(worker, "embed") is True
    assert worker._breakers["embed"].snapshot()["last_outcome"] == "off_contract"
    assert _paused(worker, "summarize") is False


def test_the_legacy_vectorization_does_not_touch_the_breaker(tmp_path, monkeypatch):
    """The `legacy` mode: the vectorization path is as it was (no breaker).

    The pilot's decision 1: the `legacy` mode changes in nothing.
    """
    settings = _settings(tmp_path, monkeypatch, mode="legacy")
    worker = _worker(settings, embedder=FailingEmbedder())
    _notes(settings).save(TEXT_C)

    assert worker.process_pending() == 0

    assert _paused(worker, "embedding") is False
    assert worker._breakers["embedding"].snapshot()["last_outcome"] is None


# --- criterion 4: the back-off is independent --------------------------------


def test_the_failure_series_is_per_operation(tmp_path, monkeypatch):
    """A series of failures of one operation grows its own counter only."""
    settings = _settings(tmp_path, monkeypatch, SLOT_PAUSE_AFTER="3")
    worker = _worker(settings)
    for _ in range(2):
        worker._note_slot_outcome(
            "merge", worker._operation_slot("merge"), SlotOutcome.OFF_CONTRACT
        )
    assert worker._breakers["merge"].snapshot()["failures"] == 2
    assert worker._breakers["summarize"].snapshot()["failures"] == 0
    assert worker._breakers["title"].snapshot()["failures"] == 0


def test_the_success_resets_only_its_own_operation(tmp_path, monkeypatch):
    """A success clears the counter of its own operation."""
    settings = _settings(tmp_path, monkeypatch, SLOT_PAUSE_AFTER="3")
    worker = _worker(settings)
    worker._note_slot_outcome(
        "merge", worker._operation_slot("merge"), SlotOutcome.OFF_CONTRACT
    )
    worker._note_slot_outcome(
        "summarize", worker._operation_slot("summarize"), SlotOutcome.EMPTY
    )

    worker._breakers["merge"].note_success()

    assert worker._breakers["merge"].snapshot()["failures"] == 0
    assert worker._breakers["summarize"].snapshot()["failures"] == 1


def test_the_loop_backoff_of_the_merge_job_is_its_own(tmp_path, monkeypatch):
    """The back-off of the loop of the merge job is not the neighbour's."""
    settings = _settings(tmp_path, monkeypatch)
    worker = _worker(settings)
    merge_spec = build_merge_job(worker, settings)
    summarize_spec = build_summarize_job(worker, settings)
    assert merge_spec is not None and summarize_spec is not None

    merge_backoff = worker._job_backoff(merge_spec)
    summarize_backoff = worker._job_backoff(summarize_spec)
    assert merge_backoff is not summarize_backoff

    merge_backoff.grow(settings.pending_retry_sec)

    assert merge_backoff.interval > summarize_backoff.interval
    assert summarize_backoff.interval == float(settings.pending_retry_sec)


# --- criterion 5: the non-transient outcomes count the attempts ---------------


def test_the_non_transient_outcome_counts_the_attempts_under_a_pause(
    tmp_path, monkeypatch
):
    """The pair's counter grows even while the operation's breaker is open.

    The invariant of the canon (design §B4): the pause is the operation's
    state, the count is the pair's property.
    """
    settings = _settings(tmp_path, monkeypatch)
    worker = _worker(settings)
    worker._create_job("summary", "merge", 1)
    with session(settings) as conn:
        job_id = int(
            conn.execute(
                "SELECT id FROM worker_jobs WHERE slot = 'summary' AND kind = 'merge' "
                "ORDER BY id DESC LIMIT 1"
            ).fetchone()["id"]
        )
    # The breaker of the merge operation is open (auth opens it at once).
    worker._note_slot_outcome(
        "merge", worker._operation_slot("merge"), SlotOutcome.AUTH
    )
    assert _paused(worker, "merge") is True

    for _ in range(3):
        worker._note_attempt(job_id, SlotOutcome.OFF_CONTRACT, detail="off contract")

    with session(settings) as conn:
        row = conn.execute(
            "SELECT attempts, attention_at FROM worker_jobs WHERE id = ?", (job_id,)
        ).fetchone()
    assert int(row["attempts"] or 0) == 3


# --- criterion 6: the `decision` operation has its own breaker ----------------


def test_the_decision_operation_has_its_own_thresholds(tmp_path, monkeypatch):
    """In the `operation` mode the decision breaker takes the operation's
    thresholds (`DECISION_PAUSE_AFTER`/`DECISION_PAUSE_SEC`)."""
    settings = _settings(
        tmp_path,
        monkeypatch,
        DECISION_ENABLED="true",
        DECISION_MODEL="test-decision-model",
        DECISION_BASE_URL="http://127.0.0.1:1",
        DECISION_PAUSE_SEC="60",
        DECISION_PAUSE_AFTER="2",
    )
    service = DecisionService(settings)
    assert service.breaker.pause_sec == 60
    assert service.breaker.open_after == 2


def test_the_legacy_decision_breaker_keeps_the_base(tmp_path, monkeypatch):
    """The `legacy` mode: the decision slot's thresholds are the base's."""
    settings = _settings(
        tmp_path,
        monkeypatch,
        mode="legacy",
        DECISION_ENABLED="true",
        DECISION_MODEL="test-decision-model",
        DECISION_BASE_URL="http://127.0.0.1:1",
        DECISION_PAUSE_SEC="60",
        DECISION_PAUSE_AFTER="2",
    )
    service = DecisionService(settings)
    assert service.breaker.pause_sec == settings.slot_pause_sec
    assert service.breaker.open_after == settings.slot_pause_after


def test_the_decision_env_prefix_is_unchanged(tmp_path, monkeypatch):
    """The env of the decision operation is `DECISION_*` — as before."""
    assert env_name("decision", "pause_sec") == "DECISION_PAUSE_SEC"
    assert env_name("decision", "pause_after") == "DECISION_PAUSE_AFTER"
    settings = _settings(tmp_path, monkeypatch, DECISION_PAUSE_SEC="60")
    assert settings.decision_pause_sec == 60
    assert settings.decision_pause_after is None


def test_the_decision_operation_is_in_the_operations_health_block(
    tmp_path, monkeypatch
):
    """In the `operation` mode `/health.slots` has the `decision` operation —
    with the state of ITS OWN breaker (the service's)."""
    settings = _settings(
        tmp_path,
        monkeypatch,
        DECISION_ENABLED="true",
        DECISION_MODEL="test-decision-model",
        DECISION_BASE_URL="http://127.0.0.1:1",
    )
    service = DecisionService(settings)
    worker = _worker(settings, decision=service)

    health = worker.slots_health()

    assert "decision" in health
    assert set(OPERATION_REGISTRY) <= set(health)
    service.breaker.note_failure(SlotOutcome.OFF_CONTRACT)
    entry = worker.slots_health()["decision"]
    # The state of the SERVICE's own breaker (the threshold is 1 — the pause
    # opens at the first failure).
    assert entry["state"] == "paused"
    assert entry["failures"] == 1
    assert entry["last_outcome"] == "off_contract"
    assert entry["paused_until"] is not None


def test_the_legacy_decision_slot_form_is_unchanged(tmp_path, monkeypatch):
    """The `legacy` mode: the `decision` slot keeps its old form."""
    settings = _settings(tmp_path, monkeypatch, mode="legacy")
    worker = _worker(settings)
    entry = worker.slots_health()["decision"]
    assert set(entry) == {
        "ok",
        "last_outcome",
        "paused_until",
        "attention",
        "enabled",
    }


# --- criterion 7: the `legacy` mode did not change ----------------------------


def _legacy_contour(settings) -> dict:
    """The `legacy` contour: the snapshot of the behaviour (the base commit)."""
    worker = _worker(settings)

    def health_view() -> dict:
        return {
            key: {
                "state": value.get("state"),
                "last_outcome": value.get("last_outcome"),
                "paused": value.get("paused_until") is not None,
                "failures": value.get("failures"),
                "attention": value.get("attention"),
            }
            for key, value in sorted(worker.slots_health().items())
        }

    def states_view() -> dict:
        return {
            key: {k: v for k, v in value.items() if k != "paused_until"}
            for key, value in sorted(worker.slot_states().items())
        }

    contour = {
        "breaker_keys": sorted(worker._breakers),
        "health_keys": sorted(worker.slots_health()),
        "health_entry_keys": {
            key: sorted(value) for key, value in sorted(worker.slots_health().items())
        },
        "slot_states": states_view(),
        "health_before": health_view(),
        "blocked": {key: worker._slot_blocked(key) for key in sorted(worker._breakers)},
    }
    # A failure of the summary slot group: the shared breaker of the `legacy`
    # mode opens.
    worker._log_slot_failure(
        "summary", "summary", SlotOutcome.OFF_CONTRACT, event="summary_failed"
    )
    contour["health_after"] = health_view()
    contour["slot_states_after"] = states_view()
    contour["paused_after"] = sorted(
        key for key, breaker in worker._breakers.items() if breaker.is_paused()
    )

    worker._create_job("summary", "merge", 1)
    with session(settings) as conn:
        job_id = int(
            conn.execute(
                "SELECT id FROM worker_jobs WHERE slot = 'summary' AND kind = 'merge' "
                "ORDER BY id DESC LIMIT 1"
            ).fetchone()["id"]
        )
    worker._note_attempt(job_id, SlotOutcome.OFF_CONTRACT, detail="off contract")
    worker._note_attempt(job_id, SlotOutcome.EMPTY, detail="empty answer")
    with session(settings) as conn:
        row = conn.execute(
            "SELECT attempts, last_outcome, attention_at FROM worker_jobs WHERE id = ?",
            (job_id,),
        ).fetchone()
    contour["attempts_are_counted"] = {
        "attempts": int(row["attempts"] or 0),
        "last_outcome": row["last_outcome"],
        "attention": row["attention_at"] is not None,
    }
    return contour


def test_the_legacy_contour_snapshot_is_unchanged(tmp_path, monkeypatch):
    """The `legacy` contour is exactly the snapshot of the base commit."""
    settings = _settings(tmp_path, monkeypatch, mode="legacy")
    assert _legacy_contour(settings) == LEGACY_BREAKER_CONTOUR
