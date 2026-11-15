"""The operation `classify` (release 3.9.0, LSBN-S-9).

    The classifier is the consumer of the `summary` slot group in the
    `operation` mode: the grooming (after a summarization), the reclass
    tasks of the `nodes` job and its sweep — plus the `warnings` job, the
    third consumer, which highlights the notes that are not in their place.
    Every operation owns its breaker (the design, B4) and every failure is
    recorded at the breaker of its own operation; the `legacy` mode keeps
    the shared breaker of the slot group (as before).

    Checkable contracts (the criteria of the step):
    1. a failure of the classifier pauses the breaker of the `classify`
       operation only;
    2. the `warnings` job works the `classify` operation (its key is the
       operation, not the `summary` slot) and its failure opens that
       breaker; the `legacy` mode keeps the slot and the journal-only
       failure;
    3. the probe of the `classify` operation is a real one — it reaches the
       client or the decision service (not the `CHECK_OK` stub);
    4. an unavailable decision blocks `classify` and nobody else;
    5. the attention counters carry the operation names only;
    6. a failure does not lose the work (the note/task waits for the next
       run);
    7. the `legacy` mode did not change (the snapshot below was taken on the
       base commit `94a13d1`, before this step).

    A unit without a network: the models are fakes, the probe is checked on
    the real `ClassificationService` over `httpx.MockTransport`.
"""

from __future__ import annotations

import asyncio
import logging

import httpx
import pytest
from fakes import (
    FailingEmbedder,
    FixedClassifier,
    FixedSummarizer,
    HashEmbedder,
)

from app.config import get_settings
from app.services.classifier import Classification, ClassificationService
from app.services.decision import DecisionService
from app.services.jobs import BackoffState, run_loop
from app.services.llm_client import (
    CHECK_OK,
    CHECK_UNREACHABLE,
    DECISION_CHECK_OK,
    LLMClient,
    SlotSpec,
)
from app.services.namespaces import NamespaceService
from app.services.notes import NoteService
from app.services.operations import OPERATION_REGISTRY
from app.services.slot_outcome import SlotOutcome
from app.services.warnings import WARNINGS_JOB, build_warnings_job
from app.services.worker import (
    NODES_JOB,
    NODES_RECLASS_KIND,
    BackgroundWorker,
    build_summary_job,
)
from app.storage.db import init_db, session, transaction

DIM = 8
#: The model of the classify operation (the operation's own env in the
#: `operation` mode, the slot's env in the `legacy` mode).
MODEL = "test-classify-model"

#: The `legacy` contour snapshot — taken on the base commit `94a13d1` BEFORE
#: this step (the `legacy` mode must not change in any respect): the breaker
#: keys, the blocked slots, the `warnings` job's slot and its gate with an
#: open breaker, the breakers that the three failure paths of the classifier
#: pause and the attention counters.
LEGACY_CLASSIFY_CONTOUR: dict = {
    "breakers": ["embedding", "judge", "summary"],
    "slot_blocked": {"embedding": False, "judge": False, "summary": False},
    "warnings_slot": "summary",
    "warnings_gate_open_breaker": False,
    "warnings_is_gated_open_breaker": True,
    "paused_after_grooming_failure": ["summary"],
    "paused_after_reclass_failure": ["summary"],
    "paused_after_warnings_failure": [],
    "attention_by_slot": {"nodes": 1},
}


def _settings(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    mode: str,
    decision_fallback: str | None = None,
):
    """The test settings: own database, the slot model, a one-failure pause."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "notes.db"))
    monkeypatch.setenv("EMBEDDING_DIM", str(DIM))
    monkeypatch.setenv("SUMMARY_SHORT_TEXT_CHARS", "0")
    # The pause opens at the first failure — otherwise the third failure
    # would open it.
    monkeypatch.setenv("SLOT_PAUSE_AFTER", "1")
    # The model of the classifier: the slot's env in the `legacy` mode, the
    # operation's own env in the `operation` mode (the slot layer does not
    # take part there — the design, B1).
    monkeypatch.setenv("SUMMARY_MODEL", MODEL)
    monkeypatch.setenv("SUMMARY_BASE_URL", "http://127.0.0.1:1")
    monkeypatch.setenv("CLASSIFY_MODEL", MODEL)
    monkeypatch.setenv("CLASSIFY_BASE_URL", "http://127.0.0.1:1")
    # The decision slot has its own env (the reserve slot — the F-12 mechanics).
    monkeypatch.setenv("DECISION_BASE_URL", "http://127.0.0.1:1")
    monkeypatch.setenv("DECISION_MODEL", "decision-test")
    if decision_fallback is not None:
        monkeypatch.setenv("DECISION_FALLBACK", decision_fallback)
    if mode == "operation":
        monkeypatch.setenv("SLOT_MODEL", "operation")
    else:
        monkeypatch.delenv("SLOT_MODEL", raising=False)
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    return settings


@pytest.fixture
def legacy_settings(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """The `legacy` mode (the default): the contour is as before."""
    return _settings(tmp_path, monkeypatch, mode="legacy")


@pytest.fixture
def operation_settings(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """The `operation` mode: the breakers are keyed by the operation."""
    return _settings(tmp_path, monkeypatch, mode="operation")


@pytest.fixture
def blocked_decision_settings(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """The `operation` mode with the decision slot down and the fallback off."""
    return _settings(
        tmp_path, monkeypatch, mode="operation", decision_fallback="false"
    )


def _worker(
    settings,
    *,
    classifier=None,
    decision=None,
) -> BackgroundWorker:
    """A worker with the classifier of the operation."""
    return BackgroundWorker(
        settings,
        HashEmbedder(DIM),
        FixedSummarizer(),
        classifier=classifier,
        decision=decision,
    )


def _paused(worker: BackgroundWorker, key: str) -> bool:
    """Is the breaker under the key paused right now?"""
    breaker = worker._breakers.get(key)
    return breaker is not None and breaker.is_paused()


def _outcome(worker: BackgroundWorker, key: str) -> str | None:
    """The last outcome of the breaker under the key."""
    breaker = worker._breakers.get(key)
    if breaker is None:
        return None
    return breaker.snapshot()["last_outcome"]


def _paused_breakers(worker: BackgroundWorker) -> list[str]:
    """The names of the paused breakers (sorted)."""
    return sorted(
        name for name, breaker in worker._breakers.items() if breaker.is_paused()
    )


def _default_note(settings) -> int:
    """A fresh `default` note (the grooming's candidate)."""
    notes = NoteService(settings, FailingEmbedder())
    return notes.save("заметка про рабочий проект")["id"]


def _placed_note(settings, note_id: int = 2) -> int:
    """A placed note with a ready summary (the `warnings` job's candidate)."""
    NamespaceService(settings).create("work", "Рабочие заметки.")
    with session(settings) as conn, transaction(conn):
        conn.execute(
            "INSERT INTO notes (id, title, text, summary, summary_status, namespace, "
            "updated_at) VALUES (?, ?, ?, ?, ?, ?, "
            "strftime('%Y-%m-%dT%H:%M:%SZ','now'))",
            (
                note_id,
                f"Заметка {note_id}",
                "текст заметки",
                "Сводка",
                "ok",
                "work",
            ),
        )
    return note_id


def _attention_row(
    worker: BackgroundWorker, slot: str, kind: str, note_id: int = 1
) -> None:
    """A task row frozen in attention (the counter's source)."""
    worker._ensure_job_table()
    with session(worker._settings) as conn, transaction(conn):
        conn.execute(
            "INSERT INTO worker_jobs (slot, kind, note_id, status, attempts, "
            "last_outcome, attention_at) VALUES (?, ?, ?, 'pending', 9, "
            "'off_contract', strftime('%Y-%m-%dT%H:%M:%SZ','now'))",
            (slot, kind, note_id),
        )


def _events(caplog, name: str) -> list:
    """The journal records of an event."""
    return [r for r in caplog.records if getattr(r, "event", None) == name]


# --- criterion 1: a failure of the classifier → its own breaker ----------------


def test_classify_failure_pauses_the_classify_breaker_only(operation_settings) -> None:
    """A failure of the classifier: the pause is only at `classify`."""
    worker = _worker(operation_settings, classifier=FixedClassifier(fail=True))
    _default_note(operation_settings)

    assert worker.process_summary_pending() == 1  # the note is summarized

    assert _paused(worker, "classify") is True
    for operation in (
        "summarize",
        "title",
        "judge",
        "structure_judge",
        "describe",
        "embed",
    ):
        assert _paused(worker, operation) is False, operation


def test_the_classify_failure_is_visible_in_its_breaker(operation_settings) -> None:
    """The failure is recorded at the breaker of the operation."""
    worker = _worker(operation_settings, classifier=FixedClassifier(fail=True))
    _default_note(operation_settings)
    worker.process_summary_pending()

    assert _outcome(worker, "classify") == SlotOutcome.OFF_CONTRACT.value
    assert _outcome(worker, "summarize") is None


def test_the_journal_names_no_fake_classify_job(operation_settings, caplog) -> None:
    """The journal: the job is the factual one, the operation — the breaker key."""
    worker = _worker(operation_settings, classifier=FixedClassifier(fail=True))
    _default_note(operation_settings)
    with caplog.at_level(logging.INFO, logger="app"):
        worker.process_summary_pending()

    records = _events(caplog, "classify_failed")
    assert len(records) == 1
    assert records[0].job == "summary"  # the factual job (the grooming)
    assert records[0].slot == "classify"  # the breaker key of the operation


def test_the_reclass_failure_pauses_the_classify_breaker(operation_settings) -> None:
    """A failure of the classifier in the reclass task — the same breaker."""
    worker = _worker(operation_settings, classifier=FixedClassifier(fail=True))
    note_id = _placed_note(operation_settings)
    worker._ensure_job(NODES_JOB, NODES_RECLASS_KIND, note_id)

    worker.process_nodes()

    assert _paused(worker, "classify") is True
    assert _paused(worker, "summarize") is False


# --- criterion 2: the `warnings` job works the `classify` operation -----------


def test_the_warnings_job_works_the_classify_operation(operation_settings) -> None:
    """The `warnings` job: the breaker key is the operation, not the slot."""
    worker = _worker(
        operation_settings, classifier=FixedClassifier(Classification("work", 0.9))
    )
    spec = build_warnings_job(worker, operation_settings)

    assert spec is not None
    assert spec.slot == "classify"
    assert spec.slot != "summary"  # the slot group is not the key any more


def test_the_warnings_failure_opens_the_classify_breaker(operation_settings) -> None:
    """A failure of the classifier in the `warnings` job → the `classify` breaker."""
    worker = _worker(operation_settings, classifier=FixedClassifier(fail=True))
    spec = build_warnings_job(worker, operation_settings)
    _placed_note(operation_settings)

    assert asyncio.run(spec.process()) == 0  # the note stays unchecked

    assert _paused(worker, "classify") is True
    assert _paused(worker, "summarize") is False
    assert _paused(worker, "title") is False


def test_the_warnings_job_keeps_the_summary_slot_in_legacy(legacy_settings) -> None:
    """The `legacy` mode: the `warnings` job stays on the `summary` slot."""
    worker = _worker(
        legacy_settings, classifier=FixedClassifier(Classification("work", 0.9))
    )
    spec = build_warnings_job(worker, legacy_settings)

    assert spec is not None
    assert spec.slot == "summary"


def test_the_warnings_failure_touches_no_breaker_in_legacy(legacy_settings) -> None:
    """The `legacy` mode: the failure of the `warnings` job stays in the journal."""
    worker = _worker(legacy_settings, classifier=FixedClassifier(fail=True))
    spec = build_warnings_job(worker, legacy_settings)
    _placed_note(legacy_settings)

    assert asyncio.run(spec.process()) == 0

    assert _paused_breakers(worker) == []


def test_the_warnings_failure_keeps_the_note_a_candidate(
    operation_settings,
) -> None:
    """The failed note is checked by the next run (the marker is not set)."""
    worker = _worker(operation_settings, classifier=FixedClassifier(fail=True))
    spec = build_warnings_job(worker, operation_settings)
    note_id = _placed_note(operation_settings)

    asyncio.run(spec.process())

    with session(operation_settings) as conn:
        row = conn.execute(
            "SELECT warnings_at FROM notes WHERE id = ?", (note_id,)
        ).fetchone()
    assert row["warnings_at"] is None


# --- criterion 3: the probe of the operation is a real one -------------------


def test_the_classifier_has_a_probe(operation_settings) -> None:
    """The classifier answers the knock of the breaker (the protocol)."""
    assert hasattr(ClassificationService, "probe")
    assert callable(ClassificationService.probe)


def test_the_probe_reaches_the_client(operation_settings) -> None:
    """The generative path: the knock goes to the client of the operation."""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"models": [{"name": MODEL}]})

    service = ClassificationService(
        operation_settings, httpx.MockTransport(handler)
    )

    assert service.probe() == CHECK_OK
    assert requests, "the client was not asked"


def test_a_failing_probe_is_not_a_silent_success(operation_settings) -> None:
    """A lying client: the probe does not answer `CHECK_OK`."""
    service = ClassificationService(
        operation_settings,
        httpx.MockTransport(lambda request: httpx.Response(500, json={})),
    )

    assert service.probe() != CHECK_OK
    assert service.probe() == CHECK_UNREACHABLE


def test_the_operation_probe_reaches_the_classifier(operation_settings) -> None:
    """In the `operation` mode the knock of `classify` goes to the classifier."""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"models": [{"name": MODEL}]})

    worker = _worker(
        operation_settings,
        classifier=ClassificationService(
            operation_settings, httpx.MockTransport(handler)
        ),
    )

    assert worker._probe_slot("classify") == CHECK_OK
    assert requests, "the classifier's client was not asked"


def test_the_probe_goes_to_the_decision_service(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The decision path: the knock goes to the decision slot."""
    settings = _settings(
        tmp_path, monkeypatch, mode="operation", decision_fallback="false"
    )
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"answers": {"probe": {"noul": 1.0}}})

    decider = DecisionService(
        settings,
        llm=LLMClient(
            SlotSpec.for_decision(settings), transport=httpx.MockTransport(handler)
        ),
    )
    service = ClassificationService(settings, decider=decider)

    assert service.probe() == DECISION_CHECK_OK
    assert requests, "the decision slot was not asked"


# --- criterion 4: the decision rule ------------------------------------------


class UndecidingDecision:
    """The decision slot: there, but unable to decide (the F-12 reserve)."""

    def __init__(self) -> None:
        self.probes = 0

    def can_decide(self) -> bool:
        return False

    def probe(self, operation: str | None = None) -> str:
        self.probes += 1
        return "unreachable"

    def snapshot(self) -> dict:
        return {
            "enabled": True,
            "ok": False,
            "breaker": {"last_outcome": None, "paused_until": None},
            "answers": 0,
            "fallbacks": 0,
            "by_reason": {},
        }


def test_an_unavailable_decision_blocks_the_classify_operation(
    blocked_decision_settings,
) -> None:
    """The decision slot is down and the fallback is off: `classify` waits."""
    worker = _worker(blocked_decision_settings, decision=UndecidingDecision())

    assert worker._operation_blocked("classify") is True
    for operation in ("summarize", "title", "merge", "describe", "embed"):
        assert worker._operation_blocked(operation) is False, operation


def test_the_warnings_job_is_gated_by_the_decision_rule(
    blocked_decision_settings,
) -> None:
    """The `warnings` job works the classify operation — it waits as well."""
    worker = _worker(
        blocked_decision_settings,
        classifier=FixedClassifier(Classification("work", 0.9)),
        decision=UndecidingDecision(),
    )
    spec = build_warnings_job(worker, blocked_decision_settings)

    assert spec.is_gated is not None and spec.is_gated() is True


def test_the_classify_operation_goes_with_the_fallback_on(operation_settings) -> None:
    """With the fallback on the decision does not block the operation."""
    worker = _worker(operation_settings, decision=UndecidingDecision())

    assert worker._operation_blocked("classify") is False


# --- criterion 5: the attention counters carry the operations only ------------


def test_the_attention_counters_carry_the_operations_only(operation_settings) -> None:
    """No non-operation key lands in the counters of `/health`."""
    worker = _worker(operation_settings)
    _attention_row(worker, NODES_JOB, NODES_RECLASS_KIND)
    _attention_row(worker, NODES_JOB, "unknown-kind", note_id=2)

    counts = worker._attention_by_operation()

    assert set(counts) <= set(OPERATION_REGISTRY)
    assert "nodes" not in counts
    assert counts["classify"] == 1  # the known pair


def test_the_legacy_attention_counts_the_slot_groups(legacy_settings) -> None:
    """The `legacy` mode: the counters are the slot groups, as before."""
    worker = _worker(legacy_settings)
    _attention_row(worker, NODES_JOB, NODES_RECLASS_KIND)

    assert worker._attention_by_slot() == {"nodes": 1}


def test_the_attention_event_names_a_real_job(operation_settings, caplog) -> None:
    """The `job_attention` event names a real job of the registry."""
    worker = _worker(operation_settings)
    _attention_row(worker, NODES_JOB, NODES_RECLASS_KIND)

    with caplog.at_level(logging.INFO, logger="app"):
        worker._note_attempt(1, SlotOutcome.OFF_CONTRACT)

    records = _events(caplog, "job_attention")
    assert len(records) == 1
    assert records[0].job == NODES_JOB  # the factual job of the reclass task
    assert records[0].slot == "classify"  # the owner — the operation


# --- criterion 6: the work is not lost ----------------------------------------


def test_a_classify_failure_leaves_the_note_unclassified(operation_settings) -> None:
    """The note stays in `default`, the markup is not written."""
    worker = _worker(operation_settings, classifier=FixedClassifier(fail=True))
    note_id = _default_note(operation_settings)

    worker.process_summary_pending()

    with session(operation_settings) as conn:
        row = conn.execute(
            "SELECT namespace, classified_at FROM notes WHERE id = ?", (note_id,)
        ).fetchone()
    assert row["namespace"] == "default"
    assert row["classified_at"] is None


def test_a_reclass_failure_leaves_the_task_pending(operation_settings) -> None:
    """The reclass task waits for the next run (the work is not lost)."""
    worker = _worker(operation_settings, classifier=FixedClassifier(fail=True))
    note_id = _placed_note(operation_settings)
    worker._ensure_job(NODES_JOB, NODES_RECLASS_KIND, note_id)

    worker.process_nodes()

    assert worker._pending_jobs(NODES_JOB, NODES_RECLASS_KIND, 10)


def test_the_work_returns_after_the_pause(operation_settings) -> None:
    """After the pause the task is taken again and the work is done."""
    classifier = FixedClassifier(fail=True)
    worker = _worker(operation_settings, classifier=classifier)
    note_id = _placed_note(operation_settings)
    worker._ensure_job(NODES_JOB, NODES_RECLASS_KIND, note_id)

    worker.process_nodes()
    assert _paused(worker, "classify") is True

    # The model is back: the breaker is resumed and the task is taken.
    classifier.fail = False
    worker._breakers["classify"].note_success()

    assert worker.process_nodes() == 1
    assert worker._pending_jobs(NODES_JOB, NODES_RECLASS_KIND, 10) == []


# --- the pause of the operation stops the classifier calls (LSBN-S-9) --------


def test_the_paused_classify_stops_the_grooming_call(operation_settings) -> None:
    """The pause of `classify`: the grooming does not call the model."""
    classifier = FixedClassifier(Classification("work", 0.9))
    worker = _worker(operation_settings, classifier=classifier)
    note_id = _default_note(operation_settings)
    worker._breakers["classify"].note_failure(SlotOutcome.AUTH)
    assert _paused(worker, "classify") is True

    assert worker.process_summary_pending() == 1  # the note is summarized

    assert classifier.calls == [], "the classifier was called on the pause"
    with session(operation_settings) as conn:
        row = conn.execute(
            "SELECT namespace, classified_at FROM notes WHERE id = ?", (note_id,)
        ).fetchone()
    assert row["classified_at"] is None  # the note waits (the sweep takes it)


def test_the_work_is_taken_after_the_pause_expires(operation_settings) -> None:
    """After the pause the note is classified (the sweep of the `nodes` job)."""
    classifier = FixedClassifier(Classification("work", 0.9))
    worker = _worker(operation_settings, classifier=classifier)
    note_id = _default_note(operation_settings)
    worker._breakers["classify"].note_failure(SlotOutcome.AUTH)
    worker.process_summary_pending()
    assert classifier.calls == []

    # The model is back: the breaker is resumed and the work is taken.
    worker._breakers["classify"].note_success()
    assert worker.process_nodes() == 1

    assert len(classifier.calls) == 1
    with session(operation_settings) as conn:
        row = conn.execute(
            "SELECT classified_at FROM notes WHERE id = ?", (note_id,)
        ).fetchone()
    assert row["classified_at"] is not None


def test_the_paused_classify_stops_the_reclass_call(operation_settings) -> None:
    """The pause of `classify`: the reclass task does not call the model."""
    classifier = FixedClassifier(Classification("work", 0.9))
    worker = _worker(operation_settings, classifier=classifier)
    note_id = _placed_note(operation_settings)
    worker._ensure_job(NODES_JOB, NODES_RECLASS_KIND, note_id)
    worker._breakers["classify"].note_failure(SlotOutcome.AUTH)

    assert worker.process_nodes() == 0

    assert classifier.calls == [], "the classifier was called on the pause"
    assert worker._pending_jobs(NODES_JOB, NODES_RECLASS_KIND, 10)


def test_the_paused_classify_stops_the_sweep_call(operation_settings) -> None:
    """The pause of `classify`: the sweep does not call the model."""
    classifier = FixedClassifier(Classification("work", 0.9))
    worker = _worker(operation_settings, classifier=classifier)
    _default_note(operation_settings)
    worker.process_summary_pending()
    classifier.calls.clear()
    worker._breakers["classify"].note_failure(SlotOutcome.AUTH)

    assert worker.process_nodes() == 0

    assert classifier.calls == [], "the classifier was called on the pause"


def test_the_paused_classify_stops_the_warnings_call(
    operation_settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pause of `classify`: the `warnings` job does not call the model."""
    classifier = FixedClassifier(Classification("work", 0.9))
    worker = _worker(operation_settings, classifier=classifier)
    spec = build_warnings_job(worker, operation_settings)
    _placed_note(operation_settings)
    worker._breakers["classify"].note_failure(SlotOutcome.AUTH)

    # The gate of the job reads the breaker of the operation — the batch is
    # not taken, the model is not called.
    assert spec.gate is not None and spec.gate() is False
    assert spec.is_gated is not None and spec.is_gated() is True

    async def instant_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", instant_sleep)
    iterations: list[int] = []

    def stopping() -> bool:
        iterations.append(1)
        return len(iterations) > 1

    asyncio.run(run_loop(spec, stopping, BackoffState(spec.interval_sec)))

    assert classifier.calls == [], "the classifier was called on the pause"


def test_the_legacy_classify_failure_stops_the_summary_job(legacy_settings) -> None:
    """The `legacy` mode: the classifier's failure stops the `summary` job.

    The shared breaker of the `summary` slot group is read by the gate of the
    job — the pause stops the job as a whole (as before). The `operation` mode
    has its own pause of the operation: the check is inside the path.
    """
    worker = _worker(legacy_settings, classifier=FixedClassifier(fail=True))
    _default_note(legacy_settings)
    worker.process_summary_pending()
    assert _paused(worker, "summary") is True

    spec = build_summary_job(worker, legacy_settings)
    assert spec is not None
    assert spec.gate is not None and spec.gate() is False
    assert spec.is_gated is not None and spec.is_gated() is True


def test_the_legacy_grooming_calls_the_classifier_on_the_pause(
    legacy_settings,
) -> None:
    """The `legacy` mode: the pause of the slot is NOT a path-level check.

    The gate of the job is the mechanism there (the whole job stops); the
    path itself is not gated by the operation (as before).
    """
    classifier = FixedClassifier(fail=True)
    worker = _worker(legacy_settings, classifier=classifier)
    _default_note(legacy_settings)
    worker._breakers["summary"].note_failure(SlotOutcome.AUTH)

    worker.process_summary_pending()

    assert classifier.calls, "the legacy path is not gated by the operation"


# --- criterion 7: the legacy mode did not change -----------------------------


def _legacy_contour(settings) -> dict:
    """The `legacy` contour: the snapshot of the behaviour (the base commit)."""
    worker = _worker(settings, classifier=FixedClassifier(fail=True))
    snapshot = {
        "breakers": sorted(worker._breakers),
        "slot_blocked": {
            name: worker._slot_blocked(name)
            for name in ("embedding", "summary", "judge")
        },
    }
    # The `warnings` job: its slot and its gate with an open breaker.
    spec = build_warnings_job(worker, settings)
    snapshot["warnings_slot"] = spec.slot if spec else None
    worker._breakers["summary"].note_failure(SlotOutcome.AUTH)
    snapshot["warnings_gate_open_breaker"] = spec.gate() if spec and spec.gate else None
    snapshot["warnings_is_gated_open_breaker"] = (
        spec.is_gated() if spec and spec.is_gated else None
    )
    worker._breakers["summary"].note_success()
    # The failure path of the grooming (the summary slot group is shared).
    _default_note(settings)
    worker.process_summary_pending()
    snapshot["paused_after_grooming_failure"] = _paused_breakers(worker)
    # The failure path of the reclass task (the same slot group).
    worker._breakers["summary"].note_success()
    note_id = _placed_note(settings, 2)
    worker._ensure_job(NODES_JOB, NODES_RECLASS_KIND, note_id)
    worker.process_nodes()
    snapshot["paused_after_reclass_failure"] = _paused_breakers(worker)
    # The failure path of the `warnings` job (no breaker at all in the legacy).
    worker._breakers["summary"].note_success()
    warnings_spec = build_warnings_job(worker, settings)
    asyncio.run(warnings_spec.process())
    snapshot["paused_after_warnings_failure"] = _paused_breakers(worker)
    # The attention counters (the legacy form: the slot groups).
    _attention_row(worker, NODES_JOB, NODES_RECLASS_KIND, note_id=3)
    snapshot["attention_by_slot"] = worker._attention_by_slot()
    return snapshot


def test_legacy_classify_contour_snapshot_is_unchanged(legacy_settings) -> None:
    """The `legacy` contour is exactly the snapshot of the base commit."""
    assert _legacy_contour(legacy_settings) == LEGACY_CLASSIFY_CONTOUR
