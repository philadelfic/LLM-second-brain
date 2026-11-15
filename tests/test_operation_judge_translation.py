"""The operations `judge` and `structure_judge` (release 3.9.0, LSBN-S-8).

    The `judge` slot group serves two judges: the dedup judge (`judge`) and
    the structural judge (`structure_judge`). In the `operation` mode
    (`SLOT_MODEL=operation`) every operation owns its breaker: a failure of
    one judge must not silence the other. The `legacy` mode keeps the shared
    breaker of the slot group (as before).

    Checkable contracts (the criteria of the step):
    1. a failure of the `judge` operation pauses its own breaker only;
    2. the breakers of `judge` and `structure_judge` are independent;
    3. a failure of the structural judge goes to its own breaker and the
       candidate is not lost (it waits for the next run);
    4. the markers are parsed as before; an answer without a marker is
       `off_contract` at the breaker of its own operation;
    5. the decision rule blocks `judge` only (the structural judge does not
       walk into the decision slot);
    6. the `legacy` mode did not change (the snapshot below was taken on the
       base commit `9631164`, before this step).

    A unit without a network: the models are fakes, the encoder —
    `HashEmbedder`; the two marker parsers are checked against the real
    services on `httpx.MockTransport`.
"""

from __future__ import annotations

import httpx
import pytest
from fakes import (
    FailingEmbedder,
    FixedClassifier,
    FixedDescriber,
    FixedSummarizer,
    HashEmbedder,
    RecordingDedup,
    ScriptedJudge,
    ScriptedStructureJudge,
    clear_seeded_skills,
)

from app.config import get_settings
from app.services.classifier import Classification
from app.services.judge import JudgeError, JudgeService
from app.services.namespaces import NamespaceService
from app.services.notes import NoteService
from app.services.promotion import (
    PromotionService,
    StructureJudgeError,
    StructureJudgeService,
    Verdict,
)
from app.services.slot_outcome import SlotOutcome
from app.services.worker import BackgroundWorker, build_judge_job
from app.storage.db import init_db, session

DIM = 8
#: The promotion threshold (the group of default notes that becomes a node).
THRESHOLD = 15

#: The `legacy` contour snapshot — taken on the base commit `9631164` BEFORE
#: this step (the `legacy` mode must not change in any respect): the breaker
#: keys, the blocked slots, the gate of the judge job with an open breaker,
#: the think flag of the structural judge and the breakers that the two
#: failure paths pause.
LEGACY_JUDGE_CONTOUR: dict = {
    "breakers": ["embedding", "judge", "summary"],
    "slot_blocked": {"embedding": False, "judge": False, "summary": False},
    "judge_gate_open_breaker": False,
    "judge_is_gated_open_breaker": True,
    "structure_judge_think": True,
    "paused_after_judge_failure": ["judge"],
    "paused_after_structure_judge_failure": [],
}


def _chat_body(content: str) -> dict:
    """The ollama chat answer (the shape the client reads)."""
    return {
        "model": "test-model",
        "created_at": "2026-08-30T12:00:00Z",
        "message": {
            "role": "assistant",
            "content": content,
            "thinking": "",
        },
        "done": True,
    }


class UndecidingDecision:
    """The decision slot: there, but unable to decide (the F-12 reserve)."""

    def __init__(self) -> None:
        self.probes = 0

    def can_decide(self) -> bool:
        return False

    def probe(self) -> str:
        self.probes += 1
        return "probe:decision"

    def snapshot(self) -> dict:
        return {
            "enabled": True,
            "ok": False,
            "breaker": {"last_outcome": None, "paused_until": None},
            "answers": 0,
            "fallbacks": 0,
            "by_reason": {},
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
    # The legacy source of the structural judge's think flag — the `operation`
    # mode must NOT take it into account (the design, B1).
    monkeypatch.setenv("NAMESPACE_JUDGE_THINK", "true")
    # The `operation` mode does not read the slot layer: the operation's own
    # env (the model and the address) is needed for a real client.
    monkeypatch.setenv("STRUCTURE_JUDGE_MODEL", "test-structure-judge-model")
    monkeypatch.setenv("STRUCTURE_JUDGE_BASE_URL", "http://127.0.0.1:1")
    if decision_fallback is not None:
        monkeypatch.setenv("DECISION_FALLBACK", decision_fallback)
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


@pytest.fixture
def blocked_decision_settings(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """The `operation` mode with the decision slot down and the fallback off."""
    return _settings(
        tmp_path, monkeypatch, mode="operation", decision_fallback="false"
    )


def _worker(
    settings,
    *,
    judge=None,
    promoter=None,
    classifier=None,
    decision=None,
) -> BackgroundWorker:
    """A worker with the judges of the two operations."""
    return BackgroundWorker(
        settings,
        HashEmbedder(DIM),
        FixedSummarizer(),
        judge=judge if judge is not None else ScriptedJudge(),
        classifier=classifier,
        promoter=promoter,
        decision=decision,
    )


def _promoter(settings, structure_judge) -> PromotionService:
    """The promotion trigger with a scriptable structural judge."""
    return PromotionService(
        settings,
        embedding=HashEmbedder(DIM),
        describer=FixedDescriber("Заметки о СУБО 2020."),
        judge=structure_judge,
        namespaces=NamespaceService(settings),
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


def _candidate_group(settings, count: int = THRESHOLD) -> None:
    """A group of default notes that reaches the promotion threshold."""
    NamespaceService(settings).create("work", "Рабочие заметки.")
    notes = NoteService(settings)
    for index in range(count):
        notes.save(f"заметка про СУБО №{index}")


def _dedup_jobs(settings, judge) -> BackgroundWorker:
    """Two notes with ready vectors → two pending dedup jobs (the judge queue)."""
    notes = NoteService(settings, FailingEmbedder())
    notes.save("первая заметка отказа судьи")
    notes.save("вторая заметка отказа судьи")
    worker = BackgroundWorker(
        settings,
        HashEmbedder(DIM),
        FixedSummarizer(),
        RecordingDedup(candidates=[(1, 0.94)]),
        judge=judge,
    )
    assert worker.process_pending() == 2  # the dedup jobs are created
    return worker


# --- criterion 1: a failure of the judge operation pauses its own breaker ----


def test_judge_failure_pauses_the_judge_breaker_only(operation_settings) -> None:
    """A failure of the dedup judge: the pause is only at `judge`."""
    worker = _dedup_jobs(operation_settings, ScriptedJudge(fail=True))
    assert worker.process_judge_pending() == 1  # the failed job stays pending
    assert _paused(worker, "judge") is True
    assert _paused(worker, "structure_judge") is False
    assert _paused(worker, "summarize") is False
    assert _paused(worker, "title") is False


def test_judge_failure_leaves_the_work_pending(operation_settings) -> None:
    """The failure of the judge does not lose the work (NFR-3)."""
    worker = _dedup_jobs(operation_settings, ScriptedJudge(fail=True))
    worker.process_judge_pending()
    assert worker._pending_jobs("judge", "dedup", 10)


def test_judge_operation_owns_its_breaker_in_operation_mode(
    operation_settings,
) -> None:
    """The breaker key of the judge operation is the operation itself."""
    worker = _worker(operation_settings)
    assert worker._operation_slot("judge") == "judge"
    assert worker._operation_slot("structure_judge") == "structure_judge"
    assert "judge" in worker._breakers
    assert "structure_judge" in worker._breakers


# --- criterion 2: the two breakers are independent ----------------------------


def test_judge_pause_does_not_block_the_promotion_path(operation_settings) -> None:
    """The dedup judge's breaker is closed — the structural judge still works."""
    structure_judge = ScriptedStructureJudge(default=Verdict("create"))
    worker = _worker(
        operation_settings,
        promoter=_promoter(operation_settings, structure_judge),
        classifier=FixedClassifier(Classification("work/subo", 0.7)),
    )
    worker._breakers["judge"].note_failure(SlotOutcome.AUTH)
    assert _paused(worker, "judge") is True

    _candidate_group(operation_settings)
    worker.process_summary_pending()

    assert structure_judge.review_calls, "the structural judge was not asked"
    assert _paused(worker, "structure_judge") is False


def test_structure_judge_pause_does_not_block_the_judge_queue(
    operation_settings,
) -> None:
    """The structural judge's breaker is closed — the judge queue goes."""
    worker = _dedup_jobs(operation_settings, ScriptedJudge())
    worker._breakers["structure_judge"].note_failure(SlotOutcome.AUTH)
    assert _paused(worker, "structure_judge") is True

    spec = build_judge_job(worker, operation_settings)
    assert spec.is_gated is not None and spec.is_gated() is False
    assert spec.gate is not None and spec.gate() is True
    assert worker._judge_queue_empty() is False


def test_structure_judge_pause_stops_the_structural_judge(operation_settings) -> None:
    """While the structural judge's breaker is closed — the judge is not asked."""
    structure_judge = ScriptedStructureJudge(default=Verdict("create"))
    worker = _worker(
        operation_settings,
        promoter=_promoter(operation_settings, structure_judge),
        classifier=FixedClassifier(Classification("work/subo", 0.7)),
    )
    worker._breakers["structure_judge"].note_failure(SlotOutcome.AUTH)

    _candidate_group(operation_settings)
    worker.process_summary_pending()

    assert structure_judge.review_calls == []
    assert _paused(worker, "structure_judge") is True


# --- criterion 3: a failure of the structural judge → its own breaker ---------


def test_structure_judge_failure_pauses_its_own_breaker(operation_settings) -> None:
    """The failure of the structural judge: the pause is only at `structure_judge`."""
    structure_judge = ScriptedStructureJudge(fail=True)
    worker = _worker(
        operation_settings,
        promoter=_promoter(operation_settings, structure_judge),
        classifier=FixedClassifier(Classification("work/subo", 0.7)),
    )
    _candidate_group(operation_settings)
    worker.process_summary_pending()

    assert _paused(worker, "structure_judge") is True
    assert _paused(worker, "judge") is False
    assert _paused(worker, "summarize") is False


def test_structure_judge_failure_does_not_lose_the_candidate(
    operation_settings,
) -> None:
    """The candidate stays without a verdict and waits for the next run."""
    structure_judge = ScriptedStructureJudge(fail=True)
    worker = _worker(
        operation_settings,
        promoter=_promoter(operation_settings, structure_judge),
        classifier=FixedClassifier(Classification("work/subo", 0.7)),
    )
    _candidate_group(operation_settings)
    worker.process_summary_pending()

    with session(operation_settings) as conn:
        node = conn.execute(
            "SELECT COUNT(*) FROM namespaces WHERE path = 'work/subo'"
        ).fetchone()[0]
        in_default = conn.execute(
            "SELECT COUNT(*) FROM notes WHERE namespace = 'default'"
        ).fetchone()[0]
    assert node == 0  # no node was created
    assert in_default == THRESHOLD  # the group stays in default (a retry later)
    assert structure_judge.review_calls, "the judge was asked (and failed)"


def test_structure_judge_failure_is_visible_in_the_breaker(operation_settings) -> None:
    """The failure is recorded at the breaker of the operation."""
    structure_judge = ScriptedStructureJudge(fail=True)
    worker = _worker(
        operation_settings,
        promoter=_promoter(operation_settings, structure_judge),
        classifier=FixedClassifier(Classification("work/subo", 0.7)),
    )
    _candidate_group(operation_settings)
    worker.process_summary_pending()

    assert _outcome(worker, "structure_judge") == SlotOutcome.OFF_CONTRACT.value
    assert _outcome(worker, "judge") is None


# --- criterion 4: the markers are parsed as before ---------------------------


def test_dedup_markers_are_parsed_as_before(operation_settings) -> None:
    """`DUPLICATE`/`NOT DUPLICATE` — the parser of the code is unchanged."""
    assert JudgeService._verdict("**DUPLICATE**") is True
    assert JudgeService._verdict("**NOT DUPLICATE**") is False
    assert JudgeService._verdict("DUPLICATE") is True
    with pytest.raises(JudgeError) as failure:
        JudgeService._verdict("maybe")
    assert failure.value.outcome is SlotOutcome.OFF_CONTRACT


def test_structure_markers_are_parsed_as_before(operation_settings) -> None:
    """`CREATE`/`MERGE <path>`/`REJECT` — the parser of the code is unchanged."""
    assert StructureJudgeService._parse("CREATE").action == "create"
    merged = StructureJudgeService._parse("**MERGE** work/subo")
    assert merged.action == "merge"
    assert merged.target == "work/subo"
    assert StructureJudgeService._parse("REJECT").action == "reject"
    with pytest.raises(StructureJudgeError) as failure:
        StructureJudgeService._parse("I do not know")
    assert failure.value.outcome is SlotOutcome.OFF_CONTRACT


def test_judge_without_marker_is_off_contract_in_its_breaker(
    operation_settings,
) -> None:
    """The dedup judge answers without a marker → `off_contract` at `judge`."""
    service = JudgeService(
        operation_settings,
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=_chat_body("no marker here"))
        ),
    )
    worker = _dedup_jobs(operation_settings, service)
    worker.process_judge_pending()
    assert _outcome(worker, "judge") == SlotOutcome.OFF_CONTRACT.value
    assert _paused(worker, "judge") is True
    assert _paused(worker, "structure_judge") is False


def test_structure_judge_without_marker_is_off_contract_in_its_breaker(
    operation_settings,
) -> None:
    """The structural judge answers without a marker → `off_contract`."""
    service = StructureJudgeService(
        operation_settings,
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=_chat_body("I do not know"))
        ),
    )
    worker = _worker(
        operation_settings,
        promoter=_promoter(operation_settings, service),
        classifier=FixedClassifier(Classification("work/subo", 0.7)),
    )
    _candidate_group(operation_settings)
    worker.process_summary_pending()

    assert _outcome(worker, "structure_judge") == SlotOutcome.OFF_CONTRACT.value
    assert _paused(worker, "structure_judge") is True
    assert _paused(worker, "judge") is False


# --- criterion 5: the decision rule blocks the judge operation only ----------


def test_unavailable_decision_blocks_the_judge_operation(
    blocked_decision_settings,
) -> None:
    """The decision slot is down and the fallback is off: `judge` waits."""
    worker = _worker(blocked_decision_settings, decision=UndecidingDecision())
    assert worker._operation_blocked("judge") is True
    assert worker._operation_blocked("classify") is True
    for operation in ("summarize", "title", "merge", "describe", "embed"):
        assert worker._operation_blocked(operation) is False, operation


def test_the_structural_judge_does_not_wait_for_the_decision(
    blocked_decision_settings,
) -> None:
    """The structural judge works on the generative path — the decision does
    not block it (the decision rule lists `judge`/`classify` only)."""
    worker = _worker(blocked_decision_settings, decision=UndecidingDecision())
    assert worker._operation_blocked("structure_judge") is False
    assert worker._decision_blocks_key("structure_judge") is False


def test_the_judge_operation_gate_follows_the_decision_rule(
    blocked_decision_settings,
) -> None:
    """The gate of the judge job is closed while the decision is unavailable."""
    worker = _worker(blocked_decision_settings, decision=UndecidingDecision())
    spec = build_judge_job(worker, blocked_decision_settings)
    assert spec.is_gated is not None and spec.is_gated() is True


# --- criterion 6: the legacy mode did not change ----------------------------


def _legacy_contour(settings) -> dict:
    """The `legacy` contour: the snapshot of the behaviour (the base commit)."""
    structure_judge = ScriptedStructureJudge(fail=True)
    worker = _worker(
        settings,
        promoter=_promoter(settings, structure_judge),
        classifier=FixedClassifier(Classification("work/subo", 0.7)),
    )
    snapshot = {
        "breakers": sorted(worker._breakers),
        "slot_blocked": {
            name: worker._slot_blocked(name)
            for name in ("embedding", "summary", "judge")
        },
    }
    # The gate of the judge job with an open breaker.
    worker._breakers["judge"].note_failure(SlotOutcome.AUTH)
    spec = build_judge_job(worker, settings)
    snapshot["judge_gate_open_breaker"] = spec.gate() if spec.gate else None
    snapshot["judge_is_gated_open_breaker"] = (
        spec.is_gated() if spec.is_gated else None
    )
    # The think flag of the structural judge (the legacy source).
    snapshot["structure_judge_think"] = StructureJudgeService(settings)._think()
    # The failure path of the dedup judge (the slot group is shared).
    worker._log_slot_failure(
        "judge", worker._operation_slot("judge"), SlotOutcome.OFF_CONTRACT,
        event="legacy_probe",
    )
    snapshot["paused_after_judge_failure"] = sorted(
        name
        for name, breaker in worker._breakers.items()
        if breaker.is_paused()
    )
    # The failure path of the structural judge (the `legacy` mode swallows it).
    worker._breakers["judge"].note_success()
    _candidate_group(settings)
    worker.process_summary_pending()
    snapshot["paused_after_structure_judge_failure"] = sorted(
        name
        for name, breaker in worker._breakers.items()
        if breaker.is_paused()
    )
    return snapshot


def test_legacy_judge_contour_snapshot_is_unchanged(legacy_settings) -> None:
    """The `legacy` contour is exactly the snapshot of the base commit."""
    assert _legacy_contour(legacy_settings) == LEGACY_JUDGE_CONTOUR


def test_legacy_think_is_the_namespace_source(legacy_settings) -> None:
    """The `legacy` mode: the structural judge thinks by NAMESPACE_JUDGE_THINK."""
    assert StructureJudgeService(legacy_settings)._think() is True


def test_operation_think_ignores_the_legacy_source(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The `operation` mode: the legacy source does not take part.

    The design, B1: the think flag of the operation `structure_judge` is
    `STRUCTURE_JUDGE_THINK`; `NAMESPACE_JUDGE_THINK` is a legacy source.
    """
    monkeypatch.setenv("SLOT_MODEL", "operation")
    monkeypatch.setenv("NAMESPACE_JUDGE_THINK", "true")
    monkeypatch.delenv("STRUCTURE_JUDGE_THINK", raising=False)
    get_settings.cache_clear()
    settings = get_settings()
    assert StructureJudgeService(settings)._think() is None


def test_operation_think_is_the_operation_source(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The `operation` mode: the think flag is the operation's own env."""
    monkeypatch.setenv("SLOT_MODEL", "operation")
    monkeypatch.setenv("NAMESPACE_JUDGE_THINK", "true")
    monkeypatch.setenv("STRUCTURE_JUDGE_THINK", "false")
    get_settings.cache_clear()
    settings = get_settings()
    assert StructureJudgeService(settings)._think() is False
