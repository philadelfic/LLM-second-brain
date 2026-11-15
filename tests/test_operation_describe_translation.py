"""The operation `describe` (release 3.9.0, LSBN-S-10).

    The description generator is the fourth consumer of the `summary` slot
    group: the promotion trigger asks it for the description of a candidate
    node. In the `operation` mode (`SLOT_MODEL=operation`) every operation owns
    its breaker: a failure of the description must not silence the
    neighbouring operations (`classify`, `summarize`, `judge`,
    `structure_judge`), and the `legacy` mode keeps the shared breaker of the
    slot group (as before).

    Checkable contracts (the criteria of the step):
    1. a failure of the description pauses the breaker of the `describe`
       operation only;
    2. the pause of the operation stops the DESCRIPTION (the candidate waits)
       and does not stop the `classify` path;
    3. a failure of the description does not lose the work (the candidate
       waits for the next run);
    4. the probe of the `describe` operation is a real one — it reaches the
       client of the description model (not the `CHECK_OK` stub);
    5. `/health.slots` shows the honest `ok` of the operation (the last
       attempt of the description);
    6. the `legacy` mode did not change (the snapshot below was taken on the
       base commit `8607647`, before this step).

    A unit without a network: the models are fakes, the probe is checked on the
    real `DescriptionService` over `httpx.MockTransport`.
"""

from __future__ import annotations

import httpx
import pytest
from fakes import (
    FixedClassifier,
    FixedDescriber,
    FixedSummarizer,
    HashEmbedder,
    clear_seeded_skills,
)

from app.config import get_settings
from app.services.classifier import Classification
from app.services.llm_client import CHECK_OK, CHECK_UNREACHABLE
from app.services.namespaces import NamespaceService
from app.services.notes import NoteService
from app.services.promotion import (
    DescriberError,
    DescriptionService,
    PromotionService,
    Verdict,
)
from app.services.slot_outcome import SlotOutcome
from app.services.worker import BackgroundWorker
from app.storage.db import init_db, session

DIM = 8
#: The promotion threshold (the group of default notes that becomes a node).
THRESHOLD = 15
#: The model of the `describe` operation (the operation's own env in the
#: `operation` mode, the slot's env in the `legacy` mode).
MODEL = "test-describe-model"

#: The `legacy` contour snapshot — taken on the base commit `8607647` BEFORE
#: this step (the `legacy` mode must not change in any respect): the breaker
#: keys, the blocked slots, the entries of `/health.slots`, the failure path
#: of the description (the breakers it pauses, the journal event, the state
#: of the candidate).
LEGACY_DESCRIBE_CONTOUR: dict = {
    "breakers": ["embedding", "judge", "summary"],
    "slot_blocked": {"embedding": False, "judge": False, "summary": False},
    "health_slots": ["decision", "embedding", "judge", "summary"],
    "describer_calls": 1,
    "paused_after_describe_failure": [],
    "promotion_failed_events": 1,
    "candidate_waits": True,
    "node_created": False,
}


class OutcomeDescriber:
    """A describer that fails with a given outcome (the transport's truth).

    The `DescriberError` of the real service carries the outcome of the
    client error (`classify_llm_error`); the fake carries it too — the
    breaker of the operation must see the same outcome.
    """

    def __init__(self, outcome: SlotOutcome) -> None:
        self.outcome = outcome
        self.calls: list[tuple[list[str], str, str]] = []

    def describe(self, summaries: list[str], slug: str, domain: str) -> str:
        self.calls.append((list(summaries), slug, domain))
        raise DescriberError("генератор описаний недоступен", outcome=self.outcome)

    def close(self) -> None:  # interface compatibility with DescriptionService
        return None


def _settings(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    mode: str,
):
    """The test settings: own database, the slot model, a one-failure pause."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "notes.db"))
    monkeypatch.setenv("EMBEDDING_DIM", str(DIM))
    monkeypatch.setenv("SUMMARY_SHORT_TEXT_CHARS", "0")
    # The pause opens at the first failure — otherwise the third failure
    # would open it.
    monkeypatch.setenv("SLOT_PAUSE_AFTER", "1")
    # The models: the slot's env in the `legacy` mode, the operation's own env
    # in the `operation` mode (the slot layer does not take part there — the
    # design, B1).
    monkeypatch.setenv("SUMMARY_MODEL", MODEL)
    monkeypatch.setenv("SUMMARY_BASE_URL", "http://127.0.0.1:1")
    monkeypatch.setenv("CLASSIFY_MODEL", MODEL)
    monkeypatch.setenv("CLASSIFY_BASE_URL", "http://127.0.0.1:1")
    monkeypatch.setenv("DESCRIBE_MODEL", MODEL)
    monkeypatch.setenv("DESCRIBE_BASE_URL", "http://127.0.0.1:1")
    # The decision slot has its own env (the reserve slot — the F-12 mechanics).
    monkeypatch.setenv("DECISION_BASE_URL", "http://127.0.0.1:1")
    monkeypatch.setenv("DECISION_MODEL", "decision-test")
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


def _worker(
    settings,
    *,
    promoter=None,
    classifier=None,
) -> BackgroundWorker:
    """A worker with the trigger of the promotion."""
    return BackgroundWorker(
        settings,
        HashEmbedder(DIM),
        FixedSummarizer(),
        classifier=classifier,
        promoter=promoter,
    )


def _promoter(settings, describer) -> PromotionService:
    """The promotion trigger with a scriptable description generator."""
    from fakes import ScriptedStructureJudge

    return PromotionService(
        settings,
        embedding=HashEmbedder(DIM),
        describer=describer,
        judge=ScriptedStructureJudge(default=Verdict("create")),
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


def _paused_breakers(worker: BackgroundWorker) -> list[str]:
    """The names of the paused breakers (sorted)."""
    return sorted(
        name for name, breaker in worker._breakers.items() if breaker.is_paused()
    )


def _candidate_group(settings, count: int = THRESHOLD) -> None:
    """A group of default notes that reaches the promotion threshold."""
    NamespaceService(settings).create("work", "Рабочие заметки.")
    notes = NoteService(settings)
    for index in range(count):
        notes.save(f"заметка про СУБО №{index}")


def _events(caplog, name: str) -> list:
    """The journal records of an event."""
    return [r for r in caplog.records if getattr(r, "event", None) == name]


def _node_count(settings, path: str = "work/subo") -> int:
    """The number of the nodes with the path."""
    with session(settings) as conn:
        return int(
            conn.execute(
                "SELECT COUNT(*) FROM namespaces WHERE path = ?", (path,)
            ).fetchone()[0]
        )


def _notes_in_default(settings) -> int:
    """The number of the notes in `default`."""
    with session(settings) as conn:
        return int(
            conn.execute(
                "SELECT COUNT(*) FROM notes WHERE namespace = 'default'"
            ).fetchone()[0]
        )


# --- criterion 1: a failure of the description → its own breaker --------------


def test_describe_failure_pauses_the_describe_breaker_only(operation_settings) -> None:
    """A failure of the description: the pause is only at `describe`."""
    worker = _worker(
        operation_settings,
        promoter=_promoter(operation_settings, FixedDescriber(fail=True)),
        classifier=FixedClassifier(Classification("work/subo", 0.7)),
    )
    _candidate_group(operation_settings)

    worker.process_summary_pending()

    assert _paused(worker, "describe") is True
    for operation in (
        "classify",
        "summarize",
        "title",
        "judge",
        "structure_judge",
        "embed",
    ):
        assert _paused(worker, operation) is False, operation


def test_the_describe_failure_is_visible_in_its_breaker(operation_settings) -> None:
    """The failure is recorded at the breaker of the operation."""
    worker = _worker(
        operation_settings,
        promoter=_promoter(operation_settings, FixedDescriber(fail=True)),
        classifier=FixedClassifier(Classification("work/subo", 0.7)),
    )
    _candidate_group(operation_settings)
    worker.process_summary_pending()

    assert _outcome(worker, "describe") == SlotOutcome.OFF_CONTRACT.value
    assert _outcome(worker, "summarize") is None
    assert _outcome(worker, "classify") is None


def test_the_describe_failure_keeps_its_own_outcome(operation_settings) -> None:
    """The error carries the outcome of the failure (the mapping of the client)."""
    worker = _worker(
        operation_settings,
        promoter=_promoter(operation_settings, OutcomeDescriber(SlotOutcome.AUTH)),
        classifier=FixedClassifier(Classification("work/subo", 0.7)),
    )
    _candidate_group(operation_settings)
    worker.process_summary_pending()

    assert _outcome(worker, "describe") == SlotOutcome.AUTH.value


def test_the_describe_failure_is_journalled(operation_settings, caplog) -> None:
    """The journal: the event of the failure of the operation."""
    worker = _worker(
        operation_settings,
        promoter=_promoter(operation_settings, FixedDescriber(fail=True)),
        classifier=FixedClassifier(Classification("work/subo", 0.7)),
    )
    _candidate_group(operation_settings)
    with caplog.at_level("WARNING", logger="app"):
        worker.process_summary_pending()

    records = _events(caplog, "promotion_describe_failed")
    assert len(records) == 1
    assert records[0].slot == "describe"  # the breaker key of the operation
    assert records[0].outcome == SlotOutcome.OFF_CONTRACT.value


# --- criterion 2: the pause of the operation stops the description -----------


def test_the_paused_describe_stops_the_description_call(operation_settings) -> None:
    """The pause of `describe`: the description model is not called."""
    describer = FixedDescriber("Заметки о СУБО 2020.")
    worker = _worker(
        operation_settings,
        promoter=_promoter(operation_settings, describer),
        classifier=FixedClassifier(Classification("work/subo", 0.7)),
    )
    worker._breakers["describe"].note_failure(SlotOutcome.AUTH)
    assert _paused(worker, "describe") is True

    _candidate_group(operation_settings)
    worker.process_summary_pending()

    assert describer.calls == [], "the description was generated on the pause"


def test_the_paused_describe_does_not_stop_the_classify_path(
    operation_settings,
) -> None:
    """The pause of `describe`: the grooming classifies the note as before."""
    classifier = FixedClassifier(Classification("work/subo", 0.7))
    worker = _worker(
        operation_settings,
        promoter=_promoter(operation_settings, FixedDescriber()),
        classifier=classifier,
    )
    worker._breakers["describe"].note_failure(SlotOutcome.AUTH)

    _candidate_group(operation_settings)
    worker.process_summary_pending()

    assert classifier.calls, "the classifier was not called on the pause"
    assert _paused(worker, "classify") is False


def test_the_legacy_describe_is_not_gated_by_the_operation(legacy_settings) -> None:
    """The `legacy` mode: the pause of the slot group is not a path check.

    The gate of the job is the mechanism there (the whole job stops); the
    path itself is not gated by the operation (as before).
    """
    describer = FixedDescriber("Заметки о СУБО 2020.")
    worker = _worker(
        legacy_settings,
        promoter=_promoter(legacy_settings, describer),
        classifier=FixedClassifier(Classification("work/subo", 0.7)),
    )
    worker._breakers["summary"].note_failure(SlotOutcome.AUTH)

    _candidate_group(legacy_settings)
    worker.process_summary_pending()

    assert describer.calls, "the legacy path is not gated by the operation"


# --- criterion 3: the work is not lost ---------------------------------------


def test_the_describe_failure_keeps_the_candidate(operation_settings) -> None:
    """The candidate stays without a description and waits for the next run."""
    describer = FixedDescriber(fail=True)
    worker = _worker(
        operation_settings,
        promoter=_promoter(operation_settings, describer),
        classifier=FixedClassifier(Classification("work/subo", 0.7)),
    )
    _candidate_group(operation_settings)
    worker.process_summary_pending()

    assert _node_count(operation_settings) == 0  # no node was created
    assert _notes_in_default(operation_settings) == THRESHOLD  # the group waits
    assert worker._promoter.candidates() != []  # the same candidate waits


def test_the_candidate_is_taken_after_the_describe_recovers(
    operation_settings,
) -> None:
    """The model is back — the next run describes the same candidate."""
    describer = FixedDescriber(fail=True)
    worker = _worker(
        operation_settings,
        promoter=_promoter(operation_settings, describer),
        classifier=FixedClassifier(Classification("work/subo", 0.7)),
    )
    _candidate_group(operation_settings)
    worker.process_summary_pending()
    assert _node_count(operation_settings) == 0

    # The model is back: the breaker is resumed and the candidate is taken.
    describer.fail = False
    worker._breakers["describe"].note_success()

    worker.process_nodes()

    assert _node_count(operation_settings) == 1
    assert describer.calls, "the description was not generated"


# --- criterion 4: the probe of the operation is a real one -------------------


def test_the_promoter_has_a_probe(operation_settings) -> None:
    """The trigger answers the knock of the breaker (the protocol)."""
    assert hasattr(PromotionService, "probe")
    assert callable(PromotionService.probe)


def test_the_description_service_has_a_probe(operation_settings) -> None:
    """The description service answers the knock (the protocol)."""
    assert hasattr(DescriptionService, "probe")
    assert callable(DescriptionService.probe)


def test_the_probe_reaches_the_description_client(operation_settings) -> None:
    """The knock of the description goes to its client (not to a stub)."""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"models": [{"name": MODEL}]})

    service = DescriptionService(operation_settings, httpx.MockTransport(handler))

    assert service.probe() == CHECK_OK
    assert requests, "the client was not asked"


def test_the_operation_probe_reaches_the_description_client(
    operation_settings,
) -> None:
    """In the `operation` mode the knock of `describe` goes to the model."""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"models": [{"name": MODEL}]})

    worker = _worker(
        operation_settings,
        promoter=PromotionService(
            operation_settings,
            embedding=HashEmbedder(DIM),
            describer=DescriptionService(
                operation_settings, httpx.MockTransport(handler)
            ),
            namespaces=NamespaceService(operation_settings),
        ),
    )

    assert worker._probe_slot("describe") == CHECK_OK
    assert requests, "the description's client was not asked"


def test_a_failing_probe_is_not_a_silent_success(operation_settings) -> None:
    """A lying client: the probe does not answer `CHECK_OK`."""
    service = DescriptionService(
        operation_settings,
        httpx.MockTransport(lambda request: httpx.Response(500, json={})),
    )

    assert service.probe() != CHECK_OK
    assert service.probe() == CHECK_UNREACHABLE


# --- criterion 5: the health shows the honest `ok` ---------------------------


def test_the_health_shows_no_attempt_yet(operation_settings) -> None:
    """Before the first description the value is unknown (None)."""
    worker = _worker(
        operation_settings,
        promoter=_promoter(operation_settings, FixedDescriber()),
    )

    assert worker.slots_health()["describe"]["ok"] is None


def test_the_health_shows_the_successful_description(operation_settings) -> None:
    """After a successful description the value is True."""
    worker = _worker(
        operation_settings,
        promoter=_promoter(operation_settings, FixedDescriber()),
        classifier=FixedClassifier(Classification("work/subo", 0.7)),
    )
    _candidate_group(operation_settings)

    worker.process_summary_pending()

    assert worker.slots_health()["describe"]["ok"] is True


def test_the_health_shows_the_failed_description(operation_settings) -> None:
    """After a failed description the value is False (not None)."""
    worker = _worker(
        operation_settings,
        promoter=_promoter(operation_settings, FixedDescriber(fail=True)),
        classifier=FixedClassifier(Classification("work/subo", 0.7)),
    )
    _candidate_group(operation_settings)

    worker.process_summary_pending()

    assert worker.slots_health()["describe"]["ok"] is False


# --- criterion 6: the legacy mode did not change -----------------------------


def _legacy_contour(settings, caplog) -> dict:
    """The `legacy` contour: the snapshot of the behaviour (the base commit)."""
    describer = FixedDescriber(fail=True)
    worker = _worker(
        settings,
        promoter=_promoter(settings, describer),
        classifier=FixedClassifier(Classification("work/subo", 0.7)),
    )
    snapshot = {
        "breakers": sorted(worker._breakers),
        "slot_blocked": {
            name: worker._slot_blocked(name)
            for name in ("embedding", "summary", "judge")
        },
        "health_slots": sorted(worker.slots_health()),
    }
    _candidate_group(settings)
    with caplog.at_level("WARNING", logger="app"):
        worker.process_summary_pending()
    snapshot["describer_calls"] = len(describer.calls)
    snapshot["paused_after_describe_failure"] = _paused_breakers(worker)
    snapshot["promotion_failed_events"] = len(_events(caplog, "promotion_failed"))
    snapshot["candidate_waits"] = bool(worker._promoter.candidates())
    snapshot["node_created"] = _node_count(settings) > 0
    return snapshot


def test_legacy_describe_contour_snapshot_is_unchanged(
    legacy_settings, caplog
) -> None:
    """The `legacy` contour is exactly the snapshot of the base commit."""
    assert _legacy_contour(legacy_settings, caplog) == LEGACY_DESCRIBE_CONTOUR
