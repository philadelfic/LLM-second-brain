"""`/health.slots` and `/health.queues` by OPERATIONS (release 3.9.0, LSBN-S-15).

In the `operation` mode the key of `/health.slots` is the OPERATION (the ten of
the registry), the four legacy slot keys STAY as the DEPRECATED aggregates (the
worst operation of the group); `/health.queues` keeps the old job keys and adds
the keys of the operations; the fields `slot_mode` and `slot_migration_pending`
are visible in BOTH modes; the `legacy` mode does not change in anything.

The criteria of the step (the design, B6):
  1. the key of `/health.slots` is the operation — an entry for every operation
     of the registry (ten);
  2. the old keys stay and are marked `deprecated` — each of them is the worst
     operation of its group; in the `legacy` mode — the old form, no mark;
  3. `/health.queues` — the old job keys stay, the operation keys are added, the
     counters of a key are the counters of its operation;
  4. `slot_mode` and `slot_migration_pending`;
  5. the consumers (the healthcheck and the MCP slot hint) work in both modes;
  6. no secrets in the answer;
  7. the `legacy` mode did not change (the snapshot of the base commit
     `8ee4eb4` — the script `/tmp/lsb-s15-legacy-snapshot.py`).

A unit without a network: the models are unreachable (the test environment),
the health assembly calls no models anyway.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from fakes import (
    FixedClassifier,
    FixedSummarizer,
    HashEmbedder,
    ScriptedJudge,
    clear_seeded_skills,
)
from mcp.types import CallToolResult, TextContent

from app import __version__
from app.config import get_settings
from app.main import create_app
from app.services.notes import NoteService
from app.services.operations import OPERATION_REGISTRY
from app.services.slot_outcome import SlotOutcome
from app.services.worker import (
    BackgroundWorker,
    ensure_worker_jobs_schema,
    migrate_slots,
    slot_group_operations,
)
from app.storage.db import init_db, session
from app.transport.mcp import SlotHintMiddleware

#: The dimension of the test embedder (the fakes).
DIM = 8

#: The legacy slot keys of `/health.slots` (the deprecated aggregates).
LEGACY_SLOT_KEYS = ("embedding", "summary", "judge", "decision")

#: The legacy job keys of `/health.queues` (the consumers of the old form).
LEGACY_QUEUE_KEYS = (
    "vector",
    "summary",
    "judge",
    "areas",
    "nodes",
    "migration",
    "warnings",
)

#: The queue keys of the operations (the `operation` mode).
OPERATION_QUEUE_KEYS = ("title", "summarize", "merge", "judge", "embed", "classify")

#: The fields of a slot's entry — the form of `SlotStat` (the canon 6.4).
SLOT_FIELDS = {
    "ok",
    "last_outcome",
    "paused_until",
    "attention",
    "last_finish_reason",
    "last_budget",
    "last_budget_source",
    "last_had_thinking",
    "last_answer_budget",
    "last_reasoning_budget",
    "last_sent_budget",
}

#: The `legacy` snapshot of `/health` — taken on the BASE commit `8ee4eb4`
#: BEFORE this step. The script: `/tmp/lsb-s15-legacy-snapshot.py` (the same
#: script on the base tree and on the branch prints the same JSON — the diff
#: is empty; the report of the step carries the proof).
LEGACY_HEALTH_SNAPSHOT: dict = json.loads(
    """
    {
      "areas": {
        "logs": {
          "enabled": true,
          "pending": 0
        },
        "skills": {
          "enabled": true,
          "pending": 0
        },
        "terms": {
          "enabled": true,
          "pending": 0
        },
        "user": {
          "enabled": true,
          "pending": 0
        }
      },
      "backups": {
        "bad": 0,
        "count": 1,
        "last_ship": null,
        "last_verify": {
          "at": "<ts>",
          "file": null,
          "result": "ok"
        },
        "newest_age_sec": null,
        "oldest_age_sec": null,
        "total_bytes": null
      },
      "decision": {
        "answers": 0,
        "by_reason": {
          "low_confidence": 0,
          "unavailable": 0,
          "unsupported": 0
        },
        "enabled": false,
        "fallbacks": 0,
        "ok": null
      },
      "decision_ok": null,
      "dedup_enabled": true,
      "embedding_ok": null,
      "judge_ok": null,
      "migration": {
        "phase": null,
        "slot": {
          "batch": 5,
          "pause_sec": 600
        }
      },
      "note_chars_limit": 2000,
      "notes_count": 0,
      "operations": {
        "classify": {
          "answer_budget": 768,
          "answer_source": "legacy",
          "model": "test-summary-model",
          "provider": "ollama",
          "reasoning_budget": 0,
          "reasoning_source": "legacy",
          "sent_budget": 768,
          "source": "slot"
        },
        "describe": {
          "answer_budget": 35000,
          "answer_source": "legacy",
          "model": "test-summary-model",
          "provider": "ollama",
          "reasoning_budget": 0,
          "reasoning_source": "legacy",
          "sent_budget": 35000,
          "source": "slot"
        },
        "embed": {
          "answer_budget": 35000,
          "answer_source": "legacy",
          "model": "qwen3-embedding:8b",
          "provider": "ollama",
          "reasoning_budget": 0,
          "reasoning_source": "legacy",
          "sent_budget": 35000,
          "source": "default"
        },
        "judge": {
          "answer_budget": 256,
          "answer_source": "legacy",
          "model": "test-judge-model",
          "provider": "ollama",
          "reasoning_budget": 0,
          "reasoning_source": "legacy",
          "sent_budget": 256,
          "source": "slot"
        },
        "merge": {
          "answer_budget": 35000,
          "answer_source": "legacy",
          "model": "test-summary-model",
          "provider": "ollama",
          "reasoning_budget": 0,
          "reasoning_source": "legacy",
          "sent_budget": 35000,
          "source": "slot"
        },
        "split": {
          "answer_budget": 35000,
          "answer_source": "legacy",
          "model": "test-summary-model",
          "provider": "ollama",
          "reasoning_budget": 0,
          "reasoning_source": "legacy",
          "sent_budget": 35000,
          "source": "slot"
        },
        "structure_judge": {
          "answer_budget": 256,
          "answer_source": "legacy",
          "model": "test-judge-model",
          "provider": "ollama",
          "reasoning_budget": 0,
          "reasoning_source": "legacy",
          "sent_budget": 256,
          "source": "slot"
        },
        "summarize": {
          "answer_budget": 35000,
          "answer_source": "legacy",
          "model": "test-summary-model",
          "provider": "ollama",
          "reasoning_budget": 0,
          "reasoning_source": "legacy",
          "sent_budget": 35000,
          "source": "slot"
        },
        "title": {
          "answer_budget": 35000,
          "answer_source": "legacy",
          "model": "test-summary-model",
          "provider": "ollama",
          "reasoning_budget": 0,
          "reasoning_source": "legacy",
          "sent_budget": 35000,
          "source": "slot"
        }
      },
      "pending_summary": 0,
      "pending_vector": 0,
      "queues": {
        "areas": {
          "attention": 0,
          "oldest_pending_sec": null,
          "pending": 0,
          "waiting": 0
        },
        "judge": {
          "attention": 0,
          "oldest_pending_sec": null,
          "pending": 0,
          "waiting": 0
        },
        "links": {
          "attention": 0,
          "oldest_pending_sec": null,
          "pending": 0,
          "waiting": 0
        },
        "migration": {
          "attention": 0,
          "oldest_pending_sec": null,
          "pending": 0,
          "waiting": 0
        },
        "nodes": {
          "attention": 0,
          "oldest_pending_sec": null,
          "pending": 0,
          "waiting": 0
        },
        "summary": {
          "attention": 0,
          "oldest_pending_sec": null,
          "pending": 0,
          "waiting": 0
        },
        "terms-normalize": {
          "attention": 0,
          "oldest_pending_sec": null,
          "pending": 0,
          "waiting": 0
        },
        "vector": {
          "attention": 0,
          "oldest_pending_sec": null,
          "pending": 0,
          "waiting": 0
        },
        "warnings": {
          "attention": 0,
          "oldest_pending_sec": null,
          "pending": 0,
          "waiting": 0
        }
      },
      "slots": {
        "decision": {
          "attention": 0,
          "last_answer_budget": null,
          "last_budget": null,
          "last_budget_source": null,
          "last_finish_reason": null,
          "last_had_thinking": null,
          "last_outcome": null,
          "last_reasoning_budget": null,
          "last_sent_budget": null,
          "ok": null,
          "paused_until": null
        },
        "embedding": {
          "attention": 0,
          "last_answer_budget": null,
          "last_budget": null,
          "last_budget_source": null,
          "last_finish_reason": null,
          "last_had_thinking": null,
          "last_outcome": null,
          "last_reasoning_budget": null,
          "last_sent_budget": null,
          "ok": null,
          "paused_until": null
        },
        "judge": {
          "attention": 0,
          "last_answer_budget": null,
          "last_budget": null,
          "last_budget_source": null,
          "last_finish_reason": null,
          "last_had_thinking": null,
          "last_outcome": null,
          "last_reasoning_budget": null,
          "last_sent_budget": null,
          "ok": null,
          "paused_until": null
        },
        "summary": {
          "attention": 0,
          "last_answer_budget": null,
          "last_budget": null,
          "last_budget_source": null,
          "last_finish_reason": null,
          "last_had_thinking": null,
          "last_outcome": null,
          "last_reasoning_budget": null,
          "last_sent_budget": null,
          "ok": null,
          "paused_until": null
        }
      },
      "status": "ok",
      "summarizer_ok": null,
      "update": {
        "checked_at": null,
        "current": "<version>",
        "enabled": false,
        "hint_shown_at": null,
        "latest": null
      },
      "version": "<version>",
      "warnings": {
        "by_kind": {},
        "total": 0
      }
    }
"""
)

#: The values that depend on the clock — normalized (the snapshot's rule).
VOLATILE = {
    "oldest_pending_sec",
    "newest_age_sec",
    "oldest_age_sec",
    "checked_at",
    "hint_shown_at",
    "paused_until",
    "file",
    # The size of the database file — the page allocation, not a form fact.
    "total_bytes",
}

#: The fields of this step — the sanctioned difference from the base.
SANCTIONED = {"slot_mode", "slot_migration_pending"}

#: The release-controlled values (LSBN-S-19) — the version string of the
#: release. They are NOT pinned by the snapshot: a version bump must not break
#: the proof that the legacy `/health` FORM did not change. The snapshot keeps
#: the place (the placeholder), the value is checked against the live
#: `app.__version__` in the same test.
RELEASE_CONTROLLED = {"version", "current"}

#: The placeholder of the release-controlled values in the snapshot.
VERSION_PLACEHOLDER = "<version>"


def _is_timestamp(value: str) -> bool:
    """An ISO-8601 stamp."""
    return len(value) >= 20 and value[4] == "-" and value[10] == "T"


def normalize(value, key: str | None = None):
    """The normalized form of a JSON value (the snapshot's rule)."""
    if isinstance(value, dict):
        return {
            name: (
                VERSION_PLACEHOLDER
                if name in RELEASE_CONTROLLED
                else normalize(item, name)
            )
            for name, item in value.items()
            if name not in SANCTIONED
        }
    if isinstance(value, list):
        return [normalize(item, key) for item in value]
    if key in VOLATILE:
        return None
    if isinstance(value, str) and _is_timestamp(value):
        return "<ts>"
    return value


def _worst(members: list[str], entries: dict[str, dict]) -> str:
    """The worst operation of the group — the rule of the deprecated aggregate.

    The severity: a paused breaker first (the work is blocked right now), then
    the failure counter; a tie — the first of the list (the registry order).
    """

    def severity(name: str) -> tuple[int, int]:
        entry = entries[name]
        return (
            1 if entry.get("state") == "paused" else 0,
            int(entry.get("failures") or 0),
        )

    return max(members, key=severity)


def _app(monkeypatch, tmp_path, name: str, mode: str):
    """The environment of the snapshot: the app in the given mode.

    The areas job is off — its vectorization would make a model attempt in the
    background (a racy `embedding_ok` in the snapshot). The seeded skill is
    cleared (the state of the test fixture of the project).
    """
    monkeypatch.setenv("DB_PATH", str(tmp_path / f"{name}.db"))
    monkeypatch.setenv("BACKUP_DIR", str(tmp_path / f"{name}-backups"))
    monkeypatch.setenv("JOB_AREAS_ENABLED", "false")
    if mode == "operation":
        monkeypatch.setenv("SLOT_MODEL", "operation")
    else:
        monkeypatch.delenv("SLOT_MODEL", raising=False)
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    clear_seeded_skills(settings)
    ensure_worker_jobs_schema(settings)
    return settings


def _settings(tmp_path, monkeypatch, *, mode: str = "operation"):
    """The settings of the worker-level test: own database, the mode."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "worker.db"))
    monkeypatch.setenv("EMBEDDING_DIM", str(DIM))
    if mode == "operation":
        monkeypatch.setenv("SLOT_MODEL", "operation")
    else:
        monkeypatch.delenv("SLOT_MODEL", raising=False)
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    ensure_worker_jobs_schema(settings)
    return settings


def _worker(settings):
    """The worker with the fakes behind the operations (no background loops)."""
    return BackgroundWorker(
        settings,
        HashEmbedder(DIM),
        FixedSummarizer(),
        judge=ScriptedJudge(),
        classifier=FixedClassifier(),
    )


@pytest.fixture
def legacy_client(test_env, tmp_path, monkeypatch):
    """The app in the `legacy` mode (the default — the flag is off)."""
    _app(monkeypatch, tmp_path, "legacy", "legacy")
    with TestClient(create_app()) as client:
        client.app.state.services.notes.set_summary_notifier(None)
        client.app.state.services.notes.set_vector_notifier(None)
        yield client


@pytest.fixture
def operation_client(test_env, tmp_path, monkeypatch):
    """The app in the `operation` mode (the flag of the release)."""
    _app(monkeypatch, tmp_path, "operation", "operation")
    with TestClient(create_app()) as client:
        client.app.state.services.notes.set_summary_notifier(None)
        client.app.state.services.notes.set_vector_notifier(None)
        yield client


# --- criterion 1: the keys are the operations --------------------------------


def test_every_operation_of_the_registry_has_an_entry(operation_client) -> None:
    """In the `operation` mode there is an entry for every operation (ten)."""
    slots = operation_client.get("/health").json()["slots"]
    for operation in OPERATION_REGISTRY:
        assert operation in slots, operation
    assert len(OPERATION_REGISTRY) == 10


# --- criterion 2: the legacy keys stay as the deprecated aggregates -----------


def test_the_legacy_keys_stay_and_are_marked_deprecated(operation_client) -> None:
    """The four legacy slot keys are there, marked, and are their group's worst."""
    worker = operation_client.app.state.worker
    entries = worker.slots_health()  # the internal form (with the breaker's state)
    slots = operation_client.get("/health").json()["slots"]
    groups = slot_group_operations()
    assert set(groups) == set(LEGACY_SLOT_KEYS)
    for group, members in groups.items():
        stat = slots[group]
        assert stat["deprecated"] is True
        assert stat["operations"] == list(members)
        worst = _worst([name for name in members if name in entries], entries)
        for field in SLOT_FIELDS:
            assert stat[field] == entries[worst][field], (group, field)


def test_the_aggregate_is_the_worst_operation_of_its_group(operation_client) -> None:
    """A failure of the `merge` operation: the `summary` aggregate is IT."""
    worker = operation_client.app.state.worker
    worker._breakers["merge"].note_failure(SlotOutcome.AUTH)
    slots = operation_client.get("/health").json()["slots"]
    assert slots["merge"]["last_outcome"] == "auth"
    # The aggregate of the summary group — the worst member of the group.
    assert slots["summary"]["last_outcome"] == "auth"
    # The neighbours of the group are untouched.
    assert slots["summarize"]["last_outcome"] is None
    assert slots["title"]["last_outcome"] is None
    assert slots["judge"]["last_outcome"] is None


def test_the_legacy_form_has_no_marker(legacy_client: TestClient) -> None:
    """The `legacy` mode — the old form: no mark, no group, no nested calls."""
    body = legacy_client.get("/health").json()
    assert set(body["slots"]) == set(LEGACY_SLOT_KEYS)
    for stat in body["slots"].values():
        assert set(stat) == SLOT_FIELDS


def test_the_nested_title_calls_of_the_split_are_marked(operation_client) -> None:
    """The title calls of the parts of a split are counted toward `split`.

    The accumulated point of the previous steps: the title of every part is a
    separate call of the `title` operation, but it goes INSIDE the call of
    `split` — it belongs to the `split` operation (one failure is counted
    once, the `title` operation's breaker does not see it).
    """
    slots = operation_client.get("/health").json()["slots"]
    assert slots["split"]["nested_calls"] == ["title"]
    # The `title` operation itself carries no nested calls of its own.
    assert "nested_calls" not in slots["title"]


# --- criterion 3: the queues ------------------------------------------------


def test_the_queues_keep_the_old_keys_and_add_the_operation_keys(
    operation_client,
) -> None:
    """The old job keys stay; the operation keys are added."""
    queues = operation_client.get("/health").json()["queues"]
    for key in LEGACY_QUEUE_KEYS:
        assert key in queues, key
    for key in OPERATION_QUEUE_KEYS:
        assert key in queues, key


def test_the_counters_of_the_operation_keys_are_their_operations(
    test_env, tmp_path, monkeypatch
) -> None:
    """The counters of a key are the counters of its operation.

    The worker without the background loops (the loops would take the tasks
    and the counters would race): the work of every operation is counted.
    """
    settings = _settings(tmp_path, monkeypatch, mode="operation")
    worker = _worker(settings)
    notes = NoteService(settings)
    # The work: a note without a title, a note with a title, a merge job, a
    # dedup job and a reclass task.
    long_text = "Заметка для проверки счётчиков очередей. " * 20
    without_title = notes.save(long_text)
    notes.save(long_text, title="Очередь")
    worker._create_job("summary", "merge", without_title["id"])
    worker._create_job("judge", "dedup", without_title["id"])
    worker._create_job("nodes", "reclass", without_title["id"])

    queues = worker.queues_health()
    with session(settings) as conn:
        title = conn.execute(
            "SELECT COUNT(*) AS c FROM notes WHERE deleted_at IS NULL "
            "AND title IS NULL"
        ).fetchone()["c"]
        summarize = conn.execute(
            "SELECT COUNT(*) AS c FROM notes WHERE deleted_at IS NULL "
            "AND summary_status = 'pending'"
        ).fetchone()["c"]
        merge = conn.execute(
            "SELECT COUNT(*) AS c FROM worker_jobs WHERE kind = 'merge' "
            "AND status = 'pending'"
        ).fetchone()["c"]
        dedup = conn.execute(
            "SELECT COUNT(*) AS c FROM worker_jobs WHERE kind = 'dedup' "
            "AND status = 'pending'"
        ).fetchone()["c"]
        reclass = conn.execute(
            "SELECT COUNT(*) AS c FROM worker_jobs WHERE kind = 'reclass' "
            "AND status = 'pending'"
        ).fetchone()["c"]
    assert queues["title"]["pending"] == title == 1
    assert queues["summarize"]["pending"] == summarize == 1
    assert queues["merge"]["pending"] == merge == 1
    assert queues["judge"]["pending"] == dedup == 1
    assert queues["classify"]["pending"] == reclass == 1
    # The `embed` operation — the notes and the areas.
    assert queues["embed"]["pending"] == (
        queues["vector"]["pending"] + queues["areas"]["pending"]
    )
    # The legacy keys stay — the `summary` key is the historical aggregate:
    # the notes awaiting a title OR a summary (each note once) plus the merge
    # jobs.
    with session(settings) as conn:
        legacy_notes = conn.execute(
            "SELECT COUNT(*) AS c FROM notes WHERE deleted_at IS NULL "
            "AND (title IS NULL OR summary_status = 'pending')"
        ).fetchone()["c"]
    assert queues["summary"]["pending"] == legacy_notes + merge


# --- criterion 4: the mode and the migration counter ------------------------


def test_slot_mode_is_the_actual_mode(legacy_client: TestClient, operation_client) -> None:
    """`slot_mode` — the flag of the slot layer, as it is."""
    assert legacy_client.get("/health").json()["slot_mode"] == "legacy"
    assert operation_client.get("/health").json()["slot_mode"] == "operation"


def test_slot_migration_pending_counts_the_rows_with_the_old_values(
    operation_client,
) -> None:
    """The counter — the rows the migration of LSBN-S-14 would still move."""
    settings = get_settings()
    worker = operation_client.app.state.worker
    # A row with the OLD value of the slot ('summary' — the merge task).
    with session(settings) as conn:
        conn.execute(
            "INSERT INTO worker_jobs (slot, kind, note_id, status) "
            "VALUES ('summary', 'merge', 1, 'pending')"
        )
    assert operation_client.get("/health").json()["slot_migration_pending"] == 1
    # After the migration the row carries the operation name — the counter is 0.
    migrate_slots(settings)
    assert operation_client.get("/health").json()["slot_migration_pending"] == 0
    assert worker.slot_migration_pending() == 0


def test_the_counter_is_visible_in_the_legacy_mode_too(legacy_client: TestClient) -> None:
    """The field is visible in BOTH modes — the consumer sees what happens."""
    body = legacy_client.get("/health").json()
    assert body["slot_mode"] == "legacy"
    assert body["slot_migration_pending"] == 0


# --- criterion 5: the consumers are not broken --------------------------------


def test_the_healthcheck_works_in_both_modes(
    legacy_client: TestClient, operation_client
) -> None:
    """The healthcheck (the docker one) — 200 and the old keys in both modes."""
    for app_client in (legacy_client, operation_client):
        response = app_client.get("/health")
        assert response.status_code == 200
        slots = response.json()["slots"]
        for key in LEGACY_SLOT_KEYS:
            assert key in slots, key


class _Ctx:
    """A minimal MCP request context: the middleware reads only the method."""

    method = "tools/call"


def _result() -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text="{}")],
        structured_content={"items": []},
        is_error=False,
    )


async def _call(mw: SlotHintMiddleware) -> CallToolResult:
    async def call_next(ctx) -> CallToolResult:
        return _result()

    return await mw(_Ctx(), call_next)


@pytest.mark.asyncio
async def test_the_mcp_slot_hint_works_in_both_modes(
    legacy_client: TestClient, operation_client
) -> None:
    """The MCP slot hint — it names the slot (the operation in the new mode)."""
    settings = get_settings()
    legacy_worker = legacy_client.app.state.worker
    legacy_worker._breakers["summary"].note_failure(SlotOutcome.AUTH)
    out = await _call(SlotHintMiddleware(settings, legacy_worker))
    assert "summary" in out.structured_content["hint"]

    operation_worker = operation_client.app.state.worker
    operation_worker._breakers["title"].note_failure(SlotOutcome.AUTH)
    out = await _call(SlotHintMiddleware(settings, operation_worker))
    assert "title" in out.structured_content["hint"]


# --- criterion 6: no secrets -------------------------------------------------


def test_no_secrets_in_the_health_form(
    operation_client, monkeypatch, tmp_path
) -> None:
    """No keys and no addresses in the answer — the old checks plus the new."""
    monkeypatch.setenv("MERGE_BASE_URL", "http://merge-private:11434")
    monkeypatch.setenv("MERGE_API_KEY", "sk-merge-test")
    monkeypatch.setenv("DB_PATH", str(tmp_path / "secrets.db"))
    get_settings.cache_clear()
    try:
        with TestClient(create_app()) as fresh:
            raw = fresh.get("/health").text
            body = fresh.get("/health").json()
    finally:
        get_settings.cache_clear()
    assert "merge-private" not in raw
    assert "sk-merge-test" not in raw
    assert "base_url" not in raw
    assert "api_key" not in raw
    # The new fields carry no addresses and no keys either.
    assert "base_url" not in json.dumps(body["slots"])
    assert "api_key" not in json.dumps(body["slots"])
    assert body["slot_mode"] == "operation"


# --- criterion 7: the legacy mode did not change -----------------------------


def test_the_legacy_health_form_matches_the_base_snapshot(
    legacy_client: TestClient,
) -> None:
    """The `legacy` form of `/health` — the snapshot of the base commit."""
    body = legacy_client.get("/health").json()
    assert normalize(body) == LEGACY_HEALTH_SNAPSHOT
    # The two fields of this step — the sanctioned difference from the base.
    assert body["slot_mode"] == "legacy"
    assert body["slot_migration_pending"] == 0
    # The version of the release — the LIVE value, not a literal (LSBN-S-19):
    # the snapshot proves the FORM of the legacy `/health`, the version is
    # checked separately, so a version bump does not break the snapshot.
    assert body["version"] == __version__
    assert body["update"]["current"] == __version__
