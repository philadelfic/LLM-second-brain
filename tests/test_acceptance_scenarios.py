"""Acceptance scenarios of the slot layer and the hooks under the operation names (LSBN-S-17).

Why this file. Two acceptance scenarios of the slot layer were red (LSBN-D-19):
"the age of the oldest task" and "the recovery after the break". Both read the
QUEUE of the slot — and the queue of an OPERATION is not the queue of its slot
group: in the `operation` mode the legacy aggregate `summary` is not gated by
the breaker of the operation, so the deferred work is invisible there. The
scenarios must therefore read the names of the MODE of the contour, and the
contour hooks (`scripts/slot_gate.sh`, `scripts/slot_fake.sh`,
`scripts/fake_slot.py`) must take the names of the OPERATIONS — the legacy slot
names stay acceptable (the `legacy` mode lives until the acceptance), with a
hint about the new name.

What is checked here (the acceptance criteria of the step):

  1. the age of the oldest task is visible and correct in BOTH modes;
  2. after a broken connection the worker returns to work and works off the
     accumulated backlog — in BOTH modes;
  3. the hooks take the operation names and work;
  4. the legacy slot names are still accepted — with a hint about the new name;
  5. the scenarios do not hardcode one mode (both branches are covered, and the
     `legacy` behaviour is the historical one);
  6. nothing is switched on the contour: the mode flag stays `legacy`, and the
     hooks change nothing without an explicit action.

The acceptance runner itself (`scripts/e2e_release32.py`) needs a live contour:
it is NOT run here. What is checked here is the naming layer it reads and the
hooks it calls — the parts that made the two scenarios red.
"""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone

import pytest
from fakes import FixedSummarizer, HashEmbedder

from app.config import get_settings
from app.services.notes import NoteService
from app.services.slot_outcome import SlotOutcome
from app.services.summary import SummaryError
from app.services.worker import BackgroundWorker
from app.storage.db import init_db, session, transaction

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"

# The two modes of the slot layer (release 3.9.0).
MODES = ("legacy", "operation")
# The queue of the summary work in each mode: the slot group's queue in the
# `legacy` mode, the operation's own queue in the `operation` mode.
SUMMARY_QUEUE = {"legacy": "summary", "operation": "summarize"}


# --- helpers ------------------------------------------------------------------


def _settings(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, mode: str):
    """The settings of a contour in the given mode of the slot layer."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "notes.db"))
    monkeypatch.setenv("EMBEDDING_DIM", "8")
    # LSBN-T-37: the short-note rule off — a note is summarized by the model
    # (the rule summarizes a short note by its own text, without a model call —
    # the probe notes of the acceptance must not leave the queue on their own).
    monkeypatch.setenv("SUMMARY_SHORT_TEXT_CHARS", "0")
    monkeypatch.setenv("SLOT_MODEL", mode)
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    return settings


def _backdate(settings, table: str, row_id: int, seconds_ago: int) -> None:
    """Age a row: the age of the oldest task is checked explicitly."""
    moment = datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)
    stamp = moment.strftime("%Y-%m-%dT%H:%M:%SZ")
    with session(settings) as conn, transaction(conn):
        conn.execute(
            f"UPDATE {table} SET updated_at = ?, created_at = ? WHERE id = ?",
            (stamp, stamp, row_id),
        )


class BrokenSummarizer:
    """The summarizer of a broken connection: every call fails as unavailable."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def summarize(self, text: str):
        self.calls.append(text)
        raise SummaryError("the connection is broken", outcome=SlotOutcome.UNAVAILABLE)

    def title(self, text: str):
        self.calls.append(text)
        raise SummaryError("the connection is broken", outcome=SlotOutcome.UNAVAILABLE)

    def merge(self, text_a: str, text_b: str, *, title: str | None = None):
        self.calls.append(text_a)
        raise SummaryError("the connection is broken", outcome=SlotOutcome.UNAVAILABLE)

    def close(self) -> None:
        return None


def _run(script: str, *args: str, env: dict[str, str] | None = None,
         timeout: float = 120.0) -> subprocess.CompletedProcess[str]:
    """Run one of the contour hooks and return its result."""
    return subprocess.run(
        ["bash", str(SCRIPTS / script), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, **(env or {})},
    )


def _stub_startup(*args: str, wait: float = 2.0) -> tuple[str, str]:
    """Start the fake provider briefly and return its (stdout, stderr)."""
    proc = subprocess.Popen(
        [sys.executable, str(SCRIPTS / "fake_slot.py"), *args, "--port", "0"],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    time.sleep(wait)
    proc.terminate()
    out, err = proc.communicate(timeout=30)
    return out, err


# --- 1. the age of the oldest task (both modes) -------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_oldest_task_age_is_visible_and_correct(tmp_path, monkeypatch, mode) -> None:
    """The queue of the summary work shows the age of the oldest task.

    The `legacy` mode reads the queue of the slot group (`summary`), the
    `operation` mode — the queue of the operation (`summarize`). The age is the
    age of the OLDEST task, and it grows while the task waits.
    """
    settings = _settings(tmp_path, monkeypatch, mode)
    notes = NoteService(settings, HashEmbedder(8))
    notes.save("первая заметка: возраст старейшей задачи",
               title="Первая заметка про возраст")
    notes.save("вторая заметка: возраст старейшей задачи",
               title="Вторая заметка про возраст")
    _backdate(settings, "notes", 1, 120)

    worker = BackgroundWorker(settings, HashEmbedder(8), FixedSummarizer("Фикс-выжимка."))
    queues = worker.queues_health()
    stat = queues[SUMMARY_QUEUE[mode]]

    assert stat["pending"] == 2
    assert stat["oldest_pending_sec"] >= 119
    # The age of the oldest task GROWS while the task waits (the old task is
    # still the oldest one).
    _backdate(settings, "notes", 1, 300)
    assert worker.queues_health()[SUMMARY_QUEUE[mode]]["oldest_pending_sec"] >= 299


@pytest.mark.parametrize("mode", MODES)
def test_the_legacy_queue_key_stays_readable_in_the_operation_mode(
    tmp_path, monkeypatch, mode
) -> None:
    """The legacy keys of `/health.queues` stay (the deprecated aggregates).

    The scenario reads the queue of the OPERATION — but the old consumers of
    the old form must not break (LSBN-S-15), so the legacy key is still there.
    """
    settings = _settings(tmp_path, monkeypatch, mode)
    notes = NoteService(settings, HashEmbedder(8))
    notes.save("заметка для очереди сводки", title="Заметка про очередь сводки")
    worker = BackgroundWorker(settings, HashEmbedder(8), FixedSummarizer("Фикс-выжимка."))
    queues = worker.queues_health()

    assert "summary" in queues
    assert queues["summary"]["pending"] == 1


# --- 2. the recovery after a break (both modes) --------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_worker_returns_to_work_after_a_broken_connection(
    tmp_path, monkeypatch, mode
) -> None:
    """After a break of the connection the worker comes back and works off
    the accumulated backlog.

    The broken connection: the batch is released, the tasks stay pending (the
    queue shows the work and its age). The connection is back: the worker
    takes the accumulated tasks and finishes them.
    """
    settings = _settings(tmp_path, monkeypatch, mode)
    notes = NoteService(settings, HashEmbedder(8))
    for index in range(3):
        notes.save(f"заметка {index}: восстановление после обрыва",
                   title=f"Заметка {index} про восстановление")

    broken = BrokenSummarizer()
    worker = BackgroundWorker(settings, HashEmbedder(8), broken)

    # The break: nothing is done, the tasks wait (the queue is not empty).
    assert worker.process_summary_pending() == 0
    assert len(broken.calls) == 1
    stat = worker.queues_health()[SUMMARY_QUEUE[mode]]
    assert stat["pending"] == 3
    assert stat["oldest_pending_sec"] is not None

    # The connection is back: the accumulated backlog is worked off.
    worker._summarizer = FixedSummarizer("Восстановленная выжимка.")
    assert worker.process_summary_pending() == 3
    after = worker.queues_health()[SUMMARY_QUEUE[mode]]
    assert after["pending"] == 0
    assert after["oldest_pending_sec"] in (0, None)


# --- 3/4. the hooks under the operation names ----------------------------------


def test_slot_gate_accepts_the_operation_names() -> None:
    """`scripts/slot_gate.sh` takes the operation names (and works with them)."""
    for operation, container in (
        ("summarize", "lsb-test-model-proxy-summary"),
        ("title", "lsb-test-model-proxy-summary"),
        ("merge", "lsb-test-model-proxy-summary"),
        ("classify", "lsb-test-model-proxy-summary"),
        ("judge", "lsb-test-model-proxy-judge"),
        ("embed", "lsb-test-model-proxy-embed"),
        ("decision", "lsb-test-model-proxy-dec"),
    ):
        proc = _run("slot_gate.sh", "--dry-run", "off", operation)
        assert proc.returncode == 0, proc.stderr
        assert f"docker stop {container}" in proc.stdout


def test_slot_gate_accepts_the_legacy_slot_names_with_a_hint() -> None:
    """The legacy slot names still work — with a hint about the operation name."""
    proc = _run("slot_gate.sh", "--dry-run", "off", "embedding")
    assert proc.returncode == 0, proc.stderr
    assert "docker stop lsb-test-model-proxy-embed" in proc.stdout
    assert "legacy slot-group name" in proc.stderr
    assert "embed" in proc.stderr

    proc = _run("slot_gate.sh", "--dry-run", "off", "summary")
    assert proc.returncode == 0, proc.stderr
    assert "docker stop lsb-test-model-proxy-summary" in proc.stdout
    assert "legacy slot-group name" in proc.stderr
    assert "summarize" in proc.stderr


def test_slot_gate_without_an_action_shows_the_state_only() -> None:
    """No action — the state is shown, nothing is changed (read-only).

    This is the run of the acceptance criteria: `scripts/slot_gate.sh summarize`
    must work with an operation name and must not touch the contour.
    """
    proc = _run("slot_gate.sh", "summarize")
    assert proc.returncode == 0, proc.stderr
    assert "lsb-test-model-proxy-summary" in proc.stdout
    assert "docker stop" not in proc.stdout
    assert "docker start" not in proc.stdout


@pytest.mark.skipif(shutil.which("docker") is None, reason="docker is not available")
def test_slot_fake_accepts_the_operation_names() -> None:
    """`scripts/slot_fake.sh` takes the operation names.

    The env of the branch follows the mode of the contour: the env of the
    operation (`<OP>_BASE_URL`) and, in the `legacy` mode, the env of the slot
    group as well (the whole group goes to the fake, as before).
    """
    env = {
        "SLOT_FAKE_PROJECT_DIR": str(REPO_ROOT),
        "SLOT_FAKE_COMPOSE": "docker-compose.yml",
        # A container that is not up: the mode of the contour is unknown, and the
        # `legacy` default applies (the hook must not depend on a live contour).
        "SLOT_FAKE_APP_CONTAINER": "lsb-not-a-container-for-the-test",
    }
    proc = _run("slot_fake.sh", "--dry-run", "on", "auth", "summarize", env=env)
    assert proc.returncode == 0, proc.stderr
    assert "SUMMARIZE_BASE_URL" in proc.stdout
    assert "SUMMARY_BASE_URL" in proc.stdout
    assert "would run: docker compose" in proc.stdout


@pytest.mark.skipif(shutil.which("docker") is None, reason="docker is not available")
def test_slot_fake_accepts_the_legacy_slot_names_with_a_hint() -> None:
    """The legacy slot names still work — with a hint about the operation name."""
    env = {
        "SLOT_FAKE_PROJECT_DIR": str(REPO_ROOT),
        "SLOT_FAKE_COMPOSE": "docker-compose.yml",
        "SLOT_FAKE_APP_CONTAINER": "lsb-not-a-container-for-the-test",
    }
    proc = _run("slot_fake.sh", "--dry-run", "on", "auth", "summary", env=env)
    assert proc.returncode == 0, proc.stderr
    assert "legacy slot-group name" in proc.stderr
    assert "summarize" in proc.stderr
    assert "SUMMARY_BASE_URL" in proc.stdout

    proc = _run("slot_fake.sh", "--dry-run", "on", "auth", "embedding", env=env)
    assert proc.returncode == 0, proc.stderr
    assert "legacy slot-group name" in proc.stderr
    assert "embed" in proc.stderr


def test_fake_slot_accepts_the_operation_names() -> None:
    """`scripts/fake_slot.py` takes the operation names and plays the role."""
    out, err = _stub_startup("--mode", "ok", "--operation", "judge")
    assert "operation=judge" in out, err

    # An unknown name — a usage error (the stub does not start).
    proc = subprocess.run(
        [sys.executable, str(SCRIPTS / "fake_slot.py"), "--operation", "nosuch"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 2
    assert "invalid choice" in proc.stderr


def test_fake_slot_accepts_the_legacy_slot_names_with_a_hint() -> None:
    """The legacy slot names still work — with a hint about the operation name."""
    out, err = _stub_startup("--mode", "ok", "--operation", "summary")
    assert "legacy slot-group name" in err
    assert "summarize" in err
    assert "operation=summarize" in out

    out, err = _stub_startup("--mode", "ok", "--operation", "embedding")
    assert "legacy slot-group name" in err
    assert "embed" in err


def test_fake_slot_answers_in_the_role_of_the_operation() -> None:
    """The operation picks the role of the answer: the judge — the verdict."""
    from scripts.fake_slot import FakeSlotServer, _chat_answer

    server = FakeSlotServer(("127.0.0.1", 0), "ok", ("m",), False, operation="judge")
    try:
        assert server.role == "verdict"
        content, _finish = _chat_answer("ok", "любой промпт", server.role)
        assert "DUPLICATE" in content
    finally:
        server.server_close()

    server = FakeSlotServer(("127.0.0.1", 0), "ok", ("m",), False, operation="summarize")
    try:
        assert server.role == "text"
        content, _finish = _chat_answer("ok", "любой промпт", server.role)
        assert "DUPLICATE" not in content
    finally:
        server.server_close()

    # Without an operation the role is read from the prompt (as before).
    server = FakeSlotServer(("127.0.0.1", 0), "ok", ("m",), False)
    try:
        assert server.role is None
    finally:
        server.server_close()


# --- 5. both branches of the mode ----------------------------------------------


def test_the_scenarios_read_the_names_of_the_mode() -> None:
    """The acceptance runner reads the queue/health/hook names of the mode.

    The `legacy` mode — the names are exactly the historical ones (the
    behaviour is unchanged); the `operation` mode — the names of the operations.
    """
    from scripts import e2e_release32 as acceptance

    legacy = acceptance.slot_names("summary", {"slot_mode": "legacy"})
    assert legacy == {"health": "summary", "queue": "summary", "target": "summary"}

    operation = acceptance.slot_names("summary", {"slot_mode": "operation"})
    assert operation == {"health": "summarize", "queue": "summarize",
                         "target": "summarize"}

    assert acceptance.slot_names("embedding", {"slot_mode": "legacy"})["queue"] == "vector"
    assert acceptance.slot_names("embedding", {"slot_mode": "operation"})["queue"] == "vector"
    assert acceptance.slot_names("judge", {"slot_mode": "operation"})["health"] == "judge"

    # An unknown mode (an old contour without `slot_mode`) — the `legacy` names.
    acceptance.MODE = acceptance.SLOT_MODEL_LEGACY
    assert acceptance.slot_names("summary", {})["queue"] == "summary"
    assert acceptance.slot_names("summary", {})["health"] == "summary"


def test_the_probe_note_of_the_scenarios_escapes_the_short_note_rule() -> None:
    """The probe note of the slot scenarios is longer than the short-note rule.

    A short probe note is summarized by its own text (LSBN-T-37) — without a
    model call — and leaves the queue on its own: the scenarios would read an
    empty queue instead of the deferred work (the false red of 13.11.2026).
    """
    from scripts import e2e_release32 as acceptance

    text = acceptance.probe_text("e2e32-2601151200", "summary outage")
    assert len(text) > 400
    assert "e2e32-2601151200" in text


# --- 6. the contour is not switched on -----------------------------------------


def test_the_mode_flag_stays_legacy_on_the_contour() -> None:
    """`SLOT_MODEL` stays `legacy` — the flag is not flipped by this step."""
    from app.config import Settings

    assert Settings.model_fields["slot_model"].default == "legacy"

    compose = (REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert 'SLOT_MODEL: "legacy"' in compose


def test_the_hooks_change_nothing_without_an_explicit_action() -> None:
    """Without `off`/`on` the hooks are read-only.

    The gate without an action shows the state (`status`); the fake hook
    without an action is a usage error. Neither changes the contour.
    """
    proc = _run("slot_gate.sh", "summarize")
    assert proc.returncode == 0, proc.stderr
    for verb in ("docker stop", "docker start", "docker pause", "iptables"):
        assert verb not in proc.stdout
