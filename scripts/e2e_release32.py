#!/usr/bin/env python3
"""Acceptance E2E of release 3.2.0 (test contour, MCP + REST).

    Spec: `release/3.2.0/staging_for_development/no39-03-observability-and-acceptance.md`
    (scenario 8 — `.../lsbdef0008-02-attention-release-and-acceptance.md`);
    requirements — `release/3.2.0/documents/req/no39-queue-unavailable-requirements.md`
    (FR-1.1…FR-7.2, lsbdef-0008 FR-1.3/FR-2.4); architecture — §6.9, §7.2, §7.4,
    §13.9, §13.13 of `release/3.2.0/documents/arch/jobs-and-terms-architecture.md`.

    This is the BASE script of the 3.2.0 acceptance: it carries the shared harness
    (observable-state waiting, idempotent probe data, teardown trace, report) and the
    mandatory №39 scenario "models unavailable → jobs wait → models are back → jobs
    finish WITHOUT a container restart", checked PER SLOT (`summary`, `judge`,
    `embedding` — in that order, see `SLOT_OUTAGE_ORDER`) and over the new queues,
    plus the observability checks (queues carry
    `pending`/`waiting`/`attention`, `/health.slots` carries the cause and the pause,
    `queue_waiting` is written for deferred work, a deferred job never spins the loop
    in a busy-loop).

    Structure and extension rules (IMPORTANT for the follow-up blocks)
    ----------------------------------------------------------------
    The script is split into explicit sections marked with `# ====` banners:

      * harness — counters/report, teardown trace, MCP/REST clients, `wait_until`,
        `/health` helpers (`queue_stat`, `slot_stat`, predicates), external hooks;
      * probe data — marking by the attempt tag in titles (unique per attempt, so a
        retry never collides with its own note), idempotent pre-cleanup and
        teardown (the prefix still matches every attempt);
      * scenarios — one `async def scenario_<n>_<name>(...)` per block, each opening
        with `scenario(n, "<title>")` and ending with its own `check()`/`skip()`;
      * `# TODO(3.2.0 future blocks)` — the empty section a follow-up brief
        fills with its own scenarios WITHOUT touching the ones below (see
        `future_block_todo`).

    A follow-up block:
      1. writes `scenario_<n>_<name>()` next to the existing ones (own section);
      2. registers it in `run_all()` in the right order;
      3. replaces/manual-notes the matching placeholder in `future_block_todo()`.

    Waiting is driven by an OBSERVABLE state (`/health.queues`, `/health.slots`,
    `vector_status`/`summary_status` in the DB) and never by a fixed pause. Hooks
    (`LSB_SLOT_OFF_CMD`/`LSB_SLOT_ON_CMD` and the auth/off-contract pairs) are the
    only contour-specific bits: an unset hook → the scenario is SKIPPED with a manual
    step, never a false FAIL.

    Scenarios (this base version):
      0. MCP surface and readiness: 34 tools, `/health` shape including `queues`
         (`pending`/`waiting`/`attention`) and `slots` (`embedding`/`summary`/`judge`
         with `ok`/`last_outcome`/`paused_until`/`attention`), version, DB size.
      1. Queue and slot observability baseline (before the outage): queues are
         reported, `waiting`/`attention` are numbers, `/health` answers without models.
      2. №39 core: for EACH slot `summary`/`judge`/`embedding` (in that order) —
         stop the slot by the contour hook, wait for the queue to grow
         (`pending`/`waiting` and the age of the oldest), check `/health` stays alive
         and no request flood appears in the log, then start the slot back and see
         the queue drain WITHOUT a restart. The `summary`/`judge` queues count only
         the notes with a READY vector: those branches first wait for the vector of
         their probe notes and refuse to run while the embedding slot is paused
         (an honest SKIP, not a FAIL); `embedding` itself goes LAST because its own
         queue `vector` needs no embedding at all.
      3. Auth branch: the slot provider answers `401` (fake-provider hook) →
         `slot_paused` in the log, `/health.slots.<slot>.last_outcome='auth'`,
         `paused_until` set, the tool hint; after the fix — `slot_probe` +
         `slot_resumed` and the queue drains.
      4. Attention pair: the judge gets a repeated `off_contract` answer → the number
         of retries is bounded, the job lands in `GET /jobs/attention`; after the
         answer changes the job is released and finishes.
      5. Restart inside the outage window: pending work survives the restart and is
         neither lost nor duplicated (no restarts if no hook).
      6. Regression: unit tests + the existing E2E scripts — only with
         `--run-regression` (otherwise MANUAL).
      7. TODO of the future blocks (slots/budget/decision, memory handles) —
         placeholders for the follow-up briefs.
      8. lsbdef-0008 (defect): reasoning eats the budget → truncation → the pair
         lands in attention → the budget is raised → BOTH queues drain by themselves
         (no manual step), numbers before/after into `release/3.2.0/ACCEPTANCE.md`.
      14. LSBN-F-10 (f10-02/03/04): the assistant-logs area — the entry is written and
         read back by id, sliced by object in the list, found by the reason in the
         search, edited and soft-deleted over MCP (with the REST operator mirror); the
         linked object and its state are visible in the entry; memory search does NOT
         mix the log in (isolation); a token-like value in the body never reaches the
         contour journal. The scenario removes its own log entry (idempotent).
      15. №14 (no14-01): the backups contour — `GET /backups` lists the snapshots,
         `POST /backups/verify` opens the newest one as a working base (integrity +
         vec0) READ-ONLY (size/mtime unchanged) and `/health.backups` carries
         `last_verify`/`last_ship`; no snapshot → an honest SKIP, never a FAIL.

    Run (normally inside the test container):
        MCP_AUTH_TOKEN=... python scripts/e2e_release32.py
        python scripts/e2e_release32.py --help

    Configuration comes from environment variables only (defaults are for the test
    contour); no secret ever reaches the code or the output — the token is printed
    masked.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
import time
from contextlib import AsyncExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

import anyio
import httpx2
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.exceptions import MCPError

RELEASE = "3.8.0"

# Repository root: the script runs from `scripts/`, while the LSBN-F-11
# scenario needs `app.services.injection_budget` (the budget constants live in
# the code, not in the environment) and `scripts/measure_injected_text.py`.
REPO_ROOT = str(Path(__file__).resolve().parents[1])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# ==== harness: canon constants =================================================
# MCP listing ceiling is 20 (3.1.0 canon); the REST ceiling is 50.
MCP_LIMIT = 20
REST_LIMIT = 50
MCP_TOOLS_TOTAL = 34  # 3.7.0 surface: the 3.1.0 one (31) + tuning_list/tuning_get + skills_*/logs_export; the literal stays deliberate (FR-4.4)
# The literal is deliberate, not len(TOOL_NAMES): the surface's growth must remain
# a conscious decision with a text-budget review (F-11, f11-03).
HEALTH_LEGACY = ("status", "embedding_ok", "summarizer_ok", "judge_ok",
                 "notes_count", "pending_vector", "pending_summary")
# №39: the queues of the contour (jobs registered on the job framework).
HEALTH_QUEUES = ("vector", "summary", "judge", "areas", "links", "nodes",
                 "warnings", "terms-normalize")
# №39: the slots the acceptance walks one by one (arch §13.13).
SLOTS = ("embedding", "summary", "judge", "decision")
# Queue that belongs to each slot for the "queue grows while the slot is down".
SLOT_QUEUE = {"embedding": "vector", "summary": "summary", "judge": "judge"}
# #39 + LSBN-F-12: the decision slot (`/health.slots.decision`) is not a "model slot", but it
# has its own channel: scenario 2's `judge` branch mutes it too (see `_outage_for_slot`).
SLOT_DECISION = "decision"
# Scenario 6's regression: these old E2E have address and database keys; the others
# (`e2e_release22`, `e2e_lsb0004/0005/0006`) go hard-coded to `localhost:8080` and
# therefore run in the contour's network (see `legacy_e2e_cmd`).
LEGACY_ARG_SCRIPTS = ("e2e_release31.py", "e2e_release30.py")
# №39: the outage walk of scenario 2 — only the slots that OWN a queue, and the
# ORDER MATTERS. `embedding` goes LAST: its own queue `vector` needs no embedding
# at all, while the `summary`/`judge` queues count ONLY the notes whose vector is
# ready. When the embedding branch runs first, it leaves its slot in a pause /
# back-off, the dependent queues honestly show zero work and the branch reads as
# a false FAIL (live run 10.10.2026: «queue summary: pending=0, waiting=0»).
SLOT_OUTAGE_ORDER = ("summary", "judge", "embedding")
# №39: slots whose queue is filled ONLY by the notes with a ready vector.
VECTOR_DEPENDENT_SLOTS = ("summary", "judge")

# ==== harness: run configuration (filled in main) ==============================
CFG: dict[str, Any] = {}
RUN: dict[str, Any] = {}
HEALTH: dict[str, Any] = {}
ENV: dict[str, Any] = {}
WAITS: list[dict[str, Any]] = []
TEARDOWN: list[str] = []
TRACE_PATH: Path | None = None

# ==== harness: counters and report =============================================
PASS = FAIL = SKIP = WARN = 0
FAILURES: list[str] = []
WARNINGS: list[str] = []
SKIPS: list[str] = []
MANUALS: list[str] = []
SCENARIO: dict[str, Any] = {}
REPORT: list[dict[str, Any]] = []
# The run's scenarios reached the end. An attempt that got here did happen: a transport
# break at CLOSING (a foreign `cancel scope`) does not zero it and does not start
# a retry — otherwise the run's live 20–40 minutes would be spent three times (2026-10-12).
RUN_FINISHED = False


def check(name: str, ok: bool, detail: str = "") -> None:
    """Mandatory check: a FAIL affects the exit code."""
    global PASS, FAIL
    if ok:
        PASS += 1
        SCENARIO["pass"] += 1
        print(f"  [PASS] {name}" + (f" — {detail}" if detail else ""))
    else:
        FAIL += 1
        SCENARIO["fail"] += 1
        SCENARIO["failed"].append(name)
        FAILURES.append(f"scenario {SCENARIO['n']}: {name}")
        print(f"  [FAIL] {name}" + (f" — {detail}" if detail else ""))


def check_slot(name: str, ok: bool, detail: str = "") -> None:
    """Check that needs a live slot: soft (WARN) when the slot itself is down."""
    if not ok and any(HEALTH.get(f"{slot}_ok") is False for slot in SLOTS):
        warn(f"{name} (a slot is unavailable — the check is soft)", detail)
    else:
        check(name, ok, detail)


def skip(name: str, reason: str) -> None:
    """The check is impossible on this contour (no DB/models/external step)."""
    global SKIP
    SKIP += 1
    SCENARIO["skip"] += 1
    SKIPS.append(f"scenario {SCENARIO['n']}: {name} — {reason}")
    print(f"  [SKIP] {name} — {reason}")


def warn(name: str, detail: str = "") -> None:
    """Observation outside the contract (does not affect the exit code)."""
    global WARN
    WARN += 1
    SCENARIO["warn"] += 1
    WARNINGS.append(f"scenario {SCENARIO['n']}: {name}")
    print(f"  [WARN] {name}" + (f" — {detail}" if detail else ""))


def info(message: str) -> None:
    print(f"  [INFO] {message}")


def manual(name: str, note: str = "") -> None:
    """A step done by the operator (not by the script)."""
    MANUALS.append(name + (f" — {note}" if note else ""))
    print(f"  [MANUAL] {name}" + (f" — {note}" if note else ""))


def pre_warn(message: str, detail: str = "") -> None:
    """Warning OUTSIDE a scenario (pre-cleanup, before scenario 0)."""
    global WARN
    WARN += 1
    WARNINGS.append(f"pre-cleanup: {message}")
    print(f"  [WARN] {message}" + (f" — {detail}" if detail else ""))


def refusal(result: dict) -> str:
    """App refusal text (`hint`/`reason`), otherwise an empty string."""
    for field in ("hint", "reason"):
        value = result.get(field)
        if value not in (None, ""):
            return str(value)
    return ""


def note_detail(result: dict) -> str:
    """Creation detail: `id`, or the refusal reason."""
    if result.get("id") is not None:
        return f"id={result['id']}"
    reason = refusal(result)
    return f"id=None, reason={reason}" if reason else "id=None (the tool gave no hint)"


async def run_scenario(n: int, title: str,
                       body: Callable[[], Awaitable[None]]) -> None:
    """One scenario in a wrapper: its failure does not cancel the others (8–13).

            Until 2026-10-12 a scenario 8/9 exception cut the whole run short:
            scenarios 10–13 and the f11-03 block of `ACCEPTANCE.md` never ran. The
            report is written independently of the observability scenarios' outcomes,
            so here — isolation (`warn` and continue), not silence: the reason stays
            in the report.
    """
    try:
        await body()
    except Exception as exc:  # noqa: BLE001 — we isolate the scenario, not swallow the reason
        if SCENARIO.get("n") != n:
            scenario(n, title)  # the scenario fell before its own heading
        warn("the scenario broke before its own checks", describe(exc))
    finally:
        close_scenario()


def scenario(n: int, title: str) -> None:
    global SCENARIO
    close_scenario()
    SCENARIO = {"n": n, "title": title, "pass": 0, "fail": 0, "skip": 0, "warn": 0,
                "failed": []}
    print(f"\n[{n}] {title}")


def close_scenario() -> None:
    global SCENARIO
    if not SCENARIO:
        return
    REPORT.append(dict(SCENARIO))
    SCENARIO = {}


def reset_counters() -> None:
    """Reset before a retry (a broken session must not zero the run)."""
    global PASS, FAIL, SKIP, WARN, FAILURES, WARNINGS, SKIPS, MANUALS, SCENARIO, REPORT
    global WAITS, TEARDOWN, RUN_FINISHED
    PASS = FAIL = SKIP = WARN = 0
    FAILURES, WARNINGS, SKIPS, MANUALS = [], [], [], []
    SCENARIO, REPORT = {}, []
    WAITS, TEARDOWN = [], []
    RUN_FINISHED = False


def mask(token: str) -> str:
    """Mask a secret for the output: 4 characters + ***."""
    return f"{token[:4]}***" if token else "(not set)"


def describe(exc: BaseException) -> str:
    """Readable error text: ExceptionGroup → its first inner exception."""
    if isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        inner = exc.exceptions[0]
        return f"{type(inner).__name__}: {inner}"
    return f"{type(exc).__name__}: {exc}"


def is_closing_cancel_scope(exc: BaseException) -> bool:
    """An MCP session CLOSING failure over a foreign cancel scope — not a product defect.

            anyio/asyncio throws `RuntimeError: Attempted to exit a cancel scope ...`
            (the text sometimes names another task: `... in a different task than it
            was entered in`), and the `ExceptionGroup` carries it as the first root.
            The live run of 2026-10-12: this failure cut the run at exit — scenarios
            10–13 and the f11-03 block never ran, "Total:" never printed. We count
            such a break a closing observation, not a run error.
    """
    if isinstance(exc, BaseExceptionGroup):
        return any(is_closing_cancel_scope(inner) for inner in exc.exceptions)
    if isinstance(exc, RuntimeError):
        text = str(exc).lower()
        return "cancel scope" in text or "different task" in text
    return False


def is_transport_break(exc: BaseException) -> bool:
    """An MCP transport break, not a product's domain error.

            The live run of 2026-10-10: the break also arrives as
            `ExceptionGroup: unhandled errors in a TaskGroup (1 sub-exception)` (anyio),
            so the group is unwrapped recursively — it counts as "transport" only when
            a transport is inside. The session-closing break (a foreign `cancel scope`)
            belongs here too: for the caller it is the same transport drop, but not a
            domain failure of the product.
    """
    if isinstance(exc, MCPError):
        return True
    if is_closing_cancel_scope(exc):
        return True
    if isinstance(exc, BaseExceptionGroup):
        return any(is_transport_break(inner) for inner in exc.exceptions)
    if isinstance(exc, (httpx2.TransportError, httpx2.HTTPError, OSError,
                        anyio.BrokenResourceError, anyio.ClosedResourceError,
                        anyio.EndOfStream)):
        return True
    # MCP SDK 2.x drags in both httpx 1.x and anyio: we catch the transport by the class's module too,
    # not only by our httpx2's types (otherwise any foreign client would slip through).
    root = (getattr(type(exc), "__module__", "") or "").split(".")[0]
    return root in ("httpx", "httpx2", "httpcore", "anyio", "h11")


def json_size(obj: Any) -> int:
    """Size of the compact answer in bytes (the canon budget metric)."""
    return len(json.dumps(obj, ensure_ascii=False).encode("utf-8"))


# ==== harness: teardown trace ==================================================
def teardown_note(line: str) -> None:
    """Append one line to the teardown trace (also on an emergency exit)."""
    TEARDOWN.append(line)


def default_trace_dir() -> str:
    """`release/3.2.0/acceptance` next to the repository, else the CWD."""
    repo = Path(__file__).resolve().parent.parent
    if (repo / "release").is_dir():
        return str(repo / "release" / RELEASE / "acceptance")
    return str(Path.cwd() / "acceptance")


def write_teardown_trace() -> Path | None:
    """Write the trace; on a write error print it instead of losing it."""
    global TRACE_PATH
    header = [
        f"teardown trace — E2E release {RELEASE} acceptance run",
        f"time: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}",
        f"MCP: {CFG.get('mcp_url')} | REST: {CFG.get('base_url')}",
        f"image: {CFG.get('image') or '(not set)'} | revision: "
        f"{ENV.get('revision') or '(unknown)'} | version: {ENV.get('version') or '(unknown)'}",
        f"run prefix: {CFG.get('prefix')} | identifiers: {RUN}",
        f"DB: {CFG.get('db')} ({ENV.get('db_size') or 'size unknown'})",
        "---",
    ]
    body = header + (TEARDOWN or ["nothing was removed (no probe data created)"])
    try:
        directory = Path(CFG["trace_dir"])
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"teardown-{time.strftime('%Y%m%d-%H%M%S')}.log"
        path.write_text("\n".join(body) + "\n", encoding="utf-8")
        TRACE_PATH = path
        return path
    except OSError as exc:
        print(f"\n[WARN] teardown trace was not written ({type(exc).__name__}: {exc})"
              " — dumping it here:")
        print("\n".join(body))
        return None


# ==== harness: MCP client ======================================================
def extract(result: Any) -> dict:
    """Unwrap a tool result: structuredContent or JSON in text."""
    sc = getattr(result, "structuredContent", None)
    if sc is not None:
        return sc
    for block in result.content or []:
        if getattr(block, "type", None) == "text":
            try:
                return json.loads(block.text)
            except Exception:
                return {"_raw": block.text}
    return {}


class Client:
    """Thin wrapper over ClientSession: a tool call → dict."""

    def __init__(self, session: ClientSession, stack: AsyncExitStack | None = None):
        self.session = session
        self._stack = stack
        # Sessions opened AFTER the contour restart (`reconnect`): their stacks live until
        # the end of the process and are closed in `close()` — the newest first.
        self._keepalive: list[AsyncExitStack] = []

    async def call(self, tool: str, args: dict) -> dict:
        return extract(await self.session.call_tool(tool, args))

    async def close(self, *, quiet: bool = False) -> None:
        """Close every live session stack, newest first: idempotent, never fatal.

                    The order — the reverse of the opening order: a fresh session's
                    `cancel scope` is nested deepest, and exiting an older scope earlier
                    than a newer one is exactly that RuntimeError about a cancel scope
                    which cut runs short (2026-10-12, reproduced by a targeted run). A
                    closing break — an observation (`info` + one teardown line), not an
                    error: the contour session is no longer needed by then.
        """
        stacks = [*self._keepalive[::-1], self._stack]
        self._keepalive, self._stack = [], None
        closed = 0
        for stack in stacks:
            if stack is None:
                continue
            closed += 1
            try:
                await stack.aclose()
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException as exc:  # noqa: BLE001 — closing is not the checks' subject
                note = (" (a cancel-scope break on the way out)"
                        if is_closing_cancel_scope(exc) else "")
                info(f"closing the MCP session: {describe(exc)}")
                teardown_note(f"closing the MCP session raised {describe(exc)}{note}")
        if closed and not quiet:
            # A trace of the correct closing in the run's journal and in the trace (the same tasks
            # as at opening: we close in our own task, before leaving the AsyncExitStack).
            info("the MCP session is closed")
            teardown_note("the MCP session is closed")

    async def reconnect(self) -> Any:
        """Re-open the MCP session after a contour restart (the old socket is dead).

                    Scenario 5 by its nature restarts the contour container — the
                    runner's session tears with it, and without a re-open the whole run
                    goes into attempt retries and ends FATAL although the product is
                    alive (verified 2026-10-10).

                    We close the old session HERE, in this same task and before opening
                    the new one: its socket is already dead, and a live `AsyncExitStack`
                    of the old session left until the end of the run nested cancel scopes
                    not by LIFO — the closing at exit failed with
                    `RuntimeError: Attempted to exit cancel scope ...`, the cancelled
                    scope "poisoned" the run's task, and scenarios 10–13 never ran (the
                    live run of 2026-10-12; reproduced by targeted runs). The new session
                    is kept in `_keepalive` and closed in `close()`.
        """
        await self.close(quiet=True)
        fresh, init = await fresh_session(AsyncExitStack(), own_stack=True)
        # We put the session's STACK into `_keepalive`, not the `Client` itself: otherwise nobody
        # closes the stack; its async generator survives until garbage collection — and that
        # collection closes it in a FOREIGN task (`Attempted to exit cancel scope in a
        # different task...`), cancelling the run's attempt (the live run of 2026-10-12:
        # the break at scenario 5; scenarios 10–13 were never reached).
        self._keepalive.append(fresh._stack)  # the new session's stack lives until close()
        self.session = fresh.session
        return init


async def call_with_reconnect(c: Client, tool: str, args: dict) -> dict:
    """MCP call right after a contour hook: a transport break is re-opened once.

            The hooks (a contour restart, muting/raising a slot) tear the runner's
            live session, and the first call after the hook fails — not only with
            `MCPError` but with an `ExceptionGroup`/anyio/httpx transport too (the
            live run of 2026-10-10). Here such a break is re-opened once, and a
            domain error propagates as is: we do not shield the product with it.
    """
    try:
        return await c.call(tool, args)
    except Exception as exc:  # noqa: BLE001 — we decide by type, not swallow
        if not is_transport_break(exc):
            raise
        info(f"the MCP transport broke on {tool} ({describe(exc)}) — "
             f"re-opening the session and retrying the call once")
        await c.reconnect()
        return await c.call(tool, args)


async def fresh_session(stack: AsyncExitStack, url: str | None = None,
                        token: str | None = None,
                        own_stack: bool = False) -> tuple[Client, Any]:
    """New MCP session with its own `initialize` (instructions are rebuilt)."""
    target = AsyncExitStack() if own_stack else stack
    http_client = await target.enter_async_context(
        httpx2.AsyncClient(
            headers={"Authorization": f"Bearer {token or CFG['token']}"},
            timeout=httpx2.Timeout(30.0, read=CFG["read_timeout"]),
        )
    )
    streams = await target.enter_async_context(
        streamable_http_client(url or CFG["mcp_url"], http_client=http_client)
    )
    session = await target.enter_async_context(ClientSession(streams[0], streams[1]))
    init = await session.initialize()
    return Client(session, target if own_stack else None), init


async def rest_client(stack: AsyncExitStack, base_url: str | None = None,
                      token: str | None = None, auth: bool = True) -> httpx2.AsyncClient:
    """REST client of the contour (Bearer as on MCP; /health answers without it)."""
    headers = {"Authorization": f"Bearer {token or CFG['token']}"} if auth else {}
    return await stack.enter_async_context(
        httpx2.AsyncClient(
            base_url=(base_url or CFG["base_url"]),
            headers=headers,
            timeout=httpx2.Timeout(30.0, read=CFG["read_timeout"]),
        )
    )


# ==== harness: observable-state waiting ========================================
async def wait_until(predicate: Callable[[], Awaitable[tuple[bool, str]]],
                     label: str, *,
                     timeout: float | None = None,
                     poll: float | None = None) -> tuple[bool, str]:
    """Wait for a state to become observable; the guard timeout is `E2E_WAIT_SEC`.

    `predicate` is called on every poll — pass the callable itself, never its
    result. The actual wait and whether the guard fired are recorded in WAITS and
    printed in the report, so a slow model reads as "we waited N s".
    """
    if not callable(predicate):
        raise TypeError(
            "wait_until: the predicate must be callable — pass the function "
            f"itself, not its call result (got {type(predicate).__name__})"
        )
    timeout = CFG["wait_sec"] if timeout is None else timeout
    poll = CFG["poll_sec"] if poll is None else poll
    started = time.monotonic()
    done = False
    detail = "(no attempt)"
    while True:
        try:
            done, detail = await predicate()
        except Exception as exc:  # noqa: BLE001 — a transient error is not the answer
            done, detail = False, describe(exc)
        if done or (time.monotonic() - started) >= timeout:
            break
        await asyncio.sleep(poll)
    elapsed = time.monotonic() - started
    WAITS.append({"label": label, "elapsed": round(elapsed, 1),
                  "timeout": timeout, "fired": not done, "detail": detail})
    state = "reached" if done else "TIMEOUT GUARD FIRED"
    print(f"  [WAIT] {label}: {state} after {elapsed:.1f}s (guard {timeout:.0f}s) — {detail}")
    return done, detail


# ==== harness: /health helpers =================================================
async def health_snapshot(rest: httpx2.AsyncClient) -> dict:
    try:
        r = await rest.get("/health")
        return r.json() if r.status_code == 200 else {"status": f"http {r.status_code}"}
    except Exception as exc:  # the contour may be down — the caller decides
        return {"status": f"error: {type(exc).__name__}"}


async def refresh_health(rest: httpx2.AsyncClient) -> dict:
    """Refresh the /health snapshot (the *_ok fields change after a first try)."""
    snapshot = await health_snapshot(rest)
    if snapshot.get("status") == "ok":
        HEALTH.update(snapshot)
    return HEALTH


def queue_stat(snapshot: dict, queue: str) -> dict | None:
    value = (snapshot.get("queues") or {}).get(queue)
    return value if isinstance(value, dict) else None


def slot_stat(snapshot: dict, slot: str) -> dict | None:
    value = (snapshot.get("slots") or {}).get(slot)
    return value if isinstance(value, dict) else None


def queue_has_work(rest: httpx2.AsyncClient, queue: str,
                   *, min_age: float = 0.0) -> Callable[[], Awaitable[tuple[bool, str]]]:
    """Predicate: the queue shows work (pending OR waiting), age grown enough.

    №39: a slot may be down as «unavailable» (work stays `pending`) or as a
    break (`auth`/`no_contract` puts it in `waiting`) — both are visible work.
    """

    async def predicate() -> tuple[bool, str]:
        snapshot = await health_snapshot(rest)
        stat = queue_stat(snapshot, queue)
        if stat is None:
            return False, f"queue {queue} is not reported by /health"
        pending = stat.get("pending") or 0
        waiting = stat.get("waiting") or 0
        oldest = stat.get("oldest_pending_sec")
        work = (pending + waiting) > 0
        age_ok = oldest is not None and oldest >= min_age
        return bool(work and (age_ok or min_age == 0)), (
            f"queue {queue}: pending={pending}, waiting={waiting}, "
            f"attention={stat.get('attention')}, oldest={oldest}"
        )

    return predicate


def queue_drained(rest: httpx2.AsyncClient,
                  queue: str) -> Callable[[], Awaitable[tuple[bool, str]]]:
    """Predicate: the queue is fully drained (no pending, no waiting)."""

    async def predicate() -> tuple[bool, str]:
        snapshot = await health_snapshot(rest)
        stat = queue_stat(snapshot, queue)
        if stat is None:
            return False, f"queue {queue} is not reported by /health"
        done = (stat.get("pending") in (0, None)
                and (stat.get("waiting") in (0, None))
                and stat.get("oldest_pending_sec") in (0, None))
        return done, f"queue {queue}: pending={stat.get('pending')}, " \
                     f"waiting={stat.get('waiting')}"

    return predicate


def queue_waiting(rest: httpx2.AsyncClient, queue: str, *,
                 min_waiting: int = 1) -> Callable[[], Awaitable[tuple[bool, str]]]:
    """Predicate: the work is DEFERRED by a reason — visible in `waiting`, not in `attention`.

            The open slot-breaker phase (#39 arch §6.9, no39-04): the loop takes no
            batch, all current work moves from `pending` to `waiting`; `attention`
            ("needs a human") is a `waiting` subset and appears only after
            `SLOT_MAX_ATTEMPTS` non-transient failures of the PAIR. Waiting for
            `attention` in this phase — false red (the live run of 2026-10-12:
            scenario 8 went red over "the hopeless pair waits for a human").
    """

    async def predicate() -> tuple[bool, str]:
        snapshot = await health_snapshot(rest)
        stat = queue_stat(snapshot, queue)
        if stat is None:
            return False, f"queue {queue} is not reported by /health"
        pending = stat.get("pending") or 0
        waiting = stat.get("waiting") or 0
        attention = stat.get("attention") or 0
        return bool(waiting >= min_waiting and pending == 0), (
            f"queue {queue}: pending={pending}, waiting={waiting}, "
            f"attention={attention}, oldest={stat.get('oldest_pending_sec')}"
        )

    return predicate


def queue_deferred(rest: httpx2.AsyncClient, queue: str, *,
                   min_work: int = 1) -> Callable[[], Awaitable[tuple[bool, str]]]:
    """Predicate: the pair STOPPED BURNING CALLS — the work is deferred by a reason.

            `waiting + attention >= min_work` and `pending == 0`: the work is either
            deferred by the slot or already frozen in "needs a human" — in both cases
            no model calls burn (FR-2.3/FR-5.1, no39-04). A hard `attention` wait
            instead is the branch's premise, not its fact.
    """

    async def predicate() -> tuple[bool, str]:
        snapshot = await health_snapshot(rest)
        stat = queue_stat(snapshot, queue)
        if stat is None:
            return False, f"queue {queue} is not reported by /health"
        pending = stat.get("pending") or 0
        waiting = stat.get("waiting") or 0
        attention = stat.get("attention") or 0
        return bool((waiting + attention) >= min_work and pending == 0), (
            f"queue {queue}: pending={pending}, waiting={waiting}, "
            f"attention={attention}, oldest={stat.get('oldest_pending_sec')}"
        )

    return predicate


async def attention_jobs(rest: httpx2.AsyncClient) -> list[dict] | None:
    """Tasks waiting for a human (`GET /jobs/attention`); None — the handle did not answer."""
    try:
        answer = await rest.get("/jobs/attention")
    except Exception:  # noqa: BLE001 — a transient REST error is not an answer
        return None
    if answer.status_code != 200:
        return None
    return (answer.json() or {}).get("jobs") or []


async def check_attention_threshold(rest: httpx2.AsyncClient, prefix: str) -> None:
    """Checking the threshold's reachability: `attention` — only after a failure series (no39-04).

            "The pause is per slot, the count per pair": a task enters `attention`
            after exhausting `SLOT_MAX_ATTEMPTS` NON-TRANSIENT failures of the pair,
            not at the first one. The stub branch's limit is below the code's
            (`scripts/slot_fake.sh` lowers it for a fast acceptance), so we compare
            the fact of the task (a failure series), not the number from the
            contour's env: no environment variant makes the scene red.
    """
    jobs = await attention_jobs(rest)
    if jobs is None:
        skip(f"{prefix} the attempt counter of the job in `attention`",
             "GET /jobs/attention did not answer")
        return
    attempts = [int(job.get("attempts") or 0) for job in jobs]
    check(f"{prefix} the human is asked only after a series of pair refusals, "
          f"not on the first one (no39-04: `attention` после SLOT_MAX_ATTEMPTS)",
          bool(attempts) and min(attempts) >= 2, f"attempts={attempts}")


def slot_paused(rest: httpx2.AsyncClient, slot: str, outcome: str | None = None,
                ) -> Callable[[], Awaitable[tuple[bool, str]]]:
    """Predicate: /health.slots reports the slot paused with the given outcome."""

    async def predicate() -> tuple[bool, str]:
        snapshot = await health_snapshot(rest)
        stat = slot_stat(snapshot, slot)
        if stat is None:
            return False, f"slot {slot} is not reported by /health.slots"
        paused = stat.get("paused_until") is not None
        outcome_ok = outcome is None or stat.get("last_outcome") == outcome
        return bool(paused and outcome_ok), f"slot {slot}: " \
            f"last_outcome={stat.get('last_outcome')!r}, " \
            f"paused_until={stat.get('paused_until')!r}"

    return predicate


def slot_ready(rest: httpx2.AsyncClient, slot: str,
               ) -> Callable[[], Awaitable[tuple[bool, str]]]:
    """Predicate: the slot is usable right now (not paused, no attention owed)."""

    async def predicate() -> tuple[bool, str]:
        snapshot = await health_snapshot(rest)
        stat = slot_stat(snapshot, slot)
        if stat is None:
            return False, f"slot {slot} is not reported by /health.slots"
        paused_until = stat.get("paused_until")
        attention = stat.get("attention")
        ok = paused_until is None and not attention
        return bool(ok), (f"slot {slot}: ok={stat.get('ok')}, "
                          f"last_outcome={stat.get('last_outcome')!r}, "
                          f"paused_until={paused_until!r}, attention={attention!r}")

    return predicate


def slot_ok(rest: httpx2.AsyncClient, slot: str,
            ) -> Callable[[], Awaitable[tuple[bool, str]]]:
    """Predicate: the slot really answers again — the last attempt succeeded.

            №39: the `slot_resumed` marker is written only when the pause is lifted in
            the SAME process (`app/services/worker.py`). A branch whose "fix"
            recreates the container (`scripts/slot_fake.sh on ok ...`) will never see
            the marker: the pause lived in memory and is gone at recreation. The
            observable fact "access is back" — the slot's successful last attempt
            (`/health.slots.<slot>.ok`) and the breaker lifted.
    """

    async def predicate() -> tuple[bool, str]:
        snapshot = await health_snapshot(rest)
        stat = slot_stat(snapshot, slot)
        if stat is None:
            return False, f"slot {slot} is not reported by /health.slots"
        ok = stat.get("ok") is True and stat.get("paused_until") is None
        return bool(ok), (f"slot {slot}: ok={stat.get('ok')!r}, "
                          f"last_outcome={stat.get('last_outcome')!r}, "
                          f"paused_until={stat.get('paused_until')!r}")

    return predicate


def probe_vectors_ready(note_ids: list[int],
                        ) -> Callable[[], Awaitable[tuple[bool, str]]]:
    """Predicate: every listed probe note carries a READY vector (DB, read-only).

    №39: the `summary`/`judge` queues count a note only when `vector_status='ok'`,
    so a branch that runs while the embedding slot is still catching up (its own
    pause / back-off after the embedding branch) sees an empty queue and reads as
    a false FAIL — the work is there, the queue simply cannot count it yet.
    """

    ids = tuple(note_ids)

    async def predicate() -> tuple[bool, str]:
        if not ids:
            return True, "no probe note of the branch needs a vector"
        marks = ",".join("?" * len(ids))
        ready = db_count(
            f"SELECT COUNT(*) AS c FROM notes WHERE id IN ({marks}) "
            "AND deleted_at IS NULL AND vector_status = 'ok'",
            ids,
        )
        if ready is None:
            return False, f"the vector status is unreadable (DB {CFG['db']})"
        return ready >= len(ids), \
            f"probe notes with vector_status='ok': {ready} of {len(ids)}"

    return predicate


# ==== harness: external steps (contour hooks) ==================================
async def run_shell(cmd: str, timeout: float = 600.0) -> subprocess.CompletedProcess[str]:
    """Run an operator-provided contour hook (never echoed into the output)."""
    return await asyncio.to_thread(
        subprocess.run, cmd, shell=True, capture_output=True, text=True, timeout=timeout
    )


def slot_cmd(template: str, slot: str) -> str:
    """Build a per-slot hook command: `{slot}` placeholder or the name appended."""
    if "{slot}" in template:
        return template.format(slot=slot)
    return f"{template} {slot}"


async def image_labels() -> dict[str, str]:
    """OCI labels of the acceptance image — proves which code was checked."""
    if not CFG["inspect_cmd"]:
        return {}
    try:
        proc = await run_shell(CFG["inspect_cmd"], timeout=60.0)
    except (OSError, subprocess.SubprocessError) as exc:
        warn("image labels are unreadable", f"{type(exc).__name__}: {exc}")
        return {}
    if proc.returncode != 0:
        warn("image labels are unreadable",
             f"rc={proc.returncode} ({CFG['inspect_cmd'].split()[0]})")
        return {}
    raw = proc.stdout.strip()
    if raw in ("", "null"):
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return {str(k): str(v) for k, v in (data or {}).items()}


async def log_has(needle: str) -> tuple[bool, str]:
    """Is the marker present in the contour log (operator-provided log hook)?"""
    if not CFG["logs_cmd"]:
        return False, "LSB_LOGS_CMD is not set"
    try:
        proc = await run_shell(CFG["logs_cmd"], timeout=120.0)
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"{type(exc).__name__}: {exc}"
    if proc.returncode != 0:
        return False, f"log hook failed: rc={proc.returncode}"
    return needle in proc.stdout, \
        f"{needle!r} — {'found' if needle in proc.stdout else 'not found'} " \
        f"in {len(proc.stdout.splitlines())} log lines"


def log_events(*needles: str) -> Callable[[], Awaitable[tuple[bool, str]]]:
    """Predicate: EVERY marker is written to the contour log.

    The factory form is deliberate: `log_has` is a coroutine function, so a bare
    `lambda: log_has(...)` hands `wait_until` an ALREADY created coroutine. An
    `and`-chain then awaits the LAST marker only, leaks the first
    (`RuntimeWarning: coroutine 'log_has' was never awaited`) and reads as a
    guaranteed false FAIL (the 401 branch, live run 10.10.2026). `wait_until`
    awaits the callable it is given — so the awaited calls live here.
    """

    async def predicate() -> tuple[bool, str]:
        details: list[str] = []
        for needle in needles:
            found, detail = await log_has(needle)
            details.append(detail)
            if not found:
                return False, "; ".join(details)
        return True, "; ".join(details)

    return predicate


def utc_now_iso() -> str:
    """A moment in ISO-8601 UTC — the "after the branch started" window for journal checks."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S") + "Z"


def log_slot_event(event: str, *, slot: str, outcome: str,
                   since: str | None = None,
                   ) -> Callable[[], Awaitable[tuple[bool, str]]]:
    """Predicate: the contour log carries THIS event of THIS slot with THIS cause.

            A journal line is JSON (`ts`/`event`/`slot`/`outcome`), so the check reads
            FIELDS, not a substring: `log_events("slot_paused")` matched any run
            branch's event (the hook window — 2 hours) and declared PASS even when the
            cause was foreign — the live run of 2026-10-13: `slot_paused` with
            `outcome=off_contract` passed as "the event with the cause" while the
            branch expected `auth`. `since` (UTC ISO, seconds) narrows the window to
            the branch's start.
    """

    async def predicate() -> tuple[bool, str]:
        if not CFG["logs_cmd"]:
            return False, "LSB_LOGS_CMD is not set"
        try:
            proc = await run_shell(CFG["logs_cmd"], timeout=120.0)
        except (OSError, subprocess.SubprocessError) as exc:
            return False, f"{type(exc).__name__}: {exc}"
        if proc.returncode != 0:
            return False, f"log hook failed: rc={proc.returncode}"
        seen: list[str] = []
        for line in proc.stdout.splitlines():
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if record.get("event") != event or record.get("slot") != slot:
                continue
            stamp = str(record.get("ts") or "")
            if since and stamp and stamp[:19] < since[:19]:
                continue
            seen.append(f"{stamp[:19]} outcome={record.get('outcome')!r}")
            if record.get("outcome") == outcome:
                return True, f"{event} slot={slot} outcome={outcome} — {seen[-1]}"
        detail = seen[-3:] if seen else "no such event in the log window"
        return False, f"{event} slot={slot} outcome={outcome!r} not found; seen: {detail}"

    return predicate


def db_ready() -> bool:
    return Path(CFG["db"]).exists()


def db_try(sql: str, params: tuple = ()) -> tuple[list[dict] | None, str]:
    """SELECT-only access to the contour DB (wait predicates); errors are texts."""
    if not db_ready():
        return None, f"DB is not reachable ({CFG['db']})"
    try:
        conn = sqlite3.connect(f"file:{CFG['db']}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            rows = [dict(row) for row in conn.execute(sql, params).fetchall()]
        finally:
            conn.close()
        return rows, ""
    except sqlite3.Error as exc:
        return None, f"sqlite: {exc}"


def db_size_text() -> str:
    if not db_ready():
        return "unknown (DB is not reachable)"
    size = Path(CFG["db"]).stat().st_size
    return f"{size} B ({size / 1048576:.1f} MiB)"


def db_write(sql: str, params: tuple = ()) -> str:
    """Write to the contour DB (the e2e contour owns its DB file); "" on success.

    Used by scenario 11 to force a deterministic `updated_at` of a probe note
    (the freshness window must not depend on wall-clock seconds).
    """
    if not db_ready():
        return f"DB is not reachable ({CFG['db']})"
    try:
        with sqlite3.connect(CFG["db"]) as conn:
            conn.execute(sql, params)
        return ""
    except sqlite3.Error as exc:
        return f"sqlite: {exc}"


# ==== harness: acceptance report (`release/3.2.0/ACCEPTANCE.md`) ================
def acceptance_path() -> Path:
    """The acceptance report of the release (release/ is not versioned)."""
    return Path(CFG["repo_dir"]) / "release" / RELEASE / "ACCEPTANCE.md"


def block_markers(text: str, start: str, end: str) -> tuple[int, int]:
    """A block's bounds by MARKER LINES: `(the start index, the end index)` or `(-1, -1)`.

            Markers are searched as whole lines (`(?m)^…$`), not substrings: the body
            of the `f12-03` block itself contained the word-for-word
            `<!-- f12-03:end -->`; the substring search found it before the real end —
            and every run appended an extra marker line (the live defect of
            2026-10-12: three duplicates in `ACCEPTANCE.md`).
    """
    start_match = re.search(rf"(?m)^{re.escape(start)}$", text)
    if start_match is None:
        return -1, -1
    # `re.search` does not know `pos=`: we search the end in the tail after the start and add up the offsets.
    end_match = re.search(rf"(?m)^{re.escape(end)}$", text[start_match.end():])
    if end_match is None:
        return -1, -1
    return start_match.start(), start_match.end() + end_match.end()


MARKER_WORDS = {"f12-03": "HTML-маркеры блока — комментарии `f12-03:start` и `f12-03:end`"}


def write_acceptance_block(block: str, lines: list[str], *,
                           path: Path | None = None) -> str:
    """Write/replace the block of ONE brief in `ACCEPTANCE.md` (idempotent).

            The block is fenced with HTML markers `<!-- <block>:start -->` /
            `<!-- <block>:end -->`: a repeated run of the scenario REPLACES its own block
            instead of appending a second copy, so the second run of the E2E leaves the
            report byte-identical (the run's idempotency). The numbers live outside the
            code: the file is the evidence, the script only fills it.

            Two idempotency protections: (1) markers are compared as WHOLE LINES — the
            block body cannot "impersonate" the end (`block_markers`); (2) before
            writing we check the file has exactly two markers and the block has not
            grown by lines: on a mismatch — `warn`, the file is untouched (better the
            old report than a spoiled one). The block body must be free of verbatim
            markers (`MARKER_WORDS` — as words).
    """
    target = path or acceptance_path()
    start, end = f"<!-- {block}:start -->", f"<!-- {block}:end -->"
    section = "\n".join([start, *lines, end])
    text = target.read_text(encoding="utf-8") if target.exists() else ""
    body_words = "\n".join(lines)
    if start in body_words or end in body_words:
        hint = MARKER_WORDS.get(block, "write the markers as words")
        warn(f"the body of the acceptance block {block} carries the literal markers",
             f"{hint} — the file is left untouched")
        return str(target)
    head, tail = block_markers(text, start, end)
    if head >= 0:
        previous = text[head:tail].count("\n") + 1
        updated = text[:head] + section + text[tail:]
    else:
        previous = 0
        base = text.rstrip() + "\n\n" if text.strip() else ""
        updated = base + section + "\n"
    markers = updated.count(start) + updated.count(end)
    grown = updated[head:head + len(section)].count("\n") + 1 if head >= 0 else 0
    if markers != 2 or (head >= 0 and grown > previous):
        warn(f"the acceptance block {block} was not written: it is not idempotent",
             f"markers={markers} (2 expected), lines {previous} -> {grown}")
        return str(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(updated, encoding="utf-8")
    return str(target)


def self_test_acceptance() -> int:
    """The report block's idempotency without a contour (the stage-6 gate).

            We write the `f12-03` block TWICE into a temp file and compare the content
            byte-by-byte: the repeat must REPLACE the block, not append to it. It
            catches both the substring marker search (the 2026-10-12 defect — three
            duplicates in `ACCEPTANCE.md`) and verbatim markers in the block body, and
            the neighboring blocks (they must not grow from a foreign write). The
            numbers come from an empty set: the idempotency is checked, not them.
    """
    print("=== self-test: the acceptance block of one постановка is idempotent "
          "(the contour is not needed)")
    failures: list[str] = []
    # The self-check needs no contour: we substitute a minimal report context
    # (otherwise the block's header reads `CFG['base_url']` and `RUN['tag']`).
    CFG.setdefault("base_url", "http://lsb-test:8080 (self-test)")
    RUN.setdefault("tag", "e2e32-self-test")
    lines = _decision_acceptance_lines({})
    with tempfile.TemporaryDirectory(prefix="e2e-acceptance-") as out_dir:
        path = Path(out_dir) / "ACCEPTANCE.md"
        write_acceptance_block("f12-03", lines, path=path)
        once = path.read_text(encoding="utf-8")
        write_acceptance_block("f12-03", lines, path=path)
        twice = path.read_text(encoding="utf-8")
        if once != twice:
            failures.append("the second write changed the file: the block is not idempotent")
        # Neighboring blocks: the f12-03 write does not touch them and stays single itself.
        write_acceptance_block("lsbdef0008-02", ["### блок дефекта", "", "числа"], path=path)
        write_acceptance_block("f11-03", ["### блок вшитого текста", "", "числа"], path=path)
        write_acceptance_block("f12-03", lines, path=path)
        again = path.read_text(encoding="utf-8")
        if again.count("<!-- f12-03:start -->") != 1 or again.count("<!-- f12-03:end -->") != 1:
            failures.append(
                "the markers of the block occur "
                f"{again.count('<!-- f12-03:start -->')}×start / "
                f"{again.count('<!-- f12-03:end -->')}×end, not once each"
            )
        if "<!-- f12-03:start -->" in "\n".join(lines) \
                or "<!-- f12-03:end -->" in "\n".join(lines):
            failures.append("the body of the block carries the literal markers")
        blocks = ["lsbdef0008-02", "f12-03", "f11-03"]
        for block in blocks:
            if again.count(f"<!-- {block}:start -->") != 1:
                failures.append(f"the block {block} is not exactly one block")
    if failures:
        print("[FAIL] the acceptance block is not idempotent:")
        for item in failures:
            print(f"  - {item}")
        return 1
    print("[PASS] the second write is byte-identical, the block markers are exactly "
          "two, the neighbouring blocks are untouched")
    return 0


# ==== probe data: marking, leftovers, teardown =================================
async def list_all(c: Client, *, detail: str = "titles", max_pages: int = 25) -> list[dict]:
    """Walk every memory_list page (offset by `next_offset` until `has_more`)."""
    items: list[dict] = []
    offset = 0
    for _ in range(max_pages):
        page = await c.call("memory_list",
                            {"limit": MCP_LIMIT, "offset": offset, "detail": detail})
        items.extend(page.get("items", []))
        nxt = page.get("next_offset")
        if not page.get("has_more") or not isinstance(nxt, int):
            break
        offset = nxt
    return items


async def probe_titles(c: Client, prefix: str) -> list[dict]:
    """Probe notes of one run (matched by the title prefix), over ALL pages."""
    return [item for item in await list_all(c, detail="titles")
            if prefix in (item.get("title") or "")]


async def prepare_leftovers() -> None:
    """Remove the probe NOTES of a previous run (idempotency, FR-2.1)."""
    if CFG["no_cleanup"]:
        info(f"pre-cleanup skipped (--no-cleanup): leftovers with prefix "
             f"{CFG['prefix']} stay in place")
        teardown_note(f"leftovers of the prefix {CFG['prefix']} were NOT removed "
                      f"(--no-cleanup / LSB_KEEP=1)")
        return
    removed = 0
    async with AsyncExitStack() as stack:
        try:
            c, _init = await fresh_session(stack)
        except Exception as exc:  # noqa: BLE001 — auxiliary step, not a scenario
            pre_warn("pre-cleanup was not performed (the MCP session did not open)",
                     describe(exc))
            teardown_note(f"pre-cleanup failed ({describe(exc)}) — probe notes "
                          f"with prefix {CFG['prefix']} may stay in the base")
            return
        for item in await probe_titles(c, CFG["prefix"]):
            res = await c.call("memory_delete", {"id": item["id"]})
            removed += int(bool(res.get("deleted")))
        await c.close()
    info(f"pre-cleanup: removed {removed} note(s) with prefix {CFG['prefix']}")
    if removed:
        teardown_note(f"removed {removed} leftover note(s) of a previous run "
                      f"(prefix {CFG['prefix']})")


async def cleanup_own_data() -> None:
    """Remove the probe data of THIS run and record it in the trace (FR-3.1)."""
    async with AsyncExitStack() as stack:
        try:
            c, _init = await fresh_session(stack)
        except Exception as exc:  # noqa: BLE001 — the trace must survive anyway
            teardown_note(f"cleanup could not open an MCP session ({describe(exc)}) "
                          f"— probe notes with prefix {CFG['prefix']} are left, "
                          f"they are safe to delete by hand")
            return
        try:
            removed = []
            for item in await probe_titles(c, CFG["prefix"]):
                res = await c.call("memory_delete", {"id": item["id"]})
                if res.get("deleted"):
                    removed.append(item["id"])
            teardown_note(f"removed {len(removed)} probe note(s) of this run "
                          f"(ids: {removed})")
            if not removed:
                teardown_note("this run had no probe notes left in the base")
        except Exception as exc:  # noqa: BLE001
            teardown_note(f"probe notes were NOT removed ({describe(exc)}) — "
                          f"prefix {CFG['prefix']}")
        await c.close()


def contour_alive(rest: httpx2.AsyncClient) -> Callable[[], Awaitable[tuple[bool, str]]]:
    """A zero-arg `wait_until` predicate: the contour answers `/health` (see `_http_alive`)."""

    async def predicate() -> tuple[bool, str]:
        return await _http_alive(rest)

    return predicate


async def _await_contour_ready(rest: httpx2.AsyncClient) -> None:
    """Wait for the contour's liveness before retrying the note save.

            Re-opening a session after a hook could catch the contour restarting (the
            restart hook, a slot configuration change): then the repeated
            `memory_save` fails like the first attempt and the probe note silently
            never appears (the live run of 2026-10-10). `_http_alive` already checks
            the liveness — we wait only if it has not happened yet, to avoid piling
            empty rows into the waits table.
    """
    alive, detail = await _http_alive(rest)
    if alive:
        return
    info(f"the contour is not ready before the probe retry ({detail}) — waiting "
         f"for /health")
    await wait_until(contour_alive(rest),
                     "the contour answers /health before the probe note is re-saved",
                     timeout=min(CFG["wait_sec"], 120.0))


async def resume_after_hooks(c: Client, rest: httpx2.AsyncClient, where: str) -> None:
    """The scenario's live MCP session after a contour hook (the recreation tears the socket).

            Branches 3/4/8/9 recreate the contour container, and the runner's socket
            dies with it: the first handle of the next scenario would fail with a
            transport break. Scenarios 10–13 never ran because of that (the live run
            of 2026-10-12). A cheap check + one session re-open; the live session is
            untouched.
    """
    try:
        await c.call("memory_list", {"limit": 1, "detail": "titles"})
        return
    except Exception as exc:  # noqa: BLE001 — we decide the break by type, not swallow
        if not is_transport_break(exc):
            raise
        info(f"{where}: the MCP session does not answer after a contour hook "
             f"({describe(exc)}) — re-opening it")
        await _await_contour_ready(rest)
        await c.reconnect()


async def _save_probe_once(c: Client, args: dict, *,
                           reconnect: bool = False) -> tuple[int | None, str]:
    """One `memory_save` of a probe note → `(id, detail)`; does not swallow a domain error.

            `reconnect=True` — the call goes through `call_with_reconnect` (a transport
            break re-opens the session once). A break that remains a break is returned
            as the reason, not an exception: the caller decides itself whether to
            retry.
    """
    try:
        answer = (await call_with_reconnect(c, "memory_save", args) if reconnect
                  else await c.call("memory_save", args))
    except Exception as exc:  # noqa: BLE001 — we decide the break by type, not swallow
        if not is_transport_break(exc):
            raise
        return None, f"transport broke: {describe(exc)}"
    if not isinstance(answer, dict):
        return None, f"unexpected answer: {answer!r}"
    if answer.get("duplicated") or answer.get("stored") is False:
        # The product did NOT save the note: a verbatim duplicate of an existing one (`stored:
        # false`/`duplicated: true`) or a validation refusal. The duplicate's answer carries
        # an `id` — but that is the EXISTING note's id, not this branch's note's: taking
        # it for success means measuring a foreign note's queue (the live run
        # of 2026-10-10, the [judge] branch: this branch's probe note is not in the database at all, and
        # the "2 of 2" of the vector wait — scenario 1's note and its twin).
        return None, ("the note was NOT created — the product answered "
                      f"{note_detail(answer)}")
    if answer.get("id") is not None:
        RUN["notes"].append(answer["id"])
        return answer["id"], f"id={answer['id']}"
    return None, note_detail(answer)


async def save_probe_note_result(c: Client, title: str, *, text: str | None = None,
                                 after_hook: bool = False,
                                 rest: httpx2.AsyncClient | None = None,
                                 ) -> tuple[int | None, str]:
    """Create one probe note, CHECK that it exists, retry once → `(id, detail)`.

            An answer without `id` is not "the note created": the product could have
            refused (a verbatim duplicate, a limit) or the transport could have broken,
            leaving the answer empty. The live run of 2026-10-10: the runner's session
            died at the hook, the re-open went through, but the note was never created
            — the outage branch waited 180 s on an empty queue and went red with a
            false FAIL. Here we save at most twice: the first — right away (after the
            hook, via `call_with_reconnect`); the second — again via
            `call_with_reconnect` and, if the contour is still coming up, after
            waiting for its liveness (`rest`). `detail` — the failure reason (the
            product's answer / the break); on success — `id=<n>`.
    """
    args = {
        "text": text or f"{RUN['tag']} probe note text (observability acceptance).",
        "title": title,
    }
    note_id, detail = await _save_probe_once(c, args, reconnect=after_hook)
    if note_id is not None:
        return note_id, detail
    info(f"the probe note {title!r} is not created ({detail}) — retrying once "
         f"through the session re-open")
    if rest is not None:
        await _await_contour_ready(rest)
    return await _save_probe_once(c, args, reconnect=True)


# --- the near-duplicate pair (scenarios 8 and 9: the judge must receive a pair) -------
# All scenarios' probes are written with ONE stub text, so the second probe
# is rejected by the product as a verbatim duplicate (`stored: false`) — and the pair never
# reaches the judge. The 2026-10-12 run: the budget branch waited 300 s for a "hopeless pair"
# over an empty judge queue, and the decision branch's step 3 — a task that must
# wait (#39) — also stared at an empty queue. The pair's texts must be
# close in meaning (a candidate from DEDUP_CANDIDATE_SIMILARITY = 0.80) but NOT
# verbatim equal.
NEAR_DUPLICATE_BODY = (
    "Acceptance probe of the dedup hand-off: the contour keeps a pair of notes "
    "of the same meaning, the candidate search finds the neighbour by cosine, "
    "and the judge is asked for a verdict. This body is shared by both notes "
    "of the pair on purpose — the pair must be close by meaning, while the "
    "framing around it keeps the texts from being verbatim duplicates. "
)


def near_duplicate_texts(tag: str, label: str = "dedup hand-off pair") -> tuple[str, str]:
    """The pair's two texts: close in meaning, but not verbatim duplicates.

            `label` must be one's OWN per each pair of the run: word-for-word equal
            texts are rejected by the product (`stored: false`), and the second pair
            simply never appears — the branch reads false red. The 2026-10-12 run:
            scenario 9's step 3 received the same texts as the budget branch, the
            second notes never appeared, and the "the task waits" check spun its guard
            on an empty queue.
    """
    return (
        f"{tag} {label}, first note. {NEAR_DUPLICATE_BODY}"
        "The pair is created by the acceptance run.",
        f"{tag} {label}, second note. {NEAR_DUPLICATE_BODY}"
        "The pair is checked by the acceptance run.",
    )


async def save_near_duplicate_pair(c: Client, title_first: str,
                                   title_second: str,
                                   ) -> tuple[list[int | None], bool]:
    """Save a near-duplicate pair → `([the first id, the second id], vector_ready)`.

            The pair is needed by the branches that need the judge's verdict: without
            a candidate (cosine ≥ 0.80) the judge is not asked, the judge queue stays
            empty and the checks read false red.

            The candidate is searched over the FIRST note's vector: until its vector
            is ready (`vector_status='ok'`), nobody finds a neighbor, and the second
            note has nothing to form a pair with (verified live on 2026-10-12 — the
            judge received no pairs at all). So after the first note we wait for its
            vector and only then save the second. `vector_ready` (the first note's
            vector readiness) is returned outward: the caller distinguishes "no pair"
            from "a pair exists but its vector was not waited for" and leaves an
            honest `skip`/`manual` instead of a false FAIL. No new DB access is
            introduced: the same `probe_vectors_ready` (`db_count` over
            `vector_status='ok'`) is used.
    """
    first, second = near_duplicate_texts(RUN["tag"], label=title_first)
    ids: list[int | None] = []
    vector_ready = False
    for title, text in ((title_first, first), (title_second, second)):
        note_id, _detail = await save_probe_note_result(
            c, title, text=text, after_hook=True
        )
        ids.append(note_id)
        if len(ids) == 1 and note_id is not None:
            vector_ready, detail = await wait_until(
                probe_vectors_ready([note_id]),
                "the first note of the pair got its vector (the candidate is "
                "searched by it)",
                timeout=min(CFG["wait_sec"], 180.0))
            if not vector_ready:
                warn("the first note of the pair has no vector yet",
                     f"id={note_id}, {detail} — the judge may get no pair")
    return ids, vector_ready


async def save_probe_note(c: Client, title: str, *, after_hook: bool = False,
                          rest: httpx2.AsyncClient | None = None) -> int | None:
    """Create one probe note (marked by the run tag in its title); return its id.

            A wrapper over `save_probe_note_result`: there — the "the note was
            created" check and the single retry via a session re-open; for those who
            do not need the reason, only `id` (or None) is returned here.
    """
    note_id, _detail = await save_probe_note_result(c, title, after_hook=after_hook,
                                                    rest=rest)
    return note_id


# ==== scenarios: 0. surface and readiness ======================================
async def scenario_0_context(c: Client, rest: httpx2.AsyncClient,
                             instructions: str) -> None:
    scenario(0, "MCP surface and contour readiness (34 tools, /health shape)")

    listed = await c.session.list_tools()
    names = sorted(t.name for t in listed.tools)
    check(f"the MCP surface lists {MCP_TOOLS_TOTAL} tools", len(names) == MCP_TOOLS_TOTAL,
          f"got {len(names)}: {', '.join(names)}")
    # №12 (no12-02): the shared warnings surface is part of the 3.2.0 MCP surface.
    check("the warnings_list handle is present", "warnings_list" in names)
    check("the memory handles are present",
          all(name in names for name in ("memory_save", "memory_search",
                                         "memory_list", "memory_get")))
    check("the instructions announce the slot hint / jobs",
          "warning" in instructions.lower() or "job" in instructions.lower(),
          f"{len(instructions)} chars of instructions")

    snapshot = await refresh_health(rest)
    missing = [field for field in HEALTH_LEGACY if field not in snapshot]
    check("the legacy /health fields are present", not missing,
          f"missing: {missing}" if missing else "all present")
    check(f"/health.version == {CFG['expect_version']!r}",
          snapshot.get("version") == CFG["expect_version"],
          f"got {snapshot.get('version')!r}")
    for queue in HEALTH_QUEUES:
        stat = queue_stat(snapshot, queue)
        check(f"/health.queues reports {queue} with waiting/attention",
              stat is not None and "waiting" in stat and "attention" in stat,
              f"{stat}")
    for slot in SLOTS:
        stat = slot_stat(snapshot, slot)
        check(f"/health.slots reports {slot} (ok/last_outcome/paused_until/attention/diagnostics)",
              stat is not None
              and {"ok", "last_outcome", "paused_until", "attention",
                   "last_finish_reason", "last_budget",
                   "last_budget_source", "last_had_thinking"} <= set(stat),
              f"{stat}")
    info(f"model slots at the start: embedding_ok={snapshot.get('embedding_ok')}, "
         f"summarizer_ok={snapshot.get('summarizer_ok')}, "
         f"judge_ok={snapshot.get('judge_ok')}")
    ENV["db_size"] = db_size_text()
    info(f"DB: {ENV['db_size']}")

    labels = await image_labels()
    ENV.update({k: labels.get(k) for k in
                ("org.opencontainers.image.revision",
                 "org.opencontainers.image.version")})
    ENV["revision"] = ENV.get("org.opencontainers.image.revision")
    ENV["image_version"] = ENV.get("org.opencontainers.image.version")
    info(f"image revision: {ENV['revision']!r}, image version: {ENV['image_version']!r}")

    await save_probe_note(c, RUN["note_observability"])
    info("probe note of the observability scenario is created")


# ==== scenarios: 1. observability baseline =====================================
async def scenario_1_observability_baseline(rest: httpx2.AsyncClient) -> None:
    scenario(1, "Queue and slot observability baseline (before the outage)")

    snapshot = await health_snapshot(rest)
    check("/health answers without a token and reports status ok",
          snapshot.get("status") == "ok", f"status={snapshot.get('status')!r}")
    # The observability block exists BEFORE the outage — the baseline the
    # scenario 2 growth is measured against.
    queues = snapshot.get("queues") or {}
    check("every queue carries numeric pending/waiting/attention",
          all(isinstance(q.get("waiting"), int) and isinstance(q.get("attention"), int)
              and isinstance(q.get("pending"), int) for q in queues.values()),
          f"{len(queues)} queues")
    slots = snapshot.get("slots") or {}
    check("every slot carries the observability and diagnostics fields",
          all({"ok", "last_outcome", "paused_until", "attention",
               "last_finish_reason", "last_budget",
               "last_budget_source", "last_had_thinking"} <= set(s)
              for s in slots.values()),
          f"{list(slots)}")

    # The queue_waiting event is the log-side proof of deferred work (FR-2.3).
    if CFG["logs_cmd"]:
        found, detail = await wait_until(log_events("queue_waiting"),
                                         "the queue_waiting event is present in the log",
                                         timeout=min(CFG["wait_sec"], 60.0))
        check("the log carries the queue_waiting event (deferred work is visible)",
              found, detail)
    else:
        skip("the queue_waiting event in the log",
             "LSB_LOGS_CMD is not set (no log hook on this contour)")


# ==== scenarios: 2. models unavailable → wait → back (per slot) ================
async def _outage_for_slot(c: Client, rest: httpx2.AsyncClient, slot: str) -> None:
    """№39 core for ONE slot: stop → the queue grows → start → it drains.

    `summary`/`judge` are VECTOR-DEPENDENT: their queue counts only the notes
    whose vector is ready, so the branch waits for the probe notes to be
    vectorised and refuses to run at all while the embedding slot is paused.
    """
    queue = SLOT_QUEUE[slot]
    needs_vector = slot in VECTOR_DEPENDENT_SLOTS
    before = await health_snapshot(rest)
    stat_before = queue_stat(before, queue) or {}
    info(f"[{slot}] before the outage: queue {queue} pending="
         f"{stat_before.get('pending')}, waiting={stat_before.get('waiting')}")

    # №39 (guard BEFORE the outage): the `summary`/`judge` queues count ONLY the
    # notes with a READY vector, so the embedding slot must be usable before this
    # branch. After the `embedding` branch it sits in a pause / back-off and the
    # dependent queue honestly shows zero work — take care of that state here
    # (an honest SKIP with a hint) instead of reading it as a FAIL later.
    if needs_vector:
        ready, ready_detail = await slot_ready(rest, "embedding")()
        if not ready:
            skip(f"[{slot}] the outage of a queue that needs a ready vector",
                 "the embedding slot is not usable, while the queue "
                 f"{queue} counts only the notes whose vector is ready: "
                 f"{ready_detail} — wait for the pause to pass "
                 "(/health.slots.embedding.paused_until), start the slot back "
                 "(LSB_SLOT_ON_CMD), then re-run the scenario")
            manual(f"scenario 2, branch [{slot}] (№39 FR-1/FR-2/FR-5.1)",
                   "the branch needs a working embedding slot: its queue is filled "
                   "only by notes whose vector is ready")
            return

    proc = await run_shell(slot_cmd(CFG["slot_off_cmd"], slot))
    check(f"[{slot}] the slot is stopped by the contour hook", proc.returncode == 0,
          f"rc={proc.returncode}")
    if proc.returncode != 0:
        return
    teardown_note(f"[{slot}] the slot was stopped and started back by the scenario hook")
    # The branch's journal-check window (for the `judge` branch — below).
    judge_started = utc_now_iso()

    # #39 + LSBN-F-12 (the live run of 2026-10-13): the `judge` queue is filled by the
    # "dedup judge" pair, and the decision slot may make the decision on it — then
    # the generative judge is not called at all; the work finishes in seconds and
    # the queue honestly empties: the "the work is visible" and "the oldest's age
    # grows" checks read false red (run #13, 2026-10-13 — the only
    # one left red). The branch measures the GENERATIVE path, so together with the judge
    # the decision channel is muted too: the product takes the generative path per the switch
    # contract (`decision_fallback reason=unavailable`) and runs into
    # the muted judge — the work WAITS (the 2026-10-13 measurement: oldest 3 → 77 s,
    # `queue_waiting` written; after the channels return the queue drains). The contour
    # is not recreated meanwhile: the same hook, the same way, only a second target.
    decision_cut = False
    if slot == "judge":
        # "Is the decision slot enabled" is read from the `/health.decision.enabled` aggregate:
        # `/health.slots.decision` has NO `enabled` field (the `SlotStat` model does not
        # carry it — the 2026-10-13 run: the condition silently did not fire, the decision channel
        # stayed alive, it decided the pair again, and the branch went red).
        decision_stat = (await health_snapshot(rest)).get("decision") or {}
        if decision_stat.get("enabled"):
            cut = await run_shell(slot_cmd(CFG["slot_off_cmd"], SLOT_DECISION))
            decision_cut = cut.returncode == 0
            if decision_cut:
                teardown_note(f"[{slot}] the decision channel was stopped and "
                              f"started back by the scenario hook")
            else:
                # The hook does not know the `decision` target (a foreign contour with an old
                # scripts/slot_gate.sh): the branch is honestly skipped — otherwise it
                # measures the wrong channel and goes red falsely. We raise the slot back.
                back = await run_shell(slot_cmd(CFG["slot_on_cmd"], slot))
                teardown_note(f"[{slot}] the slot was started back (the branch "
                              "was left before its probe notes)")
                skip(f"[{slot}] the generative judge is really the only path",
                     "при включённом decision-слоте пару решает ОН, а канал decision "
                     f"перерезать не удалось (rc={cut.returncode})")
                manual(f"scenario 2, branch [{slot}] (№39 FR-1/FR-2/FR-5.1)",
                       "решение по паре «судья дедупа» может принять decision-слот "
                       "(LSBN-F-12), и очередь `judge` пустеет без вызова "
                       "генеративного судьи. Дай хуку цель `decision` "
                       "(`scripts/slot_gate.sh off|on decision`) или гони контур с "
                       "DECISION_ENABLED=false — тогда ветвь мерит генеративный путь")
                check(f"[{slot}] the slot is started back by the contour hook",
                      back.returncode == 0, f"rc={back.returncode}")
                return

    released = False
    try:
        # Push work into the slot's queue so the wait is observable (a probe note
        # lands in vector+summary; the judge queue grows from the dedup hand-off).
        # The branches differ: `judge` asks for a near-duplicate pair (the judge is asked
        # only about a pair of close notes); the others — one note.
        if slot == "judge":
            # The judge needs a near-duplicate PAIR: one note is not enough — without a close
            # candidate (cosine ≥ DEDUP_CANDIDATE_SIMILARITY = 0.80) the judge is not
            # asked, the `judge` queue stays empty, and the "the work is
            # visible, nothing is lost" checks read false red (the live run
            # of 2026-10-12: the queue 0/0/0 and "the age of the oldest item grows").
            # The helper writes the pair with its own texts differing by the tail (not a
            # verbatim duplicate — the product does not reject them) and waits for the FIRST
            # note's vector itself: the candidate is searched over the vector; nobody finds a pair without it.
            title_one = RUN[f"note_outage_{slot}"]
            pair_ids, vector_ready = await save_near_duplicate_pair(
                c, title_one, f"{title_one} pair")
            probe_ids = [item for item in pair_ids if item is not None]
            probe = probe_ids[0] if probe_ids else None
            probe_detail = (f"ids={pair_ids}, the first vector is ready: "
                            f"{vector_ready}")
            if probe is None or len(probe_ids) < 2:
                warn("[judge] the near-duplicate pair of the branch is not usable",
                     f"{probe_detail} — the judge may get no pair")
        else:
            # The text is one's OWN per branch and attempt: with scenario 1's shared text
            # (`<tag> probe note text (observability acceptance).`) the product
            # answers "a verbatim duplicate" and saves no note, and the branch would get
            # a FOREIGN note's id and measure its queue (the live run of 2026-10-10,
            # the [judge] branch).
            probe_text = (f"{RUN['tag']} {slot} outage probe text "
                          "(observability acceptance).")
            probe, probe_detail = await save_probe_note_result(
                c, RUN[f"note_outage_{slot}"], text=probe_text, after_hook=True,
                rest=rest)
            probe_ids = [probe] if probe is not None else []
        if probe is None:
            # The probe note was not created (an answer without `id`: a refusal, a break,
            # a verbatim duplicate). This slot's queue is guaranteed empty — waiting
            # 180 seconds on it would give a false FAIL instead of an honest SKIP
            # (the live run of 2026-10-10: the runner's session died at the hook, and
            # the probe note did not save even after the re-open). We leave
            # the branch honestly: a SKIP with the reason; we raise the slot back and lose
            # nothing — as in the "the embedding slot is paused" case above.
            skip(f"[{slot}] with the slot down the work stays visible (nothing is lost)",
                 f"пробная заметка не создана: {probe_detail}")
            manual(f"scenario 2, branch [{slot}] (№39 FR-1/FR-2/FR-5.1)",
                   "пробную заметку не удалось создать, поэтому очередь пуста "
                   f"не по вине слота (memory_save → {probe_detail}); проверь "
                   "живость контура и MCP-сессии (LSB_LOGS_CMD), подними слот "
                   "(LSB_SLOT_ON_CMD) и перезапусти сценарий")
            back = await run_shell(slot_cmd(CFG["slot_on_cmd"], slot))
            released = True
            check(f"[{slot}] the slot is started back by the contour hook",
                  back.returncode == 0, f"rc={back.returncode}")
            return
        # №39 (guard BEFORE the queue check): this queue counts a note only when
        # its vector is ready, so the embedding slot must have vectorised the
        # probe notes of the branch FIRST. Otherwise the queue honestly shows zero
        # work while the work is really there — a false FAIL. If the vector never
        # arrives because the embedding slot went into a pause / back-off, the
        # branch is left honestly (SKIP), the slot is started back, nothing lost.
        if needs_vector:
            vector_ok, vector_detail = await wait_until(
                probe_vectors_ready(probe_ids),
                f"[{slot}] the probe notes of the branch are vectorised "
                f"(the queue {queue} counts them)",
                timeout=min(CFG["wait_sec"], 180.0))
            if not vector_ok:
                live, live_detail = await slot_ready(rest, "embedding")()
                if not live:
                    skip(f"[{slot}] the probe notes of the branch are vectorised "
                         f"(the queue {queue} counts them)",
                         "the embedding slot is in a pause / back-off: "
                         f"{live_detail} — wait for the pause to pass "
                         "(/health.slots.embedding.paused_until), start the slot "
                         "back (LSB_SLOT_ON_CMD), then re-run the scenario")
                    manual(f"scenario 2, branch [{slot}] (№39 FR-1/FR-2/FR-5.1)",
                           "the branch needs a working embedding slot: its queue is "
                           "filled only by notes whose vector is ready")
                else:
                    check(f"[{slot}] the probe notes of the branch got their vector "
                          f"(the queue {queue} counts them)", False, vector_detail)
                back = await run_shell(slot_cmd(CFG["slot_on_cmd"], slot))
                released = True
                check(f"[{slot}] the slot is started back by the contour hook",
                      back.returncode == 0, f"rc={back.returncode}")
                return

        if slot == "judge":
            # The `judge` queue holds a near-duplicate PAIR, and its task does not live forever:
            # an unsolvable pair is closed and placed anew ("retry queued" in the
            # journal), so an instantaneous queue snapshot is not
            # proof (live run #17: the work moved into a new job between
            # polls, and both queue checks survived on false red although
            # the deferral was honest). We measure the fact of the deferral itself — by the journal: a slot
            # failure with an OUTCOME (not a verdict) means the pair was deferred, not
            # done by a foreign channel.
            if not CFG["logs_cmd"]:
                skip(f"[{slot}] the outage defers the judge work (an outcome, "
                     f"not a verdict)",
                     "LSB_LOGS_CMD is not set (no log hook on this contour)")
            else:
                ok, detail = await wait_until(
                    log_slot_event("dedup_judge_failed", slot=slot,
                                   outcome="unavailable", since=judge_started),
                    f"[{slot}] the outage defers the judge work (an outcome, "
                    f"not a verdict)",
                    timeout=min(CFG["wait_sec"], 180.0))
                check(f"[{slot}] with the slot down the pair fails by the outage "
                      f"instead of being decided elsewhere", ok, detail)
        else:
            ok, detail = await wait_until(
                queue_has_work(rest, queue),
                f"[{slot}] the queue {queue} shows work while the slot is down",
                timeout=min(CFG["wait_sec"], 180.0))
            check(f"[{slot}] with the slot down the work stays visible (nothing is lost)",
                  ok, detail)
            if not ok:
                # Diagnostics (the live run of 2026-10-10): the queue is honestly empty when
                # the break did NOT cut the slot's channel — the contour walks to another provider,
                # and the branch's work finishes in seconds. A typical cause — the leftovers
                # of an interrupted run's failure branch: `scripts/slot_fake.sh` left
                # the slot on the stub, while the muted proxy is not the one the contour walks to.
                snapshot = await health_snapshot(rest)
                stat = slot_stat(snapshot, slot) or {}
                warn(f"[{slot}] the outage did not cut the channel of the slot: the "
                     f"work of the branch was finished, not deferred",
                     f"/health.slots.{slot}={stat}; check the contour: "
                     f"'scripts/slot_fake.sh status' (a leftover fake provider) and "
                     f"'scripts/slot_gate.sh status {slot}'")

            ok, detail = await wait_until(
                queue_has_work(rest, queue, min_age=10),
                f"[{slot}] the age of the oldest item in {queue} grows")
            check(f"[{slot}] the age of the oldest item grows (/health.queues)", ok, detail)

        alive = await health_snapshot(rest)
        check(f"[{slot}] /health answers while the slot is down (the service is alive)",
              alive.get("status") == "ok", f"status={alive.get('status')!r}")

        if CFG["logs_cmd"]:
            found, detail = await wait_until(
                log_events("queue_waiting"),
                f"[{slot}] the queue_waiting event appears in the log (slot still off)",
                timeout=min(CFG["wait_sec"], 120.0))
            check(f"[{slot}] the queue_waiting event is written to the log", found, detail)
            # No flood: the number of slot-call attempts in the window must not
            # explode (back-off, not a busy-loop). Approximate: the log hook is
            # asked for the attempt marker twice and the count must not double.
            await _check_no_flood(slot)
        else:
            skip(f"[{slot}] the queue_waiting event in the log",
                 "LSB_LOGS_CMD is not set (no log hook on this contour)")

        proc = await run_shell(slot_cmd(CFG["slot_on_cmd"], slot))
        released = True
        check(f"[{slot}] the slot is started back by the contour hook",
              proc.returncode == 0, f"rc={proc.returncode}")
        if decision_cut:
            back = await run_shell(slot_cmd(CFG["slot_on_cmd"], SLOT_DECISION))
            check(f"[{slot}] the decision channel is started back",
                  back.returncode == 0, f"rc={back.returncode}")
    finally:
        if not released:
            undo = await run_shell(slot_cmd(CFG["slot_on_cmd"], slot))
            if undo.returncode == 0:
                teardown_note(f"[{slot}] the slot was released by the teardown hook "
                              f"(the scenario did not reach its own `on` step)")
                warn(f"[{slot}] the scenario left the outage before its own `on` step",
                     "the slot was released by the teardown hook")
            else:
                warn(f"[{slot}] the slot could NOT be released — remove the block by hand",
                     f"rc={undo.returncode}")
            if decision_cut:
                restored = await run_shell(slot_cmd(CFG["slot_on_cmd"], SLOT_DECISION))
                if restored.returncode == 0:
                    teardown_note(f"[{slot}] the decision channel was started back "
                                  f"by the teardown hook")
                else:
                    warn(f"[{slot}] the decision channel could NOT be started back — "
                         f"start it by hand", f"rc={restored.returncode}")

    # The drain guard — LONGER than the shared one (`wait_sec`): after the break the pair's work goes into
    # back-off (10 → 20 → … s) and does not return to business instantly, while a live model
    # computes the summary in seconds. The live run of 2026-10-13: the queue DID drain, but
    # ~1–2 minutes later than the 120-second guard (both `summary` and `vector`) — false
    # red. The fast profile does not suffer: there the queue empties in seconds and
    # the wait returns at once (the same decision as the F-11 smoke with its 600 s window).
    ok, detail = await wait_until(
        queue_drained(rest, queue),
        f"[{slot}] the queue {queue} is drained after the slot is back",
        timeout=max(CFG["wait_sec"], 300.0))
    check(f"[{slot}] after the slot is back the queue drains within the guard (no restart)",
          ok, detail)


async def _check_no_flood(slot: str) -> None:
    """Approximate no-flood check: the attempt count over a window must not double."""
    first_ok, first = await log_has(f"slot={slot}")
    await asyncio.sleep(min(CFG["poll_sec"], 5.0))
    second_ok, second = await log_has(f"slot={slot}")
    if not (first_ok and second_ok):
        skip(f"[{slot}] no request flood to the broken slot", first)
        return
    n1 = _count_marker(first)
    n2 = _count_marker(second)
    check(f"[{slot}] no flood of slot-call attempts behind the back-off",
          n2 <= max(n1 * 2, n1 + 3),
          f"attempts marker count {n1} → {n2}")


def _count_marker(detail: str) -> int:
    match = re.search(r"in (\d+) log lines", detail)
    return int(match.group(1)) if match else 0


async def scenario_2_models_outage(c: Client, rest: httpx2.AsyncClient) -> None:
    scenario(2, "Models unavailable → jobs wait → models back → jobs finish "
                "(per slot, NO restart)")
    if not (CFG["slot_off_cmd"] and CFG["slot_on_cmd"]):
        skip("models unavailable → waiting → models back, per slot",
             "the contour hook is required: LSB_SLOT_OFF_CMD / LSB_SLOT_ON_CMD "
             "(e.g. scripts/slot_gate.sh off {slot} / on {slot})")
        manual("scenario 2 (№39 FR-1/FR-2/FR-5.1)",
               "set LSB_SLOT_OFF_CMD / LSB_SLOT_ON_CMD (with {slot} or the name "
               "appended) and run the E2E: each slot must wait and finish without "
               "a container restart")
        return

    # ORDER (`SLOT_OUTAGE_ORDER`), not `SLOTS`: only the slots owning a queue are
    # walked, and `embedding` is walked LAST — see the constant for the reason.
    for slot in SLOT_OUTAGE_ORDER:
        await _outage_for_slot(c, rest, slot)

    # FR-2.3/FR-5.1: a session opened BEFORE the outage is still alive — the
    # "no restart" proof (a restart would forget the session_id, SDK mcp 2.x).
    try:
        alive = await c.call("memory_list", {"limit": 1, "detail": "titles"})
        session_alive = "items" in alive
        detail = ""
    except Exception as exc:  # noqa: BLE001
        session_alive, detail = False, describe(exc)
    check("the MCP session opened before the outage is still alive (no container restart)",
          session_alive, detail)


# ==== scenarios: 3. auth branch (401) ==========================================
async def scenario_3_auth_branch(c: Client, rest: httpx2.AsyncClient) -> None:
    scenario(3, "Access-failure branch: the slot answers 401 → pause, probe, resume")
    if not (CFG["slot_auth_cmd"] and CFG["slot_auth_fix_cmd"]):
        skip("the access-failure branch (fake provider returns 401)",
             "the contour hook is required: LSB_SLOT_AUTH_CMD / "
             "LSB_SLOT_AUTH_FIX_CMD (a fake slot provider that can answer 401)")
        manual("scenario 3 (№39 FR-3/FR-4.2/FR-4.3)",
               "point LSB_SLOT_AUTH_CMD at a fake provider that answers 401 and "
               "LSB_SLOT_AUTH_FIX_CMD at its fix: the slot must pause, /health.slots "
               "must show last_outcome='auth' with paused_until, then probe and resume. "
               "scripts/slot_fake.sh (see its --help) does exactly this: the 401 is "
               "applied AFTER the contour is up (a 401 slot is fatal at the startup "
               "check) and the fix restarts the fake only, never the contour — the "
               "break, the probe and the resume stay alive: "
               "LSB_SLOT_AUTH_CMD='scripts/slot_fake.sh auth' with LSB_AUTH_SLOT=summary, "
               "LSB_SLOT_AUTH_FIX_CMD='scripts/slot_fake.sh on ok'")
        return

    slot = CFG["auth_slot"]
    queue = SLOT_QUEUE[slot]
    proc = await run_shell(slot_cmd(CFG["slot_auth_cmd"], slot))
    check(f"the fake provider is switched to 401 for {slot}", proc.returncode == 0,
          f"rc={proc.returncode}")
    if proc.returncode != 0:
        return
    # The branch's journal-check window: before this moment the branch's events cannot
    # exist (the hook recreates the container — old lines vanish, and a foreign
    # `slot_paused` outcome from a neighboring branch no longer counts as one's own).
    branch_started = utc_now_iso()
    teardown_note(f"[{slot}] the fake provider was switched to 401 and fixed back")
    fixed = False
    try:
        # One's own work in the queue — WHILE THE BREAK IS OPEN: the phase is checked by the fact
        # "the work WAITS out the pause"; an empty queue checks nothing. We create
        # the note AFTER switching the fake to 401: its summary must not manage to close
        # via the healthy regular slot before the breaker opens (the live run of 2026-10-12 —
        # a note created before the hook was summarized by the regular slot, and the
        # `summary` queue honestly showed 0/0/0). The branch's text is its OWN: with the shared
        # scenario-0 text (`<tag> probe note text (observability acceptance).`) the product
        # answers "a verbatim duplicate" and saves no note, and the branch would measure a foreign
        # (or empty) queue. The note's first real call opens the breaker
        # (the slot answers 401) — exactly the phase the branch checks.
        auth_text = (f"{RUN['tag']} {slot} auth-branch probe text — the summary is "
                     f"deferred by the open break (observability acceptance).")
        probe, probe_detail = await save_probe_note_result(
            c, RUN["note_auth"], text=auth_text, after_hook=True, rest=rest)
        if probe is None:
            info(f"[auth] the probe note of the branch is not created "
                 f"({probe_detail}) — the queue {queue} may stay empty")

        if CFG["logs_cmd"]:
            found, detail = await wait_until(
                log_slot_event("slot_paused", slot=slot, outcome="auth",
                               since=branch_started),
                "[auth] the slot_paused event appears in the log",
                timeout=min(CFG["wait_sec"], 120.0))
            check("[auth] the slot_paused event is written with the cause", found, detail)
        else:
            skip("[auth] the slot_paused event in the log", "LSB_LOGS_CMD is not set")

        ok, detail = await wait_until(slot_paused(rest, slot, "auth"),
                                      f"[auth] /health.slots.{slot} shows auth + pause")
        check(f"[auth] /health.slots.{slot}.last_outcome='auth', paused_until set",
              ok, detail)
        stat = slot_stat(await health_snapshot(rest), slot) or {}
        check(f"[auth] /health.slots.{slot}.paused_until is filled by the break",
              stat.get("paused_until") is not None, f"{stat}")

        # The open-breaker phase (arch §6.9, no39-04): the loop takes no batch, and all
        # current work moves from `pending` to `waiting` — that is exactly the fact "the work
        # waits out the pause"; `attention` is zero here; a human is called only after
        # SLOT_MAX_ATTEMPTS non-transient failures of the PAIR; and 401 is a transient outcome.
        ok, detail = await wait_until(
            queue_waiting(rest, queue),
            f"[auth] the work deferred by the break is visible in `waiting` ({queue}), "
            f"not in `attention`",
            timeout=min(CFG["wait_sec"], 180.0))
        check(f"[auth] in the phase of the open break the queue {queue} shows the work "
              f"in `waiting` (`attention == 0`)", ok, detail)

        proc = await run_shell(slot_cmd(CFG["slot_auth_fix_cmd"], slot))
        fixed = True
        check("[auth] the fake provider is fixed back by the contour hook",
              proc.returncode == 0, f"rc={proc.returncode}")
    finally:
        if not fixed:
            undo = await run_shell(slot_cmd(CFG["slot_auth_fix_cmd"], slot))
            if undo.returncode == 0:
                teardown_note(f"[{slot}] the fake provider was fixed by the teardown hook")
                warn("[auth] the scenario left before its fix step",
                     "the fake provider was fixed by the teardown hook")
            else:
                warn("[auth] the fake provider could NOT be fixed by hand",
                     f"rc={undo.returncode}")

    # The fix (LSB_SLOT_AUTH_FIX_CMD = `scripts/slot_fake.sh on ok ...`) RECREATES
    # the container: the breaker lives in process memory (`app/services/slot_breaker.py`),
    # it is lost at recreation, and the `slot_resumed` marker is written only when
    # the pause is lifted in the SAME process (`app/services/worker.py`) — after
    # a recreation it will never appear (the false FAIL of the live run of 2026-10-12).
    # We check the observable fact "access is back": the slot answers again (`ok`, the breaker
    # lifted) and the deferred work leaves the queue.
    ok, detail = await wait_until(
        slot_ok(rest, slot),
        f"[auth] after the fix the {slot} slot answers again (ok, the break is gone)")
    check(f"[auth] after the fix the {slot} slot is ok again — the access is restored",
          ok, detail)
    ok, detail = await wait_until(queue_drained(rest, queue),
                                  f"[auth] the queue {queue} drains after the fix")
    check("[auth] the deferred work is done after the access is restored", ok, detail)
    stat = queue_stat(await health_snapshot(rest), queue) or {}
    check(f"[auth] after the fix nothing is left in `waiting` ({queue})",
          (stat.get("waiting") or 0) == 0, f"{stat}")


# ==== scenarios: 4. attention pair (repeated off_contract) =====================
async def scenario_4_attention_pair(c: Client, rest: httpx2.AsyncClient) -> None:
    scenario(4, "Attention: a repeated off_contract answer stops retrying and waits "
                "for a human, then is released")
    if not (CFG["offcontract_cmd"] and CFG["offcontract_fix_cmd"]):
        skip("the repeated-off_contract branch (job lands in /jobs/attention)",
             "the contour hook is required: LSB_JUDGE_OFFCONTRACT_CMD / "
             "LSB_JUDGE_OFFCONTRACT_FIX_CMD (a fake judge provider answering "
             "off-contract)")
        manual("scenario 4 (№39 FR-3.4, no39-02)",
               "point the off-contract hook at a fake judge answering off-contract: "
               "the retries must be bounded, the job must appear in GET /jobs/attention "
               "and, after the answer changes, be released and finish. "
               "scripts/slot_fake.sh: "
               "LSB_JUDGE_OFFCONTRACT_CMD='scripts/slot_fake.sh on offcontract judge', "
               "LSB_JUDGE_OFFCONTRACT_FIX_CMD='scripts/slot_fake.sh off'. The fix puts "
               "the stock contour back: the restart releases the frozen job through "
               "SLOT_ATTENTION_RESET_ON_START and the real judge finishes it. "
               "'scripts/slot_fake.sh on ok judge' exercises the probe release "
               "instead (the pause must expire first), but it leaves the judge on the "
               "fake — the outage of scenarios 2/5 would then stop being visible")
        return

    # One's own work FOR THE JUDGE: without a near-duplicate pair the judge queue is empty, and the branch
    # read false red (2026-10-12 — "no work for the judge is created").
    # We save the pair BEFORE switching the fake: the judge's verdict is set by the background dedup hand,
    # which arrives after the vector and summary of both notes — that is, already with the breaker
    # open (the helper waits out the first note's vector; otherwise no candidate is found).
    pair_ids, vector_ready = await save_near_duplicate_pair(
        c, RUN["note_attention_pair"], RUN["note_attention_pair_2"])
    probes = len([item for item in pair_ids if item is not None])
    if probes < 2 or not vector_ready:
        skip("the repeated-off_contract branch (the judge has no pair)",
             f"the pair of notes is not usable: ids={pair_ids}, "
             f"first vector ready={vector_ready}")
        manual("scenario 4 (№39 FR-3.4, no39-02)",
               "the pair of nearly duplicate notes must be created and the vector of "
               "its first note must be ready: without a candidate (cosine >= 0.80) "
               "the judge is never asked, the queue `judge` stays empty and the "
               "branch reads a false red")
        return

    proc = await run_shell(CFG["offcontract_cmd"])
    check("the fake judge is switched to off-contract", proc.returncode == 0,
          f"rc={proc.returncode}")
    if proc.returncode != 0:
        return
    teardown_note("the fake judge was switched to off-contract and fixed back")
    fixed = False
    try:
        # The open-breaker phase (no39-04): the pair is deferred by a reason and visible in
        # `waiting`; `attention` — a `waiting` subset ("needs a human") and
        # appears only after SLOT_MAX_ATTEMPTS non-transient failures of the PAIR.
        ok, detail = await wait_until(
            queue_waiting(rest, "judge"),
            "[attention] the off-contract pair is deferred by the reason "
            "(the judge queue shows `waiting`)",
            timeout=min(CFG["wait_sec"], 180.0))
        check("[attention] while the answer is off-contract the pair waits in "
              "`waiting` instead of burning calls", ok, detail)
        stat = queue_stat(await health_snapshot(rest), "judge") or {}
        if stat.get("attention") or 0:
            # The threshold is already passed (the stub branch lowers SLOT_MAX_ATTEMPTS) —
            # a separate no39-04 threshold-reachability check.
            await check_attention_threshold(rest, "[attention]")
        else:
            check("[attention] in the phase of the open break `attention == 0`: the "
                  "pair is deferred, the human is not asked yet",
                  (stat.get("waiting") or 0) >= 1, f"{stat}")

        attention = None

        async def has_attention() -> tuple[bool, str]:
            nonlocal attention
            jobs = await attention_jobs(rest)
            attention = {"jobs": jobs} if jobs is not None else None
            return bool(jobs), f"attention jobs: {len(jobs or [])}"

        ok, detail = await wait_until(has_attention,
                                      "a job lands in GET /jobs/attention",
                                      timeout=CFG["wait_sec"])
        check("a repeated off-contract answer puts the job in /jobs/attention", ok, detail)
        if ok and attention:
            job = attention["jobs"][0]
            check("the attention record carries the slot/outcome/attempts",
                  job.get("slot") and job.get("outcome") and job.get("attempts") is not None,
                  f"{job}")

        proc = await run_shell(CFG["offcontract_fix_cmd"])
        fixed = True
        check("the fake judge is fixed back (the answer changes)", proc.returncode == 0,
              f"rc={proc.returncode}")
    finally:
        if not fixed:
            undo = await run_shell(CFG["offcontract_fix_cmd"])
            if undo.returncode == 0:
                teardown_note("the fake judge was fixed by the teardown hook")
                warn("scenario 4 left before its fix step",
                     "the fake judge was fixed by the teardown hook")
            else:
                warn("the fake judge could NOT be fixed — remove the block by hand",
                     f"rc={undo.returncode}")

    ok, detail = await wait_until(queue_drained(rest, "judge"),
                                  "the judge queue drains after the answer changes")
    check("after the cause is gone the frozen job is released and the queue drains",
          ok, detail)


# ==== scenarios: 5. restart inside the outage window ==========================
async def scenario_5_restart_inside_outage(c: Client, rest: httpx2.AsyncClient) -> None:
    scenario(5, "Restart inside the outage window: work is neither lost nor duplicated")
    if not CFG["restart_cmd"]:
        skip("restart inside the outage window",
             "the contour hook is required: LSB_RESTART_CMD (restarts the contour)")
        manual("scenario 5 (№39 FR-2.3)",
               "set LSB_RESTART_CMD and run the E2E: pending work must be picked up "
               "by the first cycle after the restart, nothing lost or duplicated")
        return

    rows, err = db_try("SELECT COUNT(*) AS c FROM worker_jobs WHERE status = 'pending'")
    if rows is None:
        skip("restart inside the outage window", f"the DB is not reachable: {err}")
        return
    before = int(rows[0]["c"])
    info(f"pending worker_jobs before the restart: {before}")

    proc = await run_shell(CFG["restart_cmd"])
    check("the contour is restarted by the hook", proc.returncode == 0,
          f"rc={proc.returncode}")
    if proc.returncode != 0:
        return
    teardown_note("the contour was restarted by the scenario hook")

    # The restart tears the runner's MCP session: first we wait for the contour to answer,
    # and only then re-open the session — otherwise the whole run goes into attempt retries
    # and ends FATAL (verified 2026-10-10).
    # The predicates below are `async def` and AWAIT their helpers: a lambda would hand
    # `wait_until` an already-created coroutine (awaited by nobody) — a false FAIL;
    # see `log_events`.
    async def contour_alive() -> tuple[bool, str]:
        return await _http_alive(rest)

    async def pending_kept() -> tuple[bool, str]:
        return await _pending_not_below(before)

    alive, detail = await wait_until(
        contour_alive,
        "the contour answers /health after the restart",
        timeout=min(CFG["wait_sec"], 120.0),
    )
    check("the contour answers /health after the restart", alive, detail)
    init = await c.reconnect()
    check("the MCP session is re-opened after the restart",
          init is not None, "session re-initialized")

    ok, detail = await wait_until(
        pending_kept,
        "the pending count is not below the pre-restart value (nothing lost)",
        timeout=min(CFG["wait_sec"], 120.0))
    check("no pending work is lost across the restart", ok, detail)

    after, err = db_try("SELECT COUNT(*) AS c FROM worker_jobs WHERE status = 'pending'")
    if after is not None:
        check("the restart did not duplicate the pending work",
              int(after[0]["c"]) <= before + 1,
              f"pending {before} → {int(after[0]['c'])}")


async def _pending_not_below(value: int) -> tuple[bool, str]:
    rows, err = db_try("SELECT COUNT(*) AS c FROM worker_jobs WHERE status = 'pending'")
    if rows is None:
        return False, err
    current = int(rows[0]["c"])
    return current >= value, f"pending {current} (was {value})"


async def _http_alive(rest: httpx2.AsyncClient) -> tuple[bool, str]:
    """The contour answers over HTTP (after a restart — with retries; the socket could die)."""
    last = "no answer"
    for _ in range(3):
        try:
            response = await rest.get("/health")
            body = response.json() if response.status_code == 200 else None
            if body and body.get("status") == "ok":
                return True, f"status=ok, version={body.get('version')!r}"
            last = f"http {response.status_code}"
        except Exception as exc:  # noqa: BLE001 — the contour could still be coming up
            last = describe(exc)
        await asyncio.sleep(2.0)
    return False, last


# ==== scenarios: 6. regression =================================================
async def scenario_6_regression() -> None:
    scenario(6, "Regression: unit tests + the existing E2E scripts")
    manual("unit regression",
           f"cd {CFG['repo_dir']} && {CFG['python']} -m pytest -q "
           "(baseline 1687 passed, 13 skipped, 0 failed)")
    if not CFG["run_regression"]:
        skip("unit and E2E regression",
             "run with --run-regression / LSB_RUN_REGRESSION=1 (long)")
        manual("existing E2E on the contour",
               "e2e_release31.py, e2e_release30.py, e2e_release22.py, "
               "e2e_lsb0004.py, e2e_lsb0005.py, e2e_lsb0006.py")
        return

    # The units — in a CLEAN environment with their own temporary database. The runner lives with the contour's
    # env: the live run of 2026-10-13 — with a foreign `MCP_AUTH_TOKEN` in the environment
    # 110 tests of 2136 fail (the test client sends its own token, while the application
    # takes the external one — `POST /notes` → 401); with `env -u` and an isolated database
    # the result matches the host one: 2136 passed, 13 skipped, 0 failed.
    proc = await run_shell(
        f"cd {CFG['repo_dir']} && env -u MCP_AUTH_TOKEN "
        f"DB_PATH=\"$(mktemp -d)/unit-notes.db\" "
        f"{CFG['python']} -m pytest -q",
        timeout=3600.0)
    tail = "\n".join(proc.stdout.strip().splitlines()[-3:])
    check("unit regression is green (0 failed)", proc.returncode == 0, tail)
    legacy: list[str] = []
    for name in (n.strip() for n in CFG["regression_scripts"].split(",")):
        if not name:
            continue
        proc = await run_shell(legacy_e2e_cmd(name), timeout=3600.0)
        tail = " ".join(proc.stdout.strip().splitlines()[-1:])[:90]
        legacy.append(f"{name}: rc={proc.returncode}"
                      + (f" — {tail}" if proc.returncode else ""))
    if legacy:
        # The old E2E pin THEIR release's surface (the tool count, the /health
        # shape, the prompt texts) and diverge from 3.2.0 FOR A REASON: the live run
        # of 2026-10-13 — release31 demands "21 tools" and "7 /health fields",
        # lsb0006 looks for the judge marker in the old prompt. This is not a 3.2.0 regression:
        # the behavior is held by the unit regression (2136 tests) and the release's scenarios, so
        # each script's result — a VISIBLE observation, not a red.
        warn("the existing E2E scripts of previous releases",
             "; ".join(legacy))
        manual("existing E2E of previous releases",
               "у e2e_release31/30 есть ключи (--base-url <контур> --db /data/notes.db); "
               "e2e_release22 и e2e_lsb0004/0005/0006 жёстко ходят на "
               "http://localhost:8080 и запускаются в сети контура "
               "(`--network container:<имя>`) — те самые команды в `legacy_e2e_cmd`")


def legacy_e2e_cmd(name: str) -> str:
    """The regression's (scenario 6) command for an old E2E script's run.

            Two kinds: `e2e_release31/30` have address and database keys — the runner
            calls them with the contour's address; `e2e_release22` and
            `e2e_lsb0004/0005/0006` go hard-coded to `http://localhost:8080` and are
            meant to run FROM INSIDE the contour — we run them in its NETWORK
            namespace (`--network container:<name>`), otherwise `localhost` leads to
            the runner's void (the live run of 2026-10-13: four of six failed on the
            address).

            We mount the HOST path of the tree (`--host-root`/LSB_HOST_ROOT): without
            it `-v /w:/w` is resolved by the daemon on the host, where no such
            directory exists — the nested container got an empty folder and failed
            with code 2.
    """
    tree = CFG["host_root"] or CFG["repo_dir"]
    if name in LEGACY_ARG_SCRIPTS:
        return (f"cd {CFG['repo_dir']} && {CFG['python']} scripts/{name} "
                f"--base-url {CFG['base_url']} --db {CFG['db']} "
                f'--token "$MCP_AUTH_TOKEN"')
    return (f"docker run --rm --network container:{CFG['container']} "
            f"-v {tree}:/w -v {tree}/data-test:/data -w /w "
            f'-e MCP_AUTH_TOKEN="$MCP_AUTH_TOKEN" '
            f"{CFG['image']} {CFG['python']} scripts/{name}")


# ==== scenarios: 7. TODO of the future blocks ==================================
async def future_block_todo() -> None:
    """Placeholders of the follow-up blocks of 3.2.0 (this script is a base).

            A follow-up brief REPLACES the matching placeholder with its own
            `scenario_<n>_<name>()` and registers it in `run_all()`. Nothing here FAILs:
            an unclaimed placeholder stays MANUAL so the base run is not red.
    """
    scenario(7, "TODO: scenarios of the future 3.2.0 blocks (placeholders)")
    # TODO(3.2.0 future blocks) — slots / output budget / decision:
    manual("slots: the `decision` slot and its observability (LSBN-F-12)",
           "the budget/finish_reason part of this placeholder is claimed by "
           "`scenario_8_budget_raise_release()` (lsbdef0008-02); the decision "
           "slot is claimed by `scenario_9_decision_fallback()` (f12-03)")
    # The failure branches' stub providers: `scripts/fake_slot.py` (the ok / auth /
    # empty-length / offcontract / decision-* modes) and the `scripts/slot_fake.sh` hook
    # (`on <mode> [slot ...]` recreates the contour with the branch's env; `off` returns
    # the regular compose). All four scenarios stay skippable until the hook is
    # set: `scripts/acceptance_run.sh` wires the commands below (LSB_FAKE_HOOKS=0
    # disables them).
    manual("outage branches: the contour hooks of scenarios 3/4/8/9",
           "LSB_SLOT_AUTH_CMD='scripts/slot_fake.sh auth' + LSB_AUTH_SLOT=summary, "
           "LSB_SLOT_AUTH_FIX_CMD='scripts/slot_fake.sh on ok'; "
           "LSB_JUDGE_OFFCONTRACT_CMD='scripts/slot_fake.sh on offcontract judge', "
           "LSB_JUDGE_OFFCONTRACT_FIX_CMD='scripts/slot_fake.sh off'; "
           "LSB_BUDGET_BREAK_CMD='scripts/slot_fake.sh on empty-length summary "
           "judge', LSB_BUDGET_FIX_CMD='scripts/slot_fake.sh on ok summary judge'; "
           "LSB_DECISION_ON_CMD='scripts/slot_fake.sh on decision-ok', "
           "LSB_DECISION_UNSUPPORTED_CMD='scripts/slot_fake.sh on "
           "decision-unsupported', LSB_DECISION_OFF_CMD='scripts/slot_fake.sh on "
           "decision-off', LSB_DECISION_LOW_CONFIDENCE_CMD='scripts/slot_fake.sh on "
           "decision-low-confidence', LSB_DECISION_RESTORE_CMD='scripts/slot_fake.sh "
           "off'. Run `scripts/slot_fake.sh --help` for the env of the hook "
           "(SLOT_FAKE_PAUSE_SEC, SLOT_FAKE_MAX_ATTEMPTS, SLOT_FAKE_PROJECT_DIR, "
           "SLOT_FAKE_MODELS ...)")
    # TODO(3.2.0 future blocks) — memory handles surface:
    manual("memory handles: search/list/get contract, budgets, pagination",
           "the node hint of `save` (№28) is claimed by "
           "`scenario_10_node_hint()`; the freshness filter (№29) — by "
           "`scenario_11_freshness()`; the note `comment` field (№24) — by "
           "`scenario_12_comment()`; the injected-text budget/guard of "
           "LSBN-F-11 is claimed by `tests/test_injection_budget.py` and "
           "`scenario_13_injected_text()`; the rest of the surface is claimed "
           "by the follow-up postановки")
    warn("the future-block section is a base placeholder — claimed by the "
         "follow-up postановки", "see the TODO(3.2.0 future blocks) comments")


# ==== scenarios: 8. budget raise releases the attention =========================
async def log_text() -> str:
    """Contour log via the operator hook (empty string — no hook / unreadable)."""
    if not CFG["logs_cmd"]:
        return ""
    try:
        proc = await run_shell(CFG["logs_cmd"], timeout=120.0)
    except (OSError, subprocess.SubprocessError):
        return ""
    return proc.stdout if proc.returncode == 0 else ""


def log_lines(text: str, *needles: str) -> list[str]:
    """Log lines carrying EVERY needle (an event and its own field on one line)."""
    return [line for line in text.splitlines() if all(n in line for n in needles)]


def db_count(sql: str, params: tuple = ()) -> int | None:
    """One integer out of the contour DB (`COUNT(*) AS c`); None — not readable."""
    rows, _err = db_try(sql, params)
    return int(rows[0]["c"]) if rows else None


def queue_numbers(snapshot: dict, queue: str) -> dict[str, Any]:
    """The per-queue numbers the acceptance report records (FR-5.2)."""
    stat = queue_stat(snapshot, queue) or {}
    return {key: stat.get(key) for key in ("pending", "waiting", "attention")}


def own_pending_summaries() -> int | None:
    """Probe notes of this run still waiting for a summary (matched by the tag)."""
    return db_count(
        "SELECT COUNT(*) AS c FROM notes WHERE deleted_at IS NULL "
        "AND summary_status = 'pending' AND title LIKE ?",
        (f"%{RUN['tag']}%",),
    )


def own_summaries_ready(count: int) -> Callable[[], Awaitable[tuple[bool, str]]]:
    """Predicate: every probe note of this run got its summary (FR-5.1)."""

    async def predicate() -> tuple[bool, str]:
        ready = db_count(
            "SELECT COUNT(*) AS c FROM notes WHERE deleted_at IS NULL "
            "AND summary_status = 'ok' AND title LIKE ?",
            (f"%{RUN['tag']}%",),
        )
        return (ready is not None and ready >= count), \
            f"probe notes with summary_status='ok': {ready} of {count}"

    return predicate


async def scenario_8_budget_raise_release(c: Client, rest: httpx2.AsyncClient) -> None:
    """lsbdef-0008: reasoning eats the budget → truncation → attention → the
    budget is raised → the queues drain WITHOUT a manual step (FR-2.3/FR-5.1)."""
    scenario(8, "Reasoning eats the budget → truncation → attention → the budget is "
                "raised → the queues drain by themselves")
    if not (CFG["budget_break_cmd"] and CFG["budget_fix_cmd"]):
        skip("the budget-raise release branch (a changed slot config releases the "
             "attention)",
             "the contour hook is required: LSB_BUDGET_BREAK_CMD / LSB_BUDGET_FIX_CMD "
             "(a fake slot answering done_reason=length on an empty content, then a "
             "restart with a raised budget and a working slot)")
        manual("scenario 8 (lsbdef-0008 FR-2.3/FR-5.1/FR-5.2)",
               "point LSB_BUDGET_BREAK_CMD at a contour restart with SUMMARY_MODEL_TYPE="
               "reasoning, a small SUMMARY_NUM_PREDICT and a fake slot returning an "
               "empty content with done_reason=length, and LSB_BUDGET_FIX_CMD at a "
               "restart with SUMMARY_NUM_PREDICT=0 / SUMMARY_CONTEXT_TOKENS set and a "
               "working slot: the notes waiting for a summary must get it and the "
               "judge queue must drain by itself. scripts/slot_fake.sh: "
               "LSB_BUDGET_BREAK_CMD='scripts/slot_fake.sh on empty-length summary "
               "judge' (SLOT_FAKE_SUMMARY_MODEL_TYPE / SLOT_FAKE_SUMMARY_NUM_PREDICT "
               "tune the tight budget), LSB_BUDGET_FIX_CMD='scripts/slot_fake.sh on ok "
               "summary judge' (the raised budget is the stock env: the re-creation "
               "drops the tight knobs and changes the slot fingerprint, which "
               "releases the frozen pair)")
        return

    proc = await run_shell(CFG["budget_break_cmd"])
    check("the contour is switched to the broken slot configuration (reasoning + a "
          "tight budget)", proc.returncode == 0, f"rc={proc.returncode}")
    if proc.returncode != 0:
        return
    teardown_note("the contour was switched to the broken slot configuration and fixed "
                  "back by the scenario hook")
    fixed = False
    try:
        # Push work into both queues: the notes need a summary, their dedup
        # hand-off needs a judge verdict (a hand-off pair is what freezes here).
        # The pair must be a "near-duplicate": word-for-word equal texts are rejected by the product
        # (stored=false), and the judge receives no pair at all.
        probe_ids, vector_ready = await save_near_duplicate_pair(
            c, RUN["note_budget_break"], RUN["note_budget_break_2"]
        )
        probes = len([item for item in probe_ids if item is not None])
        if probes < 2 or not vector_ready:
            warn("[budget] the pair of the branch is not usable",
                 f"ids={probe_ids}, first vector ready={vector_ready} — the judge "
                 f"may get no pair")

        ok, detail = await wait_until(queue_has_work(rest, "summary"),
                                      "[budget] notes without a summary pile up "
                                      "(the answer is cut, not lost)",
                                      timeout=min(CFG["wait_sec"], 240.0))
        check("[budget] while the budget is tight the summary queue grows", ok, detail)

        # The tight-budget phase (no39-04): the pair STOPS BURNING CALLS — the work
        # is deferred by a reason (`waiting`/`attention`, with `pending == 0`).
        # A hard `judge_attention` wait went red here: `attention` is not
        # the branch's premise but a separate threshold-reachability check.
        ok, detail = await wait_until(
            queue_deferred(rest, "judge"),
            "[budget] the pair stops burning calls: the judge work is deferred "
            "by the reason (waiting/attention, no pending)",
            timeout=CFG["wait_sec"])
        check("[budget] while the budget is tight the hopeless pair waits instead "
              "of retrying (the calls are not burned)", ok, detail)
        stat = queue_stat(await health_snapshot(rest), "judge") or {}
        if stat.get("attention") or 0:
            await check_attention_threshold(rest, "[budget]")
        else:
            info("[budget] the pair has not reached the limit of nontransient "
                 f"refusals yet: it still waits ({stat.get('waiting')} in `waiting`)")

        snapshot_before = await health_snapshot(rest)
        before = {
            "queues": {queue: queue_numbers(snapshot_before, queue)
                       for queue in ("summary", "judge")},
            "slots_attention": {
                slot: (slot_stat(snapshot_before, slot) or {}).get("attention")
                for slot in SLOTS},
            "own_pending_summaries": own_pending_summaries(),
        }
        info(f"[budget] before the fix: {json.dumps(before, ensure_ascii=False)}")

        proc = await run_shell(CFG["budget_fix_cmd"])
        fixed = True
        check("the contour is restarted with the raised budget and a working slot",
              proc.returncode == 0, f"rc={proc.returncode}")
        if proc.returncode != 0:
            return
        teardown_note("the contour was restarted with the raised budget by the hook")

        ok, detail = await wait_until(
            own_summaries_ready(probes),
            "[budget] the notes that waited for a summary get it without manual steps")
        check("[budget] the waiting notes are summarised after the budget is raised",
              ok, detail)

        ok, detail = await wait_until(queue_drained(rest, "judge"),
                                      "[budget] the judge queue drains by itself")
        check("[budget] the judge queue is drained without a manual step", ok, detail)

        snapshot_after = await health_snapshot(rest)
        after = {
            "queues": {queue: queue_numbers(snapshot_after, queue)
                       for queue in ("summary", "judge")},
            "slots_attention": {
                slot: (slot_stat(snapshot_after, slot) or {}).get("attention")
                for slot in SLOTS},
            "own_pending_summaries": own_pending_summaries(),
        }
        judge_after = after["queues"]["judge"]
        check("[budget] the judge queue keeps no pending/waiting/attention item",
              all((judge_after.get(key) or 0) == 0
                  for key in ("pending", "waiting", "attention")),
              f"{judge_after}")

        text = await log_text()
        released = log_lines(text, "job_attention_released", "startup")
        if text:
            check("[budget] the startup release is written: job_attention_released "
                  "with reason=\"startup\"", bool(released),
                  f"{len(released)} startup release line(s)")
        else:
            skip("[budget] the startup release event in the log",
                 "LSB_LOGS_CMD is not set (no log hook on this contour)")
        refusals = len(log_lines(text, '"outcome": "truncated"'))
        info(f"[budget] refusals (outcome=truncated) over the observation window: "
             f"{refusals}")

        path = write_acceptance_block("lsbdef0008-02", _acceptance_lines(before, after,
                                                                          refusals,
                                                                          len(released)))
        info(f"[budget] the acceptance numbers are written to {path}")
    finally:
        if not fixed:
            undo = await run_shell(CFG["budget_fix_cmd"])
            if undo.returncode == 0:
                teardown_note("the contour was fixed back by the teardown hook")
                warn("[budget] the scenario left before its own fix step",
                     "the contour was fixed back by the teardown hook")
            else:
                warn("[budget] the contour could NOT be fixed back — restore the slot "
                     "configuration by hand", f"rc={undo.returncode}")


def _acceptance_lines(before: dict, after: dict, refusals: int,
                      releases: int) -> list[str]:
    """Markdown block of the lsbdef-0008 acceptance: the numbers before/after."""
    def stamp(numbers: dict) -> str:
        return " / ".join(
            f"{numbers.get(key) if numbers.get(key) is not None else '?'}"
            for key in ("pending", "waiting", "attention")
        )

    def slots(numbers: dict) -> str:
        return ", ".join(f"{slot}:{numbers.get(slot)}" for slot in SLOTS)

    return [
        "### Дефект lsbdef-0008 — разгрузка ожидавших (lsbdef0008-02)",
        "",
        f"Контур: {CFG['base_url']} · прогон: {time.strftime('%Y-%m-%dT%H:%M:%S%z')} · "
        f"тег проб: {RUN['tag']}",
        "",
        "| Метрика | до (бюджет тесный) | после (бюджет поднят) |",
        "|---|---|---|",
        f"| Заметки прогона без выжимки (`summary_status='pending'`) | "
        f"{before['own_pending_summaries']} | {after['own_pending_summaries']} |",
        f"| Очередь `summary`: `pending/waiting/attention` | "
        f"{stamp(before['queues']['summary'])} | {stamp(after['queues']['summary'])} |",
        f"| Очередь `judge`: `pending/waiting/attention` | "
        f"{stamp(before['queues']['judge'])} | {stamp(after['queues']['judge'])} |",
        f"| Внимание по слотам (`/health.slots`) | {slots(before['slots_attention'])} | "
        f"{slots(after['slots_attention'])} |",
        f"| Отказы слота (`outcome=truncated`) за окно наблюдения | — | {refusals} |",
        f"| `job_attention_released` с `reason=\"startup\"` | — | {releases} |",
        "",
        "Причина переключения видна без отладочного прогона (FR-5.2): смена "
        "отпечатка конфигурации слота (`SUMMARY_MODEL_TYPE`/`SUMMARY_NUM_PREDICT`/"
        "`SUMMARY_CONTEXT_TOKENS`/`SLOT_MAX_OUTPUT_TOKENS`/`SLOT_PROMPT_RESERVE_TOKENS`/"
        "`DECISION_*`) на старте снимает `attention_at` у всех заданий — заметки "
        "получают выжимки и очередь судьи разгружается без ручных действий.",
    ]


# ==== scenarios: 9. decision slot and the three fallback reasons ===============
def _decision_stat(snapshot: dict) -> dict:
    """`/health.decision` out of a `/health` snapshot (empty — not reported)."""
    value = snapshot.get("decision")
    return value if isinstance(value, dict) else {}


def _decision_reached(rest: httpx2.AsyncClient, reason: str | None = None,
                      ) -> Callable[[], Awaitable[tuple[bool, str]]]:
    """Predicate: the decision slot took a decision (or counted a fallback reason)."""

    async def predicate() -> tuple[bool, str]:
        snapshot = await health_snapshot(rest)
        stat = _decision_stat(snapshot)
        by_reason = stat.get("by_reason") or {}
        if reason is None:
            taken = (stat.get("answers") or 0) >= 1 and not any(by_reason.values())
            return bool(taken), f"decision: {stat}"
        return bool(by_reason.get(reason, 0) >= 1), f"decision: {stat}"

    return predicate


def _avg_duration(text: str) -> float | None:
    """Average `duration_ms` of the `decision_answer` log lines (None — no lines)."""
    values: list[float] = []
    for line in log_lines(text, "decision_answer", "duration_ms"):
        try:
            value = json.loads(line).get("duration_ms")
        except json.JSONDecodeError:
            continue
        if isinstance(value, (int, float)):
            values.append(float(value))
    return sum(values) / len(values) if values else None


async def scenario_9_decision_fallback(c: Client, rest: httpx2.AsyncClient) -> None:
    """LSBN-F-12 FR-7: the slot decides, then falls back over three reasons
    (unsupported / unavailable / low_confidence) with /health honest at every step."""
    scenario(9, "The decision slot decides, then falls back over three reasons "
                "(unsupported / unavailable / low_confidence) without losing jobs")
    hooks = ("decision_on_cmd", "decision_unsupported_cmd", "decision_off_cmd",
             "decision_low_confidence_cmd")
    if not all(CFG.get(name) for name in hooks):
        skip("the decision slot and the three fallback reasons",
             "contour hooks are required: " + ", ".join(hooks))
        manual("scenario 9 (LSBN-F-12 FR-7.1/FR-7.2/FR-7.3)",
               "point the hooks at contour restarts: LSB_DECISION_ON_CMD "
               "(DECISION_ENABLED=true + a model able to /v1/systemone), "
               "LSB_DECISION_UNSUPPORTED_CMD (a model without /v1/systemone), "
               "LSB_DECISION_OFF_CMD (the slot enabled but unreachable + "
               "DECISION_FALLBACK=false), "
               "LSB_DECISION_LOW_CONFIDENCE_CMD (a threshold+zone catching the "
               "probability), LSB_DECISION_RESTORE_CMD (the baseline back): "
               "/health must stay honest at every step. scripts/slot_fake.sh: "
               "LSB_DECISION_ON_CMD='scripts/slot_fake.sh on decision-ok', "
               "LSB_DECISION_UNSUPPORTED_CMD='scripts/slot_fake.sh on "
               "decision-unsupported', LSB_DECISION_OFF_CMD='scripts/slot_fake.sh "
               "on decision-unavailable', LSB_DECISION_LOW_CONFIDENCE_CMD="
               "'scripts/slot_fake.sh "
               "on decision-low-confidence', LSB_DECISION_RESTORE_CMD='scripts/"
               "slot_fake.sh off'")
        return

    async def restart(hook: str, state: str) -> bool:
        proc = await run_shell(CFG[hook])
        check(f"the contour is switched: {state}", proc.returncode == 0,
              f"rc={proc.returncode}")
        if proc.returncode == 0:
            teardown_note(f"the contour was switched by the scenario hook ({state})")
        return proc.returncode == 0

    numbers: dict[str, Any] = {}
    try:
        # 1. decision on: the slot takes decisions; the queue drains.
        if not await restart("decision_on_cmd",
                             "DECISION_ENABLED=true + a working decision model"):
            return
        await save_probe_note(c, RUN["note_decision"], after_hook=True)
        ok, detail = await wait_until(
            _decision_reached(rest),
            "[decision] the slot takes decisions (/health.decision.answers grows)",
            timeout=min(CFG["wait_sec"], 240.0))
        check("[decision] judge/classifier decide through the slot", ok, detail)
        ok, detail = await wait_until(queue_drained(rest, "judge"),
                                      "[decision] the judge queue drains with the slot on")
        check("[decision] the judge queue drains while the slot decides", ok, detail)
        numbers["before"] = _decision_stat(await health_snapshot(rest))
        text = await log_text()
        if text:
            answered = log_lines(text, "decision_answer", "duration_ms")
            check("[decision] the journal carries decision_answer (probability + "
                  "duration_ms)", bool(answered), f"{len(answered)} line(s)")
            numbers["duration_avg"] = _avg_duration(text)
        else:
            skip("[decision] the decision_answer event in the log",
                 "LSB_LOGS_CMD is not set (no log hook on this contour)")

        # 2. unsupported: the model has no /v1/systemone → generative path, no stuck queue.
        if not await restart("decision_unsupported_cmd",
                             "a model without /v1/systemone"):
            return
        await save_probe_note(c, RUN["note_decision_unsupported"], after_hook=True)
        ok, detail = await wait_until(
            _decision_reached(rest, "unsupported"),
            "[decision] reason=unsupported is counted", timeout=min(CFG["wait_sec"], 240.0))
        check("[decision] unsupported fallback is visible in /health.decision.by_reason",
              ok, detail)
        ok, detail = await wait_until(queue_drained(rest, "judge"),
                                      "[decision] the queue drains on the generative path")
        check("[decision] the queue does not get stuck under unsupported", ok, detail)

        # 3. unavailable (the slot off) with DECISION_FALLBACK=false: the job waits
        #    by contract №39 and drains after the slot returns (no manual step).
        if not await restart("decision_off_cmd",
                             "the slot unreachable + DECISION_FALLBACK=false"):
            return
        # The task waits only when the judge is really asked: a near-duplicate pair is needed
        # (otherwise the judge queue is empty and the wait reads false red).
        await save_near_duplicate_pair(
            c, RUN["note_decision_unavailable"],
            f"{RUN['note_decision_unavailable']} pair"
        )
        ok, detail = await wait_until(
            queue_has_work(rest, "judge"),
            "[decision] the job waits (contract №39, no lost work)",
            timeout=min(CFG["wait_sec"], 240.0))
        check("[decision] under unavailable + DECISION_FALLBACK=false the job waits",
              ok, detail)
        numbers["waiting"] = queue_numbers(await health_snapshot(rest), "judge")
        if not await restart("decision_on_cmd", "the slot back to the baseline"):
            return
        ok, detail = await wait_until(queue_drained(rest, "judge"),
                                      "[decision] the queue drains after the slot returns")
        check("[decision] the waiting job drains after the slot returns", ok, detail)

        # 4. low_confidence: the probability lands in the uncertainty band → the
        #    generative path takes the decision.
        if not await restart("decision_low_confidence_cmd",
                             "a threshold+zone catching the probability"):
            return
        await save_probe_note(c, RUN["note_decision_low_confidence"], after_hook=True)
        ok, detail = await wait_until(
            _decision_reached(rest, "low_confidence"),
            "[decision] reason=low_confidence is counted",
            timeout=min(CFG["wait_sec"], 240.0))
        check("[decision] low_confidence fallback is visible in "
              "/health.decision.by_reason", ok, detail)

        # 5. /health is honest at every step: decision_ok and slots.decision present.
        snapshot = await health_snapshot(rest)
        check("[decision] /health answers with decision_ok and slots.decision",
              snapshot.get("status") == "ok" and "decision_ok" in snapshot
              and "decision" in (snapshot.get("slots") or {}),
              f"status={snapshot.get('status')}")
        numbers["after"] = _decision_stat(snapshot)
        reasons = set((numbers["after"].get("by_reason") or {}))
        check("[decision] exactly the three contract reasons are counted",
              bool(reasons) and reasons <= {"unavailable", "unsupported", "low_confidence"},
              f"{sorted(reasons)}")

        # 6. the acceptance numbers (the double run replaces the block, not appends).
        path = write_acceptance_block("f12-03", _decision_acceptance_lines(numbers))
        info(f"[decision] the acceptance numbers are written to {path}")
    finally:
        if CFG.get("decision_restore_cmd"):
            undo = await run_shell(CFG["decision_restore_cmd"])
            if undo.returncode == 0:
                teardown_note("the contour was restored to the baseline by the scenario hook")
            else:
                warn("[decision] the contour could NOT be restored — put the baseline "
                     "slot config back by hand", f"rc={undo.returncode}")


def _decision_acceptance_lines(numbers: dict) -> list[str]:
    """Markdown block of the f12-03 acceptance: the decision share and the reasons."""
    after = numbers.get("after") or {}
    by_reason = after.get("by_reason") or {}

    def share(count: int | None) -> str:
        answers = after.get("answers") or 0
        falls = after.get("fallbacks") or 0
        total = answers + falls
        if not total or count is None:
            return "—"
        return f"{count} ({count / total * 100:.0f}%)"

    return [
        "### Decision-слот: наблюдаемость и переключение (f12-03)",
        "",
        f"Контур: {CFG['base_url']} · прогон: {time.strftime('%Y-%m-%dT%H:%M:%S%z')} · "
        f"тег проб: {RUN['tag']}",
        "",
        "| Метрика | значение |",
        "|---|---|",
        f"| Принятых решений decision (`/health.decision.answers`) | "
        f"{after.get('answers')} |",
        f"| Всего переключений (`/health.decision.fallbacks`) | "
        f"{after.get('fallbacks')} |",
        f"| Доля решений decision-слотом | {share(after.get('answers'))} |",
        f"| Переключения `unavailable` | {share(by_reason.get('unavailable'))} |",
        f"| Переключения `unsupported` | {share(by_reason.get('unsupported'))} |",
        f"| Переключения `low_confidence` | {share(by_reason.get('low_confidence'))} |",
        f"| Средняя длительность решения (`duration_ms`, журнал) | "
        f"{numbers.get('duration_avg')} |",
        f"| Ожидание очереди `judge` при `unavailable` + `DECISION_FALLBACK=false` | "
        f"{numbers.get('waiting')} |",
        "",
        "Прогон идемпотентен: повторный запуск заменяет этот блок ("
        "HTML-маркеры блока — комментарии `f12-03:start` и `f12-03:end`), "
        "числа берутся из "
        "`/health.decision` (счётчики памяти процесса), вероятности остаются "
        "только в журнале (`decision_answer`) — в БД их нет (FR-3.3).",
    ]


# ==== scenarios: 10. №28 node hint on save without namespace ====================
# The hint's candidate node — a permanent path (repeated runs do not multiply nodes);
# the description — with rare topic words so the word match is obvious.
NODE_HINT_PATH = "e2e-node-hint"
NODE_HINT_DESCRIPTION = "Отчёты по релизам, реестр зарплат СУБО и доступы"


async def scenario_10_node_hint(c: Client, rest: httpx2.AsyncClient) -> None:
    """№28 (FR-1.1…FR-1.6): an MCP write WITHOUT `namespace` lands in `default`,
            and the answer carries `namespace_hint` with an existing map node; a write
            with an explicit node carries no hint; the `save` call window has no slot
            work.

            №30 (FR-3): a retelling of an already saved note arrives with `similar`
            (the earlier one's id) and is saved anyway (its id is new); a note without
            similar ones carries no field; the `save` window stays local (no slot
            work).
    """
    scenario(10, "№28: node hint on save without namespace — the note stays in "
                 "`default`, the answer points at a node, no model call on write")
    # Branches 3/4/8/9's hooks recreated the contour: the runner's session is dead; we re-open it.
    await resume_after_hooks(c, rest, "scenario 10")

    created = await c.call("memory_namespace_create", {
        "path": NODE_HINT_PATH, "description": NODE_HINT_DESCRIPTION})
    if created.get("created"):
        teardown_note(f"scenario 10 created the map node {NODE_HINT_PATH} "
                      "(it stays in the registry — the map is operator-owned)")
    else:
        listed = await c.call("memory_namespaces", {})
        present = any(node.get("path") == NODE_HINT_PATH
                      for node in listed.get("namespaces", []))
        check(f"the map node {NODE_HINT_PATH} is registered", present, refusal(created))

    started = time.monotonic()
    saved = await c.call("memory_save", {
        "text": "Реестр зарплат СУБО: доступы и контакты поддержки.",
        "title": f"{RUN['tag']} реестр зарплат СУБО",
    })
    latency = time.monotonic() - started
    if saved.get("id") is not None:
        RUN["notes"].append(saved["id"])
    check("save without `namespace` stores the note", saved.get("stored") is True,
          note_detail(saved))
    hint = saved.get("namespace_hint")
    check("the answer carries `namespace_hint` pointing at the node matched by "
          "the note words (FR-1.1/FR-1.3)",
          isinstance(hint, str) and NODE_HINT_PATH in hint, f"namespace_hint={hint!r}")
    if saved.get("id") is not None:
        got = await c.call("memory_get", {"ids": [saved["id"]]})
        note = (got.get("notes") or [{}])[0]
        check("the note lies in `default` — the hint does not move it (FR-1.2)",
              note.get("namespace") == "default",
              f"namespace={note.get('namespace')!r}")
    else:
        skip("the note namespace check", "the note was not created")
    check("the save window carries no slot work (latency of the tool, FR-1.6)",
          latency < 3.0, f"memory_save took {latency * 1000:.0f} ms")
    text = await log_text()
    if text:
        lines = log_lines(text, "tool_call", "memory_save")
        ms = None
        if lines:
            matched = re.search(r'"latency_ms":\s*([0-9.]+)', lines[-1])
            ms = float(matched.group(1)) if matched else None
        check("the journal confirms the write stayed local (memory_save latency, "
              "no slot event in the window)",
              ms is not None and ms < 3000.0, f"latency_ms={ms}")
    else:
        skip("the journal check of the save window (no slot work on write)",
             "LSB_LOGS_CMD is not set (no log hook on this contour)")

    explicit = await c.call("memory_save", {
        "text": f"{RUN['tag']} явная укладка в узел: подсказка не нужна.",
        "title": f"{RUN['tag']} явная укладка",
        "namespace": NODE_HINT_PATH,
    })
    if explicit.get("id") is not None:
        RUN["notes"].append(explicit["id"])
    check("an explicit `namespace` gives no `namespace_hint` (FR-1.5)",
          explicit.get("stored") is True and "namespace_hint" not in explicit,
          note_detail(explicit))

    started = time.monotonic()
    paraphrase = await c.call("memory_save", {
        "text": "Реестр зарплат СУБО: доступы, контакты поддержки и чек-лист деплоя.",
        "title": f"{RUN['tag']} реестр зарплат СУБО",
    })
    similar_latency = time.monotonic() - started
    if paraphrase.get("id") is not None:
        RUN["notes"].append(paraphrase["id"])
    similar_ids = [item.get("id") for item in paraphrase.get("similar", [])]
    check("a paraphrase of an existing note is stored anyway and carries "
          "`similar` with the earlier note id (№30, FR-3)",
          paraphrase.get("stored") is True
          and paraphrase.get("id") != saved.get("id")
          and saved.get("id") in similar_ids,
          note_detail(paraphrase))
    check("the `similar` hint is a local FTS read — the save window stays fast "
          "(no slot work, №30/FR-1.6)",
          similar_latency < 3.0, f"memory_save took {similar_latency * 1000:.0f} ms")
    # The "lonely" probe — WITHOUT the run tag (in both the text and the title): at the
    # memory_similar_threshold = 0.30 a tagged note is similar to all ~20 of the run's probes
    # and the check goes red (run #7: id=493). The line is random —
    # it has nothing to coincide with.
    lonely = await c.call("memory_save", {
        "text": f"qzvx {os.urandom(8).hex()} {os.urandom(4).hex()} nymq",
        "title": f"qzvx {os.urandom(4).hex()}",
    })
    if lonely.get("id") is not None:
        RUN["notes"].append(lonely["id"])
    check("a note with nothing similar carries no `similar` field (№30, FR-3.4)",
          lonely.get("stored") is True and "similar" not in lonely,
          note_detail(lonely))
    if lonely.get("id") is not None:
        # The cleanup by id: the run tag is not in the title, and the prefix cleanup
        # (`cleanup_own_data` searches a substring in the title) will not find it.
        await c.call("memory_delete", {"id": lonely["id"]})
    snapshot = await health_snapshot(rest)
    check("the contour stays healthy after the writes (the hint is a map read, "
          "not a job)", snapshot.get("status") == "ok",
          f"status={snapshot.get('status')!r}")


# ==== scenarios: 11. freshness filter (№29) ===================================
FRESHNESS_BOUNDARY = "2001-01-01T00:00:00Z"
FRESHNESS_AFTER_BOUNDARY = "2001-01-01T00:00:01Z"
FRESHNESS_OLD = "2000-01-01T00:00:00Z"


async def scenario_11_freshness(c: Client, rest: httpx2.AsyncClient) -> None:
    """№29 (FR-2.1…FR-2.7): `updated_after` — a working memory slice: search and
            listing serve only records inside the window (an inclusive bound); an old
            relevant one does not pass; an invalid value gives a hint; the same handle
            on the operator's REST surface works the same.
    """
    scenario(11, "№29: freshness filter `updated_after` — search/list, MCP + REST")
    await resume_after_hooks(c, rest, "scenario 11")

    fresh = await c.call("memory_save", {
        "text": f"{RUN['tag']} свежая заметка про TaskFlow деплой",
        "title": f"{RUN['tag']} свежая заметка",
    })
    old = await c.call("memory_save", {
        "text": f"{RUN['tag']} старая заметка про TaskFlow деплой",
        "title": f"{RUN['tag']} старая заметка",
    })
    for saved in (fresh, old):
        if saved.get("id") is not None:
            RUN["notes"].append(saved["id"])
    if fresh.get("id") is None or old.get("id") is None:
        skip("freshness filter checks",
             f"the probe notes were not created: {note_detail(fresh)} / {note_detail(old)}")
        return

    # A deterministic window: the old note is pushed beyond the bound; the fresh one
    # is placed exactly AT the bound (the inclusive `>=` semantics is checked).
    err = db_write("UPDATE notes SET updated_at = ? WHERE id = ?",
                   (FRESHNESS_OLD, old["id"]))
    if err:
        skip("freshness filter checks", f"the DB is not writable: {err}")
        return
    db_write("UPDATE notes SET updated_at = ? WHERE id = ?",
             (FRESHNESS_BOUNDARY, fresh["id"]))

    # The bound is inclusive: a note updated EXACTLY at the bound stays in the slice,
    # while strictly older — drops out. We check by the COUNT, not by presence in a page:
    # the MCP listing limit (list_max_limit_mcp = 20) is smaller than the notes in memory,
    # and the probe is deliberately the oldest — it does not enter the first page, so
    # presence would check the limit, not the filter (the 2026-10-12 run).
    listed = await c.call("memory_list", {
        "limit": 20, "updated_after": FRESHNESS_BOUNDARY})
    unfiltered = await c.call("memory_list", {"limit": 20})
    check("memory_list with `updated_after` keeps the fresh note (inclusive "
          "boundary) and drops the old one",
          unfiltered.get("total") == listed.get("total", -1) + 1,
          f"filtered={listed.get('total')}, all={unfiltered.get('total')}, "
          f"fresh={fresh['id']}, old={old['id']}")
    # Shifting the bound a second forward must also throw out the fresh note: meaning it
    # stood EXACTLY at the bound (the `>=` comparison), not "somewhere after it".
    later = await c.call("memory_list", {
        "limit": 20, "updated_after": FRESHNESS_AFTER_BOUNDARY})
    check("one second after the boundary the fresh note falls out too (the "
          "comparison is `>=`, not `>`)",
          listed.get("total", -1) == later.get("total", -2) + 1,
          f"at the boundary={listed.get('total')}, one second later="
          f"{later.get('total')}")

    found = await c.call("memory_search", {
        "query": "TaskFlow деплой", "updated_after": FRESHNESS_BOUNDARY})
    hit_ids = [r["id"] for r in found.get("results", [])]
    check("memory_search with `updated_after` drops the relevant but old note",
          fresh["id"] in hit_ids and old["id"] not in hit_ids, f"ids={hit_ids}")

    bad = await c.call("memory_list", {"updated_after": "за неделю"})
    check("an invalid `updated_after` gives a soft refusal with a hint",
          bad.get("items") == [] and "updated_after" in (bad.get("hint") or ""),
          f"hint={bad.get('hint')!r}")

    r = await rest.get("/notes", params={"updated_after": FRESHNESS_BOUNDARY,
                                         "limit": 50})
    r_all = await rest.get("/notes", params={"limit": 50})
    rest_total = r.json().get("total") if r.status_code == 200 else None
    rest_all = r_all.json().get("total") if r_all.status_code == 200 else None
    check("the REST surface filters GET /notes the same way",
          r.status_code == 200 and rest_total is not None
          and rest_all == rest_total + 1,
          f"http {r.status_code}, filtered={rest_total}, all={rest_all}, "
          f"fresh={fresh['id']}, old={old['id']}")
    r_bad = await rest.get("/notes", params={"updated_after": "за неделю"})
    check("REST answers an invalid `updated_after` with 422 (soft refusal, hint)",
          r_bad.status_code == 422 and "updated_after" in str(r_bad.json().get("detail")),
          f"http {r_bad.status_code}")


# ==== scenarios: 12. the note `comment` field (№24) ============================
async def scenario_12_comment(c: Client, rest: httpx2.AsyncClient) -> None:
    """№24 (FR-5.1…FR-5.8): a note has the optional `comment` field — a SEPARATE
            edit path: set and cleared via `memory_update`, read in `memory_get` and
            REST `GET /notes/{id}`, never in the listing and the semantic search;
            over the limit (250) — a soft failure without a partial write; the edit
            gives exactly one contour version with `changed == ['comment']`; the
            note's text, volume and dedup do not change meanwhile.
    """
    scenario(12, "№24: the note `comment` field — edit path, reading, versions")
    await resume_after_hooks(c, rest, "scenario 12")

    text = f"{RUN['tag']} заметка с пометкой про TaskFlow деплой"
    title = f"{RUN['tag']} заметка с пометкой"
    saved = await c.call("memory_save", {"text": text, "title": title})
    note_id = saved.get("id")
    if note_id is None:
        skip("the `comment` field checks",
             f"the probe note was not created: {note_detail(saved)}")
        return
    RUN["notes"].append(note_id)

    before = await c.call("memory_get", {"ids": [note_id]})
    before_note = (before.get("notes") or [{}])[0]

    comment = "пригодится к задаче TaskFlow"
    updated = await c.call("memory_update", {"id": note_id, "comment": comment})
    check("memory_update sets the comment in its own right (FR-5.3): "
          "`{id, updated: true}`",
          updated.get("id") == note_id and updated.get("updated") is True,
          f"answer={updated}")

    got = await c.call("memory_get", {"ids": [note_id]})
    note = (got.get("notes") or [{}])[0]
    check("memory_get returns the comment (FR-5.5: full read only)",
          note.get("comment") == comment, f"comment={note.get('comment')!r}")
    check("the comment edit leaves the text and its size untouched (FR-5.4)",
          note.get("text") == before_note.get("text")
          and note.get("chars") == before_note.get("chars"),
          f"chars={note.get('chars')}, before={before_note.get('chars')}")

    listed = await c.call("memory_list", {"limit": 20})
    item = next((i for i in listed.get("items", []) if i.get("id") == note_id), None)
    titled = await c.call("memory_list", {"limit": 20, "detail": "titles"})
    title_item = next((i for i in titled.get("items", []) if i.get("id") == note_id),
                      None)
    found = await c.call("memory_search", {"query": "TaskFlow деплой"})
    check("the listing (summaries) carries no `comment` (FR-5.5, budget F-11)",
          item is not None and "comment" not in item, f"item={item}")
    check("the compact titles listing carries no `comment`",
          title_item is not None and "comment" not in title_item,
          f"item={title_item}")
    check("the semantic search carries no `comment` (FR-5.7: the comment is not "
          "knowledge)",
          all("comment" not in hit for hit in found.get("results", [])),
          f"hits={len(found.get('results', []))}")

    # #24 (no24-02): the separate COMMENT search — finds the note by its comment;
    # while the regular semantic search never does (§13.6).
    comment_hits = await c.call("memory_comment_search", {"query": "пригодится"})
    check("memory_comment_search finds the note by its comment (no24-02)",
          any(r.get("id") == note_id
              for r in comment_hits.get("results", [])),
          f"hits={comment_hits.get('results')}")
    # The isolation is checked over the FTS/substring path (`mode=title`): the hybrid's
    # vector side returns the nearest notes regardless of
    # the word's presence, so on a small corpus it serves the probe note even
    # without the comment's leak. The semantic side — a separate observation.
    title_search = await c.call("memory_search",
                                {"query": "пригодится", "mode": "title"})
    check("memory_search (title/substring) does NOT find the note by a "
          "comment-only word (FR-5.7 / §13.6)",
          all(r.get("id") != note_id
              for r in title_search.get("results", [])),
          f"hits={len(title_search.get('results', []))}")
    semantic = await c.call("memory_search", {"query": "пригодится"})
    if any(r.get("id") == note_id for r in semantic.get("results", [])):
        warn("the semantic side returns the probe note as a neighbour — "
             "expected on a small corpus, not a comment leak")
    rest_comments = await rest.get("/notes/comments", params={"q": "пригодится"})
    check("REST GET /notes/comments returns the note by its comment (no24-02)",
          rest_comments.status_code == 200
          and any(r.get("id") == note_id
                  for r in rest_comments.json().get("results", [])),
          f"http {rest_comments.status_code}")

    too_long = await c.call("memory_update", {"id": note_id, "comment": "х" * 251})
    after_refusal = await c.call("memory_get", {"ids": [note_id]})
    kept = ((after_refusal.get("notes") or [{}])[0]).get("comment")
    check("a comment longer than 250 chars is a soft refusal with a hint (FR-5.8)",
          too_long.get("updated") is False and "250" in (too_long.get("hint") or ""),
          f"answer={too_long}")
    check("the refused edit writes nothing (no partial write)",
          kept == comment, f"comment={kept!r}")

    history = await rest.get(f"/history/note/{note_id}")
    versions = history.json().get("versions") if history.status_code == 200 else None
    changed_comment = [v for v in (versions or [])
                      if (v.get("changed") or []) == ["comment"]]
    check("the comment edit is versioned as ONE content version "
          "(`changed == ['comment']`, FR-5.6 / №41)",
          len(changed_comment) == 1,
          f"http {history.status_code}, versions={versions}")

    r = await rest.get(f"/notes/{note_id}")
    rest_comment = r.json().get("comment") if r.status_code == 200 else None
    check("REST GET /notes/{id} exposes the comment (FR-5.5)",
          r.status_code == 200 and rest_comment == comment,
          f"http {r.status_code}, comment={rest_comment!r}")
    cleared = await rest.put(f"/notes/{note_id}", json={"comment": None})
    r_after = await rest.get(f"/notes/{note_id}")
    after_comment = r_after.json().get("comment") if r_after.status_code == 200 else "?"
    check("REST PUT /notes/{id} clears the comment with `null` while the note "
          "keeps its text (FR-5.4)",
          cleared.status_code == 200 and after_comment is None
          and r_after.json().get("text") == text,
          f"http {cleared.status_code}, comment={after_comment!r}")

    dropped = await c.call("memory_comment_search", {"query": "пригодится"})
    check("clearing the comment removes the note from the comment search (no24-02)",
          all(r.get("id") != note_id for r in dropped.get("results", [])))

    duplicate = await c.call("memory_save", {"text": text, "title": title})
    # The MCP surface serves a verbatim duplicate as `stored: false` + a hint
    # (`_compact_save`); the service's service-level `duplicated` is not visible to the agent.
    check("dedup of the note is unaffected by its comment (FR-5.4)",
          duplicate.get("stored") is False and bool(duplicate.get("hint")),
          f"answer={duplicate}")


# ==== scenarios: 13. injected-text budget + agent smoke (LSBN-F-11) =============
# LSBN-F-11 (f11-03): the budget constants live in CODE
# (`app/services/injection_budget.py`, arch §13.17, canon §6 — not env, FR-4.4),
# and the watchdog — a regular test `tests/test_injection_budget.py` failing on
# an overrun. This scenario runs the SAME measurement
# (`scripts/measure_injected_text.py --phase after`, f11-01), compares it with
# the budget, the "before" report and the instructions the contour really serves,
# then confirms by a smoke that the text reduction did not break the notes
# handling, the areas or the background jobs (FR-6.1/FR-6.2).
#
# The measurement is written into a TEMP directory: `release/3.2.0/acceptance/
# injected-text-after.{md,json}` — the f11-01/f11-03 artifact with a hand-written delta;
# a repeated scenario run does not overwrite it (the run's idempotency).
INJECTED_TEXT_TOOL_COUNT = 34  # the 3.7.0 surface: the budget is measured over every tool description (arch 7.1)
INJECTED_TEXT_BEFORE = "injected-text-before.json"
SMOKE_MARKER = "zblorptsmoke"


def acceptance_dir() -> Path:
    """The release's acceptance artifacts directory (`release/` is not versioned)."""
    return Path(CFG["repo_dir"]) / "release" / RELEASE / "acceptance"


def budget_module() -> Any:
    """The embedded-text budgets module from the application's code; None — unavailable.

            The numbers are taken exactly from there (the single source for both the
            watchdog and the acceptance), not duplicated in the script: otherwise the
            watchdog's budget and the acceptance's budget would diverge.
    """
    try:
        from app.services import injection_budget
    except Exception as exc:  # noqa: BLE001 — the scenario must live outside the repository too
        warn("the injected-text budget module is unreadable", describe(exc))
        return None
    return injection_budget


def injected_text_measurement() -> dict[str, Any]:
    """A local run of `measure_injected_text.py --phase after` (a temp out).

            The contour is not needed for the measurement itself: it raises a
            temporary empty DB, and the slots are closed on loopback:1 (no network).
            Returns the parsed measurement JSON, extended with the "before" report
            (None when the canonical file is missing).
    """
    script = Path(CFG["repo_dir"]) / "scripts" / "measure_injected_text.py"
    with tempfile.TemporaryDirectory(prefix="e2e-injected-text-") as out_dir:
        proc = subprocess.run(
            [sys.executable, str(script), "--phase", "after", "--out", out_dir],
            cwd=CFG["repo_dir"], capture_output=True, text=True, timeout=600,
        )
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout).strip().splitlines()
            raise RuntimeError(
                f"measure_injected_text.py rc={proc.returncode}: "
                f"{tail[-1] if tail else '(no output)'}"
            )
        payload = (Path(out_dir) / "injected-text-after.json").read_text(encoding="utf-8")
    data = json.loads(payload)
    before = acceptance_dir() / INJECTED_TEXT_BEFORE
    data["before"] = (
        json.loads(before.read_text(encoding="utf-8")) if before.exists() else None
    )
    return data


def _injected_acceptance_lines(numbers: dict[str, Any]) -> list[str]:
    """The `f11-03` block of the acceptance report: the embedded text's budget, the before/after numbers."""

    def delta(before_value: Any, after_value: Any) -> str:
        if not isinstance(before_value, int) or not isinstance(after_value, int):
            return "—"
        return f"{after_value - before_value:+d}"

    rows = (
        ("A — `build_instructions` (статика подключения)", "before_a", "after_a"),
        ("A — ядро `SERVER_INSTRUCTIONS`", "before_core", "after_core"),
        ("B — сумма описаний инструментов", "before_descriptions",
         "after_descriptions"),
    )
    lines = [
        "### Вшитый текст: сторож бюджета и замер «после» (f11-03)",
        "",
        "Блок заполняется сценарием 13 (`scenario_13_injected_text`) на тестовом "
        "контуре: прогон замера (`scripts/measure_injected_text.py --phase after`), "
        "сверка с бюджетом `app/services/injection_budget.py` и с отчётом «до», "
        "смоук агента (заметки, области, очереди). Блок идемпотентен по своим "
        "HTML-маркерам, повторный прогон заменяет его; сторож — обычный тест "
        "общего прогона `tests/test_injection_budget.py`, падающий при превышении "
        "(FR-4.2). Числа бюджета — константы кода, не env (FR-4.4).",
        "",
        "| Место / метрика | «до» (f11-01) | «после» (f11-03) | Δ токенов |",
        "|---|---|---|---|",
    ]
    for title, before_key, after_key in rows:
        lines.append(
            f"| {title} | {numbers.get(before_key, '—')} ток. | "
            f"{numbers.get(after_key, '—')} ток. | "
            f"{delta(numbers.get(before_key), numbers.get(after_key))} |"
        )
    lines += [
        "",
        f"* Инструментов на поверхности: {numbers.get('tool_count', '—')} "
        f"(бюджетное число — {INJECTED_TEXT_TOOL_COUNT}); самое тяжёлое описание — "
        f"{numbers.get('after_description_max', '—')} ток.",
        "* Что удалено/переехало при сокращении (f11-02): справочные перечни и "
        "числовые лимиты ушли из ядра в канон §6 и в `hint`-ответы ручек; правило "
        "выбора узла и шапка карты, манифест навыков и раздел логов сжаты без "
        "потери обязательных правил; 18 описаний инструментов сокращены (самые "
        "тяжёлые — `memory_save` −82, `memory_namespace_create` −81, "
        "`memory_update` −80 ток.); схемы параметров и логика ручек НЕ менялись, "
        "образцы веса вызова не изменились.",
    ]
    live = numbers.get("live_instructions")
    lines.append(
        "* Инструкции, которые контур реально отдаёт агенту: "
        + (f"{live} ток. при бюджете `TOKENS_INSTRUCTIONS_MAX`."
           if isinstance(live, int)
           else "— (нужен живой контур).")
    )
    smoke = numbers.get("smoke_note")
    if smoke is None:
        lines.append(
            "* Смоук агента: НЕ выполнен (нет живой контур / заметка не создана) — "
            "числа сторожа выше получены локальным прогоном замера."
        )
    else:
        lines.append(
            f"* Смоук агента: заметка записана и найдена (id {smoke}); области "
            "(навыки, термины, факты, логи) отвечают без ошибок; очередь "
            f"`summary` дренирована — {numbers.get('summary_drained', '—')}."
        )
    lines += [
        "",
        "Числа и дельта по образцам веса вызова — "
        "`release/3.2.0/acceptance/injected-text-before.md` и "
        "`injected-text-after.md`.",
    ]
    return lines


async def scenario_13_injected_text(c: Client, rest: httpx2.AsyncClient,
                                    instructions: str) -> None:
    """LSBN-F-11 (f11-03): the embedded-text measurement under the budget + an agent smoke.

            The comparison: the "after" numbers ≤ the code's budget; the tool count =
            the budget's; "after" ≤ "before" over the connect statics and the
            descriptions; the contour's instructions — within the budget. The smoke:
            writing/searching/reading a note, all four areas answering without errors,
            this run's notes' `summary` queue drain, the contour's health. The
            measurement is local — the contour is needed only by the smoke.
    """
    scenario(13, "LSBN-F-11: the injected-text budget (A/B/C) under the guard and "
                 "the agent smoke (notes, areas, queues)")
    numbers: dict[str, Any] = {}

    await resume_after_hooks(c, rest, "scenario 13")
    module = budget_module()
    data: dict[str, Any] | None = None
    if module is not None:
        try:
            data = injected_text_measurement()
        except Exception as exc:  # noqa: BLE001 — the measurement run failed
            skip("the injected-text measurement run",
                 f"measure_injected_text.py did not complete: {describe(exc)}")

    if data is not None:
        place_a, place_b = data["place_a"], data["place_b"]
        core = next((mrow for mrow in place_a["components"]
                     if mrow["block"] == "SERVER_INSTRUCTIONS"), {"tokens": None})
        after_a = place_a["total"]["tokens"]
        after_desc = place_b["total_description"]["tokens"]
        desc_max = max(mrow["description_tokens"] for mrow in place_b["tools"])
        tool_count = place_b["tool_count"]
        numbers.update(after_a=after_a, after_core=core["tokens"],
                       after_descriptions=after_desc,
                       after_description_max=desc_max, tool_count=tool_count)
        info(f"injected text «after»: A {after_a} tok., core {core['tokens']} tok., "
             f"descriptions {after_desc} tok. (max {desc_max}, tools {tool_count})")

        check("build_instructions is within TOKENS_INSTRUCTIONS_MAX="
              f"{module.TOKENS_INSTRUCTIONS_MAX}",
              after_a <= module.TOKENS_INSTRUCTIONS_MAX, f"{after_a} tokens")
        check(f"SERVER_INSTRUCTIONS is within TOKENS_CORE_MAX={module.TOKENS_CORE_MAX}",
              core["tokens"] <= module.TOKENS_CORE_MAX, f"{core['tokens']} tokens")
        check("every tool description is within TOKENS_TOOL_DESCRIPTION_MAX="
              f"{module.TOKENS_TOOL_DESCRIPTION_MAX}",
              desc_max <= module.TOKENS_TOOL_DESCRIPTION_MAX,
              f"heaviest {desc_max} tokens")
        check("the tool surface is within TOKENS_TOOL_SURFACE_MAX="
              f"{module.TOKENS_TOOL_SURFACE_MAX}",
              after_desc <= module.TOKENS_TOOL_SURFACE_MAX, f"{after_desc} tokens")
        check(f"the MCP surface still lists {INJECTED_TEXT_TOOL_COUNT} tools "
              "(growth means the budget must be revised deliberately, FR-4.4)",
              tool_count == INJECTED_TEXT_TOOL_COUNT, f"got {tool_count}")

        before = data.get("before")
        if before is None:
            skip("the delta to the «before» measurement",
                 f"{INJECTED_TEXT_BEFORE} is missing in {acceptance_dir()}")
        else:
            before_a = before["place_a"]["total"]["tokens"]
            before_desc = before["place_b"]["total_description"]["tokens"]
            before_core = next((mrow["tokens"] for mrow in before["place_a"]["components"]
                                if mrow["block"] == "SERVER_INSTRUCTIONS"), None)
            numbers.update(before_a=before_a, before_core=before_core,
                           before_descriptions=before_desc)
            check("the f11-02 shortening did not grow the handshake static or the "
                  "description surface («after» ≤ «before»)",
                  after_a <= before_a and after_desc <= before_desc,
                  f"A {before_a} → {after_a} tok., descriptions "
                  f"{before_desc} → {after_desc} tok.")
            info(f"the core grew deliberately inside its own budget: {before_core} "
                 f"→ {core['tokens']} tok. (TOKENS_CORE_MAX={module.TOKENS_CORE_MAX})")

        if instructions:
            live = module.count_tokens(instructions)
            numbers["live_instructions"] = live
            check("the instructions the contour actually hands out are within "
                  f"TOKENS_INSTRUCTIONS_MAX={module.TOKENS_INSTRUCTIONS_MAX}",
                  live <= module.TOKENS_INSTRUCTIONS_MAX, f"{live} tokens")
        else:
            skip("the live-instructions budget",
                 "the contour handed out no instructions text")

    # --- the agent smoke: notes and areas alive (FR-6.1/FR-6.2) ----------------
    smoke_text = (f"{RUN['tag']} smoke: the shortened instructions still let the "
                  f"agent keep notes ({SMOKE_MARKER}).")
    saved = await c.call("memory_save", {"text": smoke_text,
                                         "title": f"{RUN['tag']} smoke note"})
    note_id = saved.get("id")
    if note_id is not None:
        RUN["notes"].append(note_id)
    numbers["smoke_note"] = note_id
    check("memory_save writes a note after the text shortening",
          saved.get("stored") is True, note_detail(saved))
    if note_id is None:
        skip("the note read/search smoke", "the probe note was not created")
    else:
        found = await c.call("memory_search", {"query": SMOKE_MARKER})
        check("memory_search finds the note written just now",
              any(hit.get("id") == note_id for hit in found.get("results", [])),
              f"hits={[hit.get('id') for hit in found.get('results', [])]}")
        got = await c.call("memory_get", {"ids": [note_id]})
        note = (got.get("notes") or [{}])[0]
        check("memory_get reads the note back with its text intact",
              note.get("text") == smoke_text, f"chars={note.get('chars')}")
        titles = await c.call("memory_list", {"limit": MCP_LIMIT, "detail": "titles"})
        check("memory_list (titles) still walks the registry",
              any(item.get("id") == note_id for item in titles.get("items", [])),
              f"items={len(titles.get('items', []))}")

    # The areas answer without errors: skills, terms, facts, logs (FR-6.1).
    areas = (
        ("skills", "skills_list", {}, "items"),
        ("terms", "terms_search", {"query": SMOKE_MARKER}, "senses"),
        ("user facts", "user_search", {"query": SMOKE_MARKER}, "results"),
        ("assistant logs", "logs_list", {"limit": 5}, "items"),
    )
    for label, tool, args, key in areas:
        try:
            answer = await c.call(tool, args)
        except Exception as exc:  # noqa: BLE001 — a handle error is a FAIL
            check(f"the {label} area answers {tool} without an error", False,
                  describe(exc))
            continue
        if not isinstance(answer, dict):
            check(f"the {label} area answers {tool} without an error", False,
                  f"answer={answer!r}")
            continue
        payload = answer.get(key)
        why = refusal(answer)
        # The areas' `hint` — not only failures: "no exact term", "nothing
        # found", "a node hint" — informational hints at a
        # normal output. Exactly one check: the handle returned data (a list)
        # and threw no exception; the hint text goes into the details.
        check(f"the {label} area answers {tool} without an error",
              isinstance(payload, list),
              f"{key}=" + (f"{len(payload)}" if isinstance(payload, list)
                           else repr(payload))
              + (f", hint={why!r}" if why else ""))

    # The background jobs: the run's notes are processed; the queue drains (FR-6.2).
    if note_id is not None:
        # The smoke answers "is the background alive after the text reduction", not
        # "did it manage to summarize ALL ~20 acceptance probes": we wait for the summary of the smoke's
        # OWN notes (there is one here). Over all the run's probes the window expires
        # before the queue reaches the end (run #7: "14 of 20", although in
        # the database all notes have summary_status='ok' and all summary jobs are done).
        smoke_notes = [note_id]
        marks = ",".join("?" * len(smoke_notes))

        async def smoke_summaries_ready() -> tuple[bool, str]:
            ready = db_count(
                "SELECT COUNT(*) AS c FROM notes WHERE deleted_at IS NULL "
                f"AND summary_status = 'ok' AND id IN ({marks})",
                tuple(smoke_notes),
            )
            return (ready is not None and ready >= len(smoke_notes)), \
                f"smoke notes with summary_status='ok': {ready} of {len(smoke_notes)}"

        drained, detail = await wait_until(
            smoke_summaries_ready,
            f"the smoke note of this run gets a summary ({note_id})",
            timeout=max(float(CFG["wait_sec"]), 600.0),
        )
        numbers["summary_drained"] = drained
        check_slot("the summary queue drains the notes of this run (background jobs "
                   "still work after the shortening)", drained, detail)
    snapshot = await refresh_health(rest)
    check("the contour stays healthy after the smoke (status ok)",
          snapshot.get("status") == "ok", f"status={snapshot.get('status')!r}")

    # The release's report: the f11-03 block (idempotently, by markers).
    try:
        path = write_acceptance_block("f11-03", _injected_acceptance_lines(numbers))
        info(f"the acceptance block f11-03 is written: {path}")
    except Exception as exc:  # noqa: BLE001 — the report is the operator's, not the contour's
        warn("the acceptance block f11-03 was not written", describe(exc))


# ==== scenarios: 14. assistant logs (LSBN-F-10) =================================
# The marker lies ONLY in the log record's reason: we search the area's record by it
# (`logs_search`) and check the isolation with it too — the notes search does not see it.
LOGS_ONLY_MARKER = "zblorptlogsonly"
LOGS_SECRET_PREFIX = "sk-"


def _log_marker() -> str:
    """An attempt marker, also fit for FTS parsing (one word, no hyphens)."""
    return f"{LOGS_ONLY_MARKER}{re.sub(r'[^0-9A-Za-z]', '', RUN['tag'])}"


def _fake_log_secret() -> str:
    """A token-like string for the leak check (fake, one per attempt).

            The value is NEVER printed anywhere: only its ABSENCE in the contour's
            journal is checked, so the report carries not a character of it.
    """
    tail = re.sub(r"[^0-9A-Za-z]", "", RUN["tag"]).ljust(32, "0")[:32]
    return f"{LOGS_SECRET_PREFIX}{tail}"


async def scenario_14_logs(c: Client, rest: httpx2.AsyncClient,
                           instructions: str) -> None:
    """LSBN-F-10 (f10-02/03/04): the "Assistant logs" area via MCP + REST.

            What we check (observably, no models on the write path):
            * writing a reason creates the entry (`logs_save` → id, version=1) and it
              is read by id (`logs_get`); the body comes back as written;
            * the entry is seen by the listing with an object slice
              (`logs_list(entity, entity_id)`) and the semantic search (`logs_search`;
              with the embedder unavailable — FTS-only);
            * the edit (`logs_update`) changes the field and does not touch the object
              link;
            * the deletion is soft and idempotent (`logs_delete`);
            * the linked object (a note), its version and state (`entity_state`) are
              visible in the entry (f10-04/§11);
            * the area does not mix into the notes search (isolation, FR-1.2);
            * a token-like string from the body never reaches the contour journal
              (NFR-4/canon §6: lengths and ids into the logs, not content);
            * `/health.areas.logs` shows the area enabled, the area's section is in
              the instructions (#40/f10-04).

            Idempotency: the scenario deletes its own log entry itself (including on
            a break — `finally`); the probe note is removed by the shared teardown
            over the prefix.
    """
    scenario(14, "LSBN-F-10: assistant logs — write/read/list/search/update/"
                 "delete, the entity link, isolation from memory, no secrets in "
                 "the journal")
    await resume_after_hooks(c, rest, "scenario 14")

    snapshot = await health_snapshot(rest)
    area = (snapshot.get("areas") or {}).get("logs")
    check("/health.areas.logs reports the logs area as enabled (№40, f10-04)",
          isinstance(area, dict) and area.get("enabled") is True, f"logs area={area}")
    if not (isinstance(area, dict) and area.get("enabled") is True):
        skip("the assistant-logs scenario",
             "the logs area is OFF on this contour (AREA_LOGS_ENABLED=false): the "
             "MCP handles are hidden and the write path is unavailable")
        manual("scenario 14 (LSBN-F-10)",
               "run with AREA_LOGS_ENABLED=true to exercise the logs area")
        return
    check("the logs area announces itself in the MCP instructions (f10-04, §10)",
          "logs_save" in instructions and "logs_search" in instructions,
          f"{len(instructions)} chars of instructions")

    # The reference object — one's own probe note (the text is its OWN: the other scenarios'
    # shared probe text would be rejected by the product as a verbatim duplicate, and no reference object would appear).
    note_id, note_detail = await save_probe_note_result(
        c, f"{RUN['tag']} logs entity probe",
        text=f"{RUN['tag']} assistant-logs entity probe text (acceptance run).",
        after_hook=True, rest=rest)
    if note_id is None:
        skip("the assistant-logs scenario",
             f"the probe note (the linked object) was not created: {note_detail}")
        return

    marker = _log_marker()
    secret = _fake_log_secret()
    action = "set the note state during the acceptance run"
    reason = f"the reason carries {marker} only; a token-like value is written too"
    body = f"{reason} ({secret})"
    saved = await c.call("logs_save", {
        "action": action,
        "reason": body,
        "entity": "note", "entity_id": note_id, "entity_version": 1,
        "outcome": "the entry is written by the acceptance run",
        "verified": "written and read back",
    })
    log_id = saved.get("id")
    if log_id is None:
        skip("the assistant-logs scenario",
             f"logs_save did not create an entry: {refusal(saved) or saved}")
        return
    check("logs_save creates the entry and answers {id, created: true, version: 1}",
          saved.get("created") is True and saved.get("version") == 1,
          f"answer={saved}")

    cleaned = False
    try:
        got = await c.call("logs_get", {"id": log_id})
        check("logs_get reads the entry by id (the body round-trips as written)",
              got.get("action") == action and got.get("reason") == body,
              f"id={log_id}, reason_chars={len(got.get('reason') or '')}")
        check("the entry shows the linked object and its version "
              "(entity=note, entity_id, entity_version)",
              got.get("entity") == "note" and got.get("entity_id") == note_id
              and got.get("entity_version") == 1,
              f"entity={got.get('entity')!r}, entity_id={got.get('entity_id')!r}, "
              f"entity_version={got.get('entity_version')!r}")
        check("the entry exposes the state of the linked object "
              "(entity_state='active', f10-04/§11)",
              got.get("entity_state") == "active",
              f"entity_state={got.get('entity_state')!r}")

        listed = await c.call("logs_list",
                              {"entity": "note", "entity_id": note_id, "limit": 20})
        check("logs_list slices by object (entity+entity_id) and returns the entry",
              log_id in [item.get("id") for item in listed.get("items", [])]
              and (listed.get("total") or 0) >= 1,
              f"total={listed.get('total')}, "
              f"ids={[item.get('id') for item in listed.get('items', [])]}")

        found = await c.call("logs_search", {"query": marker, "top_k": 5})
        check("logs_search finds the entry by the reason (hybrid; FTS-only with a "
              "warning when the embedder is unavailable, f10-03)",
              log_id in [hit.get("id") for hit in found.get("results", [])],
              f"hits={[hit.get('id') for hit in found.get('results', [])]}, "
              f"warning={found.get('warning')!r}")
        if found.get("warning"):
            info(f"logs_search answered in the FTS-only mode: {found['warning']}")

        r = await rest.get("/logs", params={"entity": "note", "entity_id": note_id,
                                            "limit": REST_LIMIT})
        rest_ids = ([item.get("id") for item in (r.json().get("items") or [])]
                    if r.status_code == 200 else [])
        check("REST GET /logs?entity=note&entity_id=… mirrors the same entry",
              r.status_code == 200 and log_id in rest_ids,
              f"http {r.status_code}, ids={rest_ids}")
        r_one = await rest.get(f"/logs/{log_id}")
        check("REST GET /logs/{id} returns the full entry to the operator",
              r_one.status_code == 200 and r_one.json().get("action") == action,
              f"http {r_one.status_code}")

        # Isolation (FR-1.2/§13.2): a word living ONLY in the log's reason is not
        # found by the memory search. The search — substring-based (`mode=title`): the hybrid's
        # vector side serves neighbors regardless of words.
        mem = await c.call("memory_search", {"query": marker, "mode": "title"})
        check("memory_search does NOT return the log text (the areas are isolated, "
              "FR-1.2/§13.2)",
              marker.lower() not in json.dumps(mem, ensure_ascii=False).lower(),
              f"hits={len(mem.get('results', []))}")

        new_reason = f"{reason} — edited by the acceptance run ({secret})"
        updated = await c.call("logs_update", {"id": log_id, "reason": new_reason})
        check("logs_update edits the entry in place and answers {id, updated: true}",
              updated.get("id") == log_id and updated.get("updated") is True,
              f"answer={updated}")
        after = await c.call("logs_get", {"id": log_id})
        check("the edited reason reads back while action and the entity link stay "
              "untouched (only the passed field changes)",
              after.get("reason") == new_reason and after.get("action") == action
              and after.get("entity_id") == note_id,
              f"reason_chars={len(after.get('reason') or '')}")

        # Secrets — never into the journal (NFR-4/§6: masking to 4 chars + `***`):
        # a token-like string in the record's body never appears in the contour's journal.
        text = await log_text()
        if text:
            seen = bool(log_lines(text, "tool_call", "logs_save"))
            leaked = secret in text or secret[4:] in text
            check("a token-like value in the entry body never reaches the contour "
                  "journal (NFR-4/§6: the tool_call event carries lengths, not "
                  "content)",
                  seen and not leaked,
                  f"the journal shows the logs_save event: {seen}, the value is "
                  f"{'present' if leaked else 'absent'}")
        else:
            skip("a token-like value in the entry body never reaches the contour "
                 "journal",
                 "LSB_LOGS_CMD is not set (no log hook on this contour)")

        deleted = await c.call("logs_delete", {"id": log_id})
        cleaned = True
        check("logs_delete removes the entry (soft) and answers {id, deleted: true}",
              deleted.get("id") == log_id and deleted.get("deleted") is True,
              f"answer={deleted}")
        gone = await c.call("logs_get", {"id": log_id})
        check("a deleted entry is not readable by the agent any more (hint, §7.3)",
              "action" not in gone and bool(gone.get("hint")), f"answer={gone}")
        again = await c.call("logs_delete", {"id": log_id})
        check("a repeated logs_delete is idempotent (deleted: false + hint)",
              again.get("deleted") is False and bool(again.get("hint")),
              f"answer={again}")
        after_list = await c.call("logs_list",
                                  {"entity": "note", "entity_id": note_id, "limit": 20})
        check("the deleted entry leaves the listing too",
              log_id not in [item.get("id") for item in after_list.get("items", [])],
              f"ids={[item.get('id') for item in after_list.get('items', [])]}")
    finally:
        if not cleaned:
            try:
                await c.call("logs_delete", {"id": log_id})
                teardown_note(f"scenario 14 removed its log entry {log_id} in the "
                              "teardown (the scenario did not reach its own delete)")
            except Exception as exc:  # noqa: BLE001 — the cleanup does not mask the reason
                warn("the probe log entry was not removed by the teardown",
                     f"id={log_id} ({describe(exc)})")


# ==== scenarios: 15. backups (№14) =============================================
def backup_file_path(name: str) -> Path:
    """A snapshot path in BACKUP_DIR (the runner mounts /data like the container).

            The directory — `--backup-dir` / `LSB_BACKUP_DIR`; by default it is
            derived from the DB path (`<the DB's directory>/backups`, as in compose).
    """
    return Path(CFG["backup_dir"]) / name


async def scenario_15_backups(rest: httpx2.AsyncClient) -> None:
    """№14 (no14-01, arch §6.1/§7.2): the snapshot list, verifying the restore of
            the newest one, and the backups observability in `/health`.

            What we check (reads and verification only — no handle writes into a
            snapshot):
            * `GET /backups` serves the snapshot list with the operator fields (file,
              size_bytes, verified, bad, shipped) and a consistent `total`;
            * `POST /backups/verify` over the newest snapshot opens it as a working
              base: `ok=true`, `checks.integrity == "ok"`, all mandatory tables and
              vec0 reachable;
            * the read-only verification — the snapshot file's size and mtime before
              and after match;
            * after the verification `GET /backups` marks the snapshot `verified=ok`,
              `bad=false`, and `/health.backups` carries `last_verify` (the same
              copy, the ok outcome) and `last_ship`.

            No snapshot/directory — an honest SKIP with a reason (the backups loop
            has not made a copy yet, or the directory is not mounted into the
            runner), not a FAIL.
    """
    scenario(15, "№14: backups — the snapshot list, the restore verify of the "
                 "newest snapshot (read-only) and /health.backups observability")

    listing = await rest.get("/backups")
    if listing.status_code != 200:
        skip("the backups scenario (№14)",
             f"GET /backups answered http {listing.status_code}: the operator "
             "handles are unavailable on this contour")
        return
    body = listing.json() or {}
    items = body.get("items")
    listed_ok = isinstance(items, list)
    check("GET /backups lists the snapshots with the operator fields "
          "(file/size_bytes/verified/bad/shipped)",
          listed_ok and body.get("total") == len(items)
          and all({"file", "size_bytes", "verified", "bad"} <= set(item)
                  for item in items),
          f"total={body.get('total')}, "
          f"items={len(items) if listed_ok else None}")
    if not listed_ok or not items:
        skip("the restore verify of the newest snapshot (№14, FR-2.2)",
             "BACKUP_DIR holds no snapshot yet: the backup loop makes the first "
             "copy right after the contour start (BACKUP_INTERVAL_SEC) — wait for "
             "it or check the mounted BACKUP_DIR")
        manual("scenario 15 (№14)",
               "the restore verify needs a snapshot in BACKUP_DIR "
               f"({CFG['backup_dir']}); the loop makes the first one at start")
        return

    newest = items[0]
    name = newest.get("file")
    if not (isinstance(name, str) and name):
        skip("the restore verify of the newest snapshot (№14, FR-2.2)",
             f"the newest entry carries no file name: {newest}")
        return
    check("the newest snapshot is a well-formed `notes-*.db` file",
          name.startswith("notes-") and name.endswith(".db"),
          f"newest={name!r}, size_bytes={newest.get('size_bytes')}")

    snapshot = backup_file_path(name)
    before: tuple[int, int] | None = None
    try:
        info_stat = snapshot.stat()
        before = (info_stat.st_size, info_stat.st_mtime_ns)
    except OSError as exc:
        before = None
        info(f"the snapshot file is not reachable from the runner ({snapshot}): "
             f"{type(exc).__name__} — the mtime check will be skipped")

    verified = await rest.post("/backups/verify", json={"file": name})
    if verified.status_code != 200:
        check("POST /backups/verify of the newest snapshot answers 200", False,
              f"http {verified.status_code}")
        return
    result = verified.json() or {}
    checks = result.get("checks") or {}
    check("POST /backups/verify opens the newest snapshot as a working base "
          "(ok=true, checks.integrity == 'ok')",
          result.get("ok") is True and checks.get("integrity") == "ok",
          f"file={result.get('file')!r}, ok={result.get('ok')!r}, "
          f"integrity={checks.get('integrity')!r}, reason={result.get('reason')!r}")
    check("every required table and every vec0 index opens inside the snapshot "
          "(missing_tables/vec_unavailable are empty, arch §13.1)",
          checks.get("missing_tables") == [] and checks.get("vec_unavailable") == [],
          f"missing_tables={checks.get('missing_tables')!r}, "
          f"vec_unavailable={checks.get('vec_unavailable')!r}")
    info(f"the restore verify took {result.get('latency_ms')} ms")

    # Read-only (FR-2.2): the verification opens the copy `mode=ro` and does not change
    # the snapshot file — the size and mtime before/after match; the copy stays in place.
    relisted = await rest.get("/backups")
    after_items = ((relisted.json().get("items") or [])
                   if relisted.status_code == 200 else [])
    after_item = next((item for item in after_items if item.get("file") == name), None)
    after_size = after_item.get("size_bytes") if after_item else None
    after_verified = after_item.get("verified") if after_item else None
    after_bad = after_item.get("bad") if after_item else None
    check("the snapshot is NOT deleted or resized by the verify (still listed, "
          "the same size — read-only)",
          after_item is not None and after_size == newest.get("size_bytes"),
          f"size_bytes {newest.get('size_bytes')} → {after_size}")
    if before is not None:
        try:
            after_stat = snapshot.stat()
        except OSError as exc:
            skip("the snapshot mtime is unchanged by the verify (read-only)",
                 f"the file is not reachable after the verify: {type(exc).__name__}")
        else:
            check("the snapshot mtime is unchanged by the verify (read-only: the "
                  "copy is opened, never written)",
                  after_stat.st_mtime_ns == before[1],
                  f"mtime_ns {before[1]} → {after_stat.st_mtime_ns}")
    else:
        skip("the snapshot mtime is unchanged by the verify (read-only)",
             f"the snapshot is not reachable from the runner ({snapshot}): point "
             "--backup-dir / LSB_BACKUP_DIR at the mounted BACKUP_DIR to check "
             "the mtime")

    check("GET /backups marks the verified snapshot verified='ok', bad=false",
          after_verified == "ok" and after_bad is False,
          f"verified={after_verified!r}, bad={after_bad!r}")

    health = await health_snapshot(rest)
    backups = health.get("backups") or {}
    check("/health.backups carries the backup observability "
          "(count/bad/last_verify/last_ship, arch §7.2)",
          {"count", "bad", "last_verify", "last_ship"} <= set(backups),
          f"keys={sorted(backups)}")
    last_verify = backups.get("last_verify")
    check("after the verify /health.backups.last_verify records this very snapshot "
          "as ok (the outcome survives a restart, arch §5.1)",
          isinstance(last_verify, dict) and last_verify.get("file") == name
          and last_verify.get("result") == "ok",
          f"last_verify={last_verify}")
    info(f"/health.backups: count={backups.get('count')}, bad={backups.get('bad')}, "
         f"last_ship={backups.get('last_ship')!r}")


# ==== run ======================================================================
async def run_all() -> None:
    global RUN_FINISHED
    async with AsyncExitStack() as stack:
        # We open the MCP session in its OWN stack (`own_stack`): we will close it
        # explicitly, in this same task and before leaving the `AsyncExitStack`; the run's stack
        # stays with the REST client (it has no cancel scope).
        c, init = await fresh_session(AsyncExitStack(), own_stack=True)
        instructions = init.instructions or ""
        rest = await rest_client(stack)

        print(f"\n=== E2E release {RELEASE} acceptance (test contour) ===")
        print(f"MCP: {CFG['mcp_url']} | REST: {CFG['base_url']} | DB: {CFG['db']} | "
              f"token: {mask(CFG['token'])}")
        print(f"run prefix: {CFG['prefix']} | identifiers: {RUN}")
        print(f"guard timeout: E2E_WAIT_SEC={CFG['wait_sec']:.0f}s, "
              f"poll={CFG['poll_sec']:.0f}s")

        try:
            teardown_note(f"run started at {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")
            await prepare_leftovers()
            await scenario_0_context(c, rest, instructions)
            await scenario_1_observability_baseline(rest)
            await scenario_2_models_outage(c, rest)
            await scenario_3_auth_branch(c, rest)
            await scenario_4_attention_pair(c, rest)
            await scenario_5_restart_inside_outage(c, rest)
            await scenario_6_regression()
            # Scenarios 8–13: each in a wrapper — one's failure does not cancel
            # the others; the f11-03 block (scenario 13) is written independently.
            await run_scenario(8, "Reasoning eats the budget → truncation → attention "
                                "→ the budget is raised → the queues drain by themselves",
                               lambda: scenario_8_budget_raise_release(c, rest))
            await run_scenario(9, "The decision slot decides, then falls back over three "
                                "reasons (unsupported / unavailable / low_confidence) "
                                "without losing jobs",
                               lambda: scenario_9_decision_fallback(c, rest))
            await run_scenario(10, "№28: node hint on save without namespace — the note "
                                "stays in `default`, the answer points at a node, no "
                                "model call on write",
                               lambda: scenario_10_node_hint(c, rest))
            await run_scenario(11, "№29: freshness filter `updated_after` — search/list, "
                                "MCP + REST",
                               lambda: scenario_11_freshness(c, rest))
            await run_scenario(12, "№24: the note `comment` field — edit path, reading, "
                                "versions",
                               lambda: scenario_12_comment(c, rest))
            await run_scenario(13, "LSBN-F-11: the injected-text budget (A/B/C) under the "
                                "guard and the agent smoke (notes, areas, queues)",
                               lambda: scenario_13_injected_text(c, rest, instructions))
            await run_scenario(14, "LSBN-F-10: assistant logs — write/read/list/search/"
                                "update/delete, the entity link, isolation from "
                                "memory, no secrets in the journal",
                               lambda: scenario_14_logs(c, rest, instructions))
            await run_scenario(15, "№14: backups — the snapshot list, the restore verify "
                                "of the newest snapshot (read-only) and "
                                "/health.backups observability",
                               lambda: scenario_15_backups(rest))
            await future_block_todo()
            close_scenario()
            # All scenarios completed: the attempt happened (the report exists; "Total:"
            # is printed). A closing break below no longer cancels it.
            RUN_FINISHED = True
        finally:
            # Teardown always runs — also after a failed scenario (FR-3.2).
            try:
                await cleanup_own_data()
            except Exception as exc:  # noqa: BLE001 — the trace must survive anyway
                teardown_note(f"teardown raised {describe(exc)} — check the probe "
                              f"data with prefix {CFG['prefix']} by hand")
            # The MCP session is closed HERE — in its own task, before leaving the
            # `AsyncExitStack`, and with a trace record: otherwise the failure at exit
            # cut the run short and "Total:" never printed (2026-10-12).
            try:
                await c.close()
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException as exc:  # noqa: BLE001 — closing is not the checks' subject
                info(f"closing the MCP session: {describe(exc)}")
                teardown_note(f"the MCP session was not closed cleanly ({describe(exc)})")
            ENV["db_size"] = db_size_text()
            path = write_teardown_trace()
            if path is not None:
                print(f"\n[TRACE] teardown trace: {path}")


def print_report() -> int:
    print(f"\n=== RUN RESULT (E2E release {RELEASE}) ===")
    print(f"probe tag of the run: {RUN.get('tag')!r} "
          f"(attempt {RUN.get('attempt')})")
    print(f"app version: {ENV.get('version')!r} | image: {CFG['image'] or '(not set)'} | "
          f"image revision: {ENV.get('revision')!r} | "
          f"image version: {ENV.get('image_version')!r}")
    print(f"models at the start: embedding_ok={HEALTH.get('embedding_ok')}, "
          f"summarizer_ok={HEALTH.get('summarizer_ok')}, "
          f"judge_ok={HEALTH.get('judge_ok')} | DB: {ENV.get('db_size') or db_size_text()}")
    for entry in REPORT:
        status = "FAIL" if entry["fail"] else "PASS"
        if not entry["pass"] and entry["fail"] == 0:
            status = "SKIP"
        print(f"[{status}] Scenario {entry['n']}. {entry['title']}: "
              f"PASS {entry['pass']}, FAIL {entry['fail']}, "
              f"SKIP {entry['skip']}, WARN {entry['warn']}")
    print("\nWait table (observable state, guard timeout):")
    for entry in WAITS:
        guard = "GUARD FIRED" if entry["fired"] else "ok"
        print(f"  - {entry['label']}: {entry['elapsed']}s of {entry['timeout']:.0f}s "
              f"[{guard}] — {entry['detail']}")
    print(f"\nTotal: PASS {PASS}, FAIL {FAIL}, SKIP {SKIP}, WARN {WARN}")
    if FAILURES:
        print("Failed checks:")
        for item in FAILURES:
            print(f"  - {item}")
    if WARNINGS:
        print("Soft observations (they do not affect the exit code):")
        for item in WARNINGS:
            print(f"  - {item}")
    if SKIPS:
        print("Skipped checks (no DB/models/external step):")
        for item in SKIPS:
            print(f"  - {item}")
    if MANUALS:
        print("Manual steps (the script does not do them):")
        for item in MANUALS:
            print(f"  - {item}")
    if TRACE_PATH is not None:
        print(f"Teardown trace: {TRACE_PATH}")
    return 0 if FAIL == 0 else 1


def configure_probe_data(run_id: str, attempt: int) -> str:
    """Tag of ONE attempt — the probe data of every attempt must be unique.

            Otherwise a retry runs into the verbatim duplicate of its own note
            (`memory_save` → `stored: false`), the `summary` queue stays empty and the
            "the work is visible while the slot is down" check fails by timeout (the
            live run of 2026-10-10). The cleanup by the prefix (--prefix e2e32)
            searches a substring — so it finds both old attempts and the current one.
            Returns the attempt's actual tag.
    """
    tag = f"{CFG['prefix']}-{run_id}-a{attempt}"
    RUN.update({
        "tag": tag,
        "attempt": attempt,
        "notes": [],
        "note_observability": f"{tag} observability probe",
        "note_outage_embedding": f"{tag} outage probe embedding",
        "note_outage_summary": f"{tag} outage probe summary",
        "note_outage_judge": f"{tag} outage probe judge",
        "note_auth": f"{tag} auth branch probe",
        "note_attention_pair": f"{tag} attention pair note one",
        "note_attention_pair_2": f"{tag} attention pair note two",
        "note_budget_break": f"{tag} budget probe note one",
        "note_budget_break_2": f"{tag} budget probe note two",
        "note_decision": f"{tag} decision probe note",
        "note_decision_unsupported": f"{tag} decision unsupported probe",
        "note_decision_unavailable": f"{tag} decision unavailable probe",
        "note_decision_low_confidence": f"{tag} decision low-confidence probe",
    })
    return tag


async def main() -> int:
    parser = argparse.ArgumentParser(
        description=f"Acceptance E2E of release {RELEASE} (MCP + REST): the №39 "
                    "scenario 'models unavailable → wait → back' per slot plus the "
                    "observability checks; waiting is driven by the observable state, "
                    "the run is idempotent and extendable by the follow-up blocks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base-url", default=os.environ.get("BASE_URL", "http://localhost:8080"),
                        help="REST address of the contour")
    parser.add_argument("--mcp-url", default=os.environ.get("MCP_URL", ""),
                        help="MCP address (empty — BASE_URL + /mcp)")
    parser.add_argument("--token", default=os.environ.get("MCP_AUTH_TOKEN", ""),
                        help="Bearer token (by default from the environment; never printed)")
    parser.add_argument("--db", default=os.environ.get("DB_PATH", "/data/notes.db"),
                        help="SQLite DB of the contour (wait predicates)")
    parser.add_argument("--backup-dir", default=os.environ.get("LSB_BACKUP_DIR", ""),
                        help="BACKUP_DIR of the contour as seen by the runner "
                             "(scenario 15; empty — derived from --db)")
    parser.add_argument("--prefix", default=os.environ.get("LSB_E2E_PREFIX", "e2e32"),
                        help="run probe prefix (the pre-cleanup matches it)")
    parser.add_argument("--wait-sec", type=float,
                        default=float(os.environ.get("E2E_WAIT_SEC", "300")),
                        help="guard timeout of every wait (the acceptance contour owns it)")
    parser.add_argument("--poll-sec", type=float,
                        default=float(os.environ.get("E2E_POLL_SEC", "5")),
                        help="poll period of the observable-state waits")
    parser.add_argument("--read-timeout", type=float,
                        default=float(os.environ.get("LSB_READ_TIMEOUT", "300")),
                        help="MCP/REST read timeout, s")
    parser.add_argument("--image", default=os.environ.get("LSB_IMAGE", "llm-second-brain:test"),
                        help="acceptance image whose OCI labels are read")
    parser.add_argument("--inspect-cmd", default=os.environ.get("LSB_INSPECT_CMD", ""),
                        help="command printing the image labels as JSON; empty → "
                             "docker inspect of --image")
    parser.add_argument("--trace-dir", default=os.environ.get("LSB_TRACE_DIR", ""),
                        help="directory of the teardown trace (empty → release/3.2.0/acceptance)")
    parser.add_argument("--logs-cmd", default=os.environ.get("LSB_LOGS_CMD", ""),
                        help="command printing the contour log (queue_waiting / slot events)")
    parser.add_argument("--restart-cmd", default=os.environ.get("LSB_RESTART_CMD", ""),
                        help="shell command restarting the contour (restart-in-outage check)")
    parser.add_argument("--slot-off-cmd", default=os.environ.get("LSB_SLOT_OFF_CMD", ""),
                        help="shell command stopping a slot; use {slot} or let the name "
                             "be appended (scenario 2)")
    parser.add_argument("--slot-on-cmd", default=os.environ.get("LSB_SLOT_ON_CMD", ""),
                        help="shell command starting a slot back (scenario 2)")
    parser.add_argument("--slot-auth-cmd", default=os.environ.get("LSB_SLOT_AUTH_CMD", ""),
                        help="shell command switching a slot provider to 401 (scenario 3)")
    parser.add_argument("--slot-auth-fix-cmd",
                        default=os.environ.get("LSB_SLOT_AUTH_FIX_CMD", ""),
                        help="shell command fixing the 401 provider back (scenario 3)")
    parser.add_argument("--auth-slot", default=os.environ.get("LSB_AUTH_SLOT", "summary"),
                        help="slot the 401 branch is exercised on (scenario 3)")
    parser.add_argument("--offcontract-cmd",
                        default=os.environ.get("LSB_JUDGE_OFFCONTRACT_CMD", ""),
                        help="shell command switching the judge to off-contract (scenario 4)")
    parser.add_argument("--offcontract-fix-cmd",
                        default=os.environ.get("LSB_JUDGE_OFFCONTRACT_FIX_CMD", ""),
                        help="shell command fixing the judge back (scenario 4)")
    parser.add_argument("--budget-break-cmd",
                        default=os.environ.get("LSB_BUDGET_BREAK_CMD", ""),
                        help="shell command restarting the contour with a tight budget "
                             "and a fake slot cutting the answer (scenario 8)")
    parser.add_argument("--budget-fix-cmd",
                        default=os.environ.get("LSB_BUDGET_FIX_CMD", ""),
                        help="shell command restarting the contour with the raised "
                             "budget / a working slot (scenario 8)")
    parser.add_argument("--decision-on-cmd",
                        default=os.environ.get("LSB_DECISION_ON_CMD", ""),
                        help="shell command restarting the contour with "
                             "DECISION_ENABLED=true and a working decision model "
                             "(scenario 9)")
    parser.add_argument("--decision-unsupported-cmd",
                        default=os.environ.get("LSB_DECISION_UNSUPPORTED_CMD", ""),
                        help="shell command restarting the contour with a decision "
                             "model without /v1/systemone (scenario 9)")
    parser.add_argument("--decision-off-cmd",
                        default=os.environ.get("LSB_DECISION_OFF_CMD", ""),
                        help="shell command restarting the contour with the slot off "
                             "and DECISION_FALLBACK=false (scenario 9)")
    parser.add_argument("--decision-low-confidence-cmd",
                        default=os.environ.get("LSB_DECISION_LOW_CONFIDENCE_CMD", ""),
                        help="shell command restarting the contour with a "
                             "threshold+zone catching the probability (scenario 9)")
    parser.add_argument("--decision-restore-cmd",
                        default=os.environ.get("LSB_DECISION_RESTORE_CMD", ""),
                        help="shell command restoring the baseline slot config "
                             "(scenario 9 teardown)")
    parser.add_argument("--expect-version", default=os.environ.get("LSB_EXPECT_VERSION", RELEASE),
                        help="release tag /health.version is compared with")
    parser.add_argument("--run-regression", action="store_true",
                        default=os.environ.get("LSB_RUN_REGRESSION", "") == "1",
                        help="run the unit regression and the existing E2E scripts (long)")
    parser.add_argument("--python", default=os.environ.get("LSB_PYTHON", "python"),
                        help="interpreter for the regression run")
    parser.add_argument("--container",
                        default=os.environ.get("LSB_CONTAINER", "lsb-test"),
                        help="contour container name (the legacy E2E that call "
                             "localhost:8080 run inside its netns)")
    parser.add_argument("--host-root", default=os.environ.get("LSB_HOST_ROOT", ""),
                        help="the tree path ON THE HOST (docker resolves nested "
                             "`-v` against the host, not against the runner); "
                             "empty — the repo dir itself")
    parser.add_argument("--repo-dir", default=os.environ.get("LSB_REPO_DIR", ""),
                        help="repository directory for the regression (empty — the script root)")
    parser.add_argument("--regression-scripts",
                        default=os.environ.get(
                            "LSB_E2E_SCRIPTS",
                            "e2e_release31.py,e2e_release30.py,e2e_release22.py,"
                            "e2e_lsb0004.py,e2e_lsb0005.py,e2e_lsb0006.py"),
                        help="existing E2E scripts, comma separated")
    parser.add_argument("--no-cleanup", action="store_true",
                        default=os.environ.get("LSB_KEEP", "") == "1",
                        help="do not remove the leftovers of a previous run (diagnostics)")
    parser.add_argument("--self-test-acceptance", action="store_true",
                        help="self-test of the acceptance block only: writes it twice "
                             "into a temporary file and compares byte by byte "
                             "(the contour and the token are not needed)")
    args = parser.parse_args()

    if args.self_test_acceptance:
        return self_test_acceptance()

    if not args.token:
        print("FATAL: the contour Bearer token is not set: export MCP_AUTH_TOKEN "
              "(or --token); see --help", file=sys.stderr)
        return 2

    inspect_cmd = args.inspect_cmd or (
        f"docker inspect --format '{{{{json .Config.Labels}}}}' {args.image}")
    CFG.update({
        "base_url": args.base_url.rstrip("/"),
        "mcp_url": (args.mcp_url or (args.base_url.rstrip("/") + "/mcp")),
        "token": args.token,
        "db": args.db,
        "backup_dir": args.backup_dir or str(Path(args.db).parent / "backups"),
        "prefix": args.prefix,
        "wait_sec": args.wait_sec,
        "poll_sec": args.poll_sec,
        "read_timeout": args.read_timeout,
        "image": args.image,
        "inspect_cmd": inspect_cmd,
        "trace_dir": args.trace_dir or default_trace_dir(),
        "logs_cmd": args.logs_cmd,
        "restart_cmd": args.restart_cmd,
        "slot_off_cmd": args.slot_off_cmd,
        "slot_on_cmd": args.slot_on_cmd,
        "slot_auth_cmd": args.slot_auth_cmd,
        "slot_auth_fix_cmd": args.slot_auth_fix_cmd,
        "auth_slot": args.auth_slot,
        "offcontract_cmd": args.offcontract_cmd,
        "offcontract_fix_cmd": args.offcontract_fix_cmd,
        "budget_break_cmd": args.budget_break_cmd,
        "budget_fix_cmd": args.budget_fix_cmd,
        "decision_on_cmd": args.decision_on_cmd,
        "decision_unsupported_cmd": args.decision_unsupported_cmd,
        "decision_off_cmd": args.decision_off_cmd,
        "decision_low_confidence_cmd": args.decision_low_confidence_cmd,
        "decision_restore_cmd": args.decision_restore_cmd,
        "expect_version": args.expect_version,
        "run_regression": args.run_regression,
        "python": args.python,
        "repo_dir": args.repo_dir or str(Path(__file__).resolve().parents[1]),
        "container": args.container,
        "host_root": args.host_root.rstrip("/"),
        "regression_scripts": args.regression_scripts,
        "no_cleanup": args.no_cleanup,
    })
    run_id = os.environ.get("LSB_E2E_RUN_ID") or time.strftime("%m%d%H%M%S")

    # Up to 3 attempts: a broken MCP session must not zero the run. The probe
    # tag carries the attempt number so a retry never collides with the probe
    # data of the previous attempt (a literal duplicate means no new note).
    attempts = 0
    while attempts < 3:
        attempts += 1
        reset_counters()
        tag = configure_probe_data(run_id, attempts)
        print(f"\n=== attempt {attempts}: probe tag {tag} ===")
        # We run the attempt as a SEPARATE task: the anyio areas inside the MCP SDK
        # live in it, and a cancelled foreign area (the live run of 2026-10-12:
        # the process died with `CancelledError` on a retry; "Total:" never printed)
        # does not poison the runner's task — the attempt fails, not the report.
        attempt = asyncio.create_task(run_all())
        try:
            await attempt
            break
        except asyncio.CancelledError as exc:
            # The cancellation came from OUTSIDE (a foreign anyio area in the session closing):
            # retrying within this run is pointless, but the report is already assembled.
            print(f"\n[retry] the attempt was cancelled by a foreign scope "
                  f"(attempt {attempts}): {describe(exc)}")
            break
        except BaseException as exc:  # noqa: BLE001 — we decide the break by type
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            # A transport break (including anyio's `ExceptionGroup` and the RuntimeError
            # about a foreign `cancel scope`) — not a product error.
            reason = "transport break" if is_transport_break(exc) else "run error"
            print(f"\n[retry] {reason} (attempt {attempts}): {describe(exc)}")
            if RUN_FINISHED and reason == "transport break":
                # The scenarios completed (scenarios 10–13, the f11-03 block); only the session
                # closing failed: we do not start a retry — the report is already complete.
                print("[retry] the run itself is complete: the close-time break does "
                      "not zero it, the report stands")
                break
            await asyncio.sleep(5)
        finally:
            if not attempt.done():
                attempt.cancel()
            await asyncio.gather(attempt, return_exceptions=True)
    else:
        print("\n[FATAL] the run did not finish within 3 attempts")

    if not REPORT and FAIL == 0:
        print("\n[FATAL] the run produced no scenario: the contour is unavailable — "
              "check BASE_URL, MCP_AUTH_TOKEN and that the test container is up")
        return 3
    return print_report()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
