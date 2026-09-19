#!/usr/bin/env python3
"""Acceptance E2E of release 3.1.0 (test contour lsb-test, MCP + REST).

Spec: `release/3.1.0/staging_for_development/techdebt-0036-01-e2e-hygiene.md`;
requirements — `documents/req/techdebt-0036-e2e-hygiene.md` (FR-1..FR-5),
architecture — `documents/arch/3.1.0-release-architecture.md` §9.

What makes it different from `e2e_release30.py`:

* waiting is driven by an OBSERVABLE state (`/health.pending_*`,
  `/health.queues`, `vector_status`/`summary_status`/`links_at`/`node_order_at`
  in the DB), never by a fixed pause; one guard timeout `E2E_WAIT_SEC`
  (default 300) protects every wait. The actual wait time and whether the guard
  fired are printed per stage: a slow model reads as "we waited", not as a
  false FAIL (FR-1.2/FR-1.3);
* the run is idempotent: probe data is marked by the run prefix (in titles) and
  by its own namespace, and is removed both at the start (leftovers of a
  previous run) and at the end (its own); no check depends on leftovers
  (FR-2.1/FR-2.2);
* teardown always writes a trace "what was removed / what was left on purpose"
  to `release/3.1.0/acceptance/teardown-<timestamp>.log` — also on an emergency
  exit, so the next run can still be started (FR-3.1/FR-3.2);
* the report shows the image revision/version (OCI labels of the acceptance
  image), the app version from `/health`, model availability, the DB volume and
  the wait table (FR-1.3/FR-2.2/FR-4.1);
* the mandatory scenario of lsb-0014 FR-2.4: models unavailable → jobs wait →
  models are back → jobs finish WITHOUT a container restart (scenario 4).

Scenarios (0-9), per the techdebt-0036-01 spec:

  0. MCP surface and contour readiness: 21 tools (8 memory + 5 skills +
     5 user + 3 terms), `/health` shape, model slots, image revision, DB size.
  1. Listing limits and pagination: both MCP listings page by 20, `limit=50` is
     a soft refusal, `total`/`has_more`/`next_offset`/`next_cursor` and exactly
     one "+N more" hint; REST pages up to 50.
  2. `chars` is consistent between `memory_search` / `memory_list` /
     `memory_get`; the chunk mode keeps its old meaning (sum of the served
     chunks).
  3. Links: level 0 immediately, level 1 after the job (and it wins over
     level 0), chunk reads carry links, batch reads do not, soft-deleted notes
     are never served, the own namespace is cut off.
  4. Background jobs: models unavailable → `pending` and its age grow
     (`/health.queues`), the `queue_waiting` event appears in the log → models
     are back → jobs finish with NO container restart.
  5. Node order: the accumulated `default` is drained (fast path plus the
     classifier within its budget), a processed note is not processed twice,
     the queue goes back to empty.
  6. `/health` = 7 previous fields + `queues` + `version`.
  7. Context budgets: `memory_search` top_k=5 ≤ 1.2 KB, `memory_list` ≤ 1.5 KB,
     the links overhead ≤ 0.5 KB.
  8. Live DB upgrade v3.0.0 → 3.1.0: the `links` table and the
     `links_at`/`node_order_at` columns are there, notes are intact, a repeated
     start is a no-op, and an MCP session comes up again.
  9. Regression: unit regression plus the existing E2E scripts — only with
     `--run-regression` (otherwise printed as MANUAL).

Run (from ~/projects/llm-second-brain/test, normally inside the test container):
    MCP_AUTH_TOKEN=... python scripts/e2e_release31.py
    python scripts/e2e_release31.py --help

Configuration comes from environment variables only (defaults are for the
lsb-test contour); no secret ever reaches the code or the output — the token is
printed masked.
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
import time
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, Awaitable, Callable

import httpx2
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.exceptions import MCPError

RELEASE = "3.1.0"

# --- canon constants (lsb-0013/0014, REQUIREMENTS §4.2.6) --------------------
MCP_LIMIT = 20          # MCP listing ceiling; REST ceiling is 50
REST_LIMIT = 50
MCP_TOOLS_TOTAL = 21    # 8 memory + 5 skills + 5 user + 3 terms
MCP_TOOL_GROUPS = {"memory_": 8, "skills_": 5, "user_": 5, "terms_": 3}
HEALTH_LEGACY = ("status", "embedding_ok", "summarizer_ok", "judge_ok",
                 "notes_count", "pending_vector", "pending_summary")
HEALTH_QUEUES = ("vector", "summary", "judge", "areas", "links", "nodes")
BUDGET_SEARCH = 1200    # bytes of the compact memory_search answer (top_k=5)
BUDGET_LIST = 1500      # bytes of one memory_list page
BUDGET_LINKS = 500      # bytes of the links array of one note
MORE_HINT = re.compile(r"^\+(\d+) more — offset=(\d+)$")
LINK_ITEM_FIELDS = {"id", "title", "namespace", "chars"}
UPGRADE_TABLES = ("links",)
UPGRADE_COLUMNS = ("links_at", "node_order_at")

# --- run configuration (filled in main) --------------------------------------
CFG: dict[str, Any] = {}
RUN: dict[str, Any] = {}
HEALTH: dict[str, Any] = {}
ENV: dict[str, Any] = {}
WAITS: list[dict[str, Any]] = []
TEARDOWN: list[str] = []
TRACE_PATH: Path | None = None

# --- counters and report ------------------------------------------------------
PASS = FAIL = SKIP = WARN = 0
FAILURES: list[str] = []
WARNINGS: list[str] = []
SKIPS: list[str] = []
MANUALS: list[str] = []
SCENARIO: dict[str, Any] = {}
REPORT: list[dict[str, Any]] = []


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


def check_llm(name: str, ok: bool, detail: str = "") -> None:
    """Check that needs the embedding slot: soft (WARN) when it is unavailable.

    Cosine-driven checks (dedup, anti-synonymy, level-0 links) cannot succeed
    without an embedder — a FAIL there would be false. Everything computed
    without the models stays strict.
    """
    if not ok and HEALTH.get("embedding_ok") is False:
        warn(f"{name} (embedder is unavailable — the check is soft)", detail)
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
    global WAITS, TEARDOWN
    PASS = FAIL = SKIP = WARN = 0
    FAILURES, WARNINGS, SKIPS, MANUALS = [], [], [], []
    SCENARIO, REPORT = {}, []
    WAITS, TEARDOWN = [], []


def mask(token: str) -> str:
    """Mask a secret for the output: 4 characters + ***."""
    return f"{token[:4]}***" if token else "(not set)"


def describe(exc: BaseException) -> str:
    """Readable error text: ExceptionGroup → its first inner exception."""
    if isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        inner = exc.exceptions[0]
        return f"{type(inner).__name__}: {inner}"
    return f"{type(exc).__name__}: {exc}"


def json_size(obj: Any) -> int:
    """Size of the compact answer in bytes (the canon budget metric)."""
    return len(json.dumps(obj, ensure_ascii=False).encode("utf-8"))


# --- teardown trace (FR-3.1/FR-3.2) -------------------------------------------

def teardown_note(line: str) -> None:
    """Append one line to the teardown trace (also on an emergency exit)."""
    TEARDOWN.append(line)


def default_trace_dir() -> str:
    """`release/3.1.0/acceptance` next to the repository, else the CWD.

    Inside the container the release/ tree is not mounted, so the operator
    points LSB_TRACE_DIR / --trace-dir at a bound path; the written path is
    always printed in the report.
    """
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


# --- MCP client --------------------------------------------------------------

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
    """Thin wrapper over ClientSession: a tool call → dict.

    `stack` (optional) is the session's own AsyncExitStack — needed when the
    session must be closed BEFORE the end of the run (the restart step).
    """

    def __init__(self, session: ClientSession, stack: AsyncExitStack | None = None):
        self.session = session
        self._stack = stack

    async def call(self, tool: str, args: dict) -> dict:
        return extract(await self.session.call_tool(tool, args))

    async def close(self) -> None:
        """Close the session and its transport: idempotent, never fatal."""
        stack, self._stack = self._stack, None
        if stack is None:
            return
        try:
            await stack.aclose()
        except Exception as exc:  # noqa: BLE001 — closing is not a subject of checks
            info(f"closing the MCP session: {describe(exc)}")


async def fresh_session(stack: AsyncExitStack, url: str | None = None,
                        token: str | None = None,
                        own_stack: bool = False) -> tuple[Client, Any]:
    """New MCP session with its own `initialize` (instructions are rebuilt).

    ClientSession caches `initialize()` (SDK mcp 2.x), so only a new connection
    produces a new handshake. Timeouts follow e2e_release22.py (30s
    connect/write/pool, long read): calls with synchronous embedding under the
    background worker must not be cut by the default 5s read.
    """
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


# --- observable-state waiting (FR-1.2/FR-1.3) ---------------------------------

async def wait_until(predicate: Callable[[], Awaitable[tuple[bool, str]]],
                     label: str, *,
                     timeout: float | None = None,
                     poll: float | None = None) -> tuple[bool, str]:
    """Wait for a state to become observable; the guard timeout is `E2E_WAIT_SEC`.

    `predicate` returns (done, detail). The actual wait and whether the guard
    fired are recorded in WAITS and printed in the report, so a slow model is
    visible as "we waited N s" while an unreachable one hits the guard.
    """
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


# --- direct DB access (SELECT only) ------------------------------------------

def db_ready() -> bool:
    return bool(CFG.get("db")) and Path(CFG["db"]).exists()


def _load_vec_extension(conn: sqlite3.Connection) -> None:
    """Load sqlite-vec into the script's connection (as app.storage.db does)."""
    try:
        import sqlite_vec
    except ImportError:
        return
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
    except (AttributeError, sqlite3.Error):
        return


def db_try(sql: str, params: tuple = ()) -> tuple[list[dict] | None, str]:
    """Read the contour DB: (rows | None, error text). Read-only, SELECT only."""
    path = str(CFG["db"])
    last = "unknown error"
    for connect in (
        lambda: sqlite3.connect(f"file:{path}?mode=ro", uri=True),
        lambda: sqlite3.connect(path),
    ):
        try:
            conn = connect()
        except sqlite3.Error as exc:
            last = f"{type(exc).__name__}: {exc}"
            continue
        try:
            conn.row_factory = sqlite3.Row
            _load_vec_extension(conn)
            return [dict(row) for row in conn.execute(sql, params).fetchall()], ""
        except sqlite3.Error as exc:
            last = f"{type(exc).__name__}: {exc}"
        finally:
            conn.close()
    return None, last


def db_scalar(sql: str, params: tuple = ()) -> tuple[Any, str]:
    rows, err = db_try(sql, params)
    if rows is None:
        return None, err
    if not rows:
        return None, "no rows"
    return list(rows[0].values())[0], ""


def db_cell(table: str, row_id: int, column: str, *,
            equals: Any = None, not_null: bool = False,
            ) -> Callable[[], Awaitable[tuple[bool, str]]]:
    """Predicate over one DB cell — an observable state, not a fixed pause.

    `table`/`column` are module constants only (never user input), so the f-string
    below cannot be an injection vector.
    """

    async def predicate() -> tuple[bool, str]:
        value, err = db_scalar(f"SELECT {column} FROM {table} WHERE id = ?", (row_id,))
        if err:
            return False, err
        if not_null:
            return value is not None, f"{column}={value!r}"
        return value == equals, f"{column}={value!r}"

    return predicate


def contains(haystack: Any, needle: str) -> bool:
    """Is the substring present in an arbitrary value (dict/list → JSON)."""
    if isinstance(haystack, str):
        return needle in haystack
    return needle in json.dumps(haystack, ensure_ascii=False)


# --- /health ------------------------------------------------------------------

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


def queue_empty(rest: httpx2.AsyncClient,
                queue: str) -> Callable[[], Awaitable[tuple[bool, str]]]:
    """Predicate: the queue of a job is empty (pending 0, no oldest age)."""

    async def predicate() -> tuple[bool, str]:
        snapshot = await health_snapshot(rest)
        stat = queue_stat(snapshot, queue)
        if stat is None:
            return False, f"queue {queue} is not reported by /health"
        done = stat.get("pending") == 0 and stat.get("oldest_pending_sec") in (0, None)
        return done, f"queue {queue}: pending={stat.get('pending')}, " \
                     f"oldest={stat.get('oldest_pending_sec')}"

    return predicate


def queue_age_at_least(rest: httpx2.AsyncClient, queue: str, seconds: float,
                       ) -> Callable[[], Awaitable[tuple[bool, str]]]:
    """Predicate: the age of the oldest pending item reached `seconds`."""

    async def predicate() -> tuple[bool, str]:
        snapshot = await health_snapshot(rest)
        stat = queue_stat(snapshot, queue)
        if stat is None:
            return False, f"queue {queue} is not reported by /health"
        age = stat.get("oldest_pending_sec")
        return bool(age and age >= seconds), \
            f"queue {queue}: pending={stat.get('pending')}, oldest={age}"

    return predicate


def legacy_pending_at_least(rest: httpx2.AsyncClient, field: str, value: int,
                            ) -> Callable[[], Awaitable[tuple[bool, str]]]:
    """Predicate: a legacy /health counter (pending_vector/pending_summary) ≥ value."""

    async def predicate() -> tuple[bool, str]:
        snapshot = await health_snapshot(rest)
        got = snapshot.get(field)
        return isinstance(got, int) and got >= value, f"{field}={got}"

    return predicate


# --- listing helpers (MCP ceiling is 20 since 3.1.0) ---------------------------

async def list_page(c: Client, *, detail: str = "summaries", limit: int = MCP_LIMIT,
                    offset: int = 0, namespace: str | None = None) -> dict:
    """One page of memory_list (the ceiling is passed explicitly: 20)."""
    args: dict[str, Any] = {"limit": limit, "offset": offset, "detail": detail}
    if namespace is not None:
        args["namespace"] = namespace
    return await c.call("memory_list", args)


async def list_all(c: Client, *, detail: str = "summaries",
                   max_pages: int = 25) -> list[dict]:
    """Walk every memory_list page (offset by `next_offset` until `has_more`)."""
    items: list[dict] = []
    offset = 0
    for _ in range(max_pages):
        page = await list_page(c, detail=detail, offset=offset)
        items.extend(page.get("items", []))
        nxt = page.get("next_offset")
        if not page.get("has_more") or not isinstance(nxt, int):
            break
        offset = nxt
    return items


# --- external steps (restart, model slots, logs, image labels) -----------------

async def run_shell(cmd: str, timeout: float = 600.0) -> subprocess.CompletedProcess[str]:
    """Run an operator-provided contour hook (never echoed into the output)."""
    return await asyncio.to_thread(
        subprocess.run, cmd, shell=True, capture_output=True, text=True, timeout=timeout
    )


async def image_labels() -> dict[str, str]:
    """OCI labels of the acceptance image — proves which code was checked (FR-4.1)."""
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


def db_size_text() -> str:
    if not db_ready():
        return "unknown (DB is not reachable)"
    size = Path(CFG["db"]).stat().st_size
    return f"{size} B ({size / 1048576:.1f} MiB)"


# --- probe data: marking, leftovers, teardown ---------------------------------

async def probe_titles(c: Client, prefix: str) -> list[dict]:
    """Probe notes of one run (matched by the title prefix), over ALL pages."""
    return [item for item in await list_all(c, detail="titles")
            if prefix in (item.get("title") or "")]


async def prepare_leftovers() -> None:
    """Remove the leftovers of a previous run (idempotency, FR-2.1).

    A repeated run must not trip over its own dedup/anti-synonymy, so the probe
    data of every earlier run (the same prefix) is deleted before the scenarios.
    `--no-cleanup` / LSB_KEEP=1 turns this off for diagnostics.
    """
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
            warn("pre-cleanup was not performed", describe(exc))
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


async def cleanup_namespaces(rest: httpx2.AsyncClient) -> None:
    """Remove the run's probe namespaces (children first, then the root)."""
    root = RUN["ns"]
    for path in (f"{root}/a", f"{root}/b", f"{root}/same", root):
        try:
            r = await rest.delete(f"/namespaces/{path}")
        except Exception as exc:  # noqa: BLE001
            teardown_note(f"namespace {path} was NOT removed ({describe(exc)})")
            continue
        if r.status_code in (200, 204, 404):
            teardown_note(f"removed namespace {path} (http {r.status_code})")
        else:
            teardown_note(f"namespace {path} was left on purpose (http "
                          f"{r.status_code}) — an operator step is needed")


# --- scenario 0: surface and readiness ---------------------------------------

async def scenario_0_context(c: Client, rest: httpx2.AsyncClient,
                             instructions: str) -> None:
    scenario(0, "MCP surface and contour readiness (version, revision, models, DB)")

    r = await rest.get("/health")
    check("/health answers without a token (200)", r.status_code == 200,
          f"status={r.status_code}")
    health = r.json() if r.status_code == 200 else {}
    HEALTH.update(health)
    check("/health: status = ok", health.get("status") == "ok",
          f"status={health.get('status')}")
    ENV["version"] = health.get("version")

    tools = (await c.session.list_tools()).tools
    check(f"MCP surface: {MCP_TOOLS_TOTAL} tools (8 memory + 5 skills + 5 user + 3 terms)",
          len(tools) == MCP_TOOLS_TOTAL, f"n={len(tools)}")
    names = {tool.name for tool in tools}
    for group, prefix in (("memory", "memory_"), ("skills", "skills_"),
                          ("user", "user_"), ("terms", "terms_")):
        got = sorted(n for n in names if n.startswith(prefix))
        need = MCP_TOOL_GROUPS[prefix]
        check(f"surface {group}_*: {need} tools", len(got) == need,
              f"n={len(got)}: {got}")
    check("instructions are not empty", bool(instructions.strip()),
          f"len={len(instructions)}")

    labels = await image_labels()
    revision = labels.get("org.opencontainers.image.revision")
    version_label = labels.get("org.opencontainers.image.version")
    ENV["revision"] = revision
    ENV["image_version"] = version_label
    if labels:
        check("the acceptance image carries the git revision label (FR-4.1)",
              bool(revision) and revision != "unknown", f"revision={revision}")
        check("the acceptance image carries the version label (FR-4.1)",
              bool(version_label) and version_label != "unknown",
              f"version={version_label}")
    else:
        skip("the acceptance image carries the git revision label (FR-4.1)",
             "docker inspect is unavailable here — run --inspect-cmd / "
             "LSB_INSPECT_CMD on the docker host, or read the labels by hand")
        manual("image revision label",
               f"docker inspect --format '{{{{json .Config.Labels}}}}' {CFG['image']}")
    check("/health reports the app version (FR-4.2)",
          isinstance(health.get("version"), str) and bool(health.get("version")),
          f"version={health.get('version')!r}")

    info(f"app version: {ENV['version']!r} | image: {CFG['image'] or '(not set)'} | "
         f"revision: {ENV['revision']!r} | image version: {ENV['image_version']!r}")
    info(f"models: embedding_ok={health.get('embedding_ok')}, "
         f"summarizer_ok={health.get('summarizer_ok')}, judge_ok={health.get('judge_ok')}")
    info(f"DB: {CFG['db']} — {db_size_text()}, notes_count={health.get('notes_count')}, "
         f"pending_vector={health.get('pending_vector')}, "
         f"pending_summary={health.get('pending_summary')}")


# --- scenario 1: limits and pagination ---------------------------------------

async def scenario_1_limits(c: Client, rest: httpx2.AsyncClient) -> None:
    scenario(1, "Listing limits and pagination (MCP 20, soft refusal at 50, REST up to 50)")

    page = await list_page(c)  # default page
    items = page.get("items", [])
    check("memory_list default page holds at most 20 records", len(items) <= MCP_LIMIT,
          f"n={len(items)}")
    check("memory_list carries the page fields",
          {"total", "has_more", "next_offset", "next_cursor"} <= set(page),
          f"keys={sorted(page)}")
    check("total is an integer", isinstance(page.get("total"), int),
          f"total={page.get('total')!r}")
    check("next_cursor is reserved as null (keyset pagination is not introduced)",
          page.get("next_cursor") is None, f"next_cursor={page.get('next_cursor')!r}")

    hint = page.get("hint") or ""
    if page.get("has_more"):
        m = MORE_HINT.match(hint)
        check("the '+N more' hint is exactly one and well formed", bool(m),
              f"hint={hint!r}")
        if m:
            check("the hint reports the real remainder and offset",
                  int(m.group(1)) == page["total"] - page["next_offset"]
                  and int(m.group(2)) == page["next_offset"],
                  f"n={m.group(1)}, offset={m.group(2)}, total={page['total']}, "
                  f"next_offset={page['next_offset']}")
            check("next_offset equals the number of returned records",
                  page["next_offset"] == len(items),
                  f"next_offset={page['next_offset']}, n={len(items)}")
        second = await list_page(c, offset=page["next_offset"])
        check("the page by next_offset is not empty and does not repeat ids",
              bool(second.get("items"))
              and not ({i["id"] for i in second["items"]} & {i["id"] for i in items}),
              f"n={len(second.get('items', []))}")
    else:
        skip("the '+N more' hint", "the base has one page of notes — no remainder")

    refusal = await list_page(c, limit=REST_LIMIT)
    check("memory_list limit=50 is a soft refusal (no schema error)",
          refusal.get("items") == [] and "hint" in refusal,
          f"keys={sorted(refusal)}")
    check("the refusal text is the service one: 'limit: expected 1..20, got 50'",
          refusal.get("hint") == f"limit: expected 1..{MCP_LIMIT}, got {REST_LIMIT}",
          f"hint={refusal.get('hint')!r}")
    boundary = await list_page(c, limit=MCP_LIMIT + 1)
    check("limit just above the ceiling is refused with the same text",
          boundary.get("items") == []
          and boundary.get("hint") == f"limit: expected 1..{MCP_LIMIT}, got {MCP_LIMIT + 1}",
          f"hint={boundary.get('hint')!r}")

    skills = await c.call("skills_list", {})
    check("skills_list carries the same page fields",
          {"items", "total", "has_more", "next_offset", "next_cursor"} <= set(skills),
          f"keys={sorted(skills)}")
    check("skills_list default page holds at most 20 records",
          len(skills.get("items", [])) <= MCP_LIMIT, f"n={len(skills.get('items', []))}")
    skills_refusal = await c.call("skills_list", {"limit": REST_LIMIT})
    check("skills_list limit=50 is a soft refusal",
          skills_refusal.get("items") == []
          and skills_refusal.get("hint") == f"limit: expected 1..{MCP_LIMIT}, got {REST_LIMIT}",
          f"hint={skills_refusal.get('hint')!r}")

    r = await rest.get("/notes", params={"limit": REST_LIMIT})
    body = r.json() if r.status_code == 200 else {}
    check("REST GET /notes?limit=50 is served (the REST ceiling is 50)",
          r.status_code == 200, f"status={r.status_code}")
    check("REST /notes carries the page fields", 
          {"total", "has_more", "next_offset"} <= set(body), f"keys={sorted(body)}")
    check("REST /notes returns no more than 50 records",
          len(body.get("items", [])) <= REST_LIMIT, f"n={len(body.get('items', []))}")
    r = await rest.get("/notes", params={"limit": REST_LIMIT + 1})
    check("REST /notes above the ceiling is refused (422, soft refusal text)",
          r.status_code == 422, f"status={r.status_code}")
    r = await rest.get("/skills", params={"limit": REST_LIMIT})
    check("REST GET /skills?limit=50 is served", r.status_code == 200,
          f"status={r.status_code}")


# --- scenario 2: chars --------------------------------------------------------

async def scenario_2_chars(c: Client, rest: httpx2.AsyncClient) -> int | None:
    scenario(2, "chars is consistent between memory_search / memory_list / memory_get")
    text = (f"{RUN['tag']} chars probe: the release acceptance checks the note volume "
            "field across the compact outputs.")
    saved = await c.call("memory_save", {"text": text, "title": RUN["note_chars"],
                                         "namespace": RUN["ns"]})
    nid = saved.get("id")
    check("probe note for chars is created", nid is not None, f"id={nid}")
    if nid is None:
        return None
    RUN["notes"].append(nid)

    got = (await c.call("memory_get", {"id": nid})).get("notes", [{}])
    full = got[0] if got else {}
    check("memory_get: chars equals the full text length",
          full.get("chars") == len(text), f"chars={full.get('chars')}, len={len(text)}")

    found = [i for i in await list_all(c, detail="summaries") if i.get("id") == nid]
    check("memory_list: chars of the same note equals the full text length",
          bool(found) and found[0].get("chars") == len(text),
          f"chars={found[0].get('chars') if found else None}, len={len(text)}")

    res = await c.call("memory_search", {"query": RUN["tag"], "top_k": 5})
    hits = [h for h in res.get("results", []) if h.get("id") == nid]
    if not hits and HEALTH.get("embedding_ok") is False:
        warn("memory_search: chars of the probe note",
             "the embedder is unavailable — the probe note is not in the semantic answer")
    else:
        check("memory_search: chars of the same note equals the full text length",
              bool(hits) and hits[0].get("chars") == len(text),
              f"chars={hits[0].get('chars') if hits else None}, len={len(text)}")

    long_text = " ".join(
        f"Paragraph {i} about the distributed cache, consistent hashing and replica "
        "placement across nodes, with eviction policies and cache invalidation."
        for i in range(60)
    )
    saved = await c.call("memory_save", {"text": long_text, "title": RUN["note_long"],
                                         "namespace": RUN["ns"]})
    long_id = saved.get("id")
    check("long probe note for the chunk mode is created", long_id is not None,
          f"id={long_id}")
    if long_id is None:
        return nid
    RUN["notes"].append(long_id)

    if not db_ready():
        skip("the chunk mode keeps its own chars meaning",
             f"the DB is not reachable ({CFG['db']}) — run the script inside the container")
        return nid

    async def chunked() -> tuple[bool, str]:
        chunk, _err = db_scalar("SELECT total_chunks FROM notes WHERE id = ?", (long_id,))
        return bool(chunk and chunk > 1), f"total_chunks={chunk}"

    ok, detail = await wait_until(chunked, "the long note is split into chunks")
    if not ok:
        skip("the chunk mode keeps its own chars meaning",
             f"the note is not chunked yet ({detail})")
        return nid
    r = await c.call("memory_get", {"id": long_id, "chunk": 0})
    chunks = r.get("chunks", [])
    check("memory_get chunk=0 returns one chunk", len(chunks) == 1,
          f"n={len(chunks)}")
    check("in the chunk mode chars is the sum of the served chunks (lsb-0003)",
          bool(chunks) and r.get("chars") == sum(len(ch.get("text") or "") for ch in chunks),
          f"chars={r.get('chars')}, sum={sum(len(ch.get('text') or '') for ch in chunks)}")
    check("in the chunk mode chars is NOT the full text length (the old meaning holds)",
          r.get("chars") != len(long_text),
          f"chars={r.get('chars')}, full={len(long_text)}")
    return nid


# --- scenario 3: links --------------------------------------------------------

async def scenario_3_links(c: Client, rest: httpx2.AsyncClient) -> int | None:
    scenario(3, "Links: level 0 at once, level 1 after the job, chunk/batch/soft-delete rules")

    ns_a, ns_b, ns_same = f"{RUN['ns']}/a", f"{RUN['ns']}/b", f"{RUN['ns']}/same"
    created = await c.call("memory_namespace_create", {
        "path": RUN["ns"], "description": f"Acceptance probe nodes of run {RUN['tag']}.",
    })
    check("the run namespace is created (confirmed)", created.get("created") is True,
          f"path={created.get('path')}")
    for path, desc in ((ns_a, f"Acceptance probe area A of run {RUN['tag']}."),
                       (ns_b, f"Acceptance probe area B of run {RUN['tag']}."),
                       (ns_same, f"Acceptance probe area C of run {RUN['tag']}.")):
        res = await c.call("memory_namespace_create", {"path": path, "description": desc})
        check(f"probe namespace {path} is created", res.get("created") is True,
              f"status={res.get('status')}")

    topic = (f"{RUN['tag']} vector index rebuild keeps the ranking stable while the "
             "embedding model queues requests one by one.")
    other = f"Another unrelated acceptance note of the same run {RUN['tag']}."
    same = f"{topic} — the twin note of the same namespace."

    sa = await c.call("memory_save", {"text": topic, "title": RUN["note_links_a"],
                                      "namespace": ns_a})
    sb = await c.call("memory_save", {"text": topic, "title": RUN["note_links_b"],
                                      "namespace": ns_b})
    ss = await c.call("memory_save", {"text": same, "title": RUN["note_links_same"],
                                      "namespace": ns_same})
    so = await c.call("memory_save", {"text": other, "title": RUN["note_other"],
                                      "namespace": ns_b})
    ids = [x.get("id") for x in (sa, sb, ss, so)]
    check("the link probe notes are created", all(i is not None for i in ids), f"ids={ids}")
    if any(i is None for i in ids):
        return
    RUN["notes"].extend(ids)
    na, nb, nsame, nother = ids

    got = await c.call("memory_get", {"id": na})
    links = got.get("links", [])
    check("memory_get carries a links field (single read)", "links" in got,
          f"keys={sorted(got)}")
    check("a related note from another namespace is served at once (level 0)",
          any(link.get("id") == nb for link in links), f"links={links}")
    check("link items carry {id, title, namespace, chars}",
          all(set(link) == LINK_ITEM_FIELDS for link in links),
          f"fields={[sorted(link) for link in links]}")
    check("the own namespace is cut off when serving",
          all(link.get("namespace") != ns_a for link in links),
          f"namespaces={[link.get('namespace') for link in links]}")
    check("the note is not linked to itself", all(link.get("id") != na for link in links))

    if not db_ready():
        skip("level 1 after the links job", f"the DB is not reachable ({CFG['db']})")
    else:
        async def vectorized() -> tuple[bool, str]:
            value, _err = db_scalar("SELECT vector_status FROM notes WHERE id = ?", (na,))
            return value == "ok", f"vector_status={value!r}"

        ok, detail = await wait_until(vectorized, "the probe note is vectorized")
        if not ok:
            skip("level 1 after the links job",
                 f"the vector is not ready within the guard ({detail})")
        else:
            ok, detail = await wait_until(db_cell("notes", na, "links_at", not_null=True),
                                          "the links job marked the note (links_at)")
            check("the links job processed the probe note (links_at is set)",
                  ok, detail)
            if ok:
                rows, err = db_try(
                    "SELECT note_a, note_b, kind, score FROM links "
                    "WHERE (note_a = ? OR note_b = ?)",
                    (na, na),
                )
                check("the links table holds the pair with a kind",
                      bool(rows) and all(r["kind"] in ("mention", "entities", "cosine")
                                         for r in rows or []),
                      f"err={err}, rows={rows}")
                check("the stored pair carries a score",
                      bool(rows) and rows[0].get("score") is not None,
                      f"score={rows[0].get('score') if rows else None}")
                got = await c.call("memory_get", {"id": na})
                links = got.get("links", [])
                stored_ids = {
                    (r["note_b"] if r["note_a"] == na else r["note_a"]) for r in rows or []
                }
                check("after the job the served links are the stored level-1 pairs "
                      "(level 1 wins over level 0)",
                      bool(links) and all(link.get("id") in stored_ids for link in links),
                      f"served={[link.get('id') for link in links]}, stored={sorted(stored_ids)}")
                check("the same-namespace twin is never served as a link",
                      all(link.get("id") != nsame for link in links),
                      f"links={[link.get('id') for link in links]}")

    chunk = await c.call("memory_get", {"id": na, "chunk": 0})
    check("the chunk read carries links as well",
          "links" in chunk and not chunk.get("hint"),
          f"keys={sorted(chunk)}, hint={chunk.get('hint')!r}")

    batch = await c.call("memory_get", {"ids": [na, nb]})
    check("the batch read carries no links", "links" not in batch, f"keys={sorted(batch)}")

    deleted = await c.call("memory_delete", {"id": nb})
    check("the linked probe note is soft-deleted", bool(deleted.get("deleted")),
          f"result={deleted}")
    got = await c.call("memory_get", {"id": na})
    check("a soft-deleted note is never served as a link",
          all(link.get("id") != nb for link in got.get("links", [])),
          f"links={[link.get('id') for link in got.get('links', [])]}")
    # probe data removal of this scenario is handled by the run teardown
    return na


# --- scenario 4: models unavailable → wait → models back ----------------------

async def scenario_4_models_outage(c: Client, rest: httpx2.AsyncClient) -> None:
    scenario(4, "Models unavailable → jobs wait → models back → jobs finish WITHOUT a restart")

    if not (CFG["slot_off_cmd"] and CFG["slot_on_cmd"]):
        skip("models unavailable → waiting → models back → jobs finished",
             "the contour hook is required: LSB_SLOT_OFF_CMD / LSB_SLOT_ON_CMD "
             "(e.g. scripts/slot_gate.sh off|on all — the default docker mode "
             "pauses/unpauses the slot proxy containers of the contour, so no "
             "privileges are needed and no LAN host is touched)")
        manual("scenario 4 (lsb-0014 FR-2.4)",
               "set LSB_SLOT_OFF_CMD / LSB_SLOT_ON_CMD to scripts/slot_gate.sh "
               "off|on all and run the E2E: the job must wait and finish without "
               "a container restart")
        return

    baseline = await health_snapshot(rest)
    info(f"before the outage: pending_vector={baseline.get('pending_vector')}, "
         f"pending_summary={baseline.get('pending_summary')}")

    proc = await run_shell(CFG["slot_off_cmd"])
    check("the model slot is stopped by the contour hook", proc.returncode == 0,
          f"rc={proc.returncode}")
    if proc.returncode != 0:
        return
    teardown_note("model slot was stopped and started back by the scenario hook")

    saved = await c.call("memory_save", {"text": f"{RUN['tag']} outage probe note text.",
                                         "title": RUN["note_outage"],
                                         "namespace": RUN["ns"]})
    nid = saved.get("id")
    check("the probe note of the outage scenario is created", nid is not None, f"id={nid}")
    if nid is not None:
        RUN["notes"].append(nid)

    released = False
    try:
        ok, detail = await wait_until(
            legacy_pending_at_least(rest, "pending_vector", 1),
            "the vector queue holds a pending item", timeout=min(CFG["wait_sec"], 120.0))
        check("with the models down the task stays pending (nothing is lost)", ok, detail)

        ok, detail = await wait_until(queue_age_at_least(rest, "vector", 10),
                                      "the age of the oldest pending item grows")
        check("the age of the oldest pending item grows (/health.queues)", ok, detail)

        found, detail = await log_has("queue_waiting")
        if CFG["logs_cmd"]:
            check("the queue_waiting event is written to the log", found, detail)
        else:
            skip("the queue_waiting event in the log",
                 "LSB_LOGS_CMD is not set (no log hook on this contour)")

        proc = await run_shell(CFG["slot_on_cmd"])
        released = True
        check("the model slot is started back by the contour hook", proc.returncode == 0,
              f"rc={proc.returncode}")
    finally:
        # The gate is never left closed: a failed check or a broken session in the
        # middle of the outage would keep the models blocked for the whole run (and
        # for the next scenarios) — release it on the scenario teardown path.
        if not released:
            undo_rc, undo_detail = 0, ""
            try:
                undo = await run_shell(CFG["slot_on_cmd"])
                undo_rc = undo.returncode
            except Exception as exc:  # noqa: BLE001 — the release must not mask the error
                undo_rc, undo_detail = None, describe(exc)
            if undo_rc == 0:
                teardown_note("model slot was released by the teardown hook (the scenario "
                              "did not reach its own `on` step)")
                warn("the scenario left the outage before its own `on` step",
                     "the model slot was released by the teardown hook")
            else:
                warn("the model slot could NOT be released — remove the block by hand",
                     undo_detail or f"rc={undo_rc}")

    ok, detail = await wait_until(queue_empty(rest, "vector"),
                                  "the vector queue is drained after the models are back")
    check("after the models are back the vector job catches up within the guard",
          ok, detail)
    ok, detail = await wait_until(legacy_pending_at_least(rest, "pending_summary", 0),
                                  "pending_summary is readable again")
    if not ok:
        warn("pending_summary after the outage", detail)

    if db_ready() and nid is not None:
        async def note_ready() -> tuple[bool, str]:
            row, err = db_try("SELECT vector_status, summary_status FROM notes WHERE id = ?",
                              (nid,))
            if not row:
                return False, err
            return (row[0]["vector_status"] == "ok"), \
                f"vector_status={row[0]['vector_status']!r}, " \
                f"summary_status={row[0]['summary_status']!r}"

        ok, detail = await wait_until(note_ready(), "the probe note is vectorized again")
        check("the deferred task was executed after the models returned", ok, detail)
    else:
        skip("the deferred task was executed after the models returned",
             f"the DB is not reachable ({CFG['db']})")

    # The session opened BEFORE the outage still works: a container restart would
    # have forgotten its session_id (SDK mcp 2.x) — this is the "no restart" proof.
    try:
        alive = await c.call("memory_list", {"limit": 1, "detail": "titles"})
        session_alive = "items" in alive
    except Exception as exc:  # noqa: BLE001
        session_alive = False
        detail = describe(exc)
    check("the MCP session opened before the outage is still alive (no container restart)",
          session_alive, "" if session_alive else f"session error: {detail}")


# --- scenario 5: node order ---------------------------------------------------

async def scenario_5_node_order(c: Client, rest: httpx2.AsyncClient) -> None:
    scenario(5, "Node order: the default sweep drains the backlog and does not repeat itself")

    if not db_ready():
        skip("the default sweep (job `nodes`)",
             f"the DB is not reachable ({CFG['db']}) — run the script inside the container")
        return

    async def column_present(column: str) -> bool:
        rows, _err = db_try("PRAGMA table_info(notes)")
        return column in {row["name"] for row in rows or []}

    if not await column_present("node_order_at"):
        skip("the default sweep (job `nodes`)",
             "notes.node_order_at is missing — the DB was not upgraded to 3.1.0")
        return

    left, _err = db_scalar(
        "SELECT COUNT(*) FROM notes WHERE deleted_at IS NULL AND namespace = 'default' "
        "AND vector_status = 'ok' AND node_order_at IS NULL")
    info(f"the default backlog at the start: {left} note(s) with a vector and no marker")

    created = []
    for i in range(3):
        saved = await c.call("memory_save", {
            "text": f"{RUN['tag']} node order probe {i}: the release acceptance checks "
                    "that a note left in default is not classified twice.",
            "title": RUN["note_nodes"][i],
        })
        if saved.get("id") is not None:
            created.append(saved["id"])
    check("the node-order probe notes are created", len(created) == 3,
          f"ids={created}")
    RUN["notes"].extend(created)
    if not created:
        return

    async def all_marked() -> tuple[bool, str]:
        placeholders = ",".join("?" * len(created))
        done, _err = db_scalar(
            f"SELECT COUNT(*) FROM notes WHERE id IN ({placeholders}) "
            "AND node_order_at IS NOT NULL", tuple(created))
        return done == len(created), f"marked {done}/{len(created)}"

    ok, detail = await wait_until(all_marked(),
                                  "the nodes job marked the probe notes (node_order_at)")
    check("the default sweep processes the backlog within the guard", ok, detail)
    if not ok:
        manual("node order",
               "the nodes job interval (JOB_NODES_INTERVAL_SEC) may be longer than "
               "the guard: re-run with a smaller interval or a bigger E2E_WAIT_SEC")
        return

    before, _err = db_try(
        "SELECT id, node_order_at FROM notes WHERE id IN "
        f"({','.join('?' * len(created))})", tuple(created))
    marks = {row["id"]: row["node_order_at"] for row in before or []}

    ok, detail = await wait_until(queue_empty(rest, "nodes"),
                                  "the nodes queue is drained")
    check("the nodes queue goes back to empty (/health.queues)", ok, detail)

    after, _err = db_try(
        "SELECT id, node_order_at FROM notes WHERE id IN "
        f"({','.join('?' * len(created))})", tuple(created))
    after_marks = {row["id"]: row["node_order_at"] for row in after or []}
    same = all(after_marks.get(nid) == mark for nid, mark in marks.items())
    check("a processed note is not processed twice (markers are unchanged)", same,
          f"before={marks}, after={after_marks}")

    budget = CFG["nodes_batch"]
    pending = queue_stat(await health_snapshot(rest), "nodes") or {}
    check("the sweep keeps within its batch budget",
          (pending.get("pending") or 0) <= max(budget, 0),
          f"pending={pending.get('pending')}, batch={budget}")


# --- scenario 6: /health ------------------------------------------------------

async def scenario_6_health(rest: httpx2.AsyncClient) -> None:
    scenario(6, "/health = 7 previous fields + queues + version")

    r = await rest.get("/health")
    health = r.json() if r.status_code == 200 else {}
    HEALTH.update(health or {})
    check("/health status = ok and no token is required",
          r.status_code == 200 and health.get("status") == "ok",
          f"status={r.status_code}")
    want = set(HEALTH_LEGACY) | {"queues", "version"}
    check("the field set is exactly 7 previous fields + queues + version",
          set(health) == want, f"extra={sorted(set(health) - want)}, "
                               f"missing={sorted(want - set(health))}")
    for field in HEALTH_LEGACY:
        check(f"/health keeps the field {field!r}", field in health,
              f"value={health.get(field)!r}")
    check("version is a dotted release string",
          isinstance(health.get("version"), str)
          and health["version"].count(".") == 2, f"version={health.get('version')!r}")
    expected = CFG["expect_version"]
    if expected and health.get("version") != expected:
        warn("/health.version differs from the expected release tag",
             f"got {health.get('version')!r}, expected {expected!r} — the version bump "
             "is the release closure step AFTER gate O (techdebt-0036-01)")
    elif expected:
        check(f"/health.version matches the release tag {expected}", True,
              f"version={health['version']}")

    queues = health.get("queues")
    check("queues is an object", isinstance(queues, dict), f"type={type(queues).__name__}")
    if isinstance(queues, dict):
        check("every job queue is reported (vector/summary/judge/areas/links/nodes)",
              set(HEALTH_QUEUES) <= set(queues),
              f"got={sorted(queues)}")
        bad = [name for name, stat in queues.items()
               if not isinstance(stat, dict)
               or set(stat) != {"pending", "oldest_pending_sec"}]
        check("every queue holds exactly {pending, oldest_pending_sec}", not bad,
              f"suspect={bad}")
    check("only SQL backs the queues (no models): /health keeps answering fast",
          r.status_code == 200)


# --- scenario 7: context budgets ---------------------------------------------

async def scenario_7_budgets(c: Client, note_with_links: int | None) -> None:
    scenario(7, "Context budgets: search ≤ 1.2 KB, list ≤ 1.5 KB, links ≤ 0.5 KB")

    res = await c.call("memory_search", {"query": f"{RUN['tag']} release acceptance budget",
                                         "top_k": 5})
    size = json_size(res)
    check(f"memory_search top_k=5 fits into {BUDGET_SEARCH} B",
          size <= BUDGET_SEARCH, f"{size} B")
    page = await list_page(c, limit=MCP_LIMIT)
    size = json_size(page)
    check(f"one memory_list page fits into {BUDGET_LIST} B", size <= BUDGET_LIST, f"{size} B")
    if note_with_links is None:
        skip("the links overhead of one note",
             "no probe note with links (scenario 3 was incomplete)")
    else:
        got = await c.call("memory_get", {"id": note_with_links})
        links = got.get("links", [])
        size = json_size(links)
        check(f"the links array of one note fits into {BUDGET_LINKS} B",
              size <= BUDGET_LINKS, f"{size} B, n={len(links)}")


# --- scenario 8: live DB upgrade ---------------------------------------------

async def scenario_8_upgrade(main: dict[str, Client], rest: httpx2.AsyncClient) -> None:
    scenario(8, "Live DB upgrade v3.0.0 → 3.1.0: schema, intact notes, repeated start")

    r = await rest.get("/health")
    health = r.json() if r.status_code == 200 else {}
    HEALTH.update(health or {})
    check("/health answers on the upgraded base", r.status_code == 200,
          f"status={r.status_code}")
    notes_count = health.get("notes_count")
    check("notes survived the upgrade (notes_count > 0)",
          isinstance(notes_count, int) and notes_count > 0, f"notes_count={notes_count}")

    if not db_ready():
        skip("the links table and the marker columns created by the upgrade",
             f"the DB is not reachable ({CFG['db']}) — run the script inside the container")
    else:
        rows, err = db_try("SELECT name FROM sqlite_master WHERE type = 'table'")
        names = {row["name"] for row in rows or []}
        check("the upgrade created the links table",
              all(table in names for table in UPGRADE_TABLES),
              f"missing={[t for t in UPGRADE_TABLES if t not in names]}, err={err}")
        idx, _err = db_try("SELECT name FROM sqlite_master WHERE type = 'index' "
                           "AND name = 'idx_links_b'")
        check("the reverse-side index idx_links_b exists (both directions are served)",
              bool(idx))
        cols, _err = db_try("PRAGMA table_info(notes)")
        colnames = {row["name"] for row in cols or []}
        check("notes carries the 3.1.0 marker columns",
              all(column in colnames for column in UPGRADE_COLUMNS),
              f"missing={[c for c in UPGRADE_COLUMNS if c not in colnames]}")
        page = await list_page(main["main"], detail="titles")
        check("MCP reads work on the upgraded base",
              "items" in page and "total" in page, f"keys={sorted(page)}")

    if not CFG["restart_cmd"]:
        skip("a repeated start of the upgraded container is a no-op",
             "an external restart is needed: set LSB_RESTART_CMD (a stage 6 step)")
        return

    before = await health_snapshot(rest)
    before_seed, _err = db_scalar(
        "SELECT COUNT(*) FROM skills WHERE deleted_at IS NULL")
    # Close the main session BEFORE the restart: the server still remembers the
    # session id, so the shutdown is clean and no dead SSE stream is left behind.
    await main["main"].close()
    proc = await run_shell(CFG["restart_cmd"])
    check("the restart hook LSB_RESTART_CMD succeeded", proc.returncode == 0,
          f"rc={proc.returncode}")
    teardown_note("the contour was restarted by the scenario hook (LSB_RESTART_CMD)")

    ok, detail = await wait_until(
        lambda: _health_ok(rest), "the contour is up again after the restart",
        timeout=min(CFG["wait_sec"], 180.0))
    check("after the restart /health is ok again", ok, detail)
    after = await health_snapshot(rest)
    check("a repeated start is a no-op: notes_count is unchanged",
          after.get("notes_count") == before.get("notes_count"),
          f"before={before.get('notes_count')}, after={after.get('notes_count')}")
    after_seed, _err = db_scalar("SELECT COUNT(*) FROM skills WHERE deleted_at IS NULL")
    check("a repeated start is a no-op: the seed rows are not duplicated",
          after_seed == before_seed, f"before={before_seed}, after={after_seed}")

    async with AsyncExitStack() as stack:
        c, init = await fresh_session(stack, own_stack=True)
        tools_count = len((await c.session.list_tools()).tools)
        check("an MCP session comes up after the restart (new handshake)",
              bool(init.instructions is not None), f"tools={tools_count}")
        await c.close()
    main["main"] = None  # the main session is gone; the teardown opens its own


async def _health_ok(rest: httpx2.AsyncClient) -> tuple[bool, str]:
    snapshot = await health_snapshot(rest)
    return snapshot.get("status") == "ok", f"status={snapshot.get('status')}"


# --- scenario 9: regression ---------------------------------------------------

async def scenario_9_regression() -> None:
    scenario(9, "Regression: unit tests + the existing E2E scripts")

    manual("unit regression",
           f"cd {CFG['repo_dir']} && {CFG['python']} -m pytest -q "
           "(baseline 1454 passed, 13 skipped, 0 failed)")
    if not CFG["run_regression"]:
        skip("unit and E2E regression",
             "run with --run-regression / LSB_RUN_REGRESSION=1 (long)")
        manual("existing E2E on the contour",
               "e2e_release30.py, e2e_release22.py, e2e_lsb0004.py, "
               "e2e_lsb0005.py, e2e_lsb0006.py")
        return

    proc = await run_shell(f"cd {CFG['repo_dir']} && {CFG['python']} -m pytest -q",
                           timeout=3600.0)
    tail = "\n".join(proc.stdout.strip().splitlines()[-3:])
    check("unit regression is green (0 failed)", proc.returncode == 0, tail)
    for script in CFG["regression_scripts"].split(","):
        name = script.strip()
        if not name:
            continue
        proc = await run_shell(
            f"cd {CFG['repo_dir']} && {CFG['python']} scripts/{name}", timeout=3600.0)
        tail = "\n".join(proc.stdout.strip().splitlines()[-2:])
        check(f"the existing E2E script {name} is green", proc.returncode == 0, tail)


# --- run ---------------------------------------------------------------------

async def run_all() -> None:
    async with AsyncExitStack() as stack:
        c, init = await fresh_session(stack)
        instructions = init.instructions or ""
        rest = await rest_client(stack)
        main: dict[str, Client] = {"main": c}

        print(f"\n=== E2E release {RELEASE} acceptance (contour lsb-test) ===")
        print(f"MCP: {CFG['mcp_url']} | REST: {CFG['base_url']} | DB: {CFG['db']} | "
              f"token: {mask(CFG['token'])}")
        print(f"run prefix: {CFG['prefix']} | identifiers: {RUN}")
        print(f"guard timeout: E2E_WAIT_SEC={CFG['wait_sec']:.0f}s, poll={CFG['poll_sec']:.0f}s")

        try:
            teardown_note(f"run started at {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")
            await prepare_leftovers()
            await scenario_0_context(c, rest, instructions)
            await scenario_1_limits(c, rest)
            await scenario_2_chars(c, rest)
            links_note = await scenario_3_links(c, rest)
            await scenario_4_models_outage(c, rest)
            await scenario_5_node_order(c, rest)
            await scenario_6_health(rest)
            await scenario_7_budgets(c, links_note)
            await scenario_8_upgrade(main, rest)
            await scenario_9_regression()
            close_scenario()
        finally:
            # Teardown always runs — also after a failed scenario (FR-3.2): the
            # probe data is marked, so a partially run scenario can still be
            # cleaned and the next run starts from the same state.
            try:
                await cleanup_own_data()
                await cleanup_namespaces(rest)
            except Exception as exc:  # noqa: BLE001 — the trace must survive anyway
                teardown_note(f"teardown raised {describe(exc)} — check the probe "
                              f"data with prefix {CFG['prefix']} by hand")
            ENV["db_size"] = db_size_text()
            path = write_teardown_trace()
            if path is not None:
                print(f"\n[TRACE] teardown trace: {path}")


def print_report() -> int:
    print(f"\n=== RUN RESULT (E2E release {RELEASE}) ===")
    print(f"app version: {ENV.get('version')!r} | image: {CFG['image'] or '(not set)'} | "
          f"image revision: {ENV.get('revision')!r} | image version: {ENV.get('image_version')!r}")
    print(f"models at the start: embedding_ok={HEALTH.get('embedding_ok')}, "
          f"summarizer_ok={HEALTH.get('summarizer_ok')}, judge_ok={HEALTH.get('judge_ok')} | "
          f"DB: {ENV.get('db_size') or db_size_text()}")
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


async def main() -> int:
    parser = argparse.ArgumentParser(
        description=f"Acceptance E2E of release {RELEASE} (MCP + REST of the lsb-test "
                    "contour): 10 scenarios of techdebt-0036-01, waiting is driven by "
                    "the observable state, the run is idempotent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base-url", default=os.environ.get("BASE_URL", "http://localhost:8080"),
                        help="REST address of the contour")
    parser.add_argument("--mcp-url", default=os.environ.get("MCP_URL", ""),
                        help="MCP address (empty — BASE_URL + /mcp)")
    parser.add_argument("--token", default=os.environ.get("MCP_AUTH_TOKEN", ""),
                        help="Bearer token (by default from the environment; never printed)")
    parser.add_argument("--db", default=os.environ.get("DB_PATH", "/data/notes.db"),
                        help="SQLite DB of the contour (wait predicates, upgrade checks)")
    parser.add_argument("--prefix", default=os.environ.get("LSB_E2E_PREFIX", "e2e31"),
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
                        help="directory of the teardown trace (empty → release/3.1.0/acceptance)")
    parser.add_argument("--logs-cmd", default=os.environ.get("LSB_LOGS_CMD", ""),
                        help="command printing the contour log (queue_waiting check)")
    parser.add_argument("--restart-cmd", default=os.environ.get("LSB_RESTART_CMD", ""),
                        help="shell command restarting the contour (repeated-start check)")
    parser.add_argument("--slot-off-cmd", default=os.environ.get("LSB_SLOT_OFF_CMD", ""),
                        help="shell command stopping the model slot (scenario 4, "
                             "e.g. scripts/slot_gate.sh off all)")
    parser.add_argument("--slot-on-cmd", default=os.environ.get("LSB_SLOT_ON_CMD", ""),
                        help="shell command starting the model slot back (scenario 4, "
                             "e.g. scripts/slot_gate.sh on all)")
    parser.add_argument("--expect-version", default=os.environ.get("LSB_EXPECT_VERSION", RELEASE),
                        help="release tag /health.version is compared with")
    parser.add_argument("--nodes-batch", type=int,
                        default=int(os.environ.get("JOB_NODES_BATCH", "20")),
                        help="nodes job batch budget of the contour")
    parser.add_argument("--run-regression", action="store_true",
                        default=os.environ.get("LSB_RUN_REGRESSION", "") == "1",
                        help="run the unit regression and the existing E2E scripts (long)")
    parser.add_argument("--python", default=os.environ.get("LSB_PYTHON", "python"),
                        help="interpreter for the regression run")
    parser.add_argument("--repo-dir", default=os.environ.get("LSB_REPO_DIR", ""),
                        help="repository directory for the regression (empty — the script root)")
    parser.add_argument("--regression-scripts",
                        default=os.environ.get(
                            "LSB_E2E_SCRIPTS",
                            "e2e_release30.py,e2e_release22.py,e2e_lsb0004.py,"
                            "e2e_lsb0005.py,e2e_lsb0006.py"),
                        help="existing E2E scripts, comma separated")
    parser.add_argument("--no-cleanup", action="store_true",
                        default=os.environ.get("LSB_KEEP", "") == "1",
                        help="do not remove the leftovers of a previous run (diagnostics)")
    args = parser.parse_args()

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
        "expect_version": args.expect_version,
        "nodes_batch": args.nodes_batch,
        "run_regression": args.run_regression,
        "python": args.python,
        "repo_dir": args.repo_dir or str(Path(__file__).resolve().parents[1]),
        "regression_scripts": args.regression_scripts,
        "no_cleanup": args.no_cleanup,
    })
    run_id = os.environ.get("LSB_E2E_RUN_ID") or time.strftime("%m%d%H%M%S")
    tag = f"{CFG['prefix']}-{run_id}"
    RUN.update({
        "tag": tag,
        "ns": tag,
        "note_chars": f"{tag} chars probe",
        "note_long": f"{tag} long chunked probe",
        "note_links_a": f"{tag} links probe A",
        "note_links_b": f"{tag} links probe B",
        "note_links_same": f"{tag} links probe same node",
        "note_other": f"{tag} unrelated probe",
        "note_outage": f"{tag} outage probe",
        "note_nodes": [f"{tag} nodes probe {i}" for i in range(3)],
        "notes": [],
    })

    # Up to 3 attempts: a broken MCP session (worker/network restart) must not
    # zero the run — the counters are reset and the pre-cleanup is idempotent.
    attempts = 0
    while attempts < 3:
        attempts += 1
        reset_counters()
        try:
            await run_all()
            break
        except MCPError as exc:
            print(f"\n[retry] the MCP session broke (attempt {attempts}): {describe(exc)}")
            await asyncio.sleep(5)
        except (httpx2.HTTPError, httpx2.TransportError, OSError) as exc:
            print(f"\n[retry] contour transport (attempt {attempts}): {describe(exc)}")
            await asyncio.sleep(5)
        except Exception as exc:
            print(f"\n[retry] run error (attempt {attempts}): {describe(exc)}")
            await asyncio.sleep(5)
    else:
        print("\n[FATAL] the run did not finish within 3 attempts")

    if not REPORT and FAIL == 0:
        # Not a single scenario started: the contour is unreachable — that is NOT
        # "everything is green".
        print("\n[FATAL] the run produced no scenario: the contour is unavailable — "
              "check BASE_URL, MCP_AUTH_TOKEN and that the lsb-test container is up")
        return 3
    return print_report()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
