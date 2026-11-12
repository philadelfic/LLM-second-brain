"""The bounded shutdown (LSBN-D-16): SIGTERM must actually kill the process.

12.11.2026 prod: `docker stop` sent SIGTERM, uvicorn stopped the HTTP
listener (18:56:32Z), but the process lived on for 20+ minutes and the
container stayed "Up" — `restart: unless-stopped` never fired and nothing
alerted ("a quiet memory outage"). Why:

- the background work runs in `asyncio.to_thread` threads (sync httpx calls
  with the read timeout up to 900 s — one batch can hold a thread for hours);
- cancelling the loop task does NOT interrupt a running thread — it works
  to the end of the batch, so "the worker wrote to the log until 19:16Z";
- the exit path joins the executor threads
  (`RuntimeWarning: executor did not finish joining its threads within
  300 seconds` at 19:05Z).

So the shutdown has an upper bound now: the cancelled background tasks get
SHUTDOWN_TIMEOUT_SEC (≤30) to unwind, then a short grace for the busy
executor threads to finish, then a forced exit — the `shutdown_forced`
journal event and `os._exit(0)`. `os._exit` skips the interpreter shutdown
that would block on the joins; exit code 0 — a docker stop is a planned
stop, and `unless-stopped` must not turn a planned stop into a restart loop.

Data safety: SQLite is transactional (WAL), committed work survives; an
interrupted batch keeps its pending statuses in the DB and is caught up
after the restart (ARCH §3.4).
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import threading
import time

# The grace for the executor threads AFTER the tasks are done (LSBN-D-16):
# a cancelled task returns promptly while its thread finishes the current
# call — milliseconds for SQL, up to the slot read timeout for an LLM call.
# The total shutdown stays within SHUTDOWN_TIMEOUT_SEC + this grace (the
# acceptance: "Exited(0) within timeout+5 s").
THREAD_GRACE_SEC = 5.0

#: The thread-name prefix of asyncio's default executor (`asyncio.to_thread`
#: runs there, the prefix is asyncio's own).
_ASYNCIO_THREAD_PREFIX = "asyncio_"

#: The executor module of CPython (the idle/busy classifier reads its
#: `_worker` frames — stable across 3.9..3.13).
_FUTURES_THREAD_MODULE = "concurrent/futures/thread.py"


def _busy_executor_threads() -> list[threading.Thread]:
    """The default-executor threads actually RUNNING app work right now.

    `asyncio.to_thread` runs in the default ThreadPoolExecutor; its IDLE
    threads park in `Queue.get()` and linger until the executor shutdown —
    not a shutdown problem (in the clean case Runner.close lets them out
    a moment later). A thread RUNNING a work item (a blocked LLM call) is.
    The two are told apart by the frame the `_worker` frame has called
    into: `_WorkItem.run` — busy; the queue wait — idle. A stack that does
    not match the known shape counts as busy (the conservative side:
    in prod that is the forced exit, in tests the seam).
    """
    frames = sys._current_frames()
    busy: list[threading.Thread] = []
    for thread in threading.enumerate():
        if not thread.is_alive() or not thread.name.startswith(
            _ASYNCIO_THREAD_PREFIX
        ):
            continue
        frame = frames.get(thread.ident)
        if frame is None:
            busy.append(thread)
            continue
        child = None
        found_worker = False
        is_busy = True  # a foreign stack shape counts as busy
        while frame is not None:
            filename = frame.f_code.co_filename.replace("\\", "/")
            if filename.endswith(_FUTURES_THREAD_MODULE) and (
                frame.f_code.co_name == "_worker"
            ):
                found_worker = True
                if child is not None and child.f_code.co_name == "run" and (
                    child.f_code.co_filename.replace("\\", "/").endswith(
                        _FUTURES_THREAD_MODULE
                    )
                ):
                    is_busy = True
                else:
                    is_busy = False  # parked in the work-queue wait
                break
            child = frame
            frame = frame.f_back
        if not found_worker or is_busy:
            busy.append(thread)
    return busy


def _force_exit(code: int) -> None:
    """The last-resort exit: os._exit skips the thread joins that would hang.

    A module-level seam: tests replace it (a real exit inside pytest would
    kill the whole run). As a second line of defense the real exit never
    fires under pytest at all — a false-positive `shutdown_forced` in a test
    must not nuke the runner (unit tests assert through the monkeypatched
    seam; the contour acceptance runs the real server, without pytest).
    """
    if "pytest" in sys.modules:
        return
    os._exit(code)


def _flush_logs() -> None:
    """Flush every handler — the forced exit skips the atexit flush."""
    seen: set[int] = set()
    for logger in (
        logging.getLogger(),
        logging.getLogger("app"),
        logging.getLogger("uvicorn"),
        logging.getLogger("uvicorn.error"),
        logging.getLogger("uvicorn.access"),
    ):
        for handler in logger.handlers:
            if id(handler) not in seen:
                seen.add(id(handler))
                handler.flush()


async def bounded_shutdown(tasks: list[asyncio.Task], *, timeout_sec: float) -> None:
    """Wait for the cancelled background tasks; force-exit past the budget.

    `tasks` — the already-cancelled background tasks (worker/backup/update).
    The wait is bounded by `timeout_sec` (SHUTDOWN_TIMEOUT_SEC): a task stuck
    in `asyncio.to_thread` (a blocked LLM call) is abandoned at the bound.
    After the tasks a short thread grace lets the busy executor threads
    finish. Anything still alive after that — one `shutdown_forced` warning
    (the pending task names and the stuck thread names) and `_force_exit(0)`.

    The clean path returns silently: the lifespan completes and uvicorn logs
    "Application shutdown complete" with the loops already cancelled.
    """
    logger = logging.getLogger("app")
    await asyncio.wait(tasks, timeout=timeout_sec)
    pending = [task for task in tasks if not task.done()]
    deadline = time.monotonic() + THREAD_GRACE_SEC
    stuck = _busy_executor_threads()
    while stuck and time.monotonic() < deadline:
        stuck[0].join(timeout=min(0.1, max(0.0, deadline - time.monotonic())))
        stuck = _busy_executor_threads()
    if not pending and not stuck:
        return
    logger.warning(
        "shutdown: the wait budget is over — forced exit",
        extra={
            "event": "shutdown_forced",
            "timeout_sec": timeout_sec,
            "thread_grace_sec": THREAD_GRACE_SEC,
            "pending_tasks": sorted(task.get_name() for task in pending),
            "stuck_threads": [thread.name for thread in stuck],
        },
    )
    _flush_logs()
    _force_exit(0)
