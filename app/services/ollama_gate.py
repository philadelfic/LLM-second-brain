"""The Ollama request queue - one request to the server at a time.

    The owner's decision (after Phase 8 end-to-end testing, the F1 finding): the
    server processes ONE generation at a time, there is no parallelism. Parallel
    HTTP requests do not speed the system up - they break it: Ollama queues them
    itself, but the client read timeout (httpx) ticks from the moment of sending,
    so a request that waited out someone else's generation dies by timeout before
    its own work starts. That is how, in the E2E, the judge (a 30 s timeout)
    timed out standing behind a merge (240 s), and the "insufficiency" of the
    timeouts turned out to be the server-side queue.

    The queue lives in the application: `ollama_slot(base_url)` - a lock per
    server address. Every call to one base_url lines up, and the read timeout
    ticks only from the actual send (waiting in the queue spends nothing). The
    servers are independent: vectorization and summary/judge do not serialize
    against each other - one queue per base_url.

    Guarantees and limits:
    - ALL outbound HTTP calls are covered: `EmbeddingService.embed_texts` (the
      single encoding point - the notes queue, the chunk queue, search),
      `SummaryService` (summarize + merge) and `JudgeService` (the dedup verdict).
    - No nested takes of one slot (the worker: embed -> release -> judge ->
      release -> merge); a deadlock is impossible; one call = one take.
    - The Lock gives no FIFO (the wake order is nondeterministic) - for a
      single-user personal scenario the order is not critical; queue fairness is
      not promised.
    - No cross-process queue is needed: one worker per process (ARCH SS3.4).

    Unit tests: `tests/test_ollama_gate.py` - serialization over one base_url and
    parallelism across different ones.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager

# The slot registry: base_url -> Lock. Grows with the number of distinct addresses (2-3 in
# production); the locks keep no state between takes.
_registry_lock = threading.Lock()
_slots: dict[str, threading.Lock] = {}


def _slot_for(base_url: str) -> threading.Lock:
    """The server's slot (created on first use; a dict under the GIL + a lock)."""
    with _registry_lock:
        lock = _slots.get(base_url)
        if lock is None:
            lock = threading.Lock()
            _slots[base_url] = lock
        return lock


@contextmanager
def ollama_slot(base_url: str) -> Iterator[None]:
    """Take the server's slot: one request per base_url at a time.

            The caller holds the slot exactly for the duration of the HTTP call
            (including the embedder's retries - that is one task); waiting in the
            queue spends no client read timeouts: they tick only after the request is
            actually sent.

    """
    slot = _slot_for(base_url)
    slot.acquire()
    try:
        yield
    finally:
        slot.release()