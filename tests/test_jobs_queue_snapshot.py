"""The queue snapshot: the wait and the attention are distinguishable (#39, 3.2.0; arch §6.9/§7.2).

    `queue_snapshot` — the backward-compatible extension of lsb-0014-03: the old
    `queue_snapshot(n, age)` call stays valid, and the snapshot also carries
    `waiting` (deferred by the slot/readiness) and `attention` (a `waiting`
    subset waiting for a human). An empty queue — zeros and
    `oldest_pending_sec=None`.

    3.3.1 (LSBN-S-3): the age — over pending+waiting (all work waiting to be
    done) — on a slot breaker the work moves pending → waiting and the age no
    longer vanishes; null — only when both categories are empty.
"""

from __future__ import annotations

from app.services.jobs import queue_snapshot


def test_snapshot_backward_compatible() -> None:
    """The old call without the counters: waiting=0, attention=0 (compatibility)."""
    assert queue_snapshot(3, 42) == {
        "pending": 3,
        "oldest_pending_sec": 42,
        "waiting": 0,
        "attention": 0,
    }


def test_snapshot_empty_queue() -> None:
    """An empty queue: the oldest's age — null; the counters zero."""
    assert queue_snapshot(0, None) == {
        "pending": 0,
        "oldest_pending_sec": None,
        "waiting": 0,
        "attention": 0,
    }
    # pending=0 with a passed age — also "empty" (null, not negative).
    assert queue_snapshot(None, 10) == {
        "pending": 0,
        "oldest_pending_sec": None,
        "waiting": 0,
        "attention": 0,
    }


def test_snapshot_passes_counters() -> None:
    """The waiting and attention counters pass unchanged (#39)."""
    assert queue_snapshot(0, None, waiting=2, attention=1) == {
        "pending": 0,
        "oldest_pending_sec": None,
        "waiting": 2,
        "attention": 1,
    }


def test_snapshot_age_survives_waiting_only_queue() -> None:
    """The 3.3.1 fix's point (LSBN-S-3): pending=0, waiting>0 — the age preserved.

                On a slot breaker the work moves pending → waiting (the same age), so
                zeroing it at an empty pending would be a false "the queue is empty":
                the age vanishes only when the work waiting to be done is gone
                entirely.
    """
    assert queue_snapshot(0, 42, waiting=3) == {
        "pending": 0,
        "oldest_pending_sec": 42,
        "waiting": 3,
        "attention": 0,
    }


def test_snapshot_age_hidden_only_when_both_empty() -> None:
    """Old calls without waiting behave as before (a regression).

                `queue_snapshot(0, age)` → null (both categories empty);
                `queue_snapshot(5, age)` → the age (a non-empty pending keeps it, as
                before 3.3.1).
    """
    assert queue_snapshot(0, 42)["oldest_pending_sec"] is None
    assert queue_snapshot(None, 42)["oldest_pending_sec"] is None
    assert queue_snapshot(5, 42) == {
        "pending": 5,
        "oldest_pending_sec": 42,
        "waiting": 0,
        "attention": 0,
    }
