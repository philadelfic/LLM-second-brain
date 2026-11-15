#!/usr/bin/env python3
"""The note-storm acceptance run on a prod snapshot (LSBN-T-35, G3; LSBN-D-14).

    Why: in 3.6.1 the split and the background dedup met for the first time on the
    live volume. A long note is cut into N self-contained parts sharing one
    `origin_note_id`; the parts are close by construction, so the dedup treated
    them as duplicates of each other: the judge said "duplicate", the merge path
    asked the model, the answers did not fit the JSON parts contract
    (`off_contract`), the summary slot paused, the pairs came back to the queue -
    a storm of merge attempts. The unit tests prove the functions one by one on
    two or three synthetic notes; this run proves the CLASS on a real snapshot:
    3 895 live notes, 3 083 parts, 33 origin groups.

    What it does (never the original file - a COPY is opened):

      1. copies the snapshot (and its non-empty -wal sidecar) to a temporary
         directory and points the application settings at the copy;
      2. reads the metrics BEFORE: live notes, parts/groups, merge jobs and
         merge attempts, `off_contract` jobs, tasks in attention, the summary and
         judge queues (the application's own accounting);
      3. reproduces the storm scenario:
         - `--dry-run` (default): for every part with a ready vector the REAL
           `DeduplicationService.find_candidates` runs (the prefilter the judge
           loop calls). Candidates of the OWN origin must be 0. As the control
           that the check has teeth, the raw KNN window of the same size is read
           next to it: those own-origin neighbours are exactly what the 3.6.1
           rule let through. No model is constructed at all - the dry run cannot
           call one;
         - `--live`: judge jobs are queued for the parts (what re-vectorization
           does on prod) and the REAL judge loop runs on the copy with a
           call budget (`--max-calls`). Every call is counted; when the budget is
           spent the run stops instead of burning the slot. The merge phase runs
           only if the run created merge jobs - that is the storm itself;
      4. reads the metrics AFTER and prints a table plus the verdict:
         0 same-origin candidates / judge pairs, 0 same-origin merge jobs,
         0 new attention on them, the queues did not grow, the parts are intact;
      5. returns 0 - PASS, 1 - FAIL, 2 - the setup is unusable (no snapshot,
         an unreadable DB, the live DB passed by mistake).

    Run (the slot env of the contour is enough; a dry run never calls a slot, so
    closed addresses are fine):

        python scripts/note_storm_run.py --db data/backups/notes-<stamp>.db --dry-run
        python scripts/note_storm_run.py --db data/backups/notes-<stamp>.db --live --max-calls 300

    The path is always an argument: nothing contour-specific is baked in.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# It works both as `python scripts/note_storm_run.py` and as
# `python -m scripts.note_storm_run` - the project root enters the import paths.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import ConfigError, get_settings
from app.services.dedup import DeduplicationService
from app.services.judge import JudgeService
from app.services.summary import SummaryService
from app.services.worker import BackgroundWorker
from app.storage import vectors
from app.storage.db import StorageError, session

#: The default model-call budget of a live run (one call - one judge question).
DEFAULT_MAX_CALLS = 300

#: The raw KNN window widening step for the control metric: the own-origin
#: neighbours sit in the near window by construction.
RAW_WINDOW_EXTRA = 1


class SnapshotError(RuntimeError):
    """The snapshot cannot be used (missing, unreadable, or the live DB passed)."""


class BudgetExhausted(Exception):
    """The model-call budget is spent: stop the run instead of burning the slot."""


class _NullEmbedder:
    """A stand-in embedder: the run reads the vectors already in the snapshot.

    Neither the dry run nor the live phase re-vectorizes anything - the queue
    snapshots and the dedup prefilter need no encoder. A call is a defect of the
    run, so it raises.
    """

    def embed(self, text: str) -> list[float]:
        raise RuntimeError("note_storm_run: the embedder must not be called")

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("note_storm_run: the embedder must not be called")

    def close(self) -> None:
        return None


# --- the metrics ----------------------------------------------------------


@dataclass
class Metrics:
    """One 'before'/'after' snapshot of the states the storm moves."""

    alive_notes: int = 0
    parts: int = 0
    groups: int = 0
    merge_jobs: int = 0
    merge_attempts: int = 0
    merge_pending: int = 0
    off_contract: int = 0
    attention_total: int = 0
    attention_summary: int = 0
    attention_judge: int = 0
    summary_pending: int = 0
    summary_attention: int = 0
    judge_pending: int = 0
    judge_attention: int = 0


def _jobs_table_exists(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='worker_jobs'"
    ).fetchone()
    return row is not None


def collect_metrics(settings) -> Metrics:
    """Read the metrics; SQL plus the application's own queue accounting."""
    metrics = Metrics()
    with session(settings) as conn:
        metrics.alive_notes = int(
            conn.execute(
                "SELECT COUNT(*) FROM notes WHERE deleted_at IS NULL"
            ).fetchone()[0]
        )
        metrics.parts = int(
            conn.execute(
                "SELECT COUNT(*) FROM notes WHERE deleted_at IS NULL "
                "AND origin_note_id IS NOT NULL"
            ).fetchone()[0]
        )
        metrics.groups = int(
            conn.execute(
                "SELECT COUNT(DISTINCT origin_note_id) FROM notes "
                "WHERE deleted_at IS NULL AND origin_note_id IS NOT NULL"
            ).fetchone()[0]
        )
        if _jobs_table_exists(conn):
            metrics.merge_jobs = int(
                conn.execute(
                    "SELECT COUNT(*) FROM worker_jobs WHERE kind = 'merge'"
                ).fetchone()[0]
            )
            metrics.merge_attempts = int(
                conn.execute(
                    "SELECT COALESCE(SUM(attempts), 0) FROM worker_jobs "
                    "WHERE kind = 'merge'"
                ).fetchone()[0]
            )
            metrics.merge_pending = int(
                conn.execute(
                    "SELECT COUNT(*) FROM worker_jobs WHERE kind = 'merge' "
                    "AND status = 'pending'"
                ).fetchone()[0]
            )
            metrics.off_contract = int(
                conn.execute(
                    "SELECT COUNT(*) FROM worker_jobs "
                    "WHERE last_outcome = 'off_contract'"
                ).fetchone()[0]
            )
            metrics.attention_total = int(
                conn.execute(
                    "SELECT COUNT(*) FROM worker_jobs "
                    "WHERE attention_at IS NOT NULL"
                ).fetchone()[0]
            )
            metrics.attention_summary = int(
                conn.execute(
                    "SELECT COUNT(*) FROM worker_jobs WHERE slot = 'summary' "
                    "AND attention_at IS NOT NULL"
                ).fetchone()[0]
            )
            metrics.attention_judge = int(
                conn.execute(
                    "SELECT COUNT(*) FROM worker_jobs WHERE slot = 'judge' "
                    "AND attention_at IS NOT NULL"
                ).fetchone()[0]
            )
    worker = BackgroundWorker(settings, _NullEmbedder())
    summary_queue = worker._summary_queue_stat()
    judge_queue = worker._judge_queue_stat()
    metrics.summary_pending = int(summary_queue["pending"])
    metrics.summary_attention = int(summary_queue["attention"])
    metrics.judge_pending = int(judge_queue["pending"])
    metrics.judge_attention = int(judge_queue["attention"])
    return metrics


# --- the snapshot ---------------------------------------------------------


def copy_snapshot(source: Path, workdir: Path, live_db: Path) -> Path:
    """Copy the snapshot (never open the original); the copy is what we touch."""
    if not source.is_file():
        raise SnapshotError(f"the snapshot does not exist: {source}")
    if source.resolve() == live_db.resolve():
        raise SnapshotError(
            f"the live database was passed as a snapshot: {source} "
            "(the run must work on a copy)"
        )
    target = workdir / "notes.db"
    shutil.copy2(source, target)
    # A snapshot taken with a non-empty WAL is only complete together with it:
    # the copy keeps the sidecar and sqlite replays it on the first open.
    for suffix in ("-wal", "-shm"):
        sidecar = Path(str(source) + suffix)
        if sidecar.is_file() and sidecar.stat().st_size > 0:
            shutil.copy2(sidecar, Path(str(target) + suffix))
    # A read of every page: a truncated/foreign file fails here, not mid-report.
    with sqlite3.connect(target) as conn:
        result = conn.execute("PRAGMA integrity_check").fetchone()
    if result is None or result[0] != "ok":
        raise SnapshotError(f"the snapshot copy is not readable: {result}")
    return target


def part_ids_with_vector(settings) -> list[int]:
    """The parts with a ready vector: the set the split put into the dedup."""
    with session(settings) as conn:
        rows = conn.execute(
            "SELECT n.id FROM notes n JOIN notes_vec v ON v.note_id = n.id "
            "WHERE n.deleted_at IS NULL AND n.origin_note_id IS NOT NULL "
            "ORDER BY n.id"
        ).fetchall()
    return [int(row["id"]) for row in rows]


def part_texts(settings) -> dict[int, tuple[str, int | None]]:
    """The parts' texts and origin groups before the run (the integrity check)."""
    with session(settings) as conn:
        rows = conn.execute(
            "SELECT id, text, origin_note_id FROM notes "
            "WHERE deleted_at IS NULL AND origin_note_id IS NOT NULL ORDER BY id"
        ).fetchall()
    return {int(row["id"]): (row["text"], row["origin_note_id"]) for row in rows}


# --- dry run: the prefilter over the real vectors -------------------------


@dataclass
class ScanResult:
    """What the dedup prefilter says about the parts of one snapshot."""

    parts_scanned: int = 0
    candidates_total: int = 0
    own_origin_candidates: int = 0
    raw_own_origin_neighbours: int = 0
    parts_with_raw_own_origin: int = 0
    elapsed_sec: float = 0.0


def scan_parts(settings, progress=None) -> ScanResult:
    """Run the REAL prefilter per part and, next to it, the unruled raw KNN window.

    The prefilter (`DeduplicationService.find_candidates`) is exactly what the
    judge loop calls: candidates of the own origin must not appear there. The raw
    window is the control: the same KNN query without the rule shows the
    neighbours 3.6.1 treated as duplicates, so a green run cannot be green by
    accident (a snapshot without such neighbours proves nothing).
    """
    dedup = DeduplicationService(settings)
    threshold = settings.dedup_candidate_similarity
    top_n = settings.dedup_candidate_top_n
    scan = ScanResult()
    started = time.monotonic()
    with session(settings) as conn:
        groups = {
            int(row["origin_note_id"]): int(row["count"])
            for row in conn.execute(
                "SELECT origin_note_id, COUNT(*) AS count FROM notes "
                "WHERE deleted_at IS NULL AND origin_note_id IS NOT NULL "
                "GROUP BY origin_note_id"
            ).fetchall()
        }
        rows = conn.execute(
            "SELECT n.id, n.namespace, n.origin_note_id FROM notes n "
            "JOIN notes_vec v ON v.note_id = n.id "
            "WHERE n.deleted_at IS NULL AND n.origin_note_id IS NOT NULL "
            "ORDER BY n.id"
        ).fetchall()
    for number, row in enumerate(rows, start=1):
        note_id = int(row["id"])
        origin = int(row["origin_note_id"])
        namespace = row["namespace"]
        with session(settings) as conn:
            vector = vectors.get_vector(conn, note_id)
            self_id = note_id  # the fresh note's own vector takes a KNN slot
            if vector is None:
                continue
            candidates = dedup.find_candidates(
                vector, exclude_id=note_id, namespace=namespace
            )
            scan.candidates_total += len(candidates)
            for candidate_id, _cosine in candidates:
                candidate_origin = conn.execute(
                    "SELECT origin_note_id FROM notes WHERE id = ?",
                    (candidate_id,),
                ).fetchone()[0]
                if candidate_origin == origin:
                    scan.own_origin_candidates += 1
            # The control: the raw window of the same width, rule not applied.
            window = top_n + groups.get(origin, 1) + RAW_WINDOW_EXTRA
            hits = vectors.knn(
                conn, vector, window, ns_filter=[namespace] if namespace else None
            )
            own_hits = 0
            for candidate_id, cosine in hits:
                if candidate_id == self_id or cosine < threshold:
                    continue
                candidate_origin = conn.execute(
                    "SELECT origin_note_id FROM notes WHERE id = ?",
                    (candidate_id,),
                ).fetchone()[0]
                if candidate_origin == origin:
                    own_hits += 1
            scan.raw_own_origin_neighbours += own_hits
            if own_hits:
                scan.parts_with_raw_own_origin += 1
        scan.parts_scanned += 1
        if progress is not None:
            progress(number, len(rows))
    scan.elapsed_sec = time.monotonic() - started
    return scan


# --- live run: the judge/merge paths with a call budget -------------------


@dataclass
class LiveResult:
    """What a live phase did: model calls and the states it moved."""

    judge_calls: int = 0
    merge_calls: int = 0
    same_origin_judge_pairs: int = 0
    same_origin_merge_calls: int = 0
    new_merge_jobs: int = 0
    new_same_origin_merge_jobs: int = 0
    new_attention: int = 0
    new_same_origin_attention: int = 0
    budget_exhausted: bool = False


class _Budget:
    """The shared model-call counter of a live run."""

    def __init__(self, limit: int) -> None:
        self.limit = max(0, int(limit))
        self.calls = 0

    def spend(self) -> None:
        if self.calls >= self.limit:
            raise BudgetExhausted(f"the model-call budget is spent: {self.limit}")
        self.calls += 1


class _BudgetedJudge(JudgeService):
    """The real judge with a hard call budget and a journal of the pairs asked."""

    def __init__(self, settings, budget: _Budget) -> None:
        super().__init__(settings)
        self._budget = budget
        self.pairs: list[tuple[str, str]] = []

    def judge(self, text_new: str, text_candidate: str):
        self._budget.spend()
        self.pairs.append((text_new, text_candidate))
        return super().judge(text_new, text_candidate)


class _BudgetedSummarizer(SummaryService):
    """The real summarizer with the same budget and a journal of the pairs merged."""

    def __init__(self, settings, budget: _Budget) -> None:
        super().__init__(settings)
        self._budget = budget
        self.pairs: list[tuple[str, str]] = []

    def merge(self, text_a: str, text_b: str, *, title: str | None = None):
        self._budget.spend()
        self.pairs.append((text_a, text_b))
        return super().merge(text_a, text_b, title=title)


def _origin_by_text(parts: dict[int, tuple[str, int | None]]) -> dict[str, int]:
    """text -> origin group (only parts; an ordinary text has no group)."""
    return {
        text: int(origin)
        for text, origin in parts.values()
        if origin is not None
    }


def _same_origin_pairs(
    pairs: list[tuple[str, str]], origins: dict[str, int]
) -> int:
    return sum(
        1
        for first, second in pairs
        if origins.get(first) is not None and origins.get(first) == origins.get(second)
    )


def _older_id_of(payload: str | None) -> Any:
    """The earlier side of a merge job's payload (None - a broken/absent payload)."""
    try:
        return json.loads(payload or "{}").get("older_id")
    except (TypeError, ValueError):
        return None


def _origin_of(conn: sqlite3.Connection, note_id: Any) -> int | None:
    if note_id is None:
        return None
    row = conn.execute(
        "SELECT origin_note_id FROM notes WHERE id = ?", (int(note_id),)
    ).fetchone()
    return None if row is None else row[0]


def run_live(
    settings, part_ids: list[int], parts: dict, max_calls: int
) -> LiveResult:
    """Queue judge jobs for the parts and run the real judge/merge loops."""
    live = LiveResult()
    budget = _Budget(max_calls)
    judge = _BudgetedJudge(settings, budget)
    summarizer = _BudgetedSummarizer(settings, budget)
    worker = BackgroundWorker(
        settings, _NullEmbedder(), summarizer, judge=judge
    )
    with session(settings) as conn:
        before_job_id = int(
            conn.execute("SELECT COALESCE(MAX(id), 0) FROM worker_jobs").fetchone()[0]
        )
    # What re-vectorization does on prod: one dedup job per part.
    for note_id in part_ids:
        worker._ensure_job("judge", "dedup", note_id)
    origins = _origin_by_text(parts)
    try:
        while worker.process_judge_pending():
            pass
        with session(settings) as conn:
            created = conn.execute(
                "SELECT id, note_id, payload FROM worker_jobs "
                "WHERE kind = 'merge' AND id > ? ORDER BY id",
                (before_job_id,),
            ).fetchall()
        live.new_merge_jobs = len(created)
        with session(settings) as conn:
            for row in created:
                older_id = _older_id_of(row["payload"])
                newer_origin = _origin_of(conn, row["note_id"])
                if (
                    newer_origin is not None
                    and newer_origin == _origin_of(conn, older_id)
                ):
                    live.new_same_origin_merge_jobs += 1
        if live.new_merge_jobs:
            # The storm itself: the merge path is entered only when the judge
            # created work for it.
            while worker.process_merge_pending():
                pass
    except BudgetExhausted:
        live.budget_exhausted = True
    live.judge_calls = len(judge.pairs)
    live.merge_calls = len(summarizer.pairs)
    live.same_origin_judge_pairs = _same_origin_pairs(judge.pairs, origins)
    live.same_origin_merge_calls = _same_origin_pairs(summarizer.pairs, origins)
    with session(settings) as conn:
        attention = conn.execute(
            "SELECT id, slot, note_id, payload FROM worker_jobs "
            "WHERE attention_at IS NOT NULL AND id > ?",
            (before_job_id,),
        ).fetchall()
        live.new_attention = len(attention)
        same_origin = 0
        for row in attention:
            older_id = _older_id_of(row["payload"])
            newer_origin = _origin_of(conn, row["note_id"])
            if (
                newer_origin is not None
                and newer_origin == _origin_of(conn, older_id)
            ):
                same_origin += 1
        live.new_same_origin_attention = same_origin
    return live


# --- the report and the verdict ------------------------------------------


def _row(label: str, before: Any, after: Any) -> str:
    return f"  {label:<44} {before!s:>10} {after!s:>10}"


def print_report(
    snapshot: Path,
    copy_path: Path,
    before: Metrics,
    after: Metrics,
    scan: ScanResult | None,
    live: LiveResult | None,
    max_calls: int,
) -> None:
    print("\n=== LSB note storm run (prod snapshot copy) ===\n")
    print(f"  snapshot: {snapshot}")
    print(f"  copy:     {copy_path}")
    print(f"  mode:     {'dry-run (no model calls)' if live is None else f'live (budget {max_calls} calls)'}")
    print("\n  metric                                       before      after")
    print("  " + "-" * 68)
    print(_row("live notes", before.alive_notes, after.alive_notes))
    print(_row("parts of split sources", before.parts, after.parts))
    print(_row("origin groups", before.groups, after.groups))
    print(_row("merge jobs", before.merge_jobs, after.merge_jobs))
    print(_row("merge attempts", before.merge_attempts, after.merge_attempts))
    print(_row("merge jobs pending", before.merge_pending, after.merge_pending))
    print(_row("jobs with outcome off_contract", before.off_contract, after.off_contract))
    print(_row("tasks in attention (all slots)", before.attention_total, after.attention_total))
    print(_row("  attention: summary / judge", f"{before.attention_summary} / {before.attention_judge}", f"{after.attention_summary} / {after.attention_judge}"))
    print(_row("summary queue pending", before.summary_pending, after.summary_pending))
    print(_row("judge queue pending", before.judge_pending, after.judge_pending))
    if scan is not None:
        print("\n  the storm scenario (the dedup prefilter over the real vectors):")
        print(f"  parts scanned with a ready vector: {scan.parts_scanned}")
        print(f"  candidates the prefilter returned: {scan.candidates_total}")
        print(f"  candidates of the OWN origin: {scan.own_origin_candidates}")
        print(
            "  the control - own-origin neighbours in the raw KNN window: "
            f"{scan.raw_own_origin_neighbours} "
            f"(parts with such neighbours: {scan.parts_with_raw_own_origin})"
        )
        print(f"  scan time: {scan.elapsed_sec:.1f} s")
    if live is not None:
        print("\n  the live phase (the real judge/merge paths):")
        print(f"  model calls: judge {live.judge_calls}, merge {live.merge_calls} (budget {max_calls})")
        print(f"  judge questions about one origin: {live.same_origin_judge_pairs}")
        print(f"  merge calls of one origin: {live.same_origin_merge_calls}")
        print(f"  merge jobs created: {live.new_merge_jobs} (of one origin: {live.new_same_origin_merge_jobs})")
        print(f"  new tasks in attention: {live.new_attention} (of one origin: {live.new_same_origin_attention})")
        if live.budget_exhausted:
            print("  the call budget was spent - the run stopped there")


def verdict(
    before: Metrics,
    after: Metrics,
    scan: ScanResult | None,
    live: LiveResult | None,
    parts: dict[int, tuple[str, int | None]],
    after_parts: dict[int, tuple[str, int | None]],
) -> list[tuple[str, bool, str]]:
    """The PASS/FAIL criteria of the gate, each with its number for the report."""
    checks: list[tuple[str, bool, str]] = []
    if scan is not None:
        checks.append(
            (
                "0 candidates of one origin in the dedup prefilter",
                scan.own_origin_candidates == 0,
                f"{scan.own_origin_candidates} candidates",
            )
        )
        checks.append(
            (
                "the control found own-origin neighbours at all",
                scan.raw_own_origin_neighbours > 0,
                (
                    f"{scan.raw_own_origin_neighbours} raw neighbours, "
                    f"{scan.parts_with_raw_own_origin} parts"
                ),
            )
        )
    if live is not None:
        checks.append(
            (
                "0 judge questions about one origin",
                live.same_origin_judge_pairs == 0,
                f"{live.same_origin_judge_pairs} pairs",
            )
        )
        checks.append(
            (
                "0 merge calls of one origin",
                live.same_origin_merge_calls == 0,
                f"{live.same_origin_merge_calls} calls",
            )
        )
        checks.append(
            (
                "0 merge jobs created for one origin",
                live.new_same_origin_merge_jobs == 0,
                f"{live.new_same_origin_merge_jobs} of {live.new_merge_jobs}",
            )
        )
        checks.append(
            (
                "0 new attention tasks of one origin",
                live.new_same_origin_attention == 0,
                f"{live.new_same_origin_attention} of {live.new_attention}",
            )
        )
    else:
        checks.append(
            (
                "no model was called (dry run)",
                live is None,
                "the run builds no slot services",
            )
        )
    checks.append(
        (
            "the summary queue did not grow",
            after.summary_pending <= before.summary_pending,
            f"{before.summary_pending} -> {after.summary_pending}",
        )
    )
    checks.append(
        (
            "the judge queue did not grow",
            after.judge_pending <= before.judge_pending,
            f"{before.judge_pending} -> {after.judge_pending}",
        )
    )
    if live is None:
        # A dry run must not move ANY state: attention included.
        checks.append(
            (
                "no new tasks in attention (dry run)",
                after.attention_total <= before.attention_total,
                f"{before.attention_total} -> {after.attention_total}",
            )
        )
    checks.append(
        (
            "every part is alive with its text unchanged",
            after_parts == parts,
            f"{len(parts)} parts",
        )
    )
    checks.append(
        (
            "no part was merged away (live notes did not shrink)",
            after.alive_notes >= before.alive_notes,
            f"{before.alive_notes} -> {after.alive_notes}",
        )
    )
    return checks


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "The note-storm acceptance run (LSBN-T-35): a split source against "
            "the background dedup on a COPY of a prod snapshot."
        )
    )
    parser.add_argument(
        "--db",
        required=True,
        help="the snapshot to work on (the run copies it, the original is never opened)",
    )
    parser.add_argument(
        "--max-calls",
        type=int,
        default=DEFAULT_MAX_CALLS,
        help=f"the model-call budget of a live run (default {DEFAULT_MAX_CALLS})",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="no model calls at all (the default): the prefilter over the real vectors",
    )
    mode.add_argument(
        "--live",
        action="store_true",
        help="also run the real judge/merge loops within --max-calls",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="keep the temporary copy of the snapshot for inspection",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source = Path(args.db).expanduser()
    try:
        settings = get_settings()
    except ConfigError as exc:
        print(f"[FAIL] the slot environment is not configured: {exc}")
        return 2
    workdir = Path(tempfile.mkdtemp(prefix="lsb-note-storm-"))
    try:
        try:
            copy_path = copy_snapshot(source, workdir, Path(settings.db_path))
        except (SnapshotError, StorageError) as exc:
            print(f"[FAIL] {exc}")
            return 2
        copy_settings = settings.model_copy(update={"db_path": str(copy_path)})
        before = collect_metrics(copy_settings)
        parts = part_texts(copy_settings)
        part_ids = part_ids_with_vector(copy_settings)
        scan: ScanResult | None = None
        live: LiveResult | None = None
        if args.live:
            live = run_live(copy_settings, part_ids, parts, args.max_calls)
        else:
            printed = {"last": -1}

            def progress(number: int, total: int) -> None:
                step = max(1, total // 10)
                if number // step != printed["last"]:
                    printed["last"] = number // step
                    print(f"  scanning parts: {number}/{total}", flush=True)

            scan = scan_parts(copy_settings, progress=progress)
        after = collect_metrics(copy_settings)
        after_parts = part_texts(copy_settings)
        print_report(
            source, copy_path, before, after, scan, live, args.max_calls
        )
        checks = verdict(before, after, scan, live, parts, after_parts)
        print("\n  criteria:")
        failed = 0
        for name, passed, detail in checks:
            if passed:
                print(f"  [PASS] {name}" + (f" — {detail}" if detail else ""))
            else:
                failed += 1
                print(f"  [FAIL] {name}" + (f" — {detail}" if detail else ""))
        print("\n=== VERDICT: " + ("PASS" if failed == 0 else f"FAIL ({failed})") + " ===")
        return 0 if failed == 0 else 1
    finally:
        if args.keep:
            print(f"\n  the snapshot copy is kept: {workdir}")
        else:
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
