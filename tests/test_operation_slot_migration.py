"""The migration of `worker_jobs.slot` to the operation names (release 3.9.0, LSBN-S-14).

    The `worker_jobs.slot` column carries the name of the SLOT GROUP today
    (`summary`, `judge`, `nodes`); in the `operation` mode the row must carry
    the name of the OPERATION (`merge`, `judge`, `classify`). The mapping is
    by the PAIR (slot, kind) — the design, B5:

        ('summary','merge') → 'merge'
        ('judge','dedup')   → 'judge'
        ('nodes','reclass') → 'classify'

    The criteria of the step:
      1. the three mappings are correct;
      2. the migration is idempotent (a repeated run changes nothing and
         writes no second journal record);
      3. an unknown combination stays in the old queue and is marked with
         attention — the work is not lost silently;
      4. the reverse map restores the moved rows (the rows marked with
         attention are not touched);
      5. the flag decides when the migration runs (in the `legacy` mode it
         does not run at all; in the `operation` mode it runs on an explicit
         call, never at the start);
      6. the schema of the columns does not change — the index
         `(slot, status, attention_at)` is added;
      7. the work is not lost: the rows, their statuses and their attention
         are the same after the migration, and the queues are readable.

    The `legacy` contour snapshot (the last test) was taken on the base commit
    `94b55cf` BEFORE this step — the `legacy` mode must not change in any
    respect (the script that produced it: `/tmp/lsb-s14-legacy-snapshot.py`).

    A unit without a network: the models are fakes, the encoder — `HashEmbedder`.
"""

from __future__ import annotations

import logging

import pytest
from fakes import (
    FixedClassifier,
    FixedSummarizer,
    HashEmbedder,
    ScriptedJudge,
)

from app.config import get_settings
from app.services.worker import (
    BackgroundWorker,
    ensure_worker_jobs_schema,
    migrate_slots,
)
from app.storage.db import init_db, session, transaction

DIM = 8

#: The columns of `worker_jobs` — the schema does not change (the criterion 6).
WORKER_JOBS_COLUMNS = [
    "id",
    "slot",
    "kind",
    "note_id",
    "payload",
    "status",
    "created_at",
    "updated_at",
    "attempts",
    "last_outcome",
    "last_error",
    "attention_at",
]

#: The `legacy` contour snapshot — taken on the base commit `94b55cf` BEFORE
#: this step. The script: `/tmp/lsb-s14-legacy-snapshot.py` (the same script on
#: the base tree and on the branch prints the same JSON).
LEGACY_SLOT_CONTOUR: dict = {
    "attention_by_slot": {"judge": 1},
    "attention_count": 1,
    "attention_jobs": [
        {"job": "judge", "note_id": 1, "outcome": None, "slot": "judge"}
    ],
    "columns": WORKER_JOBS_COLUMNS,
    "indexes": ["idx_worker_jobs_attention", "idx_worker_jobs_queue"],
    "jobs": [
        "areas",
        "judge",
        "links",
        "migration",
        "nodes",
        "summary",
        "vector",
        "warnings",
    ],
    "queue_empty": {"judge": True, "merge": False, "summary": False},
    "queues": {
        "judge_dedup": [],
        "nodes_reclass": [3],
        "summary_merge": [2],
    },
    "rows": [
        {"attention": True, "kind": "dedup", "note_id": 1, "slot": "judge",
         "status": "pending"},
        {"attention": False, "kind": "merge", "note_id": 2, "slot": "summary",
         "status": "pending"},
        {"attention": False, "kind": "reclass", "note_id": 3, "slot": "nodes",
         "status": "pending"},
    ],
}


def _settings(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    mode: str = "operation",
):
    """The test settings: own database, the slot model of the case."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "notes.db"))
    monkeypatch.setenv("EMBEDDING_DIM", str(DIM))
    monkeypatch.setenv("SLOT_PAUSE_AFTER", "1")
    monkeypatch.setenv("SLOT_PAUSE_SEC", "900")
    if mode == "operation":
        monkeypatch.setenv("SLOT_MODEL", "operation")
    else:
        monkeypatch.delenv("SLOT_MODEL", raising=False)
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings)
    # The queue table is the worker's own zone — it is created on the first
    # access (the same DDL the worker uses).
    ensure_worker_jobs_schema(settings)
    return settings


def _worker(settings) -> BackgroundWorker:
    """A worker with the fakes behind the operations."""
    return BackgroundWorker(
        settings,
        HashEmbedder(DIM),
        FixedSummarizer(),
        judge=ScriptedJudge(),
        classifier=FixedClassifier(),
    )


def _add_job(
    settings,
    slot: str,
    kind: str,
    note_id: int,
    *,
    status: str = "pending",
    attention: str | None = None,
) -> int:
    """A task row of the queue — the values the migration works on."""
    with session(settings) as conn, transaction(conn):
        conn.execute(
            "INSERT INTO worker_jobs (slot, kind, note_id, status, attention_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (slot, kind, note_id, status, attention),
        )
        return int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])


def _rows(settings) -> list[dict]:
    """The rows of `worker_jobs` (the order by id)."""
    with session(settings) as conn:
        return [
            {
                "id": int(row["id"]),
                "slot": row["slot"],
                "kind": row["kind"],
                "note_id": int(row["note_id"]),
                "status": row["status"],
                "attention_at": row["attention_at"],
            }
            for row in conn.execute(
                "SELECT id, slot, kind, note_id, status, attention_at "
                "FROM worker_jobs ORDER BY id"
            ).fetchall()
        ]


def _journal(caplog) -> list:
    """The journal records of the migration (`slot_migrated`)."""
    return [
        record
        for record in caplog.records
        if getattr(record, "event", None) == "slot_migrated"
    ]


# --- criterion 1: the three mappings are correct -----------------------------


def test_the_three_mappings_are_correct(tmp_path, monkeypatch) -> None:
    """The pair (slot, kind) -> the operation name: the three mappings.

    The `('judge','dedup') → 'judge'` mapping is the identity one — the
    operation name of the dedup judge IS the legacy slot group, so the row
    does not change its value (and the counter of the moved rows does not
    count it).
    """
    settings = _settings(tmp_path, monkeypatch)
    _add_job(settings, "summary", "merge", 1)
    _add_job(settings, "judge", "dedup", 2)
    _add_job(settings, "nodes", "reclass", 3)

    counters = migrate_slots(settings)

    slots = {row["kind"]: row["slot"] for row in _rows(settings)}
    assert slots == {"merge": "merge", "dedup": "judge", "reclass": "classify"}
    assert counters["migrated"] == 2  # the identity mapping moves nothing
    assert counters["unknown"] == 0
    assert counters["attention"] == 0


# --- criterion 2: the migration is idempotent --------------------------------


def test_the_migration_is_idempotent(tmp_path, monkeypatch, caplog) -> None:
    """Two runs in a row: the second changes nothing and writes no journal."""
    settings = _settings(tmp_path, monkeypatch)
    _add_job(settings, "summary", "merge", 1)
    _add_job(settings, "nodes", "reclass", 2)

    with caplog.at_level(logging.INFO, logger="app"):
        first = migrate_slots(settings)
        after_first = _rows(settings)
        second = migrate_slots(settings)
        after_second = _rows(settings)

    assert first["migrated"] == 2
    assert second["migrated"] == 0
    assert second["attention"] == 0
    assert after_first == after_second
    assert len(_journal(caplog)) == 1  # no second journal record


# --- criterion 3: an unknown combination is not lost silently -----------------


def test_an_unknown_combination_stays_and_is_marked_with_attention(
    tmp_path, monkeypatch, caplog
) -> None:
    """An unknown (slot, kind): the row stays and gets `attention_at`.

    The work is not lost silently: the row keeps its place in the old queue,
    it is marked with attention (the operator sees it in `/jobs/attention`
    and in the attention counter) and the journal of the migration reports
    the unknown combination.
    """
    settings = _settings(tmp_path, monkeypatch)
    _add_job(settings, "summary", "merge", 1)
    unknown_id = _add_job(settings, "summary", "unknown_kind", 2)

    with caplog.at_level(logging.INFO, logger="app"):
        counters = migrate_slots(settings)

    rows = {row["id"]: row for row in _rows(settings)}
    assert rows[unknown_id]["slot"] == "summary"  # it stays in the old queue
    assert rows[unknown_id]["attention_at"] is not None
    assert rows[1]["slot"] == "merge"  # the known pair is moved
    assert counters["unknown"] == 1
    assert counters["attention"] == 1

    # The row is visible in the attention counters and in the listing.
    worker = _worker(settings)
    assert worker.attention_count() == 1
    assert [job["job_id"] for job in worker.attention_jobs()] == [unknown_id]

    # And in the journal of the migration.
    records = _journal(caplog)
    assert len(records) == 1
    assert records[0].unknown == 1
    assert records[0].attention == 1
    assert records[0].migrated == 1


# --- criterion 4: the reverse map restores the moved rows --------------------


def test_the_reverse_map_restores_the_moved_rows(tmp_path, monkeypatch) -> None:
    """`migrate_slots(reverse=True)`: the moved rows return to their values.

    The reverse map is deterministic by `kind`; the rows marked with attention
    are not touched.
    """
    settings = _settings(tmp_path, monkeypatch)
    merge_id = _add_job(settings, "summary", "merge", 1)
    reclass_id = _add_job(settings, "nodes", "reclass", 2)
    unknown_id = _add_job(settings, "weird", "kind", 3)

    migrate_slots(settings)
    counters = migrate_slots(settings, reverse=True)

    rows = {row["id"]: row for row in _rows(settings)}
    assert rows[merge_id]["slot"] == "summary"
    assert rows[reclass_id]["slot"] == "nodes"
    # The row marked with attention is not touched.
    assert rows[unknown_id]["slot"] == "weird"
    assert rows[unknown_id]["attention_at"] is not None
    assert counters["migrated"] == 2


def test_the_reverse_does_not_touch_the_rows_that_were_not_moved(
    tmp_path, monkeypatch
) -> None:
    """Only the rows of the forward run are returned back.

    A row that carries the operation name but was NOT moved by the forward
    run (it appeared after it) keeps its value — the rollback returns only
    what it moved.
    """
    settings = _settings(tmp_path, monkeypatch)
    _add_job(settings, "summary", "merge", 1)
    migrate_slots(settings)
    later_id = _add_job(settings, "merge", "merge", 2)

    migrate_slots(settings, reverse=True)

    rows = {row["id"]: row for row in _rows(settings)}
    assert rows[later_id]["slot"] == "merge"


# --- criterion 5: the flag decides when the migration runs --------------------


def test_in_the_legacy_mode_the_migration_does_not_run(
    tmp_path, monkeypatch
) -> None:
    """The `legacy` mode: the values in the DB stay the old ones."""
    settings = _settings(tmp_path, monkeypatch, mode="legacy")
    _add_job(settings, "summary", "merge", 1)
    _add_job(settings, "nodes", "reclass", 2)

    counters = migrate_slots(settings)

    assert counters["skipped"] is True
    assert counters["reason"] == "legacy_mode"
    assert counters["migrated"] == 0
    assert [row["slot"] for row in _rows(settings)] == ["summary", "nodes"]


def test_in_the_operation_mode_the_migration_runs_on_an_explicit_call(
    tmp_path, monkeypatch
) -> None:
    """The `operation` mode: the migration runs on an explicit call only.

    The worker starts (the schema of the table, the queues) — nothing migrates
    itself; the migration is called explicitly.
    """
    settings = _settings(tmp_path, monkeypatch)
    _add_job(settings, "summary", "merge", 1)
    worker = _worker(settings)
    worker._ensure_job_table()

    assert [row["slot"] for row in _rows(settings)] == ["summary"]

    counters = migrate_slots(settings)

    assert counters["skipped"] is False
    assert counters["migrated"] == 1
    assert [row["slot"] for row in _rows(settings)] == ["merge"]


# --- criterion 6: the schema does not change, the index is added --------------


def test_the_columns_are_unchanged_and_the_new_index_exists(
    tmp_path, monkeypatch
) -> None:
    """The columns of `worker_jobs` are the same; the new index exists.

    The index `(slot, status, attention_at)` serves the attention counter
    (the design, B5); the queue index `(slot, kind, status, id)` stays valid
    (the slot is the leading column).
    """
    settings = _settings(tmp_path, monkeypatch)
    worker = _worker(settings)
    worker._ensure_job_table()

    with session(settings) as conn:
        columns = [
            row["name"]
            for row in conn.execute("PRAGMA table_info(worker_jobs)").fetchall()
        ]
        indexes: dict[str, list[str]] = {}
        for row in conn.execute("PRAGMA index_list(worker_jobs)").fetchall():
            name = row["name"]
            indexes[name] = [
                info["name"]
                for info in conn.execute(f"PRAGMA index_info({name})").fetchall()
            ]

    assert columns == WORKER_JOBS_COLUMNS
    assert indexes["idx_worker_jobs_slot_attention"] == [
        "slot",
        "status",
        "attention_at",
    ]
    assert indexes["idx_worker_jobs_queue"] == ["slot", "kind", "status", "id"]


# --- criterion 7: the work is not lost ---------------------------------------


def test_the_work_is_not_lost_after_the_migration(tmp_path, monkeypatch) -> None:
    """After the migration the rows, their statuses and their attention are
    the same — and the queues of the operations are readable.

    The reader covers both forms (the legacy value and the operation name):
    the migrated rows are taken by the same loops as before.
    """
    settings = _settings(tmp_path, monkeypatch)
    _add_job(settings, "summary", "merge", 1)
    _add_job(settings, "nodes", "reclass", 2)
    _add_job(settings, "judge", "dedup", 3)
    _add_job(settings, "summary", "merge", 4, attention="2026-01-01T00:00:00Z")
    _add_job(settings, "judge", "dedup", 5, status="done")

    before = _rows(settings)
    migrate_slots(settings)
    after = _rows(settings)

    assert len(after) == len(before) == 5
    assert [row["status"] for row in after] == [row["status"] for row in before]
    assert [row["attention_at"] for row in after] == [
        row["attention_at"] for row in before
    ]
    assert [row["kind"] for row in after] == [row["kind"] for row in before]
    # The values are the operation names now.
    assert [row["slot"] for row in after] == [
        "merge",
        "classify",
        "judge",
        "merge",
        "judge",
    ]

    worker = _worker(settings)
    # The queues are read under the new values — the work is not orphaned.
    assert [
        int(job["note_id"]) for job in worker._pending_jobs("summary", "merge", 10)
    ] == [1]
    assert [
        int(job["note_id"]) for job in worker._pending_jobs("nodes", "reclass", 10)
    ] == [2]
    assert [
        int(job["note_id"]) for job in worker._pending_jobs("judge", "dedup", 10)
    ] == [3]
    # The frozen pair is still frozen and still counted under its operation.
    assert worker._attention_by_operation() == {"merge": 1}


# --- the writer: a new task carries the operation name in the operation mode --


def test_a_new_task_carries_the_operation_name_in_the_operation_mode(
    tmp_path, monkeypatch
) -> None:
    """In the `operation` mode the new row carries the OPERATION name.

    The release's brief: "a slot = an operation — its own queue
    (`worker_jobs.slot` = the operation name)". The map is the reader's map of
    the pairs: the merge task — `merge`, the reclass task — `classify`, the
    dedup task — `judge`.
    """
    settings = _settings(tmp_path, monkeypatch)
    worker = _worker(settings)

    worker._create_job("summary", "merge", 1)
    worker._create_job("nodes", "reclass", 2)
    worker._create_job("judge", "dedup", 3)

    slots = {row["kind"]: row["slot"] for row in _rows(settings)}
    assert slots == {"merge": "merge", "reclass": "classify", "dedup": "judge"}


def test_a_new_task_carries_the_slot_group_in_the_legacy_mode(
    tmp_path, monkeypatch
) -> None:
    """In the `legacy` mode the new row carries the slot group, as before."""
    settings = _settings(tmp_path, monkeypatch, mode="legacy")
    worker = _worker(settings)

    worker._create_job("summary", "merge", 1)
    worker._create_job("nodes", "reclass", 2)
    worker._create_job("judge", "dedup", 3)

    slots = {row["kind"]: row["slot"] for row in _rows(settings)}
    assert slots == {"merge": "summary", "reclass": "nodes", "dedup": "judge"}


def test_a_pair_that_is_not_in_the_map_keeps_the_previous_value(
    tmp_path, monkeypatch
) -> None:
    """A pair outside the map: the row keeps the value it had.

    The work is not lost (the same rule as for the "hanging" rows of the
    migration) — the reader of the pair finds it under its own value.
    """
    settings = _settings(tmp_path, monkeypatch)
    worker = _worker(settings)

    worker._create_job("weird", "thing", 1)

    assert _rows(settings)[0]["slot"] == "weird"


# --- the `legacy` contour did not change -------------------------------------


def test_the_legacy_contour_is_unchanged(tmp_path, monkeypatch) -> None:
    """The `legacy` contour: the snapshot of the base commit `94b55cf`.

    The values the worker writes, the queues it reads, the attention counters
    and the schema of the table — the same as before the step.
    """
    settings = _settings(tmp_path, monkeypatch, mode="legacy")
    worker = _worker(settings)

    worker._create_job("judge", "dedup", 1)
    worker._create_job("summary", "merge", 2)
    worker._ensure_job("nodes", "reclass", 3)
    with session(settings) as conn, transaction(conn):
        conn.execute(
            "UPDATE worker_jobs SET attention_at = '2026-01-01T00:00:00Z' "
            "WHERE kind = 'dedup'"
        )

    with session(settings) as conn:
        columns = [
            row["name"]
            for row in conn.execute("PRAGMA table_info(worker_jobs)").fetchall()
        ]
        indexes = sorted(
            row["name"]
            for row in conn.execute("PRAGMA index_list(worker_jobs)").fetchall()
            if row["name"] != "idx_worker_jobs_slot_attention"
        )

    contour = {
        "rows": [
            {
                "slot": row["slot"],
                "kind": row["kind"],
                "note_id": row["note_id"],
                "status": row["status"],
                "attention": row["attention_at"] is not None,
            }
            for row in _rows(settings)
        ],
        "queues": {
            "judge_dedup": [
                int(job["note_id"])
                for job in worker._pending_jobs("judge", "dedup", 10)
            ],
            "summary_merge": [
                int(job["note_id"])
                for job in worker._pending_jobs("summary", "merge", 10)
            ],
            "nodes_reclass": [
                int(job["note_id"])
                for job in worker._pending_jobs("nodes", "reclass", 10)
            ],
        },
        "queue_empty": {
            "summary": worker._summary_queue_empty(),
            "merge": worker._merge_queue_empty(),
            "judge": worker._judge_queue_empty(),
        },
        "attention_by_slot": worker._attention_by_slot(),
        "attention_count": worker.attention_count(),
        "attention_jobs": [
            {
                "job": job["job"],
                "slot": job["slot"],
                "note_id": job["note_id"],
                "outcome": job["outcome"],
            }
            for job in worker.attention_jobs()
        ],
        "columns": columns,
        "indexes": indexes,
        "jobs": sorted(worker.queues_health()),
    }

    assert contour == LEGACY_SLOT_CONTOUR
