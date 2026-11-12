"""The storage (ARCHITECTURE §3.3): the SQLite schema notes + FTS5 (trigram) + vec0.

A layer without domain rules: only the schema, connections and transactions.
CRUD semantics (validations, statuses, soft delete) live in
`app.services.notes`; vector operations (vec0 serialization, KNN) — in
`app.storage.vectors` (Phase 3).

Key decisions:
- The schema is created idempotently at startup (`init_db`, `IF NOT EXISTS`).
- `notes_fts` — an external-content FTS5 (`content='notes'`,
  `content_rowid='id'`, `tokenize='trigram'`) indexing the note's TITLE and
  TEXT (the title and text columns — lsb-0001 FR-1.2/FR-1.3), kept in sync by
  AFTER INSERT / AFTER UPDATE OF text, title triggers. A DELETE trigger is
  unnecessary: deletion is soft (`deleted_at`), the row and the FTS index
  physically stay in trash.
- The title migration (lsb-0001-01, gate decision R4 — a startup migration,
  not a job): for live databases created before the title was indexed,
  notes_fts is rebuilt with the title column and re-poured from notes;
  notes_vec is dropped — the full vectors were computed over the text without
  the title and are invalid (every note, trash included, →
  vector_status='pending'; no vectors are encoded in this step — the
  background worker catches up); notes_chunks_vec is NOT touched: chunk
  vectors are built over chunk texts, the title does not affect them.
  Idempotency — the meta key `title_index_version` (the value "2").
- `notes_vec` — a vec0 table whose dimension is pinned at database creation
  (ARCH §3.3); when the configuration diverges from the one recorded in the
  DB (EMBEDDING_DIM, or an EMBEDDING_MODEL change stored in the meta table), a
  FULL auto-reindex runs at startup: the index is rebuilt, every note (trash
  included) becomes vector_status='pending' and is caught up by the
  background worker.
- `notes_chunks` + `notes_chunks_vec` (Phase 7): vectors are built over note
  chunks; the schema and operations — `app.storage.chunks`. The
  note_id+CASCADE FK and PRAGMA foreign_keys=ON live in session(); orphans
  left by the operator's direct edits are healed at startup. A chunk-parameter
  change (meta) — re-chunking (step 6); a model/dimension change — dropping
  BOTH vector indexes.
- WAL + busy_timeout (ARCH §3.3): reads are concurrent, writers serialize,
  lock contention resolves by waiting up to BUSY_TIMEOUT_MS rather than an
  instant "database is locked".
- One connection per operation (cheap for SQLite), so it is always created,
  used and closed within one thread (compatible with `asyncio.to_thread`).
  Transactions are explicit (`transaction()`, autocommit): BEGIN IMMEDIATE
  serializes writers through busy_timeout.
- `synchronous=NORMAL` — the standard WAL practice: survives a process crash;
  power-loss scenarios are covered by the periodic backup (Phase 5).
- At startup the FTS index is verified against `notes` (the operator has
  direct access to the DB file, REQUIREMENTS §4); desync is healed by a
  rebuild, other integrity errors are fatal (StorageError).
- Note links (lsb-0010-02, 3.1.0): the `links` table (one row per pair, the
  canonical order `note_a < note_b`) + the `notes.links_at` computed marker —
  an idempotent startup migration; an embedding model/dimension change (the
  auto-reindex below) clears the links too.
- The summary length limit (the 3.1.0 gate): active notes with a ready summary
  longer than MAX_SUMMARY_CHARS are marked for regeneration
  (`summary_status='pending'`) — the summary text is NOT damaged by
  truncation; the background `summary` job replaces it; idempotent (after
  regeneration a repeated start is a no-op), notes with
  `summary_status != 'ok'` are already in the queue.
- The 3.0.0 areas (the substrate, ARCH substrate §3.1–3.2): separate tables,
  FTS5 indexes and vec0 indexes for the skills/terms/user areas in the same
  DB. No note object is recreated; old records are not migrated (area data
  starts from zero). FTS — external content with AFTER INSERT / AFTER UPDATE
  triggers; `terms` carries a PARTIAL UNIQUE over live records
  (`WHERE deleted_at IS NULL`). Area vecs are dropped together with notes_vec
  in the model/dimension branch — every area record → pending, the worker's
  `areas` loop catches up.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import sqlite_vec

from app.config import Settings
from app.storage import area_vectors, chunks, expirations, vectors

# How long to wait for a write lock before giving up (both the sqlite3
# timeout and PRAGMA busy_timeout).
BUSY_TIMEOUT_MS = 5000

# How to pull the CHECK(length(text) BETWEEN 1 AND ?) limit out of the live
# notes DDL (checked against MAX_NOTE_CHARS at startup — see init_db).
_CHECK_LIMIT_RE = re.compile(r"length\(\s*text\s*\)\s+BETWEEN\s+1\s+AND\s+(\d+)")

# The "notes_fts already indexes title" marker in the sqlite_master DDL: the
# new schema starts with the title column, the legacy one (v2.1.1) — with
# text (the lsb-0001-01 migration — see _sync_fts_title_index; modeled after
# _PARTITION_RE in app.storage.vectors).
_FTS_TITLE_RE = re.compile(r"fts5\(\s*title")


class StorageError(RuntimeError):
    """A storage error: the DB is unavailable or the schema is damaged.

    Fatal at startup; while serving requests it reaches the transport layers
    as a regular exception (the response format belongs to Phase 5).
    """


# --- schema --------------------------------------------------------------

# The ARCHITECTURE §3.3 DDL draft (without the vec0 table — that is Phase 3).
# The CHECK limit is substituted from env at the first DB initialization
# (ARCH §3.3: "limits are substituted from env"); an integer is substituted —
# safe.
_NOTES_DDL = """
CREATE TABLE IF NOT EXISTS notes (
  id             INTEGER PRIMARY KEY,
  text           TEXT    NOT NULL CHECK(length(text) BETWEEN 1 AND {max_note_chars}),
  title          TEXT    NULL,
  comment        TEXT    NULL,
  summary        TEXT    NOT NULL DEFAULT '',
  author         TEXT    NOT NULL DEFAULT 'unknown',
  vector_status  TEXT    NOT NULL DEFAULT 'pending',
  summary_status TEXT    NOT NULL DEFAULT 'pending',
  created_at     TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
  updated_at     TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
  deleted_at     TEXT    NULL,
  namespace      TEXT    NOT NULL DEFAULT 'default',
  classified_at  TEXT    NULL,
  hint_path      TEXT    NULL,
  confidence     REAL    NULL,
  expires_at     TEXT    NULL,
  links_at       TEXT    NULL,
  node_order_at  TEXT    NULL,
  warnings_at    TEXT    NULL,
  origin_note_id INTEGER NULL  -- the origin group of a split/merge part, no FK
)
"""

# The namespace registry (Phase 10, REQUIREMENTS §5.7): hierarchy nodes (path
# — a slash path, at most 2 levels), description — required (the <=2-sentence
# contract is validated by the service), status — confirmed | provisional
# (auto-created nodes of the structure judge). Existing notes on migration go
# to 'default' (the column default + the node INSERT below).
_NAMESPACES_DDL = """
CREATE TABLE IF NOT EXISTS namespaces (
  path        TEXT PRIMARY KEY,
  description TEXT NOT NULL,
  status      TEXT NOT NULL DEFAULT 'confirmed'
              CHECK(status IN ('confirmed', 'provisional')),
  created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
  updated_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
)
"""

# lsb-0001-01 (FR-1.2/FR-1.3): FTS indexes both the title and the text.
_FTS_DDL = """
CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5(
  title, text, content='notes', content_rowid='id', tokenize='trigram'
)
"""

# Structure-judge verdicts (Phase 10, Step 5): the trigger cooldown. A
# hint_path group the judge has already ruled on (merge/reject) does not call
# the LLM again — otherwise a rejected candidate would loop judge calls on
# every run (notes with a rejected hint stay in default forever — honestly
# shared). A merge stores the canonical node; a verdict reset belongs to the
# operator (REST, Step 6). A created node needs no record:
# requires: the group leaves default by retro-placement.
#
# lsb-0005-03: the key is the single full placement path hint_path (1..3
# levels), replacing the (domain, subdomain) pair. The old schema (the
# (domain, subdomain) PK) is migrated in _migrate_promotions_key — existing
# verdicts get hint_path = domain/subdomain.
_PROMOTIONS_DDL = """
CREATE TABLE IF NOT EXISTS promotions (
  hint_path      TEXT NOT NULL,
  status         TEXT NOT NULL CHECK(status IN ('merged', 'rejected')),
  canonical_path TEXT NULL,
  decided_at     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
  PRIMARY KEY (hint_path)
)
"""

# lsb-0004-02 (FR-5…FR-8): the expired-note deletion queue. note_id is the PK
# (one row per note), expires_at — the absolute ISO-8601 (UTC) lifetime.
# Created idempotently (IF NOT EXISTS); synchronization with notes is at the
# service level (insert on a TTL, update on a change, delete on clear/note
# deletion).
_NOTE_EXPIRATIONS_DDL = """
CREATE TABLE IF NOT EXISTS note_expirations (
  note_id    INTEGER PRIMARY KEY,
  expires_at TEXT    NOT NULL
)
"""

# Note links, level 1 (lsb-0010-02, arch §3.2): one row per note pair; a pair
# is stored in the canonical order `note_a < note_b` (a link is symmetric and
# is read both ways), `kind` is the link kind with the priority
# mention > entities > cosine, `score` — the cosine/shared-word share (NULL
# for mention). Rows are written only by the computation
# (`LinksService.compute_for_note`) — one transaction per note: delete the
# pairs → insert them again.
_LINKS_DDL = """
CREATE TABLE IF NOT EXISTS links (
  note_a     INTEGER NOT NULL,          -- the smaller id of the pair
  note_b     INTEGER NOT NULL,          -- the larger id of the pair
  kind       TEXT    NOT NULL,          -- 'mention' | 'entities' | 'cosine'
  score      REAL,                      -- the cosine / shared-word share / NULL
  created_at TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
  PRIMARY KEY (note_a, note_b)
)
"""

# The index over the pair's other side: output and recomputation look up a
# note's links both ways (`note_a = ? OR note_b = ?`), the PRIMARY KEY covers
# only note_a.
_LINKS_INDEX_DDL = "CREATE INDEX IF NOT EXISTS idx_links_b ON links(note_b)"

# The warnings table (No12 + No32, arch §5.1): ONE shared warnings surface
# for memory and terms, readable without a model call. Substrate (the shared
# `meta` precedent): access by the key `(subject_type, subject_id)`; the area
# isolation SQL never addresses it. At most one OPEN warning per (subject,
# kind) — a repeated job run updates `updated_at`/`payload` instead of
# spawning rows (§13.7).
_WARNINGS_DDL = """
CREATE TABLE IF NOT EXISTS warnings (
  id           INTEGER PRIMARY KEY,
  kind         TEXT    NOT NULL,
  subject_type TEXT    NOT NULL,
  subject_id   INTEGER NOT NULL,
  message      TEXT    NOT NULL,
  payload      TEXT    NOT NULL DEFAULT '{}',
  status       TEXT    NOT NULL DEFAULT 'open' CHECK(status IN ('open','dismissed')),
  created_at   TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
  updated_at   TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
  resolved_at  TEXT    NULL
)
"""

# A partial unique index: at most one OPEN warning per (subject, kind) — the
# upsert idempotency (§5.1, §13.7).
_WARNINGS_UNIQUE_INDEX_DDL = (
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_warnings_open_unique "
    "ON warnings(subject_type, subject_id, kind) "
    "WHERE status = 'open' AND resolved_at IS NULL"
)

# The working index of listing/aggregates (/health.warnings, GET /warnings).
_WARNINGS_STATUS_INDEX_DDL = (
    "CREATE INDEX IF NOT EXISTS idx_warnings_status_kind "
    "ON warnings(status, kind, id)"
)

# Vectors (Phase 3): the physical schema lives in `app.storage.vectors` (the
# dimension and the cosine metric — there); init_db — creation/verification at
# startup. The embedding model the index was built with is stored in the meta
# table (see below): a model/dimension change over a live DB = an automatic
# reindex.
_META_DDL = """
CREATE TABLE IF NOT EXISTS meta (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
)
"""

# FTS synchronization with notes. For external content, removing from the
# index is the special 'delete' command with the OLD values of the indexed
# columns. UPDATE OF text, title (lsb-0001-01): changes of other columns
# (summary, statuses, deleted_at on soft delete) do not touch FTS.
_TRIGGERS = (
    """
    CREATE TRIGGER IF NOT EXISTS notes_fts_ai AFTER INSERT ON notes BEGIN
      INSERT INTO notes_fts(rowid, title, text) VALUES (new.id, new.title, new.text);
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS notes_fts_au AFTER UPDATE OF text, title ON notes BEGIN
      INSERT INTO notes_fts(notes_fts, rowid, title, text) VALUES ('delete', old.id, old.title, old.text);
      INSERT INTO notes_fts(rowid, title, text) VALUES (new.id, new.title, new.text);
    END
    """,
)

# No24 (no24-02, 3.2.0): a SEPARATE FTS index of comments (arch §5.2). The
# semantic `notes_fts` indexes title/text and does NOT see comments: comments
# are searched separately (MCP `memory_comment_search` / REST
# `GET /notes/comments`) — the results never mix (§13.6).
_COMMENT_FTS_DDL = """
CREATE VIRTUAL TABLE IF NOT EXISTS notes_comment_fts USING fts5(
  comment, content='notes', content_rowid='id', tokenize='trigram'
)
"""

# Comment-index synchronization with notes (modeled after _TRIGGERS): INSERT
# always (including comment IS NULL — "no comment"), UPDATE OF comment — the
# special 'delete' command with the OLD value + inserting the new one. No
# DELETE trigger (like notes_fts): notes are soft-deleted; the hard deletion
# is the TTL cleanup job (lsb-0004-02), and the comment index is cleaned there
# by the special 'delete' command in `delete_note_physical` (LSBN-D-8, 3.4.0).
_COMMENT_TRIGGERS = (
    """
    CREATE TRIGGER IF NOT EXISTS notes_comment_fts_ai AFTER INSERT ON notes BEGIN
      INSERT INTO notes_comment_fts(rowid, comment) VALUES (new.id, new.comment);
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS notes_comment_fts_au AFTER UPDATE OF comment ON notes BEGIN
      INSERT INTO notes_comment_fts(notes_comment_fts, rowid, comment) VALUES ('delete', old.id, old.comment);
      INSERT INTO notes_comment_fts(rowid, comment) VALUES (new.id, new.comment);
    END
    """,
)


# --- the 3.0.0 areas (substrate: skills / terms / user) ----------------------

# Once per release: an isolated area layer inside the shared notes.db.
# No area SQL ever addresses notes* (isolation is tested both ways).
SKILLS_TABLE = "skills"
SKILLS_FTS_TABLE = "skills_fts"
SKILLS_FTS_COLUMNS = ("name", "description", "steps", "text")
TERMS_TABLE = "terms"
TERMS_FTS_TABLE = "terms_fts"
TERMS_FTS_COLUMNS = ("term", "context", "definition")
USER_FACTS_TABLE = "user_facts"
USER_FACTS_FTS_TABLE = "user_facts_fts"
USER_FACTS_FTS_COLUMNS = ("name", "body")

# The "Assistant logs" area (LSBN-F-10, 3.2.0, arch §5.1/§5.2): one record
# table + an FTS5 index over the same semantic text fields. The structural
# filters (`entity*`, `source`, `session`, `created_at`) are plain columns and
# stay out of FTS.
ASSISTANT_LOGS_TABLE = "assistant_logs"
ASSISTANT_LOGS_FTS_TABLE = "assistant_logs_fts"
ASSISTANT_LOGS_FTS_COLUMNS = ("action", "reason", "entity_ref", "outcome", "verified")

# Area records (without vec tables): the common columns + the area's own fields.
AREA_RECORD_TABLES = (
    SKILLS_TABLE,
    TERMS_TABLE,
    USER_FACTS_TABLE,
    ASSISTANT_LOGS_TABLE,
)

# (record table, FTS table, indexed columns) — the FTS/trigger registry: one
# DDL and one trigger set per area (no per-area copy-paste).
_AREA_FTS_SPECS = (
    (SKILLS_TABLE, SKILLS_FTS_TABLE, SKILLS_FTS_COLUMNS),
    (TERMS_TABLE, TERMS_FTS_TABLE, TERMS_FTS_COLUMNS),
    (USER_FACTS_TABLE, USER_FACTS_FTS_TABLE, USER_FACTS_FTS_COLUMNS),
    (ASSISTANT_LOGS_TABLE, ASSISTANT_LOGS_FTS_TABLE, ASSISTANT_LOGS_FTS_COLUMNS),
)

# The mandatory "core" tables for the snapshot check (no14-01, 3.2.0): the
# names of every non-vector table `init_db` creates in the production DB (the
# notes core + the area records/FTS). The vector (vec0) tables live in their
# own registries (`vectors.VEC_TABLE`, `chunks.CHUNKS_VEC_TABLE`,
# `area_vectors.VEC_TABLES`) and are not listed here: `BackupService.verify`
# assembles the mandatory set from all `app.storage` registries instead of
# hardcoding it. The single source of core names — this tuple.
CORE_TABLES: tuple[str, ...] = (
    "notes",
    "notes_fts",
    "notes_comment_fts",
    "namespaces",
    "promotions",
    "note_expirations",
    "links",
    "warnings",
    "meta",
    chunks.CHUNKS_TABLE,
    "record_versions",
    SKILLS_TABLE,
    SKILLS_FTS_TABLE,
    "skill_versions",
    "skills_meta",
    TERMS_TABLE,
    TERMS_FTS_TABLE,
    USER_FACTS_TABLE,
    USER_FACTS_FTS_TABLE,
    ASSISTANT_LOGS_TABLE,
    ASSISTANT_LOGS_FTS_TABLE,
)

# Record fields — per the lsb-0007/0008/0009 feature architectures; common to
# all — id, vector_status (default pending), created_at, updated_at,
# deleted_at. No CHECK constraints, unlike notes: form limits are service
# validation (a soft refusal with a hint), the DDL carries structure only.
_AREA_DDL = (
    """
CREATE TABLE IF NOT EXISTS skills (
  id            INTEGER PRIMARY KEY,
  name          TEXT    NOT NULL DEFAULT '',
  description   TEXT    NOT NULL DEFAULT '',
  example       TEXT    NULL,
  steps         TEXT    NOT NULL DEFAULT '',
  text          TEXT    NOT NULL DEFAULT '',
  extra         TEXT    NULL,
  version       INTEGER NOT NULL DEFAULT 1,
  vector_status TEXT    NOT NULL DEFAULT 'pending',
  created_at    TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
  updated_at    TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
  deleted_at    TEXT    NULL
)
""",
    """
CREATE TABLE IF NOT EXISTS skill_versions (
  skill_id    INTEGER NOT NULL,
  version     INTEGER NOT NULL,
  name        TEXT    NOT NULL DEFAULT '',
  description TEXT    NOT NULL DEFAULT '',
  example     TEXT    NULL,
  steps       TEXT    NOT NULL DEFAULT '',
  text        TEXT    NOT NULL DEFAULT '',
  extra       TEXT    NULL,
  created_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
  PRIMARY KEY (skill_id, version)
)
""",
    """
CREATE TABLE IF NOT EXISTS skills_meta (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
)
""",
    """
CREATE TABLE IF NOT EXISTS terms (
  id            INTEGER PRIMARY KEY,
  term          TEXT    NOT NULL DEFAULT '',
  term_norm     TEXT    NOT NULL DEFAULT '',
  context       TEXT    NOT NULL DEFAULT '',
  context_norm  TEXT    NOT NULL DEFAULT '',
  definition    TEXT    NOT NULL DEFAULT '',
  vector_status TEXT    NOT NULL DEFAULT 'pending',
  created_at    TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
  updated_at    TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
  deleted_at    TEXT    NULL
)
""",
    """
CREATE TABLE IF NOT EXISTS user_facts (
  id            INTEGER PRIMARY KEY,
  name          TEXT    NOT NULL DEFAULT '',
  body          TEXT    NOT NULL DEFAULT '',
  vector_status TEXT    NOT NULL DEFAULT 'pending',
  created_at    TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
  updated_at    TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
  deleted_at    TEXT    NULL
)
""",
    """
CREATE TABLE IF NOT EXISTS assistant_logs (
  id             INTEGER PRIMARY KEY,
  action         TEXT    NOT NULL DEFAULT '',   -- what was done (required, <= LOGS_ACTION_MAX_CHARS)
  reason         TEXT    NOT NULL DEFAULT '',   -- why (required, <= LOGS_REASON_MAX_CHARS)
  entity         TEXT    NULL,                  -- the object-of-change kind (a closed dictionary, §5.4)
  entity_id      INTEGER NULL,                  -- the object id within its area (versioned kinds)
  entity_version INTEGER NULL,                  -- the object version at write time (No41)
  entity_ref     TEXT    NULL,                  -- a text reference/path (id-less objects; <= LOGS_ENTITY_REF_MAX_CHARS)
  outcome        TEXT    NULL,                  -- the result (what came out; <= LOGS_OUTCOME_MAX_CHARS)
  verified       TEXT    NULL,                  -- what was checked (<= LOGS_VERIFIED_MAX_CHARS)
  source         TEXT    NOT NULL DEFAULT 'mcp',-- the write channel: mcp|rest|job:<name>|merge|migration
  session        TEXT    NULL,                  -- a call/agent identifier, not a person's name
  vector_status  TEXT    NOT NULL DEFAULT 'pending',
  created_at     TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
  updated_at     TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
  deleted_at     TEXT    NULL
)
""",
)

# The skills-area global template (lsb-0007 §3.1/§3.8): one skills_meta record
# — "how to execute the steps" (never duplicated per skill). The seed key and
# text: the value is created only when the key is absent (the edit belongs to
# the operator via REST, lsb-0007-05; the seed never overwrites it).
INSTRUCTION_TEMPLATE_KEY = "instruction_template"
INSTRUCTION_TEMPLATE_SEED = (
    "Execute the steps in order: the content of each step in `text` says what "
    "exactly to do (result, format, rule). Do not skip or reorder steps; if a "
    "step cannot be executed, stop and report what is missing instead of "
    "improvising."
)

# The skill-creator seed (lsb-0007 §3.6/§3.8): a `skills` record — the "how to
# create skills" procedure; the texts are the arch-doc §3.8 canon verbatim (in
# the doc they are wrapped at ~80 chars — layout only; lines are joined with
# spaces, like INSTRUCTION_TEMPLATE_SEED). `extra` is empty,
# `vector_status='pending'` — the worker's `areas` loop catches the vector up
# (substrate §3.3).
# The seed marker in `skills_meta` is written at the first seed attempt: while
# it exists, init_db does NOT resurrect the seed — deleting the skill (by the
# operator or a model) stays meaningful (the marker outlives the soft-deleted
# row in the DB).
CREATOR_SKILL_SEED_KEY = "creator_skill_seed"
CREATOR_SKILL_SEED_VALUE = "seeded"
CREATOR_SKILL_NAME = "Create skills"
CREATOR_SKILL_DESCRIPTION = "How to add or update a skill in this memory"
CREATOR_SKILL_STEPS = (
    "1) search for a similar skill; 2) name and description; 3) steps and "
    "text; 4) save; 5) read back and check."
)
CREATOR_SKILL_TEXT = (
    "1) Call skills_search with the task wording: a similar skill exists → "
    "update it (skills_save with id), never create a duplicate. 2) name ≤65 "
    "characters (≤5 words recommended); description ≤400 — what the procedure "
    "does. 3) steps ≤1000 — the order (what after what); text ≤7500 — what "
    "exactly each step does (result/format/rule); keep the \"how to execute\" "
    "wording in the global instruction_template, never duplicate it per skill. "
    "4) Optional class fields (trigger, mode, preconditions, fallbacks, "
    "invariant, exceptions, guardrails, references, output_contract, "
    "behavior_contract) and example (≤2000) — only when this skill class needs "
    "them. 5) skills_save, then skills_get to check that the procedure reads "
    "as a ready-to-execute routine. 6) Self-improvement: when you are sure a "
    "skill can be improved, propose the exact edit (what and why) and ask the "
    "user for approval; apply it only after approval — the server keeps the "
    "previous version as a copy automatically."
)

# A partial UNIQUE (lsb-0008 §3.3): the term key (term_norm + context_norm) is
# unique among LIVE records only — a soft-deleted row does not hold the key
# (the deleted key is freed; "the undo belongs to the operator").
_TERMS_ACTIVE_KEY_INDEX_DDL = (
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_terms_key_active "
    "ON terms(term_norm, context_norm) WHERE deleted_at IS NULL"
)

# The logs-area reverse-lookup indexes (arch §5.1, FR-4.1/FR-5.2): both
# partial — over live records only (the idx_terms_key_active precedent).
# The first — "why did the object change"; the second — the absolute-date
# slice. No separate session/source indexes: a narrow domain, a small volume
# (§4).
_ASSISTANT_LOGS_INDEX_DDL = (
    "CREATE INDEX IF NOT EXISTS idx_assistant_logs_entity_active "
    "ON assistant_logs(entity, entity_id, created_at DESC) WHERE deleted_at IS NULL",
    "CREATE INDEX IF NOT EXISTS idx_assistant_logs_created_active "
    "ON assistant_logs(created_at DESC) WHERE deleted_at IS NULL",
)

# Version history (No41, 3.2.0, substrate-versioning §5.1): one shared table
# for every entity, the PK `(entity, entity_id, version)` covers the per-record
# lookup and the reverse order (newest first). No indexes beyond the PK (the
# §5.1 decision). `version=1` is the baseline (`snapshot` filled, `delta`
# NULL); `version>=2` is a delta, except a re-based one after prune.
_RECORD_VERSIONS_DDL = """
CREATE TABLE IF NOT EXISTS record_versions (
  entity       TEXT    NOT NULL,                     -- 'note'|'skill'|'skill_template'|'term'|'user_fact'
  entity_id    INTEGER NOT NULL,                     -- the live-table row id; 0 for skill_template
  version      INTEGER NOT NULL,                     -- 1..N, baseline = 1
  event        TEXT    NOT NULL DEFAULT 'content'
               CHECK(event IN ('baseline','content','delete','restore','rollback')),
  source       TEXT    NOT NULL,                     -- 'mcp'|'rest'|'job:<name>'|'merge'|'migration'
  session      TEXT    NULL,                         -- a call/agent identifier, not a person's name
  changed      TEXT    NOT NULL DEFAULT '[]',        -- a JSON array of the changed field names
  snapshot     TEXT    NULL,                         -- a JSON object: whole fields (baseline / after prune)
  delta        TEXT    NULL,                         -- a JSON object: field → value/lines (§5.2)
  restore_from INTEGER NULL,                         -- a rollback: the restored version number
  created_at   TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
  PRIMARY KEY (entity, entity_id, version)
)
"""


def _area_fts_ddl(table: str, fts_table: str, columns: tuple[str, ...]) -> str:
    """The area FTS5 index DDL: external content, trigram (like notes_fts)."""
    return (
        f"CREATE VIRTUAL TABLE IF NOT EXISTS {fts_table} USING fts5("
        f"  {', '.join(columns)}, content='{table}', content_rowid='id', "
        "tokenize='trigram')"
    )


def _area_trigger_ddls(
    table: str, fts_table: str, columns: tuple[str, ...]
) -> tuple[str, ...]:
    """The area FTS sync triggers: AFTER INSERT and AFTER UPDATE.

    No DELETE trigger: an area deletion is soft (`deleted_at`), the row and
    the FTS index physically stay in trash (the notes precedent). For external
    content, removing from the index is the special 'delete' command with the
    OLD values of the indexed columns (in UPDATE only).
    """
    joined = ", ".join(columns)
    new_values = ", ".join(f"new.{column}" for column in columns)
    old_values = ", ".join(f"old.{column}" for column in columns)
    return (
        f"""
CREATE TRIGGER IF NOT EXISTS {fts_table}_ai AFTER INSERT ON {table} BEGIN
  INSERT INTO {fts_table}(rowid, {joined}) VALUES (new.id, {new_values});
END
""",
        f"""
CREATE TRIGGER IF NOT EXISTS {fts_table}_au AFTER UPDATE ON {table} BEGIN
  INSERT INTO {fts_table}({fts_table}, rowid, {joined}) VALUES ('delete', old.id, {old_values});
  INSERT INTO {fts_table}(rowid, {joined}) VALUES (new.id, {new_values});
END
""",
    )


def _create_area_schema(conn: sqlite3.Connection, settings: Settings) -> None:
    """Create the area schema at startup — idempotently (ARCH substrate §3.1).

    Record tables + external-content FTS5 + triggers + the partial UNIQUE of
    the terms key + the vec0 tables of the current dimension. Live databases
    upgrade without manual migrations (everything is `IF NOT EXISTS`);
    existing note objects are not touched at all.
    """
    for ddl in _AREA_DDL:
        conn.execute(ddl)
    for table, fts_table, columns in _AREA_FTS_SPECS:
        conn.execute(_area_fts_ddl(table, fts_table, columns))
        for trigger in _area_trigger_ddls(table, fts_table, columns):
            conn.execute(trigger)
    conn.execute(_TERMS_ACTIVE_KEY_INDEX_DDL)
    for ddl in _ASSISTANT_LOGS_INDEX_DDL:
        conn.execute(ddl)
    area_vectors.create_vec_tables(conn, settings.embedding_dim)


def _create_area_vec_if_missing(conn: sqlite3.Connection, dim: int) -> None:
    """Create the missing area vecs (pre-3.0.0 inheritors, manual edits)."""
    if not area_vectors.vec_tables_exist(conn):
        area_vectors.create_vec_tables(conn, dim)


def _reset_area_vectors(conn: sqlite3.Connection, settings: Settings) -> None:
    """A model/dimension change: area vecs are dropped and recreated.

    Another model's vectors are incompatible: every area record (trash
    included) → `vector_status='pending'`, the worker's `areas` loop catches
    up — modeled after notes_vec (NFR-3: no data is lost, search degrades to
    FTS).
    """
    area_vectors.drop_vec_tables(conn)
    area_vectors.create_vec_tables(conn, settings.embedding_dim)
    for table in AREA_RECORD_TABLES:
        conn.execute(f"UPDATE {table} SET vector_status = 'pending'")


def _count_area_records(conn: sqlite3.Connection) -> int:
    """The number of area records (auto-reindex events, observation)."""
    return sum(
        int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        for table in AREA_RECORD_TABLES
    )


# --- connections ---------------------------------------------------------

@contextmanager
def session(settings: Settings) -> Iterator[sqlite3.Connection]:
    """One connection per operation + the WAL pragmas (ARCH §3.3).

    Autocommit (`isolation_level=None`): every statement is atomic on its
    own, and multi-statement changes are wrapped in `transaction()`.
    """
    try:
        conn = sqlite3.connect(
            settings.db_path,
            timeout=BUSY_TIMEOUT_MS / 1000,
            isolation_level=None,
        )
        conn.row_factory = sqlite3.Row  # name-based cell access
    except sqlite3.Error as exc:  # for example, the path is unreachable
        raise StorageError(
            f"не удалось открыть БД {settings.db_path}: {exc}"
        ) from exc
    try:
        conn.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
        # Phase 7: FK ON — without it the notes_chunks ON DELETE CASCADE
        # silently does not fire (SQLite ships with foreign keys off).
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        _load_vec_extension(conn)
        yield conn
    except sqlite3.Error as exc:
        raise StorageError(f"ошибка БД ({settings.db_path}): {exc}") from exc
    finally:
        conn.close()


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """A multi-statement write: BEGIN IMMEDIATE … COMMIT, ROLLBACK on error.

    BEGIN IMMEDIATE takes the write lock right away — two concurrent writers
    serialize through busy_timeout without an upgrade deadlock.
    """
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")


# --- the sqlite-vec extension ------------------------------------------------

def _load_vec_extension(conn: sqlite3.Connection) -> None:
    """Load sqlite-vec into the connection (needed on every connection).

    Extension loading is a property of the connection, not of the DB file; the
    cost is microseconds. Without the extension the vec0 tables do not open at
    all, so a load failure is a StorageError (fatal at startup).
    """
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
    except (AttributeError, sqlite3.Error) as exc:
        raise StorageError(
            f"расширение sqlite-vec недоступно (vec0): {exc}"
        ) from exc


# --- initialization ------------------------------------------------------


def note_chars_limit(conn: sqlite3.Connection) -> int | None:
    """The live CHECK(length(text) BETWEEN 1 AND ?) limit of the notes table.

    None — the table does not exist yet (a fresh DB) or there is no CHECK (a
    hand-made schema): nothing to compare against. The source of truth is the
    DDL in sqlite_master, not meta: the limit is baked into the schema at
    first creation.
    """
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'notes'"
    ).fetchone()
    if row is None or row[0] is None:
        return None  # no table — init_db failed above with its own error
    match = _CHECK_LIMIT_RE.search(row[0])
    if match is None:
        return None  # no CHECK (a hand-made schema) — nothing to compare
    return int(match.group(1))


def _assert_note_chars_limit(conn: sqlite3.Connection, expected: int) -> int | None:
    """Compare the expected limit with the live CHECK of notes — WITHOUT a fatal.

    History: before LSBN-F-22 an env/CHECK mismatch killed the start (the
    limit is pinned in the schema, a CHECK cannot change on the fly).
    Self-migration (LSBN-F-22) replaces the fatal: the function returns the
    live CHECK limit (None — nothing to compare), and the pour decision is
    made by `app.storage.migration` next to `init_db` in the lifespan. Here —
    observability only: a divergence is a startup-log warning.

    Returns:
        The live CHECK limit, or None (no table/CHECK).
    """
    stored = note_chars_limit(conn)
    if stored is not None and stored != expected:
        logging.getLogger("app").warning(
            "MAX_NOTE_CHARS разошёлся с живым CHECK; старт продолжается, "
            "решение о самомиграции — при старте сервиса (LSBN-F-22)",
            extra={
                "event": "note_chars_mismatch",
                "schema_limit": stored,
                "env_limit": expected,
            },
        )
    return stored


def init_db(settings: Settings, *, note_chars: int | None = None) -> None:
    """Create the schema at startup (the Phase 2 acceptance test); idempotent.

    `note_chars` — the CHECK(length(text)) limit FOR A NEW database
    (LSBN-F-22, M1): None — from env (`settings.max_note_chars`), as before.
    Self-migration creates the target-limit base through this parameter (a
    Settings copy with another db_path is not needed for the limit). Over an
    existing DB the parameter plays no role (IF NOT EXISTS), and an env/CHECK
    divergence is no longer fatal: `_assert_note_chars_limit` returns the live
    limit, the pour decision lives in `app.storage.migration` (lifespan).

    An EMBEDDING_MODEL or EMBEDDING_DIM change over an existing DB
    automatically rebuilds BOTH vector indexes (the full text and the chunks):
    another model's vectors are incompatible, every note goes to pending and
    is caught up by the worker (NFR-3 — no data is lost, search degrades to
    FTS until ready).

    The title-index migration (lsb-0001-01, gate decision R4 — at startup,
    not a job): the notes_fts of live databases is rebuilt to index the title
    (title + text, FR-1.2/FR-1.3), notes_vec is dropped — the full vectors
    were computed over the text without the title and are invalid, every note
    (trash included) goes to pending and is caught up by the worker; chunk
    vectors (notes_chunks_vec) stay valid — they are built over chunk texts.

    Raises:
        StorageError: the DB is unavailable, no FTS5/vec0, a damaged schema.
    """
    limit = settings.max_note_chars if note_chars is None else note_chars
    path = Path(settings.db_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with session(settings) as conn:
            conn.execute(_NOTES_DDL.format(max_note_chars=limit))
            # The CHECK limit is baked into the DDL at first creation
            # (ARCH §3.3: "limits are substituted from env"). Before LSBN-F-22
            # an env divergence killed the start; now the live limit is
            # returned (observability — a warning), the pour decision lives in
            # app.storage.migration.
            _assert_note_chars_limit(conn, limit)
            conn.execute(_FTS_DDL)
            for trigger in _TRIGGERS:
                conn.execute(trigger)
            # Note titles (Phase 11): notes.title TEXT (nullable).
            # BEFORE the FTS title migration: by the time _sync_fts_title_index
            # runs, the title column already exists — the index pour reads
            # notes.title.
            _migrate_title_column(conn)
            # The note comment (No24, 3.2.0): notes.comment TEXT (nullable).
            # The column is needed before the service inserts/reads; the
            # comment FTS is no24-02.
            _migrate_comment_column(conn)
            # The origin group of parts (3.6.2): notes.origin_note_id INTEGER
            # (nullable, no FK) — right after the other zero ADD COLUMN
            # migrations, before anything reads the notes row whole.
            _migrate_origin_note_column(conn)
            # The comment FTS (No24, no24-02): the separate notes_comment_fts
            # table + triggers — right after the comment column and BEFORE the
            # embedding check (it never addresses the table). No rebuild
            # needed: there are no comments yet.
            _migrate_comment_fts(conn)
            # Namespaces (Phase 10): the notes columns + the registry + the
            # default node.
            _migrate_namespace_columns(conn)
            _migrate_classification_columns(conn)
            # lsb-0005-02: the single full placement path hint_path (replacing
            # the domain_hint/subdomain_hint pair) — a backfill from the old
            # signature.
            _migrate_hint_path_columns(conn)
            conn.execute(_NAMESPACES_DDL)
            # Lifetimes (lsb-0004-02): the notes.expires_at column + the
            # note_expirations deletion queue (idempotent).
            _migrate_expiration_columns(conn)
            # Note links (lsb-0010-02): the links table + index + the
            # notes.links_at marker — idempotent, before the auto-reindex
            # branch (it clears links on a model/dimension change).
            _migrate_link_columns(conn)
            # Order in nodes (lsb-0011-01): the notes.node_order_at service
            # marker — idempotent; existing notes have NULL (the first `nodes`
            # run sweeps the accumulated default — a retro sweep, arch §4).
            _migrate_node_order_columns(conn)
            # Warnings (No12, 3.2.0): the notes.warnings_at service marker —
            # idempotent; existing notes have NULL (the first `warnings` run
            # checks the accumulated ones).
            _migrate_warnings_columns(conn)
            # The summary length limit (the 3.1.0 gate): ready summaries longer
            # than MAX_SUMMARY_CHARS → pending (a background regeneration).
            _migrate_summary_length(conn, settings)
            # List indexes (pool 15): the old idx_notes_namespace replaced
            # (the namespace prefix; deleted_at is covered by the new ns
            # index).
            conn.execute("DROP INDEX IF EXISTS idx_notes_namespace")
            conn.execute(_INDEX_DELETED_UPDATED_DDL)
            conn.execute(_INDEX_NS_DELETED_UPDATED_DDL)
            _ensure_default_namespace(conn)
            # No34 (3.2.0): the `default` description is EN; existing databases
            # are brought over one-off and only when the text exactly equals the
            # former RU default (an operator edit is preserved). The stamp in
            # `meta` — see the function.
            _migrate_default_namespace_description(conn)
            conn.execute(_PROMOTIONS_DDL)
            # lsb-0005-03: the promotions key (domain, subdomain) → the single
            # hint_path; the old-verdict migration (idempotent).
            _migrate_promotions_key(conn)
            # The title index (lsb-0001-01, Part A): rebuilding notes_fts for
            # title indexing on live databases — BEFORE the integrity check:
            # the pour above leaves the index consistent.
            fts_migrated = _sync_fts_title_index(conn)
            _check_fts_integrity(conn)
            # The namespace partition (Phase 10): the live databases' vec
            # tables without `+ns` are recreated with the partition, every note
            # — to pending. BEFORE the model/chunk checks: both branches create
            # the tables already partitioned.
            _sync_namespace_partition(conn, settings)
            # Chunks (Phase 7): the chunk-text table — before the vector
            # check; on a configuration mismatch notes_chunks_vec is dropped
            # too.
            chunks.create_table(conn)
            # The 3.0.0 areas (substrate): the area tables/FTS/indexes/vec0 —
            # before the model/dimension check: its branch drops the area vecs
            # and recreates them under the current dimension
            # (_reset_area_vectors).
            _create_area_schema(conn, settings)
            # Warnings (No12/No32): the shared `warnings` table + indexes —
            # substrate, affects nothing, touches no models; BEFORE the
            # embedding check (arch §5.4).
            _migrate_warnings_schema(conn)
            # The skills global template seed (lsb-0007 §3.8): right after
            # skills_meta is created, idempotent, existing values untouched.
            _ensure_skills_meta(conn)
            # The skill-creator seed (lsb-0007 §3.6/§3.8): right after the
            # template, idempotent, with the skills_meta marker (a deleted one
            # will not resurrect).
            _ensure_creator_skill(conn)
            # Version history (No41, 3.2.0, substrate-versioning §5.1/§5.4):
            # the record_versions table, baselining existing records and moving
            # the skill_versions archive into the shared shape — idempotent,
            # AFTER the area seeds (the skill seed gets a baseline) and BEFORE
            # the embedding check: history is about content and never addresses
            # vector tables.
            _migrate_history_schema(conn, settings)
            # Vectors (Phase 3 + the 2026-08-29 decision; Phase 7: + chunk
            # vectors): created at the first start; on a mismatch between the
            # recorded configuration (model/dimension) and env — a full
            # auto-reindex of both indexes.
            _sync_embedding_meta(conn, settings)
            # The title index (lsb-0001-01, Part B): the full note vectors are
            # invalid (built over the text without the title) — notes_vec is
            # dropped, every note to pending; the meta-key stamp — here.
            _sync_title_vectors(conn, settings, fts_migrated)
            selfheal_chunk_orphans(conn)
    except (sqlite3.Error, OSError) as exc:
        raise StorageError(
            f"не удалось инициализировать БД {settings.db_path}: {exc}"
        ) from exc


# --- namespaces (Phase 10) --------------------------------------------------

DEFAULT_NAMESPACE = "default"
# The system node description is model-facing text, hence EN (No34, 3.2.0).
# The former RU default stays a separate constant: the one-off migration uses
# it to tell a "seed" from an operator edit (see
# _migrate_default_namespace_description).
DEFAULT_NAMESPACE_DESCRIPTION = "general notes not tied to any domain"
DEFAULT_NAMESPACE_DESCRIPTION_LEGACY = "общие заметки, не привязанные к доменам"

# The one-off `default`-description migration stamp (No34, 3.2.0): the key in
# `meta`.
DEFAULT_DESCRIPTION_META_KEY = "default_description_version"
DEFAULT_DESCRIPTION_VERSION = "en-v1"


def _migrate_title_column(conn: sqlite3.Connection) -> None:
    """The Phase 11 zero migration: notes.title TEXT (nullable).

    Fresh databases get the column from _NOTES_DDL; inherited ones — ALTER
    TABLE ADD COLUMN (old notes stay untitled — only they may be; new ones
    always carry a title, the contract is validated by the service).
    """
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(notes)")}
    if "title" not in columns:
        conn.execute("ALTER TABLE notes ADD COLUMN title TEXT")


def _migrate_comment_column(conn: sqlite3.Connection) -> None:
    """The No24 zero migration (3.2.0): notes.comment TEXT (nullable).

    Fresh databases get the column from _NOTES_DDL; inherited ones — ALTER
    TABLE ADD COLUMN (idempotent, applicable from any earlier version;
    NULL = no comment). The 250 limit is NOT a schema CHECK — the validation
    is domain-level (CommentValidationError), like `title`.
    """
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(notes)")}
    if "comment" not in columns:
        conn.execute("ALTER TABLE notes ADD COLUMN comment TEXT")


def _migrate_origin_note_column(conn: sqlite3.Connection) -> None:
    """The 3.6.2 zero migration: notes.origin_note_id INTEGER (nullable).

    Fresh databases get the column from _NOTES_DDL; inherited ones — ALTER
    TABLE ADD COLUMN (idempotent, applicable from any earlier version;
    NULL = an ordinary note, not a part of a split/merge group). The value is
    NOT a reference: no FK, no index — it only groups the parts of one origin
    (the backfill of already split notes — app.storage.migration).
    """
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(notes)")}
    if "origin_note_id" not in columns:
        conn.execute("ALTER TABLE notes ADD COLUMN origin_note_id INTEGER")


def _migrate_comment_fts(conn: sqlite3.Connection) -> None:
    """The comment FTS index (no24-02, 3.2.0): the table + both triggers.

    Idempotent (IF NOT EXISTS) — a repeated start is a no-op. No rebuild:
    at migration time there are no comments (the column was just created by
    _migrate_comment_column), and later comment writes go through
    `AFTER UPDATE OF comment`. The semantic `notes_fts` is untouched.
    """
    conn.execute(_COMMENT_FTS_DDL)
    for trigger in _COMMENT_TRIGGERS:
        conn.execute(trigger)


# The list-output indexes (phase 2, pool 15; a 50k benchmark: LIST p95 204 ms):
# they cover ORDER BY updated_at DESC, id DESC — SQLite walks the index
# (COVERING, already sorted) and stops after LIMIT, with no full scan and no
# sort. idx_notes_ns_deleted_updated supersedes idx_notes_namespace (the
# namespace prefix; deleted_at — the same dedup/counter lookups), the old
# index is dropped at startup.
_INDEX_DELETED_UPDATED_DDL = (
    "CREATE INDEX IF NOT EXISTS idx_notes_deleted_updated "
    "ON notes(deleted_at, updated_at DESC, id DESC)"
)
_INDEX_NS_DELETED_UPDATED_DDL = (
    "CREATE INDEX IF NOT EXISTS idx_notes_ns_deleted_updated "
    "ON notes(namespace, deleted_at, updated_at DESC, id DESC)"
)


def _migrate_namespace_columns(conn: sqlite3.Connection) -> None:
    """The Phase 10 zero migration over live databases: the notes.namespace and
    notes.classified_at columns. Fresh databases get them from _NOTES_DDL;
    inherited ones — ALTER TABLE ADD COLUMN (every existing note → 'default',
    classified_at NULL — grooming will sort them out later, the Phase 10 brief,
    US-12). SQLite allows ADD COLUMN NOT NULL only with a constant DEFAULT —
    'default' is exactly that.
    """
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(notes)")}
    if "namespace" not in columns:
        conn.execute(
            "ALTER TABLE notes ADD COLUMN namespace TEXT NOT NULL DEFAULT 'default'"
        )
    if "classified_at" not in columns:
        conn.execute("ALTER TABLE notes ADD COLUMN classified_at TEXT")


def _migrate_classification_columns(conn: sqlite3.Connection) -> None:
    """The Phase 10 zero migration (Step 4): the grooming placement column
    confidence. Fresh databases get it from _NOTES_DDL; inherited ones —
    ALTER TABLE ADD COLUMN (NULL — grooming will sort them out later). The
    placement details are internal data, NOT in the MCP contracts (§5.7).

    lsb-0005-04: the domain_hint/subdomain_hint columns left the schema
    (grooming moved to the single hint_path — the pair is read/written
    nowhere anymore), so only confidence remains here. The old-pair backfill
    for inherited databases lives in _migrate_hint_path_columns."""
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(notes)")}
    if "confidence" not in columns:
        conn.execute("ALTER TABLE notes ADD COLUMN confidence REAL")


def _migrate_hint_path_columns(conn: sqlite3.Connection) -> None:
    """The lsb-0005-02 zero migration: the single full placement path hint_path.

    Replaces the domain_hint+subdomain_hint pair with one full path. Fresh
    databases get the column from _NOTES_DDL; inherited ones — ALTER TABLE
    ADD COLUMN + a backfill from the old pair (`X/Y` → `hint_path='X/Y'`,
    `X`+NULL → `hint_path='X'`, both NULL → NULL). Idempotent: the backfill
    writes only where `hint_path` is still NULL, so a repeated run never
    overwrites already-placed notes.

    The domain_hint/subdomain_hint columns left the schema (lsb-0005-04:
    grooming moved to hint_path — the pair is no longer read/written anywhere
    in app/, a clean removal), so the backfill runs only for OLD databases
    where the pair still physically exists (the `domain_hint` presence check
    in the table structure); fresh databases do not create those columns and
    skip the backfill.
    """
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(notes)")}
    if "hint_path" not in columns:
        conn.execute("ALTER TABLE notes ADD COLUMN hint_path TEXT")
    if "domain_hint" in columns:  # an old database with the pair — backfill
        conn.execute(
            """
            UPDATE notes SET hint_path = (
              CASE
                WHEN domain_hint IS NOT NULL AND subdomain_hint IS NOT NULL
                  THEN domain_hint || '/' || subdomain_hint
                WHEN domain_hint IS NOT NULL THEN domain_hint
                ELSE NULL
              END
            )
            WHERE hint_path IS NULL
              AND (domain_hint IS NOT NULL OR subdomain_hint IS NOT NULL)
            """
        )


def _migrate_promotions_key(conn: sqlite3.Connection) -> None:
    """The lsb-0005-03 migration: the promotions key (domain, subdomain) → the
    single hint_path (idempotent).

    Fresh databases get the hint_path PRIMARY KEY schema straight from
    _PROMOTIONS_DDL; inherited ones (the (domain, subdomain) key) are
    recreated: existing verdicts migrate as
    `hint_path = domain || '/' || subdomain`. The domain/subdomain columns are
    not carried over — the full path describes the verdict completely.
    Idempotency — by the presence of the old domain column: after the
    recreation it is gone, a repeated run is a no-op.
    """
    columns = {
        row["name"] for row in conn.execute("PRAGMA table_info(promotions)")
    }
    if "domain" not in columns:
        return  # a fresh schema (the hint_path key) or no table yet — no-op
    with transaction(conn):
        conn.execute("ALTER TABLE promotions RENAME TO promotions_legacy")
        conn.execute(_PROMOTIONS_DDL)
        conn.execute(
            "INSERT INTO promotions (hint_path, status, canonical_path, decided_at) "
            "SELECT domain || '/' || subdomain, status, canonical_path, decided_at "
            "FROM promotions_legacy"
        )
        conn.execute("DROP TABLE promotions_legacy")


def _migrate_expiration_columns(conn: sqlite3.Connection) -> None:
    """The lsb-0004-02 zero migration over live databases: the notes.expires_at
    column and the note_expirations table. Fresh databases get the column from
    _NOTES_DDL and the table from _NOTE_EXPIRATIONS_DDL; inherited ones —
    ALTER TABLE ADD COLUMN (existing notes without a TTL → NULL, permanent) +
    CREATE TABLE IF NOT EXISTS (the deletion queue). Idempotent: a repeated
    run is harmless (the column exists — ALTER is skipped, the table —
    IF NOT EXISTS)."""
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(notes)")}
    if "expires_at" not in columns:
        conn.execute("ALTER TABLE notes ADD COLUMN expires_at TEXT")
    conn.execute(_NOTE_EXPIRATIONS_DDL)


def _migrate_node_order_columns(conn: sqlite3.Connection) -> None:
    """The lsb-0011-01 zero migration over live databases: the
    notes.node_order_at marker.

    Fresh databases get the column from _NOTES_DDL; inherited ones — ALTER
    TABLE ADD COLUMN (node_order_at NULL on every note: the first `nodes` run
    naturally sweeps the accumulated `default` — a retro sweep). The marker is
    service-level: it never reaches MCP/REST outputs and is not a warning.
    Idempotent: a repeated run (a service restart) is a no-op.
    """
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(notes)")}
    if "node_order_at" not in columns:
        conn.execute("ALTER TABLE notes ADD COLUMN node_order_at TEXT")


def _migrate_warnings_columns(conn: sqlite3.Connection) -> None:
    """The No12 (3.2.0) zero migration over live databases: the
    notes.warnings_at marker.

    Fresh databases get the column from `_NOTES_DDL`; inherited ones — ALTER
    TABLE ADD COLUMN (warnings_at NULL on every note: the first `warnings` run
    naturally checks them). The marker is service-level (sweep scheduling):
    it creates no versions, never reaches MCP/REST outputs, and the warning is
    NOT stored on the note (only the row in `warnings`, arch §5.3).
    Idempotent: a repeated run (a service restart) is a no-op.
    """
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(notes)")}
    if "warnings_at" not in columns:
        conn.execute("ALTER TABLE notes ADD COLUMN warnings_at TEXT")


def _migrate_warnings_schema(conn: sqlite3.Connection) -> None:
    """The No12 (3.2.0) zero migration: the `warnings` table + both indexes
    (§5.1).

    The shared warnings surface of memory and terms (No12 + No32).
    CREATE TABLE/INDEX IF NOT EXISTS — idempotent, a repeated start is a
    no-op; it touches no models and no other tables (arch §5.4).
    """
    conn.execute(_WARNINGS_DDL)
    conn.execute(_WARNINGS_UNIQUE_INDEX_DDL)
    conn.execute(_WARNINGS_STATUS_INDEX_DDL)


def _migrate_link_columns(conn: sqlite3.Connection) -> None:
    """The lsb-0010-02 zero migration over live databases: the links table, the
    idx_links_b index and the notes.links_at computed marker.

    Fresh databases get the column from _NOTES_DDL and the table from
    _LINKS_DDL; inherited ones — ALTER TABLE ADD COLUMN (links_at NULL on
    every note: the first `links` run naturally backfills, arch §3.4) +
    CREATE TABLE/INDEX IF NOT EXISTS. Idempotent: a repeated run is a no-op.
    """
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(notes)")}
    if "links_at" not in columns:
        conn.execute("ALTER TABLE notes ADD COLUMN links_at TEXT")
    conn.execute(_LINKS_DDL)
    conn.execute(_LINKS_INDEX_DDL)


def _migrate_summary_length(
    conn: sqlite3.Connection, settings: Settings
) -> None:
    """The 3.1.0 gate migration: summaries above the limit go to regeneration.

    The summary length limit dropped to MAX_SUMMARY_CHARS (150): a live DB's
    already-saved longer summaries must be redone. The stored text is NOT
    touched here (a crude `substr` is forbidden: a summary cut mid-sentence is
    worse than none): the note is marked `summary_status='pending'`, and the
    background `summary` job places the new one
    (`worker.process_summary_pending` — `cap_summary` enforces the limit).
    While a summary regenerates, outputs serve the fallback truncation of the
    note text (§5.5) — already within the limit.

    Idempotency: after regeneration the summary is <= the limit — a repeated
    start finds nothing (a no-op), no meta stamp needed.

    What is not touched:
    - notes with `summary_status != 'ok'` — they are already in the
      summarization queue (a repeated mark changes nothing, the status is not
      rewritten);
    - trash (`deleted_at IS NOT NULL`) — the `summary` job does not serve it,
      the note would stay pending forever; trash summaries do not matter.

    The schema carries no summary-origin marker (model or explicit), so
    explicitly written long summaries also go to regeneration — the limit is
    one for the whole `summary` field.
    """
    cursor = conn.execute(
        "UPDATE notes SET summary_status = 'pending' "
        "WHERE summary_status = 'ok' AND deleted_at IS NULL "
        "AND length(summary) > ?",
        (settings.max_summary_chars,),
    )
    if cursor.rowcount:
        logging.getLogger("app").warning(
            "summaries above the length limit queued for regeneration",
            extra={
                "event": "summary_regen_queued",
                "notes": cursor.rowcount,
                "reason": "summary_length",
                "max_summary_chars": settings.max_summary_chars,
            },
        )


def _ensure_skills_meta(conn: sqlite3.Connection) -> None:
    """Seed the skills-area global instruction_template (idempotent).

    Modeled after _ensure_default_namespace: the record is created only when
    the key is absent. An existing value is NEVER overwritten — the operator
    edits the template via REST (lsb-0007-05), a repeated init_db (a restart
    of the service) never wipes that edit.
    """
    conn.execute(
        "INSERT INTO skills_meta (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO NOTHING",
        (INSTRUCTION_TEMPLATE_KEY, INSTRUCTION_TEMPLATE_SEED),
    )


def _ensure_creator_skill(conn: sqlite3.Connection) -> None:
    """Seed the skills-area skill-creator (lsb-0007 §3.6/§3.8), idempotent.

    The seed is ONE attempt per database lifetime (the `creator_skill_seed`
    marker in `skills_meta`): while the marker is absent and no live record
    named `CREATOR_SKILL_NAME` exists — the skill is inserted; the marker is
    written whenever the seed passes the gate. This way a repeated init_db (a
    service restart) spawns no copies, and a seed deleted by the operator or a
    model does NOT resurrect (the soft-deleted row stays in trash, the marker —
    in skills_meta).
    """
    seeded = conn.execute(
        "SELECT 1 FROM skills_meta WHERE key = ?", (CREATOR_SKILL_SEED_KEY,)
    ).fetchone()
    if seeded is not None:
        return
    exists = conn.execute(
        "SELECT 1 FROM skills WHERE name = ? AND deleted_at IS NULL",
        (CREATOR_SKILL_NAME,),
    ).fetchone()
    if exists is None:
        conn.execute(
            "INSERT INTO skills (name, description, steps, text, extra, "
            "version, vector_status) VALUES (?, ?, ?, ?, NULL, 1, 'pending')",
            (
                CREATOR_SKILL_NAME,
                CREATOR_SKILL_DESCRIPTION,
                CREATOR_SKILL_STEPS,
                CREATOR_SKILL_TEXT,
            ),
        )
    conn.execute(
        "INSERT INTO skills_meta (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO NOTHING",
        (CREATOR_SKILL_SEED_KEY, CREATOR_SKILL_SEED_VALUE),
    )


# --- version history (No41, 3.2.0) --------------------------------------

def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    """Whether the table object exists in the DB (step idempotency by presence)."""
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
        ).fetchone()
        is not None
    )


def _history_projection(entity: str, row: sqlite3.Row, fields: tuple[str, ...]) -> dict:
    """The record projection for history (§5.3): fields as is, `extra` as an object.

    A skill's `extra` in the live table is the JSON text of the class fields;
    in a version it lives as the parsed object (one shape for the baseline and
    the deltas), empty — `None`.
    """
    projection: dict = {}
    for field in fields:
        value = row[field]
        if entity == "skill" and field == "extra":
            value = None if not value else json.loads(value)
        projection[field] = value
    return projection


def _insert_baseline(
    conn: sqlite3.Connection, entity: str, entity_id: int, projection: dict
) -> None:
    """`INSERT … WHERE NOT EXISTS`: a baseline only where history does not exist yet.

    Self-healing: a row that appeared after the baseline migration picks up its
    baseline at the next start; a repeated pass is a no-op (§5.4).
    """
    conn.execute(
        "INSERT INTO record_versions (entity, entity_id, version, event, source, "
        "changed, snapshot) SELECT ?, ?, 1, 'baseline', 'migration', '[]', ? "
        "WHERE NOT EXISTS (SELECT 1 FROM record_versions WHERE entity = ? "
        "AND entity_id = ?)",
        (entity, entity_id, json.dumps(projection, ensure_ascii=False), entity, entity_id),
    )


def _backfill_history_baseline(conn: sqlite3.Connection) -> None:
    """Baseline every existing record: `version=1` = the current content.

    The projection is assembled in Python (`json.dumps`) — no SQLite json1
    dependency; a separate pass per entity, `WHERE NOT EXISTS` keeps it
    idempotent (FR-8.1/FR-8.2, §5.4). Trash records get a baseline too:
    history is operator-facing, a soft delete does not undo it.
    """
    from app.services.history import (
        CONTENT_FIELDS,
        ENTITY_ASSISTANT_LOG,
        ENTITY_NOTE,
        ENTITY_SKILL,
        ENTITY_SKILL_TEMPLATE,
        ENTITY_TERM,
        ENTITY_USER_FACT,
    )

    entity_tables = {
        ENTITY_NOTE: "notes",
        ENTITY_SKILL: "skills",
        ENTITY_TERM: "terms",
        ENTITY_USER_FACT: "user_facts",
        ENTITY_ASSISTANT_LOG: ASSISTANT_LOGS_TABLE,
    }
    for entity, table in entity_tables.items():
        fields = CONTENT_FIELDS[entity]
        columns = ", ".join(("id", *fields))
        for row in conn.execute(f"SELECT {columns} FROM {table}").fetchall():
            _insert_baseline(
                conn,
                entity,
                int(row["id"]),
                _history_projection(entity, row, fields),
            )
    # The skills global template — a separate history entity (id 0, the key
    # skills_meta.instruction_template, §5.3).
    template = conn.execute(
        "SELECT value FROM skills_meta WHERE key = ?", (INSTRUCTION_TEMPLATE_KEY,)
    ).fetchone()
    if template is not None:
        _insert_baseline(
            conn, ENTITY_SKILL_TEMPLATE, 0, {"instruction_template": template["value"]}
        )


def _merge_skill_versions(conn: sqlite3.Connection, settings: Settings) -> None:
    """Move the `skill_versions` archive into the shared history shape (§5.4, §13.7).

    The chain `1..N`: v1 — the earliest archived copy whole (`snapshot`,
    `event='baseline'`, `source='migration'`), the middle ones — deltas
    computed from the archived copies during the move, vN — the current `skills`
    row (a delta from the previous one). The numbers come from
    `skill_versions.version` and `skills.version`; missing links are not
    invented — the delta is computed
    between the neighboring versions THAT EXIST. The move runs in the
    caller's single transaction; the `skill_versions` table stays a legacy
    imprint (§13.7) and is no longer read by code. A skill that already has
    history is skipped: the step is idempotent; no table — a no-op.
    """
    from app.services.history import CONTENT_FIELDS, ENTITY_SKILL, encode_delta

    if not _table_exists(conn, "skill_versions"):
        return
    fields = CONTENT_FIELDS[ENTITY_SKILL]
    skill_ids = [
        int(row[0])
        for row in conn.execute("SELECT DISTINCT skill_id FROM skill_versions")
    ]
    for skill_id in skill_ids:
        has_history = conn.execute(
            "SELECT 1 FROM record_versions WHERE entity = ? AND entity_id = ? LIMIT 1",
            (ENTITY_SKILL, skill_id),
        ).fetchone()
        if has_history is not None:
            continue
        chain: dict[int, dict] = {}
        for row in conn.execute(
            "SELECT * FROM skill_versions WHERE skill_id = ? ORDER BY version",
            (skill_id,),
        ).fetchall():
            chain[int(row["version"])] = _history_projection(ENTITY_SKILL, row, fields)
        live = conn.execute("SELECT * FROM skills WHERE id = ?", (skill_id,)).fetchone()
        if live is not None:
            chain[int(live["version"])] = _history_projection(ENTITY_SKILL, live, fields)
        if not chain:
            continue
        previous: dict | None = None
        for version in sorted(chain):
            current = chain[version]
            if previous is None:
                conn.execute(
                    "INSERT INTO record_versions (entity, entity_id, version, event, "
                    "source, changed, snapshot) VALUES (?, ?, ?, 'baseline', "
                    "'migration', '[]', ?)",
                    (
                        ENTITY_SKILL,
                        skill_id,
                        version,
                        json.dumps(current, ensure_ascii=False),
                    ),
                )
            else:
                delta, changed = encode_delta(
                    previous, current, inline_max=settings.history_delta_inline_max
                )
                conn.execute(
                    "INSERT INTO record_versions (entity, entity_id, version, event, "
                    "source, changed, delta) VALUES (?, ?, ?, 'content', "
                    "'migration', ?, ?)",
                    (
                        ENTITY_SKILL,
                        skill_id,
                        version,
                        json.dumps(changed, ensure_ascii=False),
                        json.dumps(delta, ensure_ascii=False),
                    ),
                )
            previous = current


def _migrate_history_schema(conn: sqlite3.Connection, settings: Settings) -> None:
    """The history schema + baselining existing records + the skill_versions move.

    The no41-01 design steps: (a) `CREATE TABLE IF NOT EXISTS record_versions`
    (§5.1 — exactly as in the architecture, no indexes beyond the PK);
    (b) moving the skills archive into the shared shape; (c) baselining every
    existing record. The "move → backfill" order is mandatory: for a skill
    with an archive the baseline is the earliest archived copy, not the
    current row (otherwise v1 would be overwritten by the current content and
    the archive chain would become unreachable). All in one transaction,
    idempotent and applicable from any earlier version.
    """
    with transaction(conn):
        conn.execute(_RECORD_VERSIONS_DDL)
        _merge_skill_versions(conn, settings)
        _backfill_history_baseline(conn)


def _ensure_default_namespace(conn: sqlite3.Connection) -> None:
    """The 'default' node always exists (idempotent): the description is the
    default one; the operator may edit it via REST (Step 6)."""
    conn.execute(
        "INSERT INTO namespaces (path, description, status) VALUES (?, ?, 'confirmed') "
        "ON CONFLICT(path) DO NOTHING",
        (DEFAULT_NAMESPACE, DEFAULT_NAMESPACE_DESCRIPTION),
    )


def _sync_namespace_partition(conn: sqlite3.Connection, settings: Settings) -> None:
    """Migrate the vec indexes under the namespace partition (Phase 10).

    Databases created before Phase 10 have notes_vec/notes_chunks_vec without
    the `+ns` column: BOTH tables are dropped and recreated with the partition
    key, every note (trash included) goes to vector_status='pending' — the
    background worker re-encodes (NFR-3: no data is lost, search degrades to
    FTS). Fresh databases are skipped: their tables are created in
    _sync_embedding_meta already partitioned. Called BEFORE
    _sync_embedding_meta (see init_db).
    """
    if vectors.has_partition(conn) and chunks.has_partition(conn):
        return
    has_tables = (
        vectors.existing_vec_dim(conn) is not None
        or chunks.existing_vec_dim(conn) is not None
    )
    if not has_tables:
        return  # a fresh DB — the tables are created below already partitioned
    notes_count = int(conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0])
    logging.getLogger("app").warning(
        "namespace partition introduced: rebuilding vector indexes",
        extra={
            "event": "reindex_started",
            "notes": notes_count,
            "reason": "namespace_partition",
        },
    )
    conn.execute("DROP TABLE IF EXISTS notes_vec")
    conn.execute("DROP TABLE IF EXISTS notes_chunks_vec")
    vectors.create_vec_table(conn, settings.embedding_dim)
    chunks.create_vec_table(conn, settings.embedding_dim)
    conn.execute("UPDATE notes SET vector_status = 'pending'")
    logging.getLogger("app").info(
        "vector indexes rebuilt with namespace partition",
        extra={
            "event": "reindex_done",
            "notes_pending": notes_count,
            "reason": "namespace_partition",
        },
    )


def _get_meta(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return None if row is None else str(row[0])


def _set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )


def _migrate_default_namespace_description(conn: sqlite3.Connection) -> None:
    """Bring the `default` node description to EN (No34, 3.2.0): one-off.

    A fresh database gets the EN text from `_ensure_default_namespace` — for
    it the migration is a no-op (the text does not equal the former RU
    default), but the stamp is set anyway. An inherited database is edited
    ONLY when the description exactly equals the former Russian default
    (DEFAULT_NAMESPACE_DESCRIPTION_LEGACY): any other edit — including a
    Russian one — by the operator is preserved; "seed" and "edit" are
    distinguished by text comparison, no new column is introduced.

    The one-off nature comes from the DEFAULT_DESCRIPTION_META_KEY stamp in
    `meta`: it is set INDEPENDENTLY of the outcome (whether the row was
    updated or not), so a repeated `init_db` searches nothing and changes
    nothing. Called in `init_db` right after `_ensure_default_namespace`.
    """
    conn.execute(_META_DDL)  # meta may not exist in ancient databases yet —
    # normally the table is created later, in _sync_embedding_meta (see init_db)
    if _get_meta(conn, DEFAULT_DESCRIPTION_META_KEY) is not None:
        return  # brought over earlier — idempotent
    with transaction(conn):
        cursor = conn.execute(
            "UPDATE namespaces SET description = ?, "
            "updated_at = strftime('%Y-%m-%dT%H:%M:%SZ','now') "
            "WHERE path = ? AND description = ?",
            (
                DEFAULT_NAMESPACE_DESCRIPTION,
                DEFAULT_NAMESPACE,
                DEFAULT_NAMESPACE_DESCRIPTION_LEGACY,
            ),
        )
        if cursor.rowcount == 1:
            logging.getLogger("app").info(
                "default namespace description migrated to english",
                extra={
                    "event": "default_description_migrated",
                    "path": DEFAULT_NAMESPACE,
                    "old": DEFAULT_NAMESPACE_DESCRIPTION_LEGACY,
                    "new": DEFAULT_NAMESPACE_DESCRIPTION,
                },
            )
        _set_meta(conn, DEFAULT_DESCRIPTION_META_KEY, DEFAULT_DESCRIPTION_VERSION)


# --- the title index (lsb-0001-01) ---------------------------------------------

# The migration stamp in meta: "2" — notes_fts indexes title and text (FR-1.2/
# FR-1.3); an absent key = a database that has not passed the title migration
# (FTS over text only). Shared by Part A (FTS) and Part B (vectors).
_TITLE_INDEX_META_KEY = "title_index_version"
_TITLE_INDEX_VERSION = "2"


def _sync_fts_title_index(conn: sqlite3.Connection) -> bool:
    """Part A of the title-index migration (lsb-0001-01): notes_fts under (title, text).

    Live databases predating lsb-0001-01 have a notes_fts with a single text
    column: the table is dropped and recreated per _FTS_DDL, the index is
    re-poured from notes (rowid, title, text — the title may be NULL, FTS5
    treats it as an empty string), the triggers are recreated for the new
    columns. The gate decision R4 — a startup migration, not a job;
    idempotency — the meta key _TITLE_INDEX_META_KEY shared with Part B (the
    stamp is set by _sync_title_vectors at the end of init_db).

    The launch condition: the key is absent AND the notes_fts DDL in
    sqlite_master has no title (the _FTS_TITLE_RE regexp). The key absent but
    the DDL already with title (a fresh DB) — a no-op. Called after the column
    migrations: by the call time notes.title already exists (the pour reads
    it) and BEFORE _check_fts_integrity — the pour leaves the index
    consistent.

    Returns True when this run found a legacy database and rebuilt it — Part B
    (_sync_title_vectors) invalidates the full vectors only following an
    actual FTS rebuild.
    """
    conn.execute(_META_DDL)  # meta may not exist in ancient databases yet —
    # normally the table is created later, in _sync_embedding_meta (see init_db)
    if _get_meta(conn, _TITLE_INDEX_META_KEY) is not None:
        return False  # migrated earlier — _sync_title_vectors sets the stamp
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'notes_fts'"
    ).fetchone()
    if row is None or row[0] is None or _FTS_TITLE_RE.search(row[0]):
        return False  # a fresh DB: notes_fts was created above with the title column
    notes_count = int(conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0])
    logging.getLogger("app").warning(
        "note title added to FTS index: rebuilding notes_fts",
        extra={
            "event": "reindex_started",
            "notes": notes_count,
            "reason": "title_index",
        },
    )
    conn.execute("DROP TRIGGER IF EXISTS notes_fts_ai")
    conn.execute("DROP TRIGGER IF EXISTS notes_fts_au")
    conn.execute("DROP TABLE IF EXISTS notes_fts")
    conn.execute(_FTS_DDL)
    conn.execute(
        "INSERT INTO notes_fts(rowid, title, text) SELECT id, title, text FROM notes"
    )
    for trigger in _TRIGGERS:
        conn.execute(trigger)
    logging.getLogger("app").info(
        "notes_fts rebuilt with title column",
        extra={
            "event": "reindex_done",
            "notes": notes_count,
            "reason": "title_index",
        },
    )
    return True


def _sync_title_vectors(
    conn: sqlite3.Connection, settings: Settings, fts_migrated: bool
) -> None:
    """Part B of the title-index migration (lsb-0001-01): the full vectors.

    The full vectors (notes_vec) were computed over the text WITHOUT the
    title — after the title entered FTS (and the encoding, step 2) they are
    invalid: notes_vec is dropped, every note (trash included) goes to
    vector_status='pending'; no vectors are encoded in this step — the
    background worker catches up. notes_chunks_vec is NOT touched (the gate
    decision R4): chunk vectors are built over chunk texts, the title does not
    affect them.

    The condition: the key is absent (shared with Part A) AND the FTS was
    rebuilt in this same run (`fts_migrated` from _sync_fts_title_index): the
    two migration parts move together — a meta stamp does not protect against
    losing the whole meta manually, and dropping valid vectors because of a
    lost stamp alone is unnecessary (without meta history the vectors count as
    fresh — as in the _sync_embedding_meta zero migration). The key stamp —
    at the end, always (including a fresh DB with zero notes — nothing to
    re-encode).

    Called after _sync_embedding_meta: its branches (a model/chunk change)
    have already recreated the tables of the current configuration — the drop
    here no longer breaks them; BEFORE selfheal_chunk_orphans.
    """
    if _get_meta(conn, _TITLE_INDEX_META_KEY) is not None:
        return
    notes_count = int(conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0])
    if fts_migrated and notes_count:
        logging.getLogger("app").warning(
            "title index migration: full-text vectors invalidated",
            extra={
                "event": "reindex_started",
                "notes": notes_count,
                "reason": "title_index",
            },
        )
        conn.execute("DROP TABLE IF EXISTS notes_vec")
        vectors.create_vec_table(conn, settings.embedding_dim)
        # The full vectors are invalid — every note (trash included) to the queue.
        conn.execute("UPDATE notes SET vector_status = 'pending'")
        logging.getLogger("app").info(
            "notes_vec rebuilt; background worker will re-encode all notes",
            extra={
                "event": "reindex_done",
                "notes_pending": notes_count,
                "reason": "title_index",
            },
        )
    _set_meta(conn, _TITLE_INDEX_META_KEY, _TITLE_INDEX_VERSION)


# The chunk-parameter descriptions (Phase 7, brief §4): pinned in meta —
# changing any of them over a live DB means re-chunking (step 6).
_CHUNK_META_KEYS = ("chunk_size", "chunk_overlap", "chunk_min_target")


def _set_chunk_meta_defaults(conn: sqlite3.Connection, settings: Settings) -> None:
    """Pin the chunk parameters in meta (the zero migration: an absent key —
    from env, no re-chunking; the env comparison and re-chunking on a change —
    step 6 of Phase 7)."""
    for key in _CHUNK_META_KEYS:
        if _get_meta(conn, key) is None:
            _set_meta(conn, key, str(getattr(settings, key)))


def _create_chunk_vec_if_missing(conn: sqlite3.Connection, dim: int) -> None:
    """notes_chunks_vec arrived in Phase 7 over live databases — create it when
    missing (without reindexing notes — their vectors are current anyway)."""
    if chunks.existing_vec_dim(conn) is None:
        chunks.create_vec_table(conn, dim)


def selfheal_chunk_orphans(conn: sqlite3.Connection) -> None:
    """Clean chunk orphans at startup (Phase 7, like the FTS integrity check).

    An operator's sqlite3 CLI without PRAGMA foreign_keys=ON does not cascade
    a hard note deletion into chunks, and vec0 has no FK support at all:
    chunks without a note and vectors without a chunk are removed by
    self-healing; the log event fires only when there was something to clean."""
    dead_chunks, dead_vectors = chunks.clean_orphans(conn)
    if dead_chunks or dead_vectors:
        logging.getLogger("app").warning(
            "cleaned chunk orphans left by direct DB edits",
            extra={
                "event": "chunks_orphans_cleaned",
                "chunks": dead_chunks,
                "chunk_vectors": dead_vectors,
            },
        )


def _sync_embedding_meta(conn: sqlite3.Connection, settings: Settings) -> None:
    """Compare the vectorization/chunking configuration with the meta records;
    on a divergence — an auto-reindex at startup.

    Two different causes, one mechanics:
    - a model/dimension/provider change (env vs the meta embedding_model/
      embedding_dim/embedding_provider; the provider — Phase 11): BOTH vector
      indexes are recreated under the current dimension, every note (trash
      included — its vectors are invalid too) goes to pending, and the worker
      re-encodes the full vectors;
    - a chunk-parameter change (env vs the meta chunk_*, Phase 7, brief §6):
      the full-text vectors stay valid (the text did not change) — only
      notes_chunks_vec is dropped; the chunk texts are RECOMPUTED for every
      note (trash included) under the current splitter parameters, and the
      worker finishes the vectors.
    Both changes at once — the combined path: one reindex_started/done, the
    recreation of both indexes, every note pending + re-chunking.

    Reuse during re-chunking: for a note with exactly one chunk <= CHUNK_SIZE
    the chunk vector = the full-text vector from notes_vec without encoding —
    a vector does not depend on the chunk parameters, and with a changed model
    notes_vec is already empty (dropped before the re-chunking) — reuse itself
    never fires where the full vector is invalid.

    Legacy: a database without meta (created before this change) gets the
    record from the current env without a reindex — the zero migration; its
    vectors were built by the same model anyway (otherwise the operator would
    have run reindex.py). The chunk parameters are written to meta the same
    way — only absent keys (absence = "never changed"; the env comparison runs
    over existing keys only). The embedding provider (Phase 11) — the same
    pattern: a database without the embedding_provider key (pre-Phase 11) gets
    the env record without a reindex; a provider change under the same model
    name is a reindex too (another provider's model vectors are incompatible).
    """
    conn.execute(_META_DDL)
    stored = _get_meta(conn, "embedding_model")
    stored_dim_raw = _get_meta(conn, "embedding_dim")
    existing_dim = vectors.existing_vec_dim(conn)
    if stored is None or stored_dim_raw is None:
        # A fresh DB (no table) or an inherited one (untouched, see the docstring).
        if existing_dim is None:
            vectors.create_vec_table(conn, settings.embedding_dim)
        _create_chunk_vec_if_missing(conn, settings.embedding_dim)
        _create_area_vec_if_missing(conn, settings.embedding_dim)
        _set_chunk_meta_defaults(conn, settings)
        _set_meta(conn, "embedding_model", settings.embedding_model)
        _set_meta(conn, "embedding_dim", str(settings.embedding_dim))
        _set_meta(conn, "embedding_provider", settings.embedding_provider)
        return
    stored_dim = int(stored_dim_raw)
    # The embedding provider (Phase 11): a database without the key (pre-Phase
    # 11) — the zero migration by the chunk-key pattern: the env record WITHOUT
    # a reindex.
    stored_provider = _get_meta(conn, "embedding_provider")
    if stored_provider is None:
        _set_meta(conn, "embedding_provider", settings.embedding_provider)
        stored_provider = settings.embedding_provider
    model_changed = (
        stored != settings.embedding_model
        or stored_dim != settings.embedding_dim
        or stored_provider != settings.embedding_provider
    )
    # Chunk parameters: compare ONLY the previously pinned keys (an absent key
    # is the zero migration — "never changed"; written below).
    chunk_changes = {
        key: (str(stored_value), str(getattr(settings, key)))
        for key in _CHUNK_META_KEYS
        if (stored_value := _get_meta(conn, key)) is not None
        and stored_value != str(getattr(settings, key))
    }
    if not model_changed and not chunk_changes:
        # A match: but notes_chunks_vec may not exist yet on a live DB
        # (Phase 7 over Phase 5) — create when missing. Area vecs — the same
        # logic (pre-3.0.0 inheritors: they have meta but no areas).
        _create_chunk_vec_if_missing(conn, settings.embedding_dim)
        _create_area_vec_if_missing(conn, settings.embedding_dim)
        _set_chunk_meta_defaults(conn, settings)
        return
    notes_count = int(
        conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
    )
    started_extra: dict[str, object] = {
        "event": "reindex_started",
        "notes": notes_count,
    }
    message = "chunk parameters changed: re-chunking all notes"
    if model_changed:
        message = "embedding provider/model/dim changed: rebuilding vector index"
        started_extra.update(
            from_model=stored,
            to_model=settings.embedding_model,
            from_dim=stored_dim,
            to_dim=settings.embedding_dim,
            from_provider=stored_provider,
            to_provider=settings.embedding_provider,
        )
    for key, (old_value, new_value) in chunk_changes.items():
        started_extra[f"from_{key}"] = old_value
        started_extra[f"to_{key}"] = new_value
    logging.getLogger("app").warning(message, extra=started_extra)
    if model_changed:
        conn.execute("DROP TABLE IF EXISTS notes_vec")
        vectors.create_vec_table(conn, settings.embedding_dim)
        # The full-text vectors are invalid — every note (trash included) to
        # the queue.
        conn.execute("UPDATE notes SET vector_status = 'pending'")
        # Level-1 links (lsb-0010-02, arch §3.4): the cosine is computed over
        # these vectors and "significant words"/mentions over the text, but
        # another model's links are invalid wholesale — the table is cleared,
        # the markers reset (the computation catches up after the worker
        # finishes re-encoding).
        conn.execute("UPDATE notes SET links_at = NULL")
        conn.execute("DELETE FROM links")
        # The 3.0.0 areas: another model's vectors are incompatible — the area
        # vecs are dropped and recreated, every area record → pending.
        _reset_area_vectors(conn, settings)
    # Chunk vectors are invalid under either cause (another model — another
    # vectorization; other parameters — other chunks whose vectors no longer
    # match their own new texts). The drop: every chunk → pending (an
    # anti-join), the worker re-encodes.
    conn.execute("DROP TABLE IF EXISTS notes_chunks_vec")
    chunks.create_vec_table(conn, settings.embedding_dim)
    # Re-chunking + the meta writes — one atomic block: a rollback returns
    # both the chunks and the previous configuration's meta if anything goes
    # wrong.
    with transaction(conn):
        rechunked, reused = _rechunk_all_notes(conn, settings)
        if model_changed:
            _set_meta(conn, "embedding_model", settings.embedding_model)
            _set_meta(conn, "embedding_dim", str(settings.embedding_dim))
            _set_meta(conn, "embedding_provider", settings.embedding_provider)
        for key in _CHUNK_META_KEYS:
            _set_meta(conn, key, str(getattr(settings, key)))
    pending_chunks = int(chunks.count_pending(conn))
    done_extra: dict[str, object] = {
        "event": "reindex_done",
        "notes_rechunked": rechunked,
        "chunk_vectors_reused": reused,
        "pending_chunks": pending_chunks,
    }
    if model_changed:
        done_extra["pending_vector"] = notes_count
        done_extra["pending_areas"] = _count_area_records(conn)
    message_done = (
        "vector index rebuilt; background worker will re-encode all notes"
        if model_changed
        else "notes re-chunked; chunk vectors pending for background worker"
    )
    logging.getLogger("app").info(message_done, extra=done_extra)


def _rechunk_all_notes(
    conn: sqlite3.Connection, settings: Settings
) -> tuple[int, int]:
    """Re-chunk every note (trash included) under the current parameters.

    A migration operation (called only from _sync_embedding_meta when the
    model/dimension or chunk parameters changed — brief §6): the texts are
    recomputed by the pure splitter, the chunks and their vectors are replaced
    (replace_note_chunks clears the vec rows explicitly). The splitter import
    is local: storage does not pull the domain at import time — re-chunking
    here is like the old reindex in init_db, one-shot at service startup.

    Reuse: a note with exactly one chunk <= CHUNK_SIZE gets the chunk vector =
    the full-text vector from notes_vec — without encoding; with a changed
    model notes_vec is empty by then — reuse itself does not fire.

    Returns (notes re-chunked, reuse vectors written).
    """
    from app.services.splitter import split_text  # local, see the docstring

    rows = conn.execute("SELECT id, text FROM notes ORDER BY id").fetchall()
    rechunked = reused = 0
    for row in rows:
        spans = split_text(
            row["text"],
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            chunk_min_target=settings.chunk_min_target,
        )
        chunk_ids = chunks.replace_note_chunks(
            conn, row["id"], [(chunk.text, chunk.tokens) for chunk in spans]
        )
        rechunked += 1
        if len(chunk_ids) == 1 and spans[0].tokens <= settings.chunk_size:
            note_vector = vectors.get_vector(conn, row["id"])
            if note_vector is not None:
                chunks.upsert_vector(conn, chunk_ids[0], note_vector)
                reused += 1
    return rechunked, reused


def _check_fts_integrity(conn: sqlite3.Connection) -> None:
    """Verify the FTS index against notes; a desync (the operator edited
    `notes.text` directly, bypassing the triggers) is healed by a rebuild —
    the index is fully derived from the notes' (title, text), no data is
    lost."""
    try:
        conn.execute("INSERT INTO notes_fts(notes_fts) VALUES('integrity-check')")
        return
    except sqlite3.DatabaseError:
        pass  # a desync/damage — self-healing below
    # rebuild: zero the index (the special 'delete' command with all current
    # values) and pour it again; insensitive to the previous state.
    conn.execute(
        "INSERT INTO notes_fts(notes_fts, rowid, title, text) "
        "SELECT 'delete', id, title, text FROM notes"
    )
    conn.execute(
        "INSERT INTO notes_fts(rowid, title, text) SELECT id, title, text FROM notes"
    )


def delete_note_physical(conn: sqlite3.Connection, note_id: int) -> None:
    """Hard-delete a note from every index (lsb-0004-02, the cleanup job).

    A full physical deletion (NOT a soft delete): an expired note must
    disappear from all indexes — notes + notes_chunks + notes_chunks_vec +
    notes_vec + notes_fts + notes_comment_fts + note_expirations + links.
    A composition of the layer's existing primitives (chunks.drop_note_chunks,
    vectors.drop, expirations.delete) + an explicit DELETE from notes_fts
    (external content: no DELETE trigger, the index is not cleaned by a
    cascade) and from notes.

    Links (3.2.0, no46-03): the note's `links` rows are torn down here, in the
    SAME caller transaction (`DELETE FROM links WHERE note_a = ? OR
    note_b = ?` — both sides of the pair, idempotent). The background
    `purge_orphans` sweep (the idle branch of the `links` job) stays a safety
    net, not the only cleaner. A soft delete does NOT touch links: the note is
    restorable — its `links` rows live on (arch §3.2).

    History (lsbdef-0014): a hard deletion takes the note's versions out of
    `record_versions` too — `notes.id` without AUTOINCREMENT is reused, and a
    leftover baseline of the reused id used to kill `record_initial` of the
    new note on the UNIQUE `(entity, entity_id, version)` → `memory_save`
    failed. A soft delete KEEPS the history (a note in trash is restorable,
    the versions remain) — do not confuse the two.

    Idempotent and safe: a missing note matches nothing — every DELETE is a
    no-op; a repeated call does not fail. Called inside the caller's open
    transaction (transaction()).

    origin_note_id (3.6.2) is not a reference, nothing to clean: the parts of
    a split/merge group keep pointing at the id of a note that may already be
    gone (the id is reused) — the value is a group label, not an edge.
    """
    # The chunks and their vectors (notes_chunks_vec + notes_chunks) — before
    # notes: notes_chunks_vec has no FK, the notes cascade would not touch it.
    chunks.drop_note_chunks(conn, note_id)
    vectors.drop(conn, note_id)  # notes_vec
    # LSBN-D-8 (3.4.0): the comment FTS — external content without a DELETE
    # trigger; the special 'delete' command with the OLD comment value (the
    # _COMMENT_TRIGGERS pattern) while the notes row is still there. A missing
    # note is skipped (idempotent); the DELETEs below simply match nothing.
    comment_row = conn.execute(
        "SELECT comment FROM notes WHERE id = ?", (note_id,)
    ).fetchone()
    if comment_row is not None:
        conn.execute(
            "INSERT INTO notes_comment_fts(notes_comment_fts, rowid, comment) "
            "VALUES ('delete', ?, ?)",
            (note_id, comment_row["comment"]),
        )
    conn.execute("DELETE FROM notes_fts WHERE rowid = ?", (note_id,))
    conn.execute("DELETE FROM notes WHERE id = ?", (note_id,))
    expirations.delete(conn, note_id)  # note_expirations
    # 3.2.0 (no46-03): the note's links go with it — both sides of the pair,
    # in the same transaction (idempotent: a repeated call matches nothing).
    conn.execute(
        "DELETE FROM links WHERE note_a = ? OR note_b = ?", (note_id, note_id)
    )
    # The note's history (lsbdef-0014): the versions go with the note, in the
    # same transaction — otherwise a freed `notes.id`, after reuse, inherited
    # someone else's baseline and `memory_save` failed on the UNIQUE. Only a
    # HARD deletion clears the history; a soft delete keeps it (the note is
    # restorable).
    conn.execute(
        "DELETE FROM record_versions WHERE entity = 'note' AND entity_id = ?",
        (note_id,),
    )