"""The version history layer (No41, 3.2.0): the shared `record_versions` table.

    The live record lives in the previous tables and is read as before; the
    history is stored alongside: `version=1` is the baseline (the content whole),
    then a chain of deltas (FR-1.1/FR-2.1/FR-2.2). A layer over the storage: it
    imports no domain services, takes `Settings` from `app.config`; the connection
    belongs to the caller.

    Key decisions (ARCH substrate-versioning §5, §6.1, §13):
    - The delta format - a per-field JSON object "field -> new value"; a long text
      field (>= `HISTORY_DELTA_INLINE_MAX`) is encoded as a line replacement
      (`{"__lines__": [[i, j, [lines]], ...]}`) - the history grows by the number
      of edits, not by the record's volume (§5.2, §13.1).
    - A write - ONLY on the caller's open connection, inside its `transaction()`:
      there is no "the edit went through, the history did not" state
      (FR-2.4, §13.5). The module opens neither connections nor transactions on
      the write path.
    - A version's source and session (`source`/`session`) come from a contextvar
      (`bind_origin`/`current_origin`): the service signatures do not change
      (§13.4).
    - The history is never rewritten: a rollback is a new version (§13.6); the
      retention policy (`HISTORY_MAX_VERSIONS`/`HISTORY_RETENTION_DAYS`) prunes
      the oldest versions and RE-BASES the earliest kept one into a full snapshot;
      the numbers are never reused (§13.8).
"""

from __future__ import annotations

import difflib
import json
import sqlite3
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from app.config import Settings, get_settings
from app.services.listing import page_fields
from app.storage.db import INSTRUCTION_TEMPLATE_KEY, session

#--- the entities and the versioned projection (§5.3) -----------------------------

ENTITY_NOTE = "note"
ENTITY_SKILL = "skill"
ENTITY_SKILL_TEMPLATE = "skill_template"
ENTITY_TERM = "term"
ENTITY_USER_FACT = "user_fact"
# The "Assistant logs" area (LSBN-F-10, arch §5.4): the logs area's
# versioned entity; the service fields (`vector_status`, times) create no versions.
ENTITY_ASSISTANT_LOG = "assistant_log"

# The §5.3 projection registry: which entity fields create a version. The service
# (`vector_status`, `summary_status`, `links_at`, `node_order_at`,
# `classified_at`, `hint_path`, `confidence`) are not in the projection and
# create no versions (FR-4.2). A note's derived (auto) `summary` is outside the
# projection too: only an explicitly passed summary enters a version (§13.3).
CONTENT_FIELDS: dict[str, tuple[str, ...]] = {
    ENTITY_NOTE: ("text", "title", "namespace", "expires_at", "comment"),
    ENTITY_SKILL: ("name", "description", "example", "steps", "text", "extra"),
    ENTITY_SKILL_TEMPLATE: ("instruction_template",),
    ENTITY_TERM: ("term", "context", "definition"),
    ENTITY_USER_FACT: ("name", "body"),
    # A log's content (arch §5.4): `source`/`session` are part of the content ("what
    # action and which channel/session produced it"), apart from
    # `record_versions.source/session` (the version's own origin, §13.3).
    ENTITY_ASSISTANT_LOG: (
        "action",
        "reason",
        "entity",
        "entity_id",
        "entity_version",
        "entity_ref",
        "outcome",
        "verified",
        "source",
        "session",
    ),
}


def note_projection(
    row: sqlite3.Row | Mapping[str, Any], *, explicit_summary: bool = False
) -> dict[str, Any]:
    """The note projection per §5.3 - the single truth for save/update/merge.

            The content: `text`/`title`/`namespace`/`expires_at`/`comment` (the comment
            is the versioning content field No24, FR-5.6). `summary` enters only on an
            EXPLICIT pass (`explicit_summary=True`) - the worker's auto-regeneration
            creates no versions (§13.3). The service fields (`vector_status`,
            `summary_status`, `links_at`, `node_order_at`, `classified_at`) are not in
            the projection and create no versions (FR-4.2).

    """
    fields: dict[str, Any] = {
        "text": row["text"],
        "title": row["title"],
        "namespace": row["namespace"],
        "expires_at": row["expires_at"],
        "comment": row["comment"],
    }
    if explicit_summary:
        fields["summary"] = row["summary"]
    return fields


def _parsed_extra(raw: Any) -> dict[str, Any] | None:
    """A skill's `extra` from the live row -> the class-fields object; NULL/broken -> `None`.

            In the live table `extra` is a JSON text; in a version it lives as the
            parsed object (one shape for the baseline and the deltas, §13.3): so a
            `null` in a delta honestly means "the field was removed". Empty/damaged
            JSON (a manual DB edit) - `None`.

    """
    if not raw:
        return None
    try:
        parsed = json.loads(str(raw))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) and parsed else None


def skill_projection(row: sqlite3.Row | Mapping[str, Any]) -> dict[str, Any]:
    """The skill projection per §5.3 - the single truth for create/update/delete.

            The content: `name`/`description`/`example`/`steps`/`text`/`extra` (the
            class fields - as the parsed object, §13.3). The service fields
            (`vector_status`; `version` - the head-version mirror) are not in the
            projection and create no versions (FR-4.2).

    """
    return {
        "name": row["name"],
        "description": row["description"],
        "example": row["example"],
        "steps": row["steps"],
        "text": row["text"],
        "extra": _parsed_extra(row["extra"]),
    }


def term_projection(row: sqlite3.Row | Mapping[str, Any]) -> dict[str, Any]:
    """The term projection per §5.3 - the single truth for save/update/delete.

            The content: `term`/`context`/`definition`. The normalized keys
            (`term_norm`, `context_norm`) and `vector_status` are service fields and
            create no versions (FR-4.2).

    """
    return {
        "term": row["term"],
        "context": row["context"],
        "definition": row["definition"],
    }


def user_fact_projection(row: sqlite3.Row | Mapping[str, Any]) -> dict[str, Any]:
    """The user-fact projection per §5.3 - the single truth for CRUD.

            The content: `name`/`body`; `vector_status` is a service field and creates
            no versions (FR-4.2).

    """
    return {"name": row["name"], "body": row["body"]}


def assistant_log_projection(row: sqlite3.Row | Mapping[str, Any]) -> dict[str, Any]:
    """The assistant-log projection per §5.4 - the single truth of the area's CRUD.

            The content: `action`/`reason`/`entity`/`entity_id`/`entity_version`/
            `entity_ref`/`outcome`/`verified`/`source`/`session`. The service fields
            (`vector_status`, `created_at`/`updated_at`/`deleted_at`) are not in the
            projection and create no versions (FR-4.2).

    """
    return {
        "action": row["action"],
        "reason": row["reason"],
        "entity": row["entity"],
        "entity_id": row["entity_id"],
        "entity_version": row["entity_version"],
        "entity_ref": row["entity_ref"],
        "outcome": row["outcome"],
        "verified": row["verified"],
        "source": row["source"],
        "session": row["session"],
    }


# The soft read answer when a version/record is absent: the transport turns it
# into a 404 (no exception - "no version" is a valid outcome).
HINT_VERSION_NOT_FOUND = (
    "version not found for this record — history_list shows the available versions"
)

# The soft read answer with the history layer off (`HISTORY_ENABLED=false`, §8):
# no versions are written or read - empty + this hint (EN), not an exception.
HINT_HISTORY_OFF = "history is off"

# The version's source when the write happened outside `bind_origin` (a direct
# layer call from tests/utilities): the true source is unknown.
DEFAULT_SOURCE = "unknown"


class HistoryValidationError(ValueError):
    """A history-layer error: an unknown entity or a version out of range."""


def _check_entity(entity: str) -> None:
    """The entity must be in the projection registry (§5.3)."""
    if entity not in CONTENT_FIELDS:
        raise HistoryValidationError(
            "unknown entity: " + entity + " (known: " + ", ".join(CONTENT_FIELDS) + ")"
        )


def _history_enabled() -> bool:
    """The history layer gate (§8, `HISTORY_ENABLED`): `false` - no versions written.

            The switch is read from the current settings on EVERY write path: an edit
            with the layer off passes as before and no rows appear in
            `record_versions`. The same gate in reads serves empty with
            `HINT_HISTORY_OFF`.

    """
    return bool(get_settings().history_enabled)


#--- the delta encoding and application (§5.2) --------------------------------

def encode_delta(
    previous: Mapping[str, Any], current: Mapping[str, Any], *, inline_max: int
) -> tuple[dict[str, Any], list[str]]:
    """The delta "field -> the new value's representation" + the changed-field list.

            A field is encoded line-by-line when both the old and the new values are
            strings and the new one is >= `inline_max`; otherwise - the whole value
            (short fields, objects, `null` on removal). The representation is chosen
            per field in every version (§5.2). The field order in the result is
            stable: the old value's fields first, then the new ones (a field's
            appearance goes to the end).

    """
    delta: dict[str, Any] = {}
    changed: list[str] = []
    for field in _ordered_fields(previous, current):
        old = previous.get(field)
        new = current.get(field)
        if field in previous and field in current and old == new:
            continue
        changed.append(field)
        delta[field] = _encode_value(old, new, inline_max=inline_max)
    return delta, changed


def _ordered_fields(
    previous: Mapping[str, Any], current: Mapping[str, Any]
) -> list[str]:
    """The fields of both values without repeats: the old order first, then the new."""
    fields = list(previous)
    fields.extend(field for field in current if field not in previous)
    return fields


def _encode_value(old: Any, new: Any, *, inline_max: int) -> Any:
    """A field value's representation: the line replacement or the whole value."""
    if isinstance(old, str) and isinstance(new, str) and len(new) >= inline_max:
        return _encode_lines(old, new)
    return new


def _encode_lines(previous: str, current: str) -> dict[str, list[list[Any]]]:
    """The line replacement: only the non-`equal` opcodes of SequenceMatcher (§5.2).

            `[i, j, [lines]]` - in the old value the positions `[i, j)` are replaced
            with the listed lines (insert - `i == j`, delete - an empty list).

    """
    previous_lines = previous.split("\n")
    current_lines = current.split("\n")
    matcher = difflib.SequenceMatcher(None, previous_lines, current_lines, autojunk=False)
    opcodes: list[list[Any]] = []
    for tag, start, stop, start_new, stop_new in matcher.get_opcodes():
        if tag == "equal":
            continue
        opcodes.append([start, stop, current_lines[start_new:stop_new]])
    return {"__lines__": opcodes}


def apply_delta(fields: dict[str, Any], delta: Mapping[str, Any]) -> dict[str, Any]:
    """Apply a delta to a field dict (the same object; §5.2).

            One algorithm for all representations: `{"__lines__": ...}` - the line
            replacement in the field's current value (not a string -> an empty
            string); any other value (including `null`) - the whole assignment.

    """
    for field, value in delta.items():
        if isinstance(value, dict) and "__lines__" in value:
            lines = (fields.get(field) or "").split("\n")
            for start, stop, replacement in value["__lines__"]:
                lines[start:stop] = replacement
            fields[field] = "\n".join(lines)
        else:
            fields[field] = value
    return fields


#--- the version origin (a contextvar, §13.4) ------------------------------

# The call's source and session live in a contextvar: the service signatures
# do not change, `asyncio.to_thread` propagates the context; jobs and the
# migration pass the source explicitly (no bind_origin).
_ORIGIN: ContextVar[tuple[str, str | None]] = ContextVar(
    "history_origin", default=(DEFAULT_SOURCE, None)
)
_ROLLBACK_TO: ContextVar[int | None] = ContextVar("history_rollback_to", default=None)


@contextmanager
def bind_origin(
    source: str, session: str | None = None, *, rollback_to: int | None = None
) -> Iterator[None]:
    """Wrap a work segment with the version origin: `source` + `session`.

            `rollback_to` marks the segment as a rollback: the versions get
            `event='rollback'` and `restore_from` (FR-5.4). A nested `bind_origin`
            overrides the outer one; leaving restores the previous value (the
            contextvar).

    """
    origin_token = _ORIGIN.set((source, session))
    rollback_token = _ROLLBACK_TO.set(rollback_to)
    try:
        yield
    finally:
        _ORIGIN.reset(origin_token)
        _ROLLBACK_TO.reset(rollback_token)


def current_origin() -> tuple[str, str | None]:
    """The current version origin (`source`, `session`) of the work segment."""
    return _ORIGIN.get()


def current_rollback_to() -> int | None:
    """The target version number when the segment is a rollback (else `None`)."""
    return _ROLLBACK_TO.get()


#--- writes: only on the caller's connection (§13.5) ----------------------

def _dump(value: Any) -> str:
    """The record's JSON string (Cyrillic as is: the operator's readability)."""
    return json.dumps(value, ensure_ascii=False)


def _origin(source: str | None, session_id: str | None) -> tuple[str, str | None]:
    """The version's source/session: explicit arguments beat the current contextvar."""
    current_source, current_session = current_origin()
    resolved_source = current_source if source is None else source
    resolved_session = current_session if session_id is None else session_id
    return resolved_source, resolved_session


def _head(conn: sqlite3.Connection, entity: str, entity_id: int) -> int:
    """A record's head version (`0` - no history yet)."""
    row = conn.execute(
        "SELECT MAX(version) FROM record_versions WHERE entity = ? AND entity_id = ?",
        (entity, entity_id),
    ).fetchone()
    return 0 if row is None or row[0] is None else int(row[0])


def record_initial(
    conn: sqlite3.Connection,
    entity: str,
    entity_id: int,
    fields: Mapping[str, Any],
    *,
    source: str | None = None,
    session: str | None = None,
) -> int:
    """A record's first version: `version=1, event='baseline'`, the content whole.

            The baseline is the only version with a `snapshot`; everything further is
            deltas (FR-2.1/FR-2.2). Writing a baseline for a record that already has
            one is the caller's error (the history is never rewritten).
            With `HISTORY_ENABLED=false` no version is written - the return is `0`
            (§8).

    """
    _check_entity(entity)
    if not _history_enabled():
        return 0
    resolved_source, resolved_session = _origin(source, session)
    conn.execute(
        "INSERT INTO record_versions (entity, entity_id, version, event, source, "
        "session, changed, snapshot) VALUES (?, ?, 1, 'baseline', ?, ?, '[]', ?)",
        (entity, int(entity_id), resolved_source, resolved_session, _dump(fields)),
    )
    return 1


def record_change(
    conn: sqlite3.Connection,
    entity: str,
    entity_id: int,
    fields: Mapping[str, Any],
    *,
    source: str | None = None,
    session: str | None = None,
    event: str = "content",
) -> int | None:
    """An edit version: a delta from the head version; `None` - nothing changed.

            An empty delta creates no version (no noise). Under
            `current_rollback_to()` the version is marked `event='rollback'` with
            `restore_from` (FR-5.4). For a skill, `skills.version` is raised to the
            head version in the SAME transaction (§13.7). After the write the
            retention policy applies (`_prune`). With `HISTORY_ENABLED=false` no
            version is written (§8).

    """
    _check_entity(entity)
    if not _history_enabled():
        return None
    settings = get_settings()
    inline_max = settings.history_delta_inline_max
    entity_id = int(entity_id)
    head = _head(conn, entity, entity_id)
    if head == 0:
        # The record appeared after the baseline migration and has no history yet:
        # write a baseline (otherwise a delta has nothing to count from).
        return record_initial(
            conn, entity, entity_id, fields, source=source, session=session
        )
    previous = reconstruct(conn, entity, entity_id, head, inline_max=inline_max) or {}
    delta, changed = encode_delta(previous, fields, inline_max=inline_max)
    if not changed:
        return None
    version = head + 1
    rollback_to = current_rollback_to()
    if rollback_to is not None:
        event = "rollback"
    resolved_source, resolved_session = _origin(source, session)
    conn.execute(
        "INSERT INTO record_versions (entity, entity_id, version, event, source, "
        "session, changed, delta, restore_from) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            entity,
            entity_id,
            version,
            event,
            resolved_source,
            resolved_session,
            _dump(changed),
            _dump(delta),
            rollback_to if event == "rollback" else None,
        ),
    )
    if entity == ENTITY_SKILL:
        conn.execute(
            "UPDATE skills SET version = ? WHERE id = ?", (version, entity_id)
        )
    _prune(conn, entity, entity_id, settings)
    return version


def record_event(
    conn: sqlite3.Connection,
    entity: str,
    entity_id: int,
    event: str,
    *,
    source: str | None = None,
    session: str | None = None,
) -> int:
    """An event version without content (`changed='[]'`, a NULL delta): FR-4.3/FR-6.

            A soft delete/restore is a history fact, not a field edit; for a skill the
            `skills.version` mirror rises the same way as on an edit.
            With `HISTORY_ENABLED=false` no version is written - the return is `0`
            (§8).

    """
    _check_entity(entity)
    if not _history_enabled():
        return 0
    settings = get_settings()
    entity_id = int(entity_id)
    version = _head(conn, entity, entity_id) + 1
    resolved_source, resolved_session = _origin(source, session)
    conn.execute(
        "INSERT INTO record_versions (entity, entity_id, version, event, source, "
        "session, changed) VALUES (?, ?, ?, ?, ?, ?, '[]')",
        (entity, entity_id, version, event, resolved_source, resolved_session),
    )
    if entity == ENTITY_SKILL:
        conn.execute(
            "UPDATE skills SET version = ? WHERE id = ?", (version, entity_id)
        )
    _prune(conn, entity, entity_id, settings)
    return version


#--- reads and restoration (FR-5.1/FR-5.2) --------------------------------

def reconstruct(
    conn: sqlite3.Connection,
    entity: str,
    entity_id: int,
    version: int,
    *,
    inline_max: int,
) -> dict[str, Any] | None:
    """A version's content: apply the deltas from the baseline up to `version` inclusive.

            A `snapshot` in any version overrides the accumulated state - that is the
            re-base anchor after prune (§13.8). `None` - no record/version.

    """
    _check_entity(entity)
    entity_id = int(entity_id)
    version = int(version)
    rows = conn.execute(
        "SELECT version, snapshot, delta FROM record_versions WHERE entity = ? "
        "AND entity_id = ? AND version <= ? ORDER BY version",
        (entity, entity_id, version),
    ).fetchall()
    if not rows or int(rows[-1]["version"]) != version:
        return None
    fields: dict[str, Any] = {}
    for row in rows:
        if row["snapshot"] is not None:
            fields = json.loads(row["snapshot"])
        elif row["delta"] is not None:
            fields = apply_delta(fields, json.loads(row["delta"]))
    return fields


def read_versions(
    entity: str,
    entity_id: int,
    settings: Settings,
    *,
    limit: int | None = None,
    offset: int = 0,
) -> dict[str, Any]:
    """A record's version list without bodies: the metadata + "what changed" (FR-5.1).

            The order - newest first (like the operator's skill archive); the page -
            per the lsb-0013 contract (`total`/`has_more`/`next_offset`).

    """
    _check_entity(entity)
    entity_id = int(entity_id)
    page_size = settings.default_list_limit if limit is None else int(limit)
    offset = int(offset)
    if not settings.history_enabled:
        # §8: the history layer switch - no versions; a read serves empty with
        # a hint (EN), not an error.
        return {
            "entity": entity,
            "id": entity_id,
            "versions": [],
            "hint": HINT_HISTORY_OFF,
            **page_fields(0, offset, 0),
        }
    with session(settings) as conn:
        total = int(
            conn.execute(
                "SELECT COUNT(*) FROM record_versions WHERE entity = ? "
                "AND entity_id = ?",
                (entity, entity_id),
            ).fetchone()[0]
        )
        rows = conn.execute(
            "SELECT version, created_at, event, source, session, changed FROM "
            "record_versions WHERE entity = ? AND entity_id = ? "
            "ORDER BY version DESC LIMIT ? OFFSET ?",
            (entity, entity_id, page_size, offset),
        ).fetchall()
    versions = [
        {
            "version": int(row["version"]),
            "created_at": row["created_at"],
            "event": row["event"],
            "source": row["source"],
            "session": row["session"],
            "changed": json.loads(row["changed"]),
        }
        for row in rows
    ]
    return {
        "entity": entity,
        "id": entity_id,
        "versions": versions,
        **page_fields(total, offset, len(versions)),
    }


def read_version(
    entity: str, entity_id: int, version: int, settings: Settings
) -> dict[str, Any]:
    """A specific version's content: the metadata + the reconstructed fields.

            No such version/record - a soft answer with a `hint` (the transport turns
            it into a 404); no exceptions: "no version" is a valid outcome (§6.1).

    """
    _check_entity(entity)
    entity_id = int(entity_id)
    if not settings.history_enabled:
        # §8: the history layer switch - versions are not read (empty + a hint).
        return {
            "entity": entity,
            "id": entity_id,
            "version": int(version),
            "hint": HINT_HISTORY_OFF,
        }
    with session(settings) as conn:
        row = conn.execute(
            "SELECT version, created_at, event, source, session, changed FROM "
            "record_versions WHERE entity = ? AND entity_id = ? AND version = ?",
            (entity, entity_id, int(version)),
        ).fetchone()
        if row is None:
            return {
                "entity": entity,
                "id": entity_id,
                "version": int(version),
                "hint": HINT_VERSION_NOT_FOUND,
            }
        fields = reconstruct(
            conn,
            entity,
            entity_id,
            int(row["version"]),
            inline_max=settings.history_delta_inline_max,
        )
    return {
        "entity": entity,
        "id": entity_id,
        "version": int(row["version"]),
        "created_at": row["created_at"],
        "event": row["event"],
        "source": row["source"],
        "session": row["session"],
        "changed": json.loads(row["changed"]),
        "fields": fields,
    }


def history_stats(settings: Settings) -> dict[str, Any]:
    """The history size: the per-entity version counters + the total (FR-10.3).

            The shape is stable: every known entity is present, even with a zero (the
            operator sees which history area is still empty). With
            `HISTORY_ENABLED=false` the counters are zero + `HINT_HISTORY_OFF` (§8).

    """
    if not settings.history_enabled:
        return {
            "entities": dict.fromkeys(CONTENT_FIELDS, 0),
            "total": 0,
            "hint": HINT_HISTORY_OFF,
        }
    with session(settings) as conn:
        counts = {
            str(row["entity"]): int(row["count"])
            for row in conn.execute(
                "SELECT entity, COUNT(*) AS count FROM record_versions GROUP BY entity"
            )
        }
    entities: dict[str, int] = dict.fromkeys(CONTENT_FIELDS, 0)
    entities.update(counts)
    return {"entities": entities, "total": sum(entities.values())}


def entity_fields(
    entity: str, entity_id: int, settings: Settings
) -> tuple[dict[str, Any], str] | None:
    """The projection of a LIVE entity row + its state - the rollback point (§7.3).

            Returns `(fields, state)`: `fields` - the live row's §5.3 projection (the
            same one that would go into a version); `state` - `"active"` or
            `"deleted"` (the row is soft-deleted); `None` - no row. It reads the live
            rows of all versioned entities - the raw SQL over foreign tables is never
            exposed (f10-04 reads the state the same way). The skills global template -
            the `skills_meta` key (`skill_template`, id 0): it "exists" while the key
            is in the table.

    """
    _check_entity(entity)
    entity_id = int(entity_id)
    with session(settings) as conn:
        if entity == ENTITY_NOTE:
            row = conn.execute(
                "SELECT text, title, namespace, expires_at, comment, summary, "
                "deleted_at "
                "FROM notes WHERE id = ?",
                (entity_id,),
            ).fetchone()
            if row is None:
                return None
            fields: dict[str, Any] = note_projection(row)
        elif entity == ENTITY_SKILL:
            row = conn.execute(
                "SELECT * FROM skills WHERE id = ?", (entity_id,)
            ).fetchone()
            if row is None:
                return None
            fields = skill_projection(row)
        elif entity == ENTITY_TERM:
            row = conn.execute(
                "SELECT term, context, definition, deleted_at FROM terms "
                "WHERE id = ?",
                (entity_id,),
            ).fetchone()
            if row is None:
                return None
            fields = term_projection(row)
        elif entity == ENTITY_USER_FACT:
            row = conn.execute(
                "SELECT name, body, deleted_at FROM user_facts WHERE id = ?",
                (entity_id,),
            ).fetchone()
            if row is None:
                return None
            fields = user_fact_projection(row)
        elif entity == ENTITY_ASSISTANT_LOG:
            row = conn.execute(
                "SELECT action, reason, entity, entity_id, entity_version, "
                "entity_ref, outcome, verified, source, session, deleted_at "
                "FROM assistant_logs WHERE id = ?",
                (entity_id,),
            ).fetchone()
            if row is None:
                return None
            fields = assistant_log_projection(row)
        else:  # ENTITY_SKILL_TEMPLATE - the global template (id 0)
            row = conn.execute(
                "SELECT value FROM skills_meta WHERE key = ?",
                (INSTRUCTION_TEMPLATE_KEY,),
            ).fetchone()
            if row is None or row["value"] is None:
                return None
            return {"instruction_template": row["value"]}, "active"
        state = "deleted" if row["deleted_at"] is not None else "active"
    return fields, state


# The area flags of the versioned entities (No40, no40-02): an entity of a
# disabled area is indistinguishable from a nonexistent one. `note` and the
# skills global template (`skill_template`) are core and have no flag; `assistant_log` - the logs area.
_ENTITY_AREA_FLAGS: dict[str, str] = {
    ENTITY_SKILL: "area_skills_enabled",
    ENTITY_TERM: "area_terms_enabled",
    ENTITY_USER_FACT: "area_user_facts_enabled",
    ENTITY_ASSISTANT_LOG: "area_logs_enabled",
}


def resolve_entity(
    entity: str, entity_id: int, settings: Settings
) -> dict[str, Any] | None:
    """A log reference object's state and its head version (arch §11, §13.4).

            Returns `{"state": str, "current_version": int | None} | None`:
            * `"active"` - a live row exists, `deleted_at IS NULL`, the area is on;
            * `"deleted"` - the row exists but is soft-deleted (the log record lives, FR-4.3);
            * `"unavailable"` - no row OR the entity's area is switched off by the flag
              (whether the area exists is not revealed);
            * `None` - a non-versioned (`file`/`commit`/`tracker_task`/`config`/
              `contour`) or unknown entity (the caller sets `"external"`), and also at
              `HISTORY_ENABLED=false`: no history - the state is not built.

            `current_version` - the head version from `record_versions` (MAX(version))
            for an active/deleted row, else `None`. The live rows of the versioned
            entities are read by the same substrate as `entity_fields`; the raw SQL
            over foreign tables is never exposed (the logs-area isolation, §4/§13.4).

    """
    if not settings.history_enabled:
        return None
    if entity not in CONTENT_FIELDS:
        return None
    flag = _ENTITY_AREA_FLAGS.get(entity)
    if flag is not None and not getattr(settings, flag):
        return {"state": "unavailable", "current_version": None}
    live = entity_fields(entity, int(entity_id), settings)
    if live is None:
        return {"state": "unavailable", "current_version": None}
    _fields, state = live
    with session(settings) as conn:
        current = _head(conn, entity, int(entity_id))
    return {"state": state, "current_version": current}


#--- the retention policy: prune with a re-base (§13.8) -----------------------------

def _prune(
    conn: sqlite3.Connection, entity: str, entity_id: int, settings: Settings
) -> None:
    """Prune a record's oldest versions and re-base the earliest kept one.

            The `HISTORY_MAX_VERSIONS` ceiling and the `HISTORY_RETENTION_DAYS` term
            (0 - unlimited) keep `version=1` (the chain anchor) and the newest
            versions. After the removal the earliest kept version is rewritten into a
            full `snapshot` (`delta=NULL`): otherwise its delta would reference a
            deleted version and the chain would not reconstruct. The numbers are
            never reused.

    """
    max_versions = settings.history_max_versions
    retention_days = settings.history_retention_days
    if max_versions <= 0 and retention_days <= 0:
        return
    rows = conn.execute(
        "SELECT version, delta, created_at FROM record_versions WHERE entity = ? "
        "AND entity_id = ? ORDER BY version",
        (entity, entity_id),
    ).fetchall()
    if not rows:
        return
    versions = [int(row["version"]) for row in rows]
    base = versions[0]
    doomed: set[int] = set()
    if max_versions > 0 and len(versions) > max_versions:
        keep_count = max_versions - 1  # version=1 always stays
        newest = versions[-keep_count:] if keep_count > 0 else []
        keep = {base, *newest}
        doomed |= {version for version in versions if version not in keep}
    if retention_days > 0:
        cutoff = conn.execute(
            "SELECT strftime('%Y-%m-%dT%H:%M:%SZ','now', ?)",
            (f"-{int(retention_days)} days",),
        ).fetchone()[0]
        doomed |= {
            int(row["version"])
            for row in rows
            if int(row["version"]) != base and str(row["created_at"]) < cutoff
        }
    if not doomed:
        return
    survivors = [version for version in versions if version not in doomed]
    # Re-base the earliest kept version only when versions below it are removed
    # (otherwise its delta would reference a removed one and the chain breaks).
    anchor = survivors[0] if survivors[0] != base else next(iter(survivors[1:]), None)
    snapshot = None
    if anchor is not None and any(version < anchor for version in doomed):
        fields = reconstruct(
            conn, entity, entity_id, anchor, inline_max=settings.history_delta_inline_max
        )
        if fields is not None:
            snapshot = _dump(fields)
    placeholders = ", ".join("?" for _ in doomed)
    conn.execute(
        f"DELETE FROM record_versions WHERE entity = ? AND entity_id = ? "
        f"AND version IN ({placeholders})",
        (entity, entity_id, *sorted(doomed)),
    )
    if snapshot is not None and anchor is not None:
        conn.execute(
            "UPDATE record_versions SET snapshot = ?, delta = NULL WHERE entity = ? "
            "AND entity_id = ? AND version = ?",
            (snapshot, entity, entity_id, anchor),
        )
