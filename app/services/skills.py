"""SkillsService — the skills area's service layer (lsb-0007, release 3.0.0).

    The skill shape and limits — arch lsb-0007 §3.1; save/edit/versions/deletion —
    §3.4; the canonical texts (soft-failure hints, the template seed) — §3.8. One
    service codebase for MCP and REST (substrate §3.4): the transport is a thin
    wrapper over the service.

    Key properties
    --------------
    * **Shape validation is mandatory** — limits from settings
      (`SKILL_*_MAX_CHARS`); a violation → `SkillValidationError`, the exception
      text = the verbatim §3.8 canon hint (the transport gives it to the client as
      fail + hint; the skill is NOT saved).
    * **No vectorization at write time** (substrate §3.3): the write goes with
      `vector_status='pending'`; the worker's `areas` loop catches up the write's
      vectors (after creation/edit the service wakes it — `set_areas_notifier`).
    * **Versions (#41, release 3.2.0):** the edit history lives in the shared
      `record_versions` table and is written by the shared `history` layer in THE
      SAME transaction as the edit itself (FR-2.4, §13.5); `skills.version` is a
      mirror of the head version, kept by the history layer (§13.7). The old copy
      in `skill_versions` is gone: the table is left as a legacy cast, the code
      does not fill it (one mechanic, §13.7). The archive takes part in no output:
      `get`/`list` read only `skills` and only active rows.
    * **Creation anti-synonymy (FR-5.2, lsb-0007-03):** at creation (without `id`)
      the candidate's text (`name + description`) is compared by cosine with the
      active skills — one `embed_texts` batch, L2 normalization, the threshold
      `SKILL_SYNONYM_SIMILARITY=0.90` → a soft failure `{created: False, hint}`
      with the verbatim §3.8 canon hint (creation does NOT happen); an edit by
      `id` does not run the prefilter (otherwise a skill could not be renamed);
      an embedder failure — the prefilter is skipped (creation passes) + an
      `skills_antiseonymy_skipped` event in the log. The mechanics follow the
      lsb-0005 pattern.
    * **Deletion is soft** (`deleted_at`): a repeated/nonexistent id → a soft
      answer with the "skill not found…" hint (restoration — by the operator; no
      handle).
    * **The read composite** (`get`) is assembled over the global
      `instruction_template` from `skills_meta` (the seed — `init_db`): the
      template is served as a separate "how to execute the steps" section, not as
      a part of a concrete skill.
    * **Search is a probe** (`search`, lsb-0007-02): the area hybrid (substrate
      §3.5) vec0-KNN (`skills_vec`, the vector = `name + description`) + FTS5/BM25
      (`skills_fts`) → RRF; the output has no bodies, empty — a soft answer with
      the §3.8 canon hint. Notes and other areas are not read (isolation).

    Response contracts (full; the MCP layer trims service fields by a whitelist —
    lsb-0007-03):
    - save (creation) → {id, created: True, version}
    - save (a too-similar creation) → {created: False, hint}
    - save (edit)   → {id, updated: True, version}
    - save (an edit of a nonexistent id) → {id, updated: False, hint}
    - get    → {id, name, description, example?, steps, text, instruction_template,
                extra?} (example/extra — only when set); not found → {id, hint}
    - list   → {items: [{id, name, description}], total, has_more, next_offset,
                next_cursor} (active only; no bodies; lsb-0013 page fields)
    - search → {results: [{id, name, description, score}], warning?} (no bodies);
                empty → {results: [], hint, warning?} with the verbatim §3.8 hint
    - delete → {id, deleted: True} | {id, deleted: False, hint}
    - instruction_template() → {instruction_template}
    - set_instruction_template(text) → {instruction_template, updated: True}
    - versions(id) → {id, versions: [{version, name, description, example, steps,
                     text, extra, created_at}]}; not found/deleted →
                     {id, versions: [], hint} (the archive is visible to the REST
                     operator only)
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from collections.abc import Callable
from typing import Any

from app.config import Settings
from app.services import history
from app.services.areas import AreaSearch, AreaSearchValidationError, SKILLS_AREA
from app.services.embedding import Embedder, EmbeddingService
from app.services.listing import page_fields
from app.services.search import MAX_TOP_K
from app.storage.db import (
    INSTRUCTION_TEMPLATE_KEY,
    INSTRUCTION_TEMPLATE_SEED,
    session,
    transaction,
)

# The skill's class fields (lsb-0007 §3.1): only these keys are valid in the JSON `extra`.
EXTRA_FIELDS: tuple[str, ...] = (
    "trigger",
    "mode",
    "preconditions",
    "fallbacks",
    "invariant",
    "exceptions",
    "guardrails",
    "references",
    "output_contract",
    "behavior_contract",
)

# Valid values of the `mode` field (§3.1): collaborative / autonomous execution.
SKILL_MODES: tuple[str, ...] = ("collaborative", "autonomous")

# --- Soft-failure hints (canon §3.8: "verbatim constants in code") ---------
# The canon's limits table; the texts are English — the model reads them as a hint.
# The numbers come from the 2026-10-12 decision (canon §6): description 400, steps 1000,
# text 7500, example 2000, an extra field 1000 (the sum 4000), the template 2000;
# `name` (65) unchanged. Here the number is the ANSWER ("how much is allowed"); the source
# of truth — the setting (`SKILL_*_MAX_CHARS`) and canon §6.
HINT_NAME_LIMIT = (
    "skill not saved: name limit is 65 characters (≤5 words recommended)"
)
HINT_DESCRIPTION_LIMIT = (
    "skill not saved: description limit is 400 characters — shorten it"
)
HINT_STEPS_LIMIT = "skill not saved: steps limit is 1000 characters — shorten it"
HINT_TEXT_LIMIT = "skill not saved: text limit is 7500 characters — shorten it"
HINT_EXAMPLE_LIMIT = (
    "skill not saved: example limit is 2000 characters — shorten it"
)
HINT_EXTRA_LIMIT = (
    "skill not saved: an optional field limit is 1000 characters, all optional "
    "fields together — 4000"
)
HINT_NOT_FOUND = "skill not found (possibly deleted); the actual list — skills_list"

# An empty `skills_search` (§3.8): the search works as a probe — "no skill" is
# a valid outcome; the model does not enter skills without a reason.
HINT_SEARCH_EMPTY = (
    "no skill found for this task — do the task as usual; if you worked out a "
    "repeatable procedure, save it via skills_save"
)

# A non-empty `skills_search` (arch 3.2.0 §7.1/§7.4, #38): the hint "the body — only
# via `skills_get`" — EXACTLY one per answer (a top-level field), NOT in the `results`
# items: per-item duplication wasted the agent's context. Short
# (≤120 chars) — the budget of the baked-in text (LSBN-F-11).
HINT_SKILLS_BODY = "skill bodies are read only via skills_get"

# The creation anti-synonymy (§3.8): the hint leads to the existing skill;
# creation does not happen. The template substitutes the nearest skill's id and name.
HINT_SIMILAR_SKILL = (
    "there is a similar skill: {id} — {name}; reuse or update it "
    "(skills_save with id=…) or make this skill clearly different"
)

# Hints the §3.8 canon table does not set (the canon describes only the limits):
# the shape fields' obligatoriness, an unknown `extra` key, the `mode` value
# and the global template's limit. The style and language — like the canonical ones.
HINT_EXTRA_OBJECT = (
    "skill not saved: extra must be a single JSON object with known fields"
)
HINT_MODE = "skill not saved: mode must be collaborative or autonomous"
HINT_TEMPLATE_LIMIT = (
    "template not saved: instruction template limit is 2000 characters — shorten it"
)


def _l2_norm(vec: list[float]) -> float:
    """The vector's Euclidean norm (L2) — like promotion's/mcp.py `_l2_norm`."""
    return math.sqrt(sum(value * value for value in vec))


class SkillValidationError(ValueError):
    """A skill-shape violation (limits, `extra`) — a soft failure with a hint.

            The exception text is a hint for the model (the MCP/REST transport gives
            it to the client as fail + hint); the skill is NOT saved: the precedents
            are `TitleValidationError`/`TITLE_HINT` (Phase 11) and `NamespaceError`.
    """


class SkillsService:
    """The skills area CRUD: shape, the version archive, the global template.

            DI: `settings` (the shape limits), `embedding` (encoding the hybrid
            search query — lsb-0007-02; the creation anti-synonymy — lsb-0007-03).
            The skill body is NOT given to the encoder by the synchronous write path
            (substrate §3.3): the write's vectors are caught up by the worker's
            `areas` loop over `vector_status='pending'`; the encoder on the write path
            is called only by the creation anti-synonymy (the `name + description`
            text of the candidate).
    """

    def __init__(self, settings: Settings, embedding: Embedder | None = None) -> None:
        self._settings = settings
        # DI for tests: a HashEmbedder/fake with a call log instead of the network.
        # The write does not give the skill body to the encoder (the areas loop
        # catches up the vectors); the encoder on the write path is called only by the
        # creation anti-synonymy (the candidate's `name + description` text).
        self._embedding: Embedder = (
            embedding if embedding is not None else EmbeddingService(settings)
        )
        # The area's hybrid search (substrate §3.5): the same mechanics as the
        # notes' — vec0-KNN over `skills_vec` + FTS5/BM25 over `skills_fts` →
        # RRF. The helper's SQL addresses only the skills tables (isolation).
        self._area_search = AreaSearch(settings, SKILLS_AREA, self._embedding)
        # A signal to the worker (main.py): wake the areas loop at once when
        # an area pending record appears, instead of waiting out the grown back-off.
        self._areas_notifier: Callable[[], None] | None = None

    def set_areas_notifier(self, notifier: Callable[[], None] | None) -> None:
        """Wire the wakeup signal of the worker's `areas` loop (main.py).

                    Modeled on `NoteService.set_summary_notifier`: the service is assembled
                    before the worker, the notifier arrives after its creation.
        """
        self._areas_notifier = notifier

    # --- writes: creation and edit (arch §3.4) --------------------------------

    def save(
        self,
        *,
        id: int | None = None,
        name: str,
        description: str,
        steps: str,
        text: str,
        example: str | None = None,
        extra: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Create a skill (without `id`) or edit an existing one (with `id`).

                    Shape validation is mandatory and precedes any write: a limit/`extra`
                    violation → `SkillValidationError` with the verbatim hint (§3.8).
                    Creation (without `id`) runs the anti-synonymy (FR-5.2, lsb-0007-03):
                    a candidate too similar to an active skill → a soft failure
                    `{created: False, hint}` without a write; an edit by `id` does not run
                    the prefilter. An edit writes a version delta into the shared history
                    (§13.5) and updates the row in one transaction; `version` is a mirror
                    of the head version, kept by the history layer (§13.7). A write always
                    leaves `vector_status='pending'` — the `areas` loop catches up the
                    vectors (the skill body is not encoded in the synchronous path).
        """
        checked_name = self._checked_field(
            "name", name, self._settings.skill_name_max_chars, HINT_NAME_LIMIT
        )
        checked_description = self._checked_field(
            "description",
            description,
            self._settings.skill_description_max_chars,
            HINT_DESCRIPTION_LIMIT,
        )
        checked_steps = self._checked_field(
            "steps", steps, self._settings.skill_steps_max_chars, HINT_STEPS_LIMIT
        )
        checked_text = self._checked_field(
            "text", text, self._settings.skill_text_max_chars, HINT_TEXT_LIMIT
        )
        checked_example = self._checked_example(example)
        checked_extra = self._checked_extra(extra)
        if id is None:
            # The creation anti-synonymy (FR-5.2, arch §3.4): the cosine of the
            # candidate's text (`name + description`) against the active skills.
            # The nearest above the threshold — a soft failure (no write); the hint leads
            # to editing the existing skill; an edit by `id` does not enter here.
            similar = self._antiseonymy_nearest(checked_name, checked_description)
            if similar is not None:
                nearest_id, nearest_name, cosine = similar
                if cosine > self._settings.skill_synonym_similarity:
                    return {
                        "created": False,
                        "hint": HINT_SIMILAR_SKILL.format(
                            id=nearest_id, name=nearest_name
                        ),
                    }
            result = self._create(
                checked_name,
                checked_description,
                checked_example,
                checked_steps,
                checked_text,
                checked_extra,
            )
        else:
            result = self._update(
                id,
                checked_name,
                checked_description,
                checked_example,
                checked_steps,
                checked_text,
                checked_extra,
            )
        if result.get("created") or result.get("updated"):
            self._notify_areas_pending()
        return result

    def _create(
        self,
        name: str,
        description: str,
        example: str | None,
        steps: str,
        text: str,
        extra: str | None,
    ) -> dict[str, Any]:
        """INSERT a new skill (version=1, vector_status='pending').

                    #41 (§13.5): the baseline version is written by THE SAME transaction
                    as the INSERT — the state "the skill exists, no history" never happens;
                    the answer's `version` comes from `skills.version` (the head-version
                    mirror, §13.7).
        """
        with session(self._settings) as conn, transaction(conn):
            cursor = conn.execute(
                "INSERT INTO skills (name, description, example, steps, text, "
                "extra, version, vector_status) "
                "VALUES (?, ?, ?, ?, ?, ?, 1, 'pending')",
                (name, description, example, steps, text, extra),
            )
            new_id = int(cursor.lastrowid or 0)
            row = conn.execute(
                "SELECT * FROM skills WHERE id = ?", (new_id,)
            ).fetchone()
            history.record_initial(
                conn, history.ENTITY_SKILL, new_id, history.skill_projection(row)
            )
            version = int(row["version"])
        return {"id": new_id, "created": True, "version": version}

    def _update(
        self,
        skill_id: int,
        name: str,
        description: str,
        example: str | None,
        steps: str,
        text: str,
        extra: str | None,
    ) -> dict[str, Any]:
        """A row edit + a version delta — in ONE transaction (§13.5).

                    The version is written by the shared history layer (`record_change`),
                    not by a copy into `skill_versions` — there is no second versioning
                    mechanic in the code anymore (§13.7; the legacy table is not filled).
                    `record_change` itself raises `skills.version` to the head version (the
                    mirror, §13.7), so the answer's number is taken from there. An
                    inactive/nonexistent id → a soft answer with a hint; an edit without
                    content changes creates no version.
        """
        with session(self._settings) as conn, transaction(conn):
            row = conn.execute(
                "SELECT * FROM skills WHERE id = ? AND deleted_at IS NULL",
                (skill_id,),
            ).fetchone()
            if row is None:
                return {"id": skill_id, "updated": False, "hint": HINT_NOT_FOUND}
            conn.execute(
                "UPDATE skills SET name = ?, description = ?, example = ?, "
                "steps = ?, text = ?, extra = ?, "
                "vector_status = 'pending', "
                "updated_at = strftime('%Y-%m-%dT%H:%M:%SZ','now') "
                "WHERE id = ? AND deleted_at IS NULL",
                (name, description, example, steps, text, extra, skill_id),
            )
            updated_row = conn.execute(
                "SELECT * FROM skills WHERE id = ?", (skill_id,)
            ).fetchone()
            new_version = history.record_change(
                conn,
                history.ENTITY_SKILL,
                skill_id,
                history.skill_projection(updated_row),
            )
            version = int(row["version"]) if new_version is None else new_version
        return {"id": skill_id, "updated": True, "version": version}

    # --- reads (arch §3.1, §3.3) ----------------------------------------------

    def get(self, id: int) -> dict[str, Any]:
        """The skill composite: the shape sections + the global `instruction_template`.

                    Sections: `name` + `description` + `example` (only when set) +
                    `steps` + `text` + `instruction_template` ("how to execute the steps" —
                    a separate section, not a part of a concrete skill, §3.1). `extra`
                    (the class fields) is added only when set — the full record for REST.
                    The version archive and deleted rows take no part in the output; not
                    found → a soft answer with the §3.8 canon hint.
        """
        with session(self._settings) as conn:
            row = conn.execute(
                "SELECT * FROM skills WHERE id = ? AND deleted_at IS NULL",
                (id,),
            ).fetchone()
            if row is None:
                return {"id": id, "hint": HINT_NOT_FOUND}
            template = self._read_template(conn)
        composite: dict[str, Any] = {
            "id": int(row["id"]),
            "name": row["name"],
            "description": row["description"],
        }
        if row["example"]:
            composite["example"] = row["example"]
        composite["steps"] = row["steps"]
        composite["text"] = row["text"]
        composite["instruction_template"] = template
        extra = _extra_dict(row["extra"])
        if extra:
            composite["extra"] = extra
        return composite

    def list(
        self,
        limit: int | None = None,
        offset: int = 0,
        max_limit: int | None = None,
    ) -> dict[str, Any]:
        """A compact listing of active skills: `{items, total, …}`, no bodies.

                    `items` — only `id`, `name`, `description` (§3.3); the version archive
                    and deleted records are not visible. Pagination is the shared listing
                    contract (lsb-0013, arch §3.1–3.2): the ceiling is set by the surface
                    (`max_limit`; None → `list_max_limit_rest`, so existing callers keep
                    working), `offset ≥ 0`, and the page fields are computed by
                    `page_fields` — exactly as for notes, with the same limit error text.
        """
        max_limit = (
            self._settings.list_max_limit_rest if max_limit is None else max_limit
        )
        limit = self._settings.default_list_limit if limit is None else limit
        if not 1 <= limit <= max_limit:
            raise SkillValidationError(
                f"limit: expected 1..{max_limit}, got {limit}"
            )
        if offset < 0:
            raise SkillValidationError(f"offset: expected ≥ 0, got {offset}")
        with session(self._settings) as conn:
            rows = conn.execute(
                "SELECT id, name, description FROM skills "
                "WHERE deleted_at IS NULL "
                "ORDER BY updated_at DESC, id DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
            total = int(
                conn.execute(
                    "SELECT COUNT(*) FROM skills WHERE deleted_at IS NULL"
                ).fetchone()[0]
            )
        return {
            "items": [
                {
                    "id": int(row["id"]),
                    "name": row["name"],
                    "description": row["description"],
                }
                for row in rows
            ],
            **page_fields(total, offset, len(rows)),
        }

    @property
    def settings(self) -> Settings:
        """The service settings — the transport reads the announce limits (§3.5).

                    The transport assembles the announce block itself (`_skills_announce`),
                    but the limits (`skill_announce_max_chars` /
                    `skill_announce_description_chars`) live in Settings:
                    `build_instructions(services)` receives no settings.
        """
        return self._settings

    def announce_items(self) -> list[dict[str, Any]]:
        """The whole registry of active skills — for the announce block (§3.5).

                    A separate method, not `list()`: the announce needs the whole registry
                    (rows `id — name: description`), while the listing contract is capped
                    by the surface ceiling (lsb-0013: 20 on MCP, 50 on REST). The order is
                    the same as the listing's: `updated_at DESC, id DESC`; the version
                    archive and deleted records are not visible. Skill bodies are not read
                    — only the compact fields.
        """
        with session(self._settings) as conn:
            rows = conn.execute(
                "SELECT id, name, description FROM skills "
                "WHERE deleted_at IS NULL "
                "ORDER BY updated_at DESC, id DESC"
            ).fetchall()
        return [
            {
                "id": int(row["id"]),
                "name": row["name"],
                "description": row["description"],
            }
            for row in rows
        ]

    def search(self, query: str, top_k: int | None = None) -> dict[str, Any]:
        """The skill hybrid search — a "probe" (arch §3.3, substrate §3.5).

                    An `AreaSearch` build for the skills area: vec0-KNN over `skills_vec`
                    (the vector = `name + description`, substrate §3.3) + FTS5/BM25 over
                    `skills_fts` (`name`/`description`/`steps`/`text`) → RRF, the
                    `deleted_at IS NULL` filter. The output is compact — no bodies:
                    `{results: [{id, name, description, score}], hint, warning?}`. The hint
                    is EXACTLY one per answer, at the top level (arch 3.2.0 §7.1, #38): a
                    non-empty result → `HINT_SKILLS_BODY` ("the body — only via
                    `skills_get`"), an empty one → `HINT_SEARCH_EMPTY` (a probe: no skill —
                    a valid outcome, no fail); the `results` items carry no `hint` field.
                    An embedder failure does not break the search: FTS-only + `warning`
                    (the notes precedent, NFR-3) — the warning is preserved in the
                    service's answer, the MCP layer will trim it (lsb-0007-03).

                    `top_k` — the bounds like the notes search (`1..MAX_TOP_K`), a
                    violation — `SkillValidationError` (a soft transport failure, like
                    `list`). The helper's domain query constraint (length) is converted to
                    the same area service failure type. Isolation: the helper's SQL
                    addresses only `skills`/`skills_fts`/`skills_vec` — notes and other
                    areas are not read.
        """
        if top_k is not None and not 1 <= top_k <= MAX_TOP_K:
            raise SkillValidationError(
                f"top_k: expected 1..{MAX_TOP_K}, got {top_k}"
            )
        try:
            found = self._area_search.search(query, top_k)
        except AreaSearchValidationError as exc:  # a query outside the area's domain
            raise SkillValidationError(str(exc)) from exc
        answer: dict[str, Any] = {
            "results": found["results"],
            "warning": found.get("warning"),
        }
        if found["results"]:
            # Canon §3.8 (#38): a non-empty skill search carries the body-reading hint
            # — exactly one, at the answer's top level (not on an item).
            answer["hint"] = HINT_SKILLS_BODY
        else:
            # Canon §3.8: an empty skill search carries exactly one hint (the substrate's
            # short-query hints are not carried into the skills output).
            answer["hint"] = HINT_SEARCH_EMPTY
        return answer

    # --- deletion (soft delete, arch §3.4) ------------------------------------

    def delete(self, id: int) -> dict[str, Any]:
        """Soft-delete a skill: `deleted_at` = now; the row/indexes stay alive.

                    Idempotent in meaning: a repeated/nonexistent id → a soft
                    `deleted: False` answer + the "skill not found…" hint (restoration —
                    the operator, the notes precedent). A deleted skill vanishes from all
                    the area's outputs.
        """
        with session(self._settings) as conn, transaction(conn):
            cursor = conn.execute(
                "UPDATE skills SET deleted_at = "
                "strftime('%Y-%m-%dT%H:%M:%SZ','now') "
                "WHERE id = ? AND deleted_at IS NULL",
                (id,),
            )
            deleted = bool(cursor.rowcount)
            if deleted:
                # #41 (FR-6.1, §13.5): the soft delete — a version event in the
                # same transaction; a repeated/nonexistent deletion gives no version.
                history.record_event(conn, history.ENTITY_SKILL, id, "delete")
        if not deleted:
            return {"id": id, "deleted": False, "hint": HINT_NOT_FOUND}
        return {"id": id, "deleted": True}

    # --- the global template (skills_meta, arch §3.1) -------------------------

    def instruction_template(self) -> dict[str, Any]:
        """Read the global `instruction_template` ("how to execute the steps")."""
        with session(self._settings) as conn:
            return {"instruction_template": self._read_template(conn)}

    def set_instruction_template(self, text: str) -> dict[str, Any]:
        """Edit the global template (the operator via REST).

                    Validation ≤ `INSTRUCTION_TEMPLATE_MAX_CHARS` (canon §3.1/§3.8):
                    a violation → `SkillValidationError` + a soft failure. The existing
                    value is overwritten deliberately (that is the operator's edit);
                    the `init_db` seed does not overwrite an edit.
        """
        normalized = text.strip() if isinstance(text, str) else ""
        if not normalized:
            raise SkillValidationError(self._required_hint("instruction_template"))
        if len(normalized) > self._settings.instruction_template_max_chars:
            raise SkillValidationError(HINT_TEMPLATE_LIMIT)
        with session(self._settings) as conn, transaction(conn):
            conn.execute(
                "INSERT INTO skills_meta (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (INSTRUCTION_TEMPLATE_KEY, normalized),
            )
            # #41 (FR-3.4, §13.5): the global template — a separate history entity
            # (`skill_template`, id 0); the version is written by the same layer and
            # the same transaction as the edit.
            history.record_change(
                conn,
                history.ENTITY_SKILL_TEMPLATE,
                0,
                {"instruction_template": normalized},
            )
        return {"instruction_template": normalized, "updated": True}

    # --- the version archive (the REST operator only, arch §3.4) --------------

    def versions(self, id: int) -> dict[str, Any]:
        """The skill's version-copy archive (`skill_versions`) — REST only.

                    Each edit copies the PREVIOUS content there with the previous version
                    number (§3.4), so the archive is "old versions as copies". The order —
                    newest first. The archive takes no part in the area's outputs: this
                    method reads it separately; neither `get`, nor `list`, nor `search`
                    see `skill_versions` (there is no MCP tool for the archive at all).
                    The skill not found/deleted → a soft answer with an empty archive and
                    the §3.8 canon hint: the transport turns it into 404.
        """
        with session(self._settings) as conn:
            row = conn.execute(
                "SELECT id FROM skills WHERE id = ? AND deleted_at IS NULL",
                (id,),
            ).fetchone()
            if row is None:
                return {"id": id, "versions": [], "hint": HINT_NOT_FOUND}
            rows = conn.execute(
                "SELECT version, name, description, example, steps, text, "
                "extra, created_at FROM skill_versions WHERE skill_id = ? "
                "ORDER BY version DESC",
                (id,),
            ).fetchall()
        return {
            "id": id,
            "versions": [
                {
                    "version": int(item["version"]),
                    "name": item["name"],
                    "description": item["description"],
                    "example": item["example"],
                    "steps": item["steps"],
                    "text": item["text"],
                    "extra": _extra_dict(item["extra"]),
                    "created_at": item["created_at"],
                }
                for item in rows
            ],
        }

    # --- internal -------------------------------------------------------------

    def _antiseonymy_nearest(
        self, name: str, description: str
    ) -> tuple[int, str, float] | None:
        """The nearest active skill by the cosine of the `name + description` text.

                    The mechanics follow `_antiseonymy_nearest` (lsb-0005, mcp.py) and the
                    promotion's `_nearest_node`: the candidate's text and the active
                    skills' texts in ONE `embed_texts` batch (the format —
                    `AreaSpec.embed_text`, the same as the area's vector:
                    `name + description`), L2 normalization,
                    dot product = cosine (insensitive to the provider's scale).

                    Returns (id, name, cosine) of the nearest or None: the active registry
                    is empty / the candidate's norm is 0 / an embedder failure — the NFR-3
                    degradation: the prefilter is skipped (creation passes), an
                    `skills_antiseonymy_skipped` event goes to the log. The caller (save)
                    compares the threshold.
        """
        with session(self._settings) as conn:
            rows = conn.execute(
                "SELECT id, name, description FROM skills "
                "WHERE deleted_at IS NULL"
            ).fetchall()
        if not rows:
            return None  # no active skills — nothing to compare with
        candidate = {"name": name, "description": description}
        try:
            vectors = self._embedding.embed_texts(
                [SKILLS_AREA.embed_text(candidate)]
                + [SKILLS_AREA.embed_text(row) for row in rows]
            )
        except Exception:
            # An embedder failure does not block creation (the lsb-0005 precedent):
            # the prefilter is skipped; the event is observable in the log (NFR-4).
            logging.getLogger("app").warning(
                "skills_create: embedding failed — antiseonymy prefilter skipped",
                extra={"event": "skills_antiseonymy_skipped"},
            )
            return None
        candidate_vec, skill_vecs = vectors[0], vectors[1:]
        candidate_norm = _l2_norm(candidate_vec)
        if candidate_norm == 0.0:
            return None
        candidate_normed = [value / candidate_norm for value in candidate_vec]
        best_index = -1
        best_cosine = -1.0
        for index, skill_vec in enumerate(skill_vecs):
            skill_norm = _l2_norm(skill_vec)
            if skill_norm == 0.0:
                continue
            cosine = sum(
                left * right
                for left, right in zip(
                    candidate_normed, [value / skill_norm for value in skill_vec]
                )
            )
            if cosine > best_cosine:
                best_cosine, best_index = cosine, index
        if best_index == -1:
            return None  # all norms are zero — nothing to compare
        nearest = rows[best_index]
        return int(nearest["id"]), str(nearest["name"]), best_cosine

    @staticmethod
    def _required_hint(field: str) -> str:
        """The hint of an empty mandatory field: the §3.8 canon table has only limits."""
        return f"skill not saved: {field} is required"

    @classmethod
    def _checked_field(cls, field: str, value: str | None, limit: int, hint: str) -> str:
        """A mandatory shape field: non-empty and ≤ the limit, else the canon hint."""
        normalized = value.strip() if isinstance(value, str) else ""
        if not normalized:
            raise SkillValidationError(cls._required_hint(field))
        if len(normalized) > limit:
            raise SkillValidationError(hint)
        return normalized

    def _checked_example(self, example: str | None) -> str | None:
        """The optional `example`: empty/absent → None, else ≤ the limit."""
        if example is None:
            return None
        normalized = str(example).strip()
        if not normalized:
            return None
        if len(normalized) > self._settings.skill_example_max_chars:
            raise SkillValidationError(HINT_EXAMPLE_LIMIT)
        return normalized

    def _checked_extra(self, extra: dict[str, str] | None) -> str | None:
        """Validate `extra` and serialize into one JSON object (§3.1).

                    Only known class keys; each field a string ≤
                    `SKILL_EXTRA_FIELD_MAX_CHARS`; the sum of all fields ≤
                    `SKILL_EXTRA_TOTAL_MAX_CHARS`; `mode` (when set) — only
                    `collaborative`|`autonomous`. An unknown key — a soft failure. An empty
                    object is stored as NULL (no class fields — no `extra`).
        """
        if extra is None:
            return None
        if not isinstance(extra, dict):
            raise SkillValidationError(HINT_EXTRA_OBJECT)
        if not extra:
            return None
        total = 0
        for key, value in extra.items():
            if key not in EXTRA_FIELDS:
                raise SkillValidationError(self._unknown_extra_hint(key))
            if not isinstance(value, str):
                raise SkillValidationError(HINT_EXTRA_OBJECT)
            if len(value) > self._settings.skill_extra_field_max_chars:
                raise SkillValidationError(HINT_EXTRA_LIMIT)
            total += len(value)
        if total > self._settings.skill_extra_total_max_chars:
            raise SkillValidationError(HINT_EXTRA_LIMIT)
        mode = extra.get("mode")
        if mode is not None and mode not in SKILL_MODES:
            raise SkillValidationError(HINT_MODE)
        return json.dumps(extra, ensure_ascii=False)

    @staticmethod
    def _unknown_extra_hint(key: str) -> str:
        """The hint of an unknown `extra` key (the known fields list — §3.1)."""
        return (
            f"skill not saved: unknown optional field «{key}»; available: "
            + ", ".join(EXTRA_FIELDS)
        )

    def _read_template(self, conn: sqlite3.Connection) -> str:
        """The `instruction_template` value from `skills_meta`.

                    No row (a DB before `init_db`/the seed) — we serve the §3.8 canon: the
                    composite always carries the "how to execute the steps" section (a
                    degradation without a failure).
        """
        row = conn.execute(
            "SELECT value FROM skills_meta WHERE key = ?",
            (INSTRUCTION_TEMPLATE_KEY,),
        ).fetchone()
        if row is None:
            return INSTRUCTION_TEMPLATE_SEED
        return str(row["value"])

    def _notify_areas_pending(self) -> None:
        """A signal to the worker: an area pending record appeared (wake at once)."""
        if self._areas_notifier is not None:
            self._areas_notifier()


def _extra_dict(raw: Any) -> dict[str, Any] | None:
    """`extra` string → dict; NULL/empty/damaged JSON → None.

            A defensive read: broken JSON (an operator's manual DB edit) does not crash
            the composite — `extra` is simply not shown (the record itself is alive).
    """
    if not raw:
        return None
    try:
        parsed = json.loads(str(raw))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) and parsed else None
