"""NamespaceService - the hierarchical namespace registry (Phase 10, REQUIREMENTS §5.7).

    The domain-rule layer over the `namespaces` table (the schema - app.storage.db).
    The description contract (the owner's decision): every node's `description` is
    required, at most 2 short sentences; the validation lives here - the single
    point for both the operator handles (REST) and the LLM description generation
    (grooming, Step 5).

    The path rules:
    - a slash path, at most 3 levels (`domain`, `domain/subdomain`,
      `domain/subdomain/section`) - deeper is a dump (§5.7);
    - each segment is a slug: latin/digits/dash, no leading/trailing dashes in a
      segment; normalization - lowercase, other characters -> dash (the classifier
      model MUST send latin - "SUBO 2020" without transliteration is not a slug,
      `subo-2020` is);
    - `default` - the system node (created in init_db), always exists; its
      description may be edited but it cannot be deleted or renamed (the REST
      handles - Step 6).

    The domain is justified by mass per topic + isolation (§5.7 "Discipline"); the
    structure judge decides on auto-node creation (Step 5), the operator edits the
    map. This service is the registry mechanics, without triggers.

    The counters (the owner's metrics): `notes_count` - the notes in the node
    itself; `subtree_count` - the whole subtree; both over the active records
    (deleted_at IS NULL). The `list_all` output - the full service-layer contract
    (REST serves it as is; the MCP layer serves a compact projection - Step 3).
"""

from __future__ import annotations

import logging
import re
import sqlite3
from typing import Any

from app.config import Settings
from app.storage.db import DEFAULT_NAMESPACE, session, transaction

MAX_DEPTH = 3
MAX_PATH_LEN = 200  # a guard against absurdly long paths (3 segments x slug <=64 + 2 slashes)
MAX_DESCRIPTION_CHARS = 500  # a rough guard against walls of text; the contract - <= 2 sentences

_SEGMENT_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+(?:\s|$)")

# The node hint on save without a namespace (No28, 3.2.0; arch §13.1):
# a significant word - >= _NODE_HINT_MIN_TOKEN chars; a match - equality
# or a shared prefix >= _STEM_MIN chars (a cheap basis of Russian morphology:
#"server"/"servers"). A path segment's weight outranks a description word's:
# the paths are latin slugs; the main signal comes from the Russian descriptions.
_NODE_HINT_MIN_TOKEN = 4
_STEM_MIN = 4
_NODE_HINT_PATH_WEIGHT = 3.0
_NODE_HINT_DESCRIPTION_WEIGHT = 2.0
_WORD_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё]+")


def _node_hint_words(text: str) -> set[str]:
    """The text's words (lowercase) of length >= _STEM_MIN - the comparison base."""
    return {word.lower() for word in _WORD_RE.findall(text) if len(word) >= _STEM_MIN}


def _node_hint_match(token: str, words: set[str]) -> bool:
    """A word matches: equality or a shared prefix >= _STEM_MIN characters."""
    head = token[:_STEM_MIN]
    return any(token == word or head == word[:_STEM_MIN] for word in words)


class NamespaceError(RuntimeError):
    """A registry operational error: the node exists / not found / a conflict."""


class NamespaceValidationError(ValueError):
    """A domain-contract violation: the path/description outside §5.7."""


def normalize_slug(value: str) -> str | None:
    """A string -> a slug segment, or None (nothing to normalize).

            Lowercase; other characters -> dash; repeated/edge dashes are trimmed.
            Cyrillic is not transliterated - the classifier must send latin.

    """
    cleaned = re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")
    if not cleaned or len(cleaned) > 64:
        return None
    return cleaned


def count_sentences(text: str) -> int:
    """The sentence count of a text (the [.!?] borders + the end of the line)."""
    return len([part for part in _SENTENCE_SPLIT_RE.split(text) if part.strip()])


class NamespaceService:
    """The hierarchical namespace registry: node CRUD, counters, subtrees."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    #--- the description contract validation (§5.7) --------------------------------

    def validate_path(self, path: str) -> str:
        """A string -> a normalized node path; a violation - NamespaceValidationError.

                    "SUBO 2020" -> '2020' (Cyrillic is not a slug); 'Work/SBOS 2020' ->
                    'work/sbos-2020'; a depth > 3, empty segments, the length, nesting
                    under default - errors.

        """
        if not path or not path.strip():
            raise NamespaceValidationError("path: path cannot be empty")
        if len(path) > MAX_PATH_LEN:
            raise NamespaceValidationError(
                f"path: length must be ≤{MAX_PATH_LEN}, got {len(path)}"
            )
        raw_segments = path.strip().strip("/").split("/")
        segments: list[str] = []
        for segment in raw_segments:
            slug = normalize_slug(segment)
            if slug is None:
                raise NamespaceValidationError(
                    f"path: segment «{segment}» does not normalize to a slug "
                    "(latin/digits/hyphen)"
                )
            segments.append(slug)
        if not 1 <= len(segments) <= MAX_DEPTH:
            raise NamespaceValidationError(
                f"path: expected 1..{MAX_DEPTH} levels, got {len(segments)}"
            )
        if len(segments) > 1 and segments[0] == "default":
            raise NamespaceValidationError(
                "path: default — system node, nesting is forbidden"
            )
        return "/".join(segments)

    def validate_description(self, description: str) -> str:
        """A node description: non-empty, >= 1 meaningful sentence, <= 2 sentences
                    (the owner's contract + the lsb-0005-08 audit).

                    The single validation point of the description contract: BOTH auto-node
                    creation paths go through it - the model (memory_namespace_create ->
                    create confirmed) and the judge (auto-promotion -> create
                    provisional). The content check "not garbage": at least one
                    letter/digit must be present - pure punctuation ("...", "!?!") does
                    not pass; the semantic meaningfulness is the judge's gate (the auto
                    path) / the operator's (the operator handles), not the registry
                    mechanics.

        """
        if not description or not description.strip():
            raise NamespaceValidationError("description: cannot be empty")
        text = " ".join(description.split())
        if not any(ch.isalnum() for ch in text):
            raise NamespaceValidationError(
                "description: garbage — no letters or digits"
            )
        sentences = count_sentences(text)
        if sentences > 2:
            raise NamespaceValidationError(
                f"description: no more than 2 sentences, got {sentences}"
            )
        return text

    #--- reads ---------------------------------------------------------------

    def exists(self, path: str) -> bool:
        """Whether the node is registered (the save/search validation - Step 3)."""
        normalized = self.validate_path(path)
        with session(self._settings) as conn:
            return (
                conn.execute(
                    "SELECT 1 FROM namespaces WHERE path = ?", (normalized,)
                ).fetchone()
                is not None
            )

    def validate_placement(self, namespace: str) -> str:
        """The write's target node (save/update): the normalized path of a registered
                    node; unregistered - NamespaceError (the Step 3 transport wraps it
                    into fail + hint). "default" always exists, so a save without a node
                    always passes.
        """
        normalized = self.validate_path(namespace)
        if not self.exists(normalized):
            raise NamespaceError(
                f"namespace «{namespace}» is not registered; create the missing "
                "domains via memory_namespace_create with a description from the purpose; "
                "up-to-date map — memory_namespaces"
            )
        return normalized

    def get(self, path: str) -> dict[str, Any] | None:
        """A registry node with counters, or None (no node)."""
        normalized = self.validate_path(path)
        with session(self._settings) as conn:
            row = conn.execute(
                self._select_node() + " WHERE np.path = ?", (normalized,)
            ).fetchone()
        return self._node_dict(row) if row is not None else None

    def list_all(self) -> dict[str, Any]:
        """The full registry: nodes + counters (the memory_namespaces contract, Step 3).

                    notes_count - the notes in the node itself; subtree_count - the whole
                    subtree; both over the active notes. The path sort - a deterministic
                    map.

        """
        with session(self._settings) as conn:
            rows = conn.execute(self._select_node() + " ORDER BY np.path").fetchall()
        return {"namespaces": [self._node_dict(row) for row in rows]}

    def subtree_nodes(self, path: str) -> list[str]:
        """A subtree's nodes: the node itself + the leaves under it (for the KNN partition filter).

                    "domain" -> ['domain', 'domain/a', 'domain/b']; a leaf ->
                    ['domain/leaf']. Only REGISTERED nodes are known: notes in paths
                    outside the registry (impossible in a routine save) belong to no
                    global search.

        """
        normalized = self.validate_path(path)
        with session(self._settings) as conn:
            return [
                row[0]
                for row in conn.execute(
                    "SELECT path FROM namespaces "
                    "WHERE path = ? OR path LIKE ? || '/%' ORDER BY path",
                    (normalized, normalized),
                )
            ]

    def filter_nodes(self, namespace: str | None, exact: bool) -> list[str] | None:
        """The partition nodes for the output filter (§5.7): None - global;
                    an exact node (`exact`) - only it; otherwise - the subtree (the node +
                    the leaves). An unregistered node - NamespaceError (the Step 3
                    transport wraps it into fail + hint). The shared point of search and
                    list.
        """
        if namespace is None:
            return None
        normalized = self.validate_path(namespace)
        if not self.exists(normalized):
            raise NamespaceError(
                f"namespace «{namespace}» is not registered; create the missing "
                "domains via memory_namespace_create with a description from the purpose; "
                "up-to-date map — memory_namespaces"
            )
        if exact:
            return [normalized]
        return self.subtree_nodes(normalized)

    def suggest(
        self, terms: str, *, limit: int = 2, min_score: float = 2.0
    ) -> list[dict[str, Any]]:
        """The map's candidate nodes for a note's words (the No28 hint, arch §13.1).

                    The significant words of `terms` (>= _NODE_HINT_MIN_TOKEN) are compared
                    with the path segments (weight 3.0) and the description words (weight
                    2.0); a match - equality or a shared prefix >= _STEM_MIN
                    ("server"/"servers"). A node's score - the sum over the matched words
                    (a word counts once, at most its weight); the return - the top by
                    score descending (the tie-break - path), at most `limit`, only
                    score >= min_score. This is a MAP READ (one SELECT without counters):
                    no model, no embedder, no writes; a read failure - an empty list, not
                    an exception (the hint has no right to fail a save).

        """
        tokens = list(dict.fromkeys(_node_hint_words(terms)))
        if not tokens:
            return []
        try:
            with session(self._settings) as conn:
                rows = conn.execute(
                    "SELECT path, description FROM namespaces"
                ).fetchall()
        except Exception:  # noqa: BLE001 - the hint never fails a save
            logging.getLogger("app").warning(
                "node hint map unavailable",
                extra={"event": "node_hint_unavailable"},
            )
            return []
        candidates: list[dict[str, Any]] = []
        for row in rows:
            path_words = {segment.lower() for segment in row["path"].split("/")}
            description_words = _node_hint_words(row["description"])
            score = 0.0
            for token in tokens:
                if _node_hint_match(token, path_words):
                    score += _NODE_HINT_PATH_WEIGHT
                elif _node_hint_match(token, description_words):
                    score += _NODE_HINT_DESCRIPTION_WEIGHT
            if score >= min_score:
                candidates.append(
                    {
                        "path": row["path"],
                        "description": row["description"],
                        "score": score,
                    }
                )
        candidates.sort(key=lambda node: (-node["score"], node["path"]))
        return candidates[:limit]

    #--- writes ---------------------------------------------------------------

    def create(
        self,
        path: str,
        description: str,
        status: str = "confirmed",
    ) -> dict[str, Any]:
        """Register a node; the node already exists - NamespaceError.

                    Called by: the operator REST handles (Step 6, confirmed) and the
                    structure judge's auto-creation (Step 5, provisional). The parent
                    (`domain`) must exist - the hierarchy does not create roots by itself
                    (§5.7).

        """
        normalized = self.validate_path(path)
        text = self.validate_description(description)
        if status not in ("confirmed", "provisional"):
            raise NamespaceValidationError(
                f"status: expected confirmed|provisional, got {status}"
            )
        segments = normalized.split("/")
        with session(self._settings) as conn:
            if len(segments) > 1:
                parent_path = "/".join(segments[:-1])
                parent = conn.execute(
                    "SELECT 1 FROM namespaces WHERE path = ?", (parent_path,)
                ).fetchone()
                if parent is None:
                    raise NamespaceError(
                        f"parent node «{parent_path}» is not registered — "
                        "create it first"
                    )
            try:
                with transaction(conn):
                    conn.execute(
                        "INSERT INTO namespaces (path, description, status) "
                        "VALUES (?, ?, ?)",
                        (normalized, text, status),
                    )
            except sqlite3.IntegrityError as exc:
                raise NamespaceError(
                    f"node «{normalized}» is already registered"
                ) from exc
        return self.get(normalized)  # type: ignore[return-value]

    def update_description(self, path: str, description: str) -> dict[str, Any] | None:
        """Edit a node's description (the operator/confirmation); None - no node."""
        normalized = self.validate_path(path)
        text = self.validate_description(description)
        with session(self._settings) as conn, transaction(conn):
            cursor = conn.execute(
                "UPDATE namespaces SET description = ?, "
                "updated_at = strftime('%Y-%m-%dT%H:%M:%SZ','now') WHERE path = ?",
                (text, normalized),
            )
            updated = cursor.rowcount
        if not updated:
            return None
        return self.get(normalized)  # type: ignore[return-value]

    def set_status(self, path: str, status: str) -> dict[str, Any] | None:
        """Change the node's status (confirm a provisional by the operator's audit)."""
        if status not in ("confirmed", "provisional"):
            raise NamespaceValidationError(
                f"status: expected confirmed|provisional, got {status}"
            )
        normalized = self.validate_path(path)
        with session(self._settings) as conn, transaction(conn):
            cursor = conn.execute(
                "UPDATE namespaces SET status = ?, "
                "updated_at = strftime('%Y-%m-%dT%H:%M:%SZ','now') WHERE path = ?",
                (status, normalized),
            )
            updated = cursor.rowcount
        if not updated:
            return None
        return self.get(normalized)  # type: ignore[return-value]

    #--- the operator moves (Phase 10, Step 6; §5.7 "structural handles") ---

    def groom(self) -> dict[str, Any]:
        """The registry grooming after the moves (§5.7): the empty is cleaned, the small is signaled.

                    Only EMPTY provisional leaves are deleted automatically (the system
                    created them - the system cleans them; the operator's confirmed nodes
                    are never removed automatically - the operator may have created a node
                    for future content). A leaf with < NAMESPACE_GROOM_MIN_NOTES notes - a
                    merge candidate: untouched, a signal in the logs and the report (the
                    operator's merge handle). default and the roots are never touched
                    automatically; empty confirmed nodes - the empty_confirmed signal (the
                    logs). The return - the report:
                    {deleted, merge_candidates, empty_confirmed}.

        """
        logger = logging.getLogger("app")
        nodes = self.list_all()["namespaces"]
        report: dict[str, Any] = {
            "deleted": [],
            "merge_candidates": [],
            "empty_confirmed": [],
        }
        min_notes = self._settings.namespace_groom_min_notes
        for node in nodes:
            path = node["path"]
            if path == DEFAULT_NAMESPACE:
                continue  # the system swap lives forever
            is_leaf = "/" in path
            if (
                is_leaf
                and node["status"] == "provisional"
                and node["notes_count"] == 0
            ):
                with session(self._settings) as conn, transaction(conn):
                    conn.execute("DELETE FROM namespaces WHERE path = ?", (path,))
                report["deleted"].append(path)
                logger.info(
                    "groom: empty provisional leaf deleted",
                    extra={
                        "event": "node_deleted",
                        "path": path,
                        "reason": "groom_empty",
                    },
                )
                continue
            if node["subtree_count"] == 0:
                report["empty_confirmed"].append(path)
                logger.warning(
                    "groom: empty node after relocations — operator decision",
                    extra={
                        "event": "groom_empty_confirmed",
                        "path": path,
                        "status": node["status"],
                    },
                )
                continue
            if is_leaf and 0 < node["notes_count"] < min_notes:
                report["merge_candidates"].append(path)
                logger.info(
                    "groom: leaf below groom minimum — merge candidate",
                    extra={
                        "event": "groom_candidate",
                        "path": path,
                        "notes_count": node["notes_count"],
                    },
                )
        return report

    def rename(self, old: str, new: str) -> dict[str, Any]:
        """Rename a node (a leaf or a root with children) with the moves (§5.7).

                    What moves: the registry paths (the node + the children), the notes'
                    namespace in the subtree (vector_status='pending' - re-encoding into
                    the new partition), the default notes' placement (hint_path - the full
                    placement path, a prefix replacement as for promotions: a root - all
                    descendants, a leaf - depth 2/3) and the promotions verdicts (hint_path
                    and canonical_path). Nothing is lost. default is never renamed; the
                    new path must be free.

        """
        old_path = self.validate_path(old)
        new_path = self.validate_path(new)
        if old_path == DEFAULT_NAMESPACE or new_path == DEFAULT_NAMESPACE:
            raise NamespaceError("default — system node, renaming is forbidden")
        if old_path == new_path:
            return self.get(old_path)  # type: ignore[return-value]
        if not self.exists(old_path):
            raise NamespaceError(f"node «{old_path}» is not registered")
        if self.exists(new_path):
            raise NamespaceError(f"node «{new_path}» is already registered")
        old_nodes = self.subtree_nodes(old_path)
        with session(self._settings) as conn:
            try:
                with transaction(conn):
                    # The registry: a node's and all its children's paths (few nodes - one query each).
                    for node_path in old_nodes:
                        conn.execute(
                            "UPDATE namespaces SET path = ?, "
                            "updated_at = strftime('%Y-%m-%dT%H:%M:%SZ','now') WHERE path = ?",
                            (new_path + node_path[len(old_path):], node_path),
                        )
                    # The subtree's notes: the namespace + re-encoding into the new partition.
                    for node_path in old_nodes:
                        conn.execute(
                            "UPDATE notes SET namespace = ?, vector_status = 'pending' "
                            "WHERE namespace = ? AND deleted_at IS NULL",
                            (new_path + node_path[len(old_path):], node_path),
                        )
                    # The default notes' placement: hint_path (the full placement path,
                    # lsb-0005-02) - a prefix replacement of the old path with the new one,
                    # as for promotions: a root - all descendants, a leaf - depth 2/3
                    # (grooming references the node's path).
                    conn.execute(
                        "UPDATE notes SET hint_path = ? || substr(hint_path, ?) "
                        "WHERE namespace = 'default' "
                        "AND (hint_path = ? OR hint_path LIKE ? || '/%')",
                        (new_path, len(old_path) + 1, old_path, old_path),
                    )
                    # The trigger verdicts: the single hint_path (lsb-0005-03) -
                    # a prefix replacement as for canonical_path (a root or a leaf
                    # with children - only the prefix changes in every subpath).
                    conn.execute(
                        "UPDATE promotions SET hint_path = ? || substr(hint_path, ?) "
                        "WHERE hint_path = ? OR hint_path LIKE ? || '/%'",
                        (new_path, len(old_path) + 1, old_path, old_path),
                    )
                    conn.execute(
                        "UPDATE promotions SET canonical_path = ? WHERE canonical_path = ?",
                        (new_path, old_path),
                    )
                    conn.execute(
                        "UPDATE promotions SET canonical_path = ? || substr(canonical_path, ?) "
                        "WHERE canonical_path LIKE ? || '/%'",
                        (new_path, len(old_path) + 1, old_path),
                    )
            except sqlite3.IntegrityError as exc:
                raise NamespaceError(
                    f"node «{new_path}» is already registered"
                ) from exc
        logging.getLogger("app").info(
            "namespace renamed",
            extra={
                "event": "node_renamed",
                "old": old_path,
                "new": new_path,
                "nodes": len(old_nodes),
            },
        )
        return self.get(new_path)  # type: ignore[return-value]

    def merge_node(self, path: str, into: str) -> dict[str, Any]:
        """Merge a LEAF into an existing node: the notes moved, the node is gone (§5.7).

                    The hint_path canonicalization: into a leaf - the full target path;
                    into a root or default - NULL ("general"). Nothing is lost: the notes
                    move whole (vector_status='pending' - re-encoding into the new
                    partition). The promotions verdicts follow the move (the source's
                    canonical_path and hint_path key -> the target). A root merge is not
                    supported (the children would be left without a parent - the operator
                    sorts the subtree out leaf by leaf); default never merges. The return
                    - {path, into, moved}.

        """
        source = self.validate_path(path)
        target = self.validate_path(into)
        if source == DEFAULT_NAMESPACE:
            raise NamespaceError("default — system node, merging is forbidden")
        if source == target:
            raise NamespaceError("a node cannot be merged with itself")
        if "/" not in source:
            raise NamespaceError(
                "only a leaf is merged; a root with children — break down by leaves"
            )
        if not self.exists(source):
            raise NamespaceError(f"node «{source}» is not registered")
        if not self.exists(target):
            raise NamespaceError(f"node «{target}» is not registered")
        target_hint = target if "/" in target else None  # a leaf -> the target's path; a root/default -> NULL
        with session(self._settings) as conn, transaction(conn):
            cursor = conn.execute(
                "UPDATE notes SET namespace = ?, hint_path = ?, "
                "vector_status = 'pending' "
                "WHERE namespace = ? AND deleted_at IS NULL",
                (target, target_hint, source),
            )
            moved = cursor.rowcount
            conn.execute("DELETE FROM namespaces WHERE path = ?", (source,))
            # The trigger verdicts: the source was merged, its content went to the target -
            # canonical_path and the hint_path key itself follow the move
            # (if the target already has its own verdict - it dominates; the INSERT
            # OR IGNORE keeps it; the source's key is dropped).
            conn.execute(
                "UPDATE promotions SET canonical_path = ? WHERE canonical_path = ?",
                (target, source),
            )
            conn.execute(
                "INSERT OR IGNORE INTO promotions "
                "(hint_path, status, canonical_path, decided_at) "
                "SELECT ?, status, canonical_path, decided_at "
                "FROM promotions WHERE hint_path = ?",
                (target, source),
            )
            conn.execute("DELETE FROM promotions WHERE hint_path = ?", (source,))
        logging.getLogger("app").info(
            "namespace merged",
            extra={
                "event": "node_merged",
                "canonical": target,
                "hint": source,
                "moved": moved,
                "by": "operator",
            },
        )
        return {"path": source, "into": target, "moved": moved}

    def delete_node(self, path: str) -> dict[str, Any]:
        """Delete a node with the note moves (§5.7, nothing is lost).

                    A leaf: the notes -> the parent root (hint_path=NULL - the domain's
                    general), vector_status='pending'; the promotions verdicts follow the
                    move (the leaf's canonical_path and hint_path key -> the parent). An
                    empty root is deleted; a root with children - NamespaceError (sort the
                    subtree out leaf by leaf first). default is never deleted. The return
                    - {path, moved}.

        """
        normalized = self.validate_path(path)
        if normalized == DEFAULT_NAMESPACE:
            raise NamespaceError("default — system node, deletion is forbidden")
        if not self.exists(normalized):
            raise NamespaceError(f"node «{normalized}» is not registered")
        with session(self._settings) as conn, transaction(conn):
            children = conn.execute(
                "SELECT COUNT(*) FROM namespaces WHERE path LIKE ? || '/%'",
                (normalized,),
            ).fetchone()[0]
            if children:
                raise NamespaceError(
                    f"node «{normalized}» has children ({children}) — "
                    "break down the subtree by leaves"
                )
            moved = 0
            if "/" in normalized:
                domain = normalized.split("/", 1)[0]
                cursor = conn.execute(
                    "UPDATE notes SET namespace = ?, hint_path = NULL, "
                    "vector_status = 'pending' "
                    "WHERE namespace = ? AND deleted_at IS NULL",
                    (domain, normalized),
                )
                moved = cursor.rowcount
                # The trigger verdicts: the leaf is deleted, its content went to the parent
                # root - canonical_path and the hint_path key follow the move
                # (if the parent already has a verdict - it dominates).
                conn.execute(
                    "UPDATE promotions SET canonical_path = ? WHERE canonical_path = ?",
                    (domain, normalized),
                )
                conn.execute(
                    "INSERT OR IGNORE INTO promotions "
                    "(hint_path, status, canonical_path, decided_at) "
                    "SELECT ?, status, canonical_path, decided_at "
                    "FROM promotions WHERE hint_path = ?",
                    (domain, normalized),
                )
                conn.execute(
                    "DELETE FROM promotions WHERE hint_path = ?", (normalized,)
                )
            conn.execute("DELETE FROM namespaces WHERE path = ?", (normalized,))
        logging.getLogger("app").info(
            "namespace deleted",
            extra={
                "event": "node_deleted",
                "path": normalized,
                "moved": moved,
                "by": "operator",
            },
        )
        return {"path": normalized, "moved": moved}

    #--- internal ------------------------------------------------------------

    @staticmethod
    def _select_node() -> str:
        """A registry node + the counters (active notes), in one query."""
        return (
            "SELECT np.path, np.description, np.status, np.created_at, np.updated_at, "
            "(SELECT COUNT(*) FROM notes n "
            " WHERE n.namespace = np.path AND n.deleted_at IS NULL) AS notes_count, "
            "(SELECT COUNT(*) FROM notes n WHERE (n.namespace = np.path "
            " OR n.namespace LIKE np.path || '/%') "
            " AND n.deleted_at IS NULL) AS subtree_count "
            "FROM namespaces np"
        )

    @staticmethod
    def _node_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "path": row["path"],
            "description": row["description"],
            "status": row["status"],
            "notes_count": int(row["notes_count"]),
            "subtree_count": int(row["subtree_count"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }