"""The `terms-normalize` job (No32, 3.2.0; arch §6.6/§7.4/§13.6/§13.10-13.12).

    The background job finds CLUSTERS of one term's contexts that are about the
    same thing and brings the wordings to a canonical one. It runs entirely
    WITHOUT an LLM: the clustering - key normalization + its own light stemmer of
    Russian word forms (`token_stem`) + the area's existing trigram metric
    (`areas.trigram_similarity`); no external morphology (`pymorphy2`) is
    introduced (§13.6).

    Key invariants:

    * **Auto - only on exact definition equality** (FR-3.1/FR-3.3): a cluster is
      normalized automatically only when `normalize_key(definition)` is equal for
      ALL of its rows; otherwise - the `term_context_cluster` warning to a human,
      no rename.
    * **The canon - by the data, deterministically** (FR-2.3): the maximum `uses`
      (the count of the area's live rows with this `context_norm`, as in
      `TermsService._contexts`), then the latest `MAX(id)`, then the shorter one,
      then the lexicographically smaller. A repeated run makes the same choice -
      a context does not "rename back and forth".
    * **A key conflict - without data loss** (FR-4.1/FR-4.2): when the canonical
      key is taken by a live record, the record with the smaller id survives and
      the loser goes to a soft delete (the data lives in trash). The history
      (No41) records both the rename (`source='job:terms-normalize'`) and the
      deletion.
    * **The excess flag** (FR-5.1/FR-5.2): `term_contexts_excess` by the number of
      DIFFERENT live `context_norm` >= `JOB_TERMS_CONTEXTS_EXCESS`,
      `payload.contexts` - the counter; cleared when the context count drops
      below the threshold.
    * **Idempotency** (§13.11): the state is the data itself, no new column. After
      an auto-normalization a group has one `context_norm` -> the selection skips
      it; the warnings - an upsert over the unique open index (no12-01).
    * **The selection - SQL only** (`HAVING COUNT(DISTINCT context_norm) > 1` or
      `>= JOB_TERMS_CONTEXTS_EXCESS`), the order by `term_norm`, the
      `JOB_TERMS_BATCH` limit; no models called.

    The job registers on the lsb-0014 framework by `build_terms_normalize_job` and
    a line in `jobs.JOB_BUILDERS` (a one-way import: terms_normalize -> jobs, no
    cycle). It additionally obeys `AREA_TERMS_ENABLED` (§6.6): a disabled area is
    not processed in the background.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from app.config import Settings
from app.services import history, jobs
from app.services.areas import normalize_key, trigram_similarity
from app.services.jobs import JobSpec, queue_snapshot
from app.services.terms import TermKeyConflictError, TermsService
from app.services.warnings import (
    KIND_TERM_CONTEXTS_EXCESS,
    KIND_TERM_CONTEXT_CLUSTER,
    WarningsService,
)
from app.storage.db import session

if TYPE_CHECKING:  # the service has no runtime dependencies (DI)
    from app.services.worker import BackgroundWorker

# The job name in the journal and the queue for `/health.queues`.
TERMS_NORMALIZE_JOB = "terms-normalize"

# The history versions' source (No41, §13.4): the job bodies carry `job:<name>`.
_ORIGIN = f"job:{TERMS_NORMALIZE_JOB}"

# A curated list of frequent Russian inflection suffixes (§13.6):
# the longest match; the remainder must stay >= 4 characters. The order -
# by DESCENDING length: `token_stem` takes the first fitting (the longest).
STEM_SUFFIXES: tuple[str, ...] = (
    # 4 characters
    "иями",
    # 3 characters
    "ами",
    "ями",
    "ыми",
    "ими",
    "иях",
    # 2 characters
    "ов",
    "ев",
    "ей",
    "ий",
    "ый",
    "ой",
    "ая",
    "ое",
    "ые",
    "ия",
    "ии",
    "ью",
    # 1 character
    "а",
    "я",
    "у",
    "ю",
    "и",
    "е",
    "ы",
    "о",
    "ь",
)

# Depersonalized English warning texts (the §6.5/§7.1 style).
MESSAGE_CLUSTER_FLAGGED = (
    "term '{term_norm}' has {count} similar contexts with different "
    "definitions — review them by hand"
)
MESSAGE_CONTEXTS_EXCESS = (
    "term '{term_norm}' has {contexts} different contexts — consider "
    "reviewing its terminology"
)

# The minimum remainder length after a suffix cut (§13.6): short words are
# not cut - "house"/"houses" stay distinct.
_MIN_STEM = 4


def token_stem(token: str) -> str:
    """A word's stem: the key normalization + cutting the longest fitting suffix.

            `normalize_key` (lower -> trim -> collapsed spaces -> e -> e) -> the first
            (longest) suffix from `STEM_SUFFIXES` when the remainder is >= 4
            characters; otherwise - the word as is. Deterministic (no external
            morphology, §13.6).

    """
    normalized = normalize_key(token)
    for suffix in STEM_SUFFIXES:
        if normalized.endswith(suffix):
            stem = normalized[: len(normalized) - len(suffix)]
            if len(stem) >= _MIN_STEM:
                return stem
    return normalized


def _stems(text: str) -> set[str]:
    """The stem set of a text's significant words (over the normalized tokens)."""
    return {token_stem(token) for token in normalize_key(text).split() if token}


def stem_overlap(left: str, right: str) -> float:
    """The shared-stem share: |A intersect B| / min(|A|, |B|); empty - 0.0."""
    left_stems = _stems(left)
    right_stems = _stems(right)
    if not left_stems or not right_stems:
        return 0.0
    return len(left_stems & right_stems) / min(len(left_stems), len(right_stems))


def context_similarity(left: str, right: str) -> float:
    """The context closeness: max(trigrams, the shared-stem share).

            The area metric is trigram-based (`trigram_similarity`), plus morphology:
            "project"/"project's" without a stemmer never converge by trigrams.
            Computed without an embedding (§6.6).

    """
    return max(trigram_similarity(left, right), stem_overlap(left, right))


class TermsNormalizerService:
    """The context clusters, the canon and the key merges (arch §6.6).

            DI: `settings` (the clustering threshold, the excess threshold), `terms`
            (the reading helper `group` + the existing `update`/`delete`), `warnings`
            (the shared warnings surface). The area keeps no connections of its own to
            the domain: the plan and the application go through `TermsService`.

    """

    def __init__(
        self,
        settings: Settings,
        *,
        terms: TermsService,
        warnings: WarningsService,
    ) -> None:
        self._settings = settings
        self._terms = terms
        self._warnings = warnings

    #--- the plan (writes nothing) ---------------------------------------------

    def plan(self, term_norm: str) -> dict[str, Any]:
        """A group's clusters + the canonical form + the verdict; the DB is untouched.

                    Returns: `{term_norm, contexts, excess, clusters}` where a cluster is
                    `{context_norm, canonical, canonical_id, members, verdict, reason}`,
                    and `members` are the cluster's live rows. `verdict` - `auto` (the
                    definitions matched, or a size-1 cluster) / `flag` (the definitions
                    differ).

        """
        rows = self._terms.group(term_norm)
        contexts = len({row["context_norm"] for row in rows})
        threshold = int(self._settings.job_terms_contexts_excess)
        clusters = self._build_clusters(term_norm, rows)
        return {
            "term_norm": term_norm,
            "contexts": contexts,
            "excess": contexts >= threshold,
            "clusters": clusters,
        }

    def _build_clusters(
        self, term_norm: str, rows: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Single-linkage clusters over `context_similarity` + the canon and the verdict."""
        threshold = float(self._settings.job_terms_cluster_similarity)
        components = _connected_components(rows, threshold)
        uses = self._uses([row["context_norm"] for row in rows])
        clusters: list[dict[str, Any]] = []
        for members in components:
            canonical = _canonical(members, uses)
            definitions = {normalize_key(member["definition"]) for member in members}
            if len(members) == 1:
                verdict, reason = "auto", "single_member"
            elif len(definitions) == 1:
                verdict, reason = "auto", "definitions_equal"
            else:
                verdict, reason = "flag", "definitions_differ"
            clusters.append(
                {
                    "context_norm": canonical["context_norm"],
                    "canonical": canonical["context"],
                    "canonical_id": canonical["id"],
                    "members": members,
                    "verdict": verdict,
                    "reason": reason,
                }
            )
        clusters.sort(key=lambda cluster: cluster["members"][0]["id"])
        return clusters

    def _uses(self, context_norms: list[str]) -> dict[str, int]:
        """The `context_norm` frequency over the area's live rows (as in `_contexts`)."""
        unique = sorted(set(context_norms))
        if not unique:
            return {}
        placeholders = ", ".join("?" for _ in unique)
        with session(self._settings) as conn:
            rows = conn.execute(
                "SELECT context_norm, COUNT(*) AS uses FROM terms "
                "WHERE deleted_at IS NULL AND context_norm IN ("
                + placeholders
                + ") GROUP BY context_norm",
                unique,
            ).fetchall()
        return {str(row["context_norm"]): int(row["uses"]) for row in rows}

    #--- applying the plan ----------------------------------------------------

    def normalize(self, term_norm: str) -> dict[str, Any]:
        """Apply the plan: an auto-rename/merge or a warning.

                    Returns the counters: `{term_norm, contexts, clusters, renamed,
                    merged, flagged, excess}`. Idempotent: a repeated run over normalized
                    data is a no-op (§13.11).

        """
        plan = self.plan(term_norm)
        renamed = 0
        merged = 0
        flagged = 0
        groups = [cluster for cluster in plan["clusters"] if len(cluster["members"]) > 1]
        with history.bind_origin(_ORIGIN):
            for cluster in groups:
                if cluster["verdict"] == "flag":
                    self._flag_cluster(plan, cluster)
                    flagged += 1
                    continue
                renamed, merged = self._apply_cluster(
                    term_norm, cluster, renamed, merged
                )
            excess = self._apply_excess(term_norm)
        return {
            "term_norm": term_norm,
            "contexts": self._context_count(term_norm),
            "clusters": len(plan["clusters"]),
            "renamed": renamed,
            "merged": merged,
            "flagged": flagged,
            "excess": excess,
        }

    def _apply_cluster(
        self,
        term_norm: str,
        cluster: dict[str, Any],
        renamed: int,
        merged: int,
    ) -> tuple[int, int]:
        """Rename the cluster's non-canonical rows; return (renamed, merged)."""
        canonical = cluster["canonical"]
        canon_norm = cluster["context_norm"]
        canonical_id = cluster["canonical_id"]
        for member in sorted(cluster["members"], key=lambda item: item["id"]):
            if member["id"] == canonical_id:
                continue
            try:
                self._terms.update(
                    member["id"], member["term"], canonical, member["definition"]
                )
                renamed += 1
                self._log(
                    "terms-normalize: context renamed",
                    "term_context_renamed",
                    term_id=member["id"],
                    from_=member["context"],
                    to=canonical,
                )
            except TermKeyConflictError:
                renamed, merged = self._merge_conflict(
                    term_norm, canon_norm, canonical, member, renamed, merged
                )
        return renamed, merged

    def _merge_conflict(
        self,
        term_norm: str,
        canon_norm: str,
        canonical: str,
        member: dict[str, Any],
        renamed: int,
        merged: int,
    ) -> tuple[int, int]:
        """Resolve a key conflict: the smaller id survives, the loser goes to trash."""
        occupant = self._occupant(term_norm, canon_norm)
        if occupant is None:  # no race: the key is freed - just rename
            self._terms.update(
                member["id"], member["term"], canonical, member["definition"]
            )
            renamed += 1
            self._log(
                "terms-normalize: context renamed",
                "term_context_renamed",
                term_id=member["id"],
                from_=member["context"],
                to=canonical,
            )
            return renamed, merged
        if member["id"] < occupant:
            survivor, loser = member["id"], occupant
            self._terms.delete(loser)
            self._terms.update(
                member["id"], member["term"], canonical, member["definition"]
            )
        else:
            survivor, loser = occupant, member["id"]
            self._terms.delete(loser)
        merged += 1
        self._log(
            "terms-normalize: context merged",
            "term_context_merged",
            term_id=survivor,
            into_id=loser,
            from_=member["context"],
            to=canonical,
        )
        return renamed, merged

    def _flag_cluster(self, plan: dict[str, Any], cluster: dict[str, Any]) -> None:
        """Different definitions - a warning to a human, no rename."""
        contexts = [member["context"] for member in cluster["members"]]
        subject_id = min(member["id"] for member in cluster["members"])
        self._warnings.open(
            KIND_TERM_CONTEXT_CLUSTER,
            "term",
            subject_id,
            MESSAGE_CLUSTER_FLAGGED.format(
                term_norm=plan["term_norm"], count=len(contexts)
            ),
            {"contexts": contexts, "reason": "definitions_differ"},
        )
        self._log(
            "terms-normalize: cluster flagged",
            "term_cluster_flagged",
            term_norm=plan["term_norm"],
            contexts=contexts,
            reason="definitions_differ",
        )

    def _apply_excess(self, term_norm: str) -> bool:
        """Open/clear `term_contexts_excess` by the distinct-context count."""
        contexts = self._context_count(term_norm)
        subject_id = self._smallest_id(term_norm)
        if subject_id is None:
            return False
        if contexts >= int(self._settings.job_terms_contexts_excess):
            self._warnings.open(
                KIND_TERM_CONTEXTS_EXCESS,
                "term",
                subject_id,
                MESSAGE_CONTEXTS_EXCESS.format(
                    term_norm=term_norm, contexts=contexts
                ),
                {"contexts": contexts},
            )
            return True
        self._warnings.resolve(KIND_TERM_CONTEXTS_EXCESS, "term", subject_id)
        return False

    #--- the job run --------------------------------------------------------

    def run(self, batch: int | None = None) -> int:
        """A batch of groups (the `JOB_TERMS_BATCH` limit, the order by `term_norm`).

                    Returns the number of groups where the data was ACTUALLY edited (a
                    rename/merge of contexts). The `terms_normalize_run` event carries the
                    run's aggregates (FR-8.1), including the drained group count
                    (`terms`).

                    The return feeds the job loop (`run_loop`: `processed > 0` - keep
                    draining immediately, no interval wait). A group where an edit is
                    impossible (two contexts diverge in definitions - only a warning) or
                    where only idempotent warnings are needed does NOT count as progress:
                    otherwise the loop spins idle (30 runs a second in the test contour,
                    lsbdef-0011), because the group selection (`contexts > 1`) by itself
                    does not guarantee there is anything to change.

        """
        limit = int(self._settings.job_terms_batch if batch is None else batch)
        rows = self._selected_groups(limit)
        clusters = renamed = merged = flagged = excess = progressed = 0
        for row in rows:
            result = self.normalize(str(row["term_norm"]))
            clusters += int(result["clusters"])
            renamed += int(result["renamed"])
            merged += int(result["merged"])
            flagged += int(result["flagged"])
            excess += 1 if result["excess"] else 0
            if int(result["renamed"]) or int(result["merged"]):
                progressed += 1
        self._log(
            "terms-normalize: run finished",
            "terms_normalize_run",
            terms=len(rows),
            clusters=clusters,
            renamed=renamed,
            merged=merged,
            flagged=flagged,
            excess=excess,
            progressed=progressed,
        )
        return progressed

    def queue_stat(self) -> dict:
        """The `terms-normalize` queue snapshot - SQL only, no models (§6.6)."""
        if not self._active():
            return queue_snapshot(0, None)
        with session(self._settings) as conn:
            rows = conn.execute(
                "SELECT COUNT(DISTINCT context_norm) AS contexts, "
                "MAX(CAST(strftime('%s','now') AS INTEGER) - "
                "CAST(strftime('%s', updated_at) AS INTEGER)) AS age "
                "FROM terms WHERE deleted_at IS NULL GROUP BY term_norm"
            ).fetchall()
        threshold = int(self._settings.job_terms_contexts_excess)
        pending = 0
        oldest: int | None = None
        for row in rows:
            contexts = int(row["contexts"])
            if contexts > 1 or contexts >= threshold:
                pending += 1
                age = row["age"]
                if age is not None:
                    oldest = max(int(age), oldest or 0)
        return queue_snapshot(pending, oldest)

    def queue_empty(self) -> bool:
        """Whether the `terms-normalize` queue is empty (a cheap recheck, pool 6)."""
        return int(self.queue_stat()["pending"] or 0) == 0

    #--- the selection and small reads (SQL, no models) ---------------------------

    def _active(self) -> bool:
        """The job is enabled and the terms area is enabled (No40, §6.6)."""
        return bool(
            self._settings.job_terms_enabled
            and self._settings.area_terms_enabled
        )

    def _selected_groups(self, limit: int | None = None) -> list[sqlite3.Row]:
        """The live groups matching the selection (SQL, §6.6)."""
        sql = (
            "SELECT term_norm FROM terms WHERE deleted_at IS NULL "
            "GROUP BY term_norm HAVING COUNT(DISTINCT context_norm) > 1 "
            "OR COUNT(DISTINCT context_norm) >= ? ORDER BY term_norm"
        )
        params: list[object] = [int(self._settings.job_terms_contexts_excess)]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        with session(self._settings) as conn:
            return list(conn.execute(sql, params).fetchall())

    def _context_count(self, term_norm: str) -> int:
        with session(self._settings) as conn:
            row = conn.execute(
                "SELECT COUNT(DISTINCT context_norm) AS n FROM terms "
                "WHERE term_norm = ? AND deleted_at IS NULL",
                (term_norm,),
            ).fetchone()
        return int(row["n"])

    def _smallest_id(self, term_norm: str) -> int | None:
        with session(self._settings) as conn:
            row = conn.execute(
                "SELECT MIN(id) AS id FROM terms "
                "WHERE term_norm = ? AND deleted_at IS NULL",
                (term_norm,),
            ).fetchone()
        value = row["id"]
        return int(value) if value is not None else None

    def _occupant(self, term_norm: str, context_norm: str) -> int | None:
        """The live record holding the canonical key (or None)."""
        with session(self._settings) as conn:
            row = conn.execute(
                "SELECT id FROM terms WHERE term_norm = ? AND context_norm = ? "
                "AND deleted_at IS NULL",
                (term_norm, context_norm),
            ).fetchone()
        return int(row["id"]) if row is not None else None

    @staticmethod
    def _log(message: str, event: str, **fields: Any) -> None:
        """The job event into the shared journal: `event` + the mandatory `job` field."""
        logging.getLogger("app").info(
            message, extra={"event": event, "job": TERMS_NORMALIZE_JOB, **fields}
        )


def _connected_components(
    rows: list[dict[str, Any]], threshold: float
) -> list[list[dict[str, Any]]]:
    """Single-linkage binding at `context_similarity >= threshold`."""
    parent = list(range(len(rows)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[max(root_left, root_right)] = min(root_left, root_right)

    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            if threshold <= context_similarity(
                rows[left]["context"], rows[right]["context"]
            ):
                union(left, right)
    buckets: dict[int, list[dict[str, Any]]] = {}
    for index, row in enumerate(rows):
        buckets.setdefault(find(index), []).append(row)
    return list(buckets.values())


def _canonical(
    members: list[dict[str, Any]], uses: dict[str, int]
) -> dict[str, Any]:
    """A cluster's canonical wording - deterministic over the data (§6.6)."""
    return min(
        members,
        key=lambda member: (
            -uses.get(member["context_norm"], 0),
            -int(member["id"]),
            len(member["context"]),
            member["context_norm"],
        ),
    )


def build_terms_normalize_job(
    worker: "BackgroundWorker", settings: Settings
) -> JobSpec | None:
    """The `terms-normalize` job builder (FR-7.1): without the terms area - inapplicable.

            The shape "by interval" (no event, no queue recheck): a run goes into
            `to_thread`; the queue is derived from `terms`. No slot: the job works
            without an LLM (§13.10) and never evicts the production slots (FR-7.4).
            The gate - its own switch AND `AREA_TERMS_ENABLED` (No40, §6.6): a
            disabled area is not processed in the background.

    """
    if worker._terms is None:  # a DI build without the area - no loop
        return None
    service = TermsNormalizerService(
        settings,
        terms=worker._terms,
        warnings=WarningsService(settings, namespaces=worker._namespaces),
    )
    batch = settings.job_terms_batch

    async def process() -> int:
        return await asyncio.to_thread(service.run, batch)

    return JobSpec(
        name=TERMS_NORMALIZE_JOB,
        queue=TERMS_NORMALIZE_JOB,
        interval_sec=settings.job_terms_interval_sec,
        batch=batch,
        enabled=settings.job_terms_enabled and settings.area_terms_enabled,
        process=process,
        queue_empty=service.queue_empty,
        wait_event=None,
        idle_hook=None,
        queue_stat=service.queue_stat,
        slot=None,
    )


# The job registration in the framework registry (FR-7.1): a job lives in its
# own module (one line in `JOB_BUILDERS`); the import is one-way - terms_normalize -> jobs.
jobs.JOB_BUILDERS += (build_terms_normalize_job,)
