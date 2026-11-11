"""Service layer (ARCHITECTURE §3.2): one codebase for MCP and REST.

    `build_services` wires all services on top of a single `Settings` (frozen for
    the lifetime of the process); the main application keeps them on `app.state`.
    Phase 3: EmbeddingService is one per process — a shared httpx pool for hybrid
    search and write-time dedup; NoteService via DeduplicationService decides
    "duplicate / save" (FR-4), SearchService — hybrid search (ARCH §4.2).
    Phase 10: NamespaceService — registry of hierarchical namespaces (the map for
    `memory_namespaces`; NoteService/SearchService keep their own registry for
    write validation and output filters); ClassificationService — background
    grooming of default notes (Step 4).
    Phase 11: three LLM slot clients are wired here too — one per slot
    embedding/summary/judge (`LLMClient` per SlotSpec from Settings, decision #1);
    the structure judge and the description generator share their slots' clients,
    and the startup provider check (main.py, decision #5) walks the same clients
    (`Services.llm_embedding/llm_summary/llm_judge`). Prompts are injected into
    the clients by a single `PromptRegistry` (decision #7: system prompts are
    editable with file overrides, user templates are baked in).
"""

from __future__ import annotations

from dataclasses import dataclass

from app.config import Settings
from app.services.backup import BackupService
from app.services.classifier import ClassificationService
from app.services.decision import DecisionService
from app.services.dedup import DeduplicationService
from app.services.embedding import EmbeddingService
from app.services.judge import JudgeService
from app.services.links import LinksService
from app.services.llm_client import LLMClient, SlotSpec
from app.services.logs import LogsService
# LSBN-F-22 (M2): importing the module registers the `migration` job in the framework
# registry (`jobs.JOB_BUILDERS`) — same as `warnings`/`terms_normalize`
# (one-way import: migration → jobs).
from app.services.migration import MigrationService
from app.services.namespaces import NamespaceService
from app.services.notes import NoteService
from app.services.operations import OPERATIONS
from app.services.promotion import DescriptionService, PromotionService, StructureJudgeService
from app.services.prompts import PromptRegistry
from app.services.search import SearchService
from app.services.skills import SkillsService
from app.services.summary import SummaryService
from app.services.terms import TermsService
# #32 (3.2.0): importing the module registers the `terms-normalize` job in the registry
# (`jobs.JOB_BUILDERS`) — same as `warnings` above (one-way import:
# terms_normalize → jobs).
from app.services.terms_normalize import TermsNormalizerService
from app.services.user_facts import UserFactsService
from app.services.warnings import WarningsService

__all__ = [
    "BackupService",
    "ClassificationService",
    "DecisionService",
    "DeduplicationService",
    "EmbeddingService",
    "JudgeService",
    "LinksService",
    "LLMClient",
    "LogsService",
    "MigrationService",
    "NamespaceService",
    "NoteService",
    "PromotionService",
    "PromptRegistry",
    "SearchService",
    "Services",
    "SkillsService",
    "SummaryService",
    "TermsNormalizerService",
    "TermsService",
    "UserFactsService",
    "WarningsService",
    "build_services",
]


@dataclass(frozen=True)
class Services:
    """Service container; assembled once at application startup.

            `llm_embedding/llm_summary/llm_judge` — clients of the three LLM slots
            (decision #1): consumers share the client of their slot, and the startup
            provider check (main.py, decision #5) walks the same clients. The fields
            are optional (None) only for test DI builds where services are replaced
            with fakes without clients: the startup check of such a build is a
            stub (brief §3).
    """

    notes: NoteService
    search: SearchService
    embedding: EmbeddingService
    dedup: DeduplicationService
    summary: SummaryService  # worker — the only consumer (mode "B")
    judge: JudgeService  # dedup judge — worker, verdicts on candidates (Stage 3.2)
    backup: BackupService  # snapshot loop — asyncio task in lifespan
    namespaces: NamespaceService  # namespace registry (Phase 10, memory_namespaces)
    classifier: ClassificationService  # grooming of default notes (Phase 10, Step 4)
    promotion: PromotionService  # domain trigger (Phase 10, Step 5, memory_namespaces + worker)
    # 3.0.0: the "user" area (lsb-0009-01) — atomic facts about the user.
    # Mandatory, like notes/search: the container is complete in prod and in DI builds.
    # embedding — shared slot instance (fact writes never call it — §3.4;
    # assembly point for the area's hybrid search).
    user_facts: UserFactsService
    # 3.0.0: the "terms" area (lsb-0008-01) — terms keyed by (term + context).
    # Mandatory, like user_facts: the container is complete in prod and in DI builds.
    # embedding — shared slot instance (writes never call it — context similarity
    # is trigram-based, §3.5; assembly point for the area's hybrid search).
    terms: TermsService
    # 3.1.0: note links (lsb-0010) — level 0 ("lazy graph"). In prod (build_services)
    # it is always wired with the shared search instance; test DI builds without links pass None.
    links: LinksService
    # 3.2.0 (LSBN-F-10, arch §6.2/§13.11): the "Assistant logs" area — a journal of
    # assistant actions ("what was done, why, the object, the outcome, what was
    # verified"). Collected ALWAYS, like notes/terms/user_facts: operator REST
    # and the history layer must work with the area disabled (§13.11) — the flag
    # `AREA_LOGS_ENABLED` gates only the MCP surface and the background. `embedding` —
    # shared slot instance (the write path never calls it; assembly point for search 03).
    logs: LogsService | None = None
    llm_embedding: LLMClient | None = None  # embedding slot client (Phase 11, decision #5)
    llm_summary: LLMClient | None = None  # summary slot client (summarize/merge/classify/describe)
    llm_judge: LLMClient | None = None  # judge slot client (dedup + structure judge)
    # LSBN-F-12 (3.2.0): the decision slot — client (/v1/systemone) and decision service.
    # The pair is built ONLY when settings.decision_enabled is on (symmetric with the area
    # flags): a disabled slot is None, consumers fall back to the generative path.
    llm_decision: LLMClient | None = None  # decision slot client
    decision: DecisionService | None = None  # decisions: threshold/zone/gap
    # 3.0.0: skills area (lsb-0007). Optional — only for test DI builds created
    # before the area existed (their Services(...) without skills remain valid);
    # in prod build_services always provides the service.
    skills: SkillsService | None = None
    # 3.2.0 (#12, no12-01/no12-02): shared warnings surface state
    # `warnings` — read by MCP `warnings_list`, REST `/warnings` and
    # `/health.warnings`. No classifier needed at build time (reads are pure SQL);
    # optional for legacy DI builds, like `skills`.
    warnings: WarningsService | None = None


def build_services(
    settings: Settings,
    summary: SummaryService | None = None,
    judge: JudgeService | None = None,
) -> Services:
    """Assemble services from settings (Settings frozen — a snapshot per process).

            `summary` — DI of e2e tests (a fake instead of a live summary slot);
            default — the real SummaryService. `judge` — DI of the dedup judge
            (Phase 8, Stage 3.1), same principle. The clients of the three slots
            are always built from Settings (the startup check #5 verifies slot
            configuration even when services are replaced with fakes).

            Area services (skills/terms/user_facts/logs) are ALWAYS built,
            regardless of the `AREA_*_ENABLED` flags (#40, release 3.2.0): operator
            REST and the history layer must be able to read a disabled area's data.
            The flag gates only the MCP surface, the instruction section/announcement
            and the background `areas` loop — not the assembly.

            Raises:
                ConfigError: the prompt registry is invalid (for example, judge_system
                    loaded from a file without the required duplicate/no-duplicate
                    markers) — service startup must abort.
    """
    registry = PromptRegistry(
        settings.prompts_dir,
        decision_max_questions=settings.decision_max_questions,
        # 3.4.1 (LSBN-T-26): the number in merge/split prompts equals the write
        # validation limit and the migration target (one number — one place).
        max_note_chars=settings.max_note_chars,
    )
    # Slot clients — one per slot (decision #1); slot services share them, and
    # the startup provider check walks the same clients (decision #5).
    llm_embedding = LLMClient(SlotSpec.for_embedding(settings))
    llm_summary = LLMClient(SlotSpec.for_summary(settings))
    llm_judge = LLMClient(SlotSpec.for_judge(settings))
    # Operation clients (LSBN-F-34): flat env `<OPERATION>_<FIELD>` (chain:
    # operation env → slot env → default) on top of slot groups. Identical specs
    # (operations without overrides) share one HTTP pool — the client layout
    # changes only where the operator overrode the model.
    operation_clients: dict[str, LLMClient] = {}
    by_spec: dict[SlotSpec, LLMClient] = {}
    for operation in OPERATIONS:
        spec, _sources = SlotSpec.for_operation(operation, settings)
        client = by_spec.get(spec)
        if client is None:
            client = LLMClient(spec)
            by_spec[spec] = client
        operation_clients[operation] = client
    # LSBN-F-12: the decision pair is built only when the slot is enabled
    # (a disabled slot makes no calls — FR-1.1).
    llm_decision: LLMClient | None = None
    decision: DecisionService | None = None
    if settings.decision_enabled:
        llm_decision = LLMClient(SlotSpec.for_decision(settings))
        decision = DecisionService(settings, llm=llm_decision, registry=registry)
    embedding = EmbeddingService(settings, llm=operation_clients["embed"])
    dedup = DeduplicationService(settings)
    namespaces = NamespaceService(settings)
    search = SearchService(settings, embedding)
    # 3.1.0: note links (lsb-0010) — level 0 runs on the same search instance
    # (shared instance: same embedder/thresholds, no second assembly point).
    links = LinksService(settings, search=search)
    promotion = PromotionService(
        settings,
        embedding=embedding,
        describer=DescriptionService(
            settings, llm=operation_clients["describe"], registry=registry
        ),
        judge=StructureJudgeService(
            settings, llm=operation_clients["structure_judge"], registry=registry
        ),
        namespaces=namespaces,
    )
    # #30 (3.2.0): save receives the "similar" hint from the same search point
    # as the hybrid path (FTS+Dice, no model); the container wires the same notes.
    notes = NoteService(settings, embedding, dedup)
    notes.set_similar_finder(search.similar_by_text)
    return Services(
        notes=notes,
        search=search,
        embedding=embedding,
        dedup=dedup,
        summary=(
            summary
            if summary is not None
            else SummaryService(
                settings,
                llm={
                    operation: operation_clients[operation]
                    # "split" — migration splitting (LSBN-F-22 M2): the same operations
                    # registry, SPLIT_* env → slot inheritance.
                    for operation in ("summarize", "title", "merge", "split")
                },
                registry=registry,
            )
        ),
        judge=(
            judge
            if judge is not None
            else JudgeService(
                settings,
                llm=operation_clients["judge"],
                registry=registry,
                decider=decision,
            )
        ),
        backup=BackupService(settings, namespaces=namespaces),
        namespaces=namespaces,
        # LSBN-F-12: the judge and the classifier receive the decision service as
        # `decider` (disabled slot — None, v3.1.0 behavior); decision probabilities
        # are visible only in the logs (written by DecisionService itself).
        classifier=ClassificationService(
            settings, llm=operation_clients["classify"], registry=registry, decider=decision
        ),
        promotion=promotion,
        llm_embedding=llm_embedding,
        llm_summary=llm_summary,
        llm_judge=llm_judge,
        llm_decision=llm_decision,
        decision=decision,
        # 3.0.0 substrate: skills area service (lsb-0007-01); embedding — shared
        # slot instance (writes never call it — assembly point for track
        # search/anti-synonymy).
        skills=SkillsService(settings, embedding=embedding),
        # 3.0.0 substrate: "user" area service (lsb-0009-01); embedding — shared
        # slot instance (fact writes never call it — trigram dedup,
        # arch lsb-0009 §3.4; assembly point for the area's hybrid search).
        user_facts=UserFactsService(settings, embedding=embedding),
        # 3.0.0 substrate: terms area service (lsb-0008-01); embedding — shared
        # slot instance (writes never call it — context similarity is
        # trigram-based, arch lsb-0008 §3.5; assembly point for hybrid search).
        terms=TermsService(settings, embedding=embedding),
        links=links,
        # 3.2.0 (LSBN-F-10): "Assistant logs" area service (f10-02); embedding —
        # shared slot instance (writes never call it — the areas loop catches up
        # the vector; area search assembly point — f10-03).
        logs=LogsService(settings, embedding=embedding),
        # 3.2.0 (#12): shared warnings surface (no12-02) — pure SQL,
        # no classifier needed (writes come from the `warnings` job, #12 FR-3.2/§13.4).
        warnings=WarningsService(settings, classifier=None, namespaces=namespaces),
    )