"""Env parser — every environment variable of REQUIREMENTS §8.

Required variables (`EMBEDDING_BASE_URL`, `SUMMARY_BASE_URL`, `SUMMARY_MODEL`,
`JUDGE_BASE_URL`, `JUDGE_MODEL`, `MCP_AUTH_TOKEN`) carry no defaults: a
missing or empty value is a fatal startup error (NFR-2). Everything else
defaults to the REQUIREMENTS §8 table.

Phase 5 (NFR-6): every constraint is validated at startup — a bad value
(out of range, wrong separators, garbage) kills the service with a readable
message instead of springing runtime surprises: a wrong limit is a
configuration error, not a per-call failure.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from pydantic import Field, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Allowed log levels (NFR-4: LOG_LEVEL); the name flows into std logging.
LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})

# Allowed LLM slot providers (Phase 11): ollama — the native API, openai —
# the OpenAI-compatible surface (Bearer).
LLM_PROVIDERS = frozenset({"ollama", "openai"})

# No49 (3.2.0, arch §13.1): the model type of a generative slot is a
# configuration parameter. `reasoning` enables the budget/thinking guards;
# `decision` belongs to the separate decision slot (its own DECISION_* config)
# and is rejected with a hint on summary/judge.
LLM_MODEL_TYPES = frozenset({"reasoning", "generation"})

# Maximum words in a note title (Phase 11 brief): a code constant, NOT an
# env. Words = len(title.split()).
TITLE_MAX_WORDS = 5

# Soft note-text limit — the default (3.4.1: 1500 → 2000 so the 176 active
# prod notes in the 1500–2000 band stop being split; the "one note = one
# cluster of facts" band holds). The SINGLE home of this number:
# Settings.max_note_chars (env MAX_NOTE_CHARS). The NOTE_TEXT_LIMIT code
# constant is gone (LSBN-T-26): save validation, merge/split part limits,
# hints, tool descriptions, instructions and prompts take the value from
# settings at build time. The 500..20000 range protects the merge-part
# contract ("a part is readable on its own") — the config must not break it.
MAX_NOTE_CHARS_DEFAULT = 2000


class ConfigError(RuntimeError):
    """Fatal configuration error: the service must refuse to start."""


class Settings(BaseSettings):
    """Service settings; the source is environment variables only.

    Field names match the REQUIREMENTS §8 env variables (pydantic-settings
    matches case-insensitively).
    """

    model_config = SettingsConfigDict(
        env_file=None,  # environment only; .env belongs to deployment (docker compose)
        extra="ignore",  # foreign variables do not break parsing
        frozen=True,  # settings are immutable after startup
    )

    # --- required (REQUIREMENTS §8: "required", no defaults) ---
    embedding_base_url: str = Field(...)  # embedding slot API address (Phase 11)
    summary_base_url: str = Field(...)  # summary slot API address (Phase 11)
    summary_model: str = Field(...)  # generative summarization model
    judge_base_url: str = Field(...)  # judge slot API address (Phase 11)
    judge_model: str = Field(...)  # dedup judge model (Phase 11)
    mcp_auth_token: str = Field(...)  # Bearer token (NFR-2)

    # --- per-slot providers (Phase 11): ollama is the default | openai ---
    embedding_provider: str = "ollama"
    summary_provider: str = "ollama"
    judge_provider: str = "ollama"

    # --- per-slot API keys (Phase 11): optional, default "" ---
    embedding_api_key: str = ""
    summary_api_key: str = ""
    judge_api_key: str = ""

    # --- editable prompts directory (Phase 11): unset — the file mechanics
    # are inactive, the built-in constants are used ---
    prompts_dir: str | None = None

    # --- vectorization ---
    embedding_model: str = "qwen3-embedding:8b"
    embedding_dim: int = 4096  # fixed at database creation (vec0 table)

    # --- chunked indexing (Phase 7): vectors per note chunk ---
    text_splitter: str = "tiktoken"  # token splitter; encoding pinned: cl100k_base
    chunk_size: int = 1024  # chunk window, tokens
    chunk_overlap: int = 180  # overlap of neighboring windows, tokens
    chunk_min_target: int = 200  # a shorter tail chunk merges into the previous one
    embedding_batch_size: int = 32  # chunks per worker /api/embed request
    embedding_concurrent_requests: int = 3  # parallel worker embed requests

    # --- summarization ---
    # The summary length limit in characters (the 3.1.0 gate decision: 150).
    # Keeps both sides on one constant: the hard guard of the MODEL summary at
    # save time (worker.py, cap_summary — word-boundary truncation) and the
    # fallback truncation of the note text in outputs (emit.py, while
    # summary_status != 'ok'). The earlier 200 was a fallback-only limit.
    max_summary_chars: int = 150
    summary_think: bool = True  # when false the call carries "think": false
    # No49 (arch §13.1): reasoning | generation — affects the budget, the
    # thinking-flag treatment and startup warnings (not the request body).
    summary_model_type: str = "generation"
    # No49 (3.2.0): 0 = "as much as fits" (the auto budget from the declared
    # model context minus the prompt and the reserve, clamped by the defensive
    # cap); the earlier 35000 — the explicit operator cap and the degradation
    # when the context is unknown.
    summary_num_predict: int = 0
    merge_num_predict: int = 0  # the separate dedup-merge cap
    # No49 (arch §13.4): 900 = the back-off/slot pause ceiling ("thinking
    # long" is not an error); was 60.
    summary_timeout_sec: int = 900  # the client read timeout of a call
    # The declared context of the summary slot model (0 — unknown → the
    # previous caps without failing); the chat-template/service-fields reserve
    # and the auto-budget defensive cap are shared across generative slots.
    # 12.11.2026 (О.): облачные модели ollama.com лимиты не принимают —
    # флаг снимает num_predицt/num_ctx/max_tokens из запроса целиком.
    summary_no_output_limits: bool = False
    summary_context_tokens: int = 0
    slot_prompt_reserve_tokens: int = 512
    slot_max_output_tokens: int = 32768
    # LSBN-T-33: the slot-level REASONING budget (0 — unset: "as much as fits"
    # within the context, never at the answer's expense). The transport caps
    # the thinking+content SUM, so LSB enforces the split itself: the call
    # carries num_predict = answer_budget + reasoning_budget (clamped by the
    # context). The per-operation level is `<OP>_REASONING_NUM_PREDICT` (block
    # below).
    slot_reasoning_num_predict: int = 0
    # No49 (arch §8): the WARN threshold "the budget leaves no room for
    # thinking" and the WARN threshold "the timeout is small for long thinking".
    slot_reasoning_min_output_tokens: int = 4096
    slot_min_timeout_sec: int = 300

    # --- operations over slots (LSBN-F-34, 3.4.0; canon 4.2.8) --------
    # Flat `<OPERATION>_<FIELD>` env (no OP_ prefix) overrides the
    # configuration of ONE process over the slot group: the resolution chain
    # per field is operation env → slot env → default. None = "not set" —
    # the slot is inherited, a partial override (only MERGE_MODEL) is the norm;
    # operations without overrides behave exactly as before. The operations
    # registry — app/services/operations.py; fields: MODEL, PROVIDER, BASE_URL,
    # API_KEY, THINK, NUM_PREDICT, REASONING_NUM_PREDICT, TIMEOUT_SEC,
    # CONTEXT_TOKENS, MODEL_TYPE. Secrets (API_KEY) never reach /health or the
    # config fingerprint. The historical `merge_num_predict` is already an
    # operation env: absorbed by the mechanism, behavior unchanged (0 = auto,
    # slot inheritance for merge stays off).
    summarize_model: str | None = None
    summarize_provider: str | None = None
    summarize_base_url: str | None = None
    summarize_api_key: str | None = None
    summarize_think: bool | None = None
    summarize_num_predict: int | None = None
    summarize_reasoning_num_predict: int | None = None
    summarize_timeout_sec: int | None = None
    summarize_context_tokens: int | None = None
    summarize_model_type: str | None = None

    title_model: str | None = None
    title_provider: str | None = None
    title_base_url: str | None = None
    title_api_key: str | None = None
    title_think: bool | None = None
    title_num_predict: int | None = None
    title_reasoning_num_predict: int | None = None
    title_timeout_sec: int | None = None
    title_context_tokens: int | None = None
    title_model_type: str | None = None

    merge_model: str | None = None
    merge_provider: str | None = None
    merge_base_url: str | None = None
    merge_api_key: str | None = None
    merge_think: bool | None = None
    merge_reasoning_num_predict: int | None = None
    merge_timeout_sec: int | None = None
    merge_context_tokens: int | None = None
    merge_model_type: str | None = None

    split_model: str | None = None
    split_provider: str | None = None
    split_base_url: str | None = None
    split_api_key: str | None = None
    split_think: bool | None = None
    split_num_predict: int | None = None
    split_reasoning_num_predict: int | None = None
    split_timeout_sec: int | None = None
    split_context_tokens: int | None = None
    split_model_type: str | None = None

    classify_model: str | None = None
    classify_provider: str | None = None
    classify_base_url: str | None = None
    classify_api_key: str | None = None
    classify_think: bool | None = None
    classify_num_predict: int | None = None
    classify_reasoning_num_predict: int | None = None
    classify_timeout_sec: int | None = None
    classify_context_tokens: int | None = None
    classify_model_type: str | None = None

    describe_model: str | None = None
    describe_provider: str | None = None
    describe_base_url: str | None = None
    describe_api_key: str | None = None
    describe_think: bool | None = None
    describe_num_predict: int | None = None
    describe_reasoning_num_predict: int | None = None
    describe_timeout_sec: int | None = None
    describe_context_tokens: int | None = None
    describe_model_type: str | None = None

    structure_judge_model: str | None = None
    structure_judge_provider: str | None = None
    structure_judge_base_url: str | None = None
    structure_judge_api_key: str | None = None
    structure_judge_think: bool | None = None
    structure_judge_num_predict: int | None = None
    structure_judge_reasoning_num_predict: int | None = None
    structure_judge_timeout_sec: int | None = None
    structure_judge_context_tokens: int | None = None
    structure_judge_model_type: str | None = None

    embed_model: str | None = None
    embed_provider: str | None = None
    embed_base_url: str | None = None
    embed_api_key: str | None = None
    embed_think: bool | None = None
    embed_num_predict: int | None = None
    embed_reasoning_num_predict: int | None = None
    embed_timeout_sec: int | None = None
    embed_context_tokens: int | None = None
    embed_model_type: str | None = None

    # --- output ---
    snippet_chars: int = 120
    max_get_batch: int = 20

    # --- HTTP / MCP ---
    port: int = 8080
    mcp_path: str = "/mcp"

    # --- storage ---
    db_path: str = "/data/notes.db"

    # --- search ---
    default_top_k: int = 5
    default_list_limit: int = 20
    # Listing ceilings per surface (lsb-0013, release 3.1.0): the MCP surface
    # keeps the model context tight (20 records), the REST surface for the
    # operator is looser (50). The order default ≤ MCP ≤ REST is validated at
    # startup (see _validate_ranges).
    list_max_limit_mcp: int = 20
    list_max_limit_rest: int = 50
    # No12 (no12-02, 3.2.0): warnings page ceilings per surface (§7.1):
    # the agent MCP surface — 20, the operator REST — 50.
    warnings_max_limit_mcp: int = 20
    warnings_max_limit_rest: int = 50
    # Calibrated on 82 real queries: 0.35 → 0.50.
    score_threshold: float = 0.50
    dedup_similarity: float = 0.92
    # --- background dedup (Phase 8, step 2.1): candidate pre-filter before the summary ---
    dedup_candidate_top_n: int = 3  # top-N candidates (the judge verifies, step 3)
    dedup_candidate_similarity: float = 0.80  # paraphrase-candidate lower threshold
    # --- LLM dedup judge (Phase 8, step 3.1; Phase 11 renamed the block to judge_*):
    # ornith-1.5:35b, think:false ---
    judge_think: bool = False  # when false the call carries "think": false
    # No49: 0 = auto budget; 256 — the previous cap (legacy when the context is unknown).
    judge_num_predict: int = 0
    # LSBN-T-33: the judge operation reasoning budget (None — unset: the global
    # SLOT_REASONING_NUM_PREDICT). The judge operation name equals its slot
    # group, so this operation field is the only layer above the global one.
    judge_reasoning_num_predict: int | None = None
    # No49 (arch §13.1): the judge slot model type (reasoning | generation).
    judge_model_type: str = "generation"
    # No49 (arch §13.4): 900 = the back-off/slot pause ceiling; was 30.
    judge_timeout_sec: int = 900  # client read timeout of the judge call
    judge_no_output_limits: bool = False
    judge_context_tokens: int = 0  # declared judge model context (0 — unknown)
    rrf_k: int = 60
    # Ranking boosts after RRF (LSBN-F-14, 3.4.0): the operator switch
    # (false — the output order is exactly pre-boost; the extra links SQL does
    # not run). Boost weights are code constants in app/services/ranking.py,
    # NOT env (pinned in the canon via changes.md).
    ranking_boosts_enabled: bool = True

    # --- note links, level 0 (lsb-0010, 3.1.0) ---
    # LINK_TOP — the per-note link ceiling (FR-1.1); LINK_LAZY_THRESHOLD —
    # the "lazy graph" cosine threshold (the default equals SCORE_THRESHOLD);
    # LINK_POOL — the KNN candidate window for links (arch §3.1).
    link_top: int = 3
    link_lazy_threshold: float = 0.50
    link_pool: int = 20

    # --- note links, level 1: the links table, computed without an LLM (lsb-0010-02) ---
    # LINK_COSINE_THRESHOLD — the `cosine` kind threshold (above search: a link
    # must be stronger than "just similar", FR-2.3); LINK_ENTITIES_MIN_COMMON —
    # how many shared significant words make a pair a link;
    # LINK_ENTITIES_MIN_WORD_CHARS — the minimum significant-word length
    # (arch §3.3).
    link_cosine_threshold: float = 0.70
    link_entities_min_common: int = 2
    link_entities_min_word_chars: int = 5

    # --- framework background jobs (lsb-0014, 3.1.0): env-scheduled ---
    # The `links` job (lsb-0010-03, FR-2.2): the queue is the service marker
    # notes.links_at; JOB_LINKS_INTERVAL_SEC — the starting wait pause (>= 30 s),
    # JOB_LINKS_BATCH — notes processed per run (>= 1), JOB_LINKS_ENABLED —
    # the operator switch (false — the job does not start, its queue stays
    # visible in /health).
    job_links_enabled: bool = True
    job_links_interval_sec: int = 300
    job_links_batch: int = 100

    # The `nodes` "order in nodes" job (lsb-0011-01, FR-1.1/FR-1.2): sweeps
    # the accumulated `default` — interval >= 30 s (default 1 hour), batch >= 1
    # (the shared per-run processing budget, default 20), a switch (false —
    # the job does not start, its queue stays visible in /health).
    job_nodes_enabled: bool = True
    job_nodes_interval_sec: int = 3600
    job_nodes_batch: int = 20
    # The classifier-call budget per run (lsb-0011-02, FR-2.3): >= 1 and not
    # above the shared batch — model calls are economized (arch §3.2/§3.4,
    # default 10).
    job_nodes_classifier_budget: int = 10

    # The `warnings` job (No12, 3.2.0; arch §6.5/§8): sweeps PLACED notes
    # (not `default`, or already groomed by `nodes`) and highlights divergence
    # from the classifier placement — WITHOUT auto-moving (§13.5). Interval
    # >= 30 s (default 1 hour), batch >= 1 (default 20), classifier-call budget
    # per run >= 1 and <= batch (default 10), confidence threshold 0..1
    # (default 0.80), strict parent/child level (default false — small noise is
    # not highlighted); a switch (false — the job does not start, its queue
    # stays visible in /health).
    job_warnings_enabled: bool = True
    job_warnings_interval_sec: int = 3600
    job_warnings_batch: int = 20
    job_warnings_classifier_budget: int = 10
    warnings_min_confidence: float = 0.80
    warnings_level_strict: bool = False

    # The `terms-normalize` job (No32, 3.2.0; arch §6.6/§8): term-context
    # clusters + canonical spelling, no LLM. Interval >= 30 s (default 1800),
    # batch (terms per run) >= 1 (default 50), single-linkage clustering
    # threshold 0..1 (default 0.75, separate from `term_context_similarity` so
    # background tuning does not weaken the soft refusal on save), the
    # contexts-excess threshold >= 2 (default 5). A switch; the job also obeys
    # `AREA_TERMS_ENABLED` (No40).
    job_terms_enabled: bool = True
    job_terms_interval_sec: int = 1800
    job_terms_batch: int = 50
    job_terms_cluster_similarity: float = 0.75
    job_terms_contexts_excess: int = 5

    # --- limits (NFR-6: env-overridable, validated; see _validate_ranges) ---
    # 3.4.1 (LSBN-T-26): the soft note limit moved from a code constant
    # (NOTE_TEXT_LIMIT, 24 occurrences in 6 modules) into settings — one number
    # feeds the CHECK of new databases + the self-migration target (F-22), the
    # save/update validation, merge/split part limits, hints, tool descriptions,
    # instructions and prompts. Changed via an override + restart; the base is
    # carried by self-migration (up — a 1:1 pour without a model, down —
    # splitting).
    max_note_chars: int = MAX_NOTE_CHARS_DEFAULT  # 2000 → 20000 (Phase 7) → 35000 → 1500 (F-22) → 2000 (3.4.1)
    max_query_chars: int = 512
    # no24-01 (3.2.0): the note `comment` limit (domain validation, not a
    # schema CHECK; soft refusal CommentValidationError).
    comment_max_chars: int = 250

    # --- background operations ---
    pending_retry_sec: int = 30  # starting re-vectorization/re-summarization interval

    # --- migration slot cadence (3.4.1, LSBN-T-27) --
    # During a rollout, splitting must not monopolize the generation server:
    # the migration job takes a MIGRATION_BATCH batch of notes, processes it,
    # yields the slot for MIGRATION_PAUSE_SEC seconds, then takes the next
    # batch. Its own numbers INSTEAD of borrowed meanings: embedding_batch_size
    # is chunks of one vectorization /api/embed request, pending_retry_sec is
    # the starting retry interval of catch-up processing — they must not be
    # tuned for migration. The rollout window depends on the machine and model,
    # so both numbers are configuration (an override edit + restart, no
    # release). No parallel workers: the splitting queue is single-threaded.
    migration_batch: int = 5  # notes per migration job run
    migration_pause_sec: int = 600  # the pause AFTER a run, before the next (10 min)

    # --- slot breaks (No39, 3.2.0) ----------------------------------
    # SLOT_PAUSE_SEC — the slot pause during a break (= the back-off ceiling),
    # one probe knock per pause; SLOT_PAUSE_AFTER — how many consecutive
    # non-transient failures of ONE outcome open a break (for `auth` —
    # immediately, the threshold does not apply); SLOT_HINT_EVERY_SEC — the
    # "warn your human" hint throttling per slot (0 — every call, not
    # recommended). Validated at startup (see _validate_ranges).
    slot_pause_sec: int = 900
    slot_pause_after: int = 3
    slot_hint_every_sec: int = 900
    # SLOT_MAX_ATTEMPTS — the attempt limit of a PAIR before "requires
    # attention": exhausted, the job freezes while staying `pending` (model
    # calls are not burned) and resumes when the cause goes away
    # (No39 §6.8, no39-02). SLOT_ATTENTION_RESET_ON_START — whether to run the
    # startup slot-config fingerprint check: true — the fingerprint
    # (`slot_config_fingerprint`, lsbdef0008-02) is compared with the one
    # stored in `meta` and attention clears when it is absent or different;
    # false — the step is disabled entirely (the operator's will; the
    # fingerprint is neither read nor written).
    slot_max_attempts: int = 8
    slot_attention_reset_on_start: bool = True

    # --- decision slot (LSBN-F-12, 3.2.0; arch §6.4/§8/§13.6) ----------
    # The third slot kind: the client branch POST /v1/systemone (state +
    # questions → probabilities). Off by default — decisions run generatively
    # as today (zero regression risk); enabled and configured from
    # docker-compose. Provider — native ollama >= 0.35.1 (the route is NOT
    # OpenAI-compatible); the key goes as Bearer only when set.
    # THRESHOLD/UNCERTAINTY_ZONE — the switch mechanics (numbers are tuned in
    # testing); MAX_QUESTIONS/MAX_LABELS — request ceilings.
    decision_enabled: bool = False
    decision_base_url: str = ""
    decision_model: str = ""
    decision_api_key: str = ""
    decision_timeout_sec: int = 300
    decision_threshold: float = 0.5
    decision_uncertainty_zone: float = 0.05
    decision_fallback: bool = True
    decision_max_questions: int = 64
    decision_max_labels: int = 40

    # --- version history (No41, 3.2.0) ---
    # HISTORY_ENABLED — the history layer kill switch: false — versions are
    # not written, reads return empty. HISTORY_MAX_VERSIONS — the per-record
    # version ceiling (0 — keep everything, the FR-10.1 default);
    # HISTORY_RETENTION_DAYS — the version retention in days (0 — unlimited).
    # Above 0, prune removes the oldest versions and re-bases the earliest
    # kept one into a full snapshot (§13.8). HISTORY_DELTA_INLINE_MAX — the
    # "whole field ↔ line delta" threshold in characters (§5.2).
    history_enabled: bool = True
    history_max_versions: int = 0
    history_retention_days: int = 0
    history_delta_inline_max: int = 2048

    # --- area flags (No40, 3.2.0) ----------------------------------
    # AREA_*_ENABLED — the operator switch of an area (default "on",
    # FR-1.1/FR-1.2): a disabled area serves no MCP handles, its section and
    # announcement are not baked into the instructions, the `areas` loop skips
    # it; the area service is still built and the operator REST stays complete
    # (owner decision 2026-10-08: disabled-area data remains reachable via
    # REST and the history layer). An invalid value — pydantic
    # ValidationError → ConfigError → a fatal start listing ALL violations
    # (FR-1.3, NFR-6), not a silent shutdown.
    area_skills_enabled: bool = True
    area_terms_enabled: bool = True
    area_user_facts_enabled: bool = True
    # The "Assistant logs" area (LSBN-F-10, arch §9): gates the MCP handles,
    # the instructions section and the background loop; data/REST/history stay
    # reachable (No40, §13.11).
    area_logs_enabled: bool = True

    # --- misc ---
    log_level: str = "INFO"
    log_file: str = ""  # JSON-лог в файл (ротация); пусто — только stdout
    log_max_bytes: int = 50_000_000
    log_backups: int = 5
    author_default: str = "unknown"

    # --- backup (NFR-3) ---
    backup_dir: str = "/data/backups"
    backup_interval_sec: int = 86400  # a day
    backup_keep: int = 7
    # No14 (3.2.0, arch §8): background-verify every Nth snapshot;
    # 0 — automatic verification is off (manual POST /backups/verify only).
    backup_verify_every: int = 1
    # No14 (3.2.0, arch §8/§13.4): ship a verified copy off the server.
    # ONLY the command is stored: destinations and credentials live OUTSIDE
    # the repository and the image (host env/bind-mount); the service passes
    # the command nothing but the file path. Empty (default) — shipping is
    # off, a silent no-op without events.
    backup_ship_cmd: str = ""
    backup_ship_timeout_sec: int = 300

    # --- new-release hint (No45, 3.2.0; arch §6.2/§8/§13.8-13.13) ----
    # Background version check: the source is a local JSON file (works without
    # the internet) preferentially; HTTP is a deliberate operator option. Off
    # by default: the container's internet access has never been probed, so
    # "network from runtime" cannot be the default (arch §13.8). The cache and
    # the "hint shown" mark live in process memory (no DB). Poll interval
    # >= 60 s, hint period >= 0 (0 — one per process), text limit 1..400.
    update_check_enabled: bool = False
    update_source_file: str = ""  # local JSON source (preferential)
    update_check_url: str = ""  # HTTP source (optional)
    update_check_interval_sec: int = 21600  # background poll period (6 h)
    update_hint_every_sec: int = 86400  # at most one hint per period
    update_hint_max_chars: int = 240  # hint text limit

    # --- namespaces (Phase 10, REQUIREMENTS §5.7/§8) ---
    namespace_auto_move_min_confidence: float = 0.80  # auto-moving a default note into an existing domain
    namespace_promotion_threshold: int = 15  # counter of one-hint default notes → auto-creating a leaf
    namespace_promotion_min_confidence: float = 0.60  # minimum placement confidence counted by the trigger
    namespace_synonym_similarity: float = 0.85  # anti-synonymy: description cosine — merge instead of create
    namespace_auto_max_per_day: int = 3  # auto-created nodes per day (storm guard)
    namespace_max_leaves_per_domain: int = 12  # leaf ceiling in a root
    namespace_groom_min_notes: int = 2  # grooming: a node with fewer notes is a merge candidate
    # The node hint on save without `namespace` (No28, 3.2.0; arch §6.2/§8):
    # the title words and the first `MEMORY_NODE_HINT_TEXT_CHARS` characters
    # of the text are compared with path segments (weight 3.0) and description
    # words (weight 2.0) — WITHOUT a model call; the hint only orients: it
    # never moves the note and never blocks the save (§13.2). `MIN_SCORE` 2.0
    # = ">= 1 matched significant word"; `MAX_NODES` 1..2 (FR-1.1); the switch
    # kills the whole field.
    memory_node_hint_enabled: bool = True
    memory_node_hint_min_score: float = 2.0
    memory_node_hint_max_nodes: int = 2
    memory_node_hint_text_chars: int = 200
    # The "similar" hint on save (No30, 3.2.0; arch §6.4/§8/§13.4):
    # candidates from the `notes_fts` FTS index + Dice over the significant
    # words of the title and the first `MEMORY_SIMILAR_TEXT_CHARS` characters
    # of the text — WITHOUT a model call (a fresh note has no vector yet).
    # The hint never blocks a save and never fails it on a search error;
    # `LIMIT` 1..3; `POOL` >= `LIMIT`.
    memory_similar_enabled: bool = True
    memory_similar_limit: int = 2
    memory_similar_pool: int = 5
    # No30: the save-time "similar" hint runs WITHOUT a model and embedding —
    # Dice over the significant-word sets of the title and the first
    # `text_chars` characters of the text (arch §6.4). The earlier 0.30
    # required matching only ~a sixth of the vocabulary (on short texts —
    # almost always), so the hint was noisy; 0.74 means "sharing about three
    # quarters of the significant words" — the same thought, rewritten.
    memory_similar_threshold: float = 0.74
    memory_similar_text_chars: int = 500
    # The structure judge (Phase 10): its own think flag — the dedup judge
    # stays thinking (the Phase 8 decision), the structure judge thoughtless
    # (Phase 7 E2E: a thinking judge "stuck" on twin pairs, starving
    # summarization; the verdict is 10–50 tokens, no reasoning needed).
    # None — inherits JUDGE_THINK.
    namespace_judge_think: bool | None = None

    # --- 3.0.0 areas (substrate: skills / terms / user) ---------------------
    # Form limits and similarity thresholds come from the lsb-0007/0008/0009
    # feature contracts; env-tunable, validated at startup (see
    # _validate_ranges): the limits are applied by the area service (schema
    # CHECK constraints, unlike notes, are not used). The skill form limits
    # were raised (canon §6): description 250→400, steps 500→1000, text
    # 4000→7500, example 1000→2000, an extra field 500→1000 (total 2000→4000),
    # instruction_template 1000→2000. Reason: real procedures did not fit the
    # earlier ceilings and the model fragmented a skill. `name` (65 chars,
    # <= 5 words) untouched: it is a naming recommendation, not a content
    # ceiling; the announce budget (2000) untouched too — it is about context.
    skill_name_max_chars: int = 65
    skill_description_max_chars: int = 400
    skill_steps_max_chars: int = 1000
    skill_text_max_chars: int = 7500
    skill_example_max_chars: int = 2000
    skill_extra_field_max_chars: int = 1000  # one skill-class field (JSON extra)
    skill_extra_total_max_chars: int = 4000  # the total of all extra fields
    instruction_template_max_chars: int = 2000  # the global template (skills_meta)
    skill_announce_max_chars: int = 2000  # the skill announce block budget
    skill_announce_description_chars: int = 120  # the description excerpt in the announce
    skill_synonym_similarity: float = 0.90  # anti-synonymy at skill creation
    term_max_chars: int = 100
    term_context_max_chars: int = 40
    term_definition_max_chars: int = 350
    term_context_similarity: float = 0.75  # trigram closeness of contexts
    term_contexts_hint_limit: int = 30  # top used contexts in the hint
    user_name_max_words: int = 5  # the note title contract (TITLE_MAX_WORDS)
    user_body_max_chars: int = 1200
    user_similar_strong: float = 0.85  # soft refusal "almost the same fact"
    user_similar_weak: float = 0.55  # the middle band: saved + a similar list
    user_search_excerpt_chars: int = 300  # body truncation in search output
    # The "Assistant logs" area (LSBN-F-10, arch §9): record form limits
    # (service validation, no CHECK constraints) and the action excerpt in
    # outputs.
    logs_action_max_chars: int = 500
    logs_reason_max_chars: int = 500
    logs_entity_ref_max_chars: int = 200
    logs_outcome_max_chars: int = 500
    logs_verified_max_chars: int = 200
    logs_search_excerpt_chars: int = 300

    @field_validator(
        "embedding_base_url",
        "summary_base_url",
        "summary_model",
        "judge_base_url",
        "judge_model",
        "mcp_auth_token",
    )
    @classmethod
    def _required_not_empty(cls, value: str) -> str:
        """Required variables: an empty/whitespace string counts as absent."""
        if not value or not value.strip():
            raise ValueError("обязательная переменная пуста — задай значение")
        return value

    @field_validator("embedding_provider", "summary_provider", "judge_provider")
    @classmethod
    def _validate_provider(cls, value: str) -> str:
        """The slot provider is one of {ollama, openai} (Phase 11)."""
        provider = value.strip().lower()
        if provider not in LLM_PROVIDERS:
            raise ValueError(
                "провайдер — один из " + ", ".join(sorted(LLM_PROVIDERS))
            )
        return provider

    @field_validator("summary_model_type", "judge_model_type")
    @classmethod
    def _validate_model_type(cls, value: str) -> str:
        """The slot model type is one of {reasoning, generation} (arch §13.1).

        `decision` is a standalone slot with its own config: declaring it on a
        generative slot points the operator to `DECISION_ENABLED`.
        """
        model_type = value.strip().lower()
        if model_type == "decision":
            raise ValueError(
                "decision — отдельный слот: enable the decision slot via "
                "DECISION_ENABLED instead"
            )
        if model_type not in LLM_MODEL_TYPES:
            raise ValueError(
                "тип модели — один из " + ", ".join(sorted(LLM_MODEL_TYPES))
            )
        return model_type

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, value: str) -> str:
        """LOG_LEVEL — a std logging level name (NFR-4), case-insensitive."""
        level = value.strip().upper()
        if level not in LOG_LEVELS:
            raise ValueError("LOG_LEVEL — один из " + ", ".join(sorted(LOG_LEVELS)))
        return level

    # --- operations over slots (LSBN-F-34): validation when set -----------
    # Optional operation fields (None = "not set" — the slot is inherited) are
    # validated by the same rules as slot fields: provider ∈ {ollama, openai},
    # model type ∈ {reasoning, generation} (decision is a standalone slot and
    # is forbidden on operations).
    @field_validator(
        "summarize_provider",
        "title_provider",
        "merge_provider",
        "split_provider",
        "classify_provider",
        "describe_provider",
        "structure_judge_provider",
        "embed_provider",
    )
    @classmethod
    def _validate_operation_provider(cls, value: str | None) -> str | None:
        """The operation provider is ollama|openai when set (None — the slot)."""
        if value is None:
            return None
        provider = value.strip().lower()
        if provider not in LLM_PROVIDERS:
            raise ValueError(
                "провайдер операции — один из " + ", ".join(sorted(LLM_PROVIDERS))
            )
        return provider

    @field_validator(
        "summarize_model_type",
        "title_model_type",
        "merge_model_type",
        "split_model_type",
        "classify_model_type",
        "describe_model_type",
        "structure_judge_model_type",
        "embed_model_type",
    )
    @classmethod
    def _validate_operation_model_type(cls, value: str | None) -> str | None:
        """The operation model type is reasoning|generation when set."""
        if value is None:
            return None
        model_type = value.strip().lower()
        if model_type == "decision":
            raise ValueError(
                "decision — отдельный слот: enable the decision slot via "
                "DECISION_ENABLED instead"
            )
        if model_type not in LLM_MODEL_TYPES:
            raise ValueError(
                "тип модели операции — один из "
                + ", ".join(sorted(LLM_MODEL_TYPES))
            )
        return model_type

    @model_validator(mode="after")
    def _validate_ranges(self) -> Settings:
        """Ranges of every limit and behavior-affecting field (NFR-6).

        ALL violations are collected at once — the operator fixes the
        environment in one restart, not one error per restart. The hard
        contract ceiling stays with top_k ≤ 20; listing ceilings (lsb-0013)
        are env parameters of the surfaces (MCP ≤ REST) — their order and the
        default are validated.
        """
        errors: list[str] = []

        def need_low(field: str, value: int, low: int) -> None:
            if value < low:
                errors.append(f"  - {field}: минимум {low}, получено {value}")

        def need_range(field: str, value: float, low: float, high: float) -> None:
            if not low <= value <= high:
                errors.append(
                    f"  - {field}: должно быть в диапазоне "
                    f"{low}..{high}, получено {value}"
                )

        # --- HTTP ---
        need_range("port", self.port, 1, 65535)
        if not self.mcp_path.startswith("/"):
            errors.append("  - mcp_path: путь должен начинаться с «/»")

        # --- external API slots: http(s) only ---
        for field in ("embedding_base_url", "summary_base_url", "judge_base_url"):
            url = getattr(self, field)
            if not url.strip().startswith(("http://", "https://")):
                errors.append(f"  - {field}: ожидается URL с http:// или https://")

        # --- output/input limits (NFR-6) ---
        # 3.4.1 (LSBN-T-26): 500..20000 — the merge-part contract ("each part
        # is readable on its own") holds on both sides: below 500 a note stops
        # being self-sufficient, above 20000 merge parts become unreadable.
        need_range("max_note_chars", self.max_note_chars, 500, 20000)
        need_low("max_query_chars", self.max_query_chars, 1)
        need_low("max_summary_chars", self.max_summary_chars, 1)
        need_low("comment_max_chars", self.comment_max_chars, 1)
        need_low("snippet_chars", self.snippet_chars, 1)
        need_low("max_get_batch", self.max_get_batch, 1)
        need_range("default_top_k", self.default_top_k, 1, 20)
        # Listing limits (lsb-0013): the ceiling is a surface parameter now, so
        # positivity is checked per field and the order default ≤ MCP ≤ REST is
        # a relational check right below.
        need_low("default_list_limit", self.default_list_limit, 1)
        need_low("list_max_limit_mcp", self.list_max_limit_mcp, 1)
        need_low("list_max_limit_rest", self.list_max_limit_rest, 1)
        # The `warnings` job (No12, 3.2.0): warnings page ceilings (§7.1).
        need_low("warnings_max_limit_mcp", self.warnings_max_limit_mcp, 1)
        need_low("warnings_max_limit_rest", self.warnings_max_limit_rest, 1)
        if self.default_list_limit > self.list_max_limit_mcp:
            errors.append(
                "  - default_list_limit: default page size above the MCP "
                "listing ceiling list_max_limit_mcp — the default page would "
                "be rejected by the MCP listing"
            )
        if self.list_max_limit_mcp > self.list_max_limit_rest:
            errors.append(
                "  - list_max_limit_mcp: MCP listing ceiling above the REST one "
                "list_max_limit_rest — MCP cannot be looser than REST"
            )

        # --- thresholds and dedup ---
        need_range("score_threshold", self.score_threshold, 0.0, 1.0)
        need_range("dedup_similarity", self.dedup_similarity, 0.0, 1.0)
        # Background dedup (Phase 8, step 2.1): the candidate threshold must
        # not exceed the duplicate threshold — otherwise candidates are found
        # but never recognized as duplicates.
        need_range(
            "dedup_candidate_similarity", self.dedup_candidate_similarity, 0.0, 1.0
        )
        # The 50 cap protects brute-force KNN from a crazy window (NFR-5).
        need_range("dedup_candidate_top_n", self.dedup_candidate_top_n, 1, 50)
        if self.dedup_candidate_similarity > self.dedup_similarity:
            errors.append(
                "  - dedup_candidate_similarity: порог кандидата выше порога "
                "«дубль» dedup_similarity — найденных кандидатов "
                "нельзя будет признать дублем"
            )
        need_low("rrf_k", self.rrf_k, 1)

        # --- note links (lsb-0010): ceiling and pool >= 1, threshold 0..1 ---
        need_low("link_top", self.link_top, 1)
        need_low("link_pool", self.link_pool, 1)
        need_range("link_lazy_threshold", self.link_lazy_threshold, 0.0, 1.0)
        # Level 1 (lsb-0010-02): the link threshold is above the "lazy graph"
        # one (FR-2.3, arch §3.1) — otherwise a level-1 link is weaker than
        # its own fallback.
        need_range("link_cosine_threshold", self.link_cosine_threshold, 0.0, 1.0)
        need_low("link_entities_min_common", self.link_entities_min_common, 1)
        need_low(
            "link_entities_min_word_chars", self.link_entities_min_word_chars, 1
        )
        if self.link_lazy_threshold > self.link_cosine_threshold:
            errors.append(
                "  - link_lazy_threshold: порог ленивого уровня выше порога "
                "связи link_cosine_threshold — связь уровня 1 оказалась бы "
                "строже фолбэка уровня 0"
            )

        # --- framework background jobs (lsb-0014): env-scheduled ---
        # The `links` job (lsb-0010-03): interval >= 30 (cheap, but not a busy
        # loop), batch >= 1 (a zero batch would never drain the backfill).
        need_low("job_links_interval_sec", self.job_links_interval_sec, 30)
        need_low("job_links_batch", self.job_links_batch, 1)
        # The `nodes` job (lsb-0011-01): the default sweep — interval >= 30,
        # batch >= 1 (the shared per-run processing budget).
        need_low("job_nodes_interval_sec", self.job_nodes_interval_sec, 30)
        need_low("job_nodes_batch", self.job_nodes_batch, 1)
        # The classifier budget (lsb-0011-02): >= 1 and not above the shared
        # batch — otherwise a run makes more model calls than processings.
        need_low(
            "job_nodes_classifier_budget", self.job_nodes_classifier_budget, 1
        )
        if self.job_nodes_classifier_budget > self.job_nodes_batch:
            errors.append(
                "  - job_nodes_classifier_budget: бюджет классификатора выше "
                "общего батча job_nodes_batch — за прогон столько заметок "
                "не разберётся"
            )

        # The `warnings` job (No12, 3.2.0): interval >= 30, batch >= 1, the
        # classifier budget >= 1 and not above the batch, confidence 0..1.
        need_low("job_warnings_interval_sec", self.job_warnings_interval_sec, 30)
        need_low("job_warnings_batch", self.job_warnings_batch, 1)
        need_low(
            "job_warnings_classifier_budget", self.job_warnings_classifier_budget, 1
        )
        if self.job_warnings_classifier_budget > self.job_warnings_batch:
            errors.append(
                "  - job_warnings_classifier_budget: бюджет классификатора выше "
                "общего батча job_warnings_batch — за прогон столько заметок "
                "не разберётся"
            )
        need_range(
            "warnings_min_confidence", self.warnings_min_confidence, 0.0, 1.0
        )

        # The `terms-normalize` job (No32, 3.2.0): interval >= 30, batch >= 1,
        # clustering threshold 0..1, contexts-excess threshold >= 2.
        need_low("job_terms_interval_sec", self.job_terms_interval_sec, 30)
        need_low("job_terms_batch", self.job_terms_batch, 1)
        need_low(
            "job_terms_contexts_excess", self.job_terms_contexts_excess, 2
        )
        need_range(
            "job_terms_cluster_similarity",
            self.job_terms_cluster_similarity,
            0.0,
            1.0,
        )

        # --- vectorization / summarization ---
        need_low("embedding_dim", self.embedding_dim, 1)
        # No49: a budget of 0 = "as much as fits" (auto), hence the minimum 0;
        # a context of 0 = unknown (degradation); reserve/cap >= 1; the reserve
        # must be smaller than the declared context (otherwise it eats the
        # whole context).
        need_low("summary_num_predict", self.summary_num_predict, 0)
        need_low("merge_num_predict", self.merge_num_predict, 0)
        need_low("summary_context_tokens", self.summary_context_tokens, 0)
        need_low("summary_timeout_sec", self.summary_timeout_sec, 1)
        # The LLM dedup judge (Phase 8, step 3.1; Phase 11 — judge_*): a budget
        # of 0 = auto, a context of 0 = unknown, timeout >= 1 (NFR-6).
        need_low("judge_num_predict", self.judge_num_predict, 0)
        need_low("judge_context_tokens", self.judge_context_tokens, 0)
        need_low("slot_prompt_reserve_tokens", self.slot_prompt_reserve_tokens, 1)
        need_low("slot_max_output_tokens", self.slot_max_output_tokens, 1)
        need_low("judge_timeout_sec", self.judge_timeout_sec, 1)
        # LSBN-T-33: a reasoning budget of 0 = "as much as fits" (auto), hence
        # the minimum 0; operation-level None = "not set" — checked below.
        need_low("slot_reasoning_num_predict", self.slot_reasoning_num_predict, 0)
        judge_reasoning = self.judge_reasoning_num_predict
        if judge_reasoning is not None:
            need_low("judge_reasoning_num_predict", judge_reasoning, 0)
        # No49 (arch §8): model-type thresholds — minimum 1; relationally, the
        # thinking threshold cannot exceed the auto-budget defensive cap.
        need_low(
            "slot_reasoning_min_output_tokens",
            self.slot_reasoning_min_output_tokens,
            1,
        )
        need_low("slot_min_timeout_sec", self.slot_min_timeout_sec, 1)
        if self.slot_reasoning_min_output_tokens > self.slot_max_output_tokens:
            errors.append(
                "  - slot_reasoning_min_output_tokens: порог "
                f"{self.slot_reasoning_min_output_tokens} выше защитного потолка "
                f"slot_max_output_tokens ({self.slot_max_output_tokens}) — "
                "предупреждение недостижимо"
            )
        for context_field, context in (
            ("summary_context_tokens", self.summary_context_tokens),
            ("judge_context_tokens", self.judge_context_tokens),
        ):
            if context > 0 and self.slot_prompt_reserve_tokens >= context:
                errors.append(
                    "  - slot_prompt_reserve_tokens: запас "
                    f"{self.slot_prompt_reserve_tokens} не меньше контекста "
                    f"{context_field} ({context}) — авто-бюджет всегда упирался бы в минимум"
                )

        # --- operations over slots (LSBN-F-34): slot rules, option fields ---
        # Set flat operation envs are validated like slot fields: budget >= 0
        # (0 = auto), timeout >= 1, context >= 0, and the reserve must not eat
        # the context. None = "not set" — nothing to check (the slot inherits).
        for prefix in (
            "summarize",
            "title",
            "merge",
            "split",
            "classify",
            "describe",
            "structure_judge",
            "embed",
        ):
            num_predict = getattr(self, f"{prefix}_num_predict")
            if num_predict is not None:
                need_low(f"{prefix}_num_predict", num_predict, 0)
            # LSBN-T-33: the operation reasoning budget — >= 0 (None = not set).
            reasoning_num_predict = getattr(self, f"{prefix}_reasoning_num_predict", None)
            if reasoning_num_predict is not None:
                need_low(f"{prefix}_reasoning_num_predict", reasoning_num_predict, 0)
            timeout_sec = getattr(self, f"{prefix}_timeout_sec")
            if timeout_sec is not None:
                need_low(f"{prefix}_timeout_sec", timeout_sec, 1)
            context_tokens = getattr(self, f"{prefix}_context_tokens")
            if context_tokens is not None:
                need_low(f"{prefix}_context_tokens", context_tokens, 0)
                if self.slot_prompt_reserve_tokens >= context_tokens:
                    errors.append(
                        f"  - slot_prompt_reserve_tokens: запас "
                        f"{self.slot_prompt_reserve_tokens} не меньше контекста "
                        f"{prefix}_context_tokens ({context_tokens}) — "
                        "авто-бюджет всегда упирался бы в минимум"
                    )

        # --- chunked indexing (Phase 7) ---
        if self.text_splitter.strip().lower() != "tiktoken":
            errors.append("  - text_splitter: only 'tiktoken' is supported")
        need_range("chunk_size", self.chunk_size, 64, 16384)
        # overlap < window (otherwise the windows never shift / no progress);
        # the upper bound is computed from chunk_size — relational checks.
        need_range("chunk_overlap", self.chunk_overlap, 0, self.chunk_size - 1)
        need_range("chunk_min_target", self.chunk_min_target, 1, self.chunk_size)
        need_low("embedding_batch_size", self.embedding_batch_size, 1)
        need_low("embedding_concurrent_requests", self.embedding_concurrent_requests, 1)

        # --- background operations (0 is allowed: unit tests use a pause-free mode) ---
        need_low("pending_retry_sec", self.pending_retry_sec, 0)

        # --- migration slot cadence (3.4.1, LSBN-T-27): a batch of 1..100
        # notes (more — the job monopolizes the slot for minutes), a pause of
        # 0..86400 s (0 — drain without yielding the slot; a day is the sane
        # rollout-window ceiling).
        need_range("migration_batch", self.migration_batch, 1, 100)
        need_range("migration_pause_sec", self.migration_pause_sec, 0, 86400)

        # --- slot breaks (No39): pause >= 30 (cheap, but not a busy loop),
        # open threshold >= 1, hint throttling >= 0 (0 — every call).
        need_low("slot_pause_sec", self.slot_pause_sec, 30)
        need_low("slot_pause_after", self.slot_pause_after, 1)
        need_low("slot_hint_every_sec", self.slot_hint_every_sec, 0)
        # --- pair attempt counter (No39 §6.8, no39-02): limit >= 1 ---
        need_low("slot_max_attempts", self.slot_max_attempts, 1)
        # no39-04: a soft reachability check of the attention threshold. If the
        # attention threshold is NOT above the pause threshold
        # (`slot_max_attempts <= slot_pause_after`), a pair looks hopeless only
        # after several "pause → knock → refuse" cycles, and the "requires
        # attention" state arrives later than the operator expects. Not fatal:
        # both values are the operator's will, the start does not fail.
        if self.slot_max_attempts <= self.slot_pause_after:
            logging.getLogger("app").warning(
                "порог внимания не выше порога паузы: пара будет казаться "
                "безнадёжной только после N циклов паузы; подними "
                "SLOT_MAX_ATTEMPTS или снизь SLOT_PAUSE_AFTER",
                extra={"event": "slot_attempts_below_pause"},
            )

        # --- decision slot (LSBN-F-12, arch §8): timeout >= 1, the question
        # ceiling within 1..64 (the model limit), the label ceiling >= 2, the
        # threshold and the zone within 0..1. With the slot enabled, a non-empty
        # http(s) address and a model are required, and threshold ± zone must
        # stay inside 0..1 (otherwise the zone leaves the bounds).
        need_low("decision_timeout_sec", self.decision_timeout_sec, 1)
        need_range("decision_max_questions", self.decision_max_questions, 1, 64)
        need_low("decision_max_labels", self.decision_max_labels, 2)
        need_range("decision_threshold", self.decision_threshold, 0.0, 1.0)
        need_range(
            "decision_uncertainty_zone", self.decision_uncertainty_zone, 0.0, 1.0
        )
        if self.decision_enabled:
            if not self.decision_base_url.strip().startswith(("http://", "https://")):
                errors.append(
                    "  - decision_base_url: ожидается URL с http:// или https:// "
                    "(обязателен при DECISION_ENABLED=true)"
                )
            if not self.decision_model.strip():
                errors.append(
                    "  - decision_model: не может быть пустой при DECISION_ENABLED=true"
                )
            if not 0.0 <= self.decision_threshold - self.decision_uncertainty_zone:
                errors.append(
                    "  - decision_threshold: порог минус зона "
                    f"{self.decision_threshold - self.decision_uncertainty_zone} "
                    "уходит ниже 0 — зона неопределённости вне 0..1"
                )
            if self.decision_threshold + self.decision_uncertainty_zone > 1.0:
                errors.append(
                    "  - decision_threshold: порог плюс зона "
                    f"{self.decision_threshold + self.decision_uncertainty_zone} "
                    "уходит выше 1 — зона неопределённости вне 0..1"
                )

        # --- version history (No41): 0 = "unlimited" for the version ceiling
        # and retention (the default is keep everything); the delta threshold
        # >= 1 (otherwise line-delta encoding of long fields would never turn
        # on).
        need_low("history_max_versions", self.history_max_versions, 0)
        need_low("history_retention_days", self.history_retention_days, 0)
        need_low("history_delta_inline_max", self.history_delta_inline_max, 1)

        # --- backup (NFR-3) ---
        need_low("backup_interval_sec", self.backup_interval_sec, 1)
        need_low("backup_keep", self.backup_keep, 1)
        # No14 (arch §8): 0 — automatic verification is off (manual only).
        need_low("backup_verify_every", self.backup_verify_every, 0)
        # No14 (arch §8): the ship-command timeout >= 1 (irrelevant when empty).
        need_low("backup_ship_timeout_sec", self.backup_ship_timeout_sec, 1)
        if not self.backup_dir.strip():
            errors.append("  - backup_dir: the path cannot be empty")
        if not self.db_path.strip():
            errors.append("  - db_path: the path cannot be empty")

        # --- new-release hint (No45, arch §8): interval >= 60, hint period
        # >= 0 (0 — one per process), text limit 1..400; with the check enabled
        # at least one source is required (otherwise a fatal start, NFR-6).
        need_low("update_check_interval_sec", self.update_check_interval_sec, 60)
        need_low("update_hint_every_sec", self.update_hint_every_sec, 0)
        need_low("update_hint_max_chars", self.update_hint_max_chars, 1)
        if self.update_hint_max_chars > 400:
            errors.append(
                "  - update_hint_max_chars: above 400 — the hint text bloats "
                "the response (arch §8), got "
                f"{self.update_hint_max_chars}"
            )
        if self.update_check_enabled and not (
            self.update_source_file.strip() or self.update_check_url.strip()
        ):
            errors.append(
                "  - update_check_enabled: the check is enabled without a "
                "source — set UPDATE_SOURCE_FILE (a local JSON) or "
                "UPDATE_CHECK_URL"
            )

        # --- namespaces (Phase 10) ---
        need_range(
            "namespace_auto_move_min_confidence",
            self.namespace_auto_move_min_confidence,
            0.0,
            1.0,
        )
        need_range(
            "namespace_promotion_min_confidence",
            self.namespace_promotion_min_confidence,
            0.0,
            1.0,
        )
        need_range(
            "namespace_synonym_similarity", self.namespace_synonym_similarity, 0.0, 1.0
        )
        need_low("namespace_promotion_threshold", self.namespace_promotion_threshold, 1)
        need_low("namespace_auto_max_per_day", self.namespace_auto_max_per_day, 1)
        need_low(
            "namespace_max_leaves_per_domain", self.namespace_max_leaves_per_domain, 1
        )
        need_low("namespace_groom_min_notes", self.namespace_groom_min_notes, 0)
        # No28: the match threshold >= 0 (2.0 — the default), the node ceiling
        # 1..2 (FR-1.1), the comparison text — at least one character.
        need_range(
            "memory_node_hint_min_score", self.memory_node_hint_min_score, 0.0, 1e6
        )
        need_low("memory_node_hint_max_nodes", self.memory_node_hint_max_nodes, 1)
        if self.memory_node_hint_max_nodes > 2:
            errors.append(
                "  - memory_node_hint_max_nodes: больше 2 узлов — ответ раздувается "
                "(FR-1.1, arch §13.1), получено "
                f"{self.memory_node_hint_max_nodes}"
            )
        need_low("memory_node_hint_text_chars", self.memory_node_hint_text_chars, 1)
        # No30: limit/pool/text >= 1, threshold 0..1; the hint ceiling 1..3
        # (arch §6.4 — the answer is <= 2, 3 is allowed with margin) and the
        # pool is not smaller than the limit.
        need_low("memory_similar_limit", self.memory_similar_limit, 1)
        need_low("memory_similar_pool", self.memory_similar_pool, 1)
        need_low("memory_similar_text_chars", self.memory_similar_text_chars, 1)
        need_range("memory_similar_threshold", self.memory_similar_threshold, 0.0, 1.0)
        if self.memory_similar_limit > 3:
            errors.append(
                "  - memory_similar_limit: больше 3 — ответ раздувается "
                f"(arch §6.4), получено {self.memory_similar_limit}"
            )
        if self.memory_similar_pool < self.memory_similar_limit:
            errors.append(
                "  - memory_similar_pool: меньше memory_similar_limit — "
                f"кандидатов меньше, чем подсказок, получено {self.memory_similar_pool}"
            )

        # --- 3.0.0 areas (substrate: skills / terms / user) ---
        for field in (
            "skill_name_max_chars",
            "skill_description_max_chars",
            "skill_steps_max_chars",
            "skill_text_max_chars",
            "skill_example_max_chars",
            "skill_extra_field_max_chars",
            "skill_extra_total_max_chars",
            "instruction_template_max_chars",
            "skill_announce_max_chars",
            "skill_announce_description_chars",
            "term_max_chars",
            "term_context_max_chars",
            "term_definition_max_chars",
            "term_contexts_hint_limit",
            "user_name_max_words",
            "user_body_max_chars",
            "user_search_excerpt_chars",
            "logs_action_max_chars",
            "logs_reason_max_chars",
            "logs_entity_ref_max_chars",
            "logs_outcome_max_chars",
            "logs_verified_max_chars",
            "logs_search_excerpt_chars",
        ):
            need_low(field, getattr(self, field), 1)
        # Similarity thresholds — cosine/trigrams: the 0..1 scale (like dedup).
        need_range("skill_synonym_similarity", self.skill_synonym_similarity, 0.0, 1.0)
        need_range("term_context_similarity", self.term_context_similarity, 0.0, 1.0)
        need_range("user_similar_strong", self.user_similar_strong, 0.0, 1.0)
        need_range("user_similar_weak", self.user_similar_weak, 0.0, 1.0)
        # Relational checks (modeled after dedup_candidate_similarity):
        # the extra budget total must fit at least one class field;
        # the announce description excerpt stays within the block budget;
        # the weak "similar" threshold is not above the strong "same fact" —
        # otherwise the middle band is empty and the hint never arrives.
        if self.skill_extra_field_max_chars > self.skill_extra_total_max_chars:
            errors.append(
                "  - skill_extra_field_max_chars: лимит одного поля extra выше "
                "суммарного skill_extra_total_max_chars — ни одно поле не влезет"
            )
        if self.skill_announce_description_chars > self.skill_announce_max_chars:
            errors.append(
                "  - skill_announce_description_chars: обрезка description выше "
                "бюджета блока skill_announce_max_chars"
            )
        if self.user_similar_weak > self.user_similar_strong:
            errors.append(
                "  - user_similar_weak: слабый порог выше сильного "
                "user_similar_strong — средняя зона окажется пустой"
            )

        # --- misc ---
        if not self.author_default.strip():
            errors.append("  - author_default: не может быть пустым")

        if errors:
            raise ValueError(
                "настройки вне допустимых диапазонов (NFR-6):\n" + "\n".join(errors)
            )
        return self


def load_settings() -> Settings:
    """Load and validate the settings from the environment.

    Raises:
        ConfigError: the environment is incomplete or invalid (for example, an
            empty MCP_AUTH_TOKEN) — the calling code must terminate the process.
    """
    try:
        return Settings()
    except ValidationError as exc:
        details = "\n".join(
            (
                f"  - {'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
                if error["loc"]
                else f"{error['msg']}"
            )
            for error in exc.errors()
        )
        raise ConfigError(
            "Некорректная конфигурация окружения (REQUIREMENTS §8):\n" + details
        ) from exc


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """The settings singleton for the process lifetime."""
    return load_settings()
