# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.0.0] - 2026-09-11

Release 3.0.0 — "Skills and knowledge": three new knowledge areas (procedural skills, terminology, facts about the user) on a shared isolated substrate — one database, separate tables and indexes per area — with MCP tools, operator REST mirrors and background vectorization.

### Added

- **Area substrate**: `skills`, `terms` and `user_facts` live in the same SQLite database as notes, each with its own tables, FTS5 (trigram) and `vec0` indexes; the schema is created idempotently at startup, so an install of any previous version upgrades without manual migrations. Areas are isolated by design: no area query reads notes or another area, and note search never returns area records.
- **Skills area** (lsb-0007): `skills_search` / `skills_list` / `skills_get` / `skills_save` / `skills_delete`. A skill is a stored procedure with a validated form: `name` ≤ 65 chars (≤ 5 words recommended), `description` ≤ 250, `steps` ≤ 500, `text` ≤ 4000, optional `example` ≤ 1000 and optional class fields (trigger, mode, preconditions, fallbacks, invariant, exceptions, guardrails, references, output_contract, behavior_contract; ≤ 500 chars each, ≤ 2000 together). Every edit keeps the previous version as a copy (`skill_versions`, operator-only via `GET /skills/{id}/versions`); a too-similar creation is refused with a hint pointing at the existing skill (cosine 0.90). The global "how to execute steps" template lives once per area (`instruction_template`). Bodies never enter the context unless `skills_get` asks for them.
- **Skills announce at initialize**: MCP instructions carry a compact tail with the available skills (id — name: description, description cut to 120 chars, block budget 2000 chars, `(+n more — skills_list)` on overflow), rebuilt on every `initialize`, so a new chat sees the current registry. The built-in "Create skills" procedure is seeded on first start.
- **Terms area** (lsb-0008): `terms_search` / `terms_save` / `terms_get`. A term is keyed by (term + context) with a mandatory context; the same term in a different context is a new sense and never overwrites the old one. Search returns ALL senses with their contexts; with no exact term the closest senses by meaning are returned and marked as not an exact match. A context too close to an existing one is refused with a hint pointing at the existing wording; successful writes return the term's senses and the contexts already used in the memory. There is no listing — search is the only way.
- **User area** (lsb-0009): `user_search` / `user_save` / `user_update` / `user_delete` / `user_get` for atomic facts (`name` ≤ 5 words + `body` ≤ 1200 chars). A strong overlap with an existing fact is refused with a hint pointing at it (refine via `user_update`, or save a new fact as a separate record); every successful save repeats the atomicity rule. Search returns ≤ 300-char excerpts; the full body is one `user_get` away. Nothing is injected into the instructions at initialize.
- **Operator REST mirrors**: `/skills` (plus `/skills/search`, `/skills/instruction-template`, `/skills/{id}/versions`), `/terms` (plus `/terms/search`) and `/user-facts` (plus `/user-facts/search`) — the same service layer and Bearer token as MCP, full contracts, status codes 201/200/401/404/409/422.
- **Area limits and thresholds** are environment-tunable and validated at startup (form limits; anti-synonymy 0.90; context similarity 0.75; fact similarity 0.85/0.55; announce budget 2000/120; search excerpt 300).

### Changed

- **MCP tool surface**: 8 → **21** tools (`memory_*` + `skills_*` + `user_*` + `terms_*`); existing tools and their outputs are unchanged.
- **Worker**: a fourth independent background loop (`areas`) vectorizes area records with its own back-off; writes stay instant (`vector_status=pending`), and an embedding failure never blocks a write or a search (full-text fallback with a warning).
- **MCP instructions**: the manifest and the namespace map are unchanged; the skills announce is appended as a tail and refreshed per `initialize` (the mechanism returns from the reverted 2.2 profile block — without any user data).
- **Vectorization inputs**: `name + description` (skills), `term + context + definition` (terms), `name + body` (user facts); changing the embedding model or dimension rebuilds the area indexes as well.

### Fixed

- **Area hybrid search returned neighbours regardless of relevance**: `vec0` KNN always returns the k nearest records, so any non-empty area answered any query and the documented "probe" semantics (empty result = no such record, soft hint) was unreachable. Vector hits are now gated by `SCORE_THRESHOLD`, the same calibration the note search uses.

### Upgrade

- Drop-in: start the new image over the existing database. The schema is created and seeded idempotently, notes, namespaces and their indexes are untouched, and the new areas start empty (except the seeded "Create skills" procedure and the instruction template). Reverting is a matter of restoring the pre-upgrade database snapshot.

## [2.2.1] - 2026-09-10

Patch release — upgrade break (lsbdef-0006): upgrading an install from v2.1.x
to v2.2.x with runtime prompts (`prompts/`) from the previous seed no longer
crashes the container into a restart-loop.

### Fixed

- **Auto-migration of prompt seeds** (lsbdef-0006): `PromptRegistry` now
  stamps the seed version in a sidecar `seed_meta.json` and, on startup,
  rewrites any editable prompt file that is byte-identical to a previously
  seeded or known legacy seed (v2.1.x RU canon) with the current canon.
  Operator-edited files are never overwritten.
- **FATAL remediation hint**: the `ConfigError` for a `judge_system` file
  missing the `DUPLICATE`/`NOT DUPLICATE` markers now tells the operator to
  delete the file to re-seed, or to add both markers.

### Changed

- **Upgrade note**: when upgrading from v2.1.x, unmodified prompt files are
  migrated automatically; files edited by hand must be updated to the new
  canon manually (see [Installation](docs/INSTALL.md) → Upgrading).

## [2.2.0] - 2026-09-09

Release 2.2 — "Making life easier for models": notes are easier to find by title, search and listing shape is controlled with a single knob, context is saved, and the prompts the model reads are clear.

### Added

- **Title index + title search** (lsb-0001): `notes.title` is included in the vector index (title+text are vectorized together) and in the full-text index `notes_fts`; a note can now be found by its title.
- **Unified search & list** (lsb-0001): one search method with `memory_search(mode=semantic|title)` and one listing method `memory_list(detail=titles|summaries)`; the namespace path is shown in every output.
- **Chunk reading** (lsb-0003): `memory_get(id, query?, chunk?, limit?)` — read a note in full (as before) or just the needed chunk, either by semantic query (`query`) or by index with pagination (`chunk`/`limit`); soft refusals when `query` and `chunk` are combined and when `chunk` is out of range.
- **Metadata edit without text** (lsb-0004): `memory_update` can change `title`/`summary`/`namespace` without rewriting the text; `summary` is kept as is (not regenerated).
- **Temporary storage (TTL)** (lsb-0004): `expires_at` can be set on save; expired notes are removed by background cleanup (`note_expirations`).
- **Namespace depth 3 + model-created namespaces** (lsb-0005): maximum depth raised from 2 to 3; models create nodes at any level via `memory_namespace_create` (description required); anti-synonymy check on node creation; audit of auto-generated descriptions.
- **English canon for model-facing texts** (lsb-0006): all prompts, hints and MCP instructions moved to an English canon and rewritten for clarity (tool descriptions, soft-refusal hints, instructions manifest).

### Fixed

- **lsbdef-0001**: flaky live classifier/summarization tests — the JSON parser now extracts valid JSON from responses wrapped in thinking/response tags.
- **lsbdef-0002**: stabilized the flaky server-independence test (`test_servers_are_independent`).
- **lsbdef-0003**: the lsb-0005 E2E script updated for the EN hints after the translation.
- **lsbdef-0004**: flaky live tests caused by structure-judge non-determinism — retry up to 3 attempts.
- **lsbdef-0005**: MCP session drop under background load — MCP timeouts (30 s connect/write/pool, 300 s read) in the E2E scripts.

## [2.1.1] - 2026-09-05

### Fixed

- **The background pipeline now fully restarts after `memory_update`**
  (requirements-vs-code audit, 2026-09-05): `NoteService.update()` also
  resets the classification marks (`classified_at`, `domain_hint`,
  `subdomain_hint`, `confidence`) in the same UPDATE. A rewritten default
  note is re-classified by the groomer (and can auto-move into an existing
  domain again), and stale hints no longer feed the namespace promotion
  trigger until a fresh classification arrives. Re-vectorization and
  re-summarization after an update were already in place; new tests lock
  the entire chain.
- **Docs**: the `/health` docstring in `rest.py` now matches the actual
  `judge_ok` semantics (an unreachable judge stays `null` until the first
  real call fails).
- **Phase-2 refactor** (code audit, 2026-09-05/06; 13 pools, 828 tests):
  - **Worker notes-queue vector guard**: the embedding loop writes the
    full-text vector and flips `vector_status` to `ok` only if the note is
    unchanged since it was read (verified in the same transaction) — a
    mid-flight `memory_update` can no longer leave a stale vector marked
    as fresh.
  - **Classifier robustness**: a non-slug `domain_hint` is rejected as an
    invalid classification, and an unexpected classifier failure is
    contained with a warning instead of killing the summary loop.
  - **Atomic write paths**: the literal-dedup check runs inside the save
    transaction (no TOCTOU duplicate); `update()` drops the stale
    full-text vector (pending notes search FTS-only, as documented);
    grooming/move is a single UPDATE guarded by `namespace = 'default'`;
    duplicate merge (earlier update + later delete) is one transaction.
  - **Worker loop supervisor**: an unexpected error in a loop iteration is
    logged with a traceback and the loop continues instead of dying until
    restart; queue re-checks after `clear()` remove lost wakeups.
  - **worker_jobs hygiene**: queue index, 7-day retention for done jobs,
    one-time DDL; background party sizes follow `EMBEDDING_BATCH_SIZE`.
  - **Security/robustness smalls**: Bearer compared on bytes (a non-ASCII
    header yields 401, not 500), query strings truncated to 80 chars in
    the access log, a namespace rename race maps to 409, L2-normalized
    embeddings in the synonym prefilter, failed MCP tool calls logged
    (`failed=true` + latency), ambiguous `memory_get` rejected loudly,
    partial backup snapshot removed on copy failure.
  - **Performance: dedup query plans under a namespace filter**
    (`NOT INDEXED` + `CROSS JOIN`): at 50k notes a save into a large
    namespace took 4–12+ s (the planner drove both dedup queries through
    the low-selectivity namespace index), now save p95 ≈ 0.34 s
    (bench, 2026-09-06).
  - **Dead legacy code removed** in dedup/search/storage; failed MCP tool
    calls are visible in the JSON logs.
  - Tests: two timing-sensitive tests stabilized, `BACKUP_DIR` points to a
    tmp dir in the test env, a dedup plan-hint regression test added;
    full suite: 828 passed.

## [2.1.0] - 2026-09-05

### Added

- **Per-slot LLM providers** (Phase 11): each of the three external-call slots
  (embedding / summary / judge) is configured independently with its own
  provider (`ollama` — default, or an OpenAI-compatible API), base URL, model
  and optional API key. New env: `EMBEDDING_PROVIDER`, `SUMMARY_PROVIDER`,
  `JUDGE_PROVIDER`, `EMBEDDING_BASE_URL`, `SUMMARY_BASE_URL`,
  `JUDGE_BASE_URL`, `EMBEDDING_API_KEY`, `SUMMARY_API_KEY`, `JUDGE_API_KEY`.
- **Unified LLM client** (`app/services/llm_client.py`): a single transport
  layer for both providers — `POST /api/chat`|`/api/embed` (ollama) and
  `POST /v1/chat/completions`|`/v1/embeddings` (openai), with `max_tokens`,
  Bearer (only when a key is set), per-provider connect timeouts, and a
  single retry for the embedder on transient failures (incl. `429`).
- **Startup provider check** (decision №5): a lightweight liveness GET to each
  slot at startup (no generation). `200` → `last_attempt_ok=true` in `/health`;
  `401/403` → fatal refusal with a hint about `{SLOT}_API_KEY`; network/5xx →
  WARN and degraded start (NFR-3).
- **Editable prompts** (decision №7): three system prompts
  (`summary_system`, `summary_merge_system`, `judge_system`) are seeded as
  files under `PROMPTS_DIR` when set; existing files are never overwritten;
  an empty file falls back to the built-in default; `judge_system` is
  validated for the judge markers (DUPLICATE / NOT DUPLICATE) at startup (fatal if missing).
- **Note titles** (decision №9): `notes.title` (nullable); new notes require a
  title of ≤ 5 words (fail + hint otherwise); titles appear in `memory_search`
  and `memory_list` (not in `memory_get`); the worker back-fills titles for
  legacy null-title notes; dedup merge keeps the earlier note's title.
- **Worker loops by slot** (decision №10): three independent background loops
  (embedding / summary / judge) with per-loop back-off; job dependencies are
  preserved (dedup after vectorization, merge after the judge verdict, etc.).
- **Reindex on embedding change**: the meta key `embedding_provider` is
  recorded; changing provider/model/dim triggers an automatic full reindex on
  startup (all notes → `pending`).

### Changed

- **Renamed env** (breaking): `OLLAMA_BASE_URL` → `EMBEDDING_BASE_URL`,
  `SUMMARY_OLLAMA_BASE_URL` → `SUMMARY_BASE_URL`,
  `DEDUP_JUDGE_OLLAMA_BASE_URL` → `JUDGE_BASE_URL`; the judge block
  `DEDUP_JUDGE_MODEL/THINK/NUM_PREDICT/TIMEOUT_SEC` →
  `JUDGE_MODEL/THINK/NUM_PREDICT/TIMEOUT_SEC`.
- **`keep_alive` removed** from all request payloads (decision №6): model
  residency is managed by the server (`OLLAMA_KEEP_ALIVE` on the Ollama side).
- **Public docs in English**: brief `README.md`, new `docs/INSTALL.md` and
  `docs/CONFIG.md`; `CHANGELOG.md` (Keep a Changelog) is the source of future
  release texts.

### Fixed

- `memory_search` returns the real note title (follow-up, decision №9a).

## [2.0.0] - 2026-09-04

### Added

- **Hierarchical namespaces** (Phase 10): the store is split into large
  sections (max 2 levels: `domain`, `domain/subdomain`). New tool
  `memory_namespaces`; optional `namespace` in save/update/search/list;
  namespace labels in outputs; the map is exposed in MCP instructions.
- **Background grooming**: a classifier labels default notes
  (`domain_hint`/`subdomain_hint`/`confidence`) and auto-moves confident ones
  into existing nodes; a structure judge gates auto-created leaves
  (anti-synonymy, meaningfulness); `provisional` nodes participate in search
  on par with `confirmed`.
- **Chunked vectorization** (Phase 7): vectors are built per chunk
  (tiktoken, `cl100k_base`), improving search for facts in the middle of long
  notes; chunk vectors live in a separate vec0 table with namespace
  partitions.

### Changed

- MCP outputs are more compact (Phase 9): summaries and metadata instead of
  full texts; full contracts are available via REST.

## [1.0.0] - 2026-08-29

### Added

- Initial release: self-hosted MCP memory server for LLMs — hybrid
  vector + full-text search, note CRUD, background vectorization and
  summarization, Bearer auth, `/health`, Docker deployment.
