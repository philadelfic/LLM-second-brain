# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
  validated for the `ДУБЛЬ`/`НЕ ДУБЛЬ` markers at startup (fatal if missing).
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
