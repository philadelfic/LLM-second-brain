# LLM Second Brain

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)
![Docker](https://img.shields.io/badge/Docker-ready-2496ed.svg)

A self-hosted **long-term memory server** for LLMs running in harnesses
(primarily Open WebUI). Models get MCP access to a shared knowledge store —
notes, three knowledge areas (skills, terms, facts about the user) and the
assistant log: they can search it (hybrid vector + full-text), read, write,
update and delete records, and follow the version history of what changed.

## What it is

- **One Docker container**, self-hosted, non-root.
- **MCP Streamable HTTP** (`/mcp`, natively supported by Open WebUI) with a
  Bearer token.
- **31 MCP tools**: 9 `memory_*` for notes, comments and namespaces; 5
  `skills_*`, 5 `user_*` and 3 `terms_*` for the knowledge areas; 6 `logs_*`
  for the assistant log; `history_list` / `history_get` for version history;
  `warnings_list` for the "notes not in their place" job.
- **Storage**: one SQLite database + `sqlite-vec` (vector search) + FTS5
  (full-text), merged via Reciprocal Rank Fusion; every knowledge area has
  its own tables and indexes, isolated from notes and from each other.
- **Three knowledge areas next to notes** (since v3.0): **skills** — stored
  procedures, read in full only when needed, with a version archive;
  **terms** — terminology keyed by (term + context), every sense returned and
  never overwritten; **user** — atomic facts about the user with dedup hints.
  Each area has MCP tools and operator REST mirrors.
- **Assistant log next to notes** (since v3.2): the assistant's own record of
  "what I did, when and why" — the changed object and the result, what was
  checked; its own tables and indexes, its own `logs_*` tools, a compose flag,
  isolated from note search.
- **Version history** (since v3.2): every content change to a note, an area
  record or an assistant log is versioned with line-level deltas for long texts
  and a full baseline for the first version; the operator reads it over REST
  (`GET /history/<entity>/<id>`) and can roll back — a rollback writes a NEW
  version and never rewrites history; the agent gets `history_list` and
  `history_get`.
- **Slots that wait instead of failing** (since v3.2): every slot call has a
  typed outcome (`unavailable` / `auth` / `timeout` / `empty` / `truncated` /
  `off_contract`); a broken slot opens a break, does one cheap probe per pause
  and resumes by itself, while its work stays visible as `waiting` instead of
  burning calls. `decision` is a third slot type: the model answers structured
  questions instead of generating text, and the product falls back to the
  generative path on `unavailable` / `unsupported` / `low_confidence`.
- **Output budget "as much as fits"** (since v3.2): a slot spends the model's
  context minus the prompt (a reserve plus a protective ceiling) instead of a
  fixed small cap; `done_reason` / `finish_reason`, thinking tokens and the
  applied budget land in the journal, and an empty answer is distinguished from
  a truncated one.
- **Measured injected text** (since v3.2): one token counter measures and guards
  everything the server injects into the agent — the instructions core, the tool
  descriptions and the call weight (`app/services/injection_budget.py`), so the
  surface cannot grow silently.
- **Vectorization & summarization** are external LLM calls. Each of the three
  slots (embedding / summary / judge) is configured independently with its own
  provider (`ollama` or an OpenAI-compatible API), base URL, model and optional
  API key.
- **Hierarchical namespaces**: the store is split into large sections; the map
  is exposed to models via MCP instructions and `memory_namespaces`.
- **Related notes** (since v3.1): reading a note also returns its `links` —
  neighbours by meaning or by shared entities, from other namespaces; search,
  list and read outputs carry `chars`, the size of the full note text.
- **Listings that page** (since v3.1): `memory_list` and `skills_list` report
  `total` / `has_more` / `next_offset` / `next_cursor` and a `+N more` hint when
  a page is capped; the ceiling is tighter on the MCP surface than on REST.
- **Background worker**: pending vectors, summaries, dedup, classification,
  title generation and the new knowledge areas are processed asynchronously;
  scheduled maintenance jobs (related-notes recompute, default-namespace
  tidy-up) run on their own intervals; failures never break CRUD (pending
  states + back-off retry).
- **Observability** (since v3.1, extended in v3.2): `/health` (no token) reports
  the running application `version`, `queues` (how many jobs wait in each
  background queue, how long the oldest has waited, and how many need a human),
  `slots` (the last outcome and the pause of every slot, plus call diagnostics),
  `decision`, `areas`, `warnings`, `update` and `backups`.
- **Backups**: periodic online SQLite snapshots with rotation.

## Why

1. **Distributed knowledge with fast access.** Knowledge lives in a separate
   store, not in the system prompt or chat history; the model fetches only what
   is relevant, on demand.
2. **Token economy.** Instead of a monolithic context — a measured, guarded
   overhead for the tool spec (the instructions core 525 tokens, the 31 tool
   descriptions 2170 tokens), a compact skills announce (budget 2000
   characters, refreshed on every connect) and targeted retrieval of **short
   summaries**, not full texts.

## Quick start

```bash
git clone <repo> llm-second-brain && cd llm-second-brain
mkdir -p data prompts
# Edit docker-compose.yml: set MCP_AUTH_TOKEN (openssl rand -hex 32) and the
# three LLM slot addresses/models. Full reference: docs/CONFIG.md.
docker compose up -d --build
curl -s http://localhost:8080/health | python -m json.tool
```

`/health` answers without a token. See [Installation](docs/INSTALL.md) for the
first-run walkthrough (compose, token, Open WebUI, `/health`).

## Documentation

- [Installation](docs/INSTALL.md) — setup, first run, Open WebUI, `/health`.
- [Configuration](docs/CONFIG.md) — environment variables, the prompt files,
  and the `OLLAMA_KEEP_ALIVE` note.
- [Changelog](CHANGELOG.md) — release history (Keep a Changelog).

## Operational notes

- **`keep_alive` is not sent by the client** (since v2.1). Model residency is
  managed by the server: set `OLLAMA_KEEP_ALIVE` on the Ollama side if you want
  models to stay loaded. With the server default (5 min) models are unloaded
  more often, and a cold start (~22.6 GB for the summarizer) returns to
  latency.
- **Changing `EMBEDDING_PROVIDER` / `EMBEDDING_MODEL` / `EMBEDDING_DIM`**
  triggers an automatic full reindex on startup: all notes go to `pending` and
  the worker re-encodes them, and the knowledge-area indexes are rebuilt the
  same way. Search/dedup thresholds are calibrated for `qwen3-embedding:8b` —
  recalibrate after changing the model.
- **Knowledge areas (since v3.0)**: skills, terms and user facts live in the
  same database but in their own tables and indexes; they are isolated from
  notes in both directions. Their form limits and similarity thresholds are
  environment-tunable and validated at startup.
- **Privacy**: an `openai` provider sends note texts to an external API (the
  dedup judge and classifier see full texts). Choose providers per slot
  deliberately.

## Restore from a snapshot

Snapshots are ordinary SQLite files in `BACKUP_DIR` (default `/data/backups`,
inside the `./data` volume); the name carries the UTC timestamp. To restore one:

1. Take the newest `notes-*.db`. `GET /backups` lists the snapshots with
   `verified` (`ok` / `failed` / `unknown`) — prefer an `ok` one; a file marked
   `failed` also has a `<file>.bad` sidecar next to it. `POST /backups/verify`
   checks one on demand.
2. Stop the service: `docker compose down` (nothing must write to the file you
   are about to replace).
3. Put the snapshot in place of the live database (`DB_PATH`, default
   `data/notes.db`), e.g. `cp data/backups/notes-<stamp>.db data/notes.db`, and
   delete the stale `notes.db-wal` / `notes.db-shm` beside it if any. To keep the
   old file untouched, drop the copy next to it and point `DB_PATH` at the copy.
4. Start the service (`docker compose up -d`) and check `/health`: `status` is
   `ok`, `notes_count` matches the snapshot, and `embedding_ok` turns `true`
   after the first successful call to the embedding slot.
5. If the snapshot was taken with a different embedding model (the verification
   note `embedding_mismatch`), a full reindex starts on its own — wait for
   `pending_vector` to reach `0` before searching.

Shipping a verified copy off the box is optional and driven by `BACKUP_SHIP_CMD`
(empty by default); `scripts/backup_ship.sh` is a secret-free template. The
command and its output never reach the logs.

## License

Distributed under the [MIT](LICENSE) license.
