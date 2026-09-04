# LLM Second Brain

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)
![Docker](https://img.shields.io/badge/Docker-ready-2496ed.svg)

A self-hosted **long-term memory server** for LLMs running in harnesses
(primarily Open WebUI). Models get MCP access to a shared note store: they can
search it (hybrid vector + full-text), read, write, update and delete notes.

## What it is

- **One Docker container**, self-hosted, non-root.
- **MCP Streamable HTTP** (`/mcp`, natively supported by Open WebUI) with a
  Bearer token.
- **7 tools** prefixed `memory_*`: search, list, get, save, update, delete,
  namespaces (map of sections).
- **Storage**: SQLite + `sqlite-vec` (vector search) + FTS5 (full-text),
  merged via Reciprocal Rank Fusion.
- **Vectorization & summarization** are external LLM calls. Each of the three
  slots (embedding / summary / judge) is configured independently with its own
  provider (`ollama` or an OpenAI-compatible API), base URL, model and optional
  API key.
- **Hierarchical namespaces**: the store is split into large sections; the map
  is exposed to models via MCP instructions and `memory_namespaces`.
- **Background worker**: pending vectors, summaries, dedup, classification and
  title generation are processed asynchronously; failures never break CRUD
  (pending states + back-off retry).
- **Backups**: periodic online SQLite snapshots with rotation.

## Why

1. **Distributed knowledge with fast access.** Knowledge lives in a separate
   store, not in the system prompt or chat history; the model fetches only what
   is relevant, on demand.
2. **Token economy.** Instead of a monolithic context — a fixed small overhead
   for the tool spec (~1200 tokens) and targeted retrieval of **short
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
- [Configuration](docs/CONFIG.md) — every environment variable (v2.1), the
  prompt files, and the `OLLAMA_KEEP_ALIVE` note.
- [Changelog](CHANGELOG.md) — release history (Keep a Changelog).
- Engineering documentation (goals, contracts, architecture, internal in
  Russian) — maintained out-of-repo and not published here.

## Operational notes

- **`keep_alive` is not sent by the client** (since v2.1). Model residency is
  managed by the server: set `OLLAMA_KEEP_ALIVE` on the Ollama side if you want
  models to stay loaded. With the server default (5 min) models are unloaded
  more often, and a cold start (~22.6 GB for the summarizer) returns to
  latency.
- **Changing `EMBEDDING_PROVIDER` / `EMBEDDING_MODEL` / `EMBEDDING_DIM`**
  triggers an automatic full reindex on startup: all notes go to `pending` and
  the worker re-encodes them. Search/dedup thresholds are calibrated for
  `qwen3-embedding:8b` — recalibrate after changing the model.
- **Privacy**: an `openai` provider sends note texts to an external API (the
  dedup judge and classifier see full texts). Choose providers per slot
  deliberately.

## License

Distributed under the [MIT](LICENSE) license.
