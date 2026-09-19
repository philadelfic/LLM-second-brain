# Installation

This guide walks through installing and running **LLM Second Brain** for the
first time: prerequisites, the compose setup, the auth token, connecting Open
WebUI, and checking `/health`.

## Prerequisites

- **Docker** with the Compose plugin (`docker compose version`).
- **LLM endpoints** for the three slots. By default the service talks to
  Ollama servers:
  - **embedding** — an Ollama with an embedding model (e.g.
    `qwen3-embedding:8b`);
  - **summary** — an Ollama with a generative model (e.g. `ornith:35b`);
  - **judge** — an Ollama with a generative model (can be the same as summary).
  Each slot can instead use any **OpenAI-compatible** API (see
  [Configuration](CONFIG.md)).

## 1. Clone and prepare

```bash
git clone <repo> llm-second-brain && cd llm-second-brain
mkdir -p data prompts
```

- `data/` — the SQLite database and backups (volume mount).
- `prompts/` — editable prompt files (created automatically on first start;
  see [Configuration](CONFIG.md) → Prompts).

## 2. Configure

All configuration lives in `docker-compose.yml` (no `.env`). At minimum set:

- `MCP_AUTH_TOKEN` — generate one: `openssl rand -hex 32`. Keep the production
  value in `docker-compose.override.yml` (gitignored); the git version holds
  `CHANGE_ME`.
- The three slot addresses and models:
  - `EMBEDDING_BASE_URL` / `EMBEDDING_MODEL`;
  - `SUMMARY_BASE_URL` / `SUMMARY_MODEL`;
  - `JUDGE_BASE_URL` / `JUDGE_MODEL`.

For an OpenAI-compatible provider, also set the corresponding `*_PROVIDER` to
`openai` and the `*_API_KEY`. See [Configuration](CONFIG.md) for the full
reference.

## 3. Start

```bash
docker compose up -d --build
```

The container is named `llm-second-brain` and listens on port `8080`.

## 4. Check health

`/health` answers **without** a token:

```bash
curl -s http://localhost:8080/health | python -m json.tool
```

```json
{"status":"ok","version":"<app version>","embedding_ok":null,"summarizer_ok":null,"judge_ok":null,
 "notes_count":0,"pending_vector":0,"pending_summary":0,
 "queues":{"vector":{"pending":0,"oldest_pending_sec":null},
           "summary":{"pending":0,"oldest_pending_sec":null},
           "judge":{"pending":0,"oldest_pending_sec":null},
           "areas":{"pending":0,"oldest_pending_sec":null},
           "links":{"pending":0,"oldest_pending_sec":null},
           "nodes":{"pending":0,"oldest_pending_sec":null}}}
```

`version` is the running application version (compared with the release tag at
acceptance). `queues` gives one entry per background queue — `pending` jobs and
`oldest_pending_sec`, the age of the oldest (`null` when the queue is idle).

`embedding_ok` / `summarizer_ok` / `judge_ok` reflect the outcome of the last
real attempt (`null` — none yet). On startup the service runs a lightweight
liveness check against each slot (no generation): a `200` sets the flag to
`true`; a network/5xx failure logs a WARN and the service starts degraded
(NFR-3); an explicit `401/403` is fatal and the service refuses to start with a
hint about the missing `*_API_KEY`.

## 5. Connect Open WebUI

Requires Open WebUI **v0.6.31+** (native MCP Streamable HTTP support).

1. **Admin Panel → Settings → Integrations** → "External Tool Servers" →
   **"+ Add Connection"**.
2. Fill the dialog:
   - **Type**: `MCP Streamable HTTP`;
   - **URL**: `http://<host>:8080/mcp` (the `MCP_PATH`, default `/mcp`);
   - **Auth**: `Bearer`; **token** — your `MCP_AUTH_TOKEN`.
3. Save. All 21 MCP tools appear and are available to all models: 8
   `memory_*`, 5 `skills_*`, 5 `user_*` and 3 `terms_*` (the surface is
   described in the [README](../README.md)).
4. Test in a chat: "find in memory …" → the model calls `memory_search`.

Other MCP clients connect the same way: URL `http://<host>:8080/mcp`, header
`Authorization: Bearer <MCP_AUTH_TOKEN>`.

## 6. Verify MCP by hand

```bash
curl -s http://localhost:8080/mcp \
  -H "Authorization: Bearer $MCP_AUTH_TOKEN" \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"curl","version":"0"}}}'
```

`tools/list` returns all 21 tools (the same surface as the
[README](../README.md)); without a token or with a wrong one — `401`.

## Upgrading

When upgrading an existing install, the service **auto-migrates** editable
prompt files (`prompts/`) that are byte-identical to a previously seeded or
known legacy seed (v2.1.x RU canon): they are rewritten with the current canon
on startup, so an upgrade from v2.1.x to v2.2.x needs no manual prompt work.

Files **edited by hand** are never overwritten. If such a file no longer
matches the current canon (e.g. a `judge_system.txt` missing the
`DUPLICATE`/`NOT DUPLICATE` markers), the service refuses to start with a
clear message. To fix: delete the file to re-seed it from the built-in default,
or edit it to add both markers (see [Configuration](CONFIG.md) → Prompts).

## Operations

- **Logs**: `docker compose logs -f second-brain` (one JSON line per event).
- **Backups**: snapshots in `BACKUP_DIR` (default `/data/backups`, inside the
  `./data` volume); first right after start, then daily; `BACKUP_KEEP` newest
  are kept.
- **Direct DB access**: `sqlite3 data/notes.db` (see Requirements §3).
- **Stop**: `docker compose down`.
