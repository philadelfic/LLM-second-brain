# Configuration

All configuration is environment variables, set in `docker-compose.yml`
(block `environment`). There is no `.env` file. The canonical defaults are
listed below; the actual values are overridden in compose.

## LLM slots (v2.1)

The service makes external LLM calls in **three slots**, each configured
independently:

| Slot | Provider | Base URL | API key | Model | Notes |
|---|---|---|---|---|---|
| **embedding** | `EMBEDDING_PROVIDER` (`ollama`, default) | `EMBEDDING_BASE_URL` (required) | `EMBEDDING_API_KEY` (optional) | `EMBEDDING_MODEL` (default `qwen3-embedding:8b`) | `EMBEDDING_DIM`, chunk params, batch/concurrent — unchanged |
| **summary** | `SUMMARY_PROVIDER` (`ollama`, default) | `SUMMARY_BASE_URL` (required) | `SUMMARY_API_KEY` (optional) | `SUMMARY_MODEL` (required) | `SUMMARY_THINK`, `SUMMARY_NUM_PREDICT`, `MERGE_NUM_PREDICT`, `SUMMARY_TIMEOUT_SEC`; think — ollama only |
| **judge** | `JUDGE_PROVIDER` (`ollama`, default) | `JUDGE_BASE_URL` (required) | `JUDGE_API_KEY` (optional) | `JUDGE_MODEL` (required) | `JUDGE_THINK`, `JUDGE_NUM_PREDICT`, `JUDGE_TIMEOUT_SEC`, `NAMESPACE_JUDGE_THINK`; think — ollama only |

- **Provider** is `ollama` (native API) or `openai` (OpenAI-compatible API
  with Bearer). Validation: `*_PROVIDER` ∈ {`ollama`, `openai`};
  `*_BASE_URL` must be `http(s)`.
- **API keys** are optional for both providers. Ollama without a key is fine;
  a cloud provider without a key will not work (you get a `401/403` hint on the
  first call). An `openai` provider with an empty key logs a WARN at startup
  (local OpenAI-compatible servers legitimately need no key); on a call,
  `401/403` produces an error with a hint to set `{SLOT}_API_KEY`.
- **`think` flags** (`SUMMARY_THINK`, `JUDGE_THINK`, `NAMESPACE_JUDGE_THINK`)
  only affect the `ollama` provider. OpenAI reasoning models reason on their
  own; any `reasoning_content` in the response is discarded — the client reads
  only `content`.

### Endpoints used

| Ollama (native) | OpenAI-compatible |
|---|---|
| `POST {base}/api/chat` | `POST {base}/v1/chat/completions` |
| `POST {base}/api/embed` | `POST {base}/v1/embeddings` |
| liveness: `GET {base}/api/tags` | liveness: `GET {base}/v1/models` |

`keep_alive` is **not** sent to any provider (see the note below).

## Full variable reference

### Required

| Variable | Meaning |
|---|---|
| `EMBEDDING_BASE_URL` | API address of the embedding slot |
| `SUMMARY_BASE_URL` | API address of the summary slot |
| `SUMMARY_MODEL` | generative model for summarization |
| `JUDGE_BASE_URL` | API address of the judge slot |
| `JUDGE_MODEL` | dedup judge model |
| `MCP_AUTH_TOKEN` | Bearer token for all endpoints except `/health` |

### Providers & keys

| Variable | Default | Meaning |
|---|---|---|
| `EMBEDDING_PROVIDER` | `ollama` | provider of the embedding slot |
| `SUMMARY_PROVIDER` | `ollama` | provider of the summary slot |
| `JUDGE_PROVIDER` | `ollama` | provider of the judge slot |
| `EMBEDDING_API_KEY` | `""` | optional API key (embedding) |
| `SUMMARY_API_KEY` | `""` | optional API key (summary) |
| `JUDGE_API_KEY` | `""` | optional API key (judge) |
| `PROMPTS_DIR` | unset | directory of editable prompt files (see Prompts) |

### Vectorization & chunking

| Variable | Default | Meaning |
|---|---|---|
| `EMBEDDING_MODEL` | `qwen3-embedding:8b` | embedding model; change → full reindex on startup |
| `EMBEDDING_DIM` | `4096` | vector dimension (fixed in DB); change → reindex |
| `TEXT_SPLITTER` | `tiktoken` | token splitter (encoding `cl100k_base`) |
| `CHUNK_SIZE` | `1024` | chunk window, tokens; change → re-chunk all notes |
| `CHUNK_OVERLAP` | `180` | overlap between windows (< `CHUNK_SIZE`) |
| `CHUNK_MIN_TARGET` | `200` | tail chunk smaller → merge with previous |
| `EMBEDDING_BATCH_SIZE` | `32` | chunks per embed request |
| `EMBEDDING_CONCURRENT_REQUESTS` | `3` | parallel embed requests (prod — `1`) |

### Summarization

| Variable | Default | Meaning |
|---|---|---|
| `MAX_SUMMARY_CHARS` | `200` | summary length limit |
| `SUMMARY_THINK` | `true` | allow summarizer reasoning (ollama only) |
| `SUMMARY_NUM_PREDICT` | `35000` | generation cap for the summary |
| `MERGE_NUM_PREDICT` | `35000` | generation cap for dedup merge |
| `SUMMARY_TIMEOUT_SEC` | `60` | client read timeout (prod — `750`) |

### Judge

| Variable | Default | Meaning |
|---|---|---|
| `JUDGE_THINK` | `false` | thinking judge (ollama only; prod — `true`) |
| `JUDGE_NUM_PREDICT` | `256` | verdict generation budget (prod — `1024`) |
| `JUDGE_TIMEOUT_SEC` | `30` | client read timeout (prod — `300`) |
| `NAMESPACE_JUDGE_THINK` | unset (inherits `JUDGE_THINK`) | structure judge think flag |

### HTTP / MCP / storage

| Variable | Default | Meaning |
|---|---|---|
| `PORT` | `8080` | HTTP port |
| `MCP_PATH` | `/mcp` | MCP server path |
| `DB_PATH` | `/data/notes.db` | SQLite path |
| `MAX_NOTE_CHARS` | `35000` | note length limit |
| `MAX_QUERY_CHARS` | `512` | search query length limit |

### Search & dedup

| Variable | Default | Meaning |
|---|---|---|
| `DEFAULT_TOP_K` | `5` | default `top_k` in `memory_search` |
| `DEFAULT_LIST_LIMIT` | `20` | default `limit` in `memory_list` |
| `SCORE_THRESHOLD` | `0.50` | cosine cutoff for search |
| `DEDUP_SIMILARITY` | `0.92` | "duplicate" cosine threshold |
| `DEDUP_CANDIDATE_TOP_N` | `3` | cosine candidates for the judge |
| `DEDUP_CANDIDATE_SIMILARITY` | `0.80` | candidate lower threshold |
| `RRF_K` | `60` | RRF merge constant |
| `SNIPPET_CHARS` | `120` | snippet length in search output |
| `MAX_GET_BATCH` | `20` | max ids in one `memory_get` |

### Background / observability / backup

| Variable | Default | Meaning |
|---|---|---|
| `PENDING_RETRY_SEC` | `30` | initial worker poll interval (×2 → 15 min) |
| `LOG_LEVEL` | `INFO` | log level |
| `AUTHOR_DEFAULT` | `unknown` | author if the harness sent none |
| `BACKUP_DIR` | `/data/backups` | backup snapshot directory |
| `BACKUP_INTERVAL_SEC` | `86400` | snapshot interval (daily) |
| `BACKUP_KEEP` | `7` | snapshots kept (rotation) |

### Namespaces (Phase 10)

| Variable | Default | Meaning |
|---|---|---|
| `NAMESPACE_AUTO_MOVE_MIN_CONFIDENCE` | `0.80` | auto-move confidence threshold |
| `NAMESPACE_PROMOTION_THRESHOLD` | `15` | default-notes counter for a new leaf |
| `NAMESPACE_PROMOTION_MIN_CONFIDENCE` | `0.60` | min classification confidence for the trigger |
| `NAMESPACE_SYNONYM_SIMILARITY` | `0.85` | anti-synonymy cosine (merge instead of create) |
| `NAMESPACE_AUTO_MAX_PER_DAY` | `3` | auto-created nodes per day (storm guard) |
| `NAMESPACE_MAX_LEAVES_PER_DOMAIN` | `12` | leaf cap per root |
| `NAMESPACE_GROOM_MIN_NOTES` | `2` | grooming: node below this → merge candidate |

## Prompts

Three system prompts are **editable** (Phase 11, decision №7). When
`PROMPTS_DIR` is set, the service creates them on first start if they are
missing (seeded with the built-in default as the starting text); existing
files are **never** overwritten; an **empty** file falls back to the built-in
default; a non-empty file wins:

- `summary_system` — note summarization;
- `summary_merge_system` — dedup merge (system);
- `judge_system` — dedup verdict.

The `judge_system` file **must** contain the markers `ДУБЛЬ` and `НЕ ДУБЛЬ`
(the dedup verdicts are parsed by them). If they are missing, the service
refuses to start with a clear message. The other seven prompts are hard-coded
and never created as files.

In compose, mount the directory and point `PROMPTS_DIR` at it:

```yaml
volumes:
  - ./prompts:/app/prompts
environment:
  PROMPTS_DIR: "/app/prompts"
```

## Notes

- **`OLLAMA_KEEP_ALIVE`** — since v2.1 the client does **not** send
  `keep_alive`; model residency is managed by the server. On the Ollama side
  set `OLLAMA_KEEP_ALIVE` (e.g. `OLLAMA_KEEP_ALIVE=30m`) if you want models to
  stay loaded. With the server default (5 min) models are unloaded more often,
  and a cold start (~22.6 GB for the summarizer) returns to latency.
- **Reindex**: changing `EMBEDDING_PROVIDER` / `EMBEDDING_MODEL` /
  `EMBEDDING_DIM` triggers an automatic full reindex on startup (all notes →
  `pending`, the worker re-encodes). Search/dedup thresholds are calibrated
  for `qwen3-embedding:8b` — recalibrate after a model change.
- **Validation**: all limits are validated at startup; an out-of-range or
  malformed value is a fatal configuration error listing every violation at
  once. Changing `MAX_NOTE_CHARS` over an existing DB is forbidden (the limit
  is baked into the CHECK schema).
