# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.2.0] - 2026-09-09

Release 2.2 — «Облегчение жизни моделям»: моделям проще находить заметку по
названию, управлять формой поиска/листинга одной ручкой, экономить контекст
и читать понятные промпты.

### Added

- **Title index + title search** (lsb-0001, №2): `notes.title` включён в
  векторный индекс (векторизуется title+text) и в полнотекстовый индекс
  `notes_fts`; заметку можно найти по названию.
- **Unified search & list** (lsb-0001, №6): один метод поиска с параметрами
  `memory_search(mode=semantic|title)` и один метод листинга
  `memory_list(detail=titles|summaries)`; путь (namespace) показывается во
  всех выдачах.
- **Chunk reading** (lsb-0003, №16): `memory_get(id, query?, chunk?, limit?)`
  — чтение заметки целиком (как раньше) или нужного чанка по смысловому
  запросу (`query`) либо по индексу с пагинацией (`chunk`/`limit`); мягкие
  отказы при `query+chunk` вместе и `chunk` вне диапазона.
- **Metadata edit without text** (lsb-0004, №5): `memory_update` может менять
  `title`/`summary`/`namespace` без перезаписи текста; `summary` сохраняется
  как есть (не перегенерируется).
- **Temporary storage (TTL)** (lsb-0004, №3): при записи можно указать
  `expires_at`; просроченные заметки удаляются фоновой зачисткой
  (`note_expirations`).
- **Namespace depth 3 + model-created namespaces** (lsb-0005, №7/№9/№10):
  максимальная вложенность увеличена с 2 до 3; модели создают узлы любого
  уровня через `memory_namespace_create` (с обязательным описанием);
  антисинонимия при создании узла; аудит автогенерированных описаний.
- **English canon for model-facing texts** (lsb-0006, №17): все промпты,
  подсказки и instructions переведены на английский канон и переписаны на
  понятность (описания инструментов, хинты мягких отказов, манифест).

### Fixed

- **lsbdef-0001** (№21): флаки живых тестов классификатора/суммаризации —
  парсер JSON извлекает валидный JSON из ответов, обёрнутых в теги
  thinking/response.
- **lsbdef-0002** (№20): стабилизирован флаки-тест параллельности серверов
  (`test_servers_are_independent`).
- **lsbdef-0003** (№24): E2E-скрипт lsb-0005 обновлён под EN-хинты после
  перевода.
- **lsbdef-0004** (№25): флаки живых тестов из-за недетерминизма судьи
  структуры — ретрай до 3 попыток.
- **lsbdef-0005** (№26): обрыв MCP-сессии под фоновой нагрузкой —
  MCP-таймауты (30 c connect/write/pool, 300 c read) в E2E-скриптах.

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
