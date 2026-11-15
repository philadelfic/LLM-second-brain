# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Publishing on GitHub next after 3.4.1

The last published GitHub release is **3.4.1** (2026-10-29). Every release below
is already in `dev` and is published as one wave — after the prod rollout is
stabilized (owner's decision: GitHub only on stabilization):

* **3.5.0** (2026-11-01) — Thinking models without losing the answer
* **3.5.1** (2026-11-11) — Tolerant parts parser and a size hint
* **3.5.2** (2026-11-11) — One note per fact
* **3.5.3** (2026-11-11) — The split goes through the judge
* **3.5.4** (2026-11-11) — Generation limits inside options
* **3.5.5** (2026-11-11) — A number written with a space
* **3.5.6** (2026-11-11) — A mechanical fact floor for the split
* **3.5.7** (2026-11-11) — The floor lets the judge rule
* **3.5.8** (2026-11-11) — The judge is the only gate
* **3.5.9** (2026-11-12) — Self-contained facts
* **3.6.0** (2026-11-12) — The judge judges facts, the answer is not capped
* **3.6.1** (2026-11-12) — A log that outlives the container
* **3.6.2** (2026-11-12) — Parts of one note are not duplicates
* **3.6.3** (2026-11-12) — Short notes summarize themselves

## [3.8.1] - 2026-11-14

Release 3.8.1 — a false alarm at startup: the config scanner mistook the global
timeout knobs for operation names. The behaviour of the service does not change.

### Fixed

* **False `unknown_operation` WARNs about the global `*_TIMEOUT_SEC` settings**
  — after the 3.8.0 rollout prod logged three `unknown operation env ...`
  warnings on every start: `BACKUP_SHIP_TIMEOUT_SEC`, `SHUTDOWN_TIMEOUT_SEC`,
  `SLOT_MIN_TIMEOUT_SEC`. The scanner matched the environment against the flat
  `<PREFIX>_<FIELD>` mask of the operation registry, so a legitimate setting
  split into a prefix (`BACKUP_SHIP`, `SHUTDOWN`, `SLOT_MIN`) and the field
  `TIMEOUT_SEC`, and the prefix was reported as an unknown operation. The
  settings themselves were applied (300/300/30) — only the log line was wrong.
  The scanner now compares the env name with the `Settings` fields first
  (`settings_env_names`): a name that is a setting is configuration, never an
  operation, and is skipped before the mask is consulted. A typo
  (`MERG_MODEL`, `FROBNICATE_MODEL`) still warns, and the registry check stays as
  a fallback for operation names that carry no own `Settings` field.

### Changed

* `unknown_operation_warnings(settings)` takes the settings and scans as before;
  called from the startup warning block of `app/main.py` with the live settings.

### Tests

* `tests/test_operations.py`: the three global timeouts produce no WARN (and are
  really read into `Settings`), an invented name still warns,
  `settings_env_names()` is built from the `Settings` field set. Full suite:
  2489 passed, 13 skipped.

## [3.8.0] - 2026-11-14

Release 3.8.0 — a knob travels with its recipe: tuning became a mechanism
rather than one-off knowledge of the owner (`LSBN-F-37`) — a value is read
together with the way to calibrate it, so a consumer model does the calibration
itself when the model or the data changes. The `choice` gate of the decision
slot judges the normalized confidence of the server instead of a bare label
probability (`LSBN-T-41`), and the journal names the scale it applied.

### Added

* **A tuning registry with the recipe (`LSBN-F-37`)** — `app/services/tuning.py`
  holds `Tunable(key, env, slot, title, unit, affects, safe, metrics, hint,
  calibrated)` and five records at once: `decision_threshold`,
  `decision_choice_threshold`, `decision_note_chars`, `decision_model`,
  `slot_pause_after`. The recipe is neither an injected memory block nor a note:
  it arrives in the hint of the handle a consumer already reads. A new knob is a
  new record with a recipe and observable metrics; an empty recipe means the
  record is not ready.

* **MCP `tuning_list` / `tuning_get`, REST `GET /tuning` / `GET /tuning/{key}`**
  — one registry behind both transports and one card shape for a future front end
  (a tooltip per handle, no separate reference list): the list answers key, value
  and effect, the card adds `hint`, `metrics`, `calibrated`, `safe`, `slot` and
  `env`; an unknown key answers a hint pointing at `tuning_list`.

* **Calibration facts inside the card** — `decision_choice_threshold` carries the
  confidence formula `(p1 - 1/N) / (1 - 1/N)`, the numbers of the 2026-11-13
  measurement (prod, n=50: p >= 0.75 gives 100% (13/13), p >= 0.55 gives 82%
  (14/17)) and the note that `temperature`/`seed` are pointless for
  `/v1/systemone`: the decision is deterministic.

### Changed

* **The `choice` gate judges the normalized confidence (`LSBN-T-41`)** —
  `DECISION_CHOICE_THRESHOLD` (default `0.6`) applies to a `choice` answer that
  carries `confidence`: the excess over chance, `(p1 - 1/N) / (1 - 1/N)`. `noul`,
  `score` and a `choice` answer without `confidence` keep `DECISION_THRESHOLD` on
  the bare label probability, and a threshold written on the question itself (the
  question registry) still wins over both knobs.

* **The prompt layout of the decision slot** — `state` now carries the note text
  only, capped by `DECISION_NOTE_CHARS` (default `800`), while the node
  descriptions moved into `criteria` (the 2026-11-13 measurement: 6.1-7.8 s
  against 8.3 s, quality not worse).

* **The journal names the scale of the threshold** — `decision_answer` writes
  `applied_threshold` and `applied_scale` (`confidence` | `probability`) instead
  of a bare `threshold`, so a number in the log is never read against a threshold
  of the other scale. A `choice` answer that arrived without `confidence` is
  reported once per process and falls back to the probability scale.

## [3.7.0] - 2026-11-12

Release 3.7.0 — "The retry economy of splitting": the live failure of
2026-11-12 (a thinking judge with a degraded 256-token budget answered nothing,
so the migration queue re-split the same notes every 10 minutes, burning the
generation slot for days) is fixed at three levels — the judge's budget, the
verification shape, and the retry policy — plus a bounded SIGTERM stop
(a container Up with a dead HTTP was a second live outage of the same day).

### Fixed

* **A half-dead container after SIGTERM (`LSBN-D-16`)** — 12.11 prod:
  uvicorn dropped the HTTP listener at 18:56:32Z, but a worker thread
  blocked on an LLM call (`asyncio.to_thread`, the read timeout up to
  900 s) held the process for 20+ minutes: the container stayed `Up` with
  a dead HTTP, the healthcheck went `unhealthy` 42 times,
  `restart: unless-stopped` never fired (the process never exited) and
  nothing alerted — a quiet memory outage for every agent. The stop is
  bounded now: the background loops are cancelled before uvicorn logs
  Application shutdown complete, anything still alive past the budget is
  abandoned with a forced exit. SQLite is transactional (WAL): committed
  work survives, an interrupted batch keeps its pending statuses and is
  caught up after the restart.

* **A thinking judge without an explicit budget is a fatal config error** —
  with `JUDGE_NUM_PREDICT=0` and `JUDGE_CONTEXT_TOKENS=0` the auto budget
  degraded to the legacy 256-token cap of the `judge` operation, the reasoning
  ate all of it (`done_reason=length`, empty `content`), and every judge call
  failed (dedup froze after 8 attempts, split verification always fell). The
  shipped compose sets `JUDGE_NUM_PREDICT=4096`; validation (NFR-6) refuses to
  start on `JUDGE_THINK=true` + auto budget + unknown context.

### Added

* **`SPLIT_VERIFY_SEGMENT_CHARS`** (default `6000`, range `1000..20000`) — the
  split verdict verifies the source segment by segment (overlap ≤ 1/4 of the
  segment); the first `MISSING` batch is the verdict. One giant prompt
  "source + all parts" exploded on real notes (75 fragments).
* **The split parts cache** — `migration_notes.parts_json` (`{source_sha1,
  parts}`): a transient judge failure verifies the same parts without calling
  the split model again; saving the parts clears the cache.
* **The split attention state** — the `SLOT_MAX_ATTEMPTS` ceiling of
  non-transient failures freezes a note (`migration_notes.attention=1`): the
  job stops taking it, the queue is not "empty" (phase stays open, the union
  read keeps the source visible), a `warnings` record carries the reason.
* **The mechanical garbage floor** — `SPLIT_FACTS_FLOOR=0.3` (`facts_coverage`
  of hard tokens): below it the parts are a loss, not a retelling, and go to
  quarantine without burning the judge's call.
* **`SHUTDOWN_TIMEOUT_SEC`** (default `30`, range `1..30`) — the upper bound
  of SIGTERM handling (`LSBN-D-16`). The cancelled background tasks
  (worker/backup/update-check) get this budget, then a short grace lets the
  busy executor threads finish, then a forced exit (`os._exit(0)` — a docker
  stop is a planned stop, so `restart: unless-stopped` keeps the container
  stopped instead of looping). The journal gets one `shutdown_forced` event
  (`timeout_sec`, `thread_grace_sec`, `pending_tasks`, `stuck_threads`).

* **The fragmentation ceiling** — `SPLIT_PARTS_SLACK=2`: more parts than
  `ceil(chars / limit) + 2` means the model is padding, not decomposing — one
  follow-up, then defer (the "one part per fact" era fragmented a live note
  into 75 scraps; the prompt now asks for dense parts).

### Changed

* **Transient failures no longer eat the attempt budget** (the #39 semantics,
  mirrored from the dedup pairs): `unavailable`/`timeout`/`auth` wait on the
  loop's back-off; only `empty`/`truncated`/`off_contract` count, and on the
  ceiling the note moves to attention.

## [3.6.3] - 2026-11-12

Release 3.6.3 — "Short notes summarize themselves": a note that fits in
`SUMMARY_SHORT_TEXT_CHARS` characters no longer waits for the summary model —
its summary is its own text, so the summary queue stops being eternal on the
parts of split notes.

### Added

* **`SUMMARY_SHORT_TEXT_CHARS` (`LSBN-T-37`)** — the short-note threshold
  (default `400`, range `0..5000`; `0` — the rule is off, every note takes the
  model path as before). The journal gets one `summary_shortcircuit` event per
  note (`note_id`, `text_chars`, `summary_chars`). The parameter is not part of
  the slot-configuration fingerprint — it changes no slot.

### Changed

* **Short notes do not spend the model** — the worker takes the summary of a
  note not longer than the threshold from the text itself: the same
  `cap_summary` cut (`MAX_SUMMARY_CHARS`, word boundary) and the same guarded
  `UPDATE ... WHERE summary_status = 'pending' AND text = ?`, `summary_status`
  goes to `ok` without a single slot call. On prod that is nearly all of it — 2855 short notes
  (the parts of split notes, median ~102 characters).

### Fixed

* **The summary queue no longer grows eternal on the parts of split notes** —
  a pass used to take ~32 notes and go to sleep on the back-off, leaving ~3000
  pending tasks behind; the short ones now leave the queue in one pass without
  touching the slot.

## [3.6.2] - 2026-11-12

Release 3.6.2 — "Parts of one note are not duplicates": the background dedup no
longer sees the parts of one split note as duplicates of each other (the
`LSBN-D-14` storm of merge attempts), and the operator gets a brake for the whole
background dedup.

### Added

* **`notes.origin_note_id` — the origin group of the split parts (`LSBN-D-14`)** —
  a part written by the splitter remembers its source note: the column is filled
  for new splits and backfilled on start from `migration_notes` (the journal gets
  the `origin_backfill` event with the `filled` counter).
* **`DEDUP_ENABLED` — the operator brake of the background dedup** — `false`
  does not start the judge job, while its queue stays visible in
  `/health.queues` (`/health.dedup_enabled` reports the flag itself): the backlog
  keeps accumulating honestly and is worked off after the flag is turned back on.
  The flag is not part of the slot-configuration fingerprint — flipping it does
  not clear the attention of the slots.

### Fixed

* **`LSBN-D-14`: the dedup merged the parts of one split note** — a split source
  left its parts mutually similar, and the background dedup kept trying to merge
  them: a storm of merge attempts, `off_contract` verdicts and pauses of the
  summary slot. Parts of one origin are no longer candidates for each other, and
  pairs of one source already frozen by the judge are closed instead of thawed.

### Changed

* **One rule on both ends of the dedup** — the merge path and the candidate
  pre-filter now follow the same rule "parts of one source are not duplicates";
  the KNN window is widened over the note's own origin group, so the rule sees
  exactly the parts it must skip.

### Upgrade notes

* On prod drop the temporary thresholds `DEDUP_SIMILARITY=1.0` and
  `DEDUP_CANDIDATE_SIMILARITY=1.0` (back to `0.92` / `0.80`): the `DEDUP_ENABLED`
  brake replaces that workaround.
* The origin backfill runs at every start; the journal gets the `origin_backfill`
  event with the `filled` counter (expected 3083).

## [3.6.1] - 2026-11-12

Release 3.6.1 — "A log that outlives the container": the same JSON journal now
also goes to a rotating file, so the analysis of a failure (the split of a long
note, the merge storm) no longer dies together with the container.

### Added

* **A rotating JSON log file** — the settings `log_file` / `log_max_bytes` /
  `log_backups` (env `LOG_FILE`, `LOG_MAX_BYTES`, `LOG_BACKUPS`): an empty
  `LOG_FILE` keeps stdout only (as before), a set one adds a
  `RotatingFileHandler` with the same `JsonFormatter`, rotation at
  `LOG_MAX_BYTES` (default 50 000 000) keeping `LOG_BACKUPS` files (default 5);
  the parent directory is created on start and the handler is built once, like
  the stdout one.

### Changed

* **Post-mortem history survives a deploy** — `docker logs` lives as long as the
  container and every deploy recreates it; the file keeps the journal across
  restarts, which is exactly what the split/merge analysis of 11.11.2026 lacked.

## [3.6.0] - 2026-11-12

Release 3.6.0 — "The judge judges facts, the answer is not capped": the split
verdict answers one question only — was a fact of the source lost (the shape of
the parts is the splitter prompt's business), and a provider that rejects
generation limits (the ollama.com cloud models) can be served without them.

### Changed

* **The split verdict checks facts only** — `SPLIT_VERDICT_SYSTEM` loses the
  self-containment half: `COMPLETE` — nothing is lost, `MISSING: <what exactly is
  lost>` — otherwise; `FRAGMENTS` is no longer read as a refusal, so a note is
  quarantined only for a lost fact. Rewording, splitting into pieces of
  different sizes and a changed shape of the text are explicitly not losses.

### Added

* **`SUMMARY_NO_OUTPUT_LIMITS` / `JUDGE_NO_OUTPUT_LIMITS` (default `false`)** —
  with the flag the request carries no answer cap at all: no `options.num_predict`
  (ollama) and no `max_tokens` (openai), the model stops when it is done.
  `options.num_ctx` stays whenever the declared context is known — without it the
  server allocates its own default (262 144 on the AI server), the model stops
  fitting the GPU and generation degrades several times.

## [3.5.9] - 2026-11-12

Release 3.5.9 — "Self-contained facts": a part of a split is a statement, not a
heading and not a phrase cut in the middle; the judge refuses fragments, and the
split never spends the slot on a note from the trash.

### Changed

* **The split prompt asks for a self-contained STATEMENT** — one to three
  sentences per part: never a bare heading, never a fragment, never a list of
  items, never a phrase cut in the middle; a part shorter than 100 characters is a
  fragment and is wrong.
* **The judge refuses fragments (`FRAGMENTS: <how many>`)** —
  `SPLIT_VERDICT_SYSTEM` gets the self-containment question next to the fact one
  and `verify_split` reads `FRAGMENTS` as a refusal: the note is quarantined with
  the judge's reason and a warning is opened, so "the words are present but the
  note is a fragment" no longer passes.

### Fixed

* **A trashed note is not split** — the migration reads `deleted_at` together with
  the note and marks a trashed one `expired` (`заметка в корзине`) instead of
  splitting it (live case 2026-11-11: the queue carried five trashed notes, one of
  them was split into 75 fragments).

## [3.5.8] - 2026-11-11

Release 3.5.8 — "The judge is the only gate": the mechanical fact coverage of
a split stops blocking anything and only measures, so a note reaches the
reviewer instead of dying on arithmetic.

### Changed

* **`SPLIT_FACTS_MIN = 0.0`** — the hard-token arithmetic no longer gates a split;
  the only gate is the judge's verdict (`verify_split`: the source and ALL the new
  parts with the "nothing missed" question). Live 2026-11-11: four attempts on one
  note landed at 0.72 / 0.90 / 0.88 / 0.67 and never reached the judge — the gate
  was stricter than the reviewer.
* **Tests** — `tests/test_migration_split.py`: the coverage is diagnostic while the
  note length limit stays hard.

## [3.5.7] - 2026-11-11

Release 3.5.7 — "The floor lets the judge rule": the mechanical fact floor drops
so that a clean fact-split with rephrased prose reaches the reviewer at all.

### Changed

* **`SPLIT_FACTS_MIN` 0.90 → 0.85** — live splits of 2026-11-11 stopped exactly at
  0.90 and never reached the judge; the floor is a gross-loss catcher, the verdict
  belongs to the judge.

## [3.5.6] - 2026-11-11

Release 3.5.6 — "A mechanical fact floor for the split": the split stops being
measured by the merge's strict word coverage (which quarantined every honest
fact-split) and gets its own floor on hard facts.

### Added

* **`SPLIT_FACTS_MIN = 0.90` and a fact-coverage gate for the split** — before
  this the split was checked with the merge's `MERGE_COVERAGE_MIN = 1.0` word
  coverage; a fact-split legitimately rephrases the prose, so the split now uses
  `facts_coverage` (numbers, latin identifiers, paths, versions) at a floor of
  0.90. Below the floor the note is refused with "the parts lost hard facts of the
  source — numbers, dates, versions or paths are missing (fact coverage 0.xx)";
  the authoritative verdict stays the judge's.

## [3.5.5] - 2026-11-11

Release 3.5.5 — "A number written with a space": digit separators are folded
before the fact comparison, and the split prompt forbids reformatting numbers.

### Fixed

* **Digit separators inside a number are folded (`_DIGIT_SEPARATOR_RE`)** —
  `5 000` / `5_000` / `5.000` instead of `5000`, and non-breaking spaces, no
  longer read as a lost fact (the live 0.72 coverage case: the facts were
  present, only written with a space).

### Changed

* **The split prompt asks for exact numbers** — copy numbers, names, dates,
  versions, paths and commands EXACTLY as written in the source: do not round,
  do not reformat, do not split a number with a space, do not translate them;
  only the connecting prose may be tightened.

## [3.5.4] - 2026-11-11

Release 3.5.4 — "Generation limits inside options (`LSBN-D-14`)": ollama ignores
unknown top-level fields, so the whole answer/reasoning budget arithmetic of
3.5.0 never reached the model.

### Fixed

* **`options.num_predict` instead of a top-level one (`LSBN-D-14`)** — ollama
  silently dropped a top-level `num_predict` and the model generated with its own
  default (unlimited); proven on the stand: a top-level `num_predict=4` returned
  45 tokens, `options.num_predict=4` returned 4 (`done_reason=length`).
* **`options.num_ctx` from the declared context** — without it the server
  allocates its own default (262 144 on the AI server), the model stops fitting
  the GPU and generation degrades several times over (the slow "thinking" of
  2026-11-11).

## [3.5.3] - 2026-11-11

Release 3.5.3 — "The split goes through the judge": the source note and all the
fact-notes it was split into are shown to the judge before the split is
accepted.

### Added

* **`JudgeService.verify_split`** — the source plus ALL new parts with one
  question ("was anything from the source lost"), always on the generative path:
  the decision slot is schema-bounded (8k context, 64 KiB input) and cannot take a
  long note plus its parts. `COMPLETE` → nothing lost, `MISSING: …` → the judge's
  reason, anything else → `OFF_CONTRACT`.
* **The verdict gates the split in the migration job** — a refusal is a
  quarantine (`карантин`, reason `судья: …`, attempts bumped, a warning opened,
  journal event `migration_split_rejected`); an unavailable judge keeps the note
  pending (`migration_split_verdict_failed`); a pass is journaled as
  `migration_split_verified`.

## [3.5.2] - 2026-11-11

Release 3.5.2 — "One note per fact (`LSBN-D-13`)": splitting stops being "copy
the text in chunks" and becomes a decomposition into atomic facts, checked on
the fact skeleton instead of the prose.

### Changed

* **The split prompt: one fact per part** — each part states exactly one
  self-contained fact and carries its own names, numbers, dates, statuses,
  configs and paths; no whole paragraphs copied, no several facts in one part;
  the new `_facts_hint` gives the model the source size and "one part per fact,
  each well under the limit".
* **`facts_coverage` instead of the word coverage for the split** — the share of
  the source's hard tokens (numbers, latin identifiers, paths, versions) found in
  the parts; Cyrillic prose may be reworded, a missing number or path is a lost
  fact. The merge keeps its strict 1.0 word coverage.

### Fixed

* **A clean fact-split is no longer quarantined by arithmetic** — the strict word
  coverage read an honest decomposition as 0.86 and refused it.

## [3.5.1] - 2026-11-11

Release 3.5.1 — "Parts survive a broken answer": the parts are extracted
tolerantly (the "not a JSON object" case of the live duplicate pair), the model
is told the source size and the minimum number of parts, and the English-only
remainder of `LSBN-T-32` lands here (moved out of 3.5.0 by the owner's decision of
11.11.2026: mechanical hygiene must not hold the core).

### Added

* **A size hint for merge and split (`_size_hint`, `LSBN-D-13`)** — "SIZE: the
  source is N characters; the note limit is M. Return at least K part(s), each
  within M characters. Dropping, shortening or summarizing the content is a
  failure": without the number the model returned one oversized part or silently
  shortened the text (the live case of 2026-11-11: the model cut the merged text
  in half, coverage 0.50).
* **`_balanced_objects`: a string-aware scanner of every JSON object in the
  answer** — the greedy "first `{` .. last `}`" slice broke on answers with prose
  braces or two objects in a row; every balanced object is tried, raw and with
  escaped quotes, and a bare array is read as `{"parts": [...]}`.
* **The English-only remainder of `LSBN-T-32`** — comments and docstrings of
  `app/`, `scripts/` and the remaining `tests/` translated; the language checker
  reported 110 files with Cyrillic at the 3.5.0 gate, 109 of them under `tests/`.

### Changed

* **The merge/split prompts ask for one JSON object only** — no markdown fences,
  no text before or after it, no trailing commas, every inner double quote
  escaped in JSON style; merge additionally gets "a shortened retelling is a
  failure, not a merge".

## [3.5.0] - 2026-11-01

Release 3.5.0 — "Thinking models without losing the answer": a call now carries
two independent budgets — the reasoning no longer eats the useful answer
(the `lsbdef-0008` class, the flint merge case), an answer eaten by reasoning
is recognized, retried once with a bigger budget and reported as its own
cause; the merge parser repairs the raw-quote answers of reasoning models;
the assistant-logs area gains a one-handle period export for release gates;
the migration health block learns to speak reverse mode, and a down-up cycle
no longer dies on a taken away-base name.

### Added

* **Answer vs reasoning budget (LSBN-T-33)** — `SLOT_REASONING_NUM_PREDICT`
  (slot level) and `<OP>_REASONING_NUM_PREDICT` (per operation) set a
  reasoning budget; the wire limit becomes `answer + reasoning` clamped by
  the context window and the defensive cap, so thinking stops taking from
  the answer. Empty content with thinking present is a distinct error
  (`LLMReasoningBudgetError`) with one auto-retry at a doubled budget (a
  request-shaping retry — back-off and the slot breaker are not involved).
  `ChatMeta`, the journal and `/health.slots` carry the split budgets;
  `/health.operations` shows them per operation; the config fingerprint
  covers the new fields (attention clears on restart after a change).
* **Period export of the assistant logs (F-26)** — `logs_export` builds the
  release-gate journal digest as a ready-made markdown file in
  `<db_dir>/exports/` (visible on the host volume) plus a short summary in
  the response: same slices as `logs_search`, at least one period bound
  required, entries oldest-first grouped by day with full field text.
  Read-only; the MCP surface grows to 32 tools and the awareness-cost
  ceiling is rebased accordingly (88500T, reason recorded).
* **`/health.migration` speaks reverse mode (D-11)** — the block carries
  `direction: forward|reverse`; reverse states show `poured_rows` (delta
  rows of all tables) and `kept_parts` instead of the forward note counter
  and skip the forward invariant — the old output read as a migration
  failure.

### Fixed

* **Raw quotes inside merge JSON values are repaired (T-33/T-23)** —
  reasoning models return a meaning-valid parts object with unescaped
  quotes inside string values; a conservative repair pass escapes them
  before parsing, parts survive with the text intact, and a still-broken
  answer is refused exactly as before.
* **Away-base name resolved before the pour (D-12)** — a down-up cycle
  leaves a chain-relative base with the same CHECK as the active one; the
  next return-up or abort died at swap with "the away-base name is taken by
  a foreign file". The name is now picked before the pour (suffix
  `.away`/`.awayN` when taken), it fans out into `state.prev_file` and the
  receiver chain, and swap keeps its fail-closed refusal as defense in
  depth.

## [3.4.1] - 2026-10-29

Release 3.4.1 — "One number for the note limit, a background that degrades politely": the note limit leaves the code for a single configuration value, the migration job gets its own slicing cadence, and the background job framework stops grinding through broken providers — a failing slot now pauses whole batches instead of timing out item by item, a broken slot gates every job that shares it (including external ones), and a freshly saved note can no longer miss its vectorization wake-up.

### Added

* **Note limit in configuration (one number, one place)** — `MAX_NOTE_CHARS` (env, default **2000**, validated 500..20000) feeds the save/update validation, merge and split part limits, hints, tool descriptions, server instructions and `/health.note_chars_limit`. The `CHECK` of new databases and the self-migration target follow the same value: raising the limit pours 1:1 without a model, lowering it splits. The `NOTE_TEXT_LIMIT` code constant is removed.
* **Migration slicing cadence** — `MIGRATION_BATCH` (default 5) and `MIGRATION_PAUSE_SEC` (default 600 s) are the migration job's own numbers: it takes a batch of notes, yields the generation slot for the pause, and repeats; an emptied queue skips the pause. Both values are visible in `/health.migration.slot` and the startup log; the old shared knobs (`EMBEDDING_BATCH_SIZE`, `PENDING_RETRY_SEC`) no longer affect the migration.
* **Framework-level slot gate** — the circuit-breaker check moved into the job framework (`JobSpec.gate`/`is_gated`, applied by the shared loop to both pass invocations); manual per-job checks are gone. External jobs (`migration`, `warnings`) are now gated by their declared `summary` slot — previously they kept calling a broken slot — and `areas` declares its embedding slot. Queue snapshots move `pending` to `waiting` uniformly for every job of a broken slot; `/health.queues` still never calls models.
* **Live integration tier runner** — `scripts/run_live_tests.sh` runs the 13 integration tests against real model hosts taken from the environment (or a gitignored local env file); host addresses never enter the repository. The tier's dimension is env-configurable (`LIVE_EMBEDDING_DIM`).

### Changed

* **Batches stop on wait-class slot failures** — `summary`, `title` and `judge` batches abort after the first `unavailable`/`timeout`/`auth` outcome instead of grinding through up to 32 positions (a hanging provider could previously hold a loop for hours of 900-second timeouts, invisibly). Processed positions count, the tail stays pending, and the queue is drained by back-off, the slot probe or the save notifier — no work is lost.
* **Note limit default 1500 → 2000** — one step that keeps 176 active prod notes (average 1735 chars) whole instead of splitting them; the "one note = one cluster of facts" band is preserved.

### Fixed

* **Lost wake-up for fresh notes** — the embedding job now rechecks its queue after clearing its wake-up event: a save landing between an empty pass and the event clear no longer waits the grown back-off (up to 15 minutes on an idle system) for vectorization while models are healthy. With models down the recheck costs one refused pass — a failed attempt is not progress, so no busy loop returns.

## [3.4.0] - 2026-10-28

Release 3.4.0 — "Less information per step and merging without loss": the memory stops paying for its own bookkeeping on every step — searches and listings return headers capped by a character budget, and a note is read point-wise by id. Long notes stop being truncated: merging returns a list of self-contained parts with a strict completeness check, the note limit drops to 1500 characters, and a schema-baked limit can now be changed over a live database by self-migration instead of failing the start.

### Added

* **Character budget and two-level reading (F-13)** — `budget_chars` on `memory_search`, `memory_list`, `memory_get` (100..100000, soft refusal outside the range); defaults are code constants (search 1200 / list 2000 / get 2000). Elements that do not fit degrade to a header with `truncated: true`; the only hint of a successful answer is "read point-wise by id" (up to 8 ids, the rest as "+k"). A single `id` (and chunk mode) still returns the full text.
* **Environment card in `memory_get` (F-16)** — `map` / `map_level` (`auto`|`1`|`0`) / `map_limit` (1..20, default 8) / `map_offset`: rows `{id, title, namespace, kind}` (direction only for `mention`), budget 1500 characters with degradation, paging with `total`; `batch`+`map` is a soft refusal.
* **Operations registry (F-34)** — nine operations (summarize, title, merge, split, classify, describe, judge, structure_judge, embed) with flat `<OPERATION>_<FIELD>` overrides and the chain "operation env → slot env → default"; `/health.operations` shows model, provider and source per operation, never addresses or keys.
* **Note-limit self-migration (F-22)** — a changed `MAX_NOTE_CHARS` over a live database now builds a new base, pours 1:1, splits what does not fit, swaps the file pointer and reads from both bases through the migration (search, get, list, dedup) until done; the chain of bases is kept, the reverse direction pours 1:1 without a model, `POST /migration/abort` stops the migration; reason and progress are visible in `/health` (block `migration`, queue `migration`).
* **Saving an array of notes (F-23)** — `memory_save(notes=[...])` takes 1..20 objects with per-element validation and answers `{stored_count, notes: [...]}`; `text` became optional.
* **Ranking boosts (F-14)** — order-only boosts after RRF: title exact 0.25, title prefix 0.15, freshness 0.10 (30-day window), linked 0.05; switch `RANKING_BOOSTS_ENABLED` (default true); `rrf_score` in the answer stays raw RRF.

### Changed

* **Contract change: MCP reads default to headers.** `memory_search` returns `{id, title, namespace, chars, score}`; `memory_list` defaults to `detail="titles"`; batch `memory_get` returns headers without text; full texts are no longer returned in lists (MCP or REST).
* **Note limit 35000 → 1500 characters** (`NOTE_TEXT_LIMIT`, `MAX_NOTE_CHARS` default and compose): saving or updating a longer text is a soft refusal with a hint to split it into several self-contained notes.
* **Merging returns self-contained parts** (`{"parts": [{"title", "text"}]}`, each part ≤1500 characters, its own title ≤5 words): completeness is checked strictly (1.0, normalized tokens), one retry in the same context, then the pair is deferred with both notes alive — silent truncation is gone and the truncation marker of merging is removed.
* **Prompt canon:** `merge_system` now carries the parts contract; a new editable `summary_split_system` splits one long note (`EDITABLE_PROMPTS` 3→4, `SEED_VERSION` 3.4.1).
* **Prompts and limits:** per-operation output budgets live in the registry (historical `MERGE_NUM_PREDICT` absorbed), `describe` inherits `SUMMARY_NUM_PREDICT`, `judge` has no layer of its own, the decision-slot reserve outranks overrides.
* Version bumped to 3.4.0 in `app.__version__` and `pyproject.toml`, so `/health.version` matches the release tag.

### Fixed

* **DEF-5** — a repeated forward migration (down → up → down) used to delete the chain base `notes.<limit>.db`, the only live copy of the parts split by the first run; the new base is now poured into `notes.<target>.db.incoming` and the chain base is never removed (event `migration_chain_base_spared`, regression test over the full forward→reverse→forward cycle).
* **DEF-6 / DEF-7a** — splitting now writes a `slot_call` per model call (a retry is a separate line) and the migration phase is visible as `split` while parts are being produced.
* **D-8** — a hard delete purges the note's comment rows from `notes_comment_fts` (no more ghosts in comment search); a soft delete keeps them.

### Known price

* Boosts trade MRR for freshness and links on paraphrase queries (0.8223 → 0.7327 on the 53-pair set, recall@5 unchanged): accepted as the stated price, weight tuning is a backlog item.
* The fixed awareness block grew with the new surfaces (79100 → 84260 tokens at 10 steps) while the per-step payload shrank: shrinking the fixed block is a backlog item.

## [3.3.1] - 2026-10-17

Release 3.3.1 — "Contour hotfix: history and observability": physically deleted notes no longer leave orphan version history that breaks the next writer of a reused id, and the age of the oldest waiting work stays visible exactly when a slot is down.

### Fixed

* **lsbdef-0014** — the hard deletion of a note (TTL cleanup, the `expiration` job) removed it from `notes`, chunks, vectors, FTS and links but not from `record_versions`; `notes.id` is not `AUTOINCREMENT`, so the freed id was reused by the next note and `record_initial` failed with `UNIQUE constraint failed: record_versions(entity, entity_id, version)` — `memory_save` stayed broken until the orphan rows were removed by hand (test contour, 2026-10-15). `delete_note_physical` now purges the note's history rows in the same transaction; a soft delete keeps them (a trashed note is restorable). Regression test: a hard delete leaves no history rows, a reused id survives save and update over the full `NoteService` path, and the deleted note's history does not resurrect.

### Changed

* **`oldest_pending_sec` covers `waiting`** (LSBN-S-3): the age of the oldest work waiting for execution is computed over `pending` OR `waiting` and is `null` only when both categories are empty. With a slot breaker open the whole queue moves to `waiting` (`pending = 0`) and the age used to disappear exactly in the state it exists for (the 3.3.0 caveat is lifted). The judge queue's age now includes attention-frozen jobs as well, so an attention-only queue reads an age instead of `null`. An empty queue still reads `null` — the snapshot never invents an age.
* Version bumped to 3.3.1 in `app.__version__` and `pyproject.toml`, so `/health.version` matches the release tag.
* **Parallel unit suite** (T-1, decision of О. 2026-10-17): `pytest-xdist` joins the dev dependencies and the full suite runs with `pytest -n auto` — 2147 tests drop from ~150 s serial to ~28–30 s on 12 workers, the passed/skip set verified identical to the serial run three times over (dev and inside the acceptance image). The only parallelization hazard found was fixed: the session fixtures of `test_mcp.py`/`test_logging.py` held fixed server ports (18765–18767) and collided across workers — ports are now allocated per `PYTEST_XDIST_WORKER`; outside xdist nothing changes. The serial run stays the default; switching the default over is a later call.

## [3.3.0] - 2026-10-16

Release 3.3.0 — "Job contour reliability": the first start over a database of a previous version no longer races its own job table into an error, and the behaviour of a queue whose model is unavailable is pinned by a reproducing test instead of being taken on trust.

### Fixed

* **lsbdef-0013** — the lazy migration of the `worker_jobs` table used to race itself on the first start over a database of a previous version: the `_jobs_table_ready` flag was checked without a lock (check-then-act) and nothing guarded the window between `PRAGMA table_info(worker_jobs)` and `ALTER TABLE … ADD COLUMN`, so the `judge` and `nodes` loops both saw a missing column, both ran the ALTER, and the second one failed with `duplicate column name` — the startup attention reset failed with it (found on the production deploy of 3.2.0, 2026-10-13). The whole DDL now runs under a `threading.Lock` with a double-checked flag, and the `ALTER` tolerates `duplicate column name`: the lock covers the threads of one process only, a competing process is not its business. Regression test: two threads — and, separately, two migrators racing the startup attention reset — work on a database without the attention columns at the same moment; no exceptions, the schema comes out complete, the reset goes through.

### Changed

* **A queue behind an unavailable model is verified, not assumed** (№39): with the slot down the work moves to `waiting` (`pending = 0`), nothing is lost and the counters stay honest; when the slot answers, the queue drains. `tests/test_worker_queue_unavailable_models.py` pins both a transient refusal and an auth break, draining in batches (five jobs, a batch of two, at least three loop iterations, then a zero drain). No product code changed — the behaviour was already correct, the release makes it provable.
* Version bumped to 3.3.0 in `app.__version__` and `pyproject.toml`, so `/health.version` matches the release tag.

## [3.2.0] - 2026-10-13

Release 3.2.0 — "Hygiene, contour and slots": the memory becomes accountable. Every change to a note, an area record or an assistant log is versioned with line-level deltas and can be rolled back by the operator; a new area keeps the assistant's own "what I did and why"; the slots learn to distinguish a broken call from a slow thinker and to wait instead of failing; and the text the server injects into the agent is measured, shortened and guarded by a budget.

### Added

* **Version history** (№41): one `record_versions` table carries content versions of everything contentful — notes, skills, terms, user facts, assistant logs, warnings — with line-level deltas for long texts, a full baseline for the first version and a one-time backfill of existing data. Content fields create a version; service fields (`vector_status`, `summary_status`, `links_at`, `classify_at`) never do. The source and session of a change come from the execution context (`mcp`, `rest`, `job:<name>`, `merge`, `migration`), not from the model. The operator gets `GET /history/<entity>/<id>`, single-version reads and an explicit rollback (which writes a NEW version, `event=rollback`, and never rewrites history); the agent gets two compact handles, `history_list` / `history_get`.
* **Per-area switches** (№40): `AREA_SKILLS_ENABLED`, `AREA_TERMS_ENABLED`, `AREA_USER_FACTS_ENABLED`, `AREA_LOGS_ENABLED` gate the agent-facing surface and the background loops of an area; the operator REST surface and the data itself stay reachable with the area off. `/health.areas` and the startup event `areas_enabled` report the composition.
* **Assistant logs — a new area** (LSBN-F-10): the assistant records what it did and why — `logs_save`, `logs_get`, `logs_update`, `logs_delete`, `logs_search`, `logs_list` on MCP and `POST/GET/PUT/DELETE /logs` for the operator. An entry names the action, the reason, the object of the change (versioned entities need `entity_id`, external ones — a reference), the outcome and whether it was verified; it is versioned by №41, searchable by meaning and by slices (object, session, source, absolute dates) and is tied to the object's state (`active` / `deleted` / `unavailable` / `external`) at read time. The area is announced in the instructions with a short EN section.
* **Warnings surface** (№12): a background job looks for notes that probably belong elsewhere (classification confidence ≥ 0.80) and opens a warning instead of moving anything silently; `warnings_list` on MCP and `GET /warnings` + `POST /warnings/{id}/dismiss` for the operator, `by_kind`/`total` in `/health`.
* **Terms context normalisation** (№32): a background job clusters the contexts of one term, renames and merges only when the canonical context is unambiguous and the definitions match exactly, flags a cluster whose definitions differ instead of merging it, and soft-deletes the losing record on a key conflict (its history survives).
* **Slot outcomes and pause with a probe** (№39): one call outcome type (`unavailable` / `auth` / `empty` / `truncated` / `timeout` / `off_contract`); a broken slot pauses (`SLOT_PAUSE_SEC`, default 900) and is re-probed cheaply; an `auth` failure opens the breaker at once. Jobs wait instead of spinning: a per-pair attempt counter with a growing back-off, the "needs attention" state and a human hint; `pending` / `waiting` / `attention` per queue and `/health.slots` with the cause and the pause; `queue_waiting` is written for work that is waiting.
* **Output budget "as much as fits"** (№49): a slot spends the model's context minus the prompt (a reserve plus a protective ceiling) instead of a fixed small cap; the model type per slot (`*_MODEL_TYPE`, default `generation`), startup warnings about an inconsistent configuration, `done_reason`/`finish_reason` and thinking tokens in the journal. An empty answer is now distinguished from a truncated one (lsbdef-0008).
* **Decision slot** (LSBN-F-12): a third slot type next to generative and embedding — one call to `/v1/systemone` returns probabilities instead of text; the dedup judge and the node classifier can run through it, with a switch to the generative slot (unavailable / does not support decision / low confidence) that never loses a job, and probabilities reaching only the journal.
* **Memory handles** (№28–30, №24): a node hint (`namespace_hint`) when a note is saved without a namespace; an absolute `updated_after` freshness filter in search and listing; a "similar notes" hint (`similar`, ≤2) computed by FTS without a model; a note `comment` (≤250 chars) that never leaks into the note text, its summary, its vector or the semantic search, is edit-only (`memory_update`), is versioned as one content version, and gets its own search (`memory_comment_search`, `GET /notes/comments`).
* **Update hint** (№45): an optional hint about a newer release, from a local JSON file or an HTTP endpoint, at most once per period, off by default; `/health.update` reports the check state.
* **Backup verification and shipping** (№14): a snapshot is verified by opening it read-only with `sqlite-vec` (integrity check, required tables and vec tables from the registries); a bad snapshot is marked `<file>.bad` (never deleted silently) and excluded from rotation; `GET /backups`, `POST /backups/verify` and `/health.backups` expose the state. A verified copy can be shipped off the box by an operator command (`BACKUP_SHIP_CMD`, credentials stay outside the repository), confirmed by the reported size; a failure never hurts the local backups. "Restore from a snapshot" is documented in README and INSTALL.
* **Injected-text measurement and budget** (LSBN-F-11): one token counter (`cl100k_base`) and `scripts/measure_injected_text.py` measure the three places that cost context — instructions at connection, tool descriptions, call weight — and `tests/test_injection_budget.py` guards the budget (the surface must stay 31 tools).

### Changed
- The save-time `similar` hint (№30) now needs Dice 0.74 over significant words
  instead of 0.30: at 0.30 it fired on almost any note that shared a sixth of
  its vocabulary, so the hint was noise rather than a signal.

* **The injected text is shorter and measured** (LSBN-F-11): the instructions became a short EN core with actionable pointers, and all 31 tool descriptions were revised — 640 → 525 tokens for the instructions, 2702 → 2170 for the descriptions (−19.7%), with the mandatory rules kept verbatim; the node map is bounded (per-node description and total budget with a `(+N more …)` pointer), so a growing registry no longer inflates the instructions.
* **One hint per search** (№38): `skills_search` answers with a single `skills_get` hint per call instead of one per item.
* **`default` node description in English** (№34) with a one-time, idempotent backfill that never overwrites an operator's edit.
* **A note with a lifetime is never merged with others** (№46): the note lifetime (a TTL in `expires_at`, deleted by the system itself) is now announced in the instructions, a note with a TTL is excluded from dedup candidates and from merging, and its links are removed when the note is physically deleted.
* `/health` reports `areas`, `queues`, `slots`, `warnings`, `decision`, `backups`, `update` and the version, so one call answers "is the contour healthy and what is it doing".
* **Skill form limits raised** (decision of 2026-10-12, canon §6): `description` 250→400, `steps` 500→1000, `text` 4000→7500, `example` 1000→2000, one `extra` class field 500→1000 (all fields together 2000→4000) and the global `instruction_template` 1000→2000. Real procedures did not fit into the 3.0.0 ceilings: the model split a skill and duplicated the "how to execute" wording instead of writing the procedure. `name` (65 characters) and the announce block keep their limits; the injected-text budgets are untouched — `skills_save` stays at 119 of 120 tokens.

### Fixed

* **lsbdef-0008** — empty slot answers: a reasoning model that spends the whole budget on thinking is no longer indistinguishable from a broken or truncated call; the cause is visible in the journal and `/health.slots`, retries are bounded, and notes waiting for a summary are released when the configuration changes.
* **lsbdef-0011** — the `terms-normalize` job spun in a busy loop: a run that can change nothing (contexts whose definitions differ) was counted as progress, so the loop restarted immediately (~30 runs per second on the acceptance contour, starving the other jobs). Only real data changes count as progress now.
* **Decision is not asked when there is nothing to choose** (LSBN-F-12): the node classifier used to build a `choice` question from a single candidate (an empty node map leaves only the fallback label `generic`), which violates the live server contract (2–26 candidates in `criteria`) and came back as a 400 — a false `off_contract` that moved the decision-slot breaker and burned a call. With fewer than two candidates the decision call is now skipped entirely: one `decision_skipped` event (`reason=single_candidate`, `candidates`), the existing generative path is used straight away, and the `/health.decision` switch counters and the breaker stay untouched. This is "the decision is not applicable", not one of the three switch reasons.

## [3.1.0] - 2026-09-20

Release 3.1.0 — "Navigation and self-organization": the system starts to know its own space — reading a note returns a compact list of related notes from other namespaces, the `default` pile is sorted out in the background, a note merged by dedup is re-classified and re-homed, and every listing is bounded so a page no longer pulls more context than it needs. None of it adds an LLM call to the write path.

### Added

- **Note links** (lsb-0010): `memory_get` carries `links` — related notes from OTHER namespaces in the shape `{id, title, namespace, chars}`, at most 3. Level 0 is a lazy graph: a KNN query over the note's own vector, without any namespace filter. Level 1 is the persisted `links` table — one row per symmetric pair with `kind` (`mention` / `entities` / `cosine`) and `score` — recomputed incrementally without a single LLM call and backfilled once for an existing database by a background job. The note's own namespace is cut at output time, so a move between namespaces never invalidates a row.
- **Text volume in outputs** (lsb-0010): `memory_search`, `memory_list` and `memory_get` return `chars` — the size of the note text in characters. In chunk mode `chars` keeps its previous meaning (the sum of the chunks returned).
- **Listing pagination** (lsb-0013): one contract for the only two listings in the system — `memory_list` and `skills_list` — carries `total`, `has_more`, `next_offset` and `next_cursor` (reserved and `null` for now), plus exactly one soft "+N more — offset=K" hint. `skills_list` gains `limit` / `offset`; the MCP default and ceiling are 20 items and the operator REST ceiling is 50.
- **Background jobs framework** (lsb-0014): a job is now a registry entry (`JobSpec`: name, queue, interval or on-demand form, enabled flag, batch size, `process`, `queue_empty`, `wait_event`, `idle_hook`) served by one shared loop — a new job is a registration, not a copy of a loop. The five existing loops (`embedding`, `summary`, `judge`, `areas`, `expiration`) moved onto it with unchanged behaviour, and `links` and `nodes` were built on top. Every journal event carries a mandatory `job` field in one shape (`event` / `job` / `note_id` / `outcome` / `reason` / `target`), and a job can be woken by an event instead of waiting out its interval.
- **Queue observability** (lsb-0014): `/health` gains `queues` — `pending` and `oldest_pending_sec` for each observable queue — and a job that falls asleep on a non-empty queue emits `queue_waiting`, so a stuck queue is visible in the journal. A queue counts only the work its job can actually take in the next run.
- **Node self-organization** (lsb-0011, lsb-0012): the `nodes` job sweeps `default` for notes that fit an existing domain and re-classifies a note merged by dedup so the merged note can be re-homed. Both sources share one budget (20 items per run), and the internal markers `links_at` / `node_order_at` keep the sweep from looping.
- **Version in `/health`** (techdebt-0036): the endpoint reports the application version taken from the package, so a running image can be matched against its release tag.

### Changed

- **Summary limit is now 150 characters** (was "up to 30 words"): the prompt and the hard cut at the save point both measure characters, and the fallback excerpt is cut to the same 150. Summaries already stored and longer than the limit are regenerated in the background at startup rather than truncated in place.
- **MCP surface**: still **21 tools** with the same names and the same contracts; only the outputs of `memory_get` / `memory_search` / `memory_list` / `skills_list` are enriched with the fields above.
- **Configuration**: new environment variables for the job switches (enable / interval / batch per job), the link thresholds and pool sizes, and the MCP/REST listing ceilings and default — all validated at startup.
- **Documentation**: README, INSTALL and CONFIG brought in line with the 21-tool surface.
- **Acceptance pipeline** (techdebt-0036): the release image carries a label with its git commit, the E2E run is idempotent (probes are created and cleaned up by marker) and waits on observable state — `/health` counters and queue ages — instead of fixed sleeps, and a slot check refuses a run while the model proxies are unavailable.

### Fixed

- **A link kind could be downgraded on recompute**: a pair keeps the strongest kind by priority (`mention` / `entities` / `cosine`) and is never lowered by a later pass, and level 1 is not served for a note whose links have not been recomputed yet.
- **Queue counters counted work a job could not take**: `pending` now counts only tasks the job can pick up in the next run — a reclassification task waiting for its summary no longer keeps the queue looking busy.
- **Lost wakeup after a queue clear**: the `links` and `nodes` jobs re-check their processable queue after clearing, closing a window where a newly pending item could sleep until the next interval.
- **A finished summary did not wake the node-order job**: the post-merge reclassification task is woken as soon as its summary is ready, so a merged note no longer waits out a full interval.
- **A fresh pending note did not wake the embedding loop**: the loop is notified on insert instead of waiting out its interval.

### Upgrade

- Drop-in: start the new image over the existing database. Migrations are idempotent and run at startup from any earlier version (the `links` table and the `links_at` / `node_order_at` markers); notes, namespaces, areas and their indexes are untouched. Links are backfilled in the background, and summaries longer than 150 characters are regenerated in the background. Reverting is a matter of restoring the pre-upgrade database snapshot.

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
