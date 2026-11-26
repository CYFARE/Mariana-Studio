# Changelog

All notable changes to Mariana Studio are recorded here, newest first.

## v2.3.3 — 2026-11-24

### Agents

- Swarm tools now use a run-scoped read-result cache with single-flight
  duplicate suppression: two members reading the same file or fetching the same
  page share one execution, and a successful write clears the cache before any
  later read. Only known read-only tools are cached, results are bounded and
  in-memory, failures are never cached, and network entries expire after 30 s.
  Independent read batches execute in parallel while writes stay ordered; the
  result reports cache hits, saved time and stored size.
- The leader has a free hand to plan a large team: the safety ceiling is 64
  agents (default 16 in AML and 8 in Chat). Model calls are metered separately
  from tool executions (defaults 16 and 32 concurrent) so a 64-agent plan queues
  safely instead of firing every provider and disk call at once. Chat and IDE
  defaults were raised to 8 and 16 agents. Planning output grows with the team
  size, and reports from more than 12 agents are condensed in parallel groups
  before the final synthesis so no member is silently dropped. One read pool
  serves every member, so a large team cannot create hundreds of per-turn
  threads.
- Member prompts now put the immutable instructions, tools and briefing first
  and mutable state (peer digest, blackboard, plan, inbox) in the user turn.
  The system prefix is byte-identical across phases and tool turns, so provider
  prompt caching can reuse it: OpenAI receives `prompt_cache_key`, llama.cpp
  receives `cache_prompt`, other compatible providers use their automatic prefix
  cache, Anthropic receives `cache_control` on the system/tools block, and
  cached-token usage is parsed and reported. A rejected cache parameter is
  retried without it.
- The leader skips planning when the caller already supplied complete agent
  tasks and tool grants; a feedback round where every review scores 4+ with no
  open issue stops without a leader decision. With a reasoning effort set,
  routine tool turns reuse low effort while the first turn and wrap-up keep the
  configured level; the swarm node exposes this as `adaptive_effort` (default
  true, false sends the configured level on every turn).
- Swarm snapshots and results expose `cache` (tool hits, saved time, stored
  bytes, prompt-cache hit rate); Chat swarm cards and the IDE's bounded swarm
  observation show the hits and hit rate. The cache is per run and never
  written to disk.
- The designer metadata for a swarm node no longer lists the `goal` field
  twice, and the protocol test now rejects duplicate field keys.
- AML is now v1.9 (additive; every 1.8 document runs unchanged). AMP 1.10
  discovery reports the new field, and its specification documents the `cache`
  result object.
- A process-wide hot cache keeps decoded workspace files in system RAM, keyed by
  absolute path plus `mtime_ns`/size so a changed file is never served stale.
  Its budget is a fifth of available memory (64 MiB–2 GiB), overridable with
  `MARIANA_HOT_CACHE_MB`; large and binary files are excluded and writes or
  shell commands that may change files invalidate entries. `cache` now reports
  `hot_hits`, `hot_misses`, `hot_entries`, `hot_chars` and `hot_budget`.
- Workspace tools handle real projects: `file.read` lists a directory instead of
  failing, follows symlinks that stay inside the sandbox, detects binary files,
  streams files above 8 MB by line range, and suggests close paths plus nearby
  entries when a path is missing. `file.list` accepts a folder, nested pattern
  and recursion; `file.tree` reports directories too, supports depth and up to
  20,000 entries.
- Context compression now uses CRAFT (Context Reduction with Anchors, Facts and
  Traces). Observations are ordered newest-first, tool results are compressed
  structure-aware, compacted turns become episode cards naming tools, anchors,
  errors and decisions, and a run-wide evidence ledger shares paths/lines seen
  by any member. Handles keep a 260-character summary and anchor list while the
  full text remains retrievable. Default budgets increased to 24,000 prompt
  characters, 8,000 observation characters, a 400,000-character context bank,
  96,000 characters per agent and 32,000 per retrieval slice; `cache` reports
  compression ratio, episode count, stored items and evidence anchors.
- `file.search` uses `git grep` in a Git work tree (respecting `.gitignore` and
  searching in milliseconds), falls back to the Python walker for non-Git
  folders, raises the scan ceiling from 2,000 to 50,000 files, and treats an
  invalid regular expression as a literal search instead of failing. Search
  results report their engine and the fallback.
- Swarm tool arguments are normalised before dispatch: `file_path`/`filePath`
  map to `path`, `startLine`/`from`/`to` to line ranges, `q`/`glob` to query and
  include, `cmd` to command, numeric strings to numbers, and quotes/backslashes
  are cleaned. Equivalent spellings now share one cache entry and one executor
  call.
- A leader/verifier reply that contains only DSML or tool-call markup is never
  used as the final answer; synthesis falls back to the members' work. A member
  whose reasoning consumes the visible output budget gets one compact,
  evidence-only repair prompt instead of returning raw observations.
- Short runs reserve 15% of their budget for synthesis, no longer extend a 6-8
  step member budget, give members a visible step-budget warning, and tell the
  planner not to add an agent that only waits for every other member. Member
  prompts put all shared instructions and briefing first and member-specific
  tools/identity last, so the first member warms the provider prefix cache for
  the rest. Planning uses low effort on runs of 300 s or less.
- A synthetic 150 ms-per-read benchmark measured 0.151 s for two identical
  reads (one executor call), and four distinct reads at 0.602 s serial versus
  0.154 s when batched. This measures the tool/cache path only; it is not a
  matched end-to-end agent benchmark.
- Live checks used the saved DeepSeek endpoint against the 3.5k-file Shotcut
  checkout. A read-only bug audit completed in 156 s with 46 tool calls, zero
  tool failures and a verified seven-row table; a targeted three-agent audit of
  two source trees completed in 56 s (6 s planning, 27 calls, zero failures)
  with a nine-row table and its rejected candidates listed separately; a
  four-agent ownership audit completed in 93 s with 16 calls and zero failures.
  A documentation swarm updated only `README.md` in a disposable copy in 188 s.
  These are observed local runs, not a matched comparison with another coding
  agent.
- Native tool markup can no longer become an answer. The tolerant DSML reader
  accepts lost slashes, duplicated closers, unclosed blocks, truncated markers,
  `<parameter name="file.read">` envelopes and an `invoke name="plan"` body;
  a batch sent as one JSON array inside `<calls>` or `<invoke name="calls">` now
  executes every call instead of only the last JSON object. A reply that still
  cannot be parsed is repaired with a concrete JSON example, and the final text,
  transcript and swarm member output run through one sanitiser that returns
  prose or nothing, never markup. Replaying a real 32-turn DeepSeek audit raised
  parsed calls from 1 to 2 on each batched turn (the model had repeatedly
  observed "only one result returned?"); a targeted read-only rerun against
  Shotcut then finished in 240 s with a 2,157-character report and no markup in
  the output.
- The leader now plans for parallelism rather than a small team: size the team to
  the work, give every independent workstream its own member and reserve
  `depends_on` for a real hand-off, so a broad audit starts with a wide first
  wave instead of a chain. The IDE's swarm hint says sixteen agents (was six),
  and the IDE runs eight read-only tool calls concurrently (was four).

### Interface

- The IDE context ring moves during a run. The server publishes a `budget` event
  after every tool round with used tokens, the context window, percent and
  remaining context; the composer still substitutes the live draft locally
  between events, so typing and tool work both move the ring before the run
  ends.
- IDE panes have no fixed maximum width or height: the explorer, editor and
  agent can each be dragged until the next pane reaches its minimum, so one
  pane can fill the window. Preferred sizes are still remembered and restored.
- Agent-panel tool cards show a failure callout with the error and recovery
  hint, and offer **Copy result** and **Copy input** beside **Open file**. The
  streaming status line distinguishes a tool call in flight, a plan and a final
  answer.
- **Copy conversation** sits above **Export as Markdown** for every
  conversation. Both include tool calls, tool output and errors, so a pasted
  transcript keeps the evidence.
- IDE interaction fixes: choosing an `@` file mention inserts the path without
  leaving the trigger character behind, the mention popover closes when the
  project changes or the view is left, a conversation from another project
  opens in that project, and a draft typed while a send is in flight is kept.
  The last answer appears immediately and survives a failed session reload,
  and pending tool cards are marked ended rather than running forever. Reload
  on a stale tab asks before discarding unsaved edits, and rapid reopen or
  duplicate-open requests cannot create a second tab or overwrite a newer
  buffer.
- Package docstrings, `mariana --help`, and the citation metadata now describe
  Mariana as a one-stop research and development studio for language models,
  matching the README. The technical model documentation keeps its DRV-LM and
  hybrid diffusion-AR terminology.

### Reliability

- Tool arguments are normalised in one shared helper used by the IDE, the flow
  runtime and the swarm: `file_path`, `filename` and `path` become `path`, `q`
  becomes `query`, `offset` becomes `start_line`, `limit` becomes `end_line`, a
  bare string becomes the tool's primary argument, list and number spellings
  are coerced, and `file_search` resolves to `file.search`. The swarm no longer
  keeps a private copy of the alias table, so the three callers cannot drift.
- An incomplete native tool call is returned to the model with a repair round
  that names the missing keys instead of running and failing. After two rounds
  the call runs so the executor's precise error is visible, and the repair
  budget resets only after a complete call, so a stubborn model cannot be
  repaired forever.
- IDE bug sweep. Session lists apply their recent-window after filtering by
  project, so a busy project cannot hide another project's older conversations;
  deleting a project reserves it until its files are gone, so a run cannot
  start in a half-deleted workspace; change snapshots are written atomically; a
  failed `file.edit` no longer creates empty parent folders; cancelling a run
  interrupts the network retry wait; pinning a file no longer filters the
  project map down to that file; hidden paths are removed from both shapes of a
  `file.list` result; and a corrupt session timestamp no longer breaks the
  conversation list.

### Training

- New AXIOM corpus codec (`mariana/axiom.py`, `train --axiom`). Its English
  semantic layer replaces connectors with one-byte mathematical operators,
  canonicalises number words and typography, and iterates to an idempotent
  normal form. Its codec layer makes the technique safe on any dataset.
- AXIOM `auto` profile detects the corpus from a spread sample and selects the
  smallest codec: a lossless 129-entry dictionary for text (words, phrases,
  Unicode math symbols, emoji, LaTeX commands and CJK n-grams), 4-bit DNA
  packing, 5-bit protein packing, or an identity codec for binary and
  incompressible data. Non-English and mixed-language corpora use the
  dictionary without the English semantic layer. `compact` and `reversible`
  force English connectors; `lossless` keeps text exact.
- The selected codec is stored in the checkpoint (`axiom_codec` JSON) and in
  the prepared-corpus manifest. `gen`, `think` and `bench` encode their prompt
  and decode their output with it; a resumed run validates the corpus against
  the stored codec and refuses a mismatch. `--show-axiom` prints glyphs,
  dictionary previews or the packing label.
- AXIOM-prepared corpora are cached under `trained-model/axiom/`; large corpora
  are sampled for codec selection and encoded line by line. Studio exposes the
  toggle and four profiles in Advanced settings for DRV-LM runs and rejects
  AXIOM with NCM.
- New `scripts/axiom_corpus.py` to measure, inspect or pre-convert a corpus, and
  `docs/whitepapers/axiom.md` plus `docs/axiom.md` for the design, measurements
  and evaluation protocol.
- AXIOM 2.2 bug sweep and escape mode: streamed and in-memory rewrites now
  agree on lone CR and mixed line endings, canonicalisation freezes line
  protection so filler removal cannot create or reclassify a fence,
  ambiguity-weighted language scores plus sentence votes stop English and
  Spanish prose being called mixed, truncated cache payloads and in-place
  rewrites are rejected, and a literal operator byte is escaped as
  `0x00 <byte>` instead of disabling the connector layer. On a 206 MB English
  corpus with a few stray C0 bytes this raised the saving from 15.5% to 26.0%
  and kept 6.2 M connector hits.
- NCM generator LR now defaults to `1e-3` (was `3e-4`), including the integrated
  `--ncm-generator-lr`; Studio Smart Options scales from `1e-3`. On a 7.6 MB
  English sample, 800 generator steps reached teacher code accuracy 0.312 versus
  0.229 at the old default; free-running code accuracy also rose. Tokenizer LR
  is unchanged.
- NCM evaluation prints `free_bytes` alongside greedy `FREE` code accuracy: the
  fraction of generated bytes that match the tokenizer's decode of the true
  codes. It separates generator drift from the frozen tokenizer ceiling.
- NCM short-run sweep. A single global concept level is high-entropy on varied
  text and drives error accumulation; the 12-level paper ladder reached 0.080
  free-byte accuracy after 800 steps, while `64,96,128` reached 0.150 with
  3 NFE and `32,...,128` reached 0.123 with 7 NFE. A single-level ladder is not
  recommended (0.071: every position loses the informative draft). The runbook
  documents the measured trade-off; the paper ladder remains the default.
- NCM opt-in stabilizers: `--prefix-noise` and `--prefix-dropout` perturb the
  cumulative prefix latent while training, and `--byte-weight` /
  `--free-byte-weight` add a frozen-decoder byte-reconstruction auxiliary. Noise
  0.5 raised free code accuracy 0.0053 -> 0.0071 on the paper ladder; both byte
  auxiliaries changed the code/byte trade-off and were mixed across ladders, so
  all four are off by default.
- NCM bug fixes: `RMSNorm` casts its weight to the activation dtype, restoring
  the fused kernel under bf16/fp16 autocast; code caching now puts the frozen
  tokenizer in eval mode and restores its previous mode, so `--cache-codes` is
  deterministic even for a tokenizer with dropout.
- DRV-LM checkpoints are written atomically (temporary file plus `os.replace`),
  and Ctrl+C saves the last completed optimizer step, so a crash, full disk or
  interrupt no longer leaves a truncated checkpoint.
- Resume is validated before training: `--resume` requires the checkpoint for
  the stage being continued, rejects an NCM checkpoint in DRV-LM, and `--steps`
  must exceed the saved step; `--init-from` and `--resume` are mutually
  exclusive. A shape mismatch reports the config hint instead of a raw
  `load_state_dict` error.
- Model geometry is checked before allocation and at the Studio boundary:
  `dim` must divide by `heads` into an even RoPE head size, and context, block
  size and loop count have explicit lower bounds.
- A non-finite DRV-LM gradient update is dropped with a warning instead of
  stepping NaN weights. NCM keeps its documented abort-on-nonfinite behavior.
- FlexAttention block masks now receive the per-row causal-only annealing
  decisions, which were previously ignored on the `--flex` path.
- Studio reads checkpoints, creates overwrite backups and copies NCM tokenizer
  files without holding the job-state lock, so `/api/state` and the UI stay
  responsive during large backups. It reports a crashed paused run as failed,
  terminates and reaps a trainer if output reading fails or Studio closes, and
  turns corrupt checkpoint/load errors into actionable messages.

## v2.3.2 — 2026-11-22

### Agents

- New `assert` guardrail node: a safe expression must be true, and a false
  result fails the run with an interpolated, actionable message.
- New `validate` guardrail node: structured output is checked against a bounded
  JSON Schema subset with precise `path: message` errors. `warn` mode exposes
  `valid`/`errors` to a following condition instead of failing the node.
- `llm` and `agent` nodes accept `fallback_model`. A failed main model call is
  retried on the fallback, and an agent keeps using that fallback for its
  remaining tool rounds.
- Swarm leaders now plan testable success criteria, a short plan for every member
  and optional `depends_on` edges. Members without dependencies run together;
  dependent waves wait for their inputs. Cycles are removed with a warning.
- New swarm fields: `criteria` (caller acceptance checks), `effort`
  (`low`/`medium`/`high` reasoning where supported) and `reflection` (one
  plan self-check before a member finishes).
- Member plans persist across turns and appear in prompts, snapshots, results and
  the swarm card. A member that finishes with open steps is asked once to
  complete or justify them.
- After a review round the leader reports `gaps`; synthesis verifies the success
  criteria and returns `unmet_criteria`. Both reach the result, warnings and UI.
- Read-only tool failures with a temporary error are retried once in swarms and
  flow agents. Writes are never retried. Deterministic failures carry a recovery
  hint, and an unknown tool id gets a closest-match suggestion.
- A member that repeats the same call three times receives an explicit anti-loop
  hint, and member state reports `tool_calls`, `tool_failures` and
  `tool_retries`.
- Swarm peer digests are ranked by relevance to the member's own task and
  identical results are sent once. A review prompt no longer repeats its
  target's result, a pipeline stage shown in full is excluded from the digest,
  and a review keeps the member's work summary instead of replacing it.
- A swarm retry after an empty or truncated turn keeps native tool calling, and
  a member may self-check an open plan once in every work phase.
- AML `agent` nodes compact older tool rounds into a ledger once the transcript
  passes about 24,000 characters, and repair a malformed or empty action with up
  to two correction turns instead of returning raw prose or `{}` as the answer.
- Swarm context reads are charged only for the slice returned; an offset at or
  past the end is reported without reserving pool space.
- AML is now v1.8 and AMP v1.10, with updated specifications, discovery metadata,
  the new `swarm_agent_plan` event, live `assistant_delta`/`reasoning_delta`
  events and a `swarm.reliability` description. The project version remains
  unchanged.

### Interface

- Chat streaming now shows a live turn row with the current phase (`Waiting`,
  `Thinking`, `Writing`), elapsed time, time to first token, an output rate and
  an inline Stop button. Reasoning streams into a disclosure that stays
  collapsed once the user closes it.
- The message area gains a `New output ↓` affordance while the reader is scrolled
  away from a live reply, and live Markdown carries an accent edge so streaming
  text is distinguishable from a finished answer without shifting the layout.
- The Agents run panel gains a **Live output** pane for the current model turn
  and every active swarm member, with reasoning, per-block status, character
  count, streaming rate, time to first delta and automatic follow when the
  reader is at the bottom. A copy button grabs the current live transcript.
  Token deltas are hidden from the event log by default; a **Hide token deltas**
  toggle brings them back for inspection.
- Agents canvas cards show a live one-line preview while their node writes, with
  a pulsing dot and an accent border. A swarm card names the active member count
  and the latest member's text.
- Run events now lead with a human-readable summary (started/finished, tool and
  path, plan, swarm review, log) and keep the full JSON in a collapsed
  **Details** disclosure. The run header adds a live indicator and a cumulative
  output-token badge. Swarm events carry their owning node id, so the canvas and
  event log can place them on the right card.
- Chat live reasoning keeps its open or closed state through the final render
  and shows a growing character count; the live row adds a running token
  estimate alongside elapsed time, first-token latency and estimated tok/s.
- A swarm run that ends without a leader synthesis now replies with a compact
  report: the outcome, one headline per agent that delivered, the agents that
  did not finish with their reason, and a pointer to the full hand-offs. The
  raw member concatenation no longer fills the chat bubble, and a synthesised
  answer is still returned unchanged.
- The swarm card now renders below the answer instead of above it. It gains an
  **Assembled** badge for runs without a leader synthesis, a **Time limit**
  badge, a collapsible **Run notes** list, a one-line result preview on finished
  agent rows, and a visible expand/collapse affordance on every row. The notice
  under a swarm reply shows one short note plus a count instead of a truncated
  concatenation of every warning.
- Chat's swarm settings now have a **Research depth** preset: Standard (5 min,
  10 rounds), Deep research (15 min, 16 rounds) and Extended (30 min, 24 rounds).
  Deep research is the new default; saved settings from the old hidden 5-minute
  default migrate to it. The server accepts up to 30 minutes and 24 rounds
  instead of the old 5-minute/12-round ceiling, and the mode tooltip and context
  line show the selected duration. A timed-out report points to the setting.
- When a Chat run writes or edits files, the last previewable file now opens in
  the preview panel automatically (when automatic previews are on), and every
  written file stays selectable there. HTML designs render script-free, code
  opens as source, and images open from a bounded data URL. A new read-only
  `GET /api/chats/{id}/file?path=…` serves conversation-workspace files with a
  resolved-path check, and previews survive a reload through the run's
  `changed_files`.
- The swarm card shows a live elapsed badge and pulsing activity dot, per-member
  token rate after a turn, and a dashed live preview that says “Preparing a tool
  call…” instead of flashing partial JSON. Live output auto-follows when the
  reader has not scrolled away.
- Home is now a compact command centre: one primary action, a four-item quick
  action strip, workspace counts, recent conversations and jobs, and system
  health. The oversized hero, duplicate stat tiles and separate runtimes card
  are gone.
- A missing chat-ready model is explained on Home with a direct route to
  Discover, instead of failing later in Chat.
- Mobile navigation is a proper disclosure: the menu button reports
  `aria-expanded`, opening moves focus into the navigation, `Escape` closes it
  and returns focus, and resizing to desktop closes it cleanly.
- The command palette now opens with workspaces and actions only; chat models
  join the list once the person types, so the resting palette stays short.
- The theme follows the operating system until an explicit choice is made, and
  the saved choice now survives reloads (it was stored as JSON but read as a
  raw string before).
- Smaller polish: calmer top bar, view-change fade, consistent pressed states,
  interactive list-row hover and a wider mobile drawer.
- Chat's empty state now has a clear recovery path for a missing or unavailable
  model and three prompt starters that fill the composer; the prompt menu uses
  the same helper.
- Mobile Chat gained a visible conversations button and a close button in the
  drawer; the drawer now fills the chat area instead of leaving a gap below the
  top bar.
- The Models empty state distinguishes an empty library from filtered results:
  filtered views offer **Clear filters**, while an empty library offers Discover
  and Add endpoint.
- Models and Fine-tune remember the last tab for the session, so returning from
  another workspace no longer jumps back to Library or Setup.
- Train and Fine-tune keep their primary action in a sticky footer while the
  form scrolls, so starting a long run does not require scrolling to the end.

### IDE

- Coding skills are now automatic: every skill's id, name and description is in
  the agent prompt (bounded to a small section budget, with any overflow ids
  still listed), and the prompt tells the agent to load the relevant ones with
  `skill.read` when a task needs them instead of waiting for the user to name a
  skill. The manual **Use** button is gone; the skills manager states that all
  skills are available automatically. Swarm members receive the same automatic
  skill list.

- Breadcrumbs above the editor show the workspace, folders and active file.
  Folders reveal and expand in the explorer; the row ends with copy-path and
  file-action controls.
- The command palette now lists IDE actions as well as global ones: quick open,
  save/save-all, find/replace, go-to-line, reopen/close/tab switching, panel
  toggles, focus mode, font scaling, reveal, pin-context, new file/folder/chat,
  project check, refresh and revert. `Ctrl+Shift+P` opens it directly.
- Quick open starts with per-project recent files, supports arrow/Home/End
  navigation, and marks results that are already open or edited.
- The tab strip preserves its horizontal position, scrolls the active tab into
  view, scrolls sideways with the wheel, and gains an open-files menu with
  active/edited markers, save-all and close actions. `Ctrl+Shift+T` reopens the
  last closed tab, and `Ctrl+PageUp`/`Ctrl+PageDown` cycle tabs.
- File and project tabs support Left/Right keyboard navigation; file tabs also
  support Home/End.
- The editor font scales between 12 and 22 pixels with `Ctrl+=`, `Ctrl+-` and
  `Ctrl+0`; changing it re-measures the gutter, overlay and virtualised window.
  The preference is remembered and applied per editor.
- The status bar keeps the file path and the cursor line/column in separate
  fields, so the path no longer disappears while typing.
- Focus mode folds both side panes and restores them in one step from the layout
  menu or the palette. `Ctrl+Shift+F` opens project search from anywhere in the
  IDE.
- The composer has a selection button that inserts the current editor selection
  as a path-labelled fenced block, capped at 40k characters.
- The empty editor offers Quick open, New file and New chat actions, and
  `Ctrl+Shift+?` opens a shortcut reference.
- A new **Outline** tab lists functions, classes, types, Markdown headings and
  HTML ids with nesting and line numbers. Rows jump to their line and highlight
  as the caret moves; a filter box narrows long files. `Ctrl+Shift+O` opens the
  searchable **Go to symbol** picker, and both actions appear in the palette.

### Fine-tuning

- Hugging Face fine-tuning now supports 4-bit NF4 and 8-bit QLoRA. The frozen
  base is quantized with double quantization, LoRA updates stay in float32, and
  older CUDA cards use fp16 with a gradient scaler. A 7B base that needed ~14 GB
  in bf16 fits a 6-8 GB card at short context.
- Added the **Low-VRAM LoRA (4-bit)** preset: batch 1, accumulation 8,
  gradient checkpointing, quantized weights and an adapter-only result. The
  preset is selectable in Fine-tune and documented in
  [`docs/low-vram-finetuning.md`](docs/low-vram-finetuning.md).
- Auto memory tuning reads the base parameter count and total VRAM before
  loading, switches to 4-bit when uncompressed weights do not fit, enables
  gradient checkpointing, and halves the micro-batch while raising accumulation
  to keep the effective batch. It logs every decision.
- The training loop recovers from a real CUDA out-of-memory: it empties the
  cache, shrinks the micro-batch (or enables checkpointing at batch 1) and
  restarts the step. An unrecoverable case stops with 4-bit and context advice.
- Validation batches shrink or skip with a warning instead of failing the run.
- Cross-entropy is computed in 256-token slices, so a large vocabulary no
  longer materialises a full float32 logits tensor for the batch.
- Chat data is streamed and text is tokenized into a compact int32 token store;
  `max_tokens` and `max_examples` caps bound memory on very large datasets and
  are adjustable in Advanced.
- Training now uses shuffled epoch batches with length grouping instead of
  independent random draws, so every example is seen and padding is lower.
- A requested merged model is built on CPU after training. If the merge runs out
  of RAM, the adapter is still published and the job completes with a warning.
- The plan estimates peak VRAM, shows a 4-bit peak when the base does not fit,
  warns about full or final-layer training on a small card, and explains a
  missing bitsandbytes or accelerate install before the run starts.

### Performance

- Studio's Python transport now keeps HTTP/1.1 connections alive, serves
  gzip-compressed text assets, and sends ETag/Last-Modified validators, so a
  reload transfers 304s instead of every script and stylesheet.
- Assembled `index.html`, every view include and every static file are cached in
  memory and invalidated by mtime, eliminating repeated disk reads and template
  splicing on each asset request. Error responses close the connection so a
  keep-alive socket can never be left with an unread body.
- The loss chart gets a WASM min/max decimator: long runs keep the spike in every
  pixel column, reducing a 100k-point series to a few thousand drawable points.
  Browsers without WebAssembly use the identical JavaScript fallback.
- When WebGPU is available, the chart grid and polylines render on the GPU and
  are composited beneath the 2D labels and markers. A small startup read-back
  self-test verifies the pipeline; failures, device loss or a blank frame fall
  back to Canvas 2D immediately.
- Chat live rendering coalesces Markdown paints as before; reasoning text is
  painted at most every 100 ms and the elapsed/rate row updates at 4 Hz without
  rebuilding the message.
- AML `llm` and `agent` streams now emit coalesced `assistant_delta` and
  `reasoning_delta` events about eight times a second, so the run panel can show
  live text without an event per token. Deltas are trimmed first when a run's
  bounded history overflows.
- The run live pane rescans events only when the latest sequence changes, and
  hidden delta rows are skipped before filtering and row construction.
- The IDE now sends the full task once as the user message and keeps only a
  600-character anchor in the system prompt, removing up to 2,400 duplicated
  characters from every model call in a long run. A truncated active-file or
  pinned-file preview is explicitly labelled instead of silently ending mid-file.
- Each IDE tool result is bounded once and the same value is used for the
  transcript and the event, removing a second JSON serialisation on every call.
- Swarm context grants use O(1) accounting instead of summing every member's
  grant on each request, and peer digests avoid repeating results already shown
  in the prompt.
- Chart extents are now found in one pass instead of allocating `steps`/`values`
  arrays and spreading them into `Math.min`/`Math.max`, which also removes the
  argument-limit failure on very long runs.
- The WebGPU path builds line and grid vertices into one flat `Float32Array`,
  reuses a grow-only vertex buffer across frames and rewrites the resolution
  uniform only when the chart size changes, removing per-frame allocation and
  upload work.
- `application/wasm` is compressed like the text assets, so the decimation
  kernel transfers gzipped alongside scripts and stylesheets.
- IDE coding runs now execute independent read-only tool calls in parallel, up
  to four at a time. Writes, deletes, shell commands and swarms stay strictly
  ordered, and results are still returned in the order the model requested.
- `file.read` now returns up to 200 lines or about 3,600 characters, reports the
  real `end_line`, and adds `next_start_line` for continuation, avoiding wasted
  re-read rounds on larger files.
- Tool-observation clipping preserves the exact included line range; shortened
  reads no longer leave a stale `end_line` that could make an agent skip code.
- Saved conversation lists use an mtime/size-keyed cache, removing repeated JSON
  parsing when a project refreshes its session list during long coding runs.
- The code editor rebuilds its line index and finds the edit boundary in a single
  pass instead of two full-buffer scans per keystroke.
- Find-in-file refresh is debounced, and bracket matching is cached by text
  version, grammar and selection, so scrolling and typing stay responsive in
  large files.
- Applying an agent edit reads the changed file once instead of twice.
- Chat streaming now repaints the live Markdown at most once per 80 ms instead
  of once per token, coalesces scroll-to-bottom to one frame, and pauses paints
  in a hidden tab until it is visible again. The final reply is still rendered
  authoritatively when the stream ends.
- Swarm cards repaint at most about eight times a second and immediately on a
  terminal event, so agent chatter no longer rebuilds the card every frame.
- Conversation search, model-library filters and the IDE session search are
  debounced; typing no longer rebuilds every sidebar row or model card per key.
- Chat renders are now incremental: sending or finishing a message rebuilds
  only the article that changed instead of re-parsing and re-highlighting the
  whole conversation. The streamed Markdown is painted block by block, reusing
  completed blocks and caching closed code fences, so live output scales with
  the text that is still growing rather than with the whole reply.
- The IDE live preview uses the same incremental Markdown path and resets its
  block cache only when the bounded preview window slides.
- The chat sidebar keeps a snapshot, reuses rows by conversation id, and skips
  rebuilding when the list and search have not changed. Saving a message only
  refreshes the row whose title, preview or timestamp changed.
- The Agents canvas reuses node and edge SVG elements between renders. Node
  status, selection, labels and positions are patched in place, and edges are
  rebuilt only when the connection structure changes; editing node data no
  longer recreates the whole diagram. Selecting an edge repaints two paths
  instead of every connection.
- Run event history merges each new batch into the existing ordered list in a
  single pass, replacing the per-frame map, sort and full-history spread. The
  event panel also prepends only newly arrived rows and trims the oldest,
  instead of rebuilding 140 rows and their listeners on every snapshot.
- Historical chat and IDE messages highlight code in idle slices; the visible
  two turns stay eager, so opening a long conversation paints immediately and
  fills in syntax colour without blocking input.
- Preview artifacts are cached per message, so the conversation-wide artifact
  list no longer re-scans every code fence on each preview or stream finish.
- Node summaries are cached against a document revision, model catalog and
  agent list; validation and run repaints no longer rebuild every summary.
- The minimap redraws at 10 fps during drag, pan and zoom instead of every
  frame, and pointer-driven edge hit testing is coalesced to one animation
  frame with a node map instead of repeated linear searches.
- Arrow-key nudging and drag completion update only the inspector's position
  value, leaving every other field and its focus untouched.
- Chat, IDE tree and transcript scroll listeners are passive, so scrolling is
  never delayed by the inline handlers.
- The Agents run panel is built once per run. Status, steps, elapsed time,
  variables and output update in place, event rows are rebuilt only when the
  event list or filter changes, and the inspector is rebuilt only when its
  selection, node outcome or breakpoints change. Editing a field or typing in
  the event filter keeps focus while a run streams.
- Run snapshots repaint node status classes in place instead of recreating
  every node element, keeping large flows responsive during a run.
- The IDE project tabs, file tabs and workspace picker skip rebuilding when
  their visible state is unchanged, which removes redundant DOM work on every
  project switch and status event. The model source picker is reused when the
  catalog has not changed.
- Symbol extraction is cached per tab and only runs while the Outline panel is
  visible. Typing coalesces refreshes to one scan per 250 ms, and identical
  outlines are not repainted.

### Reliability

- Parallel tool batches preserve the model's call order, transcript order and
  per-call signatures. A batch containing a write, delete, shell command or
  swarm falls back to ordered execution.
- Parallel tool workers are daemon threads, so a slow request cannot keep the
  Studio process alive during shutdown or cancellation.
- Read-result truncation now distinguishes a whole-line cut from
  `content_cut_midline`, giving the agent an actionable hint for minified files.
- Explicit find/replace actions flush a pending debounced search before using
  match offsets.
- Session-list cache entries are invalidated on save, rename, duplication and
  deletion, including workspace removal.
- Coalesced chat and swarm paints are cancelled on completion, error, abort and
  view changes, so the authoritative final render always wins and never writes
  into a conversation the user has since left.
- The Agents run filter and human-input fields keep their focus and caret while
  snapshots arrive, because the panel is no longer recreated per snapshot.
- Selecting or clearing an edge updates only the affected paths, and starting
  a marquee now clears the previous edge highlight instead of leaving it stale.
- A fresh incremental Markdown stream clears leftover content first, so a
  preview that switches between raw tool-call text and Markdown can never
  duplicate or strand blocks.
- Closed-tab history and recent files are per project; forced closes caused by
  deletion or a missing file are not remembered. Tab rebuilds restore the
  horizontal scroll position and then reveal the active tab.
- Outline rows keep working after edits: a changed text invalidates only that
  tab's cached symbols, and jumping from a rendered preview switches the tab to
  Code before moving the caret. Binary files and unknown languages show a clear
  empty state instead of stale rows.
- A transient read-only tool failure is retried once with its attempt count in
  the result; writes are never retried, so a timeout cannot duplicate an edit.
  Tool failures now carry a recovery hint, and unknown tool ids suggest the
  closest granted id instead of only listing the allow-list.
- A swarm dependency cycle cannot deadlock a run: unknown edges, self-edges and
  cycles are removed before work starts and reported as a warning.
- Asset paths are normalised to POSIX separators after `resolve()`, so the
  "view fragments are never served standalone" check also holds when Studio runs
  on Windows, where `Path.relative_to()` otherwise returns backslashes.
- Cancelling an IDE run now stops waiting for parallel read workers and for tool
  retries as soon as the cancel flag is set, instead of blocking on a slow read.
- `file.edit` refuses a non-UTF-8 file with an actionable error instead of
  decoding it with replacement characters and writing corrupted bytes back.
- A review no longer overwrites the reviewer's work summary, so the swarm card
  and synthesis keep the member's findings.
- A member cannot silently skip its own plan: finishing with open steps triggers
  one reflection turn, and a still-open plan is recorded in the run warnings.
- Coalesced live deltas are split into events of at most 4000 characters, so a
  large burst keeps every character; previously only the first slice was sent
  and the tail was lost. Live panels count the full stream, not the preview
  window.
- An empty model reply gets a corrective turn, and an empty `{"final": ""}` is
  repaired instead of silently ending the run with no summary.
- A swarm member that returns an empty final is asked once for its complete
  finding before the fallback observation is used.
- Events from a worker that arrive after an IDE run reached a terminal state are
  ignored, so the event log, counters and replays cannot grow past the run's
  `run_<status>` event.
- Agent runs now append their terminal event and flip the final status under one
  lock. A stream waiter can no longer observe `finished` before `run_<status>`
  exists, and `LiveRun.event()` ignores late worker events the same way the IDE
  run does.
- Swarm event delivery is serialized, so a worker callback cannot land after
  `swarm_end` and reopen a finished run's stream. The time budget restarts when
  `run()` begins, and a member finish arriving after cancellation or timeout is
  ignored instead of rewriting the result.
- Coalesced swarm member deltas are split into events of at most 4000 characters,
  so a large burst reaches the live card intact.
- A swarm member that returns an empty reply is asked for its findings instead
  of finishing with no output. Correction turns never append an empty assistant
  message, which strict providers reject, so one bad reply cannot fail the run.
- Swarm native tool schemas now match the member's actual grant (and review
  calls see read tools only), instead of offering every tool the swarm owns.
- A revision prompt includes the member's previous answer and its result handle,
  so feedback is applied to the work being revised rather than to an unseen
  answer.
- Repeated empty replies move to the tool-free wrap-up instead of finishing the
  member as successful with no finding; coordinator nudges are filtered out of
  the partial-result fallback so they are never presented as evidence.
- A JSON turn with no usable action now returns a corrective observation. A
  partial reply that hits the output limit is kept when its larger retry returns
  nothing, and a final truncation is recorded as a warning.
- Member ids that would shadow channel participants (`leader`, `all`, operator)
  are renamed, and explicit no-tool grants are no longer widened by task text.
- Absolute and escaping write scopes are rejected before a member starts; a
  plan that supplied only such scopes records a read-only warning. Malformed
  tool parameter metadata no longer breaks prompt construction.
- Review-assignment failures are isolated so one bad target does not cancel the
  reviewer's remaining assignments, and compacted history keeps the handles from
  earlier compactions.
- Swarm leaders now see the run's total time budget in the plan prompt and are
  told that pipeline stages run strictly in sequence and share that budget. For
  fan-out then verification they are directed to `map_reduce` with `depends_on`,
  so collectors run together and the verifier waits for both. A pipeline plan
  with no dependencies is flagged in the run notes. Members in a parallel plan
  now receive the full output of the teammates they depend on.
- A swarm now reserves the last part of its time budget for the finish: tool
  rounds stop at the work deadline (10% of `max_seconds`, at least 5 s and at
  most 90 s), members answer from observations they already collected, later
  dependency waves and peer review are skipped with a warning, and the leader
  still synthesises. If the reserve is also gone, the answer is assembled from
  the members' work and the run completes with `timed_out: true`; only a run
  where no agent produced output still fails with the time-budget error. This
  fixes a tool-looping member turning a partly successful swarm into a total
  failure.
- A worker abandoned at the work deadline can no longer reopen its member or
  rewrite the assembled result, and skipped members are finalised before
  synthesis instead of staying `running`.
- Swarm results, snapshots and `swarm_end` expose `timed_out`, and the swarm
  card shows a **Time limit** badge. A run that succeeded with one failed member
  reads **Done · N failed** instead of **Finished with failures**.

### Documentation and tests

- Added tests for gzip/ETag/304 static delivery, the immutable asset cache,
  WebAssembly decimation equivalence, compressed WASM delivery and the CSP's
  `wasm-unsafe-eval` grant.
- Added [`docs/web-acceleration.md`](docs/web-acceleration.md) covering the WASM
  kernel, WebGPU fallbacks, asset caching and the `chart.wasm` rebuild command.
- Updated `docs/ide.md` for parallel read batches and the new `file.read`
  continuation contract.
- Added regression tests for parallel read execution, ordered write/read
  batches, accurate read continuations and session-cache refresh.
- Added symbol-outline tests for Python, JavaScript arrow functions, Markdown
  fences, HTML ids, unknown languages and the outline/go-to-symbol wiring.
- Updated the AML and AMP specifications, `docs/agentic.md` and the agents
  whitepaper for guardrail nodes, swarm planning, dependency waves, reasoning
  effort and tool retry rules, and added schema/runtime tests for the new agent
  features.
- Added tests for relevance-ranked and deduplicated peer digests, review-target
  exclusion, pipeline prompt deduplication, native tools on a retried swarm
  turn, per-phase plan reflection, review summary preservation, flow-agent
  compaction, malformed-action repair, non-UTF-8 edit refusal and labelled
  truncated previews. Updated `docs/agentic.md` and `docs/ide.md` for the new
  context and repair behaviour.
- Added tests for coalesced AML model deltas and the run panel's live-block
  derivation, plus source and stylesheet checks for the Chat live turn row,
  live reasoning, event summaries and delta filtering. AMP is now v1.10 with
  the additive `assistant_delta`/`reasoning_delta` events documented in the
  specification and discovery metadata.
- Added regression tests for large-burst delta delivery without truncation,
  empty-reply and empty-final recovery in the agent loop and swarm, and late
  IDE events after a terminal run event. New tests also cover an empty member
  reply without an empty assistant message and the terminal-event invariant for
  concurrent waiters on an agent run.
- Added [`docs/low-vram-finetuning.md`](docs/low-vram-finetuning.md) with the
  QLoRA quick path, expected 6 GB/8 GB footprints, large-dataset caps, merge
  behaviour and troubleshooting.
- Added low-VRAM planner tests for 4-bit selection, explicit settings, batch
  shrinkage and 6 GB/8 GB reports; dataset tests for the compact token store,
  epoch coverage and streamed text/chat data; and trainer tests for the chunked
  loss equivalence and new job options.
- Added swarm regression tests for a fresh run budget, member-scoped native
  tools, burst delta delivery, reserved member ids, revision context, absolute
  write-scope rejection, retained compaction handles, empty-reply wrap-up,
  no-fake-findings fallback, malformed-action correction, truncation/partial
  retry handling, late-finish protection and the new-phase live-text reset.
- AMP now documents the reserved channel roles and the member-id rename rule.
- Added swarm regression tests for the bounded finish reserve, tool refusal past
  the work deadline, skipped peer review, synthesis-timeout fallback and a
  timed-out tool loop that still returns the productive member's work. Updated
  the AMP `swarm_end` payload and reliability notes and `docs/ide.md` for the
  work deadline and graceful timeout behaviour.
- Added Chat reply tests for the assembled report, headline digests and compact
  notices, and swarm-state tests for `timed_out`/`assembled` propagation.
- Added tests for the 30-minute/24-round research bounds, the larger finish
  reserve on long runs, and prerequisite output reaching a dependent member in a
  parallel plan.
- Added tests for conversation-workspace file previews (text, image, truncation,
  traversal and binary refusal), the new chat-file route, and workspace files
  becoming preview artifacts.
- Added tests that every skill reaches the coding prompt and swarm briefing, and
  that the manual skill Use button is gone.
