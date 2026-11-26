# Agents in Mariana Studio: AML and AMP

> **Whitepaper:** [AML and AMP](whitepapers/aml-amp.md) covers the design goals,
> document and execution models, security boundary, and tradeoffs in full. This
> page is the practical guide.

Mariana Studio's **Agents** workspace is a visual and programmatic automation suite.
It has two public specifications:

- [Agent Management Language (AML) v1.9](../mariana/agents/specs/AML.md) — the
  portable agent/flow document.
- [Agent Management Protocol (AMP) v1.10](../mariana/agents/specs/AMP.md) — the
  HTTP/JSON protocol for creating, validating, running, testing, debugging and
  scheduling AML agents.

Agent and swarm nodes accept canonical JSON tool calls, DeepSeek DSML markup and
provider-native function calls. AMP discovery publishes the accepted formats
under `tool_calling`; every format shares the same allow-list, sandbox and event
stream.

Public web retrieval is available to Chat and the IDE: `http.fetch` reads a
specific page and `news.headlines` reads an RSS/Atom feed (Google News by
default, with Bing News as the automatic alternate) as dated headlines. Shell
commands, a headless browser (`browser.fetch`) and private-network HTTP are a
server-wide opt-in: enable **OS access** in Models → Runtimes, or start Studio
with `MARIANA_AGENT_ALLOW_OS=1` / `MARIANA_AGENT_ALLOW_SHELL=1`. A node input or
an imported document cannot widen that permission.

For the coding workspace and its dedicated `/api/ide` protocol, see
[IDE: agentic coding in Studio](ide.md).

Both are also served live by the Studio server and are readable in the designer's
**Spec** panel:

```text
GET http://127.0.0.1:8765/api/agents/spec/aml
GET http://127.0.0.1:8765/api/agents/spec/amp
GET http://127.0.0.1:8765/api/agents/protocol
```

## Designer workspace

The designer has four panes: the node palette on the left, the flow canvas in
the middle, the inspector on the right and the details tabs (problems, run
events, schedule and specs) below the canvas. Drag a divider to resize a pane,
double-click or press Enter on it to fold the pane, and use the panel menu to
restore it. `Ctrl+B` toggles the palette, `Ctrl+Shift+B` toggles the inspector
and `Ctrl+J` toggles the details panel. Sizes and folded state are stored in the
browser; the panel menu can reset both or focus the canvas by folding all three
side panes. The details panel starts folded and opens automatically when a run,
a pause or a validation problem needs attention.

A **Swarm** node uses the same inspector: pick the member model and goal, choose
a strategy and feedback rounds, and optionally list member tasks, tools and
write scopes. The document-wide **Swarm policy** in the agent settings caps
agents, steps, rounds, time, messages, shared memory, write access and the
run-wide context pool (`max_context_chars`, `max_agent_context_chars`,
`max_context_item_chars`) for every swarm in that document. Members borrow from
that pool when they need more of a long file or a peer result, and long content
remains readable behind `ctx-…` handles after compaction. A request is charged
only for the slice it actually returns, and an offset at or past the end is
reported without touching the pool. Peer digests are ranked against the member's
own task, identical results are sent once, and a result already shown in full (a
review target, or an earlier pipeline stage) is not repeated. A review records
its own feedback and keeps the member's work summary. A member that is still
researching when it reaches its step budget gets one bounded extension, so a long
coding or news task is not cut off mid-search; the run's time budget still caps
the work.

The swarm inspector also exposes **Acceptance criteria**, **Reasoning effort**
and **Self-check unfinished plans**. The leader derives testable criteria when
none are supplied, gives every member a short plan, and may order members with
`depends_on`; independent members run together while dependent waves wait.
Members update their plans as they work, and a member that tries to finish with
open steps gets one chance to complete or justify them. After a review round the
leader reports open gaps, which feed the next revision and the synthesis. The
swarm card shows each plan, the criteria and any gap that remained. When a run
ends without a leader synthesis (the time budget or a leader failure), Chat
replies with a compact report instead of the raw member hand-offs: the outcome,
one headline per agent that delivered, the agents that did not finish, and a
pointer to the card. The card keeps every full hand-off behind its agent row and
lists run notes in a collapsible section. Files a swarm writes are added to
Chat's preview panel; when automatic previews are on, the last previewable file
opens as soon as the run finishes, so a built page, design or image is visible
without selecting it by hand.

### Latency and cache reuse

A swarm reuses work instead of repeating it. Known read-only tools (file
list/read/search/tree/inspect, diagnostics, `git.inspect`, CSV/skill reads and
the web/browser fetchers) keep a bounded, in-memory result cache for the run.
Concurrent identical calls are single-flight: one member executes, the others
wait and reuse the result. Independent reads in one batched tool call run in
parallel up to eight at a time; writes stay ordered and serialised. A successful
write, edit, delete or shell call clears the cache and invalidates in-flight
read results, so a later read never sees a stale file. Failures are not cached;
network entries expire after 30 seconds and local entries after two minutes.
The run result and snapshot report hits, saved time, stored bytes and the
prompt-cache hit rate; Chat swarm cards and the IDE's bounded swarm observation
surface the hits and hit rate while the run is live. The cache is never written
to disk.

The leader has a free hand to plan a large team. The safety ceiling is 64
agents (default 16 in AML, 8 in Chat and 16 in the IDE); the leader is told to
size the team to the work, give every independent workstream its own member in
one wave, and never add filler. Model calls
and tool executions are metered by separate semaphores (defaults 16 and 32
concurrent, overridable with `MARIANA_SWARM_MODEL_CONCURRENCY` and
`MARIANA_SWARM_TOOL_CONCURRENCY`) so a large plan queues safely instead of
firing every provider and disk call at once. A single read pool serves every
member, so a large team cannot create hundreds of per-turn threads. Planning
output grows with the team size, and reports from more than 12 agents are
condensed in parallel groups before the final synthesis, so every member stays
represented without exceeding the leader's prompt.

Member prompts separate immutable instructions, tool descriptions and briefing
from mutable state. Peer digests, the blackboard, the member plan, unread
messages and feedback live in the user turn; the system prefix stays
byte-identical across tool turns and phases. OpenAI receives a stable
`prompt_cache_key`, llama.cpp receives `cache_prompt`, other compatible
providers use their automatic prefix cache, Anthropic receives `cache_control`
on the system and tool blocks, and cached-token counts flow back into the run
usage. A provider that rejects a cache parameter is retried once without it. If
a caller supplies complete member tasks and explicit tool grants,
the leader planning call is skipped; a feedback round where every review scores
4+ with no open issue also stops without another leader decision. When a
reasoning effort is configured, routine tool turns run at low effort while the
first turn and the final wrap-up keep the configured level. Set
`adaptive_effort: false` on a swarm node to send the configured level on every
turn instead.

These are algorithmic reductions in model and tool work. They are not a
published wall-clock comparison against a separate coding agent; use a matched
corpus, model, tool set and time budget before claiming an external speed record.

Tool calls are reliability-checked in `agent`, `swarm` and the IDE coding loop.
A read-only failure that looks temporary (timeout, connection reset, rate limit,
5xx) is retried once; writes are never retried. Deterministic failures come
back with a short recovery hint, an unknown tool id gets a closest-match
suggestion, and a third identical call is called out explicitly. These rules are
published under `swarm.reliability` in AMP discovery.

### Real-project reads and the hot cache

Below the run cache, a process-wide hot cache keeps decoded workspace files in
RAM, keyed by absolute path and `mtime_ns`/size so a changed file is never served
stale. Its budget is a fifth of available memory (64 MiB–2 GiB) and can be set
with `MARIANA_HOT_CACHE_MB`. Writes and shell commands that may change files
invalidate entries; large and binary files are never stored.

`file.read` now lists a directory instead of failing, follows symlinks that stay
inside the sandbox, reports binary files, streams files above 8 MB by line range,
and suggests close paths and nearby entries when a path is missing. `file.list`
takes a folder, nested pattern and recursion, and `file.tree` reports folders
too, with a depth limit and up to 20,000 entries. `file.search` uses `git grep`
in a Git work tree and falls back to the Python walker elsewhere; it scans up to
50,000 files and treats an invalid regular expression as a literal search.

Swarm tool arguments are normalised before dispatch (`file_path`, `filePath`,
`startLine`, `from`/`to`, `q`, `glob`, `cmd`, numeric strings), so a model's
spelling habit is not a tool failure and equivalent calls share the cache.
Synthesis never accepts tool-call markup as the final answer, and a member whose
reasoning consumes the visible output budget gets one compact evidence-only
repair prompt. Short runs reserve 15% of their budget for synthesis, and member
prompts keep the shared instruction/briefing prefix first so the first member
warms the provider cache for the rest.

On the 3.5k-file Shotcut checkout with the saved DeepSeek endpoint, a read-only
bug audit finished in 156 s with 46 tool calls and no failures, a targeted
three-agent audit in 56 s with 27 calls and no failures, and a four-agent
ownership audit in 93 s with 16 calls and no failures. A documentation swarm
edited only `README.md` in a disposable copy. These are observed local runs;
they are not a matched comparison against another coding agent.

`python scripts/swarm_bench.py --workspace /path/to/project --goal "..."` runs
the same path against any folder, using a saved Studio endpoint (default
`endpoint:deepseek`); add `--write` for edits and `--json` for a machine-readable
summary with phase timings and cache counters.

### Context compression (CRAFT)

CRAFT — **C**ontext **R**eduction with **A**nchors, **F**acts and **T**races —
follows the provenance-first direction of the agent-context survey: the failure
mode that matters is not only losing bytes, but losing the path or line that
proves a later claim. It has three layers:

1. **Anchors** — URLs, file paths, `path:line` references, commands, quoted
   symbols and error lines are extracted verbatim before anything is shortened.
2. **Facts** — units are ranked by goal overlap, error/decision markers, recency
   and anchor density, then deduplicated; head and tail units are reserved so a
   conclusion is never crowded out by filler.
3. **Traces** — the original text stays behind a `ctx-…` handle; the compacted
   prompt says how to request it back. Nothing is silently destroyed.

The compressor spends no model call, so it runs on every observation inside a
large team. Tool results are compressed structure-aware: search matches keep
`path:line` and match text, reads keep range/error metadata plus a bounded body,
trees keep counts and representative paths. The member observation path is
newest-first, so a long older result can no longer push the latest evidence out
of the window.

Compacted turns become an **episode card** naming the tools used, anchors, errors
and decisions, instead of only a list of handles. A run-wide evidence ledger
collects anchors seen by any member and places a compact line in every member
prompt, so the second agent can reuse the first agent's provenance instead of
re-reading the file. Handles also carry a 260-character summary and anchor list.

Default budgets are larger: 24,000 prompt characters, 8,000 observation
characters, a 400,000-character context bank, 96,000 characters per agent and
32,000 per retrieval slice. `cache` reports `compacted_raw_chars`,
`compacted_chars`, `compression_ratio`, `context_episodes`, `context_items` and
`evidence_anchors`.

Reference direction: *Context Compression for LLM Agents: A Survey of Methods,
Failure Modes, and Evaluation* and its curated index
[`YerbaPage/Awesome-Agent-Context-Compression`](https://github.com/YerbaPage/Awesome-Agent-Context-Compression).

An `agent` node keeps its own history bounded. After about 24,000 characters it
replaces the oldest tool rounds with a ledger naming the tools already called and
keeps the goal and newest observations verbatim. An empty reply gets a
correction turn. A reply that looks like an action but cannot be parsed, or a
JSON object with no tool call, plan or final answer (including an empty
`{"final": ""}`), gets up to two correction turns instead of being returned as
the answer. Correction turns never append an empty assistant message, so a
strict provider does not reject the retry.

Run events are ordered. The terminal `run_<status>` event is appended under the
same lock that flips the final status, so a client that sees the run finish also
sees its terminal event, and late worker events are dropped instead of landing
after it.

Streaming `llm` and `agent` nodes publish coalesced `assistant_delta` and
`reasoning_delta` events (AMP 1.10). The Agents **Run & events** panel uses them
for its **Live output** pane, hides token deltas from the event log by default,
and keeps the full JSON behind each row's **Details** disclosure. The same
deltas feed a one-line live preview on the flow canvas node and the swarm card,
with character count, streaming rate and first-delta latency. Chat uses the same
affordances for its live turn row and reasoning panel.

The **New agent** button offers the short starting paths: describe the agent in
plain language, use a working template, import AML, or start blank. The same
choices appear above the agent settings when nothing is selected. Templates,
AI drafts, import/export, schedules, specs and deletion live in the **More**
menu, keeping the toolbar's primary actions to the agent picker, New, Save and
Run.

Canvas interaction stays out of the way until asked for. Double-click empty
space, press `N` or use the add-node button to search node types and place one
at the pointer. Adding a step while another node is selected wires it
automatically: into that node's single outgoing connection, or directly after
it when nothing follows yet. Dropping a palette node onto a connection inserts
it between the two nodes. Drag from a port and drop anywhere on another step to
connect, or drop on empty space (or click the port) to add the next node
already connected. `Ctrl+F` searches labels, types, ids and configuration, and
Enter cycles the matches. Double-click a node title (or press `F2`) to rename
it, click a connection to inspect or insert into it, and right-click any node
or connection for its actions. Space drags the canvas temporarily, `H` toggles a
persistent pan mode, and the minimap is draggable. Nodes follow the pointer
freely while dragging and snap to other nodes' edges and centres with alignment
guides; hold Alt to place one exactly. Multi-selections can be aligned, spread
and fitted from the floating toolbar or the inspector, and arrow keys nudge by
20px (Shift nudges one pixel).

Node cards show a one-line summary of their most important setting, so the flow
stays readable without opening each node. The connection hit area is wider than
the visible line, so thin edges remain easy to select at any zoom. The inspector
shows only essential fields first; sampling, retries, ids, positions and other
advanced options sit behind collapsed disclosures. Template and expression fields have an **Insert
variable** picker for `input`, `last`, `vars.*`, `nodes.*` and loop
`item`/`index`; condition nodes also have a **Build condition** helper for
common comparisons. JSON fields validate inline and offer a Format action.

**Run** reuses the agent's last input. The first run asks once for input, and
the caret menu offers Run with input, a dry-run Test, the Debugger and Stop.
The run panel filters events, follows the running node, focuses a node when its
event is clicked and shows step/resume controls when a debugger pauses. The
status bar reports node and connection counts, selection, run state, zoom, save
state and validation; `?` opens the full shortcut sheet.

## Concept

An agent is a directed flowchart:

- exactly one `start` node and at least one `end` node;
- `llm`, `agent` (an LLM that picks abilities itself), `tool`, `condition`,
  `assert` (a guardrail), `validate` (structured-output schema check),
  `transform`, `delay`, `human` (approval or input), `subagent`, `foreach`,
  `memory`, and `note` nodes;
- edges with an optional `true` / `false` branch;
- explicitly granted abilities (HTTP, sandboxed files with edit/search/tree,
  SQLite, CSV/TSV, Git inspection, memory, math, time, diagnostics, and opt-in
  shell);
- optional top-level `vars` for initial values and `format: "json"` on `llm`
  nodes for parsed structured replies; `fallback_model` on `llm` and `agent`
  nodes keeps a run alive when a provider fails;
- a durable schedule (`interval`, `cron`, or `once`).

Runs, run events, memory, sandbox files, and schedule state live in
`agent-workspace/`. Add that directory to backups if you want to preserve
automations.

**Guardrails.** An `assert` node evaluates the same safe expression language as
a condition and fails the run with an explicit message when the expression is
false. A `validate` node checks the previous output (or a rendered target)
against a documented JSON Schema subset — types, required fields, properties,
arrays, enums, numeric/length bounds, patterns and `anyOf` — and can either
stop the node or continue in warn mode with `valid` and `errors` available to a
condition. Unknown schema keywords are rejected instead of silently ignored.
Both are read-only and work in dry-run mode, so a flow can be tested without
side effects. `llm` and `agent` nodes may also set `fallback_model`; a provider
failure is logged and the call is retried on the fallback model.

While a run is active, `GET /api/agents/runs/{run_id}/events?since=<seq>` streams
complete snapshots as server-sent events. The Agents view uses it; a script can
still poll `GET /api/agents/runs/{run_id}?since=<seq>` and get the same state.

## Quick programmatic loop

```bash
BASE=http://127.0.0.1:8765

# 1. Ask the server what it supports
curl -s $BASE/api/agents/protocol | jq '{version, node_types, abilities}'

# 2. Validate and save an AML document
curl -s -X POST $BASE/api/agents/validate \
  -H 'Content-Type: application/json' --data-binary @agent.json | jq .
curl -s -X POST $BASE/api/agents \
  -H 'Content-Type: application/json' --data-binary @agent.json | jq .

# 3. Test the logic without side effects
curl -s -X POST $BASE/api/agents/my-agent/runs \
  -H 'Content-Type: application/json' \
  -d '{"input":{"message":"hello"},"dry_run":true}' | jq .id

# 4. Poll the run
curl -s "$BASE/api/agents/runs/<run-id>?since=0" | jq '{status, steps, output}'

# 4b. Or follow complete snapshots as server-sent events
curl -N "$BASE/api/agents/runs/<run-id>/events?since=0"

# 5. Debug with a breakpoint
curl -s -X POST $BASE/api/agents/my-agent/runs \
  -H 'Content-Type: application/json' \
  -d '{"input":{},"debug":true,"breakpoints":["llm-1"]}' | jq .id
curl -s -X POST "$BASE/api/agents/runs/<run-id>/step" \
  -H 'Content-Type: application/json' -d '{}' | jq .
curl -s -X POST "$BASE/api/agents/runs/<run-id>/resume" \
  -H 'Content-Type: application/json' -d '{}' | jq .

# 6. Answer a human node that is waiting
curl -s "$BASE/api/agents/runs/<run-id>" | jq .awaiting_input
curl -s -X POST "$BASE/api/agents/runs/<run-id>/input" \
  -H 'Content-Type: application/json' -d '{"value": true}' | jq .status

# 7. Let a model draft an agent (returned for review, not saved)
curl -s -X POST $BASE/api/agents/generate \
  -H 'Content-Type: application/json' \
  -d '{"description":"Fetch a URL and summarise it in three bullets"}' | jq .agent
```

Every request that changes state needs `Content-Type: application/json`, even
with an empty `{}` body — the server uses it to refuse cross-site requests.

## Security

- Keep Studio bound to `127.0.0.1`.
- File abilities are sandboxed to `agent-workspace/files`.
- `sqlite.query` is read-only unless the ability config sets `write: true`.
  `ATTACH` is denied and every query has a deadline.
- `git.inspect` is limited to fixed read-only status, diff and log commands.
- Private HTTP targets are blocked unless the specific ability sets
  `allow_private: true`.
- `shell.exec` is disabled unless the server starts with
  `MARIANA_AGENT_ALLOW_SHELL=1`.
- Imported AML is executable. Review abilities and URLs before running.
