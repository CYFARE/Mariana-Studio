# AML and AMP: a portable language and protocol for local agent automation

**Whitepaper · Mariana v2.3.2 · 2026-11-20**

| | |
| --- | --- |
| Status | Implemented; AML 1.9 and AMP 1.10 |
| Canonical specifications | [`mariana/agents/specs/AML.md`](../../mariana/agents/specs/AML.md), [`mariana/agents/specs/AMP.md`](../../mariana/agents/specs/AMP.md) |
| Guide | [agentic.md](../agentic.md) |
| Live specifications | `GET /api/agents/spec/aml`, `GET /api/agents/spec/amp` |

## Abstract

Mariana Studio's Agents workspace has two public specifications:

- the **Agent Management Language (AML)**, a portable JSON document that
  describes an agent as a directed flowchart with abilities, model settings,
  schedules, and run controls;
- the **Agent Management Protocol (AMP)**, an HTTP/JSON protocol for creating,
  validating, running, testing, debugging, and scheduling AML agents.

The pair is designed for a local-first automation suite that a person or another
language model can author, validate, run, and inspect entirely through HTTP. AML
is the document; AMP is the runtime contract. The design emphasizes explicit
graphs, least-privilege abilities, deterministic validation, complete run
snapshots, server-sent-event streaming, and a security model that assumes the
server is local and the imported document is executable.

This paper describes the design, the document and execution models, the
security boundary, and the tradeoffs. The canonical specifications remain the
source of truth for field-level behavior; this paper explains why the design
looks the way it does.

## 1. Introduction

Agent tooling tends to split into two unsatisfying shapes. A visual builder
produces a proprietary project that only its UI can run. A code framework
produces a program that is powerful but difficult to inspect, validate, or
author from another model. AML and AMP aim at a middle point:

- the agent is a **portable document**, not a Python program;
- the document is a **directed graph**, not an opaque prompt chain;
- every capability is an **explicit grant**, not an ambient process permission;
- every run emits a **complete snapshot** and an ordered event stream;
- the server publishes its own **live schema**, so clients do not hardcode node
  types, ability parameters, or protocol details.

The result is small enough for an LLM to author from a prompt and structured
enough for a UI to render, validate, debug, and schedule without a browser.

## 2. Design goals

| Goal | Consequence |
| --- | --- |
| Portable | Canonical JSON with a media type; YAML accepted on import when PyYAML is installed. |
| Inspectable | A flowchart with typed nodes, named edges, and a normalized document the server can validate. |
| LLM-authorable | A live discovery document; deterministic validation with node and field locations; a dry-run mode. |
| Local-first | Loopback binding by default; no cloud control plane; run state on disk. |
| Least privilege | Abilities are granted on the document and referenced by nodes; the runtime cannot widen a grant. |
| Observable | Pollable run snapshots with `since` cursors and a server-sent-event stream of the same snapshots. |
| Versioned | AML and AMP are additive public specs with changelogs; the server serves the current specs. |
| Recoverable | Debug breakpoints, single-step execution, human-input pauses, cancellation, and persisted run history. |

A non-goal is a distributed workflow engine. AML describes one local agent or a
bounded swarm; it does not attempt to be a cluster scheduler, a message broker,
or a multi-tenant service.

## 3. AML: the document model

### 3.1 Shape

An AML document is a JSON object. The minimal shape is a version, a kind, a
name, nodes, and edges:

```json
{
  "aml": "1.9",
  "kind": "agent",
  "id": "daily-digest",
  "name": "Daily digest",
  "model": "trained-model/mariana.pt",
  "nodes": [
    {"id": "start", "type": "start", "label": "Start", "x": 60, "y": 160},
    {"id": "summary", "type": "llm", "label": "Summarise", "x": 320, "y": 140,
     "data": {"prompt": "Summarise: {{ input.message }}", "max_tokens": 500}},
    {"id": "end", "type": "end", "label": "Return", "x": 600, "y": 160,
     "data": {"output": "{{ nodes.summary.text }}"}}
  ],
  "edges": [
    {"from": "start", "to": "summary"},
    {"from": "summary", "to": "end"}
  ]
}
```

The media type is `application/vnd.mariana.aml+json`. Documents are normalized
to the current version; unknown versions validate with a warning.

Top-level fields include:

| Field | Purpose |
| --- | --- |
| `aml`, `kind`, `id`, `name`, `description`, `version` | Identity and versioning. |
| `model`, `model_settings` | Default model and sampling/device defaults. |
| `vars` | Initial flow variables, available as `vars.<name>`. |
| `settings` | `max_steps`, `timeout`, `on_error`, `max_output_chars`. |
| `swarm` | Defaults and caps for every swarm node in the document. |
| `abilities` | Explicitly granted tools. |
| `nodes`, `edges` | The flow graph. |
| `trigger`, `schedule` | Manual, interval, cron, or once scheduling. |

### 3.2 Nodes

Every node has an `id`, a `type`, a human `label`, designer coordinates `x`/`y`,
and type-specific `data`. Coordinates are ignored by the runtime. Unknown
top-level node keys are ignored.

The node set is intentionally small:

| Category | Types |
| --- | --- |
| Trigger and output | `start`, `end` |
| Intelligence | `llm`, `agent`, `swarm` |
| Action | `tool`, `subagent`, `memory` |
| Flow and logic | `condition`, `transform`, `delay`, `human`, `foreach` |
| Annotation | `note` |

- An `llm` node calls a model once.
- An `agent` node gives a model the granted abilities as tools and loops until
  it produces a final answer or reaches `max_iterations` (1–40).
- A `swarm` node is a leader-led team: the leader plans, members work, exchange
  messages, share a run-scoped blackboard, review each other, and the leader
  synthesizes.
- A `tool` node executes one granted ability.
- A `condition` node evaluates the safe expression language.
- A `transform` node renders or assigns variables.
- A `human` node pauses for approval, text, or a choice.
- A `foreach` node runs another agent once per list item, in order.
- A `memory` node reads or writes persistent per-agent key/value memory.

Any node may include `retry` and `retry_delay` in its `data`. Retries run the
same node again before the error policy applies.

### 3.3 Edges and control flow

```json
{"from": "source-id", "to": "target-id", "when": "always"}
```

`when` is `always`, `true`, or `false`. Ordinary nodes follow their first
outgoing edge. A `condition` node chooses the matching branch. Cycles are
allowed and bounded by `settings.max_steps`. Exactly one edge is followed per
visited node; parallel fan-out is not part of AML. Use a sequence, a
`subagent`, or a `foreach` node instead.

This is a deliberate restriction. A graph with implicit parallel fan-out is
harder to validate, harder to debug, and harder to make deterministic. The
swarm node provides bounded parallelism where it is explicitly requested.

### 3.4 Templates and the safe expression language

Template placeholders use `{{ ... }}`. If the whole string is one placeholder,
the native JSON value is returned; mixed text interpolates stringified values.

The expression subset supports constants, names, lists, tuples, dicts,
subscripts and slices, unary `not`/`-`/`+`, arithmetic, comparisons,
`and`/`or`, and conditional expressions. There is no `eval`, no arbitrary
attribute access, no imports, and no I/O. Helpers cover string transforms,
regular-expression search, dotted lookup, time, and basic arithmetic.

The runtime exposes `input`, `vars`, `last`, `nodes`, `agent`, `run`, `now`,
`true`, `false`, and `null`. The expression language is one of the main reasons
a document can be validated before execution: a condition is data, not a
Python callable.

### 3.5 Abilities and grants

An ability is granted on the document and referenced by a node:

```json
{"id": "fetch", "type": "http.fetch", "label": "Fetch", "enabled": true,
 "config": {"method": "GET", "timeout": 15, "allow_private": false}}
```

A node cannot use an ability that is not listed. The ability's `config` is the
security envelope; a node input or a model decision cannot widen it. The
implemented set includes:

| Area | Abilities |
| --- | --- |
| Web | `http.fetch`, `news.headlines`, `browser.fetch` |
| Files | `file.read`, `file.write`, `file.list`, `file.edit`, `file.search`, `file.tree`, `file.delete` |
| Code and data | `code.diagnostics`, `sqlite.query`, `csv.read`, `git.inspect` |
| Utility | `memory`, `math.eval`, `time.now` |
| Opt-in OS | `shell.exec` |

File abilities are sandboxed to `agent-workspace/files`. HTTP blocks
localhost/private/link-local/reserved addresses by default. `sqlite.query` is
read-only unless the ability config sets `write: true`, and `ATTACH` is denied.
`git.inspect` runs fixed read-only status, diff, and log commands. `shell.exec`
requires server-wide OS access; a document cannot grant it to itself.

### 3.6 Settings and the swarm policy

The document-level `settings` object bounds a run:

| Key | Default | Meaning |
| --- | ---: | --- |
| `max_steps` | 100 | Node executions allowed per run. |
| `timeout` | 120 | Wall-clock seconds, excluding debug pauses. |
| `on_error` | `stop` | Stop, or continue and follow the first outgoing edge. |
| `max_output_chars` | 60000 | Run-history output truncation limit. |

The top-level `swarm` object is both the default and the security envelope for
every swarm node. A node may narrow it, never widen it. It caps member count,
model rounds, feedback rounds, active seconds, messages, message size,
blackboard entries and value size, the run-wide context pool, and write/IPC/
shared-memory permissions.

### 3.7 Schedules

Schedules are durable and persist across restarts. The supported forms are:

| Type | Fields | Semantics |
| --- | --- | --- |
| `manual` | none | No automatic runs; must be disabled. |
| `interval` | `interval_minutes` | First run after one interval, then repeating. |
| `cron` | `cron` | Five-field cron with lists, ranges, steps, and names. |
| `once` | `once_at` | One run at a future local time. |

Scheduled runs use `trigger: "schedule"` and pass `schedule.input` as the run
input. Overlapping runs for the same agent are skipped. Schedules run while the
Studio server is up, even if the browser is closed.

### 3.8 Validation

Validation is deterministic and returns structured issues:

```json
{
  "ok": false,
  "errors": ["<actionable validation message>"],
  "warnings": ["Node 'note-1' is unreachable from start"],
  "issues": [
    {"level": "error", "message": "<actionable validation message>",
     "node": "summary", "edge": null, "field": "prompt"},
    {"level": "warning", "message": "Node 'note-1' is unreachable from start",
     "node": "note-1", "edge": null, "field": null}
  ]
}
```

Warnings never block saving or running. Each issue names its node, edge, or
field when applicable; the exact message wording is implementation-defined,
but the structure is the contract. An LLM authoring loop validates, fixes the
located errors, validates again, then saves.

## 4. AMP: the runtime protocol

### 4.1 Discovery

`GET /api/agents/protocol` returns the protocol version, AML version, node
schemas, ability schemas, common fields, field types, swarm schema and policies,
run statuses, event names, polling guidance, and the security model. The same
document is the source of truth for a model that writes an agent; clients should
not hardcode the appendix.

The full specifications are also served for display or download:

```text
GET /api/agents/spec/aml
GET /api/agents/spec/amp
```

### 4.2 Resources

The protocol covers:

- list, read, create, clone, delete, import, validate, and generate agents;
- list templates;
- run an agent, poll a run with `since`, or stream its events;
- answer a waiting human node;
- debug with breakpoints, step, and resume;
- test with `dry_run`;
- read and set a schedule, or trigger it once;
- inspect the model catalog used by agents.

Create or replace an agent by posting the AML document itself. Validation runs
before writing. A successful response returns the normalized agent and any
warnings.

### 4.3 Runs and snapshots

`POST /api/agents/{id}/runs` accepts:

```json
{
  "input": {"message": "The export button is broken"},
  "trigger": "manual",
  "dry_run": false,
  "debug": false,
  "breakpoints": ["classify", "branch"]
}
```

The response is a run snapshot with HTTP 202. Execution continues in a
background thread. Only one active run per agent is allowed.

A run snapshot carries:

- `status`: `queued`, `running`, `paused`, `completed`, `failed`, or
  `cancelled`;
- `current_node`, `waiting_node`, `paused`, and `awaiting_input`;
- per-node status, flow variables, input, output, and error;
- an ordered `events` array with sequence numbers.

`GET /api/agents/runs/{run_id}?since=<seq>` returns only events with
`seq > since`. This is the polling form used by scripts and LLM loops.

`GET /api/agents/runs/{run_id}/events?since=<seq>` is the streaming form. Every
frame is a complete snapshot whose `events` array contains only newer events.
The stream sends an immediate snapshot on connect, then another whenever the
run changes, with keep-alive comments during quiet periods. It ends after the
terminal frame. A finished run that has left the live registry is replayed once
from history. Clients track the last `seq` and reconnect; snapshots always carry
full `nodes`, `vars`, `steps`, and `output` state, so a reconnect loses no
progress.

Event types include `run_started`, `node_started`, `node_input`,
`node_finished`, `node_failed`, `tool_call`, `tool_result`, `input_requested`,
`input_received`, `input_timeout`, `breakpoint`, `resumed`, `log`,
`run_completed`, `run_failed`, `run_cancelled`, and the swarm events described
below.

### 4.4 Debugging, testing, and human input

- **Dry run** mocks every LLM call and skips side effects: HTTP, file writes,
  SQLite writes, shell commands, memory writes, and delays. File reads, CSV and
  SQLite reads, Git inspection, and expression evaluation still run, so logic
  can be tested.
- **Debug runs** pause before a node whose id is in `breakpoints`.
  `POST /step` executes the paused node and pauses before the next one;
  `POST /resume` continues.
- **Human nodes** pause until a person approves, enters text, or chooses an
  option. `POST /api/agents/runs/{run_id}/input` supplies the answer. The
  snapshot exposes `awaiting_input` while the run waits.

### 4.5 The swarm extension

A swarm node is a bounded, leader-driven run. The leader plans the team, its
success criteria, each member's short plan and any dependency edges. Members
whose dependencies are unfinished wait; each ready wave runs concurrently, or
all members run in order under the pipeline strategy. Members exchange
addressed messages, share a run-scoped blackboard, and review each other when
feedback rounds are configured. The leader synthesizes the final answer and
reports criteria it could not verify. Plans are persisted in the snapshot and
result, and a member that finishes with open steps is asked once to complete or
justify them.

The protocol exposes the swarm snapshot at
`GET /api/agents/runs/{run_id}/swarm`, including the leader, members, message
log, blackboard, reviews, plans, success criteria, gaps, changed files, usage,
and the dynamic context pool.
Operator messages can be queued for the leader with
`POST /api/agents/runs/{run_id}/swarm/messages`; the text is plain data, never a
tool call.

The runtime also shares knowledge without waiting for a request: it seeds the
blackboard with the goal, criteria, dependencies and plans, publishes completed
member results, and adds a peer-results digest plus team tasks to every member
prompt. Duplicate plan tasks are dropped with a warning.

Tool execution is defensive in agent, swarm and IDE coding loops. Read-only
calls that fail with a temporary error are retried once; writes are never
retried because a timed-out write may already have landed. Deterministic
failures return an actionable hint, an unknown tool id is matched against the
granted ids, and a repeated identical call receives an explicit anti-loop hint.

### 4.6 Tool calling

Agent and swarm nodes accept three spellings of a tool call:

- canonical JSON actions;
- DeepSeek DSML `<invoke>` markup;
- provider-native function calls, where the backend supports them.

The discovery document publishes the accepted formats under `tool_calling`.
Native definitions are derived from each ability's `params`; provider-safe
aliases translate back to the ability id. Every spelling shares the same
allow-list, write scope, sandbox, and event stream. An endpoint that rejects
native tools falls back to the prompted JSON form.

### 4.7 Errors

Errors use the wire shape

```json
{"error": "<actionable detail>", "code": "<coarse code>"}
```

with codes for `bad_request` (400), `forbidden` (403), `not_found` (404),
`method_not_allowed` (405), `conflict` (409), `too_large` (413),
`server_error` (500), `upstream_error` (502), and `unavailable` (503). The
message says what is wrong and what to do next; the code is for clients.

## 5. Security model

The security model assumes the server is local and the imported document is
executable code. It is defense in depth, not a claim of multi-tenant isolation.

- **Network.** Studio binds to `127.0.0.1` by default. Requests with a foreign
  `Host`, a foreign `Origin`, or a state-changing body without a JSON content
  type are refused. The server sends no CORS headers.
- **Authentication.** There is no authentication. Use a reverse proxy with auth
  if remote access is unavoidable, and do not expose the server to untrusted
  networks.
- **Files.** File abilities resolve paths inside `agent-workspace/files`; path
  traversal and symlink escapes are rejected.
- **HTTP.** Public targets only by default. Redirect targets and connected peers
  are validated so a public URL cannot redirect to or resolve to a private
  address. Cloud metadata addresses are always blocked. `allow_private` is
  taken from the ability config.
- **SQL and Git.** SQLite is read-only unless the ability config enables writes;
  `ATTACH` is denied and queries have deadlines. Git inspection uses fixed
  read-only arguments with external diffs, text conversion, pager, and fsmonitor
  disabled.
- **OS access.** Shell commands, private-network HTTP, and `browser.fetch` are a
  server-wide opt-in (`allow_os_access`, `MARIANA_AGENT_ALLOW_OS=1`, or
  `MARIANA_AGENT_ALLOW_SHELL=1`). A node input or an imported document cannot
  widen that permission.
- **Secrets.** Endpoint keys are write-only at the API boundary; the response
  exposes `has_api_key`, never the key.
- **Swarms.** IPC is run-scoped. Recipients, message kinds, sizes, quotas, and
  reply links are validated. Messages are data, never tool calls. Write-like
  member calls require both `swarm.allow_write` and the node's `write` flag,
  and write scopes are enforced before the sandbox runs.
- **Dry runs.** Test mode skips side effects, so an author can exercise control
  flow before granting network or filesystem access.

The most important operational rule is simple: AML is executable automation.
Only import agents you trust.

## 6. LLM authoring loop

The protocol is designed so another model can author an agent without a browser:

1. Read `GET /api/agents/spec/aml` and `GET /api/agents/spec/amp`.
2. Read `GET /api/agents/protocol` for the live node and ability catalog.
3. Draft AML that references only the abilities it grants.
4. `POST /api/agents/validate` and fix every `issues` entry with
   `level: "error"`.
5. `POST /api/agents` to save.
6. `POST /api/agents/{id}/runs` with `dry_run: true`.
7. Poll the run and inspect `error` and `events`.
8. If a branch needs inspection, start a debug run with breakpoints, then
   `step` or `resume`.
9. Run once for real and inspect the output. For a swarm, read the swarm
   snapshot for the plan, messages, blackboard, and reviews.
10. Only then enable a schedule.

This loop is the practical argument for deterministic validation and complete
snapshots. A model that can read the schema, validate a document, and inspect a
run can operate the system without guessing at hidden state.

## 7. Status and limitations

- **Local single-user by design.** No authentication, no tenant isolation, no
  distributed scheduler, and no cross-run message bus.
- **Bounded concurrency.** One active run per agent; swarms have explicit member,
  step, round, message, time, and context limits.
- **No arbitrary code nodes.** Computation is the template/expression language,
  the LLM nodes, and the granted abilities. A `shell.exec` ability is the
  explicit escape hatch, and it is server-opt-in.
- **Additive versioning.** AML and AMP changes are meant to be backward
  compatible. Clients should use discovery rather than hardcoding the schema.
- **Schedules require the server.** A schedule runs while Studio is up; stopping
  the server stops schedules.
- **Executable import.** Importing AML is importing a program. Review abilities,
  URLs, and shell commands before running.
- **Provider variance.** Native tool calling depends on the backend. The prompted
  JSON fallback exists precisely because not every endpoint supports native
  function definitions.

## 8. Related work

AML's document shape is closest to a workflow definition language with a
JSON-Schema-like field model. AMP intentionally resembles a small OpenAPI-style
resource protocol with a discovery document, JSON errors, and server-sent
events. The swarm extension borrows the leader/worker pattern from multi-agent
systems but keeps it bounded and run-scoped.

The design is also a response to the ergonomics of agent frameworks that are
code-first. AML does not replace code for arbitrary computation; it makes the
common automation shape reviewable, portable, and machine-authorable while
leaving a clearly marked escape hatch for OS-level work.

## 9. Conclusion

AML and AMP separate the agent document from the agent runtime. AML gives a
human or a model a portable graph with explicit grants and bounded settings.
AMP gives a client discovery, deterministic validation, complete run snapshots,
live streaming, debugging, scheduling, and a security model that treats imported
documents as executable. The canonical specifications and the live discovery
endpoint remain the source of truth; this paper explains the design choices
behind them.

## References

1. Mariana. *Agent Management Language (AML) v1.9.*
   [`mariana/agents/specs/AML.md`](../../mariana/agents/specs/AML.md).
2. Mariana. *Agent Management Protocol (AMP) v1.10.*
   [`mariana/agents/specs/AMP.md`](../../mariana/agents/specs/AMP.md).
3. Mariana. *Agents in Mariana Studio: AML and AMP.*
   [`docs/agentic.md`](../agentic.md).
4. OpenAPI Specification 3.1. <https://spec.openapis.org/oas/v3.1.0>.
5. JSON Schema 2020-12. <https://json-schema.org/specification>.
6. WHATWG. *Server-Sent Events.* <https://html.spec.whatwg.org/multipage/server-sent-events.html>.

## Citation

```bibtex
@techreport{mariana_aml_amp_2026,
  title       = {AML and AMP: A Portable Language and Protocol for Local
                 Agent Automation},
  author      = {{Mariana contributors}},
  institution = {Mariana},
  year        = {2026},
  version     = {2.3.2},
  url         = {https://github.com/CYFARE/Mariana/blob/main/docs/whitepapers/aml-amp.md}
}
```
