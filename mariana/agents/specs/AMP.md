# Agent Management Protocol (AMP) v1.10

AMP is the HTTP/JSON protocol for creating, validating, running, testing,
debugging and scheduling AML agents in Mariana Studio. It is intentionally
small enough for another LLM to use from a prompt and complete enough for a
local automation suite.

- Current version: `1.10` (a superset of `1.9`, `1.8`, `1.7`, `1.6`, `1.5`, `1.4`, `1.3`, `1.2`, `1.1` and `1.0`; see [Changes in 1.10](#12-changes-in-110))
- Transport: HTTP/1.1, JSON bodies, UTF-8
- Base path: `/api/agents`
- Language: [Agent Management Language (AML) v1.9](./AML.md)
- Discovery: `GET /api/agents/protocol`
- Server default: `http://127.0.0.1:8765`

All timestamps are ISO-8601 local time unless marked UTC. Errors are returned
as `{"error": "<actionable message>", "code": "<machine-code>"}` with a 4xx/5xx
status. A request body larger than 1 MB is rejected. Requests that change state
must send `Content-Type: application/json`; requests carrying an `Origin` other
than the server's own are refused (see §11).

---

## 1. Discovery

### `GET /api/agents/protocol`

Returns protocol version, node and ability schemas, operations, run payloads,
event names, polling guidance and the security model. Clients should use this
instead of hardcoding the schema.

```json
{
  "protocol": "AMP",
  "version": "1.10",
  "aml_version": "1.9",
  "tool_calling": {
    "formats": ["json", "dsml", "native"],
    "native_backends": ["openai-compatible", "anthropic", "ollama"],
    "fallback": "prompted JSON",
    "note": "Every format uses the same allow-list, sandbox and tool_call/tool_result events."
  },
  "guardrails": {
    "assert": "A safe expression must be true; message explains the failure.",
    "validate": "A bounded JSON schema subset validates structured output.",
    "model_fallback": "llm and agent nodes accept fallback_model."
  },
  "transport": "HTTP/JSON",
  "base_path": "/api/agents",
  "node_types": {"llm": {
    "label": "LLM", "category": "intelligence", "icon": "✦", "description": "...",
    "fields": [{"key": "prompt", "label": "Prompt", "type": "template", "rows": 6, "help": "..."},
               {"key": "fallback_model", "label": "Fallback model", "type": "model"}],
    "defaults": {"model": "", "system": "", "prompt": "", "format": "text",
                 "temperature": 0.7, "max_tokens": 512, "fallback_model": ""}
  }},
  "common_fields": [{"key": "retry", "label": "Retries on error", "type": "number", "min": 0, "max": 5}],
  "field_types": ["text", "template", "expression", "number", "select", "boolean", "json",
                  "model", "ability", "abilities", "agent"],
  "abilities": {"http.fetch": {"label": "HTTP request", "description": "...", "config": {},
                               "params": [{"key": "url", "label": "URL", "type": "template"}]},
                "news.headlines": {"label": "News headlines", "description": "...", "config": {},
                                   "params": [{"key": "query", "label": "Search or topic", "type": "text"}]},
                "browser.fetch": {"label": "Headless browser", "description": "...", "config": {},
                                  "params": [{"key": "url", "label": "URL", "type": "template"}]}},
  "swarm": {
    "node_type": "swarm",
    "strategies": ["auto", "map_reduce", "peer_review", "debate", "pipeline"],
    "message_kinds": ["message", "request", "response", "review", "feedback", "status", "handoff"],
    "policy": {"max_agents": 16, "max_steps": 8, "max_rounds": 2, "max_seconds": 300,
               "max_messages": 200, "max_context_chars": 400000,
               "max_agent_context_chars": 96000, "max_context_item_chars": 32000,
               "allow_write": false, "allow_ipc": true, "shared_memory": true},
    "leader_protocol": {"plan": "...", "reviews": "...", "continue": "...", "synthesis": "..."},
    "member_protocol": {"tool": "...", "say": "...", "memory": "...", "context": "...",
                        "review": "...", "plan": "...", "final": "..."},
    "events": ["swarm_start", "swarm_plan", "swarm_leader", "swarm_round",
               "swarm_agent_start", "swarm_agent_step", "swarm_agent_delta",
               "swarm_agent_status", "swarm_agent_tool", "swarm_agent_plan",
               "swarm_agent_end",
               "swarm_message", "swarm_memory", "swarm_review", "swarm_context", "swarm_end"]
  },
  "run_statuses": ["queued", "running", "paused", "completed", "failed", "cancelled"],
  "polling": {"run": "GET /api/agents/runs/{run_id}?since=<seq>",
              "stream": "GET /api/agents/runs/{run_id}/events?since=<seq>"}
}
```

`node_types[type].fields` and `abilities[type].params` use one field schema:
`key`, `label`, `type` (one of `field_types`), and optionally `help`, `rows`,
`min` / `max` / `step` (numbers) or `options` (`[[value, label], …]`, selects).
`defaults` is the `data` of a new node. A client can render a complete editor
for any node, including types added later, from this metadata alone.

AMP 1.4 added `sqlite.query`, `csv.read` and `git.inspect` to the same discovery
document, alongside the earlier coding abilities. The **IDE** workspace has its
own discovery endpoint, `GET /api/ide`, which lists IDE workspaces, modes, limits,
tool schemas and the shell opt-in. IDE runs use the same model gateway and
ability sandbox, but stream through `GET /api/ide/runs/{run_id}/events`; see
`docs/ide.md`.

AMP 1.5 also publishes the `swarm` node schema, both sides of the swarm
conversation (`leader_protocol`, `member_protocol`), the event names and the
security rules. The same document is the source of truth for a model that
writes a swarm, so clients should not hardcode this appendix.

AMP 1.6 adds the `tool_calling` object. `agent` and `swarm` nodes accept the
canonical JSON action, DeepSeek DSML `<invoke>` markup, and provider-native
function calls; native definitions are derived from ability `params` and every
format shares one allow-list, sandbox and event stream. An endpoint that
rejects native tools falls back to the prompted JSON form.

AMP 1.7 adds dynamic swarm context management. Members may request more context
(`context` action), a run-wide pool grants it only while space remains, long
content is retained behind `ctx-…` handles, and failed grants return an
actionable used/free report instead of a truncated prompt. The swarm snapshot
gains a `context` object and the event stream gains `swarm_context`.

AMP 1.7 also publishes the `news.headlines` ability. It reads a public RSS/Atom
feed (Google News by default) and returns dated headlines with sources and
links. Public web tools are available to Chat and the IDE; shell access is
controlled by the server-wide `allow_os_access` setting or
`MARIANA_AGENT_ALLOW_OS=1` / `MARIANA_AGENT_ALLOW_SHELL=1`.

### `GET /api/agents/spec/aml` and `GET /api/agents/spec/amp`

Return the full specifications as `text/markdown`. The Studio Spec panel loads
these endpoints and offers Copy / Download.

---

## 2. Agent resources

### `GET /api/agents`

```json
{"agents": [
  {"id": "support-triage", "name": "Support triage", "nodes": 6, "edges": 6,
   "abilities": 1, "schedule": {"enabled": false, "type": "manual"},
   "next_run": null, "last_run": null, "last_error": null,
   "active": false, "updated": "..."}
]}
```

### `GET /api/agents/runs?limit=30`

Run summaries across every agent, newest first. The response is
`{"runs": [...]}` with the same summary fields as `GET /api/agents/{id}/runs`.
Use this for a global activity view or an external controller.

### `POST /api/agents`

Create or replace an agent. The body is the AML document itself. Validation is
performed before writing. On success:

```json
{"agent": {"id": "support-triage", "...": "..."}, "warnings": []}
```

### `POST /api/agents/validate`

Validate without saving. Always returns HTTP 200 with:

```json
{"ok": true, "errors": [], "warnings": ["Node note-1 is unreachable from start"],
 "issues": [{"level": "warning", "message": "Node note-1 is unreachable from start",
             "node": "note-1", "edge": null, "field": null}],
 "agent": {}}
```

`ok` is false when `errors` is non-empty. Warnings never block saving or runs.
`issues` carries the same messages with their location: `node` (a node id),
`edge` (`[from, to]`) and `field` (a `data` key, or `"schedule"`), each `null`
when it does not apply. `errors` and `warnings` are derived from it.

### `GET /api/agents/{id}`

Returns the normalized AML document plus `{"_meta": {"active_run": "run-..."}}`.

### `POST /api/agents/{id}/clone`

Duplicate a saved agent. The body is optional: `{"id": "copy-id", "name": "Copy name"}`.
The new id defaults to `<old-id>-copy`, the new name to `<old-name> copy`, and the
copied schedule is always disabled. Returns `201` with the same shape as
`POST /api/agents`. A missing source is `404`; an id that already exists is `409`.

### `DELETE /api/agents/{id}`

Deletes the agent, its run history and schedule state. Fails if a run is active.

```json
{"deleted": true, "id": "support-triage"}
```

### `POST /api/agents/import`

Body: `{"text": "<JSON or YAML AML>"}`. Parses, validates and saves. Returns the
same shape as `POST /api/agents`.

### `GET /api/agents/templates`

Returns the starter gallery. `GET /api/agents/templates/{id}` returns one AML
document. Creating a template is `POST /api/agents` with that document.

### `POST /api/agents/generate`

Body: `{"description": "<what the agent should do>", "model": "<model id>"}`.
A chat model drafts an AML document from the description; the server validates
it and feeds errors back for up to two repair rounds. Returns
`{"agent", "warnings", "issues", "attempts", "model"}`. **Nothing is saved** —
review the draft, then `POST /api/agents`. `model` may be omitted when exactly
one chat model is available. Responds 400 with an actionable message when the
model cannot produce valid AML.

---

## 3. Running agents

### `POST /api/agents/{id}/runs`

Body:

```json
{
  "input": {"message": "The export button is broken"},
  "trigger": "manual",
  "dry_run": false,
  "debug": false,
  "breakpoints": ["classify", "branch"]
}
```

| Field | Type | Default | Meaning |
|---|---|---|---|
| `input` | any JSON | `{}` | Passed to the start node. |
| `trigger` | string | `manual` | `manual`, `schedule`, or `subagent`. |
| `dry_run` | bool | `false` | Mock LLMs; skip HTTP/file/shell/memory writes and delays. |
| `debug` | bool | `false` | Enable breakpoints and single-step pauses. |
| `breakpoints` | array | `[]` | Node ids that pause the run before executing. |

The response is a run snapshot with HTTP 202. Execution continues in a
background thread. Only one active run per agent is allowed. Field types are
strict: a non-boolean `dry_run` or `debug`, an unknown `trigger`, a non-list
`breakpoints`, or a breakpoint that names no node is rejected with 400. When
Studio stops, active runs are cancelled; runs left active by a crash are marked
failed on the next start.

### `GET /api/agents/runs/{run_id}?since=<seq>`

Poll a run. `since` returns only events with `seq > since`, which is ideal for
live UIs and LLM loops.

```json
{
  "id": "run-18f...-a1b2c3",
  "agent_id": "support-triage",
  "status": "running",
  "current_node": "classify",
  "waiting_node": null,
  "paused": false,
  "awaiting_input": null,
  "steps": 3,
  "nodes": {"start": "ok", "classify": "running", "branch": "pending"},
  "vars": {"topic": "exports"},
  "input": {"message": "..."},
  "output": null,
  "error": null,
  "events": [
    {"seq": 7, "at": "2026-01-01T09:00:00.123", "type": "node_started",
     "node": "classify", "label": "Classify", "node_type": "llm"}
  ]
}
```

Run statuses: `queued`, `running`, `paused`, `completed`, `failed`, `cancelled`.
Node statuses: `pending`, `running`, `ok`, `error`, `paused`.

Event types: `run_started`, `node_started`, `node_input`, `node_finished`,
`node_failed`, `assistant_delta`, `reasoning_delta`, `tool_call`, `tool_result`,
`validation`, `input_requested`, `input_received`, `input_timeout`, `breakpoint`,
`resumed`, `log`, `run_completed`, `run_failed`, `run_cancelled`, and the swarm
events in §3.1.

- `node_input` carries the rendered `input` of a node (an LLM prompt, an ability
  input), truncated.
- `node_finished` carries `output` and `elapsed_ms`.
- `assistant_delta` and `reasoning_delta` are coalesced, best-effort progress
  from a streaming `llm` or `agent` call. They carry `node`, `iteration` and a
  `text` slice; the authoritative answer remains the node result.
- `validation` carries `ok`, `mode`, `error_count` and a bounded `errors` list
  from a `validate` node; assert failures surface through `node_failed`.
- `tool_call` / `tool_result` report each ability an `agent` node used:
  `iteration`, `ability`, `input`, then `ok` and `output`. The event shape does
  not depend on how the model expressed the call: canonical JSON, DSML and
  provider-native function calls produce the same pair.
- Swarm events carry `swarm` (the run-scoped swarm id), `node` (the owning flow
  node), `agent`, `round` and `phase`, so a UI can place every update on the
  right node, member and round.

`vars` holds the flow variables assigned so far, including the document's
initial `vars`. `awaiting_input` is `null`, or
`{"node", "prompt", "kind", "choices", "default"}` while a `human` node waits.

`settings.timeout` is checked between nodes and inside model and tool loops.
Cancellation stops the current model stream where the backend supports it; a
cancelled run ends with `run_cancelled`.

### `GET /api/agents/runs/{run_id}/events?since=<seq>`

Server-sent events for one run. This is the streaming form of the polling
endpoint and is what the Studio Agents view uses. Every frame is a complete run
snapshot whose `events` array contains only events with `seq > since`:

```text
data: {"type": "snapshot", "id": "run-...", "status": "running", "steps": 2, "nodes": {...}, "events": [...]}

data: {"type": "snapshot", "id": "run-...", "status": "completed", "output": {...}, "events": [...]}

```

The stream sends an immediate snapshot on connect, then another whenever the
run changes, with a keep-alive comment during quiet periods. It ends after the
frame whose `status` is `completed`, `failed` or `cancelled`. A finished run that
has already been removed from the live registry is replayed as one final
snapshot from history. Clients should track the last event `seq` and reconnect
with `since=<seq>`; snapshots always carry full `nodes`, `vars`, `steps` and
`output` state, so a reconnect loses no progress.

### `POST /api/agents/runs/{run_id}/input`

Body: `{"value": <answer>}`. Answers the `human` node the run is waiting on and
returns the updated snapshot. `kind: "approve"` takes a boolean (or `yes` /
`no`), `choice` one of the node's `choices`, `text` any string. Responds 409
when the answer does not fit, or when the run is not waiting for input.

### `GET /api/agents/{id}/runs?limit=30`

Recent run summaries, newest first. Run history is retained per agent (60 most
recent runs).

### 3.1 Swarm runs

A `swarm` node is planned and led by an LLM. The run snapshot includes a
`swarm` object while the node runs and afterwards; it contains the leader, the
members, the message log, the shared blackboard, reviews and changed files.
The same object is available on its own:

#### `GET /api/agents/runs/{run_id}/swarm`

```json
{
  "id": "swarm-team-18f...",
  "status": "reviewing",
  "goal": "Find the performance regression",
  "strategy": "peer_review",
  "round": 1,
  "requested_rounds": 1,
  "phase": "review",
  "leader_model": "qwen2.5-coder",
  "model": "qwen2.5-coder",
  "allow_write": false,
  "allow_ipc": true,
  "shared_memory": true,
  "success_criteria": ["The answer names the hot loop", "Measurements are cited"],
  "gaps": [],
  "unmet_criteria": [],
  "agents": [
    {"id": "profiler", "label": "Profiler", "role": "investigator",
     "task": "Profile the training loop", "status": "done", "score": 4.5,
     "version": 2, "depends_on": [], "tool_calls": 3, "tool_retries": 1,
     "plan": [{"text": "Sample the loop", "status": "done"},
              {"text": "Measure the hot path", "status": "done"}],
     "output": "..."}
  ],
  "messages": [
    {"id": "msg-1a2b3c4d5e", "from": "profiler", "to": "leader",
     "kind": "status", "text": "Found a hot loop", "round": 0}
  ],
  "blackboard": [{"key": "finding.hot-loop", "version": 1, "author": "profiler"}],
  "reviews": [{"reviewer": "reviewer", "target": "profiler", "score": 4.5,
                "issues": ["measure again"], "round": 1}],
  "changed_files": [],
  "usage": {"prompt_tokens": 1200, "completion_tokens": 340},
  "context": {"capacity": 400000, "per_agent": 96000, "used": 24000, "free": 376000,
              "agents": {"profiler": 24000}},
  "context_items": 3,
  "context_handles": [{"handle": "ctx-1a2b3c4d5e", "kind": "tool", "label": "file.read result",
                       "chars": 18400, "author": "profiler"}],
  "started": 1760000000.0, "updated": 1760000004.2
}
```

`status` is `planning`, `running`, `reviewing`, `synthesizing`, `completed`,
`failed` or `cancelled`. A run with no swarm node answers 404. The object is
persisted with the run history, so `GET` also works after the run ends.
State-changing events refresh this snapshot immediately; high-frequency progress
events (`swarm_agent_step`, `swarm_agent_tool`, `swarm_agent_status`,
`swarm_message`) refresh it at most every 250 ms. `swarm_agent_plan` is a
state-changing event and refreshes immediately. The event stream itself is
never throttled.

#### `POST /api/agents/runs/{run_id}/swarm/messages`

Body: `{"text": "Please also check the data loader"}`. Queues an operator
message for the **leader** of a live swarm. The leader reads queued messages in
its next planning, review-decision or synthesis prompt. The text is limited to
2000 characters and is plain data, never a tool call. Only `text` is accepted;
there is no recipient field, so an operator cannot address members directly.
Answers 409 when the run is not active or has no live swarm, and 400 when the
text is empty or too long.

```json
{"queued": true, "to": "leader"}
```

#### Events

| Event | Payload | Meaning |
|---|---|---|
| `swarm_start` | `swarm`, `leader`, `strategy`, `requested_rounds`, `tools`, `limits` | The leader started planning. |
| `swarm_plan` | `agents`, `strategy`, `rounds`, `success_criteria`, `notes` | The final team, plan, criteria and strategy. |
| `swarm_leader` | `phase`, `text` | Leader status or plan notes. |
| `swarm_round` | `round`, `phase` (`work`/`review`/`revise`/`stop`), `reviews` | A phase of the run started or ended. |
| `swarm_agent_start` | `agent`, `label`, `role`, `phase`, `version` | A member started a phase. |
| `swarm_agent_step` | `agent`, `iteration`, `phase` | A model round began. |
| `swarm_agent_delta` | `agent`, `iteration`, `text` | Streamed member content. |
| `swarm_agent_status` | `agent`, `message` | Reasoning or a status note. |
| `swarm_agent_tool` | `agent`, `tool`, `input`, `ok`, `attempts`, `output` | A member's tool call and its result; `attempts` includes an automatic retry. |
| `swarm_agent_plan` | `agent`, `plan`, `version`, `done`, `total` | A member replaced its plan. |
| `swarm_message` | `from`, `to`, `message_kind`, `text`, `message_id`, `reply_to`, `round` | Agent-to-agent IPC. |
| `swarm_memory` | `agent`, `action`, `key`, `version`, `automatic` | A shared-blackboard change; `automatic` marks a runtime-published member result. |
| `swarm_review` | `reviewer`, `target`, `score`, `issues`, `round` | A peer review was recorded. |
| `swarm_context` | `agent`, `action` (`get`/`grant`/`denied`/`find`/`release`), `label`, `reference`, `granted`, `offset`, `total`, `context` | A context grant, denial, search or release. `context` is the current pool snapshot. |
| `swarm_agent_end` | `agent`, `ok`, `status`, `output`, `summary`, `score`, `version`, `steps`, `usage` | A member finished or failed a phase. |
| `swarm_end` | `ok`, `status`, `output`, `summary`, `agents`, `reviews` (count), `changed_files`, `timed_out` | The swarm finished and the leader returned its synthesis. `timed_out` is true when the work budget ended before every phase completed. |

Deltas are best-effort progress; the authoritative result is the `swarm_end`
payload and the run snapshot. A client tracks the last event `seq` and reconnects
as it does for any other run.

#### Leader and member protocol

The language specification defines the JSON shapes. In short:

- The leader plans `{strategy, rounds, agents, notes}`, assigns reviews
  `{reviews: [{reviewer, target, focus}]}`, decides whether another feedback
  round helps `{continue, reason}`, and synthesises
  `{final, summary, warnings, confidence}`. The leader may choose fewer than the
  node's `rounds`, never more.
- A member calls tools (`tool`, `calls`), sends messages (`say`), reads or writes
  the blackboard (`memory`), requests or releases shared context (`context`),
  updates a plan, produces a review during a review phase (`review`), and
  finishes with `final`.
- A reply that is plain text is a final answer. A malformed protocol turn is
  reported back to the model with the exact expected shape, so a small model can
  recover without failing the run.

The built-in fallback keeps a swarm useful with models that ignore JSON: the
leader's plan becomes one researcher, a member's plain text becomes its result,
and the synthesis concatenates the member results if the leader returns no
structured answer.

Reliability is part of the contract: an empty model reply is retried once before
it fails a step; a failed review is reported in `warnings` without discarding the
member's completed work; and `max_seconds` is enforced even when a backend
ignores cancellation. The last 10% of the budget (at least 5 s, at most 90 s) is
reserved for wrap-up and synthesis: tool rounds stop at that work deadline, a
member answers from its collected observations, later dependency waves and peer
reviews are skipped with a warning, and the leader still synthesises. If the
reserve is also gone, the result is assembled from the members' work and the run
completes with `timed_out: true`; it fails only when no member produced output.
Unfinished members are finalised as failed (or kept as done when they already
produced output) before the result and `swarm_end` are emitted, and late worker
events after the run ends are dropped. A `SwarmCoordinator` runs once; start a
new one for another run.

The runtime also shares knowledge without waiting for a member to ask: it seeds
the blackboard with `swarm.goal` and `swarm.plan` before work starts, publishes
each completed result as `result.<member-id>` (emitted as a `swarm_memory` event
with `automatic: true`), and adds a peer-results digest plus the team's tasks to
every member and reviewer prompt. Duplicate plan tasks are dropped with a
warning. These seeded and automatic writes obey the same key, JSON and
`max_memory_entries` limits as member writes; when the board is full they are
skipped and the run continues.

#### Swarm security

- IPC exists only inside one `SwarmCoordinator`. There is no network transport,
  no global registry and no cross-run delivery. A sender must be a member; a
  recipient must be `leader`, `all` or a known member; self-messages are refused.
- `leader`, `all`, `operator`, `system` and `user` are reserved channel roles. A
  planned member id that would shadow one is renamed with a warning instead of
  sharing the leader's mailbox.
- Message kind is allow-listed, text is stripped of control characters, size is
  capped by `max_message_chars`, per-sender quotas and a per-swarm total apply,
  and `reply_to` must reference a message the sender actually received.
- Messages are data. They are never executed as tool calls and cannot widen an
  ability, a write scope or a network policy.
- The shared blackboard is run-scoped, keyed by a restricted name, JSON-only and
  capped by entry count and serialized size. It is not persisted by the runtime.
- Context grants and the context bank are run-scoped. Handles are opaque, reads
  are size-capped by the pool and never become tool calls, and the bank is
  discarded with the coordinator.
- `news.headlines` and `http.fetch` reach only public addresses by default.
  Cloud metadata addresses are always blocked. The server owner may enable OS
  access to allow shell commands and private-network targets; a node input or an
  imported AML document cannot widen that permission.
- Members can use only the tools in their plan. When the node and
  `swarm.allow_write` both allow writes, write-like calls are serialized across
  members and a member's `writes` prefixes are enforced before the sandbox runs.
  A member without a write scope is read-only unless it explicitly sets
  `read_only: false`; `writes: ["*"]` grants unrestricted sandbox writes, and a
  scoped member cannot use `shell.exec`.
- The time budget is checked before every model and tool call; cancellation stops
  new work and the run reports `cancelled`.
- Operator messages are accepted only for a live swarm, only for the leader and
  only up to 2000 characters.

---

## 4. Debugging

Debug runs pause **before** a node executes when its id is in `breakpoints` or
after a single-step request.

### `POST /api/agents/runs/{run_id}/step`

Executes the paused node, then pauses before the next node. Returns the updated
snapshot. If the run is not paused, sets a pending single-step mode.

### `POST /api/agents/runs/{run_id}/resume`

Clears pause/step mode and continues until the next breakpoint or completion.

### `POST /api/agents/runs/{run_id}/cancel`

Requests cancellation. The run becomes `cancelled` at the next safe boundary.
Cancellation is cooperative for tool calls (they complete or time out first).

### `POST /api/agents/runs/{run_id}/breakpoints`

Body: `{"breakpoints": ["node-id", …]}`. Replaces the breakpoints of a live
debug run, so they can be toggled while it is paused. Returns the snapshot.

The run snapshot always includes `paused`, `waiting_node`, `breakpoints`,
`nodes`, `vars` and `current_node`, so a debugger UI can highlight the flow and
show its state exactly. `paused` is true only for debugger pauses; a run waiting
on a `human` node has `status: "paused"`, `paused: false` and `awaiting_input`
set, and `step` / `resume` do not release it.

---

## 5. Testing

Use `dry_run: true`:

- `llm`, `agent` and `swarm` nodes return `"[test mode] <prompt preview>"` with
  `dry_run: true`; no model or tool is called. A swarm in test mode reports an
  empty member list and does not plan or spend model calls.
- `human` nodes do not wait: they use their `default`, else approve / the first
  choice / an empty string.
- `http.fetch`, `file.write`, `sqlite.query` with `write: true`, `shell.exec`,
  memory writes and `delay` are skipped.
- `file.read`, `file.list`, `csv.read`, read-only `sqlite.query`, `git.inspect`,
  `math.eval`, `time.now` and expressions still run.
- The run is persisted with `dry_run: true` for inspection.

This makes it safe to test branches, retries, templates and subagent wiring on a
new corpus or a new machine.

---

## 6. Scheduling

### `GET /api/agents/{id}/schedule`

```json
{
  "schedule": {"enabled": true, "type": "cron", "cron": "0 9 * * 1-5", "input": {}},
  "next_run": "2026-01-02T09:00:00",
  "last_run": "2026-01-01T09:00:00",
  "last_error": null,
  "server_time": "2026-01-01T10:00:00"
}
```

### `POST /api/agents/{id}/schedule`

Body: `{"enabled": true}`. Enable or disable the schedule persisted in the AML
document. The schedule *type* and expression are changed by updating the agent
(`POST /api/agents`), then toggling here. The scheduler runs every 5 seconds,
survives page reloads, and skips overlapping runs for the same agent.

### `POST /api/agents/{id}/trigger`

Starts one manual run using `schedule.input` (or an override `{"input": ...}`)
without changing the schedule. Useful for cron “Run now”.

---

## 7. Models used by agents

LLM nodes call the same model gateway as Chat:

- Mariana DRV-LM and NCM checkpoints (`.pt`)
- Hugging Face directories (when `transformers` is installed)
- GGUF files, including models installed by Ollama, LM Studio and the Hugging
  Face cache (through a `llama-server` binary or `llama-cpp-python`)
- ONNX directories (when `optimum` and `onnxruntime` are installed)
- Ollama models and other servers running on this machine, found automatically
- OpenAI-compatible, Ollama and Anthropic endpoints saved in Models

`GET /api/models` returns the catalog:

```json
{"models": [
  {"id": "trained-model/mariana.pt", "name": "mariana.pt", "kind": "mariana-drv",
   "kind_label": "Mariana DRV-LM", "available": true, "reason": "",
   "capabilities": {"chat": true, "finetune": true},
   "detail": {"parameters": 4200000, "ctx_len": 256}}
]}
```

Set `model` in agent settings or override it per `llm` / `agent` node. Remote
endpoints are managed in the Models workspace; API keys may be stored locally in
`agent-workspace/endpoints.json` or referenced from an environment variable with
`api_key_env`. Catalogs never return the key value, only `has_api_key`.

---

## 8. End-to-end example

```bash
BASE=http://127.0.0.1:8765

# 1. Discover the protocol
curl -s $BASE/api/agents/protocol | jq .version

# 2. Validate AML
curl -s -X POST $BASE/api/agents/validate \
  -H 'Content-Type: application/json' \
  --data-binary @agent.json | jq .

# 3. Create it
curl -s -X POST $BASE/api/agents \
  -H 'Content-Type: application/json' \
  --data-binary @agent.json | jq .agent.id

# 4. Test the logic without side effects
curl -s -X POST $BASE/api/agents/support-triage/runs \
  -H 'Content-Type: application/json' \
  -d '{"input":{"message":"The export button is broken"},"dry_run":true}' | jq .id

# 5. Poll until it finishes
curl -s "$BASE/api/agents/runs/<run-id>?since=0" | jq '{status, steps, output}'

# 6. Debug with a breakpoint
curl -s -X POST $BASE/api/agents/support-triage/runs \
  -H 'Content-Type: application/json' \
  -d '{"debug":true,"breakpoints":["classify"],"input":{"message":"hello"}}' | jq .id
curl -s -X POST $BASE/api/agents/runs/<run-id>/step | jq .current_node
curl -s -X POST $BASE/api/agents/runs/<run-id>/resume | jq .status

# 7. Schedule it
curl -s -X POST $BASE/api/agents/support-triage \
  -H 'Content-Type: application/json' \
  -d @agent-with-schedule.json | jq .agent.schedule
curl -s -X POST $BASE/api/agents/support-triage/schedule \
  -H 'Content-Type: application/json' -d '{"enabled":true}' | jq .next_run
```

---

## 9. LLM authoring loop

Recommended loop for another model:

1. `GET /api/agents/spec/aml` and `GET /api/agents/spec/amp`.
2. `GET /api/agents/protocol` for the live node/ability catalog and model list.
3. Draft AML referencing only the abilities you grant.
4. `POST /api/agents/validate` and fix every `issues` entry with `level: "error"`
   (each names its `node` and `field`).
5. `POST /api/agents` to save.
6. `POST /api/agents/{id}/runs` with `dry_run: true`.
7. Poll `GET /api/agents/runs/{run_id}` and inspect `error` and `events`.
8. If a branch needs inspection, start a debug run with breakpoints, then
   `step`/`resume`.
9. Run once for real, inspect the output. For a swarm, also read
   `GET /api/agents/runs/{run_id}/swarm` to see the plan, messages, blackboard
   and reviews; the `swarm_end` event carries the leader's synthesis.
10. Only then enable a schedule.

Validation returns deterministic, human-readable errors, so this loop can run
without a browser.

---

## 10. Error codes

Errors are `{"error": "<actionable detail>", "code": "<coarse code>"}`.

| Code | HTTP | Meaning |
|---|---:|---|
| `bad_request` | 400 | Invalid JSON, AML, model selection, ability config, or request values. `error` contains the actionable detail. |
| `not_found` | 404 | Agent, run, template, model, endpoint or route missing. |
| `conflict` | 409 | The agent already has an active run, the run is not active/debuggable, deletion conflicts with an active run, or a clone id already exists. |
| `forbidden` | 403 | The request came from another web origin or was addressed to a non-local host name. |
| `method_not_allowed` | 405 | The path exists, but not for this HTTP method. |
| `too_large` | 413 | Request exceeds 1 MB. |
| `server_error` | 500 | Unexpected failure; run history still records node-level failures. |
| `upstream_error` | 502 | A model endpoint or local runtime answered with a failure. |
| `unavailable` | 503 | A required backend or runtime is not installed or not running. |

---

## 11. Security and operation

- Studio binds to `127.0.0.1` by default. Do not expose it to untrusted networks.
- There is no authentication. Use a reverse proxy with auth if remote access is
  unavoidable.
- Browser-borne attacks are refused: a request whose `Host` is not a loopback
  name, whose `Origin` is another site, or that writes without a JSON content
  type gets `403`/`400`. The server sends no CORS headers. Non-browser clients
  (curl, scripts, other agents) are unaffected.
- Run history, agents, memory and sandbox files live in `agent-workspace/`.
  Delete that directory to reset Agentic state.
- HTTP redirect targets and the connected peer are validated, so a public URL
  cannot redirect to or resolve to a private address. `allow_private` is taken
  from the ability config and cannot be widened by a node or model input.
- Shell access is opt-in per server via `MARIANA_AGENT_ALLOW_SHELL=1`.
- `sqlite.query` is read-only unless the ability config sets `write: true`;
  `ATTACH` is denied and a query deadline is enforced.
- `git.inspect` runs fixed read-only status, diff and log commands with external
  diffs, text conversion, pager and fsmonitor disabled.
- Imported AML is executable. Review `abilities`, URLs and shell commands before
  running.
- A swarm cannot reach another swarm, another run, the network or a wider file
  scope through IPC. Recipients, message kinds, sizes, quotas, reply links and
  the shared-memory keys are validated; write-like member tools are serialized
  and need both `swarm.allow_write` and the node's `write` flag.
- Native and text tool calls share the same allow-list, write scope and sandbox.
  A native schema is generated from the ability's `params` and never widens the
  ability's config; a rejected or malformed native call is reported to the model
  as a tool error.
- Schedules keep running while the Studio server is up, even if the browser is
  closed. Stop the server to stop schedules.

---

## 12. Changes in 1.10

AMP 1.10 is additive; 1.9 clients keep working.

- `llm` and `agent` model streams emit coalesced `assistant_delta` and
  `reasoning_delta` events with `node`, `iteration` and `text`. A live UI can
  render the model's current turn without polling the provider; the final answer
  is still the `node_finished` output.
- Deltas are trimmed first when a run keeps its bounded event history, so a
  long stream cannot push control events out of a reconnect snapshot.
- Swarm events add the owning flow `node` id, so a UI can place a member update
  on the node that launched the swarm.
- AML 1.9 adds `swarm.adaptive_effort`; discovery reports it with the other
  swarm node fields.
- Swarm policy now defaults to 16 agents and allows a ceiling of 64. The leader
  chooses the team. `swarm_start` limits report the separate model and tool
  concurrency caps (defaults 16 and 32), and teams larger than 12 are condensed
  in parallel group summaries before the final synthesis.
- Swarm snapshots, `swarm_end` events and run results add `cache`: tool-cache
  `hits`, `misses`, `saved_seconds`, `entries`, `chars` and `inflight`, plus
  `prompt_cached_tokens` and `prompt_cache_hit_rate`.

## 13. Changes in 1.9

AMP 1.9 is additive; 1.8 clients keep working.

- AML 1.8 adds swarm success criteria, per-member plans and `depends_on`
  waves, plus the `criteria`, `effort` and `reflection` node fields.
- Discovery's leader protocol includes `success_criteria`, `depends_on`, `plan`,
  `gaps` and `unmet_criteria`; swarm results expose all five.
- New `swarm_agent_plan` event. `swarm_agent_tool` adds `attempts`; member state
  adds `tool_calls`, `tool_failures`, `tool_retries`, `plan` and `depends_on`.
- Read-only tool calls that fail with a temporary error are retried once in both
  `swarm` and `agent` nodes; writes are never retried. Failure observations may
  carry a `hint`, and an unknown tool id gets a closest-match suggestion.

## 14. Changes in 1.8

AMP 1.8 is additive; 1.7 clients keep working.

- AML 1.7 adds the `assert` and `validate` guardrail nodes. Discovery already
  publishes every node schema from one table, so clients need no protocol change
  to render them.
- Discovery adds a `guardrails` object describing assertions, bounded schema
  validation and model fallback.
- `llm` and `agent` node defaults add `fallback_model`; a failed main model call
  retries with the fallback and an agent keeps the fallback for its remaining
  tool rounds.
- New run event: `validation` (the result and bounded error list of a `validate`
  node). Assert failures use the existing `node_failed` event and the normal
  retry/`on_error` policy.

## 15. Changes in 1.7

AMP 1.7 is additive; 1.6 clients keep working.

- Swarm members accept a `context` action (`get`, `find`, `release`, `status`).
  A run-wide pool grants extra prompt characters while it has free space and
  releases a member's grant at the end of a phase.
- Long tool results, peer results and compacted turns are stored behind opaque
  `ctx-…` handles; `context get` reads a slice with `chars` and `offset`.
- The swarm snapshot and node result gain `context` (capacity, used, free and
  per-member grants) and `context_items`.
- New event: `swarm_context` (grant, denial, search, release).
- New `news.headlines` ability and public web access for Chat/IDE agents.
  Results are compact; `source: auto` tries Google News then Bing News, and a
  requested date with no matches returns the newest items with a
  `date_matched: false` note.
- New `browser.fetch` ability renders JavaScript pages with Playwright or a
  Chromium binary under OS access.
- Swarm plans accept `tools: ["*"]` so a broad researcher can use every granted
  tool instead of being limited to one retrieval path.
- A swarm member that exhausts its tool budget gets one tool-free wrap-up call;
  if that fails too, its collected observations are shared with the leader, so
  the Chat turn still produces an answer.
- `allow_os_access` (or `MARIANA_AGENT_ALLOW_OS=1` / `MARIANA_AGENT_ALLOW_SHELL=1`)
  enables shell commands and private-network targets.
- AML 1.6 adds the matching `swarm` policy keys and the `context_chars` plan
  field.

## 16. Changes in 1.6

AMP 1.6 is additive; 1.5 clients keep working.

- Discovery adds `tool_calling`: the accepted formats (`json`, `dsml`, `native`),
  the native backends and the prompted fallback.
- `agent` and `swarm` nodes accept provider-native function calls and DeepSeek
  DSML `<invoke>` markup as well as canonical JSON. `tool_call`, `tool_result`
  and `swarm_agent_tool` are unchanged.
- Native and text calls share one allow-list, sandbox and write scope. An
  endpoint that rejects native tools falls back automatically.
- AML 1.5 document syntax is unchanged; this is an interoperability addition.

## 17. Changes in 1.5

AMP 1.5 is additive; 1.4 clients keep working.

- AML 1.4 adds the `swarm` node and the document-wide `swarm` policy. A leader
  model plans members, runs them in parallel or in a pipeline, and synthesises
  one answer.
- Members have an addressed IPC channel, a run-scoped shared blackboard and a
  peer-review/revision loop. Leader and member JSON protocols are published in
  `GET /api/agents/protocol` under `swarm`.
- Run snapshots include `swarm`. `GET /api/agents/runs/{run_id}/swarm` returns
  that state and `POST /api/agents/runs/{run_id}/swarm/messages` queues an
  operator message for a live leader.
- New events: `swarm_start`, `swarm_plan`, `swarm_leader`, `swarm_round`,
  `swarm_agent_start`, `swarm_agent_step`, `swarm_agent_delta`,
  `swarm_agent_status`, `swarm_agent_tool`, `swarm_agent_end`, `swarm_message`,
  `swarm_memory`, `swarm_review` and `swarm_end`.
- Swarm IPC, shared memory and write scopes have explicit quotas, allow-lists
  and sanitisation; a malformed model turn is reported back to that model.

## 18. Changes in 1.4

AMP 1.4 is additive; 1.3 clients keep working.

- New discovery document fields: AML 1.3, `sqlite.query`, `csv.read` and
  `git.inspect`; `llm` nodes gain the `format` field.
- `GET /api/agents/runs/{run_id}/events` streams complete run snapshots as
  server-sent events, so clients no longer need to poll during a run.
- `GET /api/agents/runs` lists recent runs across every agent.
- `POST /api/agents/{id}/clone` duplicates an agent and disables the copied
  schedule.
- Top-level AML `vars` seed the run variables, and `llm` nodes with
  `format: "json"` return a parsed `json` value.

## 19. Changes in 1.3

AMP 1.3 is additive; 1.2 clients keep working.

- `GET /api/agents` and `GET /api/agents/{id}/schedule` report `last_error` from
  the most recent schedule attempt.
- A due schedule that cannot start advances to its next slot and records the
  reason instead of retrying every scheduler tick.
- Run payloads are validated strictly; malformed booleans, triggers,
  breakpoints and unknown node ids return 400.
- `settings.timeout` is enforced inside model and tool loops, not only between
  nodes. Cancellation interrupts streaming model calls where the backend can
  stop.
- Stopping Studio cancels active runs; runs left active by an interrupted
  process are marked failed on the next start.
- HTTP abilities validate every redirect target and the connected peer address.

## 20. Changes in 1.2

AMP 1.2 is additive; 1.1 clients keep working.

- AML 1.2 adds `file.edit`, `file.search`, `file.tree`, `file.delete` and
  `code.diagnostics`; discovery exposes them automatically in `abilities`.
- `agent` nodes can run up to 40 tool rounds, return several calls in one reply
  (`calls`) and publish a plan (`plan`).
- New run event `plan`, with the normalized `[{text, status}]` list.
- Discovery reports `security.shell_enabled` so a client can tell whether
  `shell.exec` will run before a model tries it.
- The IDE layer is documented in `docs/ide.md`. It adds workspace, file,
  search, diagnostic, session, revert and streaming-run operations under
  `/api/ide` without changing any `/api/agents` contract.

## 21. Changes in 1.1

AMP 1.1 is additive; 1.0 clients keep working.

- Discovery returns the full node schema (`icon`, typed `fields`, `defaults`),
  `common_fields`, `field_types` and ability `params`.
- Validation returns structured `issues`.
- New endpoints: `POST /api/agents/runs/{run_id}/input`,
  `POST /api/agents/runs/{run_id}/breakpoints`, `POST /api/agents/generate`.
- Run snapshots add `vars` and `awaiting_input`; new events `node_input`,
  `tool_call`, `tool_result`, `input_requested`, `input_received`,
  `input_timeout`; `node_finished` adds `elapsed_ms`.
- New error codes `forbidden`, `method_not_allowed`, `upstream_error`,
  `unavailable`. Pausing or cancelling an unknown job is now `404`, and one that
  is not active `409` (both were `400`).
