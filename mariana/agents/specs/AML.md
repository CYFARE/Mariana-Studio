# Agent Management Language (AML) v1.8

AML is a portable, human-readable format for describing an agent as a directed
flowchart with abilities, model settings, schedules and run controls. The
canonical encoding is JSON; YAML is accepted on import when PyYAML is
installed. Studio stores the normalized JSON form.

- Media type: `application/vnd.mariana.aml+json`
- Current version: `1.8` (a superset of `1.7`, `1.6`, `1.5`, `1.4`, `1.3`, `1.2`, `1.1` and `1.0`; see [Changelog](#9-changelog))
- Transport for execution: [Agent Management Protocol (AMP)](./AMP.md)
- Studio location: **Agents** workspace → **Spec** panel

AML is designed so a person or another LLM can author, validate, run, test and
debug an automation entirely through HTTP.

---

## 1. Document shape

```json
{
  "aml": "1.8",
  "kind": "agent",
  "id": "daily-digest",
  "name": "Daily digest",
  "description": "Summarise the front page every morning.",
  "version": "1.0.0",
  "icon": "⇗",
  "model": "trained-model/mariana.pt",
  "vars": {"team": "support", "dry_run": false},
  "swarm": {
    "max_agents": 6, "max_steps": 8, "max_rounds": 2, "max_seconds": 300,
    "max_messages": 200, "allow_write": false, "allow_ipc": true,
    "shared_memory": true
  },
  "settings": {
    "max_steps": 100,
    "timeout": 120,
    "on_error": "stop",
    "max_output_chars": 60000
  },
  "abilities": [
    {
      "id": "fetch",
      "type": "http.fetch",
      "label": "Fetch a page",
      "enabled": true,
      "config": {"method": "GET", "timeout": 15, "allow_private": false}
    }
  ],
  "nodes": [
    {"id": "start", "type": "start", "label": "Start", "x": 60, "y": 160},
    {"id": "summary", "type": "llm", "label": "Summarise", "x": 320, "y": 140,
     "data": {"prompt": "Summarise: {{ input.message }}", "temperature": 0.4, "max_tokens": 500}},
    {"id": "end", "type": "end", "label": "Return", "x": 600, "y": 160,
     "data": {"output": "{{ nodes.summary.text }}"}}
  ],
  "edges": [
    {"from": "start", "to": "summary"},
    {"from": "summary", "to": "end"}
  ],
  "trigger": {"type": "manual"},
  "schedule": {
    "enabled": false,
    "type": "cron",
    "cron": "0 9 * * *",
    "input": {"message": "Morning news"}
  }
}
```

### Top-level fields

| Field | Type | Required | Notes |
|---|---|---|---|
| `aml` | string | yes | `"1.8"`, `"1.7"`, `"1.6"`, `"1.5"`, `"1.4"`, `"1.3"`, `"1.2"`, `"1.1"` or `"1.0"`. Documents are normalized to `1.8`; any other value validates with a warning. |
| `kind` | string | yes | Must be `"agent"`. |
| `id` | slug | recommended | Lowercase letters, digits and dashes. Derived from `name` when omitted. |
| `name` | string | yes | Display name, max 120 chars. |
| `description` | string | no | Max 2000 chars. |
| `version` | string | no | Author-managed semantic version. |
| `icon` | string | no | One or two display characters. |
| `model` | string | no | Default model id from `GET /api/models`. Empty = first available chat model. |
| `vars` | object | no | Initial flow variables, available as `vars.<name>` before any node runs. At most 100 keys; values are plain JSON. |
| `swarm` | object | no | Defaults and caps for every `swarm` node in the document. See **Swarm policy** below. |
| `model_settings` | object | no | `device`, `top_p` defaults for LLM nodes. |
| `settings` | object | no | Runtime limits, see below. |
| `abilities` | array | no | Explicitly granted tools. |
| `nodes` | array | yes | Flow nodes, at least one `start`. |
| `edges` | array | yes | Directed connections. |
| `trigger` | object | no | Informational (`manual`, `schedule`, `subagent`). |
| `schedule` | object | no | Durable automation schedule. |

### Settings

| Key | Default | Range | Meaning |
|---|---:|---:|---|
| `max_steps` | 100 | 1–1000 | Node executions allowed per run. |
| `timeout` | 120 | 1–3600 | Wall-clock seconds, excluding debug pauses. |
| `on_error` | `stop` | `stop`, `continue` | `continue` stores `{error, node}` and follows the first outgoing edge. |
| `max_output_chars` | 60000 | ≥ 1000 | Run history output truncation limit. |

### Swarm policy

The top-level `swarm` object is both the default and the security envelope for
every `swarm` node. A node may narrow it, never widen it.

| Key | Default | Range | Meaning |
|---|---:|---:|---|
| `max_agents` | 6 | 1–12 | Members in one swarm. |
| `max_steps` | 8 | 1–24 | Model rounds per member per phase. |
| `max_rounds` | 2 | 0–4 | Feedback rounds available to a node. |
| `max_seconds` | 300 | 10–3600 | Active time for one swarm node. |
| `max_messages` | 200 | 0–1000 | Total IPC messages in one swarm. |
| `max_message_chars` | 4000 | 200–20000 | Characters in one message. |
| `max_memory_entries` | 120 | 0–1000 | Shared-blackboard keys. |
| `max_memory_chars` | 8000 | 200–100000 | Serialized size of one memory value. |
| `max_context_chars` | 200000 | 0–2000000 | Run-wide extra prompt characters members may borrow; `0` disables dynamic context. |
| `max_agent_context_chars` | 64000 | 0–500000 | Most extra context one member may hold at once. |
| `max_context_item_chars` | 24000 | 500–200000 | Largest slice one context request may return. |
| `allow_write` | `false` | bool | Members may use write-like tools. |
| `allow_ipc` | `true` | bool | Members may message each other. |
| `shared_memory` | `true` | bool | Members may use the blackboard. |

A `swarm` node with `write: true` is rejected unless `allow_write` is true.

---

## 2. Nodes

Every node has:

```json
{"id": "unique-node-id", "type": "llm", "label": "Human label", "x": 320, "y": 140, "data": {}}
```

`x` and `y` are designer coordinates; the runtime ignores them. `data` is
type-specific. Unknown top-level node keys are ignored.

### Node types

| Type | Category | Purpose | `data` |
|---|---|---|---|
| `start` | trigger | Exactly one per agent. Receives the run input as `input`. | `{}` |
| `end` | output | Finishes the run. | `{"output": "<template>"}` |
| `llm` | intelligence | Calls a model once. | `model`, `system`, `prompt`, `format` (`text`/`json`), `temperature`, `max_tokens`, `fallback_model` |
| `agent` | intelligence | **AI agent** (1.1): a model that chooses abilities itself until it has an answer. | `model`, `system`, `goal`, `abilities` (ids; empty = all granted), `max_iterations` (1–40), `temperature`, `max_tokens`, `fallback_model` |
| `swarm` | intelligence | **Agent swarm** (1.4): a leader model plans a team, members work, talk, remember and review, then the leader synthesises. | `model`, `leader_model`, `system`, `goal`, `agents`, `tools`, `strategy`, `rounds`, `max_agents`, `max_steps`, `max_seconds`, `write`, `memory`, `ipc` |
| `tool` | action | Executes a granted ability. | `ability` (ability id), `input` (template/JSON) |
| `condition` | logic | Boolean branch. | `{"expression": "<safe expression>"}` |
| `assert` | logic | **Assert** (1.7): fail the run when a safe expression is false; a guardrail for a contract. | `expression`, `message` (template; `{{ expression }}` and `{{ value }}` are available) |
| `validate` | logic | **Validate JSON** (1.7): check a value against the bounded schema subset, then stop or warn. | `target` (template; empty = `last`), `schema`, `mode` (`fail`/`warn`) |
| `transform` | logic | Render or assign variables. | `template`, `format` (`text`/`json`), `assign` |
| `delay` | flow | Wait 0–300 seconds. | `{"seconds": 2}` |
| `human` | flow | **Human input** (1.1): pause until a person approves or answers. | `prompt`, `kind` (`approve`/`text`/`choice`), `choices`, `default`, `timeout` (0–86400 s, 0 = forever) |
| `subagent` | action | Run another agent (max depth 3). | `agent`, `input` |
| `foreach` | flow | **For each** (1.1): run another agent once per list item, in order. | `items` (template → list), `agent`, `input` (`{{ item }}`, `{{ index }}`), `max_items` (1–200) |
| `memory` | action | Persistent per-agent key/value memory. | `action` (`set`/`get`/`list`/`delete`), `key`, `value` |
| `note` | annotation | Designer note; passes `last` through. | `{"text": "..."}` |

Any node may include `"retry": 0..5` and `"retry_delay": seconds` in `data`.
Retries run the same node again before the error policy applies.

The live schema — every type with its label, icon, field definitions and
default `data` — is served by `GET /api/agents/protocol` (`node_types`,
`common_fields`). Designers and LLM authors should read it instead of
hardcoding this table.

#### `agent`: an LLM that uses abilities on its own

The node gives a chat model the granted abilities as tools and loops: the model
answers with a tool call, the runtime executes the ability and returns the
result, and so on until the model answers with a final result or
`max_iterations` is reached. The protocol is plain chat, so it works with any
model in the library:

```json
{"tool": "<ability id>", "input": {"<key>": "<value>"}}
{"calls": [{"tool": "<id>", "input": {}}, {"tool": "<id>", "input": {}}]}
{"plan": [{"text": "<step>", "status": "pending|active|done"}]}
{"final": "<the complete answer>"}
```

The runtime also accepts DeepSeek's DSML `<invoke>` markup and provider-native
function calls. Native definitions are derived from the ability's `params`;
provider-safe aliases (`file_read`) are translated back to the ability id.
Every spelling produces the same `tool_call` and `tool_result` events and obeys
the same allow-list, write scope and sandbox. An endpoint that rejects native
tools falls back to the prompted JSON form.

A `calls` array runs several tools in order and returns all observations in one
message. A `plan` updates `events` (type `plan`) without ending the loop. A reply
that is not such an object is taken as the final answer. An unknown, disabled or
failing tool is reported back to the model, which may recover. All sandbox rules
of §5 apply to every call.

AML 1.2 raises the tool-round limit to 40 and adds the coding abilities below,
so an `agent` node can inspect, edit and verify a project directly.

Both `llm` and `agent` nodes accept `fallback_model`. If the main model call
fails, the runtime logs the failure and retries that call with the fallback; an
agent node keeps using the fallback for its remaining tool rounds. An empty
fallback, or one equal to the main model, is ignored and reported as a warning.

#### `swarm`: a leader-led team of agents

A `swarm` node runs one bounded, leader-driven run. A leader model reads the
goal and plans the team; members work in parallel (or in a pipeline), exchange
addressed messages, share a run-scoped blackboard, and — when feedback rounds
are configured — review and revise each other's work. The leader synthesises
the final answer. The leader may be a different model (`leader_model`).

```json
{
  "id": "team",
  "type": "swarm",
  "label": "Research team",
  "data": {
    "model": "",
    "goal": "{{ input.message }}",
    "criteria": "Every claim cites a file path.\nThe answer states what was not verified.",
    "system": "Prefer primary sources and quote exact paths.",
    "agents": [
      {"id": "reader", "label": "Reader", "role": "investigator",
       "task": "Read the affected module", "tools": ["read", "search"],
       "writes": [], "plan": ["Locate the entry point", "Collect evidence"]},
      {"id": "checker", "label": "Checker", "role": "verifier",
       "task": "Check the tests and call sites", "read_only": true,
       "depends_on": ["reader"]}
    ],
    "tools": [],
    "strategy": "peer_review",
    "rounds": 1,
    "effort": "high",
    "reflection": true,
    "max_agents": 4,
    "max_steps": 6,
    "max_seconds": 240,
    "write": false,
    "memory": true,
    "ipc": true
  }
}
```

| Key | Default | Meaning |
|---|---:|---|
| `model` | agent model | Model for members. |
| `leader_model` | `model` | Model for the leader. |
| `goal` | `last` | Task for the whole swarm. |
| `criteria` | – | Optional acceptance checks, one per line. The leader must verify them in the final answer; empty lets it derive testable criteria from the goal. |
| `system` | – | Extra instructions shared by every member. |
| `agents` | `[]` | Optional `[{id, label, role, task, instruction, tools, writes, read_only, context_chars, depends_on, plan}]`. Empty lets the leader choose the team, its plans and dependencies. A writing member needs a `writes` scope; omit it to keep that member read-only. |
| `tools` | all granted | Ability ids members may use. `[]` means every granted ability. |
| `strategy` | `auto` | `auto`, `map_reduce`, `peer_review`, `debate` or `pipeline`. |
| `rounds` | `1` | Peer-review/revision rounds, 0–4. The leader may choose fewer, never more. Each round reviews then revises. |
| `effort` | `""` | `low`, `medium`, `high` or empty (provider default). Sent as `reasoning_effort` where the endpoint supports it, for planning, member work, review decisions and synthesis. |
| `reflection` | `true` | Ask a member once to finish or justify open plan steps before it returns its answer. |
| `max_agents` | `4` | Team size, 1–8. |
| `max_steps` | `6` | Model rounds per member per phase, 1–12. |
| `max_seconds` | `240` | Active time budget, 10–1800. |
| `write` | `false` | Members may use write-like tools. Requires top-level `swarm.allow_write`. |
| `memory` | `true` | Shared blackboard. |
| `ipc` | `true` | Agent-to-agent messages. |

The team shape is the leader's decision, using the goal and any `agents` the
caller supplied. Explicit member tasks win; the leader can add roles, tools,
write scopes, plans and extra members up to `max_agents`. `depends_on` turns the
team into a small DAG: members whose dependencies are still running wait, and
each dependency wave runs concurrently. Unknown ids, self-dependencies and
cycles are removed by the coordinator with a warning. `strategy: "pipeline"`
runs members in order and gives each one the previous outputs; every other
strategy uses dependency waves (members without dependencies start together).

**Leader protocol.** The leader answers with one JSON object per phase:

```json
{"strategy": "map_reduce|peer_review|debate|pipeline", "rounds": 0,
 "success_criteria": ["testable check"],
 "agents": [{"id": "...", "label": "...", "role": "...", "task": "...",
              "instruction": "...", "tools": ["ability-id"], "writes": ["src/"],
              "read_only": true, "context_chars": 16000,
              "depends_on": ["<member-id>"],
              "plan": [{"text": "<step>", "status": "pending|active|done"}]}],
 "notes": "..."}

{"reviews": [{"reviewer": "member-id", "target": "member-id", "focus": "..."}],
 "reason": "..."}

{"continue": true, "reason": "...", "gaps": ["unmet requirement"]}

{"final": "...", "summary": "...", "warnings": [],
 "unmet_criteria": ["criterion without evidence"], "confidence": 0.0}
```

**Member protocol.** A member answers with one JSON object per turn; a plain
reply becomes its final answer. Actions may be supplied together in an
`actions` array or as top-level keys:

```json
{"tool": "<ability id>", "input": {"<key>": "<value>"}}
{"calls": [{"tool": "...", "input": {...}}]}
{"say": {"to": "leader|all|<member-id>", "kind": "message|request|response|review|feedback|status|handoff",
         "text": "...", "reply_to": "<message id, optional>"}}
{"memory": {"action": "set|get|list|delete", "key": "finding.x", "value": {"ok": true},
            "note": "why it matters"}}
{"plan": [{"text": "<step>", "status": "pending|active|done"}]}
{"context": {"action": "get|find|release|status", "handle": "ctx-…",
             "key": "finding.x", "peer": "<member-id>", "query": "<text>",
             "chars": 8000, "offset": 0}}
{"review": {"target": "<member-id>", "score": 1-5, "strengths": [], "issues": [],
             "suggestions": [], "summary": "..."}}
{"final": "<complete findings>"}
```

A member sees its unread messages, a compact blackboard digest, its own peer-results
digest, its current plan, the leader's success criteria, the leader's per-member
instruction (if any), and the tools it was assigned before every model round. A
malformed turn is reported back with the expected shape, so a small model can
recover. The leader's `rounds` choice is capped by the node's `rounds` value.

**Planning and reflection.** A member's `plan` action replaces its plan, is
persisted in the snapshot and result, and may contain at most 12 steps; if a
model marks several active steps, only the first stays active. The plan is
included in every following prompt. When `reflection` is true and a member is
about to finish with open steps, the coordinator asks it once to complete them
or state why they are unnecessary. If it still finishes, the run records a
warning. The leader's plan seeds the team's starting plans on the blackboard.

**Tool grants.** The runtime never offers a tool the node did not grant. A member
that names any retrieval tool receives the whole retrieval family
(`news.headlines`, `http.fetch`, `browser.fetch`) when those abilities are
available, so an empty or blocked feed has a fallback. A task that asks for a
file or a blackboard hand-off receives `file.write` when the swarm allows writes
and the member has a write scope (or the server grants OS access); the executor
still enforces the scope on every write. `tools: ["*"]` on a member grants every
tool the node was given, an omitted list does the same, and `tools: ["none"]`
grants no tools to a member that only reasons over the briefing.

**Tool reliability.** A read-only call whose failure looks temporary (timeout,
connection reset, rate limit, 5xx) is retried once with a status event and the
attempt count in `swarm_agent_tool`. Writes are never retried automatically,
because a timed-out write may already have landed. Deterministic failures
(missing path, sandbox denial, invalid input, unknown tool) return immediately
with a short next-action hint; an unknown tool id is matched to the closest
granted id when one is unambiguous. Repeated identical calls earn a hint after
the third attempt. Every member reports `tool_calls`, `tool_failures` and
`tool_retries`, and the same retry/hint rules apply to an `agent` node.

**Shared knowledge.** Before the work phase the runtime seeds the blackboard with
`swarm.goal` and `swarm.plan` (strategy, rounds, success criteria and each
member's id, label, role, task, instruction, dependencies and plan), authored by
`leader`. When a member finishes a work or
revision phase its result is written to `result.<member-id>` (label, role, score,
version, summary and output) as that member, so peers never need to ask for it.
Member and reviewer prompts include a short peer-results digest
(`label (id, role, vN, score S): output`) and the team list with every member's
task; long values remain on the blackboard and are read with `memory get`. These
writes count against `max_memory_entries`; when the board is full, automatic
writes are skipped and the run continues. Duplicate plan tasks (case- and
whitespace-insensitive) are dropped with a warning so two members never do the
same work.

**Dynamic context.** Context is a shared, run-wide resource. Each member starts
with `max_prompt_chars` and may borrow extra characters from a pool of
`max_context_chars` (no member holds more than `max_agent_context_chars`). A
borrowed grant is released when its phase ends, so idle or finished members stop
consuming space and another member can use it. A prompt is also capped by the
selected model's real context window (when the gateway reports one), so a grant
cannot push a request past what the model accepts.

Truncation is loss-free. Long tool results, peer results and compacted turns are
stored under `ctx-…` handles; the prompt shows a preview plus the handle. A
member reads a stored item, a blackboard key or another member's complete result
with the `context` action:

```json
{"context": {"action": "get", "handle": "ctx-1a2b3c4d5e", "chars": 8000, "offset": 0}}
{"context": {"action": "get", "key": "finding.hot-loop", "chars": 12000}}
{"context": {"action": "get", "peer": "profiler", "chars": 8000}}
{"context": {"action": "find", "query": "retry helper"}}
{"context": {"action": "status"}}  or  {"context": {"action": "release"}}
```

A request is granted only while the pool has free space; the reply reports the
exact granted slice, the total size and the offset to request next, or an
actionable used/free report when it cannot grant anything. Compaction keeps a
ledger message naming the handles for the turns it removed, so a long coding run
can still recall an earlier path or tool output instead of guessing. The leader
may pre-allocate `context_chars` in a plan for a member that must read unusually
long files or logs.

**Dynamic step budget.** A member that is still gathering evidence when it
reaches `max_steps` gets one bounded extension (half the original budget, at most
8 extra rounds and never above 24) provided it has collected at least one
observation. The extension is announced with `swarm_agent_status` and recorded in
`warnings`; a model that only loops errors does not receive it. A one-round
budget is never extended. `max_seconds` still bounds the run.

**Feedback loop.** With `rounds > 0` the leader assigns peer reviews (or a
safe ring if the model returns no valid assignment). Reviewers may use read-only
tools to verify a claim, then record a score, issues and concrete suggestions.
Those reviews are delivered to the target over IPC and stored in the blackboard.
The target revises its work in the next phase, incrementing its output version.
Between rounds the leader decides whether another round would help; the loop
stops at `rounds`, on the leader's decision, or when no review was produced.
A review that fails is recorded in `warnings` and never discards the member's
already completed work; a reviewer that names the wrong target is corrected to
its assignment and asked to retry.

**Security.** IPC and memory are private to one swarm run. There is no network
transport, no global registry and no delivery across runs; recipients, message
kinds, sizes, per-sender quotas, `reply_to` links and memory keys are checked.
Messages are data and never become tool calls. Members can use only their
assigned abilities; when writes are enabled, write calls are serialized and a
member's `writes` prefixes are enforced before the sandbox runs. A member with
no `writes` scope is read-only unless it explicitly sets `read_only: false`;
`writes: ["*"]` grants unrestricted sandbox writes. A member with
a write scope cannot use `shell.exec`, because a command cannot be constrained
to that scope. The time budget is checked before every model and tool call, and
the run stops waiting at the deadline even when a backend ignores cancellation;
unfinished members are finalised in the result. An empty model reply is retried
once before it fails a member step or a leader step.

#### `assert`: a contract that stops a bad run

An `assert` node evaluates its `expression` with the same safe expression
language as a condition. A truthy result passes the previous output through as
`{ok: true, value, expression}`. A false result fails the node with
`Assertion failed at <node>: <message>`; the message is a template and may use
`{{ expression }}` and `{{ value }}`. The normal `retry` policy and
top-level `on_error` setting still apply.

```json
{
  "id": "has-sources",
  "type": "assert",
  "data": {
    "expression": "contains(lower(last.text), 'sources')",
    "message": "The report must include a Sources section"
  }
}
```

#### `validate`: structured-output guardrail

A `validate` node checks `target` (or `last` when the target is empty) against a
bounded JSON Schema subset. The supported keywords are `type`, `required`,
`properties`, `additionalProperties` (boolean), `items`, `enum`, `const`,
`anyOf`, `minimum`, `maximum`, `minLength`, `maxLength`, `minItems`, `maxItems`
and `pattern`. Unknown keywords are validation errors rather than silently
ignored constraints.

`mode: "fail"` fails the node with up to five `path: message` details.
`mode: "warn"` continues and returns `{valid, errors, value, mode}`, so a
condition can branch on `nodes.<id>.valid`. Every execution emits a
`validation` run event with the error list bounded for the stream.

```json
{
  "id": "check-report",
  "type": "validate",
  "data": {
    "target": "{{ nodes.report.json }}",
    "mode": "fail",
    "schema": {
      "type": "object",
      "required": ["title", "items"],
      "properties": {
        "title": {"type": "string", "minLength": 3},
        "items": {"type": "array", "minItems": 1, "items": {"type": "string"}}
      },
      "additionalProperties": false
    }
  }
}
```

#### `human`: wait for a person

The run reports `status: "paused"` with `awaiting_input` set until someone
answers through `POST /api/agents/runs/{run_id}/input`. The wait does not count
toward `settings.timeout`. When `timeout` elapses the node uses `default`, or
fails if there is none. In test mode it never waits: it uses `default`, else
`true` / the first choice / an empty string.

### Node output and state

The interpreter keeps a context available to every template:

```json
{
  "input": "<the run input>",
  "vars": {"<name>": "<value>", "...": "..."},
  "last": "<previous node output>",
  "nodes": {"<node-id>": "<that node's output>", "...": "..."},
  "agent": {"id": "...", "name": "...", "model": "..."},
  "run": {"id": "...", "trigger": "manual", "dry_run": false},
  "now": "2026-01-01T09:00:00"
}
```

If the run input is not an object, it is wrapped as `{"value": <input>}`.

Node outputs:

- `start`: the run input.
- `llm`: `{"text", "model", "usage", "elapsed", "prompt_format"}`. With `format: "json"`, the parsed value is also returned as `json`; a reply with no JSON value fails the node with an actionable error.
- `agent`: `{"text", "iterations", "steps": [{"iteration", "ability", "input", "output" | "error"}], "model", "usage", "exhausted"}`.
- `swarm`: `{"ok", "swarm", "final", "summary", "text", "strategy", "round", "phase", "agents", "completed", "failed", "messages", "reviews", "blackboard", "context", "context_items", "context_handles", "changed_files", "success_criteria", "unmet_criteria", "gaps", "usage", "warnings", "elapsed"}`. `text` is the leader's synthesised answer, suitable for an `end` node. `success_criteria` lists the caller's and leader's checks; `unmet_criteria` and `gaps` report what the leader could not verify. `context` reports the pool capacity, used and free characters plus each member's grant; `context_handles` lists recent temporary handles by label and size.
- `tool`: the ability result (for example `{"status", "headers", "body", "json"}` from `http.fetch`).
- `condition`: `{"result": true|false, "value", "expression"}`.
- `assert`: `{"ok": true, "value", "expression"}`; a false assertion fails the node with the rendered message.
- `validate`: `{"valid", "errors": [{"path", "rule", "message"}], "value", "mode"}`; conditions can branch on `valid` in warn mode.
- `transform`: `{"value", "assigned"}`.
- `delay`: `{"delayed": seconds, "dry_run": bool}`.
- `memory`: `{"key", "value", ...}` or `{"keys", "values"}`.
- `human`: `{"value", "text", "timed_out"}` plus `"approved": bool` for `kind: "approve"`.
- `subagent`: the child agent's final output.
- `foreach`: `{"items": [child outputs], "count", "truncated"}`.
- `end`: `{"output": rendered-output}`; the run result is the unwrapped `output`.

---

## 3. Edges and branching

```json
{"from": "source-id", "to": "target-id", "when": "always"}
```

- `when` is `always` (default), `true`, or `false`.
- Ordinary nodes follow their first outgoing edge. Additional outgoing edges are
  never followed; validation warns about them.
- A `condition` node chooses the edge whose `when` matches the boolean result.
  If no matching branch exists but an `always` edge exists, that edge is used;
  otherwise the run fails with an actionable error.
- Cycles are allowed and bounded by `settings.max_steps`.
- Exactly one edge is followed per visited node. Parallel fan-out is not part of
  AML; use a sequence, a `subagent`, or `foreach` instead.

---

## 4. Templates and expressions

Template placeholders use `{{ ... }}`.

- If the whole string is one placeholder (`"{{ nodes.summary.text }}"`), the
  native value is returned (object, list, number, bool).
- Mixed text (`"Report: {{ last }}"`) interpolates stringified values. Several
  placeholders in one string are evaluated independently.
- `vars.<name>` is available from the first node and starts with the document's
  top-level `vars` object.
- `null` stringifies to empty, booleans to `true`/`false`, objects to JSON.

### Safe expression language

Conditions and templates use a sandboxed expression subset. There is no
`eval`, attribute access to arbitrary Python objects, imports, or I/O.

Supported syntax: constants, names from the context, lists, tuples, dicts,
subscripts/slices, unary `not`/`-`/`+`, arithmetic `+ - * / // % **`
(exponent limited), comparisons, `and`/`or`, and `x if test else y`.
`and`/`or` short-circuit and return the deciding operand, as in Python, so
`{{ input.name or 'anonymous' }}` yields the name or the fallback.

Names: `input`, `vars`, `last`, `nodes`, `agent`, `run`, `now`, `true`,
`false`, `null`.

Helpers:

| Helper | Meaning |
|---|---|
| `contains(value, needle)` | substring test |
| `startswith(value, prefix)` / `endswith(value, suffix)` | string prefix/suffix |
| `lower`, `upper`, `trim` | string transforms |
| `split(value, sep?)`, `join(values, sep?)` | string/list conversion |
| `len(value)` / `length(value)` | size |
| `str`, `int`, `float`, `bool`, `json`, `dumps` | conversion |
| `coalesce(a, b, ...)` | first non-empty value |
| `matches(value, regex)` | regular-expression search |
| `get(value, "a.b.c", default)` | dotted lookup |
| `now()`, `today()` | local time helpers |
| `min`, `max`, `round`, `abs`, `floor`, `ceil` | arithmetic helpers |

Safe methods: string methods `lower`, `upper`, `strip`, `lstrip`, `rstrip`,
`startswith`, `endswith`, `split`, `replace`, `find`, `count`, `join`; dict
methods `get`, `keys`, `values`, `items`; list methods `count`, `index`.

Examples:

```text
contains(lower(nodes.classify.text), 'urgent')
nodes.fetch.json.hits[0].title
input.message if input.message else 'empty request'
len(nodes.list.files) > 3 and not vars.paused
```

---

## 5. Abilities

Abilities are **granted** on the agent document and referenced by node `data.ability`.
A node cannot use an ability that is not listed. `enabled: false` disables it.

```json
{"id": "fetch", "type": "http.fetch", "label": "Fetch", "enabled": true,
 "config": {"method": "GET", "timeout": 15, "allow_private": false}}
```

| Type | Config | Output |
|---|---|---|
| `http.fetch` | `url` or node input, `method`, `headers`, `body`, `timeout`, `allow_private`, `extract` (`raw`/`text`), `max_chars` | `status`, `headers`, `body`, `json` when parseable; with `extract: "text"`, `text` (readable text, HTML stripped) replaces `body` |
| `news.headlines` | `url` (empty = Google News), `query`, `source` (`auto`/`google`/`bing`), `region`, `language`, `date`, `limit` (1–50), `include_summary`, `timeout` | `url`, `source_used`, `feed_title`, `items[{title, link, source, published, date}]`, `count`, `total`, `date_matched`, `latest_date`, `dates`, `note`; `auto` tries Google News then Bing News |
| `browser.fetch` | `url`, `timeout`, `max_chars` | `title`, `text`, `chars`, `truncated`, `engine`; renders JavaScript in Playwright or a Chromium binary and requires OS access |
| `file.read` | `path`, `start_line`, `end_line` (default 200 lines) | `path`, `content`, `start_line`, `end_line`, `total_lines`, `truncated` |
| `file.write` | `path`, `content` | `path`, `bytes`, `written` |
| `file.list` | `pattern` | `pattern`, `files[{name, bytes, modified}]` |
| `file.edit` | `path`, `old`, `new`, `replace_all` | `path`, `replacements`, `bytes`, `changed`; the old text must be unique unless `replace_all` |
| `file.search` | `query`, `regex` (default true), `case_sensitive`, `include` (glob), `max_results` | `query`, `matches[{path, line, text}]`, `files_scanned`, `truncated` |
| `file.tree` | `max_entries` | `entries[{path, type, bytes?}]`, `truncated`; dependency folders are skipped |
| `file.delete` | `path` | `path`, `deleted` |
| `code.diagnostics` | `path` (file, folder or empty) | `files_checked`, `issues[{path, line, column, message, rule, severity}]`, `ok`; checks Python syntax, JSON and optional pyflakes rules |
| `sqlite.query` | `path`, `sql`, `params` (list or object), `max_rows`, `timeout`; ability config `write` (default false) | `columns`, `rows`, `row_count`, `truncated`, `changes`; one parameterized SQLite statement in the sandbox |
| `csv.read` | `path`, `delimiter`, `header` (default true), `max_rows` | `columns`, `rows` (objects), `row_count`, `truncated`; CSV/TSV in the sandbox |
| `git.inspect` | `action` (`status`/`diff`/`log`), `path`, `staged`, `limit`, `max_chars` | Read-only `branch`, `files`, `diff` or `commits` for a Git repository inside the sandbox (git must be installed) |
| `memory` | `action`, `key`, `value` | memory result |
| `math.eval` | `expression` | `expression`, `value` |
| `time.now` | – | `local`, `utc`, `epoch` |
| `shell.exec` | `command`, `timeout` (1–120 s) | `returncode`, `stdout`, `stderr`, `ok` |

### Security model

- File abilities are sandboxed to `agent-workspace/files`. Absolute paths are
  allowed only when they resolve inside that directory. Path traversal and
  symlink escapes are rejected.
- HTTP blocks localhost/private/link-local/reserved addresses by default. Set
  `allow_private: true` on that specific ability to opt in. Cloud metadata
  addresses (`169.254.169.254`, `metadata.google.internal`) are always blocked.
- `news.headlines` reads a public RSS/Atom feed and returns structured, dated
  headlines. With no URL it uses Google News for `region` and `language`; a
  `query` becomes a Google News search or filters a supplied feed. `source: auto`
  retries the query on Bing News when Google returns nothing, and `source: bing`
  selects Bing directly. Both news and HTTP can reach private addresses when the
  server owner enables **OS access**.
- `browser.fetch` renders a page with Playwright or a Chromium binary and returns
  its visible text. It requires **OS access** and reports an actionable install
  hint when no browser is available.
- `sqlite.query` opens a sandbox database read-only unless the ability config sets
  `write: true`; a model or node input cannot widen that permission. `ATTACH` is
  denied, and every query runs with a deadline.
- `git.inspect` runs fixed, read-only `git status`, `git diff` or `git log`
  arguments with external diffs, text conversion, pager and fsmonitor disabled.
  It never accepts a shell command.
- `shell.exec` requires the server owner to opt in, either with
  `allow_os_access` in Studio settings, `MARIANA_AGENT_ALLOW_OS=1`, or
  `MARIANA_AGENT_ALLOW_SHELL=1`; otherwise the node fails with an actionable
  error. OS access also lets HTTP tools reach private addresses and enables
  `browser.fetch`. A model cannot widen this permission from a node input or an
  imported AML document.
- A `swarm` shares no channels or memory with another swarm or run. Member
  messages, blackboard entries and write scopes are validated by the runtime;
  write-like member calls are serialized and require both `swarm.allow_write`
  and the node's `write` flag. A swarm cannot use an ability it was not granted,
  and an inability to parse a model's JSON turn is reported back to that model.
- Test mode (`dry_run`) mocks every LLM call and skips all side effects:
  HTTP, file writes, SQLite writes, shell commands, memory writes, and delays.
  File reads, CSV/SQLite reads, Git inspection and expression evaluation still
  run so logic can be tested.
- AML is executable automation. Only import agents you trust, and keep Studio
  bound to `127.0.0.1`.

---

## 6. Schedules

Schedules are durable; Studio persists the next run time and a background
scheduler starts runs even when the Agentic tab is not open.

```json
"schedule": {
  "enabled": true,
  "type": "cron",
  "cron": "0 9 * * 1-5",
  "input": {"message": "Weekday report"}
}
```

| Type | Fields | Semantics |
|---|---|---|
| `manual` | – | No automatic runs. Must be disabled. |
| `interval` | `interval_minutes` (1–525600) | First run after one interval, then repeating. |
| `cron` | `cron` (5 fields) | Standard minute/hour/day/month/weekday expression. |
| `once` | `once_at` (ISO date-time) | Runs once at a future local time. |

Cron supports `*`, numbers, names for months/weekdays, lists (`1,15`), ranges
(`9-17`), and steps (`*/5`, `0-30/10`). Sunday is `0` or `7`. The server uses
local time. Day-of-month and day-of-week are ORed when both are restricted,
matching common cron behavior.

Scheduled runs use `trigger: "schedule"` and pass `schedule.input` as the run
input. Overlapping runs for the same agent are skipped.

---

## 7. Authoring checklist (for humans and LLMs)

1. Use a fresh lowercase `id` and a display `name`.
2. Include exactly one `start` and at least one `end`.
3. Give every node a unique id. Connect from `start` through every reachable node.
4. Use a `condition` node for branches; add one outgoing `when: "true"` and one
   `when: "false"` edge.
5. Grant only the abilities the flow actually uses, and reference the ability id
   from each `tool` node.
6. Prefer templates that reference `nodes.<id>` or `last`; test with `dry_run`.
7. Set `model` at agent level or on each `llm` / `agent` node.
8. Keep `max_steps` and `timeout` as tight as practical.
9. Enable a schedule only after a successful manual test.
10. Validate with `POST /api/agents/validate` before saving or running; fix every
    entry of `issues` with `level: "error"`.
11. Prefer one `agent` node over a long chain of `tool` nodes when the steps
    depend on what earlier steps return; keep `max_iterations` low.
12. Put a `human` node before any irreversible action.
13. Use a `swarm` node when a task has independent parts, needs several areas
    investigated at once, or benefits from independent review; keep `rounds` at
    1 unless a second review pass is clearly useful.
14. Enable `swarm` writes only on a document whose `swarm.allow_write` is true,
    and give each writing member a disjoint `writes` scope. A member without a
    scope stays read-only; use `writes: ["*"]` only when unrestricted sandbox
    writes are intended.

---

## 8. Complete example: support triage

```json
{
  "aml": "1.8",
  "kind": "agent",
  "id": "support-triage",
  "name": "Support triage",
  "model": "",
  "settings": {"max_steps": 40, "timeout": 180, "on_error": "stop"},
  "abilities": [{"id": "memory", "type": "memory", "config": {}}],
  "nodes": [
    {"id": "start", "type": "start", "label": "Message", "x": 40, "y": 240},
    {"id": "classify", "type": "llm", "label": "Classify", "x": 240, "y": 220,
     "data": {"temperature": 0, "max_tokens": 8,
              "prompt": "Answer urgent or normal for: {{ input.message }}"}},
    {"id": "branch", "type": "condition", "label": "Urgent?", "x": 470, "y": 240,
     "data": {"expression": "contains(lower(nodes.classify.text), 'urgent')"}},
    {"id": "escalate", "type": "memory", "label": "Escalate", "x": 700, "y": 120,
     "data": {"action": "set", "key": "last-urgent", "value": "{{ input.message }}"}},
    {"id": "queue", "type": "memory", "label": "Queue", "x": 700, "y": 360,
     "data": {"action": "set", "key": "last-normal", "value": "{{ input.message }}"}},
    {"id": "end", "type": "end", "label": "Return", "x": 930, "y": 240,
     "data": {"output": "{{ nodes.classify.text }}"}}
  ],
  "edges": [
    {"from": "start", "to": "classify"},
    {"from": "classify", "to": "branch"},
    {"from": "branch", "to": "escalate", "when": "true"},
    {"from": "branch", "to": "queue", "when": "false"},
    {"from": "escalate", "to": "end"},
    {"from": "queue", "to": "end"}
  ],
  "schedule": {"enabled": false, "type": "manual"}
}
```

---

## 9. Changelog

### 1.8

AML 1.8 is additive. Every 1.7, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1 and 1.0
document is valid and runs unchanged.

- Swarm leaders now produce testable `success_criteria`, a per-member `plan`
  and optional `depends_on` edges. Members without dependencies run together;
  dependent waves wait for their inputs. Cycles are removed with a warning.
- New `swarm` node fields: `criteria` (caller acceptance checks), `effort`
  (`low`, `medium`, `high` or provider default) and `reflection` (one plan
  self-check before a member finishes).
- A member's `plan` action is persisted, exposed in snapshots/results and
  replayed in every prompt; finishing with open steps triggers one reflection
  turn and a warning if the steps remain open.
- Leaders report `gaps` after a review round and `unmet_criteria` at synthesis;
  both reach revisions, the final result and the swarm card.
- Read-only tool failures with a temporary error are retried once in swarms and
  agent nodes. Writes are never retried. Deterministic failures carry a short
  recovery hint, and an unknown tool id is matched to the closest granted id.
- New `swarm_agent_plan` event carries the structured plan, its version and its
  completion count.

### 1.7

AML 1.7 is additive. Every 1.6, 1.5, 1.4, 1.3, 1.2, 1.1 and 1.0 document is
valid and runs unchanged.

- New `assert` node: evaluate a safe expression and fail with an actionable
  message when it is false. The message can interpolate `{{ expression }}` and
  `{{ value }}`.
- New `validate` node: validate a value against a bounded JSON Schema subset
  (`type`, `required`, `properties`, `additionalProperties`, `items`, `enum`,
  `const`, `anyOf`, numeric and length bounds, `pattern`). `mode: "fail"` stops
the node; `mode: "warn"` exposes `valid` and `errors` for a condition.
- The new `validation` run event reports the validation result without making a
template parse the full error list.
- `llm` and `agent` nodes accept `fallback_model`. A provider failure retries the
  call with the fallback; an agent keeps the fallback for its remaining rounds.

### 1.6

AML 1.6 is additive. Every 1.5, 1.4, 1.3, 1.2, 1.1 and 1.0 document is valid
and runs unchanged.

- Swarm members gain a `context` action (`get`, `find`, `release`, `status`) for
  dynamic context management. A run-wide pool grants extra prompt characters
  while it has free space, and long content is retained behind `ctx-…` handles
  instead of being dropped.
- New top-level `swarm` policy keys: `max_context_chars`,
  `max_agent_context_chars` and `max_context_item_chars`.
- Swarm plans and explicit member specs accept `context_chars` to pre-allocate
  a larger context grant to one member.
- Compaction of long member conversations keeps a retrievable ledger, and prompt
  fitting uses the model's real context window when it is known.
- New `news.headlines` ability for public RSS/Atom feeds, including Google News
  search, Bing News fallback, region/language selection and date filtering, so
  agents can retrieve current headlines with sources and dates. Results are
  compact by default; a requested date with no items returns the newest items
  plus a `date_matched: false` note.
- New `browser.fetch` ability renders JavaScript pages with Playwright or a
  Chromium binary when OS access is enabled.
- Swarm plans accept `tools: ["*"]` to grant a broad researcher every tool the
  swarm was given, and the leader is instructed to use it rather than starving a
  researcher of alternate retrieval tools.
- A swarm member that reaches its tool budget is asked once more, without tools,
  to answer from the observations it already collected. If that also fails, the
  collected observations are shared with the leader instead of failing the whole
  turn.
- Public web tools are available to Chat and the IDE; shell access is controlled
  by the server-wide `allow_os_access` setting (or `MARIANA_AGENT_ALLOW_OS=1` /
  `MARIANA_AGENT_ALLOW_SHELL=1`) and can be granted to swarm members without a
  file write scope.
- The `swarm` node output adds `context` and `context_items`; the new
  `swarm_context` event reports grants, denials, searches and releases.

### 1.5

AML 1.5 is additive. Every 1.4, 1.3, 1.2, 1.1 and 1.0 document is valid and
runs unchanged.

- The `agent` and `swarm` nodes accept provider-native function calls and
  DeepSeek DSML `<invoke>` markup in addition to the canonical JSON action.
  Native definitions are derived from ability `params`; provider-safe aliases
  are translated back to the ability id.
- Every spelling shares one allow-list, write scope, sandbox and event stream:
  `tool_call`, `tool_result` and `swarm_agent_tool` are unchanged.
- An endpoint that rejects native tools automatically falls back to the
  prompted JSON path.

### 1.4

AML 1.4 is additive. Every 1.3, 1.2, 1.1 and 1.0 document is valid and runs
unchanged.

- New `swarm` node: a leader model plans a team, members work in parallel or in
  a pipeline, exchange addressed IPC messages, share a run-scoped blackboard,
  review and revise each other for a configurable number of rounds, and the
  leader synthesises one answer.
- New top-level `swarm` policy with agent, step, round, time, message and memory
  limits plus `allow_write`, `allow_ipc` and `shared_memory` caps.
- Swarm members can use every granted ability, including coding abilities when
  writing is explicitly enabled; write calls are serialized and scoped.
- The node output includes the leader's synthesis, member state, messages,
  reviews, shared memory and changed files.
- New run events: `swarm_start`, `swarm_plan`, `swarm_leader`, `swarm_round`,
  `swarm_agent_start`, `swarm_agent_step`, `swarm_agent_delta`,
  `swarm_agent_status`, `swarm_agent_tool`, `swarm_agent_end`, `swarm_message`,
  `swarm_memory`, `swarm_review` and `swarm_end`.

### 1.3

AML 1.3 is additive. Every 1.2, 1.1 and 1.0 document is valid and runs unchanged.

- New sandbox integrations: `sqlite.query`, `csv.read` and `git.inspect`.
  SQLite is read-only unless the ability config opts into writes; Git inspection
  is fixed to read-only status, diff and log commands.
- `llm` nodes accept `format: "json"`, which adds a parsed `json` field to the
  node output and fails the node when the reply contains no JSON value.
- The document accepts a top-level `vars` object with the initial flow variables.
- Fix: every `{{ ... }}` placeholder in a mixed string is now interpolated;
  previously the outer delimiters were captured as one expression when a string
  contained several placeholders.

### 1.2

AML 1.2 is additive. Every 1.1 and 1.0 document is valid and runs unchanged.

- New coding abilities: `file.edit`, `file.search`, `file.tree`, `file.delete`
  and `code.diagnostics`. File writes and edits are atomic; search and tree skip
  dependency folders (`.git`, `node_modules`, virtualenvs, caches).
- `agent` nodes accept `max_iterations` up to 40 and may return a `calls` array
  (several tools in order) or a `plan` object in addition to `tool` and `final`.
  Plans are emitted as the new `plan` run event.
- `shell.exec` accepts a timeout up to 120 seconds for test runs.
- The file sandbox may be pointed at a project folder by embedding callers (the
  Studio IDE does this), while the documented sandbox stays
  `agent-workspace/files` for AML flows.

### 1.1

AML 1.1 is additive. Every 1.0 document is valid and runs unchanged.

- New node types: `agent` (an LLM that chooses abilities), `human` (approval or
  input from a person) and `foreach` (a sub-agent per list item).
- `http.fetch` accepts `extract: "text"` and `max_chars` to return readable
  text instead of raw markup.
- Validation returns structured `issues` (`level`, `message`, `node`, `edge`,
  `field`) next to `errors` and `warnings`, and warns when a non-condition node
  has more than one outgoing edge.
- `and` / `or` follow Python semantics (short-circuit, return the deciding
  operand). In 1.0 they returned a boolean; a condition is unaffected because
  its result is converted to a boolean either way.
- Fix: abilities run by a flow now use `agent-workspace/files` and
  `agent-workspace/memory` as documented; 1.0 runs wrote one directory deeper.
