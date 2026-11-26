# IDE: agentic coding in Studio

The **IDE** workspace is a project folder plus a coding agent. It reuses the
model catalog, chat gateway and sandboxed abilities that power the rest of
Studio, and adds a multi-project explorer, a versioned editor, unified diffs, a
plan and one-click session revert.

The design goal is a safe autonomous loop: the agent can read, edit, search and
verify a project, every file-tool change is snapshotted first and every tool is
constrained to one workspace folder. Several projects can stay loaded, each with
its own tabs, conversations and live agent run.

---

## Opening a workspace

The toolbar offers three ways to start:

- **New** creates an empty project folder (`agent-workspace/ide/workspaces/<id>/`).
  New workspaces include a tiny Python sample with a passing test.
- **Open folder** opens an existing project **where it lives**. The agent reads
  and edits the original files; nothing is copied into `agent-workspace`. The
  folder is registered in `agent-workspace/ide/workspaces.json`, along with its
  IDE context rules, so no `.mariana-ide.json` is written into your project.
- **Open project → Manage recent projects… → Import a copy…** copies a folder
  into a managed workspace. The original folder is never modified. Dependency
  and runtime folders are skipped (`.git`, `node_modules`, virtualenvs,
  `trained-model/`, `models/`, `datasets/`, `exports/`, `backups/`,
  `agent-workspace/`). The copy is limited to 3000 files and 64 MB. A folder
  inside `agent-workspace/` is refused, because it holds chats, sessions,
  endpoints and settings that the agent must not read.

Removing an in-place project from **Manage recent projects…** only unregisters
it and removes its IDE conversations, runs and snapshots; the folder and its
files stay on disk. A managed copy can be deleted from the same manager; only
the copy inside `agent-workspace` is removed, never the project it came from.
If Studio's own `agent-workspace` is nested inside an opened project, the IDE
hides it and the agent cannot read or write it.

Workspace paths are always resolved and checked against the workspace root.
Absolute paths are accepted only when they resolve inside it; symlinks and
`..` escapes are rejected, as is the workspace manifest.

The editor opens UTF-8 files up to 2 MB and keeps the file's line endings when
saving. Saves send the SHA-256 version that was read with the file; if the file
changed on disk, the API answers `409` and the view offers **Reload from disk**
instead of silently overwriting.

Files are read in line ranges. The editor first asks for 4,000 lines; when the
answer says `truncated`, the tab opens read-only with a **Load full file**
action, so a generated or minified file cannot freeze the browser on click. Tab
switches reuse a cached editor instance; only the visible lines are highlighted
and numbered.

### Several projects at once

The toolbar opens projects; each becomes a project tab. Every project keeps its
own open files, active file, expanded folders, editor scroll positions,
conversations and live run. Switching projects never stops a run: the agent
panel, project tab and session list show which project is still working, and a
finished background run reports its result.

The **Open project** menu lists the workspaces on disk. **Manage recent
projects…** opens a manager that shows the projects open in this session and any
entries hidden from the menu; a collapsed **All projects on disk** section keeps
the full list. From here an entry can be shown or hidden again, the list can be
refreshed, and **Clean session** can be chosen. Cleaning removes every project
from the recent list together with its open tabs, unsaved drafts, pinned context
and per-project preferences; files, conversations and run history stay on disk.

The explorer is lazy. Opening a project lists one level, and expanding a folder
requests one level for that folder. Rows are virtualised, so a folder with
thousands of entries costs the same as a small one. **Unload folder** discards a
folder's cached children without hiding it from the agent.

### Files without the mouse

Right-click a file or folder for open, new file/folder, rename, delete, copy
path, download, pin to context, hide from the agent and close-tab actions.
`Ctrl+P` opens a file picker that starts with recent files, follows the arrow
keys and marks files that are already open or edited. `Ctrl+S` saves; `Ctrl+W`
closes a tab; `Ctrl+Shift+T` reopens the last closed tab; `Ctrl+PageUp` and
`Ctrl+PageDown` cycle tabs; `Ctrl+Shift+F` jumps to project search; and
`Ctrl+Shift+P` (or `Ctrl+K`) opens the command palette with every IDE action.
`Ctrl+Enter` sends a task. Typing `@` in the task box searches files and inserts
the chosen path as pinned context. The explorer is keyboard-navigable:
Up/Down moves between rows, Right expands or enters a folder, Left collapses or
moves to the parent, and Home/End jump to the ends. `Ctrl+Shift+E` focuses the
file filter.

The tab strip keeps its horizontal position across saves, scrolls the active tab
into view, and scrolls sideways with the wheel. The menu button on the view bar
lists every open file with its active/edited state and offers reopen, save-all
and close actions.

### Editor

The editor highlights syntax for Python, JavaScript/TypeScript, JSON, HTML/XML,
CSS/SCSS, Markdown, YAML, TOML, INI, shell, SQL, C/C++/C#/Java/Kotlin/Swift,
Go, Rust, Ruby, PHP, Lua, R, Julia, Haskell, Elixir, Erlang, Clojure,
Dockerfile, Makefile, diff, KML, GPX, GeoJSON, Mermaid and more. The language
follows the file extension
and can be changed from the language button in the status bar for the open tab;
the indentation button offers two, four or eight spaces or tabs. A file's own
indentation is detected until you choose a preference, **Detect from file**
applies it again later, and the indentation preference is remembered across
files.

Only the visible lines are highlighted. Lexer state is checkpointed every 64
lines, so scrolling a large file does not re-scan it from the top and editing a
line only invalidates the state below that line. Multiline comments, template
literals, triple-quoted strings, Markdown fences and `<script>`/`<style>` blocks
keep their state across lines.

A bar above the editor switches between **Code**, **Rendered** and **Hex** when
the file supports more than one view. Markdown renders in place (Mermaid fences
become flowcharts, sequence diagrams or pie charts), `.mmd`/`.mermaid` files
render directly, images and SVG show their pixels, and CSV/TSV files become a
table. Binary files stay read-only and open in Hex by default: offsets, bytes and
ASCII, plus the file's MIME type, size, modified time and image dimensions. The
metadata button on the same bar lists the complete details; code edits remain a
click away on the Code tab.

Breadcrumbs above the view bar show the workspace, folders and file. Click a
folder to reveal and expand it in the explorer, or use the actions at the end
of the row to copy the path or open the tab menu. `Ctrl+=`, `Ctrl+-` and
`Ctrl+0` scale the editor font between 12 and 22 pixels; the size is remembered
for every project. The selection button in the agent composer inserts the
current editor selection as a path-labelled, language-tagged fenced block, so
reviewing a specific function does not require copying code by hand. The
status bar keeps the file path and the cursor position in separate fields.
`Ctrl+Shift+?` opens the shortcut reference; every command is also searchable in
the palette.

Click a line number to select that line. The status bar reports the cursor,
the selected character and line count, the indentation and the file's line
endings. Matching brackets are marked, unmatched ones are flagged, and the
current line is tinted.

| Shortcut | Action |
|---|---|
| `Ctrl+F` / `Ctrl+H` | Find / replace in the file (case and regex toggles) |
| `Enter` / `Shift+Enter` | Next / previous match while the find bar is open |
| `Escape` | Close the find bar |
| `Ctrl+G` | Go to line |
| `Ctrl+L` | Select the current line |
| `Ctrl+/` | Toggle a line or block comment |
| `Ctrl+[` / `Ctrl+]` | Outdent / indent the selection |
| `Ctrl+Shift+D` | Duplicate the selected lines |
| `Ctrl+Shift+K` | Delete the selected lines |
| `Alt+Up` / `Alt+Down` | Move the selected lines |
| `Shift+Alt+Up` / `Shift+Alt+Down` | Duplicate the selected lines |
| `Ctrl+Enter` / `Ctrl+Shift+Enter` | Insert a line below / above |
| `Tab` / `Shift+Tab` | Indent / outdent, including a multi-line selection |
| `Ctrl+M` | Toggle whether `Tab` inserts indentation or moves focus to the status-bar controls |

Typing an opening bracket or quote inserts its pair and keeps the caret inside;
selecting text first wraps the selection. Backspace removes an empty pair, and
`Enter` between a matched pair opens an indented block. The editor's right-click
menu exposes clipboard actions, selection, find/replace, go-to-line, line
commands, comment toggling, case conversion and the language/indent menus.

### Outline and symbols

The **Outline** tab in the explorer lists the declarations of the active file:
functions and methods, classes, types, `impl` blocks, Markdown headings from
H1 to H6, and headings or `id`-carrying elements in HTML/XML. Indentation shows
nesting, and each row carries its kind and line number. Clicking a row switches
a preview back to **Code** when needed and jumps the caret to that line; the
symbol around the caret stays highlighted as you move through the file.

`Ctrl+Shift+O` opens **Go to symbol**, a searchable list of the same symbols.
Arrow keys, `Home`/`End` and `Enter` work as in quick open, and the picker
jumps straight to the chosen line. The filter box above the outline narrows the
list without touching the file. The palette also offers both actions.

Extraction is a fast, language-aware heuristic rather than a syntax tree: it
recognises declarations at the start of a line and deliberately skips comments,
Markdown fences and unknown languages. It is capped at 400 symbols and cached
per tab until the text or language changes, so typing with the panel open costs
at most one rescan per 250 ms and the outline is never scanned while the panel
is hidden.

### Window layout

The explorer, editor and agent are three resizable panes. Drag a divider to set
a width, double-click it (or press `Enter` on it) to fold the pane, and use the
arrow keys for 8-pixel steps. `Home` folds and `End` maximises; `Shift` makes the
arrow steps larger. The layout menu pins the panes side by side, stacks the
agent below the editor, returns to automatic stacking, hides either pane, or
toggles focus mode, which folds both side panes and restores them in one step.
`Ctrl+B` toggles the explorer and `Ctrl+Shift+B` the agent panel.

Sizes stick: a preferred width is remembered across reloads and restored when
space returns. There is no fixed maximum — a pane can be grown until another one
reaches its minimum, so the editor or the agent can take the whole window when
that is what the task needs. When the window is too narrow the panes shrink to
their readable minimum and then fold to a divider, which can be dragged or
tabbed back open. Dragging a divider past its minimum folds the pane, and the
divider left behind is the restore handle.

### Voice input and narration

The microphone button dictates into the task box at the cursor. Studio prefers
a **local** speech-to-text backend when one is installed; otherwise it uses the
browser's speech recogniser where available. Browser dictation may send audio to
the browser vendor's service, so install a local backend if the project must stay
on this machine:

```bash
pip install faster-whisper
# or point at whisper.cpp:
export MARIANA_WHISPER_CPP=/path/to/whisper-cli
export MARIANA_WHISPER_CPP_MODEL=/path/to/ggml-base.en.bin
```

The first local run may download the selected Whisper model. Recordings are
bounded, transcribed by a background job under `agent-workspace/voice`, and the
temporary audio is deleted when the job ends. The current run appears in the
Jobs panel and can be cancelled there.

Assistant replies have a **read aloud** button. The speaker toggle in the agent
header reads each completed reply automatically; both use the browser's local
speech synthesis and the preference is remembered.

### Context meter and token analysis

The meter in the composer footer is a compact ring, like an editor's context
indicator. It updates immediately as you type: the ring fills as the next
request approaches the selected model's context window and turns amber at 80%.
Hover it for the exact window and source, or click it for a section-by-section
breakdown: agent instructions, project instructions, project map, pinned context,
active file, open files, saved history and the current draft. The dialog reports
the model window, truncation, pinned-file caps, any over-budget warning and
whether history has been consolidated, and offers a refresh.

The ring also tracks a live run: the server publishes a `budget` event after
each tool round with the tokens actually used and the remaining context, so the
meter moves while the agent works instead of only when the run ends. Your draft
is substituted into that figure locally, so typing still updates the ring
between events.

Each completed run stores the provider's reported `prompt_tokens` and
`completion_tokens` on the conversation. The status bar shows the last run's total,
and the token dialog contrasts that exact count with the next-request estimate.

---

## Modes

The mode is enforced on the server, not only in the prompt.

| Mode | Tools | Use for |
|---|---|---|
| **Ask** | `file.read`, `file.inspect`, `file.list`, `file.tree`, `file.search`, `skill.read`, `code.diagnostics`, `csv.read`, `sqlite.query`, `git.inspect`, `http.fetch`, `news.headlines`, `browser.fetch`, `agent.swarm` | Reviews, explanations, current news and data inspection. Nothing is written; SQLite is read-only and a swarm is read-only. |
| **Edit** | Ask + `file.write`, `file.edit` | Normal implementation work. Every file-tool write is snapshotted, including swarm member writes. Shell changes are not snapshotted. |
| **Auto** | Edit + `file.delete`, `shell.exec` | Larger tasks and test runs. Shell still needs the server opt-in; SQLite remains read-only. |

`shell.exec` is removed from Auto's tool list until the server owner enables OS
access in **Models → Runtimes** (`allow_os_access`), or starts Studio with
`MARIANA_AGENT_ALLOW_OS=1` or `MARIANA_AGENT_ALLOW_SHELL=1`. The toolbar shows
**Shell off** in that case. OS access also lets `http.fetch` and
`news.headlines` reach private addresses; file writes still follow the workspace
sandbox and swarm write scopes.

`news.headlines` reads a public RSS/Atom feed (Google News by default, with
Bing News as the automatic alternate) and returns dated headlines with sources
and links. Use `region: "IN"` and `language: "en"` for India, and `query` for a
topic. `http.fetch` with `extract: "text"` reads one specific page.
`browser.fetch` renders a JavaScript page with Playwright or a Chromium binary;
it appears only when OS access is enabled.

---

## Tools

Most tools use the AML ability implementations, with the workspace as the file
root, so the same limits and sandbox checks apply to visual agents. `skill.read`
and `file.inspect` are IDE built-ins that only read skill files or file
metadata; they never write or execute anything.

The loop asks each model for one JSON action per turn. Models whose chat
template ignores that instruction and emit native markup — DeepSeek V4's
`<｜｜DSML｜｜ invoke …>` blocks, for example — are parsed into the same canonical
calls, including several parallel calls in one reply.

Read-only tool calls are retried once when the failure looks temporary
(timeout, connection reset, rate limit or a 5xx). Writes, deletes, shell
commands and swarms are never retried automatically, because a timed-out write
may already have landed. Deterministic failures return with a short recovery
hint, an unknown tool id is matched to the closest enabled id, and a third
identical call receives an explicit anti-loop hint. The retry count is visible
as `attempts` on the `tool_result` event and in the transcript step.

When tools are granted, Studio also sends native function definitions to
OpenAI-compatible endpoints and to the Anthropic Messages API. Kimi, Grok,
OpenAI, Qwen, Mistral, Groq, OpenRouter, Fireworks, Together and Claude can
therefore call tools directly; the aliases are provider-safe (`file_read`) and
are translated back to the Mariana tool id. If an endpoint rejects native
tools, Studio retries without them and the prompted JSON path takes over. A
malformed tool call is reported back to the model, which can correct it.

Arguments are normalised before a call runs, so provider- or model-specific
spellings still reach the right key: `file_path`, `filename` and `path` become
`path`, `q` becomes `query`, `offset` becomes `start_line` and `limit` becomes
`end_line`, a bare string becomes the tool's primary argument, and
punctuation/case variants of an id (`file_search`) resolve to the canonical
tool. A call that is still missing a required key is not executed: Studio
returns the reply to the model with one repair round that names the missing
keys and asks for the complete calls, so an empty input object costs a round
trip instead of a failed tool.

| Tool | What it does |
|---|---|
| `file.tree` | Project listing with sizes; dependency folders are skipped. |
| `file.read` | UTF-8 file contents in line ranges (200 lines or about 3,600 characters by default); the result reports `next_start_line` when more remain. |
| `file.inspect` | Metadata, MIME type, dimensions, hex dump and a short preview for any file, including images. |
| `file.write` | Atomic full write. |
| `file.edit` | Exact, unique block replacement. Ambiguous matches are refused with the occurrence count. |
| `file.search` | Literal or regular-expression search with file/line matches. |
| `skill.read` | Load one relevant `SKILL.md` automatically when its description matches the task; the agent follows the loaded instructions. |
| `file.delete` | Delete a file (snapshotted before the call). |
| `code.diagnostics` | Python syntax, JSON validity and optional `pyflakes` rules. Edits also run diagnostics automatically and return the issues with the tool result. |
| `csv.read` | Read CSV/TSV records with column names, delimiters and a row cap. |
| `sqlite.query` | One parameterized SQL statement against a workspace database; read-only in every IDE mode. |
| `git.inspect` | Read-only status, diff and recent log for a repository in the workspace (when git is installed). |
| `agent.swarm` | Run a leader-planned team of up to six agents. Members research or edit in parallel, message each other, share a run-scoped blackboard, review and revise each other, and return one synthesised answer. Ask mode is read-only. |
| `shell.exec` | Run a command in the workspace, up to 120 s, when enabled. |

The agent sees the tool schemas in its system prompt and replies with one JSON
object per turn:

```json
{"tool": "file.edit", "input": {"path": "main.py", "old": "a", "new": "b"}}
{"calls": [{"tool": "file.read", "input": {"path": "a.py"}}]}
{"plan": [{"text": "Update the parser", "status": "active"}]}
{"final": "Changed the parser and diagnostics pass."}
```

Several calls run in order for writes, deletes, shell commands and swarms.
Independent read-only calls in one round (`file.read`, `file.search`, `file.tree`,
`file.inspect`, diagnostics, data and web reads) run in parallel, up to four at a
time, and their results are returned in the order the model asked for them. A
plan updates the sticky checklist and emits a `plan` run event; it does not end
the loop. Repeated identical calls produce a hint in the observation so the model
changes approach. After a Python or JSON edit the service runs diagnostics on
that file and appends an `ok`/`issues` block to the observation, so a broken
change is visible without spending another tool round. The transcript renders
tool results as summaries, clickable paths, issues and syntax-coloured content
instead of one large JSON blob. A completed
tool or swarm card collapses to its header; expanding it shows the input,
result, diff and file action. Reasoning tokens stream into a collapsed
`Thought` row that keeps its newest four lines visible while the model thinks
and expands on demand. The transcript follows new output while the reader is at
the bottom; scrolling up pauses that and shows a **Jump to latest** button. A
swarm call renders as a live card: the leader's plan, one row per agent with its role,
score, version and streamed work, the review loop, the messages agents sent each
other, shared-memory keys and every file the swarm changed.

### Swarm coding

`agent.swarm` is a full swarm runtime, not just a fan-out helper. The parent
agent supplies tasks (or lets the leader choose them), a strategy
(`auto`, `map_reduce`, `peer_review`, `debate`, `pipeline`), a feedback round
count and a time budget. The leader then:

1. plans roles, tools, testable success criteria, a short plan per member and
   — when writing is enabled — disjoint `writes` scopes;
2. runs independent members in parallel, waits for `depends_on` waves, and
   otherwise keeps the pipeline sequential, with progress events for every model
   round, tool call, plan update and message;
3. assigns peer reviews against the success criteria, collects scores and
   concrete issues, and has each reviewed member revise its work;
4. synthesises one answer, reports the changed files and lists any criterion it
   could not verify.

In Edit and Auto the members use the same sandbox, snapshots, diagnostics and
shell opt-in as the parent, so every swarm file edit is visible in the session and
removed by **Revert edits**. In Ask the tool list contains no write tools and
the runtime also refuses writes, even if the model asks for them. Writes from
different members are serialized; a member with `writes` prefixes can only
touch those paths, a member with no scope is read-only, and `writes: ["*"]`
means unrestricted sandbox writes. Shell edits remain outside the snapshot
history, and a scoped member cannot use `shell.exec`.

The swarm has its own IPC and memory:

- Messages are addressed to `leader`, `all` or a known member, limited in size
  and count, stripped of control characters, and delivered only inside this
  run. They are data and never become tool calls.
- The shared blackboard is a keyed JSON store limited by entry count and value
  size; members see a compact digest before each model round.
- The runtime seeds the blackboard with `swarm.goal` and `swarm.plan`, and
  publishes each completed result as `result.<member-id>` so peers start from
  the team's knowledge instead of rediscovering it. Member and reviewer prompts
  also carry a short peer-results digest and every teammate's task.
- Context is a shared, run-wide pool. A member that needs more of a long file, a
  peer's result or an earlier turn requests it with the `context` action; the
  coordinator grants it while the pool has free space and releases the grant
  when that member's phase ends. Prompts are also capped by the model's real
  context window, so a long coding run does not fail on a context limit.
- Truncation is loss-free. Long tool results and compacted turns are kept behind
  `ctx-…` handles; the member reads a slice with `chars` and `offset`, and the
  reply names the next offset. A denied request reports the exact used/free
  characters and a smaller-request suggestion instead of a silent cut.
- If the board is full the automatic writes are skipped and the coding session
  continues; duplicate plan tasks are dropped with a warning instead of being
  run twice. A member's own plan persists across rounds and appears in the card;
  a member that finishes with open steps is asked once to complete or justify
  them, and an unresolved plan is reported as a warning.
- Read-only swarm tools retry one temporary failure; writes and shell calls are
  never retried. Deterministic failures carry a next-action hint, an unknown
  tool id suggests the closest granted id, and a third identical call is called
  out explicitly.
- The parent sees a compact result: the synthesis, each member's status and
  output, review scores, message and blackboard counts, and changed files.
  Full live state is in the run snapshot and the swarm progress events. The
  snapshot refreshes immediately on state changes and at most four times a
  second for high-frequency progress; the event stream is never throttled.
- The run is bounded even when a backend stops responding: the time budget is
  enforced on the coordinator side, unfinished members are finalised in the
  result, and late events from abandoned workers are dropped. An empty model
  reply or member final is retried once, a member that returns an empty final is
  asked once for its complete finding before the fallback is used, and a failed
  peer review is reported without discarding completed work. Repeated empty
  replies go to a tool-free wrap-up; coordinator prompts are never presented as
  the member's findings.
- Native tool schemas cover only the tools a member was granted, and review
  calls see read tools only. A revision prompt contains the member's previous
  answer (with its context handle), so feedback is applied to the work itself.
  Absolute or escaping write scopes are rejected before the member runs.
- A member that keeps calling tools until its budget ends gets one bounded
  extension (half its budget, up to 8 extra rounds, never above 24) when it has
  already collected observations, then is asked once for a final answer
  **without tools**. If that call also fails, its last observations are shared
  with the leader, which still synthesises a reply. A research turn therefore
  does not fail just because a model was slow to stop.
- The work deadline is earlier than the hard time budget: the last 10% (at
  least 5 s, at most 90 s) is reserved for wrap-up and synthesis. At the work
  deadline tools are refused, each member answers from the observations it has
  collected, later dependency waves and the feedback loop are skipped with a
  warning, and the leader still synthesises. If the reserve is exhausted too,
  the answer is assembled from the members' results and the run completes with
  `timed_out: true` instead of failing; only a run where no member produced
  output fails with the time-budget error. The result, snapshot and `swarm_end`
  event carry `timed_out`, and the card shows a **Time limit** badge.
- Retrieval is not a single feed. A member granted `news.headlines`,
  `http.fetch` or `browser.fetch` receives the whole retrieval family when those
  tools are enabled, so an empty or blocked feed has a fallback. A task that
  writes a file or a blackboard note receives `file.write` when it has a write
  scope (or OS access is enabled). `browser.fetch` uses Playwright or Chromium
  when present and otherwise falls back to reading the page over HTTP, reporting
  `engine: "http"` in its result.
- `news.headlines` returns compact titles, sources, dates and links; asking for
  a date with no items returns the newest items and a `date_matched: false`
  note, so the agent can state the mismatch instead of retrying forever.

### Context engineering

The system prompt contains the operating rules, the enabled tools, the current
task, project instructions from `AGENTS.md` or `.mariana-ide.md` (first 4,000
characters), a compact project map (file paths and sizes, about 6,000
characters), every available skill (ids, names and descriptions) with an
instruction to load the relevant ones automatically, the current plan
from earlier turns and every file this conversation already changed, and, when
one is open in the editor, the first 6,000 characters of that file. Up to four
other open files contribute 1,500 characters each. Pinned context adds up to 24
files with 4,000 characters each, capped at 12,000 characters total. The whole
prompt is capped at 23,000 characters so it survives the chat gateway's message
limit. The selected model's output budget is reserved as well: a request that
would leave no room for the reply is flagged in the token dialog and consolidated
before sending when possible. Tool observations are capped at 6,000 characters
per call and 20,000 characters per round; long fields are shortened in place so
the path, line range and truncation flags survive. A shortened `file.read`
reports the real `end_line`, adds `next_start_line`, and flags
`content_cut_midline` instead of silently skipping the rest of a very long line.
Stored conversation history is trimmed to the last 40 messages of 8,000
characters each.

Context is explicit:

- **Pin context** (`Pin context`, right-click, or `@path`) adds files and folders
to the next prompt. A folder expands to the first 24 files inside it.
- **Hidden from agent** (right-click, or the context dialog) removes a path from
the project map, `file.tree`, `file.list` and `file.search` results. Hidden
paths are saved in the workspace manifest, so they apply to every conversation.
- **Automatic consolidation** starts when the next request reaches 80% of the
model window **or when more than 32 messages would be sent**. The service
replaces older turns with a compact recap (user tasks, plans, tool calls and
final answers), keeps the newest eight messages verbatim, records how much it
freed and emits a `context_consolidated` event. The system prompt also carries
the current plan and this conversation's changed files, so paths and next steps
survive consolidation. The run continues without a separate summarisation round,
and the token dialog can consolidate a saved conversation on demand.
- The task is sent once as the user message; the system prompt carries only a
  short anchor, so a long request is not duplicated on every turn. A truncated
  active-file or pinned-file preview is labelled and points to `file.read`, so
  the model never edits from a silently shortened view.
- The context dialog shows the estimated token cost before a run. Its window is
  the selected model's latest known maximum: the larger of catalog metadata
  (GGUF, Transformers config, Mariana checkpoints) and a live probe of a saved
  endpoint, local OpenAI-compatible server or Ollama model. The probe is cached
  for five minutes; **Refresh estimate** forces a fresh one. With no model
  selected it resolves the same default chat model a run would use. If no
  source reports a window, 8,192 is assumed and the dialog says so.
- Session-level pins are stored with the conversation; clearing them explicitly
  also clears a workspace default for that conversation.

A run has a step budget (the IDE sends 40 by default, the request default is 24
and the ceiling is 128) and a time budget (the IDE requests 3,600 s; the request
default is 900 s, up to 3,600 s). When a run reaches its step budget and the last
few rounds produced distinct successful work, the service extends it by 24 steps,
up to three times. A stalled loop (the same call over and over) is not extended.
When either budget really runs out the run stops cleanly, keeps its transcript,
and tells the user to ask for the next step. A reasoning model that spends its
whole output budget on hidden thinking returns no visible text; the service
retries that turn once with double the output tokens and a lower reasoning
effort, and if it still has nothing to show it fails with an actionable error
instead of recording an empty answer. Malformed JSON actions and empty final
summaries are sent back to the model for correction instead of silently ending
the run. A successful run marks every remaining plan step done and publishes the
updated plan; an exhausted, timed-out or failed run leaves the remaining steps
open so the next turn shows what is still left.

---

## Snapshots, diffs and revert

Before the first write to a path in a session, the IDE stores the original bytes
in `agent-workspace/ide/snapshots/<session-id>/`. A unified diff is computed
after each tool call and attached to the tool result, so the UI can show
`+added / −removed` without another request.

**Revert edits** restores every file the session changed to its pre-session
content and deletes files the session created. Reverting is idempotent; a later
edit after a revert starts a new snapshot baseline. If a file was edited after
the agent's last write (by you, or by another session), the revert skips that
file and reports it, so newer work is never silently discarded.

---

## Sessions and runs

A session is one conversation with the agent. Each user message starts a run in
that session; the previous model messages are reused, so follow-ups keep context.
A session records the canonical transcript, model messages, plan, pinned context
and change manifest in `agent-workspace/ide/sessions/<session-id>.json`.

Conversations support the full lifecycle from the agent panel: create, rename,
pin to the top, duplicate (a fork that keeps the transcript but starts with no
change snapshots), copy the whole conversation as Markdown to the clipboard,
export it as a `.md` file and delete. Copy and export include tool calls, tool
output and errors, so a transcript pasted into an issue keeps the evidence.
Only one run per workspace is active at a time; different projects run
independently. Rename and pin use `PATCH /api/ide/sessions/{id}`; duplicate uses
`POST /api/ide/sessions/{id}/duplicate`.

The model picker lists every catalog chat model, including saved endpoints. The
**providers** button beside it opens the shared provider flow: add an
OpenAI-compatible, Ollama or Anthropic endpoint, manage saved endpoints, or
refresh the catalog. The same dialog is used by Chat, Agents and the Models view,
so an external API added anywhere appears in every model picker.

A run streams server-sent events:

| Event | Meaning |
|---|---|
| `run_started` | Model, mode and the granted tools. |
| `assistant_start`, `assistant_delta`, `reasoning_delta`, `assistant_end` | Model output as it arrives. |
| `tool_call`, `tool_result` | One tool, including `change` for writes. |
| `plan` | The normalized plan list. |
| `file_changed` | A file was written; payload has the diff and counts. |
| `context_consolidated` | Older turns became a recap; payload has before/after token counts, percent and cause. |
| `budget` | Live context usage after a tool round: `used_tokens`, `context_window`, `percent` and `remaining_tokens`, reported in the run's token `unit`. |
| `swarm_start`, `swarm_plan`, `swarm_leader`, `swarm_round`, `swarm_agent_start`, `swarm_agent_step`, `swarm_agent_delta`, `swarm_agent_status`, `swarm_agent_tool`, `swarm_agent_end`, `swarm_message`, `swarm_memory`, `swarm_review`, `swarm_context`, `swarm_end` | Swarm progress; every event carries `swarm`, `round`, `phase` and, when it concerns one member, `agent`. `swarm_context` carries the grant or denial and the current pool snapshot. |
| `log` | Status or a recoverable warning. |
| `run_completed`, `run_failed`, `run_cancelled` | Terminal state; `output` is the final answer. |

A run's event log ends at its `run_<status>` event. Progress reported by a
worker after completion or cancellation is ignored, so snapshots, counters and
`?since=` replays cannot grow past the terminal event.

`assistant_end` carries the round's `reasoning`, `finish_reason` and the
`empty`, `truncated` and `retrying` flags, so a client can distinguish a
reasoning-only placeholder from a real answer.

The session is checkpointed every five seconds and immediately after a write, a
plan update or a consolidation, so a crash keeps the model messages and
transcript up to the last checkpoint. If the page reloads during a run, the
session reports `active_run_id` and `last_event_seq`; the view renders the
checkpointed transcript and reconnects with `?since=<last_event_seq>`, so tool
cards are neither lost nor duplicated.

---

## HTTP API

All routes are loopback-only and follow Studio's error contract
(`{"error", "code"}`).

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/api/ide` | Discovery: workspaces, modes, tools, limits, shell availability. |
| `GET` | `/api/ide/spec` | This document as markdown. |
| `GET` | `/api/ide/workspaces` | List workspaces. |
| `POST` | `/api/ide/workspaces` | Create (`name`, optional `source`, `template`). |
| `GET` | `/api/ide/workspaces/{id}` | One workspace with file, byte and language counts. |
| `DELETE` | `/api/ide/workspaces/{id}` | Delete a workspace, its sessions, runs and snapshots. |
| `GET` | `/api/ide/workspaces/{id}/tree` | Project tree; `path` and `depth=1` list one folder lazily. |
| `GET` | `/api/ide/workspaces/{id}/files?query=&limit=` | File and folder name lookup for filters and @-mentions. |
| `GET` | `/api/ide/workspaces/{id}/file?path=` | Read a file and its version; `start` and `limit` read a line range. |
| `GET` | `/api/ide/workspaces/{id}/preview?path=` | Metadata plus a source, image or hex preview for any file, including binary; `start`/`limit` apply to text. |
| `GET` | `/api/ide/workspaces/{id}/skills` | Workspace and user skills the coding agent can load. |
| `GET` | `/api/ide/workspaces/{id}/skills/{name}` | One skill with its editable instructions. |
| `POST` | `/api/ide/workspaces/{id}/skills` | Create `{id?, name, description?, instructions, scope?}`. |
| `PUT` | `/api/ide/workspaces/{id}/skills/{name}` | Update `{name?, description?, instructions}`. |
| `DELETE` | `/api/ide/workspaces/{id}/skills/{name}` | Delete one skill directory. |
| `PUT` | `/api/ide/workspaces/{id}/file` | Save `{path, content, version, force?}`. |
| `DELETE` | `/api/ide/workspaces/{id}/file?path=` | Delete a file. |
| `POST` | `/api/ide/workspaces/{id}/folder` | Create `{path}`. |
| `DELETE` | `/api/ide/workspaces/{id}/folder?path=` | Delete a folder and its contents. |
| `POST` | `/api/ide/workspaces/{id}/move` | Rename or move `{from, to}`. |
| `GET` | `/api/ide/workspaces/{id}/settings` | Hidden paths and default pinned context. |
| `PUT` | `/api/ide/workspaces/{id}/settings` | Replace `{excluded, context}` rules. |
| `POST` | `/api/ide/workspaces/{id}/context` | Preview pinned files and the full next-request budget; accepts `context`, `active_file`, `context_files`, `model`, `mode`, `session_id`, `message` and `refresh` (force a fresh model-window lookup). |
| `GET` | `/api/ide/voice` | Local speech backend, install hint, recording limits and privacy note. |
| `POST` | `/api/ide/voice/transcribe` | Start a transcription job for `{audio, format, language?, model?}`; follow `GET /api/jobs/{id}`. |
| `POST` | `/api/ide/workspaces/{id}/search` | `{query, regex?, case_sensitive?, include?, max_results?}`. |
| `POST` | `/api/ide/workspaces/{id}/diagnostics` | `{path?}` project checks. |
| `GET` | `/api/ide/workspaces/{id}/sessions` | Conversation summaries, pinned first. |
| `POST` | `/api/ide/workspaces/{id}/runs` | Start a turn: `{message, session_id?, mode?, model?, active_file?, context?, max_steps?}`. |
| `GET` | `/api/ide/runs/{run_id}` | Run snapshot and events after `since`. |
| `GET` | `/api/ide/runs/{run_id}/events` | Server-sent events. |
| `POST` | `/api/ide/runs/{run_id}/cancel` | Stop after the current tool. |
| `GET` | `/api/ide/sessions/{session_id}` | Transcript, plan, context and changes. |
| `PATCH` | `/api/ide/sessions/{session_id}` | Rename `{title}`, pin `{pinned}` or update `{context}`. |
| `POST` | `/api/ide/sessions/{session_id}/duplicate` | Fork a conversation. |
| `POST` | `/api/ide/sessions/{session_id}/consolidate` | Replace older turns with a recap; accepts optional `{model}`. |
| `DELETE` | `/api/ide/sessions/{session_id}` | Delete a session, its saved runs and snapshots. |
| `POST` | `/api/ide/sessions/{session_id}/revert` | Restore pre-session files. |

Example:

```bash
BASE=http://127.0.0.1:8765
WS=$(curl -s -X POST $BASE/api/ide/workspaces \
  -H 'Content-Type: application/json' \
  -d '{"name":"demo","template":"python"}' | jq -r .workspace.id)

RUN=$(curl -s -X POST $BASE/api/ide/workspaces/$WS/runs \
  -H 'Content-Type: application/json' \
  -d '{"message":"Add a farewell function and a test","mode":"edit"}' | jq -r .run.id)

curl -sN $BASE/api/ide/runs/$RUN/events
```

---

## Security

- Workspaces and snapshots live under `agent-workspace/ide/`; no route writes
  outside that tree.
- The agent cannot reach the manifest, ignored dependency folders or paths
  outside its workspace.
- Hidden paths are a context rule, not a security boundary: the explorer, map,
  search and agent listings omit them, and a pinned context never reads them,
  but an explicit `file.read` for that path still works for the person driving
  the agent.
- Shell access remains a process-wide opt-in (`MARIANA_AGENT_ALLOW_SHELL=1`);
  the IDE cannot grant it per session.
- A swarm is read-only in Ask mode. In Edit and Auto its member tools come from
  the parent mode, and file-tool writes are serialized and snapshotted. The
  runtime enforces each member's write scope; shell access is process-wide and
  shell edits are outside the snapshot history. IPC is run-scoped with fixed
  recipients, size and quota limits; messages never become tool calls.
- `sqlite.query` is always read-only in the IDE: a model cannot enable writes,
  and `ATTACH` is denied. Use the agent designer's ability config when a visual
  agent needs a writable database.
- `git.inspect` runs fixed read-only Git commands with external diffs, text
  conversion, pager and fsmonitor disabled. It does not accept a shell string.
- Skills are `SKILL.md` instruction packs discovered in the workspace's
  `.agents/skills` and the user's `~/.agents/skills`. The coding agent chooses
  which relevant skills to load; there is no manual apply step. Every skill's id,
  name and description enters the prompt, and the model calls `skill.read` for
  the ones that fit. The skills popup creates, edits and deletes workspace or
  user skills through explicit CRUD routes. Only names and descriptions enter
  the prompt until the model calls `skill.read`; nothing from a skill is
  executed, and symlinked or oversized files are skipped.
- The preview route resolves workspace paths with the same sandbox as the
  editor. Binary files are read-only, image data URLs are size-capped, and HTML
  or SVG is shown as an image rather than executed as a document.
- The agent has no HTTP tool by default. Give it a URL-reading ability by
  building an AML agent if a task needs the web, and review the policy first.
- Importing a folder copies its files into the workspace, including dotfiles
  such as `.env`; the agent can read every text file there. Remove secrets before
  importing. `AGENTS.md` and `.mariana-ide.md` are fed to the agent as project
  instructions, so review them after importing an unfamiliar project.
- Export a file or copy a session before deleting a workspace; deletion removes
  sessions, saved runs and snapshots too.

---

## Extending the loop

To add a coding tool:

1. Add the ability to `ABILITY_TYPES` in `mariana/agents/aml.py` with typed
   `params`, a danger flag and a default config.
2. Implement the runner method in `mariana/agents/tools.py` (the dispatcher maps
   `a.b` to `a_b`). Keep paths inside `self.files`, enforce a size limit and
   write atomically.
3. Add the tool id to the relevant mode in `MODE_INFO` (`mariana/studio/ide.py`).
   Snapshot before write-like tools in `_WRITE_TOOLS`.
4. Update `mariana/agents/specs/AML.md` and this document.
5. Add a regression test in `tests/test_studio_ide.py` (and the AML tool tests)
   before using it on a real project.
