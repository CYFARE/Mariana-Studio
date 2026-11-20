# Mariana engineering standard

The single reference for anyone — human or LLM agent — changing this repository.
Read it before writing code. `tests/test_standards.py` enforces the checkable parts.

## 1. Map

| Path | Owns | Must not |
| --- | --- | --- |
| `mariana/*.py` | Core DRV-LM: config, model, train, generate, latent, saint, cli | import `studio` or `agents` |
| `mariana/ncm/` | Next-Concept Modeling | import `studio` |
| `mariana/export/` | GGUF / ONNX / safetensors converters | import `studio` |
| `mariana/agents/` | AML language, flow runtime, swarm coordinator and IPC, abilities, store, scheduler | speak HTTP |
| `mariana/studio/` | Mariana Studio backend (see §3) | put logic in `api/` or `server.py` |
| `mariana/studio/web/` | Studio frontend: `index.html`, `views/`, `css/`, `js/` | load anything off-origin |
| `scripts/` | Standalone utilities runnable as `python scripts/<name>.py` | be imported by the package |
| `tests/` | pytest suite, one file per source path (see §5) | touch the network or need a GPU |
| `docs/` | Design notes and runbooks, kebab-case `.md` | duplicate the README |
| `docs/whitepapers/` | One technical whitepaper per technique plus its index | replace the runbooks |

Runtime data is git-ignored: `trained-model/` (checkpoints, prepared corpora),
`models/` (downloaded and fine-tuned third-party models), `datasets/` (fine-tuning
datasets downloaded from hubs), `exports/`, `backups/`,
`agent-workspace/` (agents, runs, chats, endpoints, settings). The repository
ships without checkpoints, corpora, or model weights; these directories are
created on demand and their generated contents can be deleted at any time.

## 2. Naming

| Thing | Rule | Example |
| --- | --- | --- |
| Python module / package | `snake_case` noun, no layer prefixes | `corpus.py`, not `gui_corpus.py` |
| Class / exception | `PascalCase`; exceptions end in `Error` | `ModelGateway`, `ModelError` |
| Function / variable | `snake_case`; functions start with a verb | `inspect_folder` |
| Module-private helper | leading underscore | `_sandbox_path` |
| Constant | `UPPER_SNAKE_CASE` at module top | `MAX_HTTP_BYTES` |
| Test file | `tests/test_<source path with _>.py` | `mariana/studio/corpus.py` → `tests/test_studio_corpus.py` |
| Test function | `test_<behaviour in words>` | `test_private_urls_are_blocked` |
| CLI flag | `--kebab-case` | `--freeze-layers` |
| HTTP route | `/api/<resource>[/{id}[/<action>]]`, plural nouns | `/api/agents/{id}/runs` |
| JSON key (requests, responses, files) | `snake_case` | `"max_tokens"` |
| Web asset | `kebab-case` | `training-viz.js` |
| JavaScript | `camelCase` functions, `PascalCase` classes, one `Studio.<name>` namespace per file | `Studio.chat` |
| CSS class / custom property | `kebab-case`; state as a second class | `.model-item.active`, `--accent-soft` |
| DOM id | `<view>-<thing>` | `chat-send` |
| Docs | `kebab-case.md` | `docs/fast-training.md` |
| Whitepaper | `kebab-case.md` in `docs/whitepapers/` | `docs/whitepapers/sprint-drv.md` |

Use the vocabulary already in the code: *checkpoint* (a Mariana `.pt`), *model*
(any catalog entry), *endpoint* (a remote model), *job* (background work),
*run* (one agent execution), *ability* (an agent tool), *workspace* (the view of
the UI, or `agent-workspace/` on disk).

## 3. Backend (`mariana/studio/`)

**Layers — dependencies point downward only.**

1. `server.py`, `http.py` — transport: sockets, static files, security headers, routing.
2. `api/<resource>.py` — handlers: read the request, call one service, return a dict.
3. Services — `training.py`, `jobs.py`, `corpus.py`, `models/`, `finetune/`, `chats.py`,
   `ide.py`, `deploy.py`, `dependencies.py`, and `mariana.agents`. Plain Python, no HTTP types, unit-testable.

**Add an endpoint**

```python
# mariana/studio/api/widgets.py
from ..errors import NotFoundError
from ..http import Router

router = Router()

@router.get("/api/widgets/{id}")
def read_widget(app, request):
    widget = app.widgets.get(request.params["id"])      # service call
    if widget is None:
        raise NotFoundError("Widget not found; refresh the list")
    return {"widget": widget}                            # dict -> 200 JSON
```

Register the router in `api/__init__.py`, add a test in `tests/test_studio_api_widgets.py`.

**Rules**

- **Errors.** Raise `StudioError` subclasses from `studio/errors.py` (`BadRequestError` 400,
  `NotFoundError` 404, `ConflictError` 409). Services may raise `ValueError`; the server
  maps it to 400. Every message says what is wrong *and* what to do next. The wire shape is
  always `{"error": "<message>", "code": "<snake_case>"}`. Never leak a traceback.
- **Validation.** Validate at the boundary with `studio.validation.number()` and explicit
  allow-lists. Reject unknown enum values; never pass request text to a shell.
- **Long work is a job.** A request returns in well under a second. GPU work runs in a
  subprocess, I/O in a daemon thread, both reported through `jobs.JobRegistry`. Trainers
  log `step <n>/<total> | total <loss> | <key> <value> | …` so every progress view works.
- **Optional dependencies.** Import them inside the function that needs them. A missing
  backend yields a catalog entry with `"available": false` and a `"reason"` containing the
  exact `pip install …` line — never an import error at startup. Tests use
  `pytest.importorskip`.
- **Filesystem.** Write only under the runtime directories in §1. Resolve, then check
  `relative_to()` before touching a user-supplied path. Write atomically (temp file +
  `replace`). Never overwrite a base model; back up before replacing a checkpoint.
- **Concurrency.** One lock per manager, held only around state changes — no network or
  disk I/O under it. Threads are daemons with names. Anything that starts a thread or a
  process has `close()`, called from `Studio.close()`.
- **Security.** Bind to loopback. The server rejects foreign `Host`/`Origin` headers and
  non-JSON writes. Secrets are write-only: return `has_api_key`, never the key. Agent file
  access stays in `agent-workspace/files`; private-network HTTP and the shell are opt-in.
- **Dependencies.** Only `dependencies.py` may run `pip`; it installs the fixed files in
  the repository root with `sys.executable -m pip`, one job at a time, and never accepts
  package names or commands from a request. The browser can hide the installer but the
  job continues; make automatic installation opt-outable through
  `auto_install_dependencies`.
- **Streaming.** Server-sent events: `data: <json>\n\n`, each object has a `type`.
- **Compatibility.** `AML`/`AMP` are public specs: changes are additive, bump the minor
  version, and update `mariana/agents/specs/*.md` in the same change.
- **Style.** `from __future__ import annotations`, type hints on public functions, a module
  docstring stating the module's single responsibility, standard library first. Comments
  explain *why*. No new required dependency without discussion; the web stack is stdlib only.

## 4. Frontend (`mariana/studio/web/`)

- No build step, no CDN, no framework. The page ships with a strict
  `Content-Security-Policy`: **no inline `<script>`, no `style=""`, no `on*=""` attributes.**
- One file per view in `js/views/`, registered with `Studio.views.register(name, {...})`.
  Shared behaviour lives in focused modules: `js/dom.js` (elements and selectors),
  `js/events.js` (event bus and shortcuts), `js/format.js`, `js/storage.js`,
  `js/util.js`, `js/feedback.js` (toasts, dialogs, busy actions), `js/api.js`
  (`Studio.api`, streams and feeds), `js/catalog.js` (`Studio.models`),
  `js/router.js` (`Studio.views`), `js/ui.js` (primitives), `js/overlay.js`
  (floating panels, menus, modals, path picker), `js/components.js` (composed
  cards, result cards, list rows, section headings), `js/form.js` (declarative
  form fields and value collection), `js/highlight.js` (the stateful, line-based
  syntax tokenizer), `js/editor.js` (the code editor used by the IDE),
  `js/split.js` (resizable, collapsible and auto-stacking panes),
  `js/tokens.js` (the context meter's client estimate), `js/voice.js`
  (dictation, transcription jobs and narration) and `js/dependencies.js` (the
  startup requirement installer). Do not re-implement them per view. Use
  `Studio.components` for cards and rows, `Studio.ui.render()` to clear and fill
  a container, and `Studio.form.field()` for settings instead of assembling
  inputs by hand.
- Build DOM with `Studio.h()` / `textContent`. `innerHTML` is allowed only for output of
  `Studio.markdown.render()`, which escapes first.
- Colours, spacing, radii and type sizes come from `css/tokens.css`. No raw hex values in
  component or view CSS. Body text is never below 12px.
- Every control is reachable by keyboard, has a visible focus ring and an accessible name.
  Destructive actions go through `Studio.confirm()`; never `window.confirm`.
- Every id passed to `$()` exists exactly once in the assembled page (test-enforced).

## 5. Tests

- pytest only: `python -m pytest`. No `sys.path` edits, no `__main__` runners.
- One test file per source path (§2). Shared fixtures live in `tests/conftest.py`;
  reusable non-fixture helpers live in `tests/helpers.py`.
- Use `tmp_path`. Mock remote services with a server on `127.0.0.1:0`. Use `device="cpu"`
  and tiny configs; a test should finish in seconds. Poll with `wait_for`, never bare `sleep`.
- Tests that write under the repository use a unique name and clean up in `finally`.
- A bug fix lands with the regression test that would have caught it.

## 6. Definition of done

```bash
python -m pytest -q                 # everything green
python -m pyflakes mariana tests    # no unused imports or undefined names
python mariana.py studio --no-browser   # starts; exercise the changed view by hand
```

Then: update `README.md` / `docs/` if behaviour changed, keep generated files out of git,
and describe what was verified — and what was not — in the hand-off.
