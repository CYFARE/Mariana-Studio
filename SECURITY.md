# Security policy

Mariana Studio is a local automation server. It can read files, call remote
model endpoints, run scheduled agents, and — only when the operator opts in —
execute shell commands. Security reports are taken seriously.

## Supported versions

Security fixes are applied to the current `main` branch and the latest release
in the 2.3.x line. If you are running an older checkout, reproduce the issue on
`main` before reporting it.

## Reporting a vulnerability

Please do **not** open a public issue with exploit details.

Use GitHub's private vulnerability reporting for the repository:

1. Open <https://github.com/CYFARE/Mariana>.
2. Choose **Security** → **Report a vulnerability**.
3. Include the affected version or commit, the configuration, and a minimal
   reproduction.

If private reporting is unavailable, contact the maintainers through the
repository without including a working exploit in public.

Useful reports include:

- a path that escapes the `agent-workspace/files` sandbox;
- a way to reach a private, loopback, link-local, or cloud-metadata address
  through `http.fetch`, `news.headlines`, redirects, or DNS rebinding when the
  ability did not opt in;
- a way for a node input, a model reply, or an imported AML document to widen an
  ability grant, a write scope, or the server-wide OS access permission;
- a cross-site request or foreign `Host`/`Origin` request that the loopback
  server accepts;
- a SQLite attach/write escape, a Git argument injection, or a shell injection;
- a leak of an endpoint API key or another secret in an API response, log, or
  checkpoint;
- a dependency-install path that accepts a package name or command from a
  request.

## Threat model and safe defaults

- Studio binds to `127.0.0.1` by default. The server rejects foreign `Host` and
  `Origin` headers and state-changing requests without a JSON content type.
- There is no authentication. If remote access is unavoidable, put Studio behind
  a reverse proxy with authentication and do not expose it to untrusted
  networks.
- File abilities resolve inside `agent-workspace/files`. Absolute paths are
  allowed only when they resolve inside that directory; traversal and symlink
  escapes are rejected.
- HTTP tools block localhost, private, link-local, and reserved addresses by
  default. Redirect targets and the connected peer are validated. Cloud
  metadata addresses are always blocked.
- `sqlite.query` is read-only unless the ability config sets `write: true`;
  `ATTACH` is denied and queries have deadlines.
- `git.inspect` runs fixed, read-only status, diff, and log arguments.
- `shell.exec`, private-network HTTP, and `browser.fetch` require a server-wide
  opt-in (`allow_os_access`, `MARIANA_AGENT_ALLOW_OS=1`, or
  `MARIANA_AGENT_ALLOW_SHELL=1`). A node input or an imported document cannot
  widen it.
- Endpoint API keys are write-only at the API boundary: responses expose
  `has_api_key`, never the key.
- Swarm IPC is run-scoped. Messages are data, never tool calls, and write-like
  member calls require both `swarm.allow_write` and the node's `write` flag.
- Dry-run mode mocks LLM calls and skips HTTP, file writes, SQLite writes, shell
  commands, memory writes, and delays.

## Out of scope

- An attacker who already has local access to the user account, the repository,
  or the `agent-workspace/` directory.
- A user intentionally enabling OS access or shell execution and then running an
  untrusted AML document. AML is executable automation; only import agents you
  trust.
- Denial of service caused by a deliberately huge local training run, a huge
  download, or a model endpoint the operator configured.
- Vulnerabilities in optional third-party runtimes (llama.cpp, Ollama, ONNX
  Runtime, Playwright, or a remote model provider) that are not reachable
  through Mariana's own boundary.

## Publishing a copy of the workspace

Git ignores runtime data, but a zip or a fresh `git init` from the working tree
can still include it. Before publishing a copy:

- remove `venv/`, `.pytest_cache/`, and `__pycache__/`;
- remove or sanitize `agent-workspace/` (chats, endpoints, settings, logs, and
  sandbox files can contain local paths or saved API keys);
- remove the contents of `trained-model/` and `backups/`;
- search the working tree and the Git history for absolute paths, usernames,
  API keys, tokens, and hostnames.

The repository's checkpoints and corpora are never required to build or test
the project.

## Disclosure

We will acknowledge a report, investigate it, and coordinate a fix and a
disclosure date with the reporter. Credit is given on request.
