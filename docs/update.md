# Updates

Mariana compares the release written in `VERSION.txt` with the latest
[GitHub release](https://github.com/CYFARE/Mariana-Studio/releases) and updates
a git checkout in place. There is no package server and no compiled payload:
`git` is the only transport.

## Command line

```bash
python mariana.py update          # check, then fast-forward the checkout
python mariana.py update --check  # report without changing any file
python -m mariana update          # the same command
```

The command runs before PyTorch is imported, so it still works when the model
runtime is missing or broken. It prints the local version, the latest release,
and the release URL before touching the checkout. When an update is installed
it asks you to restart:

```bash
python mariana.py studio
```

Version comparison is numeric (`2.10.0` is newer than `2.9.9`), and a leading
`v` in a tag is ignored.

## Studio

Studio checks once after boot on a daemon thread and shows a non-modal panel
when a release is available. The panel follows the git job and asks for a
restart when it completes. Automatic checks can be turned off in
**Models → Runtimes & settings**; **Check for updates** next to that setting
runs a manual check either way.

The HTTP surface is small:

| Route | Purpose |
| --- | --- |
| `GET /api/updates` | Cached status (local version, release, git facts, latest job); starts a background check when none exists. |
| `POST /api/updates/check` | Starts a fresh background check. |
| `POST /api/updates/apply` | Starts the update job (`202`), visible in the job centre. |

A completed update reports `restart_required: true` when the checkout changed;
the running process keeps the code it imported until it is restarted.

## What the update does

1. Refuses a non-git copy, a missing `git`, a detached `HEAD`, or uncommitted
   changes to tracked files.
2. `git fetch --tags --prune origin`
3. `git merge --ff-only origin/<branch>`

Only fast-forwards are allowed, so local commits are never merged or rewritten.
Untracked files are ignored because a fast-forward aborts rather than
overwriting them. If `git` asks for credentials the command fails instead of
waiting for input (`GIT_TERMINAL_PROMPT=0`).

If the checkout has diverged, the usual recovery is:

```bash
git status
git stash          # or commit the work
git pull --ff-only
git stash pop
```

Then run `python mariana.py update` again.

## Network and privacy

The check contacts `api.github.com` only. It sends no workspace data, model
names, or telemetry. Set `check_for_updates` to `false` in Studio settings to
disable the automatic check; the `update` command remains available.
