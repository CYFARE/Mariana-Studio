"""Check GitHub releases and update a Mariana Studio git checkout.

The module is deliberately torch-free: ``python mariana.py update`` has to work
even when the model runtime is missing or broken.  Studio reuses the same
functions from a background job, so nothing here prints unless the CLI asks it
to.

The version on disk lives in ``VERSION.txt`` (the tag of the release); the
version a running process loaded is :data:`mariana.__version__`.  Comparing
both tells Studio whether a restart is still pending.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

REPOSITORY = "CYFARE/Mariana-Studio"
RELEASES_URL = f"https://github.com/{REPOSITORY}/releases"
LATEST_RELEASE_URL = f"https://api.github.com/repos/{REPOSITORY}/releases/latest"
VERSION_FILE = "VERSION.txt"
USER_AGENT = "Mariana-Studio-updater"
NETWORK_TIMEOUT = 10.0
MAX_DETAIL = 1200
_VERSION = re.compile(r"\d+(?:\.\d+)*")

LineCallback = Callable[[str], None]
Runner = Callable[[list[str], LineCallback | None], tuple[int, str]]


class UpdateError(RuntimeError):
    """A check or update could not finish; the message says what to do next."""


# ---------------------------------------------------------------------------
# Versions
# ---------------------------------------------------------------------------
def repository_root() -> Path:
    """The checkout that contains this package."""
    return Path(__file__).resolve().parents[1]


def version_tuple(value: object) -> tuple[int, ...]:
    """The numeric parts of a version or tag (``v2.4.0`` -> ``(2, 4, 0)``)."""
    match = _VERSION.search(str(value or ""))
    if not match:
        return ()
    return tuple(int(part) for part in match.group(0).split("."))


def is_newer(latest: object, local: object) -> bool:
    """True when ``latest`` names a later version than ``local``."""
    left, right = version_tuple(latest), version_tuple(local)
    if not left or not right:
        return False
    width = max(len(left), len(right))
    return left + (0,) * (width - len(left)) > right + (0,) * (width - len(right))


def local_version(root: Path | str | None = None) -> str:
    """The release stored in ``VERSION.txt``, or the packaged version."""
    path = Path(root or repository_root()) / VERSION_FILE
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        text = ""
    if text:
        return text.splitlines()[0].strip()
    from . import __version__

    return str(__version__)


# ---------------------------------------------------------------------------
# GitHub releases
# ---------------------------------------------------------------------------
def fetch_latest_release(timeout: float = NETWORK_TIMEOUT) -> dict:
    """Read the newest published (non-draft, non-prerelease) GitHub release."""
    request = urllib.request.Request(
        LATEST_RELEASE_URL,
        headers={"Accept": "application/vnd.github+json", "User-Agent": USER_AGENT},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read(2_000_000).decode("utf-8"))
    except urllib.error.HTTPError as error:
        if error.code == 404:
            raise UpdateError(
                f"{REPOSITORY} has no published GitHub release yet; nothing to compare against."
            ) from None
        if error.code in {403, 429}:
            raise UpdateError(
                "GitHub rate-limited the update check; wait a few minutes and try again."
            ) from None
        raise UpdateError(
            f"GitHub returned HTTP {error.code} while checking the latest release; try again later."
        ) from None
    except urllib.error.URLError as error:
        raise UpdateError(
            f"Cannot reach GitHub to check for updates ({error.reason}); check your connection."
        ) from None
    except (TimeoutError, OSError) as error:
        raise UpdateError(
            f"Cannot reach GitHub to check for updates ({error}); check your connection."
        ) from None
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise UpdateError("GitHub returned an unreadable release description; try again.") from None

    if not isinstance(payload, dict):
        raise UpdateError("GitHub returned an unexpected release description.")
    tag = str(payload.get("tag_name") or "").strip()
    if not tag:
        raise UpdateError("The latest GitHub release has no tag, so its version cannot be compared.")
    version = ".".join(str(part) for part in version_tuple(tag))
    if not version:
        raise UpdateError(f"Release tag {tag!r} has no version number to compare.")
    url = str(payload.get("html_url") or "").strip()
    if not url.startswith(f"https://github.com/{REPOSITORY}/"):
        url = RELEASES_URL  # never hand the browser an unexpected origin
    return {
        "tag": tag,
        "version": version,
        "name": str(payload.get("name") or tag)[:200],
        "url": url,
        "notes": str(payload.get("body") or "")[:4000],
        "published_at": str(payload.get("published_at") or ""),
        "prerelease": bool(payload.get("prerelease", False)),
    }


def check_for_update(root: Path | str | None = None, timeout: float = NETWORK_TIMEOUT,
                     fetch: Callable[..., dict] = fetch_latest_release) -> dict:
    """Compare the checkout's version with the newest GitHub release.

    Network failures are data, not exceptions: the result carries an ``error``
    message the caller can show, and the update never starts from a failed check.
    """
    current = local_version(root)
    checked = time.time()
    try:
        release = fetch(timeout=timeout)
    except UpdateError as error:
        return _result(current, None, None, checked, str(error))
    except Exception as error:  # noqa: BLE001 - an injected fetcher must not crash the caller
        return _result(current, None, None, checked, f"Could not check for updates: {error}")

    if not isinstance(release, dict):
        return _result(current, None, None, checked,
                       "GitHub returned an unexpected release description; try again.")
    latest = release.get("version") or release.get("tag")
    return _result(current, str(latest) if latest else None, release, checked, None)


def _result(current: str, latest: str | None, release: dict | None,
            checked: float, error: str | None) -> dict:
    return {
        "local_version": current,
        "latest_version": latest,
        "update_available": bool(latest and is_newer(latest, current)),
        "release": release,
        "checked_at": checked,
        "error": error,
        "ok": error is None,
    }


# ---------------------------------------------------------------------------
# Git update
# ---------------------------------------------------------------------------
def is_git_checkout(root: Path | str | None = None) -> bool:
    """A clone or worktree carries a ``.git`` directory or file."""
    marker = Path(root or repository_root()) / ".git"
    return marker.is_dir() or marker.is_file()


def git_available() -> bool:
    """Whether the ``git`` executable is on ``PATH``."""
    return shutil.which("git") is not None


def run_git(command: list[str], on_line: LineCallback | None = None,
            cwd: Path | str | None = None) -> tuple[int, str]:
    """Run one git command, streaming its combined output to ``on_line``."""
    environment = {
        **os.environ,
        "GIT_TERMINAL_PROMPT": "0",   # fail instead of waiting for credentials
        "GIT_MERGE_AUTOEDIT": "no",
    }
    try:
        process = subprocess.Popen(
            command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=environment,
        )
    except OSError as error:
        raise UpdateError(f"Could not run git: {error}") from None
    lines: list[str] = []
    try:
        for raw in process.stdout or ():
            line = raw.rstrip()
            lines.append(line)
            if on_line:
                on_line(line)
        return process.wait(), "\n".join(lines)
    finally:
        if process.stdout is not None:
            process.stdout.close()


def _detail(output: str) -> str:
    lines = [line.strip() for line in str(output or "").splitlines() if line.strip()]
    return "\n".join(lines[-8:])[-MAX_DETAIL:]


def _step(command: list[str], runner: Runner, what: str,
          on_line: LineCallback | None = None) -> str:
    code, output = runner(command, on_line)
    if code != 0:
        message = f"{what} failed (git exited with {code})."
        detail = _detail(output)
        raise UpdateError(f"{message}\n{detail}" if detail else message)
    return output


def apply_update(root: Path | str | None = None, runner: Runner | None = None,
                 on_line: LineCallback | None = None,
                 on_step: Callable[[int, str], None] | None = None) -> dict:
    """Fast-forward the checkout to the latest release with git.

    Only tracked, committed changes are allowed: a dirty tree would make the
    fast-forward ambiguous.  Untracked files are ignored because they cannot be
    overwritten by a fast-forward (git aborts if one would be).
    """
    root = Path(root or repository_root()).resolve()
    if not is_git_checkout(root):
        raise UpdateError(
            f"This copy is not a git checkout; download the new release from {RELEASES_URL}."
        )
    if runner is None:
        if not git_available():
            raise UpdateError(
                "git is not installed; install git and try again, or download the release."
            )

        def _runner(command: list[str], line_callback: LineCallback | None = None) -> tuple[int, str]:
            return run_git(command, line_callback, cwd=root)

        runner = _runner

    report = on_line or (lambda line: None)
    step = on_step or (lambda index, label: None)
    previous = local_version(root)

    step(1, "Checking the checkout")
    status = _step(["git", "status", "--porcelain", "--untracked-files=no"], runner,
                   "Checking the checkout", report)
    if any(line.strip() for line in status.splitlines()):
        raise UpdateError(
            "The checkout has uncommitted changes. Commit or stash them, then try again."
        )
    branch = _step(["git", "rev-parse", "--abbrev-ref", "HEAD"], runner,
                   "Reading the current branch", report).strip()
    if branch == "HEAD":
        raise UpdateError(
            "This checkout is on a detached HEAD. Switch to a branch "
            "(git switch main) and try again."
        )

    step(2, "Fetching the latest release")
    _step(["git", "fetch", "--tags", "--prune", "origin"], runner,
          "Fetching from GitHub", report)

    step(3, "Fast-forwarding to the latest release")
    _step(["git", "merge", "--ff-only", f"origin/{branch}"], runner,
          "Updating the checkout", report)

    current = local_version(root)
    updated = current != previous
    return {
        "previous_version": previous,
        "version": current,
        "branch": branch,
        "updated": updated,
        "restart_required": updated,
    }


# ---------------------------------------------------------------------------
# CLI (reached before PyTorch is imported)
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    """``mariana update [--check]``; returns a process exit code."""
    parser = argparse.ArgumentParser(
        prog="mariana update",
        description="Compare this checkout with the latest GitHub release and update it with git.",
    )
    parser.add_argument("--check", action="store_true",
                        help="only report whether a newer release exists")
    parser.add_argument("--timeout", type=float, default=NETWORK_TIMEOUT,
                        help="seconds to wait for GitHub (default: 10)")
    args = parser.parse_args(argv)

    root = repository_root()
    current = local_version(root)
    print(f"Mariana {current} (local)")
    print(f"Checking {RELEASES_URL}…")
    try:
        result = check_for_update(root, timeout=args.timeout)
    except KeyboardInterrupt:
        print("\nUpdate check cancelled.")
        return 130
    if result["error"]:
        print(f"Update check failed: {result['error']}")
        return 1

    release = result["release"] or {}
    if not result["update_available"]:
        print(f"Up to date: {current} is the latest release.")
        return 0

    print(f"New release: {release.get('version') or result['latest_version']} — {release.get('name', '')}".rstrip())
    if release.get("url"):
        print(release["url"])
    if args.check:
        print("Run `python mariana.py update` to install it.")
        return 0

    if not is_git_checkout(root):
        print("This copy is not a git checkout; download the release from the URL above.")
        return 1
    if not git_available():
        print("git is not installed; install git and try again, or download the release.")
        return 1

    print(f"Updating {root} with git…")
    try:
        outcome = apply_update(root, on_line=lambda line: print(f"  {line}"))
    except KeyboardInterrupt:
        print("\nUpdate cancelled.")
        return 130
    except UpdateError as error:
        print(f"Update failed: {error}")
        return 1

    print(f"Mariana is now {outcome['version']}.")
    if not outcome.get("updated"):
        print("The checkout was already current; nothing to restart.")
        return 0
    print("Restart Mariana Studio (or any running Mariana process) to load the new code:")
    print("    python mariana.py studio")
    return 0


if __name__ == "__main__":  # pragma: no cover - the wrapper calls main()
    sys.exit(main())
