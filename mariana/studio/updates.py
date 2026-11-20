"""Check GitHub releases and update the git checkout that runs Studio.

Studio never blocks a request on GitHub or on git: checks and updates run on
daemon threads, and the update itself is a normal :class:`~.jobs.Job` so the
job centre streams its output.  A completed update asks for a restart whenever
the checkout's version changed, because the running process still holds the
previous code.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

from .. import __version__
from ..update import (
    RELEASES_URL,
    UpdateError,
    apply_update,
    check_for_update,
    fetch_latest_release,
    git_available,
    is_git_checkout,
    local_version,
)
from .errors import BadRequestError, ConflictError, UnavailableError

CACHE_SECONDS = 600.0


class UpdateService:
    """Cached release status plus one background git update job at a time."""

    def __init__(self, root: Path, registry, settings):
        self.root = Path(root).resolve()
        self.registry = registry
        self.settings = settings
        self.fetch = fetch_latest_release
        self.run_git = None  # optional runner override; None uses subprocess
        self._lock = threading.RLock()
        self._result: dict | None = None
        self._checked_at = 0.0
        self._thread: threading.Thread | None = None
        self._restart_required = False
        self._closed = False

    # -- checks ----------------------------------------------------------
    def start(self) -> None:
        """Run the automatic check once when Studio boots."""
        if self._auto_check():
            self.start_check()

    def _auto_check(self) -> bool:
        try:
            return bool(self.settings().get("check_for_updates", True))
        except Exception:  # noqa: BLE001 - a settings failure must not block the check
            return True

    def _cache_fresh(self) -> bool:
        return self._result is not None and time.monotonic() - self._checked_at < CACHE_SECONDS

    def start_check(self, force: bool = False) -> bool:
        """Start one background check; return whether a thread was started."""
        with self._lock:
            if self._closed:
                return False
            if self._thread is not None and self._thread.is_alive():
                return False
            if not force and self._cache_fresh():
                return False
            self._thread = threading.Thread(
                target=self._check, name="studio-update-check", daemon=True,
            )
            self._thread.start()
            return True

    def _check(self) -> dict:
        try:
            result = check_for_update(self.root, fetch=self.fetch)
        except Exception as error:  # noqa: BLE001 - a failed check must settle visibly
            result = {
                "local_version": local_version(self.root), "latest_version": None,
                "update_available": False, "release": None, "checked_at": time.time(),
                "error": f"Could not check for updates: {error}", "ok": False,
            }
        with self._lock:
            self._result = result
            self._checked_at = time.monotonic()
        return result

    def status(self) -> dict:
        """The cached result, local git facts and the latest update job."""
        auto = self._auto_check()
        with self._lock:
            unchecked = self._result is None and not self._closed
        if unchecked and auto:
            self.start_check()
        with self._lock:
            result = self._result
            checking = bool(self._thread is not None and self._thread.is_alive())

        local = local_version(self.root)
        latest_job = max(
            (job for job in self.registry.list() if job.get("kind") == "update"),
            key=lambda item: item.get("created") or 0, default=None,
        )
        return {
            "local_version": local,
            "running_version": __version__,
            "latest_version": (result or {}).get("latest_version"),
            "update_available": bool(result and result.get("update_available")),
            "release": (result or {}).get("release"),
            "checked_at": (result or {}).get("checked_at"),
            "error": (result or {}).get("error"),
            "checking": checking,
            "auto_check": auto,
            "restart_required": self._restart_required or local != __version__,
            "git": {"available": git_available(), "checkout": is_git_checkout(self.root)},
            "releases_url": RELEASES_URL,
            "job": latest_job,
        }

    # -- update ----------------------------------------------------------
    def apply(self) -> dict:
        """Start the git update job (or return the one already running)."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise ConflictError("An update check is still running; try again once it finishes")
            active = self.registry.active("update")
            if active:
                return {"job": active[0], "started": False}
            result = self._result
            if result is None:
                raise ConflictError("Studio has not checked for updates yet; check for updates first")
            if result.get("error"):
                raise ConflictError("The last update check failed; fix the connection and check again")
            if not result.get("update_available"):
                raise ConflictError(f"Mariana Studio {local_version(self.root)} is already up to date")
            if not is_git_checkout(self.root):
                raise BadRequestError(
                    f"This copy is not a git checkout; download the new release from {RELEASES_URL}"
                )
            if self.run_git is None and not git_available():
                raise UnavailableError(
                    "git is not installed; install git and try again, or download the new release"
                )
            release = result.get("release") or {}
            job = self.registry.create("update", "Update Mariana Studio", meta={
                "from": result.get("local_version"),
                "to": result.get("latest_version"),
                "tag": release.get("tag"),
                "view": "models",
            })
            self.registry.start(job, self._apply_job)
        return {"job": job.snapshot(), "started": True}

    def _apply_job(self, job) -> dict:
        def run(command: list[str], on_line=None) -> tuple[int, str]:
            lines: list[str] = []

            def collect(line: str) -> None:
                text = str(line).rstrip()
                if text:
                    lines.append(text)
                if on_line:
                    on_line(text)

            code = self.registry.run_process(
                job, command, cwd=self.root,
                env={"GIT_TERMINAL_PROMPT": "0", "GIT_MERGE_AUTOEDIT": "no"},
                on_line=collect,
            )
            return code, "\n".join(lines)

        def step(index: int, label: str) -> None:
            job.update(stage=label, current=max(0, index - 1), total=3, unit="step")

        try:
            outcome = apply_update(self.root, runner=self.run_git or run, on_step=step)
        except UpdateError as error:
            raise RuntimeError(str(error)) from None

        with self._lock:
            self._result = None
            self._checked_at = 0.0
            self._restart_required = bool(outcome.get("updated"))
        job.update(
            stage="Update installed — restart Studio" if outcome.get("updated") else "Already up to date",
            current=3, total=3,
        )
        return outcome

    # -- lifecycle -------------------------------------------------------
    def close(self) -> None:
        with self._lock:
            self._closed = True
