"""Background work as observable jobs.

A :class:`Job` is the single progress model of Studio: training runs,
fine-tunes, downloads and exports all appear as jobs with a status, a stage,
numeric progress and a log tail.  :class:`JobRegistry` owns them, runs their
targets on daemon threads and wakes event-stream subscribers on every change.

Managers that keep richer state of their own (the training manager) register
as *providers* and are merged into the same listing.
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable

from .errors import ConflictError, NotFoundError

ACTIVE_STATUSES = frozenset({"queued", "preparing", "running"})
FINAL_STATUSES = frozenset({"completed", "failed", "cancelled", "paused"})
MAX_LOG_LINES = 2000
MAX_FINISHED_JOBS = 40


class JobCancelled(Exception):
    """Raised inside a job target to stop it after a cancel request."""


class Job:
    """Thread-safe state for one unit of background work."""

    def __init__(self, registry: "JobRegistry", kind: str, title: str, meta: dict | None = None,
                 can_pause: bool = False):
        self.registry = registry
        self.id = uuid.uuid4().hex[:12]
        self.kind = kind
        self.title = title
        self.meta = dict(meta or {})
        self.status = "queued"
        self.stage = "Queued"
        self.current = 0.0
        self.total = 0.0
        self.unit = ""
        self.created = time.time()
        self.started: float | None = None
        self.ended: float | None = None
        self.error: str | None = None
        self.result: dict = {}
        self.can_pause = can_pause
        self.cancel_requested = False
        self.pause_requested = False
        self.process: subprocess.Popen | None = None
        self._logs: deque[str] = deque(maxlen=MAX_LOG_LINES)
        self._log_count = 0
        self._on_cancel: Callable[[], None] | None = None

    # -- mutation (called from the worker thread) -------------------------
    def log(self, line: str) -> None:
        with self.registry.condition:
            self._logs.append(str(line).rstrip())
            self._log_count += 1
            self.registry._changed()

    def update(self, *, stage: str | None = None, current: float | None = None,
               total: float | None = None, unit: str | None = None,
               status: str | None = None, **meta) -> None:
        with self.registry.condition:
            if stage is not None:
                self.stage = stage
            if current is not None:
                self.current = float(current)
            if total is not None:
                self.total = float(total)
            if unit is not None:
                self.unit = unit
            if status is not None:
                self.status = status
                if status == "running" and self.started is None:
                    self.started = time.time()
            self.meta.update(meta)
            self.registry._changed()

    def check_cancelled(self) -> None:
        if self.cancel_requested:
            raise JobCancelled()

    def on_cancel(self, callback: Callable[[], None]) -> None:
        """Register a hook that interrupts blocking work (closes a socket…)."""
        self._on_cancel = callback

    def finish(self, status: str, *, error: str | None = None, result: dict | None = None,
               stage: str | None = None) -> None:
        with self.registry.condition:
            if self.status in FINAL_STATUSES:
                return
            self.status = status
            self.ended = time.time()
            self.error = error
            if result:
                self.result.update(result)
            self.stage = stage or {
                "completed": "Complete", "failed": "Failed",
                "cancelled": "Cancelled", "paused": "Paused",
            }.get(status, status.title())
            self.process = None
            self.registry._changed()

    # -- views -----------------------------------------------------------
    @property
    def active(self) -> bool:
        return self.status in ACTIVE_STATUSES

    def snapshot(self, logs: bool = False, since: int = 0) -> dict:
        with self.registry.condition:
            end = self.ended or time.time()
            percent = 0.0
            if self.total > 0:
                percent = max(0.0, min(100.0, 100.0 * self.current / self.total))
            elif self.status == "completed":
                percent = 100.0
            data = {
                "id": self.id, "kind": self.kind, "title": self.title,
                "status": self.status, "stage": self.stage, "active": self.active,
                "progress": {
                    "current": self.current, "total": self.total,
                    "unit": self.unit, "percent": round(percent, 2),
                },
                "created": self.created, "started": self.started, "ended": self.ended,
                "elapsed": max(0.0, end - (self.started or self.created)),
                "error": self.error, "result": dict(self.result), "meta": dict(self.meta),
                "can_pause": self.can_pause,
                "cancel_requested": self.cancel_requested,
                "pause_requested": self.pause_requested,
                "log_count": self._log_count,
            }
            if logs:
                kept = list(self._logs)
                first = self._log_count - len(kept)
                data["logs"] = kept[max(0, since - first):]
                data["log_start"] = max(first, since)
            return data


class JobRegistry:
    """Owns generic jobs and merges provider jobs into one stream."""

    def __init__(self):
        self.condition = threading.Condition()
        self.version = 0
        self._jobs: dict[str, Job] = {}
        self._providers: list = []
        self._closed = False

    # -- change notification --------------------------------------------
    def _changed(self) -> None:
        self.version += 1
        self.condition.notify_all()

    def touch(self) -> None:
        """Wake subscribers after a provider changed its own state."""
        with self.condition:
            self._changed()

    def wait(self, version: int, timeout: float = 15.0) -> int:
        with self.condition:
            self.condition.wait_for(
                lambda: self.version != version or self._closed, timeout=timeout,
            )
            return self.version

    # -- providers -------------------------------------------------------
    def add_provider(self, provider) -> None:
        """``provider`` implements ``jobs()``, ``job(id)``, ``cancel(id)``, ``pause(id)``."""
        self._providers.append(provider)

    def _provider_for(self, job_id: str):
        for provider in self._providers:
            if provider.job(job_id) is not None:
                return provider
        return None

    # -- lifecycle -------------------------------------------------------
    def create(self, kind: str, title: str, *, meta: dict | None = None,
               can_pause: bool = False) -> Job:
        job = Job(self, kind, title, meta, can_pause)
        with self.condition:
            self._jobs[job.id] = job
            self._trim_locked()
            self._changed()
        return job

    def start(self, job: Job, target: Callable[[Job], dict | None]) -> Job:
        """Run ``target(job)`` on a daemon thread and settle the final status."""

        def runner():
            job.update(status="running", stage=job.stage if job.stage != "Queued" else "Starting")
            try:
                result = target(job)
            except JobCancelled:
                job.finish("cancelled")
            except Exception as error:  # noqa: BLE001 - a job failure is data, not a crash
                if job.cancel_requested:
                    job.finish("cancelled")
                else:
                    job.log(f"ERROR: {error}")
                    job.finish("failed", error=str(error))
            else:
                if job.cancel_requested and job.status not in FINAL_STATUSES:
                    job.finish("cancelled")
                elif job.pause_requested and job.status not in FINAL_STATUSES:
                    job.finish("paused", result=result if isinstance(result, dict) else None)
                else:
                    job.finish("completed", result=result if isinstance(result, dict) else None)

        threading.Thread(target=runner, name=f"job-{job.kind}-{job.id}", daemon=True).start()
        return job

    def run_process(self, job: Job, command: list[str], *, cwd=None, env: dict | None = None,
                    on_line: Callable[[str], None] | None = None) -> int:
        """Run a subprocess inside a job target, streaming its output to the log."""
        job.check_cancelled()
        process = subprocess.Popen(
            command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env={**os.environ, "PYTHONUNBUFFERED": "1", **(env or {})},
        )
        with self.condition:
            job.process = process
            if job.cancel_requested and process.poll() is None:
                process.terminate()
        try:
            for line in process.stdout:
                job.log(line)
                if on_line:
                    on_line(line)
            return process.wait()
        finally:
            with self.condition:
                job.process = None

    def _trim_locked(self) -> None:
        finished = sorted(
            (job for job in self._jobs.values() if not job.active),
            key=lambda item: item.ended or item.created,
        )
        for job in finished[:-MAX_FINISHED_JOBS] if len(finished) > MAX_FINISHED_JOBS else []:
            self._jobs.pop(job.id, None)

    # -- queries ---------------------------------------------------------
    def list(self) -> list[dict]:
        with self.condition:
            jobs = [job.snapshot() for job in self._jobs.values()]
        for provider in self._providers:
            jobs.extend(provider.jobs())
        jobs.sort(key=lambda item: item.get("created") or 0, reverse=True)
        return jobs

    def get(self, job_id: str, since: int = 0) -> dict:
        with self.condition:
            job = self._jobs.get(job_id)
        if job is not None:
            return job.snapshot(logs=True, since=since)
        for provider in self._providers:
            found = provider.job(job_id, since=since)
            if found is not None:
                return found
        raise NotFoundError("Job not found; it may have been dismissed")

    def find(self, job_id: str) -> Job | None:
        with self.condition:
            return self._jobs.get(job_id)

    def active(self, kind: str | None = None) -> list[dict]:
        return [job for job in self.list() if job["active"] and (kind is None or job["kind"] == kind)]

    # -- control ---------------------------------------------------------
    def cancel(self, job_id: str) -> dict:
        with self.condition:
            job = self._jobs.get(job_id)
            if job is not None:
                if not job.active:
                    raise ConflictError("This job is not running")
                job.cancel_requested = True
                process, hook = job.process, job._on_cancel
                self._changed()
        if job is None:
            provider = self._provider_for(job_id)
            if provider is None:
                raise NotFoundError("Job not found; it may have been dismissed")
            return provider.cancel(job_id)
        if hook:
            try:
                hook()
            except Exception:  # noqa: BLE001 - a failing hook must not block the cancel
                pass
        if process is not None and process.poll() is None:
            process.terminate()
            threading.Timer(5.0, lambda: process.poll() is None and process.kill()).start()
        return job.snapshot(logs=True)

    def pause(self, job_id: str) -> dict:
        with self.condition:
            job = self._jobs.get(job_id)
            if job is not None:
                if not job.active:
                    raise ConflictError("This job is not running")
                if not job.can_pause:
                    raise ConflictError("This job cannot be paused; cancel it instead")
                job.pause_requested = True
                self._changed()
                return job.snapshot(logs=True)
        provider = self._provider_for(job_id)
        if provider is None:
            raise NotFoundError("Job not found; it may have been dismissed")
        return provider.pause(job_id)

    def remove(self, job_id: str) -> None:
        with self.condition:
            job = self._jobs.get(job_id)
            if job is None:
                if self._provider_for(job_id) is not None:
                    raise ConflictError("Training runs stay visible until the next run starts")
                raise NotFoundError("Job not found; it may have been dismissed")
            if job.active:
                raise ConflictError("Cancel the job before dismissing it")
            self._jobs.pop(job_id, None)
            self._changed()

    def clear_finished(self) -> int:
        with self.condition:
            finished = [job_id for job_id, job in self._jobs.items() if not job.active]
            for job_id in finished:
                self._jobs.pop(job_id, None)
            if finished:
                self._changed()
            return len(finished)

    def close(self) -> None:
        with self.condition:
            self._closed = True
            jobs = [job for job in self._jobs.values() if job.active]
            for job in jobs:
                job.cancel_requested = True
            self.condition.notify_all()
        for job in jobs:
            hook, process = job._on_cancel, job.process
            if hook:
                try:
                    hook()
                except Exception:  # noqa: BLE001
                    pass
            if process is not None and process.poll() is None:
                process.terminate()
