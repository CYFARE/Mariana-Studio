"""The job registry: lifecycle, logs, subprocesses, cancellation and providers."""

from __future__ import annotations

import sys
import threading

import pytest

from mariana.studio import jobs as jobs_module
from mariana.studio.errors import ConflictError, NotFoundError
from mariana.studio.jobs import JobCancelled, JobRegistry
from tests.helpers import wait_for


@pytest.fixture()
def registry():
    instance = JobRegistry()
    yield instance
    instance.close()


def settled(registry, job_id):
    return wait_for(lambda: (lambda job: job if not job["active"] else None)(registry.get(job_id)))


def test_a_job_reports_progress_logs_and_its_result(registry):
    gate = threading.Event()

    def work(job):
        job.update(stage="Working", current=3, total=12, unit="files", note="extra")
        job.log("first line\n")
        job.log("second line")
        gate.wait(5)
        return {"answer": 42}

    job = registry.create("download", "Fetch things", meta={"repo": "org/model"})
    assert job.snapshot()["status"] == "queued"
    version = registry.version
    registry.start(job, work)
    running = wait_for(lambda: (lambda snap: snap if snap["stage"] == "Working" else None)(registry.get(job.id)))
    assert registry.wait(version, timeout=1) != version
    assert running["active"] and running["kind"] == "download" and running["title"] == "Fetch things"
    assert running["progress"] == {"current": 3.0, "total": 12.0, "unit": "files", "percent": 25.0}
    assert running["meta"] == {"repo": "org/model", "note": "extra"} and running["started"]
    assert running["logs"] == ["first line", "second line"] and running["log_count"] == 2
    assert registry.get(job.id, since=1)["logs"] == ["second line"]
    assert [item["id"] for item in registry.active("download")] == [job.id] and registry.active("export") == []
    with pytest.raises(ConflictError, match="Cancel the job"):
        registry.remove(job.id)
    gate.set()
    done = settled(registry, job.id)
    assert done["status"] == "completed" and done["stage"] == "Complete"
    assert done["result"] == {"answer": 42} and done["progress"]["percent"] == 25.0 and done["ended"]
    assert "logs" not in registry.list()[0]
    registry.remove(job.id)
    with pytest.raises(NotFoundError):
        registry.get(job.id)


def test_failures_and_cancellation_settle_the_job(registry):
    def explode(job):
        raise RuntimeError("disk on fire")

    failed = settled(registry, registry.start(registry.create("export", "Explode"), explode).id)
    assert failed["status"] == "failed" and failed["error"] == "disk on fire"
    assert failed["logs"] == ["ERROR: disk on fire"]

    hooked = threading.Event()

    def wait_for_cancel(job):
        job.on_cancel(hooked.set)
        hooked.wait(5)
        job.check_cancelled()

    job = registry.start(registry.create("download", "Cancel me"), wait_for_cancel)
    wait_for(lambda: registry.get(job.id)["status"] == "running")
    assert registry.cancel(job.id)["cancel_requested"] is True
    assert settled(registry, job.id)["status"] == "cancelled" and hooked.is_set()
    with pytest.raises(ConflictError, match="not running"):
        registry.cancel(job.id)
    with pytest.raises(ConflictError, match="not running"):
        registry.pause(job.id)

    quiet = settled(registry, registry.start(
        registry.create("download", "Raise"), lambda job: (_ for _ in ()).throw(JobCancelled())).id)
    assert quiet["status"] == "cancelled"
    assert registry.clear_finished() == 3 and registry.list() == []


def test_pause_is_only_for_jobs_that_support_it(registry):
    gate = threading.Event()
    plain = registry.start(registry.create("download", "No pause"), lambda job: gate.wait(5))
    pausable = registry.start(registry.create("train", "Pause", can_pause=True), lambda job: gate.wait(5))
    wait_for(lambda: registry.get(pausable.id)["status"] == "running")
    with pytest.raises(ConflictError, match="cannot be paused"):
        registry.pause(plain.id)
    assert registry.pause(pausable.id)["pause_requested"] is True
    gate.set()
    assert settled(registry, pausable.id)["status"] == "paused"
    assert settled(registry, plain.id)["status"] == "completed"


def test_subprocess_output_is_streamed_and_a_cancel_terminates_it(registry):
    lines = []

    def run(job):
        code = registry.run_process(
            job, [sys.executable, "-c", "print('alpha'); print('beta'); raise SystemExit(3)"],
            on_line=lines.append)
        return {"code": code}

    done = settled(registry, registry.start(registry.create("export", "Process"), run).id)
    assert done["result"] == {"code": 3} and done["logs"] == ["alpha", "beta"]
    assert lines == ["alpha\n", "beta\n"]

    def hang(job):
        registry.run_process(job, [sys.executable, "-c", "import time; print('up', flush=True); time.sleep(60)"])

    job = registry.start(registry.create("export", "Hang"), hang)
    wait_for(lambda: registry.get(job.id)["logs"] == ["up"])
    registry.cancel(job.id)
    assert settled(registry, job.id)["status"] == "cancelled"


def test_providers_are_merged_and_finished_jobs_are_trimmed(registry, monkeypatch):
    class Provider:
        def __init__(self):
            self.calls = []

        def jobs(self):
            return [{"id": "external", "kind": "train", "created": 9e12, "active": True}]

        def job(self, job_id, since=0):
            return {"id": "external", "logs": []} if job_id == "external" else None

        def cancel(self, job_id):
            self.calls.append(("cancel", job_id))
            return {"id": job_id, "status": "cancelled"}

        def pause(self, job_id):
            self.calls.append(("pause", job_id))
            return {"id": job_id}

    provider = Provider()
    registry.add_provider(provider)
    local = registry.create("download", "Local")
    assert [item["id"] for item in registry.list()] == ["external", local.id]  # newest first
    assert registry.get("external") == {"id": "external", "logs": []}
    assert registry.cancel("external")["status"] == "cancelled" and registry.pause("external") == {"id": "external"}
    assert provider.calls == [("cancel", "external"), ("pause", "external")]
    with pytest.raises(ConflictError, match="stay visible"):
        registry.remove("external")
    for unknown in (registry.cancel, registry.pause, registry.remove, registry.get):
        with pytest.raises(NotFoundError):
            unknown("nope")

    monkeypatch.setattr(jobs_module, "MAX_FINISHED_JOBS", 3)
    for index in range(6):
        registry.create("download", f"Old {index}").finish("completed")
    registry.create("download", "Trigger trim")
    finished = [item for item in registry.list() if item.get("status") == "completed"]
    assert [item["title"] for item in finished] == ["Old 5", "Old 4", "Old 3"]
