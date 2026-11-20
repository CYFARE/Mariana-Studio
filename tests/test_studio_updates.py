"""Release checks and one-at-a-time update jobs in Studio."""

from __future__ import annotations

import pytest

from mariana.studio.errors import ConflictError
from mariana.studio.jobs import JobRegistry
from mariana.studio.updates import UpdateService
from tests.helpers import wait_for

RELEASE = {
    "tag": "9.9.9",
    "version": "9.9.9",
    "name": "v9.9.9",
    "url": "https://github.com/CYFARE/Mariana-Studio/releases/tag/9.9.9",
    "notes": "",
    "published_at": "",
    "prerelease": False,
}


def make_service(tmp_path, **settings):
    (tmp_path / ".git").mkdir(exist_ok=True)
    (tmp_path / "VERSION.txt").write_text("2.3.0", encoding="utf-8")
    registry = JobRegistry()
    service = UpdateService(
        tmp_path, registry, lambda: {"check_for_updates": True, **settings},
    )
    service.fetch = lambda timeout=10: dict(RELEASE)
    return service, registry


def checked(service):
    service.start_check(force=True)
    return wait_for(lambda: (lambda status: status if not status["checking"] else None)(service.status()))


def test_status_starts_one_background_check_and_caches_it(tmp_path):
    service, _ = make_service(tmp_path)
    calls = []

    def fetch(timeout=10):
        calls.append(1)
        return dict(RELEASE)

    service.fetch = fetch
    status = checked(service)
    assert status["update_available"] is True
    assert status["latest_version"] == "9.9.9"
    assert status["git"]["checkout"] is True
    assert len(calls) == 1

    service.status()
    assert len(calls) == 1  # cached for CACHE_SECONDS


def test_auto_check_can_be_disabled_without_blocking_manual_checks(tmp_path):
    service, _ = make_service(tmp_path, check_for_updates=False)
    calls = []
    service.fetch = lambda timeout=10: calls.append(1) or dict(RELEASE)

    status = service.status()
    assert status["auto_check"] is False and status["checking"] is False
    assert calls == []

    status = checked(service)
    assert status["update_available"] is True
    assert calls == [1]


def test_apply_runs_a_job_and_requests_a_restart(tmp_path):
    service, registry = make_service(tmp_path)
    checked(service)
    commands = []

    def run(command, on_line=None):
        commands.append(command)
        if "--porcelain" in command:
            return 0, ""
        if "--abbrev-ref" in command:
            return 0, "main\n"
        if "merge" in command:
            (tmp_path / "VERSION.txt").write_text("9.9.9", encoding="utf-8")
        return 0, "ok"

    service.run_git = run
    started = service.apply()
    assert started["started"] is True
    job = wait_for(lambda: (lambda snapshot: snapshot if not snapshot["active"] else None)(
        registry.get(started["job"]["id"])))
    assert job["status"] == "completed"
    assert job["result"]["version"] == "9.9.9"
    assert job["result"]["restart_required"] is True
    assert ["git", "merge", "--ff-only", "origin/main"] in commands
    assert service.status()["restart_required"] is True


def test_apply_requires_a_finished_check_and_an_available_release(tmp_path):
    service, _ = make_service(tmp_path)
    with pytest.raises(ConflictError, match="has not checked"):
        service.apply()

    service._result = {
        "local_version": "2.3.0", "latest_version": "2.3.0", "update_available": False,
        "release": None, "checked_at": 0, "error": None, "ok": True,
    }
    with pytest.raises(ConflictError, match="already up to date"):
        service.apply()
