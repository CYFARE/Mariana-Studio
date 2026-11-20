"""HTTP routes for release checks and the git update job."""

from __future__ import annotations

from tests.helpers import request, request_error, studio, wait_for

RELEASE = {
    "tag": "9.9.9",
    "version": "9.9.9",
    "name": "v9.9.9",
    "url": "https://github.com/CYFARE/Mariana-Studio/releases/tag/9.9.9",
    "notes": "Release notes",
    "published_at": "",
    "prerelease": False,
}


def test_update_routes_check_and_apply(tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / "VERSION.txt").write_text("2.3.0", encoding="utf-8")
    with studio(tmp_path) as (server, base):
        server.app.updates.fetch = lambda timeout=10: dict(RELEASE)

        status = request(base + "/api/updates")
        assert status["local_version"] == "2.3.0"
        status = wait_for(lambda: (lambda value: value if not value["checking"] else None)(
            request(base + "/api/updates")))
        assert status["update_available"] is True
        assert status["release"]["tag"] == "9.9.9"

        def run(command, on_line=None):
            if "--porcelain" in command:
                return 0, ""
            if "--abbrev-ref" in command:
                return 0, "main\n"
            if "merge" in command:
                (tmp_path / "VERSION.txt").write_text("9.9.9", encoding="utf-8")
            return 0, "ok"

        server.app.updates.run_git = run
        result = request(base + "/api/updates/apply", {})
        assert result["started"] is True
        job = wait_for(lambda: (lambda value: value if not value["active"] else None)(
            request(base + f"/api/jobs/{result['job']['id']}")["job"]))
        assert job["status"] == "completed" and job["result"]["restart_required"] is True

        status = wait_for(lambda: (lambda value: value if not value["checking"] else None)(
            request(base + "/api/updates")))
        assert status["restart_required"] is True
        assert status["local_version"] == "9.9.9"


def test_apply_route_rejects_an_up_to_date_checkout(tmp_path):
    (tmp_path / "VERSION.txt").write_text("2.3.0", encoding="utf-8")
    with studio(tmp_path) as (server, base):
        server.app.updates.fetch = lambda timeout=10: {**RELEASE, "tag": "2.3.0", "version": "2.3.0"}
        wait_for(lambda: (lambda value: value if not value["checking"] else None)(
            request(base + "/api/updates")))

        status, error = request_error(base + "/api/updates/apply", {})
        assert status == 409
        assert "already up to date" in error["error"]
