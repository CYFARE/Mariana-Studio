"""HTTP routes for dependency detection and background installation."""

from __future__ import annotations

import json

import pytest

from tests.helpers import request, request_error, studio, wait_for


def test_dependency_status_and_install_route(tmp_path):
    pytest.importorskip("pip")
    (tmp_path / "agent-workspace").mkdir()
    (tmp_path / "agent-workspace" / "settings.json").write_text(
        json.dumps({"auto_install_dependencies": False, "discover_local": False}),
        encoding="utf-8",
    )
    with studio(tmp_path) as (server, base):
        status = request(base + "/api/dependencies")
        assert [group["id"] for group in status["groups"]] == ["core", "finetune", "export", "data"]
        assert status["auto_install"] is False and status["pip"] is True
        assert "job" in status and status["busy"] is False

        commands = []

        def fake_run(job, command, **kwargs):
            commands.append(command)
            return 0

        server.app.dependencies.registry.run_process = fake_run
        result = request(base + "/api/dependencies/install", {"groups": ["data"]})
        assert result["started"] is True
        job = wait_for(lambda: (lambda snapshot: snapshot if not snapshot["active"] else None)(
            request(base + f"/api/jobs/{result['job']['id']}")["job"]))
        assert job["status"] == "completed" and job["meta"]["groups"] == ["data"]
        assert commands and commands[0][-1].endswith("requirements-data.txt")

        status, error = request_error(base + "/api/dependencies/install", {"groups": ["unknown"]})
        assert status == 400 and "Unknown requirement group" in error["error"]
