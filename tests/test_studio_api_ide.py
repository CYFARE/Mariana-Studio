"""IDE HTTP routes: workspace files, search, sessions, runs and the event stream."""

from __future__ import annotations

from tests.helpers import read_events, request, request_error, studio, wait_for


class ScriptedGateway:
    def __init__(self, replies):
        self.replies = list(replies)

    def catalog(self, force=False):
        return [{"id": "fake-coder", "name": "Fake coder", "available": True,
                 "capabilities": {"chat": True}}]

    def stream(self, config, cancelled=None):
        reply = self.replies.pop(0) if self.replies else '{"final": "done"}'
        yield {"type": "start", "model_id": "fake-coder", "model": "fake"}
        yield {"type": "delta", "text": reply}
        yield {"type": "done", "reply": reply, "usage": {"completion_tokens": 2},
               "elapsed": 0.01}


class NoModelGateway:
    def catalog(self, force=False):
        return []


def create(base, name="api-demo"):
    response = request(base + "/api/ide/workspaces", {"name": name, "template": "python"})
    return response["workspace"]


def wait_for_run(base, run_id, timeout=10.0):
    return wait_for(lambda: (lambda run: run if run["status"] in
                             {"completed", "failed", "cancelled"} else None)(
        request(base + f"/api/ide/runs/{run_id}")), timeout=timeout)


def test_ide_discovery_and_workspace_file_routes(tmp_path):
    with studio(tmp_path) as (_, base):
        overview = request(base + "/api/ide")
        assert overview["protocol"] == "IDE"
        assert {mode["id"] for mode in overview["modes"]} == {"ask", "edit", "auto"}
        assert "file.edit" in overview["tool_schemas"]
        assert "agent.swarm" in overview["tool_schemas"]
        assert overview["features"]["context_consolidation"] is True
        assert overview["features"]["research_swarm"] is True
        assert "context_consolidated" in overview["event_types"]
        assert "The **IDE** workspace" in request(base + "/api/ide/spec")
        voice = request(base + "/api/ide/voice")
        assert "available" in voice and voice["max_bytes"] > 0
        assert "models" in voice and voice["default_model"]

        workspace = create(base)
        assert workspace["files"] >= 3
        listing = request(base + "/api/ide/workspaces")
        assert workspace["id"] in {item["id"] for item in listing["workspaces"]}

        tree = request(base + f"/api/ide/workspaces/{workspace['id']}/tree")
        assert tree["files"] >= 3 and tree["children"]
        data = request(base + f"/api/ide/workspaces/{workspace['id']}/file?path=main.py")
        assert "def greet" in data["content"]
        preview = request(base + f"/api/ide/workspaces/{workspace['id']}/preview?path=main.py")
        assert preview["kind"] == "text" and preview["hex"].startswith("00000000")
        assert "def greet" in preview["content"]
        request(base + f"/api/ide/workspaces/{workspace['id']}/file", {
            "path": ".agents/skills/review/SKILL.md",
            "content": "---\nname: Review\ndescription: Check a diff\n---\nRead the change.\n",
            "version": "",
        }, method="PUT")
        skills = request(base + f"/api/ide/workspaces/{workspace['id']}/skills")["skills"]
        assert any(item["id"] == "review" and item["scope"] == "workspace" for item in skills)
        assert "skill.read" in overview["tool_schemas"]
        skill = request(base + f"/api/ide/workspaces/{workspace['id']}/skills/review")["skill"]
        assert "Read the change." in skill["instructions"]
        request(base + f"/api/ide/workspaces/{workspace['id']}/skills/review", {
            "name": "Review", "description": "Check a diff",
            "instructions": "Read the change and the tests.",
        }, method="PUT")
        created_skill = request(base + f"/api/ide/workspaces/{workspace['id']}/skills", {
            "name": "Release notes", "description": "Ship it",
            "instructions": "# Steps\nRead the changelog.",
        })["skill"]
        assert created_skill["id"] == "release-notes"
        request(base + f"/api/ide/workspaces/{workspace['id']}/skills/release-notes",
                method="DELETE")
        status, _ = request_error(
            base + f"/api/ide/workspaces/{workspace['id']}/skills/release-notes")
        assert status == 404

        saved = request(base + f"/api/ide/workspaces/{workspace['id']}/file", {
            "path": "main.py", "content": "print('api')\n", "version": data["version"],
        }, method="PUT")
        assert saved["written"] and saved["version"] != data["version"]
        status, error = request_error(base + f"/api/ide/workspaces/{workspace['id']}/file", {
            "path": "main.py", "content": "stale\n", "version": data["version"],
        }, method="PUT")
        assert status == 409 and "reload" in error["error"]

        found = request(base + f"/api/ide/workspaces/{workspace['id']}/search",
                        {"query": "api", "regex": False})
        assert found["matches"] and found["matches"][0]["path"] == "main.py"
        diagnostics = request(base + f"/api/ide/workspaces/{workspace['id']}/diagnostics", {})
        assert diagnostics["ok"] is True and diagnostics["files_checked"] >= 1

        request(base + f"/api/ide/workspaces/{workspace['id']}/file?path=main.py",
                method="DELETE")
        status, error = request_error(
            base + f"/api/ide/workspaces/{workspace['id']}/file?path=main.py")
        assert status == 404
        request(base + f"/api/ide/workspaces/{workspace['id']}", method="DELETE")
        status, error = request_error(base + f"/api/ide/workspaces/{workspace['id']}/tree")
        assert status == 404


def test_run_stream_session_and_revert_routes(tmp_path):
    with studio(tmp_path) as (server, base):
        workspace = create(base)
        server.app.ide.gateway = ScriptedGateway([
            '{"tool": "file.edit", "input": {"path": "main.py", '
            '"old": "return f\\"Hello, {name}!\\"", "new": "return f\\"Hey, {name}!\\""}}',
            '{"final": "Renamed the greeting."}',
        ])
        started = request(base + f"/api/ide/workspaces/{workspace['id']}/runs", {
            "message": "Change the greeting", "mode": "edit",
        })
        run_id = started["run"]["id"]
        session_id = started["session"]["id"]
        events = read_events(base + f"/api/ide/runs/{run_id}/events", limit=200)
        assert any(event["type"] == "tool_call" for event in events)
        final = wait_for_run(base, run_id)
        assert final["status"] == "completed", final.get("error")
        assert "Renamed" in final["output"]

        session = request(base + f"/api/ide/sessions/{session_id}")["session"]
        assert session["transcript"] and session["changes"]["main.py"]["edits"] == 1
        listed = request(base + f"/api/ide/workspaces/{workspace['id']}/sessions")
        assert session_id in {item["id"] for item in listed["sessions"]}
        assert "Hey, {name}" in request(
            base + f"/api/ide/workspaces/{workspace['id']}/file?path=main.py")["content"]

        reverted = request(base + f"/api/ide/sessions/{session_id}/revert", {})
        assert reverted == {"restored": 1, "deleted": 0, "skipped": 0}
        assert "Hello, {name}" in request(
            base + f"/api/ide/workspaces/{workspace['id']}/file?path=main.py")["content"]

        consolidated = request(base + f"/api/ide/sessions/{session_id}/consolidate", {})
        assert consolidated["consolidated"] is False

        request(base + f"/api/ide/sessions/{session_id}", method="DELETE")
        status, _ = request_error(base + f"/api/ide/sessions/{session_id}")
        assert status == 404


def test_workspace_operations_context_and_session_crud_routes(tmp_path):
    with studio(tmp_path) as (server, base):
        workspace = create(base)
        root = request(base + f"/api/ide/workspaces/{workspace['id']}/tree?depth=1")
        assert root["children"]
        settings = request(base + f"/api/ide/workspaces/{workspace['id']}/settings")
        assert settings["excluded"] == []
        updated = request(base + f"/api/ide/workspaces/{workspace['id']}/settings", {
            "excluded": ["hidden"],
            "context": {"include": ["main.py"]},
        }, method="PUT")
        assert updated["excluded"] == ["hidden"]

        request(base + f"/api/ide/workspaces/{workspace['id']}/folder", {"path": "pkg/deep"})
        request(base + f"/api/ide/workspaces/{workspace['id']}/file", {
            "path": "pkg/deep/mod.py", "content": "x = 1\n", "version": "",
        }, method="PUT")
        found = request(base + f"/api/ide/workspaces/{workspace['id']}/files?query=mod")
        assert [item["path"] for item in found["files"]] == ["pkg/deep/mod.py"]
        moved = request(base + f"/api/ide/workspaces/{workspace['id']}/move", {
            "from": "pkg/deep/mod.py", "to": "pkg/deep/renamed.py",
        })
        assert moved["to"] == "pkg/deep/renamed.py"
        preview = request(base + f"/api/ide/workspaces/{workspace['id']}/context", {
            "context": {"include": ["pkg"], "exclude": []},
        })
        assert preview["selected"] == 1 and preview["files"][0]["path"] == "pkg/deep/renamed.py"

        session = server.app.ide._new_session(workspace["id"], "Route task", "edit", "")
        renamed = request(base + f"/api/ide/sessions/{session['id']}", {
            "title": "Renamed route", "pinned": True,
        }, method="PATCH")
        assert renamed["session"]["title"] == "Renamed route"
        assert renamed["session"]["pinned"] is True
        pinned = request(base + f"/api/ide/sessions/{session['id']}", {
            "context": {"include": ["main.py"], "exclude": ["pkg"]},
        }, method="PATCH")
        assert pinned["session"]["context"] == {"include": 1, "exclude": 1}
        copy = request(base + f"/api/ide/sessions/{session['id']}/duplicate", {})
        assert copy["session"]["id"] != session["id"]

        request(base + f"/api/ide/workspaces/{workspace['id']}/folder?path=pkg", method="DELETE")
        gone = request(base + f"/api/ide/workspaces/{workspace['id']}/files?query=renamed")
        assert gone["files"] == []


def test_run_without_a_chat_model_fails_with_actionable_events(tmp_path):
    with studio(tmp_path) as (server, base):
        server.app.ide.gateway = NoModelGateway()
        workspace = create(base)
        started = request(base + f"/api/ide/workspaces/{workspace['id']}/runs", {
            "message": "Do anything", "mode": "edit",
        })
        run = wait_for_run(base, started["run"]["id"])
        assert run["status"] == "failed"
        assert "No chat model" in run["error"]
        streamed = read_events(base + f"/api/ide/runs/{run['id']}/events", limit=100)
        assert any(event["type"] == "run_failed" for event in streamed)
        status, _ = request_error(base + f"/api/ide/runs/{run['id']}/cancel", {})
        assert status == 409
