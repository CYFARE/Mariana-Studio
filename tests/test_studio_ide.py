"""IDE workspaces: sandboxed files, versions, search, snapshots and coding runs."""

from __future__ import annotations

import base64
import copy
import json
import os
import shutil
import threading
import time
from collections import Counter
from pathlib import Path

import pytest

from mariana.studio import ide as ide_module
from mariana.studio.errors import (BadRequestError, ConflictError, NotFoundError,
                                   UnavailableError)
from mariana.studio.ide import IdeRun, IdeService, MAX_FORMAT_REPAIRS
from mariana.agents.tools import ToolError, execute_ability
from tests.helpers import wait_for

FINISHED = {"completed", "failed", "cancelled"}


class ScriptedGateway:
    """A chat gateway whose replies drive the coding loop without a model."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls: list[dict] = []

    def catalog(self, force=False):
        return [{"id": "fake-coder", "name": "Fake coder", "available": True,
                 "capabilities": {"chat": True}, "reason": ""}]

    def stream(self, config, cancelled=None):
        self.calls.append(copy.deepcopy(config))
        reply = self.replies.pop(0) if self.replies else '{"final": "done"}'
        yield {"type": "start", "model_id": "fake-coder", "model": "fake"}
        yield {"type": "delta", "text": reply}
        yield {"type": "done", "reply": reply, "usage": {"prompt_tokens": 5,
                                                          "completion_tokens": 3},
               "elapsed": 0.01}


class BlockingGateway(ScriptedGateway):
    """Hold the first stream open until the test releases it."""

    def __init__(self):
        super().__init__([])
        self.started = threading.Event()
        self.release = threading.Event()

    def stream(self, config, cancelled=None):
        self.started.set()
        self.release.wait(5)
        reply = '{"final": "done"}'
        yield {"type": "done", "reply": reply, "usage": {}}


class RequestScopedGateway(ScriptedGateway):
    """Set the per-request event when a stream ends, exactly like ModelGateway."""

    def stream(self, config, cancelled=None):
        request = cancelled or threading.Event()
        try:
            yield from super().stream(config, request)
        finally:
            request.set()


class ReasoningGateway(ScriptedGateway):
    """A reasoning model that first returns hidden thinking only."""

    def __init__(self, silent: bool = False):
        super().__init__([])
        self.silent = silent
        self.calls: list[dict] = []

    def stream(self, config, cancelled=None):
        self.calls.append(copy.deepcopy(config))
        yield {"type": "start", "model_id": "fake-coder", "model": "fake"}
        if self.silent or len(self.calls) == 1:
            yield {"type": "reasoning", "text": "Read every file before changing anything."}
            yield {"type": "done", "reply": "", "finish_reason": "length",
                   "usage": {"prompt_tokens": 12,
                             "completion_tokens": config["max_tokens"]}}
            return
        reply = '{"final": "Restructured the vault."}'
        yield {"type": "reasoning", "text": "The structure is clear now."}
        yield {"type": "delta", "text": reply}
        yield {"type": "done", "reply": reply, "finish_reason": "stop",
               "usage": {"prompt_tokens": 12, "completion_tokens": 8}}


class CheckpointGateway(ScriptedGateway):
    """Run one tool round, then hold the next model call for inspection."""

    def __init__(self):
        super().__init__([])
        self.blocked = threading.Event()
        self.release = threading.Event()

    def stream(self, config, cancelled=None):
        self.calls.append(copy.deepcopy(config))
        yield {"type": "start", "model_id": "fake-coder", "model": "fake"}
        if len(self.calls) == 1:
            reply = '{"tool": "file.read", "input": {"path": "main.py"}}'
            yield {"type": "delta", "text": reply}
            yield {"type": "done", "reply": reply, "finish_reason": "stop",
                   "usage": {"prompt_tokens": 5, "completion_tokens": 3}}
            return
        self.blocked.set()
        self.release.wait(5)
        reply = '{"final": "Finished after the checkpoint."}'
        yield {"type": "delta", "text": reply}
        yield {"type": "done", "reply": reply, "finish_reason": "stop",
               "usage": {"prompt_tokens": 5, "completion_tokens": 3}}


class FlakyGateway(ScriptedGateway):
    """Fail the first request as if the endpoint dropped the connection."""

    def __init__(self):
        super().__init__([])
        self.calls: list[dict] = []

    def stream(self, config, cancelled=None):
        self.calls.append(copy.deepcopy(config))
        if len(self.calls) == 1:
            raise UnavailableError("endpoint offline")
        yield {"type": "start", "model_id": "fake-coder", "model": "fake"}
        reply = '{"final": "Recovered from a network failure."}'
        yield {"type": "delta", "text": reply}
        yield {"type": "done", "reply": reply, "finish_reason": "stop",
               "usage": {"prompt_tokens": 5, "completion_tokens": 3}}


@pytest.fixture()
def service(tmp_path):
    instance = IdeService(tmp_path, ScriptedGateway([]), skills_roots=[])
    yield instance
    instance.close()


def make_workspace(service, name="demo"):
    return service.create_workspace(name, template="python")


def run_until_finished(service, run_id, timeout=10.0):
    return wait_for(lambda: (lambda snap: snap if snap["status"] in FINISHED else None)(
        service.get_run(run_id)), timeout=timeout)


def test_language_detection_covers_popular_extensions_and_filenames():
    assert ide_module._language(Path("app.py")) == "Python"
    assert ide_module._language(Path("web.tsx")) == "TypeScript"
    assert ide_module._language(Path("main.rs")) == "Rust"
    assert ide_module._language(Path("Dockerfile")) == "Dockerfile"
    assert ide_module._language(Path("CMakeLists.txt")) == "CMake"
    assert ide_module._language(Path("styles.scss")) == "SCSS"
    assert ide_module._language(Path("query.graphql")) == "GraphQL"
    assert ide_module._language(Path("notes.unknown")) == "UNKNOWN"


def test_workspace_files_expose_source_image_hex_and_metadata_previews(service):
    workspace = make_workspace(service)
    service.write_file(workspace["id"], "notes.md", "# Plan\n\n- one\n", version="")
    service.write_file(workspace["id"], "flow.mmd", "flowchart LR\n  A[Start] --> B[Done]\n", version="")
    service.write_file(workspace["id"], "map.kml", "<kml><Placemark/></kml>\n", version="")
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    )
    (service.workspace_path(workspace["id"]) / "dot.png").write_bytes(png)

    markdown = service.preview_file(workspace["id"], "notes.md")
    assert markdown["kind"] == "text" and markdown["render"] == "markdown"
    assert markdown["content"].startswith("# Plan") and markdown["version"]
    assert markdown["hex"].startswith("00000000") and markdown["hex_bytes"] > 0

    mermaid = service.preview_file(workspace["id"], "flow.mmd")
    assert mermaid["language"] == "Mermaid" and mermaid["render"] == "mermaid"
    assert ide_module._language(Path("map.kml")) == "KML"
    kml = service.preview_file(workspace["id"], "map.kml")
    assert kml["kind"] == "text" and "<kml>" in kml["content"]

    image = service.preview_file(workspace["id"], "dot.png")
    assert image["kind"] == "binary" and image["mime"] == "image/png"
    assert image["dimensions"] == {"width": 1, "height": 1}
    assert image["data_url"].startswith("data:image/png;base64,") and image["editable"] is False

    with pytest.raises(NotFoundError):
        service.preview_file(workspace["id"], "missing.png")

    service.gateway = ScriptedGateway([
        '{"tool": "file.inspect", "input": {"path": "dot.png"}}',
        '{"final": "Inspected the image."}',
    ])
    started = service.start_run(workspace["id"], {"message": "Inspect the image", "mode": "ask"})
    final = run_until_finished(service, started["run"]["id"])
    assert final["status"] == "completed", final.get("error")
    result = [event for event in final["events"] if event["type"] == "tool_result"][0]
    assert result["ok"] is True and result["output"]["mime"] == "image/png"
    assert result["output"]["dimensions"] == {"width": 1, "height": 1}
    assert result["output"]["hex"].startswith("00000000") and "data_url" not in result["output"]


def test_workspace_and_user_skills_are_discovered_and_read(tmp_path):
    user_skills = tmp_path / "user-skills"
    (user_skills / "code-review").mkdir(parents=True)
    (user_skills / "code-review" / "SKILL.md").write_text(
        "---\nname: Code review\ndescription: Review a diff for regressions\n---\n# Review\nCheck tests.\n",
        encoding="utf-8",
    )
    service = IdeService(tmp_path / "studio", ScriptedGateway([]), skills_roots=[user_skills])
    try:
        workspace = make_workspace(service)
        skill_dir = service.workspace_path(workspace["id"]) / ".agents" / "skills" / "release"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: Release flow\ndescription: Ship the package safely\n---\nStep one.\n",
            encoding="utf-8",
        )
        skills = service.list_skills(workspace["id"])
        assert [item["id"] for item in skills] == ["release", "code-review"]
        assert skills[0]["scope"] == "workspace" and skills[1]["scope"] == "user"
        assert skills[1]["name"] == "Code review"

        content = service.read_skill(workspace["id"], "Code review")
        assert content["id"] == "code-review" and "Check tests." in content["content"]
        with pytest.raises(NotFoundError, match="Available skills"):
            service.read_skill(workspace["id"], "missing")
        tool_ids = {tool["id"] for tool in service._tools_for_mode("edit", workspace["id"])}
        assert {"skill.read", "file.inspect"} <= tool_ids
        prompt = service._system_prompt(
            workspace["id"], "edit", service._tools_for_mode("edit", workspace["id"]),
            service.workspace_path(workspace["id"]), None,
        )
        assert "available automatically" in prompt
        assert "skill.read" in prompt and "release" in prompt and "code-review" in prompt
    finally:
        service.close()


def test_every_skill_is_exposed_automatically_in_the_prompt(service):
    workspace = make_workspace(service)
    root = service.workspace_path(workspace["id"]) / ".agents" / "skills"
    for index in range(14):
        directory = root / f"skill-{index:02d}"
        directory.mkdir(parents=True)
        (directory / "SKILL.md").write_text(
            f"---\nname: Skill {index}\ndescription: Handles task {index}\n---\nBody.\n",
            encoding="utf-8",
        )
    tools = service._tools_for_mode("edit", workspace["id"])
    prompt = service._system_prompt(
        workspace["id"], "edit", tools, service.workspace_path(workspace["id"]), None)
    assert "available automatically" in prompt
    assert "skill-13" in prompt and prompt.count("- skill-") == 14
    briefing = service._swarm_briefing(workspace["id"], [])
    assert "available automatically" in briefing and "skill-13" in briefing


def test_reading_a_skill_through_a_run_returns_its_instructions(service, tmp_path):
    workspace = make_workspace(service)
    skill_dir = service.workspace_path(workspace["id"]) / ".agents" / "skills" / "diagnose"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("# Diagnose\nRun the smallest reproduction first.\n", encoding="utf-8")
    service.gateway = ScriptedGateway([
        '{"tool": "skill.read", "input": {"name": "diagnose"}}',
        '{"final": "Used the skill."}',
    ])
    started = service.start_run(workspace["id"], {"message": "Use the diagnosis skill", "mode": "edit"})
    final = run_until_finished(service, started["run"]["id"])
    assert final["status"] == "completed", final.get("error")
    results = [event for event in final["events"] if event["type"] == "tool_result"]
    assert results[0]["ok"] is True and "smallest reproduction" in results[0]["output"]["content"]
    assert "skill.read" in next(event for event in final["events"] if event["type"] == "run_started")["tools"]


def test_successful_edits_report_diagnostics_without_a_separate_tool_call(service):
    workspace = make_workspace(service)
    service.gateway = ScriptedGateway([
        '{"tool": "file.write", "input": {"path": "main.py", "content": "def broken(:\\n"}}',
        '{"final": "Wrote it."}',
    ])
    started = service.start_run(workspace["id"], {"message": "Write a broken file", "mode": "edit"})
    final = run_until_finished(service, started["run"]["id"])
    assert final["status"] == "completed", final.get("error")
    result = [event for event in final["events"] if event["type"] == "tool_result"][0]
    assert result["output"]["diagnostics"]["ok"] is False
    assert result["output"]["diagnostics"]["issues"][0]["path"] == "main.py"


def test_deepseek_dsml_tool_calls_run_in_the_ide(service):
    workspace = make_workspace(service)
    gateway = ScriptedGateway([
        '<｜｜DSML｜｜ calls>\n'
        '<｜｜DSML｜｜ invoke name="calls">\n'
        '<｜｜DSML｜｜ parameter name="input" string="false">{"path": "main.py"}</｜｜DSML｜｜ parameter>\n'
        '<｜｜DSML｜｜ parameter name="tool" string="true">file.read</｜｜DSML｜｜ parameter>\n'
        '</｜｜DSML｜｜ invoke>\n'
        '</｜｜DSML｜｜ calls>',
        '{"final": "main.py defines greet."}',
    ])
    service.gateway = gateway
    started = service.start_run(workspace["id"], {"message": "Explain the project", "mode": "ask"})
    final = run_until_finished(service, started["run"]["id"])
    assert final["status"] == "completed", final.get("error")
    calls = [event for event in final["events"] if event["type"] == "tool_call"]
    assert calls and calls[0]["tool"] == "file.read" and calls[0]["input"]["path"] == "main.py"
    results = [event for event in final["events"] if event["type"] == "tool_result"]
    assert results and results[0]["ok"] is True
    assert "def greet" in json.dumps(results[0]["output"])
    assert final["output"] == "main.py defines greet."
    assert any(tool["id"] == "file.read" for tool in gateway.calls[0]["tools"])


def test_tool_calls_accept_model_argument_and_name_spellings(service):
    workspace = make_workspace(service)
    service.gateway = ScriptedGateway([
        '{"tool": "file_read", "input": {"file_path": "main.py", "startLine": "1", "to": 2}}',
        '{"final": "Read the first two lines."}',
    ])
    started = service.start_run(workspace["id"], {"message": "Read main.py", "mode": "ask"})
    final = run_until_finished(service, started["run"]["id"])
    assert final["status"] == "completed", final.get("error")
    call = next(event for event in final["events"] if event["type"] == "tool_call")
    assert call["tool"] == "file.read"
    assert call["input"] == {"path": "main.py", "start_line": 1, "end_line": 2}
    result = next(event for event in final["events"] if event["type"] == "tool_result")
    assert result["ok"] is True
    assert result["output"]["start_line"] == 1 and result["output"]["end_line"] == 2


def test_an_incomplete_tool_call_is_repaired_before_it_runs(service):
    workspace = make_workspace(service)
    service.gateway = ScriptedGateway([
        '{"tool": "file.search", "input": {}}',
        '{"tool": "file.search", "input": {"q": "def greet"}}',
        '{"final": "Found the definition."}',
    ])
    started = service.start_run(workspace["id"], {"message": "Find greet", "mode": "ask"})
    final = run_until_finished(service, started["run"]["id"])
    assert final["status"] == "completed", final.get("error")
    calls = [event for event in final["events"] if event["type"] == "tool_call"]
    results = [event for event in final["events"] if event["type"] == "tool_result"]
    assert len(calls) == 1 and calls[0]["tool"] == "file.search"
    assert calls[0]["input"]["query"] == "def greet"
    assert len(results) == 1 and results[0]["ok"] is True
    assert any("missing required input" in event.get("message", "")
               for event in final["events"] if event["type"] == "log")


def test_a_stubborn_incomplete_tool_call_is_repaired_only_twice(service):
    workspace = make_workspace(service)
    service.gateway = ScriptedGateway([
        '{"tool": "file.search", "input": {}}',
        '{"tool": "file.search", "input": {}}',
        '{"tool": "file.search", "input": {}}',
        '{"tool": "file.search", "input": {}}',
        '{"final": "Could not repair the call."}',
    ])
    started = service.start_run(workspace["id"], {"message": "Find greet", "mode": "ask"})
    final = run_until_finished(service, started["run"]["id"])
    assert final["status"] == "completed", final.get("error")
    repairs = [event for event in final["events"]
               if event["type"] == "log"
               and "missing required input" in event.get("message", "")]
    assert len(repairs) == MAX_FORMAT_REPAIRS
    results = [event for event in final["events"] if event["type"] == "tool_result"]
    assert results and results[0]["ok"] is False
    assert "query" in results[0]["output"]["error"]


def test_a_run_publishes_live_context_usage(service):
    workspace = make_workspace(service)
    service.gateway = ScriptedGateway([
        '{"tool": "file.read", "input": {"path": "main.py"}}',
        '{"final": "Done."}',
    ])
    started = service.start_run(workspace["id"], {"message": "Read main.py", "mode": "ask"})
    final = run_until_finished(service, started["run"]["id"])
    assert final["status"] == "completed", final.get("error")
    budgets = [event for event in final["events"] if event["type"] == "budget"]
    assert budgets and budgets[-1]["used_tokens"] > 0
    assert budgets[-1]["context_window"] > 0
    assert 0 <= budgets[-1]["percent"] <= 1000


def test_workspace_files_are_sandboxed_and_versioned(service):
    workspace = make_workspace(service)
    tree = service.tree(workspace["id"])
    assert tree["workspace"]["name"] == "demo" and tree["files"] >= 3
    assert "main.py" in {node["path"] for node in tree["children"] if node["type"] == "file"}

    data = service.read_file(workspace["id"], "main.py")
    assert data["language"] == "Python" and data["version"]

    with pytest.raises(ConflictError, match="changed on disk"):
        service.write_file(workspace["id"], "main.py", "changed\n", version="stale")
    saved = service.write_file(workspace["id"], "main.py", "changed\n", version=data["version"])
    assert saved["version"] != data["version"]
    assert service.read_file(workspace["id"], "main.py")["content"] == "changed\n"

    with pytest.raises(BadRequestError, match="inside this IDE workspace"):
        service.read_file(workspace["id"], "../outside.txt")
    with pytest.raises(BadRequestError, match="manifest"):
        service.read_file(workspace["id"], ".mariana-ide.json")
    with pytest.raises(NotFoundError):
        service.read_file(workspace["id"], "missing.py")

    service.delete_file(workspace["id"], "main.py")
    with pytest.raises(NotFoundError):
        service.read_file(workspace["id"], "main.py")


def test_saving_a_file_keeps_its_permissions(service):
    workspace = make_workspace(service)
    path = service.workspace_path(workspace["id"]) / "run.sh"
    path.write_text("#!/bin/sh\n", encoding="utf-8")
    path.chmod(0o755)

    current = service.read_file(workspace["id"], "run.sh")
    service.write_file(workspace["id"], "run.sh", "#!/bin/sh\necho hi\n", version=current["version"])
    assert path.stat().st_mode & 0o777 == 0o755


def test_files_keep_their_line_endings(service):
    workspace = make_workspace(service)
    service.write_file(workspace["id"], "crlf.txt", "one\r\ntwo\r\n", version="")
    assert service.read_file(workspace["id"], "crlf.txt")["content"] == "one\r\ntwo\r\n"


def test_a_tree_deeper_than_the_limit_is_marked_truncated(service):
    workspace = make_workspace(service)
    nested = "/".join("abcdefghi") + "/deep.txt"
    service.write_file(workspace["id"], nested, "deep\n", version="")

    assert service.tree(workspace["id"])["truncated"] is True


def test_lazy_tree_range_reads_and_name_lookup(service):
    workspace = make_workspace(service)
    service.create_folder(workspace["id"], "src/components")
    service.write_file(workspace["id"], "src/components/button.py", "a\nb\nc\nd\n", version="")

    root = service.tree(workspace["id"], depth=1)
    src = next(node for node in root["children"] if node["path"] == "src")
    assert src["loaded"] is False and src["has_children"] is True
    assert src["children"] == []

    level = service.tree(workspace["id"], "src/components", depth=1)
    assert [node["path"] for node in level["children"]] == ["src/components/button.py"]

    partial = service.read_file(workspace["id"], "src/components/button.py", start=1, limit=2)
    assert partial["content"] == "b\nc"
    assert partial["total_lines"] == 5 and partial["truncated"] is True
    assert partial["start_line"] == 2 and partial["end_line"] == 3

    found = service.find_files(workspace["id"], "button", 50)
    assert [item["path"] for item in found["files"]] == ["src/components/button.py"]
    assert "src" in {item["path"] for item in service.find_files(workspace["id"], "", 50)["folders"]}
    assert "src" in {item["path"] for item in service.find_files(workspace["id"], "src", 50)["folders"]}


def test_context_rules_hide_paths_from_the_map_tree_and_search(service):
    workspace = make_workspace(service)
    service.create_folder(workspace["id"], "vendor")
    service.write_file(workspace["id"], "vendor/generated.py", "greet = 'hidden'\n", version="")

    settings = service.update_settings(workspace["id"], {"excluded": ["vendor"]})
    assert settings["excluded"] == ["vendor"]
    assert service.settings(workspace["id"])["excluded"] == ["vendor"]

    tree = service.tree(workspace["id"])
    assert "vendor" not in {node["path"] for node in tree["children"]}
    found = service.search(workspace["id"], {"query": "hidden"})
    assert found["matches"] == []
    assert "vendor" not in {item["path"] for item in service.find_files(workspace["id"], "", 50)["folders"]}

    # file.list returns the same directory twice (entries and the files subset),
    # and both must drop a hidden path so the model never sees it.
    listed = {
        "entries": [{"name": "main.py", "path": "main.py", "type": "file"},
                    {"name": "secret.py", "path": "vendor/secret.py", "type": "file"},
                    {"name": "win.py", "path": "vendor\\win.py", "type": "file"}],
        "files": [{"name": "main.py", "path": "main.py"},
                  {"name": "secret.py", "path": "vendor/secret.py"}],
    }
    filtered = service._filter_observation("file.list", listed, ["vendor"])
    assert [item["path"] for item in filtered["entries"]] == ["main.py"]
    assert [item["path"] for item in filtered["files"]] == ["main.py"]
    assert "2 entries hidden" in filtered["context_note"]

    service.update_settings(workspace["id"], {"context": {"include": ["main.py"], "exclude": []}})
    preview = service.context_preview(workspace["id"], {})
    assert preview["selected"] == 1 and preview["files"][0]["path"] == "main.py"
    assert preview["tokens"] > 0
    assert service.settings(workspace["id"])["context"]["include"] == ["main.py"]

    with pytest.raises(BadRequestError):
        service.update_settings(workspace["id"], {"excluded": ["../outside"]})


def test_folder_operations_and_renames_stay_inside_the_workspace(service):
    workspace = make_workspace(service)
    created = service.create_folder(workspace["id"], "pkg/nested")
    assert created["path"] == "pkg/nested" and created["created"]

    service.write_file(workspace["id"], "pkg/nested/old.py", "value = 1\n", version="")
    moved = service.move_entry(workspace["id"], "pkg/nested/old.py", "pkg/nested/new.py")
    assert moved["to"] == "pkg/nested/new.py"
    assert service.read_file(workspace["id"], "pkg/nested/new.py")["content"] == "value = 1\n"

    with pytest.raises(ConflictError):
        service.move_entry(workspace["id"], "pkg/nested/new.py", "pkg/nested/new.py")
    with pytest.raises(BadRequestError):
        service.move_entry(workspace["id"], "pkg", "pkg/nested/pkg")
    with pytest.raises(BadRequestError):
        service.create_folder(workspace["id"], "../escape")

    deleted = service.delete_folder(workspace["id"], "pkg")
    assert deleted["deleted"]
    assert "pkg" not in {node["path"] for node in service.tree(workspace["id"])["children"]}


def test_sessions_can_be_renamed_pinned_and_duplicated(service):
    workspace = make_workspace(service)
    session = service._new_session(workspace["id"], "First task", "edit", "fake-coder")
    session["transcript"].append({"role": "user", "text": "First task"})
    service._save_session(session)

    renamed = service.rename_session(session["id"], {"title": "Release notes", "pinned": True})
    assert renamed["title"] == "Release notes" and renamed["pinned"] is True
    assert service.list_sessions(workspace["id"])[0]["pinned"] is True
    service.rename_session(session["id"], {"context": {"include": ["main.py"], "exclude": []}})
    assert service.get_session(session["id"])["context"]["include"] == ["main.py"]

    copy_summary = service.duplicate_session(session["id"])
    assert copy_summary["id"] != session["id"] and copy_summary["title"].endswith("(copy)")
    copy_document = service.get_session(copy_summary["id"], include_messages=True)
    assert copy_document["transcript"] == session["transcript"]
    assert copy_document["changes"] == {}

    with pytest.raises(BadRequestError):
        service.rename_session(session["id"], {"title": "   "})


def test_the_session_list_cache_refreshes_after_a_save(service):
    workspace = make_workspace(service)
    session = service._new_session(workspace["id"], "First task", "edit", "fake-coder")
    assert service.list_sessions(workspace["id"])[0]["title"] == "First task"
    # The cached file is re-read only after it actually changes.
    assert service.list_sessions(workspace["id"])[0]["title"] == "First task"

    session["title"] = "Changed task"
    service._save_session(session)
    assert service.list_sessions(workspace["id"])[0]["title"] == "Changed task"


def test_a_busy_workspace_does_not_hide_another_projects_sessions(service):
    kept = make_workspace(service, "kept")
    other = make_workspace(service, "other")
    target = service._new_session(kept["id"], "Older conversation", "edit", "fake-coder")
    os.utime(service._session_path(target["id"]), (1_000_000, 1_000_000))
    for index in range(205):
        session = service._new_session(other["id"], f"Noise {index}", "edit", "fake-coder")
        stamp = 2_000_000 + index
        os.utime(service._session_path(session["id"]), (stamp, stamp))

    # The older conversation is still listed even though every session in the
    # recent global window belongs to the other, busier project.
    assert [item["id"] for item in service.list_sessions(kept["id"])] == [target["id"]]


def test_pinned_context_and_hidden_paths_reach_the_prompt(service):
    workspace = make_workspace(service)
    service.create_folder(workspace["id"], "vendor")
    service.write_file(workspace["id"], "vendor/secret.py", "VENDOR_SECRET = 'hidden'\n", version="")
    service.update_settings(workspace["id"], {"excluded": ["vendor"]})

    gateway = ScriptedGateway(['{"final": "done"}'])
    service.gateway = gateway
    started = service.start_run(workspace["id"], {
        "message": "Use the pinned file", "mode": "ask",
        "context": {"include": ["main.py"], "exclude": ["vendor"]},
    })
    run_until_finished(service, started["run"]["id"])
    prompt = gateway.calls[0]["messages"][0]["content"]
    assert "Pinned context: main.py" in prompt
    assert "def greet" in prompt
    assert "vendor/secret.py" not in prompt
    assert "VENDOR_SECRET" not in prompt
    assert "excluded this context" in prompt

    session = service.get_session(started["session"]["id"])
    assert session["context"]["include"] == ["main.py"]


def test_pinning_one_file_does_not_shrink_the_project_map(service):
    workspace = make_workspace(service)
    gateway = ScriptedGateway(['{"final": "done"}'])
    service.gateway = gateway
    started = service.start_run(workspace["id"], {
        "message": "Explain the pinned file", "mode": "ask",
        "context": {"include": ["main.py"], "exclude": []},
    })
    run_until_finished(service, started["run"]["id"])
    prompt = gateway.calls[0]["messages"][0]["content"]
    assert "Pinned context: main.py" in prompt
    # Pinned files add their contents; they must not filter the project map.
    assert "test_main.py" in prompt
    assert "README.md" in prompt


def test_different_workspaces_can_run_coding_agents_at_the_same_time(service):
    first = make_workspace(service, "first")
    second = make_workspace(service, "second")
    gateway = BlockingGateway()
    service.gateway = gateway
    started_first = service.start_run(first["id"], {"message": "One", "mode": "ask"})
    started_second = service.start_run(second["id"], {"message": "Two", "mode": "ask"})
    assert started_first["run"]["id"] != started_second["run"]["id"]
    gateway.release.set()
    assert run_until_finished(service, started_first["run"]["id"])["status"] == "completed"
    assert run_until_finished(service, started_second["run"]["id"])["status"] == "completed"


def test_search_diagnostics_and_import(service, tmp_path):
    workspace = make_workspace(service)
    service.write_file(workspace["id"], "bad.py", "def broken(:\n    pass\n", force=True)
    found = service.search(workspace["id"], {"query": "greet", "regex": False})
    assert any(match["path"] == "main.py" for match in found["matches"])
    report = service.diagnostics(workspace["id"])
    assert report["ok"] is False
    assert any(issue["path"] == "bad.py" and issue["rule"] == "syntax"
               for issue in report["issues"])

    source = tmp_path / "project"
    (source / "pkg").mkdir(parents=True)
    (source / "pkg" / "app.py").write_text("print('hello')\n", encoding="utf-8")
    (source / "node_modules").mkdir()
    (source / "node_modules" / "huge.js").write_text("ignored\n", encoding="utf-8")
    imported = service.create_workspace("Imported", source=str(source))
    tree = service.tree(imported["id"])
    paths = {node["path"] for node in tree["children"]}
    assert "pkg" in paths and "node_modules" not in paths


def test_coding_run_streams_tools_plan_edits_and_reverts(service):
    workspace = make_workspace(service)
    original = service.read_file(workspace["id"], "main.py")["content"]
    service.gateway = ScriptedGateway([
        '{"tool": "file.read", "input": {"path": "main.py"}}',
        '{"plan": [{"text": "Update the greeting", "status": "active"}]}',
        '{"tool": "file.edit", "input": {"path": "main.py", '
        '"old": "return f\\"Hello, {name}!\\"", "new": "return f\\"Hi, {name}!\\""}}',
        '{"final": "Changed the greeting and verified nothing else."}',
    ])
    started = service.start_run(workspace["id"], {"message": "Say hi instead", "mode": "edit"})
    run_id = started["run"]["id"]
    final = run_until_finished(service, run_id)
    assert final["status"] == "completed", final.get("error")
    assert "Changed the greeting" in final["output"]
    kinds = [event["type"] for event in final["events"]]
    assert {"run_started", "assistant_delta", "tool_call", "tool_result", "plan",
            "file_changed", "run_completed"} <= set(kinds)
    assert "Hi, {name}" in service.read_file(workspace["id"], "main.py")["content"]

    session = service.get_session(started["session"]["id"])
    assert any(entry["role"] == "tool" and entry["tool"] == "file.edit"
               for entry in session["transcript"])
    change = session["changes"]["main.py"]
    assert change["edits"] == 1 and change["added"] >= 1 and change["removed"] >= 1
    assert session["plan"][0]["text"] == "Update the greeting"

    result = service.revert_session(session["id"])
    assert result["restored"] == 1
    assert service.read_file(workspace["id"], "main.py")["content"] == original


def test_a_late_event_cannot_arrive_after_a_run_finishes():
    run = IdeRun("run-late", "session-late", "workspace-late", "go")
    run.event("log", message="worker started")
    run.finish("completed", output="done")
    snapshot = run.snapshot()
    count = snapshot["event_count"]

    run.event("log", message="late worker")

    after = run.snapshot()
    assert after["event_count"] == count
    assert [event["type"] for event in after["events"]] == [
        "log", "run_completed"
    ]


def test_independent_read_tools_run_in_parallel_and_keep_result_order(service, monkeypatch):
    workspace = make_workspace(service)
    for name in ("a.py", "b.py", "c.py"):
        service.write_file(workspace["id"], name, f"# {name}\n", version="")

    lock = threading.Lock()
    active = 0
    peak = 0
    real = ide_module.execute_ability

    def slow(*args, **kwargs):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            time.sleep(0.08)
            return real(*args, **kwargs)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(ide_module, "execute_ability", slow)
    service.gateway = ScriptedGateway([
        '{"calls": ['
        '{"tool": "file.read", "input": {"path": "a.py"}}, '
        '{"tool": "file.read", "input": {"path": "b.py"}}, '
        '{"tool": "file.read", "input": {"path": "c.py"}}]}',
        '{"final": "Read them."}',
    ])
    started = service.start_run(workspace["id"], {"message": "Read three files", "mode": "ask"})
    final = run_until_finished(service, started["run"]["id"])

    assert final["status"] == "completed", final.get("error")
    assert peak >= 2
    results = [event for event in final["events"] if event["type"] == "tool_result"]
    assert [result["output"]["path"] for result in results] == ["a.py", "b.py", "c.py"]


def test_cancelling_a_parallel_batch_does_not_wait_for_a_blocked_tool(service):
    workspace = make_workspace(service)
    run = IdeRun("run-cancel", "session-cancel", workspace["id"], "go")
    started = threading.Event()
    release = threading.Event()

    def blocked(*_args, **_kwargs):
        started.set()
        release.wait(5)
        return {"content": "late"}, True, None, {}, 1

    original = service._invoke_with_retry
    service._invoke_with_retry = blocked
    tools = [{"id": "file.read", "type": "file.read"},
             {"id": "file.list", "type": "file.list"}]
    calls = [{"tool": "file.read", "input": {"path": "main.py"}},
             {"tool": "file.list", "input": {}}]
    outcome = []

    def batch():
        try:
            service._run_tool_batch(
                run, workspace["id"], {"mode": "ask"}, tools, calls, 1,
                Counter(), [])
        except ide_module._Cancelled:
            outcome.append("cancelled")

    worker = threading.Thread(target=batch, daemon=True)
    try:
        worker.start()
        assert started.wait(2)
        run.cancelled.set()
        worker.join(timeout=2)
    finally:
        release.set()
        service._invoke_with_retry = original

    assert outcome == ["cancelled"]
    assert worker.is_alive() is False


def test_a_transient_read_failure_is_retried_once(service, monkeypatch):
    workspace = make_workspace(service)
    service.write_file(workspace["id"], "x.py", "value = 1\n", version="")
    calls = []
    real = ide_module.execute_ability

    def flaky(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise TimeoutError("connection timed out")
        return real(*args, **kwargs)

    monkeypatch.setattr(ide_module, "execute_ability", flaky)
    service.gateway = ScriptedGateway([
        '{"tool": "file.read", "input": {"path": "x.py"}}',
        '{"final": "Recovered."}',
    ])
    started = service.start_run(workspace["id"], {"message": "Read it", "mode": "ask"})
    final = run_until_finished(service, started["run"]["id"])

    assert final["status"] == "completed", final.get("error")
    results = [event for event in final["events"] if event["type"] == "tool_result"]
    assert len(results) == 1 and results[0]["attempts"] == 2 and results[0]["ok"] is True
    assert any(event["type"] == "log"
               and "Retrying file.read" in event.get("message", "")
               for event in final["events"])


def test_a_batch_with_a_write_keeps_call_order(service):
    workspace = make_workspace(service)
    service.gateway = ScriptedGateway([
        '{"calls": ['
        '{"tool": "file.write", "input": {"path": "created.py", "content": "value = 1\\n"}}, '
        '{"tool": "file.read", "input": {"path": "created.py"}}]}',
        '{"final": "Wrote and read it."}',
    ])
    started = service.start_run(workspace["id"], {"message": "Write then read", "mode": "edit"})
    final = run_until_finished(service, started["run"]["id"])

    assert final["status"] == "completed", final.get("error")
    results = [event for event in final["events"] if event["type"] == "tool_result"]
    assert [result["ok"] for result in results] == [True, True]
    assert "value = 1" in results[1]["output"]["content"]


def test_a_reasoning_only_reply_is_retried_with_a_larger_budget(service):
    workspace = make_workspace(service)
    gateway = ReasoningGateway()
    service.gateway = gateway
    started = service.start_run(workspace["id"], {
        "message": "Restructure the notes", "mode": "edit",
    })
    final = run_until_finished(service, started["run"]["id"])

    assert final["status"] == "completed", final.get("error")
    assert final["output"] == "Restructured the vault."
    assert len(gateway.calls) == 2
    assert gateway.calls[1]["max_tokens"] == gateway.calls[0]["max_tokens"] * 2
    assert gateway.calls[1]["reasoning_effort"] == "low"
    ends = [event for event in final["events"] if event["type"] == "assistant_end"]
    assert ends[0]["empty"] is True and ends[0]["retrying"] is True
    assert ends[-1]["empty"] is False and ends[-1]["retrying"] is False
    assert ends[-1]["reasoning"] == "The structure is clear now."
    messages = [event["message"] for event in final["events"] if event["type"] == "log"]
    assert any("no visible text" in message for message in messages)

    session = service.get_session(started["session"]["id"])
    thinking = [entry for entry in session["transcript"] if entry["role"] == "thinking"]
    assert len(thinking) == 1 and "structure is clear" in thinking[0]["text"]


def test_a_silent_reasoning_model_fails_with_actionable_guidance(service):
    workspace = make_workspace(service)
    gateway = ReasoningGateway(silent=True)
    service.gateway = gateway
    started = service.start_run(workspace["id"], {
        "message": "Restructure the notes", "mode": "edit",
    })
    final = run_until_finished(service, started["run"]["id"])

    assert len(gateway.calls) == 2
    assert final["status"] == "failed"
    assert "output budget" in final["error"]
    transcript = service.get_session(started["session"]["id"])["transcript"]
    assert any(entry["role"] == "error" for entry in transcript)
    assert not any(entry["role"] == "assistant" for entry in transcript)


def test_a_long_run_extends_its_step_budget_when_it_is_making_progress(service):
    workspace = make_workspace(service)
    service.gateway = ScriptedGateway([
        '{"tool": "file.read", "input": {"path": "main.py"}}',
        '{"tool": "file.read", "input": {"path": "test_main.py"}}',
        '{"tool": "file.read", "input": {"path": "main.py", '
        '"start_line": 1, "end_line": 2}}',
        '{"final": "Read everything."}',
    ])
    started = service.start_run(workspace["id"], {
        "message": "Read the project", "mode": "ask", "max_steps": 2,
    })
    final = run_until_finished(service, started["run"]["id"])

    assert final["status"] == "completed", final.get("error")
    assert final["output"] == "Read everything."
    logs = [event["message"] for event in final["events"] if event["type"] == "log"]
    assert any("Extended the step budget" in message for message in logs)


def test_a_stalled_long_run_stops_at_its_step_budget(service):
    workspace = make_workspace(service)
    service.gateway = ScriptedGateway([
        '{"tool": "file.read", "input": {"path": "main.py"}}',
        '{"tool": "file.read", "input": {"path": "main.py"}}',
        '{"final": "never reached"}',
    ])
    started = service.start_run(workspace["id"], {
        "message": "Loop on one file", "mode": "ask", "max_steps": 2,
    })
    final = run_until_finished(service, started["run"]["id"])

    assert final["status"] == "completed"
    assert final["exhausted"] is True
    assert "without a final answer" in final["output"]
    logs = [event["message"] for event in final["events"] if event["type"] == "log"]
    assert not any("Extended the step budget" in message for message in logs)


def test_a_malformed_action_is_repaired_instead_of_ending_the_run(service):
    workspace = make_workspace(service)
    service.gateway = ScriptedGateway([
        '{"tool": "file.read", "input": {"path": "main.py"',
        '{"final": "Recovered after the repair."}',
    ])
    started = service.start_run(workspace["id"], {"message": "Read main", "mode": "ask"})
    final = run_until_finished(service, started["run"]["id"])

    assert final["status"] == "completed", final.get("error")
    assert final["output"] == "Recovered after the repair."
    logs = [event["message"] for event in final["events"] if event["type"] == "log"]
    assert any("malformed" in message for message in logs)


def test_an_empty_final_answer_is_asked_for_again(service):
    workspace = make_workspace(service)
    service.gateway = ScriptedGateway([
        '{"final": ""}',
        '{"final": "The work is done."}',
    ])
    started = service.start_run(workspace["id"], {"message": "Finish", "mode": "ask"})
    final = run_until_finished(service, started["run"]["id"])

    assert final["status"] == "completed", final.get("error")
    assert final["output"] == "The work is done."
    logs = [event["message"] for event in final["events"] if event["type"] == "log"]
    assert any("empty summary" in message for message in logs)


def test_repeated_malformed_actions_fail_with_guidance(service):
    workspace = make_workspace(service)
    service.gateway = ScriptedGateway(['{"tool":', '{"tool":', '{"tool":'])
    started = service.start_run(workspace["id"], {"message": "Break the format", "mode": "ask"})
    final = run_until_finished(service, started["run"]["id"])

    assert final["status"] == "failed"
    assert "malformed actions" in final["error"]


# A real DeepSeek reply that lost the slashes on its closing tags.  It must be
# executed as a tool call, never stored or shown as the assistant's answer.
DAMAGED_MARKUP_READ = """<｜｜DSML｜｜ calls>
<｜｜DSML｜｜ invoke name="file.read">
<｜｜DSML｜｜ parameter name="end_line" string="false">3</｜｜DSML｜｜ parameter>
<｜｜DSML｜｜ parameter name="path" string="true">main.py</｜｜DSML｜｜ parameter>
<｜｜DSML｜｜ parameter name="start_line" string="false">1</｜｜DSML｜｜ parameter>
</｜｜DSML｜｜ parameter>
</｜｜DSML｜｜ parameter>
<｜｜DSML｜｜ invoke>
<｜｜DSML｜｜ calls>"""


def test_damaged_tool_markup_runs_the_tool_instead_of_ending_the_run(service):
    workspace = make_workspace(service)
    service.gateway = ScriptedGateway([
        DAMAGED_MARKUP_READ,
        '{"final": "main.py starts with the greeting."}',
    ])
    started = service.start_run(workspace["id"], {"message": "Read main", "mode": "ask"})
    final = run_until_finished(service, started["run"]["id"])

    assert final["status"] == "completed", final.get("error")
    assert final["output"] == "main.py starts with the greeting."
    results = [event for event in final["events"] if event["type"] == "tool_result"]
    assert results and results[0]["tool"] == "file.read" and results[0]["ok"] is True
    assert "DSML" not in final["output"]


def test_repeated_tool_markup_fails_without_leaking_it(service):
    workspace = make_workspace(service)
    truncated = ('<｜｜DSML｜｜ parameter name="path" string="true">main.py'
                 '</｜｜DSML｜｜ parameter>\n<｜｜DSML')
    service.gateway = ScriptedGateway([truncated] * 3)
    started = service.start_run(workspace["id"], {"message": "Read main", "mode": "ask"})
    final = run_until_finished(service, started["run"]["id"])

    assert final["status"] == "failed"
    assert "malformed actions" in final["error"]
    assert "tool markup" in final["error"]
    assert "DSML" not in final["error"]


def test_unparseable_tool_markup_gets_a_repair_turn(service):
    workspace = make_workspace(service)
    truncated = ('<｜｜DSML｜｜ parameter name="path" string="true">main.py'
                 '</｜｜DSML｜｜ parameter>\n<｜｜DSML')
    service.gateway = ScriptedGateway([
        truncated,
        '{"final": "Recovered with a plain answer."}',
    ])
    started = service.start_run(workspace["id"], {"message": "Read main", "mode": "ask"})
    final = run_until_finished(service, started["run"]["id"])

    assert final["status"] == "completed", final.get("error")
    assert final["output"] == "Recovered with a plain answer."
    logs = [event["message"] for event in final["events"] if event["type"] == "log"]
    assert any("tool-call markup" in message for message in logs)


def test_markup_inside_a_final_answer_is_removed(service):
    workspace = make_workspace(service)
    service.gateway = ScriptedGateway([
        json.dumps({"final": "Found it.\n" + DAMAGED_MARKUP_READ}),
        '{"final": "not used"}',
    ])
    started = service.start_run(workspace["id"], {"message": "Report", "mode": "ask"})
    final = run_until_finished(service, started["run"]["id"])

    assert final["status"] == "completed", final.get("error")
    assert final["output"] == "Found it."
    assert "DSML" not in final["output"]


def test_a_batched_markup_reply_runs_every_call(service):
    workspace = make_workspace(service)
    service.write_file(workspace["id"], "second.py", "print('two')\n", version="")
    batch = ('<｜｜DSML｜｜ calls>\n'
             '[{"tool": "file.read", "input": {"path": "main.py"}}, '
             '{"tool": "file.read", "input": {"path": "second.py"}}]'
             '</｜｜DSML｜｜ parameter>\n</｜｜DSML｜｜ invoke>\n'
             '</｜｜DSML｜｜ calls>')
    service.gateway = ScriptedGateway([batch, '{"final": "Read both files."}'])
    started = service.start_run(workspace["id"], {"message": "Read both", "mode": "ask"})
    final = run_until_finished(service, started["run"]["id"])

    assert final["status"] == "completed", final.get("error")
    assert final["output"] == "Read both files."
    results = [event for event in final["events"] if event["type"] == "tool_result"]
    assert [event["tool"] for event in results] == ["file.read", "file.read"]
    assert all(event["ok"] for event in results)
    paths = [event["output"]["path"] for event in results]
    assert paths == ["main.py", "second.py"]


def test_a_swarm_internal_error_is_returned_to_the_model(service, monkeypatch):
    workspace = make_workspace(service)
    service.gateway = ScriptedGateway([
        '{"tool": "agent.swarm", "input": {}}',
        '{"final": "Continued after the swarm error."}',
    ])

    def explode(*args, **kwargs):
        raise RuntimeError("swarm exploded")

    monkeypatch.setattr(service, "_run_swarm", explode)
    started = service.start_run(workspace["id"], {"message": "Use a swarm", "mode": "ask"})
    final = run_until_finished(service, started["run"]["id"])

    assert final["status"] == "completed", final.get("error")
    assert final["output"] == "Continued after the swarm error."
    results = [event for event in final["events"] if event["type"] == "tool_result"]
    assert results and results[0]["ok"] is False
    assert "swarm exploded" in json.dumps(results[0]["output"])


def test_observation_truncation_keeps_file_control_fields(service):
    observation = {
        "path": "main.py", "content": "x" * 50_000,
        "start_line": 1, "end_line": 200, "total_lines": 900,
        "truncated": False, "language": "Python",
    }
    bounded = service._bounded_observation(observation, 6_000)

    assert bounded["path"] == "main.py"
    assert bounded["start_line"] == 1 and bounded["end_line"] == 1
    assert bounded["total_lines"] == 900
    assert bounded["truncated"] is True and bounded["content_cut_midline"] is True
    assert "content" in bounded["truncated_fields"]
    assert "next_start_line" not in bounded
    assert len(bounded["content"]) < 50_000
    assert len(json.dumps(bounded)) <= 6_200


def test_observation_truncation_reports_the_real_last_included_line(service):
    content = "".join(
        f"value_{index} = {index}  # " + "x" * 60 + "\n" for index in range(1, 201)
    )
    observation = {
        "path": "main.py", "content": content,
        "start_line": 1, "end_line": 200, "total_lines": 900,
        "truncated": False, "language": "Python",
    }
    bounded = service._bounded_observation(observation, 6_000)

    included = bounded["content"].count("\n")
    assert included >= 1
    assert bounded["end_line"] == included
    assert bounded["next_start_line"] == included + 1
    assert bounded["end_line"] < 200 and bounded["truncated"] is True
    assert "content_cut_midline" not in bounded


def test_a_multi_call_round_has_a_total_observation_budget(service):
    observations = [
        {"tool": "file.read", "ok": True,
         "result": {"path": f"file{index}.py", "content": "x" * 5_000,
                    "start_line": 1}}
        for index in range(8)
    ]
    message = service._observation_message(observations)

    assert len(message) <= ide_module.MAX_OBSERVATION_TOTAL_CHARS + 1_000
    assert "Some results were shortened" in message


def test_the_message_window_forces_a_recap_before_messages_are_dropped(service):
    workspace = make_workspace(service)
    session = service._new_session(workspace["id"], "Seed", "ask", "fake-coder")
    messages = [{"role": "system", "content": "system"}]
    for index in range(30):
        messages.append({"role": "assistant", "content": json.dumps({
            "tool": "file.read", "input": {"path": f"file{index}.py"},
        })})
        messages.append({"role": "user", "content": "Tool results: " + "x" * 400})
    budget = {"unit": "tokens", "context_window": 1_000_000}

    combined, info = service._consolidate_for_budget(
        messages, budget, session, "auto",
        max_messages=ide_module.MESSAGE_RECAP_TRIGGER,
    )

    assert info is not None and info["after_tokens"] < info["before_tokens"]
    assert len(combined) <= ide_module.MESSAGE_RECAP_TRIGGER
    assert combined[1]["content"].startswith("Earlier conversation, consolidated")
    assert "file0.py" in combined[1]["content"]


def test_follow_up_context_keeps_the_plan_and_changed_files(service):
    workspace = make_workspace(service)
    session = service._new_session(workspace["id"], "Continue", "edit", "fake-coder")
    session["plan"] = [
        {"text": "Read the notes", "status": "done"},
        {"text": "Rewrite the index", "status": "active"},
    ]
    session["changes"] = {"Notes/Index.md": {"added": 12, "removed": 3}}
    tools = service._tools_for_mode("edit", workspace["id"])

    prompt = service._system_prompt(
        workspace["id"], "edit", tools, service.workspace_path(workspace["id"]),
        None, None, session,
    )
    assert "Current plan from earlier turns" in prompt
    assert "Rewrite the index" in prompt
    assert "Notes/Index.md (+12/−3)" in prompt

    message = service._observation_message(
        [{"tool": "file.read", "ok": True, "result": {"path": "main.py"}}], session,
    )
    assert message.startswith("Progress:")
    assert "Notes/Index.md" in message


def test_a_follow_up_run_keeps_the_previous_plan(service):
    workspace = make_workspace(service)
    session = service._new_session(workspace["id"], "Continue", "edit", "fake-coder")
    session["plan"] = [{"text": "Keep me", "status": "active"}]
    service._save_session(session)
    gateway = ScriptedGateway(['{"final": "continued"}'])
    service.gateway = gateway

    started = service.start_run(workspace["id"], {
        "message": "Continue the work", "mode": "edit", "session_id": session["id"],
    })
    final = run_until_finished(service, started["run"]["id"])

    assert final["status"] == "completed", final.get("error")
    prompt = gateway.calls[0]["messages"][0]["content"]
    assert "Current plan from earlier turns" in prompt and "Keep me" in prompt
    # The follow-up saw the preserved plan and its successful finish closed it.
    assert service.get_session(session["id"])["plan"] == [
        {"text": "Keep me", "status": "done"},
    ]


def test_a_successful_run_marks_every_plan_step_done(service):
    workspace = make_workspace(service)
    service.gateway = ScriptedGateway([
        '{"plan": [{"text": "Read the notes", "status": "done"}, '
        '{"text": "Rewrite the index", "status": "active"}, '
        '{"text": "Check the links", "status": "pending"}]}',
        '{"final": "Restructured the vault."}',
    ])
    started = service.start_run(workspace["id"], {"message": "Restructure", "mode": "edit"})
    final = run_until_finished(service, started["run"]["id"])

    assert final["status"] == "completed", final.get("error")
    session = service.get_session(started["session"]["id"])
    assert [item["status"] for item in session["plan"]] == ["done", "done", "done"]
    plan_events = [event for event in final["events"] if event["type"] == "plan"]
    assert plan_events[-1]["message"] == "Plan completed"
    assert [item["status"] for item in plan_events[-1]["plan"]] == ["done", "done", "done"]
    plan_entries = [entry for entry in session["transcript"] if entry.get("role") == "plan"]
    assert plan_entries[-1]["todos"] == session["plan"]


def test_an_exhausted_run_leaves_the_remaining_plan_steps_open(service):
    workspace = make_workspace(service)
    service.gateway = ScriptedGateway([
        '{"plan": [{"text": "Read one", "status": "active"}, '
        '{"text": "Read two", "status": "pending"}]}',
        '{"tool": "file.read", "input": {"path": "main.py"}}',
        '{"tool": "file.read", "input": {"path": "main.py"}}',
    ])
    started = service.start_run(workspace["id"], {
        "message": "Read", "mode": "ask", "max_steps": 2,
    })
    final = run_until_finished(service, started["run"]["id"])

    assert final["exhausted"] is True
    session = service.get_session(started["session"]["id"])
    assert [item["status"] for item in session["plan"]] == ["active", "pending"]


def test_a_failed_run_leaves_the_plan_open(service):
    workspace = make_workspace(service)
    service.gateway = ScriptedGateway([
        '{"plan": [{"text": "Read", "status": "active"}]}',
        '{"tool":', '{"tool":', '{"tool":',
    ])
    started = service.start_run(workspace["id"], {"message": "Break", "mode": "ask"})
    final = run_until_finished(service, started["run"]["id"])

    assert final["status"] == "failed"
    session = service.get_session(started["session"]["id"])
    assert [item["status"] for item in session["plan"]] == ["active"]


def test_a_long_run_checkpoints_its_history_before_it_finishes(service, monkeypatch):
    monkeypatch.setattr(ide_module, "CHECKPOINT_SECONDS", 0)
    workspace = make_workspace(service)
    gateway = CheckpointGateway()
    service.gateway = gateway
    started = service.start_run(workspace["id"], {"message": "Read main", "mode": "ask"})

    assert gateway.blocked.wait(5)
    saved = json.loads(service._session_path(started["session"]["id"])
                       .read_text(encoding="utf-8"))
    assert saved["status"] == "running"
    assert saved["last_event_seq"] > 0
    assert saved["messages"]
    assert any(entry.get("role") == "tool" for entry in saved["transcript"])

    gateway.release.set()
    final = run_until_finished(service, started["run"]["id"])
    assert final["status"] == "completed"


def test_renaming_a_conversation_while_a_run_is_active_survives_the_final_save(service):
    workspace = make_workspace(service)
    gateway = BlockingGateway()
    service.gateway = gateway
    started = service.start_run(workspace["id"], {"message": "Wait", "mode": "ask"})

    assert gateway.started.wait(5)
    service.rename_session(started["session"]["id"], {
        "title": "Renamed mid-run", "pinned": True,
    })
    gateway.release.set()
    final = run_until_finished(service, started["run"]["id"])

    assert final["status"] == "completed"
    reloaded = service.get_session(started["session"]["id"])
    assert reloaded["title"] == "Renamed mid-run"
    assert reloaded["pinned"] is True


def test_a_transient_model_failure_is_retried_without_spending_a_step(service):
    workspace = make_workspace(service)
    gateway = FlakyGateway()
    service.gateway = gateway
    started = service.start_run(workspace["id"], {
        "message": "Recover", "mode": "ask", "max_steps": 1,
    })
    final = run_until_finished(service, started["run"]["id"])

    assert final["status"] == "completed", final.get("error")
    assert final["output"] == "Recovered from a network failure."
    assert len(gateway.calls) == 2


def test_cancelling_a_run_interrupts_a_network_retry_wait(service):
    class RetryGateway(ScriptedGateway):
        def __init__(self):
            super().__init__([])
            self.calls = 0

        def stream(self, config, cancelled=None):
            self.calls += 1
            if self.calls == 1:
                raise TimeoutError("temporary timeout")
            yield from super().stream(config, cancelled)

    workspace = make_workspace(service)
    gateway = RetryGateway()
    service.gateway = gateway
    started = service.start_run(workspace["id"], {"message": "Say hello"})
    run_id = started["run"]["id"]
    wait_for(lambda: any(event["type"] == "log"
                         and "retrying in" in event.get("message", "")
                         for event in service.get_run(run_id)["events"]), timeout=5)

    began = time.monotonic()
    service.cancel_run(run_id)
    final = run_until_finished(service, run_id)
    elapsed = time.monotonic() - began

    assert final["status"] == "cancelled"
    assert elapsed < 0.8, f"cancellation waited {elapsed:.2f}s for the retry sleep"
    assert gateway.calls == 1
    logs = [event["message"] for event in final["events"] if event["type"] == "log"]
    assert any("retrying" in message.lower() for message in logs)


def test_ask_mode_rejects_writes_and_created_files_are_reverted(service):
    workspace = make_workspace(service)
    service.gateway = ScriptedGateway([
        '{"tool": "file.write", "input": {"path": "new.txt", "content": "nope"}}',
        '{"final": "I cannot write in Ask mode."}',
    ])
    started = service.start_run(workspace["id"], {"message": "Write a file", "mode": "ask"})
    final = run_until_finished(service, started["run"]["id"])
    assert final["status"] == "completed"
    results = [event for event in final["events"] if event["type"] == "tool_result"]
    assert results[0]["ok"] is False and "not enabled" in results[0]["output"]["error"]
    assert not (service.workspace_path(workspace["id"]) / "new.txt").exists()

    service.gateway = ScriptedGateway([
        '{"tool": "file.write", "input": {"path": "created.txt", "content": "one"}}',
        '{"tool": "file.write", "input": {"path": "created.txt", "content": "two"}}',
        '{"final": "Created it."}',
    ])
    started = service.start_run(workspace["id"], {"message": "Create a file", "mode": "edit"})
    run_until_finished(service, started["run"]["id"])
    created = service.workspace_path(workspace["id"]) / "created.txt"
    assert created.read_text() == "two"
    session = service.get_session(started["session"]["id"])
    assert session["changes"]["created.txt"]["existed"] is False
    result = service.revert_session(session["id"])
    assert result["deleted"] == 1 and not created.exists()


def test_revert_skips_files_changed_after_the_agent(service):
    workspace = make_workspace(service)
    service.gateway = ScriptedGateway([
        '{"tool": "file.write", "input": {"path": "main.py", "content": "agent\\n"}}',
        '{"final": "Wrote it."}',
    ])
    started = service.start_run(workspace["id"], {"message": "Rewrite main", "mode": "edit"})
    run_until_finished(service, started["run"]["id"])
    current = service.read_file(workspace["id"], "main.py")
    service.write_file(workspace["id"], "main.py", "person\n", version=current["version"])
    result = service.revert_session(started["session"]["id"])
    assert result == {"restored": 0, "deleted": 0, "skipped": 1}
    assert service.read_file(workspace["id"], "main.py")["content"] == "person\n"
    # Reverting again must not repeat the same skip forever.
    assert service.revert_session(started["session"]["id"]) == {
        "restored": 0, "deleted": 0, "skipped": 0,
    }
    assert service.get_session(started["session"]["id"])["changes"]["main.py"]["reverted"] is True


def test_context_preview_includes_a_prompt_budget(service):
    workspace = make_workspace(service)
    preview = service.context_preview(workspace["id"], {
        "context": {"include": ["main.py"], "exclude": []},
        "message": "Explain the greeting", "mode": "ask", "model": "fake-coder",
    })
    budget = preview["budget"]
    assert budget["model"]["id"] == "fake-coder"
    assert budget["context_window"] == 8192
    assert budget["model"]["source"] == "default"
    assert budget["unit"] == "tokens"
    assert budget["used_tokens"] > budget["draft"]["tokens"] > 0
    assert budget["output_reserve"] == 4_096
    labels = {section["id"] for section in budget["sections"]}
    assert {"instructions", "map", "pinned", "task"} <= labels
    assert budget["pinned"]["selected"] == 1
    assert preview["files"][0]["tokens"] > 0
    assert budget["prompt"]["chars"] > 0
    assert budget["limits"]["context_files"] > 0


def test_the_full_task_is_sent_once_and_the_prompt_keeps_only_an_anchor(service):
    workspace = make_workspace(service)
    gateway = ScriptedGateway(['{"final": "done"}'])
    service.gateway = gateway
    marker = "Refactor the login flow. " + "Detail. " * 200 + " UNIQUE-TAIL"
    started = service.start_run(workspace["id"], {"message": marker, "mode": "ask"})
    final = run_until_finished(service, started["run"]["id"])

    assert final["status"] == "completed", final.get("error")
    system = gateway.calls[0]["messages"][0]["content"]
    assert "Refactor the login flow." in system
    assert "UNIQUE-TAIL" not in system
    assert gateway.calls[0]["messages"][-1]["content"] == marker

    budget = service.context_budget(workspace["id"], {"message": marker, "mode": "ask"})
    assert any(section["id"] == "task" for section in budget["sections"])
    assert budget["task"]["chars"] == len(marker)


def test_a_truncated_active_file_is_labelled_in_the_prompt(service):
    workspace = make_workspace(service)
    service.write_file(workspace["id"], "long.txt", "x" * 7_000, version="")
    active = service._active_file(workspace["id"], "long.txt")
    assert active["truncated"] is True
    assert len(active["content"]) == 6_000

    tools = service._tools_for_mode("ask", workspace["id"])
    sections = service._system_sections(
        workspace["id"], "ask", tools, service.workspace_path(workspace["id"]),
        active, {}, {})
    text = next(section["text"] for section in sections if section["id"] == "active")
    assert "truncated at line" in text and "file.read" in text

    pieces, _prompt, _truncated = service._prompt_pieces(
        workspace["id"], "ask", tools, service.workspace_path(workspace["id"]),
        None, {}, "Review the notes", ["long.txt"], {})
    opened = next(piece for piece in pieces if piece["id"] == "open:long.txt")
    assert "Preview only" in opened["text"]


def test_budget_uses_the_catalog_context_window_and_bytes(service):
    class ByteGateway(ScriptedGateway):
        def catalog(self, force=False):
            return [{"id": "tiny-byte", "name": "Tiny byte model", "available": True,
                     "kind": "mariana-drv",
                     "detail": {"ctx_len": 2048, "ctx_unit": "bytes"},
                     "capabilities": {"chat": True}}]

    service.gateway = ByteGateway([])
    workspace = make_workspace(service)
    budget = service.context_budget(workspace["id"], {
        "model": "tiny-byte", "message": "héllo", "mode": "ask",
    })
    assert budget["unit"] == "bytes"
    assert budget["context_window"] == 2048
    assert budget["model"]["source"] == "catalog"
    assert budget["draft"]["tokens"] == len("héllo".encode()) + 4


def test_budget_uses_the_selected_or_default_model_window(service):
    class WideGateway(ScriptedGateway):
        def catalog(self, force=False):
            return [
                {"id": "wide", "name": "Wide model", "available": True, "kind": "openai",
                 "detail": {"context_window": 131072}, "capabilities": {"chat": True}},
                {"id": "tiny", "name": "Tiny model", "available": False,
                 "capabilities": {"chat": True}},
            ]

    service.gateway = WideGateway([])
    workspace = make_workspace(service)
    budget = service.context_budget(workspace["id"], {"message": "hello", "mode": "ask"})
    assert budget["model"]["id"] == "wide"
    assert budget["context_window"] == 131072
    assert budget["model"]["source"] == "catalog"

    class LiveGateway(WideGateway):
        def context_window(self, model_id=None, force=False):
            assert force is True
            return {"id": "wide", "name": "Wide model", "kind": "openai", "found": True,
                    "context_window": 200000, "source": "endpoint", "unit": "tokens"}

    service.gateway = LiveGateway([])
    budget = service.context_budget(workspace["id"], {
        "model": "wide", "message": "hello", "mode": "ask", "refresh": True,
    })
    assert budget["context_window"] == 200000
    assert budget["model"]["source"] == "endpoint"


def test_budget_warns_when_the_prompt_is_clipped(service, monkeypatch):
    monkeypatch.setattr(ide_module, "MAX_PROMPT_CHARS", 400)
    workspace = make_workspace(service)
    budget = service.context_budget(workspace["id"], {
        "message": "x" * 3000, "mode": "ask", "model": "fake-coder",
    })
    assert budget["truncated"] is True
    assert any("clipped" in warning["text"] for warning in budget["warnings"])


def test_finished_runs_store_usage_and_publish_a_budget(service):
    workspace = make_workspace(service)
    service.gateway = ScriptedGateway(['{"final": "All done." }'])
    started = service.start_run(workspace["id"], {"message": "Do the task", "mode": "ask"})
    run = run_until_finished(service, started["run"]["id"])
    assert run["usage"] == {"prompt_tokens": 5, "completion_tokens": 3}
    started_event = next(event for event in run["events"] if event["type"] == "run_started")
    assert started_event["budget"]["used_tokens"] > 0
    assert started_event["budget"]["context_window"] == 8192
    session = service.get_session(started["session"]["id"])
    assert session["usage"] == {"prompt_tokens": 5, "completion_tokens": 3}
    assert service._session_summary(session)["usage"]["prompt_tokens"] == 5


def test_project_instructions_are_included_in_the_system_prompt(service):
    workspace = make_workspace(service)
    service.write_file(workspace["id"], "AGENTS.md", "Always add a test.\n", version="")
    prompt = service._system_prompt(
        workspace["id"], "edit", service._tools_for_mode("edit"),
        service.workspace_path(workspace["id"]), None,
    )
    assert "Project instructions" in prompt and "Always add a test." in prompt
    assert "file.edit" in prompt and "Current mode: edit" in prompt


def test_shell_is_only_offered_when_the_server_opts_in(service, monkeypatch):
    monkeypatch.delenv("MARIANA_AGENT_ALLOW_SHELL", raising=False)
    monkeypatch.delenv("MARIANA_AGENT_ALLOW_OS", raising=False)
    assert "shell.exec" not in [tool["id"] for tool in service._tools_for_mode("auto")]
    monkeypatch.setenv("MARIANA_AGENT_ALLOW_SHELL", "1")
    assert "shell.exec" in [tool["id"] for tool in service._tools_for_mode("auto")]


def test_public_web_and_news_tools_are_available_in_every_mode(service):
    for mode in ("ask", "edit", "auto"):
        ids = {tool["id"] for tool in service._tools_for_mode(mode)}
        assert {"http.fetch", "news.headlines"} <= ids, mode


def test_the_os_access_setting_offers_shell_and_private_web(tmp_path, monkeypatch):
    monkeypatch.delenv("MARIANA_AGENT_ALLOW_SHELL", raising=False)
    monkeypatch.delenv("MARIANA_AGENT_ALLOW_OS", raising=False)
    instance = IdeService(tmp_path, ScriptedGateway([]), skills_roots=[],
                          settings=lambda: {"allow_os_access": True})
    try:
        tools = {tool["id"]: tool for tool in instance._tools_for_mode("auto")}
        assert {"shell.exec", "news.headlines", "browser.fetch"} <= set(tools)
        assert tools["http.fetch"]["config"]["allow_private"] is True
        assert instance.os_access is True
    finally:
        instance.close()


def test_data_and_git_inspection_tools_are_shared_with_the_coding_agent(service):
    expected = {"csv.read", "sqlite.query", "git.inspect"}
    for mode in ("ask", "edit", "auto"):
        tools = {tool["id"]: tool for tool in service._tools_for_mode(mode)}
        assert expected <= set(tools)
        assert tools["sqlite.query"]["config"]["write"] is False
    assert "file.write" not in {tool["id"] for tool in service._tools_for_mode("ask")}


def test_sessions_are_persisted_listed_and_deleted(service):
    workspace = make_workspace(service)
    service.gateway = ScriptedGateway(['{"final": "All done."}'])
    started = service.start_run(workspace["id"], {"message": "Do nothing"})
    run_until_finished(service, started["run"]["id"])
    sessions = service.list_sessions(workspace["id"])
    assert sessions and sessions[0]["turns"] == 1 and sessions[0]["status"] == "completed"
    assert service.get_session(sessions[0]["id"])["transcript"]
    assert service.delete_session(sessions[0]["id"]) is True
    assert service.list_sessions(workspace["id"]) == []
    with pytest.raises(NotFoundError):
        service.get_session(sessions[0]["id"])


def test_model_selection_reports_missing_catalog_entries(service):
    workspace = make_workspace(service)
    service.gateway = ScriptedGateway(['{"final": "unused"}'])
    service.gateway.catalog = lambda force=False: []
    started = service.start_run(workspace["id"], {"message": "Try"})
    final = run_until_finished(service, started["run"]["id"])
    assert final["status"] == "failed" and "No chat model" in final["error"]
    session = service.get_session(started["session"]["id"])
    assert session["transcript"][-1]["role"] == "error"


def test_reverting_an_orphaned_session_does_not_resurrect_its_workspace(service):
    workspace = make_workspace(service)
    service.gateway = ScriptedGateway([
        '{"tool": "file.write", "input": {"path": "main.py", "content": "agent\\n"}}',
        '{"final": "Wrote it."}',
    ])
    started = service.start_run(workspace["id"], {"message": "Rewrite main", "mode": "edit"})
    run_until_finished(service, started["run"]["id"])

    shutil.rmtree(service.workspace_path(workspace["id"]))
    result = service.revert_session(started["session"]["id"])
    assert result == {"restored": 0, "deleted": 0, "skipped": 1}
    assert not (service.workspaces_dir / workspace["id"]).exists()


def test_a_corrupt_manifest_does_not_break_the_workspace(service):
    workspace = make_workspace(service)
    manifest = service.workspaces_dir / workspace["id"] / ".mariana-ide.json"
    manifest.write_text("[1]", encoding="utf-8")

    assert [item["id"] for item in service.list_workspaces()] == [workspace["id"]]
    current = service.read_file(workspace["id"], "main.py")
    saved = service.write_file(workspace["id"], "main.py", "print(1)\n", version=current["version"])
    assert saved["written"]
    assert service.read_file(workspace["id"], "main.py")["content"] == "print(1)\n"

    manifest.write_text(
        json.dumps({"name": "Demo", "created": [], "updated": "soon"}), encoding="utf-8"
    )
    listed = service.list_workspaces()
    assert [item["id"] for item in listed] == [workspace["id"]]
    assert listed[0]["name"] == "Demo" and listed[0]["updated"] > 0


def test_a_non_string_import_source_is_rejected(service):
    with pytest.raises(BadRequestError, match="Folder not found"):
        service.create_workspace("bad", source=["not", "a", "path"])
    assert service.list_workspaces() == []


def test_open_workspace_edits_the_original_folder_in_place(service, tmp_path):
    source = tmp_path / "original-project"
    source.mkdir()
    (source / "app.py").write_text("print('one')\n", encoding="utf-8")

    opened = service.open_workspace(None, str(source))
    assert opened["external"] is True
    assert opened["root"] == str(source.resolve())
    assert opened["id"] == "original-project"
    assert not (service.workspaces_dir / opened["id"]).exists()

    current = service.read_file(opened["id"], "app.py")
    service.write_file(opened["id"], "app.py", "print('two')\n", version=current["version"])
    assert (source / "app.py").read_text(encoding="utf-8") == "print('two')\n"

    again = service.open_workspace(None, str(source))
    assert again["id"] == opened["id"]
    assert [item["id"] for item in service.list_workspaces()] == [opened["id"]]


def test_open_workspace_rejects_private_and_invalid_folders(service, tmp_path):
    with pytest.raises(BadRequestError, match="outside agent-workspace"):
        service.open_workspace(None, str(service.agent_workspace))
    with pytest.raises(BadRequestError, match="Folder not found"):
        service.open_workspace(None, str(tmp_path / "missing-project"))
    with pytest.raises(BadRequestError, match="absolute"):
        service.open_workspace(None, "relative/project")
    note = tmp_path / "notes.txt"
    note.write_text("x", encoding="utf-8")
    with pytest.raises(BadRequestError, match="not a folder"):
        service.open_workspace(None, str(note))


def test_external_workspace_settings_live_in_the_registry(service, tmp_path):
    source = tmp_path / "settings-project"
    source.mkdir()
    (source / "app.py").write_text("x = 1\n", encoding="utf-8")
    opened = service.open_workspace(None, str(source))

    service.update_settings(opened["id"], {
        "excluded": ["private"], "context": {"include": ["app.py"]},
    })
    assert not (source / ".mariana-ide.json").exists()
    assert service.settings(opened["id"])["excluded"] == ["private"]
    assert service.settings(opened["id"])["context"]["include"] == ["app.py"]


def test_deleting_an_in_place_workspace_keeps_its_files(service, tmp_path):
    source = tmp_path / "keep-me"
    source.mkdir()
    (source / "app.py").write_text("print(1)\n", encoding="utf-8")
    opened = service.open_workspace(None, str(source))

    assert service.delete_workspace(opened["id"]) is True
    assert (source / "app.py").read_text(encoding="utf-8") == "print(1)\n"
    assert service.list_workspaces() == []
    with pytest.raises(NotFoundError):
        service.read_file(opened["id"], "app.py")


def test_the_runtime_folder_is_private_inside_an_in_place_project(service, tmp_path):
    opened = service.open_workspace("root", str(tmp_path))
    tree = service.tree(opened["id"])
    assert "agent-workspace" not in [node["name"] for node in tree["children"]]
    with pytest.raises(BadRequestError, match="private"):
        service.read_file(opened["id"], "agent-workspace/settings.json")

    definition = {"id": "read", "type": "file.read", "enabled": True, "config": {}}
    with pytest.raises(ToolError, match="private"):
        execute_ability(definition, {"path": "agent-workspace/settings.json"},
                        {"agent": {"id": "ide"}}, service.root,
                        files_root=service.workspace_path(opened["id"]),
                        deny=service._tool_deny(opened["id"]))


def test_a_corrupt_saved_run_is_not_served(service):
    run_id = "ide-run-abcdef123456"
    (service.runs_dir / f"{run_id}.json").write_text(
        json.dumps({"session_id": "ide-abcdef", "events": "oops"}), encoding="utf-8"
    )
    with pytest.raises(NotFoundError):
        service.get_run(run_id)
    with pytest.raises(NotFoundError):
        service.stream_run(run_id)


def test_a_failed_edit_after_a_successful_one_has_no_change(service):
    workspace = make_workspace(service)
    service.gateway = ScriptedGateway([
        '{"tool": "file.edit", "input": {"path": "main.py", "old": "Hello", "new": "Hi"}}',
        '{"tool": "file.edit", "input": {"path": "main.py", "old": "missing", "new": "x"}}',
        '{"final": "Done."}',
    ])
    started = service.start_run(workspace["id"], {"message": "Edit twice", "mode": "edit"})
    final = run_until_finished(service, started["run"]["id"])

    results = [event for event in final["events"] if event["type"] == "tool_result"]
    assert [result["ok"] for result in results] == [True, False]
    assert results[0]["change"]["added"] == 1
    assert not results[1].get("change")
    steps = [item for item in service.get_session(started["session"]["id"])["transcript"]
             if item.get("role") == "tool"]
    assert "change" not in steps[1]


def test_a_failed_edit_does_not_create_missing_folders(service):
    workspace = make_workspace(service)
    service.gateway = ScriptedGateway([
        '{"tool": "file.edit", "input": {"path": "missing/dir/main.py", '
        '"old": "a", "new": "b"}}',
        '{"final": "Could not edit the file."}',
    ])
    started = service.start_run(workspace["id"], {"message": "Edit a missing file", "mode": "edit"})
    final = run_until_finished(service, started["run"]["id"])

    assert final["status"] == "completed", final.get("error")
    results = [event for event in final["events"] if event["type"] == "tool_result"]
    assert results and results[0]["ok"] is False
    assert not (service.workspace_path(workspace["id"]) / "missing").exists()


def test_a_run_that_cannot_start_does_not_leave_a_running_session(service, monkeypatch):
    workspace = make_workspace(service)

    def fail(thread):
        raise RuntimeError("cannot start a thread")

    monkeypatch.setattr(threading.Thread, "start", fail)
    with pytest.raises(RuntimeError, match="cannot start"):
        service.start_run(workspace["id"], {"message": "Try to run"})
    sessions = service.list_sessions(workspace["id"])
    assert len(sessions) == 1 and sessions[0]["status"] == "failed"
    assert sessions[0]["active"] is False

    monkeypatch.undo()
    service.gateway = ScriptedGateway(['{"final": "done"}'])
    started = service.start_run(workspace["id"], {"message": "Run now"})
    assert run_until_finished(service, started["run"]["id"])["status"] == "completed"


def test_a_rejected_concurrent_run_does_not_create_a_session(service):
    workspace = make_workspace(service)
    gateway = BlockingGateway()
    service.gateway = gateway
    first = service.start_run(workspace["id"], {"message": "first"})
    assert gateway.started.wait(5)

    with pytest.raises(ConflictError, match="Another coding run"):
        service.start_run(workspace["id"], {"message": "second"})

    gateway.release.set()
    run_until_finished(service, first["run"]["id"])
    assert [item["id"] for item in service.list_sessions(workspace["id"])] == [
        first["session"]["id"],
    ]


def test_a_finished_model_stream_does_not_cancel_the_run(service):
    workspace = make_workspace(service)
    service.gateway = RequestScopedGateway([
        '{"tool": "file.write", "input": {"path": "created.txt", "content": "one"}}',
        '{"tool": "file.write", "input": {"path": "created.txt", "content": "two"}}',
        '{"final": "Created it."}',
    ])
    started = service.start_run(workspace["id"], {"message": "Create a file", "mode": "edit"})
    final = run_until_finished(service, started["run"]["id"])

    assert final["status"] == "completed", final.get("error")
    assert final["output"] == "Created it."
    assert (service.workspace_path(workspace["id"]) / "created.txt").read_text() == "two"


def test_repeated_edits_report_counts_that_match_the_diff(service):
    workspace = make_workspace(service)
    service.gateway = ScriptedGateway([
        '{"tool": "file.edit", "input": {"path": "main.py", "old": "Hello", "new": "Hi"}}',
        '{"tool": "file.edit", "input": {"path": "main.py", "old": "Hi", "new": "Hey"}}',
        '{"final": "Renamed twice."}',
    ])
    started = service.start_run(workspace["id"], {"message": "Rename", "mode": "edit"})
    run_until_finished(service, started["run"]["id"])

    change = service.get_session(started["session"]["id"])["changes"]["main.py"]
    lines = change["diff"].splitlines()
    added = sum(1 for line in lines if line.startswith("+") and not line.startswith("+++"))
    removed = sum(1 for line in lines if line.startswith("-") and not line.startswith("---"))
    assert (change["added"], change["removed"]) == (added, removed) == (1, 1)


def test_import_refuses_the_private_studio_workspace(service):
    (service.agent_workspace / "settings.json").write_text('{"api_key": "secret"}', encoding="utf-8")

    with pytest.raises(BadRequestError, match="agent-workspace"):
        service.create_workspace("stolen", source=str(service.agent_workspace))
    with pytest.raises(BadRequestError, match="agent-workspace"):
        service.create_workspace("nested", source=str(service.base))
    assert service.list_workspaces() == []


class SlowFinishRun(IdeRun):
    """Hold the terminal event briefly so a streaming waiter can race the status."""

    def event(self, kind, **fields):
        if kind.startswith("run_"):
            time.sleep(0.05)
        return super().event(kind, **fields)


def test_a_finished_run_still_delivers_its_terminal_event():
    run = SlowFinishRun("ide-run-stream", "ide-session", "workspace", "task")
    seen: list[str] = []
    finished = threading.Event()

    def watch():
        while True:
            pending, done = run.wait_events(0, timeout=5)
            seen.extend(item["type"] for item in pending)
            if done:
                finished.set()
                return

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    time.sleep(0.02)
    run.finish("completed", output="done")
    assert finished.wait(5)
    watcher.join(timeout=5)
    assert "run_completed" in seen


def test_a_corrupt_session_does_not_break_listing_or_reverting(service):
    workspace = make_workspace(service)
    service.gateway = ScriptedGateway(['{"final": "All done."}'])
    started = service.start_run(workspace["id"], {"message": "Do nothing"})
    session_id = started["session"]["id"]
    run_until_finished(service, started["run"]["id"])

    path = service.sessions_dir / f"{session_id}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data.update({"transcript": "oops", "changes": ["oops"], "plan": "oops",
                 "updated": "yesterday", "created": None})
    path.write_text(json.dumps(data), encoding="utf-8")

    summary = service.list_sessions(workspace["id"])[0]
    assert summary["turns"] == 0 and summary["changes"] == 0
    assert summary["updated"] == 0.0 and summary["created"] == 0.0
    session = service.get_session(session_id)
    assert session["plan"] == [] and session["changes"] == {} and session["transcript"] == []

    service.gateway = ScriptedGateway(['{"final": "Again."}'])
    data["messages"] = ["oops"]
    path.write_text(json.dumps(data), encoding="utf-8")
    started = service.start_run(workspace["id"], {"message": "Continue", "session_id": session_id})
    assert run_until_finished(service, started["run"]["id"])["status"] == "completed"
    assert service.revert_session(session_id) == {"restored": 0, "deleted": 0, "skipped": 0}


def test_deleting_a_session_removes_its_saved_runs(service):
    workspace = make_workspace(service)
    service.gateway = ScriptedGateway(['{"final": "All done."}'])
    started = service.start_run(workspace["id"], {"message": "Do nothing"})
    run_id = started["run"]["id"]
    session_id = started["session"]["id"]
    run_until_finished(service, run_id)
    wait_for(lambda: (service.runs_dir / f"{run_id}.json").is_file())

    assert service.delete_session(session_id) is True
    assert list(service.runs_dir.glob("ide-run-*.json")) == []
    with pytest.raises(NotFoundError):
        service.get_run(run_id)


def test_deleting_a_workspace_removes_its_sessions_and_runs(service):
    workspace = make_workspace(service)
    service.gateway = ScriptedGateway(['{"final": "All done."}'])
    started = service.start_run(workspace["id"], {"message": "Do nothing"})
    run_id = started["run"]["id"]
    run_until_finished(service, run_id)
    wait_for(lambda: (service.runs_dir / f"{run_id}.json").is_file())

    assert service.delete_workspace(workspace["id"]) is True
    assert list(service.sessions_dir.glob("ide-*.json")) == []
    assert list(service.runs_dir.glob("ide-run-*.json")) == []
    with pytest.raises(NotFoundError):
        service.get_run(run_id)


def test_a_workspace_is_reserved_while_its_files_are_deleted(service, monkeypatch):
    workspace = make_workspace(service)
    seen = {}
    original = service._load_sessions

    def load(*args, **kwargs):
        seen["deleting"] = workspace["id"] in service._deleting_workspaces
        return original(*args, **kwargs)

    monkeypatch.setattr(service, "_load_sessions", load)
    assert service.delete_workspace(workspace["id"]) is True
    assert seen["deleting"] is True
    assert workspace["id"] not in service._deleting_workspaces


def test_a_run_cannot_start_while_its_workspace_is_being_deleted(service):
    workspace = make_workspace(service)
    service._deleting_workspaces.add(workspace["id"])
    with pytest.raises(ConflictError, match="being deleted"):
        service.start_run(workspace["id"], {"message": "late"})
    service._deleting_workspaces.discard(workspace["id"])


def test_unknown_session_and_workspace_ids_are_not_found(service):
    with pytest.raises(NotFoundError):
        service.get_session("ide-not-real")
    with pytest.raises(NotFoundError):
        service.tree("../escape")
    with pytest.raises(NotFoundError):
        service.start_run("missing", {"message": "hi"})


def test_skills_can_be_created_edited_read_and_deleted(service):
    workspace = make_workspace(service)
    created = service.create_skill(workspace["id"], {
        "name": "Release notes", "description": "Draft a release note",
        "instructions": "# Steps\nRead the changelog.",
    })
    assert created["id"] == "release-notes" and created["scope"] == "workspace"
    assert "Read the changelog." in created["instructions"]
    path = (service.workspace_path(workspace["id"]) / ".agents" / "skills"
            / "release-notes" / "SKILL.md")
    assert path.is_file() and "name: Release notes" in path.read_text(encoding="utf-8")
    assert [item["id"] for item in service.list_skills(workspace["id"])] == ["release-notes"]

    fetched = service.read_skill(workspace["id"], "release-notes", include_instructions=True)
    assert fetched["name"] == "Release notes" and fetched["instructions"] == "# Steps\nRead the changelog."
    updated = service.update_skill(workspace["id"], "release-notes", {
        "name": "Release notes v2", "description": "Ship it",
        "instructions": "# Steps\nCheck the tags.",
    })
    assert updated["name"] == "Release notes v2" and "Check the tags." in updated["instructions"]
    assert "Check the tags." in path.read_text(encoding="utf-8")

    with pytest.raises(ConflictError, match="already exists"):
        service.create_skill(workspace["id"], {"name": "Release notes"})
    with pytest.raises(BadRequestError, match="Skill id"):
        service.create_skill(workspace["id"], {"id": "../", "name": "Escape"})

    deleted = service.delete_skill(workspace["id"], "release-notes")
    assert deleted == {"deleted": True, "id": "release-notes", "scope": "workspace"}
    assert not path.parent.exists()
    with pytest.raises(NotFoundError):
        service.read_skill(workspace["id"], "release-notes")


def test_context_consolidation_recaps_older_turns_and_persists(service):
    workspace = make_workspace(service)
    session = service._new_session(workspace["id"], "Seed history", "ask", "fake-coder")
    session["messages"] = []
    for index in range(8):
        session["messages"].append({
            "role": "user", "content": f"Question {index}: " + "x" * 3000})
        session["messages"].append({
            "role": "assistant", "content": f"Answer {index}: " + "y" * 3000})
    service._save_session(session)

    result = service.consolidate_session(session["id"], "fake-coder")
    assert result["consolidated"] is True
    assert result["after_tokens"] < result["before_tokens"]
    reloaded = service.get_session(session["id"], include_messages=True)
    assert len(reloaded["messages"]) == 9
    assert reloaded["messages"][0]["content"].startswith("Earlier conversation, consolidated")
    assert reloaded["consolidation"]["count"] == 1
    assert reloaded["consolidation"]["trigger"] == "manual"
    assert any(entry.get("kind") == "consolidation" for entry in reloaded["transcript"])


def test_a_run_consolidates_history_before_it_overflows(service):
    class SmallGateway(ScriptedGateway):
        def catalog(self, force=False):
            return [{"id": "fake-coder", "name": "Fake coder", "available": True,
                     "detail": {"ctx_len": 15000},
                     "capabilities": {"chat": True}, "reason": ""}]

    service.gateway = SmallGateway([])
    workspace = make_workspace(service)
    session = service._new_session(workspace["id"], "Seed history", "ask", "fake-coder")
    session["messages"] = []
    for index in range(8):
        session["messages"].append({
            "role": "user", "content": f"Question {index}: " + "x" * 3000})
        session["messages"].append({
            "role": "assistant", "content": f"Answer {index}: " + "y" * 3000})
    service._save_session(session)

    started = service.start_run(workspace["id"], {
        "message": "Continue the work", "mode": "ask", "session_id": session["id"],
        "model": "fake-coder",
    })
    final = run_until_finished(service, started["run"]["id"])
    assert final["status"] == "completed", final.get("error")
    kinds = [event["type"] for event in final["events"]]
    assert "context_consolidated" in kinds
    reloaded = service.get_session(session["id"], include_messages=True)
    assert reloaded["messages"][0]["content"].startswith("Earlier conversation, consolidated")
    assert reloaded["consolidation"]["count"] >= 1
    assert len(reloaded["messages"]) <= 12


def test_swarm_spawns_agents_reviews_and_reports_progress(service):
    class SwarmGateway(ScriptedGateway):
        def stream(self, config, cancelled=None):
            self.calls.append(copy.deepcopy(config))
            system = config["messages"][0]["content"]
            user = config["messages"][-1]["content"]
            if "Plan a focused team" in system:
                reply = json.dumps({
                    "strategy": "peer_review", "rounds": 1,
                    "agents": [
                        {"id": "code", "label": "Code", "role": "reader",
                         "task": "read main.py"},
                        {"id": "docs", "label": "Docs", "role": "checker",
                         "task": "check the docs"},
                    ],
                })
            elif "You are one member of the swarm" in system:
                if "Phase: revise" in user:
                    reply = '{"final": "revised findings with line references"}'
                elif "returned" in user:
                    reply = '{"final": "main.py defines greet."}'
                else:
                    reply = '{"tool": "file.read", "input": {"path": "main.py"}}'
            elif "Assign focused peer reviews" in system:
                reply = json.dumps({"reviews": [
                    {"reviewer": "code", "target": "docs"},
                    {"reviewer": "docs", "target": "code"},
                ]})
            elif "You are reviewing" in system:
                target = "docs" if "reviewing Docs" in system else "code"
                reply = json.dumps({"review": {
                    "target": target, "score": 4, "issues": ["add line refs"],
                    "suggestions": ["quote code"], "summary": "useful",
                }})
            elif "Synthesise the members' work" in system:
                reply = json.dumps({"final": "main.py defines greet.",
                                    "summary": "done", "confidence": 0.9})
            else:
                reply = self.replies.pop(0) if self.replies else '{"final": "done"}'
            yield {"type": "start", "model_id": "fake-coder", "model": "fake"}
            yield {"type": "delta", "text": reply}
            yield {"type": "done", "reply": reply, "usage": {"prompt_tokens": 5,
                                                              "completion_tokens": 3}}

    workspace = make_workspace(service)
    service.gateway = SwarmGateway([
        '{"tool": "agent.swarm", "input": {"tasks": ['
        '{"task": "read main entry", "label": "Code"}, '
        '{"task": "check docs", "label": "Docs"}], '
        '"strategy": "peer_review", "rounds": 1}}',
        '{"final": "Swarm complete."}',
    ])
    assert "file.write" not in {tool["id"] for tool in service._swarm_tools(workspace["id"])}

    started = service.start_run(workspace["id"], {"message": "Explore the project", "mode": "edit"})
    final = run_until_finished(service, started["run"]["id"])
    assert final["status"] == "completed", final.get("error")
    events = final["events"]
    swarm_start = [event for event in events if event["type"] == "swarm_start"]
    plan = [event for event in events if event["type"] == "swarm_plan"]
    assert len(swarm_start) == 1 and swarm_start[0]["strategy"] == "peer_review"
    assert len(plan) == 1 and len(plan[0]["agents"]) == 2
    starts = [event for event in events if event["type"] == "swarm_agent_start"]
    assert len(starts) >= 2
    reviews = [event for event in events if event["type"] == "swarm_review"]
    assert len(reviews) == 2 and all(event["score"] == 4.0 for event in reviews)
    ends = [event for event in events if event["type"] == "swarm_agent_end"]
    assert ends and all(event["ok"] for event in ends)
    assert [event for event in events if event["type"] == "swarm_message"]
    result = [event for event in events if event["type"] == "tool_result"][0]
    assert result["output"]["completed"] == 2 and result["output"]["failed"] == 0
    assert len(result["output"]["reviews"]) == 2
    assert result["output"]["messages"] >= 2
    assert any(event["type"] == "swarm_agent_tool"
               and (event.get("input") or {}).get("path") == "main.py" for event in events)
    assert all(agent.get("output") for agent in result["output"]["agents"])
    assert final["swarm"] and final["swarm"]["strategy"] == "peer_review"


def test_swarm_parent_observation_stays_bounded():
    observation = {
        "ok": True, "strategy": "peer_review", "round": 1, "rounds": 1,
        "output": "answer " * 3000, "summary": "summary " * 100,
        "agents": [{"agent": f"a{i}", "label": "Agent " * 20, "role": "role",
                    "ok": True, "score": 4, "version": 2,
                    "output": "finding " * 1000, "error": "error " * 100}
                   for i in range(6)],
        "reviews": [{"reviewer": "a1", "target": "a2", "score": 4,
                     "issues": ["issue " * 100], "summary": "review " * 100}
                    for _ in range(6)],
        "messages": 50, "blackboard": 20,
        "changed_files": [f"src/file{i}.py" for i in range(40)],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        "cache": {"hits": 3, "misses": 1, "saved_seconds": 1.5,
                  "prompt_cache_hit_rate": 0.5, "entries": 2, "chars": 500,
                  "inflight": 0},
        "elapsed": 12.5,
    }
    compact = ide_module.IdeService._swarm_parent_observation(observation)
    assert len(json.dumps(compact)) < 6000
    assert compact["final"].startswith("answer ")
    assert "output" not in compact["agents"][0]
    assert len(compact["changed_files"]) == 6
    assert compact["cache"] == {"hits": 3, "misses": 1, "saved_seconds": 1.5,
                                "prompt_cache_hit_rate": 0.5}


def test_swarm_coding_edits_are_snapshotted_and_revertible(service):
    class CodingSwarmGateway(ScriptedGateway):
        def stream(self, config, cancelled=None):
            system = config["messages"][0]["content"]
            user = config["messages"][-1]["content"]
            if "Plan a focused team" in system:
                reply = json.dumps({
                    "strategy": "map_reduce", "rounds": 0,
                    "agents": [
                        {"id": "coder", "label": "Coder", "role": "implementer",
                         "task": "add a farewell function to main.py",
                         "tools": ["file.read", "file.edit", "file.write"],
                         "writes": ["main.py"], "read_only": False},
                        {"id": "checker", "label": "Checker", "role": "reviewer",
                         "task": "read the tests",
                         "tools": ["file.read", "file.tree"], "read_only": True},
                    ],
                })
            elif "You are one member of the swarm" in system:
                if "add a farewell function" in user and "returned" not in user:
                    reply = json.dumps({"tool": "file.write", "input": {
                        "path": "main.py",
                        "content": "def farewell(name):\n    return f'Bye, {name}!\n'",
                    }})
                else:
                    reply = '{"final": "main.py now has farewell()."}'
            elif "Synthesise the members' work" in system:
                reply = json.dumps({"final": "Added farewell to main.py.",
                                    "summary": "code changed", "confidence": 0.9})
            else:
                reply = self.replies.pop(0) if self.replies else '{"final": "done"}'
            yield {"type": "done", "reply": reply, "usage": {}}

    workspace = make_workspace(service)
    original = (service.workspace_path(workspace["id"]) / "main.py").read_text(encoding="utf-8")
    service.gateway = CodingSwarmGateway([
        '{"tool": "agent.swarm", "input": {"goal": "add farewell", '
        '"strategy": "map_reduce", "rounds": 0, "write": true}}',
        '{"final": "Swarm coding complete."}',
    ])
    started = service.start_run(workspace["id"], {"message": "add farewell", "mode": "edit"})
    final = run_until_finished(service, started["run"]["id"])
    assert final["status"] == "completed", final.get("error")
    content = (service.workspace_path(workspace["id"]) / "main.py").read_text(encoding="utf-8")
    assert "farewell" in content
    assert any(event["type"] == "file_changed" for event in final["events"])
    session = service.get_session(started["session"]["id"], include_messages=True)
    assert "main.py" in session["changes"]
    service.revert_session(started["session"]["id"])
    restored = (service.workspace_path(workspace["id"]) / "main.py").read_text(encoding="utf-8")
    assert restored == original


def test_swarm_is_read_only_in_ask_mode(service):
    workspace = make_workspace(service)
    tools = service._swarm_tools(workspace["id"], write=False, mode="ask")
    assert not {"file.write", "file.edit", "file.delete", "shell.exec"} & {tool["id"] for tool in tools}
    writable = service._swarm_tools(workspace["id"], write=True, mode="edit")
    assert {"file.write", "file.edit"} <= {tool["id"] for tool in writable}
    assert "shell.exec" not in {tool["id"] for tool in writable}
