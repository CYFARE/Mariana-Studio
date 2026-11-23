"""The HTTP API end to end: agents (AMP), chat, conversations, models, fine-tuning, deploy."""

from __future__ import annotations

import json

import pytest

from mariana.workspace import chat_files_path
from tests.helpers import (
    mock_provider,
    read_events,
    request,
    request_error,
    serving,
    studio,
    tiny_drv_checkpoint,
    wait_for,
)
from tests.test_agents import ScriptedGateway, chain, minimal_agent

ACTIVE = {"queued", "running", "paused"}


@pytest.fixture()
def api(tmp_path):
    (tmp_path / "agent-workspace").mkdir()
    (tmp_path / "agent-workspace" / "settings.json").write_text(json.dumps({"discover_local": False}))
    with studio(tmp_path) as (server, base):
        yield server, base, tmp_path


def run_finished(base, run_id):
    return wait_for(lambda: (lambda snap: snap if snap["status"] not in ACTIVE else None)(
        request(base + f"/api/agents/runs/{run_id}")), timeout=15)


def test_ide_opens_a_folder_in_place_over_http(api):
    _, base, tmp_path = api
    source = tmp_path / "linked-project"
    source.mkdir()
    (source / "app.py").write_text("print('linked')\n", encoding="utf-8")

    opened = request(base + "/api/ide/workspaces/open", {"source": str(source)})
    workspace = opened["workspace"]
    assert opened["external"] is True
    assert workspace["external"] is True and workspace["root"] == str(source.resolve())
    assert not (tmp_path / "agent-workspace" / "ide" / "workspaces"
                / workspace["id"]).exists()

    listed = request(base + "/api/ide/workspaces")["workspaces"]
    assert [(item["id"], item["external"]) for item in listed] == [(workspace["id"], True)]
    file_data = request(
        base + f"/api/ide/workspaces/{workspace['id']}/file?path=app.py")
    assert file_data["content"] == "print('linked')\n"

    request(base + f"/api/ide/workspaces/{workspace['id']}", method="DELETE")
    assert (source / "app.py").read_text(encoding="utf-8") == "print('linked')\n"
    assert request(base + "/api/ide/workspaces")["workspaces"] == []


def test_amp_protocol_workflow(api):
    _, base, _ = api
    protocol = request(base + "/api/agents/protocol")
    assert protocol["protocol"] == "AMP" and protocol["version"] == "1.10"
    assert "assistant_delta" in protocol["event_types"]
    assert "reasoning_delta" in protocol["event_types"]
    assert protocol["aml_version"] == "1.8"
    assert protocol["tool_calling"]["formats"] == ["json", "dsml", "native"]
    assert "swarm" in protocol["node_types"]
    assert protocol["swarm"]["node_type"] == "swarm"
    assert "llm" in protocol["node_types"] and "http.fetch" in protocol["abilities"]
    assert "Agent Management Language" in request(base + "/api/agents/spec/aml")
    assert "Agent Management Protocol" in request(base + "/api/agents/spec/amp")
    template = request(base + "/api/agents/templates")["templates"][0]["aml"]
    assert request(base + "/api/agents/templates/hello-agent")["id"] == "hello-agent"
    assert request_error(base + "/api/agents/templates/nope")[0] == 404
    validated = request(base + "/api/agents/validate", template)
    assert validated["ok"] is True and isinstance(validated["issues"], list)
    saved = request(base + "/api/agents", template)
    assert saved["agent"]["id"] == template["id"]
    run = request(base + f"/api/agents/{template['id']}/runs", {
        "input": {"message": "http"}, "dry_run": True, "trigger": "manual",
    })
    assert run["status"] in {"queued", "running", "paused", "completed"}
    final = run_finished(base, run["id"])
    assert final["status"] == "completed" and "[test mode]" in final["output"]
    assert request(base + f"/api/agents/runs/{run['id']}?since=9999")["events"] == []
    streamed = read_events(base + f"/api/agents/runs/{run['id']}/events?since=0")
    assert streamed and streamed[-1]["id"] == run["id"]
    assert streamed[0]["type"] == "snapshot"
    assert streamed[-1]["status"] == "completed"
    assert request(base + "/api/agents/runs")["runs"][0]["id"] == run["id"]
    assert request(base + "/api/agents")["agents"][0]["id"] == template["id"]
    assert request(base + f"/api/agents/{template['id']}/runs?limit=1")["runs"][0]["id"] == run["id"]
    assert request(base + f"/api/agents/{template['id']}/schedule")["schedule"]["enabled"] is False
    assert request(base + f"/api/agents/{template['id']}")["_meta"] == {"active_run": None}
    cloned = request(base + f"/api/agents/{template['id']}/clone", {"id": "hello-copy", "name": "Hello copy"})
    assert cloned["agent"]["id"] == "hello-copy" and cloned["agent"]["name"] == "Hello copy"
    assert cloned["agent"]["schedule"]["enabled"] is False
    assert request_error(base + f"/api/agents/{template['id']}/clone", {"id": "hello-copy"})[0] == 409
    triggered = request(base + f"/api/agents/{template['id']}/trigger",
                        {"dry_run": True, "input": {"message": "again"}})
    assert run_finished(base, triggered["id"])["status"] == "completed"

    invalid = request(base + "/api/agents/validate", {"name": "x", "nodes": []})
    assert invalid["ok"] is False and invalid["issues"][0]["level"] == "error"
    status, error = request_error(base + "/api/agents", {"name": "x", "nodes": []})
    assert status == 400 and "start node" in error["error"]
    assert request_error(base + "/api/agents/import", {"text": "{not json"})[0] == 400
    imported = request(base + "/api/agents/import", {"text": json.dumps(minimal_agent("imported"))})
    assert imported["agent"]["id"] == "imported"
    assert request_error(base + "/api/agents/missing")[0] == 404
    assert request_error(base + "/api/agents/missing/runs", {})[0] == 404
    assert request_error(base + "/api/agents/runs/run-missing")[0] == 404
    assert request_error(base + "/api/agents/runs/run-missing/events")[0] == 404
    assert request_error(base + f"/api/agents/runs/{run['id']}/swarm")[0] == 404
    assert request_error(base + f"/api/agents/runs/{run['id']}/swarm/messages",
                         {"text": "hello"})[0] == 409
    assert request_error(base + f"/api/agents/runs/{run['id']}/step", {})[0] == 409
    assert request(base + "/api/agents/imported", method="DELETE") == {"deleted": True, "id": "imported"}
    assert request_error(base + "/api/agents/imported", method="DELETE")[0] == 404


def test_amp_swarm_state_is_available_over_http(api):
    _, base, _ = api
    document = minimal_agent("api-swarm", nodes=[
        {"id": "start", "type": "start"},
        {"id": "team", "type": "swarm", "data": {
            "goal": "{{ input.message }}", "strategy": "peer_review", "rounds": 1,
        }},
        {"id": "end", "type": "end", "data": {"output": "{{ nodes.team.text }}"}},
    ], edges=[{"from": "start", "to": "team"}, {"from": "team", "to": "end"}])
    document["aml"] = "1.4"
    document["model"] = "fake"
    document["swarm"] = {"allow_write": False, "max_rounds": 1}
    request(base + "/api/agents", document)
    run = request(base + "/api/agents/api-swarm/runs", {
        "input": {"message": "hello"}, "dry_run": True,
    })
    final = run_finished(base, run["id"])
    assert final["status"] == "completed" and final["output"] == "[test mode] hello"
    swarm = request(base + f"/api/agents/runs/{run['id']}/swarm")
    assert swarm["status"] == "completed" and swarm["strategy"] == "peer_review"
    assert swarm["agents"] == [] and swarm["messages"] == []
    assert request_error(base + f"/api/agents/runs/{run['id']}/swarm/messages",
                         {"text": "hello"})[0] == 409


def test_human_input_breakpoints_and_drafting_over_http(api):
    server, base, _ = api
    nodes, edges = chain(
        {"id": "review", "type": "human", "data": {"prompt": "Proceed?", "kind": "approve"}},
        {"id": "after", "type": "note", "data": {"text": "n"}},
        output="{{ nodes.review.approved }}",
    )
    request(base + "/api/agents", minimal_agent("gate", nodes=nodes, edges=edges))
    run = request(base + "/api/agents/gate/runs", {"input": {}, "debug": True, "breakpoints": []})
    asked = wait_for(lambda: (lambda snap: snap if snap["awaiting_input"] else None)(
        request(base + f"/api/agents/runs/{run['id']}")))
    assert asked["awaiting_input"]["prompt"] == "Proceed?"
    assert request_error(base + "/api/agents/gate", method="DELETE")[0] == 409  # a run is active
    status, error = request_error(base + f"/api/agents/runs/{run['id']}/input", {"value": "perhaps"})
    assert status == 409 and "true (approve)" in error["error"]
    status, error = request_error(base + f"/api/agents/runs/{run['id']}/breakpoints", {"breakpoints": "after"})
    assert status == 400 and "list of node ids" in error["error"]
    status, error = request_error(base + "/api/agents/gate/runs", {"dry_run": "yes"})
    assert status == 400 and "dry_run must be" in error["error"]
    status, error = request_error(base + "/api/agents/gate/runs", {"breakpoints": "after"})
    assert status == 400 and "list of node ids" in error["error"]
    changed = request(base + f"/api/agents/runs/{run['id']}/breakpoints", {"breakpoints": ["after"]})
    assert changed["breakpoints"] == ["after"]
    request(base + f"/api/agents/runs/{run['id']}/input", {"value": True})
    paused = wait_for(lambda: (lambda snap: snap if snap["waiting_node"] == "after" else None)(
        request(base + f"/api/agents/runs/{run['id']}")))
    assert paused["paused"] is True
    request(base + f"/api/agents/runs/{run['id']}/resume", {})
    final = run_finished(base, run["id"])
    assert final["status"] == "completed" and final["output"] is True
    assert request_error(base + f"/api/agents/runs/{run['id']}/input", {"value": True})[0] == 409

    draft = {"name": "Echo", "nodes": [{"id": "start", "type": "start"}, {"id": "end", "type": "end"}],
             "edges": [{"from": "start", "to": "end"}]}
    server.agents.gateway = ScriptedGateway([json.dumps(draft)])
    generated = request(base + "/api/agents/generate", {"description": "Return whatever comes in."})
    assert generated["agent"]["name"] == "Echo" and generated["attempts"] == 1
    assert request(base + "/api/agents")["agents"][0]["id"] == "gate"  # a draft is not saved
    status, error = request_error(base + "/api/agents/generate", {"description": "x"})
    assert status == 400 and "Describe what the agent should do" in error["error"]


def test_chat_completes_streams_and_validates_before_streaming(api):
    _, base, root = api
    tiny_drv_checkpoint(root / "trained-model" / "tiny.pt")
    config = {"model": "trained-model/tiny.pt", "message": "Hello", "max_tokens": 2,
              "temperature": 0, "device": "cpu"}
    result = request(base + "/api/chat", config)
    assert result["kind"] == "DRV-LM" and result["reply"] and result["model_id"] == "trained-model/tiny.pt"
    events = read_events(base + "/api/chat/stream", config)
    assert events[0]["type"] == "start" and events[0]["name"] == "tiny.pt"
    assert events[-1]["type"] == "done" and events[-1]["usage"]["completion_tokens"] == 2
    assert events[-1]["reply"] == "".join(e["text"] for e in events if e["type"] == "delta") or events[-1]["empty"]
    status, error = request_error(base + "/api/chat/stream", {**config, "message": ""})
    assert status == 400 and error == {"error": "Enter a message first", "code": "bad_request"}
    status, error = request_error(base + "/api/chat/stream", {**config, "model": "missing"})
    assert status == 400 and "Model not found" in error["error"]


def test_a_failure_after_the_stream_opened_arrives_as_an_error_event(api):
    server, base, _ = api
    provider = mock_provider({
        ("POST", "/v1/chat/completions"): lambda handler, payload: (500, {"error": {"message": "boom"}}),
    })
    with serving(provider) as upstream:
        request(base + "/api/models/endpoints", {
            "name": "Flaky", "kind": "openai", "base_url": f"{upstream}/v1", "model": "m"})
        events = read_events(base + "/api/chat/stream", {"model": "endpoint:flaky", "message": "hi"})
    assert [event["type"] for event in events] == ["start", "error"]
    assert events[1]["code"] == "upstream_error" and "HTTP 500: boom" in events[1]["error"]


def test_conversations_settings_and_browsing(api):
    _, base, root = api
    created = request(base + "/api/chats", {
        "model": "m", "messages": [{"role": "user", "content": "First question"}]})["chat"]
    assert created["title"] == "First question"
    big = {"messages": [{"role": "user", "content": "x" * 300_000}] * 5}  # above the 1 MB default
    assert len(request(base + f"/api/chats/{created['id']}", big, method="PUT")["chat"]["messages"]) == 5
    assert request(base + "/api/chats")["chats"][0]["messages"] == 5
    assert request(base + f"/api/chats/{created['id']}")["chat"]["id"] == created["id"]
    workspace = chat_files_path(root, created["id"], create=True)
    (workspace / "index.html").write_text("<!DOCTYPE html><title>Hi</title>", encoding="utf-8")
    preview = request(base + f"/api/chats/{created['id']}/file?path=index.html")["file"]
    assert preview["kind"] == "text" and preview["content"].startswith("<!DOCTYPE html>")
    status, error = request_error(base + f"/api/chats/{created['id']}/file?path=../settings.json")
    assert status == 400 and "outside the conversation workspace" in error["error"]
    assert request(base + f"/api/chats/{created['id']}", method="DELETE")["deleted"] is True
    assert request_error(base + f"/api/chats/{created['id']}")[0] == 404
    assert request_error(base + "/api/chats/..%2F..%2Fsettings")[0] == 404
    assert request(base + "/api/chats", method="DELETE") == {"deleted": 0}

    assert request(base + "/api/settings")["settings"]["discover_local"] is False
    assert request(base + "/api/settings")["settings"]["allow_os_access"] is False
    updated = request(base + "/api/settings", {"idle_unload_minutes": 3, "model_dirs": [str(root)],
                                                "allow_os_access": True},
                      method="PUT")["settings"]
    assert updated["idle_unload_minutes"] == 3 and updated["model_dirs"] == [str(root)]
    assert updated["allow_os_access"] is True
    for patch, message in (({"unknown": 1}, "Unknown setting"), ({"model_dirs": ["/no/such"]}, "does not exist"),
                           ({"gguf_runtime": "magic"}, "gguf_runtime must be"),
                           ({"idle_unload_minutes": -1}, "between 0 and 1440")):
        status, error = request_error(base + "/api/settings", patch, method="PUT")
        assert status == 400 and message in error["error"]

    (root / "data").mkdir()
    (root / "data" / "notes.txt").write_text("x")
    (root / ".hidden").mkdir()
    listing = request(base + f"/api/browse?path={root}&files=1")
    assert listing["path"] == str(root) and listing["parent"] == str(root.parent)
    assert {"name": "data", "path": str(root / "data")} in listing["directories"]
    assert not any(item["name"].startswith(".") for item in listing["directories"])
    inside = request(base + f"/api/browse?path={root / 'data' / 'notes.txt'}&files=1")
    assert inside["path"] == str(root / "data") and inside["files"][0]["name"] == "notes.txt"
    assert request(base + f"/api/browse?path={root / 'data'}")["files"] == []
    assert request_error(base + "/api/browse?path=/no/such/place")[0] == 400


def test_models_endpoints_runtimes_and_starter_lists(api):
    _, base, root = api
    tiny_drv_checkpoint(root / "trained-model" / "tiny.pt")
    listing = request(base + "/api/models?refresh=1")
    assert [entry["name"] for entry in listing["models"]] == ["tiny.pt"] and listing["endpoints"] == []
    assert request(base + "/api/models/describe?id=trained-model/tiny.pt")["kind"] == "mariana-drv"
    assert request_error(base + "/api/models/describe?id=nope")[0] == 404
    providers = {item["id"]: item for item in request(base + "/api/models/providers")["providers"]}
    assert providers["anthropic"]["default_model"] == "claude-opus-5-5" and "custom" in providers
    runtimes = request(base + "/api/models/runtimes")["runtimes"]
    assert {"llama_cpp", "transformers", "anthropic", "huggingface", "servers", "loaded"} <= set(runtimes)
    assert runtimes["llama_cpp"]["managed"]["running"] is False
    featured = request(base + "/api/models/featured")
    assert all(item["format"] in {"gguf", "transformers"} for item in featured["huggingface"])
    assert featured["ollama"] and request(base + "/api/models/unload", {})["unloaded"] == 0

    provider = mock_provider({
        ("GET", "/v1/models"): lambda handler, payload: (200, {"data": [{"id": "served-model"}]}),
        ("POST", "/v1/chat/completions"): lambda handler, payload: (200, {
            "choices": [{"message": {"content": "ready"}}], "usage": {"completion_tokens": 1}}),
    })
    with serving(provider) as upstream:
        models = request(base + "/api/models/providers/models", {"kind": "openai", "base_url": f"{upstream}/v1"})
        assert models == {"models": [{"id": "served-model", "context_length": None}]}
        saved = request(base + "/api/models/endpoints", {
            "name": "Mock", "kind": "openai", "base_url": f"{upstream}/v1", "model": "served-model",
            "api_key": "never-returned"})
        assert saved["endpoints"][0]["has_api_key"] is True and "api_key" not in saved["endpoints"][0]
        assert request(base + "/api/models/test", {"id": "endpoint:mock"})["reply"] == "ready"
    assert "never-returned" not in json.dumps(request(base + "/api/models"))
    assert request(base + "/api/models/endpoints/endpoint%3Amock", method="DELETE") == {"endpoints": []}
    assert request_error(base + "/api/models/endpoints/endpoint%3Amock", method="DELETE")[0] == 404
    assert request_error(base + "/api/models/download", {"repo": "../etc", "files": ["x"]})[0] == 400
    assert request_error(base + "/api/models/search?format=exe")[0] == 400
    status, error = request_error(base + "/api/models/providers/models",
                                  {"kind": "openai", "base_url": "http://127.0.0.1:9/v1"})
    assert status == 503 and error["code"] == "unavailable"


def test_fine_tuning_jobs_and_deploy_routes(api):
    _, base, root = api
    tiny_drv_checkpoint(root / "trained-model" / "base.pt")
    (root / "models").mkdir()
    (root / "models" / "quantized.gguf").write_bytes(b"gguf")
    corpus = root / "corpus"
    corpus.mkdir()
    (corpus / "notes.txt").write_text("Fine tuning text. " * 100)
    bases = request(base + "/api/finetune/bases?refresh=1")
    assert [item["name"] for item in bases["tunable"]] == ["base.pt"]
    assert bases["other"][0]["name"] == "quantized.gguf" and "quantized" in bases["other"][0]["reason"]
    assert request(base + "/api/finetune/inspect?id=trained-model/base.pt")["engine"] == "mariana"
    assert request(base + "/api/finetune/inspect?path=trained-model/base.pt")["name"] == "base.pt"
    plan = request(base + "/api/finetune/plan", {"base": "trained-model/base.pt", "folder": str(corpus)})
    assert plan["output"] == "trained-model/base-tuned.pt" and plan["estimate"]["trainable"] > 0
    status, error = request_error(base + "/api/finetune/plan", {"base": "models/quantized.gguf",
                                                                "folder": str(corpus)})
    assert status == 400 and "quantized inference format" in error["error"]
    assert request_error(base + "/api/finetune/start", {"base": "trained-model/base.pt", "folder": ""})[0] == 400

    assert request(base + "/api/jobs") == {"jobs": [], "version": request(base + "/api/jobs")["version"]}
    assert request_error(base + "/api/jobs/nope")[0] == 404
    assert request_error(base + "/api/jobs/nope/cancel", {})[0] == 404
    assert request(base + "/api/jobs/clear", {}) == {"cleared": 0}

    discovery = request(base + "/api/finetune/discovery")
    assert discovery["installed"] == [] and discovery["featured"]["datasets"]
    assert discovery["featured"]["models"] and discovery["featured"]["external"]
    status, error = request_error(base + "/api/finetune/discovery?platform=magic")
    assert status == 400 and "Platform must be one of" in error["error"]
    assert request_error(base + "/api/finetune/datasets/download", {"repo": "../etc"})[0] == 400

    deploy = request(base + "/api/deploy/exports")
    assert deploy["exports"] == [] and deploy["checkpoints"][0]["name"] == "base.pt"
    assert request_error(base + "/api/deploy/export", {"model": "models/quantized.gguf"})[0] == 400
    assert request_error(base + "/api/deploy/ollama", {"file": "nope.gguf"})[0] == 404


def test_openai_compatible_api_serves_catalog_models(api):
    _, base, root = api
    tiny_drv_checkpoint(root / "trained-model" / "tiny.pt")
    models = request(base + "/v1/models")
    assert models["object"] == "list" and models["data"][0]["id"] == "trained-model/tiny.pt"
    body = {"model": "trained-model/tiny.pt", "messages": [{"role": "user", "content": "Hi"}],
            "max_completion_tokens": 2, "temperature": 0, "device": "cpu"}
    completion = request(base + "/v1/chat/completions", body)
    assert completion["object"] == "chat.completion" and completion["id"].startswith("chatcmpl-")
    assert completion["choices"][0]["message"]["role"] == "assistant"
    assert completion["usage"]["completion_tokens"] == 2
    assert completion["usage"]["total_tokens"] == completion["usage"]["prompt_tokens"] + 2
    chunks = read_events(base + "/v1/chat/completions", {**body, "stream": True})
    assert chunks[-1] == "[DONE]" and chunks[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    assert all(chunk["object"] == "chat.completion.chunk" for chunk in chunks[:-1])
    assert chunks[-2]["choices"][0]["finish_reason"] == "stop" and chunks[-2]["usage"]["completion_tokens"] == 2
    status, error = request_error(base + "/v1/chat/completions", {**body, "model": "missing"})
    assert status == 400 and error["error"]["type"] == "invalid_request_error"
    assert "Model not found" in error["error"]["message"]


def test_system_overview_and_route_index(api):
    _, base, root = api
    system = request(base + "/api/system")
    assert system["name"] == "Mariana Studio" and system["root"] == str(root)
    assert system["counts"] == {"models": 0, "chat_ready": 0, "tunable": 0, "agents": 0,
                                "chats": 0, "active_jobs": 0}
    assert system["disk"]["free"] > 0 and system["software"]["torch"] and system["device"]["type"]
    events = read_events(base + "/api/system/events", limit=1)
    assert len(events) == 1 and events[0]["type"] == "stats"
    assert "memory" in events[0] and events[0]["disk"]["free"] > 0
    routes = request(base + "/api/routes")["routes"]
    index = {(route["method"], route["path"]) for route in routes}
    assert {("POST", "/api/chat/stream"), ("GET", "/api/models"), ("POST", "/api/finetune/start"),
            ("POST", "/api/agents/{id}/runs"), ("POST", "/v1/chat/completions"),
            ("GET", "/api/chats/{id}/file"),
            ("PUT", "/api/settings"), ("GET", "/api/jobs/events"),
            ("GET", "/api/system/events")} <= index
    assert all(route["summary"] for route in routes), [r for r in routes if not r["summary"]]
    assert len(index) == len(routes)  # no route is registered twice
