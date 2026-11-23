"""AML language, the agent runtime, abilities, schedules, debugging and authoring."""

from __future__ import annotations

import copy
import json
import threading
import time
from datetime import datetime, timedelta

import pytest

from mariana.agents import aml, runtime
from mariana.agents.aml import (
    ExpressionError,
    evaluate_expression,
    next_cron,
    parse_cron,
    render_template,
    validate_agent,
)
from mariana.agents.authoring import generate_agent
from mariana.agents.manager import AgentManager
from mariana.agents.runtime import LiveRun, extract_json_object, extract_json_value
from mariana.agents.store import AgentStore
from mariana.agents.templates import list_templates
from mariana.agents.tools import ToolError, execute_ability, html_to_text
from tests.helpers import mock_provider, serving, wait_for

ACTIVE = {"queued", "running", "paused"}


class ScriptedGateway:
    """A chat gateway that answers from a script and records what it was asked."""

    def __init__(self, replies=(), models=("fake",)):
        self.replies = list(replies)
        self.models = models
        self.calls: list[dict] = []

    def catalog(self, force=False):
        return [{"id": model, "available": True, "capabilities": {"chat": True}} for model in self.models]

    def chat(self, config):
        self.calls.append(copy.deepcopy(config))  # the runtime keeps appending to its message list
        reply = self.replies.pop(0) if self.replies else '{"final": "script exhausted"}'
        return {"reply": reply, "usage": {"prompt_tokens": 3, "completion_tokens": 2}, "elapsed": 0.01}


class FallbackGateway(ScriptedGateway):
    """Fail one model id and answer normally on the fallback."""

    def __init__(self, broken="primary", reply="backup answer"):
        super().__init__(models=(broken, "backup"))
        self.broken = broken
        self.reply = reply

    def chat(self, config):
        self.calls.append(copy.deepcopy(config))
        if config["model"] == self.broken:
            raise ValueError(f"{self.broken} is offline")
        return {"reply": self.reply, "usage": {"prompt_tokens": 3, "completion_tokens": 2},
                "elapsed": 0.01}


class StreamingGateway(ScriptedGateway):
    """A gateway that streams slowly so cancellation can interrupt it."""

    def __init__(self):
        super().__init__()
        self.started = threading.Event()
        self.stopped = False

    def stream(self, config, cancelled=None):
        self.calls.append(copy.deepcopy(config))
        self.started.set()
        yield {"type": "start"}
        for _ in range(500):
            if cancelled.is_set():
                self.stopped = True
                break
            time.sleep(0.01)
        yield {"type": "done", "reply": "partial", "usage": {"completion_tokens": 0}, "elapsed": 0.0}


class SlowGateway(ScriptedGateway):
    """A gateway that ignores cancellation long enough to test run deadlines."""

    def __init__(self, delay: float = 1.2):
        super().__init__()
        self.delay = delay

    def stream(self, config, cancelled=None):
        self.calls.append(copy.deepcopy(config))
        time.sleep(self.delay)
        yield {"type": "done", "reply": "too late",
               "usage": {"completion_tokens": 1}, "elapsed": self.delay}


@pytest.fixture()
def manager(tmp_path):
    instance = AgentManager(tmp_path, ScriptedGateway())
    yield instance
    instance.close()


def minimal_agent(agent_id="probe", schedule=None, nodes=None, edges=None, abilities=None):
    return {
        "aml": "1.1", "kind": "agent", "id": agent_id, "name": "Probe",
        "settings": {"max_steps": 30, "timeout": 30},
        "abilities": abilities or [],
        "nodes": nodes or [
            {"id": "start", "type": "start", "x": 0, "y": 0},
            {"id": "end", "type": "end", "x": 220, "y": 0, "data": {"output": "{{ input.message }}"}},
        ],
        "edges": edges or [{"from": "start", "to": "end", "when": "always"}],
        "schedule": schedule or {"enabled": False, "type": "manual"},
    }


def chain(*middle, output="{{ last }}"):
    """start -> the given nodes in order -> end."""
    nodes = [{"id": "start", "type": "start"}, *middle,
             {"id": "end", "type": "end", "data": {"output": output}}]
    edges = [{"from": a["id"], "to": b["id"]} for a, b in zip(nodes, nodes[1:])]
    return nodes, edges


def finished(manager, run_id, timeout=10.0):
    return wait_for(lambda: (lambda snap: snap if snap["status"] not in ACTIVE else None)(
        manager.get_run(run_id)), timeout=timeout)


def waiting(manager, run_id):
    return wait_for(lambda: (lambda snap: snap if snap["awaiting_input"] else None)(manager.get_run(run_id)))


def test_every_starter_template_completes_in_test_mode(manager):
    templates = list_templates()
    for template in templates:
        manager.save_agent(template["aml"])
    for template in templates:
        final = finished(manager, manager.create_run(template["id"], {
            "input": {"message": "hello"}, "dry_run": True,
        })["id"])
        assert final["status"] == "completed", (template["id"], final["error"])


def test_templates_and_validation():
    templates = list_templates()
    known = {item["id"] for item in templates}
    assert known >= {"hello-agent", "support-triage", "daily-research-digest", "content-pipeline",
                     "research-assistant", "approval-gate"}
    for template in templates:
        result = validate_agent(template["aml"], known)
        assert result["ok"], (template["id"], result["errors"])
        assert template["nodes"] == len(template["aml"]["nodes"])
    duplicate = minimal_agent()
    duplicate["nodes"].append({"id": "end", "type": "end", "x": 1, "y": 1})
    with pytest.raises(ValueError, match="Duplicate node id"):
        validate_agent(duplicate)
    bad_ability = minimal_agent()
    bad_ability["nodes"].append({"id": "tool-1", "type": "tool", "x": 1, "y": 1,
                                 "data": {"ability": "missing"}})
    assert "missing ability" in " ".join(validate_agent(bad_ability)["errors"])
    legacy = minimal_agent()
    legacy["aml"] = "1.0"
    assert validate_agent(legacy)["ok"] and validate_agent(legacy)["agent"]["aml"] == "1.8"


def test_validation_issues_name_the_node_and_field():
    nodes, edges = chain(
        {"id": "ask", "type": "human", "data": {"kind": "choice", "timeout": "soon"}},
        {"id": "think", "type": "agent", "data": {"abilities": ["ghost"], "max_iterations": 99}},
        {"id": "each", "type": "foreach", "data": {"agent": "probe", "max_items": 0}},
    )
    edges.append({"from": "think", "to": "end"})  # a second edge out of a non-condition node
    document = minimal_agent(nodes=nodes, edges=edges)
    document["aml"] = "9.0"
    result = validate_agent(document)
    assert result["ok"] is False
    issues: dict[tuple, str] = {}
    for issue in result["issues"]:
        key = (issue["node"], issue["field"])
        issues[key] = issues.get(key, "") + f"[{issue['level']}] {issue['message']} "
    assert issues[("ask", "choices")].startswith("[error]")
    assert "timeout must be a number" in issues[("ask", "timeout")]
    assert "missing ability ghost" in issues[("think", "abilities")]
    assert "[warning] AI agent node think has no abilities" in issues[("think", "abilities")]
    assert "between 1 and 40" in issues[("think", "max_iterations")]
    assert "cannot call itself" in issues[("each", "agent")]
    assert "between 1 and 200" in issues[("each", "max_items")]
    branching = next(issue for issue in result["issues"] if "only the first" in issue["message"])
    assert branching["level"] == "warning" and branching["node"] == "think"
    assert any("declares AML 9.0" in warning for warning in result["warnings"])
    assert result["errors"] == [issue["message"] for issue in result["issues"] if issue["level"] == "error"]
    assert set(result["issues"][0]) == {"level", "message", "node", "edge", "field"}
    dangling = minimal_agent(edges=[{"from": "start", "to": "nowhere"}])
    assert validate_agent(dangling)["issues"][0]["edge"] == ["start", "nowhere"]


def test_safe_expression_language_and_templates():
    context = {
        "input": {"message": "Hello URGENT", "name": ""},
        "nodes": {"classify": {"text": "urgent"}},
        "vars": {"count": 3},
    }
    assert evaluate_expression("contains(lower(input.message), 'urgent')", context) is True
    assert evaluate_expression("nodes.classify.text", context) == "urgent"
    assert evaluate_expression("vars.count * 2", context) == 6
    assert render_template("{{ input.message }}", context) == "Hello URGENT"
    assert render_template("Count: {{ vars.count }}", context) == "Count: 3"
    assert render_template("{{ vars.count }} and {{ input.message }}", context) == "3 and Hello URGENT"
    assert render_template("{{ {'a': 1} }}", context) == {"a": 1}
    # and / or behave as in Python: they short-circuit and return the deciding operand.
    assert render_template("{{ input.name or 'anonymous' }}", context) == "anonymous"
    assert evaluate_expression("vars.count and vars.count + 1", context) == 4
    assert evaluate_expression("vars.missing and vars.missing.deep", {"vars": {}}, lenient=True) is None
    assert evaluate_expression("input.name and input.name.upper()", context) == ""
    assert evaluate_expression("1 and 2 and 3", {}) == 3 and evaluate_expression("0 or [] or 5", {}) == 5
    with pytest.raises(ExpressionError):
        evaluate_expression("__import__('os').system('true')", context)
    with pytest.raises(ExpressionError):
        evaluate_expression("open('/etc/passwd')", context)
    with pytest.raises(ExpressionError):
        evaluate_expression("input.__class__", context)
    # Lenient mode is used for validation: unknown runtime fields must not block saving.
    assert evaluate_expression("nodes.not_run_yet.text", {"nodes": {}}, lenient=True) is None


def test_cron_parser_and_next_run():
    fields = parse_cron("*/15 9-10 * * mon-fri")
    assert 0 in fields["minute"] and 45 in fields["minute"] and 7 not in fields["minute"]
    monday = datetime(2026, 1, 5, 8, 59)
    assert next_cron("0 9 * * 1-5", monday) == datetime(2026, 1, 5, 9, 0)
    with pytest.raises(ValueError):
        parse_cron("0 9 * *")


def test_tool_sandbox_and_dry_run(tmp_path):
    context = {"agent": {"id": "tool-test"}, "input": {}}
    write = {"id": "write", "type": "file.write", "config": {}}
    with pytest.raises(ToolError, match="restricted"):
        execute_ability(write, {"path": "../escape.txt", "content": "x"}, context, tmp_path)
    result = execute_ability(
        write, {"path": "safe.txt", "content": "hello"}, context, tmp_path, dry_run=True,
    )
    assert result["dry_run"] is True
    assert not (tmp_path / "agent-workspace" / "files" / "safe.txt").exists()
    execute_ability(write, {"path": "safe.txt", "content": "hello"}, context, tmp_path)
    assert (tmp_path / "agent-workspace" / "files" / "safe.txt").read_text() == "hello"
    read = {"id": "read", "type": "file.read", "config": {}}
    assert execute_ability(read, {"path": "safe.txt"}, context, tmp_path)["content"] == "hello"
    fetch = {"id": "fetch", "type": "http.fetch", "config": {}}
    with pytest.raises(ToolError, match="Private addresses"):
        execute_ability(fetch, {"url": "http://127.0.0.1:9/"}, context, tmp_path)


def test_http_fetch_can_return_readable_text(tmp_path):
    page = ("<html><head><title>T</title><style>p{color:red}</style></head><body>"
            "<script>alert(1)</script><h1>Heading</h1><p>First   paragraph &amp; more.</p>"
            "<ul><li>One</li><li>Two</li></ul></body></html>")
    provider = mock_provider({
        ("GET", "/page"): lambda handler, payload: (200, page.encode(), {"Content-Type": "text/html"}),
        ("GET", "/plain"): lambda handler, payload: (200, b"just text " * 50, {"Content-Type": "text/plain"}),
    })
    fetch = {"id": "fetch", "type": "http.fetch", "config": {"allow_private": True}}
    context = {"agent": {"id": "reader"}}
    with serving(provider) as base:
        readable = execute_ability(fetch, {"url": f"{base}/page", "extract": "text"}, context, tmp_path)
        raw = execute_ability(fetch, {"url": f"{base}/page"}, context, tmp_path)
        short = execute_ability(fetch, {"url": f"{base}/plain", "extract": "text", "max_chars": 120},
                                context, tmp_path)
    assert readable["text"] == "Heading\nFirst paragraph & more.\nOne\nTwo"
    assert "body" not in readable and readable["ok"] is True
    assert raw["body"] == page and "text" not in raw
    assert len(short["text"]) == 120
    assert html_to_text("<p>" + "word " * 100 + "</p>", max_chars=40).endswith("…")
    assert html_to_text("<div>unclosed <b>markup") == "unclosed markup"


def test_agent_runtime_dry_run_and_debugger(manager):
    manager.save_agent(list_templates()[0]["aml"])
    run = manager.create_run("hello-agent", {"input": {"message": "hi"}, "dry_run": True})
    final = finished(manager, run["id"])
    assert final["status"] == "completed" and "[test mode]" in final["output"]
    assert final["steps"] == 3

    debug = manager.create_run("hello-agent", {
        "input": {"message": "step me"}, "dry_run": True, "debug": True, "breakpoints": ["greet"],
    })
    paused = wait_for(lambda: (lambda snap: snap if snap["paused"] else None)(manager.get_run(debug["id"])))
    assert paused["status"] == "paused" and paused["waiting_node"] == "greet"
    assert any(event["type"] == "breakpoint" for event in paused["events"])
    manager.step_run(debug["id"])
    wait_for(lambda: (lambda snap: snap if snap["event_count"] > paused["event_count"] else None)(
        manager.get_run(debug["id"])))
    manager.resume_run(debug["id"])
    assert finished(manager, debug["id"])["status"] == "completed"


def test_debugger_exposes_variables_timing_and_live_breakpoints(manager):
    nodes, edges = chain(
        {"id": "remember", "type": "transform", "data": {"assign": {"topic": "{{ input.message }}"}}},
        {"id": "ask", "type": "llm", "data": {"prompt": "About {{ vars.topic }}"}},
        {"id": "pause-here", "type": "note", "data": {"text": "n"}},
        output="{{ vars.topic }}",
    )
    manager.save_agent(minimal_agent("debuggable", nodes=nodes, edges=edges))
    run = manager.create_run("debuggable", {
        "input": {"message": "tides"}, "dry_run": True, "debug": True, "breakpoints": ["ask"],
    })
    paused = wait_for(lambda: (lambda snap: snap if snap["paused"] else None)(manager.get_run(run["id"])))
    assert paused["vars"] == {"topic": "tides"} and paused["awaiting_input"] is None
    assert paused["nodes"]["remember"] == "ok" and paused["nodes"]["ask"] == "paused"
    changed = manager.set_breakpoints(run["id"], ["pause-here"])
    assert changed["breakpoints"] == ["pause-here"]
    with pytest.raises(ValueError, match="Unknown breakpoint"):
        manager.set_breakpoints(run["id"], ["missing"])
    manager.resume_run(run["id"])
    again = wait_for(lambda: (lambda snap: snap if snap["waiting_node"] == "pause-here" else None)(
        manager.get_run(run["id"])))
    assert again["paused"] is True
    manager.resume_run(run["id"])
    final = finished(manager, run["id"])
    assert final["status"] == "completed" and final["output"] == "tides"
    events = {event["type"]: event for event in final["events"]}
    assert events["node_input"]["node"] == "ask" and events["node_input"]["input"] == "About tides"
    assert isinstance(events["node_finished"]["elapsed_ms"], int)
    with pytest.raises(ValueError, match="not active"):
        manager.set_breakpoints(run["id"], [])


def test_ai_agent_node_uses_abilities_until_it_has_an_answer(tmp_path):
    gateway = ScriptedGateway([
        '{"tool": "calc", "input": {"expression": "6 * 7"}}',
        'I will save it.\n```json\n{"tool": "nonexistent", "input": {}}\n```',
        '```json\n{"tool": "save", "input": {"path": "answer.txt", "content": "42"}}\n```',
        '{"final": "The answer is 42 and it is saved."}',
    ])
    manager = AgentManager(tmp_path, gateway)
    try:
        nodes, edges = chain({"id": "solve", "type": "agent", "data": {
            "system": "Be exact.", "goal": "Compute {{ input.message }} and save it.",
            "abilities": ["calc", "save"], "max_iterations": 6, "temperature": 0, "max_tokens": 200,
        }}, output="{{ nodes.solve.text }}")
        manager.save_agent(minimal_agent("solver", nodes=nodes, edges=edges, abilities=[
            {"id": "calc", "type": "math.eval"}, {"id": "save", "type": "file.write"},
            {"id": "hidden", "type": "time.now"},
        ]))
        final = finished(manager, manager.create_run("solver", {"input": {"message": "6 times 7"}})["id"])
    finally:
        manager.close()
    assert final["status"] == "completed", final["error"]
    assert final["output"] == "The answer is 42 and it is saved."
    assert (tmp_path / "agent-workspace" / "files" / "answer.txt").read_text() == "42"
    finish = next(event for event in final["events"]
                  if event["type"] == "node_finished" and event["node"] == "solve")["output"]
    assert finish["iterations"] == 4 and finish["exhausted"] is False and finish["model"] == "fake"
    assert finish["usage"] == {"prompt_tokens": 12, "completion_tokens": 8}
    assert [step["ability"] for step in finish["steps"]] == ["calc", "nonexistent", "save"]
    assert finish["steps"][0]["output"]["value"] == 42
    assert "Unknown tool 'nonexistent'" in finish["steps"][1]["error"]
    calls = [event for event in final["events"] if event["type"] == "tool_call"]
    results = [event for event in final["events"] if event["type"] == "tool_result"]
    assert [event["ability"] for event in calls] == ["calc", "nonexistent", "save"]
    assert [event["ok"] for event in results] == [True, False, True]
    system = gateway.calls[0]["messages"][0]["content"]
    assert system.startswith("Be exact.") and "- calc (math.eval)" in system and "- save (file.write)" in system
    assert "hidden" not in system  # only the abilities the node was given
    assert gateway.calls[0]["messages"][1] == {"role": "user", "content": "Compute 6 times 7 and save it."}
    assert gateway.calls[0]["temperature"] == 0.0 and gateway.calls[0]["max_tokens"] == 200
    assert "Tool result for calc" in gateway.calls[1]["messages"][-1]["content"]
    assert '"value": 42' in gateway.calls[1]["messages"][-1]["content"]


def test_ai_agent_node_accepts_deepseek_dsml_tool_calls(tmp_path):
    gateway = ScriptedGateway([
        '<｜｜DSML｜｜ calls>\n'
        '<｜｜DSML｜｜ invoke name="calls">\n'
        '<｜｜DSML｜｜ parameter name="input" string="false">{"expression": "6 * 7"}</｜｜DSML｜｜ parameter>\n'
        '<｜｜DSML｜｜ parameter name="tool" string="true">calc</｜｜DSML｜｜ parameter>\n'
        '</｜｜DSML｜｜ invoke>\n'
        '</｜｜DSML｜｜ calls>',
        '{"final": "42"}',
    ])
    manager = AgentManager(tmp_path, gateway)
    try:
        nodes, edges = chain({"id": "solve", "type": "agent", "data": {
            "goal": "Compute", "abilities": ["calc"], "max_iterations": 4,
        }}, output="{{ nodes.solve.text }}")
        manager.save_agent(minimal_agent("solver", nodes=nodes, edges=edges, abilities=[
            {"id": "calc", "type": "math.eval"},
        ]))
        final = finished(manager, manager.create_run("solver", {"input": {}})["id"])
    finally:
        manager.close()
    assert final["status"] == "completed", final["error"]
    assert final["output"] == "42"
    result = next(event for event in final["events"]
                  if event["type"] == "node_finished" and event["node"] == "solve")["output"]
    assert result["iterations"] == 2
    assert [step["ability"] for step in result["steps"]] == ["calc"]
    assert result["steps"][0]["output"]["value"] == 42
    assert any(tool["id"] == "calc" for tool in gateway.calls[0]["tools"])


def test_ai_agent_node_plans_and_batches_tool_calls(tmp_path):
    gateway = ScriptedGateway([
        '{"plan": [{"text": "Read the clock", "status": "active"}]}',
        '{"calls": [{"tool": "clock", "input": {}}, '
        '{"tool": "calc", "input": {"expression": "2 + 2"}}]}',
        '{"final": "Both tools ran."}',
    ])
    manager = AgentManager(tmp_path, gateway)
    try:
        nodes, edges = chain({"id": "solve", "type": "agent", "data": {
            "abilities": ["clock", "calc"], "max_iterations": 5,
        }}, output="{{ nodes.solve.text }}")
        manager.save_agent(minimal_agent("batcher", nodes=nodes, edges=edges, abilities=[
            {"id": "clock", "type": "time.now"}, {"id": "calc", "type": "math.eval"},
        ]))
        run_id = manager.create_run("batcher", {"input": {"message": "go"}})["id"]
        final = finished(manager, run_id)
    finally:
        manager.close()
    assert final["output"] == "Both tools ran."
    plan_events = [event for event in final["events"] if event["type"] == "plan"]
    assert plan_events and plan_events[0]["plan"][0]["status"] == "active"
    finish = next(event for event in final["events"]
                  if event["type"] == "node_finished" and event["node"] == "solve")["output"]
    assert [step["ability"] for step in finish["steps"]] == ["clock", "calc"]
    assert "Tool results:" in gateway.calls[2]["messages"][-1]["content"]


def test_ai_agent_node_retries_a_temporary_read_failure(tmp_path, monkeypatch):
    gateway = ScriptedGateway([
        '{"tool": "read", "input": {"path": "x.txt"}}',
        '{"final": "Recovered."}',
    ])
    calls = []

    def flaky(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise TimeoutError("connection timed out")
        return {"path": "x.txt", "content": "hello"}

    monkeypatch.setattr(runtime, "execute_ability", flaky)
    manager = AgentManager(tmp_path, gateway)
    try:
        nodes, edges = chain({"id": "solve", "type": "agent", "data": {
            "abilities": ["read"], "max_iterations": 4,
        }}, output="{{ nodes.solve.text }}")
        manager.save_agent(minimal_agent("retrying", nodes=nodes, edges=edges, abilities=[
            {"id": "read", "type": "file.read"},
        ]))
        final = finished(manager, manager.create_run("retrying", {"input": {"message": "go"}})["id"])
    finally:
        manager.close()

    assert final["status"] == "completed", final.get("error")
    assert final["output"] == "Recovered."
    assert len(calls) == 2
    results = [event for event in final["events"] if event["type"] == "tool_result"]
    assert len(results) == 1 and results[0]["attempts"] == 2 and results[0]["ok"] is True
    assert any(event["type"] == "log" and "Retrying read" in event.get("message", "")
               for event in final["events"])


def test_ai_agent_node_hints_against_identical_repeated_calls(tmp_path, monkeypatch):
    gateway = ScriptedGateway([
        '{"tool": "echo", "input": {"value": 1}}',
        '{"tool": "echo", "input": {"value": 1}}',
        '{"tool": "echo", "input": {"value": 1}}',
        '{"final": "Done."}',
    ])
    monkeypatch.setattr(runtime, "execute_ability",
                        lambda *args, **kwargs: {"value": kwargs.get("value", 1)})
    manager = AgentManager(tmp_path, gateway)
    try:
        nodes, edges = chain({"id": "solve", "type": "agent", "data": {
            "abilities": ["echo"], "max_iterations": 6,
        }}, output="{{ nodes.solve.text }}")
        manager.save_agent(minimal_agent("looping", nodes=nodes, edges=edges, abilities=[
            {"id": "echo", "type": "math.eval"},
        ]))
        final = finished(manager, manager.create_run("looping", {"input": {"message": "go"}})["id"])
    finally:
        manager.close()

    assert final["status"] == "completed", final.get("error")
    results = [event for event in final["events"] if event["type"] == "tool_result"]
    assert len(results) == 3
    assert "repeated the same tool call" in results[-1]["output"]["hint"]


def test_a_flow_agent_compacts_old_tool_rounds(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "AGENT_CONTEXT_CHARS", 3_000)
    monkeypatch.setattr(runtime, "execute_ability",
                        lambda *args, **kwargs: {"path": "x.txt", "content": "x" * 2_000})
    gateway = ScriptedGateway([
        '{"tool": "read", "input": {"path": "x.txt"}}',
        '{"tool": "read", "input": {"path": "x.txt"}}',
        '{"tool": "read", "input": {"path": "x.txt"}}',
        '{"tool": "read", "input": {"path": "x.txt"}}',
        '{"final": "Done."}',
    ])
    manager = AgentManager(tmp_path, gateway)
    try:
        nodes, edges = chain({"id": "solve", "type": "agent", "data": {
            "abilities": ["read"], "max_iterations": 6,
        }}, output="{{ nodes.solve.text }}")
        manager.save_agent(minimal_agent("compacting", nodes=nodes, edges=edges, abilities=[
            {"id": "read", "type": "file.read"},
        ]))
        final = finished(manager, manager.create_run(
            "compacting", {"input": {"message": "go"}})["id"])
    finally:
        manager.close()

    assert final["status"] == "completed", final.get("error")
    messages = gateway.calls[3]["messages"]
    assert messages[0]["role"] == "system"
    assert messages[2]["content"].startswith(runtime.COMPACTED_AGENT_PREFIX)
    assert "read" in messages[2]["content"]
    observations = [item for item in messages
                    if "Tool result for read" in str(item.get("content"))]
    assert len(observations) == 2  # the oldest round was replaced by the ledger


def test_a_streaming_model_node_emits_live_deltas(tmp_path):
    class DeltaGateway(ScriptedGateway):
        def stream(self, config, cancelled=None):
            self.calls.append(copy.deepcopy(config))
            yield {"type": "start"}
            yield {"type": "reasoning", "text": "Planning."}
            yield {"type": "delta", "text": "Hello "}
            yield {"type": "delta", "text": "world"}
            yield {"type": "done", "reply": "Hello world",
                   "usage": {"prompt_tokens": 2, "completion_tokens": 2}}

    gateway = DeltaGateway()
    manager = AgentManager(tmp_path, gateway)
    try:
        nodes, edges = chain({"id": "answer", "type": "llm",
                              "data": {"prompt": "Say hello"}},
                             output="{{ nodes.answer }}")
        manager.save_agent(minimal_agent("streaming", nodes=nodes, edges=edges))
        final = finished(manager, manager.create_run("streaming", {"input": {}})["id"])
    finally:
        manager.close()

    assert final["status"] == "completed", final.get("error")
    deltas = [event for event in final["events"] if event["type"] == "assistant_delta"]
    assert "".join(event["text"] for event in deltas) == "Hello world"
    assert all(event["node"] == "answer" for event in deltas)
    reasoning = [event for event in final["events"] if event["type"] == "reasoning_delta"]
    assert "".join(event["text"] for event in reasoning) == "Planning."


def test_ai_agent_node_recovers_from_an_empty_reply(tmp_path):
    gateway = ScriptedGateway(["", '{"final": "Recovered."}'])
    manager = AgentManager(tmp_path, gateway)
    try:
        nodes, edges = chain({"id": "answer", "type": "agent",
                              "data": {"max_iterations": 4}},
                             output="{{ nodes.answer.text }}")
        manager.save_agent(minimal_agent("empty-reply", nodes=nodes, edges=edges))
        final = finished(manager, manager.create_run(
            "empty-reply", {"input": {"message": "go"}})["id"])
    finally:
        manager.close()

    assert final["status"] == "completed", final.get("error")
    assert final["output"] == "Recovered."
    assert any(event["type"] == "log" and "empty reply" in event.get("message", "")
               for event in final["events"])


def test_ai_agent_node_recovers_from_an_empty_final(tmp_path):
    gateway = ScriptedGateway(['{"final": ""}', '{"final": "Recovered."}'])
    manager = AgentManager(tmp_path, gateway)
    try:
        nodes, edges = chain({"id": "answer", "type": "agent",
                              "data": {"max_iterations": 4}},
                             output="{{ nodes.answer.text }}")
        manager.save_agent(minimal_agent("empty-final", nodes=nodes, edges=edges))
        final = finished(manager, manager.create_run(
            "empty-final", {"input": {"message": "go"}})["id"])
    finally:
        manager.close()

    assert final["status"] == "completed", final.get("error")
    assert final["output"] == "Recovered."
    assert any(event["type"] == "log" and "empty answer" in event.get("message", "")
               for event in final["events"])


def test_live_deltas_keep_every_character_of_a_large_burst(tmp_path):
    text = "x" * 5_000

    class BurstGateway(ScriptedGateway):
        def stream(self, config, cancelled=None):
            self.calls.append(copy.deepcopy(config))
            yield {"type": "delta", "text": text}
            yield {"type": "done", "reply": text,
                   "usage": {"prompt_tokens": 1, "completion_tokens": 1}}

    manager = AgentManager(tmp_path, BurstGateway())
    try:
        nodes, edges = chain({"id": "answer", "type": "llm",
                              "data": {"prompt": "Write"}},
                             output="{{ nodes.answer }}")
        manager.save_agent(minimal_agent("burst", nodes=nodes, edges=edges))
        final = finished(manager, manager.create_run("burst", {"input": {}})["id"])
    finally:
        manager.close()

    assert final["status"] == "completed", final.get("error")
    deltas = [event for event in final["events"] if event["type"] == "assistant_delta"]
    assert len(deltas) == 2
    assert "".join(event["text"] for event in deltas) == text


def test_ai_agent_node_repairs_a_malformed_action(tmp_path):
    gateway = ScriptedGateway([
        'I will call the clock: ```json\n{"tool": clock, "input": {}}\n```',
        '{"final": "Recovered."}',
    ])
    manager = AgentManager(tmp_path, gateway)
    try:
        nodes, edges = chain({"id": "solve", "type": "agent", "data": {
            "abilities": ["clock"], "max_iterations": 4,
        }}, output="{{ nodes.solve.text }}")
        manager.save_agent(minimal_agent("repairing", nodes=nodes, edges=edges, abilities=[
            {"id": "clock", "type": "time.now"},
        ]))
        final = finished(manager, manager.create_run(
            "repairing", {"input": {"message": "go"}})["id"])
    finally:
        manager.close()

    assert final["status"] == "completed", final.get("error")
    assert final["output"] == "Recovered."
    assert len(gateway.calls) == 2
    assert "did not contain a tool call" in gateway.calls[1]["messages"][-1]["content"]
    assert any(event["type"] == "log" and "empty" in event.get("message", "")
               for event in final["events"])


def test_ai_agent_node_plain_answers_exhaustion_and_test_mode(tmp_path):
    def run_with(replies, **payload):
        gateway = ScriptedGateway(replies)
        manager = AgentManager(tmp_path / str(len(replies)), gateway)
        try:
            nodes, edges = chain({"id": "solve", "type": "agent",
                                  "data": {"max_iterations": 2}}, output="{{ nodes.solve }}")
            manager.save_agent(minimal_agent("solver", nodes=nodes, edges=edges,
                                             abilities=[{"id": "clock", "type": "time.now"}]))
            final = finished(manager, manager.create_run("solver", {"input": {"message": "hi"}, **payload})["id"])
        finally:
            manager.close()
        return final, gateway

    plain, gateway = run_with(["Just a normal sentence, no JSON."])
    assert plain["output"]["text"] == "Just a normal sentence, no JSON." and plain["output"]["iterations"] == 1
    assert '{"message": "hi"}' in gateway.calls[0]["messages"][1]["content"].replace("\n", "").replace("  ", "")
    looping, _ = run_with(['{"tool": "clock"}', '{"tool": "clock"}', '{"tool": "clock"}'])
    assert looping["output"]["exhausted"] is True and looping["output"]["iterations"] == 2
    assert "Stopped after 2 tool rounds" in looping["output"]["text"]
    dry, gateway = run_with(["never used", "x"], dry_run=True)
    assert dry["output"]["dry_run"] is True and dry["output"]["text"].startswith("[test mode]")
    assert gateway.calls == []
    assert extract_json_object('noise {"a": {"b": "}"}} trailing {"c": 1}') == {"a": {"b": "}"}}
    assert extract_json_object("no object here") is None and extract_json_object("[1, 2]") is None
    assert extract_json_object('{bad json} then {"ok": true}') == {"ok": True}
    assert extract_json_value("[1, 2]") == [1, 2]
    assert extract_json_value(" \"hello\" ") == "hello"
    assert extract_json_value("42") == 42
    assert extract_json_value("not json") is None


def test_aml_validation_covers_initial_vars_and_reply_format():
    nodes, edges = chain({"id": "think", "type": "llm", "data": {"format": "yaml"}})
    document = minimal_agent(nodes=nodes, edges=edges)
    result = validate_agent(document)
    assert any(issue["field"] == "format" and "text or json" in issue["message"]
               for issue in result["issues"])
    document["vars"] = {str(index): index for index in range(101)}
    too_many = validate_agent(document)
    assert any(issue["field"] == "vars" and "100 keys" in issue["message"]
               for issue in too_many["issues"])


def test_llm_json_format_returns_the_parsed_value(tmp_path):
    gateway = ScriptedGateway(['```json\n{"answer": 42, "ok": true}\n```'])
    manager = AgentManager(tmp_path, gateway)
    try:
        nodes, edges = chain({
            "id": "think", "type": "llm",
            "data": {"prompt": "Return the answer", "format": "json"},
        })
        manager.save_agent(minimal_agent("json-llm", nodes=nodes, edges=edges))
        final = finished(manager, manager.create_run("json-llm", {"dry_run": False})["id"])
    finally:
        manager.close()
    assert final["status"] == "completed", final["error"]
    assert final["output"]["json"] == {"answer": 42, "ok": True}
    assert final["output"]["format"] == "json"
    assert "valid JSON value" in gateway.calls[0]["messages"][0]["content"]


def test_llm_json_format_fails_with_an_actionable_error(tmp_path):
    manager = AgentManager(tmp_path, ScriptedGateway(["not json at all"]))
    try:
        nodes, edges = chain({
            "id": "think", "type": "llm", "data": {"prompt": "go", "format": "json"},
        })
        manager.save_agent(minimal_agent("bad-json", nodes=nodes, edges=edges))
        final = finished(manager, manager.create_run("bad-json", {})["id"])
    finally:
        manager.close()
    assert final["status"] == "failed"
    assert "asked for JSON" in final["error"] and "format to text" in final["error"]


def test_assert_node_blocks_a_false_contract(tmp_path):
    manager = AgentManager(tmp_path, ScriptedGateway())
    try:
        nodes, edges = chain(
            {"id": "check", "type": "assert", "data": {
                "expression": "contains(input.message, 'ok')",
                "message": "Message must contain ok; expression was {{ expression }}",
            }},
            output="{{ input.message }}",
        )
        manager.save_agent(minimal_agent("guard", nodes=nodes, edges=edges))
        bad = finished(manager, manager.create_run(
            "guard", {"input": {"message": "nope"}})["id"])
        good = finished(manager, manager.create_run(
            "guard", {"input": {"message": "ok"}})["id"])
    finally:
        manager.close()

    assert bad["status"] == "failed"
    assert "Assertion failed at check" in bad["error"]
    assert "Message must contain ok" in bad["error"]
    assert good["status"] == "completed" and good["output"] == "ok"

    broken = minimal_agent("broken-assert", nodes=[
        {"id": "start", "type": "start"},
        {"id": "check", "type": "assert", "data": {"expression": "contains("}},
        {"id": "end", "type": "end"},
    ], edges=[{"from": "start", "to": "check"}, {"from": "check", "to": "end"}])
    result = validate_agent(broken)
    assert result["ok"] is False
    assert any(issue["field"] == "expression" for issue in result["issues"])


def test_validate_node_reports_schema_paths_and_warn_mode(tmp_path):
    schema = {
        "type": "object", "required": ["title", "items"],
        "properties": {
            "title": {"type": "string", "minLength": 3},
            "items": {"type": "array", "minItems": 1, "items": {"type": "string"}},
        },
        "additionalProperties": False,
    }

    def document(agent_id, payload, mode="fail"):
        nodes, edges = chain(
            {"id": "data", "type": "transform", "data": {
                "template": payload, "format": "json",
            }},
            {"id": "check", "type": "validate", "data": {
                "target": "{{ nodes.data.value }}", "schema": schema, "mode": mode,
            }},
            output="{{ nodes.check.valid }}",
        )
        return minimal_agent(agent_id, nodes=nodes, edges=edges)

    manager = AgentManager(tmp_path, ScriptedGateway())
    try:
        manager.save_agent(document("valid-json", '{"title":"Report","items":["a"]}'))
        good = finished(manager, manager.create_run("valid-json", {})["id"])

        manager.save_agent(document("bad-json", '{"title":"Report"}'))
        bad = finished(manager, manager.create_run("bad-json", {})["id"])

        manager.save_agent(document("warn-json", '{"title":"Report"}', mode="warn"))
        warned = finished(manager, manager.create_run("warn-json", {})["id"])
    finally:
        manager.close()

    assert good["status"] == "completed" and good["output"] is True
    assert bad["status"] == "failed"
    assert "missing required property 'items'" in bad["error"]
    assert warned["status"] == "completed" and warned["output"] is False
    validation = [event for event in warned["events"] if event["type"] == "validation"]
    assert validation and validation[0]["ok"] is False and validation[0]["mode"] == "warn"

    unsupported = document("unsupported-schema", '{"title":"Report"}')
    unsupported["nodes"][2]["data"]["schema"] = {"type": "object", "$ref": "#/x"}
    result = validate_agent(unsupported)
    assert result["ok"] is False
    assert "unsupported keyword" in " ".join(result["errors"])


def test_llm_and_agent_nodes_fall_back_when_the_primary_model_fails(tmp_path):
    gateway = FallbackGateway(reply="backup answer")
    manager = AgentManager(tmp_path, gateway)
    try:
        nodes, edges = chain({
            "id": "think", "type": "llm",
            "data": {"model": "primary", "fallback_model": "backup", "prompt": "go"},
        }, output="{{ nodes.think.text }}")
        manager.save_agent(minimal_agent("fallback-llm", nodes=nodes, edges=edges))
        llm = finished(manager, manager.create_run("fallback-llm", {})["id"])

        nodes, edges = chain({
            "id": "think", "type": "agent",
            "data": {"model": "primary", "fallback_model": "backup", "goal": "answer"},
        }, output="{{ nodes.think.text }}")
        manager.save_agent(minimal_agent("fallback-agent", nodes=nodes, edges=edges))
        agent = finished(manager, manager.create_run("fallback-agent", {})["id"])
    finally:
        manager.close()

    assert llm["status"] == "completed" and llm["output"] == "backup answer"
    assert agent["status"] == "completed" and agent["output"] == "backup answer"
    models = [call["model"] for call in gateway.calls]
    assert models[0] == "primary" and models[1] == "backup"
    logs = [event.get("message", "") for event in llm["events"] if event["type"] == "log"]
    assert any("falling back to backup" in message for message in logs)
    assert any(call["model"] == "backup" for call in gateway.calls)

    same = minimal_agent("same-fallback", nodes=[
        {"id": "start", "type": "start"},
        {"id": "think", "type": "llm", "data": {
            "model": "primary", "fallback_model": "primary", "prompt": "go"}},
        {"id": "end", "type": "end"},
    ], edges=[{"from": "start", "to": "think"}, {"from": "think", "to": "end"}])
    assert any("same as its main model" in warning
               for warning in validate_agent(same)["warnings"])


def test_initial_variables_are_available_before_the_first_node(manager):
    agent = minimal_agent("vars")
    agent["vars"] = {"greeting": "hi", "count": 2, "nested": {"enabled": True}}
    nodes, edges = chain(output="{{ vars.greeting }} {{ vars.count }} {{ vars.nested.enabled }}")
    agent["nodes"], agent["edges"] = nodes, edges
    manager.save_agent(agent)
    final = finished(manager, manager.create_run("vars", {"dry_run": True})["id"])
    assert final["status"] == "completed" and final["output"] == "hi 2 true"
    assert final["vars"] == agent["vars"]

    invalid = minimal_agent("bad-vars")
    invalid["vars"] = ["not", "an", "object"]
    result = validate_agent(invalid)
    assert result["ok"] is False
    assert any(issue["field"] == "vars" for issue in result["issues"])


def test_clone_creates_a_new_agent_and_disables_its_schedule(manager):
    source = minimal_agent("source", schedule={
        "enabled": True, "type": "interval", "interval_minutes": 15, "input": {"message": "go"},
    })
    manager.save_agent(source)
    clone = manager.clone_agent("source", "source-copy", "Copy of source")
    assert clone["agent"]["id"] == "source-copy"
    assert clone["agent"]["name"] == "Copy of source"
    assert clone["agent"]["schedule"]["enabled"] is False
    assert {item["id"] for item in manager.list_agents()} == {"source", "source-copy"}
    with pytest.raises(ValueError, match="already exists"):
        manager.clone_agent("source", "source-copy")


def test_run_stream_emits_snapshots_and_replays_history(manager):
    manager.save_agent(minimal_agent("streamed"))
    created = manager.create_run("streamed", {"input": {"message": "hi"}, "dry_run": True})
    frames = list(manager.stream_run(created["id"], 0))
    assert frames and frames[-1]["status"] == "completed"
    assert all(frame["type"] == "snapshot" for frame in frames)
    assert frames[-1]["output"] == "hi"
    assert all(frame["id"] == created["id"] for frame in frames)
    replay = list(manager.stream_run(created["id"], 0))
    assert len(replay) == 1 and replay[0]["status"] == "completed"


def test_finish_is_idempotent_so_a_late_reply_cannot_reopen_a_run():
    run = LiveRun("run-test", minimal_agent(), {})
    run.start()
    run.finish("cancelled", error="stop")
    run.finish("completed", output="late")
    assert run.status == "cancelled" and run.output is None


def test_a_waiter_never_sees_finished_without_the_terminal_event():
    run = LiveRun("run-terminal", minimal_agent(), {})
    run.start()
    seen: list[str] = []
    done = threading.Event()

    def watch():
        while True:
            pending, finished = run.wait_events(0, timeout=5)
            seen.extend(event["type"] for event in pending)
            if finished:
                done.set()
                return

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    run.finish("completed", output="done")

    assert done.wait(5)
    watcher.join(5)
    assert seen[-1] == "run_completed"
    # A late worker event must not land after the terminal event either.
    run.event("log", message="late worker")
    assert [event["type"] for event in run.snapshot()["events"]][-1] == "run_completed"


def test_cancel_stops_a_streaming_model_call(tmp_path):
    gateway = StreamingGateway()
    manager = AgentManager(tmp_path, gateway)
    try:
        nodes, edges = chain({"id": "think", "type": "llm", "data": {"prompt": "go"}})
        manager.save_agent(minimal_agent("cancellable", nodes=nodes, edges=edges))
        run = manager.create_run("cancellable", {})
        assert gateway.started.wait(5)
        manager.cancel_run(run["id"])
        final = finished(manager, run["id"])
    finally:
        manager.close()
    assert final["status"] == "cancelled" and gateway.stopped


def test_a_run_deadline_is_enforced_inside_an_llm_call(tmp_path):
    manager = AgentManager(tmp_path, SlowGateway())
    try:
        nodes, edges = chain({"id": "think", "type": "llm", "data": {"prompt": "go"}})
        agent = minimal_agent("slow", nodes=nodes, edges=edges)
        agent["settings"]["timeout"] = 1
        manager.save_agent(agent)
        final = finished(manager, manager.create_run("slow", {})["id"])
    finally:
        manager.close()
    assert final["status"] == "failed" and "exceeded timeout" in final["error"]


def test_manager_close_cancels_active_runs_and_rejects_new_ones(manager):
    nodes, edges = chain(
        {"id": "review", "type": "human", "data": {"kind": "approve", "timeout": 0}},
        output="{{ nodes.review }}",
    )
    manager.save_agent(minimal_agent("closing", nodes=nodes, edges=edges))
    run = manager.create_run("closing", {})
    waiting(manager, run["id"])
    manager.close()
    snapshot = manager.store.get_run(run["id"])
    assert snapshot["status"] == "cancelled" and snapshot["awaiting_input"] is None
    assert manager._runs == {}
    with pytest.raises(ValueError, match="shutting down"):
        manager.create_run("closing", {})


def test_a_history_write_failure_does_not_fail_a_run(manager, monkeypatch):
    manager.save_agent(list_templates()[0]["aml"])

    def explode(run):
        raise OSError("disk full")

    monkeypatch.setattr(manager, "_maybe_save", explode)
    final = finished(manager, manager.create_run("hello-agent", {
        "input": {"message": "hi"}, "dry_run": True,
    })["id"])
    assert final["status"] == "completed" and "[test mode]" in final["output"]


def test_interrupted_runs_are_failed_on_restart(tmp_path):
    store = AgentStore(tmp_path)
    store.save_run({
        "id": "run-stale", "agent_id": "probe", "status": "running",
        "events": [], "nodes": {"start": "ok", "think": "running"},
        "awaiting_input": {"node": "think"}, "waiting_node": "think", "paused": True,
    })
    assert store.settle_interrupted_runs() == 1
    snapshot = store.get_run("run-stale")
    assert snapshot["status"] == "failed" and "stopped" in snapshot["error"]
    assert snapshot["nodes"]["think"] == "error" and snapshot["awaiting_input"] is None
    assert snapshot["paused"] is False and snapshot["waiting_node"] is None
    assert snapshot["events"][-1]["type"] == "run_failed"
    assert store.settle_interrupted_runs() == 0


def test_corrupt_agent_metadata_does_not_break_the_agent_list(manager):
    path = manager.workspace / "agents" / "broken.json"
    path.write_text(json.dumps({
        "id": "broken", "name": "Broken", "nodes": "oops",
        "edges": 7, "abilities": {}, "schedule": "later",
    }))
    listed = next(agent for agent in manager.list_agents() if agent["id"] == "broken")
    assert listed["nodes"] == 0 and listed["edges"] == 0
    assert listed["abilities"] == 0 and listed["schedule"] == {}


def test_corrupt_run_history_is_skipped_or_repaired(tmp_path):
    store = AgentStore(tmp_path)
    runs = tmp_path / "agent-workspace" / "runs"
    (runs / "run-bad.json").write_text(json.dumps({
        "id": "run-bad", "agent_id": "probe", "status": "running",
        "created": "yesterday", "nodes": {"think": "running"},
        "events": [{"seq": "oops", "type": "log"}],
    }))
    (runs / "run-noid.json").write_text(json.dumps({"status": "completed"}))
    listed = store.list_runs()
    assert [run["id"] for run in listed] == ["run-bad"]
    assert store.settle_interrupted_runs() == 1
    repaired = store.get_run("run-bad")
    assert repaired["status"] == "failed" and repaired["events"][-1]["seq"] == 1


def test_schedule_updates_merge_fields(tmp_path):
    store = AgentStore(tmp_path)
    store.set_schedule_state("probe", {"last_run": "a"})
    store.set_schedule_state("probe", {"next_run": "b"})
    assert store.schedule_state("probe") == {"last_run": "a", "next_run": "b"}
    assert not list((tmp_path / "agent-workspace").glob(".schedules.json.*.tmp"))


def test_scheduler_records_why_a_due_run_could_not_start(manager, monkeypatch):
    agent = minimal_agent("blocked", {
        "enabled": True, "type": "interval", "interval_minutes": 60, "input": {},
    })
    manager.save_agent(agent)
    manager.store.set_schedule_state("blocked", {
        "next_run": (datetime.now() - timedelta(seconds=1)).isoformat(timespec="seconds"),
    })

    def explode(*args, **kwargs):
        raise ValueError("no model configured")

    monkeypatch.setattr(manager, "create_run", explode)
    manager._scheduler_tick()
    state = manager.store.schedule_state("blocked")
    assert "no model configured" in state["last_error"]
    assert datetime.fromisoformat(state["next_run"]) > datetime.now()
    assert manager.schedule_info("blocked")["last_error"] == "no model configured"
    listed = next(agent for agent in manager.list_agents() if agent["id"] == "blocked")
    assert listed["last_error"] == "no model configured"


def test_run_payload_is_validated_before_a_thread_starts(manager):
    manager.save_agent(minimal_agent("payload"))
    with pytest.raises(ValueError, match="breakpoints must be a list"):
        manager.create_run("payload", {"breakpoints": "start"})
    with pytest.raises(ValueError, match="Unknown breakpoint"):
        manager.create_run("payload", {"breakpoints": ["ghost"]})
    with pytest.raises(ValueError, match="dry_run must be"):
        manager.create_run("payload", {"dry_run": "yes"})
    with pytest.raises(ValueError, match="trigger must be"):
        manager.create_run("payload", {"trigger": "later"})
    assert manager._runs == {}


def test_human_node_waits_for_an_answer(manager):
    nodes, edges = chain(
        {"id": "review", "type": "human", "data": {"prompt": "Ship {{ input.message }}?", "kind": "approve"}},
        output="{{ nodes.review }}",
    )
    manager.save_agent(minimal_agent("gate", nodes=nodes, edges=edges))
    run = manager.create_run("gate", {"input": {"message": "v2"}})
    asked = waiting(manager, run["id"])
    assert asked["status"] == "paused" and asked["paused"] is False  # not a debugger pause
    assert asked["awaiting_input"] == {"node": "review", "prompt": "Ship v2?", "kind": "approve",
                                       "choices": [], "default": ""}
    assert asked["nodes"]["review"] == "paused"
    manager.resume_run(run["id"])  # the debugger cannot answer for a person
    assert manager.get_run(run["id"])["awaiting_input"] is not None
    with pytest.raises(ValueError, match="true .approve. or false"):
        manager.provide_input(run["id"], "maybe")
    manager.provide_input(run["id"], "yes")
    final = finished(manager, run["id"])
    assert final["status"] == "completed" and final["awaiting_input"] is None
    assert final["output"] == {"value": True, "text": "approved", "timed_out": False, "approved": True}
    assert [event["type"] for event in final["events"] if event["type"].startswith("input_")] == [
        "input_requested", "input_received"]
    with pytest.raises(ValueError, match="not active"):
        manager.provide_input(run["id"], True)

    cancelled = manager.create_run("gate", {"input": {"message": "v3"}})
    waiting(manager, cancelled["id"])
    manager.cancel_run(cancelled["id"])
    assert finished(manager, cancelled["id"])["status"] == "cancelled"


def test_human_node_choices_defaults_and_timeouts(manager):
    def gate(agent_id, **data):
        nodes, edges = chain({"id": "pick", "type": "human", "data": data}, output="{{ nodes.pick }}")
        manager.save_agent(minimal_agent(agent_id, nodes=nodes, edges=edges))
        return agent_id

    choice = manager.create_run(gate("choice", kind="choice", choices=["red", "blue"], prompt="Colour?"), {})
    waiting(manager, choice["id"])
    with pytest.raises(ValueError, match="Choose one of: red, blue"):
        manager.provide_input(choice["id"], "green")
    manager.provide_input(choice["id"], "blue")
    assert finished(manager, choice["id"])["output"] == {"value": "blue", "text": "blue", "timed_out": False}

    defaulted = manager.create_run(gate("defaulted", kind="text", default="fallback", timeout=0.2), {})
    final = finished(manager, defaulted["id"])
    assert final["output"] == {"value": "fallback", "text": "fallback", "timed_out": True}
    assert any(event["type"] == "input_timeout" for event in final["events"])

    strict = finished(manager, manager.create_run(gate("strict", kind="approve", timeout=0.2), {})["id"])
    assert strict["status"] == "failed" and "No answer arrived" in strict["error"]

    dry = finished(manager, manager.create_run(gate("dry", kind="approve"), {"dry_run": True})["id"])
    assert dry["output"] == {"value": True, "text": "approved", "timed_out": False,
                             "approved": True, "dry_run": True}
    dry_default = finished(manager, manager.create_run(
        gate("dry-default", kind="approve", default="false"), {"dry_run": True})["id"])
    assert dry_default["output"]["approved"] is False


def test_foreach_runs_a_sub_agent_per_item(manager):
    manager.save_agent(minimal_agent("tenfold", nodes=[
        {"id": "start", "type": "start"},
        {"id": "end", "type": "end", "data": {"output": "{{ input.value * 10 }}"}},
    ]))
    nodes, edges = chain({"id": "each", "type": "foreach", "data": {
        "items": "{{ input.numbers }}", "agent": "tenfold", "input": "{{ item + index }}", "max_items": 3,
    }}, output="{{ nodes.each }}")
    manager.save_agent(minimal_agent("fan-out", nodes=nodes, edges=edges))
    final = finished(manager, manager.create_run("fan-out", {"input": {"numbers": [1, 2, 3, 4]}})["id"])
    assert final["status"] == "completed", final["error"]
    assert final["output"] == {"items": [10, 30, 50], "count": 3, "truncated": True}
    children = [run for run in manager.list_runs("tenfold") if run["parent_run_id"] == final["id"]]
    assert len(children) == 3 and all(run["trigger"] == "subagent" for run in children)
    wrong = finished(manager, manager.create_run("fan-out", {"input": {"numbers": "not a list"}})["id"])
    assert wrong["status"] == "failed" and "needs a list" in wrong["error"]


def test_scheduler_tick_runs_due_once_agent(manager):
    agent = minimal_agent("scheduled", {
        "enabled": True, "type": "once",
        "once_at": (datetime.now() - timedelta(minutes=1)).isoformat(timespec="seconds"),
        "input": {"message": "scheduled"},
    })
    agent["nodes"][1]["data"]["output"] = "scheduled-ok"
    manager.save_agent(agent)
    manager.store.set_schedule_state("scheduled", {
        "last_run": None,
        "next_run": (datetime.now() - timedelta(seconds=1)).isoformat(timespec="seconds"),
    })
    manager._scheduler_tick()
    run = wait_for(lambda: (manager.list_runs("scheduled") or [None])[0])
    final = wait_for(lambda: (lambda snap: snap if snap["status"] == "completed" else None)(
        manager.get_run(run["id"])))
    assert final["trigger"] == "schedule" and final["output"] == "scheduled-ok"
    assert manager.store.schedule_state("scheduled")["next_run"] is None


def test_swarm_node_validates_and_runs_in_test_mode(manager):
    document = {
        "aml": "1.4", "kind": "agent", "id": "swarm-probe", "name": "Swarm probe",
        "model": "fake",
        "swarm": {"allow_write": False, "max_rounds": 1, "max_agents": 3},
        "abilities": [{"id": "read", "type": "file.read"}],
        "nodes": [
            {"id": "start", "type": "start"},
            {"id": "team", "type": "swarm", "data": {
                "goal": "{{ input.message }}", "strategy": "peer_review", "rounds": 1,
                "criteria": "read the file\ncite the path", "effort": "high",
                "reflection": False,
                "agents": [{"id": "reader", "task": "read the sandbox",
                            "depends_on": [], "plan": ["open the file"]},
                           {"id": "checker", "task": "check the reader",
                            "depends_on": ["reader"]}],
            }},
            {"id": "end", "type": "end", "data": {"output": "{{ nodes.team.text }}"}},
        ],
        "edges": [{"from": "start", "to": "team"}, {"from": "team", "to": "end"}],
    }
    assert validate_agent(document)["ok"] is True
    document["nodes"][1]["data"]["write"] = True
    result = validate_agent(document)
    assert result["ok"] is False
    assert "swarm.allow_write" in " ".join(result["errors"])
    document["nodes"][1]["data"]["write"] = False

    saved = manager.save_agent(document)["agent"]
    assert saved["swarm"]["allow_write"] is False and saved["swarm"]["max_agents"] == 3
    run = manager.create_run("swarm-probe", {"input": {"message": "hello"}, "dry_run": True})
    final = finished(manager, run["id"])
    assert final["status"] == "completed", final.get("error")
    assert final["output"] == "[test mode] hello"
    swarm = manager.get_swarm(run["id"])
    assert swarm["status"] == "completed" and swarm["strategy"] == "peer_review"
    assert swarm["success_criteria"] == ["read the file", "cite the path"]
    assert final["output"] and final["nodes"]["team"] == "ok"
    with pytest.raises(ValueError, match="not active"):
        manager.inject_swarm_message(run["id"], {"text": "stop"})


def test_protocol_metadata_is_complete_for_the_designer(manager):
    protocol = manager.protocol_metadata()
    assert protocol["protocol"] == "AMP" and protocol["version"] == "1.10"
    assert protocol["aml_version"] == "1.8"
    assert protocol["tool_calling"]["formats"] == ["json", "dsml", "native"]
    assert "openai-compatible" in protocol["tool_calling"]["native_backends"]
    assert "validation" in protocol["event_types"]
    assert "model_fallback" in protocol["guardrails"]
    assert set(protocol["node_types"]) == set(aml.NODE_TYPES) >= {"agent", "human", "foreach", "swarm"}
    allowed = set(protocol["field_types"])
    for name, node in protocol["node_types"].items():
        assert {"label", "description", "category", "icon", "fields", "defaults"} <= set(node), name
        keys = {field["key"] for field in node["fields"]}
        assert set(node["defaults"]) == keys, name  # every default is editable, every field has one
        for field in node["fields"]:
            assert field["type"] in allowed and field["label"], (name, field)
            if field["type"] == "select":
                assert field["options"] and node["defaults"][field["key"]] in dict(field["options"])
    assert protocol["node_types"]["subagent"]["label"] == "Sub-agent"
    assert [field["key"] for field in protocol["common_fields"]] == ["retry", "retry_delay"]
    for name, ability in protocol["abilities"].items():
        assert isinstance(ability["params"], list) and ability["label"], name
        assert all(field["type"] in allowed for field in ability["params"])
    assert {"tool_call", "tool_result", "input_requested", "node_input"} <= set(protocol["event_types"])
    paths = {operation["path"] for operation in protocol["operations"]}
    assert {"/api/agents/runs/{run_id}/input", "/api/agents/runs/{run_id}/breakpoints",
            "/api/agents/runs/{run_id}/events", "/api/agents/{id}/clone", "/api/agents/runs",
            "/api/agents/generate"} <= paths
    assert {"sqlite.query", "csv.read", "git.inspect"} <= set(protocol["abilities"])
    assert protocol["node_types"]["llm"]["defaults"]["format"] == "text"
    swarm = protocol["swarm"]
    assert swarm["node_type"] == "swarm"
    assert "peer_review" in swarm["strategies"] and "message" in swarm["message_kinds"]
    assert {"plan", "say", "memory", "review", "final"} <= set(swarm["member_protocol"])
    assert {"swarm_start", "swarm_review", "swarm_message"} <= set(protocol["event_types"])
    assert "swarm_agent_plan" in protocol["event_types"]
    assert "retries" in swarm["reliability"] and "plan_check" in swarm["reliability"]
    assert "depends_on" in swarm["leader_protocol"]["plan"]
    assert "success_criteria" in swarm["leader_protocol"]["plan"]
    defaults = protocol["node_types"]["swarm"]["defaults"]
    assert defaults["criteria"] == "" and defaults["reflection"] is True
    assert defaults["effort"] == ""
    assert {"/api/agents/runs/{run_id}/swarm",
            "/api/agents/runs/{run_id}/swarm/messages"} <= paths


def test_agents_can_be_drafted_from_a_description():
    valid = {
        "name": "Summariser", "abilities": [{"id": "fetch", "type": "http.fetch"}],
        "nodes": [{"id": "start", "type": "start"},
                  {"id": "get", "type": "tool", "data": {"ability": "fetch", "input": {"url": "{{ input.url }}"}}},
                  {"id": "sum", "type": "llm", "data": {"prompt": "Summarise {{ nodes.get.body }}"}},
                  {"id": "end", "type": "end"}],
        "edges": [{"from": "start", "to": "get"}, {"from": "get", "to": "sum"}, {"from": "sum", "to": "end"}],
        "schedule": {"enabled": True, "type": "interval", "interval_minutes": 1},
    }
    broken = {**valid, "nodes": valid["nodes"][:-1]}  # no end node
    gateway = ScriptedGateway(["Sure! " + json.dumps(broken), "```json\n" + json.dumps(valid) + "\n```"])
    result = generate_agent(gateway, "Fetch a URL and summarise the page.", None, set())
    assert result["attempts"] == 2 and result["model"] == "fake"
    agent = result["agent"]
    assert agent["name"] == "Summariser" and agent["schedule"]["enabled"] is False
    assert [node["x"] for node in agent["nodes"]] == [60.0, 310.0, 560.0, 810.0]  # laid out left to right
    repair = gateway.calls[1]["messages"][-1]["content"]
    assert "not valid AML" in repair and "at least one end node" in repair
    prompt = gateway.calls[0]["messages"][0]["content"]
    assert "- agent:" in prompt and "- http.fetch:" in prompt and "shell.exec" not in prompt

    with pytest.raises(ValueError, match="could not produce a valid agent"):
        generate_agent(ScriptedGateway(["nope", "still nope", "nope again"]), "Do something useful.", "fake", set())
    with pytest.raises(ValueError, match="Describe what the agent should do"):
        generate_agent(ScriptedGateway(), "hi", None, set())
    with pytest.raises(ValueError, match="Choose which model"):
        generate_agent(ScriptedGateway(models=("a", "b")), "Fetch a URL and summarise it.", None, set())
    with pytest.raises(ValueError, match="No chat model is available"):
        generate_agent(ScriptedGateway(models=()), "Fetch a URL and summarise it.", None, set())
