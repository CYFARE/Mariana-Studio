"""Chat swarm mode: validation, tools, streaming, failure and cancellation."""

from __future__ import annotations

import copy
import json
import threading

import pytest

from mariana.studio.chat_swarm import (
    READ_TOOLS,
    WEB_TOOLS,
    WRITE_TOOLS,
    ChatSwarm,
    _digest,
    _done,
    _event_value,
    _executor,
    _reply_text,
    _summary,
    _tool_definitions,
)
from mariana.studio.errors import BadRequestError, ConflictError, NotFoundError
from mariana.workspace import chat_files_path
from tests.test_agents_swarm import SwarmGateway


class ChatGateway(SwarmGateway):
    """A scripted model gateway that answers Chat swarm prompts directly."""

    def __init__(self):
        super().__init__()
        self.member_calls: list[dict] = []

    def prepare(self, config, cancelled=None):
        model = str(config.get("model") or "")
        if model not in {"fake", "leader"}:
            raise BadRequestError("Model not found; choose an available chat model")
        return {"id": model, "name": model.title(), "available": True,
                "capabilities": {"chat": True}}, None

    def chat(self, config):
        system = config["messages"][0]["content"]
        if "You are one member of the swarm" in system:
            self.member_calls.append(copy.deepcopy(config))
            return {"reply": json.dumps({"final": "member finding"}),
                    "usage": {"prompt_tokens": 10, "completion_tokens": 2}}
        return super().chat(config)


def chat_config(**overrides) -> dict:
    config = {
        "model": "fake",
        "messages": [
            {"role": "user", "content": "My name is Ada."},
            {"role": "assistant", "content": "Hello Ada."},
            {"role": "user", "content": "What is my name?"},
        ],
        "system_prompt": "Be concise.",
    }
    return {**config, **overrides}


def assembled_result() -> dict:
    """A timed-out swarm whose leader never synthesised; members wrote raw hand-offs."""
    return {
        "ok": True,
        "assembled": True,
        "timed_out": True,
        "final": "> The time budget ended before the leader could synthesise this run.\n\n"
                 "## Feed Collector\n\n" + "raw hand-off " * 500,
        "agents": [
            {"id": "feeds", "label": "Feed Collector", "status": "done",
             "output": ("RETRIEVAL RESULT — India headlines, 22 Nov 2026 (IST).\n\n"
                        "TOP CANDIDATES (ranked by significance):\n\n"
                        "    Hyderabad on high alert after an email warns of attacks\n"
                        "        Hindustan Times — https://example.test/a\n")},
            {"id": "web", "label": "Web Collector", "status": "done",
             "output": ("PATHS USED\n\nTOP 5 (ranked by significance)\n\n"
                        "    1. Rahul Gandhi calls RSS-BJP 'greedy, casteist male "
                        "chauvinists'; BJP says 'Don't need lectures' — Times Now, "
                        "22 Nov 2026 22:50 IST.\n")},
            {"id": "verify", "label": "Verifier & Ranker", "status": "failed",
             "output": "", "error": "Work budget exhausted before completion"},
        ],
    }


def test_an_assembled_swarm_reply_is_a_readable_report(tmp_path):
    result = assembled_result()
    summary = _summary(result)
    request = ChatSwarm(tmp_path, ChatGateway()).prepare(chat_config())
    payload = _done(request, result, summary)
    reply = payload["reply"]

    assert summary["assembled"] is True
    assert reply.startswith("**Time limit reached.** 2 of 3 agents delivered findings "
                            "before the deadline; the leader ran out of time")
    assert "**Delivered**" in reply
    assert "**Feed Collector** — Hyderabad on high alert after an email warns of attacks" in reply
    assert "**Web Collector** — Rahul Gandhi calls RSS-BJP" in reply
    assert "Times Now" not in reply
    assert "**Not delivered**" in reply
    assert "**Verifier & Ranker** — Work budget exhausted before completion" in reply
    assert "agent card below" in reply and "Research depth" in reply
    assert len(reply) < 1_500
    assert "notice" not in payload


def test_a_synthesised_reply_is_used_unchanged():
    assert _reply_text({"final": "merged answer", "assembled": False}) == "merged answer"
    assert _reply_text({"final": "", "assembled": False}) == "The swarm produced no output."


def test_the_handoff_digest_prefers_a_headline():
    assert _digest("") == ""
    assert _digest("A short finding.\n\nMore detail.") == "A short finding."
    assert _digest("## Findings\n\n- The first result\n- A second result") == "The first result"
    assert _digest("TOP 3\n\n1. A headline long enough to be useful — Source, 22 Nov") == (
        "A headline long enough to be useful")


def test_conversation_workspace_files_can_be_previewed(tmp_path):
    service = ChatSwarm(tmp_path, ChatGateway())
    workspace = chat_files_path(tmp_path, "chat-abcdef", create=True)
    (workspace / "index.html").write_text(
        "<!DOCTYPE html><title>Home</title><style>body{color:#111}</style>", encoding="utf-8")
    (workspace / "photo.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 16)
    (workspace / "data.bin").write_bytes(b"\x00\x01\x02")
    (workspace / "long.txt").write_text("x" * 250_000, encoding="utf-8")

    page = service.read_file("chat-abcdef", "index.html")
    assert page["name"] == "index.html" and page["kind"] == "text"
    assert page["language"] == "html" and page["mime"] == "text/html"
    assert page["content"].startswith("<!DOCTYPE html>") and page["truncated"] is False
    assert isinstance(page["version"], int) and page["version"] > 0

    image = service.read_file("chat-abcdef", "photo.png")
    assert image["kind"] == "image"
    assert image["data_url"].startswith("data:image/png;base64,")

    long_file = service.read_file("chat-abcdef", "long.txt")
    assert len(long_file["content"]) == 200_000 and long_file["truncated"] is True

    with pytest.raises(BadRequestError, match="binary"):
        service.read_file("chat-abcdef", "data.bin")
    with pytest.raises(BadRequestError, match="outside the conversation workspace"):
        service.read_file("chat-abcdef", "../secret.txt")
    with pytest.raises(BadRequestError, match="Invalid conversation id"):
        service.read_file("../etc", "passwd")
    with pytest.raises(NotFoundError, match="no workspace files"):
        service.read_file("chat-zzzzzz", "index.html")
    with pytest.raises(NotFoundError, match="File not found"):
        service.read_file("chat-abcdef", "missing.html")


def test_a_synthesised_run_keeps_one_short_note(tmp_path):
    request = ChatSwarm(tmp_path, ChatGateway()).prepare(chat_config())
    summary = {"warnings": ["first warning", "second warning"],
               "changed_files": ["a.py"]}
    payload = _done(request, {"final": "answer", "assembled": False}, summary)
    assert payload["notice"] == "first warning (+1 more in the agent card) · Files changed: a.py"

    assembled = _done(request, {"final": "raw", "assembled": True, "timed_out": True},
                      {"warnings": ["late"], "changed_files": []})
    assert "notice" not in assembled


def test_sse_payloads_are_finite_and_bounded():
    nested = 1
    for _ in range(20):
        nested = [nested]
    event = _event_value({
        "type": "done", "nan": float("nan"), "inf": [float("inf"), 1.5],
        "text": "x" * 100_000, "deep": nested, "items": list(range(300)),
    })
    assert event["nan"] is None and event["inf"] == [None, 1.5]
    assert len(event["text"]) == 20_000
    assert event["items"] == list(range(200))
    encoded = json.dumps(event, allow_nan=False)
    assert "NaN" not in encoded and "Infinity" not in encoded
    assert json.loads(encoded) == event


def test_prepare_refuses_when_too_many_swarms_are_running(tmp_path):
    service = ChatSwarm(tmp_path, ChatGateway())
    service._jobs.update(object() for _ in range(3))
    with pytest.raises(ConflictError, match="At most 3"):
        service.prepare(chat_config())


def test_prepare_validates_and_bounds_a_chat_swarm(tmp_path):
    service = ChatSwarm(tmp_path, ChatGateway())
    request = service.prepare(chat_config())
    assert request.model == "fake" and request.model_name == "Fake"
    assert request.strategy == "auto" and request.rounds == 1
    assert request.limits.max_agents == 8 and request.limits.max_steps == 10
    assert request.limits.max_seconds == 300.0
    assert request.goal == "What is my name?"
    assert "Conversation system prompt:\nBe concise." in request.briefing
    assert "User: My name is Ada." in request.briefing
    assert "What is my name?" not in request.briefing

    for patch, message in (
        ({"model": ""}, "Choose a model"),
        ({"messages": []}, "Enter a message first"),
        ({"messages": [{"role": "assistant", "content": "hi"}]},
         "last message must be from you"),
        ({"messages": "nope"}, "messages must be a list"),
        ({"strategy": "magic"}, "strategy must be one of"),
        ({"max_agents": 999}, "max_agents must be between 1 and 64"),
        ({"max_steps": 0}, "max_steps must be between 1 and 24"),
        ({"rounds": 3}, "rounds must be between 0 and 2"),
        ({"max_seconds": 5}, "max_seconds must be between"),
        ({"leader_model": "missing"}, "Model not found"),
    ):
        with pytest.raises(BadRequestError, match=message):
            service.prepare({**chat_config(), **patch})
    with pytest.raises(BadRequestError, match="JSON object"):
        service.prepare(["not", "a", "dict"])

    leader = service.prepare(chat_config(leader_model="leader"))
    assert leader.leader_model == "leader"

    extended = service.prepare(chat_config(max_seconds=1800, max_steps=24))
    assert extended.limits.max_seconds == 1800.0 and extended.limits.max_steps == 24
    for patch, message in (
        ({"max_seconds": 1801}, "max_seconds must be between"),
        ({"max_steps": 25}, "max_steps must be between"),
    ):
        with pytest.raises(BadRequestError, match=message):
            service.prepare({**chat_config(), **patch})


def test_stream_runs_a_tool_less_swarm_and_ends_with_the_synthesis(tmp_path):
    gateway = ChatGateway()
    service = ChatSwarm(tmp_path, gateway)
    events = [event for event in service.stream(chat_config()) if event is not None]
    kinds = [event["type"] for event in events]
    assert kinds[0] == "swarm_start"
    assert "swarm_plan" in kinds
    assert kinds.count("swarm_agent_end") >= 2
    assert kinds[-1] == "done"

    done = events[-1]
    assert done["reply"] == "merged answer"
    assert done["kind"] == "Swarm"
    assert done["swarm"]["status"] == "completed"
    assert len(done["swarm"]["agents"]) == 2
    assert done["swarm"]["reviews"]
    assert done["usage"]["completion_tokens"] > 0
    assert done["tokens_per_second"] > 0
    assert any("snapshot" in event for event in events)

    member = gateway.member_calls[0]
    system = member["messages"][0]["content"]
    assert "Available tools:\n- none" in system
    assert "User: My name is Ada." in system
    assert "no project workspace and no tools" in system
    assert "What is my name?" in member["messages"][-1]["content"]


def test_stream_reports_a_failed_swarm_in_band(tmp_path):
    class FailingGateway(ChatGateway):
        def chat(self, config):
            if "You are one member of the swarm" in config["messages"][0]["content"]:
                raise RuntimeError("model down")
            return super().chat(config)

    service = ChatSwarm(tmp_path, FailingGateway())
    events = [event for event in service.stream(chat_config()) if event is not None]
    final = events[-1]
    assert final["type"] == "error"
    assert "Every swarm member failed" in final["error"]
    assert final["swarm"]["status"] == "failed"
    assert all(agent["ok"] is False for agent in final["swarm"]["agents"])


def test_cancellation_ends_the_stream_with_an_error_event(tmp_path):
    class BlockingGateway(ChatGateway):
        def __init__(self):
            super().__init__()
            self.started = threading.Event()
            self.release = threading.Event()

        def chat(self, config):
            if "Plan a focused team" in config["messages"][0]["content"]:
                self.started.set()
                self.release.wait(5)
            return super().chat(config)

    gateway = BlockingGateway()
    service = ChatSwarm(tmp_path, gateway)
    cancelled = threading.Event()
    iterator = service.stream(chat_config(), cancelled)
    assert next(iterator)["type"] == "swarm_start"
    assert gateway.started.wait(2)
    cancelled.set()
    gateway.release.set()
    rest = [event for event in iterator if event is not None]
    assert rest[-1]["type"] == "error"
    assert "cancelled" in rest[-1]["error"].lower()


def test_close_cancels_running_swarms(tmp_path):
    class BlockingGateway(ChatGateway):
        def __init__(self):
            super().__init__()
            self.started = threading.Event()
            self.release = threading.Event()

        def chat(self, config):
            if "Plan a focused team" in config["messages"][0]["content"]:
                self.started.set()
                self.release.wait(5)
            return {"reply": json.dumps({"final": "too late"}), "usage": {}}

    gateway = BlockingGateway()
    service = ChatSwarm(tmp_path, gateway)
    iterator = service.stream(chat_config())
    assert next(iterator)["type"] == "swarm_start"
    assert gateway.started.wait(2)
    service.close()
    gateway.release.set()
    rest = [event for event in iterator if event is not None]
    assert rest and rest[-1]["type"] == "error"


def test_prepare_enables_read_tools_with_a_conversation_workspace(tmp_path):
    service = ChatSwarm(tmp_path, ChatGateway())
    request = service.prepare(chat_config(chat_id="chat-tool01", tools={"files": True}))
    assert [tool["id"] for tool in request.tools] == [*READ_TOOLS, *WEB_TOOLS]
    assert request.workspace == tmp_path / "agent-workspace" / "files" / "chats" / "chat-tool01"
    assert request.workspace.is_dir()
    assert request.allow_write is False and request.web is True
    assert "no tools" not in request.system
    assert "file.read" in request.system and "Public web access is enabled" in request.system
    assert "agent-workspace/files/chats/chat-tool01" in request.briefing
    assert "no project workspace" not in request.briefing


def test_write_and_web_flags_extend_the_tool_set(tmp_path):
    service = ChatSwarm(tmp_path, ChatGateway())
    request = service.prepare(chat_config(
        chat_id="chat-tool02", tools={"files": False, "write": True, "web": True},
    ))
    ids = [tool["id"] for tool in request.tools]
    assert all(tool_id in ids for tool_id in READ_TOOLS)  # writes imply reading
    assert all(tool_id in ids for tool_id in WRITE_TOOLS)
    assert all(tool_id in ids for tool_id in WEB_TOOLS)
    assert request.allow_write is True and request.web is True
    web = next(tool for tool in request.tools if tool["id"] == "http.fetch")
    assert web["config"]["allow_private"] is False
    assert any(tool["id"] == "news.headlines" for tool in request.tools)
    assert "Public web access is enabled" in request.system


def test_tools_need_a_valid_saved_conversation(tmp_path):
    service = ChatSwarm(tmp_path, ChatGateway())
    with pytest.raises(BadRequestError, match="Save the conversation"):
        service.prepare(chat_config(tools={"files": True}))
    with pytest.raises(BadRequestError, match="conversation id is not valid"):
        service.prepare(chat_config(chat_id="../../outside", tools={"files": True}))
    with pytest.raises(BadRequestError, match="tools must be an object"):
        service.prepare(chat_config(chat_id="chat-tool03", tools=["file.read"]))
    # No tools key means the older tool-less behaviour, even without an id.
    legacy = service.prepare(chat_config())
    assert legacy.tools == [] and legacy.workspace is None


def test_the_executor_reads_the_workspace_and_blocks_escapes(tmp_path):
    workspace = tmp_path / "agent-workspace" / "files" / "chats" / "chat-tool04"
    workspace.mkdir(parents=True)
    (workspace / "notes.txt").write_text("hello from the workspace", encoding="utf-8")
    execute = _executor(tmp_path, workspace, _tool_definitions(True, False, False),
                        threading.Event())

    observation, ok = execute({"id": "a1"}, "file.read", {"path": "notes.txt"})
    assert ok is True and observation["content"] == "hello from the workspace"

    observation, ok = execute({"id": "a1"}, "file.read", {"path": "../outside.txt"})
    assert ok is False and "inside the file sandbox" in observation["error"]

    observation, ok = execute({"id": "a1"}, "shell.exec", {"command": "id"})
    assert ok is False and "not available" in observation["error"]


def test_web_fetch_stays_public_only(tmp_path):
    workspace = tmp_path / "agent-workspace" / "files" / "chats" / "chat-tool05"
    workspace.mkdir(parents=True)
    execute = _executor(tmp_path, workspace, _tool_definitions(False, False, True),
                        threading.Event())
    observation, ok = execute({"id": "a1"}, "http.fetch", {"url": "http://127.0.0.1:1/"})
    assert ok is False and "Private addresses are blocked" in observation["error"]


def test_web_tools_include_news_and_os_access_adds_shell(tmp_path):
    tools = {tool["id"]: tool for tool in _tool_definitions(True, False, True)}
    assert {"http.fetch", "news.headlines"} <= set(tools)
    assert "shell.exec" not in tools

    os_tools = {tool["id"]: tool for tool in _tool_definitions(True, False, True, True)}
    assert {"shell.exec", "browser.fetch"} <= set(os_tools)
    assert "file.write" in os_tools  # OS access includes workspace writes
    assert os_tools["http.fetch"]["config"]["allow_private"] is True
    assert os_tools["news.headlines"]["config"]["region"] == "IN"


def test_os_access_is_a_server_setting(tmp_path):
    service = ChatSwarm(tmp_path, ChatGateway(),
                        settings=lambda: {"allow_os_access": True})
    request = service.prepare(chat_config(chat_id="chat-tool08", tools={"files": True}))
    ids = {tool["id"] for tool in request.tools}
    assert {"shell.exec", "browser.fetch", "file.write", "http.fetch", "news.headlines"} <= ids
    assert request.os_access is True and request.allow_write is True
    assert "OS access is enabled" in request.system


def test_writes_require_the_write_flag(tmp_path):
    workspace = tmp_path / "agent-workspace" / "files" / "chats" / "chat-tool07"
    workspace.mkdir(parents=True)
    read_only = _executor(tmp_path, workspace, _tool_definitions(True, False, False),
                          threading.Event())
    observation, ok = read_only({"id": "a1"}, "file.write",
                                {"path": "out.txt", "content": "hi"})
    assert ok is False and "not available" in observation["error"]
    assert not (workspace / "out.txt").exists()

    writable = _executor(tmp_path, workspace, _tool_definitions(True, True, False),
                         threading.Event())
    observation, ok = writable({"id": "a1"}, "file.write",
                               {"path": "out.txt", "content": "hi"})
    assert ok is True and (workspace / "out.txt").read_text(encoding="utf-8") == "hi"


def test_a_swarm_member_can_read_a_workspace_file(tmp_path):
    class ToolGateway(ChatGateway):
        def __init__(self):
            super().__init__()
            self.member_steps = 0

        def chat(self, config):
            system = config["messages"][0]["content"]
            if "Plan a focused team" in system:
                return {"reply": json.dumps({
                    "strategy": "map_reduce", "rounds": 0,
                    "agents": [{"id": "a1", "task": "read the notes"}],
                }), "usage": {}}
            if "You are one member of the swarm" in system:
                self.member_calls.append(copy.deepcopy(config))
                self.member_steps += 1
                if self.member_steps == 1:
                    return {"reply": json.dumps({
                        "tool": "file.read", "input": {"path": "notes.txt"},
                    }), "usage": {}}
                return {"reply": json.dumps({"final": "the notes say hello"}), "usage": {}}
            return super().chat(config)

    gateway = ToolGateway()
    service = ChatSwarm(tmp_path, gateway)
    workspace = tmp_path / "agent-workspace" / "files" / "chats" / "chat-tool06"
    workspace.mkdir(parents=True)
    (workspace / "notes.txt").write_text("hello", encoding="utf-8")
    events = [event for event in service.stream(chat_config(
        chat_id="chat-tool06", tools={"files": True},
    )) if event is not None]

    tool_events = [event for event in events if event["type"] == "swarm_agent_tool"]
    assert any(event.get("ok") is True and event.get("tool") == "file.read"
               for event in tool_events)
    assert any("hello" in json.dumps(event.get("output"), ensure_ascii=False)
               for event in tool_events)
    assert any(tool["id"] == "file.read" for tool in gateway.member_calls[0]["tools"])
    done = events[-1]
    assert done["type"] == "done"
    assert "file.read" in done["swarm"]["tools"]
