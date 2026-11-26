"""The swarm runtime: IPC, shared memory, leader planning, feedback and limits."""

from __future__ import annotations

import copy
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from mariana.agents.swarm import (
    COMPACTED_PREFIX,
    CONTEXT_HANDLE,
    RESERVED_MEMBER_IDS,
    SwarmBlackboard,
    SwarmCancelled,
    SwarmChannel,
    SwarmContextBank,
    SwarmContextPool,
    SwarmCoordinator,
    SwarmError,
    SwarmLimitError,
    SwarmLimits,
    SwarmMember,
    SwarmTimeout,
    _elide,
    _finish_reserve,
    _json_object,
    _normalise_tool_arguments,
    _relative_target,
    _retry_tokens,
    _usable_text,
    snapshot_due,
)

LIMITS = SwarmLimits(max_agents=2, max_steps=4, max_rounds=1, max_seconds=20,
                     max_messages=40, max_message_chars=200, max_memory_entries=20)


class SwarmGateway:
    """Answers each swarm protocol phase from its system prompt marker."""

    def __init__(self, free_text: bool = False):
        self.free_text = free_text
        self.calls: list[dict] = []

    def catalog(self, force=False):
        return [{"id": "fake", "available": True, "capabilities": {"chat": True}}]

    def chat(self, config):
        self.calls.append(copy.deepcopy(config))
        if self.free_text:
            return {"reply": "free text answer", "usage": {}}
        system = config["messages"][0]["content"]
        user = config["messages"][-1]["content"]
        if "Plan a focused team" in system:
            reply = json.dumps({
                "strategy": "peer_review", "rounds": 1,
                "agents": [
                    {"id": "a1", "label": "Reader", "role": "reader",
                     "task": "read the file"},
                    {"id": "a2", "label": "Checker", "role": "checker",
                     "task": "check the file"},
                ],
                "notes": "split the reading",
            })
        elif "Assign focused peer reviews" in system:
            reply = json.dumps({"reviews": [
                {"reviewer": "a1", "target": "a2", "focus": "accuracy"},
                {"reviewer": "a2", "target": "a1", "focus": "completeness"},
            ]})
        elif "Decide whether another revision round" in system:
            reply = json.dumps({"continue": False, "reason": "good enough"})
        elif "You are reviewing" in system:
            target = "a2" if "reviewing Checker" in system else "a1"
            reply = json.dumps({"review": {
                "target": target, "score": 4, "strengths": ["clear"],
                "issues": ["missing proof"], "suggestions": ["add a citation"],
                "summary": "solid but add evidence",
            }})
        elif "Synthesise the members' work" in system:
            reply = json.dumps({"final": "merged answer", "summary": "done",
                                "confidence": 0.8})
        elif "You are one member of the swarm" in system:
            if "Phase: revise" in user:
                reply = json.dumps({"final": "revised findings"})
            elif "returned" in user:
                reply = json.dumps({
                    "say": {"to": "all", "kind": "status", "text": "done reading"},
                    "memory": {"action": "set", "key": "finding.reader",
                               "value": {"path": "x.txt"}},
                    "final": "findings for the leader",
                })
            else:
                reply = json.dumps({"tool": "file.read", "input": {"path": "x.txt"}})
        else:
            reply = json.dumps({"final": "unexpected phase"})
        return {"reply": reply, "usage": {"prompt_tokens": 3, "completion_tokens": 2}}


class DependencyGateway(SwarmGateway):
    """A leader plan with criteria, per-member plans and a dependency edge."""

    def chat(self, config):
        system = config["messages"][0]["content"]
        if "Plan a focused team" in system:
            self.calls.append(copy.deepcopy(config))
            return {"reply": json.dumps({
                "strategy": "map_reduce", "rounds": 0,
                "success_criteria": ["both files read", "answer cites evidence"],
                "agents": [
                    {"id": "reader", "label": "Reader", "role": "reader",
                     "task": "read the first file", "tools": ["file.read"],
                     "plan": [{"text": "open the file", "status": "active"}]},
                    {"id": "checker", "label": "Checker", "role": "checker",
                     "task": "check the reader's evidence", "tools": ["file.read"],
                     "depends_on": ["reader"],
                     "plan": [{"text": "compare the findings", "status": "pending"}]},
                ],
                "notes": "read first, then check",
            }), "usage": {}}
        return super().chat(config)


def test_leader_plan_runs_dependencies_in_waves_with_criteria():
    events = []
    gateway = DependencyGateway()
    coordinator = SwarmCoordinator(
        gateway=gateway, swarm_id="swarm-waves", goal="compare two files", model="fake",
        tools=[{"id": "file.read", "type": "file.read"}],
        executor=lambda member, tool_id, arguments: ({"content": "hello"}, True),
        limits=LIMITS, on_event=lambda kind, **fields: events.append((kind, fields)),
    )
    result = coordinator.run(strategy="auto", rounds=0)

    assert result["ok"] is True, result.get("error")
    assert result["success_criteria"] == ["both files read", "answer cites evidence"]
    plan_event = next(fields for kind, fields in events if kind == "swarm_plan")
    assert plan_event["success_criteria"] == result["success_criteria"]
    agents = {item["id"]: item for item in result["agents"]}
    assert agents["checker"]["depends_on"] == ["reader"]
    assert agents["reader"]["plan"][0] == {"text": "open the file", "status": "active"}
    wave = next(fields for kind, fields in events
                if kind == "swarm_leader" and fields.get("wave") == 2)
    assert "dependencies finished" in wave["message"]
    member_calls = [call for call in gateway.calls
                    if "You are one member of the swarm" in call["messages"][0]["content"]]
    first_reader = next(index for index, call in enumerate(member_calls)
                        if "You are Reader (reader)" in call["messages"][0]["content"])
    first_checker = next(index for index, call in enumerate(member_calls)
                         if "You are Checker (checker)" in call["messages"][0]["content"])
    assert first_reader < first_checker
    assert result["blackboard"]
    stored = next(item for item in result["blackboard"] if item["key"] == "swarm.plan")
    assert stored["value"]["success_criteria"] == result["success_criteria"]
    assert stored["value"]["members"][1]["depends_on"] == ["reader"]


def test_a_dependency_cycle_is_ignored_with_a_warning():
    class CycleGateway(SwarmGateway):
        def chat(self, config):
            system = config["messages"][0]["content"]
            if "Plan a focused team" in system:
                self.calls.append(copy.deepcopy(config))
                return {"reply": json.dumps({
                    "strategy": "map_reduce", "rounds": 0,
                    "agents": [
                        {"id": "a1", "label": "A", "role": "r", "task": "one",
                         "tools": ["file.read"], "depends_on": ["a2"]},
                        {"id": "a2", "label": "B", "role": "r", "task": "two",
                         "tools": ["file.read"], "depends_on": ["a1"]},
                    ],
                }), "usage": {}}
            return super().chat(config)

    gateway = CycleGateway()
    coordinator = SwarmCoordinator(
        gateway=gateway, swarm_id="swarm-cycle", goal="cycle", model="fake",
        tools=[{"id": "file.read", "type": "file.read"}],
        executor=lambda member, tool_id, arguments: ({"content": "ok"}, True),
        limits=LIMITS,
    )
    result = coordinator.run(strategy="map_reduce", rounds=0)
    assert result["ok"] is True
    assert any("cycle" in warning.lower() for warning in result["warnings"])
    assert all(agent["depends_on"] == [] for agent in result["agents"])


class ReflectionGateway(SwarmGateway):
    """One member that plans, reads, drafts an answer, then answers the nudged check."""

    def chat(self, config):
        system = config["messages"][0]["content"]
        user = config["messages"][-1]["content"]
        if "Plan a focused team" in system:
            self.calls.append(copy.deepcopy(config))
            return {"reply": json.dumps({
                "strategy": "map_reduce", "rounds": 0,
                "agents": [{"id": "worker", "label": "Worker", "role": "researcher",
                            "task": "read and answer", "tools": ["file.read"]}],
            }), "usage": {}}
        if "You are one member of the swarm" in system:
            self.calls.append(copy.deepcopy(config))
            if "You are about to finish" in user:
                reply = json.dumps({"final": "checked answer"})
            elif "returned:" in user:
                reply = json.dumps({"final": "draft answer"})
            else:
                reply = json.dumps({
                    "plan": [{"text": "read the evidence", "status": "active"}],
                    "tool": "file.read", "input": {"path": "x.txt"},
                })
            return {"reply": reply, "usage": {}}
        return super().chat(config)


def test_a_member_plan_is_persisted_and_rechecked_before_finishing():
    events = []
    gateway = ReflectionGateway()
    coordinator = SwarmCoordinator(
        gateway=gateway, swarm_id="swarm-plan", goal="answer from the file", model="fake",
        tools=[{"id": "file.read", "type": "file.read"}],
        executor=lambda member, tool_id, arguments: ({"content": "evidence"}, True),
        limits=LIMITS, on_event=lambda kind, **fields: events.append((kind, fields)),
    )
    result = coordinator.run(strategy="map_reduce", rounds=0)

    assert result["ok"] is True, result.get("error")
    agent = result["agents"][0]
    assert agent["plan"] == [{"text": "read the evidence", "status": "active"}]
    assert agent["plan_version"] == 1
    assert agent["output"] == "checked answer"
    plan_events = [fields for kind, fields in events if kind == "swarm_agent_plan"]
    assert plan_events and plan_events[0]["plan"][0]["status"] == "active"
    assert any("unfinished plan" in fields.get("message", "").lower()
               for kind, fields in events if kind == "swarm_agent_status")
    assert any("unfinished plan step" in warning for warning in result["warnings"])


def test_a_temporary_read_failure_is_retried_once():
    events = []
    calls = []

    def executor(member, tool_id, arguments):
        calls.append(tool_id)
        if len(calls) == 1:
            return {"error": "connection reset by peer"}, False
        return {"content": "hello"}, True

    coordinator = SwarmCoordinator(
        gateway=SwarmGateway(), swarm_id="swarm-retry", goal="read x", model="fake",
        tools=[{"id": "file.read", "type": "file.read"}], executor=executor,
        limits=SwarmLimits(max_agents=1, max_steps=4, max_rounds=1, max_seconds=20,
                            max_messages=40, max_message_chars=200, max_memory_entries=20),
        on_event=lambda kind, **fields: events.append((kind, fields)),
    )
    result = coordinator.run(agents=[{"id": "a1", "task": "read the file"}],
                             strategy="map_reduce", rounds=0)

    assert result["ok"] is True, result.get("error")
    assert calls == ["file.read", "file.read"]
    agent = result["agents"][0]
    assert agent["tool_calls"] == 1 and agent["tool_failures"] == 0
    assert agent["tool_retries"] == 1
    assert any("retrying file.read" in fields.get("message", "").lower()
               for kind, fields in events if kind == "swarm_agent_status")
    tool_events = [fields for kind, fields in events if kind == "swarm_agent_tool"]
    assert tool_events[-1]["attempts"] == 2 and tool_events[-1]["ok"] is True


def test_a_rejected_write_is_not_retried():
    calls = []

    class FinalGateway(SwarmGateway):
        def __init__(self):
            super().__init__()
            self.member_turns = 0

        def chat(self, config):
            system = config["messages"][0]["content"]
            if "You are one member of the swarm" in system:
                self.calls.append(copy.deepcopy(config))
                self.member_turns += 1
                reply = (json.dumps({"tool": "file.write",
                                     "input": {"path": "notes/a.txt", "content": "x"}})
                         if self.member_turns == 1 else json.dumps({"final": "reported"}))
                return {"reply": reply, "usage": {}}
            return super().chat(config)

    def executor(member, tool_id, arguments):
        calls.append(tool_id)
        return {"error": "Path is outside the member's write scope"}, False

    coordinator = SwarmCoordinator(
        gateway=FinalGateway(), swarm_id="swarm-no-retry", goal="write x", model="fake",
        tools=[{"id": "file.write", "type": "file.write"}], executor=executor,
        limits=SwarmLimits(max_agents=1, max_steps=4, max_rounds=1, max_seconds=20,
                           max_messages=40, max_message_chars=200, max_memory_entries=20),
        allow_write=True,
    )
    result = coordinator.run(
        agents=[{"id": "a1", "task": "write a file", "writes": ["notes/"],
                 "read_only": False}],
        strategy="map_reduce", rounds=0,
    )
    assert result["ok"] is True, result.get("error")
    assert calls == ["file.write"]
    agent = result["agents"][0]
    assert agent["tool_failures"] == 1 and agent["tool_retries"] == 0
    assert agent["output"] == "reported"


def test_leader_gap_analysis_reaches_the_synthesis():
    class GapGateway(SwarmGateway):
        def chat(self, config):
            system = config["messages"][0]["content"]
            if "Plan a focused team" in system:
                self.calls.append(copy.deepcopy(config))
                return {"reply": json.dumps({
                    "strategy": "peer_review", "rounds": 2,
                    "agents": [
                        {"id": "a1", "label": "Reader", "role": "reader",
                         "task": "read the file"},
                        {"id": "a2", "label": "Checker", "role": "checker",
                         "task": "check the file"},
                    ],
                }), "usage": {}}
            if "Decide whether another revision round" in system:
                self.calls.append(copy.deepcopy(config))
                return {"reply": json.dumps({
                    "continue": False, "reason": "one gap remains",
                    "gaps": ["no source for the second claim"],
                }), "usage": {}}
            if "Synthesise the members' work" in system:
                self.calls.append(copy.deepcopy(config))
                user = config["messages"][-1]["content"]
                assert "Open gaps from the review decision" in user
                assert "Success criteria (check each against the answer)" in user
                return {"reply": json.dumps({
                    "final": "answer with a caveat", "summary": "done",
                    "unmet_criteria": ["both files read"], "confidence": 0.6,
                }), "usage": {}}
            return super().chat(config)

    gateway = GapGateway()
    coordinator = SwarmCoordinator(
        gateway=gateway, swarm_id="swarm-gaps", goal="explain x", model="fake",
        tools=[{"id": "file.read", "type": "file.read"}],
        executor=lambda member, tool_id, arguments: ({"content": "hello"}, True),
        limits=SwarmLimits(max_agents=2, max_steps=4, max_rounds=2, max_seconds=20,
                           max_messages=40, max_message_chars=200, max_memory_entries=20),
        criteria="every claim needs evidence",
    )
    result = coordinator.run(
        agents=[{"id": "a1", "task": "read the file"},
                {"id": "a2", "task": "check the file"}],
        strategy="peer_review", rounds=2,
    )

    assert result["ok"] is True, result.get("error")
    assert result["success_criteria"] == ["every claim needs evidence"]
    assert result["gaps"] == ["no source for the second claim"]
    assert result["unmet_criteria"] == ["both files read"]
    assert any("Unmet success criteria" in warning for warning in result["warnings"])


def test_channel_rejects_unknown_recipients_oversize_and_quota():
    channel = SwarmChannel(["a", "b"], SwarmLimits(max_messages=15, max_message_chars=20))
    message = channel.send("a", "b", "hello")
    assert message["from"] == "a" and message["to"] == "b"
    assert channel.unread("b")[0]["text"] == "hello"
    assert channel.unread("b") == []  # reading marks it delivered

    with pytest.raises(SwarmLimitError, match="Unknown recipient"):
        channel.send("a", "outsider", "hi")
    with pytest.raises(SwarmLimitError, match="needs some text"):
        channel.send("a", "b", "   ")
    with pytest.raises(SwarmLimitError, match="at most 20"):
        channel.send("a", "b", "x" * 100)
    with pytest.raises(SwarmLimitError, match="reply_to"):
        channel.send("a", "b", "hi", reply_to="msg-missing")
    with pytest.raises(SwarmLimitError, match="cannot send a message to itself"):
        channel.send("a", "a", "hi")
    with pytest.raises(SwarmLimitError, match="kind"):
        channel.send("a", "b", "hi", kind="command")

    broadcast = channel.send("b", "all", "broadcast")
    assert channel.unread("a")[0]["to"] == "all"
    reply = channel.send("a", "b", "reply", reply_to=broadcast["id"])
    assert reply["reply_to"] == broadcast["id"]
    for _ in range(3):
        channel.send("a", "b", "one")
    with pytest.raises(SwarmLimitError, match="quota"):
        channel.send("a", "b", "too many")


def test_blackboard_versions_limits_and_digest():
    board = SwarmBlackboard(max_entries=2, max_chars=60)
    first = board.write("a", "finding.one", {"value": 1}, note="first")
    assert first["version"] == 1 and board.read("finding.one")["value"] == {"value": 1}
    second = board.write("b", "finding.one", {"value": 2})
    assert second["version"] == 2 and second["author"] == "b"
    board.write("a", "finding.two", "two")
    with pytest.raises(SwarmLimitError, match="at most 2 entries"):
        board.write("a", "finding.three", "three")
    with pytest.raises(SwarmLimitError, match="key"):
        board.write("a", "../escape", "nope")
    with pytest.raises(SwarmLimitError, match="at most 60"):
        board.write("a", "finding.large", "x" * 200)
    assert "finding.one (v2, b)" in board.digest()
    assert len(board) == 2
    assert board.delete("finding.two") and not board.delete("finding.two")


def test_coordinator_plans_works_reviews_revises_and_synthesises():
    executor_calls = []
    events = []

    def executor(member, tool_id, arguments):
        executor_calls.append((member["id"], tool_id, arguments))
        return {"path": "x.txt", "content": "hello"}, True

    coordinator = SwarmCoordinator(
        gateway=SwarmGateway(), swarm_id="swarm-test", goal="understand x", model="fake",
        tools=[{"id": "file.read", "type": "file.read"}], executor=executor,
        limits=LIMITS, on_event=lambda kind, **fields: events.append((kind, fields)),
    )
    result = coordinator.run(
        agents=[{"id": "a1", "task": "read the file"},
                {"id": "a2", "task": "check the file"}],
        strategy="peer_review", rounds=1,
    )

    assert result["ok"] is True, result.get("error")
    assert result["final"] == "merged answer"
    assert result["strategy"] == "peer_review" and result["round"] == 1
    assert len(result["agents"]) == 2
    assert all(agent["status"] == "done" for agent in result["agents"])
    assert all(agent["version"] == 2 and agent["revisions"] == 1 for agent in result["agents"])
    assert len(result["reviews"]) == 2
    assert all(review["score"] == 4.0 for review in result["reviews"])
    assert result["messages"] >= 4
    assert any(entry["key"] == "finding.reader" for entry in result["blackboard"])
    assert executor_calls and executor_calls[0][1] == "file.read"

    kinds = [kind for kind, _ in events]
    assert {"swarm_start", "swarm_plan", "swarm_round", "swarm_agent_start",
            "swarm_agent_tool", "swarm_message", "swarm_memory", "swarm_review",
            "swarm_agent_end", "swarm_end"} <= set(kinds)
    end = [fields for kind, fields in events if kind == "swarm_end"][0]
    assert end["ok"] is True and end["status"] == "completed"
    snapshot = coordinator.snapshot()
    assert snapshot["status"] == "completed" and snapshot["strategy"] == "peer_review"
    assert len(snapshot["messages"]) >= 4


def test_coordinator_uses_a_safe_default_when_models_return_free_text():
    coordinator = SwarmCoordinator(
        gateway=SwarmGateway(free_text=True), swarm_id="swarm-text", goal="answer",
        model="fake", tools=[], limits=LIMITS,
    )
    result = coordinator.run(agents=None, strategy="auto", rounds=1)
    assert result["ok"] is True
    assert result["final"]
    assert result["warnings"]
    assert len(result["agents"]) == 1


def test_swarm_rejects_write_tools_when_write_is_disabled():
    calls = []

    def executor(member, tool_id, arguments):
        calls.append(tool_id)
        return {"written": True}, True

    class WriteGateway(SwarmGateway):
        def chat(self, config):
            system = config["messages"][0]["content"]
            if "Plan a focused team" in system:
                return {"reply": json.dumps({
                    "agents": [{"id": "a1", "task": "edit x",
                                "tools": ["file.write"], "writes": ["x.txt"]}],
                }), "usage": {}}
            if "You are one member" in system:
                return {"reply": json.dumps({
                    "tool": "file.write", "input": {"path": "x.txt", "content": "x"},
                }), "usage": {}}
            if "Synthesise" in system:
                return {"reply": json.dumps({"final": "read only"}), "usage": {}}
            return {"reply": "free text", "usage": {}}

    coordinator = SwarmCoordinator(
        gateway=WriteGateway(), swarm_id="swarm-write", goal="edit",
        model="fake", tools=[{"id": "file.write", "type": "file.write"}],
        executor=executor, limits=LIMITS, allow_write=False,
    )
    result = coordinator.run(agents=[{"id": "a1", "task": "edit x"}],
                             strategy="map_reduce", rounds=0)
    assert calls == []
    # The write is refused; the member still answers from the refusal instead of
    # failing the whole Chat turn.
    assert result["ok"] is True
    assert result["agents"][0]["output"]
    assert any("tool budget" in warning.lower() for warning in result["warnings"]), result


def test_swarm_rejects_writes_outside_a_members_scope():
    calls = []

    def executor(member, tool_id, arguments):
        calls.append((member["id"], tool_id, arguments.get("path")))
        return {"written": True}, True

    class ScopeGateway(SwarmGateway):
        def chat(self, config):
            system = config["messages"][0]["content"]
            user = config["messages"][-1]["content"]
            if "Plan a focused team" in system:
                return {"reply": json.dumps({"agents": [{
                    "id": "a1", "task": "edit the source",
                    "tools": ["file.write"], "writes": ["src"],
                }]}), "usage": {}}
            if "You are one member" in system:
                if "rejected" in user.lower() or "outside its scope" in user.lower():
                    return {"reply": json.dumps({"final": "blocked as expected"}), "usage": {}}
                return {"reply": json.dumps({"tool": "file.write",
                                             "input": {"path": "other/x.txt",
                                                       "content": "x"}}), "usage": {}}
            if "Synthesise" in system:
                return {"reply": json.dumps({"final": "done"}), "usage": {}}
            return {"reply": "free text", "usage": {}}

    coordinator = SwarmCoordinator(
        gateway=ScopeGateway(), swarm_id="swarm-scope", goal="edit", model="fake",
        tools=[{"id": "file.write", "type": "file.write"}], executor=executor,
        limits=LIMITS, allow_write=True,
    )
    result = coordinator.run(agents=[{"id": "a1", "task": "edit the source"}],
                             strategy="map_reduce", rounds=0)
    assert result["ok"] is True
    assert calls == []


def test_write_capability_requires_an_explicit_scope():
    calls = []

    def executor(member, tool_id, arguments):
        calls.append((member["id"], tool_id))
        return {"written": True}, True

    class NoScopeGateway(SwarmGateway):
        def chat(self, config):
            system = config["messages"][0]["content"]
            if "Plan a focused team" in system:
                return {"reply": json.dumps({"agents": [{
                    "id": "a1", "task": "edit without a scope",
                    "tools": ["file.write"],
                }]}), "usage": {}}
            if "You are one member" in system:
                return {"reply": json.dumps({"tool": "file.write",
                                             "input": {"path": "x.txt",
                                                       "content": "x"}}), "usage": {}}
            if "Synthesise" in system:
                return {"reply": json.dumps({"final": "no write scope"}), "usage": {}}
            return {"reply": "free text", "usage": {}}

    coordinator = SwarmCoordinator(
        gateway=NoScopeGateway(), swarm_id="swarm-no-scope", goal="edit", model="fake",
        tools=[{"id": "file.write", "type": "file.write"}], executor=executor,
        limits=LIMITS, allow_write=True,
    )
    result = coordinator.run(agents=[{"id": "a1", "task": "edit without a scope"}],
                             strategy="map_reduce", rounds=0)
    assert calls == []
    # No write scope means no write tool, but the run explains that instead of
    # aborting the whole swarm.
    assert result["ok"] is True
    assert result["agents"][0]["output"]
    assert any("tool budget" in warning.lower() for warning in result["warnings"]), result


def test_shell_cannot_bypass_a_members_write_scope():
    calls = []

    def executor(member, tool_id, arguments):
        calls.append((tool_id, arguments))
        return {"ok": True}, True

    class ShellGateway(SwarmGateway):
        def chat(self, config):
            system = config["messages"][0]["content"]
            user = config["messages"][-1]["content"]
            if "Plan a focused team" in system:
                return {"reply": json.dumps({"agents": [{
                    "id": "a1", "task": "run a command",
                    "tools": ["shell.exec"], "writes": ["src"],
                }]}), "usage": {}}
            if "You are one member" in system:
                if "write scope" in user.lower():
                    return {"reply": json.dumps({"final": "shell refused"}), "usage": {}}
                return {"reply": json.dumps({
                    "tool": "shell.exec",
                    "input": {"command": "src/x; rm -rf build"},
                }), "usage": {}}
            if "Synthesise" in system:
                return {"reply": json.dumps({"final": "done"}), "usage": {}}
            return {"reply": "free text", "usage": {}}

    coordinator = SwarmCoordinator(
        gateway=ShellGateway(), swarm_id="swarm-shell-scope", goal="run", model="fake",
        tools=[{"id": "shell.exec", "type": "shell.exec"}], executor=executor,
        limits=LIMITS, allow_write=True,
    )
    result = coordinator.run(agents=[{"id": "a1", "task": "run a command"}],
                             strategy="map_reduce", rounds=0)
    assert result["ok"] is True
    assert calls == []


def test_synthesis_without_a_final_answer_falls_back_to_member_results():
    class NoFinalGateway(SwarmGateway):
        def chat(self, config):
            system = config["messages"][0]["content"]
            if "Plan a focused team" in system:
                return {"reply": json.dumps({"agents": [{"id": "a1", "task": "answer"}],
                                             "rounds": 0}), "usage": {}}
            if "You are one member" in system:
                return {"reply": json.dumps({"final": "member finding"}), "usage": {}}
            if "Synthesise" in system:
                return {"reply": json.dumps({"summary": "no final field"}), "usage": {}}
            return {"reply": "free text", "usage": {}}

    coordinator = SwarmCoordinator(
        gateway=NoFinalGateway(), swarm_id="swarm-no-final", goal="answer", model="fake",
        tools=[], limits=LIMITS,
    )
    result = coordinator.run(agents=[{"id": "a1", "task": "answer"}],
                             strategy="map_reduce", rounds=0)
    assert result["ok"] is True
    assert "member finding" in result["final"]
    assert any("without a final" in warning for warning in result["warnings"])


def test_synthesis_accepts_a_non_string_final_value():
    class NumberFinalGateway(SwarmGateway):
        def chat(self, config):
            system = config["messages"][0]["content"]
            if "Plan a focused team" in system:
                return {"reply": json.dumps({"agents": [{"id": "a1", "task": "answer"}],
                                             "rounds": 0}), "usage": {}}
            if "You are one member" in system:
                return {"reply": json.dumps({"final": "finding"}), "usage": {}}
            if "Synthesise" in system:
                return {"reply": json.dumps({"final": 42}), "usage": {}}
            return {"reply": "free text", "usage": {}}

    coordinator = SwarmCoordinator(
        gateway=NumberFinalGateway(), swarm_id="swarm-number-final", goal="answer",
        model="fake", tools=[], limits=LIMITS,
    )
    result = coordinator.run(agents=[{"id": "a1", "task": "answer"}],
                             strategy="map_reduce", rounds=0)
    assert result["ok"] is True and result["final"] == "42"


def test_leader_cannot_raise_the_requested_feedback_rounds():
    class FinalGateway(SwarmGateway):
        def chat(self, config):
            system = config["messages"][0]["content"]
            if "Plan a focused team" in system:
                return {"reply": json.dumps({"rounds": 2, "agents": [
                    {"id": "a1", "task": "one"}, {"id": "a2", "task": "two"},
                ]}), "usage": {}}
            if "You are one member" in system:
                return {"reply": json.dumps({"final": "done"}), "usage": {}}
            if "Synthesise" in system:
                return {"reply": json.dumps({"final": "merged"}), "usage": {}}
            return {"reply": "free text", "usage": {}}

    coordinator = SwarmCoordinator(
        gateway=FinalGateway(), swarm_id="swarm-round-cap", goal="answer", model="fake",
        tools=[], limits=LIMITS,
    )
    result = coordinator.run(agents=[{"id": "a1", "task": "one"},
                                     {"id": "a2", "task": "two"}],
                             strategy="peer_review", rounds=0)
    assert result["ok"] is True
    assert result["round"] == 0
    assert result["reviews"] == []


def test_pipeline_strategy_hands_each_member_the_previous_work():
    class PipelineGateway(SwarmGateway):
        def __init__(self):
            super().__init__()
            self.saw_previous = False

        def chat(self, config):
            system = config["messages"][0]["content"]
            user = config["messages"][-1]["content"]
            if "Plan a focused team" in system:
                return {"reply": json.dumps({
                    "strategy": "pipeline", "rounds": 0,
                    "agents": [{"id": "a1", "task": "first stage"},
                               {"id": "a2", "task": "second stage"}],
                }), "usage": {}}
            if "You are one member" in system:
                if "Previous stages to build on" in user:
                    self.saw_previous = True
                return {"reply": json.dumps({"final": "stage output"}), "usage": {}}
            if "Synthesise" in system:
                return {"reply": json.dumps({"final": "pipeline result"}), "usage": {}}
            return {"reply": "free text", "usage": {}}

    gateway = PipelineGateway()
    coordinator = SwarmCoordinator(
        gateway=gateway, swarm_id="swarm-pipeline", goal="stage the work", model="fake",
        tools=[], limits=LIMITS,
    )
    result = coordinator.run(agents=[{"id": "a1", "task": "first stage"},
                                     {"id": "a2", "task": "second stage"}],
                             strategy="auto", rounds=0)
    assert result["ok"] is True and result["final"] == "pipeline result"
    assert result["strategy"] == "pipeline"
    assert gateway.saw_previous
    assert any("pipeline has no dependencies" in warning
               for warning in result["warnings"])


def test_operator_message_is_refused_after_the_swarm_finishes():
    coordinator = SwarmCoordinator(
        gateway=SwarmGateway(), swarm_id="swarm-op", goal="answer", model="fake",
        tools=[], limits=LIMITS,
    )
    coordinator.run(agents=[{"id": "a1", "task": "answer"}], strategy="map_reduce", rounds=0)
    with pytest.raises(SwarmLimitError, match="finished"):
        coordinator.post_operator_message("please stop")


def test_a_failed_leader_degrades_to_member_results():
    class FailingLeaderGateway(SwarmGateway):
        def chat(self, config):
            system = config["messages"][0]["content"]
            if "Plan a focused team" in system or "Synthesise" in system:
                raise RuntimeError("leader model unavailable")
            if "You are one member of the swarm" in system:
                return {"reply": json.dumps({"final": "member finding"}), "usage": {}}
            return super().chat(config)

    coordinator = SwarmCoordinator(
        gateway=FailingLeaderGateway(), swarm_id="swarm-leader-down", goal="answer",
        model="fake", tools=[], limits=LIMITS,
    )
    result = coordinator.run(agents=[{"id": "a1", "task": "answer"}],
                             strategy="map_reduce", rounds=0)
    assert result["ok"] is True
    assert "member finding" in result["final"]
    assert any("synthesis failed" in warning.lower() for warning in result["warnings"])


def test_a_time_budget_ends_a_slow_swarm():
    class SlowGateway(SwarmGateway):
        def chat(self, config):
            raise SwarmTimeout("Swarm time budget exhausted")

    coordinator = SwarmCoordinator(
        gateway=SlowGateway(), swarm_id="swarm-slow", goal="answer", model="fake",
        tools=[], limits=SwarmLimits(max_seconds=10, max_steps=1),
    )
    result = coordinator.run(agents=[{"id": "a1", "task": "answer"}],
                             strategy="map_reduce", rounds=0)
    assert result["ok"] is False
    assert "time budget" in result["error"].lower()


def test_json_parser_accepts_fences_and_plain_text():
    assert _json_object('```json\n{"tool": "file.read"}\n```') == {"tool": "file.read"}
    assert _json_object("no object") is None


def test_write_scope_targets_are_lexically_normalised():
    assert _relative_target("src/../etc/passwd") == "etc/passwd"
    assert _relative_target("./src/app.py") == "src/app.py"
    assert _relative_target("../escape") == "../"
    assert _relative_target("/absolute") == "../"


def test_blackboard_starts_with_goal_and_plan_and_shares_results():
    events = []
    coordinator = SwarmCoordinator(
        gateway=SwarmGateway(), swarm_id="swarm-shared", goal="understand x",
        model="fake", tools=[{"id": "file.read", "type": "file.read"}],
        executor=lambda member, tool_id, arguments: ({"content": "hello"}, True),
        limits=LIMITS, on_event=lambda kind, **fields: events.append((kind, fields)),
    )
    result = coordinator.run(
        agents=[{"id": "a1", "task": "read the file"},
                {"id": "a2", "task": "check the file"}],
        strategy="map_reduce", rounds=0,
    )

    assert result["ok"] is True, result.get("error")
    board = {entry["key"]: entry for entry in result["blackboard"]}
    assert board["swarm.goal"]["author"] == "leader"
    assert board["swarm.goal"]["value"] == "understand x"
    assert board["swarm.plan"]["author"] == "leader"
    assert [member["id"] for member in board["swarm.plan"]["value"]["members"]] == ["a1", "a2"]
    assert board["result.a1"]["author"] == "a1"
    assert board["result.a1"]["value"]["output"] == "findings for the leader"
    assert board["result.a2"]["author"] == "a2"
    automatic = [fields for kind, fields in events
                 if kind == "swarm_memory" and fields.get("automatic")]
    assert {fields["key"] for fields in automatic} == {"result.a1", "result.a2"}


def test_member_and_review_prompts_include_peer_results():
    limits = SwarmLimits(max_agents=3, max_steps=4, max_rounds=1, max_seconds=20,
                         max_messages=40, max_message_chars=200, max_memory_entries=20)
    coordinator = SwarmCoordinator(
        gateway=SwarmGateway(), swarm_id="swarm-digest", goal="understand x",
        model="fake", tools=[], limits=limits,
    )
    specs = coordinator._normalise_agents([
        {"id": "a1", "task": "read the file"},
        {"id": "a2", "task": "check the file"},
        {"id": "a3", "task": "check the logs"},
    ])
    coordinator._build_members(specs)
    first, second, third = coordinator._members
    first.output = "the file says hello"
    first.version = 2
    first.score = 4.5
    third.output = "the logs say goodbye"

    member_prompt = coordinator._member_messages(second, "work", [])
    member_text = "\n".join(item["content"] for item in member_prompt)
    assert "Peer results so far" in member_text
    assert "the file says hello" in member_text
    assert "(a1, researcher, v2, score 4.5)" in member_text
    assert "read the file" in member_text  # the team block names every member's task

    review_prompt = coordinator._review_messages(second, first, {"focus": "accuracy"}, 1)
    review_system, review_user = review_prompt[0]["content"], review_prompt[1]["content"]
    assert "Peer results (compare the target against them" in review_user
    assert "the logs say goodbye" in review_user
    assert "the file says hello" not in review_system  # the target is shown in full in the user turn
    assert "the file says hello" in review_user
    assert "read the file" in review_system


def test_duplicate_plan_tasks_are_skipped_with_a_warning():
    coordinator = SwarmCoordinator(
        gateway=SwarmGateway(), swarm_id="swarm-duplicate", goal="answer", model="fake",
        tools=[], limits=LIMITS,
    )
    specs = coordinator._normalise_agents([
        {"id": "a1", "task": "read the file"},
        {"id": "a2", "task": "  Read   the FILE "},
    ])
    assert [spec["id"] for spec in specs] == ["a1"]
    assert any("duplicate" in warning for warning in coordinator.warnings)


def test_result_sharing_survives_a_full_blackboard():
    limits = SwarmLimits(max_agents=2, max_steps=4, max_rounds=0, max_seconds=20,
                         max_messages=40, max_message_chars=200,
                         max_memory_entries=1, max_memory_chars=200)
    coordinator = SwarmCoordinator(
        gateway=SwarmGateway(), swarm_id="swarm-full-board", goal="answer", model="fake",
        tools=[{"id": "file.read", "type": "file.read"}],
        executor=lambda member, tool_id, arguments: ({"content": "hello"}, True),
        limits=limits,
    )
    result = coordinator.run(
        agents=[{"id": "a1", "task": "read the file"},
                {"id": "a2", "task": "check the file"}],
        strategy="map_reduce", rounds=0,
    )
    assert result["ok"] is True, result.get("error")
    assert result["blackboard_count"] <= 1


def test_prompt_trimming_keeps_the_newest_knowledge():
    assert _elide("short text", 80) == "short text"
    trimmed = _elide("HEAD " + "x" * 500 + " TAIL-MARKER", 120)
    assert trimmed.startswith("HEAD") and trimmed.endswith("TAIL-MARKER")
    assert "…" in trimmed and len(trimmed) <= 120

    coordinator = SwarmCoordinator(
        gateway=SwarmGateway(), swarm_id="swarm-budget", goal="answer", model="fake",
        tools=[], limits=SwarmLimits(max_prompt_chars=1_000),
    )
    fitted = coordinator._fit_messages([
        {"role": "system", "content": "s" * 1_500},
        {"role": "user", "content": "newest request"},
    ])
    assert fitted[-1]["content"] == "newest request"
    assert len(fitted[0]["content"]) <= 1_000
    assert sum(len(item["content"]) for item in fitted) <= 1_000

    fitted = coordinator._fit_messages([
        {"role": "system", "content": "s" * 900},
        {"role": "user", "content": "START " + "y" * 3_000 + " LATEST-FINDING"},
    ])
    assert fitted[-1]["content"].startswith("START")
    assert fitted[-1]["content"].endswith("LATEST-FINDING")
    assert sum(len(item["content"]) for item in fitted) <= 1_000


def test_a_hung_model_call_cannot_outlive_the_time_budget():
    class HungGateway(SwarmGateway):
        def __init__(self):
            super().__init__()
            self.release = threading.Event()

        def chat(self, config):
            system = config["messages"][0]["content"]
            if "You are one member of the swarm" in system:
                self.release.wait(5)
                return {"reply": json.dumps({"final": "late"}), "usage": {}}
            if "Plan a focused team" in system:
                return {"reply": json.dumps({
                    "strategy": "map_reduce", "rounds": 0,
                    "agents": [{"id": "a1", "task": "answer"}],
                }), "usage": {}}
            return super().chat(config)

    gateway = HungGateway()
    coordinator = SwarmCoordinator(
        gateway=gateway, swarm_id="swarm-hung", goal="answer", model="fake", tools=[],
        limits=SwarmLimits(max_agents=2, max_steps=2, max_rounds=0, max_seconds=1,
                           max_messages=20, max_message_chars=200),
    )
    started = time.monotonic()
    try:
        result = coordinator.run(agents=[{"id": "a1", "task": "answer"}],
                                 strategy="map_reduce", rounds=0)
    finally:
        gateway.release.set()
    elapsed = time.monotonic() - started

    assert result["ok"] is False
    assert "time budget" in result["error"].lower()
    assert elapsed < 4
    assert [agent["status"] for agent in result["agents"]] == ["failed"]
    assert result["agents"][0]["ended"]


def test_a_timed_out_tool_loop_keeps_the_work_already_collected():
    events = []

    class BusyGateway(SwarmGateway):
        def chat(self, config):
            self.calls.append(copy.deepcopy(config))
            system = config["messages"][0]["content"]
            user = config["messages"][-1]["content"]
            if "Plan a focused team" in system:
                return {"reply": json.dumps({
                    "strategy": "map_reduce", "rounds": 0,
                    "agents": [
                        {"id": "a1", "label": "Prod", "role": "reader",
                         "task": "find the answer"},
                        {"id": "a2", "label": "Busy", "role": "reader",
                         "task": "keep looking", "tools": ["file.read"]},
                    ],
                }), "usage": {}}
            if "Synthesise the members' work" in system:
                final = ("merged: prod finding" if "prod finding" in user else "merged")
                return {"reply": json.dumps({"final": final, "summary": "done",
                                             "confidence": 0.5}), "usage": {}}
            if "You are Prod (a1)" in system:
                return {"reply": json.dumps({"final": "prod finding"}), "usage": {}}
            if "You are Busy (a2)" in system:
                if "Tool budget reached" in user:
                    return {"reply": json.dumps({"final": "busy partial"}), "usage": {}}
                return {"reply": json.dumps({
                    "tool": "file.read", "input": {"path": "x.txt"}}), "usage": {}}
            return super().chat(config)

    coordinator = SwarmCoordinator(
        gateway=BusyGateway(), swarm_id="swarm-budget-partial", goal="answer",
        model="fake", tools=[{"id": "file.read", "type": "file.read"}],
        limits=SwarmLimits(max_agents=2, max_steps=10, max_rounds=0, max_seconds=10,
                           max_messages=40, max_message_chars=200),
        on_event=lambda kind, **fields: events.append(fields)
        if kind == "swarm_end" else None,
    )

    def executor(member, tool_id, arguments):
        if member.get("id") == "a2":
            # The next round is past the work deadline, but the finish reserve is intact.
            coordinator.work_deadline = time.monotonic() - 1
        return {"content": "observation"}, True

    coordinator.executor = executor
    result = coordinator.run(strategy="map_reduce", rounds=0)

    assert result["ok"] is True
    assert result["timed_out"] is True
    assert result["assembled"] is False
    assert "prod finding" in result["final"]
    assert any("work deadline" in warning for warning in result["warnings"])
    agents = {item["id"]: item for item in result["agents"]}
    assert agents["a1"]["status"] == "done"
    assert agents["a2"]["status"] == "done"
    assert agents["a2"]["output"] == "busy partial"
    assert events and events[-1]["timed_out"] is True and events[-1]["ok"] is True


def test_a_hung_member_does_not_discard_a_finished_members_work():
    class OneHungGateway(SwarmGateway):
        def __init__(self):
            super().__init__()
            self.release = threading.Event()

        def chat(self, config):
            system = config["messages"][0]["content"]
            if "Plan a focused team" in system:
                return {"reply": json.dumps({
                    "strategy": "map_reduce", "rounds": 0,
                    "agents": [
                        {"id": "fast", "label": "Fast", "role": "reader",
                         "task": "answer quickly"},
                        {"id": "hung", "label": "Hung", "role": "reader",
                         "task": "answer slowly"},
                    ],
                }), "usage": {}}
            if "Synthesise the members' work" in system:
                return {"reply": json.dumps({"final": "synthesised"}), "usage": {}}
            if "You are Fast (fast)" in system:
                return {"reply": json.dumps({"final": "fast finding"}), "usage": {}}
            if "You are Hung (hung)" in system:
                self.release.wait(5)
                return {"reply": json.dumps({"final": "late"}), "usage": {}}
            return super().chat(config)

    gateway = OneHungGateway()
    coordinator = SwarmCoordinator(
        gateway=gateway, swarm_id="swarm-hung-partial", goal="answer", model="fake",
        tools=[], limits=SwarmLimits(max_agents=2, max_steps=2, max_rounds=0, max_seconds=1,
                                     max_messages=20, max_message_chars=200),
        on_event=None,
    )
    started = time.monotonic()
    try:
        result = coordinator.run(strategy="map_reduce", rounds=0)
    finally:
        gateway.release.set()
    elapsed = time.monotonic() - started

    assert result["ok"] is True
    assert result["timed_out"] is True
    assert result["assembled"] is True
    assert result["final"].startswith("> The time budget ended before the leader")
    assert "fast finding" in result["final"]
    agents = {item["id"]: item for item in result["agents"]}
    assert agents["fast"]["status"] == "done"
    assert agents["hung"]["status"] == "failed"
    assert any("continues with the results it has" in warning
               for warning in result["warnings"])
    assert elapsed < 4


def test_member_tools_are_refused_once_the_work_budget_is_over():
    called = []
    coordinator = SwarmCoordinator(
        gateway=None, swarm_id="swarm-tool-deadline", goal="answer", model="fake",
        tools=[{"id": "file.read", "type": "file.read"}],
        executor=lambda *arguments: called.append(arguments) or ({"content": "x"}, True),
        limits=SwarmLimits(max_agents=1),
    )
    member = SwarmMember(id="a1", label="A", role="researcher", task="answer",
                         tools=["file.read"])
    coordinator.work_deadline = time.monotonic() - 1

    reply = coordinator._member_tool(
        member, {"tool": "file.read", "input": {"path": "x.txt"}}, {}, 1)

    assert "work budget is over" in reply.lower()
    assert coordinator.timed_out is True
    assert called == []


def test_the_finish_reserve_is_bounded_for_short_and_long_runs():
    assert _finish_reserve(0) == 0
    assert _finish_reserve(2) == 1.0
    assert _finish_reserve(20) == 5.0
    assert _finish_reserve(300) == 45.0
    assert _finish_reserve(1000) == 120.0
    assert _finish_reserve(1800) == 120.0


def test_peer_reviews_are_skipped_once_the_work_budget_is_over():
    coordinator = SwarmCoordinator(
        gateway=None, swarm_id="swarm-review-deadline", goal="answer", model="fake",
        limits=SwarmLimits(max_agents=1, max_reviews=2),
    )
    coordinator.work_deadline = time.monotonic() - 1

    assert coordinator._review_round(1) is False
    assert coordinator.timed_out is True
    assert any("reviews were skipped" in warning for warning in coordinator.warnings)


def test_a_synthesis_timeout_returns_the_members_work():
    class SynthesisTimeoutGateway(SwarmGateway):
        def chat(self, config):
            system = config["messages"][0]["content"]
            if "Plan a focused team" in system:
                return {"reply": json.dumps({
                    "strategy": "map_reduce", "rounds": 0,
                    "agents": [{"id": "a1", "label": "Reader", "role": "reader",
                                "task": "answer"}],
                }), "usage": {}}
            if "Synthesise the members' work" in system:
                raise SwarmTimeout("Swarm time budget exhausted")
            if "You are one member of the swarm" in system:
                return {"reply": json.dumps({"final": "member answer"}), "usage": {}}
            return super().chat(config)

    coordinator = SwarmCoordinator(
        gateway=SynthesisTimeoutGateway(), swarm_id="swarm-synthesis-timeout",
        goal="answer", model="fake", tools=[],
        limits=SwarmLimits(max_agents=1, max_steps=2, max_rounds=0, max_seconds=20),
    )
    result = coordinator.run(strategy="map_reduce", rounds=0)

    assert result["ok"] is True
    assert result["timed_out"] is True
    assert result["assembled"] is True
    assert "member answer" in result["final"]
    assert "## Reader" in result["final"]
    assert any("time budget ended before the leader" in warning
               for warning in result["warnings"])


def test_late_worker_events_are_ignored_after_the_run_ends():
    events = []
    coordinator = SwarmCoordinator(
        gateway=SwarmGateway(), swarm_id="swarm-late", goal="answer", model="fake",
        tools=[], limits=LIMITS, on_event=lambda kind, **fields: events.append(kind),
    )
    coordinator.status = "completed"
    coordinator._emit("swarm_agent_step", agent="a1")
    coordinator._emit("swarm_end", ok=True)
    assert events == ["swarm_end"]


def test_a_false_string_stops_the_feedback_loop():
    class DecisionGateway(SwarmGateway):
        def chat(self, config):
            system = config["messages"][0]["content"]
            if "Plan a focused team" in system:
                return {"reply": json.dumps({
                    "strategy": "peer_review", "rounds": 2,
                    "agents": [{"id": "a1", "task": "read the file"},
                               {"id": "a2", "task": "check the file"}],
                }), "usage": {}}
            if "Decide whether another revision round" in system:
                return {"reply": json.dumps({"continue": "false", "reason": "good enough"}),
                        "usage": {}}
            return super().chat(config)

    limits = SwarmLimits(max_agents=2, max_steps=4, max_rounds=2, max_seconds=20,
                         max_messages=40, max_message_chars=200, max_memory_entries=20)
    coordinator = SwarmCoordinator(
        gateway=DecisionGateway(), swarm_id="swarm-bool", goal="understand x", model="fake",
        tools=[{"id": "file.read", "type": "file.read"}],
        executor=lambda member, tool_id, arguments: ({"content": "hello"}, True),
        limits=limits,
    )
    result = coordinator.run(
        agents=[{"id": "a1", "task": "read the file"},
                {"id": "a2", "task": "check the file"}],
        strategy="peer_review", rounds=2,
    )
    assert result["ok"] is True
    assert result["round"] == 1
    assert len(result["reviews"]) == 2


def test_garbage_rounds_do_not_raise_before_the_run_starts():
    coordinator = SwarmCoordinator(
        gateway=SwarmGateway(), swarm_id="swarm-rounds", goal="answer", model="fake",
        tools=[{"id": "file.read", "type": "file.read"}],
        executor=lambda member, tool_id, arguments: ({"content": "hello"}, True),
        limits=LIMITS,
    )
    result = coordinator.run(agents=[{"id": "a1", "task": "read the file"}],
                             strategy="map_reduce", rounds="lots")
    assert result["ok"] is True
    assert result["requested_rounds"] == 0


def test_a_swarm_cannot_run_twice():
    coordinator = SwarmCoordinator(
        gateway=SwarmGateway(), swarm_id="swarm-once", goal="answer", model="fake",
        tools=[{"id": "file.read", "type": "file.read"}],
        executor=lambda member, tool_id, arguments: ({"content": "hello"}, True),
        limits=LIMITS,
    )
    coordinator.run(agents=[{"id": "a1", "task": "read the file"}],
                    strategy="map_reduce", rounds=0)
    with pytest.raises(SwarmError, match="already run"):
        coordinator.run(agents=[{"id": "a1", "task": "read the file"}],
                        strategy="map_reduce", rounds=0)


def test_a_failed_review_keeps_the_members_completed_work():
    class ReviewFailureGateway(SwarmGateway):
        def chat(self, config):
            system = config["messages"][0]["content"]
            if "You are reviewing" in system:
                raise RuntimeError("review model offline")
            return super().chat(config)

    limits = SwarmLimits(max_agents=2, max_steps=4, max_rounds=1, max_seconds=20,
                         max_messages=40, max_message_chars=200, max_memory_entries=20)
    coordinator = SwarmCoordinator(
        gateway=ReviewFailureGateway(), swarm_id="swarm-review-down", goal="understand x",
        model="fake", tools=[{"id": "file.read", "type": "file.read"}],
        executor=lambda member, tool_id, arguments: ({"content": "hello"}, True),
        limits=limits,
    )
    result = coordinator.run(
        agents=[{"id": "a1", "task": "read the file"},
                {"id": "a2", "task": "check the file"}],
        strategy="peer_review", rounds=1,
    )
    assert result["ok"] is True
    assert result["failed"] == 0
    assert all(agent["status"] == "done" for agent in result["agents"])
    assert all(agent["output"] for agent in result["agents"])
    assert any("review failed" in warning.lower() for warning in result["warnings"])


def test_a_review_with_a_wrong_target_is_corrected_to_the_assignment():
    class WrongTargetGateway(SwarmGateway):
        def chat(self, config):
            system = config["messages"][0]["content"]
            if "You are reviewing" in system:
                return {"reply": json.dumps({"review": {
                    "target": "a2", "score": 5, "strengths": ["clear"],
                    "issues": [], "suggestions": [], "summary": "good",
                }}), "usage": {}}
            return super().chat(config)

    limits = SwarmLimits(max_agents=2, max_steps=4, max_rounds=1, max_seconds=20,
                         max_messages=40, max_message_chars=200, max_memory_entries=20)
    coordinator = SwarmCoordinator(
        gateway=WrongTargetGateway(), swarm_id="swarm-wrong-target", goal="understand x",
        model="fake", tools=[{"id": "file.read", "type": "file.read"}],
        executor=lambda member, tool_id, arguments: ({"content": "hello"}, True),
        limits=limits,
    )
    result = coordinator.run(
        agents=[{"id": "a1", "task": "read the file"},
                {"id": "a2", "task": "check the file"}],
        strategy="peer_review", rounds=1,
    )
    assert result["ok"] is True
    pairs = {(review["reviewer"], review["target"]) for review in result["reviews"]}
    assert pairs == {("a1", "a2"), ("a2", "a1")}


def test_an_empty_model_reply_is_retried_once():
    class EmptyOnceGateway(SwarmGateway):
        def __init__(self):
            super().__init__()
            self.empty = False

        def chat(self, config):
            system = config["messages"][0]["content"]
            if "You are one member of the swarm" in system:
                if not self.empty:
                    self.empty = True
                    return {"reply": "", "usage": {}}
                return {"reply": json.dumps({"final": "recovered answer"}), "usage": {}}
            if "Plan a focused team" in system:
                return {"reply": json.dumps({
                    "strategy": "map_reduce", "rounds": 0,
                    "agents": [{"id": "a1", "task": "answer"}],
                }), "usage": {}}
            if "Synthesise" in system:
                return {"reply": json.dumps({"final": "final answer"}), "usage": {}}
            return super().chat(config)

    events = []
    coordinator = SwarmCoordinator(
        gateway=EmptyOnceGateway(), swarm_id="swarm-empty", goal="answer", model="fake",
        tools=[], limits=LIMITS,
        on_event=lambda kind, **fields: events.append((kind, fields)),
    )
    result = coordinator.run(agents=[{"id": "a1", "task": "answer"}],
                             strategy="map_reduce", rounds=0)
    assert result["ok"] is True
    assert result["agents"][0]["output"] == "recovered answer"
    assert result["agents"][0]["steps"] == 1
    assert result["final"] == "final answer"
    assert any(kind == "swarm_agent_status" and "empty reply" in fields.get("message", "")
               for kind, fields in events)


def test_a_truncated_reasoning_reply_retries_with_a_larger_budget():
    class ReasonerGateway(SwarmGateway):
        def __init__(self):
            super().__init__()
            self.member_calls = []

        def chat(self, config):
            system = config["messages"][0]["content"]
            if "Plan a focused team" in system:
                return {"reply": json.dumps({
                    "strategy": "map_reduce", "rounds": 0,
                    "agents": [{"id": "a1", "task": "answer"}],
                }), "usage": {}}
            if "You are one member of the swarm" in system:
                self.member_calls.append((config["max_tokens"], config.get("reasoning_effort")))
                if len(self.member_calls) == 1:
                    # A reasoning model can spend the whole budget thinking and write
                    # nothing visible, or stop mid-sentence at the limit.
                    return {
                        "reply": "",
                        "finish_reason": "length",
                        "usage": {"completion_tokens": config["max_tokens"]},
                    }
                return {"reply": json.dumps({"final": "recovered answer"}), "usage": {}}
            return super().chat(config)

    gateway = ReasonerGateway()
    coordinator = SwarmCoordinator(
        gateway=gateway, swarm_id="swarm-reasoner", goal="answer", model="fake",
        tools=[], limits=LIMITS,
    )
    result = coordinator.run(agents=[{"id": "a1", "task": "answer"}],
                             strategy="map_reduce", rounds=0)
    assert result["ok"] is True
    assert result["agents"][0]["output"] == "recovered answer"
    assert gateway.member_calls == [(2_500, None), (6_000, "low")]


def test_an_empty_synthesis_retries_with_a_larger_budget():
    class EmptySynthesis(SwarmGateway):
        def __init__(self):
            super().__init__()
            self.synthesis_calls = []

        def chat(self, config):
            system = config["messages"][0]["content"]
            if "Plan a focused team" in system:
                return {"reply": json.dumps({
                    "strategy": "map_reduce", "rounds": 0,
                    "agents": [{"id": "a1", "task": "answer"}],
                }), "usage": {}}
            if "Synthesise the members' work" in system:
                self.synthesis_calls.append((config["max_tokens"], config.get("reasoning_effort")))
                if len(self.synthesis_calls) == 1:
                    return {"reply": "", "finish_reason": "length",
                            "usage": {"completion_tokens": config["max_tokens"]}}
            if "You are one member of the swarm" in system:
                return {"reply": json.dumps({"final": "member finding"}), "usage": {}}
            return super().chat(config)

    gateway = EmptySynthesis()
    coordinator = SwarmCoordinator(
        gateway=gateway, swarm_id="swarm-empty-synthesis", goal="answer", model="fake",
        tools=[], limits=LIMITS,
    )
    result = coordinator.run(agents=[{"id": "a1", "task": "answer"}],
                             strategy="map_reduce", rounds=0)
    assert result["ok"] is True and result["final"] == "merged answer"
    assert gateway.synthesis_calls == [(4_000, None), (8_000, "low")]


def test_the_retry_budget_grows_without_shrinking_a_large_budget():
    assert _retry_tokens(300) == 6_000
    assert _retry_tokens(1_200) == 6_000
    assert _retry_tokens(4_000) == 8_000
    assert _retry_tokens(8_000) == 12_000
    assert _retry_tokens(20_000) == 20_000


def test_an_exhausted_reasoning_budget_reports_an_actionable_error():
    class EmptyReasoner(SwarmGateway):
        def chat(self, config):
            system = config["messages"][0]["content"]
            if "You are one member of the swarm" in system:
                return {"reply": "", "finish_reason": "length",
                        "usage": {"completion_tokens": config["max_tokens"]}}
            return super().chat(config)

    coordinator = SwarmCoordinator(
        gateway=EmptyReasoner(), swarm_id="swarm-empty-budget", goal="answer",
        model="fake", tools=[], limits=LIMITS,
    )
    result = coordinator.run(agents=[{"id": "a1", "task": "answer"}],
                             strategy="map_reduce", rounds=0)
    assert result["ok"] is False
    assert "output budget" in result["agents"][0]["error"]
    assert "Solo" in result["agents"][0]["error"]


def test_a_reworded_duplicate_is_not_added_as_a_new_member():
    coordinator = SwarmCoordinator(
        gateway=SwarmGateway(), swarm_id="swarm-merge", goal="answer", model="fake",
        tools=[], limits=LIMITS,
    )
    explicit = coordinator._normalise_agents([{"id": "a1", "task": "read the file"}])
    planned = coordinator._normalise_agents([{"id": "a2", "task": "  READ   the FILE "}])
    merged = coordinator._merge_agents(explicit, planned)
    assert [item["id"] for item in merged] == ["a1"]


def test_disabled_tools_are_not_offered_to_members():
    coordinator = SwarmCoordinator(
        gateway=SwarmGateway(), swarm_id="swarm-disabled", goal="answer", model="fake",
        tools=[{"id": "file.read", "type": "file.read", "enabled": "false"},
               {"id": "file.search", "type": "file.search", "enabled": True}],
        limits=LIMITS,
    )
    specs = coordinator._normalise_agents([{"id": "a1", "task": "read the file"}])
    coordinator._build_members(specs)
    assert coordinator._members[0].tools == ["file.search"]


def test_a_wildcard_tool_grant_offers_every_tool():
    coordinator = SwarmCoordinator(
        gateway=SwarmGateway(), swarm_id="swarm-wildcard", goal="research", model="fake",
        tools=[{"id": "file.read", "type": "file.read"},
               {"id": "http.fetch", "type": "http.fetch"}],
        limits=LIMITS,
    )
    specs = coordinator._normalise_agents([
        {"id": "a1", "task": "research broadly", "tools": ["*"]}])
    coordinator._build_members(specs)
    assert set(coordinator._members[0].tools) == {"file.read", "http.fetch"}


def test_a_single_web_tool_grant_extends_to_the_retrieval_family():
    coordinator = SwarmCoordinator(
        gateway=SwarmGateway(), swarm_id="swarm-web-family", goal="research", model="fake",
        tools=[{"id": "news.headlines", "type": "news.headlines"},
               {"id": "http.fetch", "type": "http.fetch"},
               {"id": "browser.fetch", "type": "browser.fetch"},
               {"id": "file.read", "type": "file.read"}],
        limits=LIMITS,
    )
    specs = coordinator._normalise_agents([
        {"id": "a1", "task": "read the news", "tools": ["news.headlines"]}])
    coordinator._build_members(specs)
    assert set(coordinator._members[0].tools) == {
        "news.headlines", "http.fetch", "browser.fetch"}


def test_the_web_family_skips_tools_that_are_disabled_or_absent():
    coordinator = SwarmCoordinator(
        gateway=SwarmGateway(), swarm_id="swarm-web-partial", goal="research", model="fake",
        tools=[{"id": "http.fetch", "type": "http.fetch"},
               {"id": "browser.fetch", "type": "browser.fetch", "enabled": False}],
        limits=LIMITS,
    )
    specs = coordinator._normalise_agents([
        {"id": "a1", "task": "read the news", "tools": ["http.fetch"]}])
    coordinator._build_members(specs)
    assert coordinator._members[0].tools == ["http.fetch"]


def test_a_blackboard_task_gets_file_write_with_a_matching_scope():
    coordinator = SwarmCoordinator(
        gateway=SwarmGateway(), swarm_id="swarm-board-scope", goal="research", model="fake",
        tools=[{"id": "news.headlines", "type": "news.headlines"},
               {"id": "file.write", "type": "file.write"}],
        limits=LIMITS, allow_write=True, allow_os_access=False,
    )
    specs = coordinator._normalise_agents([
        {"id": "a1", "task": "write the best items to blackboard/final.md",
         "tools": ["news.headlines"], "writes": ["blackboard"]}])
    coordinator._build_members(specs)
    member = coordinator._members[0]
    assert "file.write" in member.tools
    assert member.read_only is False


def test_a_blackboard_task_gets_file_write_without_a_scope_under_os_access():
    coordinator = SwarmCoordinator(
        gateway=SwarmGateway(), swarm_id="swarm-board-os", goal="research", model="fake",
        tools=[{"id": "news.headlines", "type": "news.headlines"},
               {"id": "file.write", "type": "file.write"}],
        limits=LIMITS, allow_write=True, allow_os_access=True,
    )
    specs = coordinator._normalise_agents([
        {"id": "a1", "task": "record the findings on the blackboard",
         "tools": ["news.headlines"]}])
    coordinator._build_members(specs)
    member = coordinator._members[0]
    assert "file.write" in member.tools
    assert member.read_only is False


def test_a_blackboard_task_stays_read_only_without_a_scope_or_os_access():
    coordinator = SwarmCoordinator(
        gateway=SwarmGateway(), swarm_id="swarm-board-closed", goal="research", model="fake",
        tools=[{"id": "news.headlines", "type": "news.headlines"},
               {"id": "file.write", "type": "file.write"}],
        limits=LIMITS, allow_write=True, allow_os_access=False,
    )
    specs = coordinator._normalise_agents([
        {"id": "a1", "task": "write the best items to blackboard/final.md",
         "tools": ["news.headlines"]}])
    coordinator._build_members(specs)
    member = coordinator._members[0]
    assert "file.write" not in member.tools
    assert member.read_only is True


def test_a_member_can_be_planned_with_no_tools():
    coordinator = SwarmCoordinator(
        gateway=SwarmGateway(), swarm_id="swarm-no-tools", goal="reason", model="fake",
        tools=[{"id": "file.read", "type": "file.read"},
               {"id": "http.fetch", "type": "http.fetch"}],
        limits=LIMITS,
    )
    specs = coordinator._normalise_agents([
        {"id": "a1", "task": "reason about the goal", "tools": ["none"]}])
    coordinator._build_members(specs)
    assert coordinator._members[0].tools == []


def test_an_explicit_no_tools_grant_is_not_overridden_by_the_task_text():
    coordinator = SwarmCoordinator(
        gateway=SwarmGateway(), swarm_id="swarm-no-tools-write", goal="write", model="fake",
        tools=[{"id": "file.read", "type": "file.read"},
               {"id": "file.write", "type": "file.write"}],
        allow_write=True, limits=LIMITS,
    )
    specs = coordinator._normalise_agents([
        {"id": "a1", "task": "write the report", "tools": ["none"],
         "writes": ["reports"]},
    ])
    coordinator._build_members(specs)
    assert coordinator._members[0].tools == []


def test_os_access_lets_a_writer_use_the_workspace_without_a_scope():
    coordinator = SwarmCoordinator(
        gateway=SwarmGateway(), swarm_id="swarm-os-write", goal="save notes", model="fake",
        tools=[{"id": "file.write", "type": "file.write"}],
        limits=LIMITS, allow_write=True, allow_os_access=True,
    )
    specs = coordinator._normalise_agents([
        {"id": "a1", "task": "write the notes", "tools": ["*"]}])
    coordinator._build_members(specs)
    member = coordinator._members[0]
    assert "file.write" in member.tools and member.read_only is False


def test_long_member_ids_still_share_reviews_on_the_blackboard():
    first, second = "a" * 48, "b" * 48
    limits = SwarmLimits(max_agents=2, max_steps=4, max_rounds=1, max_seconds=20,
                         max_messages=40, max_message_chars=200, max_memory_entries=20)
    coordinator = SwarmCoordinator(
        gateway=SwarmGateway(), swarm_id="swarm-long-ids", goal="understand x",
        model="fake", tools=[{"id": "file.read", "type": "file.read"}],
        executor=lambda member, tool_id, arguments: ({"content": "hello"}, True),
        limits=limits,
    )
    result = coordinator.run(
        agents=[{"id": first, "task": "read the file"},
                {"id": second, "task": "check the file"}],
        strategy="peer_review", rounds=1,
    )
    assert result["ok"] is True
    assert len(result["reviews"]) == 2
    review_keys = [entry["key"] for entry in result["blackboard"]
                   if entry["key"].startswith("review.")]
    assert len(review_keys) == 2
    assert all(len(key) <= 80 for key in review_keys)
    assert len(set(review_keys)) == 2


def test_blackboard_tracks_an_epoch_and_prioritises_recent_updates():
    board = SwarmBlackboard(max_entries=4, max_chars=200)
    assert board.epoch == 0
    board.write("a", "one", 1)
    board.write("b", "two", 2)
    board.write("a", "one", 3)
    assert board.epoch == 3
    assert [entry["key"] for entry in board.snapshot()] == ["two", "one"]
    assert board.delete("two") and board.epoch == 4
    assert not board.delete("missing") and board.epoch == 4


def test_blackboard_digest_can_skip_prefetched_results():
    board = SwarmBlackboard(max_entries=4, max_chars=200)
    board.write("a", "finding.x", "kept")
    board.write("a", "result.a1", "shown elsewhere")
    digest = board.digest(skip=("result.",))
    assert "finding.x" in digest and "result.a1" not in digest
    assert board.digest(skip=("finding.", "result.")) == "(no other shared notes)"


def test_member_observation_repeats_the_blackboard_only_when_it_changes():
    coordinator = SwarmCoordinator(
        gateway=SwarmGateway(), swarm_id="swarm-incremental", goal="understand x",
        model="fake", tools=[], limits=LIMITS,
    )
    specs = coordinator._normalise_agents([{"id": "a1", "task": "read the file"}])
    coordinator._build_members(specs)
    member = coordinator._members[0]
    coordinator._member_messages(member, "work", [])

    assert "Blackboard" not in coordinator._member_observation(member, "work")
    coordinator._blackboard.write("a1", "finding.new", {"ok": True})
    changed = coordinator._member_observation(member, "work")
    assert "finding.new" in changed and "Blackboard now" in changed
    assert "Blackboard" not in coordinator._member_observation(member, "work")


def test_member_prompt_does_not_repeat_shared_results_from_the_blackboard():
    coordinator = SwarmCoordinator(
        gateway=SwarmGateway(), swarm_id="swarm-no-repeat", goal="understand x",
        model="fake", tools=[], limits=LIMITS,
    )
    specs = coordinator._normalise_agents([
        {"id": "a1", "task": "read the file"},
        {"id": "a2", "task": "check the file"},
    ])
    coordinator._build_members(specs)
    first, second = coordinator._members
    first.output = "unique peer output"
    coordinator._publish_result(first)

    prompt = coordinator._member_messages(second, "work", [])
    text = "\n".join(item["content"] for item in prompt)
    assert "unique peer output" in text  # the peer digest
    assert "result.a1" not in text  # the blackboard digest skips automatic results


def test_read_only_tool_results_are_reused_and_writes_invalidate():
    calls = []
    events = []

    def executor(member, tool_id, arguments):
        calls.append(tool_id)
        if tool_id == "file.write":
            return {"written": True, "path": "x.txt"}, True
        return {"path": "x.txt", "content": "hello"}, True

    coordinator = SwarmCoordinator(
        gateway=SwarmGateway(), swarm_id="swarm-cache", goal="read x", model="fake",
        tools=[{"id": "file.read", "type": "file.read"},
               {"id": "file.write", "type": "file.write"}],
        executor=executor, allow_write=True,
        limits=SwarmLimits(max_agents=1, max_steps=4, max_rounds=0, max_seconds=20,
                           max_messages=40, max_message_chars=200, max_memory_entries=20),
        on_event=lambda kind, **fields: events.append((kind, fields)),
    )
    member = SwarmMember(id="a1", label="A1", role="reader", task="read",
                         tools=["file.read", "file.write"], writes=["x.txt"],
                         read_only=False, phase="work")
    coordinator._members = [member]
    coordinator._by_id = {"a1": member}

    read = {"tool": "file.read", "input": {"path": "x.txt"}}
    coordinator._member_tool(member, dict(read), {}, 1)
    coordinator._member_tool(member, dict(read), {}, 2)
    assert calls == ["file.read"]
    assert coordinator._cache_stats()["hits"] == 1
    assert any(fields.get("cached") for kind, fields in events
               if kind == "swarm_agent_tool")

    coordinator._member_tool(
        member, {"tool": "file.write", "input": {"path": "x.txt", "content": "new"}}, {}, 3)
    coordinator._member_tool(member, dict(read), {}, 4)
    assert calls == ["file.read", "file.write", "file.read"]


def test_identical_concurrent_reads_run_once():
    calls = []
    entered = threading.Event()
    release = threading.Event()

    def executor(member, tool_id, arguments):
        calls.append(tool_id)
        entered.set()
        release.wait(2)
        return {"content": "hello"}, True

    coordinator = SwarmCoordinator(
        gateway=SwarmGateway(), swarm_id="swarm-single-flight", goal="read", model="fake",
        tools=[{"id": "file.read", "type": "file.read"}], executor=executor,
        limits=SwarmLimits(max_agents=1, max_steps=4, max_rounds=0, max_seconds=20,
                           max_messages=40, max_message_chars=200, max_memory_entries=20),
    )
    member = SwarmMember(id="a1", label="A1", role="reader", task="read",
                         tools=["file.read"], phase="work")
    errors = []

    def run():
        try:
            coordinator._member_tool(
                member, {"tool": "file.read", "input": {"path": "x.txt"}}, {}, 1)
        except Exception as failure:  # pragma: no cover - reported below
            errors.append(failure)

    threads = [threading.Thread(target=run) for _ in range(2)]
    for thread in threads:
        thread.start()
    assert entered.wait(1)
    release.set()
    for thread in threads:
        thread.join(2)
    assert not errors and all(not thread.is_alive() for thread in threads)
    assert calls == ["file.read"]


def test_batched_read_tools_run_in_parallel_and_keep_their_order():
    barrier = threading.Barrier(2, timeout=3)

    def executor(member, tool_id, arguments):
        barrier.wait()
        return {"path": arguments["path"], "content": arguments["path"]}, True

    coordinator = SwarmCoordinator(
        gateway=SwarmGateway(), swarm_id="swarm-parallel", goal="read", model="fake",
        tools=[{"id": "file.read", "type": "file.read"}], executor=executor,
        limits=SwarmLimits(max_agents=1, max_steps=4, max_rounds=0, max_seconds=20,
                           max_messages=40, max_message_chars=200, max_memory_entries=20),
    )
    member = SwarmMember(id="a1", label="A1", role="reader", task="read",
                         tools=["file.read"], phase="work")
    final, results = coordinator._member_actions(member, "work", {"calls": [
        {"tool": "file.read", "input": {"path": "a.txt"}},
        {"tool": "file.read", "input": {"path": "b.txt"}},
    ]}, {}, 1)
    assert final is None
    assert len(results) == 2
    assert "a.txt" in results[0] and "b.txt" in results[1]


def test_member_system_prompt_is_stable_across_phases():
    coordinator = SwarmCoordinator(
        gateway=SwarmGateway(), swarm_id="swarm-stable", goal="understand x",
        model="fake", tools=[], limits=LIMITS,
    )
    member = SwarmMember(id="a1", label="A1", role="reader", task="read")
    coordinator._members = [member]
    coordinator._by_id = {"a1": member}

    first_system, first_user = coordinator._member_messages(member, "work", [])
    coordinator.round = 3
    member.plan = [{"text": "read x", "status": "active"}]
    coordinator._blackboard.write("a1", "finding.x", "value", note="note")
    second_system, second_user = coordinator._member_messages(member, "revise", [])

    assert first_system["content"] == second_system["content"]
    assert "Phase: work" in first_user["content"]
    assert "Phase: revise" in second_user["content"]
    assert "read x" in second_user["content"]


def test_prompt_cache_key_tracks_the_stable_system_prefix():
    gateway = SwarmGateway()
    coordinator = SwarmCoordinator(
        gateway=gateway, swarm_id="swarm-prefix", goal="answer", model="fake",
        tools=[], limits=LIMITS,
    )
    member = SwarmMember(id="a1", label="A1", role="reader", task="answer")
    base = [{"role": "system", "content": "stable instructions"},
            {"role": "user", "content": "one"}]
    coordinator._complete("fake", base, temperature=0, max_tokens=16, label="a1",
                          phase="work", member=member)
    coordinator._complete("fake", [base[0], {"role": "user", "content": "two"}],
                          temperature=0, max_tokens=16, label="a1", phase="work",
                          member=member)
    first, second = gateway.calls[-2], gateway.calls[-1]
    assert first["prompt_cache"] is True
    assert first["prompt_cache_key"] == second["prompt_cache_key"]
    coordinator._complete("fake", [{"role": "system", "content": "changed"},
                                    {"role": "user", "content": "one"}],
                          temperature=0, max_tokens=16, label="a1", phase="work",
                          member=member)
    assert gateway.calls[-1]["prompt_cache_key"] != first["prompt_cache_key"]


def test_complete_caller_supplied_agents_skip_the_leader_plan():
    def executor(member, tool_id, arguments):
        return {"content": "hello"}, True

    gateway = SwarmGateway()
    coordinator = SwarmCoordinator(
        gateway=gateway, swarm_id="swarm-fast-plan", goal="read x", model="fake",
        tools=[{"id": "file.read", "type": "file.read"}], executor=executor,
        limits=SwarmLimits(max_agents=1, max_steps=4, max_rounds=0, max_seconds=20,
                           max_messages=40, max_message_chars=200, max_memory_entries=20),
        criteria=["read the file"],
    )
    result = coordinator.run(
        agents=[{"id": "a1", "task": "read x", "tools": ["file.read"]}],
        strategy="map_reduce", rounds=0,
    )
    assert result["ok"] is True
    assert result["success_criteria"] == ["read the file"]
    assert not any("Plan a focused team" in call["messages"][0]["content"]
                   for call in gateway.calls)


def test_high_score_reviews_stop_without_a_leader_decision():
    gateway = SwarmGateway()
    coordinator = SwarmCoordinator(
        gateway=gateway, swarm_id="swarm-stop", goal="answer", model="fake",
        tools=[], limits=LIMITS,
    )
    coordinator._requested_rounds = 2
    coordinator._reviews = [{"round": 1, "reviewer": "a1", "target": "a2",
                             "score": 5, "issues": []}]
    assert coordinator._should_continue(1) is False
    assert gateway.calls == []
    assert any("deterministic" in warning for warning in coordinator.warnings)


def test_adaptive_effort_spends_reasoning_on_first_and_final_turns():
    coordinator = SwarmCoordinator(
        gateway=SwarmGateway(), swarm_id="swarm-effort", goal="answer", model="fake",
        tools=[], limits=LIMITS, effort="high",
    )
    assert coordinator._turn_effort("work", 1) == "high"
    assert coordinator._turn_effort("work", 2) == "low"
    assert coordinator._turn_effort("review", 1) == "low"
    coordinator.adaptive_effort = False
    assert coordinator._turn_effort("work", 2) == "high"
    coordinator.effort = ""
    assert coordinator._turn_effort("work", 2) is None


    assert snapshot_due("swarm_end", 0.0, now=1.0)
    assert snapshot_due("swarm_agent_end", 10.0, now=10.01)
    assert not snapshot_due("swarm_agent_step", 10.0, now=10.1)
    assert snapshot_due("swarm_agent_step", 10.0, now=10.3)


def test_synthesis_receives_the_callers_system_guidance():
    class FinalGateway(SwarmGateway):
        def chat(self, config):
            system = config["messages"][0]["content"]
            if "You are one member of the swarm" in system:
                self.calls.append(copy.deepcopy(config))
                return {"reply": json.dumps({"final": "member finding"}), "usage": {}}
            return super().chat(config)

    gateway = FinalGateway()
    coordinator = SwarmCoordinator(
        gateway=gateway, swarm_id="swarm-guidance", goal="answer", model="fake",
        system="Always answer in French.", tools=[], limits=LIMITS,
    )
    result = coordinator.run(agents=[{"id": "a1", "task": "answer"}],
                             strategy="map_reduce", rounds=0)
    assert result["ok"] is True
    synthesis = [call for call in gateway.calls
                 if "Synthesise the members' work" in call["messages"][0]["content"]]
    assert synthesis
    assert "Always answer in French." in synthesis[0]["messages"][0]["content"]


def test_context_pool_grants_only_while_space_remains():
    pool = SwarmContextPool(capacity=100, per_agent=80)
    assert pool.grant("a", 70) == 70
    assert pool.grant("b", 70) == 30  # only the run-wide remainder is free
    assert pool.free() == 0
    assert pool.grant("a", 10) == 0  # the pool, not just the agent cap, is exhausted
    assert pool.grant("c", 10) == 0
    assert pool.release("a") == 70
    assert pool.free() == 70
    assert pool.snapshot()["agents"] == {"b": 30}
    assert pool.grant("b", 60) == 50  # b may grow to its per-agent ceiling
    assert pool.used() == 80 and pool.free() == 20


def test_context_bank_evicts_to_stay_inside_its_capacity():
    bank = SwarmContextBank(max_chars=1_000, max_items=10)
    for index in range(20):
        bank.put("a", "tool", f"result {index}", "x" * 900, max_chars=5_000)
        assert bank.snapshot()["chars"] <= 1_000
    assert len(bank) <= 1
    handle = bank.put("a", "result", "small", "saved")
    assert bank.get(handle)["text"] == "saved"
    assert bank.handles(limit=1)[0]["handle"] == handle
    assert bank.search("saved")[0]["handle"] == handle


def test_agent_context_grants_are_capped_by_the_model_window():
    class WindowGateway(SwarmGateway):
        def context_window(self, model_id=None, force=False):
            return {"context_window": 8_000, "unit": "tokens", "found": True}

    limits = SwarmLimits(max_prompt_chars=10_000, max_context_chars=100_000,
                         max_agent_context_chars=50_000, max_context_item_chars=50_000)
    coordinator = SwarmCoordinator(
        gateway=WindowGateway(), swarm_id="swarm-window", goal="answer", model="fake",
        tools=[], limits=limits,
    )
    assert coordinator._context_window("fake") == 24_000
    assert coordinator._prompt_budget(None, "fake", 1_000) == 20_000  # room for output
    specs = coordinator._normalise_agents([
        {"id": "a1", "task": "read", "context_chars": 20_000}])
    coordinator._build_members(specs)
    member = coordinator._members[0]
    assert coordinator._context_pool.used("a1") == 20_000
    assert coordinator._prompt_budget(member, "fake", 1_000) == 20_000  # window wins


def test_context_requests_are_refused_when_the_pool_is_disabled():
    coordinator = SwarmCoordinator(
        gateway=SwarmGateway(), swarm_id="swarm-off", goal="answer", model="fake",
        tools=[], limits=SwarmLimits(max_prompt_chars=1_000, max_context_chars=0,
                                     max_agent_context_chars=0),
    )
    specs = coordinator._normalise_agents([{"id": "a1", "task": "read"}])
    coordinator._build_members(specs)
    member = coordinator._members[0]
    coordinator._blackboard.write("a1", "finding.one", "short")
    answer = coordinator._member_context(
        member, {"action": "get", "key": "finding.one", "chars": 1_000})
    assert "disabled" in answer
    assert "Dynamic context is disabled" in coordinator._context_lines(member)


def test_a_member_reads_a_long_blackboard_value_through_the_pool():
    limits = SwarmLimits(max_prompt_chars=1_000, max_observation_chars=100,
                         max_memory_chars=50_000, max_context_chars=5_000,
                         max_agent_context_chars=5_000, max_context_item_chars=5_000)
    coordinator = SwarmCoordinator(
        gateway=SwarmGateway(), swarm_id="swarm-read", goal="understand x",
        model="fake", tools=[], limits=limits,
    )
    specs = coordinator._normalise_agents([{"id": "a1", "task": "read the file"}])
    coordinator._build_members(specs)
    member = coordinator._members[0]
    value = "START " + "x" * 3_000 + " FINISH"
    coordinator._blackboard.write("a1", "finding.large", value)

    first = coordinator._member_context(
        member, {"action": "get", "key": "finding.large", "chars": 2_000})
    assert "START" in first and "FINISH" not in first
    assert "characters remain" in first and "offset 2000" in first
    assert coordinator._context_pool.used("a1") == 2_000

    tail = coordinator._member_context(
        member, {"action": "get", "key": "finding.large", "chars": 2_000, "offset": 2_000})
    assert "FINISH" in tail
    assert member.context_granted == len(value)  # only the remaining slice was reserved


def test_a_context_request_is_denied_with_an_actionable_report():
    limits = SwarmLimits(max_prompt_chars=1_000, max_context_chars=1_000,
                         max_agent_context_chars=1_000, max_context_item_chars=1_000,
                         max_memory_chars=10_000)
    coordinator = SwarmCoordinator(
        gateway=SwarmGateway(), swarm_id="swarm-deny", goal="answer", model="fake",
        tools=[], limits=limits,
    )
    specs = coordinator._normalise_agents([{"id": "a1", "task": "read"}])
    coordinator._build_members(specs)
    member = coordinator._members[0]
    coordinator._blackboard.write("a1", "finding.large", "y" * 4_000)

    granted = coordinator._member_context(
        member, {"action": "get", "key": "finding.large", "chars": 1_000})
    assert "granted 1,000" in granted
    denied = coordinator._member_context(
        member, {"action": "get", "key": "finding.large", "chars": 1_000})
    assert "not granted" in denied
    assert "1,000/1,000" in denied
    assert "release" in denied.lower()


def test_a_long_tool_result_becomes_a_shared_handle():
    limits = SwarmLimits(max_prompt_chars=1_000, max_observation_chars=200,
                         max_context_chars=20_000, max_agent_context_chars=20_000,
                         max_context_item_chars=8_000)
    coordinator = SwarmCoordinator(
        gateway=SwarmGateway(), swarm_id="swarm-handle", goal="read", model="fake",
        tools=[{"id": "file.read", "type": "file.read"}],
        executor=lambda member, tool_id, arguments: ({"content": "A" * 4_000 + "TAIL"}, True),
        limits=limits,
    )
    specs = coordinator._normalise_agents([{"id": "a1", "task": "read the file"}])
    coordinator._build_members(specs)
    member = coordinator._members[0]

    observation = coordinator._member_tool(member, {"tool": "file.read", "input": {}}, {}, 1)
    assert "shortened" in observation
    handles = CONTEXT_HANDLE.findall(observation)
    assert handles
    stored = coordinator._context_bank.get(handles[0])
    assert stored is not None and "TAIL" in stored["text"]

    expanded = coordinator._member_context(
        member, {"action": "get", "handle": handles[0], "chars": 5_000})
    assert "TAIL" in expanded and "end of this context" in expanded


def test_a_member_can_read_a_peers_full_result():
    limits = SwarmLimits(max_prompt_chars=1_000, max_context_chars=20_000,
                         max_agent_context_chars=20_000, max_context_item_chars=8_000)
    coordinator = SwarmCoordinator(
        gateway=SwarmGateway(), swarm_id="swarm-peer", goal="answer", model="fake",
        tools=[], limits=limits,
    )
    specs = coordinator._normalise_agents([
        {"id": "a1", "task": "first"}, {"id": "a2", "task": "second"}])
    coordinator._build_members(specs)
    first, second = coordinator._members
    first.output = "PEER-START " + "z" * 4_000 + " PEER-END"
    first.result_handle = coordinator._context_bank.put(
        first.id, "result", "first result", first.output)

    text = coordinator._member_context(
        second, {"action": "get", "peer": "a1", "chars": 1_500})
    assert "PEER-START" in text and "PEER-END" not in text
    assert first.result_handle in text  # the label names the full-text handle

    text = coordinator._member_context(
        second, {"action": "get", "peer": "a1", "chars": 1_500, "offset": 4_000})
    assert "PEER-END" in text


def test_context_find_ranks_blackboard_and_peer_results():
    coordinator = SwarmCoordinator(
        gateway=SwarmGateway(), swarm_id="swarm-find", goal="answer", model="fake",
        tools=[], limits=SwarmLimits(max_prompt_chars=1_000),
    )
    specs = coordinator._normalise_agents([
        {"id": "a1", "task": "first"}, {"id": "a2", "task": "second"}])
    coordinator._build_members(specs)
    first, second = coordinator._members
    coordinator._blackboard.write("a1", "finding.retry", "the retry helper lives in backoff.py")
    first.output = "I traced the timeout to the retry helper in backoff.py."

    found = coordinator._member_context(
        second, {"action": "find", "query": "retry helper"})
    assert "finding.retry" in found and "peer a1" in found
    assert "context get" in found

    empty = coordinator._member_context(
        second, {"action": "find", "query": "nothing-matches-this"})
    assert "No shared context matches" in empty


def test_compaction_keeps_a_retrievable_ledger():
    limits = SwarmLimits(max_prompt_chars=1_500, max_context_chars=30_000,
                         max_agent_context_chars=30_000, max_context_item_chars=10_000)
    coordinator = SwarmCoordinator(
        gateway=SwarmGateway(), swarm_id="swarm-compact", goal="answer", model="fake",
        tools=[], limits=limits,
    )
    specs = coordinator._normalise_agents([{"id": "a1", "task": "read"}])
    coordinator._build_members(specs)
    member = coordinator._members[0]
    messages = [{"role": "system", "content": "system prompt"},
                {"role": "user", "content": "do the long task"}]
    for index in range(8):
        messages.append({"role": "assistant",
                         "content": json.dumps({"tool": "file.read",
                                                "input": {"path": f"file-{index}.py"}})})
        messages.append({"role": "user",
                         "content": f"result {index} " + "z" * 400 + " ctx-abcdef12"})

    coordinator._compact_messages(messages, member, max_tokens=500)
    assert messages[0]["content"] == "system prompt"
    assert messages[1]["content"] == "do the long task"
    assert messages[2]["content"].startswith(COMPACTED_PREFIX)
    assert len(messages) < 18

    ledger = messages[2]["content"]
    handles = CONTEXT_HANDLE.findall(ledger)
    assert handles
    history = [coordinator._context_bank.get(handle) for handle in handles]
    history = [item for item in history if item and item["kind"] == "history"]
    assert history and "result 0" in history[0]["text"]


def test_a_context_grant_is_released_when_the_phase_finishes():
    coordinator = SwarmCoordinator(
        gateway=SwarmGateway(), swarm_id="swarm-release", goal="answer", model="fake",
        tools=[], limits=SwarmLimits(max_prompt_chars=1_000,
                                     max_context_chars=12_000,
                                     max_agent_context_chars=12_000,
                                     max_context_item_chars=8_000),
    )
    specs = coordinator._normalise_agents([
        {"id": "a1", "task": "read", "context_chars": 5_000}])
    coordinator._build_members(specs)
    member = coordinator._members[0]
    assert coordinator._context_pool.used("a1") == 5_000

    coordinator._finish_member(member, "findings", "work", True)
    assert coordinator._context_pool.used("a1") == 0
    assert member.context_granted == 0
    assert member.result_handle
    assert coordinator._context_bank.get(member.result_handle)["text"] == "findings"


def test_planned_context_is_renewed_for_a_later_phase():
    coordinator = SwarmCoordinator(
        gateway=SwarmGateway(), swarm_id="swarm-renew", goal="answer", model="fake",
        tools=[], limits=SwarmLimits(max_prompt_chars=1_000,
                                     max_context_chars=12_000,
                                     max_agent_context_chars=12_000,
                                     max_context_item_chars=8_000),
    )
    specs = coordinator._normalise_agents([
        {"id": "a1", "task": "read", "context_chars": 5_000}])
    coordinator._build_members(specs)
    member = coordinator._members[0]
    assert coordinator._context_pool.used("a1") == 5_000

    coordinator._finish_member(member, "first findings", "work", True)
    assert coordinator._context_pool.used("a1") == 0

    coordinator._renew_context(member, "revise")
    assert coordinator._context_pool.used("a1") == 5_000
    assert member.context_granted == 5_000


def test_context_protocol_is_wired_into_member_actions():
    coordinator = SwarmCoordinator(
        gateway=SwarmGateway(), swarm_id="swarm-action", goal="answer", model="fake",
        tools=[], limits=SwarmLimits(max_prompt_chars=1_000),
    )
    specs = coordinator._normalise_agents([{"id": "a1", "task": "read"}])
    coordinator._build_members(specs)
    member = coordinator._members[0]

    operations = coordinator._normalise_actions({"context": {"action": "status"}}, "work")
    assert [operation["op"] for operation in operations] == ["context"]
    final, results = coordinator._member_actions(
        member, "work", {"context": {"action": "status"}}, {}, 1)
    assert final is None
    assert "granted_chars" in results[0]


def test_a_member_requests_context_mid_run_and_uses_it():
    class ContextGateway(SwarmGateway):
        def chat(self, config):
            system = config["messages"][0]["content"]
            user = config["messages"][-1]["content"]
            if "Plan a focused team" in system:
                return {"reply": json.dumps({
                    "strategy": "map_reduce", "rounds": 0,
                    "agents": [{"id": "a1", "task": "read the big finding"}],
                }), "usage": {}}
            if "Synthesise the members' work" in system:
                return {"reply": json.dumps({"final": "merged with long context"}),
                        "usage": {}}
            if "You are one member of the swarm" in system:
                if "Context from blackboard finding.big" in user:
                    return {"reply": json.dumps({"final": "used the long context"}),
                            "usage": {}}
                return {"reply": json.dumps({
                    "context": {"action": "get", "key": "finding.big", "chars": 2_000}
                }), "usage": {}}
            return super().chat(config)

    events = []
    limits = SwarmLimits(max_agents=1, max_steps=3, max_rounds=0, max_seconds=20,
                         max_messages=20, max_message_chars=200, max_memory_entries=10,
                         max_memory_chars=50_000, max_prompt_chars=2_000,
                         max_observation_chars=100, max_context_chars=20_000,
                         max_agent_context_chars=20_000, max_context_item_chars=10_000)
    coordinator = SwarmCoordinator(
        gateway=ContextGateway(), swarm_id="swarm-context-run", goal="read the finding",
        model="fake", tools=[], limits=limits,
        on_event=lambda kind, **fields: events.append((kind, fields)),
    )
    coordinator._blackboard.write("leader", "finding.big", "START " + "x" * 4_000 + " TAIL")
    result = coordinator.run(agents=[{"id": "a1", "task": "read the big finding"}],
                             strategy="map_reduce", rounds=0)

    assert result["ok"] is True, result.get("error")
    assert result["agents"][0]["output"] == "used the long context"
    granted = [fields for kind, fields in events
               if kind == "swarm_context" and fields.get("action") == "get"]
    assert granted and granted[0]["granted"] == 2_000
    assert granted[0]["total"] > 2_000


def test_a_step_budget_end_asks_for_a_final_answer_without_tools():
    class ToolLoopGateway(SwarmGateway):
        def __init__(self):
            super().__init__()
            self.wrap_calls = 0

        def chat(self, config):
            system = config["messages"][0]["content"]
            user = config["messages"][-1]["content"]
            if "Plan a focused team" in system:
                return {"reply": json.dumps({
                    "strategy": "map_reduce", "rounds": 0,
                    "agents": [{"id": "a1", "task": "read the file"}],
                }), "usage": {}}
            if "Synthesise the members' work" in system:
                return {"reply": json.dumps({"final": "merged answer"}), "usage": {}}
            if "You are one member of the swarm" in system:
                if "Tool budget reached" in user:
                    self.wrap_calls += 1
                    return {"reply": json.dumps({"final": "answer from collected results"}),
                            "usage": {}}
                return {"reply": json.dumps({
                    "tool": "file.read", "input": {"path": "x.txt"}}), "usage": {}}
            return super().chat(config)

    gateway = ToolLoopGateway()
    limits = SwarmLimits(max_agents=1, max_steps=3, max_rounds=0, max_seconds=20,
                         max_messages=40, max_message_chars=200, max_memory_entries=20)
    coordinator = SwarmCoordinator(
        gateway=gateway, swarm_id="swarm-wrap", goal="read the file", model="fake",
        tools=[{"id": "file.read", "type": "file.read"}],
        executor=lambda member, tool_id, arguments: ({"content": "FILE DATA"}, True),
        limits=limits,
    )
    result = coordinator.run(agents=[{"id": "a1", "task": "read the file"}],
                             strategy="map_reduce", rounds=0)

    assert result["ok"] is True, result.get("error")
    assert gateway.wrap_calls == 1
    assert result["agents"][0]["status"] == "done"
    assert result["agents"][0]["output"] == "answer from collected results"
    assert any("tool budget" in warning.lower() for warning in result["warnings"])


def test_a_working_member_gets_one_bounded_step_extension():
    class ResearchGateway(SwarmGateway):
        def __init__(self):
            super().__init__()
            self.member_calls = 0
            self.wrap_calls = 0

        def chat(self, config):
            system = config["messages"][0]["content"]
            user = config["messages"][-1]["content"]
            if "Plan a focused team" in system:
                return {"reply": json.dumps({
                    "strategy": "map_reduce", "rounds": 0,
                    "agents": [{"id": "a1", "task": "research the topic"}],
                }), "usage": {}}
            if "Synthesise the members' work" in system:
                return {"reply": json.dumps({"final": "merged answer"}), "usage": {}}
            if "You are one member of the swarm" in system:
                if "Tool budget reached" in user:
                    self.wrap_calls += 1
                    return {"reply": json.dumps({"final": "wrapped too early"}), "usage": {}}
                self.member_calls += 1
                if self.member_calls >= 4:
                    return {"reply": json.dumps({"final": "finished after research"}),
                            "usage": {}}
                return {"reply": json.dumps({
                    "tool": "file.read", "input": {"path": "notes.md"}}), "usage": {}}
            return super().chat(config)

    gateway = ResearchGateway()
    limits = SwarmLimits(max_agents=1, max_steps=2, max_rounds=0, max_seconds=20,
                         max_messages=40, max_message_chars=200, max_memory_entries=20)
    coordinator = SwarmCoordinator(
        gateway=gateway, swarm_id="swarm-extend", goal="research the topic", model="fake",
        tools=[{"id": "file.read", "type": "file.read"}],
        executor=lambda member, tool_id, arguments: ({"content": "MORE DATA"}, True),
        limits=limits,
    )
    result = coordinator.run(agents=[{"id": "a1", "task": "research the topic"}],
                             strategy="map_reduce", rounds=0)

    assert result["ok"] is True, result.get("error")
    assert gateway.member_calls == 4 and gateway.wrap_calls == 0
    assert result["agents"][0]["output"] == "finished after research"
    assert result["agents"][0]["steps"] == 4
    assert any("raised to 4 rounds" in warning for warning in result["warnings"])


def test_the_leader_reads_files_written_by_the_swarm():
    calls = []

    def executor(member, tool_id, arguments):
        calls.append((member["id"], tool_id, arguments["path"]))
        return {"path": arguments["path"], "content": "TOP 3 CONTENT"}, True

    coordinator = SwarmCoordinator(
        gateway=SwarmGateway(), swarm_id="swarm-files", goal="rank the news", model="fake",
        tools=[{"id": "file.read", "type": "file.read"}], executor=executor,
        limits=LIMITS,
    )
    coordinator.changed_files = ["blackboard/final-top3.md"]
    text = coordinator._leader_file_sections()
    assert "TOP 3 CONTENT" in text and "blackboard/final-top3.md" in text
    assert calls == [("leader", "file.read", "blackboard/final-top3.md")]


def test_the_run_result_and_snapshot_report_context_usage():
    limits = SwarmLimits(max_agents=2, max_steps=4, max_rounds=0, max_seconds=20,
                         max_messages=40, max_message_chars=200, max_memory_entries=20,
                         max_context_chars=40_000, max_agent_context_chars=20_000,
                         max_context_item_chars=10_000)
    coordinator = SwarmCoordinator(
        gateway=SwarmGateway(), swarm_id="swarm-report", goal="understand x",
        model="fake", tools=[{"id": "file.read", "type": "file.read"}],
        executor=lambda member, tool_id, arguments: ({"content": "hello"}, True),
        limits=limits,
    )
    result = coordinator.run(agents=[{"id": "a1", "task": "read the file"}],
                             strategy="map_reduce", rounds=0)
    assert result["ok"] is True
    assert result["context"]["capacity"] == 40_000
    assert result["context"]["per_agent"] == 20_000
    assert result["context_items"] >= 1
    snapshot = coordinator.snapshot()
    assert snapshot["context"]["capacity"] == 40_000
    assert snapshot["context_items"] >= 1


def test_peer_digest_ranks_results_against_the_members_task():
    coordinator = SwarmCoordinator(
        gateway=SwarmGateway(), swarm_id="swarm-peer-rank", goal="fix the bug",
        model="fake", tools=[], limits=SwarmLimits(max_prompt_chars=1_000),
    )
    specs = coordinator._normalise_agents([
        {"id": "a1", "task": "trace the retry helper"},
        {"id": "a2", "task": "check styling"},
        {"id": "a3", "task": "check the retry backoff"},
        {"id": "a4", "task": "check docs"},
        {"id": "a5", "task": "check tests"},
    ])
    coordinator._build_members(specs)
    viewer = coordinator._members[0]
    coordinator._members[1].output = "CSS variables look consistent."
    retry = coordinator._members[2]
    retry.output = "Retry helper lives in backoff.py and retries forever."
    coordinator._members[3].output = "README describes setup."
    coordinator._members[4].output = "Suite covers parser."

    digest = coordinator._peer_digest(viewer, limit=2, chars=60)
    lines = digest.splitlines()
    assert len(lines) == 2
    assert "a3" in lines[0] and "retry" in lines[0].lower()

    coordinator._members[4].output = retry.output
    deduped = coordinator._peer_digest(viewer, limit=4, chars=80)
    assert deduped.count("Retry helper lives") == 1


def test_a_member_recovers_from_an_empty_final():
    class EmptyFinalGateway(SwarmGateway):
        def __init__(self):
            super().__init__()
            self.empty = False

        def chat(self, config):
            system = config["messages"][0]["content"]
            if "You are one member of the swarm" in system:
                if not self.empty:
                    self.empty = True
                    return {"reply": '{"final": ""}', "usage": {}}
                return {"reply": '{"final": "recovered findings"}', "usage": {}}
            return super().chat(config)

    events = []
    coordinator = SwarmCoordinator(
        gateway=EmptyFinalGateway(), swarm_id="swarm-empty-final", goal="answer",
        model="fake", tools=[], limits=LIMITS,
        on_event=lambda kind, **fields: events.append((kind, fields)),
    )
    result = coordinator.run(agents=[{"id": "a1", "task": "answer"}],
                             strategy="map_reduce", rounds=0)

    assert result["ok"] is True, result.get("error")
    assert result["agents"][0]["output"] == "recovered findings"
    assert any(kind == "swarm_agent_status" and "empty final" in fields.get("message", "")
               for kind, fields in events)


def test_a_member_recovers_from_an_empty_reply():
    class EmptyReplyGateway(SwarmGateway):
        def __init__(self):
            super().__init__()
            self.empty = False

        def chat(self, config):
            system = config["messages"][0]["content"]
            if "You are one member of the swarm" in system:
                self.calls.append(copy.deepcopy(config))
                if not self.empty:
                    self.empty = True
                    return {"reply": "", "usage": {}}
                return {"reply": '{"final": "recovered findings"}', "usage": {}}
            return super().chat(config)

    gateway = EmptyReplyGateway()
    events = []
    coordinator = SwarmCoordinator(
        gateway=gateway, swarm_id="swarm-empty-reply", goal="answer",
        model="fake", tools=[], limits=LIMITS,
        on_event=lambda kind, **fields: events.append((kind, fields)),
    )
    result = coordinator.run(agents=[{"id": "a1", "task": "answer"}],
                             strategy="map_reduce", rounds=0)

    assert result["ok"] is True, result.get("error")
    assert result["agents"][0]["output"] == "recovered findings"
    assert any(kind == "swarm_agent_status" and "empty reply" in fields.get("message", "")
               for kind, fields in events)
    # The correction turn must not send an empty assistant message: strict
    # backends reject it, which would turn one bad reply into a failed run.
    assert gateway.calls
    for call in gateway.calls:
        assert all(message["content"] for message in call["messages"]
                   if message["role"] == "assistant")


def test_a_retried_swarm_turn_keeps_native_tools():
    class RetryGateway:
        def __init__(self):
            self.calls = []

        def chat(self, config):
            self.calls.append(copy.deepcopy(config))
            if len(self.calls) == 1:
                return {"reply": "", "finish_reason": "length",
                        "usage": {"completion_tokens": 1}}
            return {"reply": '{"final": "ok"}', "usage": {}}

    gateway = RetryGateway()
    coordinator = SwarmCoordinator(
        gateway=gateway, swarm_id="swarm-tools-retry", goal="answer", model="fake",
        tools=[{"id": "file.read", "type": "file.read"}],
        limits=SwarmLimits(max_seconds=20),
    )
    reply, _usage = coordinator._complete(
        "fake", [{"role": "user", "content": "read x"}], temperature=0.2,
        max_tokens=100, label="a1", phase="work", with_tools=True,
    )
    assert reply == '{"final": "ok"}'
    assert len(gateway.calls) == 2
    assert gateway.calls[0]["tools"] == gateway.calls[1]["tools"]
    assert [tool["id"] for tool in gateway.calls[1]["tools"]] == ["file.read"]


def test_a_context_offset_past_the_end_is_reported_without_charging_the_pool():
    limits = SwarmLimits(max_prompt_chars=1_000, max_context_chars=5_000,
                         max_agent_context_chars=5_000, max_context_item_chars=1_000,
                         max_memory_chars=10_000)
    coordinator = SwarmCoordinator(
        gateway=SwarmGateway(), swarm_id="swarm-offset", goal="answer", model="fake",
        tools=[], limits=limits,
    )
    specs = coordinator._normalise_agents([{"id": "a1", "task": "read"}])
    coordinator._build_members(specs)
    member = coordinator._members[0]
    coordinator._blackboard.write("a1", "finding.short", "abcdef")

    answer = coordinator._member_context(
        member, {"action": "get", "key": "finding.short", "offset": 999})
    assert "past the end" in answer and "6 characters" in answer
    assert coordinator._context_pool.used("a1") == 0


def test_review_prompt_shows_the_target_output_once():
    coordinator = SwarmCoordinator(
        gateway=SwarmGateway(), swarm_id="swarm-review-once", goal="answer", model="fake",
        tools=[], limits=SwarmLimits(max_prompt_chars=2_000),
    )
    specs = coordinator._normalise_agents([
        {"id": "a1", "task": "first"},
        {"id": "a2", "task": "second"},
        {"id": "a3", "task": "third"},
    ])
    coordinator._build_members(specs)
    target, reviewer = coordinator._members[0], coordinator._members[1]
    target.output = "UNIQUE TARGET OUTPUT"

    messages = coordinator._review_messages(
        reviewer, target, {"focus": "accuracy"}, 1)
    combined = "\n".join(str(item["content"]) for item in messages)
    assert combined.count("UNIQUE TARGET OUTPUT") == 1


def test_a_review_keeps_the_members_work_summary():
    coordinator = SwarmCoordinator(
        gateway=SwarmGateway(), swarm_id="swarm-review-summary", goal="answer",
        model="fake", tools=[], limits=SwarmLimits(max_prompt_chars=2_000),
    )
    specs = coordinator._normalise_agents([
        {"id": "a1", "task": "first"},
        {"id": "a2", "task": "second"},
    ])
    coordinator._build_members(specs)
    first, second = coordinator._members
    first.output = "the first result"
    second.output = "the second result"
    second.summary = "second work summary"

    coordinator._finish_member(second, "review text", "review", True)
    assert second.summary == "second work summary"

    coordinator._run_member_review_only(second, first, {"focus": "accuracy"}, 1)
    assert second.summary == "second work summary"


def test_pipeline_prompt_does_not_repeat_a_previous_stage():
    coordinator = SwarmCoordinator(
        gateway=SwarmGateway(), swarm_id="swarm-pipeline-once", goal="answer", model="fake",
        tools=[], limits=SwarmLimits(max_prompt_chars=2_000),
    )
    specs = coordinator._normalise_agents([
        {"id": "a1", "task": "first stage"},
        {"id": "a2", "task": "second stage"},
    ])
    coordinator._build_members(specs)
    coordinator.strategy = "pipeline"
    first, second = coordinator._members
    first.output = "STAGE-ONE-OUTPUT"

    system, user = coordinator._member_messages(second, "work", [])
    assert "STAGE-ONE-OUTPUT" in user["content"]
    assert "STAGE-ONE-OUTPUT" not in system["content"]


def test_run_starts_a_fresh_time_budget_and_accepts_string_usage():
    class FinalGateway:
        def chat(self, config):
            return {"reply": '{"final": "ok"}',
                    "usage": {"prompt_tokens": "2", "completion_tokens": 1.0}}

    coordinator = SwarmCoordinator(
        gateway=FinalGateway(), swarm_id="swarm-deadline", goal="answer", model="fake",
        tools=[], limits=SwarmLimits(max_agents=1, max_steps=2, max_rounds=0,
                                     max_seconds=20, max_reviews=0),
        dry_run=True,
    )
    coordinator.deadline = time.monotonic() - 60  # created long before start
    result = coordinator.run(agents=[{"id": "a1", "task": "answer"}],
                             strategy="map_reduce", rounds=0)

    assert result["ok"] is True
    assert result["agents"][0]["output"] == "ok"
    assert result["agents"][0]["usage"]["prompt_tokens"] == 2


def test_native_tools_never_exceed_the_member_grant():
    class CaptureGateway:
        def __init__(self):
            self.calls = []

        def chat(self, config):
            self.calls.append(config)
            return {"reply": '{"final": "ok"}', "usage": {}}

    gateway = CaptureGateway()
    coordinator = SwarmCoordinator(
        gateway=gateway, swarm_id="swarm-tools", goal="answer", model="fake",
        tools=[{"id": "file.read", "type": "file.read"},
               {"id": "file.write", "type": "file.write"}],
        allow_write=True, limits=SwarmLimits(max_agents=1, max_steps=2, max_rounds=0),
    )
    member = SwarmMember(id="a1", label="A", role="researcher", task="answer",
                         tools=["file.read"])
    messages = [{"role": "user", "content": "go"}]

    coordinator._complete("fake", messages, temperature=0, max_tokens=10, label="a1",
                          phase="work", member=member, with_tools=True)
    assert [tool["id"] for tool in gateway.calls[-1]["tools"]] == ["file.read"]

    member.tools = ["file.read", "file.write"]
    coordinator._complete("fake", messages, temperature=0, max_tokens=10, label="a1",
                          phase="review", member=member, with_tools=True)
    assert [tool["id"] for tool in gateway.calls[-1]["tools"]] == ["file.read"]

    member.tools = []
    coordinator._complete("fake", messages, temperature=0, max_tokens=10, label="a1",
                          phase="work", member=member, with_tools=True)
    assert "tools" not in gateway.calls[-1]


def test_a_large_streamed_burst_keeps_every_character():
    class BurstGateway:
        def stream(self, config, cancel):
            yield {"type": "delta", "text": "x" * 10_000}
            yield {"type": "done", "reply": '{"final": "ok"}', "usage": {}}

    events = []
    coordinator = SwarmCoordinator(
        gateway=BurstGateway(), swarm_id="swarm-burst", goal="answer", model="fake",
        tools=[], limits=SwarmLimits(max_agents=1, max_steps=1, max_rounds=0),
        on_event=lambda kind, **fields: events.append((kind, fields)),
    )
    member = SwarmMember(id="a1", label="A", role="researcher", task="answer")
    coordinator._complete("fake", [{"role": "user", "content": "go"}],
                          temperature=0, max_tokens=20, label="a1", phase="work",
                          member=member)
    streamed = "".join(fields.get("text", "") for kind, fields in events
                       if kind == "swarm_agent_delta")
    assert streamed == "x" * 10_000


def test_absolute_or_escaping_write_scopes_are_rejected():
    coordinator = SwarmCoordinator(
        gateway=None, swarm_id="swarm-scope", goal="answer", model="fake",
        allow_write=True, limits=SwarmLimits(max_agents=2),
    )
    assert coordinator._clean_scope("/etc/passwd") == ""
    assert coordinator._clean_scope("C:\\secret") == ""
    assert coordinator._clean_scope("src/../secret") == ""
    assert coordinator._clean_scope("./src/app.py") == "src/app.py"
    assert coordinator._clean_scope("*") == "*"
    specs = coordinator._normalise_agents([
        {"id": "a1", "task": "write a file", "writes": ["/etc/passwd"]},
    ])
    assert specs[0]["writes"] == []
    assert any("outside the workspace" in warning for warning in coordinator.warnings)


def test_compaction_keeps_earlier_context_handles():
    coordinator = SwarmCoordinator(
        gateway=None, swarm_id="swarm-compact", goal="answer", model="fake",
        limits=SwarmLimits(max_agents=1, max_prompt_chars=600,
                           max_context_chars=20_000, max_agent_context_chars=5_000),
    )
    member = SwarmMember(id="a1", label="A", role="researcher", task="answer")
    messages = [{"role": "system", "content": "system prompt"},
                {"role": "user", "content": "task"}]
    for _ in range(10):
        messages += [{"role": "assistant", "content": "a" * 120},
                     {"role": "user", "content": "u" * 120}]
    coordinator._compact_messages(messages, member, max_tokens=500)
    ledger = next(item for item in messages
                  if str(item["content"]).startswith(COMPACTED_PREFIX))
    first_handles = CONTEXT_HANDLE.findall(ledger["content"])
    assert first_handles

    for _ in range(10):
        messages += [{"role": "assistant", "content": "b" * 120},
                     {"role": "user", "content": "v" * 120}]
    coordinator._compact_messages(messages, member, max_tokens=500)
    ledger = next(item for item in messages
                  if str(item["content"]).startswith(COMPACTED_PREFIX))
    assert set(first_handles) <= set(CONTEXT_HANDLE.findall(ledger["content"]))


def test_repeated_empty_replies_recover_through_the_wrap_up():
    class EmptyGateway:
        def chat(self, config):
            if "Tool budget reached" in str(config["messages"][-1].get("content") or ""):
                return {"reply": '{"final": "recovered"}', "usage": {}}
            return {"reply": "", "usage": {}}

    coordinator = SwarmCoordinator(
        gateway=EmptyGateway(), swarm_id="swarm-empty-reply", goal="answer", model="fake",
        tools=[], limits=SwarmLimits(max_agents=1, max_steps=3, max_rounds=0,
                                     max_seconds=20, max_reviews=0),
        dry_run=True,
    )
    result = coordinator.run(agents=[{"id": "a1", "task": "answer"}],
                             strategy="map_reduce", rounds=0)

    assert result["ok"] is True
    assert result["agents"][0]["status"] == "done"
    assert result["agents"][0]["output"] == "recovered"
    assert any("repeated empty replies" in warning for warning in result["warnings"])


def test_empty_replies_never_become_fake_findings():
    class AlwaysEmpty:
        def chat(self, config):
            return {"reply": "", "usage": {}}

    coordinator = SwarmCoordinator(
        gateway=AlwaysEmpty(), swarm_id="swarm-empty-fallback", goal="answer", model="fake",
        tools=[], limits=SwarmLimits(max_agents=1, max_steps=2, max_rounds=0,
                                     max_seconds=20, max_reviews=0),
        dry_run=True,
    )
    result = coordinator.run(agents=[{"id": "a1", "task": "answer"}],
                             strategy="map_reduce", rounds=0)
    agent = result["agents"][0]

    assert result["ok"] is False
    assert agent["status"] == "failed"
    assert "Your reply was empty" not in (agent["output"] or "")
    assert "Your final answer was empty" not in (agent["output"] or "")
    assert "No final answer" in (agent["output"] or "")


def test_member_ids_cannot_shadow_channel_participants():
    coordinator = SwarmCoordinator(
        gateway=None, swarm_id="swarm-reserved", goal="answer", model="fake",
        limits=SwarmLimits(max_agents=6),
    )
    specs = coordinator._normalise_agents([
        {"id": "leader", "task": "one"},
        {"id": "all", "task": "two"},
        {"id": "operator", "task": "three"},
    ])
    ids = [spec["id"] for spec in specs]
    assert len(ids) == 3 and len(set(ids)) == 3
    assert not set(ids) & RESERVED_MEMBER_IDS
    assert sum("reserved" in warning for warning in coordinator.warnings) == 3


def test_a_context_read_without_an_action_defaults_to_get():
    coordinator = SwarmCoordinator(
        gateway=None, swarm_id="swarm-context-default", goal="answer", model="fake",
        limits=SwarmLimits(max_agents=1),
    )
    member = SwarmMember(id="a1", label="A", role="researcher", task="answer")
    handle = coordinator._context_bank.put(member.id, "tool", "stored result", "hello world")
    reply = coordinator._member_context(member, {"op": "context", "handle": handle})
    assert "hello world" in reply
    assert "not supported" not in reply


def test_malformed_tool_parameters_do_not_break_the_prompt():
    coordinator = SwarmCoordinator(
        gateway=None, swarm_id="swarm-tool-params", goal="answer", model="fake",
        tools=[{"id": "file.read", "type": "file.read",
                "params": ["not-a-field", {"key": "path"}]}],
        limits=SwarmLimits(max_agents=1),
    )
    member = SwarmMember(id="a1", label="A", role="researcher", task="answer",
                         tools=["file.read"])
    assert "Input keys: path" in coordinator._tool_lines(member)
    specs = coordinator._normalise_agents([
        {"id": "a2", "task": "read", "tools": [{"id": "file.read"}]},
    ])
    assert specs[0]["tools"] == ["file.read"]


def test_a_revision_prompt_includes_the_previous_answer():
    coordinator = SwarmCoordinator(
        gateway=None, swarm_id="swarm-revise-prompt", goal="answer", model="fake",
        limits=SwarmLimits(max_agents=1),
    )
    member = SwarmMember(id="a1", label="A", role="researcher", task="answer",
                         output="OLD ANSWER", version=2)
    feedback = [{"reviewer": "a2", "score": 3, "issues": ["missing proof"],
                 "suggestions": ["cite the source"]}]
    _system, user = coordinator._member_messages(member, "revise", feedback)
    assert "OLD ANSWER" in user["content"]
    assert "missing proof" in user["content"]


def test_an_edit_that_changes_a_file_is_recorded_as_changed():
    coordinator = SwarmCoordinator(
        gateway=None, swarm_id="swarm-changed-files", goal="answer", model="fake",
        tools=[], limits=SwarmLimits(max_agents=1),
    )
    coordinator._observe_changes({"path": "index.html", "changed": True, "replacements": 2})
    coordinator._observe_changes({"path": "same.html", "changed": False, "replacements": 0})
    coordinator._observe_changes({"path": "old.html", "deleted": True})
    assert coordinator.changed_files == ["index.html", "old.html"]


def test_a_parallel_plan_gives_dependents_their_prerequisite_output():
    coordinator = SwarmCoordinator(
        gateway=None, swarm_id="swarm-dependent-prior", goal="answer", model="fake",
        limits=SwarmLimits(max_agents=2),
    )
    first = SwarmMember(id="a1", label="A1", role="reader", task="collect",
                        output="PRIOR RESULT", status="done")
    dependent = SwarmMember(id="a2", label="A2", role="checker", task="check",
                            depends_on=["a1"])
    independent = SwarmMember(id="a3", label="A3", role="reader", task="other")
    coordinator._members = [first, dependent, independent]
    coordinator._by_id = {member.id: member for member in coordinator._members}
    coordinator.strategy = "map_reduce"

    _system, user = coordinator._member_messages(dependent, "work", [])
    assert "Previous stages to build on" in user["content"]
    assert "PRIOR RESULT" in user["content"]

    _system, user = coordinator._member_messages(independent, "work", [])
    assert "Previous stages to build on" not in user["content"]


def test_a_finish_after_cancellation_does_not_rewrite_the_member():
    coordinator = SwarmCoordinator(
        gateway=None, swarm_id="swarm-late-finish", goal="answer", model="fake",
        limits=SwarmLimits(max_agents=1),
    )
    member = SwarmMember(id="a1", label="A", role="researcher", task="answer",
                         status="cancelled")
    coordinator.status = "cancelled"
    coordinator._finish_member(member, "late output", "work", True)
    assert member.status == "cancelled" and member.output == ""
    with pytest.raises(SwarmCancelled):
        coordinator._check_cancel()


def test_a_json_object_without_a_usable_action_gets_a_correction():
    coordinator = SwarmCoordinator(
        gateway=None, swarm_id="swarm-empty-action", goal="answer", model="fake",
        limits=SwarmLimits(max_agents=1),
    )
    member = SwarmMember(id="a1", label="A", role="researcher", task="answer")
    final, results = coordinator._member_actions(member, "work", {"unexpected": 1}, {}, 1)
    assert final is None
    assert results and "no usable action" in results[0]


def test_a_truncated_reply_is_reported_in_warnings():
    class TruncatedGateway:
        def chat(self, config):
            return {"reply": '{"final": "partial"}', "finish_reason": "length",
                    "usage": {"completion_tokens": 20}}

    coordinator = SwarmCoordinator(
        gateway=TruncatedGateway(), swarm_id="swarm-truncated", goal="answer", model="fake",
        limits=SwarmLimits(max_agents=1),
    )
    reply, usage = coordinator._complete(
        "fake", [{"role": "user", "content": "go"}], temperature=0,
        max_tokens=20, label="leader", phase="work")

    assert reply == '{"final": "partial"}'
    assert usage["completion_tokens"] == 40  # both attempts are counted
    assert any("output limit" in warning for warning in coordinator.warnings)


def test_a_truncated_reply_survives_an_empty_retry():
    class EmptyRetry:
        def __init__(self):
            self.calls = 0

        def chat(self, config):
            self.calls += 1
            if self.calls == 1:
                return {"reply": "partial answer", "finish_reason": "length",
                        "usage": {"completion_tokens": 20}}
            return {"reply": "", "usage": {}}

    coordinator = SwarmCoordinator(
        gateway=EmptyRetry(), swarm_id="swarm-partial-retry", goal="answer", model="fake",
        limits=SwarmLimits(max_agents=1),
    )
    reply, usage = coordinator._complete(
        "fake", [{"role": "user", "content": "go"}], temperature=0,
        max_tokens=20, label="a1", phase="work", empty_ok=True)

    assert reply == "partial answer"
    assert usage["completion_tokens"] == 20
    assert any("partial reply" in warning for warning in coordinator.warnings)


def test_swarm_limits_allow_a_free_hand_with_a_safety_ceiling():
    assert SwarmLimits().max_agents == 16
    assert SwarmLimits.from_mapping({"max_agents": 64}).max_agents == 64
    assert SwarmLimits.from_mapping({"max_agents": 999}).max_agents == 64
    limits = SwarmLimits.from_mapping({"max_model_concurrency": 4,
                                       "max_tool_concurrency": 8})
    assert limits.max_model_concurrency == 4 and limits.max_tool_concurrency == 8
    direct = SwarmCoordinator(
        gateway=SwarmGateway(), swarm_id="swarm-cap", goal="answer", model="fake",
        tools=[], limits=SwarmLimits(max_agents=999, max_model_concurrency=999,
                                     max_tool_concurrency=999))
    assert direct.limits.max_agents == 64
    assert direct.limits.max_model_concurrency == 64
    assert direct.limits.max_tool_concurrency == 128


def test_a_large_team_runs_in_one_parallel_wave():
    class ManyAgentGateway(SwarmGateway):
        def chat(self, config):
            system = config["messages"][0]["content"]
            if "Plan a focused team" in system:
                self.calls.append(copy.deepcopy(config))
                return {"reply": json.dumps({
                    "strategy": "map_reduce", "rounds": 0,
                    "agents": [{"id": f"agent-{index}",
                                "task": f"inspect area {index}"} for index in range(12)],
                }), "usage": {}}
            if "You are one member of the swarm" in system:
                return {"reply": json.dumps({"final": "verified finding"}), "usage": {}}
            return super().chat(config)

    coordinator = SwarmCoordinator(
        gateway=ManyAgentGateway(), swarm_id="swarm-many", goal="audit everything",
        model="fake", tools=[],
        limits=SwarmLimits(max_agents=16, max_steps=2, max_rounds=0, max_seconds=20),
    )
    result = coordinator.run(strategy="map_reduce", rounds=0)
    assert result["ok"] is True, result.get("error")
    assert len(result["agents"]) == 12
    assert all(agent["status"] == "done" for agent in result["agents"])


def test_model_concurrency_is_capped_across_agents():
    class SlowGateway:
        def __init__(self):
            self.active = 0
            self.peak = 0
            self.lock = threading.Lock()

        def stream(self, config, cancel):
            with self.lock:
                self.active += 1
                self.peak = max(self.peak, self.active)
            try:
                time.sleep(0.03)
                yield {"type": "done", "reply": "done", "usage": {}}
            finally:
                with self.lock:
                    self.active -= 1

    gateway = SlowGateway()
    coordinator = SwarmCoordinator(
        gateway=gateway, swarm_id="swarm-slots", goal="answer", model="fake",
        tools=[], limits=SwarmLimits(max_agents=4, max_model_concurrency=2,
                                     max_seconds=10),
    )

    def call(_index):
        return coordinator._complete(
            "fake", [{"role": "user", "content": "go"}], temperature=0,
            max_tokens=16, label="a", phase="work")

    with ThreadPoolExecutor(max_workers=4) as pool:
        replies = list(pool.map(call, range(4)))
    assert all(reply == "done" for reply, _usage in replies)
    assert 1 < gateway.peak <= 2


def test_large_teams_are_condensed_into_group_summaries_before_synthesis():
    class GroupGateway:
        def __init__(self):
            self.calls = []

        def chat(self, config):
            system = config["messages"][0]["content"]
            self.calls.append("\n".join(
                str(message.get("content") or "") for message in config["messages"]))
            if "condense one group" in system.lower():
                return {"reply": json.dumps({"summary": "group digest"}), "usage": {}}
            if "Synthesise the members' work" in system:
                return {"reply": json.dumps({"final": "merged group result"}), "usage": {}}
            return {"reply": json.dumps({"final": "member result"}), "usage": {}}

    gateway = GroupGateway()
    coordinator = SwarmCoordinator(
        gateway=gateway, swarm_id="swarm-groups", goal="audit everything", model="fake",
        tools=[], limits=SwarmLimits(max_agents=20, max_model_concurrency=4,
                                     max_seconds=20),
    )
    coordinator._members = [
        SwarmMember(id=f"a{index}", label=f"Agent {index}", role="researcher",
                    task=f"area {index}", output=f"finding {index}")
        for index in range(20)
    ]
    result = coordinator._synthesize()
    assert result["final"] == "merged group result"
    assert sum("condense one group" in system.lower() for system in gateway.calls) == 3
    final_call = next(system for system in gateway.calls if "Synthesise" in system)
    assert "condensed into 3 group summaries" in final_call


def test_reasoning_chunks_emit_one_status_not_a_flood():
    class ReasoningGateway:
        def stream(self, config, cancel):
            for _ in range(20):
                yield {"type": "reasoning", "text": "thinking"}
            yield {"type": "done", "reply": '{"final": "ok"}', "usage": {}}

    events = []
    coordinator = SwarmCoordinator(
        gateway=ReasoningGateway(), swarm_id="swarm-reasoning", goal="answer", model="fake",
        tools=[], limits=SwarmLimits(max_agents=1),
        on_event=lambda kind, **fields: events.append((kind, fields)),
    )
    member = SwarmMember(id="a1", label="A", role="researcher", task="answer")
    coordinator._complete("fake", [{"role": "user", "content": "go"}],
                          temperature=0, max_tokens=20, label="a1", phase="work",
                          member=member)
    reasoning = [fields for kind, fields in events
                 if kind == "swarm_agent_status" and "reasoning" in fields.get("message", "")]
    assert len(reasoning) == 1


def test_member_observation_keeps_the_newest_evidence_under_pressure():
    coordinator = SwarmCoordinator(
        gateway=SwarmGateway(), swarm_id="swarm-craft", goal="find the bug",
        model="fake", tools=[],
        limits=SwarmLimits(max_agents=1, max_observation_chars=300,
                           max_prompt_chars=2_000),
    )
    member = SwarmMember(id="a1", label="A", role="researcher", task="find the bug")
    results = [
        f"old result {index}: " + "x" * 500 + "\nOlder filler"
        for index in range(5)
    ]
    results.append("file.read returned:\n" + json.dumps({
        "path": "src/models/foo.cpp", "start_line": 40, "end_line": 42,
        "content": "NEWEST-MARKER: unchecked index\n" + "y" * 900,
        "error": "undefined value",
    }))
    rendered = coordinator._member_observation(member, "work", results, iteration=3, limit=4)
    assert "NEWEST-MARKER" in rendered
    assert "src/models/foo.cpp" in rendered
    assert len(rendered) <= 1_200
    assert coordinator._cache_stats()["compacted_raw_chars"] > 0


def test_compaction_ledger_keeps_tools_and_anchors():
    coordinator = SwarmCoordinator(
        gateway=SwarmGateway(), swarm_id="swarm-craft-ledger", goal="answer",
        model="fake", tools=[],
        limits=SwarmLimits(max_agents=1, max_prompt_chars=1_200,
                           max_context_chars=30_000, max_agent_context_chars=10_000),
    )
    member = SwarmMember(id="a1", label="A", role="researcher", task="answer")
    messages = [{"role": "system", "content": "system"},
                {"role": "user", "content": "task"}]
    for index in range(10):
        messages += [
            {"role": "assistant", "content": json.dumps({
                "tool": "file.read", "input": {"path": f"src/file-{index}.cpp"}})},
            {"role": "user", "content": (
                f"src/file-{index}.cpp:12 returned: error: undefined value\n"
                + "z" * 300)},
        ]
    coordinator._compact_messages(messages, member, max_tokens=500)
    ledger = next(item["content"] for item in messages
                  if str(item["content"]).startswith(COMPACTED_PREFIX))
    assert "Tools: file.read" in ledger
    assert "src/file-" in ledger and "error" in ledger.lower()
    assert CONTEXT_HANDLE.search(ledger)
    assert coordinator._cache_stats()["context_episodes"] >= 1


def test_peer_digest_and_blackboard_keep_anchors_when_shortened():
    coordinator = SwarmCoordinator(
        gateway=SwarmGateway(), swarm_id="swarm-craft-digest", goal="fix the crash",
        model="fake", tools=[], limits=SwarmLimits(max_agents=2),
    )
    first = SwarmMember(id="a1", label="A", role="researcher", task="find the crash")
    second = SwarmMember(id="a2", label="B", role="checker", task="fix the crash")
    first.output = "x" * 800 + "\nCrash at src/models/foo.cpp:42 because value is undefined."
    coordinator._members = [first, second]
    coordinator._by_id = {"a1": first, "a2": second}
    digest = coordinator._peer_digest(second, chars=220)
    assert "src/models/foo.cpp:42" in digest
    coordinator._blackboard.write("a1", "finding.crash",
                                 {"path": "src/models/foo.cpp", "line": 42,
                                  "note": "n" * 500})
    board = coordinator._blackboard.digest(limit=4, value_chars=160)
    assert "src/models/foo.cpp" in board


def test_shared_evidence_anchors_reach_the_member_prompt():
    coordinator = SwarmCoordinator(
        gateway=SwarmGateway(), swarm_id="swarm-craft-evidence", goal="answer",
        model="fake", tools=[], limits=SwarmLimits(max_agents=1),
    )
    member = SwarmMember(id="a1", label="A", role="researcher", task="answer")
    coordinator._members = [member]
    coordinator._by_id = {"a1": member}
    coordinator._remember_evidence(json.dumps({
        "path": "src/models/foo.cpp", "line": 42, "error": "undefined value"}))
    prompt = coordinator._member_messages(member, "work", [])
    text = "\n".join(item["content"] for item in prompt)
    assert "Shared evidence anchors" in text
    assert "src/models/foo.cpp" in text
    handles = coordinator._context_bank.put(
        "a1", "tool", "long result",
        "src/models/foo.cpp:42 " + "payload " * 500)
    item = coordinator._context_bank.get(handles)
    assert item["anchors"] and len(item["summary"]) <= 260


def test_tool_arguments_accept_model_key_spellings():
    read = _normalise_tool_arguments("file.read", {
        "filePath": "'src/main.cpp'", "startLine": "12", "to": 40,
    })
    assert read == {"path": "src/main.cpp", "start_line": 12, "end_line": 40}

    write = _normalise_tool_arguments("file.write", {
        "file_path": "README.md", "text": "body",
    })
    assert write["path"] == "README.md" and write["content"] == "body"

    search = _normalise_tool_arguments("file.search", {
        "q": "needle", "glob": "*.py", "maxResults": "20",
    })
    assert search["query"] == "needle" and search["include"] == "*.py"
    assert search["max_results"] == 20

    command = _normalise_tool_arguments("shell.exec", {"cmd": "git status"})
    assert command["command"] == "git status"


def test_usable_text_rejects_tool_markup_but_keeps_prose():
    markup = ('<｜｜DSML｜｜ calls>\n<｜｜DSML｜｜ invoke name="file.search">\n'
              '<｜｜DSML｜｜ parameter name="query" string="true">x</｜｜DSML｜｜ parameter>\n'
              '</｜｜DSML｜｜ invoke>\n</｜｜DSML｜｜ calls>')
    assert _usable_text(markup) == ""
    assert _usable_text("A real finding with enough words to be useful.") == (
        "A real finding with enough words to be useful.")


def test_synthesis_with_tool_markup_falls_back_to_member_results():
    class MarkupGateway(SwarmGateway):
        def chat(self, config):
            system = config["messages"][0]["content"]
            if "Synthesise the members' work" in system:
                self.calls.append(copy.deepcopy(config))
                return {"reply": (
                    '<｜｜DSML｜｜ calls>\n<｜｜DSML｜｜ invoke name="file.search">\n'
                    '<｜｜DSML｜｜ parameter name="query" string="true">x</｜｜DSML｜｜ parameter>\n'
                    '</｜｜DSML｜｜ invoke>\n</｜｜DSML｜｜ calls>'), "usage": {}}
            if "You are one member of the swarm" in system:
                return {"reply": json.dumps({"final": "member finding"}), "usage": {}}
            return super().chat(config)

    coordinator = SwarmCoordinator(
        gateway=MarkupGateway(), swarm_id="swarm-markup", goal="answer", model="fake",
        tools=[], limits=LIMITS,
    )
    result = coordinator.run(agents=[{"id": "a1", "task": "answer"}],
                             strategy="map_reduce", rounds=0)
    assert result["ok"] is True
    assert "member finding" in result["final"]
    assert "DSML" not in result["final"] and "invoke" not in result["final"]
    assert result["warnings"]


def test_leader_prompt_prefers_independent_workstreams_in_one_wave():
    coordinator = SwarmCoordinator(
        gateway=SwarmGateway(), swarm_id="swarm-parallel-prompt", goal="audit the project",
        model="fake", tools=[], limits=LIMITS,
    )
    system = coordinator._leader_plan_prompt([], "auto", 1)[0]["content"]
    assert "sized to" in system
    assert "as many agents as the work needs" in system
    assert "separate members" in system
    assert "same wave" in system
    assert "depends_on only when" in system
