"""Chat swarm HTTP route: validation, SSE forwarding and in-band failures."""

from __future__ import annotations

import json
import time
import urllib.request

import pytest

from mariana.studio.errors import ConflictError
from tests.helpers import (
    mock_provider, read_events, request, request_error, serving, studio, wait_for,
)

PAYLOAD = {
    "model": "fake",
    "messages": [{"role": "user", "content": "Explain this"}],
}


@pytest.fixture()
def api(tmp_path):
    (tmp_path / "agent-workspace").mkdir()
    (tmp_path / "agent-workspace" / "settings.json").write_text(
        json.dumps({"discover_local": False}))
    with studio(tmp_path) as (server, base):
        yield server, base, tmp_path


class FakeSwarm:
    """A stand-in service so the route can be tested without running models."""

    def __init__(self, events=None, failure=None):
        self.events = events or []
        self.failure = failure
        self.prepared = []
        self.cancelled = []

    def prepare(self, config):
        self.prepared.append(config)
        return {"model": config.get("model")}

    def stream(self, config, cancelled):
        self.cancelled.append(cancelled)
        yield {"type": "swarm_start", "swarm": "s1", "message": "The leader is planning."}
        for event in self.events:
            yield event
        if self.failure:
            raise self.failure


def test_chat_endpoints_reject_non_object_json(api):
    _, base, _ = api
    for path in ("/api/chat", "/api/chat/stream", "/api/chat/swarm/stream"):
        status, error = request_error(base + path, ["not", "an", "object"])
        assert status == 400, path
        assert "JSON object" in error["error"], (path, error)
        assert error["code"] == "bad_request"


def test_swarm_stream_validates_before_opening_the_stream(api):
    _, base, _ = api
    routes = {(route["method"], route["path"]) for route in request(base + "/api/routes")["routes"]}
    assert ("POST", "/api/chat/swarm/stream") in routes

    status, error = request_error(base + "/api/chat/swarm/stream", {
        "model": "missing", "messages": [{"role": "user", "content": "hi"}]})
    assert status == 400 and "Model not found" in error["error"]
    assert error["code"] == "bad_request"
    status, error = request_error(base + "/api/chat/swarm/stream", {
        "model": "missing", "messages": []})
    assert status == 400 and "Enter a message first" in error["error"]


def test_a_disconnect_cancels_the_swarm_stream(api):
    server, base, _ = api
    cancelled = []

    class SlowSwarm:
        def prepare(self, config):
            return config

        def stream(self, config, token):
            cancelled.append(token)
            yield {"type": "swarm_start", "swarm": "s1", "message": "Started"}
            while not token.is_set():
                time.sleep(0.05)
                yield {"type": "swarm_agent_step", "agent": "a1"}

    server.app.chat_swarm = SlowSwarm()
    req = urllib.request.Request(
        base + "/api/chat/swarm/stream", data=json.dumps(PAYLOAD).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    response = urllib.request.urlopen(req, timeout=10)
    try:
        assert b"swarm_start" in response.readline()
    finally:
        response.close()
    assert wait_for(lambda: cancelled and cancelled[0].is_set(), timeout=5)


def test_swarm_route_forwards_events_and_closes_the_stream(api):
    server, base, _ = api
    fake = FakeSwarm([{"type": "done", "reply": "answer", "swarm": {"status": "completed"}}])
    server.app.chat_swarm = fake
    events = read_events(base + "/api/chat/swarm/stream", PAYLOAD)
    assert [event["type"] for event in events] == ["swarm_start", "done"]
    assert events[-1]["reply"] == "answer"
    assert fake.prepared == [PAYLOAD]
    assert wait_for(lambda: fake.cancelled and fake.cancelled[0].is_set())


def test_a_studio_error_after_the_stream_opens_is_sent_in_band(api):
    server, base, _ = api
    server.app.chat_swarm = FakeSwarm(
        failure=ConflictError("The model is busy; retry in a moment"))
    events = read_events(base + "/api/chat/swarm/stream", PAYLOAD)
    assert events[-1] == {
        "type": "error", "error": "The model is busy; retry in a moment", "code": "conflict",
    }


def test_an_unexpected_failure_after_the_stream_opens_never_leaks_a_traceback(api):
    server, base, _ = api
    server.app.chat_swarm = FakeSwarm(failure=RuntimeError("boom"))
    events = read_events(base + "/api/chat/swarm/stream", PAYLOAD)
    assert events[-1] == {"type": "error", "error": "boom", "code": "server_error"}


def test_a_real_chat_swarm_streams_over_http(api):
    _, base, _ = api

    def answer(handler, payload):
        messages = payload.get("messages") or []
        system = messages[0]["content"] if messages else ""
        if "Plan a focused team" in system:
            reply = json.dumps({
                "strategy": "map_reduce", "rounds": 0,
                "agents": [
                    {"id": "reader", "label": "Reader", "role": "researcher",
                     "task": "find the argument"},
                    {"id": "checker", "label": "Checker", "role": "reviewer",
                     "task": "verify the argument"},
                ],
                "notes": "two independent members",
            })
        elif "You are one member of the swarm" in system:
            reply = json.dumps({"final": "member finding"})
        elif "Synthesise the members' work" in system:
            reply = json.dumps({"final": "**Merged answer**", "summary": "done", "confidence": 0.9})
        else:
            reply = json.dumps({"final": "unexpected phase"})
        return 200, {
            "id": "chatcmpl-swarm", "object": "chat.completion",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": reply},
                          "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

    provider = mock_provider({("POST", "/v1/chat/completions"): answer})
    with serving(provider) as upstream:
        request(base + "/api/models/endpoints", {
            "name": "Swarm provider", "kind": "openai",
            "base_url": f"{upstream}/v1", "model": "m",
        })
        events = read_events(base + "/api/chat/swarm/stream", {
            "model": "endpoint:swarm-provider",
            "messages": [{"role": "user", "content": "Summarise the argument."}],
            "strategy": "map_reduce", "rounds": 0, "max_agents": 2,
        }, limit=200)

    assert events[0]["type"] == "swarm_start"
    assert any(event["type"] == "swarm_plan" for event in events)
    assert any(event["type"] == "swarm_agent_end" for event in events)
    done = events[-1]
    assert done["type"] == "done"
    assert "Merged answer" in done["reply"]
    assert done["swarm"]["status"] == "completed"
    assert len(done["swarm"]["agents"]) == 2
    assert done["usage"]["completion_tokens"] > 0
