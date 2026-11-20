"""The HTTP transport: request security, the error contract, headers and event streams."""

from __future__ import annotations

import http.client
import json
import urllib.error
import urllib.request

import pytest

from mariana.studio.server import _hostname
from tests.helpers import read_events, request, request_error, studio


@pytest.fixture()
def server(tmp_path):
    with studio(tmp_path) as (instance, base):
        yield instance, base


def raw(base, method, path, body=None, headers=None):
    """Send a request with exact headers; returns ``(status, headers, body bytes)``."""
    host, port = base.rsplit("//", 1)[1].split(":")
    connection = http.client.HTTPConnection(host, int(port), timeout=15)
    try:
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()


def test_other_web_pages_cannot_drive_studio(server):
    _, base = server
    json_headers = {"Content-Type": "application/json"}
    forbidden = [
        ("POST", "/api/jobs/clear", b"{}", {**json_headers, "Origin": "http://evil.example"}),
        ("GET", "/api/state", None, {"Origin": "http://evil.example"}),
        ("POST", "/api/jobs/clear", b"{}", {**json_headers, "Origin": "null"}),
        ("POST", "/api/jobs/clear", b"{}", {**json_headers, "Sec-Fetch-Site": "cross-site"}),
        ("GET", "/api/state", None, {"Host": "attacker.example"}),       # DNS rebinding
        ("GET", "/api/state", None, {"Host": "attacker.example:8765"}),
        ("OPTIONS", "/api/train", None, {"Origin": "http://evil.example",
                                         "Access-Control-Request-Method": "POST"}),
    ]
    for method, path, body, headers in forbidden:
        status, response_headers, payload = raw(base, method, path, body, headers)
        assert status == 403, (method, path, headers)
        assert json.loads(payload)["code"] == "forbidden"
        assert "Access-Control-Allow-Origin" not in response_headers
    status, _, payload = raw(base, "POST", "/api/jobs/clear", b"{}", {"Content-Type": "text/plain"})
    assert status == 400 and "application/json" in json.loads(payload)["error"]
    status, _, payload = raw(base, "POST", "/api/jobs/clear", b"x=1",
                             {"Content-Type": "application/x-www-form-urlencoded"})
    assert status == 400  # an HTML form cannot post here either

    allowed = [
        ("POST", "/api/jobs/clear", b"{}", {**json_headers, "Origin": base}),
        ("POST", "/api/jobs/clear", b"{}", {**json_headers, "Sec-Fetch-Site": "same-origin"}),
        ("POST", "/api/jobs/clear", None, {"Content-Length": "0"}),
        ("GET", "/api/state", None, {"Host": f"localhost:{base.rsplit(':', 1)[1]}"}),
    ]
    for method, path, body, headers in allowed:
        assert raw(base, method, path, body, headers)[0] == 200, headers
    assert _hostname("[::1]:8765") == "::1" and _hostname("LOCALHOST:80") == "localhost"
    assert _hostname("127.0.0.1") == "127.0.0.1"


def test_errors_share_one_shape_and_never_leak_tracebacks(server, monkeypatch, capsys):
    instance, base = server
    assert request_error(base + "/api/unknown") == (404, {"error": "Not found", "code": "not_found"})
    status, body = request_error(base + "/api/train")
    assert status == 405 and body["code"] == "method_not_allowed" and "use POST" in body["error"]
    status, _, payload = raw(base, "POST", "/api/chat", b"{broken", {"Content-Type": "application/json"})
    assert status == 400 and json.loads(payload) == {"error": "Invalid JSON", "code": "bad_request"}
    status, _, payload = raw(base, "POST", "/api/chat", None,
                             {"Content-Type": "application/json", "Content-Length": "99999999"})
    assert status == 413 and json.loads(payload)["code"] == "too_large"
    status, body = request_error(base + "/api/corpus/inspect", {"path": "/definitely/not/here"})
    assert status == 400 and "does not exist" in body["error"]  # a service ValueError

    def explode():
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(instance.app.training, "state", explode)
    status, body = request_error(base + "/api/job")
    assert status == 500 and body["code"] == "server_error"
    assert body["error"] == "Internal error: RuntimeError: secret internal detail"
    assert "Traceback" not in json.dumps(body)
    assert "Traceback" in capsys.readouterr().err  # the operator still gets it on stderr


def test_responses_carry_security_headers(server):
    _, base = server
    status, headers, body = raw(base, "GET", "/api/state")
    assert status == 200 and headers["X-Content-Type-Options"] == "nosniff"
    assert headers["Cache-Control"] == "no-store" and headers["Referrer-Policy"] == "no-referrer"
    assert headers["Content-Type"] == "application/json; charset=utf-8"
    status, headers, body = raw(base, "HEAD", "/api/state")
    assert status == 200 and body == b"" and int(headers["Content-Length"]) > 0
    status, headers, _ = raw(base, "GET", "/api/agents/spec/aml")
    assert headers["Content-Type"].startswith("text/markdown")
    assert headers["Content-Security-Policy"] == "default-src 'none'"
    assert "Server" not in headers or "Python" not in headers["Server"]


def test_event_streams_send_the_current_state_first(server):
    instance, base = server
    (first,) = read_events(base + "/api/events", limit=1)
    assert first["job"] is None and isinstance(first["version"], int)
    (jobs,) = read_events(base + "/api/jobs/events", limit=1)
    assert jobs["jobs"] == []
    job = instance.app.jobs.create("download", "Visible")
    assert read_events(base + "/api/jobs/events", limit=1)[0]["jobs"][0]["id"] == job.id


def test_the_server_owns_and_closes_its_services(tmp_path):
    with studio(tmp_path) as (instance, base):
        assert instance.manager is instance.app.training and instance.gateway is instance.app.gateway
        assert instance.agents is instance.app.agents and instance.app.port == instance.server_address[1]
        assert request(base + "/api/state")["root"] == str(tmp_path)
    assert instance.app.jobs._closed is True
    assert instance.app.agents._stop.is_set()
    with pytest.raises(urllib.error.URLError):
        urllib.request.urlopen(base + "/api/state", timeout=2)
