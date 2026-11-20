"""Routing and request parsing, independent of any socket."""

from __future__ import annotations

import pytest

from mariana.studio.errors import BadRequestError, MethodNotAllowedError, NotFoundError, TooLargeError
from mariana.studio.http import EventStream, Request, Response, Router, accepted, created


def test_literal_segments_win_over_parameters_and_values_are_decoded():
    router = Router()

    @router.get("/api/things/{id}")
    def read(app, request):
        """Read one thing.

        More detail that discovery must not show.
        """

    @router.get("/api/things/special")
    def special(app, request):
        pass

    @router.post("/api/things/{id}/runs")
    def run(app, request):
        pass

    router.add("DELETE", "/api/things/{id}", read)
    assert router.resolve("GET", "/api/things/special") == (special, {})
    assert router.resolve("GET", "/api/things/a%20b%2Fc") == (read, {"id": "a b/c"})
    assert router.resolve("get", "/api/things/endpoint%3Ax/") == (read, {"id": "endpoint:x"})
    assert router.resolve("POST", "/api/things/7/runs") == (run, {"id": "7"})
    with pytest.raises(MethodNotAllowedError, match="use DELETE, GET"):
        router.resolve("PUT", "/api/things/7")
    for missing in ("/api/things", "/api/things/7/runs/extra", "/api/other"):
        with pytest.raises(NotFoundError):
            router.resolve("GET", missing)
    described = router.describe()
    assert {"method": "GET", "path": "/api/things/{id}", "summary": "Read one thing."} in described
    assert [route["path"] for route in described] == sorted(route["path"] for route in described)
    merged = Router()
    merged.include(router)
    assert len(merged.routes) == 4


def test_request_reads_query_and_json_with_limits():
    request = Request("post", "/api/x?name=a&name=b&flag=TRUE&n=250&bad=x", {"Content-Type": "application/json"},
                      b'{"key": 1}', {"id": "7"})
    assert request.method == "POST" and request.path == "/api/x" and request.params == {"id": "7"}
    assert request.arg("name") == "a" and request.arg("missing", "d") == "d"
    assert request.flag("flag") and not request.flag("missing")
    assert request.int_arg("n", 5, 0, 100) == 100 and request.int_arg("bad", 5, 0, 100) == 5
    assert request.json == {"key": 1} and request.header("content-type") == "application/json"
    assert Request("POST", "/", body=b"").json == {}
    deep = b"[" * 200_000 + b"1" + b"]" * 200_000
    for body, message in ((b"{nope", "Invalid JSON"), (b"[1]", "must be a JSON object"),
                          (b"\xff\xfe", "Invalid JSON"), (deep, "Invalid JSON")):
        with pytest.raises(BadRequestError, match=message):
            Request("POST", "/", body=body).json
    big = b'{"k": "' + b"x" * 1_000_001 + b'"}'
    with pytest.raises(TooLargeError):
        Request("POST", "/", body=big).json
    assert len(Request("POST", "/", body=big).large_json(2_000_000)["k"]) == 1_000_001


def test_response_helpers():
    assert created({"a": 1}).status == 201 and accepted({"a": 1}).status == 202
    response = Response.json({"text": "é"}, 418)
    assert response.status == 418 and response.body == '{"text": "é"}'.encode()
    text = Response.text("# title", "text/markdown; charset=utf-8")
    assert text.content_type.startswith("text/markdown") and text.body == b"# title"
    assert list(EventStream(iter([{"a": 1}])).events) == [{"a": 1}]
