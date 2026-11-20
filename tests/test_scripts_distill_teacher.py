"""The teacher-distillation script: response parsing, providers and the HTTP client."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import threading

import distill_teacher as dt


def test_strip_think():
    assert dt._strip_think("<think>secret</think>Answer") == "Answer"
    assert dt._strip_think("<think>unfinished") == ""
    assert dt._strip_think("plain") == "plain"


def test_parse_openai():
    data = {
        "choices": [{
            "text": " Paris",
            "logprobs": {
                "content": [
                    {"logprob": -0.5},
                    {"logprob": -1.5},
                ]
            },
            "finish_reason": "stop",
        }]
    }
    out = dt.TeacherClient._parse_openai(data)
    assert out["text"] == " Paris"
    assert abs(out["mean_logprob"] - (-1.0)) < 1e-9
    assert out["finish_reason"] == "stop"


def test_parse_llama():
    data = {
        "content": "hello",
        "completion_probabilities": [{"logprob": -0.2}, {"logprob": -0.4}],
        "stop_type": "limit",
    }
    out = dt.TeacherClient._parse_llama(data)
    assert out["text"] == "hello"
    assert abs(out["mean_logprob"] - (-0.3)) < 1e-9
    assert out["finish_reason"] == "limit"


def test_parse_chat():
    data = {
        "choices": [{
            "message": {"content": "answer", "reasoning_content": "hidden"},
            "finish_reason": "stop",
        }]
    }
    out = dt.TeacherClient._parse_chat(data)
    assert out["text"] == "answer"
    assert out["reasoning"] == "hidden"


class _FakeClient:
    def complete(self, *args, **kwargs):
        return {"text": "Answer.", "mean_logprob": -0.2, "finish_reason": "stop"}


class _Args:
    candidates = 1
    max_tokens = 8
    temperature = 0.5
    top_p = 0.9
    n_probs = 4
    seed = None
    stop = None
    include_prompt = True
    keep_think = False
    drop_truncated = False
    prompt_separator = "\n\n"
    min_chars = 1
    max_chars = 0
    min_mean_logprob = None


def test_sample_once_adds_separator():
    out = dt.sample_once(_FakeClient(), "Question?", _Args())
    assert out["text"] == "Question?\n\nAnswer."


class _Handler(BaseHTTPRequestHandler):
    last_auth = None
    last_path = None

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        _Handler.last_auth = self.headers.get("Authorization")
        body = {"data": [{"id": "mock-teacher"}]}
        self._send(body)

    def do_POST(self):
        _Handler.last_auth = self.headers.get("Authorization")
        _Handler.last_path = self.path
        length = int(self.headers.get("Content-Length", "0"))
        _ = json.loads(self.rfile.read(length) or b"{}")
        if self.path.endswith("/chat/completions"):
            body = {
                "choices": [{
                    "message": {"content": "mock answer", "reasoning_content": ""},
                    "finish_reason": "stop",
                }]
            }
        elif self.path.endswith("/completions"):
            body = {
                "choices": [{
                    "text": "mock completion",
                    "logprobs": {"content": [{"logprob": -0.1}]},
                    "finish_reason": "stop",
                }]
            }
        else:
            body = {"content": "native completion", "completion_probabilities": [{"logprob": -0.2}]}
        self._send(body)

    def _send(self, body):
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def test_resolve_teacher_provider():
    class Args:
        provider = "deepseek"
        api = "chat"
        url = None
        model = None
        api_key = None

    old_key = os.environ.pop("DEEPSEEK_API_KEY", None)
    old_model = os.environ.pop("DEEPSEEK_MODEL", None)
    try:
        try:
            dt._resolve_teacher(Args())
            assert False, "missing DeepSeek key should fail"
        except SystemExit:
            pass

        os.environ["DEEPSEEK_API_KEY"] = "sk-test"
        url, model, key = dt._resolve_teacher(Args())
        assert url == dt.DEEPSEEK_URL
        assert model == dt.DEEPSEEK_MODEL
        assert key == "sk-test"

        Args.model = "deepseek-chat"
        assert dt._resolve_teacher(Args())[1] == "deepseek-chat"

        Args.model = None
        os.environ["DEEPSEEK_MODEL"] = "deepseek-env-model"
        assert dt._resolve_teacher(Args())[1] == "deepseek-env-model"

        Args.model = "auto"
        assert dt._resolve_teacher(Args())[1] is None

        Args.model = None
        Args.api = "openai"
        try:
            dt._resolve_teacher(Args())
            assert False, "DeepSeek requires the chat API"
        except SystemExit:
            pass
    finally:
        os.environ.pop("DEEPSEEK_API_KEY", None)
        os.environ.pop("DEEPSEEK_MODEL", None)
        if old_key is not None:
            os.environ["DEEPSEEK_API_KEY"] = old_key
        if old_model is not None:
            os.environ["DEEPSEEK_MODEL"] = old_model

    class LocalArgs:
        provider = "local"
        api = "chat"
        url = None
        model = None
        api_key = None

    url, model, key = dt._resolve_teacher(LocalArgs())
    assert url == "http://127.0.0.1:8080"
    assert model is None


def test_http_client():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}"
        chat = dt.TeacherClient(
            url, api="chat", api_key="secret", retries=1, timeout=5
        )
        out = chat.complete("hi", max_tokens=8, temperature=0.5, top_p=0.9)
        assert out["text"] == "mock answer"
        assert chat.model == "mock-teacher"
        assert _Handler.last_auth == "Bearer secret"

        native = dt.TeacherClient(url, api="llama", retries=1, timeout=5)
        out = native.complete("hi", max_tokens=8, temperature=0.5, top_p=0.9)
        assert out["text"] == "native completion"

        deepseek = dt.TeacherClient(
            url, api="chat", model="deepseek-flash", api_key="secret",
            provider="deepseek", retries=1, timeout=5,
        )
        out = deepseek.complete("hi", max_tokens=8, temperature=0.5, top_p=0.9)
        assert out["text"] == "mock answer"
        assert _Handler.last_path == "/chat/completions"
    finally:
        server.shutdown()
        server.server_close()
