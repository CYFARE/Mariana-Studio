"""Managed runtimes: a llama-server and an Ollama server started and stopped by Studio."""

from __future__ import annotations

import json
import socket
import sys

import pytest

from mariana.studio.errors import UnavailableError
from mariana.studio.models import ModelGateway
from mariana.studio.models.runtimes import LlamaServer, OllamaServer

FAKE_LLAMA_SERVER = r'''#!{python}
"""A stand-in for llama-server: health check plus an echoing chat completion stream."""
import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer

args = sys.argv[1:]
model, port = args[args.index("-m") + 1], int(args[args.index("--port") + 1])
with open(model + ".args", "w") as handle:
    json.dump(args, handle)
if "broken" in model:
    print("error: unknown model architecture: 'mllama'")
    sys.exit(1)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = b'{{"status": "ok"}}'
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        reply = "echo: " + payload["messages"][-1]["content"]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for chunk in (
            {{"choices": [{{"delta": {{"content": reply}}}}]}},
            {{"choices": [{{"delta": {{}}, "finish_reason": "stop"}}],
              "usage": {{"prompt_tokens": 3, "completion_tokens": 2}}}},
        ):
            self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
        self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, *args):
        pass


HTTPServer(("127.0.0.1", port), Handler).serve_forever()
'''

FAKE_OLLAMA = r'''#!{python}
import os, sys
from http.server import BaseHTTPRequestHandler, HTTPServer

assert sys.argv[1:] == ["serve"], sys.argv


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = b'{{"version": "0.0.test"}}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


HTTPServer(("127.0.0.1", int(os.environ["OLLAMA_HOST"].rsplit(":", 1)[1])), Handler).serve_forever()
'''


def executable(path, template):
    path.write_text(template.format(python=sys.executable))
    path.chmod(0o755)
    return path


@pytest.fixture()
def workspace(tmp_path):
    settings = {"discover_local": False,
                "llama_server_path": str(executable(tmp_path / "llama-server", FAKE_LLAMA_SERVER))}
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "tiny-Q4_0.gguf").write_bytes(b"not really gguf")
    gateway = ModelGateway(tmp_path, settings=lambda: settings)
    yield tmp_path, gateway, settings
    gateway.close()


def launch_arguments(tmp_path, name="tiny-Q4_0.gguf"):
    return json.loads((tmp_path / "models" / f"{name}.args").read_text())


def test_gguf_models_chat_through_a_managed_llama_server(workspace):
    tmp_path, gateway, settings = workspace
    entry = next(item for item in gateway.catalog(force=True) if item["name"] == "tiny-Q4_0.gguf")
    assert entry["available"] and entry["runtime"] == "llama-server"
    assert entry["detail"]["quantization"] == "Q4_0"

    result = gateway.chat({"model": entry["id"], "message": "hello"})
    assert result["reply"] == "echo: hello" and result["prompt_format"] == "chat_template"
    assert result["usage"] == {"prompt_tokens": 3, "completion_tokens": 2}
    assert result["context"] == 4096  # the default when the file states none
    status = gateway.llama.status()
    assert status["running"] and status["model"] == "tiny-Q4_0.gguf" and status["context"] == 4096
    assert gateway.runtimes()["loaded"] == [{"runtime": "llama-server", "path": "tiny-Q4_0.gguf"}]
    arguments = launch_arguments(tmp_path)
    assert arguments[arguments.index("-c") + 1] == "4096" and "-ngl" not in arguments

    first = status["pid"]
    gateway.chat({"model": entry["id"], "message": "again"})
    assert gateway.llama.status()["pid"] == first  # the loaded model is reused

    settings["llama_gpu_layers"] = 12
    gateway.chat({"model": entry["id"], "message": "bigger", "n_ctx": 2048})
    assert gateway.llama.status()["pid"] != first  # new settings need a new process
    arguments = launch_arguments(tmp_path)
    assert arguments[arguments.index("-c") + 1] == "2048"
    assert arguments[arguments.index("-ngl") + 1] == "12"

    assert gateway.release_gpu()["llama_server_stopped"] is True
    assert gateway.llama.status()["running"] is False


def test_a_model_llama_cpp_cannot_load_reports_its_output(workspace):
    tmp_path, gateway, _ = workspace
    (tmp_path / "models" / "broken-Q4_0.gguf").write_bytes(b"x")
    entry = next(item for item in gateway.catalog(force=True) if item["name"] == "broken-Q4_0.gguf")
    with pytest.raises(UnavailableError) as failure:
        gateway.chat({"model": entry["id"], "message": "hello"})
    assert "could not load broken-Q4_0.gguf" in str(failure.value)
    assert "unknown model architecture" in str(failure.value)
    assert gateway.llama.status() == {
        "running": False, "model": "", "url": "", "loading": "", "pid": None,
        "idle_seconds": None, "context": None,
    }


def test_llama_server_needs_a_binary_and_picks_a_sane_context(tmp_path, monkeypatch):
    from mariana.studio.models import discovery

    monkeypatch.setattr(discovery, "LLAMA_SERVER_CANDIDATES", [])
    monkeypatch.setattr(discovery.shutil, "which", lambda name: None)
    monkeypatch.delenv("MARIANA_LLAMA_SERVER", raising=False)
    monkeypatch.delenv("LLAMA_CPP_DIR", raising=False)
    server = LlamaServer(tmp_path)
    try:
        with pytest.raises(UnavailableError, match="llama.cpp is not installed"):
            server.ensure(str(tmp_path / "model.gguf"), 4096)
        assert server.context_for(None, None) == 4096
        assert server.context_for(None, 131072) == 8192   # huge native contexts are capped
        assert server.context_for(None, 2048) == 2048
        assert server.context_for("16384", 2048) == 16384  # an explicit request wins
        assert server.context_for("nonsense", 0) == 4096
    finally:
        server.close()


def test_ollama_can_be_started_and_stopped(tmp_path, monkeypatch):
    with socket.socket() as handle:
        handle.bind(("127.0.0.1", 0))
        port = handle.getsockname()[1]
    monkeypatch.setenv("OLLAMA_HOST", f"127.0.0.1:{port}")
    server = OllamaServer(tmp_path)
    try:
        assert server.status() == {"running": False, "managed": False, "url": f"http://127.0.0.1:{port}"}
        with pytest.raises(UnavailableError, match="not installed"):
            server.start(None)
        started = server.start(str(executable(tmp_path / "ollama", FAKE_OLLAMA)))
        assert started["running"] is True and started["managed"] is True
        assert server.start(None)["running"] is True  # already up: nothing to launch
        assert server.stop() is True and server.running() is False
    finally:
        server.close()
