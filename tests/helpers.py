"""Shared, dependency-free helpers for the test suite.

Fixtures live in ``conftest.py``; plain functions that tests call directly live
here and are imported as ``from tests.helpers import wait_for``.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def wait_for(predicate, timeout: float = 10.0, interval: float = 0.05):
    """Poll ``predicate`` until it returns a truthy value, then return it."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(interval)
    raise TimeoutError("condition was not met")


def request(url: str, payload=None, method: str | None = None, headers: dict | None = None,
            timeout: float = 30):
    """Call a JSON HTTP endpoint and return the decoded body.

    JSON responses are parsed; anything else is returned as text.  HTTP errors
    propagate as ``urllib.error.HTTPError`` so tests can assert on status codes.
    """
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json", **(headers or {})},
        method=method or ("POST" if data is not None else "GET"),
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        content_type = response.headers.get("Content-Type", "")
        body = response.read()
    return json.loads(body) if "application/json" in content_type else body.decode()


def request_error(url: str, payload=None, method: str | None = None,
                  headers: dict | None = None) -> tuple[int, dict]:
    """Return ``(status, json_body)`` for a request that is expected to fail."""
    try:
        request(url, payload, method=method, headers=headers)
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        try:
            return error.code, json.loads(body)
        except json.JSONDecodeError:
            return error.code, {"error": body}
    raise AssertionError(f"{url} was expected to fail")


@contextmanager
def serving(server):
    """Run any ``socketserver`` instance on a daemon thread for the block."""
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    try:
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


class RecordingHandler(BaseHTTPRequestHandler):
    """Base class for mock providers: records requests and answers from a route table.

    ``routes`` maps ``(method, path)`` to ``callable(handler, payload)``; a path ending in
    ``*`` matches by prefix.  The callable returns ``(status, body)`` or
    ``(status, body, headers)`` where ``body`` is a dict/list (sent as JSON), ``bytes``,
    or a list of ``str`` chunks that are streamed one by one.
    """

    routes: dict = {}
    calls: list = []

    def _payload(self):
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            return json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            return {"_raw": raw.decode("utf-8", errors="replace")}

    def _route(self, method: str, path: str):
        handler = self.routes.get((method, path))
        if handler is None:
            for (route_method, pattern), candidate in self.routes.items():
                if route_method == method and pattern.endswith("*") and path.startswith(pattern[:-1]):
                    return candidate
        return handler

    def _dispatch(self, method: str):
        payload = self._payload() if method != "GET" else {}
        path = self.path.split("?", 1)[0]
        type(self).calls.append({
            "method": method, "path": self.path, "payload": payload,
            "headers": {key.lower(): value for key, value in self.headers.items()},
        })
        handler = self._route(method, path)
        result = (404, {"error": f"no mock route for {method} {path}"}) if handler is None \
            else handler(self, payload)
        status, body = result[0], result[1]
        headers = dict(result[2]) if len(result) > 2 else {}
        if isinstance(body, list) and body and all(isinstance(item, str) for item in body):
            self.send_response(status)
            self.send_header("Content-Type", headers.pop("Content-Type", "text/event-stream"))
            for name, value in headers.items():
                self.send_header(name, value)
            self.end_headers()
            try:
                for chunk in body:
                    self.wfile.write(chunk.encode("utf-8"))
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            return
        data = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", headers.pop(
            "Content-Type", "application/octet-stream" if isinstance(body, bytes) else "application/json"))
        self.send_header("Content-Length", str(len(data)))
        for name, value in headers.items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):  # noqa: N802 - http.server naming
        self._dispatch("GET")

    def do_POST(self):  # noqa: N802
        self._dispatch("POST")

    def do_DELETE(self):  # noqa: N802
        self._dispatch("DELETE")

    def log_message(self, *args):
        return


def mock_provider(routes: dict):
    """Create an unstarted mock HTTP provider; use it with :func:`serving`."""
    handler = type("MockProvider", (RecordingHandler,), {"routes": routes, "calls": []})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    server.calls = handler.calls
    return server


CHAT_TEMPLATE = (
    "{% for message in messages %}<|{{ message['role'] }}|>{{ message['content'] }}<eos>"
    "{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}"
)


def make_tiny_transformer(folder: Path, chat_template: bool = True) -> Path:
    """Write a random two-layer GPT-2 with a locally trained tokenizer (no downloads).

    Requires ``transformers`` and ``tokenizers``; callers guard with ``pytest.importorskip``.
    """
    import torch
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=320,
        special_tokens=["<unk>", "<eos>", "<pad>", "<|system|>", "<|user|>", "<|assistant|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    sample = "The quick brown fox jumps over the lazy dog. Hello there, how are you today? "
    tokenizer.train_from_iterator([sample] * 20, trainer)
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="<unk>", eos_token="<eos>", pad_token="<pad>",
    )
    if chat_template:
        fast.chat_template = CHAT_TEMPLATE
    fast.save_pretrained(folder)
    config = GPT2Config(
        vocab_size=len(fast), n_positions=64, n_embd=32, n_layer=2, n_head=2,
        bos_token_id=fast.eos_token_id, eos_token_id=fast.eos_token_id,
        pad_token_id=fast.pad_token_id,
    )
    torch.manual_seed(0)
    GPT2LMHeadModel(config).save_pretrained(folder, safe_serialization=True)
    return folder


def make_corpus(folder: Path) -> Path:
    """A small mixed-format corpus: markdown, JSON, JSONL, XML, extensionless text and a binary."""
    folder = Path(folder)
    (folder / "nested").mkdir(parents=True)
    (folder / "notes.md").write_text("# Notes\nPlain text and **Markdown**.", encoding="utf-8")
    (folder / "nested" / "data.json").write_text(
        json.dumps({"text": "JSON text", "nested": ["More text"]}), encoding="utf-8"
    )
    (folder / "rows.jsonl").write_text('{"text":"First row"}\n{"text":"Second row"}\n', encoding="utf-8")
    (folder / "page.xml").write_text("<root><item>XML text</item><item>Another value</item></root>", encoding="utf-8")
    (folder / "extensionless").write_text("Text without an extension", encoding="utf-8")
    (folder / "binary.bin").write_bytes(b"\x00\x01\x02\x03")
    return folder


@contextmanager
def studio(root: Path):
    """Run a Studio server for ``root`` on a free port; yields ``(server, base_url)``."""
    from mariana.studio import make_server

    server = make_server("127.0.0.1", 0, Path(root))
    with serving(server) as base:
        yield server, base


def wait_for_job(base: str, progress: bool = False, timeout: float = 90) -> dict:
    """Poll the training job until it finishes (or, with ``progress``, takes its first step)."""
    deadline = time.time() + timeout
    job = request(base + "/api/job")["job"]
    while time.time() < deadline:
        if progress and job["current_step"] >= 1:
            return job
        if not job["active"]:
            return job
        time.sleep(0.05)
        job = request(base + "/api/job")["job"]
    raise TimeoutError("\n".join(job["logs"]))


def tiny_drv_checkpoint(path: Path, step: int = 3, **overrides) -> Path:
    """Save a random, tiny DRV-LM checkpoint (CPU, well under a megabyte)."""
    import torch

    from mariana.config import Config
    from mariana.model import MarianaLM

    settings = dict(ctx_len=32, dim=32, n_heads=4, n_pre=1, n_core=1, n_post=1,
                    max_loops=1, block_size=8, use_mtp=False)
    settings.update(overrides)
    cfg = Config(**settings)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"step": step, "model": MarianaLM(cfg).state_dict(), "cfg": cfg.to_dict(), "format": 2}, path)
    return path


def tiny_ncm_checkpoints(folder: Path, name: str = "ncm", **overrides) -> tuple[Path, Path]:
    """Save a tiny NCM tokenizer + generator pair; returns ``(generator, tokenizer)`` paths."""
    import torch

    from mariana.device import Amp
    from mariana.ncm.config import NCMConfig
    from mariana.ncm.generator import NCMGenerator
    from mariana.ncm.tokenizer import MultiscaleTokenizer
    from mariana.ncm.train import save_checkpoint

    settings = dict(ctx_len=32, prefix_len=16, levels=(1, 4, 8, 16, 32), latent_dim=32,
                    codebook_size=64, dim=32, n_heads=4, encoder_layers=1, decoder_layers=1,
                    refiner_layers=0, generator_layers=1)
    settings.update(overrides)
    cfg = NCMConfig(**settings)
    tokenizer = MultiscaleTokenizer(cfg)
    generator = NCMGenerator(tokenizer, cfg)
    amp = Amp(torch.device("cpu"))
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    tokenizer_path, generator_path = folder / f"{name}-tokenizer.pt", folder / f"{name}.pt"
    save_checkpoint(str(tokenizer_path), tokenizer,
                    torch.optim.AdamW(tokenizer.parameters(), lr=1e-3), cfg, 1, amp, "tokenizer")
    save_checkpoint(str(generator_path), generator,
                    torch.optim.AdamW([p for p in generator.parameters() if p.requires_grad], lr=1e-3),
                    cfg, 2, amp, "generator")
    return generator_path, tokenizer_path


def read_events(url: str, payload=None, limit: int = 50, timeout: float = 30) -> list:
    """Read server-sent events until the stream ends or ``limit`` events arrived.

    JSON payloads are decoded; anything else (``[DONE]``) is returned as text.
    """
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST" if data is not None else "GET",
    )
    events = []
    with urllib.request.urlopen(req, timeout=timeout) as response:
        assert response.headers.get("Content-Type", "").startswith("text/event-stream")
        for raw in response:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            text = line[5:].strip()
            try:
                events.append(json.loads(text))
            except json.JSONDecodeError:
                events.append(text)
            if len(events) >= limit:
                break
    return events
