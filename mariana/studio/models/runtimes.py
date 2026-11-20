"""Processes Studio starts on the user's behalf: llama-server and ``ollama serve``.

Both are children of Studio: they are started on demand, logged to
``agent-workspace/logs/``, and stopped when Studio exits.  The llama.cpp server
holds one model at a time (it owns the GPU) and unloads after an idle period.
"""

from __future__ import annotations

import os
import socket
import subprocess
import threading
import time
from pathlib import Path

from ...workspace import ensure_workspace
from ..errors import UnavailableError
from .discovery import fetch_json, find_llama_server, ollama_url

LOAD_TIMEOUT = 900.0
DEFAULT_IDLE_MINUTES = 10
DEFAULT_CONTEXT = 4096
MAX_AUTO_CONTEXT = 8192


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        return handle.getsockname()[1]


def _log_path(root: Path, name: str) -> Path:
    folder = ensure_workspace(root) / "logs"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / name


def _tail(path: Path, lines: int = 12) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    kept = [line for line in text.splitlines() if line.strip()][-lines:]
    return "\n".join(kept)


def _stop(process: subprocess.Popen | None, grace: float = 5.0) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        process.kill()
        try:
            process.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            pass


class LlamaServer:
    """A managed ``llama-server`` serving one GGUF model over the OpenAI API."""

    def __init__(self, root: Path, settings=None):
        self.root = Path(root).resolve()
        self._settings = settings or (lambda: {})
        self._lock = threading.Lock()      # guards state
        self._start_lock = threading.Lock()  # serialises model loads
        self._process: subprocess.Popen | None = None
        self._key: tuple | None = None
        self._url = ""
        self._model = ""
        self._loading = ""
        self._last_used = 0.0
        self._busy = 0
        self._closed = threading.Event()
        self._watchdog = threading.Thread(target=self._watch, name="llama-idle", daemon=True)
        self._watchdog.start()

    # -- public ----------------------------------------------------------
    def context_for(self, requested, model_context) -> int:
        """Pick a context size: the request, else the model's (capped), else a default."""
        try:
            value = int(requested or 0)
        except (TypeError, ValueError):
            value = 0
        if value > 0:
            return max(256, min(value, 1_048_576))
        try:
            native = int(model_context or 0)
        except (TypeError, ValueError):
            native = 0
        return max(256, min(native, MAX_AUTO_CONTEXT)) if native else DEFAULT_CONTEXT

    def ensure(self, model_path: str, n_ctx: int, on_status=None, cancelled=None,
               label: str | None = None) -> str:
        """Start (or reuse) the server for ``model_path`` and return its base URL.

        ``label`` is the name shown to the user; Ollama blobs are named by hash on disk.
        """
        settings = self._settings() or {}
        binary = find_llama_server(settings)
        if not binary:
            raise UnavailableError(
                "llama.cpp is not installed. Build it so `llama-server` is on PATH, set its "
                "path in Settings, or run: pip install llama-cpp-python"
            )
        gpu_layers = settings.get("llama_gpu_layers")
        key = (str(model_path), int(n_ctx), str(gpu_layers), binary)
        with self._start_lock:
            with self._lock:
                alive = self._process is not None and self._process.poll() is None
                if alive and self._key == key:
                    self._last_used = time.time()
                    return self._url
            self.stop()
            return self._start(binary, key, on_status, cancelled, label)

    def acquire(self) -> None:
        with self._lock:
            self._busy += 1
            self._last_used = time.time()

    def release(self) -> None:
        with self._lock:
            self._busy = max(0, self._busy - 1)
            self._last_used = time.time()

    def stop(self) -> bool:
        with self._lock:
            process, self._process = self._process, None
            self._key, self._url, self._model, self._loading = None, "", "", ""
        _stop(process)
        return process is not None

    def status(self) -> dict:
        with self._lock:
            running = self._process is not None and self._process.poll() is None
            return {
                "running": running, "model": self._model if running else "",
                "url": self._url if running else "", "loading": self._loading,
                "pid": self._process.pid if running else None,
                "idle_seconds": round(time.time() - self._last_used, 1) if running else None,
                "context": self._key[1] if running and self._key else None,
            }

    def close(self) -> None:
        self._closed.set()
        self.stop()

    # -- internals -------------------------------------------------------
    def _start(self, binary: str, key: tuple, on_status, cancelled, label: str | None = None) -> str:
        model_path, n_ctx, gpu_layers, _ = key
        if not Path(model_path).is_file():
            raise UnavailableError(f"Model file not found: {model_path}")
        port = _free_port()
        url = f"http://127.0.0.1:{port}"
        command = [binary, "-m", model_path, "--host", "127.0.0.1", "--port", str(port),
                   "-c", str(n_ctx)]
        if gpu_layers not in (None, "", "auto", "None"):
            command += ["-ngl", str(gpu_layers)]
        log = _log_path(self.root, "llama-server.log")
        name = label or Path(model_path).name
        with self._lock:
            self._loading = name
        try:
            with log.open("w", encoding="utf-8") as handle:
                handle.write("$ " + " ".join(command) + "\n")
                handle.flush()
                process = subprocess.Popen(
                    command, stdout=handle, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                    env={**os.environ, "LLAMA_ARG_NO_WEBUI": "1"},
                )
            started = time.time()
            while True:
                if process.poll() is not None:
                    raise UnavailableError(
                        f"llama.cpp could not load {name}:\n{_tail(log) or 'no output'}"
                    )
                if cancelled is not None and cancelled.is_set():
                    _stop(process)
                    raise UnavailableError("Model loading was cancelled")
                health = fetch_json(f"{url}/health", timeout=1.0)
                if isinstance(health, dict) and health.get("status") == "ok":
                    break
                waited = time.time() - started
                if waited > LOAD_TIMEOUT:
                    _stop(process)
                    raise UnavailableError(f"llama.cpp took too long to load {name}")
                if on_status:
                    on_status(f"Loading {name} into llama.cpp… {waited:.0f}s")
                time.sleep(0.4)
        except BaseException:
            with self._lock:
                self._loading = ""
            raise
        with self._lock:
            self._process, self._key, self._url = process, key, f"{url}/v1"
            self._model, self._loading, self._last_used = name, "", time.time()
        return f"{url}/v1"

    def _watch(self) -> None:
        while not self._closed.wait(20.0):
            settings = self._settings() or {}
            try:
                minutes = float(settings.get("idle_unload_minutes", DEFAULT_IDLE_MINUTES))
            except (TypeError, ValueError):
                minutes = DEFAULT_IDLE_MINUTES
            if minutes <= 0:
                continue
            with self._lock:
                idle = (
                    self._process is not None and self._busy == 0
                    and time.time() - self._last_used > minutes * 60
                )
            if idle:
                self.stop()


class OllamaServer:
    """Starts ``ollama serve`` when the user asks, and stops it on exit."""

    def __init__(self, root: Path, settings=None):
        self.root = Path(root).resolve()
        self._settings = settings or (lambda: {})
        self._lock = threading.Lock()
        self._process: subprocess.Popen | None = None

    def url(self) -> str:
        return ollama_url(self._settings() or {})

    def running(self) -> bool:
        return isinstance(fetch_json(f"{self.url()}/api/version", timeout=1.0), dict)

    def start(self, binary: str | None, timeout: float = 30.0) -> dict:
        if self.running():
            return self.status()
        if not binary:
            raise UnavailableError("Ollama is not installed; get it from https://ollama.com/download")
        log = _log_path(self.root, "ollama.log")
        with self._lock:
            if self._process is None or self._process.poll() is not None:
                with log.open("w", encoding="utf-8") as handle:
                    self._process = subprocess.Popen(
                        [binary, "serve"], stdout=handle, stderr=subprocess.STDOUT,
                        stdin=subprocess.DEVNULL,
                    )
            process = self._process
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.running():
                return self.status()
            if process.poll() is not None:
                raise UnavailableError(f"Ollama exited while starting:\n{_tail(log) or 'no output'}")
            time.sleep(0.3)
        raise UnavailableError("Ollama did not become ready in time; check agent-workspace/logs/ollama.log")

    def stop(self) -> bool:
        with self._lock:
            process, self._process = self._process, None
        _stop(process)
        return process is not None

    def status(self) -> dict:
        with self._lock:
            managed = self._process is not None and self._process.poll() is None
        return {"running": self.running(), "managed": managed, "url": self.url()}

    def close(self) -> None:
        self.stop()
