"""The model gateway: one catalog and one chat API over every backend.

``ModelGateway`` lists models from the workspace, from tools already installed
on the machine and from saved endpoints, and runs chats against any of them,
streamed or not.  It owns the caches and the managed runtimes, and it is the
only object the HTTP layer and the agent runtime talk to about models.
"""

from __future__ import annotations

import gc
import threading
import time
from collections import OrderedDict
from pathlib import Path

from ...device import describe as describe_device
from ...device import empty_cache, pick_device
from ...workspace import ensure_workspace
from ..errors import ConflictError
from ..validation import number
from . import catalog as catalog_module
from .backends import EMPTY_REPLY, ChatRequest
from .backends import claude, huggingface, llama, ollama, onnx, openai
from .backends import mariana as mariana_backend
from .catalog import ModelError, checkpoint_entry, gguf_detail, scan_folder
from .context import entry_context, positive_int
from .discovery import Discovery, find_llama_server, gguf_runtime
from .endpoints import EndpointStore, endpoint_entry, normalize_base_url, providers, resolve_key
from .runtimes import LlamaServer, OllamaServer

MAX_CONTEXT_MESSAGES = 40
MAX_MESSAGE_CHARS = 24000
MAX_LOCAL_MODELS = 2
WORKSPACE_CACHE_SECONDS = 2.0
CONTEXT_CACHE_SECONDS = 300.0
CONTEXT_PROBE_TIMEOUT = 2.5
WORKSPACE_FOLDERS = ("trained-model", "models", "exports")
BACKENDS = {
    "mariana": mariana_backend.stream,
    "transformers": huggingface.stream,
    "llama-server": llama.stream,
    "llama-cpp-python": llama.stream,
    "onnxruntime": onnx.stream,
    "openai": openai.stream,
    "ollama": ollama.stream,
    "anthropic": claude.stream,
}
# Runtimes that load weights into this process or onto this machine's GPU.
LOCAL_RUNTIMES = frozenset({"mariana", "transformers", "llama-server", "llama-cpp-python", "onnxruntime"})
DEFAULT_MAX_TOKENS = {"mariana": 80, "anthropic": claude.DEFAULT_MAX_TOKENS, "openai": 4_096}
FALLBACK_MAX_TOKENS = 1024


class ModelGateway:
    """Catalog + chat execution for every supported backend."""

    def __init__(self, root: Path, settings=None, own_port=None):
        self.root = Path(root).resolve()
        ensure_workspace(self.root)
        self._settings = settings or (lambda: {})
        self.endpoints = EndpointStore(self.root)
        self.discovery = Discovery(self.root, self._settings, own_port)
        self.llama = LlamaServer(self.root, self._settings)
        self.ollama = OllamaServer(self.root, self._settings)
        self.local_lock = threading.Lock()   # one in-process generation at a time
        self._cache_lock = threading.Lock()
        self._runtime_cache: OrderedDict[tuple, object] = OrderedDict()
        self._workspace_cache: tuple[float, list[dict]] | None = None
        self._context_cache: dict[str, tuple[float, int, str]] = {}
        self._context_lock = threading.Lock()
        self.training_active = lambda: False

    @property
    def settings(self) -> dict:
        return self._settings() or {}

    # -- loaded-model cache ----------------------------------------------
    def is_cached(self, key: tuple) -> bool:
        with self._cache_lock:
            return key in self._runtime_cache

    def cached(self, key: tuple, loader):
        """Return the loaded model for ``key``, loading and evicting as needed."""
        with self._cache_lock:
            if key in self._runtime_cache:
                self._runtime_cache.move_to_end(key)
                return self._runtime_cache[key]
        value = loader()
        with self._cache_lock:
            self._runtime_cache[key] = value
            evicted = []
            while len(self._runtime_cache) > MAX_LOCAL_MODELS:
                evicted.append(self._runtime_cache.popitem(last=False))
        if evicted:
            del evicted
            self._empty_device_cache()
        return value

    @staticmethod
    def _empty_device_cache() -> None:
        try:
            empty_cache(pick_device("auto"))
        except Exception:  # noqa: BLE001 - freeing memory is best effort
            pass

    def release_gpu(self) -> dict:
        """Unload every local model (called before training or on shutdown)."""
        with self._cache_lock:
            count = len(self._runtime_cache)
            self._runtime_cache.clear()
        try:
            stopped = self.llama.stop()
        except Exception:  # noqa: BLE001 - freeing memory is best effort
            stopped = False
        gc.collect()
        self._empty_device_cache()
        return {"unloaded": count, "llama_server_stopped": stopped}

    def loaded(self) -> list[dict]:
        with self._cache_lock:
            keys = list(self._runtime_cache)
        models = [{"runtime": key[0], "path": key[1]} for key in keys]
        server = self.llama.status()
        if server["running"]:
            models.append({"runtime": "llama-server", "path": server["model"]})
        return models

    # -- catalog ---------------------------------------------------------
    def invalidate(self) -> None:
        self._workspace_cache = None
        with self._context_lock:
            self._context_cache.clear()
        self.discovery.invalidate()

    def _workspace_entries(self, force: bool) -> tuple[list[dict], set[Path]]:
        now = time.time()
        cached = self._workspace_cache
        if cached and not force and now - cached[0] < WORKSPACE_CACHE_SECONDS:
            return [dict(entry) for entry in cached[1]], set()
        runtime = gguf_runtime(self.settings)
        seen: set[Path] = set()
        entries: list[dict] = []
        for name in WORKSPACE_FOLDERS:
            entries += scan_folder(self.root, self.root / name, "workspace", runtime, seen)
        for path in sorted(self.root.glob("*.pt")):
            if path.resolve() not in seen and not path.name.endswith(".tmp"):
                seen.add(path.resolve())
                entries.append(checkpoint_entry(self.root, path))
        self._workspace_cache = (now, entries)
        return [dict(entry) for entry in entries], seen

    def catalog(self, force: bool = False) -> list[dict]:
        """Every model Studio can see, newest local models first."""
        entries, _ = self._workspace_entries(force)
        known = {entry["absolute_path"] for entry in entries if entry["absolute_path"]}
        saved = self.endpoints.load()
        saved_targets = {
            (endpoint.get("base_url", "").rstrip("/"), endpoint.get("model")) for endpoint in saved
        }
        for entry in self.discovery.scan(force)["entries"]:
            if entry["absolute_path"] and entry["absolute_path"] in known and entry["kind"] != "ollama":
                continue
            target = ((entry["detail"].get("base_url") or "").rstrip("/"), entry["detail"].get("model"))
            if entry["source"] == "server" and target in saved_targets:
                continue  # the user already saved this exact server model
            entries.append(dict(entry))
        entries += [endpoint_entry(endpoint) for endpoint in saved]
        for entry in entries:
            entry["defaults"] = {"max_tokens": self._default_max_tokens(entry)}
        entries.sort(key=lambda item: (-(item["modified"] or 0), item["name"].lower()))
        return entries

    def _default_max_tokens(self, entry: dict) -> int:
        default = DEFAULT_MAX_TOKENS.get(entry.get("runtime"), FALLBACK_MAX_TOKENS)
        limit = (entry.get("detail") or {}).get("max_output_tokens")
        return min(default, int(limit)) if limit else default

    def context_window(self, model_id: str | None = None, force: bool = False) -> dict:
        """The largest context window known for a model.

        Ask the catalog, then the provider or local runtime at most once per
        cache window (or on ``force``), keeping whichever value is larger.
        With no model chosen, resolve the same first available chat model a
        run would use.
        """
        entry = None
        if model_id:
            try:
                entry = self.get(model_id)
            except ModelError:
                entry = None
        else:
            catalog = self.catalog()
            chat = [item for item in catalog
                    if item.get("available") and (item.get("capabilities") or {}).get("chat")]
            entry = chat[0] if chat else (catalog[0] if catalog else None)
        if entry is None:
            return {"id": str(model_id or ""), "name": "", "kind": "", "found": False,
                    "context_window": 0, "source": "not-found", "unit": "tokens"}
        detail = entry.get("detail") or {}
        kind = str(entry.get("kind") or "")
        unit = "bytes" if detail.get("ctx_unit") == "bytes" or kind in {
            "mariana-drv", "mariana-ncm",
        } else "tokens"
        window = entry_context(entry)
        source = "catalog" if window else "unknown"
        probeable = kind in {"openai", "ollama", "anthropic"} and bool(entry.get("available"))
        if probeable:
            cache_key = str(entry.get("id") or "")
            now = time.time()
            with self._context_lock:
                cached = self._context_cache.get(cache_key)
            if cached and not force and now - cached[0] < CONTEXT_CACHE_SECONDS:
                live, live_source = cached[1], cached[2]
            else:
                try:
                    live, live_source = self._probe_context(entry)
                except Exception:  # noqa: BLE001 - a probe must not fail the budget
                    live, live_source = 0, "unavailable"
                with self._context_lock:
                    self._context_cache[cache_key] = (now, live, live_source)
            if live > window:
                window, source = live, live_source
            elif window <= 0:
                window, source = live, live_source
        return {
            "id": str(entry.get("id") or model_id or ""),
            "name": str(entry.get("name") or ""),
            "kind": kind,
            "found": True,
            "context_window": window,
            "source": source,
            "unit": unit,
        }

    def _probe_context(self, entry: dict) -> tuple[int, str]:
        """Ask a saved endpoint, a local server or Ollama for its window."""
        detail = entry.get("detail") or {}
        kind = str(entry.get("kind") or "")
        if not entry.get("available"):
            return 0, "unavailable"
        saved = str(entry.get("id") or "").startswith("endpoint:")
        if kind == "ollama":
            if detail.get("server_stopped") and not saved:
                return 0, "unavailable"
            model = detail.get("model") or entry.get("name") or ""
            base_url = detail.get("base_url") or "http://127.0.0.1:11434"
            return ollama.context_length(base_url, model, timeout=CONTEXT_PROBE_TIMEOUT), "runtime"
        if kind == "openai":
            headers = {}
            if saved:
                endpoint = self.endpoints.get(entry["id"])
                base_url = endpoint["base_url"]
                model = endpoint["model"]
                headers = dict(endpoint.get("headers") or {})
                key = resolve_key(endpoint)
                if key:
                    headers.setdefault("Authorization", f"Bearer {key}")
            else:
                base_url = detail.get("base_url") or ""
                model = detail.get("model") or entry.get("name") or ""
            window = openai.context_length(base_url, model, headers, timeout=CONTEXT_PROBE_TIMEOUT)
            return window, "endpoint" if saved or detail.get("remote") else "runtime"
        if kind == "anthropic" and saved:
            endpoint = self.endpoints.get(entry["id"])
            model = detail.get("model") or entry.get("name") or ""
            models = claude.list_models(endpoint)
            match = next((item for item in models if item.get("id") == model), None)
            return (positive_int(match.get("context_length")) if match else 0), "endpoint"
        return 0, "unavailable"

    def get(self, model_id: str | None = None, checkpoint: str | None = None) -> dict:
        """Resolve a model id (or a legacy checkpoint path) to a catalog entry."""
        entries = self.catalog()
        if checkpoint:
            wanted = str(Path(checkpoint))
            for entry in entries:
                if wanted in (entry["path"], entry["id"], entry["name"]):
                    return entry
            candidate = Path(checkpoint)
            if not candidate.is_absolute():
                candidate = self.root / candidate
            if candidate.exists() and candidate.suffix == ".pt":
                return checkpoint_entry(self.root, candidate)
            raise ModelError("Select a model from the catalog")
        if not model_id:
            if len(entries) == 1:
                return entries[0]
            raise ModelError("Select a model first")
        for attempt in range(2):
            for entry in entries:
                if entry["id"] == model_id:
                    return entry
            named = [entry for entry in entries if entry["name"] == model_id]
            if len(named) == 1:
                return named[0]
            if attempt == 0:
                entries = self.catalog(force=True)  # it may have been created a moment ago
        raise ModelError("Model not found; refresh the library")

    def describe(self, model_id: str | None = None, checkpoint: str | None = None) -> dict:
        """A catalog entry with the more expensive details filled in."""
        entry = dict(self.get(model_id, checkpoint))
        if entry["absolute_path"] and (entry["kind"] == "gguf" or entry["detail"].get("blob")):
            entry["detail"] = gguf_detail(entry)
        return entry

    def runtimes(self, force: bool = False) -> dict:
        """Status of the local runtimes Studio can use or manage."""
        status = dict(self.discovery.scan(force)["runtimes"])
        status["llama_cpp"] = {**status["llama_cpp"], "managed": self.llama.status()}
        if "ollama" in status:
            status["ollama"] = {**status["ollama"], "managed": self.ollama.status()["managed"]}
        status["loaded"] = self.loaded()
        return status

    # -- endpoints -------------------------------------------------------
    def add_endpoint(self, config: dict) -> list[dict]:
        self.endpoints.upsert(config)
        self.invalidate()
        return self.endpoints.public()

    def delete_endpoint(self, endpoint_id: str) -> list[dict]:
        if not self.endpoints.delete(endpoint_id):
            raise ModelError("Endpoint not found")
        self.invalidate()
        return self.endpoints.public()

    def providers(self) -> list[dict]:
        return providers()

    def provider_models(self, config: dict) -> list[dict]:
        """List the models a server offers, before or after it is saved as an endpoint."""
        kind = config.get("kind", "openai")
        endpoint = {
            "kind": kind, "base_url": normalize_base_url(kind, config.get("base_url")),
            "api_key": str(config.get("api_key") or "").strip(),
            "api_key_env": str(config.get("api_key_env") or "").strip(),
            "headers": config.get("headers") if isinstance(config.get("headers"), dict) else {},
        }
        if not endpoint["api_key"] and config.get("id"):
            try:
                endpoint["api_key"] = self.endpoints.get(str(config["id"])).get("api_key", "")
            except ModelError:
                pass
        if kind == "anthropic":
            return claude.list_models(endpoint)
        if kind == "ollama":
            return ollama.list_models(endpoint["base_url"], endpoint["headers"])
        headers = dict(endpoint["headers"])
        key = resolve_key(endpoint)
        if key:
            headers.setdefault("Authorization", f"Bearer {key}")
        return openai.list_models(endpoint["base_url"], headers)

    def test_endpoint(self, endpoint_id: str) -> dict:
        entry = self.get(endpoint_id)
        result = self.chat({
            "model": entry["id"],
            "messages": [{"role": "user", "content": "Reply with the single word: ready"}],
            "temperature": 0, "max_tokens": 512 if entry["kind"] == "anthropic" else 16,
        })
        return {"ok": True, "reply": result["reply"], "kind": result["kind"],
                "elapsed": result["elapsed"]}

    # -- runtimes --------------------------------------------------------
    def start_ollama(self) -> dict:
        import shutil

        status = self.ollama.start(shutil.which("ollama"))
        self.invalidate()
        return status

    def stop_ollama(self) -> dict:
        self.ollama.stop()
        self.invalidate()
        return self.ollama.status()

    # -- chat ------------------------------------------------------------
    def _messages(self, config: dict) -> list[dict]:
        raw = config.get("messages")
        messages: list[dict] = []
        if isinstance(raw, list):
            for item in raw[-MAX_CONTEXT_MESSAGES:]:
                if isinstance(item, dict) and item.get("role") in {"user", "assistant", "system"}:
                    content = _content_text(item.get("content"))[:MAX_MESSAGE_CHARS]
                    if content:
                        messages.append({"role": item["role"], "content": content})
        else:
            for item in (config.get("history") or [])[-MAX_CONTEXT_MESSAGES:]:
                if isinstance(item, dict) and item.get("role") in {"user", "assistant"}:
                    content = _content_text(item.get("content"))[:MAX_MESSAGE_CHARS]
                    if content:
                        messages.append({"role": item["role"], "content": content})
            message = str(config.get("message", "")).strip()
            if message:
                messages.append({"role": "user", "content": message[:MAX_MESSAGE_CHARS]})
        if not messages or messages[-1]["role"] != "user":
            raise ModelError("Enter a message first")
        system = str(config.get("system_prompt", "") or "").strip()
        if system and messages[0]["role"] != "system":
            messages.insert(0, {"role": "system", "content": system[:MAX_MESSAGE_CHARS]})
        return messages

    def prepare(self, config: dict, cancelled: threading.Event | None = None):
        """Validate a chat request; returns ``(entry, request)`` or raises."""
        messages = self._messages(config)
        entry = self.get(config.get("model"), config.get("checkpoint"))
        if not entry["available"]:
            raise ModelError(entry.get("reason") or "This model backend is unavailable")
        if not entry["capabilities"].get("chat"):
            raise ModelError(entry.get("reason") or "This checkpoint cannot chat")
        runtime = entry.get("runtime") or entry["kind"]
        if runtime not in BACKENDS:
            raise ModelError(f"Unsupported model kind: {entry['kind']}")
        if runtime in LOCAL_RUNTIMES and self.training_active():
            raise ConflictError(
                "A training run is using this machine. Wait for it to finish (or pause it) "
                "before chatting with a local model; remote models stay available."
            )
        options = dict(config)
        provider = str((entry.get("detail") or {}).get("provider") or "")
        if provider:
            options["provider"] = provider
        request = ChatRequest(
            messages=messages,
            temperature=number(config, "temperature", 0.8, 0.0, 5.0, float, ModelError),
            top_p=number(config, "top_p", 0.95, 0.01, 1.0, float, ModelError),
            top_k=number(config, "top_k", 0, 0, 65536, int, ModelError),
            max_tokens=number(config, "max_tokens", self._default_max_tokens(entry), 1, 131072,
                              int, ModelError),
            device=str(config.get("device") or "auto"),
            options=options,
            cancelled=cancelled or threading.Event(),
        )
        return entry, request

    def stream(self, config: dict, cancelled: threading.Event | None = None):
        """Yield chat events; the last one is ``{"type": "done", ...}`` with the full reply."""
        entry, request = self.prepare(config, cancelled)
        runtime = entry.get("runtime") or entry["kind"]
        started = time.time()
        first_token = None  # the first output of any kind, reasoning included
        produced = 0
        parts: list[str] = []
        thoughts: list[str] = []
        final: dict = {}
        yield {"type": "start", "model_id": entry["id"], "model": entry["path"] or entry["name"],
               "name": entry["name"], "kind": entry["kind_label"]}
        events = BACKENDS[runtime](self, entry, request)
        try:
            for event in events:
                kind = event.get("type")
                if kind in {"delta", "reasoning"}:
                    if first_token is None:
                        first_token = time.time()
                    produced += 1
                    (parts if kind == "delta" else thoughts).append(event["text"])
                    yield event
                elif kind == "status":
                    yield event
                elif kind == "done":
                    final = event
        finally:
            request.cancelled.set()  # stop any worker thread a backend still runs
            close = getattr(events, "close", None)
            if close:
                close()
        ended = time.time()
        reply = "".join(parts).strip()
        usage = dict(final.get("usage") or {})
        completion = int(usage.get("completion_tokens") or 0)
        if not completion and reply:
            completion = max(1, round(len(reply) / 4))  # providers that report no usage
            usage["completion_tokens"] = completion
            usage["estimated"] = True
        usage.setdefault("prompt_tokens", 0)
        elapsed = float(final.get("elapsed") or max(ended - started, 1e-6))
        generation = float(final.get("elapsed") or 0.0)
        if not generation:
            # Reasoning tokens count toward the completion, so time from the first output of
            # either kind; a reply that arrived in one piece is timed from the request.
            generation = ended - first_token if first_token and produced > 1 else ended - started
        device = pick_device(request.device) if runtime in {"mariana", "transformers"} else None
        result = {
            "type": "done",
            "reply": reply or EMPTY_REPLY,
            "empty": not reply,
            "kind": final.get("kind") or entry["kind_label"],
            "elapsed": elapsed,
            "first_token_seconds": (first_token - started) if first_token else None,
            "tokens_per_second": completion / max(generation, 1e-6),
            "usage": usage,
            "prompt_format": final.get("prompt_format", ""),
            "finish_reason": final.get("finish_reason", "stop"),
            "model": entry["path"] or entry["name"],
            "model_id": entry["id"],
            "model_name": entry["name"],
            "runtime": runtime,
            "device": describe_device(device) if device is not None else runtime,
        }
        if thoughts:
            result["reasoning"] = "".join(thoughts)
        for key in ("ncm", "notice", "served_by", "context"):
            if final.get(key) is not None:
                result[key] = final[key]
        yield result

    def chat(self, config: dict) -> dict:
        """Run a chat to completion and return the final result."""
        result: dict = {}
        messages = None
        for event in self.stream(config):
            if event["type"] == "done":
                result = event
        if messages is None:
            messages = self._messages(config)
        result = {key: value for key, value in result.items() if key != "type"}
        result["messages"] = messages
        return result

    def close(self) -> None:
        """Unload models and stop every runtime Studio owns before the server exits."""
        try:
            self.llama.close()
        except Exception:  # noqa: BLE001 - a stuck child must not keep models resident
            pass
        try:
            self.ollama.close()
        except Exception:  # noqa: BLE001
            pass
        with self._cache_lock:
            self._runtime_cache.clear()
        gc.collect()
        self._empty_device_cache()


def _content_text(content) -> str:
    """Message content as text; accepts OpenAI-style lists of text parts."""
    if isinstance(content, list):
        return "".join(
            str(part.get("text", "")) for part in content
            if isinstance(part, dict) and part.get("type", "text") == "text"
        )
    return "" if content is None else str(content)


__all__ = ["ModelGateway", "ModelError", "catalog_module", "find_llama_server"]
