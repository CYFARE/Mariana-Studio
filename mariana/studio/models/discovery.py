"""Find models that other tools already installed on this machine.

Sources: the Hugging Face cache, Ollama (running server or on-disk manifests),
llama.cpp, LM Studio and friends (model folders and running OpenAI-compatible
servers), plus any folders the user added in settings.  Every probe is bounded
by a short timeout and the combined result is cached, so listing models never
blocks on a dead port.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from . import gguf
from .context import model_context
from .catalog import (
    FINETUNE_REASONS,
    KIND_LABELS,
    SOURCE_LABELS,
    gguf_entry,
    has_module,
    scan_folder,
    transformers_entry,
)
from .endpoints import DEFAULT_OLLAMA_URL

CACHE_SECONDS = 20.0
PROBE_TIMEOUT = 0.6
MAX_FILES_PER_SOURCE = 200
# (label, port) of OpenAI-compatible servers commonly run on localhost.
LOCAL_SERVERS = [
    ("llama.cpp", 8080), ("LM Studio", 1234), ("vLLM", 8000), ("Jan", 1337),
    ("KoboldCpp", 5001), ("text-generation-webui", 5000), ("GPT4All", 4891),
]
OWNER_LABELS = {"llamacpp": "llama.cpp", "vllm": "vLLM", "koboldcpp": "KoboldCpp"}
# (source id, directories that hold GGUF files for that tool)
GGUF_FOLDERS = [
    ("llama.cpp", ["~/.cache/llama.cpp", "~/llama.cpp/models", "~/Library/Caches/llama.cpp"]),
    ("lm-studio", ["~/.lmstudio/models", "~/.cache/lm-studio/models"]),
    ("gpt4all", ["~/.local/share/nomic.ai/GPT4All",
                 "~/Library/Application Support/nomic.ai/GPT4All"]),
    ("jan", ["~/jan/models", "~/.local/share/Jan/data/models",
             "~/Library/Application Support/Jan/data/models"]),
    ("text-generation-webui", ["~/text-generation-webui/models",
                               "~/text-generation-webui/user_data/models"]),
]
LLAMA_SERVER_CANDIDATES = [
    "~/llama.cpp/build/bin/llama-server", "~/llama.cpp/llama-server",
    "~/llama.cpp/build/bin/Release/llama-server.exe", "~/.local/bin/llama-server",
    "/usr/local/bin/llama-server", "/opt/homebrew/bin/llama-server",
]


def fetch_json(url: str, timeout: float = PROBE_TIMEOUT, headers: dict | None = None):
    """GET a JSON document, or ``None`` on any failure (probes must not raise)."""
    request = urllib.request.Request(url, headers={"Accept": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read(4_000_000).decode("utf-8", errors="replace"))
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return None


def _expand(path: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(path)))


def _executable(path: Path) -> bool:
    return path.is_file() and os.access(path, os.X_OK)


def find_llama_server(settings: dict | None = None) -> str | None:
    """Locate a ``llama-server`` binary: setting, environment, PATH, common builds."""
    candidates = []
    configured = (settings or {}).get("llama_server_path")
    if configured:
        candidates.append(configured)
    if os.environ.get("MARIANA_LLAMA_SERVER"):
        candidates.append(os.environ["MARIANA_LLAMA_SERVER"])
    found = shutil.which("llama-server")
    if found:
        candidates.append(found)
    if os.environ.get("LLAMA_CPP_DIR"):
        base = os.environ["LLAMA_CPP_DIR"]
        candidates += [f"{base}/build/bin/llama-server", f"{base}/llama-server"]
    candidates += LLAMA_SERVER_CANDIDATES
    for candidate in candidates:
        path = _expand(str(candidate))
        if _executable(path):
            return str(path)
    return None


def gguf_runtime(settings: dict | None = None) -> str | None:
    """Which runtime will execute GGUF files: a llama-server binary or the Python binding."""
    preference = (settings or {}).get("gguf_runtime", "auto")
    binary = find_llama_server(settings)
    binding = has_module("llama_cpp")
    if preference == "llama-cpp-python" and binding:
        return "llama-cpp-python"
    if preference == "llama-server" and binary:
        return "llama-server"
    if binary:
        return "llama-server"
    return "llama-cpp-python" if binding else None


def ollama_url(settings: dict | None = None) -> str:
    """The Ollama server address from settings, ``OLLAMA_HOST``, or the default."""
    value = str((settings or {}).get("ollama_url") or os.environ.get("OLLAMA_HOST") or "").strip()
    if not value:
        return DEFAULT_OLLAMA_URL
    if "://" not in value:
        value = "http://" + value
    parsed = urllib.parse.urlparse(value)
    host = parsed.hostname or "127.0.0.1"
    if host in {"0.0.0.0", "::"}:  # a listen address is not a connect address
        host = "127.0.0.1"
    port = parsed.port or 11434
    return f"{parsed.scheme or 'http'}://{host}:{port}"


def ollama_model_dirs() -> list[Path]:
    candidates = []
    if os.environ.get("OLLAMA_MODELS"):
        candidates.append(os.environ["OLLAMA_MODELS"])
    candidates += ["~/.ollama/models", "/usr/share/ollama/.ollama/models",
                   "/var/lib/ollama/.ollama/models"]
    return [_expand(item) for item in candidates]


def _ollama_name(parts: tuple[str, ...]) -> str:
    """``registry/namespace/model/tag`` -> the name ``ollama run`` accepts."""
    registry, namespace, model, tag = parts
    name = model if namespace == "library" else f"{namespace}/{model}"
    if registry != "registry.ollama.ai":
        name = f"{registry}/{name}"
    return f"{name}:{tag}"


def ollama_disk_models() -> list[dict]:
    """Models in Ollama's store, read from manifests (works while Ollama is stopped)."""
    models: dict[str, dict] = {}
    for base in ollama_model_dirs():
        manifests = base / "manifests"
        try:
            files = [path for path in manifests.glob("*/*/*/*") if path.is_file()]
        except OSError:
            continue
        for path in files[:MAX_FILES_PER_SOURCE]:
            try:
                manifest = json.loads(path.read_text(encoding="utf-8"))
                stat = path.stat()
            except (OSError, ValueError):
                continue
            name = _ollama_name(path.relative_to(manifests).parts)
            blob, size = None, 0
            for layer in manifest.get("layers") or []:
                size += int(layer.get("size") or 0)
                if layer.get("mediaType") == "application/vnd.ollama.image.model":
                    digest = str(layer.get("digest", "")).replace(":", "-")
                    candidate = base / "blobs" / digest
                    if candidate.is_file():
                        blob = candidate
            models.setdefault(name, {
                "name": name, "bytes": size, "modified": stat.st_mtime,
                "blob": str(blob) if blob else None,
            })
    return sorted(models.values(), key=lambda item: item["name"])


def _timestamp(value) -> float:
    try:
        text = str(value)
        if "." in text:  # trim nanoseconds that fromisoformat rejects
            head, tail = text.split(".", 1)
            zone = ""
            for marker in ("+", "-", "Z"):
                if marker in tail:
                    index = tail.index(marker)
                    zone, tail = tail[index:], tail[:index]
                    break
            text = f"{head}.{tail[:6]}{zone}"
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError):
        return 0.0


def _base_entry(model_id: str, name: str, kind: str, source: str, **fields) -> dict:
    entry = {
        "id": model_id, "name": name, "kind": kind, "kind_label": KIND_LABELS[kind],
        "source": source, "source_label": SOURCE_LABELS.get(source, source), "runtime": kind,
        "path": "", "absolute_path": "", "bytes": 0, "modified": 0, "available": True,
        "reason": "", "detail": {}, "capabilities": {"chat": True, "finetune": False, "stream": True},
        "finetune_reason": FINETUNE_REASONS.get(kind, ""),
    }
    entry.update(fields)
    return entry


def hf_cache_dirs() -> list[Path]:
    candidates = [os.environ.get("HF_HUB_CACHE"), os.environ.get("HUGGINGFACE_HUB_CACHE")]
    if os.environ.get("HF_HOME"):
        candidates.append(os.path.join(os.environ["HF_HOME"], "hub"))
    candidates.append("~/.cache/huggingface/hub")
    seen, result = set(), []
    for item in candidates:
        if not item:
            continue
        path = _expand(item)
        if path not in seen:
            seen.add(path)
            result.append(path)
    return result


class Discovery:
    """Cached view of external model sources and local runtimes."""

    def __init__(self, root: Path, settings=None, own_port=None):
        self.root = Path(root).resolve()
        self._settings = settings or (lambda: {})
        self._own_port = own_port or (lambda: None)
        self._lock = threading.Lock()
        self._cache: tuple[float, dict] | None = None

    def invalidate(self) -> None:
        with self._lock:
            self._cache = None

    def scan(self, force: bool = False) -> dict:
        """Return ``{"entries": [...], "runtimes": {...}}`` (cached briefly)."""
        with self._lock:
            cached = self._cache
        if cached and not force and time.time() - cached[0] < CACHE_SECONDS:
            return cached[1]
        result = self._scan()
        with self._lock:
            self._cache = (time.time(), result)
        return result

    # -- implementation --------------------------------------------------
    def _scan(self) -> dict:
        settings = self._settings() or {}
        runtime = gguf_runtime(settings)
        seen: set[Path] = set()
        entries: list[dict] = []
        if settings.get("discover_local", True) is False:
            return {"entries": [], "runtimes": self._runtimes(settings, runtime, None, [], [])}
        with ThreadPoolExecutor(max_workers=8) as pool:
            ollama_future = pool.submit(self._probe_ollama, settings)
            server_futures = [
                pool.submit(self._probe_server, label, port) for label, port in LOCAL_SERVERS
                if port != self._own_port()
            ]
            huggingface = self._scan_huggingface(runtime, seen)
            folders = self._scan_folders(settings, runtime, seen)
            ollama = ollama_future.result()
            servers = [future.result() for future in server_futures]
        servers = [server for server in servers if server]
        entries += huggingface + folders
        entries += self._ollama_entries(ollama, runtime)
        for server in servers:
            entries += server["entries"]
        return {
            "entries": entries,
            "runtimes": self._runtimes(settings, runtime, ollama, servers, huggingface),
        }

    def _scan_huggingface(self, runtime: str | None, seen: set[Path]) -> list[dict]:
        entries = []
        for cache in hf_cache_dirs():
            try:
                repos = sorted(cache.glob("models--*"))[:MAX_FILES_PER_SOURCE]
            except OSError:
                continue
            for repo_dir in repos:
                repo = repo_dir.name[len("models--"):].replace("--", "/")
                try:
                    snapshots = sorted(
                        (path for path in (repo_dir / "snapshots").iterdir() if path.is_dir()),
                        key=lambda path: path.stat().st_mtime, reverse=True,
                    )
                except OSError:
                    continue
                for snapshot in snapshots[:1]:  # the newest revision only
                    for path in sorted(snapshot.rglob("*.gguf"))[:20]:
                        resolved = path.resolve()
                        if gguf.is_auxiliary(path) or resolved in seen or not path.is_file():
                            continue
                        seen.add(resolved)
                        entries.append(gguf_entry(self.root, path, "huggingface", runtime, repo=repo))
                    if snapshot.resolve() not in seen:
                        entry = transformers_entry(self.root, snapshot, "huggingface", name=repo)
                        if entry:
                            seen.add(snapshot.resolve())
                            entry["detail"]["repo"] = repo
                            entries.append(entry)
        return entries

    def _scan_folders(self, settings: dict, runtime: str | None, seen: set[Path]) -> list[dict]:
        entries = []
        for source, folders in GGUF_FOLDERS:
            for folder in folders:
                base = _expand(folder)
                if not base.is_dir():
                    continue
                try:
                    files = sorted(base.rglob("*.gguf"))[:MAX_FILES_PER_SOURCE]
                except OSError:
                    continue
                for path in files:
                    try:
                        resolved, size = path.resolve(), path.stat().st_size
                    except OSError:
                        continue
                    if gguf.is_auxiliary(path) or resolved in seen or size < 8_000_000:
                        continue
                    seen.add(resolved)
                    entries.append(gguf_entry(self.root, path, source, runtime))
        for folder in settings.get("model_dirs") or []:
            base = _expand(str(folder))
            if base.is_dir():
                entries += scan_folder(self.root, base, "folder", runtime, seen,
                                       limit=MAX_FILES_PER_SOURCE)
        return entries

    def _probe_ollama(self, settings: dict) -> dict:
        url = ollama_url(settings)
        tags = fetch_json(f"{url}/api/tags")
        running = isinstance(tags, dict) and isinstance(tags.get("models"), list)
        version = (fetch_json(f"{url}/api/version") or {}).get("version") if running else None
        return {
            "url": url, "running": running, "version": version,
            "binary": shutil.which("ollama"),
            "models": tags["models"] if running else [],
            "disk": [] if running else ollama_disk_models(),
        }

    def _ollama_entries(self, ollama: dict, runtime: str | None) -> list[dict]:
        entries = []
        url = ollama["url"]
        for model in ollama["models"]:
            name = str(model.get("name") or model.get("model") or "")
            if not name:
                continue
            details = model.get("details") or {}
            entries.append(_base_entry(
                f"ollama:{name}", name, "ollama", "ollama", path=url,
                bytes=int(model.get("size") or 0), modified=_timestamp(model.get("modified_at")),
                detail={
                    "architecture": details.get("family") or "Ollama", "model": name,
                    "size_label": details.get("parameter_size"),
                    "quantization": details.get("quantization_level"),
                    "base_url": url, "remote": False, "ctx_unit": "tokens",
                },
            ))
        for model in ollama["disk"]:
            blob = model.get("blob")
            can_fall_back = bool(blob and runtime)
            if can_fall_back:
                reason = ""
                note = (
                    "Ollama's server is not running, so Studio runs this model file through "
                    "llama.cpp. Start Ollama to use its own runtime and settings."
                )
            else:
                note = ""
                reason = (
                    "Ollama is installed but its server is not running. Start it from "
                    "Models → Runtimes, or run `ollama serve`."
                    if ollama["binary"] else
                    "This model is in Ollama's store but Ollama is not installed. Install "
                    "Ollama, or install llama.cpp to run the model file directly."
                )
            entries.append(_base_entry(
                f"ollama:{model['name']}", model["name"], "ollama", "ollama",
                runtime="llama-server" if can_fall_back and runtime == "llama-server"
                else ("llama-cpp-python" if can_fall_back else "ollama"),
                path=url, absolute_path=blob or "", bytes=model["bytes"],
                modified=model["modified"], available=can_fall_back, reason=reason,
                detail={
                    "architecture": "Ollama", "model": model["name"], "base_url": url,
                    "remote": False, "ctx_unit": "tokens", "server_stopped": True,
                    "startable": bool(ollama["binary"]), "blob": blob, "note": note,
                },
            ))
        return entries

    def _probe_server(self, label: str, port: int) -> dict | None:
        base_url = f"http://127.0.0.1:{port}/v1"
        listing = fetch_json(f"{base_url}/models")
        if not isinstance(listing, dict) or not isinstance(listing.get("data"), list):
            return None
        entries = []
        for model in listing["data"][:60]:
            model_id = str((model or {}).get("id") or "") if isinstance(model, dict) else ""
            if not model_id:
                continue
            owner = str(model.get("owned_by") or "").lower()
            server = OWNER_LABELS.get(owner, label)
            name = model_id.replace("\\", "/").rstrip("/").split("/")[-1] or model_id
            detail = {
                "architecture": model_id, "model": model_id, "base_url": base_url,
                "remote": False, "server": server, "ctx_unit": "tokens",
            }
            context = model_context(model)
            if context:
                detail["ctx_len"] = context
            entries.append(_base_entry(
                f"server:{port}:{model_id}", name, "openai", "server",
                source_label=f"{server} server", path=base_url, detail=detail,
            ))
        if not entries:
            return None
        return {"label": entries[0]["detail"]["server"], "port": port, "url": base_url,
                "models": len(entries), "entries": entries}

    def _runtimes(self, settings, runtime, ollama, servers, huggingface) -> dict:
        binary = find_llama_server(settings)
        result = {
            "llama_cpp": {
                "label": "llama.cpp", "installed": bool(binary or has_module("llama_cpp")),
                "binary": binary, "python_binding": has_module("llama_cpp"), "runtime": runtime,
                "hint": "" if runtime else (
                    "Build llama.cpp (https://github.com/ggml-org/llama.cpp) so `llama-server` "
                    "is on PATH, or run: pip install llama-cpp-python"
                ),
            },
            "transformers": {
                "label": "Transformers", "installed": has_module("transformers"),
                "hint": "" if has_module("transformers")
                else "pip install -r requirements-finetune.txt",
            },
            "onnx": {
                "label": "ONNX Runtime",
                "installed": has_module("optimum") and has_module("onnxruntime"),
                "hint": 'pip install "optimum[onnxruntime]" transformers',
            },
            "anthropic": {
                "label": "Anthropic SDK", "installed": has_module("anthropic"),
                "hint": "" if has_module("anthropic") else "pip install anthropic",
            },
            "huggingface": {
                "label": "Hugging Face cache", "installed": True,
                "cache": str(hf_cache_dirs()[0]) if hf_cache_dirs() else "",
                "models": len(huggingface),
                "token": bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")),
            },
            "servers": [
                {"label": server["label"], "url": server["url"], "models": server["models"]}
                for server in servers
            ],
        }
        if ollama is not None:
            result["ollama"] = {
                "label": "Ollama", "installed": bool(ollama["binary"]) or ollama["running"],
                "binary": ollama["binary"], "running": ollama["running"], "url": ollama["url"],
                "version": ollama["version"],
                "models": len(ollama["models"]) or len(ollama["disk"]),
                "hint": "" if ollama["running"] else (
                    "Start the server to chat with Ollama models through Ollama itself."
                    if ollama["binary"] else "Install Ollama from https://ollama.com/download"
                ),
            }
        return result
