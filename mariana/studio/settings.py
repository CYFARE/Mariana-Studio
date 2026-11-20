"""Studio settings: a small validated JSON document in the workspace."""

from __future__ import annotations

import threading
from pathlib import Path

from ..workspace import read_json, settings_path, write_json
from .errors import BadRequestError

DEFAULTS = {
    "model_dirs": [],            # extra folders scanned for models
    "discover_local": True,      # look for Ollama, llama.cpp, LM Studio, the HF cache…
    "gguf_runtime": "auto",      # auto | llama-server | llama-cpp-python
    "llama_server_path": "",     # explicit path to a llama-server binary
    "llama_gpu_layers": "auto",  # auto, or the number of layers to offload
    "ollama_url": "",            # empty = OLLAMA_HOST or http://127.0.0.1:11434
    "idle_unload_minutes": 10,   # stop the managed llama.cpp server when idle (0 = never)
    "auto_install_dependencies": True,  # install missing optional requirement groups on startup
    "check_for_updates": True,   # compare VERSION.txt with the latest GitHub release on startup
    "allow_os_access": False,    # let shell.exec run and HTTP reach private addresses
}
GGUF_RUNTIMES = ("auto", "llama-server", "llama-cpp-python")


def _clean(patch: dict) -> dict:
    clean = {}
    for key, value in patch.items():
        if key not in DEFAULTS:
            raise BadRequestError(f"Unknown setting: {key}")
        if key == "model_dirs":
            if not isinstance(value, list):
                raise BadRequestError("model_dirs must be a list of folders")
            folders = []
            for item in value[:32]:
                folder = Path(str(item)).expanduser()
                if not folder.is_dir():
                    raise BadRequestError(f"Model folder does not exist: {item}")
                resolved = str(folder.resolve())
                if resolved not in folders:
                    folders.append(resolved)
            clean[key] = folders
        elif key == "discover_local":
            clean[key] = bool(value)
        elif key == "gguf_runtime":
            if value not in GGUF_RUNTIMES:
                raise BadRequestError("gguf_runtime must be auto, llama-server, or llama-cpp-python")
            clean[key] = value
        elif key == "llama_server_path":
            text = str(value or "").strip()
            if text and not Path(text).expanduser().is_file():
                raise BadRequestError("llama_server_path does not point to a file")
            clean[key] = text
        elif key == "llama_gpu_layers":
            if value in (None, "", "auto"):
                clean[key] = "auto"
            else:
                try:
                    layers = int(value)
                except (TypeError, ValueError):
                    raise BadRequestError("llama_gpu_layers must be auto or a number") from None
                if not -1 <= layers <= 1000:
                    raise BadRequestError("llama_gpu_layers must be between -1 and 1000")
                clean[key] = layers
        elif key == "ollama_url":
            text = str(value or "").strip().rstrip("/")
            if text and not text.startswith(("http://", "https://")):
                raise BadRequestError("ollama_url must start with http:// or https://")
            clean[key] = text
        elif key == "idle_unload_minutes":
            try:
                minutes = float(value)
            except (TypeError, ValueError):
                raise BadRequestError("idle_unload_minutes must be a number") from None
            if not 0 <= minutes <= 1440:
                raise BadRequestError("idle_unload_minutes must be between 0 and 1440")
            clean[key] = minutes
        elif key == "auto_install_dependencies":
            clean[key] = bool(value)
        elif key == "check_for_updates":
            clean[key] = bool(value)
        elif key == "allow_os_access":
            clean[key] = bool(value)
    return clean


class SettingsStore:
    """Thread-safe access to ``agent-workspace/settings.json``."""

    def __init__(self, root: Path):
        self.path = settings_path(root)
        self._lock = threading.Lock()
        self._value: dict | None = None

    def get(self) -> dict:
        with self._lock:
            if self._value is None:
                stored = read_json(self.path, {})
                value = dict(DEFAULTS)
                if isinstance(stored, dict):
                    value.update({key: stored[key] for key in DEFAULTS if key in stored})
                self._value = value
            return dict(self._value)

    def update(self, patch: dict) -> dict:
        if not isinstance(patch, dict):
            raise BadRequestError("Settings must be a JSON object")
        clean = _clean(patch)
        current = self.get()
        current.update(clean)
        with self._lock:
            write_json(self.path, current)
            self._value = current
            return dict(current)
