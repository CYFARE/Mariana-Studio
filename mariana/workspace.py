"""Shared local workspace paths and small JSON persistence helpers.

Everything user-created (agents, runs, chats, endpoints, settings, agent files)
lives in one visible directory so it is easy to inspect, back up, or delete.
The directory is inside the project root and ignored by git.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from pathlib import Path

WORKSPACE_DIR = "agent-workspace"
_SUBDIRECTORIES = ("agents", "runs", "files")
_write_lock = threading.Lock()


def workspace_path(root: Path | str) -> Path:
    return Path(root).resolve() / WORKSPACE_DIR


def ensure_workspace(root: Path | str) -> Path:
    path = workspace_path(root)
    path.mkdir(parents=True, exist_ok=True)
    for name in _SUBDIRECTORIES:
        (path / name).mkdir(parents=True, exist_ok=True)
    return path


def settings_path(root: Path | str) -> Path:
    return ensure_workspace(root) / "settings.json"


def endpoints_path(root: Path | str) -> Path:
    return ensure_workspace(root) / "endpoints.json"


def chats_path(root: Path | str) -> Path:
    path = ensure_workspace(root) / "chats"
    path.mkdir(parents=True, exist_ok=True)
    return path


def files_path(root: Path | str) -> Path:
    """The agent file sandbox shared by AML flows and Chat workspaces."""
    path = ensure_workspace(root) / "files"
    path.mkdir(parents=True, exist_ok=True)
    return path


def chat_files_path(root: Path | str, chat_id: str, create: bool = True) -> Path:
    """One conversation's tool workspace, inside the agent file sandbox."""
    path = files_path(root) / "chats" / str(chat_id)
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def jobs_path(root: Path | str) -> Path:
    path = ensure_workspace(root) / "jobs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path | str, default=None):
    """Return the parsed JSON file, or ``default`` if it is missing or corrupt."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError, UnicodeDecodeError,
            RecursionError):
        return default


def write_json(path: Path | str, value, *, indent: int | None = 2) -> None:
    """Write JSON atomically: a crash leaves either the old or the new file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    payload = json.dumps(value, indent=indent, ensure_ascii=False, default=str)
    with _write_lock:
        try:
            temporary.write_text(payload, encoding="utf-8")
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
