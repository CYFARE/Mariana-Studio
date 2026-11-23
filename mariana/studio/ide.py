"""IDE workspaces and the streaming coding agent behind the Studio IDE view.

A workspace is a project folder inside ``agent-workspace/ide/workspaces``.  The
coding loop reuses the AML ability sandbox (:mod:`mariana.agents.tools`) with the
workspace as its file root, so IDE sessions and visual agents share one set of
tool definitions, limits and the same shell opt-in.

Every agent edit is snapshotted before it happens.  A session therefore owns a
complete undo history: ``revert`` restores the files as they were before the
first edited byte, not merely the last tool call.
"""

from __future__ import annotations

import base64
import copy
import difflib
import hashlib
import json
import mimetypes
import os
import re
import shutil
import struct
import threading
import time
import uuid
from collections import Counter, deque
from datetime import datetime
from pathlib import Path

from ..agents.aml import ABILITY_TYPES, slugify
from ..agents.runtime import EMPTY_REPLIES, normalize_plan, parse_model_action
from ..agents.swarm import SwarmCoordinator, SwarmLimits, snapshot_due
from ..agents.tools import (
    TOOL_ATTEMPTS, execute_ability, json_safe, nearest_tool_id, retryable_tool_failure,
    tool_failure_hint,
)
from ..workspace import ensure_workspace, read_json, write_json
from .models.context import entry_context
from .errors import (BadRequestError, ConflictError, NotFoundError, UnavailableError,
                     UpstreamError)
from .tokens import estimate_tokens, utf8_length
from .validation import choice, flag, number, text

IDE_VERSION = "1.0"
MODES = ("ask", "edit", "auto")
MODE_INFO = {
    "ask": {
        "label": "Ask",
        "description": "Read and diagnose only; no file is changed.",
        "tools": ("file.read", "file.inspect", "file.list", "file.tree", "file.search", "skill.read",
                  "code.diagnostics", "csv.read", "sqlite.query", "git.inspect",
                  "http.fetch", "news.headlines", "browser.fetch", "agent.swarm"),
    },
    "edit": {
        "label": "Edit",
        "description": "Read, search and make precise file edits, with snapshots.",
        "tools": ("file.read", "file.inspect", "file.list", "file.tree", "file.search", "skill.read",
                  "code.diagnostics", "csv.read", "sqlite.query", "git.inspect",
                  "http.fetch", "news.headlines", "browser.fetch", "agent.swarm",
                  "file.write", "file.edit"),
    },
    "auto": {
        "label": "Auto",
        "description": "Edit, delete, diagnose and run commands when shell access is enabled.",
        "tools": ("file.read", "file.inspect", "file.list", "file.tree", "file.search", "skill.read",
                  "code.diagnostics", "csv.read", "sqlite.query", "git.inspect",
                  "http.fetch", "news.headlines", "browser.fetch", "agent.swarm",
                  "file.write", "file.edit", "file.delete", "shell.exec"),
    },
}
FINISHED = frozenset({"completed", "failed", "cancelled"})
DELTA_EVENTS = frozenset({"assistant_delta", "reasoning_delta", "swarm_agent_delta"})
MAX_FILE_BYTES = 2_000_000
MAX_PREVIEW_BYTES = 8_000_000
MAX_PREVIEW_TEXT_BYTES = 2_000_000
MAX_HEX_BYTES = 4_096
MAX_SKILLS = 48
MAX_PROMPT_SKILLS = MAX_SKILLS
MAX_SKILL_PROMPT_CHARS = 6_000
MAX_SKILL_BYTES = 48_000
MAX_SKILL_CHARS = 5_000
MAX_SNAPSHOT_BYTES = 4_000_000
MAX_TREE_ENTRIES = 1500
MAX_TREE_DEPTH = 8
MAX_SEARCH_RESULTS = 500
MAX_OBSERVATION_CHARS = 6_000
MAX_OBSERVATION_TOTAL_CHARS = 20_000
OBSERVATION_FIELD_CHARS = 4_000
MAX_PROMPT_CHARS = 23_000
MAX_DIFF_CHARS = 24_000
MAX_MESSAGES = 40
MAX_MESSAGE_CHARS = 8_000
MAX_READ_LINES = 20_000
MAX_FIND_RESULTS = 1000
MAX_INDEX_ENTRIES = 50_000
MAX_SETTING_PATHS = 200
MAX_SETTING_PATH_CHARS = 400
MAX_CONTEXT_FILES = 24
MAX_CONTEXT_CHARS = 12_000
MAX_CONTEXT_FILE_CHARS = 4_000
MODEL_MESSAGE_WINDOW = 40  # mirrors the chat gateway's context window
MESSAGE_RECAP_TRIGGER = MODEL_MESSAGE_WINDOW - 8
DEFAULT_CONTEXT_WINDOW = 8_192
BUDGET_WARNING_PERCENT = 80
CONSOLIDATE_PERCENT = 80
CONSOLIDATE_KEEP_MESSAGES = 8
CONSOLIDATE_SUMMARY_CHARS = 4_000
MAX_SWARM_AGENTS = 6
MAX_SWARM_TASK_CHARS = 4_000
MAX_SWARM_STEPS = 8
MAX_SWARM_ROUNDS = 2
MAX_SWARM_SECONDS = 600
MAX_SWARM_OUTPUT_CHARS = 8_000
MESSAGE_TOKEN_OVERHEAD = 4
DEFAULT_MAX_STEPS = 24
MAX_STEPS = 128
STEP_EXTENSION = 24
MAX_STEP_EXTENSIONS = 3
MAX_FORMAT_REPAIRS = 2
MAX_NETWORK_RETRIES = 2
CHECKPOINT_SECONDS = 5.0
DEFAULT_MAX_SECONDS = 900
DEFAULT_OUTPUT_TOKENS = 4_096
MAX_OUTPUT_TOKENS = 16_384
MAX_REASONING_CHARS = 12_000
MAX_SECONDS = 3600
IMPORT_MAX_FILES = 3000
IMPORT_MAX_BYTES = 64_000_000
IGNORED_DIRS = frozenset({
    ".git", ".hg", ".svn", "__pycache__", "node_modules", ".venv", "venv", "env",
    "dist", "build", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".idea", ".vscode",
})
IMPORT_IGNORED = IGNORED_DIRS | frozenset({
    "trained-model", "models", "datasets", "exports", "backups", "agent-workspace",
})
MANIFEST = ".mariana-ide.json"
WORKSPACE_ID = re.compile(r"[a-z0-9][a-z0-9-]{0,63}")
SESSION_ID = re.compile(r"ide-[a-z0-9]{6,40}")
_LANGUAGES = {
    # Web and data
    ".py": "Python", ".pyw": "Python", ".pyi": "Python",
    ".js": "JavaScript", ".mjs": "JavaScript", ".cjs": "JavaScript", ".jsx": "JavaScript",
    ".ts": "TypeScript", ".tsx": "TypeScript", ".mts": "TypeScript", ".cts": "TypeScript",
    ".vue": "Vue", ".svelte": "Svelte", ".astro": "Astro",
    ".html": "HTML", ".htm": "HTML", ".xhtml": "HTML", ".xml": "XML", ".svg": "SVG",
    ".css": "CSS", ".scss": "SCSS", ".sass": "Sass", ".less": "Less",
    ".json": "JSON", ".jsonc": "JSON with Comments", ".json5": "JSON with Comments",
    ".jsonl": "JSON", ".ndjson": "JSON",
    ".md": "Markdown", ".markdown": "Markdown", ".mdx": "MDX", ".txt": "Text", ".log": "Text",
    ".csv": "CSV", ".tsv": "TSV", ".ipynb": "JSON", ".lock": "Text",
    ".yml": "YAML", ".yaml": "YAML", ".toml": "TOML",
    ".ini": "INI", ".cfg": "Config", ".conf": "Config", ".properties": "Config",
    # Shells and scripting
    ".sh": "Shell", ".bash": "Shell", ".zsh": "Shell", ".fish": "Shell", ".ksh": "Shell",
    ".ps1": "PowerShell", ".psm1": "PowerShell", ".psd1": "PowerShell",
    ".bat": "Batch", ".cmd": "Batch",
    ".rb": "Ruby", ".rake": "Ruby", ".php": "PHP", ".pl": "Perl", ".pm": "Perl",
    ".lua": "Lua", ".r": "R", ".jl": "Julia",
    # Compiled languages
    ".c": "C", ".h": "C", ".cpp": "C++", ".cc": "C++", ".cxx": "C++",
    ".hpp": "C++", ".hh": "C++", ".hxx": "C++", ".cu": "CUDA", ".cuh": "CUDA",
    ".cs": "C#", ".fs": "F#", ".fsx": "F#",
    ".m": "Objective-C", ".mm": "Objective-C",
    ".java": "Java", ".kt": "Kotlin", ".kts": "Kotlin",
    ".scala": "Scala", ".groovy": "Groovy", ".gradle": "Groovy",
    ".go": "Go", ".rs": "Rust", ".swift": "Swift", ".dart": "Dart",
    ".zig": "Zig", ".nim": "Nim", ".d": "D",
    ".hs": "Haskell", ".ex": "Elixir", ".exs": "Elixir", ".erl": "Erlang", ".hrl": "Erlang",
    ".clj": "Clojure", ".cljs": "Clojure", ".cljc": "Clojure", ".edn": "Clojure",
    ".lisp": "Lisp", ".scm": "Scheme", ".rkt": "Racket",
    ".elm": "Elm", ".ml": "OCaml", ".mli": "OCaml",
    ".f": "Fortran", ".f90": "Fortran", ".f95": "Fortran", ".f03": "Fortran",
    ".pas": "Pascal", ".adb": "Ada", ".ads": "Ada", ".cob": "COBOL", ".cbl": "COBOL",
    ".asm": "Assembly", ".s": "Assembly",
    ".sol": "Solidity", ".v": "Verilog", ".sv": "SystemVerilog", ".vhd": "VHDL", ".vhdl": "VHDL",
    ".glsl": "GLSL", ".vert": "GLSL", ".frag": "GLSL",
    # Schemas, infrastructure and markup
    ".sql": "SQL", ".proto": "Protocol Buffers", ".graphql": "GraphQL", ".gql": "GraphQL",
    ".tf": "Terraform", ".tfvars": "Terraform", ".hcl": "HCL",
    ".cmake": "CMake", ".mk": "Makefile", ".nix": "Nix", ".tex": "LaTeX", ".sty": "LaTeX",
    ".diff": "Diff", ".patch": "Diff",
    # Geospatial, diagrams and notebooks
    ".kml": "KML", ".kmz": "KMZ", ".gpx": "GPX", ".geojson": "GeoJSON",
    ".topojson": "GeoJSON", ".mmd": "Mermaid", ".mermaid": "Mermaid",
    ".drawio": "Draw.io", ".dio": "Draw.io", ".puml": "PlantUML",
}
_LANGUAGE_NAMES = {
    "dockerfile": "Dockerfile", "containerfile": "Dockerfile",
    "makefile": "Makefile", "gnumakefile": "Makefile",
    "cmakelists.txt": "CMake", "rakefile": "Ruby", "gemfile": "Ruby",
    "vagrantfile": "Ruby", "podfile": "Ruby", "brewfile": "Ruby",
    "justfile": "Shell", ".env": "Config", ".npmrc": "Config",
    ".gitignore": "Config", ".dockerignore": "Config", ".editorconfig": "INI",
    ".babelrc": "JSON", ".eslintrc": "JSON", ".prettierrc": "JSON",
}

_WRITE_TOOLS = frozenset({"file.write", "file.edit", "file.delete"})
# These abilities only read or fetch; a batch of them may run concurrently.
# Writes, deletes, shell commands and swarms always stay in call order.
_PARALLEL_TOOLS = frozenset({
    "file.read", "file.inspect", "file.list", "file.tree", "file.search",
    "skill.read", "code.diagnostics", "csv.read", "sqlite.query", "git.inspect",
    "http.fetch", "news.headlines", "browser.fetch",
})
MAX_PARALLEL_TOOLS = 4
_SKILL_ID = re.compile(r"[a-z0-9][a-z0-9-]{0,63}")
_SKILL_DIRS = (".agents/skills", ".mariana/skills")
_DIAGNOSTIC_EXTENSIONS = frozenset({".py", ".pyw", ".json", ".jsonc", ".json5"})
_MIME_OVERRIDES = {
    ".md": "text/markdown", ".mdx": "text/markdown", ".markdown": "text/markdown",
    ".mmd": "text/vnd.mermaid", ".mermaid": "text/vnd.mermaid",
    ".kml": "application/vnd.google-earth.kml+xml", ".gpx": "application/gpx+xml",
    ".geojson": "application/geo+json", ".topojson": "application/geo+json",
    ".ipynb": "application/x-ipynb+json", ".drawio": "application/vnd.jgraph.mxfile",
    ".csv": "text/csv", ".tsv": "text/tab-separated-values",
}


def shell_enabled() -> bool:
    """Shell access is opt-in for the whole server, never a per-request flag."""
    return (os.environ.get("MARIANA_AGENT_ALLOW_SHELL") == "1"
            or os.environ.get("MARIANA_AGENT_ALLOW_OS") == "1")


def _tool_call_label(call) -> str:
    """A compact ``tool(path or command)`` label for a consolidated recap."""
    if not isinstance(call, dict):
        return "tool"
    name = str(call.get("tool") or call.get("name") or "tool")[:60]
    arguments = call.get("input", call.get("arguments"))
    detail = ""
    if isinstance(arguments, dict):
        detail = (arguments.get("path") or arguments.get("command")
                  or arguments.get("query") or arguments.get("name")
                  or arguments.get("sql") or "")
    detail = " ".join(str(detail).split())[:80]
    return f"{name}({detail})" if detail else name


def _looks_like_action(value) -> bool:
    """Whether a reply that failed to parse was probably meant as an action.

    Prose is a legitimate final answer for models that do not follow the JSON
    contract.  Broken JSON, a JSON code fence or an action key near the start is
    instead asked for again, so a typo never silently ends a long coding run.
    """
    text = str(value or "").strip()
    if not text:
        return False
    if text.startswith("```"):
        text = text[3:]
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    if text.startswith(("{", "[")):
        return True
    head = text[:240]
    return any(marker in head for marker in ('"tool"', '"calls"', '"plan"', '"final"'))


def _clean_relative(value) -> str:
    """A workspace-relative path without escapes, for settings and context lists."""
    text = str(value or "").strip().replace("\\", "/")
    if text.startswith("/") or re.match(r"^[a-zA-Z]:", text):
        raise BadRequestError("Context paths must be relative to the workspace")
    while text.startswith("./"):
        text = text[2:]
    text = text.strip("/")
    if not text:
        return ""
    parts = text.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise BadRequestError("Context paths must stay inside the workspace")
    if parts == [MANIFEST]:
        raise BadRequestError("The workspace manifest cannot be used as context")
    return "/".join(parts)


def _path_list(value, field: str) -> list[str]:
    """Normalise a setting list, rejecting malformed paths instead of guessing."""
    if value in (None, ""):
        return []
    if not isinstance(value, list):
        raise BadRequestError(f"{field} must be a list of workspace-relative paths")
    if len(value) > MAX_SETTING_PATHS:
        raise BadRequestError(f"{field} holds at most {MAX_SETTING_PATHS} paths")
    result: list[str] = []
    for item in value:
        try:
            cleaned = _clean_relative(item)
        except BadRequestError:
            raise
        if len(cleaned) > MAX_SETTING_PATH_CHARS:
            raise BadRequestError(f"Each {field} path is limited to {MAX_SETTING_PATH_CHARS} characters")
        if cleaned and cleaned not in result:
            result.append(cleaned)
    return result


def _under(relative: str, prefixes) -> bool:
    """True when ``relative`` is one of ``prefixes`` or lives below one."""
    for prefix in prefixes or ():
        if relative == prefix or relative.startswith(f"{prefix}/"):
            return True
    return False


def _boolean(value, default: bool = True) -> bool:
    """Read a model-supplied flag as a boolean, including the usual strings."""
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "off"}
    return bool(value)


def _iso(timestamp: float | None = None) -> str:
    return datetime.fromtimestamp(timestamp or time.time()).isoformat(timespec="seconds")


def _version(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()[:32]


def _file_version(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:32]


def _language(path: Path) -> str:
    named = _LANGUAGE_NAMES.get(path.name.lower())
    if named:
        return named
    return _LANGUAGES.get(path.suffix.lower(), path.suffix.lstrip(".").upper() or "Text")


def _mime(path: Path) -> str:
    """A stable content type for the viewer, beyond what mimetypes knows."""
    override = _MIME_OVERRIDES.get(path.suffix.lower())
    if override:
        return override
    guessed, _ = mimetypes.guess_type(path.name)
    return guessed or "application/octet-stream"


def _render_kind(path: Path, mime: str) -> str | None:
    """How the viewer can render a file without executing it."""
    suffix = path.suffix.lower()
    if mime == "image/svg+xml":
        return "svg"
    if mime.startswith("image/"):
        return "image"
    if suffix in {".mmd", ".mermaid"}:
        return "mermaid"
    if suffix in {".md", ".mdx", ".markdown"}:
        return "markdown"
    if suffix in {".csv", ".tsv"}:
        return "table"
    return None


def _looks_text(payload: bytes) -> bool:
    """True when a file decodes as UTF-8 and has no NUL in its first block."""
    sample = payload[:8192]
    if b"\x00" in sample:
        return False
    try:
        sample.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def _image_dimensions(payload: bytes, mime: str) -> dict | None:
    """Read image dimensions from the header, without decoding the pixels."""
    try:
        if mime == "image/png" and payload[:8] == b"\x89PNG\r\n\x1a\n" and len(payload) >= 24:
            width, height = struct.unpack(">II", payload[16:24])
            return {"width": width, "height": height}
        if mime == "image/gif" and payload[:6] in {b"GIF87a", b"GIF89a"} and len(payload) >= 10:
            width, height = struct.unpack("<HH", payload[6:10])
            return {"width": width, "height": height}
        if mime == "image/bmp" and payload[:2] == b"BM" and len(payload) >= 26:
            width, height = struct.unpack("<ii", payload[18:26])
            return {"width": abs(width), "height": abs(height)}
        if mime == "image/webp" and payload[:4] == b"RIFF" and payload[8:12] == b"WEBP":
            chunk = payload[12:16]
            if chunk == b"VP8X" and len(payload) >= 30:
                width = 1 + int.from_bytes(payload[24:27], "little")
                height = 1 + int.from_bytes(payload[27:30], "little")
                return {"width": width, "height": height}
            if chunk == b"VP8 " and len(payload) >= 30:
                width, height = struct.unpack("<HH", payload[26:30])
                return {"width": width & 0x3FFF, "height": height & 0x3FFF}
            if chunk == b"VP8L" and len(payload) >= 25:
                bits = int.from_bytes(payload[21:25], "little")
                return {"width": (bits & 0x3FFF) + 1, "height": ((bits >> 14) & 0x3FFF) + 1}
        if mime == "image/jpeg":
            index = 2
            while index + 9 < len(payload) and index < 262_144:
                if payload[index] != 0xFF:
                    index += 1
                    continue
                marker = payload[index + 1]
                if marker in {0xD8, 0xD9} or 0xD0 <= marker <= 0xD7:
                    index += 2
                    continue
                size = struct.unpack(">H", payload[index + 2:index + 4])[0]
                if marker in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                              0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}:
                    height, width = struct.unpack(">HH", payload[index + 5:index + 9])
                    return {"width": width, "height": height}
                index += 2 + size
    except (struct.error, IndexError, ValueError):
        return None
    return None


def _skill_meta(text: str, fallback: str) -> tuple[str, str]:
    """Read the name and description from SKILL.md front matter or its opening."""
    name = fallback.replace("-", " ").replace("_", " ").title()
    description = ""
    lines = text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if lines and lines[0].strip() == "---":
        for line in lines[1:]:
            if line.strip() == "---":
                break
            key, _, value = line.partition(":")
            key = key.strip().lower()
            value = value.strip().strip('"\'')
            if key == "name" and value:
                name = value
            elif key == "description" and value:
                description = value
    if not description:
        for index, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("# "):
                description = stripped[2:].strip()
                continue
            if description and stripped:
                description = stripped
                break
    return name[:120], description[:300]


def _skill_body(text: str) -> str:
    """The editable instructions below a SKILL.md front matter block."""
    lines = text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if lines and lines[0].strip() == "---":
        for index, line in enumerate(lines[1:], start=1):
            if line.strip() == "---":
                lines = lines[index + 1:]
                break
    return "\n".join(lines).strip("\n")


def _skill_document(name: str, description: str, content: str) -> str:
    """Serialise a skill the same way the editor and discovery expect it."""
    clean_name = " ".join(str(name or "").split())[:120]
    clean_description = " ".join(str(description or "").split())[:300]
    body = _skill_body(str(content or ""))
    front = ["---", f"name: {clean_name}"]
    if clean_description:
        front.append(f"description: {clean_description}")
    front.append("---")
    return "\n".join(front) + "\n" + body.rstrip() + "\n"


def _hex_dump(payload: bytes, limit: int = MAX_HEX_BYTES) -> str:
    """A classic offset / hex / ASCII dump for the viewer and the agent."""
    sample = payload[:limit]
    lines = []
    for offset in range(0, len(sample), 16):
        chunk = sample[offset:offset + 16]
        hexes = " ".join(f"{byte:02x}" for byte in chunk)
        printable = "".join(chr(byte) if 32 <= byte < 127 else "." for byte in chunk)
        lines.append(f"{offset:08x}  {hexes:<47}  |{printable}|")
    return "\n".join(lines)


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        mode = path.stat().st_mode & 0o777
    except OSError:
        mode = None
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        temporary.write_bytes(payload)
        if mode is not None:
            try:
                os.chmod(temporary, mode)
            except OSError:
                pass  # a save must not fail because the mode could not be copied
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_snapshot(snapshot: Path) -> str:
    try:
        return snapshot.read_bytes().decode("utf-8", errors="replace")
    except OSError:
        return ""


def _modified(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _stamp(value, fallback: float) -> float:
    """A manifest timestamp, or ``fallback`` when it is missing or not a number."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return fallback
    return parsed if 0 < parsed < float("inf") else fallback


class _Cancelled(Exception):
    pass


class _RequestCancel:
    """A per-request cancel signal that never writes to the run's own event.

    ``ModelGateway`` sets the event it is given when a stream ends so worker
    threads stop; handing it ``IdeRun.cancelled`` would make every finished
    model turn look like a user cancellation.
    """

    def __init__(self, run_cancelled: threading.Event):
        self._run = run_cancelled
        self._request = threading.Event()

    def is_set(self) -> bool:
        return self._request.is_set() or self._run.is_set()

    def set(self) -> None:
        self._request.set()

    def clear(self) -> None:
        self._request.clear()

    def wait(self, timeout: float | None = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + timeout
        while not self.is_set():
            if deadline is not None and time.monotonic() >= deadline:
                return False
            time.sleep(0.05)
        return True


class IdeRun:
    """Thread-safe live state for one coding turn."""

    def __init__(self, run_id: str, session_id: str, workspace_id: str, message: str):
        self.id = run_id
        self.session_id = session_id
        self.workspace_id = workspace_id
        self.message = message
        self.session: dict | None = None
        self.status = "running"
        self.output = None
        self.error = None
        self.plan: list[dict] = []
        self.plan_set = False
        self.swarm: dict | None = None
        self.swarm_snapshot_at = 0.0
        self.usage: dict = {}
        self.exhausted = False
        self.timed_out = False
        self.model = ""
        self.created = time.time()
        self.started = self.created
        self.ended = None
        self.events: list[dict] = []
        self.cancelled = threading.Event()
        self.condition = threading.Condition()
        self._seq = 0

    def event(self, kind: str, **fields) -> None:
        with self.condition:
            if self.status in FINISHED:
                return
            self._append(kind, fields)
            self.condition.notify_all()

    def _append(self, kind: str, fields: dict) -> None:
        """Append one event; the caller holds ``condition``."""
        self._seq += 1
        event = {"seq": self._seq, "at": _iso(), "type": kind}
        event.update({key: json_safe(value, 200_000) for key, value in fields.items()})
        self.events.append(event)
        if len(self.events) > 4000:
            self._trim_events()

    def _trim_events(self) -> None:
        """Drop old streamed deltas before control events a reconnect needs."""
        excess = len(self.events) - 3000
        if excess <= 0:
            return
        dropped = set()
        for event in self.events:
            if excess <= 0:
                break
            if event.get("type") in DELTA_EVENTS:
                dropped.add(event["seq"])
                excess -= 1
        if dropped:
            self.events = [event for event in self.events if event["seq"] not in dropped]
        else:
            del self.events[:1000]

    def finish(self, status: str, output=None, error: str | None = None, **extra) -> None:
        with self.condition:
            if self.status in FINISHED:
                return
            self.output = output
            self.error = error
            self.ended = time.time()
            # Append and flip under one lock: a waiter that sees ``finished``
            # also sees ``run_<status>``, and no worker event can arrive after
            # the terminal one.
            self._append(f"run_{status}", {"output": output, "error": error, **extra})
            self.status = status
            self.condition.notify_all()

    def cancel(self) -> None:
        with self.condition:
            if self.status in FINISHED or self.cancelled.is_set():
                return
            self.cancelled.set()
        self.event("log", level="warning", message="Cancellation requested")

    def set_swarm(self, snapshot: dict | None) -> None:
        """Publish the latest swarm state for snapshots and the IDE panel."""
        bounded = json_safe(snapshot, 160_000) if snapshot else None
        with self.condition:
            self.swarm = bounded
            self.condition.notify_all()

    def elapsed(self) -> float:
        return max(0.0, (self.ended or time.time()) - self.started)

    def snapshot(self, since: int = 0, include_events: bool = True) -> dict:
        with self.condition:
            data = {
                "id": self.id,
                "session_id": self.session_id,
                "workspace_id": self.workspace_id,
                "status": self.status,
                "message": self.message,
                "output": self.output,
                "error": self.error,
                "plan": list(self.plan),
                "swarm": self.swarm,
                "usage": dict(self.usage),
                "exhausted": self.exhausted,
                "timed_out": self.timed_out,
                "created": self.created,
                "started": self.started,
                "ended": self.ended,
                "elapsed": self.elapsed(),
                "event_count": self._seq,
            }
            if include_events:
                data["events"] = [item for item in self.events if item["seq"] > since]
            return data

    def wait_events(self, since: int, timeout: float = 1.0) -> tuple[list[dict], bool]:
        with self.condition:
            pending = [item for item in self.events if item["seq"] > since]
            finished = self.status in FINISHED
            if not pending and not finished:
                self.condition.wait(timeout)
                pending = [item for item in self.events if item["seq"] > since]
                finished = self.status in FINISHED
            return pending, finished


class IdeService:
    """Workspaces, sessions, files and the coding agent loop."""

    def __init__(self, root: Path, gateway, skills_roots: list[Path] | None = None,
                 settings=None):
        self.root = Path(root).resolve()
        self._server_settings = settings if callable(settings) else (lambda: settings or {})
        if skills_roots is None:
            skills_roots = [Path.home() / ".agents" / "skills"]
        self.user_skill_roots = [Path(item).expanduser().resolve() for item in skills_roots]
        workspace = ensure_workspace(self.root)
        self.agent_workspace = workspace
        self.base = workspace / "ide"
        self.workspaces_dir = self.base / "workspaces"
        self.workspaces_file = self.base / "workspaces.json"
        self.sessions_dir = self.base / "sessions"
        self.runs_dir = self.base / "runs"
        self.snapshots_dir = self.base / "snapshots"
        for folder in (self.workspaces_dir, self.sessions_dir, self.runs_dir, self.snapshots_dir):
            folder.mkdir(parents=True, exist_ok=True)
        self._external_cache: dict[str, dict] | None = None
        self.gateway = gateway
        self._lock = threading.RLock()
        self._runs: dict[str, IdeRun] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._active_sessions: set[str] = set()
        self._active_workspaces: set[str] = set()
        self._stats_cache: dict[str, dict] = {}
        self._index_cache: dict[str, dict] = {}
        self._session_cache: dict[Path, tuple[int, int, dict]] = {}
        self._closed = False

    # -- workspaces ------------------------------------------------------
    def _load_external(self) -> dict[str, dict]:
        """The registry of folders opened in place, validated on read."""
        data = read_json(self.workspaces_file, {})
        raw = data.get("workspaces") if isinstance(data, dict) else {}
        entries: dict[str, dict] = {}
        if not isinstance(raw, dict):
            return entries
        for key, value in raw.items():
            workspace_id = str(key)
            if not WORKSPACE_ID.fullmatch(workspace_id) or not isinstance(value, dict):
                continue
            root = value.get("root")
            if not isinstance(root, str) or not root.strip():
                continue
            context = value.get("context")
            context = context if isinstance(context, dict) else {}
            entries[workspace_id] = {
                "id": workspace_id,
                "name": str(value.get("name") or workspace_id)[:80],
                "root": root,
                "created": _stamp(value.get("created"), 0.0),
                "updated": _stamp(value.get("updated"), 0.0),
                "excluded": [item for item in value.get("excluded") or []
                             if isinstance(item, str)][:MAX_SETTING_PATHS],
                "context": {
                    "include": [item for item in context.get("include") or []
                                if isinstance(item, str)][:MAX_SETTING_PATHS],
                    "exclude": [item for item in context.get("exclude") or []
                                if isinstance(item, str)][:MAX_SETTING_PATHS],
                },
            }
        return entries

    def _external_entries(self) -> dict[str, dict]:
        with self._lock:
            cached = self._external_cache
        if cached is not None:
            return cached
        entries = self._load_external()
        with self._lock:
            if self._external_cache is None:
                self._external_cache = entries
            return self._external_cache

    def _save_external(self, entries: dict[str, dict]) -> None:
        write_json(self.workspaces_file, {"version": 1, "workspaces": entries})
        with self._lock:
            self._external_cache = entries

    def _external_entry(self, workspace_id: str) -> dict | None:
        return self._external_entries().get(str(workspace_id or ""))

    def _inside_agent_workspace(self, path: Path) -> bool:
        try:
            path.resolve().relative_to(self.agent_workspace)
            return True
        except (OSError, ValueError):
            return False

    def _private_subtree(self, workspace: Path) -> Path | None:
        """Studio's runtime folder when it is nested inside an in-place project."""
        if self._inside_agent_workspace(workspace):
            return None
        try:
            self.agent_workspace.relative_to(workspace)
        except ValueError:
            return None
        return self.agent_workspace

    def _tool_deny(self, workspace_id: str) -> list[Path]:
        """Subtrees abilities must not read; managed workspaces are not affected."""
        workspace = self.workspace_path(workspace_id)
        return [] if self._inside_agent_workspace(workspace) else [self.agent_workspace]

    def workspace_path(self, workspace_id: str) -> Path:
        workspace_id = str(workspace_id or "")
        if not WORKSPACE_ID.fullmatch(workspace_id):
            raise NotFoundError("Workspace not found")
        managed = self.workspaces_dir / workspace_id
        if managed.is_dir():
            path = managed.resolve()
            try:
                path.relative_to(self.workspaces_dir)
            except ValueError:
                raise NotFoundError("Workspace not found") from None
            return path
        entry = self._external_entry(workspace_id)
        if entry is None:
            raise NotFoundError("Workspace not found; open a folder from the IDE toolbar")
        try:
            path = Path(entry["root"]).expanduser().resolve(strict=True)
        except OSError:
            raise NotFoundError("The workspace folder is missing; open it again") from None
        if not path.is_dir() or self._inside_agent_workspace(path):
            raise NotFoundError("Workspace not found")
        return path

    def _workspace_entry(self, workspace_id: str, relative: str, for_write: bool = False) -> Path:
        """Resolve one sandboxed file or folder; shared by every file operation."""
        workspace = self.workspace_path(workspace_id)
        value = str(relative or "").strip()
        if not value:
            raise BadRequestError("A file path is required")
        candidate = Path(value)
        candidate = candidate if candidate.is_absolute() else (workspace / candidate)
        if candidate.is_symlink():
            raise BadRequestError("Symlinked files are not editable in the IDE")
        candidate = candidate.resolve()
        try:
            candidate.relative_to(workspace)
        except ValueError:
            raise BadRequestError("Paths must stay inside this IDE workspace") from None
        if (not self._inside_agent_workspace(workspace)
                and self._inside_agent_workspace(candidate)):
            raise BadRequestError("This folder is private to Mariana Studio")
        parts = candidate.relative_to(workspace).parts
        if not parts:
            raise BadRequestError("A file path is required")
        if any(part in IGNORED_DIRS for part in parts):
            raise BadRequestError("This folder is ignored by the IDE")
        if parts == (MANIFEST,):
            raise BadRequestError("The workspace manifest is not editable")
        if for_write:
            candidate.parent.mkdir(parents=True, exist_ok=True)
        return candidate

    def _workspace_file(self, workspace_id: str, relative: str, for_write: bool = False) -> Path:
        return self._workspace_entry(workspace_id, relative, for_write=for_write)

    def _settings(self, workspace_id: str) -> dict:
        """Workspace context rules: hidden paths and default pinned context."""
        manifest = self._manifest(workspace_id)
        excluded = manifest.get("excluded")
        excluded = [str(item) for item in excluded] if isinstance(excluded, list) else []
        context = manifest.get("context")
        context = dict(context) if isinstance(context, dict) else {}
        include = context.get("include")
        exclude = context.get("exclude")
        return {
            "excluded": [item for item in excluded if isinstance(item, str) and item][:MAX_SETTING_PATHS],
            "context": {
                "include": [item for item in include if isinstance(item, str) and item][:MAX_SETTING_PATHS]
                if isinstance(include, list) else [],
                "exclude": [item for item in exclude if isinstance(item, str) and item][:MAX_SETTING_PATHS]
                if isinstance(exclude, list) else [],
            },
        }

    def settings(self, workspace_id: str) -> dict:
        """The context rules for one workspace."""
        self.workspace_path(workspace_id)
        return self._settings(workspace_id)

    def _apply_settings(self, manifest: dict, payload: dict) -> dict:
        """Validate context-rule changes without writing them yet."""
        manifest = dict(manifest)
        if "excluded" in payload:
            manifest["excluded"] = _path_list(payload.get("excluded"), "excluded")
        if "context" in payload:
            supplied = payload.get("context")
            if not isinstance(supplied, dict):
                raise BadRequestError("context must be an object with include and exclude lists")
            current = manifest.get("context")
            current = dict(current) if isinstance(current, dict) else {}
            if "include" in supplied:
                current["include"] = _path_list(supplied.get("include"), "context include")
            if "exclude" in supplied:
                current["exclude"] = _path_list(supplied.get("exclude"), "context exclude")
            manifest["context"] = current
        manifest["updated"] = time.time()
        return manifest

    def update_settings(self, workspace_id: str, payload: dict) -> dict:
        """Replace the context rules a person controls for this workspace."""
        self.workspace_path(workspace_id)
        entry = self._external_entry(workspace_id)
        if entry is not None:
            # An in-place project never gets a manifest: its rules live in the registry.
            entries = dict(self._external_entries())
            entries[workspace_id] = self._apply_settings(entry, payload)
            self._save_external(entries)
        else:
            path = self.workspaces_dir / workspace_id / MANIFEST
            manifest = read_json(path, {})
            manifest = manifest if isinstance(manifest, dict) else {}
            write_json(path, self._apply_settings(manifest, payload))
        with self._lock:
            self._stats_cache.pop(workspace_id, None)
            self._index_cache.pop(workspace_id, None)
        return self._settings(workspace_id)

    def _excluded_paths(self, workspace_id: str, context: dict | None = None) -> list[str]:
        rules = self._settings(workspace_id)
        context = context if isinstance(context, dict) else {}
        session_exclude = context.get("exclude") if isinstance(context.get("exclude"), list) else []
        values = list(rules["excluded"]) + list(rules["context"]["exclude"]) + list(session_exclude)
        return list(dict.fromkeys(str(item) for item in values if item))

    def _context_include(self, workspace_id: str, context: dict | None = None) -> list[str]:
        rules = self._settings(workspace_id)
        context = context if isinstance(context, dict) else {}
        supplied = context.get("include")
        if isinstance(supplied, list):
            # An explicit empty list clears a workspace default for this conversation.
            return _path_list(supplied, "context include")
        return list(rules["context"]["include"])

    def _manifest(self, workspace_id: str) -> dict:
        entry = self._external_entry(workspace_id)
        if entry is not None:
            return {
                "id": entry["id"], "name": entry["name"],
                "created": entry["created"], "updated": entry["updated"],
                "external": True, "root": entry["root"],
                "excluded": list(entry["excluded"]),
                "context": dict(entry["context"]),
            }
        data = read_json(self.workspaces_dir / workspace_id / MANIFEST, {})
        return data if isinstance(data, dict) else {}

    def _touch(self, workspace_id: str) -> None:
        entry = self._external_entry(workspace_id)
        if entry is not None:
            entries = dict(self._external_entries())
            entries[workspace_id] = {**entry, "updated": time.time()}
            self._save_external(entries)
            with self._lock:
                self._stats_cache.pop(workspace_id, None)
                self._index_cache.pop(workspace_id, None)
            return
        path = self.workspaces_dir / workspace_id / MANIFEST
        if not path.parent.is_dir():
            return  # a deleted workspace must not come back because of a stale session
        manifest = read_json(path, {})
        manifest = manifest if isinstance(manifest, dict) else {}
        manifest["updated"] = time.time()
        write_json(path, manifest)
        with self._lock:
            self._stats_cache.pop(workspace_id, None)
            self._index_cache.pop(workspace_id, None)

    def _describe(self, workspace_id: str, with_stats: bool = True) -> dict:
        manifest = self._manifest(workspace_id)
        external = bool(manifest.get("external"))
        try:
            path = self.workspace_path(workspace_id)
        except NotFoundError:
            if not external:
                raise
            path = None
        data = {
            "id": workspace_id,
            "name": manifest.get("name") or workspace_id.replace("-", " ").title(),
            "created": _stamp(manifest.get("created"), 0.0),
            "updated": _stamp(manifest.get("updated"), 0.0),
            "external": external,
        }
        if external:
            data["root"] = str(manifest.get("root") or "")
        if path is None:
            data["missing"] = True
        else:
            try:
                created, modified = path.stat().st_ctime, path.stat().st_mtime
            except OSError:
                created = modified = 0.0
            data["created"] = _stamp(manifest.get("created"), created)
            data["updated"] = _stamp(manifest.get("updated"), modified)
        if with_stats:
            if path is None:
                data.update({"files": 0, "bytes": 0, "languages": []})
            else:
                try:
                    stats = self._stats(workspace_id)
                except (OSError, NotFoundError):
                    stats = {"files": 0, "bytes": 0, "languages": []}
                data.update({"files": stats["files"], "bytes": stats["bytes"],
                             "languages": stats["languages"]})
        return data

    def list_workspaces(self) -> list[dict]:
        items = []
        seen = set()
        for path in self.workspaces_dir.iterdir():
            if path.is_dir() and WORKSPACE_ID.fullmatch(path.name):
                try:
                    items.append(self._describe(path.name))
                    seen.add(path.name)
                except (OSError, ValueError):
                    continue
        for workspace_id in self._external_entries():
            if workspace_id in seen:
                continue
            try:
                items.append(self._describe(workspace_id))
            except (OSError, ValueError):
                continue
        items.sort(key=lambda item: -(item.get("updated") or 0))
        return items

    def describe_workspace(self, workspace_id: str) -> dict:
        """One workspace with its files, bytes, language mix and context rules."""
        self.workspace_path(workspace_id)
        data = self._describe(workspace_id)
        data.update(self._settings(workspace_id))
        return data

    def create_workspace(self, name: str, source: str | None = None, template: str = "empty") -> dict:
        name = str(name or "").strip()[:80]
        if not name:
            raise BadRequestError("Give the workspace a name")
        template = choice({"template": template}, "template", "empty", ("empty", "python"))
        source = str(source or "").strip()[:4096]
        base = slugify(name, "workspace")[:56].strip("-") or "workspace"
        external = self._external_entries()
        workspace_id, counter = base, 2
        while ((self.workspaces_dir / workspace_id).exists()
               or workspace_id in external):
            workspace_id = f"{base}-{counter}"
            counter += 1
        destination = self.workspaces_dir / workspace_id
        destination.mkdir(parents=True)
        try:
            write_json(destination / MANIFEST, {
                "id": workspace_id, "name": name, "created": time.time(), "updated": time.time(),
            })
            if source:
                self._import_folder(Path(source).expanduser(), destination)
            elif template == "python":
                self._seed_python(destination)
        except Exception:
            shutil.rmtree(destination, ignore_errors=True)
            raise
        with self._lock:
            self._stats_cache.pop(workspace_id, None)
            self._index_cache.pop(workspace_id, None)
        return self._describe(workspace_id)

    def open_workspace(self, name: str | None, source: str) -> dict:
        """Register an existing folder and edit it in place; nothing is copied."""
        source = str(source or "").strip()[:4096]
        if not source:
            raise BadRequestError("Choose the project folder to open")
        candidate = Path(source).expanduser()
        if not candidate.is_absolute():
            raise BadRequestError("Choose an absolute project folder")
        try:
            root = candidate.resolve(strict=True)
        except OSError:
            raise BadRequestError(f"Folder not found: {source}") from None
        if not root.is_dir():
            raise BadRequestError("The selected path is not a folder")
        if self._inside_agent_workspace(root):
            raise BadRequestError(
                "Choose a project folder outside agent-workspace; its chats, "
                "sessions and settings are private"
            )
        entries = dict(self._external_entries())
        for workspace_id, entry in entries.items():
            try:
                if Path(entry["root"]).expanduser().resolve() == root:
                    return self._describe(workspace_id)
            except OSError:
                continue
        label = str(name or "").strip()[:80] or root.name or "workspace"
        base = slugify(label, "workspace")[:56].strip("-") or "workspace"
        workspace_id, counter = base, 2
        while (self.workspaces_dir / workspace_id).exists() or workspace_id in entries:
            workspace_id = f"{base}-{counter}"
            counter += 1
        now = time.time()
        entries[workspace_id] = {
            "id": workspace_id, "name": label, "root": str(root),
            "created": now, "updated": now, "excluded": [],
            "context": {"include": [], "exclude": []},
        }
        self._save_external(entries)
        with self._lock:
            self._stats_cache.pop(workspace_id, None)
            self._index_cache.pop(workspace_id, None)
        return self._describe(workspace_id)

    def _import_folder(self, source: Path, destination: Path) -> None:
        source = source.resolve()
        if not source.is_dir():
            raise BadRequestError(f"Folder not found: {source}")
        try:
            source.relative_to(self.agent_workspace)
        except ValueError:
            pass
        else:
            raise BadRequestError(
                "Choose a project folder outside agent-workspace; its chats, "
                "sessions and settings are private"
            )
        try:
            source.relative_to(destination.resolve())
        except ValueError:
            pass
        else:
            raise BadRequestError("Choose a folder outside the IDE workspace")
        files = total = 0
        for folder, directories, names in os.walk(source):
            directories[:] = sorted(
                name for name in directories
                if name not in IMPORT_IGNORED and not (Path(folder) / name).is_symlink()
            )
            for name in sorted(names):
                origin = Path(folder) / name
                if name == MANIFEST or origin.is_symlink() or not origin.is_file():
                    continue
                size = origin.stat().st_size
                files += 1
                total += size
                if files > IMPORT_MAX_FILES or total > IMPORT_MAX_BYTES:
                    raise BadRequestError(
                        "This folder is too large to import; the limit is "
                        f"{IMPORT_MAX_FILES} files and {IMPORT_MAX_BYTES // 1_000_000} MB"
                    )
                target = destination / origin.relative_to(source)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(origin, target)

    def _seed_python(self, destination: Path) -> None:
        (destination / "main.py").write_text(
            '"""A small starting point for the Mariana IDE."""\n\n\n'
            "def greet(name: str) -> str:\n"
            '    """Return a friendly greeting."""\n'
            '    return f"Hello, {name}!"\n\n\n'
            'if __name__ == "__main__":\n'
            '    print(greet("world"))\n',
            encoding="utf-8",
        )
        (destination / "test_main.py").write_text(
            "from main import greet\n\n\n"
            "def test_greeting_uses_the_name():\n"
            '    assert greet("Mariana") == "Hello, Mariana!"\n',
            encoding="utf-8",
        )
        (destination / "README.md").write_text(
            "# New workspace\n\n"
            "Ask the coding agent to change this project, or edit files directly.\n\n"
            "Run the tests with `python -m pytest` when shell access is enabled.\n",
            encoding="utf-8",
        )
        (destination / ".gitignore").write_text("__pycache__/\n.pytest_cache/\n", encoding="utf-8")

    def delete_workspace(self, workspace_id: str) -> bool:
        entries = dict(self._external_entries())
        external = workspace_id in entries
        path = None if external else self.workspace_path(workspace_id)
        with self._lock:
            if any(run.workspace_id == workspace_id and run.status not in FINISHED
                   for run in self._runs.values()):
                raise ConflictError("Stop the active IDE run before deleting this workspace")
            sessions = [
                data.get("id") for data in self._load_sessions(newest=None)
                if data.get("workspace_id") == workspace_id
                and SESSION_ID.fullmatch(str(data.get("id") or ""))
            ]
        if external:
            # An in-place folder is only unregistered; its files are never deleted.
            entries.pop(workspace_id, None)
            self._save_external(entries)
        else:
            shutil.rmtree(path, ignore_errors=True)
        for session_id in sessions:
            self._session_path(session_id).unlink(missing_ok=True)
            shutil.rmtree(self.snapshots_dir / str(session_id), ignore_errors=True)
        self._forget_sessions({str(item) for item in sessions})
        with self._lock:
            self._stats_cache.pop(workspace_id, None)
            self._index_cache.pop(workspace_id, None)
        return True

    def _stats(self, workspace_id: str) -> dict:
        with self._lock:
            cached = self._stats_cache.get(workspace_id)
        if cached:
            return cached
        workspace = self.workspace_path(workspace_id)
        private = self._private_subtree(workspace)
        files = total = 0
        languages: Counter = Counter()
        try:
            for folder, directories, names in os.walk(workspace):
                directories[:] = sorted(
                    name for name in directories
                    if name not in IGNORED_DIRS
                    and not (Path(folder) / name).is_symlink()
                    and (private is None or (Path(folder) / name) != private)
                )
                for name in sorted(names):
                    path = Path(folder) / name
                    if (name == MANIFEST or path.is_symlink() or not path.is_file()
                            or (name.startswith(".") and name.endswith(".tmp"))):
                        continue
                    files += 1
                    try:
                        total += path.stat().st_size
                    except OSError:
                        pass
                    languages[_language(path)] += 1
                    if files >= 5000:
                        break
                if files >= 5000:
                    break
        except OSError:
            pass
        stats = {
            "files": files, "bytes": total,
            "languages": [name for name, _ in languages.most_common(6)],
        }
        with self._lock:
            self._stats_cache[workspace_id] = stats
        return stats

    # -- files -----------------------------------------------------------
    def _visible_entries(self, folder: Path, workspace: Path, excluded: list[str]) -> list[Path]:
        """Sorted, sandbox-safe children of one folder, with hidden rules applied."""
        try:
            entries = sorted(folder.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower()))
        except OSError:
            return []
        visible = []
        private = self._private_subtree(workspace)
        for entry in entries:
            if (entry.name in IGNORED_DIRS or entry.name == MANIFEST or entry.is_symlink()
                    or entry == private
                    or (entry.name.startswith(".") and entry.name.endswith(".tmp"))):
                continue
            try:
                relative = entry.relative_to(workspace).as_posix()
            except ValueError:
                continue
            if _under(relative, excluded):
                continue
            visible.append(entry)
        return visible

    def tree(self, workspace_id: str, path: str = "", depth: int = 0) -> dict:
        """The project tree; ``depth`` above zero lists one level for lazy loading."""
        workspace = self.workspace_path(workspace_id)
        root = workspace
        relative_root = ""
        if str(path or "").strip() not in {"", ".", "./"}:
            root = self._workspace_entry(workspace_id, path)
            if not root.is_dir():
                raise BadRequestError(f"{path} is a file, not a folder")
            relative_root = root.relative_to(workspace).as_posix()
        depth = max(0, min(MAX_TREE_DEPTH, int(depth or 0)))
        excluded = self._excluded_paths(workspace_id)
        state = {"entries": 0, "truncated": False}

        def walk(folder: Path, level: int) -> list[dict]:
            children = []
            for entry in self._visible_entries(folder, workspace, excluded):
                if state["entries"] >= MAX_TREE_ENTRIES:
                    state["truncated"] = True
                    break
                state["entries"] += 1
                relative = entry.relative_to(workspace).as_posix()
                if entry.is_dir():
                    lazy = depth > 0 and level + 1 >= depth
                    if lazy:
                        has_children = bool(self._visible_entries(entry, workspace, excluded))
                        child_nodes, child_loaded = [], False
                        if not has_children:
                            child_loaded = True
                    else:
                        child_nodes = walk(entry, level + 1) if level < MAX_TREE_DEPTH else []
                        child_loaded = True
                        has_children = bool(child_nodes)
                        if level >= MAX_TREE_DEPTH:
                            state["truncated"] = True
                    children.append({
                        "name": entry.name, "path": relative, "type": "dir",
                        "children": child_nodes, "loaded": child_loaded,
                        "has_children": has_children,
                    })
                elif entry.is_file():
                    try:
                        size = entry.stat().st_size
                    except OSError:
                        continue
                    children.append({
                        "name": entry.name, "path": relative, "type": "file",
                        "bytes": size, "language": _language(entry),
                    })
            return children

        children = walk(root, 0)
        data = {
            "workspace": self._describe(workspace_id, with_stats=depth == 0),
            "path": relative_root,
            "depth": depth,
            "children": children,
            "truncated": state["truncated"],
            "excluded": excluded,
        }
        if depth == 0:
            stats = self._stats(workspace_id)
            data["files"] = stats["files"]
            data["bytes"] = stats["bytes"]
        return data

    def _path_index(self, workspace_id: str) -> dict:
        """A cached name index so filters and @-mentions do not re-walk the project."""
        with self._lock:
            cached = self._index_cache.get(workspace_id)
        if cached is not None:
            return cached
        workspace = self.workspace_path(workspace_id)
        private = self._private_subtree(workspace)
        files: list[dict] = []
        folders: list[dict] = []
        truncated = False
        try:
            for folder, directories, names in os.walk(workspace):
                relative_folder = Path(folder).relative_to(workspace).as_posix()
                base = "" if relative_folder == "." else f"{relative_folder}/"
                directories[:] = sorted(
                    name for name in directories
                    if name not in IGNORED_DIRS and not (Path(folder) / name).is_symlink()
                    and (private is None or (Path(folder) / name) != private)
                )
                for name in directories:
                    folders.append({"path": f"{base}{name}", "name": name})
                for name in sorted(names):
                    origin = Path(folder) / name
                    if (name == MANIFEST or origin.is_symlink() or not origin.is_file()
                            or (name.startswith(".") and name.endswith(".tmp"))):
                        continue
                    try:
                        size = origin.stat().st_size
                    except OSError:
                        continue
                    files.append({"path": f"{base}{name}", "name": name, "bytes": size,
                                  "language": _language(origin)})
                    if len(files) >= MAX_INDEX_ENTRIES:
                        truncated = True
                        break
                if len(files) >= MAX_INDEX_ENTRIES:
                    break
        except OSError:
            pass
        files.sort(key=lambda item: item["path"])
        folders.sort(key=lambda item: item["path"])
        index = {"files": files, "folders": folders, "truncated": truncated}
        with self._lock:
            self._index_cache[workspace_id] = index
        return index

    def find_files(self, workspace_id: str, query: str = "", limit: int = 200) -> dict:
        """Fast path matching for the explorer filter and the @-mention picker."""
        self.workspace_path(workspace_id)
        needle = str(query or "").strip().lower()[:200]
        limit = max(1, min(MAX_FIND_RESULTS, int(limit or 200)))
        excluded = self._excluded_paths(workspace_id)
        index = self._path_index(workspace_id)
        files = [item for item in index["files"]
                 if not _under(item["path"], excluded)
                 and (not needle or needle in item["path"].lower())]
        folders = [item for item in index["folders"]
                   if not _under(item["path"], excluded)
                   and (not needle or needle in item["path"].lower())]
        truncated = index["truncated"] or len(files) > limit or len(folders) > limit
        return {"files": files[:limit], "folders": folders[:limit], "truncated": truncated,
                "query": needle, "excluded": excluded}

    def preview_file(self, workspace_id: str, relative: str, start: int = 0,
                     limit: int = 0) -> dict:
        """Metadata plus a source, image or hex preview for any file.

        Text uses the same version and line-range contract as :meth:`read_file`,
        so the editor can save it.  Binary files stay read-only and carry a hex
        dump, a safe data URL when they are images, and header dimensions.
        """
        path = self._workspace_file(workspace_id, relative)
        if not path.is_file():
            raise NotFoundError(f"File not found: {relative}")
        size = path.stat().st_size
        mime = _mime(path)
        language = _language(path)
        render = _render_kind(path, mime)
        with path.open("rb") as handle:
            payload = handle.read(MAX_PREVIEW_BYTES)
        complete = size <= MAX_PREVIEW_BYTES
        bitmap = mime.startswith("image/") and mime != "image/svg+xml"
        text = _looks_text(payload) and not bitmap
        data = {
            "path": path.relative_to(self.workspace_path(workspace_id)).as_posix(),
            "name": path.name,
            "bytes": size,
            "modified": _iso(path.stat().st_mtime),
            "mime": mime,
            "language": language,
            "render": render,
            "kind": "text" if text else "binary",
            "editable": False,
            "truncated": not complete,
            "hex": _hex_dump(payload),
            "hex_bytes": min(len(payload), MAX_HEX_BYTES),
            "version": None,
            "content": None,
            "total_lines": None,
            "start_line": 1,
            "end_line": 0,
        }
        if text:
            content = payload[:MAX_PREVIEW_TEXT_BYTES].decode("utf-8", errors="replace")
            text_truncated = not complete or size > MAX_PREVIEW_TEXT_BYTES
            total_lines = content.count("\n") + 1 if not text_truncated else None
            start = max(0, int(start or 0))
            limit = max(0, min(MAX_READ_LINES, int(limit or 0)))
            if total_lines is not None:
                start = min(start, max(0, total_lines - 1))
            if limit:
                lines = content.split("\n")
                content = "\n".join(lines[start:start + limit])
                truncated = text_truncated or start + limit < len(lines)
            else:
                truncated = text_truncated
            end_line = start + content.count("\n") + 1
            data.update({
                "content": content,
                "version": _version(payload) if complete and not text_truncated else None,
                "editable": complete and not text_truncated and size <= MAX_FILE_BYTES,
                "truncated": truncated,
                "total_lines": total_lines,
                "start_line": start + 1,
                "end_line": end_line if content else start + 1,
            })
            if mime == "image/svg+xml" and size <= MAX_PREVIEW_BYTES:
                data["data_url"] = "data:image/svg+xml;base64," + base64.b64encode(payload).decode("ascii")
        else:
            data["dimensions"] = _image_dimensions(payload, mime)
            if (bitmap or mime == "image/svg+xml") and size <= MAX_PREVIEW_BYTES:
                data["data_url"] = f"data:{mime};base64," + base64.b64encode(payload).decode("ascii")
        return data

    def read_file(self, workspace_id: str, relative: str, start: int = 0,
                  limit: int = 0) -> dict:
        """One UTF-8 file, or a line range for a file too large to open at once."""
        path = self._workspace_file(workspace_id, relative)
        if not path.is_file():
            raise NotFoundError(f"File not found: {relative}")
        size = path.stat().st_size
        if size > MAX_FILE_BYTES:
            raise BadRequestError(
                f"{relative} is {size // 1000} KB; the editor opens files up to "
                f"{MAX_FILE_BYTES // 1000} KB"
            )
        payload = path.read_bytes()
        if b"\x00" in payload[:4096]:
            raise BadRequestError(f"{relative} looks binary; the IDE editor handles text files")
        try:
            content = payload.decode("utf-8")
        except UnicodeDecodeError:
            raise BadRequestError(f"{relative} is not UTF-8 text") from None
        total_lines = content.count("\n") + 1
        start = max(0, min(total_lines - 1, int(start or 0)))
        limit = max(0, min(MAX_READ_LINES, int(limit or 0)))
        truncated = False
        if limit and start + limit < total_lines:
            lines = content.split("\n")
            content = "\n".join(lines[start:start + limit])
            truncated = True
        elif limit:
            lines = content.split("\n")
            content = "\n".join(lines[start:start + limit])
        end_line = start + (content.count("\n") + 1)
        return {
            "path": path.relative_to(self.workspace_path(workspace_id)).as_posix(),
            "content": content,
            "version": _version(payload),
            "bytes": size,
            "language": _language(path),
            "modified": _iso(path.stat().st_mtime),
            "total_lines": total_lines,
            "start_line": start + 1,
            "end_line": end_line,
            "truncated": truncated,
        }

    def write_file(self, workspace_id: str, relative: str, content: str,
                   version: str | None = None, force: bool = False) -> dict:
        if not isinstance(content, str):
            raise BadRequestError("content must be a string")
        path = self._workspace_file(workspace_id, relative, for_write=True)
        if path.exists() and not path.is_file():
            raise BadRequestError(f"{relative} is a folder, not a file")
        payload = content.encode("utf-8")
        if len(payload) > MAX_FILE_BYTES:
            raise BadRequestError(f"Content exceeds the {MAX_FILE_BYTES // 1000} KB editor limit")
        if path.exists():
            if not force:
                if version is None:
                    raise ConflictError(
                        f"{relative} already exists; reload it before saving to avoid overwriting changes"
                    )
                if _file_version(path) != str(version):
                    raise ConflictError(
                        f"{relative} changed on disk; reload it before saving, or save with force"
                    )
        elif version not in (None, ""):
            raise ConflictError(f"{relative} was deleted on disk; create it again from the editor")
        _atomic_write(path, payload)
        self._touch(workspace_id)
        return {
            "path": path.relative_to(self.workspace_path(workspace_id)).as_posix(),
            "version": _version(payload),
            "bytes": len(payload),
            "language": _language(path),
            "modified": _iso(),
            "written": True,
        }

    def delete_file(self, workspace_id: str, relative: str) -> dict:
        path = self._workspace_file(workspace_id, relative)
        if not path.is_file():
            raise NotFoundError(f"File not found: {relative}")
        path.unlink()
        self._touch(workspace_id)
        return {"path": relative, "deleted": True}

    def create_folder(self, workspace_id: str, relative: str) -> dict:
        """Create one folder; an existing folder is reported instead of failing."""
        path = self._workspace_entry(workspace_id, relative)
        if path.exists() and not path.is_dir():
            raise ConflictError(f"{relative} already exists as a file")
        path.mkdir(parents=True, exist_ok=True)
        self._touch(workspace_id)
        return {"path": path.relative_to(self.workspace_path(workspace_id)).as_posix(),
                "created": True}

    def delete_folder(self, workspace_id: str, relative: str) -> dict:
        """Delete a folder and everything inside it, but never the workspace root."""
        path = self._workspace_entry(workspace_id, relative)
        if path == self.workspace_path(workspace_id):
            raise BadRequestError("The workspace root cannot be deleted from the explorer")
        if not path.is_dir():
            raise NotFoundError(f"Folder not found: {relative}")
        shutil.rmtree(path)
        self._touch(workspace_id)
        return {"path": relative, "deleted": True}

    def move_entry(self, workspace_id: str, source: str, destination: str) -> dict:
        """Rename or move a file or folder inside the workspace."""
        origin = self._workspace_entry(workspace_id, source)
        target = self._workspace_entry(workspace_id, destination)
        if not origin.exists():
            raise NotFoundError(f"Path not found: {source}")
        if origin == self.workspace_path(workspace_id):
            raise BadRequestError("The workspace root cannot be moved")
        if target.exists():
            raise ConflictError(f"{destination} already exists; choose another name")
        if origin.is_dir() and (target == origin or origin in target.parents):
            raise BadRequestError("A folder cannot be moved inside itself")
        target.parent.mkdir(parents=True, exist_ok=True)
        origin.rename(target)
        self._touch(workspace_id)
        return {
            "from": origin.relative_to(self.workspace_path(workspace_id)).as_posix(),
            "to": target.relative_to(self.workspace_path(workspace_id)).as_posix(),
            "moved": True,
        }

    # -- pinned context ---------------------------------------------------
    def _context_candidates(self, workspace_id: str, include: list[str],
                            exclude: list[str],
                            limit: int = MAX_CONTEXT_FILES + 1) -> list[Path]:
        workspace = self.workspace_path(workspace_id)
        private = self._private_subtree(workspace)
        if not include:
            return []  # the map names the project; only pinned paths add file contents
        roots: list[Path] = []
        for item in include:
            try:
                candidate = self._workspace_entry(workspace_id, item)
            except (BadRequestError, NotFoundError):
                continue
            if candidate.exists():
                roots.append(candidate)
        files: list[Path] = []
        exhausted = False
        for root in roots:
            if exhausted:
                break
            if root.is_file():
                relative = root.relative_to(workspace).as_posix()
                if not _under(relative, exclude):
                    files.append(root)
                    if len(files) >= limit:
                        break
                continue
            for folder, directories, names in os.walk(root):
                directories[:] = sorted(
                    name for name in directories
                    if name not in IGNORED_DIRS and not (Path(folder) / name).is_symlink()
                    and (private is None or (Path(folder) / name) != private)
                )
                for name in sorted(names):
                    if name == MANIFEST or name.endswith(".tmp"):
                        continue
                    path = Path(folder) / name
                    if not path.is_file() or path.is_symlink():
                        continue
                    try:
                        relative = path.relative_to(workspace).as_posix()
                    except ValueError:
                        continue
                    if _under(relative, exclude):
                        continue
                    files.append(path)
                    if len(files) >= limit:
                        exhausted = True
                        break
                if exhausted:
                    break
        unique: dict[str, Path] = {}
        for path in files:
            unique[path.relative_to(workspace).as_posix()] = path
        return [unique[key] for key in sorted(unique)]

    def _context_scope(self, workspace_id: str, payload: dict) -> tuple[list[str], list[str], dict]:
        """Resolve pinned and hidden paths from a request without touching files."""
        supplied = payload.get("context")
        context = supplied if isinstance(supplied, dict) else payload
        include = _path_list(context.get("include"), "context include") \
            if "include" in context else self._context_include(workspace_id)
        exclude = self._excluded_paths(workspace_id, context)
        return include, exclude, context

    def context_preview(self, workspace_id: str, payload: dict | None = None) -> dict:
        """What the agent will read for the current context rules, without sending tokens."""
        self.workspace_path(workspace_id)
        payload = payload if isinstance(payload, dict) else {}
        include, exclude, _context = self._context_scope(workspace_id, payload)
        workspace = self.workspace_path(workspace_id)
        candidates = self._context_candidates(workspace_id, include, exclude)
        selected = candidates[:MAX_CONTEXT_FILES]
        chars = 0
        files = []
        for path in selected:
            try:
                size = path.stat().st_size
            except OSError:
                continue
            budget = min(size, MAX_CONTEXT_FILE_CHARS)
            chars += budget
            files.append({
                "path": path.relative_to(workspace).as_posix(),
                "bytes": size,
                "included_chars": budget,
            })
        budget = self.context_budget(workspace_id, payload)
        unit = budget.get("unit", "tokens")
        for item in files:
            item["tokens"] = item["included_chars"] if unit == "bytes" \
                else max(0, item["included_chars"] // 4)
        budget["pinned"] = {
            "file_count": len(candidates), "selected": len(selected),
            "chars": chars, "truncated": len(candidates) > len(selected),
        }
        if len(candidates) > len(selected):
            budget["warnings"].append({
                "tone": "warning",
                "text": f"Only the first {MAX_CONTEXT_FILES} pinned files are included; "
                        "unpin files to reach the rest.",
            })
        if chars > MAX_CONTEXT_CHARS:
            budget["warnings"].append({
                "tone": "warning",
                "text": f"Pinned context is capped at {MAX_CONTEXT_CHARS:,} characters; "
                        "later files are omitted.",
            })
        return {
            "include": include,
            "exclude": exclude,
            "files": files,
            "file_count": len(candidates),
            "selected": len(selected),
            "chars": chars,
            "tokens": max(0, chars // 4),
            "truncated": len(candidates) > len(selected),
            "limits": {"files": MAX_CONTEXT_FILES, "chars": MAX_CONTEXT_CHARS},
            "budget": budget,
        }

    def _context_blocks(self, workspace_id: str, include: list[str],
                        exclude: list[str]) -> str:
        """Read the pinned context files, bounded so the prompt stays affordable."""
        if not include:
            return ""
        workspace = self.workspace_path(workspace_id)
        blocks: list[str] = []
        used = 0
        for path in self._context_candidates(workspace_id, include, exclude)[:MAX_CONTEXT_FILES]:
            try:
                payload = path.read_bytes()[:MAX_CONTEXT_FILE_CHARS + 1]
            except OSError:
                continue
            if b"\x00" in payload[:4096]:
                continue
            text = payload.decode("utf-8", errors="replace")
            if len(text) > MAX_CONTEXT_FILE_CHARS:
                text = text[:MAX_CONTEXT_FILE_CHARS] + "\n… (context snippet truncated)"
            block = f"--- {path.relative_to(workspace).as_posix()} ---\n{text}"
            if used + len(block) > MAX_CONTEXT_CHARS:
                blocks.append("… (more pinned context omitted to keep the prompt small)")
                break
            blocks.append(block)
            used += len(block)
        return "\n\n".join(blocks)

    def search(self, workspace_id: str, payload: dict) -> dict:
        workspace = self.workspace_path(workspace_id)
        deny = self._tool_deny(workspace_id)
        query = text(payload, "query", "", 300)
        if not query:
            raise BadRequestError("Type something to search for")
        limit = number(payload, "max_results", 100, 1, MAX_SEARCH_RESULTS)
        ability = {"id": "search", "type": "file.search", "config": {}}
        result = execute_ability(ability, {
            "query": query,
            "regex": flag(payload, "regex", False),
            "case_sensitive": flag(payload, "case_sensitive", False),
            "include": text(payload, "include", "", 120),
            "max_results": limit,
        }, {"agent": {"id": "ide"}}, self.root, files_root=workspace, deny=deny)
        excluded = self._excluded_paths(workspace_id)
        if excluded and isinstance(result, dict) and isinstance(result.get("matches"), list):
            result["matches"] = [match for match in result["matches"]
                                 if not _under(str(match.get("path") or ""), excluded)]
        return result

    def diagnostics(self, workspace_id: str, relative: str = "") -> dict:
        workspace = self.workspace_path(workspace_id)
        deny = self._tool_deny(workspace_id)
        relative = text({"path": relative}, "path", "", 300)
        if relative and relative not in {".", "./"}:
            self._workspace_file(workspace_id, relative)
        ability = {"id": "diagnostics", "type": "code.diagnostics", "config": {}}
        return execute_ability(ability, {"path": relative},
                               {"agent": {"id": "ide"}}, self.root, files_root=workspace,
                               deny=deny)

    # -- agent skills ----------------------------------------------------
    def _skill_records(self, workspace_id: str) -> list[dict]:
        """Workspace and user skills, workspace first, without executing anything."""
        workspace = self.workspace_path(workspace_id)
        roots = [(workspace / relative, "workspace") for relative in _SKILL_DIRS]
        roots.extend((root, "user") for root in self.user_skill_roots)
        records: list[dict] = []
        seen: set[str] = set()
        for root, scope in roots:
            try:
                if not root.is_dir() or root.is_symlink():
                    continue
                if scope == "workspace":
                    root.resolve().relative_to(workspace)
                children = sorted(root.iterdir())[:MAX_SKILLS * 2]
            except (OSError, ValueError):
                continue
            for child in children:
                if len(records) >= MAX_SKILLS:
                    break
                if (child.name in seen or not _SKILL_ID.fullmatch(child.name)
                        or child.is_symlink() or not child.is_dir()):
                    continue
                path = child / "SKILL.md"
                try:
                    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_SKILL_BYTES:
                        continue
                    text = path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                name, description = _skill_meta(text, child.name)
                seen.add(child.name)
                records.append({"id": child.name, "name": name, "description": description,
                                "scope": scope, "path": path})
        return records

    def _skill_list_lines(self, skills: list[dict], *, include_scope: bool = False) -> list[str]:
        """Compact skill lines for a prompt; every id stays visible.

        Descriptions are bound to a small section budget so a large skill
        collection cannot crowd pinned files and the active file out of the
        system prompt.  When space runs out the remaining ids are listed so the
        model can still load one with skill.read.
        """
        lines: list[str] = []
        size = 0
        for index, skill in enumerate(skills[:MAX_PROMPT_SKILLS]):
            scope = f" ({skill['scope']})" if include_scope else ""
            name = str(skill["name"])[:80]
            description = str(skill["description"] or "no description")[:120]
            line = f"- {skill['id']}{scope}: {name} — {description}"
            if lines and size + len(line) + 1 > MAX_SKILL_PROMPT_CHARS:
                remaining = [item["id"] for item in skills[index:]]
                if remaining:
                    lines.append("- Also available (call skill.read with an id): "
                                 + ", ".join(remaining))
                break
            lines.append(line)
            size += len(line) + 1
        return lines

    def list_skills(self, workspace_id: str) -> list[dict]:
        """Skills the coding agent can load for this workspace."""
        return [{key: record[key] for key in ("id", "name", "description", "scope")}
                for record in self._skill_records(workspace_id)]

    def _skill_record(self, workspace_id: str, name: str) -> dict:
        """Resolve a skill id or display name to its record."""
        wanted = str(name or "").strip().lower()
        if not wanted:
            raise BadRequestError("Choose a skill by id, for example skill.read {\"name\": \"code-review\"}")
        slug = slugify(wanted, fallback="")
        records = self._skill_records(workspace_id)
        record = next((item for item in records
                       if item["id"] == wanted or item["id"] == slug
                       or item["name"].lower() == wanted), None)
        if record is None:
            available = ", ".join(item["id"] for item in records) or "none"
            raise NotFoundError(f"Skill not found: {name}. Available skills: {available}")
        return record

    def _skill_root(self, workspace_id: str, scope: str) -> Path:
        """The folder that owns one scope's skills; only known roots are writable."""
        scope = str(scope or "workspace").strip().lower()
        if scope == "user":
            if not self.user_skill_roots:
                raise BadRequestError(
                    "No user skills folder is configured; create this skill in the workspace"
                )
            return self.user_skill_roots[0]
        if scope != "workspace":
            raise BadRequestError("Skill scope must be workspace or user")
        root = self.workspace_path(workspace_id) / ".agents" / "skills"
        if root.is_symlink():
            raise BadRequestError("A symlinked skills folder is read-only; remove the link first")
        return root

    def _skill_directory(self, workspace_id: str, scope: str, identifier: str) -> Path:
        root = self._skill_root(workspace_id, scope)
        if not _SKILL_ID.fullmatch(identifier):
            raise BadRequestError(
                "Skill id must use lowercase letters, digits and hyphens, starting with a letter or digit"
            )
        directory = (root / identifier).resolve()
        try:
            directory.relative_to(root.resolve())
        except ValueError:
            raise BadRequestError("Skill id must stay inside the skills folder") from None
        return directory

    def create_skill(self, workspace_id: str, payload: dict) -> dict:
        """Create a workspace or user skill from the editor."""
        self.workspace_path(workspace_id)
        payload = payload if isinstance(payload, dict) else {}
        name = str(payload.get("name") or "").strip()
        if not name:
            raise BadRequestError("Give the skill a name, for example Code review")
        identifier = slugify(str(payload.get("id") or name), fallback="")
        if not identifier:
            raise BadRequestError("Skill id must contain at least one letter or digit")
        scope = str(payload.get("scope") or "workspace").strip().lower()
        directory = self._skill_directory(workspace_id, scope, identifier)
        if directory.exists():
            raise ConflictError(
                f"Skill {identifier!r} already exists; edit it or choose another id"
            )
        content = payload.get("instructions", payload.get("content", ""))
        if not isinstance(content, str):
            raise BadRequestError("Skill instructions must be text")
        description = str(payload.get("description") or "").strip()
        document = _skill_document(name, description, content)
        if len(document.encode("utf-8")) > MAX_SKILL_BYTES:
            raise BadRequestError(f"A skill must be smaller than {MAX_SKILL_BYTES} bytes")
        _atomic_write(directory / "SKILL.md", document.encode("utf-8"))
        return {
            "id": identifier, "name": " ".join(name.split())[:120],
            "description": " ".join(description.split())[:300], "scope": scope,
            "instructions": _skill_body(document),
        }

    def update_skill(self, workspace_id: str, name: str, payload: dict) -> dict:
        """Replace a skill's display fields and instructions without moving it."""
        record = self._skill_record(workspace_id, name)
        payload = payload if isinstance(payload, dict) else {}
        if record["path"].is_symlink():
            raise BadRequestError("Symlinked skills are read-only")
        try:
            current = record["path"].read_text(encoding="utf-8", errors="replace")
        except OSError as error:
            raise NotFoundError(f"Could not read skill {record['id']}: {error}") from None
        body = payload.get("instructions")
        if body is None:
            body = payload.get("content")
        if body is None:
            body = _skill_body(current)
        if not isinstance(body, str):
            raise BadRequestError("Skill instructions must be text")
        display = str(payload.get("name") or record["name"]).strip() or record["name"]
        description = str(payload.get("description") or record["description"]).strip()
        document = _skill_document(display, description, body)
        if len(document.encode("utf-8")) > MAX_SKILL_BYTES:
            raise BadRequestError(f"A skill must be smaller than {MAX_SKILL_BYTES} bytes")
        _atomic_write(record["path"], document.encode("utf-8"))
        return {
            "id": record["id"], "name": " ".join(display.split())[:120],
            "description": " ".join(description.split())[:300], "scope": record["scope"],
            "instructions": _skill_body(document),
        }

    def delete_skill(self, workspace_id: str, name: str) -> dict:
        """Delete one skill directory from a known workspace or user root."""
        record = self._skill_record(workspace_id, name)
        directory = record["path"].parent
        if directory.is_symlink() or record["path"].is_symlink():
            raise BadRequestError("Symlinked skills are read-only")
        shutil.rmtree(directory)
        return {"deleted": True, "id": record["id"], "scope": record["scope"]}

    def read_skill(self, workspace_id: str, name: str, include_instructions: bool = False) -> dict:
        """The full instructions of one skill, bounded and read-only."""
        record = self._skill_record(workspace_id, name)
        try:
            text = record["path"].read_text(encoding="utf-8", errors="replace")
        except OSError as error:
            raise NotFoundError(f"Could not read skill {record['id']}: {error}") from None
        result = {
            "id": record["id"], "name": record["name"], "description": record["description"],
            "scope": record["scope"], "content": text[:MAX_SKILL_CHARS],
            "truncated": len(text) > MAX_SKILL_CHARS,
        }
        if include_instructions:
            result["instructions"] = _skill_body(text)[:MAX_SKILL_CHARS]
        return result

    # -- sessions --------------------------------------------------------
    def _session_path(self, session_id: str) -> Path:
        if not SESSION_ID.fullmatch(str(session_id or "")):
            raise NotFoundError("Conversation not found")
        return self.sessions_dir / f"{session_id}.json"

    def _load_sessions(self, newest: int | None = 200) -> list[dict]:
        """Recent sessions; unchanged files are served from a stat-keyed cache."""
        files = sorted(self.sessions_dir.glob("ide-*.json"), key=_modified, reverse=True)
        if newest is not None:
            files = files[:newest]
        sessions = []
        for path in files:
            try:
                stat = path.stat()
            except OSError:
                continue
            stamp, size = stat.st_mtime_ns, stat.st_size
            with self._lock:
                cached = self._session_cache.get(path)
            if cached is not None and cached[0] == stamp and cached[1] == size:
                data = cached[2]
            else:
                data = read_json(path)
                if not isinstance(data, dict) or not data.get("id"):
                    with self._lock:
                        self._session_cache.pop(path, None)
                    continue
                with self._lock:
                    self._session_cache[path] = (stamp, size, data)
            sessions.append(data)
        with self._lock:
            if len(self._session_cache) > 512:
                self._session_cache.clear()
        return sessions

    def _session_summary(self, session: dict) -> dict:
        with self._lock:
            active = any(run.session_id == session["id"] and run.status not in FINISHED
                         for run in self._runs.values())
        transcript = session.get("transcript")
        transcript = list(transcript) if isinstance(transcript, list) else []
        changes = session.get("changes")
        changes = dict(changes) if isinstance(changes, dict) else {}
        context = session.get("context") if isinstance(session.get("context"), dict) else {}
        pinned_context = context.get("include") if isinstance(context.get("include"), list) else []
        excluded_context = context.get("exclude") if isinstance(context.get("exclude"), list) else []
        usage = session.get("usage")
        usage = dict(usage) if isinstance(usage, dict) else {}
        status = session.get("status", "idle")
        if status == "running" and not active:
            status = "interrupted"
        return {
            "id": session["id"],
            "workspace_id": session.get("workspace_id"),
            "title": session.get("title") or "Untitled session",
            "created": session.get("created"),
            "updated": session.get("updated"),
            "model": session.get("model"),
            "mode": session.get("mode"),
            "status": status,
            "pinned": bool(session.get("pinned")),
            "preview": next((str(item.get("text") or "")[:140] for item in reversed(transcript)
                             if isinstance(item, dict) and item.get("role") == "user"), ""),
            "context": {"include": len(pinned_context), "exclude": len(excluded_context)},
            "turns": sum(1 for item in transcript
                         if isinstance(item, dict) and item.get("role") == "user"),
            "changes": sum(1 for change in changes.values()
                           if isinstance(change, dict) and not change.get("reverted")),
            "usage": usage,
            "active": active,
        }

    def list_sessions(self, workspace_id: str) -> list[dict]:
        self.workspace_path(workspace_id)
        summaries = [self._session_summary(session) for session in self._load_sessions()
                     if session.get("workspace_id") == workspace_id]
        summaries.sort(key=lambda item: (not item.get("pinned"), -(item.get("updated") or 0)))
        return summaries

    def get_session(self, session_id: str, include_messages: bool = False) -> dict:
        data = read_json(self._session_path(session_id))
        if not isinstance(data, dict) or not data.get("id"):
            raise NotFoundError("Conversation not found")
        for key in ("transcript", "messages", "plan", "runs"):
            if not isinstance(data.get(key, []), list):
                data[key] = []
        if not isinstance(data.get("changes", {}), dict):
            data["changes"] = {}
        if not include_messages:
            data.pop("messages", None)
        live = None
        with self._lock:
            for run in self._runs.values():
                if run.session_id == session_id and run.status not in FINISHED:
                    live = run
                    break
        data["active_run_id"] = live.id if live else None
        if data.get("status") == "running" and live is None:
            data["status"] = "interrupted"
        return data

    def delete_session(self, session_id: str) -> bool:
        session = self.get_session(session_id)
        if session.get("active_run_id"):
            raise ConflictError("Stop the active run before deleting this session")
        self._session_path(session_id).unlink(missing_ok=True)
        self._forget_sessions({session_id})
        shutil.rmtree(self.snapshots_dir / session_id, ignore_errors=True)
        return True

    def rename_session(self, session_id: str, payload: dict) -> dict:
        """Rename, pin or update the pinned context of a conversation."""
        session = self.get_session(session_id, include_messages=True)
        if "title" in payload:
            title = str(payload.get("title") or "").strip()
            if not title:
                raise BadRequestError("A conversation title cannot be empty")
            session["title"] = title[:120]
            session["titled"] = True
        if "pinned" in payload:
            session["pinned"] = bool(payload.get("pinned"))
        if "context" in payload:
            session["context"] = self._context_payload(payload.get("context"))
        with self._lock:
            live = next((run for run in self._runs.values()
                         if run.session_id == session_id and run.status not in FINISHED), None)
        if live is not None and live.session is not None:
            # A long run checkpoints the session itself; keep a rename or pin that
            # happened while it was working instead of overwriting it at the end.
            for key in ("title", "titled", "pinned", "context"):
                if key in session:
                    live.session[key] = session[key]
        self._save_session(session)
        return self._session_summary(session)

    def duplicate_session(self, session_id: str) -> dict:
        """Fork a conversation for another attempt; snapshots stay with the original."""
        source = self.get_session(session_id, include_messages=True)
        now = time.time()
        title = str(source.get("title") or "Untitled session")[:100]
        session = {
            "id": f"ide-{uuid.uuid4().hex[:12]}",
            "workspace_id": source.get("workspace_id"),
            "title": f"{title} (copy)",
            "created": now,
            "updated": now,
            "model": source.get("model"),
            "mode": source.get("mode"),
            "messages": copy.deepcopy(source.get("messages") or []),
            "transcript": copy.deepcopy(source.get("transcript") or []),
            "plan": copy.deepcopy(source.get("plan") or []),
            "context": copy.deepcopy(source.get("context") or {}),
            "changes": {},
            "runs": [],
            "status": "idle",
            "pinned": False,
            "titled": True,
        }
        self._save_session(session)
        return self._session_summary(session)

    def _new_session(self, workspace_id: str, message: str, mode: str, model: str,
                     context: dict | None = None) -> dict:
        session = {
            "id": f"ide-{uuid.uuid4().hex[:12]}",
            "workspace_id": workspace_id,
            "title": (message.strip().splitlines() or ["Untitled session"])[0][:80],
            "created": time.time(),
            "updated": time.time(),
            "model": model,
            "mode": mode,
            "messages": [],
            "transcript": [],
            "plan": [],
            "context": context or {},
            "changes": {},
            "runs": [],
            "status": "idle",
            "pinned": False,
        }
        self._save_session(session)
        return session

    def _save_session(self, session: dict) -> None:
        session["updated"] = time.time()
        path = self._session_path(session["id"])
        write_json(path, session, indent=None)
        with self._lock:
            self._session_cache.pop(path, None)

    # -- runs ------------------------------------------------------------
    @property
    def shell_enabled(self) -> bool:
        return shell_enabled() or self.os_access

    @property
    def os_access(self) -> bool:
        """Server-wide opt-in for shell commands and local-network targets."""
        if (os.environ.get("MARIANA_AGENT_ALLOW_OS") == "1"
                or os.environ.get("MARIANA_AGENT_ALLOW_SHELL") == "1"):
            return True
        try:
            return bool(self._server_settings().get("allow_os_access"))
        except Exception:  # noqa: BLE001 - a settings failure keeps the safe default
            return False

    def overview(self) -> dict:
        return {
            "protocol": "IDE",
            "version": IDE_VERSION,
            "workspaces": self.list_workspaces(),
            "modes": [
                {"id": key, **MODE_INFO[key], "tools": list(MODE_INFO[key]["tools"])}
                for key in MODES
            ],
            "shell_enabled": self.shell_enabled,
            "os_access": self.os_access,
            "limits": {
                "file_bytes": MAX_FILE_BYTES,
                "read_lines": MAX_READ_LINES,
                "tree_entries": MAX_TREE_ENTRIES,
                "search_results": MAX_SEARCH_RESULTS,
                "context_files": MAX_CONTEXT_FILES,
                "steps": MAX_STEPS,
                "seconds": MAX_SECONDS,
                "preview_bytes": MAX_PREVIEW_BYTES,
                "hex_bytes": MAX_HEX_BYTES,
                "skills": MAX_SKILLS,
                "swarm_agents": MAX_SWARM_AGENTS,
                "swarm_steps": MAX_SWARM_STEPS,
                "swarm_rounds": MAX_SWARM_ROUNDS,
                "swarm_seconds": MAX_SWARM_SECONDS,
                "consolidate_percent": CONSOLIDATE_PERCENT,
                "parallel_tools": MAX_PARALLEL_TOOLS,
            },
            "features": {
                "lazy_tree": True,
                "file_operations": True,
                "context_rules": True,
                "session_crud": True,
                "multi_project_runs": True,
                "file_preview": True,
                "hex_and_metadata": True,
                "agent_skills": True,
                "skill_crud": True,
                "context_consolidation": True,
                "research_swarm": True,
                "swarm_coding": True,
                "swarm_feedback_loop": True,
                "swarm_shared_memory": True,
                "swarm_ipc": True,
                "parallel_reads": True,
            },
            "security": {
                "workspaces": "agent-workspace/ide/workspaces",
                "snapshots": "every agent edit is restored by a session revert",
                "shell": "disabled unless MARIANA_AGENT_ALLOW_SHELL=1",
                "bind": "loopback only by default",
                "swarm": "members share one run-scoped channel and blackboard; recipients, "
                          "message sizes and quotas are checked; Ask mode is read-only; writes "
                          "are serialized and snapshotted",
            },
            "run_payload": {
                "message": "the task for the coding agent",
                "session_id": "continue this conversation when given",
                "mode": "ask, edit or auto",
                "model": "catalog model id; empty chooses the first chat model",
                "active_file": "workspace-relative file shown in the editor",
                "context_files": "up to 8 other files to include in the system prompt",
                "context": "{include: [paths], exclude: [paths]}; pinned context for this turn",
                "max_steps": f"1..{MAX_STEPS}, default {DEFAULT_MAX_STEPS}",
                "max_seconds": f"30..{MAX_SECONDS}, default {DEFAULT_MAX_SECONDS}",
            },
            "event_types": [
                "run_started", "assistant_start", "assistant_delta", "reasoning_delta",
                "assistant_end", "tool_call", "tool_result", "plan", "file_changed",
                "context_consolidated", "swarm_start", "swarm_plan", "swarm_leader",
                "swarm_round", "swarm_agent_start", "swarm_agent_step", "swarm_agent_delta",
                "swarm_agent_status", "swarm_agent_tool", "swarm_agent_end", "swarm_message",
                "swarm_memory", "swarm_review", "swarm_end", "log", "run_completed",
                "run_failed", "run_cancelled",
            ],
            "specification": "docs/ide.md",
            "spec_url": "/api/ide/spec",
            "tool_schemas": {
                **ABILITY_TYPES,
                "skill.read": {
                    "label": "Read a skill",
                    "description": ("Load one relevant skill's full instructions. Call it "
                                    "automatically when a task matches a skill in the prompt."),
                    "params": [{"key": "name", "label": "Skill id", "type": "text"}],
                },
                "file.inspect": {
                    "label": "Inspect a file",
                    "description": "Metadata, dimensions, hex and a short text preview for any file.",
                    "params": [{"key": "path", "label": "Path", "type": "text"}],
                },
                "agent.swarm": {
                    "label": "Run an agent swarm",
                    "description": "A leader model plans a team that works in parallel, exchanges "
                                   "messages, shares a blackboard, reviews and revises the work, "
                                   "and returns one answer. Members may edit files in Edit and "
                                   "Auto mode; Ask mode is read-only.",
                    "params": [
                        {"key": "tasks", "label": "Tasks", "type": "json"},
                        {"key": "strategy", "label": "Strategy", "type": "select",
                         "options": [["auto", "Auto"], ["map_reduce", "Map reduce"],
                                     ["peer_review", "Peer review"], ["debate", "Debate"],
                                     ["pipeline", "Pipeline"]]},
                        {"key": "rounds", "label": "Feedback rounds", "type": "number"},
                        {"key": "write", "label": "Members may edit", "type": "boolean"},
                        {"key": "max_agents", "label": "Max agents", "type": "number"},
                        {"key": "max_steps", "label": "Max rounds per agent", "type": "number"},
                        {"key": "max_seconds", "label": "Time budget", "type": "number"},
                        {"key": "goal", "label": "Goal", "type": "text"},
                        {"key": "system", "label": "Member instructions", "type": "text"},
                        {"key": "leader_model", "label": "Leader model", "type": "model"},
                        {"key": "ipc", "label": "Agent messages", "type": "boolean"},
                        {"key": "memory", "label": "Shared memory", "type": "boolean"},
                    ],
                },
            },
        }

    def _context_payload(self, value) -> dict:
        """Validate the optional per-turn context selection."""
        if value in (None, ""):
            return {}
        if not isinstance(value, dict):
            raise BadRequestError("context must be an object with include and exclude lists")
        result: dict = {}
        if "include" in value:
            result["include"] = _path_list(value.get("include"), "context include")
        if "exclude" in value:
            result["exclude"] = _path_list(value.get("exclude"), "context exclude")
        return result

    def start_run(self, workspace_id: str, payload: dict) -> dict:
        if self._closed:
            raise ConflictError("Studio is shutting down")
        self.workspace_path(workspace_id)
        message = text(payload, "message", "", 20_000)
        if not message:
            raise BadRequestError("Write a task for the coding agent")
        mode = choice(payload, "mode", "edit", MODES)
        model = text(payload, "model", "", 2000)
        max_steps = number(payload, "max_steps", DEFAULT_MAX_STEPS, 1, MAX_STEPS)
        max_seconds = number(payload, "max_seconds", DEFAULT_MAX_SECONDS, 30, MAX_SECONDS)
        active_file = text(payload, "active_file", "", 1000)
        context_files = payload.get("context_files")
        context_files = [str(item)[:1000] for item in context_files[:8]] \
            if isinstance(context_files, list) else []
        context_given = "context" in payload
        context = self._context_payload(payload.get("context"))
        session_id = text(payload, "session_id", "", 60)
        with self._lock:
            if workspace_id in self._active_workspaces:
                raise ConflictError(
                    "Another coding run is active in this workspace; wait for it to finish"
                )
        if session_id:
            session = self.get_session(session_id, include_messages=True)
            if session.get("workspace_id") != workspace_id:
                raise BadRequestError("That conversation belongs to another workspace")
            if session.get("active_run_id"):
                raise ConflictError("This conversation already has an active run")
        else:
            session = self._new_session(workspace_id, message, mode, model, context=context)
        if context_given:
            session["context"] = context
        try:
            with self._lock:
                if workspace_id in self._active_workspaces:
                    raise ConflictError(
                        "Another coding run is active in this workspace; wait for it to finish"
                    )
                if session["id"] in self._active_sessions:
                    raise ConflictError("This conversation already has an active run")
                self._active_workspaces.add(workspace_id)
                self._active_sessions.add(session["id"])
        except Exception:
            if not session_id:
                # A losing race created a session that has no run; do not leave it in the list.
                try:
                    self._session_path(session["id"]).unlink(missing_ok=True)
                except OSError:
                    pass
            raise
        run = IdeRun(f"ide-run-{uuid.uuid4().hex[:12]}", session["id"], workspace_id, message)
        run.session = session
        try:
            session.setdefault("transcript", []).append(
                {"role": "user", "text": message, "at": _iso()}
            )
            session["title"] = session.get("title") or message.splitlines()[0][:80]
            session["mode"] = mode
            session["model"] = model or session.get("model") or ""
            session["status"] = "running"
            session["last_event_seq"] = 0
            session["active_run_id"] = run.id
            session.setdefault("runs", []).append(run.id)
            self._save_session(session)
            with self._lock:
                self._runs[run.id] = run
            options = {
                "mode": mode, "model": session["model"], "message": message,
                "max_steps": max_steps, "max_seconds": max_seconds,
                "active_file": active_file, "context_files": context_files,
                "context": context,
            }
            thread = threading.Thread(
                target=self._execute, args=(run, workspace_id, session, options),
                name=run.id, daemon=True,
            )
            with self._lock:
                self._threads[run.id] = thread
            thread.start()
        except Exception:
            session["status"] = "failed"
            session["active_run_id"] = None
            try:
                self._save_session(session)
            except Exception:  # noqa: BLE001 - report the original start failure
                pass
            with self._lock:
                self._active_workspaces.discard(workspace_id)
                self._active_sessions.discard(session["id"])
                self._runs.pop(run.id, None)
                self._threads.pop(run.id, None)
            raise
        return {"run": run.snapshot(), "session": self._session_summary(session)}

    def cancel_run(self, run_id: str) -> dict:
        run = self._live_run(run_id)
        if run.status in FINISHED:
            raise ConflictError("That run has already finished")
        run.cancel()
        return run.snapshot()

    def _live_run(self, run_id: str) -> IdeRun:
        with self._lock:
            run = self._runs.get(run_id)
        if run is None:
            saved = self._load_run(run_id)
            if saved and saved.get("status") in FINISHED:
                raise ConflictError("That run has already finished")
            raise NotFoundError("Run not found")
        return run

    def get_run(self, run_id: str, since: int = 0) -> dict:
        with self._lock:
            run = self._runs.get(run_id)
        if run is not None:
            return run.snapshot(since)
        saved = self._load_run(run_id)
        if saved is None:
            raise NotFoundError("Run not found")
        saved["events"] = [event for event in saved.get("events", [])
                           if event.get("seq", 0) > since]
        return saved

    def _load_run(self, run_id: str) -> dict | None:
        if not re.fullmatch(r"ide-run-[a-z0-9]{6,40}", str(run_id or "")):
            return None
        data = read_json(self.runs_dir / f"{run_id}.json")
        if not isinstance(data, dict):
            return None
        events = data.get("events", [])
        if not isinstance(events, list) or not all(
                isinstance(item, dict) and isinstance(item.get("seq"), int) for item in events):
            return None
        return data

    def stream_run(self, run_id: str, since: int = 0):
        with self._lock:
            run = self._runs.get(run_id)
        if run is None:
            saved = self._load_run(run_id)
            if saved is None:
                raise NotFoundError("Run not found")

            def replay():
                for event in saved.get("events", []):
                    if event.get("seq", 0) > since:
                        yield event

            return replay()

        def events():
            last = max(0, since)
            while True:
                pending, finished = run.wait_events(last)
                for event in pending:
                    last = event["seq"]
                    yield event
                if finished:
                    remaining, still = run.wait_events(last, timeout=0)
                    for event in remaining:
                        last = event["seq"]
                        yield event
                    if still and not remaining:
                        return
                if not pending:
                    yield None

        return events()

    def _execute(self, run: IdeRun, workspace_id: str, session: dict, options: dict) -> None:
        final_text = ""
        final_recorded = False
        status = "completed"
        error = None
        timer = None
        usage = {"prompt_tokens": 0, "completion_tokens": 0}
        try:
            workspace = self.workspace_path(workspace_id)
            model = self._pick_model(options.get("model"))
            run.model = model
            tools = self._tools_for_mode(options["mode"], workspace_id)
            active = self._active_file(workspace_id, options.get("active_file"))
            context = options.get("context") or session.get("context") or {}
            excluded = self._excluded_paths(workspace_id, context)
            include = self._context_include(workspace_id, context)
            pieces, prompt, prompt_truncated = self._prompt_pieces(
                workspace_id, options["mode"], tools, workspace, active, context,
                options["message"], options.get("context_files") or [], session,
            )
            history = self._history(session)[-(MODEL_MESSAGE_WINDOW - 1):]
            messages = [{"role": "system", "content": prompt}]
            messages.extend(history)
            messages.append({"role": "user", "content": options["message"]})
            budget = self._budget_from_parts(
                model, pieces, prompt, prompt_truncated, history, options["message"],
            )
            messages, consolidated = self._consolidate_for_budget(
                messages, budget, session, "auto", max_messages=MESSAGE_RECAP_TRIGGER,
            )
            if consolidated:
                history = [item for item in messages[1:-1]
                           if item.get("role") in {"user", "assistant"}]
                budget = self._budget_from_parts(
                    model, pieces, prompt, prompt_truncated, history, options["message"],
                )
            budget["last_usage"] = session.get("usage") if isinstance(session.get("usage"), dict) else {}
            run.event("run_started", session_id=session["id"], workspace_id=workspace_id,
                      model=model, mode=options["mode"], tools=[tool["id"] for tool in tools],
                      context={"include": include, "exclude": excluded}, budget=budget,
                      message="Coding agent started")
            if consolidated:
                run.event(
                    "context_consolidated", trigger="auto", iteration=0,
                    **consolidated,
                    message=(f"Consolidated older turns to free about "
                             f"{max(0, consolidated['before_tokens'] - consolidated['after_tokens']):,} "
                             f"{consolidated['unit']} before the run."),
                )
            started = time.time()
            limit = options["max_steps"]
            repeated: Counter = Counter()
            recent_calls: deque[tuple[str, bool, str]] = deque(maxlen=8)
            format_repairs = 0
            network_retries = 0
            extensions = 0
            iteration = 0
            last_checkpoint = 0.0
            action_help = (
                "Reply with only one JSON object: "
                '{"tool": ...}, {"calls": [...]}, {"plan": [...]} or {"final": "..."}. '
                "Do not wrap it in prose or a code fence."
            )

            def checkpoint(force: bool = False) -> None:
                """Persist the live transcript and model history during a long run."""
                nonlocal last_checkpoint
                now = time.time()
                if not force and now - last_checkpoint < CHECKPOINT_SECONDS:
                    return
                last_checkpoint = now
                if messages is not None:
                    session["messages"] = self._trim_history(
                        [item for item in messages if item.get("role") != "system"]
                    )
                session["usage"] = dict(usage)
                session["last_event_seq"] = run.snapshot(0, include_events=False)["event_count"]
                session["status"] = "running"
                session["active_run_id"] = run.id
                try:
                    self._save_session(session)
                except Exception:  # noqa: BLE001 - a checkpoint must not end the run
                    pass

            def consolidate_history(step: int) -> bool:
                nonlocal messages
                if messages is None:
                    return False
                messages, info = self._consolidate_for_budget(
                    messages, budget, session, "auto", max_messages=MESSAGE_RECAP_TRIGGER,
                )
                if not info:
                    return False
                run.event(
                    "context_consolidated", trigger="auto", iteration=step, **info,
                    message=(f"Consolidated older turns to free about "
                             f"{max(0, info['before_tokens'] - info['after_tokens']):,} "
                             f"{info['unit']} during the run."),
                )
                return True

            def extend_if_needed() -> None:
                nonlocal limit, extensions
                if iteration < limit or extensions >= MAX_STEP_EXTENSIONS:
                    return
                if limit >= MAX_STEPS or not self._should_extend(recent_calls):
                    return
                added = min(MAX_STEPS, limit + STEP_EXTENSION) - limit
                limit += added
                extensions += 1
                run.event(
                    "log", level="info", iteration=iteration,
                    message=(f"Extended the step budget by {added} to {limit}; "
                             "the run is still making progress."),
                )

            timer = threading.Timer(options["max_seconds"], self._expire_run,
                                    args=(run, options["max_seconds"]))
            timer.daemon = True
            timer.start()
            while iteration < limit:
                iteration += 1
                if run.cancelled.is_set():
                    raise _Cancelled()
                if time.time() - started > options["max_seconds"]:
                    run.exhausted = True
                    run.timed_out = True
                    final_text = (
                        f"Stopped after {iteration - 1} steps because the time budget "
                        f"({options['max_seconds']} s) ran out."
                    )
                    break
                reply, round_usage, round_meta = None, None, None
                try:
                    reply, round_usage, round_meta = self._stream_reply(
                        run, model, messages, iteration, tools,
                    )
                except (UnavailableError, UpstreamError, TimeoutError, OSError) as failure:
                    status_code = int(getattr(failure, "detail", {}).get("status") or 0)
                    retryable = not (400 <= status_code < 500
                                     and status_code not in {408, 429})
                    if (retryable and network_retries < MAX_NETWORK_RETRIES
                            and not run.cancelled.is_set()):
                        network_retries += 1
                        delay = float(network_retries)
                        run.event(
                            "log", level="warning", iteration=iteration,
                            message=(f"Model endpoint failed ({failure}); retrying in "
                                     f"{delay:.0f}s"),
                        )
                        checkpoint(force=True)
                        time.sleep(delay)
                        iteration -= 1  # a transport retry does not consume a step
                        continue
                    raise
                for key in usage:
                    usage[key] += int(round_usage.get(key) or 0)
                network_retries = 0
                reasoning = str(round_meta.get("reasoning") or "").strip()
                if reasoning:
                    # Keep the thinking next to the tool calls it produced, the way
                    # Zed shows a collapsed thought row before each action.
                    self._append_transcript(session, {"role": "thinking", "text": reasoning})
                if not reply.strip():
                    raise ValueError(
                        "The model returned an empty reply. Choose another model in the IDE toolbar."
                    )
                action, calls = parse_model_action(reply)
                malformed = action is None and _looks_like_action(reply)
                unknown = (action is not None and not calls
                           and "final" not in action and "plan" not in action)
                if malformed or unknown:
                    if format_repairs < MAX_FORMAT_REPAIRS:
                        format_repairs += 1
                        messages.append({"role": "assistant", "content": reply})
                        messages.append({"role": "user", "content":
                                         f"That reply was not a valid action. {action_help}"})
                        run.event(
                            "log", level="warning", iteration=iteration,
                            message=("The model's action was malformed; asking for a "
                                     "corrected JSON reply"),
                        )
                        consolidate_history(iteration)
                        checkpoint(force=True)
                        extend_if_needed()
                        continue
                    raise ValueError(
                        "The model kept replying with malformed actions. Choose a model "
                        "with reliable JSON tool calling or start a new conversation."
                    )
                format_repairs = 0
                if action is None:
                    final_text = reply.strip()[:20_000]
                    self._append_transcript(session, {"role": "assistant", "text": final_text})
                    final_recorded = True
                    break
                if "plan" in action:
                    plan = normalize_plan(action.get("plan"))
                    session["plan"] = plan
                    run.plan = plan
                    run.plan_set = True
                    run.event("plan", plan=plan, message=f"Plan updated ({len(plan)} steps)")
                    self._append_transcript(session, {"role": "plan", "todos": plan})
                if not calls:
                    if "final" in action:
                        final_text = self._final_text(action, reply)
                        if not final_text.strip():
                            if format_repairs < MAX_FORMAT_REPAIRS:
                                format_repairs += 1
                                messages.append({"role": "assistant", "content": reply})
                                messages.append({"role": "user", "content": (
                                    "Your final answer was empty. Reply with "
                                    '{"final": "<summary>"} describing what was done, '
                                    "or continue with a tool call."
                                )})
                                run.event(
                                    "log", level="warning", iteration=iteration,
                                    message=("The model finished with an empty summary; "
                                             "asking for one"),
                                )
                                checkpoint(force=True)
                                extend_if_needed()
                                continue
                            final_text = "The task finished without a summary."
                        self._append_transcript(
                            session, {"role": "assistant", "text": final_text})
                        final_recorded = True
                        break
                    if "plan" in action:
                        messages.append({"role": "assistant", "content": reply})
                        messages.append({"role": "user", "content": (
                            "Plan recorded. Continue with a tool call, update the plan, "
                            "or reply with the final answer."
                        )})
                        consolidate_history(iteration)
                        checkpoint(force=True)
                        extend_if_needed()
                        continue
                    final_text = self._final_text(action, reply)
                    self._append_transcript(session, {"role": "assistant", "text": final_text})
                    final_recorded = True
                    break
                messages.append({"role": "assistant", "content": reply})
                observations = self._run_tool_batch(
                    run, workspace_id, session, tools, calls, iteration, repeated, excluded,
                )
                for result in observations:
                    recent_calls.append((str(result.get("signature") or ""),
                                         bool(result.get("ok")),
                                         str(result.get("tool") or "")))
                messages.append({"role": "user", "content":
                                 self._observation_message(observations, session)})
                consolidated = consolidate_history(iteration)
                wrote = any(result.get("ok") and result.get("tool") in _WRITE_TOOLS
                            for result in observations)
                checkpoint(force=wrote or consolidated)
                extend_if_needed()
            else:
                run.exhausted = True
                final_text = (
                    f"Stopped after {limit} steps without a final answer. "
                    "Ask for the next step to continue."
                )
        except _Cancelled:
            if run.timed_out:
                status = "completed"
                run.exhausted = True
                final_text = (
                    f"Stopped after {run.elapsed():.0f}s because the time budget ran out. "
                    "Ask for the next step to continue."
                )
            else:
                status, error = "cancelled", "Run cancelled"
        except Exception as failure:  # noqa: BLE001 - the run records its own failure
            status, error = "failed", str(failure) or type(failure).__name__
            self._append_transcript(session, {"role": "error", "text": error})
        finally:
            if timer is not None:
                timer.cancel()
            if final_text and not final_recorded and status == "completed":
                self._append_transcript(session, {"role": "assistant", "text": final_text})
            if (final_recorded and status == "completed"
                    and not run.exhausted and not run.timed_out):
                self._finish_plan(run, session)
            if "messages" in locals():
                if final_text and status == "completed":
                    messages.append({"role": "assistant", "content": final_text[:MAX_MESSAGE_CHARS]})
                session["messages"] = self._trim_history(
                    [item for item in messages if item.get("role") != "system"]
                )
            session["plan"] = run.plan if run.plan_set else (session.get("plan") or [])
            session["usage"] = dict(usage)
            session["status"] = status
            session["active_run_id"] = None
            try:
                self._save_session(session)
            except Exception:  # noqa: BLE001 - persistence must not leave a run "running"
                pass
            run.usage = usage
            try:
                run.finish(status, output=final_text or None, error=error,
                           usage=usage, exhausted=run.exhausted, timed_out=run.timed_out,
                           session_id=session["id"], workspace_id=workspace_id,
                           duration=run.elapsed())
            finally:
                with self._lock:
                    self._active_sessions.discard(session["id"])
                    self._active_workspaces.discard(workspace_id)
                    self._threads.pop(run.id, None)
                    if len(self._runs) > 24:
                        finished = [item for item in self._runs.values()
                                    if item.status in FINISHED]
                        for item in finished[:len(self._runs) - 24]:
                            self._runs.pop(item.id, None)
            self._save_run(run)
            run.session = None

    def _expire_run(self, run: IdeRun, seconds: float) -> None:
        """A timer ends a run that outlived its wall-clock budget."""
        with run.condition:
            if run.status in FINISHED:
                return
            run.timed_out = True
            run.cancelled.set()
        run.event("log", level="warning",
                  message=f"Time budget of {seconds:.0f}s exhausted")

    def _stream_reply(self, run: IdeRun, model: str, messages: list[dict],
                      iteration: int, tools: list[dict], *,
                      max_tokens: int | None = None, allow_retry: bool = True,
                      effort: str | None = None) -> tuple[str, dict, dict]:
        """One streamed model turn, retrying an empty or truncated reply once.

        A reasoning endpoint can spend the whole output budget on hidden thinking
        and return no visible content; ``finish_reason`` is then ``length`` and
        the gateway reports its empty-reply placeholder.  Treating that as the
        final answer silently ended the run, so the turn is retried with double
        the output budget and a lower reasoning effort before failing loudly.
        """
        limit = int(max_tokens or self._output_limit(model))
        config = {
            "model": model,
            "messages": self._model_messages(messages),
            "temperature": 0.2,
            "top_p": 0.95,
            "max_tokens": limit,
        }
        if effort:
            config["reasoning_effort"] = effort
        if tools:
            # Native function calling where the provider supports it; the prompted
            # JSON path and DSML parsing remain the fallback.
            config["tools"] = tools
        run.event("assistant_start", iteration=iteration)
        parts: list[str] = []
        thoughts: list[str] = []
        result: dict = {}
        pending: dict[str, str] = {}
        last_emit = 0.0

        def flush_pending():
            nonlocal last_emit
            for kind, text in pending.items():
                run.event("assistant_delta" if kind == "delta" else "reasoning_delta",
                          iteration=iteration, text=text[:4000])
            pending.clear()
            last_emit = time.time()

        stream = getattr(self.gateway, "stream", None)
        request_cancel = _RequestCancel(run.cancelled)
        try:
            if stream is not None:
                for event in stream(config, request_cancel):
                    if run.cancelled.is_set():
                        raise _Cancelled()
                    kind = event.get("type")
                    if kind in {"delta", "reasoning"}:
                        text = str(event.get("text") or "")
                        (parts if kind == "delta" else thoughts).append(text)
                        pending[kind] = pending.get(kind, "") + text
                        if time.time() - last_emit >= 0.12:
                            flush_pending()
                    elif kind == "status":
                        run.event("log", level="info", iteration=iteration,
                                  message=str(event.get("message") or event.get("status") or "Working"))
                    elif kind == "done":
                        result = event
            else:
                result = self.gateway.chat(config)
                if result.get("reply"):
                    parts.append(str(result["reply"]))
            if pending:
                flush_pending()
        except Exception as failure:
            run.event("assistant_end", iteration=iteration, text="", error=str(failure),
                      empty=True, truncated=False, retrying=False)
            raise
        reply = str(result.get("reply") or "".join(parts))
        usage = dict(result.get("usage") or {})
        finish = str(result.get("finish_reason") or "")
        completion = int(usage.get("completion_tokens") or 0)
        empty = not reply.strip() or reply.strip() in EMPTY_REPLIES
        truncated = finish == "length" or (not empty and completion >= limit)
        reasoning = "".join(thoughts)[-MAX_REASONING_CHARS:]
        retrying = bool((empty or truncated) and allow_retry and not result.get("error"))
        run.event("assistant_end", iteration=iteration, text=reply[:2000],
                  chars=len(reply), usage=usage, elapsed=float(result.get("elapsed") or 0.0),
                  finish_reason=finish, empty=empty, truncated=truncated,
                  retrying=retrying, reasoning=reasoning)
        if retrying:
            retry_limit = min(MAX_OUTPUT_TOKENS, max(limit * 2, limit + 256))
            reason = ("The model returned no visible text" if empty
                      else "The model hit its output limit")
            run.event(
                "log", level="warning", iteration=iteration,
                message=(
                    f"{reason}"
                    f"{f' (finish: {finish})' if finish else ''}; retrying this step with "
                    f"{retry_limit:,} output tokens and lower reasoning effort"
                ),
            )
            retry_reply, retry_usage, retry_meta = self._stream_reply(
                run, model, messages, iteration, tools,
                max_tokens=retry_limit, allow_retry=False, effort="low",
            )
            for key in usage:
                usage[key] = int(usage.get(key) or 0) + int(retry_usage.get(key) or 0)
            return retry_reply, usage, retry_meta
        if empty:
            if result.get("error"):
                raise ValueError(str(result["error"]))
            if finish == "length":
                raise ValueError(
                    "The model used its whole output budget before writing a visible "
                    "answer. Lower its reasoning effort, raise Maximum output tokens on "
                    "the saved endpoint, or choose a model that returns visible text."
                )
            raise ValueError(
                "The model returned an empty reply. Choose another model in the IDE toolbar."
            )
        return reply, usage, {
            "reasoning": reasoning, "finish_reason": finish, "truncated": truncated,
        }

    def _swarm_tools(self, workspace_id: str, write: bool = False,
                     mode: str = "ask") -> list[dict]:
        """Tools a swarm member may use; writes follow the parent run's mode."""
        source = "auto" if write and mode == "auto" else "edit" if write else "ask"
        tools = []
        for tool in self._tools_for_mode(source, workspace_id):
            if tool["id"] == "agent.swarm":
                continue
            if tool["type"] == "skill.read":
                description = ("Load one relevant skill's full instructions by id from the "
                               "workspace or the user's skills folder. Call it automatically "
                               "when a task matches a skill description.")
                params = [{"key": "name", "label": "Skill id", "type": "text"}]
            elif tool["type"] == "file.inspect":
                description = ("Metadata, MIME type, dimensions, hex and a short text preview "
                               "for any file, including binary.")
                params = [{"key": "path", "label": "Path", "type": "text"}]
            else:
                meta = ABILITY_TYPES.get(tool["type"], {})
                description = meta.get("description", "")
                params = meta.get("params", [])
            tools.append({**tool, "description": description, "params": params})
        return tools

    def _swarm_briefing(self, workspace_id: str, excluded: list[str] | None = None) -> str:
        """The shared project context every swarm member receives."""
        workspace = self.workspace_path(workspace_id)
        name = self._manifest(workspace_id).get("name") or workspace_id
        lines = [
            f"Workspace: {name}. Every tool path is relative to the workspace root.",
            "Project map:",
            self._map(workspace, limit=80, exclude=excluded or []),
        ]
        skills = self.list_skills(workspace_id)
        if skills:
            lines.append(
                "All skills below are available automatically. Call skill.read for each "
                "one that fits the task before relying on it:")
            lines.extend(self._skill_list_lines(skills))
        if excluded:
            lines.append("Hidden by the user's context rules: " + ", ".join(excluded[:16]))
        return "\n".join(lines)

    def _swarm_specs(self, arguments: dict) -> list[dict]:
        """Normalise the optional member list supplied by the parent model."""
        raw = arguments.get("tasks", arguments.get("agents"))
        if raw in (None, ""):
            return []
        if isinstance(raw, str):
            raw = [raw]
        elif isinstance(raw, dict):
            raw = [raw]
        if not isinstance(raw, list):
            raise ValueError(
                'agent.swarm needs a tasks list, for example {"tasks": '
                '[{"task": "Find the entry point"}, {"task": "List the tests"}]}'
            )
        specs = []
        seen = set()
        for index, item in enumerate(raw[:MAX_SWARM_AGENTS]):
            if isinstance(item, str):
                task, label = item, ""
            elif isinstance(item, dict):
                task = item.get("task") or item.get("goal") or item.get("prompt") or ""
                label = item.get("label") or item.get("name") or ""
            else:
                continue
            task = str(task).strip()[:MAX_SWARM_TASK_CHARS]
            if not task:
                continue
            base_source = (item.get("id") if isinstance(item, dict) else "") or label or f"agent-{index + 1}"
            base = slugify(base_source, f"agent-{index + 1}")[:48]
            member_id, counter = base, 2
            while member_id in seen:
                member_id = f"{base}-{counter}"
                counter += 1
            seen.add(member_id)
            spec = {
                "id": member_id,
                "label": " ".join(str(label).split())[:80] or f"Agent {index + 1}",
                "task": task,
            }
            if isinstance(item, dict):
                if item.get("role"):
                    spec["role"] = str(item["role"])[:40]
                if item.get("instruction"):
                    spec["instruction"] = str(item["instruction"])[:600]
                if isinstance(item.get("tools"), list):
                    spec["tools"] = [str(tool) for tool in item["tools"]][:24]
                if isinstance(item.get("writes"), list):
                    spec["writes"] = [str(path) for path in item["writes"]][:40]
                if "read_only" in item:
                    spec["read_only"] = _boolean(item["read_only"], True)
                if item.get("context_chars") not in (None, ""):
                    spec["context_chars"] = item["context_chars"]
            specs.append(spec)
        return specs

    def _swarm_executor(self, run: IdeRun, workspace_id: str, session: dict,
                        tools: list[dict], excluded: list[str] | None):
        """Route member tool calls through the IDE sandbox, snapshots and diagnostics."""
        allowed = {tool["id"]: tool for tool in tools}

        def execute(member: dict, tool_id: str, arguments: dict):
            definition = allowed.get(tool_id)
            if definition is None:
                return {
                    "error": (f"Tool {tool_id!r} is not available to this swarm. Use one of: "
                              + ", ".join(sorted(allowed))),
                }, False
            observation, ok, change, _ = self._execute_defined_tool(
                run, workspace_id, session, definition, tool_id, arguments, excluded)
            if change and isinstance(observation, dict):
                observation = {
                    **observation,
                    "change": {key: change[key] for key in
                               ("path", "added", "removed", "diff") if key in change},
                }
            return observation, ok

        return execute

    def _swarm_event(self, run: IdeRun, coordinator: SwarmCoordinator,
                     kind: str, **fields) -> None:
        """Forward swarm progress and keep the run's swarm snapshot current."""
        if kind != "swarm_agent_delta":
            now = time.monotonic()
            if snapshot_due(kind, run.swarm_snapshot_at, now):
                try:
                    run.set_swarm(coordinator.snapshot())
                    run.swarm_snapshot_at = now
                except Exception:  # noqa: BLE001 - a progress snapshot must not stop a swarm
                    pass
        run.event(kind, **fields)

    def _run_swarm(self, run: IdeRun, workspace_id: str, session: dict, arguments,
                   call_id: str, excluded: list[str] | None = None) -> dict:
        """Run a leader-driven swarm: research, coding, review and synthesis."""
        arguments = arguments if isinstance(arguments, dict) else {}
        mode = str(session.get("mode") or "edit")
        allow_write = mode in {"edit", "auto"} and _boolean(arguments.get("write"), True)
        if mode == "ask":
            allow_write = False
        specs = self._swarm_specs(arguments)
        try:
            max_agents = int(arguments.get("max_agents", 4))
        except (TypeError, ValueError):
            max_agents = 4
        try:
            rounds = int(arguments.get("rounds", 1))
        except (TypeError, ValueError):
            rounds = 1
        try:
            max_steps = int(arguments.get("max_steps", 6))
        except (TypeError, ValueError):
            max_steps = 6
        try:
            max_seconds = int(arguments.get("max_seconds", 300))
        except (TypeError, ValueError):
            max_seconds = 300
        max_agents = max(1, min(MAX_SWARM_AGENTS, max_agents))
        rounds = max(0, min(MAX_SWARM_ROUNDS, rounds))
        max_steps = max(1, min(MAX_SWARM_STEPS, max_steps))
        max_seconds = max(30, min(MAX_SWARM_SECONDS, max_seconds))
        allow_ipc = _boolean(arguments.get("ipc"), True)
        shared_memory = _boolean(arguments.get("memory"), True)
        goal = str(arguments.get("goal") or run.message or "").strip()
        if not goal:
            raise ValueError("agent.swarm needs a goal or a parent task")
        model = str(arguments.get("model") or run.model or "").strip()
        model = self._pick_model(model)
        leader_model = str(arguments.get("leader_model") or "").strip()
        if leader_model:
            leader_model = self._pick_model(leader_model)
        tools = self._swarm_tools(workspace_id, allow_write, mode)
        limits = SwarmLimits(
            max_agents=max_agents, max_steps=max_steps, max_rounds=MAX_SWARM_ROUNDS,
            max_seconds=float(max_seconds), max_messages=240,
            max_message_chars=MAX_SWARM_OUTPUT_CHARS,
            max_memory_entries=160, max_memory_chars=12_000,
            max_observation_chars=MAX_OBSERVATION_CHARS,
            max_prompt_chars=MAX_PROMPT_CHARS,
            max_output_chars=MAX_SWARM_OUTPUT_CHARS,
            max_reviews=min(8, max(2, max_agents * 2)),
        )
        coordinator = SwarmCoordinator(
            gateway=self.gateway, swarm_id=call_id, goal=goal, model=model,
            leader_model=leader_model,
            system=str(arguments.get("system") or ""),
            tools=tools,
            executor=self._swarm_executor(run, workspace_id, session, tools, excluded),
            limits=limits, allow_write=allow_write, allow_ipc=allow_ipc,
            allow_os_access=self.os_access,
            shared_memory=shared_memory, dry_run=False,
            on_event=lambda kind, **fields: self._swarm_event(run, coordinator, kind, **fields),
            cancelled=lambda: run.cancelled.is_set(),
            briefing=self._swarm_briefing(workspace_id, excluded),
        )
        try:
            result = coordinator.run(
                agents=specs,
                strategy=str(arguments.get("strategy") or "auto"),
                rounds=rounds,
            )
        finally:
            run.set_swarm(coordinator.snapshot())
        if run.cancelled.is_set():
            raise _Cancelled()
        if not result.get("ok"):
            raise ValueError(result.get("error") or "The swarm failed")
        agents = []
        for member in result.get("agents", []):
            agents.append({
                "agent": member.get("id"), "label": member.get("label"),
                "role": member.get("role"), "task": member.get("task"),
                "ok": member.get("status") == "done",
                "status": member.get("status"), "output": (member.get("output") or "")[:MAX_SWARM_OUTPUT_CHARS],
                "error": member.get("error"), "score": member.get("score"),
                "version": member.get("version"), "steps": member.get("steps"),
                "usage": member.get("usage"),
            })
        reviews = [
            {"reviewer": item.get("reviewer"), "target": item.get("target"),
             "score": item.get("score"), "summary": item.get("summary"),
             "issues": item.get("issues", [])[:4]}
            for item in result.get("reviews", [])[-12:]
        ]
        return {
            "ok": True,
            "swarm": result.get("swarm") or call_id,
            "strategy": result.get("strategy"),
            "round": result.get("round"),
            "rounds": result.get("requested_rounds", result.get("round")),
            "leader": result.get("leader"),
            "model": result.get("model"),
            "summary": result.get("summary") or "",
            "output": result.get("final") or "",
            "text": result.get("final") or "",
            "agents": agents,
            "completed": sum(1 for item in agents if item["ok"]),
            "failed": sum(1 for item in agents if not item["ok"]),
            "reviews": reviews,
            "messages": result.get("messages", 0),
            "blackboard": result.get("blackboard_count", len(result.get("blackboard", []))),
            "context": result.get("context", {}),
            "context_items": result.get("context_items", 0),
            "changed_files": result.get("changed_files", []),
            "usage": result.get("usage", {}),
            "elapsed": result.get("elapsed", 0.0),
            "tools": [tool["id"] for tool in tools],
        }

    @staticmethod
    def _tool_signature(tool_type: str, arguments: dict) -> str:
        return json.dumps({"tool": tool_type, "input": arguments}, sort_keys=True,
                          ensure_ascii=False, default=str)

    def _invoke_tool(self, run: IdeRun, workspace_id: str, session: dict,
                     tools: list[dict], tool_type: str, arguments: dict,
                     definition: dict | None, call_id: str,
                     excluded: list[str] | None) -> tuple[dict, bool, dict | None, dict]:
        """Run one call; safe for read-only tools to execute in a worker thread."""
        change = None
        if tool_type == "agent.swarm":
            try:
                observation = self._run_swarm(
                    run, workspace_id, session, arguments, call_id, excluded)
                ok = True
            except _Cancelled:
                raise
            except Exception as failure:  # noqa: BLE001 - a swarm bug must not end the run
                observation = {"error": str(failure) or type(failure).__name__}
                ok = False
        elif definition is None:
            suggestion = nearest_tool_id(tool_type, [tool["id"] for tool in tools])
            hint = f" Did you mean {suggestion!r}?" if suggestion else ""
            observation = {
                "error": (
                    f"Tool {tool_type!r} is not enabled in {MODE_INFO[session.get('mode', 'edit')]['label']} "
                    f"mode.{hint} Use one of: " + ", ".join(tool["id"] for tool in tools)
                )
            }
            ok = False
        else:
            observation, ok, change, arguments = self._execute_defined_tool(
                run, workspace_id, session, definition, tool_type, arguments, excluded)
        return observation, ok, change, arguments

    def _complete_tool(self, run: IdeRun, session: dict, tool_type: str, arguments: dict,
                       call_id: str, iteration: int, observation, ok: bool,
                       change: dict | None, noted: bool, signature: str,
                       attempts: int = 1) -> dict:
        """Publish one finished call, preserving call order and the transcript."""
        if noted:
            observation = dict(observation) if isinstance(observation, dict) else {"result": observation}
            observation["hint"] = (
                "You repeated the same tool call three times. Read the result, "
                "change your approach, or finish with the final answer." if ok else
                "You repeated the same failing tool call three times. Change the input or "
                "use another tool instead of retrying it unchanged."
            )
        bound = self._bounded_observation(observation, MAX_OBSERVATION_CHARS)
        step = {"role": "tool", "tool": tool_type, "input": json_safe(arguments, 4000),
                "output": bound, "ok": ok, "call": call_id, "at": _iso()}
        if attempts > 1:
            step["attempts"] = attempts
        if change:
            step["change"] = change
        if tool_type == "agent.swarm" and isinstance(observation, dict):
            step["swarmAgents"] = observation.get("agents", [])
            step["swarmReviews"] = observation.get("reviews", [])
            step["swarmChanged"] = observation.get("changed_files", [])
            step["swarmMeta"] = {
                "strategy": observation.get("strategy"),
                "rounds": observation.get("rounds", observation.get("round")),
                "round": observation.get("round", observation.get("rounds")),
                "phase": "synthesis", "finished": True,
                "final": observation.get("output") or observation.get("text") or "",
                "summary": observation.get("summary") or "",
                "messageCount": observation.get("messages", 0),
                "reviewCount": len(observation.get("reviews", [])),
                "memoryCount": observation.get("blackboard", 0),
            }
        self._append_transcript(session, step)
        run.event("tool_result", call=call_id, iteration=iteration, tool=tool_type, ok=ok,
                  attempts=attempts, output=bound,
                  change=change, message=f"{tool_type} {'returned' if ok else 'failed'}")
        if tool_type == "agent.swarm" and isinstance(observation, dict):
            parent_result = self._swarm_parent_observation(observation)
        else:
            parent_result = observation
        return {"tool": tool_type, "ok": ok, "signature": signature,
                "result": parent_result}

    def _run_tool(self, run: IdeRun, workspace_id: str, session: dict, tools: list[dict],
                  call: dict, iteration: int, index: int,
                  repeated: Counter, excluded: list[str] | None = None) -> dict:
        tool_type = call["tool"]
        arguments = call["input"] if isinstance(call["input"], dict) else {"input": call["input"]}
        call_id = f"{iteration}-{index}"
        definition = next((tool for tool in tools if tool["id"] == tool_type), None)
        signature = self._tool_signature(tool_type, arguments)
        repeated[signature] += 1
        noted = repeated[signature] >= 3
        run.event("tool_call", call=call_id, iteration=iteration, tool=tool_type,
                  input=json_safe(arguments, 4000), message=f"Using {tool_type}")
        observation, ok, change, arguments, attempts = self._invoke_with_retry(
            run, workspace_id, session, tools, tool_type, arguments, definition,
            call_id, excluded, iteration)
        return self._complete_tool(run, session, tool_type, arguments, call_id, iteration,
                                   observation, ok, change, noted, signature, attempts)

    def _invoke_with_retry(self, run: IdeRun, workspace_id: str, session: dict,
                           tools: list[dict], tool_type: str, arguments: dict,
                           definition: dict | None, call_id: str,
                           excluded: list[str] | None, iteration: int = 0) -> tuple:
        """Invoke a tool, retrying a read-only call once after a temporary failure."""
        attempts = 0
        while True:
            if run.cancelled.is_set():
                raise _Cancelled()
            attempts += 1
            observation, ok, change, arguments = self._invoke_tool(
                run, workspace_id, session, tools, tool_type, arguments, definition,
                call_id, excluded)
            error = observation.get("error") if isinstance(observation, dict) else observation
            retryable = (tool_type in _PARALLEL_TOOLS and attempts < TOOL_ATTEMPTS
                         and retryable_tool_failure(error))
            if ok or not retryable:
                break
            run.event("log", call=call_id, iteration=iteration,
                      tool=tool_type, attempt=attempts + 1,
                      message=f"Retrying {tool_type} after a temporary failure")
        if not ok:
            hint = tool_failure_hint(tool_type, error)
            if hint:
                observation = ({**observation, "hint": hint}
                               if isinstance(observation, dict)
                               else {"error": str(observation or "failed"), "hint": hint})
        return observation, ok, change, arguments, attempts

    def _run_tool_batch(self, run: IdeRun, workspace_id: str, session: dict,
                        tools: list[dict], calls: list[dict], iteration: int,
                        repeated: Counter, excluded: list[str] | None = None) -> list[dict]:
        """Run independent read-only calls together; everything else in order."""
        definitions = {tool["id"]: tool for tool in tools}
        parallel = (len(calls) > 1 and not run.cancelled.is_set()
                    and all(call.get("tool") in _PARALLEL_TOOLS
                            and call.get("tool") in definitions for call in calls))
        if not parallel:
            results = []
            for index, call in enumerate(calls):
                if run.cancelled.is_set():
                    raise _Cancelled()
                results.append(self._run_tool(
                    run, workspace_id, session, tools, call, iteration, index, repeated, excluded))
            return results

        prepared = []
        for index, call in enumerate(calls):
            if run.cancelled.is_set():
                raise _Cancelled()
            tool_type = call["tool"]
            arguments = (call["input"] if isinstance(call["input"], dict)
                         else {"input": call["input"]})
            call_id = f"{iteration}-{index}"
            signature = self._tool_signature(tool_type, arguments)
            repeated[signature] += 1
            noted = repeated[signature] >= 3
            run.event("tool_call", call=call_id, iteration=iteration, tool=tool_type,
                      input=json_safe(arguments, 4000), message=f"Using {tool_type}")
            prepared.append((tool_type, arguments, call_id, signature, noted))

        workers = min(MAX_PARALLEL_TOOLS, len(prepared))
        slots = threading.BoundedSemaphore(workers)
        outcomes: list[tuple | None] = [None] * len(prepared)

        def work(position: int, tool_type: str, arguments: dict, call_id: str,
                 noted: bool) -> None:
            with slots:
                try:
                    observation, ok, change, arguments, attempts = self._invoke_with_retry(
                        run, workspace_id, session, tools, tool_type, arguments,
                        definitions[tool_type], call_id, excluded, iteration)
                    outcomes[position] = (observation, ok, change, arguments, attempts)
                except Exception as failure:  # noqa: BLE001 - report it like a tool failure
                    outcomes[position] = (
                        {"error": str(failure) or type(failure).__name__}, False, None,
                        arguments, 1)

        threads = []
        for position, (tool_type, arguments, call_id, _signature, noted) in enumerate(prepared):
            thread = threading.Thread(
                target=work, args=(position, tool_type, arguments, call_id, noted),
                name=f"ide-tool-{position}", daemon=True,
            )
            threads.append(thread)
            thread.start()
        while any(thread.is_alive() for thread in threads):
            if run.cancelled.is_set():
                # The workers are daemons.  A read stuck on a slow mount must not
                # keep a cancelled run alive until the filesystem responds.
                raise _Cancelled()
            for thread in threads:
                thread.join(timeout=0.1)
        if run.cancelled.is_set():
            raise _Cancelled()
        results = []
        for (tool_type, original_arguments, call_id, signature, noted), outcome in zip(prepared, outcomes):
            observation, ok, change, arguments, attempts = outcome or (
                {"error": "The tool did not return a result"}, False, None,
                original_arguments, 1)
            results.append(self._complete_tool(
                run, session, tool_type, arguments, call_id, iteration,
                observation, ok, change, noted, signature, attempts))
        return results

    @staticmethod
    def _swarm_parent_observation(observation: dict) -> dict:
        """The bounded view the parent model needs; full state stays in the run."""
        agents = []
        for agent in observation.get("agents", [])[:6]:
            if not isinstance(agent, dict):
                continue
            item = {
                "agent": agent.get("agent") or agent.get("id"),
                "label": str(agent.get("label") or "")[:40],
                "role": str(agent.get("role") or "")[:20],
                "ok": agent.get("ok"), "score": agent.get("score"),
                "version": agent.get("version"),
            }
            if agent.get("error"):
                item["error"] = str(agent["error"])[:60]
            agents.append(item)
        reviews = []
        for review in observation.get("reviews", [])[:2]:
            if not isinstance(review, dict):
                continue
            reviews.append({
                "reviewer": review.get("reviewer"), "target": review.get("target"),
                "score": review.get("score"),
            })
        return {
            "ok": observation.get("ok"),
            "strategy": observation.get("strategy"),
            "round": observation.get("round"),
            "rounds": observation.get("rounds"),
            "final": str(observation.get("output") or "")[:3_200],
            "summary": str(observation.get("summary") or "")[:150],
            "agents": agents,
            "reviews": reviews,
            "messages": observation.get("messages", 0),
            "blackboard": observation.get("blackboard", 0),
            "changed_files": [str(item) for item in
                              (observation.get("changed_files") or [])[:6]],
            "usage": observation.get("usage") or {},
            "elapsed": observation.get("elapsed"),
        }

    def _execute_defined_tool(self, run: IdeRun, workspace_id: str, session: dict,
                              definition: dict, tool_type: str, arguments: dict,
                              excluded: list[str] | None = None
                              ) -> tuple[dict, bool, dict | None, dict]:
        """Run one named tool for the parent agent or a swarm member.

        Returns ``(observation, ok, change, normalised arguments)``.  Writes are
        snapshotted and reported exactly as they are for the parent agent, so a
        swarm edit is visible in the panel, revertible and diffable.
        """
        if tool_type == "skill.read":
            name = str(arguments.get("name") or arguments.get("id") or "").strip()
            try:
                return self.read_skill(workspace_id, name), True, None, arguments
            except (BadRequestError, NotFoundError) as failure:
                return {"error": str(failure)}, False, None, arguments
        if tool_type == "file.inspect":
            try:
                observation = self.preview_file(
                    workspace_id, str(arguments.get("path") or "").strip(), limit=0,
                )
                observation.pop("data_url", None)
                if isinstance(observation.get("content"), str):
                    observation["content"] = observation["content"][:4000]
                if isinstance(observation.get("hex"), str):
                    lines = observation["hex"].splitlines()
                    observation["hex"] = "\n".join(lines[:32])
                    observation["hex_truncated"] = len(lines) > 32
                return observation, True, None, arguments
            except (BadRequestError, NotFoundError, ValueError, OSError) as failure:
                return {"error": str(failure)}, False, None, arguments
        relative = str(arguments.get("path") or "").strip()
        try:
            if relative and relative not in {".", "./"}:
                resolved = self._workspace_file(
                    workspace_id, relative, for_write=tool_type in {"file.write", "file.edit"},
                )
                normalised = resolved.relative_to(
                    self.workspace_path(workspace_id)
                ).as_posix()
                if normalised != relative:
                    arguments = {**arguments, "path": normalised}
                relative = normalised
            if tool_type in _WRITE_TOOLS and relative:
                self._prepare_change(session, workspace_id, relative)
            observation = execute_ability(
                definition, arguments, {"agent": {"id": "ide"}, "run": {"id": run.id}},
                self.root, files_root=self.workspace_path(workspace_id),
                cancelled=run.cancelled, deny=self._tool_deny(workspace_id),
                os_access=self.os_access,
            )
            observation = self._filter_observation(tool_type, observation, excluded or [])
            change = None
            if tool_type in _WRITE_TOOLS and relative:
                change = self._record_change(session, workspace_id, relative)
                if isinstance(observation, dict) and change:
                    observation = {
                        **observation,
                        "added": change.get("added", 0),
                        "removed": change.get("removed", 0),
                    }
                    if change.get("diff"):
                        observation["diff"] = change["diff"][:4000]
                if change:
                    run.event("file_changed", path=change["path"], added=change["added"],
                              removed=change["removed"], diff=change["diff"],
                              truncated=change.get("truncated", False),
                              tracked=change.get("tracked", True))
                    if (isinstance(observation, dict) and tool_type in {"file.write", "file.edit"}
                            and Path(relative).suffix.lower() in _DIAGNOSTIC_EXTENSIONS):
                        try:
                            report = self.diagnostics(workspace_id, relative)
                            issues = report.get("issues") if isinstance(report, dict) else None
                            if issues:
                                observation["diagnostics"] = {
                                    "ok": False, "issues": issues[:20],
                                    "more": max(0, len(issues) - 20),
                                }
                            elif isinstance(report, dict):
                                observation["diagnostics"] = {
                                    "ok": True, "files_checked": report.get("files_checked", 1),
                                }
                        except Exception:  # noqa: BLE001 - verification must not fail the edit
                            pass
            return observation, True, change, arguments
        except Exception as failure:  # noqa: BLE001 - the model sees the failure
            if tool_type in _WRITE_TOOLS and relative:
                existing = (session.get("changes") or {}).get(relative)
                if existing and existing.get("edits", 0) == 0:
                    session["changes"].pop(relative, None)
            return {"error": str(failure)}, False, None, arguments

    def _filter_observation(self, tool_type: str, observation, excluded: list[str]):
        """Keep hidden folders out of listings the model reads, saving prompt tokens."""
        if not excluded or not isinstance(observation, dict):
            return observation
        if tool_type not in {"file.tree", "file.search", "file.list"}:
            return observation
        hidden = 0
        if tool_type == "file.tree" and isinstance(observation.get("entries"), list):
            kept = []
            for entry in observation["entries"]:
                relative = str(entry.get("path") or "") if isinstance(entry, dict) else ""
                if relative and _under(relative, excluded):
                    hidden += 1
                else:
                    kept.append(entry)
            observation = {**observation, "entries": kept}
        elif tool_type == "file.search" and isinstance(observation.get("matches"), list):
            kept = []
            for match in observation["matches"]:
                relative = str(match.get("path") or "") if isinstance(match, dict) else ""
                if relative and _under(relative, excluded):
                    hidden += 1
                else:
                    kept.append(match)
            observation = {**observation, "matches": kept}
        elif tool_type == "file.list" and isinstance(observation.get("files"), list):
            kept = []
            for item in observation["files"]:
                relative = str(item.get("name") or "") if isinstance(item, dict) else ""
                if relative and _under(relative, excluded):
                    hidden += 1
                else:
                    kept.append(item)
            observation = {**observation, "files": kept}
        if hidden:
            observation["context_note"] = (
                f"{hidden} entr{'y' if hidden == 1 else 'ies'} hidden by the user's "
                "context rules; do not ask for them."
            )
        return observation

    @staticmethod
    def _shrink_value(value, field_cap: int, list_cap: int,
                      path: str = "") -> tuple[object, list[str]]:
        """Copy a JSON value with long strings and lists shortened."""
        if isinstance(value, str):
            if len(value) > field_cap:
                return value[:field_cap], [path or "value"]
            return value, []
        if isinstance(value, list):
            kept = value[:list_cap]
            fields = [path or "list"] if len(value) > list_cap else []
            result = []
            for index, item in enumerate(kept):
                shrunk, nested = IdeService._shrink_value(
                    item, field_cap, list_cap,
                    f"{path}[{index}]" if path else f"[{index}]",
                )
                result.append(shrunk)
                fields.extend(nested)
            return result, fields
        if isinstance(value, dict):
            result = {}
            fields = []
            for key, item in value.items():
                if key in {"truncated", "truncated_fields"}:
                    result[key] = item
                    continue
                shrunk, nested = IdeService._shrink_value(
                    item, field_cap, list_cap, f"{path}.{key}" if path else str(key),
                )
                result[key] = shrunk
                fields.extend(nested)
            return result, fields
        return value, []

    @staticmethod
    def _is_read_result(value) -> bool:
        """True for the line-range shape returned by ``file.read``."""
        return (isinstance(value, dict) and isinstance(value.get("content"), str)
                and isinstance(value.get("start_line"), int)
                and isinstance(value.get("end_line"), int)
                and isinstance(value.get("total_lines"), int))

    def _trim_read_result(self, value: dict, limit: int) -> dict | None:
        """Shorten a file read without losing which lines were actually sent.

        The generic shrinker cuts ``content`` in the middle and leaves the old
        ``end_line``, so a model that continues at that line silently skips
        code.  This keeps whole lines where possible and reports the real
        included range plus ``next_start_line`` for the continuation.
        """
        content = value["content"]
        start = max(1, value["start_line"])
        original_end = max(start, value["end_line"])
        total = max(0, value["total_lines"])
        shell = {key: item for key, item in value.items() if key != "content"}
        try:
            overhead = len(json.dumps(shell, ensure_ascii=False, default=str)) + 128
        except (TypeError, ValueError):
            return None
        budget = max(32, limit - overhead)
        for _attempt in range(4):
            kept: list[str] = []
            used = 0
            midline = False
            for line in content.splitlines(keepends=True):
                cost = len(json.dumps(line, ensure_ascii=False))
                if not kept and cost > budget:
                    cut = max(1, budget // 2)
                    while cut > 1 and len(json.dumps(line[:cut], ensure_ascii=False)) > budget:
                        cut //= 2
                    kept.append(line[:cut])
                    midline = True
                    break
                if kept and used + cost > budget:
                    break
                kept.append(line)
                used += cost
                if used >= budget:
                    break
            body = "".join(kept)
            if not body and content:
                body = content[:max(1, budget // 2)]
                midline = True
            if len(body) >= len(content) and not midline:
                return None
            covered = (body.count("\n") + (0 if body.endswith("\n") else 1)) if body else 0
            actual_end = min(original_end, start + covered - 1) if covered else start
            if midline:
                actual_end = start
            result = {**value, "content": body, "end_line": actual_end, "truncated": True}
            fields = [str(field) for field in value.get("truncated_fields") or []]
            if "content" not in fields:
                fields.append("content")
            result["truncated_fields"] = fields[:16]
            if midline:
                result["content_cut_midline"] = True
                result["hint"] = (f"The content was cut inside line {start}; use file.search "
                                  "or file.inspect to inspect the rest.")
                result.pop("next_start_line", None)
            elif actual_end < total:
                result["next_start_line"] = actual_end + 1
            else:
                result.pop("next_start_line", None)
            try:
                if len(json.dumps(result, ensure_ascii=False, default=str)) <= limit:
                    return result
            except (TypeError, ValueError):
                return None
            budget = max(32, int(budget * 0.7))
        return None

    def _bounded_observation(self, value, limit: int = MAX_OBSERVATION_CHARS):
        """Shrink a tool result but keep the keys the agent needs to continue.

        ``json_safe`` replaces a large result with one opaque preview, losing the
        path, line range and truncation flags.  A long coding run needs those to
        ask for the next range, so long fields are shortened in place and the
        structure is preserved.
        """
        if not isinstance(value, dict):
            return json_safe(value, limit)
        try:
            encoded = json.dumps(value, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return json_safe(value, limit)
        if len(encoded) <= limit:
            return value
        if self._is_read_result(value):
            trimmed = self._trim_read_result(value, limit)
            if trimmed is not None:
                value = trimmed
                try:
                    encoded = json.dumps(value, ensure_ascii=False, default=str)
                except (TypeError, ValueError):
                    return json_safe(value, limit)
                if len(encoded) <= limit:
                    return value
        for field_cap, list_cap in ((OBSERVATION_FIELD_CHARS, 60), (2_000, 40),
                                    (1_000, 20), (500, 10), (200, 5)):
            shrunk, fields = self._shrink_value(value, field_cap, list_cap)
            if not isinstance(shrunk, dict):
                continue
            if fields:
                shrunk["truncated"] = True
                shrunk["truncated_fields"] = list(dict.fromkeys(fields))[:16]
            try:
                encoded = json.dumps(shrunk, ensure_ascii=False, default=str)
            except (TypeError, ValueError):
                return json_safe(value, limit)
            if len(encoded) <= limit:
                return shrunk
        return json_safe(value, limit)

    @staticmethod
    def _progress_line(session: dict | None) -> str:
        """The plan and changed files, kept in front of every tool round."""
        if not isinstance(session, dict):
            return ""
        plan = [item for item in session.get("plan") or [] if isinstance(item, dict)]
        changes = session.get("changes") if isinstance(session.get("changes"), dict) else {}
        changed = [(path, change) for path, change in changes.items()
                   if isinstance(change, dict) and not change.get("reverted")]
        if not plan and not changed:
            return ""
        parts = []
        if plan:
            done = sum(1 for item in plan if item.get("status") == "done")
            active = next((item for item in plan if item.get("status") == "active"), None)
            text = f"Progress: {done}/{len(plan)} plan steps done"
            if active:
                now = " ".join(str(active.get("text") or "").split())[:160]
                text += f" (now: {now})"
            parts.append(text + ".")
        if changed:
            files = ", ".join(
                f"{str(path)[:120]} (+{int(change.get('added') or 0)}/"
                f"−{int(change.get('removed') or 0)})"
                for path, change in changed[:6]
            )
            more = f" and {len(changed) - 6} more" if len(changed) > 6 else ""
            parts.append(f"Files changed: {files}{more}.")
        return " ".join(parts)

    def _observation_message(self, observations: list[dict],
                             session: dict | None = None) -> str:
        count = max(1, len(observations))
        per_result = max(1_200, min(MAX_OBSERVATION_CHARS,
                                   MAX_OBSERVATION_TOTAL_CHARS // count))
        blocks = []
        limited = False
        for item in observations:
            bounded = self._bounded_observation(item["result"], per_result)
            if isinstance(bounded, dict) and (bounded.get("truncated_fields")
                                              or bounded.get("truncated")):
                limited = True
            rendered = json.dumps(bounded, ensure_ascii=False, default=str)
            blocks.append(
                f"{item['tool']} {'returned' if item['ok'] else 'failed'}:\n{rendered}"
            )
        prefix = self._progress_line(session)
        footer = ("Continue with a tool call, update the plan, or reply with "
                  '{"final": "..."} when the task is done.')
        if limited:
            footer += (" Some results were shortened; request a narrower line range "
                       "or a more specific path when you need the rest.")
        body = "Tool results:\n\n" + "\n\n".join(blocks) + "\n\n" + footer
        return f"{prefix}\n\n{body}" if prefix else body

    @staticmethod
    def _should_extend(recent) -> bool:
        """Extend only when the last few rounds produced new, successful work.

        A model looping on one read or failing repeatedly should stop at its
        budget, not be rewarded with more steps.
        """
        if len(recent) < 2:
            return False
        successes = [(signature, tool) for signature, ok, tool in recent if ok and signature]
        if not successes:
            return False
        return (len({signature for signature, _ in successes}) >= 2
                or any(tool in _WRITE_TOOLS for _, tool in successes))

    def _final_text(self, action: dict, reply: str) -> str:
        value = action.get("final") if "final" in action else reply
        if isinstance(value, str):
            return value.strip()[:20_000]
        return json.dumps(value, indent=2, ensure_ascii=False, default=str)[:20_000]

    def _finish_plan(self, run: IdeRun, session: dict) -> None:
        """Mark the current plan complete when a run ends with a final answer.

        Models often forget the last few status updates.  Exhausted, timed-out
        and failed runs keep their remaining steps so a continuation shows what
        is still left to do.
        """
        plan = session.get("plan") if isinstance(session.get("plan"), list) else run.plan
        todos = [item for item in plan or [] if isinstance(item, dict)]
        if not todos or all(item.get("status") == "done" for item in todos):
            return
        completed = [{**item, "status": "done"} for item in todos]
        run.plan = completed
        run.plan_set = True
        session["plan"] = completed
        for entry in reversed(session.get("transcript") or []):
            if isinstance(entry, dict) and entry.get("role") == "plan":
                entry["todos"] = completed
                break
        run.event("plan", plan=completed, message="Plan completed")

    def _append_transcript(self, session: dict, entry: dict) -> None:
        entry.setdefault("at", _iso())
        session.setdefault("transcript", []).append(entry)
        if len(session["transcript"]) > 1000:
            del session["transcript"][:200]

    # -- context ---------------------------------------------------------
    def _output_limit(self, model_id: str) -> int:
        """The output budget for one coding step.

        Saved endpoints may declare ``max_output_tokens``; it is honoured up to
        the agent's own ceiling so one step cannot monopolise a long run.
        """
        try:
            catalog = self.gateway.catalog()
        except Exception:  # noqa: BLE001 - a catalog failure falls back to the default
            return DEFAULT_OUTPUT_TOKENS
        entry = next((item for item in catalog if item.get("id") == model_id), None)
        detail = (entry or {}).get("detail")
        detail = detail if isinstance(detail, dict) else {}
        try:
            declared = int(detail.get("max_output_tokens") or 0)
        except (TypeError, ValueError):
            declared = 0
        if declared:
            return max(64, min(declared, MAX_OUTPUT_TOKENS))
        resolver = getattr(self.gateway, "output_limit", None)
        if callable(resolver):
            try:
                default = int(resolver(model_id) or 0)
            except Exception:  # noqa: BLE001 - a missing limit falls back below
                default = 0
            if default > 0:
                return max(64, min(default, MAX_OUTPUT_TOKENS))
        return DEFAULT_OUTPUT_TOKENS

    def _pick_model(self, requested: str) -> str:
        catalog = self.gateway.catalog()
        chat = [entry for entry in catalog
                if entry.get("available") and entry.get("capabilities", {}).get("chat")]
        if requested:
            entry = next((item for item in catalog if item.get("id") == requested), None)
            if entry is None:
                raise ValueError(
                    f"Model {requested!r} is not in the catalog. Pick one in the IDE toolbar."
                )
            if not entry.get("available"):
                raise ValueError(entry.get("reason") or f"Model {requested} is unavailable")
            if not entry.get("capabilities", {}).get("chat"):
                raise ValueError(f"Model {requested} cannot run chat completions")
            return requested
        if not chat:
            raise ValueError(
                "No chat model is available. Download or connect one in Models first."
            )
        return chat[0]["id"]

    def _tools_for_mode(self, mode: str, workspace_id: str | None = None) -> list[dict]:
        tools = []
        skills = bool(workspace_id and self.list_skills(workspace_id))
        for tool_type in MODE_INFO[mode]["tools"]:
            if tool_type == "shell.exec" and not self.shell_enabled:
                continue
            if tool_type == "browser.fetch" and not self.os_access:
                continue
            if tool_type == "agent.swarm":
                tools.append({
                    "id": "agent.swarm", "type": "agent.swarm", "label": "Spawn a research swarm",
                    "enabled": True, "config": {},
                })
                continue
            if tool_type == "skill.read":
                if not skills:
                    continue
                tools.append({
                    "id": "skill.read", "type": "skill.read", "label": "Read a skill",
                    "enabled": True, "config": {},
                })
                continue
            if tool_type == "file.inspect":
                tools.append({
                    "id": "file.inspect", "type": "file.inspect", "label": "Inspect a file",
                    "enabled": True, "config": {},
                })
                continue
            meta = ABILITY_TYPES.get(tool_type, {})
            config = dict(meta.get("config") or {})
            if tool_type == "shell.exec":
                config["timeout"] = 120
            elif tool_type in {"http.fetch", "news.headlines", "browser.fetch"} and self.os_access:
                config["allow_private"] = True
            tools.append({
                "id": tool_type, "type": tool_type, "label": meta.get("label", tool_type),
                "enabled": True, "config": config,
            })
        return tools

    def _active_file(self, workspace_id: str, relative: str) -> dict | None:
        if not relative:
            return None
        try:
            data = self.read_file(workspace_id, relative, limit=400)
        except (BadRequestError, NotFoundError, ValueError, OSError):
            return None
        if len(data["content"]) > 6_000:
            data["content"] = data["content"][:6_000]
            covered = (data["content"].count("\n")
                       + (0 if data["content"].endswith("\n") else 1))
            data["end_line"] = data["start_line"] + max(0, covered - 1)
            data["truncated"] = True
        return data

    def _map(self, workspace: Path, limit: int = 160,
             include: list[str] | None = None,
             exclude: list[str] | None = None) -> str:
        """A compact file map, pruned so large ignored folders cost nothing."""
        lines: list[str] = []
        include = include or []
        exclude = exclude or []
        try:
            walker = os.walk(workspace)
            private = self._private_subtree(workspace)
            for folder, directories, names in walker:
                directories[:] = sorted(
                    name for name in directories
                    if name not in IGNORED_DIRS
                    and not (Path(folder) / name).is_symlink()
                    and (private is None or (Path(folder) / name) != private)
                    and not _under((Path(folder) / name).relative_to(workspace).as_posix(), exclude)
                )
                relative_folder = Path(folder).relative_to(workspace).as_posix()
                base = "" if relative_folder == "." else f"{relative_folder}/"
                for name in sorted(names):
                    if name == MANIFEST or (name.startswith(".") and name.endswith(".tmp")):
                        continue
                    path = Path(folder) / name
                    if path.is_symlink() or not path.is_file():
                        continue
                    relative = f"{base}{name}"
                    if _under(relative, exclude):
                        continue
                    if include and not _under(relative, include):
                        continue
                    try:
                        lines.append(f"{relative} ({path.stat().st_size} B)")
                    except OSError:
                        continue
                    if len(lines) >= limit:
                        lines.append("… (map truncated)")
                        return ("\n".join(lines))[:6_000]
        except OSError:
            return "(project map unavailable)"
        return ("\n".join(lines) or "(empty workspace)")[:6_000]

    def _system_sections(self, workspace_id: str, mode: str, tools: list[dict],
                         workspace: Path, active: dict | None,
                         context: dict | None = None,
                         session: dict | None = None) -> list[dict]:
        """The system prompt split into labelled pieces for the budget view."""
        name = self._manifest(workspace_id).get("name") or workspace_id
        excluded = self._excluded_paths(workspace_id, context)
        include = self._context_include(workspace_id, context)
        lines = [
            "You are Mariana IDE, an expert software engineer working inside Studio. "
            "You complete coding tasks by inspecting the project, making precise edits, "
            "and verifying the result.",
            f"Workspace: {name}. Every tool path is relative to the workspace root. "
            f"Current mode: {mode} ({MODE_INFO[mode]['description']})",
            "",
            "Operating rules:",
            "1. Inspect before you edit: use file.tree, file.search and file.read.",
            "   file.read returns up to 200 lines or about 3,600 characters; when truncated "
            "is true, continue from next_start_line.",
            "   Use file.inspect for images and other binary files; file.read handles UTF-8 text only.",
            "2. Make the smallest change that solves the task. Use file.edit for a targeted "
            "change; use file.write only for a new file or a full rewrite after reading "
            "every line.",
            "3. Never invent file contents. Read a file immediately before editing it.",
            "4. Do not write secrets, credentials, or unrelated formatting churn.",
            "5. Verify: call code.diagnostics after edits. Run tests with shell.exec only "
            "when it is listed below.",
            "6. Keep a short plan when the task has several steps.",
            "7. Use agent.swarm when the task has independent parts, needs several areas "
            "investigated at once, or benefits from independent review. Do not spawn a swarm "
            "for a one-file change.",
            "8. Batch independent reads in one turn: use {\"calls\": [...]} instead "
            "of one round per file. Independent read-only calls run in parallel; "
            "then edit as soon as you have the needed context.",
            "9. Finish with a concise summary of what changed and what was verified.",
            "",
            "Reply with only one JSON object per turn:",
            '  {"tool": "<tool id>", "input": {<input keys>}}',
            '  {"calls": [{"tool": "<tool id>", "input": {...}}, ...]}',
            '  {"plan": [{"text": "<step>", "status": "pending|active|done"}]}',
            '  {"final": "<short summary>"}',
            "Hidden reasoning alone is not a reply: every turn must include a tool call, "
            "a plan, or a final object.",
            "",
            "Available tools:",
        ]
        for tool in tools:
            if tool["type"] == "skill.read":
                lines.append(
                    "- skill.read: Load one relevant skill's full instructions by exact id "
                    "(input keys: name). Decide from the skills section when a skill fits the "
                    "task, then call this before following it; do not wait to be told."
                )
                continue
            if tool["type"] == "file.inspect":
                lines.append(
                    "- file.inspect: Metadata for any file, including binary images; returns "
                    "the MIME type, size, dimensions, a hex dump and, for text, the first "
                    "4,000 characters (input keys: path)."
                )
                continue
            if tool["type"] == "agent.swarm":
                lines.append(
                    "- agent.swarm: Run a leader-chosen team of up to six agents that work in "
                    "parallel, message each other, share a blackboard, review and revise each "
                    "other's work, and return one synthesised answer. In Edit and Auto the "
                    "members may edit files; file-tool writes are serialized and snapshotted, "
                    "and in Ask the swarm is read-only. Input keys: tasks (array of strings or "
                    "{id, label, role, task, tools, writes} objects; empty lets the leader plan), "
                    "strategy (auto, map_reduce, peer_review, debate, pipeline), rounds (0-2), "
                    "write (boolean), max_agents, max_steps, max_seconds, goal, system, "
                    "leader_model, model, ipc, memory."
                )
                continue
            meta = ABILITY_TYPES.get(tool["type"], {})
            keys = ", ".join(field["key"] for field in meta.get("params", [])) or "none"
            lines.append(f"- {tool['id']}: {meta.get('description', '')} Input keys: {keys}.")
        if not self.shell_enabled:
            lines.append(
                "shell.exec is disabled on this server. Do not ask for it; verify with "
                "code.diagnostics instead."
            )
        elif self.os_access:
            lines.append(
                "OS access is enabled: shell.exec may run any command and tools may reach "
                "private addresses. Use the smallest command that answers the request and "
                "never run a destructive command without explicit instruction."
            )
        if excluded:
            shown = ", ".join(excluded[:20])
            lines.append(
                "The user excluded this context from the agent; it is hidden from listings, "
                f"searches and the project map: {shown}."
            )
        sections = [{"id": "instructions", "label": "Agent instructions",
                     "text": "\n".join(lines)}]
        instructions = ""
        for filename in ("AGENTS.md", ".mariana-ide.md"):
            path = workspace / filename
            if path.is_file() and not path.is_symlink():
                try:
                    instructions = path.read_bytes()[:4000].decode("utf-8", errors="replace")
                except OSError:
                    instructions = ""
                break
        if instructions:
            sections.append({
                "id": "project", "label": "Project instructions",
                "text": f"Project instructions:\n{instructions}",
            })
        sections.append({
            "id": "map", "label": "Project map",
            "text": f"Project map:\n{self._map(workspace, include=include, exclude=excluded)}",
        })
        session = session if isinstance(session, dict) else {}
        plan = [item for item in session.get("plan") or [] if isinstance(item, dict)]
        if plan:
            plan_lines = [
                "Current plan from earlier turns. Continue it, or replace it with a new "
                "plan when this task differs:"
            ]
            markers = {"done": "x", "active": ">", "pending": " "}
            for todo in plan[:20]:
                status = str(todo.get("status") or "pending")
                plan_lines.append(
                    f"- [{markers.get(status, ' ')}] {str(todo.get('text') or '')[:200]}"
                )
            sections.append({"id": "plan", "label": "Current plan",
                             "text": "\n".join(plan_lines)})
        changes = session.get("changes") if isinstance(session.get("changes"), dict) else {}
        changed = [(path, change) for path, change in changes.items()
                   if isinstance(change, dict) and not change.get("reverted")]
        if changed:
            progress_lines = [
                "Files this conversation already changed. Revert restores the originals, "
                "so read before editing again:"
            ]
            for path, change in changed[:20]:
                progress_lines.append(
                    f"- {str(path)[:200]} (+{int(change.get('added') or 0)}/"
                    f"−{int(change.get('removed') or 0)})"
                )
            if len(changed) > 20:
                progress_lines.append(f"- … and {len(changed) - 20} more")
            sections.append({"id": "progress", "label": "Session progress",
                             "text": "\n".join(progress_lines)})
        skills = self.list_skills(workspace_id)
        if skills:
            skill_lines = [
                "All skills below are available automatically. Decide which skills fit the "
                "request, then call skill.read for each relevant one before doing the work. "
                "Do not wait for the user to name a skill, and do not load unrelated skills. "
                "Follow the loaded instructions; never claim to use a skill you have not read."
            ]
            skill_lines.extend(self._skill_list_lines(skills, include_scope=True))
            sections.append({"id": "skills", "label": f"Skills ({len(skills)})",
                             "text": "\n".join(skill_lines)})
        if include:
            blocks = self._context_blocks(workspace_id, include, excluded) or "(no readable files)"
            sections.append({
                "id": "pinned", "label": "Pinned context",
                "text": f"Pinned context: {', '.join(include)}\n\n{blocks}",
            })
        if active and active.get("content"):
            note = ""
            if active.get("truncated"):
                note = (f"\n\n(The editor preview is truncated at line "
                        f"{active.get('end_line')} of {active.get('total_lines')}; "
                        "use file.read to see the rest before editing.)")
            sections.append({
                "id": "active", "label": f"Active file: {active['path']}",
                "text": (f"File currently open in the editor: {active['path']}\n"
                         f"```\n{active['content']}\n```{note}"),
            })
        return sections

    def _system_prompt(self, workspace_id: str, mode: str, tools: list[dict],
                       workspace: Path, active: dict | None,
                       context: dict | None = None,
                       session: dict | None = None) -> str:
        return "\n\n".join(
            section["text"]
            for section in self._system_sections(workspace_id, mode, tools, workspace,
                                                 active, context, session)
        )

    def _prompt_pieces(self, workspace_id: str, mode: str, tools: list[dict],
                       workspace: Path, active: dict | None, context: dict | None,
                       message: str, context_files: list[str],
                       session: dict | None = None) -> tuple[list[dict], str, bool]:
        """The exact prompt pieces shared by the live request and the budget view."""
        pieces = self._system_sections(workspace_id, mode, tools, workspace, active,
                                       context, session)
        pieces.append({
            "id": "task", "label": "Current task",
            "text": f"Current task:\n{str(message or '')[:3000]}",
        })
        excluded = self._excluded_paths(workspace_id, context)
        seen = {active["path"]} if active else set()
        for relative in context_files or []:
            relative = str(relative or "")[:1000]
            if relative in seen or not relative or _under(relative, excluded):
                continue
            seen.add(relative)
            try:
                extra = self.read_file(workspace_id, relative, limit=200)
            except (BadRequestError, NotFoundError, ValueError, OSError):
                continue
            body = extra["content"][:1500]
            note = ""
            if extra.get("truncated") or len(extra["content"]) > 1500:
                note = "\n(Preview only; use file.read for the complete file.)"
            pieces.append({
                "id": f"open:{relative}", "label": f"Open file: {relative}",
                "text": (f"Other file open in the editor: {extra['path']}\n"
                         f"```\n{body}\n```{note}"),
            })
        raw_parts = [piece["text"] for piece in pieces if piece["id"] != "task"]
        task_text = str(message or "").strip()
        if task_text:
            # Keep a short anchor in the system prompt, but let the full task
            # travel once as the user message instead of duplicating it.
            raw_parts.insert(0, "Current task:\n" + task_text[:600])
        raw = "\n\n".join(raw_parts)
        return pieces, raw[:MAX_PROMPT_CHARS], len(raw) > MAX_PROMPT_CHARS

    def _model_budget(self, model_id: str, force: bool = False) -> dict:
        """The selected model's latest known context window.

        The gateway resolves the model that would actually answer (including the
        default when no model is selected), then compares its catalog metadata
        with a cached live probe of the provider or local runtime and keeps the
        larger window; ``force`` refreshes that probe.  Gateways used by tests
        may only expose ``catalog``; that path still resolves the default model
        and reads the same metadata fields.
        """
        resolver = getattr(self.gateway, "context_window", None)
        if callable(resolver):
            resolved = None
            try:
                resolved = resolver(model_id or None, force=force)
            except TypeError:
                try:
                    resolved = resolver(model_id or None)
                except Exception:  # noqa: BLE001 - fall back to catalog metadata
                    resolved = None
            except Exception:  # noqa: BLE001 - fall back to catalog metadata
                resolved = None
            if isinstance(resolved, dict):
                return self._budget_from_resolved(resolved)
        return self._catalog_model_budget(model_id)

    @staticmethod
    def _budget_from_resolved(resolved: dict) -> dict:
        window = int(resolved.get("context_window") or 0)
        source = str(resolved.get("source") or "unknown")
        if window <= 0:
            window = DEFAULT_CONTEXT_WINDOW
            source = "unavailable" if source in {"endpoint", "runtime"} else "default"
        return {
            "id": str(resolved.get("id") or ""),
            "name": str(resolved.get("name") or ""),
            "kind": str(resolved.get("kind") or ""),
            "found": bool(resolved.get("found")),
            "context_window": window,
            "source": source,
            "unit": str(resolved.get("unit") or "tokens"),
        }

    def _catalog_model_budget(self, model_id: str) -> dict:
        """Fallback for gateways that only publish a catalog."""
        try:
            catalog = self.gateway.catalog()
        except Exception:  # noqa: BLE001 - a catalog failure must not hide the budget
            catalog = []
        entry = None
        if model_id:
            entry = next((item for item in catalog if item.get("id") == model_id), None)
        else:
            chat = [item for item in catalog
                    if item.get("available") and (item.get("capabilities") or {}).get("chat")]
            entry = chat[0] if chat else (catalog[0] if catalog else None)
        detail = (entry or {}).get("detail")
        detail = detail if isinstance(detail, dict) else {}
        kind = str((entry or {}).get("kind") or "")
        unit = "bytes" if detail.get("ctx_unit") == "bytes" or kind in {
            "mariana-drv", "mariana-ncm",
        } else "tokens"
        window = entry_context(entry) if entry else 0
        source = "catalog" if window > 0 else "default"
        if window <= 0:
            window = DEFAULT_CONTEXT_WINDOW
        return {
            "id": str((entry or {}).get("id") or model_id or ""),
            "name": str((entry or {}).get("name") or ""),
            "kind": kind,
            "found": entry is not None,
            "context_window": window,
            "source": source,
            "unit": unit,
        }

    def _section_budget(self, pieces: list[dict], unit: str) -> list[dict]:
        """Count each prompt piece, marking where the character ceiling cuts it."""
        sections = []
        remaining = MAX_PROMPT_CHARS
        for index, piece in enumerate(pieces):
            value = str(piece.get("text") or "")
            if index:
                remaining = max(0, remaining - 2)  # the "\n\n" joiner
            included = value[:remaining]
            remaining = max(0, remaining - len(included))
            sections.append({
                "id": str(piece.get("id") or ""),
                "label": str(piece.get("label") or ""),
                "chars": utf8_length(value),
                "tokens": estimate_tokens(value, unit),
                "included_chars": utf8_length(included),
                "included_tokens": estimate_tokens(included, unit),
                "clipped": len(included) < len(value),
            })
        return sections

    def _budget_from_parts(self, model_id: str, pieces: list[dict], prompt: str,
                           truncated: bool, history: list[dict], message: str,
                           force: bool = False) -> dict:
        """One budget calculation for both the live run and the preview route."""
        model = self._model_budget(model_id, force=force)
        unit = model["unit"]
        sections = self._section_budget(pieces, unit)
        task = next((section for section in sections if section["id"] == "task"), None)
        task_tokens = task["included_tokens"] if task else estimate_tokens(
            str(message or "")[:3000], unit)
        history_tokens = sum(
            estimate_tokens(item.get("content") or "", unit) + MESSAGE_TOKEN_OVERHEAD
            for item in history
        )
        draft_tokens = estimate_tokens(message, unit) + MESSAGE_TOKEN_OVERHEAD
        prompt_tokens = estimate_tokens(prompt, unit)
        used = prompt_tokens + history_tokens + draft_tokens
        window = model["context_window"]
        output_reserve = self._output_limit(model["id"] or model_id)
        if unit == "bytes":
            output_reserve *= 4  # conservative bytes per output token
        percent = round(used * 100 / window, 1) if window else 0.0
        warnings: list[dict] = []
        if model_id and not model["found"]:
            warnings.append({
                "tone": "warning",
                "text": "That model is not in the catalog; refresh the library.",
            })
        if model["source"] == "default":
            label = model["name"] or "The selected model"
            warnings.append({
                "tone": "info",
                "text": f"{label} does not report a context window; assuming "
                        f"{window:,} {unit}.",
            })
        elif model["source"] == "unavailable":
            label = model["name"] or "the provider"
            warnings.append({
                "tone": "info",
                "text": f"Could not read a context window from {label}; assuming "
                        f"{window:,} {unit}. Set it on the saved endpoint if the "
                        "provider does not expose one.",
            })
        if truncated:
            warnings.append({
                "tone": "warning",
                "text": f"The system prompt hit the {MAX_PROMPT_CHARS:,}-character ceiling; "
                        "later sections are clipped.",
            })
        if used > window:
            warnings.append({
                "tone": "danger",
                "text": f"The next request is estimated at {used:,} {unit}, above the "
                        f"{window:,} {unit} window. Start a new conversation, unpin files, "
                        "or choose a larger model.",
            })
        elif percent >= BUDGET_WARNING_PERCENT:
            warnings.append({
                "tone": "warning",
                "text": f"{percent:.0f}% of the model context is used; older turns "
                        f"consolidate automatically at {CONSOLIDATE_PERCENT}% and pinned files "
                        "can be unpinned to make room for tool results.",
            })
        if window and used <= window and used + output_reserve > window:
            warnings.append({
                "tone": "danger",
                "text": f"The next request uses {used:,} {unit} and reserves "
                        f"{output_reserve:,} {unit} for output, above the "
                        f"{window:,} {unit} window. Unpin files or choose a larger model.",
            })
        if len(history) >= MESSAGE_RECAP_TRIGGER:
            warnings.append({
                "tone": "info",
                "text": f"Older turns are consolidated into a recap when more than "
                        f"{MESSAGE_RECAP_TRIGGER} messages would be sent; the newest eight "
                        "stay verbatim.",
            })
        return {
            "model": model,
            "unit": unit,
            "estimator": "bytes" if unit == "bytes" else "heuristic",
            "sections": sections,
            "prompt": {"chars": utf8_length(prompt), "tokens": prompt_tokens},
            "history": {"messages": len(history), "tokens": history_tokens},
            "draft": {"chars": len(str(message or "")), "tokens": draft_tokens},
            "task": {"chars": len(str(message or "")[:3000]), "tokens": task_tokens},
            "used_tokens": used,
            "context_window": window,
            "output_reserve": output_reserve,
            "remaining_tokens": max(0, window - used),
            "percent": percent,
            "over": used > window,
            "truncated": truncated,
            "warnings": warnings,
            "limits": {
                "prompt_chars": MAX_PROMPT_CHARS,
                "context_files": MAX_CONTEXT_FILES,
                "context_chars": MAX_CONTEXT_CHARS,
                "messages": MODEL_MESSAGE_WINDOW - 1,
                "message_chars": MAX_MESSAGE_CHARS,
            },
        }

    def context_budget(self, workspace_id: str, payload: dict | None = None) -> dict:
        """Estimate the next request and break it down by prompt section."""
        self.workspace_path(workspace_id)
        payload = payload if isinstance(payload, dict) else {}
        _include, _exclude, context = self._context_scope(workspace_id, payload)
        message = text(payload, "message", "", 20_000)
        mode = choice(payload, "mode", "edit", MODES)
        model_id = text(payload, "model", "", 2000)
        active_file = text(payload, "active_file", "", 1000)
        context_files = payload.get("context_files")
        context_files = [str(item)[:1000] for item in context_files[:8]] \
            if isinstance(context_files, list) else []
        session_id = text(payload, "session_id", "", 60)
        refresh = flag(payload, "refresh", False)
        session: dict = {}
        if session_id:
            session = self.get_session(session_id, include_messages=True)
            if session.get("workspace_id") != workspace_id:
                raise BadRequestError("That conversation belongs to another workspace")
        workspace = self.workspace_path(workspace_id)
        active = self._active_file(workspace_id, active_file)
        tools = self._tools_for_mode(mode, workspace_id)
        pieces, prompt, truncated = self._prompt_pieces(
            workspace_id, mode, tools, workspace, active, context, message,
            context_files, session,
        )
        history = self._history(session)[-(MODEL_MESSAGE_WINDOW - 1):]
        budget = self._budget_from_parts(
            model_id or str(session.get("model") or ""), pieces, prompt, truncated,
            history, message, force=refresh,
        )
        usage = session.get("usage")
        budget["last_usage"] = usage if isinstance(usage, dict) else {}
        budget["consolidation"] = dict(session.get("consolidation") or {})
        budget["auto_consolidate_percent"] = CONSOLIDATE_PERCENT
        return budget

    def _model_messages(self, messages: list[dict]) -> list[dict]:
        """Keep the system prompt and the newest turns within the gateway's window."""
        if not messages:
            return []
        if messages[0].get("role") == "system":
            return [messages[0]] + messages[1:][-(MODEL_MESSAGE_WINDOW - 1):]
        return messages[-MODEL_MESSAGE_WINDOW:]

    def _history(self, session: dict) -> list[dict]:
        history = []
        for item in session.get("messages") or []:
            if not isinstance(item, dict):
                continue
            if item.get("role") in {"user", "assistant"} and str(item.get("content") or "").strip():
                history.append({"role": item["role"], "content": str(item["content"])[:MAX_MESSAGE_CHARS]})
        return history[-MAX_MESSAGES:]

    def _trim_history(self, messages: list[dict]) -> list[dict]:
        trimmed = []
        for item in messages[-MAX_MESSAGES:]:
            if item.get("role") not in {"user", "assistant"}:
                continue
            content = str(item.get("content") or "")
            if not content:
                continue
            trimmed.append({"role": item["role"], "content": content[:MAX_MESSAGE_CHARS]})
        return trimmed

    @staticmethod
    def _message_tokens(messages: list[dict], unit: str) -> int:
        return sum(
            estimate_tokens(str(item.get("content") or ""), unit) + MESSAGE_TOKEN_OVERHEAD
            for item in messages
        )

    @staticmethod
    def _consolidation_summary(messages: list[dict]) -> str:
        """A compact, deterministic record of turns that no longer fit verbatim."""
        lines = ["Earlier conversation, consolidated to keep the context small:"]
        for item in messages:
            role = str(item.get("role") or "")
            content = str(item.get("content") or "").strip()
            if not content:
                continue
            one_line = " ".join(content.split())
            if role == "assistant":
                action, calls = parse_model_action(content)
                if isinstance(action, dict):
                    plan = action.get("plan")
                    if isinstance(plan, list):
                        steps = "; ".join(
                            str(step.get("text") if isinstance(step, dict) else step)
                            for step in plan[:6]
                        )
                        if steps:
                            lines.append(f"- Plan: {steps}")
                    if calls:
                        names = ", ".join(_tool_call_label(call) for call in calls[:8])
                        lines.append(f"- Tool calls: {names}")
                    if "final" in action and not calls:
                        final = action.get("final")
                        final_text = final if isinstance(final, str) else json.dumps(
                            final, ensure_ascii=False, default=str)
                        lines.append(f"- Answer: {' '.join(final_text.split())[:600]}")
                    elif not calls and plan is None:
                        lines.append(f"- Assistant: {one_line[:500]}")
                else:
                    lines.append(f"- Assistant: {one_line[:500]}")
            elif role == "user":
                if "Tool results:" in content[:400]:
                    lines.append(f"- Tool results: {one_line[:800]}")
                else:
                    lines.append(f"- User: {one_line[:500]}")
        return "\n".join(lines)[:CONSOLIDATE_SUMMARY_CHARS]

    def _consolidate_messages(self, messages: list[dict], unit: str, window: int,
                              keep: int = CONSOLIDATE_KEEP_MESSAGES,
                              force: bool = False) -> tuple[list[dict], int, int]:
        """Replace older turns with one recap, keeping the newest turns verbatim."""
        before = self._message_tokens(messages, unit)
        leading = 0
        while leading < len(messages) and messages[leading].get("role") == "system":
            leading += 1
        body = messages[leading:]
        if len(body) <= keep:
            return list(messages), before, before
        summary = self._consolidation_summary(body[:-keep])
        combined = list(messages[:leading]) + [{"role": "user", "content": summary}] + list(body[-keep:])
        after = self._message_tokens(combined, unit)
        if after >= before and not force:
            return list(messages), before, before
        return combined, before, after

    def _record_consolidation(self, session: dict, before: int, after: int,
                              unit: str, trigger: str) -> dict:
        record = session.get("consolidation")
        record = dict(record) if isinstance(record, dict) else {}
        record.update({
            "count": int(record.get("count") or 0) + 1,
            "saved_tokens": int(record.get("saved_tokens") or 0) + max(0, before - after),
            "last_before": before,
            "last_after": after,
            "unit": unit,
            "trigger": trigger,
            "at": _iso(),
        })
        session["consolidation"] = record
        freed = max(0, before - after)
        session.setdefault("transcript", []).append({
            "role": "notice", "kind": "consolidation", "at": _iso(),
            "text": f"Consolidated older turns: about {freed:,} {unit} freed.",
        })
        return record

    def _consolidate_for_budget(self, messages: list[dict], budget: dict, session: dict,
                                trigger: str,
                                max_messages: int | None = None) -> tuple[list[dict], dict | None]:
        """Consolidate when the request would overflow tokens or the message window."""
        unit = str(budget.get("unit") or "tokens")
        window = int(budget.get("context_window") or 0)
        before = self._message_tokens(messages, unit)
        over_messages = bool(max_messages) and len(messages) > max_messages
        reserve = max(0, int(budget.get("output_reserve") or 0))
        over_tokens = window > 0 and (
            before * 100 >= window * CONSOLIDATE_PERCENT
            or (reserve > 0 and before + reserve >= window)
        )
        if not over_messages and not over_tokens:
            return messages, None
        combined, before, after = self._consolidate_messages(
            messages, unit, window, force=over_messages and not over_tokens,
        )
        if after >= before and not over_messages:
            return messages, None
        if len(combined) >= len(messages):
            return messages, None
        self._record_consolidation(session, before, after, unit, trigger)
        return combined, {
            "before_tokens": before, "after_tokens": after, "unit": unit,
            "percent": round(after * 100 / window, 1) if window else 0.0,
        }

    def consolidate_session(self, session_id: str, model_id: str = "") -> dict:
        """Consolidate a saved conversation on demand from the token dialog."""
        session = self.get_session(session_id, include_messages=True)
        if session.get("active_run_id"):
            raise ConflictError("Stop the active run before consolidating this conversation")
        model = self._model_budget(model_id or str(session.get("model") or ""))
        history = self._history(session)
        combined, before, after = self._consolidate_messages(
            history, model["unit"], model["context_window"],
        )
        if after >= before:
            return {
                "consolidated": False,
                "message": "There is not enough older history to consolidate yet.",
                "before_tokens": before, "after_tokens": after, "unit": model["unit"],
            }
        session["messages"] = self._trim_history(combined)
        self._record_consolidation(session, before, after, model["unit"], "manual")
        self._save_session(session)
        return {
            "consolidated": True,
            "message": f"Consolidated older turns; about {before - after:,} "
                       f"{model['unit']} freed.",
            "before_tokens": before, "after_tokens": after, "unit": model["unit"],
            "messages": len(session["messages"]),
        }

    # -- snapshots and diffs ---------------------------------------------
    def _snapshot_path(self, session_id: str, token) -> Path | None:
        value = str(token or "")
        if not re.fullmatch(r"[a-f0-9]{32}", value):
            return None
        return self.snapshots_dir / session_id / f"{value}.orig"

    def _prepare_change(self, session: dict, workspace_id: str, relative: str) -> None:
        changes = session.setdefault("changes", {})
        key = str(relative)
        existing = changes.get(key)
        if existing is not None and existing.get("reverted"):
            changes.pop(key, None)
            existing = None
        if existing is not None:
            return
        path = self._workspace_file(workspace_id, key)
        existed = path.is_file()
        token = None
        tracked = True
        if existed:
            if path.stat().st_size > MAX_SNAPSHOT_BYTES:
                tracked = False
            else:
                token = uuid.uuid4().hex
                snapshot = self.snapshots_dir / session["id"] / f"{token}.orig"
                snapshot.parent.mkdir(parents=True, exist_ok=True)
                snapshot.write_bytes(path.read_bytes())
        changes[key] = {
            "path": key, "existed": existed, "snapshot": token, "tracked": tracked,
            "edits": 0, "added": 0, "removed": 0, "diff": "", "truncated": False,
            "reverted": False,
        }

    def _record_change(self, session: dict, workspace_id: str, relative: str) -> dict | None:
        change = (session.get("changes") or {}).get(str(relative))
        if change is None:
            return None
        if not change.get("tracked", True):
            change["edits"] += 1
            return {key: change[key] for key in
                    ("path", "edits", "added", "removed", "diff", "truncated", "tracked")}
        before = ""
        if change.get("snapshot"):
            snapshot = self._snapshot_path(session["id"], change["snapshot"])
            if snapshot is not None:
                before = _read_snapshot(snapshot)
        path = self._workspace_file(workspace_id, relative)
        current = path.read_bytes() if path.is_file() else b""
        after = current.decode("utf-8", errors="replace")
        diff = "".join(difflib.unified_diff(
            before.splitlines(keepends=True), after.splitlines(keepends=True),
            fromfile=f"a/{relative}", tofile=f"b/{relative}", n=3,
        ))
        added = sum(1 for line in diff.splitlines()
                    if line.startswith("+") and not line.startswith("+++"))
        removed = sum(1 for line in diff.splitlines()
                      if line.startswith("-") and not line.startswith("---"))
        change["edits"] += 1
        # The stored diff is cumulative, so the line counts must describe that diff too.
        change["added"] = added
        change["removed"] = removed
        change["diff"] = diff[:MAX_DIFF_CHARS]
        change["truncated"] = len(diff) > MAX_DIFF_CHARS
        change["after_version"] = _version(current)
        with self._lock:
            self._stats_cache.pop(workspace_id, None)
            self._index_cache.pop(workspace_id, None)
        return {key: change[key] for key in
                ("path", "edits", "added", "removed", "diff", "truncated", "tracked")}

    def revert_session(self, session_id: str) -> dict:
        session = self.get_session(session_id, include_messages=True)
        if session.get("active_run_id"):
            raise ConflictError("Stop the active run before reverting its changes")
        restored = deleted = skipped = 0
        changes = session.get("changes")
        changes = changes if isinstance(changes, dict) else {}
        for relative, change in changes.items():
            if not isinstance(change, dict):
                continue
            if change.get("reverted"):
                continue
            try:
                path = self._workspace_file(session["workspace_id"], relative, for_write=True)
                if change.get("tracked") is False:
                    skipped += 1
                    change["reverted"] = True
                    continue
                current = path.read_bytes() if path.is_file() else b""
                if (change.get("after_version")
                        and _version(current) != change["after_version"]):
                    # A person (or another session) edited the file after the agent;
                    # their work wins over a bulk revert. Mark it handled so a second
                    # revert does not keep reporting the same skip.
                    skipped += 1
                    change["reverted"] = True
                    continue
                if change.get("snapshot"):
                    snapshot = self._snapshot_path(session_id, change["snapshot"])
                    if snapshot is None or not snapshot.is_file():
                        skipped += 1
                        change["reverted"] = True
                        continue
                    _atomic_write(path, snapshot.read_bytes())
                    restored += 1
                elif not change.get("existed"):
                    if path.exists():
                        path.unlink()
                        deleted += 1
                change["reverted"] = True
            except (BadRequestError, NotFoundError, OSError):
                skipped += 1
        session["reverted"] = True
        session["status"] = "reverted"
        self._save_session(session)
        self._touch(session["workspace_id"])
        return {"restored": restored, "deleted": deleted, "skipped": skipped}

    # -- persistence -----------------------------------------------------
    def _forget_sessions(self, session_ids: set[str]) -> None:
        """Drop in-memory and saved runs that belong to deleted sessions."""
        if not session_ids:
            return
        with self._lock:
            for run_id, run in list(self._runs.items()):
                if run.session_id in session_ids:
                    self._runs.pop(run_id, None)
            for session_id in session_ids:
                self._session_cache.pop(self.sessions_dir / f"{session_id}.json", None)
        for path in self.runs_dir.glob("ide-run-*.json"):
            data = read_json(path)
            if isinstance(data, dict) and data.get("session_id") in session_ids:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    continue

    def _save_run(self, run: IdeRun) -> None:
        session_file = self._session_path(run.session_id)
        if not session_file.is_file():
            return  # the session was deleted while this run was finishing
        snapshot = run.snapshot(0)
        snapshot["events"] = snapshot.get("events", [])[-800:]
        try:
            write_json(self.runs_dir / f"{run.id}.json", snapshot, indent=None)
            files = sorted(self.runs_dir.glob("ide-run-*.json"), key=_modified, reverse=True)
            for old in files[200:]:
                old.unlink(missing_ok=True)
        except OSError:
            pass
        finally:
            if not session_file.is_file():
                try:
                    (self.runs_dir / f"{run.id}.json").unlink(missing_ok=True)
                except OSError:
                    pass

    def close(self) -> None:
        self._closed = True
        with self._lock:
            runs = list(self._runs.values())
            threads = list(self._threads.values())
        for run in runs:
            if run.status not in FINISHED:
                run.cancel()
        for thread in threads:
            thread.join(timeout=2.0)
