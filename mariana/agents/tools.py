"""Granted abilities for agent flows.

Abilities are explicitly listed on the agent document.  This module enforces
the sandbox and the dry-run contract so a workflow can be tested before it is
allowed to touch files, the network, or the shell.
"""

from __future__ import annotations

import base64
import csv
import difflib
import http.client
import io
import ipaddress
import json
import os
import re
import shutil
import signal
import socket
import sqlite3
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from pathlib import Path

from .aml import ABILITY_TYPES, evaluate_expression
from .hotcache import HOT_CACHE, read_cached
from ..workspace import ensure_workspace

PRIVATE_BLOCKLIST = {"169.254.169.254", "metadata.google.internal"}
BROWSER_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36 Mariana-Agent/1.0"
)
MAX_HTTP_BYTES = 1_000_000
MAX_FILE_BYTES = 2_000_000
MAX_OUTPUT_CHARS = 64_000
DEFAULT_TEXT_CHARS = 20_000
MAX_SEARCH_RESULTS = 500
# A real project is larger than the old 2,000-file ceiling (Shotcut is ~3.5k
# files).  Git grep handles big trees in milliseconds; the Python fallback is
# bounded by this count and the hot cache makes repeated scans cheap.
MAX_SEARCH_FILES = 50_000
MAX_SEARCH_FILE_BYTES = 2_000_000
MAX_TREE_ENTRIES = 20_000
MAX_LIST_ENTRIES = 1_000
MAX_READ_SCAN_BYTES = 64 * 1024 * 1024
MAX_READ_STREAM_BYTES = 8 * 1024 * 1024
READ_CHUNK_LINES = 200
READ_CHUNK_CHARS = 3_600
MAX_DIAGNOSTIC_FILES = 200
MAX_DIAGNOSTIC_ISSUES = 1000
IGNORED_DIRS = frozenset({
    ".git", ".hg", ".svn", "__pycache__", "node_modules", ".venv", "venv", "env",
    "dist", "build", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".idea", ".vscode",
})
_SKIPPED_TAGS = frozenset({"script", "style", "noscript", "svg", "head", "template", "iframe"})
_BLOCK_TAGS = frozenset({
    "p", "div", "br", "li", "ul", "ol", "tr", "table", "section", "article", "header", "footer",
    "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "pre", "hr", "nav", "main", "aside",
})


class ToolError(ValueError):
    pass


# Tools that change state.  They are never retried automatically: a timeout after
# a write may still have applied the change, and repeating it could duplicate it.
WRITE_ABILITY_TYPES = frozenset({"file.write", "file.edit", "file.delete", "shell.exec"})
TOOL_ATTEMPTS = 2
_RETRYABLE_MARKERS = (
    "timed out", "timeout", "temporar", "try again", "rate limit", "too many requests",
    "connection reset", "connection refused", "connection aborted", "broken pipe",
    "network is unreachable", "name or service not known", "eof occurred",
    "service unavailable", "bad gateway", "gateway timeout", "429", "502", "503", "504",
)


def retryable_tool_failure(error) -> bool:
    """Whether a read-only tool failure is worth one automatic retry.

    Deterministic failures (missing path, sandbox denial, unknown tool, invalid
    input) are returned to the model immediately so it can correct the call.
    """
    text = str(error or "").lower()
    return bool(text) and any(marker in text for marker in _RETRYABLE_MARKERS)


def tool_failure_hint(tool_id: str, error) -> str:
    """A short next action the model can use instead of repeating a failed call."""
    text = str(error or "").lower()
    if not text:
        return ""
    if "not found" in text or "no such file" in text or "does not exist" in text:
        return "List the directory or project tree, then retry with an exact existing path."
    if "outside" in text or "denied" in text or "permission" in text or "read-only" in text:
        return "Choose a path or operation inside the granted workspace and scope."
    if "unknown" in text and "tool" in text:
        return "Use one of the available tool ids exactly as listed."
    if "needs" in text or "required" in text or "invalid" in text or "must be" in text:
        return "Provide the required input keys, then retry the call."
    if retryable_tool_failure(text):
        return "The endpoint did not answer reliably; retry once, or use another source."
    return "Change the input or approach instead of repeating the same call."


def nearest_tool_id(value: str, known) -> str:
    """The closest available tool id, for an actionable unknown-tool message."""
    import difflib

    wanted = str(value or "").strip().lower()
    candidates = [str(item) for item in known]
    if not wanted or not candidates:
        return ""
    by_key = {re.sub(r"[^a-z0-9]", "", item.lower()): item for item in candidates}
    key = re.sub(r"[^a-z0-9]", "", wanted)
    if key in by_key:
        return by_key[key]
    matches = difflib.get_close_matches(key, list(by_key), n=1, cutoff=0.62)
    return by_key[matches[0]] if matches else ""


def resolve_tool_id(value: str, known) -> str:
    """An exact tool id, or one that differs only in punctuation and case.

    Models sometimes spell ``file.search`` as ``file_search`` or ``fileSearch``.
    Correcting that is safe because the normalised keys are identical; anything
    less certain is left untouched so the model gets a suggestion instead of an
    unintended tool running.
    """
    wanted = str(value or "").strip()
    candidates = [str(item) for item in known]
    if wanted in candidates:
        return wanted
    key = re.sub(r"[^a-z0-9]", "", wanted.lower())
    if not key:
        return ""
    for candidate in candidates:
        if re.sub(r"[^a-z0-9]", "", candidate.lower()) == key:
            return candidate
    return ""


_PATH_ALIASES = frozenset({"path", "file", "filename", "filepath", "file_path", "target",
                           "location", "dir", "folder"})

# The argument spellings models commonly invent, mapped onto the keys the
# abilities actually read.  Keeping this table next to the executor means the
# IDE, the flow runtime and the swarm all accept the same calls.
_TOOL_ALIASES = {
    "http.fetch": {
        "url": {"url", "uri", "link", "href", "endpoint", "address"},
        "method": {"method", "verb"},
        "headers": {"headers", "header"},
        "body": {"body", "payload", "data"},
        "extract": {"extract", "format", "as"},
        "max_chars": {"maxchars", "max_chars", "maxlength", "max_length"},
        "timeout": {"timeout", "timeout_seconds", "seconds"},
    },
    "news.headlines": {
        "url": {"url", "feed", "feed_url"},
        "query": {"query", "q", "topic", "search", "term", "keywords"},
        "source": {"source", "provider"},
        "region": {"region", "country", "locale"},
        "language": {"language", "lang"},
        "limit": {"limit", "count", "max", "max_results", "maxresults"},
        "date": {"date", "day", "published"},
        "include_summary": {"include_summary", "includesummary", "summary", "summaries"},
        "timeout": {"timeout", "timeout_seconds", "seconds"},
    },
    "browser.fetch": {
        "url": {"url", "uri", "link", "href", "endpoint", "address"},
        "timeout": {"timeout", "timeout_seconds", "seconds"},
        "max_chars": {"maxchars", "max_chars", "maxlength", "max_length"},
    },
    "file.read": {
        "path": _PATH_ALIASES,
        "start_line": {"start", "startline", "start_line", "from", "firstline",
                       "first_line", "begin", "offset"},
        "end_line": {"end", "endline", "end_line", "to", "lastline", "last_line", "stop"},
        "limit": {"limit", "count", "lines", "num_lines"},
    },
    "file.write": {
        "path": _PATH_ALIASES,
        "content": {"content", "text", "body", "data", "value", "contents", "file_content"},
    },
    "file.edit": {
        "path": _PATH_ALIASES,
        "old": {"old", "oldstring", "old_string", "oldtext", "old_text", "find",
                "search", "before"},
        "new": {"new", "newstring", "new_string", "newtext", "new_text",
                "replace", "replacement", "after", "with"},
        "replace_all": {"replaceall", "replace_all", "all"},
    },
    "file.delete": {"path": _PATH_ALIASES},
    "file.list": {
        "path": _PATH_ALIASES,
        "pattern": {"pattern", "glob", "include", "filter", "file_pattern"},
        "recursive": {"recursive", "recurse", "deep"},
        "max_entries": {"maxentries", "max_entries", "limit", "count", "max"},
    },
    "file.tree": {
        "path": _PATH_ALIASES,
        "max_entries": {"maxentries", "max_entries", "limit", "count", "max"},
        "depth": {"depth", "max_depth", "maxdepth", "levels"},
    },
    "file.search": {
        "query": {"query", "q", "search", "term", "needle", "text", "expression"},
        "include": {"include", "include_pattern", "includepattern", "glob", "pattern",
                    "filepattern", "file_pattern", "path_glob", "filter"},
        "max_results": {"maxresults", "maxresult", "max_results", "max", "limit", "count",
                        "take"},
    },
    "code.diagnostics": {"path": _PATH_ALIASES},
    "git.inspect": {
        "path": _PATH_ALIASES,
        "action": {"action", "command", "op", "mode", "subcommand"},
        "limit": {"limit", "count", "max", "number"},
        "max_chars": {"maxchars", "max_chars", "maxlength", "max_length"},
    },
    "csv.read": {
        "path": _PATH_ALIASES,
        "delimiter": {"delimiter", "sep", "separator"},
        "header": {"header", "headers", "has_header"},
        "max_rows": {"maxrows", "max_rows", "limit", "rows"},
    },
    "sqlite.query": {
        "path": _PATH_ALIASES,
        "sql": {"sql", "query", "statement", "sql_query"},
        "params": {"params", "parameters", "bind", "values"},
        "max_rows": {"maxrows", "max_rows", "limit", "rows"},
        "timeout": {"timeout", "timeout_seconds", "seconds"},
    },
    "shell.exec": {
        "command": {"command", "cmd", "shell", "script", "line"},
        "timeout": {"timeout", "timeout_seconds", "seconds"},
    },
    "math.eval": {"expression": {"expression", "expr", "formula", "math", "input"}},
    "memory": {
        "action": {"action", "op", "operation"},
        "key": {"key", "name", "id"},
        "value": {"value", "content", "text", "data"},
    },
    "skill.read": {"name": {"name", "id", "skill", "skill_id", "skillid"}},
    "file.inspect": {"path": _PATH_ALIASES},
}

# The key a bare string argument means for each tool ({"tool": "file.read",
# "input": "src/main.py"} is a common model shortcut).
_TOOL_PRIMARY_ARGUMENT = {
    "http.fetch": "url", "news.headlines": "query", "browser.fetch": "url",
    "file.read": "path", "file.write": "path", "file.edit": "path",
    "file.delete": "path", "file.list": "path", "file.tree": "path",
    "file.search": "query", "code.diagnostics": "path", "git.inspect": "path",
    "csv.read": "path", "sqlite.query": "sql", "shell.exec": "command",
    "math.eval": "expression", "skill.read": "name", "file.inspect": "path",
}

# Keys every call to the tool needs before it can do useful work.  Catching a
# missing one here produces a precise correction prompt instead of a failed
# execution that the model has to diagnose from an error string.
_REQUIRED_TOOL_INPUTS = {
    "http.fetch": ("url",), "browser.fetch": ("url",),
    "file.read": ("path",), "file.write": ("path", "content"),
    "file.edit": ("path", "old", "new"), "file.delete": ("path",),
    "file.search": ("query",), "sqlite.query": ("sql",),
    "shell.exec": ("command",), "math.eval": ("expression",),
    "csv.read": ("path",), "skill.read": ("name",), "file.inspect": ("path",),
}

_TOOL_INT_ARGUMENTS = frozenset({
    "start_line", "end_line", "limit", "max_results", "max_entries", "depth",
    "timeout", "max_chars", "max_rows", "max_agents", "max_steps", "max_seconds",
    "rounds", "max_tokens", "rows", "count",
})
_TOOL_FLAG_ARGUMENTS = frozenset({
    "regex", "case_sensitive", "recursive", "replace_all", "staged", "header",
    "include_summary", "write", "memory", "ipc", "translate",
})
_TOOL_LIST_ARGUMENTS = frozenset({"tasks", "agents", "tools", "writes", "headers", "params"})
_STRIPPED_STRING_ARGUMENTS = frozenset({
    "path", "query", "include", "pattern", "command", "url", "sql", "expression",
    "name", "old", "new", "action",
})


def normalise_tool_arguments(tool_id: str, arguments) -> dict:
    """Map the argument spellings models use onto the ability's real keys.

    A missing ``file_path`` or a camelCase ``startLine`` is a model habit, not a
    tool failure.  Normalising here also lets the read cache deduplicate calls
    that were semantically identical but spelled differently, and makes the IDE,
    the flow runtime and the swarm behave the same way.
    """
    if isinstance(arguments, str):
        text = arguments.strip()
        try:
            parsed = json.loads(text) if text[:1] in {"{", "["} else None
        except (TypeError, ValueError):
            parsed = None
        if isinstance(parsed, dict):
            arguments = parsed
        elif isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], dict):
            arguments = parsed[0]
        else:
            primary = _TOOL_PRIMARY_ARGUMENT.get(tool_id)
            arguments = {primary: arguments} if primary else {"input": arguments}
    if not isinstance(arguments, dict):
        arguments = {"input": arguments}
    result = dict(arguments)
    key_lookup = {re.sub(r"[-_\s]", "", str(key)).lower(): key for key in result}
    for canonical, spellings in _TOOL_ALIASES.get(tool_id, {}).items():
        if canonical in result:
            continue
        for spelling in spellings:
            original = key_lookup.get(re.sub(r"[-_\s]", "", str(spelling)).lower())
            if original is not None and original in result:
                result[canonical] = result.pop(original)
                break
    primary = _TOOL_PRIMARY_ARGUMENT.get(tool_id)
    if primary and primary not in result:
        # Many models send a bare string as the only argument.
        for key in ("input", "file", "value"):
            value = result.get(key)
            if isinstance(value, str) and value.strip():
                result[primary] = value
                result.pop(key, None)
                break
    for key in list(result):
        value = result[key]
        if isinstance(value, str) and key in _STRIPPED_STRING_ARGUMENTS:
            stripped = value.strip()
            if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in "`'\"":
                stripped = stripped[1:-1]
            result[key] = stripped
        if key in {"path", "include", "pattern"} and isinstance(result[key], str):
            result[key] = result[key].replace("\\", "/")
    for key in _TOOL_LIST_ARGUMENTS:
        value = result.get(key)
        if isinstance(value, str) and value.strip()[:1] in {"[", "{"}:
            try:
                result[key] = json.loads(value)
            except (TypeError, ValueError):
                pass
    for key in _TOOL_INT_ARGUMENTS:
        if key in result and not isinstance(result[key], (int, float)):
            try:
                result[key] = int(float(str(result[key]).strip()))
            except (TypeError, ValueError):
                pass
    for key in _TOOL_FLAG_ARGUMENTS:
        if key in result and not isinstance(result[key], bool):
            text = str(result[key]).strip().lower()
            if text in {"1", "true", "yes", "on"}:
                result[key] = True
            elif text in {"0", "false", "no", "off"}:
                result[key] = False
    if tool_id == "file.read" and "limit" in result and "end_line" not in result:
        start = result.get("start_line")
        count = result.get("limit")
        if isinstance(start, int) and isinstance(count, int) and count > 0:
            result["end_line"] = start + count - 1
            result.pop("limit", None)
    return result


def missing_required_arguments(tool_id: str, arguments) -> list[str]:
    """Required input keys the call omitted or left blank."""
    required = _REQUIRED_TOOL_INPUTS.get(tool_id, ())
    if not required:
        return []
    values = arguments if isinstance(arguments, dict) else {}
    missing = []
    for key in required:
        value = values.get(key)
        if value is None or (isinstance(value, str) and not value.strip()):
            missing.append(key)
    return missing


def required_arguments(tool_id: str) -> tuple[str, ...]:
    """The keys a model must provide for this tool (for correction prompts)."""
    return _REQUIRED_TOOL_INPUTS.get(tool_id, ())


def describe_abilities() -> dict:
    return ABILITY_TYPES


class _ReadableText(HTMLParser):
    """Collect the visible text of an HTML document, one block per line."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skipping = 0

    def handle_starttag(self, tag, attrs):
        if tag in _SKIPPED_TAGS:
            self.skipping += 1
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in _SKIPPED_TAGS:
            self.skipping = max(0, self.skipping - 1)
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.skipping:
            self.parts.append(data)


def html_to_text(document: str, max_chars: int = DEFAULT_TEXT_CHARS) -> str:
    """Readable text of an HTML page: no scripts or styles, whitespace collapsed."""
    parser = _ReadableText()
    try:
        parser.feed(document)
        parser.close()
    except Exception:  # noqa: BLE001 - malformed markup still yields the text seen so far
        pass
    lines = (" ".join(line.split()) for line in "".join(parser.parts).splitlines())
    text = "\n".join(line for line in lines if line)
    return text if len(text) <= max_chars else text[:max_chars].rstrip() + "…"


def _feed_local_name(tag) -> str:
    """An XML tag without its namespace, lowercased."""
    return str(tag or "").rsplit("}", 1)[-1].lower()


def _feed_text(value, limit: int = 600) -> str:
    """Readable, whitespace-collapsed feed text (descriptions often contain HTML)."""
    text = html_to_text(str(value or ""), limit)
    return " ".join(text.split())[:limit]


def _feed_date(value) -> str:
    """A feed timestamp as ``YYYY-MM-DD``, for filtering and grouping."""
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        parsed = parsedate_to_datetime(text)
        if parsed is not None:
            return parsed.date().isoformat()
    except (TypeError, ValueError, OverflowError):
        pass
    match = re.search(r"\d{4}-\d{2}-\d{2}", text)
    return match.group(0) if match else ""


def _feed_child(node, names: tuple[str, ...], limit: int = 600) -> str:
    """The first non-empty child whose local name is in ``names``."""
    for child in list(node):
        if _feed_local_name(child.tag) not in names:
            continue
        value = child.text or child.get("href") or ""
        if str(value).strip():
            return _feed_text(value, limit)
    return ""


def _feed_items(root, include_summary: bool = False) -> list[dict]:
    """RSS ``item`` and Atom ``entry`` elements as compact headline records."""
    items = []
    for node in root.iter():
        if _feed_local_name(node.tag) not in {"item", "entry"}:
            continue
        title = _feed_child(node, ("title",), 300)
        link = _feed_child(node, ("link", "id"), 1_000)
        source = _feed_child(node, ("source",), 120)
        published = _feed_child(node, ("pubdate", "published", "updated", "date"), 120)
        if not title and not link:
            continue
        item = {"title": title, "link": link, "source": source,
                "published": published, "date": _feed_date(published)}
        if include_summary:
            item["summary"] = _feed_child(node, ("description", "summary", "content"), 400)
        items.append(item)
    return items


def _feed_title(root) -> str:
    """The channel or feed title, if the document names one."""
    for node in root.iter():
        if _feed_local_name(node.tag) not in {"channel", "feed"}:
            continue
        title = _feed_child(node, ("title",), 200)
        if title:
            return title
    return ""


def execute_ability(ability: dict, node_input, context: dict, root: Path,
                    dry_run: bool = False, files_root: Path | None = None,
                    cancelled=None, deny: list[str | Path] | None = None,
                    os_access: bool = False) -> dict:
    """Run one granted ability.

    ``root`` is the Studio root; files, memory and the shell are confined to
    ``<root>/agent-workspace``.  ``files_root`` overrides the file sandbox for
    callers that own a project folder (the IDE workspace), so the same ability
    definitions and safety checks serve both AML flows and the coding agent.
    ``deny`` names subtrees of an in-place project the abilities must never
    touch (Studio's private runtime folder).  ``cancelled`` is an optional
    event that long-running abilities poll.  ``os_access`` is a server-level
    opt-in that lets ``shell.exec`` run and lets HTTP reach private addresses;
    it is never read from model input or ability configuration.
    """
    ability_type = ability.get("type")
    if ability_type not in ABILITY_TYPES:
        raise ToolError(f"Unknown ability type: {ability_type}")
    if ability.get("enabled") is False:
        raise ToolError(f"Ability {ability.get('id')} is disabled")
    params = dict(ability.get("config") or {})
    if isinstance(node_input, dict):
        params.update(node_input)
    elif node_input not in (None, ""):
        if ability_type == "news.headlines" and node_input not in (None, ""):
            params.setdefault("query", node_input)
        else:
            params["input"] = node_input
            if ability_type in {"http.fetch", "browser.fetch", "file.read", "file.write",
                                "file.edit", "file.delete", "shell.exec", "math.eval",
                                "sqlite.query", "csv.read", "git.inspect"}:
                params.setdefault(
                    {"http.fetch": "url", "browser.fetch": "url", "file.read": "path",
                     "file.write": "path", "file.edit": "path", "file.delete": "path",
                     "shell.exec": "command", "math.eval": "expression",
                     "sqlite.query": "sql", "csv.read": "path", "git.inspect": "path"}
                    [ability_type],
                    node_input,
                )
    if ability_type in {"http.fetch", "news.headlines", "browser.fetch"}:
        # Network reach is an author decision, never something a model can widen per call.
        params["allow_private"] = (
            _truthy((ability.get("config") or {}).get("allow_private"), False) or os_access
        )
    if ability_type == "sqlite.query":
        # A model cannot turn a read-only database ability into a write one.
        params["write"] = _truthy((ability.get("config") or {}).get("write"), False)
    runner = _AbilityRunner(root, dry_run, files_root=files_root, cancelled=cancelled,
                            deny=deny, os_access=os_access)
    result = getattr(runner, ability_type.replace(".", "_"))(params, context)
    result.setdefault("ability", ability.get("id"))
    result.setdefault("type", ability_type)
    return result


def _truthy(value, default: bool = False) -> bool:
    """Accept JSON booleans and the usual string spellings from a model."""
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


_SHELL_READ_ONLY = frozenset({
    "ls", "cat", "head", "tail", "grep", "egrep", "fgrep", "rg", "find", "wc",
    "pwd", "tree", "stat", "file", "du", "df", "which", "whereis", "echo",
})
_GIT_READ_ONLY = frozenset({
    "status", "diff", "log", "show", "grep", "ls-files", "rev-parse", "blame",
    "describe", "shortlog", "branch", "remote", "tag",
})


def _shell_may_write(command: str) -> bool:
    """Whether a shell command may change files and must invalidate the cache."""
    parts = str(command or "").strip().split()
    if not parts:
        return True
    head = os.path.basename(parts[0])
    if head in _SHELL_READ_ONLY:
        return False
    if head == "git":
        return len(parts) < 2 or parts[1] not in _GIT_READ_ONLY
    return True


def _text_bytes(path: Path, max_bytes: int = MAX_SEARCH_FILE_BYTES) -> bytes | None:
    """Read a small textual file through the process-wide hot cache."""
    data, _ = read_cached(
        path, max_bytes, accept=lambda value: b"\x00" not in value[:4096])
    return data


def _write_text_atomic(path: Path, text: str) -> None:
    """A crash leaves either the old file or the complete new one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        temporary.write_bytes(text.encode("utf-8"))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    HOT_CACHE.invalidate(str(path))


def _pyflakes_issues(text: str, relative: str) -> list[dict]:
    """Optional static checks: a missing pyflakes never breaks diagnostics."""
    try:
        from pyflakes import api as pyflakes_api
    except ImportError:
        return []

    class Collector:
        def __init__(self):
            self.issues: list[dict] = []

        def flake(self, message):
            try:
                detail = message.message % message.message_args
            except (TypeError, ValueError):
                detail = str(message.message)
            self.issues.append({
                "path": relative, "line": int(message.lineno or 1),
                "column": int(getattr(message, "col", 1) or 1),
                "message": detail, "rule": str(message.code or "pyflakes"),
                "severity": "warning",
            })

        def unexpectedError(self, filename, message):  # noqa: N802 - pyflakes API
            self.issues.append({"path": relative, "line": 1, "column": 1,
                                "message": str(message), "rule": "pyflakes",
                                "severity": "error"})

        def syntaxError(self, filename, message, lineno, column, source):  # noqa: N802
            self.issues.append({"path": relative, "line": int(lineno or 1),
                                "column": int(column or 1), "message": str(message),
                                "rule": "syntax", "severity": "error"})

    reporter = Collector()
    try:
        pyflakes_api.check(text, relative, reporter)
    except Exception:  # noqa: BLE001 - diagnostics must never break a run
        pass
    return reporter.issues


def _private_address(address: str) -> bool:
    """True for loopback, private, link-local, metadata and other non-public IPs."""
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return True
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return not ip.is_global


class _GuardedRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Validate every redirect target, not just the URL the flow supplied."""

    def __init__(self, validate):
        super().__init__()
        self._validate = validate

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        self._validate(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class _GuardedHTTPHandler(urllib.request.HTTPHandler):
    def __init__(self, connection):
        super().__init__()
        self._connection = connection

    def http_open(self, req):
        return self.do_open(self._connection, req)


class _GuardedHTTPSHandler(urllib.request.HTTPSHandler):
    def __init__(self, connection):
        super().__init__()
        self._connection = connection

    def https_open(self, req):
        return self.do_open(self._connection, req, context=self._context)


def _guarded_connection(base, allow_private: bool):
    """A connection class that re-checks the peer address after DNS resolution."""

    class Guarded(base):
        def connect(self):
            super().connect()
            if allow_private:
                return
            try:
                address = self.sock.getpeername()[0]
            except (AttributeError, OSError):
                return
            if _private_address(address):
                self.close()
                raise ToolError(
                    "Private addresses are blocked; set allow_private on this ability to opt in"
                )

    Guarded.__name__ = f"Guarded{base.__name__}"
    return Guarded


class _AbilityRunner:
    def __init__(self, root: Path, dry_run: bool, files_root: Path | None = None,
                 cancelled=None, deny: list[str | Path] | None = None,
                 os_access: bool = False):
        self.root = ensure_workspace(root)  # <root>/agent-workspace
        self.files = (Path(files_root).resolve() if files_root
                      else self.root / "files")
        self.files.mkdir(parents=True, exist_ok=True)
        self.deny = [Path(item).expanduser().resolve() for item in (deny or [])]
        self.dry_run = dry_run
        self.cancelled = cancelled
        self.os_access = bool(os_access)

    def _denied(self, path: Path) -> bool:
        for private in self.deny:
            try:
                path.relative_to(private)
                return True
            except ValueError:
                continue
        return False

    @staticmethod
    def _clean_path(value) -> str:
        """A model-supplied path without quotes, ``./`` noise or Windows slashes."""
        text = str(value or "").strip().strip("`'\"")
        if text.startswith("file://"):
            text = urllib.parse.unquote(text[7:])
        text = text.replace("\\", "/")
        while text.startswith("./"):
            text = text[2:]
        return text

    def _suggest_path(self, value: str, limit: int = 3) -> list[str]:
        """Close existing paths, so a missing file does not become a dead end."""
        target = self._clean_path(value)
        if not target:
            return []
        candidates = []
        try:
            for path in self._walk_files():
                try:
                    candidates.append(path.relative_to(self.files).as_posix())
                except ValueError:
                    continue
                if len(candidates) >= 5_000:
                    break
        except OSError:
            pass
        close = difflib.get_close_matches(target, candidates, n=limit, cutoff=0.55)
        if close:
            return close
        base = Path(target).name
        return [item for item in candidates if Path(item).name == base][:limit]

    def _missing_path_message(self, path: Path) -> str:
        relative = self._relative(path)
        suggestions = self._suggest_path(relative)
        nearby = []
        parent = path.parent
        if parent.is_dir():
            try:
                nearby = [item.name + ("/" if item.is_dir() else "")
                          for item in sorted(parent.iterdir())[:12]
                          if not item.name.startswith(".")]
            except OSError:
                nearby = []
        message = f"File not found: {relative}."
        if suggestions:
            message += " Did you mean: " + ", ".join(suggestions) + "?"
        if nearby:
            message += " Nearby: " + ", ".join(nearby) + "."
        if not suggestions and not nearby:
            message += " Use file.tree or file.list to find the right path."
        return message

    def _directory_listing(self, path: Path, limit: int = 200) -> dict:
        entries = []
        truncated = False
        try:
            children = sorted(path.iterdir(),
                              key=lambda item: (not item.is_dir(), item.name.lower()))
        except OSError as error:
            raise ToolError(f"Cannot list {self._relative(path)}: {error}") from error
        for child in children:
            if len(entries) >= limit:
                truncated = True
                break
            if child.name == ".mariana-ide.json":
                continue
            try:
                resolved = child.resolve()
                resolved.relative_to(self.files)
            except (OSError, ValueError):
                continue
            if self._denied(resolved):
                continue
            if resolved.is_dir():
                entries.append({"path": self._relative(child) + "/", "type": "dir"})
            elif resolved.is_file():
                try:
                    entries.append({"path": self._relative(child), "type": "file",
                                    "bytes": resolved.stat().st_size})
                except OSError:
                    continue
        return {"path": self._relative(path), "directory": True, "entries": entries,
                "truncated": truncated,
                "hint": "This path is a directory. Use file.tree or pass a file path."}

    def _relative(self, path: Path) -> str:
        try:
            return path.relative_to(self.files).as_posix()
        except ValueError:
            return str(path)

    def _sandbox_path(self, value: str, for_write: bool = False) -> Path:
        raw = self._clean_path(value)
        if not raw:
            raise ToolError("A file path is required")
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = self.files / candidate
            # Models often prefix paths with the project folder name; accept that
            # spelling when the direct path does not exist.
            prefix = f"{self.files.name}/"
            if not candidate.exists() and raw.startswith(prefix):
                alternative = self.files / raw[len(prefix):]
                if alternative.exists() or (for_write and alternative.parent.exists()):
                    candidate = alternative
        try:
            candidate = candidate.resolve(strict=False)
        except OSError as error:
            raise ToolError(f"Cannot resolve {raw}: {error}") from error
        try:
            candidate.relative_to(self.files)
        except ValueError:
            raise ToolError(
                "File abilities are restricted to paths inside the file sandbox; "
                f"{raw} resolves outside {self.files}"
            ) from None
        if self._denied(candidate):
            raise ToolError("This folder is private to Mariana Studio")
        if for_write and not self.dry_run:
            candidate.parent.mkdir(parents=True, exist_ok=True)
        return candidate

    def _walk_paths(self):
        """Paths under the sandbox, pruning ignored folders and denied subtrees."""
        for folder, directories, names in os.walk(self.files):
            base = Path(folder)
            directories[:] = sorted(
                name for name in directories
                if name not in IGNORED_DIRS
                and not (base / name).is_symlink()
                and not self._denied(base / name))
            for name in sorted(names):
                yield base / name

    def _check_cancelled(self) -> None:
        if self.cancelled is not None and self.cancelled.is_set():
            raise ToolError("Ability cancelled")

    def http_fetch(self, params: dict, context: dict) -> dict:
        url = str(params.get("url") or params.get("input") or "").strip()
        if not url:
            raise ToolError("http.fetch needs a url")
        self._check_cancelled()
        method = str(params.get("method") or "GET").upper()
        if method not in {"GET", "POST", "PUT", "PATCH", "DELETE"}:
            raise ToolError("Unsupported HTTP method")
        headers = params.get("headers") if isinstance(params.get("headers"), dict) else {}
        headers = {str(k): str(v) for k, v in headers.items()}
        # A browser-like identity avoids needless 403s from public publishers while
        # still keeping the caller's explicit headers authoritative.
        headers.setdefault("User-Agent", BROWSER_USER_AGENT)
        headers.setdefault("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8")
        headers.setdefault("Accept-Language", "en-US,en;q=0.9")
        body = params.get("body") or params.get("data")
        timeout = min(30.0, max(1.0, float(params.get("timeout", 15))))
        allow_private = _truthy(params.get("allow_private"), False)
        self._validate_public_url(url, allow_private, resolve=not self.dry_run)
        if self.dry_run:
            extract_text = str(params.get("extract") or "raw").lower() == "text"
            result = {
                "dry_run": True, "ok": True, "status": 200, "url": url, "method": method,
                "headers": {"Content-Type": "text/plain" if extract_text else "application/json"},
                "note": "HTTP request skipped in test mode",
            }
            if extract_text:
                result["text"] = f"[test mode] HTTP response from {url}"
            else:
                mock = json.dumps({"dry_run": True, "url": url})
                result["body"] = mock
                result["json"] = {"dry_run": True, "url": url}
            return result
        payload = None
        if body is not None:
            if isinstance(body, (dict, list)):
                payload = json.dumps(body).encode("utf-8")
                headers.setdefault("Content-Type", "application/json")
            else:
                payload = str(body).encode("utf-8")
        request = urllib.request.Request(url, data=payload, method=method, headers=headers)
        validate = lambda target: self._validate_public_url(target, allow_private)  # noqa: E731
        opener = urllib.request.build_opener(
            _GuardedRedirectHandler(validate),
            _GuardedHTTPHandler(_guarded_connection(http.client.HTTPConnection, allow_private)),
            _GuardedHTTPSHandler(_guarded_connection(http.client.HTTPSConnection, allow_private)),
        )
        try:
            with opener.open(request, timeout=timeout) as response:
                raw = response.read(MAX_HTTP_BYTES + 1)
                status = response.status
                response_headers = dict(response.headers.items())
                content_type = response.headers.get("Content-Type", "")
        except urllib.error.HTTPError as error:
            body_text = error.read(MAX_HTTP_BYTES).decode("utf-8", errors="replace")
            return {
                "ok": False, "status": error.code, "url": url,
                "body": body_text[:MAX_OUTPUT_CHARS], "error": f"HTTP {error.code}",
            }
        except urllib.error.URLError as error:
            raise ToolError(f"HTTP request failed: {error.reason}") from error
        except (TimeoutError, socket.timeout) as error:
            raise ToolError("HTTP request timed out") from error
        truncated = len(raw) > MAX_HTTP_BYTES
        text = raw[:MAX_HTTP_BYTES].decode("utf-8", errors="replace")
        result = {
            "ok": 200 <= status < 400, "status": status, "url": url,
            "headers": response_headers, "body": text, "truncated": truncated,
        }
        if "json" in content_type.lower():
            try:
                result["json"] = json.loads(text)
            except json.JSONDecodeError:
                pass
        if str(params.get("extract") or "raw").lower() == "text":
            # Readable text instead of markup: raw HTML wastes a model's context window.
            try:
                limit = max(100, min(200_000, int(params.get("max_chars") or DEFAULT_TEXT_CHARS)))
            except (TypeError, ValueError):
                limit = DEFAULT_TEXT_CHARS
            markup = "html" in content_type.lower() or text.lstrip()[:15].lower().startswith(("<!doctype", "<html"))
            result["text"] = html_to_text(text, limit) if markup else text[:limit]
            result.pop("body")
            result["headers"] = {"Content-Type": content_type}
        return result

    def news_headlines(self, params: dict, context: dict) -> dict:
        """Read a public RSS/Atom feed as dated, linked headlines.

        Google News is the default source.  A caller can pass any public feed
        URL; a ``query`` filters that feed, and is also sent to Google News as
        its search term.  Only public addresses are reached unless the server
        is in OS-access mode or the ability explicitly opts in.
        """
        url = str(params.get("url") or "").strip()
        query = str(params.get("query") or "").strip()
        source = str(params.get("source") or "auto").strip().lower()
        requested_date = str(params.get("date") or "").strip()
        include_summary = _truthy(params.get("include_summary"), False)
        region = str(params.get("region") or "").strip().upper()
        language = str(params.get("language") or "en").strip().lower() or "en"
        try:
            limit = max(1, min(50, int(params.get("limit") or 10)))
        except (TypeError, ValueError):
            limit = 10
        try:
            timeout = min(30.0, max(1.0, float(params.get("timeout") or 15)))
        except (TypeError, ValueError):
            timeout = 15.0

        def google_url() -> str:
            base = "https://news.google.com/rss"
            separator = "?"
            if query:
                base = ("https://news.google.com/rss/search?q="
                        + urllib.parse.quote_plus(query))
                separator = "&"
            if region:
                code = language.split("-")[0]
                hl = language if "-" in language else f"{code}-{region}"
                return (f"{base}{separator}hl={urllib.parse.quote_plus(hl)}"
                        f"&gl={urllib.parse.quote_plus(region)}"
                        f"&ceid={urllib.parse.quote_plus(region)}:"
                        f"{urllib.parse.quote_plus(code)}")
            return base

        def bing_url() -> str:
            search = urllib.parse.quote_plus(query or "world news")
            market = f"&setmkt=en-{region}" if region else ""
            return f"https://www.bing.com/news/search?q={search}&format=RSS{market}"

        if url:
            candidates = [(url, "custom")]
            if source == "auto" and query:
                candidates.extend([(google_url(), "google"), (bing_url(), "bing")])
        elif source == "bing":
            if not query:
                raise ToolError("news.headlines with source=bing needs a query")
            candidates = [(bing_url(), "bing")]
        elif source == "google":
            candidates = [(google_url(), "google")]
        else:
            candidates = [(google_url(), "google")]
            if query:
                candidates.append((bing_url(), "bing"))
        if self.dry_run:
            return {"dry_run": True, "url": candidates[0][0],
                    "urls_tried": [candidate for candidate, _ in candidates],
                    "source": source, "query": query, "region": region,
                    "requested_date": requested_date, "date_matched": None,
                    "items": [], "count": 0, "total": 0, "truncated": False,
                    "note": "News feed skipped in test mode"}
        items: list[dict] = []
        feed_title = ""
        used_url = candidates[0][0]
        used_source = candidates[0][1]
        failures: list[str] = []
        for candidate_url, name in candidates:
            try:
                fetched = self.http_fetch({
                    "url": candidate_url, "method": "GET", "timeout": timeout,
                    "allow_private": _truthy(params.get("allow_private"), False),
                }, context)
                if fetched.get("ok") is False:
                    raise ToolError(str(fetched.get("error") or fetched.get("status")))
                raw = str(fetched.get("body") or "")
                if not raw.strip():
                    raise ToolError("empty body")
                try:
                    root = ET.fromstring(raw)
                except ET.ParseError as error:
                    raise ToolError("not valid RSS or Atom XML") from error
                parsed = _feed_items(root, include_summary=include_summary)
                if not any(_feed_local_name(node.tag) in {"rss", "feed", "channel", "rdf"}
                           for node in root.iter()):
                    raise ToolError("XML is not an RSS or Atom feed")
                if query and name == "custom":
                    words = [word for word in re.split(r"\W+", query.lower())
                             if len(word) >= 2]
                    if words:
                        parsed = [item for item in parsed
                                  if all(word in f"{item['title']} "
                                         f"{item.get('summary') or ''}".lower()
                                         for word in words)]
                        if not parsed:
                            failures.append("custom: no items matched the query")
                            continue
                if parsed:
                    items = parsed
                    feed_title = _feed_title(root)
                    used_url, used_source = candidate_url, name
                    break
                failures.append(f"{name}: no items")
            except ToolError as failure:
                failures.append(f"{name}: {failure}")
        if not items:
            raise ToolError("No news items returned; " + "; ".join(failures or ["empty feed"]))
        total = len(items)
        dates: dict[str, int] = {}
        for item in items:
            if item.get("date"):
                dates[item["date"]] = dates.get(item["date"], 0) + 1
        latest_date = max(dates, default="")
        date_matched = None
        note = ""
        if requested_date:
            matched = [item for item in items if item.get("date") == requested_date]
            date_matched = bool(matched)
            if matched:
                items = matched
            else:
                note = (f"No items dated {requested_date}; the newest available item is "
                        f"{latest_date or 'undated'}. Returned the newest items instead; "
                        "state the date mismatch in the answer.")
        available = len(items)
        items = items[:limit]
        return {
            "ok": True, "as_of": datetime.now(timezone.utc).date().isoformat(),
            "url": used_url, "urls_tried": [candidate for candidate, _ in candidates],
            "source_used": used_source, "feed_title": feed_title,
            "query": query, "region": region,
            "requested_date": requested_date, "date_matched": date_matched,
            "latest_date": latest_date, "dates": dict(sorted(dates.items(), reverse=True)[:10]),
            "items": items, "count": len(items), "total": total,
            "truncated": len(items) < available, "note": note,
            "fallbacks": failures,
        }

    def browser_fetch(self, params: dict, context: dict) -> dict:
        """Render a page in a headless browser and return its visible text.

        Playwright is used when installed; otherwise a Chromium binary is driven
        with ``--dump-dom``.  Launching a browser is an OS action, so this ability
        requires the server-wide OS-access opt-in.
        """
        if not self.os_access:
            raise ToolError(
                "A headless browser requires OS access. Enable allow_os_access in Models "
                "settings, or start Studio with MARIANA_AGENT_ALLOW_OS=1."
            )
        url = str(params.get("url") or params.get("input") or "").strip()
        if not url:
            raise ToolError("browser.fetch needs a url")
        self._check_cancelled()
        try:
            timeout = min(60.0, max(5.0, float(params.get("timeout") or 25)))
        except (TypeError, ValueError):
            timeout = 25.0
        try:
            max_chars = max(500, min(200_000, int(params.get("max_chars") or 20_000)))
        except (TypeError, ValueError):
            max_chars = 20_000
        allow_private = _truthy(params.get("allow_private"), False)
        self._validate_public_url(url, allow_private, resolve=not self.dry_run)
        if self.dry_run:
            return {"dry_run": True, "ok": True, "url": url, "title": "", "text": "",
                    "chars": 0, "truncated": False, "engine": "test",
                    "note": "Headless browser skipped in test mode"}
        title, text, engine, problem = "", "", "", ""
        sync_playwright = None
        try:
            from playwright.sync_api import sync_playwright as _sync_playwright
            sync_playwright = _sync_playwright
        except ImportError:
            pass
        if sync_playwright is not None:
            try:
                with sync_playwright() as runtime:
                    browser = runtime.chromium.launch(headless=True)
                    try:
                        page = browser.new_page()
                        page.goto(url, wait_until="domcontentloaded",
                                  timeout=int(timeout * 1000))
                        try:
                            page.wait_for_load_state("networkidle",
                                                     timeout=min(5_000, int(timeout * 1000)))
                        except Exception:  # noqa: BLE001 - some pages never go idle
                            pass
                        title = str(page.title() or "")
                        text = str(page.inner_text("body") or "")
                        engine = "playwright"
                    finally:
                        browser.close()
            except Exception as failure:  # noqa: BLE001 - fall back to a Chromium binary
                problem = f"Playwright failed: {failure}"
        if not text:
            binary = self._browser_binary()
            if not binary:
                # A headless browser is optional.  Falling back to HTTP text keeps the
                # tool useful on a small install while telling the model what happened.
                fetched = self.http_fetch({
                    "url": url, "method": "GET", "timeout": timeout,
                    "extract": "text", "allow_private": allow_private,
                }, context)
                if fetched.get("ok") is False:
                    raise ToolError(
                        f"Browser fallback request failed: "
                        f"{fetched.get('error') or fetched.get('status')}"
                    )
                text = str(fetched.get("text") or fetched.get("body") or "")
                return {
                    "ok": True, "url": url, "title": "",
                    "text": text[:max_chars], "chars": len(text),
                    "truncated": len(text) > max_chars, "engine": "http",
                    "note": (("No headless browser is installed; returned HTTP text instead. "
                              "Install Playwright or Chromium for JavaScript pages. ")
                             + (problem + " " if problem else "")),
                }
            try:
                completed = subprocess.run(
                    [binary, "--headless=new", "--disable-gpu", "--no-sandbox",
                     "--dump-dom", url],
                    capture_output=True, text=True, timeout=timeout,
                )
            except (OSError, subprocess.TimeoutExpired) as failure:
                raise ToolError(f"Headless browser failed: {failure}") from failure
            dom = completed.stdout or ""
            if completed.returncode != 0 and not dom.strip():
                detail = (completed.stderr or "returned no DOM").strip()[:500]
                raise ToolError(f"Headless browser failed: {detail}")
            match = re.search(r"<title[^>]*>(.*?)</title>", dom, re.IGNORECASE | re.DOTALL)
            title = html_to_text(match.group(1), 300) if match else ""
            text = html_to_text(dom, max_chars)
            engine = "chromium"
        self._check_cancelled()
        text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
        full_length = len(text)
        text = text[:max_chars]
        result = {
            "ok": True, "url": url, "title": title, "text": text, "chars": full_length,
            "truncated": full_length > len(text), "engine": engine,
        }
        if problem:
            result["note"] = problem
        return result

    @staticmethod
    def _browser_binary() -> str:
        override = os.environ.get("MARIANA_BROWSER_BIN", "").strip()
        if override:
            return override
        for name in ("chromium", "chromium-browser", "google-chrome", "google-chrome-stable",
                     "microsoft-edge", "microsoft-edge-stable"):
            found = shutil.which(name)
            if found:
                return found
        return ""

    def _validate_public_url(self, url: str, allow_private: bool,
                             resolve: bool = True) -> None:
        try:
            parsed = urllib.parse.urlparse(url)
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
        except ValueError as error:
            raise ToolError(f"Invalid URL: {error}") from error
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ToolError("URL must be a full http(s) URL")
        host = parsed.hostname.lower().rstrip(".")
        if host in PRIVATE_BLOCKLIST:
            raise ToolError("Cloud metadata addresses are always blocked")
        try:
            literal = ipaddress.ip_address(host)
        except ValueError:
            literal = None
        if literal is not None and not allow_private and _private_address(str(literal)):
            raise ToolError("Private addresses are blocked; set allow_private on this ability to opt in")
        if allow_private:
            return
        if host == "localhost" or host.endswith(".localhost"):
            raise ToolError("Private addresses are blocked; set allow_private on this ability to opt in")
        if not resolve:
            return  # test mode never touches the network
        try:
            addresses = {item[4][0] for item in socket.getaddrinfo(host, port)}
        except socket.gaierror as error:
            raise ToolError(f"Cannot resolve host: {host}") from error
        for address in addresses:
            if _private_address(address):
                raise ToolError(
                    "Private addresses are blocked; set allow_private on this ability to opt in"
                )

    def _stream_read(self, path: Path, relative: str, start: int, end: int) -> dict:
        """Read a large file line by line, never loading it into memory."""
        selected: list[str] = []
        used = 0
        midline = False
        included_end = start
        has_more = False
        try:
            with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
                for number, line in enumerate(handle, 1):
                    if number < start:
                        continue
                    if number > end:
                        has_more = True
                        break
                    if not selected and len(line) > READ_CHUNK_CHARS:
                        selected.append(line[:READ_CHUNK_CHARS])
                        midline = True
                        has_more = True
                        included_end = number
                        break
                    if selected and used + len(line) > READ_CHUNK_CHARS:
                        has_more = True
                        break
                    selected.append(line)
                    used += len(line)
                    included_end = number
                    if used >= READ_CHUNK_CHARS:
                        has_more = True
                        break
        except OSError as error:
            raise ToolError(f"Cannot read {relative}: {error}") from error
        body = "".join(selected)
        result = {
            "path": relative, "content": body, "start_line": start,
            "end_line": included_end, "truncated": has_more, "large_file": True,
            "total_lines": None,
        }
        if not selected:
            result["hint"] = (
                f"Line {start:,} is past the end of {relative}; read from a smaller "
                "start_line or use file.search."
            )
        elif has_more and not midline:
            result["next_start_line"] = included_end + 1
        if midline:
            result["content_cut_midline"] = True
            result["hint"] = (
                "This line is longer than the read budget; use file.search to inspect "
                "the rest instead of re-reading the same line."
            )
        return result

    def file_read(self, params: dict, context: dict) -> dict:
        path = self._sandbox_path(str(params.get("path") or params.get("input") or ""))
        relative = self._relative(path)
        if path.is_dir():
            return self._directory_listing(path)
        if not path.is_file():
            raise ToolError(self._missing_path_message(path))
        try:
            size = path.stat().st_size
        except OSError as error:
            raise ToolError(f"Cannot stat {relative}: {error}") from error
        if size > MAX_READ_SCAN_BYTES:
            raise ToolError(
                f"{relative} is {size / 1e6:.1f} MB, above the "
                f"{MAX_READ_SCAN_BYTES / 1e6:.0f} MB read limit; use file.search or "
                "git.inspect to inspect it."
            )
        try:
            with path.open("rb") as handle:
                head = handle.read(4096)
        except OSError as error:
            raise ToolError(f"Cannot read {relative}: {error}") from error
        if b"\x00" in head:
            return {"path": relative, "binary": True, "bytes": size,
                    "hint": "This is a binary file; use file.search for text matches or "
                            "another tool that understands its format."}
        try:
            start = max(1, int(params.get("start_line") or 1))
        except (TypeError, ValueError):
            start = 1
        try:
            end = int(params.get("end_line") or start + READ_CHUNK_LINES - 1)
        except (TypeError, ValueError):
            end = start + READ_CHUNK_LINES - 1
        end = max(start, end)
        if size > MAX_READ_STREAM_BYTES:
            return self._stream_read(path, relative, start, end)
        data, _ = read_cached(
            path, MAX_READ_STREAM_BYTES,
            accept=lambda value: b"\x00" not in value[:4096])
        if data is None:
            return self._stream_read(path, relative, start, end)
        content = data.decode("utf-8", errors="replace")
        if content.startswith("\ufeff"):
            content = content[1:]
        lines = content.splitlines(keepends=True)
        total = len(lines)
        start = min(start, max(1, total))
        end = max(start, min(max(1, total), end))
        selected: list[str] = []
        used = 0
        midline = False
        for index in range(start - 1, end):
            line = lines[index]
            if not selected and len(line) > READ_CHUNK_CHARS:
                # One generated line can exceed the whole read budget.  Return its
                # prefix and say so, instead of pretending the rest was delivered.
                selected.append(line[:READ_CHUNK_CHARS])
                midline = True
                break
            if selected and used + len(line) > READ_CHUNK_CHARS:
                break
            selected.append(line)
            used += len(line)
            if used >= READ_CHUNK_CHARS:
                break
        body = "".join(selected)
        included_end = start + len(selected) - 1 if selected else start
        truncated = included_end < total
        result = {
            "path": relative, "content": body,
            "start_line": start, "end_line": included_end, "total_lines": total,
            "truncated": truncated,
        }
        if truncated and not midline:
            result["next_start_line"] = included_end + 1
        if midline:
            result["content_cut_midline"] = True
            result["hint"] = (
                "This line is longer than the read budget; use file.search to inspect "
                "the rest instead of re-reading the same line."
            )
        return result

    def file_write(self, params: dict, context: dict) -> dict:
        path = self._sandbox_path(
            str(params.get("path") or params.get("input") or ""), for_write=True
        )
        content = params.get("content", params.get("text", ""))
        if not isinstance(content, str):
            content = json.dumps(content, indent=2, ensure_ascii=False)
        if len(content.encode("utf-8")) > MAX_FILE_BYTES:
            raise ToolError("Content exceeds the 2 MB ability limit")
        if self.dry_run:
            return {"dry_run": True, "path": str(path.relative_to(self.files)),
                    "bytes": len(content.encode("utf-8"))}
        _write_text_atomic(path, content)
        return {"path": str(path.relative_to(self.files)),
                "bytes": len(content.encode("utf-8")), "written": True}

    def file_list(self, params: dict, context: dict) -> dict:
        directory = self._sandbox_path(
            str(params.get("path") or params.get("input") or "."))
        relative = self._relative(directory)
        if not directory.exists():
            raise ToolError(self._missing_path_message(directory))
        pattern = str(params.get("pattern") or "*").strip() or "*"
        if pattern.startswith("/") or ".." in Path(pattern).parts:
            raise ToolError("List patterns cannot traverse outside the sandbox")
        recursive = _truthy(params.get("recursive"), False) or "**" in pattern
        try:
            limit = max(1, min(MAX_LIST_ENTRIES, int(params.get("max_entries") or 200)))
        except (TypeError, ValueError):
            limit = 200
        if directory.is_file():
            paths = [directory]
        elif pattern in {"*", "."} and not recursive:
            paths = sorted(directory.iterdir())
        elif recursive:
            paths = sorted(directory.rglob(pattern))
        else:
            paths = sorted(directory.glob(pattern))
        entries: list[dict] = []
        files: list[dict] = []
        truncated = False
        for child in paths:
            if len(entries) >= limit:
                truncated = True
                break
            if child.name == ".mariana-ide.json" or child.is_symlink():
                continue
            try:
                resolved = child.resolve()
                resolved.relative_to(self.files)
            except (OSError, ValueError):
                continue
            if self._denied(resolved):
                continue
            item = {"name": child.name, "path": self._relative(child)}
            if resolved.is_dir():
                item.update({"type": "dir"})
                entries.append(item)
            elif resolved.is_file():
                try:
                    info = resolved.stat()
                except OSError:
                    continue
                modified = datetime.fromtimestamp(info.st_mtime).isoformat(timespec="seconds")
                item.update({"type": "file", "bytes": info.st_size, "modified": modified})
                entries.append(item)
                files.append({"name": child.name, "bytes": info.st_size,
                              "modified": modified, "path": self._relative(child)})
        return {"path": relative, "pattern": pattern, "entries": entries,
                "files": files, "truncated": truncated}

    def file_edit(self, params: dict, context: dict) -> dict:
        """Replace one exact block; refusing ambiguity is safer than guessing."""
        path = self._sandbox_path(str(params.get("path") or params.get("input") or ""))
        if not path.is_file():
            raise ToolError(f"File not found: {path.relative_to(self.files)}")
        old = params.get("old", params.get("find"))
        new = params.get("new", params.get("replace", ""))
        if not isinstance(old, str) or old == "":
            raise ToolError("file.edit needs the exact text to replace in 'old'")
        if not isinstance(new, str):
            new = json.dumps(new, indent=2, ensure_ascii=False)
        if path.stat().st_size > MAX_FILE_BYTES:
            raise ToolError("File exceeds the 2 MB ability limit")
        raw = path.read_bytes()
        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError:
            raise ToolError(
                f"{path.relative_to(self.files)} is not UTF-8 text; "
                "file.edit refuses to rewrite it and lose bytes"
            ) from None
        count = content.count(old)
        if count == 0:
            raise ToolError(
                f"Text to replace was not found in {path.relative_to(self.files)}; read the file again"
            )
        replace_all = _truthy(params.get("replace_all"), False)
        if count > 1 and not replace_all:
            raise ToolError(
                f"Text occurs {count} times in {path.relative_to(self.files)}; "
                "include more surrounding text, or set replace_all"
            )
        updated = content.replace(old, new) if replace_all else content.replace(old, new, 1)
        encoded = updated.encode("utf-8")
        if len(encoded) > MAX_FILE_BYTES:
            raise ToolError("Result exceeds the 2 MB ability limit")
        if self.dry_run:
            return {"dry_run": True, "path": str(path.relative_to(self.files)),
                    "replacements": count if replace_all else 1, "changed": content != updated}
        _write_text_atomic(path, updated)
        return {
            "path": str(path.relative_to(self.files)),
            "replacements": count if replace_all else 1,
            "bytes": len(encoded), "changed": content != updated,
        }

    def _git_grep(self, query: str, use_regex: bool, case_sensitive: bool,
                  include: str, limit: int) -> dict | None:
        """Search a Git work tree with git grep, or return ``None`` to fall back."""
        git_dir = self.files / ".git"
        if not git_dir.exists() or git_dir.is_symlink():
            return None
        git = shutil.which("git")
        if not git:
            return None
        arguments = [
            git, "-C", str(self.files), "grep", "-n", "-I", "--no-color",
            "--untracked", "-m", str(max(1, limit + 1)),
        ]
        if not case_sensitive:
            arguments.append("-i")
        arguments.append("-E" if use_regex else "-F")
        arguments += ["-e", query]
        paths = [item.strip() for item in str(include or "").split(",") if item.strip()]
        arguments += ["--", *(paths or ["."])]
        environment = {key: value for key, value in os.environ.items()
                       if not key.startswith("GIT_")}
        environment.update({"GIT_OPTIONAL_LOCKS": "0", "GIT_TERMINAL_PROMPT": "0",
                            "GIT_PAGER": "cat", "GIT_CONFIG_NOSYSTEM": "1",
                            "GIT_CONFIG_GLOBAL": os.devnull})
        matches = []
        truncated = False
        process = None
        try:
            process = subprocess.Popen(
                arguments, cwd=self.files, env=environment, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            assert process.stdout is not None
            for line in process.stdout:
                if len(matches) >= limit:
                    truncated = True
                    break
                parts = line.rstrip("\n").split(":", 2)
                if len(parts) < 3:
                    continue
                path, number, text = parts
                try:
                    line_number = int(number)
                except ValueError:
                    continue
                matches.append({"path": path.replace("\\", "/"), "line": line_number,
                                "text": text.strip()[:300]})
            if truncated:
                # The requested result count is already reached; stopping git now
                # avoids buffering a huge match stream for a broad query.
                process.terminate()
            try:
                returncode = process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                returncode = process.wait(timeout=2)
        except (OSError, subprocess.SubprocessError):
            if process is not None:
                process.kill()
            return None
        finally:
            if process is not None:
                if process.stdout is not None:
                    process.stdout.close()
                if process.stderr is not None:
                    process.stderr.close()
        if returncode not in {0, 1, -15} and not matches:
            # Git versions before --untracked, or a regex dialect Git rejects:
            # the Python walker below is the portable path.
            return None
        return {"query": query, "matches": matches, "files_scanned": None,
                "truncated": truncated, "engine": "git"}

    def file_search(self, params: dict, context: dict) -> dict:
        query = str(params.get("query") or params.get("input") or "")
        if not query:
            raise ToolError("file.search needs a query")
        use_regex = _truthy(params.get("regex"), True)
        case_sensitive = _truthy(params.get("case_sensitive"), False)
        flags = 0 if case_sensitive else re.IGNORECASE
        if len(query) > 300:
            raise ToolError("Search query is limited to 300 characters")
        regex_fallback = False
        if use_regex:
            try:
                pattern = re.compile(query, flags)
            except re.error:
                # Models frequently send a literal snippet such as "strlen(".
                # Falling back to a literal search is more useful than failing.
                regex_fallback = True
                use_regex = False
                pattern = None
            if use_regex:
                match_line = lambda line: pattern.search(line) is not None  # noqa: E731
            else:
                needle = query if case_sensitive else query.lower()
                match_line = (lambda line: needle in line if case_sensitive  # noqa: E731
                              else needle in line.lower())
        else:
            needle = query if case_sensitive else query.lower()
            match_line = (lambda line: needle in line if case_sensitive  # noqa: E731
                          else needle in line.lower())
        try:
            limit = max(1, min(MAX_SEARCH_RESULTS, int(params.get("max_results") or 100)))
        except (TypeError, ValueError):
            limit = 100
        include = str(params.get("include") or "").replace("\\", "/").strip()
        self._check_cancelled()
        fast = self._git_grep(query, use_regex, case_sensitive, include, limit)
        if fast is not None:
            if regex_fallback:
                fast.update({"regex_fallback": True,
                             "hint": "The query was not a valid regular expression; "
                                     "it was searched as literal text."})
            return fast
        matches, scanned, truncated = [], 0, False
        visited = 0
        includes = [item.strip() for item in include.split(",") if item.strip()]
        for path in self._walk_files():
            visited += 1
            if visited > MAX_SEARCH_FILES:
                truncated = True
                break
            if visited % 250 == 0:
                self._check_cancelled()
            if includes:
                try:
                    relative = path.relative_to(self.files)
                except ValueError:
                    continue
                if not any(relative.match(item) for item in includes):
                    continue
            data = _text_bytes(path)
            if data is None:
                continue
            scanned += 1
            text = data.decode("utf-8", errors="replace")
            for number, line in enumerate(text.splitlines(), 1):
                if not match_line(line):
                    continue
                matches.append({
                    "path": str(path.relative_to(self.files)),
                    "line": number,
                    "text": line.strip()[:300],
                })
                if len(matches) >= limit:
                    truncated = True
                    break
            if truncated:
                break
        result = {"query": query, "matches": matches, "files_scanned": scanned,
                  "truncated": truncated, "engine": "python"}
        if regex_fallback:
            result.update({"regex_fallback": True,
                           "hint": "The query was not a valid regular expression; "
                                   "it was searched as literal text."})
        return result

    def file_tree(self, params: dict, context: dict) -> dict:
        try:
            limit = max(1, min(MAX_TREE_ENTRIES, int(params.get("max_entries") or 600)))
        except (TypeError, ValueError):
            limit = 600
        try:
            depth = max(1, min(32, int(params.get("depth") or 12)))
        except (TypeError, ValueError):
            depth = 12
        entries, truncated = [], False
        for base, directories, names in os.walk(self.files, followlinks=False):
            base_path = Path(base)
            try:
                relative_base = base_path.relative_to(self.files)
            except ValueError:
                directories[:] = []
                continue
            directories[:] = sorted(
                name for name in directories
                if name not in IGNORED_DIRS
                and not (base_path / name).is_symlink()
                and not self._denied((base_path / name).resolve()))
            if len(relative_base.parts) >= depth:
                directories[:] = []
            for name in directories:
                if len(entries) >= limit:
                    truncated = True
                    break
                entries.append({"path": (relative_base / name).as_posix(), "type": "dir"})
            if truncated:
                break
            for name in sorted(names):
                if len(entries) >= limit:
                    truncated = True
                    break
                path = base_path / name
                if name == ".mariana-ide.json" or path.is_symlink():
                    continue
                if name.startswith(".") and name.endswith(".tmp"):
                    continue
                try:
                    resolved = path.resolve()
                    resolved.relative_to(self.files)
                except (OSError, ValueError):
                    continue
                if self._denied(resolved) or not resolved.is_file():
                    continue
                try:
                    size = resolved.stat().st_size
                except OSError:
                    continue
                entries.append({"path": (relative_base / name).as_posix(),
                                "type": "file", "bytes": size})
            if truncated:
                break
        return {"entries": entries, "truncated": truncated, "depth": depth}

    def file_delete(self, params: dict, context: dict) -> dict:
        path = self._sandbox_path(str(params.get("path") or params.get("input") or ""))
        if not path.is_file():
            raise ToolError(f"File not found: {path.relative_to(self.files)}")
        if self.dry_run:
            return {"dry_run": True, "path": str(path.relative_to(self.files)), "deleted": True}
        path.unlink()
        HOT_CACHE.invalidate(str(path))
        return {"path": str(path.relative_to(self.files)), "deleted": True}

    def code_diagnostics(self, params: dict, context: dict) -> dict:
        """Check Python syntax and JSON validity; pyflakes adds deeper rules when installed."""
        target = str(params.get("path") or params.get("input") or "").strip()
        if target and target not in {".", "./"}:
            candidate = self._sandbox_path(target)
            if not candidate.exists():
                raise ToolError(f"Path not found: {target}")
            if candidate.is_file():
                files = [candidate]
            else:
                files = []
                for path in self._walk_files():
                    try:
                        path.relative_to(candidate)
                    except ValueError:
                        continue
                    files.append(path)
        else:
            files = list(self._walk_files())
        issues, checked, skipped = [], 0, 0
        truncated = len(files) > MAX_DIAGNOSTIC_FILES
        for path in files[:MAX_DIAGNOSTIC_FILES]:
            if len(issues) >= MAX_DIAGNOSTIC_ISSUES:
                truncated = True
                break
            suffix = path.suffix.lower()
            if suffix not in {".py", ".json"}:
                continue
            data = _text_bytes(path)
            if data is None:
                skipped += 1
                continue
            checked += 1
            text = data.decode("utf-8", errors="replace")
            relative = str(path.relative_to(self.files))
            if suffix == ".json":
                try:
                    json.loads(text)
                except json.JSONDecodeError as error:
                    issues.append({"path": relative, "line": error.lineno,
                                   "column": error.colno, "message": error.msg,
                                   "rule": "json", "severity": "error"})
                continue
            try:
                compile(text, relative, "exec")
            except SyntaxError as error:
                issues.append({"path": relative, "line": error.lineno or 1,
                               "column": error.offset or 1, "message": error.msg,
                               "rule": "syntax", "severity": "error"})
                continue
            issues.extend(_pyflakes_issues(text, relative))
        return {"files_checked": checked, "files_skipped": skipped, "issues": issues,
                "ok": not issues, "truncated": truncated}

    def sqlite_query(self, params: dict, context: dict) -> dict:
        """Run one bound SQL statement against a sandboxed SQLite database."""
        write = _truthy(params.get("write"), False)
        path_value = str(params.get("path") or params.get("input") or "")
        if not path_value:
            raise ToolError("sqlite.query needs a database path; set it on the ability config or pass path")
        path = self._sandbox_path(path_value, for_write=write)
        if not write and not path.is_file():
            raise ToolError(f"Database not found: {path.relative_to(self.files)}")
        sql = str(params.get("sql") or params.get("query") or "").strip()
        if not sql:
            raise ToolError("sqlite.query needs a SQL statement")
        if len(sql) > 10_000:
            raise ToolError("SQL statement is limited to 10,000 characters")
        parameters = params.get("params", params.get("parameters"))
        if parameters is None:
            parameters = []
        if not isinstance(parameters, (list, tuple, dict)):
            raise ToolError("params must be a JSON list or object")
        try:
            limit = max(1, min(5000, int(params.get("max_rows") or 200)))
        except (TypeError, ValueError):
            limit = 200
        try:
            timeout = max(1.0, min(30.0, float(params.get("timeout") or 5)))
        except (TypeError, ValueError):
            timeout = 5.0
        relative = str(path.relative_to(self.files))
        if self.dry_run and write:
            return {"dry_run": True, "path": relative, "sql": sql, "write": True,
                    "note": "Database write skipped in test mode"}
        try:
            if write:
                connection = sqlite3.connect(str(path), timeout=timeout)
            else:
                connection = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True,
                                             timeout=timeout)
        except sqlite3.Error as error:
            raise ToolError(f"Could not open database: {error}") from error
        deadline = time.monotonic() + timeout

        def progress() -> int:
            if self.cancelled is not None and self.cancelled.is_set():
                return 1
            return 1 if time.monotonic() > deadline else 0

        attach = getattr(sqlite3, "SQLITE_ATTACH", 24)
        detach = getattr(sqlite3, "SQLITE_DETACH", 25)
        deny = getattr(sqlite3, "SQLITE_DENY", 1)

        def authorizer(action, arg1, arg2, dbname, source):
            # ATTACH would escape the sandbox; the read-only URI already blocks writes.
            return deny if action in {attach, detach} else sqlite3.SQLITE_OK

        connection.set_authorizer(authorizer)
        connection.set_progress_handler(progress, 5000)
        try:
            cursor = connection.execute(sql, parameters)
            if cursor.description:
                columns = [str(item[0]) for item in cursor.description]
                fetched = cursor.fetchmany(limit + 1)
                truncated = len(fetched) > limit
                rows = [[_sqlite_cell(value) for value in row] for row in fetched[:limit]]
                result = {"path": relative, "sql": sql, "write": write,
                          "columns": columns, "rows": rows, "row_count": len(rows),
                          "truncated": truncated}
            else:
                changes = cursor.rowcount
                result = {"path": relative, "sql": sql, "write": write,
                          "columns": [], "rows": [], "row_count": 0,
                          "changes": changes, "truncated": False}
            if write:
                connection.commit()
            return result
        except sqlite3.OperationalError as error:
            if "interrupted" in str(error).lower():
                if self.cancelled is not None and self.cancelled.is_set():
                    raise ToolError("Database query cancelled") from error
                raise ToolError(f"Database query timed out after {timeout:.0f}s") from error
            if not write and "readonly" in str(error).lower():
                raise ToolError(
                    "This SQLite ability is read-only; set write to true in its config to modify the database"
                ) from error
            raise ToolError(f"SQLite error: {error}") from error
        except sqlite3.ProgrammingError as error:
            raise ToolError(
                "sqlite.query accepts one statement at a time; bind values through params"
            ) from error
        except sqlite3.Error as error:
            raise ToolError(f"SQLite error: {error}") from error
        finally:
            connection.close()

    def csv_read(self, params: dict, context: dict) -> dict:
        """Read a CSV/TSV file as named columns and records, without touching the network."""
        path_value = str(params.get("path") or params.get("input") or "")
        if not path_value:
            raise ToolError("csv.read needs a path to a CSV or TSV file")
        path = self._sandbox_path(path_value)
        if not path.is_file():
            raise ToolError(f"File not found: {path.relative_to(self.files)}")
        data = _text_bytes(path)
        if data is None:
            raise ToolError("CSV file is too large or is not text")
        text = data.decode("utf-8-sig", errors="replace")
        delimiter = str(params.get("delimiter") or "")
        if not delimiter:
            delimiter = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
        if len(delimiter) != 1:
            raise ToolError("delimiter must be exactly one character")
        header = _truthy(params.get("header"), True)
        try:
            limit = max(1, min(10000, int(params.get("max_rows") or 1000)))
        except (TypeError, ValueError):
            limit = 1000
        try:
            reader = csv.reader(io.StringIO(text), delimiter=delimiter)
            first = next(reader, None)
        except csv.Error as error:
            raise ToolError(f"CSV error: {error}") from error
        columns: list[str] = []
        seen: dict[str, int] = {}
        if header and first is not None:
            for index, item in enumerate(first):
                name = str(item).strip() or f"column_{index + 1}"
                if name in seen:
                    seen[name] += 1
                    name = f"{name}_{seen[name]}"
                else:
                    seen[name] = 1
                columns.append(name)

        def source_rows():
            if first is not None and not header:
                yield first
            yield from reader

        rows: list[dict] = []
        truncated = False
        try:
            for row in source_rows():
                if len(rows) >= limit:
                    truncated = True
                    break
                record = {}
                for index, value in enumerate(row):
                    if index < len(columns):
                        key = columns[index]
                    else:
                        key = f"column_{index + 1}"
                        columns.append(key)
                    record[key] = value
                rows.append(record)
        except csv.Error as error:
            raise ToolError(f"CSV error: {error}") from error
        return {"path": str(path.relative_to(self.files)), "delimiter": delimiter,
                "header": header, "columns": columns, "rows": rows,
                "row_count": len(rows), "truncated": truncated}

    def git_inspect(self, params: dict, context: dict) -> dict:
        """Read-only git status, diff or log with fixed arguments and a sanitized environment."""
        action = str(params.get("action") or "status").strip().lower()
        action = action.split()[0] if action else "status"
        aliases = {
            "changes": "status", "working": "status", "short": "status",
            "files": "status", "staged": "diff", "history": "log",
            "commits": "log", "show": "log", "head": "log", "last": "log",
        }
        action = aliases.get(action, action)
        if action not in {"status", "diff", "log"}:
            raise ToolError("git.inspect action must be status, diff, or log")
        git_dir = self.files / ".git"
        if not git_dir.exists() or git_dir.is_symlink():
            # A missing repository is a normal property of a project, not a tool
            # failure; returning a structured miss lets the model choose another
            # source instead of retrying the same call.
            return {"action": action, "repository": False, "clean": None,
                    "files": [], "truncated": False,
                    "hint": "No Git repository in this folder; use file.tree, file.list "
                            "or file.search instead."}
        git = shutil.which("git")
        if not git:
            return {"action": action, "repository": True, "git": False,
                    "clean": None, "files": [], "truncated": False,
                    "hint": "Git is not installed; use file.tree or file.search instead."}
        try:
            limit = max(1, min(200, int(params.get("limit") or 30)))
        except (TypeError, ValueError):
            limit = 30
        try:
            max_chars = max(500, min(50000, int(params.get("max_chars") or 12000)))
        except (TypeError, ValueError):
            max_chars = 12000
        arguments = [
            git, "--no-pager", "-C", str(self.files),
            "-c", "core.fsmonitor=false", "-c", "core.pager=cat",
            "-c", "advice.detachedHead=false",
        ]
        path_filter = str(params.get("path") or "").strip()
        relative_filter = ""
        if path_filter and path_filter not in {".", "./"}:
            relative_filter = self._sandbox_path(path_filter).relative_to(self.files).as_posix()
        if action == "status":
            arguments += ["status", "--porcelain=v1", "-b"]
        elif action == "diff":
            arguments += ["diff", "--no-ext-diff", "--no-textconv", "--no-color", "--unified=3"]
            if _truthy(params.get("staged"), False):
                arguments.append("--cached")
            if relative_filter:
                arguments += ["--", relative_filter]
        else:
            arguments += ["log", "--no-decorate", "--no-color", "--date=short",
                          "--pretty=format:%h%x09%ad%x09%s", "-n", str(limit)]
            if relative_filter:
                arguments += ["--", relative_filter]
        environment = {key: value for key, value in os.environ.items()
                       if not key.startswith("GIT_")}
        environment.update({
            "GIT_OPTIONAL_LOCKS": "0", "GIT_TERMINAL_PROMPT": "0", "GIT_PAGER": "cat",
            "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
        })
        try:
            completed = subprocess.run(
                arguments, cwd=self.files, env=environment, capture_output=True,
                text=True, timeout=10,
            )
        except subprocess.TimeoutExpired as error:
            raise ToolError("Git command timed out") from error
        except OSError as error:
            raise ToolError(f"Could not start git: {error}") from error
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()[:1000]
            raise ToolError(f"Git {action} failed: {detail or 'unknown error'}")
        output = completed.stdout
        truncated = len(output) > max_chars
        output = output[:max_chars].rstrip()
        if truncated:
            output += "…"
        if action == "status":
            branch = ""
            files = []
            for line in output.splitlines():
                if line.startswith("## "):
                    branch = line[3:].strip()
                elif len(line) >= 3 and line[:2].strip():
                    files.append({"status": line[:2].strip(), "path": line[3:].strip()})
            return {"action": "status", "branch": branch, "files": files,
                    "clean": not files, "truncated": truncated}
        if action == "diff":
            return {"action": "diff", "diff": output, "staged": _truthy(params.get("staged"), False),
                    "path": relative_filter, "truncated": truncated}
        commits = []
        for line in output.splitlines():
            parts = line.split("\t", 2)
            if len(parts) == 3:
                commits.append({"hash": parts[0], "date": parts[1], "subject": parts[2]})
        return {"action": "log", "commits": commits, "truncated": truncated}

    def _walk_files(self):
        """Every regular file under the sandbox, without ignored or denied folders."""
        for path in self._walk_paths():
            try:
                resolved = path.resolve()
                resolved.relative_to(self.files)
            except (OSError, ValueError):
                continue
            if resolved.is_symlink() or not resolved.is_file():
                continue
            if resolved.name == ".mariana-ide.json":
                continue
            if resolved.name.startswith(".") and resolved.name.endswith(".tmp"):
                continue
            if self._denied(resolved):
                continue
            yield resolved

    def _memory_path(self, context: dict) -> Path:
        agent_id = (context.get("agent") or {}).get("id") or "agent"
        folder = self.root / "memory"
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f"{re.sub(r'[^a-z0-9-]+', '-', str(agent_id).lower())}.json"

    def _memory(self, context: dict) -> dict:
        path = self._memory_path(context)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {}
        return data if isinstance(data, dict) else {}

    def memory(self, params: dict, context: dict) -> dict:
        action = str(params.get("action") or "get").lower()
        key = str(params.get("key") or "")
        data = self._memory(context)
        if action == "list":
            return {"keys": sorted(data), "values": data}
        if not key:
            raise ToolError("Memory action needs a key")
        if action == "get":
            return {"key": key, "value": data.get(key), "found": key in data}
        if action == "delete":
            if self.dry_run:
                return {"dry_run": True, "key": key, "deleted": key in data}
            existed = data.pop(key, None) is not None
            self._write_memory(context, data)
            return {"key": key, "deleted": existed}
        if action == "set":
            value = params.get("value", params.get("input"))
            if self.dry_run:
                return {"dry_run": True, "key": key, "value": value}
            data[key] = value
            self._write_memory(context, data)
            return {"key": key, "value": value, "stored": True}
        raise ToolError("Memory action must be get, set, list, or delete")

    def _write_memory(self, context: dict, data: dict) -> None:
        path = self._memory_path(context)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)

    def math_eval(self, params: dict, context: dict) -> dict:
        expression = str(params.get("expression") or params.get("input") or "")
        if not expression:
            raise ToolError("math.eval needs an expression")
        return {"expression": expression, "value": evaluate_expression(expression, context)}

    def time_now(self, params: dict, context: dict) -> dict:
        now = datetime.now()
        return {
            "local": now.isoformat(timespec="seconds"),
            "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "epoch": time.time(),
        }

    def shell_exec(self, params: dict, context: dict) -> dict:
        if not self.os_access and os.environ.get("MARIANA_AGENT_ALLOW_SHELL") != "1":
            raise ToolError(
                "Shell ability is disabled. Enable OS access in Models settings, "
                "set allow_os_access in agent-workspace/settings.json, or start Studio "
                "with MARIANA_AGENT_ALLOW_SHELL=1 to opt in."
            )
        command = str(params.get("command") or params.get("input") or "").strip()
        if not command:
            raise ToolError("shell.exec needs a command")
        timeout = min(120.0, max(1.0, float(params.get("timeout", 30))))
        if self.dry_run:
            return {"dry_run": True, "command": command, "returncode": 0,
                    "stdout": "", "stderr": "", "ok": True,
                    "note": "Shell command skipped in test mode"}
        try:
            process = subprocess.Popen(
                command, shell=True, cwd=self.files, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, start_new_session=(os.name != "nt"),
            )
        except OSError as error:
            raise ToolError(f"Could not start command: {error}") from error
        deadline = time.monotonic() + timeout
        while True:
            if self.cancelled is not None and self.cancelled.is_set():
                self._stop_process(process)
                raise ToolError("Command cancelled")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._stop_process(process)
                raise ToolError(f"Command timed out after {timeout:.0f}s")
            try:
                stdout, stderr = process.communicate(timeout=min(0.1, remaining))
                break
            except subprocess.TimeoutExpired:
                continue
        if _shell_may_write(command):
            HOT_CACHE.clear()
        return {
            "command": command, "returncode": process.returncode,
            "stdout": stdout[-MAX_OUTPUT_CHARS:],
            "stderr": stderr[-MAX_OUTPUT_CHARS:],
            "ok": process.returncode == 0,
        }

    @staticmethod
    def _stop_process(process: subprocess.Popen) -> None:
        if os.name == "nt":
            process.kill()
        else:
            try:
                os.killpg(process.pid, signal.SIGKILL)  # the shell and its children
            except (ProcessLookupError, PermissionError):
                process.kill()
        try:
            process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            pass


def json_safe(value, limit: int = MAX_OUTPUT_CHARS):
    """Make ability outputs safe for run history and JSON responses."""
    try:
        if isinstance(value, bytes):
            value = base64.b64encode(value).decode("ascii")
        encoded = json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        encoded = json.dumps(str(value), ensure_ascii=False)
    if len(encoded) > limit:
        return {"truncated": True, "chars": len(encoded), "preview": encoded[:limit]}
    return json.loads(encoded)


def _sqlite_cell(value):
    """SQLite values are JSON-friendly except blobs, which become base64 text."""
    return base64.b64encode(value).decode("ascii") if isinstance(value, bytes) else value
