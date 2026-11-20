"""Agent Management Language (AML) v1.3.

AML is a portable JSON (or YAML, when PyYAML is available) document that
describes an agent as a directed flowchart plus abilities, schedules and run
settings.  This module owns the schema, normalization, validation, template
rendering and the sandboxed expression language used by conditions.

The node and ability tables below are the single source of truth: the runtime
executes them, the validator checks them, and the visual designer renders its
palette and inspector forms from them through the protocol metadata.
"""

from __future__ import annotations

import ast
import json
import math
import os
import re
import time
from datetime import datetime, timedelta

AML_VERSION = "1.6"
SUPPORTED_VERSIONS = ("1.0", "1.1", "1.2", "1.3", "1.4", "1.5", "1.6")
MAX_DOCUMENT_BYTES = 1_000_000

FIELD_TYPES = (
    "text", "template", "expression", "number", "select", "boolean", "json",
    "model", "ability", "abilities", "agent",
)
_MODEL_FIELDS = [
    {"key": "model", "label": "Model", "type": "model",
     "help": "Leave empty to use the agent's default model."},
    {"key": "system", "label": "System prompt", "type": "template", "rows": 3},
]
_SAMPLING_FIELDS = [
    {"key": "temperature", "label": "Temperature", "type": "number", "min": 0, "max": 5, "step": 0.1},
    {"key": "max_tokens", "label": "Max tokens", "type": "number", "min": 1, "max": 131072, "step": 1},
]
SWARM_STRATEGIES = ("auto", "map_reduce", "peer_review", "debate", "pipeline")
SWARM_STRATEGY_LABELS = {
    "auto": "Automatic (the leader decides)",
    "map_reduce": "Map reduce (parallel, then merge)",
    "peer_review": "Peer review (work, review, revise)",
    "debate": "Debate (challenge and strengthen)",
    "pipeline": "Pipeline (each member builds on the last)",
}

NODE_TYPES = {
    "start": {
        "label": "Start", "category": "trigger", "icon": "▶",
        "description": "Entry point. Receives the run input.",
        "fields": [],
    },
    "end": {
        "label": "End", "category": "output", "icon": "■",
        "description": "Finish the run and return a value.",
        "fields": [{"key": "output", "label": "Output", "type": "template", "rows": 3,
                    "help": "What the run returns. {{ last }} is the previous node's output."}],
    },
    "llm": {
        "label": "LLM", "category": "intelligence", "icon": "✦",
        "description": "Call any model from the Studio model library once.",
        "fields": [
            *_MODEL_FIELDS,
            {"key": "prompt", "label": "Prompt", "type": "template", "rows": 6,
             "help": "Leave empty to send the previous node's output."},
            {"key": "format", "label": "Reply format", "type": "select",
             "options": [["text", "Text"], ["json", "JSON value"]],
             "help": "JSON adds a parsed value under output.json and fails the node when the reply is not JSON."},
            *_SAMPLING_FIELDS,
        ],
    },
    "agent": {
        "label": "AI agent", "category": "intelligence", "icon": "✺",
        "description": "A model that decides which abilities to use, step by step, until it has an answer.",
        "fields": [
            *_MODEL_FIELDS,
            {"key": "goal", "label": "Goal", "type": "template", "rows": 5,
             "help": "The task to accomplish. Leave empty to use the previous node's output."},
            {"key": "abilities", "label": "Abilities it may use", "type": "abilities",
             "help": "None selected means every ability granted to this agent."},
            {"key": "max_iterations", "label": "Max tool rounds", "type": "number",
             "min": 1, "max": 40, "step": 1},
            *_SAMPLING_FIELDS,
        ],
    },
    "swarm": {
        "label": "Agent swarm", "category": "intelligence", "icon": "⌬",
        "description": "A leader model plans a team, members work in parallel and exchange messages, "
                       "then review and revise each other's work before the leader synthesises the result.",
        "fields": [
            *_MODEL_FIELDS,
            {"key": "leader_model", "label": "Leader model", "type": "model",
             "help": "Empty uses the member model. The leader plans, decides and synthesises."},
            {"key": "goal", "label": "Swarm goal", "type": "template", "rows": 5,
             "help": "The task the whole swarm must complete. Empty uses the previous node's output."},
            {"key": "agents", "label": "Member tasks", "type": "json", "rows": 6,
             "help": "Optional JSON list of {id, label, role, task, instruction, tools, writes, "
                     "read_only, context_chars}. Empty lets the leader choose the team."},
            {"key": "tools", "label": "Abilities members may use", "type": "abilities",
             "help": "None selected means every ability granted to this agent."},
            {"key": "strategy", "label": "Strategy", "type": "select",
             "options": [[name, SWARM_STRATEGY_LABELS[name]] for name in SWARM_STRATEGIES]},
            {"key": "rounds", "label": "Feedback rounds", "type": "number",
             "min": 0, "max": 4, "step": 1,
             "help": "Each round adds peer review and a revision by the reviewed members."},
            {"key": "max_agents", "label": "Max agents", "type": "number",
             "min": 1, "max": 8, "step": 1},
            {"key": "max_steps", "label": "Max model rounds per agent", "type": "number",
             "min": 1, "max": 12, "step": 1},
            {"key": "max_seconds", "label": "Time budget (seconds)", "type": "number",
             "min": 10, "max": 1800, "step": 10},
            {"key": "write", "label": "Members may edit files", "type": "boolean",
             "help": "Requires swarm.allow_write at the top of the document. Write calls are serialized."},
            {"key": "memory", "label": "Shared blackboard", "type": "boolean"},
            {"key": "ipc", "label": "Agent-to-agent messages", "type": "boolean"},
        ],
    },
    "tool": {
        "label": "Ability", "category": "action", "icon": "⚙",
        "description": "Run one of the agent's granted abilities.",
        "fields": [
            {"key": "ability", "label": "Ability", "type": "ability"},
            {"key": "input", "label": "Input", "type": "json", "rows": 5,
             "help": "A template or a JSON object of the ability's input keys. "
                     "Empty passes the previous node's output."},
        ],
    },
    "condition": {
        "label": "Condition", "category": "logic", "icon": "◇",
        "description": "Branch on a safe boolean expression.",
        "fields": [{"key": "expression", "label": "Expression", "type": "expression", "rows": 3,
                    "help": "For example: contains(lower(last.text), 'urgent'). "
                            "Read input, last, vars and nodes.<id>."}],
    },
    "transform": {
        "label": "Transform", "category": "logic", "icon": "⇄",
        "description": "Render a template or assign variables.",
        "fields": [
            {"key": "template", "label": "Template", "type": "template", "rows": 4},
            {"key": "format", "label": "Result format", "type": "select",
             "options": [["text", "Text"], ["json", "Parse as JSON"]]},
            {"key": "assign", "label": "Assign variables", "type": "json", "rows": 4,
             "help": "A JSON object; each value is a template stored under vars.<name>."},
        ],
    },
    "delay": {
        "label": "Delay", "category": "flow", "icon": "◷",
        "description": "Wait before continuing (maximum 300 seconds).",
        "fields": [{"key": "seconds", "label": "Seconds", "type": "number",
                    "min": 0, "max": 300, "step": 1}],
    },
    "human": {
        "label": "Human input", "category": "flow", "icon": "✋",
        "description": "Pause the run until a person approves or supplies a value.",
        "fields": [
            {"key": "prompt", "label": "Question", "type": "template", "rows": 3},
            {"key": "kind", "label": "Answer type", "type": "select",
             "options": [["approve", "Approve / reject"], ["text", "Free text"],
                         ["choice", "One of several choices"]]},
            {"key": "choices", "label": "Choices", "type": "json", "rows": 3,
             "help": "A JSON list of strings, used when the answer type is a choice."},
            {"key": "default", "label": "Default answer", "type": "text",
             "help": "Used in test runs and when the wait times out."},
            {"key": "timeout", "label": "Wait at most (seconds)", "type": "number",
             "min": 0, "max": 86400, "step": 1, "help": "0 waits forever."},
        ],
    },
    "subagent": {
        "label": "Sub-agent", "category": "action", "icon": "◈",
        "description": "Run another agent and use its output.",
        "fields": [
            {"key": "agent", "label": "Agent", "type": "agent"},
            {"key": "input", "label": "Input", "type": "template", "rows": 3},
        ],
    },
    "foreach": {
        "label": "For each", "category": "flow", "icon": "⟳",
        "description": "Run another agent once for every item of a list.",
        "fields": [
            {"key": "items", "label": "Items", "type": "template", "rows": 2,
             "help": "A template that renders to a list, for example {{ nodes.fetch.json.items }}."},
            {"key": "agent", "label": "Agent to run per item", "type": "agent"},
            {"key": "input", "label": "Input per item", "type": "template", "rows": 2,
             "help": "{{ item }} is the current element and {{ index }} its position."},
            {"key": "max_items", "label": "Max items", "type": "number", "min": 1, "max": 200, "step": 1},
        ],
    },
    "memory": {
        "label": "Memory", "category": "action", "icon": "▤",
        "description": "Read or write persistent agent memory.",
        "fields": [
            {"key": "action", "label": "Action", "type": "select",
             "options": [["get", "Get"], ["set", "Set"], ["list", "List"], ["delete", "Delete"]]},
            {"key": "key", "label": "Key", "type": "template", "rows": 1},
            {"key": "value", "label": "Value", "type": "template", "rows": 3},
        ],
    },
    "note": {
        "label": "Note", "category": "annotation", "icon": "✎",
        "description": "A designer note; passes data through unchanged.",
        "fields": [{"key": "text", "label": "Note", "type": "template", "rows": 4}],
    },
}

# Options every node accepts in ``data`` in addition to its own fields.
COMMON_FIELDS = [
    {"key": "retry", "label": "Retries on error", "type": "number", "min": 0, "max": 5, "step": 1},
    {"key": "retry_delay", "label": "Seconds between retries", "type": "number",
     "min": 0, "max": 30, "step": 1},
]

ABILITY_TYPES = {
    "http.fetch": {
        "label": "HTTP request", "icon": "⇗",
        "description": "GET or POST a public URL. Private-network targets require allow_private.",
        "danger": False,
        "config": {"method": "GET", "headers": {}, "timeout": 15, "allow_private": False},
        "params": [
            {"key": "url", "label": "URL", "type": "template", "rows": 1},
            {"key": "method", "label": "Method", "type": "select",
             "options": [[name, name] for name in ("GET", "POST", "PUT", "PATCH", "DELETE")]},
            {"key": "headers", "label": "Headers", "type": "json", "rows": 2},
            {"key": "body", "label": "Body", "type": "json", "rows": 3},
            {"key": "extract", "label": "Return", "type": "select",
             "options": [["raw", "Raw body"], ["text", "Readable text (HTML stripped)"]]},
            {"key": "max_chars", "label": "Max characters of text", "type": "number",
             "min": 100, "max": 200000, "step": 100},
            {"key": "timeout", "label": "Timeout (seconds)", "type": "number", "min": 1, "max": 30, "step": 1},
        ],
    },
    "news.headlines": {
        "label": "News headlines", "icon": "🗞",
        "description": ("Read a public RSS/Atom news feed and return dated headlines with "
                        "sources and links. Defaults to Google News; set region=IN for India."),
        "danger": False,
        "config": {"url": "", "query": "", "region": "IN", "language": "en",
                   "source": "auto", "limit": 10, "date": "",
                   "include_summary": False, "timeout": 15, "allow_private": False},
        "params": [
            {"key": "url", "label": "Feed URL", "type": "template", "rows": 1,
             "help": "Empty uses Google News for the region and query."},
            {"key": "query", "label": "Search or topic", "type": "text"},
            {"key": "source", "label": "Feed source", "type": "select",
             "options": [["auto", "Auto (Google, then Bing)"], ["google", "Google News"],
                         ["bing", "Bing News"]]},
            {"key": "region", "label": "Region", "type": "text",
             "help": "ISO country code such as IN or US; empty uses the feed default."},
            {"key": "language", "label": "Language", "type": "text",
             "help": "Language code such as en or hi."},
            {"key": "limit", "label": "Headlines", "type": "number", "min": 1, "max": 50},
            {"key": "date", "label": "Only this date", "type": "text",
             "help": "YYYY-MM-DD. If no item matches, the newest items are returned with a note."},
            {"key": "include_summary", "label": "Include summaries", "type": "boolean",
             "help": "Off by default: titles, sources, dates and links cost far fewer tokens."},
            {"key": "timeout", "label": "Timeout (seconds)", "type": "number",
             "min": 1, "max": 30},
        ],
    },
    "browser.fetch": {
        "label": "Headless browser", "icon": "🌐",
        "description": ("Render a page in a headless browser and return its visible text. "
                        "Requires OS access; uses Playwright or Chromium and falls back "
                        "to HTTP text when neither is installed."),
        "danger": True,
        "config": {"url": "", "timeout": 25, "max_chars": 20_000, "allow_private": False},
        "params": [
            {"key": "url", "label": "URL", "type": "template", "rows": 1},
            {"key": "timeout", "label": "Timeout (seconds)", "type": "number",
             "min": 5, "max": 60, "step": 1},
            {"key": "max_chars", "label": "Maximum text characters", "type": "number",
             "min": 500, "max": 200_000, "step": 500},
        ],
    },
    "file.read": {
        "label": "Read file", "icon": "▤",
        "description": "Read a UTF-8 file from the agent sandbox, in line ranges.",
        "danger": False, "config": {},
        "params": [
            {"key": "path", "label": "Path", "type": "template", "rows": 1},
            {"key": "start_line", "label": "First line", "type": "number",
             "min": 1, "max": 1000000, "step": 1},
            {"key": "end_line", "label": "Last line", "type": "number",
             "min": 1, "max": 1000000, "step": 1,
             "help": "Defaults to 200 lines from start_line."},
        ],
    },
    "file.write": {
        "label": "Write file", "icon": "▥",
        "description": "Write a UTF-8 file inside the agent sandbox.",
        "danger": False, "config": {},
        "params": [
            {"key": "path", "label": "Path", "type": "template", "rows": 1},
            {"key": "content", "label": "Content", "type": "template", "rows": 4},
        ],
    },
    "file.list": {
        "label": "List files", "icon": "≡",
        "description": "List sandbox files with an optional glob pattern.",
        "danger": False, "config": {"pattern": "*"},
        "params": [{"key": "pattern", "label": "Pattern", "type": "text"}],
    },
    "file.edit": {
        "label": "Edit file", "icon": "✎",
        "description": "Replace one exact, unique block of text in a sandbox file.",
        "danger": False, "config": {},
        "params": [
            {"key": "path", "label": "Path", "type": "template", "rows": 1},
            {"key": "old", "label": "Replace this text", "type": "template", "rows": 4},
            {"key": "new", "label": "With this text", "type": "template", "rows": 4},
            {"key": "replace_all", "label": "Replace every occurrence", "type": "boolean"},
        ],
    },
    "file.search": {
        "label": "Search files", "icon": "⌕",
        "description": "Search sandbox text files by regular expression or literal text.",
        "danger": False, "config": {"regex": True, "case_sensitive": False, "include": ""},
        "params": [
            {"key": "query", "label": "Search for", "type": "template", "rows": 1},
            {"key": "regex", "label": "Treat as a regular expression", "type": "boolean"},
            {"key": "case_sensitive", "label": "Match case", "type": "boolean"},
            {"key": "include", "label": "File glob", "type": "text",
             "help": "For example *.py; empty searches every text file."},
            {"key": "max_results", "label": "Maximum matches", "type": "number",
             "min": 1, "max": 500, "step": 1},
        ],
    },
    "file.tree": {
        "label": "File tree", "icon": "▤",
        "description": "List the sandbox project tree with file sizes.",
        "danger": False, "config": {"max_entries": 600},
        "params": [
            {"key": "max_entries", "label": "Maximum entries", "type": "number",
             "min": 1, "max": 2000, "step": 1},
        ],
    },
    "file.delete": {
        "label": "Delete file", "icon": "⌫",
        "description": "Delete a sandbox file. The IDE snapshots it first.",
        "danger": True, "config": {},
        "params": [{"key": "path", "label": "Path", "type": "template", "rows": 1}],
    },
    "memory": {
        "label": "Agent memory", "icon": "▤",
        "description": "Persistent key/value memory for this agent.",
        "danger": False, "config": {},
        "params": [
            {"key": "action", "label": "Action", "type": "select",
             "options": [["get", "Get"], ["set", "Set"], ["list", "List"], ["delete", "Delete"]]},
            {"key": "key", "label": "Key", "type": "template", "rows": 1},
            {"key": "value", "label": "Value", "type": "template", "rows": 2},
        ],
    },
    "math.eval": {
        "label": "Math / expression", "icon": "∑",
        "description": "Evaluate a sandboxed arithmetic expression.",
        "danger": False, "config": {},
        "params": [{"key": "expression", "label": "Expression", "type": "expression", "rows": 1}],
    },
    "time.now": {
        "label": "Current time", "icon": "◷",
        "description": "Return the current local and UTC time.",
        "danger": False, "config": {}, "params": [],
    },
    "code.diagnostics": {
        "label": "Code diagnostics", "icon": "⚕",
        "description": "Check Python syntax (and pyflakes rules when installed) and JSON validity.",
        "danger": False, "config": {},
        "params": [{"key": "path", "label": "File or folder", "type": "text",
                   "help": "Empty checks the whole sandbox."}],
    },
    "sqlite.query": {
        "label": "SQLite query", "icon": "▦",
        "description": "Run one parameterized SQL statement against a SQLite database inside the sandbox. "
                       "Read-only unless the ability config sets write to true.",
        "danger": False, "config": {"path": "", "write": False, "timeout": 5, "max_rows": 200},
        "params": [
            {"key": "path", "label": "Database file", "type": "template", "rows": 1},
            {"key": "sql", "label": "SQL statement", "type": "template", "rows": 5},
            {"key": "params", "label": "Bound parameters", "type": "json", "rows": 2,
             "help": "A JSON list or object bound to ? or :name placeholders."},
            {"key": "max_rows", "label": "Maximum rows", "type": "number",
             "min": 1, "max": 5000, "step": 1},
            {"key": "timeout", "label": "Timeout (seconds)", "type": "number",
             "min": 1, "max": 30, "step": 1},
        ],
    },
    "csv.read": {
        "label": "Read CSV / TSV", "icon": "▤",
        "description": "Read a delimited text file from the sandbox as columns and records.",
        "danger": False, "config": {"delimiter": "", "header": True, "max_rows": 1000},
        "params": [
            {"key": "path", "label": "Path", "type": "template", "rows": 1},
            {"key": "delimiter", "label": "Delimiter", "type": "text",
             "help": "Leave empty to detect from the file name; use \\t for tabs."},
            {"key": "header", "label": "First row is a header", "type": "boolean"},
            {"key": "max_rows", "label": "Maximum rows", "type": "number",
             "min": 1, "max": 10000, "step": 1},
        ],
    },
    "git.inspect": {
        "label": "Inspect Git", "icon": "⑂",
        "description": "Read-only git status, diff or log for a project folder inside the sandbox.",
        "danger": False, "config": {"action": "status", "limit": 30, "max_chars": 12000},
        "params": [
            {"key": "action", "label": "Action", "type": "select",
             "options": [["status", "Status"], ["diff", "Diff"], ["log", "Recent commits"]]},
            {"key": "path", "label": "Path filter", "type": "text",
             "help": "Optional workspace-relative file or folder."},
            {"key": "staged", "label": "Diff the index", "type": "boolean"},
            {"key": "limit", "label": "Maximum commits", "type": "number",
             "min": 1, "max": 200, "step": 1},
            {"key": "max_chars", "label": "Maximum output characters", "type": "number",
             "min": 500, "max": 50000, "step": 500},
        ],
    },
    "shell.exec": {
        "label": "Shell command", "icon": "⌘",
        "description": ("Run a command on this machine. Requires OS access "
                        "(allow_os_access or MARIANA_AGENT_ALLOW_OS / ALLOW_SHELL)."),
        "danger": True, "config": {"timeout": 30},
        "params": [
            {"key": "command", "label": "Command", "type": "template", "rows": 2},
            {"key": "timeout", "label": "Timeout (seconds)", "type": "number", "min": 1, "max": 120, "step": 1},
        ],
    },
}

DEFAULT_NODE_DATA = {
    "start": {},
    "end": {"output": "{{ last }}"},
    "llm": {"model": "", "system": "", "prompt": "", "format": "text",
            "temperature": 0.7, "max_tokens": 512},
    "agent": {"model": "", "system": "", "goal": "", "abilities": [], "max_iterations": 6,
              "temperature": 0.3, "max_tokens": 1024},
    "swarm": {"model": "", "leader_model": "", "system": "", "goal": "",
              "agents": [], "tools": [], "strategy": "auto", "rounds": 1,
              "max_agents": 4, "max_steps": 6, "max_seconds": 240,
              "write": False, "memory": True, "ipc": True},
    "tool": {"ability": "", "input": {}},
    "condition": {"expression": "true"},
    "transform": {"template": "", "format": "text", "assign": {}},
    "delay": {"seconds": 1},
    "human": {"prompt": "", "kind": "approve", "choices": [], "default": "", "timeout": 3600},
    "subagent": {"agent": "", "input": "{{ last }}"},
    "foreach": {"items": "{{ last }}", "agent": "", "input": "{{ item }}", "max_items": 20},
    "memory": {"action": "set", "key": "", "value": ""},
    "note": {"text": ""},
}
HUMAN_KINDS = ("approve", "text", "choice")
# Node types that must lead somewhere; ``end`` and ``condition`` are checked separately.
_FLOWING_TYPES = frozenset(NODE_TYPES) - {"end", "condition"}

DEFAULT_SETTINGS = {
    "max_steps": 100,
    "timeout": 120,
    "on_error": "stop",
    "max_output_chars": 60000,
}

DEFAULT_SWARM = {
    "max_agents": 6,
    "max_steps": 8,
    "max_rounds": 2,
    "max_seconds": 300,
    "max_messages": 200,
    "max_message_chars": 4000,
    "max_memory_entries": 120,
    "max_memory_chars": 8000,
    "max_context_chars": 200000,
    "max_agent_context_chars": 64000,
    "max_context_item_chars": 24000,
    "allow_write": False,
    "allow_ipc": True,
    "shared_memory": True,
}


class AMLValidationError(ValueError):
    def __init__(self, errors, warnings=None):
        self.errors = errors if isinstance(errors, list) else [errors]
        self.warnings = warnings or []
        super().__init__("; ".join(self.errors))


def slugify(value: str, fallback: str = "agent") -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-")
    return slug or fallback


def new_id(prefix: str = "n") -> str:
    return f"{prefix}-{int(time.time() * 1000):x}-{int.from_bytes(os.urandom(2), 'big'):x}"


def parse_document(text: str) -> dict:
    if len(text.encode("utf-8")) > MAX_DOCUMENT_BYTES:
        raise AMLValidationError("AML document exceeds 1 MB")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # optional
        except ImportError as error:
            raise AMLValidationError(
                "Invalid JSON; install PyYAML to import YAML AML"
            ) from error
        try:
            value = yaml.safe_load(text)
        except Exception as error:  # noqa: BLE001 - YAML parser errors vary
            raise AMLValidationError(f"Invalid JSON or YAML: {error}") from error
    if not isinstance(value, dict):
        raise AMLValidationError("AML document must be an object")
    return value


def normalize_agent(document: dict) -> dict:
    if not isinstance(document, dict):
        raise AMLValidationError("Agent document must be an object")
    name = str(document.get("name") or "Untitled agent").strip()[:120] or "Untitled agent"
    agent_id = slugify(document.get("id") or name)
    abilities = []
    seen_abilities = set()
    for index, item in enumerate(document.get("abilities") or []):
        if not isinstance(item, dict):
            continue
        ability_type = str(item.get("type") or "").strip()
        ability_id = slugify(item.get("id") or f"{ability_type}-{index}", "ability")
        if ability_id in seen_abilities:
            raise AMLValidationError(f"Duplicate ability id: {ability_id}")
        seen_abilities.add(ability_id)
        defaults = ABILITY_TYPES.get(ability_type, {}).get("config", {})
        config = dict(defaults)
        if isinstance(item.get("config"), dict):
            config.update(item["config"])
        abilities.append({
            "id": ability_id,
            "type": ability_type,
            "label": str(item.get("label") or ABILITY_TYPES.get(ability_type, {}).get("label") or ability_type),
            "enabled": item.get("enabled", True) is not False,
            "config": config,
        })
    nodes = []
    node_ids = set()
    for index, item in enumerate(document.get("nodes") or []):
        if not isinstance(item, dict):
            continue
        node_type = str(item.get("type") or "").strip()
        node_id = slugify(item.get("id") or f"{node_type}-{index}", "node")
        if node_id in node_ids:
            raise AMLValidationError(f"Duplicate node id: {node_id}")
        node_ids.add(node_id)
        data = dict(DEFAULT_NODE_DATA.get(node_type, {}))
        if isinstance(item.get("data"), dict):
            data.update(item["data"])
        nodes.append({
            "id": node_id,
            "type": node_type,
            "label": str(item.get("label") or NODE_TYPES.get(node_type, {}).get("label") or node_type)[:80],
            "x": _number(item.get("x"), 80.0),
            "y": _number(item.get("y"), 80.0),
            "data": data,
        })
    edges = []
    seen_edges = set()
    for item in document.get("edges") or []:
        if not isinstance(item, dict):
            continue
        source, target = str(item.get("from") or ""), str(item.get("to") or "")
        when = str(item.get("when") or "always")
        key = (source, target, when)
        if key in seen_edges:
            continue
        seen_edges.add(key)
        edge = {"from": source, "to": target, "when": when}
        if item.get("label"):
            edge["label"] = str(item["label"])[:80]
        edges.append(edge)
    settings = dict(DEFAULT_SETTINGS)
    if isinstance(document.get("settings"), dict):
        settings.update(document["settings"])
    schedule = document.get("schedule") if isinstance(document.get("schedule"), dict) else {}
    schedule = {
        "enabled": bool(schedule.get("enabled", False)),
        "type": str(schedule.get("type") or "manual"),
        "interval_minutes": _number(schedule.get("interval_minutes"), 60.0),
        "cron": str(schedule.get("cron") or ""),
        "once_at": str(schedule.get("once_at") or ""),
        "input": schedule.get("input") if isinstance(schedule.get("input"), dict) else {},
    }
    trigger = document.get("trigger") if isinstance(document.get("trigger"), dict) else {}
    return {
        "aml": AML_VERSION,
        "kind": "agent",
        "id": agent_id,
        "name": name,
        "description": str(document.get("description") or "")[:2000],
        "version": str(document.get("version") or "1.0.0")[:40],
        "icon": str(document.get("icon") or "◈")[:4],
        "model": str(document.get("model") or "").strip(),
        "model_settings": document.get("model_settings") if isinstance(document.get("model_settings"), dict) else {},
        "vars": _initial_vars(document.get("vars")),
        "swarm": _swarm_policy(document.get("swarm")),
        "settings": {
            "max_steps": int(_number(settings.get("max_steps"), 100.0)),
            "timeout": int(_number(settings.get("timeout"), 120.0)),
            "on_error": str(settings.get("on_error") or "stop"),
            "max_output_chars": int(_number(settings.get("max_output_chars"), 60000.0)),
        },
        "abilities": abilities,
        "nodes": nodes,
        "edges": edges,
        "trigger": {"type": str(trigger.get("type") or "manual")},
        "schedule": schedule,
        "created": str(document.get("created") or "") or datetime.now().isoformat(timespec="seconds"),
        "updated": datetime.now().isoformat(timespec="seconds"),
    }


def _number(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _initial_vars(value):
    """Top-level starting variables: plain JSON, bounded so a document stays portable."""
    if not isinstance(value, dict):
        return {}
    result = {}
    for key, item in list(value.items())[:100]:
        name = str(key).strip()[:80]
        if not name:
            continue
        try:
            result[name] = json.loads(json.dumps(item, ensure_ascii=False, default=str))
        except (TypeError, ValueError):
            result[name] = str(item)
    return result


def _swarm_policy(value) -> dict:
    """Document-wide swarm defaults and security caps; a swarm node may narrow them."""
    result = dict(DEFAULT_SWARM)
    if not isinstance(value, dict):
        return result
    for key in ("max_agents", "max_steps", "max_rounds", "max_seconds",
                "max_messages", "max_message_chars", "max_memory_entries",
                "max_memory_chars", "max_context_chars", "max_agent_context_chars",
                "max_context_item_chars"):
        result[key] = int(_number(value.get(key), float(DEFAULT_SWARM[key])))
    for key in ("allow_write", "allow_ipc", "shared_memory"):
        result[key] = value.get(key, DEFAULT_SWARM[key]) is True
    return result


def validate_agent(document: dict, known_agent_ids: set[str] | None = None) -> dict:
    """Normalize and check an AML document.

    Returns ``{"ok", "errors", "warnings", "issues", "agent"}``.  ``issues`` is the
    structured form — each entry names the node, edge or field it concerns so a
    designer can pin it to the canvas; ``errors``/``warnings`` are the same
    messages as plain strings.
    """
    agent = normalize_agent(document)
    issues: list[dict] = []

    def problem(level: str, message: str, node=None, edge=None, field=None) -> None:
        issues.append({"level": level, "message": message, "node": node, "edge": edge, "field": field})

    def error(message, **where):
        problem("error", message, **where)

    def warning(message, **where):
        problem("warning", message, **where)

    declared = str(document.get("aml") or AML_VERSION)
    if declared not in SUPPORTED_VERSIONS:
        warning(f"Document declares AML {declared}; this runtime implements {AML_VERSION}")
    raw_vars = document.get("vars")
    if raw_vars is not None and not isinstance(raw_vars, dict):
        error("vars must be a JSON object", field="vars")
    elif isinstance(raw_vars, dict):
        if len(raw_vars) > 100:
            error("vars may contain at most 100 keys", field="vars")
        for key in raw_vars:
            if not str(key).strip() or len(str(key)) > 80:
                error("Every vars key must be 1-80 characters", field="vars")
                break
    if not agent["name"]:
        error("Agent name is required")
    if agent["settings"]["on_error"] not in {"stop", "continue"}:
        error("settings.on_error must be stop or continue")
    if not 1 <= agent["settings"]["max_steps"] <= 1000:
        error("settings.max_steps must be between 1 and 1000")
    if not 1 <= agent["settings"]["timeout"] <= 3600:
        error("settings.timeout must be between 1 and 3600 seconds")
    raw_swarm = document.get("swarm")
    if raw_swarm is not None and not isinstance(raw_swarm, dict):
        error("swarm must be a JSON object", field="swarm")
    policy = agent["swarm"]
    for key, low, high in (("max_agents", 1, 12), ("max_steps", 1, 24),
                           ("max_rounds", 0, 4), ("max_seconds", 10, 3600),
                           ("max_messages", 0, 1000), ("max_message_chars", 200, 20000),
                           ("max_memory_entries", 0, 1000), ("max_memory_chars", 200, 100000),
                           ("max_context_chars", 0, 2_000_000),
                           ("max_agent_context_chars", 0, 500_000),
                           ("max_context_item_chars", 500, 200_000)):
        if not low <= policy[key] <= high:
            error(f"swarm.{key} must be between {low} and {high}", field="swarm")
    if isinstance(raw_swarm, dict):
        for key in ("allow_write", "allow_ipc", "shared_memory"):
            if key in raw_swarm and not isinstance(raw_swarm[key], bool):
                error(f"swarm.{key} must be true or false", field="swarm")
    ability_ids = {item["id"] for item in agent["abilities"]}
    ability_types = {item["id"]: item["type"] for item in agent["abilities"]}
    for ability in agent["abilities"]:
        if ability["type"] not in ABILITY_TYPES:
            error(f"Ability {ability['id']} has unknown type {ability['type']}")
    nodes = {node["id"]: node for node in agent["nodes"]}
    if not nodes:
        error("Agent needs at least one node")
    starts = [node for node in agent["nodes"] if node["type"] == "start"]
    ends = [node for node in agent["nodes"] if node["type"] == "end"]
    if len(starts) != 1:
        error("Agent must have exactly one start node")
    if not ends:
        error("Agent needs at least one end node")
    known_agent_ids = known_agent_ids or set()
    for node in agent["nodes"]:
        node_type, node_id = node["type"], node["id"]
        if node_type not in NODE_TYPES:
            error(f"Node {node_id} has unknown type {node_type}", node=node_id)
            continue
        data = node["data"]
        if node_type == "tool":
            if not data.get("ability"):
                error(f"Ability node {node_id} needs an ability id", node=node_id, field="ability")
            elif data["ability"] not in ability_ids:
                error(f"Node {node_id} references missing ability {data['ability']}",
                      node=node_id, field="ability")
        if node_type in {"subagent", "foreach"}:
            what = "Agent" if node_type == "subagent" else "For-each"
            if not data.get("agent"):
                error(f"{what} node {node_id} needs a target agent id", node=node_id, field="agent")
            elif data["agent"] == agent["id"]:
                error(f"{what} node {node_id} cannot call itself", node=node_id, field="agent")
            elif known_agent_ids and data["agent"] not in known_agent_ids:
                warning(f"{what} node {node_id} references agent {data['agent']} which does not exist yet",
                        node=node_id, field="agent")
        if node_type == "foreach":
            try:
                if not 1 <= int(data.get("max_items", 20)) <= 200:
                    error(f"For-each node {node_id} max_items must be between 1 and 200",
                          node=node_id, field="max_items")
            except (TypeError, ValueError):
                error(f"For-each node {node_id} max_items must be a number", node=node_id, field="max_items")
        if node_type == "condition":
            try:
                sample = _sample_context()
                sample["nodes"] = {
                    other: {"text": "", "result": False, "value": None, "output": ""} for other in nodes
                }
                sample["vars"] = {"example": True}
                evaluate_expression(str(data.get("expression") or "true"), sample, lenient=True)
            except Exception as failure:  # noqa: BLE001
                error(f"Condition node {node_id}: {failure}", node=node_id, field="expression")
        if node_type == "delay":
            try:
                if not 0 <= float(data.get("seconds", 0)) <= 300:
                    error(f"Delay node {node_id} must be between 0 and 300 seconds",
                          node=node_id, field="seconds")
            except (TypeError, ValueError):
                error(f"Delay node {node_id} seconds must be a number", node=node_id, field="seconds")
        if node_type == "human":
            kind = data.get("kind", "approve")
            if kind not in HUMAN_KINDS:
                error(f"Human node {node_id} kind must be approve, text, or choice",
                      node=node_id, field="kind")
            choices = data.get("choices")
            if kind == "choice" and not (isinstance(choices, list) and choices):
                error(f"Human node {node_id} needs a list of choices", node=node_id, field="choices")
            try:
                if not 0 <= float(data.get("timeout", 3600)) <= 86400:
                    error(f"Human node {node_id} timeout must be between 0 and 86400 seconds",
                          node=node_id, field="timeout")
            except (TypeError, ValueError):
                error(f"Human node {node_id} timeout must be a number", node=node_id, field="timeout")
        if node_type == "agent":
            try:
                if not 1 <= int(data.get("max_iterations", 6)) <= 40:
                    error(f"AI agent node {node_id} max_iterations must be between 1 and 40",
                          node=node_id, field="max_iterations")
            except (TypeError, ValueError):
                error(f"AI agent node {node_id} max_iterations must be a number",
                      node=node_id, field="max_iterations")
            chosen = data.get("abilities") or []
            if not isinstance(chosen, list):
                error(f"AI agent node {node_id} abilities must be a list of ability ids",
                      node=node_id, field="abilities")
                chosen = []
            for ability_id in chosen:
                if ability_id not in ability_ids:
                    error(f"Node {node_id} references missing ability {ability_id}",
                          node=node_id, field="abilities")
            if not ability_ids:
                warning(f"AI agent node {node_id} has no abilities to use; it will answer like a plain LLM call",
                        node=node_id, field="abilities")
        if node_type == "swarm":
            _validate_swarm_node(node, data, ability_types, policy, error, warning)
        if node_type in {"llm", "agent", "swarm"} and not agent["model"] and not str(data.get("model") or ""):
            label = {"llm": "LLM", "agent": "AI agent", "swarm": "Swarm"}[node_type]
            warning(f"{label} node {node_id} has no model; the run will use the first available chat model",
                    node=node_id, field="model")
        if node_type == "llm" and data.get("format", "text") not in {"text", "json"}:
            error(f"LLM node {node_id} format must be text or json", node=node_id, field="format")
    node_ids = set(nodes)
    for edge in agent["edges"]:
        pair = [edge["from"], edge["to"]]
        if edge["from"] not in node_ids:
            error(f"Edge starts at missing node {edge['from']}", edge=pair)
        if edge["to"] not in node_ids:
            error(f"Edge points at missing node {edge['to']}", edge=pair)
        if edge["when"] not in {"always", "true", "false"}:
            error(f"Edge {edge['from']}→{edge['to']} has invalid when={edge['when']}", edge=pair)
    for node in agent["nodes"]:
        outgoing = [edge for edge in agent["edges"] if edge["from"] == node["id"]]
        if node["type"] == "condition":
            branches = {edge["when"] for edge in outgoing}
            if "true" not in branches or "false" not in branches:
                warning(f"Condition {node['id']} does not define both true and false branches",
                        node=node["id"])
        elif node["type"] in _FLOWING_TYPES:
            if not outgoing:
                warning(f"Node {node['id']} has no outgoing edge", node=node["id"])
            elif len(outgoing) > 1:
                warning(
                    f"Node {node['id']} has {len(outgoing)} outgoing edges; only the first "
                    f"(to {outgoing[0]['to']}) is followed. Use a condition to branch",
                    node=node["id"],
                )
    reachable = _reachable(nodes, agent["edges"], starts[0]["id"] if starts else None)
    for node in agent["nodes"]:
        if node["id"] not in reachable:
            warning(f"Node {node['id']} is unreachable from start", node=node["id"])
    if _has_cycle(nodes, agent["edges"]):
        warning("The flow contains a cycle; max_steps protects the run but make sure it terminates")
    if agent["schedule"]["enabled"]:
        if agent["schedule"]["type"] == "manual":
            error("Enable a schedule type (interval, cron, or once) or disable the schedule",
                  field="schedule")
        for message in _validate_schedule(agent["schedule"]):
            error(message, field="schedule")
    if any(ability["type"] == "shell.exec" for ability in agent["abilities"]):
        warning("This agent requests shell access; it stays disabled unless MARIANA_AGENT_ALLOW_SHELL=1")
    errors = [issue["message"] for issue in issues if issue["level"] == "error"]
    warnings = [issue["message"] for issue in issues if issue["level"] == "warning"]
    return {"ok": not errors, "errors": errors, "warnings": warnings, "issues": issues, "agent": agent}


def _validate_swarm_node(node: dict, data: dict, ability_types: dict[str, str], policy: dict,
                         error, warning) -> None:
    """Check a swarm node against the language and the document-wide swarm policy."""
    node_id = node["id"]
    strategy = str(data.get("strategy") or "auto")
    if strategy not in SWARM_STRATEGIES:
        error(f"Swarm node {node_id} strategy must be one of {', '.join(SWARM_STRATEGIES)}",
              node=node_id, field="strategy")
    try:
        rounds = int(data.get("rounds", 1))
        if not 0 <= rounds <= 4:
            error(f"Swarm node {node_id} rounds must be between 0 and 4",
                  node=node_id, field="rounds")
        elif rounds > policy["max_rounds"]:
            warning(f"Swarm node {node_id} asks for {rounds} feedback rounds; "
                    f"swarm.max_rounds caps it at {policy['max_rounds']}",
                    node=node_id, field="rounds")
    except (TypeError, ValueError):
        error(f"Swarm node {node_id} rounds must be a number", node=node_id, field="rounds")
    for key, low, high in (("max_agents", 1, 8), ("max_steps", 1, 12), ("max_seconds", 10, 1800)):
        try:
            value = int(data.get(key, DEFAULT_NODE_DATA["swarm"][key]))
            if not low <= value <= high:
                error(f"Swarm node {node_id} {key} must be between {low} and {high}",
                      node=node_id, field=key)
            elif key == "max_agents" and value > policy["max_agents"]:
                warning(f"Swarm node {node_id} may use at most {policy['max_agents']} agents "
                        "because of the document's swarm policy", node=node_id, field=key)
            elif key == "max_steps" and value > policy["max_steps"]:
                warning(f"Swarm node {node_id} max_steps is capped at {policy['max_steps']} "
                        "by the document's swarm policy", node=node_id, field=key)
            elif key == "max_seconds" and value > policy["max_seconds"]:
                warning(f"Swarm node {node_id} time budget is capped at {policy['max_seconds']} s "
                        "by the document's swarm policy", node=node_id, field=key)
        except (TypeError, ValueError):
            error(f"Swarm node {node_id} {key} must be a number", node=node_id, field=key)
    for key in ("write", "memory", "ipc"):
        if key in data and not isinstance(data[key], bool):
            error(f"Swarm node {node_id} {key} must be true or false", node=node_id, field=key)
    tools = data.get("tools") or []
    if not isinstance(tools, list):
        error(f"Swarm node {node_id} tools must be a list of ability ids",
              node=node_id, field="tools")
        tools = []
    for ability_id in tools:
        if ability_id != "*" and ability_id not in ability_types:
            error(f"Node {node_id} references missing ability {ability_id}",
                  node=node_id, field="tools")
    write = data.get("write") is True
    if write and not policy["allow_write"]:
        error(f"Swarm node {node_id} asks to edit files, but swarm.allow_write is false; "
              "set it to true at the top of the document or disable write",
              node=node_id, field="write")
    if write:
        chosen = list(ability_types) if "*" in tools else tools if tools else list(ability_types)
        write_abilities = {ability_id for ability_id in chosen
                           if ability_types.get(ability_id) in
                           {"file.write", "file.edit", "file.delete", "shell.exec"}}
        if not write_abilities:
            warning(f"Swarm node {node_id} allows writing, but no write ability is granted",
                    node=node_id, field="write")
    agents = data.get("agents") or []
    if isinstance(agents, str):
        try:
            agents = json.loads(agents)
        except json.JSONDecodeError:
            if "{{" in agents:
                agents = []  # rendered from the flow context at run time
            else:
                agents = None
    if agents is not None and not isinstance(agents, list):
        error(f"Swarm node {node_id} agents must be a JSON list", node=node_id, field="agents")
        agents = []
    if len(agents) > 12:
        error(f"Swarm node {node_id} may define at most 12 agents", node=node_id, field="agents")
    for index, item in enumerate(agents[:12]):
        if isinstance(item, str):
            if not item.strip():
                error(f"Swarm node {node_id} agent {index + 1} has an empty task",
                      node=node_id, field="agents")
        elif isinstance(item, dict):
            if not str(item.get("task") or item.get("goal") or "").strip():
                error(f"Swarm node {node_id} agent {index + 1} needs a task",
                      node=node_id, field="agents")
        else:
            error(f"Swarm node {node_id} agent {index + 1} must be a task string or an object",
                  node=node_id, field="agents")
    if data.get("ipc") is False:
        warning(f"Swarm node {node_id} disables agent-to-agent messages; members only share the blackboard",
                node=node_id, field="ipc")


def _validate_schedule(schedule: dict) -> list[str]:
    kind = schedule.get("type")
    errors = []
    if kind == "interval":
        if not 1 <= float(schedule.get("interval_minutes", 0)) <= 525600:
            errors.append("Schedule interval must be between 1 minute and 1 year")
    elif kind == "cron":
        try:
            parse_cron(schedule.get("cron", ""))
        except ValueError as error:
            errors.append(str(error))
    elif kind == "once":
        try:
            datetime.fromisoformat(schedule.get("once_at", ""))
        except ValueError:
            errors.append("Schedule once_at must be an ISO date-time")
    elif kind not in {"manual", "interval", "cron", "once"}:
        errors.append("Schedule type must be manual, interval, cron, or once")
    return errors


def _reachable(nodes: dict, edges: list[dict], start_id: str | None) -> set[str]:
    if not start_id:
        return set()
    outgoing: dict[str, list[str]] = {}
    for edge in edges:
        outgoing.setdefault(edge["from"], []).append(edge["to"])
    seen, stack = set(), [start_id]
    while stack:
        node = stack.pop()
        if node in seen or node not in nodes:
            continue
        seen.add(node)
        stack.extend(outgoing.get(node, []))
    return seen


def _has_cycle(nodes: dict, edges: list[dict]) -> bool:
    outgoing: dict[str, list[str]] = {}
    for edge in edges:
        outgoing.setdefault(edge["from"], []).append(edge["to"])
    visiting, done = set(), set()

    def visit(node):
        if node in visiting:
            return True
        if node in done:
            return False
        visiting.add(node)
        if any(visit(child) for child in outgoing.get(node, [])):
            return True
        visiting.discard(node)
        done.add(node)
        return False

    return any(visit(node) for node in nodes)


# ---------------------------------------------------------------------------
# Safe expressions and templates
# ---------------------------------------------------------------------------
class ExpressionError(ValueError):
    pass


def _stringify(value) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(value, ensure_ascii=False)


def _matches(value, pattern) -> bool:
    if len(str(pattern)) > 300:
        raise ExpressionError("Regular expression is too long")
    return re.search(str(pattern), str(value)) is not None


def _get_path(value, path: str, default=None):
    current = value
    for part in str(path).split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, (list, tuple)) and part.isdigit():
            try:
                current = current[int(part)]
            except IndexError:
                return default
        else:
            return default
    return current


_HELPERS = {
    "contains": lambda value, needle: str(needle) in str(value),
    "startswith": lambda value, prefix: str(value).startswith(str(prefix)),
    "endswith": lambda value, suffix: str(value).endswith(str(suffix)),
    "lower": lambda value: str(value).lower(),
    "upper": lambda value: str(value).upper(),
    "trim": lambda value: str(value).strip(),
    "split": lambda value, sep=None: str(value).split(sep),
    "join": lambda values, sep="": str(sep).join(str(item) for item in values),
    "len": lambda value: len(value),
    "length": lambda value: len(value),
    "str": lambda value: _stringify(value),
    "int": lambda value: int(value),
    "float": lambda value: float(value),
    "bool": lambda value: bool(value),
    "json": lambda value: json.loads(value) if isinstance(value, str) else value,
    "dumps": lambda value: json.dumps(value, ensure_ascii=False),
    "coalesce": lambda *values: next((value for value in values if value not in (None, "")), None),
    "matches": _matches,
    "get": _get_path,
    "now": lambda: datetime.now().isoformat(timespec="seconds"),
    "today": lambda: datetime.now().date().isoformat(),
    "min": min,
    "max": max,
    "round": round,
    "abs": abs,
    "floor": math.floor,
    "ceil": math.ceil,
}

_ALLOWED_METHODS = {
    str: {"lower", "upper", "strip", "lstrip", "rstrip", "startswith", "endswith",
          "split", "replace", "find", "count", "join"},
    dict: {"get", "keys", "values", "items"},
    list: {"count", "index"},
}


def _sample_context() -> dict:
    return {
        "input": {"message": "hello", "value": 1},
        "vars": {"example": True},
        "last": "hello",
        "nodes": {"start": {"message": "hello"}},
        "agent": {"id": "example", "name": "Example"},
        "now": "2026-01-01T00:00:00",
    }


class _SafeEvaluator(ast.NodeVisitor):
    def __init__(self, context: dict, lenient: bool = False):
        self.context = context
        self.helpers = _HELPERS
        self.lenient = lenient

    def evaluate(self, expression: str):
        if len(expression) > 800:
            raise ExpressionError("Expression is too long (max 800 characters)")
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as error:
            raise ExpressionError(f"Invalid expression: {error.msg}") from error
        return self.visit(tree.body)

    def generic_visit(self, node):
        raise ExpressionError(f"Unsupported expression element: {type(node).__name__}")

    def visit_Constant(self, node):
        return node.value

    def visit_Name(self, node):
        if node.id in {"true", "false", "null"}:
            return {"true": True, "false": False, "null": None}[node.id]
        if node.id in self.context:
            return self.context[node.id]
        if node.id in self.helpers:
            return self.helpers[node.id]
        if self.lenient:
            return None
        raise ExpressionError(f"Unknown name: {node.id}")

    def visit_List(self, node):
        return [self.visit(item) for item in node.elts]

    def visit_Tuple(self, node):
        return tuple(self.visit(item) for item in node.elts)

    def visit_Dict(self, node):
        return {self.visit(key): self.visit(value) for key, value in zip(node.keys, node.values)}

    def visit_Attribute(self, node):
        value = self.visit(node.value)
        if isinstance(value, dict) and node.attr in value:
            return value[node.attr]
        if self.lenient:
            return None
        raise ExpressionError(f"Cannot read .{node.attr} from {type(value).__name__}")

    def visit_Subscript(self, node):
        value = self.visit(node.value)
        key = self.visit(node.slice)
        try:
            return value[key]
        except (KeyError, IndexError, TypeError) as error:
            if self.lenient:
                return None
            raise ExpressionError(f"Invalid index: {error}") from error

    def visit_Slice(self, node):
        return slice(
            self.visit(node.lower) if node.lower else None,
            self.visit(node.upper) if node.upper else None,
            self.visit(node.step) if node.step else None,
        )

    def visit_UnaryOp(self, node):
        value = self.visit(node.operand)
        if isinstance(node.op, ast.Not):
            return not value
        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.UAdd):
            return +value
        raise ExpressionError("Unsupported unary operator")

    def visit_BinOp(self, node):
        left, right = self.visit(node.left), self.visit(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.FloorDiv):
            return left // right
        if isinstance(node.op, ast.Mod):
            return left % right
        if isinstance(node.op, ast.Pow):
            if abs(right) > 1000:
                raise ExpressionError("Exponent is too large")
            return left ** right
        raise ExpressionError("Unsupported binary operator")

    def visit_BoolOp(self, node):
        """``and``/``or`` as in Python: short-circuit and return the deciding operand.

        ``{{ input.name or 'anonymous' }}`` therefore yields the name, not ``True``,
        and ``a and a.b`` never evaluates ``a.b`` when ``a`` is empty.
        """
        if not isinstance(node.op, (ast.And, ast.Or)):
            raise ExpressionError("Unsupported boolean operator")
        stop_on = isinstance(node.op, ast.Or)
        value = None
        for operand in node.values:
            value = self.visit(operand)
            if bool(value) == stop_on:
                return value
        return value

    def visit_Compare(self, node):
        left = self.visit(node.left)
        for operation, comparator in zip(node.ops, node.comparators):
            right = self.visit(comparator)
            if not _compare(operation, left, right):
                return False
            left = right
        return True

    def visit_IfExp(self, node):
        return self.visit(node.body) if self.visit(node.test) else self.visit(node.orelse)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            if node.func.id not in self.helpers:
                raise ExpressionError(f"Unknown function: {node.func.id}")
            function = self.helpers[node.func.id]
        elif isinstance(node.func, ast.Attribute):
            owner = self.visit(node.func.value)
            allowed = _ALLOWED_METHODS.get(type(owner))
            if not allowed or node.func.attr not in allowed:
                raise ExpressionError(f"Method .{node.func.attr} is not allowed")
            function = getattr(owner, node.func.attr)
        else:
            raise ExpressionError("Only named helpers and safe methods can be called")
        args = [self.visit(arg) for arg in node.args]
        kwargs = {
            keyword.arg: self.visit(keyword.value)
            for keyword in node.keywords if keyword.arg
        }
        try:
            result = function(*args, **kwargs)
        except ExpressionError:
            raise
        except Exception as error:  # noqa: BLE001 - surface a readable expression error
            if self.lenient:
                return None
            raise ExpressionError(f"Expression failed: {error}") from error
        if isinstance(result, (str, int, float, bool, list, tuple, dict)) or result is None:
            return result
        if type(result).__name__ in {"dict_keys", "dict_values", "dict_items"}:
            return list(result)
        raise ExpressionError("Expression returned an unsupported value")


def _compare(operation, left, right) -> bool:
    if isinstance(operation, ast.Eq):
        return left == right
    if isinstance(operation, ast.NotEq):
        return left != right
    if isinstance(operation, ast.Lt):
        return left < right
    if isinstance(operation, ast.LtE):
        return left <= right
    if isinstance(operation, ast.Gt):
        return left > right
    if isinstance(operation, ast.GtE):
        return left >= right
    if isinstance(operation, ast.In):
        return left in right
    if isinstance(operation, ast.NotIn):
        return left not in right
    if isinstance(operation, ast.Is):
        return left is right
    if isinstance(operation, ast.IsNot):
        return left is not right
    raise ExpressionError("Unsupported comparison")


def evaluate_expression(expression: str, context: dict, lenient: bool = False):
    return _SafeEvaluator(context, lenient=lenient).evaluate(expression)


_TEMPLATE_PATTERN = re.compile(r"\{\{\s*(.+?)\s*\}\}", re.S)


def render_template(value, context: dict):
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    # A whole-string expression may legally contain braces (dict/list literals),
    # but more than one opening marker always means interpolation.
    if stripped.startswith("{{") and stripped.endswith("}}") and stripped.count("{{") == 1:
        try:
            return evaluate_expression(stripped[2:-2].strip(), context)
        except ExpressionError as error:
            raise ExpressionError(f"Template {value!r}: {error}") from error

    def replace(found):
        try:
            return _stringify(evaluate_expression(found.group(1), context))
        except ExpressionError as error:
            raise ExpressionError(f"Template {found.group(0)!r}: {error}") from error

    return _TEMPLATE_PATTERN.sub(replace, value)


def render_data(value, context: dict):
    if isinstance(value, str):
        return render_template(value, context)
    if isinstance(value, list):
        return [render_data(item, context) for item in value]
    if isinstance(value, dict):
        return {key: render_data(item, context) for key, item in value.items()}
    return value


# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------
_MONTHS = {name: index for index, name in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1
)}
_WEEKDAYS = {
    name: (index + 1) % 7
    for index, name in enumerate(["mon", "tue", "wed", "thu", "fri", "sat", "sun"])
}


def _cron_field(text: str, minimum: int, maximum: int, names: dict | None = None) -> tuple[set[int], bool]:
    text = text.strip().lower()
    if names:
        for name, value in names.items():
            text = re.sub(rf"\b{name}\b", str(value), text)
    values: set[int] = set()
    restricted = text != "*"
    for part in text.split(","):
        step = 1
        if "/" in part:
            part, step_text = part.split("/", 1)
            step = int(step_text)
            if step <= 0:
                raise ValueError("Cron step must be positive")
        if part in {"*", ""}:
            start, end = minimum, maximum
        elif "-" in part:
            start_text, end_text = part.split("-", 1)
            start, end = int(start_text), int(end_text)
        else:
            start = end = int(part)
        if start < minimum or end > maximum or start > end:
            raise ValueError(f"Cron value {part} is outside {minimum}-{maximum}")
        values.update(range(start, end + 1, step))
    if not values:
        raise ValueError("Cron field is empty")
    return values, restricted


def parse_cron(expression: str) -> dict:
    fields = str(expression).split()
    if len(fields) != 5:
        raise ValueError("Cron expression must have 5 fields: minute hour day month weekday")
    minute, _ = _cron_field(fields[0], 0, 59)
    hour, _ = _cron_field(fields[1], 0, 23)
    day, day_restricted = _cron_field(fields[2], 1, 31)
    month, _ = _cron_field(fields[3], 1, 12, _MONTHS)
    weekday, weekday_restricted = _cron_field(fields[4].replace("7", "0"), 0, 6, _WEEKDAYS)
    return {
        "minute": minute, "hour": hour, "day": day, "month": month,
        "weekday": weekday, "day_restricted": day_restricted,
        "weekday_restricted": weekday_restricted,
    }


def _cron_matches(moment: datetime, fields: dict) -> bool:
    if moment.minute not in fields["minute"] or moment.hour not in fields["hour"]:
        return False
    if moment.month not in fields["month"]:
        return False
    day_ok = moment.day in fields["day"]
    weekday_ok = (moment.weekday() + 1) % 7 in fields["weekday"]
    if fields["day_restricted"] and fields["weekday_restricted"]:
        return day_ok or weekday_ok
    return day_ok and weekday_ok


def next_cron(expression: str, after: datetime | None = None) -> datetime:
    fields = parse_cron(expression)
    moment = (after or datetime.now()).replace(second=0, microsecond=0) + timedelta(minutes=1)
    limit = moment + timedelta(days=366 * 2)
    while moment < limit:
        if _cron_matches(moment, fields):
            return moment
        moment += timedelta(minutes=1)
    raise ValueError("Cron expression never matches in the next two years")


def next_run_at(agent: dict, after: datetime | None = None,
                last_run: datetime | None = None) -> datetime | None:
    schedule = agent.get("schedule")
    if not isinstance(schedule, dict) or not schedule.get("enabled"):
        return None
    now = after or datetime.now()
    kind = schedule.get("type", "manual")
    if kind == "interval":
        minutes = max(1.0, float(schedule.get("interval_minutes", 60)))
        base = last_run or now
        candidate = base + timedelta(minutes=minutes)
        if candidate <= now:
            candidate = now + timedelta(minutes=minutes)
        return candidate
    if kind == "cron":
        return next_cron(schedule.get("cron", ""), now)
    if kind == "once":
        try:
            moment = datetime.fromisoformat(schedule.get("once_at", ""))
        except ValueError:
            return None
        return moment if moment > now else None
    return None


def summarize(agent: dict) -> dict:
    nodes = agent.get("nodes")
    edges = agent.get("edges")
    abilities = agent.get("abilities")
    schedule = agent.get("schedule")
    return {
        "id": agent.get("id"),
        "name": agent.get("name"),
        "description": agent.get("description", ""),
        "icon": agent.get("icon", "◈"),
        "version": agent.get("version", "1.0.0"),
        "model": agent.get("model", ""),
        "nodes": len(nodes) if isinstance(nodes, (list, tuple)) else 0,
        "edges": len(edges) if isinstance(edges, (list, tuple)) else 0,
        "abilities": len(abilities) if isinstance(abilities, (list, tuple)) else 0,
        "schedule": schedule if isinstance(schedule, dict) else {},
        "updated": agent.get("updated", ""),
        "created": agent.get("created", ""),
    }
