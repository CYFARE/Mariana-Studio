"""Chat's swarm mode: one conversation turn answered by a leader-led team.

Each conversation owns a small tool workspace under
``agent-workspace/files/chats/<chat-id>``.  Members may inspect it with the
read-only file tools (and, when the user turns it on, edit files or fetch public
web pages).  The saved conversation is the briefing and the newest user message
is the goal; members reason in parallel, may exchange addressed messages and
share a blackboard, and the leader's synthesis becomes the assistant reply.
This module owns request validation, the streaming bridge between the swarm
coordinator and Chat's server-sent events, and the compact summary that is
stored with the reply so the agent card survives a reload.
"""

from __future__ import annotations

import math
import os
import queue
import re
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from ..agents.aml import ABILITY_TYPES, SWARM_STRATEGIES
from ..agents.swarm import SwarmCoordinator, SwarmLimits, snapshot_due
from ..agents.tools import execute_ability
from ..workspace import chat_files_path, ensure_workspace
from .errors import BadRequestError, ConflictError
from .validation import choice, flag, number, text

DEFAULTS = SwarmLimits()
MAX_CONCURRENT_SWARMS = 3
MAX_MESSAGES = 40
MAX_MESSAGE_CHARS = 24_000
MAX_GOAL_CHARS = 12_000
MAX_BRIEFING_CHARS = 12_000
MAX_BRIEFING_TURN_CHARS = 4_000
MAX_SYSTEM_CHARS = 4_000
MAX_WARNINGS = 8
MAX_EVENT_DEPTH = 12
MAX_EVENT_ITEMS = 200
MAX_EVENT_STRING = 20_000
_END = object()
_CHAT_ID = re.compile(r"chat-[a-z0-9]{6,32}")
READ_TOOLS = (
    "file.list", "file.tree", "file.read", "file.search", "code.diagnostics",
    "csv.read", "time.now",
)
WRITE_TOOLS = ("file.write", "file.edit", "file.delete")
WEB_TOOLS = ("http.fetch", "news.headlines")
OS_TOOLS = ("shell.exec", "browser.fetch")
TOOL_META = {tool_id: ABILITY_TYPES[tool_id]
            for tool_id in (*READ_TOOLS, *WRITE_TOOLS, *WEB_TOOLS, *OS_TOOLS)
            if tool_id in ABILITY_TYPES}


@dataclass
class ChatSwarmRequest:
    """A validated Chat swarm request, ready for :class:`SwarmCoordinator`."""

    model: str
    model_name: str
    leader_model: str
    goal: str
    briefing: str
    system: str
    strategy: str
    rounds: int
    limits: SwarmLimits
    ipc: bool
    memory: bool
    tools: list[dict]
    workspace: Path | None
    allow_write: bool
    web: bool
    os_access: bool


class _SwarmJob:
    """One running swarm: its event queue, finish flag and cancellation flag."""

    def __init__(self, cancelled: threading.Event):
        self.cancelled = cancelled
        self.events: queue.Queue = queue.Queue()
        self.finished = threading.Event()


class ChatSwarm:
    """Validate and run Chat agent swarms, with per-conversation tools."""

    def __init__(self, root: Path, gateway, settings=None):
        self.root = Path(root).resolve()
        self.gateway = gateway
        self._settings = settings if callable(settings) else (lambda: settings or {})
        self._lock = threading.Lock()
        self._jobs: set[_SwarmJob] = set()

    @property
    def os_access(self) -> bool:
        """Server-wide opt-in for shell commands and local-network fetches."""
        if (os.environ.get("MARIANA_AGENT_ALLOW_OS") == "1"
                or os.environ.get("MARIANA_AGENT_ALLOW_SHELL") == "1"):
            return True
        try:
            return bool(self._settings().get("allow_os_access"))
        except Exception:  # noqa: BLE001 - a settings failure keeps the safe default
            return False

    # -- validation ------------------------------------------------------
    def prepare(self, config: dict) -> ChatSwarmRequest:
        """Return a normalised request or raise a boundary error."""
        if not isinstance(config, dict):
            raise BadRequestError("A swarm request must be a JSON object")
        messages = _messages(config)
        model = text(config, "model", "", 200)
        if not model:
            raise BadRequestError("Choose a model before starting a swarm")
        entry, _ = self.gateway.prepare({"model": model, "messages": messages})
        leader_model = text(config, "leader_model", "", 200)
        if leader_model:
            leader_entry, _ = self.gateway.prepare(
                {"model": leader_model, "messages": messages})
            leader_model = str(leader_entry.get("id") or leader_model)
        system_prompt = text(config, "system_prompt", "", MAX_SYSTEM_CHARS)
        chat_id = text(config, "chat_id", "", 64).strip()
        if chat_id and not _CHAT_ID.fullmatch(chat_id):
            raise BadRequestError(
                "The conversation id is not valid; refresh the conversation list and try again"
            )
        raw_tools = config.get("tools")
        if raw_tools is None:
            tools_config, tools_given = {}, False
        elif isinstance(raw_tools, dict):
            tools_config, tools_given = raw_tools, True
        else:
            raise BadRequestError("tools must be an object with files, write and web flags")
        wants_files = flag(tools_config, "files", True) if tools_given else False
        wants_write = flag(tools_config, "write", False) if tools_given else False
        wants_web = flag(tools_config, "web", True) if tools_given else False
        os_access = self.os_access
        if os_access:
            wants_web = True  # OS access includes the public web and local network
            wants_files = True
            wants_write = True
        if wants_write:
            wants_files = True  # editing files without reading them is not offered
        if (wants_files or wants_write or wants_web or os_access) and not chat_id:
            raise BadRequestError(
                "Save the conversation before using tools; the first message creates it"
            )
        tools = _tool_definitions(wants_files, wants_write, wants_web, os_access)
        workspace = (chat_files_path(self.root, chat_id)
                     if chat_id and tools else None)
        strategy = choice(config, "strategy", "auto", SWARM_STRATEGIES)
        max_agents = number(config, "max_agents", 4, 1, DEFAULTS.max_agents)
        max_steps = number(config, "max_steps", 10, 1, 12)
        rounds = number(config, "rounds", 1, 0, DEFAULTS.max_rounds)
        max_seconds = number(config, "max_seconds", 300.0, 20.0, DEFAULTS.max_seconds, float)
        limits = SwarmLimits(
            max_agents=max_agents,
            max_steps=max_steps,
            max_rounds=DEFAULTS.max_rounds,
            max_seconds=float(max_seconds),
            max_messages=160,
            max_message_chars=2_000,
            max_memory_entries=60,
            max_memory_chars=6_000,
            max_observation_chars=8_000,
            max_prompt_chars=32_000,
            max_output_chars=6_000,
            max_reviews=min(8, max(2, max_agents * 2)),
        )
        with self._lock:
            if len(self._jobs) >= MAX_CONCURRENT_SWARMS:
                raise ConflictError(
                    f"At most {MAX_CONCURRENT_SWARMS} Chat swarms can run at once; "
                    "wait for one to finish or stop it before starting another"
                )
        return ChatSwarmRequest(
            model=str(entry.get("id") or model),
            model_name=str(entry.get("name") or model),
            leader_model=leader_model,
            goal=_goal(messages),
            briefing=_briefing(messages[:-1], system_prompt, workspace, tools),
            system=_guidance(system_prompt, tools, wants_web, os_access),
            strategy=strategy,
            rounds=rounds,
            limits=limits,
            ipc=flag(config, "ipc", True),
            memory=flag(config, "memory", True),
            tools=tools,
            workspace=workspace,
            allow_write=wants_write or os_access,
            web=wants_web,
            os_access=os_access,
        )

    # -- streaming -------------------------------------------------------
    def stream(self, config: dict | ChatSwarmRequest,
               cancelled: threading.Event | None = None):
        """Yield swarm events; progress events may carry a ``snapshot``.

        The last event is ``done`` with the synthesised reply, or ``error`` when
        the swarm could not finish.  ``None`` is yielded while idle so the
        transport can keep the connection alive and notice a disconnect.
        """
        request = config if isinstance(config, ChatSwarmRequest) else self.prepare(config)
        cancelled = cancelled or threading.Event()
        job = _SwarmJob(cancelled)
        worker = threading.Thread(
            target=self._run, args=(job, request),
            name=f"chat-swarm-{uuid.uuid4().hex[:8]}", daemon=True,
        )
        with self._lock:
            self._jobs.add(job)
        try:
            worker.start()
            while True:
                try:
                    event = job.events.get(timeout=2.0)
                except queue.Empty:
                    if job.finished.is_set() and job.events.empty():
                        return
                    yield None
                    continue
                if event is _END:
                    return
                yield event
        finally:
            cancelled.set()
            with self._lock:
                self._jobs.discard(job)

    def close(self) -> None:
        """Cancel every running Chat swarm; called from ``Studio.close()``."""
        with self._lock:
            jobs = list(self._jobs)
            self._jobs.clear()
        for job in jobs:
            job.cancelled.set()

    # -- worker ----------------------------------------------------------
    def _run(self, job: _SwarmJob, request: ChatSwarmRequest) -> None:
        state = {"coordinator": None, "snapshot_at": 0.0}

        def on_event(kind: str, **fields) -> None:
            payload = {"type": kind, **fields}
            coordinator = state["coordinator"]
            now = time.monotonic()
            if (coordinator is not None and kind != "swarm_agent_delta"
                    and snapshot_due(kind, state["snapshot_at"], now)):
                try:
                    payload["snapshot"] = coordinator.snapshot()
                    state["snapshot_at"] = now
                except Exception:  # noqa: BLE001 - progress must not stop a swarm
                    pass
            job.events.put(_event_value(payload))

        try:
            coordinator = SwarmCoordinator(
                gateway=self.gateway,
                swarm_id=f"chat-swarm-{uuid.uuid4().hex[:10]}",
                goal=request.goal,
                model=request.model,
                leader_model=request.leader_model,
                system=request.system,
                tools=request.tools,
                executor=(_executor(self.root, request.workspace, request.tools,
                                    job.cancelled, request.os_access)
                          if request.tools and request.workspace is not None else None),
                limits=request.limits,
                allow_write=request.allow_write,
                allow_os_access=request.os_access,
                allow_ipc=request.ipc,
                shared_memory=request.memory,
                dry_run=False,
                on_event=on_event,
                cancelled=job.cancelled.is_set,
                briefing=request.briefing,
            )
            state["coordinator"] = coordinator
            result = coordinator.run(
                agents=None, strategy=request.strategy, rounds=request.rounds)
            summary = _summary(result)
            summary["tools"] = [tool["id"] for tool in request.tools]
            if result.get("ok"):
                job.events.put(_event_value({"type": "done", **_done(request, result, summary)}))
            else:
                job.events.put(_event_value({
                    "type": "error",
                    "error": str(result.get("error") or "The swarm failed"),
                    "code": "swarm_failed",
                    "swarm": summary,
                }))
        except Exception as failure:  # noqa: BLE001 - the stream reports failures in band
            job.events.put(_event_value({
                "type": "error",
                "error": str(failure) or type(failure).__name__,
                "code": "swarm_error",
            }))
        finally:
            job.events.put(_END)
            job.finished.set()


def _event_value(value, depth: int = 0):
    """Make an SSE payload browser-parseable: finite numbers, bounded nesting."""
    if depth > MAX_EVENT_DEPTH:
        return None
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return value[:MAX_EVENT_STRING]
    if isinstance(value, (list, tuple)):
        return [_event_value(item, depth + 1) for item in value[:MAX_EVENT_ITEMS]]
    if isinstance(value, dict):
        return {str(key)[:80]: _event_value(item, depth + 1)
                for key, item in list(value.items())[:MAX_EVENT_ITEMS]}
    return str(value)[:500]


def _messages(config: dict) -> list[dict]:
    """Read the transcript Chat sends, keeping the newest bounded turns."""
    raw = config.get("messages")
    if not isinstance(raw, list):
        raise BadRequestError("messages must be a list of role and content objects")
    messages = []
    for item in raw[-MAX_MESSAGES:]:
        if not isinstance(item, dict) or item.get("role") not in {"user", "assistant", "system"}:
            continue
        content = item.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        messages.append({"role": item["role"], "content": content[:MAX_MESSAGE_CHARS]})
    if not messages:
        raise BadRequestError("Enter a message first")
    if messages[-1]["role"] != "user":
        raise BadRequestError("The last message must be from you before a swarm can run")
    return messages


def _goal(messages: list[dict]) -> str:
    return messages[-1]["content"].strip()[:MAX_GOAL_CHARS]


def _tool_definitions(files: bool, write: bool, web: bool,
                      os_access: bool = False) -> list[dict]:
    """The Chat tool catalogue: read-only by default, writes/web/OS opt-in."""
    ids = []
    if files:
        ids.extend(READ_TOOLS)
    if write or os_access:
        ids.extend(WRITE_TOOLS)
    if web:
        ids.extend(WEB_TOOLS)
    if os_access:
        ids.extend(OS_TOOLS)
    tools = []
    for tool_id in ids:
        meta = TOOL_META.get(tool_id)
        if not meta:
            continue
        config = dict(meta.get("config") or {})
        description = meta.get("description") or ""
        if tool_id == "http.fetch":
            # Network reach is an author decision: Chat can fetch public pages only
            # unless the server explicitly enabled OS access.
            config.update({"allow_private": bool(os_access), "timeout": 15})
            description = ("Fetch a public http(s) URL. Private and local addresses "
                           "are blocked unless OS access is enabled; use extract=text "
                           "for readable output.")
        elif tool_id == "news.headlines":
            config.update({"allow_private": bool(os_access), "region": "IN",
                           "language": "en", "limit": 10, "include_summary": False,
                           "timeout": 15})
            description = ("Read a public RSS/Atom news feed and return dated headlines with "
                           "sources and links. Defaults to Google News for India; set region "
                           "and query as needed.")
        elif tool_id == "shell.exec":
            description = ("Run any command on this machine. OS access is enabled by the "
                           "server owner; use it only for the user's task.")
        elif tool_id == "browser.fetch":
            config["allow_private"] = bool(os_access)
            description = ("Render a page in a headless browser and return its visible text. "
                           "Useful for pages that need JavaScript; requires Playwright or "
                           "Chromium plus OS access.")
        elif tool_id == "file.delete":
            description = ("Delete a file from the conversation workspace. This cannot be "
                           "undone.")
        tools.append({
            "id": tool_id, "type": tool_id, "label": meta.get("label") or tool_id,
            "description": description,
            "params": meta.get("params") or [], "config": config, "enabled": True,
        })
    return tools


def _executor(root: Path, workspace: Path | None, tools: list[dict], cancelled,
              os_access: bool = False):
    """Run one member tool call through the AML sandbox for this conversation."""
    allowed = {tool["id"]: tool for tool in tools}
    files_root = workspace or (ensure_workspace(root) / "files")

    def execute(member, tool_id, arguments):
        name = str(tool_id or "")
        definition = allowed.get(name)
        if definition is None:
            return {
                "error": (f"Tool {name!r} is not available to this swarm. Use one of: "
                          + ", ".join(sorted(allowed))),
            }, False
        values = arguments if isinstance(arguments, dict) else {"input": arguments}
        try:
            observation = execute_ability(
                definition, values, {"agent": member, "run": {"id": "chat-swarm"}},
                root, files_root=files_root, cancelled=cancelled, os_access=os_access,
            )
        except Exception as failure:  # noqa: BLE001 - the member reads the failure and retries
            return {"error": str(failure) or type(failure).__name__}, False
        failed = isinstance(observation, dict) and (
            observation.get("ok") is False or bool(observation.get("error")))
        if not failed and isinstance(observation, dict) and name in WRITE_TOOLS:
            # The coordinator records changed files from a `written` flag; file.edit
            # reports `changed` instead, so normalise both here.
            if observation.get("changed") is not False:
                observation = {**observation, "written": True}
        return observation, not failed

    return execute


def _workspace_map(workspace: Path | None, limit: int = 40) -> str:
    """A short, bounded file listing so members know what exists before reading."""
    if workspace is None:
        return ""
    rows = []
    visited = 0
    try:
        for base, folders, names in os.walk(workspace, followlinks=False):
            visited += 1
            if visited > 200:
                break
            folders[:] = sorted(name for name in folders
                                if not (Path(base) / name).is_symlink())
            for name in sorted(names):
                item = Path(base) / name
                if item.is_symlink() or not item.is_file():
                    continue
                try:
                    relative = item.relative_to(workspace).as_posix()
                    size = item.stat().st_size
                except OSError:
                    continue
                rows.append(f"- {relative} ({size} bytes)")
                if len(rows) >= limit:
                    return "\n".join(rows)
    except OSError:
        return "(the workspace could not be read)"
    return "\n".join(rows) or "(the workspace is empty)"


def _guidance(system_prompt: str, tools: list[dict], web: bool,
              os_access: bool = False) -> str:
    """The conversation's instructions plus Chat's tool rules."""
    if not tools:
        note = ("Chat swarm: members have no tools and no workspace. Never emit a tool call; "
                "answer from the conversation text and say when it does not contain the answer.")
    else:
        names = ", ".join(tool["id"] for tool in tools)
        note = (f"Chat swarm tools: {names}. File paths are relative to the conversation "
                "workspace. Call a tool only when its result is needed, read the observation, "
                "and never claim a result you did not receive. ")
        if web:
            note += ("Public web access is enabled. For current headlines or a news list, call "
                     "news.headlines (set region=IN and language=en for India, and use query "
                     "for a topic); for a specific page use http.fetch with extract=text, and "
                     "cite the URL. News results include title, source, date and link; set "
                     "include_summary only when you need article summaries. If a requested "
                     "date has no items, use the newest items and state the date mismatch "
                     "instead of searching again. news.headlines uses Google News then Bing "
                     "News automatically; set source=bing or a feed URL if you need an "
                     "alternate source. If a page needs JavaScript, use browser.fetch when "
                     "it is available. A fetch is not a search unless it returned results. ")
        else:
            note += "Web access is disabled; do not claim to browse or know live data. "
        note += ("Stop calling tools as soon as the observations are enough to answer; do "
                 "not repeat the same call with the same input. Batch independent calls in one "
                 "round (the calls form) so one step covers several searches or fetches. If the "
                 "tool budget is reached, answer from the results already collected. ")
        if os_access:
            note += ("OS access is enabled: shell.exec can run any command on this machine, "
                     "browser.fetch and private addresses are also reachable. Use the smallest "
                     "tool that answers the request, never run destructive commands without "
                     "explicit instruction, and report failures honestly. ")
    if not system_prompt:
        return note[:MAX_SYSTEM_CHARS]
    # Keep the tool rules intact even when the conversation's system prompt is long.
    room = max(0, MAX_SYSTEM_CHARS - len(note) - 2)
    return f"{system_prompt[:room]}\n\n{note}"


def _briefing(history: list[dict], system_prompt: str,
              workspace: Path | None, tools: list[dict]) -> str:
    """Turn earlier conversation turns and the workspace into the shared briefing."""
    sections = []
    if system_prompt:
        sections.append(f"Conversation system prompt:\n{system_prompt}")
    if tools and workspace is not None:
        label = f"agent-workspace/files/chats/{workspace.name}"
        sections.append(
            f"Conversation workspace: {label}. File tool paths are relative to it.\n"
            f"Files:\n{_workspace_map(workspace)}"
        )
        sections.append(
            "Use file.list or file.tree to confirm what exists before reading. Never invent a "
            "path or claim a tool result you did not receive."
        )
    turns = []
    for message in history[-12:]:
        if message["role"] == "system":
            continue
        speaker = "User" if message["role"] == "user" else "Assistant"
        turns.append(f"{speaker}: {message['content'][:MAX_BRIEFING_TURN_CHARS]}")
    if turns:
        sections.append("Conversation so far:\n" + "\n\n".join(turns))
    if not tools:
        sections.append(
            "This is a text-only Chat conversation. There is no project workspace and no tools: "
            "work from the conversation text alone, never emit a tool call, and never claim to "
            "read, run or change anything."
        )
    return "\n\n".join(sections)[:MAX_BRIEFING_CHARS]


def _done(request: ChatSwarmRequest, result: dict, summary: dict) -> dict:
    """The terminal event Chat's reply path understands."""
    usage = dict(result.get("usage") or {})
    elapsed = float(result.get("elapsed") or 0.0)
    completion = int(usage.get("completion_tokens") or 0)
    payload = {
        "reply": str(result.get("final") or "The swarm produced no output."),
        "kind": "Swarm",
        "model": request.model,
        "model_name": request.model_name,
        "usage": usage,
        "elapsed": elapsed,
        "finish_reason": "stop",
        "swarm": summary,
    }
    if completion and elapsed > 0:
        payload["tokens_per_second"] = completion / elapsed
    warnings = [str(item).strip() for item in result.get("warnings") or [] if str(item).strip()]
    notes = []
    if warnings:
        notes.append("Swarm notes: " + "; ".join(warnings[:MAX_WARNINGS]))
    changed = [str(item) for item in summary.get("changed_files") or [] if item]
    if changed:
        notes.append("Files changed: " + ", ".join(changed[:8]))
    if notes:
        payload["notice"] = " · ".join(notes)[:600]
    return payload


def _summary(result: dict) -> dict:
    """Keep the parts of a swarm result that the Chat card can render later."""
    agents = []
    for item in (result.get("agents") or [])[:12]:
        if not isinstance(item, dict):
            continue
        started, ended = item.get("started"), item.get("ended")
        agents.append({
            "agent": item.get("id") or item.get("agent"),
            "label": str(item.get("label") or item.get("id") or "Agent")[:80],
            "role": str(item.get("role") or "")[:40],
            "task": str(item.get("task") or "")[:2_000],
            "status": str(item.get("status") or "queued")[:20],
            "ok": item.get("status") == "done",
            "output": str(item.get("output") or "")[:6_000],
            "summary": str(item.get("summary") or "")[:600],
            "error": str(item.get("error") or "")[:600],
            "score": item.get("score"),
            "version": item.get("version"),
            "steps": item.get("steps"),
            "elapsed": (round(ended - started, 2)
                        if isinstance(started, (int, float)) and isinstance(ended, (int, float))
                        else None),
            "usage": dict(item.get("usage") or {}),
        })
    reviews = []
    for item in (result.get("reviews") or [])[-12:]:
        if not isinstance(item, dict):
            continue
        issues = item.get("issues") if isinstance(item.get("issues"), list) else []
        suggestions = (item.get("suggestions")
                       if isinstance(item.get("suggestions"), list) else [])
        reviews.append({
            "round": item.get("round"),
            "reviewer": item.get("reviewer"),
            "target": item.get("target"),
            "score": item.get("score"),
            "issues": [str(value)[:300] for value in issues[:4]],
            "suggestions": [str(value)[:300] for value in suggestions[:4]],
            "summary": str(item.get("summary") or "")[:400],
        })
    memory = []
    for item in (result.get("blackboard") or [])[-20:]:
        if not isinstance(item, dict):
            continue
        memory.append({
            "key": str(item.get("key") or "")[:80],
            "value": item.get("value"),
            "version": item.get("version"),
            "author": item.get("author"),
        })
    return {
        "id": result.get("swarm"),
        "status": "completed" if result.get("ok") else "failed",
        "ok": bool(result.get("ok")),
        "strategy": result.get("strategy"),
        "round": result.get("round"),
        "rounds": result.get("requested_rounds"),
        "leader": result.get("leader"),
        "model": result.get("model"),
        "summary": str(result.get("summary") or "")[:1_000],
        "agents": agents,
        "reviews": reviews,
        "messages": int(result.get("messages") or 0),
        "review_count": len(result.get("reviews") or []),
        "memory": memory,
        "memory_count": int(result.get("blackboard_count") or len(memory)),
        "context": result.get("context") if isinstance(result.get("context"), dict) else {},
        "context_items": int(result.get("context_items") or 0),
        "usage": dict(result.get("usage") or {}),
        "elapsed": result.get("elapsed") or 0.0,
        "warnings": [str(item)[:400] for item in (result.get("warnings") or [])[:MAX_WARNINGS]],
        "changed_files": [str(item)[:400]
                          for item in (result.get("changed_files") or [])[:20]],
    }


__all__ = ["ChatSwarm", "ChatSwarmRequest"]
