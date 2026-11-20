"""Chat backends: one module per way of running a model.

Every backend exposes ``stream(gateway, entry, request)`` and yields events:

* ``{"type": "status", "message": str}``    progress before the first token
* ``{"type": "delta", "text": str}``         visible reply text
* ``{"type": "reasoning", "text": str}``     model reasoning, shown collapsed
* ``{"type": "done", ...}``                  exactly once, with ``usage`` and
  ``prompt_format``; the gateway adds the reply, timing and model identity.

Backends stop promptly when ``request.cancelled`` is set or the generator is
closed, and release whatever they hold in a ``finally`` block.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field

EMPTY_REPLY = "(No readable text generated)"
TRANSCRIPT_LABELS = {"system": "System", "user": "User", "assistant": "Assistant"}
TRANSCRIPT_STOPS = ("\nUser:", "\nSystem:")
MAX_NATIVE_CALLS = 8
_TOOL_NAME = re.compile(r"[^A-Za-z0-9_-]+")
_REQUIRED_PARAMS = frozenset({
    "path", "content", "old", "new", "query", "url", "sql", "command",
    "expression", "name",
})


@dataclass
class ChatRequest:
    """A validated chat request, independent of any backend."""

    messages: list[dict]
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 0
    max_tokens: int = 256
    device: str = "auto"
    options: dict = field(default_factory=dict)
    cancelled: threading.Event = field(default_factory=threading.Event)


def transcript(messages: list[dict]) -> str:
    """Render messages as ``Role: text`` lines — the layout for models without a chat template.

    Must stay byte-identical to ``mariana.studio.finetune.dataset.transcript`` so a model
    fine-tuned on chat data sees the same layout at inference time.
    """
    return "".join(
        f"{TRANSCRIPT_LABELS.get(message['role'], 'User')}: {message['content']}\n"
        for message in messages
    )


def transcript_prompt(messages: list[dict]) -> str:
    return transcript(messages) + "Assistant:"


def status(message: str) -> dict:
    return {"type": "status", "message": message}


def delta(text: str) -> dict:
    return {"type": "delta", "text": text}


def reasoning(text: str) -> dict:
    return {"type": "reasoning", "text": text}


def done(**fields) -> dict:
    return {"type": "done", **fields}


def tool_alias(name: str, used: set | None = None) -> str:
    """A provider-safe function name; dots become underscores."""
    alias = _TOOL_NAME.sub("_", str(name or "")).strip("_")[:64] or "tool"
    if used is None:
        return alias
    candidate, index = alias, 2
    while candidate in used:
        suffix = f"_{index}"
        candidate = f"{alias[:64 - len(suffix)]}{suffix}"
        index += 1
    used.add(candidate)
    return candidate


def _parameter_schema(param: dict) -> dict:
    """One Mariana tool parameter as JSON Schema."""
    kind = str(param.get("type") or "string")
    schema: dict = {}
    if kind == "number":
        schema["type"] = "number"
        for key, keyword in (("min", "minimum"), ("max", "maximum"), ("step", "multipleOf")):
            value = param.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                schema[keyword] = value
    elif kind == "boolean":
        schema["type"] = "boolean"
    elif kind == "json":
        schema["type"] = "object"
    elif kind == "abilities":
        schema.update({"type": "array", "items": {"type": "string"}})
    elif kind == "select":
        schema["type"] = "string"
        values = []
        for option in param.get("options") or []:
            if isinstance(option, (list, tuple)) and option:
                values.append(str(option[0]))
            elif isinstance(option, str):
                values.append(option)
        if values:
            schema["enum"] = values
    else:
        schema["type"] = "string"
    description = ". ".join(part for part in (
        str(param.get("label") or ""), str(param.get("help") or ""),
    ) if part)
    if description:
        schema["description"] = description[:300]
    return schema


def tool_parameters(parameters) -> dict:
    """A JSON Schema object for one tool's parameters."""
    properties = {}
    for param in parameters if isinstance(parameters, list) else []:
        if isinstance(param, dict) and str(param.get("key") or "").strip():
            properties[str(param["key"]).strip()] = _parameter_schema(param)
    schema = {"type": "object", "properties": properties}
    required = [key for key in properties if key in _REQUIRED_PARAMS]
    if required:
        schema["required"] = required
    return schema


def native_tools(tools) -> tuple[list[dict], dict[str, str]]:
    """OpenAI-style function schemas and the provider alias to Mariana id map."""
    schemas: list[dict] = []
    aliases: dict[str, str] = {}
    used: set[str] = set()
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        tool_id = str(tool.get("id") or tool.get("type") or "").strip()
        if not tool_id or tool.get("enabled") is False:
            continue
        alias = tool_alias(tool_id, used)
        aliases[alias] = tool_id
        schemas.append({
            "type": "function",
            "function": {
                "name": alias,
                "description": str(tool.get("description") or tool.get("label")
                                   or tool_id)[:1024],
                "parameters": tool_parameters(tool.get("params")),
            },
        })
    return schemas, aliases


def anthropic_tools(tools) -> tuple[list[dict], dict[str, str]]:
    """The same tools in the Anthropic Messages API shape."""
    schemas, aliases = native_tools(tools)
    return ([{
        "name": item["function"]["name"],
        "description": item["function"]["description"],
        "input_schema": item["function"]["parameters"],
    } for item in schemas], aliases)


def parse_tool_arguments(value) -> dict:
    """A provider's JSON-string arguments as a dict; malformed text is kept visible."""
    if isinstance(value, dict):
        return value
    if value in (None, ""):
        return {}
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    stripped = text.strip()
    try:
        parsed = json.loads(stripped)
    except (TypeError, ValueError):
        return {"input": stripped}
    return parsed if isinstance(parsed, dict) else {"input": parsed}


def canonical_calls(slots, aliases: dict[str, str]) -> list[dict]:
    """Turn provider tool-call slots into Mariana's ``{tool, input}`` calls."""
    calls = []
    for slot in slots if isinstance(slots, list) else []:
        if not isinstance(slot, dict):
            continue
        name = str(slot.get("name") or "").strip()
        if not name:
            continue
        calls.append({
            "tool": (aliases.get(name)
                     or aliases.get(name.rsplit(".", 1)[-1])
                     or name),
            "input": parse_tool_arguments(slot.get("arguments")),
        })
        if len(calls) >= MAX_NATIVE_CALLS:
            break
    return calls


def tool_calls_text(calls: list[dict]) -> str:
    """The canonical JSON the agent loops already understand."""
    return json.dumps({"calls": calls}, ensure_ascii=False, default=str)


class StopFilter:
    """Cut streamed text at the first stop string without leaking a partial match."""

    def __init__(self, stops=TRANSCRIPT_STOPS):
        self.stops = tuple(stop for stop in stops if stop)
        self.hold = max((len(stop) for stop in self.stops), default=1) - 1
        self.buffer = ""
        self.stopped = False

    def feed(self, text: str) -> str:
        """Return the part of ``text`` that is safe to show now."""
        if self.stopped or not text:
            return ""
        self.buffer += text
        cut = min((index for index in (self.buffer.find(stop) for stop in self.stops) if index >= 0),
                  default=-1)
        if cut >= 0:
            self.stopped = True
            ready, self.buffer = self.buffer[:cut], ""
            return ready
        if self.hold <= 0:
            ready, self.buffer = self.buffer, ""
            return ready
        ready, self.buffer = self.buffer[:-self.hold], self.buffer[-self.hold:]
        return ready

    def flush(self) -> str:
        ready, self.buffer = ("" if self.stopped else self.buffer), ""
        return ready
