"""Read tool-call spellings that models emit as text.

Mariana's agent loops ask models for one JSON action per turn (``{"tool": ...}``
or ``{"calls": [...]}``), which works with every backend.  Some chat templates
ignore that instruction and emit their native markup instead; DeepSeek V4, for
example, writes

    <｜｜DSML｜｜ invoke name="file.read">
    <｜｜DSML｜｜ parameter name="path" string="true">notes.txt</｜｜DSML｜｜ parameter>
    </｜｜DSML｜｜ invoke>

Both the standard shape and the ``invoke name="calls"`` shape (where the tool
and its input are parameters) are accepted and reduced to the same canonical
``{"tool": id, "input": {...}}`` calls.  Values keep their JSON types, so a
line number stays an integer.
"""

from __future__ import annotations

import html
import json
import re

MAX_CALLS = 8

# The markers around "DSML" vary by chat template (full-width bars in current
# DeepSeek models), so the tag prefix and suffix are skipped rather than spelled.
_INVOKE = re.compile(
    r"<[^>]*?invoke\s+name\s*=\s*[\"']([^\"']+)[\"'][^>]*>(.*?)</[^>]*?invoke\s*>",
    re.IGNORECASE | re.DOTALL,
)
_PARAMETER = re.compile(
    r"<[^>]*?parameter\s+name\s*=\s*[\"']([^\"']+)[\"']([^>]*)>(.*?)</[^>]*?parameter\s*>",
    re.IGNORECASE | re.DOTALL,
)
_GENERIC_INVOKES = frozenset({"call", "calls", "function", "invoke", "tool_call", "tool_calls"})
ACTION_KEYS = frozenset({
    "tool", "calls", "tools", "final", "plan", "say", "memory", "actions",
})


def _value(text: str, is_string: bool = False):
    """A parameter value with its JSON type, unless the tag marks it as a string."""
    stripped = html.unescape(text).strip()
    if not stripped or is_string:
        return stripped
    try:
        return json.loads(stripped)
    except (TypeError, ValueError):
        return stripped


def parse_dsml_tool_calls(text: str) -> list[dict]:
    """Canonical tool calls from DeepSeek-style DSML markup, or ``[]``."""
    if not isinstance(text, str) or "invoke" not in text.lower():
        return []
    calls = []
    for match in _INVOKE.finditer(text):
        name = html.unescape(match.group(1)).strip()
        parameters = {}
        for item in _PARAMETER.finditer(match.group(2)):
            key = html.unescape(item.group(1)).strip()
            if key:
                is_string = bool(re.search(
                    r"string\s*=\s*[\"']true[\"']", item.group(2) or "", re.IGNORECASE,
                ))
                parameters[key] = (item.group(3), is_string)
        raw_tool, _ = parameters.pop("tool", (None, True))
        tool_id = html.unescape(raw_tool or "").strip()
        arguments = {}
        raw_input, input_is_string = parameters.pop("input", (None, False))
        if raw_input is not None:
            parsed = _value(raw_input, input_is_string)
            if isinstance(parsed, dict):
                arguments = parsed
            elif parsed not in (None, ""):
                arguments = {"input": parsed}
            else:
                arguments = {"input": html.unescape(raw_input).strip()}
        for key, (raw, is_string) in parameters.items():
            if key in {"arguments", "args"}:
                parsed = _value(raw, is_string)
                if isinstance(parsed, dict):
                    arguments.update(parsed)
                    continue
            if key not in arguments:
                arguments[key] = _value(raw, is_string)
        if not tool_id:
            if name.lower() not in _GENERIC_INVOKES:
                tool_id = name
            elif isinstance(arguments.get("tool"), str):
                tool_id = arguments.pop("tool").strip()
        if tool_id:
            calls.append({"tool": tool_id[:200], "input": arguments})
        if len(calls) >= MAX_CALLS:
            break
    return calls


def extract_json_objects(text: str):
    """Every balanced JSON object in a reply, in order."""
    if not isinstance(text, str):
        return
    decoder = json.JSONDecoder()
    index = 0
    while index < len(text):
        start = text.find("{", index)
        if start < 0:
            return
        try:
            value, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            index = start + 1
            continue
        if isinstance(value, dict):
            yield value
        index = start + end


def extract_action_object(text: str) -> dict | None:
    """The last action-shaped JSON object; falls back to the last object.

    Native tool calls are appended after any streamed commentary, so the last
    action wins over a JSON snippet the model quoted in its prose.
    """
    objects = list(extract_json_objects(text))
    if not objects:
        return None
    for value in reversed(objects):
        if ACTION_KEYS & set(value):
            return value
    return objects[-1]


__all__ = ["extract_action_object", "extract_json_objects", "parse_dsml_tool_calls"]
