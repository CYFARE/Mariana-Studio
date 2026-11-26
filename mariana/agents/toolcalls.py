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

Real replies are often damaged: a run that hits its output budget loses the
last closing tag, and some clients rewrite ``</invoke>`` as ``<invoke>``.  The
scanner therefore tolerates missing slashes, duplicated closers, unclosed
blocks, and marker variants, so a call still executes instead of being shown to
the user as a final answer.  :func:`usable_text` is the counterpart for the
answer path: it returns prose without markup, never markup itself.
"""

from __future__ import annotations

import html
import json
import re
from typing import NamedTuple

MAX_CALLS = 8

_CALL_TAGS = frozenset({"invoke", "tool_call", "function_call"})
_WRAPPERS = frozenset({"calls", "tool_calls"})
_GENERIC_INVOKES = frozenset({
    "call", "calls", "function", "function_call", "invoke", "tool_call", "tool_calls",
})
ACTION_KEYS = frozenset({
    "tool", "calls", "tools", "final", "plan", "say", "memory", "actions",
})

# A tag name may sit behind a chat-template marker ("<｜｜DSML｜｜ invoke") or a
# namespace ("<antml:invoke"); only the tag name itself is significant.
_TAG = re.compile(
    r"<\s*(?P<slash>/?)\s*(?:[^<>]{0,80}?\b)?"
    r"(?P<kind>invoke|parameter|tool_calls|tool_call|function_call|calls)\b"
    r"(?P<attrs>[^<>]*)>",
    re.IGNORECASE,
)
_ATTRIBUTE = re.compile(
    r"([A-Za-z_][\w:.-]*)\s*=\s*(?:\"([^\"]*)\"|'([^']*)'|([^\s\"'<>`]+))")
# A marker or tag cut off mid-stream has no closing ">"; it is still markup.
_TRUNCATED_TAG = re.compile(r"<[^<>]{0,80}(?:dsml|invoke|parameter)[^<>]{0,80}$",
                            re.IGNORECASE)
_DSML_FRAGMENT = re.compile(r"<\s*[^<>]{0,40}dsml", re.IGNORECASE)


class _Tag(NamedTuple):
    kind: str
    closing: bool
    self_closing: bool
    attributes: dict[str, str]
    start: int
    end: int

    @property
    def name(self) -> str:
        return self.attributes.get("name", "").strip()


def _attributes(raw: str) -> dict[str, str]:
    found: dict[str, str] = {}
    for match in _ATTRIBUTE.finditer(raw or ""):
        value = next((group for group in match.groups()[1:] if group is not None), "")
        found[match.group(1).lower()] = html.unescape(value)
    return found


def _iter_tags(text: str):
    """Every call-related tag, in order, with tolerant closing-tag detection."""
    for match in _TAG.finditer(text):
        raw = match.group("attrs") or ""
        kind = match.group("kind").lower()
        attributes = _attributes(raw)
        closing = bool(match.group("slash"))
        if not closing and kind in {"invoke", "parameter"} and not attributes:
            # ``<invoke>`` / ``<parameter>`` with no attributes is a closer whose
            # slash was lost; an opening tag always carries a name.
            closing = True
        yield _Tag(kind, closing, raw.rstrip().endswith("/"), attributes,
                   match.start(), match.end())


def _value(text: str, is_string: bool = False):
    """A parameter value with its JSON type, unless the tag marks it as a string."""
    stripped = html.unescape(text).strip()
    if not stripped or is_string:
        return stripped
    try:
        return json.loads(stripped)
    except (TypeError, ValueError):
        return stripped


def _canonical_call(name: str,
                    parameters: dict[str, tuple[str, bool]]) -> dict | None:
    """One invoke reduced to ``{"tool": id, "input": {...}}``."""
    raw_tool, _ = parameters.pop("tool", (None, True))
    tool_id = html.unescape(raw_tool or "").strip()
    arguments: dict = {}
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
    if not tool_id:
        return None
    return {"tool": tool_id[:200], "input": arguments}


def parse_dsml_action(text: str) -> tuple[dict, list[dict]]:
    """Action hints and tool calls from DeepSeek-style tool markup.

    Models trained on the JSON protocol do not always spell the markup the way
    the chat template documents.  Observed shapes include a normal ``invoke``
    with parameters, one ``invoke name="calls"`` whose body is a JSON array of
    calls, a bare array between ``<calls>`` tags, nested ``parameter`` tags
    naming the tool, and an ``invoke name="plan"`` whose body is the plan array.
    All of them reduce to the same canonical calls, plus ``plan``/``final``
    hints when the envelope carries them.
    """
    if not isinstance(text, str):
        return {}, []
    lowered = text.lower()
    if not any(marker in lowered for marker in
               ("invoke", "tool_call", "function_call", "parameter", "calls")):
        return {}, []
    action: dict = {}
    calls: list[dict] = []
    invoke: dict | None = None
    param: dict | None = None
    wrapper: dict | None = None

    def body_of(start: int, end: int) -> str:
        raw = text[start:max(end, start)]
        return _TAG.sub("", html.unescape(raw)).strip()

    def close_parameter(end: int) -> None:
        nonlocal param
        if param is not None and invoke is not None and param["key"]:
            invoke["parameters"].setdefault(
                param["key"], (text[param["start"]:max(end, param["start"])],
                               param["string"]))
        param = None

    def close_invoke(end: int) -> None:
        nonlocal invoke
        if invoke is None:
            return
        name = str(invoke["name"] or "").strip()
        lowered_name = name.lower()
        parameters = invoke["parameters"]
        if parameters:
            call = _canonical_call(name, parameters)
            if call:
                calls.append(call)
        else:
            body = body_of(invoke["start"], end)
            value = _json_value(body)
            if lowered_name == "plan":
                plan = value.get("plan") if isinstance(value, dict) else value
                if isinstance(plan, dict):
                    plan = plan.get("items") or plan.get("steps") or plan
                if isinstance(plan, list) and plan:
                    action["plan"] = plan
            elif lowered_name in {"final", "answer", "result", "summary"}:
                if body:
                    action["final"] = value if isinstance(value, str) else body
            else:
                expanded = _calls_from_value(value)
                if expanded:
                    calls.extend(expanded)
                elif name and lowered_name not in _GENERIC_INVOKES:
                    calls.append({"tool": name[:200], "input": {}})
        invoke = None

    for tag in _iter_tags(text):
        if len(calls) >= MAX_CALLS:
            break
        if tag.kind in _CALL_TAGS:
            close_parameter(tag.start)
            close_invoke(tag.start)  # a new opening tag also ends an unclosed block
            if not tag.closing:
                invoke = {"name": tag.name, "parameters": {}, "start": tag.end}
        elif tag.kind == "parameter":
            close_parameter(tag.start)
            if tag.closing:
                continue
            if invoke is None and "." in tag.name and " " not in tag.name:
                # ``<parameter name="file.read">`` with nested parameters is a
                # tool call whose opening tag lost the ``invoke`` spelling.
                invoke = {"name": tag.name, "parameters": {}, "start": tag.end}
                continue
            if invoke is None:
                continue
            if tag.self_closing:
                if tag.name:
                    invoke["parameters"].setdefault(tag.name, ("", False))
                continue
            param = {
                "key": tag.name,
                "string": tag.attributes.get("string", "").strip().lower() == "true",
                "start": tag.end,
            }
        else:  # a calls/tool_calls wrapper
            close_parameter(tag.start)
            if tag.closing:
                close_invoke(tag.start)
                if (wrapper is not None and len(calls) == wrapper["before"]
                        and not action):
                    # ``<calls>[{"tool": ...}, ...]</calls>``: the array itself
                    # is the batch, not a parameter of one call.
                    calls.extend(_calls_from_value(
                        _json_value(body_of(wrapper["start"], tag.start))))
                wrapper = None
            elif wrapper is None:
                wrapper = {"start": tag.end, "before": len(calls)}
    close_parameter(len(text))
    close_invoke(len(text))
    if wrapper is not None and len(calls) == wrapper["before"] and not action:
        calls.extend(_calls_from_value(
            _json_value(body_of(wrapper["start"], len(text)))))
    if not calls and not action and _DSML_FRAGMENT.search(text):
        # A wrapper tag may be missing entirely; a JSON array of call objects
        # beside DSML markers is still the batch the model meant to send.
        calls.extend(_call_array_in(text))
    return action, calls[:MAX_CALLS]


def parse_dsml_tool_calls(text: str) -> list[dict]:
    """Canonical tool calls from DeepSeek-style DSML markup, or ``[]``."""
    return parse_dsml_action(text)[1]


def _json_value(text: str):
    """A JSON value, or ``None`` when the text is not JSON."""
    stripped = str(text or "").strip()
    if not stripped or stripped[0] not in "[{\"":
        return None
    try:
        return json.loads(stripped)
    except (TypeError, ValueError):
        return None


def _calls_from_value(value) -> list[dict]:
    """Expand a call object, a list of them, or a ``{"calls": [...]}`` envelope."""
    if isinstance(value, dict):
        value = value.get("calls") if isinstance(value.get("calls"), list) else [value]
    if not isinstance(value, list):
        return []
    calls: list[dict] = []
    for item in value:
        call = _call_from_object(item)
        if call:
            calls.append(call)
        if len(calls) >= MAX_CALLS:
            break
    return calls


def _call_from_object(item) -> dict | None:
    """One ``{"tool": id, "input": {...}}`` object from a JSON batch."""
    if not isinstance(item, dict):
        return None
    tool_id = str(item.get("tool") or item.get("name") or "").strip()
    if not tool_id:
        return None
    arguments = item.get("input", item.get("arguments", item.get("args", {})))
    if isinstance(arguments, str):
        arguments = _json_value(arguments) or arguments
    if isinstance(arguments, dict):
        payload = arguments
    elif arguments in (None, ""):
        payload = {}
    else:
        payload = {"input": arguments}
    return {"tool": tool_id[:200], "input": payload}


def _call_array_in(text: str) -> list[dict]:
    """The first JSON array of call objects in the text, or ``[]``."""
    decoder = json.JSONDecoder()
    index = 0
    while index < len(text):
        start = text.find("[", index)
        if start < 0:
            return []
        try:
            value, length = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            index = start + 1
            continue
        calls = _calls_from_value(value)
        if calls:
            return calls
        index = start + length
    return []


def has_tool_markup(value) -> bool:
    """Whether a reply contains native tool-call markup rather than only prose.

    Ordinary XML and HTML answers are not flagged: a tag must be a call tag,
    parameter or call wrapper, or carry a DSML-style marker for the reply to
    count as an attempted tool call.
    """
    if not isinstance(value, str) or not value:
        return False
    for tag in _iter_tags(value):
        if tag.kind in _CALL_TAGS or tag.kind in _WRAPPERS or tag.kind == "parameter":
            return True
    return bool(_DSML_FRAGMENT.search(value))


def _markup_spans(text: str) -> list[tuple[int, int]]:
    """Character ranges covered by call blocks, wrappers and parameters, merged."""
    spans: list[tuple[int, int]] = []
    call_at: int | None = None
    param_at: int | None = None
    wrapper_at: int | None = None
    loose_at: int | None = None  # a top-level parameter that names its tool

    def close_parameter(at: int) -> None:
        nonlocal param_at
        if param_at is not None:
            spans.append((param_at, max(at, param_at)))
            param_at = None

    def close_loose(at: int) -> None:
        nonlocal loose_at
        if loose_at is not None:
            spans.append((loose_at, max(at, loose_at)))
            loose_at = None

    for tag in _iter_tags(text):
        if tag.kind in _CALL_TAGS:
            close_parameter(tag.start)
            if tag.closing:
                close_loose(tag.end)
                if call_at is not None:
                    spans.append((call_at, tag.end))
                    call_at = None
            else:
                close_loose(tag.start)
                if call_at is not None:
                    spans.append((call_at, tag.start))
                call_at = tag.start
        elif tag.kind == "parameter":
            close_parameter(tag.start)
            if not tag.closing and not tag.self_closing and tag.name:
                if call_at is None and loose_at is None:
                    loose_at = tag.start
                param_at = tag.start
        else:  # wrapper
            close_parameter(tag.start)
            if tag.closing:
                close_loose(tag.end)
                if call_at is not None:
                    spans.append((call_at, tag.start))
                    call_at = None
                if wrapper_at is not None:
                    spans.append((wrapper_at, tag.end))
                    wrapper_at = None
            elif wrapper_at is None:
                wrapper_at = tag.start
    close_parameter(len(text))
    close_loose(len(text))
    if call_at is not None:
        spans.append((call_at, len(text)))
    if wrapper_at is not None:
        spans.append((wrapper_at, len(text)))
    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged
    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def strip_tool_markup(value) -> str:
    """Text with every call block, parameter, and stray tag removed."""
    text = str(value or "")
    if not text:
        return ""
    for start, end in reversed(_markup_spans(text)):
        text = text[:start] + text[end:]
    text = _TAG.sub("", text)
    text = _TRUNCATED_TAG.sub("", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def usable_text(value) -> str:
    """Prose from a reply, or ``""`` when the reply is only tool-call markup.

    A model that runs out of steps sometimes emits its native call syntax where
    an answer belongs; that text must never reach the user as a finding.
    """
    text = str(value or "").strip()
    if not text or not has_tool_markup(text):
        return text
    cleaned = strip_tool_markup(text)
    return cleaned


def extract_json_objects(text: str):
    """Every balanced JSON object in a reply, in order."""
    if not isinstance(text, str):
        return
    for value, _start, _end in _json_objects(text):
        yield value


def _json_objects(text: str):
    """Every balanced JSON object with its character span."""
    decoder = json.JSONDecoder()
    index = 0
    while index < len(text):
        start = text.find("{", index)
        if start < 0:
            return
        try:
            value, length = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            index = start + 1
            continue
        if isinstance(value, dict):
            yield value, start, start + length
        index = start + length


def extract_action_span(text: str) -> tuple[dict, int, int] | None:
    """The last action-shaped object and the span it occupies, or ``None``.

    Native tool calls are appended after any streamed commentary, so the last
    action wins over a JSON snippet the model quoted in its prose.
    """
    if not isinstance(text, str):
        return None
    found = list(_json_objects(text))
    if not found:
        return None
    for value, start, end in reversed(found):
        if ACTION_KEYS & set(value):
            return value, start, end
    return found[-1]


def extract_action_object(text: str) -> dict | None:
    """The last action-shaped JSON object; falls back to the last object."""
    found = extract_action_span(text)
    return found[0] if found else None


def first_call_span(text: str) -> tuple[int, int] | None:
    """Span of the first tool-call tag in a reply, if any."""
    for tag in _iter_tags(str(text or "")):
        if tag.kind in _CALL_TAGS:
            return tag.start, tag.end
    return None


def json_action_outranks_markup(text: str, action_span: tuple[int, int]) -> bool:
    """Whether the JSON action encloses the markup, so the JSON is the contract.

    A model that answers ``{"final": "…<invoke…>"}`` quoted markup inside its
    answer; a model that emits DSML writes the tags before any parameter JSON.
    """
    call = first_call_span(text)
    return bool(call and action_span[0] <= call[0] < action_span[1])


__all__ = [
    "extract_action_object", "extract_action_span", "extract_json_objects",
    "first_call_span", "has_tool_markup", "json_action_outranks_markup",
    "parse_dsml_action", "parse_dsml_tool_calls", "strip_tool_markup", "usable_text",
]
