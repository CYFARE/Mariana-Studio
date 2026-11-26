"""CRAFT: anchor-preserving, recoverable context compression for agent memory.

The reference literature converges on a few failures that plain summarization
causes: provenance is lost, a summary drifts from the source, and a later agent
cannot recover the part that was dropped.  CRAFT keeps three layers:

* **Anchors** — file paths, ``path:line`` references, symbols, URLs, commands and
  error lines are extracted verbatim before anything is shortened.
* **Facts** — units are ranked by goal overlap, error/decision markers, recency
  and anchor density, then deduplicated.
* **Traces** — head and tail units survive, and the original text stays behind a
  ``ctx-…`` handle in :class:`SwarmContextBank` so any member can read it back.

The compressor is deliberately extractive and deterministic.  It spends no
model call, so it can run on every observation and inside a 64-agent swarm.
"""

from __future__ import annotations

import json
import re

MAX_ANCHORS = 32
MARKER = "[CRAFT]"
_URL = re.compile(r"https?://[^\s<>)\"']+")
_PATH = re.compile(
    r"(?<![\w/.-])(?:\.{0,2}/)?(?:[\w.-]+/)+[\w.-]+\.[A-Za-z0-9]{1,12}"
    r"(?::\d+(?:-\d+)?)?")
_FILE = re.compile(
    r"(?<![\w/.-])[\w.-]+\.(?:py|cpp|cc|cxx|h|hpp|qml|js|ts|tsx|jsx|md|json|"
    r"yaml|yml|toml|ini|cfg|sh|rs|go|java|kt|swift|m|mm|ui|xml|html|css|txt|"
    r"log|sql|csv)(?::\d+(?:-\d+)?)?",
    re.IGNORECASE)
_LINE = re.compile(r"\bline\s+(\d+)\b", re.IGNORECASE)
_ERROR = re.compile(
    r"(?:error|exception|traceback|failed|failure|timeout|timed out|denied|"
    r"not found|segfault|nullptr|nan|undefined|cannot|can't|refused|panic|"
    r"overflow|dangling|use-after-free|double free)", re.IGNORECASE)
_DECISION = re.compile(
    r"\b(must|should|need to|because|therefore|root cause|fix|bug|decision|"
    r"verified|repro|expected|actual|breaking|regression|constraint)\b",
    re.IGNORECASE)
_COMMAND = re.compile(r"^\s*(?:\$|>|PS>)\s*(\S.{0,140})$")
_QUOTED = re.compile(r"`([^`\n]{2,100})`")
_TOOL_LINE = re.compile(r"^([\w.-]+)\s+(returned|failed):\s*$", re.IGNORECASE)
_BOILERPLATE = re.compile(
    r"^(?:[-*#\s]*)(?:ok|done|success|continue|working|started|returned|using)\b",
    re.IGNORECASE)


def _text(value) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)


def _dedupe(values) -> list[str]:
    result, seen = [], set()
    for value in values:
        text = str(value or "").strip()
        key = re.sub(r"\s+", " ", text).strip().lower()
        if not text or key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def extract_anchors(value, limit: int = MAX_ANCHORS) -> list[str]:
    """Verbatim evidence markers: URLs, paths, path:line, errors, commands, code."""
    text = _text(value)
    if not text:
        return []
    limit = max(1, int(limit))
    anchors: list[str] = []

    def add(item: str, cap: int = 180) -> None:
        item = re.sub(r"\s+", " ", str(item or "")).strip(" \t-*#>")
        if 2 <= len(item) <= cap:
            anchors.append(item)

    for match in _URL.finditer(text):
        add(match.group(0), 220)
    for pattern in (_PATH, _FILE):
        for match in pattern.finditer(text):
            add(match.group(0), 160)
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if _ERROR.search(stripped):
            add(stripped, 180)
        command = _COMMAND.match(stripped)
        if command:
            add(command.group(1), 180)
        if len(anchors) >= limit * 2:
            break
    for match in _QUOTED.finditer(text):
        add(match.group(1), 100)
    for match in _LINE.finditer(text):
        add(match.group(0), 40)
    return _dedupe(anchors)[:limit]


def anchor_line(value, limit: int = 18, chars: int = 900) -> str:
    """A compact, bounded block of anchors for a prompt."""
    anchors = extract_anchors(value, limit=limit)
    if not anchors:
        return ""
    text = " | ".join(anchors)
    return text[:chars].rstrip(" |")


def _units(text: str) -> list[str]:
    """Lines for structured output, sentences for prose."""
    if text.count("\n") >= 2 or text.lstrip().startswith(("{", "[", "```")):
        raw = text.splitlines()
    else:
        raw = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9`\"'])", text)
    units = []
    for item in raw:
        cleaned = str(item or "").strip()
        if not cleaned or set(cleaned) <= {"-", "*", "=", "_", " "}:
            continue
        if _BOILERPLATE.match(cleaned) and len(cleaned) < 40:
            continue
        units.append(cleaned)
    return _dedupe(units)


def _goal_terms(goal: str) -> list[str]:
    return [word.lower() for word in re.split(r"\W+", str(goal or "")) if len(word) >= 3]


def _score(unit: str, terms: list[str], index: int, total: int) -> float:
    lowered = unit.lower()
    score = 0.0
    score += 4.0 * (_PATH.search(unit) is not None or _FILE.search(unit) is not None)
    score += 4.0 * (_ERROR.search(unit) is not None)
    score += 3.0 * (_DECISION.search(unit) is not None)
    score += 2.0 * (sum(lowered.count(term) for term in terms))
    score += 1.5 * (any(ch.isdigit() for ch in unit))
    if total > 1:
        score += 2.0 * (index / max(1, total - 1))
    if len(unit) > 500:
        score -= 1.0
    return score


def _elide_units(units: list[str], budget: int) -> str:
    selected, used = [], 0
    for unit in units:
        line = f"- {unit}" if not unit.startswith(("- ", "* ")) else unit
        if used + len(line) + 1 > budget:
            continue
        selected.append(line)
        used += len(line) + 1
    return "\n".join(selected)


def _clip(text: str, budget: int) -> str:
    """Head and tail survive when no full unit fits a very small budget."""
    budget = max(0, int(budget))
    if len(text) <= budget:
        return text
    if budget <= 5:
        return text[:budget]
    head = max(1, (budget - 1) * 11 // 20)
    tail = max(1, budget - 1 - head)
    return text[:head].rstrip() + "…" + text[-tail:].lstrip()


def compress_text(value, budget: int, goal: str = "", prefix: bool = True) -> str:
    """Compress prose/code to ``budget`` characters while keeping anchors."""
    text = _text(value)
    budget = max(0, int(budget))
    if budget <= 0:
        return ""
    if len(text) <= budget:
        return text
    anchors = extract_anchors(text, limit=min(16, max(3, budget // 140)))
    header = ""
    if anchors:
        header = "[anchors] " + " | ".join(anchors)
        header = header[: max(80, int(budget * 0.45))].rstrip(" |")
    units = _units(text)
    terms = _goal_terms(goal)
    total = max(1, len(units))
    head = units[:2]
    tail = units[-2:] if len(units) > 2 else []
    middle = [unit for unit in units if unit not in head and unit not in tail]
    ranked = sorted(enumerate(middle),
                    key=lambda row: -_score(row[1], terms, row[0], total))
    head_text = _elide_units(head, max(40, budget // 5))
    # The newest unit matters most; a long penultimate line must not crowd it out.
    tail_text = _elide_units(list(reversed(tail)), max(40, budget // 4))
    reserve = len(header) + len(head_text) + len(tail_text) + 80
    body = _elide_units([unit for _, unit in ranked], max(40, budget - reserve))
    marker = f"{MARKER} {len(text):,} chars"
    result = "\n".join(part for part in ((marker if prefix else ""), header,
                                         head_text, body, tail_text) if part)
    if len(result.strip()) < 40:
        return _clip(text, budget)
    if len(result) > budget:
        result = _clip(result, budget)
    return result


def _compact_json_value(value, budget: int, goal: str):
    if isinstance(value, str):
        return compress_text(value, budget, goal)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    rendered = json.dumps(value, ensure_ascii=False, default=str)
    if len(rendered) <= budget:
        return value
    return compress_text(rendered, budget, goal)


def compress_observation(value, budget: int, goal: str = "", tool_id: str = "") -> str:
    """Compress one tool observation, preserving paths, errors and match lines."""
    text = str(value or "")
    budget = max(80, int(budget))
    if len(text) <= budget:
        return text
    label = str(tool_id or "")
    payload = text
    first, separator, rest = text.partition("\n")
    tool_match = _TOOL_LINE.match(first.strip())
    if tool_match and separator:
        label = tool_match.group(1)
        payload = rest
    parsed = None
    try:
        parsed = json.loads(payload)
    except (TypeError, ValueError):
        parsed = None
    if not isinstance(parsed, dict):
        prefix = (first + "\n") if tool_match and separator else ""
        return compress_text(prefix + payload, budget, goal)
    keep_keys = ("path", "error", "hint", "ok", "binary", "directory", "truncated",
                 "start_line", "end_line", "total_lines", "next_start_line",
                 "files_scanned", "engine", "query", "written", "changed", "deleted")
    compact: dict = {}
    for key in keep_keys:
        if key in parsed:
            compact[key] = _compact_json_value(parsed[key], 300, goal)
    header = json.dumps({**({"tool": label} if label else {}), **compact},
                        ensure_ascii=False, separators=(",", ":"))
    remaining = max(200, budget - len(header) - 2)
    if isinstance(parsed.get("matches"), list):
        rows = []
        for match in parsed["matches"][:60]:
            if not isinstance(match, dict):
                continue
            rows.append(f"{match.get('path', '?')}:{match.get('line', '?')} "
                        f"{str(match.get('text') or '').strip()}")
        if rows:
            selected = rows[:12] + (rows[-4:] if len(rows) > 16 else [])
            compact["matches"] = "\n".join(_dedupe(selected))
    elif isinstance(parsed.get("entries"), list):
        entries = parsed["entries"]
        paths = [str(item.get("path") or "") for item in entries if isinstance(item, dict)]
        compact["entries"] = f"{len(entries)} entries: " + ", ".join(paths[:20])
    elif isinstance(parsed.get("files"), list):
        files = parsed["files"]
        names = [str(item.get("name") or "") for item in files if isinstance(item, dict)]
        compact["files"] = f"{len(files)} files: " + ", ".join(names[:20])
    if isinstance(parsed.get("content"), str):
        compact["content"] = compress_text(parsed["content"], remaining, goal)
    elif isinstance(parsed.get("text"), str):
        compact["text"] = compress_text(parsed["text"], remaining, goal)
    elif isinstance(parsed.get("stdout"), str):
        compact["stdout"] = compress_text(parsed["stdout"], remaining // 2, goal)
        if isinstance(parsed.get("stderr"), str):
            compact["stderr"] = compress_text(parsed["stderr"], remaining // 2, goal)
    if "matches" not in compact and "entries" not in compact and "files" not in compact \
            and "content" not in compact and "text" not in compact \
            and "stdout" not in compact:
        body = compress_text(json.dumps(parsed, ensure_ascii=False, default=str),
                             remaining, goal)
        compact["body"] = body
    rendered = json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
    if len(rendered) > budget:
        rendered = compress_text(rendered, budget, goal)
    return rendered


def compact_turns(messages, budget: int, goal: str = "") -> str:
    """A structured episode card for dropped turns: tools, anchors, errors, decisions."""
    budget = max(200, int(budget))
    parts = []
    raw = 0
    tools: list[str] = []
    text_parts: list[str] = []
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        content = str(message.get("content") or "")
        raw += len(content)
        text_parts.append(content)
        tools.extend(re.findall(r'"tool"\s*:\s*"([\w.-]+)"', content))
        tools.extend(re.findall(r'invoke\s+name\s*=\s*"([\w.-]+)"', content, re.I))
    tools = _dedupe(tools)[:12]
    combined = "\n".join(text_parts)
    anchors = extract_anchors(combined, limit=16)
    errors = []
    decisions = []
    for line in combined.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if _ERROR.search(stripped) and len(errors) < 5:
            errors.append(stripped[:180])
        elif _DECISION.search(stripped) and len(decisions) < 5:
            decisions.append(stripped[:180])
    if tools:
        parts.append("Tools: " + ", ".join(tools))
    if anchors:
        parts.append("Anchors: " + " | ".join(anchors))
    if errors:
        parts.append("Errors:\n" + "\n".join(f"- {item}" for item in _dedupe(errors)))
    if decisions:
        parts.append("Decisions/constraints:\n"
                     + "\n".join(f"- {item}" for item in _dedupe(decisions)))
    last = next((str(message.get("content") or "") for message in reversed(messages or [])
                 if str(message.get("content") or "").strip()), "")
    header = f"{MARKER} episode: {len(messages or [])} turns, {raw:,} chars compacted."
    used = len(header) + sum(len(part) + 1 for part in parts)
    if last and used < budget - 120:
        parts.append("Last compacted turn:\n"
                     + compress_text(last, min(budget - used - 20, max(160, budget // 5)),
                                     goal))
    result = "\n".join([header, *parts])
    if len(result) > budget:
        result = compress_text(result, budget, goal)
    return result


__all__ = [
    "MARKER", "anchor_line", "compact_turns", "compress_observation",
    "compress_text", "extract_anchors",
]
