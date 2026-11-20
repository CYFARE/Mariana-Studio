"""Conversation storage: one JSON file per chat under ``agent-workspace/chats/``.

Conversations live on the server, not in the browser, so they survive a
change of port or browser and can be searched and exported.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import time
import uuid
from pathlib import Path

from ..workspace import chat_files_path, chats_path, read_json, write_json
from .errors import BadRequestError, NotFoundError, TooLargeError

MAX_MESSAGES = 2000
MAX_STORED_BYTES = 64_000_000
MAX_CONTENT_CHARS = 400_000
MAX_TITLE_CHARS = 120
MAX_ATTACHMENTS = 8
MAX_ATTACHMENT_NAME_CHARS = 200
MAX_ATTACHMENT_DATA_CHARS = 8_000_000
MAX_ATTACHMENT_TEXT_CHARS = 200_000
MAX_ATTACHMENTS_TOTAL_CHARS = 12_000_000
MAX_SWARM_CHARS = 200_000
MAX_SWARM_AGENTS = 12
MAX_SWARM_ITEMS = 20
MAX_SWARM_AGENT_OUTPUT = 8_000
MAX_SWARM_AGENT_TEXT = 4_000
MAX_SWARM_TEXT_CHARS = 2_000
MAX_SWARM_MEMORY_CHARS = 1_000
ROLES = ("system", "user", "assistant")
ATTACHMENT_KINDS = ("image", "text", "file")
MESSAGE_FIELDS = (
    "model", "model_name", "kind", "usage", "elapsed", "tokens_per_second",
    "first_token_seconds", "reasoning", "notice", "finish_reason", "served_by", "at", "error",
)
_ID = re.compile(r"chat-[a-z0-9]{6,32}")


def _clean_attachments(raw) -> list[dict]:
    """Keep the previewable parts of an attachment list, bounded and typed."""
    if not isinstance(raw, list):
        return []
    attachments = []
    total = 0
    for item in raw[:MAX_ATTACHMENTS]:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind") or "file").lower()
        if kind not in ATTACHMENT_KINDS:
            kind = "file"
        try:
            size = max(0, int(item.get("size") or 0))
        except (TypeError, ValueError, OverflowError):
            size = 0
        entry = {
            "id": _bounded_text(item.get("id") or f"a{len(attachments)}", 64),
            "name": _bounded_text(item.get("name") or "attachment", MAX_ATTACHMENT_NAME_CHARS),
            "kind": kind,
            "mime": _bounded_text(item.get("mime"), 120),
            "size": size,
        }
        language = item.get("language")
        if isinstance(language, str) and language.strip():
            entry["language"] = language.strip()[:40]
        if kind == "image":
            data_url = item.get("data_url")
            if (not isinstance(data_url, str) or not data_url.startswith("data:image/")
                    or "," not in data_url or data_url.endswith(",")):
                continue
            if len(data_url) > MAX_ATTACHMENT_DATA_CHARS:
                continue
            total += len(data_url)
            if total > MAX_ATTACHMENTS_TOTAL_CHARS:
                break
            entry["data_url"] = data_url
        elif kind == "text":
            text = item.get("text")
            if not isinstance(text, str):
                continue
            text = text[:MAX_ATTACHMENT_TEXT_CHARS]
            total += len(text)
            if total > MAX_ATTACHMENTS_TOTAL_CHARS:
                break
            entry["text"] = text
        attachments.append(entry)
    return attachments


def _title_from(messages: list[dict]) -> str:
    for message in messages:
        if message["role"] == "user":
            line = message["content"].strip().splitlines()[0] if message["content"].strip() else ""
            return (line[:60] + "…") if len(line) > 60 else (line or "New chat")
    return "New chat"


def _finite(value) -> bool:
    """Whether ``value`` is a JSON number that survives a browser parse."""
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def _bounded_json(value, limit: int):
    """Return ``value`` when its finite JSON form fits, else a bounded preview."""
    try:
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, default=str)
    except (TypeError, ValueError, RecursionError):
        return None
    if len(encoded) <= limit:
        try:
            return json.loads(encoded)
        except (TypeError, ValueError, RecursionError):
            return None
    return {"truncated": True, "preview": encoded[:limit]}


def _clean_usage(value) -> dict:
    """Usage counters and flags that are safe to send to the browser."""
    if not isinstance(value, dict):
        return {}
    usage = {}
    for key, item in list(value.items())[:32]:
        key = str(key)[:40]
        if isinstance(item, bool) or _finite(item):
            usage[key] = item
        elif isinstance(item, str):
            usage[key] = item[:200]
    return usage


def _clean_settings(value) -> dict:
    """Chat settings are scalars and short string lists; keep only those."""
    if not isinstance(value, dict):
        return {}
    settings = {}
    for key, item in list(value.items())[:64]:
        key = str(key)[:80]
        if isinstance(item, bool) or _finite(item):
            settings[key] = item
        elif isinstance(item, str):
            settings[key] = item[:20_000]
        elif isinstance(item, list):
            cleaned = []
            for entry in item[:64]:
                if isinstance(entry, str):
                    cleaned.append(entry[:200])
                elif isinstance(entry, bool) or _finite(entry):
                    cleaned.append(entry)
            settings[key] = cleaned
    return settings


def _bounded_text(value, limit: int) -> str:
    """Stringify untrusted data without letting deep nesting raise."""
    if value is None:
        return ""
    try:
        return str(value)[:limit]
    except (RecursionError, TypeError, ValueError):
        return ""


def _swarm_text(value, limit: int) -> str:
    return _bounded_text(value, limit)


def _clean_swarm(raw) -> dict | None:
    """Keep the bounded, renderable parts of a Chat swarm card."""
    if not isinstance(raw, dict):
        return None
    agents = []
    raw_agents = raw.get("agents") if isinstance(raw.get("agents"), list) else []
    for item in raw_agents[:MAX_SWARM_AGENTS]:
        if not isinstance(item, dict):
            continue
        agent = {
            "agent": _swarm_text(item.get("agent") or item.get("id"), 48),
            "label": _swarm_text(item.get("label"), 80),
            "role": _swarm_text(item.get("role"), 40),
            "task": _swarm_text(item.get("task"), MAX_SWARM_TEXT_CHARS),
            "status": _swarm_text(item.get("status"), 20),
            "output": _swarm_text(item.get("output"), MAX_SWARM_AGENT_OUTPUT),
            "text": _swarm_text(item.get("text"), MAX_SWARM_AGENT_TEXT),
            "summary": _swarm_text(item.get("summary"), 600),
            "error": _swarm_text(item.get("error"), 600),
            "note": _swarm_text(item.get("note"), 600),
        }
        if item.get("lastTool"):
            agent["lastTool"] = _swarm_text(item["lastTool"], 80)
        if "ok" in item:
            agent["ok"] = bool(item["ok"])
        for key in ("score", "elapsed"):
            if _finite(item.get(key)):
                agent[key] = item[key]
        for key in ("version", "steps"):
            if isinstance(item.get(key), int):
                agent[key] = item[key]
        if agent["agent"] or agent["label"] or agent["output"]:
            agents.append(agent)
    meta = {}
    raw_meta = raw.get("meta") if isinstance(raw.get("meta"), dict) else {}
    for key, limit in (("id", 80), ("strategy", 40), ("leader", 200), ("model", 200),
                       ("phase", 20), ("status", 20), ("summary", 1_000),
                       ("reason", 600), ("leaderNote", 600)):
        if raw_meta.get(key) is not None:
            meta[key] = _swarm_text(raw_meta[key], limit)
    for key in ("round", "rounds", "messageCount", "reviewCount", "memoryCount"):
        if isinstance(raw_meta.get(key), int):
            meta[key] = raw_meta[key]
    if _finite(raw_meta.get("elapsed")):
        meta["elapsed"] = raw_meta["elapsed"]
    for key in ("finished", "ok"):
        if key in raw_meta:
            meta[key] = bool(raw_meta[key])
    if isinstance(raw_meta.get("warnings"), list):
        meta["warnings"] = [_swarm_text(value, 400)
                            for value in raw_meta["warnings"][:8] if value]
    if isinstance(raw_meta.get("tools"), list):
        meta["tools"] = [_swarm_text(value, 80) for value in raw_meta["tools"][:12]
                        if value]
    reviews = []
    raw_reviews = raw.get("reviews") if isinstance(raw.get("reviews"), list) else []
    for item in raw_reviews[-MAX_SWARM_ITEMS:]:
        if not isinstance(item, dict):
            continue
        review = {
            "reviewer": _swarm_text(item.get("reviewer"), 48),
            "target": _swarm_text(item.get("target"), 48),
            "summary": _swarm_text(item.get("summary"), 400),
            "issues": [_swarm_text(value, 300) for value in (item.get("issues") or [])[:4]
                       if value] if isinstance(item.get("issues"), list) else [],
            "suggestions": [_swarm_text(value, 300)
                            for value in (item.get("suggestions") or [])[:4]
                            if value] if isinstance(item.get("suggestions"), list) else [],
        }
        if isinstance(item.get("round"), int):
            review["round"] = item["round"]
        if _finite(item.get("score")):
            review["score"] = item["score"]
        reviews.append(review)
    messages = []
    raw_messages = raw.get("messages") if isinstance(raw.get("messages"), list) else []
    for item in raw_messages[-MAX_SWARM_ITEMS:]:
        if not isinstance(item, dict):
            continue
        messages.append({
            "from": _swarm_text(item.get("from"), 48),
            "to": _swarm_text(item.get("to"), 48),
            "kind": _swarm_text(item.get("kind") or item.get("message_kind"), 20),
            "text": _swarm_text(item.get("text"), 1_500),
            "round": item.get("round") if isinstance(item.get("round"), int) else 0,
        })
    memory = []
    raw_memory = raw.get("memory") if isinstance(raw.get("memory"), list) else []
    for item in raw_memory[-MAX_SWARM_ITEMS:]:
        if not isinstance(item, dict):
            continue
        entry = {"key": _swarm_text(item.get("key"), 80)}
        if "value" in item:
            entry["value"] = _bounded_json(item["value"], MAX_SWARM_MEMORY_CHARS)
        for key in ("version",):
            if isinstance(item.get(key), int):
                entry[key] = item[key]
        if item.get("author"):
            entry["author"] = _swarm_text(item["author"], 48)
        memory.append(entry)
    if not (agents or reviews or messages or memory or meta):
        return None
    entry = {"agents": agents, "meta": meta, "reviews": reviews,
             "messages": messages, "memory": memory}
    if len(json.dumps(entry, ensure_ascii=False, allow_nan=False, default=str)) > MAX_SWARM_CHARS:
        for agent in agents:
            agent.pop("text", None)
            agent["output"] = agent["output"][:2_000]
        entry["reviews"] = reviews[:6]
        entry["messages"] = messages[:10]
        entry["memory"] = memory[:10]
        if (len(json.dumps(entry, ensure_ascii=False, allow_nan=False, default=str))
                > MAX_SWARM_CHARS):
            return None
    return entry


def _clean_messages(raw) -> list[dict]:
    if not isinstance(raw, list):
        raise BadRequestError("messages must be a list")
    if len(raw) > MAX_MESSAGES:
        raise BadRequestError(
            f"A conversation holds at most {MAX_MESSAGES} messages; export it and start a new one"
        )
    messages = []
    for item in raw:
        if not isinstance(item, dict) or item.get("role") not in ROLES:
            continue
        content = item.get("content")
        if not isinstance(content, str):
            continue
        message = {"role": item["role"], "content": content[:MAX_CONTENT_CHARS]}
        for field in MESSAGE_FIELDS:
            value = item.get(field)
            if value is None:
                continue
            if field == "usage":
                usage = _clean_usage(value)
                if usage:
                    message[field] = usage
            elif field in {"elapsed", "tokens_per_second", "first_token_seconds"}:
                if _finite(value):
                    message[field] = value
            else:
                message[field] = _bounded_text(value, 20_000)
        attachments = _clean_attachments(item.get("attachments"))
        if attachments:
            message["attachments"] = attachments
        swarm = _clean_swarm(item.get("swarm"))
        if swarm:
            message["swarm"] = swarm
        messages.append(message)
    return messages


def _read_messages(raw) -> list[dict]:
    """Sanitise stored messages without rejecting the whole conversation."""
    if not isinstance(raw, list):
        return []
    try:
        return _clean_messages(raw[:MAX_MESSAGES])
    except BadRequestError:
        return []


def _read_document(data: dict, chat_id: str) -> dict:
    """Return a stored conversation in the shape the API promises."""
    document = dict(data)
    document["id"] = chat_id  # the file name wins over a stale internal id
    document["messages"] = _read_messages(document.get("messages"))
    for key in ("model", "model_name", "system_prompt"):
        value = document.get(key)
        document[key] = _bounded_text(value, 20_000)
    if not isinstance(document.get("settings"), dict):
        document["settings"] = {}
    else:
        document["settings"] = _clean_settings(document["settings"])
    for key in ("created", "updated"):
        value = document.get(key)
        document[key] = value if _finite(value) else None
    title = document.get("title")
    if not isinstance(title, str) or not title.strip():
        document["title"] = _title_from(document["messages"])
    else:
        document["title"] = title[:MAX_TITLE_CHARS]
    if "titled" in document:
        document["titled"] = bool(document["titled"])
    else:
        # Infer a user-chosen title for documents saved before the flag existed.
        derived = _title_from(document["messages"])
        document["titled"] = (bool(document["title"]) and document["title"] != "New chat"
                              and document["title"] != derived)
    document["pinned"] = bool(document.get("pinned"))
    return document


class ChatStore:
    """Create, read, update and delete saved conversations."""

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.folder = chats_path(root)
        # Listing only needs the title, preview and timestamps.  Cache that
        # metadata by file state so image attachments are not parsed again on
        # every refresh.
        self._listing: dict[Path, tuple[tuple[int, int], dict | None]] = {}

    def _path(self, chat_id: str) -> Path:
        if not _ID.fullmatch(str(chat_id or "")):
            raise NotFoundError("Conversation not found")
        return self.folder / f"{chat_id}.json"

    def _metadata(self, path: Path) -> dict | None:
        try:
            stat = path.stat()
        except OSError:
            return None
        key = (stat.st_mtime_ns, stat.st_size)
        cached = self._listing.get(path)
        if cached and cached[0] == key:
            return cached[1]
        if stat.st_size > MAX_STORED_BYTES:
            self._listing[path] = (key, None)
            return None
        data = read_json(path)
        if not isinstance(data, dict):
            self._listing[path] = (key, None)
            return None
        chat_id = path.stem  # the file name is the conversation's identity
        if not _ID.fullmatch(chat_id):
            self._listing[path] = (key, None)
            return None
        messages = data.get("messages") or []
        if not isinstance(messages, list):
            messages = []
        last = next((m.get("content", "") for m in reversed(messages)
                     if isinstance(m, dict) and isinstance(m.get("content"), str)
                     and m.get("content", "").strip() and m.get("role") != "system"), "")
        updated = data.get("updated")
        if not _finite(updated):
            updated = 0
        created = data.get("created")
        if not _finite(created):
            created = None
        title = data.get("title")
        if not isinstance(title, str) or not title.strip():
            title = "New chat"
        else:
            title = title[:MAX_TITLE_CHARS]
        metadata = {
            "id": chat_id, "title": title,
            "model": data.get("model") if isinstance(data.get("model"), str) else "",
            "model_name": data.get("model_name") if isinstance(data.get("model_name"), str) else "",
            "created": created, "updated": updated,
            "messages": min(len(messages), MAX_MESSAGES), "pinned": bool(data.get("pinned")),
            "preview": " ".join(str(last)[:2_000].split())[:140],
        }
        self._listing[path] = (key, metadata)
        return metadata

    def list(self) -> list[dict]:
        chats = []
        for path in self.folder.glob("chat-*.json"):
            metadata = self._metadata(path)
            if metadata:
                chats.append(dict(metadata))
        chats.sort(key=lambda item: (not item["pinned"], -(item["updated"] or 0)))
        return chats

    def get(self, chat_id: str) -> dict:
        path = self._path(chat_id)
        try:
            too_large = path.stat().st_size > MAX_STORED_BYTES
        except OSError:
            raise NotFoundError("Conversation not found") from None
        if too_large:
            raise TooLargeError(
                "This conversation is too large to open. Move or delete its file in "
                "agent-workspace/chats and refresh the list."
            )
        data = read_json(path)
        if not isinstance(data, dict):
            raise NotFoundError("Conversation not found")
        return _read_document(data, chat_id)

    def save(self, document: dict, chat_id: str | None = None) -> dict:
        """Create a conversation, or replace the fields given for an existing one."""
        if not isinstance(document, dict):
            raise BadRequestError("A conversation must be a JSON object")
        now = time.time()
        if chat_id:
            current = self.get(chat_id)
        else:
            current = {"id": f"chat-{uuid.uuid4().hex[:12]}", "created": now, "messages": [],
                       "title": "", "titled": False, "pinned": False}
        if "messages" in document:
            current["messages"] = _clean_messages(document["messages"])
        for key in ("model", "model_name", "system_prompt"):
            if key in document:
                current[key] = None if document[key] is None else str(document[key])[:20_000]
        if isinstance(document.get("settings"), dict):
            current["settings"] = _clean_settings(document["settings"])
        if "pinned" in document:
            current["pinned"] = bool(document["pinned"])
        title = str(document.get("title") or "").strip()[:MAX_TITLE_CHARS]
        if title:
            current["title"] = title
            current["titled"] = True
        elif not current.get("titled"):
            current["title"] = _title_from(current["messages"])
        current["updated"] = now
        try:
            json.dumps(current, ensure_ascii=False, allow_nan=False, default=str)
        except (TypeError, ValueError, RecursionError):
            raise BadRequestError(
                "The conversation contains a value JSON cannot store (NaN, infinity or "
                "excessive nesting); remove it and save again"
            ) from None
        write_json(self._path(current["id"]), current, indent=None)
        self._listing.pop(self._path(current["id"]), None)
        return current

    def delete(self, chat_id: str) -> None:
        path = self._path(chat_id)
        if not path.exists():
            raise NotFoundError("Conversation not found")
        path.unlink()
        self._listing.pop(path, None)
        shutil.rmtree(chat_files_path(self.root, chat_id, create=False), ignore_errors=True)

    def clear(self) -> int:
        count = 0
        for path in self.folder.glob("chat-*.json"):
            shutil.rmtree(chat_files_path(self.root, path.stem, create=False), ignore_errors=True)
            path.unlink(missing_ok=True)
            count += 1
        return count
