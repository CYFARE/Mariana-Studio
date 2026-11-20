"""OpenAI-compatible HTTP servers: cloud providers and local runtimes alike.

Streams ``/chat/completions`` with server-sent events and degrades gracefully:
servers that ignore ``stream`` and answer with one JSON document still work,
and parameters a provider rejects are dropped or renamed and the call retried.
"""

from __future__ import annotations

import json
import socket
import urllib.error
import urllib.parse
import urllib.request

from ...errors import UnavailableError, UpstreamError
from ..context import model_context, positive_int
from ..endpoints import resolve_key
from . import (
    canonical_calls,
    delta,
    done,
    native_tools,
    reasoning,
    tool_calls_text,
)

TIMEOUT = 300
MAX_PARAMETER_RETRIES = 3


def chat_url(base_url: str, api_path: str = "") -> str:
    base = base_url.rstrip("/")
    path = api_path or "/chat/completions"
    if path.startswith("http"):
        return path
    if base.endswith("/chat/completions"):
        return base
    return base + ("" if path.startswith("/") else "/") + path


def open_json(url: str, payload: dict | None, headers: dict, timeout: float = TIMEOUT,
              method: str | None = None):
    """Open a JSON request and return the live response, mapping failures to Studio errors."""
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url, data=body, method=method or ("POST" if body is not None else "GET"),
        headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream",
                 **headers},
    )
    try:
        return urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:600]
        raise UpstreamError(
            f"Endpoint returned HTTP {error.code}: {error_text(detail)}",
            detail={"status": error.code, "body": detail},
        ) from error
    except urllib.error.URLError as error:
        raise UnavailableError(f"Cannot reach endpoint: {error.reason}") from error
    except (TimeoutError, socket.timeout) as error:
        raise UpstreamError("Endpoint timed out") from error


def error_text(body: str) -> str:
    """The human message inside a provider error body, or the body itself."""
    try:
        value = json.loads(body)
    except ValueError:
        return body.strip() or "no details"
    error = value.get("error") if isinstance(value, dict) else None
    if isinstance(error, dict):
        return str(error.get("message") or error)
    if isinstance(error, str):
        return error
    if isinstance(value, dict) and value.get("message"):
        return str(value["message"])
    return body.strip() or "no details"


def read_json(response) -> dict:
    try:
        return json.loads(response.read().decode("utf-8"))
    except ValueError as error:
        raise UpstreamError("Endpoint returned non-JSON data") from error


def _text(content) -> str:
    """Message content as text: a string, or a list of ``{"text": ...}`` parts."""
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return "" if content is None else str(content)


def adapt_payload(payload: dict, error: UpstreamError) -> dict | None:
    """Return a payload without the parameter a provider just rejected, or ``None``."""
    if error.detail.get("status") not in {400, 422}:
        return None
    text = str(error.detail.get("body", "")).lower()
    adapted = dict(payload)
    if "stream_options" in text and "stream_options" in adapted:
        adapted.pop("stream_options")
    elif "max_completion_tokens" in text and "max_tokens" in adapted:
        adapted["max_completion_tokens"] = adapted.pop("max_tokens")
    elif "top_k" in text and "top_k" in adapted:
        adapted.pop("top_k")
    elif "top_p" in text and "top_p" in adapted:
        adapted.pop("top_p")
    elif "temperature" in text and "temperature" in adapted:
        adapted.pop("temperature")
    elif "seed" in text and "seed" in adapted:
        adapted.pop("seed")
    elif "reasoning_effort" in text and "reasoning_effort" in adapted:
        adapted.pop("reasoning_effort")
    elif ("reasoning_content" in text or "reasoning content" in text) and "tools" in adapted:
        # DeepSeek requires prior thinking to be sent back with native tools; the
        # prompted path is the safe fallback when that history is not available.
        adapted.pop("tools", None)
        adapted.pop("tool_choice", None)
    elif "tool" in text and ("tools" in adapted or "tool_choice" in adapted):
        # An endpoint that cannot call functions still gets the prompted JSON path.
        adapted.pop("tools", None)
        adapted.pop("tool_choice", None)
    else:
        return None
    return adapted


def build_payload(model: str, request, local: bool) -> dict:
    payload = {
        "model": model, "messages": request.messages, "temperature": request.temperature,
        "top_p": request.top_p, "max_tokens": request.max_tokens, "stream": True,
        "stream_options": {"include_usage": True},
    }
    if local and request.top_k:
        payload["top_k"] = request.top_k  # not part of the OpenAI API; local servers accept it
    options = request.options
    if options.get("seed") not in (None, ""):
        try:
            payload["seed"] = int(options["seed"])
        except (TypeError, ValueError):
            pass
    stops = options.get("stop")
    if isinstance(stops, list) and stops:
        payload["stop"] = [str(item) for item in stops[:4]]
    effort = options.get("reasoning_effort")
    if effort in {"minimal", "low", "medium", "high", "max"}:
        # Reasoning endpoints (DeepSeek, o-series) honour this; adapt_payload drops it
        # automatically when a provider rejects the parameter.
        payload["reasoning_effort"] = effort
    tools, _ = native_tools(options.get("tools"))
    if tools and options.get("provider") != "deepseek":
        # DeepSeek requires prior reasoning to be replayed when native tools are sent;
        # its DSML/text tool calls are parsed instead, so the text path is used.
        payload["tools"] = tools
    return payload


def stream_chat(url: str, payload: dict, headers: dict, request, prompt_format: str = "openai-chat"):
    """Yield chat events from one OpenAI-compatible completion call."""
    response = None
    for _ in range(MAX_PARAMETER_RETRIES + 1):
        try:
            response = open_json(url, payload, headers)
            break
        except UpstreamError as error:
            adapted = adapt_payload(payload, error)
            if adapted is None:
                raise
            payload = adapted
    if response is None:
        raise UpstreamError("Endpoint rejected every parameter combination Studio tried")
    usage: dict = {}
    finish = None
    slots: dict[int, dict] = {}
    _, aliases = native_tools(request.options.get("tools"))
    try:
        content_type = response.headers.get("Content-Type", "")
        if "text/event-stream" not in content_type:
            body = read_json(response)
            if isinstance(body.get("error"), (dict, str)):
                raise UpstreamError(f"Endpoint error: {error_text(json.dumps(body))}")
            try:
                choice = body["choices"][0]
                message = choice["message"]
            except (KeyError, IndexError, TypeError) as error:
                raise UpstreamError(
                    "Endpoint response did not contain choices[0].message.content"
                ) from error
            thinking = _text(message.get("reasoning_content") or message.get("reasoning"))
            if thinking:
                yield reasoning(thinking)
            text = _text(message.get("content"))
            if text:
                yield delta(text)
            for index, call in enumerate(message.get("tool_calls") or []):
                if isinstance(call, dict):
                    function = call.get("function") or {}
                    slots[index] = {"name": function.get("name"),
                                    "arguments": function.get("arguments")}
            legacy = message.get("function_call")
            if isinstance(legacy, dict) and legacy.get("name"):
                slots[len(slots)] = {"name": legacy.get("name"),
                                     "arguments": legacy.get("arguments")}
            usage, finish = body.get("usage") or {}, choice.get("finish_reason")
        else:
            for raw in response:
                if request.cancelled.is_set():
                    finish = "cancelled"
                    break
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                try:
                    chunk = json.loads(data)
                except ValueError:
                    continue
                if not isinstance(chunk, dict):
                    continue
                if chunk.get("error"):
                    raise UpstreamError(f"Endpoint error: {error_text(json.dumps(chunk))}")
                if isinstance(chunk.get("usage"), dict):
                    usage = chunk["usage"]
                for choice in chunk.get("choices") or []:
                    part = choice.get("delta") or {}
                    thinking = _text(part.get("reasoning_content") or part.get("reasoning"))
                    if thinking:
                        yield reasoning(thinking)
                    text = _text(part.get("content"))
                    if text:
                        yield delta(text)
                    for call in part.get("tool_calls") or []:
                        if not isinstance(call, dict):
                            continue
                        index = call.get("index")
                        if not isinstance(index, int):
                            # A provider that omits the index streams one call; keep
                            # appending its fragments to that slot.
                            index = len(slots) - 1 if slots else 0
                        slot = slots.setdefault(index, {"name": "", "arguments": ""})
                        function = call.get("function") or {}
                        name = function.get("name")
                        if name:
                            if not slot["name"]:
                                slot["name"] = str(name)
                            elif slot["name"] != str(name):
                                slot["name"] += str(name)
                        arguments = function.get("arguments")
                        if isinstance(arguments, dict):
                            slot["arguments"] = arguments
                        elif arguments:
                            current = slot.get("arguments")
                            if not isinstance(current, dict):
                                slot["arguments"] = f"{current or ''}{arguments}"
                    legacy = part.get("function_call")
                    if isinstance(legacy, dict):
                        slot = slots.setdefault(0, {"name": "", "arguments": ""})
                        if legacy.get("name") and not slot["name"]:
                            slot["name"] = str(legacy["name"])
                        arguments = legacy.get("arguments")
                        if isinstance(arguments, dict):
                            slot["arguments"] = arguments
                        elif arguments:
                            current = slot.get("arguments")
                            if not isinstance(current, dict):
                                slot["arguments"] = f"{current or ''}{arguments}"
                    finish = choice.get("finish_reason") or finish
    finally:
        response.close()
    calls = canonical_calls([slots[index] for index in sorted(slots)], aliases)
    if calls and finish != "cancelled":
        # The loops already understand this canonical JSON; the text path and DSML
        # remain available when an endpoint ignores or rejects native tools.
        yield delta(tool_calls_text(calls))
        finish = "tool_calls"
    yield done(
        usage={"prompt_tokens": int(usage.get("prompt_tokens") or 0),
               "completion_tokens": int(usage.get("completion_tokens") or 0)},
        prompt_format=prompt_format, finish_reason=finish or "stop",
    )


def connection(gateway, entry: dict) -> tuple[str, str, dict, bool]:
    """Return ``(url, model, headers, is_local)`` for a saved endpoint or a discovered server."""
    detail = entry.get("detail") or {}
    if entry["id"].startswith("endpoint:"):
        endpoint = gateway.endpoints.get(entry["id"])
        headers = dict(endpoint.get("headers") or {})
        key = resolve_key(endpoint)
        if key:
            headers.setdefault("Authorization", f"Bearer {key}")
        return (chat_url(endpoint["base_url"], endpoint.get("api_path", "")), endpoint["model"],
                headers, not detail.get("remote", True))
    return chat_url(detail["base_url"]), detail.get("model") or entry["name"], {}, True


def stream(gateway, entry, request):
    url, model, headers, local = connection(gateway, entry)
    yield from stream_chat(url, build_payload(model, request, local), headers, request)


def list_models(base_url: str, headers: dict, timeout: float = 15) -> list[dict]:
    """Model ids offered by an OpenAI-compatible server (``GET /models``)."""
    with open_json(base_url.rstrip("/") + "/models", None, headers, timeout=timeout) as response:
        body = read_json(response)
    items = body.get("data") if isinstance(body, dict) else body
    if not isinstance(items, list):
        raise UpstreamError("Endpoint did not return a model list")
    models = []
    for item in items:
        model_id = item.get("id") if isinstance(item, dict) else item
        if model_id:
            models.append({
                "id": str(model_id),
                "context_length": model_context(item) if isinstance(item, dict) else None,
            })
    return sorted(models, key=lambda model: model["id"].lower())


def _local_host(base_url: str) -> bool:
    host = urllib.parse.urlparse(base_url).hostname or ""
    return host in {"127.0.0.1", "localhost", "::1"} or host.endswith(".local")


def context_length(base_url: str, model: str, headers: dict | None = None,
                   timeout: float = 2.5) -> int:
    """Ask an OpenAI-compatible server for a model's context window.

    The ``/models`` payload is the first source (vLLM, LM Studio and llama.cpp
    expose different field names there).  Local llama.cpp servers also answer
    ``/props`` with the configured ``n_ctx`` when their model list is silent.
    """
    try:
        for item in list_models(base_url, headers or {}, timeout=timeout):
            if item["id"] == model:
                value = positive_int(item.get("context_length"))
                if value:
                    return value
    except (UnavailableError, UpstreamError, OSError, ValueError):
        pass
    if not _local_host(base_url):
        return 0
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    try:
        with open_json(root + "/props", None, headers or {}, timeout=timeout) as response:
            body = read_json(response)
    except (UnavailableError, UpstreamError, OSError, ValueError):
        return 0
    if not isinstance(body, dict):
        return 0
    values = [positive_int(body.get("n_ctx"))]
    settings = body.get("default_generation_settings")
    if isinstance(settings, dict):
        values.append(positive_int(settings.get("n_ctx")))
        values.append(model_context(settings) or 0)
    return max(values)
