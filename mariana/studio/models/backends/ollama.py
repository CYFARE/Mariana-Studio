"""Ollama's native chat API (``/api/chat``), streamed as newline-delimited JSON."""

from __future__ import annotations

import json

from ...errors import UnavailableError, UpstreamError
from ..context import model_context, positive_int
from . import (
    canonical_calls,
    delta,
    done,
    native_tools,
    reasoning,
    tool_calls_text,
)
from .openai import open_json, read_json

TIMEOUT = 600


def api_url(base_url: str, path: str) -> str:
    base = base_url.rstrip("/")
    for suffix in ("/api/chat", "/api"):
        if base.endswith(suffix):
            base = base[:-len(suffix)]
            break
    return f"{base}/api/{path}"


def connection(gateway, entry: dict) -> tuple[str, str, dict]:
    detail = entry.get("detail") or {}
    if entry["id"].startswith("endpoint:"):
        endpoint = gateway.endpoints.get(entry["id"])
        return endpoint["base_url"], endpoint["model"], dict(endpoint.get("headers") or {})
    return detail["base_url"], detail.get("model") or entry["name"], {}


def stream(gateway, entry, request):
    base_url, model, headers = connection(gateway, entry)
    options = {
        "temperature": request.temperature, "top_p": request.top_p,
        "num_predict": request.max_tokens,
    }
    if request.top_k:
        options["top_k"] = request.top_k
    if request.options.get("n_ctx"):
        try:
            options["num_ctx"] = int(request.options["n_ctx"])
        except (TypeError, ValueError):
            pass
    if request.options.get("seed") not in (None, ""):
        try:
            options["seed"] = int(request.options["seed"])
        except (TypeError, ValueError):
            pass
    payload = {"model": model, "messages": request.messages, "stream": True, "options": options}
    tools, aliases = native_tools(request.options.get("tools"))
    if tools:
        payload["tools"] = tools
    try:
        response = open_json(api_url(base_url, "chat"), payload, headers, timeout=TIMEOUT)
    except UpstreamError as error:
        if tools and "tool" in str(error).lower():
            payload.pop("tools", None)  # older Ollama: keep the prompted JSON path
            response = open_json(api_url(base_url, "chat"), payload, headers, timeout=TIMEOUT)
        else:
            raise
    final: dict = {}
    finish = None
    slots = []
    seen = set()
    try:
        for raw in response:
            if request.cancelled.is_set():
                finish = "cancelled"
                break
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except ValueError:
                continue
            if not isinstance(chunk, dict):
                continue
            if chunk.get("error"):
                raise UpstreamError(f"Ollama error: {chunk['error']}")
            message = chunk.get("message") or {}
            if message.get("thinking"):
                yield reasoning(str(message["thinking"]))
            if message.get("content"):
                yield delta(str(message["content"]))
            for call in message.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                function = call.get("function") or {}
                signature = json.dumps({"name": function.get("name"),
                                        "arguments": function.get("arguments")},
                                       sort_keys=True, default=str)
                if signature in seen:
                    continue
                seen.add(signature)
                slots.append({"name": function.get("name"),
                              "arguments": function.get("arguments")})
            if chunk.get("done") or "eval_count" in chunk:
                final = chunk
    finally:
        response.close()
    reason = final.get("done_reason")
    calls = canonical_calls(slots, aliases)
    if calls and finish != "cancelled":
        yield delta(tool_calls_text(calls))
        finish = "tool_calls"
    yield done(
        usage={"prompt_tokens": int(final.get("prompt_eval_count") or 0),
               "completion_tokens": int(final.get("eval_count") or 0)},
        prompt_format="ollama-chat",
        finish_reason=finish or ("length" if reason == "length" else "stop"),
    )


def list_models(base_url: str, headers: dict | None = None, timeout: float = 10) -> list[dict]:
    with open_json(api_url(base_url, "tags"), None, headers or {}, timeout=timeout) as response:
        body = read_json(response)
    models = body.get("models") if isinstance(body, dict) else None
    if not isinstance(models, list):
        raise UpstreamError("Ollama did not return a model list")
    return sorted(
        ({"id": str(item.get("name") or item.get("model")), "context_length": None}
         for item in models if isinstance(item, dict) and (item.get("name") or item.get("model"))),
        key=lambda model: model["id"].lower(),
    )


def context_length(base_url: str, model: str, headers: dict | None = None,
                   timeout: float = 2.5) -> int:
    """The architecture maximum from Ollama's ``/api/show``, when it reports one."""
    payload = {"model": str(model or "")}
    if not payload["model"]:
        return 0
    try:
        with open_json(api_url(base_url, "show"), payload, headers or {}, timeout=timeout) as response:
            body = read_json(response)
    except (UnavailableError, UpstreamError, OSError, ValueError):
        return 0
    if not isinstance(body, dict):
        return 0
    values = [positive_int(body.get("context_length")), model_context(body) or 0]
    return max(values)
