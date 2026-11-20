"""Claude models through the official Anthropic SDK (optional dependency).

Differences from the generic backends, all dictated by the Messages API:

* sampling parameters are never sent — current Claude models reject
  ``temperature``/``top_p``/``top_k``; depth is controlled with ``effort``;
* ``max_tokens`` covers thinking as well as the visible reply, so the default
  is generous;
* a safety refusal is a normal response with ``stop_reason == "refusal"``; for
  models that support it Studio opts into server-side refusal fallbacks and
  reports which model actually answered.
"""

from __future__ import annotations

from ...errors import UnavailableError, UpstreamError
from ..catalog import ModelError
from ..endpoints import ANTHROPIC_HINT, resolve_key
from . import (
    anthropic_tools,
    canonical_calls,
    delta,
    done,
    reasoning,
    status,
    tool_calls_text,
)

DEFAULT_MODEL = "claude-opus-5-5"
DEFAULT_MAX_TOKENS = 16000
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")
FALLBACK_BETA = "server-side-fallback-2026-07-01"
# Models whose refusals can be re-run server-side on a fallback model.
FALLBACK_MODELS = frozenset({
    "claude-fable-5-2", "claude-fable-5-1", "claude-opus-5-5", "claude-opus-5",
    "claude-sonnet-5-5",
})


def _sdk():
    try:
        import anthropic
    except ImportError as error:
        raise UnavailableError(f"The Anthropic SDK is not installed; {ANTHROPIC_HINT}") from error
    return anthropic


def client_for(endpoint: dict):
    """Build a client; with no stored key the SDK resolves credentials itself
    (``ANTHROPIC_API_KEY``, ``ANTHROPIC_AUTH_TOKEN``, or an ``ant auth login`` profile)."""
    anthropic = _sdk()
    arguments = {}
    key = resolve_key(endpoint)
    if key:
        arguments["api_key"] = key
    if endpoint.get("base_url"):
        arguments["base_url"] = endpoint["base_url"]
    if endpoint.get("headers"):
        arguments["default_headers"] = dict(endpoint["headers"])
    return anthropic.Anthropic(**arguments)


def split_messages(messages: list[dict]) -> tuple[str, list[dict]]:
    """Separate the system prompt and make the history start with a user turn."""
    system = "\n\n".join(item["content"] for item in messages if item["role"] == "system")
    turns = [
        {"role": item["role"], "content": item["content"]}
        for item in messages if item["role"] in {"user", "assistant"}
    ]
    while turns and turns[0]["role"] != "user":
        turns.pop(0)
    if not turns:
        raise ModelError("Enter a message first")
    return system, turns


def _friendly(anthropic, error) -> Exception:
    """Translate SDK exceptions (most specific first) into actionable Studio errors."""
    if isinstance(error, anthropic.AuthenticationError):
        return UnavailableError(
            "Anthropic rejected the credentials. Set ANTHROPIC_API_KEY, run `ant auth login`, "
            "or store a key on this endpoint."
        )
    if isinstance(error, anthropic.PermissionDeniedError):
        return UpstreamError(f"This credential may not use the model: {error.message}")
    if isinstance(error, anthropic.NotFoundError):
        return UpstreamError(
            "Anthropic does not recognise this model id, or your organization cannot use it. "
            "Use Fetch models to pick a valid one."
        )
    if isinstance(error, anthropic.RateLimitError):
        retry = error.response.headers.get("retry-after") if error.response is not None else None
        return UpstreamError(
            "Anthropic rate limit reached" + (f"; retry in {retry}s." if retry else "; retry shortly.")
        )
    if isinstance(error, anthropic.BadRequestError):
        return UpstreamError(f"Anthropic rejected the request: {error.message}",
                             detail={"status": 400, "body": str(error.message)})
    if isinstance(error, anthropic.APIStatusError):
        if error.status_code >= 500:
            return UpstreamError(f"Anthropic service error ({error.status_code}); retry shortly.")
        return UpstreamError(f"Anthropic API error ({error.status_code}): {error.message}")
    if isinstance(error, anthropic.APIConnectionError):
        return UnavailableError("Cannot reach the Anthropic API; check the network connection.")
    return error


def _refusal_notice(message) -> str:
    details = getattr(message, "stop_details", None)
    category = getattr(details, "category", None)
    explanation = getattr(details, "explanation", None)
    notice = "The model declined this request"
    if category:
        notice += f" (safety category: {category})"
    notice += "."
    if explanation:
        notice += f" {explanation}"
    return notice


def _served_by_fallback(message) -> bool:
    iterations = getattr(getattr(message, "usage", None), "iterations", None) or []
    return any(getattr(entry, "type", "") == "fallback_message" for entry in iterations)


def _events(stream_manager, request):
    """Yield Studio events from one SDK message stream, then its final message."""
    with stream_manager as stream:
        for event in stream:
            if request.cancelled.is_set():
                break
            if event.type == "content_block_start":
                block = event.content_block
                if block.type == "thinking":
                    yield status("Thinking…")
                elif block.type == "fallback":
                    yield status(
                        f"{block.from_.model} declined; {block.to.model} is answering instead"
                    )
            elif event.type == "content_block_delta":
                if event.delta.type == "text_delta":
                    yield delta(event.delta.text)
                elif event.delta.type == "thinking_delta" and event.delta.thinking:
                    yield reasoning(event.delta.thinking)
        yield stream.get_final_message() if not request.cancelled.is_set() else None


def stream(gateway, entry, request):
    anthropic = _sdk()
    endpoint = gateway.endpoints.get(entry["id"])
    model = endpoint.get("model") or DEFAULT_MODEL
    system, turns = split_messages(request.messages)
    arguments = {"model": model, "max_tokens": request.max_tokens, "messages": turns}
    if system:
        arguments["system"] = system
    tools, aliases = anthropic_tools(request.options.get("tools"))
    native = bool(tools)
    if native:
        arguments["tools"] = tools
    effort = request.options.get("effort")
    if effort in EFFORT_LEVELS:
        arguments["output_config"] = {"effort": effort}
    if request.options.get("show_reasoning"):
        arguments["thinking"] = {"type": "adaptive", "display": "summarized"}
    use_fallback = (
        endpoint.get("refusal_fallback", True) and model in FALLBACK_MODELS
        and not request.options.get("disable_fallback")
    )
    client = client_for(endpoint)
    attempts = [True, False] if use_fallback else [False]
    final = None
    index = 0
    while index < len(attempts):
        with_fallback = attempts[index]
        emitted = False
        try:
            if with_fallback:
                manager = client.beta.messages.stream(
                    betas=[FALLBACK_BETA], fallbacks="default", **arguments,
                )
            else:
                manager = client.messages.stream(**arguments)
            for item in _events(manager, request):
                if isinstance(item, dict):
                    emitted = True
                    yield item
                else:
                    final = item
            break
        except anthropic.BadRequestError as error:
            # An endpoint that rejects native tools still gets the prompted JSON path.
            message = str(error).lower()
            if (native and not emitted
                    and any(word in message for word in ("tool", "schema", "function"))):
                native = False
                arguments.pop("tools", None)
                continue
            # An organization without the fallback beta rejects the header; retry plainly.
            if with_fallback and not emitted:
                index += 1
                continue
            raise _friendly(anthropic, error) from error
        except (anthropic.APIStatusError, anthropic.APIConnectionError) as error:
            raise _friendly(anthropic, error) from error
    if final is None:  # cancelled by the user
        yield done(usage={}, prompt_format="anthropic-messages", finish_reason="cancelled")
        return
    usage = final.usage
    result = {
        "usage": {"prompt_tokens": int(getattr(usage, "input_tokens", 0) or 0),
                  "completion_tokens": int(getattr(usage, "output_tokens", 0) or 0)},
        "prompt_format": "anthropic-messages",
        "finish_reason": {"end_turn": "stop", "max_tokens": "length"}.get(
            final.stop_reason, final.stop_reason or "stop"),
    }
    if final.stop_reason == "refusal":
        result["notice"] = _refusal_notice(final)
    elif final.stop_reason == "max_tokens":
        result["notice"] = (
            "The reply stopped at the max-tokens limit. Thinking counts toward that limit; "
            "raise Max tokens to let the model finish."
        )
    if _served_by_fallback(final) and final.stop_reason != "refusal":
        result["served_by"] = final.model
        result["notice"] = f"{model} declined this request; {final.model} answered instead."
    slots = []
    for block in getattr(final, "content", None) or []:
        if getattr(block, "type", "") == "tool_use":
            slots.append({"name": getattr(block, "name", ""),
                          "arguments": getattr(block, "input", {})})
    calls = canonical_calls(slots, aliases)
    if calls:
        yield delta(tool_calls_text(calls))
        result["finish_reason"] = "tool_calls"
    yield done(**result)


def list_models(endpoint: dict) -> list[dict]:
    """Models visible to this credential, with context and output limits."""
    anthropic = _sdk()
    try:
        return [
            {"id": model.id, "name": getattr(model, "display_name", "") or model.id,
             "context_length": getattr(model, "max_input_tokens", None),
             "max_output_tokens": getattr(model, "max_tokens", None)}
            for model in client_for(endpoint).models.list()
        ]
    except (anthropic.APIStatusError, anthropic.APIConnectionError) as error:
        raise _friendly(anthropic, error) from error
