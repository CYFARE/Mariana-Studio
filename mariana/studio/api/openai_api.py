"""An OpenAI-compatible API over every model in the catalog.

Point any OpenAI client at ``http://127.0.0.1:<port>/v1`` to use Mariana
checkpoints, local GGUF or Transformers models and saved endpoints through one
interface.  Errors follow the OpenAI shape (``{"error": {"message": …}}``).
"""

from __future__ import annotations

import threading
import time
import uuid

from ..errors import StudioError
from ..http import EventStream, Response, Router

router = Router()


def _error(error: StudioError) -> Response:
    return Response.json({"error": {
        "message": error.message, "type": "invalid_request_error" if error.status < 500 else "api_error",
        "code": error.code,
    }}, error.status)


def _config(body: dict) -> dict:
    config = {
        "model": body.get("model"), "messages": body.get("messages"),
        "max_tokens": body.get("max_completion_tokens") or body.get("max_tokens"),
    }
    for key in ("temperature", "top_p", "top_k", "seed", "stop", "device"):
        if body.get(key) is not None:
            config[key] = body[key]
    if isinstance(config.get("stop"), str):
        config["stop"] = [config["stop"]]
    return {key: value for key, value in config.items() if value is not None}


@router.get("/v1/models")
def list_models(app, request):
    """Chat-ready models in OpenAI list format."""
    return {"object": "list", "data": [
        {"id": entry["id"], "object": "model", "created": int(entry["modified"] or 0),
         "owned_by": entry["source"], "name": entry["name"]}
        for entry in app.gateway.catalog()
        if entry["available"] and entry["capabilities"].get("chat")
    ]}


@router.post("/v1/chat/completions")
def chat_completions(app, request):
    """Create a chat completion; set ``stream: true`` for server-sent chunks."""
    try:
        body = request.json
        config = _config(body)
        entry, _ = app.gateway.prepare(config)
    except StudioError as error:
        return _error(error)
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    base = {"id": completion_id, "created": created, "model": entry["id"]}
    if not body.get("stream"):
        try:
            result = app.gateway.chat(config)
        except StudioError as error:
            return _error(error)
        usage = result["usage"]
        return {
            **base, "object": "chat.completion",
            "choices": [{
                "index": 0, "finish_reason": result.get("finish_reason", "stop"),
                "message": {"role": "assistant", "content": "" if result.get("empty") else result["reply"]},
            }],
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0),
            },
        }
    cancelled = threading.Event()

    def chunk(delta: dict, finish=None, **extra) -> dict:
        return {**base, "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}], **extra}

    def events():
        try:
            yield chunk({"role": "assistant", "content": ""})
            for event in app.gateway.stream(config, cancelled):
                if event["type"] == "delta":
                    yield chunk({"content": event["text"]})
                elif event["type"] == "reasoning":
                    yield chunk({"reasoning_content": event["text"]})
                elif event["type"] == "done":
                    usage = event["usage"]
                    yield chunk({}, event.get("finish_reason", "stop"), usage={
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0),
                        "total_tokens": usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0),
                    })
        except StudioError as error:
            yield {"error": {"message": error.message, "type": "api_error", "code": error.code}}
        except Exception as error:  # noqa: BLE001 - the stream has started; report in-band
            yield {"error": {"message": str(error), "type": "api_error", "code": "server_error"}}
        finally:
            cancelled.set()
        yield "[DONE]"

    return EventStream(events())
