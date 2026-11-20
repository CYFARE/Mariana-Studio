"""Chat routes: completions (streamed or not) and saved conversations."""

from __future__ import annotations

import threading

from ..errors import BadRequestError, StudioError
from ..http import EventStream, Router, created

router = Router()
MAX_CHAT_BYTES = 16_000_000


def _chat_config(request) -> dict:
    """A chat request must be a JSON object, not an array or a scalar."""
    config = request.json
    if not isinstance(config, dict):
        raise BadRequestError("A chat request must be a JSON object with a model and messages")
    return config


@router.post("/api/chat")
def complete_chat(app, request):
    """Run a chat to completion and return the reply."""
    return app.gateway.chat(_chat_config(request))


@router.post("/api/chat/stream")
def stream_chat(app, request):
    """Server-sent events: ``start``, ``status``, ``delta``, ``reasoning``, then ``done``."""
    config = _chat_config(request)
    app.gateway.prepare(config)  # fail with a proper status before the stream opens
    cancelled = threading.Event()

    def events():
        try:
            yield from app.gateway.stream(config, cancelled)
        except StudioError as error:
            yield {"type": "error", **error.payload()}
        except Exception as error:  # noqa: BLE001 - the stream has started; report in-band
            yield {"type": "error", "error": str(error) or type(error).__name__, "code": "server_error"}
        finally:
            cancelled.set()

    return EventStream(events())


@router.post("/api/chat/swarm/stream")
def stream_chat_swarm(app, request):
    """Server-sent events: ``swarm_*`` progress, then ``done`` with the reply."""
    config = request.json
    prepared = app.chat_swarm.prepare(config)  # fail with a proper status before the stream opens
    cancelled = threading.Event()

    def events():
        try:
            yield from app.chat_swarm.stream(prepared, cancelled)
        except StudioError as error:
            yield {"type": "error", **error.payload()}
        except Exception as error:  # noqa: BLE001 - the stream has started; report in-band
            yield {"type": "error", "error": str(error) or type(error).__name__, "code": "server_error"}
        finally:
            cancelled.set()

    return EventStream(events())


@router.get("/api/chats")
def list_chats(app, request):
    """Saved conversations, pinned first, then newest."""
    return {"chats": app.chats.list()}


@router.post("/api/chats")
def create_chat(app, request):
    """Save a new conversation."""
    return created({"chat": app.chats.save(request.large_json(MAX_CHAT_BYTES))})


@router.delete("/api/chats")
def clear_chats(app, request):
    """Delete every saved conversation."""
    return {"deleted": app.chats.clear()}


@router.get("/api/chats/{id}")
def read_chat(app, request):
    """One conversation with all its messages."""
    return {"chat": app.chats.get(request.params["id"])}


@router.put("/api/chats/{id}")
def update_chat(app, request):
    """Replace the given fields of a conversation."""
    return {"chat": app.chats.save(request.large_json(MAX_CHAT_BYTES), request.params["id"])}


@router.delete("/api/chats/{id}")
def delete_chat(app, request):
    """Delete one conversation."""
    app.chats.delete(request.params["id"])
    return {"deleted": True, "id": request.params["id"]}
