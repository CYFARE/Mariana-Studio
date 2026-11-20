"""GGUF models through llama.cpp: a managed ``llama-server`` or the Python binding."""

from __future__ import annotations

import threading
from pathlib import Path

from ..catalog import GGUF_HINT, ModelError
from .. import gguf
from . import StopFilter, delta, done, status, transcript_prompt
from .openai import build_payload, chat_url, stream_chat


def _context(gateway, entry: dict, request) -> int:
    native = gguf.describe(Path(entry["absolute_path"])).get("ctx_len")
    return gateway.llama.context_for(request.options.get("n_ctx"), native)


def stream_server(gateway, entry, request):
    """Load the model into the managed llama-server, then stream over its OpenAI API."""
    n_ctx = _context(gateway, entry, request)
    progress: dict = {}

    def load():
        try:
            progress["url"] = gateway.llama.ensure(
                entry["absolute_path"], n_ctx,
                on_status=lambda message: progress.__setitem__("status", message),
                cancelled=request.cancelled, label=entry["name"],
            )
        except BaseException as error:  # noqa: BLE001 - re-raised on the request thread
            progress["error"] = error

    loader = threading.Thread(target=load, name="llama-load", daemon=True)
    loader.start()
    shown = None
    while loader.is_alive():
        loader.join(0.5)
        message = progress.get("status")
        if message and message != shown:
            shown = message
            yield status(message)
    if "error" in progress:
        raise progress["error"]
    gateway.llama.acquire()
    try:
        payload = build_payload(entry["name"], request, local=True)
        for event in stream_chat(chat_url(progress["url"]), payload, {}, request, "chat_template"):
            if event["type"] == "done":
                event["context"] = n_ctx
            yield event
    finally:
        gateway.llama.release()


def load_binding(entry: dict, n_ctx: int, gpu_layers):
    try:
        from llama_cpp import Llama
    except ImportError as error:
        raise ModelError(f"GGUF backend missing; {GGUF_HINT}") from error
    arguments = {"model_path": entry["absolute_path"], "n_ctx": int(n_ctx), "verbose": False}
    if gpu_layers not in (None, "", "auto"):
        arguments["n_gpu_layers"] = int(gpu_layers)
    return Llama(**arguments)


def stream_binding(gateway, entry, request):
    """Run the model in-process with llama-cpp-python."""
    path = Path(entry["absolute_path"])
    n_ctx = _context(gateway, entry, request)
    gpu_layers = gateway.settings.get("llama_gpu_layers")
    key = ("gguf", str(path), path.stat().st_mtime_ns, n_ctx, str(gpu_layers))
    if not gateway.is_cached(key):
        yield status(f"Loading {entry['name']}…")
    parameters = {"temperature": request.temperature, "top_p": request.top_p,
                  "max_tokens": request.max_tokens}
    if request.top_k:
        parameters["top_k"] = request.top_k
    usage: dict = {}
    with gateway.local_lock:
        model = gateway.cached(key, lambda: load_binding(entry, n_ctx, gpu_layers))
        try:
            chunks = model.create_chat_completion(messages=request.messages, stream=True, **parameters)
            first = next(iter(chunks), None)
            prompt_format, stop_filter = "chat_template", None
        except Exception:  # noqa: BLE001 - base models have no chat template; continue the transcript
            chunks = model.create_completion(
                prompt=transcript_prompt(request.messages), stream=True, **parameters,
            )
            first = next(iter(chunks), None)
            prompt_format, stop_filter = "transcript", StopFilter()

        def pieces():
            if first is not None:
                yield first
            yield from chunks

        count = 0
        for chunk in pieces():
            if request.cancelled.is_set():
                break
            choice = (chunk.get("choices") or [{}])[0]
            text = (choice.get("delta") or {}).get("content") if "delta" in choice else choice.get("text")
            if isinstance(chunk.get("usage"), dict):
                usage = chunk["usage"]
            if not text:
                continue
            count += 1
            if stop_filter is not None:
                text = stop_filter.feed(text)
                if stop_filter.stopped:
                    if text:
                        yield delta(text)
                    break
            if text:
                yield delta(text)
        if stop_filter is not None:
            tail = stop_filter.flush()
            if tail:
                yield delta(tail)
    yield done(
        usage={"prompt_tokens": int(usage.get("prompt_tokens") or 0),
               "completion_tokens": int(usage.get("completion_tokens") or count)},
        prompt_format=prompt_format, context=n_ctx,
    )


def stream(gateway, entry, request):
    if not entry.get("absolute_path"):
        raise ModelError("This model has no local file to load")
    if entry.get("runtime") == "llama-cpp-python":
        yield from stream_binding(gateway, entry, request)
    else:
        yield from stream_server(gateway, entry, request)
