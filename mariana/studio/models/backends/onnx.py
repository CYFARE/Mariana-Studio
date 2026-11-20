"""ONNX causal language models through optimum + onnxruntime."""

from __future__ import annotations

from pathlib import Path

from ..catalog import ONNX_HINT, ModelError
from . import delta, done, status
from .huggingface import build_prompt


def load_model(entry: dict):
    try:
        from optimum.onnxruntime import ORTModelForCausalLM
        from transformers import AutoTokenizer
    except ImportError as error:
        raise ModelError(f"ONNX backend missing; {ONNX_HINT}") from error
    folder = Path(entry["absolute_path"])
    if folder.suffix == ".onnx":
        folder = folder.parent
    tokenizer = AutoTokenizer.from_pretrained(str(folder), local_files_only=True)
    return tokenizer, ORTModelForCausalLM.from_pretrained(str(folder), local_files_only=True)


def stream(gateway, entry, request):
    folder = Path(entry["absolute_path"])
    key = ("onnx", str(folder), folder.stat().st_mtime_ns)
    if not gateway.is_cached(key):
        yield status(f"Loading {entry['name']}…")
    with gateway.local_lock:
        tokenizer, model = gateway.cached(key, lambda: load_model(entry))
        prompt, prompt_format = build_prompt(
            tokenizer, request.messages, request.options.get("prompt_format", "auto"),
        )
        encoded = tokenizer(prompt, return_tensors="pt")
        arguments = dict(
            **encoded, max_new_tokens=request.max_tokens, do_sample=request.temperature > 0,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id,
        )
        if request.temperature > 0:
            arguments.update(temperature=request.temperature, top_p=request.top_p)
            if request.top_k:
                arguments["top_k"] = request.top_k
        output = model.generate(**arguments)
        generated = output[0][encoded["input_ids"].shape[1]:]
        reply = tokenizer.decode(generated, skip_special_tokens=True)
    if prompt_format == "transcript":
        reply = reply.split("\nUser:", 1)[0]
    reply = reply.strip()
    if reply:
        yield delta(reply)
    yield done(
        usage={"prompt_tokens": int(encoded["input_ids"].shape[1]),
               "completion_tokens": int(generated.shape[0])},
        prompt_format=prompt_format,
    )
