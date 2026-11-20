"""Hugging Face Transformers causal language models (in-process, PyTorch)."""

from __future__ import annotations

import threading
from pathlib import Path

import torch

from ....device import pick_device
from ..catalog import TRANSFORMERS_HINT, ModelError
from . import StopFilter, delta, done, status, transcript_prompt


def load_model(entry: dict, device):
    """Return ``(tokenizer, model)`` for a model directory or a LoRA adapter directory."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise ModelError(f"Transformers backend missing; {TRANSFORMERS_HINT}") from error
    folder = Path(entry["absolute_path"])
    if folder.suffix in {".safetensors", ".bin", ".pt"}:
        folder = folder.parent
    detail = entry.get("detail") or {}
    source = Path(detail["base"]) if detail.get("adapter") else folder
    tokenizer = AutoTokenizer.from_pretrained(str(source), local_files_only=True)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(source), local_files_only=True, dtype=dtype, low_cpu_mem_usage=True,
        )
    except TypeError:  # older Transformers spell the argument torch_dtype
        model = AutoModelForCausalLM.from_pretrained(
            str(source), local_files_only=True, torch_dtype=dtype, low_cpu_mem_usage=True,
        )
    if detail.get("adapter"):
        from ...finetune.lora import load_adapter

        load_adapter(model, folder / "adapter")
    return tokenizer, model.to(device).eval()


def build_prompt(tokenizer, messages: list[dict], prompt_format: str) -> tuple[str, str]:
    """Return ``(prompt, format name)`` using the chat template when there is one."""
    has_template = bool(getattr(tokenizer, "chat_template", None)) and hasattr(
        tokenizer, "apply_chat_template"
    )
    if prompt_format in {"auto", "chatml"} and has_template:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return prompt, "chat_template"
    return transcript_prompt(messages), "transcript"


def stream(gateway, entry, request):
    device = pick_device(request.device)
    folder = Path(entry["absolute_path"])
    key = ("transformers", str(folder), folder.stat().st_mtime_ns, str(device))
    if not gateway.is_cached(key):
        yield status(f"Loading {entry['name']}…")
    with gateway.local_lock:
        tokenizer, model = gateway.cached(key, lambda: load_model(entry, device))
        from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

        prompt, prompt_format = build_prompt(
            tokenizer, request.messages, request.options.get("prompt_format", "auto"),
        )
        encoded = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_tokens = int(encoded["input_ids"].shape[1])
        stop = threading.Event()
        stop_filter = StopFilter() if prompt_format == "transcript" else None

        class Cancelled(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return stop.is_set() or request.cancelled.is_set()

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        arguments = dict(
            **encoded, max_new_tokens=request.max_tokens, do_sample=request.temperature > 0,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id,
            use_cache=True, streamer=streamer, stopping_criteria=StoppingCriteriaList([Cancelled()]),
        )
        if request.temperature > 0:
            arguments.update(temperature=request.temperature, top_p=request.top_p)
            if request.top_k:
                arguments["top_k"] = request.top_k
        outcome: dict = {}

        def run():
            try:
                with torch.no_grad():
                    outcome["output"] = model.generate(**arguments)
            except BaseException as error:  # noqa: BLE001 - re-raised on the request thread
                outcome["error"] = error
            finally:
                streamer.end()

        worker = threading.Thread(target=run, name="transformers-generate", daemon=True)
        worker.start()
        try:
            for piece in streamer:
                if stop_filter is not None:
                    piece = stop_filter.feed(piece)
                    if stop_filter.stopped:
                        stop.set()
                if piece:
                    yield delta(piece)
            if stop_filter is not None:
                tail = stop_filter.flush()
                if tail:
                    yield delta(tail)
        finally:
            stop.set()
            worker.join(timeout=30)
        if "error" in outcome:
            raise ModelError(f"Generation failed: {outcome['error']}") from outcome["error"]
        output = outcome.get("output")
        completion = int(output[0].shape[0]) - prompt_tokens if output is not None else 0
        truncated = completion >= request.max_tokens
    yield done(
        usage={"prompt_tokens": prompt_tokens, "completion_tokens": max(0, completion)},
        prompt_format=prompt_format, finish_reason="length" if truncated else "stop",
    )
