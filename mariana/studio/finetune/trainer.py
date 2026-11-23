"""Fine-tune any Hugging Face causal language model: LoRA, full, or last-N-layers.

Runs as a subprocess of Studio::

    python -m mariana.studio.finetune.trainer --config job.json

and reports progress on stdout in the line format every Studio progress view
parses (``step <n>/<total> | total <loss> | lr <lr> | tok/s <rate>``).  Expected
failures print one ``ERROR: <what to do>`` line and exit with status 1.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from . import memory
from .dataset import (
    IGNORE_INDEX, ExampleBuilder, collate, encode_chat, iter_batches, iter_chat_examples,
    iter_text_blocks, split_examples,
)
from .lora import count_parameters, inject_lora, load_adapter, merge_adapter, save_adapter

METADATA_NAME = "mariana-finetune.json"
METHODS = ("lora", "full", "last_layers")
FINAL_NORM_NAMES = frozenset({
    "ln_f", "norm", "final_layer_norm", "final_norm", "norm_f", "layer_norm", "final_layernorm",
})
DEFAULTS = {
    "method": "lora",
    "lora": {"r": 16, "alpha": 32, "dropout": 0.05, "targets": "auto"},
    "trainable_layers": 2,
    "steps": 200, "batch": 4, "grad_accum": 1, "lr": 2e-4, "warmup": 10,
    "weight_decay": 0.0, "block_size": 256, "eval_every": 50, "eval_batches": 8,
    "log_every": 0, "seed": 0, "device": "auto", "precision": "auto",
    "quantization": "auto", "auto_memory": True, "parameters": 0,
    "max_tokens": 20_000_000, "max_examples": 500_000, "loss_chunk": 256,
    "gradient_checkpointing": False, "merge": True, "train_on": "assistant",
    "allow_download": False, "pause_file": None,
}
QUANTIZATION_MODES = ("auto", "none", "4bit", "8bit")


class TrainerError(Exception):
    """An expected failure with a message the user can act on."""


def say(message: str) -> None:
    print(message, flush=True)


def load_config(path: str) -> dict:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise TrainerError(f"Cannot read the job file {path}: {error}") from error
    if not isinstance(raw, dict):
        raise TrainerError("The job file must contain a JSON object")
    config = {**DEFAULTS, **raw}
    config["lora"] = {**DEFAULTS["lora"], **(raw.get("lora") or {})}
    data = raw.get("data") or {}
    config["data"] = {"corpus": data.get("corpus"), "chat_files": list(data.get("chat_files") or [])}
    if not config.get("base") or not config.get("output"):
        raise TrainerError("The job file needs both a base model and an output folder")
    if config["method"] not in METHODS:
        raise TrainerError(f"Unknown method {config['method']!r}; use lora, full, or last_layers")
    if config["train_on"] not in {"assistant", "all"}:
        raise TrainerError("train_on must be assistant or all")
    if config["precision"] not in {"auto", "fp32", "bf16", "fp16"}:
        raise TrainerError("precision must be auto, fp32, bf16 or fp16")
    if config["quantization"] not in QUANTIZATION_MODES:
        raise TrainerError("quantization must be auto, none, 4bit or 8bit")
    if config["method"] != "lora" and config["quantization"] not in {"auto", "none"}:
        raise TrainerError("4-bit and 8-bit quantization need the LoRA method")
    try:
        for key in ("steps", "batch", "grad_accum", "block_size", "eval_every", "eval_batches",
                    "warmup", "log_every", "seed", "trainable_layers", "parameters",
                    "max_tokens", "max_examples", "loss_chunk"):
            config[key] = int(config[key])
        for key in ("lr", "weight_decay"):
            config[key] = float(config[key])
        config["lora"]["r"] = int(config["lora"]["r"])
        config["lora"]["alpha"] = float(config["lora"]["alpha"])
        config["lora"]["dropout"] = float(config["lora"]["dropout"])
    except (TypeError, ValueError) as error:
        raise TrainerError(f"A numeric setting in the job file is not a number: {error}") from error
    if config["steps"] < 1 or config["batch"] < 1 or config["grad_accum"] < 1:
        raise TrainerError("steps, batch and grad_accum must each be at least 1")
    if config["block_size"] < 8:
        raise TrainerError("block_size must be at least 8 tokens")
    if config["max_tokens"] < 1_000 or config["max_examples"] < 1:
        raise TrainerError("max_tokens must be at least 1000 and max_examples at least 1")
    if config["loss_chunk"] < 16:
        raise TrainerError("loss_chunk must be at least 16 tokens")
    config["loss_chunk"] = max(16, min(config["loss_chunk"], config["block_size"]))
    if config["lr"] <= 0:
        raise TrainerError("lr must be greater than zero")
    return config


def check_paths(config: dict) -> tuple[str, Path]:
    """Refuse to write into the base model; return ``(base, output)``."""
    base = str(config["base"])
    output = Path(config["output"]).expanduser().resolve()
    base_path = Path(base).expanduser()
    if base_path.exists():
        base_path = base_path.resolve()
        if output == base_path or base_path in output.parents:
            raise TrainerError(
                "The output folder is the base model or inside it; choose a separate folder so "
                "the base stays untouched"
            )
        base = str(base_path)
    return base, output


def import_transformers():
    try:
        import transformers
    except ImportError as error:
        raise TrainerError(
            "Transformers is not installed. Run: pip install -r requirements-finetune.txt"
        ) from error
    transformers.logging.set_verbosity_error()
    return transformers


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def base_parameter_count(base: str) -> int:
    """Parameter count from safetensors headers, or 0 when it cannot be known."""
    path = Path(base)
    if not path.is_dir():
        return 0
    try:
        from ...models.catalog import safetensors_summary, transformers_weights
    except ImportError:
        return 0
    total = 0
    for weight in transformers_weights(path):
        if weight.suffix == ".safetensors":
            total += int(safetensors_summary(weight).get("parameters") or 0)
    return total


def model_context_limit(base: str) -> int:
    try:
        config = json.loads((Path(base) / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return 0
    return int(config.get("max_position_embeddings") or config.get("n_positions") or 0)


def load_pretrained(loader, base: str, local_only: bool, dtype,
                    quantization: str = "none", compute_dtype=None, device=None):
    """``from_pretrained`` across Transformers versions and the quantized path."""
    options: dict = {"local_files_only": local_only}
    if quantization in {"4bit", "8bit"}:
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as error:
            raise TrainerError(
                "Quantized fine-tuning needs a Transformers build with bitsandbytes support. "
                "Run: pip install -r requirements-finetune.txt") from error
        if quantization == "4bit":
            options["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype or torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            options["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        if device is not None and _has_module("accelerate"):
            options["device_map"] = {"": str(device)}
    else:
        options["dtype"] = dtype
    try:
        try:
            return loader.from_pretrained(base, **options)
        except TypeError as error:
            if "dtype" in options:
                options["torch_dtype"] = options.pop("dtype")
                return loader.from_pretrained(base, **options)
            raise TrainerError(
                f"This Transformers build cannot load the model with these settings: {error}") from error
    except OSError as error:
        hint = "" if not local_only else " Download it in Studio (Models → Discover) first."
        raise TrainerError(f"Cannot load the base model {base}: {error}.{hint}") from error
    except ValueError as error:
        raise TrainerError(f"Cannot load the base model {base}: {error}") from error


def resolve_precision(requested: str, device) -> str:
    """bf16 where supported, fp16 on older CUDA cards, fp32 on CPU/MPS."""
    if requested == "fp32":
        return "fp32"
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return "bf16"
        if requested == "bf16":
            say("bf16 is not supported on this device; using fp16")
        return "fp16"
    if requested in {"bf16", "fp16"}:
        say(f"{requested} needs a CUDA GPU; using fp32 on {device.type}")
    return "fp32"


def precision_dtype(precision: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16}.get(precision, torch.float32)


def find_layer_stack(model: nn.Module) -> tuple[str, nn.ModuleList]:
    """The ``ModuleList`` holding the transformer blocks: the one with the most parameters."""
    best_name, best, best_size = "", None, -1
    for name, module in model.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) > 0:
            size = sum(parameter.numel() for parameter in module.parameters())
            if size > best_size:
                best_name, best, best_size = name, module, size
    if best is None:
        raise TrainerError("Could not find the transformer layers of this model; use LoRA instead")
    return best_name, best


def unfreeze_last_layers(model: nn.Module, count: int) -> int:
    """Train only the last ``count`` blocks and the final norm; return how many blocks."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    name, stack = find_layer_stack(model)
    count = max(1, min(count, len(stack)))
    for block in list(stack)[-count:]:
        for parameter in block.parameters():
            parameter.requires_grad_(True)
    parent = model.get_submodule(name.rpartition(".")[0]) if "." in name else model
    for child_name, child in parent.named_children():
        if child_name in FINAL_NORM_NAMES:
            for parameter in child.parameters():
                parameter.requires_grad_(True)
    return count


def build_examples(tokenizer, config: dict):
    """Tokenize data into a compact store, streaming instead of loading it twice."""
    data = config["data"]
    block_size = config["block_size"]
    builder = ExampleBuilder(config["max_tokens"], config["max_examples"])
    chat_count = text_count = 0
    if data["chat_files"]:
        for messages in iter_chat_examples(data["chat_files"]):
            if builder.full:
                break
            example = encode_chat(tokenizer, messages, config["train_on"], block_size)
            if example is not None and builder.add(example):
                chat_count += 1
    if data["corpus"] and not builder.full:
        if not Path(data["corpus"]).is_file():
            raise TrainerError(f"The prepared corpus {data['corpus']} does not exist")
        remaining = max(0, config["max_tokens"] - builder.tokens)
        for block in iter_text_blocks(tokenizer, data["corpus"], block_size, remaining):
            if not builder.add(block):
                break
            text_count += 1
    examples = builder.build()
    if not len(examples):
        raise TrainerError(
            "No training examples were found. Provide a folder with text files, or JSONL chat "
            "data with messages / prompt+completion / instruction+output rows"
        )
    tokens = builder.tokens
    say(f"Data: {len(examples):,} examples | {tokens:,} tokens | "
        f"{chat_count:,} chat | {text_count:,} text blocks")
    if builder.skipped or builder.full:
        say(f"Data: stopped at the memory cap ({config['max_tokens']:,} tokens / "
            f"{config['max_examples']:,} examples); raise it to use more of a large dataset")
    return examples, {"examples": len(examples), "tokens": tokens,
                      "chat_examples": chat_count, "text_blocks": text_count,
                      "capped": bool(builder.skipped or builder.full)}


def compute_loss(model, batch: dict, autocast: bool, device,
                 autocast_dtype: torch.dtype = torch.bfloat16,
                 chunk_tokens: int = 256) -> tuple[torch.Tensor, int]:
    """Next-token cross-entropy in float32, computed one slice at a time.

    A 7B model's full logits tensor for a 512-token batch is ~300 MB in float32.
    Summing the slices keeps that temporary small without changing the loss.
    """
    with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast):
        logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
    logits = logits[:, :-1]
    targets = batch["labels"][:, 1:]
    supervised = int((targets != IGNORE_INDEX).sum())
    if supervised == 0:
        return logits.sum() * 0.0, 0
    positions = logits.shape[1]
    chunk = max(16, int(chunk_tokens))
    total = torch.zeros((), dtype=torch.float32, device=logits.device)
    for start in range(0, positions, chunk):
        part = logits[:, start:start + chunk].float().reshape(-1, logits.shape[-1])
        target = targets[:, start:start + chunk].reshape(-1)
        total = total + F.cross_entropy(part, target, ignore_index=IGNORE_INDEX, reduction="sum")
    return total / supervised, supervised


@torch.no_grad()
def evaluate(model, examples, config: dict, pad_id: int, autocast: bool, device,
             autocast_dtype: torch.dtype = torch.bfloat16) -> float | None:
    if not examples:
        return None
    model.eval()
    total = weight = 0.0
    size = max(1, config["batch"])
    index = done = 0
    try:
        while index * size < len(examples) and done < config["eval_batches"]:
            batch = collate(examples[index * size:(index + 1) * size], pad_id)
            try:
                batch = {key: value.to(device) for key, value in batch.items()}
                loss, supervised = compute_loss(
                    model, batch, autocast, device, autocast_dtype, config["loss_chunk"])
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if size > 1:
                    size = max(1, size // 2)
                    say(f"Memory: validation batch reduced to {size}")
                    continue
                say("Warning: validation skipped because the GPU is out of memory")
                return None
            total += float(loss) * supervised
            weight += supervised
            index += 1
            done += 1
    finally:
        model.train()
    return total / weight if weight else None


def learning_rate(step: int, config: dict) -> float:
    """Linear warmup, then cosine decay to 10% of the peak."""
    if step < config["warmup"]:
        return config["lr"] * (step + 1) / max(config["warmup"], 1)
    progress = (step - config["warmup"]) / max(config["steps"] - config["warmup"], 1)
    return config["lr"] * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0))))


def prepare_output(output: Path) -> Path:
    """Return an empty staging folder next to ``output``; refuse to clobber foreign folders."""
    if output.exists():
        if not output.is_dir() or (any(output.iterdir()) and not (output / METADATA_NAME).exists()):
            raise TrainerError(
                f"{output} already exists and was not written by Mariana; choose another name"
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.{os.getpid()}.tmp"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    return staging


def publish(staging: Path, output: Path) -> None:
    """Swap the finished staging folder into place."""
    if output.exists():
        shutil.rmtree(output)
    os.replace(staging, output)


def _oom_error() -> TrainerError:
    return TrainerError(
        "The GPU ran out of memory even at batch 1 with gradient checkpointing. "
        "Use 4-bit quantization (QLoRA), a smaller base model, or a shorter context, then start again."
    )


def run(config: dict) -> int:
    base, output = check_paths(config)
    transformers = import_transformers()
    from ...device import describe, empty_cache, pick_device

    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])
    device = pick_device(config["device"])
    say(describe(device))
    precision = resolve_precision(config["precision"], device)
    method = config["method"]
    local_only = not config["allow_download"]

    parameters = int(config.get("parameters") or 0) or base_parameter_count(base)
    vram = torch.cuda.get_device_properties(device).total_memory if device.type == "cuda" else 0
    quantize_available = device.type == "cuda" and _has_module("bitsandbytes")
    quantization = config["quantization"]
    memory_plan = None
    if config["auto_memory"] and device.type == "cuda":
        preferred = precision if precision != "fp32" else (
            "bf16" if torch.cuda.is_bf16_supported() else "fp16")
        tuned = memory.auto_tune(
            parameters, vram=vram, method=method, precision=preferred,
            quantization=quantization, batch=config["batch"],
            grad_accum=config["grad_accum"], block_size=config["block_size"],
            quantize_available=quantize_available,
        )
        config["batch"] = tuned["batch"]
        config["grad_accum"] = tuned["grad_accum"]
        config["gradient_checkpointing"] = tuned["gradient_checkpointing"]
        quantization = tuned["quantization"]
        memory_plan = tuned
        for note in tuned["notes"]:
            say(f"Memory: {note}")
        say(f"Memory: {memory.describe_bytes(tuned['estimate']['weights'])} weights | "
            f"~{memory.describe_bytes(tuned['estimate']['total'])} estimated peak | "
            f"batch {config['batch']} × {config['grad_accum']} | "
            f"{vram / memory.GIB:.1f} GB VRAM")
    elif quantization == "auto":
        quantization = "none"
    if method != "lora":
        if quantization != "none":
            say("Quantization only applies to LoRA; it was turned off")
        quantization = "none"
    if quantization != "none":
        if device.type != "cuda":
            raise TrainerError(
                "4-bit and 8-bit fine-tuning need a CUDA GPU; choose Precision FP32 or BF16 instead")
        if not _has_module("bitsandbytes"):
            raise TrainerError(
                "Quantized fine-tuning needs bitsandbytes. Run: "
                "pip install -r requirements-finetune.txt")
        if not _has_module("accelerate"):
            say("Note: installing accelerate improves quantized model placement: "
                "pip install accelerate")
        if precision == "fp32":
            say("Quantized LoRA computes in bf16/fp16 while the adapter weights stay float32")
        compute_dtype = precision_dtype(
            precision if precision != "fp32" else ("bf16" if torch.cuda.is_bf16_supported() else "fp16"))
    else:
        compute_dtype = precision_dtype(precision)
    autocast = precision != "fp32" and method != "lora"
    load_dtype = (None if quantization != "none"
                  else (precision_dtype(precision) if method == "lora" else torch.float32))

    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(base, local_files_only=local_only)
    except (OSError, ValueError) as error:
        hint = "" if not local_only else " Download it in Studio (Models → Discover) first."
        raise TrainerError(f"Cannot load the tokenizer of {base}: {error}.{hint}") from error
    tokenizer.model_max_length = int(1e30)  # blocks are cut here, not by the tokenizer
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    limit = model_context_limit(base)
    if limit and config["block_size"] > limit:
        say(f"block_size reduced to the model's limit of {limit} tokens")
        config["block_size"] = limit

    # Data is checked and tokenized before the weights are loaded: a huge base
    # model should not cost minutes of download/load just to fail on empty data.
    examples, data_stats = build_examples(tokenizer, config)
    training, validation = split_examples(examples, seed=config["seed"])
    per_step = config["batch"] * config["grad_accum"]
    say(f"One epoch ≈ {max(1, math.ceil(len(training) / per_step)):,} steps "
        f"({len(training):,} training / {len(validation):,} validation examples)")

    try:
        model = load_pretrained(
            transformers.AutoModelForCausalLM, base, local_only, load_dtype,
            quantization=quantization, compute_dtype=compute_dtype, device=device,
        )
    except torch.cuda.OutOfMemoryError as error:
        empty_cache(device)
        raise TrainerError(
            "The base model did not fit in VRAM. Enable 4-bit quantization, use a smaller "
            "base model, or close other GPU applications") from error
    except RuntimeError as error:
        if "out of memory" in str(error).lower():
            empty_cache(device)
            raise TrainerError(
                "The base model did not fit in VRAM. Enable 4-bit quantization, use a smaller "
                "base model, or close other GPU applications") from error
        raise TrainerError(f"Cannot load the base model {base}: {error}") from error
    if quantization == "none":
        try:
            model.to(device)
        except torch.cuda.OutOfMemoryError as error:
            empty_cache(device)
            raise TrainerError(
                "The base model did not fit in VRAM. Enable 4-bit quantization, use a smaller "
                "base model, or close other GPU applications") from error
    elif not _has_module("accelerate"):
        try:
            model.to(device)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
            empty_cache(device)
            raise TrainerError(
                "Could not place the quantized model on the GPU. Install accelerate "
                "(pip install accelerate) or choose a smaller base model") from error
    limit = getattr(model.config, "max_position_embeddings", None) or getattr(
        model.config, "n_positions", None)
    if limit and config["block_size"] > int(limit):
        say(f"block_size reduced to the model's limit of {int(limit)} tokens")
        config["block_size"] = int(limit)
    model.config.use_cache = False

    lora_names: list[str] = []
    if method == "lora":
        try:
            lora_names = inject_lora(
                model, config["lora"]["r"], config["lora"]["alpha"], config["lora"]["dropout"],
                config["lora"]["targets"],
            )
        except ValueError as error:
            raise TrainerError(str(error)) from error
    elif method == "last_layers":
        trained = unfreeze_last_layers(model, config["trainable_layers"])
        say(f"Training the last {trained} transformer block(s)")
    else:
        for parameter in model.parameters():
            parameter.requires_grad_(True)
    if config["gradient_checkpointing"]:
        try:
            model.gradient_checkpointing_enable()
        except (ValueError, NotImplementedError):
            say("This architecture does not support gradient checkpointing; continuing without it")
        if method != "full" and hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()  # frozen inputs would cut the graph otherwise
    total, trainable = count_parameters(model)
    say(f"Parameters: {total / 1e6:.2f}M total | {trainable / 1e6:.2f}M trainable "
        f"({100.0 * trainable / max(total, 1):.2f}%)")
    if trainable == 0:
        raise TrainerError("Nothing is trainable with these settings; choose another method")

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=config["lr"], weight_decay=config["weight_decay"], betas=(0.9, 0.999),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(precision == "fp16" and device.type == "cuda"))
    sampler = random.Random(config["seed"])
    steps = config["steps"]
    log_every = config["log_every"] or (1 if steps <= 50 else max(1, steps // 100))
    pause_file = config.get("pause_file")
    model.train()
    window_loss = window_count = window_tokens = 0
    window_started = time.time()
    train_loss = val_loss = None
    completed = 0
    paused = False
    oom_events = 0
    batches = iter_batches(training, config["batch"], sampler)
    while completed < steps:
        rate = learning_rate(completed, config)
        for group in optimizer.param_groups:
            group["lr"] = rate
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        micro = 0
        accum = max(1, config["grad_accum"])
        restart = False
        while micro < accum:
            indices = next(batches)
            try:
                batch = collate([training[index] for index in indices], pad_id)
                batch = {key: value.to(device) for key, value in batch.items()}
                loss, _ = compute_loss(
                    model, batch, autocast, device, compute_dtype, config["loss_chunk"])
                if not torch.isfinite(loss):
                    raise TrainerError(
                        "The loss became infinite or NaN. Lower the learning rate, or use fp32 precision")
                scaled = loss / accum
                if scaler.is_enabled():
                    scaler.scale(scaled).backward()
                else:
                    scaled.backward()
                window_tokens += int(batch["attention_mask"].sum())
                step_loss += float(loss) / accum
            except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
                if (not isinstance(error, torch.cuda.OutOfMemoryError)
                        and "out of memory" not in str(error).lower()):
                    raise
                empty_cache(device)
                optimizer.zero_grad(set_to_none=True)
                oom_events += 1
                if oom_events > 8:
                    raise _oom_error() from error
                if config["batch"] > 1:
                    previous = config["batch"]
                    config["batch"] = max(1, config["batch"] // 2)
                    config["grad_accum"] = max(1, min(4096, config["grad_accum"] * 2))
                    batches = iter_batches(training, config["batch"], sampler)
                    say(f"Memory: out of memory; micro-batch {previous} -> {config['batch']}, "
                        f"accumulation -> {config['grad_accum']}; restarting the step")
                    restart = True
                    break
                if not config["gradient_checkpointing"]:
                    config["gradient_checkpointing"] = True
                    try:
                        model.gradient_checkpointing_enable()
                    except (ValueError, NotImplementedError):
                        pass
                    if method != "full" and hasattr(model, "enable_input_require_grads"):
                        model.enable_input_require_grads()
                    say("Memory: out of memory; enabled gradient checkpointing; restarting the step")
                    restart = True
                    break
                raise _oom_error() from error
            micro += 1
        if restart:
            continue
        if scaler.is_enabled():
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(parameters, 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
        completed += 1
        train_loss = step_loss
        window_loss += step_loss
        window_count += 1
        if completed % log_every == 0 or completed == steps:
            elapsed = max(time.time() - window_started, 1e-6)
            say(f"step {completed}/{steps} | total {window_loss / window_count:.4f} | "
                f"lr {rate:.2e} | tok/s {window_tokens / elapsed:.0f}")
            window_loss = window_count = window_tokens = 0
            window_started = time.time()
        if pause_file and os.path.exists(pause_file):
            paused = True
        if completed % config["eval_every"] == 0 or completed == steps or paused:
            measured = evaluate(model, validation, config, pad_id, autocast, device, compute_dtype)
            if measured is not None:
                val_loss = measured
                say(f"VAL step {completed}: total {measured:.4f} | ppl {math.exp(min(measured, 50)):.2f}")
        if paused:
            break

    staging = prepare_output(output)
    try:
        metadata = {
            "base": base, "method": method, "steps_completed": completed, "steps_planned": steps,
            "train_loss": train_loss, "val_loss": val_loss, "paused": paused,
            "created": datetime.now().isoformat(timespec="seconds"),
            "data": data_stats, "parameters": {"total": total, "trainable": trainable},
            "precision": precision, "quantization": quantization,
            "batch": config["batch"], "grad_accum": config["grad_accum"],
            "gradient_checkpointing": bool(config["gradient_checkpointing"]),
        }
        if memory_plan is not None:
            metadata["memory"] = {
                "estimated_peak": memory_plan["estimate"]["total"],
                "fits": memory_plan["fits"], "notes": memory_plan["notes"],
            }
        merge_error = None
        if method == "lora":
            metadata["lora"] = {**config["lora"], "target_modules": len(lora_names),
                                "merged": bool(config["merge"])}
            save_adapter(
                model, staging / "adapter", base_model=base, r=config["lora"]["r"],
                alpha=config["lora"]["alpha"], dropout=config["lora"]["dropout"],
                target_modules=lora_names,
            )
            if config["merge"]:
                del model, optimizer, parameters
                empty_cache(device)
                merge_dir = staging / ".merged"
                merge_dir.mkdir(parents=True, exist_ok=True)
                try:
                    say("Merging the adapter into a copy of the base model on CPU")
                    merge_dtype = torch.float32 if precision == "fp32" else compute_dtype
                    merged = load_pretrained(
                        transformers.AutoModelForCausalLM, base, local_only, merge_dtype)
                    load_adapter(merged, staging / "adapter")
                    merge_adapter(merged)
                    merged.config.use_cache = True
                    merged.save_pretrained(merge_dir, safe_serialization=True)
                    tokenizer.save_pretrained(merge_dir)
                    for item in merge_dir.iterdir():
                        os.replace(item, staging / item.name)
                    merge_dir.rmdir()
                    metadata["lora"]["merged"] = True
                except Exception as error:  # noqa: BLE001 - a failed merge must keep the adapter
                    merge_error = str(error)[:500] or type(error).__name__
                    metadata["lora"]["merged"] = False
                    say("Warning: could not merge the adapter into a full model "
                        "(not enough CPU memory or an unsupported architecture). "
                        "The adapter itself was saved and is usable.")
                finally:
                    shutil.rmtree(merge_dir, ignore_errors=True)
        else:
            model.config.use_cache = True
            model.save_pretrained(staging, safe_serialization=True)
            tokenizer.save_pretrained(staging)
        if merge_error:
            metadata["merge_error"] = merge_error
        (staging / METADATA_NAME).write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        publish(staging, output)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    if method == "lora":
        say(f"Saved adapter to {output / 'adapter'}")
    if paused:
        say(f"Paused at step {completed}")
    say(f"Saved model to {output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m mariana.studio.finetune.trainer",
        description="Fine-tune a Hugging Face causal language model from a Studio job file.",
    )
    parser.add_argument("--config", required=True, help="path to the JSON job file")
    arguments = parser.parse_args(argv)
    try:
        return run(load_config(arguments.config))
    except TrainerError as error:
        say(f"ERROR: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
