"""Fine-tune any Hugging Face causal language model: LoRA, full, or last-N-layers.

Runs as a subprocess of Studio::

    python -m mariana.studio.finetune.trainer --config job.json

and reports progress on stdout in the line format every Studio progress view
parses (``step <n>/<total> | total <loss> | lr <lr> | tok/s <rate>``).  Expected
failures print one ``ERROR: <what to do>`` line and exit with status 1.
"""

from __future__ import annotations

import argparse
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

from .dataset import IGNORE_INDEX, build_text_blocks, collate, encode_chat, load_chat_examples, split_examples
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
    "gradient_checkpointing": False, "merge": True, "train_on": "assistant",
    "allow_download": False, "pause_file": None,
}


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
    try:
        for key in ("steps", "batch", "grad_accum", "block_size", "eval_every", "eval_batches",
                    "warmup", "log_every", "seed", "trainable_layers"):
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


def load_pretrained(loader, base: str, local_only: bool, dtype):
    """``from_pretrained`` across Transformers versions (``dtype`` was ``torch_dtype``)."""
    try:
        try:
            return loader.from_pretrained(base, local_files_only=local_only, dtype=dtype)
        except TypeError:
            return loader.from_pretrained(base, local_files_only=local_only, torch_dtype=dtype)
    except OSError as error:
        hint = "" if not local_only else " Download it in Studio (Models → Discover) first."
        raise TrainerError(f"Cannot load the base model {base}: {error}.{hint}") from error


def resolve_precision(requested: str, device) -> str:
    if requested == "fp32":
        return "fp32"
    supported = device.type == "cuda" and torch.cuda.is_bf16_supported()
    if requested == "bf16" and not supported:
        say("bf16 is not supported on this device; using fp32")
    return "bf16" if supported and requested in {"auto", "bf16"} else "fp32"


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
    data = config["data"]
    block_size = config["block_size"]
    chat = text = []
    if data["chat_files"]:
        conversations = load_chat_examples(data["chat_files"])
        chat = [
            example for example in (
                encode_chat(tokenizer, messages, config["train_on"], block_size)
                for messages in conversations
            ) if example is not None
        ]
    if data["corpus"]:
        if not Path(data["corpus"]).is_file():
            raise TrainerError(f"The prepared corpus {data['corpus']} does not exist")
        text = build_text_blocks(tokenizer, data["corpus"], block_size)
    examples = list(chat) + list(text)
    if not examples:
        raise TrainerError(
            "No training examples were found. Provide a folder with text files, or JSONL chat "
            "data with messages / prompt+completion / instruction+output rows"
        )
    tokens = sum(len(ids) for ids, _ in examples)
    say(f"Data: {len(examples):,} examples | {tokens:,} tokens | "
        f"{len(chat):,} chat | {len(text):,} text blocks")
    return examples, {"examples": len(examples), "tokens": tokens,
                      "chat_examples": len(chat), "text_blocks": len(text)}


def compute_loss(model, batch: dict, autocast: bool, device) -> tuple[torch.Tensor, int]:
    """Next-token cross-entropy in float32, and the number of supervised tokens."""
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast):
        logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
    logits = logits[:, :-1].float()
    targets = batch["labels"][:, 1:]
    supervised = int((targets != IGNORE_INDEX).sum())
    if supervised == 0:
        return logits.sum() * 0.0, 0
    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=IGNORE_INDEX,
    )
    return loss, supervised


@torch.no_grad()
def evaluate(model, examples, config: dict, pad_id: int, autocast: bool, device) -> float | None:
    if not examples:
        return None
    model.eval()
    total = weight = 0.0
    size = config["batch"]
    for index in range(min(config["eval_batches"], math.ceil(len(examples) / size))):
        batch = collate(examples[index * size:(index + 1) * size], pad_id)
        batch = {key: value.to(device) for key, value in batch.items()}
        loss, supervised = compute_loss(model, batch, autocast, device)
        total += float(loss) * supervised
        weight += supervised
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


def run(config: dict) -> int:
    base, output = check_paths(config)
    transformers = import_transformers()
    from ...device import describe, empty_cache, pick_device

    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    device = pick_device(config["device"])
    say(describe(device))
    precision = resolve_precision(config["precision"], device)
    method = config["method"]
    local_only = not config["allow_download"]

    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(base, local_files_only=local_only)
    except (OSError, ValueError) as error:
        hint = "" if not local_only else " Download it in Studio (Models → Discover) first."
        raise TrainerError(f"Cannot load the tokenizer of {base}: {error}.{hint}") from error
    tokenizer.model_max_length = int(1e30)  # blocks are cut here, not by the tokenizer
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    # LoRA keeps the frozen base in the compact dtype; full training keeps fp32 weights and
    # uses autocast so optimizer updates are not rounded away.
    load_dtype = torch.bfloat16 if method == "lora" and precision == "bf16" else torch.float32
    autocast = precision == "bf16" and method != "lora"
    model = load_pretrained(transformers.AutoModelForCausalLM, base, local_only, load_dtype)
    limit = getattr(model.config, "max_position_embeddings", None) or getattr(
        model.config, "n_positions", None)
    if limit and config["block_size"] > int(limit):
        say(f"block_size reduced to the model's limit of {int(limit)} tokens")
        config["block_size"] = int(limit)
    model.config.use_cache = False
    model.to(device)

    examples, data_stats = build_examples(tokenizer, config)
    training, validation = split_examples(examples, seed=config["seed"])
    per_step = config["batch"] * config["grad_accum"]
    say(f"One epoch ≈ {max(1, math.ceil(len(training) / per_step)):,} steps "
        f"({len(training):,} training / {len(validation):,} validation examples)")

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
        model.gradient_checkpointing_enable()
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
    for step in range(steps):
        rate = learning_rate(step, config)
        for group in optimizer.param_groups:
            group["lr"] = rate
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(config["grad_accum"]):
            batch = collate([sampler.choice(training) for _ in range(config["batch"])], pad_id)
            window_tokens += int(batch["attention_mask"].sum())
            batch = {key: value.to(device) for key, value in batch.items()}
            loss, _ = compute_loss(model, batch, autocast, device)
            if not torch.isfinite(loss):
                raise TrainerError(
                    "The loss became infinite or NaN. Lower the learning rate, or use fp32 precision"
                )
            (loss / config["grad_accum"]).backward()
            step_loss += float(loss) / config["grad_accum"]
        nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        completed = step + 1
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
            measured = evaluate(model, validation, config, pad_id, autocast, device)
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
        }
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
                say("Merging the adapter into a copy of the base model")
                merged = load_pretrained(transformers.AutoModelForCausalLM, base, local_only, "auto")
                load_adapter(merged, staging / "adapter")
                merge_adapter(merged)
                merged.save_pretrained(staging, safe_serialization=True)
                tokenizer.save_pretrained(staging)
        else:
            model.config.use_cache = True
            model.save_pretrained(staging, safe_serialization=True)
            tokenizer.save_pretrained(staging)
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
