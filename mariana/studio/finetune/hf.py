"""Fine-tuning plans for third-party Hugging Face causal language models.

A plan validates the request, decides how the data will be read (plain text or
chat rows), estimates how many weights will train, and produces the job the
training manager hands to ``finetune.trainer``.  Nothing here imports
Transformers, so planning works before the optional dependency is installed.
"""

from __future__ import annotations

import re
from pathlib import Path

from ..models.catalog import (
    TRANSFORMERS_HINT,
    ModelError,
    has_module,
    safetensors_summary,
    transformers_weights,
)
from ..validation import choice, flag, number
from .dataset import find_chat_files
from .lora import TARGET_MODES, estimate_lora_parameters, select_target_names

PRESETS = {
    "quick": {
        "label": "Quick LoRA",
        "description": "A small adapter and a short run. Adjusts tone or teaches a format "
                       "without touching the base weights.",
        "method": "lora", "steps": 100, "lr": 0.0002, "warmup": 5, "grad_accum": 1, "lora_r": 8,
    },
    "balanced": {
        "label": "Balanced LoRA",
        "description": "A larger adapter for a new domain or task. The usual place to start.",
        "method": "lora", "steps": 400, "lr": 0.0002, "warmup": 20, "grad_accum": 2, "lora_r": 16,
    },
    "deep": {
        "label": "Full fine-tune",
        "description": "Updates every weight. Needs more data and memory; keep the learning "
                       "rate low to avoid forgetting.",
        "method": "full", "steps": 1000, "lr": 0.00002, "warmup": 50, "grad_accum": 2, "lora_r": 16,
    },
}
METHODS = {
    "lora": "LoRA adapter",
    "last_layers": "Last layers only",
    "full": "Full model",
}
DATA_MODES = ("auto", "text", "chat")
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,80}")
_LAYER_INDEX = re.compile(r"(?:^|\.)(?:h|layers|layer|blocks|block)\.(\d+)\.")
_FINAL_NORM = re.compile(r"(?:^|\.)(?:ln_f|norm|final_layer_norm|final_norm|norm_f)\.")


def _slug(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", name.split("/")[-1]).strip("-.")
    return slug or "model"


def inspect(entry: dict) -> dict:
    """Describe a Transformers catalog entry as a fine-tuning base."""
    detail = entry.get("detail") or {}
    installed = has_module("transformers")
    return {
        "engine": "hf", "id": entry["id"], "path": entry["path"], "name": entry["name"],
        "technique": "hf", "architecture": detail.get("architecture") or "Transformers",
        "step": None, "ctx_len": detail.get("ctx_len"), "ctx_unit": "tokens",
        "dim": detail.get("dim"), "heads": detail.get("heads"), "layers": detail.get("layers"),
        "parameters": detail.get("parameters"), "chat_template": bool(detail.get("chat_template")),
        "finetuned_from": detail.get("finetuned_from"), "latent": False, "moe": False,
        "tokenizer": None, "default_name": f"{_slug(entry['name'])}-tuned",
        "presets": PRESETS, "methods": METHODS,
        "available": installed,
        "requirement": "" if installed else f"Transformers is not installed; {TRANSFORMERS_HINT}",
    }


def _tensor_shapes(folder: Path) -> dict[str, tuple]:
    shapes: dict[str, tuple] = {}
    for path in transformers_weights(folder):
        if path.suffix == ".safetensors":
            shapes.update(safetensors_summary(path)["shapes"])
    return shapes


def _count(shape: tuple) -> int:
    total = 1
    for size in shape:
        total *= size
    return total


def estimate_trainable(folder: Path, method: str, targets, rank: int, layers: int) -> tuple[int, int]:
    """Return ``(total, trainable)`` weights from safetensors headers (0, 0 if unknown)."""
    shapes = _tensor_shapes(folder)
    total = sum(_count(shape) for shape in shapes.values())
    if not total:
        return 0, 0
    if method == "full":
        return total, total
    if method == "lora":
        linear = {name[:-len(".weight")]: shape for name, shape in shapes.items()
                  if name.endswith(".weight") and len(shape) == 2}
        try:
            names = select_target_names(linear, targets)
        except ValueError:
            names = []
        embedding_like = {name for name, shape in linear.items() if shape[0] > 4 * shape[1]}
        chosen = {name: linear[name] for name in names if name not in embedding_like}
        return total, estimate_lora_parameters(chosen, rank)
    indices = sorted({int(match.group(1)) for name in shapes if (match := _LAYER_INDEX.search(name))})
    wanted = set(indices[-max(1, layers):])
    trainable = sum(
        _count(shape) for name, shape in shapes.items()
        if ((match := _LAYER_INDEX.search(name)) and int(match.group(1)) in wanted)
        or (not _LAYER_INDEX.search(name) and _FINAL_NORM.search(name))
    )
    return total, trainable


def plan(root: Path, entry: dict, config: dict) -> dict:
    """Validate a fine-tune request for a Transformers model and build its job."""
    root = Path(root).resolve()
    info = inspect(entry)
    base = Path(entry["absolute_path"]).resolve()
    preset_name = config.get("preset") or "balanced"
    if preset_name not in PRESETS:
        raise ModelError("Unknown fine-tune preset")
    preset = PRESETS[preset_name]
    method = choice(config, "method", preset["method"], METHODS, "Method")

    folder = Path(str(config.get("folder", ""))).expanduser()
    if not str(config.get("folder", "")).strip() or not folder.exists():
        raise ModelError("Select a valid fine-tuning data folder")
    corpus_mode = choice(config, "corpus_mode", "smart", ("smart", "raw"), "Extraction")
    data_mode = choice(config, "data_mode", "auto", DATA_MODES, "Data format")
    chat_files = find_chat_files(folder) if data_mode in {"auto", "chat"} else []
    if data_mode == "chat" and not chat_files:
        raise ModelError(
            "No chat-format data found. Expected .jsonl/.json rows with messages, "
            "prompt + completion, or instruction + output. Switch Data format to Plain text "
            "to train on raw text instead."
        )
    resolved_mode = "chat" if chat_files else "text"

    name = str(config.get("name") or info["default_name"]).strip()
    if name.endswith(".pt"):
        name = name[:-3]
    if not _NAME.fullmatch(name):
        raise ModelError("Model name may contain only letters, numbers, dots, dashes, and underscores")
    output = root / "models" / name
    if output.resolve() == base or base in output.resolve().parents:
        raise ModelError("Choose a new output name so the base model is preserved")
    backup = flag(config, "backup", True)
    if output.exists() and not backup:
        raise ModelError("Output model exists; choose a new name or enable backups")

    steps = number(config, "steps", preset["steps"], 1, 10_000_000)
    learning_rate = number(config, "lr", preset["lr"], 1e-8, 1.0, float)
    batch = number(config, "batch", 4, 1, 4096)
    grad_accum = number(config, "grad_accum", preset["grad_accum"], 1, 4096)
    warmup = number(config, "warmup", preset["warmup"], 0, 1_000_000)
    eval_every = number(config, "eval_every", max(10, min(100, steps)), 1, 10_000_000)
    context = int(info["ctx_len"] or 0)
    block_size = number(config, "block_size", min(512, context) if context else 512, 16, 32768)
    rank = number(config, "lora_r", preset["lora_r"], 1, 512)
    alpha = number(config, "lora_alpha", 2 * rank, 1, 4096, float)
    dropout = number(config, "lora_dropout", 0.05, 0.0, 0.9, float)
    targets = choice(config, "lora_targets", "auto", TARGET_MODES, "LoRA targets")
    layer_count = int(info["layers"] or 0)
    trainable_layers = number(config, "trainable_layers", 2, 1, max(1, layer_count or 1024))
    train_on = choice(config, "train_on", "assistant", ("assistant", "all"), "Train on")
    precision = choice(config, "precision", "auto", ("auto", "fp32", "bf16"), "Precision")
    merge = flag(config, "merge", True)

    total, trainable = estimate_trainable(base, method, targets, rank, trainable_layers)
    total = total or int(info["parameters"] or 0)
    warnings = []
    if not info["available"]:
        warnings.append(f"{info['requirement']} — the run cannot start until it is installed.")
    if context and block_size > context:
        warnings.append(f"Block size exceeds the model's {context}-token context; it will be reduced.")
    if resolved_mode == "chat":
        others = "" if data_mode == "chat" else " Other files in the folder are ignored."
        warnings.append(
            f"Found {len(chat_files)} chat-format file(s); training on conversations.{others}"
        )
        if not info["chat_template"]:
            warnings.append(
                "This model has no chat template, so conversations are trained as a plain "
                "\"User: / Assistant:\" transcript — the same layout Studio's chat uses for it."
            )
    if method == "full" and learning_rate > 0.0001:
        warnings.append("Full fine-tuning above 1e-4 often forgets the base model's behaviour.")
    if method == "lora" and learning_rate < 0.00002:
        warnings.append("LoRA usually needs a higher learning rate (1e-4 to 3e-4) to learn.")
    if output.exists():
        warnings.append("The output model exists and will be moved to backups/ before writing.")

    scope = {
        "lora": f"LoRA r={rank}, alpha={alpha:g} on {targets} layers",
        "last_layers": f"Last {trainable_layers} of {layer_count or '?'} blocks",
        "full": "Every weight",
    }[method]
    trainer = {
        "method": method,
        "lora": {"r": rank, "alpha": alpha, "dropout": dropout, "targets": targets},
        "trainable_layers": trainable_layers, "steps": steps, "batch": batch,
        "grad_accum": grad_accum, "lr": learning_rate, "warmup": warmup,
        "block_size": block_size, "eval_every": eval_every, "device": config.get("device", "auto"),
        "precision": precision, "gradient_checkpointing": flag(config, "gradient_checkpointing"),
        "merge": merge, "train_on": train_on,
    }
    normalized = {
        "engine": "hf", "mode": "finetune", "base": str(base), "base_name": entry["name"],
        "name": name, "output": str(output), "folder": str(folder), "corpus_mode": corpus_mode,
        "data_mode": resolved_mode, "chat_files": [str(path) for path in chat_files],
        "steps": steps, "backup": backup, "preset": preset_name, "trainer": trainer,
    }
    summary = [
        ["Base", entry["name"]],
        ["Output", str(output.relative_to(root))],
        ["Preset", preset["label"]],
        ["Method", METHODS[method]],
        ["Scope", scope],
        ["Data", f"{len(chat_files)} chat file(s)" if resolved_mode == "chat" else "Plain text"],
        ["Steps", f"{steps:,}"],
        ["Learning rate", f"{learning_rate:g}"],
        ["Batch × accumulation", f"{batch} × {grad_accum}"],
        ["Block size", f"{block_size} tokens"],
    ]
    if method == "lora":
        summary.append(["Result", "Adapter + merged model" if merge else "Adapter only"])
    return {
        "engine": "hf", "base": info, "config": normalized,
        "preset": {"name": preset_name, **preset}, "summary": summary,
        "method": {"name": method, "label": METHODS[method]},
        "data": {"mode": resolved_mode, "chat_files": len(chat_files)},
        "estimate": {
            "parameters": total, "trainable": trainable, "frozen": max(0, total - trainable),
            "trainable_percent": (100.0 * trainable / total) if total else 0.0,
        },
        "output": str(output.relative_to(root)), "warnings": warnings,
        "ready": info["available"],
    }
