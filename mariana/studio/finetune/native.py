"""Fine-tuning plans for Mariana's own checkpoints (DRV-LM and NCM generators).

Deliberately opinionated: choose a base checkpoint and data, then a preset.
The plan validates the architecture, computes a freeze scope and launches the
training manager in ``finetune`` mode, so the live run panel, pause/resume,
backups and logs work exactly as they do for training.
"""

from __future__ import annotations

import re
from pathlib import Path

import torch

from ..models.catalog import ModelError, inspect_checkpoint

PRESETS = {
    "quick": {
        "label": "Quick refresh",
        "description": "A short, gentle pass that nudges style or vocabulary without disturbing the base model.",
        "steps": 200,
        "lr": 0.0001,
        "warmup": 10,
        "grad_accum": 2,
        "freeze": "last_block",
    },
    "balanced": {
        "label": "Balanced adaptation",
        "description": "Adapts the last few layers to a new domain while keeping most of the base knowledge intact.",
        "steps": 600,
        "lr": 0.0003,
        "warmup": 25,
        "grad_accum": 2,
        "freeze": "last_three",
    },
    "deep": {
        "label": "Deep adaptation",
        "description": "Updates the whole model. Use a larger corpus and a low learning rate to avoid forgetting.",
        "steps": 1500,
        "lr": 0.0005,
        "warmup": 50,
        "grad_accum": 1,
        "freeze": "full",
    },
}

SCOPES = {
    "full": (False, 0.0),
    "last_block": (True, 1.0),
    "last_three": (True, 3.0),
    "freeze_embeddings": (True, 0.0),
    "embeddings": (True, 0.0),
}
SCOPE_LABELS = {
    "full": "Full model",
    "last_block": "Last block",
    "last_three": "Last three layers",
    "freeze_embeddings": "All blocks, frozen embeddings",
    "embeddings": "All blocks, frozen embeddings",
}


def _checkpoint_name(value: str, base: Path) -> str:
    value = (value or f"{base.stem}-tuned.pt").strip()
    if Path(value).name != value or not re.fullmatch(r"[A-Za-z0-9._-]+", value):
        raise ModelError("Model name may contain only letters, numbers, dots, dashes, and underscores")
    return value if value.endswith(".pt") else value + ".pt"


def _resolve_base(root: Path, value: str | None) -> Path:
    if not value:
        raise ModelError("Select a base checkpoint to fine-tune")
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if path.suffix != ".pt" or not path.is_file():
        raise ModelError("Base checkpoint does not exist")
    try:
        path.relative_to(root)
    except ValueError:
        raise ModelError("Base checkpoint must be inside the Mariana workspace")
    return path


def inspect(root: Path, value: str | None) -> dict:
    root = Path(root).resolve()
    path = _resolve_base(root, value)
    info = inspect_checkpoint(path)
    cfg = info.get("config") or {}
    stage = info.get("stage")
    if stage == "tokenizer":
        raise ModelError("NCM tokenizer checkpoints cannot be fine-tuned as chat models")
    technique = "ncm" if stage == "generator" else ("fast" if cfg.get("fast_train") else "legacy")
    tokenizer = path.with_name(path.stem + "-tokenizer.pt")
    layers = int(cfg.get("generator_layers", 0) if stage == "generator"
                 else int(cfg.get("n_pre", 0)) + int(cfg.get("n_core", 0)) + int(cfg.get("n_post", 0)))
    return {
        "engine": "mariana",
        "id": str(path.relative_to(root)),
        "path": str(path.relative_to(root)),
        "name": path.name,
        "available": True,
        "ctx_unit": "bytes",
        "technique": technique,
        "architecture": "NCM generator" if stage == "generator" else "DRV-LM",
        "stage": stage,
        "step": info.get("step"),
        "ctx_len": cfg.get("ctx_len"),
        "dim": cfg.get("dim"),
        "heads": cfg.get("n_heads"),
        "layers": layers,
        "parameters": info.get("generator_parameters") or info.get("parameters"),
        "latent": bool(cfg.get("use_latent")),
        "moe": bool(cfg.get("use_moe")),
        "tokenizer": str(tokenizer.relative_to(root)) if tokenizer.exists() else None,
        "default_name": f"{path.stem}-tuned.pt",
        "presets": PRESETS,
        "config": cfg,
    }


def _freeze_count(info: dict, scope: str) -> tuple[int, bool]:
    layers = int(info.get("layers") or 0)
    if scope == "full":
        return 0, False
    if scope in {"embeddings", "freeze_embeddings"}:
        return 0, True
    if scope == "last_block":
        return max(0, layers - 1), True
    if scope == "last_three":
        return max(0, layers - 3), True
    raise ModelError("Unknown freeze scope")


def _estimate_trainable(path: Path, is_ncm: bool, freeze_layers: int,
                        freeze_embeddings: bool, cfg: dict) -> tuple[int, int]:
    try:
        state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except Exception:  # noqa: BLE001
        return 0, 0
    n_pre = int(cfg.get("n_pre", 0) or 0)
    n_core = int(cfg.get("n_core", 0) or 0)
    total = trainable = 0
    for name, tensor in state.get("model", {}).items():
        if not torch.is_tensor(tensor):
            continue
        if is_ncm and name.startswith("tokenizer."):
            continue
        count = tensor.numel()
        total += count
        frozen = name.startswith("tokenizer.")
        if freeze_embeddings and any(
            name.startswith(prefix)
            for prefix in ("tok_emb.", "time_emb.", "region_emb.", "time_mlp.",
                           "prompt_tokens.", "level_embedding.")
        ):
            frozen = True
        block_index = None
        if is_ncm:
            match = re.match(r"blocks\.(\d+)\.", name)
            if match:
                block_index = int(match.group(1))
        else:
            match = re.match(r"(pre|core|post)\.(\d+)\.", name)
            if match:
                prefix, index = match.group(1), int(match.group(2))
                if prefix == "pre":
                    block_index = index
                elif prefix == "core":
                    block_index = n_pre + index
                else:
                    block_index = n_pre + n_core + index
        if block_index is not None and block_index < freeze_layers:
            frozen = True
        if not frozen:
            trainable += count
    return total, trainable


def plan(root: Path, config: dict) -> dict:
    root = Path(root).resolve()
    base_path = _resolve_base(root, config.get("base"))
    info = inspect(root, config.get("base"))
    preset_name = config.get("preset", "balanced")
    if preset_name not in PRESETS:
        raise ModelError("Unknown fine-tune preset")
    preset = PRESETS[preset_name]
    scope = config.get("freeze") or preset["freeze"]
    if scope not in SCOPES:
        raise ModelError("Unknown freeze scope")
    freeze_layers, freeze_embeddings = _freeze_count(info, scope)

    folder = Path(str(config.get("folder", ""))).expanduser()
    # An empty value must not fall through to Path("") — that is the current directory.
    if not str(config.get("folder", "")).strip() or not folder.exists():
        raise ModelError("Select a valid fine-tuning corpus folder")
    corpus_mode = config.get("corpus_mode", "smart")
    if corpus_mode not in {"smart", "raw"}:
        raise ModelError("Unknown corpus extraction mode")

    output = _checkpoint_name(config.get("name") or info["default_name"], base_path)
    if output == base_path.name:
        raise ModelError("Choose a new output name so the base checkpoint is preserved")
    target = root / "trained-model" / output
    if target.exists() and not config.get("backup", True):
        raise ModelError("Output checkpoint exists; choose a new name or enable backups")

    def number(key, default, minimum, maximum, cast=float):
        try:
            value = cast(config.get(key, default))
        except (TypeError, ValueError):
            raise ModelError(f"{key} must be a number")
        if not minimum <= value <= maximum:
            raise ModelError(f"{key} must be between {minimum} and {maximum}")
        return value

    steps = number("steps", preset["steps"], 1, 10_000_000, int)
    lr = number("lr", preset["lr"], 1e-8, 1.0)
    batch = number("batch", 8, 1, 4096, int)
    grad_accum = number("grad_accum", preset["grad_accum"], 1, 4096, int)
    warmup = number("warmup", preset["warmup"], 0, 1_000_000, int)
    total, trainable = _estimate_trainable(
        base_path, info["architecture"].startswith("NCM"), freeze_layers,
        freeze_embeddings, info.get("config") or {},
    )
    warnings = []
    if info["step"] and steps > 2500:
        warnings.append("This is a long fine-tune; consider fewer steps or more data.")
    if scope == "full" and lr > 0.001:
        warnings.append("Full-model fine-tuning above 0.001 can forget the base behaviour.")
    if target.exists():
        warnings.append("The output checkpoint exists and will be backed up before writing.")
    if info["architecture"].startswith("NCM") and not info.get("tokenizer"):
        raise ModelError("NCM base checkpoint is missing its companion -tokenizer.pt file")

    cfg = info.get("config") or {}
    normalized = {
        "mode": "finetune",
        "base": str(base_path),
        "name": output,
        "folder": str(folder),
        "corpus_mode": corpus_mode,
        "technique": info["technique"],
        "preset": preset_name,
        "freeze": scope,
        "freeze_layers": freeze_layers,
        "freeze_embeddings": freeze_embeddings,
        "steps": steps,
        "lr": lr,
        "batch": batch,
        "grad_accum": grad_accum,
        "warmup": warmup,
        "eval_every": number("eval_every", max(50, min(200, steps)), 1, 10_000_000, int),
        "device": config.get("device", "auto"),
        "backup": bool(config.get("backup", True)),
        "ctx": cfg.get("ctx_len", 256),
        "dim": cfg.get("dim", 256),
        "heads": cfg.get("n_heads", 4),
        "n_pre": cfg.get("n_pre", 2),
        "n_core": cfg.get("n_core", 2),
        "n_post": cfg.get("n_post", 2),
        "loops": cfg.get("max_loops", 2),
        "block_size": cfg.get("block_size", 32),
        "moe": bool(cfg.get("use_moe")),
        "grouped_moe": bool(cfg.get("grouped_moe", True)),
        "latent_mode": (
            "saint" if cfg.get("latent_saint") else
            "legacy" if cfg.get("use_latent") else "none"
        ),
        "thoughts": cfg.get("n_thoughts", 8),
        "latent_symbols": cfg.get("n_latent_symbols", 256),
        "latent_window": cfg.get("latent_local_window", 32),
        "latent_min_thoughts": cfg.get("latent_min_thoughts", 2),
        "latent_block_size": cfg.get("latent_block_size", 2),
        "latent_continuous_slots": cfg.get("latent_continuous_slots", 2),
        "latent_refine_steps": cfg.get("latent_refine_steps", 2),
        "latent_grammar_prob": cfg.get("latent_grammar_prob", 0.25),
        "w_latent_rl": cfg.get("w_latent_rl", 0.1),
        "tokenizer_steps": 1,
        "generator_lr": lr,
        "compile": False,
        "flex": False,
        "cache_codes": False,
        "smart": False,
    }
    return {
        "engine": "mariana",
        "base": info,
        "config": normalized,
        "preset": {"name": preset_name, **preset},
        "summary": [
            ["Base", info["name"]],
            ["Output", str(target.relative_to(root))],
            ["Preset", preset["label"]],
            ["Scope", f"{SCOPE_LABELS[scope]} · {freeze_layers}/{info['layers']} layers frozen"],
            ["Steps", f"{steps:,}"],
            ["Learning rate", f"{lr:g}"],
            ["Batch × accumulation", f"{batch} × {grad_accum}"],
        ],
        "freeze": {
            "scope": scope,
            "layers": freeze_layers,
            "embeddings": freeze_embeddings,
            "total_layers": info["layers"],
        },
        "estimate": {
            "parameters": total,
            "trainable": trainable,
            "frozen": max(0, total - trainable),
            "trainable_percent": (100.0 * trainable / total) if total else 0.0,
        },
        "output": str(target.relative_to(root)),
        "warnings": warnings,
    }


def start(manager, config: dict) -> dict:
    """Validate and launch a fine-tune through the shared training manager."""
    planned = plan(manager.root, config)
    job_config = dict(planned["config"])
    job_config["smart"] = False
    return manager.start(job_config)
