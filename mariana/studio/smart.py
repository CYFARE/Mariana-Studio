"""Smart Options: size a training run from the corpus and the hardware.

Given extracted byte-tokens, the technique and the device memory this module
recommends context, width, batch, accumulation and step counts, and explains
the estimate so the user can override it knowingly.
"""

from __future__ import annotations

import math
import os

import torch

from ..device import pick_device


def _system_memory():
    try:
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        return None


def _device_memory(device):
    try:
        if device.type == "cuda":
            return torch.cuda.get_device_properties(device).total_memory
        if device.type == "xpu" and hasattr(torch, "xpu"):
            return torch.xpu.get_device_properties(device).total_memory
    except Exception:
        pass
    return _system_memory()


def _levels(ctx):
    from ..ncm.config import default_levels

    return ",".join(str(value) for value in default_levels(ctx))


def _parameter_estimate(config, technique, options=None):
    options = options or {}
    layers = config["n_pre"] + config["n_core"] * config["loops"] + config["n_post"]
    if technique == "ncm":
        parameters = layers * 16 * config["dim"] ** 2 + 300 * config["dim"]
        return parameters + 4 * 16 * 128**2 + 512 * 128 * (config["levels"].count(",") + 1)

    dim = config["dim"]
    latent_mode = options.get("latent_mode", "none")
    blocks = config["n_pre"] + config["n_core"] + config["n_post"]
    block_parameters = 58 if options.get("moe") else 22
    if latent_mode == "saint":
        block_parameters += 4
    parameters = blocks * block_parameters * dim**2 + 2 * dim**2 + 300 * dim
    if latent_mode in {"saint", "legacy"}:
        symbols = int(options.get("latent_symbols", 256))
        parameters += 2 * symbols * dim
    if latent_mode == "saint":
        slots = int(options.get("latent_continuous_slots", 2))
        refiners = int(options.get("latent_refine_steps", 2))
        parameters += refiners * 16 * dim**2 + 2 * slots * dim**2
        parameters += 4 * symbols * dim + 7 * dim**2
    return parameters


def recommend_training_options(corpus, technique="fast", requested_device="auto", options=None):
    if technique not in {"fast", "legacy", "ncm"}:
        raise ValueError("Unknown training technique")
    options = dict(options or {})
    latent_mode = options.get("latent_mode", "none")
    if latent_mode not in {"none", "saint", "legacy"}:
        raise ValueError("Latent mode must be none, saint, or legacy")
    if technique == "ncm":
        latent_mode = "none"
    options["latent_mode"] = latent_mode
    options["moe"] = technique != "ncm" and bool(options.get("moe"))
    size = max(1, int(corpus.get("output_bytes") or corpus.get("bytes") or 1))
    if size < 128_000:
        ctx = 64
    elif size < 2_000_000:
        ctx = 128
    elif size < 32_000_000:
        ctx = 256
    else:
        ctx = 512
    if size < 50_000:
        dim = 128
    elif size < 250_000:
        dim = 192
    elif size < 4_000_000:
        dim = 256
    elif size < 64_000_000:
        dim = 384
    else:
        dim = 512
    heads = 4 if dim <= 256 else 8
    device = pick_device(requested_device)
    memory = _device_memory(device)
    target_epochs = 3.0 if size < 2_000_000 else 2.0 if size < 64_000_000 else 1.0
    lr_scale = math.sqrt(256 / dim)
    config = {
        "batch": 1, "grad_accum": 32, "ctx": ctx, "dim": dim, "heads": heads,
        "lr": round(min(3e-3, max(5e-4, 3e-3 * lr_scale)), 8),
        "n_pre": 1 if dim <= 192 else 2,
        "n_core": 1 if dim <= 192 else 2,
        "n_post": 1 if dim <= 192 else 2,
        "loops": 2, "block_size": min(32, ctx),
    }
    if technique == "legacy":
        config["lr"] = round(min(2e-3, max(3e-4, 1.5e-3 * lr_scale)), 8)
    flex_available = False
    if technique == "ncm":
        from ..ncm.layers import HAS_FLEX

        flex_available = HAS_FLEX and device.type == "cuda" and dim // heads >= 16
        config.update({
            "generator_lr": round(min(1e-3, max(1e-4, 1e-3 * lr_scale)), 8),
            "prefix": min(64, max(1, ctx // 2)), "levels": _levels(ctx),
        })
    else:
        config.update({"moe": options["moe"], "latent_mode": latent_mode})

    parameters = _parameter_estimate(config, technique, options)
    layers = config["n_pre"] + config["n_core"] * config["loops"] + config["n_post"]
    per_sequence = ctx * layers * dim * 320 + ctx * ctx * heads * 8
    if technique == "legacy":
        per_sequence *= 2
    elif technique == "ncm":
        per_sequence = int(per_sequence * 1.5)
    if device.type == "cpu":
        memory_batch = max(1, int((memory or 8 * 1024**3) * 0.5 / max(per_sequence, 1)))
        cpu_batch = max(1, min(8, (os.cpu_count() or 4) // 4))
        batch = min(cpu_batch, memory_batch)
    else:
        budget = (memory or 4 * 1024**3) * 0.65
        fixed = parameters * 16 + 512 * 1024**2
        batch = int((budget - fixed) / max(per_sequence, 1))
        batch = max(1, min(16, batch))
    measured_8gb = (
        device.type == "cuda" and memory
        and 6 * 1024**3 <= memory <= 10 * 1024**3
        and size >= 64_000_000 and ctx == 512 and dim == 512
        and not options["moe"] and latent_mode == "none"
    )
    if measured_8gb:
        batch = 6 if technique == "legacy" else 12 if technique == "fast" else 4
    grad_accum = min(32, max(1, math.ceil(32 / batch)))
    effective_batch = batch * grad_accum
    positions_per_step = effective_batch * ctx
    steps = max(10, min(100_000, round(size * target_epochs / positions_per_step)))
    estimated_epochs = steps * positions_per_step / size
    config.update({
        "batch": batch, "grad_accum": grad_accum, "steps": steps,
        "warmup": max(1, min(200, round(steps * 0.03))),
        "eval_every": max(1, round(steps / 10)),
        "compile": device.type == "cuda" and (technique == "ncm" or latent_mode == "none"),
        "flex": flex_available,
        "cache_codes": technique == "ncm" and bool(options.get("cache_codes")),
    })
    if technique == "ncm":
        tokenizer_batch = 24 if measured_8gb else min(24 if device.type != "cpu" else 8, max(batch, batch * 4))
        tokenizer_grad_accum = min(32, max(1, math.ceil(32 / tokenizer_batch)))
        tokenizer_positions = tokenizer_batch * tokenizer_grad_accum * ctx
        config.update({
            "tokenizer_batch": tokenizer_batch,
            "tokenizer_grad_accum": tokenizer_grad_accum,
            "tokenizer_steps": max(10, round(steps * positions_per_step * 0.5 / tokenizer_positions)),
        })
    warnings = []
    if size < 128_000:
        warnings.append("This is a very small corpus. Limiting passes cannot by itself prevent memorization; more data is recommended.")
    if estimated_epochs > 5:
        warnings.append(f"The small dataset requires about {estimated_epochs:.1f} passes. Watch validation loss closely for overfitting.")
    if steps == 100_000 and estimated_epochs < target_epochs:
        warnings.append(f"The 100k step cap limits training to about {estimated_epochs:.2f} epochs; raise steps manually for a longer run.")
    estimated_memory = parameters * 16 + 512 * 1024**2 + per_sequence * batch
    if memory and estimated_memory > memory * 0.9:
        warnings.append("The recommended batch may be tight on this device. If it OOMs, lower batch size or context length.")
    ram = _system_memory()
    if device.type == "cpu" and ram and size * 1.5 > ram:
        warnings.append("The extracted corpus is large relative to available RAM. Use a subset or train in shards.")
    return {
        "config": config,
        "analysis": {
            "corpus_bytes": size, "byte_tokens": size, "files": corpus.get("files", 0),
            "documents": corpus.get("documents", corpus.get("files", 0)),
            "signature": corpus.get("signature"), "device": str(device),
            "moe": options["moe"], "latent_mode": latent_mode,
            "device_bytes": memory, "target_epochs": target_epochs,
            "estimated_epochs": round(estimated_epochs, 3),
            "effective_batch": effective_batch, "positions_per_step": positions_per_step,
            "estimated_parameters": parameters, "estimated_memory": estimated_memory,
            "warnings": warnings,
        },
    }
