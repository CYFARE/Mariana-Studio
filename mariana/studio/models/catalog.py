"""Catalog entries: one uniform description for every model Studio can see.

An entry is a plain dict::

    {"id", "name", "kind", "kind_label", "source", "source_label", "runtime",
     "path", "absolute_path", "bytes", "modified", "available", "reason",
     "detail": {...}, "capabilities": {"chat", "finetune", "remote", ...}}

``available`` is false when the model exists but its backend does not; then
``reason`` says exactly how to fix that.  Nothing here loads model weights.
"""

from __future__ import annotations

import importlib.util
import json
import struct
from pathlib import Path

import torch

from ..errors import BadRequestError
from . import gguf
from .context import model_context

KIND_LABELS = {
    "mariana-drv": "Mariana DRV-LM",
    "mariana-ncm": "Mariana NCM",
    "mariana-ncm-tokenizer": "NCM tokenizer",
    "transformers": "Transformers",
    "gguf": "GGUF",
    "onnx": "ONNX",
    "openai": "OpenAI-compatible",
    "ollama": "Ollama",
    "anthropic": "Anthropic",
}
SOURCE_LABELS = {
    "workspace": "This workspace",
    "huggingface": "Hugging Face cache",
    "ollama": "Ollama",
    "llama.cpp": "llama.cpp",
    "lm-studio": "LM Studio",
    "gpt4all": "GPT4All",
    "jan": "Jan",
    "text-generation-webui": "text-generation-webui",
    "folder": "Model folder",
    "endpoint": "Saved endpoint",
    "server": "Local server",
}
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt")
TRANSFORMERS_HINT = "pip install -r requirements-finetune.txt"
GGUF_HINT = "pip install llama-cpp-python"
ONNX_HINT = 'pip install "optimum[onnxruntime]" transformers'
TOKENIZER_ONLY_REASON = (
    "This is an NCM tokenizer-only checkpoint, not a chat model. "
    "Select the NCM generator checkpoint trained in stage 2."
)
FINETUNE_REASONS = {
    "gguf": (
        "GGUF is a quantized inference format and cannot be trained directly. Download the "
        "original Transformers weights of this model, fine-tune those, then export to GGUF."
    ),
    "onnx": (
        "ONNX graphs are inference-only. Fine-tune the original Transformers weights, then "
        "export again."
    ),
    "ollama": (
        "Ollama serves quantized GGUF models that cannot be trained directly. Download the "
        "original Transformers weights from Hugging Face and fine-tune those."
    ),
    "openai": (
        "This model runs on a remote server, so its weights are not available locally. "
        "Use the provider's own fine-tuning service, or pick a downloadable model."
    ),
    "anthropic": (
        "This model runs on a remote server, so its weights are not available locally."
    ),
    "mariana-ncm-tokenizer": "Fine-tune the NCM generator checkpoint, not its tokenizer.",
}

_checkpoint_cache: dict[tuple, dict] = {}
_safetensors_cache: dict[tuple, dict] = {}


class ModelError(BadRequestError):
    """A user-actionable model selection or inference error."""


def has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def relative(root: Path, path: Path) -> str:
    """The path relative to ``root`` when inside it, else the absolute path."""
    try:
        return str(path.absolute().relative_to(root))
    except ValueError:
        return str(path)


def _entry(root: Path, path: Path, kind: str, source: str, **fields) -> dict:
    try:
        stat = path.stat()
        size, modified = stat.st_size, stat.st_mtime
    except OSError:
        size, modified = 0, 0
    entry = {
        "id": relative(root, path), "name": path.name, "kind": kind,
        "kind_label": KIND_LABELS[kind], "source": source,
        "source_label": SOURCE_LABELS.get(source, source), "runtime": "",
        "path": relative(root, path), "absolute_path": str(path),
        "bytes": size, "modified": modified, "available": True, "reason": "",
        "detail": {}, "capabilities": {"chat": True, "finetune": False},
    }
    entry.update(fields)
    if not entry["capabilities"].get("finetune") and kind in FINETUNE_REASONS:
        entry["finetune_reason"] = FINETUNE_REASONS[kind]
    return entry


# -- Mariana checkpoints ----------------------------------------------------
def inspect_checkpoint(path: Path) -> dict:
    """Read checkpoint metadata without materialising the model."""
    path = Path(path)
    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    cached = _checkpoint_cache.get(key)
    if cached is not None:
        return dict(cached)
    info = {
        "step": None, "stage": None, "format": None, "parameters": None,
        "generator_parameters": None, "ctx_len": None, "dim": None,
        "heads": None, "layers": None, "latent": False, "moe": False,
        "fast_train": False, "tokenizer": None,
    }
    try:
        state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except Exception as error:  # noqa: BLE001 - report any corrupt checkpoint
        raise ModelError(f"Cannot read {path.name}: {error}") from error
    if not isinstance(state, dict) or "model" not in state or "cfg" not in state:
        raise ModelError(f"{path.name} is not a Mariana checkpoint")
    cfg = state.get("cfg") or {}
    info["stage"] = state.get("stage")
    info["step"] = int(state.get("step", 0) or 0)
    info["format"] = state.get("format")
    info["ctx_len"] = cfg.get("ctx_len")
    info["dim"] = cfg.get("dim")
    info["heads"] = cfg.get("n_heads")
    info["latent"] = bool(cfg.get("use_latent"))
    info["moe"] = bool(cfg.get("use_moe"))
    info["fast_train"] = bool(cfg.get("fast_train"))
    info["layers"] = int(cfg.get("n_pre", 0)) + int(cfg.get("n_core", 0)) + int(cfg.get("n_post", 0))
    is_ncm = info["stage"] == "generator" or "levels" in cfg
    if is_ncm:
        info["layers"] = int(cfg.get("generator_layers", info["layers"] or 0))
    total = generator_total = 0
    for name, tensor in state["model"].items():
        if not torch.is_tensor(tensor):
            continue
        count = tensor.numel()
        total += count
        if not name.startswith("tokenizer."):
            generator_total += count
    info["parameters"] = total
    if is_ncm:
        info["generator_parameters"] = generator_total
    info["config"] = cfg
    if len(_checkpoint_cache) > 256:
        _checkpoint_cache.clear()
    _checkpoint_cache[key] = info
    return dict(info)


def checkpoint_entry(root: Path, path: Path, source: str = "workspace") -> dict:
    entry = _entry(
        root, path, "mariana-drv", source, runtime="mariana",
        capabilities={"chat": True, "finetune": True},
    )
    try:
        info = inspect_checkpoint(path)
    except (ModelError, OSError) as error:
        entry.update({
            "available": False, "reason": str(error),
            "capabilities": {"chat": False, "finetune": False},
            "finetune_reason": str(error),
        })
        return entry
    stage = info.get("stage")
    if stage == "tokenizer":
        kind, available, reason = "mariana-ncm-tokenizer", False, TOKENIZER_ONLY_REASON
        capabilities = {"chat": False, "finetune": False}
    elif stage == "generator":
        kind, available, reason = "mariana-ncm", True, ""
        capabilities = {"chat": True, "finetune": True, "ncm": True}
    else:
        kind, available, reason = "mariana-drv", True, ""
        capabilities = {
            "chat": True, "finetune": True, "latent": info["latent"],
            "export": not info["moe"],
        }
    detail = {
        "stage": stage, "step": info["step"],
        "parameters": info["generator_parameters"] if stage == "generator" else info["parameters"],
        "total_parameters": info["parameters"], "ctx_len": info["ctx_len"],
        "ctx_unit": "bytes", "dim": info["dim"], "heads": info["heads"],
        "layers": info["layers"], "latent": info["latent"], "moe": info["moe"],
        "fast_train": info["fast_train"],
        "architecture": "NCM" if stage == "generator" else "DRV-LM",
    }
    if stage == "generator":
        detail["generator_parameters"] = info["generator_parameters"]
        tokenizer = path.with_name(path.stem + "-tokenizer.pt")
        detail["tokenizer"] = relative(root, tokenizer) if tokenizer.exists() else None
    entry.update({
        "kind": kind, "kind_label": KIND_LABELS[kind], "available": available,
        "reason": reason, "detail": detail, "capabilities": capabilities,
    })
    if kind in FINETUNE_REASONS:
        entry["finetune_reason"] = FINETUNE_REASONS[kind]
    return entry


# -- Hugging Face Transformers directories ------------------------------------
def safetensors_summary(path: Path) -> dict:
    """Parameter count and tensor shapes from a safetensors header (no tensor data)."""
    path = Path(path)
    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    cached = _safetensors_cache.get(key)
    if cached is not None:
        return cached
    summary = {"parameters": 0, "shapes": {}, "dtypes": {}}
    try:
        with path.open("rb") as handle:
            length = struct.unpack("<Q", handle.read(8))[0]
            if length <= 0 or length > 512 * 1024 * 1024:
                raise ValueError("implausible header length")
            header = json.loads(handle.read(length))
    except (OSError, ValueError, struct.error):
        return summary
    for name, tensor in header.items():
        if name == "__metadata__" or not isinstance(tensor, dict):
            continue
        shape = tuple(int(size) for size in tensor.get("shape") or ())
        count = 1
        for size in shape:
            count *= size
        summary["parameters"] += count
        summary["shapes"][name] = shape
        dtype = str(tensor.get("dtype", ""))
        summary["dtypes"][dtype] = summary["dtypes"].get(dtype, 0) + count
    if len(_safetensors_cache) > 128:
        _safetensors_cache.clear()
    _safetensors_cache[key] = summary
    return summary


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def transformers_weights(folder: Path) -> list[Path]:
    try:
        return sorted(
            path for path in folder.iterdir()
            if path.suffix in WEIGHT_SUFFIXES and path.is_file()
            and not path.name.startswith(("training_args", "optimizer", "scheduler", "adapter_"))
        )
    except OSError:
        return []


def is_transformers_folder(folder: Path) -> bool:
    config = _read_json(folder / "config.json")
    return bool(config.get("model_type") or config.get("architectures"))


def has_chat_template(folder: Path) -> bool:
    if (folder / "chat_template.jinja").is_file() or (folder / "chat_template.json").is_file():
        return True
    return bool(_read_json(folder / "tokenizer_config.json").get("chat_template"))


def transformers_entry(root: Path, folder: Path, source: str = "workspace",
                       name: str | None = None) -> dict | None:
    """Describe a Transformers model directory, or a LoRA adapter-only directory."""
    adapter_config = _read_json(folder / "adapter" / "adapter_config.json")
    is_full = is_transformers_folder(folder)
    if not is_full and not adapter_config:
        return None
    metadata = _read_json(folder / "config.json") if is_full else {}
    weights = transformers_weights(folder) if is_full else []
    installed = has_module("transformers")
    detail = {
        "architecture": metadata.get("model_type") or (metadata.get("architectures") or [None])[0],
        "parameters": None, "ctx_unit": "tokens",
        "ctx_len": model_context(metadata),
        "dim": metadata.get("hidden_size") or metadata.get("n_embd") or metadata.get("d_model"),
        "layers": (metadata.get("num_hidden_layers") or metadata.get("n_layer")
                   or metadata.get("num_layers")),
        "heads": metadata.get("num_attention_heads") or metadata.get("n_head"),
        "vocab_size": metadata.get("vocab_size"),
        "chat_template": has_chat_template(folder),
        "quantization": str(metadata.get("torch_dtype") or metadata.get("dtype") or "") or None,
    }
    size = sum(path.stat().st_size for path in weights)
    parameters = sum(
        safetensors_summary(path)["parameters"] for path in weights if path.suffix == ".safetensors"
    )
    if parameters:
        detail["parameters"] = parameters
    tuned = _read_json(folder / "mariana-finetune.json")
    if tuned:
        detail["finetuned_from"] = tuned.get("base")
        detail["finetune"] = {
            key: tuned.get(key) for key in
            ("method", "steps_completed", "train_loss", "val_loss", "created", "paused")
        }
    reason = ""
    adapter_only = not is_full
    if adapter_only:
        base = str(adapter_config.get("base_model_name_or_path") or "")
        detail.update({"adapter": True, "base": base, "architecture": "LoRA adapter"})
        adapter_file = folder / "adapter" / "adapter_model.safetensors"
        if adapter_file.is_file():
            size = adapter_file.stat().st_size
            detail["parameters"] = safetensors_summary(adapter_file)["parameters"]
        if not base or not Path(base).is_dir():
            reason = f"The base model of this adapter is missing: {base or 'unknown'}"
    elif not weights:
        reason = "No model weights found in this directory"
    if not installed and not reason:
        reason = f"Transformers backend missing; {TRANSFORMERS_HINT}"
    tunable = is_full and bool(weights)
    entry = _entry(
        root, folder, "transformers", source, runtime="transformers",
        name=name or folder.name, bytes=size, available=not reason, reason=reason,
        detail={key: value for key, value in detail.items() if value is not None},
        capabilities={"chat": True, "finetune": tunable, "stream": True},
    )
    if not tunable:
        entry["finetune_reason"] = (
            "Fine-tune the merged model or the original base instead of an adapter."
            if adapter_only else "No model weights found in this directory"
        )
    elif not installed:
        entry["finetune_reason"] = f"Transformers is not installed; {TRANSFORMERS_HINT}"
    return entry


# -- GGUF / ONNX --------------------------------------------------------------
def gguf_entry(root: Path, path: Path, source: str = "workspace", runtime: str | None = None,
               name: str | None = None, repo: str | None = None) -> dict:
    """``runtime`` is ``"llama-server"``, ``"llama-cpp-python"`` or ``None``."""
    detail = {"architecture": "GGUF", "ctx_unit": "tokens",
              "quantization": gguf.quantization_from_name(name or path.name)}
    if repo:
        detail["repo"] = repo
    entry = _entry(
        root, path, "gguf", source, runtime=runtime or "", name=name or path.name,
        available=bool(runtime),
        reason="" if runtime else (
            "No GGUF runtime found. Build or install llama.cpp so `llama-server` is on PATH, "
            f"or run: {GGUF_HINT}"
        ),
        detail={key: value for key, value in detail.items() if value is not None},
        capabilities={"chat": True, "finetune": False, "stream": True},
    )
    return entry


def gguf_detail(entry: dict) -> dict:
    """Enrich a GGUF entry with header metadata (cached; safe on huge files)."""
    summary = gguf.describe(Path(entry["absolute_path"]))
    detail = dict(entry.get("detail") or {})
    for key in ("architecture", "model_name", "size_label", "quantization", "ctx_len",
                "layers", "dim", "vocab_size", "chat_template", "license", "base_model"):
        if summary.get(key) is not None:
            detail[key] = summary[key]
    return detail


def onnx_entry(root: Path, path: Path, source: str = "workspace") -> dict:
    tokenizer_present = any(
        (path.parent / name).exists() for name in ("tokenizer.json", "tokenizer_config.json")
    )
    available = has_module("optimum") and has_module("onnxruntime") and has_module("transformers")
    reason = ""
    if not available:
        reason = f"ONNX backend missing; {ONNX_HINT}"
    elif not tokenizer_present:
        reason = "Place a tokenizer.json / tokenizer_config.json next to the .onnx file."
        available = False
    return _entry(
        root, path, "onnx", source, runtime="onnxruntime", available=available, reason=reason,
        detail={"architecture": "ONNX", "quantization": "onnx", "ctx_unit": "tokens"},
        capabilities={"chat": True, "finetune": False},
    )


# -- workspace scan -----------------------------------------------------------
def scan_folder(root: Path, base: Path, source: str, gguf_runtime: str | None,
                seen: set[Path], recurse: bool = True, limit: int = 400) -> list[dict]:
    """Find every model under one directory.  ``seen`` de-duplicates across scans."""
    entries: list[dict] = []
    if not base.is_dir():
        return entries

    def fresh(path: Path) -> bool:
        try:
            resolved = path.resolve()
        except OSError:
            return False
        if resolved in seen:
            return False
        seen.add(resolved)
        return True

    pattern = "**/*" if recurse else "*"
    try:
        checkpoints = sorted(base.glob(f"{pattern}.pt"))[:limit]
        gguf_files = sorted(base.glob(f"{pattern}.gguf"))[:limit]
        onnx_files = sorted(base.glob(f"{pattern}.onnx"))[:limit]
        folders = {config.parent for config in list(base.glob(f"{pattern}/config.json"))[:limit]}
        folders.update(
            config.parent.parent
            for config in list(base.glob(f"{pattern}/adapter/adapter_config.json"))[:limit]
        )
        folders.update(child for child in base.iterdir() if child.is_dir())
    except OSError:
        return entries
    transformer_folders = set()
    for folder in sorted(folders):
        if not fresh(folder):
            continue
        entry = transformers_entry(root, folder, source)
        if entry:
            entries.append(entry)
            transformer_folders.add(folder.resolve())
    for path in checkpoints:
        if path.name.endswith(".tmp") or path.parent.resolve() in transformer_folders:
            continue  # weights of a Transformers directory are not Mariana checkpoints
        if path.parent.name == "adapter" or not fresh(path):
            continue
        entries.append(checkpoint_entry(root, path, source))
    for path in gguf_files:
        if gguf.is_auxiliary(path) or not fresh(path):
            continue
        entries.append(gguf_entry(root, path, source, gguf_runtime))
    for path in onnx_files:
        if fresh(path):
            entries.append(onnx_entry(root, path, source))
    return entries
