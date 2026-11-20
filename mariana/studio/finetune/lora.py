"""Native LoRA adapters for causal language models: inject, save, load and merge.

Needs only torch (plus safetensors to read or write adapter files), so adapters can be
planned, merged and inspected without Transformers or PEFT.  Adapters are stored in the PEFT
layout -- ``adapter_config.json`` next to ``adapter_model.safetensors`` -- and load in
either library.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import torch
from torch import nn

ADAPTER_CONFIG_NAME = "adapter_config.json"
ADAPTER_WEIGHTS_NAME = "adapter_model.safetensors"
# PEFT prefixes every adapter tensor with the attribute path of the wrapped model.
PEFT_KEY_PREFIX = "base_model.model."
TARGET_MODES = ("auto", "attention", "all-linear")
ATTENTION_TARGETS = frozenset({
    "q_proj", "k_proj", "v_proj", "o_proj", "out_proj",
    "query", "key", "value", "c_attn", "qkv_proj",
})
# The output head is usually tied to the input embedding; adapting it would make a merge
# rewrite the embedding table as well.
HEAD_NAMES = frozenset({"lm_head", "embed_out", "classifier", "score"})
# PEFT options that change the maths of the update and are not implemented here.
UNSUPPORTED_PEFT_OPTIONS = ("use_dora", "use_rslora", "rank_pattern", "alpha_pattern", "modules_to_save")
INSTALL_HINT = "pip install -r requirements-finetune.txt"


def _is_conv1d(module: nn.Module) -> bool:
    # Matched by name so that Transformers does not have to be importable.
    return type(module).__name__ == "Conv1D"


def _is_linear(module: nn.Module) -> bool:
    return isinstance(module, nn.Linear) or _is_conv1d(module)


class LoRALinear(nn.Module):
    """A linear layer plus a trainable low-rank update: ``base(x) + alpha / r * B(A(x))``.

    ``base`` is an ``nn.Linear`` or a GPT-2 style ``Conv1D``.  The update is kept in float32
    whatever the dtype of ``base``, so small optimizer steps are not rounded away when the
    base model is loaded in bf16 or fp16.
    """

    def __init__(self, base: nn.Module, r: int = 16, alpha: float = 32, dropout: float = 0.05):
        super().__init__()
        if not _is_linear(base):
            raise ValueError(
                f"LoRA can wrap nn.Linear or Conv1D layers, not {type(base).__name__}; "
                "pick linear layers as targets"
            )
        if r < 1:
            raise ValueError(f"LoRA rank r must be at least 1 (got {r}); try r=8 or r=16")
        # Conv1D stores its weight as (in, out), nn.Linear as (out, in).
        self.fan_in_fan_out = _is_conv1d(base)
        rows, columns = base.weight.shape
        in_features, out_features = (rows, columns) if self.fan_in_fan_out else (columns, rows)
        self.base = base
        self.r = int(r)
        self.alpha = alpha
        self.scaling = alpha / self.r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        device = base.weight.device
        self.lora_A = nn.Linear(in_features, self.r, bias=False, device=device, dtype=torch.float32)
        self.lora_B = nn.Linear(self.r, out_features, bias=False, device=device, dtype=torch.float32)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)  # the update starts at exactly zero
        self.is_merged = False

    @property
    def weight(self) -> torch.Tensor:
        # Some architectures read ``layer.weight.dtype`` of their projections directly.
        return self.base.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.base.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base(x)
        if self.is_merged:  # the update already lives in the base weight
            return result
        update = self.lora_B(self.lora_A(self.dropout(x.to(self.lora_A.weight.dtype))))
        return result + (self.scaling * update).to(result.dtype)

    @torch.no_grad()
    def merged(self) -> nn.Module:
        """Fold the update into the base weight (once) and return the base layer."""
        if not self.is_merged:
            weight = self.base.weight
            delta = self.scaling * (self.lora_B.weight.float() @ self.lora_A.weight.float())
            if self.fan_in_fan_out:
                delta = delta.T
            # Sum in float32 and round once, so a bf16/fp16 base loses no extra precision.
            weight.copy_((weight.float() + delta.to(weight.device)).to(weight.dtype))
            self.is_merged = True
        return self.base

    def extra_repr(self) -> str:
        return f"r={self.r}, alpha={self.alpha}, fan_in_fan_out={self.fan_in_fan_out}"


def _is_attention_projection(name: str) -> bool:
    parent, _, leaf = name.rpartition(".")
    if leaf in ATTENTION_TARGETS:
        return True
    # GPT-2 names both the attention output and the MLP output ``c_proj``; only the first counts.
    return leaf == "c_proj" and any(marker in parent.lower() for marker in ("attn", "attention"))


def select_target_names(names: Iterable[str], targets: str | Sequence[str] = "auto") -> list[str]:
    """Pick the LoRA targets among linear-layer names.

    Works on names alone, so a planner can apply the same rules to the tensor names of a
    safetensors header.  ``targets`` is ``"auto"``, ``"attention"``, ``"all-linear"`` or a list
    of name suffixes such as ``["q_proj", "v_proj"]``.  Output heads are never selected.
    """
    candidates = [name for name in names if name.rpartition(".")[2] not in HEAD_NAMES]
    if isinstance(targets, str):
        if targets not in TARGET_MODES:
            raise ValueError(
                f"Unknown LoRA targets {targets!r}; use one of {', '.join(TARGET_MODES)} "
                "or a list of layer name suffixes"
            )
        if targets == "all-linear":
            return candidates
        attention = [name for name in candidates if _is_attention_projection(name)]
        return attention if attention or targets == "attention" else candidates
    suffixes = [str(target) for target in targets]
    return [
        name for name in candidates
        if any(name == suffix or name.endswith("." + suffix) for suffix in suffixes)
    ]


def _find_output_head(model: nn.Module) -> nn.Module | None:
    getter = getattr(model, "get_output_embeddings", None)
    if getter is None:
        return None
    try:
        return getter()
    except (NotImplementedError, AttributeError, TypeError):  # not every architecture has one
        return None


def _list_linear_names(model: nn.Module) -> list[str]:
    """Names of the linear layers LoRA may wrap: no output head, nothing inside an adapter."""
    head = _find_output_head(model)
    wrapped = {name for name, module in model.named_modules() if isinstance(module, LoRALinear)}
    return [
        name for name, module in model.named_modules()
        if name and _is_linear(module) and module is not head
        and name.rpartition(".")[0] not in wrapped
    ]


def find_target_modules(model: nn.Module, targets: str | Sequence[str] = "auto") -> list[str]:
    """Return the full dotted names of the layers that ``targets`` selects in ``model``."""
    return select_target_names(_list_linear_names(model), targets)


def _wrap_modules(model: nn.Module, names: Sequence[str], r: int, alpha: float, dropout: float) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for name in names:
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, LoRALinear(getattr(parent, child_name), r, alpha, dropout))


def _require_no_adapter(model: nn.Module) -> None:
    if any(isinstance(module, LoRALinear) for module in model.modules()):
        raise ValueError("The model already has LoRA layers; call merge_adapter() before adding more")


def inject_lora(model: nn.Module, r: int = 16, alpha: float = 32, dropout: float = 0.05,
                targets: str | Sequence[str] = "auto") -> list[str]:
    """Freeze ``model`` and wrap its target layers in :class:`LoRALinear`; return their names."""
    if r < 1:
        raise ValueError(f"LoRA rank r must be at least 1 (got {r}); try r=8 or r=16")
    _require_no_adapter(model)
    names = find_target_modules(model, targets)
    if not names:
        leaves = sorted({name.rpartition(".")[2] for name in _list_linear_names(model)})
        if not leaves:
            raise ValueError(
                "The model has no linear layers LoRA can adapt; use full or last-layers training instead"
            )
        raise ValueError(
            f"No layer matches LoRA targets {targets!r}; use 'all-linear' or name suffixes "
            f"from this model: {', '.join(leaves[:12])}"
        )
    _wrap_modules(model, names, r, alpha, dropout)
    return names


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return the adapter weights under PEFT-compatible keys."""
    state: dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            state[f"{PEFT_KEY_PREFIX}{name}.lora_A.weight"] = module.lora_A.weight.detach().cpu()
            state[f"{PEFT_KEY_PREFIX}{name}.lora_B.weight"] = module.lora_B.weight.detach().cpu()
    return state


def _import_safetensors():
    try:
        from safetensors.torch import load_file, save_file
    except ImportError as error:
        raise ImportError(f"safetensors is not installed. Run: {INSTALL_HINT}") from error
    return load_file, save_file


def _replace_atomically(path: Path, write) -> None:
    """Run ``write(temporary_path)`` and move the result over ``path`` in one step."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        write(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_adapter(model: nn.Module, directory: str | Path, *, base_model: str, r: int, alpha: float,
                 dropout: float, target_modules: Sequence[str], extra: Mapping | None = None) -> None:
    """Write the adapter of ``model`` to ``directory`` in the PEFT file layout.

    ``extra`` adds top-level keys to ``adapter_config.json``; PEFT ignores keys it does not
    know (with a warning), so prefer PEFT's own field names.
    """
    state = {key: value.to(torch.float32).contiguous() for key, value in lora_state_dict(model).items()}
    if not state:
        raise ValueError("The model has no LoRA layers to save; call inject_lora() first")
    config = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": str(base_model),
        "r": int(r),
        "lora_alpha": alpha,
        "lora_dropout": dropout,
        "target_modules": list(target_modules),
        "fan_in_fan_out": any(
            module.fan_in_fan_out for module in model.modules() if isinstance(module, LoRALinear)
        ),
        "bias": "none",
        "inference_mode": True,
    }
    clashes = sorted(set(extra or {}) & set(config))
    if clashes:
        raise ValueError(f"extra may not redefine adapter settings: {', '.join(clashes)}; remove those keys")
    config.update(extra or {})
    _, save_file = _import_safetensors()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # Weights first: a folder only counts as an adapter once its config exists.
    _replace_atomically(
        directory / ADAPTER_WEIGHTS_NAME,
        lambda path: save_file(state, str(path), metadata={"format": "pt"}),
    )
    _replace_atomically(
        directory / ADAPTER_CONFIG_NAME,
        lambda path: path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"),
    )


def _read_adapter_config(directory: Path) -> dict:
    path = directory / ADAPTER_CONFIG_NAME
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(
            f"Cannot read {path}; choose a folder written by save_adapter() or PEFT"
        ) from error
    if not isinstance(config, dict) or config.get("peft_type") != "LORA":
        raise ValueError(f"{path} does not describe a LoRA adapter; only LoRA adapters can be loaded")
    unsupported = [option for option in UNSUPPORTED_PEFT_OPTIONS if config.get(option)]
    if config.get("bias", "none") != "none":
        unsupported.append("bias")
    if unsupported:
        raise ValueError(
            f"{path} uses PEFT options Mariana cannot apply ({', '.join(unsupported)}); "
            "merge this adapter with PEFT instead"
        )
    targets, r = config.get("target_modules"), config.get("r")
    if not isinstance(targets, list) or not targets or not isinstance(r, int) or r < 1:
        raise ValueError(f"{path} needs a positive integer r and a list of target_modules; save the adapter again")
    return config


def _resolve_stored_targets(model: nn.Module, targets: Sequence[str]) -> list[str]:
    """Map stored targets to module names: exact names (ours) or PEFT-style suffixes."""
    targets = list(dict.fromkeys(str(target) for target in targets))
    modules = dict(model.named_modules())
    if all(target in modules and _is_linear(modules[target]) for target in targets):
        return targets
    return find_target_modules(model, targets)


def load_adapter(model: nn.Module, directory: str | Path) -> list[str]:
    """Inject the adapter stored in ``directory`` into ``model``; return the wrapped names.

    Loading is strict: every stored tensor must belong to a target layer of ``model`` and
    have the right shape, otherwise a ``ValueError`` explains the mismatch and ``model`` is
    left untouched.
    """
    directory = Path(directory)
    config = _read_adapter_config(directory)
    weights_path = directory / ADAPTER_WEIGHTS_NAME
    if not weights_path.is_file():
        raise ValueError(f"{weights_path} is missing; the adapter folder is incomplete, save it again")
    load_file, _ = _import_safetensors()
    state = load_file(str(weights_path))
    _require_no_adapter(model)
    r = config["r"]
    names = _resolve_stored_targets(model, config["target_modules"])
    base_name = config.get("base_model_name_or_path") or "the model the adapter was trained on"
    expected: dict[str, tuple[int, int]] = {}
    for name in names:
        layer = model.get_submodule(name)
        rows, columns = layer.weight.shape
        in_features, out_features = (rows, columns) if _is_conv1d(layer) else (columns, rows)
        expected[f"{PEFT_KEY_PREFIX}{name}.lora_A.weight"] = (r, in_features)
        expected[f"{PEFT_KEY_PREFIX}{name}.lora_B.weight"] = (out_features, r)
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    if missing or unexpected:
        raise ValueError(
            f"The adapter in {directory} does not fit this model "
            f"(missing weights: {_preview(missing)}; unexpected weights: {_preview(unexpected)}). "
            f"Load it onto {base_name}"
        )
    for key, shape in expected.items():
        if tuple(state[key].shape) != shape:
            raise ValueError(
                f"Adapter weight {key} has shape {tuple(state[key].shape)} but this model needs "
                f"{shape}. Load the adapter onto {base_name}"
            )
    # Missing settings fall back to PEFT's own defaults.
    _wrap_modules(model, names, r, config.get("lora_alpha", 8), config.get("lora_dropout", 0.0))
    with torch.no_grad():
        for name in names:
            layer = model.get_submodule(name)
            layer.lora_A.weight.copy_(state[f"{PEFT_KEY_PREFIX}{name}.lora_A.weight"])
            layer.lora_B.weight.copy_(state[f"{PEFT_KEY_PREFIX}{name}.lora_B.weight"])
    return names


def _preview(keys: Sequence[str], limit: int = 3) -> str:
    if not keys:
        return "none"
    more = f" and {len(keys) - limit} more" if len(keys) > limit else ""
    return ", ".join(keys[:limit]) + more


def merge_adapter(model: nn.Module) -> nn.Module:
    """Replace every :class:`LoRALinear` in ``model`` by its merged base layer."""
    if isinstance(model, LoRALinear):
        return model.merged()
    wrapped = [(name, module) for name, module in model.named_modules() if isinstance(module, LoRALinear)]
    for name, module in wrapped:
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, module.merged())
    return model


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return ``(total, trainable)`` parameter counts; tied weights count once."""
    total = trainable = 0
    for parameter in model.parameters():
        total += parameter.numel()
        if parameter.requires_grad:
            trainable += parameter.numel()
    return total, trainable


def estimate_lora_parameters(shapes: Mapping[str, tuple[int, int]], r: int) -> int:
    """Count the weights LoRA adds to layers with the given ``(in, out)`` weight shapes."""
    return sum(r * (first + second) for first, second in shapes.values())
