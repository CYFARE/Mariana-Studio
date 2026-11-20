from __future__ import annotations

import json
from dataclasses import asdict, dataclass


DEFAULT_LEVELS = (1, 8, 16, 24, 32, 40, 52, 64, 76, 92, 108, 128)


def default_levels(ctx_len: int) -> tuple[int, ...]:
    if ctx_len == 128:
        return DEFAULT_LEVELS
    if ctx_len == 512:
        return (1, 4, 9, 16, 25, 36, 49, 64, 96, 128, 192, 256, 384, 512)
    count = min(12, ctx_len)
    values = {1, ctx_len}
    for index in range(1, count):
        fraction = index / max(count - 1, 1)
        values.add(1 + round((ctx_len - 1) * fraction**2))
    return tuple(sorted(values))


@dataclass
class NCMConfig:
    ctx_len: int = 128
    prefix_len: int = 64
    levels: tuple[int, ...] | None = None
    vocab_size: int = 259
    bos_token: int = 256
    pad_token: int = 257
    mask_token: int = 258

    latent_dim: int = 128
    codebook_size: int = 512
    encoder_layers: int = 2
    decoder_layers: int = 2
    refiner_layers: int = 1
    commitment: float = 0.25
    ema_decay: float = 0.99
    dead_threshold: float = 0.1
    nested_dropout: float = 0.5
    full_recon_weight: float = 0.7
    refiner_weight: float = 0.2
    all_level_vq: bool = True

    dim: int = 256
    n_heads: int = 8
    generator_layers: int = 6
    mlp_mult: int = 4
    dropout: float = 0.0
    rope_base: float = 10000.0
    level_alpha: float = 1.0
    codebook_head: str = "distance"
    corruption: float = 0.5
    self_condition: float = 0.25
    consistency_weight: float = 0.05
    prompt_dropout: float = 0.1

    def __post_init__(self):
        if self.levels is None:
            self.levels = default_levels(self.ctx_len)
        self.levels = tuple(int(v) for v in self.levels)
        if self.ctx_len <= 0 or self.prefix_len <= 0:
            raise ValueError("ctx_len and prefix_len must be positive")
        if not self.levels or self.levels[-1] != self.ctx_len:
            raise ValueError("levels must end at ctx_len")
        if self.levels[0] <= 0 or any(b <= a for a, b in zip(self.levels, self.levels[1:])):
            raise ValueError("levels must be strictly increasing and positive")
        if self.dim <= 0 or self.latent_dim <= 0 or self.codebook_size <= 1:
            raise ValueError("model dimensions must be positive and codebook_size > 1")
        if self.n_heads <= 0 or self.generator_layers <= 0 or self.mlp_mult <= 0:
            raise ValueError("n_heads, generator_layers, and mlp_mult must be positive")
        if self.rope_base <= 0:
            raise ValueError("rope_base must be positive")
        if min(self.encoder_layers, self.decoder_layers, self.refiner_layers) < 0:
            raise ValueError("tokenizer layer counts cannot be negative")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.codebook_head not in {"distance", "cosine"}:
            raise ValueError("codebook_head must be 'distance' or 'cosine'")
        if self.dim % self.n_heads:
            raise ValueError("dim must be divisible by n_heads")
        if self.latent_dim % self.n_heads:
            raise ValueError("latent_dim must be divisible by n_heads")
        if (self.dim // self.n_heads) % 2 or (self.latent_dim // self.n_heads) % 2:
            raise ValueError("attention head dimensions must be even for RoPE")
        for name in (
            "nested_dropout", "full_recon_weight", "corruption", "self_condition",
            "prompt_dropout", "level_alpha",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError("ema_decay must be in [0, 1)")
        for name in ("commitment", "dead_threshold", "refiner_weight", "consistency_weight"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if min(self.bos_token, self.pad_token, self.mask_token) < 256:
            raise ValueError("special tokens must not overlap the byte vocabulary")
        if len({self.bos_token, self.pad_token, self.mask_token}) != 3:
            raise ValueError("special tokens must be distinct")
        if max(self.bos_token, self.pad_token, self.mask_token) >= self.vocab_size:
            raise ValueError("vocab_size does not include every special token")

    @property
    def n_levels(self) -> int:
        return len(self.levels)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> "NCMConfig":
        keys = cls.__dataclass_fields__
        return cls(**{k: v for k, v in values.items() if k in keys})

    @classmethod
    def from_checkpoint(cls, state: dict) -> "NCMConfig":
        values = state["cfg"].copy()
        values.setdefault("all_level_vq", False)
        if state.get("format", 1) < 2:
            values.setdefault("codebook_head", "cosine")
        return cls.from_dict(values)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)
