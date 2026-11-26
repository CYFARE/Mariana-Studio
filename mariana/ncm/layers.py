from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.nn.attention.flex_attention import flex_attention
    HAS_FLEX = True
except Exception:
    flex_attention = None
    HAS_FLEX = False


class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        weight = self.weight
        if weight.dtype != x.dtype:
            weight = weight.to(x.dtype)
        return F.rms_norm(x, (x.shape[-1],), weight, 1e-6)


class Rotary(nn.Module):
    def __init__(self, dim: int, base: float):
        super().__init__()
        self.register_buffer(
            "inv_freq",
            1.0 / (base ** (torch.arange(0, dim, 2).float() / dim)),
            persistent=False,
        )

    def forward(self, x, positions):
        angles = positions.to(self.inv_freq.dtype).unsqueeze(-1) * self.inv_freq
        cos, sin = angles.cos().unsqueeze(1), angles.sin().unsqueeze(1)
        even, odd = x[..., 0::2], x[..., 1::2]
        out = torch.empty_like(x)
        out[..., 0::2] = even * cos - odd * sin
        out[..., 1::2] = even * sin + odd * cos
        return out


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_mult: int, dropout: float, rope_base: float,
                 residual_layers: int | None = None):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.rope = Rotary(self.head_dim, rope_base)
        hidden = mlp_mult * dim
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)
        self.drop = nn.Dropout(dropout)
        if residual_layers is not None:
            std = 0.02 / math.sqrt(2 * residual_layers)
            nn.init.normal_(self.out.weight, std=std)
            nn.init.normal_(self.down.weight, std=std)

    def forward(self, x, positions, mask=None, past_kv=None, return_kv=False,
                block_mask=None, score_mod=None):
        B, T, D = x.shape
        qkv = self.qkv(self.norm1(x)).view(B, T, 3, self.heads, self.head_dim)
        q, k, v = (qkv[:, :, i].transpose(1, 2) for i in range(3))
        q, k = self.q_norm(q), self.k_norm(k)
        q, k = self.rope(q, positions), self.rope(k, positions)
        if past_kv is not None:
            k = torch.cat((past_kv[0], k), dim=2)
            v = torch.cat((past_kv[1], v), dim=2)
        if block_mask is not None and HAS_FLEX:
            options = {} if score_mod is None else {"score_mod": score_mod}
            attended = flex_attention(q, k, v, block_mask=block_mask, **options)
        else:
            attended = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        attended = attended.transpose(1, 2).contiguous().view(B, T, D)
        x = x + self.drop(self.out(attended))
        h = self.norm2(x)
        x = x + self.drop(self.down(F.silu(self.gate(h)) * self.up(h)))
        if return_kv:
            return x, (k, v)
        return x


class ZeroConv(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, padding=1)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return x + self.conv(x.transpose(1, 2)).transpose(1, 2)


def resize(x, length: int, mode: str = "area"):
    if x.shape[1] == length:
        return x
    return F.interpolate(
        x.transpose(1, 2), size=length, mode=mode,
        align_corners=False if mode == "linear" else None,
    ).transpose(1, 2)


def sample_logits(logits, temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0):
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and non-negative")
    if top_k < 0:
        raise ValueError("top_k must be non-negative")
    if not math.isfinite(top_p) or not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be finite and in (0, 1]")
    if temperature == 0:
        return logits.argmax(-1)
    logits = logits.float() / temperature
    if top_k > 0:
        k = min(top_k, logits.shape[-1])
        threshold = logits.topk(k, dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < threshold, -torch.inf)
    if top_p < 1.0:
        sorted_logits, indices = logits.sort(dim=-1, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        remove = probs.cumsum(-1) - probs > top_p
        sorted_logits = sorted_logits.masked_fill(remove, -torch.inf)
        logits = torch.full_like(logits, -torch.inf).scatter(-1, indices, sorted_logits)
    return torch.multinomial(F.softmax(logits, dim=-1).flatten(0, -2), 1).view(logits.shape[:-1])


def mbr_select(candidates: torch.Tensor) -> torch.Tensor:
    if candidates.ndim != 3:
        raise ValueError("candidates must have shape (candidates, batch, length)")
    if candidates.shape[0] < 1:
        raise ValueError("at least one candidate is required")
    if candidates.shape[0] == 1:
        return candidates[0]
    same = (candidates[:, None] == candidates[None, :]).float().mean(-1)
    scores = (same.sum(0) - 1.0) / (candidates.shape[0] - 1)
    return candidates[scores.argmax(0), torch.arange(candidates.shape[1], device=candidates.device)]
