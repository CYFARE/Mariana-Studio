"""Mariana DRV-LM model.

Architectural upgrades over the original single-file version (see
``architecture.txt``):

* **adaLN-zero timestep conditioning** (DiT-style FiLM scale/shift/gate on
  every block) instead of a plain additive time embedding.  The modulation
  is computed *relative to the t=0 baseline*, so committed/AR tokens (time
  bucket 0) get an exact identity transform -- this keeps the AR path
  bit-exact exportable to vanilla-LLaMA formats (GGUF/ONNX).
* **Full-spectrum LLaMA-exact position RoPE** + additive time-axis rotation
  on odd pair slots (see ``rope.py``).
* **Recurrent-depth core with inference-time adaptive halting**: the shared
  core blocks loop ``max_loops`` times, exiting early when the hidden state
  converges (cheap "deep recursive thinking").  Training samples the loop
  count (loop dropout) so every depth is a valid compute point.
* **Optional sparse-MoE FFN** (top-k routing, Switch-style load balancing) with
  exact grouped CUDA AMP GEMMs and fused gate/up projections; CPU/fp32/compile
  paths retain per-expert reference kernels.
* **Multi-token-prediction head** for lookahead supervision.
* **Verifier head** scoring per-token draft acceptance (BCE-trained).
* Optional **latent thought** (``cfg.use_latent``): the legacy private discrete
  code or SAINT's adaptive discrete/continuous hierarchical memory. SAINT also
  adds zero-gated cross-attention used only by diffusion drafts; ordinary AR and
  exported models never read the private memory.
* Optional **FlexAttention** kernel path for the hybrid training mask, with
  automatic SDPA fallback.
* Fused **RMSNorm** and RoPE cosine/sine tables computed once per forward and
  shared by Q, K, and every block (explicit RMSNorm retained for ONNX export).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config
from .rope import apply_rope, build_rope_angles, precompute_rope

try:  # FlexAttention is available in torch >= 2.5
    from torch.nn.attention.flex_attention import flex_attention

    HAS_FLEX = True
except Exception:  # pragma: no cover
    flex_attention = None
    HAS_FLEX = False


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        # The native fused kernel avoids several eager reductions and temporaries.
        # Keep the explicit form for ONNX, whose opset-17 exporter cannot lower
        # aten::rms_norm, and for runtimes without a native implementation.
        if torch.onnx.is_in_onnx_export():
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)


class SwiGLU(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        hidden = cfg.mlp_mult * cfg.dim
        self.fc1 = nn.Linear(cfg.dim, hidden, bias=False)  # gate
        self.fc2 = nn.Linear(cfg.dim, hidden, bias=False)  # up
        self.fc3 = nn.Linear(hidden, cfg.dim, bias=False)  # down

    def forward(self, x):
        return self.fc3(F.silu(self.fc1(x)) * self.fc2(x))


class LatentCrossAttention(nn.Module):
    """Draft-only attention from byte tokens into the private thought memory."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.nh = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.norm = RMSNorm(cfg.dim)
        self.q = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.k = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.v = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.o = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.gate = nn.Parameter(torch.zeros(cfg.dim))

    def forward(self, x, memory, memory_mask, draft_mask):
        B, T, D = x.shape
        M = memory.shape[1]
        h = self.norm(x)
        q = self.q(h).view(B, T, self.nh, self.head_dim).transpose(1, 2)
        k = self.k(memory).view(B, M, self.nh, self.head_dim).transpose(1, 2)
        v = self.v(memory).view(B, M, self.nh, self.head_dim).transpose(1, 2)
        allow = memory_mask[:, None, None, :].expand(B, 1, T, M)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=allow)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return x + self.gate * self.o(out) * draft_mask.unsqueeze(-1)


class LatentRefinerBlock(nn.Module):
    """A residual self-attention/FFN pass over the continuous thought slots."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.norm1 = RMSNorm(cfg.dim)
        self.attn = nn.MultiheadAttention(
            cfg.dim, cfg.n_heads, dropout=cfg.dropout, batch_first=True
        )
        self.norm2 = RMSNorm(cfg.dim)
        self.mlp = SwiGLU(cfg)
        self.gate1 = nn.Parameter(torch.zeros(cfg.dim))
        self.gate2 = nn.Parameter(torch.zeros(cfg.dim))

    def forward(self, x, valid):
        h = self.norm1(x)
        attended, _ = self.attn(
            h, h, h, key_padding_mask=~valid, need_weights=False
        )
        x = x + self.gate1 * attended
        x = x + self.gate2 * self.mlp(self.norm2(x))
        return x * valid.unsqueeze(-1)


def _grouped_mm_forward(ctx, x, weight, offsets, sizes):
    ctx.save_for_backward(x, weight, offsets)
    ctx.sizes = sizes
    return torch._grouped_mm(x, weight, offs=offsets)


def _grouped_mm_backward(ctx, grad):
    x, weight, offsets = ctx.saved_tensors
    grad_x = torch._grouped_mm(grad, weight.transpose(-2, -1), offs=offsets)
    grad_weight = torch.zeros_like(weight)
    start = 0
    for expert, size in enumerate(ctx.sizes):
        end = start + size
        if size:
            grad_weight[expert] = x[start:end].T @ grad[start:end]
        start = end
    return grad_x, grad_weight, None, None


class GroupedMMBF16(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.bfloat16)
    def forward(ctx, x, weight, offsets, sizes):
        return _grouped_mm_forward(ctx, x, weight, offsets, sizes)

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad):
        return _grouped_mm_backward(ctx, grad)


class GroupedMMFP16(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float16)
    def forward(ctx, x, weight, offsets, sizes):
        return _grouped_mm_forward(ctx, x, weight, offsets, sizes)

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad):
        return _grouped_mm_backward(ctx, grad)


class SparseMoEFFN(nn.Module):
    """Top-k routed SwiGLU experts with Switch-style load-balancing loss."""

    def __init__(self, cfg: Config):
        super().__init__()
        E, dim = cfg.moe_experts, cfg.dim
        hidden = cfg.mlp_mult * cfg.dim
        self.k = cfg.moe_topk
        self.n_experts = E
        self.router = nn.Linear(dim, E, bias=False)
        self.gate = nn.Parameter(torch.empty(E, dim, hidden))
        self.up = nn.Parameter(torch.empty(E, dim, hidden))
        self.down = nn.Parameter(torch.empty(E, hidden, dim))
        for p in (self.gate, self.up, self.down):
            nn.init.normal_(p, std=0.02)
        self.grouped = getattr(cfg, "grouped_moe", True) and hasattr(torch, "_grouped_mm")
        self._gate_up_cache = None

    def _fused_gate_up(self):
        version = (self.gate._version, self.up._version)
        if self._gate_up_cache is None or self._gate_up_cache[0] != version:
            self._gate_up_cache = (version, torch.cat((self.gate, self.up), dim=2))
        return self._gate_up_cache[1]

    def _load_from_state_dict(self, *args, **kwargs):
        self._gate_up_cache = None
        super()._load_from_state_dict(*args, **kwargs)

    def forward(self, x):
        B, T, D = x.shape
        flat = x.reshape(-1, D)
        logits = self.router(flat)
        probs = F.softmax(logits, dim=-1)
        top_w, top_i = probs.topk(self.k, dim=-1)
        top_w = top_w / top_w.sum(-1, keepdim=True).clamp_min(1e-9)
        counts = torch.bincount(top_i.reshape(-1), minlength=self.n_experts)

        use_grouped = (
            self.grouped and x.is_cuda and torch.is_autocast_enabled("cuda")
            and not torch._dynamo.is_compiling()
        )
        if use_grouped:
            dtype = torch.get_autocast_dtype("cuda")
            grouped_mm = (
                GroupedMMBF16 if dtype == torch.bfloat16 else GroupedMMFP16
            )
            sizes = tuple(counts.tolist())
            offsets = counts.cumsum(0, dtype=torch.int32)
            order = torch.argsort(top_i.reshape(-1))
            sorted_x = flat[order // self.k]
            gate_up = grouped_mm.apply(
                sorted_x, self._fused_gate_up(), offsets, sizes
            )
            gate, up = gate_up.split(self.gate.shape[2], dim=-1)
            hidden = F.silu(gate) * up
            sorted_out = grouped_mm.apply(hidden, self.down, offsets, sizes)
            sorted_weights = top_w.reshape(-1)[order]
            out = torch.zeros_like(flat)
            out.index_add_(
                0, order // self.k, sorted_out * sorted_weights.unsqueeze(-1)
            )
        else:
            out = torch.zeros_like(flat)
            for e in range(self.n_experts):
                sel = top_i == e
                idx, slot = sel.nonzero(as_tuple=True)
                if idx.numel() == 0:
                    continue
                xe = flat[idx]
                he = F.silu(xe @ self.gate[e]) * (xe @ self.up[e])
                ye = he @ self.down[e]
                out.index_add_(0, idx, ye * top_w[idx, slot].unsqueeze(-1))

        f = counts.float() / counts.sum().clamp_min(1.0)
        lb = self.n_experts * (f * probs.mean(0)).sum()
        return out.view(B, T, D), lb


# ---------------------------------------------------------------------------
# Transformer block with adaLN-zero conditioning
# ---------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.head_dim = cfg.head_dim
        self.nh = cfg.n_heads
        self.qkv = nn.Linear(cfg.dim, 3 * cfg.dim, bias=False)
        self.o = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.ln1 = RMSNorm(cfg.dim)
        self.ln2 = RMSNorm(cfg.dim)
        if cfg.use_moe:
            self.mlp = SparseMoEFFN(cfg)
        else:
            self.mlp = SwiGLU(cfg)
        self.dropout = nn.Dropout(cfg.dropout)
        if cfg.use_latent and cfg.latent_saint and cfg.latent_diffusion:
            self.latent_attn = LatentCrossAttention(cfg)
        if cfg.use_adaln:
            # produces (scale1, shift1, gate1, scale2, shift2, gate2) deltas
            self.adaln = nn.Linear(cfg.dim, 6 * cfg.dim, bias=True)
            nn.init.zeros_(self.adaln.weight)
            nn.init.zeros_(self.adaln.bias)

    def _modulate(self, h, scale, shift):
        return h * (1.0 + scale) + shift

    def forward(
        self,
        x,
        cond,                 # (B, T, dim) adaLN delta (0 at time bucket 0)
        angles_pos,
        angles_time,
        times,
        mask=None,            # bool (B,1,T,S), True = attend
        kv_cache=None,
        return_kv=False,
        block_mask=None,      # FlexAttention BlockMask (full-seq training only)
        rope=None,            # shared (cos, sin), computed once by MarianaLM
        latent_memory=None,
        latent_memory_mask=None,
        draft_mask=None,
    ):
        B, T, _ = x.shape
        lb = None

        if self.cfg.use_adaln:
            # ``cond`` is already relative to the t=0 baseline.  Subtract the
            # linear bias as well: otherwise a trained bias would make the
            # modulation non-identity at t=0 and break lossless AR export.
            mod = self.adaln(cond)
            if self.adaln.bias is not None:
                mod = mod - self.adaln.bias
            s1, b1, g1, s2, b2, g2 = mod.chunk(6, dim=-1)
        else:
            s1 = b1 = g1 = s2 = b2 = g2 = 0.0

        residual = x
        h = self._modulate(self.ln1(x), s1, b1) if self.cfg.use_adaln else self.ln1(x)

        qkv = self.qkv(h).view(B, T, 3, self.nh, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # absolute positions: new tokens start after the cached prefix
        off = kv_cache[0].shape[2] if kv_cache is not None else 0
        ap = angles_pos[off : off + T]

        if rope is None:
            rope = precompute_rope(ap, angles_time, times)
        q = apply_rope(q, ap, angles_time, times, rope)
        k = apply_rope(k, ap, angles_time, times, rope)

        if kv_cache is not None:
            ck, cv = kv_cache
            k = torch.cat([ck, k], dim=2)
            v = torch.cat([cv, v], dim=2)

        if block_mask is not None and HAS_FLEX:
            out = flex_attention(q, k, v, block_mask=block_mask)
        else:
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False
            )
        out = out.transpose(1, 2).contiguous().view(B, T, self.cfg.dim)
        out = self.o(out)
        x = residual + (1.0 + g1) * self.dropout(out) if self.cfg.use_adaln else residual + self.dropout(out)

        if latent_memory is not None:
            if not hasattr(self, "latent_attn"):
                raise ValueError("latent memory requires a SAINT diffusion model")
            if latent_memory_mask is None or draft_mask is None:
                raise ValueError("latent memory mask and draft mask are required")
            x = self.latent_attn(x, latent_memory, latent_memory_mask, draft_mask)

        residual = x
        h = self._modulate(self.ln2(x), s2, b2) if self.cfg.use_adaln else self.ln2(x)
        if isinstance(self.mlp, SparseMoEFFN):
            mlp_out, lb = self.mlp(h)
        else:
            mlp_out = self.mlp(h)
        x = residual + (1.0 + g2) * self.dropout(mlp_out) if self.cfg.use_adaln else residual + self.dropout(mlp_out)

        if return_kv:
            return x, (k, v), lb
        return x, lb


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------
class MarianaLM(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.time_emb = nn.Embedding(cfg.n_time_buckets, cfg.dim)
        self.region_emb = nn.Embedding(2, cfg.dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(cfg.dim, cfg.dim), nn.SiLU(), nn.Linear(cfg.dim, cfg.dim)
        )

        self.pre = nn.ModuleList([Block(cfg) for _ in range(cfg.n_pre)])
        self.core = nn.ModuleList([Block(cfg) for _ in range(cfg.n_core)])
        self.post = nn.ModuleList([Block(cfg) for _ in range(cfg.n_post)])

        self.ln = RMSNorm(cfg.dim)
        self.verifier = nn.Linear(cfg.dim, 1, bias=False)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # tied embeddings
        self.mtp_head = (
            nn.Linear(cfg.dim, cfg.vocab_size, bias=False) if cfg.use_mtp else None
        )

        # --- private latent alphabet ------------------------------------------
        # These parameters are deliberately outside the byte vocabulary: the
        # symbols can only be emitted into the thought channel, never as text,
        # and the exported AR path never touches them.
        if cfg.use_latent:
            self.thought_bos = nn.Parameter(torch.empty(cfg.dim))
            self.thought_pos = nn.Embedding(cfg.n_thoughts, cfg.dim)
            if cfg.latent_saint:
                self.n_latent_parents = (
                    cfg.n_thoughts + cfg.latent_block_size - 1
                ) // cfg.latent_block_size
                self.codebook = nn.Embedding(cfg.n_latent_symbols + 2, cfg.dim)
                self.latent_head = nn.Linear(
                    cfg.dim, cfg.n_latent_symbols + 1, bias=False
                )
                self.continuous_head = nn.Linear(
                    cfg.dim, 2 * cfg.latent_continuous_slots * cfg.dim
                )
                self.continuous_pos = nn.Embedding(
                    cfg.n_thoughts * cfg.latent_continuous_slots, cfg.dim
                )
                self.continuous_norm = RMSNorm(cfg.dim)
                self.latent_refiners = nn.ModuleList(
                    LatentRefinerBlock(cfg) for _ in range(cfg.latent_refine_steps)
                )
                self.parent_proj = nn.Linear(2 * cfg.dim, cfg.dim)
                self.parent_pos = nn.Embedding(self.n_latent_parents, cfg.dim)
                self.parent_codebook = nn.Embedding(cfg.n_latent_symbols, cfg.dim)
                self.parent_head = nn.Linear(cfg.dim, cfg.n_latent_symbols, bias=False)
                self.parent_readout = nn.Linear(cfg.dim, cfg.n_latent_symbols, bias=False)
                self.grammar_pair = nn.Linear(3 * cfg.dim, cfg.dim)
                self.grammar_head = nn.Linear(
                    cfg.dim, cfg.n_latent_symbols + 1, bias=False
                )
                self.order_head = nn.Linear(3 * cfg.dim, 2)
                self.latent_value = nn.Sequential(
                    nn.Linear(cfg.dim, cfg.dim), nn.SiLU(), nn.Linear(cfg.dim, 1)
                )
                self.latent_type = nn.Embedding(3, cfg.dim)
            else:
                self.codebook = nn.Embedding(cfg.n_latent_symbols, cfg.dim)
                self.latent_head = nn.Linear(cfg.dim, cfg.n_latent_symbols, bias=False)

        head_dim = cfg.head_dim
        position_len = cfg.ctx_len * 2
        if cfg.use_latent and cfg.latent_saint:
            memory_len = (
                cfg.n_thoughts * (1 + cfg.latent_continuous_slots)
                + self.n_latent_parents
            )
            position_len = max(
                position_len, memory_len + cfg.ctx_len - cfg.n_thoughts
            )
        ap, at = build_rope_angles(
            head_dim, position_len, cfg.n_time_buckets,
            cfg.rope_base_pos, cfg.rope_base_time,
        )
        self.register_buffer("angles_pos", ap, persistent=False)
        self.register_buffer("angles_time", at, persistent=False)

        self.apply(self._init_weights)
        self._rescale_residual_init()
        if cfg.use_latent:
            nn.init.normal_(self.thought_bos, std=0.02)

    # -- init -----------------------------------------------------------------
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        # re-assert adaLN-zero after generic init
        for blk in list(self.pre) + list(self.core) + list(self.post):
            if self.cfg.use_adaln:
                nn.init.zeros_(blk.adaln.weight)
                nn.init.zeros_(blk.adaln.bias)

    def _rescale_residual_init(self):
        # GPT-2 style: shrink residual-path output projections by depth
        scale = 0.02 / (2 * self.cfg.n_layers) ** 0.5
        for blk in list(self.pre) + list(self.core) + list(self.post):
            nn.init.normal_(blk.o.weight, std=scale)
            if isinstance(blk.mlp, SwiGLU):
                nn.init.normal_(blk.mlp.fc3.weight, std=scale)
            else:
                nn.init.normal_(blk.mlp.down, std=scale)

    # -- conditioning -----------------------------------------------------------
    def _cond_delta(self, times):
        """adaLN input relative to the t=0 baseline (exact identity at t=0)."""
        emb = self.time_emb(times.clamp(0, self.cfg.n_time_buckets - 1))
        base = self.time_emb.weight[0]
        return self.time_mlp(emb) - self.time_mlp(base)

    # -- forward ------------------------------------------------------------------
    def blocks_for_loops(self, n_loops: int):
        return list(self.pre) + list(self.core) * n_loops + list(self.post)

    def forward(
        self,
        tokens=None,
        times=None,
        region=None,
        mask=None,
        kv_cache=None,
        return_kv=False,
        n_loops=None,
        block_mask=None,
        halt_tol: float = 0.0,
        need_hidden: bool = False,
        input_embeds=None,
        latent_memory=None,
        latent_memory_mask=None,
    ):
        """Returns ``(logits, verify, extras)``.

        Pass either integer ``tokens`` or pre-embedded ``input_embeds``.  The
        latter is how latent symbols enter the model: they are vectors from the
        private codebook rather than rows of the byte embedding table.

        ``extras`` may contain ``kv`` (list of (k, v)), ``hidden`` (final
        normalised hidden states) and ``moe_lb`` (summed load-balancing loss).
        """
        if n_loops is None:
            n_loops = self.cfg.max_loops
        if input_embeds is not None and tokens is not None:
            raise ValueError("pass either tokens or input_embeds, not both")
        if input_embeds is None and tokens is None:
            raise ValueError("tokens or input_embeds is required")
        if times is None or region is None:
            raise ValueError("times and region are required")

        # Embeddings are defined *relative* to the t=0 / region=0 baseline, so
        # the committed/AR path uses exactly tok_emb (+ nothing else) and the
        # tied LM head is exact.  This is what makes the AR path losslessly
        # exportable as a vanilla LLaMA-architecture model.
        base = self.tok_emb(tokens) if tokens is not None else input_embeds
        x = base + (
            self.time_emb(times.clamp(0, self.cfg.n_time_buckets - 1))
            - self.time_emb.weight[0]
        ) + (self.region_emb(region) - self.region_emb.weight[0])
        cond = self._cond_delta(times) if self.cfg.use_adaln else None

        extras: dict = {}
        new_cache = [] if return_kv else None
        moe_lb = 0.0
        layer = 0
        off = kv_cache[0][0].shape[2] if kv_cache is not None and len(kv_cache) else 0
        required_positions = off + x.shape[1]
        angles_pos = self.angles_pos
        if required_positions > angles_pos.shape[0]:
            angles_pos, _ = build_rope_angles(
                self.cfg.head_dim, required_positions, self.cfg.n_time_buckets,
                self.cfg.rope_base_pos, self.cfg.rope_base_time,
            )
            angles_pos = angles_pos.to(device=x.device, dtype=self.angles_pos.dtype)
        rope = precompute_rope(
            angles_pos[off:required_positions], self.angles_time, times
        )

        def run_block(block, x, cache):
            nonlocal moe_lb, layer
            if return_kv:
                x, kv, lb = block(
                    x, cond, angles_pos, self.angles_time, times,
                    mask, cache, True, block_mask, rope,
                    latent_memory, latent_memory_mask, region.bool(),
                )
                new_cache.append(kv)
            else:
                x, lb = block(
                    x, cond, angles_pos, self.angles_time, times,
                    mask, cache, False, block_mask, rope,
                    latent_memory, latent_memory_mask, region.bool(),
                )
            if lb is not None:
                moe_lb = moe_lb + lb
            layer += 1
            return x

        def cache_for(_layer):
            return kv_cache[_layer] if kv_cache is not None else None

        for blk in self.pre:
            x = run_block(blk, x, cache_for(layer))

        # Never halt while a KV cache is involved: the cache length must
        # always match the canonical depth (pre + core*max_loops + post).
        can_halt = (
            halt_tol > 0.0 and kv_cache is None and not return_kv and not self.training
        )
        x_prev = None
        for _ in range(n_loops):
            for blk in self.core:
                x = run_block(blk, x, cache_for(layer))
            if can_halt:
                if x_prev is not None and (x - x_prev).abs().max().item() < halt_tol:
                    break
                x_prev = x

        for blk in self.post:
            x = run_block(blk, x, cache_for(layer))

        h = self.ln(x)
        logits = self.lm_head(h)
        verify = self.verifier(h).squeeze(-1)
        if return_kv:
            extras["kv"] = new_cache
        if need_hidden:
            extras["hidden"] = h
        if isinstance(moe_lb, torch.Tensor):
            extras["moe_lb"] = moe_lb
        return logits, verify, extras
