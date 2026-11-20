"""Export: GGUF, safetensors and exact LLaMA-conversion parity."""

from __future__ import annotations

import os
import tempfile

import torch

from mariana.config import Config
from mariana.model import MarianaLM
from mariana.train import corrupt_batch, hybrid_losses


def tiny_cfg() -> Config:
    return Config(
        ctx_len=32, dim=64, n_heads=4, n_pre=1, n_core=1, n_post=1,
        max_loops=2, block_size=16, mlp_mult=2,
    )


def test_export_gguf_and_safetensors():
    from mariana.export.gguf_export import export_gguf
    from mariana.export.safetensors import export_safetensors

    cfg = tiny_cfg()
    model = MarianaLM(cfg)
    sd = model.state_dict()
    with tempfile.TemporaryDirectory() as d:
        gguf_path = export_gguf(sd, cfg, os.path.join(d, "m.gguf"), dtype="f32")
        assert os.path.getsize(gguf_path) > 1000
        with open(gguf_path, "rb") as f:
            assert f.read(4) == b"GGUF"
        assert os.path.exists(os.path.join(d, "Modelfile"))

        st_path = export_safetensors(sd, cfg, os.path.join(d, "m.safetensors"))
        from safetensors import safe_open

        with safe_open(st_path, framework="pt") as f:
            keys = list(f.keys())
        assert any(k.startswith("blk.") for k in keys)
        assert "token_embd.weight" in keys
        n_layers = cfg.n_pre + cfg.n_core * cfg.max_loops + cfg.n_post
        assert f"blk.{n_layers - 1}.attn_q.weight" in keys


def test_llama_compat_parity():
    """Converted LLaMA-format weights must reproduce the AR forward exactly."""
    from mariana.export.compat import to_llama_state

    cfg = tiny_cfg()
    model = MarianaLM(cfg)

    # Train briefly so the adaLN bias is nonzero.  The export identity must
    # hold after training, not only at initialization.
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3)
    train_x = torch.randint(0, 256, (4, cfg.ctx_len + 1))
    train_mask = torch.tril(torch.ones(cfg.ctx_len, cfg.ctx_len)).bool().unsqueeze(0).unsqueeze(0)
    for _ in range(5):
        inputs, targets, originals, times, region, attn_mask, _ = corrupt_batch(train_x, cfg)
        losses = hybrid_losses(model, inputs, targets, originals, times, region, attn_mask,
                               None, cfg, train_mask, cfg.max_loops)
        opt.zero_grad()
        losses["total"].backward()
        opt.step()
    # Simulate a checkpoint with a nonzero adaLN bias.  The export identity
    # must hold when the bias is present, because the block subtracts it.
    with torch.no_grad():
        for blk in list(model.pre) + list(model.core) + list(model.post):
            if blk.adaln.bias is not None:
                blk.adaln.bias.normal_(0.0, 0.05)
    model.eval()

    sd = model.state_dict()
    llama_sd = to_llama_state(sd, cfg)

    T = 16
    x = torch.randint(0, 256, (1, T))
    zeros = torch.zeros(1, T, dtype=torch.long)
    causal = torch.tril(torch.ones(T, T)).bool().unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        ref, _, _ = model(x, zeros, zeros, causal)

    # manual LLaMA-architecture forward using the converted weights
    import torch.nn.functional as F

    def rms(h, w, eps=1e-6):
        return h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps) * w

    h = llama_sd["token_embd.weight"][x]
    half = cfg.head_dim // 2
    pos = torch.arange(T, dtype=torch.float32)
    freqs = 1.0 / (cfg.rope_base_pos ** (torch.arange(half).float() / half))
    ang = torch.outer(pos, freqs)  # (T, half) NeoX layout
    cos, sin = torch.cos(ang)[None, None], torch.sin(ang)[None, None]

    def rope(t):  # (1, nh, T, hd) half-split rotation
        t1, t2 = t[..., :half], t[..., half:]
        return torch.cat([t1 * cos - t2 * sin, t1 * sin + t2 * cos], dim=-1)

    n_layers = cfg.n_layers
    for l in range(n_layers):
        p = f"blk.{l}."
        r = h
        hn = rms(h, llama_sd[p + "attn_norm.weight"])
        q = (hn @ llama_sd[p + "attn_q.weight"].T).view(1, T, cfg.n_heads, cfg.head_dim).transpose(1, 2)
        k = (hn @ llama_sd[p + "attn_k.weight"].T).view(1, T, cfg.n_heads, cfg.head_dim).transpose(1, 2)
        v = (hn @ llama_sd[p + "attn_v.weight"].T).view(1, T, cfg.n_heads, cfg.head_dim).transpose(1, 2)
        q, k = rope(q), rope(k)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        o = o.transpose(1, 2).reshape(1, T, cfg.dim) @ llama_sd[p + "attn_output.weight"].T
        h = r + o
        r = h
        hn = rms(h, llama_sd[p + "ffn_norm.weight"])
        h = r + (F.silu(hn @ llama_sd[p + "ffn_gate.weight"].T)
                 * (hn @ llama_sd[p + "ffn_up.weight"].T)) @ llama_sd[p + "ffn_down.weight"].T
    logits = rms(h, llama_sd["output_norm.weight"]) @ llama_sd["output.weight"].T
    torch.testing.assert_close(logits, ref, atol=1e-3, rtol=1e-3)
