"""The DRV-LM model: forward shapes, KV-cache exactness, recurrent depth."""

from __future__ import annotations

import torch

from mariana.config import Config
from mariana.model import MarianaLM
from mariana.train import make_block_mask


def tiny_cfg() -> Config:
    return Config(
        ctx_len=32, dim=64, n_heads=4, n_pre=1, n_core=1, n_post=1,
        max_loops=2, block_size=16, mlp_mult=2,
    )


def test_forward_shapes():
    cfg = tiny_cfg()
    model = MarianaLM(cfg)
    x = torch.randint(0, 256, (2, cfg.ctx_len))
    times = torch.zeros(2, cfg.ctx_len, dtype=torch.long)
    region = torch.zeros(2, cfg.ctx_len, dtype=torch.long)
    mask = make_block_mask(cfg.ctx_len, 16, cfg.block_size).unsqueeze(0).unsqueeze(0)
    logits, verify, extras = model(x, times, region, mask, need_hidden=True)
    assert logits.shape == (2, cfg.ctx_len, cfg.vocab_size)
    assert verify.shape == (2, cfg.ctx_len)
    assert "hidden" in extras


def test_kv_cache_consistency():
    """Cached decode must match a full forward pass (AR path exactness)."""
    cfg = tiny_cfg()
    model = MarianaLM(cfg).eval()
    T = 12
    x = torch.randint(0, 256, (1, T))
    zeros = torch.zeros(1, T, dtype=torch.long)
    causal = torch.tril(torch.ones(T, T)).bool().unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        full_logits, _, _ = model(x, zeros, zeros, causal)
        # prefill first half, decode rest token by token
        t0 = torch.zeros(1, 6, dtype=torch.long)
        m0 = torch.tril(torch.ones(6, 6)).bool().unsqueeze(0).unsqueeze(0)
        logits, _, ex = model(x[:, :6], t0, t0, m0, return_kv=True)
        kv = ex["kv"]
        outs = [logits[:, -1]]
        for i in range(6, T):
            tok = x[:, i : i + 1]
            t1 = torch.zeros(1, 1, dtype=torch.long)
            m1 = torch.ones(1, 1, 1, i + 1, dtype=torch.bool)
            logits, _, ex = model(tok, t1, t1, m1, kv_cache=kv, return_kv=True)
            kv = ex["kv"]
            outs.append(logits[:, -1])
        cached = torch.stack(outs, dim=1)  # logits at positions 5..T-1
    torch.testing.assert_close(cached, full_logits[:, 5:T], atol=1e-4, rtol=1e-4)


def test_loop_dropout_and_halting():
    cfg = tiny_cfg()
    model = MarianaLM(cfg).eval()
    x = torch.randint(0, 256, (1, cfg.ctx_len))
    zeros = torch.zeros(1, cfg.ctx_len, dtype=torch.long)
    causal = torch.tril(torch.ones(cfg.ctx_len, cfg.ctx_len)).bool().unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        for n in (1, 2):
            logits, _, _ = model(x, zeros, zeros, causal, n_loops=n)
            assert logits.shape[-1] == cfg.vocab_size
        logits, _, _ = model(x, zeros, zeros, causal, halt_tol=1e-3)
        assert torch.isfinite(logits).all()
