"""Generation: Draft-Refine-Verify and the autoregressive baseline."""

from __future__ import annotations

import torch

from mariana.config import Config
from mariana.generate import ar_generate, draft_refine_verify
from mariana.model import MarianaLM


def tiny_cfg() -> Config:
    return Config(
        ctx_len=32, dim=64, n_heads=4, n_pre=1, n_core=1, n_post=1,
        max_loops=2, block_size=16, mlp_mult=2,
    )


def test_generation_runs():
    cfg = tiny_cfg()
    model = MarianaLM(cfg).eval()
    out, acc, tps, _ = draft_refine_verify(
        model, b"HELLO: ", max_new_tokens=16, n_steps=2,
        device=torch.device("cpu"), cfg=cfg,
    )
    assert len(out) > 0
    out_ar, _, _ = ar_generate(model, b"HELLO: ", max_new_tokens=8,
                               device=torch.device("cpu"), cfg=cfg)
    assert len(out_ar) > 0
