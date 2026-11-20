"""Training objectives: legacy two-pass losses and Sprint-DRV single-pass training."""

from __future__ import annotations

import os
import tempfile

import pytest
import torch

from mariana.cli import cfg_from_args, parse_args
from mariana.config import Config
from mariana.data import build_dataset
from mariana.model import MarianaLM
from mariana.train import (
    corrupt_batch,
    hybrid_losses,
    make_block_mask,
    make_causal_mask,
    train as train_model,
)


def tiny_cfg(fast: bool = True) -> Config:
    return Config(
        ctx_len=32,
        dim=64,
        n_heads=4,
        n_pre=1,
        n_core=1,
        n_post=1,
        max_loops=2,
        block_size=8,
        mlp_mult=2,
        fast_train=fast,
    )


class CountingLM(MarianaLM):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.forward_calls = 0

    def forward(self, *args, **kwargs):
        self.forward_calls += 1
        return super().forward(*args, **kwargs)


def make_batch(model, cfg):
    x = torch.randint(0, 256, (3, cfg.ctx_len + 1))
    batch = corrupt_batch(x, cfg)
    causal = make_causal_mask(cfg.ctx_len).unsqueeze(0).unsqueeze(0)
    return x, batch, causal


def test_fast_uses_one_forward_and_backpropagates():
    cfg = tiny_cfg(fast=True)
    model = CountingLM(cfg)
    _, batch, causal = make_batch(model, cfg)
    inputs, targets, originals, times, region, mask, flex = batch
    losses = hybrid_losses(
        model, inputs, targets, originals, times, region, mask, flex,
        cfg, causal, cfg.max_loops,
    )
    assert model.forward_calls == 1
    assert float(losses["align"].detach()) == 0.0
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    assert model.tok_emb.weight.grad is not None


def test_fixed_shape_losses_match_selected_reference():
    cfg = tiny_cfg(fast=False)  # guarantee a non-empty draft for the reference
    model = MarianaLM(cfg).eval()
    _, batch, causal = make_batch(model, cfg)
    cfg.fast_train = True
    inputs, targets, originals, times, region, mask, flex = batch
    losses = hybrid_losses(
        model, inputs, targets, originals, times, region, mask, flex,
        cfg, causal, cfg.max_loops,
    )
    with torch.no_grad():
        logits, verify, extras = model(
            inputs, times, region, mask, n_loops=cfg.max_loops, need_hidden=True
        )
        ar = region == 0
        diff = ~ar
        expected = {
            "ar": torch.nn.functional.cross_entropy(logits[ar], targets[ar]),
            "diff": torch.nn.functional.cross_entropy(logits[diff], originals[diff]),
            "ver": torch.nn.functional.binary_cross_entropy_with_logits(
                verify[diff], (logits[diff].argmax(-1) == originals[diff]).float()
            ),
            "mtp": torch.nn.functional.cross_entropy(
                model.mtp_head(extras["hidden"][:, :-2][ar[:, :-2]]),
                originals[:, 2:][ar[:, :-2]],
            ),
        }
    for key, value in expected.items():
        torch.testing.assert_close(losses[key], value, atol=1e-6, rtol=1e-6)


def test_legacy_keeps_two_forward_passes():
    cfg = tiny_cfg(fast=False)
    model = CountingLM(cfg)
    _, batch, causal = make_batch(model, cfg)
    inputs, targets, originals, times, region, mask, flex = batch
    losses = hybrid_losses(
        model, inputs, targets, originals, times, region, mask, flex,
        cfg, causal, cfg.max_loops,
    )
    assert model.forward_calls == 2
    assert torch.isfinite(losses["align"])


def test_fast_prefix_ar_is_exact_clean_ar():
    cfg = tiny_cfg(fast=True)
    model = MarianaLM(cfg).eval()
    _, batch, causal = make_batch(model, cfg)
    inputs, _, originals, times, region, mask, _ = batch
    zeros = torch.zeros_like(region)
    with torch.no_grad():
        hybrid, _, _ = model(inputs, times, region, mask, n_loops=cfg.max_loops)
        clean, _, _ = model(originals, zeros, zeros, causal, n_loops=cfg.max_loops)
    prefix = region == 0
    torch.testing.assert_close(hybrid[prefix], clean[prefix], atol=1e-5, rtol=1e-5)


def test_fast_fully_clean_batch_has_zero_denoising_losses():
    cfg = tiny_cfg(fast=True)
    model = MarianaLM(cfg)
    x = torch.randint(0, 256, (2, cfg.ctx_len + 1))
    originals = x[:, :-1]
    targets = x[:, 1:]
    times = torch.zeros_like(originals)
    region = torch.zeros_like(originals)
    mask = make_causal_mask(cfg.ctx_len).unsqueeze(0).unsqueeze(0)
    losses = hybrid_losses(
        model, originals, targets, originals, times, region, mask, None,
        cfg, mask, cfg.max_loops,
    )
    assert float(losses["diff"].detach()) == 0.0
    assert float(losses["ver"].detach()) == 0.0
    assert torch.isfinite(losses["total"])
    losses["total"].backward()


def test_vectorized_corruption_and_masks():
    cfg = tiny_cfg(fast=False)
    x = torch.randint(0, 256, (4, cfg.ctx_len + 1))
    before = x.clone()
    inputs, _, originals, times, region, masks, flex = corrupt_batch(x, cfg)
    assert torch.equal(x, before), "corruption must not mutate the dataset batch"
    assert flex is None
    assert inputs.shape == originals.shape == (4, cfg.ctx_len)
    assert (inputs[:, 0] == cfg.bos_token).all()
    assert (times[region == 0] == 0).all()
    assert (times[region == 1] > 0).all()
    for b in range(x.shape[0]):
        cutoff = int(region[b].argmax())
        expected = make_block_mask(cfg.ctx_len, cutoff, cfg.block_size)
        assert torch.equal(masks[b, 0], expected)
        # A single noise level is shared by each relative draft block.
        draft_times = times[b, cutoff:]
        for start in range(0, len(draft_times), cfg.block_size):
            block = draft_times[start : start + cfg.block_size]
            assert (block == block[0]).all()


def test_fast_cli_and_checkpoint_default():
    args = parse_args(["train", "--fast"])
    assert cfg_from_args(args).fast_train
    args = parse_args(["train", "--legacy"])
    assert not cfg_from_args(args).fast_train
    assert not Config.from_dict({}).fast_train


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA grouped-MM test")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_grouped_moe_matches_reference(dtype):
    from mariana.model import SparseMoEFFN

    cfg = tiny_cfg()
    cfg.dim = 64
    cfg.mlp_mult = 2
    cfg.use_moe = True
    cfg.moe_experts = 4
    cfg.moe_topk = 2
    cfg.grouped_moe = False
    reference = SparseMoEFFN(cfg).cuda()
    cfg.grouped_moe = True
    grouped = SparseMoEFFN(cfg).cuda()
    grouped.load_state_dict(reference.state_dict())
    x = torch.randn(4, 16, 64, device="cuda", requires_grad=True)
    y = x.detach().clone().requires_grad_(True)
    with torch.autocast("cuda", dtype=dtype):
        expected, expected_lb = reference(x)
        actual, actual_lb = grouped(y)
    expected.sum().backward()
    actual.sum().backward()
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(actual_lb, expected_lb, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(y.grad, x.grad, atol=2e-3, rtol=2e-3)
    for name, parameter in reference.named_parameters():
        actual_parameter = dict(grouped.named_parameters())[name]
        torch.testing.assert_close(
            actual_parameter.grad, parameter.grad, atol=2e-3, rtol=2e-3
        )

    optimizer = torch.optim.AdamW(grouped.parameters(), lr=1e-3)
    previous_version = grouped._gate_up_cache[0]
    optimizer.step()
    assert grouped._fused_gate_up() is not None
    assert grouped._gate_up_cache[0] != previous_version


def test_grouped_moe_cli_fallback():
    args = parse_args(["train", "--moe", "--no-grouped-moe"])
    assert not cfg_from_args(args).grouped_moe
    assert cfg_from_args(parse_args(["train", "--moe"])).grouped_moe


def test_small_corpus_validation_is_source_disjoint():
    train, val = build_dataset(bytes(range(100)), ctx=9)
    train_rows = {tuple(row.tolist()) for row in train}
    assert train.shape[0] > 0 and val.shape[0] > 0
    assert not any(tuple(row.tolist()) in train_rows for row in val)
    with pytest.raises(ValueError, match="disjoint train/validation split"):
        build_dataset(b"abc", ctx=9)


def test_oversized_batch_has_actionable_loader_error():
    with tempfile.TemporaryDirectory() as directory:
        corpus = os.path.join(directory, "corpus.txt")
        with open(corpus, "wb") as handle:
            handle.write(bytes(range(100)))
        args = parse_args([
            "train", "--data", corpus, "--checkpoint", os.path.join(directory, "model.pt"),
            "--ctx", "8", "--dim", "32", "--heads", "4", "--n-pre", "1",
            "--n-core", "1", "--n-post", "1", "--loops", "1",
            "--batch", "100000", "--steps", "1", "--no-gen", "--device", "cpu",
        ])
        with pytest.raises(ValueError, match="training loader is empty.*reduce --batch"):
            train_model(args, cfg_from_args(args), torch.device("cpu"))


def legacy_cfg() -> Config:
    return Config(
        ctx_len=32, dim=64, n_heads=4, n_pre=1, n_core=1, n_post=1,
        max_loops=2, block_size=16, mlp_mult=2,
    )


def test_corrupt_and_loss():
    cfg = legacy_cfg()
    model = MarianaLM(cfg)
    x = torch.randint(0, 256, (4, cfg.ctx_len + 1))
    inputs, targets, originals, times, region, attn_mask, flex_bm = corrupt_batch(x, cfg)
    assert inputs.shape == (4, cfg.ctx_len)
    assert (inputs[:, 0] == cfg.bos_token).all()
    assert (times[region == 0] == 0).all()
    causal = torch.tril(torch.ones(cfg.ctx_len, cfg.ctx_len)).bool().unsqueeze(0).unsqueeze(0)
    losses = hybrid_losses(model, inputs, targets, originals, times, region,
                           attn_mask, None, cfg, causal, cfg.max_loops)
    for k in ("ar", "diff", "align", "ver", "total"):
        assert k in losses and torch.isfinite(losses[k])
    losses["total"].backward()


def test_training_reduces_loss():
    cfg = legacy_cfg()
    model = MarianaLM(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    x = torch.randint(0, 256, (8, cfg.ctx_len + 1))
    causal = torch.tril(torch.ones(cfg.ctx_len, cfg.ctx_len)).bool().unsqueeze(0).unsqueeze(0)
    first = last = None
    for _ in range(8):
        inputs, targets, originals, times, region, attn_mask, _ = corrupt_batch(x, cfg)
        losses = hybrid_losses(model, inputs, targets, originals, times, region,
                               attn_mask, None, cfg, causal, cfg.max_loops)
        opt.zero_grad()
        losses["total"].backward()
        opt.step()
        first = first or float(losses["total"].detach())
        last = float(losses["total"].detach())
    assert last < first, f"loss did not decrease: {first} -> {last}"
