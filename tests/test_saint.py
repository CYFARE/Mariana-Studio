"""SAINT adaptive hierarchical thought: policy, grammar, memory, and AR/export invariance."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

import torch

from mariana.cli import cfg_from_args, parse_args
from mariana.config import Config
from mariana.export.compat import to_llama_state
from mariana.export.onnx_export import ARWrapper
import mariana.generate as generation
from mariana.latent import generate_latent, latent_batch_loss, summarize
from mariana.model import MarianaLM
from mariana.saint import (
    _attention_mask,
    _cached_byte_forward,
    build_memory,
    make_grammar_batch,
    memory_prefill,
    prompt_memory,
)
from mariana.train import make_causal_mask


def tiny_cfg() -> Config:
    return Config(
        ctx_len=64,
        dim=32,
        n_heads=4,
        n_pre=1,
        n_core=1,
        n_post=1,
        max_loops=1,
        block_size=8,
        mlp_mult=2,
        use_latent=True,
        latent_saint=True,
        n_latent_symbols=16,
        n_thoughts=4,
        latent_min_thoughts=1,
        latent_local_window=8,
        latent_continuous_slots=2,
        latent_refine_steps=1,
        latent_recall_prob=0.0,
        latent_grammar_prob=0.0,
        latent_window_dropout=0.0,
    )


def test_saint_shapes_masks_and_hierarchy():
    cfg = tiny_cfg()
    model = MarianaLM(cfg).eval()
    context = torch.randint(0, 256, (2, 24))
    with torch.no_grad():
        sym = summarize(model, context, cfg, torch.device("cpu"))
        memory, memory_mask = build_memory(model, sym, cfg)

    N, S, C = cfg.n_thoughts, cfg.n_latent_symbols, cfg.latent_continuous_slots
    assert sym["symbols"].shape == (2, N)
    assert sym["logits"].shape == (2, N, S + 1)
    assert sym["continuous"].shape == (2, N, C, cfg.dim)
    assert sym["parent_symbols"].shape == (2, model.n_latent_parents)
    assert sym["lengths"].dtype == torch.long
    torch.testing.assert_close(sym["lengths"], sym["content_mask"].sum(dim=1))
    expected_parents = (sym["lengths"] + cfg.latent_block_size - 1) // cfg.latent_block_size
    torch.testing.assert_close(sym["parent_mask"].sum(dim=1), expected_parents)
    expected_memory = sym["lengths"] * (1 + C) + expected_parents
    torch.testing.assert_close(memory_mask.sum(dim=1), expected_memory)
    assert int(sym["symbols"].max()) <= S + 1
    assert torch.isfinite(memory).all()


def test_eos_is_real_and_stops_encoder_and_generation():
    cfg = replace(tiny_cfg(), n_thoughts=5, latent_min_thoughts=2)
    model = MarianaLM(cfg).eval()

    def force_eos(_module, _args, output):
        output = output.clone()
        output[..., cfg.n_latent_symbols] = 10.0
        return output

    handle = model.latent_head.register_forward_hook(force_eos)
    try:
        context = torch.randint(0, 256, (2, 24))
        with torch.no_grad():
            sym = summarize(model, context, cfg, torch.device("cpu"))
        assert sym["n_steps"] == 3
        assert sym["terminated"].all()
        torch.testing.assert_close(sym["lengths"], torch.full((2,), 2))
        assert (sym["symbols"][:, 2] == cfg.n_latent_symbols).all()
        assert (sym["symbols"][:, 3:] == cfg.n_latent_symbols + 1).all()
        assert sym["content_mask"][:, :2].all()
        assert not sym["content_mask"][:, 2:].any()

        out, symbols, stats = generate_latent(
            model, b"HELLO", cfg, torch.device("cpu"),
            max_new_tokens=2, temperature=0.0,
        )
    finally:
        handle.remove()
    assert len(out) == 2
    assert symbols.shape == (1, 3)  # two content actions plus EOS
    assert stats["n_thoughts"] == 2
    assert stats["max_thoughts"] == 5
    assert stats["stopped"] is True
    assert stats["n_steps"] == 3


def test_minimum_length_eos_has_finite_grammar_and_policy_losses():
    cfg = replace(tiny_cfg(), latent_min_thoughts=2)
    model = MarianaLM(cfg).train()

    def force_eos(_module, _args, output):
        result = output * 0.0
        result[..., cfg.n_latent_symbols] = 10.0
        return result

    handle = model.latent_head.register_forward_hook(force_eos)
    try:
        x = torch.randint(0, 256, (2, cfg.ctx_len + 1))
        losses = latent_batch_loss(model, x, cfg, torch.device("cpu"), training=True)
    finally:
        handle.remove()
    assert float(losses["lat_len"]) == 2.0
    for key in ("lat_grammar", "lat_rl", "lat_total"):
        assert torch.isfinite(losses[key]), key
    losses["lat_total"].backward()
    assert torch.isfinite(model.latent_head.weight.grad).all()


def test_cached_decode_preserves_masked_memory_and_matches_full_forward():
    cfg = tiny_cfg()
    model = MarianaLM(cfg).eval()

    def force_eos(_module, _args, output):
        output = output.clone()
        output[..., cfg.n_latent_symbols] = 10.0
        return output

    handle = model.latent_head.register_forward_hook(force_eos)
    try:
        with torch.no_grad():
            sym = summarize(
                model, torch.randint(0, 256, (1, 24)), cfg, torch.device("cpu")
            )
            memory, memory_mask = build_memory(model, sym, cfg)
            local = torch.randint(0, 256, (1, cfg.latent_local_window))
            local[:, 0] = cfg.bos_token
            token = torch.tensor([[65]])
            byte_in = torch.cat([local, token], dim=1)
            emb = torch.cat([memory, model.tok_emb(byte_in)], dim=1)
            zeros = torch.zeros(1, emb.shape[1], dtype=torch.long)
            full, _, _ = model(
                input_embeds=emb, times=zeros, region=zeros,
                mask=_attention_mask(memory_mask, byte_in.shape[1]),
            )
            _, kv, _ = memory_prefill(model, memory, memory_mask, local, cfg)
            cached, _, _ = _cached_byte_forward(
                model, token, kv, memory_mask, byte_in.shape[1]
            )
    finally:
        handle.remove()
    assert not memory_mask.all(), "test requires EOS/PAD memory positions"
    torch.testing.assert_close(cached[:, -1], full[:, -1], atol=1e-5, rtol=1e-5)


def test_all_saint_losses_backpropagate():
    cfg = tiny_cfg()
    model = MarianaLM(cfg).train()
    x = torch.randint(0, 256, (2, cfg.ctx_len + 1))
    losses = latent_batch_loss(model, x, cfg, torch.device("cpu"), training=True)
    for key in (
        "lat_cont", "lat_grammar", "lat_pair", "lat_order", "lat_hier",
        "lat_rl", "lat_value", "lat_reward", "lat_kl", "lat_len", "lat_total",
    ):
        assert torch.isfinite(losses[key]), key
    losses["lat_total"].backward()
    for parameter in (
        model.latent_head.weight,
        model.codebook.weight,
        model.continuous_head.weight,
        model.parent_head.weight,
        model.parent_readout.weight,
        model.grammar_pair.weight,
        model.order_head.weight,
        model.latent_value[-1].weight,
    ):
        assert parameter.grad is not None
        assert parameter.grad.abs().sum() > 0


def test_policy_gradient_alone_trains_symbol_policy_including_eos_row():
    cfg = tiny_cfg()
    for name in (
        "w_latent", "w_latent_entropy", "w_latent_grammar", "w_latent_hierarchy",
        "w_latent_value", "w_latent_kl", "w_latent_policy_entropy", "w_moe_lb",
    ):
        setattr(cfg, name, 0.0)
    cfg.w_latent_rl = 1.0
    model = MarianaLM(cfg).train()
    x = torch.randint(0, 256, (2, cfg.ctx_len + 1))
    losses = latent_batch_loss(model, x, cfg, torch.device("cpu"), training=True)
    torch.testing.assert_close(losses["lat_total"], losses["lat_rl"])
    losses["lat_total"].backward()
    grad = model.latent_head.weight.grad
    assert grad is not None and grad.abs().sum() > 0
    assert grad[cfg.n_latent_symbols].abs().sum() > 0
    parent_grad = model.parent_head.weight.grad
    assert parent_grad is not None and parent_grad.abs().sum() > 0


def test_synthetic_curriculum_memory_does_not_condition_natural_diffusion():
    cfg = replace(
        tiny_cfg(), n_latent_symbols=256, n_thoughts=8,
        latent_local_window=12, latent_recall_pairs=2,
        latent_continuous_slots=1, latent_refine_steps=0,
        latent_recall_prob=1.0,
    )
    model = MarianaLM(cfg).train()
    x = torch.randint(0, 256, (2, cfg.ctx_len + 1))
    recall = latent_batch_loss(model, x, cfg, torch.device("cpu"), training=True)
    assert float(recall["lat_recall"]) == 1.0
    assert recall["diffusion_memory"] is None
    assert recall["diffusion_memory_mask"] is None

    cfg.latent_recall_prob = 0.0
    cfg.latent_grammar_prob = 1.0
    grammar = latent_batch_loss(model, x, cfg, torch.device("cpu"), training=True)
    assert float(grammar["lat_grammar_task"]) == 1.0
    assert grammar["diffusion_memory"] is None

    cfg.latent_grammar_prob = 0.0
    natural = latent_batch_loss(
        model, x, cfg, torch.device("cpu"), cutoff=32, training=False
    )
    assert natural["diffusion_memory"] is not None
    assert natural["diffusion_memory_mask"] is not None


def test_compositional_grammar_batch_computes_exact_operator_results():
    cfg = replace(
        tiny_cfg(), n_latent_symbols=256, n_thoughts=8,
        latent_local_window=12, latent_recall_pairs=2,
    )
    key_scores = torch.arange(240, dtype=torch.float32).unsqueeze(0)
    order_scores = torch.tensor([[0.1, 0.2]])
    values = torch.tensor([[10, 20]])
    right_indices = torch.tensor([[0, 1]])
    operations = torch.tensor([[0, 3]])
    with (
        patch("mariana.saint.torch.rand", side_effect=[key_scores, order_scores]),
        patch(
            "mariana.saint.torch.randint",
            side_effect=[values, right_indices, operations],
        ),
        patch("mariana.saint.random.randint", side_effect=[30, 5]),
    ):
        x, cutoff = make_grammar_batch(1, cfg, torch.device("cpu"), cfg.ctx_len)

    assert cutoff == 30
    assert x.shape == (1, 32)
    # Left queries reverse the table; right operands independently use [0, 1].
    torch.testing.assert_close(x[0, 5:9], torch.tensor([255, 10, 254, 20]))
    torch.testing.assert_close(
        x[0, 18:24], torch.tensor([254, 1, 255, 255, 4, 254])
    )
    assert x[0, 18] != x[0, 20] and x[0, 21] != x[0, 23]
    # Query 0: 20 XOR 10 = 30. Query 1: rotate 10 left by (20 & 7) = 160.
    torch.testing.assert_close(x[0, cutoff:], torch.tensor([30, 160]))


def test_channel_dropout_never_removes_both_channels():
    cfg = replace(tiny_cfg(), latent_discrete_dropout=1.0, latent_continuous_dropout=1.0)
    model = MarianaLM(cfg).eval()
    context = torch.randint(0, 256, (2, 24))
    with torch.no_grad():
        sym = summarize(model, context, cfg, torch.device("cpu"), training=True)
        _, mask = build_memory(model, sym, cfg, training=True)
    N, C = cfg.n_thoughts, cfg.latent_continuous_slots
    assert not mask[:, :N].any()
    assert not mask[:, N + N * C:].any()
    assert mask[:, N:N + N * C].any(dim=1).all()


def test_long_prompt_uses_same_slice_for_memory_and_ar():
    cfg = replace(tiny_cfg(), latent_continuous_slots=1, latent_refine_steps=0)
    model = MarianaLM(cfg).eval()
    seen = {}

    def memory_spy(model, prompt, cfg, device, *args, **kwargs):
        result = prompt_memory(model, prompt, cfg, device, *args, **kwargs)
        seen["memory"] = result[3][0, 1:].tolist()
        return result

    real_prefill = generation.prefill

    def prefill_spy(model, tokens, device, cfg):
        seen["ar"] = tokens[0, 1:].tolist()
        return real_prefill(model, tokens, device, cfg)

    with (
        patch("mariana.saint.prompt_memory", side_effect=memory_spy),
        patch("mariana.generate.prefill", side_effect=prefill_spy),
    ):
        generation.draft_refine_verify(
            model, bytes(range(100)), max_new_tokens=2, n_steps=1,
            temperature=0.0, draft_temperature=0.0,
            device=torch.device("cpu"), cfg=cfg,
        )
    expected = list(range(48, 100))
    assert seen["memory"] == expected
    assert seen["ar"] == expected


def test_extended_memory_training_and_generation_beyond_position_table():
    cfg = replace(
        tiny_cfg(), n_thoughts=20, latent_continuous_slots=4,
        latent_refine_steps=0,
    )
    model = MarianaLM(cfg)
    x = torch.randint(0, 256, (1, cfg.ctx_len + 1))
    losses = latent_batch_loss(
        model, x, cfg, torch.device("cpu"), cutoff=27, training=False,
    )
    assert torch.isfinite(losses["lat_total"])
    assert model.angles_pos.shape[0] >= 154

    model.eval()
    out, _, _ = generate_latent(
        model, b"HELLO", cfg, torch.device("cpu"),
        max_new_tokens=cfg.ctx_len, temperature=0.0,
    )
    assert len(out) == cfg.ctx_len


def test_critic_loss_is_independent_of_policy_gradient_switch():
    cfg = replace(tiny_cfg(), w_latent_rl=0.0, w_latent_value=0.5)
    model = MarianaLM(cfg).train()
    x = torch.randint(0, 256, (2, cfg.ctx_len + 1))
    losses = latent_batch_loss(model, x, cfg, torch.device("cpu"), training=True)
    assert float(losses["lat_rl"].detach()) == 0.0
    assert float(losses["lat_value"].detach()) > 0.0
    losses["lat_total"].backward()
    grad = model.latent_value[-1].weight.grad
    assert grad is not None and grad.abs().sum() > 0


def test_latent_cross_attention_is_draft_only():
    cfg = tiny_cfg()
    model = MarianaLM(cfg).eval()
    for block in list(model.pre) + list(model.core) + list(model.post):
        with torch.no_grad():
            block.latent_attn.gate.fill_(0.1)
    tokens = torch.randint(0, 256, (2, 12))
    memory = torch.randn(2, 6, cfg.dim)
    memory_mask = torch.ones(2, 6, dtype=torch.bool)
    causal = make_causal_mask(12).unsqueeze(0).unsqueeze(0)
    zeros = torch.zeros(2, 12, dtype=torch.long)
    ones = torch.ones(2, 12, dtype=torch.long)
    with torch.no_grad():
        clean, _, _ = model(tokens, zeros, zeros, causal)
        clean_conditioned, _, _ = model(
            tokens, zeros, zeros, causal,
            latent_memory=memory, latent_memory_mask=memory_mask,
        )
        draft, _, _ = model(tokens, ones, ones, causal)
        draft_conditioned, _, _ = model(
            tokens, ones, ones, causal,
            latent_memory=memory, latent_memory_mask=memory_mask,
        )
    torch.testing.assert_close(clean, clean_conditioned, atol=0.0, rtol=0.0)
    assert not torch.allclose(draft, draft_conditioned)


def test_saint_ar_path_and_llama_export_are_unchanged():
    cfg = tiny_cfg()
    plain_cfg = replace(cfg, use_latent=False, latent_saint=False)
    plain = MarianaLM(plain_cfg).eval()
    saint = MarianaLM(cfg).eval()
    plain_sd = plain.state_dict()
    common = {
        key: value for key, value in saint.state_dict().items()
        if key in plain_sd and plain_sd[key].shape == value.shape
    }
    plain.load_state_dict(common, strict=False)
    tokens = torch.randint(0, 256, (1, 12))
    zeros = torch.zeros(1, 12, dtype=torch.long)
    causal = make_causal_mask(12).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        expected, _, _ = plain(tokens, zeros, zeros, causal)
        actual, _, _ = saint(tokens, zeros, zeros, causal)
        wrapped = ARWrapper(saint)(tokens)
    torch.testing.assert_close(expected, actual, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(actual, wrapped, atol=1e-5, rtol=1e-5)
    llama = to_llama_state(saint.state_dict(), cfg)
    assert "token_embd.weight" in llama
    assert not any("parent" in key or "continuous" in key for key in llama)


def test_cli_opts_into_saint_and_preserves_legacy_config():
    args = parse_args(["train", "--latent"])
    assert cfg_from_args(args).latent_saint
    args = parse_args(["train", "--latent", "--latent-legacy"])
    assert not cfg_from_args(args).latent_saint
    args = parse_args(["train", "--latent", "--no-latent-diffusion"])
    assert not cfg_from_args(args).latent_diffusion
    assert not Config.from_dict({"use_latent": True}).latent_saint
