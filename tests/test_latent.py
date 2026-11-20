"""The learned latent code: shapes, losses, recall curriculum, ablations, export parity."""

from __future__ import annotations

from unittest.mock import patch

import torch

from mariana.config import Config
from mariana.export.compat import to_llama_state
from mariana.export.onnx_export import ARWrapper
from mariana.latent import (
    _make_recall_batch,
    _mask_local_windows,
    _prepare_batch,
    _shuffle_symbols,
    _symbol_usage_entropy,
    code_ablation,
    format_code,
    generate_latent,
    latent_batch_loss,
    latent_prefill,
    summarize,
)
from mariana.model import MarianaLM
from mariana.train import make_causal_mask


def tiny_cfg(use_latent: bool = True) -> Config:
    return Config(
        ctx_len=48,
        dim=64,
        n_heads=4,
        n_pre=1,
        n_core=1,
        n_post=1,
        max_loops=2,
        block_size=16,
        mlp_mult=2,
        use_latent=use_latent,
        n_latent_symbols=32,
        n_thoughts=4,
        latent_local_window=8,
    )


def test_format_code():
    code = format_code(torch.tensor([[0, 1, 36]]))
    assert code == "⟨00⟩⟨01⟩⟨10⟩"


def test_summarize_shapes():
    cfg = tiny_cfg()
    model = MarianaLM(cfg).eval()
    context = torch.randint(0, 256, (2, 20))
    context[:, 0] = cfg.bos_token
    with torch.no_grad():
        out = summarize(
            model, context, cfg, torch.device("cpu"), training=False
        )
    assert out["symbols"].shape == (2, cfg.n_thoughts)
    assert out["symbol_embeds"].shape == (2, cfg.n_thoughts, cfg.dim)
    assert out["logits"].shape == (2, cfg.n_thoughts, cfg.n_latent_symbols)
    assert int(out["symbols"].min()) >= 0
    assert int(out["symbols"].max()) < cfg.n_latent_symbols
    assert torch.isfinite(out["logits"]).all()


def test_latent_loss_backward():
    cfg = tiny_cfg()
    model = MarianaLM(cfg).train()
    x = torch.randint(0, 256, (2, cfg.ctx_len + 1))
    losses = latent_batch_loss(model, x, cfg, torch.device("cpu"), training=True)
    for key in ("lat_cont", "lat_local", "lat_ent", "lat_total"):
        assert torch.isfinite(losses[key]), key
    losses["lat_total"].backward()
    assert model.latent_head.weight.grad is not None
    assert model.codebook.weight.grad is not None
    assert model.latent_head.weight.grad.abs().sum() > 0
    assert model.codebook.weight.grad.abs().sum() > 0


def test_latent_moe_load_balancing_reaches_total_and_routers():
    cfg = tiny_cfg()
    cfg.use_moe = True
    cfg.latent_recall_prob = 0.0
    cfg.latent_window_dropout = 0.0
    cfg.w_latent = 0.0
    cfg.w_latent_entropy = 0.0
    cfg.w_moe_lb = 0.25
    model = MarianaLM(cfg).train()
    x = torch.randint(0, 256, (2, cfg.ctx_len))

    losses = latent_batch_loss(model, x, cfg, torch.device("cpu"), training=True)
    assert torch.isfinite(losses["lat_moe_lb"])
    assert float(losses["lat_moe_lb"].detach()) > 0.0
    torch.testing.assert_close(
        losses["lat_total"], cfg.w_moe_lb * losses["lat_moe_lb"]
    )
    losses["lat_total"].backward()
    blocks = list(model.pre) + list(model.core) + list(model.post)
    assert any(
        block.mlp.router.weight.grad is not None
        and block.mlp.router.weight.grad.abs().sum() > 0
        for block in blocks
    )


def test_recall_batch_has_unique_keys_and_no_padding_targets():
    cfg = tiny_cfg()
    cfg.ctx_len = 64
    cfg.n_thoughts = 8
    cfg.n_latent_symbols = 256
    cfg.latent_recall_pairs = 4

    x, cutoff = _make_recall_batch(3, cfg, torch.device("cpu"), cfg.ctx_len)
    assert x.shape == (3, cutoff + cfg.latent_recall_pairs)
    keys = x[:, cutoff - cfg.latent_local_window:cutoff - cfg.latent_local_window + 4]
    assert all(len(set(row.tolist())) == 4 for row in keys)

    _, _, targets, prepared_cutoff = _prepare_batch(
        x, cfg, torch.device("cpu"), cutoff=cutoff
    )
    assert prepared_cutoff == cutoff
    assert targets.shape == (3, cfg.latent_recall_pairs)
    torch.testing.assert_close(targets, x[:, cutoff:])


def test_recall_queries_permute_keys_and_values_together():
    cfg = tiny_cfg()
    cfg.ctx_len = 64
    cfg.n_thoughts = 8
    cfg.n_latent_symbols = 256
    cfg.latent_recall_pairs = 4
    keys = torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23]])
    values = torch.tensor([[101, 102, 103, 104], [111, 112, 113, 114]])
    key_scores = torch.full((2, 256), -1.0)
    for row, row_keys in enumerate(keys):
        for index, key in enumerate(row_keys):
            key_scores[row, key] = 1.0 - index * 0.01
    order_scores = torch.tensor([
        [0.1, 0.2, 0.3, 0.4],
        [0.4, 0.3, 0.2, 0.1],
    ])

    with (
        patch("mariana.latent.torch.rand", side_effect=[key_scores, order_scores]),
        patch("mariana.latent.torch.randint", return_value=values),
        patch("mariana.latent.random.randint", side_effect=[24, 4, 4]),
    ):
        x, cutoff = _make_recall_batch(2, cfg, torch.device("cpu"), cfg.ctx_len)

    order = torch.tensor([[3, 0, 1, 2], [3, 2, 1, 0]])
    assert cutoff == 24
    assert x.shape == (2, cutoff + cfg.latent_recall_pairs)
    torch.testing.assert_close(x[:, 4:12:2], keys)
    torch.testing.assert_close(x[:, 5:12:2], values)
    query = x[:, cutoff - cfg.latent_local_window:cutoff - cfg.latent_local_window + 4]
    torch.testing.assert_close(query, keys.gather(1, order))
    torch.testing.assert_close(x[:, cutoff:], values.gather(1, order))


def test_latent_recall_loss_is_marked():
    cfg = tiny_cfg()
    cfg.latent_recall_prob = 1.0
    model = MarianaLM(cfg).train()
    x = torch.randint(0, 256, (2, cfg.ctx_len + 1))
    losses = latent_batch_loss(model, x, cfg, torch.device("cpu"), training=True)
    assert float(losses["lat_recall"]) == 1.0
    assert losses["context_len"] < cfg.ctx_len
    assert torch.isfinite(losses["lat_total"])


def test_local_window_masking_is_per_example():
    local = torch.arange(24).view(4, 6)
    random_values = torch.tensor([[0.1], [0.9], [0.2], [0.8]])
    with patch("mariana.latent.torch.rand", return_value=random_values):
        masked = _mask_local_windows(local, 0.5, 257)
    assert (masked[0] == 257).all()
    assert (masked[2] == 257).all()
    torch.testing.assert_close(masked[1], local[1])
    torch.testing.assert_close(masked[3], local[3])


def test_ablation_shuffle_is_per_row_and_not_identity():
    symbols = torch.arange(4).repeat(3, 1)
    random_values = torch.tensor([
        [0.1, 0.2, 0.3, 0.4],
        [0.4, 0.3, 0.2, 0.1],
        [0.2, 0.4, 0.1, 0.3],
    ])
    with patch("mariana.latent.torch.rand", return_value=random_values):
        shuffled = _shuffle_symbols(symbols)
    expected = torch.tensor([
        [3, 0, 1, 2],
        [3, 2, 1, 0],
        [2, 0, 3, 1],
    ])
    torch.testing.assert_close(shuffled, expected)


def test_symbol_entropy_measures_emitted_usage():
    collapsed = torch.zeros(4, 4, dtype=torch.long)
    balanced = torch.arange(32).repeat(2)
    assert float(_symbol_usage_entropy(collapsed, 32)) < 1e-6
    torch.testing.assert_close(
        _symbol_usage_entropy(balanced, 32), torch.tensor(1.0)
    )


def test_latent_generate():
    cfg = tiny_cfg()
    model = MarianaLM(cfg).eval()
    out, symbols, stats = generate_latent(
        model, b"HELLO", cfg, torch.device("cpu"),
        max_new_tokens=8, temperature=0.9,
    )
    assert len(out) > 0
    assert symbols.shape == (1, cfg.n_thoughts)
    assert stats["n_thoughts"] == cfg.n_thoughts
    assert stats["context_bytes"] >= 1


def test_latent_prefill_matches_full_forward():
    cfg = tiny_cfg()
    model = MarianaLM(cfg).eval()
    symbols = torch.randint(0, cfg.n_latent_symbols, (1, cfg.n_thoughts))
    local = torch.randint(0, 256, (1, cfg.latent_local_window))
    local[:, 0] = cfg.bos_token

    thought = model.codebook(symbols) + model.thought_pos.weight[: cfg.n_thoughts].unsqueeze(0)
    emb = torch.cat([thought, model.tok_emb(local)], dim=1)
    seq_len = emb.shape[1]
    zeros = torch.zeros(1, seq_len, dtype=torch.long)
    causal = make_causal_mask(seq_len).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        full, _, _ = model(input_embeds=emb, times=zeros, region=zeros, mask=causal)

        pre, kv, pre_len = latent_prefill(
            model, symbols, local[:, :-1], cfg, torch.device("cpu")
        )
        z = torch.zeros(1, 1, dtype=torch.long)
        mask = torch.ones(1, 1, 1, pre_len + 1, dtype=torch.bool)
        with torch.no_grad():
            last, _, _ = model(
                local[:, -1:], z, z, mask, kv_cache=kv, return_kv=True
            )

    torch.testing.assert_close(pre, full[:, -2, :], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(last[:, -1], full[:, -1, :], atol=1e-4, rtol=1e-4)


def test_code_ablation():
    cfg = tiny_cfg()
    model = MarianaLM(cfg).eval()
    x = torch.randint(0, 256, (2, cfg.ctx_len + 1))
    out = code_ablation(model, x, cfg, torch.device("cpu"))
    for key in (
        "true", "shuffled", "random", "blank",
        "gain_over_blank", "shuffle_penalty", "random_penalty",
    ):
        assert torch.isfinite(torch.tensor(out[key])), key
    assert out["symbols"].shape == (2, cfg.n_thoughts)

    # The synthetic recall task must fit and run as well.
    rec = code_ablation(
        model, x, cfg, torch.device("cpu"), recall=True
    )
    assert rec["recall"] is True
    assert torch.isfinite(torch.tensor(rec["true"]))

    masked = code_ablation(
        model, x, cfg, torch.device("cpu"), mask_local=True
    )
    assert masked["mask_local"] is True
    assert torch.isfinite(torch.tensor(masked["local_gain"]))


def test_ar_path_unchanged_and_export_ignores_latent():
    cfg_plain = tiny_cfg(use_latent=False)
    cfg_latent = tiny_cfg(use_latent=True)
    plain = MarianaLM(cfg_plain).eval()
    latent = MarianaLM(cfg_latent).eval()

    # copy every shared tensor so only the presence of latent modules differs
    plain_sd = plain.state_dict()
    latent_sd = latent.state_dict()
    common = {
        k: v for k, v in latent_sd.items()
        if k in plain_sd and plain_sd[k].shape == v.shape
    }
    plain.load_state_dict(common, strict=False)

    x = torch.randint(0, 256, (1, 16))
    zeros = torch.zeros(1, 16, dtype=torch.long)
    causal = make_causal_mask(16).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        a, _, _ = plain(x, zeros, zeros, causal)
        b, _, _ = latent(x, zeros, zeros, causal)
        wrapper = ARWrapper(latent)
        c = wrapper(x)
    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(b, c, atol=1e-5, rtol=1e-5)

    llama_sd = to_llama_state(latent.state_dict(), cfg_latent)
    assert "codebook.weight" not in llama_sd
    assert "thought_bos" not in llama_sd
    assert "token_embd.weight" in llama_sd
