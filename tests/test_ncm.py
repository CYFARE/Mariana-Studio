"""Next-Concept Modeling: tokenizer, generator, training, checkpoints and the CLI."""

from __future__ import annotations

import os
import tempfile
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from mariana.cli import (
    cmd_gen as mariana_cmd_gen,
    cmd_train as mariana_cmd_train,
    parse_args as mariana_parse_args,
)
from mariana.device import Amp
from mariana.ncm import MultiscaleTokenizer, NCMConfig, NCMGenerator
from mariana.ncm.cli import build_parser, cfg_from_args, load_generator, prepare_prompt
from mariana.ncm.generator import CodebookHead
from mariana.ncm.layers import mbr_select
from mariana.ncm.train import free_running_metrics, save_checkpoint


def tiny_cfg(**overrides):
    values = dict(
        ctx_len=8, prefix_len=4, levels=(1, 2, 4, 8),
        latent_dim=16, dim=32, n_heads=4, codebook_size=16,
        encoder_layers=1, decoder_layers=1, refiner_layers=1,
        generator_layers=1, mlp_mult=2, self_condition=1.0,
        nested_dropout=0.5, corruption=0.2,
    )
    values.update(overrides)
    return NCMConfig(**values)


def make_models(cfg=None):
    cfg = cfg or tiny_cfg()
    tokenizer = MultiscaleTokenizer(cfg)
    generator = NCMGenerator(tokenizer, cfg)
    return cfg, tokenizer, generator


def make_batch(cfg, batch=2):
    target = torch.randint(0, 256, (batch, cfg.ctx_len))
    prompt = torch.randint(0, 256, (batch, cfg.prefix_len))
    return prompt, target


def test_config_and_cli():
    args = build_parser().parse_args([
        "tokenizer", "--ctx", "16", "--prefix", "8", "--levels", "2,4,8,16"
    ])
    cfg = cfg_from_args(args)
    assert cfg.levels == (2, 4, 8, 16)
    args = build_parser().parse_args(["tokenizer", "--ctx", "2"])
    assert cfg_from_args(args).levels == (1, 2)
    assert cfg_from_args(build_parser().parse_args(["tokenizer"])).levels == (
        1, 8, 16, 24, 32, 40, 52, 64, 76, 92, 108, 128
    )
    args = build_parser().parse_args(["tokenizer", "--ctx", "512"])
    assert cfg_from_args(args).levels == (1, 4, 9, 16, 25, 36, 49, 64, 96, 128, 192, 256, 384, 512)
    assert cfg_from_args(args).corruption == 0.5
    assert NCMConfig(ctx_len=1).levels == (1,)
    assert NCMConfig(ctx_len=2).levels == (1, 2)
    assert NCMConfig(ctx_len=512).levels == cfg_from_args(args).levels
    assert build_parser().parse_args(["train", "--steps", "2"]).generator_lr == 1e-3
    assert build_parser().parse_args(["generator", "--steps", "2"]).lr == 1e-3
    try:
        NCMConfig(ctx_len=8, levels=(2, 4))
    except ValueError as e:
        assert "end at ctx_len" in str(e)
    else:
        raise AssertionError("invalid hierarchy was accepted")


def test_integrated_ncm_flags_and_two_stage_dispatch():
    args = mariana_parse_args(["train", "--ncm", "--no-gen"])
    with patch("mariana.cli._run_ncm") as dispatch:
        mariana_cmd_train(args)
    argv = dispatch.call_args.args[0]
    assert argv[0] == "train"
    assert argv[argv.index("--checkpoint") + 1].endswith("mariana-ncm.pt")
    assert "--tokenizer-steps" in argv and "--no-gen" in argv

    args = mariana_parse_args(["gen", "--ncm", "--gen-len", "12"])
    with patch("mariana.cli._run_ncm") as dispatch:
        mariana_cmd_gen(args)
    argv = dispatch.call_args.args[0]
    assert argv[0] == "gen" and "12" in argv
    assert mariana_parse_args(["--ncm"]).cmd == "train"
    assert mariana_parse_args(["ncm", "generator"]).ncm_args == ["generator"]

    from mariana.ncm.cli import cmd_train_all

    args = build_parser().parse_args([
        "train", "--tokenizer-steps", "3", "--steps", "5", "--no-gen"
    ])
    with patch("mariana.ncm.cli.cmd_tokenizer") as tokenizer, \
            patch("mariana.ncm.cli.cmd_generator") as generator:
        cmd_train_all(args)
    assert tokenizer.call_args.args[0].steps == 3
    assert tokenizer.call_args.args[0].checkpoint != generator.call_args.args[0].checkpoint
    assert generator.call_args.args[0].steps == 5
    assert generator.call_args.args[0].lr == 1e-3


def test_tokenizer_forward_backward_and_hierarchy():
    cfg = tiny_cfg()
    tokenizer = MultiscaleTokenizer(cfg)
    _, target = make_batch(cfg)
    before = target.clone()
    losses = tokenizer(target)
    assert torch.equal(target, before)
    for name in ("total", "recon", "vq", "refine", "acc", "usage", "perplexity"):
        assert torch.isfinite(losses[name]), name
    losses["total"].backward()
    assert tokenizer.tokens.weight.grad is not None
    assert tokenizer.quantizer.down_refiners[0].conv.weight.grad is not None
    assert all(layer.conv.weight.grad is None for layer in tokenizer.quantizer.down_refiners[1:])
    assert all(layer.conv.weight.grad is not None for layer in tokenizer.quantizer.up_refiners)
    assert all(not emb.weight.requires_grad for emb in tokenizer.quantizer.embeddings)
    codes = tokenizer.encode(target)
    assert [tuple(code.shape) for code in codes] == [
        (target.shape[0], length) for length in cfg.levels
    ]


def test_partial_reconstruction_is_exact_prefix():
    cfg, tokenizer, _ = make_models()
    tokenizer.eval()
    _, target = make_batch(cfg)
    codes = tokenizer.encode(target)
    full = tokenizer.quantizer.reconstruct(codes)
    for level in range(1, cfg.n_levels + 1):
        prefix = tokenizer.quantizer.reconstruct(codes[:level])
        explicit = tokenizer.quantizer.reconstruct(codes, upto=level)
        torch.testing.assert_close(prefix, explicit)
    torch.testing.assert_close(full, tokenizer.quantizer.reconstruct(codes, upto=cfg.n_levels))


def test_corruption_preserves_targets():
    cfg, _, generator = make_models()
    _, target = make_batch(cfg)
    codes = tokenizer_codes = generator.tokenizer.encode(target)
    before = [code.clone() for code in codes]
    corrupted = generator.corrupt_codes(codes)
    assert all(torch.equal(a, b) for a, b in zip(codes, before))
    assert all(torch.equal(a, b) for a, b in zip(codes, tokenizer_codes))
    assert all(a.shape == b.shape for a, b in zip(codes, corrupted))


def test_levels_do_not_leak_backward():
    cfg, _, generator = make_models()
    generator.eval()
    prompt, target = make_batch(cfg)
    codes = generator.tokenizer.encode(target)
    with torch.no_grad():
        baseline = generator(prompt, codes)
        changed = [code.clone() for code in codes]
        changed[0] = (changed[0] + 1) % cfg.codebook_size
        altered = generator(prompt, changed)
    torch.testing.assert_close(baseline[0], altered[0])
    assert any(not torch.allclose(a, b) for a, b in zip(baseline[1:], altered[1:]))


def test_self_conditioned_loss_and_frozen_tokenizer():
    cfg, _, generator = make_models()
    generator.train()
    prompt, target = make_batch(cfg)
    codes = generator.tokenizer.encode(target)
    losses = generator.loss(prompt, codes)
    assert float(losses["self"]) == 1.0
    assert torch.isfinite(losses["total"])
    assert float(losses["consistency"].detach()) >= 0
    losses["total"].backward()
    assert generator.latent_proj.weight.grad is not None
    assert generator.heads[0].proj.weight.grad is not None
    assert all(p.grad is None for p in generator.tokenizer.parameters())


def test_consistency_loss_is_averaged_per_code_position():
    cfg, _, generator = make_models(tiny_cfg(prompt_dropout=0.0, corruption=0.0))
    generator.train()
    prompt, target = make_batch(cfg)
    codes = generator.tokenizer.encode(target)
    teacher = [torch.randn(target.shape[0], length, cfg.codebook_size) for length in cfg.levels]
    student = [torch.randn(target.shape[0], length, cfg.codebook_size) for length in cfg.levels]
    with patch.object(generator, "forward", side_effect=[teacher, student]):
        losses = generator.loss(prompt, codes)
    expected = torch.stack([
        F.kl_div(
            F.log_softmax(student_logits, dim=-1),
            F.log_softmax(teacher_logits, dim=-1),
            log_target=True, reduction="batchmean",
        ) / length
        for student_logits, teacher_logits, length in zip(student, teacher, cfg.levels)
    ]).mean()
    torch.testing.assert_close(losses["consistency"], expected)


def test_codebook_head_matches_quantizer_distance():
    head = CodebookHead(8, 4, 6)
    x = torch.randn(2, 3, 8)
    codebook = torch.randn(6, 4)
    projected = head.proj(x).float()
    entries_sq = codebook.square().sum(-1)
    expected = (2.0 * projected @ codebook.T - entries_sq) / entries_sq.mean()
    actual = head(x, codebook)
    torch.testing.assert_close(actual, expected)
    distances = (
        projected.square().sum(-1, keepdim=True) + entries_sq
        - 2.0 * projected @ codebook.T
    )
    assert torch.equal(actual.argmax(-1), distances.argmin(-1))
    actual.mean().backward()
    assert head.proj.weight.grad is not None


def test_generator_loss_uses_ncm_level_weights():
    for alpha in (0.0, 1.0):
        cfg, _, generator = make_models(tiny_cfg(level_alpha=alpha))
        generator.eval()
        prompt, target = make_batch(cfg)
        codes = generator.tokenizer.encode(target)
        logits = [
            torch.randn(target.shape[0], length, cfg.codebook_size)
            for length in cfg.levels
        ]
        with patch.object(generator, "forward", return_value=logits):
            losses = generator.loss(prompt, codes)
        level_losses = torch.stack([
            F.cross_entropy(logit.reshape(-1, cfg.codebook_size), code.reshape(-1))
            for logit, code in zip(logits, codes)
        ])
        if alpha == 0.0:
            lengths = torch.tensor(cfg.levels, dtype=level_losses.dtype)
            expected = (lengths * level_losses).sum() / lengths.sum()
        else:
            expected = level_losses.mean()
        torch.testing.assert_close(losses["code"], expected)


def test_legacy_checkpoint_retains_cosine_head():
    cfg = tiny_cfg()
    values = cfg.to_dict()
    values.pop("codebook_head")
    legacy = NCMConfig.from_checkpoint({"format": 1, "cfg": values})
    current = NCMConfig.from_checkpoint({"format": 2, "cfg": cfg.to_dict()})
    assert legacy.codebook_head == "cosine"
    assert current.codebook_head == "distance"


def test_generator_resume_preserves_checkpoint_head():
    from mariana.ncm.cli import cmd_generator

    args = build_parser().parse_args([
        "generator", "--resume", "--checkpoint", "legacy.pt", "--device", "cpu",
    ])
    with patch("mariana.ncm.cli.os.path.exists", return_value=True), \
            patch("mariana.ncm.cli.load_cfg", return_value=(tiny_cfg(codebook_head="cosine"), {})), \
            patch("mariana.ncm.cli.train_generator") as train:
        cmd_generator(args)
    assert train.call_args.args[1].codebook_head == "cosine"


def test_generator_inherits_legacy_tokenizer_head():
    from mariana.ncm.cli import cmd_generator

    args = build_parser().parse_args([
        "generator", "--tokenizer", "legacy-tokenizer.pt", "--device", "cpu",
    ])
    with patch("mariana.ncm.cli.load_cfg", return_value=(tiny_cfg(codebook_head="cosine"), {})), \
            patch("mariana.ncm.cli.train_generator") as train:
        cmd_generator(args)
    assert train.call_args.args[1].codebook_head == "cosine"


def test_legacy_generator_resume_can_migrate_to_distance():
    from mariana.ncm.train import run_training

    legacy_cfg = tiny_cfg(codebook_head="cosine")
    _, _, legacy = make_models(legacy_cfg)
    optimizer = torch.optim.AdamW(legacy.parameters(), lr=1e-3)
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "legacy-generator.pt")
        save_checkpoint(path, legacy, optimizer, legacy_cfg, 1, Amp(torch.device("cpu")), "generator")
        state = torch.load(path, weights_only=False)
        state["format"] = 1
        state["cfg"].pop("codebook_head")
        torch.save(state, path)

        cfg, _, generator = make_models()
        prompt, target = make_batch(cfg, batch=4)
        rows = torch.cat((prompt, target), dim=1)
        args = build_parser().parse_args([
            "generator", "--resume", "--checkpoint", path, "--steps", "2",
            "--batch", "2", "--eval-every", "2", "--log-every", "1",
        ])
        with patch("torch.optim.AdamW.load_state_dict") as load_optimizer:
            run_training(generator, rows, rows, cfg, args, torch.device("cpu"), path, "generator")
        load_optimizer.assert_not_called()
        migrated = torch.load(path, weights_only=False)
        assert migrated["format"] == 2
        assert migrated["step"] == 2
        assert migrated["optimizer"]["param_groups"][0]["lr"] > 0
        assert migrated["cfg"]["codebook_head"] == "distance"
        assert all(head.mode == "distance" for head in generator.heads)


def test_free_running_metrics_cover_every_level():
    cfg, _, generator = make_models()
    generator.eval()
    prompt, target = make_batch(cfg)
    metrics = free_running_metrics(generator, (torch.cat((prompt, target), dim=1),), cfg, torch.device("cpu"))
    assert set(metrics) == {"free", "free_bytes", *(f"free_l{i}" for i in range(cfg.n_levels))}
    assert all(0.0 <= float(value) <= 1.0 for value in metrics.values())


def test_prefix_perturbation_is_training_only():
    cfg, _, generator = make_models(tiny_cfg(prefix_noise=0.5, prefix_dropout=0.5))
    prompt, target = make_batch(cfg)
    codes = generator.tokenizer.encode(target)
    generator.eval()
    with torch.no_grad():
        first, second = generator(prompt, codes), generator(prompt, codes)
    for a, b in zip(first, second):
        torch.testing.assert_close(a, b)
    generator.train()
    with torch.no_grad():
        first, second = generator(prompt, codes), generator(prompt, codes)
        steady, again = (
            generator(prompt, codes, perturb=False),
            generator(prompt, codes, perturb=False),
        )
    assert any(not torch.allclose(a, b) for a, b in zip(first, second))
    for a, b in zip(steady, again):
        torch.testing.assert_close(a, b)


def test_byte_auxiliary_losses_are_consistent_and_trainable():
    cfg, tokenizer, generator = make_models(tiny_cfg(self_condition=0.0))
    generator.eval()
    prompt, target = make_batch(cfg)
    codes = tokenizer.encode(target)
    valid = target != cfg.pad_token
    with torch.no_grad():
        true_logits = [
            F.one_hot(code, cfg.codebook_size).float() * 20.0 for code in codes
        ]
        latent = generator._latent_from_logits(true_logits)
        expected = tokenizer.quantizer.reconstruct(codes)
        torch.testing.assert_close(latent, expected, atol=1e-4, rtol=1e-4)
        assert torch.isfinite(generator._byte_loss(true_logits, target, valid))

    cfg.byte_weight = 0.5
    cfg.free_byte_weight = 0.25
    with pytest.raises(ValueError, match="target bytes"):
        generator.loss(prompt, codes)
    losses = generator.loss(prompt, codes, target=target)
    for key in ("byte", "free_byte"):
        assert torch.isfinite(losses[key]), key
    expected_total = (
        losses["code"] + cfg.consistency_weight * losses["consistency"]
        + cfg.byte_weight * losses["byte"] + cfg.free_byte_weight * losses["free_byte"]
    )
    torch.testing.assert_close(losses["total"], expected_total)
    losses["total"].backward()
    assert generator.heads[0].proj.weight.grad is not None
    assert all(p.grad is None for p in generator.tokenizer.parameters())


def test_new_training_options_validate_and_reach_the_parser():
    cfg = tiny_cfg()
    assert cfg.prefix_noise == 0.0 and cfg.prefix_dropout == 0.0
    assert cfg.byte_weight == 0.0 and cfg.free_byte_weight == 0.0
    for values in (
        {"prefix_noise": -0.1}, {"prefix_dropout": 1.5},
        {"byte_weight": -1.0}, {"free_byte_weight": float("nan")},
    ):
        with pytest.raises(ValueError):
            tiny_cfg(**values)
    args = build_parser().parse_args([
        "generator", "--prefix-noise", "0.5", "--prefix-dropout", "0.25",
        "--byte-weight", "0.5", "--free-byte-weight", "0.25",
    ])
    assert args.prefix_noise == 0.5 and args.prefix_dropout == 0.25
    assert args.byte_weight == 0.5 and args.free_byte_weight == 0.25


def test_rms_norm_accepts_mixed_precision_inputs():
    from mariana.ncm.layers import RMSNorm

    norm = RMSNorm(8)
    x = torch.randn(2, 3, 8, dtype=torch.bfloat16, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = norm(x)
    assert output.dtype == torch.bfloat16
    output.float().sum().backward()
    assert x.grad is not None and norm.weight.grad is not None


def test_cached_generation_matches_uncached():
    cfg, _, generator = make_models()
    generator.eval()
    prompt, _ = make_batch(cfg, batch=1)
    cached, a = generator.generate(prompt, temperature=0, refine_steps=1)
    uncached, b = generator.generate(prompt, temperature=0, use_cache=False, refine_steps=1)
    assert torch.equal(cached, uncached)
    assert all(torch.equal(x, y) for x, y in zip(a["codes"], b["codes"]))
    assert a["level_nfe"] == cfg.n_levels
    assert a["generator_nfe"] == cfg.n_levels + 1
    assert a["decoder_nfe"] == 1


def test_guidance_cache_parity():
    cfg, _, generator = make_models()
    generator.eval()
    prompt, _ = make_batch(cfg, batch=1)
    cached, a = generator.generate(prompt, temperature=0, guidance=0.5)
    uncached, b = generator.generate(prompt, temperature=0, guidance=0.5, use_cache=False)
    assert torch.equal(cached, uncached)
    assert a["level_nfe"] == 2 * cfg.n_levels
    assert a["generator_nfe"] == 2 * (cfg.n_levels + 1)


def test_early_exit_uses_partial_hierarchy():
    cfg, _, generator = make_models()
    generator.eval()
    prompt, _ = make_batch(cfg, batch=1)
    tokens, stats = generator.generate(prompt, temperature=0, early_exit=1e-6)
    assert stats["levels_used"] == 2
    assert len(stats["codes"]) == 2
    assert stats["level_nfe"] == 2
    assert stats["generator_nfe"] == 3
    assert stats["decoder_nfe"] == 2
    assert tokens.shape == (1, cfg.ctx_len)


def test_padding_and_refinement_shapes():
    cfg, tokenizer, _ = make_models()
    tokenizer.eval()
    _, target = make_batch(cfg)
    target[0, 5:] = cfg.pad_token
    valid = target != cfg.pad_token
    codes = tokenizer.encode(target)
    decoded, stats = tokenizer.decode(codes, valid=valid, refine_steps=2, refine_threshold=1.1)
    assert decoded.shape == target.shape
    assert stats["confidence"].shape == target.shape
    assert (decoded[~valid] == cfg.pad_token).all()


def test_mbr_selects_consensus_candidate():
    candidates = torch.tensor([
        [[1, 2, 3, 4]], [[1, 2, 3, 4]], [[9, 9, 9, 9]], [[1, 2, 3, 0]],
    ])
    selected = mbr_select(candidates)
    assert torch.equal(selected, torch.tensor([[1, 2, 3, 4]]))
    assert torch.equal(mbr_select(candidates[:1]), candidates[0])


def test_checkpoint_roundtrip_and_prompt_padding():
    cfg, _, generator = make_models()
    generator.eval()
    optimizer = torch.optim.AdamW([p for p in generator.parameters() if p.requires_grad], lr=1e-3)
    prompt, target = make_batch(cfg, batch=1)
    codes = generator.tokenizer.encode(target)
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "generator.pt")
        save_checkpoint(path, generator, optimizer, cfg, 3, Amp(torch.device("cpu")), "generator")
        _, loaded, state = load_generator(path, torch.device("cpu"))
        assert state["step"] == 3
        with torch.no_grad():
            expected = generator(prompt, codes)
            actual = loaded(prompt, codes)
        for a, b in zip(expected, actual):
            torch.testing.assert_close(a, b)
    prompt, raw = prepare_prompt("abc", cfg, torch.device("cpu"))
    assert prompt.shape == (1, cfg.prefix_len)
    assert raw == b"abc"
    assert prompt[0, -4].item() == cfg.bos_token
    assert prompt[0, -3:].tolist() == [97, 98, 99]


def test_resume_rejects_architecture_overrides_before_loading():
    from mariana.ncm.cli import cmd_generator

    cfg, _, generator = make_models()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in generator.parameters() if parameter.requires_grad],
        lr=1e-3,
    )
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "generator.pt")
        save_checkpoint(path, generator, optimizer, cfg, 1, Amp(torch.device("cpu")), "generator")
        args = build_parser().parse_args([
            "generator", "--resume", "--checkpoint", path, "--steps", "2",
            "--dim", "64", "--device", "cpu",
        ])
        with pytest.raises(SystemExit, match="--dim.*--resume"):
            cmd_generator(args)


def test_small_corpus_validation_is_source_disjoint():
    from mariana.ncm.train import build_pairs

    cfg = tiny_cfg()
    train, val = build_pairs(bytes(range(120)), cfg)
    train_rows = {tuple(row.tolist()) for row in train}
    assert train.shape[0] > 0 and val.shape[0] > 0
    assert not any(tuple(row.tolist()) in train_rows for row in val)


def test_empty_training_loader_has_actionable_error():
    from mariana.ncm.train import run_training

    cfg, _, generator = make_models()
    rows = torch.randint(0, 256, (1, cfg.prefix_len + cfg.ctx_len))
    args = build_parser().parse_args([
        "generator", "--steps", "1", "--batch", "2", "--device", "cpu",
    ])
    with tempfile.TemporaryDirectory() as directory:
        with pytest.raises(ValueError, match="training loader is empty.*reduce --batch"):
            run_training(
                generator, rows, rows, cfg, args, torch.device("cpu"),
                os.path.join(directory, "generator.pt"), "generator",
            )


def test_empty_corpus_is_rejected():
    from mariana.ncm.train import build_pairs

    try:
        build_pairs(b"", tiny_cfg())
    except ValueError as e:
        assert "corpus is empty" in str(e)
    else:
        raise AssertionError("empty corpus was accepted")


def test_large_file_corpus_is_memory_mapped_and_trainable():
    import numpy as np
    from mariana.data import build_dataset
    from mariana.ncm.train import build_pairs, run_training

    cfg, _, generator = make_models()
    with tempfile.TemporaryDirectory() as directory:
        corpus = os.path.join(directory, "corpus.txt")
        with open(corpus, "wb") as handle:
            handle.write(bytes(range(256)) * 2500)
        with patch("mariana.data.np.memmap", wraps=np.memmap) as mapped:
            train_ids, val_ids = build_pairs(corpus, cfg)
            drv_train, drv_val = build_dataset(corpus, cfg.ctx_len)
        assert mapped.call_count == 2
        assert train_ids.shape[1] == cfg.prefix_len + cfg.ctx_len
        assert drv_train.shape[1] == cfg.ctx_len + 1
        assert train_ids[0, :4].tolist() == [0, 1, 2, 3]
        assert len(val_ids) > 0 and len(drv_val) > 0

        checkpoint = os.path.join(directory, "generator.pt")
        pause_file = os.path.join(directory, "pause")
        with open(pause_file, "w", encoding="utf-8") as handle:
            handle.write("pause\n")
        args = build_parser().parse_args([
            "generator", "--steps", "100", "--batch", "2", "--eval-every", "100",
            "--log-every", "1", "--pause-file", pause_file,
        ])
        run_training(generator, train_ids, val_ids, cfg, args, torch.device("cpu"), checkpoint, "generator")
        assert torch.load(checkpoint, weights_only=False)["step"] == 1


def test_resume_requires_additional_steps():
    from mariana.ncm.train import run_training

    cfg, _, generator = make_models()
    optimizer = torch.optim.AdamW([p for p in generator.parameters() if p.requires_grad], lr=1e-3)
    prompt, target = make_batch(cfg, batch=4)
    rows = torch.cat((prompt, target), dim=1)
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "generator.pt")
        save_checkpoint(path, generator, optimizer, cfg, 2, Amp(torch.device("cpu")), "generator")
        args = build_parser().parse_args([
            "generator", "--resume", "--checkpoint", path, "--steps", "2", "--batch", "2",
        ])
        try:
            run_training(generator, rows, rows, cfg, args, torch.device("cpu"), path, "generator")
        except ValueError as e:
            assert "must exceed checkpoint step" in str(e)
        else:
            raise AssertionError("exhausted resume was accepted")


def test_candidate_nfe_includes_decoder_and_refiner():
    from mariana.ncm.cli import cmd_gen

    cfg, _, generator = make_models()
    generator.eval()
    args = build_parser().parse_args([
        "gen", "--checkpoint", "unused", "--device", "cpu", "--candidates", "3",
        "--gen-len", "8", "--refine-threshold", "1.1",
    ])
    with patch("mariana.ncm.cli.load_generator", return_value=(cfg, generator, {})), \
            patch("mariana.ncm.cli.print") as output:
        cmd_gen(args)
    message = next(call.args[0] for call in output.call_args_list if call.args[0].startswith("NCM:"))
    assert "15 generator NFE | 3 decoder + 3 refiner NFE" in message


def test_reconstruct_uses_valid_mask_and_rejects_empty_text():
    from mariana.ncm.cli import cmd_reconstruct

    cfg, tokenizer, _ = make_models()
    tokenizer.eval()
    state = {"model": tokenizer.state_dict()}
    args = build_parser().parse_args([
        "reconstruct", "--checkpoint", "unused", "--device", "cpu", "--text", "A",
        "--refine-steps", "0", "--show-levels",
    ])
    with patch("mariana.ncm.cli.load_cfg", return_value=(cfg, state)), \
            patch("mariana.ncm.cli.MultiscaleTokenizer", return_value=tokenizer), \
            patch.object(tokenizer, "decode", wraps=tokenizer.decode) as decode, \
            patch("mariana.ncm.cli.print"):
        cmd_reconstruct(args)
        assert decode.call_count == cfg.n_levels + 1
        for call in decode.call_args_list:
            valid = call.kwargs["valid"]
            assert valid.shape == (1, cfg.ctx_len)
            assert int(valid.sum()) == 1 and bool(valid[0, 0])
        args.text = ""
        try:
            cmd_reconstruct(args)
        except SystemExit as e:
            assert "--text must not be empty" in str(e)
        else:
            raise AssertionError("empty reconstruction text was accepted")


def test_evaluation_refinement_is_deterministic_and_skips_empty_repairs():
    cfg, tokenizer, _ = make_models()
    tokenizer.eval()
    _, target = make_batch(cfg)

    def perfect(logits, valid):
        return torch.where(valid, target, cfg.pad_token)

    with patch.object(tokenizer, "_greedy_bytes", side_effect=perfect), \
            patch.object(tokenizer.refiner, "forward", wraps=tokenizer.refiner.forward) as refiner:
        torch.manual_seed(123)
        first = tokenizer(target)
        torch.manual_seed(456)
        second = tokenizer(target)
    torch.testing.assert_close(first["total"], second["total"])
    assert float(first["refine"]) == 0.0
    assert refiner.call_count == 0


def test_fast_encoding_and_grouped_reconstruction_match_reference_paths():
    cfg, tokenizer, _ = make_models()
    tokenizer.eval()
    _, target = make_batch(cfg)
    z, _ = tokenizer.encode_z(target)
    expected = tokenizer.quantizer._quantize(z, False)[0]
    actual = tokenizer.quantizer.encode(z)
    assert all(torch.equal(a, b) for a, b in zip(expected, actual))

    cumulative = torch.zeros(
        target.shape[0], cfg.ctx_len, cfg.latent_dim, dtype=torch.float,
    )
    sequential = []
    for level, code in enumerate(actual):
        cumulative = tokenizer.quantizer.add_level(cumulative, code, level)
        sequential.append(cumulative)
    grouped = tokenizer.quantizer.reconstruct_prefixes(actual)
    for level, expected_prefix in enumerate(sequential):
        torch.testing.assert_close(grouped[:, level], expected_prefix)


def test_ncm_windows_cover_all_target_positions():
    from mariana.ncm.train import build_pairs, build_targets

    cfg = tiny_cfg()
    train, val = build_pairs(bytes(range(120)), cfg)
    assert train[0].tolist() == list(range(12))
    assert train[1].tolist() == list(range(8, 20))
    assert train[0, cfg.prefix_len:].tolist() == list(range(4, 12))
    assert train[1, cfg.prefix_len:].tolist() == list(range(12, 20))
    train_rows = {tuple(row.tolist()) for row in train}
    assert not any(tuple(row.tolist()) in train_rows for row in val)

    targets, _ = build_targets(bytes(range(120)), cfg)
    assert targets.shape[1] == cfg.ctx_len
    assert targets[0].tolist() == list(range(8))
    assert targets[1].tolist() == list(range(8, 16))


def test_cached_code_training_matches_uncached_training():
    from mariana.ncm.train import run_training

    cfg, _, first = make_models(tiny_cfg(self_condition=0.0, dropout=0.2))
    _, _, second = make_models(cfg)
    second.load_state_dict(first.state_dict())
    prompt, target = make_batch(cfg, batch=4)
    rows = torch.cat((prompt, target), dim=1)
    with tempfile.TemporaryDirectory() as directory:
        for name, model, cached in (("raw", first, False), ("cached", second, True)):
            argv = [
                "generator", "--steps", "1", "--batch", "2", "--eval-every", "1",
                "--log-every", "1", "--device", "cpu",
            ]
            if cached:
                argv.append("--cache-codes")
            args = build_parser().parse_args(argv)
            torch.manual_seed(1234)
            run_training(
                model, rows, rows, cfg, args, torch.device("cpu"),
                os.path.join(directory, name + ".pt"), "generator",
            )
    for a, b in zip(first.parameters(), second.parameters()):
        torch.testing.assert_close(a, b)


def test_optimizer_migration_preserves_state_and_separates_scalar_decay():
    from mariana.ncm.train import load_optimizer_state, make_optimizer

    cfg = tiny_cfg()
    tokenizer = MultiscaleTokenizer(cfg)
    parameters = [p for p in tokenizer.parameters() if p.requires_grad]
    legacy = torch.optim.AdamW(parameters, lr=1e-3, betas=(0.9, 0.95), weight_decay=0.05)
    for parameter in parameters:
        parameter.grad = torch.ones_like(parameter)
    legacy.step()
    expected = legacy.state_dict()

    optimizer = make_optimizer(tokenizer, 3e-4, torch.device("cpu"))
    load_optimizer_state(optimizer, expected)
    assert [group["weight_decay"] for group in optimizer.param_groups] == [0.05, 0.0]
    assert all(p.ndim < 2 for p in optimizer.param_groups[1]["params"])
    assert len(optimizer.state) == len(legacy.state)
    torch.testing.assert_close(
        optimizer.state[parameters[0]]["exp_avg"],
        expected["state"][0]["exp_avg"],
    )


def test_sampling_rejects_invalid_constraints():
    from mariana.ncm.layers import sample_logits

    logits = torch.randn(1, 2, 8)
    for kwargs in (
        {"top_p": 0.0}, {"top_p": -0.1}, {"top_p": float("nan")},
        {"top_k": -1}, {"temperature": -0.1}, {"temperature": float("inf")},
    ):
        with pytest.raises(ValueError):
            sample_logits(logits, **kwargs)
    assert torch.equal(sample_logits(logits, temperature=0), logits.argmax(-1))


def test_integrated_ncm_forwards_performance_flags():
    args = mariana_parse_args([
        "train", "--ncm", "--compile", "--ncm-cache-codes", "--no-gen",
    ])
    with patch("mariana.cli._run_ncm") as dispatch:
        mariana_cmd_train(args)
    argv = dispatch.call_args.args[0]
    assert "--compile" in argv
    assert argv[argv.index("--compile-mode") + 1] == "default"
    assert "--cache-codes" in argv


def test_ncm_stages_share_the_same_source_partition():
    from mariana.data import _tensor_rows
    from mariana.ncm.train import build_pairs, build_targets

    calls = []

    def no_repeat(data, width, stride=None):
        calls.append((width, len(data)))
        ids = torch.tensor(bytearray(data), dtype=torch.uint8)
        return _tensor_rows(ids, width, stride)

    cfg = NCMConfig()
    with patch("mariana.data._repeated_rows", side_effect=no_repeat):
        build_pairs(bytes(1000), cfg)
        build_targets(bytes(1000), cfg)
    assert calls == [(192, 808), (192, 192), (128, 808), (128, 192)]


def test_all_level_vq_matches_reference_objective_and_legacy_default():
    cfg = tiny_cfg(nested_dropout=0.0, refiner_layers=0, all_level_vq=True)
    tokenizer = MultiscaleTokenizer(cfg).train()
    _, target = make_batch(cfg)
    z, _ = tokenizer.encode_z(target)
    selected, full, codes, vq, _ = tokenizer.quantizer(z, 0.0)
    partials = tokenizer.quantizer.reconstruct_prefixes(codes)
    expected = torch.stack([
        cfg.commitment * F.mse_loss(z, partials[:, level].detach())
        + F.mse_loss(z.detach(), partials[:, level])
        for level in range(cfg.n_levels)
    ]).mean()
    torch.testing.assert_close(vq, expected, atol=1e-5, rtol=1e-5)
    assert torch.equal(selected, full)

    values = cfg.to_dict()
    values.pop("all_level_vq")
    legacy = NCMConfig.from_checkpoint({"format": 2, "cfg": values})
    assert legacy.all_level_vq is False
    assert cfg.all_level_vq is True


def test_nested_reconstruction_skips_duplicate_and_batches_selected_decode():
    cfg = tiny_cfg(nested_dropout=0.0, refiner_layers=0)
    tokenizer = MultiscaleTokenizer(cfg).train()
    _, target = make_batch(cfg)
    with patch.object(tokenizer, "decode_latent", wraps=tokenizer.decode_latent) as decode:
        tokenizer(target)
        assert decode.call_count == 1

    tokenizer.cfg.nested_dropout = 0.5
    with patch.object(tokenizer, "decode_latent", wraps=tokenizer.decode_latent) as decode:
        tokenizer(target)
        assert decode.call_count == 1
        assert decode.call_args.args[0].shape[0] == 2 * target.shape[0]


def test_completed_resume_starts_and_preserves_additional_lr_cycle():
    from mariana.ncm.train import make_optimizer, run_training

    cfg = tiny_cfg(nested_dropout=0.0, refiner_layers=0)
    tokenizer = MultiscaleTokenizer(cfg)
    optimizer = make_optimizer(tokenizer, 3e-3, torch.device("cpu"))
    rows = torch.randint(0, 256, (4, cfg.ctx_len))
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "tokenizer.pt")
        pause = os.path.join(directory, "pause")
        save_checkpoint(
            path, tokenizer, optimizer, cfg, 1, Amp(torch.device("cpu")),
            "tokenizer", schedule_steps=1, schedule_start=0,
        )
        with open(pause, "w", encoding="utf-8") as handle:
            handle.write("pause\n")
        args = build_parser().parse_args([
            "tokenizer", "--resume", "--checkpoint", path, "--steps", "3",
            "--batch", "2", "--eval-every", "3", "--log-every", "1",
            "--pause-file", pause,
        ])
        run_training(tokenizer, rows, rows, cfg, args, torch.device("cpu"), path, "tokenizer")
        state = torch.load(path, weights_only=False)
        assert state["step"] == 2
        assert state["schedule_start"] == 1 and state["schedule_steps"] == 3

        os.remove(pause)
        args.steps = 4
        run_training(tokenizer, rows, rows, cfg, args, torch.device("cpu"), path, "tokenizer")
        state = torch.load(path, weights_only=False)
        assert state["step"] == 4
        assert state["schedule_start"] == 1 and state["schedule_steps"] == 4


def test_nonfinite_gradients_abort_unscaled_training():
    from mariana.ncm.train import run_training

    cfg = tiny_cfg(nested_dropout=0.0, refiner_layers=0)
    tokenizer = MultiscaleTokenizer(cfg)
    rows = torch.randint(0, 256, (4, cfg.ctx_len))
    args = build_parser().parse_args([
        "tokenizer", "--steps", "1", "--batch", "2", "--device", "cpu",
    ])
    loss = tokenizer.tokens.weight.sum() * float("nan")
    with tempfile.TemporaryDirectory() as directory, \
            patch.object(tokenizer, "forward", return_value={"total": loss}), \
            pytest.raises(RuntimeError, match="non-finite"):
        run_training(
            tokenizer, rows, rows, cfg, args, torch.device("cpu"),
            os.path.join(directory, "invalid.pt"), "tokenizer",
        )


def test_guidance_ramps_from_coarse_to_fine():
    cfg, _, generator = make_models()
    generator.eval()
    prompt, _ = make_batch(cfg, batch=1)
    choices = []

    def fake_level(prompt, level, cumulative, prefix_cache=None):
        logits = torch.zeros(2, cfg.levels[level], cfg.codebook_size)
        logits[0, :, 1] = 0.1
        logits[1, :, 1] = 1.0
        return logits

    def fake_sample(logits, *args):
        choices.append(logits[0].argmax(-1))
        return torch.zeros(logits.shape[:-1], dtype=torch.long)

    with patch.object(generator, "forward_level", side_effect=fake_level), \
            patch("mariana.ncm.generator.sample_logits", side_effect=fake_sample):
        generator.generate(prompt, temperature=0, guidance=1.0, use_cache=False)
        assert choices[0][0].item() == 1
        assert all(int(choice.sum()) == 0 for choice in choices[1:])
        choices.clear()
        generator.generate(
            prompt, temperature=0, guidance=1.0, use_cache=False,
            guidance_ramp=False,
        )
        assert all(int(choice.sum()) == 0 for choice in choices)


def test_integrated_ncm_forwards_flex_attention():
    args = mariana_parse_args(["train", "--ncm", "--flex", "--no-gen"])
    with patch("mariana.cli._run_ncm") as dispatch:
        mariana_cmd_train(args)
    assert "--flex" in dispatch.call_args.args[0]


def test_flex_rejects_unsupported_head_dimensions_early():
    _, _, generator = make_models()
    with pytest.raises(ValueError, match="head dimension >= 16"):
        generator.enable_flex(torch.device("cpu"))
