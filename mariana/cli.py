"""Mariana command line interface.

Subcommands
-----------
    python -m mariana train  [--steps N] [--batch B] [--device auto] ...
    python -m mariana train  --ncm [--tokenizer-steps N] [--steps N]
    python -m mariana gen    --prompt "KING: " [--gen-len 50] [--latent] [--ncm]
    python -m mariana ncm    tokenizer|train|generator|gen|reconstruct ...
    python -m mariana studio [--host HOST] [--port PORT] [--no-browser]
    python -m mariana think  --prompt "KING: "       # show the private code
    python -m mariana bench  [--gen-len 50]          # hybrid vs AR speed test
    python -m mariana export --format gguf|onnx|safetensors|all [--out DIR]
    python -m mariana setup [--python 3.14]         # pyenv + venv + core requirements
    python -m mariana update [--check]              # compare with GitHub releases, then update

Backward compatibility: ``python mariana.py --steps 3000`` (no subcommand)
still works and is treated as ``train``.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

from .config import Config
from .device import describe, pick_device

DEFAULT_CHECKPOINT = os.path.join("trained-model", "mariana.pt")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def _add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--ctx", type=int, default=256)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--n-pre", type=int, default=2)
    p.add_argument("--n-core", type=int, default=2)
    p.add_argument("--n-post", type=int, default=2)
    p.add_argument("--loops", type=int, default=None, help="max_loops override")
    p.add_argument("--block-size", type=int, default=32)
    p.add_argument("--moe", action="store_true", help="enable sparse-MoE FFN")
    p.add_argument("--no-grouped-moe", action="store_true",
                   help="use the reference per-expert kernels instead of grouped CUDA AMP GEMMs")
    p.add_argument("--no-mtp", action="store_true", help="disable MTP head")
    p.add_argument("--no-adaln", action="store_true", help="disable adaLN conditioning")
    p.add_argument("--latent", action="store_true",
                   help="enable SAINT adaptive hierarchical continuous/discrete thought")
    p.add_argument("--latent-legacy", action="store_true",
                   help="use the original fixed-length discrete-only latent code")
    p.add_argument("--latent-symbols", type=int, default=256,
                   help="size of the model's private alphabet")
    p.add_argument("--thoughts", type=int, default=8,
                   help="maximum latent symbols per context (SAINT can halt earlier)")
    p.add_argument("--latent-min-thoughts", type=int, default=2,
                   help="minimum content symbols before SAINT may emit EOS")
    p.add_argument("--latent-block-size", type=int, default=2,
                   help="child thoughts summarized by each SAINT parent symbol")
    p.add_argument("--latent-continuous-slots", type=int, default=2,
                   help="variational continuous vectors per SAINT thought")
    p.add_argument("--latent-refine-steps", type=int, default=2,
                   help="self-attention reasoning passes over continuous slots")
    p.add_argument("--latent-window", type=int, default=32,
                   help="raw bytes kept beside the latent summary")
    p.add_argument("--latent-window-dropout", type=float, default=0.25,
                   help="probability of masking each example's local window")
    p.add_argument("--latent-recall-prob", type=float, default=0.25,
                   help="fraction of latent batches using an associative recall task")
    p.add_argument("--latent-recall-pairs", type=int, default=2,
                   help="key/value pairs to memorize in recall/grammar tasks")
    p.add_argument("--latent-grammar-prob", type=float, default=0.25,
                   help="fraction of SAINT batches using compositional operator tasks")
    p.add_argument("--w-latent", type=float, default=0.5,
                   help="weight of the compressed-context loss")
    p.add_argument("--w-latent-rl", type=float, default=0.1,
                   help="weight of baseline-centered policy gradients on thought actions")
    p.add_argument("--w-latent-grammar", type=float, default=0.1,
                   help="weight of next-symbol, composition, parent, and order objectives")
    p.add_argument("--w-latent-length", type=float, default=0.02,
                   help="return penalty that makes easy contexts use fewer thoughts")
    p.add_argument("--no-latent-diffusion", action="store_true",
                   help="disable SAINT cross-attention in the diffusion draft")


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    p.add_argument("--device", type=str, default="auto",
                   help="auto|cuda|xpu|mps|cpu (auto prefers GPU, falls back to CPU)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mariana", description="Mariana tiny hybrid Diffusion-AR LM (DRV-LM)"
    )
    sub = parser.add_subparsers(dest="cmd")

    # -- train ---------------------------------------------------------------
    p = sub.add_parser("train", help="train the hybrid model")
    _add_common_args(p)
    _add_model_args(p)
    p.add_argument("--ncm", action="store_true",
                   help="train Next-Concept Modeling instead of DRV-LM (both stages)")
    p.add_argument("--ncm-tokenizer", default=os.path.join("trained-model", "mariana-ncm-tokenizer.pt"),
                   help="NCM stage-1 checkpoint")
    p.add_argument("--ncm-tokenizer-steps", type=int, default=1000,
                   help="NCM stage-1 updates; --steps controls stage 2")
    p.add_argument("--ncm-tokenizer-batch", type=int)
    p.add_argument("--ncm-tokenizer-grad-accum", type=int)
    p.add_argument("--ncm-generator-lr", type=float, default=3e-4)
    p.add_argument("--ncm-cache-codes", action="store_true",
                   help="cache frozen NCM codes for repeated-epoch generator training")
    p.add_argument("--ncm-prefix", type=int, default=64, help="NCM prompt byte length")
    p.add_argument("--ncm-levels", help="comma-separated coarse-to-fine code lengths")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--init-from", metavar="CHECKPOINT",
                   help="initialize weights from a DRV-LM checkpoint; resets optimizer and step")
    p.add_argument("--freeze-layers", type=int, default=0,
                   help="freeze the leading N transformer blocks (fine-tuning)")
    p.add_argument("--freeze-embeddings", action="store_true",
                   help="freeze token/time/region embeddings (fine-tuning)")
    p.add_argument("--pause-file")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--compile-mode", type=str, default="default",
                   choices=["default", "max-autotune-no-cudagraphs"],
                   help="torch.compile mode; CUDA-graph modes are unsafe for the "
                        "dynamic hybrid training step")
    p.add_argument("--flex", action="store_true", help="use FlexAttention kernel")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--fast", dest="fast_train", action="store_true",
        help="Sprint-DRV single-pass training (prefix AR + diffusion + verifier + MTP)",
    )
    mode.add_argument(
        "--legacy", dest="fast_train", action="store_false",
        help="legacy two-pass full-sequence AR + alignment training",
    )
    p.set_defaults(fast_train=None)
    p.add_argument("--amp", action="store_true", help="force AMP on CPU too")
    p.add_argument("--data", type=str, default=None,
                   help="training corpus file (default: built-in demo text)")
    p.add_argument("--anneal-frac", type=float, default=0.5,
                   help="fraction of training over which causal-mask annealing decays")
    p.add_argument("--gen-len", type=int, default=50)
    p.add_argument("--prompt", type=str, default="KING RICHARD III:\n")
    p.add_argument("--no-gen", action="store_true", help="skip final demo generation")
    p.add_argument("--bench", action="store_true", help="run AR baseline after training")

    # -- gen ------------------------------------------------------------------
    p = sub.add_parser("gen", help="generate from a checkpoint")
    _add_common_args(p)
    p.add_argument("--ncm", action="store_true", help="use an NCM generator checkpoint")
    p.add_argument("--prompt", type=str, default="KING RICHARD III:\n")
    p.add_argument("--gen-len", type=int, default=50)
    p.add_argument("--draft-len", type=int, default=None)
    p.add_argument("--refine-steps", type=int, default=2)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--draft-temperature", type=float, default=None,
                   help="draft sampling temperature (0 = greedy drafting; "
                        "default: match --temperature)")
    p.add_argument("--ar", action="store_true", help="plain AR decoding (no diffusion)")
    p.add_argument("--no-adaptive", action="store_true", help="fixed draft length")
    p.add_argument("--latent", action="store_true",
                   help="decode from the learned private code instead of the full prompt")
    p.add_argument("--show-code", action="store_true",
                   help="print the latent symbol sequence")
    p.add_argument("--code-temperature", type=float, default=0.0,
                   help="latent-code sampling temperature (0 = greedy)")

    # -- think ----------------------------------------------------------------
    p = sub.add_parser("think", help="inspect the model's private latent code")
    _add_common_args(p)
    p.add_argument("--prompt", type=str, default="KING RICHARD III:\n")
    p.add_argument("--gen-len", type=int, default=80)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--code-temperature", type=float, default=0.0,
                   help="latent-code sampling temperature (0 = greedy)")

    # -- bench ----------------------------------------------------------------
    p = sub.add_parser("bench", help="hybrid vs AR speed comparison")
    _add_common_args(p)
    p.add_argument("--prompt", type=str, default="KING RICHARD III:\n")
    p.add_argument("--gen-len", type=int, default=50)
    p.add_argument("--refine-steps", type=int, default=2)
    p.add_argument("--draft-temperature", type=float, default=None,
                   help="draft sampling temperature (0 = greedy drafting)")

    # -- export ---------------------------------------------------------------
    p = sub.add_parser("export", help="convert checkpoint to GGUF/ONNX/safetensors")
    _add_common_args(p)
    p.add_argument("--format", type=str, default="gguf",
                   choices=["gguf", "onnx", "safetensors", "all"])
    p.add_argument("--out", type=str, default="exports")
    p.add_argument("--dtype", type=str, default="f16", choices=["f16", "f32", "bf16"])
    p.add_argument("--name", type=str, default="mariana")
    p.add_argument("--no-verify", action="store_true")
    p.add_argument("--raw", action="store_true",
                   help="safetensors: save raw Mariana weights instead of LLaMA-converted")

    for name in ("studio", "gui"):
        p = sub.add_parser(
            name, help="open Mariana Studio in the browser" if name == "studio"
            else "alias of studio (kept for existing scripts)",
        )
        p.add_argument("--host", default="127.0.0.1",
                       help="interface to listen on (keep the default; Studio has no authentication)")
        p.add_argument("--port", type=int, default=8765)
        p.add_argument("--no-browser", action="store_true")

    p = sub.add_parser("ncm", add_help=False, help="advanced Next-Concept Modeling commands")
    p.add_argument("ncm_args", nargs=argparse.REMAINDER)

    p = sub.add_parser("setup", add_help=False,
                       help="create a pyenv Python and virtualenv, then install requirements")
    p.add_argument("setup_args", nargs=argparse.REMAINDER)

    p = sub.add_parser("update", add_help=False,
                       help="check GitHub releases and update this git checkout")
    p.add_argument("update_args", nargs=argparse.REMAINDER)

    return parser


def parse_args(argv=None) -> argparse.Namespace:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and (argv[0] in {"--gui", "--studio"} or argv[:2] in (["train", "--gui"], ["train", "--studio"])):
        argv = ["gui"] + (argv[1:] if argv[0] in {"--gui", "--studio"} else argv[2:])
    if argv and argv[0] == "ncm":
        return argparse.Namespace(cmd="ncm", ncm_args=argv[1:])
    if argv and argv[0] == "setup":
        return argparse.Namespace(cmd="setup", setup_args=argv[1:])
    if argv and argv[0] == "update":
        return argparse.Namespace(cmd="update", update_args=argv[1:])
    cmds = {"train", "gen", "think", "bench", "export", "ncm", "setup", "update", "gui", "studio"}
    if argv and argv[0] not in cmds and not argv[0].startswith("-h") and argv[0] != "--help":
        argv = ["train"] + argv  # legacy flag style -> train
    elif not argv:
        argv = ["train"]
    return build_parser().parse_args(argv)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def load_cfg_and_model(args, device):
    from .model import MarianaLM
    from .train import load_checkpoint

    if not os.path.exists(args.checkpoint):
        raise SystemExit(f"Checkpoint {args.checkpoint} not found. Train first.")
    ckpt = load_checkpoint(args.checkpoint, map_location="cpu")
    cfg = Config.from_dict(ckpt["cfg"])
    if getattr(args, "loops", None) is not None:
        cfg.max_loops = args.loops
    model = MarianaLM(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint from step {ckpt.get('step', '?')}")
    return cfg, model, ckpt


def cfg_from_args(args) -> Config:
    cfg = Config(
        ctx_len=args.ctx,
        dim=args.dim,
        n_heads=args.heads,
        n_pre=args.n_pre,
        n_core=args.n_core,
        n_post=args.n_post,
        block_size=args.block_size,
        use_moe=args.moe,
        grouped_moe=not args.no_grouped_moe,
        use_mtp=not args.no_mtp,
        use_adaln=not args.no_adaln,
        use_latent=args.latent,
        n_latent_symbols=args.latent_symbols,
        n_thoughts=args.thoughts,
        latent_local_window=args.latent_window,
        latent_window_dropout=args.latent_window_dropout,
        latent_recall_prob=args.latent_recall_prob,
        latent_recall_pairs=args.latent_recall_pairs,
        w_latent=args.w_latent,
        latent_saint=args.latent and not args.latent_legacy,
        latent_min_thoughts=args.latent_min_thoughts,
        latent_block_size=args.latent_block_size,
        latent_continuous_slots=args.latent_continuous_slots,
        latent_refine_steps=args.latent_refine_steps,
        latent_grammar_prob=args.latent_grammar_prob,
        latent_diffusion=not args.no_latent_diffusion,
        w_latent_rl=args.w_latent_rl,
        w_latent_grammar=args.w_latent_grammar,
        w_latent_length=args.w_latent_length,
        fast_train=bool(getattr(args, "fast_train", False)),
    )
    if args.loops is not None:
        cfg.max_loops = args.loops
    return cfg


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
NCM_GENERATOR_CHECKPOINT = os.path.join("trained-model", "mariana-ncm.pt")


def _run_ncm(argv):
    from .ncm.cli import main

    main(argv)


def _cmd_ncm_train(args):
    if (args.moe or args.no_grouped_moe or args.latent or args.bench
            or args.no_mtp or args.no_adaln or args.fast_train is not None):
        raise SystemExit("DRV-LM-only training flags cannot be combined with --ncm.")
    if getattr(args, "init_from", None) or getattr(args, "freeze_layers", 0) or getattr(args, "freeze_embeddings", False):
        raise SystemExit(
            "Fine-tune options for NCM use: python mariana.py ncm generator "
            "--init-from BASE --tokenizer TOKENIZER --checkpoint OUTPUT ..."
        )
    checkpoint = NCM_GENERATOR_CHECKPOINT if args.checkpoint == DEFAULT_CHECKPOINT else args.checkpoint
    argv = [
        "train", "--checkpoint", checkpoint, "--tokenizer", args.ncm_tokenizer,
        "--tokenizer-steps", str(args.ncm_tokenizer_steps),
        "--generator-lr", str(args.ncm_generator_lr),
        "--steps", str(args.steps), "--batch", str(args.batch),
        "--grad-accum", str(args.grad_accum), "--lr", str(args.lr),
        "--warmup", str(args.warmup), "--eval-every", str(args.eval_every),
        "--device", args.device, "--ctx", str(args.ctx), "--dim", str(args.dim),
        "--heads", str(args.heads), "--prefix", str(args.ncm_prefix),
        "--prompt", args.prompt, "--gen-len", str(args.gen_len),
    ]
    if args.ncm_levels:
        argv += ["--levels", args.ncm_levels]
    if args.ncm_tokenizer_batch is not None:
        argv += ["--tokenizer-batch", str(args.ncm_tokenizer_batch)]
    if args.ncm_tokenizer_grad_accum is not None:
        argv += ["--tokenizer-grad-accum", str(args.ncm_tokenizer_grad_accum)]
    if args.data:
        argv += ["--data", args.data]
    if args.resume:
        argv.append("--resume")
    if args.pause_file:
        argv += ["--pause-file", args.pause_file]
    if args.amp:
        argv.append("--amp")
    if args.compile:
        argv += ["--compile", "--compile-mode", args.compile_mode]
    if args.flex:
        argv.append("--flex")
    if args.ncm_cache_codes:
        argv.append("--cache-codes")
    if args.no_gen:
        argv.append("--no-gen")
    _run_ncm(argv)


def _cmd_ncm_gen(args):
    if (args.ar or args.latent or args.show_code or args.no_adaptive
            or args.draft_len is not None or args.draft_temperature is not None
            or args.code_temperature != 0.0):
        raise SystemExit("DRV-LM-only generation flags cannot be combined with --ncm.")
    checkpoint = NCM_GENERATOR_CHECKPOINT if args.checkpoint == DEFAULT_CHECKPOINT else args.checkpoint
    argv = [
        "gen", "--checkpoint", checkpoint, "--device", args.device,
        "--prompt", args.prompt, "--gen-len", str(args.gen_len),
        "--temperature", str(args.temperature), "--top-p", str(args.top_p),
        "--refine-steps", str(args.refine_steps),
    ]
    _run_ncm(argv)


def cmd_ncm(args):
    _run_ncm(args.ncm_args)


def cmd_studio(args):
    from .studio import run

    run(args.host, args.port, open_browser=not args.no_browser)


def cmd_train(args) -> None:
    if args.ncm:
        return _cmd_ncm_train(args)
    from .generate import ar_generate, draft_refine_verify
    from .latent import format_code, generate_latent
    from .train import load_checkpoint, train

    device = pick_device(args.device)
    print(describe(device))
    if device.type == "cuda":
        vram = torch.cuda.get_device_properties(device).total_memory / 1024**3
        if vram < 6:
            print("Low VRAM detected. Consider --batch 4 --grad-accum 4 --ctx 128.")

    # when resuming, prefer the checkpoint config so dimensions match
    if getattr(args, "init_from", None):
        if not os.path.exists(args.init_from):
            raise SystemExit(f"Base checkpoint {args.init_from} not found.")
        base = load_checkpoint(args.init_from, map_location="cpu")
        if base.get("stage") not in {None, "drv"}:
            raise SystemExit("--init-from must point to a DRV-LM checkpoint.")
        cfg = Config.from_dict(base["cfg"])
        if args.loops is not None:
            cfg.max_loops = args.loops
        if args.fast_train is not None:
            cfg.fast_train = args.fast_train
    elif args.resume and os.path.exists(args.checkpoint):
        ckpt = load_checkpoint(args.checkpoint, map_location="cpu")
        cfg = Config.from_dict(ckpt["cfg"])
        if args.loops is not None:
            cfg.max_loops = args.loops
        if args.latent and not cfg.use_latent:
            print(
                "Checkpoint was trained without --latent; keeping the latent "
                "code disabled. Start a new checkpoint to enable it."
            )
        if args.fast_train is not None:
            cfg.fast_train = args.fast_train
    else:
        cfg = cfg_from_args(args)

    model = train(args, cfg, device)

    if not args.no_gen and not (args.pause_file and os.path.exists(args.pause_file)):
        from .device import empty_cache

        empty_cache(device)
        print("\n--- Final generation demo (Draft-Refine-Verify) ---")
        model.eval()
        out, acc, tps, elapsed = draft_refine_verify(
            model, args.prompt.encode(), max_new_tokens=args.gen_len,
            device=device, cfg=cfg,
        )
        print(f"Hybrid: {tps:.1f} tok/s, acceptance {acc:.2%}, time {elapsed:.2f}s")
        print(out.decode("utf-8", errors="replace"))

        if cfg.use_latent:
            print("\n--- Latent code demo (full context -> private symbols -> bytes) ---")
            lat_out, symbols, stats = generate_latent(
                model, args.prompt.encode(), cfg, device,
                max_new_tokens=args.gen_len,
            )
            print(f"Code: {format_code(symbols)}")
            print(
                f"{stats['context_bytes']} context bytes -> "
                f"{stats['n_thoughts']} symbols "
                f"({stats['bytes_per_symbol']:.1f} bytes/symbol, "
                f"{stats['unique']} unique)"
            )
            print(lat_out.decode("utf-8", errors="replace"))

    if args.bench:
        out_ar, tps_ar, elapsed_ar = ar_generate(
            model, args.prompt.encode(), max_new_tokens=args.gen_len,
            device=device, cfg=cfg,
        )
        print(f"\nAR baseline: {tps_ar:.1f} tok/s, time {elapsed_ar:.2f}s")
        print(out_ar.decode("utf-8", errors="replace"))


def cmd_gen(args) -> None:
    if args.ncm:
        return _cmd_ncm_gen(args)
    from .generate import ar_generate, draft_refine_verify
    from .latent import format_code, generate_latent

    device = pick_device(args.device)
    print(describe(device))
    cfg, model, _ = load_cfg_and_model(args, device)

    if args.latent and not cfg.use_latent:
        raise SystemExit(
            "This checkpoint has no latent code. Train one with "
            "`python -m mariana train --latent`."
        )

    if args.latent:
        out, symbols, stats = generate_latent(
            model, args.prompt.encode(), cfg, device,
            max_new_tokens=args.gen_len,
            temperature=args.temperature, top_p=args.top_p,
            code_temperature=args.code_temperature,
        )
        if args.show_code:
            print(f"\n--- Private code ({stats['unique']}/{stats['n_thoughts']} unique) ---")
            print(format_code(symbols))
            if "parent_symbols" in stats:
                print(f"Parent code: {format_code(stats['parent_symbols'])}")
        print(
            f"\n--- Latent-code AR ({stats['context_bytes']} bytes -> "
            f"{stats['n_thoughts']} symbols) ---"
        )
        max_context = max(1, cfg.ctx_len - cfg.n_thoughts - 8)
        used_prompt = args.prompt.encode()[-max_context:]
        print(
            used_prompt.decode("utf-8", errors="replace")
            + out.decode("utf-8", errors="replace")
        )
        return

    if args.ar:
        out, tps, elapsed = ar_generate(
            model, args.prompt.encode(), max_new_tokens=args.gen_len,
            temperature=args.temperature, top_p=args.top_p, device=device, cfg=cfg,
        )
        print(f"\n--- AR generation ({tps:.1f} tok/s) ---")
    else:
        out, acc, tps, elapsed = draft_refine_verify(
            model, args.prompt.encode(), max_new_tokens=args.gen_len,
            draft_len=args.draft_len, n_steps=args.refine_steps,
            temperature=args.temperature, top_p=args.top_p, device=device, cfg=cfg,
            adaptive_draft=not args.no_adaptive,
            draft_temperature=args.draft_temperature,
        )
        print(f"\n--- Hybrid generation ({tps:.1f} tok/s, accept {acc:.2%}) ---")
    print(out.decode("utf-8", errors="replace"))


def cmd_think(args) -> None:
    """Print the private symbol sequence and the continuation it encodes."""
    from .latent import format_code, generate_latent

    device = pick_device(args.device)
    print(describe(device))
    cfg, model, _ = load_cfg_and_model(args, device)
    if not cfg.use_latent:
        raise SystemExit(
            "This checkpoint has no latent code. Train one with "
            "`python -m mariana train --latent`."
        )

    out, symbols, stats = generate_latent(
        model, args.prompt.encode(), cfg, device,
        max_new_tokens=args.gen_len,
        temperature=args.temperature, top_p=args.top_p,
        code_temperature=args.code_temperature,
    )
    print("\n--- Private latent code ---")
    print(format_code(symbols))
    if "parent_symbols" in stats:
        print(f"Parent code: {format_code(stats['parent_symbols'])}")
    print(
        f"{stats['context_bytes']} context bytes -> {stats['n_thoughts']} symbols "
        f"({stats['bytes_per_symbol']:.1f} bytes/symbol, "
        f"{stats['bits_per_symbol']:.1f} bits/symbol, "
        f"{stats['unique']} unique)"
    )
    print("\n--- Decoded continuation ---")
    max_context = max(1, cfg.ctx_len - cfg.n_thoughts - 8)
    used_prompt = args.prompt.encode()[-max_context:]
    print(
        used_prompt.decode("utf-8", errors="replace")
        + out.decode("utf-8", errors="replace")
    )


def cmd_bench(args) -> None:
    from .generate import ar_generate, draft_refine_verify

    device = pick_device(args.device)
    print(describe(device))
    cfg, model, _ = load_cfg_and_model(args, device)
    prompt = args.prompt.encode()

    out, acc, tps, el = draft_refine_verify(
        model, prompt, max_new_tokens=args.gen_len, device=device, cfg=cfg,
        n_steps=args.refine_steps, draft_temperature=args.draft_temperature,
    )
    print(f"Hybrid DRV: {tps:.1f} tok/s | acceptance {acc:.2%} | {el:.2f}s")
    print(out.decode("utf-8", errors="replace"))
    out_ar, tps_ar, el_ar = ar_generate(
        model, prompt, max_new_tokens=args.gen_len, device=device, cfg=cfg
    )
    print(f"\nAR baseline: {tps_ar:.1f} tok/s | {el_ar:.2f}s")
    print(out_ar.decode("utf-8", errors="replace"))
    print(f"\nSpeedup: {tps / max(tps_ar, 1e-9):.2f}x")


def cmd_export(args) -> None:
    device = torch.device("cpu")  # export is device-independent; keep it CPU
    cfg, model, ckpt = load_cfg_and_model(args, device)
    sd = ckpt["model"]
    if cfg.use_latent:
        print(
            "Note: GGUF/ONNX and LLaMA-format safetensors export the lossless "
            "AR path and omit the latent modules; --raw safetensors keeps them. "
            "The AR path itself is unchanged by the latent code."
        )
    os.makedirs(args.out, exist_ok=True)
    fmts = ["gguf", "onnx", "safetensors"] if args.format == "all" else [args.format]

    for fmt in fmts:
        if fmt == "gguf":
            from .export.gguf_export import export_gguf

            export_gguf(sd, cfg, os.path.join(args.out, f"{args.name}-{args.dtype}.gguf"),
                        dtype=args.dtype, name=args.name)
        elif fmt == "onnx":
            from .export.onnx_export import export_onnx

            export_onnx(model, cfg, os.path.join(args.out, f"{args.name}.onnx"),
                        verify=not args.no_verify)
        elif fmt == "safetensors":
            from .export.safetensors import export_safetensors

            suffix = "raw" if args.raw else "llama"
            export_safetensors(sd, cfg, os.path.join(args.out, f"{args.name}-{suffix}.safetensors"),
                               llama_compat=not args.raw)


def cmd_setup(args) -> None:
    from .setup import main as setup_main

    code = setup_main(list(getattr(args, "setup_args", [])))
    if code:
        raise SystemExit(code)


def cmd_update(args) -> None:
    from .update import main as update_main

    code = update_main(list(getattr(args, "update_args", [])))
    if code:
        raise SystemExit(code)


def main(argv=None) -> None:
    args = parse_args(argv)
    {
        "train": cmd_train,
        "gen": cmd_gen,
        "think": cmd_think,
        "bench": cmd_bench,
        "export": cmd_export,
        "ncm": cmd_ncm,
        "setup": cmd_setup,
        "update": cmd_update,
        "gui": cmd_studio,
        "studio": cmd_studio,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
