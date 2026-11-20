from __future__ import annotations

import argparse
import os
import random

import torch

from ..device import describe, pick_device
from .config import NCMConfig, default_levels
from .generator import NCMGenerator
from .layers import mbr_select
from .tokenizer import MultiscaleTokenizer
from .train import load_checkpoint, train_generator, train_tokenizer

TOKENIZER_PATH = os.path.join("trained-model", "mariana-ncm-tokenizer.pt")
GENERATOR_PATH = os.path.join("trained-model", "mariana-ncm.pt")


def add_training(p, checkpoint):
    p.add_argument("--data", help="training corpus file (default: built-in demo text)")
    p.add_argument("--checkpoint", default=checkpoint)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--init-from", metavar="CHECKPOINT",
                   help="initialize from a checkpoint of the same stage; resets optimizer and step")
    p.add_argument("--freeze-layers", type=int, default=0,
                   help="freeze the leading N generator blocks")
    p.add_argument("--freeze-embeddings", action="store_true",
                   help="freeze generator prompt/level embeddings")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--pause-file")


def add_compilation(p):
    p.add_argument("--compile", action="store_true", help="compile the generator forward pass with PyTorch")
    p.add_argument("--flex", action="store_true", help="use sparse FlexAttention for hierarchy masks")
    p.add_argument(
        "--compile-mode", default="default",
        choices=["default", "max-autotune-no-cudagraphs"],
        help="torch.compile mode without dynamic-shape CUDA graphs",
    )


def add_architecture(p):
    p.add_argument("--ctx", type=int, default=128)
    p.add_argument("--prefix", type=int, default=64)
    p.add_argument("--levels", help="comma-separated coarse-to-fine code lengths")
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--codebook-size", type=int, default=512)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--encoder-layers", type=int, default=2)
    p.add_argument("--decoder-layers", type=int, default=2)
    p.add_argument("--refiner-layers", type=int, default=1)
    p.add_argument("--generator-layers", type=int, default=6)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--nested-dropout", type=float, default=0.5)
    p.add_argument("--corruption", type=float, default=0.5)
    p.add_argument("--self-condition", type=float, default=0.25)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="mariana ncm",
        description="Improved byte-level Next-Concept Modeling for Mariana",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("tokenizer", aliases=["tokenize"], help="train stage 1: multiscale residual VQ tokenizer")
    add_training(p, TOKENIZER_PATH)
    add_architecture(p)

    p = sub.add_parser("train", help="train both NCM stages in one command")
    add_training(p, GENERATOR_PATH)
    add_compilation(p)
    add_architecture(p)
    p.add_argument("--tokenizer", default=TOKENIZER_PATH)
    p.add_argument("--tokenizer-steps", type=int, default=1000)
    p.add_argument("--tokenizer-batch", type=int)
    p.add_argument("--tokenizer-grad-accum", type=int)
    p.add_argument("--generator-lr", type=float, default=3e-4)
    p.add_argument("--cache-codes", action="store_true", help="cache frozen tokenizer codes for repeated epochs")
    p.add_argument("--prompt", default="KING RICHARD III:\n")
    p.add_argument("--gen-len", type=int, default=50)
    p.add_argument("--no-gen", action="store_true")

    p = sub.add_parser("generator", help="train only stage 2: frozen-tokenizer next-concept generator")
    add_training(p, GENERATOR_PATH)
    add_compilation(p)
    p.set_defaults(lr=3e-4)
    p.add_argument("--tokenizer", default=TOKENIZER_PATH)
    p.add_argument("--cache-codes", action="store_true", help="cache frozen tokenizer codes for repeated epochs")
    p.add_argument("--dim", type=int)
    p.add_argument("--heads", type=int)
    p.add_argument("--generator-layers", type=int)
    p.add_argument("--dropout", type=float)
    p.add_argument("--corruption", type=float)
    p.add_argument("--self-condition", type=float)

    p = sub.add_parser("gen", aliases=["generate"], help="generate a coarse-to-fine continuation")
    p.add_argument("--checkpoint", default=GENERATOR_PATH)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prompt", default="KING RICHARD III:\n")
    p.add_argument("--gen-len", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--guidance", type=float, default=0.0)
    p.add_argument("--constant-guidance", action="store_true", help="disable the coarse-to-fine guidance ramp")
    p.add_argument("--candidates", type=int, default=1, help="sample N candidates and select the MBR medoid")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--early-exit", type=float, default=0.0, help="decoder confidence threshold in [0,1]; 0 uses every level")
    p.add_argument("--refine-steps", type=int, default=1)
    p.add_argument("--refine-threshold", type=float, default=0.8)
    p.add_argument("--show-levels", action="store_true")

    p = sub.add_parser("reconstruct", help="measure tokenizer reconstruction on text")
    p.add_argument("--checkpoint", default=TOKENIZER_PATH)
    p.add_argument("--device", default="auto")
    p.add_argument("--text", default="Mariana learns coarse concepts before fine details.")
    p.add_argument("--refine-steps", type=int, default=1)
    p.add_argument("--refine-threshold", type=float, default=0.8)
    p.add_argument("--show-levels", action="store_true")
    return parser


def cfg_from_args(args):
    if args.levels:
        levels = tuple(int(v.strip()) for v in args.levels.split(",") if v.strip())
    else:
        levels = default_levels(args.ctx)
    return NCMConfig(
        ctx_len=args.ctx, prefix_len=args.prefix, levels=levels,
        latent_dim=args.latent_dim, codebook_size=args.codebook_size,
        dim=args.dim, n_heads=args.heads,
        encoder_layers=args.encoder_layers, decoder_layers=args.decoder_layers,
        refiner_layers=args.refiner_layers, generator_layers=args.generator_layers,
        dropout=args.dropout, nested_dropout=args.nested_dropout,
        corruption=args.corruption, self_condition=args.self_condition,
    )


def load_cfg(path, expected_stage=None):
    state = load_checkpoint(path)
    if expected_stage and state.get("stage") != expected_stage:
        raise SystemExit(f"{path} is a {state.get('stage')} checkpoint, expected {expected_stage}")
    return NCMConfig.from_checkpoint(state), state


def load_generator(path, device):
    cfg, state = load_cfg(path, "generator")
    tokenizer = MultiscaleTokenizer(cfg)
    model = NCMGenerator(tokenizer, cfg)
    model.load_state_dict(state["model"])
    return cfg, model.to(device).eval(), state


def prepare_prompt(text, cfg, device):
    raw = text.encode("utf-8")[-cfg.prefix_len:]
    if len(raw) < cfg.prefix_len:
        ids = [cfg.pad_token] * (cfg.prefix_len - len(raw) - 1) + [cfg.bos_token] + list(raw)
    else:
        ids = list(raw)
    return torch.tensor([ids], dtype=torch.long, device=device), raw


def as_text(tokens, length=None):
    values = tokens.reshape(-1).tolist()
    if length is not None:
        values = values[:length]
    return bytes(v for v in values if 0 <= v < 256).decode("utf-8", errors="replace")


def cmd_tokenizer(args):
    if args.resume and getattr(args, "init_from", None):
        raise SystemExit("--resume and --init-from are mutually exclusive")
    if args.resume and os.path.exists(args.checkpoint):
        cfg, _ = load_cfg(args.checkpoint, "tokenizer")
    elif getattr(args, "init_from", None):
        cfg, _ = load_cfg(args.init_from, "tokenizer")
    else:
        cfg = cfg_from_args(args)
    device = pick_device(args.device)
    print(describe(device))
    train_tokenizer(args, cfg, device)


def _derived_args(args, **updates):
    values = vars(args).copy()
    values.update(updates)
    return argparse.Namespace(**values)


def _training_demo(args):
    if args.no_gen or (args.pause_file and os.path.exists(args.pause_file)):
        return
    demo = build_parser().parse_args(["gen"])
    demo.checkpoint = args.checkpoint
    demo.device = args.device
    demo.prompt = args.prompt
    demo.gen_len = args.gen_len
    cmd_gen(demo)


def cmd_train_all(args):
    overrides = {
        name: None for name in (
            "dim", "heads", "generator_layers", "dropout", "corruption", "self_condition"
        )
    }
    generator_args = _derived_args(
        args, lr=args.generator_lr, **overrides,
    )
    if args.resume and os.path.exists(args.checkpoint):
        cmd_generator(generator_args)
    else:
        tokenizer_args = _derived_args(
            args, checkpoint=args.tokenizer, steps=args.tokenizer_steps,
            batch=getattr(args, "tokenizer_batch", None) or args.batch,
            grad_accum=getattr(args, "tokenizer_grad_accum", None) or args.grad_accum,
            compile=False, flex=False,
        )
        cmd_tokenizer(tokenizer_args)
        if args.pause_file and os.path.exists(args.pause_file):
            return
        generator_args.resume = False
        cmd_generator(generator_args)
    _training_demo(args)


def cmd_generator(args):
    if args.resume and getattr(args, "init_from", None):
        raise SystemExit("--resume and --init-from are mutually exclusive")
    resuming = args.resume and os.path.exists(args.checkpoint)
    initializing = bool(getattr(args, "init_from", None)) and os.path.exists(args.init_from)
    if resuming or initializing:
        blocked = [
            f"--{name.replace('_', '-')}"
            for name in ("dim", "heads", "generator_layers", "dropout")
            if getattr(args, name) is not None
        ]
        if blocked:
            raise SystemExit(
                f"{', '.join(blocked)} cannot be used with --resume or --init-from; "
                "checkpoint architecture is preserved"
            )
        cfg, _ = load_cfg(args.checkpoint if resuming else args.init_from, "generator")
    else:
        cfg, _ = load_cfg(args.tokenizer, "tokenizer")
    for name in ("corruption", "self_condition"):
        value = getattr(args, name)
        if value is not None:
            setattr(cfg, name, value)
    if not resuming:
        for name in ("dim", "heads", "generator_layers", "dropout"):
            value = getattr(args, name)
            if value is not None:
                setattr(cfg, name, value)
    cfg.__post_init__()
    device = pick_device(args.device)
    print(describe(device))
    train_generator(args, cfg, device)


def cmd_gen(args):
    if not 1 <= args.gen_len:
        raise SystemExit("--gen-len must be positive")
    if args.candidates < 1:
        raise SystemExit("--candidates must be positive")
    device = pick_device(args.device)
    print(describe(device))
    cfg, model, state = load_generator(args.checkpoint, device)
    if args.gen_len > cfg.ctx_len:
        raise SystemExit(f"--gen-len cannot exceed this checkpoint's ctx_len={cfg.ctx_len}")
    prompt, raw = prepare_prompt(args.prompt, cfg, device)
    prompt = prompt.repeat(args.candidates, 1)
    with torch.no_grad():
        tokens, stats = model.generate(
            prompt, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
            guidance=args.guidance, use_cache=not args.no_cache,
            early_exit=args.early_exit, refine_steps=args.refine_steps,
            refine_threshold=args.refine_threshold,
            guidance_ramp=not args.constant_guidance,
        )
    if args.candidates > 1:
        tokens = mbr_select(tokens.unsqueeze(1))
    output = as_text(tokens[0], args.gen_len)
    nfe = stats["generator_nfe"] * args.candidates
    decoder_nfe = stats["decoder_nfe"] * args.candidates
    refiner_nfe = stats["refiner_nfe"] * args.candidates
    print(
        f"NCM: {stats['levels_used']}/{cfg.n_levels} levels | {nfe} generator NFE | "
        f"{decoder_nfe} decoder + {refiner_nfe} refiner NFE | "
        f"{args.gen_len / max(stats['elapsed'], 1e-6):.1f} byte/s"
    )
    if args.show_levels:
        for level in range(stats["levels_used"]):
            partial, _ = model.tokenizer.decode([c[:1] for c in stats["codes"][:level + 1]])
            print(f"--- level {level + 1} ({cfg.levels[level]} codes) ---")
            print(as_text(partial[0], args.gen_len))
    print("--- continuation ---")
    print(raw.decode("utf-8", errors="replace") + output)


def cmd_reconstruct(args):
    device = pick_device(args.device)
    print(describe(device))
    cfg, state = load_cfg(args.checkpoint, "tokenizer")
    model = MultiscaleTokenizer(cfg).to(device).eval()
    model.load_state_dict(state["model"])
    raw = args.text.encode("utf-8")[:cfg.ctx_len]
    if not raw:
        raise SystemExit("--text must not be empty")
    ids = list(raw) + [cfg.pad_token] * (cfg.ctx_len - len(raw))
    tokens = torch.tensor([ids], dtype=torch.long, device=device)
    valid = tokens != cfg.pad_token
    codes = model.encode(tokens)
    if args.show_levels:
        for level in range(cfg.n_levels):
            partial, _ = model.decode(codes[:level + 1], valid=valid)
            print(f"--- level {level + 1} ---\n{as_text(partial[0], len(raw))}")
    decoded, stats = model.decode(
        codes, valid=valid, refine_steps=args.refine_steps, refine_threshold=args.refine_threshold,
    )
    result = as_text(decoded[0], len(raw))
    accuracy = (decoded[0, :len(raw)] == tokens[0, :len(raw)]).float().mean()
    print(f"Reconstruction accuracy: {accuracy:.2%} | refiner NFE {stats['refiner_nfe']}")
    print(result)


def main(argv=None):
    args = build_parser().parse_args(argv)
    if hasattr(args, "seed"):
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    if getattr(args, "steps", 1) < 1 or getattr(args, "tokenizer_steps", 1) < 1:
        raise SystemExit("--steps and --tokenizer-steps must be positive")
    {
        "tokenizer": cmd_tokenizer, "tokenize": cmd_tokenizer,
        "generator": cmd_generator, "train": cmd_train_all,
        "gen": cmd_gen, "generate": cmd_gen,
        "reconstruct": cmd_reconstruct,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
