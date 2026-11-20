#!/usr/bin/env python3
"""Measure whether a checkpoint's private latent code carries information.

A low compressed-context loss (``lat``) is not enough: the local byte window
could be doing all the work.  This script compares continuation CE for:

  true       the symbols the model actually wrote
  shuffled   the same symbols in a different order
  random     random symbols from the private alphabet
  blank      the same thought positions, zero content

Positive ``gain_over_blank`` and ``shuffle_penalty`` mean the code matters.
Near-zero penalties mean the model is reading only the local window.

Usage:
    python scripts/latent_ablation.py --checkpoint mariana-latent.pt
    python scripts/latent_ablation.py --checkpoint mariana-latent.pt \
        --data /path/to/corpus.txt --batch 8 --cutoff 128
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mariana.config import Config
from mariana.data import build_dataset, fetch_corpus
from mariana.device import pick_device
from mariana.latent import code_ablation, format_code
from mariana.model import MarianaLM
from mariana.train import load_checkpoint


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="latent_ablation",
        description="Test whether Mariana's private latent code is functional.",
    )
    p.add_argument("--checkpoint", type=str, default="trained-model/mariana.pt")
    p.add_argument("--data", type=str, default=None,
                   help="local corpus file (default: built-in demo corpus)")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--cutoff", type=int, default=None,
                   help="context/continuation split (default: ctx_len // 2)")
    p.add_argument("--recall", action="store_true",
                   help="use a synthetic associative-recall batch instead of text")
    p.add_argument("--mask-local", action="store_true",
                   help="mask the local window and test reconstruction from the code")
    p.add_argument("--device", type=str, default="cpu")
    return p


def main() -> None:
    args = build_parser().parse_args()
    ckpt = load_checkpoint(args.checkpoint, map_location="cpu")
    cfg = Config.from_dict(ckpt["cfg"])
    if not cfg.use_latent:
        raise SystemExit(
            "This checkpoint has no latent code. Train with --latent first."
        )

    device = pick_device(args.device)
    model = MarianaLM(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    data = fetch_corpus(args.data)
    _, val_ids = build_dataset(data, cfg.ctx_len)
    rows = val_ids[: args.batch]
    cutoff = args.cutoff if args.cutoff is not None else cfg.ctx_len // 2

    out = code_ablation(
        model, rows, cfg, device, cutoff=cutoff,
        recall=args.recall, mask_local=args.mask_local,
    )
    task = "recall" if args.recall else "natural text"
    if args.mask_local:
        task += ", masked local"
    print(f"Checkpoint step {ckpt.get('step', '?')} | device {device} | {task}")
    print(
        f"Code: {format_code(out['symbols'][0])}  "
        f"(context {out['cutoff']} bytes -> {cfg.n_thoughts} symbols)"
    )
    print()
    print(f"  true code       {out['true']:.4f}")
    print(f"  shuffled code   {out['shuffled']:.4f}  (penalty {out['shuffle_penalty']:+.4f})")
    print(f"  random code     {out['random']:.4f}  (penalty {out['random_penalty']:+.4f})")
    print(f"  blank code      {out['blank']:.4f}  (gain {out['gain_over_blank']:+.4f})")
    print()
    print(f"  local recon     true {out['true_local']:.4f} | "
          f"shuffled {out['shuffled_local']:.4f} | "
          f"random {out['random_local']:.4f} | "
          f"blank {out['blank_local']:.4f}  (local gain {out['local_gain']:+.4f})")
    print()
    # With a masked local window the reconstruction term is the direct test;
    # otherwise use the continuation gain.  Take the stronger signal.
    code_gain = out["gain_over_blank"]
    if args.mask_local:
        code_gain = max(code_gain, out["local_gain"])

    if code_gain > 0.02 and out["shuffle_penalty"] > 0.02:
        print("Verdict: the code carries information; order/content matter.")
    elif code_gain > 0.02:
        print("Verdict: the code stores information, but shuffling it does not "
              "hurt much (a bag-of-symbols memory rather than a grammar).")
    else:
        print("Verdict: the code is not contributing beyond the local window. "
              "Train longer, shrink --latent-window, or increase --w-latent.")


if __name__ == "__main__":
    main()
