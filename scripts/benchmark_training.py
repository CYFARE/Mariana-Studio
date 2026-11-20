"""Compare legacy and Sprint-DRV training throughput on synthetic byte batches.

Example (from the repository root):
    python scripts/benchmark_training.py --device cuda --batch 16 --steps 12

No corpus, checkpoint, network, or test dependencies are required. Loss values
are intentionally irrelevant; this benchmark measures the complete train step.
"""

from __future__ import annotations

import argparse
import gc
import os
import random
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mariana.config import Config
from mariana.device import Amp, empty_cache, pick_device
from mariana.model import MarianaLM
from mariana.train import make_causal_mask, train_step


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "xpu" and hasattr(torch, "xpu"):
        torch.xpu.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def run(mode: str, args, device):
    random.seed(1234)
    torch.manual_seed(1234)
    cfg = Config(ctx_len=args.ctx, dim=args.dim, fast_train=mode == "fast")
    model = MarianaLM(cfg).to(device)
    model.train()
    if args.compile:
        model = torch.compile(model, mode="default")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    try:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=1e-4, betas=(0.9, 0.95), fused=device.type == "cuda"
        )
    except (TypeError, RuntimeError):
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95))
    amp = Amp(device)
    causal = make_causal_mask(cfg.ctx_len).to(device).unsqueeze(0).unsqueeze(0)
    batch = torch.randint(0, 256, (args.batch, cfg.ctx_len + 1), dtype=torch.uint8)

    def step():
        return train_step(
            model, batch, device, cfg, amp, causal, False, 0.0,
            1, True, optimizer, 0, args.steps,
        )

    for _ in range(args.warmup):
        step()
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    for _ in range(args.steps):
        losses = step()
    synchronize(device)
    elapsed = time.perf_counter() - start
    seconds = elapsed / args.steps
    tokens = args.batch * cfg.ctx_len
    memory = (
        f" | peak {torch.cuda.max_memory_allocated(device) / 2**20:.0f} MiB"
        if device.type == "cuda" else ""
    )
    label = mode + ("+compile" if args.compile else "")
    print(
        f"{label:13s} | {seconds * 1000:7.2f} ms/step | "
        f"{tokens / seconds:10.0f} byte-tokens/s{memory} | "
        f"loss {float(losses['total']):.3f}"
    )
    del model, optimizer, amp, causal, batch, losses
    gc.collect()
    empty_cache(device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--ctx", type=int, default=256)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--compile", action="store_true", help="compile the model with Inductor")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()
    device = pick_device(args.device)
    print(f"Device: {device} | batch {args.batch} | ctx {args.ctx} | dim {args.dim}")
    for mode in ("legacy", "fast"):
        run(mode, args, device)


if __name__ == "__main__":
    main()
