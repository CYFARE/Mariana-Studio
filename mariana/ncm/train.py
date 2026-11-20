from __future__ import annotations

import math
import os
import time

import torch
from torch.utils.data import DataLoader, TensorDataset

from ..data import corpus_rows_split, fetch_corpus
from ..device import Amp, empty_cache
from .config import NCMConfig
from .generator import NCMGenerator
from .tokenizer import MultiscaleTokenizer
from ..freeze import freeze_ncm_generator


def build_pairs(data: bytes | str | os.PathLike, cfg: NCMConfig, val_frac: float = 0.05):
    if not data:
        raise ValueError("NCM training corpus is empty")
    return corpus_rows_split(
        data, cfg.prefix_len + cfg.ctx_len, val_frac, stride=cfg.ctx_len,
    )


def build_targets(data: bytes | str | os.PathLike, cfg: NCMConfig, val_frac: float = 0.05):
    if not data:
        raise ValueError("NCM training corpus is empty")
    return corpus_rows_split(
        data, cfg.ctx_len, val_frac, split_width=cfg.prefix_len + cfg.ctx_len,
    )


def save_checkpoint(path, model, optimizer, cfg, step, amp, stage,
                    schedule_steps=None, schedule_start=0):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = path + ".tmp"
    state = {
        "format": 2, "stage": stage, "step": step,
        "cfg": cfg.to_dict(), "model": model.state_dict(),
        "optimizer": optimizer.state_dict(), "amp": amp.state_dict(),
        "schedule_start": schedule_start,
    }
    if schedule_steps is not None:
        state["schedule_steps"] = schedule_steps
    torch.save(state, temporary)
    os.replace(temporary, path)


def load_checkpoint(path, map_location="cpu"):
    return torch.load(path, map_location=map_location, weights_only=False)


def make_optimizer(model, lr, device):
    decay, no_decay = [], []
    for parameter in model.parameters():
        if parameter.requires_grad:
            (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    groups = [
        {"params": decay, "weight_decay": 0.05},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    kwargs = dict(lr=lr, betas=(0.9, 0.95))
    if device.type == "cuda":
        try:
            return torch.optim.AdamW(groups, fused=True, **kwargs)
        except (TypeError, RuntimeError):
            pass
    return torch.optim.AdamW(groups, **kwargs)


def load_optimizer_state(optimizer, state):
    try:
        optimizer.load_state_dict(state)
        return
    except ValueError:
        current = optimizer.state_dict()
        old_order = [p for group in state["param_groups"] for p in group["params"]]
        new_order = [p for group in current["param_groups"] for p in group["params"]]
        if len(old_order) != len(new_order) or len(set(old_order)) != len(old_order):
            raise
        old_to_new = dict(zip(old_order, new_order))
        new_to_old = {new: old for old, new in old_to_new.items()}
        old_group_for = {
            parameter: group for group in state["param_groups"] for parameter in group["params"]
        }
        groups = []
        for group in current["param_groups"]:
            old_parameter = new_to_old[group["params"][0]]
            migrated = old_group_for[old_parameter].copy()
            migrated["params"] = group["params"]
            migrated["weight_decay"] = group["weight_decay"]
            groups.append(migrated)
        optimizer.load_state_dict({
            "state": {old_to_new[p]: value for p, value in state["state"].items()},
            "param_groups": groups,
        })


def free_running_metrics(model, batch, cfg, device):
    if len(batch) == cfg.n_levels + 1:
        prompt = batch[0].to(device, non_blocking=True).long()
        codes = [code.to(device, non_blocking=True).long() for code in batch[1:]]
    else:
        rows = batch[0].to(device, non_blocking=True).long()
        prompt, target = rows[:, :cfg.prefix_len], rows[:, cfg.prefix_len:]
        with torch.no_grad():
            codes = model.tokenizer.encode(target)
    with torch.no_grad():
        _, stats = model.generate(prompt, temperature=0, refine_steps=0)
        correct = torch.cat([
            (predicted == truth).reshape(prompt.shape[0], -1)
            for predicted, truth in zip(stats["codes"], codes)
        ], dim=1).float().mean()
        result = {"free": correct}
        result.update({
            f"free_l{level}": (predicted == truth).float().mean()
            for level, (predicted, truth) in enumerate(zip(stats["codes"], codes))
        })
    return result


def cache_generator_codes(model, rows, cfg, args, device, amp):
    started = time.time()
    dtype = torch.int16 if cfg.codebook_size <= 32768 else torch.int32
    codes = [
        torch.empty(len(rows), length, dtype=dtype)
        for length in cfg.levels
    ]
    loader = DataLoader(
        TensorDataset(rows), batch_size=args.batch,
        pin_memory=device.type == "cuda",
    )
    offset = 0
    with torch.no_grad():
        for batch in loader:
            target = batch[0][:, cfg.prefix_len:].to(device, non_blocking=True).long()
            with amp.autocast():
                encoded = model.tokenizer.encode(target)
            count = target.shape[0]
            for destination, source in zip(codes, encoded):
                destination[offset:offset + count].copy_(source.cpu())
            offset += count
    size = sum(code.numel() * code.element_size() for code in codes) / 1024**2
    print(f"Cached {len(rows)} frozen-code rows ({size:.1f} MiB) in {time.time() - started:.1f}s")
    return TensorDataset(rows[:, :cfg.prefix_len], *codes)


def run_training(model, train_ids, val_ids, cfg, args, device, checkpoint, stage):
    for name in ("steps", "batch", "grad_accum", "log_every", "eval_every"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("--lr must be finite and positive")
    pin = device.type == "cuda"
    train_loader = DataLoader(
        TensorDataset(train_ids), batch_size=args.batch, shuffle=True,
        drop_last=True, pin_memory=pin,
    )
    if len(train_loader) == 0:
        raise ValueError(
            f"training loader is empty: {len(train_ids)} rows with "
            f"batch={args.batch} and drop_last=True; reduce --batch or provide more data"
        )
    val_loader = DataLoader(TensorDataset(val_ids), batch_size=args.batch, pin_memory=pin)
    amp = Amp(device, force=args.amp)
    start = 0
    schedule_origin = 0
    saved_optimizer = None
    legacy_generator = False
    if args.resume and os.path.exists(checkpoint):
        state = load_checkpoint(checkpoint, map_location=device)
        if state.get("stage") != stage:
            raise ValueError(f"checkpoint is {state.get('stage')}, not {stage}")
        model.load_state_dict(state["model"])
        start = state.get("step", 0)
        legacy_generator = state.get("format", 1) < 2 and stage == "generator"
        if legacy_generator:
            if cfg.codebook_head == "distance":
                for head in model.heads:
                    head.log_scale.detach().zero_()
            schedule_origin = start
            print("Migrating legacy generator objective; optimizer and LR schedule reset")
        else:
            saved_optimizer = state.get("optimizer")
            schedule_origin = state.get("schedule_start", 0)
            previous_total = state.get("schedule_steps")
            if previous_total is not None and start >= previous_total:
                schedule_origin = start
                print("Previous LR cycle completed; starting a new cycle over the additional steps")
        amp.load_state_dict(state.get("amp", {}))
        print(f"Resumed {stage} from step {start}")
    if args.steps <= start:
        raise ValueError(f"--steps ({args.steps}) must exceed checkpoint step ({start})")
    if stage == "generator":
        trainable = freeze_ncm_generator(
            model, getattr(args, "freeze_embeddings", False), getattr(args, "freeze_layers", 0),
        )
        if trainable <= 0:
            raise ValueError("Freeze settings leave no trainable generator parameters")
        if getattr(args, "freeze_layers", 0) or getattr(args, "freeze_embeddings", False):
            print(f"Fine-tune scope: {trainable / 1e6:.2f}M trainable parameters")
    else:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = make_optimizer(model, args.lr, device)
    if saved_optimizer is not None:
        try:
            load_optimizer_state(optimizer, saved_optimizer)
        except (ValueError, KeyError):
            print("Checkpoint optimizer state does not match the current freeze scope; starting fresh")
    if stage == "generator" and getattr(args, "flex", False):
        model.enable_flex(device)
    if stage == "generator" and getattr(args, "cache_codes", False):
        with torch.random.fork_rng(devices=[]):
            train_dataset = cache_generator_codes(model, train_ids, cfg, args, device, amp)
            val_dataset = cache_generator_codes(model, val_ids, cfg, args, device, amp)
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch, shuffle=True,
            drop_last=True, pin_memory=pin,
        )
        val_loader = DataLoader(val_dataset, batch_size=args.batch, pin_memory=pin)
    if getattr(args, "compile", False):
        model.forward = torch.compile(
            model.forward, mode=getattr(args, "compile_mode", "default"),
        )

    def losses_for(batch, training=True):
        with amp.autocast():
            if stage == "tokenizer":
                rows = batch[0].to(device, non_blocking=True).long()
                return model(rows[:, -cfg.ctx_len:])
            if len(batch) == cfg.n_levels + 1:
                prompt = batch[0].to(device, non_blocking=True).long()
                codes = [code.to(device, non_blocking=True).long() for code in batch[1:]]
            else:
                rows = batch[0].to(device, non_blocking=True).long()
                prompt, target = rows[:, :cfg.prefix_len], rows[:, cfg.prefix_len:]
                with torch.no_grad():
                    codes = model.tokenizer.encode(target)
            return model.loss(prompt, codes)

    steps_per_epoch = max(1, math.ceil(len(train_loader) / args.grad_accum))
    print(
        f"Data: {len(train_ids) * cfg.ctx_len / 1e6:.2f} target MB | "
        f"epoch ≈ {steps_per_epoch} steps | run ≈ {args.steps / steps_per_epoch:.3f} epochs"
    )
    model.train()
    optimizer.zero_grad(set_to_none=True)
    iterator = iter(train_loader)
    started = last_log = time.time()
    for step in range(start, args.steps):
        schedule_step = step - schedule_origin
        schedule_steps = args.steps - schedule_origin
        progress = schedule_step / max(schedule_steps, 1)
        lr = args.lr * min(1.0, (schedule_step + 1) / max(args.warmup, 1))
        lr *= 0.5 * (1.0 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = lr
        metrics = {}
        for accumulation in range(args.grad_accum):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch = next(iterator)
            losses = losses_for(batch)
            amp.scale_backward(losses["total"] / args.grad_accum)
            for key, value in losses.items():
                metrics[key] = metrics.get(key, 0.0) + value.detach() / args.grad_accum
        if amp.use_scaler:
            amp.scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0,
            error_if_nonfinite=not amp.use_scaler,
        )
        amp.step(optimizer)
        optimizer.zero_grad(set_to_none=True)

        if getattr(args, "pause_file", None) and os.path.exists(args.pause_file):
            save_checkpoint(
                checkpoint, model, optimizer, cfg, step + 1, amp, stage,
                schedule_steps=args.steps, schedule_start=schedule_origin,
            )
            print(f"Paused at step {step + 1}/{args.steps}; saved {checkpoint}", flush=True)
            break

        if (step + 1) % args.log_every == 0 or step == start:
            now = time.time()
            positions = args.batch * cfg.ctx_len * args.grad_accum
            positions *= min(step + 1 - start, args.log_every)
            rate = positions / max(now - last_log, 1e-6)
            last_log = now
            message = " | ".join(f"{k} {v:.3f}" for k, v in metrics.items())
            print(f"step {step + 1}/{args.steps} | {message} | lr {lr:.2e} | pos/s {rate:.0f}", flush=True)

        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            save_checkpoint(
                checkpoint, model, optimizer, cfg, step + 1, amp, stage,
                schedule_steps=args.steps, schedule_start=schedule_origin,
            )
            print(f"Saved {checkpoint}", flush=True)
            model.eval()
            totals, batches, examples = {}, 0, 0
            first_batch = None
            with torch.no_grad():
                for batch in val_loader:
                    if first_batch is None:
                        first_batch = batch
                    losses = losses_for(batch, training=False)
                    batch_size = batch[0].shape[0]
                    for key, value in losses.items():
                        totals[key] = totals.get(key, 0.0) + value.detach() * batch_size
                    batches += 1
                    examples += batch_size
                    if batches >= 10:
                        break
                if stage == "generator" and first_batch is not None:
                    with amp.autocast():
                        free = free_running_metrics(model, first_batch, cfg, device)
                    print("FREE " + " | ".join(f"{k} {v:.3f}" for k, v in free.items()), flush=True)
            print("VAL " + " | ".join(f"{k} {v / max(examples, 1):.3f}" for k, v in totals.items()), flush=True)
            model.train()
    empty_cache(device)
    print(f"Finished {stage} in {time.time() - started:.1f}s")
    return model


def train_tokenizer(args, cfg, device):
    model = MultiscaleTokenizer(cfg).to(device)
    init_from = getattr(args, "init_from", None)
    if init_from and os.path.exists(init_from):
        state = load_checkpoint(init_from, map_location=device)
        if state.get("stage") != "tokenizer":
            raise ValueError("--init-from must point to a tokenizer checkpoint")
        model.load_state_dict(state["model"])
        print(f"Initialized tokenizer from {init_from} (step {state.get('step', '?')})")
    print(f"Tokenizer parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M | levels {cfg.levels}")
    train_ids, val_ids = build_targets(args.data if args.data else fetch_corpus(None), cfg)
    return run_training(
        model, train_ids, val_ids, cfg, args, device, args.checkpoint, "tokenizer"
    )


def train_generator(args, cfg, device):
    tokenizer = MultiscaleTokenizer(cfg)
    if not (args.resume and os.path.exists(args.checkpoint)):
        state = load_checkpoint(args.tokenizer)
        if state.get("stage") != "tokenizer":
            raise ValueError("--tokenizer must point to a stage-1 tokenizer checkpoint")
        tokenizer.load_state_dict(state["model"])
        print(f"Loaded frozen tokenizer from step {state.get('step', '?')}")
    tokenizer = tokenizer.to(device)
    model = NCMGenerator(tokenizer, cfg).to(device)
    init_from = getattr(args, "init_from", None)
    if init_from:
        state = load_checkpoint(init_from, map_location=device)
        if state.get("stage") != "generator":
            raise ValueError("--init-from must point to an NCM generator checkpoint")
        model.load_state_dict(state["model"])
        print(f"Initialized generator from {init_from} (step {state.get('step', '?')})")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Generator trainable parameters: {trainable / 1e6:.2f}M | levels {cfg.levels}")
    train_ids, val_ids = build_pairs(args.data if args.data else fetch_corpus(None), cfg)
    return run_training(
        model, train_ids, val_ids, cfg, args, device, args.checkpoint, "generator"
    )
