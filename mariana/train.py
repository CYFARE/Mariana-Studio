"""Training: hybrid causal + block-diffusion objective.

Upgrades over the original:

* **Block-wise corruption & timesteps** -- the draft region is split into
  ``block_size`` blocks, each with an independent noise level.  Attention is
  *block-causal* (bidirectional inside a block, causal across blocks), which
  is the modern block-diffusion design and stays KV-cache compatible.
* **Attention-mask annealing** -- early in training a fraction of batches use
  a purely causal mask, annealing to the block-hybrid mask (stabilises the
  causal -> bidirectional transition).
* **L_align consistency loss** -- KL pulling diffusion marginals toward the
  (detached) AR conditionals; directly optimises speculative acceptance.
* **Loop dropout** -- the recurrent core depth is sampled per step so every
  loop count is a valid operating point (enables inference-time halting).
* **Sprint-DRV (--fast)** -- one shared hybrid pass trains exact causal-prefix
  AR, diffusion, verifier and MTP; vectorized on-device batch construction.
* **SAINT latent thought** -- adaptive discrete/continuous memory with grammar,
  hierarchy, policy-gradient, and draft-only cross-attention objectives.
* **Auxiliary MTP loss** and optional MoE load-balancing loss.
"""

from __future__ import annotations

import math
import os
import random
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .config import Config
from .data import build_dataset, fetch_corpus
from .device import Amp, empty_cache
from .freeze import freeze_mariana
from .latent import latent_batch_loss
from .model import HAS_FLEX, MarianaLM

if HAS_FLEX:
    from torch.nn.attention.flex_attention import create_block_mask


# ---------------------------------------------------------------------------
# Masks
# ---------------------------------------------------------------------------
def make_causal_mask(L: int) -> torch.Tensor:
    return torch.tril(torch.ones(L, L, dtype=torch.bool))


def make_block_mask(L: int, cutoff: int, block_size: int) -> torch.Tensor:
    """Causal prefix + block-causal draft (bidirectional inside each block)."""
    idx = torch.arange(L)
    # causal everywhere by default
    allow = idx.unsqueeze(1) >= idx.unsqueeze(0)
    # draft rows additionally see their whole block (future positions inside it)
    if cutoff < L:
        block_end = ((idx - cutoff).clamp_min(0) // block_size + 1) * block_size + cutoff
        block_end = block_end.clamp_max(L)
        draft_rows = idx >= cutoff
        allow = allow | (draft_rows.unsqueeze(1) & (idx.unsqueeze(0) < block_end.unsqueeze(1)))
    return allow


def build_flex_block_mask(cutoffs: torch.Tensor, ctx: int, block_size: int, device):
    """FlexAttention BlockMask equivalent of :func:`make_block_mask` (batched)."""
    cutoffs = cutoffs.to(device)

    def mask_mod(b, h, q, kv):
        k = cutoffs[b]
        causal = q >= kv
        qd = q >= k
        block_end = ((q - k).clamp_min(0) // block_size + 1) * block_size + k
        return causal | (qd & (kv < block_end))

    return create_block_mask(mask_mod, B=len(cutoffs), H=None, Q_LEN=ctx, KV_LEN=ctx, device=device)


# ---------------------------------------------------------------------------
# Corruption
# ---------------------------------------------------------------------------
def corrupt_batch(x, cfg: Config, max_t: float = 1.0, p_causal: float = 0.0,
                  use_flex: bool = False, device=None):
    """Return inputs/targets/etc for one hybrid training batch.

    * prefix [0, cutoff): clean, causal, AR loss
    * draft  [cutoff, L): split into blocks with independent noise levels

    Corruption and masks are built with tensor ops directly on the training
    device.  The old B x L Python loop was a surprisingly large host-side cost
    for small models and also forced a synchronous CPU -> GPU handoff.
    """
    B, _ = x.shape
    L = cfg.ctx_len
    work_device = torch.device(device) if device is not None else x.device
    x = x.to(work_device, non_blocking=True).long().clone()
    x[:, 0] = cfg.bos_token

    originals = x[:, :-1].clone()  # self-prediction targets (diffusion)
    targets = x[:, 1:].clone()     # next-token targets (AR)
    inputs = x[:, :-1].clone()

    # Sprint-DRV occasionally uses cutoff=L: a fully clean causal batch gives
    # AR coverage even at the final byte. Empty-draft losses are zero-safe.
    cutoff_high = L + 1 if cfg.fast_train else L - 4
    cutoffs = torch.randint(L // 4, cutoff_high, (B,), device=work_device)
    pos = torch.arange(L, device=work_device)
    region = (pos.unsqueeze(0) >= cutoffs.unsqueeze(1)).long()
    draft = region.bool()

    # One random noise level per sample and relative draft block, gathered back
    # to byte positions.  This matches the original per-block corruption.
    block_idx = (pos.unsqueeze(0) - cutoffs.unsqueeze(1)).clamp_min(0) // cfg.block_size
    n_blocks = (L + cfg.block_size - 1) // cfg.block_size + 1
    block_t = torch.rand(B, n_blocks, device=work_device) * max_t
    token_t = block_t.gather(1, block_idx)
    buckets = (token_t * (cfg.n_time_buckets - 1)).long() + 1
    times = torch.where(draft, buckets, torch.zeros_like(buckets))

    p_mask = 0.1 + 0.8 * token_t
    noise = torch.rand(B, L, device=work_device)
    mask_hit = draft & (noise < p_mask)
    random_hit = draft & ~mask_hit & (noise < p_mask + 0.1)
    inputs = torch.where(mask_hit, torch.full_like(inputs, cfg.mask_token), inputs)
    random_bytes = torch.randint(0, 256, inputs.shape, device=work_device)
    inputs = torch.where(random_hit, random_bytes, inputs)

    # Batched causal/block-causal mask, with optional per-row causal annealing.
    q = pos.view(1, L, 1)
    kv = pos.view(1, 1, L)
    causal = q >= kv
    block_end = ((q - cutoffs[:, None, None]).clamp_min(0) // cfg.block_size + 1)
    block_end = block_end * cfg.block_size + cutoffs[:, None, None]
    block_end = block_end.clamp_max(L)
    allow = causal | (draft.unsqueeze(2) & (kv < block_end))
    causal_only = torch.rand(B, device=work_device) < p_causal
    allow = torch.where(causal_only[:, None, None], causal, allow)
    attn_mask = allow.unsqueeze(1)  # (B,1,L,L)

    flex_bm = None
    if use_flex and HAS_FLEX and device is not None:
        flex_bm = build_flex_block_mask(cutoffs, L, cfg.block_size, work_device)
        attn_mask = None  # FlexAttention path takes precedence

    return inputs, targets, originals, times, region, attn_mask, flex_bm


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------
def _to(device, *tensors):
    return [t.to(device, non_blocking=True) if t is not None else None for t in tensors]


def fast_hybrid_losses(model, inputs, targets, originals, times, region, attn_mask,
                       flex_bm, cfg: Config, n_loops: int,
                       latent_memory=None, latent_memory_mask=None):
    """Sprint-DRV: train every head from one shared hybrid forward.

    The hybrid mask is strictly causal in the clean prefix, where time and
    region are both zero.  Its logits are therefore exactly the model's AR
    conditionals at those positions; a second clean full-sequence forward is
    redundant.  Random cutoffs cover different AR positions on later batches.

    Full-sequence L_align is intentionally a legacy-only objective: computing
    its clean teacher logits in the draft region would require the second pass
    that Sprint-DRV eliminates.  The returned zero keeps metric schemas stable.
    """
    logits, verify, extras = model(
        inputs, times, region, attn_mask, n_loops=n_loops, block_mask=flex_bm,
        need_hidden=cfg.use_mtp, latent_memory=latent_memory,
        latent_memory_mask=latent_memory_mask,
    )
    ar_mask = region == 0
    diff_mask = ~ar_mask
    B, L = region.shape
    n_ar = ar_mask.sum().clamp_min(1)
    n_diff = diff_mask.sum().clamp_min(1)

    # Each position has exactly one target: next-byte AR in the prefix or
    # self-byte denoising in the draft.  One fixed-shape CE therefore computes
    # both losses.  Masked reductions avoid boolean advanced indexing, whose
    # variable-sized output can synchronise the GPU and defeat compilation.
    target = torch.where(ar_mask, targets, originals)
    ce = F.cross_entropy(
        logits.reshape(-1, cfg.vocab_size), target.reshape(-1), reduction="none"
    ).view(B, L)
    ar_loss = (ce * ar_mask).sum() / n_ar
    diff_loss = (ce * diff_mask).sum() / n_diff

    with torch.no_grad():
        correct = (logits.argmax(-1) == originals).float()
    ver = F.binary_cross_entropy_with_logits(verify, correct, reduction="none")
    ver_loss = (ver * diff_mask).sum() / n_diff

    align_loss = logits.new_zeros(())
    losses = {
        "ar": ar_loss,
        "diff": diff_loss,
        "align": align_loss,
        "ver": ver_loss,
        "total": cfg.w_ar * ar_loss + cfg.w_diff * diff_loss + cfg.w_ver * ver_loss,
    }

    # Prefix hidden states are causal, so they also provide exact MTP features.
    if cfg.use_mtp and model.mtp_head is not None:
        mtp_logits = model.mtp_head(extras["hidden"][:, :-2])
        mtp_ce = F.cross_entropy(
            mtp_logits.reshape(-1, cfg.vocab_size), originals[:, 2:].reshape(-1),
            reduction="none",
        ).view(B, L - 2)
        mtp_mask = ar_mask[:, :-2]
        mtp_loss = (mtp_ce * mtp_mask).sum() / mtp_mask.sum().clamp_min(1)
        losses["mtp"] = mtp_loss
        losses["total"] = losses["total"] + cfg.w_mtp * mtp_loss

    if "moe_lb" in extras:
        losses["total"] = losses["total"] + cfg.w_moe_lb * extras["moe_lb"]
    return losses


def hybrid_losses(model, inputs, targets, originals, times, region, attn_mask,
                  flex_bm, cfg: Config, causal_mask, n_loops_diff: int,
                  latent_memory=None, latent_memory_mask=None):
    """Dispatch to Sprint-DRV or legacy two-pass AR + block diffusion."""
    if cfg.fast_train:
        return fast_hybrid_losses(
            model, inputs, targets, originals, times, region, attn_mask,
            flex_bm, cfg, n_loops_diff, latent_memory, latent_memory_mask,
        )

    zeros = torch.zeros_like(region)

    # 1) Causal AR pass on the clean sequence (canonical full depth)
    logits_ar, _, extras = model(originals, zeros, zeros, causal_mask,
                                 need_hidden=cfg.use_mtp)
    ar_loss = F.cross_entropy(logits_ar.reshape(-1, cfg.vocab_size), targets.reshape(-1))

    # 2) Hybrid block-diffusion pass on the corrupted sequence
    logits, verify, extras_d = model(
        inputs, times, region, attn_mask, n_loops=n_loops_diff, block_mask=flex_bm,
        latent_memory=latent_memory, latent_memory_mask=latent_memory_mask,
    )

    diff_mask = region == 1
    flat_mask = diff_mask.reshape(-1).float()
    n_diff = diff_mask.sum().clamp_min(1)

    diff_loss = F.cross_entropy(
        logits.reshape(-1, cfg.vocab_size), originals.reshape(-1), reduction="none"
    )
    diff_loss = (diff_loss * flat_mask).sum() / n_diff

    # 3) L_align: pull diffusion marginals toward detached AR conditionals.
    #    AR logit at position j-1 is the conditional for token j. Evaluate all
    #    j>=1 and mask, rather than slicing at k_min via a synchronising .item().
    with torch.no_grad():
        ar_cond = F.log_softmax(logits_ar[:, :-1].detach(), dim=-1)
    diff_logp = F.log_softmax(logits[:, 1:], dim=-1)
    align_loss = F.kl_div(diff_logp, ar_cond, log_target=True, reduction="none").sum(-1)
    align_loss = (align_loss * diff_mask[:, 1:].float()).sum() / n_diff

    # 4) Verifier: predict whether the drafted token equals the original
    with torch.no_grad():
        correct = (logits.argmax(-1) == originals).float()
    ver_loss = F.binary_cross_entropy_with_logits(verify, correct, reduction="none")
    ver_loss = (ver_loss * diff_mask.float()).sum() / n_diff

    losses = {
        "ar": ar_loss,
        "diff": diff_loss,
        "align": align_loss,
        "ver": ver_loss,
        "total": cfg.w_ar * ar_loss + cfg.w_diff * diff_loss
        + cfg.w_align * align_loss + cfg.w_ver * ver_loss,
    }

    # 5) Multi-token prediction: hidden state at i predicts token i+2
    if cfg.use_mtp and model.mtp_head is not None:
        mtp_logits = model.mtp_head(extras["hidden"][:, :-2])
        mtp_target = originals[:, 2:]
        mtp_loss = F.cross_entropy(
            mtp_logits.reshape(-1, cfg.vocab_size), mtp_target.reshape(-1)
        )
        losses["mtp"] = mtp_loss
        losses["total"] = losses["total"] + cfg.w_mtp * mtp_loss

    if "moe_lb" in extras_d:
        losses["total"] = losses["total"] + cfg.w_moe_lb * extras_d["moe_lb"]

    return losses


def _merge_latent_losses(losses, lat):
    losses["lat"] = lat["lat_cont"]
    losses["lat_win"] = lat["lat_local"]
    losses["lat_ent"] = lat["lat_ent"]
    losses["lat_recall"] = lat["lat_recall"]
    for key in (
        "lat_grammar_task", "lat_grammar", "lat_pair", "lat_order", "lat_hier",
        "lat_rl", "lat_value", "lat_reward", "lat_kl", "lat_len", "lat_parents",
        "lat_stop", "lat_moe_lb",
    ):
        if key in lat:
            losses[key] = lat[key]
    losses["total"] = losses["total"] + lat["lat_total"]


def train_step(model, xb, device, cfg: Config, amp: Amp, causal_mask,
               use_flex: bool, p_causal: float, grad_accum: int, do_step: bool,
               optimizer, step: int, total_steps: int):
    inputs, targets, originals, times, region, attn_mask, flex_bm = corrupt_batch(
        xb, cfg, p_causal=p_causal, use_flex=use_flex, device=device
    )
    inputs, targets, originals, times, region, attn_mask = _to(
        device, inputs, targets, originals, times, region, attn_mask
    )

    n_loops = (
        random.randint(1, cfg.max_loops) if cfg.loop_dropout else cfg.max_loops
    )

    with amp.autocast():
        lat = None
        if cfg.use_latent:
            lat = latent_batch_loss(
                model, xb[:, : cfg.ctx_len], cfg, device, training=True
            )
        losses = hybrid_losses(
            model, inputs, targets, originals, times, region, attn_mask,
            flex_bm, cfg, causal_mask, n_loops,
            latent_memory=lat.get("diffusion_memory") if lat else None,
            latent_memory_mask=lat.get("diffusion_memory_mask") if lat else None,
        )
        if lat is not None:
            _merge_latent_losses(losses, lat)
        loss = losses["total"] / grad_accum

    amp.scale_backward(loss)
    if do_step:
        if amp.use_scaler:
            amp.scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        amp.step(optimizer)
        optimizer.zero_grad(set_to_none=True)

    # Keep metrics on-device. Converting every loss to float caused one GPU
    # synchronisation per metric per microbatch; logging now syncs once/25 steps.
    return {k: v.detach() for k, v in losses.items()}


@torch.no_grad()
def evaluate(model, val_loader, device, cfg: Config, amp: Amp, causal_mask,
             max_batches: int = 20):
    model.eval()
    sums: dict[str, torch.Tensor] = {}
    n = 0
    for xb in val_loader:
        inputs, targets, originals, times, region, attn_mask, _ = corrupt_batch(
            xb[0], cfg, max_t=0.5
        )
        inputs, targets, originals, times, region, attn_mask = _to(
            device, inputs, targets, originals, times, region, attn_mask
        )
        with amp.autocast():
            lat = None
            if cfg.use_latent:
                lat = latent_batch_loss(
                    model, xb[0][:, : cfg.ctx_len], cfg, device,
                    cutoff=cfg.ctx_len // 2, training=False,
                )
            losses = hybrid_losses(
                model, inputs, targets, originals, times, region, attn_mask,
                None, cfg, causal_mask, cfg.max_loops,
                latent_memory=lat.get("diffusion_memory") if lat else None,
                latent_memory_mask=lat.get("diffusion_memory_mask") if lat else None,
            )
            if lat is not None:
                _merge_latent_losses(losses, lat)
        for k, v in losses.items():
            sums[k] = sums.get(k, 0.0) + v.detach()
        n += 1
        if n >= max_batches:
            break
    model.train()
    return {k: float(v / max(n, 1)) for k, v in sums.items()}


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def save_checkpoint(path, model, optimizer, cfg: Config, step: int, amp: Amp,
                    compiled: bool):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    sd = model._orig_mod.state_dict() if compiled and hasattr(model, "_orig_mod") else model.state_dict()
    torch.save(
        {
            "step": step,
            "model": sd,
            "optimizer": optimizer.state_dict(),
            "amp": amp.state_dict(),
            "cfg": cfg.to_dict(),
            "format": 2,
        },
        path,
    )


def load_checkpoint(path, map_location="cpu"):
    return torch.load(path, map_location=map_location, weights_only=False)


def train(args, cfg: Config, device) -> MarianaLM:
    if device.type == "cuda":
        # TF32 is a large win for the matmuls here; the export parity test is
        # CPU-side, so it is unaffected.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    model = MarianaLM(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params / 1e6:.2f}M  Depth: {cfg.n_layers} layers "
          f"({cfg.n_pre}+{cfg.n_core}x{cfg.max_loops}+{cfg.n_post})")
    if cfg.fast_train:
        print("Sprint-DRV: one shared forward/step; full-sequence L_align disabled.")
    if cfg.use_latent:
        print(
            f"Latent code: {cfg.n_latent_symbols} symbols x {cfg.n_thoughts} "
            f"thoughts | local window {cfg.latent_local_window} bytes "
            f"(dropout {cfg.latent_window_dropout:.0%}, recall {cfg.latent_recall_prob:.0%}) "
            f"({math.log2(cfg.n_latent_symbols) * cfg.n_thoughts:.1f} max discrete bits)"
        )
        if cfg.latent_saint:
            print(
                f"SAINT: EOS after {cfg.latent_min_thoughts}+ thoughts | "
                f"parents/{cfg.latent_block_size} thoughts | "
                f"{cfg.latent_continuous_slots} continuous slots x {cfg.latent_refine_steps} refinements | "
                f"grammar {cfg.latent_grammar_prob:.0%} | RL {cfg.w_latent_rl} | "
                f"diffusion {'on' if cfg.latent_diffusion else 'off'}"
            )

    data = getattr(args, "data", None) or fetch_corpus(None)
    train_ids, val_ids = build_dataset(data, cfg.ctx_len)
    del data  # small in-memory corpora are copied into rows; large files stay mapped
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        TensorDataset(train_ids), batch_size=args.batch, shuffle=True, drop_last=True,
        pin_memory=pin_memory,
    )
    if len(train_loader) == 0:
        raise ValueError(
            f"training loader is empty: {len(train_ids)} rows with "
            f"batch={args.batch} and drop_last=True; reduce --batch or provide more data"
        )
    val_loader = DataLoader(
        TensorDataset(val_ids), batch_size=args.batch, pin_memory=pin_memory
    )

    steps_per_epoch = max(1, len(train_loader) // max(1, args.grad_accum))
    epochs = args.steps / steps_per_epoch
    data_mb = len(train_ids) * cfg.ctx_len / 1e6
    print(
        f"Data: {data_mb:.2f} MB | 1 epoch ≈ {steps_per_epoch} steps | "
        f"{args.steps} steps ≈ {epochs:.1f} epochs"
    )
    if epochs > 5:
        print(
            "WARNING: more than 5 epochs; a 2M-50M model will memorise a small "
            "corpus. Add data or reduce --steps."
        )

    opt_kwargs = dict(lr=args.lr, betas=(0.9, 0.95), weight_decay=0.05)
    amp = Amp(device, force=getattr(args, "amp", False))

    start_step = 0
    resume_state = None
    init_from = getattr(args, "init_from", None)
    if init_from:
        if args.resume:
            raise ValueError("--init-from and --resume are mutually exclusive")
        source = load_checkpoint(init_from, map_location=device)
        if source.get("stage") not in {None, "drv"}:
            raise ValueError("--init-from must point to a DRV-LM checkpoint")
        model.load_state_dict(source["model"])
        print(
            f"Initialized from {init_from} (step {source.get('step', '?')}); "
            "optimizer and step counter reset"
        )
    if args.resume and os.path.exists(args.checkpoint):
        resume_state = load_checkpoint(args.checkpoint, map_location=device)
        model.load_state_dict(resume_state["model"])
        start_step = resume_state.get("step", 0)
        print(f"Resumed from step {start_step}")

    trainable = freeze_mariana(
        model, getattr(args, "freeze_embeddings", False),
        getattr(args, "freeze_layers", 0),
    )
    if trainable <= 0:
        raise ValueError("Freeze settings leave no trainable parameters")
    if getattr(args, "freeze_layers", 0) or getattr(args, "freeze_embeddings", False):
        print(f"Fine-tune scope: {trainable / 1e6:.2f}M trainable parameters")
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    if device.type == "cuda":
        try:
            optimizer = torch.optim.AdamW(trainable_parameters, fused=True, **opt_kwargs)
        except (TypeError, RuntimeError):
            optimizer = torch.optim.AdamW(trainable_parameters, **opt_kwargs)
    else:
        optimizer = torch.optim.AdamW(trainable_parameters, **opt_kwargs)
    if resume_state is not None:
        saved_count = sum(
            len(group.get("params", []))
            for group in resume_state.get("optimizer", {}).get("param_groups", [])
        )
        if saved_count == len(trainable_parameters):
            optimizer.load_state_dict(resume_state["optimizer"])
            amp.load_state_dict(resume_state.get("amp", {}))
        else:
            print(
                "Checkpoint optimizer state does not match the current "
                "freeze scope; starting a fresh optimizer cycle"
            )

    compiled = False
    if args.compile:
        if cfg.use_latent:
            # The latent thought loop has dynamic lengths and a growing KV
            # cache; compiling it causes repeated recompiles and can be
            # slower than eager.  The hybrid AR/diffusion path is unaffected.
            print("torch.compile skipped for latent-code training (dynamic thought loop).")
        else:
            compile_mode = getattr(args, "compile_mode", "default")
            try:
                model = torch.compile(model, mode=compile_mode)
                compiled = True
                print(f"torch.compile enabled (mode={compile_mode}, no CUDA graphs)")
            except Exception as e:
                print(f"torch.compile unavailable: {e}")

    use_flex = getattr(args, "flex", False) and HAS_FLEX and device.type in ("cuda", "xpu")
    if getattr(args, "flex", False) and not use_flex:
        # FlexAttention has no CPU backward kernel; SDPA masks are equivalent
        print("FlexAttention unavailable on this device/torch; using SDPA masks.")

    causal_mask = (
        make_causal_mask(cfg.ctx_len).to(device).unsqueeze(0).unsqueeze(0)
    )

    model.train()
    optimizer.zero_grad(set_to_none=True)
    t_start = time.time()
    t_last = t_start
    train_iter = iter(train_loader)

    for step in range(start_step, args.steps):
        progress = step / max(args.steps, 1)
        lr = (
            args.lr
            * min(1.0, (step + 1) / max(args.warmup, 1))
            * 0.5 * (1 + math.cos(math.pi * progress))
        )
        for g in optimizer.param_groups:
            g["lr"] = lr

        # mask annealing: start ~50% causal-only batches, anneal to 0
        p_causal = 0.5 * max(0.0, 1.0 - progress / max(args.anneal_frac, 1e-6))

        metrics: dict[str, torch.Tensor] = {}
        for accum_idx in range(args.grad_accum):
            try:
                xb = next(train_iter)[0]
            except StopIteration:
                train_iter = iter(train_loader)
                xb = next(train_iter)[0]
            losses = train_step(
                model, xb, device, cfg, amp, causal_mask, use_flex, p_causal,
                args.grad_accum, accum_idx == args.grad_accum - 1,
                optimizer, step, args.steps,
            )
            for k, v in losses.items():
                metrics[k] = metrics.get(k, 0.0) + v / args.grad_accum

        if getattr(args, "pause_file", None) and os.path.exists(args.pause_file):
            save_checkpoint(args.checkpoint, model, optimizer, cfg, step + 1, amp, compiled)
            print(f"Paused at step {step + 1}/{args.steps}; saved {args.checkpoint}")
            break

        if (step + 1) % 25 == 0 or step == start_step:
            now = time.time()
            tok_per_sec = (
                args.batch * cfg.ctx_len * args.grad_accum * min(step + 1 - start_step, 25)
                / max(now - t_last, 1e-6)
            )
            t_last = now
            msg = " | ".join(f"{k} {v:.3f}" for k, v in metrics.items())
            print(f"step {step + 1:5d}/{args.steps} | {msg} | lr {lr:.2e} | tok/s {tok_per_sec:.0f}")

        if (step + 1) % args.eval_every == 0 or step == args.steps - 1:
            # Save first: a failing evaluation must not throw away the run.
            save_checkpoint(args.checkpoint, model, optimizer, cfg, step + 1, amp, compiled)
            print(f"Saved {args.checkpoint}")
            vlosses = evaluate(model, val_loader, device, cfg, amp, causal_mask)
            msg = " | ".join(f"{k} {v:.3f}" for k, v in vlosses.items())
            print(f"VAL step {step + 1}: {msg}")
            model.train()

    empty_cache(device)
    return model
