"""Legacy latent code plus the public dispatch to SAINT thought.

New CLI latent runs use :mod:`mariana.saint`; ``latent_saint=False`` retains the
original implementation below for checkpoint and API compatibility.

Why a separate channel?
-----------------------
English (or any human tokenizer) forces every internal step through a
low-bandwidth, human-shaped alphabet.  This module gives the model a private
alphabet of ``cfg.n_latent_symbols`` learned symbols and a fixed number of
thought positions.  A thought is emitted from the model's hidden state and its
codebook vector is fed back as the next input.  The sequence of symbols is
therefore a real, discrete, autoregressive language -- but it has no byte
spelling, no dictionary, and no reason to resemble English.

The compressed-context objective
--------------------------------
``latent_batch_loss`` shows the model a context, makes it emit ``N`` symbols,
and then asks it to predict the continuation from:

    [thought_0 ... thought_{N-1}] + [last W raw bytes] + [continuation]

The raw context is *not* visible during the second pass.  The symbols are the
only long-range channel, so gradient descent must make them useful.  The small
local window preserves recency (spelling, immediate syntax) while the symbols
store the rest of the context.  This is a fixed-size learned memory, not a
human-readable summary.

Symbols are discrete at inference.  During training we use the
straight-through Gumbel estimator so the continuation loss can train the
symbol choices and the codebook directly:
``hard_one_hot + soft - soft.detach()``.
"""

from __future__ import annotations

import math
import random

import torch
import torch.nn.functional as F

from .config import Config


_BASE36 = "0123456789abcdefghijklmnopqrstuvwxyz"


def symbol_glyph(symbol: int) -> str:
    """Compact display glyph for a private symbol (not a model token)."""
    symbol = int(symbol)
    if 0 <= symbol < 36 * 36:
        return "⟨" + _BASE36[symbol // 36] + _BASE36[symbol % 36] + "⟩"
    return f"⟨{symbol}⟩"


def format_code(symbols) -> str:
    """Render a symbol tensor (B, N) or sequence as ``⟨0a⟩⟨1f⟩...``."""
    if torch.is_tensor(symbols):
        rows = symbols.detach().cpu().tolist()
    else:
        rows = list(symbols)
    if not rows:
        return ""
    if isinstance(rows[0], int):
        rows = [rows]
    return " ".join("".join(symbol_glyph(i) for i in row) for row in rows)


def _check_cfg(cfg: Config) -> None:
    if not cfg.use_latent:
        raise ValueError(
            "This checkpoint was not trained with the latent code. "
            "Train with --latent (or load a latent-enabled checkpoint)."
        )
    if cfg.n_thoughts < 1:
        raise ValueError("n_thoughts must be >= 1")
    if cfg.n_latent_symbols < 2:
        raise ValueError("n_latent_symbols must be >= 2")
    if cfg.latent_local_window < 1:
        raise ValueError("latent_local_window must be >= 1")


def _causal_mask(length: int, device) -> torch.Tensor:
    return torch.tril(
        torch.ones(length, length, dtype=torch.bool, device=device)
    ).unsqueeze(0).unsqueeze(0)


def _normalised_entropy(probabilities: torch.Tensor) -> torch.Tensor:
    probabilities = probabilities.clamp_min(1e-9)
    entropy = -(probabilities * probabilities.log()).sum()
    return entropy / math.log(probabilities.numel())


def _symbol_usage_entropy(symbols: torch.Tensor, n_symbols: int) -> torch.Tensor:
    counts = torch.bincount(symbols.reshape(-1), minlength=n_symbols).float()
    return _normalised_entropy(counts / counts.sum().clamp_min(1)).detach()


def _mask_local_windows(local: torch.Tensor, probability: float,
                        mask_token: int) -> torch.Tensor:
    masked_rows = torch.rand(local.shape[0], 1, device=local.device) < probability
    return torch.where(
        masked_rows, torch.full_like(local, mask_token), local
    )


def _row_permutations(batch: int, length: int, device) -> torch.Tensor:
    perm = torch.argsort(torch.rand(batch, length, device=device), dim=1)
    if length > 1:
        identity = (perm == torch.arange(length, device=device)).all(dim=1)
        perm = torch.where(
            identity.unsqueeze(1), torch.roll(perm, 1, dims=1), perm
        )
    return perm


def _shuffle_symbols(symbols: torch.Tensor) -> torch.Tensor:
    perm = _row_permutations(*symbols.shape, symbols.device)
    return symbols.gather(1, perm)


def _sample_symbols(logits, codebook, temperature: float = 0.0,
                    training: bool = False):
    """Return ``(indices, embeddings, probabilities)``.

    Training uses the straight-through Gumbel estimator.  Inference is greedy
    when ``temperature <= 0`` and categorical otherwise.  The returned
    embeddings carry gradients to the codebook and to the symbol logits during
    training; at inference they are ordinary codebook lookups.
    """
    logits = logits.float()
    if training:
        tau = max(float(temperature), 1e-3)
        straight = F.gumbel_softmax(logits, tau=tau, hard=True, dim=-1)
        idx = straight.argmax(dim=-1)
        embeds = straight @ codebook.weight.float()
        probs = F.softmax(logits, dim=-1)
        return idx, embeds, probs

    if temperature is not None and temperature > 0:
        probs = F.softmax(logits / max(float(temperature), 1e-6), dim=-1)
        idx = torch.multinomial(
            probs.reshape(-1, probs.shape[-1]), num_samples=1
        ).view(probs.shape[:-1])
    else:
        probs = F.softmax(logits, dim=-1)
        idx = probs.argmax(dim=-1)
    embeds = codebook(idx)
    return idx, embeds, probs


def summarize(model, context, cfg: Config, device,
              training: bool = False, temperature: float | None = None):
    """Read a byte context and emit ``cfg.n_thoughts`` private symbols.

    The context is prefilled once with a causal mask; each thought is then
    appended one at a time using the KV cache.  Thought ``i`` sees the full
    context and all earlier thoughts, never future bytes.  This makes the
    latent code an autoregressive language of its own.
    """
    if cfg.latent_saint:
        from .saint import summarize_saint

        return summarize_saint(
            model, context, cfg, device, training,
            cfg.latent_temperature if temperature is None else float(temperature),
        )
    _check_cfg(cfg)
    device = torch.device(device)
    context = context.to(device=device, dtype=torch.long)
    B, C = context.shape
    C = min(int(C), cfg.ctx_len - cfg.n_thoughts)
    if C < 1:
        raise ValueError("context is too short for the configured number of thoughts")
    context = context[:, -C:]

    zeros = torch.zeros(B, C, dtype=torch.long, device=device)
    causal = _causal_mask(C, device)
    _, _, extras = model(
        context, zeros, zeros, causal, return_kv=True,
    )
    kv = extras["kv"]
    moe_losses = []
    if "moe_lb" in extras:
        moe_losses.append(extras["moe_lb"])

    n_thoughts = cfg.n_thoughts
    emb = (
        model.thought_bos.view(1, 1, -1).expand(B, 1, -1)
        + model.thought_pos.weight[0].view(1, 1, -1)
    )
    temp = cfg.latent_temperature if temperature is None else float(temperature)

    symbols: list[torch.Tensor] = []
    embeds: list[torch.Tensor] = []
    all_logits: list[torch.Tensor] = []
    all_probs: list[torch.Tensor] = []

    for i in range(n_thoughts):
        z = torch.zeros(B, 1, dtype=torch.long, device=device)
        mask = torch.ones(B, 1, 1, C + i + 1, dtype=torch.bool, device=device)
        _, _, ex = model(
            input_embeds=emb, times=z, region=z, mask=mask,
            kv_cache=kv, return_kv=True, need_hidden=True,
        )
        if "moe_lb" in ex:
            moe_losses.append(ex["moe_lb"])
        hidden = ex["hidden"][:, -1]              # (B, D)
        step_logits = model.latent_head(hidden)   # (B, S)
        idx, sym_emb, probs = _sample_symbols(
            step_logits, model.codebook, temp, training=training
        )
        symbols.append(idx)
        embeds.append(sym_emb)
        all_logits.append(step_logits)
        all_probs.append(probs)
        kv = ex["kv"]
        if i + 1 < n_thoughts:
            emb = (sym_emb + model.thought_pos.weight[i + 1]).unsqueeze(1)

    result = {
        "symbols": torch.stack(symbols, dim=1),        # (B, N)
        "symbol_embeds": torch.stack(embeds, dim=1),   # (B, N, D)
        "logits": torch.stack(all_logits, dim=1),      # (B, N, S)
        "probs": torch.stack(all_probs, dim=1),        # (B, N, S)
        "context_len": C,
    }
    if moe_losses:
        result["moe_lb"] = torch.stack(moe_losses).mean()
    return result


def _make_recall_batch(batch: int, cfg: Config, device, length: int):
    """Synthetic associative-recall batch for the latent channel.

    The context holds ``K`` random key/value pairs, followed by a local window
    containing only the query keys (never the values), permuted independently
    from the table.  The continuation is exactly the values in query order,
    with no padding targets to dilute the recall loss.  Because the values are
    independent random bytes, the previous target bytes do not reveal the next
    one: the private code must carry the table and its requested ordering.
    Returns ``(x, cutoff)`` or ``None``.
    """
    n_thoughts = cfg.n_thoughts
    window = cfg.latent_local_window
    c_min = n_thoughts + window - 1
    c_max = min(length - n_thoughts, length - 8)
    if c_min > c_max:
        return None

    # Keep the table within the discrete alphabet's plausible capacity
    # (log2(S) bits per symbol) and inside the local window.
    capacity = max(1, int(n_thoughts * math.log2(cfg.n_latent_symbols)) // 8)
    pairs = min(
        cfg.latent_recall_pairs, capacity // 2, window, length - window - 9, 256
    )
    if pairs < 1:
        return None

    c_low = max(c_min, window + 2 * pairs + 1)
    c_max = min(c_max, length - pairs)
    if c_low > c_max:
        return None
    cutoff = random.randint(c_low, c_max)

    x = torch.full((batch, length), 32, dtype=torch.long, device=device)  # spaces
    x[:, 0] = cfg.bos_token
    keys = torch.rand((batch, 256), device=device).topk(pairs, dim=1).indices
    values = torch.randint(0, 256, (batch, pairs), device=device)
    order = _row_permutations(batch, pairs, device)
    query_keys = keys.gather(1, order)
    query_values = values.gather(1, order)
    for b in range(batch):
        start = random.randint(1, cutoff - window - 2 * pairs)
        for i in range(pairs):
            x[b, start + 2 * i] = keys[b, i]
            x[b, start + 2 * i + 1] = values[b, i]
        # query keys are the only part of the table visible locally
        x[b, cutoff - window:cutoff - window + pairs] = query_keys[b]
        x[b, cutoff:cutoff + pairs] = query_values[b]
    return x[:, :cutoff + pairs], cutoff


def _prepare_batch(x, cfg: Config, device, cutoff: int | None = None):
    """Split byte rows into ``(context, local_window, continuation, cutoff)``."""
    _check_cfg(cfg)
    device = torch.device(device)
    length = min(int(x.shape[1]), cfg.ctx_len)
    x = x[:, :length].to(device).long().clone()
    x[:, 0] = cfg.bos_token

    n_thoughts = cfg.n_thoughts
    window = cfg.latent_local_window
    c_min = n_thoughts + window - 1
    continuation_reserve = 8 if cutoff is None else 1
    c_max = min(cfg.ctx_len - n_thoughts, length - continuation_reserve)
    if c_min > c_max:
        needed = max(
            2 * n_thoughts + window - 1,
            n_thoughts + window - 1 + continuation_reserve,
        )
        raise ValueError(
            f"ctx_len={length} is too small for n_thoughts={n_thoughts} "
            f"and latent_local_window={window}; need at least {needed} bytes."
        )
    if cutoff is None:
        cutoff = random.randint(c_min, c_max)
    cutoff = max(c_min, min(c_max, int(cutoff)))
    if length - cutoff < 1:
        raise ValueError("the continuation must contain at least one byte")

    return (
        x[:, :cutoff],
        x[:, cutoff - window:cutoff],
        x[:, cutoff:],
        cutoff,
    )


def _decode_loss(model, symbol_embeds, local, targets, cfg: Config, device,
                 local_target=None, compute_local: bool = True):
    """Continuation CE from symbols + local window (or local window alone)."""
    window = cfg.latent_local_window
    n_thoughts = 0 if symbol_embeds is None else symbol_embeds.shape[1]
    byte_in = torch.cat([local, targets[:, :-1]], dim=1)
    byte_emb = model.tok_emb(byte_in)
    if symbol_embeds is not None:
        thought = symbol_embeds + model.thought_pos.weight[:n_thoughts].unsqueeze(0)
        emb = torch.cat([thought, byte_emb], dim=1)
    else:
        emb = byte_emb
    seq_len = emb.shape[1]

    zeros = torch.zeros(1, seq_len, dtype=torch.long, device=device)
    causal = _causal_mask(seq_len, device)
    logits, _, extras = model(
        input_embeds=emb, times=zeros, region=zeros, mask=causal,
        n_loops=cfg.max_loops,
    )
    byte_logits = logits[:, n_thoughts:]                 # (B, W+T-1, V)
    cont_logits = byte_logits[:, window - 1:]            # (B, T, V)
    lat_cont = F.cross_entropy(
        cont_logits.reshape(-1, cfg.vocab_size), targets.reshape(-1)
    )
    if compute_local and window > 1:
        target = local if local_target is None else local_target
        local_logits = byte_logits[:, : window - 1]
        lat_local = F.cross_entropy(
            local_logits.reshape(-1, cfg.vocab_size), target[:, 1:].reshape(-1)
        )
    else:
        lat_local = lat_cont.new_zeros(())
    return lat_cont, lat_local, extras.get("moe_lb")


def latent_batch_loss(model, x, cfg: Config, device, cutoff: int | None = None,
                      training: bool = True):
    """Compressed-context loss + alphabet-usage regulariser.

    ``x`` is a batch of raw byte rows (at least ``cfg.ctx_len`` long).  A shared
    cutoff splits each row into a context and a continuation.  The continuation
    is predicted from the emitted symbols plus only the last
    ``cfg.latent_local_window`` bytes of the context.
    """
    if cfg.latent_saint:
        from .saint import latent_batch_loss_saint

        return latent_batch_loss_saint(
            model, x, cfg, device, cutoff=cutoff, training=training
        )
    _check_cfg(cfg)
    device = torch.device(device)
    length = min(int(x.shape[1]), cfg.ctx_len)
    is_recall = False
    if (
        training
        and cutoff is None
        and cfg.latent_recall_prob > 0.0
        and random.random() < cfg.latent_recall_prob
    ):
        recall = _make_recall_batch(int(x.shape[0]), cfg, device, length)
        if recall is not None:
            x, cutoff = recall
            is_recall = True

    context, local, targets, cutoff = _prepare_batch(x, cfg, device, cutoff)

    sym = summarize(
        model, context, cfg, device, training=training,
        temperature=cfg.latent_temperature if training else 0.0,
    )

    local_input = local
    if training and cfg.latent_window_dropout > 0.0:
        local_input = _mask_local_windows(
            local, cfg.latent_window_dropout, cfg.mask_token
        )

    # Second pass: the long-range context is available *only* through the
    # private symbols.  The local window supplies spelling and recency.
    lat_cont, lat_local, decode_moe_lb = _decode_loss(
        model, sym["symbol_embeds"], local_input, targets, cfg, device,
        local_target=local, compute_local=True,
    )

    # Encourage the model to use its alphabet instead of collapsing to one
    # symbol.  Marginal entropy is normalised to [0, 1] by log(alphabet size).
    marginal = sym["probs"].mean(dim=(0, 1))
    probability_entropy = _normalised_entropy(marginal)
    usage_entropy = _symbol_usage_entropy(sym["symbols"], cfg.n_latent_symbols)

    # The local term is a reconstruction loss whenever the window was masked.
    total = cfg.w_latent * (lat_cont + 0.25 * lat_local)
    total = total - cfg.w_latent_entropy * probability_entropy
    moe_losses = [
        loss for loss in (sym.get("moe_lb"), decode_moe_lb) if loss is not None
    ]
    latent_moe_lb = torch.stack(moe_losses).mean() if moe_losses else None
    if latent_moe_lb is not None:
        total = total + cfg.w_moe_lb * latent_moe_lb

    result = {
        "lat_cont": lat_cont,
        "lat_local": lat_local,
        "lat_ent": usage_entropy,
        "lat_total": total,
        "lat_recall": lat_cont.new_tensor(float(is_recall)),
        "symbols": sym["symbols"].detach(),
        "context_len": cutoff,
    }
    if latent_moe_lb is not None:
        result["lat_moe_lb"] = latent_moe_lb
    return result


@torch.no_grad()
def code_ablation(model, x, cfg: Config, device, cutoff: int | None = None,
                  recall: bool = False, mask_local: bool = False):
    """Measure whether the symbols actually carry information.

    Returns continuation CE for the real code, a within-row shuffle of the
    code, random symbols, and a blank code (same positions, zero vectors).
    ``shuffle_penalty`` and ``gain_over_blank`` are the useful numbers:

    * positive ``shuffle_penalty`` -> symbol order/content matters,
    * positive ``gain_over_blank`` -> the code beats a no-content placeholder.

    A model that ignores its symbols and reads only the local window will show
    near-zero penalties even if ``true`` CE is low.
    """
    if cfg.latent_saint:
        from .saint import code_ablation_saint

        return code_ablation_saint(
            model, x, cfg, device, cutoff=cutoff,
            recall=recall, mask_local=mask_local,
        )
    _check_cfg(cfg)
    device = torch.device(device)
    if recall:
        length = min(int(x.shape[1]), cfg.ctx_len)
        made = _make_recall_batch(int(x.shape[0]), cfg, device, length)
        if made is None:
            raise ValueError("no room for a recall batch with this configuration")
        x, cutoff = made
    context, local, targets, cutoff = _prepare_batch(x, cfg, device, cutoff)
    local_input = (
        torch.full_like(local, cfg.mask_token) if mask_local else local
    )
    local_target = local

    was_training = model.training
    model.eval()
    try:
        sym = summarize(
            model, context, cfg, device, training=False, temperature=0.0
        )
        true_loss, true_local, _ = _decode_loss(
            model, sym["symbol_embeds"], local_input, targets, cfg, device,
            local_target=local_target,
        )

        symbols = sym["symbols"]
        batch, n_thoughts = symbols.shape
        shuffled = _shuffle_symbols(symbols)
        shuffled_loss, shuffled_local, _ = _decode_loss(
            model, model.codebook(shuffled), local_input, targets, cfg, device,
            local_target=local_target,
        )

        random_symbols = torch.randint(
            0, cfg.n_latent_symbols, (batch, n_thoughts), device=device
        )
        random_loss, random_local, _ = _decode_loss(
            model, model.codebook(random_symbols), local_input, targets, cfg, device,
            local_target=local_target,
        )

        blank_loss, blank_local, _ = _decode_loss(
            model, torch.zeros_like(sym["symbol_embeds"]), local_input, targets,
            cfg, device, local_target=local_target,
        )
    finally:
        if was_training:
            model.train()

    return {
        "true": float(true_loss),
        "shuffled": float(shuffled_loss),
        "random": float(random_loss),
        "blank": float(blank_loss),
        "gain_over_blank": float(blank_loss - true_loss),
        "shuffle_penalty": float(shuffled_loss - true_loss),
        "random_penalty": float(random_loss - true_loss),
        "true_local": float(true_local),
        "shuffled_local": float(shuffled_local),
        "random_local": float(random_local),
        "blank_local": float(blank_local),
        "local_gain": float(blank_local - true_local),
        "symbols": symbols,
        "cutoff": cutoff,
        "recall": recall,
        "mask_local": mask_local,
    }


@torch.no_grad()
def latent_prefill(model, symbols, local, cfg: Config, device):
    """Prefill a compressed sequence ``[symbols, local bytes]``.

    Returns ``(last_logits, kv_cache, sequence_length)``.  The mask matches
    training: thoughts form a causal block, then local bytes are causal.
    """
    n_thoughts = symbols.shape[1]
    thought = model.codebook(symbols) + model.thought_pos.weight[:n_thoughts].unsqueeze(0)
    emb = torch.cat([thought, model.tok_emb(local)], dim=1)
    seq_len = emb.shape[1]
    device = torch.device(device)
    zeros = torch.zeros(1, seq_len, dtype=torch.long, device=device)
    causal = _causal_mask(seq_len, device)
    logits, _, extras = model(
        input_embeds=emb, times=zeros, region=zeros, mask=causal, return_kv=True,
    )
    return logits[:, -1, :], extras["kv"], seq_len


@torch.no_grad()
def generate_latent(model, prompt_bytes, cfg: Config, device,
                    max_new_tokens: int = 200, temperature: float = 0.9,
                    top_p: float = 0.95, code_temperature: float = 0.0):
    """Generate from the private code instead of the full raw prompt.

    The prompt is read once to emit symbols; decoding then sees the symbols
    plus a short local byte window.  Returns ``(continuation_bytes, symbols,
    stats)``.
    """
    if cfg.latent_saint:
        from .saint import generate_saint

        return generate_saint(
            model, prompt_bytes, cfg, device, max_new_tokens,
            temperature, top_p, code_temperature,
        )
    _check_cfg(cfg)
    from .generate import sample_logits, _autocast

    model.eval()
    device = torch.device(device)
    ctx = cfg.ctx_len
    n_thoughts = cfg.n_thoughts
    window = cfg.latent_local_window
    if n_thoughts + window >= ctx:
        raise ValueError("n_thoughts + latent_local_window must be < ctx_len")

    max_context = max(1, ctx - n_thoughts - 8)
    # Keep the most recent bytes; the local window and continuation care about
    # recency more than the oldest part of a very long prompt.
    prompt = list(prompt_bytes)[-max_context:]
    context = torch.tensor(
        [[cfg.bos_token] + prompt], dtype=torch.long, device=device
    )
    context_len = context.shape[1]

    with _autocast(device):
        sym = summarize(
            model, context, cfg, device, training=False,
            temperature=code_temperature,
        )
    symbols = sym["symbols"]

    local = context[:, -window:]
    if local.shape[1] < window:
        pad = torch.full(
            (1, window - local.shape[1]), cfg.bos_token,
            dtype=torch.long, device=device,
        )
        local = torch.cat([pad, local], dim=1)

    with _autocast(device):
        last_logits, kv, seq_len = latent_prefill(model, symbols, local, cfg, device)

    output: list[int] = []
    banned = [cfg.bos_token, cfg.mask_token]
    for _ in range(max_new_tokens):
        if seq_len + len(output) >= ctx:
            break
        tok, _ = sample_logits(
            last_logits, temperature, top_p,
            greedy=(temperature <= 0.0), banned=banned,
        )
        tok = int(tok.item())
        output.append(tok)

        z = torch.zeros(1, 1, dtype=torch.long, device=device)
        mask = torch.ones(
            1, 1, 1, seq_len + len(output), dtype=torch.bool, device=device
        )
        with _autocast(device):
            logits, _, extras = model(
                torch.tensor([[tok]], dtype=torch.long, device=device),
                z, z, mask, kv_cache=kv, return_kv=True,
            )
        kv = extras["kv"]
        last_logits = logits[:, -1, :]

    unique = len(set(symbols[0].detach().cpu().tolist()))
    bits_per_symbol = math.log2(cfg.n_latent_symbols)
    stats = {
        "unique": unique,
        "n_thoughts": n_thoughts,
        "context_bytes": context_len,
        "bits_per_symbol": bits_per_symbol,
        "total_bits": bits_per_symbol * n_thoughts,
        "bytes_per_symbol": context_len / max(n_thoughts, 1),
    }
    return bytes(output), symbols, stats
