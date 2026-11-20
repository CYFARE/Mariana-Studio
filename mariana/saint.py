"""SAINT: adaptive hierarchical discrete/continuous thought for Mariana.

The ordinary AR backbone is deliberately unaware of this module. SAINT writes a
private memory and reads it through compressed decoding or draft-only diffusion
cross-attention. Padded thoughts are masked everywhere; they are not silently
counted as content.
"""

from __future__ import annotations

import math
import random

import torch
import torch.nn.functional as F

from .config import Config


def _sample(logits, codebook, temperature: float, training: bool):
    logits = logits.float()
    if training:
        tau = max(float(temperature), 1e-3)
        straight = F.gumbel_softmax(logits, tau=tau, hard=True, dim=-1)
        indices = straight.argmax(dim=-1)
        embeds = straight @ codebook.weight[: logits.shape[-1]].float()
        probabilities = F.softmax(logits, dim=-1)
        return indices, embeds, probabilities
    if temperature > 0.0:
        probabilities = F.softmax(logits / max(float(temperature), 1e-6), dim=-1)
        indices = torch.multinomial(
            probabilities.reshape(-1, probabilities.shape[-1]), 1
        ).view(probabilities.shape[:-1])
    else:
        probabilities = F.softmax(logits, dim=-1)
        indices = probabilities.argmax(dim=-1)
    return indices, codebook(indices), probabilities


def _pad_steps(items, target: int, fill=0):
    current = torch.stack(items, dim=1)
    missing = target - current.shape[1]
    if missing <= 0:
        return current
    shape = list(current.shape)
    shape[1] = missing
    if torch.is_tensor(fill):
        padding = fill.reshape(1, 1, *([-1] if current.ndim == 3 else []))
        padding = padding.expand(*shape)
    else:
        padding = current.new_full(shape, fill)
    return torch.cat([current, padding], dim=1)


def _masked_mean(loss: torch.Tensor, mask: torch.Tensor):
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def summarize_saint(model, context, cfg: Config, device, training: bool,
                    temperature: float):
    from .latent import _causal_mask, _check_cfg

    _check_cfg(cfg)
    if cfg.latent_continuous_slots < 1:
        raise ValueError("latent_continuous_slots must be >= 1")
    if cfg.latent_block_size < 1:
        raise ValueError("latent_block_size must be >= 1")

    device = torch.device(device)
    context = context.to(device=device, dtype=torch.long)
    B, C = context.shape
    C = min(int(C), cfg.ctx_len - cfg.n_thoughts)
    if C < 1:
        raise ValueError("context is too short for the configured number of thoughts")
    context = context[:, -C:]
    zeros = torch.zeros(B, C, dtype=torch.long, device=device)
    _, _, extras = model(
        context, zeros, zeros, _causal_mask(C, device),
        return_kv=True, need_hidden=True,
    )
    kv = extras["kv"]
    context_hidden = extras["hidden"].mean(dim=1)
    moe_losses = [extras["moe_lb"]] if "moe_lb" in extras else []

    eos = cfg.n_latent_symbols
    pad = eos + 1
    minimum = max(1, min(cfg.latent_min_thoughts, cfg.n_thoughts))
    ended = torch.zeros(B, dtype=torch.bool, device=device)
    lengths = torch.zeros(B, dtype=torch.long, device=device)
    emb = (
        model.thought_bos.view(1, 1, -1).expand(B, 1, -1)
        + model.thought_pos.weight[0].view(1, 1, -1)
    )

    symbols, symbol_embeds, logits, probabilities = [], [], [], []
    grammar_logits = []
    actions, contents, continuous, kl_steps = [], [], [], []
    n_steps = cfg.n_thoughts
    for i in range(cfg.n_thoughts):
        z = torch.zeros(B, 1, dtype=torch.long, device=device)
        mask = torch.ones(B, 1, 1, C + i + 1, dtype=torch.bool, device=device)
        _, _, ex = model(
            input_embeds=emb, times=z, region=z, mask=mask,
            kv_cache=kv, return_kv=True, need_hidden=True,
        )
        if "moe_lb" in ex:
            moe_losses.append(ex["moe_lb"])
        hidden = ex["hidden"][:, -1]
        raw_step_logits = model.latent_head(hidden)
        step_logits = raw_step_logits
        if i < minimum:
            step_logits = step_logits.clone()
            step_logits[:, eos] = float("-inf")
        idx, sampled_emb, probs = _sample(
            step_logits, model.codebook, temperature, training
        )
        active = ~ended
        content = active & (idx != eos)
        emitted = torch.where(active, idx, torch.full_like(idx, pad))
        pad_emb = model.codebook.weight[pad].view(1, -1).expand(B, -1)
        raw_emb = torch.where(active.unsqueeze(-1), sampled_emb, pad_emb)

        params = model.continuous_head(hidden).view(
            B, cfg.latent_continuous_slots, 2, cfg.dim
        )
        mu, log_var = params[:, :, 0], params[:, :, 1].clamp(-6.0, 6.0)
        if training:
            noise = torch.randn_like(mu)
            z_cont = mu + noise * (0.5 * log_var).exp()
        else:
            z_cont = mu
        content_3d = content[:, None, None]
        z_cont = z_cont * content_3d
        kl = 0.5 * (log_var.exp() + mu.square() - 1.0 - log_var).mean(dim=(1, 2))
        kl_steps.append(kl * content)

        symbols.append(emitted)
        symbol_embeds.append(raw_emb)
        logits.append(step_logits)
        grammar_logits.append(raw_step_logits)
        probabilities.append(probs)
        actions.append(active)
        contents.append(content)
        continuous.append(z_cont)
        lengths = lengths + content.long()
        ended = ended | (active & (idx == eos))

        if not training and bool(ended.all()):
            n_steps = i + 1
            break
        if i + 1 < cfg.n_thoughts:
            emb = (raw_emb + model.thought_pos.weight[i + 1]).unsqueeze(1)
        kv = ex["kv"]

    symbols = _pad_steps(symbols, cfg.n_thoughts, pad)
    symbol_embeds = _pad_steps(
        symbol_embeds, cfg.n_thoughts, model.codebook.weight[pad]
    )
    logits = _pad_steps(logits, cfg.n_thoughts)
    grammar_logits = _pad_steps(grammar_logits, cfg.n_thoughts)
    probabilities = _pad_steps(probabilities, cfg.n_thoughts)
    actions = _pad_steps(actions, cfg.n_thoughts, False)
    contents = _pad_steps(contents, cfg.n_thoughts, False)
    continuous = _pad_steps(continuous, cfg.n_thoughts)
    kl_per_example = torch.stack(kl_steps, dim=1).sum(dim=1) / cfg.n_thoughts

    slots = cfg.latent_continuous_slots
    continuous_valid = contents[:, :, None].expand(B, cfg.n_thoughts, slots)
    flat = continuous.reshape(B, cfg.n_thoughts * slots, cfg.dim)
    flat_valid = continuous_valid.reshape(B, cfg.n_thoughts * slots)
    flat = flat + model.continuous_pos.weight.unsqueeze(0)
    flat = flat * flat_valid.unsqueeze(-1)
    for refiner in model.latent_refiners:
        flat = refiner(flat, flat_valid)
    continuous = model.continuous_norm(flat).reshape(
        B, cfg.n_thoughts, slots, cfg.dim
    ) * continuous_valid.unsqueeze(-1)

    child = (symbol_embeds + continuous.mean(dim=2)) * contents.unsqueeze(-1)
    P = model.n_latent_parents
    block = cfg.latent_block_size
    padded_n = P * block
    child_blocks = F.pad(child, (0, 0, 0, padded_n - cfg.n_thoughts)).view(
        B, P, block, cfg.dim
    )
    mask_blocks = F.pad(
        contents, (0, padded_n - cfg.n_thoughts), value=False
    ).view(B, P, block)
    counts = mask_blocks.sum(dim=2).clamp_min(1).unsqueeze(-1)
    means = child_blocks.sum(dim=2) / counts
    maxima = child_blocks.masked_fill(~mask_blocks.unsqueeze(-1), -1e9).amax(dim=2)
    parent_mask = mask_blocks.any(dim=2)
    maxima = torch.where(parent_mask.unsqueeze(-1), maxima, torch.zeros_like(maxima))
    parent_hidden = (
        model.parent_proj(torch.cat([means, maxima], dim=-1))
        + model.parent_pos.weight.unsqueeze(0)
    )
    parent_logits = model.parent_head(parent_hidden)
    parent_symbols, parent_embeds, parent_probs = _sample(
        parent_logits, model.parent_codebook, temperature, training
    )
    parent_embeds = parent_embeds * parent_mask.unsqueeze(-1)
    parent_symbols = torch.where(
        parent_mask, parent_symbols, torch.zeros_like(parent_symbols)
    )

    result = {
        "symbols": symbols,
        "symbol_embeds": symbol_embeds,
        "logits": logits,
        "grammar_logits": grammar_logits,
        "probs": probabilities,
        "action_mask": actions,
        "content_mask": contents,
        "lengths": lengths,
        "terminated": ended,
        "n_steps": n_steps,
        "continuous": continuous,
        "continuous_kl": kl_per_example,
        "parent_symbols": parent_symbols,
        "parent_embeds": parent_embeds,
        "parent_hidden": parent_hidden,
        "parent_logits": parent_logits,
        "parent_probs": parent_probs,
        "parent_mask": parent_mask,
        "value": model.latent_value(context_hidden.detach()).squeeze(-1),
        "context_len": C,
    }
    if moe_losses:
        result["moe_lb"] = torch.stack(moe_losses).mean()
    return result


def build_memory(model, sym, cfg: Config, training: bool = False):
    B, N = sym["symbols"].shape
    thought = (
        sym["symbol_embeds"]
        + model.thought_pos.weight[:N].unsqueeze(0)
        + model.latent_type.weight[0]
    )
    thought = thought * sym["content_mask"].unsqueeze(-1)

    slots = cfg.latent_continuous_slots
    continuous = sym["continuous"] + model.latent_type.weight[1]
    continuous_mask = sym["content_mask"][:, :, None].expand(B, N, slots)
    continuous = continuous.reshape(B, N * slots, cfg.dim)
    continuous_mask = continuous_mask.reshape(B, N * slots)

    parents = (
        sym["parent_embeds"]
        + model.parent_pos.weight.unsqueeze(0)
        + model.latent_type.weight[2]
    ) * sym["parent_mask"].unsqueeze(-1)

    memory = torch.cat([thought, continuous, parents], dim=1)
    memory_mask = torch.cat(
        [sym["content_mask"], continuous_mask, sym["parent_mask"]], dim=1
    )
    if training:
        keep_discrete = torch.rand(B, 1, device=memory.device) >= cfg.latent_discrete_dropout
        keep_continuous = torch.rand(B, 1, device=memory.device) >= cfg.latent_continuous_dropout
        keep_continuous = keep_continuous | ~keep_discrete
        discrete_positions = torch.cat([
            torch.ones(B, N, dtype=torch.bool, device=memory.device),
            torch.zeros(B, N * slots, dtype=torch.bool, device=memory.device),
            torch.ones(B, model.n_latent_parents, dtype=torch.bool, device=memory.device),
        ], dim=1)
        keep = torch.where(discrete_positions, keep_discrete, keep_continuous)
        memory_mask = memory_mask & keep
    memory = memory * memory_mask.unsqueeze(-1)
    return memory, memory_mask


def _memory_key_mask(memory_mask, byte_length: int):
    return torch.cat([
        memory_mask,
        torch.ones(
            memory_mask.shape[0], byte_length, dtype=torch.bool,
            device=memory_mask.device,
        ),
    ], dim=1)


def _attention_mask(memory_mask, byte_length: int):
    B, M = memory_mask.shape
    L = M + byte_length
    causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=memory_mask.device))
    key_valid = _memory_key_mask(memory_mask, byte_length)
    return causal[None, None] & key_valid[:, None, None, :]


def _decode_memory(model, memory, memory_mask, local, targets, cfg: Config,
                   local_target=None, compute_local: bool = True):
    B, M, _ = memory.shape
    window = cfg.latent_local_window
    byte_in = torch.cat([local, targets[:, :-1]], dim=1)
    emb = torch.cat([memory, model.tok_emb(byte_in)], dim=1)
    zeros = torch.zeros(B, emb.shape[1], dtype=torch.long, device=emb.device)
    logits, _, extras = model(
        input_embeds=emb, times=zeros, region=zeros,
        mask=_attention_mask(memory_mask, byte_in.shape[1]), n_loops=cfg.max_loops,
    )
    byte_logits = logits[:, M:]
    cont_logits = byte_logits[:, window - 1:]
    cont_nll = F.cross_entropy(
        cont_logits.reshape(-1, cfg.vocab_size), targets.reshape(-1), reduction="none"
    ).view(B, -1).mean(dim=1)
    if compute_local and window > 1:
        target = local if local_target is None else local_target
        local_logits = byte_logits[:, :window - 1]
        local_nll = F.cross_entropy(
            local_logits.reshape(-1, cfg.vocab_size), target[:, 1:].reshape(-1),
            reduction="none",
        ).view(B, -1).mean(dim=1)
    else:
        local_nll = cont_nll.new_zeros(B)
    accuracy = (cont_logits.argmax(dim=-1) == targets).float().mean(dim=1)
    return {
        "cont": cont_nll.mean(),
        "cont_per_example": cont_nll,
        "local": local_nll.mean(),
        "accuracy": accuracy,
        "moe_lb": extras.get("moe_lb"),
    }


def make_grammar_batch(batch: int, cfg: Config, device, length: int):
    """Key/operator/key tasks whose values must be retrieved and composed."""
    from .latent import _row_permutations

    N, window = cfg.n_thoughts, cfg.latent_local_window
    queries = min(max(1, cfg.latent_recall_pairs), max(1, window // 3))
    capacity = max(1, int(N * math.log2(cfg.n_latent_symbols)) // 8)
    pairs = min(max(2, cfg.latent_recall_pairs), capacity // 2, 64)
    queries = min(queries, pairs)
    c_min = N + window - 1
    c_low = max(c_min, window + 2 * pairs + 1)
    c_max = min(length - N, length - queries, length - 8)
    if pairs < 2 or c_low > c_max:
        return None

    cutoff = random.randint(c_low, c_max)
    if cutoff - window - 2 * pairs < 1:
        return None
    x = torch.full((batch, length), 32, dtype=torch.long, device=device)
    x[:, 0] = cfg.bos_token
    keys = torch.rand(batch, 240, device=device).topk(pairs, dim=1).indices + 16
    values = torch.randint(0, 256, (batch, pairs), device=device)
    query_order = _row_permutations(batch, pairs, device)
    left = keys.gather(1, query_order)
    right_order = torch.randint(0, pairs, (batch, pairs), device=device)
    right = keys.gather(1, right_order)
    operations = torch.randint(0, 4, (batch, queries), device=device)
    left = left[:, :queries]
    right = right[:, :queries]
    left_values = values.gather(1, query_order[:, :queries])
    right_values = values.gather(1, right_order[:, :queries])

    results = torch.empty_like(left_values)
    results = torch.where(operations == 0, left_values ^ right_values, results)
    results = torch.where(operations == 1, left_values + right_values, results)
    results = torch.where(operations == 2, left_values - right_values, results)
    shifts = right_values & 7
    rotated = (left_values << shifts) | (left_values >> ((8 - shifts) & 7))
    results = torch.where(operations == 3, rotated, results) & 255

    for b in range(batch):
        start = random.randint(1, cutoff - window - 2 * pairs)
        for i in range(pairs):
            x[b, start + 2 * i] = keys[b, i]
            x[b, start + 2 * i + 1] = values[b, i]
        for i in range(queries):
            base = cutoff - window + 3 * i
            x[b, base] = left[b, i]
            x[b, base + 1] = operations[b, i] + 1
            x[b, base + 2] = right[b, i]
    x[:, cutoff:cutoff + queries] = results
    return x[:, :cutoff + queries], cutoff


def _grammar_losses(model, sym, cfg: Config):
    logits = sym["grammar_logits"].float()
    actions = sym["action_mask"]
    contents = sym["content_mask"]
    symbols = sym["symbols"]
    eos = cfg.n_latent_symbols
    target = symbols[:, 1:].clamp_max(eos)
    next_loss = _masked_mean(
        F.cross_entropy(
            logits[:, :-1].reshape(-1, eos + 1), target.reshape(-1), reduction="none"
        ).view(target.shape), actions[:, 1:].float(),
    )

    left, right = sym["symbol_embeds"][:, :-1], sym["symbol_embeds"][:, 1:]
    ordered_pair = torch.cat([left, right, left * right], dim=-1)
    reversed_pair = torch.cat([right, left, left * right], dim=-1)
    if cfg.n_thoughts >= 3:
        pair_target = symbols[:, 2:].clamp_max(eos)
        pair_mask = contents[:, :-2] & contents[:, 1:-1] & actions[:, 2:]
        pair_logits = model.grammar_head(
            F.silu(model.grammar_pair(ordered_pair[:, :-1]))
        )
        pair_loss = _masked_mean(
            F.cross_entropy(
                pair_logits.reshape(-1, eos + 1), pair_target.reshape(-1),
                reduction="none",
            ).view(pair_target.shape), pair_mask.float(),
        )
    else:
        pair_loss = next_loss.new_zeros(())

    adjacent = contents[:, :-1] & contents[:, 1:]
    order_logits = model.order_head(ordered_pair)
    reverse_logits = model.order_head(reversed_pair)
    order_target = torch.ones_like(adjacent, dtype=torch.long)
    reverse_target = torch.zeros_like(adjacent, dtype=torch.long)
    order_loss = 0.5 * (
        _masked_mean(
            F.cross_entropy(
                order_logits.reshape(-1, 2), order_target.reshape(-1), reduction="none"
            ).view(adjacent.shape), adjacent.float(),
        ) + _masked_mean(
            F.cross_entropy(
                reverse_logits.reshape(-1, 2), reverse_target.reshape(-1), reduction="none"
            ).view(adjacent.shape), adjacent.float(),
        )
    )

    parent_target = sym["parent_symbols"][:, 1:]
    parent_mask = sym["parent_mask"][:, 1:]
    parent_loss = _masked_mean(
        F.cross_entropy(
            sym["parent_logits"][:, :-1].reshape(-1, eos),
            parent_target.reshape(-1), reduction="none",
        ).view(parent_target.shape), parent_mask.float(),
    ) if model.n_latent_parents > 1 else next_loss.new_zeros(())
    grammar = next_loss + pair_loss + 0.25 * order_loss + parent_loss
    return grammar, pair_loss, order_loss


def _hierarchy_loss(model, sym, cfg: Config):
    B, N = sym["symbols"].shape
    block = cfg.latent_block_size
    readout = model.parent_readout(sym["parent_hidden"])
    expanded = readout[:, :, None, :].expand(B, model.n_latent_parents, block, cfg.n_latent_symbols)
    child_logits = expanded.reshape(B, -1, cfg.n_latent_symbols)[:, :N]
    child_target = sym["symbols"].clamp_max(cfg.n_latent_symbols - 1)
    return _masked_mean(
        F.cross_entropy(
            child_logits.reshape(-1, cfg.n_latent_symbols), child_target.reshape(-1),
            reduction="none",
        ).view(B, N), sym["content_mask"].float(),
    )


def latent_batch_loss_saint(model, x, cfg: Config, device, cutoff=None,
                            training: bool = True):
    from .latent import (
        _check_cfg, _make_recall_batch, _mask_local_windows,
        _normalised_entropy, _prepare_batch, _symbol_usage_entropy,
    )

    _check_cfg(cfg)
    device = torch.device(device)
    length = min(int(x.shape[1]), cfg.ctx_len)
    is_recall = is_grammar = False
    if training and cutoff is None:
        draw = random.random()
        if draw < cfg.latent_recall_prob:
            made = _make_recall_batch(int(x.shape[0]), cfg, device, length)
            if made is not None:
                x, cutoff = made
                is_recall = True
        elif draw < cfg.latent_recall_prob + cfg.latent_grammar_prob:
            made = make_grammar_batch(int(x.shape[0]), cfg, device, length)
            if made is not None:
                x, cutoff = made
                is_grammar = True

    context, local, targets, cutoff = _prepare_batch(x, cfg, device, cutoff)
    sym = summarize_saint(
        model, context, cfg, device, training,
        cfg.latent_temperature if training else 0.0,
    )
    memory, memory_mask = build_memory(model, sym, cfg, training=training)
    local_input = local
    if training and cfg.latent_window_dropout > 0.0:
        local_input = _mask_local_windows(local, cfg.latent_window_dropout, cfg.mask_token)
    decoded = _decode_memory(
        model, memory, memory_mask, local_input, targets, cfg, local_target=local
    )
    lat_cont, lat_local = decoded["cont"], decoded["local"]

    content = sym["content_mask"]
    content_probs = sym["probs"][..., :cfg.n_latent_symbols]
    content_probs = content_probs / content_probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    marginal = (content_probs * content.unsqueeze(-1)).sum(dim=(0, 1))
    marginal = marginal / content.sum().clamp_min(1)
    probability_entropy = _normalised_entropy(marginal)
    usage_entropy = _symbol_usage_entropy(
        sym["symbols"][content], cfg.n_latent_symbols
    )
    grammar, pair_grammar, order = _grammar_losses(model, sym, cfg)
    hierarchy = _hierarchy_loss(model, sym, cfg)
    kl = sym["continuous_kl"].mean()

    reward = (
        -decoded["cont_per_example"].detach()
        + cfg.w_latent_task_reward * decoded["accuracy"].detach()
        - cfg.w_latent_length * sym["lengths"].float() / cfg.n_thoughts
    )
    value = sym["value"]
    policy_gradient = lat_cont.new_zeros(())
    value_loss = lat_cont.new_zeros(())
    if training and cfg.w_latent_rl > 0.0:
        actions = sym["action_mask"]
        chosen = sym["symbols"].clamp_max(cfg.n_latent_symbols)
        log_probs = F.log_softmax(sym["logits"].float(), dim=-1).gather(
            -1, chosen.unsqueeze(-1)
        ).squeeze(-1)
        parent_log_probs = F.log_softmax(
            sym["parent_logits"].float(), dim=-1
        ).gather(-1, sym["parent_symbols"].unsqueeze(-1)).squeeze(-1)
        logp_sum = (log_probs * actions).sum(dim=1)
        logp_sum = logp_sum + (parent_log_probs * sym["parent_mask"]).sum(dim=1)
        action_count = actions.sum(dim=1) + sym["parent_mask"].sum(dim=1)
        sequence_logp = logp_sum / action_count.clamp_min(1)
        advantage = reward - value.detach()
        scale = advantage.detach().std(unbiased=False).clamp_min(1.0)
        policy_gradient = -(sequence_logp * advantage / scale).mean()
    if training and cfg.w_latent_value > 0.0:
        value_loss = F.smooth_l1_loss(value, reward)

    action_probs = sym["probs"].clamp_min(1e-9)
    action_entropy = -(action_probs * action_probs.log()).sum(dim=-1)
    action_entropy = action_entropy / math.log(cfg.n_latent_symbols + 1)
    action_entropy = _masked_mean(action_entropy, sym["action_mask"].float())

    total = cfg.w_latent * (lat_cont + 0.25 * lat_local)
    total = total - cfg.w_latent_entropy * probability_entropy
    total = total + cfg.w_latent_grammar * grammar
    total = total + cfg.w_latent_hierarchy * hierarchy
    total = total + cfg.w_latent_rl * policy_gradient
    total = total + cfg.w_latent_value * value_loss
    total = total + cfg.w_latent_kl * kl
    total = total - cfg.w_latent_policy_entropy * action_entropy

    moe_losses = [v for v in (sym.get("moe_lb"), decoded["moe_lb"]) if v is not None]
    latent_moe_lb = torch.stack(moe_losses).mean() if moe_losses else None
    if latent_moe_lb is not None:
        total = total + cfg.w_moe_lb * latent_moe_lb

    result = {
        "lat_cont": lat_cont,
        "lat_local": lat_local,
        "lat_ent": usage_entropy,
        "lat_total": total,
        "lat_recall": lat_cont.new_tensor(float(is_recall)),
        "lat_grammar_task": lat_cont.new_tensor(float(is_grammar)),
        "lat_grammar": grammar,
        "lat_pair": pair_grammar,
        "lat_order": order,
        "lat_hier": hierarchy,
        "lat_rl": policy_gradient,
        "lat_value": value_loss,
        "lat_reward": reward.mean(),
        "lat_kl": kl,
        "lat_len": sym["lengths"].float().mean(),
        "lat_parents": sym["parent_mask"].float().sum(dim=1).mean(),
        "lat_stop": sym["terminated"].float().mean(),
        "symbols": sym["symbols"].detach(),
        "parent_symbols": sym["parent_symbols"].detach(),
        "context_len": cutoff,
        "diffusion_memory": (
            memory if cfg.latent_diffusion and not (is_recall or is_grammar) else None
        ),
        "diffusion_memory_mask": (
            memory_mask if cfg.latent_diffusion and not (is_recall or is_grammar) else None
        ),
    }
    if latent_moe_lb is not None:
        result["lat_moe_lb"] = latent_moe_lb
    return result


def memory_prefill(model, memory, memory_mask, local, cfg: Config):
    B, M, _ = memory.shape
    emb = torch.cat([memory, model.tok_emb(local)], dim=1)
    zeros = torch.zeros(B, emb.shape[1], dtype=torch.long, device=emb.device)
    logits, _, extras = model(
        input_embeds=emb, times=zeros, region=zeros,
        mask=_attention_mask(memory_mask, local.shape[1]), return_kv=True,
    )
    return logits[:, -1], extras["kv"], emb.shape[1]


def _cached_byte_forward(model, token, kv_cache, memory_mask, byte_length: int):
    zeros = torch.zeros_like(token)
    key_valid = _memory_key_mask(memory_mask, byte_length)
    return model(
        token, zeros, zeros, key_valid[:, None, None, :],
        kv_cache=kv_cache, return_kv=True,
    )


@torch.no_grad()
def prompt_memory(model, prompt_bytes, cfg: Config, device,
                  code_temperature: float = 0.0):
    device = torch.device(device)
    max_context = max(1, cfg.ctx_len - cfg.n_thoughts - 8)
    prompt = list(prompt_bytes)[-max_context:]
    context = torch.tensor(
        [[cfg.bos_token] + prompt], dtype=torch.long, device=device
    )
    sym = summarize_saint(
        model, context, cfg, device, training=False, temperature=code_temperature
    )
    memory, memory_mask = build_memory(model, sym, cfg)
    return memory, memory_mask, sym, context


@torch.no_grad()
def generate_saint(model, prompt_bytes, cfg: Config, device,
                   max_new_tokens: int, temperature: float, top_p: float,
                   code_temperature: float):
    from .generate import _autocast, sample_logits

    model.eval()
    device = torch.device(device)
    with _autocast(device):
        memory, memory_mask, sym, context = prompt_memory(
            model, prompt_bytes, cfg, device, code_temperature
        )
        window = cfg.latent_local_window
        local = context[:, -window:]
        if local.shape[1] < window:
            pad = torch.full(
                (1, window - local.shape[1]), cfg.bos_token,
                dtype=torch.long, device=device,
            )
            local = torch.cat([pad, local], dim=1)
        last_logits, kv, _ = memory_prefill(
            model, memory, memory_mask, local, cfg
        )

    output = []
    for _ in range(max_new_tokens):
        tok, _ = sample_logits(
            last_logits, temperature, top_p, greedy=temperature <= 0.0,
            banned=[cfg.bos_token, cfg.mask_token],
        )
        tok = int(tok.item())
        output.append(tok)
        with _autocast(device):
            token = torch.tensor([[tok]], dtype=torch.long, device=device)
            logits, _, extras = _cached_byte_forward(
                model, token, kv, memory_mask, local.shape[1] + len(output)
            )
        kv = extras["kv"]
        last_logits = logits[:, -1]

    n_thoughts = int(sym["lengths"][0])
    terminated = bool(sym["terminated"][0])
    display_n = n_thoughts + int(terminated)
    symbols = sym["symbols"][:, :display_n]
    content_symbols = sym["symbols"][0][sym["content_mask"][0]]
    unique = len(set(content_symbols.detach().cpu().tolist()))
    n_parents = int(sym["parent_mask"][0].sum())
    bits = math.log2(cfg.n_latent_symbols)
    stats = {
        "unique": unique,
        "n_thoughts": n_thoughts,
        "max_thoughts": cfg.n_thoughts,
        "n_parents": n_parents,
        "parent_symbols": sym["parent_symbols"][:, :n_parents],
        "n_continuous": n_thoughts * cfg.latent_continuous_slots,
        "stopped": terminated,
        "n_steps": sym["n_steps"],
        "context_bytes": context.shape[1],
        "bits_per_symbol": bits,
        "total_bits": bits * (n_thoughts + n_parents),
        "bytes_per_symbol": context.shape[1] / max(n_thoughts, 1),
    }
    return bytes(output), symbols, stats


@torch.no_grad()
def code_ablation_saint(model, x, cfg: Config, device, cutoff=None,
                        recall: bool = False, mask_local: bool = False):
    from .latent import (
        _make_recall_batch, _prepare_batch, _row_permutations,
    )

    device = torch.device(device)
    if recall:
        made = _make_recall_batch(int(x.shape[0]), cfg, device, min(x.shape[1], cfg.ctx_len))
        if made is None:
            raise ValueError("no room for a recall batch with this configuration")
        x, cutoff = made
    context, local, targets, cutoff = _prepare_batch(x, cfg, device, cutoff)
    local_input = torch.full_like(local, cfg.mask_token) if mask_local else local
    was_training = model.training
    model.eval()
    try:
        sym = summarize_saint(model, context, cfg, device, False, 0.0)
        memory, memory_mask = build_memory(model, sym, cfg)
        true = _decode_memory(model, memory, memory_mask, local_input, targets, cfg, local)

        B, M, _ = memory.shape
        perm = _row_permutations(B, M, device)
        shuffled_memory = memory.gather(1, perm.unsqueeze(-1).expand(B, M, cfg.dim))
        shuffled_mask = memory_mask.gather(1, perm)
        shuffled = _decode_memory(
            model, shuffled_memory, shuffled_mask, local_input, targets, cfg, local
        )

        random_sym = dict(sym)
        random_symbols = torch.randint(
            0, cfg.n_latent_symbols, sym["symbols"].shape, device=device
        )
        random_sym["symbol_embeds"] = model.codebook(random_symbols)
        random_parents = torch.randint(
            0, cfg.n_latent_symbols, sym["parent_symbols"].shape, device=device
        )
        random_sym["parent_embeds"] = model.parent_codebook(random_parents)
        random_memory, random_mask = build_memory(model, random_sym, cfg)
        random = _decode_memory(
            model, random_memory, random_mask, local_input, targets, cfg, local
        )
        blank = _decode_memory(
            model, torch.zeros_like(memory), memory_mask, local_input, targets, cfg, local
        )
    finally:
        if was_training:
            model.train()

    return {
        "true": float(true["cont"]),
        "shuffled": float(shuffled["cont"]),
        "random": float(random["cont"]),
        "blank": float(blank["cont"]),
        "gain_over_blank": float(blank["cont"] - true["cont"]),
        "shuffle_penalty": float(shuffled["cont"] - true["cont"]),
        "random_penalty": float(random["cont"] - true["cont"]),
        "true_local": float(true["local"]),
        "shuffled_local": float(shuffled["local"]),
        "random_local": float(random["local"]),
        "blank_local": float(blank["local"]),
        "local_gain": float(blank["local"] - true["local"]),
        "symbols": sym["symbols"],
        "parent_symbols": sym["parent_symbols"],
        "cutoff": cutoff,
        "recall": recall,
        "mask_local": mask_local,
    }
