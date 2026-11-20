"""Mariana DRV-LM and NCM generator checkpoints (in-process, PyTorch)."""

from __future__ import annotations

from pathlib import Path

import torch

from ....device import pick_device
from ...validation import number
from ..catalog import TOKENIZER_ONLY_REASON, ModelError
from . import delta, done


def load_model(path: Path, device):
    """Materialise a checkpoint and return ``(kind, cfg, model)``."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    stage = state.get("stage")
    if stage == "tokenizer":
        raise ModelError(TOKENIZER_ONLY_REASON)
    if stage not in {None, "generator"} or "model" not in state or "cfg" not in state:
        raise ModelError("Select a chat-compatible Mariana DRV-LM or NCM generator checkpoint")
    cfg_values = state.get("cfg", {})
    if stage == "generator" and "levels" in cfg_values:
        from ....ncm.config import NCMConfig
        from ....ncm.generator import NCMGenerator
        from ....ncm.tokenizer import MultiscaleTokenizer

        cfg = NCMConfig.from_checkpoint(state)
        model = NCMGenerator(MultiscaleTokenizer(cfg), cfg)
        kind = "NCM"
    else:
        from ....config import Config
        from ....model import MarianaLM

        cfg = Config.from_dict(cfg_values)
        model = MarianaLM(cfg)
        kind = "DRV-LM"
    model.load_state_dict(state["model"])
    return kind, cfg, model.to(device).eval()


def _generate_ncm(cfg, model, prompt_text, request, device):
    from ....ncm.cli import prepare_prompt
    from ....ncm.layers import mbr_select

    options = request.options
    guidance = number(options, "guidance", 0.0, 0.0, 5.0, float, ModelError)
    early_exit = number(options, "early_exit", 0.0, 0.0, 1.0, float, ModelError)
    refine_steps = number(options, "refine_steps", 1, 0, 8, int, ModelError)
    refine_threshold = number(options, "refine_threshold", 0.8, 0.0, 5.0, float, ModelError)
    candidates = number(options, "candidates", 1, 1, 16, int, ModelError)
    guidance_mode = options.get("guidance_mode", "ramp")
    if guidance_mode not in {"ramp", "constant"}:
        raise ModelError("Guidance mode must be ramp or constant")
    max_tokens = min(request.max_tokens, cfg.ctx_len)
    prompt, _ = prepare_prompt(prompt_text, cfg, device)
    prompt = prompt.repeat(candidates, 1)
    tokens, stats = model.generate(
        prompt, temperature=request.temperature, top_k=request.top_k, top_p=request.top_p,
        guidance=guidance, use_cache=True, early_exit=early_exit,
        refine_steps=refine_steps, refine_threshold=refine_threshold,
        guidance_ramp=guidance_mode == "ramp",
    )
    if candidates > 1:
        tokens = mbr_select(tokens.unsqueeze(1))
    values = tokens[0, :max_tokens].tolist()
    reply = bytes(value for value in values if 0 <= value < 256).decode("utf-8", errors="replace")
    details = {
        "levels_used": stats["levels_used"], "levels": cfg.n_levels, "candidates": candidates,
        "generator_nfe": stats["generator_nfe"] * candidates,
        "decoder_nfe": stats["decoder_nfe"] * candidates,
        "refiner_nfe": stats["refiner_nfe"] * candidates,
    }
    return reply, stats["elapsed"], {"completion_tokens": len(values)}, details


def _generate_drv(cfg, model, prompt_text, request, device):
    from ....generate import _canonical_prompt, ar_generate

    max_tokens = min(request.max_tokens, max(1, cfg.ctx_len - 9))
    raw = prompt_text.encode("utf-8")
    used = _canonical_prompt(raw, cfg, max_tokens)
    output, _, elapsed = ar_generate(
        model, raw, max_new_tokens=max_tokens, temperature=request.temperature,
        top_p=request.top_p, device=device, cfg=cfg,
    )
    reply = output[len(used):].decode("utf-8", errors="replace")
    usage = {"prompt_tokens": len(used), "completion_tokens": max(0, len(output) - len(used))}
    return reply, elapsed, usage, None


def stream(gateway, entry, request):
    from . import transcript_prompt

    path = Path(entry["absolute_path"])
    device = pick_device(request.device)
    prompt_text = transcript_prompt(request.messages)
    with gateway.local_lock:
        key = ("mariana", str(path), path.stat().st_mtime_ns, str(device))
        kind, cfg, model = gateway.cached(key, lambda: load_model(path, device))
        with torch.no_grad():
            if kind == "NCM":
                reply, elapsed, usage, details = _generate_ncm(cfg, model, prompt_text, request, device)
            else:
                reply, elapsed, usage, details = _generate_drv(cfg, model, prompt_text, request, device)
    reply = reply.split("\nUser:", 1)[0].strip()
    if reply:
        yield delta(reply)
    result = {"kind": kind, "usage": usage, "prompt_format": "transcript", "elapsed": elapsed}
    if details is not None:
        result["ncm"] = details
    yield done(**result)
