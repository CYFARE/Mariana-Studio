"""Parameter-freezing helpers shared by full training and fine-tuning."""

from __future__ import annotations


def _freeze(module) -> None:
    if module is None:
        return
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def freeze_mariana(model, freeze_embeddings: bool = False, freeze_layers: int = 0) -> int:
    """Freeze the leading blocks in execution order (pre -> core -> post)."""
    if freeze_embeddings:
        for name in ("tok_emb", "time_emb", "region_emb", "time_mlp"):
            _freeze(getattr(model, name, None))
    blocks = list(model.pre) + list(model.core) + list(model.post)
    for block in blocks[:max(0, int(freeze_layers))]:
        _freeze(block)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def freeze_ncm_generator(model, freeze_embeddings: bool = False,
                         freeze_layers: int = 0) -> int:
    """Freeze the leading generator blocks; the tokenizer is always frozen."""
    if freeze_embeddings:
        for name in ("prompt_tokens", "level_embedding"):
            _freeze(getattr(model, name, None))
    for block in list(model.blocks)[:max(0, int(freeze_layers))]:
        _freeze(block)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
