from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import NCMConfig
from .layers import Block, RMSNorm
from .quantizer import ResidualQuantizer


def _padding_mask(valid):
    return None if bool(valid.all()) else valid[:, None, None, :]


def masked_ce(logits, targets, mask=None, ignore_index=257):
    losses = F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1),
        ignore_index=ignore_index, reduction="none",
    ).view_as(targets)
    if mask is None:
        mask = targets != ignore_index
    return (losses * mask).sum() / mask.sum().clamp_min(1)


class ByteRefiner(nn.Module):
    def __init__(self, cfg: NCMConfig):
        super().__init__()
        self.cfg = cfg
        self.tokens = nn.Embedding(cfg.vocab_size, cfg.latent_dim)
        self.latent = nn.Linear(cfg.latent_dim, cfg.latent_dim, bias=False)
        self.blocks = nn.ModuleList([
            Block(cfg.latent_dim, cfg.n_heads, cfg.mlp_mult, cfg.dropout, cfg.rope_base)
            for _ in range(cfg.refiner_layers)
        ])
        self.norm = RMSNorm(cfg.latent_dim)
        self.head = nn.Linear(cfg.latent_dim, cfg.vocab_size, bias=False)
        self.head.weight = self.tokens.weight

    def forward(self, tokens, latent, valid):
        x = self.tokens(tokens) + self.latent(latent)
        positions = torch.arange(tokens.shape[1], device=tokens.device).expand_as(tokens)
        mask = _padding_mask(valid)
        for block in self.blocks:
            x = block(x, positions, mask)
        return self.head(self.norm(x))


class MultiscaleTokenizer(nn.Module):
    def __init__(self, cfg: NCMConfig):
        super().__init__()
        self.cfg = cfg
        self.tokens = nn.Embedding(cfg.vocab_size, cfg.latent_dim)
        self.encoder = nn.ModuleList([
            Block(cfg.latent_dim, cfg.n_heads, cfg.mlp_mult, cfg.dropout, cfg.rope_base)
            for _ in range(cfg.encoder_layers)
        ])
        self.decoder = nn.ModuleList([
            Block(cfg.latent_dim, cfg.n_heads, cfg.mlp_mult, cfg.dropout, cfg.rope_base)
            for _ in range(cfg.decoder_layers)
        ])
        self.enc_norm = RMSNorm(cfg.latent_dim)
        self.dec_norm = RMSNorm(cfg.latent_dim)
        self.quantizer = ResidualQuantizer(
            cfg.latent_dim, cfg.levels, cfg.codebook_size,
            cfg.commitment, cfg.ema_decay, cfg.dead_threshold, cfg.all_level_vq,
        )
        self.output = nn.Linear(cfg.latent_dim, cfg.vocab_size, bias=False)
        self.output.weight = self.tokens.weight
        self.refiner = ByteRefiner(cfg) if cfg.refiner_layers > 0 else None

    def _positions(self, tokens):
        return torch.arange(tokens.shape[1], device=tokens.device).expand_as(tokens)

    def encode_z(self, tokens):
        if tokens.shape[1] != self.cfg.ctx_len:
            raise ValueError(f"expected sequences of length {self.cfg.ctx_len}")
        valid = tokens != self.cfg.pad_token
        x = self.tokens(tokens)
        positions = self._positions(tokens)
        mask = _padding_mask(valid)
        for block in self.encoder:
            x = block(x, positions, mask)
        return F.rms_norm(self.enc_norm(x), (self.cfg.latent_dim,)), valid

    def decode_latent(self, latent, valid):
        positions = torch.arange(latent.shape[1], device=latent.device).expand(latent.shape[:2])
        mask = _padding_mask(valid)
        for block in self.decoder:
            latent = block(latent, positions, mask)
        return self.output(self.dec_norm(latent))

    def _greedy_bytes(self, logits, valid):
        tokens = logits[..., :256].argmax(-1)
        return torch.where(valid, tokens, self.cfg.pad_token)

    def forward(self, tokens):
        z, valid = self.encode_z(tokens)
        selected, full, codes, vq, qmetrics = self.quantizer(
            z, self.cfg.nested_dropout
        )
        nested_recon = (
            self.training and self.cfg.full_recon_weight < 1.0
            and self.cfg.nested_dropout > 0
        )
        if nested_recon:
            batch = tokens.shape[0]
            paired_logits = self.decode_latent(
                torch.cat((full, selected), dim=0),
                torch.cat((valid, valid), dim=0),
            )
            full_logits, selected_logits = paired_logits.split(batch, dim=0)
        else:
            full_logits = self.decode_latent(full, valid)
        full_loss = masked_ce(full_logits, tokens, valid, self.cfg.pad_token)
        if nested_recon:
            selected_loss = masked_ce(selected_logits, tokens, valid, self.cfg.pad_token)
            recon = (
                self.cfg.full_recon_weight * full_loss
                + (1.0 - self.cfg.full_recon_weight) * selected_loss
            )
        else:
            recon = full_loss

        with torch.no_grad():
            initial = self._greedy_bytes(full_logits, valid)
            repair_mask = valid & (initial != tokens)
            if self.training:
                repair_mask |= valid & (torch.rand_like(tokens.float()) < 0.1)
            corrupted = initial.masked_fill(repair_mask, self.cfg.mask_token)
        if self.refiner is not None and bool(repair_mask.any()):
            refined_logits = self.refiner(corrupted, full, valid)
            refine = masked_ce(refined_logits, tokens, repair_mask, self.cfg.pad_token)
            with torch.no_grad():
                refined = self._greedy_bytes(refined_logits, valid)
                refine_acc = (refined[repair_mask] == tokens[repair_mask]).float().mean()
        else:
            refine = full_loss.new_zeros(())
            refine_acc = full_loss.new_zeros(())
        with torch.no_grad():
            if valid.any():
                recon_acc = (initial[valid] == tokens[valid]).float().mean()
            else:
                recon_acc = full_loss.new_zeros(())
        total = recon + vq + self.cfg.refiner_weight * refine
        return {
            "total": total, "recon": recon, "vq": vq, "refine": refine,
            "acc": recon_acc, "repair_acc": refine_acc, **qmetrics,
        }

    @torch.no_grad()
    def encode(self, tokens):
        z, _ = self.encode_z(tokens)
        return self.quantizer.encode(z)

    @torch.no_grad()
    def decode(self, codes, upto: int | None = None, valid=None,
               refine_steps: int = 0, refine_threshold: float = 0.8):
        latent = self.quantizer.reconstruct(codes, upto)
        if valid is None:
            valid = torch.ones(
                codes[0].shape[0], self.cfg.ctx_len,
                dtype=torch.bool, device=latent.device,
            )
        logits = self.decode_latent(latent, valid)
        tokens = self._greedy_bytes(logits, valid)
        confidence = logits[..., :256].float().softmax(-1).max(-1).values
        repairs = 0
        for _ in range(refine_steps):
            if self.refiner is None:
                break
            low = valid & (confidence < refine_threshold)
            if not low.any():
                break
            corrupted = tokens.masked_fill(low, self.cfg.mask_token)
            refined_logits = self.refiner(corrupted, latent, valid)
            refined = self._greedy_bytes(refined_logits, valid)
            refined_conf = refined_logits[..., :256].float().softmax(-1).max(-1).values
            accepted = low & (refined_conf >= confidence)
            tokens = torch.where(accepted, refined, tokens)
            confidence = torch.where(accepted, refined_conf, confidence)
            repairs += 1
            if not bool(accepted.any()):
                break
        return tokens, {"confidence": confidence, "refiner_nfe": repairs}
