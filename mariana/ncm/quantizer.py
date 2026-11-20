from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import ZeroConv, resize


class ResidualQuantizer(nn.Module):
    def __init__(self, dim: int, levels: tuple[int, ...], codebook_size: int,
                 commitment: float, decay: float, dead_threshold: float,
                 all_levels: bool = True):
        super().__init__()
        self.dim = dim
        self.levels = tuple(levels)
        self.codebook_size = codebook_size
        self.commitment = commitment
        self.decay = decay
        self.dead_threshold = dead_threshold
        self.all_levels = all_levels
        self.embeddings = nn.ModuleList([
            nn.Embedding(codebook_size, dim) for _ in levels
        ])
        for emb in self.embeddings:
            nn.init.normal_(emb.weight)
            emb.weight.requires_grad_(False)
        self.down_refiners = nn.ModuleList([ZeroConv(dim) for _ in levels])
        self.up_refiners = nn.ModuleList([ZeroConv(dim) for _ in levels])
        for i, emb in enumerate(self.embeddings):
            self.register_buffer(f"cluster_{i}", torch.ones(codebook_size))
            self.register_buffer(f"ema_sum_{i}", emb.weight.detach().clone())
        self.register_buffer("updates", torch.zeros((), dtype=torch.long))

    def _distances(self, x, codebook):
        return (
            x.square().sum(-1, keepdim=True)
            + codebook.square().sum(-1).unsqueeze(0)
            - 2.0 * x @ codebook.T
        )

    @torch.no_grad()
    def _update(self, level: int, vectors, indices, distances, recover: bool):
        emb = self.embeddings[level]
        cluster = getattr(self, f"cluster_{level}")
        ema_sum = getattr(self, f"ema_sum_{level}")
        counts = torch.bincount(indices, minlength=self.codebook_size).float()
        sums = torch.zeros_like(ema_sum).index_add_(0, indices, vectors)
        cluster.mul_(self.decay).add_(counts, alpha=1.0 - self.decay)
        ema_sum.mul_(self.decay).add_(sums, alpha=1.0 - self.decay)
        n = cluster.sum()
        smooth = (cluster + 1e-5) / (n + self.codebook_size * 1e-5) * n
        emb.weight.copy_(ema_sum / smooth.clamp_min(1e-8).unsqueeze(1))

        if not recover:
            return
        dead = cluster < self.dead_threshold
        ranks = dead.long().cumsum(0) - 1
        errors = distances.gather(1, indices[:, None]).squeeze(1)
        order = errors.argsort(descending=True)
        sources = order[ranks.clamp_min(0).remainder(len(order))]
        chosen = vectors[sources]
        repeated = (ranks >= len(order)).unsqueeze(1)
        chosen = chosen + repeated * 0.01 * torch.randn_like(chosen)
        emb.weight.copy_(torch.where(dead[:, None], chosen, emb.weight))
        ema_sum.copy_(torch.where(
            dead[:, None], chosen * self.dead_threshold, ema_sum,
        ))
        cluster.copy_(torch.where(dead, cluster.new_full((), self.dead_threshold), cluster))

    def _nearest(self, residual, level: int):
        down = self.down_refiners[level](resize(residual, self.levels[level]))
        flat = down.reshape(-1, self.dim)
        distances = self._distances(flat, self.embeddings[level].weight)
        return down, flat, distances, distances.argmin(-1)

    def _quantize(self, z, update: bool):
        z = z.float()
        B, L, D = z.shape
        residual = z.detach()
        cumulative = z.new_zeros(B, L, D)
        codes, partials, usages, perplexities = [], [], [], []
        recover = update and self.dead_threshold > 0 and bool(self.updates >= 10)
        for level, length in enumerate(self.levels):
            down, flat, distances, indices = self._nearest(residual, level)
            if update:
                self._update(level, flat.detach(), indices, distances.detach(), recover)
            codes.append(indices.view(B, length))
            quantized = self.embeddings[level](codes[-1])
            if level == 0:
                quantized = down + (quantized - down).detach()
            contribution = self._contribution(codes[-1], level, quantized)
            cumulative = cumulative + contribution
            partials.append(cumulative)
            residual = z.detach() - cumulative.detach()
            counts = torch.bincount(indices, minlength=self.codebook_size).float()
            usages.append((counts > 0).float().mean())
            probs = counts / counts.sum().clamp_min(1.0)
            perplexities.append(torch.exp(-(probs * probs.clamp_min(1e-9).log()).sum()))
        if update:
            self.updates.add_(1)
        return codes, partials, usages, perplexities

    def _contribution(self, codes, level: int, quantized=None):
        q = self.embeddings[level](codes) if quantized is None else quantized
        up = resize(q, self.levels[-1], "linear")
        return self.up_refiners[level](up)

    def add_level(self, cumulative, codes, level: int):
        return cumulative + self._contribution(codes, level)

    def forward(self, z, nested_dropout: float = 0.0):
        codes, partials, usages, perplexities = self._quantize(z, self.training)
        B = z.shape[0]
        K = len(self.levels)
        stacked = torch.stack(partials, dim=1)
        full = partials[-1]
        truncated = self.training and nested_dropout > 0
        if truncated:
            drop = torch.rand(B, device=z.device) < nested_dropout
            cutoffs = torch.randint(K, (B,), device=z.device)
            cutoffs = torch.where(drop, cutoffs, torch.full_like(cutoffs, K - 1))
            selected = stacked.gather(
                1, cutoffs.view(B, 1, 1, 1).expand(-1, 1, *z.shape[1:])
            ).squeeze(1)
        else:
            selected = full
        if self.all_levels:
            target = z.unsqueeze(1)
            commitment = (target - stacked.detach()).square().mean(dim=(0, 2, 3))
            quantization = (stacked - target.detach()).square().mean(dim=(0, 2, 3))
            vq_loss = (self.commitment * commitment + quantization).mean()
        else:
            if truncated:
                commitment = 0.5 * (
                    F.mse_loss(z, selected.detach()) + F.mse_loss(z, full.detach())
                )
                quantization = 0.5 * (
                    F.mse_loss(z.detach(), selected) + F.mse_loss(z.detach(), full)
                )
            else:
                commitment = F.mse_loss(z, full.detach())
                quantization = F.mse_loss(z.detach(), full)
            vq_loss = self.commitment * commitment + quantization
        selected = z + (selected - z).detach()
        full = z + (full - z).detach()
        metrics = {
            "usage": torch.stack(usages).mean(),
            "perplexity": torch.stack(perplexities).mean() / self.codebook_size,
        }
        return selected, full, codes, vq_loss, metrics

    @torch.no_grad()
    def encode(self, z):
        z = z.float()
        residual = z.detach()
        cumulative = z.new_zeros(z.shape)
        codes = []
        for level, length in enumerate(self.levels):
            down, _, _, indices = self._nearest(residual, level)
            code = indices.view(z.shape[0], length)
            codes.append(code)
            if level + 1 < len(self.levels):
                quantized = self.embeddings[level](code)
                if level == 0:
                    quantized = down + (quantized - down).detach()
                cumulative = cumulative + self._contribution(code, level, quantized)
                residual = z.detach() - cumulative.detach()
        return codes

    def reconstruct_prefixes(self, codes):
        if not 0 < len(codes) <= len(self.levels):
            raise ValueError("expected a non-empty prefix of the code hierarchy")
        values = torch.stack([
            resize(self.embeddings[level](code), self.levels[-1], "linear")
            for level, code in enumerate(codes)
        ], dim=1)
        batch, count, length, dim = values.shape
        refiners = self.up_refiners[:count]
        weight = torch.cat([refiner.conv.weight for refiner in refiners], dim=0)
        bias = torch.cat([refiner.conv.bias for refiner in refiners], dim=0)
        correction = F.conv1d(
            values.permute(0, 1, 3, 2).reshape(batch, count * dim, length),
            weight, bias, padding=1, groups=count,
        )
        correction = correction.view(batch, count, dim, length).permute(0, 1, 3, 2)
        return (values + correction).cumsum(dim=1)

    def reconstruct(self, codes, upto: int | None = None):
        if not 0 < len(codes) <= len(self.levels):
            raise ValueError("expected a non-empty prefix of the code hierarchy")
        upto = len(codes) if upto is None else upto
        if not 0 <= upto <= len(codes):
            raise ValueError("upto must be within the available code prefix")
        if upto == 0:
            return codes[0].new_zeros(
                codes[0].shape[0], self.levels[-1], self.dim, dtype=torch.float
            )
        return self.reconstruct_prefixes(codes[:upto])[:, -1]
