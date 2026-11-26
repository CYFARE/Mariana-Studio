from __future__ import annotations

import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import NCMConfig
from .layers import HAS_FLEX, Block, RMSNorm, ZeroConv, resize, sample_logits
from .tokenizer import MultiscaleTokenizer, masked_ce


class CodebookHead(nn.Module):
    def __init__(self, dim: int, latent_dim: int, codebook_size: int, mode: str = "distance"):
        super().__init__()
        self.mode = mode
        self.proj = nn.Linear(dim, latent_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(codebook_size))
        initial_scale = 0.0 if mode == "distance" else math.log(16.0)
        self.log_scale = nn.Parameter(torch.tensor(initial_scale))

    def forward(self, x, codebook, entries_sq=None, scale=None):
        projected = self.proj(x).float()
        entries = codebook.float()
        if self.mode == "distance":
            if entries_sq is None:
                entries_sq = entries.square().sum(-1)
            if scale is None:
                scale = entries_sq.mean().clamp_min(1e-12)
            logits = (2.0 * projected @ entries.T - entries_sq) / scale
            return logits * self.log_scale.exp() + self.bias
        projected = F.normalize(projected, dim=-1, eps=1e-6)
        entries = F.normalize(entries, dim=-1, eps=1e-6)
        scale = self.log_scale.clamp(-2.3, 4.6).exp()
        return (projected @ entries.T) * scale + self.bias


class NCMGenerator(nn.Module):
    def __init__(self, tokenizer: MultiscaleTokenizer, cfg: NCMConfig):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.tokenizer.eval()
        for parameter in self.tokenizer.parameters():
            parameter.requires_grad_(False)
        self.prompt_tokens = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.latent_proj = nn.Linear(cfg.latent_dim, cfg.dim, bias=False)
        self.level_embedding = nn.Embedding(cfg.n_levels, cfg.dim)
        self.queries = nn.Parameter(torch.empty(1, cfg.levels[0], cfg.latent_dim))
        nn.init.normal_(self.queries, std=0.02)
        self.input_refiners = nn.ModuleList([
            ZeroConv(cfg.latent_dim) for _ in range(cfg.n_levels)
        ])
        self.blocks = nn.ModuleList([
            Block(
                cfg.dim, cfg.n_heads, cfg.mlp_mult, cfg.dropout, cfg.rope_base,
                residual_layers=cfg.generator_layers,
            )
            for _ in range(cfg.generator_layers)
        ])
        self.norm = RMSNorm(cfg.dim)
        self.heads = nn.ModuleList([
            CodebookHead(cfg.dim, cfg.latent_dim, cfg.codebook_size, cfg.codebook_head)
            for _ in range(cfg.n_levels)
        ])
        self.register_buffer(
            "codebook_entries_sq",
            torch.empty(cfg.n_levels, cfg.codebook_size), persistent=False,
        )
        self.register_buffer("codebook_scales", torch.empty(cfg.n_levels), persistent=False)
        self._refresh_codebook_stats()
        chunks, target_positions = self._target_layout(cfg.prefix_len)
        sequence_positions = torch.cat((
            torch.arange(cfg.prefix_len), *target_positions,
        ))
        self.register_buffer("sequence_positions", sequence_positions, persistent=False)
        total_length = cfg.prefix_len + sum(cfg.levels)
        attention_mask = torch.zeros(1, 1, total_length, total_length, dtype=torch.bool)
        attention_mask[:, :, :cfg.prefix_len, :cfg.prefix_len] = True
        for start, end in chunks:
            attention_mask[:, :, start:end, :cfg.prefix_len] = True
            attention_mask[:, :, start:end, start:end] = True
        self.register_buffer("attention_mask", attention_mask, persistent=False)
        self.register_buffer(
            "level_ids",
            torch.repeat_interleave(torch.arange(cfg.n_levels), torch.tensor(cfg.levels)),
            persistent=False,
        )
        self.use_flex = False
        self.flex_block_mask = None

    @torch.no_grad()
    def _refresh_codebook_stats(self):
        entries_sq = torch.stack([
            embedding.weight.float().square().sum(-1)
            for embedding in self.tokenizer.quantizer.embeddings
        ])
        self.codebook_entries_sq.copy_(entries_sq)
        self.codebook_scales.copy_(entries_sq.mean(dim=1).clamp_min(1e-12))

    def load_state_dict(self, *args, **kwargs):
        result = super().load_state_dict(*args, **kwargs)
        self._refresh_codebook_stats()
        return result

    def enable_flex(self, device, block_size: int = 128):
        if not HAS_FLEX:
            raise RuntimeError("FlexAttention requires a PyTorch build with flex_attention support")
        head_dim = self.cfg.dim // self.cfg.n_heads
        if head_dim < 16:
            raise ValueError(
                f"FlexAttention requires attention head dimension >= 16; got {head_dim}. "
                "Reduce --heads or omit --flex"
            )
        from torch.nn.attention.flex_attention import create_block_mask

        device = torch.device(device)
        length = self.attention_mask.shape[-1]
        levels = torch.cat((
            torch.full((self.cfg.prefix_len,), -1, dtype=torch.long),
            self.level_ids.cpu(),
        )).to(device)

        def mask_mod(b, h, q, kv):
            q = q.clamp(max=length - 1)
            kv = kv.clamp(max=length - 1)
            q_prompt = q < self.cfg.prefix_len
            k_prompt = kv < self.cfg.prefix_len
            same_level = (levels[q] == levels[kv]) & (levels[kv] >= 0)
            return torch.where(q_prompt, k_prompt, k_prompt | same_level)

        self.flex_block_mask = create_block_mask(
            mask_mod, B=None, H=None, Q_LEN=length, KV_LEN=length,
            device=device, BLOCK_SIZE=block_size,
        )
        self.use_flex = True

    def train(self, mode=True):
        super().train(mode)
        self.tokenizer.eval()
        return self

    def null_prompt(self, prompt):
        null = torch.full_like(prompt, self.cfg.pad_token)
        null[:, 0] = self.cfg.bos_token
        return null

    def _prompt_data(self, prompt):
        if prompt.ndim != 2 or prompt.shape[1] != self.cfg.prefix_len:
            raise ValueError(f"prompt must have shape (batch, {self.cfg.prefix_len})")
        valid = prompt != self.cfg.pad_token
        if not valid.any(dim=1).all():
            raise ValueError("each prompt needs at least one non-padding token")
        positions = torch.arange(self.cfg.prefix_len, device=prompt.device).expand_as(prompt)
        return self.prompt_tokens(prompt), positions, valid

    def _level_input(self, cumulative, level: int, batch_size: int):
        if level == 0:
            x = self.queries.expand(batch_size, -1, -1)
        else:
            x = resize(cumulative, self.cfg.levels[level])
        x = self.input_refiners[level](x)
        return self.latent_proj(x) + self.level_embedding.weight[level]

    def _perturb_prefix(self, cumulative):
        """Make training prefixes resemble the generator's own free-running errors."""
        if not self.training:
            return cumulative
        if self.cfg.prefix_dropout > 0:
            drop = torch.rand(
                cumulative.shape[0], 1, 1, device=cumulative.device,
            ) < self.cfg.prefix_dropout
            cumulative = cumulative.masked_fill(drop, 0.0)
        if self.cfg.prefix_noise > 0:
            cumulative = cumulative + torch.randn_like(cumulative) * self.cfg.prefix_noise
        return cumulative

    def _target_layout(self, prompt_length):
        chunks, positions = [], []
        for level, length in enumerate(self.cfg.levels):
            chunks.append((prompt_length, prompt_length + length))
            positions.append(torch.arange(length, device=self.queries.device))
            prompt_length += length
        return chunks, positions

    def forward(self, prompt, codes, has_padding=None, perturb=True):
        if len(codes) != self.cfg.n_levels:
            raise ValueError("expected one target code tensor per level")
        prompt_x, _, prompt_valid = self._prompt_data(prompt)
        B = prompt.shape[0]
        with torch.no_grad():
            if len(codes) > 1:
                partials = self.tokenizer.quantizer.reconstruct_prefixes(codes[:-1])
            else:
                partials = None
        inputs = []
        for level in range(self.cfg.n_levels):
            if level == 0:
                cumulative = torch.zeros(
                    B, self.cfg.ctx_len, self.cfg.latent_dim,
                    device=prompt.device, dtype=self.queries.dtype,
                )
            else:
                cumulative = partials[:, level - 1]
            if perturb:
                cumulative = self._perturb_prefix(cumulative)
            inputs.append(self._level_input(cumulative, level, B))
        x = torch.cat((prompt_x, *inputs), dim=1)
        P = prompt.shape[1]
        chunks, _ = self._target_layout(P)
        positions = self.sequence_positions.expand(B, -1)
        if has_padding is None:
            has_padding = not bool(prompt_valid.all())
        if has_padding:
            key_valid = F.pad(prompt_valid, (0, x.shape[1] - P), value=True)
            mask = self.attention_mask & key_valid[:, None, None, :]
        else:
            mask = self.attention_mask
        score_mod = None
        if self.use_flex and has_padding:
            def mask_padded_prompt(score, batch, head, q, kv):
                prompt_key = kv.clamp(max=P - 1)
                valid = (kv >= P) | prompt_valid[batch, prompt_key]
                return torch.where(valid, score, float("-inf"))

            score_mod = mask_padded_prompt
        for block in self.blocks:
            x = block(
                x, positions, mask,
                block_mask=self.flex_block_mask if self.use_flex else None,
                score_mod=score_mod,
            )
        x = self.norm(x)
        return [
            self.heads[level](
                x[:, start:end], self.tokenizer.quantizer.embeddings[level].weight,
                self.codebook_entries_sq[level], self.codebook_scales[level],
            )
            for level, (start, end) in enumerate(chunks)
        ]

    def corrupt_codes(self, codes):
        flat = torch.cat(codes, dim=1)
        B = flat.shape[0]
        fractions = 1.0 - 0.5 * self.level_ids.float() / max(self.cfg.n_levels - 1, 1)
        rate = 2.0 * self.cfg.corruption * fractions * torch.rand(B, 1, device=flat.device)
        replace = torch.rand_like(flat.float()) < rate
        random_codes = torch.randint_like(flat, self.cfg.codebook_size)
        corrupted = torch.where(replace, random_codes, flat)
        return list(corrupted.split(self.cfg.levels, dim=1))

    def _latent_from_logits(self, logits):
        """Differentiable hard-code latent: forward uses argmax, gradients use the softmax."""
        cumulative = None
        for level, score in enumerate(logits):
            probs = score.float().softmax(-1)
            hard = F.one_hot(
                probs.argmax(-1), self.cfg.codebook_size,
            ).to(probs.dtype)
            soft = hard + probs - probs.detach()
            quantized = soft @ self.tokenizer.quantizer.embeddings[level].weight
            contribution = self.tokenizer.quantizer._contribution(
                None, level, quantized=quantized,
            )
            cumulative = contribution if cumulative is None else cumulative + contribution
        return cumulative

    def _byte_loss(self, logits, target, valid):
        latent = self._latent_from_logits(logits)
        byte_logits = self.tokenizer.decode_latent(latent, valid)
        return masked_ce(byte_logits, target, valid, self.cfg.pad_token)

    def _free_byte_loss(self, prompt, target, valid):
        """Byte loss for the codes the model chooses under its own free-running prefix."""
        with torch.no_grad():
            cache = self.encode_prompt(prompt)
            cumulative = torch.zeros(
                prompt.shape[0], self.cfg.ctx_len, self.cfg.latent_dim,
                device=prompt.device, dtype=self.queries.dtype,
            )
            predicted = []
            for level in range(self.cfg.n_levels):
                logits = self.forward_level(prompt, level, cumulative, cache)
                code = logits.argmax(-1)
                predicted.append(code)
                cumulative = self.tokenizer.quantizer.add_level(cumulative, code, level)
        student = self.forward(prompt, predicted, perturb=False)
        return self._byte_loss(student, target, valid)

    def loss(self, prompt, codes, target=None):
        B = prompt.shape[0]
        if (self.cfg.byte_weight > 0 or self.cfg.free_byte_weight > 0) and target is None:
            raise ValueError("byte_weight and free_byte_weight require target bytes")
        if self.training and self.cfg.prompt_dropout > 0:
            drop = torch.rand(B, device=prompt.device) < self.cfg.prompt_dropout
            prompt = torch.where(drop[:, None], self.null_prompt(prompt), prompt)
        has_padding = bool((prompt == self.cfg.pad_token).any())
        conditioning = self.corrupt_codes(codes) if self.training else [c.clone() for c in codes]
        self_step = self.training and self.cfg.self_condition > 0 and bool(torch.rand(()) < self.cfg.self_condition)
        consistency = prompt.new_zeros((), dtype=self.queries.dtype)
        if self_step:
            with torch.no_grad():
                teacher = self.forward(prompt, codes, has_padding=has_padding)
                teacher_flat = torch.cat(teacher, dim=1)
                predicted = teacher_flat.argmax(-1)
            corrupted = torch.cat(conditioning, dim=1)
            use_predicted = torch.rand(
                B, self.cfg.n_levels, device=prompt.device,
            ) < 0.5
            use_predicted = use_predicted[:, self.level_ids]
            conditioning = list(torch.where(
                use_predicted, predicted, corrupted,
            ).split(self.cfg.levels, dim=1))
            logits = self.forward(prompt, conditioning, has_padding=has_padding)
            student_log = F.log_softmax(torch.cat(logits, dim=1).float(), dim=-1)
            teacher_log = F.log_softmax(teacher_flat.float(), dim=-1)
            point_kl = (teacher_log.exp() * (teacher_log - student_log)).sum(-1)
            consistency = torch.stack([
                level.mean() for level in point_kl.split(self.cfg.levels, dim=1)
            ]).mean()
        else:
            logits = self.forward(prompt, conditioning, has_padding=has_padding)

        flat_logits = torch.cat(logits, dim=1)
        flat_codes = torch.cat(codes, dim=1)
        position_losses = F.cross_entropy(
            flat_logits.float().reshape(-1, flat_logits.shape[-1]),
            flat_codes.reshape(-1), reduction="none",
        ).view(B, -1)
        level_losses = torch.stack([
            level.mean() for level in position_losses.split(self.cfg.levels, dim=1)
        ])
        lengths = torch.tensor(
            self.cfg.levels, device=prompt.device, dtype=level_losses.dtype,
        )
        weights = lengths ** (1.0 - self.cfg.level_alpha)
        code_loss = (weights / weights.sum() * level_losses).sum()
        total = code_loss + self.cfg.consistency_weight * consistency
        with torch.no_grad():
            correct = (flat_logits.argmax(-1) == flat_codes).float().mean()
        result = {
            "total": total, "code": code_loss, "consistency": consistency,
            "acc": correct, "self": total.new_tensor(float(self_step)),
        }
        if target is not None and (self.cfg.byte_weight > 0 or self.cfg.free_byte_weight > 0):
            target = target.long()
            valid = target != self.cfg.pad_token
            if self.cfg.byte_weight > 0:
                byte_loss = self._byte_loss(logits, target, valid)
                total = total + self.cfg.byte_weight * byte_loss
                result["byte"] = byte_loss.detach()
            if self.cfg.free_byte_weight > 0:
                free_byte = self._free_byte_loss(prompt, target, valid)
                total = total + self.cfg.free_byte_weight * free_byte
                result["free_byte"] = free_byte.detach()
            result["total"] = total
        result.update({f"l{i}": value for i, value in enumerate(level_losses)})
        return result

    def encode_prompt(self, prompt):
        x, positions, valid = self._prompt_data(prompt)
        mask = None if bool(valid.all()) else valid[:, None, None, :]
        cache = []
        for block in self.blocks:
            x, kv = block(x, positions, mask, return_kv=True)
            cache.append(kv)
        return cache, mask

    def forward_level(self, prompt, level: int, cumulative, prefix_cache=None):
        B = prompt.shape[0]
        x = self._level_input(cumulative, level, B)
        length = x.shape[1]
        positions = torch.arange(length, device=x.device).expand(B, length)
        if prefix_cache is not None:
            prompt_mask = prefix_cache[1]
            cache = prefix_cache[0]
            if prompt_mask is None:
                mask = None
            else:
                mask = torch.ones(
                    B, 1, length, prompt.shape[1] + length,
                    dtype=torch.bool, device=x.device,
                )
                mask[:, :, :, :prompt.shape[1]] = prompt_mask
            for i, block in enumerate(self.blocks):
                x = block(x, positions, mask, past_kv=cache[i])
        else:
            prompt_x, prompt_pos, prompt_valid = self._prompt_data(prompt)
            mask = torch.ones(
                B, 1, prompt.shape[1] + length, prompt.shape[1] + length,
                dtype=torch.bool, device=x.device,
            )
            mask[:, :, :prompt.shape[1], prompt.shape[1]:] = False
            mask[:, :, :, :prompt.shape[1]] &= prompt_valid[:, None, None, :]
            positions = torch.cat((prompt_pos, positions), dim=1)
            x = torch.cat((prompt_x, x), dim=1)
            for block in self.blocks:
                x = block(x, positions, mask)
            x = x[:, prompt.shape[1]:]
        return self.heads[level](
            self.norm(x), self.tokenizer.quantizer.embeddings[level].weight,
            self.codebook_entries_sq[level], self.codebook_scales[level],
        )

    @torch.no_grad()
    def generate(self, prompt, temperature=1.0, top_k=0, top_p=1.0,
                 guidance=0.0, use_cache=True, early_exit=0.0,
                 refine_steps=0, refine_threshold=0.8, guidance_ramp=True):
        if not math.isfinite(guidance) or guidance < 0:
            raise ValueError("guidance must be finite and non-negative")
        if not math.isfinite(early_exit) or not 0.0 <= early_exit <= 1.0:
            raise ValueError("early_exit must be in [0, 1]")
        if refine_steps < 0:
            raise ValueError("refine_steps must be non-negative")
        if not math.isfinite(refine_threshold) or refine_threshold < 0.0:
            raise ValueError("refine_threshold must be finite and non-negative")
        started = time.time()
        original_batch = prompt.shape[0]
        if guidance > 0:
            prompt = torch.cat((prompt, self.null_prompt(prompt)), dim=0)
        B = prompt.shape[0]
        cache = self.encode_prompt(prompt) if use_cache else None
        cumulative = torch.zeros(
            B, self.cfg.ctx_len, self.cfg.latent_dim,
            device=prompt.device, dtype=self.queries.dtype,
        )
        generated = []
        levels_used = self.cfg.n_levels
        decoder_nfe = 0
        for level in range(self.cfg.n_levels):
            logits = self.forward_level(prompt, level, cumulative, cache)
            if guidance > 0:
                scale = guidance * level / max(self.cfg.n_levels - 1, 1) if guidance_ramp else guidance
                logits = (1.0 + scale) * logits[:original_batch] - scale * logits[original_batch:]
            codes = sample_logits(logits, temperature, top_k, top_p)
            if guidance > 0:
                codes = torch.cat((codes, codes), dim=0)
            generated.append(codes)
            cumulative = self.tokenizer.quantizer.add_level(cumulative, codes, level)
            enough_levels = level + 1 >= max(2, self.cfg.n_levels // 2)
            if early_exit > 0 and enough_levels and level + 1 < self.cfg.n_levels:
                valid = torch.ones(B, self.cfg.ctx_len, dtype=torch.bool, device=prompt.device)
                partial_logits = self.tokenizer.decode_latent(cumulative, valid)
                confidence = partial_logits[..., :256].float().softmax(-1).max(-1).values
                confidence = confidence.mean(-1).min()
                decoder_nfe += 1
                if confidence >= early_exit:
                    levels_used = level + 1
                    break
        codes = [code[:original_batch] for code in generated]
        tokens, decode_stats = self.tokenizer.decode(
            codes, refine_steps=refine_steps, refine_threshold=refine_threshold,
        )
        branches = 2 if guidance > 0 else 1
        level_nfe = levels_used * branches
        prompt_nfe = branches if use_cache else 0
        return tokens, {
            "codes": codes,
            "levels_used": levels_used,
            "level_nfe": level_nfe,
            "prompt_nfe": prompt_nfe,
            "generator_nfe": level_nfe + prompt_nfe,
            "decoder_nfe": decoder_nfe + 1,
            "confidence": decode_stats["confidence"],
            "refiner_nfe": decode_stats["refiner_nfe"],
            "elapsed": time.time() - started,
        }
