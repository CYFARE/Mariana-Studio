# Sprint-DRV: single-pass training for a hybrid diffusion–autoregressive language model

**Whitepaper · Mariana v2.3.1 · 2026-11-20**

| | |
| --- | --- |
| Status | Implemented and measured on one local GPU |
| Implementation | `mariana/train.py`, `mariana/model.py`, `mariana/config.py` |
| Runbook | [fast-training.md](../fast-training.md) |
| Related | [Architecture spec](../../architecture.txt), [SAINT paper](saint.md), [NCM paper](ncm.md) |

## Abstract

Mariana is a byte-level language model trained with two objectives at once: a
causal autoregressive (AR) next-byte objective on a clean prefix, and a
block-wise masked-diffusion objective on a corrupted draft region. The original
trainer computed those objectives with two backbone passes over the same batch:
one clean causal pass for AR supervision and one hybrid pass for diffusion,
verifier, and alignment supervision. This paper describes **Sprint-DRV**, a
training architecture that computes every objective from a single hybrid
forward.

The key observation is structural. For a cutoff `k`, positions `i < k` have
clean inputs, diffusion time bucket zero, region zero, and a strictly causal
attention mask. The draft-only bidirectional attention that begins at `k`
cannot change any earlier row. Consequently the prefix hidden states and logits
of the hybrid forward are exactly the clean causal AR forward at the same loop
count. Sprint-DRV therefore supervises AR on the clean prefix, diffusion and the
verifier on the draft, and multi-token prediction on causal prefix states, all
from one pass. Random cutoffs move the boundary so every byte is eventually
covered; a fully clean cutoff supplies final-byte AR coverage.

On an 8 GB consumer laptop GPU with a 9M-parameter dense model, batch 16, and
context 256, eager Sprint-DRV measured 134,835 byte-tokens/s versus 30,024 for
the original trainer (**4.49×**) and reduced peak CUDA memory from 1,645 MiB to
819 MiB. With `torch.compile`, a long run measured 167,132 byte-tokens/s
(**5.57×**) at 641 MiB, about **61% less peak memory** than the original
trainer. These are throughput measurements, not a claim of identical sample
efficiency or final quality.

## 1. Introduction

A diffusion–autoregressive hybrid has a useful training property: the same
weights can learn a causal next-token distribution and a block-wise denoising
distribution. It also has an obvious inefficiency. If the two objectives are
computed by separate backbone forwards over overlapping tokens, most of the
projections, attention, SwiGLU work, and activations are paid twice.

Sprint-DRV removes the duplicate pass without changing the model, the inference
algorithm, or the export format. It is a training-architecture change, not a
new set of weights and not a distillation recipe. The contribution is a
correctness argument plus a systems implementation:

1. **Mask identity.** The clean prefix of a hybrid batch is indistinguishable
   from a clean causal batch at the same positions and loop count.
2. **Shared supervision.** One hybrid forward supplies AR, diffusion, verifier,
   and multi-token-prediction losses, plus MoE load balancing when enabled.
3. **Coverage.** Random cutoffs and the `k = L` case give AR supervision at every
   position across a run.
4. **Systems work.** Fused normalization, shared RoPE trigonometry, vectorized
   on-device corruption, fixed-shape losses, and deferred metric synchronization
   make the single pass efficient in practice.
5. **Measured tradeoffs.** The paper reports local throughput and memory, and is
   explicit about what the fast objective does not train.

The legacy two-pass objective remains available as the baseline and as a
fine-tuning mode. Both modes read and write the same checkpoint layout.

## 2. The training problem

### 2.1 Backbone and conditioning

The dense Mariana model is a pre-norm transformer with RMSNorm, SwiGLU, and
RoPE. It uses two additional conditioning signals for diffusion:

- a **time** bucket, embedded and applied through adaLN-zero modulation, and
- a **region** embedding that distinguishes committed/AR tokens from draft
  tokens.

Both are defined relative to a baseline so that time bucket zero and region zero
are an exact identity. That property is what lets the committed AR path export
to a vanilla LLaMA-architecture file. The recurrent core is weight-shared across
`max_loops`; loop dropout samples a smaller loop count per training step.

A byte vocabulary of 256 tokens plus `<|bos|>` and `<|mask|>` is used
throughout. The sequence length is `L`, the model width is `d`, and there are
`n_heads` attention heads.

### 2.2 Batch construction

For each row, a cutoff `k` splits the sequence into

- a clean prefix `P = {i : i < k}` and
- a draft region `D = {i : i ≥ k}`.

The draft is split into blocks of `block_size` bytes. Each block receives an
independent noise level `t ∈ (0, 1]`, discretized into `n_time_buckets` buckets.
For a draft position in a block with level `t`:

- the input is replaced by `[MASK]` with probability `0.1 + 0.8t`, or
- replaced by a uniform random byte with an additional 10% probability, or
- left unchanged.

The attention mask is strictly causal in the prefix. Draft rows may attend
bidirectionally inside their own block and causally across earlier blocks.
Mask annealing occasionally uses a fully causal mask to stabilize early
training. The corruption and mask construction are tensor operations on the
training device.

### 2.3 The legacy two-pass objective

Let `x` be the clean byte sequence, `x̃` the corrupted hybrid input, `r_i` the
region indicator, `t_i` the time bucket, and `f_θ` the backbone. The legacy
trainer computes

$$
\mathcal{L}_{\text{AR}} = \frac{1}{L-1}\sum_{i=0}^{L-2}
  \mathrm{CE}\big(f_\theta(x)_i,\; x_{i+1}\big),
$$

$$
\mathcal{L}_{\text{diff}} = \frac{1}{|D|}\sum_{i \in D}
  \mathrm{CE}\big(f_\theta(\tilde{x},t,r)_i,\; x_i\big),
$$

$$
\mathcal{L}_{\text{align}} = \frac{1}{|D|}\sum_{i \in D}
  \mathrm{KL}\Big(\mathrm{softmax}\big(f_\theta(\tilde{x},t,r)_i\big)
  \;\Big\|\;\mathrm{softmax}\big(f_\theta(x)_{i-1}\big)\Big),
$$

$$
\mathcal{L}_{\text{ver}} = \frac{1}{|D|}\sum_{i \in D}
  \mathrm{BCE}\big(v_i,\; \mathbb{1}[\arg\max f_\theta(\tilde{x},t,r)_i = x_i]\big),
$$

$$
\mathcal{L}_{\text{MTP}} = \frac{1}{L-2}\sum_{i=0}^{L-3}
  \mathrm{CE}\big(g(h_i),\; x_{i+2}\big),
$$

with total

$$
\mathcal{L}_{\text{legacy}} =
  w_{\text{ar}}\mathcal{L}_{\text{AR}}
  + w_{\text{diff}}\mathcal{L}_{\text{diff}}
  + w_{\text{align}}\mathcal{L}_{\text{align}}
  + w_{\text{ver}}\mathcal{L}_{\text{ver}}
  + w_{\text{mtp}}\mathcal{L}_{\text{MTP}}
  + w_{\text{moe}}\mathcal{L}_{\text{MoE-LB}}.
$$

The first two terms need different attention patterns: the clean pass is
strictly causal, while the draft pass is block-causal and carries corruption and
conditioning. The alignment term additionally needs clean AR logits in the draft
region as a detached teacher. That is why the legacy trainer performs two
backbone passes.

## 3. Sprint-DRV

### 3.1 Mask identity

Fix a cutoff `k` and a loop count. For every prefix row `i < k`:

1. `r_i = 0` and `t_i = 0`, so time and region conditioning are the baseline
   identity.
2. The corruption mask only touches `D`, so the input at `i` is the clean byte.
3. The attention mask for row `i` allows only `j ≤ i`, because the extra
   block-causal term requires `r_i = 1`.

These are exactly the inputs, conditioning, and mask of a clean causal forward
at the same positions and loop count. By induction over layers, every prefix
hidden state and logit is identical. Draft rows cannot alter earlier rows
because the prefix mask has no edge from a draft column to a prefix row.

> **Lemma (prefix identity).** At a fixed loop count, the hybrid forward and a
> clean causal forward produce identical hidden states and logits at every
> position `i < k`.

This is a statement about the computation graph, not an approximation. It is
tested directly by the training correctness suite: prefix AR losses and logits
must match a separate clean forward, and empty-draft cases must be zero-safe.

### 3.2 Shared objective

Sprint-DRV builds one hybrid batch and performs one forward. Let `z_i` be the
logits at position `i`. The same cross-entropy tensor is used for both
objectives by selecting the target per position:

$$
\tilde{y}_i =
\begin{cases}
x_{i+1}, & i \in P \quad \text{(next-byte AR target)}\\
x_i, & i \in D \quad \text{(self-byte denoising target)}
\end{cases}
$$

$$
\mathcal{L}_{\text{AR}} = \frac{1}{|P|}\sum_{i \in P}
  \mathrm{CE}(z_i, \tilde{y}_i), \qquad
\mathcal{L}_{\text{diff}} = \frac{1}{|D|}\sum_{i \in D}
  \mathrm{CE}(z_i, \tilde{y}_i).
$$

The verifier and MTP terms are computed from the same forward:

$$
\mathcal{L}_{\text{ver}} = \frac{1}{|D|}\sum_{i \in D}
  \mathrm{BCE}\big(v_i, \mathbb{1}[\arg\max z_i = x_i]\big),
$$

$$
\mathcal{L}_{\text{MTP}} = \frac{1}{|P'|}\sum_{i \in P'}
  \mathrm{CE}\big(g(h_i), x_{i+2}\big),
$$

where `P'` is the causal prefix restricted to positions that have a valid
`i+2` target. The fast objective is

$$
\mathcal{L}_{\text{fast}} =
  w_{\text{ar}}\mathcal{L}_{\text{AR}}
  + w_{\text{diff}}\mathcal{L}_{\text{diff}}
  + w_{\text{ver}}\mathcal{L}_{\text{ver}}
  + w_{\text{mtp}}\mathcal{L}_{\text{MTP}}
  + w_{\text{moe}}\mathcal{L}_{\text{MoE-LB}},
$$

with `L_align = 0`. A single fixed-shape cross-entropy and masked reductions
avoid variable-sized boolean indexing, whose output can synchronize the GPU and
defeat compilation.

### 3.3 Coverage and empty drafts

Each batch samples a cutoff. In fast mode the sampling range includes `k = L`,
which creates a fully clean causal batch and supplies AR supervision at the
final byte. Across steps, every position has a positive probability of falling
in the prefix, so AR coverage is not restricted to a fixed early region.

When `k = L`, the draft is empty. Every draft loss is guarded by
`clamp_min(1)` in its denominator and contributes zero; the verifier and
diffusion heads receive no gradient for that batch, while AR and MTP still do.
This makes the boundary case a normal training step rather than a special path.

### 3.4 What Sprint-DRV does not train

Two things are intentionally different from the legacy objective:

- **Full-sequence `L_align` is zero.** The alignment term needs clean AR
  conditionals in the draft region. Producing them would require the second
  pass Sprint-DRV removes. Fast training therefore optimizes diffusion and
  verifier losses directly, but does not explicitly pull the draft marginal
  toward the AR conditional at draft positions. A short `--legacy` fine-tune can
  restore that objective before measuring speculative acceptance.
- **AR is supervised at the step's loop count.** Loop dropout samples a depth
  per step. The prefix identity holds for that sampled depth. The legacy clean
  pass uses the canonical full depth, so fast-mode AR loss is not numerically
  identical to legacy AR loss even before the prefix weighting.

Fast validation AR is prefix-weighted. It should not be compared directly with
legacy full-sequence validation AR.

## 4. Systems implementation

The identity removes one forward; the following changes make the remaining
forward cheap. None of them changes parameter names, state-dict layouts, or the
inference graph.

| Change | Why it matters |
| --- | --- |
| Native fused RMSNorm | Removes a separate normalization launch per block; export keeps an explicit form for ONNX. |
| RoPE cosine/sine computed once per forward | The same tables are shared by Q, K, and all blocks instead of recomputed per block. |
| Vectorized corruption and mask construction on the training device | Removes the Python loop over the batch and the synchronous host-to-device handoff. |
| Pinned CUDA batches and nonblocking copies | Hides host-to-device latency behind compute. |
| One fixed-shape CE with masked reductions | Avoids variable-sized boolean indexing and GPU synchronization; compilation-friendly. |
| Detached on-device metrics | Host synchronization happens once at logging instead of once per metric per micro-batch. |
| Corpus buffer released after row construction | Large in-memory corpora are not retained while training. |
| Optional grouped MoE GEMMs | Exact top-k grouped CUDA AMP GEMMs with a custom backward; dense and reference paths remain available. |

The fast path is implemented by `fast_hybrid_losses`; the legacy path is
`hybrid_losses`. `hybrid_losses` dispatches on `cfg.fast_train`, so a checkpoint
can switch modes without changing its weights. The corruption and mask builder
is shared by both paths.

## 5. Measurements

### 5.1 Dense training

Hardware and software: an 8 GB consumer laptop GPU, BF16, PyTorch 2.13+cu130,
default dense 9M-parameter model, batch 16, context 256. A step includes corruption,
forward, losses, backward, gradient clipping, and fused AdamW. Synthetic
batches avoid data-download and disk variance.

| Version | ms/step | byte-tokens/s | peak CUDA memory |
| --- | ---: | ---: | ---: |
| Original trainer (initial measurement) | 136.4 | 30,024 | 1,645 MiB |
| Optimized legacy two-pass | 65.02 | 62,997 | 1,482 MiB |
| Sprint-DRV `--fast` | 30.38 | 134,835 | 819 MiB |
| Sprint-DRV `--fast --compile` | 24.51 | 167,132 | 641 MiB |

Relative to the original trainer, eager Sprint-DRV is **4.49×** the throughput
and about **50%** lower peak memory. With compilation it is **5.57×** and about
**61%** lower.
Compilation has startup cost and should be amortized over a long run. Batch 32
was also measured but did not beat batch 16 after the fixed-shape loss change.

Reproduce with:

```bash
python scripts/benchmark_training.py --device cuda --batch 16 \
    --steps 25 --warmup 10 --compile
```

### 5.2 MoE systems measurement

On the same 8 GB laptop GPU with a 110.22M MoE + SAINT model, dim 512, context
512, natural Sprint batches and varied Legacy curricula, including backward and
the optimizer step:

| Configuration | Reference | Grouped + fused gate/up | Change | Peak grouped memory |
| --- | ---: | ---: | ---: | ---: |
| Sprint-DRV, batch 4 | 0.741 s/step | 0.480 s/step | 35% less time (1.54×) | 6.20 GiB |
| Legacy DRV, batch 2 | 0.724 s/step | 0.625 s/step | 14% less time (1.16×) | 5.60 GiB |

These are local step-time measurements, not a claim of final model quality or
throughput on another GPU. Numerical parity tests and multi-step memory checks
are the correctness and safety checks; re-run the reference-versus-grouped
benchmark before enabling grouped MoE on a different backend.

### 5.3 Threats to validity

- One GPU, one software stack, one model size. The relative speedup depends on
  the ratio of backbone work to host and kernel-launch overhead.
- Synthetic batches measure training throughput, not data loading, tokenization,
  evaluation, or checkpoint I/O.
- The benchmark does not establish equal sample efficiency. Sprint-DRV changes
  which objectives are optimized per step, most visibly by dropping
  full-sequence alignment.
- `--compile` can recompile for changing sequence lengths and is not free on
  short runs.

## 6. Usage and compatibility

```bash
# Fast training for long runs.
python -m mariana train --fast --compile \
  --data trained-model/teacher-corpus.txt \
  --checkpoint trained-model/mariana-fast.pt \
  --batch 16 --ctx 256 --steps 5000 --no-gen --device cuda

# Resume the same weights with the full-sequence alignment objective.
python -m mariana train --legacy --resume \
  --data trained-model/teacher-corpus.txt \
  --checkpoint trained-model/mariana-fast.pt \
  --steps 5500 --no-gen --device cuda
```

`--steps` is the target total optimizer step, not an increment. The checkpoint
stores the full config, optimizer, and AMP state, so resuming does not require
repeating the mode flags. Model weights, generation, export, and the GGUF/ONNX
numerical parity tests are unchanged.

SAINT latent training can run with either mode. The same hybrid forward receives
thought memory through draft-only cross-attention; the clean prefix is masked
out of that sublayer, so the prefix identity is preserved. SAINT encoding and
compressed decoding are additional passes and are not included in the dense
benchmark above.

## 7. Limitations and open work

- **Dropped alignment.** The largest quality question is whether the missing
  full-sequence `L_align` materially slows speculative acceptance. The intended
  workflow is fast pretraining followed by a short legacy alignment fine-tune,
  but this repository does not claim a matched quality study.
- **Prefix-weighted validation.** Fast AR validation covers the clean prefix,
  not the whole sequence. It is useful for monitoring but not directly
  comparable to legacy validation.
- **Loop-count mismatch.** Fast AR is trained at the sampled loop count while
  legacy AR uses full depth. A controlled comparison should hold loop dropout
  and depth fixed.
- **Compilation and dynamic shapes.** `torch.compile` helps long, stable runs;
  it is a poor fit for very short runs or frequently changing shapes.
- **MoE export.** Grouped MoE kernels are a training optimization. MoE
  checkpoints still cannot be exported to the dense GGUF/ONNX path.
- **Broader hardware.** The measurements are from one consumer GPU. A multi-GPU
  or data-center evaluation should report scaling, not only per-step time.

## 8. Conclusion

Sprint-DRV turns a structural property of the hybrid attention mask into a
training optimization: the clean prefix already contains the exact causal AR
computation. One shared forward therefore trains AR, diffusion, verifier, MTP,
and MoE balancing. The result is a simpler training graph, lower activation
memory, and a measured 4.5–5.6× throughput improvement on the development GPU,
with a clearly bounded tradeoff in full-sequence alignment.

## References

1. Mariana. *Architecture v2.3: Sprint-DRV + hybrid diffusion-AR + SAINT thought.*
   [`architecture.txt`](../../architecture.txt).
2. Mariana. *Sprint-DRV: fast training on a limited GPU.*
   [`docs/fast-training.md`](../fast-training.md).
3. A. Chowdhery et al. *PaLM: Scaling Language Modeling with Pathways.*
   arXiv:2204.02311, 2022.
4. W. Peebles and S. Xie. *Scalable Diffusion Models with Transformers.*
   arXiv:2212.09748, 2023.
5. H. Touvron et al. *LLaMA: Open and Efficient Foundation Language Models.*
   arXiv:2302.13971, 2023.
6. Y. Leviathan, M. Kalman, and Y. Matias. *Fast Inference from Transformers via
   Speculative Decoding.* arXiv:2211.17192, 2023.
7. X. Li, J. Thickstun, I. Gulrajani, P. Liang, and T. Hashimoto.
   *Diffusion-LM Improves Controllable Text Generation.* arXiv:2205.14217, 2022.
8. S. Sahoo et al. *Simple and Effective Masked Diffusion Language Models.*
   arXiv:2406.07524, 2024.
9. A. Pagnoni, R. Pasunuru, P. Rodriguez, et al. *Byte Latent Transformer:
   Patches Scale Better Than Tokens.* arXiv:2412.09871, 2024.
10. P. Nawrot et al. *Hierarchical Transformers Are More Efficient Language
    Models.* Findings of NAACL, 2022.
11. Q. Yu et al. *MEGABYTE: Predicting Million-byte Sequences with Multiscale
    Transformers.* arXiv:2305.07185, 2023.

## Citation

```bibtex
@techreport{mariana_sprint_drv_2026,
  title       = {Sprint-DRV: Single-Pass Training for a Hybrid
                 Diffusion-Autoregressive Language Model},
  author      = {{Mariana contributors}},
  institution = {Mariana},
  year        = {2026},
  version     = {2.3.1},
  url         = {https://github.com/CYFARE/Mariana/blob/main/docs/whitepapers/sprint-drv.md}
}
```
