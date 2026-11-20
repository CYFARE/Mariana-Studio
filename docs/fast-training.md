# Sprint-DRV: fast training on a limited GPU

> **Whitepaper:** [Sprint-DRV](whitepapers/sprint-drv.md) covers the mask
> identity, the shared objective, the systems work, the measurements, and the
> tradeoffs in full. This page is the practical runbook.

Sprint-DRV is a training-architecture change, not a new checkpoint weight format.
Enable it with `python -m mariana train --fast`. Old checkpoints default to the
legacy objective; either mode can resume the same weights.

## Why this design

The original trainer performs a clean full-sequence AR forward and a second
hybrid diffusion forward. That duplicates most backbone FLOPs and activations.
For a 256-byte context, replacing attention with an exotic linear-time module
would not address this duplication: the small model's projections, SwiGLU,
losses, host work, and kernel launches are also substantial.

These research directions were evaluated:

- **Byte Latent Transformer / Hourglass / MEGABYTE:** patching reduces expensive
  global sequence lengths and is attractive for long byte streams. At Mariana's
  short context it also adds local encoder/decoder work and a different AR
  runtime/export path. BLT itself notes that FFNs, not just attention, dominate
  at scale. Fixed patching was therefore not a good minimal-change fit here.
- **PaLM parallel attention+FFN blocks:** reported to be about 15% faster at
  large scale by fusing input projections. A parallel block is not a vanilla
  sequential LLaMA block, so adopting it would invalidate Mariana's GGUF
  conversion. A fused-SwiGLU-only prototype showed no material gain on the
  development GPU and was not retained.
- **Shared-trunk supervision:** the hybrid batch already contains a clean,
  causal, time-zero prefix. Its logits are exact AR conditionals, so the second
  clean pass is unnecessary at those positions. This gives the largest speedup
  without changing model weights, inference, or export.

References:

- Byte Latent Transformer: https://arxiv.org/abs/2412.09871
- Hourglass Transformers: https://aclanthology.org/2022.findings-naacl.117/
- MEGABYTE: https://arxiv.org/abs/2305.07185
- PaLM parallel layers: https://arxiv.org/abs/2204.02311

## Sprint-DRV objective

For cutoff `k`, positions `i < k` have clean inputs, `time=0`, `region=0`, and a
strictly causal mask. Draft-only bidirectional attention begins at rows `i >= k`
and cannot change earlier rows. Consequently:

```
H_hybrid[i] == H_clean_AR[i]        for every i < k
```

One hybrid forward therefore supplies:

- **AR cross-entropy** on clean-prefix next-byte targets;
- **diffusion cross-entropy** on corrupted draft positions (one shared
  fixed-shape CE with AR, followed by masked reductions);
- **verifier BCE** on draft positions;
- **MTP cross-entropy** on exact causal prefix hidden states;
- **MoE load balancing**, when enabled.

Random cutoffs move the AR/diffusion boundary. Sprint-DRV also permits `k=L`,
which creates a fully clean causal batch and gives the final byte AR coverage.
Empty-draft diffusion and verifier losses are zero-safe.

`L_align` is reported as zero in fast mode. Its full-sequence clean teacher
logits in the draft region would require the second pass that Sprint-DRV removes.
Use `--legacy` when that objective is more important than throughput, or do most
pretraining with `--fast` and a shorter `--legacy` fine-tune for speculative
acceptance. Fast validation AR is prefix-weighted and is not numerically
comparable to legacy full-sequence validation AR.

With SAINT latent training, the same hybrid forward also receives thought memory
through draft-only cross-attention. The clean prefix is masked out of that
sublayer, so its exact-AR property is unchanged. SAINT encoding and compressed
decoding are additional passes and are not included in the dense benchmark below.

## Systems changes (math-preserving)

- Native fused `RMSNorm` (explicit form retained during ONNX export).
- RoPE cosine/sine computed once per model forward and shared by Q, K, and all
  blocks, instead of recomputed twice per block.
- Vectorized corruption and mask construction directly on the training device.
- Pinned CUDA batches and nonblocking host-to-device transfer.
- Fixed-shape losses without variable-sized boolean indexing; a single CE
  computes both AR and diffusion terms.
- Detached metrics accumulated on-device; host synchronization only at logging.
- Corpus byte buffer released after constructing its uint8 dataset rows.

These changes also accelerate the legacy objective. No parameter names or state
dict layouts changed.

## Measured on a development GPU

Hardware/software: an 8 GB consumer laptop GPU, BF16, PyTorch 2.13+cu130, default
dense 9M-parameter model, batch 16, context 256. Figures are complete training steps:
corruption, forward, losses, backward, gradient clipping, and fused AdamW.
Synthetic batches avoid data-download and disk variance.

| Version | ms/step | byte-tokens/s | peak CUDA memory |
| --- | ---: | ---: | ---: |
| Original trainer (initial measurement) | 136.4 | 30,024 | 1,645 MiB |
| Optimized legacy two-pass | 65.02 | 62,997 | 1,482 MiB |
| Sprint-DRV `--fast` | 30.38 | 134,835 | 819 MiB |
| Sprint-DRV `--fast --compile` | 24.51 | 167,132 | 641 MiB |

Thus eager Sprint-DRV is **4.49×** the original throughput; adding compilation
for a long run reaches **5.57×** and uses about **61% less peak memory**.
Compilation has startup cost and should be amortised over a long run. Batch 32
was also measured but did not beat batch 16 after the fixed-shape loss change.
These are throughput measurements,
not a claim of identical sample efficiency or final quality. Re-run:

```bash
python scripts/benchmark_training.py --device cuda --batch 16 --steps 25 --warmup 10 --compile
```

## Usage

```bash
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

`--steps` remains the target total optimizer step, not an increment.

## Grouped MoE + SAINT kernels

MoE uses exact top-k grouped CUDA GEMMs in AMP, with a custom autograd backward
because this torch build's `_grouped_mm` backward fails. Gate and up projections
are concatenated into one grouped GEMM and cached by parameter version until the
next optimizer update. Routing, expert weights, outputs, and objectives are
unchanged; bf16 and fp16 tests compare outputs and all gradients with the
reference implementation. CPU, fp32, and `torch.compile` execution retain the
reference path. Use `--no-grouped-moe` to force that reference path explicitly.

Compiling the whole hybrid model was not retained: it was slower than eager and
used more memory. The dynamic SAINT thought loop remains eager.

Measured on an 8 GB consumer laptop GPU with the 110.22M MoE + SAINT model,
dim 512, context 512, natural Sprint batches and varied Legacy curricula,
including backward and the optimizer step:

| Configuration | Reference | Grouped + fused gate/up | Change | Peak grouped memory |
| --- | ---: | ---: | ---: | ---: |
| Sprint-DRV, batch 4 | 0.741 s/step | 0.480 s/step | 35% less time (1.54×) | 6.20 GiB |
| Legacy DRV, batch 2 | 0.724 s/step | 0.625 s/step | 14% less time (1.16×) | 5.60 GiB |

These are local step-time measurements, not a claim of final model quality or
throughput on another GPU. Numerical parity tests and multi-step memory checks
are the correctness/safety checks; use the same reference-vs-grouped benchmark
before enabling it on a different backend.
