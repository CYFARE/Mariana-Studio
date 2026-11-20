# Mariana-NCM

> **Whitepaper:** [NCM](whitepapers/ncm.md) covers the tokenizer and generator
> design, objectives, inference controls, and evaluation protocol in full. This
> page is the practical runbook.

Mariana's `--ncm` mode is a byte-level implementation of **Next-Concept Modeling** (NCM),
with additional training and inference mechanisms aimed at the failure modes of
the original design. It is separate from Mariana's DRV-LM checkpoints and export
path.

References: [paper](https://arxiv.org/abs/2610.14162),
[reference implementation](https://github.com/HazyResearch/ncm), and
[project blog](https://hazyresearch.stanford.edu/blog/2026-10-12-ncm).

## Model

Stage 1 is a multiscale residual VQ autoencoder. For target bytes `x`, its encoder
produces `z`. Level `k` downsamples the remaining residual to `l_k` positions,
assigns each position to that level's nearest codebook entry, and upsamples the
correction. Cumulative reconstructions are therefore coarse to fine:

```
z_hat(k) = z_hat(k-1) + Upsample_k(codebook_k(r_k))
r_k = argmin distance(Downsample_k(z - z_hat(k-1)), codebook_k)
```

The default 128-byte ladder is `(1, 8, 16, 24, 32, 40, 52, 64, 76, 92, 108, 128)`,
matching the reference TinyStories/LM1B hierarchy. The 512-byte recipe uses 14
gradual levels instead of the former five-level `1,32,128,256,512` ladder.
Changing the ladder requires retraining both stages because the tokenizer is frozen.
Stage 2 predicts every level from the previous cumulative reconstruction. A training
forward contains every level with block-diagonal target attention: positions in a
level attend to each other and the prompt, but not to other target levels. Thus
all positions in a level are predicted in parallel, and sampling needs one
generator evaluation per level rather than one per byte.

Prompt bytes use prefix-LM attention. Prompt positions attend to other valid
prompt positions, never to targets. Target positions attend to the prompt and
their own level. RoPE positions restart at zero for each target level.

The stage-2 loss follows equation 10 of the NCM paper. If `L_k` is the mean
cross-entropy at level `k`, its weight is
`l_k^(1-alpha) / sum_j(l_j^(1-alpha))`. Thus `alpha=0` weights every code
position equally and `alpha=1` weights every level equally.

## Improvements over the basic recipe

- **Per-example nested dropout:** each tokenizer batch mixes different hierarchy
  cutoffs across rows, with separate full and truncated reconstruction losses.
  Full and truncated latents share one batched decoder call; disabling nested
  dropout removes the duplicate decoder call entirely.
- **All-prefix VQ supervision:** new tokenizers average commitment and quantization
  error over every cumulative reconstruction, matching the reference objective
  instead of supervising only one random prefix and the final result. Older
  checkpoints retain their original selected/full-prefix VQ objective on resume.
- **Stable, self-maintaining quantization:** initialized EMA statistics avoid
  unused-code explosions; expired codes are reassigned to high-error residual
  vectors. Codebooks use EMA updates rather than optimizer gradients.
- **Learned local resampling:** zero-initialized convolutional refinements start
  from area/linear interpolation. Straight-through gradients are limited to
  level-0 downsampling, as in the reference; applying them at higher residual
  levels is unstable. Upsampling refinements train at every level.
- **Codebook-consistent generator heads:** new tokenizers default to the reference
  implementation's normalized negative squared Euclidean distance, matching the
  quantizer's nearest-neighbor rule. A learned inverse temperature and per-code
  bias remain trainable. Frozen normalization statistics are cached and refreshed
  after checkpoint loads. Generators inherit tokenizer compatibility settings, so
  format-1 tokenizers use legacy cosine heads; existing generators retain their
  trained head type on resume. New generators also use depth-scaled residual
  projection initialization for stable deep-stack training.
- **Model-error training:** in addition to level-dependent random corruption,
  self-conditioned passes feed back the generator's own clean-prefix predictions
  while keeping clean target codes. A small consistency loss limits distribution
  drift.
- **Reconstruction-error refiner:** a bidirectional byte refiner is trained on
  actual autoencoder mistakes plus random masking. Evaluation is deterministic;
  random repair masks are training-only. At inference it only rewrites
  low-confidence byte positions and accepts refinements that do not reduce
  confidence.
- **Prompt KV reuse:** prompt keys/values are computed once and shared by all
  levels. Cached and uncached greedy decoding are tested for exact agreement.
  Metrics count this prompt-prefill call separately from per-level calls.
- **Adaptive computation:** an optional decoder-confidence threshold can stop at
  a partial hierarchy. The minimum useful prefix is enforced, and batched runs
  use the least-confident row so one easy candidate cannot truncate harder rows.
  Extra decoder evaluations are reported separately from generator evaluations.
- **Guidance and consensus:** classifier-free guidance is available after prompt
  dropout training. Guidance ramps from zero at the coarsest level to the requested
  scale at the finest level, matching the reference; `--constant-guidance` retains
  the old fixed-scale behavior. Multiple candidates can be generated as a batch
  and selected by byte-agreement minimum Bayes risk (MBR).
- **Full-corpus target coverage:** generator windows advance by `ctx_len`, so their
  non-overlapping targets cover the corpus instead of skipping every prompt region.
  Stage 1 trains on full `ctx_len` rows rather than discarding prompt bytes. Both
  stages use identical source cutoffs, including small-corpus boundary cases, so
  stage-1 training cannot leak into stage-2 validation.
- **Higher-throughput training:** frozen upsampling is grouped across levels,
  code losses and corruption are vectorized, full prompt batches share broadcast
  attention masks, and code-only encoding avoids unused reconstruction metrics.
  Optional `torch.compile` fuses the remaining generator forward/backward path;
  `--flex` additionally skips cross-level attention blocks with dynamic padding
  handled exactly by a score modifier.
- **Reusable frozen codes:** `--cache-codes` stores stage-1 codes in compact CPU
  tensors for repeated-epoch generator training. Cache construction is isolated
  from the training RNG, and cached/uncached updates are regression-tested.
- **Selective regularization:** AdamW decay applies to matrix/tensor weights, not
  norms, biases, or scalar temperatures. Older one-group optimizer checkpoints
  resume with their moments intact and migrate to the new parameter groups.
- **Operational safety:** device-independent AMP, gradient accumulation, atomic
  checkpoints, resume support, and no tokenizer or transformer dependency beyond
  PyTorch. Unscaled nonfinite gradients abort before they can corrupt weights;
  fp16 GradScaler retains its normal overflow-skip behavior. Checkpoints record
  LR-cycle boundaries so completed and partially completed runs resume correctly.

These are implemented design improvements, not a claim of a measured quality or
speed advantage on a particular dataset. That requires matched training and
wall-clock benchmarks. In particular, the two-step smoke run used in development
is a correctness check, not a language-quality result.

## Train

Activate the repository environment first. The examples use CPU for portability;
omit `--device cpu` to select CUDA/XPU/MPS automatically.

### One-command training

`train --ncm` runs both stages. `--ncm-tokenizer-steps` controls the tokenizer
and `--steps` controls the generator; `--no-gen` skips the final demo:

```bash
python mariana.py train --ncm \
  --ncm-tokenizer-steps 10000 --steps 20000 --batch 32 --grad-accum 2 \
  --ctx 128 --dim 256 --heads 8 --ncm-prefix 64 \
  --ncm-levels 1,8,16,24,32,40,52,64,76,92,108,128 \
  --ncm-tokenizer trained-model/mariana-ncm-tokenizer.pt \
  --checkpoint trained-model/mariana-ncm.pt --no-gen
```

With no `--data`, the command uses Mariana's small built-in demo corpus; pass
`--data path/to/corpus.txt` for real training. Generator checkpoints include a
frozen copy of the tokenizer, so generation needs only `--checkpoint`.

For long generator runs, `--compile` removes Python/kernel overhead; its first
steps include a one-time compilation cost. On supported PyTorch builds, combine
it with `--flex` to exploit the block-sparse hierarchy mask. FlexAttention requires
an attention head dimension of at least 16; unsupported sizes fail before training
with an actionable error. FlexAttention without compilation uses an unfused fallback
and is not the recommended throughput path.
Advanced `ncm` commands also offer `--cache-codes` for repeated-epoch generator
training; the integrated
`train --ncm` spelling is `--ncm-cache-codes`. This avoids repeatedly encoding the
same frozen targets. The cache uses 16-bit codes for codebooks up to 32,768 entries
and 32-bit codes above that. It is most useful when its startup and CPU-memory
costs amortize over several epochs; single-epoch runs normally leave it disabled.
Both options work with resume and do not change the checkpoint format.

Mariana Studio exposes the same controls under **Advanced settings** when NCM is
selected. Smart options recommend compile and FlexAttention on compatible CUDA
systems, while tokenizer-only modes suppress generator-only flags. The chat view's
**NCM inference options** exposes guidance scheduling, top-k, early exit, byte
refinement, and MBR candidates; ordinary DRV chat is unaffected.

### Tiny correctness run

The advanced `ncm train` form exposes every architecture option while retaining
the same two-stage behavior:

```bash
python mariana.py ncm train \
  --device cpu --tokenizer-steps 20 --steps 20 --batch 4 \
  --ctx 32 --prefix 16 --levels 2,4,8,16,32 \
  --latent-dim 32 --dim 64 --heads 4 --codebook-size 64 \
  --encoder-layers 1 --decoder-layers 1 --refiner-layers 1 \
  --generator-layers 1 --eval-every 20 --no-gen \
  --tokenizer trained-model/ncm-tiny-tokenizer.pt \
  --checkpoint trained-model/ncm-tiny-generator.pt

python mariana.py ncm gen \
  --device cpu --checkpoint trained-model/ncm-tiny-generator.pt \
  --prompt "KING: " --gen-len 32 --show-levels
```

Use `python mariana.py ncm tokenizer` or `python mariana.py ncm generator` to
run only one stage. `tokenize` remains an alias for `tokenizer`.

Resume with the same integrated command and a new total generator step count:

```bash
python mariana.py train --ncm --resume --steps 30000 \
  --checkpoint trained-model/mariana-ncm.pt \
  --ncm-tokenizer trained-model/mariana-ncm-tokenizer.pt \
  --data path/to/corpus.txt --no-gen
```

This resumes the generator when its checkpoint exists, or the tokenizer when only
that stage exists. Checkpoint architecture takes precedence on resume; passing
`--dim`, `--heads`, `--generator-layers`, or `--dropout` with `--resume` is
rejected before loading. Objective overrides remain available. The requested total
step count must exceed the saved step. Resuming a format-1
generator applies the corrected objective, resets its optimizer, and restarts
warmup/cosine scheduling over the additional steps without replacing its trained
heads. It writes format-2 checkpoints; format-1 inference remains backward-compatible.
Current checkpoints also record the active LR-cycle start and total steps. Extending
a completed cycle starts a fresh warmup/cosine cycle over only the additional steps;
resuming an unfinished cycle preserves its origin instead of introducing an LR jump.

## Generate and inspect

```bash
# Standard NCM: one parallel code prediction per level.
python mariana.py ncm gen --prompt "KING: " --gen-len 128 --temperature 0.8

# Greedy cached baseline and a visible coarse-to-fine trajectory.
python mariana.py ncm gen --prompt "KING: " --temperature 0 --show-levels

# Optional adaptive/refined/guided inference.
python mariana.py ncm gen --early-exit 0.995 --refine-steps 2 --guidance 0.5

# Eight candidates, selected by average agreement with the other candidates.
python mariana.py ncm gen --candidates 8 --temperature 1.0 --gen-len 128

# Isolate the autoencoder's reconstruction ceiling before training a generator.
python mariana.py ncm reconstruct \
  --checkpoint trained-model/mariana-ncm-tokenizer.pt \
  --text "A reconstruction example." --show-levels
```

`--gen-len` is capped by the checkpoint's fixed `ctx_len`; this version does not
claim variable-length or multi-window autoregressive decoding. `--no-cache` is a
correctness/benchmarking option, not the recommended default.

## Metrics and evaluation

Tokenizer logs include reconstruction loss/accuracy, VQ loss, refiner loss,
batch codebook utilization, and normalized codebook perplexity. Generator logs
include clean-code cross-entropy by level, code accuracy, self-conditioning
frequency, and consistency loss. Each generator evaluation also prints `FREE`:
greedy free-running code accuracy overall and per level. A large gap between
`VAL` and `FREE` exposes error accumulation that teacher-forced loss hides.
The default conditioning corruption is `0.5`, following the reference's selected
TinyStories recipe. Keep it enabled on short runs; `--self-condition 0` disables
the additional model-error training pass. Consistency KL is averaged per code
position before applying its weight. Check the printed epoch estimate as well:
runs covering only a few hundredths of an epoch are correctness checks, not
language-quality training.

A useful evaluation must report separately:

1. Full-hierarchy and per-level tokenizer reconstruction accuracy.
2. Codebook utilization/perplexity and dead-code recovery over training.
3. Teacher-forced and self-conditioned code loss by level.
4. Generated-text quality and diversity against a matched AR/diffusion baseline.
5. Wall-clock time and memory, not just NFE. Include candidate multiplicity,
   guidance, early-exit decoder calls, and refiner calls in the inference budget.
6. Cached versus uncached outputs and timing. Do not compare a cached NCM number
   with an uncached baseline.

The fixed byte hierarchy also has a hard information ceiling: generated text
cannot be expected to exceed the frozen tokenizer's reconstruction fidelity.
Measure that ceiling first. Larger codebooks, a finer final ladder, or a stronger
refiner may be needed for exact formats such as code or Sudoku.

## Files

```
mariana/cli.py              train/gen --ncm flags and advanced ncm subcommand
mariana/ncm/config.py       serializable architecture and training configuration
mariana/ncm/layers.py       RoPE transformer blocks, sampling, MBR
mariana/ncm/quantizer.py    multiscale residual VQ, EMA, dead-code recovery
mariana/ncm/tokenizer.py    autoencoder and confidence-guided byte refiner
mariana/ncm/generator.py    parallel level prediction and cached generation
mariana/ncm/train.py        two-stage data/training/checkpoint loops
mariana/ncm/cli.py          tokenizer/generator/gen/reconstruct commands
tests/test_ncm.py          tiny CPU correctness tests
```
