# Next-Concept Modeling in Mariana: coarse-to-fine residual VQ and parallel concept generation

**Whitepaper · Mariana v2.3.2 · 2026-11-20**

| | |
| --- | --- |
| Status | Implemented; quality and wall-clock advantage not yet established |
| Implementation | `mariana/ncm/` |
| Runbook | [ncm.md](../ncm.md) |
| Related | [Architecture spec](../../architecture.txt), [Sprint-DRV paper](sprint-drv.md) |

## Abstract

Next-Concept Modeling (NCM) changes the unit and the order of generation. A
multiscale residual vector-quantized tokenizer maps a byte sequence to a
hierarchy of codes: a coarse level has a few positions, each finer level adds a
correction, and the final level reconstructs the sequence. A second-stage
generator predicts every level from the previous cumulative reconstruction. All
positions in a level are predicted in parallel, so sampling needs one generator
evaluation per level rather than one per byte.

Mariana's `--ncm` mode is a byte-level implementation of NCM with additional
training and inference mechanisms aimed at the failure modes of the basic
recipe: nested dropout, all-prefix VQ supervision, EMA codebooks with dead-code
recovery, learned resampling, codebook-consistent heads, model-error training, a
reconstruction-error refiner, prompt KV reuse, adaptive early exit, guidance,
minimum-Bayes-risk candidate selection, and full-corpus target coverage.

NCM checkpoints are independent of Mariana's DRV-LM checkpoints and of the
GGUF/ONNX export path. The implemented improvements are design and correctness
work, not an unbenchmarked claim of better language quality or wall-clock speed.
This paper states the model, objectives, implementation, and the evaluation that
would establish such a claim.

## 1. Introduction

Autoregressive byte-level generation is simple and exact, but it is sequential:
one forward per byte, or one forward per block with speculative verification.
NCM asks a different question. Can the model predict a *concept* at a coarse
temporal resolution, then refine it, instead of predicting every byte in order?

The original NCM paper proposes a two-stage answer:

1. **Tokenizer.** A multiscale residual VQ autoencoder turns a sequence into a
   coarse-to-fine hierarchy of discrete codes.
2. **Generator.** A parallel generator predicts the codes level by level,
   conditioning each level on the cumulative reconstruction so far.

The appeal is computational: a level with `l_k` positions costs one generator
evaluation, not `l_k`. The risk is information loss: the generator cannot
recover detail the frozen tokenizer cannot represent, and errors at a coarse
level can accumulate into finer levels.

Mariana's implementation keeps the two-stage structure and adds mechanisms that
address the practical failure modes observed in small runs. It is deliberately
separate from DRV-LM: a different checkpoint format, a different training loop,
and a different generation path. The export path remains the DRV-LM AR model.

## 2. Stage 1: the multiscale tokenizer

### 2.1 Encoder and residual hierarchy

Let `x` be a byte sequence of length `L`, and let `E_θ` be the encoder. The
encoder produces a latent sequence

$$
z = E_\theta(x) \in \mathbb{R}^{L \times d_z}.
$$

A ladder of increasing lengths

$$
1 = l_1 < l_2 < \dots < l_K = L
$$

defines the hierarchy. The default 128-byte ladder is

```text
1, 8, 16, 24, 32, 40, 52, 64, 76, 92, 108, 128
```

and the 512-byte recipe uses 14 gradual levels:

```text
1, 4, 9, 16, 25, 36, 49, 64, 96, 128, 192, 256, 384, 512
```

Changing the ladder requires retraining both stages because the tokenizer is
frozen during stage 2.

For level `k`, the residual quantizer computes

$$
r_k = z - \hat{z}_{k-1},
$$

$$
c_k = \operatorname*{arg\,min}_{j}\;
  \left\|D_k(r_k) - e_{k,j}\right\|_2^2,
$$

$$
\hat{z}_k = \hat{z}_{k-1} + U_k(e_{k,c_k}),
$$

with `ẑ_0 = 0`. `D_k` downsamples the remaining residual to `l_k` positions,
`e_{k,j}` is codebook entry `j` at level `k`, and `U_k` upsamples the correction
back to the full latent length. The cumulative reconstruction `ẑ_k` is therefore
coarse to fine by construction.

A decoder `G_φ` reconstructs bytes from the final cumulative latent:

$$
\hat{x} = G_\phi(\hat{z}_K).
$$

### 2.2 Tokenizer objective

The tokenizer is trained with a reconstruction loss, a vector-quantization
loss, and an optional refiner loss:

$$
\mathcal{L}_{\text{tok}} =
  \mathcal{L}_{\text{recon}}
  + \mathcal{L}_{\text{VQ}}
  + \lambda_{\text{ref}}\mathcal{L}_{\text{refine}}.
$$

Mariana's implementation adds several stabilizers:

- **Per-example nested dropout.** Each tokenizer batch mixes different hierarchy
  cutoffs across rows, with separate full and truncated reconstruction losses.
  Full and truncated latents share one batched decoder call; disabling nested
  dropout removes the duplicate decoder call entirely.
- **All-prefix VQ supervision.** New tokenizers average commitment and
  quantization error over every cumulative reconstruction, matching the
  reference objective instead of supervising only one random prefix and the
  final result. Older checkpoints retain their original selected/full-prefix VQ
  objective on resume.
- **Stable, self-maintaining quantization.** Initialized EMA statistics avoid
  unused-code explosions; expired codes are reassigned to high-error residual
  vectors. Codebooks use EMA updates rather than optimizer gradients.
- **Learned local resampling.** Zero-initialized convolutional refinements start
  from area/linear interpolation. Straight-through gradients are limited to
  level-0 downsampling, as in the reference; applying them at higher residual
  levels is unstable. Upsampling refinements train at every level.
- **Reconstruction-error refiner.** A bidirectional byte refiner is trained on
  actual autoencoder mistakes plus random masking. Evaluation is deterministic;
  random repair masks are training-only. At inference it only rewrites
  low-confidence byte positions and accepts refinements that do not reduce
  confidence.

The refiner does not change the code hierarchy. It reduces visible byte errors
at decode time and is counted separately in the inference budget.

## 3. Stage 2: the parallel concept generator

### 3.1 Inputs and attention

The generator receives a prompt of `prefix_len` bytes and predicts the codes for
every level of the target sequence. For level `k`, the input is the previous
cumulative reconstruction, resized to `l_k` positions, plus a level embedding.
A small learned query bank supplies the initial level-0 positions.

The attention mask is block-diagonal over target levels:

- prompt positions attend to other valid prompt positions;
- target positions attend to the prompt and to their own level;
- target positions do not attend to other target levels;
- RoPE positions restart at zero for each target level.

Therefore every position in a level is predicted in parallel from the same
conditioning, and all levels can be trained in one forward pass. Sampling needs
one generator evaluation per level. This is the central computational claim of
NCM.

### 3.2 Codebook-consistent heads

Each level has a head that predicts a codebook index. New tokenizers default to
the reference implementation's normalized negative squared Euclidean distance:

$$
\ell_j = \frac{2\,p^\top e_j - \|e_j\|^2}{\mathrm{scale}},
$$

where `p` is the projected hidden state, `e_j` is codebook entry `j`, and
`scale` is a cached normalization statistic. This matches the quantizer's
nearest-neighbor rule. A learned inverse temperature and per-code bias remain
trainable. Frozen normalization statistics are cached and refreshed after
checkpoint loads.

Generators inherit tokenizer compatibility settings, so format-1 tokenizers use
legacy cosine heads; existing generators retain their trained head type on
resume. New generators also use depth-scaled residual projection initialization
for stable deep-stack training.

### 3.3 Generator objective

Let `L_k` be the mean cross-entropy at level `k`. The stage-2 loss follows
equation 10 of the NCM paper:

$$
\mathcal{L}_{\text{gen}} =
  \sum_{k=1}^{K} w_k L_k,
\qquad
w_k = \frac{l_k^{1-\alpha}}{\sum_{j=1}^{K} l_j^{1-\alpha}}.
$$

With `α = 0`, every code position is weighted equally. With `α = 1`, every
level is weighted equally. The default is `α = 1`.

Training also uses:

- **Level-dependent random corruption** of the conditioning codes.
- **Self-conditioning:** a second pass feeds back the generator's own
  clean-prefix predictions while keeping clean target codes. A small consistency
  loss limits distribution drift.
- **Prompt dropout** so classifier-free guidance is available at inference.
- **Full-corpus target coverage:** generator windows advance by `ctx_len`, so
  their non-overlapping targets cover the corpus instead of skipping every
  prompt region. Stage 1 trains on full `ctx_len` rows rather than discarding
  prompt bytes. Both stages use identical source cutoffs, including small-corpus
  boundary cases, so stage-1 training cannot leak into stage-2 validation.

### 3.4 Frozen codes and throughput

The tokenizer is frozen during stage 2. Optional `--cache-codes` stores stage-1
codes in compact CPU tensors for repeated-epoch training. Cache construction is
isolated from the training RNG, and cached/uncached updates are regression-tested.
The cache uses 16-bit codes for codebooks up to 32,768 entries and 32-bit codes
above that. It is most useful when its startup and CPU-memory costs amortize over
several epochs.

Frozen upsampling is grouped across levels, code losses and corruption are
vectorized, full prompt batches share broadcast attention masks, and code-only
encoding avoids unused reconstruction metrics. Optional `torch.compile` fuses
the remaining generator forward/backward path. `--flex` skips cross-level
attention blocks with dynamic padding handled by a score modifier.

## 4. Training workflow

The integrated command trains both stages:

```bash
python mariana.py train --ncm \
  --ncm-tokenizer-steps 10000 --steps 20000 \
  --batch 32 --grad-accum 2 --ctx 128 --dim 256 --heads 8 \
  --ncm-prefix 64 \
  --ncm-levels 1,8,16,24,32,40,52,64,76,92,108,128 \
  --ncm-tokenizer trained-model/mariana-ncm-tokenizer.pt \
  --checkpoint trained-model/mariana-ncm.pt --no-gen
```

`--ncm-tokenizer-steps` controls stage 1 and `--steps` controls stage 2. The
advanced `ncm` subcommand exposes every architecture option:

```bash
python mariana.py ncm train \
  --device cpu --tokenizer-steps 20 --steps 20 --batch 4 \
  --ctx 32 --prefix 16 --levels 2,4,8,16,32 \
  --latent-dim 32 --dim 64 --heads 4 --codebook-size 64 \
  --encoder-layers 1 --decoder-layers 1 --refiner-layers 1 \
  --generator-layers 1 --eval-every 20 --no-gen \
  --tokenizer trained-model/ncm-tiny-tokenizer.pt \
  --checkpoint trained-model/ncm-tiny-generator.pt
```

Resume with the same integrated command and a new total generator step count:

```bash
python mariana.py train --ncm --resume --steps 30000 \
  --checkpoint trained-model/mariana-ncm.pt \
  --ncm-tokenizer trained-model/mariana-ncm-tokenizer.pt \
  --data path/to/corpus.txt --no-gen
```

Checkpoint architecture takes precedence on resume; passing `--dim`, `--heads`,
`--generator-layers`, or `--dropout` with `--resume` is rejected before loading.
Objective overrides remain available. A format-1 generator checkpoint applies
the corrected objective, resets its optimizer, and restarts warmup/cosine
scheduling over the additional steps without replacing its trained heads. It
writes format-2 checkpoints; format-1 inference remains backward-compatible.

## 5. Inference

```bash
# One parallel code prediction per level.
python mariana.py ncm gen --prompt "KING: " --gen-len 128 --temperature 0.8

# Greedy cached baseline and a visible coarse-to-fine trajectory.
python mariana.py ncm gen --prompt "KING: " --temperature 0 --show-levels

# Optional adaptive/refined/guided inference.
python mariana.py ncm gen --early-exit 0.995 --refine-steps 2 --guidance 0.5

# Eight candidates, selected by average agreement with the others.
python mariana.py ncm gen --candidates 8 --temperature 1.0 --gen-len 128
```

Prompt keys/values are computed once and shared by all levels. Cached and
uncached greedy decoding are tested for exact agreement. Metrics count the
prompt-prefill call separately from per-level calls.

Additional controls:

- **Adaptive computation.** An optional decoder-confidence threshold stops at a
  partial hierarchy. The minimum useful prefix is enforced, and batched runs use
  the least-confident row so one easy candidate cannot truncate harder rows.
- **Classifier-free guidance.** Guidance ramps from zero at the coarsest level to
  the requested scale at the finest level, matching the reference.
  `--constant-guidance` retains the old fixed-scale behavior.
- **Consensus.** Multiple candidates can be generated as a batch and selected by
  byte-agreement minimum Bayes risk (MBR).
- **Byte refinement.** The tokenizer's refiner rewrites low-confidence byte
  positions and accepts only refinements that do not reduce confidence.

`--gen-len` is capped by the checkpoint's fixed `ctx_len`; this version does not
claim variable-length or multi-window autoregressive decoding. `--no-cache` is a
correctness and benchmarking option, not the recommended default.

### 5.1 Inference cost

For `K` levels, standard NCM generation uses

- one prompt-prefill call,
- one generator evaluation per level,
- one tokenizer decode call, and
- optional refiner calls and candidate multiplicity.

This is the correct comparison unit. A report that compares NCM per-level calls
against AR per-byte calls without counting the decoder, refiner, candidate, and
early-exit behavior is not comparing the same work.

## 6. Evaluation protocol

A useful NCM evaluation separates six questions:

1. **Tokenizer ceiling.** Full-hierarchy and per-level reconstruction accuracy.
   Generated text cannot be expected to exceed the frozen tokenizer's fidelity.
2. **Codebook health.** Utilization, normalized perplexity, and dead-code
   recovery over training.
3. **Generator fit.** Teacher-forced and self-conditioned code loss by level.
4. **Free-running drift.** Greedy free-running code accuracy overall and per
   level. A large gap between teacher-forced `VAL` and free-running `FREE`
   exposes error accumulation that teacher-forced loss hides.
5. **Sample quality and diversity.** Generated text against a matched
   AR/diffusion baseline, with the same data, compute budget, and evaluation
   protocol.
6. **Wall-clock and memory.** Include candidate multiplicity, guidance,
   early-exit decoder calls, refiner calls, and cached versus uncached decoding
   in the inference budget.

The repository's tiny correctness runs are not language-quality results. The
default conditioning corruption is `0.5`, following the reference's selected
TinyStories recipe. Check the printed epoch estimate: runs covering only a few
hundredths of an epoch are correctness checks, not training.

## 7. Implementation map

```text
mariana/cli.py              train/gen --ncm flags and advanced ncm subcommand
mariana/ncm/config.py       serializable architecture and training configuration
mariana/ncm/layers.py       RoPE transformer blocks, sampling, MBR
mariana/ncm/quantizer.py    multiscale residual VQ, EMA, dead-code recovery
mariana/ncm/tokenizer.py    autoencoder and confidence-guided byte refiner
mariana/ncm/generator.py    parallel level prediction and cached generation
mariana/ncm/train.py        two-stage data/training/checkpoint loops
mariana/ncm/cli.py          tokenizer/generator/gen/reconstruct commands
tests/test_ncm.py           tiny CPU correctness tests
```

NCM checkpoints use `stage: "tokenizer"` or `stage: "generator"` and are
independent of DRV-LM checkpoints. The generator checkpoint includes a frozen
copy of the tokenizer, so generation needs only `--checkpoint`.

## 8. Limitations and open work

- **Frozen-tokenizer ceiling.** The generator cannot exceed the tokenizer's
  reconstruction fidelity. Exact formats such as code or Sudoku may need a
  larger codebook, a finer final ladder, or a stronger refiner.
- **Error accumulation.** Coarse-level mistakes can propagate into finer levels.
  Free-running accuracy is the diagnostic; teacher-forced loss alone is not
  enough.
- **Fixed context.** `--gen-len` is capped by `ctx_len`. Variable-length and
  multi-window autoregressive decoding are not implemented.
- **Separate export.** NCM checkpoints do not use the DRV-LM GGUF/ONNX path.
  The NCM generator and tokenizer are custom PyTorch graphs.
- **Compute accounting.** Parallel generation reduces the number of generator
  evaluations, but the tokenizer decoder, refiner, guidance, and candidate
  multiplicity are real costs.
- **Quality claim.** The repository does not claim that NCM beats a matched AR
  or hybrid baseline in quality or wall-clock time. That requires the evaluation
  above.
- **Scale.** The default models are tiny. A larger study should report matched
  data, tokens, compute, and inference budgets.

## 9. Related work

NCM builds on discrete representation learning and multiscale sequence
modeling. VQ-VAE supplies the discrete latent bottleneck; residual vector
quantization supplies the coarse-to-fine residual hierarchy; MEGABYTE and the
Byte Latent Transformer explore multiscale byte and patch prediction; masked
generative modeling supplies the parallel-within-a-block intuition. The
Mariana implementation is byte-level and keeps the tokenizer and generator
separate, so it can be evaluated stage by stage.

## 10. Conclusion

Mariana's NCM mode implements a byte-level multiscale tokenizer and a parallel
next-concept generator, with a large set of practical stabilizers around the
basic recipe. Its value is not assumed. The repository provides the training
commands, checkpoint compatibility rules, inference controls, and evaluation
protocol needed to measure whether one generator evaluation per level is a
better trade than one per byte for a given corpus and deployment target.

## References

1. Mariana. *Mariana-NCM.* [`docs/ncm.md`](../ncm.md).
2. Next-Concept Modeling paper: <https://arxiv.org/abs/2610.14162>.
3. Reference implementation: <https://github.com/HazyResearch/ncm>.
4. Project blog: <https://hazyresearch.stanford.edu/blog/2026-10-12-ncm>.
5. A. van den Oord, O. Vinyals, and K. Kavukcuoglu. *Neural Discrete
   Representation Learning.* arXiv:1711.00937, 2017.
6. N. Zeghidour, A. Luebs, A. Omran, J. Skoglund, and M. Tagliasacchi.
   *SoundStream: An End-to-End Neural Audio Codec.* arXiv:2107.03312, 2021.
7. Q. Yu et al. *MEGABYTE: Predicting Million-byte Sequences with Multiscale
   Transformers.* arXiv:2305.07185, 2023.
8. A. Pagnoni, R. Pasunuru, P. Rodriguez, et al. *Byte Latent Transformer:
   Patches Scale Better Than Tokens.* arXiv:2412.09871, 2024.
9. H. Chang, H. Zhang, L. Jiang, C. Liu, and W. T. Freeman. *MaskGIT: Masked
   Generative Image Transformer.* arXiv:2202.04200, 2022.

## Citation

```bibtex
@techreport{mariana_ncm_2026,
  title       = {Next-Concept Modeling in Mariana: Coarse-to-Fine Residual VQ
                 and Parallel Concept Generation},
  author      = {{Mariana contributors}},
  institution = {Mariana},
  year        = {2026},
  version     = {2.3.2},
  url         = {https://github.com/CYFARE/Mariana/blob/main/docs/whitepapers/ncm.md}
}
```
