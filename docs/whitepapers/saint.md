# SAINT: structured, adaptive, hierarchical, reinforced thought

**Whitepaper · Mariana v2.3.0 · 2026-11-20**

| | |
| --- | --- |
| Status | Implemented; capability effect not yet established |
| Implementation | `mariana/saint.py`, `mariana/latent.py`, `mariana/model.py` |
| Runbook | [latent-language.md](../latent-language.md) |
| Related | [Architecture spec](../../architecture.txt), [Sprint-DRV paper](sprint-drv.md) |

## Abstract

SAINT is a private latent-memory mechanism for a byte-level language model. The
ordinary model sees at most a short local window and a continuation. A causal
prefill of the long context emits a learned thought memory: content symbols from
a private alphabet, an EOS decision, variational continuous slots, and parent
symbols that summarize blocks of child thoughts. The compressed
decoder then predicts the continuation from that memory plus a small raw-byte
window. The long raw context is not visible to the decoder.

The design targets five failure modes of a fixed discrete code: it can be too
short or too long for a context, it carries no explicit composition pressure, it
has no hierarchy, it is low bandwidth, and its symbols are trained only by
differentiable likelihood. SAINT adds an adaptive EOS policy with a compute
penalty, compositional key/operator/value curricula, parent codes pooled from
child blocks, variational continuous slots with refinement, and a
REINFORCE-style policy gradient on the code decisions. A zero-gated
cross-attention sublayer lets the diffusion draft read the same memory without
making the ordinary AR or export path depend on it.

This paper describes the mechanism and the evaluation protocol, not a capability
result. Earlier tiny experiments with the legacy fixed code found that natural
text often ignored the memory, and no completed study in this repository shows
that SAINT solves grammar, reasoning, or sample efficiency. The repository
therefore reports the implemented objectives, the metrics, and the ablations
that would falsify the design.

## 1. Introduction

A byte-level model with a 256-byte context has a trivial form of memory: keep
the bytes. That does not scale. The interesting question is whether a model can
learn a compact internal language that carries what later bytes need, without
being constrained to a human tokenizer or an English summary.

Mariana's original latent code gave the model a private alphabet and a fixed
number of thought positions. The compressed decoder saw `[thoughts][local
window][continuation]`. The design was intentionally simple, and its limits were
visible:

- a fixed thought count spends the same compute on an easy and a hard context;
- a flat codebook does not force a tree or a composition;
- discrete symbols are a narrow channel;
- the only training signal was the continuation likelihood, which can be
  minimized by ignoring the code when the local window is sufficient;
- a code with no explicit order objective can become an unordered bag.

SAINT keeps the private-alphabet idea and adds structure where those failures
occur. The name is descriptive: **S**tructured, **A**daptive,
**H**ierarchical, **R**einforced **T**hought.

## 2. Problem setting

Let `c` be a context of up to `ctx_len` bytes, `w` the last
`latent_local_window` bytes, and `y` the continuation. A latent model computes

$$
m = \mathrm{Enc}_\theta(c), \qquad
p_\theta(y \mid c) \approx p_\theta(y \mid m, w).
$$

The decoder is not allowed to attend to `c` directly. The local window `w`
preserves spelling and immediate syntax. The memory `m` must carry the rest.

A useful memory has two properties:

1. **Information preservation.** Removing, shuffling, or randomizing `m` should
   hurt continuation likelihood.
2. **Adaptive compute.** The model should be able to stop emitting thoughts when
   more computation no longer pays for itself.

SAINT adds a third: **structure**. Symbols should support composition and
hierarchy, not merely retrieval of a single value.

## 3. Architecture

```text
                              ┌─ discrete symbol + EOS decision
raw context ─ causal prefill ─┤
                              └─ variational continuous thought slots
                                         │
                              iterative slot self-attention
                                         │
                     child thoughts ─ block pooling ─ parent symbols
                                         │
                 ┌───────────────────────┴────────────────────┐
                 │                                            │
   compressed AR: [memory][local bytes]          diffusion draft cross-attention
   continuation CE + auxiliary losses             (draft tokens only; never AR verify)
```

### 3.1 Causal prefill and the thought loop

The context is prefilled with the ordinary causal backbone. At step `i`, the
last hidden state is projected to a policy over

- `n_latent_symbols` content actions,
- one EOS action, and
- a separate PAD state that is never a policy action.

EOS is masked until `latent_min_thoughts` content symbols have been emitted.
This prevents a trivial empty memory from being the first solution the optimizer
finds. Each row returns explicit action, content, and parent masks. Padded
positions contribute neither predictions nor loss.

At inference, the loop stops as soon as every row in the batch has emitted EOS.
`lat_stop`, `lat_len`, and `n_steps` report actual rather than maximum compute.

### 3.2 Discrete symbols

Content symbols use a codebook of learned embeddings. During training, the
content embedding uses the straight-through Gumbel estimator:

$$
\mathrm{hard}(\mathrm{softmax}(z)) + \mathrm{softmax}(z)
- \mathrm{softmax}(z).\mathrm{detach}().
$$

The hard one-hot selects the code; the soft term supplies a gradient to the
policy logits and the codebook. At inference the model samples from the
temperature-scaled policy or takes the argmax.

### 3.3 Continuous slots

Each content thought also emits `latent_continuous_slots` vectors. The projection
produces a Gaussian mean and log-variance; training reparameterizes

$$
z = \mu + \sigma \odot \epsilon, \qquad \sigma = \exp(\tfrac{1}{2}\log\sigma^2),
$$

and inference uses `μ`. A normalized KL penalty limits unbounded information and
scale:

$$
\mathcal{L}_{\text{KL}} =
  \frac{1}{N}\sum_{i=1}^{N}
  \frac{1}{2}\left(\sigma_i^2 + \mu_i^2 - 1 - \log\sigma_i^2\right).
$$

Before decoding, the continuous slots pass through `latent_refine_steps`
residual self-attention/FFN blocks. This lets slots exchange information without
access to future bytes. Slot positions are masked so padded thoughts do not
influence the refinement.

### 3.4 Tree-shaped memory

`latent_block_size = K` groups child thoughts into blocks. Masked mean and max
pooling over each block's discrete and continuous child states produces a parent
hidden state, which emits a second-level symbol from its own codebook and policy.
Parents have their own positions and a next-parent grammar objective. A
parent-to-child readout reconstructs the child symbols with a masked
cross-entropy:

$$
\mathcal{L}_{\text{hier}} =
  \frac{1}{|C|}\sum_{i \in C}
  \mathrm{CE}\big(\mathrm{Readout}(p_{\lfloor i/K \rfloor}),\; s_i\big),
$$

where `C` is the set of content child positions and `p` is the parent hidden
state. Invalid child blocks emit no parent memory. The decoder receives child
symbols, continuous slots, and parents as distinct memory types with type
embeddings; this is a two-level learned tree, not a flat codebook with a
hierarchical label attached.

### 3.5 Channel dropout

During training, discrete and continuous channels are dropped independently per
row, with the constraint that at least one channel remains available. This
prevents either representation from becoming a permanent bypass around the
other.

### 3.6 Diffusion-native thought

Every backbone block in a SAINT checkpoint has a zero-gated latent
cross-attention sublayer:

$$
h' = h + g \odot \mathrm{CrossAttn}(h, m),
$$

with `g` initialized to zero. It is active only where `region = 1`, the
diffusion draft. Clean AR prefixes, compressed AR continuation, AR verification,
and export never read the memory. Synthetic recall and grammar batches train
only the compressed objective and do not condition the unrelated natural
diffusion batch.

This is the key isolation property: the ordinary AR backbone remains the
ordinary AR model. The exported GGUF/ONNX and LLaMA-format safetensors paths
omit every latent module. Draft-conditioned generation still verifies and
repairs tokens against the unconditioned AR distribution.

## 4. Training

### 4.1 Curricula

Two synthetic curricula make the memory learn something other than a local
language-model shortcut.

**Associative recall.** A random key/value table is placed in the distant
context. Queries in the local window ask for values in an independently
permuted order. The memory must store the table, not merely copy the query.

**Compositional grammar.** The local window contains `key op key` queries, while
values remain in the distant table. Targets include XOR, modular
addition/subtraction, and rotations. Left queries preserve ordered recall; right
operands are sampled independently with replacement, so distinct and repeated
operands both occur. An operator changes how its adjacent keys are interpreted,
and reversing subtraction changes the answer.

The mixture is controlled by `latent_recall_prob` and `latent_grammar_prob`.
Three auxiliary objectives reinforce the grammar pressure:

- next-action prediction, including EOS;
- adjacent-pair composition predicting the following action;
- forward-versus-reversed pair discrimination and next-parent prediction.

### 4.2 Compressed-context objective

The compressed decoder receives `[memory][local bytes][continuation]` and
predicts the continuation with a causal mask. Let `L_cont` be the continuation
cross-entropy and `L_win` the local reconstruction cross-entropy. The primary
latent objective is

$$
\mathcal{L}_{\text{latent}} =
  w_{\text{latent}}\left(\mathcal{L}_{\text{cont}}
  + \tfrac{1}{4}\mathcal{L}_{\text{win}}\right).
$$

The local window is additionally masked with probability
`latent_window_dropout` during training, which forces reconstruction through the
memory instead of a copy of the recent bytes.

### 4.3 Regularizers and structure

The complete latent objective adds

$$
\begin{aligned}
\mathcal{L}_{\text{SAINT}} ={}&
  \mathcal{L}_{\text{latent}}
  - w_{\text{entropy}}\,\mathcal{L}_{\text{alphabet entropy}}
  + w_{\text{grammar}}\,\mathcal{L}_{\text{grammar}} \\
  &+ w_{\text{hierarchy}}\,\mathcal{L}_{\text{hier}}
  + w_{\text{kl}}\,\mathcal{L}_{\text{KL}}
  + w_{\text{rl}}\,\mathcal{L}_{\text{policy}}
  + w_{\text{value}}\,\mathcal{L}_{\text{value}} \\
  &- w_{\text{policy entropy}}\,\mathcal{L}_{\text{action entropy}}
  + w_{\text{moe}}\,\mathcal{L}_{\text{MoE-LB}}.
\end{aligned}
$$

`L_grammar` combines next-symbol, pair-composition, order, and next-parent
losses. The first entropy term is the marginal content-symbol distribution; the
second is the normalized conditional action entropy. Together they encourage the
policy to use the alphabet without collapsing to a single symbol. `L_KL` is the
continuous-slot penalty above. `L_value` is the critic loss described next.

### 4.4 Policy learning on the code

Straight-through Gumbel training is a differentiable signal, but it optimizes
the content embedding, not the downstream value of choosing a particular symbol
or stopping. SAINT adds a per-context return:

$$
R = -\overline{\mathrm{NLL}}_{\text{continuation}}
  + w_{\text{task}}\cdot\mathrm{Accuracy}_{\text{byte}}
  - w_{\text{length}}\cdot\frac{\text{used thoughts}}{\text{max thoughts}}.
$$

All terms are detached when forming the return. The last term makes computation
useful only when its likelihood or accuracy benefit pays for the extra thought.

A value head estimates the return from a detached context summary. The policy
loss uses the centered advantage and a detached batch standard deviation:

$$
A = R - V(s), \qquad
\hat{A} = \frac{A}{\max(\mathrm{std}(A), 1)},
$$

$$
\mathcal{L}_{\text{policy}} =
  -\mathbb{E}\left[\hat{A}\cdot
  \overline{\log \pi_\theta(a \mid s)}\right],
$$

where the log-probability is the joint mean over child and parent actions. The
critic minimizes Smooth L1 loss against `R`:

$$
\mathcal{L}_{\text{value}} =
  \mathrm{SmoothL1}(V(s), R).
$$

This trains child symbols, parent symbols, and EOS choices using downstream
performance. It is REINFORCE-style policy learning with a learned baseline, not
PPO or GRPO. The exploration bonus is a small normalized conditional entropy
term.

### 4.5 Interaction with Sprint-DRV

SAINT can run with either Sprint-DRV or the legacy two-pass objective. On
natural-context batches, the same differentiable memory is passed to the
draft-only cross-attention. The clean prefix is masked out of that sublayer, so
the Sprint-DRV prefix identity is unchanged. SAINT encoding, compressed
decoding, and refinement are additional passes; they are not included in the
dense Sprint-DRV benchmark.

## 5. Inference

```text
prompt bytes
    │
    ├─ causal prefill ──> thought loop ──> child symbols + continuous slots + EOS
    │                                  └─> parent symbols
    │
    ├─ compressed AR: [memory][last 32 bytes][continuation] ──> bytes
    │
    └─ diffusion draft: cross-attention to memory ──> draft bytes
                                   │
                         ordinary AR verifier ──> accept / repair
```

- The prompt is summarized once; the thought loop stops early when the whole
  batch has emitted EOS.
- The compressed AR decoder uses the memory plus a configurable local window.
  It never sees the raw long context.
- The diffusion draft may read the memory through the zero-gated cross-attention
  path. The verifier and the committed AR distribution do not.
- `--show-code` prints the child and parent symbols; `--no-latent-diffusion`
  disables draft conditioning.
- The ordinary AR path, and therefore speculative verification, remains lossless
  with respect to the AR distribution.

The latent memory is lossy. GGUF/ONNX and LLaMA-format safetensors export only
the ordinary AR path. Deploying compressed thought requires the custom PyTorch
graph.

## 6. Evaluation protocol

The operational question is not whether a glyph looks like an English word. It
is whether removing, randomizing, or reordering the memory harms prediction, and
whether adaptive compute improves the return/quality tradeoff.

### 6.1 Ablations

`scripts/latent_ablation.py` compares four memory conditions:

| Condition | What it measures |
| --- | --- |
| True memory | Baseline compressed-context loss. |
| Shuffled memory | Whether order or composition matters. Shuffling permutes the mixed memory. |
| Randomized symbols | Whether the discrete code carries content. Continuous slots remain intact. |
| Blank memory | The information gap between memory and no memory. |

The `lat - ar` metric is the compressed-path information gap: the difference
between the compressed decoder's continuation loss and the ordinary AR loss.
A small gap means the memory is nearly as good as the full context; a gap near
zero *and* a zero blank-memory penalty means the memory is being ignored.

### 6.2 Adaptation

Compare `lat_len`, `lat_stop`, and return on easy repeated text, natural text,
recall, and operator tasks. A shorter code is useful only if likelihood is
preserved. A longer code is not automatically better. Sweep the bottleneck:
thoughts, block size, local window, continuous slots, and alphabet size.

### 6.3 Checkpoint parity

Shared AR weights must produce identical logits with latent modules present and
absent, and LLaMA conversion must omit every SAINT tensor. The repository's
latent tests cover symbol shapes, straight-through backward, compressed-context
generation, prefill/KV exactness, and export isolation.

### 6.4 What the repository does and does not show

Earlier small-corpus experiments with the legacy fixed code found that
natural-text continuation often ignored the code. An earlier ordered-table
curriculum produced functional unordered memory but little shuffle penalty.
Those results motivate SAINT; they are not results demonstrating that SAINT has
solved grammar or reasoning. New mechanisms require new training runs and
ablations. The repository's default models are tiny, and the two-step smoke run
used in development is a correctness check, not a language-quality result.

## 7. Limitations and open work

- **No capability result.** The paper describes a training harness. It does not
  claim emergent language, systematic compositionality, or better sample
  efficiency.
- **Policy-gradient noise.** REINFORCE-style learning can be high variance.
  Adaptive halting can collapse to too few or too many thoughts.
- **Channel bypass.** A continuous channel can bypass the discrete code even
  with dropout. The ablations must separate the two channels.
- **Lossy memory.** The compressed decoder is grounded in byte prediction and
  synthetic tasks. Removing the raw context is an information bottleneck by
  construction.
- **Compute overhead.** Context encoding, slot refinement, compressed decoding,
  and cross-attention are additional passes. `torch.compile` remains disabled
  for dynamic latent loops.
- **Export.** The latent path is not exportable to GGUF/ONNX; only the ordinary
  AR path is.
- **Synthetic curricula.** Recall and grammar tasks use random values to prevent
  a language-modeling shortcut. That also means they are not natural-language
  evidence.
- **Scale.** The experiments are small. A larger study should report matched
  tokens, wall-clock time, and memory, not only loss.

## 8. Related work

SAINT draws on discrete representation learning, variational inference,
Gumbel-softmax reparameterization, policy gradients, and adaptive computation.
The key difference from a standard VQ memory is the combination of a private
symbol alphabet, an explicit EOS/length policy, a learned parent level, and a
draft-only cross-attention path that leaves the exported AR model unchanged.

Related lines of work include VQ-VAE and residual vector quantization for
discrete latents, Gumbel-softmax for differentiable categorical training,
REINFORCE and actor-critic methods for non-differentiable decisions, adaptive
computation for learned halting, and compressed-prompt or memory-token methods
for long-context inference. The implementation is deliberately small enough to
run on a consumer GPU, so it should be read as a testbed rather than a scaled
architecture study.

## 9. Conclusion

SAINT turns a fixed private code into an adaptive, hierarchical, multi-channel
memory with a downstream policy signal. It preserves the most important
engineering property of the original design: the ordinary AR path does not read
the memory, so export and speculative verification remain unchanged. The open
question is empirical. The repository provides the objectives, metrics, and
ablations needed to test whether the structure actually helps.

## References

1. Mariana. *SAINT and the private latent language.*
   [`docs/latent-language.md`](../latent-language.md).
2. Mariana. *Architecture v2.3.* [`architecture.txt`](../../architecture.txt).
3. D. P. Kingma and M. Welling. *Auto-Encoding Variational Bayes.*
   arXiv:1312.6114, 2014.
4. E. Jang, S. Gu, and B. Poole. *Categorical Reparameterization with
   Gumbel-Softmax.* arXiv:1611.01144, 2017.
5. A. van den Oord, O. Vinyals, and K. Kavukcuoglu. *Neural Discrete
   Representation Learning.* arXiv:1711.00937, 2017.
6. R. J. Williams. *Simple Statistical Gradient-Following Algorithms for
   Connectionist Reinforcement Learning.* Machine Learning, 8, 1992.
7. A. Graves. *Adaptive Computation Time for Recurrent Neural Networks.*
   arXiv:1603.08983, 2016.
8. A. Bulatov, Y. Kuratov, and M. Burtsev. *Recurrent Memory Transformer.*
   arXiv:2207.06881, 2022.
9. Y. Wu, M. N. Rabe, D. Hutchins, and C. Szegedy. *Memorizing Transformers.*
   arXiv:2203.08913, 2022.
10. J. Mu, X. Li, and N. Goodman. *Learning to Compress Prompts with Gist
    Tokens.* arXiv:2304.08467, 2023.

## Citation

```bibtex
@techreport{mariana_saint_2026,
  title       = {SAINT: Structured, Adaptive, Hierarchical, Reinforced
                 Thought for a Byte-Level Language Model},
  author      = {{Mariana contributors}},
  institution = {Mariana},
  year        = {2026},
  version     = {2.3.0},
  url         = {https://github.com/CYFARE/Mariana/blob/main/docs/whitepapers/saint.md}
}
```
