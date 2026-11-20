# SAINT and the private latent language

> **Whitepaper:** [SAINT](whitepapers/saint.md) covers the architecture,
> objectives, policy learning, evaluation protocol, and limitations in full.
> This page is the practical runbook.

Mariana can learn an internal language that is not English, not a byte tokenizer,
and not a human-readable summary. Its job is to carry information needed after
the raw context has been removed. New `train --latent` runs enable **SAINT**:
*structured, adaptive, hierarchical, reinforced thought*. Use
`--latent-legacy` for the original fixed discrete code.

SAINT is a mechanism and training harness, not a promise of emergent
superintelligence. The ablations and validation losses below are the test of
whether a particular checkpoint has learned anything useful.

## Architecture

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

The ordinary AR backbone does not read the memory. The exported GGUF/ONNX and
LLaMA-format safetensors paths omit every latent module and remain the ordinary
AR model. Draft-conditioned generation still verifies and repairs tokens against
the unconditioned AR distribution.

### 1. Compositional grammar, not just retrieval

The associative-recall curriculum still stores random key/value tables and asks
for values in independently permuted query order. SAINT adds a compositional
curriculum: the local window contains `key op key` queries, while values remain
in the distant table. Targets include XOR, modular addition/subtraction, and
rotations. Left queries preserve ordered recall; right operands are sampled
independently with replacement, so distinct and repeated operands both occur.
An operator changes how its adjacent keys are interpreted, and reversing
subtraction changes the answer.

Three auxiliary objectives reinforce this pressure:

- next-action prediction, including prediction of EOS;
- adjacent-pair composition predicting the following action;
- forward-versus-reversed pair discrimination and next-parent prediction.

The synthetic batch fraction is `--latent-grammar-prob` and the auxiliary weight
is `--w-latent-grammar`. Ordered recall is controlled separately by
`--latent-recall-prob`.

### 2. Adaptive-length thought and a compute budget

The policy alphabet has `n_latent_symbols` content actions, an EOS action, and a
separate PAD state. PAD is never a policy action. EOS is disabled until
`--latent-min-thoughts` content symbols exist. Every row returns explicit action,
content, and parent masks, so padded positions contribute neither predictions
nor loss.

At inference, a fully halted batch stops executing the thought loop early. The
return used for policy learning includes a `--w-latent-length` penalty per used
thought. More computation is therefore useful only when its likelihood/accuracy
benefit pays for the extra thought. `lat_len`, `lat_stop`, and `n_steps` expose
actual rather than maximum compute.

### 3. Tree-shaped memory

`--latent-block-size K` groups child thoughts into blocks. Masked mean and max
pooling over each block's discrete and continuous states produces a parent
hidden state, which emits a second-level discrete symbol. Parents have their own
codebook, positions, next-parent grammar loss, and parent-to-child reconstruction
loss. Invalid child blocks emit no parent memory.

The decoder receives children, continuous slots, and parents as distinct memory
types. This is a two-level learned tree, not an extra flat codebook with a
hierarchical label.

### 4. High-bandwidth continuous reasoning

Each content thought also emits `--latent-continuous-slots` vectors. During
training they use a reparameterized Gaussian posterior; inference uses its mean.
A normalized KL penalty (`w_latent_kl`) limits unbounded information and scale.
`--latent-refine-steps` residual self-attention/FFN passes let slots exchange
information before decoding, without access to future bytes.

Discrete and continuous channel dropout prevent either representation from
becoming a permanent bypass. At least one channel remains available per row.
The exported AR path never calls any of these modules.

### 5. Policy learning on the code

Straight-through Gumbel training remains useful for differentiable content
embeddings, but it is no longer the only signal. SAINT computes a per-context
return from detached compressed-context NLL, byte accuracy, and the thought
budget:

```text
return = -mean continuation NLL
         + w_task_reward * exact byte accuracy
         - w_length * used_thoughts / max_thoughts
```

A learned value head estimates this return from a detached context summary.
The policy-gradient loss uses the centered advantage, joint child/parent
per-action mean log probability, and detached batch standard-deviation scaling.
This trains child symbols, parent symbols, *and EOS choices* using downstream
performance. The critic uses Smooth L1 loss with its own independent weight;
a small conditional entropy bonus preserves exploration. `lat_rl`, `lat_reward`,
and `lat_value` make this signal visible. It is REINFORCE-style policy learning,
not a claim of PPO or GRPO.

### 6. Diffusion-native thought

Every backbone block has a zero-gated latent cross-attention sublayer in SAINT
checkpoints. It is active only where `region=1` (the diffusion draft). Clean AR
prefixes, compressed AR continuation, AR verification, and export do not use it.
Both Sprint-DRV and legacy training pass the same differentiable memory to the
draft on natural-context batches. Synthetic recall/grammar batches train only the
compressed objective and do not condition the unrelated natural diffusion batch.
Ordinary `gen` automatically summarizes the prompt for this path when the
checkpoint has SAINT diffusion enabled; `--no-latent-diffusion` disables it for
new training runs.

## Additional pressures

The complete latent objective combines:

- compressed continuation CE and masked-local reconstruction CE;
- marginal alphabet usage and conditional policy entropy bonuses;
- grammar, order, hierarchy, and continuous KL terms;
- baseline-centered policy gradient and value loss;
- MoE load balancing when MoE is enabled.

The raw long-range context is never visible to the compressed decoder. The local
window preserves spelling and recency, and window dropout forces reconstruction
through memory. The grammar and recall tasks use random values to prevent a
language-modeling shortcut.

## Training and inspection

```bash
# Full SAINT architecture (enabled by --latent for new runs)
python -m mariana train --latent \
    --thoughts 12 --latent-min-thoughts 2 --latent-block-size 3 \
    --latent-continuous-slots 2 --latent-refine-steps 2 \
    --latent-symbols 256 --latent-window 32 \
    --latent-recall-prob 0.25 --latent-grammar-prob 0.25 \
    --w-latent-rl 0.1 --w-latent-length 0.02 \
    --steps 5000 --checkpoint trained-model/mariana-saint.pt

# Original fixed-length, discrete-only implementation
python -m mariana train --latent --latent-legacy \
    --thoughts 8 --steps 5000 --checkpoint trained-model/mariana-latent-legacy.pt

# Inspect adaptive child and parent codes
python -m mariana think --checkpoint trained-model/mariana-saint.pt \
    --prompt "KING RICHARD III:\n" --gen-len 80

# Generate through compressed AR memory
python -m mariana gen --checkpoint trained-model/mariana-saint.pt \
    --latent --show-code --prompt "KING: " --gen-len 80

# Measure natural-text storage, masked reconstruction, and recall
python scripts/latent_ablation.py --checkpoint trained-model/mariana-saint.pt --mask-local
python scripts/latent_ablation.py --checkpoint trained-model/mariana-saint.pt --recall
```

Resuming always uses the checkpoint's stored architecture. An old checkpoint is
not silently upgraded to SAINT, which would invalidate its codebook and optimizer
state. `Config(use_latent=True)` retains the legacy default for API compatibility;
the CLI opts new latent runs into SAINT.

`torch.compile` remains disabled for latent training because the thought loop
and KV lengths are dynamic. Position tables are pre-sized for the longest
compressed training batch and extended on demand for longer cached generation;
incremental decoding preserves the EOS/PAD key mask. Hybrid drafting and AR use
one canonical trailing prompt slice. SAINT adds context encoding, slot refinement,
compressed decoding, and latent cross-attention work; measure throughput before
a long run.

## Metrics and experiments

| metric | meaning |
| --- | --- |
| `lat` / `lat_win` | compressed continuation CE / local reconstruction CE |
| `lat_ent` | emitted content-symbol usage entropy |
| `lat_len`, `lat_stop` | mean used thoughts / fraction that emitted EOS |
| `lat_parents` | mean valid parent count |
| `lat_grammar`, `lat_pair`, `lat_order` | grammar, composition, and ordering losses |
| `lat_hier` | parent-to-child reconstruction loss |
| `lat_rl`, `lat_reward`, `lat_value` | policy loss, achieved return, critic loss |
| `lat_kl` | continuous posterior information penalty |
| `lat_grammar_task`, `lat_recall` | curriculum mixture indicators |
| `lat_moe_lb` | latent-stage MoE load balancing |
| `lat - ar` | compressed-path information gap |

Useful checks:

1. **Ablate content and order.** `latent_ablation.py` compares true, shuffled,
   randomized, and blank memory. For SAINT, shuffling permutes the mixed memory;
   randomization replaces discrete codes but deliberately leaves continuous
   slots intact. Interpret the penalties accordingly.
2. **Measure adaptation.** Compare `lat_len` and return on easy repeated text,
   natural text, recall, and operator tasks. A shorter code is useful only if
   likelihood is preserved; a larger code is not automatically better.
3. **Sweep the bottleneck.** Vary thoughts, block size, local window, continuous
   slots, and alphabet size. Monitor validation `lat`, emitted entropy, and KL.
4. **Test both decoders.** Compare draft acceptance and speed with a checkpoint
   trained with and without latent diffusion. The AR verifier remains the
   correctness reference.
5. **Check checkpoint parity.** Shared AR weights must produce identical logits
   with latent modules present and absent, and LLaMA conversion must omit all
   SAINT tensors.

The operational question is not whether a glyph resembles an English word. It is
whether removing, randomizing, or reordering memory harms prediction, and whether
adaptive compute improves the return/quality tradeoff.

## Measured status and limitations

Earlier small TinyShakespeare experiments found that natural-text continuation
often ignored the fixed discrete code. An earlier ordered-table curriculum
produced functional unordered memory but little shuffle penalty. Those results
motivate SAINT; they are **not** results demonstrating that SAINT has solved
grammar or reasoning. New mechanisms require new training runs and ablations.

The repository's default models are tiny. Variational slots, hierarchy, RL, and
cross-attention increase capacity and optimization choices, but do not guarantee
emergent language, systematic compositionality, or better sample efficiency.
Policy gradients can be noisy, adaptive halting can collapse to too few or too
many thoughts, and a continuous channel can bypass the discrete code. Use the
metrics and ablations rather than interpreting architectural complexity as
capability.

The code remains lossy and grounded primarily in byte prediction and synthetic
tasks. GGUF/ONNX and LLaMA-format safetensors export only the lossless ordinary
AR path; deploying compressed thought requires the custom PyTorch graph.
