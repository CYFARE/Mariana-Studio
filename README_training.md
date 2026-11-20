# Mariana training quickstart

Run from the repository root with the venv active. Checkpoints and corpora go
to `trained-model/`; the directory is created automatically. With no `--data`,
a short run uses the built-in demo corpus; pass `--data` for real training.
**The repository ships without checkpoints or prepared corpora.** For the
external-disk Common Corpus MoE + latent run, see
[`docs/low-vram.md`](docs/low-vram.md).

For the fastest training on the limited GPU, add `--fast` (Sprint-DRV). It uses
one shared forward for AR/diffusion/verifier/MTP and measured 5.57× the original
trainer's throughput at batch 16 / context 256 with compilation, using 61% less
peak memory:

```bash
python -m mariana train --fast \
    --data trained-model/teacher-corpus.txt \
    --checkpoint trained-model/mariana-fast.pt \
    --batch 16 --ctx 256 --steps 5000 --eval-every 250 --compile \
    --no-gen --device cuda
```

Fast mode disables full-sequence `L_align`; `--legacy` restores it and can
resume the same checkpoint. Compilation has startup cost, so omit `--compile`
for very short runs. See `docs/fast-training.md` before comparing fast and
legacy validation losses.

## SAINT latent training

New `--latent` runs enable adaptive EOS-based thought, parent codes, variational
continuous slots, grammar/order tasks, policy gradients with a value baseline,
and diffusion-draft cross-attention. Start with:

```bash
python -m mariana train --latent --fast \
    --data trained-model/teacher-corpus.txt \
    --checkpoint trained-model/mariana-saint.pt \
    --thoughts 12 --latent-min-thoughts 2 --latent-block-size 3 \
    --latent-continuous-slots 2 --latent-refine-steps 2 \
    --latent-recall-prob 0.25 --latent-grammar-prob 0.25 \
    --w-latent-rl 0.1 --w-latent-length 0.02 \
    --steps 5000 --eval-every 250 --no-gen --device cuda
```

Use `--latent-legacy` for the lighter fixed discrete code. Compilation is skipped
for either latent mode. Watch `lat_len`, `lat_reward`, `lat_grammar`, `lat_hier`,
and `lat_kl`, and run `scripts/latent_ablation.py`; do not treat lower training
loss alone as evidence of useful thought. See `docs/latent-language.md`.

## NCM: single-command two-stage training

`--ncm` trains the multiscale tokenizer and then the next-concept generator;
`--ncm-tokenizer-steps` controls stage 1 and `--steps` controls stage 2:

```bash
python mariana.py train --ncm --data trained-model/teacher-corpus.txt --checkpoint trained-model/mariana-ncm.pt --ncm-tokenizer trained-model/mariana-ncm-tokenizer.pt --ncm-tokenizer-steps 10000 --steps 20000 --ctx 128 --dim 256 --heads 8 --batch 32 --grad-accum 2 --no-gen
```

Generate with `python mariana.py gen --ncm --checkpoint trained-model/mariana-ncm.pt`.
Add `--resume` to the training command to resume the generator when its checkpoint
exists, or the tokenizer otherwise. Format-1 generator checkpoints retain their
trained head type and reset their optimizer and LR schedule for the corrected objective.
Generator evaluation prints `FREE` greedy code accuracy per level; use it together
with teacher-forced `VAL` loss to detect coarse-to-fine error accumulation. Use
`python mariana.py ncm --help` for stage-specific training and advanced inference
options; see `docs/ncm.md`.

## GUI: folder → training → chat

Run `python mariana.py studio` (or the `--gui` compatibility alias) and open the
local Mariana Studio page. Select a corpus folder, choose Sprint-DRV, Legacy DRV,
or NCM, configure the model, and watch the data-first training/validation loss
chart, stage progress, and throughput. Raw terminal logs are collapsed by default
and can be shown with **Logs and command**. The
corpus scanner is recursive and supports readable text,
Markdown, source code, CSV, JSON/JSONL, HTML, XML, extensionless files, and Parquet.
Smart extraction handles JSON strings, markup text, and Parquet rows; Raw mode preserves
text files and converts structured rows to JSON. Generated `gui-corpus-*` artifacts
are excluded from scans, and corpus publication uses unique temporary files plus
atomic replacement so concurrent preparation cannot recursively ingest or corrupt
output. Parquet support requires `pyarrow`
(`pip install pyarrow`, or install `requirements-data.txt`). Scans include per-file
reasons for unsupported or unreadable inputs.

For Sprint/Legacy DRV, **Advanced settings** includes sparse MoE and an
Off/SAINT/Legacy latent selector. SAINT exposes maximum/minimum thoughts, parent
block size, continuous slots and refinements, grammar probability, and RL weight;
legacy latent exposes the shared alphabet, thought count, and local window. These
architecture choices apply to fresh runs; resumed checkpoints keep their stored
architecture. NCM has separate tokenizer/generator controls and does not accept
these DRV enhancement flags.

**Training modes and pause/resume.** DRV supports fresh and continued training.
NCM additionally supports updating the tokenizer, retraining the generator from
its current tokenizer, updating both, or running either stage alone. Continue
model is the recommended way to add data while preserving the tokenizer and code
hierarchy. Updating a tokenizer changes code meanings, so the generator must be
retrained; that workflow does not preserve generator knowledge. Mix new data
with representative old data to reduce catastrophic forgetting.

Steps entered for continue/update modes are additional steps. **Pause & save**
writes the model, optimizer, and AMP state after the current optimizer step and
then stops, allowing a later continue run. Overwrite backups are enabled and
stored under `backups/<model-name>/`. The model-name field offers existing checkpoints for repeated
updates to one file.

After training, open **Chat**, select the new `.pt` checkpoint, and generate
locally. Checkpoints from both DRV-LM and NCM are detected automatically. The
frontend uses local browser assets with no additional frontend packages or CDN.
Light theme is the default; the theme toggle
saves a dark mode based on the One Dark palette. Use `--no-browser` to suppress
auto-launch and `--port 8765` to change the port. Keep the default `127.0.0.1` host unless you
explicitly intend to expose the training and file-browsing API to other machines.

### Smart Options

**Auto-apply smart options** is enabled by default. Scanning a folder analyzes the
actual extracted local file contents and applies recommendations; **Recalculate
smart options** repeats the analysis after corpus or hardware changes. Start
Training also recalculates when the folder, extraction mode, technique, or device
has changed. No Hugging Face metadata, URL, or remote service is used. Large
prepared corpora are memory-mapped during training instead of being copied into RAM.

Train/validation splitting happens on disjoint source bytes before any
small-corpus repetition. Repetitions never cross partitions; corpora too small for
two complete rows are rejected rather than given a leaky validation set. Both
trainers also reject empty batch loaders with an actionable batch-size error.

The calculator uses UTF-8 byte-tokens after extraction, dataset passes (three for
small corpora, two for medium, one for large), context/model size, device memory,
estimated optimizer/activation memory, CPU count, and available RAM. It sets
steps, batch and accumulation, context and hidden sizes, attention heads, learning
rates, warmup/evaluation cadence, DRV block architecture, and NCM levels/tokenizer
steps. JSONL extraction is streamed so large files are not loaded wholesale during
analysis. These are conservative heuristics, not a guarantee against overfitting
or underfitting: monitor validation loss and adjust the applied values as needed.

On the 8 GiB laptop-GPU profile with a large corpus and 512-byte context,
measured speed settings are Sprint-DRV batch 12 × 3 with compilation, Legacy
batch 6 × 6 with compilation, NCM generator batch 4 × 8, and NCM tokenizer
batch 24 × 2.
NCM tokenizer steps are scaled independently to preserve its token budget. The
GUI enables compilation for DRV on CUDA; compilation has substantial startup
cost, so disable it for short runs. FlexAttention and TF32 were slower in this
profile and are not enabled by Smart Options.

MoE and latent/SAINT settings are included in parameter and memory estimates.
Their batches are calculated rather than reusing dense-only presets, and latent
runs disable compilation. On the 8 GiB profile, the 110.22M MoE + SAINT
configuration measured 2.50 GiB at batch 1, 3.49 GiB at batch 2, and 5.12 GiB at
batch 4 peak allocated memory. Sprint-DRV therefore recommends **batch 4 ×
grad accumulation 8**. Legacy retains a second full AR graph and alignment
activations, so it uses a separate 2× activation estimate: after six varied steps,
batch 2 peaked at 4.95 GiB, batch 3 at 6.02 GiB, and batch 4 reached OOM.
Legacy recommends **batch 2 × grad accumulation 16**. Use batch 1 × 32 if other
GPU processes are resident or if memory is still tight.

For MoE speed, grouped CUDA AMP kernels are enabled by default, with fused
gate/up projections and exact reference fallbacks for CPU/fp32/compile. On this
110.22M configuration they reduced Sprint step time from 0.741s to 0.480s and
Legacy from 0.724s to 0.625s, including backward and optimizer work. Peak memory
was 6.20 GiB and 5.60 GiB respectively. Use `--no-grouped-moe` or uncheck grouped
MoE kernels in the GUI for the reference implementation. See
`docs/fast-training.md` for measurements and numerical-parity coverage.

## 1. Generate distillation data

### Local llama.cpp

```bash
python scripts/distill_teacher.py \
    --url http://127.0.0.1:8080 --api chat \
    --out trained-model/teacher-corpus.txt \
    --samples-per-prompt 4 --max-tokens 512 \
    --workers 2 --drop-truncated
```

The local model is auto-discovered. `--append` resumes an existing corpus.
For local top-k logprobs, best-of-n and confidence filtering, use
`--api openai --candidates 2 --min-mean-logprob -1.2`.

### DeepSeek V4.1 Flash

```bash
export DEEPSEEK_API_KEY=sk-...

python scripts/distill_teacher.py \
    --provider deepseek --model deepseek-flash \
    --out trained-model/teacher-corpus-deepseek.txt \
    --samples-per-prompt 4 --max-tokens 512 \
    --workers 4 --drop-truncated
```

`deepseek-flash` is served by DeepSeek-V4.1-Flash-1017. If your account uses a
different model ID, pass `--model` or set `DEEPSEEK_MODEL`; `--model auto`
queries the provider's model list and picks the newest.
DeepSeek uses the chat endpoint; for logprobs use a local teacher with
`--api openai`. Requests leave the machine and are billed by DeepSeek.

Smoke test either provider with `--max-samples 1 --max-tokens 512 --workers 1`.

## 2. Train MoE + MTP

```bash
python -m mariana train \
    --data trained-model/teacher-corpus.txt \
    --checkpoint trained-model/mariana-moe.pt \
    --moe --ctx 256 --dim 384 \
    --batch 8 --grad-accum 2 \
    --steps 500 --eval-every 250 \
    --compile --no-gen --device cuda
```

Low-VRAM version (fits while the local teacher is loaded):

```bash
python -m mariana train \
    --data trained-model/teacher-corpus.txt \
    --checkpoint trained-model/mariana-moe-small.pt \
    --moe --ctx 128 --dim 128 \
    --batch 4 --grad-accum 4 \
    --steps 500 --eval-every 250 \
    --no-gen --device cuda
```

MTP is on by default (`--no-mtp` disables it). Choose steps from the corpus:
`steps ≈ corpus_bytes / (batch * ctx) * epochs`; use 1–3 epochs. The trainer
prints the one-epoch estimate and warns above 5 epochs. The current 71 KB
teacher corpus is ~120 steps/epoch, so 500 steps is about 4 epochs; 20,000
steps memorises it. Unload the local teacher first if both share the GPU (the
35B teacher used ~6.4 GB).

## 3. Resume, generate, export

```bash
python -m mariana train --resume \
    --checkpoint trained-model/mariana-moe.pt \
    --data trained-model/teacher-corpus.txt \
    --steps 1000 --no-gen --device cuda

python -m mariana gen --checkpoint trained-model/mariana-moe.pt \
    --prompt "Explain photosynthesis in two paragraphs." --gen-len 200

# dense checkpoints only: MoE export is not implemented
python -m mariana export --checkpoint trained-model/mariana-dense.pt \
    --format all --out exports
```
