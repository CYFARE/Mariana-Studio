# Distillation from a local teacher server

For the copy-paste command runbook, see [`../README_training.md`](../README_training.md).

Mariana can learn from a larger open model running behind `llama.cpp`'s HTTP
server.  The practical route is **sequence-level distillation**: the teacher
writes text, the student trains on it.  This works across tokenizers, which
matters here because a 35B Llama-family teacher uses a ~248k-token BPE
vocabulary while Mariana is byte-level (258 tokens).

The teacher's log-probabilities are still useful.  The script can generate
several candidates per prompt, keep the one with the highest mean
log-probability, and reject low-confidence or truncated samples.

## 1. Verify the teacher

```bash
curl -s http://127.0.0.1:8080/v1/models
```

The response includes the model name and context length.  For a reasoning
model, the OpenAI **chat** endpoint is usually the cleanest source: it returns
the final answer in `message.content` and puts hidden reasoning in
`message.reasoning_content`, so Mariana does not learn to emit thinking tags.

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"MODEL","messages":[{"role":"user","content":"Explain photosynthesis."}],"max_tokens":256}'
```

If `finish_reason` is `length`, increase `--max-tokens`; reasoning models spend
tokens before the visible answer.

### DeepSeek V4.1 Flash (cloud)

DeepSeek exposes an OpenAI-compatible chat endpoint.  Use the provider shortcut
instead of a local URL:

```bash
export DEEPSEEK_API_KEY=sk-...

python scripts/distill_teacher.py \
    --provider deepseek \
    --model deepseek-flash \
    --out trained-model/teacher-corpus-deepseek.txt \
    --samples-per-prompt 4 --max-tokens 512 \
    --workers 4 --drop-truncated
```

- The default URL is `https://api.deepseek.com`; override with `--url`.
- `deepseek-flash` is the API model served by DeepSeek-V4.1-Flash-1017.
  Override with `--model` or `DEEPSEEK_MODEL` if your account exposes a
  different ID.  `--model auto` queries the provider's model list and picks the
  newest.
- DeepSeek uses the chat endpoint, so `--api openai` logprobs, `--candidates`
  scoring and `--min-mean-logprob` are not available there.  Use the local
  llama.cpp teacher for those.
- Cloud generation leaves the machine and is billed by DeepSeek.

## 2. Generate a corpus

```bash
# the safe default: clean chat answers
python scripts/distill_teacher.py \
    --url http://127.0.0.1:8080 \
    --api chat \
    --out trained-model/teacher-corpus.txt \
    --samples-per-prompt 4 \
    --max-tokens 512 \
    --workers 2

# if the server exposes top-k logprobs, request them and filter
python scripts/distill_teacher.py \
    --url http://127.0.0.1:8080 \
    --api openai \
    --out trained-model/teacher-corpus.txt \
    --samples-per-prompt 4 --candidates 2 \
    --min-mean-logprob -1.2 \
    --max-tokens 512
```

Useful flags:

| flag | effect |
| --- | --- |
| `--prompts FILE` | one prompt per line; otherwise built-in diverse prompts |
| `--samples-per-prompt N` | completions per prompt |
| `--candidates N` | generate N and keep the highest mean logprob |
| `--min-mean-logprob X` | reject low-confidence samples (OpenAI/native logprobs) |
| `--drop-truncated` | reject samples that hit `max_tokens` |
| `--no-prompt` | train only on answers, not prompt + answer |
| `--prompt-separator` | text between prompt and answer (default blank line) |
| `--keep-think` | keep `<think>...</think>` blocks (default strips them) |
| `--append` | append and skip already-written samples |
| `--dry-run` | show the discovered model and planned output |

The script writes:

- `teacher-corpus.txt` — plain UTF-8 text, ready for `--data`
- `teacher-corpus.txt.manifest.json` — teacher URL/model, counts, parameters
- `teacher-corpus.txt.seen.json` — dedup hashes for `--append`

## 3. Train MoE + MTP

MTP is on by default; `--no-mtp` disables it.  MoE is enabled with `--moe`.

```bash
python -m mariana train \
    --data trained-model/teacher-corpus.txt \
    --checkpoint trained-model/mariana-moe.pt \
    --moe \
    --ctx 256 --dim 384 \
    --batch 8 --grad-accum 2 \
    --steps 500 --eval-every 250 \
    --compile --no-gen --device cuda
```

Suggested sequence:

1. **Bulk pretraining** on real/open data (e.g. Common Corpus) for general
   coverage.
2. **Distillation fine-tune** on teacher data for style, instruction following,
   and the domains you care about.
3. **Latent-code training** separately with `--latent` if you want the private
   symbol channel.  A checkpoint cannot gain latent modules by resuming an
   older non-latent run; start a new latent run (optionally initialising from a
   base checkpoint in the future).

## What this does and does not buy you

**It can buy data efficiency.** Teacher text is cleaner and more diverse than
scraped text, so the student can reach a target validation loss in fewer
steps.  Best-of-n and confidence filtering remove many low-quality samples.

**It does not make each step cheaper.** MoE adds capacity and routing overhead;
MTP adds a small head and a second prediction target.  Distillation changes the
data, not the arithmetic.  If raw wall-clock per step matters, use AMP (already
automatic), `--compile`, a larger batch, and `--grad-accum` to fit your GPU.

**It does not transfer 35B capacity.** A 2M–20M-parameter byte model can learn
the teacher's style and some factual patterns, not its full knowledge or
reasoning.  Scale the student or narrow the domain before expecting more.

**Generation is the bottleneck.** A 35B Q4 model on a local GPU may produce
5–25 tokens/s.  A 1M-token corpus can take many hours.  Generate targeted
domains, use multiple server slots (`--workers`), and reuse the corpus for
several epochs rather than trying to mirror a huge generic dataset.

## Measured on a small local GPU

These numbers come from a 35B Q4_K_M teacher on an 8 GB consumer laptop GPU:

| path | observed |
| --- | --- |
| `/v1/completions` generation | ~11 tokens/s |
| `/v1/chat/completions` generation | ~22 tokens/s |
| teacher VRAM while loaded | ~6.4 GB |
| dense + MTP student, 2.28M, batch 4 | ~7.4k tokens/s |
| MoE + MTP student, 5.82M, batch 4 | ~5.7k tokens/s |

The teacher and the student compete for the same GPU.  A batch-16 MoE run
OOM'd while the teacher was loaded; batch 4 fit.  Either unload the teacher
during student training, lower the student batch/dim, or run the teacher on a
different host.

MoE is **not** faster per token here: it is roughly 30% slower than the dense
student at the same width, while carrying 2.5× the parameters.  Its value is
capacity per step.  Distillation's value is fewer steps to a target quality;
do not expect both to make wall-clock training cheaper.

## Why not full logit distillation?

Logit distillation needs a shared vocabulary.  The teacher emits a distribution
over ~248k BPE tokens; Mariana predicts one of 258 bytes.  A token's first byte
is not the same as its full token, and the server does not expose the teacher's
conditional distribution after a partial token.  Sequence-level distillation
sidesteps the mismatch and is robust to any teacher.

The server does return `bytes` for each token and top-k alternatives, so a
future extension can add **first-byte soft targets at token boundaries**: group
the top-k teacher probabilities by their first byte and add a KL term at those
positions.  That is a useful auxiliary signal, not a replacement for the hard
byte targets.

## Safety and privacy

- The teacher is local; no text leaves the machine unless you point `--url` at
  a remote endpoint.
- An uncensored/abliterated teacher can generate unsafe content.  Filter the
  corpus before training, especially if the model will be shared.
- `--min-mean-logprob` is a confidence filter, not a safety filter.
- Keep API keys in `LLAMA_API_KEY`, not in shell history or committed files.
