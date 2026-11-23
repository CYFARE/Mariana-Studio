# Low-VRAM fine-tuning

Studio's Hugging Face trainer can adapt a large model on a 6-8 GB GPU with
**4-bit QLoRA**. The small-card path is a preset rather than a separate engine:
it chooses the same native LoRA adapter format, just with a quantized frozen
base and a memory-aware training loop.

## Quick path

1. **Models → Discover** or a local folder provides the base model.
2. Open **Fine-tune**, select the base, and choose **Low-VRAM LoRA (4-bit)**.
3. Keep **Global batch** at 1 and **Gradient accumulation** at 8 unless you know
   the card has headroom. The preset already does this.
4. Review the plan. It shows the estimated peak, the 4-bit peak when the
   uncompressed model does not fit, and any missing backend.
5. Start. The trainer reports `Memory: ...` decisions in the job log, then the
   usual `step n/total` and `VAL step` lines.

The output folder contains `adapter/` and `mariana-finetune.json`. With **Also
save a merged model** off (the preset default), Studio can still chat with the
adapter: the model catalog and chat backend load the base plus `adapter/`.

## Install the quantized backend

```bash
pip install -r requirements-finetune.txt
```

This adds `bitsandbytes` (CUDA platforms) and `accelerate`. Without
bitsandbytes, planning still works and explains the missing package, but a
4-bit run cannot start. On a GPU without bf16 (many 6-8 GB cards), the trainer
uses **fp16** with loss scaling automatically. CPU and macOS MPS runs use fp32
and are intended for small bases; quantization is CUDA-only.

## What the low-VRAM path changes

| Change | Why it matters |
| --- | --- |
| NF4 4-bit frozen weights with double quantization | A 7B base drops from ~14 GB (bf16) to ~3.9 GB. |
| Native LoRA updates in float32 | Adapter math stays accurate even when the base is 4-bit. |
| bf16, or fp16 with `GradScaler` | Older CUDA cards get half-precision weights without underflow. |
| Gradient checkpointing | Recomputes activations instead of storing them; the preset enables it. |
| Auto memory tuning | Before loading, the trainer estimates the footprint and may switch to 4-bit, enable checkpointing, or halve the micro-batch. The effective batch is preserved with accumulation. |
| Chunked cross-entropy | Full logits for a 7B model can cost ~300 MB per batch; the loss is summed in 256-token slices. |
| Epoch batches with length grouping | Every example is seen before an epoch repeats, and batches pad less. |
| Compact token store | Corpus examples live in two int32 tensors instead of one Python int per token. |
| Streaming data caps | `max_tokens` (20M) and `max_examples` (500k) bound RAM; raise them in **Advanced** for a large dataset. |
| OOM recovery | On an allocator failure the trainer empties the cache, halves the micro-batch (doubling accumulation), and restarts the step; at batch 1 it enables checkpointing. |
| Validation retry | A validation batch that does not fit is halved or skipped with a warning instead of failing the run. |
| CPU merge | A requested merged model is built on the CPU after training, so it does not need a second copy in VRAM. If RAM is insufficient, the adapter is still saved and the job completes. |

## Expected footprint (4-bit LoRA, batch 1, 512-token blocks)

| Base | Weights | Estimated peak | 6 GB | 8 GB |
| --- | --- | --- | --- | --- |
| 1.5B | ~0.8 GB | ~2.1 GB | yes | yes |
| 3B | ~1.7 GB | ~3.2 GB | yes | yes |
| 7B | ~3.9 GB | ~5.2 GB | tight; use 256-token blocks | yes |
| 13B | ~7.2 GB | ~8.6 GB | no | no |

These are planning estimates, not promises: sequence length, vocabulary,
trainable layer count, allocator fragmentation and other GPU applications all
move the real number. The trainer logs its estimate and reacts to an actual
out-of-memory error.

## Full and final-layer training

Full fine-tuning still keeps fp32 weights and AdamW state (about 16 bytes per
parameter before activations), so a 7B model needs far more than 8 GB. The plan
says so and recommends LoRA + 4-bit. Final-layer training also keeps the fp32
base; it is useful for small and medium models, not for fitting a large base on
a small card.

## Large datasets

- Chat files are read row by row; the trainer never builds a list of every row
  before tokenizing.
- Text is tokenized about 1 MB at a time and packed into blocks.
- Both chat and text share the token and example caps. When a cap is reached,
  the log says so and the caps can be raised in **Advanced**.
- JSON Lines is the cheapest format for a large corpus. A very large `.json`
  array that exceeds the 256 MB whole-document limit falls back to line-based
  parsing; convert it to JSONL for reliable streaming.

## Troubleshooting

| Message | Fix |
| --- | --- |
| `Quantized fine-tuning needs bitsandbytes` | `pip install -r requirements-finetune.txt` |
| `Could not place the quantized model on the GPU` | Install `accelerate`, or use a smaller base. |
| `out of memory; micro-batch N -> M` | Normal recovery. The run continues with more accumulation. |
| `The GPU ran out of memory even at batch 1` | Shorten **Context tokens**, use a smaller base, or close other GPU work. |
| `could not merge the adapter into a full model` | The adapter is saved and usable. Free CPU RAM or turn off **Also save a merged model**. |
| `validation skipped because the GPU is out of memory` | Training continues. Lower the batch or context for validation headroom. |

## Verifying the implementation

```bash
python -m pytest -q tests/test_studio_finetune_memory.py \
  tests/test_studio_finetune_dataset.py tests/test_studio_finetune.py \
  tests/test_studio_finetune_trainer.py
```

The memory tests cover 6 GB and 8 GB decisions, the dataset tests cover the
compact store and epoch coverage, and the trainer tests run real CPU fine-tunes
plus a chunked-loss equivalence check.
