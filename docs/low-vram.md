# Low-VRAM MoE + MTP + latent run on an external disk

> **Related:** [training quickstart](../README_training.md) ·
> [Sprint-DRV](fast-training.md) · [SAINT](latent-language.md) ·
> [documentation index](README.md)

Following steps explain on how to train MOE+Latent on very low VRAM GPU, example: 8 GB VRAM 
GPU. Stop the local llama.cpp teacher first if it shares the GPU (a 35B Q4
teacher can use ~6.4 GB). Set `DATA_DIR` to a directory on an external drive
with enough free space; everything stays there.

```bash
DATA_DIR=/mnt/mariana-data

# 1. verify the external drive is mounted, then create directories
df -h "$DATA_DIR"
mkdir -p "$DATA_DIR/trained-model"

# 2. 2 GiB English Common Corpus (shards also go to the external drive)
pip install -r requirements-data.txt
python scripts/prepare_common_corpus.py --gb 2 \
    --out "$DATA_DIR/common_corpus_en.txt" \
    --shard-dir "$DATA_DIR/parquet"

# 3. MoE + MTP + latent, effective batch 32 via grad accumulation
python -m mariana train \
    --data "$DATA_DIR/common_corpus_en.txt" \
    --checkpoint "$DATA_DIR/trained-model/mariana-moe-latent.pt" \
    --moe --latent \
    --thoughts 16 --latent-symbols 512 \
    --latent-window 32 --latent-window-dropout 0.5 \
    --latent-recall-prob 0.5 --w-latent 1.0 \
    --ctx 256 --dim 384 \
    --batch 2 --grad-accum 16 \
    --steps 100000 --eval-every 5000 \
    --no-gen --device cuda

# 4. MoE + MTP + latent, fast compile
python mariana.py train \
    --data "$DATA_DIR/common_corpus_en.txt" \
    --checkpoint "$DATA_DIR/trained-model/mariana-moe-latent.pt" \
    --moe --latent --fast --compile \
    --thoughts 16 --latent-symbols 512 \
    --latent-window 32 --latent-window-dropout 0.5 \
    --latent-recall-prob 0.5 --w-latent 1.0 \
    --ctx 256 --dim 384 \
    --batch 2 --grad-accum 16 \
    --steps 100000 --eval-every 5000 \
    --no-gen --device cuda
```

MTP is on by default; `--moe` adds top-2 of 4 experts; `--latent` enables SAINT.
It adds adaptive context encoding, continuous-slot refinement, compressed decode,
and draft cross-attention. `--compile` is skipped automatically. For the lighter
original code, add `--latent-legacy`.

**If it OOMs:** `--ctx 128 --dim 128 --batch 2 --grad-accum 16 --thoughts 8`.

**If you have more VRAM:** raise `--batch` and lower `--grad-accum` so the
effective batch stays ~32.

## Steps and resume

One epoch is `corpus_bytes / (batch * ctx * grad_accum)` steps. At 2 GiB and
an effective batch of 32 × 256 tokens, 100k steps is only ~0.4 epoch; raise
`--steps` for more. The trainer prints the one-epoch estimate and warns above
5 epochs.

```bash
# continue the same run (config comes from the checkpoint)
python -m mariana train --resume \
    --data "$DATA_DIR/common_corpus_en.txt" \
    --checkpoint "$DATA_DIR/trained-model/mariana-moe-latent.pt" \
    --steps 200000 --no-gen --device cuda

# inspect the private code / generate
python -m mariana think \
    --checkpoint "$DATA_DIR/trained-model/mariana-moe-latent.pt" \
    --prompt "The capital of France is" --gen-len 80
python -m mariana gen \
    --checkpoint "$DATA_DIR/trained-model/mariana-moe-latent.pt" \
    --prompt "Explain photosynthesis." --gen-len 200
```

The corpus is stored as uint8 (1 byte/token), so a 2 GiB corpus fits in RAM.
Checkpoints are saved before validation, so a failed eval keeps the run.
MoE checkpoints cannot be exported to GGUF/ONNX; train a dense model for
export. This is the capacity ceiling of this repository on an 8 GB GPU, not a
frontier model — more data/steps help, but the student size is the limit.
