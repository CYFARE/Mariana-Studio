# Checkpoint directory

`trained-model/` is the default location for Mariana checkpoints and prepared
corpora. It is runtime data, not source.

**A fresh checkout contains only this README.** The files below are created on
demand by training, Studio, or the distillation scripts. Everything in this
directory except this README is git-ignored; no model weights or prepared
corpora are shipped with the repository.

## Conventional layout

| Path | Created by | Meaning |
| --- | --- | --- |
| `mariana.pt` | `python -m mariana train` | Default DRV-LM checkpoint. |
| `mariana-ncm.pt` | `train --ncm` or `ncm generator` | Default NCM generator. |
| `mariana-ncm-tokenizer.pt` | `train --ncm` or `ncm tokenizer` | Default NCM tokenizer (also embedded in the generator checkpoint). |
| `teacher-corpus.txt` | `scripts/distill_teacher.py` | Optional corpus produced from a teacher model. |
| `common-corpus/` | `scripts/prepare_common_corpus.py` | Downloaded Common Corpus shards and extracted text. |
| `gui-corpus-*.txt` | Studio corpus scanner | Corpora prepared for a Studio training run. |
| `*.manifest.json` | Studio corpus tools | Optional corpus metadata. |

Names are conventions, not requirements. Any `--checkpoint` or `--data` path
works, including a directory outside the repository.

## Checkpoint contents

A DRV-LM checkpoint stores the full `Config`, model and optimizer state, AMP
state, and the current step. An NCM checkpoint additionally stores `stage`
(`"tokenizer"` or `"generator"`), format version, and the frozen tokenizer in a
generator checkpoint. Resuming uses the stored architecture rather than the
flags on the command line.

## Cleaning up

Checkpoints and prepared corpora are safe to delete. Remove individual files, or
remove the whole directory; the trainer recreates it on demand. This README is
documentation only and can also be removed if the directory is not needed.

Studio writes a backup before overwriting a checkpoint under
`backups/<model-name>/`. Those backups are also runtime data and are safe to
delete.

## Backups and exports

- Overwriting a checkpoint from Studio writes a backup under
  `backups/<model-name>/`.
- GGUF, ONNX, and safetensors exports go to `exports/` by default.
- Downloaded and fine-tuned third-party models live under `models/`; downloaded
  datasets live under `datasets/`.

These directories are all git-ignored. See the main
[README](../README.md), [`README_training.md`](../README_training.md), and
[`docs/`](../docs/) for commands and the whitepapers.
