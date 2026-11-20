"""The Hugging Face fine-tune trainer, run as the real subprocess Studio launches."""

from __future__ import annotations

import json
import re
import subprocess
import sys

import pytest
import torch

from tests.helpers import ROOT, make_tiny_transformer

pytest.importorskip("transformers")

STEP_LINE = re.compile(r"^step (\d+)/(\d+) \| total \d+\.\d{4} \| lr \d\.\d{2}e[-+]\d{2} \| tok/s \d+$")
VAL_LINE = re.compile(r"^VAL step \d+: total \d+\.\d{4} \| ppl \d+\.\d{2}$")
PARAMETERS_LINE = re.compile(
    r"^Parameters: (\d+\.\d{2})M total \| (\d+\.\d{2})M trainable \((\d+\.\d{2})%\)$"
)


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    return make_tiny_transformer(tmp_path_factory.mktemp("base") / "tiny-gpt2")


@pytest.fixture()
def corpus(tmp_path):
    path = tmp_path / "corpus.txt"
    path.write_text("The quick brown fox jumps over the lazy dog. " * 120, encoding="utf-8")
    return path


def run_trainer(tmp_path, **config):
    job = tmp_path / "job.json"
    settings = {"steps": 3, "batch": 2, "block_size": 32, "eval_every": 2, "device": "cpu",
                "warmup": 1, **config}
    job.write_text(json.dumps(settings), encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, "-m", "mariana.studio.finetune.trainer", "--config", str(job)],
        cwd=ROOT, capture_output=True, text=True, timeout=240,
    )
    return completed, completed.stdout.splitlines()


def logits(folder, tokens):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(folder).eval()
    with torch.no_grad():
        return model(tokens).logits


def test_lora_run_reports_progress_and_writes_adapter_and_merged_model(tmp_path, base, corpus):
    output = tmp_path / "models" / "tuned"
    completed, lines = run_trainer(
        tmp_path, base=str(base), output=str(output), data={"corpus": str(corpus)},
        method="lora", lr=0.05, lora={"r": 4, "alpha": 8, "dropout": 0.0},
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    steps = [line for line in lines if STEP_LINE.match(line)]
    assert [STEP_LINE.match(line).group(1) for line in steps] == ["1", "2", "3"]
    assert sum(bool(VAL_LINE.match(line)) for line in lines) == 2  # step 2 and the final step
    parameters = next(PARAMETERS_LINE.match(line) for line in lines if PARAMETERS_LINE.match(line))
    assert 0 < float(parameters.group(3)) < 50
    assert lines[-1] == f"Saved model to {output}"
    for name in ("config.json", "model.safetensors", "tokenizer.json", "mariana-finetune.json",
                 "adapter/adapter_config.json", "adapter/adapter_model.safetensors"):
        assert (output / name).is_file(), name
    metadata = json.loads((output / "mariana-finetune.json").read_text())
    assert metadata["method"] == "lora" and metadata["steps_completed"] == 3
    assert metadata["paused"] is False and metadata["data"]["text_blocks"] > 0
    assert metadata["lora"]["r"] == 4 and metadata["lora"]["target_modules"] > 0
    assert metadata["val_loss"] is not None and metadata["base"] == str(base)

    from transformers import AutoModelForCausalLM

    from mariana.studio.finetune.lora import load_adapter

    tokens = torch.randint(0, 200, (1, 12))
    merged = logits(output, tokens)
    assert not torch.allclose(merged, logits(base, tokens), atol=1e-4)  # training changed it
    adapted = AutoModelForCausalLM.from_pretrained(base).eval()
    load_adapter(adapted, output / "adapter")
    with torch.no_grad():
        assert torch.allclose(adapted(tokens).logits, merged, atol=1e-4)
    assert not list(output.parent.glob(".tuned.*"))  # no staging folder is left behind


def test_full_and_last_layer_methods(tmp_path, base, corpus):
    full = tmp_path / "full"
    completed, lines = run_trainer(
        tmp_path, base=str(base), output=str(full), data={"corpus": str(corpus)},
        method="full", steps=2, lr=0.001,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert any(PARAMETERS_LINE.match(line).group(3) == "100.00" for line in lines
               if PARAMETERS_LINE.match(line))
    assert (full / "model.safetensors").is_file() and not (full / "adapter").exists()

    partial = tmp_path / "partial"
    completed, lines = run_trainer(
        tmp_path, base=str(base), output=str(partial), data={"corpus": str(corpus)},
        method="last_layers", trainable_layers=1, steps=2, lr=0.001,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    share = float(next(PARAMETERS_LINE.match(line) for line in lines
                       if PARAMETERS_LINE.match(line)).group(3))
    assert 0 < share < 100
    assert "Training the last 1 transformer block(s)" in lines
    assert json.loads((partial / "mariana-finetune.json").read_text())["method"] == "last_layers"


def test_chat_data_and_pause(tmp_path, base):
    chat = tmp_path / "chat.jsonl"
    rows = [{"messages": [{"role": "user", "content": f"Say {word}"},
                          {"role": "assistant", "content": f"{word}!"}]}
            for word in ("fox", "dog", "quick", "lazy", "brown", "jumps")]
    chat.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    pause = tmp_path / "pause"
    pause.write_text("pause\n")
    output = tmp_path / "chat-tuned"
    completed, lines = run_trainer(
        tmp_path, base=str(base), output=str(output), data={"chat_files": [str(chat)]},
        method="lora", steps=50, merge=False, pause_file=str(pause),
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert any(line.startswith("Data: 6 examples") and "6 chat | 0 text blocks" in line for line in lines)
    assert "Paused at step 1" in lines and lines[-1] == f"Saved model to {output}"
    metadata = json.loads((output / "mariana-finetune.json").read_text())
    assert metadata["paused"] is True and metadata["steps_completed"] == 1
    assert metadata["data"]["chat_examples"] == 6
    assert (output / "adapter" / "adapter_model.safetensors").is_file()
    assert not (output / "model.safetensors").exists()  # merge was turned off


def test_expected_failures_print_one_actionable_error(tmp_path, base, corpus):
    empty = tmp_path / "empty.txt"
    empty.write_text("", encoding="utf-8")
    completed, lines = run_trainer(
        tmp_path, base=str(base), output=str(tmp_path / "out"), data={"corpus": str(empty)},
    )
    assert completed.returncode == 1
    assert lines[-1].startswith("ERROR: No training examples were found")
    assert "Traceback" not in completed.stdout + completed.stderr

    completed, lines = run_trainer(
        tmp_path, base=str(base), output=str(base / "inside"), data={"corpus": str(corpus)},
    )
    assert completed.returncode == 1 and "inside it" in lines[-1]

    completed, lines = run_trainer(
        tmp_path, base=str(tmp_path / "no-such-model"), output=str(tmp_path / "out"),
        data={"corpus": str(corpus)},
    )
    assert completed.returncode == 1 and lines[-1].startswith("ERROR: Cannot load")

    completed, lines = run_trainer(
        tmp_path, base=str(base), output=str(tmp_path / "out"), data={"corpus": str(corpus)},
        method="lora", lora={"targets": ["no_such_layer"]},
    )
    assert completed.returncode == 1 and "No layer matches" in lines[-1]
    assert not (tmp_path / "out").exists()
