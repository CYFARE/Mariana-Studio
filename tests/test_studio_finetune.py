"""Fine-tune planning and execution for Mariana checkpoints and any Hugging Face model."""

from __future__ import annotations

import json

import pytest
import torch

from mariana.studio.finetune import FinetuneService, discovery, hf, native
from mariana.studio.jobs import JobRegistry
from mariana.studio.models import ModelError, ModelGateway
from mariana.studio.training import TrainingManager
from tests.helpers import make_tiny_transformer, tiny_drv_checkpoint, tiny_ncm_checkpoints, wait_for

OFFLINE = {"discover_local": False}


@pytest.fixture()
def service(tmp_path):
    gateway = ModelGateway(tmp_path, settings=lambda: OFFLINE)
    yield FinetuneService(tmp_path, gateway, TrainingManager(tmp_path))
    gateway.close()


@pytest.fixture()
def corpus(tmp_path):
    folder = tmp_path / "corpus"
    folder.mkdir()
    (folder / "notes.txt").write_text("Fine tuning corpus. " * 160)
    return folder


def write_safetensors(path, tensors: dict[str, tuple]) -> None:
    header = json.dumps({
        name: {"dtype": "F32", "shape": list(shape), "data_offsets": [0, 0]}
        for name, shape in tensors.items()
    }).encode()
    path.write_bytes(len(header).to_bytes(8, "little") + header)


def fake_transformer(folder, chat_template=False):
    """A Transformers directory described only by config and safetensors headers."""
    folder.mkdir(parents=True)
    (folder / "config.json").write_text(json.dumps({
        "model_type": "gpt2", "n_positions": 128, "n_embd": 16, "n_layer": 3, "n_head": 2}))
    if chat_template:
        (folder / "tokenizer_config.json").write_text(json.dumps({"chat_template": "{{ messages }}"}))
    tensors = {"wte.weight": (100, 16), "ln_f.weight": (16,)}
    for index in range(3):
        tensors[f"h.{index}.attn.c_attn.weight"] = (16, 48)
        tensors[f"h.{index}.attn.c_proj.weight"] = (16, 16)
        tensors[f"h.{index}.mlp.c_fc.weight"] = (16, 64)
        tensors[f"h.{index}.mlp.c_proj.weight"] = (64, 16)
    write_safetensors(folder / "model.safetensors", tensors)
    return folder


def test_native_plans_for_drv_and_ncm_checkpoints(tmp_path, service, corpus):
    tiny_drv_checkpoint(tmp_path / "trained-model" / "base.pt")
    info = service.inspect("trained-model/base.pt")
    assert info["engine"] == "mariana" and info["architecture"] == "DRV-LM" and info["layers"] == 3
    assert native.inspect(tmp_path, "trained-model/base.pt")["name"] == "base.pt"
    plan = service.plan({
        "base": "trained-model/base.pt", "folder": str(corpus),
        "preset": "quick", "freeze": "last_block", "name": "tuned.pt",
    })
    assert plan["engine"] == "mariana" and plan["config"]["mode"] == "finetune"
    assert plan["freeze"] == {"scope": "last_block", "layers": 2, "embeddings": True, "total_layers": 3}
    assert 0 < plan["estimate"]["trainable"] < plan["estimate"]["parameters"]
    assert ["Scope", "Last block · 2/3 layers frozen"] in plan["summary"]
    assert plan["output"] == "trained-model/tuned.pt"
    with pytest.raises(ModelError, match="new output name"):
        service.plan({"base": "trained-model/base.pt", "folder": str(corpus), "name": "base.pt"})
    for blank in ("", "   ", None):  # never fall back to the current directory
        with pytest.raises(ModelError, match="valid fine-tuning corpus folder"):
            service.plan({"base": "trained-model/base.pt", "folder": blank})

    tiny_ncm_checkpoints(tmp_path / "trained-model", "ncm-base")
    ncm = service.inspect("trained-model/ncm-base.pt")
    assert ncm["technique"] == "ncm" and ncm["tokenizer"]
    ncm_plan = service.plan({"base": "trained-model/ncm-base.pt", "folder": str(corpus),
                             "preset": "balanced", "name": "ncm-tuned.pt"})
    assert ncm_plan["config"]["technique"] == "ncm" and ncm_plan["config"]["base"].endswith("ncm-base.pt")


def test_every_model_is_listed_with_why_it_can_or_cannot_be_tuned(tmp_path, service):
    tiny_drv_checkpoint(tmp_path / "trained-model" / "base.pt")
    tiny_ncm_checkpoints(tmp_path / "trained-model", "ncm")
    fake_transformer(tmp_path / "models" / "third-party")
    (tmp_path / "models" / "quantized-Q4_0.gguf").write_bytes(b"gguf")
    service.gateway.add_endpoint({"name": "Cloud", "kind": "openai",
                                  "base_url": "https://api.example.com/v1", "model": "m"})
    bases = service.bases(force=True)
    tunable = {item["name"]: item for item in bases["tunable"]}
    assert set(tunable) == {"base.pt", "ncm.pt", "third-party"}
    assert tunable["third-party"]["engine"] == "hf" and tunable["base.pt"]["engine"] == "mariana"
    other = {item["name"]: item["reason"] for item in bases["other"]}
    assert set(other) == {"quantized-Q4_0.gguf", "Cloud"}  # the NCM tokenizer is not offered at all
    assert "quantized inference format" in other["quantized-Q4_0.gguf"]
    assert "remote server" in other["Cloud"]
    with pytest.raises(ModelError, match="quantized inference format"):
        service.inspect("models/quantized-Q4_0.gguf")
    with pytest.raises(ModelError, match="Select a base model"):
        service.plan({"folder": "x"})


def test_hugging_face_plans_estimate_what_trains(tmp_path, service, corpus):
    fake_transformer(tmp_path / "models" / "third-party")
    info = service.inspect("models/third-party")
    assert info["engine"] == "hf" and info["layers"] == 3 and info["ctx_len"] == 128
    assert info["default_name"] == "third-party-tuned" and info["chat_template"] is False
    assert set(info["presets"]) == {"quick", "balanced", "deep"}

    plan = service.plan({"base": "models/third-party", "folder": str(corpus), "preset": "quick"})
    assert plan["engine"] == "hf" and plan["config"]["engine"] == "hf"
    assert plan["config"]["output"] == str(tmp_path / "models" / "third-party-tuned")
    assert plan["config"]["data_mode"] == "text" and plan["config"]["chat_files"] == []
    trainer = plan["config"]["trainer"]
    assert trainer["method"] == "lora" and trainer["lora"] == {
        "r": 8, "alpha": 16.0, "dropout": 0.05, "targets": "auto"}
    assert trainer["steps"] == 100 and trainer["block_size"] == 128 and trainer["merge"] is True
    total = 100 * 16 + 16 + 3 * (16 * 48 + 16 * 16 + 16 * 64 + 64 * 16)
    attention = 3 * (8 * (16 + 48) + 8 * (16 + 16))  # c_attn and the attention c_proj
    assert plan["estimate"]["parameters"] == total and plan["estimate"]["trainable"] == attention
    assert ["Method", "LoRA adapter"] in plan["summary"] and plan["output"] == "models/third-party-tuned"

    full = service.plan({"base": "models/third-party", "folder": str(corpus), "preset": "deep", "lr": 0.001})
    assert full["estimate"]["trainable"] == total and full["config"]["trainer"]["method"] == "full"
    assert any("forgets" in warning for warning in full["warnings"])
    last = service.plan({"base": "models/third-party", "folder": str(corpus), "method": "last_layers",
                         "trainable_layers": 1})
    assert last["estimate"]["trainable"] == 16 * 48 + 16 * 16 + 16 * 64 + 64 * 16 + 16
    wide = service.plan({"base": "models/third-party", "folder": str(corpus), "lora_targets": "all-linear",
                         "lora_r": 4, "block_size": 512, "name": "wide.pt"})
    assert wide["estimate"]["trainable"] > 3 * (4 * 64 + 4 * 32)
    assert wide["config"]["name"] == "wide"  # a stray .pt suffix is not a folder name
    assert any("will be reduced" in warning for warning in wide["warnings"])
    assert hf.estimate_trainable(tmp_path / "missing", "lora", "auto", 8, 1) == (0, 0)


def test_hugging_face_plans_detect_chat_data_and_validate(tmp_path, service, corpus):
    fake_transformer(tmp_path / "models" / "third-party")
    rows = [{"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}] * 4
    (corpus / "chat.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    base = {"base": "models/third-party", "folder": str(corpus)}
    auto = service.plan(base)
    assert auto["data"] == {"mode": "chat", "chat_files": 1}
    assert auto["config"]["chat_files"] == [str(corpus / "chat.jsonl")]
    assert any("no chat template" in warning for warning in auto["warnings"])
    assert service.plan({**base, "data_mode": "text"})["data"]["mode"] == "text"
    (corpus / "chat.jsonl").unlink()
    for config, message in (
        ({**base, "data_mode": "chat"}, "No chat-format data found"),
        ({**base, "folder": str(tmp_path / "missing")}, "valid fine-tuning data folder"),
        ({**base, "name": "bad name"}, "letters, numbers"),
        ({**base, "name": "third-party"}, "new output name"),
        ({**base, "method": "magic"}, "Method must be one of"),
        ({**base, "preset": "nope"}, "Unknown fine-tune preset"),
        ({**base, "lora_r": 0}, "lora_r must be between"),
    ):
        with pytest.raises(ValueError, match=message):
            service.plan(config)
    (tmp_path / "models" / "taken").mkdir()
    with pytest.raises(ModelError, match="enable backups"):
        service.plan({**base, "name": "taken", "backup": False})
    assert any("moved to backups" in warning for warning in service.plan({**base, "name": "taken"})["warnings"])


@pytest.mark.slow
def test_one_step_drv_finetune_runs_from_base_weights(tmp_path, service, corpus):
    base = tiny_drv_checkpoint(tmp_path / "trained-model" / "base.pt", step=3)
    job = service.start({
        "base": "trained-model/base.pt", "folder": str(corpus), "preset": "quick",
        "freeze": "last_block", "name": "tuned.pt", "steps": 1, "batch": 1, "grad_accum": 1,
        "warmup": 0, "eval_every": 1, "device": "cpu",
    })
    assert job["finetune"] is True and job["title"] == "Fine-tune · base.pt"
    job = wait_for(lambda: (lambda state: state if not state["active"] else None)(service.training.state()),
                   timeout=120)
    assert job["status"] == "completed", "\n".join(job["logs"][-20:])
    assert any("Fine-tune scope" in line for line in job["logs"])
    assert torch.load(base, map_location="cpu", weights_only=True)["step"] == 3
    tuned = torch.load(tmp_path / "trained-model" / "tuned.pt", map_location="cpu", weights_only=True)
    assert tuned["step"] == 1


@pytest.mark.slow
def test_a_hugging_face_model_is_fine_tuned_end_to_end_and_then_chats(tmp_path, service, corpus):
    pytest.importorskip("transformers")
    base = make_tiny_transformer(tmp_path / "models" / "tiny-gpt2")
    plan = service.plan({"base": "models/tiny-gpt2", "folder": str(corpus), "preset": "quick"})
    assert plan["ready"] is True and plan["estimate"]["trainable"] > 0
    job = service.start({
        "base": "models/tiny-gpt2", "folder": str(corpus), "preset": "quick", "name": "tiny-tuned",
        "steps": 2, "batch": 2, "block_size": 32, "eval_every": 1, "device": "cpu", "lr": 0.01,
        "lora_r": 4,
    })
    assert job["technique"] == "hf" and job["finetune"] is True
    assert job["title"] == "Fine-tune · tiny-gpt2" and job["checkpoint"] == "models/tiny-tuned"
    job = wait_for(lambda: (lambda state: state if not state["active"] else None)(service.training.state()),
                   timeout=180)
    assert job["status"] == "completed", "\n".join(job["logs"][-25:])
    assert job["current_step"] == 2 and job["total_steps"] == 2 and job["checkpoint_ready"] is True
    assert job["metrics"]["points"][-1][1] == 2 and job["metrics"]["validation"]
    output = tmp_path / "models" / "tiny-tuned"
    assert (output / "model.safetensors").is_file() and (output / "adapter" / "adapter_config.json").is_file()
    assert (base / "model.safetensors").is_file()  # the base is untouched

    entry = service.gateway.get("models/tiny-tuned")
    assert entry["kind"] == "transformers" and entry["detail"]["finetuned_from"] == str(base)
    assert entry["capabilities"]["finetune"] is True  # a merged result can be tuned again
    reply = service.gateway.chat({"model": entry["id"], "message": "Hello", "max_tokens": 3,
                                  "temperature": 0, "device": "cpu"})
    assert reply["runtime"] == "transformers" and reply["usage"]["completion_tokens"] <= 3

    again = service.start({
        "base": "models/tiny-gpt2", "folder": str(corpus), "name": "tiny-tuned", "steps": 1,
        "batch": 2, "block_size": 32, "device": "cpu", "method": "last_layers", "trainable_layers": 1,
    })
    assert again["backup"] and (tmp_path / again["backup"] / "tiny-tuned" / "model.safetensors").is_file()
    done = wait_for(lambda: (lambda state: state if not state["active"] else None)(service.training.state()),
                    timeout=180)
    assert done["status"] == "completed", "\n".join(done["logs"][-25:])
    assert not (output / "adapter").exists()  # the new run replaced the old output


def test_discovery_marks_saved_datasets_and_local_bases(tmp_path, service, monkeypatch):
    fake_transformer(tmp_path / "models" / "org--model")
    folder = tmp_path / "datasets" / "org--data"
    folder.mkdir(parents=True)
    (folder / "data.jsonl").write_text(
        '{"messages":[{"role":"user","content":"q"},{"role":"assistant","content":"a"}]}\n'
    )
    (folder / "dataset.json").write_text(json.dumps({
        "repo": "org/data", "config": "default", "split": "train", "rows": 1, "bytes": 76,
    }))

    def search(platform, query, limit):
        return [
            {"platform": "huggingface", "kind": "dataset", "id": "org/data", "repo": "org/data",
             "title": "org/data", "summary": "", "url": "https://huggingface.co/datasets/org/data",
             "tags": [], "downloads": 1, "likes": 0, "updated": "", "gated": False},
            {"platform": "huggingface", "kind": "model", "id": "org/model", "repo": "org/model",
             "title": "org/model", "summary": "", "url": "https://huggingface.co/org/model",
             "tags": [], "downloads": 1, "likes": 0, "updated": "", "gated": False},
            {"platform": "openml", "kind": "dataset", "id": "openml:1", "repo": "org/data",
             "title": "org/data", "summary": "OpenML data", "url": "https://www.openml.org/d/1",
             "tags": [], "downloads": 0, "likes": 0, "updated": "", "gated": False},
        ], []

    monkeypatch.setattr(discovery, "search", search)
    registry = JobRegistry()
    try:
        result = FinetuneService(tmp_path, service.gateway, service.training, registry).discover(
            "all", "org", limit=10)
    finally:
        registry.close()
    dataset, model, openml = result["results"]
    assert dataset["local_folder"] == str(folder.resolve()) and dataset["rows"] == 1
    assert "local_folder" not in openml  # an OpenML name is not a Hub repository
    assert model["local_id"] == "models/org--model" and model["local"] is True
    assert model["finetune"] is True and isinstance(model["reason"], str)
    assert result["installed"][0]["repo"] == "org/data"
    assert result["featured"]["datasets"] and result["platforms"][0]["id"] == "datasets"


def test_starting_without_transformers_names_the_fix(tmp_path, service, corpus, monkeypatch):
    fake_transformer(tmp_path / "models" / "third-party")
    monkeypatch.setattr(hf, "has_module", lambda name: False)
    plan = service.plan({"base": "models/third-party", "folder": str(corpus)})
    assert plan["ready"] is False
    assert any("requirements-finetune.txt" in warning for warning in plan["warnings"])
    with pytest.raises(ModelError, match="pip install -r requirements-finetune.txt"):
        service.start({"base": "models/third-party", "folder": str(corpus)})
    assert service.training.state() is None
