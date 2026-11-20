"""The model gateway: catalog, model resolution, endpoints and Mariana chat."""

from __future__ import annotations

import json
import os
from unittest.mock import Mock, patch

import pytest
import torch

from mariana.studio.errors import ConflictError
from mariana.studio.models import ModelGateway, ModelError
from mariana.studio.models import gateway as gateway_module
from mariana.studio.models.catalog import safetensors_summary, transformers_entry
from mariana.studio.models.endpoints import endpoint_entry, endpoint_id
from tests.helpers import tiny_drv_checkpoint, tiny_ncm_checkpoints

OFFLINE = {"discover_local": False}


@pytest.fixture()
def gateway(tmp_path):
    instance = ModelGateway(tmp_path, settings=lambda: OFFLINE)
    yield instance
    instance.close()


def write_safetensors(path, tensors: dict[str, tuple]) -> None:
    header = json.dumps({
        name: {"dtype": "F32", "shape": list(shape), "data_offsets": [0, 0]}
        for name, shape in tensors.items()
    }).encode()
    path.write_bytes(len(header).to_bytes(8, "little") + header)


def test_catalog_lists_mariana_checkpoints_and_optional_backends(tmp_path, gateway):
    tiny_drv_checkpoint(tmp_path / "trained-model" / "tiny.pt")
    (tmp_path / "trained-model" / "fake.gguf").write_bytes(b"GGUF")
    catalog = gateway.catalog(force=True)
    drv = next(entry for entry in catalog if entry["name"] == "tiny.pt")
    assert drv["kind"] == "mariana-drv" and drv["available"] and drv["runtime"] == "mariana"
    assert drv["source"] == "workspace" and drv["id"] == "trained-model/tiny.pt"
    assert drv["detail"]["parameters"] > 0 and drv["detail"]["ctx_len"] == 32
    assert drv["capabilities"] == {"chat": True, "finetune": True, "latent": False, "export": True}
    assert drv["defaults"] == {"max_tokens": 80}
    gguf = next(entry for entry in catalog if entry["name"] == "fake.gguf")
    assert gguf["kind"] == "gguf" and gguf["capabilities"]["finetune"] is False
    assert gguf["available"] or "llama" in gguf["reason"]
    assert "quantized inference format" in gguf["finetune_reason"]


def test_transformers_folders_adapters_and_lookalikes(tmp_path, gateway):
    model = tmp_path / "models" / "tiny-gpt"
    model.mkdir(parents=True)
    (model / "config.json").write_text(json.dumps({
        "model_type": "gpt2", "n_positions": 64, "n_embd": 32, "n_layer": 2, "n_head": 2,
    }))
    (model / "tokenizer_config.json").write_text(json.dumps({"chat_template": "{{ messages }}"}))
    write_safetensors(model / "model.safetensors", {"wte.weight": (100, 32), "h.0.attn.c_attn.weight": (32, 96)})
    (model / "mariana-finetune.json").write_text(json.dumps({"base": "/base", "method": "lora"}))
    adapter = tmp_path / "models" / "adapter-only"
    (adapter / "adapter").mkdir(parents=True)
    (adapter / "adapter" / "adapter_config.json").write_text(json.dumps({
        "peft_type": "LORA", "base_model_name_or_path": str(model),
    }))
    write_safetensors(adapter / "adapter" / "adapter_model.safetensors", {"a.lora_A.weight": (4, 32)})
    other = tmp_path / "models" / "not-a-model"
    other.mkdir()
    (other / "config.json").write_text(json.dumps({"some": "application config"}))

    entries = {entry["name"]: entry for entry in gateway.catalog(force=True)}
    assert set(entries) == {"tiny-gpt", "adapter-only"}  # weights are not listed as checkpoints
    full = entries["tiny-gpt"]
    assert full["kind"] == "transformers" and full["capabilities"]["finetune"] is True
    assert full["detail"]["parameters"] == 100 * 32 + 32 * 96
    assert full["detail"]["ctx_len"] == 64 and full["detail"]["chat_template"] is True
    assert full["detail"]["finetuned_from"] == "/base" and full["detail"]["finetune"]["method"] == "lora"
    assert full["defaults"] == {"max_tokens": 1024}
    lora = entries["adapter-only"]
    assert lora["detail"]["adapter"] is True and lora["detail"]["base"] == str(model)
    assert lora["capabilities"]["finetune"] is False and "adapter" in lora["finetune_reason"]
    assert transformers_entry(tmp_path, other) is None
    assert safetensors_summary(model / "model.safetensors")["shapes"]["wte.weight"] == (100, 32)
    (adapter / "adapter" / "adapter_config.json").write_text(json.dumps({
        "peft_type": "LORA", "base_model_name_or_path": str(tmp_path / "gone"),
    }))
    orphan = transformers_entry(tmp_path, adapter)
    assert orphan["available"] is False and "base model of this adapter is missing" in orphan["reason"]


def test_models_resolve_by_id_name_and_fresh_files(tmp_path, gateway):
    assert gateway.catalog() == []
    with pytest.raises(ModelError, match="Select a model first"):
        gateway.get(None)
    tiny_drv_checkpoint(tmp_path / "trained-model" / "late.pt")  # created after the first listing
    assert gateway.get("trained-model/late.pt")["name"] == "late.pt"
    assert gateway.get("late.pt")["id"] == "trained-model/late.pt"  # unique name
    assert gateway.get(None)["name"] == "late.pt"  # the only model
    assert gateway.get(checkpoint="trained-model/late.pt")["kind"] == "mariana-drv"
    with pytest.raises(ModelError, match="refresh the library"):
        gateway.get("missing")
    with pytest.raises(ModelError, match="from the catalog"):
        gateway.get(checkpoint="trained-model/missing.pt")


def test_chat_with_a_drv_checkpoint_and_stream_events(tmp_path, gateway):
    tiny_drv_checkpoint(tmp_path / "trained-model" / "tiny.pt")
    result = gateway.chat({
        "model": "trained-model/tiny.pt", "message": "Hello", "history": [],
        "max_tokens": 2, "temperature": 0, "device": "cpu",
    })
    assert result["kind"] == "DRV-LM" and result["reply"] and result["runtime"] == "mariana"
    assert result["model"].endswith("tiny.pt") and result["model_id"] == "trained-model/tiny.pt"
    assert result["messages"][-1] == {"role": "user", "content": "Hello"}
    assert result["usage"]["completion_tokens"] == 2 and result["prompt_format"] == "transcript"
    assert result["tokens_per_second"] > 0 and "type" not in result

    events = list(gateway.stream({
        "model": "tiny.pt", "max_tokens": 2, "temperature": 0, "device": "cpu",
        "system_prompt": "Be brief.",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Hi"}, {"type": "image"}]}],
    }))
    assert events[0] == {"type": "start", "model_id": "trained-model/tiny.pt",
                         "model": "trained-model/tiny.pt", "name": "tiny.pt", "kind": "Mariana DRV-LM"}
    assert events[-1]["type"] == "done" and all(event["type"] == "delta" for event in events[1:-1])
    assert gateway.loaded() == [
        {"runtime": "mariana", "path": str(tmp_path / "trained-model" / "tiny.pt")},
    ]
    assert gateway.release_gpu()["unloaded"] == 1 and gateway.loaded() == []


def test_close_unloads_models_and_stops_every_runtime(gateway, monkeypatch):
    gateway._runtime_cache[("mariana", "cached.pt")] = object()
    calls = []
    monkeypatch.setattr(gateway.llama, "close", lambda: calls.append("llama"))
    monkeypatch.setattr(gateway.ollama, "close", lambda: calls.append("ollama"))
    monkeypatch.setattr(gateway, "_empty_device_cache", lambda: calls.append("device"))
    gateway.close()
    assert gateway.loaded() == []
    assert calls == ["llama", "ollama", "device"]


def test_chat_requests_are_validated(tmp_path, gateway):
    tiny_drv_checkpoint(tmp_path / "trained-model" / "tiny.pt")
    base = {"model": "tiny.pt", "device": "cpu"}
    for config, message in (
        ({**base, "message": "  "}, "Enter a message first"),
        ({**base, "messages": [{"role": "assistant", "content": "hi"}]}, "Enter a message first"),
        ({**base, "message": "hi", "temperature": 9}, "temperature must be between"),
        ({**base, "message": "hi", "max_tokens": "many"}, "max_tokens must be a number"),
    ):
        with pytest.raises(ModelError, match=message):
            gateway.chat(config)
    gateway.training_active = lambda: True
    with pytest.raises(ConflictError, match="training run"):
        gateway.chat({**base, "message": "hi"})


def test_only_two_local_models_stay_loaded(tmp_path, gateway):
    for name in ("a.pt", "b.pt", "c.pt"):
        tiny_drv_checkpoint(tmp_path / "trained-model" / name)
        gateway.chat({"model": name, "message": "x", "max_tokens": 1, "temperature": 0, "device": "cpu"})
    loaded = [item["path"].rsplit("/", 1)[1] for item in gateway.loaded()]
    assert loaded == ["b.pt", "c.pt"]


def test_chat_with_an_ncm_generator(tmp_path, gateway):
    tiny_ncm_checkpoints(tmp_path / "trained-model", "ncm")
    catalog = {entry["name"]: entry for entry in gateway.catalog(force=True)}
    assert catalog["ncm.pt"]["kind"] == "mariana-ncm" and catalog["ncm.pt"]["capabilities"]["ncm"]
    assert catalog["ncm.pt"]["detail"]["tokenizer"] == "trained-model/ncm-tokenizer.pt"
    tokenizer = catalog["ncm-tokenizer.pt"]
    assert tokenizer["kind"] == "mariana-ncm-tokenizer" and tokenizer["available"] is False
    result = gateway.chat({
        "model": catalog["ncm.pt"]["id"], "message": "Hello", "history": [],
        "max_tokens": 2, "temperature": 0, "device": "cpu",
    })
    assert result["kind"] == "NCM" and result["reply"] and result["ncm"]["levels"] == 5
    with pytest.raises(ValueError) as refused:
        gateway.chat({"checkpoint": "trained-model/ncm-tokenizer.pt", "message": "Hello",
                      "max_tokens": 1, "device": "cpu"})
    assert "tokenizer-only" in str(refused.value) and "generator checkpoint" in str(refused.value)
    assert "state_dict" not in str(refused.value)


def test_saint_chat_strips_only_the_canonical_prompt(tmp_path, gateway):
    from mariana.config import Config
    from mariana.generate import _canonical_prompt

    cfg = Config(
        ctx_len=16, dim=32, n_heads=4, n_pre=1, n_core=1, n_post=1, max_loops=1, block_size=4,
        mlp_mult=2, use_latent=True, latent_saint=True, n_latent_symbols=8, n_thoughts=4,
        latent_min_thoughts=1, latent_local_window=4, latent_continuous_slots=1,
        latent_refine_steps=0,
    )
    tiny_drv_checkpoint(tmp_path / "trained-model" / "saint.pt", **cfg.to_dict())

    def fake_generate(model, raw, max_new_tokens, **kwargs):
        prompt = _canonical_prompt(raw, cfg, max_new_tokens)
        assert len(prompt) == 4
        return prompt + b"xy", 0.0, 0.01

    with patch("mariana.generate.ar_generate", side_effect=fake_generate):
        result = gateway.chat({
            "checkpoint": "trained-model/saint.pt",
            "message": "This is a deliberately long prompt", "history": [],
            "max_tokens": 2, "temperature": 0, "device": "cpu",
        })
    assert result["reply"] == "xy"


def test_ncm_chat_forwards_options_and_selects_the_mbr_consensus(tmp_path, gateway):
    from mariana.ncm import NCMConfig

    cfg = NCMConfig(
        ctx_len=8, prefix_len=4, levels=(1, 2, 4, 8), latent_dim=16, dim=32, n_heads=4,
        codebook_size=16, encoder_layers=1, decoder_layers=1, refiner_layers=1, generator_layers=1,
    )
    model = Mock()
    seen = {}

    def generate(prompt, **kwargs):
        seen["batch"] = prompt.shape[0]
        seen.update(kwargs)
        tokens = torch.tensor([[65, 66, 67, 68], [65, 66, 67, 68], [90, 90, 90, 90]])
        return tokens, {"elapsed": 0.01, "levels_used": 2, "generator_nfe": 5,
                        "decoder_nfe": 1, "refiner_nfe": 2}

    model.generate.side_effect = generate
    path = tmp_path / "trained-model" / "ncm.pt"
    path.parent.mkdir()
    path.touch()
    entry = {
        "id": "trained-model/ncm.pt", "name": "ncm.pt", "kind": "mariana-ncm",
        "kind_label": "Mariana NCM", "runtime": "mariana", "path": "trained-model/ncm.pt",
        "absolute_path": str(path), "available": True, "capabilities": {"chat": True}, "detail": {},
    }
    config = {
        "model": entry["id"], "message": "Hello", "history": [], "max_tokens": 4,
        "temperature": 0.8, "top_p": 0.9, "top_k": 7, "guidance": 0.6,
        "guidance_mode": "constant", "early_exit": 0.9, "refine_steps": 2,
        "refine_threshold": 1.1, "candidates": 3, "device": "cpu",
    }
    with (
        patch.object(gateway, "get", return_value=entry),
        patch("mariana.studio.models.backends.mariana.load_model", return_value=("NCM", cfg, model)),
    ):
        result = gateway.chat(config)
        assert seen["batch"] == 3 and seen["top_k"] == 7 and seen["top_p"] == 0.9
        assert seen["guidance"] == 0.6 and seen["guidance_ramp"] is False
        assert seen["early_exit"] == 0.9 and seen["use_cache"] is True
        assert seen["refine_steps"] == 2 and seen["refine_threshold"] == 1.1
        assert result["reply"] == "ABCD" and result["ncm"]["candidates"] == 3
        assert result["ncm"]["generator_nfe"] == 15
        assert result["ncm"]["levels_used"] == 2 and result["ncm"]["levels"] == 4
        model.generate.reset_mock()
        with pytest.raises(ValueError, match="ramp or constant"):
            gateway.chat({**config, "guidance_mode": "invalid"})
        model.generate.assert_not_called()


def test_endpoint_secrets_are_write_only_and_ids_are_stable(tmp_path, gateway):
    gateway.add_endpoint({
        "name": "Example", "kind": "openai", "base_url": "https://example.com/v1/",
        "model": "example-model", "api_key": "secret-value", "context_length": "8192",
    })
    (public,) = gateway.endpoints.public()
    assert public["id"] == "endpoint:example" and public["has_api_key"] is True
    assert "api_key" not in public and public["base_url"] == "https://example.com/v1"
    assert public["context_length"] == 8192
    assert "secret-value" not in json.dumps(gateway.catalog(force=True))
    assert oct(os.stat(gateway.endpoints.path).st_mode & 0o777) == "0o600"
    # Updating through the id keeps the id and, without a new key, the stored key.
    gateway.add_endpoint({"id": public["id"], "name": "Renamed", "kind": "openai",
                          "base_url": "https://example.com/v1", "model": "other-model"})
    (updated,) = gateway.endpoints.load()
    assert updated["id"] == "endpoint:example" and updated["name"] == "Renamed"
    assert updated["api_key"] == "secret-value" and updated["model"] == "other-model"
    gateway.add_endpoint({"id": public["id"], "name": "Renamed", "kind": "openai",
                          "base_url": "https://example.com/v1", "model": "m", "clear_api_key": True})
    assert "api_key" not in gateway.endpoints.load()[0]
    assert endpoint_id("endpoint:example") == "endpoint:example"
    assert endpoint_id("My Server #2") == "endpoint:my-server-2"
    gateway.delete_endpoint(public["id"])
    assert gateway.endpoints.public() == []
    with pytest.raises(ModelError, match="Endpoint not found"):
        gateway.delete_endpoint(public["id"])
    for bad, message in (
        ({"kind": "openai", "base_url": "https://x/v1"}, "Give the endpoint a name"),
        ({"name": "x", "kind": "smtp", "base_url": "https://x/v1", "model": "m"}, "kind must be"),
        ({"name": "x", "kind": "openai", "base_url": "ftp://x", "model": "m"}, "full http"),
        ({"name": "x", "kind": "openai", "base_url": "https://x/v1"}, "Model name is required"),
    ):
        with pytest.raises(ModelError, match=message):
            gateway.add_endpoint(bad)


def test_endpoint_entries_describe_what_the_backend_supports(monkeypatch):
    monkeypatch.delenv("EXAMPLE_KEY", raising=False)
    local = endpoint_entry({"id": "endpoint:local", "name": "Local", "kind": "openai",
                            "base_url": "http://127.0.0.1:8080/v1", "model": "m"})
    assert local["available"] and local["capabilities"]["remote"] is False
    keyed = endpoint_entry({"id": "endpoint:cloud", "name": "Cloud", "kind": "openai",
                            "base_url": "https://api.example.com/v1", "model": "m",
                            "api_key_env": "EXAMPLE_KEY"})
    assert keyed["available"] is False and "EXAMPLE_KEY" in keyed["reason"]
    monkeypatch.setenv("EXAMPLE_KEY", "value")
    assert endpoint_entry({**keyed, "kind": "openai", "base_url": "https://api.example.com/v1",
                           "model": "m", "api_key_env": "EXAMPLE_KEY"})["available"] is True
    claude = endpoint_entry({"id": "endpoint:claude", "name": "Claude", "kind": "anthropic",
                             "base_url": "", "model": "claude-opus-5-5", "max_output_tokens": 128000})
    assert claude["capabilities"]["sampling"] is False and claude["capabilities"]["effort"] is True
    assert claude["detail"]["base_url"] == "https://api.anthropic.com"
    assert claude["detail"]["refusal_fallback"] is True


def test_provider_presets_and_default_token_limits(gateway):
    presets = {preset["id"]: preset for preset in gateway.providers()}
    assert presets["anthropic"]["kind"] == "anthropic"
    assert presets["anthropic"]["default_model"] == "claude-opus-5-5"
    assert presets["ollama"]["local"] is True and presets["openai"]["key_env"] == "OPENAI_API_KEY"
    assert all(preset["kind"] in {"openai", "ollama", "anthropic"} for preset in presets.values())
    assert gateway._default_max_tokens({"runtime": "anthropic", "detail": {}}) == 16000
    assert gateway._default_max_tokens({"runtime": "anthropic", "detail": {"max_output_tokens": 4096}}) == 4096
    assert gateway._default_max_tokens({"runtime": "ollama", "detail": {}}) == gateway_module.FALLBACK_MAX_TOKENS
    assert gateway.output_limit() == gateway_module.FALLBACK_MAX_TOKENS
