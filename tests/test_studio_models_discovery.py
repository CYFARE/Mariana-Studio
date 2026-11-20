"""Discovery of models installed by other tools: Ollama, Hugging Face cache, local servers."""

from __future__ import annotations

import json

import pytest

from mariana.studio.models import catalog, discovery
from mariana.studio.models.discovery import Discovery
from tests.helpers import mock_provider, serving, tiny_drv_checkpoint


@pytest.fixture()
def isolated(monkeypatch, tmp_path):
    """Point every discovery source at empty temporary locations."""
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hf"))
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.setenv("OLLAMA_MODELS", str(tmp_path / "ollama"))
    monkeypatch.setenv("OLLAMA_HOST", "127.0.0.1:9")  # nothing listens on the discard port
    monkeypatch.delenv("MARIANA_LLAMA_SERVER", raising=False)
    monkeypatch.delenv("LLAMA_CPP_DIR", raising=False)
    monkeypatch.setattr(discovery, "ollama_model_dirs", lambda: [tmp_path / "ollama"])
    monkeypatch.setattr(discovery, "hf_cache_dirs", lambda: [tmp_path / "hf"])
    monkeypatch.setattr(discovery, "GGUF_FOLDERS", [])
    monkeypatch.setattr(discovery, "LOCAL_SERVERS", [])
    monkeypatch.setattr(discovery, "LLAMA_SERVER_CANDIDATES", [])
    monkeypatch.setattr(discovery.shutil, "which", lambda name: None)
    monkeypatch.setattr(discovery, "has_module", lambda name: False)
    return tmp_path


def add_ollama_model(store, namespace, model, tag, blob=b"GGUF-model-bytes"):
    digest = f"sha256:{model}{tag}".replace(".", "")
    blob_path = store / "blobs" / digest.replace(":", "-")
    blob_path.parent.mkdir(parents=True, exist_ok=True)
    blob_path.write_bytes(blob)
    manifest = store / "manifests" / "registry.ollama.ai" / namespace / model / tag
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"layers": [
        {"mediaType": "application/vnd.ollama.image.model", "digest": digest, "size": len(blob)},
        {"mediaType": "application/vnd.ollama.image.template", "digest": "sha256:t", "size": 10},
    ]}))
    return blob_path


def test_ollama_models_are_read_from_disk_when_the_server_is_stopped(isolated):
    blob = add_ollama_model(isolated / "ollama", "library", "qwen3", "8b")
    add_ollama_model(isolated / "ollama", "someone", "custom", "latest")
    (isolated / "ollama" / "manifests" / "registry.ollama.ai" / "library" / "broken").mkdir(parents=True)
    (isolated / "ollama" / "manifests" / "registry.ollama.ai" / "library" / "broken" / "x").write_text("{")
    result = Discovery(isolated).scan(force=True)
    entries = {entry["name"]: entry for entry in result["entries"]}
    assert set(entries) == {"qwen3:8b", "someone/custom:latest"}
    stopped = entries["qwen3:8b"]
    assert stopped["id"] == "ollama:qwen3:8b" and stopped["kind"] == "ollama"
    assert stopped["available"] is False and "not installed" in stopped["reason"]
    assert stopped["absolute_path"] == str(blob) and stopped["bytes"] == len(b"GGUF-model-bytes") + 10
    assert result["runtimes"]["ollama"]["running"] is False
    assert result["runtimes"]["ollama"]["models"] == 2


def test_stopped_ollama_models_run_through_llama_cpp_when_it_exists(isolated, monkeypatch):
    add_ollama_model(isolated / "ollama", "library", "qwen3", "8b")
    server = isolated / "llama-server"
    server.write_text("#!/bin/sh\n")
    server.chmod(0o755)
    monkeypatch.setattr(discovery.shutil, "which", lambda name: "/usr/bin/ollama" if name == "ollama" else None)
    result = Discovery(isolated, settings=lambda: {"llama_server_path": str(server)}).scan(force=True)
    (entry,) = result["entries"]
    assert entry["available"] is True and entry["runtime"] == "llama-server"
    assert "llama.cpp" in entry["detail"]["note"] and entry["detail"]["startable"] is True
    assert result["runtimes"]["llama_cpp"]["binary"] == str(server)
    assert result["runtimes"]["ollama"]["installed"] is True


def test_running_ollama_and_local_servers_are_listed(isolated, monkeypatch):
    provider = mock_provider({
        ("GET", "/api/tags"): lambda handler, payload: (200, {"models": [{
            "name": "llama3.2:3b", "size": 2_000_000_000, "modified_at": "2026-10-01T10:00:00.123456789Z",
            "details": {"family": "llama", "parameter_size": "3.2B", "quantization_level": "Q4_K_M"},
        }]}),
        ("GET", "/api/version"): lambda handler, payload: (200, {"version": "0.40.0"}),
        ("GET", "/v1/models"): lambda handler, payload: (200, {"data": [
            {"id": "/models/qwen.gguf", "owned_by": "llamacpp"}, {"id": ""},
        ]}),
    })
    with serving(provider) as base:
        port = int(base.rsplit(":", 1)[1])
        monkeypatch.setenv("OLLAMA_HOST", f"0.0.0.0:{port}")
        monkeypatch.setattr(discovery, "LOCAL_SERVERS", [("LM Studio", port), ("vLLM", 9)])
        result = Discovery(isolated).scan(force=True)
        skipped = Discovery(isolated, own_port=lambda: port).scan(force=True)
    entries = {entry["id"]: entry for entry in result["entries"]}
    ollama = entries["ollama:llama3.2:3b"]
    assert ollama["available"] and ollama["runtime"] == "ollama"
    assert ollama["detail"]["base_url"] == f"http://127.0.0.1:{port}"
    assert ollama["detail"]["quantization"] == "Q4_K_M" and ollama["modified"] > 1_700_000_000
    server = entries[f"server:{port}:/models/qwen.gguf"]
    assert server["name"] == "qwen.gguf" and server["kind"] == "openai"
    assert server["source_label"] == "llama.cpp server"  # owned_by beats the port's default label
    assert server["detail"]["base_url"] == f"http://127.0.0.1:{port}/v1"
    assert result["runtimes"]["ollama"] | {"binary": None} == result["runtimes"]["ollama"]
    assert result["runtimes"]["ollama"]["version"] == "0.40.0"
    assert result["runtimes"]["servers"] == [
        {"label": "llama.cpp", "url": f"http://127.0.0.1:{port}/v1", "models": 1},
    ]
    assert not any(entry["source"] == "server" for entry in skipped["entries"])  # never probe itself


def test_hugging_face_cache_and_extra_folders(isolated, monkeypatch):
    # The workspace must explain how to enable optional backends even when the
    # test environment happens to have some of them installed.
    monkeypatch.setattr(catalog, "has_module", lambda name: False)
    monkeypatch.setattr(discovery, "has_module", lambda name: False)
    snapshot = isolated / "hf" / "models--Org--Model-GGUF" / "snapshots" / "abc"
    snapshot.mkdir(parents=True)
    (snapshot / "model-Q4_K_M.gguf").write_bytes(b"x" * 100)
    (snapshot / "mmproj-model.gguf").write_bytes(b"x")
    weights = isolated / "hf" / "models--Org--Weights" / "snapshots" / "def"
    weights.mkdir(parents=True)
    (weights / "config.json").write_text(json.dumps({"model_type": "gpt2", "n_layer": 2}))
    (weights / "model.safetensors").write_bytes(b"\x02\x00\x00\x00\x00\x00\x00\x00{}")
    extra = isolated / "extra"
    tiny_drv_checkpoint(extra / "mine.pt")
    (extra / "big.gguf").write_bytes(b"y" * 10)
    result = Discovery(isolated, settings=lambda: {"model_dirs": [str(extra)]}).scan(force=True)
    entries = {entry["name"]: entry for entry in result["entries"]}
    assert set(entries) == {"model-Q4_K_M.gguf", "Org/Weights", "mine.pt", "big.gguf"}
    gguf = entries["model-Q4_K_M.gguf"]
    assert gguf["source"] == "huggingface" and gguf["detail"]["repo"] == "Org/Model-GGUF"
    assert gguf["available"] is False and "llama" in gguf["reason"]
    assert entries["Org/Weights"]["kind"] == "transformers"
    assert "requirements-finetune.txt" in entries["Org/Weights"]["reason"]
    assert entries["mine.pt"]["kind"] == "mariana-drv" and entries["mine.pt"]["source"] == "folder"
    assert result["runtimes"]["huggingface"]["models"] == 2
    disabled = Discovery(isolated, settings=lambda: {"discover_local": False}).scan(force=True)
    assert disabled["entries"] == []


def test_address_and_binary_resolution(isolated, monkeypatch):
    assert discovery.ollama_url({}) == "http://127.0.0.1:9"
    assert discovery.ollama_url({"ollama_url": "https://ollama.example:8443/"}) == "https://ollama.example:8443"
    monkeypatch.delenv("OLLAMA_HOST")
    assert discovery.ollama_url({}) == "http://127.0.0.1:11434"
    assert discovery.ollama_url({"ollama_url": "0.0.0.0"}) == "http://127.0.0.1:11434"
    assert discovery.find_llama_server({}) is None and discovery.gguf_runtime({}) is None
    binary = isolated / "bin" / "llama-server"
    binary.parent.mkdir()
    binary.write_text("#!/bin/sh\n")
    assert discovery.find_llama_server({"llama_server_path": str(binary)}) is None  # not executable
    binary.chmod(0o755)
    monkeypatch.setenv("MARIANA_LLAMA_SERVER", str(binary))
    assert discovery.find_llama_server({}) == str(binary)
    assert discovery.gguf_runtime({}) == "llama-server"
    monkeypatch.setattr(discovery, "has_module", lambda name: name == "llama_cpp")
    assert discovery.gguf_runtime({"gguf_runtime": "llama-cpp-python"}) == "llama-cpp-python"
    assert discovery._timestamp("not a date") == 0.0
