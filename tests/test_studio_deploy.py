"""Deployment: exporting checkpoints as jobs and registering them with Ollama."""

from __future__ import annotations

import pytest

from mariana.studio.deploy import DeployService
from mariana.studio.errors import BadRequestError, NotFoundError, UnavailableError
from mariana.studio.jobs import JobRegistry
from mariana.studio.models import ModelGateway
from tests.helpers import tiny_drv_checkpoint, tiny_ncm_checkpoints, wait_for


@pytest.fixture()
def deploy(tmp_path):
    registry = JobRegistry()
    gateway = ModelGateway(tmp_path, settings=lambda: {"discover_local": False})
    yield DeployService(tmp_path, registry, gateway)
    registry.close()
    gateway.close()


def test_only_dense_drv_checkpoints_are_exportable(tmp_path, deploy):
    tiny_drv_checkpoint(tmp_path / "trained-model" / "dense.pt")
    tiny_drv_checkpoint(tmp_path / "trained-model" / "sparse.pt", use_moe=True)
    tiny_ncm_checkpoints(tmp_path / "trained-model", "ncm")
    listed = {item["name"]: item for item in deploy.exportable()}
    assert set(listed) == {"dense.pt", "sparse.pt"}
    assert listed["dense.pt"]["exportable"] is True and listed["dense.pt"]["reason"] == ""
    assert listed["sparse.pt"]["exportable"] is False and "dense model" in listed["sparse.pt"]["reason"]
    for config, message in (
        ({"model": "trained-model/sparse.pt"}, "Sparse-MoE"),
        ({"model": "trained-model/ncm.pt"}, "Only Mariana DRV-LM"),
        ({"model": "trained-model/dense.pt", "format": "exe"}, "Format must be one of"),
        ({"model": "trained-model/dense.pt", "dtype": "f8"}, "Precision must be one of"),
        ({"model": "trained-model/dense.pt", "name": "../escape"}, "Export name"),
    ):
        with pytest.raises(BadRequestError, match=message):
            deploy.start_export(config)
    assert deploy.exports() == []


@pytest.mark.slow
def test_a_checkpoint_is_exported_to_safetensors_as_a_job(tmp_path, deploy):
    pytest.importorskip("safetensors")
    tiny_drv_checkpoint(tmp_path / "trained-model" / "dense.pt")
    job = deploy.start_export({"model": "trained-model/dense.pt", "format": "safetensors", "name": "tiny"})
    assert job["kind"] == "export" and job["title"] == "Export dense.pt → SAFETENSORS"
    done = wait_for(lambda: (lambda state: state if not state["active"] else None)(
        deploy.registry.get(job["id"])), timeout=120)
    assert done["status"] == "completed", "\n".join(done["logs"][-15:])
    assert done["result"]["files"] == ["tiny-llama.safetensors", "tiny-llama.config.json"]
    (artifact,) = deploy.exports()
    assert artifact["name"] == "tiny-llama.safetensors" and artifact["format"] == "safetensors"
    assert artifact["path"] == "exports/tiny-llama.safetensors" and artifact["bytes"] > 0
    assert artifact["modelfile"] is False


def test_registering_with_ollama_needs_a_gguf_and_a_running_server(tmp_path, deploy, monkeypatch):
    from mariana.studio import deploy as deploy_module

    (tmp_path / "exports").mkdir()
    (tmp_path / "exports" / "model-f16.gguf").write_bytes(b"gguf")
    (tmp_path / "exports" / "model.onnx").write_bytes(b"onnx")
    assert {item["name"] for item in deploy.exports()} == {"model-f16.gguf", "model.onnx"}
    for missing in ("nope.gguf", "model.onnx", "../trained-model/model-f16.gguf", ""):
        with pytest.raises(NotFoundError, match="GGUF file from exports"):
            deploy.start_ollama({"file": missing})
    with pytest.raises(BadRequestError, match="lowercase"):
        deploy.start_ollama({"file": "model-f16.gguf", "name": "Bad Name"})
    monkeypatch.setattr(deploy_module.shutil, "which", lambda name: None)
    with pytest.raises(UnavailableError, match="not installed"):
        deploy.start_ollama({"file": "model-f16.gguf"})
    monkeypatch.setattr(deploy_module.shutil, "which", lambda name: "/usr/bin/ollama")
    monkeypatch.setattr(deploy.gateway.ollama, "running", lambda: False)
    with pytest.raises(UnavailableError, match="not running"):
        deploy.start_ollama({"file": "model-f16.gguf"})
