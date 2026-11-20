"""Model downloads as jobs: Hugging Face files (resume, checksum) and Ollama pulls."""

from __future__ import annotations

import hashlib
import json

import pytest

from mariana.studio.errors import BadRequestError, ConflictError, NotFoundError
from mariana.studio.jobs import JobRegistry
from mariana.studio.models import downloads
from mariana.studio.models.downloads import DownloadManager
from tests.helpers import mock_provider, serving, wait_for

WEIGHTS = bytes(range(256)) * 40
FILES = {
    "model-Q4_K_M.gguf": WEIGHTS,
    "model-Q8_0.gguf": WEIGHTS * 2,
    "mmproj-model.gguf": b"projector",
    "config.json": b'{"model_type": "gpt2"}',
    "tokenizer.json": b"{}",
    "model.safetensors": b"safetensors-bytes",
    "pytorch_model.bin": b"legacy",
    "onnx/model.onnx": b"onnx",
    "README.md": b"# readme",
}


def repository_info(corrupt: str = ""):
    return {
        "id": "org/model", "sha": "rev1", "gated": False, "downloads": 5, "likes": 2,
        "pipeline_tag": "text-generation", "tags": ["gguf", "license:mit"],
        "cardData": {"license": "mit"},
        "siblings": [
            {"rfilename": name, "size": len(data), "lfs": {
                "size": len(data),
                "sha256": "0" * 64 if name == corrupt else hashlib.sha256(data).hexdigest(),
            }} if name.endswith((".gguf", ".safetensors", ".bin")) else {"rfilename": name, "size": len(data)}
            for name, data in FILES.items()
        ],
    }


def resolve(handler, payload):
    name = handler.path.split("/resolve/rev1/", 1)[1]
    data = FILES[name]
    header = handler.headers.get("Range")
    if header:
        start = int(header.split("=")[1].rstrip("-"))
        return 206, data[start:], {"Content-Range": f"bytes {start}-{len(data) - 1}/{len(data)}"}
    return 200, data


@pytest.fixture()
def hub(monkeypatch, tmp_path):
    state = {"corrupt": ""}
    provider = mock_provider({
        ("GET", "/api/models/org/model"): lambda handler, payload: (200, repository_info(state["corrupt"])),
        ("GET", "/api/models/org/gone"): lambda handler, payload: (404, {"error": "not found"}),
        ("GET", "/api/models"): lambda handler, payload: (200, [
            {"id": "org/model", "downloads": 9, "likes": 1, "tags": ["gguf", "license:mit"], "gated": False},
            {"modelId": "org/other", "gated": "auto"}, "junk",
        ]),
        ("GET", "/org/model/resolve/*"): resolve,
    })
    with serving(provider) as base:
        monkeypatch.setattr(downloads, "HUB", base)
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
        registry = JobRegistry()
        completed = []
        manager = DownloadManager(tmp_path, registry, on_complete=lambda: completed.append(True))
        yield {"base": base, "provider": provider, "registry": registry, "manager": manager,
               "root": tmp_path, "completed": completed, "state": state}
        registry.close()


def finished(registry, job_id):
    return wait_for(lambda: (lambda job: job if not job["active"] else None)(registry.get(job_id)))


def test_repository_listing_suggests_the_right_files(hub):
    listing = downloads.repository("https://huggingface.co/org/model/" if False else "org/model")
    assert [item["path"] for item in listing["gguf"]] == ["model-Q4_K_M.gguf", "model-Q8_0.gguf"]
    assert [item["path"] for item in listing["transformers"]] == [
        "config.json", "tokenizer.json", "model.safetensors"]  # no .bin, no onnx/, no README
    assert listing["revision"] == "rev1" and listing["license"] == "mit" and listing["gated"] is False
    assert listing["transformers_bytes"] == sum(len(FILES[name]) for name in (
        "config.json", "tokenizer.json", "model.safetensors"))
    legacy = downloads.select_transformers_files([
        {"path": "config.json"}, {"path": "pytorch_model.bin"}, {"path": "sub/model.safetensors"}])
    assert [item["path"] for item in legacy] == ["config.json", "pytorch_model.bin"]
    with pytest.raises(NotFoundError):
        downloads.repository("org/gone")
    for bad in ("", "../etc", "a/b/c", "has space/x"):
        with pytest.raises(BadRequestError):
            downloads.validate_repo(bad)
    assert downloads.validate_repo(" org/model/ ") == "org/model"
    assert downloads.destination(hub["root"], "org/model") == hub["root"] / "models" / "org--model"


def test_search_maps_results_and_filters_by_format(hub):
    results = downloads.search("model", "gguf")
    assert [item["repo"] for item in results] == ["org/model", "org/other"]
    assert results[0]["tags"] == ["gguf"] and results[1]["gated"] is True
    assert "filter=gguf" in hub["provider"].calls[-1]["path"]
    downloads.search("model", "transformers")
    assert "pipeline_tag=text-generation" in hub["provider"].calls[-1]["path"]


def test_a_gguf_download_is_verified_and_appears_under_models(hub):
    manager, registry, root = hub["manager"], hub["registry"], hub["root"]
    job = manager.start_huggingface("org/model", ["model-Q4_K_M.gguf"])
    assert job["kind"] == "download" and job["meta"]["target"] == "models/org--model"
    done = finished(registry, job["id"])
    assert done["status"] == "completed", done
    target = root / "models" / "org--model" / "model-Q4_K_M.gguf"
    assert target.read_bytes() == WEIGHTS and not target.with_name(target.name + ".part").exists()
    assert done["progress"]["current"] == len(WEIGHTS) == done["progress"]["total"]
    assert done["result"] == {"path": "models/org--model", "files": 1, "model_id": "models/org--model"}
    assert hub["completed"] == [True]
    again = finished(registry, manager.start_huggingface("org/model", ["model-Q4_K_M.gguf"])["id"])
    assert again["status"] == "completed" and "Already present: model-Q4_K_M.gguf" in again["logs"]

    with pytest.raises(BadRequestError, match="at least one file"):
        manager.start_huggingface("org/model", [])
    with pytest.raises(BadRequestError, match="not in this repository"):
        manager.start_huggingface("org/model", ["nope.gguf"])


def test_an_interrupted_download_resumes_from_its_partial_file(hub):
    manager, registry, root = hub["manager"], hub["registry"], hub["root"]
    target = root / "models" / "org--model" / "model-Q8_0.gguf"
    target.parent.mkdir(parents=True)
    target.with_name(target.name + ".part").write_bytes(FILES["model-Q8_0.gguf"][:5000])
    done = finished(registry, manager.start_huggingface("org/model", ["model-Q8_0.gguf"])["id"])
    assert done["status"] == "completed", done
    assert target.read_bytes() == FILES["model-Q8_0.gguf"]
    ranged = [call for call in hub["provider"].calls if call["headers"].get("range")]
    assert ranged and ranged[-1]["headers"]["range"] == "bytes=5000-"


def test_a_corrupt_download_is_discarded(hub):
    hub["state"]["corrupt"] = "model-Q4_K_M.gguf"
    done = finished(hub["registry"], hub["manager"].start_huggingface("org/model", ["model-Q4_K_M.gguf"])["id"])
    assert done["status"] == "failed" and "failed its checksum" in done["error"]
    folder = hub["root"] / "models" / "org--model"
    assert not (folder / "model-Q4_K_M.gguf").exists() and not list(folder.glob("*.part"))
    assert hub["completed"] == []


def test_transformers_downloads_fetch_config_tokenizer_and_weights(hub):
    manager, registry, root = hub["manager"], hub["registry"], hub["root"]
    done = finished(registry, manager.start_huggingface("org/model", model_format="transformers")["id"])
    assert done["status"] == "completed", done
    folder = root / "models" / "org--model"
    assert sorted(path.name for path in folder.iterdir()) == ["config.json", "model.safetensors", "tokenizer.json"]
    assert (folder / "model.safetensors").read_bytes() == FILES["model.safetensors"]


def test_the_hub_token_is_not_forwarded_to_another_host(hub, monkeypatch):
    cdn = mock_provider({("GET", "/blob"): lambda handler, payload: (200, b"cdn-bytes")})
    with serving(cdn) as cdn_base:
        hub["provider"].RequestHandlerClass.routes[("GET", "/redirect")] = (
            lambda handler, payload: (302, b"", {"Location": f"{cdn_base.replace('127.0.0.1', 'localhost')}/blob"}))
        monkeypatch.setenv("HF_TOKEN", "hub-secret")
        with downloads._open(f"{hub['base']}/redirect") as response:
            assert response.read() == b"cdn-bytes"
    assert hub["provider"].calls[-1]["headers"]["authorization"] == "Bearer hub-secret"
    assert "authorization" not in cdn.calls[-1]["headers"]


def test_ollama_pull_reports_progress_and_errors(tmp_path):
    def pull(handler, payload):
        if payload["model"] == "missing:1b":
            return 200, [json.dumps({"error": "pull model manifest: file does not exist"}) + "\n"]
        return 200, [json.dumps(event) + "\n" for event in (
            {"status": "pulling manifest"},
            {"status": "pulling abc", "digest": "sha256:abc", "total": 1000, "completed": 400},
            {"status": "pulling abc", "digest": "sha256:abc", "total": 1000, "completed": 1000},
            {"status": "pulling def", "digest": "sha256:def", "total": 50, "completed": 50},
            {"status": "success"},
        )]

    provider = mock_provider({("POST", "/api/pull"): pull})
    registry = JobRegistry()
    completed = []
    with serving(provider) as base:
        manager = DownloadManager(tmp_path, registry, on_complete=lambda: completed.append(True),
                                  ollama_url=lambda: base)
        done = finished(registry, manager.start_ollama("qwen3:0.6b")["id"])
        assert done["status"] == "completed" and done["result"] == {
            "model": "qwen3:0.6b", "model_id": "ollama:qwen3:0.6b"}
        assert done["progress"]["total"] == 1050 and done["progress"]["percent"] == 100.0
        assert done["logs"] == ["pulling manifest", "pulling abc", "pulling def", "success"]
        assert completed == [True]
        failed = finished(registry, manager.start_ollama("missing:1b")["id"])
        assert failed["status"] == "failed" and "file does not exist" in failed["error"]
    stopped = DownloadManager(tmp_path, registry, ollama_url=lambda: "http://127.0.0.1:9")
    offline = finished(registry, stopped.start_ollama("qwen3:0.6b")["id"])
    assert offline["status"] == "failed" and "Ollama is not running" in offline["error"]
    with pytest.raises(BadRequestError, match="Ollama model name"):
        manager.start_ollama("bad name!")
    registry.close()


def test_the_same_download_cannot_run_twice(hub, monkeypatch):
    import threading

    gate = threading.Event()
    original = DownloadManager._run_huggingface

    def slow(self, job, *arguments):
        gate.wait(10)
        return original(self, job, *arguments)

    monkeypatch.setattr(DownloadManager, "_run_huggingface", slow)
    first = hub["manager"].start_huggingface("org/model", ["model-Q4_K_M.gguf"])
    with pytest.raises(ConflictError, match="already running"):
        hub["manager"].start_huggingface("org/model", ["model-Q8_0.gguf"])
    cancelled = hub["registry"].cancel(first["id"])
    assert cancelled["cancel_requested"] is True
    gate.set()
    assert finished(hub["registry"], first["id"])["status"] == "cancelled"
