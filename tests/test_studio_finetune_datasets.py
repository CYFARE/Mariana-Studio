"""Dataset downloads through the Hugging Face viewer as background jobs."""

from __future__ import annotations

import json
import urllib.parse

import pytest

from mariana.studio.errors import BadRequestError
from mariana.studio.finetune import discovery
from mariana.studio.finetune.datasets import DatasetManager
from mariana.studio.jobs import JobRegistry
from tests.helpers import mock_provider, serving, wait_for


def finished(registry, job_id):
    return wait_for(lambda: (lambda job: job if not job["active"] else None)(registry.get(job_id)))


@pytest.fixture()
def hub(monkeypatch, tmp_path):
    state = {"empty": False}

    def rows(handler, payload):
        if state["empty"]:
            return 200, {"rows": [], "num_rows_total": 0}
        query = urllib.parse.parse_qs(urllib.parse.urlparse(handler.path).query)
        offset = int(query["offset"][0])
        length = int(query["length"][0])
        return 200, {
            "rows": [{
                "row_idx": offset + index,
                "row": {"messages": [
                    {"role": "user", "content": f"q{offset + index}"},
                    {"role": "assistant", "content": "a"},
                ]},
            } for index in range(length)],
            "num_rows_total": 1000,
        }

    provider = mock_provider({
        ("GET", "/splits"): lambda handler, payload: (200, {"splits": [
            {"config": "default", "split": "validation"},
            {"config": "default", "split": "train"},
        ]}),
        ("GET", "/rows"): rows,
        ("GET", "/api/datasets/org/data"): lambda handler, payload: (200, {
            "gated": False, "downloads": 3, "likes": 1, "tags": ["chat"],
            "cardData": {"license": "apache-2.0"},
        }),
    })
    with serving(provider) as base:
        monkeypatch.setattr(discovery, "HUB", base)
        monkeypatch.setattr(discovery, "DATASETS_SERVER", base)
        registry = JobRegistry()
        manager = DatasetManager(tmp_path, registry)
        yield {"manager": manager, "registry": registry, "root": tmp_path,
               "provider": provider, "state": state}
        registry.close()


def test_a_dataset_split_becomes_json_lines(hub):
    manager, registry, root = hub["manager"], hub["registry"], hub["root"]
    job = manager.start_huggingface("org/data", max_rows=250)
    assert job["kind"] == "dataset" and job["title"] == "Download data · org/data"
    assert job["meta"]["target"] == "datasets/org--data" and job["meta"]["view"] == "finetune"
    done = finished(registry, job["id"])
    assert done["status"] == "completed", done
    assert done["result"]["rows"] == 250 and done["result"]["config"] == "default"
    assert done["result"]["split"] == "train" and done["result"]["folder"].endswith("datasets/org--data")

    folder = root / "datasets" / "org--data"
    rows = [json.loads(line) for line in (folder / "data.jsonl").read_text().splitlines()]
    assert len(rows) == 250
    assert rows[0]["messages"][0]["content"] == "q0" and rows[-1]["messages"][0]["content"] == "q249"
    metadata = json.loads((folder / "dataset.json").read_text())
    assert metadata["repo"] == "org/data" and metadata["rows"] == 250
    assert metadata["config"] == "default" and metadata["split"] == "train"
    assert not (folder / "data.jsonl.part").exists()

    saved = manager.installed()
    assert len(saved) == 1 and saved[0]["repo"] == "org/data" and saved[0]["rows"] == 250
    assert saved[0]["folder"] == str(folder.resolve()) and saved[0]["bytes"] > 0


def test_an_explicit_split_skips_the_split_lookup(hub):
    manager, registry = hub["manager"], hub["registry"]
    before = sum("/splits" in call["path"] for call in hub["provider"].calls)
    job = manager.start_huggingface("org/data", "default", "validation", max_rows=100)
    done = finished(registry, job["id"])
    assert done["status"] == "completed", done
    after = sum("/splits" in call["path"] for call in hub["provider"].calls)
    assert after == before
    assert done["result"]["split"] == "validation"


def test_an_empty_split_fails_without_replacing_old_data(hub):
    manager, registry, root = hub["manager"], hub["registry"], hub["root"]
    hub["state"]["empty"] = True
    job = manager.start_huggingface("org/data", "default", "train", max_rows=100)
    done = finished(registry, job["id"])
    assert done["status"] == "failed" and "returned no rows" in done["error"]
    assert not (root / "datasets" / "org--data" / "data.jsonl").exists()
    assert manager.installed() == []


def test_download_requests_are_validated(hub):
    manager = hub["manager"]
    for repo in ("", "../etc", "a/b/c", "has space/x"):
        with pytest.raises(BadRequestError):
            manager.start_huggingface(repo)
    with pytest.raises(BadRequestError, match="max_rows must be between"):
        manager.start_huggingface("org/data", "default", "train", max_rows=10)
    with pytest.raises(BadRequestError, match="Configuration may contain only"):
        manager.start_huggingface("org/data", "bad config", "train")
