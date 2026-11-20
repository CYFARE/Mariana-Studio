"""Discovery of fine-tuning datasets, base models and GitHub repositories."""

from __future__ import annotations

import pytest

from mariana.studio.errors import BadRequestError, UpstreamError
from mariana.studio.finetune import discovery
from tests.helpers import mock_provider, serving


@pytest.fixture()
def hub(monkeypatch):
    provider = mock_provider({
        ("GET", "/api/datasets"): lambda handler, payload: (200, [
            {"id": "tatsu-lab/alpaca", "downloads": 1000, "likes": 12,
             "tags": ["instruction", "license:mit"], "lastModified": "2025-01-01", "gated": False},
            {"id": "broken"},
            "junk",
        ]),
        ("GET", "/api/models"): lambda handler, payload: (200, [
            {"id": "Qwen/Qwen3-0.6B", "downloads": 500, "likes": 3,
             "tags": ["text-generation", "license:apache-2.0"], "lastModified": "2025-02-01"},
        ]),
        ("GET", "/search/repositories"): lambda handler, payload: (200, {"items": [
            {"full_name": "org/data", "description": "A dataset for testing",
             "html_url": "https://github.com/org/data", "stargazers_count": 10,
             "topics": ["dataset", "nlp"], "license": {"spdx_id": "MIT"},
             "pushed_at": "2025-03-01"},
        ]}),
        ("GET", "/api/v1/json/data/list/data_name/*"): lambda handler, payload: (200, {"data": {"dataset": [
            {"did": 7, "name": "iris", "status": "active", "format": "ARFF",
             "url": "https://www.openml.org/d/7"},
        ]}}),
        ("GET", "/api/records"): lambda handler, payload: (200, {"hits": {"hits": [
            {"id": 42, "metadata": {"title": "Research corpus", "description": "<p>Plain <b>text</b></p>",
             "license": {"id": "cc-by-4.0"}, "keywords": ["text"], "publication_date": "2025-04-01",
             "resource_type": {"type": "dataset", "title": "Dataset"}},
             "stats": {"unique_downloads": 4}, "links": {"self_html": "https://zenodo.org/records/42"}},
        ]}}),
        ("GET", "/splits"): lambda handler, payload: (200, {"splits": [
            {"config": "default", "split": "validation"},
            {"config": "default", "split": "train"},
        ]}),
        ("GET", "/api/datasets/tatsu-lab/alpaca"): lambda handler, payload: (200, {
            "gated": False, "downloads": 9, "likes": 2, "tags": ["instruction", "license:mit"],
            "cardData": {"license": "mit"},
        }),
    })
    with serving(provider) as base:
        monkeypatch.setattr(discovery, "HUB", base)
        monkeypatch.setattr(discovery, "DATASETS_SERVER", base)
        monkeypatch.setattr(discovery, "GITHUB_API", base)
        monkeypatch.setattr(discovery, "OPENML_API", base + "/api/v1/json")
        monkeypatch.setattr(discovery, "ZENODO_API", base + "/api")
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        yield {"provider": provider, "base": base}


def test_search_normalises_results_from_every_platform(hub):
    datasets, warnings = discovery.search("datasets", "alpaca")
    assert warnings == [] and len(datasets) == 1
    item = datasets[0]
    assert item["kind"] == "dataset" and item["repo"] == "tatsu-lab/alpaca"
    assert item["url"].endswith("/datasets/tatsu-lab/alpaca")
    assert item["downloads"] == 1000 and item["tags"] == ["instruction"]
    assert "search=alpaca" in hub["provider"].calls[-1]["path"]

    models, _ = discovery.search("models", "qwen")
    assert models[0]["kind"] == "model" and models[0]["repo"] == "Qwen/Qwen3-0.6B"
    assert "pipeline_tag=text-generation" in hub["provider"].calls[-1]["path"]

    repositories, _ = discovery.search("github", "alpaca")
    repository = repositories[0]
    assert repository["platform"] == "github" and repository["stars"] == 10
    assert repository["url"] == "https://github.com/org/data"
    assert repository["license"] == "MIT" and repository["tags"] == ["dataset", "nlp"]
    assert "dataset" in hub["provider"].calls[-1]["path"]

    every, warnings = discovery.search("all", "alpaca", limit=9)
    assert warnings == []
    assert {item["platform"] for item in every} == {"huggingface", "github", "openml", "zenodo"}
    assert {item["kind"] for item in every} == {"dataset", "model", "repository"}

    openml, _ = discovery.search("openml", "iris")
    assert openml[0]["platform"] == "openml" and openml[0]["title"] == "iris"
    assert openml[0]["url"].endswith("/d/7") and "ARFF" in openml[0]["summary"]

    zenodo, _ = discovery.search("zenodo", "text")
    assert zenodo[0]["platform"] == "zenodo" and zenodo[0]["downloads"] == 4
    assert zenodo[0]["summary"] == "Plain text" and zenodo[0]["license"] == "cc-by-4.0"


def test_blank_search_never_touches_the_network(hub):
    before = len(hub["provider"].calls)
    assert discovery.search("all", "   ") == ([], [])
    assert len(hub["provider"].calls) == before
    with pytest.raises(BadRequestError, match="Platform must be one of"):
        discovery.search("magic", "alpaca")


def test_dataset_info_lists_splits_and_prefers_train(hub):
    info = discovery.dataset_info("tatsu-lab/alpaca")
    assert info["splits"] == [
        {"config": "default", "split": "validation"},
        {"config": "default", "split": "train"},
    ]
    assert info["license"] == "mit" and info["downloads"] == 9
    assert discovery.select_split(info["splits"])["split"] == "train"
    assert discovery.select_split(info["splits"], "default", "validation")["split"] == "validation"
    with pytest.raises(BadRequestError, match="not found"):
        discovery.select_split(info["splits"], "default", "test")
    with pytest.raises(BadRequestError, match="no viewer splits"):
        discovery.select_split([])


def test_dataset_names_are_validated_before_they_reach_a_url(hub):
    assert discovery.validate_split("wikitext-2-raw-v1", "Split") == "wikitext-2-raw-v1"
    for bad in ("../split", "has space", "a/b"):
        with pytest.raises(BadRequestError, match="may contain only"):
            discovery.validate_split(bad, "Split")
    with pytest.raises(BadRequestError, match="repository id"):
        discovery.dataset_info("../etc")


def test_remote_urls_must_be_http(hub):
    github = discovery._github_item({"full_name": "org/x", "html_url": "javascript:alert(1)"})
    zenodo = discovery._zenodo_item({
        "id": 3, "metadata": {"title": "x"}, "links": {"html": "data:text/html,x"}})
    openml = discovery._openml_item({"did": 4, "name": "x", "url": "file:///etc/passwd"})
    assert github["url"] == "https://github.com/org/x"
    assert zenodo["url"] == "https://zenodo.org/records/3"
    assert openml["url"] == "https://www.openml.org/d/4"


def test_a_github_rate_limit_becomes_actionable_advice(monkeypatch):
    provider = mock_provider({
        ("GET", "/api/datasets"): lambda handler, payload: (200, []),
        ("GET", "/api/models"): lambda handler, payload: (200, []),
        ("GET", "/api/v1/json/data/list/data_name/*"): lambda handler, payload: (200, {"data": {"dataset": []}}),
        ("GET", "/api/records"): lambda handler, payload: (200, {"hits": {"hits": []}}),
        ("GET", "/search/repositories"): lambda handler, payload: (403, {"error": "rate limited"}),
    })
    with serving(provider) as base:
        monkeypatch.setattr(discovery, "HUB", base)
        monkeypatch.setattr(discovery, "GITHUB_API", base)
        monkeypatch.setattr(discovery, "OPENML_API", base + "/api/v1/json")
        monkeypatch.setattr(discovery, "ZENODO_API", base + "/api")
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        with pytest.raises(UpstreamError, match="GitHub's search limit"):
            discovery.search("github", "alpaca")
        results, warnings = discovery.search("all", "alpaca")
        assert results == []
        assert len(warnings) == 1 and "GitHub" in warnings[0] and "search limit" in warnings[0]
