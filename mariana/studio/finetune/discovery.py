"""Find fine-tuning material on public hubs: datasets, base models and repositories.

Only the standard library is used.  ``fetch_json`` is the one place that talks
to the network, so tests point the module constants at a local mock server.
Every result has the same shape, whichever platform it came from::

    {"platform", "kind", "id", "repo", "title", "summary", "url", "tags",
     "downloads", "likes", "stars", "updated", "gated", "license"}
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from html import unescape

from ..errors import (
    BadRequestError,
    NotFoundError,
    StudioError,
    UnavailableError,
    UpstreamError,
)
from ..models.downloads import validate_repo

HUB = "https://huggingface.co"
DATASETS_SERVER = "https://datasets-server.huggingface.co"
GITHUB_API = "https://api.github.com"
OPENML_API = "https://www.openml.org/api/v1/json"
ZENODO_API = "https://zenodo.org/api"
USER_AGENT = "mariana-studio/1.0"
MAX_RESPONSE_BYTES = 8 << 20
MAX_SUMMARY_CHARS = 280
ROWS_PAGE = 100
PLATFORMS = {
    "datasets": "Hugging Face datasets",
    "models": "Hugging Face models",
    "github": "GitHub repositories",
    "openml": "OpenML datasets",
    "zenodo": "Zenodo records",
    "all": "All sources",
}
_SPLIT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,119}")
_TAGS = re.compile(r"<[^>]+>")
_WHITESPACE = re.compile(r"\s+")
_URL = re.compile(r"https?://\S+", re.IGNORECASE)

# Small, well-known starting points.  The Hub is read live for search; these
# only save a person from having to know a repository id up front.
FEATURED_DATASETS = (
    {"repo": "tatsu-lab/alpaca", "title": "Alpaca",
     "summary": "Instruction-following examples in the instruction/input/output layout.",
     "tags": ["instruction", "json"]},
    {"repo": "databricks/databricks-dolly-15k", "title": "Dolly 15k",
     "summary": "Human-written instruction and response pairs from Databricks.",
     "tags": ["instruction", "chat"]},
    {"repo": "HuggingFaceH4/ultrachat_200k", "title": "UltraChat 200k",
     "summary": "Multi-turn conversations used to train modern chat models.",
     "tags": ["conversations", "chat"]},
    {"repo": "roneneldan/TinyStories", "title": "TinyStories",
     "summary": "Short synthetic stories for training very small language models.",
     "tags": ["text", "tiny"]},
    {"repo": "openai/gsm8k", "title": "GSM8K",
     "summary": "Grade-school math problems with worked solutions.",
     "tags": ["math", "reasoning"]},
    {"repo": "sahil2801/CodeAlpaca-20k", "title": "CodeAlpaca 20k",
     "summary": "Code instruction and response pairs for programming assistants.",
     "tags": ["code", "instruction"]},
)
FEATURED_MODELS = (
    {"repo": "HuggingFaceTB/SmolLM2-135M-Instruct", "title": "SmolLM2 135M Instruct",
     "summary": "The quickest third-party model to fine-tune end to end.",
     "tags": ["fine-tune", "tiny"]},
    {"repo": "HuggingFaceTB/SmolLM2-360M-Instruct", "title": "SmolLM2 360M Instruct",
     "summary": "Small chat model that fine-tunes in minutes.",
     "tags": ["fine-tune", "tiny"]},
    {"repo": "Qwen/Qwen3-0.6B", "title": "Qwen3 0.6B",
     "summary": "Original weights for LoRA or full fine-tuning.",
     "tags": ["fine-tune", "chat"]},
    {"repo": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "title": "TinyLlama 1.1B Chat",
     "summary": "A compact Llama-architecture chat model for experiments.",
     "tags": ["fine-tune", "chat"]},
)
QUERIES = (
    {"label": "Instruction tuning", "query": "instruction", "platform": "datasets"},
    {"label": "Multi-turn chat", "query": "chat conversations", "platform": "datasets"},
    {"label": "Code", "query": "code", "platform": "datasets"},
    {"label": "Math", "query": "math", "platform": "datasets"},
    {"label": "Small text corpora", "query": "tiny stories", "platform": "datasets"},
    {"label": "Fine-tunable bases", "query": "instruct", "platform": "models"},
)
EXTERNAL_SOURCES = (
    {"name": "Kaggle Datasets", "url": "https://www.kaggle.com/datasets",
     "summary": "Download a CSV or JSON dataset and point the data folder at it."},
    {"name": "Google Dataset Search", "url": "https://datasetsearch.research.google.com/",
     "summary": "Search datasets published across the web by topic."},
    {"name": "Papers with Code", "url": "https://paperswithcode.com/datasets",
     "summary": "Datasets linked to benchmarks and research papers."},
    {"name": "Data.gov", "url": "https://catalog.data.gov/dataset",
     "summary": "Public government datasets, mostly CSV."},
    {"name": "Common Crawl", "url": "https://commoncrawl.org/",
     "summary": "Web-scale text for continued pretraining."},
    {"name": "GitHub dataset topic", "url": "https://github.com/topics/dataset",
     "summary": "Open dataset repositories and collection projects."},
)


class _StripAuthOnRedirect(urllib.request.HTTPRedirectHandler):
    """Follow redirects, but never forward a token to another host."""

    def redirect_request(self, request, fp, code, msg, headers, newurl):
        redirected = super().redirect_request(request, fp, code, msg, headers, newurl)
        if redirected is not None:
            origin = urllib.parse.urlparse(request.full_url).netloc
            target = urllib.parse.urlparse(newurl).netloc
            if origin != target:
                redirected.headers.pop("Authorization", None)
                redirected.unredirected_hdrs.pop("Authorization", None)
        return redirected


_opener = urllib.request.build_opener(_StripAuthOnRedirect)


def github_token() -> str:
    return os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or ""


def _hub_token() -> str:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or ""


def _headers(url: str) -> dict[str, str]:
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    if url.startswith(HUB):
        token = _hub_token()
    elif url.startswith(GITHUB_API):
        token = github_token()
    else:
        token = ""
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def fetch_json(url: str, timeout: float = 20) -> object:
    """Read one JSON document, mapping transport failures to Studio advice."""
    try:
        with _opener.open(urllib.request.Request(url, headers=_headers(url)), timeout=timeout) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as error:
        _raise_http_error(url, error)
    except urllib.error.URLError as error:
        raise UnavailableError(
            f"Cannot reach the discovery service ({error.reason}). Check the network connection "
            "and try again."
        ) from error
    except TimeoutError as error:
        raise UnavailableError("The discovery service did not respond in time. Try again in a moment.") from error
    if len(raw) > MAX_RESPONSE_BYTES:
        raise UpstreamError("The discovery service returned more data than Studio can read; narrow the search.")
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise UpstreamError("The discovery service returned an unreadable response.") from error


def _raise_http_error(url: str, error: urllib.error.HTTPError):
    headers = getattr(error, "headers", None)
    remaining = headers.get("X-RateLimit-Remaining") if headers is not None else None
    if url.startswith(GITHUB_API) and (error.code in {403, 429} or remaining == "0"):
        raise UpstreamError(
            "GitHub's search limit was reached. Wait an hour, set GITHUB_TOKEN, or search "
            "Hugging Face instead."
        ) from error
    if url.startswith((HUB, DATASETS_SERVER)) and error.code in {401, 403}:
        raise UpstreamError(
            "Hugging Face refused access. This repository may be gated: accept its licence "
            "on huggingface.co and export HF_TOKEN before starting Studio."
        ) from error
    if error.code == 404:
        raise NotFoundError("That repository was not found; check its identifier and try again.") from error
    if error.code in {401, 403}:
        raise UpstreamError("The discovery service refused the request; check its credentials.") from error
    raise UpstreamError(f"The discovery service returned HTTP {error.code}; try again in a moment.") from error


def _optional_json(url: str, timeout: float = 15) -> dict:
    """Hub card metadata is a bonus; a missing card must not block a download."""
    try:
        value = fetch_json(url, timeout)
    except StudioError:
        return {}
    return value if isinstance(value, dict) else {}


def _tags(values, limit: int = 8) -> list[str]:
    if not isinstance(values, list):
        return []
    cleaned = []
    for value in values:
        if isinstance(value, str) and value.strip() and ":" not in value:
            cleaned.append(value.strip())
        if len(cleaned) >= limit:
            break
    return cleaned


def _license(value) -> str:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value if item)
    return str(value or "")


def _safe_url(value, fallback: str) -> str:
    """Remote data may not turn an anchor into a ``javascript:`` link."""
    value = str(value or "").strip()
    return value if _URL.fullmatch(value) else fallback


def dataset_url(repo: str) -> str:
    """The public page of one Hub dataset."""
    return f"{HUB}/datasets/{repo}"


def _model_url(repo: str) -> str:
    return f"{HUB}/{repo}"


def _dataset_item(item: dict) -> dict | None:
    repo = item.get("id") or item.get("datasetId")
    if not isinstance(repo, str) or "/" not in repo:
        return None
    return {
        "platform": "huggingface", "kind": "dataset", "id": repo, "repo": repo,
        "title": repo, "summary": "", "url": dataset_url(repo),
        "downloads": int(item.get("downloads") or 0), "likes": int(item.get("likes") or 0),
        "updated": item.get("lastModified") or "", "gated": bool(item.get("gated")),
        "tags": _tags(item.get("tags")),
    }


def _model_item(item: dict) -> dict | None:
    repo = item.get("id") or item.get("modelId")
    if not isinstance(repo, str) or "/" not in repo:
        return None
    return {
        "platform": "huggingface", "kind": "model", "id": repo, "repo": repo,
        "title": repo, "summary": "", "url": _model_url(repo),
        "downloads": int(item.get("downloads") or 0), "likes": int(item.get("likes") or 0),
        "updated": item.get("lastModified") or "", "gated": bool(item.get("gated")),
        "tags": _tags(item.get("tags")),
    }


def _github_item(item: dict) -> dict | None:
    repo = item.get("full_name")
    if not isinstance(repo, str) or "/" not in repo:
        return None
    licence = item.get("license") if isinstance(item.get("license"), dict) else {}
    return {
        "platform": "github", "kind": "repository", "id": repo, "repo": repo,
        "title": repo, "summary": str(item.get("description") or ""),
        "url": _safe_url(item.get("html_url"), f"https://github.com/{repo}"),
        "stars": int(item.get("stargazers_count") or 0),
        "updated": item.get("pushed_at") or item.get("updated_at") or "",
        "license": _license(licence.get("spdx_id") if licence.get("spdx_id") != "NOASSERTION" else ""),
        "tags": _tags(item.get("topics")),
    }


def _plain(value, limit: int = MAX_SUMMARY_CHARS) -> str:
    """Zenodo descriptions are HTML; cards show readable text only."""
    text = _WHITESPACE.sub(" ", unescape(_TAGS.sub(" ", str(value or "")))).strip()
    return text[:limit]


def _openml_item(item: dict) -> dict | None:
    name = item.get("name")
    identifier = item.get("did")
    if not isinstance(name, str) or not name or identifier in (None, ""):
        return None
    url = _safe_url(item.get("url"), f"https://www.openml.org/d/{identifier}")
    status = str(item.get("status") or "active")
    version = item.get("version")
    summary = f"{status.capitalize()} OpenML dataset · {item.get('format') or 'ARFF'}"
    if version:
        summary += f" · version {version}"
    return {
        "platform": "openml", "kind": "dataset", "id": f"openml:{identifier}", "repo": name,
        "title": name, "summary": summary,
        "url": url, "downloads": 0, "likes": 0, "updated": "", "gated": False,
        "tags": [tag for tag in (item.get("format"), "openml") if tag],
    }


def _zenodo_item(item: dict) -> dict | None:
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    identifier = item.get("id")
    title = metadata.get("title") or (f"Zenodo record {identifier}" if identifier else "")
    if not title or identifier in (None, ""):
        return None
    licence = metadata.get("license")
    if isinstance(licence, dict):
        licence = licence.get("id") or licence.get("title")
    stats = item.get("stats") if isinstance(item.get("stats"), dict) else {}
    links = item.get("links") if isinstance(item.get("links"), dict) else {}
    return {
        "platform": "zenodo", "kind": "dataset", "id": f"zenodo:{identifier}", "repo": str(title),
        "title": str(title), "summary": _plain(metadata.get("description")),
        "url": _safe_url(links.get("self_html") or links.get("html"),
                          f"https://zenodo.org/records/{identifier}"),
        "downloads": int(stats.get("unique_downloads") or 0),
        "likes": 0, "updated": metadata.get("publication_date") or item.get("created") or "",
        "gated": False, "license": _license(licence),
        "tags": _tags(metadata.get("keywords")),
    }


def _search_datasets(query: str, limit: int) -> list[dict]:
    parameters = {
        "search": query, "limit": limit, "sort": "downloads", "direction": "-1",
    }
    payload = fetch_json(f"{HUB}/api/datasets?{urllib.parse.urlencode(parameters)}")
    items = payload if isinstance(payload, list) else []
    return [item for entry in items if isinstance(entry, dict) and (item := _dataset_item(entry))]


def _search_models(query: str, limit: int) -> list[dict]:
    parameters = {
        "search": query, "limit": limit, "sort": "downloads", "direction": "-1",
        "pipeline_tag": "text-generation", "library": "transformers",
    }
    payload = fetch_json(f"{HUB}/api/models?{urllib.parse.urlencode(parameters)}")
    items = payload if isinstance(payload, list) else []
    return [item for entry in items if isinstance(entry, dict) and (item := _model_item(entry))]


def _search_github(query: str, limit: int) -> list[dict]:
    parameters = {
        "q": f"{query} dataset", "sort": "stars", "order": "desc", "per_page": limit,
    }
    payload = fetch_json(f"{GITHUB_API}/search/repositories?{urllib.parse.urlencode(parameters)}")
    items = payload.get("items") if isinstance(payload, dict) else []
    return [item for entry in (items or []) if isinstance(entry, dict) and (item := _github_item(entry))]


def _search_openml(query: str, limit: int) -> list[dict]:
    url = f"{OPENML_API}/data/list/data_name/{urllib.parse.quote(query, safe='')}/limit/{limit}"
    payload = fetch_json(url)
    data = payload.get("data") if isinstance(payload, dict) else None
    items = data.get("dataset") if isinstance(data, dict) else None
    if isinstance(items, dict):
        items = [items]
    return [item for entry in (items or []) if isinstance(entry, dict) and (item := _openml_item(entry))]


def _search_zenodo(query: str, limit: int) -> list[dict]:
    parameters = {"q": query, "size": min(100, max(10, limit * 3)), "sort": "mostrecent"}
    payload = fetch_json(f"{ZENODO_API}/records?{urllib.parse.urlencode(parameters)}")
    hits = payload.get("hits") if isinstance(payload, dict) else None
    items = hits.get("hits") if isinstance(hits, dict) else None
    results = []
    for entry in items or []:
        if not isinstance(entry, dict):
            continue
        metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
        resource_type = metadata.get("resource_type")
        if not isinstance(resource_type, dict) or resource_type.get("type") != "dataset":
            continue  # Zenodo also indexes papers; this centre lists training data
        item = _zenodo_item(entry)
        if item:
            results.append(item)
        if len(results) >= limit:
            break
    return results


_SEARCHERS = {
    "datasets": _search_datasets,
    "models": _search_models,
    "github": _search_github,
    "openml": _search_openml,
    "zenodo": _search_zenodo,
}


def search(platform: str = "datasets", query: str = "", limit: int = 24) -> tuple[list[dict], list[str]]:
    """Search one platform, or every platform; return ``(results, warnings)``.

    A blank query returns nothing so opening the workspace never touches the
    network.  With ``platform="all"`` a failing source becomes a warning instead
    of hiding the sources that answered.
    """
    platform = str(platform or "datasets")
    if platform not in PLATFORMS:
        raise BadRequestError(f"Platform must be one of: {', '.join(PLATFORMS)}")
    query = str(query or "").strip()[:120]
    limit = max(1, min(50, int(limit)))
    if not query:
        return [], []
    if platform != "all":
        return _SEARCHERS[platform](query, limit), []
    names = [name for name in PLATFORMS if name != "all"]
    base, remainder = divmod(limit, len(names))
    results, warnings = [], []
    for index, name in enumerate(names):
        share = max(1, base + (1 if index < remainder else 0))
        try:
            results.extend(_SEARCHERS[name](query, share))
        except StudioError as error:
            warnings.append(f"{PLATFORMS[name]}: {error}")
    return results[:limit], warnings


def select_split(splits: list[dict], config: str = "", split: str = "") -> dict:
    """Pick the requested dataset configuration/split, preferring ``train``."""
    if not splits:
        raise BadRequestError(
            "This dataset has no viewer splits; download its files and point the data folder at them."
        )
    if config and split:
        match = next((item for item in splits if item["config"] == config and item["split"] == split), None)
    elif config:
        match = next((item for item in splits if item["config"] == config), None)
    elif split:
        match = next((item for item in splits if item["split"] == split), None)
    else:
        match = next((item for item in splits if item["split"] == "train"), splits[0])
    if match is None:
        options = ", ".join(f"{item['config']}/{item['split']}" for item in splits[:8])
        raise BadRequestError(f"That configuration and split were not found. Choose one of: {options}")
    return match


def validate_split(value: str, label: str) -> str:
    """A dataset configuration or split name from an untrusted request."""
    value = str(value or "").strip()
    if value and not _SPLIT.fullmatch(value):
        raise BadRequestError(f"{label} may contain only letters, numbers, dots, dashes, and underscores")
    return value


def dataset_info(repo: str) -> dict:
    """Splits and card metadata for one Hugging Face dataset."""
    repo = validate_repo(repo)
    try:
        payload = fetch_json(f"{DATASETS_SERVER}/splits?{urllib.parse.urlencode({'dataset': repo})}")
    except NotFoundError as error:
        raise NotFoundError(
            "The Hugging Face dataset viewer did not return splits for this repository. "
            "Check the repository id, or download its files and point the data folder at them."
        ) from error
    raw = payload.get("splits") if isinstance(payload, dict) else []
    splits, seen = [], set()
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        config = str(item.get("config") or "default")
        split = str(item.get("split") or "train")
        if (config, split) in seen:
            continue
        seen.add((config, split))
        splits.append({"config": config, "split": split})
    if not splits:
        raise NotFoundError(
            "The Hugging Face dataset viewer did not return splits for this repository. "
            "Check the repository id, or download its files and point the data folder at them."
        )
    metadata = _optional_json(f"{HUB}/api/datasets/{repo}")
    card = metadata.get("cardData") if isinstance(metadata.get("cardData"), dict) else {}
    return {
        "repo": repo, "splits": splits, "url": dataset_url(repo),
        "gated": bool(metadata.get("gated")), "private": bool(metadata.get("private")),
        "license": _license(card.get("license")),
        "downloads": int(metadata.get("downloads") or 0),
        "likes": int(metadata.get("likes") or 0),
        "tags": _tags(metadata.get("tags"), 12),
    }


def dataset_rows(repo: str, config: str, split: str, offset: int = 0, length: int = ROWS_PAGE) -> dict:
    """One page of rows from the Hugging Face dataset viewer."""
    repo = validate_repo(repo)
    config = validate_split(config, "Configuration")
    split = validate_split(split, "Split")
    if not config or not split:
        raise BadRequestError("Choose a dataset configuration and split first")
    parameters = {
        "dataset": repo, "config": config, "split": split,
        "offset": max(0, int(offset)), "length": max(1, min(ROWS_PAGE, int(length))),
    }
    payload = fetch_json(
        f"{DATASETS_SERVER}/rows?{urllib.parse.urlencode(parameters)}", timeout=60,
    )
    if not isinstance(payload, dict):
        raise UpstreamError("The dataset viewer returned an unexpected response; try another split.")
    return payload


def _featured(raw: dict, kind: str) -> dict:
    repo = raw["repo"]
    url = dataset_url(repo) if kind == "dataset" else _model_url(repo)
    return {
        "platform": "huggingface", "kind": kind, "id": repo, "repo": repo,
        "title": raw.get("title") or repo, "summary": raw.get("summary") or "",
        "url": url, "tags": list(raw.get("tags") or []),
        "downloads": 0, "likes": 0, "updated": "", "gated": False,
    }


def featured() -> dict:
    """Static starting points returned with the first discovery request."""
    return {
        "datasets": [_featured(item, "dataset") for item in FEATURED_DATASETS],
        "models": [_featured(item, "model") for item in FEATURED_MODELS],
        "queries": [dict(item) for item in QUERIES],
        "external": [dict(item) for item in EXTERNAL_SOURCES],
    }
