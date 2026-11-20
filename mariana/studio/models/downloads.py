"""Download models as jobs: Hugging Face Hub files and ``ollama pull``.

Hub downloads use only the standard library: resumable (HTTP ``Range``),
verified (size, and SHA-256 when the Hub publishes one), written to
``models/<org>--<name>/`` via a ``.part`` file so an interrupted download never
looks complete.  The Hub token is sent only to the Hub, never to the CDN it
redirects to.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from ..errors import (
    BadRequestError,
    ConflictError,
    NotFoundError,
    StudioError,
    UnavailableError,
    UpstreamError,
)
from ..jobs import JobCancelled

HUB = "https://huggingface.co"
USER_AGENT = "model-downloader/1.0"
CHUNK = 1 << 20
_REPO = re.compile(r"[A-Za-z0-9][\w.-]{0,95}(/[A-Za-z0-9][\w.-]{0,95})?")
_OLLAMA_NAME = re.compile(r"[A-Za-z0-9][\w./-]{0,120}(:[\w.-]{1,80})?")
_TRANSFORMERS_FILES = re.compile(
    r"(config\.json|generation_config\.json|tokenizer[^/]*|special_tokens_map\.json|"
    r"vocab\.[^/]+|merges\.txt|added_tokens\.json|chat_template\.[^/]+|[^/]+\.model|"
    r"[^/]+\.safetensors)"
)
_PYTORCH_WEIGHTS = re.compile(r"pytorch_model[^/]*\.bin")

# Small, permissively licensed models that run on modest hardware.  Files and
# sizes are read live from the Hub, so only repository ids are fixed here.
FEATURED = [
    {"repo": "Qwen/Qwen3-0.6B-GGUF", "format": "gguf", "title": "Qwen3 0.6B",
     "summary": "Tiny reasoning-capable chat model; runs anywhere.", "tags": ["chat", "tiny"]},
    {"repo": "Qwen/Qwen2.5-0.5B-Instruct-GGUF", "format": "gguf", "title": "Qwen2.5 0.5B Instruct",
     "summary": "Fast general assistant under 500 MB.", "tags": ["chat", "tiny"]},
    {"repo": "ggml-org/gemma-3-1b-it-GGUF", "format": "gguf", "title": "Gemma 3 1B",
     "summary": "Google's compact instruction model.", "tags": ["chat", "small"]},
    {"repo": "bartowski/Llama-3.2-1B-Instruct-GGUF", "format": "gguf", "title": "Llama 3.2 1B Instruct",
     "summary": "Meta's smallest Llama 3.2; many quantizations.", "tags": ["chat", "small"]},
    {"repo": "bartowski/Llama-3.2-3B-Instruct-GGUF", "format": "gguf", "title": "Llama 3.2 3B Instruct",
     "summary": "A stronger everyday assistant for 4 GB+ of memory.", "tags": ["chat"]},
    {"repo": "Qwen/Qwen3-4B-GGUF", "format": "gguf", "title": "Qwen3 4B",
     "summary": "Capable reasoning model that fits an 8 GB GPU.", "tags": ["chat", "reasoning"]},
    {"repo": "ggml-org/SmolLM3-3B-GGUF", "format": "gguf", "title": "SmolLM3 3B",
     "summary": "Multilingual 3B model with long context.", "tags": ["chat"]},
    {"repo": "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF", "format": "gguf", "title": "Qwen2.5 Coder 1.5B",
     "summary": "Code completion and explanation.", "tags": ["code"]},
    {"repo": "microsoft/Phi-3-mini-4k-instruct-gguf", "format": "gguf", "title": "Phi-3 Mini 4K",
     "summary": "Microsoft's 3.8B instruction model.", "tags": ["chat"]},
    {"repo": "HuggingFaceTB/SmolLM2-135M-Instruct", "format": "transformers",
     "title": "SmolLM2 135M Instruct", "summary": "The quickest model to fine-tune end to end.",
     "tags": ["fine-tune", "tiny"]},
    {"repo": "HuggingFaceTB/SmolLM2-360M-Instruct", "format": "transformers",
     "title": "SmolLM2 360M Instruct", "summary": "Small chat model, fine-tunes in minutes.",
     "tags": ["fine-tune", "tiny"]},
    {"repo": "Qwen/Qwen3-0.6B", "format": "transformers", "title": "Qwen3 0.6B (weights)",
     "summary": "Original weights for LoRA or full fine-tuning.", "tags": ["fine-tune"]},
    {"repo": "Qwen/Qwen2.5-0.5B-Instruct", "format": "transformers",
     "title": "Qwen2.5 0.5B Instruct (weights)", "summary": "A solid 0.5B base for adaptation.",
     "tags": ["fine-tune"]},
    {"repo": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "format": "transformers",
     "title": "TinyLlama 1.1B Chat", "summary": "Llama-architecture chat model for experiments.",
     "tags": ["fine-tune"]},
    {"repo": "openai-community/gpt2", "format": "transformers", "title": "GPT-2 (124M)",
     "summary": "The classic base model; no chat template.", "tags": ["fine-tune", "base"]},
]
OLLAMA_FEATURED = [
    {"name": "qwen3:0.6b", "summary": "Tiny reasoning-capable chat model."},
    {"name": "llama3.2:1b", "summary": "Meta's smallest Llama 3.2."},
    {"name": "llama3.2:3b", "summary": "Everyday assistant for 4 GB+."},
    {"name": "gemma3:1b", "summary": "Google's compact instruction model."},
    {"name": "qwen3:4b", "summary": "Capable reasoning model for 8 GB GPUs."},
    {"name": "qwen2.5-coder:1.5b", "summary": "Code completion and explanation."},
    {"name": "phi3:mini", "summary": "Microsoft's 3.8B instruction model."},
]


def hub_token() -> str:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or ""


class _StripAuthOnRedirect(urllib.request.HTTPRedirectHandler):
    """Follow redirects, but never forward the Hub token to another host."""

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


def _open(url: str, headers: dict | None = None, timeout: float = 30):
    request_headers = {"User-Agent": USER_AGENT, **(headers or {})}
    token = hub_token()
    if token and urllib.parse.urlparse(url).netloc == urllib.parse.urlparse(HUB).netloc:
        request_headers["Authorization"] = f"Bearer {token}"
    try:
        return _opener.open(urllib.request.Request(url, headers=request_headers), timeout=timeout)
    except urllib.error.HTTPError as error:
        if error.code in {401, 403}:
            raise UpstreamError(
                "Hugging Face refused access. This repository may be gated: accept its licence "
                "on huggingface.co and export HF_TOKEN before starting Studio."
            ) from error
        if error.code == 404:
            raise NotFoundError("Repository or file not found on Hugging Face") from error
        if error.code == 416:
            raise
        raise UpstreamError(f"Hugging Face returned HTTP {error.code}") from error
    except urllib.error.URLError as error:
        raise UnavailableError(f"Cannot reach Hugging Face: {error.reason}") from error
    except TimeoutError as error:
        raise UnavailableError("Hugging Face did not respond in time") from error


def _api(path: str, timeout: float = 20):
    with _open(f"{HUB}/api/{path}", timeout=timeout) as response:
        try:
            return json.loads(response.read().decode("utf-8"))
        except ValueError as error:
            raise UpstreamError("Hugging Face returned an unreadable response") from error


def validate_repo(repo: str) -> str:
    repo = str(repo or "").strip().strip("/")
    if repo.startswith(HUB):
        repo = repo[len(HUB):].strip("/")
    if not _REPO.fullmatch(repo):
        raise BadRequestError("Enter a Hugging Face repository id such as Qwen/Qwen3-0.6B-GGUF")
    return repo


def _safe_relative(path: str) -> Path:
    relative = Path(path)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise BadRequestError(f"Refusing unsafe file path from the repository: {path}")
    return relative


def search(query: str, model_format: str = "gguf", limit: int = 24) -> list[dict]:
    """Search the Hub for text-generation models in one format."""
    parameters = {
        "search": str(query or "").strip()[:120], "limit": max(1, min(50, int(limit))),
        "sort": "downloads", "direction": "-1",
    }
    if model_format == "gguf":
        parameters["filter"] = "gguf"
    else:
        parameters["pipeline_tag"] = "text-generation"
        parameters["library"] = "transformers"
    results = _api("models?" + urllib.parse.urlencode(parameters))
    if not isinstance(results, list):
        raise UpstreamError("Hugging Face returned an unexpected search result")
    return [
        {"repo": item.get("id") or item.get("modelId"), "format": model_format,
         "downloads": item.get("downloads") or 0, "likes": item.get("likes") or 0,
         "gated": bool(item.get("gated")), "updated": item.get("lastModified") or "",
         "tags": [tag for tag in (item.get("tags") or []) if ":" not in tag][:6]}
        for item in results if isinstance(item, dict) and (item.get("id") or item.get("modelId"))
    ]


def select_transformers_files(files: list[dict]) -> list[dict]:
    """The files a Transformers model needs: config, tokenizer and one set of weights."""
    root = [item for item in files if "/" not in item["path"]]
    chosen = [item for item in root if _TRANSFORMERS_FILES.fullmatch(item["path"])]
    if not any(item["path"].endswith(".safetensors") for item in chosen):
        chosen += [item for item in root if _PYTORCH_WEIGHTS.fullmatch(item["path"])]
    return chosen


def repository(repo: str) -> dict:
    """List a repository's downloadable files with sizes and a suggested selection."""
    repo = validate_repo(repo)
    info = _api(f"models/{repo}?blobs=true")
    files = []
    for sibling in info.get("siblings") or []:
        path = sibling.get("rfilename")
        if not path:
            continue
        lfs = sibling.get("lfs") or {}
        files.append({
            "path": path, "bytes": int(lfs.get("size") or sibling.get("size") or 0),
            "sha256": lfs.get("sha256") or "",
        })
    gguf_files = [
        item for item in files
        if item["path"].lower().endswith(".gguf") and "mmproj" not in item["path"].lower()
    ]
    transformers_files = select_transformers_files(files)
    has_weights = any(
        item["path"].endswith((".safetensors", ".bin")) for item in transformers_files
    ) and any(item["path"] == "config.json" for item in transformers_files)
    return {
        "repo": repo, "gated": bool(info.get("gated")), "private": bool(info.get("private")),
        "license": (info.get("cardData") or {}).get("license"),
        "downloads": info.get("downloads") or 0, "likes": info.get("likes") or 0,
        "pipeline": info.get("pipeline_tag"), "tags": (info.get("tags") or [])[:12],
        "gguf": sorted(gguf_files, key=lambda item: item["bytes"]),
        "transformers": transformers_files if has_weights else [],
        "transformers_bytes": sum(item["bytes"] for item in transformers_files) if has_weights else 0,
        "revision": info.get("sha") or "main",
    }


def destination(root: Path, repo: str) -> Path:
    return Path(root).resolve() / "models" / repo.replace("/", "--")


class DownloadManager:
    """Starts and tracks download jobs."""

    def __init__(self, root: Path, registry, on_complete=None, ollama_url=None):
        self.root = Path(root).resolve()
        self.registry = registry
        self._on_complete = on_complete or (lambda: None)
        self._ollama_url = ollama_url or (lambda: "http://127.0.0.1:11434")

    # -- Hugging Face ----------------------------------------------------
    def start_huggingface(self, repo: str, files: list[str] | None = None,
                          model_format: str = "gguf") -> dict:
        repo = validate_repo(repo)
        listing = repository(repo)
        if model_format == "transformers":
            selected = listing["transformers"]
            if not selected:
                raise BadRequestError(
                    "This repository has no Transformers weights. Choose a GGUF file instead, "
                    "or pick the original (non-GGUF) repository of the model."
                )
        else:
            wanted = set(files or [])
            if not wanted:
                raise BadRequestError("Choose at least one file to download")
            selected = [item for item in listing["gguf"] if item["path"] in wanted]
            if len(selected) != len(wanted):
                raise BadRequestError("Some of the requested files are not in this repository")
        for active in self.registry.active("download"):
            if active["meta"].get("repo") == repo and active["meta"].get("format") == model_format:
                raise ConflictError("This download is already running")
        target = destination(self.root, repo)
        total = sum(item["bytes"] for item in selected)
        target.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(target).free
        if total and free < total + 256 * 1024 * 1024:
            raise ConflictError(
                f"Not enough disk space: {total / 1e9:.1f} GB needed, {free / 1e9:.1f} GB free"
            )
        job = self.registry.create(
            "download", f"Download {repo}",
            meta={"repo": repo, "format": model_format, "source": "huggingface",
                  "files": [item["path"] for item in selected],
                  "target": str(target.relative_to(self.root)), "view": "models"},
        )
        job.update(total=total, unit="bytes", stage="Starting download")
        revision = listing["revision"]
        self.registry.start(job, lambda job: self._run_huggingface(job, repo, revision, selected, target))
        return job.snapshot()

    def _run_huggingface(self, job, repo, revision, selected, target: Path) -> dict:
        done_bytes = 0
        started = time.time()
        for index, item in enumerate(selected, 1):
            relative = _safe_relative(item["path"])
            final = target / relative
            final.parent.mkdir(parents=True, exist_ok=True)
            label = f"{relative.name} ({index}/{len(selected)})"
            if final.is_file() and item["bytes"] and final.stat().st_size == item["bytes"]:
                done_bytes += item["bytes"]
                job.log(f"Already present: {relative}")
                job.update(current=done_bytes, stage=f"Already have {label}")
                continue
            url = f"{HUB}/{repo}/resolve/{revision}/{urllib.parse.quote(item['path'])}"
            self._fetch(job, url, final, item, done_bytes, label, started)
            done_bytes += item["bytes"] or final.stat().st_size
            job.log(f"Saved {relative}")
        self._on_complete()
        return {"path": str(target.relative_to(self.root)), "files": len(selected),
                "model_id": str(target.relative_to(self.root))}

    def _fetch(self, job, url, final: Path, item: dict, base: int, label: str, started: float):
        part = final.with_name(final.name + ".part")
        offset = part.stat().st_size if part.is_file() else 0
        expected = item["bytes"]
        if expected and offset > expected:
            part.unlink()
            offset = 0
        digest = hashlib.sha256() if item.get("sha256") else None
        if digest is not None and offset:
            job.update(stage=f"Verifying partial {label}")
            with part.open("rb") as handle:
                while chunk := handle.read(CHUNK):
                    job.check_cancelled()
                    digest.update(chunk)
        if not expected or offset < expected:
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            try:
                response = _open(url, headers, timeout=60)
            except urllib.error.HTTPError as error:  # 416: the partial file is already complete
                if error.code != 416:
                    raise
                response = None
            if response is not None:
                job.on_cancel(response.close)
                try:
                    if offset and response.status != 206:
                        offset = 0  # the server ignored the range; start over
                        if digest is not None:
                            digest = hashlib.sha256()
                    with part.open("ab" if offset else "wb") as handle:
                        written = offset
                        last = 0.0
                        while True:
                            job.check_cancelled()
                            try:
                                chunk = response.read(CHUNK)
                            except (OSError, ValueError) as error:
                                if job.cancel_requested:
                                    raise JobCancelled() from error
                                raise UpstreamError(
                                    f"The download of {final.name} was interrupted; start it "
                                    "again to resume where it stopped."
                                ) from error
                            if not chunk:
                                break
                            handle.write(chunk)
                            if digest is not None:
                                digest.update(chunk)
                            written += len(chunk)
                            now = time.time()
                            if now - last > 0.25:
                                last = now
                                rate = (base + written) / max(now - started, 1e-6)
                                job.update(current=base + written, stage=f"Downloading {label}",
                                           bytes_per_second=round(rate))
                finally:
                    response.close()
                    job.on_cancel(lambda: None)
        size = part.stat().st_size
        if expected and size != expected:
            raise UpstreamError(
                f"{final.name} is incomplete ({size} of {expected} bytes); start the download "
                "again to resume."
            )
        if digest is not None and digest.hexdigest() != item["sha256"]:
            part.unlink(missing_ok=True)
            raise UpstreamError(f"{final.name} failed its checksum and was discarded; download it again")
        os.replace(part, final)
        job.update(current=base + size)

    # -- Ollama ----------------------------------------------------------
    def start_ollama(self, name: str) -> dict:
        name = str(name or "").strip()
        if not _OLLAMA_NAME.fullmatch(name):
            raise BadRequestError("Enter an Ollama model name such as llama3.2:3b")
        for active in self.registry.active("download"):
            if active["meta"].get("source") == "ollama" and active["meta"].get("model") == name:
                raise ConflictError("This model is already being pulled")
        job = self.registry.create(
            "download", f"Pull {name}",
            meta={"source": "ollama", "model": name, "view": "models"},
        )
        job.update(unit="bytes", stage="Contacting Ollama")
        self.registry.start(job, lambda job: self._run_ollama(job, name))
        return job.snapshot()

    def _run_ollama(self, job, name: str) -> dict:
        url = self._ollama_url().rstrip("/") + "/api/pull"
        body = json.dumps({"model": name, "stream": True}).encode("utf-8")
        request = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            response = urllib.request.urlopen(request, timeout=120)
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:300]
            raise UpstreamError(f"Ollama refused the pull (HTTP {error.code}): {detail}") from error
        except (urllib.error.URLError, OSError) as error:
            raise UnavailableError(
                "Ollama is not running. Start it from Models → Runtimes, then pull again."
            ) from error
        job.on_cancel(response.close)
        layers: dict[str, tuple[int, int]] = {}
        last_status = ""
        try:
            for raw in response:
                job.check_cancelled()
                try:
                    event = json.loads(raw.decode("utf-8", errors="replace"))
                except ValueError:
                    continue
                if event.get("error"):
                    raise UpstreamError(f"Ollama: {event['error']}")
                state = str(event.get("status") or "")
                if event.get("digest") and event.get("total"):
                    layers[event["digest"]] = (int(event.get("completed") or 0), int(event["total"]))
                current = sum(done for done, _ in layers.values())
                total = sum(size for _, size in layers.values())
                if state != last_status:
                    last_status = state
                    job.log(state)
                job.update(current=current, total=total, stage=state.capitalize() or "Pulling")
        except StudioError:
            raise  # an error Ollama reported; StudioError is a ValueError, so it must not fall through
        except (OSError, ValueError) as error:
            if job.cancel_requested:
                raise JobCancelled() from error
            raise UpstreamError("The connection to Ollama was lost during the pull") from error
        finally:
            response.close()
        if last_status != "success":
            raise UpstreamError("Ollama ended the pull without confirming success")
        self._on_complete()
        return {"model": name, "model_id": f"ollama:{name}"}
