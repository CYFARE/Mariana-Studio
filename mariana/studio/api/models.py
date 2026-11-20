"""Model routes: the catalog, endpoints, local runtimes and downloads."""

from __future__ import annotations

from ..errors import BadRequestError, NotFoundError
from ..http import Router, accepted, created
from ..models import downloads
from ..models.catalog import ModelError

router = Router()


@router.get("/api/models")
def list_models(app, request):
    """Every model Studio can see, with saved endpoints."""
    return {
        "models": app.gateway.catalog(force=request.flag("refresh")),
        "endpoints": app.gateway.endpoints.public(),
    }


@router.get("/api/models/describe")
def describe_model(app, request):
    """One model with header metadata (context length, quantization…) filled in."""
    try:
        return app.gateway.describe(request.arg("id"))
    except ModelError as error:
        raise NotFoundError(str(error)) from error


@router.get("/api/models/providers")
def list_providers(app, request):
    """Provider presets for the add-endpoint dialog."""
    return {"providers": app.gateway.providers()}


@router.post("/api/models/providers/models")
def list_provider_models(app, request):
    """Ask a server which models it offers."""
    return {"models": app.gateway.provider_models(request.json)}


@router.post("/api/models/endpoints")
def save_endpoint(app, request):
    """Create or update a saved endpoint."""
    return created({"endpoints": app.gateway.add_endpoint(request.json)})


@router.delete("/api/models/endpoints/{id}")
def delete_endpoint(app, request):
    """Remove a saved endpoint."""
    try:
        return {"endpoints": app.gateway.delete_endpoint(request.params["id"])}
    except ModelError as error:
        raise NotFoundError(str(error)) from error


@router.post("/api/models/test")
def test_endpoint(app, request):
    """Send one short message to check that a model answers."""
    return app.gateway.test_endpoint(request.json.get("id", ""))


@router.get("/api/models/runtimes")
def list_runtimes(app, request):
    """Local runtimes: llama.cpp, Ollama, Transformers, running servers."""
    return {"runtimes": app.gateway.runtimes(force=request.flag("refresh"))}


@router.post("/api/models/runtimes/ollama/start")
def start_ollama(app, request):
    """Start `ollama serve` as a child of Studio."""
    return {"ollama": app.gateway.start_ollama()}


@router.post("/api/models/runtimes/ollama/stop")
def stop_ollama(app, request):
    """Stop the Ollama server Studio started."""
    return {"ollama": app.gateway.stop_ollama()}


@router.post("/api/models/unload")
def unload_models(app, request):
    """Free memory: unload in-process models and stop the managed llama.cpp server."""
    return app.gateway.release_gpu()


@router.get("/api/models/featured")
def featured_models(app, request):
    """Small starter models worth downloading."""
    return {"huggingface": downloads.FEATURED, "ollama": downloads.OLLAMA_FEATURED}


@router.get("/api/models/search")
def search_models(app, request):
    """Search the Hugging Face Hub (needs network access)."""
    model_format = request.arg("format", "gguf")
    if model_format not in {"gguf", "transformers"}:
        raise BadRequestError("format must be gguf or transformers")
    return {"results": downloads.search(request.arg("q"), model_format)}


@router.get("/api/models/repository")
def read_repository(app, request):
    """Files of a Hub repository with sizes."""
    return downloads.repository(request.arg("repo"))


@router.post("/api/models/download")
def start_download(app, request):
    """Start a download job from Hugging Face or pull a model with Ollama."""
    body = request.json
    if body.get("source") == "ollama":
        return accepted({"job": app.downloads.start_ollama(body.get("name", ""))})
    return accepted({"job": app.downloads.start_huggingface(
        body.get("repo", ""), body.get("files"), body.get("format", "gguf"),
    )})
