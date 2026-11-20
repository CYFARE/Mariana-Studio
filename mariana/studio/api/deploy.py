"""Deployment routes: export checkpoints and register them with Ollama."""

from __future__ import annotations

from ..http import Router, accepted

router = Router()


@router.get("/api/deploy/exports")
def list_exports(app, request):
    """Exported artifacts and the checkpoints that can be exported."""
    return {"exports": app.deploy.exports(), "checkpoints": app.deploy.exportable()}


@router.post("/api/deploy/export")
def start_export(app, request):
    """Convert a Mariana checkpoint to GGUF, ONNX or safetensors."""
    return accepted({"job": app.deploy.start_export(request.json)})


@router.post("/api/deploy/ollama")
def create_ollama_model(app, request):
    """Register an exported GGUF file as an Ollama model."""
    return accepted({"job": app.deploy.start_ollama(request.json)})
