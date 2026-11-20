"""Fine-tuning routes for any model in the catalog."""

from __future__ import annotations

from ..http import Router, accepted

router = Router()


@router.get("/api/finetune/bases")
def list_bases(app, request):
    """Models that can be fine-tuned, and the reason the others cannot."""
    return app.finetune.bases(force=request.flag("refresh"))


@router.get("/api/finetune/inspect")
def inspect_base(app, request):
    """Architecture and presets of one base model."""
    return app.finetune.inspect(request.arg("id") or request.arg("path"))


@router.post("/api/finetune/plan")
def plan_finetune(app, request):
    """Validate a fine-tune and estimate what will train, without starting it."""
    return app.finetune.plan(request.json)


@router.post("/api/finetune/start")
def start_finetune(app, request):
    """Start a fine-tune; it appears as the active training run."""
    return accepted({"job": app.finetune.start(request.json)})


@router.get("/api/finetune/discovery")
def discover_material(app, request):
    """Search Hugging Face datasets/models and GitHub for fine-tuning material."""
    return app.finetune.discover(
        request.arg("platform", "datasets"), request.arg("q"),
        request.int_arg("limit", 24, 1, 50),
    )


@router.get("/api/finetune/datasets/info")
def describe_dataset(app, request):
    """Splits, licence and saved copy of one Hugging Face dataset."""
    return app.finetune.dataset_info(request.arg("repo"))


@router.post("/api/finetune/datasets/download")
def download_dataset(app, request):
    """Start a job that saves one dataset split as ready-to-train JSON Lines."""
    return accepted({"job": app.finetune.start_dataset(request.json)})
