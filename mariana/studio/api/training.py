"""Training routes: start, pause and cancel the single training run."""

from __future__ import annotations

from ..http import EventStream, Router, accepted

router = Router()


@router.post("/api/train")
def start_training(app, request):
    """Start a DRV-LM or NCM training run."""
    return accepted({"job": app.training.start(request.json)})


@router.post("/api/pause")
def pause_training(app, request):
    """Ask the run to save a checkpoint and stop."""
    return {"job": app.training.pause(request.json.get("id", ""))}


@router.post("/api/cancel")
def cancel_training(app, request):
    """Stop the run without saving a checkpoint."""
    return {"job": app.training.cancel(request.json.get("id", ""))}


@router.get("/api/job")
def read_job(app, request):
    """The current (or last) training run with its log tail and loss history."""
    return {"job": app.training.state(), "version": app.training.version}


@router.get("/api/checkpoints")
def list_checkpoints(app, request):
    """Mariana checkpoints in the workspace, newest first."""
    return {"checkpoints": app.training.checkpoints()}


@router.get("/api/events")
def stream_training(app, request):
    """Server-sent events: the training run, re-sent whenever it changes."""

    def events():
        version = -1
        while True:
            version, job = app.training.wait(version)
            yield {"version": version, "job": job}

    return EventStream(events())
