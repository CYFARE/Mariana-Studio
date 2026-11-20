"""Job routes: one list for training runs, fine-tunes, downloads and exports."""

from __future__ import annotations

from ..http import EventStream, Router

router = Router()


@router.get("/api/jobs")
def list_jobs(app, request):
    """Every job: training runs, fine-tunes, downloads and exports."""
    return {"jobs": app.jobs.list(), "version": app.jobs.version}


@router.get("/api/jobs/events")
def stream_jobs(app, request):
    """Server-sent events: the job list, re-sent whenever any job changes."""

    def events():
        version = -1
        while True:
            version = app.jobs.wait(version)
            yield {"version": version, "jobs": app.jobs.list()}

    return EventStream(events())


@router.post("/api/jobs/clear")
def clear_jobs(app, request):
    """Dismiss every finished job."""
    return {"cleared": app.jobs.clear_finished()}


@router.get("/api/jobs/{id}")
def read_job(app, request):
    """One job with its log lines after ``since``."""
    return {"job": app.jobs.get(request.params["id"], request.int_arg("since", 0, 0, 10**9))}


@router.post("/api/jobs/{id}/cancel")
def cancel_job(app, request):
    """Cancel an active job."""
    return {"job": app.jobs.cancel(request.params["id"])}


@router.post("/api/jobs/{id}/pause")
def pause_job(app, request):
    """Ask a pausable job to save and stop."""
    return {"job": app.jobs.pause(request.params["id"])}


@router.delete("/api/jobs/{id}")
def dismiss_job(app, request):
    """Remove a finished job from the list."""
    app.jobs.remove(request.params["id"])
    return {"dismissed": True, "id": request.params["id"]}
