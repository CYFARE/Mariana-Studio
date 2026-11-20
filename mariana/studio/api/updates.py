"""Update routes: compare the checkout with GitHub releases and run the git update."""

from __future__ import annotations

from ..http import Router, accepted

router = Router()


@router.get("/api/updates")
def read_updates(app, request):
    """Cached release status; a background check starts when none exists yet."""
    return app.updates.status()


@router.post("/api/updates/check")
def check_updates(app, request):
    """Start a fresh background check against the latest GitHub release."""
    app.updates.start_check(force=True)
    return app.updates.status()


@router.post("/api/updates/apply")
def apply_updates(app, request):
    """Start the background git update; restart Studio when the job completes."""
    return accepted(app.updates.apply())
