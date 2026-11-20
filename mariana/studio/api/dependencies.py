"""Dependency routes: inspect requirement groups and start one pip installer job."""

from __future__ import annotations

from ..http import Router, accepted

router = Router()


@router.get("/api/dependencies")
def read_dependencies(app, request):
    """Requirement groups, missing packages, the auto-install preference and latest job."""
    return app.dependencies.status()


@router.post("/api/dependencies/install")
def install_dependencies(app, request):
    """Start a background job that installs the selected requirement groups."""
    return accepted(app.dependencies.install(request.json.get("groups")))
