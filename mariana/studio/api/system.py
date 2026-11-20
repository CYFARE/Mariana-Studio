"""System routes: application state, hardware overview, folder browsing, settings."""

from __future__ import annotations

import platform
from pathlib import Path

import torch

from ... import __version__
from ...device import describe, pick_device
from ..app import NAME
from ..errors import BadRequestError
from ..http import EventStream, Router
from ..training import TECHNIQUES

router = Router()
HIDDEN_FOLDERS = {"venv", "node_modules", "__pycache__"}
MAX_LISTED = 400


@router.get("/api/state")
def read_state(app, request):
    """Static facts the page needs at start-up, plus the current training job."""
    return {
        "name": NAME, "version": __version__, "techniques": TECHNIQUES,
        "device": describe(pick_device("auto")), "root": str(app.root),
        "job": app.training.state(),
    }


@router.get("/api/system")
def read_system(app, request):
    """Hardware, runtime and workspace overview for the dashboard."""
    device = pick_device("auto")
    stats = app.monitor.snapshot()
    models = app.gateway.catalog()
    return {
        "name": NAME, "version": __version__, "root": str(app.root),
        "uptime": stats["uptime"],
        "device": {"type": device.type, "label": describe(device), "gpu": stats["gpu"],
                   "threads": torch.get_num_threads()},
        "memory": stats["memory"],
        "disk": stats["disk"],
        "software": {"python": platform.python_version(), "torch": torch.__version__,
                     "platform": platform.system()},
        "counts": {
            "models": len(models),
            "chat_ready": sum(1 for entry in models if entry["available"] and entry["capabilities"].get("chat")),
            "tunable": sum(1 for entry in models if entry["capabilities"].get("finetune")),
            "agents": len(app.agents.list_agents()),
            "chats": len(app.chats.list()),
            "active_jobs": len(app.jobs.active()),
        },
        "runtimes": app.gateway.runtimes(),
    }


@router.get("/api/system/events")
def stream_system(app, request):
    """Server-sent events: live memory, GPU and disk usage for the dashboard."""
    return EventStream(app.monitor.events())


@router.get("/api/browse")
def browse(app, request):
    """List sub-folders (and optionally files) so the page can offer a picker."""
    path = Path(request.arg("path") or str(app.root)).expanduser().resolve()
    if path.is_file():
        path = path.parent
    if not path.is_dir():
        raise BadRequestError("Folder does not exist")
    try:
        children = sorted(path.iterdir(), key=lambda child: child.name.lower())
    except PermissionError:
        raise BadRequestError("This folder cannot be read; choose another one") from None
    directories, files = [], []
    for child in children:
        if child.name.startswith("."):
            continue
        try:
            if child.is_dir():
                if child.name not in HIDDEN_FOLDERS:
                    directories.append({"name": child.name, "path": str(child)})
            elif request.flag("files") and child.is_file():
                files.append({"name": child.name, "path": str(child), "bytes": child.stat().st_size})
        except OSError:
            continue
    return {
        "path": str(path), "parent": str(path.parent), "home": str(Path.home()),
        "directories": directories[:MAX_LISTED], "files": files[:MAX_LISTED],
    }


@router.get("/api/settings")
def read_settings(app, request):
    """Current Studio settings."""
    return {"settings": app.settings.get()}


@router.put("/api/settings")
def update_settings(app, request):
    """Change settings; model discovery is refreshed because its inputs changed."""
    settings = app.settings.update(request.json)
    app.gateway.invalidate()
    return {"settings": settings}


@router.get("/api/routes")
def list_routes(app, request):
    """Every HTTP route the server answers, for discovery."""
    from . import router as root_router

    return {"routes": root_router.describe()}
