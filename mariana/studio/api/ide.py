"""IDE routes: workspaces, files, sessions and the streaming coding agent.

Handlers translate HTTP into calls on :class:`mariana.studio.ide.IdeService`.
The service owns the sandbox, snapshots and the tool loop; nothing here touches
the filesystem directly.
"""

from __future__ import annotations

from pathlib import Path

from ..errors import BadRequestError
from ..http import EventStream, Response, Router, accepted, created
from ..ide import MAX_FILE_BYTES, MAX_READ_LINES
from ..validation import flag

router = Router()
SPEC = Path(__file__).resolve().parents[3] / "docs" / "ide.md"
MARKDOWN = "text/markdown; charset=utf-8"


@router.get("/api/ide")
def read_ide(app, request):
    """IDE discovery: workspaces, modes, limits, tool schemas and shell availability."""
    return app.ide.overview()


@router.get("/api/ide/spec")
def read_spec(app, request):
    """The IDE protocol and workflow documentation as markdown."""
    return Response.text(SPEC.read_text(encoding="utf-8") if SPEC.is_file() else "", MARKDOWN)


@router.get("/api/ide/workspaces")
def list_workspaces(app, request):
    """Every IDE workspace with its size and language mix."""
    return {"workspaces": app.ide.list_workspaces()}


@router.post("/api/ide/workspaces")
def create_workspace(app, request):
    """Create a workspace, optionally copying an existing folder into it."""
    body = request.json
    return created({"workspace": app.ide.create_workspace(
        body.get("name"), source=body.get("source") or None,
        template=body.get("template") or "empty",
    )})


@router.post("/api/ide/workspaces/open")
def open_workspace(app, request):
    """Open an existing folder in place; the files are not copied."""
    body = request.json
    workspace = app.ide.open_workspace(body.get("name"), body.get("source"))
    return created({"workspace": workspace, "external": workspace.get("external", True)})


@router.get("/api/ide/workspaces/{id}")
def read_workspace(app, request):
    """One workspace with its file and language statistics."""
    return {"workspace": app.ide.describe_workspace(request.params["id"])}


@router.delete("/api/ide/workspaces/{id}")
def delete_workspace(app, request):
    """Delete a workspace, its sessions, their runs and their snapshots."""
    app.ide.delete_workspace(request.params["id"])
    return {"deleted": True, "id": request.params["id"]}


@router.get("/api/ide/workspaces/{id}/tree")
def read_tree(app, request):
    """The workspace tree; ``depth`` and ``path`` list one folder for lazy loading."""
    return app.ide.tree(
        request.params["id"], request.arg("path"),
        request.int_arg("depth", 0, 0, 8),
    )


@router.get("/api/ide/workspaces/{id}/files")
def find_files(app, request):
    """File and folder paths matching a query, for filters and @-mentions."""
    return app.ide.find_files(
        request.params["id"], request.arg("query"),
        request.int_arg("limit", 200, 1, 1000),
    )


@router.get("/api/ide/workspaces/{id}/file")
def read_file(app, request):
    """One UTF-8 file with the version the editor must send back on save."""
    return app.ide.read_file(
        request.params["id"], request.arg("path"),
        start=request.int_arg("start", 0, 0, 10**9),
        limit=request.int_arg("limit", 0, 0, MAX_READ_LINES),
    )


@router.get("/api/ide/workspaces/{id}/preview")
def preview_file(app, request):
    """Metadata and a source, image or hex preview for any file, including binary."""
    return app.ide.preview_file(
        request.params["id"], request.arg("path"),
        start=request.int_arg("start", 0, 0, 10**9),
        limit=request.int_arg("limit", 0, 0, MAX_READ_LINES),
    )


@router.get("/api/ide/workspaces/{id}/skills")
def list_skills(app, request):
    """Workspace and user skills the coding agent can load."""
    return {"skills": app.ide.list_skills(request.params["id"])}


@router.get("/api/ide/workspaces/{id}/skills/{name}")
def read_skill(app, request):
    """One skill with its editable instructions."""
    return {"skill": app.ide.read_skill(
        request.params["id"], request.params["name"], include_instructions=True)}


@router.post("/api/ide/workspaces/{id}/skills")
def create_skill(app, request):
    """Create a workspace or user skill from the skills editor."""
    return created({"skill": app.ide.create_skill(request.params["id"], request.json)})


@router.put("/api/ide/workspaces/{id}/skills/{name}")
def update_skill(app, request):
    """Replace a skill's display fields and instructions."""
    return {"skill": app.ide.update_skill(
        request.params["id"], request.params["name"], request.json)}


@router.delete("/api/ide/workspaces/{id}/skills/{name}")
def delete_skill(app, request):
    """Delete one skill directory from a known skills folder."""
    return app.ide.delete_skill(request.params["id"], request.params["name"])


@router.put("/api/ide/workspaces/{id}/file")
def write_file(app, request):
    """Save a file; a stale version is rejected instead of overwriting changes."""
    body = request.large_json(MAX_FILE_BYTES * 3 + 10_000)
    content = body.get("content")
    if not isinstance(content, str):
        raise BadRequestError("content must be a string")
    version = body.get("version")
    return app.ide.write_file(
        request.params["id"], str(body.get("path") or ""), content,
        version=None if version is None else str(version),
        force=flag(body, "force", False),
    )


@router.delete("/api/ide/workspaces/{id}/file")
def delete_file(app, request):
    """Delete one file from a workspace."""
    return app.ide.delete_file(request.params["id"], request.arg("path"))


@router.post("/api/ide/workspaces/{id}/folder")
def create_folder(app, request):
    """Create a folder in the workspace explorer."""
    return created(app.ide.create_folder(request.params["id"], request.json.get("path")))


@router.delete("/api/ide/workspaces/{id}/folder")
def delete_folder(app, request):
    """Delete a folder and its contents from the workspace."""
    return app.ide.delete_folder(request.params["id"], request.arg("path"))


@router.post("/api/ide/workspaces/{id}/move")
def move_entry(app, request):
    """Rename or move a file or folder inside the workspace."""
    body = request.json
    return app.ide.move_entry(
        request.params["id"], body.get("from"), body.get("to"),
    )


@router.get("/api/ide/workspaces/{id}/settings")
def read_settings(app, request):
    """The workspace context rules shared by every conversation."""
    return app.ide.settings(request.params["id"])


@router.put("/api/ide/workspaces/{id}/settings")
def write_settings(app, request):
    """Replace hidden paths or the default pinned context."""
    return app.ide.update_settings(request.params["id"], request.json)


@router.post("/api/ide/workspaces/{id}/context")
def preview_context(app, request):
    """Estimate the files and tokens a context selection would add to a prompt."""
    return app.ide.context_preview(request.params["id"], request.json)


@router.get("/api/ide/voice")
def read_voice(app, request):
    """Local speech-to-text availability, accepted formats and privacy notes."""
    return app.voice.status()


@router.post("/api/ide/voice/transcribe")
def transcribe_voice(app, request):
    """Start a background transcription of one short microphone recording."""
    return accepted(app.voice.transcribe(request.large_json(12_000_000)))


@router.post("/api/ide/workspaces/{id}/search")
def search_files(app, request):
    """Search workspace text files by literal text or regular expression."""
    return app.ide.search(request.params["id"], request.json)


@router.post("/api/ide/workspaces/{id}/diagnostics")
def check_diagnostics(app, request):
    """Check Python syntax, JSON validity and optional pyflakes rules."""
    body = request.json
    return app.ide.diagnostics(request.params["id"], str(body.get("path") or ""))


@router.get("/api/ide/workspaces/{id}/sessions")
def list_sessions(app, request):
    """Coding conversations in this workspace, newest first."""
    return {"sessions": app.ide.list_sessions(request.params["id"])}


@router.post("/api/ide/sessions/{session_id}/consolidate")
def consolidate_session(app, request):
    """Consolidate older turns in one saved conversation when the context is full."""
    body = request.json
    return app.ide.consolidate_session(
        request.params["session_id"], str(body.get("model") or ""),
    )


@router.post("/api/ide/workspaces/{id}/runs")
def start_run(app, request):
    """Start one coding turn; follows the run stream at ``/api/ide/runs/{run_id}/events``."""
    return accepted(app.ide.start_run(request.params["id"], request.json))


@router.get("/api/ide/runs/{run_id}")
def read_run(app, request):
    """Run state with the events after ``since``."""
    return app.ide.get_run(
        request.params["run_id"], request.int_arg("since", 0, 0, 10**9),
    )


@router.get("/api/ide/runs/{run_id}/events")
def stream_run(app, request):
    """Server-sent events for one coding run; the stream ends with the run."""
    return EventStream(app.ide.stream_run(
        request.params["run_id"], request.int_arg("since", 0, 0, 10**9),
    ))


@router.post("/api/ide/runs/{run_id}/cancel")
def cancel_run(app, request):
    """Ask a running coding agent to stop after its current tool."""
    return app.ide.cancel_run(request.params["run_id"])


@router.get("/api/ide/sessions/{session_id}")
def read_session(app, request):
    """One coding conversation with its transcript, plan and changes."""
    return {"session": app.ide.get_session(request.params["session_id"])}


@router.patch("/api/ide/sessions/{session_id}")
def rename_session(app, request):
    """Rename, pin or update the pinned context of a coding conversation."""
    return {"session": app.ide.rename_session(request.params["session_id"], request.json)}


@router.post("/api/ide/sessions/{session_id}/duplicate")
def duplicate_session(app, request):
    """Fork a coding conversation for another attempt."""
    return created({"session": app.ide.duplicate_session(request.params["session_id"])})


@router.delete("/api/ide/sessions/{session_id}")
def delete_session(app, request):
    """Delete a coding conversation, its saved runs and its snapshots."""
    app.ide.delete_session(request.params["session_id"])
    return {"deleted": True, "id": request.params["session_id"]}


@router.post("/api/ide/sessions/{session_id}/revert")
def revert_session(app, request):
    """Restore every file this conversation changed to its first snapshot."""
    return app.ide.revert_session(request.params["session_id"])
