"""Agent routes: the Agent Management Protocol (AMP) over HTTP.

The protocol is specified in ``mariana/agents/specs/AMP.md``; handlers here
only translate between HTTP and :class:`mariana.agents.manager.AgentManager`.
"""

from __future__ import annotations

from pathlib import Path

from ...agents import aml
from ...agents.aml import AMLValidationError
from ...agents.runtime import RunPayloadError
from ...agents.templates import get_template, list_templates
from ..errors import BadRequestError, ConflictError, NotFoundError
from ..http import EventStream, Response, Router, accepted, created

router = Router()
SPECS = Path(aml.__file__).resolve().parent / "specs"
MARKDOWN = "text/markdown; charset=utf-8"


def _agent(app, agent_id: str) -> dict:
    try:
        return app.agents.get_agent(agent_id)
    except ValueError as error:
        raise NotFoundError(str(error)) from error


def _conflict(action, *arguments):
    """Run a manager call whose ``ValueError`` means "not in this state"."""
    try:
        return action(*arguments)
    except RunPayloadError as error:
        raise BadRequestError(str(error)) from error
    except AMLValidationError as error:
        raise BadRequestError(str(error)) from error
    except ValueError as error:
        raise ConflictError(str(error)) from error


def _spec(name: str) -> Response:
    path = SPECS / name
    return Response.text(path.read_text(encoding="utf-8") if path.exists() else "", MARKDOWN)


@router.get("/api/agents")
def list_agents(app, request):
    """List saved agents with schedule and run status."""
    return {"agents": app.agents.list_agents()}


@router.get("/api/agents/runs")
def list_all_runs(app, request):
    """Runs across every agent, newest first."""
    return {"runs": app.agents.list_runs(None, request.int_arg("limit", 30, 1, 200))}


@router.post("/api/agents")
def save_agent(app, request):
    """Create or replace an agent; the body is the AML document."""
    return created(app.agents.save_agent(request.json))


@router.get("/api/agents/protocol")
def read_protocol(app, request):
    """Discovery: node types, abilities, operations, events."""
    return app.agents.protocol_metadata()


@router.get("/api/agents/templates")
def list_agent_templates(app, request):
    """Starter agents to copy."""
    return {"templates": list_templates()}


@router.get("/api/agents/templates/{id}")
def read_template(app, request):
    """One starter agent as an AML document."""
    try:
        return get_template(request.params["id"])
    except ValueError as error:
        raise NotFoundError(str(error)) from error


@router.get("/api/agents/spec/aml")
def read_aml_spec(app, request):
    """The AML specification as markdown."""
    return _spec("AML.md")


@router.get("/api/agents/spec/amp")
def read_amp_spec(app, request):
    """The AMP specification as markdown."""
    return _spec("AMP.md")


@router.post("/api/agents/validate")
def validate_agent(app, request):
    """Validate AML without saving; always answers 200 with ``ok``."""
    body = request.json
    try:
        return app.agents.validate(body)
    except Exception as error:  # noqa: BLE001 - validation reports, it never fails
        return {"ok": False, "errors": [str(error)], "warnings": [], "issues": [
            {"level": "error", "message": str(error), "node": None, "edge": None, "field": None},
        ], "agent": body}


@router.post("/api/agents/import")
def import_agent(app, request):
    """Import AML given as JSON or YAML text."""
    return app.agents.import_aml(str(request.json.get("text", "")))


@router.post("/api/agents/generate")
def generate_agent(app, request):
    """Draft an agent from a plain-language description using a chat model."""
    body = request.json
    return app.agents.generate_agent(str(body.get("description", "")), body.get("model") or None)


@router.get("/api/agents/runs/{run_id}")
def read_run(app, request):
    """Run state with the events after ``since``."""
    try:
        return app.agents.get_run(request.params["run_id"], request.int_arg("since", 0, 0, 10**9))
    except ValueError as error:
        raise NotFoundError(str(error)) from error


@router.get("/api/agents/runs/{run_id}/events")
def stream_run(app, request):
    """Server-sent run snapshots after ``since``; ends with the final state."""
    try:
        events = app.agents.stream_run(
            request.params["run_id"], request.int_arg("since", 0, 0, 10**9),
        )
    except ValueError as error:
        raise NotFoundError(str(error)) from error
    return EventStream(events)


@router.get("/api/agents/runs/{run_id}/swarm")
def read_swarm(app, request):
    """The live or last swarm state of a run: leader, members, messages, memory and reviews."""
    try:
        return app.agents.get_swarm(request.params["run_id"])
    except ValueError as error:
        raise NotFoundError(str(error)) from error


@router.post("/api/agents/runs/{run_id}/swarm/messages")
def post_swarm_message(app, request):
    """Queue an operator message for the leader of a live swarm."""
    return _conflict(app.agents.inject_swarm_message, request.params["run_id"], request.json)


@router.post("/api/agents/runs/{run_id}/step")
def step_run(app, request):
    """Run one node of a paused debug run."""
    return _conflict(app.agents.step_run, request.params["run_id"])


@router.post("/api/agents/runs/{run_id}/resume")
def resume_run(app, request):
    """Continue a paused debug run to the next breakpoint."""
    return _conflict(app.agents.resume_run, request.params["run_id"])


@router.post("/api/agents/runs/{run_id}/cancel")
def cancel_run(app, request):
    """Cancel an active run."""
    return _conflict(app.agents.cancel_run, request.params["run_id"])


@router.post("/api/agents/runs/{run_id}/input")
def provide_run_input(app, request):
    """Answer a ``human`` node that is waiting for approval or a value."""
    return _conflict(app.agents.provide_input, request.params["run_id"], request.json.get("value"))


@router.post("/api/agents/runs/{run_id}/breakpoints")
def set_run_breakpoints(app, request):
    """Replace the breakpoints of a live debug run."""
    points = request.json.get("breakpoints")
    if not isinstance(points, list):
        raise BadRequestError("breakpoints must be a list of node ids")
    return _conflict(app.agents.set_breakpoints, request.params["run_id"], points)


@router.get("/api/agents/{id}")
def read_agent(app, request):
    """One agent document and its active run, if any."""
    agent = _agent(app, request.params["id"])
    active = app.agents._active_for_agent(request.params["id"])
    return {"agent": agent, "_meta": {"active_run": active.id if active else None}}


@router.post("/api/agents/{id}/clone")
def clone_agent(app, request):
    """Duplicate an agent under a new id, with its schedule disabled."""
    _agent(app, request.params["id"])
    body = request.json
    new_id = body.get("id")
    new_name = body.get("name")
    if new_id is not None and not isinstance(new_id, str):
        raise BadRequestError("id must be a string when given")
    if new_name is not None and not isinstance(new_name, str):
        raise BadRequestError("name must be a string when given")
    return created(_conflict(app.agents.clone_agent, request.params["id"], new_id, new_name))


@router.delete("/api/agents/{id}")
def delete_agent(app, request):
    """Delete an agent and its run history."""
    if not _conflict(app.agents.delete_agent, request.params["id"]):
        raise NotFoundError("Agent not found")
    return {"deleted": True, "id": request.params["id"]}


@router.get("/api/agents/{id}/runs")
def list_runs(app, request):
    """Recent runs of an agent."""
    return {"runs": app.agents.list_runs(request.params["id"], request.int_arg("limit", 30, 1, 200))}


@router.post("/api/agents/{id}/runs")
def create_run(app, request):
    """Run, test (``dry_run``) or debug (``debug`` + ``breakpoints``) an agent."""
    _agent(app, request.params["id"])
    return accepted(_conflict(app.agents.create_run, request.params["id"], request.json))


@router.get("/api/agents/{id}/schedule")
def read_schedule(app, request):
    """An agent's schedule with its next and last run."""
    try:
        return app.agents.schedule_info(request.params["id"])
    except ValueError as error:
        raise NotFoundError(str(error)) from error


@router.post("/api/agents/{id}/schedule")
def set_schedule(app, request):
    """Enable or disable an agent's schedule."""
    enabled = request.json.get("enabled", False)
    if not isinstance(enabled, bool):
        raise BadRequestError("enabled must be true or false")
    return app.agents.set_schedule_enabled(request.params["id"], enabled)


@router.post("/api/agents/{id}/trigger")
def trigger_agent(app, request):
    """Run now with the schedule's input (or the input given)."""
    agent = _agent(app, request.params["id"])
    body = request.json
    schedule = agent.get("schedule")
    default_input = schedule.get("input", {}) if isinstance(schedule, dict) else {}
    payload = {
        "input": body.get("input", default_input),
        "trigger": "manual", "dry_run": body.get("dry_run", False),
    }
    return accepted(_conflict(app.agents.create_run, request.params["id"], payload))
