"""Agent runtime manager: CRUD, runs, debugger, subagents and schedules."""

from __future__ import annotations

import copy
import os
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

from . import aml
from .aml import AMLValidationError, next_run_at, summarize, validate_agent
from .runtime import FINAL_STATUSES, LiveRun, RunCancelled, RunPayloadError, execute_agent
from .store import AgentStore
from .swarm import MESSAGE_KINDS, SWARM_STRATEGIES

ACTIVE_STATUSES = frozenset({"queued", "running", "paused"})


class AgentManager:
    def __init__(self, root: Path, gateway, settings=None):
        self.root = Path(root).resolve()
        self.gateway = gateway
        self._settings = settings if callable(settings) else (lambda: settings or {})
        self.store = AgentStore(self.root)
        self.workspace = self.store.workspace
        self._runs: dict[str, LiveRun] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._run_lock = threading.Lock()
        self._last_save: dict[str, float] = {}
        self._close_lock = threading.Lock()
        self._closed = False
        self._stop = threading.Event()
        try:
            self.store.settle_interrupted_runs()
        except Exception:  # noqa: BLE001 - unreadable history must not block Studio startup
            pass
        self._scheduler = threading.Thread(target=self._scheduler_loop, name="aml-scheduler", daemon=True)
        self._scheduler.start()

    @property
    def os_access(self) -> bool:
        """Server-wide opt-in for shell commands and local-network targets."""
        if (os.environ.get("MARIANA_AGENT_ALLOW_OS") == "1"
                or os.environ.get("MARIANA_AGENT_ALLOW_SHELL") == "1"):
            return True
        try:
            return bool(self._settings().get("allow_os_access"))
        except Exception:  # noqa: BLE001 - a settings failure keeps the safe default
            return False

    # -- agents ----------------------------------------------------------
    def known_ids(self) -> set[str]:
        return {agent["id"] for agent in self.store.list_agents()}

    def list_agents(self) -> list[dict]:
        result = []
        states = self.store.all_schedules()
        for agent in self.store.list_agents():
            item = summarize(agent)
            state = states.get(agent["id"], {})
            item["next_run"] = state.get("next_run")
            item["last_run"] = state.get("last_run")
            item["last_error"] = state.get("last_error")
            item["active"] = self._active_for_agent(agent["id"]) is not None
            result.append(item)
        return result

    def get_agent(self, agent_id: str) -> dict:
        return self.store.get_agent(agent_id)

    def validate(self, document: dict) -> dict:
        try:
            return validate_agent(document, self.known_ids())
        except AMLValidationError as error:
            issues = [{"level": "error", "message": message, "node": None, "edge": None, "field": None}
                      for message in error.errors]
            return {"ok": False, "errors": error.errors, "warnings": error.warnings,
                    "issues": issues, "agent": document}

    def save_agent(self, document: dict) -> dict:
        result = self.validate(document)
        if not result["ok"]:
            raise AMLValidationError(result["errors"], result["warnings"])
        agent = result["agent"]
        try:
            existing = self.store.get_agent(agent["id"])
            agent["created"] = existing.get("created", agent["created"])
        except ValueError:
            pass
        self.store.save_agent(agent)
        self.store.clear_schedule(agent["id"])
        self._ensure_schedule(agent)
        return {"agent": agent, "warnings": result["warnings"]}

    def delete_agent(self, agent_id: str) -> bool:
        active = self._active_for_agent(agent_id)
        if active is not None:
            raise ValueError("Cancel the active run before deleting this agent")
        return self.store.delete_agent(agent_id)

    def import_aml(self, text: str) -> dict:
        document = aml.parse_document(text)
        return self.save_agent(document)

    def clone_agent(self, agent_id: str, new_id: str | None = None,
                    new_name: str | None = None) -> dict:
        """Duplicate an agent under a fresh id, with its schedule disabled."""
        source = self.store.get_agent(agent_id)
        document = copy.deepcopy(source)
        document.pop("created", None)
        document.pop("updated", None)
        document["id"] = aml.slugify(new_id or f"{agent_id}-copy")
        document["name"] = str(new_name or f"{source.get('name') or agent_id} copy")[:120]
        if self._agent_exists(document["id"]):
            raise ValueError(f"Agent {document['id']} already exists; choose another id")
        if isinstance(document.get("schedule"), dict):
            document["schedule"]["enabled"] = False
        return self.save_agent(document)

    def _agent_exists(self, agent_id: str) -> bool:
        try:
            self.store.get_agent(agent_id)
        except ValueError:
            return False
        return True

    def generate_agent(self, description: str, model: str | None = None) -> dict:
        """Draft an agent from a description with a chat model; nothing is saved."""
        from .authoring import generate_agent

        return generate_agent(self.gateway, description, model, self.known_ids())

    # -- runs ------------------------------------------------------------
    def _active_for_agent(self, agent_id: str) -> LiveRun | None:
        with self._run_lock:
            for run in self._runs.values():
                if run.agent_id == agent_id and run.status in ACTIVE_STATUSES:
                    return run
        return None

    def list_runs(self, agent_id: str | None = None, limit: int = 30) -> list[dict]:
        runs = self.store.list_runs(agent_id, limit)
        with self._run_lock:
            for run in self._runs.values():
                if agent_id and run.agent_id != agent_id:
                    continue
                if any(item["id"] == run.id for item in runs):
                    continue
                runs.insert(0, {
                    "id": run.id, "agent_id": run.agent_id, "agent_name": run.agent_name,
                    "status": run.status, "trigger": run.trigger,
                    "dry_run": run.dry_run, "debug": run.debug, "steps": run.steps,
                    "error": run.error, "created": run.created, "started": run.started,
                    "ended": run.ended, "parent_run_id": run.parent_run_id,
                })
        runs.sort(key=lambda item: item.get("created") or 0, reverse=True)
        return runs[:max(1, limit)]

    def create_run(self, agent_id: str, payload: dict | None = None) -> dict:
        payload = dict(payload or {})
        agent = self.store.get_agent(agent_id)
        validation = validate_agent(agent, self.known_ids())
        if not validation["ok"]:
            raise AMLValidationError(validation["errors"], validation["warnings"])
        agent = validation["agent"]
        trigger = payload.get("trigger", "manual") or "manual"
        if trigger not in {"manual", "schedule", "subagent"}:
            raise RunPayloadError("Run trigger must be manual, schedule, or subagent")
        for key in ("debug", "dry_run"):
            if key in payload and not isinstance(payload[key], bool):
                raise RunPayloadError(f"{key} must be true or false")
        points = payload.get("breakpoints") or []
        if not isinstance(points, list):
            raise RunPayloadError("breakpoints must be a list of node ids")
        points = [str(item) for item in points]
        unknown = sorted(set(points) - {node["id"] for node in agent["nodes"]})
        if unknown:
            raise RunPayloadError(f"Unknown breakpoint node(s): {', '.join(unknown)}")
        if len(points) > 200:
            raise RunPayloadError("At most 200 breakpoints are allowed")
        payload["trigger"] = trigger
        payload["breakpoints"] = points
        with self._run_lock:
            if self._closed:
                raise ValueError("Studio is shutting down; try again after restart")
            if any(run.agent_id == agent_id and run.status in ACTIVE_STATUSES
                   for run in self._runs.values()):
                raise ValueError("This agent already has an active run")
            run = self._new_run_locked(agent, payload)
            thread = threading.Thread(
                target=self._execute_live, args=(run, agent, payload, 0),
                name=f"aml-run-{run.id}", daemon=True,
            )
            self._threads[run.id] = thread
        self._save_run(run)
        thread.start()
        return run.snapshot()

    def _new_run_locked(self, agent: dict, payload: dict,
                        parent_run_id: str | None = None) -> LiveRun:
        run_id = f"run-{int(time.time() * 1000):x}-{uuid.uuid4().hex[:6]}"
        run = LiveRun(run_id, agent, payload, parent_run_id=parent_run_id)
        run.on_event = lambda live: self._maybe_save(live)
        self._runs[run_id] = run
        return run

    def _make_run(self, agent: dict, payload: dict, parent_run_id: str | None = None) -> LiveRun:
        with self._run_lock:
            if self._closed:
                raise ValueError("Studio is shutting down")
            run = self._new_run_locked(agent, payload, parent_run_id)
        self._save_run(run)
        return run

    def _execute_live(self, run: LiveRun, agent: dict, payload: dict, depth: int) -> None:
        try:
            execute_agent(
                run, agent, self.gateway, self.root,
                invoke_child=self._invoke_child, depth=depth,
                os_access=self.os_access,
            )
        except Exception as error:  # noqa: BLE001 - always close a run
            if run.status in ACTIVE_STATUSES:
                if run.cancel_requested:
                    run.finish("cancelled", error="Run cancelled")
                else:
                    run.finish("failed", error=str(error))
        finally:
            try:
                self._save_run(run, force=True)
            except Exception:  # noqa: BLE001 - history persistence is best effort
                pass
            try:
                self.store.trim_runs(run.agent_id)
            except OSError:
                pass
            try:
                if run.trigger == "schedule" and run.parent_run_id is None:
                    self._update_schedule_after_run(run)
            except Exception:  # noqa: BLE001 - a schedule write must not strand the run
                pass
            with self._run_lock:
                if run.status in FINAL_STATUSES:
                    self._runs.pop(run.id, None)
                self._threads.pop(run.id, None)
            self._last_save.pop(run.id, None)

    def _invoke_child(self, agent_id: str, child_input, parent_run: LiveRun, depth: int):
        if depth > 3:
            raise ValueError("Agent nesting exceeds the maximum depth of 3")
        agent = self.store.get_agent(agent_id)
        validation = validate_agent(agent, self.known_ids())
        if not validation["ok"]:
            raise ValueError("Child agent is invalid: " + "; ".join(validation["errors"]))
        payload = {
            "input": child_input, "trigger": "subagent", "dry_run": parent_run.dry_run,
            "debug": False, "breakpoints": [],
        }
        child = self._make_run(validation["agent"], payload, parent_run_id=parent_run.id)
        self._execute_live(child, validation["agent"], payload, depth)
        if child.status != "completed":
            if parent_run.cancel_requested:
                raise RunCancelled()
            raise ValueError(f"Child agent {agent_id} {child.status}: {child.error or 'no output'}")
        return child.output

    def get_run(self, run_id: str, since: int = 0) -> dict:
        with self._run_lock:
            live = self._runs.get(run_id)
        if live is not None:
            return live.snapshot(since=since)
        snapshot = self.store.get_run(run_id)
        if since:
            snapshot["events"] = [
                event for event in snapshot.get("events", [])
                if isinstance(event, dict) and isinstance(event.get("seq"), int)
                and event["seq"] > since
            ]
        return snapshot

    def get_swarm(self, run_id: str) -> dict:
        """The latest swarm state of a run, live or from history."""
        with self._run_lock:
            live = self._runs.get(run_id)
        if live is not None:
            snapshot = live.swarm
            if not snapshot:
                raise ValueError("This run has not started a swarm node")
            return snapshot
        snapshot = self.store.get_run(run_id)
        swarm = snapshot.get("swarm")
        if not swarm:
            raise ValueError("This run has no swarm state")
        return swarm

    def inject_swarm_message(self, run_id: str, payload: dict) -> dict:
        """Queue an operator message for the leader of an active swarm."""
        text = payload.get("text") if isinstance(payload, dict) else None
        if not isinstance(text, str) or not text.strip():
            raise RunPayloadError("A swarm operator message needs some text")
        if len(text) > 2_000:
            raise RunPayloadError("A swarm operator message is limited to 2000 characters")
        live = self._live(run_id)
        coordinator = live.swarm_coordinator
        if coordinator is None:
            raise ValueError("This run does not currently have a live swarm")
        return coordinator.post_operator_message(text)

    def stream_run(self, run_id: str, since: int = 0):
        """Server-sent snapshots for one run; the last frame has a final status.

        A live run yields a fresh snapshot whenever an event arrives, plus an
        immediate snapshot so a client that just connected has full state.  A
        completed run is replayed from history in a single frame.
        """
        with self._run_lock:
            live = self._runs.get(run_id)
        if live is None:
            snapshot = self.store.get_run(run_id)
            snapshot["events"] = [
                event for event in snapshot.get("events", [])
                if isinstance(event, dict) and isinstance(event.get("seq"), int)
                and event["seq"] > max(0, since)
            ]
            snapshot["type"] = "snapshot"
            return iter([snapshot])

        def frame(cursor: int) -> dict:
            snapshot = live.snapshot(since=cursor)
            snapshot["type"] = "snapshot"
            return snapshot

        def snapshots():
            cursor = max(0, since)
            first = True
            while True:
                pending, finished = live.wait_events(cursor, timeout=0.0 if first else 15.0)
                if pending or first:
                    snapshot = frame(cursor)
                    if pending:
                        cursor = pending[-1]["seq"]
                    first = False
                    yield snapshot
                elif finished:
                    yield frame(cursor)
                if finished:
                    return
                if not pending:
                    yield None

        return snapshots()

    def step_run(self, run_id: str) -> dict:
        return self._control_run(run_id, "step")

    def resume_run(self, run_id: str) -> dict:
        return self._control_run(run_id, "resume")

    def cancel_run(self, run_id: str) -> dict:
        return self._control_run(run_id, "cancel")

    def provide_input(self, run_id: str, value) -> dict:
        """Answer the ``human`` node a run is waiting on."""
        live = self._live(run_id)
        live.provide_input(value)
        return live.snapshot()

    def set_breakpoints(self, run_id: str, node_ids) -> dict:
        """Replace the breakpoints of a live debug run."""
        live = self._live(run_id)
        live.set_breakpoints(node_ids)
        return live.snapshot()

    def _live(self, run_id: str) -> LiveRun:
        with self._run_lock:
            live = self._runs.get(run_id)
            if live is not None and live.status in ACTIVE_STATUSES:
                return live
        self.store.get_run(run_id)  # raises "Run not found" for an unknown id
        raise ValueError("Run is not active")

    def _control_run(self, run_id: str, action: str) -> dict:
        with self._run_lock:
            live = self._runs.get(run_id)
            if live is not None and live.status not in ACTIVE_STATUSES:
                live = None
        if live is None:
            snapshot = self.store.get_run(run_id)
            if snapshot.get("status") in ACTIVE_STATUSES:
                raise ValueError("Run is not attached to this server session")
            raise ValueError("Run is not active")
        getattr(live, action)()
        return live.snapshot()

    def _save_run(self, run: LiveRun, force: bool = False) -> None:
        now = time.monotonic()
        last = self._last_save.get(run.id, 0.0)
        if not force and now - last < 0.2:
            return
        self._last_save[run.id] = now
        try:
            self.store.save_run(run.snapshot())
        except Exception:  # noqa: BLE001 - a run must not fail because history is unwritable
            pass

    def _maybe_save(self, run: LiveRun) -> None:
        self._save_run(run)

    # -- schedules -------------------------------------------------------
    def schedule_info(self, agent_id: str) -> dict:
        agent = self.store.get_agent(agent_id)
        state = self.store.schedule_state(agent_id)
        schedule = agent.get("schedule")
        return {
            "schedule": schedule if isinstance(schedule, dict) else {},
            "next_run": state.get("next_run"),
            "last_run": state.get("last_run"),
            "last_error": state.get("last_error"),
            "server_time": datetime.now().isoformat(timespec="seconds"),
        }

    def set_schedule_enabled(self, agent_id: str, enabled: bool) -> dict:
        agent = self.store.get_agent(agent_id)
        schedule = agent.get("schedule")
        if not isinstance(schedule, dict):
            schedule = {}
        schedule["enabled"] = bool(enabled)
        agent["schedule"] = schedule
        saved = self.save_agent(agent)
        return self.schedule_info(saved["agent"]["id"])

    def _ensure_schedule(self, agent: dict) -> None:
        schedule = agent.get("schedule")
        if not isinstance(schedule, dict) or not schedule.get("enabled"):
            self.store.clear_schedule(agent["id"])
            return
        state = self.store.schedule_state(agent["id"])
        try:
            next_run = next_run_at(agent, datetime.now(),
                                   _parse_dt(state.get("last_run")))
        except Exception:  # noqa: BLE001 - invalid schedules are rejected by validation
            next_run = None
        self.store.set_schedule_state(agent["id"], {
            "last_run": state.get("last_run"),
            "next_run": next_run.isoformat(timespec="seconds") if next_run else None,
        })

    def _update_schedule_after_run(self, run: LiveRun) -> None:
        agent_id = run.agent_id
        try:
            agent = self.store.get_agent(agent_id)
        except ValueError:
            return
        if not isinstance(agent.get("schedule"), dict) or not agent["schedule"].get("enabled"):
            self.store.clear_schedule(agent_id)
            return
        now = datetime.now()
        try:
            next_run = next_run_at(agent, now, now)
        except Exception:  # noqa: BLE001
            next_run = None
        self.store.set_schedule_state(agent_id, {
            "last_run": now.isoformat(timespec="seconds"),
            "next_run": next_run.isoformat(timespec="seconds") if next_run else None,
            "last_error": None if run.status == "completed" else (run.error or run.status),
        })

    def _scheduler_loop(self) -> None:
        while not self._stop.wait(5.0):
            try:
                self._scheduler_tick()
            except Exception:  # noqa: BLE001 - a bad agent must not stop the scheduler
                continue

    def _scheduler_tick(self) -> None:
        now = datetime.now()
        for agent in self.store.list_agents():
            schedule = agent.get("schedule")
            if not isinstance(schedule, dict) or not schedule.get("enabled"):
                self.store.clear_schedule(agent["id"])
                continue
            state = self.store.schedule_state(agent["id"])
            next_run = _parse_dt(state.get("next_run"))
            if next_run is None:
                try:
                    next_run = next_run_at(agent, now, _parse_dt(state.get("last_run")))
                except Exception:  # noqa: BLE001
                    continue
                self.store.set_schedule_state(agent["id"], {
                    "last_run": state.get("last_run"),
                    "next_run": next_run.isoformat(timespec="seconds") if next_run else None,
                })
            if next_run is None or next_run > now:
                continue
            if self._active_for_agent(agent["id"]):
                try:
                    later = next_run_at(agent, now, now)
                except Exception:  # noqa: BLE001
                    later = None
                self.store.set_schedule_state(agent["id"], {
                    "next_run": later.isoformat(timespec="seconds") if later else None,
                })
                continue
            payload = {
                "input": schedule.get("input") or {},
                "trigger": "schedule",
            }
            try:
                self.create_run(agent["id"], payload)
            except Exception as error:  # noqa: BLE001 - record why the schedule did not start
                self._advance_schedule(agent, now, error=str(error) or type(error).__name__,
                                       started=False)
                continue
            self._advance_schedule(agent, now)

    def _advance_schedule(self, agent: dict, now: datetime, error: str | None = None,
                          started: bool = True) -> None:
        """Move a schedule past a fired slot, recording why it could not run."""
        try:
            later = next_run_at(agent, now, now)
        except Exception:  # noqa: BLE001
            later = None
        state = self.store.schedule_state(agent["id"])
        self.store.set_schedule_state(agent["id"], {
            "last_run": now.isoformat(timespec="seconds") if started else state.get("last_run"),
            "next_run": later.isoformat(timespec="seconds") if later else None,
            "last_error": error,
        })

    def close(self) -> None:
        """Cancel active runs and stop the scheduler without leaving work behind."""
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        self._stop.set()
        with self._run_lock:
            runs = [run for run in self._runs.values() if run.status in ACTIVE_STATUSES]
            threads = [self._threads.get(run.id) for run in runs]
        for run in runs:
            try:
                run.cancel()
            except Exception:  # noqa: BLE001 - shutdown must reach every run
                pass
        deadline = time.monotonic() + 5.0
        for thread in threads:
            if (thread is None or thread is threading.current_thread()
                    or thread.ident is None):
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            thread.join(timeout=min(2.0, remaining))
        for run in runs:
            if run.status in ACTIVE_STATUSES:
                run.finish("cancelled", error="Studio shut down before the run finished")
                self._save_run(run, force=True)
            with self._run_lock:
                self._runs.pop(run.id, None)
                self._threads.pop(run.id, None)
            self._last_save.pop(run.id, None)
        if self._scheduler is not threading.current_thread():
            self._scheduler.join(timeout=2.0)

    # -- protocol discovery ---------------------------------------------
    def protocol_metadata(self) -> dict:
        return {
            "protocol": "AMP",
            "version": "1.10",
            "aml_version": aml.AML_VERSION,
            "tool_calling": {
                "formats": ["json", "dsml", "native"],
                "native_backends": ["openai-compatible", "anthropic", "ollama"],
                "fallback": "prompted JSON",
                "note": ("Every format uses the same allow-list, sandbox and "
                         "tool_call/tool_result events."),
            },
            "guardrails": {
                "assert": "A safe expression must be true; message explains the failure.",
                "validate": "A bounded JSON schema subset validates structured output; "
                            "mode fail stops the node, mode warn exposes valid/errors.",
                "model_fallback": "llm and agent nodes accept fallback_model for provider failures.",
            },
            "transport": "HTTP/JSON",
            "base_path": "/api/agents",
            "specifications": {
                "aml": "/api/agents/spec/aml",
                "amp": "/api/agents/spec/amp",
            },
            "workspace": str(self.workspace.relative_to(self.root)),
            "server_time": datetime.now().isoformat(timespec="seconds"),
            "operations": [
                {"method": "GET", "path": "/api/agents/protocol", "summary": "Discovery metadata"},
                {"method": "GET", "path": "/api/agents/spec/aml", "summary": "AML specification (markdown)"},
                {"method": "GET", "path": "/api/agents/spec/amp", "summary": "AMP specification (markdown)"},
                {"method": "GET", "path": "/api/agents", "summary": "List agents"},
                {"method": "POST", "path": "/api/agents", "summary": "Create or update an agent"},
                {"method": "POST", "path": "/api/agents/import", "summary": "Import AML text"},
                {"method": "POST", "path": "/api/agents/validate", "summary": "Validate AML without saving"},
                {"method": "GET", "path": "/api/agents/{id}", "summary": "Read an agent"},
                {"method": "POST", "path": "/api/agents/{id}/clone", "summary": "Duplicate an agent"},
                {"method": "DELETE", "path": "/api/agents/{id}", "summary": "Delete an agent"},
                {"method": "GET", "path": "/api/agents/{id}/runs", "summary": "List recent runs"},
                {"method": "GET", "path": "/api/agents/runs", "summary": "List runs across agents"},
                {"method": "POST", "path": "/api/agents/{id}/runs", "summary": "Run, test, or debug"},
                {"method": "GET", "path": "/api/agents/runs/{run_id}", "summary": "Read run state and events"},
                {"method": "GET", "path": "/api/agents/runs/{run_id}/events",
                 "summary": "Server-sent run snapshots"},
                {"method": "POST", "path": "/api/agents/runs/{run_id}/step", "summary": "Step one paused node"},
                {"method": "POST", "path": "/api/agents/runs/{run_id}/resume", "summary": "Resume a paused run"},
                {"method": "POST", "path": "/api/agents/runs/{run_id}/cancel", "summary": "Cancel a run"},
                {"method": "POST", "path": "/api/agents/runs/{run_id}/input",
                 "summary": "Answer a human node that is waiting"},
                {"method": "POST", "path": "/api/agents/runs/{run_id}/breakpoints",
                 "summary": "Replace the breakpoints of a live debug run"},
                {"method": "POST", "path": "/api/agents/generate",
                 "summary": "Draft an agent from a description (not saved)"},
                {"method": "GET", "path": "/api/agents/runs/{run_id}/swarm",
                 "summary": "Read the live or last swarm state of a run"},
                {"method": "POST", "path": "/api/agents/runs/{run_id}/swarm/messages",
                 "summary": "Queue an operator message for a live swarm leader"},
                {"method": "GET", "path": "/api/agents/{id}/schedule", "summary": "Read schedule state"},
                {"method": "POST", "path": "/api/agents/{id}/schedule", "summary": "Enable/disable schedule"},
                {"method": "POST", "path": "/api/agents/{id}/trigger", "summary": "Run now with schedule input"},
            ],
            "node_types": {
                key: {"label": value["label"], "description": value["description"],
                      "category": value["category"], "icon": value["icon"],
                      "fields": value["fields"], "defaults": aml.DEFAULT_NODE_DATA.get(key, {})}
                for key, value in aml.NODE_TYPES.items()
            },
            "common_fields": aml.COMMON_FIELDS,
            "field_types": list(aml.FIELD_TYPES),
            "abilities": aml.ABILITY_TYPES,
            "run_payload": {
                "input": "any JSON value passed to the start node",
                "dry_run": "boolean; mock LLM calls and skip side effects",
                "debug": "boolean; pause at breakpoints",
                "breakpoints": "array of node ids",
                "trigger": "manual, schedule, or subagent",
            },
            "run_statuses": ["queued", "running", "paused", "completed", "failed", "cancelled"],
            "event_types": [
                "run_started", "node_started", "node_input", "node_finished", "node_failed",
                "assistant_delta", "reasoning_delta",
                "tool_call", "tool_result", "plan", "input_requested", "input_received",
                "input_timeout", "breakpoint", "resumed", "log", "validation",
                "run_completed", "run_failed", "run_cancelled",
                "swarm_start", "swarm_plan", "swarm_leader", "swarm_round",
                "swarm_agent_start", "swarm_agent_step", "swarm_agent_delta",
                "swarm_agent_status", "swarm_agent_tool", "swarm_agent_plan",
                "swarm_agent_end",
                "swarm_message", "swarm_memory", "swarm_review", "swarm_context", "swarm_end",
            ],
            "swarm": {
                "node_type": "swarm",
                "strategies": list(SWARM_STRATEGIES),
                "strategy_labels": aml.SWARM_STRATEGY_LABELS,
                "message_kinds": list(MESSAGE_KINDS),
                "policy": dict(aml.DEFAULT_SWARM),
                "limits": aml.DEFAULT_SWARM,
                "leader_protocol": {
                    "plan": '{"strategy", "rounds", "success_criteria", "agents": '
                            '[{"id", "label", "role", "task", "instruction", "tools", '
                            '"writes", "read_only", "context_chars", "depends_on", '
                            '"plan"}], "notes"}',
                    "reviews": '{"reviews": [{"reviewer", "target", "focus"}], "reason"}',
                    "continue": '{"continue": true|false, "reason", "gaps": ["..."]}',
                    "synthesis": '{"final", "summary", "warnings", "unmet_criteria", '
                                  '"confidence"}',
                },
                "member_protocol": {
                    "tool": '{"tool": "<id>", "input": {...}}',
                    "calls": '{"calls": [{"tool", "input"}]}',
                    "say": '{"say": {"to": "leader|all|<id>", "kind": "<message kind>", '
                           '"text": "...", "reply_to": "<id>"}}',
                    "memory": '{"memory": {"action": "set|get|list|delete", "key", "value"}}',
                    "context": '{"context": {"action": "get|find|release|status", '
                               '"handle"|"key"|"peer"|"query", "chars", "offset"}}',
                    "review": '{"review": {"target", "score": 1-5, "strengths", "issues", '
                              '"suggestions", "summary"}}',
                    "plan": '{"plan": [{"text", "status"}]}',
                    "final": '{"final": "<complete findings>"}',
                },
                "reliability": {
                    "retries": "Read-only tool failures with a temporary error are retried once; "
                               "writes are never retried automatically.",
                    "duplicates": "A repeated identical call earns an explicit hint instead of a loop.",
                    "plan_check": "A member that finishes with open plan steps is asked once to "
                                  "complete or justify them; the node disables this with reflection=false.",
                    "time_budget": "Tool rounds stop at the work deadline (the last 10% of "
                                   "max_seconds, at least 5 s and at most 90 s); members answer "
                                   "from the work already collected, peer review and later "
                                   "waves are skipped, and the leader still synthesises. A run "
                                   "that reaches the hard deadline returns partial work with "
                                   "timed_out=true instead of failing.",
                },
                "context": {
                    "pool": "Run-wide extra prompt characters members may borrow",
                    "grant": "Granted only while the pool has free space; released when a phase ends",
                    "bank": "Long tool results, peer results and compacted turns stay readable via ctx-… handles",
                    "actions": ["get", "find", "release", "status"],
                    "sources": ["handle", "key", "peer", "query"],
                    "slicing": "A get returns chars from offset; the reply names the next offset",
                },
                "events": [
                    "swarm_start", "swarm_plan", "swarm_leader", "swarm_round",
                    "swarm_agent_start", "swarm_agent_step", "swarm_agent_delta",
                    "swarm_agent_status", "swarm_agent_tool", "swarm_agent_plan",
                    "swarm_agent_end",
                    "swarm_message", "swarm_memory", "swarm_review", "swarm_context", "swarm_end",
                ],
                "security": {
                    "ipc": "swarm-scoped only; recipients, kinds, sizes, quotas and reply_to are checked",
                    "messages": "messages are data and are never executed as tool calls",
                    "memory": "run-scoped, keyed, size-capped and never persisted by the runtime",
                    "context": "run-scoped and size-capped; reads are grants, never a tool call",
                    "writes": "members may write only when the node and swarm.allow_write both allow it; "
                              "write calls are serialized and write scopes are enforced",
                    "operator": "messages are accepted only for a live swarm and are limited to 2000 chars",
                },
            },
            "run_state": {
                "vars": "flow variables assigned so far",
                "awaiting_input": "the question a human node is waiting on, or null",
                "nodes": "status per node id: pending, running, paused, ok, error",
                "swarm": "live swarm state while a swarm node runs, or the last snapshot",
            },
            "polling": {
                "run": "GET /api/agents/runs/{run_id}?since=<event_seq>",
                "stream": "GET /api/agents/runs/{run_id}/events?since=<event_seq> (server-sent events)",
                "interval_ms": 400,
            },
            "security": {
                "bind": "127.0.0.1 by default; do not expose Studio to untrusted networks",
                "files": "agent-workspace/files",
                "shell": "disabled unless MARIANA_AGENT_ALLOW_SHELL=1",
                "shell_enabled": os.environ.get("MARIANA_AGENT_ALLOW_SHELL") == "1",
                "sqlite": "read-only unless the ability config sets write=true; ATTACH is denied",
                "git": "fixed read-only status, diff and log commands; no shell string",
                "private_http": "blocked unless the ability sets allow_private; redirects and resolved peers are checked",
                "swarm_ipc": "per-run mailboxes only; no network transport, no cross-run delivery, known recipients only",
                "swarm_write": "opt-in through swarm.allow_write plus the node write flag; serialized",
            },
        }


def _parse_dt(value) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None
