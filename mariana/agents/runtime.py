"""AML flow interpreter with live events, breakpoints and test mode."""

from __future__ import annotations

import copy
import json
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

from .aml import ABILITY_TYPES, HUMAN_KINDS, render_data, render_template
from .swarm import SwarmCoordinator, SwarmError, SwarmLimits, snapshot_due
from .toolcalls import extract_action_object, parse_dsml_tool_calls
from .tools import execute_ability, json_safe

MAX_AGENT_ITERATIONS = 40
OBSERVATION_CHARS = 6000
EMPTY_REPLIES = frozenset({"", "(No readable text generated)"})
FINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})
DELTA_EVENTS = frozenset({"assistant_delta", "swarm_agent_delta"})
_APPROVE_WORDS = {"true": True, "yes": True, "approve": True, "approved": True, "y": True, "ok": True,
                  "false": False, "no": False, "reject": False, "rejected": False, "n": False}


class RunCancelled(Exception):
    pass


class RunPayloadError(ValueError):
    """A malformed run request, as opposed to a run-state conflict."""


class LiveRun:
    """Thread-safe live state for one agent execution."""

    def __init__(self, run_id: str, agent: dict, payload: dict, parent_run_id: str | None = None):
        self.id = run_id
        self.agent_id = agent["id"]
        self.agent_name = agent.get("name", agent["id"])
        self.parent_run_id = parent_run_id
        self.trigger = str(payload.get("trigger") or "manual")
        self.input = json_safe(payload.get("input", {}))
        self.debug = bool(payload.get("debug", False))
        self.dry_run = bool(payload.get("dry_run", False))
        self.breakpoints = set(str(item) for item in payload.get("breakpoints") or [])
        self.status = "queued"
        self.output = None
        self.error = None
        self.current_node = None
        self.waiting_node = None
        self.events: list[dict] = []
        self.nodes = {node["id"]: "pending" for node in agent.get("nodes", [])}
        self.steps = 0
        self.created = time.time()
        self.started = None
        self.ended = None
        self.updated = self.created
        self.cancel_requested = False
        self.paused = False
        self.single_step = False
        self.condition = threading.Condition()
        self.paused_started = None
        self.paused_total = 0.0
        self.vars: dict = {}
        self.awaiting_input: dict | None = None
        self.swarm: dict | None = None
        self.swarm_coordinator = None
        self.swarm_snapshot_at = 0.0
        self._input_ready = False
        self._input_value = None
        self._seq = 0
        self._max_output = int(agent.get("settings", {}).get("max_output_chars", 60000))
        settings = agent.get("settings") or {}
        try:
            self.timeout = max(1.0, float(settings.get("timeout", 120)))
        except (TypeError, ValueError):
            self.timeout = 120.0
        try:
            self.max_steps = max(1, int(settings.get("max_steps", 100)))
        except (TypeError, ValueError):
            self.max_steps = 100
        self.stop = threading.Event()

    # -- event & state ---------------------------------------------------
    def event(self, kind: str, **fields) -> None:
        with self.condition:
            self._seq += 1
            event = {
                "seq": self._seq,
                "at": datetime.now().isoformat(timespec="milliseconds"),
                "type": kind,
            }
            event.update({key: json_safe(value, self._max_output) for key, value in fields.items()})
            self.events.append(event)
            if len(self.events) > 800:
                self._trim_events()
            self.updated = time.time()
            if kind == "node_started":
                self.nodes[fields.get("node")] = "running"
                self.current_node = fields.get("node")
            elif kind == "node_finished":
                self.nodes[fields.get("node")] = "ok"
            elif kind == "node_failed":
                self.nodes[fields.get("node")] = "error"
            elif kind in {"breakpoint", "paused", "input_requested"}:
                self.nodes[fields.get("node")] = "paused"
                self.current_node = fields.get("node")
            elif kind in {"resumed", "stepped", "input_received", "input_timeout"}:
                self.nodes[fields.get("node")] = "running"
            self.condition.notify_all()
        callback = getattr(self, "on_event", None)
        if callback:
            try:
                callback(self)
            except Exception:  # noqa: BLE001 - a failed save must not fail the run
                pass

    def _trim_events(self) -> None:
        """Drop old streamed deltas before control events a reconnect needs."""
        excess = len(self.events) - 700
        if excess <= 0:
            return
        dropped = set()
        for event in self.events:
            if excess <= 0:
                break
            if event.get("type") in DELTA_EVENTS:
                dropped.add(event["seq"])
                excess -= 1
        if dropped:
            self.events = [event for event in self.events if event["seq"] not in dropped]
        else:
            del self.events[:200]

    def snapshot(self, since: int = 0, include_events: bool = True) -> dict:
        with self.condition:
            data = {
                "id": self.id,
                "agent_id": self.agent_id,
                "agent_name": self.agent_name,
                "parent_run_id": self.parent_run_id,
                "status": self.status,
                "trigger": self.trigger,
                "input": self.input,
                "output": self.output,
                "error": self.error,
                "current_node": self.current_node,
                "waiting_node": self.waiting_node,
                "paused": self.paused,
                "debug": self.debug,
                "dry_run": self.dry_run,
                "breakpoints": sorted(self.breakpoints),
                "steps": self.steps,
                "nodes": dict(self.nodes),
                "vars": json_safe(self.vars, 20000),
                "awaiting_input": dict(self.awaiting_input) if self.awaiting_input else None,
                "swarm": json_safe(self.swarm, 160_000) if self.swarm else None,
                "created": self.created,
                "started": self.started,
                "ended": self.ended,
                "updated": self.updated,
                "elapsed": self.active_elapsed_locked(),
                "event_count": self._seq,
            }
            if include_events:
                data["events"] = [event for event in self.events if event["seq"] > since]
            return data

    def wait_events(self, since: int, timeout: float = 1.0) -> tuple[list[dict], bool]:
        """Wait for events after ``since``; returns ``(events, finished)``."""
        with self.condition:
            pending = [event for event in self.events if event["seq"] > since]
            finished = self.status in FINAL_STATUSES
            if not pending and not finished:
                self.condition.wait(timeout)
                pending = [event for event in self.events if event["seq"] > since]
                finished = self.status in FINAL_STATUSES
            return pending, finished

    def active_elapsed_locked(self) -> float:
        if self.started is None:
            return 0.0
        end = self.ended or time.time()
        paused = self.paused_total
        if self.paused_started:
            paused += max(0.0, time.time() - self.paused_started)
        return max(0.0, end - self.started - paused)

    def active_elapsed(self) -> float:
        with self.condition:
            return self.active_elapsed_locked()

    # -- lifecycle -------------------------------------------------------
    def start(self) -> None:
        with self.condition:
            if self.status in FINAL_STATUSES:
                return
            self.status = "running"
            self.started = time.time()
        self.event("run_started", message="Run started", dry_run=self.dry_run, debug=self.debug)

    def is_final(self) -> bool:
        with self.condition:
            return self.status in FINAL_STATUSES

    def finish(self, status: str, output=None, error: str | None = None) -> None:
        with self.condition:
            if self.status in FINAL_STATUSES:
                return
            self.status = status
            self.output = json_safe(output, self._max_output)
            self.error = error
            self.ended = time.time()
            self.updated = self.ended
            self.stop.set()
            if self.paused_started:
                self.paused_total += max(0.0, self.ended - self.paused_started)
                self.paused_started = None
            self.paused = False
            self.single_step = False
            self.awaiting_input = None
            self.condition.notify_all()
        self.event(f"run_{status}", message=error or status.title(), output=self.output)

    # -- debug controls --------------------------------------------------
    def before_node(self, node_id: str) -> None:
        if not self.debug:
            self._check_cancel()
            return
        paused_here = False
        while True:
            with self.condition:
                if self.cancel_requested:
                    raise RunCancelled()
                if not (node_id in self.breakpoints or self.single_step):
                    break
                paused_here = True
                self.paused = True
                self.status = "paused"
                self.waiting_node = node_id
                self.paused_started = time.time()
                self.condition.notify_all()
            self.event(
                "breakpoint", node=node_id,
                message=f"Paused before {node_id}",
                breakpoint=node_id in self.breakpoints,
                single_step=self.single_step,
            )
            with self.condition:
                self.condition.wait_for(lambda: not self.paused or self.cancel_requested)
                if self.paused_started:
                    self.paused_total += max(0.0, time.time() - self.paused_started)
                    self.paused_started = None
                self.paused = False
                self.waiting_node = None
                cancelled = self.cancel_requested
                if not cancelled:
                    self.status = "running"
                self.condition.notify_all()
            if cancelled:
                raise RunCancelled()
            break
        if paused_here:
            self.event("resumed", node=node_id, message=f"Continue at {node_id}",
                       single_step=self.single_step)

    def pause_at(self, node_id: str, reason: str) -> None:
        with self.condition:
            self.paused = True
            self.waiting_node = node_id
            self.paused_started = time.time()
            self.condition.notify_all()
        self.event("breakpoint", node=node_id, message=reason)

    def resume(self) -> None:
        with self.condition:
            self.single_step = False
            self.paused = False
            self.condition.notify_all()

    def step(self) -> None:
        with self.condition:
            self.single_step = True
            self.paused = False
            self.condition.notify_all()

    def cancel(self) -> None:
        with self.condition:
            if self.status in FINAL_STATUSES:
                return
            self.cancel_requested = True
            self.paused = False
            self.stop.set()
            self.condition.notify_all()

    def set_breakpoints(self, node_ids) -> None:
        """Replace the breakpoints of a live debug run."""
        wanted = {str(item) for item in node_ids or []}
        with self.condition:
            unknown = sorted(wanted - set(self.nodes))
            if unknown:
                raise RunPayloadError(f"Unknown breakpoint node(s): {', '.join(unknown)}")
            if len(wanted) > 200:
                raise RunPayloadError("At most 200 breakpoints are allowed")
            self.breakpoints = wanted
            self.updated = time.time()

    # -- human input -----------------------------------------------------
    def request_input(self, node_id: str, spec: dict, timeout: float):
        """Block until a person answers; returns ``(value, timed_out)``.

        The run reports ``paused`` with ``awaiting_input`` set.  The wait does not
        count toward the run timeout, and only :meth:`provide_input` or
        :meth:`cancel` end it — the debugger's resume and step do not.
        """
        with self.condition:
            self.awaiting_input = {"node": node_id, **spec}
            self._input_ready, self._input_value = False, None
            self.status = "paused"
            self.waiting_node = node_id
            self.paused_started = time.time()
            self.condition.notify_all()
        self.event("input_requested", node=node_id, message=spec.get("prompt", ""), request=spec)
        with self.condition:
            answered = self.condition.wait_for(
                lambda: self._input_ready or self.cancel_requested,
                timeout=timeout if timeout and timeout > 0 else None,
            )
            self.paused_total += max(0.0, time.time() - (self.paused_started or time.time()))
            self.paused_started = None
            self.awaiting_input = None
            self.waiting_node = None
            cancelled = self.cancel_requested
            value = self._input_value
            if not cancelled:
                self.status = "running"
            self.condition.notify_all()
        if cancelled:
            raise RunCancelled()
        if not answered:
            self.event("input_timeout", node=node_id, message="No answer arrived in time")
            return None, True
        self.event("input_received", node=node_id, value=value, message="Answer received")
        return value, False

    def provide_input(self, value) -> None:
        """Answer the ``human`` node this run is waiting on."""
        with self.condition:
            if self.awaiting_input is None or self._input_ready:
                raise ValueError("This run is not waiting for input")
            self._input_value = normalize_answer(self.awaiting_input, value)
            self._input_ready = True
            self.condition.notify_all()

    def _check_cancel(self) -> None:
        self.check_limits()

    def set_swarm(self, snapshot: dict | None) -> None:
        """Publish the latest swarm state on the run for snapshots and the protocol."""
        bounded = json_safe(snapshot, 160_000) if snapshot else None
        with self.condition:
            self.swarm = bounded
            self.updated = time.time()
            self.condition.notify_all()

    def check_limits(self, timeout: float | None = None) -> None:
        """Raise when the run was cancelled or its active time exceeded ``timeout``."""
        with self.condition:
            if self.cancel_requested:
                raise RunCancelled()
            if (timeout is not None and self.started is not None
                    and self.active_elapsed_locked() > timeout):
                raise ValueError(f"Run exceeded timeout={timeout:.0f}s")


def _string(value) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def normalize_answer(spec: dict, value):
    """Check a human answer against the kind of question that was asked."""
    kind = spec.get("kind", "approve")
    if kind == "approve":
        if isinstance(value, bool):
            return value
        answer = _APPROVE_WORDS.get(str(value).strip().lower())
        if answer is None:
            raise ValueError("Answer with true (approve) or false (reject)")
        return answer
    if kind == "choice":
        choices = [str(choice) for choice in spec.get("choices") or []]
        if str(value) not in choices:
            raise ValueError(f"Choose one of: {', '.join(choices)}")
        return str(value)
    return "" if value is None else str(value)


def extract_json_object(text: str) -> dict | None:
    """Return the first balanced JSON object in ``text`` (code fences are fine), or ``None``."""
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def extract_json_value(text: str):
    """Return the first JSON object or array in text, or a bare JSON value."""
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character not in "{[":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        return value
    stripped = text.strip()
    if not stripped:
        return None
    try:
        return decoder.raw_decode(stripped)[0]
    except json.JSONDecodeError:
        return None


def _choose_next(node: dict, output, outgoing: dict, steps: int, run: LiveRun):
    edges = outgoing.get(node["id"], [])
    if not edges:
        return None
    if node["type"] == "condition":
        branch = "true" if bool(isinstance(output, dict) and output.get("result")) else "false"
        for edge in edges:
            if edge.get("when") == branch:
                return edge
        for edge in edges:
            if edge.get("when") == "always":
                return edge
        raise ValueError(
            f"Condition {node['id']} evaluated {branch} but has no matching branch"
        )
    return edges[0]


def execute_agent(run: LiveRun, agent: dict, gateway, root: Path,
                  invoke_child=None, depth: int = 0, os_access: bool = False) -> None:
    """Execute one AML flow until completion, failure, or cancellation.

    ``root`` is the Studio root; abilities are confined to ``<root>/agent-workspace``.
    """
    nodes = {node["id"]: node for node in agent.get("nodes", [])}
    outgoing: dict[str, list[dict]] = {}
    for edge in agent.get("edges", []):
        outgoing.setdefault(edge["from"], []).append(edge)
    starts = [node for node in agent.get("nodes", []) if node["type"] == "start"]
    if not starts:
        run.finish("failed", error="Agent has no start node")
        return
    start_id = starts[0]["id"]
    context = {
        "input": run.input if isinstance(run.input, dict) else {"value": run.input},
        "vars": copy.deepcopy(agent.get("vars") or {}),
        "last": None,
        "nodes": {},
        "agent": {"id": agent["id"], "name": agent.get("name"), "model": agent.get("model", "")},
        "run": {"id": run.id, "trigger": run.trigger, "dry_run": run.dry_run},
        "now": datetime.now().isoformat(timespec="seconds"),
    }
    abilities = {ability["id"]: ability for ability in agent.get("abilities", [])}
    settings = agent.get("settings", {})
    max_steps = run.max_steps
    timeout = run.timeout
    on_error = settings.get("on_error", "stop")
    run.vars = context["vars"]
    run.start()
    if run.is_final():
        return
    current_id = start_id
    try:
        while current_id:
            if run.steps >= max_steps:
                raise ValueError(f"Run exceeded max_steps={max_steps}")
            run.check_limits(timeout)
            node = nodes.get(current_id)
            if node is None:
                raise ValueError(f"Flow points at missing node {current_id}")
            run.before_node(node["id"])
            run.steps += 1
            run.event("node_started", node=node["id"], label=node.get("label"),
                      node_type=node["type"])
            data = node.get("data") or {}
            retries = max(0, min(5, int(data.get("retry", 0) or 0)))
            retry_delay = max(0.0, min(30.0, float(data.get("retry_delay", 1) or 1)))
            attempt = 0
            node_started = time.time()
            while True:
                try:
                    output = _run_node(
                        node, data, context, run, agent, gateway, root,
                        abilities, invoke_child, depth, os_access,
                    )
                    break
                except RunCancelled:
                    raise
                except Exception as error:  # noqa: BLE001 - node failures are data
                    run.check_limits(timeout)
                    if attempt < retries:
                        attempt += 1
                        run.event(
                            "log", node=node["id"], level="warning",
                            message=f"Retry {attempt}/{retries} after {error}",
                        )
                        if retry_delay and not run.dry_run:
                            _sleep_cancelable(run, retry_delay)
                        continue
                    if on_error == "continue":
                        output = {"error": str(error), "node": node["id"]}
                        run.event("node_failed", node=node["id"], error=str(error),
                                  message="Continuing after error")
                        break
                    run.event("node_failed", node=node["id"], error=str(error))
                    raise
            context["nodes"][node["id"]] = output
            context["last"] = output
            run.event("node_finished", node=node["id"], label=node.get("label"),
                      output=json_safe(output, 12000),
                      elapsed_ms=round((time.time() - node_started) * 1000))
            if node["type"] == "end":
                final = output
                if isinstance(output, dict) and "output" in output:
                    final = output["output"]
                run.finish("completed", output=final)
                return
            edge = _choose_next(node, output, outgoing, run.steps, run)
            if edge is None:
                if node["type"] != "end":
                    raise ValueError(f"Node {node['id']} has no outgoing edge")
                run.finish("completed", output=output)
                return
            current_id = edge["to"]
        run.finish("completed", output=context.get("last"))
    except RunCancelled:
        run.finish("cancelled", error="Run cancelled")
    except Exception as error:  # noqa: BLE001 - persisted as run failure
        if run.cancel_requested:
            run.finish("cancelled", error="Run cancelled")
            return
        detail = str(error)
        run.event("log", level="error", message=detail,
                  traceback=traceback.format_exc(limit=4))
        run.finish("failed", error=detail)


def _run_node(node, data, context, run, agent, gateway, root,
              abilities, invoke_child, depth, os_access: bool = False):
    node_type = node["type"]
    if node_type in {"start", "note"}:
        return context.get("last") if node_type == "note" and context.get("last") is not None else run.input
    if node_type == "end":
        template = data.get("output", "{{ last }}")
        return {"output": render_template(template, context)}
    if node_type == "condition":
        expression = str(data.get("expression") or "true")
        result = render_template("{{ " + expression + " }}", context)
        return {"result": bool(result), "value": result, "expression": expression}
    if node_type == "llm":
        return _run_llm(node, data, context, run, agent, gateway)
    if node_type == "agent":
        return _run_agent(node, data, context, run, agent, gateway, root, abilities,
                          os_access)
    if node_type == "swarm":
        return _run_swarm_node(node, data, context, run, agent, gateway, root, abilities,
                               os_access)
    if node_type == "human":
        return _run_human(node, data, context, run)
    if node_type == "foreach":
        return _run_foreach(node, data, context, run, invoke_child, depth)
    if node_type == "tool":
        ability_id = str(data.get("ability") or "")
        ability = abilities.get(ability_id)
        if not ability:
            raise ValueError(f"Unknown ability: {ability_id}")
        if not ability.get("enabled", True):
            raise ValueError(f"Ability {ability_id} is disabled")
        raw_input = render_data(data.get("input"), context)
        if raw_input in (None, "", {}):
            raw_input = context.get("last")
        run.event("node_input", node=node["id"], input=json_safe(raw_input, 4000))
        run.check_limits(run.timeout)
        result = execute_ability(ability, raw_input, context, root, run.dry_run,
                                 cancelled=run.stop, os_access=os_access)
        run.check_limits(run.timeout)
        return result
    if node_type == "transform":
        rendered = None
        if str(data.get("template") or "").strip():
            rendered = render_data(data.get("template"), context)
            if str(data.get("format") or "text").lower() == "json" and isinstance(rendered, str):
                try:
                    rendered = json.loads(rendered)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Transform output is not valid JSON: {error}") from error
        assigned = {}
        assignments = data.get("assign")
        if isinstance(assignments, dict):
            for key, value in assignments.items():
                assigned[str(key)] = render_data(value, context)
            context["vars"].update(assigned)
        return {
            "value": rendered if rendered is not None else assigned if assigned else context.get("last"),
            "assigned": assigned,
        }
    if node_type == "delay":
        seconds = min(300.0, max(0.0, float(data.get("seconds", 1) or 0)))
        if not run.dry_run:
            _sleep_cancelable(run, seconds)
        return {"delayed": seconds, "dry_run": run.dry_run}
    if node_type == "memory":
        ability = {"id": "memory", "type": "memory", "config": {}, "enabled": True}
        params = {
            "action": data.get("action", "get"),
            "key": render_data(data.get("key", ""), context),
            "value": render_data(data.get("value", ""), context),
        }
        return execute_ability(ability, params, context, root, run.dry_run,
                               cancelled=run.stop, os_access=os_access)
    if node_type == "subagent":
        if invoke_child is None:
            raise ValueError("Child agents are unavailable in this runtime")
        if depth >= 3:
            raise ValueError("Agent nesting exceeds the maximum depth of 3")
        target = str(data.get("agent") or "")
        if not target:
            raise ValueError("Agent node needs a target agent id")
        child_input = render_data(data.get("input", "{{ last }}"), context)
        run.event("node_input", node=node["id"], input=json_safe(child_input, 4000))
        return invoke_child(target, child_input, run, depth + 1)
    raise ValueError(f"Unsupported node type: {node_type}")


def _resolve_model(data: dict, agent: dict, gateway) -> str:
    """The model for an ``llm`` or ``agent`` node: node, then agent, then the only one."""
    model = str(data.get("model") or agent.get("model") or "").strip()
    if model:
        return model
    candidates = [
        entry for entry in gateway.catalog()
        if entry.get("available") and entry.get("capabilities", {}).get("chat")
    ]
    if len(candidates) == 1:
        return candidates[0]["id"]
    if not candidates:
        raise ValueError("No chat model is available; select a model for this agent")
    raise ValueError("Choose a model in agent settings or on the node")


class _RunCancel:
    """Gateway cancellation whose own ``set()`` does not cancel the whole run.

    ``ModelGateway.stream`` sets the event in its ``finally`` block.  Exposing the
    run's event directly would therefore mark a normally completed call as
    cancelled; this proxy keeps the gateway's local stop separate.
    """

    def __init__(self, run: LiveRun):
        self.run = run
        self.requested = threading.Event()
        self.expired = False

    def is_set(self) -> bool:
        return self.requested.is_set() or self.run.stop.is_set()

    def set(self) -> None:
        self.requested.set()

    def expire(self) -> None:
        self.expired = True
        self.requested.set()


def _chat(gateway, agent: dict, model: str, messages: list[dict], data: dict,
          default_temperature: float, default_tokens: int,
          run: LiveRun | None = None, tools: list[dict] | None = None) -> dict:
    settings = agent.get("model_settings") or {}
    temperature = data.get("temperature")
    config = {
        "model": model,
        "messages": messages,
        "temperature": float(default_temperature if temperature in (None, "") else temperature),
        "top_p": float(settings.get("top_p", 0.95)),
        "max_tokens": int(data.get("max_tokens") or default_tokens),
        "device": settings.get("device", "auto"),
    }
    if tools:
        # Native function calling where supported; the prompt and DSML parser stay
        # as the fallback for every other backend.
        config["tools"] = tools
    stream = getattr(gateway, "stream", None)
    if run is not None and callable(stream):
        scope = _RunCancel(run)
        remaining = run.timeout - run.active_elapsed()
        if remaining <= 0:
            run.check_limits(run.timeout)
        timer = threading.Timer(max(0.01, remaining), scope.expire)
        timer.daemon = True
        timer.start()
        final: dict = {}
        try:
            for event in stream(config, scope):
                if not isinstance(event, dict):
                    continue
                if event.get("type") == "done":
                    final = event
                elif event.get("type") == "error":
                    raise ValueError(str(event.get("error") or "The model call failed"))
        finally:
            timer.cancel()
        run.check_limits(run.timeout)
        if scope.expired:
            raise ValueError(f"Run exceeded timeout={run.timeout:.0f}s")
        if not final:
            raise ValueError("The model returned no final response")
        return final
    result = gateway.chat(config)
    if run is not None:
        run.check_limits(run.timeout)
    return result


def _run_llm(node, data, context, run, agent, gateway):
    prompt = render_data(data.get("prompt", ""), context)
    if prompt in (None, "", {}):
        prompt = context.get("last")
    prompt_text = _string(prompt)
    system = _string(render_data(data.get("system", ""), context)).strip()
    response_format = str(data.get("format") or "text").lower()
    if response_format == "json":
        system = (system + "\n\n" if system else "") + (
            "Reply with one valid JSON value and no other text. "
            "Use an object when the result has named fields."
        )
    run.event("node_input", node=node["id"], input=prompt_text[:4000])
    if run.dry_run:
        result = {
            "text": f"[test mode] {prompt_text[:240]}",
            "model": str(data.get("model") or agent.get("model") or "") or "test-mode",
            "dry_run": True,
            "usage": {"completion_tokens": 0},
        }
        if response_format == "json":
            result["json"] = None
            result["format"] = "json"
        return result
    model = _resolve_model(data, agent, gateway)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt_text})
    run.check_limits(run.timeout)
    result = _chat(gateway, agent, model, messages, data, 0.7, 512, run)
    run.check_limits(run.timeout)
    reply = str(result.get("reply") or "")
    output = {
        "text": reply,
        "model": model,
        "usage": result.get("usage", {}),
        "elapsed": result.get("elapsed", 0.0),
        "prompt_format": result.get("prompt_format", ""),
    }
    if response_format == "json":
        parsed = extract_json_value(reply)
        if parsed is None:
            raise ValueError(
                f"LLM node {node['id']} was asked for JSON but the model returned no JSON value; "
                "tell it to reply with a single JSON value, or change the node format to text"
            )
        output["json"] = parsed
        output["format"] = "json"
    return output


def _tool_instructions(system: str, tools: list[dict]) -> str:
    """The system prompt that teaches any chat model the JSON tool protocol."""
    lines = [system] if system else []
    if tools:
        lines.append("You can use tools to complete the task. Available tools:")
        for tool in tools:
            meta = ABILITY_TYPES.get(tool["type"], {})
            keys = ", ".join(field["key"] for field in meta.get("params", [])) or "none"
            lines.append(
                f"- {tool['id']} ({tool['type']}): {meta.get('description', '')} "
                f"Input keys: {keys}."
            )
        lines.append(
            "Reply with ONLY one JSON object and nothing else, using one of these forms:\n"
            '{"tool": "<tool id>", "input": {<input keys>}}\n'
            '{"calls": [{"tool": "<tool id>", "input": {...}}, ...]}\n'
            '{"plan": [{"text": "<step>", "status": "pending|active|done"}]}\n'
            "You will receive the result of each tool and may then call another tool. "
            "Use a plan when the task has several steps, and keep it current."
        )
    else:
        lines.append("No tools are available for this task.")
    lines.append('When you have the answer, reply with ONLY:\n{"final": "<your complete answer>"}')
    return "\n\n".join(lines)


def normalize_plan(value) -> list[dict]:
    """Normalise a model's plan into ``[{text, status}]`` for the run event log."""
    if isinstance(value, dict):
        value = value.get("todos", value.get("steps", value.get("plan", [])))
    if not isinstance(value, (list, tuple)):
        return []
    items = []
    for entry in list(value)[:40]:
        if isinstance(entry, str):
            items.append({"text": entry[:300], "status": "pending"})
        elif isinstance(entry, dict):
            text = str(entry.get("text") or entry.get("title") or "").strip()[:300]
            if not text:
                continue
            status = str(entry.get("status") or "pending").lower()
            items.append({"text": text, "status": status if status in
                          {"pending", "active", "done"} else "pending"})
    return items


def normalize_tool_calls(action: dict) -> list[dict]:
    """Accept the single-call and the parallel-call spelling."""
    raw = action.get("calls", action.get("tools"))
    if isinstance(raw, list):
        calls = [item for item in raw[:8] if isinstance(item, dict)]
    elif "tool" in action:
        calls = [action]
    else:
        calls = []
    normalised = []
    for call in calls:
        tool_id = str(call.get("tool") or call.get("name") or "").strip()
        arguments = call.get("input", call.get("arguments", call.get("args", {})))
        if tool_id:
            normalised.append({"tool": tool_id, "input": arguments})
    return normalised


def parse_model_action(text: str) -> tuple[dict | None, list[dict]]:
    """An action and its tool calls from JSON or a model's native tool markup.

    DSML is checked first: its parameters contain JSON objects, and a plain
    JSON scan would mistake one of them for the whole action.
    """
    calls = parse_dsml_tool_calls(text)
    if calls:
        return {"calls": calls}, calls
    action = extract_action_object(text)
    if action is None:
        return None, []
    return action, normalize_tool_calls(action)


def _run_agent(node, data, context, run, agent, gateway, root, abilities,
               os_access: bool = False):
    """An LLM that chooses abilities itself: reason, call a tool, observe, repeat."""
    goal = render_data(data.get("goal", ""), context)
    if goal in (None, "", {}):
        goal = context.get("last")
    goal_text = _string(goal)
    system = _string(render_data(data.get("system", ""), context)).strip()
    chosen = [str(item) for item in data.get("abilities") or []] or list(abilities)
    tools = [abilities[item] for item in chosen
             if item in abilities and abilities[item].get("enabled", True)]
    allowed = {tool["id"]: tool for tool in tools}
    limit = max(1, min(MAX_AGENT_ITERATIONS, int(data.get("max_iterations", 6) or 6)))
    run.event("node_input", node=node["id"], input=goal_text[:4000])
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    if run.dry_run:
        return {"text": f"[test mode] {goal_text[:240]}", "dry_run": True, "iterations": 0,
                "steps": [], "model": str(data.get("model") or agent.get("model") or "") or "test-mode",
                "usage": usage, "exhausted": False}
    model = _resolve_model(data, agent, gateway)
    messages = [{"role": "system", "content": _tool_instructions(system, tools)},
                {"role": "user", "content": goal_text}]
    steps: list[dict] = []
    reply = ""
    for iteration in range(1, limit + 1):
        run.check_limits(run.timeout)
        result = _chat(gateway, agent, model, messages, data, 0.3, 1024, run, tools)
        run.check_limits(run.timeout)
        reply = str(result.get("reply") or "")
        for key in usage:
            usage[key] += int((result.get("usage") or {}).get(key) or 0)
        action, calls = parse_model_action(reply)
        if action is None:
            text = "" if str(reply).strip() in EMPTY_REPLIES else _string(reply)
            return {"text": text, "iterations": iteration, "steps": steps, "model": model,
                    "usage": usage, "exhausted": False}
        plan = normalize_plan(action.get("plan")) if "plan" in action else None
        if plan is not None:
            run.event("plan", node=node["id"], iteration=iteration, plan=plan,
                      message=f"Plan updated ({len(plan)} steps)")
        if not calls:
            if "final" in action:
                final = action.get("final")
                text = "" if str(final).strip() in EMPTY_REPLIES else _string(final)
                return {"text": text, "iterations": iteration, "steps": steps, "model": model,
                        "usage": usage, "exhausted": False}
            if plan is not None:
                messages.append({"role": "assistant", "content": reply})
                messages.append({"role": "user", "content": (
                    "Plan recorded. Continue with a tool call, update the plan, "
                    "or reply with the final answer."
                )})
                continue
            final = reply
            text = "" if str(final).strip() in EMPTY_REPLIES else _string(final)
            return {"text": text, "iterations": iteration, "steps": steps, "model": model,
                    "usage": usage, "exhausted": False}
        observations: list[dict] = []
        for index, call in enumerate(calls):
            tool_id = call["tool"]
            tool_input = call["input"]
            run.event("tool_call", node=node["id"], iteration=iteration, index=index,
                      ability=tool_id, input=json_safe(tool_input, 4000),
                      message=f"Calling {tool_id}")
            step = {"iteration": iteration, "ability": tool_id,
                    "input": json_safe(tool_input, 4000)}
            try:
                run.check_limits(run.timeout)
                if tool_id not in allowed:
                    raise ValueError(
                        f"Unknown tool {tool_id!r}. Available tools: {', '.join(allowed) or 'none'}"
                    )
                observation = execute_ability(allowed[tool_id], tool_input, context, root,
                                               run.dry_run, cancelled=run.stop,
                                               os_access=os_access)
                run.check_limits(run.timeout)
                step["output"] = json_safe(observation, OBSERVATION_CHARS)
                succeeded = True
            except RunCancelled:
                raise
            except Exception as error:  # noqa: BLE001 - the model sees the failure and can adapt
                run.check_limits(run.timeout)
                observation = {"error": str(error)}
                step["error"] = str(error)
                succeeded = False
            steps.append(step)
            observations.append({"tool": tool_id, "ok": succeeded, "result": observation})
            run.event("tool_result", node=node["id"], iteration=iteration, index=index,
                      ability=tool_id, ok=succeeded, output=json_safe(observation, 4000),
                      message=f"{tool_id} {'returned' if succeeded else 'failed'}")
        messages.append({"role": "assistant", "content": reply})
        if len(observations) == 1:
            tool_id = observations[0]["tool"]
            rendered = json.dumps(json_safe(observations[0]["result"], OBSERVATION_CHARS),
                                  ensure_ascii=False)
            content = (f"Tool result for {tool_id}:\n{rendered[:OBSERVATION_CHARS]}\n\n"
                       "Use another tool, or reply with the final answer.")
        else:
            blocks = []
            for item in observations:
                rendered = json.dumps(json_safe(item["result"], OBSERVATION_CHARS),
                                      ensure_ascii=False)
                status = "returned" if item["ok"] else "failed"
                blocks.append(f"{item['tool']} {status}:\n{rendered[:OBSERVATION_CHARS]}")
            content = ("Tool results:\n\n" + "\n\n".join(blocks) +
                       "\n\nUse another tool, or reply with the final answer.")
        messages.append({"role": "user", "content": content})
    summary = f"Stopped after {limit} tool rounds without a final answer."
    return {"text": summary, "iterations": limit, "steps": steps, "model": model,
            "usage": usage, "exhausted": True}


def _run_swarm_node(node, data, context, run, agent, gateway, root, abilities,
                    os_access: bool = False):
    """Run a leader-driven swarm as one flow node."""
    goal = render_data(data.get("goal", ""), context)
    if goal in (None, "", {}):
        goal = context.get("last")
    goal_text = _string(goal)
    if not goal_text.strip():
        raise ValueError(f"Swarm node {node['id']} needs a goal")
    system = _string(render_data(data.get("system", ""), context)).strip()
    policy = agent.get("swarm") or {}
    if not isinstance(policy, dict):
        policy = {}
    max_agents = min(int(data.get("max_agents", 4) or 4), int(policy.get("max_agents", 6)))
    max_steps = min(int(data.get("max_steps", 6) or 6), int(policy.get("max_steps", 8)))
    rounds = min(int(data.get("rounds", 1) or 0), int(policy.get("max_rounds", 2)))
    max_seconds = min(int(data.get("max_seconds", 240) or 240),
                      int(policy.get("max_seconds", 300)))
    max_seconds = max(1, min(max_seconds, int(max(1.0, run.timeout - run.active_elapsed()))))
    allow_write = bool(data.get("write")) and bool(policy.get("allow_write"))
    allow_ipc = data.get("ipc") is not False and policy.get("allow_ipc", True) is not False
    shared_memory = (data.get("memory") is not False
                     and policy.get("shared_memory", True) is not False)
    chosen = [str(item) for item in data.get("tools") or []]
    if not chosen:
        chosen = [ability_id for ability_id, ability in abilities.items()
                  if ability.get("enabled", True)]
    tools = [abilities[ability_id] for ability_id in chosen
             if ability_id in abilities and abilities[ability_id].get("enabled", True)]
    if not allow_write:
        tools = [tool for tool in tools
                 if tool.get("type") not in {"file.write", "file.edit", "file.delete", "shell.exec"}]
    specs = _render_swarm_agents(data.get("agents"), context)
    model = _resolve_model(data, agent, gateway)
    leader_model = str(data.get("leader_model") or "").strip() or model
    limits = SwarmLimits.from_mapping({
        **{key: policy.get(key) for key in policy},
        "max_agents": max_agents,
        "max_steps": max_steps,
        "max_rounds": int(policy.get("max_rounds", 2)),
        "max_seconds": max_seconds,
        "max_messages": int(policy.get("max_messages", 200)),
        "max_message_chars": int(policy.get("max_message_chars", 4000)),
        "max_memory_entries": int(policy.get("max_memory_entries", 120)),
        "max_memory_chars": int(policy.get("max_memory_chars", 8000)),
    })
    run.event("node_input", node=node["id"], input=goal_text[:4000])
    if run.dry_run:
        output = {
            "text": f"[test mode] {goal_text[:240]}",
            "summary": "Test mode swarm", "agents": [], "completed": 0, "failed": 0,
            "messages": 0, "reviews": [], "blackboard": [], "changed_files": [],
            "strategy": str(data.get("strategy") or "auto"), "usage": {},
            "elapsed": 0.0, "dry_run": True,
        }
        run.set_swarm({
            "id": f"swarm-{node['id']}-test", "status": "completed",
            "goal": goal_text[:4000], "strategy": output["strategy"],
            "round": 0, "phase": "synthesis", "model": model,
            "leader_model": leader_model, "agents": [], "messages": [],
            "blackboard": [], "reviews": [], "changed_files": [],
            "usage": {}, "warnings": ["dry run: no model was called"],
        })
        return output
    coordinator = SwarmCoordinator(
        gateway=gateway, swarm_id=f"swarm-{node['id']}-{int(time.time() * 1000):x}",
        goal=goal_text, model=model, leader_model=leader_model, system=system,
        tools=tools, executor=lambda member, tool_id, arguments: _swarm_ability(
            abilities, member, tool_id, arguments, context, root, run, os_access),
        limits=limits, allow_write=allow_write, allow_ipc=allow_ipc,
        allow_os_access=os_access,
        shared_memory=shared_memory, dry_run=run.dry_run,
        on_event=lambda kind, **fields: _swarm_event(run, coordinator, kind, **fields),
        cancelled=lambda: run.stop.is_set(), briefing="",
    )
    run.swarm_coordinator = coordinator
    try:
        result = coordinator.run(agents=specs, strategy=str(data.get("strategy") or "auto"),
                                 rounds=rounds)
    finally:
        run.swarm_coordinator = None
    run.check_limits(run.timeout)
    if run.stop.is_set():
        raise RunCancelled()
    if not result.get("ok"):
        raise SwarmError(result.get("error") or f"Swarm node {node['id']} failed")
    output = {**result, "text": result.get("final") or result.get("summary") or ""}
    return output


def _render_swarm_agents(value, context: dict) -> list[dict]:
    """Render the optional member templates against the flow context."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            rendered = render_data(value, context)
            if isinstance(rendered, str):
                try:
                    value = json.loads(rendered)
                except json.JSONDecodeError:
                    raise ValueError(
                        "Swarm agents must be a JSON list of member objects"
                    ) from None
            else:
                value = rendered
    if not isinstance(value, list):
        return []
    result = []
    for item in value[:12]:
        if isinstance(item, str):
            result.append({"task": _string(render_data(item, context))})
        elif isinstance(item, dict):
            rendered = {}
            for key in ("id", "label", "role", "read_only", "context_chars"):
                if key in item:
                    rendered[key] = (render_data(item[key], context)
                                     if isinstance(item[key], str) else item[key])
            rendered["task"] = _string(render_data(item.get("task") or item.get("goal") or "",
                                                  context))
            if item.get("instruction"):
                rendered["instruction"] = _string(render_data(item["instruction"], context))
            if item.get("tools"):
                rendered["tools"] = list(item["tools"])[:24]
            if item.get("writes"):
                rendered["writes"] = list(item["writes"])[:40]
            result.append(rendered)
    return result


def _swarm_ability(abilities: dict, member: dict, tool_id: str, arguments: dict,
                   context: dict, root: Path, run: LiveRun, os_access: bool = False):
    """Execute one swarm member tool through the same sandbox as flow abilities."""
    ability = abilities.get(tool_id)
    if ability is None:
        raise ValueError(f"Unknown ability: {tool_id}")
    if not ability.get("enabled", True):
        raise ValueError(f"Ability {tool_id} is disabled")
    run.check_limits(run.timeout)
    observation = execute_ability(ability, arguments, context, root, run.dry_run,
                                  cancelled=run.stop, os_access=os_access)
    run.check_limits(run.timeout)
    return observation, True


def _swarm_event(run: LiveRun, coordinator, kind: str, **fields) -> None:
    """Forward a coordinator event and keep the run's swarm snapshot current."""
    if kind != "swarm_agent_delta":
        now = time.monotonic()
        if snapshot_due(kind, run.swarm_snapshot_at, now):
            try:
                run.set_swarm(coordinator.snapshot())
                run.swarm_snapshot_at = now
            except Exception:  # noqa: BLE001 - a state snapshot must not stop a swarm
                pass
    run.event(kind, **fields)


def _run_human(node, data, context, run):
    """Ask a person to approve, choose or type something, and wait for the answer."""
    kind = data.get("kind", "approve")
    if kind not in HUMAN_KINDS:
        raise ValueError("Human node kind must be approve, text, or choice")
    choices = [str(choice) for choice in data.get("choices") or []] if kind == "choice" else []
    default = data.get("default")
    spec = {
        "prompt": _string(render_data(data.get("prompt", ""), context)).strip() or "Continue?",
        "kind": kind, "choices": choices, "default": default,
    }
    has_default = default not in (None, "")
    fallback = True if kind == "approve" else (choices[0] if choices else "")
    if run.dry_run:
        value, dry_run, timed_out = normalize_answer(spec, default if has_default else fallback), True, False
    else:
        dry_run = False
        value, timed_out = run.request_input(node["id"], spec, float(data.get("timeout", 3600) or 0))
        if timed_out:
            if not has_default:
                raise ValueError(f"No answer arrived for {node['id']} and it has no default")
            value = normalize_answer(spec, default)
    output = {"value": value, "text": ("approved" if value else "rejected") if kind == "approve"
              else _string(value), "timed_out": timed_out}
    if kind == "approve":
        output["approved"] = bool(value)
    if dry_run:
        output["dry_run"] = True
    return output


def _run_foreach(node, data, context, run, invoke_child, depth):
    """Run a sub-agent once per item of a list, in order."""
    if invoke_child is None:
        raise ValueError("Child agents are unavailable in this runtime")
    target = str(data.get("agent") or "")
    if not target:
        raise ValueError("For-each node needs a target agent id")
    items = render_data(data.get("items", "{{ last }}"), context)
    if isinstance(items, str):
        try:
            items = json.loads(items)
        except json.JSONDecodeError:
            pass
    if not isinstance(items, (list, tuple)):
        raise ValueError(
            f"For-each node {node['id']} needs a list, but its items rendered to "
            f"{type(items).__name__}. Point it at a list such as {{{{ nodes.<id>.files }}}}"
        )
    limit = max(1, min(200, int(data.get("max_items", 20) or 20)))
    selected = list(items)[:limit]
    run.event("node_input", node=node["id"], input=json_safe(selected, 4000))
    outputs = []
    for index, item in enumerate(selected):
        run.check_limits(run.timeout)
        scope = {**context, "item": item, "index": index}
        child_input = render_data(data.get("input", "{{ item }}"), scope)
        run.event("log", node=node["id"], level="info",
                  message=f"Item {index + 1}/{len(selected)}")
        outputs.append(invoke_child(target, child_input, run, depth + 1))
        run.check_limits(run.timeout)
    return {"items": outputs, "count": len(outputs), "truncated": len(items) > len(selected)}


def _sleep_cancelable(run: LiveRun, seconds: float) -> None:
    deadline = time.time() + max(0.0, seconds)
    while time.time() < deadline:
        run.check_limits(run.timeout)
        time.sleep(min(0.1, max(0.0, deadline - time.time())))
