"""Persistence for AML agents, runs and schedule state."""

from __future__ import annotations

import json
import re
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

from ..workspace import ensure_workspace

MAX_RUNS_PER_AGENT = 60
ACTIVE_STATUSES = frozenset({"queued", "running", "paused"})


def _number(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


class AgentStore:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.workspace = ensure_workspace(self.root)
        self.agents_dir = self.workspace / "agents"
        self.runs_dir = self.workspace / "runs"
        self.schedule_path = self.workspace / "schedules.json"
        self._schedule_lock = threading.RLock()
        self.agents_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    # -- agents ----------------------------------------------------------
    def _agent_path(self, agent_id: str) -> Path:
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,119}", agent_id or ""):
            raise ValueError("Invalid agent id")
        return self.agents_dir / f"{agent_id}.json"

    def list_agents(self) -> list[dict]:
        agents = []
        for path in sorted(self.agents_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(data, dict) and isinstance(data.get("id"), str) and data["id"]:
                agents.append(data)
        agents.sort(key=lambda item: str(item.get("updated") or ""), reverse=True)
        return agents

    def get_agent(self, agent_id: str) -> dict:
        path = self._agent_path(agent_id)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise ValueError(f"Agent not found: {agent_id}") from None
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Agent {agent_id} is unreadable: {error}") from error
        if not isinstance(data, dict):
            raise ValueError(f"Agent {agent_id} is invalid")
        return data

    def save_agent(self, agent: dict) -> dict:
        path = self._agent_path(agent["id"])
        self._write_atomic(path, json.dumps(agent, indent=2, ensure_ascii=False))
        return agent

    def _write_atomic(self, path: Path, text: str) -> None:
        """Write through a unique temporary file so concurrent saves cannot interleave."""
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
        try:
            temporary.write_text(text, encoding="utf-8")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def delete_agent(self, agent_id: str) -> bool:
        path = self._agent_path(agent_id)
        if not path.exists():
            return False
        path.unlink()
        for run in self.list_runs(agent_id, limit=10_000):
            self._run_path(run["id"]).unlink(missing_ok=True)
        self.clear_schedule(agent_id)
        return True

    # -- runs ------------------------------------------------------------
    def _run_path(self, run_id: str) -> Path:
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,119}", run_id or ""):
            raise ValueError("Invalid run id")
        return self.runs_dir / f"{run_id}.json"

    def save_run(self, snapshot: dict) -> None:
        run_id = snapshot["id"]
        path = self._run_path(run_id)
        self._write_atomic(path, json.dumps(snapshot, ensure_ascii=False, default=str))

    def get_run(self, run_id: str) -> dict:
        path = self._run_path(run_id)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise ValueError(f"Run not found: {run_id}") from None
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Run {run_id} is unreadable: {error}") from error
        if not isinstance(data, dict):
            raise ValueError(f"Run {run_id} is invalid")
        return data

    def list_runs(self, agent_id: str | None = None, limit: int = 30) -> list[dict]:
        runs = []
        for path in self.runs_dir.glob("run-*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(data, dict) or not isinstance(data.get("id"), str):
                continue
            if agent_id and data.get("agent_id") != agent_id:
                continue
            runs.append({
                "id": data.get("id"),
                "agent_id": data.get("agent_id"),
                "agent_name": data.get("agent_name"),
                "status": data.get("status"),
                "trigger": data.get("trigger"),
                "dry_run": data.get("dry_run"),
                "debug": data.get("debug"),
                "steps": data.get("steps"),
                "error": data.get("error"),
                "created": data.get("created"),
                "started": data.get("started"),
                "ended": data.get("ended"),
                "elapsed": data.get("elapsed"),
                "parent_run_id": data.get("parent_run_id"),
            })
        runs.sort(key=lambda item: _number(item.get("created")), reverse=True)
        return runs[:max(1, limit)]

    def trim_runs(self, agent_id: str) -> None:
        runs = [item for item in self.list_runs(agent_id, limit=10_000)]
        for item in runs[MAX_RUNS_PER_AGENT:]:
            self._run_path(item["id"]).unlink(missing_ok=True)

    def settle_interrupted_runs(self) -> int:
        """Fail runs left active by a previous Studio process.

        A persisted ``running`` run cannot actually be running after a restart;
        leaving it in the history would make the UI report phantom work forever.
        """
        settled = 0
        now = time.time()
        stamp = datetime.now().isoformat(timespec="milliseconds")
        message = "Studio stopped before this run finished"
        for path in self.runs_dir.glob("run-*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if (not isinstance(data, dict) or not isinstance(data.get("id"), str)
                    or data.get("status") not in ACTIVE_STATUSES):
                continue
            data.update({"status": "failed", "error": message, "ended": data.get("ended") or now,
                         "updated": now, "awaiting_input": None, "waiting_node": None,
                         "paused": False, "current_node": None})
            nodes = data.get("nodes")
            if isinstance(nodes, dict):
                data["nodes"] = {
                    key: "error" if value in {"running", "paused"} else value
                    for key, value in nodes.items()
                }
            events = data.get("events") if isinstance(data.get("events"), list) else []
            seq = max((event.get("seq") for event in events
                       if isinstance(event, dict) and isinstance(event.get("seq"), int)),
                      default=0) + 1
            events.append({"seq": seq, "at": stamp, "type": "run_failed",
                           "message": message, "output": None})
            data["events"] = events[-800:]
            data["event_count"] = seq
            try:
                self.save_run(data)
            except (OSError, ValueError):
                continue
            settled += 1
        return settled

    # -- schedules -------------------------------------------------------
    def _schedules(self) -> dict:
        try:
            data = json.loads(self.schedule_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return {}
        if not isinstance(data, dict):
            return {}
        return {key: value for key, value in data.items() if isinstance(value, dict)}

    def _save_schedules(self, data: dict) -> None:
        self._write_atomic(self.schedule_path, json.dumps(data, indent=2))

    def schedule_state(self, agent_id: str) -> dict:
        with self._schedule_lock:
            return dict(self._schedules().get(agent_id, {}))

    def set_schedule_state(self, agent_id: str, state: dict) -> None:
        with self._schedule_lock:
            data = self._schedules()
            data[agent_id] = {**data.get(agent_id, {}), **state}
            self._save_schedules(data)

    def clear_schedule(self, agent_id: str) -> None:
        with self._schedule_lock:
            data = self._schedules()
            if agent_id in data:
                data.pop(agent_id)
                self._save_schedules(data)

    def all_schedules(self) -> dict:
        with self._schedule_lock:
            return {key: dict(value) for key, value in self._schedules().items()}
