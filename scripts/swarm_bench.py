"""Run a real Mariana swarm against any project folder.

Standalone: ``python scripts/swarm_bench.py --workspace /path/to/project --goal "..."``.
The workspace is used as the file sandbox; the saved endpoint/model comes from
Studio's endpoint store.  Prints a compact event log and a JSON summary.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mariana.agents.aml import ABILITY_TYPES  # noqa: E402
from mariana.agents.swarm import SwarmCoordinator, SwarmLimits  # noqa: E402
from mariana.agents.tools import IGNORED_DIRS, execute_ability  # noqa: E402
from mariana.studio.models.gateway import ModelGateway  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
READ_TOOLS = ("file.list", "file.tree", "file.read", "file.search",
              "code.diagnostics", "git.inspect", "csv.read", "time.now")
WRITE_TOOLS = ("file.write", "file.edit", "file.delete")


def tool_definitions(write: bool, shell: bool) -> list[dict]:
    ids = list(READ_TOOLS)
    if write:
        ids.extend(WRITE_TOOLS)
    if shell:
        ids.append("shell.exec")
    tools = []
    for tool_id in ids:
        meta = ABILITY_TYPES[tool_id]
        config = dict(meta.get("config") or {})
        if tool_id == "shell.exec":
            config["timeout"] = 60
        tools.append({
            "id": tool_id, "type": tool_id, "label": meta.get("label") or tool_id,
            "description": meta.get("description") or "",
            "params": meta.get("params") or [], "config": config, "enabled": True,
        })
    return tools


def project_briefing(workspace: Path, limit: int = 4_000) -> str:
    counts: Counter[str] = Counter()
    folders: Counter[str] = Counter()
    files = 0
    total = 0
    for base, directories, names in os.walk(workspace):
        directories[:] = sorted(name for name in directories
                                 if name not in IGNORED_DIRS
                                 and not (Path(base) / name).is_symlink())
        top = Path(base).relative_to(workspace).parts
        for name in names:
            path = Path(base) / name
            if path.is_symlink() or not path.is_file():
                continue
            files += 1
            try:
                total += path.stat().st_size
            except OSError:
                pass
            counts[path.suffix.lower() or "(none)"] += 1
            if top:
                folders[top[0]] += 1
        if files > 50_000:
            break
    lines = [f"Project: {workspace.name} ({files:,} files, {total / 1e6:.1f} MB)."]
    if folders:
        lines.append("Top folders: " + ", ".join(
            f"{name}/ ({count})" for name, count in folders.most_common(12)))
    if counts:
        lines.append("File types: " + ", ".join(
            f"{name} {count}" for name, count in counts.most_common(12)))
    readme = next((workspace / name for name in
                   ("README.md", "README.rst", "README.txt", "README") if (workspace / name).is_file()),
                  None)
    if readme:
        try:
            head = readme.read_text(encoding="utf-8", errors="replace")[:1_500]
            lines.append(f"README {readme.name} (first 1500 chars):\n{head}")
        except OSError:
            pass
    try:
        git = subprocess.run(["git", "-C", str(workspace), "log", "-5",
                              "--pretty=%h %ad %s", "--date=short"],
                             capture_output=True, text=True, timeout=5)
        if git.returncode == 0 and git.stdout.strip():
            lines.append("Recent commits:\n" + git.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return "\n".join(lines)[:limit]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a real swarm on a project folder")
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--goal", required=True)
    parser.add_argument("--model", default="endpoint:deepseek")
    parser.add_argument("--agents", type=int, default=4)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seconds", type=float, default=240.0)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--strategy", default="peer_review")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--shell", action="store_true")
    parser.add_argument("--effort", default="high")
    parser.add_argument("--model-concurrency", type=int, default=16,
                        help="Concurrent provider calls across the team")
    parser.add_argument("--tool-concurrency", type=int, default=32,
                        help="Concurrent tool executions across the team")
    parser.add_argument("--briefing", action="store_true",
                        help="prefetch a project map and pass it to the team")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    workspace = Path(args.workspace).expanduser().resolve()
    if not workspace.is_dir():
        parser.error(f"workspace is not a directory: {workspace}")
    gateway = ModelGateway(ROOT)
    entry = gateway.get(args.model)
    print(f"workspace: {workspace}")
    print(f"model: {entry['id']} ({entry.get('name')}) context={entry.get('detail', {}).get('context_window')}")

    tools = tool_definitions(args.write, args.shell)
    definitions = {tool["id"]: tool for tool in tools}
    cancelled = threading.Event()
    started = time.time()
    events: list[dict] = []

    def executor(member, tool_id, arguments):
        definition = definitions.get(str(tool_id))
        if definition is None:
            return {"error": f"Unknown tool {tool_id!r}"}, False
        values = arguments if isinstance(arguments, dict) else {"input": arguments}
        try:
            observation = execute_ability(
                definition, values,
                {"agent": member, "run": {"id": "project-bench"}},
                ROOT, files_root=workspace, cancelled=cancelled, os_access=args.shell)
        except Exception as failure:  # noqa: BLE001 - the member reads the error
            return {"error": str(failure) or type(failure).__name__}, False
        failed = isinstance(observation, dict) and (
            observation.get("ok") is False or bool(observation.get("error")))
        if not failed and isinstance(observation, dict) and str(tool_id) in WRITE_TOOLS:
            if observation.get("changed") is not False:
                observation = {**observation, "written": True}
        return observation, not failed

    def on_event(kind, **fields):
        events.append({"type": kind, "_t": round(time.time() - started, 2), **fields})
        if not args.json and kind in {"swarm_start", "swarm_plan", "swarm_round",
                                      "swarm_agent_start", "swarm_agent_tool",
                                      "swarm_agent_end", "swarm_review", "swarm_end"}:
            detail = fields.get("message") or fields.get("tool") or ""
            agent = fields.get("agent") or ""
            print(f"[{time.time() - started:7.2f}s] {kind} {agent} {detail}"[:300])

    limits = SwarmLimits.from_mapping({
        "max_agents": args.agents, "max_steps": args.steps,
        "max_seconds": args.seconds, "max_rounds": max(1, args.rounds + 1),
        "max_prompt_chars": 28_000, "max_observation_chars": 8_000,
        "max_output_chars": 20_000, "max_context_chars": 300_000,
        "max_agent_context_chars": 80_000, "max_context_item_chars": 32_000,
        "max_model_concurrency": args.model_concurrency,
        "max_tool_concurrency": args.tool_concurrency,
    })
    briefing = project_briefing(workspace) if args.briefing else ""
    coordinator = SwarmCoordinator(
        gateway=gateway, swarm_id=f"project-{int(time.time()):x}", goal=args.goal,
        model=entry["id"], leader_model=entry["id"], tools=tools, executor=executor,
        limits=limits, allow_write=args.write, allow_os_access=args.shell,
        effort=args.effort, reflection=True, shared_memory=True,
        on_event=on_event, cancelled=cancelled.is_set, briefing=briefing,
    )
    result = coordinator.run(strategy=args.strategy, rounds=args.rounds)
    elapsed = time.time() - started
    cache = result.get("cache") or {}
    tool_events = [item for item in events if item["type"] == "swarm_agent_tool"]
    failed_tools = [item for item in tool_events
                    if item.get("ok") is False or item.get("error")]

    def tool_error(item):
        output = item.get("output")
        if isinstance(output, dict):
            return output.get("error") or output.get("hint")
        return item.get("error")

    def tool_input(item):
        value = item.get("input")
        return value if value not in (None, {}) else None

    plan_event = next((item for item in events if item["type"] == "swarm_plan"), None)
    agent_end = [item for item in events if item["type"] == "swarm_agent_end"]
    final_event = next((item for item in events if item["type"] == "swarm_end"), None)
    plan_seconds = plan_event.get("_t") if plan_event else None
    work_seconds = (round(agent_end[-1]["_t"] - plan_event["_t"], 2)
                    if agent_end and plan_event else None)
    finish_seconds = (round(final_event["_t"] - agent_end[-1]["_t"], 2)
                      if agent_end and final_event else None)

    summary = {
        "ok": result.get("ok"), "error": result.get("error"),
        "elapsed": round(elapsed, 2), "plan_seconds": plan_seconds,
        "work_seconds": work_seconds, "finish_seconds": finish_seconds,
        "agents": len(result.get("agents") or []),
        "tool_calls": len([item for item in tool_events if item.get("ok") is not None]),
        "tool_failures": len(failed_tools),
        "tool_errors": [{"tool": item.get("tool"), "error": tool_error(item),
                         "input": tool_input(item)} for item in failed_tools[:8]],
        "cache": cache, "usage": result.get("usage"),
        "changed_files": result.get("changed_files"),
        "agents_detail": [{"id": item.get("id") or item.get("agent"),
                           "status": item.get("status"), "error": item.get("error"),
                           "tool_calls": item.get("tool_calls"),
                           "tool_failures": item.get("tool_failures"),
                           "output": str(item.get("output") or "")[:1_200]}
                          for item in (result.get("agents") or [])],
        "warnings": result.get("warnings"),
        "final": result.get("final") or "",
    }
    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        print("\n=== summary ===")
        print(json.dumps({key: value for key, value in summary.items() if key != "final"},
                         indent=2, ensure_ascii=False))
        print("\n=== final ===\n" + summary["final"][:12_000])
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
