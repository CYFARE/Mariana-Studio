"""AI-assisted authoring: turn a plain-language description into a valid AML agent.

A chat model is shown the live node and ability tables and asked for one AML
document.  The reply is validated; errors are fed back for up to two repair
rounds.  The result is returned for review and is never saved here.
"""

from __future__ import annotations

import json

from . import aml
from .runtime import extract_json_object

MAX_DESCRIPTION_CHARS = 4000
MAX_REPAIRS = 2
EXAMPLE = {
    "aml": aml.AML_VERSION, "kind": "agent", "id": "summarise-page", "name": "Summarise a page",
    "description": "Fetch a URL and summarise it.",
    "abilities": [{"id": "fetch", "type": "http.fetch", "config": {"method": "GET"}}],
    "nodes": [
        {"id": "start", "type": "start", "label": "Start"},
        {"id": "get", "type": "tool", "label": "Fetch page",
         "data": {"ability": "fetch", "input": {"url": "{{ input.url }}", "extract": "text"}}},
        {"id": "summarise", "type": "llm", "label": "Summarise",
         "data": {"prompt": "Summarise in three bullet points:\n\n{{ nodes.get.text }}"}},
        {"id": "end", "type": "end", "label": "Done", "data": {"output": "{{ nodes.summarise.text }}"}},
    ],
    "edges": [{"from": "start", "to": "get"}, {"from": "get", "to": "summarise"},
              {"from": "summarise", "to": "end"}],
}


def _reference() -> str:
    lines = ["Node types (type: purpose; data keys):"]
    for key, meta in aml.NODE_TYPES.items():
        fields = ", ".join(field["key"] for field in meta["fields"]) or "none"
        lines.append(f"- {key}: {meta['description']} Data keys: {fields}.")
    lines.append("")
    lines.append("Ability types (type: purpose; input keys):")
    for key, meta in aml.ABILITY_TYPES.items():
        if meta.get("danger"):
            continue  # never suggest the shell
        params = ", ".join(field["key"] for field in meta.get("params", [])) or "none"
        lines.append(f"- {key}: {meta['description']} Input keys: {params}.")
    return "\n".join(lines)


def system_prompt() -> str:
    return "\n\n".join([
        "You design automation agents in AML, a JSON flowchart format. Reply with ONE JSON "
        "object and nothing else: no prose, no code fence.",
        _reference(),
        "Rules:\n"
        "- Exactly one node of type start and at least one of type end.\n"
        "- Every node has a unique lowercase id using letters, digits and dashes.\n"
        "- Edges are {\"from\": id, \"to\": id}. Only a condition node branches: give it two "
        "edges with \"when\": \"true\" and \"when\": \"false\". Every other node has one "
        "outgoing edge.\n"
        "- A tool node names an ability by the id declared in \"abilities\".\n"
        "- Optional top-level \"vars\" seeds flow variables available as {{ vars.<name> }}.\n"
        "- Set an llm node's \"format\" to \"json\" when a later node needs structured fields.\n"
        "- A swarm node delegates work to a leader-planned team. Give it a clear goal, keep "
        "rounds at 1, and keep the top-level swarm.allow_write false unless the request "
        "explicitly needs parallel file edits.\n"
        "- Templates use {{ }}: {{ input.<key> }} is the run input, {{ last }} the previous "
        "node's output, {{ nodes.<id>.<field> }} an earlier node's output (an llm node's "
        "text is {{ nodes.<id>.text }}).\n"
        "- Leave \"model\" empty and keep schedules disabled.",
        "Example:\n" + json.dumps(EXAMPLE, indent=1),
    ])


def _layout(agent: dict) -> None:
    """Place nodes left to right by distance from the start when none were positioned."""
    nodes = agent["nodes"]
    if any((node.get("x"), node.get("y")) != (80.0, 80.0) for node in nodes):
        return
    outgoing: dict[str, list[str]] = {}
    for edge in agent["edges"]:
        outgoing.setdefault(edge["from"], []).append(edge["to"])
    depth: dict[str, int] = {}
    start = next((node["id"] for node in nodes if node["type"] == "start"), nodes[0]["id"])
    queue = [(start, 0)]
    while queue:
        node_id, level = queue.pop(0)
        if node_id in depth:
            continue
        depth[node_id] = level
        queue += [(child, level + 1) for child in outgoing.get(node_id, [])]
    deepest = max(depth.values(), default=0)
    rows: dict[int, int] = {}
    for node in nodes:
        level = depth.get(node["id"], deepest + 1)
        row = rows.get(level, 0)
        rows[level] = row + 1
        node["x"], node["y"] = 60.0 + 250.0 * level, 120.0 + 140.0 * row


def generate_agent(gateway, description: str, model: str | None, known_agent_ids: set[str]) -> dict:
    """Return ``{"agent", "warnings", "issues", "attempts", "model"}`` for a description."""
    description = str(description or "").strip()[:MAX_DESCRIPTION_CHARS]
    if len(description) < 8:
        raise ValueError("Describe what the agent should do in a sentence or two")
    if not model:
        candidates = [entry for entry in gateway.catalog()
                      if entry.get("available") and entry.get("capabilities", {}).get("chat")]
        if not candidates:
            raise ValueError("No chat model is available; add one in Models, then try again")
        if len(candidates) > 1:
            raise ValueError("Choose which model should draft the agent")
        model = candidates[0]["id"]
    messages = [{"role": "system", "content": system_prompt()},
                {"role": "user", "content": f"Design an agent for this request:\n\n{description}"}]
    problems: list[str] = []
    for attempt in range(1, MAX_REPAIRS + 2):
        reply = str(gateway.chat({
            "model": model, "messages": messages, "temperature": 0.2, "max_tokens": 4096,
        }).get("reply") or "")
        document = extract_json_object(reply)
        if document is None:
            problems = ["The reply was not a JSON object"]
        else:
            document.pop("schedule", None)  # a drafted agent never schedules itself
            try:
                result = aml.validate_agent(document, known_agent_ids)
            except aml.AMLValidationError as error:
                problems = list(error.errors)
            else:
                if result["ok"]:
                    _layout(result["agent"])
                    return {"agent": result["agent"], "warnings": result["warnings"],
                            "issues": result["issues"], "attempts": attempt, "model": model}
                problems = result["errors"]
        messages.append({"role": "assistant", "content": reply[:6000]})
        messages.append({"role": "user", "content": (
            "That document is not valid AML:\n- " + "\n- ".join(problems[:8])
            + "\nReply with the corrected JSON object only."
        )})
    raise ValueError(
        "The model could not produce a valid agent (" + "; ".join(problems[:3])
        + "). Try a more capable model, or describe the steps more concretely."
    )
