"use strict";

/* Swarm state and the agent card shared by Chat (and available to the IDE).
   A state object is plain data so it can be saved with a conversation:
   { agents, meta, reviews, messages, memory }.  Events from the server are
   applied incrementally; snapshots replace the whole picture while keeping
   the live text accumulated from agent deltas. */
(function () {
  const Studio = window.Studio;
  const { h } = Studio;

  function create(extra) {
    return { agents: [], meta: { ...(extra || {}) }, reviews: [], messages: [], memory: [] };
  }

  function normalize(value) {
    const swarm = value && typeof value === "object" ? value : {};
    if (!Array.isArray(swarm.agents)) swarm.agents = [];
    if (!Array.isArray(swarm.reviews)) swarm.reviews = [];
    if (!Array.isArray(swarm.messages)) swarm.messages = [];
    if (!Array.isArray(swarm.memory)) swarm.memory = [];
    if (!swarm.meta || typeof swarm.meta !== "object") swarm.meta = {};
    if (swarm.meta.tools !== undefined && !Array.isArray(swarm.meta.tools)) {
      delete swarm.meta.tools;
    }
    return swarm;
  }

  function agentId(agent) {
    return agent ? agent.agent || agent.id || "" : "";
  }

  function findAgent(swarm, id) {
    return swarm.agents.find(agent => agentId(agent) === id);
  }

  function agentState(agent) {
    if (agent && agent.status) return String(agent.status);
    if (agent && agent.ok === false) return "failed";
    if (agent && agent.ok) return "done";
    return "queued";
  }

  function compactChars(value) {
    const number = Number(value) || 0;
    if (number >= 1000000) return `${(number / 1000000).toFixed(1)}M`;
    if (number >= 1000) return `${Math.round(number / 1000)}k`;
    return String(number);
  }

  function phaseLabel(phase) {
    return {
      planning: "Planning", work: "Working", review: "Reviewing",
      revise: "Revising", synthesis: "Synthesizing", stop: "Stopping",
    }[phase] || "Working";
  }

  function statusLabel(status) {
    if (status === "done") return "Done";
    if (status === "failed") return "Failed";
    if (status === "reviewing") return "Reviewing";
    if (status === "revising") return "Revising";
    if (["running", "thinking", "writing", "tool"].includes(status)) return "Working";
    return "Queued";
  }

  function oneLine(value, limit) {
    const line = String(value === null || value === undefined ? "" : value)
      .replace(/\s+/g, " ").trim();
    return line.length > limit ? `${line.slice(0, limit - 1)}…` : line;
  }

  function setAgents(swarm, list) {
    const prior = new Map();
    for (const agent of swarm.agents) prior.set(agentId(agent), agent);
    swarm.agents = [];
    for (const item of Array.isArray(list) ? list : []) {
      if (!item || typeof item !== "object") continue;
      const id = agentId(item);
      if (!id) continue;
      const old = prior.get(id) || {};
      swarm.agents.push({ ...old, ...item, agent: id, text: item.text || old.text || "" });
    }
  }

  function ensureAgent(swarm, event) {
    const id = String(event.agent || "leader");
    let agent = findAgent(swarm, id);
    if (!agent) {
      agent = { agent: id, label: event.label || id, task: event.task || "", status: "queued" };
      swarm.agents.push(agent);
    }
    return agent;
  }

  function applySnapshot(value, snapshot) {
    if (!snapshot || typeof snapshot !== "object") return value;
    const swarm = normalize(value);
    setAgents(swarm, snapshot.agents || []);
    if (Array.isArray(snapshot.reviews)) swarm.reviews = snapshot.reviews;
    if (Array.isArray(snapshot.messages)) swarm.messages = snapshot.messages;
    if (Array.isArray(snapshot.blackboard)) swarm.memory = snapshot.blackboard;
    const meta = swarm.meta;
    if (snapshot.id) meta.id = snapshot.id;
    if (snapshot.strategy) meta.strategy = snapshot.strategy;
    if (snapshot.round !== undefined) meta.round = snapshot.round;
    if (snapshot.requested_rounds !== undefined) meta.rounds = snapshot.requested_rounds;
    if (snapshot.phase) meta.phase = snapshot.phase;
    if (snapshot.leader_model) meta.leader = snapshot.leader_model;
    if (snapshot.model) meta.model = snapshot.model;
    if (snapshot.status) meta.status = snapshot.status;
    if (snapshot.reason) meta.leaderNote = snapshot.reason;
    if (snapshot.context && typeof snapshot.context === "object") meta.context = snapshot.context;
    if (Array.isArray(snapshot.warnings) && snapshot.warnings.length) meta.warnings = snapshot.warnings;
    meta.messageCount = Math.max(swarm.messages.length, Number(meta.messageCount) || 0);
    meta.reviewCount = Math.max(swarm.reviews.length, Number(meta.reviewCount) || 0);
    meta.memoryCount = Math.max(swarm.memory.length, Number(meta.memoryCount) || 0);
    return swarm;
  }

  function applyEvent(value, event) {
    if (!event || typeof event !== "object") return value;
    const swarm = normalize(value);
    const meta = swarm.meta;
    switch (event.type) {
      case "swarm_start": {
        if (event.swarm) meta.id = event.swarm;
        if (event.strategy) meta.strategy = event.strategy;
        if (event.leader) meta.leader = event.leader;
        if (event.model) meta.model = event.model;
        if (event.requested_rounds !== undefined) meta.rounds = event.requested_rounds;
        if (Array.isArray(event.tools)) meta.tools = event.tools;
        meta.phase = "planning";
        meta.status = "planning";
        if (Array.isArray(event.agents) && event.agents.length) setAgents(swarm, event.agents);
        break;
      }
      case "swarm_plan": {
        setAgents(swarm, event.agents || []);
        if (event.strategy) meta.strategy = event.strategy;
        if (event.rounds !== undefined) meta.rounds = event.rounds;
        if (event.notes) meta.leaderNote = event.notes;
        meta.phase = "work";
        break;
      }
      case "swarm_leader": {
        if (event.text || event.message) meta.leaderNote = event.text || event.message;
        if (event.phase) meta.phase = event.phase;
        break;
      }
      case "swarm_context": {
        if (event.context && typeof event.context === "object") meta.context = event.context;
        if (event.agent) {
          const agent = ensureAgent(swarm, event);
          const pool = event.context && event.context.agents;
          agent.contextGranted = pool && Number.isFinite(pool[event.agent]) ? pool[event.agent] : 0;
        }
        break;
      }
      case "swarm_round": {
        if (event.round !== undefined) meta.round = event.round;
        if (event.phase && event.phase !== "stop") meta.phase = event.phase;
        break;
      }
      case "swarm_agent_start": {
        const agent = ensureAgent(swarm, event);
        agent.label = event.label || agent.label;
        agent.role = event.role || agent.role;
        agent.task = event.task || agent.task;
        agent.version = event.version || agent.version;
        agent.status = event.phase === "review" ? "reviewing"
          : event.phase === "revise" ? "revising" : "running";
        agent.note = event.message || "";
        if (event.phase) meta.phase = event.phase;
        if (event.round !== undefined) meta.round = event.round;
        break;
      }
      case "swarm_agent_step":
      case "swarm_agent_status": {
        const agent = ensureAgent(swarm, event);
        agent.status = event.phase === "review" ? "reviewing"
          : event.phase === "revise" ? "revising" : "thinking";
        agent.note = event.message || agent.note || "";
        break;
      }
      case "swarm_agent_delta": {
        const agent = ensureAgent(swarm, event);
        agent.status = event.phase === "review" ? "reviewing"
          : event.phase === "revise" ? "revising" : "writing";
        agent.text = `${agent.text || ""}${event.text || ""}`.slice(-8000);
        break;
      }
      case "swarm_agent_tool": {
        const agent = ensureAgent(swarm, event);
        agent.status = "tool";
        agent.lastTool = event.tool || "";
        agent.note = event.message || "";
        break;
      }
      case "swarm_agent_end": {
        const agent = ensureAgent(swarm, event);
        agent.status = event.status || (event.ok ? "done" : "failed");
        if (event.output) agent.output = event.output;
        if (event.summary) agent.note = event.summary;
        agent.error = event.error || "";
        if (event.elapsed !== undefined) agent.elapsed = event.elapsed;
        if (event.score !== undefined) agent.score = event.score;
        if (event.version !== undefined) agent.version = event.version;
        if (event.steps !== undefined) agent.steps = event.steps;
        if (event.phase) meta.phase = event.phase;
        break;
      }
      case "swarm_review": {
        const exists = swarm.reviews.some(item => item.reviewer === event.reviewer
          && item.target === event.target && item.round === event.round);
        if (!exists) {
          swarm.reviews.push({
            round: event.round, reviewer: event.reviewer, target: event.target,
            score: event.score, issues: event.issues || [], summary: "",
          });
        }
        meta.reviewCount = Math.max(meta.reviewCount || 0, swarm.reviews.length);
        break;
      }
      case "swarm_message": {
        const seen = event.message_id
          && swarm.messages.some(item => item.id === event.message_id);
        if (!seen) {
          swarm.messages.push({
            id: event.message_id, from: event.from, to: event.to,
            kind: event.message_kind || "message", text: event.text || "", round: event.round || 0,
          });
        }
        meta.messageCount = Math.max(meta.messageCount || 0, swarm.messages.length);
        break;
      }
      case "swarm_memory": {
        if (event.action === "delete") {
          swarm.memory = swarm.memory.filter(item => item.key !== event.key);
        } else if (event.key) {
          const existing = swarm.memory.find(item => item.key === event.key);
          if (existing) existing.version = event.version;
          else swarm.memory.push({ key: event.key, version: event.version });
        }
        meta.memoryCount = swarm.memory.length;
        break;
      }
      case "swarm_end": {
        meta.finished = true;
        meta.ok = event.ok !== false;
        meta.status = event.status || (meta.ok ? "completed" : "failed");
        meta.phase = "synthesis";
        if (Array.isArray(event.agents) && event.agents.length) setAgents(swarm, event.agents);
        if (event.output) meta.final = event.output;
        if (event.summary) meta.summary = event.summary;
        if (event.messages !== undefined) meta.messageCount = event.messages;
        if (event.reviews !== undefined) meta.reviewCount = event.reviews;
        if (event.blackboard !== undefined) meta.memoryCount = event.blackboard;
        if (event.error) meta.error = event.error;
        break;
      }
      default:
        break;
    }
    return swarm;
  }

  function finish(value, summary) {
    const swarm = normalize(value);
    const meta = swarm.meta;
    meta.finished = true;
    if (!summary || typeof summary !== "object") return swarm;
    if (Array.isArray(summary.agents) && summary.agents.length) setAgents(swarm, summary.agents);
    if (Array.isArray(summary.reviews)) swarm.reviews = summary.reviews;
    if (Array.isArray(summary.memory)) swarm.memory = summary.memory;
    meta.ok = summary.ok !== false;
    meta.status = summary.status || (meta.ok ? "completed" : "failed");
    if (summary.strategy) meta.strategy = summary.strategy;
    if (summary.round !== undefined) meta.round = summary.round;
    if (summary.rounds !== undefined) meta.rounds = summary.rounds;
    if (summary.leader) meta.leader = summary.leader;
    if (summary.summary) meta.summary = summary.summary;
    if (summary.elapsed !== undefined) meta.elapsed = summary.elapsed;
    if (summary.usage) meta.usage = summary.usage;
    if (Array.isArray(summary.warnings) && summary.warnings.length) meta.warnings = summary.warnings;
    if (Array.isArray(summary.tools)) meta.tools = summary.tools.slice(0, 24).map(String);
    if (summary.messages !== undefined) meta.messageCount = summary.messages;
    if (summary.review_count !== undefined) meta.reviewCount = summary.review_count;
    if (summary.memory_count !== undefined) meta.memoryCount = summary.memory_count;
    if (summary.context && typeof summary.context === "object") meta.context = summary.context;
    meta.messageCount = Math.max(swarm.messages.length, Number(meta.messageCount) || 0);
    meta.reviewCount = Math.max(swarm.reviews.length, Number(meta.reviewCount) || 0);
    meta.memoryCount = Math.max(swarm.memory.length, Number(meta.memoryCount) || 0);
    return swarm;
  }

  function detailsState(container) {
    const open = new Set();
    const closed = new Set();
    if (!container) return { open, closed };
    for (const details of container.querySelectorAll("details[data-swarm-key]")) {
      const key = details.getAttribute("data-swarm-key");
      if (!key) continue;
      if (details.open) open.add(key);
      else closed.add(key);
    }
    return { open, closed };
  }

  function render(container, value, options = {}) {
    const swarm = normalize(value);
    const meta = swarm.meta;
    const live = Boolean(options.live);
    const details = options.details || detailsState(container);
    const isOpen = (key, fallback) => (details.open.has(key)
      ? true : details.closed.has(key) ? false : fallback);
    const finished = Boolean(meta.finished)
      || ["completed", "failed", "cancelled"].includes(meta.status);
    const active = live && !finished;
    const failed = swarm.agents.filter(agent => agentState(agent) === "failed").length;
    const settled = swarm.agents.filter(agent => ["done", "failed"].includes(agentState(agent))).length;
    const cancelled = meta.status === "cancelled";
    const ok = meta.ok !== false;
    const badgeText = active ? "Running"
      : cancelled ? "Stopped"
        : failed ? "Finished with failures"
          : finished ? (ok ? "Done" : "Failed")
            : swarm.agents.length || meta.error ? "Stopped" : "Queued";
    const badgeTone = active ? "running"
      : cancelled ? "muted"
        : failed || (!ok && finished) ? "failed"
          : finished ? "completed" : "muted";
    const role = meta.strategy ? String(meta.strategy).replace(/_/g, " ") : "leader-planned";
    const root = Studio.ui.render(container);
    const card = h("article", { class: `swarm-card${failed ? " failed" : ""}` });
    root.append(card);

    card.append(h("div", { class: "swarm-head" },
      Studio.icon("agents", "small"),
      h("strong", { class: "grow", text: "Agent swarm" }),
      h("span", { class: "muted tiny", text: swarm.agents.length ? `${settled}/${swarm.agents.length} finished` : "" }),
      Studio.ui.badge(badgeText, badgeTone)));

    const metaRow = h("div", { class: "swarm-meta" },
      Studio.ui.badge(role, "accent"),
      meta.round ? Studio.ui.badge(`Round ${meta.round}${meta.rounds ? `/${meta.rounds}` : ""}`, "muted") : null,
      meta.phase ? Studio.ui.badge(phaseLabel(meta.phase), active ? "running" : "muted") : null,
      meta.reviewCount ? Studio.ui.badge(`${meta.reviewCount} review${meta.reviewCount === 1 ? "" : "s"}`,
        "completed") : null,
      meta.messageCount ? Studio.ui.badge(`${meta.messageCount} message${meta.messageCount === 1 ? "" : "s"}`,
        "muted") : null,
      meta.memoryCount ? Studio.ui.badge(`${meta.memoryCount} shared key${meta.memoryCount === 1 ? "" : "s"}`,
        "muted") : null,
      meta.context && Number(meta.context.capacity) > 0
        ? Studio.ui.badge(`${compactChars(meta.context.used || 0)}/${compactChars(meta.context.capacity)} context`,
          "muted")
        : null,
      Array.isArray(meta.tools)
        ? Studio.ui.badge(meta.tools.length
          ? `${meta.tools.length} tool${meta.tools.length === 1 ? "" : "s"}`
          : "no tools", "muted")
        : null);
    card.append(metaRow);

    if (meta.leaderNote) {
      card.append(h("p", { class: "swarm-note", text: meta.leaderNote }));
    }
    if (!swarm.agents.length) {
      card.append(h("p", { class: "muted tiny",
        text: active ? "The leader is planning the team…" : "No agents reported." }));
    }

    const list = h("div", { class: "swarm-agents" });
    for (const agent of swarm.agents) {
      const state = agentState(agent);
      const running = active && ["running", "thinking", "writing", "tool", "reviewing",
        "revising", "queued"].includes(state);
      const glyph = running
        ? h("span", { class: "spinner" })
        : h("span", { class: "swarm-glyph" },
          Studio.icon(state === "failed" ? "alert" : state === "done" ? "check" : "sparkle", "small"));
      const label = agent.label || agentId(agent) || "Agent";
      const facts = [];
      if (agent.role) facts.push(agent.role);
      if (agent.score !== undefined && agent.score !== null) facts.push(`${agent.score}/5`);
      if (agent.version && agent.version > 1) facts.push(`v${agent.version}`);
      if (agent.steps) facts.push(`${agent.steps} step${agent.steps === 1 ? "" : "s"}`);
      const granted = agent.contextGranted ?? agent.context_granted;
      if (granted) facts.push(`${compactChars(granted)} context`);
      if (agent.elapsed) facts.push(`${agent.elapsed}s`);
      const detail = h("div", { class: "swarm-body" });
      if (facts.length) detail.append(h("p", { class: "muted tiny swarm-facts", text: facts.join(" · ") }));
      if (agent.task) detail.append(h("p", { class: "muted tiny", text: agent.task }));
      if (agent.lastTool) detail.append(h("p", { class: "muted tiny", text: `Last tool: ${agent.lastTool}` }));
      if (agent.note && !agent.output) detail.append(h("p", { class: "muted tiny", text: agent.note }));
      if (agent.error) {
        detail.append(Studio.ui.callout("danger", h("span", { text: agent.error })));
      } else if (agent.output) {
        const output = h("div", { class: "swarm-output", "data-swarm-output": agentId(agent) });
        Studio.markdown.into(output, agent.output);
        detail.append(output);
      } else if (agent.text) {
        detail.append(h("pre", { class: "swarm-preview", "data-swarm-output": agentId(agent),
          text: agent.text.slice(-4000) }));
      }
      list.append(h("details", {
        class: `swarm-agent ${state}`, "data-swarm-key": `agent:${agentId(agent)}`,
        open: isOpen(`agent:${agentId(agent)}`, running || state === "failed"),
      },
        h("summary", {},
          glyph,
          h("span", { class: "grow" },
            h("strong", { text: label }),
            agent.task ? h("span", { class: "swarm-task", text: agent.task }) : null),
          Studio.ui.badge(statusLabel(state),
            state === "done" ? "completed" : state === "failed" ? "failed"
              : state === "queued" ? "muted" : "running")),
        detail));
    }
    card.append(list);

    if (swarm.reviews.length) {
      const rows = h("div", { class: "stack tight" });
      for (const review of swarm.reviews.slice(-8)) {
        const reviewer = findAgent(swarm, review.reviewer);
        const target = findAgent(swarm, review.target);
        rows.append(h("div", { class: "swarm-review" },
          Studio.ui.badge(`${review.score === undefined || review.score === null ? "?" : review.score}/5`,
            review.score >= 4 ? "completed" : "warning"),
          h("span", { class: "grow tiny",
            text: `${reviewer ? reviewer.label : review.reviewer} → ${target ? target.label : review.target}` }),
          h("span", { class: "muted tiny truncate",
            text: oneLine((review.issues || []).join("; ") || review.summary || "reviewed", 110) })));
      }
      card.append(h("details", {
        class: "swarm-section", "data-swarm-key": "reviews",
        open: isOpen("reviews", swarm.reviews.length <= 3),
      },
        h("summary", { text: `Peer review · ${swarm.reviews.length} review${swarm.reviews.length === 1 ? "" : "s"}` }),
        rows));
    }

    if (swarm.messages.length) {
      const rows = h("div", { class: "stack tight" });
      for (const message of swarm.messages.slice(-12)) {
        rows.append(h("div", { class: "swarm-message" },
          h("span", { class: "mono tiny muted", text: `${message.from} → ${message.to}` }),
          h("span", { class: "grow tiny", text: oneLine(message.text, 220) })));
      }
      card.append(h("details", { class: "swarm-section", "data-swarm-key": "messages",
        open: isOpen("messages", false) },
        h("summary", { text: `Agent messages · ${swarm.messages.length}` }), rows));
    }

    if (swarm.memory.length) {
      const rows = h("div", { class: "row wrap" });
      for (const item of swarm.memory.slice(-16)) rows.append(Studio.ui.badge(item.key || String(item), "muted"));
      card.append(h("details", { class: "swarm-section", "data-swarm-key": "memory",
        open: isOpen("memory", false) },
        h("summary", { text: `Shared memory · ${swarm.memory.length} key${swarm.memory.length === 1 ? "" : "s"}` }),
        rows));
    }

    if (options.showFinal && meta.final && finished) {
      const summary = h("div", { class: "swarm-final" });
      Studio.markdown.into(summary, meta.final);
      card.append(h("details", { class: "swarm-section", "data-swarm-key": "final",
        open: isOpen("final", false) },
        h("summary", { text: "Synthesised result" }), summary));
    }
    return root;
  }

  Studio.swarm = {
    create, agentState, phaseLabel, statusLabel, detailsState,
    applySnapshot, applyEvent, finish, render,
  };
})();
