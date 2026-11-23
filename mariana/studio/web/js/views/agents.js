"use strict";

(function () {
  const Studio = window.Studio;
  const { $, h, api, format } = Studio;
  const SVG_NS = "http://www.w3.org/2000/svg";
  const NODE_WIDTH = 190;
  const NODE_HEIGHT = 82;
  const raf = typeof requestAnimationFrame === "function" ? requestAnimationFrame : task => setTimeout(task, 16);
  const caf = typeof cancelAnimationFrame === "function" ? cancelAnimationFrame : handle => { if (typeof clearTimeout === "function") clearTimeout(handle); };
  const frameTasks = new Map();
  /* Large flows look up nodes constantly (edges, snapping, run updates).  The
     index follows the document/array identity, so nodes added, removed or
     replaced in place invalidate it without a manual rebuild call. */
  const nodesById = { doc: null, list: null, length: 0, map: new Map() };
  const ACTIVE_RUNS = new Set(["queued", "running", "paused"]);
  const DELTA_EVENTS = new Set(["assistant_delta", "reasoning_delta", "swarm_agent_delta"]);
  const SWARM_DEFAULTS = {
    max_agents: 6, max_steps: 8, max_rounds: 2, max_seconds: 300,
    max_messages: 200, max_message_chars: 4000, max_memory_entries: 120,
    max_memory_chars: 8000, max_context_chars: 200000, max_agent_context_chars: 64000,
    max_context_item_chars: 24000, allow_write: false, allow_ipc: true, shared_memory: true,
  };
  const DESIGNER_CONTROLS = ["agents-select", "agents-new", "agents-save", "agents-add-ability", "agents-add-node", "agents-find-toggle", "agents-run-menu", "agents-more"];
  const RUN_CONTROLS = ["agents-run"];
  const PRIMARY_FIELDS = {
    end: ["output"],
    llm: ["model", "prompt", "format", "fallback_model"],
    agent: ["model", "goal", "abilities", "fallback_model"],
    swarm: ["model", "goal", "criteria", "strategy", "rounds", "effort"],
    tool: ["ability", "input"],
    condition: ["expression"],
    assert: ["expression", "message"],
    validate: ["target", "schema", "mode"],
    transform: ["template", "format"],
    delay: ["seconds"],
    human: ["prompt", "kind", "choices"],
    subagent: ["agent", "input"],
    foreach: ["items", "agent", "input"],
    memory: ["action", "key", "value"],
    note: ["text"],
  };
  const state = {
    meta: null, agents: [], doc: null, selected: new Set(), edge: null, issues: [], ready: false,
    undo: [], redo: [], clipboard: null, dirty: false, zoom: 1, pan: { x: 30, y: 30 },
    breakpoints: new Set(), drag: null, connection: null, run: null, feed: null, runToken: 0, eventSeq: 0,
    validationTimer: null, bottomTabs: null, sectionState: new Map(), recentTypes: [],
    docVersion: 0,
    find: { open: false, query: "", matches: [], index: -1, ui: null },
    panMode: false, spaceHeld: false, follow: false,
    guides: [], dropEdge: null, dragPoint: null, runFilter: "", lastRunNode: null, renaming: null, inspectorSelection: "", runPanelKey: "", lastClick: null,
    runHideDeltas: null,
    nodeElements: new Map(), edgeElements: new Map(), connectionGhost: null, guideElements: { x: null, y: null }, nodeRenderPending: false,
  };
  let layout = null;
  let dockLayout = null;
  let minimapPalette = null;
  let minimapTimer = 0;

  /* The minimap is decorative; 10 fps is enough while dragging or panning and
     keeps its per-node canvas work off the pointer-critical path. */
  function scheduleMinimap(immediate = false) {
    if (immediate) { cancelMinimap(); drawMinimapNow(); return; }
    if (minimapTimer) return;
    minimapTimer = setTimeout(() => {
      minimapTimer = 0;
      drawMinimapNow();
    }, 100);
  }

  function cancelMinimap() {
    if (!minimapTimer) return;
    clearTimeout(minimapTimer);
    minimapTimer = 0;
  }

  function copy(value) { return JSON.parse(JSON.stringify(value)); }

  /* Coalesce DOM updates to one callback per animation frame.  Pointer events
     arrive faster than the compositor can use them, so painting each one makes
     dragging feel heavy without looking any smoother. */
  function frame(key, task) {
    const pending = frameTasks.get(key);
    if (pending) { pending.task = task; return; }
    const entry = { task, handle: 0 };
    entry.handle = raf(() => {
      frameTasks.delete(key);
      if (entry.task) entry.task();
    });
    frameTasks.set(key, entry);
  }

  function cancelFrame(key) {
    const pending = frameTasks.get(key);
    if (!pending) return;
    caf(pending.handle);
    frameTasks.delete(key);
  }

  function cancelAllFrames() {
    for (const pending of frameTasks.values()) caf(pending.handle);
    frameTasks.clear();
  }
  function nodeById(id) {
    if (!state.doc) return undefined;
    const list = state.doc.nodes;
    if (nodesById.doc !== state.doc || nodesById.list !== list || nodesById.length !== list.length) {
      nodesById.doc = state.doc; nodesById.list = list; nodesById.length = list.length;
      nodesById.map = new Map(list.map(node => [node.id, node]));
    }
    return nodesById.map.get(id);
  }
  function activeRun() { return state.run && ACTIVE_RUNS.has(state.run.status); }
  function svg(tag, attributes) {
    const element = document.createElementNS(SVG_NS, tag);
    for (const [key, value] of Object.entries(attributes || {})) element.setAttribute(key, value);
    return element;
  }
  function uniqueId(prefix, reserved) {
    const used = reserved || new Set(state.doc.nodes.map(node => node.id));
    let index = 1;
    while (used.has(`${prefix}-${index}`)) index += 1;
    const id = `${prefix}-${index}`;
    used.add(id);
    return id;
  }

  /* The run dialog accepts JSON or plain text.  Plain text becomes input.message,
     the key every starter template uses; JSON values pass through unchanged. */
  function parseRunInput(raw) {
    let value = raw;
    try {
      value = JSON.parse(raw);
    } catch (error) {
      return { message: raw };
    }
    return typeof value === "string" ? { message: value } : value;
  }
  function blankAgent() {
    return {
      aml: "1.4", kind: "agent", id: "untitled-agent", name: "Untitled agent", description: "", version: "1.0.0", icon: "◈",
      model: "", model_settings: {}, vars: {}, swarm: copy(SWARM_DEFAULTS),
      settings: { max_steps: 100, timeout: 120, on_error: "stop", max_output_chars: 60000 },
      abilities: [], nodes: [
        { id: "start", type: "start", label: "Start", x: 80, y: 160, data: {} },
        { id: "end", type: "end", label: "End", x: 420, y: 160, data: { output: "{{ last }}" } },
      ], edges: [{ from: "start", to: "end", when: "always" }], trigger: { type: "manual" },
      schedule: { enabled: false, type: "manual", interval_minutes: 60, cron: "", once_at: "", input: {} },
    };
  }

  function markDirty(value = true) {
    state.dirty = value;
    $("agents-save").textContent = value ? "Save *" : "Save";
    $("agents-save").title = value ? "Unsaved changes" : "Saved";
    renderStatus();
    renderHistoryControls();
  }

  function setDesignerReady(value) {
    state.ready = value;
    for (const id of DESIGNER_CONTROLS) $(id).disabled = !value;
  }

  function pushHistory(label = "Edit") {
    if (!state.doc) return;
    state.undo.push({ doc: copy(state.doc), label });
    if (state.undo.length > 60) state.undo.shift();
    state.redo = [];
  }

  function mutate(task, render = true, label = "Edit") {
    pushHistory(label);
    task();
    state.docVersion += 1;
    markDirty();
    if (render) renderAll();
    else { renderNodes(); renderEdges(); renderSelectionToolbar(); renderStatus(); renderHistoryControls(); }
    scheduleValidation();
  }

  function undo() {
    const entry = state.undo.pop();
    if (!entry) return;
    state.redo.push({ doc: copy(state.doc), label: entry.label });
    state.doc = entry.doc;
    state.selected.clear(); state.edge = null;
    markDirty(); renderAll(); scheduleValidation(); refreshFind();
  }

  function redo() {
    const entry = state.redo.pop();
    if (!entry) return;
    state.undo.push({ doc: copy(state.doc), label: entry.label });
    state.doc = entry.doc;
    state.selected.clear(); state.edge = null;
    markDirty(); renderAll(); scheduleValidation(); refreshFind();
  }

  function loadDocument(document) {
    stopWatching();
    cancelAllFrames();
    cancelRunEvents();
    cancelMinimap();
    state.drag = null;
    state.connection = null;
    state.guides = [];
    renderGuides();
    removeConnectionGhost();
    $("agents-canvas").classList.remove("panning", "connecting");
    state.doc = copy(document || blankAgent());
    state.doc.nodes = state.doc.nodes || [];
    for (const node of state.doc.nodes) {
      node.data = node.data && typeof node.data === "object" && !Array.isArray(node.data) ? node.data : {};
    }
    state.doc.edges = state.doc.edges || [];
    state.doc.abilities = state.doc.abilities || [];
    state.doc.model_settings = state.doc.model_settings || {};
    state.doc.vars = state.doc.vars && typeof state.doc.vars === "object" && !Array.isArray(state.doc.vars)
      ? state.doc.vars : {};
    state.doc.settings = { max_steps: 100, timeout: 120, on_error: "stop", max_output_chars: 60000, ...(state.doc.settings || {}) };
    state.doc.swarm = { ...SWARM_DEFAULTS, ...(state.doc.swarm || {}) };
    state.doc.schedule = { enabled: false, type: "manual", interval_minutes: 60, cron: "", once_at: "", input: {}, ...(state.doc.schedule || {}) };
    state.selected.clear(); state.edge = null; state.issues = []; state.undo = []; state.redo = [];
    state.breakpoints = new Set(); state.run = null; state.eventSeq = 0; state.zoom = 1; state.pan = { x: 30, y: 30 };
    state.follow = false; state.runFilter = ""; state.lastRunNode = null;
    state.renaming = null; state.inspectorSelection = ""; state.runPanelKey = "";
    state.lastClick = null;
    state.panMode = false; state.spaceHeld = false; updatePanMode();
    closeFind();
    setDesignerReady(true);
    markDirty(false); renderAll(); renderRun(); renderSchedule(); renderSpec(); fitFlow(); scheduleValidation();
  }

  async function refreshAgents(preferred) {
    const data = await api("/api/agents");
    state.agents = data.agents;
    const select = Studio.ui.render($("agents-select"));
    select.append(h("option", { value: "", text: "Unsaved / new agent" }));
    for (const agent of state.agents) select.append(h("option", { value: agent.id, text: `${agent.name}${agent.active ? " · running" : ""}` }));
    select.value = preferred || (state.doc && state.agents.some(agent => agent.id === state.doc.id) ? state.doc.id : "");
    if (state.doc) renderInspector();
  }

  async function confirmDiscard() {
    return !state.dirty || Studio.confirm({ title: "Discard unsaved changes?", message: "This agent has unsaved changes. Continue without saving them?", confirmLabel: "Discard changes", danger: true });
  }

  async function openAgent(id) {
    if (!id || !await confirmDiscard()) { $("agents-select").value = state.doc && state.agents.some(agent => agent.id === state.doc.id) ? state.doc.id : ""; return; }
    const data = await Studio.run(() => api(`/api/agents/${encodeURIComponent(id)}`));
    if (data) { loadDocument(data.agent); if (data._meta.active_run) watchRun(data._meta.active_run, true); }
  }

  async function saveAgent() {
    if (!state.doc) return null;
    const data = await Studio.run(() => api("/api/agents", { body: state.doc }));
    if (!data) return null;
    const id = data.agent.id;
    state.doc = data.agent;
    markDirty(false);
    await refreshAgents(id);
    renderAll();
    Studio.toast("Agent saved and validated.", "success");
    return data.agent;
  }

  async function ensureSaved() {
    if (!state.doc) return null;
    return state.dirty || !state.agents.some(agent => agent.id === state.doc.id) ? saveAgent() : state.doc;
  }

  function scheduleValidation() {
    clearTimeout(state.validationTimer);
    state.validationSeq = (state.validationSeq || 0) + 1;
    const sequence = state.validationSeq;
    state.validationTimer = setTimeout(() => validate(sequence), 250);
  }

  async function validate(sequence) {
    if (!state.doc) return;
    const agentDocument = copy(state.doc);
    try {
      const result = await api("/api/agents/validate", { body: agentDocument });
      if (sequence !== state.validationSeq) return;
      const previousIssues = JSON.stringify(state.issues);
      state.issues = result.issues || [];
      renderProblems();
      renderHistoryControls();
      if (JSON.stringify(state.issues) !== previousIssues && !document.querySelector(".agent-node-rename")) renderNodes();
    } catch (error) { Studio.toast(error.message, "error"); }
  }

  function renderProblems() {
    const errors = state.issues.filter(issue => issue.level === "error").length;
    $("agents-problem-count").textContent = String(state.issues.length);
    $("agents-problem-count").className = `badge square${errors ? " danger" : state.issues.length ? " warning" : " completed"}`;
    const root = Studio.ui.render($("agents-problems"));
    if (!state.issues.length) {
      root.append(Studio.ui.empty({ compact: true, icon: "check", title: "No validation problems", text: "Save or run to validate the flow again." }));
      renderStatus();
      return;
    }
    for (const issue of state.issues) root.append(h("div", { class: "problem-row" }, Studio.ui.badge(issue.level, issue.level === "error" ? "danger" : "warning"),
      h("button", { class: "link-button grow", type: "button", onclick: () => {
        if (issue.node && nodeById(issue.node)) { selectNode(issue.node); focusNode(issue.node); }
      }, text: `${issue.node ? `${issue.node}: ` : ""}${issue.message}${issue.field ? ` (${issue.field})` : ""}` })));
    renderStatus();
  }

  function renderPalette() {
    if (!state.meta) return;
    const query = $("agents-node-search").value.trim().toLowerCase();
    const root = Studio.ui.render($("agents-node-palette"));
    let category = "";
    let shown = 0;
    for (const [type, meta] of Object.entries(state.meta.node_types)) {
      if (type === "start" && state.doc && state.doc.nodes.some(node => node.type === "start")) continue;
      const words = query.split(/\s+/).filter(Boolean);
      const text = `${meta.label} ${meta.description} ${meta.category} ${type}`.toLowerCase();
      if (query && !words.every(word => text.includes(word))) continue;
      if (meta.category !== category) { category = meta.category; root.append(h("div", { class: "agent-palette-group", text: category })); }
      const item = h("button", { class: "agent-palette-item", type: "button", draggable: true, title: `${meta.description} · click to add at the centre, or drag onto the canvas`,
        onclick: () => addNode(type), ondragstart: event => { event.dataTransfer.setData("application/x-mariana-node", type); event.dataTransfer.effectAllowed = "copy"; } },
        h("span", { class: "node-glyph", text: meta.icon }), h("span", { text: meta.label }));
      root.append(item);
      shown += 1;
    }
    if (!shown) root.append(h("p", { class: "help", text: query ? `No node type matches “${query}”.` : "No node types are available." }));
  }

  function renderAbilities() {
    $("agents-ability-count").textContent = String(state.doc.abilities.length);
    const root = Studio.ui.render($("agents-abilities"));
    if (!state.doc.abilities.length) root.append(h("p", { class: "help", text: "Grant file, HTTP, memory, math and other abilities here, then use them in Ability or AI agent nodes." }));
    for (const ability of state.doc.abilities) {
      const meta = state.meta.abilities[ability.type] || {};
      const keys = Object.keys(ability.config || {});
      root.append(h("div", { class: "ability-item stack tight" }, h("div", { class: "row" }, h("input", { type: "checkbox", checked: ability.enabled !== false, "aria-label": `Enable ${ability.id}`, onchange: event => mutate(() => { ability.enabled = event.currentTarget.checked; }, true, "Toggle ability") }),
        h("strong", { class: "grow truncate", text: ability.label || ability.id }), Studio.ui.iconButton("edit", `Edit ${ability.id}`, () => abilityDialog(ability), { small: true }),
        Studio.ui.iconButton("copy", `Duplicate ${ability.id}`, () => {
          const base = `${ability.id}-copy`; let index = 1; let id = base;
          while (state.doc.abilities.some(item => item.id === id)) { index += 1; id = `${base}-${index}`; }
          mutate(() => state.doc.abilities.push({ ...copy(ability), id, label: `${ability.label || ability.id} copy` }), true, "Duplicate ability");
        }, { small: true }),
        Studio.ui.iconButton("trash", `Remove ${ability.id}`, () => Studio.confirm({ title: "Remove ability?", message: `Nodes using ${ability.id} will fail validation until they pick another ability.`, confirmLabel: "Remove", danger: true }).then(ok => { if (ok) mutate(() => { state.doc.abilities = state.doc.abilities.filter(item => item !== ability); }, true, "Remove ability"); }), { small: true })),
        h("span", { class: "muted tiny", text: `${ability.id} · ${meta.label || ability.type}${keys.length ? ` · ${keys.join(", ")}` : ""}${ability.enabled === false ? " · disabled" : ""}` })));
    }
  }

  function abilityDialog(existing) {
    const current = existing || { id: "", type: "http.fetch", label: "", enabled: true, config: {} };
    const type = h("select", {}, Object.entries(state.meta.abilities).map(([id, meta]) => h("option", { value: id, text: meta.label, selected: id === current.type })));
    const id = h("input", { value: current.id, placeholder: "unique-ability-id", readOnly: Boolean(existing) });
    const label = h("input", { value: current.label || "" });
    const config = h("textarea", { rows: 7, value: JSON.stringify(current.config || state.meta.abilities[current.type].config || {}, null, 2), spellcheck: false });
    const description = h("p", { class: "help" });
    const update = () => {
      const meta = state.meta.abilities[type.value];
      description.textContent = `${meta.description}${meta.danger ? " Shell access is disabled unless explicitly enabled on the server." : ""}`;
      if (!existing) config.value = JSON.stringify(meta.config || {}, null, 2);
    };
    const view = Studio.overlay.modal({ key: existing ? `ability:${existing.id}` : "ability:new", title: existing ? `Edit ${current.id}` : "Grant an ability", subtitle: "Ability configuration is separate from the input supplied by each node.", wide: true });
    if (view.reused) return;
    type.addEventListener("change", update); update();
    view.body.append(Studio.ui.field("Ability type", type, description.textContent), description, Studio.ui.field("Id", id), Studio.ui.field("Label", label), Studio.ui.field("Configuration (JSON)", config));
    view.footer.append(Studio.ui.button(existing ? "Save ability" : "Add ability", { kind: "primary", onclick: () => {
      try {
        const abilityId = id.value.trim();
        if (!/^[a-z0-9][a-z0-9-]*$/.test(abilityId)) throw new Error("Use a lowercase id with letters, numbers and dashes");
        if (!existing && state.doc.abilities.some(item => item.id === abilityId)) throw new Error("That ability id is already used");
        const parsed = JSON.parse(config.value || "{}");
        if (!parsed || Array.isArray(parsed) || typeof parsed !== "object") throw new Error("Configuration must be a JSON object");
        mutate(() => {
          if (existing) Object.assign(existing, { type: type.value, label: label.value.trim(), config: parsed });
          else state.doc.abilities.push({ id: abilityId, type: type.value, label: label.value.trim() || state.meta.abilities[type.value].label, enabled: true, config: parsed });
        });
        view.close();
      } catch (error) { Studio.toast(error.message, "error"); }
    } }));
  }

  function outputPosition(node, when) {
    return { x: node.x + NODE_WIDTH, y: node.y + (node.type === "condition" ? (when === "false" ? 58 : 32) : 41) };
  }
  function inputPosition(node) { return { x: node.x, y: node.y + 41 }; }
  function edgePath(source, target) {
    const distance = Math.max(45, Math.abs(target.x - source.x) * 0.45);
    return `M ${source.x} ${source.y} C ${source.x + distance} ${source.y}, ${target.x - distance} ${target.y}, ${target.x} ${target.y}`;
  }

  /* -- pure helpers (exported for tests) ----------------------------------- */

  function findNodes(nodes, query) {
    const words = String(query || "").trim().toLowerCase().split(/\s+/).filter(Boolean);
    if (!words.length) return [];
    return nodes.filter(node => {
      const text = `${node.label || ""} ${node.type || ""} ${node.id || ""} ${JSON.stringify(node.data || {})}`.toLowerCase();
      return words.every(word => text.includes(word));
    }).map(node => node.id);
  }

  function rankNodeTypes(nodeTypes, query) {
    const words = String(query || "").trim().toLowerCase().split(/\s+/).filter(Boolean);
    return Object.entries(nodeTypes || {}).map(([type, meta]) => {
      const label = String(meta.label || type).toLowerCase();
      const text = `${label} ${meta.description || ""} ${type} ${meta.category || ""}`.toLowerCase();
      const score = !words.length ? 1
        : !words.every(word => text.includes(word)) ? 0
        : words.every(word => label.includes(word)) ? 3 : 1;
      return { type, meta, score };
    }).filter(entry => entry.score > 0)
      .sort((left, right) => words.length ? right.score - left.score || left.meta.label.localeCompare(right.meta.label) : 0);
  }

  function alignPositions(nodes, mode) {
    if (nodes.length < 2) return [];
    const left = Math.min(...nodes.map(node => node.x));
    const right = Math.max(...nodes.map(node => node.x + NODE_WIDTH));
    const top = Math.min(...nodes.map(node => node.y));
    const bottom = Math.max(...nodes.map(node => node.y + NODE_HEIGHT));
    return nodes.map(node => {
      let x = node.x;
      let y = node.y;
      if (mode === "left") x = left;
      else if (mode === "hcenter") x = (left + right - NODE_WIDTH) / 2;
      else if (mode === "right") x = right - NODE_WIDTH;
      else if (mode === "top") y = top;
      else if (mode === "vcenter") y = (top + bottom - NODE_HEIGHT) / 2;
      else if (mode === "bottom") y = bottom - NODE_HEIGHT;
      return { id: node.id, x: Math.round(x), y: Math.round(y) };
    });
  }

  function distributePositions(nodes, axis) {
    if (nodes.length < 3) return [];
    const key = axis === "y" ? "y" : "x";
    const sorted = [...nodes].sort((left, right) => left[key] - right[key]);
    const first = sorted[0][key];
    const last = sorted[sorted.length - 1][key];
    const step = (last - first) / (sorted.length - 1);
    return sorted.map((node, index) => ({
      id: node.id,
      x: key === "x" ? Math.round(first + step * index) : node.x,
      y: key === "y" ? Math.round(first + step * index) : node.y,
    }));
  }

  /* Sample the cubic used by edgePath so a drop can land on a connection. */
  function cubicPoint(source, target, t) {
    const distance = Math.max(45, Math.abs(target.x - source.x) * 0.45);
    const x1 = source.x + distance; const y1 = source.y;
    const x2 = target.x - distance; const y2 = target.y;
    const u = 1 - t;
    return {
      x: u * u * u * source.x + 3 * u * u * t * x1 + 3 * u * t * t * x2 + t * t * t * target.x,
      y: u * u * u * source.y + 3 * u * u * t * y1 + 3 * u * t * t * y2 + t * t * t * target.y,
    };
  }

  function nearestEdge(doc, point, threshold = 24) {
    if (!doc) return -1;
    const nodes = new Map((doc.nodes || []).map(node => [node.id, node]));
    let best = -1;
    let bestDistance = threshold * threshold;
    (doc.edges || []).forEach((edge, index) => {
      const from = nodes.get(edge.from);
      const to = nodes.get(edge.to);
      if (!from || !to) return;
      const source = outputPosition(from, edge.when);
      const target = inputPosition(to);
      for (let step = 0; step <= 24; step += 1) {
        const sample = cubicPoint(source, target, step / 24);
        const dx = sample.x - point.x; const dy = sample.y - point.y;
        const distance = dx * dx + dy * dy;
        if (distance < bestDistance) { bestDistance = distance; best = index; }
      }
    });
    return best;
  }

  function updateDropEdge() {
    const point = state.dragPoint;
    state.dragPoint = null;
    if (!point || !state.doc) return;
    const index = nearestEdge(state.doc, point, Math.max(14, 20 / state.zoom));
    if (index !== state.dropEdge) { state.dropEdge = index; renderEdges(); }
  }

  function edgeMidpoint(doc, index) {
    const edge = (doc.edges || [])[index];
    if (!edge) return null;
    const from = (doc.nodes || []).find(node => node.id === edge.from);
    const to = (doc.nodes || []).find(node => node.id === edge.to);
    if (!from || !to) return null;
    return cubicPoint(outputPosition(from, edge.when), inputPosition(to), 0.5);
  }

  function marker(id, className) {
    const element = svg("marker", {
      id, viewBox: "0 0 10 10", refX: "9", refY: "5", markerWidth: "7", markerHeight: "7", orient: "auto-start-reverse",
    });
    element.append(svg("path", { d: "M 0 0 L 10 5 L 0 10 z", class: className }));
    return element;
  }

  let edgesStructure = null;

  function renderEdges() {
    const root = $("agents-edges");
    const parts = [];
    let drawable = 0;
    for (const edge of state.doc.edges) {
      const from = nodeById(edge.from);
      const to = nodeById(edge.to);
      if (!from || !to) { parts.push("missing"); continue; }
      drawable += 1;
      parts.push(`${edge.from}\u0000${edge.to}\u0000${edge.when || ""}\u0000${edge.label || ""}`);
    }
    const structure = parts.join("\u0001");
    /* Node edits and moves do not change the edge list; update the existing
       SVG nodes instead of recreating every path and listener. */
    if (edgesStructure !== null && structure === edgesStructure && state.edgeElements.size === drawable) {
      for (const index of state.edgeElements.keys()) updateEdgeElement(index);
      if (state.connection) ensureConnectionGhost();
      return;
    }
    edgesStructure = structure;
    Studio.ui.render(root);
    state.edgeElements.clear();
    state.connectionGhost = null;
    const defs = svg("defs");
    defs.append(
      marker("agent-arrow-always", "agent-arrow-always"), marker("agent-arrow-true", "agent-arrow-true"),
      marker("agent-arrow-false", "agent-arrow-false"), marker("agent-arrow-selected", "agent-arrow-selected"),
      marker("agent-arrow-drop", "agent-arrow-drop"));
    root.append(defs);
    state.doc.edges.forEach((edge, index) => {
      const from = nodeById(edge.from); const to = nodeById(edge.to);
      if (!from || !to) return;
      const tone = state.edge === index ? "selected"
        : state.dropEdge === index ? "drop"
        : edge.when === "true" ? "true" : edge.when === "false" ? "false" : "always";
      const middle = edgeMidpoint(state.doc, index);
      const d = edgePath(outputPosition(from, edge.when), inputPosition(to));
      /* The visible stroke is 2px, which is hard to hit.  A transparent path
         directly underneath widens the click target without changing the look. */
      const hit = svg("path", { d, class: "agent-edge-hit", "aria-hidden": "true" });
      const path = svg("path", {
        d,
        class: `agent-edge agent-edge-${edge.when || "always"}${state.edge === index ? " selected" : ""}${state.dropEdge === index ? " drop-target" : ""}`,
        "marker-end": `url(#agent-arrow-${tone})`,
      });
      const cached = { path, hit, label: null, middle };
      hit.addEventListener("pointerdown", event => {
        if (event.button !== 0 || state.panMode || state.spaceHeld) return;
        event.preventDefault();
        $("agents-canvas").focus({ preventScroll: true });
        event.stopPropagation();
        if (registerClick("edge", index, { x: event.clientX, y: event.clientY })) { openQuickAdd({ edge: index, point: cached.middle }); return; }
        selectEdge(index);
      });
      hit.addEventListener("dblclick", event => {
        event.preventDefault(); event.stopPropagation();
        openQuickAdd({ edge: index, point: cached.middle });
      });
      hit.addEventListener("contextmenu", event => { event.preventDefault(); event.stopPropagation(); selectEdge(index); edgeMenu(event, index); });
      root.append(hit, path);
      if (edge.when && edge.when !== "always" || edge.label) {
        const label = svg("text", { x: middle.x, y: middle.y - 7, class: "agent-edge-label", "text-anchor": "middle" });
        label.textContent = edge.label || edge.when;
        root.append(label);
        cached.label = label;
      }
      state.edgeElements.set(index, cached);
    });
    if (state.connection) ensureConnectionGhost();
  }

  function updateEdgeElement(index, paintState = true) {
    const cached = state.edgeElements.get(index);
    const edge = state.doc.edges[index];
    if (!cached || !edge) return;
    const from = nodeById(edge.from); const to = nodeById(edge.to);
    if (!from || !to) return;
    const source = outputPosition(from, edge.when);
    const target = inputPosition(to);
    const d = edgePath(source, target);
    cached.path.setAttribute("d", d);
    if (cached.hit) cached.hit.setAttribute("d", d);
    const middle = cubicPoint(source, target, 0.5);
    cached.middle = middle;
    if (cached.label) {
      cached.label.setAttribute("x", middle.x);
      cached.label.setAttribute("y", middle.y - 7);
    }
    if (!paintState) return;
    const selected = state.edge === index;
    const tone = selected ? "selected"
      : state.dropEdge === index ? "drop"
      : edge.when === "true" ? "true" : edge.when === "false" ? "false" : "always";
    cached.path.setAttribute("class", `agent-edge agent-edge-${edge.when || "always"}${selected ? " selected" : ""}${state.dropEdge === index ? " drop-target" : ""}`);
    cached.path.setAttribute("marker-end", `url(#agent-arrow-${tone})`);
  }

  function updateEdgesTouching(ids) {
    state.doc.edges.forEach((edge, index) => {
      if (ids.has(edge.from) || ids.has(edge.to)) updateEdgeElement(index, false);
    });
  }

  function ensureConnectionGhost() {
    if (!state.connection) return null;
    if (state.connectionGhost && state.connectionGhost.isConnected) return state.connectionGhost;
    const path = svg("path", { class: "agent-edge selected", "marker-end": "url(#agent-arrow-selected)" });
    $("agents-edges").append(path);
    state.connectionGhost = path;
    return path;
  }

  function paintConnectionGhost() {
    if (!state.connection) return;
    const source = nodeById(state.connection.from);
    const path = ensureConnectionGhost();
    if (source && path) path.setAttribute("d", edgePath(outputPosition(source, state.connection.when), state.connection.point));
  }

  function removeConnectionGhost() {
    if (state.connectionGhost) state.connectionGhost.remove();
    state.connectionGhost = null;
  }

  function toggleBreakpoint(id) {
    if (state.breakpoints.has(id)) state.breakpoints.delete(id); else state.breakpoints.add(id);
    if (state.run && state.run.debug && activeRun()) api(`/api/agents/runs/${encodeURIComponent(state.run.id)}/breakpoints`, { body: { breakpoints: Array.from(state.breakpoints) } })
      .catch(error => Studio.toast(error.message, "error"));
    renderNodes();
    renderInspector();
  }

  /* Selection only changes classes and the inspector, so it never needs to
     rebuild the canvas.  Re-rendering here replaced every node on each click,
     which made clicks feel sluggish and killed native double-click targets. */
  function paintEdgeSelection(previous, next) {
    for (const index of new Set([previous, next])) {
      if (index === null || index === undefined) continue;
      const cached = state.edgeElements.get(index);
      const edge = state.doc && state.doc.edges[index];
      if (!cached || !edge) continue;
      const selected = index === next;
      cached.path.classList.toggle("selected", selected);
      cached.path.classList.toggle("drop-target", index === state.dropEdge);
      const tone = selected ? "selected"
        : index === state.dropEdge ? "drop"
        : edge.when === "true" ? "true" : edge.when === "false" ? "false" : "always";
      cached.path.setAttribute("marker-end", `url(#agent-arrow-${tone})`);
    }
  }

  function selectNode(id, additive = false) {
    const previousEdge = state.edge;
    state.edge = null;
    if (additive) { if (state.selected.has(id)) state.selected.delete(id); else state.selected.add(id); }
    else state.selected = new Set([id]);
    if (previousEdge !== null) paintEdgeSelection(previousEdge, null);
    paintSelection();
    renderInspector(); renderSelectionToolbar(); renderStatus();
  }

  function selectEdge(index) {
    const previous = state.edge;
    state.edge = index;
    state.selected.clear();
    paintEdgeSelection(previous, index);
    paintSelection();
    renderInspector(); renderSelectionToolbar(); renderStatus();
  }

  function shortText(value, limit = 46) {
    const text = String(value === undefined || value === null ? "" : value).replace(/\s+/g, " ").trim();
    return text.length > limit ? `${text.slice(0, limit - 1)}…` : text;
  }

  function modelLabel(id) {
    if (!id) return "Default model";
    const entry = (Studio.models.list || []).find(item => item.id === id);
    return entry ? entry.name : id;
  }

  function abilityLabel(id) {
    if (!id) return "No ability";
    const ability = (state.doc.abilities || []).find(item => item.id === id);
    return ability ? ability.label || ability.id : id;
  }

  function agentLabel(id) {
    if (!id) return "No agent";
    const agent = (state.agents || []).find(item => item.id === id);
    return agent ? agent.name : id;
  }

  /* A one-line description rendered on the node card, ComfyUI style: the graph
     should be readable without opening the inspector for every step. */
  function nodeSummary(node) {
    const data = node.data || {};
    switch (node.type) {
      case "start": return "Run input";
      case "end": return shortText(data.output || "Returns the previous output");
      case "llm": return `${modelLabel(data.model)}${data.fallback_model ? " + fallback" : ""} · ${shortText(data.prompt || "Uses the previous output", 28)}`;
      case "agent": return `${modelLabel(data.model)}${data.fallback_model ? " + fallback" : ""} · ${shortText(data.goal || "Goal from previous output", 28)}`;
      case "tool": return `${abilityLabel(data.ability)}${data.input && Object.keys(data.input).length ? " · configured" : ""}`;
      case "condition": return shortText(data.expression || "true", 52);
      case "assert": return shortText(data.expression || "true", 52);
      case "validate": {
        const schema = data.schema || {};
        const type = Array.isArray(schema.type) ? schema.type.join(" / ") : schema.type || "value";
        return `${data.mode === "warn" ? "Warn on" : "Require"} ${type}`;
      }
      case "transform": return shortText(data.template || (data.assign && Object.keys(data.assign).length ? `Assigns ${Object.keys(data.assign).join(", ")}` : "Template"), 52);
      case "delay": return `Waits ${data.seconds === undefined ? 1 : data.seconds}s`;
      case "human": return shortText(data.prompt || (data.kind === "choice" ? "Choose an option" : data.kind === "text" ? "Ask for a response" : "Approve or reject"), 52);
      case "subagent": return `Run ${agentLabel(data.agent)}`;
      case "foreach": return `${agentLabel(data.agent)} per item`;
      case "memory": return `${data.action || "get"}${data.key ? ` · ${shortText(data.key, 24)}` : ""}`;
      case "note": return shortText(data.text || "Note", 52);
      default: return node.label;
    }
  }

  let nodesRenderedDoc = null;
  let nodesRenderedMeta = null;
  let inspectorPosition = null;
  let nodeSummaryCache = { doc: null, version: -1, agents: null, meta: null, models: -1, values: new Map() };

  function summariesForNodes() {
    const models = Studio.models ? Studio.models.sequence : 0;
    if (nodeSummaryCache.doc !== state.doc || nodeSummaryCache.version !== state.docVersion
        || nodeSummaryCache.agents !== state.agents || nodeSummaryCache.meta !== state.meta
        || nodeSummaryCache.models !== models) {
      nodeSummaryCache = {
        doc: state.doc, version: state.docVersion, agents: state.agents,
        meta: state.meta, models, values: new Map(),
      };
    }
    return nodeSummaryCache.values;
  }

  function buildNodeElement(node, meta, summary, flags) {
    const element = h("div", {
      class: `agent-node agent-node-${meta.category || "flow"}${flags.selected ? " selected" : ""}${flags.issue ? " has-issue" : ""}${flags.breakpoint ? " breakpoint" : ""}${flags.found ? " find-match" : ""}${flags.current ? " find-current" : ""} ${flags.status}`,
      dataset: { node: node.id }, style: { left: "0px", top: "0px", transform: `translate(${node.x}px, ${node.y}px)` }, tabIndex: 0, role: "button",
      "aria-label": `${node.label}, ${meta.label}`, "aria-selected": String(flags.selected), title: `${meta.description} · Click to edit · Double-click to rename`, "aria-description": meta.description,
      onkeydown: event => {
        if (event.key === "Enter" || event.key === " ") { event.preventDefault(); selectNode(node.id, event.shiftKey); }
        else if (event.key === "F2") { event.preventDefault(); renameNode(node); }
        else if (event.key === "ContextMenu") { event.preventDefault(); selectNode(node.id); const box = element.getBoundingClientRect(); nodeMenu({ clientX: box.left + 8, clientY: box.top + 8 }, node); }
      },
      ondblclick: event => { event.preventDefault(); event.stopPropagation(); startInlineRename(node, element); },
      oncontextmenu: event => { event.preventDefault(); selectNode(node.id); nodeMenu(event, node); },
    });
    element.append(h("div", { class: "agent-node-head" }, h("span", { class: "node-glyph", text: meta.icon }), h("span", { class: "agent-node-title", text: node.label }),
      Studio.ui.iconButton("bug", `Toggle breakpoint on ${node.label}`, event => { event.stopPropagation(); state.lastClick = null; toggleBreakpoint(node.id); }, { small: true }),
      Studio.ui.iconButton("more", `More actions for ${node.label}`, event => {
        event.stopPropagation();
        state.lastClick = null;
        const box = event.currentTarget.getBoundingClientRect();
        selectNode(node.id);
        nodeMenu({ clientX: box.left, clientY: box.bottom + 4 }, node);
      }, { small: true })),
      h("div", { class: "agent-node-type", title: `${meta.label} · ${node.id}`, text: summary }),
      h("div", { class: "agent-node-live hidden", "aria-hidden": "true" }));
    if (node.type !== "start") element.append(h("button", { class: "node-port input", type: "button", "aria-label": `Input to ${node.label}`, title: "Incoming connection", dataset: { input: node.id } }));
    if (node.type !== "end") {
      if (node.type === "condition") {
        for (const when of ["true", "false"]) element.append(h("button", { class: `node-port output-${when}`, type: "button", "aria-label": `${when} output from ${node.label}`, title: when, dataset: { output: node.id, when } }));
      } else element.append(h("button", { class: "node-port output", type: "button", "aria-label": `Output from ${node.label}`, title: "Drag to the next step", dataset: { output: node.id, when: "always" } }));
    }
    element.addEventListener("pointerdown", event => beginNodeDrag(event, node));
    element.__agentView = {
      node, id: node.id, label: node.label, summary, type: node.type,
      category: meta.category || "flow", metaLabel: meta.label, metaIcon: meta.icon,
      metaDescription: meta.description || "",
      live: element.querySelector(".agent-node-live"),
    };
    return element;
  }

  function sameNodeView(element, node, meta, summary) {
    const view = element.__agentView;
    return Boolean(view)
      && view.node === node
      && view.id === node.id
      && view.label === node.label
      && view.summary === summary
      && view.type === node.type
      && view.category === (meta.category || "flow")
      && view.metaLabel === meta.label
      && view.metaIcon === meta.icon
      && view.metaDescription === (meta.description || "");
  }

  function paintNodeElement(element, node, flags) {
    const transform = `translate(${node.x}px, ${node.y}px)`;
    if (element.style.transform !== transform) element.style.transform = transform;
    element.classList.toggle("selected", flags.selected);
    element.classList.toggle("has-issue", flags.issue);
    element.classList.toggle("breakpoint", flags.breakpoint);
    element.classList.toggle("find-match", flags.found);
    element.classList.toggle("find-current", flags.current);
    for (const name of RUN_NODE_STATES) element.classList.toggle(name, name === flags.status);
    const selected = String(flags.selected);
    if (element.getAttribute("aria-selected") !== selected) element.setAttribute("aria-selected", selected);
  }

  function renderNodes() {
    if (state.renaming) return;
    /* A run may stream updates or validation may land mid-drag.  Replacing the
       node elements then would throw away the elements being moved, so coalesce
       the rebuild until the pointer is released. */
    if (state.drag) { state.nodeRenderPending = true; return; }
    state.nodeRenderPending = false;
    const root = $("agents-nodes");
    if (nodesRenderedDoc !== state.doc || nodesRenderedMeta !== state.meta) {
      nodesRenderedDoc = state.doc;
      nodesRenderedMeta = state.meta;
      state.nodeElements.clear();
      root.replaceChildren();
    }
    const issueNodes = new Set(state.issues.filter(issue => issue.level === "error" && issue.node).map(issue => issue.node));
    const found = new Set(state.find.matches || []);
    const current = state.find.index >= 0 ? state.find.matches[state.find.index] : null;
    const desired = [];
    const live = new Set();
    const summaries = summariesForNodes();
    for (const node of state.doc.nodes) {
      live.add(node.id);
      const meta = state.meta.node_types[node.type] || { label: node.type, icon: "?" };
      let summary = summaries.get(node.id);
      if (summary === undefined) {
        summary = nodeSummary(node);
        summaries.set(node.id, summary);
      }
      const flags = {
        selected: state.selected.has(node.id),
        issue: issueNodes.has(node.id),
        breakpoint: state.breakpoints.has(node.id),
        found: found.has(node.id),
        current: current === node.id,
        status: state.run && state.run.nodes ? state.run.nodes[node.id] || "pending" : "pending",
      };
      let element = state.nodeElements.get(node.id);
      if (!element || !sameNodeView(element, node, meta, summary)) {
        element = buildNodeElement(node, meta, summary, flags);
        state.nodeElements.set(node.id, element);
      } else paintNodeElement(element, node, flags);
      desired.push(element);
    }
    for (const id of [...state.nodeElements.keys()]) {
      if (!live.has(id)) state.nodeElements.delete(id);
    }
    const existing = Array.from(root.children);
    let keep = 0;
    while (keep < existing.length && keep < desired.length && existing[keep] === desired[keep]) keep += 1;
    if (keep < existing.length || keep < desired.length) {
      for (let index = existing.length - 1; index >= keep; index -= 1) existing[index].remove();
      for (let index = keep; index < desired.length; index += 1) root.append(desired[index]);
    }
    renderHint();
  }

  const RUN_NODE_STATES = ["pending", "running", "ok", "error", "paused"];

  /* Run snapshots change node status frequently.  Toggle the state classes on
     the existing cards instead of replacing every node element each frame. */
  let nodePreviewSeq = -1;
  let nodePreviewRun = "";
  let nodePreviewMap = new Map();

  function runNodePreviews() {
    if (!state.run) {
      nodePreviewSeq = -1;
      nodePreviewRun = "";
      nodePreviewMap = new Map();
      return nodePreviewMap;
    }
    const events = state.run.events || [];
    const latest = events.length ? events[events.length - 1].seq || 0 : 0;
    if (latest !== nodePreviewSeq || state.run.id !== nodePreviewRun) {
      nodePreviewSeq = latest;
      nodePreviewRun = state.run.id;
      nodePreviewMap = nodeLivePreviews(state.run);
    }
    return nodePreviewMap;
  }

  function paintNodeLive(element, preview) {
    const live = element.__agentView && element.__agentView.live;
    if (!live) return;
    const text = preview && preview.live && preview.text
      ? `${preview.group ? `${preview.label}: ` : ""}${preview.text}`.slice(0, 160)
      : "";
    live.classList.toggle("hidden", !text);
    if (live.textContent !== text) live.textContent = text;
    element.classList.toggle("streaming", Boolean(text));
  }

  function paintRunNodes() {
    const statuses = state.run && state.run.nodes ? state.run.nodes : null;
    const previews = runNodePreviews();
    for (const [id, element] of state.nodeElements) {
      const status = (statuses && statuses[id]) || "pending";
      for (const name of RUN_NODE_STATES) element.classList.toggle(name, name === status);
      element.classList.toggle("breakpoint", state.breakpoints.has(id));
      paintNodeLive(element, previews.get(id));
    }
  }

  function viewportCenter() {
    const box = $("agents-canvas").getBoundingClientRect();
    return {
      x: (box.width / 2 - state.pan.x) / state.zoom - NODE_WIDTH / 2,
      y: (box.height / 2 - state.pan.y) / state.zoom - NODE_HEIGHT / 2,
    };
  }

  function nextOpenPosition() {
    const center = viewportCenter();
    const x = Math.round(center.x / 20) * 20;
    const y = Math.round(center.y / 20) * 20;
    let offset = 0;
    while (state.doc.nodes.some(node => Math.abs(node.x - (x + offset)) < NODE_WIDTH && Math.abs(node.y - (y + offset)) < NODE_HEIGHT) && offset < 200) offset += 30;
    return { x: x + offset, y: y + offset };
  }

  function beginNodeDrag(event, node) {
    if (event.button !== 0 || state.panMode || state.spaceHeld
      || event.target.closest(".node-port, .icon-button, .agent-node-rename")) return;
    event.preventDefault();
    const double = registerClick("node", node.id, { x: event.clientX, y: event.clientY });
    if (double && !event.shiftKey && !event.ctrlKey && !event.metaKey) {
      startInlineRename(node, event.currentTarget);
      return;
    }
    $("agents-canvas").focus({ preventScroll: true });
    if (event.shiftKey || event.ctrlKey || event.metaKey) { selectNode(node.id, true); return; }
    if (!state.selected.has(node.id)) {
      state.selected = new Set([node.id]);
      const previousEdge = state.edge;
      state.edge = null;
      if (previousEdge !== null) paintEdgeSelection(previousEdge, null);
    }
    const selectedNodes = state.doc.nodes.filter(item => state.selected.has(item.id));
    const others = state.doc.nodes.filter(item => !state.selected.has(item.id));
    state.drag = {
      kind: "nodes", start: { x: event.clientX, y: event.clientY }, primary: node.id,
      positions: selectedNodes.map(item => ({ node: item, x: item.x, y: item.y })),
      xTargets: others.flatMap(item => [item.x, item.x + NODE_WIDTH / 2, item.x + NODE_WIDTH]),
      yTargets: others.flatMap(item => [item.y, item.y + NODE_HEIGHT / 2, item.y + NODE_HEIGHT]),
      moved: false, history: false,
    };
    paintSelection();
    for (const item of selectedNodes) state.nodeElements.get(item.id)?.classList.add("dragging");
    document.addEventListener("pointermove", onPointerMove);
    document.addEventListener("pointerup", onPointerUp, { once: true });
  }

  function screenPosition(event) {
    return {
      x: event.clientX !== undefined ? event.clientX : event.x,
      y: event.clientY !== undefined ? event.clientY : event.y,
    };
  }

  function worldPoint(point, pan, zoom, origin) {
    return {
      x: (point.x - origin.left - pan.x) / zoom,
      y: (point.y - origin.top - pan.y) / zoom,
    };
  }

  /* Keep the world point under the cursor stationary while zooming. */
  function zoomedPan(pan, zoom, nextZoom, point, origin) {
    const world = worldPoint(point, pan, zoom, origin);
    return {
      x: point.x - origin.left - world.x * nextZoom,
      y: point.y - origin.top - world.y * nextZoom,
    };
  }

  function screenToWorld(event) {
    const box = $("agents-canvas").getBoundingClientRect();
    return worldPoint(screenPosition(event), state.pan, state.zoom, { left: box.left, top: box.top });
  }

  /* Re-rendering the canvas replaces the clicked element, so the native dblclick
     cannot be relied on.  Track the last pointerdown ourselves instead. */
  function registerClick(kind, id, point) {
    const now = Date.now();
    const near = state.lastClick && Math.abs(point.x - state.lastClick.x) < 10 && Math.abs(point.y - state.lastClick.y) < 10;
    const double = Boolean(near && state.lastClick.kind === kind && state.lastClick.id === id
      && now - state.lastClick.time < 400);
    state.lastClick = { kind, id, x: point.x, y: point.y, time: now };
    return double;
  }

  /* Snap the dragged group to another node's edges or centre and report the
     guide lines to paint.  Returns the adjusted delta in world units. */
  function snapDrag(drag, dx, dy) {
    const primary = drag.positions.find(item => item.node.id === drag.primary) || drag.positions[0];
    if (!primary) return { dx, dy, guides: [] };
    const threshold = 6 / state.zoom;
    const rawX = primary.x + dx; const rawY = primary.y + dy;
    const selfX = [rawX, rawX + NODE_WIDTH / 2, rawX + NODE_WIDTH];
    const selfY = [rawY, rawY + NODE_HEIGHT / 2, rawY + NODE_HEIGHT];
    let bestX = null; let bestY = null;
    for (const target of drag.xTargets) for (const point of selfX) {
      const difference = target - point;
      if (Math.abs(difference) <= threshold && (bestX === null || Math.abs(difference) < Math.abs(bestX.difference))) bestX = { difference, value: target };
    }
    for (const target of drag.yTargets) for (const point of selfY) {
      const difference = target - point;
      if (Math.abs(difference) <= threshold && (bestY === null || Math.abs(difference) < Math.abs(bestY.difference))) bestY = { difference, value: target };
    }
    const guides = [];
    if (bestX) guides.push({ axis: "x", value: bestX.value });
    if (bestY) guides.push({ axis: "y", value: bestY.value });
    return { dx: dx + (bestX ? bestX.difference : 0), dy: dy + (bestY ? bestY.difference : 0), guides };
  }

  function guideLine(axis) {
    let line = state.guideElements[axis];
    if (!line) {
      line = h("div", { class: `agent-guide ${axis} hidden` });
      $("agents-guides").append(line);
      state.guideElements[axis] = line;
    }
    return line;
  }

  function renderGuides() {
    const active = { x: false, y: false };
    for (const guide of state.guides) {
      const screen = guide.value * state.zoom + (guide.axis === "x" ? state.pan.x : state.pan.y);
      const line = guideLine(guide.axis);
      active[guide.axis] = true;
      line.classList.remove("hidden");
      if (guide.axis === "x") Studio.setStyle(line, { left: `${Math.round(screen)}px` });
      else Studio.setStyle(line, { top: `${Math.round(screen)}px` });
    }
    for (const axis of ["x", "y"]) {
      if (!active[axis] && state.guideElements[axis]) state.guideElements[axis].classList.add("hidden");
    }
  }

  function onPointerMove(event) {
    const drag = state.drag;
    if (!drag || !state.doc) return;
    const dx = (event.clientX - drag.start.x) / state.zoom; const dy = (event.clientY - drag.start.y) / state.zoom;
    if (drag.kind === "nodes" && Math.abs(dx) + Math.abs(dy) > 3 && !drag.history) {
      pushHistory("Move nodes"); drag.history = true; drag.moved = true; state.lastClick = null;
    }
    if (drag.kind === "nodes") {
      const snapped = event.altKey ? { dx, dy, guides: [] } : snapDrag(drag, dx, dy);
      state.guides = snapped.guides;
      /* Move freely and snap only when an alignment guide is in range: a
         20px grid made dragging look stepped and fought the pointer. */
      for (const item of drag.positions) {
        item.node.x = Math.round(item.x + snapped.dx);
        item.node.y = Math.round(item.y + snapped.dy);
      }
      frame("drag", paintNodeDrag);
    } else if (drag.kind === "marquee") {
      const box = $("agents-canvas").getBoundingClientRect();
      const left = Math.min(drag.start.x, event.clientX) - box.left; const top = Math.min(drag.start.y, event.clientY) - box.top;
      const width = Math.abs(event.clientX - drag.start.x); const height = Math.abs(event.clientY - drag.start.y);
      if (width + height > 6) state.lastClick = null;
      Studio.setStyle(drag.box, { left: `${left}px`, top: `${top}px`, width: `${width}px`, height: `${height}px` });
      const hits = state.doc.nodes.filter(node => {
        const x = node.x * state.zoom + state.pan.x; const y = node.y * state.zoom + state.pan.y;
        return x + NODE_WIDTH * state.zoom >= left && x <= left + width && y + NODE_HEIGHT * state.zoom >= top && y <= top + height;
      }).map(node => node.id);
      state.selected = drag.additive ? new Set([...drag.base, ...hits]) : new Set(hits);
      frame("selection", paintSelection);
    } else if (drag.kind === "pan") {
      state.pan.x = drag.pan.x + event.clientX - drag.start.x; state.pan.y = drag.pan.y + event.clientY - drag.start.y; applyTransform();
    }
  }

  function onPointerUp() {
    document.removeEventListener("pointermove", onPointerMove);
    const drag = state.drag;
    if (drag && drag.kind === "nodes" && drag.moved) paintNodeDrag();
    cancelFrame("drag"); cancelFrame("selection");
    state.drag = null;
    for (const element of state.nodeElements.values()) element.classList.remove("dragging");
    if (drag && drag.box) drag.box.remove();
    state.guides = []; renderGuides();
    $("agents-canvas").classList.remove("panning");
    if (!drag) return;
    if (state.nodeRenderPending) { state.nodeRenderPending = false; renderNodes(); }
    if (drag.kind === "nodes") {
      if (drag.moved) { markDirty(); scheduleValidation(); }
      refreshInspectorAfterDrag(); renderSelectionToolbar(); renderStatus();
    } else if (drag.kind === "marquee") {
      paintSelection();
      renderInspector(); renderSelectionToolbar(); renderStatus();
    }
  }

  function paintSelection() {
    for (const [id, element] of state.nodeElements) {
      const selected = state.selected.has(id);
      element.classList.toggle("selected", selected);
      element.setAttribute("aria-selected", String(selected));
    }
  }

  function paintNodePositions(ids) {
    for (const id of ids) {
      const node = nodeById(id);
      const element = state.nodeElements.get(id);
      if (node && element) Studio.setStyle(element, { transform: `translate(${node.x}px, ${node.y}px)` });
    }
  }

  function paintNodeDrag() {
    const drag = state.drag;
    if (!drag || drag.kind !== "nodes") return;
    paintNodePositions(drag.positions.map(item => item.node.id));
    updateEdgesTouching(state.selected);
    renderGuides();
    scheduleMinimap();
  }

  function beginCanvasPointer(event) {
    if (!state.doc || !state.meta) return;
    const panning = state.panMode || state.spaceHeld;
    if (event.target.closest(".agent-minimap, .agent-canvas-toolbar, .agent-find, .agent-selection-toolbar")) return;
    if (event.target.closest(".agent-node, .agent-edge, .agent-edge-hit") && !panning) return;
    if (event.button === 0 && !panning && registerClick("canvas", "canvas", { x: event.clientX, y: event.clientY })) {
      event.preventDefault();
      openQuickAdd({ point: screenToWorld(event) });
      return;
    }
    /* Space, H or the middle button pan; Shift and Ctrl both extend a
       marquee selection, which is what most flow editors teach. */
    const pan = event.button === 1 || panning;
    if (pan) {
      state.drag = { kind: "pan", start: { x: event.clientX, y: event.clientY }, pan: { ...state.pan } };
      $("agents-canvas").classList.add("panning");
    } else if (event.button === 0) {
      const additive = event.ctrlKey || event.metaKey || event.shiftKey;
      if (!additive) state.selected.clear();
      const box = h("div", { class: "agent-selection-box" }); $("agents-canvas").append(box);
      state.drag = { kind: "marquee", start: { x: event.clientX, y: event.clientY }, box, moved: true, additive, base: new Set(state.selected) };
      const previousEdge = state.edge;
      state.edge = null;
      if (previousEdge !== null) paintEdgeSelection(previousEdge, null);
      paintSelection();
    } else return;
    event.preventDefault();
    $("agents-canvas").focus({ preventScroll: true });
    document.addEventListener("pointermove", onPointerMove);
    document.addEventListener("pointerup", onPointerUp, { once: true });
  }

  /* Any visible node except Start (and the source itself) is a valid drop
     target: aiming at a 14px port was unnecessarily precise. */
  function connectionTarget(element, fromId) {
    if (!element || !fromId) return null;
    const nodeElement = element.closest(".agent-node");
    if (!nodeElement || nodeElement.dataset.node === fromId) return null;
    const node = nodeById(nodeElement.dataset.node);
    return node && node.type !== "start" ? nodeElement : null;
  }

  function cancelConnection() {
    if (!state.connection) return;
    cancelFrame("connection");
    state.connection = null;
    removeConnectionGhost();
    $("agents-canvas").classList.remove("connecting");
    for (const element of state.nodeElements.values()) element.classList.remove("connect-source", "connect-target");
  }

  function beginConnection(event) {
    if (!state.doc || !state.meta) return;
    const port = event.target.closest("[data-output]");
    if (!port || state.panMode || state.spaceHeld) return;
    if (event.button !== 0) return;
    event.preventDefault(); event.stopPropagation();
    state.lastClick = null;
    $("agents-canvas").focus({ preventScroll: true });
    state.connection = { from: port.dataset.output, when: port.dataset.when || "always", point: screenToWorld(event) };
    $("agents-canvas").classList.add("connecting");
    ensureConnectionGhost();
    const sourceElement = port.closest(".agent-node");
    if (sourceElement) sourceElement.classList.add("connect-source");
    const start = { x: event.clientX, y: event.clientY };
    let hovered = null;
    let moved = false;
    const highlight = target => {
      if (hovered === target) return;
      if (hovered) hovered.classList.remove("connect-target");
      hovered = target;
      if (hovered) hovered.classList.add("connect-target");
    };
    const move = moveEvent => {
      if (!state.connection) return;
      if (!moved && Math.abs(moveEvent.clientX - start.x) + Math.abs(moveEvent.clientY - start.y) > 4) moved = true;
      state.connection.point = screenToWorld(moveEvent);
      const target = document.elementFromPoint(moveEvent.clientX, moveEvent.clientY);
      highlight(connectionTarget(target, state.connection.from));
      frame("connection", paintConnectionGhost);
    };
    const up = upEvent => {
      document.removeEventListener("pointermove", move);
      cancelFrame("connection");
      const connection = state.connection; state.connection = null; $("agents-canvas").classList.remove("connecting");
      removeConnectionGhost();
      if (sourceElement) sourceElement.classList.remove("connect-source");
      highlight(null);
      const target = connectionTarget(document.elementFromPoint(upEvent.clientX, upEvent.clientY), connection && connection.from);
      if (connection && moved && target) {
        createEdge(connection.from, target.dataset.node, connection.when);
      } else if (connection) {
        /* Clicking a port (or dropping on empty space or the source) offers the
           next step instead of silently doing nothing. */
        openQuickAdd({ point: screenToWorld(upEvent), from: connection.from, when: connection.when });
      }
    };
    document.addEventListener("pointermove", move);
    document.addEventListener("pointerup", up, { once: true });
  }

  function createEdge(from, to, when = "always") {
    if (!from || !to || from === to) { Studio.toast("A node cannot connect to itself.", "info"); return; }
    const target = nodeById(to);
    if (!target || target.type === "start") { Studio.toast("The Start node has no input. Choose another node.", "info"); return; }
    if (state.doc.edges.some(edge => edge.from === from && edge.to === to && (edge.when || "always") === when)) {
      Studio.toast("That connection already exists.", "info"); return;
    }
    mutate(() => state.doc.edges.push({ from, to, when }), true, "Connect nodes");
  }

  function makeNode(type, position) {
    const meta = state.meta.node_types[type];
    if (!meta) return null;
    if (type === "start" && state.doc.nodes.some(node => node.type === "start")) {
      Studio.toast("This flow already has a Start node. Connect it to the new node instead.", "info");
      return null;
    }
    const spot = position || nextOpenPosition();
    return {
      id: uniqueId(type), type, label: meta.label,
      x: Math.max(0, Math.round(spot.x / 20) * 20), y: Math.max(0, Math.round(spot.y / 20) * 20),
      data: copy(meta.defaults || {}),
    };
  }

  function rememberType(type) {
    state.recentTypes = [type, ...state.recentTypes.filter(item => item !== type)].slice(0, 6);
    Studio.store.set("agents-recent-nodes", state.recentTypes);
  }

  /* A brand-new flow is just Start → End.  Adding the first real step there
     should wire it in automatically instead of making the user draw two
     connections before the agent can run. */
  function simpleChain() {
    const starts = state.doc.nodes.filter(node => node.type === "start");
    const ends = state.doc.nodes.filter(node => node.type === "end");
    if (state.doc.nodes.length !== 2 || starts.length !== 1 || ends.length !== 1) return null;
    const edgeIndex = state.doc.edges.findIndex(edge => edge.from === starts[0].id && edge.to === ends[0].id);
    return edgeIndex < 0 ? null : { edgeIndex, start: starts[0], end: ends[0] };
  }

  function addNode(type, position, options = {}) {
    const meta = state.meta.node_types[type];
    const manual = ["start", "end", "note"];
    if (!manual.includes(type)) {
      const selected = selectedNodes();
      if (selected.length === 1 && selected[0].type !== "end") {
        const source = selected[0];
        const outgoing = state.doc.edges.map((edge, index) => ({ edge, index })).filter(item => item.edge.from === source.id);
        if (outgoing.length === 1) {
          const middle = edgeMidpoint(state.doc, outgoing[0].index);
          const node = insertNodeOnEdge(type, outgoing[0].index,
            position || (middle ? { x: middle.x - NODE_WIDTH / 2, y: middle.y - NODE_HEIGHT / 2 } : undefined));
          if (node) fitFlow();
          return node;
        }
        if (outgoing.length === 0) {
          const node = addConnectedNode(type, source.id, type === "condition" ? "true" : "always", position || { x: source.x + NODE_WIDTH + 80, y: source.y });
          if (node) fitFlow();
          return node;
        }
      }
    }
    const chain = simpleChain();
    if (chain && !manual.includes(type)) {
      const node = makeNode(type, position || { x: chain.start.x + NODE_WIDTH + 80, y: chain.start.y });
      if (!node) return null;
      if (!position) {
        node.x = Math.max(0, Math.round((chain.start.x + NODE_WIDTH + 80) / 20) * 20);
        node.y = chain.start.y;
        if (chain.end.x <= node.x + NODE_WIDTH) chain.end.x = node.x + NODE_WIDTH + 80;
      }
      state.edge = null; state.selected = new Set([node.id]);
      mutate(() => {
        state.doc.nodes.push(node);
        state.doc.edges.splice(chain.edgeIndex, 1,
          { from: chain.start.id, to: node.id, when: "always" },
          { from: node.id, to: chain.end.id, when: type === "condition" ? "true" : "always" });
      }, true, `Add ${meta.label}`);
      rememberType(type);
      fitFlow();
      return node;
    }
    const node = makeNode(type, position);
    if (!node) return null;
    state.edge = null; state.selected = new Set([node.id]);
    mutate(() => state.doc.nodes.push(node), true, `Add ${meta.label}`);
    rememberType(type);
    if (options.fit) fitFlow();
    return node;
  }

  function addConnectedNode(type, from, when, position) {
    const meta = state.meta.node_types[type];
    const node = makeNode(type, position);
    if (!node) return null;
    state.edge = null; state.selected = new Set([node.id]);
    mutate(() => { state.doc.nodes.push(node); state.doc.edges.push({ from, to: node.id, when }); }, true, `Add ${meta.label}`);
    rememberType(type);
    return node;
  }

  function insertNodeOnEdge(type, edgeIndex, position) {
    const meta = state.meta.node_types[type];
    const edge = state.doc.edges[edgeIndex];
    const node = makeNode(type, position);
    if (!meta || !edge || !node) return null;
    state.edge = null; state.selected = new Set([node.id]);
    mutate(() => {
      state.doc.nodes.push(node);
      state.doc.edges.splice(edgeIndex, 1,
        { from: edge.from, to: node.id, when: edge.when || "always" },
        { from: node.id, to: edge.to, when: type === "condition" ? "true" : "always" });
    }, true, `Insert ${meta.label}`);
    rememberType(type);
    return node;
  }

  /* A searchable node picker used by the toolbar, double-click, empty-port
     drops and the edge menu.  It keeps the palette out of the way until the
     user actually wants to add something. */
  function openQuickAdd(options = {}) {
    if (!state.doc || !state.meta) return;
    /* Callers pass the pointer position; nodes are positioned by their
       top-left corner, so centre the new card under that point. */
    const point = options.point ? { x: options.point.x - NODE_WIDTH / 2, y: options.point.y - NODE_HEIGHT / 2 } : null;
    const subtitle = options.edge !== undefined ? "Insert on the selected connection"
      : options.from ? `Connect from ${nodeById(options.from)?.label || options.from}${options.when && options.when !== "always" ? ` · ${options.when}` : ""}`
      : "Search by name, category or purpose.";
    const view = Studio.overlay.modal({ key: "agent-quick-add", title: "Add a node", subtitle, wide: true });
    if (view.reused) return;
    const selected = selectedNodes();
    const autoHint = selected.length === 1 ? ` · connects after ${selected[0].label} when possible` : "";
    const search = h("input", { type: "search", class: "agent-quick-search", placeholder: "Find a node type…", "aria-label": "Find a node type", autocomplete: "off", spellcheck: false });
    const list = h("div", { class: "agent-quick-list", role: "listbox", "aria-label": "Node types" });
    let entries = [];
    let index = 0;
    const choose = entry => {
      view.close();
      if (!entry) return;
      if (options.edge !== undefined) insertNodeOnEdge(entry.type, options.edge, point || undefined);
      else if (options.from) addConnectedNode(entry.type, options.from, options.when || "always", point || viewportCenter());
      else addNode(entry.type, point || undefined);
    };
    const paint = () => {
      const hasStart = state.doc.nodes.some(node => node.type === "start");
      const blockStart = hasStart || Boolean(options.from) || options.edge !== undefined;
      const ranked = rankNodeTypes(state.meta.node_types, search.value).filter(entry => entry.type !== "start" || !blockStart);
      const recents = search.value ? [] : state.recentTypes.filter(type => state.meta.node_types[type] && (type !== "start" || !blockStart));
      const recentEntries = recents.map(type => ({ type, meta: state.meta.node_types[type] }));
      const ordered = recentEntries.concat(ranked.filter(entry => !recents.includes(entry.type)));
      entries = ordered;
      index = Math.max(0, Math.min(index, entries.length - 1));
      Studio.ui.render(list);
      if (!entries.length) { list.append(h("div", { class: "agent-quick-empty", text: "No node type matches. Try a purpose such as “http”, “branch” or “model”." })); return; }
      let category = null;
      entries.forEach((entry, position) => {
        if (position === 0 && recentEntries.length) { list.append(h("div", { class: "agent-quick-group", text: "Recent" })); category = "__recent__"; }
        if ((category === "__recent__" && position >= recentEntries.length) || (category !== "__recent__" && entry.meta.category !== category)) {
          category = entry.meta.category;
          list.append(h("div", { class: "agent-quick-group", text: category }));
        }
        const item = h("button", {
          class: "agent-quick-item", type: "button", role: "option", "aria-selected": String(position === index),
          onclick: () => choose(entry),
        }, h("span", { class: "node-glyph", text: entry.meta.icon }),
          h("span", { class: "agent-quick-item-copy" }, h("strong", { text: entry.meta.label }), h("span", { text: entry.meta.description })));
        item.dataset.quickIndex = String(position);
        list.append(item);
      });
      const active = list.querySelector('[aria-selected="true"]');
      if (active) active.scrollIntoView({ block: "nearest" });
    };
    const step = delta => { index = Math.max(0, Math.min(entries.length - 1, index + delta)); paint(); };
    search.addEventListener("input", () => { index = 0; paint(); });
    search.addEventListener("keydown", event => {
      if (event.key === "ArrowDown") { event.preventDefault(); step(1); }
      else if (event.key === "ArrowUp") { event.preventDefault(); step(-1); }
      else if (event.key === "Home") { event.preventDefault(); index = 0; paint(); }
      else if (event.key === "End") { event.preventDefault(); index = entries.length - 1; paint(); }
      else if (event.key === "Enter") { event.preventDefault(); if (entries[index]) choose(entries[index]); }
    });
    view.body.append(search, list, h("p", { class: "help", text: `↑ ↓ choose · Enter add · Esc close.${autoHint}` }));
    paint();
    search.focus();
  }

  function deleteSelection() {
    if (!state.doc) return;
    if (state.edge !== null && state.doc.edges[state.edge]) { const index = state.edge; state.edge = null; mutate(() => state.doc.edges.splice(index, 1), true, "Delete connection"); return; }
    if (!state.selected.size) return;
    const ids = new Set(state.selected);
    state.selected.clear(); state.edge = null;
    mutate(() => { state.doc.nodes = state.doc.nodes.filter(node => !ids.has(node.id)); state.doc.edges = state.doc.edges.filter(edge => !ids.has(edge.from) && !ids.has(edge.to)); }, true, "Delete nodes");
  }

  function duplicateSelection() {
    if (!state.doc || !state.selected.size) return;
    const originals = state.doc.nodes.filter(node => state.selected.has(node.id)); const mapping = {};
    const reserved = new Set(state.doc.nodes.map(node => node.id));
    const clones = originals.map(node => { const clone = copy(node); clone.id = uniqueId(node.type, reserved); mapping[node.id] = clone.id; clone.x += 40; clone.y += 40; clone.label += " copy"; return clone; });
    const edges = state.doc.edges.filter(edge => mapping[edge.from] && mapping[edge.to]).map(edge => ({ ...edge, from: mapping[edge.from], to: mapping[edge.to] }));
    state.edge = null; state.selected = new Set(clones.map(node => node.id));
    mutate(() => { state.doc.nodes.push(...clones); state.doc.edges.push(...edges); }, true, "Duplicate nodes");
  }

  function copySelection() {
    if (!state.doc) return;
    const nodes = state.doc.nodes.filter(node => state.selected.has(node.id));
    if (!nodes.length) return;
    const ids = new Set(state.selected);
    state.clipboard = { nodes: copy(nodes), edges: copy(state.doc.edges.filter(edge => ids.has(edge.from) && ids.has(edge.to))) };
    Studio.toast(`Copied ${nodes.length} node${nodes.length === 1 ? "" : "s"}.`, "success");
  }

  function pasteSelection(position) {
    if (!state.doc || !state.clipboard) return;
    const mapping = {};
    const reserved = new Set(state.doc.nodes.map(node => node.id));
    const minX = Math.min(...state.clipboard.nodes.map(node => node.x));
    const minY = Math.min(...state.clipboard.nodes.map(node => node.y));
    const offset = position ? { x: position.x - minX, y: position.y - minY } : { x: 40, y: 40 };
    const nodes = state.clipboard.nodes.map(original => {
      const node = copy(original); node.id = uniqueId(node.type, reserved); mapping[original.id] = node.id;
      node.x = Math.max(0, Math.round((original.x + offset.x) / 20) * 20);
      node.y = Math.max(0, Math.round((original.y + offset.y) / 20) * 20);
      return node;
    });
    const edges = state.clipboard.edges.map(edge => ({ ...edge, from: mapping[edge.from], to: mapping[edge.to] }));
    state.edge = null; state.selected = new Set(nodes.map(node => node.id));
    mutate(() => { state.doc.nodes.push(...nodes); state.doc.edges.push(...edges); }, true, "Paste nodes");
  }

  function selectedNodes() {
    return state.doc ? state.doc.nodes.filter(node => state.selected.has(node.id)) : [];
  }

  function selectAllNodes() {
    if (!state.doc) return;
    const previousEdge = state.edge;
    state.edge = null;
    state.selected = new Set(state.doc.nodes.map(node => node.id));
    if (previousEdge !== null) paintEdgeSelection(previousEdge, null);
    paintSelection();
    renderInspector(); renderSelectionToolbar(); renderStatus();
  }

  function applyPositions(positions) {
    for (const position of positions) {
      const node = nodeById(position.id);
      if (node) { node.x = position.x; node.y = position.y; }
    }
  }

  function alignSelection(mode) {
    const positions = alignPositions(selectedNodes(), mode);
    if (!positions.length) return;
    mutate(() => applyPositions(positions), true, "Align nodes");
  }

  function distributeSelection(axis) {
    const positions = distributePositions(selectedNodes(), axis);
    if (!positions.length) return;
    mutate(() => applyPositions(positions), true, "Distribute nodes");
  }

  function nudgeSelection(dx, dy) {
    if (!state.selected.size) return;
    const last = state.undo[state.undo.length - 1];
    if (!state.nudgeAt || Date.now() - state.nudgeAt > 900 || !last || last.label !== "Nudge nodes") pushHistory("Nudge nodes");
    for (const node of selectedNodes()) {
      node.x = Math.max(0, node.x + dx);
      node.y = Math.max(0, node.y + dy);
    }
    state.nudgeAt = Date.now();
    markDirty();
    paintNodePositions(state.selected);
    updateEdgesTouching(state.selected);
    scheduleMinimap();
    scheduleValidation();
  }

  function focusNode(id) {
    const node = nodeById(id);
    if (!node) return;
    const box = $("agents-canvas").getBoundingClientRect();
    if (!box.width) return;
    const zoom = Math.max(state.zoom, 0.55);
    state.zoom = zoom;
    state.pan = {
      x: box.width / 2 - (node.x + NODE_WIDTH / 2) * zoom,
      y: box.height / 2 - (node.y + NODE_HEIGHT / 2) * zoom,
    };
    applyTransform(); renderStatus();
  }

  function fitNodes(nodes) {
    if (!nodes.length) return;
    const box = $("agents-canvas").getBoundingClientRect();
    if (!box.width || !box.height) return;
    const minX = Math.min(...nodes.map(node => node.x)); const minY = Math.min(...nodes.map(node => node.y));
    const maxX = Math.max(...nodes.map(node => node.x + NODE_WIDTH)); const maxY = Math.max(...nodes.map(node => node.y + NODE_HEIGHT));
    const padding = 80;
    state.zoom = Math.max(0.25, Math.min(1.3, Math.min(
      (box.width - padding) / Math.max(1, maxX - minX),
      (box.height - padding) / Math.max(1, maxY - minY))));
    state.pan = {
      x: (box.width - (maxX - minX) * state.zoom) / 2 - minX * state.zoom,
      y: (box.height - (maxY - minY) * state.zoom) / 2 - minY * state.zoom,
    };
    applyTransform(); renderStatus();
  }

  function fitSelection() {
    const nodes = selectedNodes();
    fitNodes(nodes.length ? nodes : (state.doc ? state.doc.nodes : []));
  }

  async function renameNode(node) {
    const value = await Studio.prompt({ title: "Rename node", message: "The display label is used in the canvas and run events.", label: "Display label", value: node.label, confirmLabel: "Rename" });
    if (value === null) return;
    const label = value.trim();
    if (!label || label === node.label) return;
    mutate(() => { node.label = label; }, true, "Rename node");
  }

  function startInlineRename(node, element) {
    const title = element.querySelector(".agent-node-title");
    if (!title || element.querySelector(".agent-node-rename")) return;
    const input = h("input", { class: "agent-node-rename", value: node.label, "aria-label": `Rename ${node.label}` });
    let done = false;
    const finish = commit => {
      if (done) return;
      done = true;
      state.renaming = null;
      const label = input.value.trim();
      if (commit && label && label !== node.label) mutate(() => { node.label = label; }, true, "Rename node");
      else renderNodes();
    };
    input.addEventListener("pointerdown", event => event.stopPropagation());
    input.addEventListener("click", event => event.stopPropagation());
    input.addEventListener("keydown", event => {
      event.stopPropagation();
      if (event.key === "Enter") { event.preventDefault(); finish(true); }
      else if (event.key === "Escape") { event.preventDefault(); finish(false); }
    });
    input.addEventListener("blur", () => finish(true));
    title.replaceWith(input);
    state.renaming = node.id;
    input.focus(); input.select();
  }

  function connectDialog(node) {
    const candidates = state.doc.nodes.filter(target => target.id !== node.id && target.type !== "start");
    if (!candidates.length) { Studio.toast("Add another node first; Start is the only targetless node.", "info"); return; }
    const target = h("select", {"aria-label": "Target node"}, candidates.map(item => h("option", { value: item.id, text: `${item.label} · ${item.type}` })));
    const when = h("select", { "aria-label": "Branch" }, [
      h("option", { value: "always", text: "Always" }), h("option", { value: "true", text: "True branch" }), h("option", { value: "false", text: "False branch" }),
    ]);
    if (node.type === "condition") when.value = "true";
    const view = Studio.overlay.modal({ key: `connect:${node.id}`, title: `Connect ${node.label}`, subtitle: "Create a connection without dragging from a port.", narrow: true });
    if (view.reused) return;
    view.body.append(Studio.ui.field("Target node", target), Studio.ui.field("Follow this branch", when, node.type === "condition" ? "Conditions normally use their true and false outputs." : "Other node types always follow the connection."));
    view.footer.append(Studio.ui.button("Connect", { kind: "primary", onclick: () => { view.close(); createEdge(node.id, target.value, node.type === "condition" ? when.value : "always"); } }));
  }

  async function disconnectNode(node) {
    const linked = state.doc.edges.filter(edge => edge.from === node.id || edge.to === node.id);
    if (!linked.length) { Studio.toast("This node has no connections.", "info"); return; }
    if (!await Studio.confirm({ title: "Disconnect node?", message: `Remove ${linked.length} connection${linked.length === 1 ? "" : "s"} to and from ${node.label}. The node itself stays in the flow.`, confirmLabel: "Disconnect", danger: true })) return;
    mutate(() => { state.doc.edges = state.doc.edges.filter(edge => edge.from !== node.id && edge.to !== node.id); }, true, "Disconnect node");
  }

  function edgeMenu(event, index) {
    const edge = state.doc.edges[index];
    if (!edge) return;
    Studio.overlay.menu({ x: event.clientX, y: event.clientY }, [
      { label: "Inspect connection", icon: "link", onSelect: () => selectEdge(index) },
      { label: "Insert a node", icon: "plus", onSelect: () => openQuickAdd({ edge: index, point: edgeMidpoint(state.doc, index) }) },
      "separator",
      { label: "Delete connection", icon: "trash", danger: true, onSelect: () => { state.edge = index; state.selected.clear(); deleteSelection(); } },
    ], { label: "Connection actions" });
  }

  function nodeMenu(event, node) {
    const linked = state.doc.edges.filter(edge => edge.from === node.id || edge.to === node.id).length;
    const position = { x: node.x + 40, y: node.y + 40 };
    Studio.overlay.menu({ x: event.clientX, y: event.clientY }, [
      { label: "Rename", icon: "edit", hint: "F2", onSelect: () => renameNode(node) },
      { label: "Connect to…", icon: "link", onSelect: () => connectDialog(node) },
      { label: "Disconnect all", icon: "x", disabled: !linked, onSelect: () => disconnectNode(node) },
      "separator",
      { label: "Duplicate", icon: "copy", hint: "Ctrl+D", onSelect: duplicateSelection },
      { label: "Copy", icon: "copy", hint: "Ctrl+C", onSelect: copySelection },
      { label: "Paste here", icon: "plus", disabled: !state.clipboard, onSelect: () => pasteSelection(position) },
      { label: state.breakpoints.has(node.id) ? "Remove breakpoint" : "Add breakpoint", icon: "bug", onSelect: () => toggleBreakpoint(node.id) },
      "separator", { label: "Delete", icon: "trash", danger: true, hint: "Del", onSelect: deleteSelection },
    ], { label: `${node.label} actions` });
  }

  /* Pointer and wheel events can outpace the compositor.  Coalesce pan/zoom
     into one paint per frame so a fast trackpad does not queue dozens of
     style writes and minimap redraws. */
  function applyTransform() {
    frame("transform", paintTransform);
  }

  function paintTransform() {
    Studio.setStyle($("agents-world"), { transform: `translate(${state.pan.x}px, ${state.pan.y}px) scale(${state.zoom})` });
    $("agents-zoom").textContent = `${Math.round(state.zoom * 100)}%`;
    scheduleMinimap();
  }

  function setZoom(value, center) {
    const box = $("agents-canvas").getBoundingClientRect();
    const point = center || { x: box.left + box.width / 2, y: box.top + box.height / 2 };
    const nextZoom = Math.max(0.25, Math.min(2, value));
    state.pan = zoomedPan(state.pan, state.zoom, nextZoom, point, { left: box.left, top: box.top });
    state.zoom = nextZoom;
    applyTransform(); frame("status", renderStatus);
  }

  function fitFlow() {
    if (!state.doc || !state.doc.nodes.length) return;
    fitNodes(state.doc.nodes);
  }

  function autoLayout() {
    if (!state.doc || !state.meta) return;
    const outgoing = {}; const incoming = {};
    for (const edge of state.doc.edges) { (outgoing[edge.from] ||= []).push(edge.to); incoming[edge.to] = (incoming[edge.to] || 0) + 1; }
    const start = state.doc.nodes.find(node => node.type === "start") || state.doc.nodes[0];
    const depth = {}; const queue = start ? [[start.id, 0]] : [];
    while (queue.length) { const [id, level] = queue.shift(); if (depth[id] !== undefined) continue; depth[id] = level; for (const child of outgoing[id] || []) queue.push([child, level + 1]); }
    const rows = {};
    mutate(() => {
      for (const node of state.doc.nodes) { const level = depth[node.id] === undefined ? Math.max(0, ...Object.values(depth)) + 1 : depth[node.id]; const row = rows[level] || 0; rows[level] = row + 1; node.x = 60 + level * 250; node.y = 60 + row * 140; }
    }, true, "Auto layout");
    fitFlow();
  }

  function renderHint() {
    const hint = $("agents-canvas-hint");
    const sparse = state.doc.nodes.length <= 2;
    hint.classList.toggle("hidden", !sparse);
    if (!sparse) return;
    hint.textContent = state.doc.nodes.length <= 1
      ? "Double-click empty space to add a node, or drag one from the palette."
      : "Drag from a port to connect · double-click a node title to rename · double-click empty space to add a node.";
  }

  function renderStatus() {
    if (!state.doc) return;
    const nodeCount = state.doc.nodes.length;
    const edgeCount = state.doc.edges.length;
    const errors = state.issues.filter(issue => issue.level === "error").length;
    const warnings = state.issues.length - errors;
    const persisted = state.agents.some(agent => agent.id === state.doc.id);
    const root = Studio.ui.render($("agents-status"));
    root.append(
      h("span", { text: `${nodeCount} node${nodeCount === 1 ? "" : "s"}` }),
      h("span", { text: `${edgeCount} connection${edgeCount === 1 ? "" : "s"}` }),
      state.selected.size ? h("span", { text: `${state.selected.size} selected` }) : null,
      state.run ? h("span", { class: state.run.error ? "agent-status-problems" : "", text: `Run ${state.run.status}${state.run.current_node ? ` · ${state.run.current_node}` : ""}${state.run.steps ? ` · ${state.run.steps} steps` : ""}` }) : null,
      h("span", { text: `${Math.round(state.zoom * 100)}% zoom` }),
      h("span", { class: state.dirty ? "agent-status-dirty" : "", text: state.dirty ? "Unsaved changes" : persisted ? "Saved" : "Draft (not saved)" }),
      state.issues.length ? Studio.ui.button(`${errors ? `${errors} error${errors === 1 ? "" : "s"}` : ""}${errors && warnings ? ", " : ""}${warnings ? `${warnings} warning${warnings === 1 ? "" : "s"}` : ""}`, {
        small: true, kind: "ghost", title: "Show validation problems", onclick: () => showDetails("problems"),
      }) : h("span", { text: "No problems" }),
      h("span", { class: "grow" }),
      Studio.ui.button("Shortcuts", { small: true, kind: "ghost", title: "Show keyboard and mouse shortcuts", onclick: showShortcuts }));
  }

  function renderSelectionToolbar() {
    const root = Studio.ui.render($("agents-selection-toolbar"));
    const nodes = state.doc ? selectedNodes() : [];
    root.classList.toggle("hidden", nodes.length < 2);
    if (nodes.length < 2) return;
    root.append(
      h("span", { class: "agent-selection-count", text: `${nodes.length} selected` }),
      Studio.ui.button("Left", { small: true, title: "Align left edges", onclick: () => alignSelection("left") }),
      Studio.ui.button("H center", { small: true, title: "Align horizontal centres", onclick: () => alignSelection("hcenter") }),
      Studio.ui.button("Right", { small: true, title: "Align right edges", onclick: () => alignSelection("right") }),
      Studio.ui.button("Top", { small: true, title: "Align top edges", onclick: () => alignSelection("top") }),
      Studio.ui.button("V center", { small: true, title: "Align vertical centres", onclick: () => alignSelection("vcenter") }),
      Studio.ui.button("Bottom", { small: true, title: "Align bottom edges", onclick: () => alignSelection("bottom") }),
      Studio.ui.button("Spread H", { small: true, disabled: nodes.length < 3, title: "Distribute horizontally", onclick: () => distributeSelection("x") }),
      Studio.ui.button("Spread V", { small: true, disabled: nodes.length < 3, title: "Distribute vertically", onclick: () => distributeSelection("y") }),
      Studio.ui.button("Fit", { small: true, title: "Fit the selection", onclick: fitSelection }),
      Studio.ui.button("Duplicate", { small: true, title: "Duplicate selection (Ctrl+D)", onclick: duplicateSelection }),
      Studio.ui.button("Delete", { small: true, kind: "danger", title: "Delete selection (Del)", onclick: deleteSelection }));
  }

  function buildFind() {
    const root = $("agents-find");
    const input = h("input", { type: "search", placeholder: "Find nodes…", "aria-label": "Find nodes by label, type or id", autocomplete: "off", spellcheck: false });
    const count = h("span", { class: "agent-find-count", "aria-live": "polite", text: "" });
    const previous = Studio.ui.iconButton("chevron-left", "Previous match", () => stepFind(-1), { small: true });
    const next = Studio.ui.iconButton("chevron-right", "Next match", () => stepFind(1), { small: true });
    const close = Studio.ui.iconButton("x", "Close find", closeFind, { small: true });
    input.addEventListener("input", () => { state.find.query = input.value; refreshFind(); });
    input.addEventListener("keydown", event => {
      event.stopPropagation();
      if (event.key === "Enter") { event.preventDefault(); stepFind(event.shiftKey ? -1 : 1); }
      else if (event.key === "Escape") { event.preventDefault(); closeFind(); $("agents-canvas").focus(); }
    });
    root.append(input, count, previous, next, close);
    state.find.ui = { root, input, count };
  }

  function openFind() {
    if (!state.doc || !state.meta) return;
    state.find.open = true;
    state.find.ui.root.classList.remove("hidden");
    $("agents-find-toggle").setAttribute("aria-pressed", "true");
    refreshFind();
    state.find.ui.input.focus();
    state.find.ui.input.select();
  }

  function closeFind() {
    state.find.open = false;
    state.find.query = "";
    state.find.matches = [];
    state.find.index = -1;
    if (state.find.ui) {
      state.find.ui.root.classList.add("hidden");
      state.find.ui.input.value = "";
      state.find.ui.count.textContent = "";
    }
    if ($("agents-find-toggle")) $("agents-find-toggle").setAttribute("aria-pressed", "false");
    if (state.doc && state.meta) renderNodes();
  }

  function refreshFind() {
    if (!state.find.open || !state.doc || !state.meta) return;
    const previous = state.find.index >= 0 ? state.find.matches[state.find.index] : null;
    state.find.matches = findNodes(state.doc.nodes, state.find.query);
    if (!state.find.matches.length) state.find.index = -1;
    else if (previous && state.find.matches.includes(previous)) state.find.index = state.find.matches.indexOf(previous);
    else state.find.index = 0;
    updateFindCount();
    renderNodes();
  }

  function updateFindCount() {
    if (!state.find.ui) return;
    state.find.ui.count.textContent = !state.find.query.trim() ? ""
      : state.find.matches.length ? `${state.find.index + 1}/${state.find.matches.length}` : "0/0";
  }

  function stepFind(delta) {
    if (!state.find.matches.length) return;
    state.find.index = (state.find.index + delta + state.find.matches.length) % state.find.matches.length;
    updateFindCount();
    renderNodes();
    focusNode(state.find.matches[state.find.index]);
  }

  function updatePanMode() {
    const canvas = $("agents-canvas");
    canvas.classList.toggle("pan-mode", state.panMode);
    canvas.classList.toggle("pan-ready", state.spaceHeld && !state.panMode);
    $("agents-pan-mode").setAttribute("aria-pressed", String(state.panMode));
    $("agents-pan-mode").title = state.panMode ? "Exit pan mode (H)" : "Pan mode (H) · hold Space to pan temporarily";
  }

  function showShortcuts() {
    const groups = [
      ["Files and history", [["Ctrl+S", "Save agent"], ["Ctrl+Enter", "Run the agent"], ["Ctrl+Z", "Undo"], ["Ctrl+Shift+Z", "Redo"], ["Ctrl+A", "Select all nodes"]]],
      ["Editing", [["N", "Add a node"], ["Ctrl+F", "Find nodes"], ["F2", "Rename selected node"], ["Ctrl+C / Ctrl+V", "Copy / paste nodes"], ["Ctrl+D", "Duplicate selection"], ["Delete", "Delete selection or connection"], ["Arrows", "Nudge by 20px (Shift: 1px)"], ["Alt + drag", "Move without snapping to guides"], ["Add while selected", "Wire the new step into the chain automatically"]]],
      ["View", [["Ctrl+B", "Toggle node palette"], ["Ctrl+Shift+B", "Toggle inspector"], ["Ctrl+J", "Toggle details panel"], ["0", "Reset zoom to 100%"], ["1", "Fit the flow"], ["H", "Toggle pan mode"], ["Space + drag", "Pan temporarily"], ["Ctrl + wheel", "Zoom at the pointer"]]],
      ["Mouse", [["Double-click canvas", "Add a node at that point"], ["Double-click node title", "Rename inline"], ["Right-click", "Node or connection actions"], ["Drag a port to a node", "Connect to that step"], ["Drag a port to empty space", "Add and connect the next step"], ["Drag onto a connection", "Insert the node between"], ["Double-click a connection", "Insert a node there"], ["Shift / Ctrl + drag", "Add to the selection"]]],
    ];
    const view = Studio.overlay.modal({ key: "agent-shortcuts", title: "Designer shortcuts", subtitle: "Keyboard and pointer shortcuts for the agent flow canvas.", wide: true });
    if (view.reused) return;
    view.body.append(h("div", { class: "grid-2" }, groups.map(([title, rows]) => h("section", { class: "agent-shortcut-group" },
      h("h3", { text: title }),
      rows.map(([keys, description]) => h("div", { class: "agent-shortcut-row" }, h("span", { text: description }), h("kbd", { text: keys })))))));
  }

  /* getComputedStyle forces style resolution; the theme changes rarely, so
     resolve the two colours once per theme and reuse them for every frame. */
  function minimapColours() {
    if (!minimapPalette) {
      const styles = getComputedStyle(document.documentElement);
      minimapPalette = {
        fill: styles.getPropertyValue("--border-bright").trim() || "currentColor",
        stroke: styles.getPropertyValue("--accent").trim() || "currentColor",
      };
    }
    return minimapPalette;
  }

  function drawMinimapNow() {
    const canvas = $("agents-minimap");
    if (!state.doc || !canvas.getContext) return;
    const width = canvas.clientWidth; const height = canvas.clientHeight;
    if (!width || !height) return;
    const context = canvas.getContext("2d");
    const ratio = window.devicePixelRatio || 1;
    if (canvas.width !== width * ratio || canvas.height !== height * ratio) { canvas.width = width * ratio; canvas.height = height * ratio; }
    context.setTransform(ratio, 0, 0, ratio, 0, 0); context.clearRect(0, 0, width, height);
    let maxX = 1000; let maxY = 700;
    for (const node of state.doc.nodes) { maxX = Math.max(maxX, node.x + NODE_WIDTH + 80); maxY = Math.max(maxY, node.y + NODE_HEIGHT + 80); }
    const scale = Math.min(width / maxX, height / maxY);
    const colours = minimapColours();
    context.fillStyle = colours.fill;
    for (const node of state.doc.nodes) context.fillRect(node.x * scale, node.y * scale, Math.max(2, NODE_WIDTH * scale), Math.max(2, NODE_HEIGHT * scale));
    const view = $("agents-canvas");
    context.strokeStyle = colours.stroke; context.lineWidth = 1;
    context.strokeRect(-state.pan.x / state.zoom * scale, -state.pan.y / state.zoom * scale, view.clientWidth / state.zoom * scale, view.clientHeight / state.zoom * scale);
  }

  function insertAtCursor(input, text, selectText) {
    const start = input.selectionStart === undefined ? input.value.length : input.selectionStart;
    const end = input.selectionEnd === undefined ? start : input.selectionEnd;
    input.setRangeText(text, start, end, "end");
    if (selectText) {
      const offset = text.indexOf(selectText);
      if (offset >= 0) input.setSelectionRange(start + offset, start + offset + selectText.length);
    }
    input.focus();
    input.dispatchEvent(new Event("change", { bubbles: true }));
  }

  function conditionDialog(input, node) {
    const sources = [
      ["last", "Previous output (last)"],
      ["last.text", "Previous text (last.text)"],
      ["input", "Run input"],
      ["input.message", "Run input message"],
    ];
    for (const name of Object.keys(state.doc.vars || {}).sort()) sources.push([`vars.${name}`, `Variable vars.${name}`]);
    for (const item of state.doc.nodes) {
      if (!node || item.id !== node.id) sources.push([`nodes.${item.id}`, `Output of ${item.label}`]);
    }
    const source = h("select", { "aria-label": "Value to test" }, sources.map(([value, label]) => h("option", { value, text: label })));
    const operator = h("select", { "aria-label": "Comparison" }, [
      ["contains", "contains"], ["not_contains", "does not contain"],
      ["equals", "equals"], ["not_equals", "does not equal"],
      ["starts", "starts with"], ["ends", "ends with"],
      ["empty", "is empty"], ["not_empty", "is not empty"],
      ["gt", "is greater than"], ["lt", "is less than"],
    ].map(([value, label]) => h("option", { value, text: label })));
    const value = h("input", { placeholder: "urgent", spellcheck: false, "aria-label": "Comparison value" });
    const valueField = Studio.ui.field("Value", value, "Quoted automatically unless it looks like a number or true/false.");
    const preview = h("code", { class: "wrap-anywhere" });
    const literal = raw => {
      const text = String(raw || "").trim();
      if (!text) return "''";
      if (/^-?\d+(?:\.\d+)?$/.test(text)) return text;
      if (/^(true|false|None)$/.test(text)) return text;
      return JSON.stringify(text);
    };
    const expression = () => {
      const left = source.value; const right = literal(value.value);
      switch (operator.value) {
        case "contains": return `contains(${left}, ${right})`;
        case "not_contains": return `not contains(${left}, ${right})`;
        case "equals": return `${left} == ${right}`;
        case "not_equals": return `${left} != ${right}`;
        case "starts": return `startswith(${left}, ${right})`;
        case "ends": return `endswith(${left}, ${right})`;
        case "empty": return `not ${left}`;
        case "not_empty": return left;
        case "gt": return `${left} > ${right}`;
        case "lt": return `${left} < ${right}`;
        default: return "true";
      }
    };
    const update = () => {
      valueField.classList.toggle("hidden", operator.value === "empty" || operator.value === "not_empty");
      preview.textContent = expression();
    };
    const view = Studio.overlay.modal({ key: "condition-helper", title: "Build a condition", subtitle: "Choose a value, a comparison and insert a safe expression.", wide: true });
    if (view.reused) return;
    for (const control of [source, operator, value]) { control.addEventListener("input", update); control.addEventListener("change", update); }
    update();
    view.body.append(h("div", { class: "grid-2" }, Studio.ui.field("Value to test", source), Studio.ui.field("Comparison", operator)), valueField,
      h("div", { class: "callout" }, h("div", { class: "grow" }, h("strong", { text: "Expression" }), h("pre", {}, preview))));
    view.footer.append(Studio.ui.button("Use expression", { kind: "primary", onclick: () => {
      input.value = expression();
      view.close();
      input.focus();
      input.dispatchEvent(new Event("change", { bubbles: true }));
    } }));
  }

  function variableMenu(anchor, input, node, expression) {
    const items = [{ heading: "Run context" }];
    const add = (label, token) => items.push({ label, onSelect: () => insertAtCursor(input, expression ? token : `{{ ${token} }}`) });
    add("input", "input");
    add("last", "last");
    if (node && node.type === "foreach") { add("item (current loop entry)", "item"); add("index (current loop position)", "index"); }
    const variables = new Set([...Object.keys(state.doc.vars || {}), ...Object.keys((state.run && state.run.vars) || {})]);
    items.push({ heading: "Variables" });
    if (!variables.size) items.push({ label: "vars (none defined yet)", disabled: true });
    for (const name of [...variables].sort()) add(`vars.${name}`, `vars.${name}`);
    items.push({ heading: "Other node outputs" });
    const others = state.doc.nodes.filter(item => !node || item.id !== node.id);
    if (!others.length) items.push({ label: "No other nodes", disabled: true });
    for (const item of others) add(`${item.label} · nodes.${item.id}`, `nodes.${item.id}`);
    Studio.overlay.menu(anchor, items, { label: "Insert variable" });
  }

  function fieldVisible(node, field) {
    if (node.type === "human" && field.key === "choices") return (node.data.kind || "approve") === "choice";
    if (node.type === "memory" && field.key === "value") return (node.data.action || "get") === "set";
    return true;
  }

  function fieldControl(field, value, onChange, context = {}) {
    const type = field.type;
    if (type === "abilities") {
      const selected = new Set(value || []);
      return h("div", { class: "stack tight" }, state.doc.abilities.map(ability => h("label", { class: "check" }, h("input", { type: "checkbox", checked: selected.has(ability.id), onchange: event => {
        if (event.currentTarget.checked) selected.add(ability.id); else selected.delete(ability.id); onChange(Array.from(selected));
      } }), `${ability.label || ability.id} (${ability.type})`)), !state.doc.abilities.length ? h("span", { class: "help", text: "Grant abilities first using the + button in the node palette." }) : null);
    }
    let input;
    let extra = null;
    if (type === "boolean") input = Studio.form.control({ type: "checkbox", value: Boolean(value) });
    else if (type === "select") input = Studio.form.control({ type: "select", options: field.options || [], value });
    else if (type === "model") input = h("select", {}, h("option", { value: "", text: "Studio default model" }), Studio.models.chat().map(entry => h("option", { value: entry.id, text: entry.name, selected: entry.id === value, disabled: !entry.available })));
    else if (type === "ability") {
      input = h("select", {}, h("option", { value: "", text: "Choose an ability" }), state.doc.abilities.map(ability => h("option", { value: ability.id, text: `${ability.label || ability.id} (${ability.type})`, selected: ability.id === value })));
      extra = Studio.ui.button("Add ability", { small: true, onclick: () => abilityDialog() });
    } else if (type === "agent") input = h("select", {}, h("option", { value: "", text: "Choose an agent" }), state.agents.map(agent => h("option", { value: agent.id, text: agent.name, selected: agent.id === value })));
    else if (["template", "expression", "json"].includes(type)) input = Studio.form.control({ type: "textarea", rows: field.rows || 3, value: type === "json" ? JSON.stringify(value === undefined ? {} : value, null, 2) : value || "", spellcheck: false });
    else if (type === "number") input = Studio.form.control({ type: "number", value: value === undefined ? "" : value, min: field.min, max: field.max, step: field.step || "any" });
    else input = Studio.form.control({ type: "text", value: value || "" });

    let error = null;
    if (type === "json") {
      error = h("span", { class: "field-error hidden" });
      const validate = () => {
        try { JSON.parse(input.value || "{}"); }
        catch (problem) { error.textContent = `Invalid JSON: ${problem.message}`; error.classList.remove("hidden"); input.setAttribute("aria-invalid", "true"); return false; }
        error.classList.add("hidden"); input.removeAttribute("aria-invalid"); return true;
      };
      input.addEventListener("input", validate);
      input.addEventListener("change", () => {
        if (!validate()) return;
        onChange(JSON.parse(input.value || "{}"));
      });
    } else {
      input.addEventListener("change", () => onChange(Studio.form.value(input)));
    }

    if (type === "template" || type === "expression" || type === "json") {
      const tools = h("div", { class: "agent-field-tools" });
      if (type !== "json") {
        tools.append(Studio.ui.button("Insert variable", { small: true, title: "Insert a run reference at the cursor", onclick: event => variableMenu(event.currentTarget, input, context.node, type === "expression") }));
        if (type === "expression") tools.append(Studio.ui.button("Build condition", { small: true, title: "Create an expression from simple choices", onclick: () => conditionDialog(input, context.node) }));
      } else {
        tools.append(Studio.ui.button("Format", { small: true, title: "Pretty-print this JSON", onclick: () => { try { input.value = JSON.stringify(JSON.parse(input.value || "{}"), null, 2); input.dispatchEvent(new Event("change", { bubbles: true })); } catch (problem) { Studio.toast("Fix the JSON before formatting it.", "error"); } } }));
      }
      if (error) tools.append(error);
      return h("div", { class: "field" }, h("span", { text: field.label }), h("div", { class: "stack tight" }, input, tools),
        field.help ? h("span", { class: "help", text: field.help }) : null);
    }
    if (extra) return h("div", { class: "field" }, h("span", { text: field.label }), h("div", { class: "input-action" }, input, extra),
      field.help ? h("span", { class: "help", text: field.help }) : null);
    return Studio.ui.field(field.label, input, field.help);
  }

  function agentSection(key, title, children, badge, openByDefault = true) {
    const open = state.sectionState.has(key) ? state.sectionState.get(key) : openByDefault;
    const details = h("details", { class: "agent-section", open });
    details.append(h("summary", {}, h("h3", { text: title }), badge || null),
      h("div", { class: "agent-section-body" }, children));
    details.addEventListener("toggle", () => {
      if (!details.isConnected) return;
      state.sectionState.set(key, details.open);
    });
    return details;
  }

  function connectionRow(edge, index, direction) {
    const other = nodeById(direction === "out" ? edge.to : edge.from);
    const label = other ? other.label : "(missing node)";
    const detail = [edge.label, edge.when !== "always" ? edge.when : ""].filter(Boolean).join(" · ");
    return h("button", { class: "agent-connection-row", type: "button", title: `Connection ${index + 1}`, onclick: () => selectEdge(index) },
      h("span", { class: "node-glyph", text: direction === "out" ? "→" : "←" }),
      h("span", { class: "grow", text: label }),
      detail ? h("span", { class: "muted tiny", text: detail }) : null);
  }

  function renderInspector() {
    const root = $("agents-inspector");
    inspectorPosition = null;
    const signature = state.edge !== null ? `edge:${state.edge}` : `nodes:${[...state.selected].sort().join(",")}`;
    const scrollTop = state.inspectorSelection === signature ? root.scrollTop : 0;
    state.inspectorSelection = signature;
    Studio.ui.render(root);
    try {
      if (!state.doc || !state.meta) return;
    if (state.edge !== null && state.doc.edges[state.edge]) {
      const edge = state.doc.edges[state.edge]; const index = state.edge;
      const from = nodeById(edge.from); const to = nodeById(edge.to);
      root.append(agentSection("connection", "Connection", [
        Studio.ui.kv([["From", from ? from.label : edge.from], ["To", to ? to.label : edge.to]]),
        fieldControl({ key: "when", label: "Branch", type: "select", options: [["always", "Always"], ["true", "True"], ["false", "False"]] }, edge.when || "always", value => mutate(() => { edge.when = value; }, true, "Edit connection")),
        fieldControl({ key: "label", label: "Label (optional)", type: "text" }, edge.label || "", value => mutate(() => { if (value) edge.label = value; else delete edge.label; }, true, "Edit connection")),
        h("div", { class: "row wrap" }, from ? Studio.ui.button("Focus source", { small: true, onclick: () => { selectNode(from.id); focusNode(from.id); } }) : null,
          to ? Studio.ui.button("Focus target", { small: true, onclick: () => { selectNode(to.id); focusNode(to.id); } }) : null),
        Studio.ui.button("Insert a node here", { small: true, onclick: () => openQuickAdd({ edge: index, point: edgeMidpoint(state.doc, index) }) }),
      ], Studio.ui.badge(edge.when || "always")));
      root.append(Studio.ui.button("Delete connection", { kind: "danger", onclick: deleteSelection }));
      return;
    }
    const nodes = selectedNodes();
    if (nodes.length > 1) {
      root.append(agentSection("multi", `${nodes.length} nodes selected`, [
        h("p", { class: "help", text: "Drag any selected node to move the group. Arrow keys nudge by 20px; hold Shift for 1px. Shift- or Ctrl-drag on the canvas adds to the selection." }),
        h("div", { class: "row wrap" }, Studio.ui.button("Align left", { small: true, onclick: () => alignSelection("left") }), Studio.ui.button("H centre", { small: true, onclick: () => alignSelection("hcenter") }), Studio.ui.button("Align right", { small: true, onclick: () => alignSelection("right") })),
        h("div", { class: "row wrap" }, Studio.ui.button("Align top", { small: true, onclick: () => alignSelection("top") }), Studio.ui.button("V centre", { small: true, onclick: () => alignSelection("vcenter") }), Studio.ui.button("Align bottom", { small: true, onclick: () => alignSelection("bottom") })),
        h("div", { class: "row wrap" }, Studio.ui.button("Spread horizontally", { small: true, disabled: nodes.length < 3, onclick: () => distributeSelection("x") }), Studio.ui.button("Spread vertically", { small: true, disabled: nodes.length < 3, onclick: () => distributeSelection("y") }), Studio.ui.button("Fit selection", { small: true, onclick: fitSelection })),
        h("div", { class: "row wrap" }, Studio.ui.button("Copy", { small: true, onclick: copySelection }), Studio.ui.button("Duplicate", { small: true, onclick: duplicateSelection }), Studio.ui.button("Delete", { small: true, kind: "danger", onclick: deleteSelection })),
      ], Studio.ui.badge(`${nodes.length} selected`, "accent")));
      return;
    }
    if (nodes.length === 1) {
      const node = nodes[0]; const meta = state.meta.node_types[node.type] || { label: node.type, description: "Unknown node type.", fields: [] };
      const positionValue = h("span", { text: `${Math.round(node.x)}, ${Math.round(node.y)}` });
      inspectorPosition = positionValue;
      const outgoing = state.doc.edges.map((edge, index) => ({ edge, index })).filter(item => item.edge.from === node.id);
      const incoming = state.doc.edges.map((edge, index) => ({ edge, index })).filter(item => item.edge.to === node.id);
      const rerenderOnChange = new Set(["kind", "action"]);
      const renderField = field => fieldControl(field, node.data[field.key], value => {
        mutate(() => { node.data[field.key] = value; }, rerenderOnChange.has(field.key), `Edit ${field.label}`);
      }, { node });
      const available = meta.fields.filter(field => fieldVisible(node, field));
      const primaryKeys = PRIMARY_FIELDS[node.type] || meta.fields.slice(0, 3).map(field => field.key);
      const primary = available.filter(field => primaryKeys.includes(field.key));
      const advanced = available.filter(field => !primaryKeys.includes(field.key));
      root.append(agentSection("node-header", "Step", [
        h("div", { class: "row between" }, h("strong", { text: node.label }), h("div", { class: "row" }, state.breakpoints.has(node.id) ? Studio.ui.badge("breakpoint", "danger") : null)),
        h("p", { class: "help", text: meta.description }),
        fieldControl({ label: "Display label", type: "text" }, node.label, value => mutate(() => { node.label = value || meta.label; }, false, "Rename node"), { node }),
        h("div", { class: "row wrap" }, Studio.ui.iconButton("edit", `Rename ${node.label}`, () => renameNode(node), { small: true }),
          Studio.ui.iconButton("link", `Connect ${node.label} to another node`, () => connectDialog(node), { small: true }),
          Studio.ui.iconButton("bug", state.breakpoints.has(node.id) ? `Remove breakpoint on ${node.label}` : `Add breakpoint on ${node.label}`, () => toggleBreakpoint(node.id), { small: true }),
          Studio.ui.iconButton("copy", `Duplicate ${node.label}`, duplicateSelection, { small: true }),
          Studio.ui.iconButton("trash", `Delete ${node.label}`, deleteSelection, { small: true })),
      ], Studio.ui.badge(meta.label, "accent")));
      root.append(agentSection("config", "Configuration", primary.length
        ? primary.map(renderField)
        : h("p", { class: "help", text: meta.fields.length ? "This node only has advanced options." : "This node has no configuration; it only marks the flow." })));
      root.append(agentSection("advanced", "Advanced options", [
        ...advanced.map(renderField),
        Studio.ui.kv([["Node id", node.id], ["Position", positionValue]]),
      ], null, false));
      root.append(agentSection("connections", "Connections", [
        h("div", { class: "stack tight" }, outgoing.length || incoming.length ? [
          ...outgoing.map(item => connectionRow(item.edge, item.index, "out")),
          ...incoming.map(item => connectionRow(item.edge, item.index, "in")),
        ] : h("p", { class: "help", text: "This node is not connected yet. Drag from a port or use Connect to…." })),
        h("div", { class: "row wrap" }, Studio.ui.button("Connect to…", { small: true, onclick: () => connectDialog(node) }), Studio.ui.button("Disconnect all", { small: true, kind: "danger", disabled: !(outgoing.length || incoming.length), onclick: () => disconnectNode(node) })),
      ], Studio.ui.badge(`${outgoing.length + incoming.length}`), false));
      root.append(agentSection("errors", "Retries and errors", state.meta.common_fields.map(field => fieldControl(field, node.data[field.key], value => mutate(() => { node.data[field.key] = value; }, false, `Edit ${field.label}`), { node })), null, false));
      if (state.run) {
        const events = (state.run.events || []).filter(event => event.node === node.id);
        const input = [...events].reverse().find(event => event.type === "node_input"); const output = [...events].reverse().find(event => event.type === "node_finished");
        root.append(agentSection("last-run", "Last run", [
          input ? Studio.ui.code(JSON.stringify(input.input, null, 2), "Input") : h("p", { class: "help", text: "No input recorded for this node in the current run." }),
          output ? Studio.ui.code(JSON.stringify(output.output, null, 2), "Output") : null,
          h("div", { class: "row wrap" }, Studio.ui.button("Focus on canvas", { small: true, onclick: () => focusNode(node.id) }), input ? Studio.ui.button("Copy input", { small: true, onclick: () => Studio.copy(JSON.stringify(input.input, null, 2), "Node input copied") }) : null),
        ], Studio.ui.badge(`${events.length} event${events.length === 1 ? "" : "s"}`)));
      }
      return;
    }
    root.append(agentSection("welcome", "Get started", [
      h("p", { class: "help", text: state.agents.some(agent => agent.id === state.doc.id)
        ? "Select a node to edit it, or add the next step to the flow."
        : "Describe what this agent should do, start from a template, or add your first node." }),
      h("div", { class: "row wrap" }, Studio.ui.button("Describe it", { small: true, kind: "primary", onclick: authorAgent }),
        Studio.ui.button("Templates", { small: true, onclick: showTemplates }),
        Studio.ui.button("Add node", { small: true, onclick: () => openQuickAdd() })),
    ]));
    root.append(agentSection("agent", "Agent settings", [
      fieldControl({ label: "Name", type: "text" }, state.doc.name, value => mutate(() => { state.doc.name = value; }, false, "Rename agent")),
      fieldControl({ label: "Description", type: "template", rows: 3 }, state.doc.description, value => mutate(() => { state.doc.description = value; }, false, "Edit description")),
      fieldControl({ label: "Default model", type: "model" }, state.doc.model || "", value => mutate(() => { state.doc.model = value; }, false, "Choose model")),
    ], Studio.ui.badge(state.doc.abilities.length ? `${state.doc.abilities.length} abilities` : "no abilities")));
    root.append(agentSection("limits", "Advanced settings", [
      fieldControl({ label: "Initial variables (JSON)", type: "json", rows: 4,
        help: "Available as vars.<name> in every template before the flow assigns anything." },
        state.doc.vars || {}, value => mutate(() => { state.doc.vars = value; }, false, "Edit variables")),
      fieldControl({ label: "Maximum steps", type: "number", min: 1, max: 10000 }, state.doc.settings.max_steps, value => mutate(() => { state.doc.settings.max_steps = value; }, false, "Edit run limits")),
      fieldControl({ label: "Timeout seconds", type: "number", min: 1, max: 86400 }, state.doc.settings.timeout, value => mutate(() => { state.doc.settings.timeout = value; }, false, "Edit run limits")),
      fieldControl({ label: "On node error", type: "select", options: [["stop", "Stop run"], ["continue", "Continue with error output"]] }, state.doc.settings.on_error, value => mutate(() => { state.doc.settings.on_error = value; }, false, "Edit run limits")),
      fieldControl({ label: "Maximum output characters", type: "number", min: 100, max: 1000000 }, state.doc.settings.max_output_chars, value => mutate(() => { state.doc.settings.max_output_chars = value; }, false, "Edit run limits")),
    ], null, false));
    const swarmFields = [
      ["Maximum agents", "max_agents", { type: "number", min: 1, max: 12 }],
      ["Maximum model rounds per agent", "max_steps", { type: "number", min: 1, max: 24 }],
      ["Maximum feedback rounds", "max_rounds", { type: "number", min: 0, max: 4 }],
      ["Time budget (seconds)", "max_seconds", { type: "number", min: 10, max: 3600 }],
      ["Maximum messages", "max_messages", { type: "number", min: 0, max: 1000 }],
      ["Maximum characters per message", "max_message_chars", { type: "number", min: 200, max: 20000 }],
      ["Maximum shared-memory keys", "max_memory_entries", { type: "number", min: 0, max: 1000 }],
      ["Maximum characters per memory value", "max_memory_chars", { type: "number", min: 200, max: 100000 }],
      ["Shared context pool characters", "max_context_chars", { type: "number", min: 0, max: 2000000, help: "Extra prompt characters members may borrow during one swarm. Agents that read long files request more only while this pool has free space." }],
      ["Context a single agent may hold", "max_agent_context_chars", { type: "number", min: 0, max: 500000 }],
      ["Largest context slice per request", "max_context_item_chars", { type: "number", min: 500, max: 200000 }],
      ["Members may edit files", "allow_write", { type: "boolean", help: "A swarm node must also set write to true; write calls are serialized and snapshotted." }],
      ["Agent-to-agent messages", "allow_ipc", { type: "boolean" }],
      ["Shared blackboard", "shared_memory", { type: "boolean" }],
    ];
    root.append(agentSection("swarm-policy", "Swarm policy", swarmFields.map(([label, key, field]) =>
      fieldControl({ label, ...field }, state.doc.swarm[key],
        value => mutate(() => { state.doc.swarm[key] = value; }, false, `Edit swarm ${label.toLowerCase()}`))), null, false));
    } finally {
      root.scrollTop = scrollTop;
    }
  }

  function renderAll() {
    if (!state.doc || !state.meta) return;
    renderNodes(); renderEdges(); renderInspector(); renderAbilities(); renderProblems(); renderSelectionToolbar(); applyTransform(); renderStatus();
    renderHistoryControls();
    if (state.find.open) refreshFind();
  }

  function renderHistoryControls() {
    const undo = state.undo[state.undo.length - 1]; const redo = state.redo[state.redo.length - 1];
    $("agents-undo").disabled = !undo;
    $("agents-redo").disabled = !redo;
    $("agents-undo").title = undo ? `Undo ${undo.label} (Ctrl+Z)` : "Nothing to undo";
    $("agents-redo").title = redo ? `Redo ${redo.label} (Ctrl+Shift+Z)` : "Nothing to redo";
  }

  function stopWatching() {
    if (state.feed) { state.feed.close(); state.feed = null; }
    state.runToken = (state.runToken || 0) + 1;
  }

  /* One live SSE feed per run.  A snapshot frame carries both the new events
     and the complete node/variable state, so reconnects never lose progress. */
  function watchRun(runId, reset = false) {
    if (!runId) return;
    if (reset) { stopWatching(); state.run = null; state.eventSeq = 0; state.lastRunNode = null; }
    else stopWatching();
    const token = (state.runToken = (state.runToken || 0) + 1);
    state.feed = Studio.subscribe(
      () => `/api/agents/runs/${encodeURIComponent(runId)}/events?since=${state.eventSeq}`,
      snapshot => {
        if (token !== state.runToken || !snapshot || !snapshot.id) return;
        acceptRun(snapshot);
        if (!ACTIVE_RUNS.has(snapshot.status)) stopWatching();
      },
    );
  }

  /* Merge the already-ordered event history with one snapshot's batch in O(n).
     The stream is `since=`-based, so only the small batch needs normalising;
     re-sorting the whole run on every frame was wasted work. */
  function mergeRunEvents(previous, incoming) {
    if (!incoming.length) return previous;
    const batch = incoming.length > 1
      ? Array.from(new Map(incoming.map(event => [event.seq, event])).values())
        .sort((a, b) => (a.seq || 0) - (b.seq || 0))
      : incoming;
    if (!previous.length) return batch;
    const merged = [];
    let left = 0;
    let right = 0;
    while (left < previous.length && right < batch.length) {
      const before = previous[left];
      const after = batch[right];
      const beforeSeq = before.seq || 0;
      const afterSeq = after.seq || 0;
      if (beforeSeq < afterSeq) { merged.push(before); left += 1; }
      else if (afterSeq < beforeSeq) { merged.push(after); right += 1; }
      else { merged.push(after); left += 1; right += 1; }
    }
    while (left < previous.length) { merged.push(previous[left]); left += 1; }
    while (right < batch.length) { merged.push(batch[right]); right += 1; }
    return merged;
  }

  function acceptRun(snapshot) {
    const sameRun = state.run && state.run.id === snapshot.id;
    const previous = sameRun ? state.run.events || [] : [];
    const previousNodes = sameRun ? state.run.nodes || null : null;
    const previousBreakpoints = state.breakpoints;
    const previousSelection = [...state.selected].sort().join(",");
    if (!sameRun) {
      nodePreviewSeq = -1;
      nodePreviewRun = "";
      nodePreviewMap = new Map();
    }
    if (state.run && state.run.id !== snapshot.id) state.lastRunNode = null;
    const unique = mergeRunEvents(previous, snapshot.events || []);
    state.run = { ...snapshot, events: unique };
    const latestSeq = unique.length ? unique[unique.length - 1].seq || 0 : 0;
    state.eventSeq = Math.max(state.eventSeq, snapshot.event_count || 0, latestSeq);
    if (snapshot.debug) state.breakpoints = new Set(snapshot.breakpoints || []);
    const paused = snapshot.status === "paused";
    if (snapshot.current_node && snapshot.current_node !== state.lastRunNode && (state.follow || paused)) {
      state.lastRunNode = snapshot.current_node;
      state.selected = new Set([snapshot.current_node]);
      const previousEdge = state.edge;
      state.edge = null;
      if (previousEdge !== null) paintEdgeSelection(previousEdge, null);
      focusNode(snapshot.current_node);
    }
    /* Snapshots can arrive several times per frame; coalesce the expensive
       panels so a streaming run cannot stall canvas dragging.  Nodes are
       repainted in place, and the inspector is only rebuilt when the data it
       shows (selection, node outcome, breakpoints) actually changed; a
       rebuild would also drop focus from a field being edited. */
    const nodes = state.run.nodes || {};
    const nodesChanged = previousNodes === null
      ? Object.keys(nodes).some(id => nodes[id] && nodes[id] !== "pending")
      : Object.keys(nodes).some(id => (previousNodes[id] || "pending") !== nodes[id]);
    const breakpointsChanged = snapshot.debug
      && [...previousBreakpoints].sort().join(",") !== [...state.breakpoints].sort().join(",");
    const selectionChanged = previousSelection !== [...state.selected].sort().join(",");
    frame("run-panel", renderRun);
    frame("run-nodes", paintRunNodes);
    if (!sameRun || nodesChanged || breakpointsChanged || selectionChanged) frame("run-inspector", renderInspector);
    if (snapshot.awaiting_input || paused) showDetails("run");
  }

  function runInputKey(id) { return `agents-input:${id}`; }

  function defaultRunInput(agent) {
    const stored = Studio.store.get(runInputKey(agent.id), null);
    if (stored !== null && stored !== undefined) return stored;
    const scheduled = agent.schedule && agent.schedule.input;
    if (scheduled && typeof scheduled === "object" && Object.keys(scheduled).length) return scheduled;
    return { message: "Hello" };
  }

  function hasRunInput(id) {
    const stored = Studio.store.get(runInputKey(id), null);
    return stored !== null && stored !== undefined;
  }

  function rememberRunInput(id, input) { Studio.store.set(runInputKey(id), input); }

  async function startRun(mode, options = {}) {
    if (!state.ready || !state.doc) return;
    if (activeRun()) { Studio.toast("A run is already active. Stop it or wait for it to finish.", "info"); return; }
    const saved = await ensureSaved();
    if (!saved) return;
    let input = defaultRunInput(saved);
    const storedInput = hasRunInput(saved.id);
    if (options.prompt || !storedInput) {
      const scheduled = saved.schedule && saved.schedule.input && Object.keys(saved.schedule.input).length;
      const promptValue = !options.prompt && !storedInput && !scheduled ? "Hello" : JSON.stringify(input, null, 2);
      const multiline = Boolean(options.prompt || storedInput || scheduled);
      const raw = await Studio.prompt({ title: `${mode === "test" ? "Test" : mode === "debug" ? "Debug" : "Run"} ${saved.name}`, label: "What should the agent receive? Plain text becomes input.message; JSON is also accepted.", value: promptValue, multiline, confirmLabel: mode === "test" ? "Start test" : mode === "debug" ? "Start debugger" : "Start run" });
      if (raw === null) return;
      input = parseRunInput(raw);
    }
    rememberRunInput(saved.id, input);
    const result = await Studio.run(() => api(`/api/agents/${encodeURIComponent(saved.id)}/runs`, { body: {
      input, trigger: "manual", dry_run: mode === "test", debug: mode === "debug", breakpoints: Array.from(state.breakpoints),
    } }));
    if (result) {
      stopWatching();
      state.run = null; state.eventSeq = 0; state.lastRunNode = null; acceptRun(result); showDetails("run");
      watchRun(result.id);
      await refreshAgents(saved.id);
    }
  }

  function openRunMenu(anchor) {
    const running = activeRun();
    Studio.overlay.menu(anchor, [
      { label: "Run with input…", icon: "edit", disabled: Boolean(running), onSelect: () => startRun("run", { prompt: true }) },
      { label: "Test without side effects", icon: "eye", disabled: Boolean(running), onSelect: () => startRun("test") },
      { label: "Debug with breakpoints", icon: "bug", disabled: Boolean(running), onSelect: () => startRun("debug") },
      "separator",
      { label: "Stop current run", icon: "stop", danger: true, disabled: !running, onSelect: () => runAction("cancel") },
      { label: "Show run history", icon: "clock", onSelect: () => { stopWatching(); state.run = null; state.eventSeq = 0; state.lastRunNode = null; state.runPanelKey = ""; showDetails("run"); renderRun(); } },
    ], { label: "Run options", align: "end" });
  }

  async function runAction(action) {
    if (!state.run) return;
    if (action === "cancel" && !await Studio.confirm({ title: "Cancel this run?", message: "Stop the agent now. Side effects from nodes that already ran cannot be undone.", confirmLabel: "Cancel run", danger: true })) return;
    const result = await Studio.run(() => api(`/api/agents/runs/${encodeURIComponent(state.run.id)}/${action}`, { body: {} }));
    if (result) { acceptRun(result); if (!state.feed) watchRun(result.id); }
  }

  async function provideInput(value) {
    if (!state.run) return;
    const result = await Studio.run(() => api(`/api/agents/runs/${encodeURIComponent(state.run.id)}/input`, { body: { value } }));
    if (result) { acceptRun(result); if (!state.feed) watchRun(result.id); }
  }

  async function loadRunHistory() {
    if (!state.doc || state.run) return;
    const root = $("agents-run-panel");
    if (!state.agents.some(agent => agent.id === state.doc.id)) {
      Studio.ui.render(root, Studio.ui.empty({ compact: true, icon: "play", title: "No runs yet", text: "Press Run to see live events, variables and node outputs here." }));
      return;
    }
    try {
      const data = await api(`/api/agents/${encodeURIComponent(state.doc.id)}/runs`);
      if (state.run) return;
      Studio.ui.render(root, h("div", { class: "stack tight" },
        h("div", { class: "row between" }, h("h3", { text: "Recent runs" }), Studio.ui.button("Refresh", { small: true, onclick: loadRunHistory })),
        data.runs.length ? h("div", { class: "stack tight" }, data.runs.map(run => h("button", { class: "list-item", type: "button", onclick: () => { watchRun(run.id, true); showDetails("run"); } },
          Studio.ui.status(run.status), h("span", { class: "grow", text: `${run.id} · ${run.trigger}${run.dry_run ? " · test" : ""}${run.debug ? " · debug" : ""}` }),
          Studio.ui.badge(`${run.steps || 0} steps`), h("span", { class: "muted tiny", text: format.ago(run.created) }))))
          : Studio.ui.empty({ compact: true, icon: "play", title: "No runs yet", text: "Run the agent once to start a history." })));
    } catch (error) { Studio.toast(error.message, "error"); }
  }

  let runView = null;
  let runEventsTimer = 0;
  let runEventsCache = null;
  const runEventText = new WeakMap();

  function cancelRunEvents() {
    if (!runEventsTimer) return;
    clearTimeout(runEventsTimer);
    runEventsTimer = 0;
  }

  function eventSummary(event) {
    const type = String(event.type || "event");
    const node = String(event.node || event.agent || "");
    const detail = event.input && typeof event.input === "object"
      ? event.input.path || event.input.command || event.input.query || event.input.name || ""
      : "";
    const tool = event.ability || event.tool || "tool";
    const tokens = Number(event.usage && event.usage.completion_tokens) || 0;
    switch (type) {
      case "node_started": return `Started ${event.label || node || "node"}`;
      case "node_finished": return `Finished ${node || "node"}${event.elapsed_ms ? ` in ${event.elapsed_ms} ms` : ""}`;
      case "node_failed": return `Failed ${node || "node"}: ${event.error || "error"}`;
      case "tool_call": return `${tool}${detail ? ` · ${detail}` : ""}`;
      case "tool_result": return `${tool} ${event.ok === false ? "failed" : "returned"}`;
      case "assistant_start": return "Model turn started";
      case "assistant_delta": return `Streaming ${String(event.text || "").length} characters`;
      case "reasoning_delta": return `Thinking · ${String(event.text || "").length} characters`;
      case "assistant_end": return `Model turn finished${tokens ? ` · ${tokens} tokens` : ""}`;
      case "plan": return `Plan updated · ${Array.isArray(event.plan) ? event.plan.length : 0} steps`;
      case "log": return event.message || event.level || "Log message";
      case "swarm_agent_start": return `${event.label || node} started`;
      case "swarm_agent_end": return `${event.label || node} ${event.ok === false ? "failed" : "finished"}`;
      case "swarm_agent_tool": return `${event.label || node} used ${event.tool || "a tool"}`;
      case "swarm_review": return `${event.reviewer} reviewed ${event.target} · ${event.score}/5`;
      case "swarm_round": return `${format.title(String(event.phase || "round"))}${event.round ? ` ${event.round}` : ""}`;
      case "swarm_end": return `Swarm ${event.ok === false ? "failed" : "finished"}`;
      default: return event.message || type;
    }
  }

  function runEventRow(event, currentNode) {
    const details = { ...event }; delete details.seq; delete details.at; delete details.type;
    return h("div", { class: `run-event${event.node ? " clickable" : ""}${event.node === currentNode ? " current" : ""}${event.level === "error" ? " error" : ""}`, title: event.node ? "Select and focus this node" : null, onclick: event.node ? () => { selectNode(event.node); focusNode(event.node); } : null },
      h("span", { class: "event-time", text: String(event.at || "").split("T").pop() }),
      h("div", { class: "grow" },
        h("div", { class: "row between" }, h("strong", { text: `${event.type}${event.node ? ` · ${event.node}` : ""}` }),
          Studio.ui.iconButton("copy", "Copy event JSON", copyEvent => { copyEvent.stopPropagation(); Studio.copy(JSON.stringify(event, null, 2), "Event copied"); }, { small: true })),
        h("p", { class: "run-event-summary", text: eventSummary(event) }),
        h("details", { class: "run-event-details", onclick: click => click.stopPropagation() },
          h("summary", { text: "Details" }),
          h("pre", { class: "muted", text: JSON.stringify(details, null, 1) }))));
  }

  /* Event rows are the expensive part of the run panel.  New events arrive in
     order, so prepend only the fresh ones and trim the oldest; a filter change
     or a reconnecting snapshot still rebuilds the short visible window. */
  function renderRunEvents(host, run) {
    const query = state.runFilter.trim().toLowerCase();
    const hideDeltas = Boolean(state.runHideDeltas);
    const all = run.events || [];
    const latestSeq = all.length ? all[all.length - 1].seq || 0 : 0;
    let cache = runEventsCache;
    const reusable = cache && cache.host === host && cache.query === query
      && cache.hideDeltas === hideDeltas && latestSeq >= cache.lastSeq && all.length >= cache.count;
    if (reusable) {
      for (const row of cache.rows) {
        row.element.classList.toggle("current", Boolean(row.node) && row.node === run.current_node);
      }
      const fresh = [];
      for (let index = all.length - 1; index >= 0; index -= 1) {
        const event = all[index];
        if ((event.seq || 0) <= cache.lastSeq) break;
        fresh.push(event);
      }
      fresh.reverse();
      for (const event of fresh) {
        if (hideDeltas && DELTA_EVENTS.has(event.type)) continue;
        if (query && !eventHaystack(event).includes(query)) continue;
        const element = runEventRow(event, run.current_node);
        host.prepend(element);
        cache.rows.unshift({ element, node: event.node });
      }
      while (cache.rows.length > 140) cache.rows.pop().element.remove();
      if (cache.rows.length && cache.placeholder) {
        cache.placeholder.remove();
        cache.placeholder = null;
      }
      cache.lastSeq = Math.max(cache.lastSeq, latestSeq);
      cache.count = all.length;
      return;
    }
    const events = all
      .filter(event => !(hideDeltas && DELTA_EVENTS.has(event.type)))
      .filter(event => !query || eventHaystack(event).includes(query))
      .slice(-140).reverse();
    Studio.ui.render(host);
    cache = { host, query, hideDeltas, lastSeq: latestSeq, count: all.length, rows: [], placeholder: null };
    runEventsCache = cache;
    if (!events.length) {
      cache.placeholder = h("p", { class: "muted small", text: query
        ? "No event matches this filter."
        : hideDeltas
          ? "Token deltas are hidden. Uncheck “Hide token deltas” to inspect them; live text is shown above."
          : "The run has not emitted events yet." });
      host.append(cache.placeholder);
      return;
    }
    for (const event of events) {
      const element = runEventRow(event, run.current_node);
      cache.rows.push({ element, node: event.node });
      host.append(element);
    }
  }

  function eventHaystack(event) {
    let text = runEventText.get(event);
    if (text === undefined) {
      text = JSON.stringify(event).toLowerCase();
      runEventText.set(event, text);
    }
    return text;
  }

  function scheduleRunEvents(run, immediate = false) {
    if (!runView) return;
    if (immediate) {
      cancelRunEvents();
      renderRunEvents(runView.eventsHost, run);
      return;
    }
    if (runEventsTimer) return;
    runEventsTimer = setTimeout(() => {
      runEventsTimer = 0;
      if (runView && state.run) renderRunEvents(runView.eventsHost, state.run);
    }, 120);
  }

  /* The live pane follows the current model turn and every active swarm
     member.  Deltas are hidden from the event log by default, so this is the
     readable tail of a streaming run instead of a wall of JSON frames. */
  function livePreview(value) {
    const text = String(value || "");
    const trimmed = text.trimStart();
    if (trimmed.startsWith("{") || trimmed.startsWith("```")) {
      return "Preparing a tool call…";
    }
    return text.slice(-6000);
  }

  function runLiveBlocks(run) {
    const blocks = new Map();
    const labels = new Map();
    let order = 0;
    let current = null;
    let previousAt = NaN;
    const ensure = (key, label, node) => {
      let block = blocks.get(key);
      if (!block) {
        block = { key, label, node, text: "", reasoning: "", note: "", live: false, order: 0,
          startedAt: 0, firstAt: 0, firstTextAt: 0, lastAt: 0, chars: 0, rate: 0, latency: null,
          totalChars: 0, reasoningChars: 0 };
        blocks.set(key, block);
      }
      return block;
    };
    const turn = (event) => {
      const key = `node:${String(event.node || "")}:${event.iteration || 0}`;
      if (!current || current.key !== key) {
        if (current) current.live = false;
        const node = String(event.node || "");
        current = ensure(key, labels.get(node) || node || "model", node);
        current.startedAt = Number.isFinite(previousAt) ? previousAt : 0;
      }
      return current;
    };
    for (const event of run.events || []) {
      const type = event.type;
      const node = String(event.node || "");
      const at = Date.parse(event.at || "");
      if (type === "node_started") {
        if (node) labels.set(node, event.label || node);
        if (current) current.live = false;
        current = null;
      } else if (type === "assistant_delta") {
        current = turn(event);
        current.text = `${current.text}${event.text || ""}`.slice(-6000);
        current.totalChars += (event.text || "").length;
        if (Number.isFinite(at)) {
          if (!current.firstAt) current.firstAt = at;
          if (!current.firstTextAt) current.firstTextAt = at;
          current.lastAt = at;
        }
        current.chars = current.totalChars;
        current.note = "writing";
        current.live = true;
        current.order = ++order;
      } else if (type === "reasoning_delta") {
        current = turn(event);
        current.reasoning = `${current.reasoning}${event.text || ""}`.slice(-4000);
        current.reasoningChars += (event.text || "").length;
        if (Number.isFinite(at)) {
          if (!current.firstAt) current.firstAt = at;
          current.lastAt = at;
        }
        current.note = "thinking";
        current.live = true;
        current.order = ++order;
      } else if (type === "tool_call" || type === "tool_result") {
        if (current) {
          current.note = type === "tool_call"
            ? `using ${event.ability || event.tool || "a tool"}…`
            : `${event.ability || event.tool || "tool"} ${event.ok === false ? "failed" : "returned"}`;
          current.order = ++order;
        }
      } else if (type === "node_finished" || type === "node_failed") {
        if (current) {
          current.live = false;
          current.note = type === "node_failed" ? "failed" : "done";
          current.order = ++order;
        }
      } else if (type === "swarm_agent_start") {
        const id = String(event.agent || event.label || "leader");
        current = ensure(`agent:${id}`, event.label || id, node);
        current.text = "";
        current.reasoning = "";
        current.note = event.message || "started";
        current.startedAt = Number.isFinite(at) ? at : 0;
        current.firstAt = 0;
        current.firstTextAt = 0;
        current.lastAt = 0;
        current.chars = 0;
        current.rate = 0;
        current.latency = null;
        current.totalChars = 0;
        current.reasoningChars = 0;
        current.live = true;
        current.order = ++order;
      } else if (type === "swarm_agent_delta") {
        const id = String(event.agent || event.label || "leader");
        const block = ensure(`agent:${id}`, event.label || id, node);
        block.text = `${block.text}${event.text || ""}`.slice(-6000);
        block.totalChars += (event.text || "").length;
        if (Number.isFinite(at)) {
          if (!block.firstAt) block.firstAt = at;
          if (!block.firstTextAt) block.firstTextAt = at;
          block.lastAt = at;
        }
        block.chars = block.totalChars;
        block.note = event.phase === "review" ? "reviewing" : "writing";
        block.live = true;
        block.order = ++order;
        current = block;
      } else if (type === "swarm_agent_status") {
        const id = String(event.agent || event.label || "leader");
        const block = ensure(`agent:${id}`, event.label || id, node);
        block.note = event.message || block.note;
        block.order = ++order;
      } else if (type === "swarm_agent_tool") {
        const id = String(event.agent || event.label || "leader");
        const block = ensure(`agent:${id}`, event.label || id, node);
        block.note = event.message || `using ${event.tool || "a tool"}…`;
        block.live = true;
        block.order = ++order;
      } else if (type === "swarm_agent_end") {
        const id = String(event.agent || event.label || "leader");
        const block = ensure(`agent:${id}`, event.label || id, node);
        block.note = event.ok === false ? "failed" : "finished";
        block.live = false;
        block.order = ++order;
      } else if (type === "swarm_end") {
        for (const block of blocks.values()) {
          if (block.key.startsWith("agent:")) block.live = false;
        }
      }
      previousAt = Number.isFinite(at) ? at : previousAt;
    }
    const rows = [...blocks.values()].filter(block => block.text || block.reasoning);
    for (const block of rows) {
      block.chars = block.totalChars || block.text.length;
      block.reasoningChars = block.reasoningChars || block.reasoning.length;
      block.latency = block.firstAt && block.startedAt
        ? Math.max(0, (block.firstAt - block.startedAt) / 1000) : null;
      block.rate = block.firstTextAt && block.lastAt > block.firstTextAt && block.chars
        ? Math.round(block.chars / ((block.lastAt - block.firstTextAt) / 1000)) : 0;
    }
    return rows
      .sort((left, right) => right.order - left.order)
      .slice(0, 6);
  }

  function nodeLivePreviews(run) {
    const blocks = runLiveBlocks(run);
    const counts = new Map();
    for (const block of blocks) {
      if (!block.node || !block.key.startsWith("agent:")) continue;
      counts.set(block.node, (counts.get(block.node) || 0) + 1);
    }
    const previews = new Map();
    for (const block of blocks) {
      if (!block.node) continue;
      const existing = previews.get(block.node);
      const better = !existing
        || (block.live && !existing.live)
        || (block.live === existing.live && block.order > existing.order);
      if (!better) continue;
      const group = block.key.startsWith("agent:");
      const count = counts.get(block.node) || 0;
      previews.set(block.node, {
        order: block.order,
        label: group && count > 1 ? `${count} agents · ${block.label}` : block.label,
        group,
        text: livePreview(block.text) || (block.reasoning ? "Thinking…" : ""),
        live: block.live,
      });
    }
    return previews;
  }

  function runTokenTotal(run) {
    let total = 0;
    for (const event of run.events || []) {
      if (!["assistant_end", "swarm_agent_end", "node_finished"].includes(event.type)) continue;
      const usage = event.usage || (event.output && event.output.usage);
      total += Number(usage && usage.completion_tokens) || 0;
    }
    return total;
  }

  function renderRunLive(view, run) {
    const active = activeRun();
    view.live.classList.toggle("hidden", !active);
    if (!active) {
      view.liveKey = "";
      view.liveSeq = -1;
      return;
    }
    const events = run.events || [];
    const latest = events.length ? events[events.length - 1].seq || 0 : 0;
    if (view.liveSeq !== latest) {
      view.liveSeq = latest;
      view.liveBlocks = runLiveBlocks(run);
    }
    const blocks = view.liveBlocks || [];
    const streaming = blocks.filter(block => block.live).length;
    if (blocks.length) {
      view.liveStatus.textContent = streaming ? `${streaming} streaming` : "Waiting for the next step…";
    } else {
      const last = events[events.length - 1];
      view.liveStatus.textContent = last ? eventSummary(last) : "Waiting for model output…";
    }
    view.liveSpinner.classList.toggle("hidden", !streaming);
    const key = blocks.map(block => [block.key, block.text.length, block.chars,
      block.reasoning.length, block.reasoningChars, block.live ? 1 : 0, block.note].join(":")).join("|");
    if (key === view.liveKey) return;
    view.liveKey = key;
    const previousTop = view.liveBody.scrollTop;
    Studio.ui.render(view.liveBody);
    for (const block of blocks) {
      const reasoning = block.reasoning
        ? h("details", { class: "run-live-reasoning", open: block.live && !block.text },
          h("summary", { text: `Thinking · ${(block.reasoningChars || block.reasoning.length).toLocaleString()} chars` }),
          h("div", { class: "reasoning-live", text: block.reasoning.slice(-3000) }))
        : null;
      const metrics = [];
      if (block.chars) metrics.push(`${block.chars.toLocaleString()} chars`);
      if (block.rate) metrics.push(`${block.rate} ch/s`);
      if (block.latency !== null && block.latency !== undefined) {
        metrics.push(`${block.latency.toFixed(1)}s to first`);
      }
      view.liveBody.append(h("section", { class: "run-live-block" },
        h("div", { class: "run-live-block-head" },
          block.live ? h("span", { class: "spinner", "aria-hidden": "true" })
            : h("span", { class: "swarm-glyph" }, Studio.icon(block.note === "failed" ? "alert" : "check", "small")),
          h("strong", { class: "grow", text: block.label }),
          h("span", { class: "muted tiny", text: block.note || "" })),
        metrics.length ? h("p", { class: "run-live-metrics muted tiny", text: metrics.join(" · ") }) : null,
        reasoning,
        block.text ? h("div", { class: "run-live-text", text: livePreview(block.text) }) : null));
    }
    if (view.liveFollow.checked) view.liveBody.scrollTop = view.liveBody.scrollHeight;
    else view.liveBody.scrollTop = previousTop;
  }

  /* Build the panel once per run.  Streaming snapshots then update the few
     values that changed instead of recreating the filter input (and the
     caret), the disclosure bodies and every event row several times a second. */
  function buildRunView(run) {
    if (state.runHideDeltas === null) {
      state.runHideDeltas = Studio.store.get("agents-run-hide-deltas", true) !== false;
    }
    const statusHost = h("span");
    statusHost.append(Studio.ui.status(run.status));
    const steps = Studio.ui.badge(`${run.steps || 0} steps`);
    const elapsed = Studio.ui.badge(format.duration(run.elapsed || 0));
    const tokens = Studio.ui.badge("0 tokens");
    tokens.classList.add("hidden");
    const liveIndicator = h("span", { class: "run-live-indicator hidden" },
      h("span", { class: "spinner", "aria-hidden": "true" }), h("span", { text: "Live" }));
    const stop = Studio.ui.button("Stop", { small: true, kind: "danger", onclick: () => runAction("cancel") });
    const nodeLabel = h("span", { class: "muted tiny" });
    const header = h("div", { class: "row between wrap" }, h("div", { class: "row wrap" }, statusHost, h("strong", { text: run.id }), steps, elapsed, tokens, liveIndicator,
      run.dry_run ? Studio.ui.badge("Test mode", "accent") : null, run.debug ? Studio.ui.badge("Debugger", "accent") : null, stop), nodeLabel);
    const extras = h("div", { class: "stack tight" });
    const filter = h("input", { id: "agents-run-filter", type: "search", value: state.runFilter, placeholder: "Filter events by node, type or text…", "aria-label": "Filter run events" });
    const eventsHost = h("div", { class: "stack tight" });
    filter.addEventListener("input", () => {
      state.runFilter = filter.value;
      cancelRunEvents();
      if (state.run) renderRunEvents(eventsHost, state.run);
    });
    const follow = Studio.form.control({ type: "checkbox", value: state.follow });
    follow.addEventListener("change", () => {
      state.follow = follow.checked;
      if (state.follow && state.run && state.run.current_node) focusNode(state.run.current_node);
    });
    const hideDeltas = Studio.form.control({ type: "checkbox", value: state.runHideDeltas });
    hideDeltas.addEventListener("change", () => {
      state.runHideDeltas = hideDeltas.checked;
      Studio.store.set("agents-run-hide-deltas", hideDeltas.checked);
      cancelRunEvents();
      if (state.run) renderRunEvents(eventsHost, state.run);
    });
    const toolbar = h("div", { class: "agent-run-toolbar" }, filter,
      h("label", { class: "check" }, follow, "Follow running node"),
      h("label", { class: "check" }, hideDeltas, "Hide token deltas"),
      Studio.ui.button("Run history", { small: true, onclick: () => { stopWatching(); state.run = null; state.eventSeq = 0; state.lastRunNode = null; state.runPanelKey = ""; renderRun(); } }),
      Studio.ui.button("Copy run JSON", { small: true, onclick: () => { if (state.run) Studio.copy(JSON.stringify(state.run, null, 2), "Run copied"); } }));
    const liveStatus = h("span", { class: "muted tiny", "aria-live": "polite" });
    const liveBody = h("div", { class: "run-live-body", "aria-live": "off" });
    const liveFollow = Studio.form.control({ type: "checkbox", value: true });
    liveFollow.addEventListener("change", () => {
      if (liveFollow.checked) liveBody.scrollTop = liveBody.scrollHeight;
    });
    liveBody.addEventListener("scroll", () => {
      const pinned = liveBody.scrollHeight - liveBody.scrollTop - liveBody.clientHeight < 24;
      liveFollow.checked = pinned;
    }, { passive: true });
    const liveSpinner = h("span", { class: "spinner", "aria-hidden": "true" });
    const live = h("div", { class: "run-live hidden" },
      h("div", { class: "run-live-head" }, liveSpinner,
        h("strong", { class: "grow", text: "Live output" }), liveStatus,
        Studio.ui.iconButton("copy", "Copy live output", () => {
          if (!state.run) return;
          const text = runLiveBlocks(state.run)
            .map(block => `${block.label}\n${block.text}`).join("\n\n");
          Studio.copy(text || "No live output yet", "Live output copied");
        }, { small: true }),
        h("label", { class: "check" }, liveFollow, "Follow")),
      liveBody);
    const outputBody = h("div", { class: "disclosure-body" });
    const output = h("details", { class: "disclosure", open: true }, h("summary", { text: "Final output" }), outputBody);
    const varsBody = h("div", { class: "disclosure-body" });
    const variables = h("details", { class: "disclosure" }, h("summary", { text: "Variables" }), varsBody);
    const root = h("div", { class: "stack tight" }, header, extras, toolbar, live, output, variables, eventsHost);
    return {
      id: run.id, root, statusHost, statusValue: run.status, steps, elapsed, tokens, liveIndicator, stop, nodeLabel,
      extras, extraKey: null, filter, eventsHost, live, liveStatus, liveBody, liveSpinner, liveFollow,
      liveKey: "", liveSeq: -1, liveBlocks: [],
      output, outputBody, outputText: null, varsBody, varsText: null,
    };
  }

  function syncRunView(view, run) {
    if (view.statusValue !== run.status) {
      view.statusValue = run.status;
      view.statusHost.replaceChildren(Studio.ui.status(run.status));
    }
    view.steps.textContent = `${run.steps || 0} steps`;
    view.elapsed.textContent = format.duration(run.elapsed || 0);
    const totalTokens = runTokenTotal(run);
    view.tokens.textContent = `${totalTokens.toLocaleString()} tokens`;
    view.tokens.classList.toggle("hidden", !totalTokens);
    view.liveIndicator.classList.toggle("hidden", !activeRun());
    view.stop.classList.toggle("hidden", !activeRun());
    view.nodeLabel.textContent = run.error || run.current_node || "";
    const request = run.awaiting_input || null;
    const extraKey = [run.error || "", run.status, run.debug ? "debug" : "", run.current_node || "",
      request ? JSON.stringify(request) : ""].join("\u0000");
    if (extraKey !== view.extraKey) {
      view.extraKey = extraKey;
      Studio.ui.render(view.extras);
      if (run.error) view.extras.append(Studio.ui.callout("danger", h("strong", { text: "Run failed" }), h("p", { class: "small", text: run.error })));
      if (run.status === "paused" && run.debug) {
        view.extras.append(h("div", { class: "callout warning stack" }, h("strong", { text: `Paused at ${run.current_node || "a breakpoint"}` }),
          h("div", { class: "row wrap" }, Studio.ui.button("Step one node", { kind: "primary", onclick: () => runAction("step") }), Studio.ui.button("Resume", { onclick: () => runAction("resume") }))));
      }
      if (request) {
        const form = h("div", { class: "callout warning stack" }, h("strong", { text: request.prompt || "Input required" }));
        if (request.kind === "approve") form.append(h("div", { class: "row" }, Studio.ui.button("Approve", { kind: "primary", onclick: () => provideInput(true) }), Studio.ui.button("Reject", { kind: "danger", onclick: () => provideInput(false) })));
        else if (request.kind === "choice") {
          const select = h("select", {}, (request.choices || []).map(choice => h("option", { value: choice, text: choice })));
          form.append(h("div", { class: "input-action" }, select, Studio.ui.button("Submit", { kind: "primary", onclick: () => provideInput(select.value) })));
        } else {
          const input = h("input", { value: request.default || "", "aria-label": "Human response" });
          form.append(h("div", { class: "input-action" }, input, Studio.ui.button("Submit", { kind: "primary", onclick: () => provideInput(input.value) })));
          input.addEventListener("keydown", event => { if (event.key === "Enter") provideInput(input.value); });
        }
        view.extras.append(form);
      }
    }
    if (run.output === null || run.output === undefined) {
      view.output.classList.add("hidden");
    } else {
      view.output.classList.remove("hidden");
      const text = JSON.stringify(run.output, null, 2);
      if (text !== view.outputText) {
        view.outputText = text;
        Studio.ui.render(view.outputBody, Studio.ui.code(text));
      }
    }
    const varsText = JSON.stringify(run.vars || {}, null, 2);
    if (varsText !== view.varsText) {
      view.varsText = varsText;
      Studio.ui.render(view.varsBody, Studio.ui.code(varsText));
    }
    renderRunLive(view, run);
    scheduleRunEvents(run, !activeRun());
  }

  function renderRun() {
    const running = activeRun();
    $("agents-run").disabled = !state.ready || running;
    $("agents-run").title = running ? "A run is already active" : "Run this agent";
    const panelElement = $("agents-run-panel");
    if (!state.run) {
      cancelRunEvents();
      runView = null;
      state.runPanelKey = "";
      paintRunNodes();
      loadRunHistory();
      renderStatus();
      return;
    }
    const run = state.run;
    if (!runView || runView.id !== run.id || !panelElement.contains(runView.root)) {
      cancelRunEvents();
      const previousScroll = panelElement.scrollTop;
      const scrollTop = state.runPanelKey === run.id ? previousScroll : 0;
      state.runPanelKey = run.id;
      runView = buildRunView(run);
      Studio.ui.render(panelElement, runView.root);
      renderRunEvents(runView.eventsHost, run);
      panelElement.scrollTop = scrollTop;
    }
    syncRunView(runView, run);
    renderStatus();
  }

  function inspectorSignature() {
    return state.edge !== null ? `edge:${state.edge}` : `nodes:${[...state.selected].sort().join(",")}`;
  }

  function refreshInspectorAfterDrag() {
    if (inspectorSignature() === state.inspectorSelection) paintInspectorPosition();
    else renderInspector();
  }

  function paintInspectorPosition() {
    const value = inspectorPosition;
    if (!value || !value.isConnected || !state.doc || state.selected.size !== 1) return;
    const node = nodeById([...state.selected][0]);
    if (!node) return;
    const text = `${Math.round(node.x)}, ${Math.round(node.y)}`;
    if (value.textContent !== text) value.textContent = text;
  }

  function renderSchedule() {
    if (!state.doc) return;
    const schedule = state.doc.schedule;
    const root = Studio.ui.render($("agents-schedule"));
    const enabled = Studio.form.control({ type: "checkbox", value: schedule.enabled });
    const type = Studio.form.control({ type: "select", options: [["manual", "Manual only"], ["interval", "Every N minutes"], ["cron", "Cron expression"], ["once", "Once at a local time"]], value: schedule.type });
    const interval = Studio.form.control({ type: "number", min: 1, max: 525600, value: schedule.interval_minutes || 60 });
    const cron = Studio.form.control({ type: "text", value: schedule.cron || "", placeholder: "*/15 9-17 * * mon-fri", spellcheck: false });
    const once = Studio.form.control({ type: "datetime-local", value: schedule.once_at || "" });
    const input = Studio.form.control({ type: "textarea", rows: 3, value: JSON.stringify(schedule.input || {}, null, 2), spellcheck: false });
    const status = h("div", { class: "muted small" });
    function update() {
      pushHistory("Edit schedule");
      try { schedule.input = JSON.parse(input.value || "{}"); }
      catch (error) { Studio.toast("Schedule input must be valid JSON.", "error"); return false; }
      Object.assign(schedule, { enabled: enabled.checked, type: type.value, interval_minutes: Number(interval.value), cron: cron.value, once_at: once.value });
      markDirty(); scheduleValidation();
      return true;
    }
    for (const control of [enabled, type, interval, cron, once, input]) control.addEventListener("change", update);
    async function triggerNow() {
      if (!update()) return;
      const saved = await ensureSaved();
      if (!saved) return;
      const result = await Studio.run(() => api(`/api/agents/${encodeURIComponent(saved.id)}/trigger`, { body: { input: schedule.input, dry_run: false } }));
      if (!result) return;
      stopWatching(); state.run = null; state.eventSeq = 0; state.lastRunNode = null; acceptRun(result); showDetails("run");
      watchRun(result.id);
      await refreshAgents(saved.id);
    }
    root.append(h("div", { class: "grid-3" }, Studio.ui.field("Schedule type", type), Studio.ui.field("Interval minutes", interval), Studio.ui.field("Cron expression", cron), Studio.ui.field("Run once at", once)),
      Studio.ui.field("Scheduled input (JSON object)", input), h("label", { class: "check" }, enabled, "Enable this schedule while Studio is running"),
      h("div", { class: "row wrap" }, Studio.ui.button("Save & apply schedule", { kind: "primary", onclick: async () => {
        if (!update()) return; const saved = await ensureSaved(); if (!saved) return;
        const data = await Studio.run(() => api(`/api/agents/${encodeURIComponent(saved.id)}/schedule`, { body: { enabled: schedule.enabled } }));
        if (data) { state.doc.schedule = data.schedule; markDirty(false); status.textContent = `Next run: ${data.next_run || "not scheduled"} · Last run: ${data.last_run || "never"}`; Studio.toast("Schedule applied.", "success"); }
      } }), Studio.ui.button("Run now with this input", { onclick: triggerNow })), status);
    if (state.agents.some(agent => agent.id === state.doc.id)) api(`/api/agents/${encodeURIComponent(state.doc.id)}/schedule`).then(data => {
      status.textContent = `Server time: ${data.server_time} · Next run: ${data.next_run || "not scheduled"} · Last run: ${data.last_run || "never"}`;
    }).catch(() => {});
  }

  function renderSpec() {
    if (!state.meta) return;
    const root = Studio.ui.render($("agents-spec"));
    const content = h("div", { class: "stack" }, h("p", { class: "muted small", text: `AML ${state.meta.aml_version} · AMP ${state.meta.version}. These public specifications are additive and machine-readable through the protocol endpoint.` }),
      h("div", { class: "row wrap" }, Studio.ui.button("Read AML", { onclick: () => showSpec("aml") }), Studio.ui.button("Read AMP", { onclick: () => showSpec("amp") }),
        Studio.ui.button("Copy protocol metadata", { onclick: () => Studio.copy(JSON.stringify(state.meta, null, 2), "Protocol metadata copied") }),
        Studio.ui.button("Export agent JSON", { onclick: () => { if (state.doc) Studio.download(`${state.doc.id}.json`, JSON.stringify(state.doc, null, 2)); } })));
    async function showSpec(name) {
      try {
        const text = await api(state.meta.specifications[name]);
        const view = Studio.overlay.modal({ key: `spec:${name}`, title: name.toUpperCase(), wide: true });
        if (view.reused) return;
        Studio.markdown.into(view.body, text);
      } catch (error) { Studio.toast(error.message, "error"); }
    }
    root.append(content);
  }

  function newAgentDialog() {
    const view = Studio.overlay.modal({ key: "new-agent", title: "Create an agent", subtitle: "Pick the fastest starting point; everything can be changed later.", wide: true });
    if (view.reused) return;
    const card = options => h("button", { class: "action-card agent-action-card", type: "button", onclick: options.onclick },
      h("span", { class: "action-icon" }, Studio.icon(options.glyph, "large")),
      h("strong", { text: options.title }),
      h("span", { class: "muted small", text: options.description }));
    view.body.append(h("div", { class: "action-grid" }, [
      card({ glyph: "sparkle", title: "Describe it", description: "Say what the agent should do; a model drafts the flow for your review.", onclick: async () => { view.close(); await authorAgent(); } }),
      card({ glyph: "agents", title: "Use a template", description: "Start from a working example and adapt each step.", onclick: async () => { view.close(); await showTemplates(); } }),
      card({ glyph: "plus", title: "Start blank", description: "A clean Start → End flow to build yourself.", onclick: async () => { view.close(); if (await confirmDiscard()) { loadDocument(blankAgent()); $("agents-select").value = ""; } } }),
      card({ glyph: "upload", title: "Import AML", description: "Paste an agent definition and continue in the designer.", onclick: () => { view.close(); importAgent(); } }),
    ]), h("p", { class: "help", text: "Nothing is saved until you press Save. Templates and drafts stay editable in the designer." }));
  }

  async function showTemplates() {
    const data = await Studio.run(() => api("/api/agents/templates"));
    if (!data) return;
    const view = Studio.overlay.modal({ key: "templates", title: "Agent templates", subtitle: "Start from a complete flow, then adapt it in the designer.", wide: true });
    if (view.reused) return;
    view.body.append(h("div", { class: "result-grid" }, data.templates.map(template => Studio.components.resultCard({
      glyph: "agents",
      title: `${template.icon} ${template.name}`,
      description: template.tagline,
      meta: `${template.nodes} nodes`,
      actions: [Studio.ui.button("Use template", { kind: "primary", small: true, onclick: async () => {
        view.close();
        if (!await confirmDiscard()) return;
        const agentDocument = template.aml || await api(`/api/agents/templates/${encodeURIComponent(template.id)}`);
        if (state.agents.some(agent => agent.id === agentDocument.id)) { agentDocument.id += "-copy"; agentDocument.name += " copy"; }
        loadDocument(agentDocument); markDirty();
      } })],
    }))));
  }

  async function authorAgent() {
    await Studio.models.ensure();
    const description = h("textarea", { rows: 6, placeholder: "Example: Fetch a product page every morning, summarize price changes, and ask me before saving the report.", "aria-label": "Agent description" });
    const model = h("select", {}, Studio.models.chat().map(entry => h("option", { value: entry.id, text: entry.name, disabled: !entry.available })));
    const view = Studio.overlay.modal({ key: "author", title: "Describe an agent", subtitle: "A model drafts AML for your review. Nothing is saved or run automatically.", wide: true });
    if (view.reused) return;
    view.body.append(Studio.ui.field("What should the agent do?", description), Studio.ui.field("Authoring model", model), Studio.ui.callout("info", "The draft is validated and repaired up to two times. Review every ability, prompt, branch and schedule before running it."));
    view.footer.append(Studio.ui.button("Generate draft", { kind: "primary", onclick: async event => {
      const result = await Studio.run(() => api("/api/agents/generate", { body: { description: description.value, model: model.value } }), event.currentTarget);
      if (!result) return;
      if (!await confirmDiscard()) return;
      if (state.agents.some(agent => agent.id === result.agent.id)) result.agent.id += "-draft";
      loadDocument(result.agent); markDirty(); view.close();
      Studio.toast(`Draft generated by ${result.model} in ${result.attempts} attempt(s). Review and save it.`, "success");
    } }));
  }

  function importAgent() {
    const view = Studio.overlay.modal({ key: "import", title: "Import AML", subtitle: "Paste JSON, or YAML when PyYAML is installed. Import validates and saves a new agent.", wide: true });
    if (view.reused) return;
    const text = h("textarea", { rows: 16, spellcheck: false, placeholder: '{"aml":"1.1","kind":"agent",…}' });
    view.body.append(text);
    view.footer.append(Studio.ui.button("Import", { kind: "primary", onclick: async event => {
      if (!await confirmDiscard()) return;
      const result = await Studio.run(() => api("/api/agents/import", { body: { text: text.value } }), event.currentTarget);
      if (result) { loadDocument(result.agent); await refreshAgents(result.agent.id); view.close(); Studio.toast("Agent imported and saved.", "success"); }
    } }));
  }

  async function deleteAgent() {
    if (!state.agents.some(agent => agent.id === state.doc.id)) { loadDocument(blankAgent()); return; }
    if (!await Studio.confirm({ title: "Delete agent?", message: `Delete ${state.doc.name} and its run history? Files written by its abilities are kept.`, confirmLabel: "Delete agent", danger: true })) return;
    const result = await Studio.run(() => api(`/api/agents/${encodeURIComponent(state.doc.id)}`, { method: "DELETE" }));
    if (result) { await refreshAgents(); loadDocument(blankAgent()); Studio.toast("Agent deleted.", "success"); }
  }

  function moreActions() {
    Studio.overlay.menu($("agents-more"), [
      { heading: "Start another" },
      { label: "Use a template", icon: "agents", onSelect: showTemplates },
      { label: "Describe an agent", icon: "sparkle", onSelect: authorAgent },
      "separator",
      { label: "Import JSON / YAML", icon: "upload", onSelect: importAgent },
      { label: "Export JSON", icon: "download", onSelect: () => Studio.download(`${state.doc.id}.json`, JSON.stringify(state.doc, null, 2)) },
      { label: "Duplicate agent", icon: "copy", onSelect: () => {
        const agentDocument = copy(state.doc); let index = 1; const base = `${state.doc.id}-copy`;
        while (state.agents.some(agent => agent.id === agentDocument.id)) agentDocument.id = `${base}${index === 1 ? "" : `-${index}`}` , index += 1;
        agentDocument.name += " copy"; loadDocument(agentDocument); markDirty();
      } },
      "separator",
      { label: "Schedule", icon: "clock", onSelect: () => showDetails("schedule") },
      { label: "AML / AMP specifications", icon: "book", onSelect: () => showDetails("spec") },
      "separator", { label: "Delete agent", icon: "trash", danger: true, onSelect: deleteAgent },
    ], { align: "end" });
  }

  /* -- workspace panes ------------------------------------------------------ */

  function paintPaneToggle(id, open, visible, label, shortcut) {
    const button = $(id);
    button.setAttribute("aria-pressed", String(open));
    button.title = visible ? `Hide ${label} (${shortcut})`
      : open ? `${label[0].toUpperCase()}${label.slice(1)} folded for space (${shortcut})` : `Show ${label} (${shortcut})`;
    button.setAttribute("aria-label", button.title);
  }

  function updateLayoutControls() {
    const dock = dockLayout && dockLayout.read();
    if (dock) paintPaneToggle("agents-toggle-details", !dock.agent.collapsed, dock.agent.visible, "details panel", "Ctrl+J");
  }

  function setupLayout() {
    const savedDock = Studio.store.get("agents-details-layout", null);
    layout = Studio.split.create({
      root: $("agents-designer"),
      key: "agents-layout",
      mode: "columns",
      gap: 6,
      centerMin: 360,
      fallbackWidth: 1280,
      fallbackHeight: 720,
      sidebar: {
        splitter: $("agents-split-palette"),
        min: 170, max: 320, defaultSize: 220, label: "Node palette",
      },
      agent: {
        splitter: $("agents-split-inspector"),
        min: 250, max: 460, defaultSize: 300, label: "Inspector",
      },
      onChange: updateLayoutControls,
      onBlocked(name) {
        Studio.toast(name === "sidebar"
          ? "The node palette needs more room: widen the window, reset the panes, or use the panel menu."
          : "The inspector needs more room: widen the window, reset the panes, or use the panel menu.", "info");
      },
    });
    dockLayout = Studio.split.create({
      root: $("agents-canvas-column"),
      key: "agents-details-layout",
      mode: "stacked",
      gap: 6,
      centerMin: 320,
      centerMinHeight: 220,
      fallbackWidth: 900,
      fallbackHeight: 700,
      sidebar: { enabled: false },
      agent: {
        splitter: $("agents-split-dock"),
        min: 280, max: 460, defaultSize: 300,
        minHeight: 120, maxHeight: 560, defaultHeight: 200,
        label: "Details panel",
        variable: "--agents-dock-size",
        heightVariable: "--agents-dock-height",
      },
      onChange: updateLayoutControls,
      onBlocked() {
        Studio.toast("The details panel needs more room: enlarge the window or collapse another pane.", "info");
      },
    });
    updateLayoutControls();
    if (!savedDock) dockLayout.setCollapsed("agent", true);
  }

  function openPaneMenu(anchor) {
    if (!layout || !dockLayout) return;
    const panels = layout.read();
    const dock = dockLayout.read();
    if (!panels || !dock) return;
    const paletteAuto = !panels.sidebar.visible && !panels.sidebar.collapsed;
    const inspectorAuto = !panels.agent.visible && !panels.agent.collapsed;
    const detailsAuto = !dock.agent.visible && !dock.agent.collapsed;
    Studio.overlay.menu(anchor, [
      { heading: "Panels" },
      { label: panels.sidebar.collapsed ? "Show node palette" : "Hide node palette", icon: "sidebar", hint: "Ctrl+B",
        disabled: paletteAuto, title: paletteAuto ? "Widen the window to make room" : null,
        onSelect: () => layout.toggle("sidebar") },
      { label: panels.agent.collapsed ? "Show inspector" : "Hide inspector", icon: "tune", hint: "Ctrl+Shift+B",
        disabled: inspectorAuto, title: inspectorAuto ? "Widen the window to make room" : null,
        onSelect: () => layout.toggle("agent") },
      { label: dock.agent.collapsed ? "Show details panel" : "Hide details panel", icon: "table", hint: "Ctrl+J",
        disabled: detailsAuto, title: detailsAuto ? "Enlarge the window to make room" : null,
        onSelect: () => dockLayout.toggle("agent") },
      "separator",
      { label: "Focus canvas", icon: "agents", onSelect: () => {
        layout.setCollapsed("sidebar", true);
        layout.setCollapsed("agent", true);
        dockLayout.setCollapsed("agent", true);
      } },
      { label: "Reset pane sizes", icon: "refresh", onSelect: () => { layout.reset(); dockLayout.reset(); } },
    ], { label: "Panel layout", align: "end" });
  }

  /* Running an agent is only useful if the event log is in view. */
  function showDetails(name) {
    if (dockLayout) {
      const dock = dockLayout.read();
      if (dock && dock.agent.collapsed) dockLayout.setCollapsed("agent", false, false);
    }
    if (state.bottomTabs) state.bottomTabs.select(name);
  }

  function bindCanvas() {
    const canvas = $("agents-canvas");
    canvas.addEventListener("pointerdown", beginCanvasPointer);
    canvas.addEventListener("pointerdown", beginConnection);
    canvas.addEventListener("dragover", event => {
      if (!state.doc || !state.meta || !event.dataTransfer.types.includes("application/x-mariana-node")) return;
      event.preventDefault();
      state.dragPoint = screenToWorld(event);
      frame("drop-edge", updateDropEdge);
    });
    canvas.addEventListener("dragleave", event => {
      if (event.relatedTarget && canvas.contains(event.relatedTarget)) return;
      cancelFrame("drop-edge");
      state.dragPoint = null;
      if (state.dropEdge !== null) { state.dropEdge = null; renderEdges(); }
    });
    canvas.addEventListener("drop", event => {
      if (!state.doc || !state.meta) return;
      const type = event.dataTransfer.getData("application/x-mariana-node");
      if (!type) return;
      event.preventDefault();
      cancelFrame("drop-edge");
      state.dragPoint = null;
      const point = screenToWorld(event);
      const index = state.dropEdge === null ? nearestEdge(state.doc, point, Math.max(14, 20 / state.zoom)) : state.dropEdge;
      state.dropEdge = null;
      if (index >= 0) insertNodeOnEdge(type, index, point);
      else addNode(type, point);
      renderEdges();
    });
    canvas.addEventListener("dblclick", event => {
      if (!state.doc || !state.meta) return;
      if (event.target.closest(".agent-node, .agent-edge, .agent-edge-hit, .agent-minimap, .agent-find, .agent-selection-toolbar")) return;
      event.preventDefault();
      openQuickAdd({ point: screenToWorld(event) });
    });
    canvas.addEventListener("wheel", event => {
      event.preventDefault();
      const unit = event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? 400 : 1;
      const dx = event.deltaX * unit; const dy = event.deltaY * unit;
      if (event.ctrlKey || event.metaKey) {
        /* Exponential steps keep a mouse notch decisive and a trackpad pinch
           gentle, instead of applying 10% per event to both. */
        const factor = Math.max(0.8, Math.min(1.25, Math.exp(-dy * 0.0022)));
        setZoom(state.zoom * factor, { x: event.clientX, y: event.clientY });
      } else {
        state.pan.x -= dx; state.pan.y -= dy;
        applyTransform();
      }
    }, { passive: false });
    canvas.addEventListener("contextmenu", event => {
      if (!state.doc || !state.meta) return;
      if (event.target.closest(".agent-node, .agent-edge, .agent-edge-hit, .agent-find, .agent-selection-toolbar, .agent-minimap")) return;
      event.preventDefault();
      const point = screenToWorld(event);
      Studio.overlay.menu({ x: event.clientX, y: event.clientY }, [
        { label: "Add a node…", icon: "plus", hint: "N", onSelect: () => openQuickAdd({ point }) },
        { label: "Paste here", icon: "copy", disabled: !state.clipboard, onSelect: () => pasteSelection(point) },
        "separator",
        { label: "Select all", icon: "check", hint: "Ctrl+A", onSelect: selectAllNodes },
        { label: "Fit flow", icon: "fit", hint: "1", onSelect: fitFlow },
        { label: "Auto layout", icon: "layout", onSelect: autoLayout },
        { label: "Shortcuts", icon: "info", hint: "?", onSelect: showShortcuts },
      ], { label: "Canvas actions" });
    });
    const minimap = $("agents-minimap");
    const jump = event => {
      if (!state.doc) return;
      const mini = minimap.getBoundingClientRect();
      const maxX = Math.max(1000, ...state.doc.nodes.map(node => node.x + NODE_WIDTH + 80)); const maxY = Math.max(700, ...state.doc.nodes.map(node => node.y + NODE_HEIGHT + 80));
      const scale = Math.min(mini.width / maxX, mini.height / maxY); const box = canvas.getBoundingClientRect();
      const x = (event.clientX - mini.left) / scale; const y = (event.clientY - mini.top) / scale;
      state.pan = { x: box.width / 2 - x * state.zoom, y: box.height / 2 - y * state.zoom };
      applyTransform();
    };
    minimap.addEventListener("pointerdown", event => {
      if (event.button !== 0) return;
      event.preventDefault();
      jump(event);
      const move = moveEvent => jump(moveEvent);
      document.addEventListener("pointermove", move);
      document.addEventListener("pointerup", () => document.removeEventListener("pointermove", move), { once: true });
    });
    minimap.addEventListener("keydown", event => {
      if (!state.doc) return;
      const step = event.shiftKey ? 120 : 40;
      if (event.key === "ArrowLeft") state.pan.x += step;
      else if (event.key === "ArrowRight") state.pan.x -= step;
      else if (event.key === "ArrowUp") state.pan.y += step;
      else if (event.key === "ArrowDown") state.pan.y -= step;
      else if (event.key === "Home" || event.key === "Enter") { fitFlow(); event.preventDefault(); return; }
      else return;
      event.preventDefault();
      applyTransform();
    });
  }

  async function applyDesignerAction(name) {
    if (!state.doc || !state.meta) return;
    switch (name) {
      case "save": await saveAgent(); break;
      case "run": await startRun("run"); break;
      case "test": await startRun("test"); break;
      case "debug": await startRun("debug"); break;
      case "fit": fitFlow(); break;
      case "layout": autoLayout(); break;
      case "find": openFind(); break;
      case "add": openQuickAdd(); break;
      case "shortcuts": showShortcuts(); break;
      case "templates": await showTemplates(); break;
      case "import": importAgent(); break;
      case "export": Studio.download(`${state.doc.id}.json`, JSON.stringify(state.doc, null, 2)); break;
      case "validate": scheduleValidation(); break;
      case "pan": state.panMode = !state.panMode; updatePanMode(); break;
      default: break;
    }
  }

  function registerDesignerCommands() {
    const go = action => () => Studio.views.go("agents", { action });
    const commands = [
      { label: "Agent designer: save", hint: "Designer", action: "save", keywords: "agent save designer" },
      { label: "Agent designer: run", hint: "Designer", action: "run", keywords: "agent run execute designer" },
      { label: "Agent designer: test without side effects", hint: "Designer", action: "test", keywords: "agent test dry run designer" },
      { label: "Agent designer: debug step by step", hint: "Designer", action: "debug", keywords: "agent debug breakpoint designer" },
      { label: "Agent designer: add a node", hint: "Designer", action: "add", keywords: "agent add node designer" },
      { label: "Agent designer: find a node", hint: "Designer", action: "find", keywords: "agent find search node designer" },
      { label: "Agent designer: fit flow", hint: "Designer", action: "fit", keywords: "agent fit zoom view designer" },
      { label: "Agent designer: auto layout", hint: "Designer", action: "layout", keywords: "agent arrange layout designer" },
      { label: "Agent designer: keyboard shortcuts", hint: "Designer", action: "shortcuts", keywords: "agent help shortcuts designer" },
      { label: "Agent designer: templates", hint: "Designer", action: "templates", keywords: "agent template starter designer" },
    ];
    for (const command of commands) Studio.palette.add({ ...command, glyph: "agents", when: () => Studio.views.current === "agents", action: go(command.action) });
  }

  Studio.views.register("agents", {
    title: "Agents", subtitle: "Design, test, debug and schedule visual model workflows",
    glyph: "agents", hint: "Visual agent designer and debugger",
    init() {
      for (const id of DESIGNER_CONTROLS.concat(RUN_CONTROLS)) $(id).disabled = true;
      $("agents-undo").disabled = true; $("agents-redo").disabled = true;
      state.recentTypes = Studio.store.get("agents-recent-nodes", []);
      if (!Array.isArray(state.recentTypes)) state.recentTypes = [];
      Studio.ui.render($("agents-node-palette"), Studio.ui.loading("Loading designer…"));
      Studio.ui.render($("agents-inspector"), Studio.ui.loading("Loading agent…"));
      Studio.ui.render($("agents-run-panel"), Studio.ui.loading("Loading runs…"));
      state.bottomTabs = Studio.ui.tabs(document.querySelector(".agent-bottom-tabs"), name => {
        if (name === "run") loadRunHistory(); if (name === "schedule") renderSchedule(); if (name === "spec") renderSpec();
      });
      setupLayout();
      bindCanvas();
      buildFind();
      updatePanMode();
      const search = $("agents-node-search");
      search.addEventListener("input", renderPalette);
      search.addEventListener("keydown", event => {
        if (event.key === "Enter") {
          event.preventDefault();
          const first = $("agents-node-palette").querySelector(".agent-palette-item");
          if (first) first.click();
        } else if (event.key === "Escape") { search.value = ""; renderPalette(); search.blur(); }
      });
      $("agents-select").addEventListener("change", () => openAgent($("agents-select").value));
      $("agents-new").addEventListener("click", newAgentDialog);
      $("agents-save").addEventListener("click", saveAgent);
      $("agents-provider").addEventListener("click", event => Studio.providers.menu(event.currentTarget, {
        onSaved: () => { if (state.doc && state.meta) renderInspector(); },
      }));
      $("agents-add-ability").addEventListener("click", () => abilityDialog());
      $("agents-undo").addEventListener("click", undo); $("agents-redo").addEventListener("click", redo);
      $("agents-more").addEventListener("click", moreActions);
      $("agents-pane-menu").addEventListener("click", event => openPaneMenu(event.currentTarget));
      $("agents-add-node").addEventListener("click", () => openQuickAdd());
      $("agents-find-toggle").addEventListener("click", () => { if (state.find.open) closeFind(); else openFind(); });
      $("agents-pan-mode").addEventListener("click", () => { state.panMode = !state.panMode; updatePanMode(); });
      $("agents-toggle-details").addEventListener("click", () => { if (dockLayout) dockLayout.toggle("agent"); });
      $("agents-run").addEventListener("click", () => startRun("run"));
      $("agents-run-menu").addEventListener("click", event => openRunMenu(event.currentTarget));
      $("agents-zoom-in").addEventListener("click", () => setZoom(state.zoom * 1.2)); $("agents-zoom-out").addEventListener("click", () => setZoom(state.zoom / 1.2));
      $("agents-zoom").addEventListener("click", () => setZoom(1));
      $("agents-fit").addEventListener("click", fitFlow); $("agents-layout").addEventListener("click", autoLayout);
      const editing = () => Studio.views.current === "agents" && !Studio.isTyping(document.activeElement);
      const idle = () => editing() && !document.querySelector("dialog[open], .popover");
      Studio.shortcut("mod+z", () => { if (editing()) undo(); else return false; });
      Studio.shortcut("mod+shift+z", () => { if (editing()) redo(); else return false; });
      Studio.shortcut("mod+y", () => { if (editing()) redo(); else return false; });
      Studio.shortcut("mod+c", () => { if (editing()) copySelection(); else return false; });
      Studio.shortcut("mod+v", () => { if (editing()) pasteSelection(); else return false; });
      Studio.shortcut("mod+d", () => { if (editing()) duplicateSelection(); else return false; });
      Studio.shortcut("mod+a", () => { if (editing()) selectAllNodes(); else return false; });
      Studio.shortcut("mod+s", () => { if (Studio.views.current !== "agents") return false; if (Studio.isTyping(document.activeElement)) document.activeElement.blur(); saveAgent(); });
      Studio.shortcut("mod+enter", () => {
        if (Studio.views.current !== "agents" || document.querySelector("dialog[open], .popover")) return false;
        if (Studio.isTyping(document.activeElement)) document.activeElement.blur();
        startRun("run");
      });
      Studio.shortcut("mod+f", () => { if (!idle()) return false; if (state.find.open) closeFind(); else openFind(); });
      Studio.shortcut("f2", () => { if (!idle()) return false; const nodes = selectedNodes(); if (nodes.length === 1) renameNode(nodes[0]); else Studio.toast("Select exactly one node to rename it.", "info"); });
      Studio.shortcut("n", () => { if (!idle()) return false; openQuickAdd(); });
      Studio.shortcut("h", () => { if (!idle()) return false; state.panMode = !state.panMode; updatePanMode(); });
      Studio.shortcut("0", () => { if (!idle()) return false; setZoom(1); });
      Studio.shortcut("1", () => { if (!idle()) return false; fitFlow(); });
      Studio.shortcut("?", () => { if (!idle()) return false; showShortcuts(); });
      Studio.shortcut("mod+b", () => {
        if (Studio.views.current !== "agents" || !layout) return false;
        layout.toggle("sidebar");
      });
      Studio.shortcut("mod+shift+b", () => {
        if (Studio.views.current !== "agents" || !layout) return false;
        layout.toggle("agent");
      });
      Studio.shortcut("mod+j", () => {
        if (Studio.views.current !== "agents" || !dockLayout) return false;
        dockLayout.toggle("agent");
      });
      const canvasKeyboardTarget = target => target === document.body || target === $("agents-canvas")
        || Boolean(target.closest && target.closest(".agent-node"));
      document.addEventListener("keydown", event => {
        if (Studio.views.current !== "agents" || document.querySelector("dialog[open], .popover")) return;
        if ((event.key === "Delete" || event.key === "Backspace") && idle()) { event.preventDefault(); deleteSelection(); return; }
        if (event.key === "Escape") {
          if (Studio.isTyping(event.target)) return;
          if (state.find.open) { closeFind(); $("agents-canvas").focus(); event.preventDefault(); return; }
          if (state.connection) { cancelConnection(); event.preventDefault(); return; }
          if (state.panMode) { state.panMode = false; updatePanMode(); event.preventDefault(); return; }
          const previousEdge = state.edge;
          state.selected.clear(); state.edge = null; state.connection = null;
          if (previousEdge !== null) paintEdgeSelection(previousEdge, null);
          paintSelection(); renderInspector(); renderSelectionToolbar(); renderStatus();
          return;
        }
        if (!idle() || !canvasKeyboardTarget(event.target)) return;
        if (event.key.startsWith("Arrow") && !event.ctrlKey && !event.metaKey && !event.altKey) {
          event.preventDefault();
          const step = event.shiftKey ? 1 : 20;
          nudgeSelection(event.key === "ArrowLeft" ? -step : event.key === "ArrowRight" ? step : 0,
            event.key === "ArrowUp" ? -step : event.key === "ArrowDown" ? step : 0);
        }
      });
      document.addEventListener("keydown", event => {
        if (event.key !== " " || Studio.views.current !== "agents" || Studio.isTyping(event.target)) return;
        if (event.target !== document.body && event.target !== $("agents-canvas")) return;
        if (!state.spaceHeld) { state.spaceHeld = true; updatePanMode(); }
        event.preventDefault();
      });
      document.addEventListener("keyup", event => {
        if (event.key === " " && state.spaceHeld) { state.spaceHeld = false; updatePanMode(); }
        else if (event.key.startsWith("Arrow") && state.selected.size && Studio.views.current === "agents" && !Studio.isTyping(event.target)) paintInspectorPosition();
      });
      window.addEventListener("resize", () => scheduleMinimap(true));
      Studio.on("theme", () => { minimapPalette = null; scheduleMinimap(true); });
      window.addEventListener("blur", () => {
        if (state.spaceHeld) { state.spaceHeld = false; updatePanMode(); }
        const drag = state.drag;
        cancelFrame("drag"); cancelFrame("selection"); cancelFrame("connection");
        state.drag = null;
        if (drag && drag.box) drag.box.remove();
        for (const element of state.nodeElements.values()) element.classList.remove("dragging", "connect-source", "connect-target");
        $("agents-canvas").classList.remove("panning");
        state.guides = []; renderGuides();
        cancelConnection();
        if (drag && drag.kind === "nodes" && drag.moved) { markDirty(); scheduleValidation(); refreshInspectorAfterDrag(); }
        if (state.nodeRenderPending) { state.nodeRenderPending = false; renderNodes(); }
      });
      window.addEventListener("beforeunload", event => { if (state.dirty) { event.preventDefault(); event.returnValue = ""; } });
      Studio.on("models", () => { if (state.doc && state.meta) renderInspector(); });
    },
    deactivate() {
      state.renaming = null; state.panMode = false; state.spaceHeld = false; state.lastClick = null;
      state.drag = null; state.guides = []; state.nodeRenderPending = false;
      cancelAllFrames(); cancelRunEvents(); cancelConnection(); cancelMinimap();
      for (const element of state.nodeElements.values()) element.classList.remove("dragging", "connect-source", "connect-target");
      renderGuides();
      $("agents-canvas").classList.remove("panning");
      updatePanMode();
    },
    async activate(parameters) {
      if (layout) layout.apply();
      if (dockLayout) dockLayout.apply();
      if (!state.meta) {
        try {
          const [meta] = await Promise.all([api("/api/agents/protocol"), Studio.models.ensure(), refreshAgents()]);
          state.meta = meta; renderPalette();
        } catch (error) {
          Studio.ui.render($("agents-node-palette"), Studio.ui.empty({ icon: "alert", title: "Could not load the designer", text: error.message,
            actions: [Studio.ui.button("Retry", { kind: "primary", onclick: () => Studio.views.go("agents") })] }));
          Studio.ui.render($("agents-inspector"), Studio.ui.empty({ compact: true, text: "The designer is unavailable until the protocol loads." }));
          Studio.toast(error.message, "error");
          return;
        }
      } else await refreshAgents();
      if (parameters.new === "1") { loadDocument(blankAgent()); return; }
      if (parameters.id && (!state.doc || parameters.id !== state.doc.id)) { await openAgent(parameters.id); return; }
      if (!state.doc) {
        if (state.agents.length) {
          const data = await api(`/api/agents/${encodeURIComponent(state.agents[0].id)}`);
          loadDocument(data.agent); if (data._meta.active_run) watchRun(data._meta.active_run, true);
        } else loadDocument(blankAgent());
      } else { renderAll(); renderRun(); renderSchedule(); renderSpec(); }
      if (parameters.action) {
        const action = parameters.action;
        window.history.replaceState(null, "", "#/agents");
        await applyDesignerAction(action);
      }
    },
  });

  Studio.agents = { parseRunInput, findNodes, rankNodeTypes, alignPositions, distributePositions, nearestEdge, edgeMidpoint, worldPoint, zoomedPan, runLiveBlocks, nodeLivePreviews };
  if (Studio.palette && Studio.palette.add) registerDesignerCommands();
})();
