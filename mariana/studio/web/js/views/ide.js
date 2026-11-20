"use strict";

/* Studio IDE: several loaded projects, a virtualised explorer, a cached editor
   per tab and a streaming coding agent per conversation.  Agent runs keep going
   when the person switches project, so long tasks never block reading code. */
(function () {
  const Studio = window.Studio;
  const { $, h, api, format, icon } = Studio;

  const ROW_HEIGHT = 28;
  const PREVIEW_LINES = 4000;
  const MAX_SAFE_EDIT_BYTES = 1_500_000;
  const MAX_RENDERED_ENTRIES = 300;
  const BUDGET_RING = 2 * Math.PI * 9;
  const MODE_LABELS = { ask: "Ask", edit: "Edit", auto: "Auto" };
  const PROMPTS = [
    ["Explain the open file", "Explain what this file does, its main entry points and any risks. Do not change anything.", "ask"],
    ["Find and fix a bug", "Find the most likely bug in the open file, fix it with the smallest edit, and verify with diagnostics.", "edit"],
    ["Write tests", "Add focused tests for the open file, run them when shell access is available, and report the result.", "edit"],
    ["Review the changes", "Review the current diff for correctness and regressions. Do not change files unless asked.", "ask"],
    ["Run an agent swarm", "Use agent.swarm with strategy peer_review and one feedback round to split this task across independent agents, let them review each other, and implement the agreed result:\n\n", "edit"],
    ["Add a feature", "Implement the feature described below, keeping the existing style and adding tests.\n\n", "edit"],
  ];

  let layout = null;

  const state = {
    info: null,
    loaded: false,
    mode: Studio.store.get("ide-mode", "edit"),
    model: Studio.store.get("ide-model", ""),
    sidebarOpen: Studio.store.get("ide-sidebar", true),
    voiceAuto: Boolean(Studio.store.get("ide-voice-auto", false)),
    voiceStatus: null,
    projects: [],
    hiddenProjects: new Set(),
    active: "",
    data: new Map(),
    viewActive: false,
    overviewToken: 0,
    treeFrame: null,
  };

  /* -- project state -------------------------------------------------------- */
  function ensureProject(id) {
    if (!id) return null;
    let project = state.data.get(id);
    if (project) return project;
    const info = (state.info && state.info.workspaces || []).find(item => item.id === id);
    project = {
      id,
      name: (info && info.name) || id.replace(/-/g, " "),
      tree: null,
      treeRoot: null,
      treeLoaded: false,
      treeToken: 0,
      expanded: new Set(Studio.store.get(`ide-expanded-${id}`, []) || []),
      loadingDirs: new Set(),
      rows: [],
      treeDom: null,
      paintStart: -1,
      paintEnd: -1,
      scrollTop: 0,
      filter: "",
      filterRows: null,
      filterLoading: false,
      filterTimer: null,
      filterToken: 0,
      planCollapsed: false,
      open: [],
      activePath: "",
      session: null,
      sessionId: "",
      sessionToken: 0,
      sessions: [],
      sessionsLoaded: false,
      sessionSearch: "",
      plan: [],
      running: false,
      runId: "",
      lastSeq: 0,
      feed: null,
      pendingTools: new Map(),
      swarmTimers: new Map(),
      live: null,
      liveText: "",
      lastPrompt: "",
      draft: "",
      statusText: "",
      model: Studio.store.get(`ide-model-${id}`, ""),
      context: { include: [], exclude: [] },
      settings: { excluded: [], context: { include: [], exclude: [] } },
      preview: null,
      previewToken: 0,
      previewTimer: null,
      contextTimer: null,
      loaded: false,
      loading: false,
      mention: null,
      treeTimer: null,
      voice: null,
      voiceState: "idle",
      voiceToken: 0,
      budgetLocked: false,
      usage: null,
      spokenText: "",
    };
    state.data.set(id, project);
    return project;
  }

  function activeProject() {
    return state.data.get(state.active) || null;
  }

  function isActive(project) {
    return Boolean(project && project.id === state.active);
  }

  function projectName(id) {
    const info = (state.info && state.info.workspaces || []).find(item => item.id === id);
    return (info && info.name) || id.replace(/-/g, " ");
  }

  function rememberProjects() {
    Studio.store.set("ide-open-projects", state.projects);
    Studio.store.set("ide-active-project", state.active);
  }

  function rememberProject(project) {
    if (!project || !state.data.has(project.id)) return;  // detached by Clean session
    Studio.store.set(`ide-tabs-${project.id}`, project.open.map(tab => tab.path).slice(0, 40));
    Studio.store.set(`ide-active-tab-${project.id}`, project.activePath || "");
    Studio.store.set(`ide-session-${project.id}`, (project.session && project.session.id) || "");
    Studio.store.set(`ide-expanded-${project.id}`, Array.from(project.expanded).slice(0, 400));
    rememberProjects();
  }

  function fileUrl(id, path, query) {
    return `/api/ide/workspaces/${encodeURIComponent(id)}/file${Studio.query({ path, ...(query || {}) })}`;
  }

  function previewUrl(id, path, query) {
    return `/api/ide/workspaces/${encodeURIComponent(id)}/preview${Studio.query({ path, ...(query || {}) })}`;
  }

  function treeUrl(id, path, depth) {
    return `/api/ide/workspaces/${encodeURIComponent(id)}/tree${Studio.query({ path, depth })}`;
  }

  function sessionUrl(id) {
    return `/api/ide/sessions/${encodeURIComponent(id)}`;
  }

  /* -- status and chrome ---------------------------------------------------- */
  function updateModeBadge() {
    $("ide-status-mode").textContent = MODE_LABELS[state.mode] || "Edit";
    $("ide-status-mode").className = `badge${state.mode === "auto" ? " running" : state.mode === "ask" ? " paused" : ""}`;
    const hint = {
      ask: "Ask mode reads and diagnoses only. Switch to Edit to let the agent change files.",
      edit: "Edit mode snapshots every change; use Revert edits to undo the session.",
      auto: state.info && state.info.shell_enabled
        ? "Auto mode can edit, delete and run shell commands. Every change is snapshotted."
        : "Auto mode can edit and delete. Shell stays off until MARIANA_AGENT_ALLOW_SHELL=1.",
    }[state.mode];
    $("ide-message").placeholder = hint;
  }

  function updateBusy(project) {
    if (!isActive(project)) return;
    $("ide-send").disabled = !project || project.running || !project.id;
    $("ide-stop").classList.toggle("hidden", !project || !project.running);
    $("ide-run-badge").classList.toggle("hidden", !project || !project.running);
    $("ide-agent-status").textContent = project && project.statusText
      ? project.statusText
      : project && project.running
        ? "The agent is working. You can switch projects; this run keeps going."
        : "Ask for a change; the agent reads, edits and verifies.";
  }

  function updateWorkspaceState() {
    const project = activeProject();
    const tab = project ? project.open.find(item => item.path === project.activePath) : null;
    $("ide-send").disabled = !project || project.running;
    $("ide-save").disabled = !tab || tab.kind !== "text" || tab.readonly || tab.partial;
    $("ide-refresh").disabled = !project;
    $("ide-check").disabled = !project;
    $("ide-revert").disabled = !project || project.running || !hasRevertableChanges(project);
    $("ide-status-file").textContent = tab
      ? (tab.kind === "binary" ? `${tab.path} · ${tab.mime || "binary"}` : tab.path)
      : "No file open";
    $("ide-voice").disabled = !project;
    $("ide-budget-open").disabled = !project;
    updateUsageBadge(project);
    updateBudgetMeter(project);
    updateVoiceControls(project);
  }

  function updateUsageBadge(project) {
    const badge = $("ide-status-tokens");
    const usage = project && project.usage;
    const prompt = Number(usage && usage.prompt_tokens) || 0;
    const completion = Number(usage && usage.completion_tokens) || 0;
    const total = prompt + completion;
    badge.classList.toggle("hidden", !total);
    if (!total) return;
    badge.textContent = `${format.compact(total)} tokens`;
    badge.title = `${format.number(prompt)} prompt + ${format.number(completion)} completion tokens (last run)`;
  }

  /* Re-estimate the draft locally so typing updates the meter without a request. */
  function liveBudget(project, draft) {
    const budget = project && project.preview && project.preview.budget;
    if (!budget) return null;
    const unit = budget.unit === "bytes" ? "bytes" : "tokens";
    if (project.budgetLocked) {
      const window = Number(budget.context_window) || 0;
      const used = Number(budget.used_tokens) || 0;
      return { budget, unit, used, window, percent: window ? used * 100 / window : 0 };
    }
    const value = String(draft === undefined ? ($("ide-message").value || "") : draft);
    const draftBase = budget.draft && Number(budget.draft.tokens) || 0;
    const taskBase = budget.task && Number(budget.task.tokens) || 0;
    const liveDraft = Studio.tokens.estimate(value, unit) + 4;
    const liveTask = Studio.tokens.estimate(`Current task:\n${value.slice(0, 3000)}`, unit);
    const used = Math.max(0, (Number(budget.used_tokens) || 0) - draftBase - taskBase
      + liveDraft + liveTask);
    const window = Number(budget.context_window) || 0;
    return {
      budget, unit, used, window,
      percent: window ? used * 100 / window : 0,
    };
  }

  function updateBudgetMeter(project, draft) {
    const track = $("ide-budget-track");
    const fill = $("ide-budget-fill");
    const button = $("ide-budget-open");
    const text = $("ide-budget-text");
    const row = track ? track.closest(".ide-budget") : null;
    const live = project ? liveBudget(project, draft) : null;
    if (fill) fill.setAttribute("stroke-dasharray", String(BUDGET_RING));
    if (!live) {
      if (row) row.classList.remove("warning", "danger");
      if (fill) fill.setAttribute("stroke-dashoffset", String(BUDGET_RING));
      if (track) {
        track.setAttribute("aria-valuenow", "0");
        track.setAttribute("aria-valuetext", project ? "Context estimate unavailable" : "No project open");
      }
      if (text) text.textContent = project ? "Estimating…" : "Open a project";
      if (button) button.setAttribute("aria-label", "Open context and token analysis");
      return;
    }
    const percent = Math.max(0, live.percent);
    const capped = Math.min(100, percent);
    if (row) {
      row.classList.toggle("warning", percent >= 80 && percent <= 100);
      row.classList.toggle("danger", percent > 100);
    }
    if (fill) fill.setAttribute("stroke-dashoffset", String(BUDGET_RING * (1 - capped / 100)));
    if (track) {
      track.setAttribute("aria-valuenow", String(Math.round(capped)));
      track.setAttribute("aria-valuetext",
        `${format.number(Math.round(live.used))} of ${format.number(live.window)} ${live.unit} used`);
    }
    if (text) {
      text.textContent = `${format.compact(live.used)} / ${format.compact(live.window)} ${live.unit} · ${format.percent(percent)}`;
    }
    if (button) {
      const modelName = live.budget.model && (live.budget.model.name || live.budget.model.id);
      button.setAttribute("aria-label", `Open context and token analysis${modelName ? ` for ${modelName}` : ""}: ${format.number(Math.round(live.used))} of ${format.number(live.window)} ${live.unit} used`);
      const warning = (live.budget.warnings || []).find(item => item.tone === "danger")
        || (live.budget.warnings || []).find(item => item.tone === "warning");
      button.title = warning ? warning.text
        : `Context for ${modelName || "the selected model"} · ${format.number(live.window)} ${live.unit} window`;
    }
  }

  function voiceIdleTitle() {
    const voiceStatus = state.voiceStatus;
    if (!voiceStatus) return "Dictate a task";
    if (voiceStatus.available) {
      return `Dictate with local speech recognition (${voiceStatus.label}); audio stays on this machine`;
    }
    if (Studio.voice && Studio.voice.browserSupported()) {
      return "Dictate with browser speech recognition; your browser may use a remote service";
    }
    const install = voiceStatus && voiceStatus.install ? ` ${voiceStatus.install}` : "";
    return `Voice input needs a local Whisper backend or browser support.${install}`;
  }

  function updateVoiceButton(project, voiceState) {
    const button = $("ide-voice");
    if (!button) return;
    const active = ["recording", "listening", "transcribing"].includes(voiceState);
    button.classList.toggle("recording", voiceState === "recording" || voiceState === "listening");
    button.classList.toggle("transcribing", voiceState === "transcribing");
    button.setAttribute("aria-pressed", String(active));
    button.title = voiceState === "transcribing" ? "Transcribing voice input…"
      : active ? "Stop dictation" : voiceIdleTitle();
    button.setAttribute("aria-label", button.title);
  }

  function updateVoiceControls(project) {
    const auto = $("ide-voice-auto");
    auto.disabled = !project;
    if (!project || !isActive(project)) {
      if (!project) updateVoiceButton(null, "idle");
      return;
    }
    updateVoiceButton(project, project.voiceState || "idle");
    auto.setAttribute("aria-pressed", String(state.voiceAuto));
    auto.title = state.voiceAuto ? "Stop reading replies aloud automatically"
      : "Read replies aloud automatically";
    auto.setAttribute("aria-label", auto.title);
  }

  function stopDictation(project, cancel = false) {
    const handle = project && project.voice;
    if (!handle) return;
    if (cancel) {
      project.voiceToken += 1;
      project.voice = null;
      project.voiceState = "idle";
      try { handle.cancel(); } catch (error) { /* already stopped */ }
      if (isActive(project)) {
        updateVoiceButton(project, "idle");
        updateVoiceControls(project);
      }
      return;
    }
    try { handle.stop(); } catch (error) { /* already stopped */ }
  }

  async function toggleDictation(project) {
    if (!project) {
      Studio.toast("Open a project first.", "info");
      return;
    }
    if (!Studio.voice) {
      Studio.toast("Voice support did not load; reload Studio and try again.", "error");
      return;
    }
    if (project.voice) {
      if (project.voiceState === "transcribing") stopDictation(project, true);
      else stopDictation(project, false);
      return;
    }
    if (project.voiceState === "starting") {
      project.voiceToken += 1;
      project.voiceState = "idle";
      if (isActive(project)) updateVoiceControls(project);
      return;
    }
    const token = ++project.voiceToken;
    project.voiceState = "starting";
    if (isActive(project)) updateVoiceControls(project);
    let failed = false;
    const input = $("ide-message");
    const insert = text => {
      if (token !== project.voiceToken || !isActive(project)) return;
      const caret = Number.isFinite(input.selectionStart) ? input.selectionStart : input.value.length;
      const before = input.value.slice(0, caret);
      const space = before && !/\s$/.test(before) ? " " : "";
      input.setRangeText(`${space}${text}`, caret, caret, "end");
      input.focus();
      project.draft = input.value;
      onComposerInput();
      updateBudgetMeter(project, input.value);
    };
    let active = null;
    try {
      active = await Studio.voice.dictate({
        language: document.documentElement.lang || navigator.language || "en-US",
        onState(voiceState) {
          if (token !== project.voiceToken) return;
          project.voiceState = voiceState;
          if (isActive(project)) updateVoiceButton(project, voiceState);
          if (voiceState === "transcribing") setStatus(project, "Transcribing voice input…");
          if (voiceState === "idle") {
            project.voice = null;
            project.voiceState = "idle";
            if (isActive(project)) {
              updateVoiceControls(project);
              updateBudgetMeter(project);
            }
            if (!failed) setStatus(project, "Voice input ready. Review the text, then send.");
          }
        },
        onText: text => insert(text),
        onInterim(text) {
          if (token === project.voiceToken && isActive(project)) {
            setStatus(project, text ? `Hearing: ${text}` : "Listening…");
          }
        },
        onError(error) {
          if (token !== project.voiceToken) return;
          failed = true;
          project.voice = null;
          project.voiceState = "idle";
          if (isActive(project)) updateVoiceControls(project);
          toastFor(project, error.message, "error");
        },
      });
      if (token !== project.voiceToken) {
        active.cancel();
        return;
      }
      project.voice = active;
      if (isActive(project)) updateVoiceControls(project);
      setStatus(project, active.mode === "local"
        ? "Recording locally. Click the microphone again to stop and transcribe."
        : "Listening with browser speech recognition. Click the microphone again to stop.");
    } catch (error) {
      project.voice = null;
      project.voiceState = "idle";
      if (isActive(project)) updateVoiceControls(project);
      toastFor(project, error.message, "error");
    }
  }

  function hasRevertableChanges(project) {
    const changes = project && project.session && project.session.changes;
    return Boolean(changes) && Object.values(changes).some(change => !change.reverted && change.tracked !== false);
  }

  function setStatus(project, text) {
    if (!project) return;
    if (text) project.statusText = text;
    if (isActive(project) && text) $("ide-agent-status").textContent = text;
  }

  function toastFor(project, message, kind) {
    const prefix = isActive(project) ? "" : `${projectName(project.id)}: `;
    Studio.toast(`${prefix}${message}`, kind);
  }

  /* -- models and providers ------------------------------------------------- */
  function fillModelSelect() {
    const select = $("ide-model");
    if (!select) return;
    Studio.models.ensure().then(() => {
      Studio.ui.fillModels(select, {
        placeholder: "Default chat model",
        filter: entry => entry.capabilities && entry.capabilities.chat
          && entry.kind !== "mariana-ncm-tokenizer",
        value: state.model,
      });
      state.model = select.value;
      const project = activeProject();
      if (project) {
        project.model = select.value;
        Studio.store.set(`ide-model-${project.id}`, select.value);
      }
    }).catch(() => {
      Studio.ui.render(select, h("option", { value: "", text: "No model catalog available" }));
    });
  }

  /* -- overview, project tabs and lifecycle -------------------------------- */
  async function loadOverview(options = {}) {
    const token = ++state.overviewToken;
    const data = await api("/api/ide");
    if (token !== state.overviewToken) return;
    state.info = data;
    state.loaded = true;
    $("ide-status-shell").textContent = data.shell_enabled ? "Shell on" : "Shell off";
    $("ide-status-shell").className = `badge${data.shell_enabled ? " running" : ""}`;
    updateModeBadge();
    fillModelSelect();
    if (options.restore === false) {
      // Refresh the workspace list without reopening a project.
      renderProjects();
      return data;
    }
    const available = new Set(data.workspaces
      .filter(item => !item.missing)
      .map(item => item.id));
    const stored = Studio.store.get("ide-open-projects", []);
    const wanted = (Array.isArray(stored) ? stored : []).filter(id => available.has(id));
    if (!wanted.length) {
      const saved = Studio.store.get("ide-workspace", "");
      const first = available.has(saved)
        ? saved
        : (data.workspaces.find(item => !item.missing) || {}).id;
      if (first) wanted.push(first);
    }
    state.projects = wanted;
    state.active = "";
    const previous = Studio.store.get("ide-active-project", "");
    const target = wanted.includes(previous) ? previous : wanted[0];
    if (target) await activateProject(target);
    else {
      renderProjects();
      renderAll();
    }
  }

  function visibleWorkspaces() {
    return ((state.info && state.info.workspaces) || [])
      .filter(item => !state.hiddenProjects.has(item.id) && !item.missing);
  }

  function selectOptions() {
    const select = $("ide-workspace");
    const previous = state.active;
    const values = [["", "Open project…"], ...visibleWorkspaces().map(item => [item.id, item.name])];
    if ((state.info && state.info.workspaces || []).length) values.push(["__manage__", "Manage recent projects…"]);
    Studio.ui.selectOptions(select, values, previous);
    select.value = previous || "";
  }

  function saveHiddenProjects() {
    Studio.store.set("ide-hidden-workspaces", Array.from(state.hiddenProjects));
  }

  /* The manager is the recent-session view: open projects and hidden entries
     come first, while the full on-disk workspace list stays collapsed. */
  function openProjectManager() {
    const view = Studio.overlay.modal({
      key: "ide-projects",
      title: "Recent projects",
      subtitle: "Projects open in this session, plus hidden entries. Files and chats stay on disk.",
      wide: true,
    });
    if (view.reused) return;
    const list = h("div", { class: "ide-context-list" });
    let diskOpen = false;

    const itemRow = (item, mode) => {
      const id = item.id;
      const hidden = state.hiddenProjects.has(id);
      const open = state.projects.includes(id);
      const missing = Boolean(item.missing);
      const toggle = hidden || mode === "disk"
        ? Studio.ui.button(hidden ? "Show" : "Hide", { small: true, onclick: () => {
          if (hidden) state.hiddenProjects.delete(id);
          else state.hiddenProjects.add(id);
          saveHiddenProjects();
          selectOptions();
          paint();
        } })
        : null;
      return h("div", { class: `ide-context-row${hidden ? " muted" : ""}` },
        h("div", { class: "grow" },
          h("div", { class: "row between" },
            h("strong", { text: item.name || projectName(id) }),
            h("div", { class: "row" },
              item.external ? Studio.ui.badge("in place", "accent") : null,
              missing ? Studio.ui.badge("missing", "warning") : null,
              hidden ? Studio.ui.badge("hidden", "warning") : null,
              open && !hidden ? Studio.ui.badge("open", "success") : null)),
          h("div", { class: "muted tiny", text: [
            `${item.files || 0} file${item.files === 1 ? "" : "s"}`,
            item.bytes ? format.bytes(item.bytes) : "",
            item.updated ? `updated ${format.ago(item.updated)}` : "",
          ].filter(Boolean).join(" · ") }),
          item.external
            ? h("div", { class: "muted tiny truncate", title: item.root || "",
              text: missing ? "Folder is missing; remove it and open it again" : (item.root || "") })
            : null),
        h("div", { class: "row" },
          Studio.ui.button("Open", { small: true, kind: "primary", disabled: missing, onclick: () => {
            view.close();
            if (open) activateProject(id);
            else {
              state.hiddenProjects.delete(id);
              saveHiddenProjects();
              openProject(id);
            }
          } }),
          toggle,
          item.external
            ? Studio.ui.button("Remove", { small: true, kind: "danger",
              onclick: () => Studio.run(async () => {
                if (await removeWorkspace(id, true)) paint();
              }) })
            : Studio.ui.button("Delete…", { small: true, kind: "danger",
              onclick: () => Studio.run(async () => {
                if (await removeWorkspace(id, false)) paint();
              }) })));
    };

    const paint = () => {
      Studio.ui.render(list);
      const workspaces = (state.info && state.info.workspaces) || [];
      if (!workspaces.length) {
        list.append(h("p", { class: "help", text: "No projects yet. Create one or import a folder." }));
        return;
      }
      const byId = new Map(workspaces.map(item => [item.id, item]));
      const openItems = state.projects.map(id => byId.get(id)).filter(Boolean);
      const hiddenItems = Array.from(state.hiddenProjects)
        .filter(id => !state.projects.includes(id))
        .map(id => byId.get(id)).filter(Boolean);
      list.append(h("h3", { class: "ide-context-heading", text: "Open in this session" }));
      if (openItems.length) {
        for (const item of openItems) list.append(itemRow(item, "open"));
      } else {
        list.append(h("p", { class: "help", text: "No projects are open in this session." }));
      }
      if (hiddenItems.length) {
        list.append(h("h3", { class: "ide-context-heading", text: "Hidden from the menu" }));
        for (const item of hiddenItems) list.append(itemRow(item, "hidden"));
      }
      const rest = workspaces.filter(item => !state.projects.includes(item.id)
        && !state.hiddenProjects.has(item.id));
      if (rest.length) {
        const details = h("details", { class: "ide-context-details", open: diskOpen },
          h("summary", { class: "muted tiny", text: `All projects on disk (${rest.length})` }));
        details.addEventListener("toggle", () => { diskOpen = details.open; });
        for (const item of rest) details.append(itemRow(item, "disk"));
        list.append(details);
      }
    };

    paint();
    view.body.append(list, h("p", { class: "help", text: "Hidden projects still open from a session or when you create them again. Cleaning removes every project from this session; files and chats stay on disk." }));
    if (state.hiddenProjects.size) {
      view.footer.append(Studio.ui.button("Show all", { small: true, onclick: () => {
        state.hiddenProjects.clear();
        saveHiddenProjects();
        selectOptions();
        paint();
      } }));
    }
    view.footer.append(
      Studio.ui.button("Import a copy…", { small: true, onclick: () => {
        view.close();
        Studio.run(importWorkspace);
      } }),
      Studio.ui.button("Refresh", { small: true, onclick: () => Studio.run(async () => {
        await loadOverview({ restore: false });
        paint();
      }) }),
      Studio.ui.button("Clean session…", { small: true, kind: "danger", onclick: () => Studio.run(async () => {
        if (await cleanSession()) view.close();
      }) }));
  }

  /* Remove an in-place folder (unregister) or delete a managed copy. */
  async function removeWorkspace(workspaceId, external) {
    const project = state.data.get(workspaceId);
    if (project && project.running) {
      Studio.toast("Stop the active agent run before removing this workspace.", "info");
      return false;
    }
    const confirmed = await Studio.confirm({
      title: external ? "Remove this folder from the IDE?" : "Delete this copied workspace?",
      message: external
        ? "Its IDE conversations, runs, snapshots and unsaved drafts are removed from Studio. The folder and its files stay on disk."
        : "The copy inside agent-workspace is deleted, along with its IDE conversations, runs, snapshots and unsaved drafts. The project it came from is not touched.",
      confirmLabel: external ? "Remove folder" : "Delete copy", danger: true,
    });
    if (!confirmed) return false;
    await api(`/api/ide/workspaces/${encodeURIComponent(workspaceId)}`, { method: "DELETE" });
    if (project) {
      closeFeed(project);
      stopDictation(project, true);
      for (const tab of project.open) destroyEditor(tab);
      state.data.delete(workspaceId);
      state.projects = state.projects.filter(id => id !== workspaceId);
      if (state.active === workspaceId) {
        state.active = "";
        const next = state.projects[0];
        if (next) await activateProject(next);
        else renderAll();
      }
    }
    state.hiddenProjects.delete(workspaceId);
    saveHiddenProjects();
    for (const key of ["ide-tabs-", "ide-active-tab-", "ide-session-",
      "ide-expanded-", "ide-model-"]) {
      Studio.store.remove(`${key}${workspaceId}`);
    }
    await loadOverview({ restore: false });
    Studio.toast(external
      ? "Folder removed from the IDE. Its files stay on disk."
      : "Copied workspace deleted.", "success");
    return true;
  }

  /* Clear the local IDE session: recents, open tabs, drafts, hidden entries and
     per-project preferences.  Files, conversations and run history stay on disk. */
  async function cleanSession() {
    if (Array.from(state.data.values()).some(project => project.running)) {
      Studio.toast("Stop the active agent run before cleaning the session.", "info");
      return false;
    }
    const confirmed = await Studio.confirm({
      title: "Clean this IDE session?",
      message: "Every project leaves the recent list, with its open tabs, unsaved drafts, pinned context and per-project preferences. Files, conversations and run history stay on disk.",
      confirmLabel: "Clean session", danger: true,
    });
    if (!confirmed) return false;
    for (const project of Array.from(state.data.values())) {
      try {
        closeFeed(project);
        stopDictation(project, true);
        if (project.previewTimer) clearTimeout(project.previewTimer);
        if (project.contextTimer) clearTimeout(project.contextTimer);
        for (const tab of project.open) destroyEditor(tab);
      } catch (error) {
        console.error("Could not fully detach a project while cleaning", error);
      }
    }
    state.data.clear();
    state.projects = [];
    state.active = "";
    state.hiddenProjects.clear();
    for (const key of ["ide-open-projects", "ide-active-project", "ide-workspace",
      "ide-hidden-workspaces"]) {
      Studio.store.remove(key);
    }
    for (const prefix of ["ide-tabs-", "ide-active-tab-", "ide-session-",
      "ide-expanded-", "ide-model-"]) {
      Studio.store.removePrefix(prefix);
    }
    // Update the visible session before refreshing the list: a slow or failed
    // request must not leave stale project tabs or menus on screen.
    renderAll();
    try {
      await loadOverview({ restore: false });
    } catch (error) {
      console.error("Refreshing the workspace list failed", error);
    }
    Studio.toast("IDE session cleaned. Files and conversations stay on disk.", "success");
    return true;
  }

  function renderProjects() {
    selectOptions();
    const root = Studio.ui.render($("ide-project-tabs"));
    if (!state.projects.length) {
      root.append(h("span", { class: "ide-project-empty muted tiny", text: "Open a project from the menu to begin." }));
      return;
    }
    for (const id of state.projects) {
      const project = ensureProject(id);
      const dirty = project.open.some(tab => tab.dirty);
      const tab = h("div", {
        class: `ide-project-tab${id === state.active ? " active" : ""}`,
        role: "tab", tabIndex: id === state.active ? 0 : -1,
        "aria-selected": String(id === state.active), title: projectName(id),
        "aria-controls": "ide-body",
        onclick: () => activateProject(id),
        onkeydown: event => {
          if (event.target !== tab) return;
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            activateProject(id);
          }
        },
        oncontextmenu: event => projectMenu(event, project),
      },
      h("span", { class: "truncate", text: projectName(id) }),
      dirty ? h("span", { class: "ide-tab-dot", title: "Unsaved files", "aria-label": "Unsaved files" }) : null,
      project.running ? h("span", { class: "ide-project-running", title: "Agent running", "aria-label": "Agent running" }) : null,
      h("button", {
        class: "icon-button small ide-tab-close", type: "button",
        "aria-label": `Close ${projectName(id)}`, title: "Close project (files stay on disk)",
        onclick: event => { event.stopPropagation(); closeProject(id); },
      }, icon("x", "small")));
      root.append(tab);
    }
  }

  function projectMenu(event, project) {
    event.preventDefault();
    Studio.overlay.menu({ x: event.clientX, y: event.clientY }, [
      { label: "Close project", icon: "x", onSelect: () => closeProject(project.id) },
      { label: "Close other projects", icon: "x", onSelect: async () => {
        for (const id of [...state.projects]) if (id !== project.id) await closeProject(id);
      } },
      "separator",
      { label: "New file here", icon: "plus", onSelect: () => newFile(project, "") },
      { label: "Add provider…", icon: "globe",
        onSelect: () => Studio.providers.addEndpointDialog({ onSaved: fillModelSelect }) },
    ], { label: `Project actions for ${projectName(project.id)}` });
  }

  async function openProject(id) {
    if (!id) return;
    if (!state.projects.includes(id)) {
      state.projects.push(id);
      ensureProject(id);
      rememberProjects();
    }
    await activateProject(id);
  }

  async function activateProject(id) {
    const project = ensureProject(id);
    if (!project) return;
    const previous = state.data.get(state.active);
    if (previous && previous.id !== id) stopDictation(previous, true);
    state.active = id;
    state.model = project.model || state.model;
    Studio.store.set("ide-workspace", id);
    rememberProjects();
    renderProjects();
    renderAll(project);
    updateBusy(project);
    fillModelSelect();
    if (!project.loaded && !project.loading) await loadProject(project);
    if (project.running && !project.feed && state.viewActive) followRun(project, project.runId, project.lastSeq);
    updateWorkspaceState();
  }

  async function closeProject(id) {
    const project = state.data.get(id);
    if (!project) return;
    if (project.open.some(tab => tab.dirty)) {
      const discard = await Studio.confirm({
        title: `Close ${projectName(id)}?`,
        message: "Its open tabs include unsaved edits. The files stay on disk, but those edits are dropped.",
        confirmLabel: "Close project", danger: true,
      });
      if (!discard) return;
    }
    if (project.running) {
      const close = await Studio.confirm({
        title: "Close a project with a running agent?",
        message: "The run keeps going on the server and its conversation is kept, but its stream and diff feed stop updating here until you reopen the project.",
        confirmLabel: "Close project", cancelLabel: "Keep open", danger: true,
      });
      if (!close) return;
    }
    closeFeed(project);
    stopDictation(project, true);
    for (const tab of project.open) destroyEditor(tab);
    project.open = [];
    state.data.delete(id);
    state.projects = state.projects.filter(item => item !== id);
    rememberProjects();
    if (state.active === id) {
      state.active = "";
      const next = state.projects[0];
      if (next) await activateProject(next);
      else {
        renderProjects();
        renderAll();
      }
    } else {
      renderProjects();
    }
  }

  async function loadProject(project) {
    project.loading = true;
    const token = ++project.treeToken;
    try {
      const [tree, sessions, settings] = await Promise.all([
        api(treeUrl(project.id, "", 1)),
        api(`/api/ide/workspaces/${encodeURIComponent(project.id)}/sessions`),
        api(`/api/ide/workspaces/${encodeURIComponent(project.id)}/settings`),
      ]);
      if (token !== project.treeToken) return;
      project.tree = tree;
      project.treeRoot = tree.children || [];
      project.treeLoaded = true;
      project.sessions = sessions.sessions || [];
      project.sessionsLoaded = true;
      project.settings = settings || project.settings;
      if (!project.session && !project.context.include.length
          && project.settings.context && project.settings.context.include) {
        project.context.include = [...project.settings.context.include];
      }
      project.loaded = true;
      await restoreExpanded(project);
      if (isActive(project)) {
        renderTree(project);
        renderSessions(project);
        renderContext(project);
      }
      await restoreTabs(project);
      const sessionId = Studio.store.get(`ide-session-${project.id}`, "");
      if (sessionId && !project.running) await loadSession(project, sessionId, { quiet: true });
      if (project.running && project.runId && isActive(project) && state.viewActive) {
        followRun(project, project.runId, project.lastSeq);
      }
    } catch (error) {
      if (token !== project.treeToken) return;
      if (isActive(project)) {
        Studio.ui.render($("ide-tree"), Studio.ui.empty({
          compact: true, icon: "alert", title: "Could not load the project", text: error.message,
          actions: [Studio.ui.button("Try again", { small: true, onclick: () => loadProject(project) })],
        }));
      }
      Studio.toast(error.message, "error");
    } finally {
      project.loading = false;
    }
  }

  async function restoreExpanded(project) {
    const expanded = Array.from(project.expanded).slice(0, 80);
    const byDepth = new Map();
    for (const path of expanded) {
      const depth = path.split("/").length;
      if (!byDepth.has(depth)) byDepth.set(depth, []);
      byDepth.get(depth).push(path);
    }
    for (const depth of [...byDepth.keys()].sort((left, right) => left - right)) {
      await Promise.all(byDepth.get(depth).map(async path => {
        const node = findNode(project, path);
        if (!node || node.type !== "dir") {
          project.expanded.delete(path);
          return;
        }
        if (node.loaded) return;
        try {
          const data = await api(treeUrl(project.id, path, 1));
          const current = findNode(project, path) || node;
          current.children = data.children || [];
          current.loaded = true;
          current.has_children = Boolean(current.children.length);
        } catch (error) {
          project.expanded.delete(path);
        }
      }));
    }
  }

  async function restoreTabs(project) {
    const paths = Studio.store.get(`ide-tabs-${project.id}`, []);
    const active = Studio.store.get(`ide-active-tab-${project.id}`, "");
    if (!Array.isArray(paths) || !paths.length) return;
    await Promise.all(paths.slice(0, 20).map(async path => {
      if (project.open.some(tab => tab.path === path)) return;
      try {
        const data = await api(previewUrl(project.id, path, { start: 0, limit: PREVIEW_LINES }));
        if (project.open.some(tab => tab.path === path)) return;
        project.open.push(makeTab(data));
      } catch (error) {
        /* a tab removed on disk simply is not restored */
      }
    }));
    if (active && project.open.some(tab => tab.path === active)) project.activePath = active;
    else if (project.open.length) project.activePath = project.open[0].path;
    if (isActive(project)) {
      renderTabs(project);
      mountEditor(project);
      updateWorkspaceState();
    }
  }

  async function newWorkspace() {
    const name = await Studio.prompt({
      title: "New IDE workspace", label: "Workspace name", placeholder: "my-project",
      confirmLabel: "Create",
    });
    if (!name) return;
    const result = await Studio.run(() => api("/api/ide/workspaces", { body: { name, template: "python" } }));
    if (!result) return;
    Studio.toast(`Created ${result.workspace.name}.`, "success");
    state.loaded = false;
    await loadOverview();
    await openProject(result.workspace.id);
  }

  async function openFolder() {
    const source = await Studio.overlay.pickPath({
      title: "Open a project folder in place",
      hint: "The agent edits this folder where it lives; nothing is copied into agent-workspace.",
      start: (Studio.state && Studio.state.root) || "",
    });
    if (!source) return;
    const result = await Studio.run(
      () => api("/api/ide/workspaces/open", { body: { source } }));
    if (!result) return;
    Studio.toast(`Opened ${result.workspace.name} in place.`, "success");
    state.loaded = false;
    await loadOverview();
    await openProject(result.workspace.id);
  }

  async function importWorkspace() {
    const source = await Studio.overlay.pickPath({
      title: "Copy a project folder into the IDE",
      hint: "The copy lives in agent-workspace/ide/workspaces; the original is not modified.",
      start: (Studio.state && Studio.state.root) || "",
    });
    if (!source) return;
    const fallback = source.split(/[\\/]/).filter(Boolean).pop() || "imported-project";
    const name = await Studio.prompt({
      title: "Import project", label: "Workspace name", value: fallback,
      confirmLabel: "Import",
    });
    if (!name) return;
    const result = await Studio.run(() => api("/api/ide/workspaces", { body: { name, source } }));
    if (!result) return;
    Studio.toast(`Imported ${result.workspace.files} files.`, "success");
    state.loaded = false;
    await loadOverview();
    await openProject(result.workspace.id);
  }

  /* -- explorer: lazy folders, virtual rows --------------------------------- */
  function findNode(project, path) {
    const stack = [...(project.treeRoot || [])];
    while (stack.length) {
      const node = stack.pop();
      if (node.path === path) return node;
      if (node.type === "dir" && Array.isArray(node.children)) stack.push(...node.children);
    }
    return null;
  }

  async function loadChildren(project, node) {
    if (!node || node.type !== "dir" || node.loaded || project.loadingDirs.has(node.path)) return;
    project.loadingDirs.add(node.path);
    paintTreeWindow(project, true);
    try {
      const data = await api(treeUrl(project.id, node.path, 1));
      const current = findNode(project, node.path);
      if (current) {
        current.children = data.children || [];
        current.loaded = true;
        current.has_children = Boolean(current.children.length);
      }
    } catch (error) {
      toastFor(project, `Could not open ${node.path}: ${error.message}`, "error");
      project.expanded.delete(node.path);
    } finally {
      project.loadingDirs.delete(node.path);
      if (isActive(project)) {
        renderTreeRows(project);
        paintTreeWindow(project, true);
      }
    }
  }

  async function toggleFolder(project, node) {
    if (project.expanded.has(node.path)) project.expanded.delete(node.path);
    else {
      project.expanded.add(node.path);
      if (!node.loaded) await loadChildren(project, node);
    }
    rememberProject(project);
    if (isActive(project)) {
      renderTreeRows(project);
      paintTreeWindow(project, true);
    }
  }

  async function unloadFolder(project, node) {
    const hadChildren = node.has_children !== false;
    project.expanded.delete(node.path);
    node.children = [];
    node.loaded = false;
    node.has_children = hadChildren;
    rememberProject(project);
    if (isActive(project)) {
      renderTreeRows(project);
      paintTreeWindow(project, true);
    }
    toastFor(project, `Unloaded ${node.path} from memory.`, "success");
  }

  function computeRows(project) {
    const rows = [];
    if (project.filter.trim() && project.filterRows) {
      for (const node of project.filterRows) rows.push({ node, depth: 0, flat: true });
      return rows;
    }
    const walk = (nodes, depth) => {
      for (const node of nodes || []) {
        rows.push({ node, depth });
        if (node.type === "dir" && project.expanded.has(node.path)
            && Array.isArray(node.children) && node.children.length) {
          walk(node.children, depth + 1);
        }
      }
    };
    walk(project.treeRoot, 0);
    return rows;
  }

  function renderTreeRows(project) {
    project.rows = computeRows(project);
    project.paintStart = -1;
    project.paintEnd = -1;
  }

  function renderTree(project) {
    const root = $("ide-tree");
    if (!project) {
      Studio.ui.render(root, Studio.ui.empty({
        compact: true, icon: "folder", title: "No project open",
        text: "Open a project from the toolbar, or ask the agent to create one.",
        actions: [Studio.ui.button("New workspace", { small: true, kind: "primary", onclick: newWorkspace })],
      }));
      return;
    }
    if (!project.treeLoaded) {
      Studio.ui.render(root, Studio.ui.loading("Loading files…"));
      return;
    }
    if (project.filter.trim() && project.filterLoading) {
      Studio.ui.render(root, Studio.ui.loading("Finding files…"));
      return;
    }
    renderTreeRows(project);
    if (!project.rows.length) {
      Studio.ui.render(root, Studio.ui.empty({
        compact: true, icon: project.filter.trim() ? "search" : "file",
        title: project.filter.trim() ? "No file matches" : "This project is empty",
        text: project.filter.trim() ? "Try a shorter filter." : "Use New file, or ask the agent to create one.",
      }));
      project.treeDom = null;
      return;
    }
    root.replaceChildren();
    const spacerTop = h("div", { class: "ide-tree-spacer" });
    const windowElement = h("div", { class: "ide-tree-window" });
    const spacerBottom = h("div", { class: "ide-tree-spacer" });
    root.append(spacerTop, windowElement, spacerBottom);
    if (project.tree && project.tree.truncated) {
      root.append(h("div", { class: "ide-tree-note muted tiny", text: "The tree is truncated; use Search or the filter to find a file." }));
    }
    project.treeDom = { spacerTop, windowElement, spacerBottom };
    const restore = project.scrollTop || 0;
    paintTreeWindow(project, true);
    root.scrollTop = restore;
    project.scrollTop = root.scrollTop;
    paintTreeWindow(project, true);
  }

  function paintTreeWindow(project, force = false) {
    if (!project || !isActive(project) || !project.treeDom) return;
    const root = $("ide-tree");
    const total = project.rows.length;
    const scrollTop = Number(root.scrollTop) || 0;
    project.scrollTop = scrollTop;
    const viewport = Number(root.clientHeight) || 600;
    const overscan = 8;
    const start = Math.max(0, Math.floor(scrollTop / ROW_HEIGHT) - overscan);
    const end = Math.min(total, Math.ceil((scrollTop + viewport) / ROW_HEIGHT) + overscan);
    if (!force && start === project.paintStart && end === project.paintEnd) return;
    project.paintStart = start;
    project.paintEnd = end;
    const { spacerTop, windowElement, spacerBottom } = project.treeDom;
    spacerTop.style.height = `${start * ROW_HEIGHT}px`;
    spacerBottom.style.height = `${Math.max(0, total - end) * ROW_HEIGHT}px`;
    Studio.ui.render(windowElement);
    for (let index = start; index < end; index += 1) {
      const row = project.rows[index];
      if (row) windowElement.append(treeRow(project, row));
    }
  }

  function treeRow(project, row) {
    const { node, depth, flat } = row;
    if (node.type === "dir") {
      const expanded = !flat && project.expanded.has(node.path);
      const loading = project.loadingDirs.has(node.path);
      const element = h("button", {
        class: `ide-tree-row dir${expanded ? " expanded" : ""}`, type: "button",
        "aria-expanded": String(expanded), title: node.path,
        onclick: () => {
          if (flat) {
            project.filter = "";
            $("ide-file-filter").value = "";
            project.filterRows = null;
            revealPath(project, node.path);
          } else toggleFolder(project, node);
        },
        oncontextmenu: event => treeMenu(event, project, node),
      },
      icon(loading ? "refresh" : expanded ? "chevron-down" : "chevron-right", "small"),
      icon("folder", "small"),
      h("span", { class: "truncate", text: node.name }),
      h("span", { class: "ide-tree-size", text: node.loaded ? String((node.children || []).length) : "" }));
      Studio.setStyle(element, { "padding-left": `calc(var(--space-2) + ${depth * 14}px)` });
      return element;
    }
    const tab = project.open.find(item => item.path === node.path);
    const element = h("button", {
      class: `ide-tree-row file${project.activePath === node.path ? " active" : ""}${tab && tab.dirty ? " dirty" : ""}`,
      type: "button", title: node.path,
      onclick: () => openFile(node.path),
      oncontextmenu: event => treeMenu(event, project, node),
    },
    icon("file", "small"),
    h("span", { class: "truncate", text: flat ? node.path : node.name }),
    node.bytes !== undefined ? h("span", { class: "ide-tree-size", text: format.bytes(node.bytes) }) : null);
    Studio.setStyle(element, { "padding-left": `calc(var(--space-2) + ${depth * 14}px)` });
    return element;
  }

  function treeMenu(event, project, node) {
    event.preventDefault();
    event.stopPropagation();
    const items = [];
    if (node.type === "dir") {
      items.push(
        { heading: node.name },
        { label: "New file here…", icon: "plus", onSelect: () => newFile(project, node.path) },
        { label: "New folder here…", icon: "folder", onSelect: () => newFolder(project, node.path) },
        "separator",
        { label: "Pin to agent context", icon: "pin", onSelect: () => addContextPath(project, node.path) },
        { label: "Toggle hidden from agent", icon: "eye", onSelect: () => toggleHidden(project, node.path) },
        { label: "Unload folder", icon: "refresh", onSelect: () => unloadFolder(project, node) },
        "separator",
        { label: "Rename…", icon: "edit", onSelect: () => renameEntry(project, node) },
        { label: "Delete folder…", icon: "trash", danger: true, onSelect: () => deleteEntry(project, node) },
        { label: "Copy path", icon: "copy", onSelect: () => Studio.copy(node.path) },
      );
    } else {
      const tab = project.open.find(item => item.path === node.path);
      items.push(
        { heading: node.name },
        { label: "Open", icon: "file", onSelect: () => openFile(node.path) },
        { label: "Pin to agent context", icon: "pin", onSelect: () => addContextPath(project, node.path) },
        { label: "Toggle hidden from agent", icon: "eye", onSelect: () => toggleHidden(project, node.path) },
        "separator",
        { label: "Rename…", icon: "edit", onSelect: () => renameEntry(project, node) },
        { label: "Copy path", icon: "copy", onSelect: () => Studio.copy(node.path) },
        { label: "Download", icon: "download", onSelect: () => downloadFile(project, node.path) },
        { label: "Delete…", icon: "trash", danger: true, onSelect: () => deleteEntry(project, node) },
      );
      if (tab) items.push("separator", { label: "Close tab", icon: "x", onSelect: () => closeTab(project, node.path) });
    }
    Studio.overlay.menu({ x: event.clientX, y: event.clientY }, items, { label: `Actions for ${node.name}` });
  }

  function onFilterInput() {
    const project = activeProject();
    if (!project) return;
    project.filter = $("ide-file-filter").value;
    clearTimeout(project.filterTimer);
    const token = ++project.filterToken;
    if (!project.filter.trim()) {
      project.filterRows = null;
      project.filterLoading = false;
      renderTree(project);
      return;
    }
    project.filterLoading = true;
    renderTree(project);
    project.filterTimer = setTimeout(async () => {
      try {
        const data = await api(`/api/ide/workspaces/${encodeURIComponent(project.id)}/files`
          + Studio.query({ query: project.filter, limit: 300 }));
        if (token !== project.filterToken) return;
        project.filterRows = [
          ...(data.folders || []).map(item => ({ ...item, type: "dir", children: [], loaded: false })),
          ...(data.files || []).map(item => ({ ...item, type: "file" })),
        ];
      } catch (error) {
        if (token !== project.filterToken) return;
        project.filterRows = [];
        Studio.toast(error.message, "error");
      } finally {
        if (token === project.filterToken) {
          project.filterLoading = false;
          if (isActive(project)) renderTree(project);
        }
      }
    }, 220);
  }

  async function revealPath(project, path) {
    if (!project) return;
    const parts = path.split("/");
    let prefix = "";
    const ancestors = parts.slice(0, -1);
    if (parts.length && findNode(project, path) && findNode(project, path).type === "dir") ancestors.push(parts[parts.length - 1]);
    for (const part of ancestors) {
      prefix = prefix ? `${prefix}/${part}` : part;
      project.expanded.add(prefix);
      const node = findNode(project, prefix);
      if (node && node.type === "dir" && !node.loaded) await loadChildren(project, node);
    }
    rememberProject(project);
    renderTreeRows(project);
    renderTree(project);
    scrollToPath(project, path);
  }

  function scrollToPath(project, path) {
    const index = project.rows.findIndex(row => row.node.path === path);
    if (index < 0 || !isActive(project)) return;
    const root = $("ide-tree");
    root.scrollTop = Math.max(0, index * ROW_HEIGHT - root.clientHeight / 2 + ROW_HEIGHT);
    paintTreeWindow(project, true);
  }

  async function refreshTree(project, options = {}) {
    if (!project) return;
    try {
      const data = await api(treeUrl(project.id, "", 1));
      if (!isActive(project)) return;
      const loaded = new Map();
      const collect = nodes => {
        for (const node of nodes || []) {
          if (node.type === "dir" && node.loaded) loaded.set(node.path, node.children || []);
          if (node.type === "dir" && Array.isArray(node.children)) collect(node.children);
        }
      };
      if (!options.deep) collect(project.treeRoot);
      project.tree = data;
      project.treeRoot = data.children || [];
      project.treeLoaded = true;
      if (!options.deep) {
        for (const [path, children] of loaded) {
          const node = findNode(project, path);
          if (node && node.type === "dir") {
            node.children = children;
            node.loaded = true;
            node.has_children = Boolean(children.length);
          }
        }
      }
      if (options.deep) await restoreExpanded(project);
      if (isActive(project)) {
        renderTree(project);
        updateWorkspaceState();
      }
    } catch (error) {
      console.error("Tree refresh failed", error);
    }
  }

  async function refreshParentDirectory(project, path) {
    if (!project) return;
    const parts = String(path || "").split("/");
    parts.pop();
    const parent = parts.join("/");
    if (!parent) {
      await refreshTree(project);
      return;
    }
    const node = findNode(project, parent);
    if (!node || !node.loaded) return;
    try {
      const data = await api(treeUrl(project.id, parent, 1));
      const current = findNode(project, parent);
      if (!current) return;
      current.children = data.children || [];
      current.has_children = Boolean(current.children.length);
      if (isActive(project)) {
        renderTreeRows(project);
        renderTree(project);
      }
    } catch (error) {
      console.error("Folder refresh failed", error);
    }
  }

  function refreshTreeSoon(project, path) {
    clearTimeout(project.treeTimer);
    project.treeTimer = setTimeout(() => refreshParentDirectory(project, path || ""), 350);
  }

  /* -- tabs and editor ------------------------------------------------------ */
  function defaultView(data) {
    if (data.kind === "binary") return data.render === "image" || data.render === "svg" ? "render" : "hex";
    if (["markdown", "mermaid", "svg"].includes(data.render)) return "render";
    return "code";
  }

  function makeTab(data) {
    const text = data.kind !== "binary";
    return {
      path: data.path,
      content: text ? Studio.editor.encodeEol(data.content || "", "\n") : "",
      version: data.version || "",
      language: data.language,
      eol: text ? Studio.editor.detectEol(data.content || "") : "\n",
      bytes: data.bytes || 0,
      mime: data.mime || "",
      modified: data.modified || "",
      kind: text ? "text" : "binary",
      render: data.render || null,
      preview: data.data_url || data.dimensions
        ? { data_url: data.data_url || "", dimensions: data.dimensions || null } : null,
      hex: data.hex || "",
      hexBytes: data.hex_bytes || 0,
      dimensions: data.dimensions || null,
      dirty: false,
      readonly: !text || Boolean(data.truncated) || data.editable === false,
      partial: text && Boolean(data.truncated),
      tooLarge: text && data.editable === false,
      stale: false,
      totalLines: data.total_lines || 0,
      loadedLines: data.end_line || 0,
      view: defaultView(data),
      editor: null,
    };
  }

  function activeTab(project) {
    return project ? project.open.find(tab => tab.path === project.activePath) || null : null;
  }

  function renderTabs(project) {
    const root = $("ide-file-tabs");
    if (!project) {
      Studio.ui.render(root);
      return;
    }
    Studio.ui.render(root);
    if (!project.open.length) {
      root.append(h("span", { class: "ide-tabs-empty muted tiny", text: "No files open" }));
      return;
    }
    for (const tab of project.open) {
      const wrapper = h("div", {
        class: `ide-file-tab${tab.path === project.activePath ? " active" : ""}${tab.dirty ? " dirty" : ""}${tab.stale ? " stale" : ""}`,
        role: "tab", tabIndex: tab.path === project.activePath ? 0 : -1,
        "aria-selected": String(tab.path === project.activePath),
        title: `${tab.path}${tab.partial ? " (large file preview)" : ""}${tab.stale ? " (changed on disk)" : ""}`,
        "aria-controls": "ide-editor-host",
        onclick: () => activateTab(project, tab.path),
        onauxclick: event => {
          if (event.button === 1) {
            event.preventDefault();
            closeTab(project, tab.path);
          }
        },
        oncontextmenu: event => tabMenu(event, project, tab),
        onkeydown: event => {
          if (event.target !== wrapper) return;
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            activateTab(project, tab.path);
          }
        },
      },
      h("span", { class: "truncate", text: tab.path.split("/").pop() }),
      tab.dirty ? h("span", { class: "ide-tab-dot", title: "Unsaved changes", "aria-label": "Unsaved changes" }) : null,
      h("button", {
        class: "icon-button small ide-tab-close", type: "button",
        "aria-label": `Close ${tab.path}`,
        onclick: event => { event.stopPropagation(); closeTab(project, tab.path); },
      }, icon("x", "small")));
      root.append(wrapper);
    }
  }

  function tabMenu(event, project, tab) {
    event.preventDefault();
    const views = viewIds(tab).filter(id => id !== tab.view).map(id => ({
      label: `Show ${viewLabel(tab, id).toLowerCase()}`, icon: id === "hex" ? "terminal" : "eye",
      onSelect: () => setTabView(project, tab, id),
    }));
    Studio.overlay.menu({ x: event.clientX, y: event.clientY }, [
      ...views,
      { label: "File details…", icon: "info", onSelect: () => openFileInfo(project, tab) },
      "separator",
      { label: "Find in file", icon: "search", disabled: !tab.editor, onSelect: () => tab.editor && tab.editor.openFind() },
      { label: "Go to line", icon: "file", disabled: !tab.editor, onSelect: () => tab.editor && tab.editor.goToLineDialog() },
      "separator",
      { label: "Close", icon: "x", onSelect: () => closeTab(project, tab.path) },
      { label: "Close others", icon: "x", onSelect: () => closeOtherTabs(project, tab.path) },
      { label: "Close all", icon: "x", onSelect: () => closeAllTabs(project) },
      "separator",
      { label: "Copy path", icon: "copy", onSelect: () => Studio.copy(tab.path) },
      { label: "Pin to agent context", icon: "pin", onSelect: () => addContextPath(project, tab.path) },
      { label: "Reload from disk", icon: "refresh", disabled: tab.dirty, onSelect: () => reloadOpenFile(project, tab.path) },
    ], { label: `Actions for ${tab.path}` });
  }

  const VIEW_LABELS = { render: "Rendered" };

  function viewLabel(tab, id) {
    if (id === "code") return "Code";
    if (id === "hex") return "Hex";
    return {
      image: "Image", svg: "Preview", markdown: "Rendered", mermaid: "Diagram", table: "Table",
    }[tab.render] || VIEW_LABELS.render;
  }

  function viewIds(tab) {
    const ids = tab.kind === "text" ? ["code"] : [];
    if (tab.render && (tab.kind === "text" || tab.render === "image" || tab.render === "svg")) ids.push("render");
    ids.push("hex");
    return ids;
  }

  function renderViewBar(project) {
    const root = $("ide-view-bar");
    const tab = activeTab(project);
    if (!tab) {
      root.classList.add("hidden");
      Studio.clear(root);
      return;
    }
    root.classList.remove("hidden");
    Studio.ui.render(root);
    const group = h("div", { class: "ide-view-tabs", role: "group", "aria-label": "File view" });
    for (const id of viewIds(tab)) {
      group.append(h("button", {
        class: `ide-view-button${tab.view === id ? " active" : ""}`,
        type: "button", "aria-pressed": String(tab.view === id),
        onclick: () => setTabView(project, tab, id),
      }, viewLabel(tab, id)));
    }
    const details = [tab.mime || tab.language || "file", format.bytes(tab.bytes)];
    if (tab.dimensions) details.push(`${tab.dimensions.width}×${tab.dimensions.height}`);
    else if (tab.kind === "text" && tab.totalLines) details.push(`${tab.totalLines} lines`);
    root.append(group, h("span", { class: "spacer" }),
      h("button", {
        class: "ide-view-meta", type: "button", title: "File details and metadata",
        onclick: () => openFileInfo(project, tab),
      }, icon("info", "small"), h("span", { class: "truncate", text: details.filter(Boolean).join(" · ") })));
  }

  function setTabView(project, tab, view) {
    if (!tab || tab.view === view) return;
    tab.view = view;
    renderViewBar(project);
    mountEditor(project);
  }

  function parseDelimited(text, delimiter) {
    const rows = [];
    let row = [];
    let field = "";
    let quoted = false;
    const source = String(text || "");
    for (let index = 0; index < source.length; index += 1) {
      const character = source[index];
      if (quoted) {
        if (character === '"' && source[index + 1] === '"') { field += '"'; index += 1; }
        else if (character === '"') quoted = false;
        else field += character;
      } else if (character === '"') quoted = true;
      else if (character === delimiter) { row.push(field); field = ""; }
      else if (character === "\n") { row.push(field); rows.push(row); row = []; field = ""; if (rows.length >= 200) break; }
      else if (character !== "\r") field += character;
    }
    if (field || row.length) { row.push(field); rows.push(row); }
    return rows;
  }

  function tablePreview(project, tab) {
    const rows = parseDelimited(tab.content, tab.path.endsWith(".tsv") ? "\t" : ",");
    if (!rows.length) return h("p", { class: "help", text: "This table is empty." });
    const width = Math.min(40, Math.max(...rows.map(row => row.length)));
    const table = h("table", { class: "ide-csv" });
    table.append(h("thead", {}, h("tr", {}, rows[0].slice(0, width).map(cell => h("th", { text: cell })))));
    const body = h("tbody", {});
    for (const row of rows.slice(1, 201)) {
      body.append(h("tr", {}, row.slice(0, width).map(cell => h("td", { text: cell }))));
    }
    table.append(body);
    return h("div", { class: "ide-table-wrap" }, table,
      rows.length > 201 ? h("p", { class: "help", text: `Showing the first 200 rows of ${rows.length}.` }) : null);
  }

  function previewMeta(tab) {
    const rows = [
      ["Type", tab.mime || tab.language || "unknown"],
      ["Size", format.bytes(tab.bytes)],
      ["Modified", tab.modified || "unknown"],
    ];
    if (tab.dimensions) rows.push(["Dimensions", `${tab.dimensions.width} × ${tab.dimensions.height}`]);
    if (tab.kind === "text" && tab.totalLines) rows.push(["Lines", String(tab.totalLines)]);
    if (tab.kind === "text") rows.push(["Encoding", "UTF-8"]);
    if (tab.render) rows.push(["Viewer", viewLabel(tab, "render")]);
    return Studio.ui.kv(rows);
  }

  function renderPreview(project, tab) {
    const body = h("div", { class: "ide-preview-body" });
    if (tab.render === "image") {
      const dataUrl = tab.preview && tab.preview.data_url;
      body.append(dataUrl
        ? h("div", { class: "ide-image-frame" }, h("img", { src: dataUrl, alt: tab.path, loading: "lazy" }))
        : h("p", { class: "help", text: "This image is too large to preview inline. Use Hex to inspect its header." }));
    } else if (tab.render === "svg") {
      const source = !tab.partial && !tab.tooLarge && tab.content
        ? `data:image/svg+xml;charset=utf-8,${encodeURIComponent(tab.content)}`
        : tab.preview && tab.preview.data_url;
      body.append(source
        ? h("div", { class: "ide-image-frame" }, h("img", { src: source, alt: tab.path }))
        : h("p", { class: "help", text: "No SVG source available." }));
    } else if (tab.render === "markdown") {
      const rendered = h("div", { class: "markdown ide-rendered" });
      Studio.markdown.into(rendered, tab.content);
      body.append(rendered);
    } else if (tab.render === "mermaid") {
      const holder = h("div", { class: "ide-rendered" });
      const svg = Studio.diagrams ? Studio.diagrams.render(tab.content) : null;
      if (svg) holder.append(h("div", { class: "diagram" }, svg));
      else holder.append(h("p", { class: "help", text: "This diagram could not be rendered. Supported previews: flowchart/graph, sequenceDiagram and pie." }),
        syntaxPre(tab.content, "mermaid"));
      body.append(holder);
    } else if (tab.render === "table") {
      body.append(tablePreview(project, tab));
    } else {
      body.append(h("p", { class: "help", text: "No rendered view for this file; switch to Code." }));
    }
    return body;
  }

  function mountPreview(project, tab, host) {
    const preview = h("div", { class: "ide-preview" });
    if (tab.view === "hex") {
      const note = tab.hexBytes && tab.bytes > tab.hexBytes
        ? `First ${format.bytes(tab.hexBytes)} of ${format.bytes(tab.bytes)}.` : "";
      preview.append(h("pre", { class: "ide-hex", text: tab.hex || "No bytes available." }),
        note ? h("p", { class: "help", text: note }) : null,
        previewMeta(tab));
    } else {
      preview.append(renderPreview(project, tab), previewMeta(tab));
    }
    host.replaceChildren(preview);
  }

  function openFileInfo(project, tab) {
    const view = Studio.overlay.modal({ key: `ide-file-info:${tab.path}`, title: tab.path, subtitle: "File metadata", wide: false });
    if (view.reused) return;
    const rows = [
      ["Path", tab.path], ["Kind", tab.kind], ["Type", tab.mime || "unknown"],
      ["Language", tab.language || "unknown"], ["Size", format.bytes(tab.bytes)],
      ["Modified", tab.modified || "unknown"],
    ];
    if (tab.dimensions) rows.push(["Dimensions", `${tab.dimensions.width} × ${tab.dimensions.height}`]);
    if (tab.kind === "text") rows.push(["Lines", String(tab.totalLines || "unknown")]);
    if (tab.render) rows.push(["Rendered as", viewLabel(tab, "render")]);
    rows.push(["Editable", tab.readonly ? "No (read-only preview)" : "Yes"]);
    view.body.append(Studio.ui.kv(rows));
    view.footer.append(Studio.ui.button("Copy path", { small: true, onclick: () => Studio.copy(tab.path) }));
  }

  function mountEditor(project) {
    if (!isActive(project)) return;
    const host = $("ide-editor-host");
    const tab = activeTab(project);
    renderViewBar(project);
    if (!tab) {
      editorHostTab = null;
      Studio.ui.render(host, Studio.ui.empty({
        compact: true, icon: "file", title: "Open a file",
        text: "Pick a file from the explorer, press Ctrl+P, or ask the agent to create one.",
      }));
      renderNotice(project);
      updateWorkspaceState();
      return;
    }
    if (tab.kind !== "text" || tab.view !== "code") {
      mountPreview(project, tab, host);
      editorHostTab = null;
      renderNotice(project);
      updateWorkspaceState();
      return;
    }
    if (tab.editor) {
      tab.editor.attach(host);
    } else {
      const created = Studio.editor.create(host, {
        value: tab.content,
        language: tab.language || "",
        path: tab.path,
        eol: tab.eol,
        readOnly: tab.readonly,
        label: tab.path,
        onLanguage(value) { tab.language = value; },
        onChange(value) {
          tab.content = value;
          if (!tab.dirty) {
            tab.dirty = true;
            renderTabs(project);
            renderProjects();
            updateWorkspaceState();
          }
        },
        onSave: saveActive,
        onCursor(line, column) {
          if (isActive(project) && activeTab(project) === tab) {
            $("ide-status-file").textContent = `${tab.path} · Ln ${line}, Col ${column}`;
          }
        },
      });
      tab.editor = created;
      if (tab.partial) created.setReadOnly(true);
    }
    editorHostTab = tab;
    renderNotice(project);
    updateWorkspaceState();
  }

  let editorHostTab = null;

  function renderNotice(project) {
    const notice = $("ide-file-notice");
    const tab = activeTab(project);
    if (!tab || (!tab.partial && !tab.stale)) {
      notice.classList.add("hidden");
      Studio.clear(notice);
      return;
    }
    notice.classList.remove("hidden");
    Studio.ui.render(notice);
    if (tab.partial) {
      notice.append(
        icon("info", "small"),
        h("span", { class: "grow", text: tab.tooLarge
          ? "This file is larger than the editor limit; showing its first bytes read-only. Use Hex for the raw content."
          : `Large file: lines ${tab.loadedLines} of ${tab.totalLines || "?"}. Opened read-only to keep the editor responsive.` }),
        tab.tooLarge ? null : Studio.ui.button("Load full file", { small: true, onclick: () => loadFullFile(project, tab) }));
    }
    if (tab.stale) {
      notice.append(
        icon("alert", "small"),
        h("span", { class: "grow", text: "This file changed on disk or was edited by the agent; your buffer was kept." }),
        Studio.ui.button("Reload", { small: true, onclick: () => reloadOpenFile(project, tab.path) }));
    }
  }

  async function loadFullFile(project, tab) {
    if (tab.kind !== "text") return;
    if (tab.tooLarge) {
      Studio.toast("This file is larger than the editor limit; keep using the preview or open it outside Studio.", "info");
      return;
    }
    if (tab.bytes > MAX_SAFE_EDIT_BYTES) {
      const load = await Studio.confirm({
        title: "Open a very large file for editing?",
        message: `${tab.path} is ${format.bytes(tab.bytes)}. Editing files this large can make the browser sluggish; a preview is usually better.`,
        confirmLabel: "Load and edit", cancelLabel: "Keep preview",
      });
      if (!load) return;
    }
    try {
      const data = await api(fileUrl(project.id, tab.path));
      tab.content = Studio.editor.encodeEol(data.content, "\n");
      tab.version = data.version;
      tab.language = data.language;
      tab.modified = data.modified || tab.modified;
      tab.eol = Studio.editor.detectEol(data.content);
      tab.partial = false;
      tab.readonly = false;
      tab.totalLines = data.total_lines;
      tab.loadedLines = data.end_line;
      if (tab.editor) {
        tab.editor.setReadOnly(false);
        tab.editor.setValue(tab.content);
        tab.editor.setLanguage(tab.language);
        tab.editor.setEol(tab.eol);
      }
      if (isActive(project)) {
        if (tab.view !== "code") mountEditor(project);
        renderNotice(project);
        updateWorkspaceState();
      }
    } catch (error) {
      Studio.toast(error.message, "error");
    }
  }

  async function openFile(path, line) {
    const project = activeProject();
    if (!project || !project.id) return;
    const existing = project.open.find(tab => tab.path === path);
    if (existing) {
      const switched = Boolean(line && existing.kind === "text" && existing.view !== "code");
      if (switched) existing.view = "code";
      activateTab(project, path);
      if (switched) { renderViewBar(project); mountEditor(project); }
      if (line && existing.editor) existing.editor.gotoLine(line);
      return;
    }
    try {
      const data = await api(previewUrl(project.id, path, { start: 0, limit: PREVIEW_LINES }));
      if (!isActive(project)) return;
      const tab = makeTab(data);
      if (line && tab.kind === "text" && tab.view !== "code") tab.view = "code";
      project.open.push(tab);
      project.activePath = tab.path;
      rememberProject(project);
      renderTabs(project);
      revealPath(project, tab.path);
      mountEditor(project);
      if (line && tab.editor) tab.editor.gotoLine(line);
      updateWorkspaceState();
      renderProjects();
    } catch (error) {
      Studio.toast(error.message, "error");
    }
  }

  function activateTab(project, path) {
    if (!project || project.activePath === path) return;
    project.activePath = path;
    rememberProject(project);
    renderTabs(project);
    paintTreeWindow(project, true);
    mountEditor(project);
    updateWorkspaceState();
    refreshContextPreview(project);
  }

  function destroyEditor(tab) {
    if (tab.editor) {
      tab.editor.destroy();
      tab.editor = null;
    }
    if (editorHostTab === tab) editorHostTab = null;
  }

  async function closeTab(project, path, options = {}) {
    const tab = project.open.find(item => item.path === path);
    if (!tab) return;
    if (tab.dirty && !options.force) {
      const discard = await Studio.confirm({
        title: "Discard unsaved changes?",
        message: `${path} has unsaved edits. Closing the tab drops them.`,
        confirmLabel: "Discard", danger: true,
      });
      if (!discard) return;
    }
    destroyEditor(tab);
    project.open = project.open.filter(item => item.path !== path);
    if (project.activePath === path) {
      project.activePath = project.open.length ? project.open[project.open.length - 1].path : "";
    }
    rememberProject(project);
    renderTabs(project);
    paintTreeWindow(project, true);
    mountEditor(project);
    updateWorkspaceState();
    renderProjects();
  }

  async function closeOtherTabs(project, keep) {
    for (const tab of [...project.open]) if (tab.path !== keep) await closeTab(project, tab.path);
  }

  async function closeAllTabs(project) {
    for (const tab of [...project.open]) await closeTab(project, tab.path);
  }

  async function reloadOpenFile(project, path) {
    try {
      const data = await api(previewUrl(project.id, path, { start: 0, limit: PREVIEW_LINES }));
      const tab = project.open.find(item => item.path === path);
      if (!tab) return;
      tab.bytes = data.bytes || 0;
      tab.mime = data.mime || tab.mime;
      tab.modified = data.modified || tab.modified;
      tab.hex = data.hex || "";
      tab.hexBytes = data.hex_bytes || 0;
      tab.dimensions = data.dimensions || null;
      tab.preview = data.data_url || data.dimensions
        ? { data_url: data.data_url || "", dimensions: data.dimensions || null } : null;
      tab.language = data.language || tab.language;
      tab.render = data.render || tab.render;
      if (tab.kind === "text") {
        tab.content = Studio.editor.encodeEol(data.content || "", "\n");
        tab.version = data.version || "";
        tab.eol = Studio.editor.detectEol(data.content || "");
        tab.partial = Boolean(data.truncated);
        tab.readonly = Boolean(data.truncated) || data.editable === false;
        tab.tooLarge = data.editable === false;
        tab.totalLines = data.total_lines || 0;
        tab.loadedLines = data.end_line || 0;
        if (tab.editor) {
          tab.editor.setReadOnly(tab.readonly);
          tab.editor.setValue(tab.content, { keepScroll: true });
          tab.editor.setLanguage(tab.language);
          tab.editor.setEol(tab.eol);
        }
      }
      tab.dirty = false;
      tab.stale = false;
      if (isActive(project)) {
        if (tab.kind !== "text" || tab.view !== "code" || !tab.editor) mountEditor(project);
        renderTabs(project);
        renderViewBar(project);
        renderNotice(project);
        updateWorkspaceState();
        if (path === project.activePath) refreshContextPreview(project);
      }
    } catch (error) {
      if (error.status === 404) {
        closeTab(project, path, { force: true });
        return;
      }
      Studio.toast(error.message, "error");
    }
  }

  async function saveTab(project, tab) {
    if (!project.id || tab.kind !== "text" || tab.readonly || tab.partial) return null;
    const result = await api(`/api/ide/workspaces/${encodeURIComponent(project.id)}/file`, {
      method: "PUT",
      body: {
        path: tab.path,
        content: Studio.editor.encodeEol(tab.content, tab.eol || "\n"),
        version: tab.version,
      },
    });
    tab.version = result.version;
    tab.bytes = result.bytes;
    tab.modified = result.modified || tab.modified;
    tab.dirty = false;
    tab.stale = false;
    if (isActive(project) && project.activePath === tab.path) {
      renderViewBar(project);
      refreshContextPreview(project);
    }
    return result;
  }

  async function saveAllDirty(project) {
    const dirty = project.open.filter(tab => tab.dirty && !tab.readonly && !tab.partial);
    for (const tab of dirty) await saveTab(project, tab);
    if (isActive(project)) {
      renderTabs(project);
      renderProjects();
      updateWorkspaceState();
    }
    return dirty.length;
  }

  async function saveActive() {
    const project = activeProject();
    const tab = project && activeTab(project);
    if (!project || !tab || tab.readonly || tab.partial || $("ide-save").disabled) return;
    $("ide-save").disabled = true;
    try {
      await saveTab(project, tab);
      renderTabs(project);
      renderProjects();
      refreshParentDirectory(project, tab.path);
      Studio.toast(`Saved ${tab.path}`, "success");
    } catch (error) {
      if (error.status === 409) {
        const reload = await Studio.confirm({
          title: "File changed on disk", message: error.message,
          confirmLabel: "Reload from disk", cancelLabel: "Keep editing",
        });
        if (reload) await reloadOpenFile(project, tab.path);
      } else {
        Studio.toast(error.message, "error");
      }
    } finally {
      updateWorkspaceState();
    }
  }

  async function newFile(project, parent) {
    if (!project || !project.id) return;
    const prefix = parent ? `${parent}/` : "";
    const path = await Studio.prompt({
      title: parent ? `New file in ${parent}` : "New file", label: "File path",
      placeholder: `${prefix}src/module.py`, value: prefix, confirmLabel: "Create",
    });
    if (!path) return;
    try {
      const result = await api(`/api/ide/workspaces/${encodeURIComponent(project.id)}/file`, {
        method: "PUT", body: { path, content: "", version: "" },
      });
      await refreshParentDirectory(project, result.path);
      await openFile(result.path);
    } catch (error) {
      Studio.toast(error.message, "error");
    }
  }

  async function newFolder(project, parent) {
    if (!project || !project.id) return;
    const path = await Studio.prompt({
      title: parent ? `New folder in ${parent}` : "New folder", label: "Folder path",
      placeholder: parent ? `${parent}/components` : "src/components", confirmLabel: "Create",
    });
    if (!path) return;
    try {
      await api(`/api/ide/workspaces/${encodeURIComponent(project.id)}/folder`, { body: { path } });
      await refreshTree(project, { deep: true });
      toastFor(project, `Created folder ${path}.`, "success");
    } catch (error) {
      Studio.toast(error.message, "error");
    }
  }

  async function renameEntry(project, node) {
    const destination = await Studio.prompt({
      title: "Rename", label: "New path", value: node.path, confirmLabel: "Rename",
    });
    if (!destination || destination === node.path) return;
    try {
      const result = await api(`/api/ide/workspaces/${encodeURIComponent(project.id)}/move`, {
        body: { from: node.path, to: destination },
      });
      remapOpenTabs(project, result.from, result.to);
      await refreshTree(project, { deep: true });
      if (isActive(project)) {
        renderTabs(project);
        mountEditor(project);
        updateWorkspaceState();
      }
      toastFor(project, `Renamed to ${result.to}.`, "success");
    } catch (error) {
      Studio.toast(error.message, "error");
    }
  }

  async function deleteEntry(project, node) {
    const folder = node.type === "dir";
    const confirmed = await Studio.confirm({
      title: folder ? `Delete ${node.path}?` : `Delete ${node.path}?`,
      message: folder
        ? "The folder and every file inside it are deleted. This is not snapshotted."
        : "The file is deleted. This is not snapshotted; use agent revert for agent-made changes.",
      confirmLabel: folder ? "Delete folder" : "Delete file", danger: true,
    });
    if (!confirmed) return;
    try {
      if (folder) {
        await api(`/api/ide/workspaces/${encodeURIComponent(project.id)}/folder`
          + Studio.query({ path: node.path }), { method: "DELETE" });
      } else {
        await api(fileUrl(project.id, node.path), { method: "DELETE" });
      }
      for (const tab of [...project.open]) {
        if (tab.path === node.path || (folder && tab.path.startsWith(`${node.path}/`))) {
          await closeTab(project, tab.path, { force: true });
        }
      }
      await refreshTree(project, { deep: true });
    } catch (error) {
      Studio.toast(error.message, "error");
    }
  }

  function remapOpenTabs(project, from, to) {
    let active = project.activePath;
    for (const tab of project.open) {
      if (tab.path === from) tab.path = to;
      else if (tab.path.startsWith(`${from}/`)) tab.path = `${to}${tab.path.slice(from.length)}`;
      if (active === tab.path || active === from) active = tab.path;
    }
    if (project.activePath === from || project.activePath.startsWith(`${from}/`)) {
      project.activePath = `${to}${project.activePath.slice(from.length)}`;
    } else project.activePath = active;
    rememberProject(project);
  }

  function downloadFile(project, path) {
    api(previewUrl(project.id, path)).then(data => {
      const name = path.split("/").pop();
      if (data.kind === "binary" || data.editable === false) {
        if (!data.data_url) {
          Studio.toast("This file is too large to download through the preview.", "info");
          return;
        }
        const link = h("a", { href: data.data_url, download: name });
        document.body.append(link);
        link.click();
        link.remove();
        return;
      }
      Studio.download(name, data.content || "", data.mime || "text/plain");
    }).catch(error => Studio.toast(error.message, "error"));
  }

  /* -- search and diagnostics ----------------------------------------------- */
  async function runSearch() {
    const project = activeProject();
    const query = $("ide-search-query").value.trim();
    const root = $("ide-search-results");
    if (!project) return;
    if (!query) {
      Studio.ui.render(root, Studio.ui.empty({ compact: true, text: "Type a search query." }));
      return;
    }
    const token = ++project.filterToken;
    Studio.ui.render(root, Studio.ui.loading("Searching…"));
    try {
      const data = await api(`/api/ide/workspaces/${encodeURIComponent(project.id)}/search`, {
        body: { query, max_results: 200 },
      });
      if (!isActive(project) || token !== project.filterToken) return;
      Studio.ui.render(root);
      if (!data.matches.length) {
        root.append(Studio.ui.empty({
          compact: true, text: `No matches in ${data.files_scanned} searched files.`,
        }));
        return;
      }
      for (const match of data.matches) {
        root.append(h("button", {
          class: "ide-search-hit", type: "button", title: `${match.path}:${match.line}`,
          onclick: () => openFile(match.path, match.line),
        },
        h("span", { class: "ide-search-path", text: `${match.path}:${match.line}` }),
        h("span", { class: "ide-search-line mono", text: match.text || "" })));
      }
      if (data.truncated) root.append(h("div", { class: "muted tiny", text: "More matches were truncated; narrow the query." }));
    } catch (error) {
      if (isActive(project)) Studio.ui.render(root, Studio.ui.empty({ compact: true, icon: "alert", text: error.message }));
    }
  }

  async function checkProject() {
    const project = activeProject();
    if (!project) return;
    const handle = Studio.overlay.modal({
      key: "ide-diagnostics", title: "Project diagnostics",
      subtitle: "Checks files on disk; save the open file first if it has unsaved edits.",
      content: Studio.ui.loading("Checking the project…"),
    });
    if (handle.reused) return;
    try {
      const result = await api(`/api/ide/workspaces/${encodeURIComponent(project.id)}/diagnostics`, { body: {} });
      const body = Studio.ui.render(handle.body);
      if (result.ok) {
        body.append(Studio.ui.callout("success",
          h("strong", { text: `No problems found in ${result.files_checked} files.` })));
        return;
      }
      body.append(h("p", { class: "muted small", text: `${result.issues.length} issue(s) across ${result.files_checked} checked files.` }));
      if (result.truncated) {
        body.append(h("p", { class: "muted small", text: "Results were truncated; check a specific folder next." }));
      }
      const list = h("div", { class: "ide-problems" });
      for (const issue of result.issues.slice(0, 200)) {
        list.append(h("button", {
          class: `ide-problem ${issue.severity === "error" ? "error" : ""}`, type: "button",
          onclick: () => { handle.close(); openFile(issue.path, issue.line); },
        },
        Studio.ui.badge(issue.severity === "error" ? "Error" : "Warning", issue.severity === "error" ? "failed" : "paused"),
        h("span", { class: "grow wrap-anywhere", text: issue.message }),
        h("span", { class: "muted tiny numeric", text: `${issue.path}:${issue.line}` })));
      }
      body.append(list);
    } catch (error) {
      Studio.ui.render(handle.body, Studio.ui.callout("danger", h("span", { text: error.message })));
    }
  }

  /* -- sessions and transcript --------------------------------------------- */
  function summaryFor(project, id) {
    return project.sessions.find(item => item.id === id) || null;
  }

  function updateSessionSummary(project, summary) {
    if (!summary) return;
    const index = project.sessions.findIndex(item => item.id === summary.id);
    if (index >= 0) project.sessions[index] = { ...project.sessions[index], ...summary };
    else project.sessions.unshift(summary);
    project.sessions.sort((left, right) => (Number(Boolean(right.pinned)) - Number(Boolean(left.pinned)))
      || ((right.updated || 0) - (left.updated || 0)));
  }

  function renderSessions(project) {
    if (!isActive(project)) return;
    const root = $("ide-sessions");
    if (!project) {
      Studio.ui.render(root);
      return;
    }
    const query = project.sessionSearch.trim().toLowerCase();
    const sessions = project.sessions.filter(session => !query
      || (session.title || "").toLowerCase().includes(query)
      || (session.preview || "").toLowerCase().includes(query));
    Studio.ui.render(root);
    if (!sessions.length) {
      root.append(Studio.ui.empty({
        compact: true,
        text: project.sessions.length ? "No conversation matches this search." : "This project has no conversations yet.",
      }));
      return;
    }
    for (const session of sessions) {
      const active = project.session && project.session.id === session.id;
      const row = h("div", {
        class: `list-row ide-session-row${active ? " active" : ""}`,
        role: "button", tabIndex: 0,
        onclick: () => loadSession(project, session.id),
        onkeydown: event => {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            loadSession(project, session.id);
          }
        },
        oncontextmenu: event => sessionMenu(event, project, session, null),
      },
      h("span", { class: "ide-session-mark" }, icon(session.pinned ? "pin" : "chat", "small")),
      h("div", { class: "list-row-copy" },
        h("strong", { class: "list-row-title", text: session.title || "Untitled session" }),
        h("span", { class: "list-row-subtitle", text: `${format.ago(session.updated)} · ${session.turns} turn(s)${session.active ? " · running" : ""}` })),
      h("button", {
        class: "icon-button small", type: "button", "aria-label": `Actions for ${session.title || "session"}`,
        onclick: event => { event.stopPropagation(); sessionMenu(event, project, session, event.currentTarget); },
      }, icon("more", "small")));
      root.append(row);
    }
  }

  function sessionMenu(event, project, session, anchor) {
    event.preventDefault();
    event.stopPropagation();
    const position = anchor ? anchor : { x: event.clientX, y: event.clientY };
    Studio.overlay.menu(position, [
      { heading: session.title || "Conversation" },
      { label: "Open", icon: "chat", disabled: project.running && (!project.session || project.session.id !== session.id),
        onSelect: () => loadSession(project, session.id) },
      { label: "Rename…", icon: "edit", onSelect: () => renameSession(project, session) },
      { label: session.pinned ? "Unpin" : "Pin to top", icon: "pin", onSelect: () => pinSession(project, session, !session.pinned) },
      { label: "Duplicate", icon: "copy", disabled: project.running, onSelect: () => duplicateSession(project, session.id) },
      { label: "Export as Markdown", icon: "download", onSelect: () => exportSession(project, session.id) },
      "separator",
      { label: "Delete conversation…", icon: "trash", danger: true, onSelect: () => deleteSession(project, session.id) },
    ], { label: "Conversation actions" });
  }

  async function loadSessions(project) {
    if (!project || !project.id) return;
    try {
      const data = await api(`/api/ide/workspaces/${encodeURIComponent(project.id)}/sessions`);
      project.sessions = data.sessions || [];
      project.sessionsLoaded = true;
      if (isActive(project)) renderSessions(project);
    } catch (error) {
      console.error("Session list failed", error);
    }
  }

  async function loadSession(project, id, options = {}) {
    if (!project || !id) return;
    if (project.running && project.session && project.session.id !== id) {
      Studio.toast("Stop the active run in this project before opening another conversation.", "info");
      return;
    }
    const token = ++project.sessionToken;
    try {
      const data = await api(sessionUrl(id));
      if (token !== project.sessionToken || !state.data.has(project.id)) return;
      const session = data.session;
      if (session.workspace_id !== project.id) {
        await openProject(session.workspace_id);
        return;
      }
      project.session = session;
      project.sessionId = session.id;
      project.plan = session.plan || [];
      project.spokenText = "";
      project.usage = session.usage && typeof session.usage === "object" ? session.usage : null;
      const storedContext = session.context && typeof session.context === "object" ? session.context : {};
      project.context = {
        include: Array.isArray(storedContext.include)
          ? [...storedContext.include]
          : [...(project.settings.context && project.settings.context.include || [])],
        exclude: Array.isArray(storedContext.exclude) ? [...storedContext.exclude] : [],
      };
      project.runId = session.active_run_id || "";
      project.running = Boolean(project.runId);
      project.lastSeq = 0;
      project.pendingTools.clear();
      project.live = null;
      project.liveText = "";
      project.statusText = project.runId ? "Resuming the live run…" : "";
      rememberProject(project);
      closeFeed(project);
      if (isActive(project)) {
        $("ide-session-title").textContent = session.title || "Session";
        renderTranscript(project);
        renderPlan(project);
        renderSessions(project);
        renderContext(project);
        updateWorkspaceState();
        updateBusy(project);
        if (project.runId) {
          setStatus(project, "Resuming the live run…");
          if (state.viewActive) followRun(project, project.runId, 0);
        }
      }
      if (!options.quiet) {
        const index = project.sessions.findIndex(item => item.id === session.id);
        const summary = summaryFor(project, session.id);
        if (index < 0 && summary === null) {
          project.sessions.unshift({
            id: session.id, title: session.title, updated: session.updated,
            turns: 0, changes: 0, active: Boolean(project.runId), pinned: Boolean(session.pinned),
            preview: "", context: project.context,
          });
          renderSessions(project);
        }
      }
    } catch (error) {
      if (token === project.sessionToken) Studio.toast(error.message, "error");
    }
  }

  function newSession(project) {
    if (!project) return;
    if (project.running) {
      Studio.toast("Stop the active run before starting a new conversation.", "info");
      return;
    }
    stopDictation(project, true);
    project.usage = null;
    project.budgetLocked = false;
    project.spokenText = "";
    project.sessionToken += 1;
    closeFeed(project);
    project.session = null;
    project.sessionId = "";
    project.plan = [];
    project.runId = "";
    project.lastSeq = 0;
    project.running = false;
    project.pendingTools.clear();
    project.live = null;
    project.liveText = "";
    project.statusText = "";
    project.context = {
      include: [...(project.settings.context && project.settings.context.include || [])],
      exclude: [],
    };
    rememberProject(project);
    if (isActive(project)) {
      $("ide-session-title").textContent = "New session";
      renderTranscript(project);
      renderPlan(project);
      renderSessions(project);
      renderContext(project);
      updateBusy(project);
      updateWorkspaceState();
      $("ide-message").focus();
    }
  }

  async function renameSession(project, session) {
    const title = await Studio.prompt({
      title: "Rename conversation", label: "Title", value: session.title || "", confirmLabel: "Rename",
    });
    if (!title) return;
    try {
      const data = await api(sessionUrl(session.id), { method: "PATCH", body: { title } });
      updateSessionSummary(project, data.session);
      if (project.session && project.session.id === session.id) {
        project.session.title = data.session.title;
        if (isActive(project)) $("ide-session-title").textContent = data.session.title;
      }
      renderSessions(project);
      rememberProject(project);
    } catch (error) {
      Studio.toast(error.message, "error");
    }
  }

  async function pinSession(project, session, pinned) {
    try {
      const data = await api(sessionUrl(session.id), { method: "PATCH", body: { pinned } });
      updateSessionSummary(project, data.session);
      if (project.session && project.session.id === session.id) project.session.pinned = pinned;
      renderSessions(project);
    } catch (error) {
      Studio.toast(error.message, "error");
    }
  }

  async function duplicateSession(project, id) {
    try {
      const data = await api(`${sessionUrl(id)}/duplicate`, { body: {} });
      updateSessionSummary(project, data.session);
      renderSessions(project);
      await loadSession(project, data.session.id);
      toastFor(project, "Conversation duplicated.", "success");
    } catch (error) {
      Studio.toast(error.message, "error");
    }
  }

  async function deleteSession(project, id) {
    if (project.running && project.session && project.session.id === id) {
      Studio.toast("Stop the active run before deleting this conversation.", "info");
      return;
    }
    const confirmed = await Studio.confirm({
      title: "Delete this conversation?",
      message: "Its transcript, run history and snapshots are removed. Files already changed on disk stay as they are.",
      confirmLabel: "Delete conversation", danger: true,
    });
    if (!confirmed) return;
    try {
      await api(sessionUrl(id), { method: "DELETE" });
      project.sessions = project.sessions.filter(item => item.id !== id);
      if (project.session && project.session.id === id) newSession(project);
      renderSessions(project);
    } catch (error) {
      Studio.toast(error.message, "error");
    }
  }

  function exportSession(project, id) {
    const session = project.session && project.session.id === id ? project.session : null;
    if (!session) {
      api(sessionUrl(id)).then(data => exportDocument(project, data.session))
        .catch(error => Studio.toast(error.message, "error"));
      return;
    }
    exportDocument(project, session);
  }

  function exportDocument(project, session) {
    const lines = [`# ${session.title || "Conversation"}`, "",
      `- Project: ${projectName(project.id)}`,
      `- Model: ${session.model || "default"}`,
      `- Mode: ${session.mode || "edit"}`,
      ""];
    for (const entry of session.transcript || []) {
      if (entry.role === "user") lines.push("## You", "", entry.text || "", "");
      else if (entry.role === "assistant") lines.push("## Mariana", "", entry.text || "", "");
      else if (entry.role === "plan") {
        lines.push("## Plan", "");
        for (const todo of entry.todos || []) lines.push(`- [${todo.status === "done" ? "x" : " "}] ${todo.text}`);
        lines.push("");
      } else if (entry.role === "tool") {
        lines.push(`### ${entry.tool}`, "", "```json", JSON.stringify(entry.input || {}, null, 2), "```", "");
      }
    }
    const name = (session.title || "mariana-session").replace(/[^\w.-]+/g, "-").slice(0, 60);
    Studio.download(`${name}.md`, lines.join("\n"), "text/markdown");
  }

  function scrollTranscript(project) {
    if (!isActive(project)) return;
    const root = $("ide-transcript");
    if (root.scrollHeight - root.scrollTop - root.clientHeight < 160) root.scrollTop = root.scrollHeight;
  }

  function transcriptEmpty(project) {
    if (!isActive(project)) return;
    const hidden = (project.settings.excluded || []).length;
    Studio.ui.render($("ide-transcript"), Studio.ui.empty({
      compact: true, icon: "sparkle", title: "Tell the agent what to build",
      text: hidden
        ? `It reads the project, makes snapshotted edits and checks diagnostics. ${hidden} hidden path(s) stay out of its context.`
        : "It reads the project, makes snapshotted edits and checks diagnostics.",
    }));
  }

  function renderTranscript(project) {
    if (!isActive(project)) return;
    const entries = (project.session && project.session.transcript) || [];
    if (!entries.length) {
      transcriptEmpty(project);
      return;
    }
    const root = Studio.ui.render($("ide-transcript"));
    const visible = entries.slice(-MAX_RENDERED_ENTRIES);
    if (visible.length < entries.length) {
      root.append(h("div", { class: "muted tiny ide-transcript-note",
        text: `Showing the last ${visible.length} of ${entries.length} entries. Export the conversation for the full log.` }));
    }
    for (const entry of visible) {
      const node = renderEntry(project, entry);
      root.append(node);
      if (entry.role === "tool" && entry.pending) {
        const pending = project.pendingTools.get(entry.call);
        if (pending) pending.card = node;
      }
    }
    root.scrollTop = root.scrollHeight;
  }

  function appendEntry(project, entry) {
    if (!project.session) project.session = { id: "", title: "", transcript: [], changes: {} };
    if (!project.session.transcript) project.session.transcript = [];
    entry.at = entry.at || new Date().toISOString();
    project.session.transcript.push(entry);
    if (!isActive(project)) return;
    const root = $("ide-transcript");
    const empty = root.querySelector(".empty");
    if (empty) root.replaceChildren();
    const node = renderEntry(project, entry);
    if (project.live && project.live.node && project.live.node.parentNode === root) {
      root.insertBefore(node, project.live.node);
    } else root.append(node);
    scrollTranscript(project);
  }

  function toolLabel(tool) {
    const schemas = (state.info && state.info.tool_schemas) || {};
    return (schemas[tool] && schemas[tool].label) || tool;
  }

  function pretty(value, limit = 12000) {
    if (value === undefined || value === null) return "";
    const text = typeof value === "string" ? value : JSON.stringify(value, null, 2);
    return text.length > limit ? `${text.slice(0, limit)}\n… truncated` : text;
  }

  function diffBlock(change) {
    if (!change) return null;
    if (!change.diff) {
      return change.tracked === false
        ? h("p", { class: "muted tiny", text: "This file was too large to snapshot." }) : null;
    }
    const pre = h("pre", { class: "ide-diff" });
    const lines = String(change.diff).split("\n").slice(0, 500);
    for (const line of lines) {
      const kind = line.startsWith("@@") ? "hunk"
        : line.startsWith("+++") || line.startsWith("---") ? "meta"
          : line.startsWith("+") ? "add" : line.startsWith("-") ? "del" : "";
      pre.append(h("span", { class: `diff-line${kind ? ` ${kind}` : ""}`, text: line || " " }));
    }
    return h("details", { class: "ide-diff-block", open: true },
      h("summary", { text: `Diff +${change.added || 0} −${change.removed || 0}${change.truncated ? " (truncated)" : ""}` }),
      pre);
  }

  function oneLine(value, limit = 90) {
    const text = String(value === undefined || value === null ? "" : value).replace(/\s+/g, " ").trim();
    return text.length > limit ? `${text.slice(0, limit - 1)}…` : text;
  }

  function toolSummary(entry) {
    const input = entry.input && typeof entry.input === "object" ? entry.input : {};
    const output = entry.output && typeof entry.output === "object" ? entry.output : {};
    const change = entry.change || {};
    switch (entry.tool) {
      case "file.read": return [input.path || output.path, output.start_line ? `lines ${output.start_line}–${output.end_line}` : ""].filter(Boolean).join(" · ");
      case "file.write":
      case "file.edit": return [input.path, change.added !== undefined ? `+${change.added || 0} −${change.removed || 0}` : ""].filter(Boolean).join(" · ");
      case "file.delete": return input.path || "";
      case "file.search": return [input.query, output.matches ? `${output.matches.length} match${output.matches.length === 1 ? "" : "es"}` : ""].filter(Boolean).join(" · ");
      case "file.tree": return input.path || "workspace";
      case "file.list": return input.path || "workspace";
      case "code.diagnostics":
        if (entry.ok === undefined) return "";
        return output.ok === false ? `${(output.issues || []).length} issue(s)` : "No issues";
      case "skill.read": return output.id || input.name || "";
      case "agent.swarm":
        return [
          output.agents ? `${output.agents.length} agent(s)` : "agent swarm",
          output.strategy ? String(output.strategy).replace("_", " ") : "",
        ].filter(Boolean).join(" · ");
      case "shell.exec": return oneLine(input.command, 100);
      case "sqlite.query": return oneLine(input.sql || input.path, 100);
      case "csv.read": return input.path || "";
      case "git.inspect": return input.action || "repository";
      default: return oneLine(input.path || input.query || "", 80);
    }
  }

  function syntaxPre(source, language) {
    const pre = h("pre", { class: "ide-tool-code" });
    const highlighter = Studio.highlight;
    const grammar = highlighter ? highlighter.resolve(language || "") : "text";
    if (!highlighter || !highlighter.hasGrammar(grammar) || source.length > 60000) {
      pre.textContent = source;
      return pre;
    }
    let state = null;
    const lines = source.split("\n");
    lines.forEach((line, index) => {
      const result = highlighter.scanLine(line, state, grammar);
      state = result.state;
      let cursor = 0;
      for (const token of result.tokens) {
        if (token.start > cursor) pre.append(document.createTextNode(line.slice(cursor, token.start)));
        pre.append(h("span", { class: `tok-${token.type}`, text: line.slice(token.start, token.end) }));
        cursor = token.end;
      }
      if (cursor < line.length) pre.append(document.createTextNode(line.slice(cursor)));
      if (index < lines.length - 1) pre.append(document.createTextNode("\n"));
    });
    return pre;
  }

  function rawDetails(label, text, language = "json") {
    return h("details", { class: "ide-tool-io" },
      h("summary", { text: label }), syntaxPre(text, language));
  }

  function issueRows(issues, fallback) {
    const rows = h("div", { class: "stack tight" });
    for (const issue of issues.slice(0, 20)) {
      const path = issue.path || fallback || "";
      rows.append(h("button", {
        class: "ide-tool-row", type: "button",
        onclick: () => { if (path) openFile(path, issue.line); },
      }, Studio.ui.badge(`${path || "issue"}${issue.line ? `:${issue.line}` : ""}`, "danger"),
        h("span", { class: "grow truncate", text: issue.message || issue.text || "Problem" })));
    }
    if (issues.length > 20) rows.append(h("p", { class: "help", text: `${issues.length - 20} more issue(s) not shown.` }));
    return rows;
  }

  function pathRows(items, fallback) {
    const rows = h("div", { class: "stack tight" });
    for (const item of items.slice(0, 40)) {
      const path = typeof item === "string" ? item : item.path || item.name || "";
      const isFile = typeof item === "string" ? true : item.type !== "dir";
      rows.append(h("button", {
        class: "ide-tool-row", type: "button", disabled: !path || !isFile,
        onclick: () => { if (path && isFile) openFile(path); },
      }, icon(isFile ? "file" : "folder", "small"), h("span", { class: "grow truncate", text: path || fallback || "" })));
    }
    if (items.length > 40) rows.append(h("p", { class: "help", text: `${items.length - 40} more not shown.` }));
    return rows;
  }

  function renderToolResult(entry) {
    const output = entry.output;
    if (output === undefined || output === null) return null;
    if (typeof output !== "object") return rawDetails("Result", pretty(output), "text");
    const input = entry.input && typeof entry.input === "object" ? entry.input : {};
    const parts = h("div", { class: "stack tight" });
    const diagnostics = output.diagnostics;
    let structured = false;
    if (diagnostics && typeof diagnostics === "object") {
      structured = true;
      if (diagnostics.ok === false) {
        parts.append(h("p", { class: "ide-tool-note warning",
          text: `Diagnostics: ${(diagnostics.issues || []).length} issue(s)${diagnostics.more ? ` (+${diagnostics.more} more)` : ""}.` }));
        if (diagnostics.issues && diagnostics.issues.length) parts.append(issueRows(diagnostics.issues, entry.change && entry.change.path));
      }
    }
    if ((entry.tool === "file.read" || entry.tool === "skill.read") && typeof output.content === "string") {
      structured = true;
      const path = output.path || input.path || "";
      const language = entry.tool === "skill.read" ? "markdown"
        : output.language || (path ? Studio.highlight.resolve("", path) : "text");
      parts.append(h("details", { class: "ide-tool-io", open: output.content.length < 2000 },
        h("summary", { text: `Contents${output.start_line ? ` · lines ${output.start_line}–${output.end_line}` : ""}` }),
        syntaxPre(output.content, language)));
    } else if (entry.tool === "file.inspect") {
      structured = true;
      const rows = [
        ["Kind", output.kind || "unknown"], ["Type", output.mime || "unknown"],
        ["Size", format.bytes(output.bytes || 0)], ["Modified", output.modified || "unknown"],
      ];
      if (output.dimensions) rows.push(["Dimensions", `${output.dimensions.width} × ${output.dimensions.height}`]);
      if (output.total_lines) rows.push(["Lines", String(output.total_lines)]);
      parts.append(Studio.ui.kv(rows));
      if (typeof output.hex === "string" && output.hex) parts.append(rawDetails("Hex", output.hex, "text"));
      if (typeof output.content === "string" && output.content) {
        parts.append(h("details", { class: "ide-tool-io" },
          h("summary", { text: "Text preview" }), syntaxPre(output.content, output.language || "text")));
      }
    } else if (Array.isArray(output.issues) && output.issues.length) {
      structured = true;
      parts.append(issueRows(output.issues, input.path));
    } else if (Array.isArray(output.matches)) {
      structured = true;
      if (!output.matches.length) parts.append(h("p", { class: "help", text: "No matches." }));
      const rows = h("div", { class: "stack tight" });
      for (const match of output.matches.slice(0, 40)) {
        rows.append(h("button", {
          class: "ide-tool-row", type: "button",
          onclick: () => { if (match.path) openFile(match.path, match.line); },
        }, Studio.ui.badge(`${match.path || "?"}:${match.line || ""}`, "accent"),
          h("span", { class: "grow truncate", text: match.text || "" })));
      }
      if (output.matches.length > 40) rows.append(h("p", { class: "help", text: `${output.matches.length - 40} more match(es) not shown.` }));
      parts.append(rows);
    } else if (Array.isArray(output.entries) || Array.isArray(output.files)) {
      structured = true;
      parts.append(pathRows(output.entries || output.files, input.path));
    } else if (typeof output.command === "string" || typeof output.stdout === "string") {
      structured = true;
      if (output.command) parts.append(h("div", { class: "ide-tool-path mono tiny", text: output.command }));
      if (typeof output.stdout === "string") parts.append(rawDetails("Output", output.stdout, "shell"));
      if (typeof output.stderr === "string" && output.stderr) parts.append(rawDetails("Errors", output.stderr, "shell"));
    }
    if (!structured) parts.append(rawDetails("Result", pretty(output)));
    else parts.append(rawDetails("Raw JSON", pretty(output)));
    return parts;
  }

  function swarmState(agent) {
    if (agent.status) return agent.status;
    if (agent.ok === false) return "failed";
    if (agent.ok) return "done";
    return "queued";
  }

  function swarmPhaseLabel(phase) {
    return {
      planning: "Planning", work: "Working", review: "Reviewing",
      revise: "Revising", synthesis: "Synthesizing", stop: "Stopping",
    }[phase] || "Working";
  }

  function swarmAgentStatusLabel(state) {
    if (state === "done") return "Done";
    if (state === "failed") return "Failed";
    if (state === "reviewing") return "Reviewing";
    if (state === "revising") return "Revising";
    if (["running", "thinking", "writing", "tool"].includes(state)) return "Working";
    return "Queued";
  }

  function renderSwarmEntry(entry, pending) {
    const agents = entry.swarmAgents
      || (entry.output && Array.isArray(entry.output.agents) ? entry.output.agents : []);
    const meta = entry.swarmMeta || {};
    const reviews = entry.swarmReviews || [];
    const messages = entry.swarmMessages || [];
    const memory = entry.swarmMemory || [];
    const changed = entry.swarmChanged || (entry.output && entry.output.changed_files) || [];
    const messageCount = Math.max(messages.length, Number(meta.messageCount) || 0);
    const reviewCount = Math.max(reviews.length, Number(meta.reviewCount) || 0);
    const memoryCount = Math.max(memory.length, Number(meta.memoryCount) || 0);
    const finished = agents.filter(agent => ["done", "failed"].includes(swarmState(agent))).length;
    const failed = agents.filter(agent => swarmState(agent) === "failed").length;
    const active = pending && !meta.finished;
    const role = meta.strategy ? String(meta.strategy).replace(/_/g, " ") : "leader-planned";
    const body = h("div", { class: "stack tight" });

    body.append(h("div", { class: "ide-swarm-meta" },
      Studio.ui.badge(role, "accent"),
      meta.round ? Studio.ui.badge(`Round ${meta.round}${meta.rounds ? `/${meta.rounds}` : ""}`, "muted") : null,
      meta.phase ? Studio.ui.badge(swarmPhaseLabel(meta.phase), pending ? "running" : "muted") : null,
      reviews.length ? Studio.ui.badge(`${reviewCount} review${reviewCount === 1 ? "" : "s"}`, "completed") : null,
      messageCount ? Studio.ui.badge(`${messageCount} message${messageCount === 1 ? "" : "s"}`, "muted") : null,
      memoryCount ? Studio.ui.badge(`${memoryCount} shared key${memoryCount === 1 ? "" : "s"}`, "muted") : null,
      changed.length ? Studio.ui.badge(`${changed.length} file${changed.length === 1 ? "" : "s"} changed`, "accent") : null));

    if (meta.leaderNote) {
      body.append(h("p", { class: "muted tiny ide-swarm-leader", text: meta.leaderNote }));
    }
    if (!agents.length) {
      body.append(h("p", { class: "muted tiny",
        text: pending ? "The leader is planning the team…" : "No agents reported." }));
    }

    const list = h("div", { class: "ide-swarm-agents" });
    for (const agent of agents) {
      const state = swarmState(agent);
      const running = ["running", "thinking", "writing", "tool", "reviewing", "revising", "queued"].includes(state);
      const glyph = running && active
        ? h("span", { class: "spinner" })
        : h("span", { class: "ide-entry-glyph" },
          icon(state === "failed" ? "alert" : state === "done" ? "check" : "sparkle", "small"));
      const label = agent.label || agent.agent || agent.id || "Agent";
      const detail = h("div", { class: "ide-swarm-body" });
      const facts = [];
      if (agent.role) facts.push(agent.role);
      if (agent.score !== undefined && agent.score !== null) facts.push(`${agent.score}/5`);
      if (agent.version && agent.version > 1) facts.push(`v${agent.version}`);
      if (agent.steps) facts.push(`${agent.steps} step${agent.steps === 1 ? "" : "s"}`);
      if (agent.elapsed) facts.push(`${agent.elapsed}s`);
      if (facts.length) detail.append(h("p", { class: "muted tiny ide-swarm-facts", text: facts.join(" · ") }));
      if (agent.task) detail.append(h("p", { class: "muted tiny", text: agent.task }));
      if (agent.lastTool) detail.append(h("p", { class: "muted tiny", text: `Last tool: ${agent.lastTool}` }));
      if (agent.note && !agent.output) detail.append(h("p", { class: "muted tiny", text: agent.note }));
      if (agent.error) {
        detail.append(Studio.ui.callout("danger", h("span", { text: agent.error })));
      } else if (agent.output) {
        const output = h("div", { class: "ide-swarm-output" });
        Studio.markdown.into(output, agent.output);
        detail.append(output);
      } else if (agent.text) {
        detail.append(h("pre", { class: "ide-swarm-preview", text: agent.text.slice(-4000) }));
      }
      list.append(h("details", { class: `ide-swarm-agent ${state}`, open: running || state === "failed" },
        h("summary", {},
          glyph,
          h("span", { class: "grow" },
            h("strong", { text: label }),
            agent.task ? h("span", { class: "ide-swarm-task", text: agent.task }) : null),
          Studio.ui.badge(swarmAgentStatusLabel(state),
            state === "done" ? "completed" : state === "failed" ? "failed"
              : state === "queued" ? "muted" : "running")),
        detail));
    }
    body.append(list);

    if (reviews.length) {
      const rows = h("div", { class: "stack tight" });
      for (const review of reviews.slice(-8)) {
        const reviewer = agents.find(agent => (agent.agent || agent.id) === review.reviewer);
        const target = agents.find(agent => (agent.agent || agent.id) === review.target);
        rows.append(h("div", { class: "ide-swarm-review" },
          Studio.ui.badge(`${review.score ?? "?"}/5`, review.score >= 4 ? "completed" : "warning"),
          h("span", { class: "grow tiny",
            text: `${reviewer ? reviewer.label : review.reviewer} → ${target ? target.label : review.target}` }),
          h("span", { class: "muted tiny truncate",
            text: oneLine((review.issues || []).join("; ") || review.summary || "reviewed", 110) })));
      }
      body.append(h("details", { class: "ide-swarm-section", open: reviews.length <= 3 },
        h("summary", { text: `Peer review · ${reviews.length} review${reviews.length === 1 ? "" : "s"}` }), rows));
    }

    if (messages.length) {
      const rows = h("div", { class: "stack tight" });
      for (const message of messages.slice(-12)) {
        rows.append(h("div", { class: "ide-swarm-message" },
          h("span", { class: "mono tiny muted", text: `${message.from} → ${message.to}` }),
          h("span", { class: "grow tiny", text: oneLine(message.text, 220) })));
      }
      body.append(h("details", { class: "ide-swarm-section" },
        h("summary", { text: `Agent messages · ${messages.length}` }), rows));
    }

    if (memory.length) {
      const rows = h("div", { class: "row wrap" });
      for (const item of memory.slice(-16)) rows.append(Studio.ui.badge(item.key || item, "muted"));
      body.append(h("details", { class: "ide-swarm-section" },
        h("summary", { text: `Shared memory · ${memory.length} key${memory.length === 1 ? "" : "s"}` }), rows));
    }

    if (changed.length) {
      const rows = h("div", { class: "stack tight" });
      for (const path of changed.slice(0, 20)) {
        rows.append(h("button", { class: "ide-tool-row", type: "button",
          onclick: () => openFile(path) },
          icon("file", "small"), h("span", { class: "grow truncate", text: path })));
      }
      body.append(h("details", { class: "ide-swarm-section" },
        h("summary", { text: `Files changed · ${changed.length}` }), rows));
    }

    if (meta.final && !active) {
      const summary = h("div", { class: "ide-swarm-final" });
      Studio.markdown.into(summary, meta.final);
      body.append(h("details", { class: "ide-swarm-section", open: false },
        h("summary", { text: "Synthesised result" }), summary));
    }

    return h("article", { class: `ide-entry tool swarm${failed ? " failed" : ""}` },
      h("div", { class: "ide-entry-head" },
        h("span", { class: "ide-entry-glyph" }, icon("agents", "small")),
        h("strong", { class: "grow", text: "Agent swarm" }),
        h("span", { class: "muted tiny", text: `${finished}/${agents.length} finished` }),
        active ? Studio.ui.badge("Running", "running")
          : Studio.ui.badge(failed ? "Finished with failures" : "Done",
            failed ? "failed" : "completed")),
      body);
  }

  function renderToolEntry(entry, pending) {
    if (entry.tool === "agent.swarm") return renderSwarmEntry(entry, pending);
    const ok = entry.ok;
    const badge = pending ? Studio.ui.badge("Running", "running")
      : Studio.ui.badge(ok ? "Done" : "Failed", ok ? "completed" : "failed");
    const input = entry.input && typeof entry.input === "object" ? entry.input : {};
    const summary = toolSummary(entry);
    const path = (entry.change && entry.change.path) || input.path || (entry.output && entry.output.path) || "";
    const startLine = entry.output && typeof entry.output === "object" ? entry.output.start_line : 0;
    const actions = path
      ? h("div", { class: "row wrap" },
        Studio.ui.button("Open file", { small: true, onclick: () => openFile(path, startLine) }))
      : null;
    return h("article", { class: `ide-entry tool${ok === false ? " failed" : ""}` },
      h("div", { class: "ide-entry-head" },
        h("span", { class: "ide-entry-glyph" }, icon(ok === false ? "alert" : "bolt", "small")),
        h("strong", { class: "grow", text: toolLabel(entry.tool) }),
        summary ? h("span", { class: "muted tiny truncate ide-tool-summary", text: summary }) : null,
        badge),
      h("div", { class: "ide-entry-body" },
        h("div", { class: "ide-tool-path mono tiny muted", text: entry.tool }),
        Object.keys(input).length ? rawDetails("Input", pretty(input)) : null,
        pending ? null : renderToolResult(entry),
        diffBlock(entry.change),
        actions));
  }

  function renderEntry(project, entry) {
    if (!entry || !entry.role) return h("div");
    if (entry.role === "user") {
      return h("article", { class: "ide-entry user" },
        h("div", { class: "ide-entry-head" },
          h("span", { class: "ide-entry-glyph" }, icon("chat", "small")),
          h("strong", { text: "You" }),
          h("button", { class: "icon-button small ide-entry-action", type: "button", "aria-label": "Copy message",
            onclick: () => Studio.copy(entry.text || "") }, icon("copy", "small"))),
        h("div", { class: "ide-bubble user", text: entry.text || "" }));
    }
    if (entry.role === "assistant") {
      const bubble = h("div", { class: "ide-bubble assistant" });
      Studio.markdown.into(bubble, entry.text || "");
      return h("article", { class: "ide-entry assistant" },
        h("div", { class: "ide-entry-head" },
          h("span", { class: "ide-entry-glyph" }, icon("sparkle", "small")),
          h("strong", { class: "grow", text: "Mariana" }),
          h("button", { class: "icon-button small ide-entry-action", type: "button",
            "aria-label": "Read this reply aloud",
            title: "Read this reply aloud; click any read button again to stop",
            onclick: () => {
              if (!Studio.voice || !Studio.voice.synthesisSupported()) {
                Studio.toast("This browser cannot read replies aloud.", "info");
                return;
              }
              if (Studio.voice.speaking()) {
                Studio.voice.stopSpeaking();
                return;
              }
              Studio.voice.speak(entry.text || "");
            } }, icon("volume", "small")),
          h("button", { class: "icon-button small ide-entry-action", type: "button", "aria-label": "Copy reply",
            onclick: () => Studio.copy(entry.text || "") }, icon("copy", "small"))),
        bubble);
    }
    if (entry.role === "tool") return renderToolEntry(entry, Boolean(entry.pending));
    if (entry.role === "plan") {
      const list = h("ol", { class: "ide-plan-list" });
      for (const todo of entry.todos || []) {
        list.append(h("li", { class: `plan-step ${todo.status || "pending"}` },
          h("span", { class: "plan-marker", text: todo.status === "done" ? "✓" : todo.status === "active" ? "▸" : "○" }),
          h("span", { text: todo.text })));
      }
      return h("article", { class: "ide-entry plan" },
        h("div", { class: "ide-entry-head" }, h("span", { class: "ide-entry-glyph" }, icon("layout", "small")), h("strong", { text: "Plan" })),
        list);
    }
    if (entry.role === "error") {
      return h("article", { class: "ide-entry error" },
        Studio.ui.callout("danger", h("span", { class: "grow", text: entry.text || "The run failed." })),
        project.lastPrompt ? h("div", { class: "row" },
          Studio.ui.button("Retry last task", {
            small: true,
            onclick: () => {
              $("ide-message").value = project.lastPrompt;
              $("ide-message").focus();
            },
          })) : null);
    }
    return h("article", { class: "ide-entry notice muted small", text: entry.text || "" });
  }

  function renderPlan(project) {
    if (!isActive(project)) return;
    const root = $("ide-plan");
    const todos = project.plan || [];
    if (!todos.length) {
      root.classList.add("hidden");
      Studio.clear(root);
      return;
    }
    const collapsed = Boolean(project.planCollapsed);
    root.classList.remove("hidden");
    root.classList.toggle("collapsed", collapsed);
    Studio.ui.render(root,
      h("div", { class: "row between" },
        h("button", {
          class: "ide-plan-toggle", type: "button", "aria-expanded": String(!collapsed),
          title: collapsed ? "Expand plan" : "Collapse plan",
          onclick: () => { project.planCollapsed = !collapsed; renderPlan(project); },
        }, icon(collapsed ? "chevron-right" : "chevron-down", "small"), h("strong", { text: "Plan" })),
        Studio.ui.badge(`${todos.filter(todo => todo.status === "done").length}/${todos.length}`, "completed")),
      collapsed ? null : h("div", { class: "ide-plan-list" },
        ...todos.slice(0, 20).map(todo => h("div", { class: `plan-step ${todo.status || "pending"}` },
          h("span", { class: "plan-marker", text: todo.status === "done" ? "✓" : todo.status === "active" ? "▸" : "○" }),
          h("span", { class: "grow", text: todo.text })))));
  }

  /* -- live run feed -------------------------------------------------------- */
  function closeFeed(project) {
    if (project && project.feed) {
      project.feed.close();
      project.feed = null;
    }
    removeLive(project);
  }

  function followRun(project, runId, since) {
    closeFeed(project);
    project.runId = runId;
    project.lastSeq = since || 0;
    project.running = true;
    updateBusy(project);
    renderProjects();
    project.feed = Studio.subscribe(
      () => `/api/ide/runs/${encodeURIComponent(runId)}/events?since=${project.lastSeq}`,
      event => {
        if (!event || event.seq === undefined || project.runId !== runId) return;
        if (event.seq <= project.lastSeq) return;
        project.lastSeq = event.seq;
        handleEvent(project, event);
      },
      connected => {
        if (!connected && project.running) setStatus(project, "Reconnecting to the run…");
      },
    );
  }

  function showLive(project) {
    if (!isActive(project)) return;
    removeLive(project);
    project.liveText = "";
    const preview = h("div", { class: "muted small", text: "Thinking…" });
    const node = h("article", { class: "ide-entry live", "aria-hidden": "true" },
      h("div", { class: "ide-entry-head" },
        h("span", { class: "spinner" }),
        h("strong", { text: "Mariana" }),
        h("span", { class: "muted tiny", text: "working" })),
      h("div", { class: "ide-live-preview" }, preview));
    project.live = { node, preview, timer: null };
    const root = $("ide-transcript");
    const empty = root.querySelector(".empty");
    if (empty) root.replaceChildren();
    root.append(node);
    scrollTranscript(project);
  }

  function removeLive(project) {
    if (project && project.live && project.live.timer) clearTimeout(project.live.timer);
    if (project && project.live && project.live.node && typeof project.live.node.remove === "function") {
      project.live.node.remove();
    }
    if (project) {
      project.live = null;
      project.liveText = "";
    }
  }

  /* Streaming text arrives many times per second; render Markdown at most
     every 120 ms and keep the raw tail while the model is writing JSON. */
  function scrollLive(project) {
    const box = project.live && project.live.preview && project.live.preview.parentElement;
    if (box) box.scrollTop = box.scrollHeight;
  }

  function scheduleLivePreview(project) {
    if (!isActive(project) || !project.live) return;
    const trimmed = project.liveText.trimStart();
    if (trimmed.startsWith("{") || trimmed.startsWith("```json")) {
      clearTimeout(project.live.timer);
      project.live.preview.classList.remove("markdown");
      project.live.preview.textContent = "Preparing a tool call…";
      scrollLive(project);
      return;
    }
    clearTimeout(project.live.timer);
    project.live.timer = setTimeout(() => {
      if (!project.live || !isActive(project)) return;
      const visible = project.liveText.slice(-20000);
      Studio.markdown.into(project.live.preview, visible, { enhance: false });
      scrollLive(project);
      scrollTranscript(project);
    }, 120);
  }

  function findPendingToolEntry(project, callId) {
    const transcript = (project.session && project.session.transcript) || [];
    for (let index = transcript.length - 1; index >= 0; index -= 1) {
      const item = transcript[index];
      if (item && item.role === "tool" && item.call === callId && item.pending) return item;
    }
    return null;
  }

  function swarmAgentState(project, event) {
    const pending = project.pendingTools.get(event.swarm);
    const entry = pending ? pending.entry : findPendingToolEntry(project, event.swarm);
    if (!entry) return null;
    if (!Array.isArray(entry.swarmAgents)) entry.swarmAgents = [];
    let agent = entry.swarmAgents.find(item => item.agent === event.agent);
    if (!agent) {
      agent = { agent: event.agent, label: event.label || event.agent, task: event.task || "" };
      entry.swarmAgents.push(agent);
    }
    return { entry, agent };
  }

  function refreshSwarmCard(project, swarm, immediate) {
    const pending = project.pendingTools.get(swarm);
    if (!pending || !pending.card || !isActive(project)) return;
    const paint = () => {
      if (!pending.card || !pending.card.isConnected) return;
      const next = renderToolEntry(pending.entry, true);
      pending.card.replaceWith(next);
      pending.card = next;
    };
    clearTimeout(project.swarmTimers.get(swarm));
    if (immediate) {
      project.swarmTimers.delete(swarm);
      paint();
      return;
    }
    project.swarmTimers.set(swarm, setTimeout(() => {
      project.swarmTimers.delete(swarm);
      paint();
    }, 100));
  }

  function handleEvent(project, event) {
    switch (event.type) {
      case "run_started":
        project.running = true;
        for (const timer of project.swarmTimers.values()) clearTimeout(timer);
        project.swarmTimers.clear();
        project.usage = { prompt_tokens: 0, completion_tokens: 0 };
        if (event.budget) {
          if (!project.preview) project.preview = {};
          project.preview.budget = event.budget;
          project.budgetLocked = true;
        }
        updateBusy(project);
        updateUsageBadge(project);
        updateBudgetMeter(project);
        renderProjects();
        setStatus(project, `Working with ${event.model || "the selected model"} (${event.mode || state.mode}).`);
        break;
      case "assistant_start":
        showLive(project);
        break;
      case "assistant_delta":
        project.liveText += event.text || "";
        if (isActive(project)) {
          if (!project.live) showLive(project);
          scheduleLivePreview(project);
        }
        break;
      case "reasoning_delta":
        if (isActive(project)) {
          if (!project.live) showLive(project);
          if (project.live) {
            clearTimeout(project.live.timer);
            project.live.preview.classList.remove("markdown");
            project.live.preview.textContent = "Reasoning…";
            scrollLive(project);
          }
        }
        break;
      case "assistant_end":
        if (event.usage) {
          const usage = project.usage || { prompt_tokens: 0, completion_tokens: 0 };
          usage.prompt_tokens = (Number(usage.prompt_tokens) || 0) + (Number(event.usage.prompt_tokens) || 0);
          usage.completion_tokens = (Number(usage.completion_tokens) || 0) + (Number(event.usage.completion_tokens) || 0);
          project.usage = usage;
          updateUsageBadge(project);
        }
        removeLive(project);
        break;
      case "tool_call": {
        const entry = { role: "tool", tool: event.tool, input: event.input, call: event.call,
          pending: true, at: new Date().toISOString() };
        if (event.tool === "agent.swarm") entry.swarmAgents = [];
        if (!project.session) project.session = { id: "", title: "", transcript: [], changes: {} };
        if (!project.session.transcript) project.session.transcript = [];
        project.session.transcript.push(entry);
        if (isActive(project)) {
          const card = renderToolEntry(entry, true);
          project.pendingTools.set(event.call, { card, entry });
          const root = $("ide-transcript");
          const empty = root.querySelector(".empty");
          if (empty) root.replaceChildren();
          root.append(card);
          scrollTranscript(project);
        }
        const detail = event.input && typeof event.input === "object"
          ? event.input.path || event.input.command || event.input.query || event.input.name || "" : "";
        setStatus(project, `${toolLabel(event.tool)}${detail ? ` · ${oneLine(detail, 60)}` : ""}…`);
        break;
      }
      case "tool_result": {
        const pending = project.pendingTools.get(event.call);
        const entry = pending ? pending.entry : findPendingToolEntry(project, event.call);
        const target = entry || { role: "tool", tool: event.tool, input: {} };
        target.output = event.output;
        target.ok = event.ok;
        target.change = event.change;
        target.pending = false;
        if (event.tool === "agent.swarm" && event.output
          && Array.isArray(event.output.agents)) {
          target.swarmAgents = event.output.agents.map(agent => ({
            ...agent, status: agent.ok === false ? "failed" : "done", text: "",
          }));
          target.swarmMeta = {
            ...(target.swarmMeta || {}),
            strategy: event.output.strategy,
            rounds: event.output.rounds,
            round: event.output.round,
            phase: "synthesis", finished: true,
            final: event.output.output || event.output.text || "",
            summary: event.output.summary || "",
            messageCount: event.output.messages || 0,
            reviewCount: Array.isArray(event.output.reviews) ? event.output.reviews.length : 0,
            memoryCount: event.output.blackboard || 0,
          };
          if (Array.isArray(event.output.reviews)) target.swarmReviews = event.output.reviews;
          target.swarmChanged = event.output.changed_files || [];
        }
        clearTimeout(project.swarmTimers.get(event.call));
        project.swarmTimers.delete(event.call);
        if (pending) {
          project.pendingTools.delete(event.call);
          if (isActive(project)) {
            pending.card.replaceWith(renderToolEntry(target, false));
            scrollTranscript(project);
          }
        } else if (!entry && isActive(project)) {
          appendEntry(project, { ...target, at: new Date().toISOString() });
        }
        break;
      }
      case "plan":
        project.plan = event.plan || [];
        if (project.session) project.session.plan = project.plan;
        renderPlan(project);
        break;
      case "context_consolidated":
        if (event.iteration && project.preview && project.preview.budget
          && event.after_tokens !== undefined) {
          const budget = project.preview.budget;
          budget.used_tokens = Math.max(0, Number(event.after_tokens) || 0);
          budget.percent = budget.context_window
            ? Math.round(budget.used_tokens * 1000 / budget.context_window) / 10 : 0;
          budget.remaining_tokens = Math.max(0, budget.context_window - budget.used_tokens);
          updateBudgetMeter(project);
        }
        setStatus(project, event.message || "Consolidated older turns to free context.");
        appendEntry(project, { role: "notice", kind: "consolidation",
          text: event.message || "Consolidated older turns to free context." });
        if (!project.running) refreshContextPreview(project);
        break;
      case "swarm_start": {
        const pending = project.pendingTools.get(event.swarm);
        const entry = pending ? pending.entry : findPendingToolEntry(project, event.swarm);
        if (entry) {
          entry.swarmMeta = {
            strategy: event.strategy || "auto", leader: event.leader || "",
            rounds: event.requested_rounds || 0, round: 0, phase: "planning",
            leaderNote: "The leader is planning the team.",
          };
          if (Array.isArray(event.agents) && event.agents.length) {
            entry.swarmAgents = event.agents.map(agent => ({
              agent: agent.id || agent.agent, label: agent.label, role: agent.role,
              task: agent.task, status: "queued",
            }));
          } else if (!entry.swarmAgents) {
            entry.swarmAgents = [];
          }
        }
        refreshSwarmCard(project, event.swarm, true);
        setStatus(project, event.message || "The leader is planning a swarm.");
        break;
      }
      case "swarm_plan": {
        const pending = project.pendingTools.get(event.swarm);
        const entry = pending ? pending.entry : findPendingToolEntry(project, event.swarm);
        if (entry) {
          entry.swarmAgents = (event.agents || []).map(agent => ({
            agent: agent.id || agent.agent, label: agent.label, role: agent.role,
            task: agent.task, tools: agent.tools, writes: agent.writes,
            status: agent.status || "queued", score: agent.score, version: agent.version,
          }));
          entry.swarmMeta = {
            ...(entry.swarmMeta || {}),
            strategy: event.strategy, rounds: event.rounds || 0,
            leaderNote: event.notes || (entry.swarmMeta && entry.swarmMeta.leaderNote),
            phase: "work",
          };
        }
        refreshSwarmCard(project, event.swarm, true);
        setStatus(project, event.message || "Swarm plan ready.");
        break;
      }
      case "swarm_leader": {
        const pending = project.pendingTools.get(event.swarm);
        const entry = pending ? pending.entry : findPendingToolEntry(project, event.swarm);
        if (entry) {
          entry.swarmMeta = {
            ...(entry.swarmMeta || {}), phase: event.phase || (entry.swarmMeta || {}).phase,
            leaderNote: event.text || event.message || (entry.swarmMeta || {}).leaderNote,
          };
        }
        refreshSwarmCard(project, event.swarm, false);
        if (event.message) setStatus(project, event.message);
        break;
      }
      case "swarm_round": {
        const pending = project.pendingTools.get(event.swarm);
        const entry = pending ? pending.entry : findPendingToolEntry(project, event.swarm);
        if (entry) {
          entry.swarmMeta = {
            ...(entry.swarmMeta || {}), round: event.round || 0,
            phase: event.phase || "work",
          };
        }
        refreshSwarmCard(project, event.swarm, true);
        if (event.message) setStatus(project, event.message);
        break;
      }
      case "swarm_agent_start": {
        const state = swarmAgentState(project, event);
        if (state) {
          state.agent.label = event.label || state.agent.label;
          state.agent.role = event.role || state.agent.role;
          state.agent.task = event.task || state.agent.task;
          state.agent.version = event.version || state.agent.version;
          state.agent.status = event.phase === "review" ? "reviewing"
            : event.phase === "revise" ? "revising" : "running";
          state.agent.note = event.message || "";
          if (event.phase) {
            state.entry.swarmMeta = { ...(state.entry.swarmMeta || {}), phase: event.phase, round: event.round || 0 };
          }
          refreshSwarmCard(project, event.swarm, true);
          setStatus(project, `${state.agent.label} is ${swarmAgentStatusLabel(state.agent.status).toLowerCase()}…`);
        }
        break;
      }
      case "swarm_agent_step": {
        const state = swarmAgentState(project, event);
        if (state) {
          state.agent.status = event.phase === "review" ? "reviewing"
            : event.phase === "revise" ? "revising" : "thinking";
          state.agent.note = event.message || state.agent.note;
          refreshSwarmCard(project, event.swarm, false);
        }
        break;
      }
      case "swarm_agent_status": {
        const state = swarmAgentState(project, event);
        if (state) {
          state.agent.status = event.phase === "review" ? "reviewing"
            : event.phase === "revise" ? "revising" : "thinking";
          state.agent.note = event.message || "";
          refreshSwarmCard(project, event.swarm, false);
        }
        break;
      }
      case "swarm_agent_delta": {
        const state = swarmAgentState(project, event);
        if (state) {
          state.agent.status = event.phase === "review" ? "reviewing"
            : event.phase === "revise" ? "revising" : "writing";
          state.agent.text = `${state.agent.text || ""}${event.text || ""}`.slice(-8000);
          refreshSwarmCard(project, event.swarm, false);
        }
        break;
      }
      case "swarm_agent_tool": {
        const state = swarmAgentState(project, event);
        if (state) {
          state.agent.status = event.phase === "review" ? "reviewing"
            : event.phase === "revise" ? "revising" : "tool";
          state.agent.lastTool = event.tool || "";
          state.agent.note = event.message || "";
          refreshSwarmCard(project, event.swarm, true);
          setStatus(project, `${state.agent.label} is using ${toolLabel(event.tool)}…`);
        }
        break;
      }
      case "swarm_message": {
        const pending = project.pendingTools.get(event.swarm);
        const entry = pending ? pending.entry : findPendingToolEntry(project, event.swarm);
        if (entry) {
          if (!Array.isArray(entry.swarmMessages)) entry.swarmMessages = [];
          entry.swarmMessages.push({
            from: event.from || "agent", to: event.to || "leader",
            kind: event.message_kind || "message", text: event.text || "",
          });
          if (entry.swarmMessages.length > 24) entry.swarmMessages.splice(0, entry.swarmMessages.length - 24);
        }
        refreshSwarmCard(project, event.swarm, true);
        setStatus(project, event.message || "Swarm agents exchanged a message.");
        break;
      }
      case "swarm_memory": {
        const pending = project.pendingTools.get(event.swarm);
        const entry = pending ? pending.entry : findPendingToolEntry(project, event.swarm);
        if (entry) {
          if (!Array.isArray(entry.swarmMemory)) entry.swarmMemory = [];
          if (event.action === "delete") {
            entry.swarmMemory = entry.swarmMemory.filter(item => item.key !== event.key);
          } else if (event.key) {
            const known = entry.swarmMemory.find(item => item.key === event.key);
            if (known) {
              known.version = event.version;
              known.author = event.agent || known.author;
            } else {
              entry.swarmMemory.push({ key: event.key, author: event.agent, version: event.version });
            }
          }
          if (entry.swarmMemory.length > 30) entry.swarmMemory.splice(0, entry.swarmMemory.length - 30);
        }
        refreshSwarmCard(project, event.swarm, false);
        setStatus(project, event.message || "Swarm memory updated.");
        break;
      }
      case "swarm_review": {
        const pending = project.pendingTools.get(event.swarm);
        const entry = pending ? pending.entry : findPendingToolEntry(project, event.swarm);
        if (entry) {
          if (!Array.isArray(entry.swarmReviews)) entry.swarmReviews = [];
          entry.swarmReviews.push({
            reviewer: event.reviewer, target: event.target, score: event.score,
            issues: event.issues || [], round: event.round,
          });
          if (entry.swarmReviews.length > 16) entry.swarmReviews.splice(0, entry.swarmReviews.length - 16);
        }
        refreshSwarmCard(project, event.swarm, true);
        setStatus(project, event.message || "A peer review was recorded.");
        break;
      }
      case "swarm_agent_end": {
        const state = swarmAgentState(project, event);
        if (state) {
          state.agent.status = event.status || (event.ok ? "done" : "failed");
          if (event.output) state.agent.output = event.output;
          if (event.summary) state.agent.note = event.summary;
          state.agent.error = event.error || "";
          state.agent.elapsed = event.elapsed;
          state.agent.score = event.score !== undefined ? event.score : state.agent.score;
          state.agent.version = event.version || state.agent.version;
          state.agent.steps = event.steps || state.agent.steps;
          state.agent.text = "";
          if (event.phase) {
            state.entry.swarmMeta = { ...(state.entry.swarmMeta || {}), phase: event.phase, round: event.round || 0 };
          }
          refreshSwarmCard(project, event.swarm, true);
          setStatus(project, event.message || `${state.agent.label} finished.`);
        }
        break;
      }
      case "swarm_end": {
        const pending = project.pendingTools.get(event.swarm);
        const entry = pending ? pending.entry : findPendingToolEntry(project, event.swarm);
        if (entry) {
          if (Array.isArray(event.agents) && event.agents.length) {
            entry.swarmAgents = event.agents.map(agent => ({
              ...agent, agent: agent.id || agent.agent, status: agent.status,
            }));
          }
          entry.swarmMeta = {
            ...(entry.swarmMeta || {}), finished: true, final: event.output || "",
            phase: "synthesis", summary: event.summary || "",
            strategy: entry.swarmMeta && entry.swarmMeta.strategy,
            messageCount: event.messages ?? (entry.swarmMeta || {}).messageCount,
            reviewCount: event.reviews ?? (entry.swarmMeta || {}).reviewCount,
            memoryCount: event.blackboard ?? (entry.swarmMeta || {}).memoryCount,
          };
          if (Array.isArray(event.changed_files)) entry.swarmChanged = event.changed_files;
        }
        refreshSwarmCard(project, event.swarm, true);
        if (event.message) setStatus(project, event.message);
        break;
      }
      case "file_changed":
        refreshTreeSoon(project, event.path || "");
        if (event.path === project.activePath) refreshContextPreview(project);
        if (event.path) {
          const tab = project.open.find(item => item.path === event.path);
          if (tab && !tab.dirty) reloadOpenFile(project, event.path);
          else if (tab && !tab.stale) {
            tab.stale = true;
            if (isActive(project)) {
              renderTabs(project);
              renderNotice(project);
              Studio.toast(`${event.path} changed by the agent; your unsaved copy was kept.`, "info");
            }
          }
        }
        break;
      case "log":
        if (event.message) setStatus(project, event.message);
        break;
      case "run_completed":
      case "run_failed":
      case "run_cancelled":
        finishRun(project, event);
        break;
      default:
        break;
    }
  }

  async function finishRun(project, event) {
    if (!project.runId) return;
    const status = event.type.replace("run_", "");
    const sessionId = event.session_id || (project.session && project.session.id);
    project.running = false;
    project.budgetLocked = false;
    project.runId = "";
    closeFeed(project);
    updateBusy(project);
    renderProjects();
    removeLive(project);
    for (const pending of project.pendingTools.values()) {
      if (isActive(project) && typeof pending.card.remove === "function") pending.card.remove();
    }
    for (const timer of project.swarmTimers.values()) clearTimeout(timer);
    project.swarmTimers.clear();
    project.pendingTools.clear();
    if (event.usage) {
      project.usage = {
        prompt_tokens: Number(event.usage.prompt_tokens) || 0,
        completion_tokens: Number(event.usage.completion_tokens) || 0,
      };
      updateUsageBadge(project);
    }
    if (event.type === "run_failed" && event.error) {
      appendEntry(project, { role: "error", text: event.error });
      toastFor(project, event.error, "error");
      setStatus(project, "The run failed. Read the error below and try again.");
    } else if (status === "cancelled") {
      setStatus(project, "Run cancelled.");
    } else {
      setStatus(project, event.exhausted
        ? (event.timed_out
          ? "The time budget ran out. Ask for the next step to continue."
          : "The step budget ran out. Ask for the next step to continue.")
        : "Finished. Review the diff, then run Check project or your tests.");
    }
    await refreshParentDirectory(project, "");
    await loadSessions(project);
    if (sessionId) {
      try {
        const data = await api(sessionUrl(sessionId));
        if (project.running || (project.session && project.session.id !== sessionId)) return;
        project.session = data.session;
        project.sessionId = sessionId;
        project.plan = data.session.plan || [];
        if (data.session.usage && typeof data.session.usage === "object") {
          project.usage = data.session.usage;
        }
        updateSessionSummary(project, {
          ...(summaryFor(project, sessionId) || {}),
          id: sessionId, title: data.session.title, updated: data.session.updated,
          turns: (data.session.transcript || []).filter(item => item.role === "user").length,
          changes: Object.values(data.session.changes || {}).filter(change => !change.reverted).length,
          active: false, pinned: Boolean(data.session.pinned),
        });
        if (isActive(project)) {
          $("ide-session-title").textContent = data.session.title || "Session";
          renderTranscript(project);
          renderPlan(project);
          renderSessions(project);
          renderContext(project);
          updateWorkspaceState();
          refreshContextPreview(project);
        }
      } catch (error) {
        /* keep the live transcript when the canonical copy cannot be read */
      }
    }
    if (status === "completed" && state.voiceAuto && state.viewActive && isActive(project)
      && Studio.voice && Studio.voice.synthesisSupported()) {
      const transcript = (project.session && project.session.transcript) || [];
      const last = [...transcript].reverse().find(item => item && item.role === "assistant" && item.text);
      if (last && last.text !== project.spokenText) {
        project.spokenText = last.text;
        Studio.voice.speak(last.text, {
          onError: error => toastFor(project, error.message, "error"),
        });
      }
    }
  }

  /* -- context -------------------------------------------------------------- */
  function sessionContext(project) {
    const include = [...project.context.include];
    const exclude = [...project.context.exclude];
    return { include, exclude };
  }

  function addContextPath(project, path) {
    if (!path) return;
    if (!project.context.include.includes(path)) {
      project.context.include.push(path);
      renderContext(project);
      persistSessionContext(project);
      toastFor(project, `${path} pinned to the agent context.`, "success");
    }
    if (project.session) project.session.context = sessionContext(project);
  }

  function removeContextPath(project, path) {
    project.context.include = project.context.include.filter(item => item !== path);
    renderContext(project);
    persistSessionContext(project);
    if (project.session) project.session.context = sessionContext(project);
  }

  function persistSessionContext(project) {
    clearTimeout(project.contextTimer);
    project.contextTimer = setTimeout(async () => {
      if (!project.session || !project.session.id || project.running) return;
      try {
        await api(sessionUrl(project.session.id), {
          method: "PATCH", body: { context: sessionContext(project) },
        });
      } catch (error) {
        console.error("Saving conversation context failed", error);
      }
    }, 400);
  }

  async function toggleHidden(project, path) {
    const hidden = (project.settings.excluded || []).some(item => path === item || path.startsWith(`${item}/`));
    const excluded = hidden
      ? (project.settings.excluded || []).filter(item => !(item === path || path.startsWith(`${item}/`)
        || item.startsWith(`${path}/`)))
      : [...(project.settings.excluded || []), path];
    try {
      const settings = await api(`/api/ide/workspaces/${encodeURIComponent(project.id)}/settings`, {
        method: "PUT", body: { excluded },
      });
      project.settings = settings;
      if (hidden) {
        project.expanded.add(path);
      } else {
        project.context.include = project.context.include.filter(item => item !== path && !item.startsWith(`${path}/`));
        persistSessionContext(project);
        for (const tab of [...project.open]) {
          if (tab.path === path || tab.path.startsWith(`${path}/`)) {
            if (tab.dirty) tab.stale = true;
            else await closeTab(project, tab.path, { force: true });
          }
        }
      }
      await refreshTree(project, { deep: true });
      renderContext(project);
      toastFor(project, hidden ? `${path} is visible to the agent again.` : `${path} is hidden from the agent.`, "success");
    } catch (error) {
      Studio.toast(error.message, "error");
    }
  }

  function renderContext(project) {
    if (!isActive(project)) return;
    const chipRoot = Studio.ui.render($("ide-context-chips"));
    for (const path of project.context.include) {
      chipRoot.append(h("span", { class: "ide-context-chip" },
        icon("pin", "small"),
        h("span", { class: "truncate", text: path }),
        h("button", {
          class: "icon-button small", type: "button", "aria-label": `Remove ${path} from context`,
          onclick: () => removeContextPath(project, path),
        }, icon("x", "small"))));
    }
    if (project.activePath) {
      chipRoot.append(h("span", { class: "ide-context-chip muted" },
        icon("file", "small"), h("span", { class: "truncate", text: `active: ${project.activePath}` })));
    }
    if (!chipRoot.children.length) {
      chipRoot.append(h("span", { class: "muted tiny", text: "No pinned context; the agent sees the project map." }));
    }
    refreshContextPreview(project);
  }

  function contextPreviewBody(project, draft, options = {}) {
    return {
      context: sessionContext(project),
      active_file: project.activePath || "",
      context_files: project.open
        .map(tab => tab.path)
        .filter(path => path && path !== project.activePath)
        .slice(0, 4),
      model: project.model || state.model || $("ide-model").value || "",
      mode: state.mode,
      session_id: project.sessionId || "",
      message: draft === undefined ? (project.draft || $("ide-message").value || "") : draft,
      refresh: Boolean(options.refresh),
    };
  }

  function updateContextSummary(project) {
    if (!isActive(project)) return;
    const data = project.preview;
    const budget = data && data.budget;
    if (!budget) {
      $("ide-context").textContent = "Estimating context…";
      return;
    }
    const pieces = [];
    if (data.selected) pieces.push(`${data.selected} file(s) pinned`);
    else pieces.push("No pinned files");
    pieces.push(`${format.compact(budget.used_tokens)} / ${format.compact(budget.context_window)} `
      + `${budget.unit} (${format.percent(budget.percent)})`);
    if (data.truncated) pieces.push("context truncated");
    const hidden = (project.settings.excluded || []).length + project.context.exclude.length;
    if (hidden) pieces.push(`${hidden} path(s) hidden`);
    if (project.activePath) pieces.push("active file included");
    $("ide-context").textContent = pieces.join(" · ");
  }

  async function loadContextPreview(project, draft, options = {}) {
    const token = ++project.previewToken;
    let data;
    try {
      data = await api(`/api/ide/workspaces/${encodeURIComponent(project.id)}/context`, {
        body: contextPreviewBody(project, draft, options),
      });
    } catch (error) {
      if (token === project.previewToken && isActive(project)) {
        $("ide-context").textContent = "Context estimate unavailable.";
      }
      throw error;
    }
    if (token !== project.previewToken) return null;
    const runBudget = project.budgetLocked && project.running
      && project.preview && project.preview.budget;
    if (runBudget) data.budget = project.preview.budget;
    project.preview = data;
    project.budgetLocked = Boolean(runBudget);
    if (isActive(project)) {
      updateBudgetMeter(project, draft);
      updateContextSummary(project);
    }
    return data;
  }

  function refreshContextPreview(project, draft, options = {}) {
    clearTimeout(project.previewTimer);
    project.previewTimer = setTimeout(() => {
      loadContextPreview(project, draft, options).catch(() => {
        /* the summary line already carries the failure */
      });
    }, 320);
  }

  function tokenRow(label, tokens, unit, total, options = {}) {
    const ratio = total ? Math.min(100, Math.max(0, tokens * 100 / total)) : 0;
    return h("div", { class: `ide-token-row${options.clipped ? " clipped" : ""}` },
      h("span", { class: "ide-token-label", text: label }),
      h("span", { class: "ide-token-value", text: `${format.compact(tokens)} ${unit}` }),
      h("div", { class: "ide-token-bar" },
        h("div", { class: "ide-token-fill", style: { width: `${ratio}%` } })));
  }

  function renderTokenAnalysis(body, project) {
    const budget = project.preview && project.preview.budget;
    if (!budget) {
      Studio.ui.render(body, Studio.ui.loading("Measuring the next request…"));
      return;
    }
    const unit = budget.unit;
    const used = Number(budget.used_tokens) || 0;
    const total = used || 1;
    const tone = budget.over || budget.percent > 100 ? "danger"
      : budget.percent >= 80 ? "warning" : "";
    const children = [
      h("div", { class: "ide-token-summary" },
        h("div", { class: "ide-token-total" },
          h("strong", { text: `~${format.compact(used)} ${unit}` }),
          h("span", { class: "muted",
            text: `of ${format.compact(budget.context_window)} · ${format.percent(budget.percent)}` }),
          Studio.ui.badge(budget.estimator === "bytes" ? "exact bytes" : "estimated", "")),
        h("div", { class: `ide-token-track ${tone}` },
          h("div", { class: "ide-token-track-fill",
            style: { width: `${Math.min(100, Math.max(0, budget.percent))}%` } })),
        h("p", { class: "ide-token-note",
          text: `${budget.model.name || budget.model.id || "No model selected"}`
            + `${budget.model.source ? ` · window from ${{ catalog: "catalog", endpoint: "provider", runtime: "runtime", default: "assumed default", unavailable: "unavailable source" }[budget.model.source] || budget.model.source}` : ""}`
            + ` · ${budget.history.messages} history message(s)` })),
      h("div", { class: "ide-token-list" },
        ...budget.sections.map(section => tokenRow(section.label, section.included_tokens,
          unit, total, { clipped: section.clipped })),
        budget.history.messages
          ? tokenRow(`Conversation (${budget.history.messages} messages)`, budget.history.tokens, unit, total)
          : null,
        tokenRow("Current draft (user message)", budget.draft.tokens, unit, total)),
    ];
    if (budget.warnings && budget.warnings.length) {
      children.push(h("div", { class: "stack" },
        ...budget.warnings.map(warning => Studio.ui.callout(
          warning.tone === "danger" ? "danger" : warning.tone === "warning" ? "warning" : "info",
          h("span", { text: warning.text })))));
    }
    const usage = project.running
      || (project.usage && (project.usage.prompt_tokens || project.usage.completion_tokens))
      ? project.usage : budget.last_usage;
    if (usage && (usage.prompt_tokens || usage.completion_tokens)) {
      children.push(h("p", { class: "ide-token-usage",
        text: `Last run: ${format.number(usage.prompt_tokens || 0)} prompt + `
          + `${format.number(usage.completion_tokens || 0)} completion tokens. `
          + "Provider counts are exact where the model reports them." }));
    }
    if (budget.consolidation && budget.consolidation.count) {
      children.push(h("p", { class: "ide-token-note",
        text: `History consolidated ${budget.consolidation.count} time(s); about `
          + `${format.compact(budget.consolidation.saved_tokens || 0)} ${unit} freed. `
          + `Older turns become a recap at ${budget.auto_consolidate_percent || 80}% context.` }));
    }
    children.push(h("p", { class: "ide-token-note",
      text: `System prompt ceiling ${format.number(budget.limits.prompt_chars)} characters · `
        + `pinned context ${format.number(budget.limits.context_chars)} characters · `
        + `up to ${budget.limits.messages} history messages.` }));
    children.push(h("div", { class: "row wrap" },
      Studio.ui.button("Refresh estimate", { small: true, onclick: async event => {
        const button = event.currentTarget;
        button.disabled = true;
        try {
          await loadContextPreview(project, undefined, { refresh: true });
          renderTokenAnalysis(body, project);
        } catch (error) {
          Studio.toast(error.message, "error");
        } finally {
          button.disabled = false;
        }
      } }),
      Studio.ui.button("Manage context", { small: true, onclick: () => openContextDialog(project) }),
      project.session && project.session.id
        && budget.percent >= (budget.auto_consolidate_percent || 80)
        ? Studio.ui.button("Consolidate now", { small: true, onclick: async event => {
          const button = event.currentTarget;
          button.disabled = true;
          try {
            const result = await api(
              `/api/ide/sessions/${encodeURIComponent(project.session.id)}/consolidate`,
              { body: { model: budget.model.id || state.model || "" } });
            Studio.toast(result.message, result.consolidated ? "success" : "info");
            await loadContextPreview(project, undefined, { refresh: true });
            renderTokenAnalysis(body, project);
          } catch (error) {
            Studio.toast(error.message, "error");
          } finally {
            button.disabled = false;
          }
        } }) : null,
      budget.pinned && budget.pinned.file_count
        ? h("span", { class: "muted tiny",
          text: `${budget.pinned.selected} of ${budget.pinned.file_count} pinned file(s) included` })
        : null));
    Studio.ui.render(body, h("div", { class: "stack" }, children));
  }

  function openTokenDialog(project) {
    if (!project) return;
    const handle = Studio.overlay.modal({
      key: `ide-tokens:${project.id}`, title: "Context and tokens", wide: true,
      subtitle: "The next request, estimated by prompt section. Typing updates the local "
        + "estimate; a run replaces it with provider usage.",
    });
    if (handle.reused) return;
    const body = h("div");
    Studio.ui.render(handle.body, body);
    if (!project.preview || !project.preview.budget) {
      Studio.ui.render(body, Studio.ui.loading("Measuring context…"));
      loadContextPreview(project).then(() => renderTokenAnalysis(body, project)).catch(error => {
        Studio.ui.render(body, Studio.ui.callout("danger", h("span", { text: error.message })));
      });
    } else {
      renderTokenAnalysis(body, project);
    }
  }

  async function openContextDialog(project) {
    const handle = Studio.overlay.modal({
      key: `ide-context:${project.id}`, title: "Agent context", wide: true,
      subtitle: "Pinned paths are added to every prompt. Hidden paths never reach the project map, searches or tool listings.",
    });
    if (handle.reused) return;
    const filter = h("input", { type: "search", placeholder: "Filter files and folders…", "aria-label": "Filter context paths" });
    const results = h("div", { class: "ide-context-results" });
    const pinned = h("div", { class: "ide-context-list" });
    const hidden = h("div", { class: "ide-context-list" });
    const summary = h("p", { class: "muted small" });

    function renderLists() {
      const fileUnit = project.preview && project.preview.budget
        && project.preview.budget.unit === "bytes" ? "B" : "t";
      Studio.ui.render(pinned);
      if (!project.context.include.length) pinned.append(h("p", { class: "muted tiny", text: "Nothing pinned yet." }));
      for (const path of project.context.include) {
        const file = project.preview && (project.preview.files || [])
          .find(item => item.path === path);
        pinned.append(h("div", { class: "ide-context-row" },
          icon("pin", "small"),
          h("span", { class: "grow wrap-anywhere mono tiny", text: path }),
          file ? h("span", { class: "muted tiny", text: `~${format.compact(file.tokens)} ${fileUnit}` }) : null,
          Studio.ui.button("Remove", { small: true, onclick: () => { removeContextPath(project, path); renderLists(); } })));
      }
      Studio.ui.render(hidden);
      const excluded = project.settings.excluded || [];
      if (!excluded.length) hidden.append(h("p", { class: "muted tiny", text: "Nothing is hidden." }));
      for (const path of excluded) {
        hidden.append(h("div", { class: "ide-context-row" },
          icon("eye", "small"),
          h("span", { class: "grow wrap-anywhere mono tiny", text: path }),
          Studio.ui.button("Show again", { small: true, onclick: () => { toggleHidden(project, path).then(renderLists); } })));
      }
      const budget = project.preview && project.preview.budget;
      summary.textContent = project.preview
        ? `${project.preview.selected} file(s) pinned · `
          + (budget
            ? `${format.compact(budget.used_tokens)} / ${format.compact(budget.context_window)} ${budget.unit} next request`
            : `about ${project.preview.tokens} tokens`)
          + ` · ${excluded.length} path(s) hidden`
        : "Estimating context…";
    }

    function resultRow(item) {
      const path = item.path;
      const isPinned = project.context.include.includes(path);
      return h("div", { class: "ide-context-row" },
        icon(item.type === "dir" ? "folder" : "file", "small"),
        h("span", { class: "grow wrap-anywhere mono tiny", text: path }),
        Studio.ui.button(isPinned ? "Pinned" : "Pin", {
          small: true, disabled: isPinned,
          onclick: () => { addContextPath(project, path); renderLists(); },
        }),
        Studio.ui.button("Hide", {
          small: true, kind: "ghost",
          onclick: async () => { await toggleHidden(project, path); renderLists(); },
        }));
    }

    async function renderResults() {
      const query = filter.value.trim();
      Studio.ui.render(results, Studio.ui.loading("Finding paths…"));
      try {
        const data = await api(`/api/ide/workspaces/${encodeURIComponent(project.id)}/files`
          + Studio.query({ query, limit: 120 }));
        Studio.ui.render(results);
        const rows = [
          ...(data.folders || []).map(item => ({ ...item, type: "dir" })),
          ...(data.files || []).map(item => ({ ...item, type: "file" })),
        ];
        if (!rows.length) {
          results.append(h("p", { class: "muted tiny", text: "No path matches this filter." }));
          return;
        }
        for (const item of rows.slice(0, 120)) results.append(resultRow(item));
        if (data.truncated) results.append(h("p", { class: "muted tiny", text: "More paths exist; narrow the filter." }));
      } catch (error) {
        Studio.ui.render(results, Studio.ui.callout("danger", h("span", { text: error.message })));
      }
    }

    let filterTimer = null;
    filter.addEventListener("input", () => {
      clearTimeout(filterTimer);
      filterTimer = setTimeout(renderResults, 220);
    });

    function addTyped() {
      const value = filter.value.trim();
      if (!value) return;
      addContextPath(project, value);
      renderLists();
    }

    Studio.ui.render(handle.body,
      h("div", { class: "stack" },
        h("section", { class: "stack" }, h("strong", { text: "Pinned context" }), pinned),
        h("section", { class: "stack" }, h("strong", { text: "Hidden from the agent" }), hidden),
        h("div", { class: "row wrap" }, Studio.ui.button("Pin typed path", { small: true, onclick: addTyped }),
          Studio.ui.button("Refresh estimate", { small: true, onclick: () => refreshContextPreview(project) })),
        summary,
        filter,
        results));
    handle.footer.append(Studio.ui.button("Done", { kind: "primary", onclick: handle.close }));
    renderLists();
    renderResults();
    refreshContextPreview(project);
    setTimeout(() => { if (summary.isConnected) renderLists(); }, 500);
  }


  /* -- skills --------------------------------------------------------------- */
  function skillsUrl(project, name) {
    const base = `/api/ide/workspaces/${encodeURIComponent(project.id)}/skills`;
    return name ? `${base}/${encodeURIComponent(name)}` : base;
  }

  function useSkill(project, skill, view) {
    const message = $("ide-message");
    const directive = `Use the skill "${skill.id}" (${skill.name}) when it fits. Read it with skill.read first.`;
    message.value = message.value ? `${directive}\n\n${message.value}` : directive;
    onComposerInput();
    if (view) view.close();
    message.focus();
  }

  function openSkillEditor(project, skill) {
    const isNew = !skill;
    const view = Studio.overlay.modal({
      key: `ide-skill-edit:${isNew ? "new" : skill.id}`,
      title: isNew ? "New coding skill" : `Edit ${skill.name}`,
      subtitle: "A skill is a SKILL.md instruction pack the agent reads only when the task matches it.",
      wide: true,
    });
    if (view.reused) return;
    const nameInput = Studio.form.control({
      type: "text", value: isNew ? "" : skill.name || "",
      placeholder: "Code review", required: true,
    });
    const idInput = Studio.form.control({
      type: "text", value: isNew ? "" : skill.id || "",
      placeholder: "code-review (generated from the name when empty)", spellcheck: false,
      disabled: !isNew, readOnly: !isNew,
    });
    const descriptionInput = Studio.form.control({
      type: "text", value: isNew ? "" : skill.description || "",
      placeholder: "What this skill helps with, in one line",
    });
    const scopeInput = isNew ? Studio.form.control({
      type: "select", value: "workspace",
      options: [["workspace", "This project (.agents/skills)"],
        ["user", "Every project (~/.agents/skills)"]],
    }) : null;
    const instructionsInput = Studio.form.control({
      type: "textarea", rows: 16, spellcheck: false,
      value: isNew ? "# When to use this skill\n\n- ...\n\n# Steps\n\n1. ..." : skill.instructions || "",
    });
    const error = h("p", { class: "muted tiny" });
    view.body.append(
      Studio.ui.field("Name", nameInput, "Shown in the skills list and in the agent prompt."),
      Studio.ui.field("Id", idInput, isNew
        ? "Lowercase letters, digits and hyphens. Generated from the name when empty."
        : "The id is fixed; create a new skill to rename a folder."),
      scopeInput ? Studio.ui.field("Scope", scopeInput,
        "Project skills travel with the workspace; user skills apply to every project.") : null,
      Studio.ui.field("Description", descriptionInput, "The agent sees this before loading the full skill."),
      Studio.ui.field("Instructions", instructionsInput, "Markdown. Keep it focused: the agent reads it as project guidance."),
      error,
    );
    const save = Studio.ui.button(isNew ? "Create skill" : "Save skill", {
      kind: "primary",
      onclick: async () => {
        const name = nameInput.value.trim();
        if (!name) {
          error.textContent = "Give the skill a name.";
          nameInput.focus();
          return;
        }
        save.disabled = true;
        error.textContent = "";
        try {
          const body = {
            name,
            description: descriptionInput.value.trim(),
            instructions: instructionsInput.value,
          };
          if (isNew) {
            body.id = idInput.value.trim();
            body.scope = scopeInput.value;
          }
          const result = await api(
            skillsUrl(project, isNew ? "" : skill.id),
            { body, method: isNew ? "POST" : "PUT" });
          Studio.toast(isNew ? `Created ${result.skill.name}.` : `Saved ${result.skill.name}.`, "success");
          openSkills(project);
        } catch (failure) {
          error.textContent = failure.message;
        } finally {
          save.disabled = false;
        }
      },
    });
    view.footer.append(save, Studio.ui.button("Cancel", { onclick: () => {
      view.close();
      openSkills(project);
    } }));
  }

  async function editSkill(project, id) {
    try {
      const data = await api(skillsUrl(project, id));
      openSkillEditor(project, data.skill);
    } catch (error) {
      Studio.toast(error.message, "error");
    }
  }

  async function removeSkill(project, skill) {
    const confirmed = await Studio.confirm({
      title: `Delete the ${skill.name} skill?`,
      message: `Removes ${skill.scope === "workspace" ? ".agents/skills" : "the user skills folder"}/${skill.id}. Files are not affected.`,
      confirmLabel: "Delete skill", danger: true,
    });
    if (!confirmed) return;
    try {
      await api(skillsUrl(project, skill.id), { method: "DELETE" });
      Studio.toast(`Deleted ${skill.name}.`, "success");
      openSkills(project);
    } catch (error) {
      Studio.toast(error.message, "error");
    }
  }

  function openSkills(project) {
    if (!project || !project.id) { Studio.toast("Open a project first.", "info"); return; }
    const view = Studio.overlay.modal({
      key: "ide-skills",
      title: "Coding skills",
      subtitle: "Skills are SKILL.md instruction packs the agent can load with skill.read.",
      wide: true,
    });
    if (view.reused) return;
    const body = h("div", { class: "stack tight" }, Studio.ui.loading("Loading skills…"));
    view.body.append(body);

    async function reload() {
      Studio.ui.render(body, Studio.ui.loading("Loading skills…"));
      try {
        const data = await api(skillsUrl(project));
        const skills = data.skills || [];
        Studio.ui.render(body,
          h("div", { class: "row between wrap" },
            h("span", { class: "muted tiny", text: "Workspace skills live in .agents/skills; user skills apply to every project." }),
            Studio.ui.button("New skill", { small: true, kind: "primary", onclick: () => openSkillEditor(project, null) })));
        if (!skills.length) {
          body.append(Studio.ui.empty({
            compact: true, icon: "sparkle", title: "No skills yet",
            text: "Create one here, or add .agents/skills/<name>/SKILL.md inside the project or ~/.agents/skills/<name>/SKILL.md for every project.",
          }));
          return;
        }
        for (const skill of skills) {
          body.append(h("div", { class: "ide-context-row" },
            h("div", { class: "grow" },
              h("div", { class: "row" }, h("strong", { text: skill.name }),
                Studio.ui.badge(skill.scope, skill.scope === "workspace" ? "accent" : "muted")),
              h("p", { class: "muted tiny", text: skill.description || "No description" }),
              h("code", { class: "tiny", text: skill.id })),
            h("div", { class: "row wrap" },
              Studio.ui.button("Use", { small: true, kind: "primary", onclick: () => useSkill(project, skill, view) }),
              Studio.ui.button("Edit", { small: true, onclick: () => editSkill(project, skill.id) }),
              Studio.ui.button("Delete", { small: true, kind: "ghost",
                onclick: () => removeSkill(project, skill) }))));
        }
      } catch (error) {
        Studio.ui.render(body, Studio.ui.callout("danger", h("span", { text: error.message })));
      }
    }
    view.footer.append(Studio.ui.button("Done", { kind: "primary", onclick: view.close }));
    reload();
  }

  /* -- quick open and @-mentions ------------------------------------------- */
  function openQuickOpen() {
    const project = activeProject();
    if (!project) return;
    const handle = Studio.overlay.modal({
      key: `ide-quick-open:${project.id}`, title: "Quick open",
      subtitle: "Search every file in this project; Enter opens the first result.",
      narrow: false,
    });
    if (handle.reused) return;
    const input = h("input", { type: "search", placeholder: "File name or path…", "aria-label": "Quick open file" });
    const list = h("div", { class: "ide-quick-list" });
    let rows = [];
    let timer = null;
    async function search() {
      try {
        const data = await api(`/api/ide/workspaces/${encodeURIComponent(project.id)}/files`
          + Studio.query({ query: input.value.trim(), limit: 80 }));
        rows = data.files || [];
        Studio.ui.render(list);
        if (!rows.length) {
          list.append(h("p", { class: "muted tiny", text: "No file matches." }));
          return;
        }
        for (const row of rows) {
          list.append(h("button", {
            class: "ide-quick-row", type: "button",
            onclick: () => { handle.close(); openFile(row.path); },
          }, icon("file", "small"), h("span", { class: "grow truncate", text: row.path }),
          h("span", { class: "muted tiny", text: format.bytes(row.bytes) })));
        }
      } catch (error) {
        Studio.ui.render(list, Studio.ui.callout("danger", h("span", { text: error.message })));
      }
    }
    input.addEventListener("input", () => {
      clearTimeout(timer);
      timer = setTimeout(search, 160);
    });
    input.addEventListener("keydown", event => {
      if (event.key !== "Enter") return;
      event.preventDefault();
      if (rows.length) {
        handle.close();
        openFile(rows[0].path);
      }
    });
    Studio.ui.render(handle.body, input, list);
    search();
    input.focus();
  }

  function closeMention(project) {
    if (!project || !project.mention) return;
    project.mention.handle.close();
    project.mention = null;
  }

  function openMention(project, start, query) {
    closeMention(project);
    const input = h("input", { type: "search", value: query || "", placeholder: "Search files to mention…", "aria-label": "Mention a file" });
    const list = h("div", { class: "ide-mention-list" });
    const panel = h("div", { class: "popover ide-mention", role: "dialog", "aria-label": "Mention a file" }, input, list);
    document.body.append(panel);
    let results = [];
    let timer = null;
    let token = 0;

    async function search() {
      const current = ++token;
      try {
        const data = await api(`/api/ide/workspaces/${encodeURIComponent(project.id)}/files`
          + Studio.query({ query: input.value.trim(), limit: 16 }));
        if (current !== token) return;
        results = data.files || [];
        Studio.ui.render(list);
        if (!results.length) {
          list.append(h("p", { class: "muted tiny", text: "No file matches." }));
          return;
        }
        for (const row of results) {
          list.append(h("button", {
            class: "ide-quick-row", type: "button",
            onclick: () => choose(row.path),
          }, icon("file", "small"), h("span", { class: "grow truncate", text: row.path })));
        }
      } catch (error) {
        Studio.ui.render(list, h("p", { class: "muted tiny", text: error.message }));
      }
    }

    function choose(path) {
      const message = $("ide-message");
      const value = message.value;
      const caret = message.selectionStart || value.length;
      message.value = `${value.slice(0, start)}${path}${value.slice(caret)}`;
      message.selectionStart = start + path.length;
      message.selectionEnd = message.selectionStart;
      addContextPath(project, path);
      closeMention(project);
      message.focus();
      message.dispatchEvent(new Event("input", { bubbles: true }));
    }

    input.addEventListener("input", () => {
      clearTimeout(timer);
      timer = setTimeout(search, 160);
    });
    input.addEventListener("keydown", event => {
      if (event.key === "Enter") {
        event.preventDefault();
        if (results.length) choose(results[0].path);
      } else if (event.key === "Escape") {
        event.preventDefault();
        closeMention(project);
      }
    });
    const handle = Studio.overlay.float(panel, $("ide-message"), {
      remove: true, onClose: () => { if (project.mention) project.mention = null; },
    });
    project.mention = { handle };
    search();
    input.focus();
  }

  function onComposerInput() {
    const project = activeProject();
    const input = $("ide-message");
    if (!project || !input) return;
    input.style.height = "auto";
    input.style.height = `${Math.min(220, Math.max(64, input.scrollHeight))}px`;
    project.draft = input.value;
    project.budgetLocked = false;
    updateBudgetMeter(project, input.value);
    const caret = input.selectionStart || 0;
    const before = input.value.slice(0, caret);
    const match = /(^|\s)@([^\s@]*)$/.exec(before);
    if (match && state.viewActive) {
      const start = caret - match[2].length;
      if (!project.mention) openMention(project, start, match[2]);
    } else if (project.mention) {
      closeMention(project);
    }
  }

  /* -- sending, stopping and reverting ------------------------------------- */
  async function sendMessage() {
    const project = activeProject();
    const input = $("ide-message");
    if (!project || !input) return;
    const content = input.value.trim();
    if (!content || project.running) return;
    if (project.voice) stopDictation(project, true);
    try {
      if (project.open.some(tab => tab.dirty && !tab.readonly && !tab.partial)) {
        setStatus(project, "Saving open files before the agent reads them…");
        await saveAllDirty(project);
      }
    } catch (error) {
      Studio.toast(`Open files could not be saved: ${error.message}`, "error");
      setStatus(project, "Save or reload the open files before asking the agent.");
      return;
    }
    const body = {
      message: content,
      mode: state.mode,
      model: $("ide-model").value || state.model || "",
      active_file: project.activePath || "",
      context_files: project.open.map(tab => tab.path)
        .filter(path => path !== project.activePath).slice(0, 4),
      context: sessionContext(project),
      max_steps: 24,
    };
    if (project.session && project.session.id) body.session_id = project.session.id;
    project.lastPrompt = content;
    $("ide-send").disabled = true;
    try {
      const result = await api(`/api/ide/workspaces/${encodeURIComponent(project.id)}/runs`, { body });
      if (!state.data.has(project.id)) return;
      if (!project.session || !project.session.id) {
        project.session = { id: result.session.id, title: result.session.title, transcript: [], changes: {} };
      }
      project.session.id = result.session.id;
      project.sessionId = result.session.id;
      updateSessionSummary(project, result.session);
      if (isActive(project)) {
        $("ide-session-title").textContent = result.session.title || "Session";
        appendEntry(project, { role: "user", text: content });
        input.value = "";
        project.draft = "";
        input.style.height = "";
        project.plan = [];
        renderPlan(project);
        renderSessions(project);
        renderContext(project);
      } else if (project.session) {
        project.session.transcript.push({ role: "user", text: content });
      }
      rememberProject(project);
      followRun(project, result.run.id, 0);
      setStatus(project, "The agent is inspecting the project…");
    } catch (error) {
      Studio.toast(error.message, "error");
      updateWorkspaceState();
    }
  }

  async function stopRun(project) {
    if (!project || !project.runId) return;
    $("ide-stop").disabled = true;
    try {
      await api(`/api/ide/runs/${encodeURIComponent(project.runId)}/cancel`, { body: {} });
      setStatus(project, "Stopping after the current tool…");
    } catch (error) {
      Studio.toast(error.message, "error");
    } finally {
      if (isActive(project)) $("ide-stop").disabled = false;
    }
  }

  async function revertSession() {
    const project = activeProject();
    if (!project || !project.session || !project.session.id) return;
    const confirmed = await Studio.confirm({
      title: "Revert this conversation's edits?",
      message: "Every file the agent changed in this conversation is restored to its state before the first edit. Files it created are deleted.",
      confirmLabel: "Revert edits", danger: true,
    });
    if (!confirmed) return;
    const result = await Studio.run(() => api(
      `${sessionUrl(project.session.id)}/revert`, { body: {} },
    ));
    if (!result) return;
    const skipped = result.skipped
      ? ` ${result.skipped} file(s) changed after the agent and were left untouched.` : "";
    Studio.toast(`Restored ${result.restored} file(s) and removed ${result.deleted} created file(s).${skipped}`,
      result.skipped ? "info" : "success");
    for (const tab of [...project.open]) {
      if (tab.dirty) {
        tab.stale = true;
        continue;
      }
      await reloadOpenFile(project, tab.path);
    }
    renderTabs(project);
    await refreshTree(project, { deep: true });
    await loadSession(project, project.session.id, { quiet: true });
  }

  /* -- layout helpers ------------------------------------------------------- */
  function putIcon(id, name) {
    const element = $(id);
    if (element && !element.children.length) element.append(icon(name, "small"));
  }

  function updateLayoutControls(view) {
    if (!view) return;
    if (state.sidebarOpen !== view.sidebar.visible) {
      state.sidebarOpen = view.sidebar.visible;
      Studio.store.set("ide-sidebar", state.sidebarOpen);
    }
    const sidebarOpen = !view.sidebar.collapsed;
    const agentOpen = !view.agent.collapsed;
    $("ide-sidebar-toggle").setAttribute("aria-pressed", String(sidebarOpen));
    $("ide-agent-toggle").setAttribute("aria-pressed", String(agentOpen));
    $("ide-sidebar-toggle").title = view.sidebar.visible ? "Hide explorer (Ctrl+B)"
      : sidebarOpen ? "Explorer folded for space (Ctrl+B)" : "Show explorer (Ctrl+B)";
    $("ide-sidebar-toggle").setAttribute("aria-label", $("ide-sidebar-toggle").title);
    $("ide-agent-toggle").title = view.agent.visible ? "Hide agent panel (Ctrl+Shift+B)"
      : agentOpen ? "Agent panel folded for space (Ctrl+Shift+B)" : "Show agent panel (Ctrl+Shift+B)";
    $("ide-agent-toggle").setAttribute("aria-label", $("ide-agent-toggle").title);
    $("ide-agent-collapse").disabled = !agentOpen;
    const mode = view.mode === "auto"
      ? `Auto layout${view.stacked ? ", stacked" : ", side by side"}`
      : view.mode === "stacked" ? "Stacked layout" : "Side-by-side layout";
    $("ide-layout-menu").title = `Layout options — ${mode}`;
    $("ide-layout-menu").setAttribute("aria-label", `Layout options: ${mode}`);
  }

  function setupLayout() {
    const previous = Studio.store.get("ide-layout", null);
    layout = Studio.split.create({
      root: $("ide-body"),
      key: "ide-layout",
      gap: 6,
      centerMin: 320,
      centerMinHeight: 240,
      stackAt: 900,
      fallbackWidth: 1200,
      fallbackHeight: 700,
      sidebar: {
        splitter: $("ide-split-sidebar"),
        min: 190, max: 420, defaultSize: 252,
      },
      agent: {
        splitter: $("ide-split-agent"),
        min: 300, max: 680, defaultSize: 380,
        minHeight: 220, maxHeight: 720, defaultHeight: 340,
      },
      onChange: updateLayoutControls,
      onBlocked(name) {
        Studio.toast(name === "sidebar"
          ? "The explorer needs more room: widen the window, reset the layout, or use the layout menu."
          : "The agent panel needs more room: widen the window, stack the panes, or reset the layout.", "info");
      },
    });
    // Keep the pre-layout-preference setting for people upgrading Studio.
    if (!previous) layout.setCollapsed("sidebar", !Studio.store.get("ide-sidebar", true), false);
  }

  function openLayoutMenu(anchor) {
    if (!layout) return;
    const view = layout.read();
    if (!view) return;
    const active = mode => (view.mode === mode ? "On" : null);
    const sidebarAuto = !view.sidebar.visible && !view.sidebar.collapsed;
    const agentAuto = !view.agent.visible && !view.agent.collapsed;
    Studio.overlay.menu(anchor, [
      { heading: "Pane layout" },
      { label: "Auto · stack when narrow", icon: "layout", hint: active("auto"),
        onSelect: () => layout.setMode("auto") },
      { label: "Side by side", icon: "sidebar", hint: active("columns"),
        onSelect: () => layout.setMode("columns") },
      { label: "Stacked · editor above agent", icon: "table", hint: active("stacked"),
        onSelect: () => layout.setMode("stacked") },
      "separator",
      { label: sidebarAuto ? "Explorer folded for space" : view.sidebar.collapsed ? "Show explorer" : "Hide explorer",
        icon: "sidebar", hint: sidebarAuto ? null : "Ctrl+B", disabled: sidebarAuto,
        title: sidebarAuto ? "Widen the window or reset the layout to make room" : null,
        onSelect: () => layout.toggle("sidebar") },
      { label: agentAuto ? "Agent panel folded for space" : view.agent.collapsed ? "Show agent panel" : "Hide agent panel",
        icon: "chat", hint: agentAuto ? null : "Ctrl+Shift+B", disabled: agentAuto,
        title: agentAuto ? "Widen the window, stack the panes or reset the layout to make room" : null,
        onSelect: () => layout.toggle("agent") },
      { label: "Editor only", icon: "file", onSelect: () => {
        layout.setCollapsed("sidebar", true);
        layout.setCollapsed("agent", true);
      } },
      "separator",
      { label: "Reset panel sizes", icon: "refresh", onSelect: () => layout.reset() },
    ], { label: "Layout options", align: "end" });
  }

  function renderAll(project) {
    renderProjects();
    renderTabs(project);
    Studio.ui.render($("ide-search-results"));
    $("ide-search-query").value = "";
    $("ide-file-filter").value = project ? project.filter : "";
    $("ide-session-search").value = project ? project.sessionSearch : "";
    $("ide-message").value = project ? project.draft : "";
    $("ide-message").style.height = "auto";
    $("ide-message").style.height = `${Math.min(220, Math.max(64, $("ide-message").scrollHeight || 64))}px`;
    if (project) {
      renderTree(project);
      renderSessions(project);
      renderTranscript(project);
      renderPlan(project);
      renderContext(project);
      if (!project.preview) $("ide-context").textContent = "Estimating context…";
      mountEditor(project);
      updateBusy(project);
      if (project.session) $("ide-session-title").textContent = project.session.title || "Session";
      else $("ide-session-title").textContent = project.id ? "New session" : "No project";
    } else {
      Studio.ui.render($("ide-file-tabs"));
      renderViewBar(null);
      $("ide-file-notice").classList.add("hidden");
      Studio.clear($("ide-file-notice"));
      Studio.ui.render($("ide-tree"), Studio.ui.empty({
        compact: true, icon: "folder", title: "No project open",
        text: "Open a project from the toolbar, or import a folder.",
        actions: [Studio.ui.button("New workspace", { small: true, kind: "primary", onclick: newWorkspace })],
      }));
      Studio.ui.render($("ide-sessions"));
      Studio.ui.render($("ide-transcript"));
      Studio.ui.render($("ide-plan"));
      Studio.ui.render($("ide-context-chips"));
      $("ide-context").textContent = "Open a project to define agent context.";
      Studio.ui.render($("ide-editor-host"), Studio.ui.empty({
        compact: true, icon: "file", title: "No project open",
        text: "Create a workspace or open a project to begin.",
      }));
      $("ide-session-title").textContent = "No project";
    }
    updateWorkspaceState();
  }

  /* -- wiring --------------------------------------------------------------- */
  Studio.views.register("ide", {
    title: "IDE", subtitle: "Agentic coding across several projects at once",
    glyph: "ide", hint: "Plan, edit and verify code with a coding agent",
    init() {
      if (state.voiceAuto && Studio.voice && !Studio.voice.synthesisSupported()) {
        state.voiceAuto = false;
        Studio.store.set("ide-voice-auto", false);
      }
      for (const [id, name] of Object.entries({
        "ide-sidebar-toggle": "sidebar", "ide-layout-menu": "layout", "ide-agent-toggle": "chat",
        "ide-agent-collapse": "chevron-right", "ide-voice": "mic", "ide-voice-auto": "volume",
        "ide-refresh": "refresh", "ide-new-file": "plus",
        "ide-new-folder": "folder", "ide-provider-menu": "globe", "ide-quick-open": "search",
        "ide-session-menu": "more", "ide-context-add": "pin", "ide-context-manage": "settings",
        "ide-prompt-menu": "sparkle", "ide-delete-session": "trash", "ide-search-run": "search",
      })) putIcon(id, name);

      const sidebarTabs = Studio.ui.tabs($("ide-files-panel").parentElement.querySelector(".ide-sidebar-tabs"));
      sidebarTabs.select("files", false);
      const mode = Studio.ui.segmented($("ide-mode"), value => {
        state.mode = value;
        Studio.store.set("ide-mode", value);
        updateModeBadge();
        const project = activeProject();
        if (project) refreshContextPreview(project);
      });
      if (!["ask", "edit", "auto"].includes(state.mode)) state.mode = "edit";
      mode.select(state.mode, false);
      updateModeBadge();

      const hidden = Studio.store.get("ide-hidden-workspaces", []);
      state.hiddenProjects = new Set(Array.isArray(hidden) ? hidden : []);
      $("ide-workspace").addEventListener("change", event => {
        const target = event.target.value;
        if (target === "__manage__") {
          event.target.value = state.active || "";
          openProjectManager();
          return;
        }
        if (target) openProject(target);
        else event.target.value = state.active || "";
      });
      $("ide-model").addEventListener("change", event => {
        state.model = event.target.value;
        Studio.store.set("ide-model", state.model);
        const project = activeProject();
        if (project) {
          project.model = state.model;
          Studio.store.set(`ide-model-${project.id}`, state.model);
          refreshContextPreview(project);
        }
      });
      $("ide-provider-menu").addEventListener("click", event => Studio.providers.menu(event.currentTarget));
      $("ide-quick-open").addEventListener("click", openQuickOpen);
      $("ide-new-workspace").addEventListener("click", newWorkspace);
      $("ide-open-folder").addEventListener("click", openFolder);
      $("ide-refresh").addEventListener("click", () => {
        const project = activeProject();
        if (project) Studio.run(() => refreshTree(project, { deep: true }));
      });
      $("ide-sidebar-toggle").addEventListener("click", () => {
        if (layout) layout.toggle("sidebar");
      });
      $("ide-agent-toggle").addEventListener("click", () => {
        if (layout) layout.toggle("agent");
      });
      $("ide-agent-collapse").addEventListener("click", () => {
        if (layout) layout.toggle("agent");
      });
      $("ide-layout-menu").addEventListener("click", event => openLayoutMenu(event.currentTarget));
      $("ide-file-filter").addEventListener("input", onFilterInput);
      $("ide-new-file").addEventListener("click", () => newFile(activeProject(), ""));
      $("ide-new-folder").addEventListener("click", () => newFolder(activeProject(), ""));
      $("ide-search-run").addEventListener("click", runSearch);
      $("ide-search-query").addEventListener("keydown", event => {
        if (event.key === "Enter") { event.preventDefault(); runSearch(); }
      });
      $("ide-session-search").addEventListener("input", event => {
        const project = activeProject();
        if (!project) return;
        project.sessionSearch = event.target.value;
        renderSessions(project);
      });
      $("ide-new-session").addEventListener("click", () => newSession(activeProject()));
      $("ide-delete-session").addEventListener("click", () => {
        const project = activeProject();
        if (project && project.session) deleteSession(project, project.session.id);
        else Studio.toast("Open a conversation first.", "info");
      });
      $("ide-session-menu").addEventListener("click", event => {
        const project = activeProject();
        if (!project) return;
        const session = project.session || { id: "", title: "New session", pinned: false };
        Studio.overlay.menu(event.currentTarget, [
          { label: "New conversation", icon: "plus", onSelect: () => newSession(project) },
          { label: "Rename…", icon: "edit", disabled: !session.id, onSelect: () => renameSession(project, session) },
          { label: session.pinned ? "Unpin" : "Pin to top", icon: "pin", disabled: !session.id,
            onSelect: () => pinSession(project, session, !session.pinned) },
          { label: "Duplicate", icon: "copy", disabled: !session.id || project.running, onSelect: () => duplicateSession(project, session.id) },
          { label: "Export as Markdown", icon: "download", disabled: !session.id, onSelect: () => exportSession(project, session.id) },
          "separator",
          { label: "Delete conversation…", icon: "trash", danger: true, disabled: !session.id,
            onSelect: () => deleteSession(project, session.id) },
          { label: "Clean IDE session…", icon: "trash", danger: true,
            onSelect: () => Studio.run(cleanSession) },
        ], { label: "Conversation actions", align: "end" });
      });
      $("ide-context-add").addEventListener("click", () => {
        const project = activeProject();
        if (project) openContextDialog(project);
        else Studio.toast("Open a project first.", "info");
      });
      $("ide-context-manage").addEventListener("click", () => {
        const project = activeProject();
        if (project) openContextDialog(project);
        else Studio.toast("Open a project first.", "info");
      });
      $("ide-voice").addEventListener("click", () => toggleDictation(activeProject()));
      $("ide-voice-auto").addEventListener("click", () => {
        if (!Studio.voice || !Studio.voice.synthesisSupported()) {
          Studio.toast("This browser cannot read replies aloud.", "info");
          return;
        }
        state.voiceAuto = !state.voiceAuto;
        Studio.store.set("ide-voice-auto", state.voiceAuto);
        updateVoiceControls(activeProject());
        if (!state.voiceAuto) Studio.voice.stopSpeaking();
        else Studio.toast("Replies will be read aloud after each completed run.", "info");
      });
      $("ide-budget-open").addEventListener("click", () => openTokenDialog(activeProject()));
      $("ide-message").addEventListener("blur", () => {
        const project = activeProject();
        if (project) refreshContextPreview(project);
      });
      $("ide-prompt-menu").addEventListener("click", event => {
        Studio.overlay.menu(event.currentTarget, [
          ...PROMPTS.map(([label, text, promptMode]) => ({
            label, icon: "sparkle",
            onSelect: () => {
              $("ide-message").value = text;
              if (promptMode && promptMode !== state.mode) {
                state.mode = promptMode;
                Studio.store.set("ide-mode", promptMode);
                mode.select(promptMode, false);
                updateModeBadge();
              }
              onComposerInput();
              $("ide-message").focus();
            },
          })),
          "separator",
          { label: "Coding skills…", icon: "sparkle", onSelect: () => openSkills(activeProject()) },
        ], { label: "Prompt presets", align: "end" });
      });
      $("ide-save").addEventListener("click", saveActive);
      $("ide-revert").addEventListener("click", revertSession);
      $("ide-check").addEventListener("click", checkProject);
      $("ide-send").addEventListener("click", sendMessage);
      $("ide-stop").addEventListener("click", () => stopRun(activeProject()));
      $("ide-message").addEventListener("input", onComposerInput);
      $("ide-message").addEventListener("keydown", event => {
        if (event.isComposing) return;
        if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
          event.preventDefault();
          sendMessage();
        }
      });
      $("ide-tree").addEventListener("scroll", () => {
        const project = activeProject();
        if (!project) return;
        project.scrollTop = $("ide-tree").scrollTop;
        if (state.treeFrame !== null) return;
        if (typeof requestAnimationFrame !== "function") {
          paintTreeWindow(project);
          return;
        }
        state.treeFrame = requestAnimationFrame(() => {
          state.treeFrame = null;
          paintTreeWindow(activeProject());
        });
      });
      $("ide-tree").addEventListener("contextmenu", event => {
        if (event.target.closest(".ide-tree-row")) return;
        const project = activeProject();
        if (!project) return;
        event.preventDefault();
        Studio.overlay.menu({ x: event.clientX, y: event.clientY }, [
          { label: "New file…", icon: "plus", onSelect: () => newFile(project, "") },
          { label: "New folder…", icon: "folder", onSelect: () => newFolder(project, "") },
          "separator",
          { label: "Refresh explorer", icon: "refresh", onSelect: () => refreshTree(project, { deep: true }) },
          { label: "Quick open…", icon: "search", onSelect: openQuickOpen },
        ], { label: "Project actions" });
      });

      Studio.shortcut("mod+s", () => {
        if (Studio.views.current !== "ide") return false;
        saveActive();
      });
      Studio.shortcut("mod+p", () => {
        if (Studio.views.current !== "ide") return false;
        openQuickOpen();
      });
      Studio.shortcut("mod+w", () => {
        if (Studio.views.current !== "ide") return false;
        const project = activeProject();
        const tab = project && activeTab(project);
        if (tab) closeTab(project, tab.path);
      });
      Studio.shortcut("mod+b", () => {
        if (Studio.views.current !== "ide" || !layout) return false;
        layout.toggle("sidebar");
      });
      Studio.shortcut("mod+shift+b", () => {
        if (Studio.views.current !== "ide" || !layout) return false;
        layout.toggle("agent");
      });
      window.addEventListener("beforeunload", event => {
        if (state.projects.some(id => (state.data.get(id) || { open: [] }).open.some(tab => tab.dirty))) {
          event.preventDefault();
          event.returnValue = "";
        }
      });
      setupLayout();
      if (Studio.voice) {
        Studio.voice.status().then(value => {
          state.voiceStatus = value;
          updateVoiceControls(activeProject());
        }).catch(() => { /* voice status is optional */ });
      }
      Studio.ui.render($("ide-editor-host"), Studio.ui.loading("Opening the IDE…"));
      Studio.on("models", () => {
        if (!state.loaded) return;
        fillModelSelect();
      });
    },

    async activate(parameters) {
      state.viewActive = true;
      if (layout) layout.apply();
      try {
        if (!state.loaded) await loadOverview();
      } catch (error) {
        Studio.ui.render($("ide-tree"), Studio.ui.empty({
          icon: "alert", title: "Could not open the IDE", text: error.message,
          actions: [Studio.ui.button("Try again", { kind: "primary", onclick: () => Studio.run(loadOverview) })],
        }));
        return;
      }
      const wanted = parameters.project || parameters.workspace;
      if (wanted) await openProject(wanted);
      else if (!activeProject() && state.projects.length) await activateProject(state.projects[0]);
      if (parameters.session) {
        const project = activeProject();
        if (project) await loadSession(project, parameters.session);
      }
      const project = activeProject();
      if (project && project.running && project.runId && !project.feed) {
        followRun(project, project.runId, project.lastSeq);
      }
    },

    deactivate() {
      state.viewActive = false;
      for (const project of state.data.values()) {
        closeFeed(project);
        stopDictation(project, true);
      }
      if (Studio.voice) Studio.voice.stopSpeaking();
    },
  });
})();
