"use strict";

/* Studio.dependencies: detect optional Python requirement groups and install
   them in one background pip job.  The panel is non-modal: Studio stays usable
   while pip works and the job is also visible in the job centre. */
(function () {
  const Studio = window.Studio;
  const { $, h, api, store, toast } = Studio;
  const state = { status: null, job: null, dismissed: false, hidden: true, started: false, preparing: false, renderKey: null };
  const TERMINAL = new Set(["failed", "cancelled", "completed"]);
  const FAILED_KEY = "dependencies-failed";

  function signature(status) {
    return status ? status.missing.join(",") : "";
  }

  function running(job) {
    return Boolean(job && job.active);
  }

  function show() {
    state.hidden = false;
    $("dependencies-panel").classList.remove("hidden");
  }

  function hide() {
    state.hidden = true;
    $("dependencies-panel").classList.add("hidden");
  }

  function dismiss() {
    if (state.status) store.set("dependencies-dismissed", signature(state.status));
    state.dismissed = true;
    hide();
  }

  async function install(groups) {
    const wanted = groups && groups.length ? groups : (state.status ? state.status.missing : []);
    if (!wanted.length) { await check({ show: true }); return; }
    state.preparing = true;
    store.set(FAILED_KEY, "");
    show();
    render();
    try {
      const data = await api("/api/dependencies/install", { body: { groups: wanted } });
      state.job = data.job || null;
      store.set("dependencies-dismissed", "");
      state.dismissed = false;
      state.preparing = false;
      if (state.job) { show(); render(); }
      else await check({ show: true });
    } catch (error) {
      state.preparing = false;
      render();
      toast(error.message, "error");
    }
  }

  async function check(options = {}) {
    let status;
    try {
      status = await api("/api/dependencies");
    } catch (error) {
      if (options.report) toast(error.message, "error");
      return;
    }
    state.status = status;
    state.dismissed = store.get("dependencies-dismissed", "") === signature(status);
    const failedBefore = store.get(FAILED_KEY, "") === signature(status);
    state.job = status.job || null;

    if (status.busy && status.job) { show(); render(); return; }
    if (status.errors && status.errors.length) { show(); render(); return; }
    if (!status.missing.length) {
      if (options.show) { show(); render(); }
      else hide();
      return;
    }
    if (status.job && TERMINAL.has(status.job.status) && (!state.dismissed || options.show)) {
      show(); render(); return;
    }
    if (status.auto_install && status.pip !== false && !state.dismissed && !failedBefore && !options.noInstall) {
      await install(status.missing);
      return;
    }
    if (!state.dismissed || options.show) { show(); render(); }
    else hide();
  }

  async function stop() {
    if (!state.job || !state.job.active) return;
    if (!await Studio.confirm({
      title: "Stop installing dependencies?",
      message: "pip will be stopped. Packages that already finished installing stay installed.",
      confirmLabel: "Stop installation", danger: true,
    })) return;
    try {
      const data = await api(`/api/jobs/${encodeURIComponent(state.job.id)}/cancel`, { body: {} });
      state.job = data.job;
      render();
    } catch (error) {
      toast(error.message, "error");
    }
  }

  function computeKey() {
    const status = state.status;
    const job = state.job;
    return JSON.stringify({
      missing: status ? status.missing : [],
      errors: status ? status.errors || [] : [],
      groups: status ? status.groups.map(group => [group.id, group.missing.length, Boolean(group.error)]) : [],
      preparing: state.preparing,
      dismissed: state.dismissed,
      job: job ? [job.id, job.status, job.error, job.meta ? job.meta.group : "",
        job.meta ? job.meta.groups || [] : []] : null,
    });
  }

  /* The title, tone and subtitle only change when the installation state does.
     Progress updates reuse them so the panel does not rebuild on every pip line. */
  function panelState() {
    const status = state.status;
    const job = state.job;
    const active = running(job);
    const terminal = job && TERMINAL.has(job.status);
    const missing = new Set(status.missing);
    const errors = new Set(status.errors || []);
    const anyError = errors.size > 0;
    let title, subtitle, tone = "";

    if (state.preparing) {
      title = "Preparing dependency installer";
      subtitle = "Studio stays usable while pip works in the background.";
      tone = "preparing";
    } else if (active) {
      title = "Installing dependencies";
      subtitle = "Studio stays usable while pip works in the background.";
      tone = "running";
    } else if (job && job.status === "failed") {
      title = "Dependency installation failed";
      subtitle = job.error || "pip could not finish. Retry or check the job centre.";
      tone = "failed";
    } else if (job && job.status === "cancelled") {
      title = "Dependency installation stopped";
      subtitle = "Packages that already finished installing stay installed.";
    } else if (anyError) {
      title = "Requirement files are missing";
      subtitle = "Restore the missing requirements file, then check again.";
      tone = "failed";
    } else if (status.missing.length && status.pip === false) {
      title = "pip is unavailable";
      subtitle = "Install pip in this Python environment, then reopen Studio.";
      tone = "failed";
    } else if (job && job.status === "completed" && status.missing.length) {
      title = "Some dependencies are still unavailable";
      subtitle = "Retry, or open the job centre to inspect the pip output.";
      tone = "failed";
    } else if (job && job.status === "completed") {
      title = "Dependencies installed";
      subtitle = "Reload Studio to activate every feature.";
      tone = "completed";
    } else if (status.missing.length) {
      title = "Optional Python features are missing";
      subtitle = `${status.missing.length} requirement group${status.missing.length === 1 ? "" : "s"} can be installed without leaving Studio.`;
    } else {
      title = "Python features are ready";
      subtitle = "Every optional requirement group is satisfied.";
      tone = "completed";
    }
    return { title, subtitle, tone, active, terminal, missing, errors, anyError };
  }

  function renderProgress(view) {
    const status = state.status;
    const job = state.job;
    let percent = 0;
    if (view.active || state.preparing) percent = state.preparing ? 0 : (job.progress.percent || 0);
    else if (view.tone === "completed") percent = 100;
    else if (view.terminal) percent = (job.progress && job.progress.percent) || 0;
    percent = Math.max(0, Math.min(100, percent));
    $("dependencies-bar").style.width = `${percent}%`;
    $("dependencies-percent").textContent = `${Math.round(percent)}%`;
    $("dependencies-progress").setAttribute("aria-valuenow", String(Math.round(percent)));

    let stage = "Ready";
    if (state.preparing) stage = "Preparing pip…";
    else if (view.active) stage = job.stage || "Installing";
    else if (job && job.status === "failed") stage = "Stopped with an error";
    else if (job && job.status === "cancelled") stage = "Stopped";
    else if (view.anyError) stage = "Missing requirement files";
    else if (job && job.status === "completed" && status.missing.length) stage = "Some groups remain missing";
    else if (job && job.status === "completed") stage = "All groups installed";
    else if (status.missing.length) stage = `${status.missing.length} group${status.missing.length === 1 ? "" : "s"} missing`;
    $("dependencies-stage").textContent = stage;
  }

  function render() {
    const status = state.status;
    const job = state.job;
    if (!status) return;

    const view = panelState();
    const currentGroup = job && job.meta ? job.meta.group : null;
    const selected = new Set((job && job.meta && job.meta.groups) || []);

    $("dependencies-title").textContent = view.title;
    $("dependencies-subtitle").textContent = view.subtitle;
    const panel = $("dependencies-panel");
    panel.classList.remove("preparing", "running", "failed", "completed");
    if (view.tone) panel.classList.add(view.tone);
    panel.setAttribute("aria-busy", String(state.preparing || view.active));

    Studio.ui.render($("dependencies-groups"), status.groups.map(group => {
      let groupTone = "installed";
      if (group.error || view.missing.has(group.id)) {
        if (view.active && selected.has(group.id)) groupTone = currentGroup === group.id ? "installing" : "queued";
        else groupTone = "missing";
      }
      const detail = group.error
        || (group.missing.length ? `Missing: ${group.missing.join(", ")}` : "Installed");
      return h("span", {
        class: `install-group ${groupTone}`.trim(), title: detail,
        text: `${group.title}${group.missing.length ? ` · ${group.missing.length}` : ""}${group.error ? " · missing file" : ""}`,
      });
    }));

    renderProgress(view);

    const actions = Studio.ui.render($("dependencies-actions"));
    if (state.preparing) {
      actions.append(Studio.ui.button("Hide", { small: true, kind: "ghost", onclick: hide }));
    } else if (active) {
      actions.append(Studio.ui.button("Hide", { small: true, kind: "ghost", onclick: hide }));
      actions.append(Studio.ui.button("Stop", { small: true, kind: "danger", onclick: stop }));
    } else if (job && job.status === "failed") {
      actions.append(Studio.ui.button("Retry", { small: true, kind: "primary", onclick: () => install(job.meta.groups || status.missing) }));
      actions.append(Studio.ui.button("Job centre", { small: true, kind: "ghost", onclick: () => { hide(); $("jobs-button").click(); } }));
      actions.append(Studio.ui.button("Dismiss", { small: true, kind: "ghost", onclick: dismiss }));
    } else if (job && job.status === "cancelled") {
      actions.append(Studio.ui.button("Try again", { small: true, kind: "primary", onclick: () => install(job.meta.groups || status.missing) }));
      actions.append(Studio.ui.button("Dismiss", { small: true, kind: "ghost", onclick: dismiss }));
    } else if (job && job.status === "completed" && status.missing.length) {
      actions.append(Studio.ui.button("Retry", { small: true, kind: "primary", onclick: () => install(status.missing) }));
      actions.append(Studio.ui.button("Dismiss", { small: true, kind: "ghost", onclick: dismiss }));
    } else if (job && job.status === "completed") {
      actions.append(Studio.ui.button("Reload Studio", { small: true, kind: "primary", onclick: () => window.location.reload() }));
      actions.append(Studio.ui.button("Dismiss", { small: true, kind: "ghost", onclick: dismiss }));
    } else if (anyError) {
      actions.append(Studio.ui.button("Dismiss", { small: true, kind: "ghost", onclick: dismiss }));
    } else if (status.missing.length && status.pip === false) {
      actions.append(Studio.ui.button("Dismiss", { small: true, kind: "ghost", onclick: dismiss }));
    } else if (status.missing.length) {
      actions.append(Studio.ui.button("Install missing", { small: true, kind: "primary", onclick: () => install(status.missing) }));
      actions.append(Studio.ui.button("Not now", { small: true, kind: "ghost", onclick: dismiss }));
    } else {
      actions.append(Studio.ui.button("Close", { small: true, kind: "ghost", onclick: dismiss }));
    }
    state.renderKey = computeKey();
  }

  function updateJob(job) {
    if (state.job && state.job.id !== job.id) {
      if (state.job.active || !job.active) return;
    }
    if (!state.job && !job.active) return;
    const previous = state.job ? state.job.status : null;
    const wasHidden = state.hidden;
    state.job = job;
    if (!previous && job.active) show();
    if (computeKey() !== state.renderKey) render();
    else if (state.status) renderProgress(panelState());
    if (!job.active && previous !== job.status) {
      if (job.status === "completed") {
        store.set(FAILED_KEY, "");
        Studio.models.load(true).catch(() => {});
        if (wasHidden) toast("Python dependencies installed. Reload Studio to activate every feature.", "success", 9000);
        check({ show: !wasHidden, noInstall: true });
      } else if (job.status === "failed" || job.status === "cancelled") {
        store.set(FAILED_KEY, signature(state.status));
        if (job.status === "failed" && wasHidden) toast(job.error || "Dependency installation failed.", "error");
      }
    }
  }

  function start() {
    if (state.started) return;
    state.started = true;
    $("dependencies-close").addEventListener("click", () => {
      if (state.preparing || running(state.job)) hide();
      else dismiss();
    });
    Studio.on("job:dependency", updateJob);
    check();
  }

  async function open() {
    state.dismissed = false;
    store.set("dependencies-dismissed", "");
    await check({ show: true, noInstall: true });
  }

  Studio.dependencies = { start, open, check };
})();
