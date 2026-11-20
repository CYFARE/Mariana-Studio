"use strict";

(function () {
  const Studio = window.Studio;
  const { $, h, icon, format, api } = Studio;
  const state = { jobs: [], feed: null, open: null, started: false, rows: new Map(), signature: null };

  function sorted() {
    return state.jobs.slice().sort((a, b) => (b.created || 0) - (a.created || 0));
  }

  function activeJobs() {
    return state.jobs.filter(job => job.active || ["queued", "preparing", "running", "paused"].includes(job.status));
  }

  function jobSignature(jobs) {
    return jobs.map(job => [
      job.id, job.status, Boolean(job.active), job.kind, job.title,
      Boolean(job.error), job.meta ? job.meta.view || "" : "",
    ].join(":")).join("|");
  }

  /* One row that updates its own text and bar in place.  The job feed emits on
     every trainer log line, so rebuilding the list would restart transitions
     and steal focus. */
  function createRow(job) {
    const status = h("span");
    const stage = h("span");
    const percent = h("span");
    const error = h("div", { class: "job-error hidden" });
    const bar = job.active || job.status === "completed" ? Studio.ui.progress(job.title) : null;
    const action = job.active ? h("button", {
      class: "icon-button small", type: "button", "aria-label": `Cancel ${job.title}`, title: "Cancel",
      onclick: () => Studio.run(() => api(`/api/jobs/${encodeURIComponent(job.id)}/cancel`, { body: {} })),
    }, icon("stop", "small")) : h("button", {
      class: "icon-button small", type: "button", "aria-label": `Dismiss ${job.title}`, title: "Dismiss",
      onclick: () => Studio.run(() => api(`/api/jobs/${encodeURIComponent(job.id)}`, { method: "DELETE" })),
    }, icon("x", "small"));
    const root = h("article", { class: "job-row" },
      h("div", { class: "job-top" }, status,
        h("button", {
          class: "link-button job-title", type: "button", title: job.title,
          onclick: () => {
            if (state.open) state.open.close();
            if (job.kind === "dependency") { Studio.dependencies.open(); return; }
            const view = job.meta && job.meta.view;
            Studio.views.go(view || (job.finetune ? "finetune" : "train"), { job: job.id });
          },
        }, job.title || job.kind),
        action),
      h("div", { class: "job-meta" }, stage, percent),
      bar,
      error);

    function update(current) {
      const statusKey = current.status || (current.active ? "running" : "idle");
      if (status.dataset.status !== statusKey) {
        status.dataset.status = statusKey;
        status.replaceChildren(Studio.ui.status(statusKey));
      }
      stage.textContent = current.stage || format.title(current.kind);
      const progress = current.progress || { percent: 0 };
      percent.textContent = `${Math.round(progress.percent || 0)}% · ${format.duration(current.elapsed)}`;
      if (bar) bar.update(progress.percent || 0, current.status === "failed" ? "failed" : current.status === "completed" ? "done" : "active");
      error.textContent = current.error || "";
      error.classList.toggle("hidden", !current.error);
    }

    update(job);
    return { root, update };
  }

  function render() {
    const jobs = sorted();
    const active = activeJobs();
    $("jobs-count").textContent = String(active.length);
    $("jobs-count").classList.toggle("hidden", active.length === 0);
    $("jobs-button").setAttribute("aria-label", active.length ? `Jobs, ${active.length} active` : "Jobs");
    $("jobs-clear").disabled = !state.jobs.some(job => !job.active);
    const next = jobSignature(jobs);
    if (next === state.signature) {
      for (const job of jobs) {
        const row = state.rows.get(job.id);
        if (row) row.update(job);
      }
      return;
    }
    state.signature = next;
    state.rows.clear();
    const list = Studio.ui.render($("jobs-list"));
    if (!jobs.length) {
      list.append(Studio.ui.empty({ compact: true, icon: "jobs", title: "No jobs", text: "Training, downloads and exports will appear here." }));
      return;
    }
    for (const job of jobs) {
      const row = createRow(job);
      state.rows.set(job.id, row);
      list.append(row.root);
    }
  }

  function accept(data) {
    state.jobs = data.jobs || [];
    render();
    Studio.emit("jobs", sorted());
    for (const job of state.jobs) Studio.emit(`job:${job.kind}`, job);
  }

  async function refresh() {
    accept(await api("/api/jobs"));
    return state.jobs;
  }

  function start() {
    if (state.started) return;
    state.started = true;
    state.feed = Studio.subscribe("/api/jobs/events", accept);
    $("jobs-button").addEventListener("click", () => {
      if (state.open) {
        state.open.close();
        return;
      }
      render();
      state.open = Studio.overlay.float($("jobs-popover"), $("jobs-button"), {
        align: "end", toggle: $("jobs-button"),
        onClose: () => { state.open = null; $("jobs-button").setAttribute("aria-expanded", "false"); },
      });
      $("jobs-button").setAttribute("aria-expanded", "true");
    });
    $("jobs-clear").addEventListener("click", async event => {
      if (!await Studio.confirm({ title: "Clear finished jobs?", message: "This removes completed, failed and cancelled jobs from the list. Output files are kept.", confirmLabel: "Clear finished" })) return;
      Studio.run(() => api("/api/jobs/clear", { body: {} }), event.currentTarget);
    });
    refresh().catch(error => Studio.toast(error.message, "error"));
  }

  Studio.jobs = { start, refresh, list: sorted, active: activeJobs, get: id => state.jobs.find(job => job.id === id) || null };
})();
