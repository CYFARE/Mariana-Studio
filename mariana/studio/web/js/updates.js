"use strict";

/* Studio.updates: compare the checkout with the newest GitHub release after
   boot, then fast-forward it with one background git job.  The panel is
   non-modal and only appears when a release, a restart or a manual check asks
   for attention. */
(function () {
  const Studio = window.Studio;
  const { $, h, api, store, toast } = Studio;
  const state = { status: null, job: null, started: false, hidden: true,
    applying: false, polls: 0, timer: null, dismissedJob: null, hiddenJob: null, ignoredJob: null };
  const TERMINAL = new Set(["failed", "cancelled", "completed"]);
  const DISMISSED = "updates-dismissed";
  const MAX_POLLS = 24;
  const POLL_MS = 1200;

  function releaseVersion(status) {
    return status && status.latest_version ? String(status.latest_version) : "";
  }

  function dismissed(status) {
    return Boolean(status) && store.get(DISMISSED, "") === releaseVersion(status);
  }

  function show() {
    state.hidden = false;
    state.hiddenJob = null;
    $("update-panel").classList.remove("hidden");
  }

  function hide() {
    state.hidden = true;
    if (state.job && state.job.active) state.hiddenJob = state.job.id;
    $("update-panel").classList.add("hidden");
  }

  function dismiss() {
    if (state.status && state.status.update_available) store.set(DISMISSED, releaseVersion(state.status));
    if (state.job) state.dismissedJob = state.job.id;
    hide();
  }

  function stopPolling() {
    clearTimeout(state.timer);
    state.timer = null;
  }

  function schedule() {
    stopPolling();
    if (state.polls >= MAX_POLLS) {
      if (state.status && state.status.checking) state.status = { ...state.status, checking: false };
      render();
      return;
    }
    state.polls += 1;
    state.timer = setTimeout(() => {
      refresh().catch(error => {
        if (state.status) state.status = { ...state.status, checking: false, error: error.message };
        render();
      });
    }, POLL_MS);
  }

  function maybeShow() {
    const status = state.status;
    const job = state.job;
    if (state.applying) { show(); return; }
    if (job && job.id !== state.dismissedJob
      && (job.active || job.status === "completed" || job.status === "failed")) {
      show();
      return;
    }
    if (job && job.id === state.dismissedJob) return;
    if (status && status.update_available && !dismissed(status)) show();
    else if (status && status.restart_required) show();
  }

  async function refresh() {
    const status = await api("/api/updates");
    state.status = status;
    if (status.job && status.job.id !== state.ignoredJob) state.job = status.job;
    if (status.checking) schedule();
    else stopPolling();
    maybeShow();
    render();
    return status;
  }

  async function open() {
    state.polls = 0;
    state.dismissedJob = null;
    if (state.job && !state.job.active) state.ignoredJob = state.job.id;
    state.job = null;
    show();
    render();
    try {
      state.status = await api("/api/updates/check", { body: {} });
      if (state.status.job && state.status.job.id !== state.ignoredJob) state.job = state.status.job;
      if (state.status.checking) schedule();
      render();
    } catch (error) {
      toast(error.message, "error");
      if (!state.status) hide();
    }
  }

  async function apply() {
    const status = state.status;
    if (!status) return;
    if (!status.git || !status.git.checkout) {
      if (status.releases_url) window.open(status.releases_url, "_blank", "noopener");
      return;
    }
    if (!status.git.available) {
      toast("git is not installed, so Studio cannot update this checkout. Install git and try again.", "error");
      return;
    }
    const target = releaseVersion(status);
    if (!await Studio.confirm({
      title: "Update Mariana Studio?",
      message: `Studio will fetch version ${target} from GitHub and fast-forward this git checkout. `
        + "Running jobs are not restarted; save your work and restart the server afterwards.",
      confirmLabel: "Update now",
    })) return;
    state.applying = true;
    state.dismissedJob = null;
    show();
    render();
    try {
      const data = await api("/api/updates/apply", { body: {} });
      state.job = data.job || null;
      if (!data.started) toast("An update is already running.", "info");
      render();
    } catch (error) {
      state.applying = false;
      toast(error.message, "error");
      render();
    }
  }

  function updateJob(job) {
    if (!job || job.id === state.ignoredJob) return;
    if (state.job && state.job.id !== job.id && !job.active) return;
    state.job = job;
    if (job.active) {
      state.applying = true;
      if (job.id !== state.hiddenJob) show();
    } else if (TERMINAL.has(job.status)) {
      state.applying = false;
      state.hiddenJob = null;
      if (job.status === "completed") {
        const version = job.result && job.result.version ? ` ${job.result.version}` : "";
        toast(`Mariana Studio${version} installed. Restart the server to use it.`, "success", 9000);
        refresh().catch(() => render());
      } else if (job.status === "failed") {
        toast(job.error || "The update failed.", "error");
      }
      show();
    }
    render();
  }

  function percentOf(job) {
    return job && job.progress ? Number(job.progress.percent) || 0 : 0;
  }

  function view() {
    const status = state.status;
    const job = state.job;
    if (job && job.active) {
      return {
        title: "Updating Mariana Studio",
        subtitle: "The checkout is fast-forwarding to the latest release. Studio stays usable until you restart it.",
        tone: "running", percent: percentOf(job), stage: job.stage || "Updating",
      };
    }
    if (job && job.status === "completed" && job.result && job.result.updated === false) {
      return {
        title: "Mariana Studio is up to date",
        subtitle: "The checkout already contains the latest release.",
        tone: "completed", percent: 100, stage: "Up to date",
      };
    }
    if ((job && job.status === "completed") || (status && status.restart_required)) {
      const version = status && status.local_version ? status.local_version : "the new version";
      return {
        title: "Update installed",
        subtitle: `Restart Mariana Studio to run ${version}. Stop the server (Ctrl+C) and start it again with python mariana.py studio.`,
        tone: "completed", percent: 100, stage: "Restart required",
      };
    }
    if (status && status.checking) {
      return {
        title: "Checking for updates",
        subtitle: "Comparing this checkout with the latest GitHub release…",
        tone: "preparing", percent: 0, stage: "Checking",
      };
    }
    if (job && job.status === "failed") {
      return {
        title: "Update failed",
        subtitle: job.error || "git could not finish. Open the job centre for the full output.",
        tone: "failed", percent: percentOf(job), stage: "Failed",
      };
    }
    if (job && job.status === "cancelled") {
      return {
        title: "Update cancelled",
        subtitle: "The checkout may be unchanged. Check again when you are ready.",
        tone: "failed", percent: percentOf(job), stage: "Cancelled",
      };
    }
    if (status && status.error) {
      return {
        title: "Update check failed",
        subtitle: status.error,
        tone: "failed", percent: 0, stage: "Not checked",
      };
    }
    if (status && status.update_available) {
      const git = status.git || {};
      const canUpdate = Boolean(git.checkout && git.available);
      const hint = canUpdate
        ? "Update with git, then restart the server."
        : git.checkout
          ? "git is not installed; install it or download the release from GitHub."
          : "Download the release from GitHub, then restart the server.";
      return {
        title: `Mariana ${releaseVersion(status)} is available`,
        subtitle: `You are running ${status.running_version || status.local_version}. ${hint}`,
        tone: "preparing", percent: 0, stage: "New release",
      };
    }
    if (status) {
      return {
        title: "Mariana Studio is up to date",
        subtitle: `Version ${status.local_version} is the latest release.`,
        tone: "completed", percent: 100, stage: "Up to date",
      };
    }
    return {
      title: "Checking for updates",
      subtitle: "Contacting GitHub for the latest release…",
      tone: "preparing", percent: 0, stage: "Checking",
    };
  }

  function render() {
    const data = view();
    if (!data) return;
    const status = state.status;
    const job = state.job;

    $("update-title").textContent = data.title;
    $("update-subtitle").textContent = data.subtitle;
    const panel = $("update-panel");
    panel.classList.remove("preparing", "running", "failed", "completed");
    if (data.tone) panel.classList.add(data.tone);
    panel.setAttribute("aria-busy", String(data.tone === "running" || data.tone === "preparing"));

    const current = status ? (status.running_version || status.local_version) : "unknown";
    const latest = releaseVersion(status);
    Studio.ui.render($("update-versions"), [
      h("span", { class: "update-version", text: `Running ${current}` }),
      latest ? h("span", { class: "update-arrow", text: "→" }) : null,
      latest ? h("span", {
        class: `update-version${status.update_available ? " next" : ""}`,
        text: `Latest ${latest}`,
      }) : null,
    ]);

    const notes = status && status.release && status.release.notes ? status.release.notes.trim() : "";
    $("update-notes").textContent = notes.slice(0, 360);
    $("update-notes").classList.toggle(
      "hidden", !notes || data.tone === "completed" || data.tone === "failed");

    const percent = Math.max(0, Math.min(100, Number(data.percent) || 0));
    $("update-bar").style.width = `${percent}%`;
    $("update-percent").textContent = `${Math.round(percent)}%`;
    $("update-progress").setAttribute("aria-valuenow", String(Math.round(percent)));
    $("update-stage").textContent = data.stage || "";

    const actions = Studio.ui.render($("update-actions"));
    if (job && job.active) {
      actions.append(Studio.ui.button("Hide", { small: true, kind: "ghost", onclick: hide }));
      actions.append(Studio.ui.button("Job centre", {
        small: true, kind: "ghost", onclick: () => { hide(); $("jobs-button").click(); },
      }));
    } else if ((job && job.status === "completed"
      && !(job.result && job.result.updated === false)) || (status && status.restart_required)) {
      actions.append(Studio.ui.button("Restart later", { small: true, kind: "primary", onclick: dismiss }));
    } else if (status && status.checking) {
      actions.append(Studio.ui.button("Hide", { small: true, kind: "ghost", onclick: hide }));
    } else if (job && job.status === "failed") {
      actions.append(Studio.ui.button("Retry", { small: true, kind: "primary", onclick: apply }));
      actions.append(Studio.ui.button("Job centre", {
        small: true, kind: "ghost", onclick: () => { hide(); $("jobs-button").click(); },
      }));
      actions.append(Studio.ui.button("Dismiss", { small: true, kind: "ghost", onclick: dismiss }));
    } else if (job && job.status === "cancelled") {
      actions.append(Studio.ui.button("Try again", { small: true, kind: "primary", onclick: apply }));
      actions.append(Studio.ui.button("Dismiss", { small: true, kind: "ghost", onclick: dismiss }));
    } else if (status && status.update_available) {
      if (status.git && status.git.checkout && status.git.available) {
        actions.append(Studio.ui.button("Update now", { small: true, kind: "primary", onclick: apply }));
      } else if (status.releases_url) {
        actions.append(h("a", {
          class: "button small primary", href: status.releases_url,
          target: "_blank", rel: "noreferrer", text: "Open release page",
        }));
      }
      if (status.release && status.release.url) {
        actions.append(h("a", {
          class: "button small ghost", href: status.release.url,
          target: "_blank", rel: "noreferrer", text: "Release notes",
        }));
      }
      actions.append(Studio.ui.button("Not now", { small: true, kind: "ghost", onclick: dismiss }));
    } else if (status && status.error) {
      actions.append(Studio.ui.button("Try again", { small: true, kind: "primary", onclick: open }));
      actions.append(Studio.ui.button("Dismiss", { small: true, kind: "ghost", onclick: hide }));
    } else {
      actions.append(Studio.ui.button("Close", { small: true, kind: "ghost", onclick: hide }));
    }
  }

  function start() {
    if (state.started) return;
    state.started = true;
    $("update-close").addEventListener("click", () => {
      if (state.applying || (state.job && state.job.active)) hide();
      else dismiss();
    });
    Studio.on("job:update", updateJob);
    refresh().catch(() => render());
  }

  Studio.updates = { start, open, refresh };
})();
