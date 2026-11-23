"use strict";

(function () {
  const Studio = window.Studio;
  const { $, h, format, api, icon } = Studio;
  const { card, listRow } = Studio.components;
  let active = false;
  let loading = false;
  let again = false;
  let loaded = false;
  let signature = null;
  let feed = null;
  let live = null;
  let jobsDigest = "";
  const refresh = Studio.debounce(() => { if (active) load({ background: loaded }); }, 800);

  function runtimeReady(runtime) {
    return Boolean(runtime.running || runtime.available || runtime.installed);
  }

  /* Only the five visible job rows matter here; log lines must not refetch the dashboard. */
  function jobDigest(jobs) {
    return JSON.stringify([jobs.filter(job => job.active).length, jobs.slice(0, 5).map(job =>
      [job.id, job.status, Boolean(job.active), job.title, job.stage])]);
  }

  function onJobs(jobs) {
    const next = jobDigest(jobs);
    if (next === jobsDigest) return;
    jobsDigest = next;
    refresh();
  }

  /* Live hardware readings only touch the meters, never the rest of the page. */
  function accept(stats) {
    if (!active || !live || !stats || stats.type !== "stats") return;
    if (live.gpu && stats.gpu) {
      live.gpu.update({ value: stats.gpu.used, total: stats.gpu.total,
        text: `${format.bytes(stats.gpu.used)} / ${format.bytes(stats.gpu.total)}` });
    }
    if (live.memory && stats.memory && stats.memory.total && Number.isFinite(stats.memory.available)) {
      const used = stats.memory.total - stats.memory.available;
      live.memory.update({ value: used, total: stats.memory.total,
        text: `${format.bytes(used)} / ${format.bytes(stats.memory.total)}` });
    }
    if (live.disk && stats.disk && stats.disk.total && Number.isFinite(stats.disk.free)) {
      live.disk.update({ value: stats.disk.total - stats.disk.free, total: stats.disk.total,
        text: `${format.bytes(stats.disk.free)} free` });
    }
    if (live.uptime && Number.isFinite(stats.uptime)) live.uptime.textContent = format.duration(stats.uptime);
  }

  function connect() {
    if (feed || !active) return;
    feed = Studio.subscribe("/api/system/events", accept);
  }

  function disconnect() {
    if (feed) feed.close();
    feed = null;
  }

  /* Home is a summary, not a live progress view.  Ignore job percentages and
     scroll-stable details so a busy training log does not re-render the page. */
  function snapshot(system, chats, jobs) {
    return JSON.stringify({
      version: system.version,
      counts: system.counts,
      device: [system.device.type, system.device.label, system.device.gpu ? system.device.gpu.name : ""],
      software: system.software,
      runtimes: Object.entries(system.runtimes).map(([id, runtime]) => [id, runtimeReady(runtime)]),
      chats: chats.chats.slice(0, 5).map(chat => [chat.id, chat.updated, chat.title]),
      jobs: jobs.slice(0, 5).map(job => [job.id, job.status, Boolean(job.active)]),
    });
  }

  async function load(options = {}) {
    if (loading) { again = true; return; }
    loading = true;
    const background = Boolean(options.background && loaded);
    const root = $("home-content");
    if (!background) Studio.ui.render(root, Studio.ui.loading("Loading system overview…"));
    else root.setAttribute("aria-busy", "true");
    try {
      const [system, chats] = await Promise.all([api("/api/system"), api("/api/chats")]);
      const jobs = Studio.jobs.list();
      if (!active) return;
      const next = snapshot(system, chats, jobs);
      jobsDigest = jobDigest(jobs);
      if (background && next === signature) return;
      signature = next;
      loaded = true;
      const counts = system.counts;
      const runtimes = Object.entries(system.runtimes).filter(([id]) => id !== "loaded");
      const gpuMeter = system.device.gpu ? Studio.ui.meter({
        label: system.device.gpu.name, value: system.device.gpu.used, total: system.device.gpu.total,
        text: `${format.bytes(system.device.gpu.used)} / ${format.bytes(system.device.gpu.total)}`,
      }) : null;
      const memoryMeter = system.memory.total ? Studio.ui.meter({
        label: "System memory", value: system.memory.total - system.memory.available,
        total: system.memory.total,
        text: `${format.bytes(system.memory.total - system.memory.available)} / ${format.bytes(system.memory.total)}`,
      }) : null;
      const diskMeter = Studio.ui.meter({
        label: "Workspace disk", value: system.disk.total - system.disk.free, total: system.disk.total,
        text: `${format.bytes(system.disk.free)} free`,
      });
      const uptime = h("span", { text: format.duration(system.uptime) });
      live = { gpu: gpuMeter, memory: memoryMeter, disk: diskMeter, uptime };
      const setup = counts.chat_ready ? null : Studio.ui.callout("warning",
        h("div", { class: "stack tight" },
          h("span", { text: "No chat-ready model yet. Download open weights or connect a provider before starting a conversation." }),
          h("div", { class: "row wrap" },
            h("a", { class: "button small", href: "#/models?tab=discover", text: "Find a model" }))));
      const recent = chats.chats[0] || null;
      const quick = [
        ["#/chat", "chat", "Chat", "Ask a model or a swarm"],
        ["#/ide", "ide", "IDE", "Agentic coding"],
        ["#/train", "train", "Train", "Build from a corpus"],
        ["#/finetune", "tune", "Fine-tune", "Adapt existing weights"],
      ];
      Studio.ui.render(root,
        h("section", { class: "home-welcome" },
          h("div", { class: "home-welcome-copy" },
            h("p", { class: "eyebrow", text: `Version ${system.version}` }),
            h("h2", { text: recent ? "Pick up where you left off." : "Your local model workspace." }),
            h("p", { class: "muted", text: recent
              ? `Last conversation: ${recent.title}`
              : "Train, adapt, chat with and ship language models without leaving this machine." }),
            h("div", { class: "row wrap" },
              h("a", {
                class: "button primary large",
                href: recent ? `#/chat?id=${encodeURIComponent(recent.id)}` : "#/chat?new=1",
                text: recent ? "Continue chatting" : "Start a chat",
              }),
              h("a", { class: "button large", href: "#/models", text: "Browse models" }))),
          h("nav", { class: "home-stats", "aria-label": "Workspace summary" },
            [
              ["models", counts.models, "#/models"],
              ["chat ready", counts.chat_ready, "#/models"],
              ["conversations", counts.chats, "#/chat"],
              ["agents", counts.agents, "#/agents"],
            ].map(([label, value, href]) => h("a", { class: "home-stat", href },
              h("strong", { text: format.number(value) }), h("span", { text: label }))))),
        setup,
        h("nav", { class: "home-actions", "aria-label": "Quick actions" },
          quick.map(([href, glyph, title, text]) => h("a", { class: "home-action", href },
            h("span", { class: "home-action-icon" }, icon(glyph, "small")),
            h("span", { class: "home-action-copy" },
              h("strong", { text: title }), h("span", { text }))))),
        h("div", { class: "home-grid" },
          h("div", { class: "home-stack" },
            card({
              title: "Recent conversations",
              actions: h("a", { class: "small", href: "#/chat", text: "Open Chat" }),
              children: chats.chats.length ? chats.chats.slice(0, 4).map(chat => listRow({
                href: `#/chat?id=${encodeURIComponent(chat.id)}`,
                title: chat.title,
                subtitle: chat.preview || "No messages",
                meta: format.ago(chat.updated),
              })) : [Studio.ui.empty({ compact: true, text: "Your saved conversations will appear here." })],
            }),
            card({
              title: "Recent jobs",
              actions: h("button", { class: "link-button small", type: "button", onclick: () => $("jobs-button").click(), text: "Job centre" }),
              children: jobs.length ? jobs.slice(0, 4).map(job => listRow({
                leading: Studio.ui.status(job.status),
                title: job.title,
                subtitle: job.stage || format.title(job.kind),
                meta: format.ago(job.created),
              })) : [Studio.ui.empty({ compact: true, text: "Training, downloads and exports will appear here." })],
            })),
          card({
            title: "System",
            actions: [
              Studio.ui.badge(system.device.type.toUpperCase(), "completed", { title: system.device.label }),
              h("a", { class: "small", href: "#/models?tab=runtimes", text: "Manage" }),
            ],
            children: [
              gpuMeter,
              memoryMeter,
              diskMeter,
              h("details", { class: "disclosure" },
                h("summary", { text: "Software and runtimes" }),
                h("div", { class: "disclosure-body stack tight" },
                  Studio.ui.kv([
                    ["Python", system.software.python], ["PyTorch", system.software.torch],
                    ["CPU threads", system.device.threads], ["Uptime", uptime],
                  ]),
                  runtimes.length ? runtimes.map(([id, runtime]) => listRow({
                    title: runtime.label || runtime.name || format.title(id),
                    subtitle: runtime.hint || runtime.detail || runtime.path || runtime.url || runtime.status || "Detected",
                    trailing: Studio.ui.badge(runtimeReady(runtime) ? "Ready" : "Stopped", runtimeReady(runtime) ? "completed" : ""),
                  })) : Studio.ui.empty({ compact: true, text: "No optional runtimes detected." }))),
            ],
          })));
    } catch (error) {
      if (!background && active) {
        live = null;
        Studio.ui.render(root, Studio.ui.empty({ icon: "alert", title: "Could not load the dashboard", text: error.message, actions: [Studio.ui.button("Try again", { kind: "primary", onclick: load })] }));
      } else console.error("Dashboard refresh failed", error);
    } finally {
      root.setAttribute("aria-busy", "false");
      loading = false;
      if (again) { again = false; refresh(); }
    }
  }

  Studio.views.register("home", {
    title: "Home", subtitle: "Your local language-model research workspace",
    glyph: "home", hint: "Overview and system health",
    activate() { active = true; connect(); return load({ background: loaded }); },
    deactivate() { active = false; disconnect(); },
    init() {
      Studio.on("jobs", onJobs);
      Studio.on("models", refresh);
    },
  });
})();
