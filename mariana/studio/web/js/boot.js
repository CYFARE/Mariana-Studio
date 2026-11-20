"use strict";

(function () {
  const Studio = window.Studio;
  const { $, icon } = Studio;

  function putIcon(id, name) {
    const element = $(id);
    if (element && !element.children.length) element.append(icon(name, "small"));
  }

  function theme() {
    const current = document.documentElement.dataset.theme === "dark" ? "dark" : "light";
    const next = current === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    Studio.store.set("theme", next);
    $("theme-icon").setAttribute("href", next === "dark" ? "#i-moon" : "#i-sun");
    $("theme-toggle").setAttribute("aria-label", `Switch to ${current} theme`);
    Studio.emit("theme", next);
  }

  async function start() {
    for (const [id, name] of Object.entries({
      "chat-more": "more", "chat-settings": "settings", "chat-settings-close": "x",
      "chat-provider": "globe",
      "agents-undo": "undo", "agents-redo": "redo", "agents-more": "more", "agents-add-ability": "plus",
      "agents-provider": "globe", "agents-pane-menu": "layout", "agents-toggle-details": "table",
      "agents-add-node": "plus", "agents-find-toggle": "search", "agents-pan-mode": "hand", "agents-run-menu": "chevron-down",
      "agents-zoom-out": "zoom-out", "agents-zoom-in": "zoom-in", "agents-fit": "fit", "agents-layout": "layout",
    })) putIcon(id, name);
    const dark = document.documentElement.dataset.theme === "dark";
    $("theme-icon").setAttribute("href", dark ? "#i-moon" : "#i-sun");
    $("theme-toggle").setAttribute("aria-label", `Switch to ${dark ? "light" : "dark"} theme`);
    $("theme-toggle").addEventListener("click", theme);
    $("shell-command-key").textContent = /Mac|iPhone|iPad/.test(navigator.platform) ? "⌘ K" : "Ctrl K";
    const collapsed = Studio.store.get("sidebar-collapsed", false);
    $("app").classList.toggle("collapsed", collapsed);
    $("shell-collapse").addEventListener("click", () => {
      const value = !$("app").classList.contains("collapsed");
      $("app").classList.toggle("collapsed", value);
      Studio.store.set("sidebar-collapsed", value);
      $("shell-collapse").setAttribute("aria-label", value ? "Expand sidebar" : "Collapse sidebar");
    });
    $("shell-menu").addEventListener("click", () => $("app").classList.toggle("menu-open"));
    $("skip-to-main").addEventListener("click", event => {
      event.preventDefault();
      $("main").focus({ preventScroll: true });
    });
    $("app").addEventListener("click", event => {
      if ($("app").classList.contains("menu-open") && !event.target.closest(".sidebar, #shell-menu")) $("app").classList.remove("menu-open");
    });
    Studio.palette.start();
    Studio.jobs.start();
    Studio.dependencies.start();
    Studio.updates.start();
    // Paint the routed workspace immediately; hardware discovery must never
    // hold the whole UI behind a slow probe.
    Studio.views.start();
    try {
      const state = await Studio.api("/api/state");
      Studio.state = { ...Studio.state, ...state };
      $("shell-device").textContent = state.device || "CPU";
      Studio.emit("state", state);
    } catch (error) {
      $("shell-device").textContent = "Studio unavailable";
      Studio.toast(error.message, "error");
    }
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
  else start();
})();
