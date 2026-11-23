"use strict";

(function () {
  const Studio = window.Studio;
  const { $, icon } = Studio;

  function putIcon(id, name) {
    const element = $(id);
    if (element && !element.children.length) element.append(icon(name, "small"));
  }

  function theme() {
    const media = window.matchMedia ? window.matchMedia("(prefers-color-scheme: dark)") : null;
    const currentMode = () => document.documentElement.dataset.themeMode || "system";
    const resolved = mode => mode === "system"
      ? (media && media.matches ? "dark" : "light") : mode;
    const apply = (mode, announce = true) => {
      const next = resolved(mode);
      document.documentElement.dataset.theme = next;
      document.documentElement.dataset.themeMode = mode;
      Studio.store.set("theme", mode);
      $("theme-icon").setAttribute("href", mode === "system" ? "#i-monitor"
        : mode === "dark" ? "#i-moon" : "#i-sun");
      $("theme-toggle").setAttribute("aria-label", `Theme: ${mode}. Choose theme`);
      $("theme-toggle").setAttribute("title", `Theme: ${mode}`);
      if (announce) Studio.emit("theme", next);
    };
    const choose = anchor => Studio.overlay.menu(anchor, [
      { heading: "Theme" },
      { label: "System", icon: "monitor", hint: currentMode() === "system" ? "Current" : "", onSelect: () => apply("system") },
      { label: "Light", icon: "sun", hint: currentMode() === "light" ? "Current" : "", onSelect: () => apply("light") },
      { label: "Dark", icon: "moon", hint: currentMode() === "dark" ? "Current" : "", onSelect: () => apply("dark") },
    ], { label: "Theme", align: "end" });
    apply(currentMode(), false);
    $("theme-toggle").addEventListener("click", event => choose(event.currentTarget));
    if (media && media.addEventListener) {
      media.addEventListener("change", () => {
        if (currentMode() === "system") apply("system");
      });
    }
  }

  async function start() {
    for (const [id, name] of Object.entries({
      "chat-more": "more", "chat-settings": "settings", "chat-settings-close": "x",
      "chat-provider": "globe", "chat-conversations": "sidebar",
      "chat-list-close": "x",
      "agents-undo": "undo", "agents-redo": "redo", "agents-more": "more", "agents-add-ability": "plus",
      "agents-provider": "globe", "agents-pane-menu": "layout", "agents-toggle-details": "table",
      "agents-add-node": "plus", "agents-find-toggle": "search", "agents-pan-mode": "hand", "agents-run-menu": "chevron-down",
      "agents-zoom-out": "zoom-out", "agents-zoom-in": "zoom-in", "agents-fit": "fit", "agents-layout": "layout",
    })) putIcon(id, name);
    theme();
    $("shell-command-key").textContent = /Mac|iPhone|iPad/.test(navigator.platform) ? "⌘ K" : "Ctrl K";
    const collapsed = Studio.store.get("sidebar-collapsed", false);
    $("app").classList.toggle("collapsed", collapsed);
    $("shell-collapse").addEventListener("click", () => {
      const value = !$("app").classList.contains("collapsed");
      $("app").classList.toggle("collapsed", value);
      Studio.store.set("sidebar-collapsed", value);
      $("shell-collapse").setAttribute("aria-label", value ? "Expand sidebar" : "Collapse sidebar");
    });
    const menuOpen = () => $("app").classList.contains("menu-open");
    const setMenu = (open, moveFocus = false) => {
      $("app").classList.toggle("menu-open", open);
      $("shell-menu").setAttribute("aria-expanded", String(open));
      $("shell-menu").setAttribute("aria-label", open ? "Close navigation" : "Open navigation");
      if (!moveFocus) return;
      requestAnimationFrame(() => {
        if (open) {
          const target = document.querySelector(".nav-item[aria-current='page']")
            || document.querySelector(".nav-item");
          if (target) target.focus();
        } else {
          $("shell-menu").focus();
        }
      });
    };
    $("shell-menu").addEventListener("click", () => setMenu(!menuOpen(), true));
    document.addEventListener("keydown", event => {
      if (event.key !== "Escape" || !menuOpen() || document.querySelector("dialog[open]")) return;
      setMenu(false, true);
    });
    window.addEventListener("resize", () => { if (window.innerWidth > 900 && menuOpen()) setMenu(false); });
    $("skip-to-main").addEventListener("click", event => {
      event.preventDefault();
      $("main").focus({ preventScroll: true });
    });
    $("app").addEventListener("click", event => {
      if (menuOpen() && !event.target.closest(".sidebar, #shell-menu")) setMenu(false);
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
