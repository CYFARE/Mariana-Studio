"use strict";

/* Studio.views: workspace registration, URL routing and lifecycle.
   Navigation is async-safe: a slow activation can never overwrite a newer one. */
(function () {
  const Studio = window.Studio;
  const { $, $$, query, toast, emit } = Studio;
  const scrollPositions = new Map();
  let sequence = 0;

  function focusMain() {
    const main = $("main");
    if (main) main.focus({ preventScroll: true });
  }

  function report(name, error) {
    console.error(`Opening ${name} failed`, error);
    toast(`The ${name} workspace hit an error: ${error.message}`, "error");
  }

  function restoreScroll(name, reset) {
    requestAnimationFrame(() => {
      window.scrollTo(0, reset ? 0 : scrollPositions.get(name) || 0);
    });
  }

  Studio.views = {
    registry: new Map(),
    current: null,
    parameters: {},
    order: [],

    register(name, definition) {
      this.registry.set(name, { name, ready: false, ...definition });
      this.order.push(name);
    },

    parse(hash) {
      const match = /^#\/([a-z-]+)(?:\?(.*))?$/.exec(hash || "");
      if (!match) return { name: "home", parameters: {} };
      return { name: match[1], parameters: Object.fromEntries(new URLSearchParams(match[2] || "")) };
    },

    /* Navigate to a view; parameters travel in the URL so links can be shared. */
    go(name, parameters) {
      const target = `#/${name}${query(parameters)}`;
      if (window.location.hash === target) {
        this.route();
        focusMain();
      } else window.location.hash = target;
    },

    async route() {
      const token = ++sequence;
      const parsed = this.parse(window.location.hash);
      const name = this.registry.has(parsed.name) ? parsed.name : "home";
      const view = this.registry.get(name);
      if (!view) return;
      if (Studio.overlay && Studio.overlay.closeModal) Studio.overlay.closeModal();
      const previous = this.current && this.current !== name ? this.registry.get(this.current) : null;
      if (previous) {
        scrollPositions.set(previous.name, window.scrollY);
        if (previous.deactivate) {
          try {
            await previous.deactivate(parsed.parameters);
          } catch (error) {
            if (token === sequence) console.error(`Leaving ${previous.name} failed`, error);
          }
          if (token !== sequence) return;
        }
      }
      this.current = name;
      this.parameters = parsed.parameters;
      for (const key of this.registry.keys()) {
        const section = document.getElementById(`${key}-view`);
        if (section) section.classList.toggle("hidden", key !== name);
      }
      for (const link of $$(".nav-item")) {
        if (link.dataset.view === name) link.setAttribute("aria-current", "page");
        else link.removeAttribute("aria-current");
      }
      $("shell-title").textContent = view.title || "Mariana Studio";
      $("shell-subtitle").textContent = view.subtitle || "";
      document.title = view.title ? `${view.title} · Mariana Studio` : "Mariana Studio";
      $("app").classList.remove("menu-open");
      const menu = $("shell-menu");
      if (menu) {
        menu.setAttribute("aria-expanded", "false");
        menu.setAttribute("aria-label", "Open navigation");
      }

      try {
        if (!view.ready) {
          view.ready = true;
          if (view.init) view.init();
          if (token !== sequence) return;
        }
        if (view.activate) await view.activate(parsed.parameters);
      } catch (error) {
        if (token === sequence) report(view.title || name, error);
      }
      if (token !== sequence) return;
      restoreScroll(name, Object.keys(parsed.parameters).length > 0);
      emit("view", { name, parameters: parsed.parameters });
    },

    start() {
      window.addEventListener("hashchange", () => {
        this.route();
        focusMain();
      });
      this.route();
    },
  };
})();
