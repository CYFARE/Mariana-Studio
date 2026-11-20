"use strict";

/* Studio.split: draggable pane dividers with keyboard control, preferred sizes
   that survive window resizing and automatic folding when the container gets
   too narrow.  The host owns the CSS grid; this module only computes sizes and
   writes custom properties, so it carries no markup or visual design. */
(function () {
  const Studio = window.Studio;
  const SNAP_MARGIN = 48;
  const MIN_STEP = 8;
  const LARGE_STEP = 32;
  const MODES = ["auto", "columns", "stacked"];

  function clamp(value, low, high) {
    if (high < low) high = low;
    return Math.max(low, Math.min(high, value));
  }

  function number(value, fallback) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : fallback;
  }

  /* Distribute the available room between an optional left/top pane, an
     optional right/bottom pane and a centre pane with its own minimum.
     Preferred sizes win; overflow eats the second pane first, then the first;
     only when even their minimums do not fit does a pane fold away.  The pure
     function is exported so the sizing rules can be tested without a DOM. */
  function fitAxis(available, gap, first, second, centerMin) {
    const room = Math.max(0, number(available, 0));
    const space = Math.max(0, number(gap, 0));
    const centre = Math.max(0, number(centerMin, 0));
    const left = first && !first.collapsed ? first : null;
    const right = second && !second.collapsed ? second : null;
    // A folded pane keeps its divider, so its gap still belongs to the layout.
    const gapCount = (first ? 1 : 0) + (second ? 1 : 0);
    const result = {
      first: { size: 0, visible: false },
      second: { size: 0, visible: false },
    };
    if (!left && !right) return result;

    let firstSize = left ? clamp(number(left.size, left.min), left.min, left.max) : 0;
    let secondSize = right ? clamp(number(right.size, right.min), right.min, right.max) : 0;
    let firstVisible = Boolean(left);
    let secondVisible = Boolean(right);

    const total = () => (firstVisible ? firstSize : 0) + (secondVisible ? secondSize : 0)
      + gapCount * space + centre;

    let over = total() - room;
    if (over > 0 && secondVisible) {
      const cut = Math.min(secondSize - right.min, over);
      secondSize -= cut;
      over -= cut;
    }
    if (over > 0 && firstVisible) {
      const cut = Math.min(firstSize - left.min, over);
      firstSize -= cut;
      over -= cut;
    }
    if (over > 0 && secondVisible) {
      secondVisible = false;
      over = total() - room;
    }
    if (over > 0) firstVisible = false;

    result.first = { size: firstVisible ? Math.round(firstSize) : 0, visible: firstVisible };
    result.second = { size: secondVisible ? Math.round(secondSize) : 0, visible: secondVisible };
    return result;
  }

  function create(options) {
    const root = options.root;
    if (!root) throw new Error("Studio.split needs a root element");
    const gap = Math.max(0, number(options.gap, 6));
    const centerMin = Math.max(0, number(options.centerMin, 320));
    const centerMinHeight = Math.max(0, number(options.centerMinHeight, 220));
    const stackAt = Math.max(0, number(options.stackAt, 1040));
    const key = options.key || "studio-split";
    const store = options.storage || Studio.store;
    const sidebar = options.sidebar || {};
    const agent = options.agent || {};
    const sidebarSplitter = sidebar.splitter;
    const agentSplitter = agent.splitter;
    const variables = {
      sidebar: sidebar.variable || "--ide-sidebar-size",
      agent: agent.variable || "--ide-agent-size",
      agentHeight: agent.heightVariable || "--ide-agent-height",
    };
    const defaultMode = MODES.includes(options.mode) ? options.mode : "auto";
    const sidebarLimits = {
      min: Math.max(0, number(sidebar.min, 180)),
      max: Math.max(0, number(sidebar.max, 480)),
      size: Math.max(0, number(sidebar.defaultSize, 252)),
    };
    const agentLimits = {
      min: Math.max(0, number(agent.min, 280)),
      max: Math.max(0, number(agent.max, 680)),
      size: Math.max(0, number(agent.defaultSize, 360)),
      minHeight: Math.max(0, number(agent.minHeight, 220)),
      maxHeight: Math.max(0, number(agent.maxHeight, 720)),
      height: Math.max(0, number(agent.defaultHeight, 320)),
    };

    let saved = {};
    try {
      saved = (store && store.get(key, {})) || {};
    } catch (error) {
      saved = {};
    }
    const state = {
      mode: MODES.includes(saved.mode) ? saved.mode : defaultMode,
      sidebar: {
        width: Math.max(0, number(saved.sidebar, sidebarLimits.size)),
        collapsed: Boolean(saved.sidebarCollapsed),
      },
      agent: {
        width: Math.max(0, number(saved.agent, agentLimits.size)),
        height: Math.max(0, number(saved.agentHeight, agentLimits.height)),
        collapsed: Boolean(saved.agentCollapsed),
      },
      stacked: false,
    };
    let snapshot = null;
    let frame = null;
    let observer = null;
    let resizing = null;
    const teardown = [];

    root.style.setProperty("--ide-split", `${gap}px`);

    function save() {
      if (!store || typeof store.set !== "function") return;
      store.set(key, {
        mode: state.mode,
        sidebar: Math.round(state.sidebar.width),
        sidebarCollapsed: state.sidebar.collapsed,
        agent: Math.round(state.agent.width),
        agentHeight: Math.round(state.agent.height),
        agentCollapsed: state.agent.collapsed,
      });
    }

    function measure() {
      const width = Math.round(root.clientWidth || 0);
      const height = Math.round(root.clientHeight || 0);
      return {
        width: width || Math.max(0, number(options.fallbackWidth, 1200)),
        height: height || Math.max(0, number(options.fallbackHeight, 700)),
      };
    }

    function splitterLabel(name) {
      if (name === "sidebar") return sidebar.label || "Explorer";
      return agent.label || "Agent panel";
    }

    function paintSplitter(splitter, name, visible, orientation, size, bounds) {
      if (!splitter) return;
      const label = splitterLabel(name);
      splitter.classList.toggle("collapsed", !visible);
      splitter.setAttribute("aria-orientation", orientation);
      splitter.setAttribute("aria-valuemin", "0");
      splitter.setAttribute("aria-valuemax", String(Math.round(bounds.max)));
      splitter.setAttribute("aria-valuenow", String(visible ? Math.round(size) : 0));
      splitter.setAttribute("aria-valuetext", visible ? `${label} ${Math.round(size)} pixels` : `${label} collapsed`);
      splitter.title = visible
        ? `${label}: drag to resize · double-click or Enter to collapse`
        : `${label}: collapsed · drag or press Enter to expand`;
    }

    function apply() {
      const box = measure();
      const stacked = state.mode === "stacked" || (state.mode === "auto" && box.width < stackAt);
      const sidebarInput = sidebar.enabled === false ? null : {
        size: state.sidebar.width, min: sidebarLimits.min, max: sidebarLimits.max,
        collapsed: state.sidebar.collapsed,
      };
      const agentInput = agent.enabled === false ? null : {
        size: state.agent.width, min: agentLimits.min, max: agentLimits.max,
        collapsed: state.agent.collapsed,
      };
      const horizontal = fitAxis(box.width, gap, sidebarInput, stacked ? null : agentInput, centerMin);
      const vertical = stacked
        ? fitAxis(box.height, gap, {
          size: state.agent.height, min: agentLimits.minHeight, max: agentLimits.maxHeight,
          collapsed: state.agent.collapsed,
        }, null, centerMinHeight)
        : null;
      const sidebarFit = horizontal.first;
      const agentFit = stacked ? vertical.first : horizontal.second;
      const bounds = {
        sidebar: { axis: "x", min: sidebarLimits.min, max: sidebarLimits.max },
        agent: {
          axis: stacked ? "y" : "x",
          min: stacked ? agentLimits.minHeight : agentLimits.min,
          max: stacked ? agentLimits.maxHeight : agentLimits.max,
        },
      };
      const otherAgent = !stacked && agentFit.visible ? agentFit.size : 0;
      const otherSidebar = sidebarFit.visible ? sidebarFit.size : 0;
      const columnGaps = stacked ? gap : 2 * gap;
      bounds.sidebar.max = Math.max(bounds.sidebar.min,
        Math.min(sidebarLimits.max, box.width - centerMin - columnGaps - otherAgent));
      if (stacked) {
        bounds.agent.max = Math.max(bounds.agent.min,
          Math.min(agentLimits.maxHeight, box.height - centerMinHeight - gap));
      } else {
        bounds.agent.max = Math.max(bounds.agent.min,
          Math.min(agentLimits.max, box.width - centerMin - 2 * gap - otherSidebar));
      }

      state.stacked = stacked;
      root.classList.toggle("stacked", stacked);
      root.classList.toggle("no-sidebar", !sidebarFit.visible);
      root.classList.toggle("no-agent", !agentFit.visible);
      root.style.setProperty(variables.sidebar,
        `${Math.round(sidebarFit.visible ? sidebarFit.size : state.sidebar.width)}px`);
      root.style.setProperty(variables.agent,
        `${Math.round(!stacked && agentFit.visible ? agentFit.size : state.agent.width)}px`);
      root.style.setProperty(variables.agentHeight,
        `${Math.round(stacked && agentFit.visible ? agentFit.size : state.agent.height)}px`);

      paintSplitter(sidebarSplitter, "sidebar", sidebarFit.visible, "vertical", sidebarFit.size, bounds.sidebar);
      paintSplitter(agentSplitter, "agent", agentFit.visible, stacked ? "horizontal" : "vertical",
        agentFit.size, bounds.agent);

      snapshot = {
        mode: state.mode,
        stacked,
        sidebar: { visible: sidebarFit.visible, size: sidebarFit.size, collapsed: state.sidebar.collapsed },
        agent: { visible: agentFit.visible, size: agentFit.size, collapsed: state.agent.collapsed },
        bounds,
        width: box.width,
        height: box.height,
      };
      if (options.onChange) options.onChange(snapshot);
      return snapshot;
    }

    function schedule() {
      if (frame !== null) return;
      if (typeof requestAnimationFrame !== "function") {
        apply();
        return;
      }
      frame = requestAnimationFrame(() => {
        frame = null;
        apply();
      });
    }

    function listen(target, type, handler) {
      target.addEventListener(type, handler);
      teardown.push(() => target.removeEventListener(type, handler));
    }

    function setCollapsed(name, collapsed, remember = true) {
      const pane = state[name];
      if (!pane || pane.collapsed === Boolean(collapsed)) return;
      pane.collapsed = Boolean(collapsed);
      if (remember) save();
      apply();
    }

    function setSize(name, value, remember = true) {
      const pane = state[name];
      if (!pane) return;
      // Sizing a folded pane is a request to see it again.
      if (number(value, 0) > 0) pane.collapsed = false;
      if (name === "sidebar") {
        pane.width = clamp(number(value, sidebarLimits.min), sidebarLimits.min, sidebarLimits.max);
      } else if (snapshot && snapshot.stacked) {
        pane.height = clamp(number(value, agentLimits.minHeight), agentLimits.minHeight, agentLimits.maxHeight);
      } else {
        pane.width = clamp(number(value, agentLimits.min), agentLimits.min, agentLimits.max);
      }
      if (remember) save();
      apply();
    }

    function currentSize(name) {
      if (!snapshot) return 0;
      return name === "sidebar" ? snapshot.sidebar.size : snapshot.agent.size;
    }

    function adjust(name, delta) {
      if (!snapshot) return;
      const bounds = snapshot.bounds[name];
      const size = currentSize(name);
      if (delta < 0 && size <= bounds.min) {
        setCollapsed(name, true);
        return;
      }
      setSize(name, clamp(size + delta, bounds.min, bounds.max));
    }

    function onKey(event, name) {
      if (!snapshot) return;
      const vertical = snapshot.bounds[name].axis === "y";
      const step = event.shiftKey ? LARGE_STEP : MIN_STEP;
      const grow = name === "sidebar"
        ? (vertical ? "ArrowDown" : "ArrowRight")
        : (vertical ? "ArrowUp" : "ArrowLeft");
      const shrink = name === "sidebar"
        ? (vertical ? "ArrowUp" : "ArrowLeft")
        : (vertical ? "ArrowDown" : "ArrowRight");
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        toggle(name);
      } else if (event.key === "Home") {
        event.preventDefault();
        setCollapsed(name, true);
      } else if (event.key === "End") {
        event.preventDefault();
        setSize(name, snapshot.bounds[name].max);
      } else if (event.key === grow) {
        event.preventDefault();
        adjust(name, step);
      } else if (event.key === shrink) {
        event.preventDefault();
        adjust(name, -step);
      }
    }

    function varFor(name, vertical) {
      if (name === "sidebar") return variables.sidebar;
      return vertical ? variables.agentHeight : variables.agent;
    }

    function beginResize(event, name) {
      if (event.button !== 0 || !snapshot) return;
      const splitter = name === "sidebar" ? sidebarSplitter : agentSplitter;
      const bounds = snapshot.bounds[name];
      const vertical = bounds.axis === "y";
      const start = vertical ? event.clientY : event.clientX;
      const startSize = currentSize(name);
      const sign = name === "sidebar" ? 1 : -1;
      const folded = startSize <= 0;
      const hiddenClass = name === "sidebar" ? "no-sidebar" : "no-agent";
      let opening = !folded;
      let preview = startSize;
      let snap = false;
      let moved = false;
      const variable = varFor(name, vertical);
      const label = splitterLabel(name);
      event.preventDefault();
      splitter.classList.add("dragging");
      root.classList.add("resizing");
      document.body.classList.add(vertical ? "pane-resizing-y" : "pane-resizing-x");
      resizing = { name, splitter };

      function paint(size) {
        splitter.setAttribute("aria-valuenow", String(Math.round(size)));
        splitter.setAttribute("aria-valuetext", `${label} ${Math.round(size)} pixels`);
        root.style.setProperty(variable, `${Math.round(size)}px`);
      }

      function onMove(moveEvent) {
        const delta = (vertical ? moveEvent.clientY : moveEvent.clientX) - start;
        if (Math.abs(delta) > 2) moved = true;
        const wanted = startSize + sign * delta;
        if (folded) {
          // Follow the pointer from the closed edge; the pane opens fully only
          // once the drag has moved far enough to mean it.
          opening = wanted > 0;
          root.classList.toggle(hiddenClass, !opening);
          preview = Math.max(0, Math.min(bounds.max, wanted));
          snap = false;
        } else {
          snap = wanted < bounds.min - SNAP_MARGIN;
          preview = clamp(wanted, bounds.min, bounds.max);
        }
        splitter.classList.toggle("snap", snap);
        paint(preview);
      }

      function finish(cancelled) {
        document.removeEventListener("pointermove", onMove);
        document.removeEventListener("pointerup", onUp);
        document.removeEventListener("pointercancel", onCancel);
        splitter.classList.remove("dragging", "snap");
        root.classList.remove("resizing");
        document.body.classList.remove("pane-resizing-x", "pane-resizing-y");
        resizing = null;
        if (cancelled || !moved || (folded && (!opening || preview < bounds.min * 0.5))) {
          apply();
        } else if (snap) {
          setCollapsed(name, true);
        } else if (folded) {
          setSize(name, Math.max(bounds.min, preview));
        } else {
          setSize(name, preview);
        }
      }

      function onUp() {
        finish(false);
      }

      function onCancel() {
        finish(true);
      }

      document.addEventListener("pointermove", onMove);
      document.addEventListener("pointerup", onUp);
      document.addEventListener("pointercancel", onCancel);
    }

    function bindSplitter(name, splitter) {
      if (!splitter) return;
      if (!splitter.getAttribute("tabindex")) splitter.tabIndex = 0;
      listen(splitter, "pointerdown", event => beginResize(event, name));
      listen(splitter, "dblclick", event => {
        event.preventDefault();
        toggle(name);
      });
      listen(splitter, "keydown", event => onKey(event, name));
    }

    function toggle(name) {
      const pane = state[name];
      const item = snapshot && snapshot[name];
      // A pane folded by the container, not by the person, has nowhere to go:
      // report that instead of recording a manual collapse the person never made.
      if (item && !item.visible && !pane.collapsed) {
        if (options.onBlocked) options.onBlocked(name);
        return;
      }
      setCollapsed(name, !pane.collapsed);
    }

    function setMode(mode) {
      if (!MODES.includes(mode) || mode === state.mode) return;
      state.mode = mode;
      save();
      apply();
    }

    function reset() {
      state.mode = defaultMode;
      state.sidebar.width = sidebarLimits.size;
      state.sidebar.collapsed = false;
      state.agent.width = agentLimits.size;
      state.agent.height = agentLimits.height;
      state.agent.collapsed = false;
      save();
      apply();
    }

    bindSplitter("sidebar", sidebarSplitter);
    bindSplitter("agent", agentSplitter);

    if (typeof window.ResizeObserver === "function") {
      observer = new window.ResizeObserver(schedule);
      observer.observe(root);
    } else if (typeof window.addEventListener === "function") {
      listen(window, "resize", schedule);
    }

    apply();

    return {
      apply,
      setMode,
      setCollapsed,
      setSize,
      reset,
      read() { return snapshot; },
      toggle,
      destroy() {
        if (frame !== null && typeof cancelAnimationFrame === "function") cancelAnimationFrame(frame);
        frame = null;
        if (observer) observer.disconnect();
        for (const remove of teardown) remove();
        teardown.length = 0;
        if (resizing) {
          resizing.splitter.classList.remove("dragging", "snap");
          root.classList.remove("resizing");
          document.body.classList.remove("pane-resizing-x", "pane-resizing-y");
          resizing = null;
        }
      },
    };
  }

  Studio.split = { create, fit: fitAxis };
})();
