"use strict";

/* Studio UI primitives: tabs, badges, meters, cards' inner pieces and shared
   form controls.  Overlays live in overlay.js; composed blocks in components.js. */
(function () {
  const Studio = window.Studio;
  const { $$, h, icon, format } = Studio;

  /* -- tabs and segmented controls ------------------------------------------ */
  /* Wire a [role=tablist] whose tabs carry data-tab and aria-controls. */
  function tabs(list, onSelect) {
    const buttons = $$('[role="tab"]', list);
    const control = {
      current: null,
      select(name, notify = true) {
        const target = buttons.find(button => button.dataset.tab === name) || buttons[0];
        if (!target) return;
        for (const button of buttons) {
          const active = button === target;
          button.setAttribute("aria-selected", String(active));
          button.tabIndex = active ? 0 : -1;
          const panel = document.getElementById(button.getAttribute("aria-controls"));
          if (panel) panel.classList.toggle("hidden", !active);
        }
        const changed = control.current !== target.dataset.tab;
        control.current = target.dataset.tab;
        if (notify && changed && onSelect) onSelect(control.current);
      },
    };
    list.addEventListener("click", event => {
      const button = event.target.closest('[role="tab"]');
      if (button && list.contains(button)) control.select(button.dataset.tab);
    });
    list.addEventListener("keydown", event => {
      const index = buttons.indexOf(document.activeElement);
      if (index < 0 || !["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
      event.preventDefault();
      const next = event.key === "Home" ? 0 : event.key === "End" ? buttons.length - 1
        : (index + (event.key === "ArrowRight" ? 1 : -1) + buttons.length) % buttons.length;
      buttons[next].focus();
      control.select(buttons[next].dataset.tab);
    });
    return control;
  }

  /* Wire a .segmented group of buttons that carry data-value. */
  function segmented(group, onSelect) {
    const control = {
      value: null,
      select(value, notify = false) {
        control.value = value;
        for (const button of $$("button", group)) button.setAttribute("aria-pressed", String(button.dataset.value === value));
        if (notify && onSelect) onSelect(value);
      },
    };
    group.addEventListener("click", event => {
      const button = event.target.closest("button");
      if (button && group.contains(button) && button.dataset.value !== control.value) control.select(button.dataset.value, true);
    });
    const pressed = group.querySelector('button[aria-pressed="true"]');
    if (pressed) control.value = pressed.dataset.value;
    return control;
  }

  /* -- small building blocks ------------------------------------------------ */
  const STATUS_TONES = {
    running: "running", preparing: "running", queued: "running", paused: "paused", completed: "completed",
    ok: "completed", failed: "failed", error: "failed", cancelled: "cancelled", pending: "", idle: "",
  };

  function badge(text, tone, options = {}) {
    return h("span", { class: `badge${tone ? ` ${tone}` : ""}`, title: options.title || null, text });
  }

  function status(value) {
    const text = format.title(value || "idle");
    return h("span", { class: `badge ${STATUS_TONES[value] || ""}`.trim(), text });
  }

  function stat(options) {
    const tag = options.href ? "a" : "div";
    return h(tag, { class: `stat${options.compact ? " compact" : ""}`, href: options.href || null },
      h("span", { class: "stat-label", text: options.label }),
      h("span", { class: "stat-value", text: options.value }),
      options.hint ? h("span", { class: "stat-hint", text: options.hint }) : null);
  }

  /* A labelled bar for "how full": the track is a lighter step of the fill colour.
     ``update()`` mutates the existing nodes, so live statistics never re-render a view. */
  function meter(options) {
    const label = h("span", { text: options.label });
    const value = h("span", { class: "meter-value" });
    const fill = h("div", { class: "meter-fill" });
    const track = h("div", { class: "meter-track", role: "meter" }, fill);
    const element = h("div", { class: "meter" },
      h("div", { class: "meter-head" }, label, value), track);
    element.update = next => {
      const state = { ...options, ...(next || {}) };
      const total = Number(state.total) || 0;
      const current = Number(state.value) || 0;
      const ratio = total > 0 ? Math.max(0, Math.min(1, current / total)) : 0;
      const tone = state.tone || (ratio >= 0.92 ? "danger" : ratio >= 0.8 ? "warning" : "");
      const percent = Math.round(ratio * 100);
      const text = state.text || `${percent}%`;
      element.className = `meter${tone ? ` ${tone}` : ""}`;
      label.textContent = state.label;
      value.textContent = text;
      track.setAttribute("aria-label", state.label);
      track.setAttribute("aria-valuemin", "0");
      track.setAttribute("aria-valuemax", "100");
      track.setAttribute("aria-valuenow", String(percent));
      track.setAttribute("aria-valuetext", text);
      fill.style.width = `${percent}%`;
      return element;
    };
    return element.update(options);
  }

  /* A progress bar with update(percent, state); state: active | done | failed | indeterminate. */
  function progress(label) {
    const fill = h("div", { class: "progress-fill" });
    const bar = h("div", {
      class: "progress", role: "progressbar", "aria-label": label || "Progress",
      "aria-valuemin": "0", "aria-valuemax": "100", "aria-valuenow": "0",
    }, fill);
    bar.update = (percent, state = "active") => {
      const value = Math.max(0, Math.min(100, Number(percent) || 0));
      bar.classList.toggle("indeterminate", state === "indeterminate");
      bar.classList.toggle("done", state === "done");
      bar.classList.toggle("failed", state === "failed");
      fill.style.width = state === "indeterminate" ? "" : `${value}%`;
      if (state === "indeterminate") bar.removeAttribute("aria-valuenow");
      else bar.setAttribute("aria-valuenow", String(Math.round(value)));
      return bar;
    };
    return bar;
  }

  function empty(options) {
    return h("div", { class: `empty${options.compact ? " compact" : ""}` },
      options.icon ? h("div", { class: "empty-icon" }, icon(options.icon, "large")) : null,
      options.title ? h("h3", { text: options.title }) : null,
      options.text ? h("p", { text: options.text }) : null,
      options.actions && options.actions.length ? h("div", { class: "row wrap" }, options.actions) : null);
  }

  function loading(text = "Loading…") {
    return h("div", { class: "empty compact" }, h("div", { class: "row" }, h("span", { class: "spinner" }), h("span", { text })));
  }

  /* Clear a container and append children; arrays and null entries are flattened. */
  function render(root, ...children) {
    root.replaceChildren();
    return Studio.append(root, children);
  }

  /* Replace a select's choices, keeping the current value when it still exists. */
  function selectOptions(select, values, current) {
    select.replaceChildren(...(values || []).map(([value, label]) => h("option", {
      value: String(value), text: label, selected: String(value) === String(current),
    })));
    return select;
  }

  function callout(tone, ...children) {
    const icons = { warning: "alert", danger: "alert", success: "check" };
    return h("div", { class: `callout${tone ? ` ${tone}` : ""}` }, icon(icons[tone] || "info"), h("div", { class: "grow" }, children));
  }

  /* rows: [[label, value], ...]; value may be text or a node. */
  function kv(rows) {
    const list = h("dl", { class: "kv" });
    for (const [label, value] of rows) {
      if (value === null || value === undefined || value === "") continue;
      list.append(h("dt", { text: label }), h("dd", {}, value));
    }
    return list;
  }

  function code(text, language) {
    return h("div", { class: "code-block" },
      language ? h("span", { class: "code-lang", text: language }) : null,
      h("button", { class: "icon-button small code-copy", type: "button", "data-copy-code": "", "aria-label": "Copy code" }, icon("copy", "small")),
      h("pre", {}, h("code", { text })));
  }

  function field(label, control, help) {
    return h("label", { class: "field" }, h("span", { text: label }), control, help ? h("span", { class: "help", text: help }) : null);
  }

  function button(label, options = {}) {
    return h("button", {
      class: `button${options.kind ? ` ${options.kind}` : ""}${options.small ? " small" : ""}`, type: "button",
      title: options.title || null, disabled: Boolean(options.disabled), onclick: options.onclick || null,
    }, options.icon ? icon(options.icon, "small") : null, label);
  }

  function iconButton(name, label, onclick, options = {}) {
    return h("button", {
      class: `icon-button${options.small ? " small" : ""}`, type: "button", "aria-label": label, title: label, onclick,
    }, icon(name, options.small ? "small" : ""));
  }

  /* -- model select --------------------------------------------------------- */
  const SOURCE_ORDER = ["workspace", "ollama", "huggingface", "llama.cpp", "lm-studio", "gpt4all", "jan",
    "text-generation-webui", "folder", "server", "endpoint"];

  /* Fill a <select> with catalog models grouped by where they live.  The current
     choice is kept when it still exists. */
  function fillModels(select, options = {}) {
    const previous = options.value !== undefined ? options.value : select.value;
    const entries = Studio.models.list.filter(options.filter || (() => true));
    const groups = new Map();
    for (const entry of entries) {
      if (!groups.has(entry.source)) groups.set(entry.source, []);
      groups.get(entry.source).push(entry);
    }
    Studio.clear(select);
    if (options.placeholder) select.append(h("option", { value: "", text: options.placeholder }));
    const sources = Array.from(groups.keys()).sort((left, right) => {
      const a = SOURCE_ORDER.indexOf(left);
      const b = SOURCE_ORDER.indexOf(right);
      return (a < 0 ? 99 : a) - (b < 0 ? 99 : b);
    });
    for (const source of sources) {
      const items = groups.get(source);
      const group = h("optgroup", { label: items[0].source_label || source });
      for (const entry of items) {
        const usable = entry.available || options.includeUnavailable;
        group.append(h("option", {
          value: entry.id, disabled: !usable,
          text: `${entry.name}${entry.available ? "" : " (unavailable)"}`,
        }));
      }
      select.append(group);
    }
    const values = Array.from(select.options).filter(option => !option.disabled).map(option => option.value);
    if (previous && values.includes(previous)) select.value = previous;
    else if (options.placeholder) select.value = "";
    else if (values.length) select.value = values[0];
    return select.value;
  }

  /* One line describing a catalog entry: kind, size, parameters, context. */
  function describeModel(entry) {
    const detail = entry.detail || {};
    const parts = [entry.kind_label];
    if (detail.quantization) parts.push(String(detail.quantization));
    if (detail.parameters) parts.push(`${format.compact(detail.parameters)} params`);
    else if (detail.size_label) parts.push(detail.size_label);
    if (detail.ctx_len) parts.push(`${format.compact(detail.ctx_len)} ${detail.ctx_unit === "bytes" ? "byte" : "token"} context`);
    if (entry.bytes) parts.push(format.bytes(entry.bytes));
    return parts.filter(Boolean).join(" · ");
  }

  Studio.ui = {
    tabs, segmented, badge, status, stat, meter, progress, empty, loading, render,
    selectOptions, callout, kv, code, field, button, iconButton, fillModels, describeModel,
  };
})();
