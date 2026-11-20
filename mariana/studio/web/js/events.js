"use strict";

/* Studio events: the in-page event bus and keyboard shortcut registry.
   Views subscribe with Studio.on and never reach into another view. */
(function () {
  const Studio = window.Studio;
  const listeners = new Map();
  const shortcuts = new Map();

  function on(name, handler) {
    if (!listeners.has(name)) listeners.set(name, new Set());
    listeners.get(name).add(handler);
    return () => listeners.get(name).delete(handler);
  }

  function emit(name, payload) {
    for (const handler of Array.from(listeners.get(name) || [])) {
      try {
        handler(payload);
      } catch (error) {
        console.error(`Listener for "${name}" failed`, error);
      }
    }
  }

  function shortcut(combo, handler) {
    const key = combo.toLowerCase();
    if (!shortcuts.has(key)) shortcuts.set(key, new Set());
    shortcuts.get(key).add(handler);
  }

  function isTyping(target) {
    return Boolean(target && target.closest && target.closest("input, textarea, select, [contenteditable='true']"));
  }

  document.addEventListener("keydown", event => {
    const modifier = event.ctrlKey || event.metaKey;
    if (!modifier && (event.altKey || isTyping(event.target))) return;
    const key = event.key.toLowerCase();
    const combo = `${modifier ? "mod+" : ""}${event.shiftKey && modifier ? "shift+" : ""}${key}`;
    const handlers = shortcuts.get(combo);
    if (!handlers) return;
    let handled = false;
    for (const handler of Array.from(handlers)) {
      try {
        if (handler(event) !== false) handled = true;
      } catch (error) {
        console.error(`Shortcut ${combo} failed`, error);
      }
    }
    if (handled) event.preventDefault();
  });

  Object.assign(Studio, { on, emit, shortcut, isTyping });
})();
