"use strict";

/* Studio overlays: floating panels, action menus, modal dialogs and the path
   picker.  They own focus, Escape handling and focus return; views only ask. */
(function () {
  const Studio = window.Studio;
  const { $, $$, h, icon, format } = Studio;

  /* -- floating panels ------------------------------------------------------ */
  let openPanel = null;

  /* Place a fixed-position element next to its anchor, kept inside the viewport. */
  function place(panel, anchor, align = "start") {
    const box = anchor.getBoundingClientRect ? anchor.getBoundingClientRect()
      : { left: anchor.x, right: anchor.x, top: anchor.y, bottom: anchor.y };
    const margin = 8;
    panel.style.left = "0px";
    panel.style.top = "0px";
    const width = panel.offsetWidth;
    const height = panel.offsetHeight;
    let left = align === "end" ? box.right - width : box.left;
    let top = box.bottom + 6;
    if (top + height > window.innerHeight - margin) top = Math.max(margin, box.top - height - 6);
    left = Math.max(margin, Math.min(left, window.innerWidth - width - margin));
    panel.style.left = `${Math.round(left)}px`;
    panel.style.top = `${Math.round(top)}px`;
  }

  function closeFloating() {
    if (openPanel) openPanel.close();
  }

  /* Show a panel until the user clicks elsewhere, presses Escape or resizes. */
  function float(panel, anchor, options = {}) {
    closeFloating();
    const returnFocus = document.activeElement;
    panel.classList.remove("hidden");
    place(panel, anchor, options.align);

    function close(restore = true) {
      if (openPanel !== handle) return;
      openPanel = null;
      document.removeEventListener("pointerdown", onPointer, true);
      document.removeEventListener("keydown", onKey, true);
      window.removeEventListener("resize", onResize);
      if (options.remove) panel.remove();
      else panel.classList.add("hidden");
      if (options.onClose) options.onClose();
      if (restore && returnFocus && returnFocus.focus && document.contains(returnFocus)) returnFocus.focus();
    }
    function onPointer(event) {
      if (panel.contains(event.target)) return;
      if (options.toggle && options.toggle.contains(event.target)) return;
      close(false);
    }
    function onKey(event) {
      if (event.key !== "Escape") return;
      event.stopPropagation();
      event.preventDefault();
      close();
    }
    function onResize() {
      place(panel, anchor, options.align);
    }
    const handle = { close, panel };
    openPanel = handle;
    document.addEventListener("pointerdown", onPointer, true);
    document.addEventListener("keydown", onKey, true);
    window.addEventListener("resize", onResize);
    return handle;
  }

  /* A menu of actions.  Items: { label, icon, hint, danger, disabled, onSelect },
     "separator", or { heading }. */
  function menu(anchor, items, options = {}) {
    const panel = h("div", { class: "popover", role: "menu", "aria-label": options.label || "Actions" });
    let handle = null;
    for (const item of items) {
      if (!item) continue;
      if (item === "separator") {
        panel.append(h("div", { class: "menu-separator", role: "separator" }));
      } else if (item.heading) {
        panel.append(h("div", { class: "menu-label", text: item.heading }));
      } else {
        panel.append(h("button", {
          class: `menu-item${item.danger ? " danger" : ""}`, type: "button", role: "menuitem",
          disabled: Boolean(item.disabled), title: item.title || null,
          onclick: () => {
            handle.close();
            if (item.onSelect) item.onSelect();
          },
        }, item.icon ? icon(item.icon, "small") : null, h("span", { text: item.label }),
        item.hint ? h("kbd", { text: item.hint }) : null));
      }
    }
    document.body.append(panel);
    handle = float(panel, anchor, { align: options.align, remove: true, onClose: options.onClose });
    panel.addEventListener("keydown", event => {
      const buttons = $$(".menu-item:not(:disabled)", panel);
      const index = buttons.indexOf(document.activeElement);
      if (event.key === "ArrowDown") buttons[(index + 1) % buttons.length].focus();
      else if (event.key === "ArrowUp") buttons[(index - 1 + buttons.length) % buttons.length].focus();
      else if (event.key === "Home") buttons[0].focus();
      else if (event.key === "End") buttons[buttons.length - 1].focus();
      else return;
      event.preventDefault();
    });
    const first = panel.querySelector(".menu-item:not(:disabled)");
    if (first) first.focus();
    return handle;
  }

  /* -- folder and file picker ----------------------------------------------- */
  const picker = { resolve: null, files: false, current: "", selected: "", home: "" };

  async function browse(path) {
    const list = $("picker-list");
    let data;
    try {
      data = await Studio.api(`/api/browse${Studio.query({ path, files: picker.files ? 1 : "" })}`);
    } catch (error) {
      Studio.toast(error.message, "error");
      return;
    }
    picker.current = data.path;
    picker.parent = data.parent;
    picker.home = data.home;
    picker.selected = "";
    $("picker-path").value = data.path;
    $("picker-up").disabled = data.parent === data.path;
    Studio.clear(list);
    for (const folder of data.directories) {
      list.append(h("button", { class: "picker-item", type: "button", onclick: () => browse(folder.path) },
        icon("folder", "small"), h("span", { class: "truncate", text: folder.name })));
    }
    for (const file of data.files || []) {
      const item = h("button", {
        class: "picker-item", type: "button",
        onclick: () => {
          for (const other of $$(".picker-item.selected", list)) other.classList.remove("selected");
          item.classList.add("selected");
          picker.selected = file.path;
          renderSelection();
        },
        ondblclick: () => finishPicker(file.path),
      }, icon("file", "small"), h("span", { class: "truncate", text: file.name }),
      h("span", { class: "meta", text: format.bytes(file.bytes) }));
      list.append(item);
    }
    if (!list.children.length) {
      list.append(Studio.ui.empty({ compact: true, text: picker.files ? "This folder is empty." : "No sub-folders here. You can use this folder." }));
    }
    renderSelection();
  }

  function renderSelection() {
    $("picker-selection").textContent = picker.selected || picker.current;
    $("picker-accept").textContent = picker.selected ? "Use this file" : "Use this folder";
  }

  function finishPicker(value) {
    const dialog = $("picker-dialog");
    if (dialog.open) dialog.close();
    const resolve = picker.resolve;
    picker.resolve = null;
    if (resolve) resolve(value);
  }

  /* Ask for a folder (or a file when options.files is set).  Resolves to a path or null. */
  function pickPath(options = {}) {
    if (picker.resolve) finishPicker(null);
    picker.files = Boolean(options.files);
    $("picker-title").textContent = options.title || (picker.files ? "Choose a folder or file" : "Choose a folder");
    $("picker-hint").textContent = options.hint || "";
    return new Promise(resolve => {
      picker.resolve = resolve;
      $("picker-dialog").showModal();
      browse(options.start || "");
    });
  }

  function initPicker() {
    $("picker-close").addEventListener("click", () => finishPicker(null));
    $("picker-cancel").addEventListener("click", () => finishPicker(null));
    $("picker-dialog").addEventListener("cancel", event => {
      event.preventDefault();
      finishPicker(null);
    });
    $("picker-accept").addEventListener("click", () => finishPicker(picker.selected || picker.current));
    $("picker-up").addEventListener("click", () => browse(picker.parent));
    $("picker-home").addEventListener("click", () => browse(picker.home));
    $("picker-workspace").addEventListener("click", () => browse(Studio.state.root || ""));
    $("picker-go").addEventListener("click", () => browse($("picker-path").value.trim()));
    $("picker-path").addEventListener("keydown", event => {
      if (event.key !== "Enter") return;
      event.preventDefault();
      browse($("picker-path").value.trim());
    });
  }

  /* Wire a "Browse" button to a text input. */
  function bindBrowse(buttonElement, input, options = {}) {
    buttonElement.addEventListener("click", async () => {
      const chosen = await pickPath({ ...options, start: input.value.trim() });
      if (chosen === null) return;
      input.value = chosen;
      input.dispatchEvent(new Event("change", { bubbles: true }));
    });
  }

  /* -- dialogs -------------------------------------------------------------- */
  let activeModal = null;

  /* Open a <dialog>; a button with data-close inside it, or Escape, closes it. */
  function openDialog(dialog) {
    if (!dialog.dataset.wired) {
      dialog.dataset.wired = "1";
      dialog.addEventListener("click", event => {
        if (event.target.closest("[data-close]")) dialog.close();
      });
    }
    if (!dialog.open) dialog.showModal();
  }

  function closeModal() {
    if (activeModal) activeModal.close();
  }

  function modal(options = {}) {
    const key = options.key || "";
    /* Repeated clicks on the same action focus the open panel instead of
       stacking another copy.  A different action replaces the old panel. */
    if (activeModal && key && activeModal.key === key && activeModal.dialog.open) {
      activeModal.dialog.focus();
      return { ...activeModal, reused: true };
    }
    if (activeModal) activeModal.close();
    const dialog = h("dialog", {
      class: `dialog${options.wide ? " wide" : ""}${options.narrow ? " narrow" : ""}`,
      tabindex: -1, "aria-label": options.title || "Dialog",
    });
    const body = h("div", { class: "dialog-body stack" }, options.content || null);
    const footer = h("div", { class: "dialog-footer" });
    const close = value => { if (dialog.open) dialog.close(value || ""); };
    dialog.append(
      h("div", { class: "dialog-header" }, h("div", {}, h("h2", { text: options.title || "" }), options.subtitle ? h("p", { class: "muted small", text: options.subtitle }) : null),
        h("button", { class: "icon-button", type: "button", "aria-label": "Close", onclick: () => close() }, icon("x"))),
      body, footer);
    const handle = { dialog, body, footer, close, key, reused: false };
    dialog.addEventListener("close", () => {
      if (activeModal === handle) activeModal = null;
      try {
        if (options.onClose) options.onClose(dialog.returnValue);
      } finally {
        dialog.remove();
      }
    }, { once: true });
    document.body.append(dialog);
    activeModal = handle;
    openDialog(dialog);
    return handle;
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", initPicker);
  else initPicker();

  Studio.overlay = { place, float, closeFloating, menu, pickPath, bindBrowse, openDialog, modal, closeModal };
})();
