"use strict";

/* Studio feedback: toasts, busy controls, clipboard and the single dialog stack.
   Every user-visible failure passes through here. */
(function () {
  const Studio = window.Studio;
  const { $, h, icon } = Studio;
  const TOAST_ICONS = { info: "info", success: "check", error: "alert" };
  let dialogQueue = Promise.resolve();

  function toast(message, kind = "info", timeout) {
    const region = $("toast-region");
    const tone = TOAST_ICONS[kind] ? kind : "info";
    const item = h("div", { class: `toast ${tone}`, role: tone === "error" ? "alert" : null },
      icon(TOAST_ICONS[tone]),
      h("div", { class: "toast-text", text: String(message) }),
      h("button", { class: "icon-button small", type: "button", "aria-label": "Dismiss", onclick: () => item.remove() },
        icon("x", "small")));
    region.append(item);
    while (region.children.length > 4) region.firstElementChild.remove();
    setTimeout(() => item.remove(), timeout || (tone === "error" ? 9000 : 4500));
    return item;
  }

  /* Run an async action for a control: disables it while pending and reports failures. */
  async function run(task, control) {
    if (control) {
      if (control.disabled) return undefined;
      control.disabled = true;
      control.classList.add("busy");
    }
    try {
      return await task();
    } catch (error) {
      if (error && error.name !== "AbortError") toast(error.message || String(error), "error");
      return undefined;
    } finally {
      if (control) {
        control.disabled = false;
        control.classList.remove("busy");
      }
    }
  }

  function ask(options) {
    const open = () => new Promise(resolve => {
      const dialog = $("confirm-dialog");
      const accept = $("confirm-accept");
      const cancel = $("confirm-cancel");
      const input = options.multiline ? $("confirm-textarea") : $("confirm-input");
      $("confirm-title").textContent = options.title || "Are you sure?";
      $("confirm-message").textContent = options.message || "";
      $("confirm-message").classList.toggle("hidden", !options.message);
      $("confirm-field").classList.toggle("hidden", !options.input);
      $("confirm-input").classList.toggle("hidden", Boolean(options.multiline));
      $("confirm-textarea").classList.toggle("hidden", !options.multiline);
      $("confirm-label").textContent = options.label || "";
      accept.textContent = options.confirmLabel || "Confirm";
      accept.className = `button ${options.danger ? "danger" : "primary"}`;
      cancel.textContent = options.cancelLabel || "Cancel";
      if (options.input) {
        input.value = options.value || "";
        input.placeholder = options.placeholder || "";
      }

      function finish(result) {
        accept.removeEventListener("click", onAccept);
        cancel.removeEventListener("click", onCancel);
        dialog.removeEventListener("cancel", onCancel);
        dialog.removeEventListener("keydown", onKey);
        if (dialog.open) dialog.close();
        resolve(result);
      }
      function onAccept() {
        finish(options.input ? input.value : true);
      }
      function onCancel(event) {
        if (event) event.preventDefault();
        finish(options.input ? null : false);
      }
      function onKey(event) {
        if (event.key === "Enter" && options.input && !options.multiline && event.target === input) {
          event.preventDefault();
          onAccept();
        }
      }
      accept.addEventListener("click", onAccept);
      cancel.addEventListener("click", onCancel);
      dialog.addEventListener("cancel", onCancel);
      dialog.addEventListener("keydown", onKey);
      dialog.showModal();
      (options.input ? input : options.danger ? cancel : accept).focus();
      if (options.input && !options.multiline) input.select();
    });
    // One dialog at a time: a second request waits for the first to close.
    const result = dialogQueue.then(open);
    dialogQueue = result.catch(() => undefined);
    return result;
  }

  /* Resolve to true when the user confirms.  Destructive actions pass danger: true. */
  function confirm(options) {
    return ask(typeof options === "string" ? { message: options } : { ...options, input: false });
  }

  /* Resolve to the entered text, or null when cancelled. */
  function prompt(options) {
    return ask({ confirmLabel: "Save", ...options, input: true });
  }

  async function copy(text, label = "Copied to clipboard") {
    try {
      await navigator.clipboard.writeText(String(text));
      toast(label, "success");
      return true;
    } catch (error) {
      toast("The browser blocked clipboard access; select the text and copy it manually.", "error");
      return false;
    }
  }

  window.addEventListener("unhandledrejection", event => {
    const reason = event.reason;
    if (!reason || reason.name === "AbortError") return;
    console.error(reason);
    toast(reason.message || String(reason), "error");
  });

  Object.assign(Studio, { toast, run, confirm, prompt, copy });
})();
