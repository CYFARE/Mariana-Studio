"use strict";

/* Studio.form: declarative form controls shared by the workspaces.
   A spec names the key, type, value and select options; these helpers build
   the control and expose its typed value. */
(function () {
  const Studio = window.Studio;
  const { h } = Studio;

  const NUMERIC = new Set(["number", "range"]);

  function attributes(spec) {
    return {
      disabled: Boolean(spec.disabled), required: Boolean(spec.required),
      name: spec.name, "aria-label": spec.ariaLabel || null,
      dataset: spec.key ? { key: spec.key } : null,
    };
  }

  /* Build the native control; the caller decides where it goes. */
  function control(spec) {
    const type = spec.type || "text";
    if (type === "select") {
      return Studio.ui.selectOptions(h("select", attributes(spec)), spec.options, spec.value);
    }
    if (type === "textarea") {
      return h("textarea", {
        ...attributes(spec), rows: spec.rows || 3, value: spec.value === undefined ? "" : spec.value,
        placeholder: spec.placeholder, spellcheck: spec.spellcheck === undefined ? false : spec.spellcheck,
        readOnly: spec.readOnly,
      });
    }
    if (type === "checkbox") return h("input", { ...attributes(spec), type, checked: Boolean(spec.value) });
    return h("input", {
      ...attributes(spec), type, value: spec.value === undefined ? "" : spec.value,
      min: spec.min, max: spec.max, step: spec.step, placeholder: spec.placeholder,
      autocomplete: spec.autocomplete, spellcheck: spec.spellcheck, readOnly: spec.readOnly,
    });
  }

  /* Labelled field.  onchange receives the typed value, not the event. */
  function field(spec) {
    const input = control(spec);
    if (spec.onchange) input.addEventListener("change", () => spec.onchange(value(input)));
    return Studio.ui.field(spec.label, input, spec.help);
  }

  /* Checkbox with its own label as the click target. */
  function check(spec) {
    const input = control({ ...spec, type: "checkbox" });
    if (spec.onchange) input.addEventListener("change", () => spec.onchange(input.checked));
    return h("label", { class: "check" }, input, spec.label);
  }

  /* Typed value of a control built here, or of any control of the same types. */
  function value(input) {
    if (input.type === "checkbox") return input.checked;
    if (NUMERIC.has(input.type)) return Number(input.value);
    return input.value;
  }

  Studio.form = { control, field, check, value };
})();
