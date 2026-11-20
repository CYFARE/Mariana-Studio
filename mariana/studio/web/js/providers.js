"use strict";

/* Studio.providers: one endpoint flow for every model picker.
   Chat, Agents, the IDE and the Models view all open the same dialog and the
   same provider menu, so an external API added anywhere appears everywhere. */
(function () {
  const Studio = window.Studio;
  const { api, h } = Studio;

  let cached = null;
  let pending = null;

  /* Provider presets are small metadata; load them once and share the cache. */
  function ensure(force = false) {
    if (force) cached = null;
    if (cached) return Promise.resolve(cached);
    if (pending) return pending;
    const request = api("/api/models/providers").then(data => {
      cached = data.providers || [];
      return cached;
    }).finally(() => {
      if (pending === request) pending = null;
    });
    pending = request;
    return request;
  }

  async function refreshCatalog() {
    try {
      await Studio.models.load(true);
      Studio.toast("Model catalog refreshed.", "success");
    } catch (error) {
      Studio.toast(error.message, "error");
    }
  }

  /* The single add/edit endpoint dialog used by every view. */
  async function addEndpointDialog(options = {}) {
    let providers;
    try {
      providers = await ensure();
    } catch (error) {
      Studio.toast(error.message, "error");
      return;
    }
    if (!providers.length) {
      Studio.toast("No provider presets are available; reopen Studio and try again.", "error");
      return;
    }
    const existing = options.existing || null;
    const current = existing || {};
    let preset = providers.find(item => item.id === current.provider) || providers[0];
    const provider = h("select", { "aria-label": "Provider" },
      providers.map(item => h("option", { value: item.id, text: item.name, selected: item.id === preset.id })));
    const name = h("input", { value: current.name || "", required: true });
    const url = h("input", { value: current.base_url || preset.base_url || "", placeholder: "https://api.example.com/v1" });
    const model = h("input", {
      value: current.model || preset.default_model || "", required: true, list: "studio-endpoint-models",
    });
    const dataList = h("datalist", { id: "studio-endpoint-models" },
      (preset.models || []).map(value => h("option", { value })));
    const key = h("input", {
      type: "password", value: "", autocomplete: "off",
      placeholder: current.has_api_key ? "Leave blank to keep the stored key" : "API key (optional for local servers)",
    });
    const keyEnv = h("input", { value: current.api_key_env || preset.key_env || "", placeholder: "Environment variable" });
    const context = h("input", { type: "number", min: 0, value: current.context_length || "" });
    const output = h("input", { type: "number", min: 0, value: current.max_output_tokens || "" });
    const headers = h("textarea", { rows: 2, value: JSON.stringify(current.headers || {}), spellcheck: false });
    const fallback = h("input", { type: "checkbox", checked: current.refusal_fallback !== false });
    const note = h("p", {
      class: "help",
      text: preset.note || (preset.key_present ? `${preset.key_env} is available in the environment.` : ""),
    });
    const view = Studio.overlay.modal({
      key: `endpoint:${current.id || "new"}`,
      title: existing ? "Edit endpoint" : "Add model provider",
      subtitle: "Connect one model on an Ollama, OpenAI-compatible or Anthropic server. It becomes available in Chat, Agents and the IDE.",
      wide: true,
    });
    if (view.reused) return;

    let fetchedModels = [];
    const fillFromModel = () => {
      const wanted = model.value.trim();
      const entry = fetchedModels.find(item => (item.id || item.name) === wanted);
      if (!entry) return;
      if (!context.value && entry.context_length) context.value = entry.context_length;
      if (!output.value && entry.max_output_tokens) output.value = entry.max_output_tokens;
    };
    model.addEventListener("change", fillFromModel);
    model.addEventListener("blur", fillFromModel);
    const modelRow = h("div", { class: "input-action" }, model,
      Studio.ui.button("Fetch models", {
        onclick: async event => {
          const result = await Studio.run(
            () => api("/api/models/providers/models", { body: config(false) }),
            event.currentTarget,
          );
          if (!result) return;
          fetchedModels = result.models || [];
          Studio.ui.render(dataList, fetchedModels.map(item => h("option", { value: item.id || item.name || item })));
          if (fetchedModels.length && !model.value) model.value = fetchedModels[0].id || fetchedModels[0].name || fetchedModels[0];
          fillFromModel();
          Studio.toast(`Found ${fetchedModels.length} model(s).`, "success");
        },
      }));

    function applyPreset() {
      preset = providers.find(item => item.id === provider.value) || providers[0];
      if (!existing || !url.value) url.value = preset.base_url || "";
      keyEnv.value = preset.key_env || "";
      if (!existing && preset.default_model) model.value = preset.default_model;
      Studio.ui.render(dataList, (preset.models || []).map(value => h("option", { value })));
      note.textContent = preset.note || (preset.key_present ? `${preset.key_env} is available in the environment.` : "");
      const holder = fallback.closest("label");
      if (holder) holder.classList.toggle("hidden", preset.kind !== "anthropic");
    }

    function config(includeHeaders = true) {
      let parsedHeaders = {};
      if (includeHeaders) {
        try {
          parsedHeaders = JSON.parse(headers.value || "{}");
        } catch (error) {
          throw new Error("Custom headers must be a JSON object");
        }
      }
      return {
        id: current.id || "", provider: provider.value, kind: preset.kind,
        name: name.value.trim(), base_url: url.value.trim(), model: model.value.trim(),
        api_key: key.value, api_key_env: keyEnv.value.trim(), headers: parsedHeaders,
        context_length: Number(context.value) || 0,
        max_output_tokens: Number(output.value) || 0,
        refusal_fallback: fallback.checked,
      };
    }

    provider.addEventListener("change", applyPreset);
    view.body.append(
      Studio.ui.field("Provider", provider),
      Studio.ui.field("Display name", name),
      Studio.ui.field("Base URL", url),
      Studio.ui.field("Model", modelRow),
      dataList,
      note,
      h("div", { class: "form-grid" },
        Studio.ui.field("API key", key, "Keys are write-only and stored with mode 0600."),
        Studio.ui.field("Key environment variable", keyEnv)),
      h("div", { class: "form-grid" },
        Studio.ui.field("Context tokens (optional)", context),
        Studio.ui.field("Maximum output tokens (optional)", output)),
      Studio.ui.field("Custom headers (JSON)", headers),
      h("label", { class: "check" }, fallback, "Enable Anthropic server-side refusal fallback for supported models"),
    );
    applyPreset();
    view.footer.append(
      Studio.ui.button("Cancel", { onclick: view.close }),
      Studio.ui.button("Save provider", {
        kind: "primary",
        onclick: async event => {
          try {
            const result = await Studio.run(
              () => api("/api/models/endpoints", { body: config() }),
              event.currentTarget,
            );
            if (!result) return;
            view.close();
            await Studio.models.load(true);
            if (options.onSaved) await options.onSaved(result.endpoints || Studio.models.endpoints);
            Studio.toast("Provider saved. It is available in every model picker.", "success");
          } catch (error) {
            Studio.toast(error.message, "error");
          }
        },
      }),
    );
  }

  /* The shared globe menu: add, manage, refresh. */
  function menu(anchor, options = {}) {
    Studio.overlay.menu(anchor, [
      { heading: "Model providers" },
      { label: "Add provider…", icon: "globe", onSelect: () => addEndpointDialog({ onSaved: options.onSaved }) },
      { label: "Manage endpoints", icon: "settings", onSelect: () => Studio.views.go("models", { tab: "endpoints" }) },
      { label: "Refresh catalog", icon: "refresh", onSelect: refreshCatalog },
    ], { label: "Model providers", align: options.align || "end" });
  }

  Studio.providers = { ensure, addEndpointDialog, menu, refreshCatalog };
})();
