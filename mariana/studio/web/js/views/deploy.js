"use strict";

(function () {
  const Studio = window.Studio;
  const { $, h, api, format } = Studio;
  const state = { exports: [], checkpoints: [], loaded: false };

  async function load() {
    const data = await api("/api/deploy/exports");
    state.exports = data.exports;
    state.checkpoints = data.checkpoints;
    state.loaded = true;
    renderCheckpoints();
    renderArtifacts();
    await Studio.models.ensure();
    Studio.ui.fillModels($("deploy-api-model"), { filter: entry => entry.available && entry.capabilities.chat, includeUnavailable: false });
    renderApi();
  }

  function renderCheckpoints() {
    const select = $("deploy-model");
    const previous = select.value;
    Studio.clear(select);
    if (!state.checkpoints.length) select.append(h("option", { value: "", text: "No Mariana DRV-LM checkpoints found" }));
    for (const checkpoint of state.checkpoints) select.append(h("option", { value: checkpoint.id, text: `${checkpoint.name}${checkpoint.exportable ? "" : " · not exportable"}`, disabled: !checkpoint.exportable }));
    const values = Array.from(select.options).filter(option => !option.disabled).map(option => option.value);
    if (previous && values.includes(previous)) select.value = previous;
    const checkpoint = state.checkpoints.find(item => item.id === select.value);
    $("deploy-export").disabled = !checkpoint || !checkpoint.exportable;
    if (checkpoint && !$("deploy-name").value) $("deploy-name").value = checkpoint.name.replace(/\.pt$/i, "");
  }

  function renderArtifacts() {
    const root = Studio.ui.render($("deploy-artifacts"));
    if (!state.exports.length) {
      root.append(Studio.ui.empty({ icon: "deploy", title: "No exports yet", text: "Export a dense DRV-LM checkpoint above, or use third-party models directly from Chat." }));
      return;
    }
    for (const artifact of state.exports) {
      root.append(Studio.components.listRow({
        leading: h("div", { class: "result-icon" }, Studio.icon("file")),
        title: artifact.name,
        subtitle: `${artifact.format.toUpperCase()} · ${format.bytes(artifact.bytes)} · ${format.ago(artifact.modified)}`,
        trailing: h("div", { class: "row wrap" },
          Studio.ui.button("Chat", { small: true, onclick: () => Studio.views.go("chat", { model: artifact.path, new: "1" }) }),
          Studio.ui.button("Copy path", { small: true, onclick: () => Studio.copy(artifact.path, "Artifact path copied") }),
          artifact.format === "gguf" ? Studio.ui.button("Create Ollama model", { small: true, kind: "primary", onclick: () => createOllama(artifact) }) : null),
      }));
    }
  }

  async function createOllama(artifact) {
    const name = await Studio.prompt({ title: "Create Ollama model", label: "Model name", value: artifact.name.replace(/\.gguf$/i, "").toLowerCase(), confirmLabel: "Create" });
    if (!name) return;
    const result = await Studio.run(() => api("/api/deploy/ollama", { body: { file: artifact.name, name } }));
    if (result) { Studio.toast("Ollama registration started; follow it in Jobs.", "success"); Studio.jobs.refresh(); }
  }

  function renderApi() {
    const model = $("deploy-api-model").value || "MODEL_ID";
    const base = `${window.location.origin}/v1`;
    const curl = `curl ${base}/chat/completions \\\n  -H 'Content-Type: application/json' \\\n  -d '{\n    "model": ${JSON.stringify(model)},\n    "messages": [{"role": "user", "content": "Hello"}],\n    "stream": false\n  }'`;
    const python = `from openai import OpenAI\n\nclient = OpenAI(base_url=${JSON.stringify(base)}, api_key="not-needed")\nreply = client.chat.completions.create(\n    model=${JSON.stringify(model)},\n    messages=[{"role": "user", "content": "Hello"}],\n)\nprint(reply.choices[0].message.content)`;
    const root = Studio.ui.render($("deploy-api-snippet"));
    const curlTab = h("button", { type: "button", role: "tab", "data-tab": "curl", "aria-controls": "deploy-curl", "aria-selected": "true", text: "cURL" });
    const pythonTab = h("button", { type: "button", role: "tab", "data-tab": "python", "aria-controls": "deploy-python", "aria-selected": "false", text: "Python" });
    const tabs = h("div", { class: "tabs api-tabs", role: "tablist", "aria-label": "API example" }, curlTab, pythonTab);
    const curlPanel = h("div", { id: "deploy-curl", role: "tabpanel" }, Studio.ui.code(curl, "bash"));
    const pythonPanel = h("div", { id: "deploy-python", role: "tabpanel", class: "hidden" }, Studio.ui.code(python, "python"));
    root.append(tabs, curlPanel, pythonPanel, h("p", { class: "help", text: "Keep this API on loopback. It uses the same local models and saved remote endpoints as Chat; no API key is required on loopback." }));
    Studio.ui.tabs(tabs);
  }

  Studio.views.register("deploy", {
    title: "Deploy", subtitle: "Export portable artifacts and serve every model through one API",
    glyph: "deploy", hint: "Export and serve models",
    init() {
      $("deploy-model").addEventListener("change", () => { $("deploy-name").value = ""; renderCheckpoints(); });
      $("deploy-api-model").addEventListener("change", renderApi);
      $("deploy-refresh").addEventListener("click", event => Studio.run(async () => {
        await load();
        Studio.toast("Exports and checkpoints refreshed.", "success");
      }, event.currentTarget));
      $("deploy-export").addEventListener("click", async event => {
        const result = await Studio.run(() => api("/api/deploy/export", { body: {
          model: $("deploy-model").value, format: $("deploy-format").value, dtype: $("deploy-dtype").value, name: $("deploy-name").value.trim(),
        } }), event.currentTarget);
        if (result) { Studio.toast("Export started; follow it in Jobs.", "success"); Studio.jobs.refresh(); }
      });
      Studio.on("job:export", job => { if (job.status === "completed") load().catch(error => Studio.toast(error.message, "error")); });
      Studio.on("job:deploy", job => { if (job.status === "completed") Studio.models.load(true); });
    },
    async activate(parameters) {
      if (!state.loaded) Studio.ui.render($("deploy-artifacts"), Studio.ui.loading("Loading exported artifacts…"));
      try {
        await load();
      } catch (error) {
        if (state.loaded) { Studio.toast(error.message, "error"); return; }
        $("deploy-export").disabled = true;
        Studio.ui.render($("deploy-model"), h("option", { value: "", text: "Could not load checkpoints" }));
        Studio.ui.render($("deploy-api-model"), h("option", { value: "", text: "Could not load models" }));
        Studio.ui.render($("deploy-artifacts"), Studio.ui.empty({ icon: "alert", title: "Could not load deploy data", text: error.message,
          actions: [Studio.ui.button("Try again", { kind: "primary", onclick: () => Studio.run(load) })] }));
        return;
      }
      if (parameters.model && state.checkpoints.some(item => item.id === parameters.model)) {
        $("deploy-model").value = parameters.model; $("deploy-name").value = ""; renderCheckpoints();
      }
    },
  });
})();
