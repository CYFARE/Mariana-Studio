"use strict";

(function () {
  const Studio = window.Studio;
  const { $, h, api, format } = Studio;
  const state = { featured: null, runtimes: {}, endpoints: [], settings: null };
  let tabs;

  function refreshCatalog(force = false) {
    if (!Studio.models.loaded) {
      Studio.ui.render($("models-list"), h("div", { class: "span-all" }, Studio.ui.loading("Loading model library…")));
      Studio.ui.render($("endpoints-list"), h("div", { class: "span-all" }, Studio.ui.loading("Loading endpoints…")));
    }
    return (force ? Studio.models.load(true) : Studio.models.ensure()).then(models => {
      state.endpoints = Studio.models.endpoints;
      renderLibrary();
      renderEndpoints();
      return models;
    }).catch(error => {
      if (Studio.models.loaded) Studio.toast(error.message, "error");
      else {
        Studio.ui.render($("models-list"), h("div", { class: "span-all" }, Studio.ui.empty({ icon: "alert", title: "Could not load models", text: error.message,
          actions: [Studio.ui.button("Try again", { kind: "primary", onclick: () => refreshCatalog(true) })] })));
        Studio.ui.render($("endpoints-list"), h("div", { class: "span-all" }, Studio.ui.empty({ icon: "alert", title: "Could not load endpoints", text: error.message })));
      }
      return null;
    });
  }

  function modelActions(entry, options = {}) {
    const navigate = action => () => { if (options.before) options.before(); action(); };
    const actions = [];
    if (entry.capabilities.chat) actions.push(Studio.ui.button("Chat", { small: true, kind: "primary", disabled: !entry.available, onclick: navigate(() => Studio.views.go("chat", { model: entry.id, new: "1" })) }));
    if (entry.capabilities.finetune) actions.push(Studio.ui.button("Fine-tune", { small: true, onclick: navigate(() => Studio.views.go("finetune", { base: entry.id })) }));
    if (entry.kind === "mariana-drv") actions.push(Studio.ui.button("Export", { small: true, onclick: navigate(() => Studio.views.go("deploy", { model: entry.id })) }));
    if (options.details !== false) actions.push(Studio.ui.button("Details", { small: true, kind: "ghost", onclick: () => describe(entry) }));
    return actions;
  }

  function modelCard(entry) {
    const detail = Studio.ui.describeModel(entry);
    return Studio.components.resultCard({
      glyph: entry.source === "endpoint" ? "globe" : "models",
      title: entry.name,
      subtitle: `${entry.source_label} · ${entry.kind_label}`,
      description: entry.available ? (detail || entry.path || "Ready to use") : (entry.reason || "This model is not currently available"),
      meta: entry.bytes ? `${format.bytes(entry.bytes)}${entry.modified ? ` · ${format.ago(entry.modified)}` : ""}` : "",
      status: Studio.ui.badge(entry.available ? "Available" : "Unavailable", entry.available ? "completed" : "warning"),
      actions: modelActions(entry),
      unavailable: !entry.available,
    });
  }

  function renderLibrary() {
    const query = $("models-search").value.trim().toLowerCase();
    const source = $("models-source").value;
    const capability = $("models-capability").value;
    const entries = Studio.models.list.filter(entry => {
      const haystack = `${entry.name} ${entry.kind_label} ${entry.source_label} ${entry.path}`.toLowerCase();
      return (!query || query.split(/\s+/).every(word => haystack.includes(word))) && (!source || entry.source === source)
        && (!capability || entry.capabilities[capability]);
    });
    $("models-count").textContent = entries.length === Studio.models.list.length ? String(entries.length) : `${entries.length}/${Studio.models.list.length}`;
    $("models-count").title = `${entries.length} of ${Studio.models.list.length} models`;
    const totals = new Map();
    for (const entry of Studio.models.list) totals.set(entry.source, (totals.get(entry.source) || 0) + 1);
    const sources = Array.from(new Set(Studio.models.list.map(entry => entry.source)));
    Studio.ui.selectOptions($("models-source"), [["", `All sources (${Studio.models.list.length})`], ...sources.map(name => {
      const entry = Studio.models.list.find(item => item.source === name);
      return [name, `${entry.source_label} (${totals.get(name)})`];
    })], source);
    const list = Studio.ui.render($("models-list"));
    if (!entries.length) {
      list.append(h("div", { class: "span-all" }, Studio.ui.empty({
        icon: "models", title: Studio.models.list.length ? "No models match" : "No models found",
        text: Studio.models.list.length ? "Clear the filters or refresh discovery." : "Download an open model, add an endpoint, or start Ollama.",
        actions: [Studio.ui.button("Discover models", { kind: "primary", onclick: () => tabs.select("discover") })],
      })));
      return;
    }
    list.append(...entries.map(modelCard));
  }

  async function describe(entry) {
    const view = Studio.overlay.modal({ key: `model:${entry.id}`, title: entry.name, subtitle: `${entry.source_label} · ${entry.kind_label}`, content: Studio.ui.loading("Reading model metadata…") });
    if (view.reused) return;
    try {
      const detail = await api(`/api/models/describe${Studio.query({ id: entry.id })}`);
      const item = detail.model || detail;
      const rows = [
        ["Format", item.kind_label], ["Source", item.source_label], ["Path", item.path],
        ["Size", item.bytes ? format.bytes(item.bytes) : null],
        ...Object.entries(item.detail || {}).filter(([, value]) => ["string", "number", "boolean"].includes(typeof value))
          .map(([key, value]) => [format.title(key), typeof value === "boolean" ? (value ? "Yes" : "No") : String(value)]),
      ];
      Studio.ui.render(view.body, Studio.ui.kv(rows), item.reason ? Studio.ui.callout("warning", item.reason) : null,
        h("div", { class: "row wrap" }, modelActions(item, { before: view.close, details: false }).concat(Studio.ui.button("Copy model id", { small: true, onclick: () => Studio.copy(item.id, "Model id copied") }))));
    } catch (error) {
      Studio.ui.render(view.body, Studio.ui.callout("danger", error.message));
    }
  }

  function resultCard(item) {
    return Studio.components.resultCard({
      glyph: "download",
      title: item.title || item.repo,
      subtitle: item.title && item.title !== item.repo ? item.repo : "",
      description: item.summary || `${format.compact(item.downloads || 0)} downloads · ${format.compact(item.likes || 0)} likes${item.gated ? " · gated" : ""}`,
      tags: item.tags,
      actions: [Studio.ui.button("Choose files", { kind: "primary", small: true, onclick: () => repository(item.repo, item.format) })],
    });
  }

  async function search() {
    const query = $("discover-query").value.trim();
    const modelFormat = $("discover-format").value;
    const results = Studio.ui.render($("discover-results"), h("div", { class: "span-all" }, Studio.ui.loading("Searching Hugging Face…")));
    try {
      const data = await api(`/api/models/search${Studio.query({ q: query, format: modelFormat })}`);
      Studio.ui.render(results, data.results.map(resultCard));
      if (!data.results.length) results.append(h("div", { class: "span-all" }, Studio.ui.empty({ title: "No repositories found", text: "Try a model family such as Qwen, Gemma or SmolLM." })));
    } catch (error) {
      Studio.ui.render(results, h("div", { class: "span-all" }, Studio.ui.callout("danger", error.message)));
    }
  }

  async function repository(repo, preferred) {
    const view = Studio.overlay.modal({ key: `repo:${repo}`, title: repo, subtitle: "Choose a portable model file or the complete Transformers weights.", wide: true, content: Studio.ui.loading("Reading repository files…") });
    if (view.reused) return;
    try {
      const data = await api(`/api/models/repository${Studio.query({ repo })}`);
      const choices = data.gguf.length ? data.gguf : [];
      const initial = preferred === "transformers" && data.transformers.length ? "transformers" : choices.length ? "gguf" : "transformers";
      const selectedFile = h("select", { "aria-label": "GGUF file" }, choices.map(file => h("option", { value: file.path, text: `${file.path} · ${format.bytes(file.bytes)}` })));
      const ggufChoice = h("label", { class: "check" }, h("input", { type: "radio", name: "repo-format", value: "gguf", checked: initial === "gguf", disabled: !choices.length }), "GGUF", selectedFile);
      const hfChoice = h("label", { class: "check" }, h("input", { type: "radio", name: "repo-format", value: "transformers", checked: initial === "transformers", disabled: !data.transformers.length }), `Transformers weights · ${format.bytes(data.transformers_bytes)}`);
      const update = () => { selectedFile.disabled = !ggufChoice.querySelector("input").checked; };
      ggufChoice.querySelector("input").addEventListener("change", update);
      hfChoice.querySelector("input").addEventListener("change", update);
      update();
      Studio.ui.render(view.body,
        Studio.ui.kv([["Pipeline", data.pipeline], ["Licence", data.license], ["Downloads", format.compact(data.downloads)], ["Revision", data.revision]]),
        data.gated ? Studio.ui.callout("warning", "This repository is gated. Accept its licence on Hugging Face and configure HF_TOKEN before downloading.") : null,
        h("div", { class: "stack" }, ggufChoice, hfChoice),
        !choices.length && !data.transformers.length ? Studio.ui.callout("warning", "No supported GGUF or Transformers weights were found in this repository.") : null);
      view.footer.append(Studio.ui.button("Download", { kind: "primary", disabled: !choices.length && !data.transformers.length, onclick: async event => {
        const modelFormat = hfChoice.querySelector("input").checked ? "transformers" : "gguf";
        const body = { source: "huggingface", repo, format: modelFormat, files: modelFormat === "gguf" ? [selectedFile.value] : [] };
        const result = await Studio.run(() => api("/api/models/download", { body }), event.currentTarget);
        if (result) { view.close(); Studio.toast("Download started; follow it in the job centre.", "success"); Studio.jobs.refresh(); }
      } }));
    } catch (error) {
      Studio.ui.render(view.body, Studio.ui.callout("danger", error.message));
    }
  }

  async function pullOllama(name, control) {
    const result = await Studio.run(() => api("/api/models/download", { body: { source: "ollama", name } }), control);
    if (result) { Studio.toast(`Pulling ${name}; follow progress in Jobs.`, "success"); Studio.jobs.refresh(); }
  }

  function renderDiscover() {
    if (!state.featured) return;
    if (!$("discover-results").children.length) $("discover-results").append(...state.featured.huggingface.map(resultCard));
    Studio.ui.render($("discover-ollama"), state.featured.ollama.map(item => Studio.components.resultCard({
      glyph: "server",
      title: item.name,
      description: item.summary,
      actions: [Studio.ui.button("Pull with Ollama", { kind: "primary", small: true, onclick: event => pullOllama(item.name, event.currentTarget) })],
    })));
  }

  function endpointDialog(existing) {
    Studio.providers.addEndpointDialog({
      existing,
      onSaved: async endpoints => {
        state.endpoints = endpoints || Studio.models.endpoints;
        await refreshCatalog(false);
      },
    });
  }

  function renderEndpoints() {
    const list = Studio.ui.render($("endpoints-list"));
    if (!state.endpoints.length) {
      list.append(h("div", { class: "span-all" }, Studio.ui.empty({ icon: "globe", title: "No saved endpoints", text: "Connect OpenAI, Anthropic, Ollama, llama.cpp, LM Studio, vLLM and compatible servers.", actions: [Studio.ui.button("Add endpoint", { kind: "primary", onclick: () => endpointDialog() })] })));
      return;
    }
    for (const endpoint of state.endpoints) {
      list.append(Studio.components.resultCard({
        glyph: "globe",
        title: endpoint.name,
        subtitle: endpoint.kind,
        description: `${endpoint.model} · ${endpoint.base_url || "Provider default"}`,
        meta: endpoint.has_api_key ? "API key saved or available in the environment." : "No stored API key.",
        actions: [
          Studio.ui.button("Chat", { small: true, kind: "primary", onclick: () => Studio.views.go("chat", { model: endpoint.id, new: "1" }) }),
          Studio.ui.button("Test", { small: true, onclick: async event => {
            const result = await Studio.run(() => api("/api/models/test", { body: { id: endpoint.id } }), event.currentTarget);
            if (result) Studio.toast(`Endpoint replied in ${format.seconds(result.elapsed)}: ${result.reply}`, "success");
          } }),
          Studio.ui.button("Edit", { small: true, onclick: () => endpointDialog(endpoint) }),
          Studio.ui.button("Delete", { small: true, kind: "danger", onclick: async () => {
            if (!await Studio.confirm({ title: "Delete endpoint?", message: `Remove ${endpoint.name} and its stored credentials? Downloaded models are not affected.`, confirmLabel: "Delete", danger: true })) return;
            const result = await Studio.run(() => api(`/api/models/endpoints/${encodeURIComponent(endpoint.id)}`, { method: "DELETE" }));
            if (result) { state.endpoints = result.endpoints; await refreshCatalog(true); }
          } }),
        ],
      }));
    }
  }

  async function loadRuntimes() {
    const [runtimeData, settingData] = await Promise.all([api("/api/models/runtimes"), api("/api/settings")]);
    state.runtimes = runtimeData.runtimes;
    state.settings = settingData.settings;
    renderRuntimes();
    renderSettings();
  }

  function renderRuntimes() {
    const list = Studio.ui.render($("runtimes-list"));
    for (const [id, runtime] of Object.entries(state.runtimes)) {
      if (id === "loaded") continue;
      const ready = Boolean(runtime.running || runtime.installed || runtime.available);
      list.append(Studio.components.card({
        title: runtime.label || format.title(id),
        description: runtime.hint || runtime.path || runtime.url || (runtime.models !== undefined ? `${runtime.models} model(s) discovered` : "Available"),
        actions: Studio.ui.badge(runtime.running ? "Running" : ready ? "Installed" : "Not found", runtime.running || ready ? "completed" : "warning"),
        children: id === "ollama" && runtime.installed ? [Studio.ui.button(runtime.running ? "Stop server" : "Start server", {
          small: true, kind: runtime.running ? "danger" : "primary", onclick: async event => {
            await Studio.run(() => api(`/api/models/runtimes/ollama/${runtime.running ? "stop" : "start"}`, { body: {} }), event.currentTarget);
            await Promise.all([loadRuntimes(), refreshCatalog(true)]);
          },
        })] : [],
      }));
    }
    list.append(Studio.components.card({
      title: "Loaded models",
      description: `${(state.runtimes.loaded || []).length} model runtime(s) currently hold memory.`,
      actions: Studio.ui.button("Unload local models", { small: true, onclick: async event => {
        const result = await Studio.run(() => api("/api/models/unload", { body: {} }), event.currentTarget);
        if (result) { Studio.toast("Local model memory released.", "success"); loadRuntimes(); }
      } }),
    }));
  }

  function renderSettings() {
    const settings = state.settings;
    const form = Studio.ui.render($("settings-form"));
    const dirs = h("textarea", { rows: 3, value: settings.model_dirs.join("\n"), spellcheck: false });
    const discover = h("input", { type: "checkbox", checked: settings.discover_local });
    const runtime = h("select", {}, ["auto", "llama-server", "llama-cpp-python"].map(value => h("option", { value, text: value, selected: settings.gguf_runtime === value })));
    const binary = h("input", { value: settings.llama_server_path, placeholder: "Auto-detect llama-server" });
    const layers = h("input", { value: String(settings.llama_gpu_layers), placeholder: "auto" });
    const url = h("input", { value: settings.ollama_url, placeholder: "http://127.0.0.1:11434" });
    const idle = h("input", { type: "number", min: 0, max: 1440, step: 1, value: settings.idle_unload_minutes });
    const autoInstall = h("input", { type: "checkbox", checked: settings.auto_install_dependencies });
    const checkUpdates = h("input", { type: "checkbox", checked: settings.check_for_updates !== false });
    const osAccess = h("input", { type: "checkbox", checked: settings.allow_os_access === true });
    form.append(h("h2", { text: "Discovery and runtime settings" }),
      Studio.ui.field("Extra model folders (one per line)", dirs), h("label", { class: "check" }, discover, "Discover Ollama, llama.cpp, LM Studio, GPT4All, Jan and Hugging Face cache models"),
      h("div", { class: "form-grid three" }, Studio.ui.field("GGUF runtime", runtime), Studio.ui.field("llama-server binary", binary), Studio.ui.field("GPU layers", layers)),
      h("div", { class: "form-grid" }, Studio.ui.field("Ollama URL", url), Studio.ui.field("Unload managed runtime after idle minutes (0 = never)", idle)),
      h("div", { class: "row between wrap" },
        h("label", { class: "check" }, autoInstall, "Install missing Python packages when Studio opens"),
        Studio.ui.button("Manage dependencies", { small: true, onclick: () => Studio.dependencies.open() })),
      h("div", { class: "row between wrap" },
        h("label", { class: "check" }, checkUpdates, "Check GitHub for a new release when Studio opens"),
        Studio.ui.button("Check for updates", { small: true, onclick: () => Studio.updates.open() })),
      h("div", { class: "stack tight" },
        h("label", { class: "check" }, osAccess, "Grant agents OS access (shell commands and private-network HTTP)"),
        h("p", { class: "muted tiny", text: "Off by default. When enabled, shell.exec can run any command this user can run, and HTTP tools may reach local addresses. Enable it only on a trusted machine and for tasks that need it." })),
      h("div", {}, Studio.ui.button("Save settings", { kind: "primary", onclick: async event => {
        const body = { model_dirs: dirs.value.split("\n").map(value => value.trim()).filter(Boolean), discover_local: discover.checked,
          gguf_runtime: runtime.value, llama_server_path: binary.value.trim(), llama_gpu_layers: layers.value.trim() || "auto",
          ollama_url: url.value.trim(), idle_unload_minutes: Number(idle.value), auto_install_dependencies: autoInstall.checked,
          check_for_updates: checkUpdates.checked, allow_os_access: osAccess.checked };
        const result = await Studio.run(() => api("/api/settings", { method: "PUT", body }), event.currentTarget);
        if (result) {
          state.settings = result.settings;
          Studio.toast("Settings saved; rediscovering models.", "success");
          await Promise.all([loadRuntimes(), refreshCatalog(true)]);
          if (autoInstall.checked) Studio.dependencies.check();
        }
      } })));
  }

  Studio.views.register("models", {
    title: "Models", subtitle: "One library for local, remote and downloadable language models",
    glyph: "models", hint: "Local, remote and downloadable models",
    init() {
      tabs = Studio.ui.tabs($("models-tabs"), name => {
        if (name !== "runtimes") return;
        if (!Object.keys(state.runtimes).length) {
          Studio.ui.render($("runtimes-list"), h("div", { class: "span-all" }, Studio.ui.loading("Detecting model runtimes…")));
          Studio.ui.render($("settings-form"), Studio.ui.loading("Loading settings…"));
        }
        loadRuntimes().catch(error => Studio.toast(error.message, "error"));
      });
      $("models-search").addEventListener("input", renderLibrary);
      $("models-source").addEventListener("change", renderLibrary);
      $("models-capability").addEventListener("change", renderLibrary);
      $("models-refresh").addEventListener("click", event => Studio.run(async () => {
        const models = await refreshCatalog(true);
        if (models) Studio.toast(`Catalog refreshed: ${models.length} model(s).`, "success");
      }, event.currentTarget));
      $("models-add").addEventListener("click", () => { tabs.select("endpoints"); endpointDialog(); });
      $("discover-search").addEventListener("click", event => Studio.run(search, event.currentTarget));
      $("discover-query").addEventListener("keydown", event => { if (event.key === "Enter") { event.preventDefault(); search(); } });
      Studio.on("job:download", job => { if (job.status === "completed") refreshCatalog(true); });
    },
    async activate(parameters) {
      tabs.select(parameters.tab || "library", false);
      await Promise.all([
        refreshCatalog(false),
        state.featured ? Promise.resolve() : api("/api/models/featured").then(data => { state.featured = data; renderDiscover(); }),
      ]);
      if (parameters.tab === "discover") renderDiscover();
      if (parameters.add === "1") endpointDialog();
    },
  });
})();
