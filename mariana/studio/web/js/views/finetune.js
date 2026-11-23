"use strict";

(function () {
  const Studio = window.Studio;
  const { $, h, api, format } = Studio;
  const state = {
    bases: [], other: [], info: null, preset: "balanced", values: {}, plan: null, nameTouched: false, run: null,
    tabs: null, discovery: null, discoveryLoaded: false, discoveryPending: null, search: null,
    pendingBase: null, pendingDataset: null,
  };

  function invalidate() {
    state.plan = null;
    $("finetune-plan-output").className = "empty compact";
    $("finetune-plan-output").textContent = "Settings changed. Review the plan to validate them before starting.";
  }

  function field(key, label, type = "number", options = {}) {
    const onchange = value => {
      state.values[key] = value;
      invalidate();
      if (key === "method") {
        if (value !== "lora") state.values.quantization = "none";
        renderFields();
      }
    };
    if (type === "checkbox") return Studio.form.check({ key, label, value: state.values[key], onchange });
    return Studio.form.field({
      key, label, type, value: state.values[key], options: options.values,
      min: options.min, max: options.max,
      step: options.step || (key === "lr" || key.includes("dropout") ? "any" : 1),
      placeholder: options.placeholder, help: options.help, onchange,
    });
  }

  function renderPresets() {
    const root = Studio.ui.render($("finetune-presets"));
    if (!state.info) return;
    for (const [name, preset] of Object.entries(state.info.presets || {})) {
      root.append(h("button", { type: "button", class: `option-card${state.preset === name ? " selected" : ""}`, "aria-pressed": String(state.preset === name), onclick: () => {
        state.preset = name;
        for (const key of ["steps", "lr", "warmup", "grad_accum", "batch", "quantization", "gradient_checkpointing", "merge", "lora_r", "lora_alpha"]) {
          if (preset[key] !== undefined) state.values[key] = preset[key];
        }
        if (state.info.engine === "hf") {
          state.values.method = preset.method;
          state.values.quantization = preset.quantization || (preset.method === "lora" ? "auto" : "none");
        } else state.values.freeze = preset.freeze;
        renderPresets(); renderFields(); invalidate();
      } }, h("strong", { text: preset.label }), h("span", { class: "muted small", text: preset.description })));
    }
  }

  function renderFields() {
    if (!state.info) return;
    const hf = state.info.engine === "hf";
    const main = Studio.ui.render($("finetune-fields"));
    const advanced = Studio.ui.render($("finetune-advanced"));
    if (hf) main.append(field("method", "Training method", "select", { values: Object.entries(state.info.methods || {}) }));
    else main.append(field("freeze", "Trainable scope", "select", { values: [["full", "Full model"], ["last_block", "Last block"], ["last_three", "Last three layers"], ["freeze_embeddings", "All blocks, frozen embeddings"]] }));
    main.append(field("steps", "Steps", "number", { min: 1 }), field("lr", "Learning rate", "number", { min: 0 }), field("batch", "Batch size", "number", { min: 1 }), field("grad_accum", "Gradient accumulation", "number", { min: 1 }));
    advanced.append(field("warmup", "Warmup steps", "number", { min: 0 }), field("eval_every", "Evaluate every N steps", "number", { min: 1 }),
      field("device", "Device", "select", { values: [["auto", "Auto"], ["cuda", "CUDA"], ["cpu", "CPU"], ["xpu", "XPU"], ["mps", "MPS"]] }));
    if (hf) {
      advanced.append(field("block_size", "Context tokens", "number", { min: 16 }), field("precision", "Precision", "select", { values: [["auto", "Auto"], ["bf16", "BFloat 16"], ["fp16", "Float 16 (older GPUs)"], ["fp32", "Float 32"]] }),
        field("train_on", "Loss on", "select", { values: [["assistant", "Assistant turns only"], ["all", "Every token"]] }),
        field("gradient_checkpointing", "Gradient checkpointing (uses less memory)", "checkbox"),
        field("auto_memory", "Auto-tune batch and precision for this GPU", "checkbox"),
        field("max_tokens", "Training token cap", "number", { min: 1000 }),
        field("max_examples", "Training example cap", "number", { min: 1 }));
      if (state.values.method === "lora") advanced.append(field("quantization", "Quantization", "select", { values: [["auto", "Auto (4-bit when needed)"], ["none", "None"], ["4bit", "4-bit QLoRA (low VRAM)"], ["8bit", "8-bit"]] }));
      if (state.values.method === "lora") advanced.append(field("lora_r", "LoRA rank", "number", { min: 1 }), field("lora_alpha", "LoRA alpha", "number", { min: 1 }),
        field("lora_dropout", "LoRA dropout", "number", { min: 0, max: 0.9 }), field("lora_targets", "LoRA target layers", "select", { values: [["auto", "Auto"], ["attention", "Attention"], ["all-linear", "All linear layers"]] }),
        field("merge", "Also save a merged model", "checkbox"));
      if (state.values.method === "last_layers") advanced.append(field("trainable_layers", "Number of trainable final layers", "number", { min: 1 }));
    }
    $("finetune-data-field").classList.toggle("hidden", !hf);
  }

  async function inspectBase(id) {
    state.info = null;
    state.plan = null;
    $("finetune-plan").disabled = true;
    $("finetune-start").disabled = true;
    Studio.ui.render($("finetune-base-info"), Studio.ui.loading("Inspecting base model…"));
    try {
      const info = await api(`/api/finetune/inspect${Studio.query({ id })}`);
      state.info = info;
      state.preset = info.presets.balanced ? "balanced" : Object.keys(info.presets)[0];
      const preset = info.presets[state.preset];
      state.values = {
        method: preset.method || "full", freeze: preset.freeze || "full", steps: preset.steps, lr: preset.lr, warmup: preset.warmup,
        grad_accum: preset.grad_accum, batch: info.engine === "hf" ? (preset.batch || 4) : 8, eval_every: Math.max(10, Math.min(100, preset.steps)), device: "auto",
        block_size: Math.min(512, info.ctx_len || 512), precision: "auto", train_on: "assistant",
        gradient_checkpointing: Boolean(preset.gradient_checkpointing),
        quantization: preset.quantization || "auto", auto_memory: true,
        max_tokens: 20000000, max_examples: 500000,
        lora_r: preset.lora_r || 16, lora_alpha: preset.lora_alpha || 32, lora_dropout: 0.05, lora_targets: "auto", trainable_layers: 2,
        merge: preset.merge === undefined ? true : Boolean(preset.merge),
      };
      if (!state.nameTouched || !$("finetune-name").value) $("finetune-name").value = info.default_name;
      Studio.ui.render($("finetune-base-info"), Studio.ui.kv([
        ["Architecture", info.architecture], ["Parameters", info.parameters ? format.parameters(info.parameters) : null],
        ["Context", info.ctx_len ? `${format.number(info.ctx_len)} ${info.ctx_unit}` : null], ["Layers", info.layers],
        ["Chat template", info.engine === "hf" ? (info.chat_template ? "Yes" : "No; User / Assistant fallback") : null],
      ]), info.requirement ? Studio.ui.callout("warning", info.requirement) : null,
      info.engine === "hf" && info.quantization_ready === false
        ? Studio.ui.callout("info", "Install bitsandbytes to use 4-bit QLoRA on a 6-8 GB GPU: pip install -r requirements-finetune.txt")
        : null);
      $("finetune-engine").textContent = info.engine === "hf" ? "Hugging Face" : "Mariana native";
      $("finetune-plan").disabled = false;
      $("finetune-start").disabled = false;
      renderPresets(); renderFields(); invalidate();
    } catch (error) {
      Studio.ui.render($("finetune-base-info"), Studio.ui.callout("danger", error.message));
    }
  }

  async function loadBases(preferred, force = false, preserve = false) {
    const data = await api(`/api/finetune/bases${force ? "?refresh=1" : ""}`);
    state.bases = data.tunable;
    state.other = data.other;
    const select = $("finetune-base");
    const previous = preferred || select.value;
    if (!data.tunable.length) Studio.ui.selectOptions(select, [["", "No fine-tunable local models found"]]);
    else Studio.ui.selectOptions(select, data.tunable.map(base => [base.id, `${base.name} · ${base.kind_label}`]), previous);
    const unavailable = Studio.ui.render($("finetune-unavailable"), data.other.map(base => h("div", { class: "card soft padded" }, h("strong", { text: base.name }), h("p", { class: "help", text: base.reason || "This runtime does not expose local weights." }))));
    if (!data.transformers_installed) unavailable.prepend(Studio.ui.callout("warning", `Third-party fine-tuning needs ${data.transformers_hint}`));
    if (!data.tunable.length) {
      state.info = null;
      state.plan = null;
      $("finetune-plan").disabled = true;
      $("finetune-start").disabled = true;
      $("finetune-engine").textContent = "";
      Studio.ui.render($("finetune-presets"));
      Studio.ui.render($("finetune-fields"));
      Studio.ui.render($("finetune-advanced"));
      Studio.ui.render($("finetune-base-info"), Studio.ui.empty({ compact: true, icon: "tune", title: "No fine-tunable models",
        text: data.transformers_installed ? "Add a local checkpoint or a Transformers model folder, or open Discovery to download a base." : "Install Transformers support and add a local model to fine-tune third-party weights." }));
      return;
    }
    if (select.value && !(preserve && state.info && state.info.id === select.value)) await inspectBase(select.value);
  }

  function config() {
    return {
      base: $("finetune-base").value, folder: $("finetune-folder").value.trim(), corpus_mode: $("finetune-corpus-mode").value,
      data_mode: $("finetune-data-mode").value, name: $("finetune-name").value.trim(), preset: state.preset,
      backup: $("finetune-backup").checked, ...state.values,
    };
  }

  async function review() {
    if (!state.info) { Studio.toast("Select a base model first.", "error"); return null; }
    const output = Studio.ui.render($("finetune-plan-output"), Studio.ui.loading("Validating data and calculating trainable parameters…"));
    output.className = "stack";
    try {
      const plan = await api("/api/finetune/plan", { body: config() });
      state.plan = plan;
      Studio.ui.render(output, Studio.ui.kv(plan.summary),
        plan.estimate && plan.estimate.parameters ? Studio.ui.meter({ label: "Trainable parameters", value: plan.estimate.trainable, total: plan.estimate.parameters,
          text: `${format.parameters(plan.estimate.trainable)} · ${format.percent(plan.estimate.trainable_percent, 2)}` }) : null,
        ...(plan.warnings || []).map(warning => h("div", { class: "plan-warning", text: warning })),
        plan.ready === false ? Studio.ui.callout("danger", plan.base.requirement || "Install the missing training backend before starting.") : Studio.ui.callout("success", "Plan validated. The base model will be preserved."));
      return plan;
    } catch (error) {
      output.className = "stack";
      Studio.ui.render(output, Studio.ui.callout("danger", error.message));
      return null;
    }
  }

  /* -- discovery centre ---------------------------------------------------- */
  function platformLabel(item) {
    if (item.platform === "github") return "GitHub";
    if (item.platform === "openml") return "OpenML dataset";
    if (item.platform === "zenodo") return "Zenodo record";
    return item.kind === "dataset" ? "Hugging Face dataset" : "Hugging Face model";
  }

  function itemStats(item) {
    const parts = [];
    if (item.downloads) parts.push(`${format.compact(item.downloads)} downloads`);
    if (item.likes) parts.push(`${format.compact(item.likes)} likes`);
    if (item.stars) parts.push(`${format.compact(item.stars)} stars`);
    if (item.updated) parts.push(`updated ${format.ago(item.updated)}`);
    return parts.join(" · ");
  }

  function resultCard(item) {
    const glyph = item.kind === "dataset" ? "table" : item.kind === "model" ? "models" : "globe";
    const actions = [];
    if (item.kind === "dataset" && item.platform === "huggingface") {
      actions.push(Studio.ui.button(item.local_folder ? "Use saved copy" : "Choose split", {
        kind: "primary", small: true,
        onclick: () => item.local_folder ? applyDataset(item) : chooseDataset(item),
      }));
    } else if (item.kind === "model") {
      if (item.local && item.finetune) actions.push(Studio.ui.button("Select base", { kind: "primary", small: true, onclick: () => selectBase(item.local_id) }));
      else if (!item.local) actions.push(Studio.ui.button("Download & use", { kind: "primary", small: true, onclick: event => downloadBase(item, event.currentTarget) }));
    }
    if (item.url) actions.push(h("a", { class: "button small ghost", href: item.url, target: "_blank", rel: "noreferrer noopener", title: item.url, "aria-label": `Open ${item.repo} in a new tab` }, "Open", Studio.icon("external", "small")));
    const tags = [Studio.ui.badge(platformLabel(item))];
    if (item.local || item.local_folder) tags.push(Studio.ui.badge(item.kind === "dataset" ? "Saved" : "Downloaded", "success"));
    if (item.gated) tags.push(Studio.ui.badge("Gated", "warning"));
    if (item.local && item.kind === "model" && !item.finetune) tags.push(Studio.ui.badge("Not tunable", "warning"));
    tags.push(...(item.tags || []).slice(0, 4));
    return Studio.components.resultCard({
      glyph,
      title: item.title || item.repo,
      subtitle: item.title && item.title !== item.repo ? item.repo : "",
      description: item.summary || itemStats(item) || "Open the page for details and licence information.",
      meta: item.summary ? itemStats(item) : "",
      tags,
      detail: item.local && item.kind === "model" && !item.finetune && item.reason ? h("span", { class: "help", text: item.reason }) : null,
      actions,
    });
  }

  function renderInstalled() {
    const container = Studio.ui.render($("finetune-discover-installed"));
    const installed = (state.discovery && state.discovery.installed) || [];
    if (!installed.length) {
      container.append(h("div", { class: "span-all" }, Studio.ui.empty({ compact: true, icon: "folder", title: "No datasets downloaded yet",
        text: "Choose a dataset above. Studio saves its rows as JSON Lines under datasets/ and fills in the data folder." })));
      return;
    }
    for (const item of installed) {
      const rows = item.rows ? `${format.number(item.rows)} rows` : "Saved rows";
      container.append(Studio.components.resultCard({
        glyph: "table",
        title: item.repo,
        subtitle: `${item.config || "default"} / ${item.split || "train"}`,
        description: `${rows} · ${format.bytes(item.bytes || 0)} · saved ${format.ago(item.created)}`,
        actions: [
          Studio.ui.button("Use for fine-tuning", { kind: "primary", small: true, onclick: () => applyDataset(item) }),
          Studio.ui.button("Copy path", { kind: "ghost", small: true, onclick: () => Studio.copy(item.folder, "Dataset folder copied") }),
        ],
      }));
    }
  }

  function renderDiscovery() {
    if (!state.discovery) return;
    const featured = state.discovery.featured || {};
    Studio.ui.render($("finetune-discover-quick"), (featured.queries || []).map(suggestion =>
      h("button", { type: "button", class: "button small ghost", onclick: () => {
        $("finetune-discover-query").value = suggestion.query;
        $("finetune-discover-platform").value = suggestion.platform || "datasets";
        searchDiscovery();
      } }, suggestion.label)));
    const list = Studio.ui.render($("finetune-discover-results"));
    const results = state.search ? state.search.results : [...(featured.datasets || []), ...(featured.models || [])];
    $("finetune-discover-heading").textContent = state.search ? `Results for “${state.search.query}”` : "Featured datasets and bases";
    const warnings = state.search ? state.search.warnings : (state.discovery.warnings || []);
    if (warnings && warnings.length) list.append(h("div", { class: "span-all stack tight" }, warnings.map(warning => Studio.ui.callout("warning", warning))));
    if (!results.length) {
      list.append(h("div", { class: "span-all" }, Studio.ui.empty({ compact: true, icon: "search", title: "No matches",
        text: "Try a repository name such as alpaca, Qwen or TinyStories, or pick another source." })));
    } else {
      list.append(...results.map(resultCard));
    }
    renderInstalled();
    Studio.ui.render($("finetune-discover-sources"), (featured.external || []).map(source =>
      h("a", { class: "option-card", href: source.url, target: "_blank", rel: "noreferrer noopener" },
        h("strong", { text: source.name }), h("span", { text: source.summary }))));
  }

  function loadDiscovery(force = false) {
    if (state.discoveryLoaded && !force) { renderDiscovery(); return Promise.resolve(state.discovery); }
    if (state.discoveryPending && !force) return state.discoveryPending;
    if (!state.search) Studio.ui.render($("finetune-discover-results"), h("div", { class: "span-all" }, Studio.ui.loading("Loading discovery…")));
    const pending = api("/api/finetune/discovery").then(data => {
      state.discovery = data;
      state.discoveryLoaded = true;
      renderDiscovery();
      return data;
    }).catch(error => {
      Studio.ui.render($("finetune-discover-results"), h("div", { class: "span-all" }, Studio.ui.callout("danger", error.message)));
      throw error;
    }).finally(() => {
      if (state.discoveryPending === pending) state.discoveryPending = null;
    });
    state.discoveryPending = pending;
    return pending;
  }

  async function searchDiscovery() {
    const query = $("finetune-discover-query").value.trim();
    const platform = $("finetune-discover-platform").value;
    if (!query) {
      state.search = null;
      try { await loadDiscovery(true); } catch (error) { /* the callout already explains the failure */ }
      return;
    }
    Studio.ui.render($("finetune-discover-results"), h("div", { class: "span-all" }, Studio.ui.loading("Searching public hubs…")));
    try {
      const data = await api(`/api/finetune/discovery${Studio.query({ q: query, platform, limit: 30 })}`);
      state.discovery = data;
      state.discoveryLoaded = true;
      state.search = { query, platform, results: data.results || [], warnings: data.warnings || [] };
      renderDiscovery();
    } catch (error) {
      Studio.ui.render($("finetune-discover-results"), h("div", { class: "span-all" }, Studio.ui.callout("danger", error.message)));
    }
  }

  function refreshDiscovery() {
    return loadDiscovery(true).catch(() => null);
  }

  function applyDataset(item) {
    const folder = item.folder || item.local_folder;
    if (!folder) { Studio.toast("This dataset has no saved folder; download it again.", "error"); return; }
    $("finetune-folder").value = folder;
    $("finetune-data-mode").value = "auto";
    invalidate();
    state.tabs.select("setup");
    Studio.toast(`${item.repo} is now the fine-tuning data folder.`, "success");
  }

  async function chooseDataset(item) {
    const view = Studio.overlay.modal({
      key: `dataset:${item.repo}`,
      title: `Use ${item.repo}`, subtitle: "Choose a configuration and split; Studio saves the rows as JSON Lines.",
      wide: true, content: Studio.ui.loading("Reading dataset splits…"),
    });
    if (view.reused) return;
    try {
      const info = await api(`/api/finetune/datasets/info${Studio.query({ repo: item.repo })}`);
      const choices = info.splits || [];
      if (!choices.length) throw new Error("This dataset has no viewer splits. Download its files and point the data folder at them.");
      const split = h("select", { "aria-label": "Dataset configuration and split" },
        choices.map((choice, index) => h("option", { value: String(index), text: `${choice.config} / ${choice.split}` })));
      const rows = h("input", { type: "number", min: 100, max: 100000, step: 100, value: 5000, "aria-label": "Maximum rows" });
      Studio.ui.render(view.body,
        Studio.ui.kv([
          ["Dataset", info.repo], ["Licence", info.license],
          ["Downloads", info.downloads ? format.compact(info.downloads) : null],
          ["Saved copy", info.local ? info.local.folder : null],
        ]),
        info.gated ? Studio.ui.callout("warning", "This dataset is gated. Accept its licence on Hugging Face and configure HF_TOKEN before downloading.") : null,
        Studio.ui.field("Configuration / split", split, "Training rows come from one split."),
        Studio.ui.field("Maximum rows", rows, "A few thousand rows are enough to adapt a small model; more rows take longer."),
        Studio.ui.callout("info", "The download runs in the job centre. When it finishes, Studio fills in the data folder and switches to Setup."));
      view.footer.append(Studio.ui.button("Download dataset", { kind: "primary", onclick: async event => {
        const choice = choices[Number(split.value)];
        const result = await Studio.run(() => api("/api/finetune/datasets/download", {
          body: { repo: item.repo, config: choice.config, split: choice.split, max_rows: Number(rows.value) },
        }), event.currentTarget);
        if (!result) return;
        state.pendingDataset = { repo: item.repo, folder: info.local ? info.local.folder : "" };
        view.close();
        Studio.ui.render($("finetune-discover-status"), Studio.ui.callout("info", `Downloading ${item.repo}. The data folder is filled in when it finishes.`));
        Studio.toast("Dataset download started; follow it in the job centre.", "success");
        Studio.jobs.refresh();
      } }));
      if (info.local) view.footer.prepend(Studio.ui.button("Use saved copy", { onclick: () => {
        view.close();
        applyDataset({ repo: item.repo, folder: info.local.folder });
      } }));
    } catch (error) {
      Studio.ui.render(view.body, Studio.ui.callout("danger", error.message));
    }
  }

  async function downloadBase(item, control) {
    const result = await Studio.run(() => api("/api/models/download", {
      body: { source: "huggingface", repo: item.repo, format: "transformers" },
    }), control);
    if (!result) return;
    state.pendingBase = { repo: item.repo, id: "models/" + item.repo.split("/").join("--") };
    Studio.ui.render($("finetune-discover-status"), Studio.ui.callout("info", `Downloading ${item.repo}. It becomes the selected base when the download finishes.`));
    Studio.toast("Base download started; follow it in the job centre.", "success");
    Studio.jobs.refresh();
  }

  function selectBase(id) {
    state.tabs.select("setup");
    loadBases(id, true).then(() => {
      if ($("finetune-base").value === id) Studio.toast("Base model selected. Choose data, then review the plan.", "success");
      else Studio.toast("The model downloaded but Studio could not select it; refresh the base list.", "error");
    }).catch(error => Studio.toast(error.message, "error"));
  }

  function onDownloadJob(job) {
    const pending = state.pendingBase;
    if (!pending || !job.meta || job.meta.repo !== pending.repo || job.meta.format !== "transformers") return;
    if (job.status === "failed" || job.status === "cancelled") {
      state.pendingBase = null;
      Studio.ui.render($("finetune-discover-status"), Studio.ui.callout("danger", job.error || "The base download did not finish. Try again or choose another repository."));
      return;
    }
    if (job.status !== "completed") return;
    state.pendingBase = null;
    Studio.ui.render($("finetune-discover-status"), Studio.ui.callout("success", `${pending.repo} is downloaded. Loading it as the base model…`));
    loadBases(pending.id, true).then(() => {
      state.tabs.select("setup");
      refreshDiscovery();
      Studio.toast("Base model ready: choose data and review the plan.", "success");
    }).catch(error => Studio.toast(error.message, "error"));
  }

  function onDatasetJob(job) {
    const pending = state.pendingDataset;
    if (!pending || !job.meta || job.meta.repo !== pending.repo) return;
    if (job.status === "failed" || job.status === "cancelled") {
      state.pendingDataset = null;
      Studio.ui.render($("finetune-discover-status"), Studio.ui.callout("danger", job.error || "The dataset download did not finish. Try again or choose another split."));
      return;
    }
    if (job.status !== "completed") return;
    state.pendingDataset = null;
    const folder = (job.result && job.result.folder) || pending.folder;
    if (!folder) {
      Studio.toast("The dataset downloaded but its folder is unknown; check the job logs.", "error");
      return;
    }
    $("finetune-folder").value = folder;
    $("finetune-data-mode").value = "auto";
    invalidate();
    const count = job.result && job.result.rows;
    Studio.ui.render($("finetune-discover-status"), Studio.ui.callout("success", `${count ? `${format.number(count)} rows` : "Dataset"} saved to ${folder}.`));
    state.tabs.select("setup");
    refreshDiscovery();
    Studio.toast("Dataset ready; the data folder is filled in.", "success");
  }

  function openDiscovery(platform, query) {
    state.tabs.select("discovery");
    $("finetune-discover-platform").value = platform || "datasets";
    if (query !== undefined) {
      $("finetune-discover-query").value = query;
      if (!query.trim()) state.search = null;
    }
    loadDiscovery().then(() => ($("finetune-discover-query").value.trim() ? searchDiscovery() : null)).catch(() => null);
  }

  Studio.views.register("finetune", {
    title: "Fine-tune", subtitle: "Adapt Mariana checkpoints and third-party causal language models",
    glyph: "tune", hint: "Adapt any local causal language model and discover datasets",
    init() {
      state.run = new Studio.trainingViz.RunPanel("finetune", {
        empty: "finetune-run-empty", content: "finetune-run-content", status: "finetune-status", title: "finetune-run-title", stage: "finetune-run-stage",
        progress: "finetune-progress", stats: "finetune-stats", chart: "finetune-chart", logs: "finetune-logs",
        pause: "finetune-pause", cancel: "finetune-cancel", chat: "finetune-chat",
      });
      state.tabs = Studio.ui.tabs($("finetune-tabs"), name => {
        state.lastTab = name;
        if (name === "discovery") loadDiscovery().catch(error => Studio.toast(error.message, "error"));
      });
      $("finetune-plan").disabled = true;
      $("finetune-start").disabled = true;
      $("finetune-base").addEventListener("change", () => inspectBase($("finetune-base").value));
      $("finetune-name").addEventListener("input", () => { state.nameTouched = true; invalidate(); });
      for (const id of ["finetune-folder", "finetune-corpus-mode", "finetune-data-mode", "finetune-backup"]) $(id).addEventListener("change", invalidate);
      Studio.overlay.bindBrowse($("finetune-browse"), $("finetune-folder"), { files: true, title: "Choose fine-tuning data" });
      $("finetune-plan").addEventListener("click", event => Studio.run(review, event.currentTarget));
      $("finetune-start").addEventListener("click", async event => {
        const plan = await review();
        if (!plan || plan.ready === false) return;
        const result = await Studio.run(() => api("/api/finetune/start", { body: config() }), event.currentTarget);
        if (result) { state.run.update(result.job); Studio.toast("Fine-tune started. The base model is preserved.", "success"); Studio.jobs.refresh(); }
      });
      $("finetune-find-base").addEventListener("click", () => openDiscovery("models", "instruct"));
      $("finetune-find-data").addEventListener("click", () => openDiscovery("datasets", ""));
      $("finetune-discover-search").addEventListener("click", event => Studio.run(searchDiscovery, event.currentTarget));
      $("finetune-discover-query").addEventListener("keydown", event => {
        if (event.key === "Enter") { event.preventDefault(); searchDiscovery(); }
      });
      $("finetune-discover-platform").addEventListener("change", () => {
        if ($("finetune-discover-query").value.trim()) searchDiscovery();
      });
      Studio.on("job:download", onDownloadJob);
      Studio.on("job:dataset", onDatasetJob);
    },
    async activate(parameters) {
      if (parameters.base) state.nameTouched = false;
      const tab = parameters.tab === "discovery" ? "discovery"
        : parameters.tab === "setup" ? "setup"
          : parameters.base ? "setup" : state.lastTab || "setup";
      state.lastTab = tab;
      if (state.tabs) state.tabs.select(tab, false);
      if (!state.info) Studio.ui.render($("finetune-base-info"), Studio.ui.loading("Loading base models…"));
      try {
        await loadBases(parameters.base, false, !parameters.base);
      } catch (error) {
        $("finetune-plan").disabled = true;
        $("finetune-start").disabled = true;
        Studio.ui.render($("finetune-base"), h("option", { value: "", text: "Could not load base models" }));
        Studio.ui.render($("finetune-base-info"), Studio.ui.callout("danger", error.message));
      }
      if (tab === "discovery") {
        if (parameters.q) {
          $("finetune-discover-query").value = parameters.q;
          $("finetune-discover-platform").value = parameters.platform || "datasets";
        }
        try {
          await loadDiscovery();
          if (parameters.q) await searchDiscovery();
        } catch (error) {
          Studio.toast(error.message, "error");
        }
      }
    },
  });
})();
