"use strict";

(function () {
  const Studio = window.Studio;
  const { $, h, api, format } = Studio;
  const state = { technique: "fast", values: {}, run: null };
  const MODES = {
    drv: [["fresh", "Fresh model"], ["continue", "Continue checkpoint"]],
    ncm: [["fresh", "Fresh tokenizer + generator"], ["continue", "Continue both stages"], ["update", "Update both stages"], ["fresh_generator", "Fresh generator, existing tokenizer"], ["update_tokenizer", "Update tokenizer only"], ["fresh_tokenizer", "Fresh tokenizer only"]],
  };

  function field(key, label, type = "number", options = {}) {
    const onchange = value => { state.values[key] = value; };
    if (type === "checkbox") return Studio.form.check({ key, label, value: state.values[key], onchange });
    return Studio.form.field({
      key, label, type, value: state.values[key], options: options.values,
      min: options.min, max: options.max,
      step: options.step || (["lr", "generator_lr", "latent_grammar_prob", "w_latent_rl"].includes(key) ? "any" : 1),
      placeholder: options.placeholder, help: options.help, onchange,
    });
  }

  function defaults() {
    state.values = {
      steps: 1000, batch: 16, grad_accum: 1, ctx: 128, dim: 256, heads: 4, lr: 0.003, warmup: 50, eval_every: 200, device: "auto",
      compile: false, n_pre: 2, n_core: 2, n_post: 2, loops: 2, block_size: 32, moe: false, grouped_moe: true, latent_mode: "none",
      thoughts: 8, latent_symbols: 256, latent_window: 32, latent_min_thoughts: 2, latent_block_size: 2, latent_continuous_slots: 2,
      latent_refine_steps: 2, latent_grammar_prob: 0.25, w_latent_rl: 0.1, tokenizer_steps: 1000, generator_lr: 0.0003,
      prefix: 64, levels: "", flex: false, cache_codes: false, tokenizer_batch: 16, tokenizer_grad_accum: 1,
      axiom: false, axiom_profile: "auto",
    };
  }

  function renderTechniques() {
    const techniques = Studio.state.techniques || {};
    const root = Studio.ui.render($("train-techniques"));
    for (const [id, technique] of Object.entries(techniques)) root.append(h("button", {
      type: "button", class: `option-card${state.technique === id ? " selected" : ""}`, "aria-pressed": String(state.technique === id),
      onclick: () => { state.technique = id; $("train-name").value = `mariana-${id}-gui.pt`; renderTechniques(); renderModes(); renderFields(); },
    }, h("div", { class: "option-top" }, h("strong", { text: technique.name }), Studio.ui.badge(technique.badge, "accent")), h("span", { text: technique.description })));
  }

  function renderModes() {
    const select = $("train-mode");
    const previous = state.mode || "fresh";
    const modes = state.technique === "ncm" ? MODES.ncm : MODES.drv;
    Studio.ui.selectOptions(select, modes, previous);
    if (!modes.some(([value]) => value === previous)) select.value = "fresh";
    state.mode = select.value;
  }

  function renderFields() {
    const main = Studio.ui.render($("train-fields"));
    const advanced = Studio.ui.render($("train-advanced"));
    main.append(field("steps", state.technique === "ncm" && ["update_tokenizer", "fresh_tokenizer"].includes(state.mode) ? "Generator steps (unused in tokenizer-only mode)" : "Training steps", "number", { min: 1 }),
      field("batch", "Batch size", "number", { min: 1 }), field("grad_accum", "Gradient accumulation", "number", { min: 1 }),
      field("ctx", "Context bytes", "number", { min: 8 }), field("dim", "Model width", "number", { min: 8 }), field("heads", "Attention heads", "number", { min: 1 }),
      field("lr", state.technique === "ncm" ? "Tokenizer learning rate" : "Learning rate", "number", { min: 0 }));
    advanced.append(field("warmup", "Warmup steps", "number", { min: 0 }), field("eval_every", "Evaluate every N steps", "number", { min: 1 }),
      field("device", "Device", "select", { values: [["auto", "Auto"], ["cuda", "CUDA"], ["cpu", "CPU"], ["xpu", "XPU"], ["mps", "MPS"]] }), field("compile", "Compile model for long runs", "checkbox"));
    if (state.technique === "ncm") {
      advanced.append(field("tokenizer_steps", "Tokenizer steps", "number", { min: 1 }), field("generator_lr", "Generator learning rate", "number", { min: 0 }),
        field("prefix", "Prefix bytes", "number", { min: 1 }), field("levels", "Residual VQ levels (comma separated, blank = default)", "text", { placeholder: "8,4,2" }),
        field("tokenizer_batch", "Tokenizer batch size", "number", { min: 1 }), field("tokenizer_grad_accum", "Tokenizer gradient accumulation", "number", { min: 1 }),
        field("flex", "FlexAttention on supported GPUs", "checkbox"), field("cache_codes", "Cache tokenizer codes", "checkbox"));
    } else {
      advanced.append(field("n_pre", "Pre blocks", "number", { min: 0 }), field("n_core", "Shared core blocks", "number", { min: 0 }), field("n_post", "Post blocks", "number", { min: 0 }),
        field("loops", "Maximum recurrent loops", "number", { min: 1 }), field("block_size", "Diffusion block size", "number", { min: 1 }),
        field("moe", "Sparse mixture-of-experts FFN", "checkbox"), field("grouped_moe", "Grouped MoE kernels when available", "checkbox"),
        field("latent_mode", "Latent thought", "select", { values: [["none", "Off"], ["saint", "SAINT"], ["legacy", "Legacy discrete"]] }),
        field("thoughts", "Thought symbols", "number", { min: 1 }), field("latent_symbols", "Latent vocabulary", "number", { min: 2 }), field("latent_window", "Local raw-byte window", "number", { min: 1 }),
        field("latent_min_thoughts", "Minimum thoughts", "number", { min: 1 }), field("latent_block_size", "Hierarchy block size", "number", { min: 1 }),
        field("latent_continuous_slots", "Continuous slots", "number", { min: 1 }), field("latent_refine_steps", "Slot refinement steps", "number", { min: 0 }),
        field("latent_grammar_prob", "Grammar curriculum probability", "number", { min: 0, max: 1 }), field("w_latent_rl", "Latent RL weight", "number", { min: 0 }),
        field("axiom", "AXIOM corpus codec (connectors, dictionary, DNA/protein packing)", "checkbox"),
        field("axiom_profile", "AXIOM codec profile", "select", { values: [["auto", "Auto (detect domain)"], ["compact", "Compact (English connectors)"], ["reversible", "Reversible (English, keep words)"], ["lossless", "Lossless (exact text)"]] }));
    }
  }

  function config(smart) {
    const values = { ...state.values };
    if (state.technique === "ncm") values.axiom = false;  // AXIOM is a DRV-LM corpus encoding
    return {
      technique: state.technique, mode: $("train-mode").value, name: $("train-name").value.trim(), folder: $("train-folder").value.trim(),
      corpus_mode: $("train-corpus-mode").value, smart: smart === undefined ? $("train-smart").checked : smart, backup: true, ...values,
    };
  }

  async function inspect() {
    const target = Studio.ui.render($("train-corpus-summary"));
    target.textContent = "Inspecting corpus…";
    try {
      const data = await api("/api/corpus/inspect", { body: { path: $("train-folder").value.trim() } });
      target.textContent = `${data.files} readable file(s) · ${format.bytes(data.bytes)} · ${data.skipped} skipped · ${data.formats.map(([suffix, count]) => `${suffix} (${count})`).join(", ")}`;
    } catch (error) { target.textContent = error.message; }
  }

  async function recommend() {
    const root = Studio.ui.render($("train-smart-summary"), Studio.ui.loading("Preparing the corpus and measuring hardware-aware options…"));
    try {
      const data = await api("/api/corpus/smart", { body: config(false) });
      Object.assign(state.values, data.config || {});
      renderFields();
      const corpus = data.corpus || {};
      Studio.ui.render(root, Studio.ui.callout("success", `Prepared ${corpus.files || 0} file(s), ${format.bytes(corpus.output_bytes || 0)} of training text. Smart options applied.`),
        Studio.ui.kv(Object.entries(data.analysis || {}).filter(([, value]) => ["string", "number", "boolean"].includes(typeof value)).slice(0, 12).map(([key, value]) => [format.title(key), String(value)])));
    } catch (error) { Studio.ui.render(root, Studio.ui.callout("danger", error.message)); }
  }

  Studio.views.register("train", {
    title: "Train", subtitle: "Build Mariana DRV-LM and Next-Concept models from your own data",
    glyph: "train", hint: "Train DRV-LM or NCM",
    init() {
      defaults();
      state.run = new Studio.trainingViz.RunPanel("train", {
        empty: "train-empty", content: "train-active", status: "train-status", title: "train-title", stage: "train-stage",
        progress: "train-progress", stats: "train-stats", chart: "train-chart", logs: "train-logs",
        pause: "train-pause", cancel: "train-cancel", chat: "train-chat",
      });
      renderTechniques(); renderModes(); renderFields();
      $("train-mode").addEventListener("change", () => { state.mode = $("train-mode").value; renderFields(); });
      Studio.overlay.bindBrowse($("train-browse"), $("train-folder"), { files: true, title: "Choose training data" });
      $("train-inspect").addEventListener("click", event => Studio.run(inspect, event.currentTarget));
      $("train-recommend").addEventListener("click", event => Studio.run(recommend, event.currentTarget));
      $("train-start").addEventListener("click", async event => {
        if (!$("train-folder").value.trim()) {
          Studio.toast("Choose a training data folder or file first.", "error");
          $("train-folder").focus();
          return;
        }
        const result = await Studio.run(() => api("/api/train", { body: config() }), event.currentTarget);
        if (result) { state.run.update(result.job); Studio.toast("Training started.", "success"); Studio.jobs.refresh(); }
      });
      api("/api/checkpoints").then(data => {
        const list = h("datalist", { id: "train-checkpoint-options" }, data.checkpoints.map(item => h("option", { value: item.name })));
        document.body.append(list); $("train-name").setAttribute("list", "train-checkpoint-options");
      }).catch(() => {});
      Studio.on("state", () => { if (Studio.state.techniques) renderTechniques(); });
    },
    activate() { state.run.refresh(); },
  });
})();
