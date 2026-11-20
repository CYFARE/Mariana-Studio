"use strict";

/* Studio.trainingViz: the live loss chart and helpers for reading trainer logs.
   One chart shows one stage on one axis; training and validation share the
   loss scale, so they are drawn together and told apart by colour + legend. */
(function () {
  const Studio = window.Studio;
  const { h, format } = Studio;
  const STAGE_LABELS = { model: "Model", tokenizer: "Tokenizer", generator: "Generator" };
  const PADDING = { left: 52, right: 16, top: 12, bottom: 26 };
  const MAX_TABLE_ROWS = 160;

  /* Fallback for jobs that only expose log lines: extract the loss curve from them. */
  function parseTrainingMetrics(logs) {
    const points = [];
    const seen = new Set();
    let stage = "model";
    let throughput = null;
    let throughputUnit = "";
    (logs || []).forEach(rawLine => {
      const line = String(rawLine).trim();
      if (line.includes("Tokenizer parameters:")) stage = "tokenizer";
      if (line.includes("Generator trainable parameters:")) stage = "generator";
      const match = line.match(/^step\s+(\d+)\/(\d+)\s*\|(.+)$/i);
      if (!match) return;
      const step = Number(match[1]);
      let loss = null;
      match[3].split("|").forEach(part => {
        const metric = part.trim().match(/^(\S+)\s+(\S+)$/);
        if (!metric) return;
        const key = metric[1].toLowerCase();
        const value = Number(metric[2]);
        if (!Number.isFinite(value)) return;
        if (key === "total") loss = value;
        if (key === "tok/s" || key === "pos/s") {
          throughput = value;
          throughputUnit = key;
        }
      });
      const key = `${stage}:${step}`;
      if (loss !== null && !seen.has(key)) {
        seen.add(key);
        points.push({ step, loss, stage });
      }
    });
    const losses = points.map(point => point.loss);
    return {
      points,
      latestLoss: losses.length ? losses[losses.length - 1] : null,
      bestLoss: losses.length ? Math.min(...losses) : null,
      throughput,
      throughputUnit,
    };
  }

  function formatLoss(value) {
    if (!Number.isFinite(value)) return "—";
    const magnitude = Math.abs(value);
    if (magnitude >= 1000 || (magnitude > 0 && magnitude < 0.001)) return value.toExponential(2);
    return value.toFixed(3);
  }

  function formatThroughput(value, unit) {
    if (!Number.isFinite(value)) return "—";
    const scaled = value >= 1e6 ? `${(value / 1e6).toFixed(1)}m` : value >= 1e3 ? `${(value / 1e3).toFixed(1)}k` : value.toFixed(0);
    return `${scaled} ${unit || "units/s"}`;
  }

  /* Round axis values: about `count` ticks covering [low, high]. */
  function niceTicks(low, high, count = 4) {
    if (!Number.isFinite(low) || !Number.isFinite(high)) return [0, 1];
    if (!(high > low)) {
      const pad = Math.abs(low) * 0.1 || 1;
      low -= pad;
      high += pad;
    }
    const raw = (high - low) / count;
    const power = Math.pow(10, Math.floor(Math.log10(raw)));
    const unit = [1, 2, 2.5, 5, 10].find(step => step * power >= raw) * power;
    const ticks = [];
    for (let value = Math.floor(low / unit) * unit; value <= high + unit * 0.5; value += unit) {
      ticks.push(Number(value.toPrecision(12)));
    }
    return ticks;
  }

  function nearestIndex(series, step) {
    let low = 0;
    let high = series.length - 1;
    while (high - low > 1) {
      const middle = (low + high) >> 1;
      if (series[middle][0] < step) low = middle;
      else high = middle;
    }
    return Math.abs(series[low][0] - step) <= Math.abs(series[high][0] - step) ? low : high;
  }

  function token(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  class LossChart {
    constructor(container) {
      this.metrics = { points: [], validation: [] };
      this.stage = null;
      this.pinned = false;
      this.hover = null;
      this.geometry = null;

      this.title = h("h3", { text: "Training loss" });
      this.stageGroup = h("div", { class: "segmented hidden", role: "group", "aria-label": "Training stage" });
      this.legend = h("div", { class: "chart-legend hidden" },
        h("span", {}, h("i", { class: "line-key" }), "Training"),
        h("span", {}, h("i", { class: "line-key second" }), "Validation"));
      this.tableButton = h("button", {
        class: "button small ghost", type: "button", "aria-pressed": "false",
        onclick: () => this.toggleTable(),
      }, Studio.icon("table", "small"), "Table");
      this.canvas = h("canvas", { class: "chart-canvas", tabIndex: 0, role: "img", "aria-label": "Loss chart" });
      this.placeholder = h("p", { class: "muted small", text: "The loss curve appears after the first training step." });
      this.tableBox = h("div", { class: "chart-table hidden", tabIndex: 0, "aria-label": "Loss values" });
      this.tooltip = h("div", { class: "tooltip hidden", role: "status" });
      container.classList.add("chart");
      container.append(
        h("div", { class: "chart-head" },
          h("div", { class: "row wrap" }, this.title, this.stageGroup),
          h("div", { class: "row wrap" }, this.legend, this.tableButton)),
        this.placeholder, this.canvas, this.tableBox);
      document.body.append(this.tooltip);

      this.stageGroup.addEventListener("click", event => {
        const button = event.target.closest("button");
        if (!button) return;
        this.stage = button.dataset.value;
        this.pinned = true;
        this.hover = null;
        this.refresh();
      });
      this.canvas.addEventListener("pointermove", event => this.onPointer(event));
      this.canvas.addEventListener("pointerleave", () => this.setHover(null));
      this.canvas.addEventListener("blur", () => this.setHover(null));
      this.canvas.addEventListener("keydown", event => this.onKey(event));
      if (window.ResizeObserver) new ResizeObserver(() => this.draw()).observe(this.canvas);
      Studio.on("theme", () => this.draw());
      this.refresh();
    }

    /* metrics: { points: [[stage, step, loss]], validation: [[stage, step, loss]] } */
    update(metrics) {
      const next = metrics && Array.isArray(metrics.points) ? metrics : { points: [], validation: [] };
      const restarted = next.points.length < this.metrics.points.length;
      this.metrics = { points: next.points, validation: next.validation || [] };
      if (restarted) {
        this.pinned = false;
        this.hover = null;
      }
      this.refresh();
    }

    stages() {
      const names = [];
      for (const point of this.metrics.points) if (!names.includes(point[0])) names.push(point[0]);
      return names;
    }

    series() {
      const pick = rows => rows.filter(row => row[0] === this.stage).map(row => [row[1], row[2]]);
      return { training: pick(this.metrics.points), validation: pick(this.metrics.validation) };
    }

    refresh() {
      const stages = this.stages();
      if (!this.pinned || !stages.includes(this.stage)) this.stage = stages.length ? stages[stages.length - 1] : null;
      Studio.clear(this.stageGroup);
      for (const name of stages) {
        this.stageGroup.append(h("button", {
          type: "button", "data-value": name, "aria-pressed": String(name === this.stage),
          text: STAGE_LABELS[name] || format.title(name),
        }));
      }
      this.stageGroup.classList.toggle("hidden", stages.length < 2);
      const { training, validation } = this.series();
      const hasData = training.length > 0;
      this.placeholder.classList.toggle("hidden", hasData);
      this.canvas.classList.toggle("hidden", !hasData || this.tableShown());
      this.legend.classList.toggle("hidden", validation.length === 0);
      this.tableButton.classList.toggle("hidden", !hasData);
      this.title.textContent = validation.length ? "Loss" : "Training loss";
      if (hasData) {
        const last = training[training.length - 1];
        const best = Math.min(...training.map(point => point[1]));
        this.canvas.setAttribute("aria-label",
          `Loss chart. Latest training loss ${formatLoss(last[1])} at step ${last[0]}; best ${formatLoss(best)}.`
          + (validation.length ? ` Latest validation loss ${formatLoss(validation[validation.length - 1][1])}.` : ""));
      }
      if (this.tableShown()) this.renderTable();
      this.draw();
    }

    tableShown() {
      return this.tableButton.getAttribute("aria-pressed") === "true";
    }

    toggleTable() {
      const show = !this.tableShown();
      this.tableButton.setAttribute("aria-pressed", String(show));
      this.tableBox.classList.toggle("hidden", !show);
      this.setHover(null);
      this.refresh();
    }

    renderTable() {
      const { training, validation } = this.series();
      const stride = Math.max(1, Math.ceil(training.length / MAX_TABLE_ROWS));
      const rows = new Map();
      training.forEach((point, index) => {
        if (index % stride === 0 || index === training.length - 1) rows.set(point[0], { training: point[1] });
      });
      for (const point of validation) rows.set(point[0], { ...(rows.get(point[0]) || {}), validation: point[1] });
      const body = h("tbody");
      for (const step of Array.from(rows.keys()).sort((left, right) => left - right)) {
        const row = rows.get(step);
        body.append(h("tr", {},
          h("td", { class: "numeric", text: format.number(step) }),
          h("td", { class: "numeric", text: row.training === undefined ? "" : formatLoss(row.training) }),
          h("td", { class: "numeric", text: row.validation === undefined ? "" : formatLoss(row.validation) })));
      }
      Studio.clear(this.tableBox).append(h("table", { class: "table" },
        h("thead", {}, h("tr", {},
          h("th", { class: "numeric", scope: "col", text: "Step" }),
          h("th", { class: "numeric", scope: "col", text: "Training" }),
          h("th", { class: "numeric", scope: "col", text: "Validation" }))),
        body));
    }

    draw() {
      const canvas = this.canvas;
      const width = canvas.clientWidth;
      const height = canvas.clientHeight;
      const { training, validation } = this.series();
      this.geometry = null;
      if (!width || !height || !training.length) return;
      const ratio = window.devicePixelRatio || 1;
      if (canvas.width !== Math.round(width * ratio) || canvas.height !== Math.round(height * ratio)) {
        canvas.width = Math.round(width * ratio);
        canvas.height = Math.round(height * ratio);
      }
      const context = canvas.getContext("2d");
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
      context.clearRect(0, 0, width, height);

      const colors = {
        training: token("--series-1"), validation: token("--series-2"), grid: token("--chart-grid"),
        text: token("--subtle"), surface: token("--panel"), cursor: token("--border-bright"),
      };
      const all = training.concat(validation);
      const steps = all.map(point => point[0]);
      let minStep = Math.min(...steps);
      let maxStep = Math.max(...steps);
      if (minStep === maxStep) {
        minStep -= 1;
        maxStep += 1;
      }
      const values = all.map(point => point[1]);
      const ticks = niceTicks(Math.min(...values), Math.max(...values));
      const low = ticks[0];
      const high = ticks[ticks.length - 1];
      const plotWidth = width - PADDING.left - PADDING.right;
      const plotHeight = height - PADDING.top - PADDING.bottom;
      const x = step => PADDING.left + ((step - minStep) / (maxStep - minStep)) * plotWidth;
      const y = value => PADDING.top + (1 - (value - low) / (high - low)) * plotHeight;
      this.geometry = { x, y, minStep, maxStep, plotWidth, training, validation };

      context.font = `12px ${token("--font-sans") || "sans-serif"}`;
      context.lineWidth = 1;
      context.strokeStyle = colors.grid;
      context.fillStyle = colors.text;
      context.textAlign = "right";
      context.textBaseline = "middle";
      for (const tick of ticks) {
        const position = Math.round(y(tick)) + 0.5;
        context.beginPath();
        context.moveTo(PADDING.left, position);
        context.lineTo(width - PADDING.right, position);
        context.stroke();
        context.fillText(formatLoss(tick), PADDING.left - 8, position);
      }
      context.textBaseline = "top";
      const stepTicks = niceTicks(minStep, maxStep, Math.max(2, Math.min(6, Math.floor(plotWidth / 90))))
        .filter(tick => tick >= minStep && tick <= maxStep && Number.isInteger(tick));
      stepTicks.forEach((tick, index) => {
        context.textAlign = index === 0 && x(tick) - PADDING.left < 16 ? "left" : "center";
        context.fillText(format.compact(tick), x(tick), height - PADDING.bottom + 8);
      });

      const line = (points, color) => {
        context.strokeStyle = color;
        context.lineWidth = 2;
        context.lineJoin = "round";
        context.lineCap = "round";
        context.beginPath();
        points.forEach((point, index) => {
          if (index === 0) context.moveTo(x(point[0]), y(point[1]));
          else context.lineTo(x(point[0]), y(point[1]));
        });
        context.stroke();
      };
      const marker = (point, color, radius = 4) => {
        context.beginPath();
        context.arc(x(point[0]), y(point[1]), radius + 2, 0, Math.PI * 2);
        context.fillStyle = colors.surface;
        context.fill();
        context.beginPath();
        context.arc(x(point[0]), y(point[1]), radius, 0, Math.PI * 2);
        context.fillStyle = color;
        context.fill();
      };

      line(training, colors.training);
      if (validation.length) line(validation, colors.validation);
      if (this.hover !== null && training[this.hover]) {
        const point = training[this.hover];
        const position = Math.round(x(point[0])) + 0.5;
        context.strokeStyle = colors.cursor;
        context.lineWidth = 1;
        context.beginPath();
        context.moveTo(position, PADDING.top);
        context.lineTo(position, height - PADDING.bottom);
        context.stroke();
        const partner = this.validationNear(point[0]);
        if (partner) marker(partner, colors.validation);
        marker(point, colors.training);
      } else {
        if (validation.length) marker(validation[validation.length - 1], colors.validation);
        marker(training[training.length - 1], colors.training);
      }
    }

    /* The validation measurement that belongs to a step: the latest one at or before it. */
    validationNear(step) {
      const validation = this.geometry ? this.geometry.validation : [];
      let found = null;
      for (const point of validation) {
        if (point[0] <= step) found = point;
        else break;
      }
      return found || validation[0] || null;
    }

    onPointer(event) {
      if (!this.geometry) return;
      const box = this.canvas.getBoundingClientRect();
      const { minStep, maxStep, plotWidth, training } = this.geometry;
      const fraction = (event.clientX - box.left - PADDING.left) / plotWidth;
      const step = minStep + Math.max(0, Math.min(1, fraction)) * (maxStep - minStep);
      this.setHover(nearestIndex(training, step));
    }

    onKey(event) {
      if (!this.geometry) return;
      const count = this.geometry.training.length;
      if (event.key === "Escape") {
        this.setHover(null);
        return;
      }
      if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
      event.preventDefault();
      const current = this.hover === null ? count - 1 : this.hover;
      const jump = event.shiftKey ? Math.max(1, Math.round(count / 10)) : 1;
      const next = event.key === "Home" ? 0 : event.key === "End" ? count - 1
        : current + (event.key === "ArrowRight" ? jump : -jump);
      this.setHover(Math.max(0, Math.min(count - 1, next)));
    }

    setHover(index) {
      if (index === this.hover) return;
      this.hover = index;
      this.draw();
      if (index === null || !this.geometry || !this.geometry.training[index]) {
        this.tooltip.classList.add("hidden");
        return;
      }
      const point = this.geometry.training[index];
      const partner = this.validationNear(point[0]);
      Studio.clear(this.tooltip).append(
        h("div", { class: "muted", text: `Step ${format.number(point[0])}` }),
        h("div", { class: "tooltip-row" }, h("i", { class: "line-key" }), h("span", { text: "Training" }),
          h("strong", { text: formatLoss(point[1]) })));
      if (partner) {
        this.tooltip.append(h("div", { class: "tooltip-row" }, h("i", { class: "line-key second" }),
          h("span", { text: partner[0] === point[0] ? "Validation" : `Validation @ ${format.number(partner[0])}` }),
          h("strong", { text: formatLoss(partner[1]) })));
      }
      this.tooltip.classList.remove("hidden");
      const box = this.canvas.getBoundingClientRect();
      const left = box.left + this.geometry.x(point[0]);
      const top = box.top + this.geometry.y(point[1]);
      const width = this.tooltip.offsetWidth;
      const flip = left + 14 + width > window.innerWidth - 8;
      this.tooltip.style.left = `${Math.round(flip ? left - width - 14 : left + 14)}px`;
      this.tooltip.style.top = `${Math.round(Math.max(8, top - this.tooltip.offsetHeight - 10))}px`;
    }
  }

  /* Headline numbers for the stat tiles next to the chart. */
  function summarize(metrics) {
    const latest = (metrics && metrics.latest) || {};
    const values = latest.metrics || {};
    return {
      loss: formatLoss(values.total),
      best: formatLoss(metrics ? metrics.best : null),
      throughput: formatThroughput(latest.throughput, latest.throughput_unit),
      extra: Object.entries(values).filter(([key]) => key !== "total" && key !== "lr")
        .map(([key, value]) => [key, formatLoss(value)]),
      learningRate: Number.isFinite(values.lr) ? values.lr.toExponential(2) : "—",
    };
  }

  const runPanels = new Set();
  let runFeed = null;

  class RunPanel {
    constructor(prefix, ids) {
      this.prefix = prefix;
      this.ids = ids;
      this.job = null;
      this.chart = new LossChart(document.getElementById(ids.chart));
      this.progress = Studio.ui.progress("Training progress");
      document.getElementById(ids.progress).append(this.progress);
      document.getElementById(ids.pause).addEventListener("click", () => this.control("pause"));
      document.getElementById(ids.cancel).addEventListener("click", () => this.control("cancel"));
      document.getElementById(ids.chat).addEventListener("click", () => {
        if (this.job && this.job.checkpoint_ready) Studio.views.go("chat", { model: (this.job.result || {}).model_id || this.job.checkpoint, new: "1" });
      });
      runPanels.add(this);
      if (!runFeed) runFeed = Studio.subscribe("/api/events", data => {
        for (const panel of runPanels) panel.update(data.job);
      });
      this.refresh();
    }

    async refresh() {
      try {
        const data = await Studio.api("/api/job");
        this.update(data.job);
      } catch (error) {
        Studio.toast(error.message, "error");
      }
    }

    async control(action) {
      if (!this.job) return;
      if (action === "cancel" && !await Studio.confirm({ title: "Cancel training?", message: "Stop without saving another checkpoint. Checkpoints written earlier are kept.", confirmLabel: "Cancel run", danger: true })) return;
      await Studio.run(() => Studio.api(`/api/${action}`, { body: { id: this.job.id } }));
    }

    update(job) {
      this.job = job;
      const empty = document.getElementById(this.ids.empty);
      const content = document.getElementById(this.ids.content);
      empty.classList.toggle("hidden", Boolean(job));
      content.classList.toggle("hidden", !job);
      if (!job) { this.chart.update({ points: [], validation: [] }); return; }
      document.getElementById(this.ids.status).replaceChildren(Studio.ui.status(job.status));
      document.getElementById(this.ids.title).textContent = job.title || "Training";
      document.getElementById(this.ids.stage).textContent = `${job.stage || "Working"}${job.error ? ` · ${job.error}` : ""}`;
      const current = job.current_step || 0;
      const total = job.total_steps || 0;
      const percent = total ? 100 * current / total : 0;
      this.progress.update(percent, job.status === "failed" ? "failed" : job.status === "completed" ? "done" : total ? "active" : "indeterminate");
      const summary = summarize(job.metrics || {});
      const stats = Studio.clear(document.getElementById(this.ids.stats));
      stats.append(Studio.ui.stat({ compact: true, label: "Steps", value: `${format.number(current)} / ${total ? format.number(total) : "—"}` }),
        Studio.ui.stat({ compact: true, label: "Latest loss", value: summary.loss }),
        Studio.ui.stat({ compact: true, label: "Best loss", value: summary.best }),
        Studio.ui.stat({ compact: true, label: "Throughput", value: summary.throughput }));
      this.chart.update(job.metrics || { points: [], validation: [] });
      const logs = document.getElementById(this.ids.logs);
      const atBottom = logs.scrollHeight - logs.scrollTop - logs.clientHeight < 30;
      logs.textContent = (job.logs || []).join("\n");
      if (atBottom) logs.scrollTop = logs.scrollHeight;
      const active = ["preparing", "running"].includes(job.status);
      document.getElementById(this.ids.pause).disabled = !active || job.pause_requested;
      document.getElementById(this.ids.cancel).disabled = !active || job.cancel_requested;
      document.getElementById(this.ids.chat).disabled = !job.checkpoint_ready;
      Studio.emit("training", job);
    }
  }

  Studio.trainingViz = { LossChart, RunPanel, parseTrainingMetrics, formatLoss, formatThroughput, niceTicks, summarize };
})();
