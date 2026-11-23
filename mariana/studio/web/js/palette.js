"use strict";

(function () {
  const Studio = window.Studio;
  const { $, h, icon } = Studio;
  const commands = [];
  let filtered = [];
  let selected = 0;

  function add(command) {
    commands.push(command);
  }

  function defaults() {
    for (const name of Studio.views.order) {
      const view = Studio.views.registry.get(name);
      add({
        label: view.title || name, hint: view.hint || view.subtitle || "Workspace",
        glyph: view.glyph || name, keywords: name, action: () => Studio.views.go(name),
      });
    }
    add({ label: "New chat", hint: "Action", glyph: "plus", action: () => Studio.views.go("chat", { new: "1" }) });
    add({ label: "Download a model", hint: "Action", glyph: "download", action: () => Studio.views.go("models", { tab: "discover" }) });
    add({ label: "Add a remote endpoint", hint: "Action", glyph: "globe", action: () => Studio.views.go("models", { tab: "endpoints", add: "1" }) });
    add({ label: "Create an agent", hint: "Action", glyph: "agents", action: () => Studio.views.go("agents", { new: "1" }) });
    add({ label: "Manage Python dependencies", hint: "Install optional requirements", glyph: "download", action: () => Studio.dependencies.open() });
    add({ label: "Choose colour theme", hint: "System, light or dark", glyph: "sun", action: () => $("theme-toggle").click() });
    add({ label: "Show background jobs", hint: "Action", glyph: "jobs", action: () => $("jobs-button").click() });
  }

  function score(command, query) {
    const text = `${command.label} ${command.keywords || ""} ${command.hint || ""}`.toLowerCase();
    if (!query) return 1;
    if (command.label.toLowerCase().startsWith(query)) return 5;
    if (command.label.toLowerCase().includes(query)) return 3;
    return query.split(/\s+/).every(word => text.includes(word)) ? 1 : 0;
  }

  function render() {
    const query = $("palette-input").value.trim().toLowerCase();
    /* Keep the resting state short: workspaces and actions.  Models join only
       once the person types, so opening the palette never scrolls a catalogue. */
    const extra = query
      ? Studio.models.list.filter(model => model.capabilities && model.capabilities.chat).map(model => ({
        label: model.name, hint: `Chat · ${model.source_label}`, glyph: "chat",
        keywords: `${model.id} ${model.kind}`,
        action: () => Studio.views.go("chat", { model: model.id, new: "1" }),
      })).slice(0, 8)
      : [];
    filtered = commands.filter(command => !command.when || command.when()).concat(extra).map(command => ({ command, value: score(command, query) }))
      .filter(item => item.value > 0).sort((a, b) => b.value - a.value)
      .slice(0, 30).map(item => item.command);
    selected = Math.max(0, Math.min(selected, filtered.length - 1));
    const list = Studio.ui.render($("palette-list"));
    $("palette-status").textContent = filtered.length
      ? `${filtered.length} result${filtered.length === 1 ? "" : "s"}`
      : "No results";
    if (!filtered.length) {
      $("palette-input").removeAttribute("aria-activedescendant");
      list.append(h("div", { class: "palette-empty", text: "No matching workspaces, models or actions." }));
      return;
    }
    filtered.forEach((command, index) => {
      const item = h("button", {
        id: `palette-option-${index}`, class: "palette-item", type: "button", role: "option",
        "aria-selected": String(index === selected), onclick: () => execute(index),
      }, icon(command.glyph || "search"), h("span", { text: command.label }), h("span", { class: "palette-hint", text: command.hint || "" }));
      list.append(item);
    });
    $("palette-input").setAttribute("aria-activedescendant", `palette-option-${selected}`);
  }

  function execute(index) {
    const command = filtered[index];
    if (!command) return;
    $("palette").close();
    command.action();
  }

  function open() {
    if ($("palette").open) return;
    Studio.models.ensure().then(() => { if ($("palette").open) render(); }).catch(() => {});
    $("palette-input").value = "";
    selected = 0;
    render();
    $("palette").showModal();
    $("palette-input").focus();
  }

  function start() {
    defaults();
    $("shell-command").addEventListener("click", open);
    Studio.shortcut("mod+k", open);
    $("palette").addEventListener("click", event => { if (event.target === $("palette")) $("palette").close(); });
    $("palette-input").addEventListener("input", () => { selected = 0; render(); });
    $("palette-input").addEventListener("keydown", event => {
      if (event.key === "ArrowDown") selected = Math.min(filtered.length - 1, selected + 1);
      else if (event.key === "ArrowUp") selected = Math.max(0, selected - 1);
      else if (event.key === "Home") selected = 0;
      else if (event.key === "End") selected = filtered.length - 1;
      else if (event.key === "Enter") { event.preventDefault(); execute(selected); return; }
      else return;
      event.preventDefault();
      render();
      const active = $("palette-list").children[selected];
      if (active) active.scrollIntoView({ block: "nearest" });
    });
  }

  Studio.palette = { start, open, add };
})();
