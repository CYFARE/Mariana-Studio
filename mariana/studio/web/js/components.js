"use strict";

/* Studio components: composed blocks shared by every workspace.
   They are deliberately plain data -> DOM functions; no view state lives here. */
(function () {
  const Studio = window.Studio;
  const { h, icon } = Studio;

  /* A page-level hero used by Home. */
  function hero(options) {
    return h("section", { class: "hero-panel" },
      h("div", { class: "hero-copy" },
        options.eyebrow ? h("div", { class: "eyebrow", text: options.eyebrow }) : null,
        h("h2", { text: options.title }),
        h("p", { text: options.text })),
      h("div", { class: "hero-actions row wrap" }, options.actions || null));
  }

  /* Step/icon + title + description + optional right-aligned action. */
  function sectionHeading(options) {
    let mark = null;
    if (options.step !== undefined) mark = h("span", { class: "step-number", text: String(options.step) });
    else if (options.glyph) mark = h("span", { class: "section-glyph" }, icon(options.glyph));
    return h("div", { class: "section-heading" },
      mark,
      h("div", { class: "grow" },
        h("h2", { text: options.title }),
        options.description ? h("p", { class: "muted small", text: options.description }) : null),
      options.action ? h("div", { class: "row wrap" }, options.action) : null);
  }

  /* A card with optional header and footer.  Content goes in the padded body. */
  function card(options = {}) {
    const element = h("section", {
      class: `card${options.soft ? " soft" : ""}${options.className ? ` ${options.className}` : ""}`,
    });
    if (options.title || options.description || options.actions) {
      element.append(h("div", { class: "card-header row between top wrap" },
        h("div", { class: "grow" },
          options.title ? h("h2", { text: options.title }) : null,
          options.description ? h("p", { text: options.description }) : null),
        options.actions ? h("div", { class: "row wrap" }, options.actions) : null));
    }
    const body = h("div", { class: options.bodyClass || "card-body stack" }, options.content || null);
    if (options.children) Studio.append(body, options.children);
    if (body.childNodes.length || !element.childNodes.length) element.append(body);
    if (options.footer) element.append(h("div", { class: "card-footer" }, options.footer));
    return element;
  }

  function toolbar(...children) {
    return h("div", { class: "toolbar" }, children);
  }

  function actionCard(options) {
    return h("a", { class: "action-card", href: options.href },
      h("span", { class: "action-icon" }, icon(options.glyph, "large")),
      h("strong", { text: options.title }),
      h("p", { text: options.description }));
  }

  function statRow(stats) {
    return h("div", { class: "stat-row" }, stats.map( options => Studio.ui.stat(options)));
  }

  function tagList(tags) {
    const list = h("div", { class: "tag-list" });
    for (const tag of tags || []) list.append(typeof tag === "string" ? Studio.ui.badge(tag) : tag);
    return list;
  }

  /* A model, dataset or repository card: icon, copy, description, tags and actions. */
  function resultCard(options) {
    const cardElement = h("article", {
      class: `result-card${options.unavailable ? " unavailable" : ""}${options.className ? ` ${options.className}` : ""}`,
    });
    cardElement.append(h("div", { class: "result-head" },
      h("div", { class: "result-icon" }, icon(options.glyph || "file")),
      h("div", { class: "grow result-copy" },
        h("h3", { class: "wrap-anywhere", text: options.title }),
        options.subtitle ? h("div", { class: "muted tiny wrap-anywhere", text: options.subtitle }) : null),
      options.status || null));
    if (options.description) cardElement.append(h("p", { class: "result-description", text: options.description }));
    if (options.meta) cardElement.append(h("div", { class: "result-meta", text: options.meta }));
    if (options.tags && options.tags.length) cardElement.append(tagList(options.tags));
    if (options.detail) cardElement.append(options.detail);
    if (options.actions && options.actions.length) cardElement.append(h("div", { class: "result-actions" }, options.actions));
    return cardElement;
  }

  /* One row in a list: title, subtitle, optional leading/trailing nodes and meta. */
  function listRow(options) {
    const tag = options.href ? "a" : options.onclick ? "button" : "div";
    return h(tag, {
      class: `list-row${options.active ? " active" : ""}${options.className ? ` ${options.className}` : ""}`,
      href: options.href || null,
      type: tag === "button" ? "button" : null,
      onclick: options.onclick || null,
    },
    options.leading || null,
    h("div", { class: "list-row-copy" },
      h("strong", { class: "list-row-title", text: options.title }),
      options.subtitle ? h("span", { class: "list-row-subtitle", text: options.subtitle }) : null),
    options.meta ? h("span", { class: "list-row-meta", text: options.meta }) : null,
    options.trailing || null);
  }

  Studio.components = {
    hero, sectionHeading, card, toolbar, actionCard, statRow, tagList, resultCard, listRow,
  };
})();
