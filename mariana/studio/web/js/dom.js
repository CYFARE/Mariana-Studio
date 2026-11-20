"use strict";

/* Studio DOM: element construction and the few selectors every module uses.
   No workspace, API or formatting knowledge lives here. */
(function () {
  const Studio = (window.Studio = {});
  const SVG_NS = "http://www.w3.org/2000/svg";
  const PROPERTIES = new Set([
    "value", "checked", "disabled", "selected", "hidden", "type", "href", "title", "id", "placeholder",
    "name", "min", "max", "step", "rows", "tabIndex", "htmlFor", "open", "readOnly", "required",
    "multiple", "src", "alt", "target", "rel", "draggable", "spellcheck", "autocomplete", "download",
    "indeterminate", "colSpan", "width", "height",
  ]);

  const $ = id => document.getElementById(id);
  const $$ = (selector, root) => Array.from((root || document).querySelectorAll(selector));

  function append(parent, children) {
    for (const child of children) {
      if (child === null || child === undefined || child === false || child === "") continue;
      if (Array.isArray(child)) append(parent, child);
      else parent.append(child instanceof Node ? child : document.createTextNode(String(child)));
    }
    return parent;
  }

  function appendChildren(parent, ...children) {
    return append(parent, children);
  }

  /* Build an element: h("button", { class: "button", onclick: handler }, "Label"). */
  function h(tag, attributes, ...children) {
    const element = document.createElement(tag);
    for (const [key, value] of Object.entries(attributes || {})) {
      if (value === null || value === undefined || value === false) continue;
      if (key === "class") element.className = value;
      else if (key === "text") element.textContent = value;
      else if (key === "dataset") Object.assign(element.dataset, value);
      else if (key === "style") setStyle(element, value);
      else if (key.startsWith("on") && typeof value === "function") element.addEventListener(key.slice(2), value);
      else if (PROPERTIES.has(key)) element[key] = value;
      else element.setAttribute(key, value === true ? "" : String(value));
    }
    return append(element, children);
  }

  /* Styles are set through the CSSOM: the page's CSP forbids style="" attributes. */
  function setStyle(element, styles) {
    for (const [name, value] of Object.entries(styles || {})) {
      if (name.startsWith("--")) element.style.setProperty(name, value);
      else element.style[name] = value;
    }
  }

  function icon(name, className) {
    const svg = document.createElementNS(SVG_NS, "svg");
    svg.setAttribute("class", className ? `icon ${className}` : "icon");
    svg.setAttribute("aria-hidden", "true");
    const use = document.createElementNS(SVG_NS, "use");
    use.setAttribute("href", `#i-${name}`);
    svg.append(use);
    return svg;
  }

  function clear(element) {
    element.replaceChildren();
    return element;
  }

  Object.assign(Studio, { $, $$, h, append: appendChildren, icon, clear, setStyle });
})();
