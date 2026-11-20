"use strict";

/* Studio.artifacts: turn model replies and chat attachments into previewable
   artifacts.  Fenced code, HTML designs, SVG, Mermaid diagrams, Markdown,
   CSV tables and images share one small model so the Chat preview panel can
   switch between rendered output and source without duplicating rendering
   rules here.  Untrusted HTML is sanitised and rendered inside a shadow root;
   scripts never run. */
(function () {
  const Studio = window.Studio;
  const { h } = Studio;
  const FENCE = /```([A-Za-z0-9_+.#-]*)[ \t]*\n([\s\S]*?)(?:\n?```|$)/g;
  const MAX_SOURCE = 200_000;
  const MAX_LINES = 5_000;
  const MAX_TABLE_ROWS = 500;
  const MAX_TABLE_COLUMNS = 40;
  const BLOCKED_TAGS = new Set([
    "script", "iframe", "object", "embed", "link", "meta", "base", "form", "input",
    "button", "textarea", "select", "option", "audio", "video", "source", "track",
    "frame", "frameset",
  ]);
  const BLOCKED_ATTRIBUTES = new Set([
    "srcdoc", "srcset", "poster", "formaction", "action", "ping", "background", "xlink:href",
  ]);
  const EXTENSIONS = {
    python: "py", py: "py", javascript: "js", js: "js", typescript: "ts", ts: "ts",
    tsx: "tsx", jsx: "jsx", html: "html", htm: "html", css: "css", scss: "scss",
    json: "json", jsonc: "jsonc", yaml: "yaml", yml: "yaml", toml: "toml",
    markdown: "md", md: "md", mermaid: "mmd", mmd: "mmd", svg: "svg",
    sh: "sh", shell: "sh", bash: "sh", powershell: "ps1", sql: "sql", csv: "csv",
    tsv: "tsv", go: "go", rust: "rs", rs: "rs", java: "java", kotlin: "kt",
    ruby: "rb", rb: "rb", php: "php", c: "c", cpp: "cpp", csharp: "cs", cs: "cs",
  };
  const MIME = {
    html: "text/html", svg: "image/svg+xml", mermaid: "text/plain", markdown: "text/markdown",
    table: "text/csv", json: "application/json", image: "application/octet-stream",
    file: "application/octet-stream", code: "text/plain",
  };
  const LABELS = {
    html: "HTML design", svg: "SVG graphic", mermaid: "Mermaid diagram", markdown: "Markdown",
    table: "Table", json: "JSON", image: "Image", file: "File", code: "Code",
  };

  function text(value) {
    return String(value === undefined || value === null ? "" : value);
  }

  function clean(value) {
    return text(value).replace(/\s+/g, " ").trim();
  }

  function slug(value, fallback = "artifact") {
    const result = clean(value).toLowerCase().replace(/[^a-z0-9._-]+/g, "-").replace(/^-+|-+$/g, "");
    return result || fallback;
  }

  function extensionFor(language, kind) {
    const name = text(language).toLowerCase();
    if (EXTENSIONS[name]) return EXTENSIONS[name];
    if (kind === "html") return "html";
    if (kind === "svg") return "svg";
    if (kind === "mermaid") return "mmd";
    if (kind === "markdown") return "md";
    if (kind === "table") return "csv";
    if (kind === "json") return "json";
    return "txt";
  }

  function kindFor(language, source) {
    const name = text(language).trim().toLowerCase();
    const body = text(source);
    if (["html", "htm", "xhtml"].includes(name)) return "html";
    if (name === "svg") return "svg";
    if (["markup", "xml", "svg+xml"].includes(name)) {
      if (/^\s*<svg[\s>]/i.test(body)) return "svg";
      if (/<(?:html|body|head|style|div|main|section|h1|p)\b/i.test(body)) return "html";
      return "code";
    }
    if (["mermaid", "mmd", "mermaid.js"].includes(name)) return "mermaid";
    if (["markdown", "md", "commonmark"].includes(name)) return "markdown";
    if (["csv", "tsv"].includes(name)) return "table";
    if (["json", "jsonc", "json5"].includes(name)) return "json";
    if (!name && /^\s*<svg[\s>]/i.test(body)) return "svg";
    if (!name && /^(?:flowchart|graph|sequenceDiagram|pie)\b/im.test(body)) return "mermaid";
    if (!name && /<\s*(?:!doctype|html|head|body|style|div|main|section|article|header|footer|nav|h[1-6]|p|span|button|table|form|ul|ol|li)\b/i.test(body)) {
      return "html";
    }
    return "code";
  }

  function previewable(kind) {
    return ["html", "svg", "mermaid", "markdown", "table", "image"].includes(kind);
  }

  function labelFor(kind, language) {
    const base = LABELS[kind] || "Code";
    if (kind === "code" && clean(language)) return clean(language);
    return base;
  }

  function titleFor(language, source, kind, index) {
    const body = text(source);
    if (kind === "html" || kind === "svg") {
      const match = /<title[^>]*>([\s\S]*?)<\/title>/i.exec(body);
      const title = match ? clean(match[1].replace(/<[^>]*>/g, " ")) : "";
      if (title) return title;
    }
    if (kind !== "code") return LABELS[kind];
    const name = clean(language);
    return name ? `${name} snippet ${index + 1}` : `Snippet ${index + 1}`;
  }

  function codeArtifacts(content, messageIndex, role) {
    const results = [];
    const source = text(content).replace(/\r\n?/g, "\n");
    FENCE.lastIndex = 0;
    let match;
    while ((match = FENCE.exec(source))) {
      const language = text(match[1]).trim().toLowerCase();
      const code = match[2].replace(/\n$/, "");
      if (!code.trim()) continue;
      const kind = kindFor(language, code);
      const order = results.length;
      results.push({
        id: `code:${messageIndex}:${order}`,
        scope: `m${messageIndex}`,
        messageIndex,
        role,
        order,
        kind,
        language,
        title: titleFor(language, code, kind, order),
        source: code,
        size: code.length,
        previewable: previewable(kind),
      });
    }
    return results;
  }

  function attachmentArtifact(attachment, scope, messageIndex, position = 0) {
    if (!attachment || typeof attachment !== "object") return null;
    const name = text(attachment.name);
    const extension = name.includes(".") ? name.split(".").pop().toLowerCase() : "";
    const semantic = {
      html: "html", htm: "html", xhtml: "html", svg: "svg", md: "markdown",
      markdown: "markdown", csv: "csv", tsv: "tsv", json: "json", jsonc: "json",
    }[extension];
    const language = semantic || text(attachment.language);
    const kind = attachment.kind === "image" ? "image"
      : attachment.kind === "text" ? kindFor(language, attachment.text) : "file";
    const source = text(attachment.text);
    return {
      id: `attach:${scope}:${position}:${text(attachment.id) || "0"}`,
      scope,
      messageIndex,
      kind,
      language,
      title: clean(attachment.name) || "Attachment",
      filename: clean(attachment.name),
      mime: text(attachment.mime),
      size: Number(attachment.size) || source.length || text(attachment.data_url).length,
      source,
      data_url: text(attachment.data_url),
      attachment: true,
      previewable: previewable(kind),
    };
  }

  function fromAttachments(list, scope, messageIndex) {
    const artifacts = [];
    (Array.isArray(list) ? list : []).forEach((attachment, position) => {
      const artifact = attachmentArtifact(attachment, scope, messageIndex, position);
      if (artifact) artifacts.push(artifact);
    });
    return artifacts;
  }

  function messageArtifacts(message, index) {
    if (!message || message.pending) return [];
    const artifacts = fromAttachments(message.attachments, `m${index}`, index);
    return artifacts.concat(codeArtifacts(message.content, index, message.role));
  }

  function collect(messages) {
    const artifacts = [];
    (Array.isArray(messages) ? messages : []).forEach((message, index) => {
      for (const artifact of messageArtifacts(message, index)) artifacts.push(artifact);
    });
    return artifacts;
  }

  function filenameFor(artifact) {
    const fallback = `${slug(artifact.title)}.${extensionFor(artifact.language, artifact.kind)}`;
    if (!artifact.filename) return fallback;
    const base = text(artifact.filename)
      .replace(/[\\/:*?"<>|\u0000-\u001f]/g, "-")
      .replace(/\.{2,}/g, ".")
      .replace(/^\.+/, "")
      .trim()
      .slice(0, 120);
    return base || fallback;
  }

  function blobFromDataUrl(dataUrl) {
    const comma = text(dataUrl).indexOf(",");
    if (comma < 0) return null;
    const meta = dataUrl.slice(5, comma).split(";");
    const mime = meta[0] || "application/octet-stream";
    const payload = dataUrl.slice(comma + 1);
    try {
      if (meta.includes("base64")) {
        const binary = atob(payload);
        const bytes = new Uint8Array(binary.length);
        for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
        return new Blob([bytes], { type: mime });
      }
      return new Blob([decodeURIComponent(payload)], { type: mime });
    } catch (error) {
      return null;
    }
  }

  function saveBlob(blob, name) {
    const url = URL.createObjectURL(blob);
    const link = h("a", { href: url, download: name });
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  function download(artifact) {
    if (!artifact) return;
    const name = filenameFor(artifact);
    if (artifact.data_url && !artifact.source) {
      const blob = blobFromDataUrl(artifact.data_url);
      if (blob) saveBlob(blob, name);
      else Studio.toast("This image could not be prepared for download.", "error");
      return;
    }
    const mime = artifact.mime || MIME[artifact.kind] || "text/plain";
    Studio.download(name, artifact.source || "", mime);
  }

  /* -- rendering ------------------------------------------------------------ */
  function sourceNode(artifact) {
    const wrap = h("div", { class: "chat-artifact-source" });
    const source = text(artifact.source).replace(/\r\n?/g, "\n");
    if (!source) {
      wrap.append(h("p", { class: "muted small", text: "This artifact has no text to show." }));
      return wrap;
    }
    const lines = source.split("\n");
    const clipped = lines.length > MAX_LINES || source.length > MAX_SOURCE;
    const body = clipped
      ? lines.slice(0, MAX_LINES).join("\n").slice(0, MAX_SOURCE)
      : source;
    const pre = h("pre", { class: "chat-artifact-code", tabindex: 0 });
    const highlighter = Studio.highlight;
    const grammar = highlighter ? highlighter.resolve(artifact.language, artifact.filename) : "text";
    if (!highlighter || !highlighter.hasGrammar(grammar) || body.length > 60_000) {
      pre.textContent = body;
    } else {
      let state = null;
      const rows = body.split("\n");
      rows.forEach((line, index) => {
        const result = highlighter.scanLine(line, state, grammar);
        state = result.state;
        let cursor = 0;
        for (const token of result.tokens) {
          if (token.start > cursor) pre.append(document.createTextNode(line.slice(cursor, token.start)));
          pre.append(h("span", { class: `tok-${token.type}`, text: line.slice(token.start, token.end) }));
          cursor = token.end;
        }
        if (cursor < line.length) pre.append(document.createTextNode(line.slice(cursor)));
        if (index < rows.length - 1) pre.append(document.createTextNode("\n"));
      });
    }
    wrap.append(pre);
    if (clipped) {
      wrap.append(h("p", { class: "help", text: `Showing the first ${MAX_LINES} lines or ${MAX_SOURCE} characters.` }));
    }
    return wrap;
  }

  function sanitizeDesign(body, token) {
    const rules = [];
    let inline = 0;
    for (const element of Array.from(body.querySelectorAll("*"))) {
      const tag = element.tagName.toLowerCase();
      if (BLOCKED_TAGS.has(tag)) {
        element.remove();
        continue;
      }
      for (const attribute of Array.from(element.attributes)) {
        const name = attribute.name.toLowerCase();
        if (name.startsWith("on") || BLOCKED_ATTRIBUTES.has(name)) {
          element.removeAttribute(attribute.name);
        } else if (name === "href") {
          element.removeAttribute(attribute.name);
        } else if (name === "src" && !/^(?:data:image\/|blob:|\/)/i.test(attribute.value)) {
          element.removeAttribute(attribute.name);
        } else if (name === "style") {
          const value = attribute.value.replace(/[<>}]/g, "").slice(0, 4000);
          if (value) {
            const className = `chat-inline-${token}-${inline}`;
            inline += 1;
            element.classList.add(className);
            rules.push(`.${className}{${value}}`);
          }
          element.removeAttribute(attribute.name);
        }
      }
    }
    return rules;
  }

  function scopeCss(css) {
    return text(css)
      .replace(/@import[^;]+;/gi, "")
      .replace(/(^|[{}])\s*:host(?:-context)?\b/gi, (match, lead) => `${lead} .chat-design-body`)
      .replace(/(^|[{}])\s*html\s*,\s*body\b/gi, (match, lead) => `${lead} .chat-design-body`)
      .replace(/(^|[{}])\s*body\s*,\s*html\b/gi, (match, lead) => `${lead} .chat-design-body`)
      .replace(/(^|[{}])\s*(html|body)\b/gi, (match, lead) => `${lead} .chat-design-body`)
      .replace(/(^|[{}])\s*:root\b/gi, (match, lead) => `${lead} .chat-design-body`);
  }

  function designNode(source) {
    const host = h("div", { class: "chat-design" });
    let shadow;
    try {
      shadow = host.attachShadow({ mode: "open" });
    } catch (error) {
      return h("div", { class: "chat-artifact-source" },
        h("p", { class: "muted small", text: "This browser cannot render the design preview; use Source." }));
    }
    const documentParser = new DOMParser();
    const parsed = documentParser.parseFromString(text(source), "text/html");
    const styles = [];
    for (const style of Array.from(parsed.querySelectorAll("style"))) {
      styles.push(style.textContent || "");
      style.remove();
    }
    const inlineRules = sanitizeDesign(parsed.body, Math.random().toString(36).slice(2, 8));
    const bodyStyle = parsed.body.getAttribute("style");
    if (bodyStyle) {
      inlineRules.push(`.chat-design-body{${bodyStyle.replace(/[<>}]/g, "").slice(0, 4000)}}`);
    }
    const bodyClasses = parsed.body.className;
    const base = ":host{display:block;min-height:100%;padding:16px;color:var(--text);"
      + "background:var(--panel);font:14px/1.5 var(--font-sans);}"
      + ":host *{box-sizing:border-box;}"
      + "img,svg,video,canvas{max-width:100%;height:auto;}"
      + "a{color:var(--accent);}";
    const designCss = styles.map(scopeCss)
      .concat(inlineRules)
      .join("\n")
      .slice(0, MAX_SOURCE);
    if (typeof CSSStyleSheet === "function" && "adoptedStyleSheets" in shadow) {
      const sheets = [];
      try {
        const baseSheet = new CSSStyleSheet();
        baseSheet.replaceSync(base);
        sheets.push(baseSheet);
      } catch (error) {
        /* Constructable stylesheets are unavailable: layout still renders unstyled. */
      }
      if (designCss.trim()) {
        try {
          const designSheet = new CSSStyleSheet();
          designSheet.replaceSync(designCss);
          sheets.push(designSheet);
        } catch (error) {
          /* Keep the base sheet when generated CSS is malformed. */
        }
      }
      if (sheets.length) shadow.adoptedStyleSheets = sheets;
    }
    const body = document.createElement("div");
    body.className = `chat-design-body${bodyClasses ? ` ${bodyClasses}` : ""}`;
    while (parsed.body.firstChild) body.append(parsed.body.firstChild);
    shadow.append(body);
    return host;
  }

  function svgNode(source, title) {
    const url = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(text(source))}`;
    return h("div", { class: "chat-artifact-image" },
      h("img", { src: url, alt: title || "SVG preview" }));
  }

  function imageNode(artifact) {
    if (!artifact.data_url) {
      return h("p", { class: "muted small", text: "This image was not stored with the conversation." });
    }
    return h("div", { class: "chat-artifact-image" },
      h("img", { src: artifact.data_url, alt: artifact.title || "Attached image" }));
  }

  function fileNode(artifact) {
    return h("div", { class: "chat-artifact-source" },
      Studio.ui.kv([
        ["Name", artifact.filename || artifact.title],
        ["Type", artifact.mime || "unknown"],
        ["Size", Studio.format.bytes(artifact.size || 0)],
      ]),
      h("p", { class: "help",
        text: "Only the file name travels with this message. Attach a text or image file to preview its contents." }));
  }

  function mermaidNode(source) {
    const element = Studio.diagrams ? Studio.diagrams.render(source) : null;
    if (!element) {
      return h("div", { class: "stack tight" },
        h("p", { class: "muted small", text: "The local diagram renderer does not support this diagram type. Use Source." }),
        sourceNode({ source, language: "mermaid", title: "Diagram" }));
    }
    const wrapper = h("div", { class: "diagram chat-artifact-diagram" }, element);
    wrapper.append(h("details", { class: "diagram-source" },
      h("summary", { text: "Diagram source" }),
      sourceNode({ source, language: "mermaid", title: "Diagram" })));
    return wrapper;
  }

  function markdownNode(source) {
    const body = h("div", { class: "markdown chat-artifact-markdown" });
    Studio.markdown.into(body, source);
    return body;
  }

  function splitRow(line, delimiter) {
    const cells = [];
    let value = "";
    let quoted = false;
    for (let index = 0; index < line.length; index += 1) {
      const char = line[index];
      if (char === '"') {
        if (quoted && line[index + 1] === '"') {
          value += '"';
          index += 1;
        } else {
          quoted = !quoted;
        }
      } else if (char === delimiter && !quoted) {
        cells.push(value.trim());
        value = "";
      } else {
        value += char;
      }
    }
    cells.push(value.trim());
    return cells;
  }

  function tableNode(source, language) {
    const lines = text(source).split(/\r?\n/).filter(line => line.trim());
    if (!lines.length) return h("p", { class: "muted small", text: "The table is empty." });
    const delimiter = text(language).toLowerCase() === "tsv" ? "\t" : ",";
    const rows = lines.slice(0, MAX_TABLE_ROWS).map(line => splitRow(line, delimiter).slice(0, MAX_TABLE_COLUMNS));
    const table = h("table", { class: "chat-artifact-table" });
    const head = h("thead");
    const headerRow = h("tr");
    for (const cell of rows[0]) headerRow.append(h("th", { text: cell || " " }));
    head.append(headerRow);
    const body = h("tbody");
    for (const row of rows.slice(1)) {
      const tr = h("tr");
      const width = rows[0].length;
      for (let index = 0; index < width; index += 1) tr.append(h("td", { text: row[index] || "" }));
      body.append(tr);
    }
    table.append(head, body);
    const wrap = h("div", { class: "chat-artifact-table-wrap" }, table);
    if (lines.length > MAX_TABLE_ROWS) {
      wrap.append(h("p", { class: "help", text: `Showing the first ${MAX_TABLE_ROWS} rows.` }));
    }
    return wrap;
  }

  function previewNode(artifact) {
    if (artifact.kind === "html") {
      const wrap = h("div", { class: "chat-artifact-rendered" }, designNode(artifact.source));
      if (/<script[\s>]/i.test(artifact.source)) {
        wrap.append(h("p", { class: "help",
          text: "Scripts are removed from the preview; layout and styles still render." }));
      }
      return wrap;
    }
    if (artifact.kind === "svg") return svgNode(artifact.source, artifact.title);
    if (artifact.kind === "image") return imageNode(artifact);
    if (artifact.kind === "mermaid") return mermaidNode(artifact.source);
    if (artifact.kind === "markdown") return markdownNode(artifact.source);
    if (artifact.kind === "table") return tableNode(artifact.source, artifact.language);
    return sourceNode(artifact);
  }

  function render(artifact, view) {
    if (!artifact) {
      return h("div", { class: "empty compact" },
        h("p", { class: "muted small", text: "Nothing selected." }));
    }
    if (artifact.kind === "file") return fileNode(artifact);
    if (view === "source" || !previewable(artifact.kind)) return sourceNode(artifact);
    return previewNode(artifact);
  }

  Studio.artifacts = {
    collect,
    messageArtifacts,
    fromAttachments,
    attachmentArtifact,
    kindFor,
    previewable,
    label: labelFor,
    filename: filenameFor,
    render,
    download,
  };
})();
