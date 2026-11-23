"use strict";

/* Studio.markdown: a small, safe Markdown renderer for model output.
   Everything is HTML-escaped before any tag is produced, so text from a model
   or a file can never inject markup.  This is the only file that may assign
   innerHTML. */
(function () {
  const Studio = window.Studio;
  const ESCAPES = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
  const FENCE = /```([A-Za-z0-9_+.#-]*)[ \t]*\n([\s\S]*?)(?:\n?```|$)/g;
  const ITEM = /^(\s*)([-*+]|\d{1,9}[.)])\s+(.*)$/;
  const RULE = /^\s*([-*_])(?:\s*\1){2,}\s*$/;
  const HEADING = /^(#{1,4})\s+(.+?)\s*#*$/;
  const TABLE_DIVIDER = /^\s*\|?\s*:?-{2,}:?\s*(?:\|\s*:?-{2,}:?\s*)*\|?\s*$/;
  const SAFE_LINK = /^(?:https?:\/\/|mailto:)[^\s<>"']+$/i;
  const COPY_BUTTON = '<button class="icon-button small code-copy" type="button" data-copy-code aria-label="Copy code">'
    + '<svg class="icon small" aria-hidden="true"><use href="#i-copy"/></svg></button>';

  function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, character => ESCAPES[character]);
  }

  function codeBlock(code, language) {
    const label = language ? `<span class="code-lang">${escapeHtml(language)}</span>` : "";
    return `<div class="code-block">${label}${COPY_BUTTON}<pre><code>${escapeHtml(code.replace(/\n$/, ""))}</code></pre></div>`;
  }

  /* Inline formatting.  The input is already escaped. */
  function inline(escaped) {
    const held = [];
    const hold = html => `\u0001${held.push(html) - 1}\u0001`;
    let text = escaped
      .replace(/```([^`\n]+)```/g, (_, code) => hold(`<code>${code}</code>`))
      .replace(/`([^`\n]+)`/g, (_, code) => hold(`<code>${code}</code>`));
    text = text.replace(/\[([^\]\n]+)\]\(([^\s)]+)\)/g, (whole, label, url) => (
      SAFE_LINK.test(url) ? hold(`<a href="${url}" target="_blank" rel="noreferrer noopener">${label}</a>`) : whole
    ));
    text = text.replace(/(^|[\s(])(https?:\/\/[^\s<>"']+[^\s<>"'.,;:!?)])/g, (_, lead, url) => (
      `${lead}${hold(`<a href="${url}" target="_blank" rel="noreferrer noopener">${url}</a>`)}`
    ));
    text = text
      .replace(/\*\*([^*\n]+(?:\*(?!\*)[^*\n]+)*)\*\*/g, "<strong>$1</strong>")
      .replace(/(^|[^\w_])__([^_\n]+)__(?![\w_])/g, "$1<strong>$2</strong>")
      .replace(/(^|[^*\w])\*([^*\s][^*\n]*?)\*(?![*\w])/g, "$1<em>$2</em>")
      .replace(/(^|[^\w_])_([^_\s][^_\n]*?)_(?![\w_])/g, "$1<em>$2</em>")
      .replace(/~~([^~\n]+)~~/g, "<del>$1</del>");
    return text.replace(/\u0001(\d+)\u0001/g, (_, index) => held[Number(index)] || "");
  }

  function indentOf(line) {
    return /^\s*/.exec(line)[0].replace(/\t/g, "    ").length;
  }

  function isOrdered(marker) {
    return /\d/.test(marker);
  }

  function cells(row) {
    return row.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map(cell => cell.trim());
  }

  function isTable(lines, index) {
    return lines[index].includes("|") && index + 1 < lines.length && TABLE_DIVIDER.test(lines[index + 1])
      && cells(lines[index]).length === cells(lines[index + 1]).length;
  }

  function startsBlock(lines, index) {
    const line = lines[index];
    return /^\u0000\d+\u0000$/.test(line.trim()) || HEADING.test(line) || RULE.test(line)
      || /^&gt;/.test(line) || ITEM.test(line) || isTable(lines, index);
  }

  function renderList(lines, start, render = true) {
    const first = ITEM.exec(lines[start]);
    const base = indentOf(lines[start]);
    const ordered = isOrdered(first[2]);
    const items = [];
    let index = start;
    while (index < lines.length) {
      const line = lines[index];
      if (!line.trim()) {
        // A blank line stays inside the list when another item of it follows.
        let next = index + 1;
        while (next < lines.length && !lines[next].trim()) next += 1;
        if (next < lines.length && ITEM.test(lines[next]) && indentOf(lines[next]) >= base) {
          index = next;
          continue;
        }
        break;
      }
      const match = ITEM.exec(line);
      const indent = indentOf(line);
      if (!match) {
        if (indent > base && items.length) {
          items[items.length - 1].text += `\n${line.trim()}`;
          index += 1;
          continue;
        }
        break;
      }
      if (indent < base) break;
      if (indent >= base + 2 && items.length) {
        const [html, next] = renderList(lines, index, render);
        items[items.length - 1].children += html;
        index = next;
        continue;
      }
      if (isOrdered(match[2]) !== ordered) break;
      items.push({ text: match[3], children: "" });
      index += 1;
    }
    if (!render) return ["", index];
    const tag = ordered ? "ol" : "ul";
    const number = ordered ? parseInt(first[2], 10) : 1;
    const open = ordered && number > 1 ? `<ol start="${number}">` : `<${tag}>`;
    const body = items.map(item => `<li>${inline(item.text).replace(/\n/g, "<br>")}${item.children}</li>`).join("");
    return [`${open}${body}</${tag}>`, index];
  }

  function renderTable(lines, start, render = true) {
    const head = cells(lines[start]);
    let index = start + 2;
    const rows = [];
    while (index < lines.length && lines[index].trim() && lines[index].includes("|")) {
      rows.push(cells(lines[index]));
      index += 1;
    }
    if (!render) return ["", index];
    const header = head.map(cell => `<th>${inline(cell)}</th>`).join("");
    const body = rows.map(row => `<tr>${head.map((_, column) => `<td>${inline(row[column] || "")}</td>`).join("")}</tr>`).join("");
    return [`<table><thead><tr>${header}</tr></thead><tbody>${body}</tbody></table>`, index];
  }

  /* Walk top-level blocks.  With `options.output` the caller receives one HTML
     string per block; `options.reuse` blocks are copied from `options.rendered`
     without running inline formatting again.  A streaming reply only grows, so
     every block before the last is final and can be reused. */
  function renderBlocks(lines, blocks, options = {}) {
    const output = options.output || [];
    const reuse = Math.max(0, options.reuse || 0);
    const rendered = options.rendered || [];
    const cachedAt = position => position < reuse && typeof rendered[position] === "string"
      ? rendered[position] : null;
    let index = 0;
    while (index < lines.length) {
      const line = lines[index];
      if (!line.trim()) {
        index += 1;
        continue;
      }
      const cached = cachedAt(output.length);
      const placeholder = /^\u0000(\d+)\u0000$/.exec(line.trim());
      if (placeholder) {
        output.push(cached !== null ? cached : (blocks[Number(placeholder[1])] || ""));
        index += 1;
        continue;
      }
      const heading = HEADING.exec(line);
      if (heading) {
        if (cached !== null) output.push(cached);
        else output.push(`<h${heading[1].length}>${inline(heading[2])}</h${heading[1].length}>`);
        index += 1;
        continue;
      }
      if (RULE.test(line)) {
        output.push(cached !== null ? cached : "<hr>");
        index += 1;
        continue;
      }
      if (/^&gt;/.test(line)) {
        const quoted = [];
        while (index < lines.length && /^&gt;/.test(lines[index])) {
          quoted.push(lines[index].replace(/^&gt;\s?/, ""));
          index += 1;
        }
        output.push(cached !== null ? cached : `<blockquote>${renderBlocks(quoted, blocks)}</blockquote>`);
        continue;
      }
      if (ITEM.test(line)) {
        const [html, next] = renderList(lines, index, cached === null);
        output.push(cached !== null ? cached : html);
        index = next;
        continue;
      }
      if (isTable(lines, index)) {
        const [html, next] = renderTable(lines, index, cached === null);
        output.push(cached !== null ? cached : html);
        index = next;
        continue;
      }
      const paragraph = [line];
      index += 1;
      while (index < lines.length && lines[index].trim() && !startsBlock(lines, index)) {
        paragraph.push(lines[index]);
        index += 1;
      }
      output.push(cached !== null ? cached
        : `<p>${inline(paragraph.join("\n")).replace(/\n/g, "<br>")}</p>`);
    }
    return options.output ? output : output.join("");
  }

  /* Markdown text -> one HTML string per top-level block.  Closed code fences
     are cached by source so an unchanged fence is escaped once, not per tick. */
  function renderDocument(text, options = {}) {
    const source = String(text === null || text === undefined ? "" : text)
      .replace(/\r\n?/g, "\n").replace(/[\u0000\u0001]/g, "");
    const blocks = [];
    const fenceCache = options.fenceCache;
    const body = source.replace(FENCE, (whole, language, code) => {
      let html;
      if (fenceCache && whole.endsWith("```")) {
        const key = `${language}\u0000${code}`;
        html = fenceCache.get(key);
        if (html === undefined) {
          html = codeBlock(code, language);
          fenceCache.set(key, html);
        }
      } else html = codeBlock(code, language);
      blocks.push(html);
      return `\n\u0000${blocks.length - 1}\u0000\n`;
    });
    return renderBlocks(escapeHtml(body).split("\n"), blocks, options);
  }

  /* Markdown text -> HTML string.  Safe to assign to innerHTML. */
  function render(text) {
    return renderDocument(text);
  }

  /* One HTML string per top-level block; joins back to render(). */
  function blocks(text) {
    return renderDocument(text, { output: [] });
  }

  /* Add syntax colour and Mermaid previews to code fences after insertion.
     Streaming callers can pass {enhance: false} and enhance the final text. */
  function highlightBlock(block) {
    if (block.dataset.highlighted || !Studio.highlight) return;
    const label = block.querySelector(".code-lang");
    const language = label ? label.textContent.trim() : "";
    if (!language) return;
    const grammar = Studio.highlight.resolve(language);
    if (!Studio.highlight.hasGrammar(grammar)) return;
    const code = block.querySelector("pre code");
    if (!code) return;
    const source = code.textContent;
    if (source.length > 40000) return;
    const lines = source.split("\n");
    let state = null;
    const fragment = document.createDocumentFragment();
    lines.forEach((line, index) => {
      const result = Studio.highlight.scanLine(line, state, grammar);
      state = result.state;
      let cursor = 0;
      for (const token of result.tokens) {
        if (token.start > cursor) fragment.append(document.createTextNode(line.slice(cursor, token.start)));
        const span = document.createElement("span");
        span.className = `tok-${token.type}`;
        span.textContent = line.slice(token.start, token.end);
        fragment.append(span);
        cursor = token.end;
      }
      if (cursor < line.length) fragment.append(document.createTextNode(line.slice(cursor)));
      if (index < lines.length - 1) fragment.append(document.createTextNode("\n"));
    });
    code.replaceChildren(fragment);
    block.dataset.highlighted = "true";
  }

  /* Offscreen history is highlighted in idle slices so opening a long chat
     paints immediately.  Visible or small renders still highlight eagerly. */
  const lazyBlocks = [];
  let lazyCursor = 0;
  let lazyScheduled = false;

  function scheduleLazyHighlight() {
    if (lazyScheduled) return;
    lazyScheduled = true;
    const run = deadline => {
      lazyScheduled = false;
      let budget = 12;
      while (lazyCursor < lazyBlocks.length && budget > 0
          && (!deadline || deadline.timeRemaining() > 2)) {
        const block = lazyBlocks[lazyCursor];
        lazyCursor += 1;
        budget -= 1;
        if (block && block.isConnected) highlightBlock(block);
      }
      if (lazyCursor < lazyBlocks.length) {
        if (lazyCursor > 256) {
          lazyBlocks.splice(0, lazyCursor);
          lazyCursor = 0;
        }
        scheduleLazyHighlight();
      } else {
        lazyBlocks.length = 0;
        lazyCursor = 0;
      }
    };
    if (typeof requestIdleCallback === "function") requestIdleCallback(run, { timeout: 1000 });
    else setTimeout(run, 16);
  }

  function queueHighlight(block) {
    if (block.dataset.enhanced || block.dataset.highlighted || block.dataset.highlightQueued) return;
    block.dataset.highlightQueued = "true";
    lazyBlocks.push(block);
    scheduleLazyHighlight();
  }

  function enhance(root, options = {}) {
    if (!root) return root;
    if (Studio.diagrams) Studio.diagrams.enhance(root);
    for (const block of Array.from(root.querySelectorAll(".code-block"))) {
      if (block.dataset.enhanced) continue;
      if (options.lazy) queueHighlight(block);
      else highlightBlock(block);
    }
    return root;
  }

  /* Render markdown into an element. */
  function into(element, text, options = {}) {
    element.classList.add("markdown");
    element.innerHTML = render(text);
    if (options.enhance !== false) enhance(element, options);
    return element;
  }

  const streamStates = new WeakMap();

  function reset(element) {
    streamStates.delete(element);
  }

  /* Incremental paint for append-only model output.  Completed top-level
     blocks are left alone; only the growing last block is replaced.  Call into()
     once the stream ends for the authoritative render and highlighting. */
  function stream(element, text) {
    element.classList.add("markdown");
    let state = streamStates.get(element);
    if (!state) {
      /* A fresh stream owns the whole content: clear any text or blocks left by
         a previous render or by the reset() before a slid preview window. */
      element.replaceChildren();
      state = { rendered: [], count: 0, tail: null, fenceCache: new Map() };
      streamStates.set(element, state);
    }
    let options = {
      output: [], reuse: state.count > 0 ? state.count - 1 : 0,
      rendered: state.rendered, fenceCache: state.fenceCache,
    };
    let pieces = renderDocument(text, options);
    /* Content was replaced or edited backwards: rebuild the block list. */
    if (pieces.length < state.count) {
      element.replaceChildren();
      state.tail = null;
      state.count = 0;
      state.rendered = [];
      pieces = renderDocument(text, { output: [], reuse: 0, rendered: [], fenceCache: state.fenceCache });
    }
    /* The old tail may have become a stable block with final content. */
    if (state.tail) {
      state.tail.remove();
      state.tail = null;
    }
    const already = state.count > 0 ? state.count - 1 : 0;
    const stable = Math.max(0, pieces.length - 1);
    if (stable > already) {
      const template = document.createElement("template");
      for (let index = already; index < stable; index += 1) {
        template.innerHTML = pieces[index];
        const node = template.content.firstElementChild;
        if (node) element.append(node);
      }
    }
    if (pieces.length) {
      const template = document.createElement("template");
      template.innerHTML = pieces[pieces.length - 1];
      const tail = template.content.firstElementChild;
      if (tail) {
        element.append(tail);
        state.tail = tail;
      }
    }
    state.count = pieces.length;
    state.rendered = pieces;
    return element;
  }

  document.addEventListener("click", event => {
    const button = event.target.closest ? event.target.closest("[data-copy-code]") : null;
    if (!button) return;
    const block = button.closest(".code-block");
    const code = block ? block.querySelector("pre") : null;
    if (code) Studio.copy(code.textContent, "Code copied");
  });

  Studio.markdown = { render, blocks, into, stream, reset, enhance, escapeHtml };
})();
