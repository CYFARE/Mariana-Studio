"use strict";

/* Studio.editor: the IDE code editor.  It keeps a textarea as the single source
   of truth (native selection, undo, IME and accessibility), then paints a
   virtualised, syntax-highlighted layer behind it.  Only visible lines are in
   the DOM, and lexer state is checkpointed so scrolling a 30k-line file never
   re-scans it from the top.  Everything is local: no build step, no CDN. */
(function () {
  const Studio = window.Studio;
  const CHECKPOINT_LINES = 64;
  const MAX_FIND_MATCHES = 5000;
  const PAIRS = { "(": ")", "[": "]", "{": "}", '"': '"', "'": "'", "`": "`" };
  const CLOSERS = new Set([")", "]", "}"]);
  const BRACKETS = { "(": ")", "[": "]", "{": "}", ")": "(", "]": "[", "}": "{" };

  /* A textarea exposes its value with LF only, so remember the file's endings
     and put them back on save.  A mixed file keeps LF. */
  function detectEol(text) {
    const value = String(text || "");
    return value.includes("\r\n") && !value.replace(/\r\n/g, "").includes("\n") ? "\r\n" : "\n";
  }

  function encodeEol(text, eol) {
    const value = String(text === undefined || text === null ? "" : text).replace(/\r\n/g, "\n");
    return eol === "\r\n" ? value.replace(/\n/g, "\r\n") : value;
  }

  function escapeRegExp(text) {
    return String(text).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  }

  /* Use the file's own indentation until the user chooses one explicitly. */
  function detectIndent(text) {
    const lines = String(text || "").split("\n", 4000);
    let tabs = 0;
    let spaces = 0;
    const widths = new Set();
    for (const line of lines) {
      const match = /^([ \t]+)\S/.exec(line);
      if (!match) continue;
      if (match[1].includes("\t")) tabs += 1;
      else {
        spaces += 1;
        if (match[1].length >= 2 && match[1].length <= 8) widths.add(match[1].length);
      }
    }
    if (tabs > spaces) return { mode: "tab", size: 4 };
    const width = widths.size ? Math.min(...widths) : 0;
    return width ? { mode: "spaces", size: width } : null;
  }

  function clamp(value, minimum, maximum) {
    return Math.max(minimum, Math.min(maximum, value));
  }

  /* All matches for the find bar.  The regex path also serves literal searches
     so case-insensitive matching keeps offsets into the original text. */
  function findMatches(text, query, options = {}) {
    const value = String(text === undefined || text === null ? "" : text);
    const needle = String(query === undefined || query === null ? "" : query);
    const limit = Math.max(1, Number(options.limit) || MAX_FIND_MATCHES);
    if (!needle) return { matches: [], invalid: false, truncated: false };
    let pattern;
    try {
      pattern = new RegExp(options.regex ? needle : escapeRegExp(needle), options.caseSensitive ? "g" : "gi");
    } catch (error) {
      return { matches: [], invalid: true, truncated: false };
    }
    const matches = [];
    let match;
    let truncated = false;
    while ((match = pattern.exec(value))) {
      if (!match[0].length) {
        pattern.lastIndex += 1;
        continue;
      }
      if (matches.length >= limit) {
        truncated = true;
        break;
      }
      matches.push({ start: match.index, end: match.index + match[0].length, text: match[0] });
    }
    return { matches, invalid: false, truncated };
  }

  function create(host, options = {}) {
    const highlighter = Studio.highlight || null;
    const store = Studio.store || null;
    const storedIndent = store && store.get ? store.get("editor-indent", null) : null;
    const savedIndent = storedIndent && typeof storedIndent === "object" ? storedIndent : null;
    let indentMode = options.indent === "tab" ? "tab"
      : options.indent === "spaces" ? "spaces"
        : savedIndent && savedIndent.mode === "tab" ? "tab" : "spaces";
    let tabSize = clamp(Number(options.tabSize || (savedIndent && savedIndent.size) || 4) || 4, 1, 8);
    if (!savedIndent && options.indent === undefined && options.tabSize === undefined) {
      const detected = detectIndent(options.value || "");
      if (detected) {
        indentMode = detected.mode;
        tabSize = clamp(detected.size, 1, 8);
      }
    }
    let indentUnit = indentMode === "tab" ? "\t" : " ".repeat(tabSize);

    const gutterCurrent = Studio.h("div", { class: "code-gutter-current", "aria-hidden": "true" });
    const gutterInner = Studio.h("div", { class: "code-gutter-inner", "aria-hidden": "true" });
    const gutter = Studio.h("div", {
      class: "code-gutter", "aria-hidden": "true", title: "Click a line number to select that line",
    }, gutterCurrent, gutterInner);
    const highlightInner = Studio.h("div", { class: "code-highlight-inner", "aria-hidden": "true" });
    const highlightLayer = Studio.h("div", { class: "code-highlight", "aria-hidden": "true" }, highlightInner);
    const lineBand = Studio.h("div", { class: "code-line-band", "aria-hidden": "true" });
    const input = Studio.h("textarea", {
      class: "code-input", spellcheck: false, autocomplete: "off", autocapitalize: "off",
      wrap: "off", "aria-label": options.label || "Code editor", "aria-multiline": "true",
    });
    const languageButton = Studio.h("button", {
      class: "code-language", type: "button", "aria-haspopup": "menu",
      title: "Change language mode",
    }, options.language || "Plain text");
    const selectionInfo = Studio.h("span", { class: "code-selection numeric" });
    const indentButton = Studio.h("button", {
      class: "code-indent", type: "button", "aria-haspopup": "menu",
      title: "Change indentation",
    });
    const initialEol = options.eol || detectEol(options.value || "");
    const eolInfo = Studio.h("span", { class: "code-eol", text: initialEol === "\r\n" ? "CRLF" : "LF" });
    const position = Studio.h("span", { class: "code-position numeric", text: "Ln 1, Col 1" });
    const scroll = Studio.h("div", { class: "code-scroll" }, lineBand, highlightLayer, input);
    const status = Studio.h("div", { class: "code-status" },
      languageButton, Studio.h("span", { class: "grow" }), selectionInfo, indentButton, eolInfo, position);
    const editor = Studio.h("div", { class: `code-editor${options.readOnly ? " readonly" : ""}` },
      Studio.h("div", { class: "code-surface" }, gutter, scroll), status);
    host.replaceChildren(editor);

    let silent = false;
    let readOnly = Boolean(options.readOnly);
    let starts = [0];
    let lineHeight = 20;
    let paddingTop = 12;
    let paddingLeft = 16;
    let measured = false;
    let charWidth = 0;
    let frame = null;
    let observer = null;
    let lastValue = "";
    let contentVersion = 0;
    let decorVersion = 0;
    let gutterFirst = -1;
    let gutterLast = -1;
    let gutterDirty = true;
    let checkpoints = [{ line: 0, state: null }];
    let rendered = { first: -1, last: -1, content: -1, decor: -1, language: "", scrollLeft: -1, scrollTop: -1 };
    let currentLine = 0;
    let brackets = [];
    // Scrolling repaints the visible range; bracket matching only changes with
    // the text, grammar or selection, so those three values key the cache.
    let bracketKey = "";
    let tabMovesFocus = false;
    let languageLabel = String(options.language || "");
    let grammarId = "text";

    const find = {
      open: false, replace: false, query: "", matches: [], index: -1,
      invalid: false, truncated: false, caseSensitive: false, regex: false, byLine: new Map(),
    };
    let findTimer = null;
    let findBar = null;
    let findInput = null;
    let findCount = null;
    let findReplaceRow = null;
    let replaceInput = null;
    let replaceButton = null;
    let replaceAllButton = null;
    let caseButton = null;
    let regexButton = null;

    function setClass(element, name, on) {
      element.classList.toggle(name, Boolean(on));
    }

    function setLabel(element, name, value) {
      if (typeof element.setAttribute === "function") element.setAttribute(name, String(value));
    }

    function setPressed(element, on) {
      setClass(element, "active", on);
      setLabel(element, "aria-pressed", on ? "true" : "false");
    }

    function onWindowResize() {
      measured = false;
      schedulePaint(false);
    }

    function toggleTabFocus() {
      tabMovesFocus = !tabMovesFocus;
      if (Studio.toast) {
        Studio.toast(tabMovesFocus
          ? "Tab now moves focus. Press Ctrl+M for indentation."
          : "Tab inserts indentation again.", "info", 2600);
      }
    }

    /* -- line index and lexer checkpoints ------------------------------------ */

    function rebuildIndex(previous = input.value) {
      const value = input.value;
      const result = [0];
      let firstDiff = -1;
      const maximum = Math.min(previous.length, value.length);
      for (let index = 0; index < value.length; index += 1) {
        if (firstDiff < 0 && index < maximum && value.charCodeAt(index) !== previous.charCodeAt(index)) {
          firstDiff = index;
        }
        if (value.charCodeAt(index) === 10) result.push(index + 1);
      }
      if (firstDiff < 0 && previous.length !== value.length) firstDiff = maximum;
      starts = result;
      return firstDiff;
    }

    function lineCount() {
      return starts.length;
    }

    function lineOf(offset) {
      const value = clamp(Number(offset) || 0, 0, input.value.length);
      let low = 0;
      let high = starts.length - 1;
      while (low < high) {
        const middle = (low + high + 1) >> 1;
        if (starts[middle] <= value) low = middle;
        else high = middle - 1;
      }
      return { index: low, column: value - starts[low] + 1 };
    }

    function lineEnd(index) {
      return index + 1 < starts.length ? starts[index + 1] - 1 : input.value.length;
    }

    function lineText(index) {
      return input.value.slice(starts[index], lineEnd(index));
    }

    function resetCheckpoints() {
      checkpoints = [{ line: 0, state: null }];
    }

    function invalidateFrom(line) {
      while (checkpoints.length > 1 && checkpoints[checkpoints.length - 1].line > line) checkpoints.pop();
    }

    function addCheckpoint(line, state) {
      const last = checkpoints[checkpoints.length - 1];
      if (last && last.line === line) last.state = state;
      else if (!last || last.line < line) checkpoints.push({ line, state });
    }

    function stateAt(line) {
      if (!highlighter || !highlighter.hasGrammar(grammarId)) return null;
      line = clamp(Number(line) || 0, 0, starts.length - 1);
      let low = 0;
      let high = checkpoints.length - 1;
      while (low < high) {
        const middle = (low + high + 1) >> 1;
        if (checkpoints[middle].line <= line) low = middle;
        else high = middle - 1;
      }
      let checkpoint = checkpoints[low];
      let state = checkpoint.state;
      let index = checkpoint.line;
      while (index < line) {
        state = highlighter.scanLine(lineText(index), state, grammarId).state;
        index += 1;
        if (index % CHECKPOINT_LINES === 0) addCheckpoint(index, state);
      }
      return state;
    }

    function applyEdit(previous) {
      const firstDiff = rebuildIndex(previous);
      if (firstDiff >= 0) {
        invalidateFrom(lineOf(firstDiff).index);
        contentVersion += 1;
        gutterDirty = true;
      }
      lastValue = input.value;
    }

    /* -- measurement, gutter and highlighting -------------------------------- */

    function measure() {
      if (measured) return;
      measured = true;
      const view = typeof window.getComputedStyle === "function" ? window.getComputedStyle(input) : null;
      lineHeight = parseFloat(view && view.lineHeight) || 20;
      paddingTop = parseFloat(view && view.paddingTop) || 12;
      paddingLeft = parseFloat(view && view.paddingLeft) || 16;
      if (highlightInner.style) highlightInner.style.lineHeight = `${lineHeight}px`;
      if (highlightInner.style) highlightInner.style.tabSize = String(tabSize);
      if (input.style) input.style.tabSize = String(tabSize);
      charWidth = 0;
    }

    function viewport() {
      return Number(input.clientHeight) || 600;
    }

    function visibleRange() {
      const total = lineCount();
      const scrollTop = Number(input.scrollTop) || 0;
      const overscan = 8;
      const first = Math.min(Math.max(0, total - 1),
        Math.max(0, Math.floor((scrollTop - paddingTop) / lineHeight) - overscan));
      const last = Math.min(total, Math.max(first + 1,
        Math.ceil((scrollTop + viewport() - paddingTop) / lineHeight) + overscan));
      return { first, last, scrollTop };
    }

    function renderGutter(force = false) {
      measure();
      const total = lineCount();
      const { first, last, scrollTop } = visibleRange();
      if (force || gutterDirty || first !== gutterFirst || last !== gutterLast) {
        gutterFirst = first;
        gutterLast = last;
        gutterDirty = false;
        const numbers = new Array(Math.max(0, last - first));
        for (let index = first; index < last; index += 1) numbers[index - first] = String(index + 1);
        gutterInner.textContent = numbers.join("\n");
      }
      gutterInner.style.transform = `translateY(${Math.round(paddingTop + first * lineHeight - scrollTop)}px)`;
    }

    function decorationsForLine(line, length) {
      const ranges = [];
      const found = find.byLine && find.byLine.get(line);
      if (found) ranges.push(...found);
      for (const bracket of brackets) {
        if (lineOf(bracket.start).index !== line) continue;
        const start = bracket.start - starts[line];
        if (start >= 0 && start < length + 1) ranges.push({ start, end: start + 1, type: bracket.type });
      }
      ranges.sort((left, right) => left.start - right.start || left.end - right.end);
      return ranges;
    }

    function renderLine(fragment, text, line, tokens) {
      const decorations = decorationsForLine(line, text.length);
      if (!tokens.length && !decorations.length) {
        if (text) fragment.append(document.createTextNode(text));
        return;
      }
      const boundaries = [0, text.length];
      for (const token of tokens) {
        boundaries.push(token.start, token.end);
      }
      for (const range of decorations) {
        boundaries.push(clamp(range.start, 0, text.length), clamp(range.end, 0, text.length));
      }
      boundaries.sort((left, right) => left - right);
      let tokenIndex = 0;
      let start = boundaries[0];
      for (let index = 1; index < boundaries.length; index += 1) {
        const end = boundaries[index];
        if (end <= start) continue;
        while (tokenIndex < tokens.length && tokens[tokenIndex].end <= start) tokenIndex += 1;
        const token = tokenIndex < tokens.length && tokens[tokenIndex].start <= start && tokens[tokenIndex].end >= end
          ? tokens[tokenIndex] : null;
        const classes = [];
        if (token && token.type) classes.push(`tok-${token.type}`);
        for (const range of decorations) {
          if (range.start <= start && range.end >= end && range.type) classes.push(range.type);
        }
        const chunk = text.slice(start, end);
        if (!chunk) {
          start = end;
          continue;
        }
        if (!classes.length) {
          fragment.append(document.createTextNode(chunk));
        } else {
          const span = document.createElement("span");
          span.className = classes.join(" ");
          span.textContent = chunk;
          fragment.append(span);
        }
        start = end;
      }
    }

    function renderHighlight(force = false) {
      measure();
      // The textarea text is painted by this layer, which keeps find and bracket
      // decorations working even for plain text.  The overlay class is only set
      // after a successful paint, so a tokenizer failure leaves native text visible.
      if (typeof document === "undefined") return;
      const syntax = Boolean(highlighter && highlighter.hasGrammar(grammarId));
      const { first, last, scrollTop } = visibleRange();
      const scrollLeft = Number(input.scrollLeft) || 0;
      highlightInner.style.transform = `translate(${Math.round(paddingLeft - scrollLeft)}px, ${Math.round(paddingTop + first * lineHeight - scrollTop)}px)`;
      if (!force && rendered.first === first && rendered.last === last
        && rendered.content === contentVersion && rendered.decor === decorVersion
        && rendered.language === grammarId) {
        setClass(editor, "overlay", true);
        return;
      }
      rendered = {
        first, last, content: contentVersion, decor: decorVersion, language: grammarId,
        scrollLeft, scrollTop,
      };
      let state = syntax ? stateAt(first) : null;
      const fragment = document.createDocumentFragment();
      try {
        for (let line = first; line < last; line += 1) {
          if (syntax && line > 0 && line % CHECKPOINT_LINES === 0) addCheckpoint(line, state);
          const text = lineText(line);
          const result = syntax
            ? highlighter.scanLine(text, state, grammarId)
            : { tokens: [], state: null };
          state = result.state;
          renderLine(fragment, text, line, result.tokens);
          fragment.append(document.createTextNode("\n"));
        }
      } catch (error) {
        setClass(editor, "overlay", false);
        highlightInner.replaceChildren();
        rendered = { first: -1, last: -1, content: -1, decor: -1, language: "", scrollLeft: -1, scrollTop: -1 };
        if (typeof console !== "undefined" && console.error) console.error("Highlighting failed", error);
        return;
      }
      highlightInner.replaceChildren(fragment);
      setClass(editor, "overlay", true);
    }

    function updateBands() {
      measure();
      const top = paddingTop + currentLine * lineHeight - (Number(input.scrollTop) || 0);
      for (const element of [lineBand, gutterCurrent]) {
        if (element.style) {
          element.style.transform = `translateY(${Math.round(top)}px)`;
          element.style.height = `${lineHeight}px`;
        }
      }
    }

    /* -- cursor, selection and brackets -------------------------------------- */

    function tokenTypeAt(offset) {
      if (!highlighter || !highlighter.hasGrammar(grammarId)) return null;
      const line = lineOf(offset).index;
      const state = stateAt(line);
      const result = highlighter.scanLine(lineText(line), state, grammarId);
      const column = offset - starts[line];
      const token = result.tokens.find(item => item.start <= column && item.end > column);
      return token ? token.type : null;
    }

    function updateBrackets() {
      const key = `${contentVersion}:${grammarId}:${input.selectionStart}:${input.selectionEnd}`;
      if (key === bracketKey) return;
      bracketKey = key;
      const next = [];
      if (input.selectionStart === input.selectionEnd) {
        const value = input.value;
        const caret = input.selectionStart;
        let index = -1;
        let char = "";
        if (caret > 0 && BRACKETS[value[caret - 1]]) {
          index = caret - 1;
          char = value[caret - 1];
        } else if (caret < value.length && BRACKETS[value[caret]]) {
          index = caret;
          char = value[caret];
        }
        if (index >= 0) {
          const ignored = tokenTypeAt(index);
          if (ignored !== "comment" && ignored !== "string" && ignored !== "code") {
            const target = BRACKETS[char];
            const forward = "([{".includes(char);
            const step = forward ? 1 : -1;
            let depth = 0;
            let found = -1;
            for (let cursor = index; cursor >= 0 && cursor < value.length; cursor += step) {
              const character = value[cursor];
              if (character === char) depth += 1;
              else if (character === target) {
                depth -= 1;
                if (depth === 0) {
                  found = cursor;
                  break;
                }
              }
              if (Math.abs(cursor - index) > 200000) break;
            }
            next.push({ start: index, end: index + 1, type: found >= 0 ? "bracket-match" : "bracket-unmatched" });
            if (found >= 0) next.push({ start: found, end: found + 1, type: "bracket-match" });
          }
        }
      }
      const previous = brackets.map(item => `${item.start}:${item.type}`).join("|");
      const current = next.map(item => `${item.start}:${item.type}`).join("|");
      brackets = next;
      if (previous !== current) {
        decorVersion += 1;
        schedulePaint(false);
      }
    }

    function updateSelectionInfo() {
      const start = input.selectionStart;
      const end = input.selectionEnd;
      if (start === end) {
        selectionInfo.textContent = "";
        return;
      }
      const first = lineOf(start).index;
      const last = lineOf(end).index;
      const lines = last - first + 1;
      selectionInfo.textContent = `${end - start} chars${lines > 1 ? `, ${lines} lines` : ""}`;
    }

    function cursor() {
      const offset = Number(input.selectionStart) || 0;
      const line = lineOf(offset);
      currentLine = line.index;
      position.textContent = `Ln ${line.index + 1}, Col ${line.column}`;
      updateSelectionInfo();
      updateBands();
      updateBrackets();
      if (options.onCursor) options.onCursor(line.index + 1, line.column);
    }

    function paint(force) {
      renderGutter(Boolean(force));
      cursor();
      renderHighlight(Boolean(force));
      updateBands();
    }

    function schedulePaint(force) {
      if (force) {
        if (frame !== null && typeof cancelAnimationFrame === "function") cancelAnimationFrame(frame);
        frame = null;
        paint(true);
        return;
      }
      if (frame !== null) return;
      if (typeof requestAnimationFrame !== "function") {
        paint(false);
        return;
      }
      frame = requestAnimationFrame(() => {
        frame = null;
        paint(false);
      });
    }

    /* Re-measure after a font or zoom change; the cached line height, character
       width and virtualised window are all invalid then. */
    function refreshMetrics() {
      measured = false;
      gutterDirty = true;
      rendered.content = -1;
      paint(true);
    }

    function emitInput() {
      applyEdit(lastValue);
      if (find.open) scheduleFindRefresh();
      schedulePaint(false);
      if (!silent && options.onChange) options.onChange(input.value);
    }

    function select(start, end) {
      if (typeof input.setSelectionRange === "function") input.setSelectionRange(start, end);
      else {
        input.selectionStart = start;
        input.selectionEnd = end;
      }
      cursor();
    }

    function insert(text) {
      const start = input.selectionStart;
      const end = input.selectionEnd;
      input.setRangeText(text, start, end, "end");
      emitInput();
    }

    function editRange(start, end, text, selectStart, selectEnd) {
      input.setRangeText(text, start, end, "end");
      if (selectStart !== undefined) {
        const from = clamp(selectStart, 0, input.value.length);
        const to = clamp(selectEnd === undefined ? selectStart : selectEnd, from, input.value.length);
        input.selectionStart = from;
        input.selectionEnd = to;
      }
      emitInput();
    }

    function selectedLineRange() {
      const start = input.selectionStart;
      const end = input.selectionEnd;
      const first = lineOf(start).index;
      let last = lineOf(end).index;
      if (end > start && end === starts[last]) last -= 1;
      return { first, last: Math.max(first, last), start, end };
    }

    function indentSelection(outdent) {
      const start = input.selectionStart;
      const end = input.selectionEnd;
      const value = input.value;
      const lineStart = value.lastIndexOf("\n", start - 1) + 1;
      // A selection that stops at the start of a line covers only the line above;
      // otherwise it covers every line up to the next line break.
      let finish = end;
      if (end > start && value[end - 1] === "\n") {
        finish = end - 1;
      } else {
        const newline = value.indexOf("\n", end);
        finish = newline === -1 ? value.length : newline;
      }
      const block = value.slice(lineStart, finish);
      const changed = block.split("\n").map(line => {
        if (outdent) return line.startsWith(indentUnit) ? line.slice(indentUnit.length) : line.replace(/^[ \t]/, "");
        return indentUnit + line;
      }).join("\n");
      input.setRangeText(changed, lineStart, finish, "select");
      emitInput();
    }

    /* -- editing commands ---------------------------------------------------- */

    function duplicateLine() {
      if (readOnly) return;
      const range = selectedLineRange();
      const blockStart = starts[range.first];
      const blockEnd = lineEnd(range.last);
      const block = input.value.slice(blockStart, blockEnd);
      if (!block && !input.value) return;
      const insertAt = blockEnd;
      const from = blockStart + block.length + 1;
      editRange(insertAt, insertAt, `\n${block}`, from, from + block.length);
    }

    function deleteLine() {
      if (readOnly) return;
      const range = selectedLineRange();
      let start = starts[range.first];
      let end = range.last + 1 < lineCount() ? starts[range.last + 1] : lineEnd(range.last);
      if (start === end && start > 0) start -= 1;
      editRange(start, end, "", start, start);
    }

    function moveLines(direction) {
      if (readOnly) return;
      const range = selectedLineRange();
      if (direction < 0 && range.first === 0) return;
      if (direction > 0 && range.last + 1 >= lineCount()) return;
      const block = input.value.slice(starts[range.first], lineEnd(range.last));
      const other = direction < 0
        ? input.value.slice(starts[range.first - 1], lineEnd(range.first - 1))
        : input.value.slice(starts[range.last + 1], lineEnd(range.last + 1));
      const regionStart = direction < 0 ? starts[range.first - 1] : starts[range.first];
      const regionEnd = direction < 0 ? lineEnd(range.last) : lineEnd(range.last + 1);
      const replacement = direction < 0 ? `${block}\n${other}` : `${other}\n${block}`;
      const shift = direction < 0 ? -(other.length + 1) : other.length + 1;
      editRange(regionStart, regionEnd, replacement, range.start + shift, range.end + shift);
    }

    function toggleComment() {
      if (readOnly) return;
      const range = selectedLineRange();
      const blockStart = starts[range.first];
      const blockEnd = lineEnd(range.last);
      const block = input.value.slice(blockStart, blockEnd);
      const token = highlighter ? highlighter.lineComment(grammarId) : null;
      if (token) {
        const lines = block.split("\n");
        const meaningful = lines.filter(line => line.trim());
        const all = meaningful.length > 0 && meaningful.every(line => line.trimStart().startsWith(token));
        const changed = lines.map(line => {
          if (!line.trim()) return line;
          const indent = (/^[ \t]*/.exec(line) || [""])[0];
          const rest = line.slice(indent.length);
          if (all) {
            let next = rest.startsWith(token) ? rest.slice(token.length) : rest;
            if (next.startsWith(" ")) next = next.slice(1);
            return indent + next;
          }
          return `${indent}${token} ${rest}`;
        }).join("\n");
        const delta = changed.length - block.length;
        const from = range.start === range.end ? clamp(range.start + delta, blockStart, blockStart + changed.length) : blockStart;
        editRange(blockStart, blockEnd, changed, from, from + (range.start === range.end ? 0 : changed.length));
        return;
      }
      const wrap = highlighter ? highlighter.blockComment(grammarId) : null;
      if (!wrap) return;
      const open = wrap.open.trim();
      const close = wrap.close.trim();
      const openAt = block.indexOf(open);
      const closeAt = block.lastIndexOf(close);
      let changed;
      if (openAt >= 0 && closeAt >= openAt + open.length) {
        let inner = block.slice(openAt + open.length, closeAt);
        if (inner.startsWith(" ")) inner = inner.slice(1);
        if (inner.endsWith(" ")) inner = inner.slice(0, -1);
        changed = block.slice(0, openAt) + inner + block.slice(closeAt + close.length);
      } else {
        changed = `${wrap.open}${block}${wrap.close}`;
      }
      editRange(blockStart, blockEnd, changed, blockStart, blockStart + changed.length);
    }

    function transformSelection(kind) {
      if (readOnly) return;
      const start = input.selectionStart;
      const end = input.selectionEnd;
      if (start === end) return;
      const selected = input.value.slice(start, end);
      let changed = selected;
      if (kind === "upper") changed = selected.toUpperCase();
      else if (kind === "lower") changed = selected.toLowerCase();
      else if (kind === "title") changed = selected.replace(/\S+/g, word => word[0].toUpperCase() + word.slice(1).toLowerCase());
      if (changed === selected) return;
      editRange(start, end, changed, start, start + changed.length);
    }

    function selectLine() {
      const range = selectedLineRange();
      input.focus();
      select(starts[range.first], lineEnd(range.last));
      revealRange(starts[range.first]);
    }

    function wrapOrInsertPair(open) {
      const close = PAIRS[open];
      const start = input.selectionStart;
      const end = input.selectionEnd;
      if (start !== end) {
        const selected = input.value.slice(start, end);
        editRange(start, end, `${open}${selected}${close}`, start + 1, start + 1 + selected.length);
        return;
      }
      const next = input.value[start] || "";
      if (next === close) {
        input.selectionStart = start + 1;
        input.selectionEnd = start + 1;
        cursor();
        return;
      }
      const quote = open === '"' || open === "'" || open === "`";
      if (quote && start > 0 && /[\w$]/.test(input.value[start - 1])) {
        insert(open);
        return;
      }
      input.setRangeText(`${open}${close}`, start, start, "end");
      input.selectionStart = start + 1;
      input.selectionEnd = start + 1;
      emitInput();
    }

    function insertLine(before) {
      if (readOnly) return;
      const line = lineOf(input.selectionStart).index;
      const indent = (/^[ \t]*/.exec(lineText(line)) || [""])[0];
      const at = before ? starts[line] : lineEnd(line);
      const text = before ? `${indent}\n` : `\n${indent}`;
      editRange(at, at, text, at + (before ? indent.length : 1 + indent.length), at + (before ? indent.length : 1 + indent.length));
    }

    /* -- find and replace ---------------------------------------------------- */

    function buildFindRanges() {
      find.byLine = new Map();
      if (!find.open || !find.matches.length) return;
      for (let index = 0; index < find.matches.length; index += 1) {
        const match = find.matches[index];
        const first = lineOf(match.start).index;
        const last = lineOf(Math.max(match.start, match.end - 1)).index;
        for (let line = first; line <= last; line += 1) {
          const start = clamp(match.start - starts[line], 0, lineEnd(line) - starts[line]);
          const end = clamp(match.end - starts[line], 0, lineEnd(line) - starts[line]);
          if (end <= start) continue;
          const ranges = find.byLine.get(line) || [];
          ranges.push({ start, end, type: index === find.index ? "find-current" : "find-match" });
          find.byLine.set(line, ranges);
        }
      }
    }

    function updateFindStatus() {
      if (!findCount) return;
      if (find.invalid) findCount.textContent = "Invalid regex";
      else if (!find.query) findCount.textContent = "";
      else if (!find.matches.length) findCount.textContent = "No results";
      else findCount.textContent = `${find.index + 1} of ${find.matches.length}${find.truncated ? "+" : ""}`;
      const disabled = !find.open || readOnly || !find.matches.length;
      if (replaceButton) replaceButton.disabled = disabled;
      if (replaceAllButton) replaceAllButton.disabled = disabled;
    }

    function revealRange(offset) {
      measure();
      const line = lineOf(offset);
      const top = paddingTop + line.index * lineHeight;
      const height = viewport();
      if (top < input.scrollTop + paddingTop || top + lineHeight > input.scrollTop + height - paddingTop) {
        input.scrollTop = Math.max(0, top - height / 2);
      }
      if (typeof document !== "undefined") {
        const width = Number(input.clientWidth) || 800;
        const column = offset - starts[line.index];
        if (!charWidth) {
          const canvas = document.createElement("canvas");
          const context = canvas.getContext && canvas.getContext("2d");
          const view = typeof window.getComputedStyle === "function" ? window.getComputedStyle(input) : null;
          if (context && view) {
            context.font = `${view.fontSize} ${view.fontFamily}`;
            charWidth = (context.measureText("MMMMMMMMMM").width / 10) || 0;
          }
        }
        const char = charWidth || Math.max(6, lineHeight * 0.6);
        const left = paddingLeft + column * char;
        if (left < input.scrollLeft + paddingLeft || left > input.scrollLeft + width - paddingLeft) {
          input.scrollLeft = Math.max(0, left - width / 2);
        }
      }
      schedulePaint(false);
    }

    function scheduleFindRefresh() {
      // Scanning the whole buffer on every keystroke makes typing in a large
      // file stall; the find bar tolerates one refresh per 120 ms.
      clearTimeout(findTimer);
      findTimer = setTimeout(() => {
        findTimer = null;
        if (find.open && findInput) refreshFind({ select: false });
      }, 120);
    }

    function flushFindRefresh() {
      if (findTimer === null) return;
      clearTimeout(findTimer);
      findTimer = null;
      if (find.open && findInput) refreshFind({ select: false });
    }

    function refreshFind(settings = {}) {
      if (!find.open || !findInput) return;
      find.query = findInput.value;
      const result = findMatches(input.value, find.query, {
        caseSensitive: find.caseSensitive, regex: find.regex, limit: MAX_FIND_MATCHES,
      });
      find.matches = result.matches;
      find.invalid = result.invalid;
      find.truncated = result.truncated;
      if (!find.matches.length) {
        find.index = -1;
      } else {
        const caret = input.selectionStart;
        const covering = find.matches.findIndex(match => match.start <= caret && caret <= match.end);
        if (covering >= 0) find.index = covering;
        else {
          const after = find.matches.findIndex(match => match.start >= caret);
          find.index = after >= 0 ? after : 0;
        }
      }
      buildFindRanges();
      updateFindStatus();
      decorVersion += 1;
      if (settings.select !== false && find.index >= 0) {
        const match = find.matches[find.index];
        select(match.start, match.end);
        revealRange(match.start);
      }
      schedulePaint(false);
    }

    function findNext(direction) {
      if (!find.open) {
        openFind(false);
        return;
      }
      flushFindRefresh();
      if (!find.matches.length) {
        refreshFind({ select: false });
        if (!find.matches.length) return;
      }
      if (find.index < 0) find.index = direction > 0 ? 0 : find.matches.length - 1;
      else find.index = (find.index + direction + find.matches.length) % find.matches.length;
      const match = find.matches[find.index];
      buildFindRanges();
      updateFindStatus();
      decorVersion += 1;
      select(match.start, match.end);
      revealRange(match.start);
      schedulePaint(false);
    }

    function replacementFor(match) {
      const replacement = replaceInput ? replaceInput.value : "";
      if (!find.regex) return replacement;
      try {
        return match.text.replace(new RegExp(find.query, find.caseSensitive ? "" : "i"), replacement);
      } catch (error) {
        return replacement;
      }
    }

    function replaceCurrent() {
      if (!find.open || readOnly) return;
      flushFindRefresh();
      if (find.index < 0 || find.index >= find.matches.length) {
        refreshFind({ select: true });
        if (find.index < 0) return;
      }
      const match = find.matches[find.index];
      const replacement = replacementFor(match);
      editRange(match.start, match.end, replacement, match.start + replacement.length, match.start + replacement.length);
      refreshFind({ select: true });
    }

    function replaceAll() {
      if (!find.open || readOnly) return;
      flushFindRefresh();
      if (!find.matches.length) return;
      const value = input.value;
      let cursor = 0;
      let next = "";
      for (const match of find.matches) {
        next += value.slice(cursor, match.start) + replacementFor(match);
        cursor = match.end;
      }
      next += value.slice(cursor);
      if (next === value) return;
      if (typeof document !== "undefined" && document.execCommand) {
        input.focus();
        input.setSelectionRange(0, value.length);
        let inserted = false;
        try {
          inserted = document.execCommand("insertText", false, next);
        } catch (error) {
          inserted = false;
        }
        if (!inserted) input.value = next;
      } else {
        input.value = next;
      }
      input.selectionStart = 0;
      input.selectionEnd = 0;
      emitInput();
      refreshFind({ select: true });
    }

    function ensureFindBar() {
      if (findBar) return;
      findInput = Studio.h("input", {
        class: "code-find-input", type: "text", placeholder: "Find", "aria-label": "Find in file",
        spellcheck: false, autocomplete: "off",
      });
      findCount = Studio.h("span", { class: "code-find-count", "aria-live": "polite" });
      caseButton = Studio.h("button", {
        class: "code-toggle", type: "button", title: "Match case", "aria-label": "Match case",
        "aria-pressed": "false", text: "Aa",
      });
      regexButton = Studio.h("button", {
        class: "code-toggle", type: "button", title: "Use regular expression", "aria-label": "Use regular expression",
        "aria-pressed": "false", text: ".*",
      });
      const previous = Studio.h("button", {
        class: "icon-button small", type: "button", title: "Previous match (Shift+Enter)", "aria-label": "Previous match", text: "↑",
      });
      const next = Studio.h("button", {
        class: "icon-button small", type: "button", title: "Next match (Enter)", "aria-label": "Next match", text: "↓",
      });
      const close = Studio.h("button", {
        class: "icon-button small", type: "button", title: "Close find (Escape)", "aria-label": "Close find", text: "×",
      });
      const row = Studio.h("div", { class: "code-find-row" },
        findInput, findCount, caseButton, regexButton, previous, next, close);
      replaceInput = Studio.h("input", {
        class: "code-find-input", type: "text", placeholder: "Replace", "aria-label": "Replace with", spellcheck: false,
      });
      replaceButton = Studio.h("button", { class: "button small", type: "button", text: "Replace" });
      replaceAllButton = Studio.h("button", { class: "button small", type: "button", text: "Replace all" });
      findReplaceRow = Studio.h("div", { class: "code-find-row hidden" }, replaceInput, replaceButton, replaceAllButton);
      findBar = Studio.h("div", { class: "code-find hidden", role: "search", "aria-label": "Find and replace" },
        row, findReplaceRow);
      editor.append(findBar);

      findInput.addEventListener("input", () => {
        find.index = -1;
        refreshFind({ select: true });
      });
      findInput.addEventListener("keydown", event => {
        if (event.key === "Escape") {
          event.preventDefault();
          closeFind(true);
        } else if (event.key === "Enter") {
          event.preventDefault();
          findNext(event.shiftKey ? -1 : 1);
        }
      });
      replaceInput.addEventListener("keydown", event => {
        if (event.key === "Escape") {
          event.preventDefault();
          closeFind(true);
        } else if (event.key === "Enter") {
          event.preventDefault();
          replaceCurrent();
        }
      });
      previous.addEventListener("click", () => findNext(-1));
      next.addEventListener("click", () => findNext(1));
      close.addEventListener("click", () => closeFind(true));
      caseButton.addEventListener("click", () => {
        find.caseSensitive = !find.caseSensitive;
        setPressed(caseButton, find.caseSensitive);
        find.index = -1;
        refreshFind({ select: true });
      });
      regexButton.addEventListener("click", () => {
        find.regex = !find.regex;
        setPressed(regexButton, find.regex);
        find.index = -1;
        refreshFind({ select: true });
      });
      replaceButton.addEventListener("click", replaceCurrent);
      replaceAllButton.addEventListener("click", replaceAll);
      findBar.addEventListener("keydown", event => {
        if (!(event.ctrlKey || event.metaKey) || event.shiftKey) return;
        const key = typeof event.key === "string" ? event.key.toLowerCase() : "";
        if (key === "f") {
          event.preventDefault();
          openFind(false);
        } else if (key === "h") {
          event.preventDefault();
          openFind(true);
        } else if (key === "g") {
          event.preventDefault();
          goToLinePrompt();
        }
      });
      updateFindAvailability();
    }

    function updateFindAvailability() {
      const disabled = readOnly;
      if (replaceInput) replaceInput.disabled = disabled;
      if (replaceButton) replaceButton.disabled = disabled;
      if (replaceAllButton) replaceAllButton.disabled = disabled;
      updateFindStatus();
    }

    function openFind(replace) {
      ensureFindBar();
      find.open = true;
      find.replace = Boolean(replace);
      setClass(findBar, "hidden", false);
      setClass(findReplaceRow, "hidden", !find.replace);
      const selected = input.value.slice(input.selectionStart, input.selectionEnd);
      if (selected && !selected.includes("\n") && selected.length <= 500) findInput.value = selected;
      find.index = -1;
      updateFindAvailability();
      refreshFind({ select: true });
      findInput.focus();
      if (typeof findInput.select === "function") findInput.select();
    }

    function closeFind(restore) {
      find.open = false;
      clearTimeout(findTimer);
      findTimer = null;
      find.replace = false;
      find.matches = [];
      find.index = -1;
      find.byLine = new Map();
      if (findBar) setClass(findBar, "hidden", true);
      decorVersion += 1;
      schedulePaint(false);
      if (restore) input.focus();
    }

    function goToLinePrompt() {
      if (!Studio.prompt) return;
      const line = lineOf(input.selectionStart).index + 1;
      Promise.resolve(Studio.prompt({
        title: "Go to line",
        label: `Line number (1-${lineCount()})`,
        value: String(line),
        confirmLabel: "Go",
      })).then(value => {
        if (value === null || value === undefined) return;
        const number = parseInt(String(value).trim(), 10);
        if (!Number.isFinite(number) || number < 1) {
          if (Studio.toast) Studio.toast("Enter a line number between 1 and " + lineCount(), "error");
          return;
        }
        api.gotoLine(number);
      });
    }

    /* -- language and indentation menus -------------------------------------- */

    function openLanguageMenu() {
      if (!Studio.overlay || !Studio.overlay.menu || !highlighter) return;
      const current = highlighter.languages.find(language => language.id === grammarId);
      const ordered = current
        ? [current, ...highlighter.languages.filter(language => language.id !== grammarId)]
        : highlighter.languages;
      const items = ordered.map(language => ({
        label: language.label,
        hint: language.id === grammarId ? "current" : null,
        onSelect: () => {
          api.setLanguage(language.label, { notify: true });
        },
      }));
      Studio.overlay.menu(languageButton, items, { label: "Language mode" });
    }

    function setIndent(mode, size, notify, persist = true) {
      indentMode = mode === "tab" ? "tab" : "spaces";
      tabSize = clamp(Number(size) || 4, 1, 8);
      indentUnit = indentMode === "tab" ? "\t" : " ".repeat(tabSize);
      if (input.style) input.style.tabSize = String(tabSize);
      if (highlightInner.style) highlightInner.style.tabSize = String(tabSize);
      indentButton.textContent = `${indentMode === "tab" ? "Tabs" : "Spaces"}: ${tabSize}`;
      setLabel(indentButton, "aria-label", `Indentation: ${indentButton.textContent}`);
      if (persist && store && store.set) store.set("editor-indent", { mode: indentMode, size: tabSize });
      if (notify && options.onIndent) options.onIndent({ mode: indentMode, size: tabSize });
    }

    function openIndentMenu() {
      if (!Studio.overlay || !Studio.overlay.menu) return;
      const items = [2, 4, 8].map(size => ({
        label: `Spaces: ${size}`,
        hint: indentMode === "spaces" && tabSize === size ? "current" : null,
        onSelect: () => setIndent("spaces", size, true),
      }));
      items.push({ label: "Tabs", hint: indentMode === "tab" ? "current" : null, onSelect: () => setIndent("tab", tabSize, true) });
      items.push({
        label: "Detect from file",
        onSelect: () => {
          const detected = detectIndent(input.value);
          if (detected) setIndent(detected.mode, detected.size, true, false);
        },
      });
      Studio.overlay.menu(indentButton, items, { label: "Indentation" });
    }

    languageButton.addEventListener("click", openLanguageMenu);
    indentButton.addEventListener("click", openIndentMenu);

    /* -- context menu -------------------------------------------------------- */

    function nativeCommand(name) {
      input.focus();
      if (typeof document === "undefined" || !document.execCommand) return false;
      try {
        return document.execCommand(name);
      } catch (error) {
        return false;
      }
    }

    function copyText(text) {
      if (typeof navigator !== "undefined" && navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).catch(() => {});
      }
    }

    function copySelection() {
      if (!nativeCommand("copy")) {
        copyText(input.value.slice(input.selectionStart, input.selectionEnd));
      }
    }

    function cutSelection() {
      if (readOnly) return;
      if (!nativeCommand("cut")) {
        const start = input.selectionStart;
        const end = input.selectionEnd;
        copyText(input.value.slice(start, end));
        editRange(start, end, "", start, start);
      }
    }

    function pasteClipboard() {
      if (readOnly) return;
      if (nativeCommand("paste")) return;
      if (typeof navigator !== "undefined" && navigator.clipboard && navigator.clipboard.readText) {
        navigator.clipboard.readText().then(text => {
          if (text) insert(text);
        }).catch(() => {});
      }
    }

    function openContextMenu(event) {
      if (!Studio.overlay || !Studio.overlay.menu) return;
      if (event && event.preventDefault) event.preventDefault();
      const selected = input.selectionStart !== input.selectionEnd;
      const canComment = Boolean(highlighter
        && (highlighter.lineComment(grammarId) || highlighter.blockComment(grammarId)));
      const items = [
        { label: "Save", hint: "Ctrl+S", disabled: !options.onSave, onSelect: () => options.onSave && options.onSave() },
        "separator",
        { label: "Undo", hint: "Ctrl+Z", disabled: readOnly, onSelect: () => nativeCommand("undo") },
        { label: "Redo", hint: "Ctrl+Shift+Z", disabled: readOnly, onSelect: () => nativeCommand("redo") },
        "separator",
        { label: "Cut", hint: "Ctrl+X", disabled: readOnly || !selected, onSelect: cutSelection },
        { label: "Copy", hint: "Ctrl+C", disabled: !selected, onSelect: copySelection },
        { label: "Paste", hint: "Ctrl+V", disabled: readOnly, onSelect: pasteClipboard },
        "separator",
        { label: "Select all", hint: "Ctrl+A", onSelect: () => { input.focus(); input.select(); cursor(); } },
        { label: "Select line", hint: "Ctrl+L", onSelect: selectLine },
        "separator",
        { label: "Find", hint: "Ctrl+F", onSelect: () => openFind(false) },
        { label: "Replace", hint: "Ctrl+H", disabled: readOnly, onSelect: () => openFind(true) },
        { label: "Go to line", hint: "Ctrl+G", onSelect: goToLinePrompt },
        "separator",
        { label: "Duplicate line", hint: "Ctrl+Shift+D", disabled: readOnly, onSelect: duplicateLine },
        { label: "Delete line", hint: "Ctrl+Shift+K", disabled: readOnly, onSelect: deleteLine },
        { label: "Move line up", hint: "Alt+Up", disabled: readOnly || selectedLineRange().first === 0, onSelect: () => moveLines(-1) },
        { label: "Move line down", hint: "Alt+Down", disabled: readOnly || selectedLineRange().last + 1 >= lineCount(), onSelect: () => moveLines(1) },
        { label: "Toggle comment", hint: "Ctrl+/", disabled: readOnly || !canComment, onSelect: toggleComment },
        "separator",
        { label: "Uppercase selection", disabled: readOnly || !selected, onSelect: () => transformSelection("upper") },
        { label: "Lowercase selection", disabled: readOnly || !selected, onSelect: () => transformSelection("lower") },
        { label: "Title case selection", disabled: readOnly || !selected, onSelect: () => transformSelection("title") },
        "separator",
        { label: "Change language mode", onSelect: openLanguageMenu },
        { label: "Change indentation", onSelect: openIndentMenu },
        { label: tabMovesFocus ? "Tab inserts indentation" : "Tab moves focus", hint: "Ctrl+M", onSelect: toggleTabFocus },
      ];
      if (options.path) {
        items.push("separator", { label: "Copy path", onSelect: () => copyText(options.path) });
      }
      let x = event.clientX || 0;
      let y = event.clientY || 0;
      if (!x && !y && input.getBoundingClientRect) {
        const box = input.getBoundingClientRect();
        x = box.left + paddingLeft + 24;
        y = box.top + paddingTop + currentLine * lineHeight - (Number(input.scrollTop) || 0) + lineHeight;
      }
      Studio.overlay.menu({ x, y }, items, { label: "Editor actions" });
    }

    /* -- events -------------------------------------------------------------- */

    input.addEventListener("input", emitInput);
    input.addEventListener("scroll", () => schedulePaint(false));
    input.addEventListener("click", cursor);
    input.addEventListener("keyup", cursor);
    input.addEventListener("select", cursor);
    input.addEventListener("contextmenu", openContextMenu);
    gutter.addEventListener("contextmenu", openContextMenu);
    input.addEventListener("focus", () => setClass(editor, "focused", true));
    input.addEventListener("blur", () => setClass(editor, "focused", false));
    gutter.addEventListener("click", event => {
      measure();
      const box = gutter.getBoundingClientRect ? gutter.getBoundingClientRect() : null;
      if (!box) return;
      const y = (event.clientY || 0) - box.top + (Number(input.scrollTop) || 0) - paddingTop;
      const line = clamp(Math.floor(y / lineHeight), 0, lineCount() - 1);
      input.focus();
      select(starts[line], lineEnd(line));
    });
    // A pane resize changes how many lines fit; repaint the virtualised layers.
    if (typeof window.ResizeObserver === "function") {
      observer = new window.ResizeObserver(() => {
        measured = false;
        schedulePaint(false);
      });
      observer.observe(input);
    }
    if (typeof window.addEventListener === "function") window.addEventListener("resize", onWindowResize);

    input.addEventListener("keydown", event => {
      if (event.isComposing) return;
      const key = typeof event.key === "string" ? event.key : "";
      const lower = key.toLowerCase();
      const mod = event.ctrlKey || event.metaKey;

      if (mod && lower === "s") {
        event.preventDefault();
        event.stopPropagation();
        if (options.onSave) options.onSave();
        return;
      }
      if (event.ctrlKey && !event.metaKey && !event.shiftKey && lower === "m") {
        event.preventDefault();
        toggleTabFocus();
        return;
      }
      if (mod && !event.shiftKey && lower === "f") {
        event.preventDefault();
        openFind(false);
        return;
      }
      if (mod && !event.shiftKey && lower === "h") {
        event.preventDefault();
        openFind(true);
        return;
      }
      if (mod && !event.shiftKey && lower === "g") {
        event.preventDefault();
        goToLinePrompt();
        return;
      }
      if (mod && !event.shiftKey && lower === "l") {
        event.preventDefault();
        selectLine();
        return;
      }
      if (key === "Escape" && find.open) {
        event.preventDefault();
        closeFind(true);
        return;
      }
      if (readOnly) return;

      if (mod && lower === "/") {
        event.preventDefault();
        toggleComment();
        return;
      }
      if (mod && event.shiftKey && lower === "d") {
        event.preventDefault();
        duplicateLine();
        return;
      }
      if (mod && event.shiftKey && lower === "k") {
        event.preventDefault();
        deleteLine();
        return;
      }
      if (mod && !event.shiftKey && key === "[") {
        event.preventDefault();
        indentSelection(true);
        return;
      }
      if (mod && !event.shiftKey && key === "]") {
        event.preventDefault();
        indentSelection(false);
        return;
      }
      if (mod && key === "Enter") {
        event.preventDefault();
        insertLine(Boolean(event.shiftKey));
        return;
      }
      if (event.altKey && key === "ArrowUp") {
        event.preventDefault();
        if (event.shiftKey) duplicateLine(); else moveLines(-1);
        return;
      }
      if (event.altKey && key === "ArrowDown") {
        event.preventDefault();
        if (event.shiftKey) duplicateLine(); else moveLines(1);
        return;
      }
      if (event.ctrlKey || event.metaKey) return;

      if (key === "Backspace" && input.selectionStart === input.selectionEnd && input.selectionStart > 0) {
        const before = input.value[input.selectionStart - 1];
        const after = input.value[input.selectionStart];
        if (PAIRS[before] === after) {
          event.preventDefault();
          const start = input.selectionStart;
          editRange(start - 1, start + 1, "", start - 1, start - 1);
          return;
        }
      }

      if (key === "Tab" && tabMovesFocus) return;
      if (key === "Tab") {
        event.preventDefault();
        if (input.selectionStart !== input.selectionEnd || input.value.slice(input.selectionStart, input.selectionEnd).includes("\n")) {
          indentSelection(event.shiftKey);
        } else if (event.shiftKey) {
          const start = input.selectionStart;
          const lineStart = input.value.lastIndexOf("\n", start - 1) + 1;
          if (input.value.slice(lineStart, lineStart + indentUnit.length) === indentUnit) {
            input.setRangeText("", lineStart, lineStart + indentUnit.length, "end");
            input.selectionStart = Math.max(lineStart, start - indentUnit.length);
            input.selectionEnd = input.selectionStart;
            emitInput();
          }
        } else {
          insert(indentUnit);
        }
        return;
      }

      if (key === "Enter" && !event.shiftKey) {
        const start = input.selectionStart;
        const value = input.value;
        const lineStart = value.lastIndexOf("\n", start - 1) + 1;
        const lineTextValue = value.slice(lineStart, start);
        const indent = (lineTextValue.match(/^[ \t]*/) || [""])[0];
        const previous = value[start - 1] || "";
        const after = value[input.selectionEnd] || "";
        if (start === input.selectionEnd && PAIRS[previous] === after && PAIRS[previous]
          && previous !== '"' && previous !== "'") {
          event.preventDefault();
          const inner = indent + indentUnit;
          const text = `\n${inner}\n${indent}`;
          input.setRangeText(text, start, start, "end");
          input.selectionStart = start + 1 + inner.length;
          input.selectionEnd = input.selectionStart;
          emitInput();
          return;
        }
        const opensBlock = /[:{[<(]\s*$/.test(lineTextValue)
          || (grammarId === "markup" && /<[^/!][^>]*>\s*$/.test(lineTextValue) && !/\/>\s*$/.test(lineTextValue));
        event.preventDefault();
        input.setRangeText(`\n${indent}${opensBlock ? indentUnit : ""}`, start, input.selectionEnd, "end");
        emitInput();
        return;
      }

      if (PAIRS[key] && !event.altKey) {
        event.preventDefault();
        wrapOrInsertPair(key);
        return;
      }
      if (CLOSERS.has(key) && input.selectionStart === input.selectionEnd && input.value[input.selectionStart] === key) {
        event.preventDefault();
        const at = input.selectionStart;
        input.selectionStart = at + 1;
        input.selectionEnd = at + 1;
        cursor();
        return;
      }
      if (CLOSERS.has(key) && input.selectionStart === input.selectionEnd) {
        const start = input.selectionStart;
        const lineStart = input.value.lastIndexOf("\n", start - 1) + 1;
        const prefix = input.value.slice(lineStart, start);
        if (/^[ \t]+$/.test(prefix) && prefix.endsWith(indentUnit)) {
          event.preventDefault();
          const at = start - indentUnit.length;
          input.setRangeText(key, at, start, "end");
          emitInput();
        }
      }
    });

    const api = {
      input,
      element: editor,
      get value() { return input.value; },
      get dirty() { return Boolean(options.onChange && input.value !== api.baseline); },
      get position() { return lineOf(input.selectionStart); },
      baseline: options.value || "",
      attach(target) {
        target.replaceChildren(editor);
        measured = false;
        schedulePaint(true);
      },
      setLanguage(label, settings = {}) {
        languageLabel = String(label === undefined || label === null ? "" : label);
        languageButton.textContent = languageLabel || "Plain text";
        setLabel(languageButton, "aria-label", `Language mode: ${languageLabel || "Plain text"}`);
        const next = highlighter ? highlighter.resolve(languageLabel, options.path) : "text";
        if (!languageLabel && next !== "text" && highlighter.grammars[next]) {
          languageLabel = highlighter.grammars[next].label;
          languageButton.textContent = languageLabel;
          setLabel(languageButton, "aria-label", `Language mode: ${languageLabel}`);
        }
        if (next !== grammarId) {
          grammarId = next;
          resetCheckpoints();
          contentVersion += 1;
          rendered.content = -1;
        }
        if (settings.notify && options.onLanguage) options.onLanguage(languageLabel);
        paint(true);
      },
      setReadOnly(value) {
        readOnly = Boolean(value);
        input.readOnly = readOnly;
        setClass(editor, "readonly", readOnly);
        updateFindAvailability();
        paint(false);
      },
      setIndent(size, mode) {
        setIndent(mode === undefined ? indentMode : mode, size, false);
        paint(false);
      },
      setEol(eol) {
        eolInfo.textContent = eol === "\r\n" ? "CRLF" : "LF";
      },
      setValue(value, settings = {}) {
        const top = settings.keepScroll === false ? 0 : input.scrollTop;
        const left = settings.keepScroll === false ? 0 : input.scrollLeft;
        const start = input.selectionStart;
        const end = input.selectionEnd;
        silent = true;
        api.baseline = String(value === undefined || value === null ? "" : value);
        input.value = api.baseline;
        lastValue = api.baseline;
        silent = false;
        rebuildIndex();
        resetCheckpoints();
        contentVersion += 1;
        gutterDirty = true;
        if (settings.keepSelection !== false) {
          input.selectionStart = Math.min(start, api.baseline.length);
          input.selectionEnd = Math.min(end, api.baseline.length);
        }
        input.scrollTop = top;
        input.scrollLeft = left;
        if (find.open) refreshFind({ select: false });
        paint(true);
      },
      focus() { input.focus(); },
      refreshMetrics,
      openFind(replace) { openFind(Boolean(replace)); },
      goToLineDialog: goToLinePrompt,
      closeFind(restore) { closeFind(restore !== false); },
      findNext(direction) { findNext(direction < 0 ? -1 : 1); },
      selectLine,
      duplicateLine,
      deleteLine,
      moveLines,
      toggleComment,
      gotoLine(line) {
        measure();
        const index = clamp((Number(line) || 1) - 1, 0, lineCount() - 1);
        const offset = starts[index] || 0;
        input.focus();
        select(offset, offset);
        const visible = viewport();
        input.scrollTop = Math.max(0, paddingTop + index * lineHeight - visible / 2);
        input.scrollLeft = 0;
        schedulePaint(true);
      },
      destroy() {
        if (observer) observer.disconnect();
        observer = null;
        clearTimeout(findTimer);
        findTimer = null;
        if (typeof window.removeEventListener === "function") window.removeEventListener("resize", onWindowResize);
        if (frame !== null && typeof cancelAnimationFrame === "function") cancelAnimationFrame(frame);
        frame = null;
        if (findBar && findBar.remove) findBar.remove();
        findBar = null;
        if (typeof editor.remove === "function") editor.remove();
      },
    };

    setIndent(indentMode, tabSize, false, false);
    api.setValue(options.value || "");
    api.setLanguage(options.language || "");
    if (options.readOnly) api.setReadOnly(true);
    return api;
  }

  Studio.editor = { create, detectEol, encodeEol, detectIndent, findMatches };
})();
