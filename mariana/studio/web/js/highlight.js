"use strict";

/* Studio.highlight: dependency-free, stateful syntax highlighting for the IDE.
   The tokenizer works one line at a time and returns the lexer state for the
   next line, so the editor can store checkpoints and only scan what is visible.
   It produces token ranges only; editor.js turns them into DOM nodes.  No
   grammar definitions contain code, and nothing here touches the network. */
(function () {
  const Studio = window.Studio = window.Studio || {};

  function words(text) {
    return new Set(String(text || "").split(/\s+/).filter(Boolean));
  }

  function sticky(source) {
    return new RegExp(source, "y");
  }

  const IDENTIFIER = sticky("[A-Za-z_$\\u00c0-\\uffff][\\w$\\u00c0-\\uffff]*");
  const NUMBER = sticky("(?:0[xX][0-9a-fA-F_]+|0[bB][01_]+|0[oO][0-7_]+|\\d[\\d_]*(?:\\.[\\d_]+)?(?:[eE][+-]?\\d+)?)(?:[uUlLfFdDnN]|_?[A-Za-z]\\w*)?");
  const OPERATOR = /[+\-*/%=<>!&|^~?:@]/;

  const STRINGS = [
    { open: '"', close: '"', escape: "\\", multiline: false },
    { open: "'", close: "'", escape: "\\", multiline: false },
  ];

  function strings(list) {
    return (list || STRINGS).slice().sort((left, right) => right.open.length - left.open.length);
  }

  function makeGrammar(config) {
    const lower = values => config.caseInsensitive
      ? new Set([...values].map(value => value.toLowerCase()))
      : values;
    return {
      id: config.id,
      label: config.label || config.id,
      lineComments: config.lineComments || [],
      lineComment: config.lineComment === undefined ? (config.lineComments || [])[0] || null : config.lineComment,
      blockComments: config.blockComments || [],
      strings: strings(config.strings),
      keywords: lower(words(config.keywords)),
      types: lower(words(config.types)),
      constants: lower(words(config.constants)),
      builtins: lower(words(config.builtins)),
      identifier: config.identifier || IDENTIFIER,
      number: config.number || NUMBER,
      variable: config.variable ? sticky(config.variable) : null,
      annotation: config.annotation ? sticky(config.annotation) : null,
      symbols: Boolean(config.symbols),
      preprocessor: Boolean(config.preprocessor),
      attributes: Boolean(config.attributes),
      lifetimes: Boolean(config.lifetimes),
      propertyKeys: Boolean(config.propertyKeys),
      assignmentKeys: Boolean(config.assignmentKeys),
      functionCalls: config.functionCalls !== false,
      dottedProperties: config.dottedProperties !== false,
      hashNeedsSpace: Boolean(config.hashNeedsSpace),
      caseInsensitive: Boolean(config.caseInsensitive),
      capitalized: config.capitalized || null,
      scan: config.scan || null,
      lineCommentWrap: config.lineCommentWrap || null,
    };
  }

  function findStringEnd(text, from, spec) {
    let index = from;
    while (index < text.length) {
      if (spec.doubleClose && text[index] === spec.close && text[index + 1] === spec.close) {
        index += 2;
        continue;
      }
      if (spec.escape && text[index] === spec.escape) {
        index += 2;
        continue;
      }
      if (text.startsWith(spec.close, index)) return index + spec.close.length;
      index += 1;
    }
    return -1;
  }

  function stringAt(text, index, grammar) {
    for (const spec of grammar.strings) {
      if (text.startsWith(spec.open, index)) return spec;
    }
    return null;
  }

  function blockAt(text, index, grammar) {
    for (const pair of grammar.blockComments) {
      if (text.startsWith(pair[0], index)) return { open: pair[0], close: pair[1] };
    }
    return null;
  }

  function lineCommentAt(text, index, grammar) {
    for (const marker of grammar.lineComments) {
      if (!text.startsWith(marker, index)) continue;
      if (marker === "#" && grammar.hashNeedsSpace && index > 0 && !/\s/.test(text[index - 1])) continue;
      return marker;
    }
    return null;
  }

  function previousWord(text, index) {
    let cursor = index - 1;
    while (cursor >= 0 && /\s/.test(text[cursor])) cursor -= 1;
    const end = cursor + 1;
    while (cursor >= 0 && /[\w$]/.test(text[cursor])) cursor -= 1;
    return text.slice(cursor + 1, end);
  }

  function nextSignificant(text, index) {
    let cursor = index;
    while (cursor < text.length && /\s/.test(text[cursor])) cursor += 1;
    return { char: text[cursor] || "", index: cursor };
  }

  function pushToken(tokens, start, end, type) {
    if (end > start) tokens.push({ start, end, type });
  }

  /* The generic scanner covers every language whose grammar is comments,
     strings, numbers, identifiers and a keyword list. */
  function scanGeneric(text, state, grammar) {
    const tokens = [];
    let index = 0;
    const length = text.length;

    if (state && state.kind === "comment") {
      const end = text.indexOf(state.close);
      if (end < 0) {
        pushToken(tokens, 0, length, "comment");
        return { tokens, state };
      }
      pushToken(tokens, 0, end + state.close.length, "comment");
      index = end + state.close.length;
      state = null;
    } else if (state && state.kind === "string") {
      const end = findStringEnd(text, 0, state);
      if (end < 0) {
        pushToken(tokens, 0, length, "string");
        return { tokens, state };
      }
      pushToken(tokens, 0, end, "string");
      index = end;
      state = null;
    } else if (state) {
      state = null;
    }

    if (grammar.preprocessor) {
      while (index < length && /\s/.test(text[index])) index += 1;
      if (text[index] === "#") {
        const match = /^#\s*[A-Za-z_]\w*/.exec(text.slice(index));
        if (match) {
          pushToken(tokens, index, index + match[0].length, "meta");
          index += match[0].length;
          const header = /^\s*([<"][^>"\n]*[>"])/.exec(text.slice(index));
          if (header) {
            const start = index + header[0].indexOf(header[1]);
            pushToken(tokens, start, start + header[1].length, "string");
            index = start + header[1].length;
          }
        }
      }
    }

    while (index < length) {
      const char = text[index];

      if (/\s/.test(char)) {
        index += 1;
        continue;
      }

      if (grammar.attributes && char === "#" && (text[index + 1] === "[" || (text[index + 1] === "!" && text[index + 2] === "["))) {
        const end = text.indexOf("]", index);
        if (end < 0) {
          pushToken(tokens, index, length, "meta");
          return { tokens, state: null };
        }
        pushToken(tokens, index, end + 1, "meta");
        index = end + 1;
        continue;
      }

      const block = blockAt(text, index, grammar);
      if (block) {
        const end = text.indexOf(block.close, index + block.open.length);
        if (end < 0) {
          pushToken(tokens, index, length, "comment");
          return { tokens, state: { kind: "comment", close: block.close } };
        }
        pushToken(tokens, index, end + block.close.length, "comment");
        index = end + block.close.length;
        continue;
      }

      const line = lineCommentAt(text, index, grammar);
      if (line) {
        pushToken(tokens, index, length, "comment");
        break;
      }

      if (grammar.annotation) {
        grammar.annotation.lastIndex = index;
        const match = grammar.annotation.exec(text);
        if (match) {
          pushToken(tokens, index, index + match[0].length, "meta");
          index += match[0].length;
          continue;
        }
      }

      if (grammar.variable) {
        grammar.variable.lastIndex = index;
        const match = grammar.variable.exec(text);
        if (match) {
          pushToken(tokens, index, index + match[0].length, "variable");
          index += match[0].length;
          continue;
        }
      }

      if (grammar.lifetimes && char === "'" && /[A-Za-z_]/.test(text[index + 1] || "")) {
        const match = /'[A-Za-z_]\w*/y;
        match.lastIndex = index;
        const lifetime = match.exec(text);
        if (lifetime && text[lifetime.index + lifetime[0].length] !== "'") {
          pushToken(tokens, index, index + lifetime[0].length, "type");
          index += lifetime[0].length;
          continue;
        }
      }

      const spec = stringAt(text, index, grammar);
      if (spec) {
        const start = index;
        const end = findStringEnd(text, index + spec.open.length, spec);
        if (end < 0) {
          pushToken(tokens, start, length, "string");
          return {
            tokens,
            state: spec.multiline
              ? { kind: "string", close: spec.close, escape: spec.escape, doubleClose: spec.doubleClose }
              : null,
          };
        }
        const next = nextSignificant(text, end);
        const key = (grammar.propertyKeys && next.char === ":")
          || (grammar.assignmentKeys && next.char === "=" && text[next.index + 1] !== "=");
        pushToken(tokens, start, end, key ? "property" : "string");
        index = end;
        continue;
      }

      if (grammar.symbols && char === ":" && /[A-Za-z_]/.test(text[index + 1] || "") && (index === 0 || !/[\w:]/.test(text[index - 1]))) {
        const match = /:[A-Za-z_]\w*[?!=]?/y;
        match.lastIndex = index;
        const symbol = match.exec(text);
        if (symbol) {
          pushToken(tokens, index, index + symbol[0].length, "constant");
          index += symbol[0].length;
          continue;
        }
      }

      if (/\d/.test(char) || (char === "." && /\d/.test(text[index + 1] || ""))) {
        grammar.number.lastIndex = index;
        const number = grammar.number.exec(text);
        if (number) {
          pushToken(tokens, index, index + number[0].length, "number");
          index += number[0].length;
          continue;
        }
      }

      grammar.identifier.lastIndex = index;
      const identifier = grammar.identifier.exec(text);
      if (identifier) {
        const word = identifier[0];
        const end = index + word.length;
        const key = grammar.caseInsensitive ? word.toLowerCase() : word;
        let type = null;
        if (grammar.keywords.has(key)) type = "keyword";
        else if (grammar.types.has(key)) type = "type";
        else if (grammar.constants.has(key)) type = "constant";
        else if (grammar.builtins.has(key)) type = "builtin";
        if (!type) {
          const next = nextSignificant(text, end);
          const previous = text[index - 1];
          if (grammar.functionCalls && next.char === "(") type = "function";
          else if (grammar.dottedProperties && previous === ".") type = "property";
          else if ((grammar.propertyKeys && next.char === ":")
            || (grammar.assignmentKeys && next.char === "=" && text[next.index + 1] !== "=")) type = "property";
          else if (grammar.capitalized === "type" && /^[A-Z]/.test(word)) type = "type";
          else if (grammar.capitalized === "variable" && /^[A-Z_]/.test(word)) type = "variable";
        }
        if (!type && grammar.functionCalls) {
          const before = previousWord(text, index);
          if (["def", "function", "fn", "func", "fun", "proc", "sub", "method"].includes(before)) type = "function";
        }
        pushToken(tokens, index, end, type);
        index = end;
        continue;
      }

      if (OPERATOR.test(char)) {
        let end = index + 1;
        while (end < length && OPERATOR.test(text[end])) end += 1;
        pushToken(tokens, index, end, "operator");
        index = end;
        continue;
      }

      index += 1;
    }

    return { tokens, state: null };
  }

  /* -- markup --------------------------------------------------------------- */

  const RAW_TAGS = new Set(["script", "style", "textarea", "title"]);

  function scanTextEntities(text, from, to, tokens) {
    const pattern = /&(?:#\d+|#x[0-9a-fA-F]+|[A-Za-z][A-Za-z0-9]+);/g;
    pattern.lastIndex = from;
    let match;
    while ((match = pattern.exec(text)) && match.index < to) {
      const end = match.index + match[0].length;
      if (end > to) break;
      pushToken(tokens, match.index, end, "constant");
    }
  }

  function parseTag(text, start, state, tokens) {
    let index = start;
    let name = state ? state.name || "" : "";
    let closing = state ? Boolean(state.closing) : false;
    let rawTag = state ? state.rawTag || "" : "";
    let quote = state ? state.quote || "" : "";
    let expectValue = state ? Boolean(state.expectValue) : false;
    const resuming = Boolean(state);
    const length = text.length;

    if (!resuming) {
      closing = text.startsWith("</", index);
      pushToken(tokens, index, index + (closing ? 2 : 1), "punctuation");
      index += closing ? 2 : 1;
      const nameMatch = /^[A-Za-z][\w:.-]*/.exec(text.slice(index));
      if (!nameMatch) return { index: index + 1, state: null, raw: null };
      name = nameMatch[0];
      pushToken(tokens, index, index + name.length, "tag");
      index += name.length;
      rawTag = !closing && RAW_TAGS.has(name.toLowerCase()) ? name.toLowerCase() : "";
    }

    while (index < length) {
      const char = text[index];
      if (quote) {
        const end = findStringEnd(text, index, { close: quote, escape: "\\" });
        if (end < 0) {
          pushToken(tokens, index, length, "string");
          return { index: length, state: { kind: "tag", name, closing, rawTag, quote, expectValue } };
        }
        pushToken(tokens, index, end, "string");
        index = end;
        quote = "";
        expectValue = false;
        continue;
      }
      if (char === ">") {
        pushToken(tokens, index, index + 1, "punctuation");
        return { index: index + 1, state: null, raw: rawTag || null };
      }
      if (char === "/" && text[index + 1] === ">") {
        pushToken(tokens, index, index + 2, "punctuation");
        return { index: index + 2, state: null, raw: null };
      }
      if (/\s/.test(char)) {
        index += 1;
        continue;
      }
      if (char === '"' || char === "'") {
        const end = findStringEnd(text, index + 1, { close: char, escape: "\\" });
        if (end < 0) {
          pushToken(tokens, index, length, "string");
          return { index: length, state: { kind: "tag", name, closing, rawTag, quote: char, expectValue: false } };
        }
        pushToken(tokens, index, end, "string");
        index = end;
        expectValue = false;
        continue;
      }
      if (char === "=") {
        pushToken(tokens, index, index + 1, "operator");
        expectValue = true;
        index += 1;
        continue;
      }
      if (expectValue) {
        const value = /^[^\s>]+/.exec(text.slice(index));
        if (value) {
          pushToken(tokens, index, index + value[0].length, "string");
          index += value[0].length;
          expectValue = false;
          continue;
        }
      }
      const attribute = /^[^\s=/>]+/.exec(text.slice(index));
      if (attribute) {
        pushToken(tokens, index, index + attribute[0].length, "attr");
        index += attribute[0].length;
        continue;
      }
      if (char === "/" || char === "?") {
        index += 1;
        continue;
      }
      index += 1;
    }

    return { index: length, state: { kind: "tag", name, closing, rawTag, quote, expectValue } };
  }

  function scanMarkupLine(text, state, grammar) {
    const tokens = [];
    let index = 0;
    const length = text.length;
    state = state || null;

    while (index < length) {
      if (state && state.kind === "comment") {
        const close = state.close || "-->";
        const end = text.indexOf(close, index);
        if (end < 0) {
          pushToken(tokens, index, length, "comment");
          return { tokens, state };
        }
        pushToken(tokens, index, end + close.length, "comment");
        index = end + close.length;
        state = null;
        continue;
      }

      if (state && state.kind === "raw") {
        const close = `</${state.tag}`;
        const at = text.toLowerCase().indexOf(close, index);
        const end = at < 0 ? length : at;
        if (end > index) {
          const chunk = text.slice(index, end);
          if (state.grammar) {
            const scan = state.grammar.scan || scanGeneric;
            const inner = scan(chunk, state.lex, state.grammar);
            for (const token of inner.tokens) {
              tokens.push({ start: token.start + index, end: token.end + index, type: token.type });
            }
            state = { ...state, lex: inner.state };
          }
        }
        if (at < 0) return { tokens, state };
        index = at;
        state = null;
        continue;
      }

      if (state && state.kind === "tag") {
        const parsed = parseTag(text, index, state, tokens);
        index = parsed.index;
        state = parsed.state;
        if (parsed.raw) {
          state = {
            kind: "raw",
            tag: parsed.raw,
            grammar: parsed.raw === "script" ? GRAMMARS.javascript : parsed.raw === "style" ? GRAMMARS.css : null,
            lex: null,
          };
        }
        continue;
      }

      const open = text.indexOf("<", index);
      if (open < 0) {
        scanTextEntities(text, index, length, tokens);
        break;
      }
      if (open > index) scanTextEntities(text, index, open, tokens);
      index = open;

      if (text.startsWith("<!--", index)) {
        const end = text.indexOf("-->", index + 4);
        if (end < 0) {
          pushToken(tokens, index, length, "comment");
          return { tokens, state: { kind: "comment" } };
        }
        pushToken(tokens, index, end + 3, "comment");
        index = end + 3;
        continue;
      }
      if (text.startsWith("<![CDATA[", index)) {
        const end = text.indexOf("]]>", index + 9);
        if (end < 0) {
          pushToken(tokens, index, length, "string");
          return { tokens, state: { kind: "comment", close: "]]>" } };
        }
        pushToken(tokens, index, end + 3, "string");
        index = end + 3;
        continue;
      }
      if (text.startsWith("<!", index) || text.startsWith("<?", index)) {
        const close = text.startsWith("<?", index) ? "?>" : ">";
        const end = text.indexOf(close, index + 2);
        if (end < 0) {
          pushToken(tokens, index, length, "meta");
          return { tokens, state: null };
        }
        pushToken(tokens, index, end + close.length, "meta");
        index = end + close.length;
        continue;
      }
      if (/[A-Za-z/]/.test(text[index + 1] || "")) {
        const parsed = parseTag(text, index, null, tokens);
        index = parsed.index;
        state = parsed.state;
        if (parsed.raw) {
          state = {
            kind: "raw",
            tag: parsed.raw,
            grammar: parsed.raw === "script" ? GRAMMARS.javascript : parsed.raw === "style" ? GRAMMARS.css : null,
            lex: null,
          };
        }
        continue;
      }
      index += 1;
    }

    return { tokens, state };
  }

  /* -- CSS ------------------------------------------------------------------ */

  const CSS_AT_RULES = words("charset import namespace media supports document page font-face keyframes viewport");
  const CSS_HEX = /#[0-9a-fA-F]{3,8}\b/y;
  const CSS_UNIT = /(?:0[xX][0-9a-fA-F_]+|\d[\d_]*(?:\.[\d_]+)?(?:[eE][+-]?\d+)?)(?:%|[a-zA-Z]{1,6})?/y;

  function scanCssLine(text, state, grammar) {
    const tokens = [];
    let index = 0;
    const length = text.length;
    let mode = state ? state.mode || "selector" : "selector";
    let depth = state ? state.depth || 0 : 0;

    if (state && state.kind === "comment") {
      const end = text.indexOf("*/");
      if (end < 0) {
        pushToken(tokens, 0, length, "comment");
        return { tokens, state };
      }
      pushToken(tokens, 0, end + 2, "comment");
      index = end + 2;
    }

    while (index < length) {
      const char = text[index];
      if (/\s/.test(char)) {
        index += 1;
        continue;
      }
      if (char === "/" && text[index + 1] === "*") {
        const end = text.indexOf("*/", index + 2);
        if (end < 0) {
          pushToken(tokens, index, length, "comment");
          return { tokens, state: { kind: "comment", mode, depth } };
        }
        pushToken(tokens, index, end + 2, "comment");
        index = end + 2;
        continue;
      }
      if (grammar.lineComments.length && char === "/" && text[index + 1] === "/" && text[index - 1] !== ":") {
        pushToken(tokens, index, length, "comment");
        break;
      }
      if (char === '"' || char === "'") {
        const close = char;
        const end = findStringEnd(text, index + 1, { close, escape: "\\" });
        if (end < 0) {
          pushToken(tokens, index, length, "string");
          break;
        }
        pushToken(tokens, index, end, "string");
        index = end;
        continue;
      }
      if (char === "@" && /[A-Za-z-]/.test(text[index + 1] || "")) {
        const match = /^@[A-Za-z-]+/y.exec(text.slice(index));
        if (match) {
          const name = match[0].slice(1);
          const next = nextSignificant(text, index + match[0].length);
          const variable = grammar.id === "scss" && next.char === ":";
          pushToken(tokens, index, index + match[0].length, variable ? "variable" : "keyword");
          index += match[0].length;
          continue;
        }
      }
      if (char === "$" && /[A-Za-z_]/.test(text[index + 1] || "")) {
        const match = /^\$[A-Za-z_][\w-]*/.exec(text.slice(index));
        pushToken(tokens, index, index + match[0].length, "variable");
        index += match[0].length;
        continue;
      }
      if (char === "!" && /^!important\b/i.test(text.slice(index))) {
        pushToken(tokens, index, index + 10, "keyword");
        index += 10;
        continue;
      }
      if (char === "#") {
        CSS_HEX.lastIndex = index;
        const hex = CSS_HEX.exec(text);
        if (hex) {
          pushToken(tokens, index, index + hex[0].length, "constant");
          index += hex[0].length;
          continue;
        }
        const id = /^#[A-Za-z_-][\w-]*/.exec(text.slice(index));
        if (id) {
          pushToken(tokens, index, index + id[0].length, mode === "selector" ? "constant" : "property");
          index += id[0].length;
          continue;
        }
      }
      if (/\d/.test(char) || (char === "." && /\d/.test(text[index + 1] || ""))) {
        CSS_UNIT.lastIndex = index;
        const number = CSS_UNIT.exec(text);
        if (number) {
          pushToken(tokens, index, index + number[0].length, "number");
          index += number[0].length;
          continue;
        }
      }
      if (char === "{") {
        pushToken(tokens, index, index + 1, "punctuation");
        depth += 1;
        mode = "block";
        index += 1;
        continue;
      }
      if (char === "}") {
        pushToken(tokens, index, index + 1, "punctuation");
        depth = Math.max(0, depth - 1);
        if (depth === 0) mode = "selector";
        index += 1;
        continue;
      }
      if (char === ":" || char === ";" || char === ",") {
        index += 1;
        continue;
      }
      const identifier = /^[A-Za-z_-][\w-]*/.exec(text.slice(index));
      if (identifier) {
        const word = identifier[0];
        const end = index + word.length;
        const next = nextSignificant(text, end);
        let type = null;
        if (mode === "selector") {
          if (text[index - 1] === ".") type = "attr";
          else if (text[index - 1] === ":") type = "builtin";
          else if (next.char === "(") type = "function";
          else type = "type";
        } else if (next.char === "(") {
          type = "function";
        } else if (next.char === ":") {
          type = "property";
        } else if (word.startsWith("--")) {
          type = "property";
        } else if (CSS_AT_RULES.has(word.toLowerCase())) {
          type = "keyword";
        }
        pushToken(tokens, index, end, type);
        index = end;
        continue;
      }
      if (/[>+~=*]/.test(char)) {
        pushToken(tokens, index, index + 1, "operator");
      }
      index += 1;
    }

    return { tokens, state: mode === "block" || depth > 0 ? { mode, depth } : null };
  }

  /* -- Markdown ------------------------------------------------------------- */

  function scanInline(text, from, to, tokens) {
    let index = from;
    while (index < to) {
      const char = text[index];
      if (char === "\\") {
        index += 2;
        continue;
      }
      if (char === "`") {
        let run = 1;
        while (text[index + run] === "`") run += 1;
        const fence = "`".repeat(run);
        const close = text.indexOf(fence, index + run);
        if (close >= 0 && close + run <= to) {
          pushToken(tokens, index, close + run, "code");
          index = close + run;
          continue;
        }
        index += run;
        continue;
      }
      if (char === "[" || (char === "!" && text[index + 1] === "[")) {
        const open = char === "!" ? index + 1 : index;
        const labelEnd = text.indexOf("]", open + 1);
        if (labelEnd >= 0 && text[labelEnd + 1] === "(") {
          const urlEnd = text.indexOf(")", labelEnd + 2);
          if (urlEnd >= 0 && urlEnd < to) {
            pushToken(tokens, index, open + 1, "punctuation");
            pushToken(tokens, open + 1, labelEnd, "link");
            pushToken(tokens, labelEnd, urlEnd + 1, "string");
            index = urlEnd + 1;
            continue;
          }
        }
        index += 1;
        continue;
      }
      if (char === "<") {
        const end = text.indexOf(">", index + 1);
        if (end >= 0 && end < to) {
          const body = text.slice(index + 1, end);
          pushToken(tokens, index, end + 1, /^(?:https?:|mailto:|www\.)/i.test(body) ? "link" : "tag");
          index = end + 1;
          continue;
        }
      }
      if (char === "*" || char === "_" || char === "~") {
        const run = text[index + 1] === char ? 2 : 1;
        const marker = char.repeat(run);
        const start = index + marker.length;
        const close = text.indexOf(marker, start);
        if (close >= 0 && close < to) {
          const type = run === 2 ? (char === "~" ? "emphasis" : "strong") : "emphasis";
          pushToken(tokens, index, close + marker.length, type);
          index = close + marker.length;
          continue;
        }
      }
      index += 1;
    }
  }

  function scanMarkdownLine(text, state) {
    const tokens = [];
    if (state && state.kind === "fence") {
      const trimmed = text.trim();
      if (trimmed.startsWith(state.marker)) {
        pushToken(tokens, 0, text.length, "meta");
        return { tokens, state: null };
      }
      const grammar = state.language && GRAMMARS[state.language];
      if (grammar && grammar.id !== "text") {
        const scan = grammar.scan || scanGeneric;
        const inner = scan(text, state.lex, grammar);
        return { tokens: inner.tokens, state: { ...state, lex: inner.state } };
      }
      pushToken(tokens, 0, text.length, "code");
      return { tokens, state };
    }

    const fence = /^\s{0,3}(`{3,}|~{3,})(.*)$/.exec(text);
    if (fence) {
      pushToken(tokens, 0, text.length, "meta");
      const language = fence[2].trim().split(/\s+/)[0];
      const id = language ? resolve(language) : "text";
      return {
        tokens,
        state: { kind: "fence", marker: fence[1][0].repeat(Math.max(3, fence[1].length)), language: id, lex: null },
      };
    }
    if (/^\s{0,3}#{1,6}\s/.test(text)) {
      pushToken(tokens, 0, text.length, "heading");
      return { tokens, state: null };
    }
    if (/^\s{0,3}(?:(?:\*\s*){3,}|(?:-\s*){3,}|(?:_\s*){3,})$/.test(text)) {
      pushToken(tokens, 0, text.length, "meta");
      return { tokens, state: null };
    }
    const quote = /^(\s{0,3}>+\s?)(.*)$/.exec(text);
    if (quote) {
      pushToken(tokens, 0, quote[1].length, "meta");
      scanInline(text, quote[1].length, text.length, tokens);
      return { tokens, state: null };
    }
    const list = /^(\s*)([-*+]|\d+[.)])\s+/.exec(text);
    if (list) {
      pushToken(tokens, 0, list[0].length, "meta");
      scanInline(text, list[0].length, text.length, tokens);
      return { tokens, state: null };
    }
    if (text.includes("|")) {
      let start = 0;
      for (let index = 0; index < text.length; index += 1) {
        if (text[index] !== "|") continue;
        if (index > start) scanInline(text, start, index, tokens);
        pushToken(tokens, index, index + 1, "meta");
        start = index + 1;
      }
      if (start < text.length) scanInline(text, start, text.length, tokens);
      tokens.sort((left, right) => left.start - right.start);
      return { tokens, state: null };
    }
    scanInline(text, 0, text.length, tokens);
    return { tokens, state: null };
  }

  /* -- YAML and data formats ------------------------------------------------ */

  function scanQuotedValue(text, index, tokens, close, type = "string") {
    const end = findStringEnd(text, index + 1, { close, escape: "\\" });
    if (end < 0) {
      pushToken(tokens, index, text.length, type);
      return text.length;
    }
    pushToken(tokens, index, end, type);
    return end;
  }

  function scanYamlValue(text, index, tokens) {
    while (index < text.length) {
      const char = text[index];
      if (/\s/.test(char)) {
        index += 1;
        continue;
      }
      if (char === "#" && (index === 0 || /\s/.test(text[index - 1]))) {
        pushToken(tokens, index, text.length, "comment");
        return text.length;
      }
      if (char === '"' || char === "'") {
        index = scanQuotedValue(text, index, tokens, char);
        continue;
      }
      if (char === "&" || char === "*" || char === "!") {
        const match = /^[&*!][\w.:-]*/.exec(text.slice(index));
        pushToken(tokens, index, index + match[0].length, "meta");
        index += match[0].length;
        continue;
      }
      if (char === "[" || char === "]" || char === "{" || char === "}") {
        pushToken(tokens, index, index + 1, "punctuation");
        index += 1;
        continue;
      }
      const word = /^[^\s,#\[\]{}]+/.exec(text.slice(index));
      if (word) {
        const value = word[0];
        const type = /^(?:true|false|null|~|yes|no|on|off)$/i.test(value) ? "constant"
          : /^[+-]?(?:\d[\d_]*(?:\.\d+)?|\.\d+)$/.test(value) ? "number" : null;
        pushToken(tokens, index, index + value.length, type);
        index += value.length;
        continue;
      }
      index += 1;
    }
    return index;
  }

  function scanYamlLine(text, state) {
    const tokens = [];
    if (state && state.kind === "block") {
      const indent = text.length - text.trimStart().length;
      if (!text.trim() || indent > state.indent) {
        pushToken(tokens, 0, text.length, "string");
        return { tokens, state };
      }
      state = null;
    }
    const trimmed = text.trimStart();
    const indent = text.length - trimmed.length;
    if (!trimmed) return { tokens, state: null };
    if (trimmed.startsWith("#")) {
      pushToken(tokens, indent, text.length, "comment");
      return { tokens, state: null };
    }
    if (/^(?:---|\.\.\.)\s*$/.test(trimmed)) {
      pushToken(tokens, indent, text.length, "meta");
      return { tokens, state: null };
    }
    let start = indent;
    const list = /^-\s+/.exec(trimmed);
    if (list) {
      pushToken(tokens, start, start + list[0].length, "meta");
      start += list[0].length;
    }
    const key = /^([^#\n][^:\n]*?):(?=\s|$)/.exec(text.slice(start));
    if (key) {
      pushToken(tokens, start, start + key[1].length, "property");
      start += key[0].length;
      const block = /^\s*[|>][+-]?\s*(?:#.*)?$/.exec(text.slice(start));
      if (block) {
        pushToken(tokens, start, text.length, "meta");
        return { tokens, state: { kind: "block", indent } };
      }
    }
    scanYamlValue(text, start, tokens);
    return { tokens, state: null };
  }

  function scanIniLine(text, state, grammar) {
    const trimmed = text.trimStart();
    const indent = text.length - trimmed.length;
    const tokens = [];
    const section = /^\[[^\]\n]*\]/.exec(trimmed);
    if (section) {
      pushToken(tokens, indent, indent + section[0].length, "type");
      return { tokens, state };
    }
    return scanGeneric(text, state, grammar);
  }

  function scanTomlLine(text, state, grammar) {
    const trimmed = text.trimStart();
    const indent = text.length - trimmed.length;
    const tokens = [];
    const table = /^\[\[?[^\]\n]*\]\]?/.exec(trimmed);
    if (table) {
      pushToken(tokens, indent, indent + table[0].length, "type");
      return { tokens, state };
    }
    return scanGeneric(text, state, grammar);
  }

  function scanDiffLine(text) {
    const tokens = [];
    let type = null;
    if (/^(?:diff |index |new file|deleted file|similarity |rename )/.test(text)) type = "meta";
    else if (/^(?:---|\+\+\+)\s/.test(text)) type = "meta";
    else if (text.startsWith("@@")) type = "meta";
    else if (text.startsWith("+")) type = "added";
    else if (text.startsWith("-")) type = "removed";
    else if (text.startsWith("\\")) type = "meta";
    pushToken(tokens, 0, text.length, type);
    return { tokens, state: null };
  }

  function scanCobolLine(text, state, grammar) {
    const trimmed = text.trimStart();
    const indent = text.length - trimmed.length;
    const tokens = [];
    if (text[6] === "*" || text[6] === "/" || trimmed.startsWith("*>")) {
      pushToken(tokens, indent, text.length, "comment");
      return { tokens, state: null };
    }
    return scanGeneric(text, state, grammar);
  }

  function scanBatchLine(text, state, grammar) {
    const trimmed = text.trimStart();
    const indent = text.length - trimmed.length;
    const tokens = [];
    if (/^(?:@?rem\b|::)/i.test(trimmed)) {
      pushToken(tokens, indent, text.length, "comment");
      return { tokens, state: null };
    }
    const label = /^:[A-Za-z_][\w-]*/.exec(trimmed);
    if (label) {
      pushToken(tokens, indent, indent + label[0].length, "meta");
      return { tokens, state: null };
    }
    return scanGeneric(text, state, grammar);
  }

  /* -- grammar table -------------------------------------------------------- */

  const GRAMMARS = {};

  function define(config) {
    GRAMMARS[config.id] = makeGrammar(config);
  }

  const C_COMMENTS = { lineComments: ["//"], blockComments: [["/*", "*/"]] };

  define({
    id: "javascript", label: "JavaScript", ...C_COMMENTS,
    strings: [
      { open: "`", close: "`", escape: "\\", multiline: true },
      { open: '"', close: '"', escape: "\\", multiline: false },
      { open: "'", close: "'", escape: "\\", multiline: false },
    ],
    keywords: "as async await break case catch class const continue debugger default delete do else export extends finally for from function get if implements import in instanceof interface let new of package private protected public return set static super switch this throw try typeof var void while with yield",
    types: "Array BigInt Boolean Date Error Function Map Number Object Promise Proxy RegExp Set String Symbol WeakMap WeakSet JSON Math",
    constants: "true false null undefined NaN Infinity",
    builtins: "console document window globalThis process require module exports",
    propertyKeys: true, capitalized: "type",
  });

  define({
    id: "typescript", label: "TypeScript", ...C_COMMENTS,
    strings: [
      { open: "`", close: "`", escape: "\\", multiline: true },
      { open: '"', close: '"', escape: "\\", multiline: false },
      { open: "'", close: "'", escape: "\\", multiline: false },
    ],
    keywords: "abstract as asserts async await break case catch class const continue declare default delete do else enum export extends finally for from function get if implements import in infer instanceof interface is keyof let namespace never new of override package private protected public readonly return satisfies set static super switch this throw try type typeof undefined unique unknown var void while with yield",
    types: "any bigint boolean Date Error Function Map Number Object Promise Record Set string symbol unknown void Array BigInt JSON Math",
    constants: "true false null undefined NaN Infinity",
    builtins: "console document window globalThis process require module exports",
    propertyKeys: true, capitalized: "type",
  });

  define({
    id: "json", label: "JSON",
    strings: [{ open: '"', close: '"', escape: "\\", multiline: false }],
    constants: "true false null", propertyKeys: true, functionCalls: false, dottedProperties: false,
  });

  define({
    id: "jsonc", label: "JSON with Comments", ...C_COMMENTS,
    strings: [{ open: '"', close: '"', escape: "\\", multiline: false }],
    constants: "true false null", propertyKeys: true, functionCalls: false, dottedProperties: false,
  });

  define({
    id: "python", label: "Python",
    lineComments: ["#"],
    strings: [
      { open: '"""', close: '"""', escape: "\\", multiline: true },
      { open: "'''", close: "'''", escape: "\\", multiline: true },
      { open: '"', close: '"', escape: "\\", multiline: false },
      { open: "'", close: "'", escape: "\\", multiline: false },
    ],
    keywords: "False None True and as assert async await break class continue def del elif else except finally for from global if import in is lambda nonlocal not or pass raise return try while with yield match case",
    types: "bool bytearray bytes complex dict float frozenset int list object set str tuple",
    constants: "True False None Ellipsis NotImplemented",
    builtins: "abs all any ascii bin breakpoint callable chr classmethod compile dir divmod enumerate eval exec filter format getattr globals hasattr hash help hex id input isinstance issubclass iter len locals map max memoryview min next oct open ord pow print property range repr reversed round setattr slice sorted staticmethod sum super type vars zip __import__",
    annotation: "@[A-Za-z_][\\w.]*", propertyKeys: true, capitalized: "type",
  });

  define({
    id: "ruby", label: "Ruby",
    lineComments: ["#"],
    strings: [
      { open: '"', close: '"', escape: "\\", multiline: false },
      { open: "'", close: "'", escape: "\\", multiline: false },
    ],
    keywords: "alias and begin break case class def defined? do else elsif end ensure for if in module next nil not or redo rescue retry return self super then true undef unless until when while yield",
    constants: "true false nil self",
    builtins: "puts print p require require_relative attr_accessor attr_reader attr_writer include extend",
    variable: "[@$][A-Za-z_][\\w]*", symbols: true, capitalized: "type",
  });

  define({
    id: "php", label: "PHP",
    lineComments: ["//", "#"],
    blockComments: [["/*", "*/"]],
    strings: [
      { open: '"', close: '"', escape: "\\", multiline: false },
      { open: "'", close: "'", escape: "\\", multiline: false },
    ],
    keywords: "abstract and array as break callable case catch class clone const continue declare default do echo else elseif empty enddeclare endfor endforeach endif endswitch endwhile eval exit extends final finally fn for foreach function global goto if implements include include_once instanceof insteadof interface isset list match namespace new or print private protected public readonly require require_once return static switch throw trait try unset use var while xor yield",
    types: "array bool int float string object mixed void never iterable",
    constants: "true false null TRUE FALSE NULL",
    variable: "\\$[A-Za-z_][\\w]*", capitalized: "type",
  });

  define({
    id: "shell", label: "Shell",
    lineComments: ["#"],
    strings: [
      { open: '"', close: '"', escape: "\\", multiline: false },
      { open: "'", close: "'", escape: "", multiline: false },
      { open: "`", close: "`", escape: "\\", multiline: false },
    ],
    keywords: "if then else elif fi for while until do done case esac function in select time coproc",
    builtins: "echo printf read cd pwd export unset source alias set shift test true false exit return local declare typeset command exec eval trap wait",
    variable: "\\$(?:\\([^)]*\\)|\\{[^}]*\\}|[A-Za-z_][\\w]*|[0-9@*#?$!-])",
    hashNeedsSpace: true,
  });

  define({
    id: "powershell", label: "PowerShell",
    lineComments: ["#"], blockComments: [["<#", "#>"]],
    strings: [
      { open: '"', close: '"', escape: "`", multiline: false },
      { open: "'", close: "'", escape: "", doubleClose: true, multiline: false },
    ],
    keywords: "begin break catch class continue data define do dynamicparam else elseif end exit filter finally for foreach from function if in param process return switch throw trap try until using var while workflow",
    builtins: "Write-Host Write-Output Get-Item Get-ChildItem Set-Item Remove-Item Where-Object ForEach-Object Select-Object Import-Module",
    variable: "\\$(?:[A-Za-z_][\\w]*|\\{[^}]*\\})", caseInsensitive: true,
  });

  define({
    id: "batch", label: "Batch", scan: scanBatchLine,
    lineComments: ["::"],
    keywords: "echo set if else for in do call goto exit rem",
    variable: "%[^%\\s]+%",
  });

  define({
    id: "perl", label: "Perl",
    lineComments: ["#"],
    strings: [
      { open: '"', close: '"', escape: "\\", multiline: false },
      { open: "'", close: "'", escape: "\\", multiline: false },
    ],
    keywords: "my our local sub if elsif else unless while for foreach continue last next redo return use require package BEGIN END and or not eq ne lt gt le ge cmp",
    builtins: "print printf say chomp chop split join push pop shift unshift keys values each sort map grep",
    variable: "[$@%](?:[A-Za-z_][\\w]*|[0-9!@#$?])", capitalized: "type",
  });

  define({
    id: "lua", label: "Lua",
    lineComments: ["--"], blockComments: [["--[[", "]]"]],
    strings: [
      { open: "[[", close: "]]", escape: "", multiline: true },
      { open: '"', close: '"', escape: "\\", multiline: false },
      { open: "'", close: "'", escape: "\\", multiline: false },
    ],
    keywords: "and break do else elseif end false for function goto if in local nil not or repeat return then true until while",
    builtins: "print pairs ipairs require type tostring tonumber setmetatable getmetatable pcall error assert next rawget rawset select",
    constants: "true false nil",
  });

  define({
    id: "r", label: "R", lineComments: ["#"],
    strings: [
      { open: '"', close: '"', escape: "\\", multiline: false },
      { open: "'", close: "'", escape: "\\", multiline: false },
    ],
    keywords: "if else repeat while function for in next break",
    constants: "TRUE FALSE NULL Inf NaN NA NA_integer_ NA_real_ NA_character_",
    builtins: "library require c list data.frame matrix vector mean median sum sd print cat",
  });

  define({
    id: "julia", label: "Julia",
    lineComments: ["#"], blockComments: [["#=", "=#"]],
    strings: [
      { open: '"""', close: '"""', escape: "\\", multiline: true },
      { open: '"', close: '"', escape: "\\", multiline: false },
      { open: "'", close: "'", escape: "\\", multiline: false },
    ],
    keywords: "function end if else elseif for while begin let local global const return break continue using import export module baremodule struct mutable primitive abstract type where do try catch finally quote macro",
    constants: "true false nothing missing",
    builtins: "println print push! pop! map filter reduce length size typeof",
  });

  define({
    id: "sql", label: "SQL",
    lineComments: ["--"], blockComments: [["/*", "*/"]],
    strings: [{ open: "'", close: "'", escape: "\\", doubleClose: true, multiline: false }],
    keywords: "SELECT FROM WHERE JOIN LEFT RIGHT INNER OUTER FULL ON GROUP BY ORDER HAVING LIMIT OFFSET INSERT INTO VALUES UPDATE SET DELETE CREATE TABLE ALTER DROP INDEX VIEW AS DISTINCT UNION ALL AND OR NOT NULL IS IN LIKE BETWEEN EXISTS CASE WHEN THEN ELSE END PRIMARY KEY FOREIGN REFERENCES DEFAULT UNIQUE CHECK CASCADE WITH RETURNING BEGIN COMMIT ROLLBACK TRANSACTION GRANT REVOKE TRUNCATE EXPLAIN",
    types: "INT INTEGER BIGINT SMALLINT TINYINT DECIMAL NUMERIC FLOAT REAL DOUBLE PRECISION VARCHAR CHAR TEXT DATE TIME TIMESTAMP BOOLEAN JSON UUID BLOB",
    constants: "TRUE FALSE NULL",
    caseInsensitive: true, capitalized: "type", functionCalls: false,
  });

  define({
    id: "c", label: "C", ...C_COMMENTS, preprocessor: true,
    strings: [
      { open: '"', close: '"', escape: "\\", multiline: false },
      { open: "'", close: "'", escape: "\\", multiline: false },
    ],
    keywords: "auto break case const continue default do else enum extern for goto if inline register restrict return sizeof static struct switch typedef union volatile while _Bool _Atomic",
    types: "char double float int long short signed unsigned void size_t ssize_t int8_t int16_t int32_t int64_t uint8_t uint16_t uint32_t uint64_t bool",
    constants: "NULL true false",
    capitalized: "type",
  });

  define({
    id: "cpp", label: "C++", ...C_COMMENTS, preprocessor: true,
    strings: [
      { open: '"', close: '"', escape: "\\", multiline: false },
      { open: "'", close: "'", escape: "\\", multiline: false },
    ],
    keywords: "alignas alignof and asm auto bitand bitor bool break case catch char class compl concept const consteval constexpr constinit const_cast continue co_await co_return co_yield decltype default delete do else enum explicit export extern false for friend goto if inline mutable namespace new noexcept not nullptr operator or private protected public register reinterpret_cast requires return sizeof static static_assert static_cast struct switch template this thread_local throw true try typedef typeid typename union using virtual void volatile while xor",
    types: "int long short signed unsigned float double char size_t ssize_t string vector map set unique_ptr shared_ptr optional variant",
    constants: "nullptr true false NULL",
    capitalized: "type",
  });

  define({
    id: "objc", label: "Objective-C", ...C_COMMENTS, preprocessor: true,
    strings: [
      { open: '"', close: '"', escape: "\\", multiline: false },
      { open: "'", close: "'", escape: "\\", multiline: false },
    ],
    keywords: "auto break case const continue default do else enum extern for goto if inline register restrict return sizeof static struct switch typedef union volatile while @interface @implementation @end @property @synthesize @protocol @class @selector @try @catch @finally @throw @autoreleasepool @synchronized",
    types: "char double float int long short signed unsigned void size_t id BOOL NSInteger NSString NSArray NSDictionary",
    constants: "nil Nil YES NO NULL",
    annotation: "@[A-Za-z_][\\w]*", capitalized: "type",
  });

  define({
    id: "csharp", label: "C#", ...C_COMMENTS,
    strings: [
      { open: '"""', close: '"""', escape: "\\", multiline: true },
      { open: '"', close: '"', escape: "\\", multiline: false },
      { open: "'", close: "'", escape: "\\", multiline: false },
    ],
    keywords: "abstract as async await base break case catch checked class const continue default delegate do else enum event explicit extern finally fixed for foreach get goto if implicit in init interface internal is lock namespace new operator out override params partial private protected public readonly record ref return sealed set sizeof stackalloc static struct switch this throw try typeof unchecked unsafe using var virtual void volatile when where while yield",
    types: "bool byte char decimal double dynamic float int long object sbyte short string uint ulong ushort nint nuint void Task List Dictionary IEnumerable",
    constants: "true false null",
    annotation: "@[A-Za-z_][\\w.]*", capitalized: "type",
  });

  define({
    id: "java", label: "Java", ...C_COMMENTS,
    strings: [
      { open: '"""', close: '"""', escape: "\\", multiline: true },
      { open: '"', close: '"', escape: "\\", multiline: false },
      { open: "'", close: "'", escape: "\\", multiline: false },
    ],
    keywords: "abstract assert break case catch class const continue default do else enum extends final finally for goto if implements import instanceof interface native new package private protected public return sealed static strictfp super switch synchronized this throw throws transient try var void volatile while yield",
    types: "boolean byte char double float int long short String Object List Map Set Optional",
    constants: "true false null",
    annotation: "@[A-Za-z_][\\w.]*", capitalized: "type",
  });

  define({
    id: "kotlin", label: "Kotlin", ...C_COMMENTS,
    strings: [
      { open: '"""', close: '"""', escape: "", multiline: true },
      { open: '"', close: '"', escape: "\\", multiline: false },
      { open: "'", close: "'", escape: "\\", multiline: false },
    ],
    keywords: "as break class continue do else false for fun if in interface is null object package return super this throw true try typealias typeof val var when while by catch constructor delegate dynamic field file finally get import init param property receiver set setparam where actual abstract annotation companion const crossinline data enum expect external final infix inline inner internal lateinit noinline open operator out override private protected public reified sealed suspend tailrec vararg",
    types: "Any Boolean Byte Char Double Float Int Long Nothing Short String Unit List Map Set",
    constants: "true false null",
    annotation: "@[A-Za-z_][\\w.]*", capitalized: "type",
  });

  define({
    id: "scala", label: "Scala", ...C_COMMENTS,
    strings: [
      { open: '"""', close: '"""', escape: "", multiline: true },
      { open: '"', close: '"', escape: "\\", multiline: false },
      { open: "'", close: "'", escape: "\\", multiline: false },
    ],
    keywords: "abstract case catch class def do else extends false final finally for forSome if implicit import lazy match new null object override package private protected return sealed super this throw trait true try type val var while with yield given using enum export then",
    types: "Any AnyRef Boolean Byte Char Double Float Int Long Nothing Short String Unit List Map Set Option Either",
    constants: "true false null",
    annotation: "@[A-Za-z_][\\w.]*", capitalized: "type",
  });

  define({
    id: "groovy", label: "Groovy", ...C_COMMENTS,
    strings: [
      { open: '"""', close: '"""', escape: "\\", multiline: true },
      { open: '"', close: '"', escape: "\\", multiline: false },
      { open: "'", close: "'", escape: "\\", multiline: false },
    ],
    keywords: "as assert break case catch class const continue def default do else enum extends finally for goto if implements import in instanceof interface new package return static super switch this throw throws trait try while",
    types: "boolean byte char double float int long short String Object List Map Set",
    constants: "true false null",
    annotation: "@[A-Za-z_][\\w.]*", capitalized: "type",
  });

  define({
    id: "go", label: "Go", ...C_COMMENTS,
    strings: [
      { open: "`", close: "`", escape: "", multiline: true },
      { open: '"', close: '"', escape: "\\", multiline: false },
      { open: "'", close: "'", escape: "\\", multiline: false },
    ],
    keywords: "break case chan const continue default defer else fallthrough for func go goto if import interface map package range return select struct switch type var",
    types: "bool byte complex64 complex128 error float32 float64 int int8 int16 int32 int64 rune string uint uint8 uint16 uint32 uint64 uintptr any",
    constants: "true false nil iota",
  });

  define({
    id: "rust", label: "Rust", ...C_COMMENTS, attributes: true, lifetimes: true,
    strings: [
      { open: '"', close: '"', escape: "\\", multiline: false },
      { open: "'", close: "'", escape: "\\", multiline: false },
    ],
    keywords: "as async await break const continue crate dyn else enum extern false fn for if impl in let loop match mod move mut pub ref return self Self static struct super trait true type unsafe use where while yield",
    types: "bool char f32 f64 i8 i16 i32 i64 i128 isize str u8 u16 u32 u64 u128 usize String Vec Option Result Box Rc Arc HashMap",
    constants: "true false None Some Ok Err",
    capitalized: "type",
  });

  define({
    id: "swift", label: "Swift", ...C_COMMENTS,
    strings: [
      { open: '"""', close: '"""', escape: "\\", multiline: true },
      { open: '"', close: '"', escape: "\\", multiline: false },
    ],
    keywords: "associatedtype class deinit enum extension fileprivate func import init inout internal let open operator private protocol public rethrows static struct subscript typealias var break case continue default defer do else fallthrough for guard if in repeat return switch where while as Any catch false is nil super self Self throw throws true try",
    types: "Bool Int Double Float String Array Dictionary Set Optional Result",
    constants: "true false nil",
    annotation: "@[A-Za-z_][\\w.]*", capitalized: "type",
  });

  define({
    id: "dart", label: "Dart", ...C_COMMENTS,
    strings: [
      { open: '"""', close: '"""', escape: "\\", multiline: true },
      { open: "'''", close: "'''", escape: "\\", multiline: true },
      { open: '"', close: '"', escape: "\\", multiline: false },
      { open: "'", close: "'", escape: "\\", multiline: false },
    ],
    keywords: "abstract as assert async await break case catch class const continue covariant default deferred do dynamic else enum export extends extension external factory false final finally for Function get hide if implements import in interface is late library mixin new null on operator part required rethrow return set show static super switch sync this throw true try typedef var void while with yield",
    types: "bool int double num String List Map Set Future Stream Object Iterable",
    constants: "true false null",
    annotation: "@[A-Za-z_][\\w.]*", capitalized: "type",
  });

  define({
    id: "zig", label: "Zig", ...C_COMMENTS,
    strings: [
      { open: '"', close: '"', escape: "\\", multiline: false },
      { open: "'", close: "'", escape: "\\", multiline: false },
    ],
    keywords: "const var fn pub use test comptime inline export extern if else switch while for break continue return defer errdefer try catch orelse and or null undefined true false struct enum union error async await nosuspend suspend resume",
    types: "i8 i16 i32 i64 i128 u8 u16 u32 u64 u128 usize isize f32 f64 bool void anyerror",
    constants: "null undefined true false",
    capitalized: "type",
  });

  define({
    id: "nim", label: "Nim",
    lineComments: ["#"], blockComments: [["#[", "]#"]],
    strings: [
      { open: '"""', close: '"""', escape: "\\", multiline: true },
      { open: '"', close: '"', escape: "\\", multiline: false },
      { open: "'", close: "'", escape: "\\", multiline: false },
    ],
    keywords: "proc func method iterator converter template macro type object enum tuple ref ptr var let const if elif else case of when for while block break continue return yield import from export as discard defer try except finally raise echo and or not xor is isnot in notin",
    types: "int int8 int16 int32 int64 uint uint8 uint16 uint32 uint64 float float32 float64 bool char string seq array",
    constants: "true false nil",
    capitalized: "type",
  });

  define({
    id: "d", label: "D", ...C_COMMENTS,
    strings: [
      { open: '"', close: '"', escape: "\\", multiline: false },
      { open: "'", close: "'", escape: "\\", multiline: false },
    ],
    keywords: "abstract alias align asm assert auto body break case cast catch class const continue debug default delegate delete deprecated do else enum export extern final finally for foreach foreach_reverse function goto if immutable import in inout interface invariant is lazy macro mixin module new nothrow null out override package pragma private protected public pure ref return scope shared static struct super switch synchronized template this throw try typeid typeof union unittest version while with",
    types: "bool byte char double float int long short string void size_t",
    constants: "true false null",
    capitalized: "type",
  });

  define({
    id: "haskell", label: "Haskell",
    lineComments: ["--"], blockComments: [["{-", "-}"]],
    strings: [
      { open: '"', close: '"', escape: "\\", multiline: false },
      { open: "'", close: "'", escape: "\\", multiline: false },
    ],
    keywords: "case class data default deriving do else if in infix infixl infixr instance let module newtype of then type where",
    types: "Bool Char Double Either Float Int Integer Maybe IO List String",
    constants: "True False Nothing Just",
    capitalized: "type",
  });

  define({
    id: "elixir", label: "Elixir", lineComments: ["#"],
    strings: [
      { open: '"""', close: '"""', escape: "\\", multiline: true },
      { open: '"', close: '"', escape: "\\", multiline: false },
    ],
    keywords: "def defp defmodule defmacro defstruct defguard do end fn case cond if else unless for with when raise rescue try catch after receive import require use alias",
    constants: "true false nil",
    symbols: true, capitalized: "type",
  });

  define({
    id: "erlang", label: "Erlang", lineComments: ["%"],
    strings: [
      { open: '"', close: '"', escape: "\\", multiline: false },
      { open: "'", close: "'", escape: "\\", multiline: false },
    ],
    keywords: "after begin case catch cond end fun if let of query receive try when",
    constants: "true false",
    capitalized: "variable",
  });

  define({
    id: "clojure", label: "Clojure", lineComments: [";"],
    strings: [{ open: '"', close: '"', escape: "\\", multiline: false }],
    keywords: "def defn defmacro fn let if do loop recur ns require import try catch throw when cond and or not",
    builtins: "println print map filter reduce conj assoc dissoc get update range",
    constants: "true false nil",
  });

  define({
    id: "elm", label: "Elm", lineComments: ["--"], blockComments: [["{-", "-}"]],
    strings: [
      { open: '"""', close: '"""', escape: "\\", multiline: true },
      { open: '"', close: '"', escape: "\\", multiline: false },
    ],
    keywords: "alias as case else exposing if import in let module of port then type where",
    constants: "True False Nothing Just",
    capitalized: "type",
  });

  define({
    id: "ocaml", label: "OCaml", blockComments: [["(*", "*)"]],
    strings: [
      { open: '"', close: '"', escape: "\\", multiline: false },
      { open: "'", close: "'", escape: "\\", multiline: false },
    ],
    keywords: "and as assert begin class constraint do done downto else end exception external false for fun function functor if in include inherit initializer lazy let match method module mutable new object of open or private rec sig struct then to true try type val virtual when while with",
    types: "bool char float int int32 int64 list option string unit array",
    constants: "true false None Some",
    capitalized: "type",
  });

  define({
    id: "fsharp", label: "F#", ...C_COMMENTS, blockComments: [["(*", "*)"], ["/*", "*/"]],
    strings: [
      { open: '"""', close: '"""', escape: "", multiline: true },
      { open: '"', close: '"', escape: "\\", multiline: false },
      { open: "'", close: "'", escape: "\\", multiline: false },
    ],
    keywords: "abstract and as assert base begin class default delegate do done downcast downto elif else end exception extern false finally for fun function global if in inherit inline interface internal lazy let match member module mutable namespace new not null of open or override private public rec return sig static struct then to true try type upcast use val void when while with yield",
    types: "bool byte char decimal double float int int64 list obj option string unit array Map Set",
    constants: "true false null None Some",
    capitalized: "type",
  });

  define({
    id: "protobuf", label: "Protocol Buffers", ...C_COMMENTS,
    strings: [
      { open: '"', close: '"', escape: "\\", multiline: false },
      { open: "'", close: "'", escape: "\\", multiline: false },
    ],
    keywords: "syntax package import option message enum service rpc returns repeated optional required oneof map reserved extend extensions group",
    types: "double float int32 int64 uint32 uint64 sint32 sint64 fixed32 fixed64 sfixed32 sfixed64 bool string bytes",
    constants: "true false",
    capitalized: "type",
  });

  define({
    id: "graphql", label: "GraphQL", lineComments: ["#"],
    strings: [
      { open: '"""', close: '"""', escape: "", multiline: true },
      { open: '"', close: '"', escape: "\\", multiline: false },
    ],
    keywords: "query mutation subscription fragment on type interface union enum input scalar schema extend directive implements repeatable",
    constants: "true false null",
    capitalized: "type",
  });

  define({
    id: "terraform", label: "Terraform", lineComments: ["#", "//"], blockComments: [["/*", "*/"]],
    strings: [
      { open: '"', close: '"', escape: "\\", multiline: false },
      { open: "<<EOF", close: "EOF", escape: "", multiline: true },
    ],
    keywords: "resource data variable output module provider terraform locals for_each count depends_on lifecycle dynamic",
    constants: "true false null",
    variable: "\\$\\{[^}]*\\}", capitalized: "type",
  });

  define({
    id: "cmake", label: "CMake", lineComments: ["#"],
    strings: [{ open: '"', close: '"', escape: "\\", multiline: false }],
    keywords: "cmake_minimum_required project add_executable add_library target_link_libraries target_include_directories find_package include_directories set unset if elseif else endif foreach endforeach function endfunction macro endmacro message option install",
    constants: "ON OFF TRUE FALSE YES NO",
    variable: "\\$\\{[^}]*\\}", caseInsensitive: true,
  });

  define({
    id: "makefile", label: "Makefile", lineComments: ["#"],
    strings: [{ open: '"', close: '"', escape: "\\", multiline: false }],
    keywords: "include -include sinclude ifeq ifneq ifdef ifndef else endif define endef override export unexport vpath",
    variable: "\\$(?:\\([^)]*\\)|\\{[^}]*\\})",
  });

  define({
    id: "dockerfile", label: "Dockerfile", lineComments: ["#"],
    strings: [{ open: '"', close: '"', escape: "\\", multiline: false }],
    keywords: "FROM RUN CMD LABEL MAINTAINER EXPOSE ENV ADD COPY ENTRYPOINT VOLUME USER WORKDIR ARG ONBUILD STOPSIGNAL HEALTHCHECK SHELL AS",
    variable: "\\$(?:[A-Za-z_][\\w]*|\\{[^}]*\\})", caseInsensitive: true,
  });

  define({
    id: "markup", label: "HTML", scan: scanMarkupLine,
  });

  define({
    id: "css", label: "CSS", scan: scanCssLine, blockComments: [["/*", "*/"]],
  });

  define({
    id: "scss", label: "SCSS", scan: scanCssLine, lineComments: ["//"], blockComments: [["/*", "*/"]],
  });

  define({
    id: "mermaid", label: "Mermaid", lineComments: ["%%"],
    strings: [{ open: '"', close: '"', escape: "\\", multiline: false }],
    keywords: "graph flowchart sequenceDiagram pie participant actor note over of left right loop alt else opt par and rect end subgraph classDef class click style linkStyle direction autonumber activate deactivate",
    constants: "TD TB BT LR RL true false",
    symbols: true, propertyKeys: false,
  });

  define({
    id: "markdown", label: "Markdown", scan: scanMarkdownLine,
  });

  define({
    id: "yaml", label: "YAML", scan: scanYamlLine, lineComments: ["#"], hashNeedsSpace: true,
  });

  define({
    id: "toml", label: "TOML", scan: scanTomlLine, lineComments: ["#"],
    strings: [
      { open: '"""', close: '"""', escape: "\\", multiline: true },
      { open: "'''", close: "'''", escape: "", multiline: true },
      { open: '"', close: '"', escape: "\\", multiline: false },
      { open: "'", close: "'", escape: "", multiline: false },
    ],
    constants: "true false inf nan", assignmentKeys: true, functionCalls: false,
  });

  define({
    id: "ini", label: "Config", scan: scanIniLine,
    lineComments: ["#", ";"],
    strings: [
      { open: '"', close: '"', escape: "\\", multiline: false },
      { open: "'", close: "'", escape: "\\", multiline: false },
    ],
    constants: "true false yes no on off null", assignmentKeys: true, functionCalls: false,
  });

  define({
    id: "diff", label: "Diff", scan: scanDiffLine,
  });

  define({
    id: "asm", label: "Assembly", lineComments: [";", "//"],
    strings: [
      { open: '"', close: '"', escape: "\\", multiline: false },
      { open: "'", close: "'", escape: "\\", multiline: false },
    ],
    keywords: "mov add sub mul div inc dec and or xor not shl shr jmp je jne jz jnz call ret push pop cmp test lea int syscall section global extern db dw dd dq equ times",
    constants: "eax ebx ecx edx esi edi esp ebp rax rbx rcx rdx rsi rdi rsp rbp r8 r9 r10 r11 r12 r13 r14 r15 ax bx cx dx ah al",
    caseInsensitive: true, capitalized: "type",
  });

  define({
    id: "fortran", label: "Fortran", lineComments: ["!"],
    strings: [{ open: '"', close: '"', escape: "", multiline: false }, { open: "'", close: "'", escape: "", multiline: false }],
    keywords: "program end module subroutine function if then else elseif endif do while for call return contains implicit none use only integer real complex logical character dimension allocatable pointer parameter intent in out inout public private interface type",
    constants: "true false", caseInsensitive: true, capitalized: "type",
  });

  define({
    id: "matlab", label: "MATLAB", lineComments: ["%"],
    strings: [{ open: '"', close: '"', escape: "", multiline: false }, { open: "'", close: "'", escape: "", multiline: false }],
    keywords: "break case catch classdef continue else elseif end for function global if otherwise parfor persistent return switch try while",
    constants: "true false NaN Inf pi", capitalized: "type",
  });

  define({
    id: "solidity", label: "Solidity", ...C_COMMENTS,
    strings: [
      { open: '"', close: '"', escape: "\\", multiline: false },
      { open: "'", close: "'", escape: "\\", multiline: false },
    ],
    keywords: "abstract address anonymous as assembly bool break calldata case catch constant constructor continue contract delete do else emit enum error event external fallback for function if immutable import indexed interface internal is library mapping memory modifier new override payable private public pure receive return returns revert storage struct super switch this throw try type unchecked unicode using view virtual while",
    types: "address bool bytes bytes32 int int8 int16 int32 int64 int128 int256 string uint uint8 uint16 uint32 uint64 uint128 uint256",
    constants: "true false", capitalized: "type",
  });

  define({
    id: "verilog", label: "Verilog", ...C_COMMENTS,
    strings: [{ open: '"', close: '"', escape: "\\", multiline: false }],
    keywords: "always and assign begin case default else end endcase endfunction endmodule for function if initial inout input integer module or output parameter posedge negedge reg repeat signed unsigned wire while",
    types: "bit logic int byte shortint longint real time reg wire",
    constants: "0 1 x z", capitalized: "type",
  });

  define({
    id: "vhdl", label: "VHDL", lineComments: ["--"],
    strings: [{ open: '"', close: '"', escape: "", multiline: false }, { open: "'", close: "'", escape: "", multiline: false }],
    keywords: "architecture begin case component downto else elsif end entity function generic if in inout is library loop map of out package port process record return signal then to type use variable when while with",
    constants: "true false", caseInsensitive: true, capitalized: "type",
  });

  define({
    id: "pascal", label: "Pascal",
    blockComments: [["{", "}"], ["(*", "*)"]], lineComments: ["//"],
    strings: [{ open: "'", close: "'", escape: "", doubleClose: true, multiline: false }],
    keywords: "and begin case const div do downto else end file for function goto if in label mod nil not of or packed procedure program record repeat set then to type until var while with",
    constants: "true false nil", caseInsensitive: true, capitalized: "type",
  });

  define({
    id: "ada", label: "Ada", lineComments: ["--"],
    strings: [{ open: '"', close: '"', escape: "", doubleClose: true, multiline: false }],
    keywords: "abort abs abstract accept access aliased all and array at begin body case constant declare delay delta digits do else elsif end entry exception exit for function generic goto if in interface is limited loop mod new not null of or others out overriding package pragma private procedure protected raise range record rem renames requeue return reverse select separate subtype synchronized tagged task terminates then type until use when while with xor",
    constants: "true false", caseInsensitive: true, capitalized: "type",
  });

  define({
    id: "cobol", label: "COBOL", scan: scanCobolLine, lineComments: ["*>"],
    strings: [{ open: '"', close: '"', escape: "", multiline: false }, { open: "'", close: "'", escape: "", multiline: false }],
    keywords: "accept add alter call cancel close compute continue copy display divide else end end-if end-perform end-read end-rewrite end-write evaluate exit file go goback if initialize inspect into move multiply open perform read rewrite search set sort start stop string subtract unstring write working-storage procedure division section paragraph pic picture value values",
    constants: "true false zero zeros spaces high-values low-values", caseInsensitive: true,
  });

  define({
    id: "nix", label: "Nix", lineComments: ["#"], blockComments: [["/*", "*/"]],
    strings: [{ open: "''", close: "''", escape: "", multiline: true }, { open: '"', close: '"', escape: "\\", multiline: false }],
    keywords: "assert else if in inherit let or rec then with",
    constants: "true false null",
  });

  define({
    id: "latex", label: "LaTeX", lineComments: ["%"],
    strings: [{ open: "$", close: "$", escape: "\\", multiline: false }, { open: "\\[", close: "\\]", escape: "", multiline: true }],
    annotation: "\\\\[A-Za-z@]+\\*?",
    keywords: "documentclass usepackage begin end section subsection subsubsection paragraph label ref cite bibliography includegraphics caption item frac sum int",
  });

  define({
    id: "glsl", label: "GLSL", ...C_COMMENTS, preprocessor: true,
    keywords: "attribute break case const continue default discard do else for if in inout layout out precision return struct switch uniform void while buffer shared coherent readonly writeonly restrict",
    types: "bool bvec2 bvec3 bvec4 float vec2 vec3 vec4 int ivec2 ivec3 ivec4 mat2 mat3 mat4 sampler2D samplerCube uint uvec2 uvec3 uvec4 double",
    constants: "true false",
  });

  define({id: "text", label: "Plain text", scan: () => ({ tokens: [], state: null })});

  const ALIASES = {
    python: "python", python3: "python", py: "python", pyw: "python",
    javascript: "javascript", js: "javascript", jsx: "javascript", mjs: "javascript", cjs: "javascript", node: "javascript",
    javascriptreact: "javascript", typescriptreact: "typescript",
    typescript: "typescript", ts: "typescript", tsx: "typescript", mts: "typescript", cts: "typescript",
    json: "json", jsonc: "jsonc", "json with comments": "jsonc", json5: "jsonc",
    html: "markup", htm: "markup", xhtml: "markup", xml: "markup", svg: "markup", vue: "markup", svelte: "markup", astro: "markup",
    css: "css", scss: "scss", sass: "scss", less: "scss", stylus: "css", styl: "css",
    markdown: "markdown", md: "markdown", mdx: "markdown",
    mermaid: "mermaid", mmd: "mermaid",
    kml: "markup", gpx: "markup", geojson: "json", topojson: "json",
    jsonl: "json", ndjson: "json", ipynb: "json", drawio: "markup", dia: "markup",
    puml: "text", plantuml: "text",
    yaml: "yaml", yml: "yaml", toml: "toml", ini: "ini", config: "ini", cfg: "ini", conf: "ini", properties: "ini",
    shell: "shell", bash: "shell", sh: "shell", zsh: "shell", fish: "shell", ksh: "shell",
    console: "shell", terminal: "shell", "shell-session": "shell",
    powershell: "powershell", ps1: "powershell", psm1: "powershell",
    batch: "batch", bat: "batch", cmd: "batch",
    sql: "sql", c: "c", h: "c", "c++": "cpp", cpp: "cpp", cxx: "cpp", cc: "cpp", hpp: "cpp", hh: "cpp",
    "objective-c": "objc", objc: "objc", m: "objc", mm: "objc",
    "c#": "csharp", csharp: "csharp", cs: "csharp",
    java: "java", kotlin: "kotlin", kt: "kotlin", kts: "kotlin", scala: "scala", groovy: "groovy", gradle: "groovy",
    go: "go", golang: "go", rust: "rust", rs: "rust", swift: "swift", dart: "dart", zig: "zig", nim: "nim", d: "d",
    ruby: "ruby", rb: "ruby", rake: "ruby", php: "php", perl: "perl", pl: "perl", pm: "perl",
    lua: "lua", r: "r", julia: "julia", jl: "julia",
    haskell: "haskell", hs: "haskell", elixir: "elixir", ex: "elixir", exs: "elixir",
    erlang: "erlang", erl: "erlang", hrl: "erlang",
    clojure: "clojure", clj: "clojure", cljs: "clojure", cljc: "clojure", edn: "clojure",
    elm: "elm", ocaml: "ocaml", ml: "ocaml", mli: "ocaml",
    lisp: "clojure", scheme: "clojure", racket: "clojure", scm: "clojure", rkt: "clojure",
    cuda: "cpp", cu: "cpp", glsl: "glsl", vert: "glsl", frag: "glsl",
    systemverilog: "verilog", sv: "verilog", v: "verilog",
    csv: "text", tsv: "text",
    "f#": "fsharp", fsharp: "fsharp", fs: "fsharp",
    protobuf: "protobuf", proto: "protobuf", "protocol buffers": "protobuf",
    graphql: "graphql", gql: "graphql",
    assembly: "asm", asm: "asm", s: "asm",
    fortran: "fortran", f90: "fortran", f95: "fortran",
    matlab: "matlab", octave: "matlab",
    solidity: "solidity", sol: "solidity",
    verilog: "verilog", v: "verilog", sv: "verilog", systemverilog: "verilog",
    vhdl: "vhdl",
    pascal: "pascal", pas: "pascal",
    ada: "ada", adb: "ada", ads: "ada",
    cobol: "cobol", cob: "cobol", cbl: "cobol",
    nix: "nix",
    latex: "latex", tex: "latex",
    terraform: "terraform", hcl: "terraform", tf: "terraform", tfvars: "terraform",
    cmake: "cmake", makefile: "makefile", make: "makefile", mk: "makefile",
    dockerfile: "dockerfile", docker: "dockerfile",
    diff: "diff", patch: "diff",
    text: "text", txt: "text", plaintext: "text", plain: "text", "plain text": "text", log: "text",
  };

  const LANGUAGE_MENU = [
    "text", "python", "javascript", "typescript", "json", "jsonc", "markup", "css", "scss",
    "markdown", "mermaid", "yaml", "toml", "ini", "shell", "powershell", "batch", "sql", "go", "rust",
    "java", "kotlin", "scala", "groovy", "c", "cpp", "csharp", "objc", "swift", "dart", "ruby", "php",
    "lua", "r", "julia", "haskell", "elixir", "erlang", "clojure", "elm", "ocaml", "fsharp",
    "protobuf", "graphql", "terraform", "cmake", "makefile", "dockerfile", "diff", "zig", "nim", "d", "perl",
    "asm", "fortran", "matlab", "solidity", "verilog", "vhdl", "pascal", "ada", "cobol", "nix", "latex", "glsl",
  ];

  const SPECIAL_NAMES = {
    dockerfile: "dockerfile", makefile: "makefile", "gnumakefile": "makefile",
    "cmakelists.txt": "cmake", rakefile: "ruby", gemfile: "ruby",
    ".gitignore": "ini", ".dockerignore": "ini", ".env": "ini", ".editorconfig": "ini",
  };

  function normalize(value) {
    return String(value || "").trim().toLowerCase().replace(/\s+file$/, "");
  }

  function extensionOf(path) {
    const name = String(path || "").split("/").pop().toLowerCase();
    if (!name || name.startsWith(".")) return "";
    const dot = name.lastIndexOf(".");
    return dot > 0 ? name.slice(dot + 1) : "";
  }

  /* Resolve a backend language label, then a filename or extension to a grammar. */
  function resolve(label, path) {
    const text = normalize(label);
    if (ALIASES[text]) return ALIASES[text];
    const name = String(path || "").split("/").pop().toLowerCase();
    if (SPECIAL_NAMES[name]) return SPECIAL_NAMES[name];
    const extension = extensionOf(path);
    if (extension && ALIASES[extension]) return ALIASES[extension];
    if (text && ALIASES[text.split(".").pop()]) return ALIASES[text.split(".").pop()];
    return "text";
  }

  function scanLine(text, state, language) {
    const grammar = GRAMMARS[language];
    if (!grammar) return { tokens: [], state: null };
    const value = String(text === undefined || text === null ? "" : text);
    if (!value) return { tokens: [], state: state || null };
    if (value.length > 20000) return { tokens: [], state: state || null };
    return grammar.scan ? grammar.scan(value, state || null, grammar) : scanGeneric(value, state || null, grammar);
  }

  function lineComment(language) {
    const grammar = GRAMMARS[language];
    return grammar ? grammar.lineComment : null;
  }

  function blockComment(language) {
    const grammar = GRAMMARS[language];
    if (!grammar) return null;
    if (grammar.id === "markup" || grammar.id === "markdown") return { open: "<!-- ", close: " -->" };
    if (grammar.blockComments.length) return { open: grammar.blockComments[0][0] + " ", close: " " + grammar.blockComments[0][1] };
    return null;
  }

  function hasGrammar(language) {
    return Boolean(GRAMMARS[language]) && language !== "text";
  }

  const languages = LANGUAGE_MENU.filter(id => GRAMMARS[id]).map(id => ({ id, label: GRAMMARS[id].label }));

  Studio.highlight = { resolve, scanLine, lineComment, blockComment, hasGrammar, languages, grammars: GRAMMARS };
})();
