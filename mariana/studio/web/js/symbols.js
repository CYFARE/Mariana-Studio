"use strict";

/* Studio.symbols: a small, language-aware outline for the IDE.  This is
   deliberately heuristic: it indexes declarations and headings, not a syntax
   tree, so it stays fast on files larger than the editor limit. */
(function () {
  const Studio = window.Studio;
  const MAX_SYMBOLS = 400;
  const SKIP_NAMES = new Set([
    "if", "for", "while", "switch", "catch", "function", "return", "else", "do",
    "try", "with", "constructor", "sizeof", "typeof", "new", "delete", "throw",
  ]);

  const PATTERNS = {
    python: [
      [/^(\s*)(?:async\s+)?def\s+([A-Za-z_]\w*)\s*\(/, "function"],
      [/^(\s*)class\s+([A-Za-z_]\w*)/, "class"],
    ],
    javascript: [
      [/^(\s*)(?:export\s+(?:default\s+)?)?(?:async\s+)?function\s*\*?\s*([A-Za-z_$][\w$]*)/, "function"],
      [/^(\s*)(?:export\s+(?:default\s+)?)?class\s+([A-Za-z_$][\w$]*)/, "class"],
      [/^(\s*)(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?(?:function\b|(?:\([^)]*\)|[A-Za-z_$][\w$]*)\s*=>)/, "function"],
      [/^(\s*)(?:export\s+)?(?:interface|type|enum)\s+([A-Za-z_$][\w$]*)/, "type"],
    ],
    go: [
      [/^(\s*)func\s+(?:\([^)]*\)\s*)?([A-Za-z_]\w*)/, "function"],
      [/^(\s*)type\s+([A-Za-z_]\w*)\s+(?:struct|interface)/, "type"],
    ],
    rust: [
      [/^(\s*)(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?(?:unsafe\s+)?fn\s+([A-Za-z_]\w*)/, "function"],
      [/^(\s*)(?:pub(?:\([^)]*\))?\s+)?(?:struct|enum|trait|mod)\s+([A-Za-z_]\w*)/, "type"],
      [/^(\s*)impl(?:\s*<[^>]*>)?\s+([A-Za-z_]\w*)/, "impl"],
    ],
    java: [
      [/^(\s*)(?:(?:public|private|protected|internal|static|final|abstract|sealed|open|data|override)\s+)*(?:class|interface|enum|record|struct|object)\s+([A-Za-z_]\w*)/, "type"],
      [/^(\s*)(?:(?:public|private|protected|internal|static|final|abstract|synchronized|native|default)\s+)+[\w<>\[\],.?]+\s+([A-Za-z_]\w*)\s*\(/, "function"],
    ],
    ruby: [
      [/^(\s*)def\s+(?:self\.)?([A-Za-z_]\w*[?!=]?)/, "function"],
      [/^(\s*)(?:class|module)\s+([A-Za-z_]\w*)/, "type"],
    ],
    php: [
      [/^(\s*)(?:(?:public|private|protected|static|final|abstract)\s+)*function\s+([A-Za-z_]\w*)/, "function"],
      [/^(\s*)(?:abstract\s+|final\s+)?(?:class|interface|trait|enum)\s+([A-Za-z_]\w*)/, "type"],
    ],
    c: [
      [/^(\s*)(?:(?:static|inline|extern|const)\s+)*[\w*&:<>,~\[\]]+\s+([A-Za-z_]\w*)\s*\([^;]*\)\s*(?:const\s*)?\{?\s*$/, "function"],
      [/^(\s*)(?:class|struct|union|enum|namespace)\s+([A-Za-z_]\w*)/, "type"],
    ],
  };

  const BRACE_FAMILIES = new Set(["javascript", "go", "rust", "java", "php", "c"]);
  const COMMENT_STARTS = {
    python: ["#"], ruby: ["#"], php: ["//", "#", "/*", "*"],
    javascript: ["//", "/*", "*"], go: ["//", "/*", "*"], rust: ["//", "/*", "*"],
    java: ["//", "/*", "*"], c: ["//", "/*", "*"], markup: ["<!--"],
  };

  function familyFor(language, path) {
    const label = String(language || "").toLowerCase();
    const name = String(path || "").toLowerCase();
    const extension = name.includes(".") ? name.split(".").pop() : "";
    if (["md", "markdown", "mdx"].includes(extension) || label.includes("markdown")) return "markdown";
    if (["html", "htm", "xhtml", "xml", "svg", "vue", "svelte"].includes(extension)
        || label.includes("html") || label.includes("xml")) return "markup";
    if (extension === "py" || label.includes("python")) return "python";
    if (["js", "jsx", "mjs", "cjs", "ts", "tsx", "mts", "cts"].includes(extension)
        || label.includes("javascript") || label.includes("typescript")) return "javascript";
    if (extension === "go" || label === "go" || label.includes("golang")) return "go";
    if (extension === "rs" || label.includes("rust")) return "rust";
    if (["java", "kt", "kts", "cs", "swift", "scala", "groovy", "dart"].includes(extension)
        || /java|kotlin|csharp|c#|swift|scala|groovy|dart/.test(label)) return "java";
    if (extension === "rb" || label.includes("ruby")) return "ruby";
    if (extension === "php" || label.includes("php")) return "php";
    if (["c", "h", "cc", "cpp", "cxx", "hpp", "hh", "m", "mm"].includes(extension)
        || /c\/c\+\+|c\+\+|objective-c/.test(label)) return "c";
    return "";
  }

  function indentWidth(text) {
    const leading = /^[ \t]*/.exec(text)[0];
    let width = 0;
    for (const character of leading) width += character === "\t" ? 4 : 1;
    return width;
  }

  function extract(text, language, path) {
    const family = familyFor(language, path);
    if (!family) return [];
    const lines = String(text === null || text === undefined ? "" : text)
      .replace(/\r\n?/g, "\n").split("\n");
    const symbols = [];
    let braceDepth = 0;
    let fenced = false;
    for (let index = 0; index < lines.length && symbols.length < MAX_SYMBOLS; index += 1) {
      const line = lines[index];
      if (family === "markdown") {
        if (/^\s*(```|~~~)/.test(line)) {
          fenced = !fenced;
          continue;
        }
        if (fenced) continue;
        const heading = /^(\s*)(#{1,6})\s+(.+?)\s*#*\s*$/.exec(line);
        if (heading) {
          symbols.push({
            name: heading[3].trim(), kind: "heading", level: heading[2].length,
            line: index + 1, column: heading[1].length + 1, depth: heading[2].length - 1,
          });
        }
        continue;
      }
      if (family === "markup") {
        if (/^\s*<!--/.test(line)) continue;
        const heading = /^(\s*)<h([1-6])[^>]*>([^<]+)<\/h\2>/i.exec(line);
        if (heading) {
          symbols.push({
            name: heading[3].trim(), kind: "heading", level: Number(heading[2]),
            line: index + 1, column: heading[1].length + 1, depth: Math.round(indentWidth(line) / 2),
          });
          continue;
        }
        const id = /^(\s*)<([A-Za-z][\w:-]*)[^>]*\sid=(["'])([^"']+)\3/.exec(line);
        if (id) {
          symbols.push({
            name: `${id[2]}#${id[4]}`, kind: "tag",
            line: index + 1, column: id[1].length + 1, depth: Math.round(indentWidth(line) / 2),
          });
        }
        continue;
      }
      const trimmed = line.trim();
      if (!trimmed) continue;
      const comments = COMMENT_STARTS[family] || [];
      if (comments.some(prefix => trimmed.startsWith(prefix))) continue;
      const indent = indentWidth(line);
      const depth = family === "python" || family === "ruby"
        ? Math.max(0, Math.round(indent / 4))
        : Math.max(0, braceDepth);
      for (const [pattern, kind] of PATTERNS[family]) {
        const match = pattern.exec(line);
        if (!match) continue;
        const name = match[2];
        if (!name || SKIP_NAMES.has(name)) continue;
        symbols.push({ name, kind, line: index + 1, column: indent + 1, depth });
        break;
      }
      if (BRACE_FAMILIES.has(family)) {
        for (const character of line) {
          if (character === "{") braceDepth += 1;
          else if (character === "}") braceDepth = Math.max(0, braceDepth - 1);
        }
      }
    }
    return symbols;
  }

  Studio.symbols = { extract, familyFor };
})();
