"use strict";

/* Studio.diagrams: a dependency-free preview for the common Mermaid diagram
   types.  Flowcharts, sequence diagrams and pie charts are parsed and drawn as
   SVG locally; unsupported syntax is left as a code block instead of failing.
   Parsing and layout are pure so they can be tested outside a browser. */
(function () {
  const Studio = window.Studio;
  const SVG_NS = "http://www.w3.org/2000/svg";
  let markerSeq = 0;

  function text(value) {
    return String(value === undefined || value === null ? "" : value);
  }

  function cleanLabel(value) {
    let label = text(value).trim();
    if ((label.startsWith('"') && label.endsWith('"')) || (label.startsWith("'") && label.endsWith("'"))) {
      label = label.slice(1, -1);
    }
    return label.replace(/<br\s*\/?>/gi, "\n").replace(/<[^>]*>/g, "").trim();
  }

  /* -- flowchart ------------------------------------------------------------ */

  const NODE_PATTERNS = [
    [/^\(\((.*)\)\)$/, "circle"],
    [/^\[\[(.*)\]\]$/, "rect"],
    [/^\[\((.*)\)\]$/, "rect"],
    [/^\((.*)\)$/, "round"],
    [/^\{(.*)\}$/, "diamond"],
    [/^>(.*)\]$/, "rect"],
    [/^\[(.*)\]$/, "rect"],
  ];

  function parseNodeToken(source, model) {
    const match = /^\s*([A-Za-z0-9_][\w-]*)\s*(.*)$/.exec(text(source));
    if (!match) return null;
    const id = match[1];
    let label = id;
    let shape = "rect";
    const rest = match[2].trim();
    if (rest) {
      for (const [pattern, kind] of NODE_PATTERNS) {
        const shapeMatch = pattern.exec(rest);
        if (shapeMatch) { label = cleanLabel(shapeMatch[1]); shape = kind; break; }
      }
    }
    if (!model.nodes.has(id)) model.nodes.set(id, { id, label, shape });
    else if (label !== id) {
      const node = model.nodes.get(id);
      node.label = label;
      node.shape = shape;
    }
    return id;
  }

  const EDGE_SPLIT = /\s*(-->|---|==>|===|-\.->|--x|--o|--\s*[^>|]+?\s*-->)\s*(?:\|([^|]*)\|)?\s*/g;

  function splitEdges(line) {
    const pieces = [];
    let last = 0;
    let match;
    EDGE_SPLIT.lastIndex = 0;
    while ((match = EDGE_SPLIT.exec(line))) {
      pieces.push({ text: line.slice(last, match.index), arrow: null, label: "" });
      let arrow = match[1];
      let label = match[2] || "";
      if (arrow.startsWith("--") && arrow.endsWith("-->") && arrow !== "-->") {
        label = label || arrow.slice(2, arrow.lastIndexOf("-->")).trim();
        arrow = "-->";
      }
      pieces.push({ text: "", arrow, label, target: line.slice(EDGE_SPLIT.lastIndex) });
      last = EDGE_SPLIT.lastIndex;
    }
    if (!pieces.length) return [{ text: line, arrow: null, label: "" }];
    const result = [{ text: pieces[0].text, arrow: null, label: "" }];
    for (let index = 1; index < pieces.length; index += 2) {
      const after = pieces[index];
      const next = pieces[index + 1] || { text: after.target || "" };
      const arrowIndex = next.text.indexOf("|");
      let target = next.text;
      let label = after.label;
      if (arrowIndex > 0) {
        const close = next.text.indexOf("|", arrowIndex + 1);
        if (close > arrowIndex) {
          label = label || next.text.slice(arrowIndex + 1, close);
          target = next.text.slice(close + 1);
        }
      }
      result.push({ text: target, arrow: after.arrow, label });
    }
    return result;
  }

  function parseFlowchart(source) {
    const lines = text(source).replace(/\r\n?/g, "\n").split("\n")
      .map(line => line.trim()).filter(Boolean);
    const header = /^(?:graph|flowchart)\s+(TD|TB|BT|LR|RL)\b/i.exec(lines.shift() || "");
    const rawDirection = header ? header[1].toUpperCase() : "TD";
    const model = {
      type: "flowchart",
      direction: rawDirection === "TB" ? "TD" : rawDirection,
      nodes: new Map(),
      edges: [],
    };
    for (const line of lines) {
      if (/^(%%|subgraph\b|end\b|classDef\b|class\b|style\b|linkStyle\b|click\b|direction\b)/i.test(line)) continue;
      const pieces = splitEdges(line);
      let previous = parseNodeToken(pieces[0].text, model);
      for (let index = 1; index < pieces.length; index += 1) {
        const piece = pieces[index];
        const next = parseNodeToken(piece.text, model);
        if (previous && next) {
          model.edges.push({
            from: previous, to: next, label: cleanLabel(piece.label),
            arrow: piece.arrow || "-->",
          });
        }
        previous = next;
      }
    }
    return model.nodes.size ? model : null;
  }

  function labelLines(node) {
    return text(node.label || node.id).split("\n").filter((line, index, list) => line || list.length === 1);
  }

  function nodeSize(node) {
    const lines = labelLines(node);
    const longest = Math.max(...lines.map(line => line.length), node.id.length);
    return {
      width: Math.max(64, Math.min(240, longest * 7.2 + 28)),
      height: Math.max(36, 18 + lines.length * 16),
      lines,
    };
  }

  function layoutFlowchart(model) {
    const direction = model.direction || "TD";
    const vertical = direction === "TD" || direction === "BT";
    const nodes = [...model.nodes.values()].map(node => ({ ...node, ...nodeSize(node), rank: 0 }));
    const byId = new Map(nodes.map(node => [node.id, node]));
    const outgoing = new Map();
    const incoming = new Map();
    for (const edge of model.edges) {
      if (!outgoing.has(edge.from)) outgoing.set(edge.from, []);
      outgoing.get(edge.from).push(edge.to);
      incoming.set(edge.to, (incoming.get(edge.to) || 0) + 1);
    }
    const roots = nodes.filter(node => !incoming.get(node.id));
    const queue = (roots.length ? roots : nodes.slice(0, 1)).map(node => [node, 0]);
    const seen = new Set();
    while (queue.length) {
      const [node, level] = queue.shift();
      if (seen.has(node.id)) continue;
      seen.add(node.id);
      node.rank = level;
      for (const id of outgoing.get(node.id) || []) {
        const child = byId.get(id);
        if (child && !seen.has(id)) queue.push([child, level + 1]);
      }
    }
    /* Longest-path relaxation gives joins a stable rank; the guard keeps a
       cycle from looping forever. */
    let changed = true;
    let guard = 0;
    while (changed && guard < nodes.length * 2) {
      changed = false;
      guard += 1;
      for (const edge of model.edges) {
        const from = byId.get(edge.from);
        const to = byId.get(edge.to);
        if (!from || !to || from === to) continue;
        if (to.rank <= from.rank && to.rank < nodes.length) {
          to.rank = from.rank + 1;
          changed = true;
        }
      }
    }
    const ranks = new Map();
    for (const node of nodes) {
      if (!ranks.has(node.rank)) ranks.set(node.rank, []);
      ranks.get(node.rank).push(node);
    }
    const rankKeys = [...ranks.keys()].sort((left, right) => left - right);
    const margin = 24;
    const gapMain = vertical ? 62 : 78;
    const gapCross = vertical ? 54 : 46;
    let cursor = margin;
    let crossMax = 0;
    if (vertical) {
      for (const rank of rankKeys) {
        const row = ranks.get(rank);
        const rowHeight = Math.max(...row.map(node => node.height));
        const rowWidth = row.reduce((sum, node) => sum + node.width, 0) + gapCross * (row.length - 1);
        crossMax = Math.max(crossMax, rowWidth);
        for (const node of row) node.rankY = cursor + (rowHeight - node.height) / 2;
        cursor += rowHeight + gapMain;
      }
      for (const rank of rankKeys) {
        const row = ranks.get(rank);
        const rowWidth = row.reduce((sum, node) => sum + node.width, 0) + gapCross * (row.length - 1);
        let x = margin + (crossMax - rowWidth) / 2;
        for (const node of row) {
          node.x = x;
          node.y = direction === "BT" ? cursor - (node.rankY + node.height) : node.rankY;
          x += node.width + gapCross;
        }
      }
    } else {
      for (const rank of rankKeys) {
        const column = ranks.get(rank);
        const columnWidth = Math.max(...column.map(node => node.width));
        const columnHeight = column.reduce((sum, node) => sum + node.height, 0) + gapCross * (column.length - 1);
        crossMax = Math.max(crossMax, columnHeight);
        for (const node of column) node.rankX = cursor + (columnWidth - node.width) / 2;
        cursor += columnWidth + gapMain;
      }
      for (const rank of rankKeys) {
        const column = ranks.get(rank);
        const columnHeight = column.reduce((sum, node) => sum + node.height, 0) + gapCross * (column.length - 1);
        let y = margin + (crossMax - columnHeight) / 2;
        for (const node of column) {
          node.x = direction === "RL" ? cursor - (node.rankX + node.width) : node.rankX;
          node.y = y;
          y += node.height + gapCross;
        }
      }
    }
    const width = Math.max(...nodes.map(node => node.x + node.width)) + margin;
    const height = Math.max(...nodes.map(node => node.y + node.height)) + margin;
    return { ...model, nodes, byId, width, height, vertical, direction };
  }

  function edgeGeometry(layout, edge) {
    const from = layout.byId.get(edge.from);
    const to = layout.byId.get(edge.to);
    if (!from || !to) return null;
    if (from === to) {
      const x = from.x + from.width;
      const y = from.y + from.height / 2;
      return {
        d: `M ${x} ${y - 10} C ${x + 42} ${y - 34}, ${x + 42} ${y + 34}, ${x} ${y + 10}`,
        mx: x + 30, my: y,
      };
    }
    if (layout.vertical) {
      const down = layout.direction !== "BT";
      const sx = from.x + from.width / 2;
      const sy = down ? from.y + from.height : from.y;
      const tx = to.x + to.width / 2;
      const ty = down ? to.y : to.y + to.height;
      const bend = Math.max(20, Math.abs(ty - sy) * 0.45);
      return {
        d: `M ${sx} ${sy} C ${sx} ${sy + (down ? bend : -bend)}, ${tx} ${ty - (down ? bend : -bend)}, ${tx} ${ty}`,
        mx: (sx + tx) / 2, my: (sy + ty) / 2,
      };
    }
    const right = layout.direction !== "RL";
    const sx = right ? from.x + from.width : from.x;
    const sy = from.y + from.height / 2;
    const tx = right ? to.x : to.x + to.width;
    const ty = to.y + to.height / 2;
    const bend = Math.max(20, Math.abs(tx - sx) * 0.45);
    return {
      d: `M ${sx} ${sy} C ${sx + (right ? bend : -bend)} ${sy}, ${tx - (right ? bend : -bend)} ${ty}, ${tx} ${ty}`,
      mx: (sx + tx) / 2, my: (sy + ty) / 2,
    };
  }

  /* -- sequence ------------------------------------------------------------- */

  function parseSequence(source) {
    const lines = text(source).replace(/\r\n?/g, "\n").split("\n")
      .map(line => line.trim()).filter(Boolean);
    lines.shift();
    const model = { type: "sequence", participants: [], messages: [], notes: [] };
    const byName = new Map();
    const add = (name, label) => {
      if (!name) return null;
      if (byName.has(name)) {
        if (label) byName.get(name).label = cleanLabel(label);
        return byName.get(name);
      }
      const participant = { id: name, label: cleanLabel(label) || name };
      byName.set(name, participant);
      model.participants.push(participant);
      return participant;
    };
    for (const line of lines) {
      const participant = /^(?:participant|actor)\s+([A-Za-z0-9_](?:[A-Za-z0-9_-]*[A-Za-z0-9_])?)(?:\s+as\s+(.+))?$/i.exec(line);
      if (participant) { add(participant[1], participant[2]); continue; }
      const note = /^note\s+(over|left of|right of)\s+([^:]+):\s*(.+)$/i.exec(line);
      if (note) {
        const ids = note[2].split(",").map(item => add(item.trim())).filter(Boolean);
        model.notes.push({ placement: note[1].toLowerCase(), ids: ids.map(item => item.id), text: cleanLabel(note[3]) });
        continue;
      }
      if (/^(autonumber|loop|alt|else|opt|par|and|rect|end|critical|break|activate|deactivate)\b/i.test(line)) continue;
      const message = /^([A-Za-z0-9_](?:[A-Za-z0-9_-]*[A-Za-z0-9_])?)\s*(-->>|->>|-->|->|--x|-x|--\)|-\))\s*([A-Za-z0-9_](?:[A-Za-z0-9_-]*[A-Za-z0-9_])?)\s*:\s*(.+)$/.exec(line);
      if (!message) continue;
      const from = add(message[1]);
      const to = add(message[3]);
      model.messages.push({
        from: from.id, to: to.id, arrow: message[2], text: cleanLabel(message[4]),
      });
    }
    return model.participants.length ? model : null;
  }

  function layoutSequence(model) {
    const participants = model.participants;
    const boxWidth = 130;
    const gap = 64;
    const left = 28;
    const top = 26;
    const boxHeight = 38;
    const rowHeight = 44;
    const byId = new Map(participants.map(participant => [participant.id, participant]));
    participants.forEach((participant, index) => {
      participant.index = index;
      participant.x = left + index * (boxWidth + gap);
      participant.y = top;
      participant.width = boxWidth;
      participant.height = boxHeight;
    });
    let row = 0;
    const notes = [];
    const messages = [];
    for (const note of model.notes) {
      const ids = note.ids.map(id => byId.get(id)).filter(Boolean);
      if (!ids.length) continue;
      const x = Math.min(...ids.map(item => item.x));
      const right = Math.max(...ids.map(item => item.x + item.width));
      notes.push({ ...note, x, y: top + boxHeight + 22 + row * rowHeight, width: right - x, height: 30 });
      row += 1;
    }
    for (const message of model.messages) {
      messages.push({
        ...message,
        y: top + boxHeight + 44 + row * rowHeight,
        fromX: byId.get(message.from).x + byId.get(message.from).width / 2,
        toX: byId.get(message.to).x + byId.get(message.to).width / 2,
      });
      row += 1;
    }
    const width = left * 2 + participants.length * boxWidth + (participants.length - 1) * gap;
    const height = top + boxHeight + 44 + Math.max(1, row) * rowHeight + 20;
    return { ...model, byId, messages, notes, width, height, top, boxHeight, lifeline: height - 24 };
  }

  /* -- pie ------------------------------------------------------------------ */

  function parsePie(source) {
    const lines = text(source).replace(/\r\n?/g, "\n").split("\n")
      .map(line => line.trim()).filter(Boolean);
    const header = /^pie(?:\s+title\s+(.+))?/i.exec(lines.shift() || "");
    const model = { type: "pie", title: header && header[1] ? cleanLabel(header[1]) : "", slices: [] };
    for (const line of lines) {
      const match = /^(?:"(.+)"|(.+?))\s*:\s*([\d.]+)$/.exec(line);
      if (!match) continue;
      const value = Number(match[3]);
      if (!Number.isFinite(value) || value < 0) continue;
      model.slices.push({ label: cleanLabel(match[1] || match[2]), value });
    }
    return model.slices.length ? model : null;
  }

  function layoutPie(model) {
    const total = model.slices.reduce((sum, slice) => sum + slice.value, 0) || 1;
    const radius = 92;
    let angle = -Math.PI / 2;
    const slices = model.slices.map((slice, index) => {
      const sweep = (slice.value / total) * Math.PI * 2;
      const item = { ...slice, index, start: angle, end: angle + sweep };
      angle += sweep;
      return item;
    });
    const legend = Math.max(0, ...slices.map(slice => `${slice.label} · ${slice.value}`.length * 6.4));
    return {
      ...model, slices, total, radius, cx: 122, cy: 118,
      width: Math.max(390, 122 + radius + 32 + legend + 24),
      height: Math.max(250, 44 + slices.length * 22),
    };
  }

  /* -- SVG rendering -------------------------------------------------------- */

  function svg(tag, attrs) {
    const element = document.createElementNS(SVG_NS, tag);
    for (const [key, value] of Object.entries(attrs || {})) {
      if (value !== null && value !== undefined) element.setAttribute(key, String(value));
    }
    return element;
  }

  function makeMarker() {
    markerSeq += 1;
    const id = `diagram-arrow-${markerSeq}`;
    const marker = svg("marker", {
      id, viewBox: "0 0 10 10", refX: "9", refY: "5", markerWidth: "7", markerHeight: "7",
      orient: "auto-start-reverse",
    });
    marker.append(svg("path", { d: "M 0 0 L 10 5 L 0 10 z", class: "diagram-arrow-head" }));
    return { id, element: marker };
  }

  function baseSvg(layout, label) {
    const element = svg("svg", {
      viewBox: `0 0 ${Math.ceil(layout.width)} ${Math.ceil(layout.height)}`,
      class: "diagram-svg", role: "img", "aria-label": label || "Diagram",
    });
    const title = svg("title");
    title.textContent = label || "Diagram";
    element.append(title);
    const marker = makeMarker();
    const defs = svg("defs");
    defs.append(marker.element);
    element.append(defs);
    return { element, marker: marker.id };
  }

  function renderFlowchart(layout) {
    const label = layout.nodes.map(node => node.label).join(", ").slice(0, 160);
    const { element, marker } = baseSvg(layout, `Flowchart: ${label}`);
    for (const edge of layout.edges) {
      const geometry = edgeGeometry(layout, edge);
      if (!geometry) continue;
      element.append(svg("path", {
        d: geometry.d, class: "diagram-edge",
        "marker-end": edge.arrow === "---" ? null : `url(#${marker})`,
      }));
      if (edge.label) {
        const textNode = svg("text", {
          x: geometry.mx, y: geometry.my - 6, class: "diagram-edge-label", "text-anchor": "middle",
        });
        textNode.textContent = edge.label;
        element.append(textNode);
      }
    }
    for (const node of layout.nodes) {
      const group = svg("g", { class: "diagram-node" });
      let shape;
      if (node.shape === "diamond") {
        const points = [
          `${node.x + node.width / 2},${node.y}`,
          `${node.x + node.width},${node.y + node.height / 2}`,
          `${node.x + node.width / 2},${node.y + node.height}`,
          `${node.x},${node.y + node.height / 2}`,
        ].join(" ");
        shape = svg("polygon", { points, class: "diagram-shape" });
      } else {
        shape = svg("rect", {
          x: node.x, y: node.y, width: node.width, height: node.height,
          rx: node.shape === "round" || node.shape === "circle" ? node.height / 2 : 8,
          class: "diagram-shape",
        });
      }
      group.append(shape);
      const labelNode = svg("text", {
        x: node.x + node.width / 2,
        y: node.y + node.height / 2 - (node.lines.length - 1) * 8,
        class: "diagram-node-label", "text-anchor": "middle",
      });
      node.lines.forEach((line, index) => {
        const span = svg("tspan", { x: node.x + node.width / 2, dy: index === 0 ? 0 : 16 });
        span.textContent = line || " ";
        labelNode.append(span);
      });
      const title = svg("title");
      title.textContent = node.label;
      group.append(title, labelNode);
      element.append(group);
    }
    return element;
  }

  function renderSequence(layout) {
    const names = layout.participants.map(participant => participant.label).join(", ");
    const { element, marker } = baseSvg(layout, `Sequence: ${names.slice(0, 160)}`);
    for (const participant of layout.participants) {
      element.append(svg("line", {
        x1: participant.x + participant.width / 2, y1: participant.y + participant.height,
        x2: participant.x + participant.width / 2, y2: layout.lifeline, class: "diagram-lifeline",
      }));
      element.append(svg("rect", {
        x: participant.x, y: participant.y, width: participant.width, height: participant.height,
        rx: 7, class: "diagram-shape",
      }));
      const labelNode = svg("text", {
        x: participant.x + participant.width / 2, y: participant.y + participant.height / 2 + 5,
        class: "diagram-node-label", "text-anchor": "middle",
      });
      labelNode.textContent = participant.label;
      element.append(labelNode);
    }
    for (const message of layout.messages) {
      if (message.from === message.to) {
        element.append(svg("path", {
          d: `M ${message.fromX} ${message.y - 10} C ${message.fromX + 44} ${message.y - 26}, ${message.fromX + 44} ${message.y + 26}, ${message.fromX} ${message.y + 10}`,
          class: "diagram-edge", "marker-end": `url(#${marker})`,
        }));
        const selfLabel = svg("text", {
          x: message.fromX + 28, y: message.y - 12, class: "diagram-edge-label", "text-anchor": "start",
        });
        selfLabel.textContent = message.text;
        element.append(selfLabel);
        continue;
      }
      const direction = message.toX >= message.fromX ? 1 : -1;
      element.append(svg("line", {
        x1: message.fromX, y1: message.y, x2: message.toX - direction * 5, y2: message.y,
        class: "diagram-edge", "marker-end": message.arrow.includes("x") ? null : `url(#${marker})`,
      }));
      if (message.arrow.includes("x")) {
        element.append(svg("path", {
          d: `M ${message.toX} ${message.y - 5} L ${message.toX - 10} ${message.y + 5} M ${message.toX} ${message.y + 5} L ${message.toX - 10} ${message.y - 5}`,
          class: "diagram-edge",
        }));
      }
      const labelNode = svg("text", {
        x: (message.fromX + message.toX) / 2, y: message.y - 7,
        class: "diagram-edge-label", "text-anchor": "middle",
      });
      labelNode.textContent = message.text;
      element.append(labelNode);
    }
    for (const note of layout.notes) {
      element.append(svg("rect", {
        x: note.x, y: note.y, width: note.width, height: note.height,
        rx: 6, class: "diagram-note",
      }));
      const labelNode = svg("text", {
        x: note.x + note.width / 2, y: note.y + note.height / 2 + 5,
        class: "diagram-note-label", "text-anchor": "middle",
      });
      labelNode.textContent = note.text;
      element.append(labelNode);
    }
    return element;
  }

  function arcPath(cx, cy, radius, start, end) {
    const x1 = cx + radius * Math.cos(start);
    const y1 = cy + radius * Math.sin(start);
    const x2 = cx + radius * Math.cos(end);
    const y2 = cy + radius * Math.sin(end);
    const large = end - start > Math.PI ? 1 : 0;
    return `M ${cx} ${cy} L ${x1} ${y1} A ${radius} ${radius} 0 ${large} 1 ${x2} ${y2} Z`;
  }

  function renderPie(layout) {
    const { element } = baseSvg(layout, layout.title ? `Pie chart: ${layout.title}` : "Pie chart");
    for (const slice of layout.slices) {
      const sweep = slice.end - slice.start;
      const shape = Math.abs(sweep - Math.PI * 2) < 0.001
        ? svg("circle", { cx: layout.cx, cy: layout.cy, r: layout.radius, class: `diagram-slice slice-${slice.index % 6}` })
        : svg("path", {
          d: arcPath(layout.cx, layout.cy, layout.radius, slice.start, slice.end),
          class: `diagram-slice slice-${slice.index % 6}`,
        });
      const title = svg("title");
      title.textContent = `${slice.label}: ${slice.value}`;
      shape.append(title);
      element.append(shape);
    }
    const legendX = layout.cx + layout.radius + 32;
    layout.slices.forEach((slice, index) => {
      const y = 34 + index * 22;
      element.append(svg("rect", {
        x: legendX, y: y - 11, width: 11, height: 11, rx: 2,
        class: `diagram-slice slice-${slice.index % 6}`,
      }));
      const labelNode = svg("text", { x: legendX + 18, y, class: "diagram-legend-label" });
      labelNode.textContent = `${slice.label} · ${slice.value}`;
      element.append(labelNode);
    });
    return element;
  }

  function render(source) {
    const model = parse(source);
    if (!model) return null;
    try {
      if (model.type === "flowchart") return renderFlowchart(layoutFlowchart(model));
      if (model.type === "sequence") return renderSequence(layoutSequence(model));
      if (model.type === "pie") return renderPie(layoutPie(model));
    } catch (error) {
      return null;
    }
    return null;
  }

  function parse(source) {
    const value = text(source).trim();
    if (!value) return null;
    if (/^(flowchart|graph)\b/i.test(value)) return parseFlowchart(value);
    if (/^sequenceDiagram\b/i.test(value)) return parseSequence(value);
    if (/^pie\b/i.test(value)) return parsePie(value);
    return null;
  }

  /* Replace mermaid fences in rendered Markdown with a diagram and keep the
     source behind a disclosure, so copy and inspection still work. */
  function enhance(root) {
    if (!root || !root.querySelectorAll) return;
    for (const block of Array.from(root.querySelectorAll(".code-block"))) {
      if (block.dataset.enhanced) continue;
      const language = block.querySelector(".code-lang");
      const name = language ? language.textContent.trim().toLowerCase() : "";
      if (!["mermaid", "mmd", "mermaid.js"].includes(name)) continue;
      const code = block.querySelector("pre code");
      if (!code) continue;
      const element = render(code.textContent);
      if (!element) continue;
      block.dataset.enhanced = "true";
      const wrapper = document.createElement("div");
      wrapper.className = "diagram";
      wrapper.append(element);
      const details = document.createElement("details");
      details.className = "diagram-source";
      const summary = document.createElement("summary");
      summary.textContent = "Diagram source";
      const clone = block.cloneNode(true);
      clone.dataset.enhanced = "source";
      details.append(summary, clone);
      wrapper.append(details);
      block.replaceWith(wrapper);
    }
  }

  Studio.diagrams = {
    parse, render, enhance,
    layoutFlowchart, layoutSequence, layoutPie,
    edgeGeometry,
  };
})();
