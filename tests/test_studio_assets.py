"""Studio's assembled frontend, safe Markdown, charts and static asset security."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

import pytest

from mariana.studio.assets import CONTENT_SECURITY_POLICY, WEB, read_asset, render_index
from tests.helpers import studio


class PageParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = []
        self.assets = []
        self.forbidden = []

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key == "id":
                self.ids.append(value)
            if key in {"style"} or key.startswith("on") or (tag == "script" and key is None):
                self.forbidden.append((tag, key, value))
            if key == "src" and tag == "script":
                self.assets.append(value)
            if key == "href" and tag == "link":
                self.assets.append(value)


def parse_page() -> PageParser:
    parser = PageParser()
    parser.feed(render_index())
    return parser


def node_script(source: str, *paths: Path) -> dict:
    if not shutil.which("node"):
        pytest.skip("Node.js is not installed")
    result = subprocess.run(
        ["node", "-e", source, *map(str, paths)], capture_output=True, text=True, timeout=20, check=True,
    )
    return json.loads(result.stdout)


def test_every_view_and_asset_is_assembled_without_inline_content():
    page = render_index()
    parser = parse_page()
    assert len(parser.ids) == len(set(parser.ids)), "DOM ids must be unique"
    assert "<!-- include views/" not in page and "<!-- missing" not in page
    assert not any(tag == "script" and content.strip() for tag, content in re.findall(r"<(script)[^>]*>(.*?)</script>", page, re.S))
    assert not [item for item in parser.forbidden if item[1] is not None]
    for name in ("home", "models", "train", "finetune", "chat", "agents", "ide", "deploy"):
        assert f"{name}-view" in parser.ids
    for asset in parser.assets:
        assert asset.startswith("/") and not asset.startswith("//")
        assert read_asset(asset) is not None, asset


def test_frontend_modules_load_in_dependency_order():
    scripts = re.findall(r'<script src="([^"]+)"', render_index())
    order = {path: index for index, path in enumerate(scripts)}
    assert "/js/core.js" not in order
    assert order["/js/dom.js"] < order["/js/feedback.js"] < order["/js/router.js"]
    assert order["/js/ui.js"] < order["/js/overlay.js"] < order["/js/components.js"]
    assert order["/js/components.js"] < order["/js/views/home.js"] < order["/js/boot.js"]
    assert order["/js/ui.js"] < order["/js/split.js"] < order["/js/views/ide.js"] < order["/js/boot.js"]
    assert order["/js/highlight.js"] < order["/js/editor.js"] < order["/js/views/ide.js"] < order["/js/boot.js"]
    assert order["/js/tokens.js"] < order["/js/views/ide.js"]
    assert order["/js/voice.js"] < order["/js/views/ide.js"]


def test_every_model_picker_uses_the_shared_provider_flow():
    for name in ("chat", "agents", "ide", "models"):
        source = (WEB / "js" / "views" / f"{name}.js").read_text(encoding="utf-8")
        assert "Studio.providers" in source, name
    providers = (WEB / "js" / "providers.js").read_text(encoding="utf-8")
    assert "addEndpointDialog" in providers and "Studio.providers =" in providers


def test_ide_modules_register_with_the_shared_studio_namespace():
    if not shutil.which("node"):
        pytest.skip("Node.js is not installed")
    script = r"""
const fs = require("fs"), vm = require("vm");
const Studio = {
  store: { get(key, fallback) { return fallback; }, set() {} },
  debounce: task => task,
  views: { register(name, definition) { this.name = name; this.definition = definition; } },
};
const sandbox = { window: { Studio }, console, setTimeout, clearTimeout };
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(process.argv[1], "utf8"), sandbox);
vm.runInContext(fs.readFileSync(process.argv[2], "utf8"), sandbox);
console.log(JSON.stringify({
  view: Studio.views.name,
  title: Studio.views.definition && Studio.views.definition.title,
  editor: typeof Studio.editor.create,
}));
"""
    result = subprocess.run(
        ["node", "-e", script, str(WEB / "js/editor.js"), str(WEB / "js/views/ide.js")],
        capture_output=True, text=True, timeout=20, check=True,
    )
    assert json.loads(result.stdout) == {"view": "ide", "title": "IDE", "editor": "function"}


def test_ide_layout_dividers_are_keyboard_accessible():
    page = render_index()
    assert 'id="ide-agent"' in page and 'id="ide-editor-column"' in page
    for splitter, controls in (
        ("ide-split-sidebar", "ide-sidebar ide-editor-column"),
        ("ide-split-agent", "ide-agent ide-editor-column"),
    ):
        match = re.search(rf'<div id="{splitter}"[^>]*>', page)
        assert match, splitter
        tag = match.group(0)
        assert 'role="separator"' in tag
        assert 'tabindex="0"' in tag
        assert 'aria-orientation="vertical"' in tag
        assert f'aria-controls="{controls}"' in tag


def test_agent_layout_dividers_are_keyboard_accessible():
    page = render_index()
    assert 'id="agents-designer"' in page
    for splitter, orientation, controls in (
        ("agents-split-palette", "vertical", "agents-palette agents-canvas-column"),
        ("agents-split-inspector", "vertical", "agents-inspector agents-canvas-column"),
        ("agents-split-dock", "horizontal", "agents-bottom"),
    ):
        match = re.search(rf'<div id="{splitter}"[^>]*>', page)
        assert match, splitter
        tag = match.group(0)
        assert 'role="separator"' in tag
        assert 'tabindex="0"' in tag
        assert f'aria-orientation="{orientation}"' in tag
        assert f'aria-controls="{controls}"' in tag


def test_split_supports_custom_labels_variables_and_a_stacked_details_pane():
    script = r"""
const fs = require("fs"), vm = require("vm");
class ClassList {
  constructor() { this.values = new Set(); }
  toggle(name, force) {
    const on = force === undefined ? !this.values.has(name) : Boolean(force);
    if (on) this.values.add(name); else this.values.delete(name);
    return on;
  }
  add(...names) { for (const name of names) this.values.add(name); }
  remove(...names) { for (const name of names) this.values.delete(name); }
  contains(name) { return this.values.has(name); }
}
class Element {
  constructor() {
    this.classList = new ClassList();
    this.attrs = {};
    this.listeners = {};
    this.style = { values: {}, setProperty(name, value) { this.values[name] = value; } };
    this.title = "";
    this.tabIndex = 0;
  }
  setAttribute(name, value) { this.attrs[name] = String(value); }
  getAttribute(name) { return name in this.attrs ? this.attrs[name] : null; }
  addEventListener(type, handler) { (this.listeners[type] = this.listeners[type] || []).push(handler); }
  removeEventListener(type, handler) { this.listeners[type] = (this.listeners[type] || []).filter(item => item !== handler); }
}
const document = { body: new Element(), addEventListener() {}, removeEventListener() {} };
const window = {
  Studio: { store: { values: {}, get(key, fallback) { return key in this.values ? this.values[key] : fallback; }, set(key, value) { this.values[key] = value; } } },
  ResizeObserver: class { observe() {} disconnect() {} },
};
const sandbox = { window, document, console, requestAnimationFrame: task => { task(); return 1; },
  cancelAnimationFrame() {} };
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(process.argv[1], "utf8"), sandbox);
const root = new Element();
root.clientWidth = 900; root.clientHeight = 700;
const splitter = new Element();
const layout = sandbox.window.Studio.split.create({
  root, key: "details-test", storage: sandbox.window.Studio.store, mode: "stacked", gap: 6,
  centerMinHeight: 220, sidebar: { enabled: false },
  agent: { splitter, minHeight: 120, maxHeight: 560, defaultHeight: 240,
    label: "Details panel", heightVariable: "--details-height" },
});
const view = layout.read();
layout.reset();
const reset = layout.read();
console.log(JSON.stringify({
  stacked: view.stacked, sidebar: view.sidebar.visible, size: view.agent.size,
  height: root.style.values["--details-height"],
  orientation: splitter.getAttribute("aria-orientation"),
  text: splitter.getAttribute("aria-valuetext"), title: splitter.title,
  noSidebar: root.classList.contains("no-sidebar"),
  reset: { stacked: reset.stacked, size: reset.agent.size,
    orientation: splitter.getAttribute("aria-orientation") },
}));
"""
    result = node_script(script, WEB / "js/split.js")
    assert result["stacked"] is True
    assert result["sidebar"] is False
    assert result["size"] == 240
    assert result["height"] == "240px"
    assert result["orientation"] == "horizontal"
    assert result["text"] == "Details panel 240 pixels"
    assert "Details panel" in result["title"]
    assert result["noSidebar"] is True
    assert result["reset"] == {"stacked": True, "size": 240, "orientation": "horizontal"}


def test_split_fit_shrinks_then_folds_panes():
    script = r"""
const fs = require("fs"), vm = require("vm");
const sandbox = { window: { Studio: {} }, console };
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(process.argv[1], "utf8"), sandbox);
const fit = sandbox.window.Studio.split.fit;
const pane = (size, min, max) => ({ size, min, max, collapsed: false });
const shape = result => [result.first.size, result.second.size, result.first.visible, result.second.visible];
console.log(JSON.stringify({
  wide: shape(fit(1600, 6, pane(252, 190, 420), pane(380, 300, 680), 320)),
  tight: shape(fit(900, 6, pane(420, 190, 420), pane(680, 300, 680), 320)),
  tiny: shape(fit(600, 6, pane(420, 190, 420), pane(680, 300, 680), 320)),
  folded: shape(fit(900, 6, { size: 252, min: 190, max: 420, collapsed: true },
    pane(380, 300, 680), 320)),
}));
"""
    result = node_script(script, WEB / "js/split.js")
    assert result == {
        "wide": [252, 380, True, True],
        "tight": [268, 300, True, True],
        "tiny": [190, 0, True, False],
        "folded": [0, 380, False, True],
    }


def test_split_panes_drag_snap_expand_and_stack():
    script = r"""
const fs = require("fs"), vm = require("vm");
class ClassList {
  constructor() { this.values = new Set(); }
  toggle(name, force) {
    const on = force === undefined ? !this.values.has(name) : Boolean(force);
    if (on) this.values.add(name); else this.values.delete(name);
    return on;
  }
  add(...names) { for (const name of names) this.values.add(name); }
  remove(...names) { for (const name of names) this.values.delete(name); }
  contains(name) { return this.values.has(name); }
}
class Element {
  constructor() {
    this.classList = new ClassList();
    this.attrs = {};
    this.listeners = {};
    this.style = { values: {}, setProperty(name, value) { this.values[name] = value; } };
    this.title = "";
    this.tabIndex = 0;
  }
  setAttribute(name, value) { this.attrs[name] = String(value); }
  getAttribute(name) { return name in this.attrs ? this.attrs[name] : null; }
  addEventListener(type, handler) { (this.listeners[type] = this.listeners[type] || []).push(handler); }
  removeEventListener(type, handler) { this.listeners[type] = (this.listeners[type] || []).filter(item => item !== handler); }
  dispatch(type, event) {
    for (const handler of this.listeners[type] || []) {
      handler({ ...event, currentTarget: this, target: this, preventDefault() {}, stopPropagation() {} });
    }
  }
}
const documentElement = new Element();
const document = {
  body: new Element(),
  addEventListener: (type, handler) => documentElement.addEventListener(type, handler),
  removeEventListener: (type, handler) => documentElement.removeEventListener(type, handler),
};
const window = {
  Studio: { store: { values: {}, get(key, fallback) { return key in this.values ? this.values[key] : fallback; }, set(key, value) { this.values[key] = value; } } },
  ResizeObserver: class { observe() {} disconnect() {} },
  addEventListener() {},
  removeEventListener() {},
};
const sandbox = { window, document, console, requestAnimationFrame: task => { task(); return 1; },
  cancelAnimationFrame() {} };
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(process.argv[1], "utf8"), sandbox);
const Studio = window.Studio;
const root = new Element();
root.clientWidth = 1200; root.clientHeight = 700;
const sidebarSplitter = new Element();
const agentSplitter = new Element();
const layout = Studio.split.create({
  root, key: "test-layout", storage: Studio.store, gap: 6,
  centerMin: 320, centerMinHeight: 240, stackAt: 900,
  sidebar: { splitter: sidebarSplitter, min: 190, max: 420, defaultSize: 252 },
  agent: { splitter: agentSplitter, min: 300, max: 680, defaultSize: 380,
    minHeight: 220, maxHeight: 720, defaultHeight: 340 },
});
const pointer = (x, y) => ({ button: 0, clientX: x, clientY: y, preventDefault() {} });
const view = () => {
  const item = layout.read();
  return { stacked: item.stacked, sidebar: item.sidebar.visible, agent: item.agent.visible,
    sidebarSize: item.sidebar.size, agentSize: item.agent.size,
    sidebarCss: root.style.values["--ide-sidebar-size"], agentCss: root.style.values["--ide-agent-size"],
    agentHeight: root.style.values["--ide-agent-height"],
    collapsedClass: root.classList.contains("no-sidebar"),
    orientation: agentSplitter.getAttribute("aria-orientation") };
};
const initial = view();
sidebarSplitter.dispatch("pointerdown", pointer(100, 100));
documentElement.dispatch("pointermove", pointer(180, 100));
documentElement.dispatch("pointerup", pointer(180, 100));
const dragged = view();
sidebarSplitter.dispatch("pointerdown", pointer(100, 100));
documentElement.dispatch("pointermove", pointer(-200, 100));
documentElement.dispatch("pointerup", pointer(-200, 100));
const snapped = view();
sidebarSplitter.dispatch("pointerdown", pointer(100, 100));
documentElement.dispatch("pointermove", pointer(90, 100));
documentElement.dispatch("pointerup", pointer(90, 100));
const wrongWay = view();
sidebarSplitter.dispatch("keydown", { key: "Enter", shiftKey: false, preventDefault() {} });
const restored = view();
layout.setMode("stacked");
const stacked = view();
root.clientHeight = 400;
layout.apply();
const short = view();
console.log(JSON.stringify({ initial, dragged, snapped, wrongWay, restored, stacked, short,
  stored: Studio.store.values["test-layout"] }));
"""
    result = node_script(script, WEB / "js/split.js")
    assert result["initial"] == {
        "stacked": False, "sidebar": True, "agent": True, "sidebarSize": 252, "agentSize": 380,
        "sidebarCss": "252px", "agentCss": "380px", "agentHeight": "340px",
        "collapsedClass": False, "orientation": "vertical",
    }
    assert result["dragged"]["sidebarSize"] == 332
    assert result["dragged"]["sidebarCss"] == "332px"
    assert result["snapped"]["sidebar"] is False
    assert result["snapped"]["collapsedClass"] is True
    assert result["wrongWay"]["sidebar"] is False
    assert result["restored"]["sidebar"] is True
    assert result["stacked"]["stacked"] is True
    assert result["stacked"]["orientation"] == "horizontal"
    assert result["short"]["agent"] is False
    assert result["stored"]["sidebarCollapsed"] is False


def test_ide_voice_and_token_meter_have_accessible_controls():
    page = render_index()
    for dom_id in ("ide-voice", "ide-voice-auto", "ide-budget-track", "ide-budget-fill",
                   "ide-budget-open", "ide-budget-text", "ide-status-tokens"):
        assert f'id="{dom_id}"' in page, dom_id
    assert 'id="ide-budget-track" class="ide-budget-track" role="meter"' in page
    assert 'aria-label="Context used"' in page


def test_tokens_module_matches_the_server_estimate():
    script = r"""
const fs = require("fs"), vm = require("vm");
const sandbox = { window: { Studio: {} }, console, TextEncoder };
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(process.argv[1], "utf8"), sandbox);
const tokens = sandbox.window.Studio.tokens;
console.log(JSON.stringify({
  empty: tokens.estimate(""),
  ascii: tokens.estimate("hello world"),
  wide: tokens.estimate("你好世界"),
  bytes: tokens.bytes("hé"),
}));
"""
    result = node_script(script, WEB / "js/tokens.js")
    assert result == {"empty": 0, "ascii": 3, "wide": 3, "bytes": 3}


def test_voice_module_strips_markdown_for_narration():
    script = r"""
const fs = require("fs"), vm = require("vm");
const sandbox = { window: { Studio: {} }, console,
  navigator: { language: "en-US" },
  document: { documentElement: { lang: "en-US" } } };
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(process.argv[1], "utf8"), sandbox);
const voice = sandbox.window.Studio.voice;
console.log(JSON.stringify({
  text: voice.plain("## Hi **there** `code` [link](https://example.test)"),
  browser: voice.browserSupported(),
  speech: voice.synthesisSupported(),
}));
"""
    result = node_script(script, WEB / "js/voice.js")
    assert result == {"text": "Hi there code link", "browser": False, "speech": False}


def test_every_static_dom_lookup_has_exactly_one_target():
    parser = parse_page()
    ids = set(parser.ids)
    for source in WEB.glob("js/**/*.js"):
        text = source.read_text(encoding="utf-8")
        for dom_id in re.findall(r"\$\([\"']([a-z][a-z0-9-]+)[\"']\)", text):
            assert dom_id in ids, f"{source.relative_to(WEB)} references missing #{dom_id}"


@pytest.mark.parametrize("source", sorted(WEB.glob("js/**/*.js"), key=lambda path: str(path)))
def test_javascript_parses(source):
    if not shutil.which("node"):
        pytest.skip("Node.js is not installed")
    subprocess.run(["node", "--check", str(source)], capture_output=True, text=True, timeout=20, check=True)


def test_shortcuts_keep_every_view_handler_for_the_same_combo():
    script = r"""
const fs = require("fs"), vm = require("vm");
const document = { addEventListener(type, handler) { this.handler = handler; }, hidden: false };
const sandbox = { window: { Studio: {} }, document, console };
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(process.argv[1], "utf8"), sandbox);
const Studio = sandbox.window.Studio;
let first = 0, second = 0, prevented = 0;
Studio.shortcut("mod+s", () => { first += 1; return false; });
Studio.shortcut("mod+s", () => { second += 1; });
function press(key, extra = {}) {
  const event = { key, ctrlKey: true, metaKey: false, altKey: false, shiftKey: false,
    target: {}, preventDefault() { prevented += 1; }, ...extra };
  document.handler(event);
}
press("s");
const afterSave = { first, second, prevented };
Studio.shortcut("mod+x", () => false);
press("x");
console.log(JSON.stringify({ afterSave, afterOptOut: prevented }));
"""
    result = node_script(script, WEB / "js/events.js")
    assert result == {
        "afterSave": {"first": 1, "second": 1, "prevented": 1},
        "afterOptOut": 1,
    }


def test_modal_reuses_the_open_panel_and_removes_it_on_close():
    script = r"""
const fs = require("fs"), vm = require("vm");
class Element {
  constructor(tag) {
    this.tag = tag; this.children = []; this.parent = null; this.open = false;
    this.dataset = {}; this.listeners = {}; this.className = ""; this.returnValue = "";
  }
  append(...nodes) {
    for (const node of nodes) {
      if (node === null || node === undefined) continue;
      node.parent = this; this.children.push(node);
    }
  }
  addEventListener(type, handler) { (this.listeners[type] = this.listeners[type] || []).push(handler); }
  removeEventListener() {}
  showModal() { this.open = true; }
  close(value) {
    if (!this.open) return;
    this.open = false; this.returnValue = value || "";
    for (const handler of this.listeners.close || []) handler();
  }
  remove() {
    if (!this.parent) return;
    const index = this.parent.children.indexOf(this);
    if (index >= 0) this.parent.children.splice(index, 1);
    this.parent = null;
  }
  focus() { this.focused = true; }
}
const body = new Element("body");
const document = { readyState: "loading", body, addEventListener() {}, activeElement: null };
function h(tag, attributes, ...children) { const element = new Element(tag); Object.assign(element, attributes || {}); element.append(...children); return element; }
const Studio = { $: () => null, $$: () => [], h, icon: () => new Element("svg"), format: {},
  ui: { empty: () => new Element("div") }, api: async () => ({}), query: () => "", state: {}, toast() {} };
const sandbox = { window: { Studio }, document, console, setTimeout, clearTimeout, requestAnimationFrame: fn => fn() };
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(process.argv[1], "utf8"), sandbox);
const first = Studio.overlay.modal({ key: "details:1", title: "Details" });
const second = Studio.overlay.modal({ key: "details:1", title: "Details" });
const other = Studio.overlay.modal({ key: "details:2", title: "Other" });
const afterOther = body.children.length;
first.close(); second.close(); other.close();
Studio.overlay.modal({ key: "details:3", title: "Third" });
Studio.overlay.closeModal();
console.log(JSON.stringify({ same: first.dialog === second.dialog, reused: second.reused,
  afterOther, remaining: body.children.length, focused: first.dialog.focused }));
"""
    result = node_script(script, WEB / "js/overlay.js")
    assert result == {"same": True, "reused": True, "afterOther": 1, "remaining": 0, "focused": True}


def test_markdown_escapes_markup_and_only_allows_safe_links():
    script = r"""
const fs = require("fs"), vm = require("vm");
const sandbox = { window: { Studio: { copy() {} } }, document: { addEventListener() {} }, console };
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(process.argv[1], "utf8"), sandbox);
const render = sandbox.window.Studio.markdown.render;
const hostile = '<script>alert(1)</script> <img src=x onerror=alert(1)> [bad](javascript:alert(1))';
const formatted = render('# Title\n\n**bold** and *italic* and `code`\n\n- one\n- two\n\n```js\n<x>\n```');
console.log(JSON.stringify({ hostile: render(hostile), formatted, link: render('[ok](https://example.com?a=1&b=2)') }));
"""
    result = node_script(script, WEB / "js/markdown.js")
    assert "<script>" not in result["hostile"] and "<img" not in result["hostile"]
    assert 'href="javascript:' not in result["hostile"] and "&lt;script&gt;" in result["hostile"]
    assert "<h1>Title</h1>" in result["formatted"] and "<strong>bold</strong>" in result["formatted"]
    assert "<ul><li>one</li><li>two</li></ul>" in result["formatted"]
    assert "&lt;x&gt;" in result["formatted"] and '<a href="https://example.com?a=1&amp;b=2"' in result["link"]


def test_meter_updates_in_place_for_live_statistics():
    script = r"""
const fs = require("fs"), vm = require("vm");
class Element {
  constructor(tag) { this.tag = tag; this.children = []; this.attrs = {}; this.className = ""; this.style = {}; this.textContent = ""; }
  setAttribute(name, value) { this.attrs[name] = String(value); }
  append(...nodes) { this.children.push(...nodes.filter(node => node !== null && node !== undefined)); }
}
function h(tag, attributes, ...children) {
  const element = new Element(tag);
  for (const [key, value] of Object.entries(attributes || {})) {
    if (key === "class") element.className = value;
    else if (key === "text") element.textContent = value;
    else element[key] = value;
  }
  element.append(...children);
  return element;
}
const Studio = { h, $$: () => [], icon: () => new Element("svg") };
const sandbox = { window: { Studio }, console };
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(process.argv[1], "utf8"), sandbox);
const meter = Studio.ui.meter({ label: "GPU", value: 1, total: 10, text: "1 / 10" });
const track = meter.children[1], fill = track.children[0], head = meter.children[0];
const before = { className: meter.className, width: fill.style.width };
meter.update({ value: 9, total: 10, text: "9 / 10" });
console.log(JSON.stringify({ before, after: { className: meter.className, width: fill.style.width,
  text: head.children[1].textContent, now: track.attrs["aria-valuenow"] }, same: meter.children.length === 2 }));
"""
    result = node_script(script, WEB / "js/ui.js")
    assert result == {
        "before": {"className": "meter", "width": "10%"},
        "after": {"className": "meter warning", "width": "90%", "text": "9 / 10", "now": "90"},
        "same": True,
    }


def test_agents_run_panel_follows_the_amp_event_stream():
    source = (WEB / "js/views/agents.js").read_text(encoding="utf-8")
    assert "Studio.subscribe" in source
    assert "/events?since=" in source
    assert "pollRun" not in source
    assert "stopWatching" in source


def test_agent_dragging_updates_in_place_once_per_frame():
    source = (WEB / "js/views/agents.js").read_text(encoding="utf-8")
    move = re.search(r"function onPointerMove\(event\) \{(.*?)\n  \}\n", source, re.S)
    assert move, "onPointerMove not found"
    assert 'frame("drag", paintNodeDrag)' in move.group(1)
    assert 'frame("selection", paintSelection)' in move.group(1)
    assert "renderNodes(); renderEdges();" not in move.group(1)
    paint = re.search(r"function paintNodeDrag\(\) \{(.*?)\n  \}\n", source, re.S)
    assert paint, "paintNodeDrag not found"
    assert "updateEdgesTouching" in paint.group(1)
    assert "drawMinimapNow" in paint.group(1)
    assert "translate(${node.x}px, ${node.y}px)" in source


def test_agent_selection_and_nudging_update_without_rebuilding_the_canvas():
    source = (WEB / "js/views/agents.js").read_text(encoding="utf-8")
    for name in ("selectNode", "selectAllNodes", "nudgeSelection"):
        body = re.search(rf"function {name}\(.*?\n  \}}\n", source, re.S)
        assert body, f"{name} not found"
        assert "renderAll()" not in body.group(0), f"{name} still rebuilds the whole designer"
    move = re.search(r"function onPointerMove\(event\) \{(.*?)\n  \}\n", source, re.S)
    assert move, "onPointerMove not found"
    assert "/ 20) * 20" not in move.group(1), "dragging should follow the pointer, not a 20px grid"


def test_agent_connections_have_a_wide_target_and_cancel_cleanly():
    source = (WEB / "js/views/agents.js").read_text(encoding="utf-8")
    style = (WEB / "css/agents.css").read_text(encoding="utf-8")
    assert 'class: "agent-edge-hit"' in source
    assert ".agent-edge-hit {" in style and "pointer-events: stroke" in style
    assert ".agent-edge-hit:hover + .agent-edge" in style
    assert "function connectionTarget" in source and 'node.type !== "start"' in source
    assert "function cancelConnection" in source
    assert source.count("cancelConnection()") >= 3, "Escape, blur and view changes must clear a half-drawn connection"


def test_agent_designer_never_shadows_the_browser_document():
    source = (WEB / "js/views/agents.js").read_text(encoding="utf-8")
    assert not re.search(r"\b(?:const|let|var|function)\s+document\b", source), \
        "A local named 'document' shadows the DOM and breaks querySelector calls"
    assert 'document.querySelector(".agent-node-rename")' in source


def test_plain_text_run_input_becomes_a_message_for_agent_templates():
    script = r"""
const fs = require("fs"), vm = require("vm");
const Studio = { views: { register() {} } };
const sandbox = { window: { Studio }, document: {}, console };
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(process.argv[1], "utf8"), sandbox);
const parse = Studio.agents.parseRunInput;
console.log(JSON.stringify({
  plain: parse("top 5 world news"),
  quoted: parse('"hello"'),
  object: parse('{"message":"hi","count":2}'),
  number: parse("42"),
  list: parse("[1,2]")
}));
"""
    result = node_script(script, WEB / "js/views/agents.js")
    assert result == {
        "plain": {"message": "top 5 world news"},
        "quoted": {"message": "hello"},
        "object": {"message": "hi", "count": 2},
        "number": 42,
        "list": [1, 2],
    }


def test_agent_designer_exposes_quick_add_find_and_selection_tooling():
    page = render_index()
    for dom_id in (
        "agents-add-node", "agents-find-toggle", "agents-pan-mode", "agents-guides",
        "agents-find", "agents-selection-toolbar", "agents-status", "agents-run-menu",
    ):
        assert f'id="{dom_id}"' in page, dom_id
    for removed in ("agents-library", "agents-author", "agents-test", "agents-debug"):
        assert f'id="{removed}"' not in page, removed
    source = (WEB / "js/views/agents.js").read_text(encoding="utf-8")
    for marker in (
        "openQuickAdd", "findNodes", "alignPositions", "distributePositions", "nearestEdge",
        "startInlineRename", "showShortcuts", "agent-selection-toolbar", "state.follow",
        "newAgentDialog", "openRunMenu", "PRIMARY_FIELDS", "nodeSummary", "fieldVisible",
        "simpleChain", "conditionDialog",
    ):
        assert marker in source, marker


def test_agent_zoom_keeps_the_point_under_the_cursor():
    script = r"""
const fs = require("fs"), vm = require("vm");
const Studio = { views: { register() {} } };
const sandbox = { window: { Studio }, document: {}, console };
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(process.argv[1], "utf8"), sandbox);
const agents = Studio.agents;
const origin = { left: 50, top: 20 };
const point = { x: 300, y: 250 };
const before = agents.worldPoint(point, { x: 100, y: 50 }, 2, origin);
const pan = agents.zoomedPan({ x: 100, y: 50 }, 2, 1.5, point, origin);
const after = agents.worldPoint(point, pan, 1.5, origin);
console.log(JSON.stringify({ before, pan, after }));
"""
    result = node_script(script, WEB / "js/views/agents.js")
    assert result["before"] == {"x": 75.0, "y": 90.0}
    assert result["after"] == result["before"]
    assert result["pan"] == {"x": 137.5, "y": 95.0}


def test_agent_geometry_search_and_layout_helpers_are_correct():
    script = r"""
const fs = require("fs"), vm = require("vm");
const Studio = { views: { register() {} } };
const sandbox = { window: { Studio }, document: {}, console };
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(process.argv[1], "utf8"), sandbox);
const agents = Studio.agents;
const nodes = [
  { id: "a", type: "start", label: "Start", x: 0, y: 0, data: {} },
  { id: "b", type: "llm", label: "Model call", x: 220, y: 120, data: { prompt: "summarize this" } },
  { id: "c", type: "end", label: "End", x: 500, y: 300, data: {} }
];
const doc = { nodes, edges: [{ from: "a", to: "b", when: "always" }] };
const middle = agents.edgeMidpoint(doc, 0);
const types = {
  llm: { label: "LLM", description: "Call a model", category: "intelligence" },
  note: { label: "Note", description: "Annotation only", category: "annotation" },
  agent: { label: "AI agent", description: "Uses tools", category: "intelligence" }
};
console.log(JSON.stringify({
  left: agents.alignPositions(nodes, "left").map(item => item.x),
  center: agents.alignPositions(nodes, "hcenter").map(item => item.x),
  distributed: agents.distributePositions(nodes, "x").map(item => item.x),
  foundType: agents.findNodes(nodes, "llm"),
  foundData: agents.findNodes(nodes, "summarize"),
  middle,
  nearEdge: agents.nearestEdge(doc, middle, 4),
  farEdge: agents.nearestEdge(doc, { x: 900, y: 900 }, 4),
  order: agents.rankNodeTypes(types, "").map(entry => entry.type),
  query: agents.rankNodeTypes(types, "model").map(entry => entry.type)
}));
"""
    result = node_script(script, WEB / "js/views/agents.js")
    assert result["left"] == [0, 0, 0]
    assert result["center"] == [250, 250, 250]
    assert result["distributed"] == [0, 250, 500]
    assert result["foundType"] == ["b"] and result["foundData"] == ["b"]
    assert result["middle"] == {"x": 205, "y": 101}
    assert result["nearEdge"] == 0 and result["farEdge"] == -1
    assert result["order"] == ["llm", "note", "agent"]
    assert result["query"] == ["llm"]


def test_chart_log_parser_and_axis_helpers_behave_on_real_metrics():
    script = r"""
const fs = require("fs"), vm = require("vm");
const sandbox = { window: { Studio: {} }, console };
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(process.argv[1], "utf8"), sandbox);
const viz = sandbox.window.Studio.trainingViz;
const parsed = viz.parseTrainingMetrics([
  "Tokenizer parameters: 10",
  "step 1/4 | total 3.5 | lr 0.001 | tok/s 120",
  "step 2/4 | total 2.5 | lr 0.001 | tok/s 130",
  "Generator trainable parameters: 20",
  "step 1/4 | total 1.75 | pos/s 90",
  "step 1/4 | total 1.75 | pos/s 90"
]);
console.log(JSON.stringify({ parsed, ticks: viz.niceTicks(0, 10), summary: viz.summarize({ latest: { metrics: { total: 1.75, lr: 0.001 }, throughput: 90, throughput_unit: "pos/s" }, best: 1.75 }) }));
"""
    result = node_script(script, WEB / "js/training-viz.js")
    assert result["parsed"]["points"] == [
        {"step": 1, "loss": 3.5, "stage": "tokenizer"}, {"step": 2, "loss": 2.5, "stage": "tokenizer"},
        {"step": 1, "loss": 1.75, "stage": "generator"},
    ]
    assert result["parsed"]["bestLoss"] == 1.75 and result["parsed"]["throughput"] == 90
    assert result["ticks"][0] == 0 and result["ticks"][-1] == 10
    assert result["summary"]["loss"] == "1.750" and result["summary"]["learningRate"] == "1.00e-3"


def test_editor_indent_keeps_the_document_intact():
    script = r"""
const fs = require("fs"), vm = require("vm");
class Element {
  constructor(tag) {
    this.tag = tag; this.children = []; this.listeners = {};
    this.classList = { toggle() {} }; this.style = {}; this.textContent = "";
    this.value = ""; this.selectionStart = 0; this.selectionEnd = 0;
    this.scrollTop = 0; this.scrollLeft = 0; this.clientHeight = 0; this.readOnly = false;
  }
  append(...nodes) { this.children.push(...nodes.filter(node => node !== null && node !== undefined)); }
  replaceChildren(...nodes) { this.children = []; this.append(...nodes); }
  addEventListener(type, handler) { (this.listeners[type] = this.listeners[type] || []).push(handler); }
  setRangeText(text, start, end, mode) {
    this.value = this.value.slice(0, start) + text + this.value.slice(end);
    const caret = start + text.length;
    this.selectionStart = mode === "select" ? start : caret;
    this.selectionEnd = caret;
  }
  focus() {}
}
function h(tag, attributes, ...children) {
  const element = new Element(tag);
  Object.assign(element, attributes || {});
  element.append(...children);
  return element;
}
const Studio = { h };
const sandbox = { window: { Studio }, console };
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(process.argv[1], "utf8"), sandbox);
const host = new Element("div");
const editor = Studio.editor.create(host, { value: "one\ntwo\nthree" });
const input = editor.input;
function press(key, shiftKey) {
  for (const handler of input.listeners.keydown || []) {
    handler({ key, shiftKey, isComposing: false, ctrlKey: false, metaKey: false, preventDefault() {} });
  }
}
input.selectionStart = 0;
input.selectionEnd = 4;  // ends at the start of the second line
press("Tab", false);
const indented = editor.value;
press("Tab", true);
console.log(JSON.stringify({ indented, restored: editor.value }));
"""
    result = node_script(script, WEB / "js/editor.js")
    assert result == {"indented": "    one\ntwo\nthree", "restored": "one\ntwo\nthree"}


def test_highlight_tokenizer_covers_popular_languages_and_keeps_state():
    script = r"""
const fs = require("fs"), vm = require("vm");
const sandbox = { window: { Studio: {} }, console };
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(process.argv[1], "utf8"), sandbox);
const h = sandbox.window.Studio.highlight;
const kinds = (text, language, state) => h.scanLine(text, state || null, language).tokens
  .map(token => text.slice(token.start, token.end) + ":" + token.type);
const inside = h.scanLine("still a comment", h.scanLine("/* open", null, "javascript").state, "javascript");
const openedFence = h.scanLine("```js", null, "markdown");
const fenced = h.scanLine("const x = 1;", openedFence.state, "markdown");
console.log(JSON.stringify({
  js: kinds("const answer = 42; // yes", "javascript"),
  py: kinds("def run(value):", "python"),
  html: kinds('<a href="/x">link</a>', "markup"),
  heading: h.scanLine("## Title", null, "markdown").tokens.map(token => token.type),
  css: kinds(".card { color: #fff; }", "css"),
  sql: kinds("SELECT * FROM users", "sql"),
  toml: kinds('name = "mariana"', "toml"),
  fence: { kind: openedFence.state.kind, marker: openedFence.state.marker, language: openedFence.state.language },
  fenceBody: fenced.tokens.map(token => "const x = 1;".slice(token.start, token.end) + ":" + token.type),
  comment: inside.tokens.map(token => token.type),
  resolve: [
    h.resolve("Python"), h.resolve("C++"), h.resolve("Unknown", "src/main.rs"),
    h.resolve("", "Dockerfile"), h.resolve("Plain text", "src/main.py"),
    h.resolve("JSON with Comments"), h.resolve("TypeScript", "x.tsx"),
    h.resolve("SystemVerilog"), h.resolve("CUDA"), h.resolve("GLSL")
  ]
}));
"""
    result = node_script(script, WEB / "js/highlight.js")
    assert "const:keyword" in result["js"] and "42:number" in result["js"] and "// yes:comment" in result["js"]
    assert "def:keyword" in result["py"] and "run:function" in result["py"]
    assert "a:tag" in result["html"] and "href:attr" in result["html"] and '"/x":string' in result["html"]
    assert result["heading"] == ["heading"]
    assert "color:property" in result["css"] and "#fff:constant" in result["css"]
    assert "SELECT:keyword" in result["sql"] and "FROM:keyword" in result["sql"]
    assert "name:property" in result["toml"] and '"mariana":string' in result["toml"]
    assert result["fence"] == {"kind": "fence", "marker": "```", "language": "javascript"}
    assert "const:keyword" in result["fenceBody"] and "1:number" in result["fenceBody"]
    assert result["comment"] == ["comment"]
    assert result["resolve"] == [
        "python", "cpp", "rust", "dockerfile", "text", "jsonc", "typescript", "verilog", "cpp", "glsl",
    ]


def test_highlight_resolves_geospatial_and_mermaid_languages():
    script = r"""
const fs = require("fs"), vm = require("vm");
const sandbox = { window: { Studio: {} }, console };
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(process.argv[1], "utf8"), sandbox);
const h = sandbox.window.Studio.highlight;
console.log(JSON.stringify([
  h.resolve("KML", "map.kml"), h.resolve("", "route.gpx"),
  h.resolve("GeoJSON", "places.geojson"), h.resolve("", "data.jsonl"),
  h.resolve("Mermaid", "flow.mmd"), h.resolve("", "chart.mermaid")
]));
"""
    result = node_script(script, WEB / "js/highlight.js")
    assert result == ["markup", "markup", "json", "json", "mermaid", "mermaid"]


def test_diagrams_parse_and_lay_out_flowcharts_sequences_and_pies():
    script = r"""
const fs = require("fs"), vm = require("vm");
class Element {
  constructor(tag) { this.tagName = tag; this.attrs = {}; this.children = []; this.textContent = ""; }
  setAttribute(key, value) { this.attrs[key] = String(value); }
  append(...children) { this.children.push(...children); }
}
const sandbox = {
  window: { Studio: {} },
  document: { createElementNS(namespace, tag) { return new Element(tag); } },
  console,
};
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(process.argv[1], "utf8"), sandbox);
const diagrams = sandbox.window.Studio.diagrams;
const flow = diagrams.parse("flowchart LR\n  A[Start] --> B{Ready?}\n  B -->|yes| C[Done]\n  B -.-> D[Retry]");
const flowLayout = diagrams.layoutFlowchart(flow);
const sequence = diagrams.parse("sequenceDiagram\n  participant A as Alice\n  participant B as Bob\n  A->>B: Hello\n  B-->>A: Hi\n  Note over A,B: Sync");
const sequenceLayout = diagrams.layoutSequence(sequence);
const pie = diagrams.parse("pie title Traffic\n  \"Direct\" : 60\n  \"Search\" : 40");
const pieLayout = diagrams.layoutPie(pie);
console.log(JSON.stringify({
  nodes: [...flow.nodes.keys()],
  labels: [...flow.nodes.values()].map(node => node.label),
  edges: flow.edges.map(edge => [edge.from, edge.to, edge.label]),
  ranks: flowLayout.nodes.map(node => [node.id, node.rank]),
  horizontal: !flowLayout.vertical && flowLayout.nodes[0].x < flowLayout.nodes[1].x,
  participants: sequence.participants.map(item => item.label),
  messages: sequenceLayout.messages.map(item => [item.from, item.to, item.text]),
  notes: sequenceLayout.notes.map(item => item.text),
  pieTitle: pieLayout.title,
  pieTotal: pieLayout.total,
  rendered: [
    diagrams.render("flowchart TD\n A-->B").tagName,
    diagrams.render("sequenceDiagram\n A->>B: Hi").tagName,
    diagrams.render("pie title X\n \"A\" : 1").tagName,
  ],
  unsupported: diagrams.parse("classDiagram\n  A <|-- B")
}));
"""
    result = node_script(script, WEB / "js/diagrams.js")
    assert result["nodes"] == ["A", "B", "C", "D"]
    assert result["labels"] == ["Start", "Ready?", "Done", "Retry"]
    assert result["edges"] == [["A", "B", ""], ["B", "C", "yes"], ["B", "D", ""]]
    assert result["ranks"] == [["A", 0], ["B", 1], ["C", 2], ["D", 2]]
    assert result["horizontal"] is True
    assert result["participants"] == ["Alice", "Bob"]
    assert result["messages"] == [["A", "B", "Hello"], ["B", "A", "Hi"]]
    assert result["notes"] == ["Sync"]
    assert result["pieTitle"] == "Traffic" and result["pieTotal"] == 100
    assert result["rendered"] == ["svg", "svg", "svg"]
    assert result["unsupported"] is None


def test_editor_autopairs_toggles_comments_and_opens_context_menu():
    script = r"""
const fs = require("fs"), vm = require("vm");
class Element {
  constructor(tag) {
    this.tag = tag; this.children = []; this.listeners = {};
    this.classList = { toggle() {} }; this.style = {}; this.textContent = "";
    this.value = ""; this.selectionStart = 0; this.selectionEnd = 0;
    this.scrollTop = 0; this.scrollLeft = 0; this.clientHeight = 0; this.clientWidth = 0;
    this.readOnly = false;
  }
  append(...nodes) { this.children.push(...nodes.filter(node => node !== null && node !== undefined)); }
  replaceChildren(...nodes) { this.children = []; this.append(...nodes); }
  addEventListener(type, handler) { (this.listeners[type] = this.listeners[type] || []).push(handler); }
  setRangeText(text, start, end, mode) {
    this.value = this.value.slice(0, start) + text + this.value.slice(end);
    const caret = start + text.length;
    this.selectionStart = mode === "select" ? start : caret;
    this.selectionEnd = mode === "select" ? start + text.length : caret;
  }
  setSelectionRange(start, end) { this.selectionStart = start; this.selectionEnd = end; }
  setAttribute(name, value) { this[name] = value; }
  focus() {}
}
function h(tag, attributes, ...children) {
  const element = new Element(tag);
  Object.assign(element, attributes || {});
  element.append(...children);
  return element;
}
const Studio = { h };
const sandbox = { window: { Studio }, console };
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(process.argv[1], "utf8"), sandbox);
vm.runInContext(fs.readFileSync(process.argv[2], "utf8"), sandbox);
const host = new Element("div");
const editor = Studio.editor.create(host, { value: "", language: "Python" });
const input = editor.input;
function press(key, extra) {
  for (const handler of input.listeners.keydown || []) {
    handler({ key, shiftKey: false, ctrlKey: false, metaKey: false, altKey: false,
      isComposing: false, preventDefault() {}, stopPropagation() {}, ...(extra || {}) });
  }
}
press("(");
const paired = [editor.value, input.selectionStart];
press(")");
const skipped = [editor.value, input.selectionStart];
input.selectionStart = 1;
input.selectionEnd = 1;
press("Backspace");
const removed = editor.value;
editor.setValue("word");
input.selectionStart = 0;
input.selectionEnd = 4;
press("(");
const wrapped = [editor.value, input.selectionStart, input.selectionEnd];
editor.setValue("x = 1");
input.selectionStart = 0;
input.selectionEnd = 5;
press("/", { ctrlKey: true });
const commented = editor.value;
press("/", { ctrlKey: true });
const uncommented = editor.value;
let labels = [];
Studio.overlay = { menu(anchor, items) { labels = items.filter(item => item && item.label).map(item => item.label); } };
for (const handler of input.listeners.contextmenu || []) handler({ clientX: 10, clientY: 20, preventDefault() {} });
console.log(JSON.stringify({
  paired, skipped, removed, wrapped, commented, uncommented, labels,
  matches: Studio.editor.findMatches("Alpha beta ALPHA", "alpha", {}).matches.length,
  sensitive: Studio.editor.findMatches("Alpha beta ALPHA", "Alpha", { caseSensitive: true }).matches.length,
  insensitive: Studio.editor.findMatches("Alpha beta ALPHA", "Alpha", {}).matches.length,
  regex: Studio.editor.findMatches("a1 b2", "a\\d", { regex: true }).matches.map(match => match.text),
  detect: [Studio.editor.detectIndent("  a\n  b"), Studio.editor.detectIndent("\ta\n\tb")],
}));
"""
    result = node_script(script, WEB / "js/highlight.js", WEB / "js/editor.js")
    assert result["paired"] == ["()", 1]
    assert result["skipped"] == ["()", 2]
    assert result["removed"] == ""
    assert result["wrapped"] == ["(word)", 1, 5]
    assert result["commented"] == "# x = 1"
    assert result["uncommented"] == "x = 1"
    assert {"Find", "Replace", "Go to line", "Toggle comment", "Change language mode"} <= set(result["labels"])
    assert result["matches"] == 2
    assert result["sensitive"] == 1
    assert result["insensitive"] == 2
    assert result["regex"] == ["a1"]
    assert result["detect"] == [
        {"mode": "spaces", "size": 2}, {"mode": "tab", "size": 4},
    ]


def test_editor_paints_syntax_and_bracket_decorations():
    script = r"""
const fs = require("fs"), vm = require("vm");
class TextNode { constructor(value) { this.textContent = value; this.isText = true; } }
class Element {
  constructor(tag) {
    this.tag = tag; this.children = []; this.listeners = {};
    this.classList = { toggle() {} }; this.style = {}; this.textContent = "";
    this.value = ""; this.selectionStart = 0; this.selectionEnd = 0;
    this.scrollTop = 0; this.scrollLeft = 0; this.clientHeight = 400; this.clientWidth = 800;
    this.readOnly = false; this.className = "";
  }
  append(...nodes) {
    for (const node of nodes) {
      if (node === null || node === undefined) continue;
      if (node.isFragment) this.append(...node.children);
      else this.children.push(node);
    }
  }
  replaceChildren(...nodes) { this.children = []; this.append(...nodes); }
  addEventListener(type, handler) { (this.listeners[type] = this.listeners[type] || []).push(handler); }
  setRangeText(text, start, end, mode) {
    this.value = this.value.slice(0, start) + text + this.value.slice(end);
    const caret = start + text.length;
    this.selectionStart = mode === "select" ? start : caret;
    this.selectionEnd = caret;
  }
  setSelectionRange(start, end) { this.selectionStart = start; this.selectionEnd = end; }
  setAttribute(name, value) { this[name] = value; }
  focus() {}
}
const document = {
  createElement: tag => new Element(tag),
  createTextNode: value => new TextNode(value),
  createDocumentFragment: () => { const fragment = new Element("#fragment"); fragment.isFragment = true; return fragment; },
};
function h(tag, attributes, ...children) {
  const element = new Element(tag);
  Object.assign(element, attributes || {});
  element.append(...children);
  return element;
}
const Studio = { h };
const sandbox = { window: { Studio }, document, console };
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(process.argv[1], "utf8"), sandbox);
vm.runInContext(fs.readFileSync(process.argv[2], "utf8"), sandbox);
const host = new Element("div");
const editor = Studio.editor.create(host, { value: "def run(value):\n    return [1]\n", language: "Python" });
const input = editor.input;
input.selectionStart = input.selectionEnd = "def run(value):\n    return [".length;
for (const handler of input.listeners.keyup || []) handler({ key: "ArrowRight" });
function classes(node, found) {
  if (node.isText) return found;
  if (node.className) for (const name of node.className.split(/\s+/)) if (name) found.push(name);
  for (const child of node.children || []) classes(child, found);
  return found;
}
const found = classes(host, []);
console.log(JSON.stringify({
  keyword: found.includes("tok-keyword"),
  function: found.includes("tok-function"),
  number: found.includes("tok-number"),
  brackets: found.filter(name => name === "bracket-match").length,
}));
"""
    result = node_script(script, WEB / "js/highlight.js", WEB / "js/editor.js")
    assert result == {"keyword": True, "function": True, "number": True, "brackets": 2}


def test_editor_round_trips_crlf_line_endings():
    script = r"""
const fs = require("fs"), vm = require("vm");
const sandbox = { window: { Studio: {} }, console };
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(process.argv[1], "utf8"), sandbox);
const editor = sandbox.window.Studio.editor;
console.log(JSON.stringify({
  crlf: editor.detectEol("a\r\nb\r\n"),
  lf: editor.detectEol("a\nb\n"),
  mixed: editor.detectEol("a\r\nb\n"),
  encoded: editor.encodeEol("a\nb\n", "\r\n"),
  encodedLf: editor.encodeEol("a\nb\n", "\n")
}));
"""
    result = node_script(script, WEB / "js/editor.js")
    assert result == {
        "crlf": "\r\n", "lf": "\n", "mixed": "\n",
        "encoded": "a\r\nb\r\n", "encodedLf": "a\nb\n",
    }


def test_form_values_are_typed_for_shared_controls():
    script = r"""
const fs = require("fs"), vm = require("vm");
const sandbox = { window: { Studio: {} }, console };
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(process.argv[1], "utf8"), sandbox);
const form = sandbox.window.Studio.form;
console.log(JSON.stringify({
  number: form.value({ type: "number", value: "12.5" }),
  blank: form.value({ type: "number", value: "" }),
  checked: form.value({ type: "checkbox", checked: true }),
  text: form.value({ type: "text", value: "hello" })
}));
"""
    result = node_script(script, WEB / "js/form.js")
    assert result == {"number": 12.5, "blank": 0, "checked": True, "text": "hello"}


def test_ide_can_hide_recent_projects_without_deleting_them():
    source = (WEB / "js/views/ide.js").read_text(encoding="utf-8")
    assert "ide-hidden-workspaces" in source
    assert "Manage recent projects" in source
    assert "function openProjectManager" in source
    assert "hiddenProjects" in source


def test_ide_exposes_source_render_hex_and_metadata_views():
    page = render_index()
    assert 'id="ide-view-bar"' in page
    source = (WEB / "js/views/ide.js").read_text(encoding="utf-8")
    for marker in (
        "previewUrl", "defaultView", "mountPreview", "renderPreview", "syntaxPre",
        "openFileInfo", "ide-preview", "ide-hex", "ide-csv",
    ):
        assert marker in source, marker
    scripts = re.findall(r'<script src="([^"]+)"', page)
    assert scripts.index("/js/markdown.js") < scripts.index("/js/diagrams.js") < scripts.index("/js/views/ide.js")


def test_ide_context_ring_skills_editor_and_swarm_cards():
    page = render_index()
    assert page.count('id="ide-budget-track"') == 1
    assert 'id="ide-budget-track" class="ide-budget-track" role="meter"' in page
    assert 'class="ide-budget-ring"' in page and '<circle id="ide-budget-fill"' in page
    source = (WEB / "js/views/ide.js").read_text(encoding="utf-8")
    for marker in (
        "BUDGET_RING", "stroke-dashoffset", "openSkillEditor", "removeSkill",
        "context_consolidated", "Consolidate now", "renderSwarmEntry",
        "swarm_agent_start", "swarm_agent_end",
    ):
        assert marker in source, marker
    css = (WEB / "css/ide.css").read_text(encoding="utf-8")
    assert ".ide-budget-ring" in css and ".ide-swarm-agent" in css


def test_ide_context_ring_sits_in_the_composer_footer_and_session_can_be_cleaned():
    page = render_index()
    assert "Ctrl+Enter sends · Ctrl+P quick open" not in page
    assert page.count('class="ide-budget"') == 1
    assert page.count('id="ide-budget-track"') == 1
    assert page.count('id="ide-open-folder"') == 1
    assert 'id="ide-import-workspace"' not in page
    # The ring replaces the old shortcut text below the composer.
    assert page.index('<textarea id="ide-message"') < page.index('id="ide-budget-track"')
    source = (WEB / "js/views/ide.js").read_text(encoding="utf-8")
    for marker in ("cleanSession", "Clean session…", "Clean IDE session…",
                   "ide-hidden-workspaces", "removePrefix", "Open in this session",
                   "All projects on disk", "openFolder", "removeWorkspace",
                   "/api/ide/workspaces/open", "Import a copy…", "Delete copy"):
        assert marker in source, marker
    storage = (WEB / "js/storage.js").read_text(encoding="utf-8")
    assert "remove(key)" in storage and "removePrefix(prefix)" in storage
    css = (WEB / "css/ide.css").read_text(encoding="utf-8")
    assert ".ide-context-heading" in css and ".ide-context-details" in css


def test_chat_exposes_attachments_voice_meter_and_artifact_panel():
    page = render_index()
    for dom_id in (
        "chat-attach", "chat-file-input", "chat-attachments", "chat-voice", "chat-voice-auto",
        "chat-prompts", "chat-budget-track", "chat-budget-fill", "chat-budget-open", "chat-budget-text",
        "chat-preview-toggle", "chat-preview", "chat-preview-list", "chat-preview-body",
        "chat-preview-copy", "chat-preview-download", "chat-preview-close",
        "chat-mode", "chat-mode-solo", "chat-mode-swarm",
    ):
        assert f'id="{dom_id}"' in page, dom_id
    match = re.search(r'<div id="chat-split-preview"[^>]*>', page)
    assert match
    tag = match.group(0)
    assert 'role="separator"' in tag and 'tabindex="0"' in tag
    assert 'aria-orientation="vertical"' in tag and 'aria-controls="chat-preview"' in tag
    assert 'id="chat-budget-track" class="chat-budget-track" role="meter"' in page
    scripts = re.findall(r'<script src="([^"]+)"', page)
    assert scripts.index("/js/artifacts.js") < scripts.index("/js/views/chat.js")
    assert scripts.index("/js/swarm.js") < scripts.index("/js/views/chat.js")
    assert scripts.index("/js/markdown.js") < scripts.index("/js/swarm.js")


def test_chat_starts_in_swarm_mode_and_offers_a_solo_retry():
    page = render_index()
    assert re.search(r'<button id="chat-mode-solo"[^>]*aria-pressed="false"', page)
    assert re.search(r'<button id="chat-mode-swarm"[^>]*aria-pressed="true"', page)
    source = (WEB / "js/views/chat.js").read_text(encoding="utf-8")
    assert 'Studio.store.get("chat-mode", "swarm")' in source
    assert "Retry in Solo" in source
    assert "agent-workspace/chats/" in source


def test_chat_swarm_tools_are_scoped_and_opt_in():
    source = (WEB / "js/views/chat.js").read_text(encoding="utf-8")
    assert 'Studio.store.get("chat-tools"' in source
    assert "agent-workspace/files/chats/" in source
    assert "Read files" in source and "Edit files" in source
    assert "Fetch public web pages" in source
    assert "tools: { ...state.tools }" in source
    page = render_index()
    assert 'id="chat-mode-swarm"' in page


def test_artifact_module_finds_previewable_chat_blocks():
    script = r"""
const fs = require("fs"), vm = require("vm");
const sandbox = { window: { Studio: {} }, console };
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(process.argv[1], "utf8"), sandbox);
const artifacts = sandbox.window.Studio.artifacts;
const message = {
  role: "assistant",
  content: "Here:\n```html\n<title>Card</title><style>body{color:red}</style><div>x</div>\n```\n"
    + "```python\nprint(1)\n```\n```mermaid\ngraph TD;\nA-->B;\n```\n",
};
const items = artifacts.messageArtifacts(message, 3).map(item => ({
  id: item.id, kind: item.kind, title: item.title, previewable: item.previewable, language: item.language,
}));
const attachments = artifacts.fromAttachments([
  { id: "a1", name: "notes.md", kind: "text", text: "# Hi", language: "markdown", size: 4 },
  { id: "a2", name: "photo.png", kind: "image", data_url: "data:image/png;base64,AAAA", size: 3 },
], "draft", -1).map(item => ({ kind: item.kind, title: item.title, previewable: item.previewable }));
const filename = artifacts.filename({
  title: "Card", filename: "../../etc/passwd.html", kind: "html", language: "html",
});
const crlf = artifacts.messageArtifacts({ role: "assistant", content: "```python\r\nprint(1)\r\n```\r\n" }, 4)
  .map(item => ({ kind: item.kind, language: item.language }));
const duplicates = artifacts.fromAttachments([
  { id: "same", name: "a.txt", kind: "text", text: "a" },
  { id: "same", name: "b.txt", kind: "text", text: "b" },
], "m0", 0).map(item => item.id);
console.log(JSON.stringify({ items, attachments, filename, crlf, duplicates }));
"""
    result = node_script(script, WEB / "js/artifacts.js")
    assert result["items"] == [
        {"id": "code:3:0", "kind": "html", "title": "Card", "previewable": True, "language": "html"},
        {"id": "code:3:1", "kind": "code", "title": "python snippet 2", "previewable": False, "language": "python"},
        {"id": "code:3:2", "kind": "mermaid", "title": "Mermaid diagram", "previewable": True, "language": "mermaid"},
    ]
    assert result["attachments"] == [
        {"kind": "markdown", "title": "notes.md", "previewable": True},
        {"kind": "image", "title": "photo.png", "previewable": True},
    ]
    assert result["filename"] == "-.-etc-passwd.html"
    assert result["crlf"] == [{"kind": "code", "language": "python"}]
    assert result["duplicates"] == ["attach:m0:0:same", "attach:m0:1:same"]


def test_chat_registers_with_artifacts_and_shared_namespace():
    if not shutil.which("node"):
        pytest.skip("Node.js is not installed")
    script = r"""
const fs = require("fs"), vm = require("vm");
const Studio = {
  store: { get(key, fallback) { return fallback; }, set() {} },
  views: { register(name, definition) { this.name = name; this.definition = definition; } },
};
const sandbox = { window: { Studio }, console, setTimeout, clearTimeout };
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(process.argv[1], "utf8"), sandbox);
vm.runInContext(fs.readFileSync(process.argv[2], "utf8"), sandbox);
vm.runInContext(fs.readFileSync(process.argv[3], "utf8"), sandbox);
console.log(JSON.stringify({
  view: Studio.views.name,
  title: Studio.views.definition && Studio.views.definition.title,
  artifacts: typeof Studio.artifacts.collect,
  swarm: typeof Studio.swarm.render,
}));
"""
    result = subprocess.run(
        ["node", "-e", script, str(WEB / "js/artifacts.js"), str(WEB / "js/swarm.js"),
         str(WEB / "js/views/chat.js")],
        capture_output=True, text=True, timeout=20, check=True,
    )
    assert json.loads(result.stdout) == {
        "view": "chat", "title": "Chat", "artifacts": "function", "swarm": "function",
    }


def test_swarm_module_tracks_agents_and_survives_snapshots():
    script = r"""
const fs = require("fs"), vm = require("vm");
const sandbox = { window: { Studio: {} }, console };
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(process.argv[1], "utf8"), sandbox);
const swarmModule = sandbox.window.Studio.swarm;
const swarm = swarmModule.create();
swarm.meta.model = "fake";
swarmModule.applyEvent(swarm, {
  type: "swarm_plan", strategy: "peer_review", rounds: 1, notes: "split",
  agents: [{id: "a1", label: "Reader", task: "read", status: "queued"}],
});
swarmModule.applyEvent(swarm, {
  type: "swarm_agent_start", agent: "a1", label: "Reader", role: "researcher",
  task: "read", phase: "work", version: 1,
});
swarmModule.applyEvent(swarm, {type: "swarm_agent_delta", agent: "a1", phase: "work", text: "hel"});
swarmModule.applyEvent(swarm, {type: "swarm_agent_delta", agent: "a1", phase: "work", text: "lo"});
swarmModule.applyEvent(swarm, {
  type: "swarm_agent_end", agent: "a1", status: "done", output: "result",
  score: 4.5, steps: 2, version: 2,
});
swarmModule.applyEvent(swarm, {
  type: "swarm_review", reviewer: "a1", target: "a1", score: 4, issues: ["more proof"],
});
swarmModule.applyEvent(swarm, {
  type: "swarm_message", from: "a1", to: "all", text: "done", message_kind: "status",
});
swarmModule.applySnapshot(swarm, {
  id: "s1", status: "running", strategy: "peer_review", round: 0, requested_rounds: 1,
  phase: "work", leader_model: "fake", model: "fake", reviews: [], messages: [], blackboard: [],
  agents: [{id: "a1", label: "Reader", task: "read", status: "running", output: ""}],
});
swarmModule.finish(swarm, {
  ok: true, status: "completed", messages: 3, memory_count: 1,
  tools: ["file.read", "file.list"],
  agents: [{agent: "a1", label: "Reader", status: "done", output: "result"}],
  reviews: [{reviewer: "a1", target: "a1", score: 4}],
});
let malformed = false;
try {
  const broken = swarmModule.create();
  swarmModule.applyEvent(broken, null);
  swarmModule.applyEvent(broken, {type: "swarm_plan", agents: "not a list"});
  swarmModule.applySnapshot(broken, {
    agents: "bad", reviews: "x", messages: 5, blackboard: "y",
  });
  swarmModule.finish(broken, {agents: "bad", reviews: "x", memory: 2});
  malformed = broken.agents.length === 0 && broken.reviews.length === 0
    && broken.messages.length === 0 && swarmModule.agentState(null) === "queued";
} catch (error) {
  malformed = false;
}
console.log(JSON.stringify({
  agents: swarm.agents.length,
  text: swarm.agents[0].text,
  output: swarm.agents[0].output,
  score: swarm.agents[0].score,
  status: swarm.agents[0].status,
  finished: swarm.meta.finished,
  strategy: swarm.meta.strategy,
  messageCount: swarm.meta.messageCount,
  reviewCount: swarm.meta.reviewCount,
  memoryCount: swarm.meta.memoryCount,
  tools: swarm.meta.tools,
  malformed,
}));
"""
    result = node_script(script, WEB / "js/swarm.js")
    assert result == {
        "agents": 1, "text": "hello", "output": "result", "score": 4.5,
        "status": "done", "finished": True, "strategy": "peer_review",
        "messageCount": 3, "reviewCount": 1, "memoryCount": 1,
        "tools": ["file.read", "file.list"], "malformed": True,
    }


def test_native_dom_append_never_receives_an_array_as_one_argument():
    direct_map = re.compile(r"\.append\(\s*(?!\.\.\.)[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*\.map\(")
    for source in WEB.glob("js/**/*.js"):
        text = source.read_text(encoding="utf-8")
        assert not direct_map.search(text), f"Spread mapped children in {source.relative_to(WEB)}"
        assert not re.search(r"\.append\(\s*\(", text), f"Spread grouped array children in {source.relative_to(WEB)}"


def test_css_uses_tokens_and_web_assets_use_standard_names():
    for source in WEB.glob("css/*.css"):
        if source.name == "tokens.css":
            continue
        assert not re.search(r"#[0-9a-fA-F]{3,8}\b", source.read_text(encoding="utf-8")), source.name
    for source in WEB.glob("**/*"):
        if source.is_file():
            assert source.suffix in {".js", ".css", ".html", ".svg"}
            assert re.fullmatch(r"[a-z0-9.-]+", source.name), source.name


def test_static_assets_are_served_with_csp_and_cannot_escape(tmp_path):
    assert read_asset("/views/chat.html") is None
    assert read_asset("/../pyproject.toml") is None
    assert read_asset("/js/../views/chat.html") is None
    assert read_asset("/js/dom.js")[1].startswith("text/javascript")
    with studio(tmp_path) as (_, base):
        with urllib.request.urlopen(base + "/", timeout=15) as response:
            page = response.read().decode()
            assert response.headers["Content-Security-Policy"] == CONTENT_SECURITY_POLICY
            assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert "Mariana Studio" in page and 'id="chat-view"' in page
        for asset in ("/js/views/agents.js", "/js/views/ide.js", "/js/editor.js", "/js/highlight.js",
                      "/css/views.css", "/css/ide.css", "/img/favicon.svg"):
            with urllib.request.urlopen(base + asset, timeout=15) as response:
                assert response.status == 200 and response.read()
