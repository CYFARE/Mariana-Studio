"use strict";

/* Studio Chat: persistent conversations with any catalog model, plus the
   Chat-only parts Anthropic- and ChatGPT-style workspaces have: attachment
   previews, a resizable code/design artifact panel, dictation, read-aloud,
   prompt presets and a live context meter. */
(function () {
  const Studio = window.Studio;
  const { $, h, api, format, icon } = Studio;

  const MAX_ATTACHMENTS = 8;
  const MAX_IMAGE_BYTES = 4_000_000;
  const MAX_ATTACHMENTS_TOTAL_BYTES = 8_000_000;
  const MAX_TEXT_BYTES = 512_000;
  const MAX_ATTACHMENT_TEXT = 200_000;
  const MAX_PROMPT_ATTACHMENT_CHARS = 20_000;
  const MAX_CONVERSATION_ATTACHMENT_CHARS = 10_000_000;
  const BUDGET_RING = 2 * Math.PI * 9;
  const ASSUMED_CONTEXT_TOKENS = 8192;
  const TEXT_EXTENSIONS = new Set([
    "txt", "text", "md", "markdown", "csv", "tsv", "json", "jsonc", "yaml", "yml",
    "toml", "ini", "cfg", "conf", "env", "html", "htm", "css", "scss", "sass", "less",
    "js", "jsx", "mjs", "cjs", "ts", "tsx", "py", "rb", "php", "go", "rs", "java", "kt",
    "kts", "c", "h", "cpp", "cc", "hpp", "cs", "swift", "dart", "lua", "r", "sql", "sh",
    "bash", "zsh", "fish", "ps1", "bat", "dockerfile", "makefile", "cmake", "xml", "svg",
    "vue", "svelte", "astro", "graphql", "gql", "proto", "tf", "hcl", "nix", "tex",
    "log", "rst", "diff", "patch", "properties", "gradle", "plist", "lock", "gitignore",
    "editorconfig", "gitattributes", "dockerignore", "npmignore", "prettierignore",
    "eslintignore", "babelrc", "npmrc", "yarnrc",
  ]);
  const PROMPTS = [
    ["Explain this", "Explain the following clearly, step by step, and call out the parts that matter most.\n\n"],
    ["Summarize the conversation", "Summarize our conversation so far. Keep the key decisions, open questions and next steps.\n\n"],
    ["Review and improve", "Review the following and improve it. Keep the meaning, fix mistakes, and explain the important changes.\n\n"],
    ["Write code", "Write code for the following. Include a short explanation and a runnable example.\n\n"],
    ["Design a UI", "Create a self-contained HTML page with embedded CSS that implements the following design. Do not run scripts; focus on layout, type and colour.\n\n"],
    ["Plan the steps", "Break the following goal into ordered, concrete steps. Note the risks and what to verify after each step.\n\n"],
    ["Compare options", "Compare the options below on cost, complexity, performance and maintenance, then recommend one.\n\n"],
    ["Translate", "Translate the following text. Preserve the tone and formatting, and list any terms you had to interpret.\n\n"],
  ];

  const SWARM_DEFAULTS = {
    strategy: "auto", max_agents: 8, rounds: 0, ipc: true, memory: true,
    depth: "deep", max_steps: 16, max_seconds: 900,
  };
  const TOOL_DEFAULTS = { files: true, write: false, web: true };
  const SWARM_STRATEGIES = [
    ["auto", "Automatic"], ["map_reduce", "Map & reduce"], ["peer_review", "Peer review"],
    ["debate", "Debate"], ["pipeline", "Pipeline"],
  ];
  /* One preset controls both bounds. Tokens are bounded mainly by rounds; the
     time budget lets slow web fetches finish instead of cutting the team off. */
  const SWARM_DEPTHS = [
    ["standard", "Standard · 5 min · 10 rounds"],
    ["deep", "Deep research · 15 min · 16 rounds"],
    ["extended", "Extended · 30 min · 24 rounds"],
  ];
  const SWARM_DEPTH_VALUES = {
    standard: { max_steps: 10, max_seconds: 300 },
    deep: { max_steps: 16, max_seconds: 900 },
    extended: { max_steps: 24, max_seconds: 1800 },
  };

  let layout = null;

  const state = {
    chats: [],
    doc: null,
    abort: null,
    runToken: 0,
    loadToken: 0,
    activateToken: 0,
    streaming: false,
    loading: false,
    needsRender: false,
    search: "",
    pinned: true,
    attachments: [],
    // Swarm is the default; only an explicit Solo choice is remembered.
    mode: Studio.store.get("chat-mode", "swarm") === "solo" ? "solo" : "swarm",
    swarmOptions: readSwarmOptions(),
    tools: readTools(),
    autoPreview: Studio.store.get("chat-auto-preview", true) !== false,
    budgetCache: null,
    preview: { open: false, selected: null, view: "preview", closedIds: new Set(), count: 0,
      rendered: "", workspace: new Map() },
    voiceStatus: null,
    voice: null,
    voiceState: "idle",
    voiceToken: 0,
    voiceAuto: Boolean(Studio.store.get("chat-voice-auto", false)),
    spokenMessage: null,
  };

  function normaliseSwarmOptions(raw) {
    const saved = raw && typeof raw === "object" ? raw : {};
    const bounded = (value, fallback, low, high) => {
      const number = Number(value);
      return Number.isFinite(number) ? Math.max(low, Math.min(high, Math.round(number))) : fallback;
    };
    const seconds = bounded(saved.max_seconds, SWARM_DEFAULTS.max_seconds, 20, 1800);
    /* Saved options from before the preset existed always had the old hidden
       five-minute default; those migrate to Deep research. A longer custom value
       keeps its closest preset instead of being reset. */
    const depth = Object.prototype.hasOwnProperty.call(SWARM_DEPTH_VALUES, saved.depth)
      ? saved.depth
      : seconds > 300 ? (seconds >= 1200 ? "extended" : "deep") : SWARM_DEFAULTS.depth;
    const budget = SWARM_DEPTH_VALUES[depth];
    return {
      strategy: SWARM_STRATEGIES.some(([value]) => value === saved.strategy)
        ? saved.strategy : SWARM_DEFAULTS.strategy,
      depth,
      max_agents: bounded(saved.max_agents, SWARM_DEFAULTS.max_agents, 1, 64),
      rounds: bounded(saved.rounds, SWARM_DEFAULTS.rounds, 0, 2),
      max_steps: budget.max_steps,
      max_seconds: budget.max_seconds,
      ipc: saved.ipc !== false,
      memory: saved.memory !== false,
    };
  }

  function readSwarmOptions() {
    return normaliseSwarmOptions(Studio.store.get("chat-swarm", null));
  }

  function normaliseTools(raw) {
    const saved = raw && typeof raw === "object" ? raw : {};
    const write = saved.write === true;
    return { files: saved.files !== false || write, write, web: saved.web !== false };
  }

  function readTools() {
    const saved = Studio.store.get("chat-tools", null);
    return normaliseTools(saved === null ? TOOL_DEFAULTS : saved);
  }

  function swarmPayload(target = state.doc) {
    const options = state.swarmOptions;
    return {
      strategy: options.strategy, rounds: options.rounds, max_agents: options.max_agents,
      max_steps: options.max_steps, max_seconds: options.max_seconds,
      ipc: options.ipc, memory: options.memory,
      chat_id: target && target.id ? target.id : "",
      tools: { ...state.tools },
    };
  }

  function updateModeControls() {
    const swarm = state.mode === "swarm";
    const disabled = state.streaming || state.loading;
    $("chat-mode-solo").disabled = disabled;
    $("chat-mode-swarm").disabled = disabled;
    $("chat-mode-solo").setAttribute("aria-pressed", String(!swarm));
    $("chat-mode-swarm").setAttribute("aria-pressed", String(swarm));
    const options = state.swarmOptions;
    $("chat-mode-solo").title = "One model answers directly · faster and cheaper";
    $("chat-mode-swarm").title = `${swarm ? "Swarm mode is on (default)" : "Use a team of agents (default)"}`
      + ` · up to ${options.max_agents} agents · ${Math.round(options.max_seconds / 60)} min · `
      + (options.rounds === 0
        ? "no peer-review rounds"
        : `${options.rounds} feedback round${options.rounds === 1 ? "" : "s"}`)
      + (state.tools.files || state.tools.web ? " · tools on" : " · no tools")
      + " · set the strategy in conversation settings";
  }

  function setMode(mode) {
    if (state.streaming || state.loading) return;
    state.mode = mode === "swarm" ? "swarm" : "solo";
    Studio.store.set("chat-mode", state.mode);
    updateModeControls();
    updateHeader();
    $("chat-input").focus();
  }

  function currentModel() {
    return state.doc && Studio.models.get(state.doc.model);
  }

  function refreshChatModel(modelId) {
    return Studio.ui.fillModels($("chat-model"), {
      filter: entry => entry.capabilities.chat,
      value: modelId || (state.doc && state.doc.model) || "",
      includeUnavailable: true,
    });
  }

  function defaults(entry) {
    const all = Studio.store.get("chat-settings", {});
    const stored = all && typeof all === "object" ? all[entry.id] : null;
    const saved = stored && typeof stored === "object" ? stored : {};
    return normaliseSettings({
      temperature: 0.8, top_p: 0.95, top_k: 0, max_tokens: (entry.defaults || {}).max_tokens || 512,
      device: "auto", prompt_format: "auto", n_ctx: 0, seed: "", stop: [], effort: "high", show_reasoning: false,
      guidance: 1, guidance_mode: "none", early_exit: true, refine_steps: 2, refine_threshold: 0.5, candidates: 1,
      ...saved,
    });
  }

  function newDocument(modelId) {
    const entries = Studio.models.chat();
    const entry = Studio.models.get(modelId) || entries.find(item => item.available) || entries[0];
    return {
      id: null, title: "New chat", titled: false, pinned: false, model: entry ? entry.id : "", model_name: entry ? entry.name : "",
      system_prompt: "", settings: entry ? defaults(entry) : {}, messages: [],
    };
  }

  let chatsToken = 0;

  async function refreshChats() {
    const token = ++chatsToken;
    let data;
    try {
      data = await api("/api/chats");
    } catch (error) {
      if (token !== chatsToken) return state.chats;
      throw error;
    }
    if (token !== chatsToken) return state.chats;
    state.chats = data && Array.isArray(data.chats)
      ? data.chats.filter(chat => chat && typeof chat === "object") : [];
    renderChatList();
    return state.chats;
  }

  let chatListTimer = 0;

  /* Typing in the sidebar search used to rebuild every row per keystroke.
     Repaint at most once per 120ms; the final value is always rendered. */
  function scheduleChatList() {
    if (chatListTimer) return;
    chatListTimer = setTimeout(() => {
      chatListTimer = 0;
      renderChatList();
    }, 120);
  }

  let chatListState = { docId: "", search: "", minute: -1, chats: [] };

  let chatRowCache = new Map();
  let chatListEmpty = "";

  function chatRowKey(chat, active, minute) {
    return `${active ? 1 : 0}|${minute}|${chat.title}|${chat.preview || ""}|${chat.model_name || ""}|${chat.pinned ? 1 : 0}|${chat.updated || ""}`;
  }

  function chatRow(chat, active, minute) {
    const key = chatRowKey(chat, active, minute);
    const cached = chatRowCache.get(chat.id);
    if (cached && cached.key === key && cached.element.isConnected) return cached.element;
    const element = Studio.components.listRow({
      className: "chat-list-item",
      active,
      title: `${chat.pinned ? "Pinned · " : ""}${chat.title}`,
      subtitle: chat.preview || chat.model_name || "No messages",
      meta: format.ago(chat.updated),
      onclick: () => {
        setConversationList(false);
        if (state.doc && state.doc.id === chat.id) return;
        Studio.views.go("chat", { id: chat.id });
      },
    });
    chatRowCache.set(chat.id, { key, element });
    return element;
  }

  function setConversationList(open) {
    const layoutElement = document.querySelector(".chat-layout");
    if (!layoutElement) return;
    layoutElement.classList.toggle("show-list", Boolean(open));
    $("chat-conversations").setAttribute("aria-expanded", String(Boolean(open)));
  }

  function renderChatList() {
    const docId = state.doc ? state.doc.id : "";
    const minute = Math.floor(Date.now() / 60000);
    const previous = chatListState;
    const unchanged = previous.docId === docId && previous.search === state.search
      && previous.minute === minute && previous.chats.length === state.chats.length
      && state.chats.every((chat, index) => {
        const old = previous.chats[index];
        return old && old.id === chat.id && old.title === chat.title
          && old.preview === chat.preview && old.model_name === chat.model_name
          && old.pinned === chat.pinned && old.updated === chat.updated;
      });
    if (unchanged) return;
    chatListState = {
      docId, search: state.search, minute,
      chats: state.chats.map(chat => ({
        id: chat.id, title: chat.title, preview: chat.preview,
        model_name: chat.model_name, pinned: chat.pinned, updated: chat.updated,
      })),
    };
    const list = $("chat-list");
    const query = state.search.toLowerCase();
    const chats = state.chats.filter(chat => !query || `${chat.title} ${chat.preview} ${chat.model_name}`.toLowerCase().includes(query));
    if (!chats.length) {
      const text = query ? "No conversations match." : "Start a new conversation.";
      if (chatListEmpty !== text) {
        chatListEmpty = text;
        Studio.ui.render(list, Studio.ui.empty({ compact: true, text }));
      }
      return;
    }
    chatListEmpty = "";
    const desired = chats.map(chat => chatRow(chat, Boolean(docId && chat.id === docId), minute));
    const existing = Array.from(list.children);
    let keep = 0;
    while (keep < existing.length && keep < desired.length && existing[keep] === desired[keep]) keep += 1;
    if (keep < existing.length || keep < desired.length) {
      for (let index = existing.length - 1; index >= keep; index -= 1) existing[index].remove();
      if (keep < desired.length) {
        const fragment = document.createDocumentFragment();
        for (const element of desired.slice(keep)) fragment.append(element);
        list.append(fragment);
      }
    }
    if (chatRowCache.size > state.chats.length) {
      const live = new Set(state.chats.map(chat => chat.id));
      for (const id of [...chatRowCache.keys()]) if (!live.has(id)) chatRowCache.delete(id);
    }
  }

  function nearBottom() {
    const messages = $("chat-messages");
    return messages.scrollHeight - messages.scrollTop - messages.clientHeight < 60;
  }

  function updateScrollButton() {
    const messages = $("chat-messages");
    const button = $("chat-scroll");
    const hidden = state.pinned || !messages.children.length;
    button.classList.toggle("hidden", hidden);
    button.classList.toggle("streaming", !hidden && state.streaming);
    button.textContent = !hidden && state.streaming ? "New output ↓" : "Jump to latest";
  }

  let scrollFrame = 0;

  function scrollMessages(force = false) {
    if (!force && !state.pinned) return;
    state.pinned = true;
    if (scrollFrame) return;
    scrollFrame = requestAnimationFrame(() => {
      scrollFrame = 0;
      const messages = $("chat-messages");
      messages.scrollTop = messages.scrollHeight;
      updateScrollButton();
    });
  }

  /* Streaming deltas arrive token by token.  Re-parsing the whole reply as
     Markdown for every token is quadratic, so coalesce the live paint to one
     per ~80ms.  The authoritative final text is rendered by renderMessages()
     when the stream ends, so nothing is lost if a paint is skipped. */
  let liveTimer = 0;
  let liveMessage = null;
  let reasoningTimer = 0;
  let reasoningMessage = null;
  let liveClock = 0;
  let liveStream = null;
  const liveParts = new WeakMap();

  function stopStream() {
    if (state.abort) state.abort.abort();
  }

  function liveEntry(message) {
    return message ? liveParts.get(message) || null : null;
  }

  /* The live row is created with the pending message so the reader sees a
     state, an elapsed time and a stop control even before the first token. */
  function streamLive(message) {
    if (!message.pending || message.role !== "assistant" || message.swarm) return null;
    const phase = h("strong", { class: "stream-phase", text: "Waiting for the model…" });
    const metrics = h("span", { class: "stream-metrics muted tiny" });
    const reasoningText = h("div", { class: "reasoning-live" });
    const summary = h("summary", { text: "Thinking" });
    const knownReasoning = message.reasoning_open !== undefined;
    const reasoning = h("details", { class: "stream-reasoning hidden",
      open: knownReasoning ? Boolean(message.reasoning_open) : false }, summary, reasoningText);
    reasoning.addEventListener("toggle", () => {
      message.reasoning_open = reasoning.open;
      const entry = liveParts.get(message);
      if (entry) entry.reasoningOpen = true;
    });
    if (message.reasoning) {
      reasoning.classList.remove("hidden");
      reasoningText.textContent = message.reasoning.slice(-4000);
    }
    const row = h("div", { class: "stream-live", "aria-live": "off" },
      h("span", { class: "spinner", "aria-hidden": "true" }),
      phase, metrics, h("span", { class: "grow" }),
      Studio.ui.button("Stop", { small: true, kind: "danger", onclick: stopStream }));
    liveParts.set(message, {
      phase, metrics, reasoning, summary, reasoningText,
      reasoningOpen: knownReasoning,
    });
    return h("div", { class: "stream-live-stack" }, row, reasoning);
  }

  function beginLive(message) {
    if (!message) return;
    if (liveStream && liveStream.message === message) return;
    liveStream = { message, started: performance.now(), firstToken: 0, chars: 0 };
    const entry = liveEntry(message);
    if (entry) entry.phase.textContent = "Waiting for the model…";
    if (!liveClock) liveClock = setInterval(paintLiveClock, 250);
  }

  function paintLiveClock() {
    if (!liveStream) {
      clearInterval(liveClock);
      liveClock = 0;
      return;
    }
    if (document.hidden) return;
    paintLiveMetrics(liveStream.message);
  }

  function paintLiveMetrics(message) {
    const entry = liveEntry(message);
    const stream = liveStream && liveStream.message === message ? liveStream : null;
    if (!entry || !stream) return;
    const elapsed = Math.max(0, (performance.now() - stream.started) / 1000);
    const parts = [`${elapsed.toFixed(1)}s`];
    if (stream.firstToken) {
      parts.push(`${((stream.firstToken - stream.started) / 1000).toFixed(1)}s first token`);
      if (elapsed > 0.5 && stream.chars) {
        const tokens = Studio.tokens && Studio.tokens.estimate
          ? Studio.tokens.estimate(message.content, "tokens") : 0;
        if (tokens) parts.push(`~${tokens.toLocaleString()} tok`);
        if (tokens && elapsed > 1) parts.push(`~${Math.round(tokens / elapsed)} tok/s`);
      }
    }
    entry.metrics.textContent = parts.join(" · ");
  }

  function endLive(message) {
    if (message && liveStream && liveStream.message !== message) return;
    liveStream = null;
    if (liveClock) clearInterval(liveClock);
    liveClock = 0;
  }

  function liveContentFor(message) {
    if (!state.doc) return null;
    const index = state.doc.messages.indexOf(message);
    if (index < 0) return null;
    const article = $("chat-messages").children[index];
    return article ? article.querySelector(".message-content") : null;
  }

  function paintLiveReply(message) {
    liveTimer = 0;
    if (!message || message !== liveMessage) return;
    const target = liveContentFor(message);
    if (!target) return;
    target.classList.add("streaming");
    Studio.markdown.stream(target, message.content);
    const entry = liveEntry(message);
    if (entry) entry.phase.textContent = "Writing";
    paintLiveMetrics(message);
    scrollMessages();
    updateScrollButton();
  }

  function scheduleLiveReply(message) {
    liveMessage = message;
    if (liveTimer || document.hidden) return;
    /* Long replies cost more to scan; relax the cadence instead of letting a
       50k-character answer compete with input and scrolling. */
    const delay = message.content.length > 24000 ? 160 : 80;
    liveTimer = setTimeout(() => paintLiveReply(message), delay);
  }

  function paintReasoning(message) {
    reasoningTimer = 0;
    if (!message || message !== reasoningMessage) return;
    const entry = liveEntry(message);
    if (!entry) return;
    entry.reasoning.classList.remove("hidden");
    if (!entry.reasoningOpen) {
      entry.reasoning.open = true;
      entry.reasoningOpen = true;
    }
    entry.reasoningText.textContent = message.reasoning.slice(-4000);
    entry.reasoningText.scrollTop = entry.reasoningText.scrollHeight;
    entry.summary.textContent = `Thinking · ${message.reasoning.length.toLocaleString()} chars`;
    entry.phase.textContent = message.content ? "Writing" : "Thinking…";
    scrollMessages();
  }

  function scheduleReasoning(message) {
    reasoningMessage = message;
    if (reasoningTimer || document.hidden) return;
    reasoningTimer = setTimeout(() => paintReasoning(message), 100);
  }

  function cancelLiveReply() {
    liveMessage = null;
    reasoningMessage = null;
    if (liveTimer) {
      clearTimeout(liveTimer);
      liveTimer = 0;
    }
    if (reasoningTimer) {
      clearTimeout(reasoningTimer);
      reasoningTimer = 0;
    }
  }

  function resizeComposer() {
    const input = $("chat-input");
    input.style.height = "auto";
    input.style.height = `${Math.min(input.scrollHeight, 180)}px`;
  }

  function updateComposerState() {
    const entry = currentModel();
    $("chat-send").disabled = state.streaming || state.loading || !entry || !entry.available
      || (!$("chat-input").value.trim() && !state.attachments.length);
  }

  /* -- attachments ---------------------------------------------------------- */
  function newAttachmentId() {
    return `a${Date.now().toString(36)}${Math.random().toString(36).slice(2, 8)}`;
  }

  function readDataUrl(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result || ""));
      reader.onerror = () => reject(new Error(`Could not read ${file.name || "the file"}.`));
      reader.readAsDataURL(file);
    });
  }

  function isTextFile(file, language) {
    const mime = String(file.type || "").toLowerCase();
    if (/^text\//.test(mime)) return true;
    if (["application/json", "application/xml", "application/javascript", "application/x-yaml",
      "application/toml"].includes(mime)) return true;
    const name = String(file.name || "").toLowerCase();
    const extension = name.includes(".") ? name.split(".").pop() : name;
    return TEXT_EXTENSIONS.has(extension) || (Studio.highlight && Studio.highlight.hasGrammar(language));
  }

  function imageMimeFor(name, mime) {
    if (/^image\//i.test(mime)) return mime;
    const extension = String(name || "").toLowerCase().split(".").pop();
    return {
      png: "image/png", jpg: "image/jpeg", jpeg: "image/jpeg", gif: "image/gif",
      webp: "image/webp", bmp: "image/bmp", svg: "image/svg+xml", ico: "image/x-icon",
      avif: "image/avif", tif: "image/tiff", tiff: "image/tiff", heic: "image/heic",
      heif: "image/heif", jfif: "image/jpeg",
    }[extension] || "image/png";
  }

  async function readAttachment(file) {
    const givenName = String(file.name || "");
    const name = givenName || (/^image\//.test(String(file.type || "")) ? "pasted-image.png" : "pasted-file");
    const mime = String(file.type || "");
    const size = Number(file.size) || 0;
    const imageName = /\.(?:png|jpe?g|gif|webp|bmp|svg|ico|avif|tiff?|heic|heif|jfif)$/i.test(givenName);
    if (/^image\//.test(mime) || imageName) {
      if (!size) {
        Studio.toast(`${name} is empty, so there is nothing to preview.`, "error");
        return null;
      }
      if (size > MAX_IMAGE_BYTES) {
        Studio.toast(`${name} is too large for an image preview (limit ${format.bytes(MAX_IMAGE_BYTES)}).`, "error");
        return null;
      }
      const imageMime = imageMimeFor(name, mime);
      let dataUrl = await readDataUrl(file);
      const comma = dataUrl.indexOf(",");
      if (comma < 0) {
        Studio.toast(`${name} could not be read as an image.`, "error");
        return null;
      }
      if (!/^data:image\//i.test(dataUrl)) {
        dataUrl = `data:${imageMime};base64,${dataUrl.slice(comma + 1)}`;
      }
      return {
        id: newAttachmentId(), name, kind: "image", mime: imageMime, size,
        data_url: dataUrl,
      };
    }
    const language = Studio.highlight ? Studio.highlight.resolve("", name) : "text";
    const extension = name.includes(".") ? name.split(".").pop().toLowerCase() : "";
    const unknownExtension = !extension || !/^[a-z0-9]+$/i.test(extension);
    if (isTextFile(file, language) || (unknownExtension && size <= MAX_TEXT_BYTES)) {
      if (size > MAX_TEXT_BYTES) {
        Studio.toast(`${name} is too large to attach as text (limit ${format.bytes(MAX_TEXT_BYTES)}).`, "error");
        return null;
      }
      const value = await file.text();
      if (value.includes("\u0000")) {
        return {
          id: newAttachmentId(), name, kind: "file", mime: mime || "application/octet-stream", size,
        };
      }
      return {
        id: newAttachmentId(), name, kind: "text", mime: mime || "text/plain", size,
        language, text: value.slice(0, MAX_ATTACHMENT_TEXT),
      };
    }
    return {
      id: newAttachmentId(), name, kind: "file", mime: mime || "application/octet-stream", size,
    };
  }

  function attachmentPayloadLength(attachment) {
    return String(attachment.data_url || "").length + String(attachment.text || "").length;
  }

  function conversationAttachmentLength() {
    let total = 0;
    for (const attachment of state.attachments) total += attachmentPayloadLength(attachment);
    for (const message of state.doc ? state.doc.messages : []) {
      for (const attachment of message.attachments || []) total += attachmentPayloadLength(attachment);
    }
    return total;
  }

  function estimatedPayloadLength(file, mime) {
    const size = Number(file.size) || 0;
    const name = String(file.name || "");
    if (/^image\//i.test(mime) || /\.(?:png|jpe?g|gif|webp|bmp|svg|ico|avif|tiff?|heic|heif|jfif)$/i.test(name)) {
      return Math.ceil(size / 3) * 4 + 64;
    }
    const extension = name.includes(".") ? name.split(".").pop().toLowerCase() : "";
    const textual = /^text\//i.test(mime) || ["application/json", "application/xml",
      "application/javascript", "application/x-yaml", "application/toml"].includes(String(mime || "").toLowerCase())
      || TEXT_EXTENSIONS.has(extension) || !extension;
    return textual ? Math.min(size, MAX_ATTACHMENT_TEXT) : 0;
  }

  async function addFiles(fileList) {
    const files = Array.from(fileList || []);
    if (!files.length) return;
    let added = 0;
    for (const file of files) {
      if (state.attachments.length >= MAX_ATTACHMENTS) {
        Studio.toast(`A message can carry at most ${MAX_ATTACHMENTS} attachments.`, "info");
        break;
      }
      const used = state.attachments.reduce((sum, attachment) => sum + (Number(attachment.size) || 0), 0);
      if (used + (Number(file.size) || 0) > MAX_ATTACHMENTS_TOTAL_BYTES) {
        Studio.toast(`A message can carry at most ${format.bytes(MAX_ATTACHMENTS_TOTAL_BYTES)} of attachments.`, "info");
        continue;
      }
      const conversation = conversationAttachmentLength();
      if (conversation + estimatedPayloadLength(file, file.type) > MAX_CONVERSATION_ATTACHMENT_CHARS) {
        Studio.toast("This conversation has reached its attachment limit. Start a new chat to add more images or files.", "info");
        continue;
      }
      try {
        const attachment = await readAttachment(file);
        if (attachment
            && conversationAttachmentLength() + attachmentPayloadLength(attachment) > MAX_CONVERSATION_ATTACHMENT_CHARS) {
          Studio.toast("This conversation has reached its attachment limit. Start a new chat to add more images or files.", "info");
          continue;
        }
        if (attachment) {
          state.attachments.push(attachment);
          added += 1;
        }
      } catch (error) {
        Studio.toast(error.message, "error");
      }
    }
    if (added) {
      renderAttachments();
      resizeComposer();
      updateComposerState();
      updateBudgetMeter();
      refreshPreviewSelection();
      if (state.autoPreview) {
        const artifacts = Studio.artifacts.fromAttachments(state.attachments, "draft", -1);
        const newest = artifacts[artifacts.length - 1];
        if (newest) openArtifact(newest);
      }
    }
  }

  function removeAttachment(id) {
    state.attachments = state.attachments.filter(attachment => attachment.id !== id);
    renderAttachments();
    updateComposerState();
    updateBudgetMeter();
    refreshPreviewSelection();
  }

  function attachmentThumb(attachment, onOpen) {
    const button = h("button", {
      class: "chat-attachment-thumb", type: "button",
      "aria-label": `Preview ${attachment.name}`, title: `Preview ${attachment.name}`,
      onclick: onOpen,
    });
    if (attachment.kind === "image" && attachment.data_url) {
      button.append(h("img", { src: attachment.data_url, alt: attachment.name }));
    } else {
      button.append(icon(attachment.kind === "text" ? "code" : "file", "small"));
    }
    return button;
  }

  function attachmentChip(attachment, options = {}) {
    const open = options.onOpen || (() => {
      const artifact = Studio.artifacts.attachmentArtifact(
        attachment, options.scope || "draft", options.index, options.position || 0);
      if (artifact) openArtifact(artifact);
    });
    const chip = h("div", { class: "chat-attachment" },
      attachmentThumb(attachment, open),
      h("div", { class: "chat-attachment-meta" },
        h("span", { class: "chat-attachment-name", title: attachment.name, text: attachment.name }),
        h("span", { class: "chat-attachment-size",
          title: attachment.kind === "image"
            ? "Image preview; the selected model receives the file name, not the pixels"
            : null,
          text: `${attachment.kind === "image" ? "Image" : attachment.kind === "text" ? "Text" : "File"} · ${format.bytes(attachment.size || 0)}` })));
    if (options.remove) {
      chip.append(Studio.ui.iconButton("x", `Remove ${attachment.name}`, () => removeAttachment(attachment.id), { small: true }));
    }
    return chip;
  }

  function renderAttachments() {
    const root = Studio.ui.render($("chat-attachments"));
    state.attachments.forEach((attachment, position) => {
      root.append(attachmentChip(attachment, { remove: true, scope: "draft", index: -1, position }));
    });
  }

  function messageAttachments(attachments, index) {
    const wrap = h("div", { class: "message-attachments" });
    (attachments || []).forEach((attachment, position) => {
      wrap.append(attachmentChip(attachment, { scope: `m${index}`, index, position }));
    });
    return wrap;
  }

  /* -- conversation text ---------------------------------------------------- */
  function fenceFor(source) {
    const runs = String(source || "").match(/`+/g) || [];
    let size = 3;
    for (const run of runs) size = Math.max(size, run.length + 1);
    return "`".repeat(size);
  }

  function safePromptName(name) {
    return String(name || "attachment").replace(/[`"\r\n]+/g, "'").slice(0, 200);
  }

  function attachmentPrompt(attachments) {
    if (!Array.isArray(attachments) || !attachments.length) return "";
    const parts = [];
    let remaining = MAX_PROMPT_ATTACHMENT_CHARS;
    for (const attachment of attachments) {
      const name = safePromptName(attachment.name);
      const mime = String(attachment.mime || "file").replace(/[^A-Za-z0-9.+/-]/g, "").slice(0, 120) || "file";
      if (attachment.kind === "text" && attachment.text) {
        const body = attachment.text.length > remaining
          ? `${attachment.text.slice(0, remaining)}\n… (truncated to fit the model context)`
          : attachment.text;
        remaining = Math.max(0, remaining - attachment.text.length);
        const fence = fenceFor(body);
        const language = String(attachment.language || "").replace(/[^\w+#.-]/g, "").slice(0, 40);
        parts.push(`Attached file "${name}":\n${fence}${language}\n${body}\n${fence}`);
      } else if (attachment.kind === "image") {
        parts.push(`[Image attached: "${name}" (${mime}). The chat shows a preview; this model receives the file name only.]`);
      } else {
        parts.push(`[File attached: "${name}" (${mime}, ${format.bytes(attachment.size || 0)}).]`);
      }
      if (remaining <= 0) break;
    }
    return parts.join("\n\n");
  }

  function messagePromptParts(message) {
    const body = String(message.content || "");
    // An error message that replaced the whole reply is not an assistant turn.
    const usable = message.error && body === message.error ? "" : body;
    const extra = attachmentPrompt(message.attachments);
    return { extra, prompt: extra && usable ? `${extra}\n\n${usable}` : extra || usable };
  }

  function messagePrompt(message) {
    return messagePromptParts(message).prompt;
  }

  function outgoingMessages(target = state.doc) {
    const messages = target ? target.messages : state.doc ? state.doc.messages : [];
    return messages
      .filter(message => !message.pending && message.role !== "system")
      .map(message => ({ role: message.role, content: messagePrompt(message) }))
      .filter(message => message.content);
  }

  function removeMessage(index) {
    if (state.streaming || !state.doc || !state.doc.messages[index]) return;
    const target = state.doc;
    const message = target.messages[index];
    Studio.confirm({ title: "Delete message?", message: "Remove this message from the conversation?", confirmLabel: "Delete", danger: true })
      .then(async confirmed => {
        if (!confirmed || state.streaming) return;
        const position = target.messages.indexOf(message);
        if (position < 0) return;
        if (state.spokenMessage === message && Studio.voice) {
          Studio.voice.stopSpeaking();
          state.spokenMessage = null;
        }
        target.messages.splice(position, 1);
        if (state.doc === target) {
          renderMessages({ keepScroll: true });
          refreshPreviewSelection();
        }
        try {
          await saveChat(target);
        } catch (error) {
          Studio.toast(error.message, "error");
        }
      });
  }

  function editMessage(index) {
    const message = state.doc.messages[index];
    const area = h("textarea", { class: "message-edit", value: message.content, "aria-label": "Edit message" });
    const save = Studio.ui.button("Save & regenerate", { kind: "primary", small: true, onclick: async () => {
      if (state.streaming || state.loading) return;
      const content = area.value.trim();
      if (!content && !(message.attachments || []).length) return;
      const entry = currentModel();
      if (!entry || !entry.available) {
        Studio.toast(entry ? entry.reason : "Choose an available model first.", "error");
        return;
      }
      if (index < state.doc.messages.length - 1) {
        const confirmed = await Studio.confirm({
          title: "Regenerate from this message?",
          message: "Every message after this one, including the reply, will be removed.",
          confirmLabel: "Save & regenerate", danger: true,
        });
        if (!confirmed) return;
      }
      const replacement = { role: "user", content };
      if ((message.attachments || []).length) replacement.attachments = message.attachments;
      state.doc.messages = state.doc.messages.slice(0, index).concat(replacement);
      const saved = await Studio.run(saveChat);
      if (!saved) return;
      renderMessages();
      refreshPreviewSelection();
      generate();
    } });
    const cancel = Studio.ui.button("Cancel", { small: true, onclick: () => {
      messageCache.delete(message);
      renderMessages();
    } });
    const holder = h("div", { class: "message-content stack" }, area, h("div", { class: "row" }, save, cancel));
    const article = $("chat-messages").children[index];
    if (article) Studio.ui.render(article.querySelector(".message-content"), holder);
  }

  function readMessage(message) {
    if (!Studio.voice || !Studio.voice.synthesisSupported()) {
      Studio.toast("This browser cannot read replies aloud.", "info");
      return;
    }
    if (Studio.voice.speaking() && state.spokenMessage === message) {
      Studio.voice.stopSpeaking();
      state.spokenMessage = null;
      return;
    }
    state.spokenMessage = null;
    const started = Studio.voice.speak(message.content || "");
    state.spokenMessage = started ? message : null;
  }

  function messageActions(message, index) {
    const actions = h("div", { class: "message-actions" });
    actions.append(Studio.ui.iconButton("copy", "Copy message", () => {
      const attachments = (message.attachments || []).map(item => `Attached: ${item.name}`).join("\n");
      Studio.copy([message.content || "", attachments].filter(Boolean).join("\n\n"), "Message copied");
    }, { small: true }));
    if (message.role === "assistant" && message.content) {
      actions.append(Studio.ui.iconButton("volume", "Read this reply aloud", () => readMessage(message), { small: true }));
    }
    if (!state.streaming) {
      if (message.role === "user") actions.append(Studio.ui.iconButton("edit", "Edit and regenerate", () => editMessage(index), { small: true }));
      if (message.role === "assistant" && index === state.doc.messages.length - 1) {
        actions.append(Studio.ui.iconButton("refresh", "Regenerate response", () => {
          const entry = currentModel();
          if (!entry || !entry.available) {
            Studio.toast(entry ? entry.reason : "Choose an available model first.", "error");
            return;
          }
          state.doc.messages = state.doc.messages.slice(0, index);
          renderMessages();
          refreshPreviewSelection();
          generate();
        }, { small: true }));
      }
      actions.append(Studio.ui.iconButton("trash", "Delete message", () => removeMessage(index), { small: true }));
    }
    return actions;
  }

  function decorateCodeBlocks(container, message, index) {
    const remaining = Studio.artifacts.messageArtifacts(message, index)
      .filter(artifact => !artifact.attachment);
    for (const block of Array.from(container.querySelectorAll(".code-block"))) {
      const code = block.querySelector("pre code");
      const source = code ? code.textContent : "";
      const position = remaining.findIndex(artifact => artifact.source === source);
      if (position < 0) continue;
      const artifact = remaining.splice(position, 1)[0];
      block.append(h("button", {
        class: "icon-button small code-preview", type: "button",
        "aria-label": `Preview ${artifact.title}`, title: "Open in the preview panel",
        onclick: () => openArtifact(artifact),
      }, icon("preview", "small")));
    }
  }

  function messageElement(message, index, options = {}) {
    const article = h("article", { class: `message ${message.role}${message.error ? " error" : ""}` });
    const name = message.role === "user" ? "You" : message.role === "assistant" ? (message.model_name || state.doc.model_name || "Assistant") : "System";
    const meta = [];
    if (message.tokens_per_second) meta.push(`${Number(message.tokens_per_second).toFixed(1)} tok/s`);
    if (message.elapsed) meta.push(format.seconds(message.elapsed));
    if (message.usage && message.usage.completion_tokens !== undefined) meta.push(`${message.usage.completion_tokens} output tokens`);
    if (message.finish_reason && message.finish_reason !== "stop") meta.push(format.title(message.finish_reason));
    const head = h("div", { class: "message-head" }, h("strong", { text: name }),
      h("span", { text: meta.join(" · ") }), messageActions(message, index));
    article.append(head);
    const live = streamLive(message);
    if (live) article.append(live);
    if (message.reasoning && !message.pending) {
      article.append(h("details", { class: "message-reasoning", open: Boolean(message.reasoning_open) },
        h("summary", { text: "Reasoning" }), h("div", { text: message.reasoning })));
    }
    if ((message.attachments || []).length) article.append(messageAttachments(message.attachments, index));
    const content = h("div", { class: `message-content${message.pending ? " streaming" : ""}` });
    if (message.pending) content.setAttribute("aria-live", "off");
    if (message.role === "assistant") {
      const placeholder = state.streaming && index === state.doc.messages.length - 1
        && !(message.swarm && message.pending) ? "…" : "";
      Studio.markdown.into(content, message.content || placeholder, options.lazy ? { lazy: true } : undefined);
      if (!message.pending && message.content) decorateCodeBlocks(content, message, index);
    } else content.textContent = message.content || "";
    article.append(content);
    if (message.notice) article.append(h("div", { class: "help", text: message.notice }));
    if (message.swarm) {
      const swarmHolder = h("div", { class: "message-swarm" });
      Studio.swarm.render(swarmHolder, message.swarm, { live: Boolean(message.pending) });
      article.append(swarmHolder);
    }
    if (message.served_by) article.append(h("div", { class: "help", text: `Served by ${message.served_by}` }));
    if (message.error && !state.streaming && index === state.doc.messages.length - 1) {
      const retry = solo => () => {
        const entry = currentModel();
        if (!entry || !entry.available) {
          Studio.toast(entry ? entry.reason : "Choose an available model first.", "error");
          return;
        }
        state.doc.messages = state.doc.messages.slice(0, index);
        renderMessages();
        refreshPreviewSelection();
        if (solo) generateReply();
        else generate();
      };
      article.append(h("div", { class: "row" },
        Studio.ui.button("Retry", { small: true, onclick: retry(false) }),
        state.mode === "swarm"
          ? Studio.ui.button("Retry in Solo", { small: true, onclick: retry(true) })
          : null));
    }
    return article;
  }

  let messageCache = new WeakMap();
  let messageCacheDoc = null;

  function messageMetrics(message) {
    return [
      message.tokens_per_second ? Number(message.tokens_per_second).toFixed(1) : "",
      message.elapsed || "",
      message.usage && message.usage.completion_tokens !== undefined ? message.usage.completion_tokens : "",
      message.finish_reason || "",
      message.model_name || "",
      message.served_by || "",
    ].join("|");
  }

  function attachmentsKey(message) {
    return (message.attachments || []).map(item =>
      `${item.id || ""}:${item.name || ""}:${item.size || 0}:${item.kind || ""}`).join("|");
  }

  function swarmKey(message) {
    if (!message.swarm) return "";
    const meta = message.swarm.meta || {};
    return `${meta.status || ""}:${meta.finished ? 1 : 0}:${meta.ok ? 1 : 0}:${(message.swarm.agents || []).length}`;
  }

  function actionsKey(message, index) {
    const last = state.doc && index === state.doc.messages.length - 1;
    return `${state.streaming ? 1 : 0}|${message.role}|${message.content ? 1 : 0}|${last ? 1 : 0}`;
  }

  function messageTail(message, index) {
    const last = state.doc && index === state.doc.messages.length - 1;
    return Boolean(last && (state.streaming || message.error));
  }

  function messageCacheMatches(entry, message, index) {
    return Boolean(entry)
      && entry.index === index
      && entry.docName === (state.doc ? state.doc.model_name || "" : "")
      && entry.role === message.role
      && entry.content === message.content
      && entry.reasoning === (message.reasoning || "")
      && entry.pending === Boolean(message.pending)
      && entry.error === (message.error || "")
      && entry.notice === (message.notice || "")
      && entry.metrics === messageMetrics(message)
      && entry.attachments === attachmentsKey(message)
      && entry.swarm === swarmKey(message)
      && entry.tail === messageTail(message, index);
  }

  function cacheMessage(message, index, element) {
    messageCache.set(message, {
      index, element,
      docName: state.doc ? state.doc.model_name || "" : "",
      role: message.role,
      content: message.content,
      reasoning: message.reasoning || "",
      pending: Boolean(message.pending),
      error: message.error || "",
      notice: message.notice || "",
      metrics: messageMetrics(message),
      attachments: attachmentsKey(message),
      swarm: swarmKey(message),
      tail: messageTail(message, index),
    });
    element.__actionsKey = actionsKey(message, index);
  }

  /* Reconcile the message list by element identity: appending a turn only
     touches the new article, and a finished stream rebuilds only the reply.
     Without this, every send and every completed reply re-parsed the whole
     conversation Markdown and re-highlighted all its code blocks. */
  function renderMessages(options = {}) {
    state.needsRender = false;
    cancelLiveReply();
    cancelSwarmCard();
    const messages = $("chat-messages");
    const wasPinned = state.pinned && nearBottom();
    const previousTop = messages.scrollTop;
    if (messageCacheDoc !== state.doc) {
      messageCacheDoc = state.doc;
      messageCache = new WeakMap();
    }
    const desired = [];
    const reused = [];
    if (state.doc && state.doc.messages.length) {
      const eagerFrom = Math.max(0, state.doc.messages.length - 2);
      state.doc.messages.forEach((message, index) => {
        const entry = messageCache.get(message);
        if (messageCacheMatches(entry, message, index)) {
          desired.push(entry.element);
          reused.push({ element: entry.element, message, index });
        } else {
          const element = messageElement(message, index, { lazy: index < eagerFrom });
          cacheMessage(message, index, element);
          desired.push(element);
        }
      });
    }
    if (!desired.length) {
      Studio.ui.render(messages);
      const entry = currentModel();
      if (!entry) {
        const hasModels = Studio.models.list.some(model => model.capabilities.chat && model.available);
        messages.append(Studio.ui.empty({
          icon: "chat", title: "Choose a model",
          text: hasModels
            ? "Pick a chat model in the toolbar to start the conversation."
            : "Add or download a chat model, or connect a provider to use one from elsewhere.",
          actions: hasModels ? [
            Studio.ui.button("Choose a model", { kind: "primary", onclick: () => $("chat-model").focus() }),
            Studio.ui.button("Model settings", { onclick: () => $("chat-settings").click() }),
          ] : [
            Studio.ui.button("Find a model", { kind: "primary", onclick: () => Studio.views.go("models", { tab: "discover" }) }),
            Studio.ui.button("Add an endpoint", { onclick: () => $("chat-provider").click() }),
          ],
        }));
      } else if (!entry.available) {
        messages.append(Studio.ui.empty({
          icon: "alert", title: `${entry.name} is unavailable`,
          text: entry.reason || "Start its runtime or choose another model.",
          actions: [
            Studio.ui.button("Choose another model", { kind: "primary", onclick: () => $("chat-model").focus() }),
            Studio.ui.button("Model settings", { onclick: () => $("chat-settings").click() }),
          ],
        }));
      } else {
        messages.append(Studio.ui.empty({
          icon: "chat", title: `Chat with ${entry.name}`,
          text: "Messages are saved on this machine. Attach a file, ask for a design, or open the preview panel for code without leaving the chat.",
          actions: [0, 2, 5].map(index => Studio.ui.button(PROMPTS[index][0], {
            small: true, onclick: () => applyPrompt(PROMPTS[index][1]),
          })),
        }));
      }
    } else {
      const existing = Array.from(messages.children);
      let keep = 0;
      while (keep < existing.length && keep < desired.length && existing[keep] === desired[keep]) keep += 1;
      if (keep < existing.length || keep < desired.length) {
        for (let index = existing.length - 1; index >= keep; index -= 1) existing[index].remove();
        if (keep < desired.length) {
          const fragment = document.createDocumentFragment();
          for (const element of desired.slice(keep)) fragment.append(element);
          messages.append(fragment);
        }
      }
      for (const item of reused) {
        const key = actionsKey(item.message, item.index);
        if (item.element.__actionsKey === key) continue;
        item.element.__actionsKey = key;
        const head = item.element.querySelector(".message-head");
        const actions = head ? head.querySelector(".message-actions") : null;
        if (actions) actions.replaceWith(messageActions(item.message, item.index));
      }
    }
    updateHeader();
    if (options.keepScroll && !wasPinned) {
      requestAnimationFrame(() => { messages.scrollTop = previousTop; updateScrollButton(); });
    } else {
      scrollMessages(true);
    }
  }

  function modelContext(entry) {
    const detail = (entry && entry.detail) || {};
    const unit = detail.ctx_unit === "bytes" ? "bytes" : "tokens";
    const settings = (state.doc && state.doc.settings) || {};
    const configured = unit === "tokens" ? Number(settings.n_ctx) || 0 : 0;
    const known = configured || Number(detail.ctx_len) || Number(detail.context_length)
      || Number(detail.context_window) || 0;
    return {
      unit,
      window: known || (unit === "tokens" ? ASSUMED_CONTEXT_TOKENS : 0),
      assumed: !known,
    };
  }

  function estimateContext() {
    const entry = currentModel();
    const context = modelContext(entry);
    const all = state.doc ? state.doc.messages : [];
    const messages = [];
    let pending = 0;
    for (const message of all) {
      if (message.pending) pending += 1;
      else if (message.role !== "system") messages.push(message);
    }
    const system = state.doc ? state.doc.system_prompt || "" : "";
    const cache = state.budgetCache;
    let history = 0;
    let sent = 0;
    let attachmentTokens = 0;
    let attachments = 0;
    let systemTokens = 0;
    if (cache && cache.ref === all && cache.length === messages.length
        && cache.pending === pending && cache.system === system
        && cache.unit === context.unit) {
      ({ history, sent, attachmentTokens, attachments, systemTokens } = cache);
    } else {
      for (const message of messages) {
        const { extra, prompt } = messagePromptParts(message);
        if (!prompt) continue;
        history += Studio.tokens.estimate(`${prompt}\n${message.role}`, context.unit) + 4;
        sent += 1;
        if (extra) {
          attachmentTokens += Studio.tokens.estimate(extra, context.unit);
          attachments += (message.attachments || []).length;
        }
      }
      systemTokens = Studio.tokens.estimate(system, context.unit);
      state.budgetCache = {
        ref: all, length: messages.length, pending, system, unit: context.unit,
        history, sent, attachmentTokens, attachments, systemTokens,
      };
    }
    const draftExtra = attachmentPrompt(state.attachments);
    const draftAttachmentTokens = Studio.tokens.estimate(draftExtra, context.unit);
    const draft = Studio.tokens.estimate($("chat-input") ? $("chat-input").value : "", context.unit);
    const used = systemTokens + history + draftAttachmentTokens + draft + 8;
    return {
      ...context,
      used,
      system: systemTokens,
      history,
      attachmentTokens: attachmentTokens + draftAttachmentTokens,
      draftAttachmentTokens,
      attachments: attachments + state.attachments.length,
      messageCount: sent,
      draft,
      percent: context.window ? used * 100 / context.window : 0,
    };
  }

  function updateBudgetMeter() {
    const track = $("chat-budget-track");
    const fill = $("chat-budget-fill");
    const button = $("chat-budget-open");
    const text = $("chat-budget-text");
    const row = track ? track.closest(".chat-budget") : null;
    const live = estimateContext();
    updateContextLine(live);
    if (fill) fill.setAttribute("stroke-dasharray", String(BUDGET_RING));
    if (!live.window) {
      if (row) row.classList.remove("warning", "danger");
      if (fill) fill.setAttribute("stroke-dashoffset", String(BUDGET_RING));
      if (track) {
        track.setAttribute("aria-valuenow", "0");
        track.setAttribute("aria-valuetext", "Context estimate unavailable");
      }
      if (text) text.textContent = "Context window unknown";
      if (button) button.title = "Open context and token analysis";
      return;
    }
    const percent = Math.max(0, live.percent);
    const capped = Math.min(100, percent);
    if (row) {
      row.classList.toggle("warning", percent >= 80 && percent <= 100);
      row.classList.toggle("danger", percent > 100);
    }
    if (fill) fill.setAttribute("stroke-dashoffset", String(BUDGET_RING * (1 - capped / 100)));
    if (track) {
      track.setAttribute("aria-valuenow", String(Math.round(capped)));
      track.setAttribute("aria-valuetext",
        `${format.number(Math.round(live.used))} of ${format.number(live.window)} ${live.unit} used`);
    }
    if (text) text.textContent = `${format.compact(live.used)} / ${format.compact(live.window)} ${live.unit} · ${format.percent(percent)}`;
    if (button) {
      button.title = `${format.number(Math.round(live.used))} of ${format.number(live.window)} ${live.unit}`
        + `${live.assumed ? " (assumed window; set Context tokens in Model settings)" : ""}`;
    }
  }

  function updateContextLine(live = estimateContext()) {
    const entry = currentModel();
    if (!state.doc) {
      $("chat-context").textContent = entry ? `${entry.kind_label} · ${entry.source_label}` : "No chat-ready model found";
      return;
    }
    if (!entry) {
      $("chat-context").textContent = "No chat-ready model found";
      return;
    }
    const pieces = [entry.kind_label, entry.source_label];
    if (state.doc.pinned) pieces.push("pinned");
    if (state.mode === "swarm") pieces.push(`swarm: ${state.swarmOptions.strategy.replace(/_/g, " ")} · ${Math.round(state.swarmOptions.max_seconds / 60)} min`);
    pieces.push(`${live.attachments} attachment${live.attachments === 1 ? "" : "s"}`);
    if (live.window) pieces.push(`${format.compact(live.used)} / ${format.compact(live.window)} ${live.unit} (${format.percent(live.percent)})`);
    else pieces.push("context window unknown");
    $("chat-context").textContent = pieces.join(" · ");
  }

  function updateHeader() {
    const entry = currentModel();
    $("chat-stop").classList.toggle("hidden", !state.streaming);
    updateModeControls();
    $("chat-runtime").classList.toggle("hidden", !entry || entry.kind !== "ollama" || entry.available);
    $("chat-settings-model").textContent = entry ? `${entry.name} · ${entry.source_label}` : "";
    const notice = Studio.ui.render($("chat-notice"));
    if (entry && !entry.available) notice.append(Studio.ui.callout("warning", entry.reason || "This model is unavailable."));
    updateBudgetMeter();
    updateComposerState();
    updateVoiceControls();
  }

  /* -- voice ---------------------------------------------------------------- */
  function voiceIdleTitle() {
    const status = state.voiceStatus;
    if (!status) return "Dictate a message";
    if (status.available) {
      return `Dictate with local speech recognition (${status.label}); audio stays on this machine`;
    }
    if (Studio.voice && Studio.voice.browserSupported()) {
      return "Dictate with browser speech recognition; your browser may use a remote service";
    }
    const install = status.install ? ` ${status.install}` : "";
    return `Voice input needs a local Whisper backend or browser support.${install}`;
  }

  function updateVoiceButton() {
    const button = $("chat-voice");
    if (!button) return;
    const active = ["recording", "listening", "transcribing"].includes(state.voiceState);
    button.disabled = state.streaming;
    button.classList.toggle("recording", state.voiceState === "recording" || state.voiceState === "listening");
    button.classList.toggle("transcribing", state.voiceState === "transcribing");
    button.setAttribute("aria-pressed", String(active));
    button.title = state.voiceState === "transcribing" ? "Transcribing voice input…"
      : active ? "Stop dictation" : voiceIdleTitle();
    button.setAttribute("aria-label", button.title);
  }

  function updateVoiceControls() {
    updateVoiceButton();
    const auto = $("chat-voice-auto");
    auto.setAttribute("aria-pressed", String(state.voiceAuto));
    auto.title = state.voiceAuto ? "Stop reading replies aloud automatically"
      : "Read replies aloud automatically";
    auto.setAttribute("aria-label", auto.title);
  }

  function stopDictation(cancel = false) {
    const handle = state.voice;
    if (cancel) {
      state.voiceToken += 1;
      state.voice = null;
      state.voiceState = "idle";
      if (handle) {
        try { handle.cancel(); } catch (error) { /* already stopped */ }
      }
      updateVoiceControls();
      return;
    }
    if (!handle) {
      state.voiceState = "idle";
      updateVoiceControls();
      return;
    }
    try {
      handle.stop();
    } catch (error) {
      state.voice = null;
      state.voiceState = "idle";
      updateVoiceControls();
    }
  }

  async function toggleDictation() {
    if (!Studio.voice) {
      Studio.toast("Voice support did not load; reload Studio and try again.", "error");
      return;
    }
    if (state.voice) {
      if (state.voiceState === "transcribing") stopDictation(true);
      else stopDictation(false);
      return;
    }
    if (state.voiceState === "starting") {
      state.voiceToken += 1;
      state.voiceState = "idle";
      updateVoiceControls();
      return;
    }
    const token = ++state.voiceToken;
    state.voiceState = "starting";
    updateVoiceControls();
    let failed = false;
    const input = $("chat-input");
    const insert = text => {
      if (token !== state.voiceToken) return;
      const caret = Number.isFinite(input.selectionStart) ? input.selectionStart : input.value.length;
      const before = input.value.slice(0, caret);
      const space = before && !/\s$/.test(before) ? " " : "";
      input.setRangeText(`${space}${text}`, caret, caret, "end");
      input.focus();
      onComposerInput();
    };
    let active = null;
    try {
      active = await Studio.voice.dictate({
        language: document.documentElement.lang || navigator.language || "en-US",
        onState(voiceState) {
          if (token !== state.voiceToken) return;
          state.voiceState = voiceState;
          updateVoiceButton();
          if (voiceState === "transcribing") $("chat-stream-status").textContent = "Transcribing voice input…";
          if (voiceState === "idle") {
            state.voice = null;
            state.voiceState = "idle";
            updateVoiceControls();
            updateBudgetMeter();
            if (!failed) $("chat-stream-status").textContent = "Voice input ready. Review the text, then send.";
          }
        },
        onText: text => insert(text),
        onInterim(text) {
          if (token === state.voiceToken) $("chat-stream-status").textContent = text ? `Hearing: ${text}` : "Listening…";
        },
        onError(error) {
          if (token !== state.voiceToken) return;
          failed = true;
          state.voice = null;
          state.voiceState = "idle";
          updateVoiceControls();
          Studio.toast(error.message, "error");
        },
      });
      if (token !== state.voiceToken) {
        active.cancel();
        return;
      }
      state.voice = active;
      updateVoiceControls();
      $("chat-stream-status").textContent = active.mode === "local"
        ? "Recording locally. Click the microphone again to stop and transcribe."
        : "Listening with browser speech recognition. Click the microphone again to stop.";
    } catch (error) {
      if (token !== state.voiceToken) return;
      state.voice = null;
      state.voiceState = "idle";
      updateVoiceControls();
      Studio.toast(error.message, "error");
    }
  }

  let budgetFrame = 0;

  /* Typing re-estimates the whole conversation; throttle it to one frame so a
     long transcript with attachments cannot make the composer lag. */
  function scheduleBudgetMeter() {
    if (budgetFrame) return;
    budgetFrame = requestAnimationFrame(() => {
      budgetFrame = 0;
      updateBudgetMeter();
    });
  }

  function onComposerInput() {
    resizeComposer();
    updateComposerState();
    scheduleBudgetMeter();
  }

  /* -- artifact preview ----------------------------------------------------- */
  function allArtifacts() {
    const artifacts = state.doc ? Studio.artifacts.collect(state.doc.messages) : [];
    return artifacts.concat(
      Studio.artifacts.fromAttachments(state.attachments, "draft", -1),
      Array.from(state.preview.workspace.values()),
    );
  }

  /* Files a swarm wrote live in the conversation workspace.  Fetch only the
     paths named by a run (bounded), and cache each artifact by path. */
  async function loadWorkspaceFiles(paths, options = {}) {
    if (!state.doc || !state.doc.id) return [];
    const chatId = state.doc.id;
    const refresh = Boolean(options.refresh);
    const unique = [...new Set((Array.isArray(paths) ? paths : [])
      .map(path => String(path || "").trim()).filter(Boolean))];
    const loaded = [];
    for (const path of unique.slice(0, 8)) {
      const cached = refresh ? null : state.preview.workspace.get(path);
      if (cached) { loaded.push(cached); continue; }
      if (refresh) state.preview.workspace.delete(path);
      try {
        const data = await api(`/api/chats/${encodeURIComponent(chatId)}/file`
          + `?path=${encodeURIComponent(path)}`);
        if (!state.doc || state.doc.id !== chatId) return loaded;
        const artifact = Studio.artifacts.workspaceArtifact(data && data.file ? data.file : data);
        if (artifact) {
          state.preview.workspace.set(path, artifact);
          loaded.push(artifact);
        }
      } catch (error) {
        // A deleted, binary or oversized file simply has no preview.
      }
    }
    return loaded;
  }

  function currentArtifact(artifacts) {
    const list = artifacts || allArtifacts();
    if (!list.length) {
      state.preview.selected = null;
      return null;
    }
    const selected = state.preview.selected
      ? list.find(artifact => artifact.id === state.preview.selected.id) : null;
    if (selected) {
      state.preview.selected = selected;
      return selected;
    }
    state.preview.selected = list[list.length - 1];
    state.preview.view = state.preview.selected.previewable ? "preview" : "source";
    return state.preview.selected;
  }

  function previewIcon(kind) {
    return {
      html: "preview", svg: "preview", mermaid: "layout", markdown: "book", table: "table",
      image: "eye", json: "code", code: "code", file: "file",
    }[kind] || "preview";
  }

  function updatePreviewControls(view) {
    if (!view) return;
    const wasOpen = state.preview.open;
    const open = Boolean(view.agent.visible && !view.agent.collapsed);
    if (wasOpen && !open) {
      // A splitter collapse counts as a deliberate close too.
      state.preview.closedIds = new Set(allArtifacts().map(artifact => artifact.id));
    }
    state.preview.open = open;
    const button = $("chat-preview-toggle");
    if (!view.agent.visible) {
      const active = document.activeElement;
      if (active && (active === $("chat-split-preview") || $("chat-preview").contains(active))) {
        button.focus();
      }
    }
    const count = state.preview.count;
    button.setAttribute("aria-pressed", String(state.preview.open));
    button.title = `${state.preview.open ? "Hide" : "Show"} preview panel`
      + `${count ? ` · ${count} artifact${count === 1 ? "" : "s"}` : ""} (Ctrl+Shift+P)`;
    button.setAttribute("aria-label", button.title);
  }

  function ensurePreviewOpen() {
    if (!layout) return;
    const view = layout.read();
    if (view && view.agent.collapsed) layout.setCollapsed("agent", false);
  }

  function selectArtifact(artifact) {
    const changed = !state.preview.selected || !artifact || state.preview.selected.id !== artifact.id;
    state.preview.selected = artifact;
    if (!artifact || !artifact.previewable) state.preview.view = "source";
    else if (artifact.kind === "image" || changed) state.preview.view = "preview";
    else if (state.preview.view !== "source") state.preview.view = "preview";
    renderPreview();
  }

  function openArtifact(artifact) {
    if (!artifact) return;
    state.preview.selected = artifact;
    state.preview.view = artifact.previewable ? "preview" : "source";
    state.preview.closedIds = new Set();
    ensurePreviewOpen();
    renderPreview();
  }

  function setAutoPreview(value) {
    state.autoPreview = Boolean(value);
    Studio.store.set("chat-auto-preview", state.autoPreview);
    Studio.toast(state.autoPreview
      ? "Previews open automatically when code, designs or attachments appear."
      : "Automatic previews are off; use the preview button or a code block button.", "info");
    renderPreview();
  }

  function renderPreviewList(artifacts, selected) {
    const list = Studio.ui.render($("chat-preview-list"));
    for (const artifact of artifacts) {
      const active = Boolean(selected && artifact.id === selected.id);
      list.append(h("button", {
        class: "chat-preview-tab", type: "button", role: "tab",
        "aria-selected": String(active),
        "aria-controls": "chat-preview-body",
        tabIndex: active ? 0 : -1,
        title: artifact.title,
        onclick: () => selectArtifact(artifact),
      }, icon(previewIcon(artifact.kind), "small"), h("span", { class: "truncate", text: artifact.title })));
    }
  }

  function renderPreviewModes(artifact) {
    const root = Studio.ui.render($("chat-preview-modes"));
    if (!artifact || !artifact.previewable || artifact.kind === "image") return;
    const group = h("div", { class: "segmented", role: "group", "aria-label": "Preview mode" });
    for (const [value, label] of [["preview", "Preview"], ["source", "Source"]]) {
      group.append(h("button", {
        type: "button", "aria-pressed": String(state.preview.view === value), text: label,
        onclick: () => { state.preview.view = value; renderPreview(); },
      }));
    }
    root.append(group);
  }

  function renderPreview() {
    const focusedTab = document.activeElement && document.activeElement.getAttribute
      && document.activeElement.getAttribute("role") === "tab"
      && $("chat-preview-list").contains(document.activeElement);
    const focusedMode = Boolean(document.activeElement && document.activeElement.getAttribute
      && $("chat-preview-modes").contains(document.activeElement));
    const artifacts = allArtifacts();
    const selected = currentArtifact(artifacts);
    state.preview.count = artifacts.length;
    $("chat-preview-toggle").classList.toggle("has-artifacts", Boolean(artifacts.length));
    updatePreviewControls(layout && layout.read());
    $("chat-preview-copy").disabled = !selected || (!selected.source && (!selected.data_url || selected.kind === "image"));
    $("chat-preview-download").disabled = !selected || (!selected.source && !selected.data_url);
    $("chat-preview-title").textContent = selected ? selected.title : "Preview";
    $("chat-preview-subtitle").textContent = selected
      ? `${Studio.artifacts.label(selected.kind, selected.language)} · ${format.bytes(selected.size || 0)}`
        + `${selected.filename ? ` · ${selected.filename}` : ""}`
      : "Code, designs, diagrams and attachments from this conversation.";
    renderPreviewModes(selected);
    renderPreviewList(artifacts, selected);
    const renderKey = selected
      ? `${selected.id}:${state.preview.view}:${selected.version || 0}` : "empty";
    const changed = renderKey !== state.preview.rendered;
    const body = $("chat-preview-body");
    if (!selected) {
      Studio.ui.render(body, Studio.ui.empty({
        compact: true, icon: "preview", title: "Nothing to preview yet",
        text: "Ask for a design or code block, attach an image, or let a swarm write a file in the conversation workspace. Previews also appear here from every reply in the conversation.",
        actions: [
          Studio.ui.button("Attach a file", { small: true, kind: "primary", onclick: () => $("chat-file-input").click() }),
          Studio.ui.button(state.autoPreview ? "Turn off automatic previews" : "Open previews automatically", {
            small: true, onclick: () => setAutoPreview(!state.autoPreview),
          }),
        ],
      }));
    } else {
      Studio.ui.render(body, Studio.artifacts.render(selected, state.preview.view));
    }
    if (changed) {
      body.scrollTop = 0;
      state.preview.rendered = renderKey;
    }
    if (focusedTab) {
      const active = $("chat-preview-list").querySelector('[aria-selected="true"]');
      if (active) active.focus();
    } else if (focusedMode) {
      const active = $("chat-preview-modes").querySelector('[aria-pressed="true"]')
        || $("chat-preview-list").querySelector('[aria-selected="true"]');
      if (active) active.focus();
    }
  }

  function refreshPreviewSelection() {
    const artifacts = allArtifacts();
    state.preview.count = artifacts.length;
    $("chat-preview-toggle").classList.toggle("has-artifacts", Boolean(artifacts.length));
    currentArtifact(artifacts);
    if (state.preview.open) renderPreview();
    else updatePreviewControls(layout && layout.read());
  }

  function togglePreview() {
    if (!layout) return;
    const view = layout.read();
    if (!view) return;
    const visible = view.agent.visible && !view.agent.collapsed;
    if (!visible) {
      state.preview.closedIds = new Set();
      ensurePreviewOpen();
      const after = layout.read();
      if (after && !after.agent.visible) {
        Studio.toast("The preview panel needs more room: widen the window or close the conversation list.", "info");
      }
    } else {
      const focusInPreview = $("chat-preview").contains(document.activeElement);
      state.preview.closedIds = new Set(allArtifacts().map(artifact => artifact.id));
      layout.setCollapsed("agent", true);
      if (focusInPreview) $("chat-preview-toggle").focus();
    }
    renderPreview();
  }

  function autoPreviewArtifact(artifact) {
    if (!state.autoPreview || !artifact) return;
    const artifacts = allArtifacts();
    if (!artifacts.some(item => !state.preview.closedIds.has(item.id))) return;
    openArtifact(artifact);
  }

  /* -- token analysis ------------------------------------------------------- */
  function tokenSummary() {
    const live = estimateContext();
    const children = [];
    const tone = !live.window ? "warning" : live.percent > 100 ? "danger" : live.percent >= 80 ? "warning" : "";
    children.push(h("div", { class: "row between wrap" },
      h("div", {},
        h("strong", { text: `~${format.compact(live.used)} ${live.unit}` }),
        h("span", { class: "muted", text: live.window
          ? ` of ${format.compact(live.window)} · ${format.percent(live.percent)}`
          : " · context window unknown" })),
      Studio.ui.badge(live.unit === "bytes" ? "exact bytes" : "estimated", live.assumed ? "warning" : "")));
    children.push(Studio.ui.meter({
      label: "Used", value: live.used, total: live.window || Math.max(live.used, 1),
      text: live.window ? format.percent(live.percent) : `${format.compact(live.used)} ${live.unit}`,
      tone,
    }));
    if (live.attachments) {
      children.push(h("p", { class: "help",
        text: `${live.attachments} attachment(s) contribute about ${format.compact(live.attachmentTokens)} ${live.unit}; they are counted inside the conversation total.` }));
    }
    return children;
  }

  function renderTokenAnalysis(body) {
    const live = estimateContext();
    const total = Math.max(live.used, 1);
    const rows = h("div", { class: "stack tight" });
    rows.append(Studio.ui.meter({
      label: "System prompt", value: live.system, total, text: `${format.compact(live.system)} ${live.unit}`,
    }));
    rows.append(Studio.ui.meter({
      label: `Conversation (${live.messageCount} message${live.messageCount === 1 ? "" : "s"})`,
      value: live.history, total, text: `${format.compact(live.history)} ${live.unit}`,
    }));
    rows.append(Studio.ui.meter({
      label: "Current draft", value: live.draft, total, text: `${format.compact(live.draft)} ${live.unit}`,
    }));
    if (live.draftAttachmentTokens) {
      rows.append(Studio.ui.meter({
        label: `Draft attachments (${state.attachments.length})`, value: live.draftAttachmentTokens, total,
        text: `${format.compact(live.draftAttachmentTokens)} ${live.unit}`,
      }));
    }
    const entry = currentModel();
    const last = state.doc && state.doc.messages.slice().reverse()
      .find(message => message.role === "assistant" && message.usage);
    const children = [
      h("div", { class: "stack" }, tokenSummary()),
      h("p", { class: "help",
        text: `${entry ? entry.name : "No model selected"}`
          + `${live.assumed ? " · context window assumed; set Context tokens in Model settings or on the endpoint" : " · context window from the catalog"}` }),
      rows,
    ];
    if (last && last.usage && (last.usage.prompt_tokens || last.usage.completion_tokens)) {
      children.push(h("p", { class: "help",
        text: `Last run: ${format.number(last.usage.prompt_tokens || 0)} prompt + `
          + `${format.number(last.usage.completion_tokens || 0)} completion tokens. `
          + "Provider counts are exact where the model reports them." }));
    }
    if (live.window && live.percent >= 80) {
      children.push(Studio.ui.callout(live.percent > 100 ? "danger" : "warning",
        h("span", { text: live.percent > 100
          ? "The next request is larger than the context window. Remove messages or attachments, or start a new chat."
          : "The next request is close to the context window. Consider starting a new chat or removing attachments." })));
    }
    Studio.ui.render(body, h("div", { class: "stack" }, children));
  }

  function openTokenDialog() {
    const handle = Studio.overlay.modal({
      key: "chat-tokens", title: "Context and tokens", wide: true,
      subtitle: "Estimated from the system prompt, saved messages, attachments and the draft. "
        + "The ring in the composer updates as you type.",
    });
    if (handle.reused) return;
    const body = h("div");
    Studio.ui.render(handle.body, body);
    renderTokenAnalysis(body);
    handle.footer.append(
      Studio.ui.button("Refresh estimate", { small: true, onclick: () => renderTokenAnalysis(body) }),
      Studio.ui.button("Done", { kind: "primary", onclick: handle.close }));
  }

  const saveQueues = new WeakMap();

  async function persistChat(target) {
    const messages = target.messages.filter(message => !message.pending);
    const payload = {
      pinned: target.pinned, model: target.model, model_name: target.model_name,
      system_prompt: target.system_prompt, settings: target.settings, messages,
    };
    // An automatic title must keep re-deriving from the first user turn; only a
    // title the user chose is sent back and frozen.
    if (target.titled) payload.title = target.title;
    const data = target.id
      ? await api(`/api/chats/${encodeURIComponent(target.id)}`, { method: "PUT", body: payload })
      : await api("/api/chats", { body: payload });
    const saved = data && data.chat && typeof data.chat === "object" ? data.chat : null;
    if (!saved || !saved.id) {
      throw new Error("The server did not return the saved conversation. Check that Studio is still running.");
    }
    const merged = { ...target, ...saved, messages: target.messages };
    Object.assign(target, merged);
    if (state.doc === target) {
      state.doc = target;
      if (target.id && Studio.views.current === "chat"
          && window.location.hash.startsWith("#/chat")) {
        const address = `#/chat${Studio.query({ id: target.id })}`;
        if (window.location.hash !== address) history.replaceState(null, "", address);
      }
    }
    try {
      await refreshChats();
    } catch (error) {
      // The conversation is saved; a failed sidebar refresh must not fail it.
      console.error("Refreshing conversations failed", error);
    }
    return target;
  }

  /* Saves are serialised per conversation: renaming or pinning while the first
     save of a new chat is still in flight must not create a second chat. */
  function saveChat(target = state.doc) {
    if (!target) return Promise.resolve(null);
    const prior = saveQueues.get(target) || Promise.resolve();
    const task = prior.catch(() => null).then(() => persistChat(target));
    saveQueues.set(target, task);
    return task;
  }

  function resetComposer() {
    state.attachments = [];
    $("chat-input").value = "";
    renderAttachments();
    resizeComposer();
    updateComposerState();
    updateBudgetMeter();
  }

  /* Stop the current stream and give up ownership of the conversation UI.
     Late stream events and saves stay attached to their own document object,
     so opening another chat mid-generation can never write into the new one. */
  function abandonGeneration() {
    const wasStreaming = state.streaming;
    state.runToken += 1;
    cancelLiveReply();
    endLive();
    cancelSwarmCard();
    const controller = state.abort;
    state.abort = null;
    state.streaming = false;
    if (wasStreaming) state.needsRender = true;
    $("chat-input").readOnly = false;
    $("chat-model").disabled = false;
    $("chat-provider").disabled = false;
    $("chat-settings").disabled = false;
    $("chat-prompts").disabled = false;
    $("chat-mode-solo").disabled = state.loading;
    $("chat-mode-swarm").disabled = state.loading;
    $("chat-stream-status").textContent = "";
    if (controller) controller.abort();
    updateComposerState();
    updateVoiceControls();
    $("chat-stop").classList.add("hidden");
  }

  function newChat(modelId) {
    abandonGeneration();
    state.loadToken += 1;
    stopDictation(true);
    if (Studio.voice) Studio.voice.stopSpeaking();
    state.spokenMessage = null;
    state.pinned = true;
    state.doc = newDocument(modelId || (state.doc && state.doc.model));
    state.preview.selected = null;
    state.preview.closedIds = new Set();
    state.preview.view = "preview";
    state.preview.rendered = "";
    state.preview.workspace = new Map();
    resetComposer();
    history.replaceState(null, "", `#/chat${Studio.query({ model: state.doc.model, new: "1" })}`);
    Studio.ui.fillModels($("chat-model"), { filter: entry => entry.capabilities.chat, value: state.doc.model, includeUnavailable: true });
    renderMessages();
    renderChatList();
    renderPreview();
    if (layout) layout.setCollapsed("agent", true, false);
    $("chat-input").focus();
  }

  async function loadChat(id) {
    if (!id) return newChat();
    if (state.doc && state.doc.id === id) {
      // Returning to the open conversation must not discard an unsent draft.
      if (!state.streaming) {
        Studio.ui.fillModels($("chat-model"), {
          filter: entry => entry.capabilities.chat, value: state.doc.model, includeUnavailable: true,
        });
        if (state.needsRender) renderMessages();
        updateHeader();
        updateComposerState();
        renderPreview();
      }
      return;
    }
    abandonGeneration();
    const token = ++state.loadToken;
    stopDictation(true);
    if (Studio.voice) Studio.voice.stopSpeaking();
    state.spokenMessage = null;
    state.preview.closedIds = new Set();
    state.preview.view = "preview";
    state.preview.rendered = "";
    let data;
    try {
      data = await api(`/api/chats/${encodeURIComponent(id)}`);
    } catch (error) {
      if (token !== state.loadToken) return;
      throw error;
    }
    if (token !== state.loadToken) return;
    state.pinned = true;
    const document = data && data.chat && typeof data.chat === "object" ? data.chat : {};
    if (!Array.isArray(document.messages)) document.messages = [];
    document.messages = document.messages.filter(message => message && typeof message === "object"
      && ["user", "assistant", "system"].includes(message.role));
    for (const message of document.messages) {
      if (typeof message.content !== "string") message.content = "";
      if (!Array.isArray(message.attachments)) delete message.attachments;
    }
    if (!document.settings || typeof document.settings !== "object") document.settings = {};
    if (typeof document.system_prompt !== "string") document.system_prompt = "";
    if (typeof document.title !== "string" || !document.title.trim()) document.title = "New chat";
    state.doc = document;
    state.preview.selected = null;
    state.preview.workspace = new Map();
    resetComposer();
    const selected = Studio.ui.fillModels($("chat-model"), { filter: entry => entry.capabilities.chat, value: state.doc.model, includeUnavailable: true });
    if (selected && selected !== state.doc.model) {
      const entry = Studio.models.get(selected);
      state.doc.model = selected;
      state.doc.model_name = entry ? entry.name : selected;
    }
    state.doc.settings = normaliseSettings({ ...defaults(Studio.models.get(state.doc.model) || { id: state.doc.model }), ...(state.doc.settings || {}) });
    renderMessages();
    renderChatList();
    resizeComposer();
    renderPreview();
    const written = [];
    for (const message of state.doc.messages) {
      const files = message.swarm && message.swarm.meta && message.swarm.meta.changedFiles;
      if (Array.isArray(files)) written.push(...files);
    }
    if (written.length) {
      loadWorkspaceFiles(written).then(() => {
        if (state.doc && state.doc.id === id) refreshPreviewSelection();
      });
    }
  }

  function setStreaming(value) {
    state.streaming = value;
    $("chat-input").readOnly = value;
    $("chat-model").disabled = value;
    $("chat-provider").disabled = value;
    $("chat-settings").disabled = value;
    $("chat-prompts").disabled = value;
    $("chat-mode-solo").disabled = value || state.loading;
    $("chat-mode-swarm").disabled = value || state.loading;
    updateHeader();
  }

  /* -- swarm replies --------------------------------------------------------- */
  function applySwarmEvent(message, event) {
    if (!event || typeof event !== "object") return;
    if (!message.swarm) message.swarm = Studio.swarm.create();
    if (event.snapshot) Studio.swarm.applySnapshot(message.swarm, event.snapshot);
    if (event.type === "done") {
      if (event.swarm) Studio.swarm.finish(message.swarm, event.swarm);
      return;
    }
    if (event.type === "error" && event.swarm) {
      Studio.swarm.finish(message.swarm, event.swarm);
      return;
    }
    Studio.swarm.applyEvent(message.swarm, event);
  }

  let swarmTimer = 0;
  let swarmCardMessage = null;

  function paintSwarmCard(message) {
    if (!message || !state.doc || !message.swarm) return;
    const index = state.doc.messages.indexOf(message);
    const article = index < 0 ? null : $("chat-messages").children[index];
    const holder = article ? article.querySelector(".message-swarm") : null;
    if (!holder) return;
    const active = document.activeElement && holder.contains(document.activeElement)
      ? document.activeElement : null;
    const section = active ? active.closest("details[data-swarm-key]") : null;
    const focusKey = section ? section.getAttribute("data-swarm-key") : null;
    const scrolls = new Map();
    const pinned = new Map();
    for (const node of holder.querySelectorAll("[data-swarm-output]")) {
      const key = node.getAttribute("data-swarm-output");
      scrolls.set(key, node.scrollTop);
      pinned.set(key, node.scrollHeight - node.scrollTop - node.clientHeight < 24);
    }
    Studio.swarm.render(holder, message.swarm, { live: Boolean(message.pending) });
    for (const node of holder.querySelectorAll("[data-swarm-output]")) {
      const key = node.getAttribute("data-swarm-output");
      if (pinned.get(key)) {
        node.scrollTop = node.scrollHeight;
      } else {
        const top = scrolls.get(key);
        if (top) node.scrollTop = top;
      }
    }
    if (focusKey) {
      const next = Array.from(holder.querySelectorAll("details[data-swarm-key]"))
        .find(node => node.getAttribute("data-swarm-key") === focusKey);
      const control = next ? next.querySelector("summary") : null;
      if (control) control.focus({ preventScroll: true });
    }
    if (message.pending) scrollMessages();
  }

  /* Agent deltas can arrive dozens of times a second.  Redraw the card at most
     ~8 times a second, immediately on a terminal event; a full message render
     would discard the reply Markdown the user is already reading. */
  function scheduleSwarmCard(message, immediate = false) {
    swarmCardMessage = message;
    if (immediate && swarmTimer) { clearTimeout(swarmTimer); swarmTimer = 0; }
    if (immediate) {
      const target = swarmCardMessage;
      swarmCardMessage = null;
      paintSwarmCard(target);
      return;
    }
    if (swarmTimer) return;
    swarmTimer = setTimeout(() => {
      swarmTimer = 0;
      const target = swarmCardMessage;
      swarmCardMessage = null;
      paintSwarmCard(target);
    }, 120);
  }

  function cancelSwarmCard() {
    swarmCardMessage = null;
    if (!swarmTimer) return;
    clearTimeout(swarmTimer);
    swarmTimer = 0;
  }

  async function generateSwarm() {
    if (state.streaming || !state.doc || !state.doc.model) return;
    if (state.voice || state.voiceState === "starting") stopDictation(true);
    if (Studio.voice && Studio.voice.speaking()) Studio.voice.stopSpeaking();
    state.spokenMessage = null;
    const target = state.doc;
    const messages = outgoingMessages(target);
    if (!messages.length || messages[messages.length - 1].role !== "user") {
      Studio.toast("Enter a message before starting a swarm.", "error");
      return;
    }
    const entry = Studio.models.get(target.model);
    if (!entry || !entry.available) {
      Studio.toast(entry ? entry.reason : "Choose an available model first.", "error");
      return;
    }
    const assistantIndex = target.messages.length;
    const assistant = {
      role: "assistant", content: "", model: entry.id, model_name: entry.name,
      kind: "Swarm", pending: true,
      swarm: Studio.swarm.create({ model: entry.id, model_name: entry.name }),
    };
    target.messages.push(assistant);
    renderMessages();
    setStreaming(true);
    const token = ++state.runToken;
    const controller = new AbortController();
    state.abort = controller;
    const ownsUi = () => token === state.runToken && state.doc === target;
    const dropAssistant = () => {
      const index = target.messages.indexOf(assistant);
      if (index >= 0) target.messages.splice(index, 1);
    };
    let completed = false;
    try {
      await saveChat(target);
      await Studio.stream("/api/chat/swarm/stream", {
        model: entry.id, messages, system_prompt: target.system_prompt || "", ...swarmPayload(target),
      }, {
        signal: controller.signal,
        onEvent(event) {
          applySwarmEvent(assistant, event);
          if (event.type === "done") {
            completed = true;
            assistant.content = event.reply || assistant.content;
            for (const key of ["kind", "usage", "elapsed", "tokens_per_second", "first_token_seconds", "notice", "finish_reason", "served_by"]) {
              if (event[key] !== undefined) assistant[key] = event[key];
            }
          }
          if (event.type === "error") {
            assistant.error = event.error || "The swarm failed";
            assistant.content = assistant.content || assistant.error;
            assistant.swarm.meta.finished = true;
            assistant.swarm.meta.ok = false;
            assistant.swarm.meta.status = "failed";
            assistant.swarm.meta.error = assistant.error;
          }
          if (!ownsUi()) return;
          if (event.type !== "swarm_agent_delta" && event.message) {
            $("chat-stream-status").textContent = event.message;
          }
          if (event.type === "done" || event.type === "error") {
            $("chat-stream-status").textContent = "";
          }
          scheduleSwarmCard(assistant, event.type === "done" || event.type === "error");
        },
      });
      if (!completed && !assistant.error && controller.signal.aborted) {
        if (assistant.swarm && assistant.swarm.agents.length) {
          assistant.notice = "The swarm was stopped before the leader synthesised a result. Agent findings are kept below.";
          assistant.swarm.meta.finished = true;
          assistant.swarm.meta.status = "cancelled";
        } else {
          dropAssistant();
        }
        if (ownsUi()) Studio.toast("Swarm stopped.", "info");
      }
    } catch (error) {
      if (error.name === "AbortError") {
        if (!completed && !(assistant.swarm && assistant.swarm.agents.length)) dropAssistant();
      } else {
        assistant.error = error.message;
        assistant.content = assistant.content || error.message;
        if (assistant.swarm) {
          assistant.swarm.meta.finished = true;
          assistant.swarm.meta.ok = false;
          assistant.swarm.meta.status = "failed";
          assistant.swarm.meta.error = error.message;
        }
      }
    } finally {
      delete assistant.pending;
      cancelSwarmCard();
      const current = ownsUi();
      if (current) {
        state.abort = null;
        setStreaming(false);
        $("chat-stream-status").textContent = "";
        renderMessages({ keepScroll: true });
        $("chat-input").focus();
        updateBudgetMeter();
      } else if (state.abort === controller) {
        state.abort = null;
      }
      try {
        await saveChat(target);
      } catch (error) {
        if (current) Studio.toast(error.message, "error");
        else console.error("Saving an abandoned conversation failed", error);
      }
      if (!current) {
        if (state.doc === target && !state.streaming && Studio.views.current === "chat") {
          renderMessages({ keepScroll: true });
          refreshPreviewSelection();
        }
        return;
      }
      refreshPreviewSelection();
      const changed = assistant.swarm && Array.isArray(assistant.swarm.meta.changedFiles)
        ? assistant.swarm.meta.changedFiles : [];
      if (completed || changed.length) {
        if (completed && state.voiceAuto && Studio.voice && Studio.voice.synthesisSupported()) {
          state.spokenMessage = Studio.voice.speak(assistant.content || "") ? assistant : null;
        }
        const messageArtifact = () => {
          const candidates = Studio.artifacts.messageArtifacts(assistant, assistantIndex)
            .filter(item => !item.attachment);
          autoPreviewArtifact(candidates.filter(item => item.previewable).pop() || candidates.pop());
        };
        const workspace = changed.length
          ? await loadWorkspaceFiles(changed, { refresh: true }) : [];
        if (!ownsUi()) return;
        const written = workspace.filter(item => item.previewable).pop() || workspace.pop();
        if (written) autoPreviewArtifact(written);
        else if (completed) messageArtifact();
        if (state.preview.open) renderPreview();
        else refreshPreviewSelection();
      }
    }
  }

  async function generateReply() {
    if (state.streaming || !state.doc || !state.doc.model) return;
    if (state.voice || state.voiceState === "starting") stopDictation(true);
    if (Studio.voice && Studio.voice.speaking()) Studio.voice.stopSpeaking();
    state.spokenMessage = null;
    const target = state.doc;
    const messages = outgoingMessages(target);
    if (!messages.length || messages[messages.length - 1].role !== "user") {
      Studio.toast("Enter a message before generating a response.", "error");
      return;
    }
    const entry = Studio.models.get(target.model);
    if (!entry || !entry.available) { Studio.toast(entry ? entry.reason : "Choose an available model first.", "error"); return; }
    const assistantIndex = target.messages.length;
    const assistant = { role: "assistant", content: "", reasoning: "", model: entry.id, model_name: entry.name, pending: true };
    target.messages.push(assistant);
    renderMessages();
    setStreaming(true);
    beginLive(assistant);
    const token = ++state.runToken;
    const controller = new AbortController();
    state.abort = controller;
    const ownsUi = () => token === state.runToken && state.doc === target;
    const dropAssistant = () => {
      const index = target.messages.indexOf(assistant);
      if (index >= 0) target.messages.splice(index, 1);
    };
    let completed = false;
    try {
      await saveChat(target);
      await Studio.stream("/api/chat/stream", {
        model: entry.id, messages, system_prompt: target.system_prompt || "", ...target.settings,
      }, {
        signal: controller.signal,
        onEvent(event) {
          if (!event || typeof event !== "object") return;
          if (event.type === "status" && ownsUi()) $("chat-stream-status").textContent = event.message || "Preparing model…";
          if (event.type === "delta") {
            assistant.content += event.text || "";
            if (ownsUi()) {
              beginLive(assistant);
              if (liveStream && liveStream.message === assistant) {
                if (!liveStream.firstToken) liveStream.firstToken = performance.now();
                liveStream.chars = assistant.content.length;
              }
              if ($("chat-stream-status").textContent) $("chat-stream-status").textContent = "";
              scheduleLiveReply(assistant);
            }
          }
          if (event.type === "reasoning") {
            assistant.reasoning += event.text || "";
            if (ownsUi()) {
              beginLive(assistant);
              $("chat-stream-status").textContent = "Reasoning…";
              scheduleReasoning(assistant);
            }
          }
          if (event.type === "error") {
            assistant.error = event.error || "Generation failed";
            assistant.content = assistant.content || assistant.error;
          }
          if (event.type === "done") {
            completed = true;
            assistant.content = event.reply || assistant.content;
            assistant.reasoning = event.reasoning || assistant.reasoning;
            for (const key of ["kind", "usage", "elapsed", "tokens_per_second", "first_token_seconds", "notice", "finish_reason", "served_by"]) {
              if (event[key] !== undefined) assistant[key] = event[key];
            }
          }
        },
      });
      if (!completed && !assistant.error && controller.signal.aborted) {
        dropAssistant();
        if (ownsUi()) Studio.toast("Generation stopped. The partial response was discarded.", "info");
      }
    } catch (error) {
      if (error.name === "AbortError") {
        if (!completed) dropAssistant();
      } else {
        assistant.error = error.message;
        assistant.content = assistant.content || error.message;
      }
    } finally {
      delete assistant.pending;
      endLive(assistant);
      cancelLiveReply();
      const current = ownsUi();
      if (current) {
        state.abort = null;
        setStreaming(false);
        $("chat-stream-status").textContent = "";
        renderMessages({ keepScroll: true });
        $("chat-input").focus();
        updateBudgetMeter();
      } else if (state.abort === controller) {
        state.abort = null;
      }
      try {
        await saveChat(target);
      } catch (error) {
        if (current) Studio.toast(error.message, "error");
        else console.error("Saving an abandoned conversation failed", error);
      }
      if (!current) {
        if (state.doc === target && !state.streaming && Studio.views.current === "chat") {
          renderMessages({ keepScroll: true });
          refreshPreviewSelection();
        }
        return;
      }
      refreshPreviewSelection();
      if (completed) {
        if (state.voiceAuto && Studio.voice && Studio.voice.synthesisSupported()) {
          state.spokenMessage = Studio.voice.speak(assistant.content || "") ? assistant : null;
        }
        const candidates = Studio.artifacts.messageArtifacts(assistant, assistantIndex)
          .filter(item => !item.attachment);
        const artifact = candidates.filter(item => item.previewable).pop() || candidates.pop();
        autoPreviewArtifact(artifact);
      }
    }
  }

  function generate() {
    if (state.loading) return undefined;
    return state.mode === "swarm" ? generateSwarm() : generateReply();
  }

  function send() {
    if (state.streaming || state.loading) return;
    const content = $("chat-input").value.trim();
    const attachments = state.attachments.slice();
    if (!content && !attachments.length) return;
    if (!state.doc) {
      const fallback = Studio.models.chat().find(item => item.available);
      if (!fallback) {
        Studio.toast("No available chat model was found. Add a provider or start a local runtime.", "error");
        return;
      }
      newChat(fallback.id);
    }
    const entry = currentModel();
    if (!entry || !entry.available) {
      Studio.toast(entry ? entry.reason : "Choose an available model first.", "error");
      return;
    }
    const message = { role: "user", content };
    if (attachments.length) message.attachments = attachments;
    state.doc.messages.push(message);
    if (!content && attachments.length && !state.doc.titled) {
      state.doc.title = `Shared ${attachments[0].name}`;
      state.doc.titled = true;
    }
    state.attachments = [];
    $("chat-input").value = "";
    renderAttachments();
    resizeComposer();
    updateComposerState();
    updateBudgetMeter();
    renderMessages();
    refreshPreviewSelection();
    generate();
  }

  function normaliseSettings(settings) {
    if (!settings || typeof settings !== "object" || Array.isArray(settings)) settings = {};
    const clamp = (value, low, high, fallback) => {
      const number = Number(value);
      return Number.isFinite(number) ? Math.max(low, Math.min(high, number)) : fallback;
    };
    const positive = (value, high, fallback) => {
      const number = Number(value);
      return Number.isFinite(number) && number >= 1
        ? Math.min(high, Math.round(number)) : fallback;
    };
    settings.temperature = clamp(settings.temperature, 0, 5, 0.8);
    settings.top_p = clamp(settings.top_p, 0.01, 1, 0.95);
    settings.top_k = Math.round(clamp(settings.top_k, 0, 65536, 0));
    settings.max_tokens = positive(settings.max_tokens, 131072, 512);
    settings.n_ctx = Math.round(clamp(settings.n_ctx, 0, 1_048_576, 0));
    if ("guidance" in settings) settings.guidance = clamp(settings.guidance, 0, 5, 1);
    if ("refine_steps" in settings) settings.refine_steps = positive(settings.refine_steps, 20, 2);
    if ("refine_threshold" in settings) settings.refine_threshold = clamp(settings.refine_threshold, 0, 1, 0.5);
    if ("candidates" in settings) settings.candidates = positive(settings.candidates, 16, 1);
    const stops = Array.isArray(settings.stop) ? settings.stop : [settings.stop];
    settings.stop = stops.filter(value => value !== undefined && value !== null && String(value).trim())
      .map(value => String(value).trim()).slice(0, 32);
    return settings;
  }

  function settingField(label, key, type, options = {}) {
    const value = state.doc.settings[key];
    if (type === "checkbox") return Studio.form.check({ key, label, value: Boolean(value) });
    return Studio.form.field({
      key, label, type,
      value: Array.isArray(value) ? value.join("\n") : value,
      min: options.min, max: options.max, step: options.step,
      options: options.values, help: options.help,
    });
  }

  function openSettings() {
    if (!state.doc || state.loading || $("chat-settings-dialog").open) return;
    const entry = currentModel();
    const body = Studio.ui.render($("chat-settings-body"));
    const system = h("textarea", { rows: 4, value: state.doc.system_prompt || "", "aria-label": "System prompt", placeholder: "Optional instructions for every message" });
    body.append(Studio.ui.field("System prompt", system));
    const storage = state.doc.id
      ? `agent-workspace/chats/${state.doc.id}.json`
      : "Not saved yet; the first message creates the file";
    body.append(Studio.components.sectionHeading({
      title: "Storage",
      description: "Each conversation is one JSON file in agent-workspace/chats, beside agents, runs and settings. It survives a restart, a port change and a browser change.",
    }));
    body.append(h("div", { class: "row wrap" },
      h("span", { class: "muted small chat-storage-path", text: storage }),
      state.doc.id ? Studio.ui.button("Copy path", {
        small: true, onclick: () => Studio.copy(storage, "Storage path copied"),
      }) : null));
    const grid = h("div", { class: "form-grid three" });
    const sampling = !entry || entry.capabilities.sampling !== false;
    if (sampling) {
      grid.append(settingField("Temperature", "temperature", "number", { min: 0, max: 5, step: 0.05 }),
        settingField("Top P", "top_p", "number", { min: 0, max: 1, step: 0.01 }), settingField("Top K (0 = disabled)", "top_k", "number", { min: 0, step: 1 }));
    }
    grid.append(settingField("Maximum output tokens", "max_tokens", "number", { min: 1, step: 1 }), settingField("Device", "device", "select", { values: [["auto", "Auto"], ["cpu", "CPU"], ["cuda", "CUDA"], ["xpu", "XPU"], ["mps", "MPS"]] }));
    body.append(grid);
    const advanced = h("div", { class: "form-grid three" },
      settingField("Prompt format", "prompt_format", "select", { values: [["auto", "Auto"], ["transcript", "User / Assistant transcript"], ["raw", "Raw completion"]] }),
      settingField("Context tokens (0 = model default)", "n_ctx", "number", { min: 0, step: 1 }), settingField("Seed (optional)", "seed", "number", { step: 1 }));
    body.append(advanced);
    const stops = Studio.form.control({ key: "stop", type: "textarea", rows: 2, value: (state.doc.settings.stop || []).join("\n"), ariaLabel: "Stop strings" });
    body.append(Studio.ui.field("Stop strings (one per line)", stops));
    const autoPreview = Studio.form.control({ type: "checkbox", value: state.autoPreview });
    body.append(Studio.ui.field("Open previews automatically", autoPreview,
      "Show code, designs, diagrams and attachments in the preview panel as soon as they appear."));
    const swarmStrategy = Studio.form.control({
      type: "select", options: SWARM_STRATEGIES, value: state.swarmOptions.strategy,
      ariaLabel: "Swarm strategy",
    });
    const swarmAgents = Studio.form.control({
      type: "number", min: 1, max: 64, step: 1, value: state.swarmOptions.max_agents,
      ariaLabel: "Swarm agents",
    });
    const swarmRounds = Studio.form.control({
      type: "number", min: 0, max: 2, step: 1, value: state.swarmOptions.rounds,
      ariaLabel: "Swarm feedback rounds",
    });
    const swarmDepth = Studio.form.control({
      type: "select", options: SWARM_DEPTHS, value: state.swarmOptions.depth,
      ariaLabel: "Swarm research depth",
    });
    const swarmIpc = Studio.form.control({
      type: "checkbox", value: state.swarmOptions.ipc, ariaLabel: "Agents can message each other",
    });
    const swarmMemory = Studio.form.control({
      type: "checkbox", value: state.swarmOptions.memory, ariaLabel: "Agents share a blackboard",
    });
    body.append(Studio.components.sectionHeading({
      title: "Swarm mode",
      description: "Swarm is the default response mode: the leader plans a small team, members work in parallel and share findings, and only the leader's synthesis becomes the reply. Choose Solo in the chat header when one direct answer is enough.",
    }));
    body.append(Studio.ui.callout("info", "A swarm makes several model calls (plan, members, synthesis). Expect more tokens and a longer wait than a solo reply, especially with paid endpoints. Read-only file tools, public web and news retrieval are on by default; file editing is opt-in below, and shell commands appear only when the server owner enables OS access."));
    body.append(h("div", { class: "form-grid" },
      Studio.ui.field("Strategy", swarmStrategy,
        "The leader may still choose a different shape. Pipeline is sequential; Map & reduce is faster for independent research."),
      Studio.ui.field("Agents", swarmAgents, "Up to 6 members, each with its own task.")));
    body.append(h("div", { class: "form-grid" },
      Studio.ui.field("Feedback rounds", swarmRounds, "0 is a single pass; 1–2 add peer review and revision."),
      Studio.ui.field("Research depth", swarmDepth,
        "Longer budgets let slow web research finish. Tokens stay bounded by the rounds per agent.")));
    body.append(h("div", { class: "row wrap" },
      h("label", { class: "check" }, swarmIpc, "Agents can message each other"),
      h("label", { class: "check" }, swarmMemory, "Agents share a blackboard")));
    const toolFiles = Studio.form.control({
      type: "checkbox", value: state.tools.files, ariaLabel: "Read files in the conversation workspace",
    });
    const toolWrite = Studio.form.control({
      type: "checkbox", value: state.tools.write, ariaLabel: "Edit files in the conversation workspace",
    });
    const toolWeb = Studio.form.control({
      type: "checkbox", value: state.tools.web, ariaLabel: "Fetch public web pages and news headlines",
    });
    body.append(Studio.components.sectionHeading({
      title: "Swarm tools",
      description: "Members work in a workspace that belongs to this conversation. Read-only file tools are on by default; editing files and fetching public web pages are opt-in. Tools apply to Swarm mode.",
    }));
    body.append(h("div", { class: "row wrap" },
      h("label", { class: "check" }, toolFiles, "Read files"),
      h("label", { class: "check" }, toolWrite, "Edit files"),
      h("label", { class: "check" }, toolWeb, "Fetch public web pages and news headlines")));
    body.append(Studio.ui.callout("info", "Files live in the conversation workspace. File edits are permanent, web access is limited to public URLs unless OS access is enabled, and shell commands follow the same OS-access setting."));
    const workspace = state.doc.id
      ? `agent-workspace/files/chats/${state.doc.id}`
      : "Saved after the first message";
    body.append(h("div", { class: "row wrap" },
      h("span", { class: "muted small chat-storage-path", text: workspace }),
      state.doc.id ? Studio.ui.button("Copy workspace path", {
        small: true, onclick: () => Studio.copy(workspace, "Workspace path copied"),
      }) : null));
    if (entry && entry.capabilities.effort) body.append(h("div", { class: "form-grid" },
      settingField("Reasoning effort", "effort", "select", { values: [["low", "Low"], ["medium", "Medium"], ["high", "High"], ["max", "Max"]] }),
      settingField("Show summarized reasoning", "show_reasoning", "checkbox")),
      Studio.ui.callout("info", "Claude does not accept temperature, top-p or top-k. Supported models use server-side refusal fallback by default; edit the endpoint to change it."));
    if (entry && entry.capabilities.ncm) body.append(h("div", { class: "form-grid three" },
      settingField("Guidance", "guidance", "number", { min: 0, step: 0.1 }), settingField("Refinement steps", "refine_steps", "number", { min: 1, step: 1 }),
      settingField("Refinement threshold", "refine_threshold", "number", { min: 0, max: 1, step: 0.05 }), settingField("Candidates", "candidates", "number", { min: 1, step: 1 }),
      settingField("Early exit", "early_exit", "checkbox"), settingField("Guidance mode", "guidance_mode", "select", { values: [["none", "None"], ["self", "Self"], ["ar", "AR"]] })));
    $("chat-settings-save").onclick = async event => {
      for (const input of Array.from(body.querySelectorAll("[data-key]"))) {
        const key = input.dataset.key;
        if (input.type === "checkbox") state.doc.settings[key] = input.checked;
        else if (key === "stop") state.doc.settings[key] = input.value.split("\n").map(value => value.trim()).filter(Boolean);
        else if (["temperature", "top_p", "guidance", "refine_threshold"].includes(key)) state.doc.settings[key] = Number(input.value);
        else if (["top_k", "max_tokens", "n_ctx", "refine_steps", "candidates"].includes(key)) state.doc.settings[key] = Number(input.value) || 0;
        else state.doc.settings[key] = input.value;
      }
      state.doc.system_prompt = system.value;
      state.doc.settings = normaliseSettings(state.doc.settings);
      state.autoPreview = autoPreview.checked;
      Studio.store.set("chat-auto-preview", state.autoPreview);
      state.swarmOptions = normaliseSwarmOptions({
        ...state.swarmOptions,
        strategy: swarmStrategy.value,
        max_agents: swarmAgents.value,
        rounds: swarmRounds.value,
        depth: swarmDepth.value,
        ipc: swarmIpc.checked,
        memory: swarmMemory.checked,
      });
      Studio.store.set("chat-swarm", state.swarmOptions);
      state.tools = normaliseTools({
        files: toolFiles.checked || toolWrite.checked,
        write: toolWrite.checked,
        web: toolWeb.checked,
      });
      Studio.store.set("chat-tools", state.tools);
      updateModeControls();
      const saved = await Studio.run(saveChat, event.currentTarget);
      if (!saved) return;
      const all = Studio.store.get("chat-settings", {});
      const settings = all && typeof all === "object" && !Array.isArray(all) ? all : {};
      if (entry) { settings[entry.id] = state.doc.settings; Studio.store.set("chat-settings", settings); }
      $("chat-settings-dialog").close();
      if (state.streaming) updateHeader();
      else renderMessages({ keepScroll: true });
      updateBudgetMeter();
    };
    $("chat-settings-dialog").showModal();
    system.focus();
  }

  function markdownExport() {
    if (!state.doc) return "";
    const lines = [`# ${state.doc.title || "Mariana chat"}`, "", `Model: ${state.doc.model_name || state.doc.model || "unknown"}`, ""];
    for (const message of state.doc.messages) {
      if (message.role === "system" || message.pending) continue;
      lines.push(`## ${message.role === "user" ? "You" : message.model_name || "Assistant"}`, "");
      for (const attachment of message.attachments || []) {
        lines.push(`- Attached ${attachment.kind}: ${attachment.name}${attachment.size ? ` (${format.bytes(attachment.size)})` : ""}`);
      }
      if ((message.attachments || []).length) lines.push("");
      lines.push(message.content || "_No text_", "");
      const swarm = message.swarm;
      if (swarm && Array.isArray(swarm.agents) && swarm.agents.length) {
        lines.push("<details><summary>Agent swarm</summary>", "");
        for (const agent of swarm.agents) {
          const label = agent.label || agent.agent || "Agent";
          const state = agent.status || (agent.ok ? "done" : "queued");
          lines.push(`#### ${label} — ${state}`, "");
          if (agent.task) lines.push(`_${agent.task}_`, "");
          if (agent.output) lines.push(agent.output, "");
          else if (agent.error) lines.push(`Error: ${agent.error}`, "");
        }
        lines.push("</details>", "");
      }
      if (message.reasoning) lines.push(`<details><summary>Reasoning</summary>`, "", message.reasoning, "", "</details>", "");
    }
    return lines.join("\n");
  }

  function exportMarkdown() {
    if (!state.doc) return;
    const name = `${(state.doc.title || "mariana-chat").replace(/[^a-z0-9._-]+/gi, "-")}.md`;
    Studio.download(name, markdownExport(), "text/markdown");
  }

  async function duplicateChat() {
    if (!state.doc) return;
    const copy = {
      title: `${state.doc.title || "New chat"} (copy)`,
      pinned: false,
      model: state.doc.model,
      model_name: state.doc.model_name,
      system_prompt: state.doc.system_prompt,
      settings: state.doc.settings,
      messages: state.doc.messages.map(message => {
        const clone = { ...message };
        delete clone.id;
        return clone;
      }),
    };
    try {
      const data = await api("/api/chats", { body: copy });
      await refreshChats();
      Studio.views.go("chat", { id: data.chat.id });
    } catch (error) {
      Studio.toast(error.message, "error");
    }
  }

  function applyPrompt(text) {
    const input = $("chat-input");
    const existing = input.value;
    if (existing.startsWith(text)) {
      input.focus();
      input.setSelectionRange(input.value.length, input.value.length);
      return;
    }
    input.value = existing.trim() ? `${text}${existing}` : text;
    onComposerInput();
    input.focus();
    input.setSelectionRange(input.value.length, input.value.length);
  }

  function openPrompts(anchor) {
    Studio.overlay.menu(anchor, [
      { heading: "Chat prompts" },
      ...PROMPTS.map(([label, text]) => ({
        label, icon: "sparkle",
        onSelect: () => applyPrompt(text),
      })),
    ], { label: "Prompt presets", align: "end" });
  }

  function moreActions(anchor) {
    const doc = state.doc;
    Studio.overlay.menu(anchor, [
      { label: "Conversations", icon: "chat", onSelect: () => setConversationList(!document.querySelector(".chat-layout").classList.contains("show-list")) },
      { label: "Preview panel", icon: "preview", hint: "Ctrl+Shift+P", onSelect: togglePreview },
      { label: state.autoPreview ? "Turn off automatic previews" : "Open previews automatically",
        icon: "preview", onSelect: () => setAutoPreview(!state.autoPreview) },
      { label: "Context and tokens…", icon: "cpu", onSelect: openTokenDialog },
      "separator",
      { label: "Export as Markdown", icon: "download", disabled: !doc, onSelect: exportMarkdown },
      { label: "Copy as Markdown", icon: "copy", disabled: !doc, onSelect: async () => {
        try {
          await Studio.copy(markdownExport(), "Conversation copied as Markdown");
        } catch (error) { Studio.toast(error.message, "error"); }
      } },
      { label: "Duplicate conversation", icon: "copy", disabled: !doc || !doc.id || state.streaming, onSelect: duplicateChat },
      { label: doc && doc.pinned ? "Unpin conversation" : "Pin conversation", icon: "pin", disabled: !doc || !doc.id, onSelect: async () => { doc.pinned = !doc.pinned; await Studio.run(saveChat); updateHeader(); } },
      { label: "Rename", icon: "edit", disabled: !doc || !doc.id, onSelect: async () => {
        const title = await Studio.prompt({ title: "Rename conversation", label: "Title", value: doc.title });
        if (title) { doc.title = title; doc.titled = true; await Studio.run(saveChat); }
      } },
      { label: "Export JSON", icon: "download", disabled: !doc, onSelect: () => Studio.download(`${(doc.title || "mariana-chat").replace(/[^a-z0-9._-]+/gi, "-")}.json`, JSON.stringify(doc, null, 2)) },
      { label: "Import JSON", icon: "upload", onSelect: () => {
        const input = h("input", { type: "file", accept: ".json,application/json" });
        input.addEventListener("change", async () => {
          if (!input.files[0]) return;
          try {
            const imported = JSON.parse(await input.files[0].text());
            const data = await api("/api/chats", { body: imported });
            await refreshChats();
            Studio.views.go("chat", { id: data.chat.id });
          } catch (error) { Studio.toast(error.message, "error"); }
        });
        input.click();
      } },
      "separator",
      { label: "Delete conversation", icon: "trash", danger: true, disabled: !doc || !doc.id || state.streaming, onSelect: async () => {
        if (!await Studio.confirm({ title: "Delete conversation?", message: "This permanently deletes the saved conversation.", confirmLabel: "Delete", danger: true })) return;
        await Studio.run(async () => {
          await api(`/api/chats/${encodeURIComponent(doc.id)}`, { method: "DELETE" });
          await refreshChats();
          newChat();
        });
      } },
      { label: "Clear all conversations", icon: "trash", danger: true, disabled: !state.chats.length || state.streaming, onSelect: async () => {
        if (!await Studio.confirm({ title: "Clear all conversations?", message: "This permanently deletes every saved conversation. Export anything you need first.", confirmLabel: "Clear all", danger: true })) return;
        await Studio.run(async () => {
          await api("/api/chats", { method: "DELETE" });
          await refreshChats();
          newChat();
        });
      } },
    ], { align: "end" });
  }

  Studio.views.register("chat", {
    title: "Chat", subtitle: "Persistent conversations with any local or remote model",
    glyph: "chat", hint: "Talk to any available model",
    init() {
      /* A hidden tab still receives the stream; paint once it becomes visible. */
      document.addEventListener("visibilitychange", () => {
        if (document.hidden || !liveMessage) return;
        if (liveTimer) { clearTimeout(liveTimer); liveTimer = 0; }
        paintLiveReply(liveMessage);
      });
      for (const [id, name] of Object.entries({
        "chat-attach": "paperclip", "chat-preview-toggle": "preview", "chat-voice": "mic",
        "chat-voice-auto": "volume", "chat-prompts": "sparkle", "chat-preview-copy": "copy",
        "chat-preview-download": "download", "chat-preview-close": "x",
      })) {
        const element = $(id);
        if (element && !element.children.length) element.append(icon(name, id === "chat-prompts" ? "small" : ""));
      }
      const savedLayout = Studio.store.get("chat-preview-layout", null);
      layout = Studio.split.create({
        root: $("chat-workspace"), key: "chat-preview-layout", gap: 6,
        centerMin: 360, centerMinHeight: 240, stackAt: 820,
        fallbackWidth: 1200, fallbackHeight: 700,
        sidebar: { enabled: false },
        agent: {
          splitter: $("chat-split-preview"),
          variable: "--chat-preview-size",
          heightVariable: "--chat-preview-height",
          min: 300, max: 760, defaultSize: 440,
          minHeight: 220, maxHeight: 720, defaultHeight: 340,
          label: "Preview panel",
        },
        onChange: updatePreviewControls,
        onBlocked() {
          Studio.toast("The preview panel needs more room: widen the window or use the stacked layout.", "info");
        },
      });
      if (!savedLayout) layout.setCollapsed("agent", true, false);
      if (state.voiceAuto && Studio.voice && !Studio.voice.synthesisSupported()) {
        state.voiceAuto = false;
        Studio.store.set("chat-voice-auto", false);
      }
      $("chat-new").addEventListener("click", () => newChat());
      $("chat-send").addEventListener("click", send);
      $("chat-mode-solo").addEventListener("click", () => setMode("solo"));
      $("chat-mode-swarm").addEventListener("click", () => setMode("swarm"));
      updateModeControls();
      $("chat-stop").addEventListener("click", stopStream);
      $("chat-input").addEventListener("keydown", event => {
        if (event.isComposing) return;
        if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); send(); }
      });
      $("chat-model").addEventListener("change", async event => {
        if (!state.doc || state.streaming || state.loading) return;
        const entry = Studio.models.get($("chat-model").value);
        if (!entry) return;
        state.doc.model = entry.id; state.doc.model_name = entry.name; state.doc.settings = defaults(entry);
        await Studio.run(saveChat, event.currentTarget);
        if (!state.streaming) renderMessages();
        renderPreview();
      });
      $("chat-settings").addEventListener("click", openSettings);
      $("chat-provider").addEventListener("click", event => Studio.providers.menu(event.currentTarget, {
        onSaved: async endpoints => {
          const endpoint = endpoints[endpoints.length - 1];
          if (state.doc && endpoint) {
            state.doc.model = endpoint.id;
            state.doc.model_name = endpoint.name;
            state.doc.settings = defaults(Studio.models.get(endpoint.id) || { id: endpoint.id });
            await Studio.run(saveChat);
          }
          refreshChatModel(state.doc && state.doc.model);
          updateHeader();
          if (!state.streaming) renderMessages();
          renderPreview();
        },
      }));
      $("chat-settings-close").addEventListener("click", () => $("chat-settings-dialog").close());
      $("chat-more").addEventListener("click", () => moreActions($("chat-more")));
      $("chat-search").addEventListener("input", () => {
        state.search = $("chat-search").value.trim();
        scheduleChatList();
      });
      $("chat-input").addEventListener("input", onComposerInput);
      $("chat-messages").addEventListener("scroll", () => { state.pinned = nearBottom(); updateScrollButton(); }, { passive: true });
      $("chat-scroll").addEventListener("click", () => scrollMessages(true));
      $("chat-conversations").addEventListener("click", () => {
        const layoutElement = document.querySelector(".chat-layout");
        setConversationList(!layoutElement.classList.contains("show-list"));
      });
      $("chat-list-close").addEventListener("click", () => {
        setConversationList(false);
        $("chat-conversations").focus();
      });
      $("chat-view").addEventListener("click", event => {
        const layoutElement = document.querySelector(".chat-layout");
        if (layoutElement.classList.contains("show-list") && !event.target.closest(".chat-sidebar, #chat-more, #chat-conversations")) setConversationList(false);
      });
      $("chat-view").addEventListener("keydown", event => {
        const layoutElement = document.querySelector(".chat-layout");
        if (event.key === "Escape" && layoutElement.classList.contains("show-list")) {
          setConversationList(false);
          $("chat-conversations").focus();
        }
      });
      $("chat-runtime").addEventListener("click", async event => {
        const result = await Studio.run(() => api("/api/models/runtimes/ollama/start", { body: {} }), event.currentTarget);
        if (result) {
          await Studio.models.load(true);
          Studio.ui.fillModels($("chat-model"), { filter: entry => entry.capabilities.chat, value: state.doc.model, includeUnavailable: true });
          if (!state.streaming) renderMessages();
          Studio.toast("Ollama server started.", "success");
        }
      });

      $("chat-preview-toggle").addEventListener("click", togglePreview);
      $("chat-preview-close").addEventListener("click", togglePreview);
      $("chat-preview-list").addEventListener("keydown", event => {
        if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
        const tabs = Array.from($("chat-preview-list").querySelectorAll('[role="tab"]'));
        const index = tabs.indexOf(document.activeElement);
        if (index < 0) return;
        event.preventDefault();
        const next = event.key === "Home" ? 0 : event.key === "End" ? tabs.length - 1
          : (index + (event.key === "ArrowRight" ? 1 : -1) + tabs.length) % tabs.length;
        tabs[next].focus();
        tabs[next].click();
      });
      $("chat-preview-copy").addEventListener("click", () => {
        const artifact = currentArtifact();
        if (!artifact) return;
        const value = artifact.source || artifact.data_url || "";
        if (!value) { Studio.toast("This artifact has no text to copy.", "info"); return; }
        Studio.copy(value, "Artifact copied");
      });
      $("chat-preview-download").addEventListener("click", () => {
        const artifact = currentArtifact();
        if (artifact) Studio.artifacts.download(artifact);
      });
      $("chat-budget-open").addEventListener("click", openTokenDialog);
      $("chat-attach").addEventListener("click", () => $("chat-file-input").click());
      $("chat-file-input").addEventListener("change", () => {
        addFiles($("chat-file-input").files);
        $("chat-file-input").value = "";
      });
      $("chat-prompts").addEventListener("click", event => openPrompts(event.currentTarget));
      $("chat-voice").addEventListener("click", toggleDictation);
      $("chat-voice-auto").addEventListener("click", () => {
        if (!Studio.voice || !Studio.voice.synthesisSupported()) {
          Studio.toast("This browser cannot read replies aloud.", "info");
          return;
        }
        state.voiceAuto = !state.voiceAuto;
        Studio.store.set("chat-voice-auto", state.voiceAuto);
        updateVoiceControls();
        if (!state.voiceAuto) Studio.voice.stopSpeaking();
        else Studio.toast("Replies will be read aloud after each completed response.", "info");
      });

      const compose = document.querySelector(".chat-compose");
      if (compose) {
        let dragDepth = 0;
        const hasFiles = event => Boolean(event.dataTransfer
          && Array.from(event.dataTransfer.types || []).includes("Files"));
        compose.addEventListener("dragenter", event => {
          if (!hasFiles(event)) return;
          event.preventDefault();
          dragDepth += 1;
          compose.classList.add("dragging");
        });
        compose.addEventListener("dragover", event => {
          if (!hasFiles(event)) return;
          event.preventDefault();
          if (event.dataTransfer) event.dataTransfer.dropEffect = "copy";
        });
        compose.addEventListener("dragleave", () => {
          dragDepth = Math.max(0, dragDepth - 1);
          if (!dragDepth) compose.classList.remove("dragging");
        });
        compose.addEventListener("drop", event => {
          if (!hasFiles(event)) return;
          event.preventDefault();
          dragDepth = 0;
          compose.classList.remove("dragging");
          addFiles(event.dataTransfer.files);
        });
      }
      $("chat-input").addEventListener("paste", event => {
        const files = event.clipboardData ? Array.from(event.clipboardData.files || []) : [];
        if (!files.length) return;
        const text = event.clipboardData ? event.clipboardData.getData("text/plain") : "";
        if (!text) event.preventDefault();
        addFiles(files);
      });
      $("chat-view").addEventListener("dragover", event => {
        if (event.dataTransfer && Array.from(event.dataTransfer.types || []).includes("Files")) event.preventDefault();
      });
      $("chat-view").addEventListener("drop", event => {
        if (!event.dataTransfer || !Array.from(event.dataTransfer.types || []).includes("Files")) return;
        event.preventDefault();
        if (!event.target.closest(".chat-compose")) addFiles(event.dataTransfer.files);
      });

      Studio.shortcut("mod+shift+p", () => {
        if (Studio.views.current !== "chat") return false;
        togglePreview();
      });
      if (Studio.voice) {
        Studio.voice.status().then(value => {
          state.voiceStatus = value;
          updateVoiceControls();
        }).catch(() => { /* voice status is optional */ });
      }
      Studio.on("voice:state", payload => {
        if (!payload || !payload.speaking) state.spokenMessage = null;
      });
      Studio.on("models", () => {
        if (!state.doc) return;
        const selected = Studio.ui.fillModels($("chat-model"), {
          filter: entry => entry.capabilities.chat, value: state.doc.model, includeUnavailable: true,
        });
        if (selected && (!state.doc.model || !Studio.models.get(state.doc.model))) {
          const entry = Studio.models.get(selected);
          state.doc.model = selected;
          state.doc.model_name = entry ? entry.name : selected;
          state.doc.settings = defaults(entry || { id: selected });
        }
        updateHeader();
        renderPreview();
      });
      renderAttachments();
      renderPreview();
      updateVoiceControls();
    },
    async activate(parameters) {
      const activation = ++state.activateToken;
      state.loading = true;
      updateComposerState();
      updateModeControls();
      if (!state.doc) {
        Studio.ui.render($("chat-list"), Studio.ui.loading("Loading conversations…"));
        Studio.ui.render($("chat-messages"), Studio.ui.loading("Preparing chat…"));
      }
      try {
        await Studio.models.ensure();
        if (activation !== state.activateToken) return;
        await refreshChats();
        if (activation !== state.activateToken) return;
        if (parameters.id) await loadChat(parameters.id);
        else if (!state.doc
          || (parameters.new === "1" && state.doc.id)
          || (parameters.model && parameters.model !== state.doc.model)) newChat(parameters.model);
        else {
          Studio.ui.fillModels($("chat-model"), { filter: entry => entry.capabilities.chat, value: state.doc.model, includeUnavailable: true });
          if (!state.streaming && state.needsRender) renderMessages();
          updateHeader();
          updateComposerState();
        }
        if (activation !== state.activateToken) return;
        if (layout) layout.apply();
        renderPreview();
      } catch (error) {
        if (activation !== state.activateToken) return;
        if (state.doc) {
          if (!state.streaming && state.needsRender) renderMessages();
          renderPreview();
          if (state.doc.id) {
            const address = `#/chat${Studio.query({ id: state.doc.id })}`;
            if (window.location.hash !== address) history.replaceState(null, "", address);
          }
          Studio.toast(error.message, "error");
          return;
        }
        Studio.ui.render($("chat-list"), Studio.ui.empty({ compact: true, text: "Could not load conversations." }));
        Studio.ui.render($("chat-messages"), Studio.ui.empty({ icon: "alert", title: "Could not open Chat", text: error.message,
          actions: [Studio.ui.button("Try again", { kind: "primary", onclick: () => Studio.views.go("chat", parameters) })] }));
        Studio.toast(error.message, "error");
      } finally {
        if (activation === state.activateToken) {
          state.loading = false;
          updateComposerState();
          updateModeControls();
        }
      }
    },
    deactivate() {
      state.activateToken += 1;
      state.loadToken += 1;
      if (chatListTimer) { clearTimeout(chatListTimer); chatListTimer = 0; }
      setConversationList(false);
      abandonGeneration();
      stopDictation(true);
      if (Studio.voice) Studio.voice.stopSpeaking();
      state.spokenMessage = null;
    },
  });
})();
