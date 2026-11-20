"use strict";

/* Studio API client: JSON requests, server-sent event streams and live feeds.
   Every view uses these; none of them knows about a particular endpoint. */
(function () {
  const Studio = window.Studio;

  class ApiError extends Error {
    constructor(message, status, code) {
      super(message);
      this.name = "ApiError";
      this.status = status;
      this.code = code || "";
    }
  }

  async function readError(response) {
    let message = `Request failed (${response.status})`;
    let code = "";
    try {
      const data = await response.json();
      if (data && data.error) message = String(data.error);
      if (data && data.code) code = String(data.code);
    } catch (error) {
      /* a non-JSON error body keeps the generic message */
    }
    return new ApiError(message, response.status, code);
  }

  function request(path, options) {
    const method = options.method || (options.body === undefined ? "GET" : "POST");
    const init = { method, headers: { Accept: options.accept || "application/json" }, signal: options.signal };
    if (method !== "GET") {
      // The server only accepts writes that carry a JSON content type.
      init.headers["Content-Type"] = "application/json";
      init.body = JSON.stringify(options.body === undefined ? {} : options.body);
    }
    return fetch(path, init).catch(error => {
      if (error.name === "AbortError") throw error;
      throw new ApiError("Cannot reach Mariana Studio. Check that the server is still running.", 0, "offline");
    });
  }

  /* JSON request.  Rejects with ApiError carrying .status and .code. */
  async function api(path, options = {}) {
    const response = await request(path, options);
    if (!response.ok) throw await readError(response);
    const type = response.headers.get("Content-Type") || "";
    return type.includes("json") ? response.json() : response.text();
  }

  /* POST a body and read the server-sent events of the reply as they arrive. */
  async function stream(path, body, options = {}) {
    const response = await request(path, { body, signal: options.signal, accept: "text/event-stream" });
    if (!response.ok) throw await readError(response);
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let end;
      while ((end = buffer.indexOf("\n\n")) >= 0) {
        const frame = buffer.slice(0, end);
        buffer = buffer.slice(end + 2);
        for (const line of frame.split("\n")) {
          if (!line.startsWith("data:")) continue;
          const payload = line.slice(5).trim();
          if (!payload || payload === "[DONE]") continue;
          let event = null;
          try {
            event = JSON.parse(payload);
          } catch (error) {
            continue;
          }
          if (options.onEvent) options.onEvent(event);
        }
      }
    }
  }

  /* Follow a server-sent event feed.  ``path`` may be a function so a consumer
     can advance a cursor when reconnecting.  The connection is dropped while
     the tab is hidden: browsers allow only a few connections per origin, and an
     idle tab holding them would stall every other Studio tab. */
  function subscribe(path, onMessage, onState) {
    let source = null;
    let closed = false;
    let delay = 1000;
    let timer = null;

    function disconnect() {
      clearTimeout(timer);
      timer = null;
      if (source) source.close();
      source = null;
    }

    function connect() {
      if (closed || source || document.hidden) return;
      source = new EventSource(typeof path === "function" ? path() : path);
      source.onopen = () => {
        delay = 1000;
        if (onState) onState(true);
      };
      source.onmessage = event => {
        let data = null;
        try {
          data = JSON.parse(event.data);
        } catch (error) {
          return;
        }
        try {
          onMessage(data);
        } catch (error) {
          console.error(`Feed handler for ${path} failed`, error);
        }
      };
      source.onerror = () => {
        disconnect();
        if (onState) onState(false);
        if (!closed) timer = setTimeout(connect, delay);
        delay = Math.min(delay * 2, 15000);
      };
    }

    function onVisibility() {
      if (document.hidden) disconnect();
      else connect();
    }

    document.addEventListener("visibilitychange", onVisibility);
    connect();
    return {
      close() {
        closed = true;
        document.removeEventListener("visibilitychange", onVisibility);
        disconnect();
      },
    };
  }

  Object.assign(Studio, { ApiError, api, stream, subscribe });
})();
