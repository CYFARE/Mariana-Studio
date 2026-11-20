"use strict";

/* Studio.tokens: the client half of the context meter.  It mirrors the server's
   estimate so typing updates the budget immediately; provider usage replaces the
   estimate once a request has run. */
(function () {
  const Studio = window.Studio;
  const encoder = typeof TextEncoder === "function" ? new TextEncoder() : null;

  function bytes(text) {
    const value = String(text === undefined || text === null ? "" : text);
    if (encoder) return encoder.encode(value).length;
    return unescape(encodeURIComponent(value)).length;
  }

  function estimate(text, unit = "tokens") {
    const value = String(text === undefined || text === null ? "" : text);
    if (!value) return 0;
    if (unit === "bytes") return bytes(value);
    let ascii = 0;
    let wide = 0;
    for (const char of value) {
      if (char.codePointAt(0) < 128) ascii += 1;
      else wide += 1;
    }
    return Math.max(1, Math.ceil(ascii / 4 + wide / 1.6));
  }

  Studio.tokens = { estimate, bytes };
})();
