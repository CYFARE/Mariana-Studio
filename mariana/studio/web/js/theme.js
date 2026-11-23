"use strict";

/* Applied before first paint so the page never flashes the wrong theme.
   System is the default; System, Light and Dark are all remembered. */
(function () {
  var media = window.matchMedia
    ? window.matchMedia("(prefers-color-scheme: dark)")
    : null;
  var mode = "system";
  try {
    var stored = window.localStorage.getItem("mariana-theme");
    try { stored = JSON.parse(stored); } catch (error) { /* legacy raw value */ }
    if (stored === "dark" || stored === "light" || stored === "system") mode = stored;
  } catch (error) {
    /* storage can be blocked; the system theme still applies */
  }
  var theme = mode === "system" ? (media && media.matches ? "dark" : "light") : mode;
  document.documentElement.dataset.theme = theme;
  document.documentElement.dataset.themeMode = mode;
})();
