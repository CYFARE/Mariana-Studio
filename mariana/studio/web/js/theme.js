"use strict";

/* Applied before first paint so the page never flashes the wrong theme.
   Light is the default; the choice is remembered per browser. */
(function () {
  var theme = "light";
  try {
    var stored = window.localStorage.getItem("mariana-theme");
    if (stored === "dark" || stored === "light") theme = stored;
  } catch (error) {
    /* storage can be blocked; the default theme still applies */
  }
  document.documentElement.dataset.theme = theme;
})();
