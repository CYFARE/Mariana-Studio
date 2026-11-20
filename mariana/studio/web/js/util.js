"use strict";

/* Studio utilities: URL query strings and debouncing. */
(function () {
  const Studio = window.Studio;

  function debounce(task, wait = 250) {
    let timer = null;
    const debounced = (...parameters) => {
      clearTimeout(timer);
      timer = setTimeout(() => task(...parameters), wait);
    };
    debounced.cancel = () => clearTimeout(timer);
    return debounced;
  }

  function query(parameters) {
    const search = new URLSearchParams();
    for (const [key, value] of Object.entries(parameters || {})) {
      if (value !== null && value !== undefined && value !== "") search.set(key, String(value));
    }
    const text = search.toString();
    return text ? `?${text}` : "";
  }

  Object.assign(Studio, { debounce, query });
})();
