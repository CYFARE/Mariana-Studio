"use strict";

/* Studio storage: browser preferences and file downloads. */
(function () {
  const Studio = window.Studio;
  const { h } = Studio;

  const store = {
    get(key, fallback) {
      try {
        const raw = window.localStorage.getItem(`mariana-${key}`);
        return raw === null ? fallback : JSON.parse(raw);
      } catch (error) {
        return fallback;
      }
    },
    set(key, value) {
      try {
        window.localStorage.setItem(`mariana-${key}`, JSON.stringify(value));
      } catch (error) {
        /* storage full or blocked: the preference simply is not remembered */
      }
    },
    remove(key) {
      try {
        window.localStorage.removeItem(`mariana-${key}`);
      } catch (error) {
        /* storage unavailable: nothing to remove */
      }
    },
    removePrefix(prefix) {
      try {
        const full = `mariana-${prefix}`;
        const keys = [];
        for (let index = 0; index < window.localStorage.length; index += 1) {
          const key = window.localStorage.key(index);
          if (key && key.startsWith(full)) keys.push(key);
        }
        for (const key of keys) window.localStorage.removeItem(key);
      } catch (error) {
        /* storage unavailable: nothing to clear */
      }
    },
  };

  function download(filename, text, type = "application/json") {
    const url = URL.createObjectURL(new Blob([text], { type }));
    const link = h("a", { href: url, download: filename });
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  Object.assign(Studio, { store, download });
})();
