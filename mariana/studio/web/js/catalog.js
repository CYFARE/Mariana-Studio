"use strict";

/* Studio.models: the shared model catalog loaded by every workspace. */
(function () {
  const Studio = window.Studio;
  const { api, emit } = Studio;

  Studio.models = {
    list: [],
    endpoints: [],
    loaded: false,
    pending: null,
    sequence: 0,
    load(force) {
      if (this.pending && !force) return this.pending;
      const sequence = ++this.sequence;
      const pending = api(`/api/models${force ? "?refresh=1" : ""}`).then(data => {
        if (sequence === this.sequence) {
          this.list = data.models || [];
          this.endpoints = data.endpoints || [];
          this.loaded = true;
          emit("models", this.list);
        }
        return this.list;
      }).finally(() => {
        if (this.pending === pending) this.pending = null;
      });
      this.pending = pending;
      return pending;
    },
    ensure() {
      return this.loaded ? Promise.resolve(this.list) : this.load();
    },
    get(id) {
      return this.list.find(entry => entry.id === id) || null;
    },
    chat() {
      return this.list.filter(entry => entry.capabilities && entry.capabilities.chat
        && entry.kind !== "mariana-ncm-tokenizer");
    },
  };
})();
