"use strict";

/* Studio formatting: human-readable bytes, counts, durations and dates. */
(function () {
  const Studio = window.Studio;
  const BYTE_UNITS = ["B", "KB", "MB", "GB", "TB"];

  const format = {
    bytes(value) {
      if (!Number.isFinite(value) || value < 0) return "—";
      let amount = value;
      let unit = 0;
      while (amount >= 1024 && unit < BYTE_UNITS.length - 1) {
        amount /= 1024;
        unit += 1;
      }
      return `${amount.toFixed(unit === 0 || amount >= 100 ? 0 : 1)} ${BYTE_UNITS[unit]}`;
    },
    number(value) {
      return Number.isFinite(value) ? value.toLocaleString("en-US") : "—";
    },
    compact(value) {
      if (!Number.isFinite(value)) return "—";
      const size = Math.abs(value);
      if (size >= 1e9) return `${(value / 1e9).toFixed(size >= 1e10 ? 0 : 1)}B`;
      if (size >= 1e6) return `${(value / 1e6).toFixed(size >= 1e7 ? 0 : 1)}M`;
      if (size >= 1e3) return `${(value / 1e3).toFixed(size >= 1e4 ? 0 : 1)}K`;
      return String(Math.round(value));
    },
    parameters(value) {
      return Number.isFinite(value) && value > 0 ? `${format.compact(value)} parameters` : "";
    },
    percent(value, digits = 0) {
      return Number.isFinite(value) ? `${value.toFixed(digits)}%` : "—";
    },
    duration(seconds) {
      if (!Number.isFinite(seconds) || seconds < 0) return "—";
      const total = Math.floor(seconds);
      const hours = Math.floor(total / 3600);
      const minutes = Math.floor((total % 3600) / 60);
      const rest = String(total % 60).padStart(2, "0");
      return hours ? `${hours}:${String(minutes).padStart(2, "0")}:${rest}` : `${minutes}:${rest}`;
    },
    seconds(value) {
      if (!Number.isFinite(value)) return "—";
      return value < 10 ? `${value.toFixed(1)} s` : `${Math.round(value)} s`;
    },
    /* Accepts seconds since the epoch or an ISO date string. */
    time(value) {
      if (value === null || value === undefined || value === "") return null;
      const time = typeof value === "number" ? value * 1000 : Date.parse(value);
      return Number.isFinite(time) ? time : null;
    },
    ago(value) {
      const time = format.time(value);
      if (time === null) return "—";
      const seconds = Math.round((Date.now() - time) / 1000);
      if (seconds < -30) return format.until(value);
      if (seconds < 45) return "just now";
      if (seconds < 3600) return `${Math.round(seconds / 60)} min ago`;
      if (seconds < 86400) return `${Math.round(seconds / 3600)} h ago`;
      if (seconds < 86400 * 30) return `${Math.round(seconds / 86400)} d ago`;
      return new Date(time).toLocaleDateString();
    },
    until(value) {
      const time = format.time(value);
      if (time === null) return "—";
      const seconds = Math.round((time - Date.now()) / 1000);
      if (seconds <= 0) return "now";
      if (seconds < 90) return `in ${seconds} s`;
      if (seconds < 3600) return `in ${Math.round(seconds / 60)} min`;
      if (seconds < 86400) return `in ${Math.round(seconds / 3600)} h`;
      return `in ${Math.round(seconds / 86400)} d`;
    },
    date(value) {
      const time = format.time(value);
      return time === null ? "—" : new Date(time).toLocaleString();
    },
    title(value) {
      const text = String(value || "").replaceAll("_", " ");
      return text ? text[0].toUpperCase() + text.slice(1) : "";
    },
  };

  Studio.format = format;
})();
