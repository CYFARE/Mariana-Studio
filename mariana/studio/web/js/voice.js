"use strict";

/* Studio.voice: microphone dictation and reply narration for the IDE.

   A local Whisper job is preferred when the server reports one, because the
   recording stays on this machine.  Otherwise the browser's speech recogniser is
   used when available; that service may be remote, so the UI labels it.  Speech
   synthesis is built into the browser and is local. */
(function () {
  const Studio = window.Studio;
  const MAX_RECORDING_MS = 120_000;
  const MAX_TEXT_CHARS = 20_000;
  const JobDelay = 400;
  let statusValue = null;
  let statusRequest = null;
  let speechGeneration = 0;

  function browserRecognition() {
    return window.SpeechRecognition || window.webkitSpeechRecognition || null;
  }

  function canRecord() {
    return Boolean(window.MediaRecorder && navigator.mediaDevices
      && typeof navigator.mediaDevices.getUserMedia === "function");
  }

  function browserSupported() {
    return Boolean(browserRecognition());
  }

  function synthesisSupported() {
    return Boolean(window.speechSynthesis && typeof window.SpeechSynthesisUtterance === "function");
  }

  function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  async function status(refresh = false) {
    if (refresh || !statusRequest) {
      statusRequest = Studio.api("/api/ide/voice").catch(error => ({
        available: false,
        reason: error.message || "Voice status is unavailable.",
        install: "",
        machine_local: false,
        default_model: "base",
        formats: ["webm", "ogg", "wav"],
        max_bytes: 8_000_000,
      })).then(value => {
        statusValue = value;
        return value;
      });
    }
    return statusRequest;
  }

  function fail(message, name = "VoiceError") {
    const error = new Error(message);
    error.name = name;
    return error;
  }

  function microphoneError(error) {
    if (!error) return fail("The microphone could not be started. Try again.");
    if (error.name === "NotAllowedError" || error.name === "SecurityError") {
      return fail("Microphone permission was denied. Allow the microphone for this page and try again.");
    }
    if (error.name === "NotFoundError" || error.name === "DevicesNotFoundError") {
      return fail("No microphone was found. Connect one and try again.");
    }
    if (error.name === "NotReadableError" || error.name === "TrackStartError") {
      return fail("The microphone is busy in another application. Close it and try again.");
    }
    return error;
  }

  function chooseMode() {
    return status().then(local => {
      if (local.available && canRecord()) return "local";
      if (browserSupported()) return "browser";
      const install = local.install ? ` ${local.install}` : "";
      throw fail(`${local.reason || "No speech recognition is available."}${install}`);
    });
  }

  function setState(options, state) {
    if (options.onState) options.onState(state);
  }

  function reportError(options, error) {
    const normalized = error instanceof Error ? error : fail(String(error || "Voice input failed."));
    if (options.onError) options.onError(normalized);
    setState(options, "idle");
  }

  /* -- browser recognition ------------------------------------------------ */
  function startBrowser(options) {
    const Recognition = browserRecognition();
    if (!Recognition) throw fail("This browser does not provide speech recognition.");
    const recognition = new Recognition();
    recognition.lang = options.language || document.documentElement.lang
      || navigator.language || "en-US";
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.maxAlternatives = 1;
    let finished = false;
    let cancelled = false;
    let heard = false;

    recognition.onresult = event => {
      let interim = "";
      for (let index = event.resultIndex; index < event.results.length; index += 1) {
        const result = event.results[index];
        const text = result && result[0] ? result[0].transcript || "" : "";
        if (text.trim()) heard = true;
        if (result && result.isFinal) {
          if (text.trim() && options.onText) options.onText(text.trim(), true);
        } else interim += text;
      }
      if (interim.trim() && options.onInterim) options.onInterim(interim.trim());
    };
    recognition.onerror = event => {
      if (finished) return;
      finished = true;
      const messages = {
        "not-allowed": "Microphone permission was denied. Allow the microphone for this page and try again.",
        "service-not-allowed": "The browser's speech service is unavailable. Use a local Whisper backend or type the task.",
        "no-speech": "No speech was heard. Try again closer to the microphone.",
        "audio-capture": "No working microphone was found. Connect one and try again.",
        network: "The browser could not reach its speech service. Use a local Whisper backend or type the task.",
      };
      reportError(options, fail(messages[event.error] || `Browser dictation failed (${event.error}).`));
    };
    recognition.onend = () => {
      if (!finished) {
        if (!heard) reportError(options, fail("No speech was heard. Try again closer to the microphone."));
        else setState(options, "idle");
      }
      finished = true;
    };
    try {
      recognition.start();
    } catch (error) {
      throw microphoneError(error);
    }
    setState(options, "listening");
    return {
      mode: "browser",
      stop() { if (!finished) recognition.stop(); },
      cancel() { cancelled = true; finished = true; recognition.abort(); setState(options, "idle"); },
      get cancelled() { return cancelled; },
    };
  }

  /* -- local recording and a server transcription job --------------------- */
  function mimeChoice() {
    const choices = [
      ["audio/webm;codecs=opus", "webm"],
      ["audio/webm", "webm"],
      ["audio/ogg;codecs=opus", "ogg"],
      ["audio/ogg", "ogg"],
      ["audio/mp4", "m4a"],
      ["audio/wav", "wav"],
    ];
    for (const [type, format] of choices) {
      if (!window.MediaRecorder.isTypeSupported || window.MediaRecorder.isTypeSupported(type)) {
        return { type, format };
      }
    }
    return { type: "", format: "webm" };
  }

  function toBase64(buffer) {
    const bytes = new Uint8Array(buffer);
    let binary = "";
    const chunk = 0x8000;
    for (let offset = 0; offset < bytes.length; offset += chunk) {
      binary += String.fromCharCode.apply(null, bytes.subarray(offset, offset + chunk));
    }
    return window.btoa(binary);
  }

  async function waitForJob(jobId, state) {
    const deadline = Date.now() + 240_000;
    while (Date.now() < deadline) {
      if (state && state.cancelled) throw fail("Voice transcription was cancelled.");
      const data = await Studio.api(`/api/jobs/${encodeURIComponent(jobId)}?since=1000000000`);
      const job = data.job || {};
      if (job.status === "completed") return job.result || { text: "" };
      if (job.status === "failed") throw fail(job.error || "The transcriber failed. Read the job log for details.");
      if (job.status === "cancelled") throw fail("Voice transcription was cancelled.");
      await delay(JobDelay);
    }
    throw fail("Voice transcription is taking too long. Open Jobs to cancel it, or try a shorter recording.");
  }

  async function transcribe(blob, format, language, options, state) {
    if (state.cancelled) return null;
    setState(options, "transcribing");
    const audio = toBase64(await blob.arrayBuffer());
    if (state.cancelled) return null;
    const voiceStatus = statusValue || {};
    const payload = await Studio.api("/api/ide/voice/transcribe", {
      body: {
        audio, format, language,
        model: options.model || voiceStatus.default_model || "base",
      },
    });
    state.jobId = payload.job && payload.job.id || "";
    if (state.cancelled) {
      if (state.jobId) {
        Studio.api(`/api/jobs/${encodeURIComponent(state.jobId)}/cancel`, { body: {} }).catch(() => {});
      }
      setState(options, "idle");
      return null;
    }
    const result = await waitForJob(state.jobId, state);
    if (state.cancelled) return null;
    if (!result.text) throw fail("No speech was recognised. Move closer to the microphone and try again.");
    if (options.onText) options.onText(String(result.text).slice(0, MAX_TEXT_CHARS), true);
    setState(options, "idle");
    return result;
  }

  async function startLocal(options) {
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (error) {
      throw microphoneError(error);
    }
    const choice = mimeChoice();
    let recorder;
    try {
      recorder = choice.type ? new window.MediaRecorder(stream, { mimeType: choice.type })
        : new window.MediaRecorder(stream);
    } catch (error) {
      stream.getTracks().forEach(track => track.stop());
      throw fail("This browser cannot record audio for local transcription. Use browser dictation or type the task.");
    }
    const chunks = [];
    const state = { cancelled: false, jobId: "" };
    let stopped = false;
    const timer = setTimeout(() => {
      if (!stopped) handle.stop();
    }, MAX_RECORDING_MS);

    recorder.ondataavailable = event => {
      if (event.data && event.data.size) chunks.push(event.data);
    };
    recorder.onstop = async () => {
      stopped = true;
      clearTimeout(timer);
      stream.getTracks().forEach(track => track.stop());
      if (state.cancelled) return;
      try {
        const blob = new Blob(chunks, { type: choice.type || "audio/webm" });
        await transcribe(blob, choice.format, options.language || "", options, state);
      } catch (error) {
        if (!state.cancelled) reportError(options, error);
      }
    };
    recorder.onerror = event => {
      clearTimeout(timer);
      stream.getTracks().forEach(track => track.stop());
      reportError(options, event.error || fail("The recorder failed."));
    };
    try {
      recorder.start();
    } catch (error) {
      clearTimeout(timer);
      stream.getTracks().forEach(track => track.stop());
      throw microphoneError(error);
    }
    setState(options, "recording");
    const handle = {
      mode: "local",
      stop() {
        if (!stopped && recorder.state !== "inactive") recorder.stop();
      },
      cancel() {
        state.cancelled = true;
        clearTimeout(timer);
        if (state.jobId) {
          Studio.api(`/api/jobs/${encodeURIComponent(state.jobId)}/cancel`, { body: {} }).catch(() => {});
        }
        if (recorder.state !== "inactive") recorder.stop();
        stream.getTracks().forEach(track => track.stop());
        setState(options, "idle");
      },
      get cancelled() { return state.cancelled; },
    };
    return handle;
  }

  /* -- narration ---------------------------------------------------------- */
  function plain(text) {
    return String(text || "")
      .replace(/```[\s\S]*?```/g, " Code block. ")
      .replace(/`([^`]+)`/g, "$1")
      .replace(/!\[[^\]]*\]\([^)]*\)/g, "")
      .replace(/\[([^\]]+)\]\([^)]*\)/g, "$1")
      .replace(/^#{1,6}\s+/gm, "")
      .replace(/[*_>#|]/g, " ")
      .replace(/\s+/g, " ")
      .trim();
  }

  function sentences(text, limit = 1600) {
    const chunks = [];
    let rest = plain(text).slice(0, MAX_TEXT_CHARS);
    while (rest.length > limit) {
      let cut = rest.lastIndexOf(". ", limit);
      if (cut < limit / 3) cut = rest.lastIndexOf(" ", limit);
      if (cut < limit / 3) cut = limit;
      chunks.push(rest.slice(0, cut + 1).trim());
      rest = rest.slice(cut + 1).trim();
    }
    if (rest) chunks.push(rest);
    return chunks;
  }

  function emitSpeaking(speaking, text = "") {
    Studio.emit("voice:state", { speaking, text });
  }

  function stopSpeaking() {
    speechGeneration += 1;
    if (window.speechSynthesis) window.speechSynthesis.cancel();
    emitSpeaking(false);
  }

  function speak(text, options = {}) {
    if (!synthesisSupported()) return false;
    const chunks = sentences(text);
    if (!chunks.length) return false;
    stopSpeaking();
    const generation = speechGeneration;
    const synth = window.speechSynthesis;
    let index = 0;
    const language = options.language || document.documentElement.lang || navigator.language || "en-US";
    emitSpeaking(true, plain(text).slice(0, 240));

    function next() {
      if (generation !== speechGeneration) return;
      if (index >= chunks.length) {
        emitSpeaking(false);
        return;
      }
      const utterance = new window.SpeechSynthesisUtterance(chunks[index]);
      index += 1;
      utterance.lang = language;
      utterance.rate = options.rate || 1;
      utterance.pitch = options.pitch || 1;
      utterance.onend = next;
      utterance.onerror = event => {
        if (generation !== speechGeneration) return;
        if (event.error && event.error !== "interrupted" && event.error !== "canceled") {
          options.onError && options.onError(fail(`Speech could not be read aloud (${event.error}).`));
        }
        emitSpeaking(false);
      };
      synth.speak(utterance);
    }
    next();
    return true;
  }

  function speaking() {
    return Boolean(window.speechSynthesis
      && (window.speechSynthesis.speaking || window.speechSynthesis.pending));
  }

  async function dictate(options = {}) {
    const mode = await chooseMode();
    if (mode === "local") {
      status().then(value => { statusValue = value; });
      return startLocal(options);
    }
    return startBrowser(options);
  }

  Studio.voice = {
    browserSupported, synthesisSupported, status, dictate, speak, stopSpeaking, speaking,
    plain,
  };
})();
