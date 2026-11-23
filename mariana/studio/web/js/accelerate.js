"use strict";

/* Optional browser accelerators.  Nothing here is required for correctness:
   every path has a plain JavaScript fallback, and a failed device or module
   disables only that accelerator.  WASM decimates long chart series; WebGPU
   rasterises the grid and polylines while the 2D context draws labels. */
(function () {
  const Studio = window.Studio;
  const WASM_URL = "/wasm/chart.wasm";
  const MAX_WASM_POINTS = 200000;
  const MAX_BINS = 8192;

  const wasm = { state: "idle", promise: null, exports: null, views: null, buffer: null };
  const gpu = {
    state: "idle", promise: null, device: null, pipeline: null, uniform: null,
    bindGroup: null, format: null, canvas: null, context: null, width: 0, height: 0,
    vertexBuffer: null, vertexCapacity: 0, resolution: null,
  };

  function warn(message) {
    if (window.console && console.warn) console.warn(`[accelerate] ${message}`);
  }

  /* -- WebAssembly: min/max decimation ------------------------------------- */
  function loadWasm() {
    if (wasm.promise) return wasm.promise;
    if (typeof WebAssembly !== "object" || typeof fetch !== "function") {
      wasm.state = "unavailable";
      return Promise.resolve(null);
    }
    wasm.state = "loading";
    wasm.promise = fetch(WASM_URL)
      .then(response => {
        if (!response.ok) throw new Error(`chart.wasm returned ${response.status}`);
        return response.arrayBuffer();
      })
      .then(bytes => WebAssembly.instantiate(bytes, {}))
      .then(({ instance }) => {
        if (typeof instance.exports.decimate !== "function") {
          throw new Error("chart.wasm has no decimate export");
        }
        wasm.exports = instance.exports;
        wasm.state = "ready";
        return wasm.exports;
      })
      .catch(error => {
        wasm.state = "unavailable";
        warn(`WASM decimation unavailable: ${error.message}`);
        return null;
      });
    return wasm.promise;
  }

  function wasmViews() {
    const exports = wasm.exports;
    const memory = exports.memory;
    if (wasm.buffer !== memory.buffer) {
      wasm.buffer = memory.buffer;
      wasm.views = {
        steps: new Float64Array(memory.buffer, exports.steps_in(), MAX_WASM_POINTS),
        values: new Float64Array(memory.buffer, exports.values_in(), MAX_WASM_POINTS),
        outSteps: new Float64Array(memory.buffer, exports.steps_out(), MAX_WASM_POINTS * 2),
        outValues: new Float64Array(memory.buffer, exports.values_out(), MAX_WASM_POINTS * 2),
      };
    }
    return wasm.views;
  }

  /* Pure fallback and testable mirror of the C kernel. */
  function decimateJs(points, minStep, maxStep, bins) {
    const count = points ? points.length : 0;
    if (!count || bins <= 0 || !(maxStep > minStep)) return points || [];
    if (count <= bins * 2) return points;
    const output = [];
    let bin = -1;
    let minIndex = 0;
    let maxIndex = 0;
    let binMin = 0;
    let binMax = 0;
    const emit = () => {
      if (minIndex === maxIndex) {
        output.push(points[minIndex]);
      } else {
        const first = minIndex < maxIndex ? minIndex : maxIndex;
        const second = first === minIndex ? maxIndex : minIndex;
        output.push(points[first], points[second]);
      }
    };
    for (let index = 0; index < count; index += 1) {
      const step = Number(points[index][0]);
      const value = Number(points[index][1]);
      if (!Number.isFinite(step) || !Number.isFinite(value)) continue;
      let next = Math.trunc(((step - minStep) / (maxStep - minStep)) * bins);
      if (next < 0) next = 0;
      if (next >= bins) next = bins - 1;
      if (next !== bin) {
        if (bin >= 0) emit();
        bin = next;
        minIndex = maxIndex = index;
        binMin = binMax = value;
      } else if (value < binMin) {
        binMin = value;
        minIndex = index;
      } else if (value > binMax) {
        binMax = value;
        maxIndex = index;
      }
    }
    if (bin >= 0) emit();
    return output;
  }

  function decimate(points, minStep, maxStep, bins) {
    const limit = Math.max(1, Math.min(MAX_BINS, Math.floor(Number(bins) || 0)));
    const count = points ? points.length : 0;
    if (!count || count <= limit * 2 || !(maxStep > minStep)) return points || [];
    if (wasm.state !== "ready" || count > MAX_WASM_POINTS) {
      return decimateJs(points, minStep, maxStep, limit);
    }
    try {
      const views = wasmViews();
      for (let index = 0; index < count; index += 1) {
        views.steps[index] = points[index][0];
        views.values[index] = points[index][1];
      }
      const produced = wasm.exports.decimate(count, minStep, maxStep, limit);
      if (produced <= 0) return decimateJs(points, minStep, maxStep, limit);
      const output = new Array(produced);
      for (let index = 0; index < produced; index += 1) {
        output[index] = [views.outSteps[index], views.outValues[index]];
      }
      return output;
    } catch (error) {
      wasm.state = "unavailable";
      warn(`WASM decimation failed: ${error.message}`);
      return decimateJs(points, minStep, maxStep, limit);
    }
  }

  /* -- WebGPU: grid and polylines ------------------------------------------ */
  const SHADER = `
struct Uniforms { resolution: vec2f };
@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexOut {
  @builtin(position) position: vec4f,
  @location(0) color: vec4f,
};

@vertex
fn vertex(@location(0) position: vec2f, @location(1) color: vec4f) -> VertexOut {
  var out: VertexOut;
  let clip = vec2f(position.x / uniforms.resolution.x * 2.0 - 1.0,
                   1.0 - position.y / uniforms.resolution.y * 2.0);
  out.position = vec4f(clip, 0.0, 1.0);
  out.color = color;
  return out;
}

@fragment
fn fragment(in: VertexOut) -> @location(0) vec4f {
  return in.color;
}
`;

  function parseColor(value) {
    const text = String(value || "").trim();
    const hex = /^#([0-9a-f]{3}|[0-9a-f]{6})$/i.exec(text);
    if (hex) {
      const raw = hex[1].length === 3
        ? hex[1].split("").map(part => part + part).join("") : hex[1];
      return [
        parseInt(raw.slice(0, 2), 16) / 255,
        parseInt(raw.slice(2, 4), 16) / 255,
        parseInt(raw.slice(4, 6), 16) / 255,
        1,
      ];
    }
    const rgb = /^rgba?\(([^)]+)\)$/i.exec(text);
    if (rgb) {
      const parts = rgb[1].split(/[,\s/]+/).filter(Boolean).map(Number);
      if (parts.length >= 3) {
        return [parts[0] / 255, parts[1] / 255, parts[2] / 255,
          Number.isFinite(parts[3]) ? parts[3] : 1];
      }
    }
    return [0.5, 0.5, 0.5, 1];
  }

  /* One flat Float32Array, filled in place: no per-point objects and no
     per-segment garbage on a path that runs for every animation frame. */
  function buildVertices(commands) {
    const list = commands || [];
    let segments = 0;
    for (const command of list) {
      segments += Math.max(0, (command.points || []).length - 1);
    }
    if (!segments) return new Float32Array(0);
    const values = new Float32Array(segments * 36);
    let offset = 0;
    for (const command of list) {
      const points = command.points || [];
      const color = parseColor(command.color);
      const half = Math.max(0.5, Number(command.width) || 1) / 2;
      const cr = color[0], cg = color[1], cb = color[2], ca = color[3];
      for (let index = 1; index < points.length; index += 1) {
        const previous = points[index - 1];
        const current = points[index];
        let x1 = Number(previous[0]);
        let y1 = Number(previous[1]);
        let x2 = Number(current[0]);
        let y2 = Number(current[1]);
        if (!(Number.isFinite(x1) && Number.isFinite(y1)
              && Number.isFinite(x2) && Number.isFinite(y2))) continue;
        const dx = x2 - x1;
        const dy = y2 - y1;
        const length = Math.hypot(dx, dy);
        if (length < 0.0001) continue;
        const ux = dx / length;
        const uy = dy / length;
        x1 -= ux * half;
        y1 -= uy * half;
        x2 += ux * half;
        y2 += uy * half;
        const nx = -uy * half;
        const ny = ux * half;
        const corners = [x1 + nx, y1 + ny, x2 + nx, y2 + ny,
          x2 - nx, y2 - ny, x1 - nx, y1 - ny];
        for (const corner of [0, 2, 4, 0, 4, 6]) {
          values[offset++] = corners[corner];
          values[offset++] = corners[corner + 1];
          values[offset++] = cr;
          values[offset++] = cg;
          values[offset++] = cb;
          values[offset++] = ca;
        }
      }
    }
    return offset === values.length ? values : values.slice(0, offset);
  }

  function disableGpu(message) {
    if (gpu.state === "unavailable") return;
    gpu.state = "unavailable";
    gpu.canvas = null;
    gpu.context = null;
    if (gpu.vertexBuffer && gpu.vertexBuffer.destroy) gpu.vertexBuffer.destroy();
    gpu.vertexBuffer = null;
    gpu.vertexCapacity = 0;
    gpu.resolution = null;
    if (message) warn(`WebGPU unavailable: ${message}`);
  }

  /* Prove the pipeline can write a pixel before the chart trusts it.  A driver
     that creates a device but renders nothing would otherwise hide the series. */
  async function gpuSelfTest(device, pipeline, uniform, bindGroup, format) {
    const vertices = buildVertices([
      { points: [[-2, -2], [2, 2]], color: "#ffffff", width: 4 },
    ]);
    const vertexBuffer = device.createBuffer({
      size: vertices.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(vertexBuffer, 0, vertices);
    device.queue.writeBuffer(uniform, 0, new Float32Array([1, 1]));
    const texture = device.createTexture({
      size: [1, 1], format,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
    const target = device.createBuffer({
      size: 256, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: texture.createView(), loadOp: "clear", storeOp: "store",
        clearValue: { r: 0, g: 0, b: 0, a: 0 },
      }],
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.setVertexBuffer(0, vertexBuffer);
    pass.draw(vertices.length / 6);
    pass.end();
    encoder.copyTextureToBuffer(
      { texture, origin: { x: 0, y: 0, z: 0 } },
      { buffer: target, bytesPerRow: 256, rowsPerImage: 1 },
      { width: 1, height: 1, depthOrArrayLayers: 1 },
    );
    device.queue.submit([encoder.finish()]);
    await target.mapAsync(GPUMapMode.READ);
    const pixel = new Uint8Array(target.getMappedRange());
    const visible = pixel[3] > 0;
    target.unmap();
    if (!visible) throw new Error("self-test produced a blank frame");
  }

  function initGpu() {
    if (gpu.promise) return gpu.promise;
    if (!navigator.gpu || typeof OffscreenCanvas !== "function") {
      gpu.state = "unavailable";
      return Promise.resolve(false);
    }
    gpu.state = "loading";
    gpu.promise = (async () => {
      try {
        const adapter = await navigator.gpu.requestAdapter({ powerPreference: "low-power" });
        if (!adapter) throw new Error("no adapter");
        const device = await adapter.requestDevice();
        if (device.lost && typeof device.lost.then === "function") {
          device.lost.then(() => disableGpu("device lost"));
        }
        if (typeof device.addEventListener === "function") {
          device.addEventListener("uncapturederror", event => {
            disableGpu(event.error ? event.error.message : "validation error");
          });
        }
        const module = device.createShaderModule({ code: SHADER });
        if (module.getCompilationInfo) {
          const info = await module.getCompilationInfo();
          if (info.messages.some(message => message.type === "error")) {
            throw new Error("shader compilation failed");
          }
        }
        const pipeline = device.createRenderPipeline({
          layout: "auto",
          vertex: {
            module,
            entryPoint: "vertex",
            buffers: [{
              arrayStride: 24,
              attributes: [
                { shaderLocation: 0, offset: 0, format: "float32x2" },
                { shaderLocation: 1, offset: 8, format: "float32x4" },
              ],
            }],
          },
          fragment: {
            module,
            entryPoint: "fragment",
            targets: [{
              format: navigator.gpu.getPreferredCanvasFormat(),
              blend: {
                color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha", operation: "add" },
                alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
              },
            }],
          },
          primitive: { topology: "triangle-list" },
        });
        const uniform = device.createBuffer({
          size: 8,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        gpu.device = device;
        gpu.pipeline = pipeline;
        gpu.uniform = uniform;
        gpu.bindGroup = device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: [{ binding: 0, resource: { buffer: uniform } }],
        });
        gpu.format = navigator.gpu.getPreferredCanvasFormat();
        await gpuSelfTest(device, pipeline, uniform, gpu.bindGroup, gpu.format);
        gpu.state = "ready";
        return true;
      } catch (error) {
        disableGpu(error.message);
        return false;
      }
    })();
    return gpu.promise;
  }

  function ensureCanvas(width, height, ratio) {
    const pixelWidth = Math.max(1, Math.round(width * ratio));
    const pixelHeight = Math.max(1, Math.round(height * ratio));
    if (gpu.canvas && gpu.width === pixelWidth && gpu.height === pixelHeight) return true;
    gpu.canvas = new OffscreenCanvas(pixelWidth, pixelHeight);
    gpu.context = gpu.canvas.getContext("webgpu");
    if (!gpu.context) throw new Error("webgpu context unavailable");
    gpu.context.configure({
      device: gpu.device, format: gpu.format, alphaMode: "premultiplied",
    });
    gpu.width = pixelWidth;
    gpu.height = pixelHeight;
    return true;
  }

  /* Grow-only buffer: a redraw reuses the previous allocation instead of
     creating and destroying one buffer per frame. */
  function vertexBufferFor(device, byteLength) {
    if (gpu.vertexBuffer && gpu.vertexCapacity >= byteLength) return gpu.vertexBuffer;
    if (gpu.vertexBuffer && gpu.vertexBuffer.destroy) gpu.vertexBuffer.destroy();
    const capacity = Math.max(4096, 2 ** Math.ceil(Math.log2(Math.max(1, byteLength))));
    gpu.vertexBuffer = device.createBuffer({
      size: capacity,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    gpu.vertexCapacity = capacity;
    return gpu.vertexBuffer;
  }

  function drawGpu(width, height, commands, ratio = 1) {
    if (gpu.state !== "ready") return null;
    try {
      const vertices = buildVertices(commands);
      if (!vertices.length) return null;
      ensureCanvas(width, height, ratio);
      if (!gpu.resolution || gpu.resolution[0] !== width || gpu.resolution[1] !== height) {
        gpu.device.queue.writeBuffer(gpu.uniform, 0, new Float32Array([width, height]));
        gpu.resolution = [width, height];
      }
      const vertexBuffer = vertexBufferFor(gpu.device, vertices.byteLength);
      gpu.device.queue.writeBuffer(vertexBuffer, 0, vertices);
      const encoder = gpu.device.createCommandEncoder();
      const pass = encoder.beginRenderPass({
        colorAttachments: [{
          view: gpu.context.getCurrentTexture().createView(),
          loadOp: "clear", storeOp: "store",
          clearValue: { r: 0, g: 0, b: 0, a: 0 },
        }],
      });
      pass.setPipeline(gpu.pipeline);
      pass.setBindGroup(0, gpu.bindGroup);
      pass.setVertexBuffer(0, vertexBuffer);
      pass.draw(vertices.length / 6);
      pass.end();
      gpu.device.queue.submit([encoder.finish()]);
      return gpu.canvas;
    } catch (error) {
      disableGpu(error.message);
      return null;
    }
  }

  function capabilities() {
    return { wasm: wasm.state, webgpu: gpu.state };
  }

  function warmup() {
    return Promise.all([loadWasm(), initGpu()]).then(() => capabilities());
  }

  Studio.accelerate = {
    capabilities, warmup, decimate, drawGpu, loadWasm, initGpu,
    gpuReady: () => (gpu.state === "ready" ? Promise.resolve(true) : initGpu()),
    test: { decimateJs, buildVertices, parseColor },
  };
})();
