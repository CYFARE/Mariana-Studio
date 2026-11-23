# Web acceleration and the asset pipeline

Studio works without any accelerator: every fast path has a correctness-equivalent
fallback, and a failed device or module disables only that fast path. This note
covers where the speed comes from, what guarantees the fallbacks provide, and how
to regenerate the WebAssembly kernel.

## Layers

| Layer | Owns | Source |
| --- | --- | --- |
| WebAssembly | Min/max decimation of long chart series | `wasm/chart.c` → `wasm/chart.wasm` |
| WebGPU | Chart grid and polyline rasterisation | `js/accelerate.js` |
| Canvas 2D | Labels, markers, cursor and the universal fallback | `js/training-viz.js` |
| HTTP transport | Keep-alive, gzip, `ETag`/`Last-Modified`, `304` | `server.py` |
| Asset cache | Assembled page and immutable file responses | `assets.py` |

## Guarantees

- `Studio.accelerate` always exposes a callable `decimate()`. When WASM is
  unavailable, fails to fetch, fails validation or throws, the identical
  JavaScript `decimateJs()` algorithm runs instead.
- `drawGpu()` returns a composited `OffscreenCanvas` or `null`. A `null` means
  "draw the same commands with Canvas 2D". A startup 1x1 read-back self-test
  catches drivers that create a device but render nothing.
- A lost device, an `uncapturederror` event or any exception disables WebGPU for
  the rest of the session. It never retries a broken path per frame.
- WebGPU draws into a private `OffscreenCanvas`; nothing is uploaded off-origin.
  The page keeps its strict CSP, which grants only `'self'` and
  `'wasm-unsafe-eval'` for the shipped module.
- View fragments are assembled into `index.html` and are never served as
  standalone files. Asset paths are resolved and re-checked after `resolve()`,
  with POSIX separators, before any bytes are read.

## Chart pipeline

1. `series()` memoises the parsed metric series by metrics identity and stage.
2. Extents are found in one pass over the points.
3. The series is decimated to one minimum/maximum pair per pixel column
   (bounded to 8192 bins), preserving spikes that naive sampling would drop.
4. Pixel-space line and grid commands are built into one flat `Float32Array`.
5. WebGPU renders those commands; the 2D context is cleared and receives the
   GPU canvas plus labels, hover markers and the cursor.
6. Without WebGPU, the same commands become 2D paths.

The GPU vertex buffer grows geometrically and is reused across frames, and the
resolution uniform is rewritten only when the chart size changes.

## Asset caching and compression

- `read_asset_entry()` returns an immutable `AssetEntry` (`body`, `content_type`,
  `etag`, `modified`, `gzip`).
- File entries are cached by `st_mtime_ns` and `st_size`; the assembled page is
  cached by a fingerprint of `index.html` plus every `views/**/*.html`.
- `ETag` is a weak BLAKE2b digest of the bytes; `Last-Modified` is the file mtime.
  Conditional requests return `304`.
- Text assets, JavaScript, JSON, SVG and `application/wasm` bodies of at least
  1 KiB are gzip-compressed once at cache-build time (`mtime=0` so the bytes are
  reproducible). Compressed bytes are only sent when `Accept-Encoding` allows it.
- Responses are HTTP/1.1 with an exact `Content-Length`. Server-sent events and
  error responses close the connection, so a keep-alive socket never waits on an
  unread body.
- `clear_asset_cache()` drops everything; tests use it and a development edit is
  picked up automatically through the mtime checks.

## Rebuilding `chart.wasm`

The kernel is freestanding C with no libc. Regenerate it from
`mariana/studio/web/wasm/`:

```bash
clang --target=wasm32 -O3 -nostdlib -fno-builtin \
  -Wl,--no-entry -Wl,--export=decimate -Wl,--export=steps_in \
  -Wl,--export=values_in -Wl,--export=steps_out -Wl,--export=values_out \
  -Wl,--export-memory -Wl,--initial-memory=12582912 \
  -o chart.wasm chart.c
```

`chart.wasm` is committed: the studio must serve it without a build step. The
initial memory covers the `MAX_POINTS` input and output arrays declared in
`chart.c`; keep the two in sync.

## Verifying

```bash
.venv/bin/python -m pytest -q tests/test_studio_assets.py
node --check mariana/studio/web/js/accelerate.js
```

The asset tests compare the WASM and JavaScript decimators on the same points,
check the vertex layout, and assert gzip, `ETag`, `304`, `application/wasm` and
the CSP grant. A browser check is still the only way to confirm a specific GPU
driver; the self-test makes that check fail safe to Canvas 2D.
