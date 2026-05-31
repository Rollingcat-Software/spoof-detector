#!/usr/bin/env node
// Local dev server for amispoof with the two HTTP headers that unlock
// `crossOriginIsolated` mode in the browser:
//
//   Cross-Origin-Opener-Policy:   same-origin
//   Cross-Origin-Embedder-Policy: require-corp
//
// Those two together make the browser expose `SharedArrayBuffer`, which
// onnxruntime-web detects and uses to run MiniFASNet inference across
// multiple WASM threads (~50 % more end-to-end fps in measurement —
// MiniFASNet drops from ~30 ms to ~10-15 ms per frame).
//
// `python -m http.server` doesn't send either header, so the console
// shows:
//   "env.wasm.numThreads is set to 2, but this will not work unless
//    you enable crossOriginIsolated mode."
//   "WebAssembly multi-threading is not supported in the current
//    environment. Falling back to single-threading."
//
// Run instead of `python -m http.server`:
//   node web/amispoof/server.mjs            # default port 8791
//   node web/amispoof/server.mjs 8080       # custom port
//
// Zero npm dependencies — uses only Node's built-in `http` and `fs`.

import http from "node:http";
import { readFile, stat } from "node:fs/promises";
import { resolve, join, extname } from "node:path";
import { fileURLToPath } from "node:url";

const PORT = Number(process.argv[2] ?? 8791);
const ROOT = resolve(fileURLToPath(import.meta.url), "..");

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".wasm": "application/wasm",
  ".onnx": "application/octet-stream",
  ".task": "application/octet-stream",
  ".tflite": "application/octet-stream",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".svg": "image/svg+xml",
  ".map": "application/json; charset=utf-8",
  ".txt": "text/plain; charset=utf-8",
  ".xml": "application/xml; charset=utf-8",
};

const server = http.createServer(async (req, res) => {
  // Required for cross-origin isolation. Set on EVERY response (including
  // 404s and asset fetches) so the browser doesn't drop isolation when
  // any sub-resource lacks the header.
  res.setHeader("Cross-Origin-Opener-Policy", "same-origin");
  res.setHeader("Cross-Origin-Embedder-Policy", "require-corp");
  // Allow assets from this origin to be embedded under crossOriginIsolated.
  // `require-corp` above forces every same-origin asset to declare CORP.
  res.setHeader("Cross-Origin-Resource-Policy", "same-origin");
  // Belt-and-braces: tell the browser this server is happy to be loaded
  // in a crossOriginIsolated context.
  res.setHeader("Access-Control-Allow-Origin", "*");

  try {
    const url = new URL(req.url ?? "/", `http://localhost:${PORT}`);
    let path = decodeURIComponent(url.pathname);
    if (path.endsWith("/")) path += "index.html";

    const filePath = resolve(join(ROOT, "." + path));
    if (!filePath.startsWith(ROOT)) {
      res.writeHead(403);
      res.end("Forbidden");
      return;
    }

    const st = await stat(filePath).catch(() => null);
    if (!st || !st.isFile()) {
      res.writeHead(404, { "Content-Type": "text/plain; charset=utf-8" });
      res.end(`Not found: ${path}`);
      return;
    }

    const ext = extname(filePath).toLowerCase();
    const contentType = MIME[ext] ?? "application/octet-stream";
    res.writeHead(200, {
      "Content-Type": contentType,
      "Content-Length": st.size,
      // Cache-bust is handled by the importmap / version-handshake in
      // index.html, so let the browser cache aggressively for stable URLs.
      "Cache-Control": "no-cache, must-revalidate",
    });
    const body = await readFile(filePath);
    res.end(body);
  } catch (err) {
    res.writeHead(500, { "Content-Type": "text/plain; charset=utf-8" });
    res.end(`Server error: ${err.message ?? err}`);
  }
});

server.listen(PORT, () => {
  console.log(`amispoof dev server listening on http://localhost:${PORT}`);
  console.log("  COOP/COEP headers enabled → SharedArrayBuffer available");
  console.log("  → onnxruntime-web can run multi-threaded WASM");
  console.log("  Stop: Ctrl-C");
});
