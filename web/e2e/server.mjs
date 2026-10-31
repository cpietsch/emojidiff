// Static server for the browser harness, with cross-origin isolation (COOP/COEP), which
// WASM threads need (SharedArrayBuffer). NOCOI=1 drops the headers, as GitHub Pages would.
//
//   /web/...        web/ (the decoder, this harness's page)
//   /ort/...        onnxruntime-web's dist/ (from web/node_modules or $MOJIDIFF_NODE_MODULES)
//   /model/...      the export directory (encoder.onnx, decoder_step.onnx, manifest.json)
//   /reference/...  the reference bundle (reference.json, images.u8)
//
//   node e2e/server.mjs --model DIR --reference DIR [--port 8951]   (browser.mjs starts its own)
// Adapted from milestone 0's server.mjs (reports/browser/m0/server.mjs).
import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";
import { packageDir, WEB } from "./modules.mjs";

const TYPES = {
  ".html": "text/html; charset=utf-8", ".js": "text/javascript", ".mjs": "text/javascript",
  ".json": "application/json", ".wasm": "application/wasm", ".onnx": "application/octet-stream",
  ".u8": "application/octet-stream",
};

/** The harness's routes for one model and one reference bundle. */
export function harnessRoutes(model, reference) {
  const ort = packageDir("onnxruntime-web");
  if (ort === null) throw new Error("onnxruntime-web is not installed (npm ci --ignore-scripts in web/)");
  return { "/web/": WEB, "/ort/": path.join(ort, "dist"), "/model/": model, "/reference/": reference };
}

/** Serve `routes` ({url prefix: directory}); resolves to {server, port, close()}. */
export function startServer({ routes, port = 0, host = "127.0.0.1", coi = true }) {
  const table = Object.entries(routes).map(([prefix, dir]) => [prefix, path.resolve(dir)]);
  const server = http.createServer((request, response) => {
    const url = decodeURIComponent(new URL(request.url, "http://localhost").pathname);
    const route = table.find(([prefix]) => url.startsWith(prefix));
    const file = route && path.resolve(route[1], "." + url.slice(route[0].length - 1));
    if (!route || !file.startsWith(route[1] + path.sep) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) {
      response.writeHead(404);
      response.end();
      return;
    }
    const headers = { "Content-Type": TYPES[path.extname(file)] || "application/octet-stream", "Cache-Control": "no-store" };
    if (coi) {
      Object.assign(headers, {
        "Cross-Origin-Opener-Policy": "same-origin",
        "Cross-Origin-Embedder-Policy": "require-corp",
        "Cross-Origin-Resource-Policy": "cross-origin",
      });
    }
    response.writeHead(200, headers);
    fs.createReadStream(file).pipe(response);
  });
  return new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(port, host, () => resolve({
      server, port: server.address().port, close: () => new Promise(done => server.close(done)),
    }));
  });
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const { values } = parseArgs({ options: { model: { type: "string" }, reference: { type: "string" }, port: { type: "string", default: "8951" } } });
  if (!values.model || !values.reference) {
    console.error("usage: node e2e/server.mjs --model DIR --reference DIR [--port N]");
    process.exit(2);
  }
  const coi = !process.env.NOCOI;
  const { port } = await startServer({ routes: harnessRoutes(values.model, values.reference), port: Number(values.port), coi });
  console.log(`http://127.0.0.1:${port}/web/e2e/page.html?ep=wasm&threads=4 (${coi ? "COOP/COEP" : "no COOP/COEP"})`);
}
