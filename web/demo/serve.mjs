// Serve the demo page (web/demo) with the harness's routes and COOP/COEP headers.
//
//   node demo/serve.mjs --model DIR --reference DIR [--host 127.0.0.1] [--port 8952]
//
// --model is the export (CACHE_ROOT/onnx/<run_id>), --reference a bundle from
// `python -m mojidiff.learning.web_e2e --icons 339`. The 4 WASM threads and WebGPU need a
// secure context, so a plain-http address other than localhost gets 1 thread and no WebGPU.
import path from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";
import { harnessRoutes, startServer } from "../e2e/server.mjs";

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const { values } = parseArgs({
    options: {
      model: { type: "string" }, reference: { type: "string" },
      host: { type: "string", default: "127.0.0.1" }, port: { type: "string", default: "8952" },
    },
  });
  if (!values.model || !values.reference) {
    console.error("usage: node demo/serve.mjs --model DIR --reference DIR [--host H] [--port N]");
    process.exit(2);
  }
  const routes = harnessRoutes(values.model, values.reference);
  const { port } = await startServer({ routes, host: values.host, port: Number(values.port) });
  console.log(`http://${values.host}:${port}/web/demo/index.html`);
}
