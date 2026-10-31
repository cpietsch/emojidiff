// Drive page.html in headless Chrome (Chrome for Testing via Playwright), one fresh browser
// per config, and record what each decode produced and how long it took.
//
//   node e2e/browser.mjs --model DIR --reference DIR --out DIR [--configs wasm4,webgpu] [--icons N]
//
// Configs: wasm4 (WASM, 4 threads), wasm1 (WASM, 1 thread), webgpu (WebGPU through the JSEP
// build, ort.all.min.mjs), webgpu-native (ort.webgpu.min.mjs). Chrome flags come from
// $CHROME_ARGS (default: milestone 0's WebGPU-on-Vulkan flags); e2e/gpubox-env.sh sets the
// container's library and Vulkan paths on gpubox-4080. nvidia-smi and the load average are
// recorded before and after each config, and nvidia-smi is sampled every 250 ms during it.
// Writes <out>/<config>.json (everything) and <out>/summary.json; exits 1 if a config fails
// or any icon's tokens, SVG or call count differ from the reference.
// Adapted from milestone 0's run_replay.mjs (reports/browser/m0/run_replay.mjs).
import { execFileSync, spawn } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";
import { summarize } from "./check.mjs";
import { packageVersion, requirePackage } from "./modules.mjs";
import { harnessRoutes, startServer } from "./server.mjs";

export const CONFIGS = {
  wasm4: "ep=wasm&threads=4",
  wasm1: "ep=wasm&threads=1",
  webgpu: "ep=webgpu&threads=1",
  "webgpu-native": "ep=webgpu&threads=1&bundle=ort.webgpu.min.mjs",
};
const DEFAULT_CHROME_ARGS = "--enable-unsafe-webgpu --enable-features=Vulkan,VulkanFromANGLE,DefaultANGLEVulkan "
  + "--use-angle=vulkan --use-vulkan=native --ignore-gpu-blocklist --disable-vulkan-surface";

function nvidiaSmi(...query) {
  try {
    return execFileSync("nvidia-smi", query, { encoding: "utf8", stdio: ["ignore", "pipe", "ignore"] }).trim();
  } catch {
    return null;
  }
}
const gpuState = () => ({
  at: new Date().toISOString(),
  gpu: nvidiaSmi("--query-gpu=name,driver_version,utilization.gpu,memory.used,memory.total", "--format=csv,noheader"),
  apps: nvidiaSmi("--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"),
  loadavg: os.loadavg(),
});

function sampleGpu() {
  let text = "", child = null;
  try {
    child = spawn("nvidia-smi", ["--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits", "-lms", "250"]);
    child.on("error", () => {});
    child.stdout.on("data", chunk => { text += chunk; });
  } catch { child = null; }
  return () => {
    if (child) child.kill();
    const rows = text.trim().split("\n").filter(Boolean).map(line => line.split(",").map(Number));
    if (!rows.length) return null;
    return {
      samples: rows.length, util_max: Math.max(...rows.map(r => r[0])),
      util_mean: Math.round(rows.reduce((a, r) => a + r[0], 0) / rows.length * 10) / 10,
      memory_max_mib: Math.max(...rows.map(r => r[1])),
    };
  };
}

export async function runConfig(chromium, base, tag, query, { icons, chromeArgs, timeoutMs = 30 * 60 * 1000 }) {
  const before = gpuState();
  const stopSampling = sampleGpu();
  const browser = await chromium.launch({ headless: true, args: chromeArgs });
  const page = await browser.newPage();
  const consoleLines = [];
  page.on("console", message => consoleLines.push(`${message.type()}: ${message.text()}`));
  page.on("pageerror", error => consoleLines.push(`pageerror: ${error}`));
  const started = Date.now();
  let out;
  try {
    const url = `${base}/web/e2e/page.html?${query}${icons ? `&icons=${icons}` : ""}`;
    await page.goto(url, { timeout: 120000 });
    await page.waitForFunction(() => document.title === "done", null, { timeout: timeoutMs, polling: 500 });
    out = await page.evaluate(() => window.__out);
  } catch (error) {
    out = { error: `driver: ${error}` };
  }
  const browserVersion = browser.version();
  await browser.close();
  out.tag = tag;
  out.query = query;
  out.chrome = browserVersion;
  out.wall_s = (Date.now() - started) / 1000;
  out.nvidia_smi = { before, during: stopSampling(), after: gpuState() };
  out.console = consoleLines.filter(line => !line.startsWith("warning")).slice(0, 40);
  if (!out.error && out.icons) out.summary = summarize(out.icons);
  return out;
}

async function main() {
  const { values: args } = parseArgs({
    options: {
      model: { type: "string" }, reference: { type: "string" }, out: { type: "string" },
      configs: { type: "string", default: "wasm4,webgpu" }, icons: { type: "string" },
    },
  });
  if (!args.model || !args.reference || !args.out) {
    console.error("usage: node e2e/browser.mjs --model DIR --reference DIR --out DIR [--configs wasm4,webgpu] [--icons N]");
    process.exit(2);
  }
  const tags = args.configs.split(",").filter(Boolean);
  for (const tag of tags) if (!(tag in CONFIGS)) throw new Error(`unknown config ${tag}; known: ${Object.keys(CONFIGS)}`);
  const { chromium } = requirePackage("playwright");
  const chromeArgs = (process.env.CHROME_ARGS ?? DEFAULT_CHROME_ARGS).split(" ").filter(Boolean);
  const coi = !process.env.NOCOI;
  const server = await startServer({ routes: harnessRoutes(args.model, args.reference), coi });
  const base = `http://127.0.0.1:${server.port}`;
  fs.mkdirSync(args.out, { recursive: true });
  const results = {};
  let ok = true;
  try {
    for (const tag of tags) {
      const out = await runConfig(chromium, base, tag, CONFIGS[tag], { icons: args.icons, chromeArgs });
      fs.writeFileSync(path.join(args.out, `${tag}.json`), JSON.stringify(out, null, 1) + "\n");
      const s = out.summary;
      const passed = !out.error && s && s.identical_tokens === s.icons && s.identical_svg === s.icons && s.equal_calls === s.icons;
      ok &&= Boolean(passed);
      results[tag] = {
        query: out.query, error: out.error ?? null, adapter: out.adapter ?? null, chrome: out.chrome,
        crossOriginIsolated: out.crossOriginIsolated, threads_effective: out.numThreads_effective,
        session_ms: out.session_ms && Math.round(out.session_ms), cold_icon_ms: out.cold && Math.round(out.cold.ms_icon),
        wall_s: out.wall_s, nvidia_smi_before: out.nvidia_smi.before.gpu, loadavg_before: out.nvidia_smi.before.loadavg,
        gpu_during: out.nvidia_smi.during, summary: s ?? null,
      };
      console.log(tag, JSON.stringify(results[tag]));
      if (out.error) console.log(out.console.join("\n"));
    }
  } finally {
    await server.close();
  }
  const summary = {
    what: "free-running greedy decode in headless Chrome: web/src grammar + loop + codec on ONNX Runtime Web",
    cross_origin_isolation: coi,
    model: args.model, reference: args.reference,
    conditions: {
      node: process.version, playwright: packageVersion("playwright"), onnxruntime_web: packageVersion("onnxruntime-web"),
      chrome_args: chromeArgs, cpu: os.cpus()[0]?.model, logical_cpus: os.cpus().length, host: os.hostname(),
    },
    configs: results,
  };
  fs.writeFileSync(path.join(args.out, "summary.json"), JSON.stringify(summary, null, 1) + "\n");
  process.exit(ok ? 0 : 1);
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) await main();
