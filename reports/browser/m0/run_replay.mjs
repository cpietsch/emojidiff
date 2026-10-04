// Drive replay.html in headless Chromium (Chrome for Testing via Playwright) for each config.
// Usage: node run_replay.mjs <tag>=<querystring> ...   (source ../env.sh first; server on PORT, default 8941)
// One fresh browser per config so session creation and the cold icon are really cold.
// nvidia-smi is recorded before and after each config, and sampled every 250 ms during it.
import { chromium } from 'playwright';
import { execSync, spawn } from 'node:child_process';
import fs from 'node:fs'; import path from 'node:path';
const here = path.dirname(new URL(import.meta.url).pathname);
const port = process.env.PORT || 8941;
const args = (process.env.CHROME_ARGS || '').split(' ').filter(Boolean);
const smi = () => execSync('nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits').toString().trim();
const apps = () => execSync('nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader').toString().trim();
for (const spec of process.argv.slice(2)) {
  const [tag, qs] = spec.split('=', 2).length === 2 ? [spec.slice(0, spec.indexOf('=')), spec.slice(spec.indexOf('=') + 1)] : [spec, spec];
  const before = {gpu: smi(), apps: apps(), at: new Date().toISOString()};
  const sampler = spawn('nvidia-smi', ['--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits', '-lms', '250']);
  let samples = ''; sampler.stdout.on('data', d => samples += d);
  const b = await chromium.launch({ headless: true, args });
  const p = await b.newPage(); const logs = []; p.on('console', m => logs.push(m.type() + ': ' + m.text()));
  p.on('pageerror', e => logs.push('pageerror: ' + e));
  const t0 = Date.now();
  let out;
  try {
    await p.goto(`http://127.0.0.1:${port}/m0/${process.env.PAGE || "replay.html"}?${qs}`, {timeout: 120000});
    await p.waitForFunction(() => document.title === 'done', null, {timeout: 3 * 3600 * 1000, polling: 1000});
    out = await p.evaluate(() => window.__out);
  } catch (e) { out = {error: 'driver: ' + e}; }
  await b.close();
  sampler.kill();
  const util = samples.trim().split('\n').filter(Boolean).map(l => l.split(',').map(Number));
  const after = {gpu: smi(), apps: apps(), at: new Date().toISOString()};
  out.tag = tag; out.query = qs; out.wall_s = (Date.now() - t0) / 1000;
  out.nvidia_smi = {before, after, during: {samples: util.length,
    util_max: Math.max(...util.map(u => u[0])), util_mean: util.reduce((a, u) => a + u[0], 0) / Math.max(1, util.length),
    mem_max_mib: Math.max(...util.map(u => u[1]))}};
  out.console = logs.filter(l => !/^warning/.test(l)).slice(0, 30);
  fs.writeFileSync(path.join(here, 'results', `replay_${tag}.json`), JSON.stringify(out, null, 1));
  const s = out.summary || {};
  console.log(tag, JSON.stringify({error: out.error, adapter: out.adapter, threads: out.numThreads_effective, coi: out.crossOriginIsolated,
    session_ms: out.session_ms && Math.round(out.session_ms), ...Object.fromEntries(Object.entries(s).map(([k, v]) => [k, typeof v === 'number' ? Math.round(v * 1000) / 1000 : v])),
    smi_before: before.gpu, smi_after: after.gpu, util_max: out.nvidia_smi.during.util_max}));
  if (out.error) console.log(out.console.join('\n'));
}
