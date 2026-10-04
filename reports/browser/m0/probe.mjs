import { chromium } from 'playwright';
const args = (process.env.CHROME_ARGS||'').split(' ').filter(Boolean);
const b = await chromium.launch({ headless: true, args, channel: process.env.CHANNEL || undefined });
const p = await b.newPage();
await p.goto("http://127.0.0.1:8941/blank.html");
const r = await p.evaluate(async () => {
  const o = {};
  for (const opt of [{}, {powerPreference:'high-performance'}, {forceFallbackAdapter:true}, {featureLevel:'compatibility'}]) {
    try { const a = await navigator.gpu.requestAdapter(opt);
      o[JSON.stringify(opt)] = a ? {vendor:a.info.vendor, arch:a.info.architecture, device:a.info.device, desc:a.info.description, fallback:a.info.isFallbackAdapter, feats:[...a.features].sort().join(','), subgroups: a.features.has('subgroups'), maxBuf:a.limits.maxBufferSize} : null;
    } catch(e) { o[JSON.stringify(opt)] = 'ERR '+e; } }
  return o; });
console.log(JSON.stringify(r,null,1));
if (process.env.GPUPAGE) { await p.goto('chrome://gpu'); await p.waitForTimeout(3000);
  const t = await p.evaluate(() => { const g=document.querySelector('info-view'); return (g && g.shadowRoot ? g.shadowRoot.textContent : document.body.innerText); });
  console.log(t.replace(/\s*\n\s*/g,'\n').slice(0,6000)); }
await b.close();
