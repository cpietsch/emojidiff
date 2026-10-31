// Render each Lottie JSON in a directory to N evenly spaced frames with lottie-web (SVG renderer)
// in headless Chromium. Usage: node render.mjs <lottie dir> <frames out dir> [frames=8] [px=160]
import { chromium } from "/home/dev/workspace/mojidiff/web/node_modules/playwright/index.mjs";
import fs from "node:fs";
import path from "node:path";

const [dir, out, nArg, pxArg] = process.argv.slice(2);
const N = Number(nArg || 8), PX = Number(pxArg || 160);
const lib = fs.readFileSync("/home/dev/.cache/mojidiff/omnilottie/lottie-web/lottie.min.js", "utf8");
fs.mkdirSync(out, { recursive: true });
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: PX, height: PX } });
await page.setContent(`<html><body style="margin:0;background:#fff"><div id="a" style="width:${PX}px;height:${PX}px"></div></body></html>`);
await page.addScriptTag({ content: lib });
const summary = {};
for (const f of fs.readdirSync(dir).filter((f) => f.endsWith(".json") && !f.includes("candidate")).sort()) {
  const name = path.basename(f, ".json");
  const data = fs.readFileSync(path.join(dir, f), "utf8");
  try {
    const info = await page.evaluate(({ data, N }) => {
      window.anim && window.anim.destroy();
      document.getElementById("a").innerHTML = "";
      const animationData = JSON.parse(data);
      window.anim = lottie.loadAnimation({ container: document.getElementById("a"), renderer: "svg",
        loop: false, autoplay: false, animationData });
      return { frames: window.anim.totalFrames, fr: animationData.fr, w: animationData.w, h: animationData.h,
               layers: (animationData.layers || []).length };
    }, { data, N });
    for (let i = 0; i < N; i++) {
      await page.evaluate(({ i, N }) => window.anim.goToAndStop(Math.floor((i * window.anim.totalFrames) / N), true), { i, N });
      await page.screenshot({ path: path.join(out, `${name}-${i}.png`) });
    }
    summary[name] = { ok: true, ...info };
  } catch (e) {
    summary[name] = { ok: false, error: String(e).slice(0, 300) };
  }
}
fs.writeFileSync(path.join(out, "render-summary.json"), JSON.stringify(summary, null, 1));
await browser.close();
console.log(Object.values(summary).filter((s) => s.ok).length, "of", Object.keys(summary).length, "rendered");
