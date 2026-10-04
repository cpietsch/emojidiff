// Static server rooted at browser/ (so /node_modules and /m0 are both reachable).
// COOP/COEP on by default (needed for SharedArrayBuffer -> WASM threads); NOCOI=1 turns them off
// to mimic GitHub Pages, which cannot send these headers.
import http from 'node:http'; import fs from 'node:fs'; import path from 'node:path';
const root = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const types = {'.html':'text/html','.js':'text/javascript','.mjs':'text/javascript','.wasm':'application/wasm','.onnx':'application/octet-stream','.json':'application/json','.bin':'application/octet-stream'};
const coi = !process.env.NOCOI;
http.createServer((req,res)=>{
  const p = path.join(root, decodeURIComponent(new URL(req.url,'http://x').pathname));
  if(!p.startsWith(root)||!fs.existsSync(p)||fs.statSync(p).isDirectory()){res.writeHead(404);return res.end();}
  const h = {'Content-Type':types[path.extname(p)]||'application/octet-stream','Cache-Control':'no-store'};
  if (coi) Object.assign(h, {'Cross-Origin-Opener-Policy':'same-origin','Cross-Origin-Embedder-Policy':'require-corp','Cross-Origin-Resource-Policy':'cross-origin'});
  res.writeHead(200,h); fs.createReadStream(p).pipe(res);
}).listen(+process.env.PORT||8941,'127.0.0.1',()=>console.log('listening', coi ? 'coi' : 'no-coi'));
