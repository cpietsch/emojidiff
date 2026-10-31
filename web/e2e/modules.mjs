// Where the end-to-end scripts find their Node packages (onnxruntime-node, onnxruntime-web,
// playwright; pinned in web/package.json): $MOJIDIFF_NODE_MODULES first, then web/node_modules
// (`npm ci --ignore-scripts` in web/). The browser code itself (web/src) needs none of them.
import { existsSync } from "node:fs";
import { createRequire } from "node:module";
import path from "node:path";
import { fileURLToPath } from "node:url";

export const WEB = fileURLToPath(new URL("../", import.meta.url));

export function moduleRoots() {
  const roots = [];
  if (process.env.MOJIDIFF_NODE_MODULES) roots.push(path.resolve(process.env.MOJIDIFF_NODE_MODULES));
  roots.push(path.join(WEB, "node_modules"));
  return roots;
}

/** The directory of an installed package, or null. */
export function packageDir(name) {
  for (const root of moduleRoots()) {
    const dir = path.join(root, name);
    if (existsSync(path.join(dir, "package.json"))) return dir;
  }
  return null;
}

/** require() a CommonJS package from the first root that has it. */
export function requirePackage(name) {
  const dir = packageDir(name);
  if (dir === null) {
    throw new Error(`${name} is not installed: run \`npm ci --ignore-scripts\` in web/ or set MOJIDIFF_NODE_MODULES`);
  }
  return createRequire(path.join(path.dirname(dir), "_"))(name);
}

/** The version field of an installed package. */
export function packageVersion(name) {
  const dir = packageDir(name);
  return dir === null ? null : createRequire(import.meta.url)(path.join(dir, "package.json")).version;
}
