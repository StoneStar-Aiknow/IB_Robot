// Mermaid parse() validator for markdown sources (skill tool).
// Authoritative syntax check: parses every ```mermaid fenced block with the
// same parser the browser uses, so parse-pass == render-pass.
//
// Dependencies: mermaid, jsdom (resolved from the current working directory's
// node_modules, searched upward). Install once with:
//   npm install mermaid jsdom
//
// Usage:
//   node parse_check.mjs <target_dir>
//
// Exit code 0 = all blocks parsed; 1 = at least one failure.

import { createRequire } from "node:module";
import path from "node:path";
import process from "node:process";
import { pathToFileURL } from "node:url";

const require = createRequire(path.join(process.cwd(), "noop.js"));

const { JSDOM } = require("jsdom");
const dom = new JSDOM("<!DOCTYPE html><html><body></body></html>", { pretendToBeVisual: true });
globalThis.window = dom.window;
globalThis.document = dom.window.document;
try { Object.defineProperty(globalThis, "navigator", { value: dom.window.navigator, configurable: true }); } catch {}
globalThis.DOMPurify = undefined;

// Windows absolute paths must go through pathToFileURL for dynamic ESM import
const { default: mermaid } = await import(pathToFileURL(require.resolve("mermaid")).href);
mermaid.initialize({ startOnLoad: false, suppressErrorRendering: true });

const root = process.argv[2];
if (!root) {
  console.error("usage: node parse_check.mjs <target_dir>");
  process.exit(2);
}
const fs = await import("node:fs");

function* walk(dir) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, entry.name);
    if (entry.isDirectory()) yield* walk(p);
    else if (entry.name.endsWith(".md")) yield p;
  }
}

const blockRe = /```mermaid\r?\n(.*?)```/gs;
let total = 0, passed = 0;
const failures = [];
for (const file of walk(root)) {
  const text = fs.readFileSync(file, "utf-8");
  const blocks = [...text.matchAll(blockRe)].map(m => m[1]);
  for (let i = 0; i < blocks.length; i++) {
    const block = blocks[i];
    total++;
    try {
      const r = await mermaid.parse(block);
      if (!r || r.error) {
        failures.push({ file, i, err: String(r && r.error || "parse returned false").slice(0, 300), block: block.slice(0, 150) });
      } else {
        passed++;
      }
    } catch (e) {
      const msg = String(e && e.message || e).replace(/\s+/g, " ").slice(0, 300);
      failures.push({ file, i, err: msg, block: block.slice(0, 150) });
    }
  }
}
console.log(`total=${total} passed=${passed} failed=${failures.length}`);
for (const f of failures) {
  console.log(`--- ${f.file} [block#${f.i + 1}]`);
  console.log(`    err: ${f.err.replace(/\n/g, " | ")}`);
  console.log(`    head: ${f.block.replace(/\n/g, " ⏎ ").slice(0, 140)}`);
}
process.exit(failures.length ? 1 : 0);
