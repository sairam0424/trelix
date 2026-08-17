#!/usr/bin/env node
// Regenerates src/generated/schema.ts from the FastAPI app's OpenAPI schema.
//
// Default source is an IN-PROCESS dump: `create_app().openapi()` in a one-shot
// Python subprocess (~0.6 s including interpreter start), no server, no port,
// no readiness race. That is what makes the drift gate in
// .github/workflows/schema-drift.yml possible — the previous version of this
// script only knew how to fetch http://127.0.0.1:8765/openapi.json, so
// regenerating required a human to start `trelix serve` in another terminal,
// and the schema went 604 -> 768 lines stale (missing /parse, ParseRequest/
// ParseResponse/ParseSymbolModel, and the /search intent_hint and
// hyde_snippet_hint params) with nothing in CI to notice.
//
// Set TRELIX_OPENAPI_URL to fetch from a running server instead — kept for the
// case where you want the schema of an already-deployed instance rather than of
// the working tree.
//
// src/generated/schema.ts is checked into git (like a lockfile) so `npm run
// build`/`npm test` work without Python installed; the drift job re-runs this
// and diffs.

import { execFileSync } from "node:child_process";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

// Not import.meta.dirname: that landed in Node 20.11 and package.json declares
// engines.node >= 18.
const repoRoot = join(dirname(fileURLToPath(import.meta.url)), "..", "..", "..");

const OUT = "src/generated/schema.ts";

/** Dump the working tree's OpenAPI schema to a temp .json file and return its path.
 *
 * The `.json` extension is load-bearing: openapi-typescript picks its parser
 * from the file extension, and an extensionless temp file is parsed as YAML.
 */
function dumpInProcess() {
  const python = process.env.PYTHON ?? "python3";
  // No sort_keys: the fetch path below gets FastAPI's insertion-ordered JSON,
  // and openapi-typescript emits members in input order — sorting here would
  // make the two paths produce different schema.ts and turn the drift gate into
  // a coin flip on which path last ran.
  const script = [
    "import json, sys",
    "from trelix.api.app import create_app",
    "sys.stdout.write(json.dumps(create_app().openapi()))",
  ].join("\n");
  const spec = execFileSync(python, ["-c", script], {
    encoding: "utf8",
    // Python resolves `trelix` from the installed package; run from the repo
    // root so an editable install of the working tree wins.
    cwd: repoRoot,
    maxBuffer: 32 * 1024 * 1024,
  });
  const file = join(mkdtempSync(join(tmpdir(), "trelix-openapi-")), "openapi.json");
  writeFileSync(file, spec);
  return file;
}

const url = process.env.TRELIX_OPENAPI_URL;
let source;
if (url) {
  console.log(`Fetching OpenAPI schema from ${url} ...`);
  source = url;
} else {
  console.log("Dumping OpenAPI schema in-process via create_app().openapi() ...");
  source = dumpInProcess();
}

execFileSync("npx", ["openapi-typescript", source, "-o", OUT], { stdio: "inherit" });
console.log(`Wrote ${OUT}`);
