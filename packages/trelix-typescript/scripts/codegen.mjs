#!/usr/bin/env node
// Regenerates src/generated/schema.ts from a running trelix serve instance's
// OpenAPI schema. Run `trelix serve <repo> --port 8765` in another terminal
// first, then `npm run codegen`.
//
// src/generated/schema.ts is checked into git (like a lockfile) so `npm run
// build`/`npm test` work in CI without needing a running server — only rerun
// this script by hand after an app.py route/model change.

import { execFileSync } from "node:child_process";

const endpoint = process.env.TRELIX_OPENAPI_URL ?? "http://127.0.0.1:8765/openapi.json";

console.log(`Fetching OpenAPI schema from ${endpoint} ...`);
execFileSync(
  "npx",
  ["openapi-typescript", endpoint, "-o", "src/generated/schema.ts"],
  { stdio: "inherit" },
);
console.log("Wrote src/generated/schema.ts");
