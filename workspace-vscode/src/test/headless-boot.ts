/**
 * Headless test bootstrap. Registered via `mocha --require`, this installs a
 * `Module._load` shim so that any `require("vscode")` (from the SUTs or the
 * test files) resolves to our in-memory mock instead of the real Extension
 * Host module — letting vscode-dependent units run under plain node/mocha.
 *
 * This is an ADDITIVE second test path; it does not touch the Extension Host
 * runner (runTest.ts / suite/index.ts) used by the real CI `test` script.
 */

/* eslint-disable @typescript-eslint/no-explicit-any, @typescript-eslint/no-var-requires */
import * as path from "path";

const Module = require("module");
const mockPath = path.join(__dirname, "vscode-mock.js");
const originalLoad = Module._load;

Module._load = function (request: string, ...rest: any[]): any {
    if (request === "vscode") {
        return require(mockPath);
    }
    return originalLoad.call(this, request, ...rest);
};
