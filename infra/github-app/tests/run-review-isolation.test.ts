/**
 * Tests that run a whole review keep its outcome directory to themselves.
 *
 * runReview creates `trelix-review-outcome-*` in the OS temp directory unless it is
 * given an `outcomeBaseDir`. Vitest runs test files in parallel, so a directory
 * that exists there for a moment made the leak check in repo-checkout.test.ts
 * fail about one run in ten, and sweepStaleWorkspaces, which removes every
 * `trelix-review-*` entry, could remove one from under a running review.
 *
 * tests/support/run-review.ts is the only way the tests call runReview. This file
 * tests that wrapper, and fails if a test file reaches past it.
 */
import {
    existsSync,
    mkdtempSync,
    readdirSync,
    readFileSync,
    rmSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { basename, dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { AppConfig } from "../src/config.js";
import { runReview, type RunReviewOptions } from "../src/review-runner.js";
import { reviewRequest } from "./support/review-fixtures.js";
import { runReviewForTest } from "./support/run-review.js";

vi.mock("../src/review-runner.js", () => ({ runReview: vi.fn() }));

const CONFIG: AppConfig = {
    appId: "1",
    privateKey: "unused",
    webhookSecret: "unused",
    port: 0,
};
const REQUEST = reviewRequest({ installationId: 9 });

describe("runReviewForTest", () => {
    const cleanups: string[] = [];

    beforeEach(() => {
        vi.mocked(runReview).mockReset();
    });

    afterEach(() => {
        for (const dir of cleanups.splice(0)) {
            rmSync(dir, { recursive: true, force: true });
        }
    });

    it("gives the review a base directory that exists during the call and is gone after it", async () => {
        let base = "";
        let existedDuringTheCall = false;
        vi.mocked(runReview).mockImplementation(
            async (_config, _request, options) => {
                base = options?.outcomeBaseDir ?? "";
                existedDuringTheCall = existsSync(base);
                return [];
            },
        );

        await runReviewForTest(CONFIG, REQUEST);

        expect(existedDuringTheCall).toBe(true);
        expect(dirname(base)).toBe(tmpdir());
        expect(basename(base)).toMatch(/^trelix-outcome-base-[A-Za-z0-9]{6}$/);
        expect(existsSync(base)).toBe(false);
    });

    it("names it so that sweepStaleWorkspaces, which removes every trelix-review-* entry, leaves it alone", async () => {
        let name = "";
        vi.mocked(runReview).mockImplementation(
            async (_config, _request, options) => {
                name = basename(options?.outcomeBaseDir ?? "");
                return [];
            },
        );

        await runReviewForTest(CONFIG, REQUEST);

        expect(name).not.toBe("");
        expect(name.startsWith("trelix-review-")).toBe(false);
    });

    it("is a different directory for each call", async () => {
        const bases: string[] = [];
        vi.mocked(runReview).mockImplementation(
            async (_config, _request, options) => {
                bases.push(options?.outcomeBaseDir ?? "");
                return [];
            },
        );

        await runReviewForTest(CONFIG, REQUEST);
        await runReviewForTest(CONFIG, REQUEST);

        expect(bases).toHaveLength(2);
        expect(bases[0]).not.toBe(bases[1]);
    });

    it("removes the directory and rethrows when the review throws", async () => {
        let base = "";
        vi.mocked(runReview).mockImplementation(
            async (_config, _request, options) => {
                base = options?.outcomeBaseDir ?? "";
                throw new Error("review failed (simulated)");
            },
        );

        await expect(runReviewForTest(CONFIG, REQUEST)).rejects.toThrow(
            "review failed (simulated)",
        );

        expect(base).not.toBe("");
        expect(existsSync(base)).toBe(false);
    });

    it("returns what runReview returns and passes the other options through", async () => {
        const findings = [
            {
                file: "a.py",
                lines: "1-1",
                severity: "INFO" as const,
                comment: "c",
            },
        ];
        vi.mocked(runReview).mockResolvedValue(findings);
        const checkoutPullRequest: RunReviewOptions["checkoutPullRequest"] =
            vi.fn();

        const result = await runReviewForTest(CONFIG, REQUEST, {
            checkoutPullRequest,
        });

        expect(result).toBe(findings);
        expect(vi.mocked(runReview).mock.calls[0][0]).toBe(CONFIG);
        expect(vi.mocked(runReview).mock.calls[0][1]).toBe(REQUEST);
        expect(vi.mocked(runReview).mock.calls[0][2]?.checkoutPullRequest).toBe(
            checkoutPullRequest,
        );
    });

    it("keeps a base directory the test chose, and does not remove it", async () => {
        const chosen = mkdtempSync(join(tmpdir(), "trelix-chosen-base-"));
        cleanups.push(chosen);
        vi.mocked(runReview).mockResolvedValue([]);

        await runReviewForTest(CONFIG, REQUEST, { outcomeBaseDir: chosen });

        expect(vi.mocked(runReview).mock.calls[0][2]?.outcomeBaseDir).toBe(
            chosen,
        );
        expect(existsSync(chosen)).toBe(true);
    });
});

const TESTS_DIR = fileURLToPath(new URL(".", import.meta.url));
const THIS_FILE = "run-review-isolation.test.ts";
const RUNNER_MODULE = String.raw`["']\.\.\/src\/review-runner(?:\.js)?["']`;
const NAMED_IMPORT = new RegExp(
    String.raw`import\s+(?:type\s+)?\{([^}]*)\}\s*from\s*${RUNNER_MODULE}`,
    "g",
);
// Ways to get at runReview that the named-import scan cannot see.
const OTHER_IMPORT = new RegExp(
    String.raw`import\s*\*\s*as\s+\w+\s+from\s*${RUNNER_MODULE}|import\s*\(\s*${RUNNER_MODULE}`,
);

/** The names a test file imports from the runner module, `as` aliases resolved to the exported name. */
function importedNames(source: string): string[] {
    return [...source.matchAll(NAMED_IMPORT)].flatMap((match) =>
        match[1]
            .split(",")
            .map((part) =>
                part
                    .trim()
                    .replace(/^type\s+/, "")
                    .split(/\s+as\s+/)[0]
                    .trim(),
            )
            .filter((name) => name !== ""),
    );
}

function reachesPastTheWrapper(source: string): boolean {
    return (
        importedNames(source).includes("runReview") || OTHER_IMPORT.test(source)
    );
}

describe("no test calls runReview except through runReviewForTest", () => {
    const sources = readdirSync(TESTS_DIR)
        .filter((name) => name.endsWith(".test.ts") && name !== THIS_FILE)
        .map((name) => ({
            name,
            text: readFileSync(join(TESTS_DIR, name), "utf8"),
        }));

    it("finds no test file that imports runReview from the source", () => {
        const offenders = sources
            .filter(({ text }) => reachesPastTheWrapper(text))
            .map(({ name }) => name);

        expect(offenders).toEqual([]);
    });

    it("looks at the files that matter: they import from the runner and from the wrapper", () => {
        const fromRunner = sources.filter(
            ({ text }) => importedNames(text).length > 0,
        );
        const fromWrapper = sources.filter(({ text }) =>
            /from\s*["']\.\/support\/run-review\.js["']/.test(text),
        );

        for (const found of [fromRunner, fromWrapper]) {
            expect(found.map(({ name }) => name)).toEqual(
                expect.arrayContaining([
                    "review-runner.test.ts",
                    "check-output-wiring.test.ts",
                ]),
            );
        }
    });

    it.each([
        ['import { runReview } from "../src/review-runner.js";'],
        ['import { a, runReview, b } from "../src/review-runner.js";'],
        ['import {\n    a,\n    runReview,\n} from "../src/review-runner";'],
        ['import { runReview as run } from "../src/review-runner.js";'],
        ['import { type X, runReview } from "../src/review-runner.js";'],
        ['import * as runner from "../src/review-runner.js";'],
        ['const runner = await import("../src/review-runner.js");'],
    ])("the scan flags %j", (text) => {
        expect(reachesPastTheWrapper(text)).toBe(true);
    });

    it.each([
        [
            'import { toAnnotations, runReviewCli } from "../src/review-runner.js";',
        ],
        ['import { runReviewForTest } from "./support/run-review.js";'],
        ['import { runReview } from "../src/other.js";'],
    ])("the scan lets %j through", (text) => {
        expect(reachesPastTheWrapper(text)).toBe(false);
    });
});
