import { readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import {
    CHECKOUT_TIMEOUT_MS,
    INDEX_TIMEOUT_MS,
    REVIEW_STAGES_TIMEOUT_MS,
    REVIEW_TIMEOUT_MS,
} from "../src/review-timeouts.js";

const SRC_DIR = fileURLToPath(new URL("../src/", import.meta.url));
const MINUTE_MS = 60 * 1000;

describe("the time limits of a review's stages", () => {
    it("are 2 minutes for the checkout, 5 for the index and 5 for the review child", () => {
        expect(CHECKOUT_TIMEOUT_MS).toBe(2 * MINUTE_MS);
        expect(INDEX_TIMEOUT_MS).toBe(5 * MINUTE_MS);
        expect(REVIEW_TIMEOUT_MS).toBe(5 * MINUTE_MS);
    });

    it("add up to the 12 minutes the queue's deadline is sized from", () => {
        expect(REVIEW_STAGES_TIMEOUT_MS).toBe(12 * MINUTE_MS);
    });

    it.each(["repo-checkout.ts", "review-runner.ts"])(
        "are imported by %s, which declares none of its own",
        (file) => {
            const source = readFileSync(join(SRC_DIR, file), "utf8");

            expect(source).toContain('from "./review-timeouts.js"');
            expect(source).not.toMatch(
                /\bconst (CHECKOUT|INDEX|REVIEW)_TIMEOUT_MS\b/,
            );
        },
    );

    // Importing them is not enough: a stage gets its limit from a default (the
    // checkout from `options.timeoutMs`, the index and the review child from a
    // parameter), and no caller passes one, so the default expression is the
    // limit a review really runs with. A literal there would leave the queue's
    // deadline sized from numbers the stage no longer uses.
    it.each([
        ["repo-checkout.ts", "options.timeoutMs ?? CHECKOUT_TIMEOUT_MS;"],
        ["review-runner.ts", "timeoutMs: number = INDEX_TIMEOUT_MS,"],
        ["review-runner.ts", "timeoutMs: number = REVIEW_TIMEOUT_MS,"],
    ])("%s gets a stage's default limit from `%s`", (file, expression) => {
        const source = readFileSync(join(SRC_DIR, file), "utf8");

        expect(source).toContain(expression);
    });
});
