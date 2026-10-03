/**
 * runReview for tests: the per-review outcome directory goes to a directory of the
 * test's own, never straight into the OS temp directory.
 *
 * runReview creates `trelix-review-outcome-*` (createOutcomeLocation) in the OS temp
 * directory unless it is given an `outcomeBaseDir`. Vitest runs test files in
 * parallel, and other files list that directory (the leak check in
 * repo-checkout.test.ts) or sweep every `trelix-review-*` entry out of it
 * (sweepStaleWorkspaces), so an outcome directory that exists for a moment in the
 * shared directory made them fail now and then. tests/run-review-isolation.test.ts
 * fails if a test file imports runReview from the source instead of from here.
 */
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
    runReview,
    type ReviewFinding,
    type ReviewRequest,
    type RunReviewOptions,
} from "../../src/review-runner.js";
import type { AppConfig } from "../../src/config.js";

/** Not a `trelix-review-` name: sweepStaleWorkspaces must not be able to remove it from under a running test. */
const OUTCOME_BASE_PREFIX = "trelix-outcome-base-";

/**
 * runReview with an `outcomeBaseDir` that exists only for this call. A test that
 * passes its own `outcomeBaseDir` keeps it (and its removal); nothing is created then.
 */
export async function runReviewForTest(
    config: AppConfig,
    request: ReviewRequest,
    options: RunReviewOptions = {},
): Promise<ReviewFinding[]> {
    if (options.outcomeBaseDir !== undefined) {
        return runReview(config, request, options);
    }
    const outcomeBaseDir = mkdtempSync(join(tmpdir(), OUTCOME_BASE_PREFIX));
    try {
        return await runReview(config, request, { ...options, outcomeBaseDir });
    } finally {
        rmSync(outcomeBaseDir, { recursive: true, force: true });
    }
}
