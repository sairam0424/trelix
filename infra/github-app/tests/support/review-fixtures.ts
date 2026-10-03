/**
 * What every test that runs a whole review (runReview) shares: the delivery it is about
 * and the commit a fake checkout is at. Literals, so an assertion that names them does
 * not depend on anything under test.
 */
import type { ReviewRequest } from "../../src/review-runner.js";

/** The commit a delivery is about, and the one a fake checkout is at unless a test says otherwise. */
export const HEAD_SHA = "0123456789abcdef0123456789abcdef01234567";

/** A different commit: what a checkout is at after a newer push. */
export const NEWER_SHA = "fedcba9876543210fedcba9876543210fedcba98";

/** A review request as the webhook handler builds it from a pull_request delivery. */
export function reviewRequest(
    overrides: Partial<ReviewRequest> = {},
): ReviewRequest {
    return {
        owner: "o",
        repo: "r",
        prNumber: 1,
        installationId: 999,
        repositoryId: 4242,
        headSha: HEAD_SHA,
        ...overrides,
    };
}
