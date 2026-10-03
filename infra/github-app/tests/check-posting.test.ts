import { describe, expect, it } from "vitest";
import {
    exitCodeOf,
    postCheckRun,
    postIncompleteCheckRun,
    postReviewFailureCheckRun,
    toAnnotations,
} from "../src/check-posting.js";
import {
    postCheckRun as postCheckRunFromRunner,
    postIncompleteCheckRun as postIncompleteCheckRunFromRunner,
    postReviewFailureCheckRun as postReviewFailureCheckRunFromRunner,
    toAnnotations as toAnnotationsFromRunner,
} from "../src/review-runner.js";

/**
 * The Check-posting helpers moved out of review-runner.ts (which was at the
 * repository's 500-line limit) into check-posting.ts. Callers, and most of the
 * tests, still import them from review-runner.ts, so the move must not change
 * what they get.
 */
describe("the move of the Check-posting helpers", () => {
    it.each([
        ["toAnnotations", toAnnotations, toAnnotationsFromRunner],
        ["postCheckRun", postCheckRun, postCheckRunFromRunner],
        [
            "postIncompleteCheckRun",
            postIncompleteCheckRun,
            postIncompleteCheckRunFromRunner,
        ],
        [
            "postReviewFailureCheckRun",
            postReviewFailureCheckRun,
            postReviewFailureCheckRunFromRunner,
        ],
    ])("review-runner re-exports %s unchanged", (_name, moved, reExported) => {
        expect(typeof moved).toBe("function");
        expect(reExported).toBe(moved);
    });

    it("exitCodeOf reads the numeric exit status of a rejected execFile call", () => {
        expect(exitCodeOf({ code: 4 })).toBe(4);
        expect(exitCodeOf({ code: "ENOENT" })).toBeNull();
        expect(exitCodeOf(new Error("no code"))).toBeNull();
        expect(exitCodeOf(null)).toBeNull();
        expect(exitCodeOf("4")).toBeNull();
    });
});
