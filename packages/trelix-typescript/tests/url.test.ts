import { describe, expect, it } from "vitest";
import { trimTrailingSlashes } from "../src/url.js";

/**
 * Bound for the linear-time check. Only a run of slashes that is NOT at the end defeats the old
 * `/\/+$/` regex (tens of seconds); it handles a trailing run in linear time.
 */
const LINEAR_TIME_BUDGET_MS = 2000;
const HOSTILE_SLASH_RUN = 160_000;

describe("trimTrailingSlashes", () => {
  it.each([
    ["http://h/", "http://h"],
    ["http://h//", "http://h"],
    ["http://h", "http://h"],
    ["http://h/api/", "http://h/api"],
    ["http://h/api//", "http://h/api"],
    ["http://h/api", "http://h/api"],
    ["http://h/a//b/", "http://h/a//b"],
    ["http://h/?q=1", "http://h/?q=1"],
    ["http://h/a?x=/", "http://h/a?x="],
    ["http://h/#/", "http://h/#"],
    ["///", ""],
    ["/", ""],
    ["", ""],
  ])("trims %j to %j", (input, expected) => {
    expect(trimTrailingSlashes(input)).toBe(expected);
  });

  it("leaves a long run of slashes that is not at the end untouched, in linear time", () => {
    const input = "http://h" + "/".repeat(HOSTILE_SLASH_RUN) + "?q";

    const started = performance.now();
    const result = trimTrailingSlashes(input);
    const elapsedMs = performance.now() - started;

    expect(result).toBe(input);
    expect(elapsedMs).toBeLessThan(LINEAR_TIME_BUDGET_MS);
  });

  it("removes a very long trailing run of slashes completely", () => {
    const input = "http://h/api" + "/".repeat(HOSTILE_SLASH_RUN);

    expect(trimTrailingSlashes(input)).toBe("http://h/api");
  });
});
