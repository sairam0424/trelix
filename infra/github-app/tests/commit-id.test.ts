import { describe, expect, it } from "vitest";
import { isCommitId } from "../src/commit-id.js";

const SHA1 = "0123456789abcdef0123456789abcdef01234567";
const SHA256 =
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

describe("isCommitId", () => {
    it("accepts a full lowercase SHA-1 and a full lowercase SHA-256 id", () => {
        expect(isCommitId(SHA1)).toBe(true);
        expect(isCommitId(SHA256)).toBe(true);
    });

    it.each([
        ["an empty string", ""],
        ["a short id", "deadbeef"],
        ["39 digits", SHA1.slice(1)],
        ["41 digits", `${SHA1}0`],
        ["63 digits", SHA256.slice(1)],
        ["65 digits", `${SHA256}0`],
        ["an uppercase id", SHA1.toUpperCase()],
        ["one non-hex character", `${SHA1.slice(0, 39)}g`],
        ["a trailing newline", `${SHA1}\n`],
        ["a leading space", ` ${SHA1.slice(1)}`],
        ["a ref name", "HEAD"],
        ["two ids", `${SHA1}\n${SHA1}`],
    ])("rejects %s", (_name, value) => {
        expect(isCommitId(value)).toBe(false);
    });

    it.each([
        ["undefined", undefined],
        ["null", null],
        ["a number", 1234],
        ["an object", {}],
        ["an array holding an id", [SHA1]],
    ])("rejects %s", (_name, value) => {
        expect(isCommitId(value)).toBe(false);
    });
});

// The id is read from a webhook payload and from git's output. A hostile value must cost
// the same as a short one: the length is checked before the pattern sees it.
describe("isCommitId on hostile input", () => {
    const SIZE = 50_000;
    const HOSTILE: Array<[string, string]> = [
        ["backticks", "`".repeat(SIZE)],
        ["tildes", "~".repeat(SIZE)],
        ["spaces", " ".repeat(SIZE)],
        ["newlines", "\n".repeat(SIZE)],
        ["hex digits", "a".repeat(SIZE)],
        ["an id followed by hex", SHA1 + "a".repeat(SIZE)],
        ["hex followed by an id", "a".repeat(SIZE) + SHA1],
        // The shape that makes a nested-quantifier pattern backtrack: a long hex run that
        // fails on its last character.
        ["hex followed by one non-hex character", "a".repeat(SIZE) + "!"],
        ["alternating hex and non-hex", "a!".repeat(SIZE / 2)],
        ["nested brackets", "[".repeat(SIZE / 2) + "]".repeat(SIZE / 2)],
        ["unterminated comment openers", "<!--".repeat(SIZE / 4)],
        ["combining marks", "a" + "\u0301".repeat(SIZE)],
        ["zero-width characters", "\u200B".repeat(SIZE)],
    ];

    it.each(HOSTILE)(
        "rejects a run of %s in under two seconds",
        (_name, value) => {
            expect(value.length).toBeGreaterThanOrEqual(SIZE);
            const started = performance.now();

            const result = isCommitId(value);

            expect(performance.now() - started).toBeLessThan(2000);
            expect(result).toBe(false);
        },
    );
});
