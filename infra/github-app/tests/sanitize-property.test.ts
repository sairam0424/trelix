import fc from "fast-check";
import { describe, expect, it, vi } from "vitest";
import { sanitizeAnnotationPath, sanitizeText } from "../src/sanitize.js";
import { activeMarkupProblem } from "./support/active-markup.js";

// Property tests: fast-check draws NUM_RUNS inputs per property, shrinks a
// failure to a minimal one and prints the seed and path that replay it. The
// example-based tests next to this file pin known attacks; these assert what
// must hold for any input at all. activeMarkupProblem is the independent oracle
// those tests use.
//
// The seed is fixed, so CI draws the same inputs on every run and a failure
// there is a regression, never a flake. To hunt for new counterexamples, change
// SEED locally.

const SEED = 20261113;
const NUM_RUNS = 1000;
const RUN_OPTIONS = { seed: SEED, numRuns: NUM_RUNS };

// Written out, not imported from src/sanitize.ts, so changing the cap there
// fails these tests until a person edits them on purpose.
const MAX_PATH_LENGTH = 1024;

// CPU bound, so a loaded runner slows the wall clock past vitest's 5 s default.
vi.setConfig({ testTimeout: 60_000 });

/** Any UTF-16 code unit: lone surrogates and every hidden BMP character can appear. */
const anyCodeUnit = fc
    .integer({ min: 0, max: 0xffff })
    .map((code) => String.fromCharCode(code));

/** `size: "max"` because the default draws strings of about 5 characters, never more than 10. */
const anyText = fc.string({ unit: anyCodeUnit, maxLength: 3000, size: "max" });

/** Letters and spaces only: nothing a rule of the sanitiser reacts to, so its output is known. */
const plainText = fc.string({
    unit: fc.constantFrom("a", "b", "c", "x", "y", "z", " "),
    maxLength: 300,
    size: "max",
});

/** One fragment per rule of the sanitiser, so random joins of them reach every rule. */
const FRAGMENTS = [
    // HTML comments, in the forms HTML5 accepts, and raw tags
    "<!--",
    "-->",
    "--!>",
    "<!-->",
    "<!--->",
    "<img src=x>",
    "<",
    ">",
    // images, links, reference definitions
    "![a](b)",
    "[x](y)",
    "](",
    "]:",
    // URLs and autolinks, also with a Markdown backslash in them
    "http://",
    "https://example.test",
    "https\\://example.test",
    "www.",
    "www\\.",
    // mentions and emails
    "@octocat",
    "@org/team",
    "a@-b.example",
    "@",
    // character references
    "&#64;",
    "&#x40;",
    "&amp;",
    "&lt;",
    "&copy;",
    "&",
    // hidden characters: zero-width, bidi override, soft hyphen, variation
    // selector, tag character, NUL, and the halves of a surrogate pair
    "\u200b",
    "\u202e",
    "\u00ad",
    "\ufe0f",
    "\u{e0041}",
    "\u0000",
    "\ud83d",
    "\ude00",
    "\u{1f600}",
    // combining marks: two of the longer fragment make a run of six
    "\u0301",
    "\u0301\u0301\u0301",
    // line breaks and whitespace
    "\r\n",
    "\r",
    "\n",
    "\t",
    "\u2028",
    "\u2029",
    " ",
    "a",
];
const markupSoup = fc
    .array(fc.constantFrom(...FRAGMENTS), { maxLength: 400, size: "max" })
    .map((parts) => parts.join(""));

/** Half the characters are two code units long, so a cap often falls inside a surrogate pair. */
const astralText = fc.string({
    unit: fc.constantFrom("a", "\u{1f600}"),
    maxLength: 3000,
    size: "max",
});
const anyInput = fc.oneof(anyText, markupSoup, astralText);

/** Caps from 0 to 4000, the largest one the App uses (a check run's summary). */
const anyLimit = fc.integer({ min: 0, max: 4000 });
const randomCase = fc.tuple(anyInput, anyLimit, fc.boolean());

/**
 * A run of letters with a cap one under, at and one over its length. An
 * off-by-one in the cap shows only there, and two independent random numbers
 * almost never land on it. Plain letters, so what is expected does not depend
 * on the code under test.
 */
const capAtBoundary = fc
    .integer({ min: 1, max: 3000 })
    .chain((length) =>
        fc.tuple(
            fc.constant("a".repeat(length)),
            fc.constantFrom(length - 1, length, length + 1),
            fc.boolean(),
        ),
    );
const sanitizeCase = fc.oneof(randomCase, capAtBoundary);

// File paths: a start (most often none), common pieces (separators and dot
// segments included), at most one piece the sanitiser must reject or clean, and
// more common pieces. One hazard at a time, so that each rule is the only thing
// standing between the path and a failure, and a fair share of the draws are
// accepted and then checked.
const PATH_STARTS = ["", "", "", "", "/", "\\", "C:\\", "c:/", "z:\\"];
const COMMON_PATH_PIECES = [
    "src",
    "a.ts",
    "@scope",
    "[id]",
    "with space",
    "/",
    "/",
    "\\",
    ".",
    "..",
];
const HAZARDOUS_PATH_PIECES = [
    "<",
    ">",
    "://",
    "](",
    "\r",
    "\n",
    "\t",
    "\u2028",
    "\u2029",
    "\u0000",
    "\u00ad",
    "\u200b",
    "\u202e",
    "\ufe0f",
    "\u{e0041}",
    "\ud83d",
];
const commonPieces = fc.array(fc.constantFrom(...COMMON_PATH_PIECES), {
    maxLength: 5,
    size: "max",
});
const pathLike = fc
    .tuple(
        fc.constantFrom(...PATH_STARTS),
        commonPieces,
        fc.constantFrom("", ...HAZARDOUS_PATH_PIECES),
        commonPieces,
    )
    .map(
        ([start, before, hazard, after]) =>
            start + before.join("") + hazard + after.join(""),
    );

// Random text cut short, and a run of one character of any length on both sides
// of the cap. A counterexample longer than the cap would otherwise be a long
// random string, which fast-check takes minutes to shrink; a run shrinks at once.
// A second run draws only lengths around the cap, where an off-by-one shows.
const shortText = fc.string({ unit: anyCodeUnit, maxLength: 200, size: "max" });
const pathOfAnyLength = fc
    .integer({ min: 1, max: 3000 })
    .map((length) => "a".repeat(length));
const pathNearTheCap = fc
    .integer({ min: MAX_PATH_LENGTH - 1, max: MAX_PATH_LENGTH + 2 })
    .map((length) => "a".repeat(length));
const anyPathInput = fc.oneof(
    shortText,
    pathLike,
    pathOfAnyLength,
    pathNearTheCap,
);

/** What an accepted path must not contain: control, line-break, bidi and invisible characters. */
const HIDDEN_CHARACTER =
    /[\u0000-\u001F\u007F-\u009F\u2028\u2029\p{Default_Ignorable_Code_Point}\p{Bidi_Control}\uD800-\uDFFF]/u;

describe("sanitizeText properties", () => {
    it("never returns more than maxLength code units", () => {
        fc.assert(
            fc.property(sanitizeCase, ([input, limit, singleLine]) => {
                const output = sanitizeText(input, limit, { singleLine });
                expect(output.length).toBeLessThanOrEqual(limit);
            }),
            RUN_OPTIONS,
        );
    });

    it("leaves no active markup, whatever the input and the cap", () => {
        fc.assert(
            fc.property(sanitizeCase, ([input, limit, singleLine]) => {
                const output = sanitizeText(input, limit, { singleLine });
                expect(activeMarkupProblem(output)).toBeNull();
            }),
            RUN_OPTIONS,
        );
    });

    it("returns plain text that fits under the cap as it is, without the outer spaces", () => {
        // The other properties are all upper bounds, which a function that
        // returned "" for every input would also satisfy.
        fc.assert(
            fc.property(
                plainText,
                fc.integer({ min: 1, max: 50 }),
                fc.boolean(),
                (input, spare, singleLine) => {
                    const output = sanitizeText(input, input.length + spare, {
                        singleLine,
                    });
                    expect(output).toBe(input.trim());
                },
            ),
            RUN_OPTIONS,
        );
    });

    it("writes a single line when asked to", () => {
        fc.assert(
            fc.property(anyInput, anyLimit, (input, limit) => {
                const output = sanitizeText(input, limit, {
                    singleLine: true,
                });
                expect(output).not.toMatch(/[\r\n\t\u2028\u2029]/);
            }),
            RUN_OPTIONS,
        );
    });
});

describe("sanitizeAnnotationPath properties", () => {
    it("returns null or a short, relative path with no parent segment, markup or hidden character", () => {
        let accepted = 0;
        fc.assert(
            fc.property(anyPathInput, (input) => {
                const path = sanitizeAnnotationPath(input);
                if (path === null) {
                    return;
                }
                accepted += 1;
                expect(path.length).toBeGreaterThan(0);
                expect(path.length).toBeLessThanOrEqual(MAX_PATH_LENGTH);
                expect(path.split(/[\\/]/)).not.toContain("..");
                expect(path).not.toMatch(/^[\\/]|^[A-Za-z]:[\\/]|[<>]/);
                expect(path).not.toMatch(HIDDEN_CHARACTER);
                expect(path).not.toContain("://");
                expect(path).not.toContain("](");
            }),
            RUN_OPTIONS,
        );
        // Guards against a vacuous pass: the checks above only run on accepted paths.
        expect(accepted).toBeGreaterThan(NUM_RUNS / 10);
    });
});
