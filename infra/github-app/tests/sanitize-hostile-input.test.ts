import { describe, expect, it, vi } from "vitest";
import type { CheckAnnotation } from "../src/review-runner.js";
import {
    sanitizeAnnotation,
    sanitizeCheckOutput,
    sanitizeText,
} from "../src/sanitize.js";
import { activeMarkupProblem } from "./support/active-markup.js";

/** The size rule 6 of this project names: a hostile input of 50,000 characters. */
const HOSTILE_SIZE = 50_000;
const TIME_LIMIT_MS = 2000;

// Sized for a busy machine: the work below is CPU bound and measured in CPU
// time, so a loaded runner slows the wall clock but not the measurement.
vi.setConfig({ testTimeout: 120_000 });

/** A hostile text of (at least) `size` characters. */
type Shape = (size: number) => string;

const repeated =
    (unit: string): Shape =>
    (size) =>
        unit.repeat(Math.ceil(size / unit.length)).slice(0, size);

// Inputs shaped to make a parser or a backtracking regular expression slow:
// long runs of backticks, tildes, spaces, newlines, nested brackets,
// unterminated comment openers, and zero-width or combining characters. The
// sanitiser has no parser and bounded repetition only, so each must finish far
// below the limit.
const HOSTILE_SHAPES: Array<[name: string, shape: Shape]> = [
    ["backticks", repeated("`")],
    ["tildes", repeated("~")],
    ["spaces", repeated(" ")],
    ["newlines", repeated("\n")],
    ["carriage returns and line feeds", repeated("\r\n")],
    ["open brackets", repeated("[")],
    ["close brackets", repeated("]")],
    ["nested brackets", (size) => "[".repeat(size / 2) + "]".repeat(size / 2)],
    ["image openers", repeated("![")],
    ["image openers with a destination opener", repeated("![](")],
    ["complete short images", repeated("![a](b)")],
    ["link openers", repeated("[a](")],
    ["reference definitions", repeated("[a]:\n")],
    ["open parentheses", (size) => "![a](" + "(".repeat(size)],
    ["close parentheses", repeated(")")],
    ["unterminated comment openers", repeated("<!--")],
    ["one unterminated opener and dashes", (size) => "<!--" + "-".repeat(size)],
    ["spaced unterminated openers", repeated("<!-- ")],
    ["empty comments", repeated("<!---->")],
    // Ordinary text kept between many complete comments. The shapes around
    // this one consume everything they repeat, so a stripper that copies the
    // text after every comment it removes (quadratic, same output) passes
    // them; it does not pass these.
    ["complete comments between text", repeated("<!--a-->b")],
    ["text before complete spaced comments", repeated("a<!-- -->")],
    ["comment closers", repeated("-->")],
    ["bang comment closers", repeated("--!>")],
    ["abrupt comments", repeated("<!-->")],
    ["angle brackets", repeated("<>")],
    ["ampersands", repeated("&")],
    ["character reference openers", repeated("&#")],
    ["character-reference-shaped text", repeated("&a;")],
    ["written-out references", repeated("&lt;")],
    ["zero-width spaces", repeated("\u200B")],
    ["zero-width joiners", repeated("\u200D")],
    ["bidi overrides", repeated("\u202E")],
    ["tag characters", repeated("\u{E0041}")],
    ["variation selectors", repeated("\uFE0F")],
    [
        "one base character and a run of combining marks",
        (size) => "a" + "\u0301".repeat(size),
    ],
    [
        "runs of exactly four combining marks",
        repeated("a\u0301\u0301\u0301\u0301"),
    ],
    [
        "runs of five combining marks",
        repeated("a\u0301\u0301\u0301\u0301\u0301"),
    ],
    ["lone surrogates", repeated("\uD800")],
    ["emoji", repeated("\u{1F600}")],
    ["mentions", repeated("@a")],
    ["at signs", repeated("@")],
    ["email addresses with an escaped domain start", repeated("a@\\.b")],
    ["www prefixes", repeated("www.")],
    ["escaped www prefixes", repeated("www\\.")],
    ["scheme separators", repeated("://")],
    ["escaped scheme separators", repeated("\\:\\/\\/")],
    ["backslashes", repeated("\\")],
    ["colons and slashes", repeated(":/")],
    ["URLs", repeated("https://x.y/")],
    ["control characters", repeated("\u0000")],
    [
        "everything at once",
        repeated("<!--![a](http://x) @u www.a &#64; \u200B\u0301-->"),
    ],
];

const okAnnotation = (
    overrides: Partial<CheckAnnotation>,
): CheckAnnotation => ({
    path: "a.py",
    start_line: 1,
    end_line: 1,
    annotation_level: "notice",
    message: "m",
    title: "t",
    ...overrides,
});

/**
 * CPU time of `work` in milliseconds. Wall-clock time would make these tests
 * fail on a busy machine that merely runs the sanitiser late; CPU time only
 * grows when the sanitiser does more work.
 */
function cpuMs(work: () => unknown): number {
    const before = process.cpuUsage();
    work();
    const used = process.cpuUsage(before);
    return (used.user + used.system) / 1000;
}

describe("linear time on hostile 50,000-character inputs", () => {
    it.each(HOSTILE_SHAPES)("%s", (_name, shape) => {
        const text = shape(HOSTILE_SIZE);
        expect(text.length).toBeGreaterThanOrEqual(HOSTILE_SIZE);

        let message = "";
        let title = "";
        const took = cpuMs(() => {
            message = sanitizeText(text, 2000);
            title = sanitizeText(text, 140, { singleLine: true });
        });

        expect(took).toBeLessThan(TIME_LIMIT_MS);
        expect(message.length).toBeLessThanOrEqual(2000);
        expect(title.length).toBeLessThanOrEqual(140);
    });

    it("sanitises a hostile annotation and a hostile output in time", () => {
        const text = HOSTILE_SHAPES[HOSTILE_SHAPES.length - 1][1](HOSTILE_SIZE);
        const took = cpuMs(() => {
            sanitizeAnnotation(okAnnotation({ path: text, message: text }));
            sanitizeAnnotation(okAnnotation({ message: text, title: text }));
            sanitizeCheckOutput({ title: text, summary: text });
        });
        expect(took).toBeLessThan(TIME_LIMIT_MS);
    });
});

// A quadratic rule can still finish 50,000 characters inside the limit on a
// fast machine (an unbounded image pattern takes about a second), so the time
// is also compared across sizes: eight times the input must cost about eight
// times the time, not sixty-four.
describe("time grows with the input, not with its square", () => {
    const SMALL = 25_000;
    const LARGE = 200_000;
    const MAX_GROWTH = 30;
    const NOISE_FLOOR_MS = 2;

    const ROUNDS = 3;

    /** The best of several rounds for each size, taken alternately so both see the same conditions. */
    function bestOfMs(small: string, large: string): [number, number] {
        let bestSmall = Infinity;
        let bestLarge = Infinity;
        for (let round = 0; round < ROUNDS; round += 1) {
            bestSmall = Math.min(
                bestSmall,
                cpuMs(() => sanitizeText(small, 2000)),
            );
            bestLarge = Math.min(
                bestLarge,
                cpuMs(() => sanitizeText(large, 2000)),
            );
        }
        return [bestSmall, bestLarge];
    }

    it.each(HOSTILE_SHAPES)("%s", (_name, shape) => {
        const small = shape(SMALL);
        const large = shape(LARGE);
        sanitizeText(small, 2000); // warm up

        const [smallMs, largeMs] = bestOfMs(small, large);

        expect(largeMs / Math.max(smallMs, NOISE_FLOOR_MS)).toBeLessThan(
            MAX_GROWTH,
        );
    });
});

describe("hidden characters, exhaustively", () => {
    const INVISIBLE = /[\p{Default_Ignorable_Code_Point}\p{Bidi_Control}]/u;

    it("removes every default-ignorable and every bidi control character", () => {
        const kept: string[] = [];
        for (let codePoint = 0; codePoint <= 0x10ffff; codePoint += 1) {
            if (codePoint >= 0xd800 && codePoint <= 0xdfff) {
                continue;
            }
            const character = String.fromCodePoint(codePoint);
            if (
                INVISIBLE.test(character) &&
                sanitizeText(`a${character}b`, 100) !== "ab"
            ) {
                kept.push(codePoint.toString(16));
            }
        }
        expect(kept).toEqual([]);
    });

    it("removes every control character except tab, newline and carriage return", () => {
        const kept: string[] = [];
        for (let codePoint = 0; codePoint <= 0x9f; codePoint += 1) {
            const isLayout = [0x09, 0x0a, 0x0d].includes(codePoint);
            const isControl = codePoint < 0x20 || codePoint >= 0x7f;
            const character = String.fromCodePoint(codePoint);
            if (
                isControl &&
                !isLayout &&
                sanitizeText(`a${character}b`, 100) !== "ab"
            ) {
                kept.push(codePoint.toString(16));
            }
        }
        expect(kept).toEqual([]);
    });

    it("removes every unpaired surrogate", () => {
        for (let unit = 0xd800; unit <= 0xdfff; unit += 1) {
            expect(sanitizeText(`a${String.fromCharCode(unit)}b`, 100)).toBe(
                "ab",
            );
        }
    });

    it("keeps printable ASCII that is not markup", () => {
        const plain = "a.b,c;d?e-f_g=h+i*j#k%l^m~n|o{p}q\"r's`t/u\\v w$x";
        expect(sanitizeText(plain, 1000)).toBe(plain);
    });
});

/** Deterministic pseudo-random numbers, so a failure reproduces. */
function lcg(seed: number): () => number {
    let state = seed >>> 0;
    return () => {
        state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
        return state / 4294967296;
    };
}

const FUZZ_TOKENS = [
    "<",
    ">",
    "<!--",
    "-->",
    "--!>",
    "<!",
    "!",
    "[",
    "]",
    "(",
    ")",
    "](",
    "]:",
    "![",
    "@",
    "a",
    "Z",
    "0",
    "-",
    "_",
    ":",
    "/",
    "//",
    "://",
    "\\",
    "\\:",
    "\\/",
    ".",
    "www.",
    "WWW",
    "http",
    "&",
    "&#",
    ";",
    "lt",
    "gt",
    "amp",
    "&lt;",
    "&amp;",
    "&gt;",
    "#",
    "x",
    "64",
    "\u200B",
    "\u200D",
    "\u202E",
    "\u2066",
    "\uFEFF",
    "\u00AD",
    "\uFE0F",
    "\u{E0041}",
    "\u{E007F}",
    "\u{E0100}",
    "\u0301",
    "\u0301\u0301",
    "\u0000",
    "\r",
    "\r\n",
    "\n",
    "\u2028",
    "\t",
    " ",
    "  ",
    "`",
    "```",
    "~~~",
    "\u{1F600}",
    "\uD800",
    "\uDC00",
    "\uFF20",
    "\uFF1C",
    "\uFF1E",
    "@-",
    "@_",
    "@.",
    "a@",
    "\\-",
    "\\_",
    "\\.",
    "\u2026",
    "\u00E9",
    "\u3164",
    "mailto",
    "javascript",
    "img",
    "src=",
    '"',
    "'",
    "=",
    "?",
    "@a",
    "@org/team",
    "https://e.example",
    "[t](u)",
    "![a](b)",
    "[r]: u",
];

function fuzzInput(next: () => number): string {
    const count = 1 + Math.floor(next() * 60);
    let text = "";
    for (let index = 0; index < count; index += 1) {
        text += FUZZ_TOKENS[Math.floor(next() * FUZZ_TOKENS.length)];
    }
    return text;
}

describe("seeded fuzz", () => {
    it("is idempotent and leaves nothing active, at every cap", () => {
        const next = lcg(20261003);
        const caps = [100_000, 4000, 40, 17, 9, 6, 5, 3, 1];
        const problems: string[] = [];
        for (let run = 0; run < 30_000; run += 1) {
            const input = fuzzInput(next);
            const cap = caps[run % caps.length];
            const singleLine = run % 5 === 0;
            const once = sanitizeText(input, cap, { singleLine });
            const twice = sanitizeText(once, cap, { singleLine });

            const problem =
                once.length > cap
                    ? "over the cap"
                    : twice !== once
                      ? "not idempotent"
                      : cap >= 40
                        ? activeMarkupProblem(once)
                        : null;
            if (problem !== null && problems.length < 5) {
                problems.push(`${problem}: ${JSON.stringify(input)}`);
            }
        }
        expect(problems).toEqual([]);
    });
});
