import { describe, expect, it } from "vitest";
import { parseFindings } from "../src/review-runner.js";

/** The size the project's rule names for a hostile input, and the time it must be handled in. */
const HOSTILE_SIZE = 50_000;
const TIME_LIMIT_MS = 2000;

const FINDING = {
    file: "src/a.py",
    lines: "3-4",
    severity: "WARN",
    comment: "smell",
};

describe("parseFindings", () => {
    it("returns the findings of an array of objects, in order", () => {
        const other = { ...FINDING, file: "src/b.py", severity: "ERROR" };

        expect(parseFindings(JSON.stringify([FINDING, other]))).toEqual([
            FINDING,
            other,
        ]);
    });

    it("returns an empty array as it is: no findings is a result, not an unreadable one", () => {
        expect(parseFindings("[]")).toEqual([]);
        expect(parseFindings(" [ ]\n")).toEqual([]);
    });

    it.each([
        ["nothing at all", ""],
        ["text", "not json"],
        ["a cut-off array", '[{"file":"a.py"'],
    ])("throws when stdout is %s", (_name, stdout) => {
        expect(() => parseFindings(stdout)).toThrow(SyntaxError);
    });

    it.each([
        ["an object", "{}"],
        ["an object with findings in it", '{"findings":[]}'],
        ["null", "null"],
        ["a string", '"x"'],
        ["a number", "5"],
        ["true", "true"],
    ])("throws when stdout is %s, not an array", (_name, stdout) => {
        expect(() => parseFindings(stdout)).toThrow(
            "trelix review did not print a JSON array",
        );
    });

    it.each([
        ["null", "[null]"],
        ["a number", "[1]"],
        ["a string", '["x"]'],
        ["true", "[true]"],
        ["an array", "[[]]"],
        ["null after a finding", `[${JSON.stringify(FINDING)},null]`],
        ["null before a finding", `[null,${JSON.stringify(FINDING)}]`],
    ])(
        "throws when the array holds %s: toAnnotations would throw on it later",
        (_name, stdout) => {
            expect(() => parseFindings(stdout)).toThrow(
                "trelix review printed a finding that is not an object",
            );
        },
    );
});

/** CPU milliseconds spent by `work`: a busy machine slows the clock, not this. */
function cpuMs(work: () => void): number {
    const before = process.cpuUsage();
    work();
    const used = process.cpuUsage(before);
    return (used.user + used.system) / 1000;
}

/** parseFindings on `text`, whether it returns or throws. */
function attempt(text: string): void {
    try {
        parseFindings(text);
    } catch {
        // Unreadable output is an answer too; only the time it took matters here.
    }
}

const repeated = (unit: string, size: number) =>
    unit.repeat(Math.ceil(size / unit.length)).slice(0, size);

/** `[{},{},...,{}]`: as many objects as fit in `size` characters; `last` is the final element. */
const manyObjects = (size: number, last = "{}") =>
    `[${"{},".repeat(Math.floor(size / 3))}${last}]`;

// Inputs that make a parser or a backtracking expression slow: long runs of
// backticks, tildes, spaces and newlines, nested brackets, unterminated comment
// openers, zero-width or combining characters, and arrays with as many
// elements as fit, which the check of every element has to walk.
const SHAPES: Array<[name: string, build: (size: number) => string]> = [
    ["backticks", (size) => repeated("`", size)],
    ["tildes", (size) => repeated("~", size)],
    ["spaces in an empty array", (size) => `[${" ".repeat(size)}]`],
    ["newlines in an empty array", (size) => `[${"\n".repeat(size)}]`],
    ["open brackets", (size) => repeated("[", size)],
    ["nested brackets", (size) => "[".repeat(size / 2) + "]".repeat(size / 2)],
    ["unterminated comment openers", (size) => repeated("<!--", size)],
    ["zero-width characters", (size) => repeated("\u200B", size)],
    ["combining marks", (size) => "a" + "\u0301".repeat(size)],
    [
        "a finding with a long comment of backticks",
        (size) => JSON.stringify([{ ...FINDING, comment: "`".repeat(size) }]),
    ],
    ["as many empty objects as fit", (size) => manyObjects(size)],
    [
        "as many empty objects as fit, then a null",
        (size) => manyObjects(size, "null"),
    ],
    [
        "as many nulls as fit",
        (size) => `[${"null,".repeat(Math.floor(size / 5))}null]`,
    ],
];

describe("parseFindings is linear on hostile input", () => {
    it.each(SHAPES)(
        "%s, 50000 characters, is read in under two seconds",
        (_name, build) => {
            const text = build(HOSTILE_SIZE);
            expect(text.length).toBeGreaterThanOrEqual(HOSTILE_SIZE);

            expect(cpuMs(() => attempt(text))).toBeLessThan(TIME_LIMIT_MS);
        },
    );

    it("reads the longest array of objects that fits, all of them", () => {
        expect(parseFindings(manyObjects(HOSTILE_SIZE))).toHaveLength(
            Math.floor(HOSTILE_SIZE / 3) + 1,
        );
    });

    // A quadratic check can still finish 50,000 characters inside the limit on a
    // fast machine, so the time is also compared across sizes: eight times the
    // input must cost about eight times the time, not sixty-four.
    describe("and its time grows with the input, not with its square", () => {
        const SMALL = 25_000;
        const LARGE = 200_000;
        const MAX_GROWTH = 30;
        const NOISE_FLOOR_MS = 2;
        const ROUNDS = 3;

        it.each(SHAPES.filter(([name]) => name.includes("empty objects")))(
            "%s",
            (_name, build) => {
                const small = build(SMALL);
                const large = build(LARGE);
                attempt(small); // warm up
                let bestSmall = Infinity;
                let bestLarge = Infinity;
                for (let round = 0; round < ROUNDS; round += 1) {
                    bestSmall = Math.min(
                        bestSmall,
                        cpuMs(() => attempt(small)),
                    );
                    bestLarge = Math.min(
                        bestLarge,
                        cpuMs(() => attempt(large)),
                    );
                }

                expect(
                    bestLarge / Math.max(bestSmall, NOISE_FLOOR_MS),
                ).toBeLessThan(MAX_GROWTH);
            },
        );
    });
});
