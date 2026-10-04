import {
    chmodSync,
    mkdirSync,
    mkdtempSync,
    readdirSync,
    rmSync,
    statSync,
    symlinkSync,
    writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
    buildIncompleteSummary,
    createOutcomeLocation,
    displayPath,
    MAX_DISPLAY_PATH_LENGTH,
    MAX_LISTED_HUNKS,
    parseOutcomeRecord,
    readOutcomeRecord,
    REVIEW_INCOMPLETE_EXIT_CODE,
    REVIEW_NOT_RUN_EXIT_CODE,
    reviewConclusion,
    type UnreviewedHunk,
} from "../src/review-outcome.js";
import { activeMarkupProblem } from "./support/active-markup.js";
import { loadConclusionCases } from "./support/conclusion-cases.js";

const CASES = loadConclusionCases();
const APP_ROWS = CASES.cases.filter((row) =>
    (row.applies_to ?? ["workflow", "app"]).includes("app"),
);

/** The size rule 6 of this project names: a hostile input of 50,000 characters. */
const HOSTILE_SIZE = 50_000;
const TIME_LIMIT_MS = 2000;
// The work below is measured in CPU time, so a busy machine slows the wall clock but
// not the measurement.
vi.setConfig({ testTimeout: 120_000 });

describe("the exit codes of trelix review", () => {
    it("are 3 for a review that did not run and 4 for one that covered only part of the diff", () => {
        // tests/unit/test_review_exit_code_contract.py ties these to src/trelix/cli/main.py.
        expect(REVIEW_NOT_RUN_EXIT_CODE).toBe(3);
        expect(REVIEW_INCOMPLETE_EXIT_CODE).toBe(4);
    });
});

describe("reviewConclusion, against the table the workflow is tested with", () => {
    it("has rows for every conclusion and every exit code", () => {
        expect(APP_ROWS.length).toBeGreaterThanOrEqual(14);
        expect(new Set(APP_ROWS.map((row) => row.conclusion))).toEqual(
            new Set(["success", "failure", "neutral"]),
        );
        expect(new Set(APP_ROWS.map((row) => row.exit_code))).toEqual(
            new Set(["0", "1", "2", "3", "4", "5"]),
        );
    });

    it.each(APP_ROWS.map((row) => [row.id, row] as const))("%s", (_id, row) => {
        // A row with stdout_text is output whose findings cannot be read: null, not [].
        const findings =
            row.stdout_text === undefined
                ? row.severities.map((severity) => ({ severity }))
                : null;

        expect(reviewConclusion(Number(row.exit_code), findings)).toBe(
            row.conclusion,
        );
    });

    it("has rows for findings that cannot be read, for exit 0 and for exit 4", () => {
        const unreadable = APP_ROWS.filter(
            (row) => row.stdout_text !== undefined,
        );

        expect(new Set(unreadable.map((row) => row.exit_code))).toEqual(
            new Set(["0", "4"]),
        );
        expect(unreadable.length).toBeGreaterThanOrEqual(6);
    });

    it("is neutral when no exit status is known", () => {
        expect(reviewConclusion(null, [])).toBe("neutral");
        expect(reviewConclusion(null, [{ severity: "ERROR" }])).toBe("neutral");
    });

    it("is neutral, never success, when the findings could not be read, whatever the exit status", () => {
        // null is "could not be read"; [] is "there are none" (exit 0: success).
        for (const exitCode of [0, 1, 2, 3, 4, 5, null]) {
            expect(reviewConclusion(exitCode, null)).toBe("neutral");
        }
        expect(reviewConclusion(0, [])).toBe("success");
    });

    it("judges by every finding, not by the first fifty that get an annotation", () => {
        const findings = [
            ...Array.from({ length: 55 }, () => ({ severity: "INFO" })),
            { severity: "ERROR" },
        ];

        expect(reviewConclusion(0, findings)).toBe("failure");
        expect(reviewConclusion(4, findings)).toBe("failure");
    });

    it("counts only the exact severity ERROR", () => {
        for (const severity of [
            "error",
            "Error",
            "FAILURE",
            "",
            undefined,
            1,
        ]) {
            expect(reviewConclusion(0, [{ severity }])).toBe("success");
            expect(reviewConclusion(4, [{ severity }])).toBe("neutral");
        }
    });
});

const VALID = CASES.valid_outcome;

describe("parseOutcomeRecord", () => {
    it("reads the record trelix writes", () => {
        expect(parseOutcomeRecord(JSON.stringify(VALID))).toEqual({
            hunksTotal: 5,
            hunksReviewed: 3,
            hunksUnreviewed: 2,
            hunksOmitted: 0,
            hunks: [
                { file: "src/a.py", line: 10, status: "truncated" },
                { file: "src/b.py", line: 20, status: "refused" },
            ],
        });
    });

    it("keeps only file, line and status of a hunk: the detail string is never read on", () => {
        const record = parseOutcomeRecord(
            JSON.stringify({
                ...VALID,
                hunks: [
                    { ...VALID.hunks[0], detail: "canary-detail" },
                    VALID.hunks[1],
                ],
            }),
        );

        expect(JSON.stringify(record)).not.toContain("canary-detail");
    });

    it("counts hunks the record does not list", () => {
        const hunks = Array.from({ length: 100 }, (_, i) => ({
            file: `f${i}.py`,
            line: i + 1,
            status: "error",
            detail: "x",
        }));

        const record = parseOutcomeRecord(
            JSON.stringify({
                ...VALID,
                hunks_total: 200,
                hunks_reviewed: 70,
                hunks_unreviewed: 130,
                hunks,
                hunks_omitted: 30,
            }),
        );

        expect(record?.hunksUnreviewed).toBe(130);
        expect(record?.hunksOmitted).toBe(30);
        expect(record?.hunks).toHaveLength(100);
    });

    it("has rows of bad records to reject", () => {
        expect(CASES.invalid_outcomes.length).toBeGreaterThanOrEqual(30);
    });

    it.each(CASES.invalid_outcomes.map((row) => [row.id, row.text] as const))(
        "reads as unknown: %s",
        (_id, text) => {
            expect(parseOutcomeRecord(text)).toBeNull();
        },
    );

    it("accepts every status of the vocabulary and no other", () => {
        for (const status of [
            "truncated",
            "refused",
            "parse_failed",
            "error",
        ]) {
            const text = JSON.stringify({
                ...VALID,
                hunks_total: 2,
                hunks_reviewed: 1,
                hunks_unreviewed: 1,
                hunks: [{ file: "a.py", line: 1, status, detail: "x" }],
            });

            expect(parseOutcomeRecord(text)?.hunks[0].status).toBe(status);
        }
    });
});

describe("readOutcomeRecord", () => {
    let dir: string;

    beforeEach(() => {
        dir = mkdtempSync(join(tmpdir(), "trelix-outcome-read-"));
    });

    afterEach(() => {
        rmSync(dir, { recursive: true, force: true });
    });

    const quiet = () => vi.spyOn(console, "warn").mockImplementation(() => {});
    const MAX_BYTES = 1024 * 1024;

    function paddedTo(size: number): string {
        const text = JSON.stringify(VALID);
        return text + " ".repeat(size - Buffer.byteLength(text));
    }

    it("reads a valid record", async () => {
        const path = join(dir, "outcome.json");
        writeFileSync(path, JSON.stringify(VALID));

        expect((await readOutcomeRecord(path))?.hunksUnreviewed).toBe(2);
    });

    it("reads a record of exactly the size limit", async () => {
        const path = join(dir, "outcome.json");
        writeFileSync(path, paddedTo(MAX_BYTES));

        expect((await readOutcomeRecord(path))?.hunksUnreviewed).toBe(2);
    });

    it("treats a record one byte over the size limit as unknown, and says so in the log", async () => {
        const warn = quiet();
        const path = join(dir, "outcome.json");
        writeFileSync(path, paddedTo(MAX_BYTES + 1));

        expect(await readOutcomeRecord(path)).toBeNull();
        expect(warn).toHaveBeenCalledTimes(1);
        warn.mockRestore();
    });

    it("treats a missing file as unknown, and says so in the log", async () => {
        const warn = quiet();

        expect(await readOutcomeRecord(join(dir, "absent.json"))).toBeNull();
        expect(warn).toHaveBeenCalledTimes(1);
        warn.mockRestore();
    });

    it("treats a directory as unknown", async () => {
        const warn = quiet();
        const path = join(dir, "outcome.json");
        mkdirSync(path);

        expect(await readOutcomeRecord(path)).toBeNull();
        warn.mockRestore();
    });

    it("treats a symlink to a valid record as unknown: the record is never followed", async () => {
        const warn = quiet();
        const target = join(dir, "target.json");
        writeFileSync(target, JSON.stringify(VALID));
        const path = join(dir, "outcome.json");
        symlinkSync(target, path);

        expect(await readOutcomeRecord(path)).toBeNull();
        warn.mockRestore();
    });

    it("treats an invalid record as unknown, and says so in the log", async () => {
        const warn = quiet();
        const path = join(dir, "outcome.json");
        writeFileSync(path, "{not json");

        expect(await readOutcomeRecord(path)).toBeNull();
        expect(warn).toHaveBeenCalledTimes(1);
        warn.mockRestore();
    });
});

describe("createOutcomeLocation", () => {
    let base: string;

    beforeEach(() => {
        base = mkdtempSync(join(tmpdir(), "trelix-outcome-base-"));
    });

    afterEach(() => {
        chmodSync(base, 0o700);
        rmSync(base, { recursive: true, force: true });
    });

    it("makes a fresh private directory named like the checkout's, so a boot-time sweep covers it", async () => {
        const first = await createOutcomeLocation(base);
        const second = await createOutcomeLocation(base);

        const names = readdirSync(base);
        expect(names).toHaveLength(2);
        for (const name of names) {
            expect(name).toMatch(/^trelix-review-outcome-[A-Za-z0-9]{6}$/);
            expect(statSync(join(base, name)).mode & 0o777).toBe(0o700);
        }
        expect(first.file).not.toBe(second.file);
        expect(first.file.startsWith(base)).toBe(true);
        expect(first.file.endsWith("/outcome.json")).toBe(true);
    });

    it("removes the directory and everything in it, and can be called again without complaint", async () => {
        const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
        const location = await createOutcomeLocation(base);
        writeFileSync(location.file, "{}");

        await location.cleanup();
        await location.cleanup();

        expect(readdirSync(base)).toEqual([]);
        expect(warn).not.toHaveBeenCalled();
        warn.mockRestore();
    });

    it.skipIf(process.getuid?.() === 0)(
        "logs instead of rejecting when the directory cannot be removed",
        async () => {
            const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
            const location = await createOutcomeLocation(base);
            writeFileSync(location.file, "{}");
            // A parent that cannot be written to: removing the directory fails.
            chmodSync(base, 0o500);

            await expect(location.cleanup()).resolves.toBeUndefined();

            expect(warn).toHaveBeenCalledTimes(1);
            warn.mockRestore();
        },
    );
});

const AT = "\uFF20"; // fullwidth commercial at
const LT = "\uFF1C"; // fullwidth less-than sign
const GT = "\uFF1E"; // fullwidth greater-than sign

describe("displayPath", () => {
    const cases: Array<[name: string, file: string, shown: string]> = [
        [
            "a heading and a link on new lines",
            "src/a.py\n# Heading\n- [x](https://attacker.example/canary)",
            "src/a.py # Heading - [x] (https[:]//attacker.example/canary)",
        ],
        [
            "carriage returns and line separators",
            "a\rb\u2028c\u2029d",
            "a b c d",
        ],
        ["a tab", "a\tb", "a b"],
        [
            "backticks that would close the span",
            "a`b`, **bold**",
            "a\u02CBb\u02CB, **bold**",
        ],
        [
            "a long run of backticks",
            "`".repeat(200),
            "\u02CB".repeat(99) + "\u2026",
        ],
        [
            "html",
            "<img src=x onerror=alert(1)><script>canary</script>",
            `${LT}img src=x onerror=alert(1)${GT}${LT}script${GT}canary${LT}/script${GT}`,
        ],
        ["an html comment opener", "src/<!-- canary", "src/"],
        [
            "a mention and a link",
            "@octo-org/team [x](https://attacker.example/canary)",
            `${AT}octo-org/team [x] (https[:]//attacker.example/canary)`,
        ],
        ["a bidi override", "src/\u202Eexe.py", "src/exe.py"],
        ["zero-width characters", "sr\u200Bc/\u2060a.py", "src/a.py"],
        ["tag characters", "a\u{E0041}\u{E0042}.py", "a.py"],
        ["control characters", "a\u0000b\u001Bc\u007Fd\u0085e.py", "abcde.py"],
        [
            "a run of combining marks",
            "a" + "\u0301".repeat(60) + ".py",
            "a" + "\u0301".repeat(4) + ".py",
        ],
        ["a lone surrogate", "a\uD800b.py", "ab.py"],
        ["a character reference", "a&#64;octocat.py", "a&amp;#64;octocat.py"],
        [
            "nothing but hidden characters",
            "\u200B\u200C\u202E",
            "(unprintable file name)",
        ],
        ["an empty name", "", "(unprintable file name)"],
        [
            "a very long name",
            "x".repeat(HOSTILE_SIZE),
            "x".repeat(99) + "\u2026",
        ],
    ];

    it.each(cases)("%s", (_name, file, shown) => {
        expect(displayPath(file)).toBe(shown);
    });

    it("never shows a backtick, however the name is built", () => {
        for (const [, file] of cases) {
            expect(displayPath(file)).not.toContain("`");
        }
    });

    it("cuts at the limit, not before it", () => {
        expect(MAX_DISPLAY_PATH_LENGTH).toBe(100);
        expect(displayPath("y".repeat(100))).toBe("y".repeat(100));
        expect(displayPath("y".repeat(101))).toBe("y".repeat(99) + "\u2026");
    });

    it("does not split a surrogate pair at the cut", () => {
        const shown = displayPath("y".repeat(98) + "\u{1F600}\u{1F600}");

        // The first emoji would have been cut in half; it is dropped whole instead.
        expect(shown).toBe("y".repeat(98) + "\u2026");
    });

    // Only the first 400 characters of a name are looked at (four times the 100 shown), so
    // what it costs to show a name does not depend on how long the name is. The hidden
    // characters below are removed before the cap, so a name that is nothing but hidden
    // characters in its first 400 reads as empty, whatever follows.
    it("reads the first 400 characters of a name and no more", () => {
        expect(displayPath("\u200B".repeat(400) + "visible.py")).toBe(
            "(unprintable file name)",
        );
        expect(displayPath("\u200B".repeat(500) + "visible.py")).toBe(
            "(unprintable file name)",
        );
    });

    it("reads the 400th character of a name", () => {
        expect(displayPath("\u200B".repeat(399) + "visible.py")).toBe("v");
    });
});

const RECORD_HEADER = {
    ...VALID,
    hunks_total: 5,
    hunks_reviewed: 3,
    hunks_unreviewed: 2,
};

function record(hunks: UnreviewedHunk[], unreviewed = hunks.length) {
    const parsed = parseOutcomeRecord(
        JSON.stringify({
            ...RECORD_HEADER,
            hunks_total: unreviewed + 3,
            hunks_unreviewed: unreviewed,
            hunks: hunks.map((hunk) => ({ ...hunk, detail: "x" })),
            hunks_omitted: unreviewed - hunks.length,
        }),
    );
    if (parsed === null) {
        throw new Error("the test record was rejected");
    }
    return parsed;
}

describe("buildIncompleteSummary", () => {
    const outcome = parseOutcomeRecord(JSON.stringify(VALID));

    it("says what was covered, how many findings there are and which hunks were left out", () => {
        expect(buildIncompleteSummary(1, 1, outcome)).toBe(
            [
                "trelix reviewed only part of this PR: 3 of 5 hunks were reviewed and 2 were not. This is not a clean result.",
                "",
                "1 issue(s) found in the hunks that were reviewed.",
                "",
                "Hunks that were not reviewed:",
                "- `src/a.py:10` (truncated)",
                "- `src/b.py:20` (refused)",
            ].join("\n"),
        );
    });

    it("says there are no findings in the hunks that were reviewed, not that there are none", () => {
        const summary = buildIncompleteSummary(0, 0, outcome);

        expect(summary).toContain(
            "No findings in the hunks that were reviewed.",
        );
        expect(summary).toContain("This is not a clean result.");
    });

    it("says the findings could not be read, which is not the same as none", () => {
        const summary = buildIncompleteSummary(null, 0, outcome);

        expect(summary).toContain(
            "The list of findings could not be read, so none are shown.",
        );
        expect(summary).not.toContain("No findings");
    });

    it("says when findings could not be shown as annotations", () => {
        const summary = buildIncompleteSummary(60, 50, outcome);

        expect(summary).toContain(
            "60 issue(s) found in the hunks that were reviewed. 10 of them could not be shown as inline annotations",
        );
    });

    it("says the extent is unknown, with no numbers and no list, when there is no record", () => {
        const summary = buildIncompleteSummary(1, 1, null);

        expect(summary).toBe(
            [
                "trelix reviewed only part of this PR, but the record of what it covered is missing or unreadable, so how much was left unreviewed is unknown. This is not a clean result.",
                "",
                "1 issue(s) found in the hunks that were reviewed.",
            ].join("\n"),
        );
    });

    it("lists ten hunks and counts the rest, including those the record does not list", () => {
        const hunks: UnreviewedHunk[] = Array.from({ length: 100 }, (_, i) => ({
            file: `src/f${i}.py`,
            line: i + 1,
            status: "truncated" as const,
        }));

        const lines = buildIncompleteSummary(0, 0, record(hunks, 130)).split(
            "\n",
        );

        const entries = lines.filter((line) => line.startsWith("- `"));
        expect(MAX_LISTED_HUNKS).toBe(10);
        expect(entries).toHaveLength(10);
        expect(entries[0]).toBe("- `src/f0.py:1` (truncated)");
        expect(entries[9]).toBe("- `src/f9.py:10` (truncated)");
        expect(lines).toContain("- and 120 more.");
    });

    it.each([
        [10, []],
        [11, ["- and 1 more."]],
    ])(
        "adds the 'more' line only when hunks are left unlisted (%i hunks)",
        (count, more) => {
            const hunks: UnreviewedHunk[] = Array.from(
                { length: count },
                (_, i) => ({
                    file: `f${i}.py`,
                    line: 1,
                    status: "error" as const,
                }),
            );

            const lines = buildIncompleteSummary(0, 0, record(hunks)).split(
                "\n",
            );

            expect(lines.filter((line) => line.startsWith("- and"))).toEqual(
                more,
            );
        },
    );

    it("fits the summary limit with ten of the longest names the sanitiser can make longer", () => {
        // "&a;" grows to "&amp;a;" in the sanitiser: the worst case for the length.
        const hunks: UnreviewedHunk[] = Array.from({ length: 10 }, () => ({
            file: "&a;".repeat(60),
            line: Number.MAX_SAFE_INTEGER,
            status: "parse_failed" as const,
        }));

        const summary = buildIncompleteSummary(60, 50, record(hunks, 100));

        expect(summary.length).toBeLessThan(3500);
    });
});

/** Every line of the summary is one buildIncompleteSummary writes itself. */
function assertOnlyKnownLines(summary: string, entries: number): string[] {
    const spans: string[] = [];
    const entry = /^- `([^`\n]+)` \((truncated|refused|parse_failed|error)\)$/;
    const known = [
        /^trelix reviewed only part of this PR: \d+ of \d+ hunks were reviewed and \d+ were not\. This is not a clean result\.$/,
        /^$/,
        /^No findings in the hunks that were reviewed\.$/,
        /^\d+ issue\(s\) found in the hunks that were reviewed\.( \d+ of them could not be shown as inline annotations .*)?$/,
        /^Hunks that were not reviewed:$/,
        /^- and \d+ more\.$/,
    ];
    for (const line of summary.split("\n")) {
        const match = entry.exec(line);
        if (match !== null) {
            spans.push(match[1]);
        } else if (!known.some((pattern) => pattern.test(line))) {
            throw new Error(
                `not a line the summary writes: ${JSON.stringify(line)}`,
            );
        }
    }
    expect(spans).toHaveLength(entries);
    expect(summary.split("`").length - 1).toBe(2 * entries);
    return spans;
}

const HOSTILE_FILES = [
    "src/a.py\n# Heading\n- [x](https://attacker.example/canary)",
    "a`b`, **bold**",
    "`".repeat(200),
    "<img src=x onerror=alert(1)><script>canary</script>",
    "src/<!-- canary",
    "@octo-org/team [x](https://attacker.example/canary)",
    "src/\u202Eexe.py",
    "sr\u200Bc/\u2060a.py",
    "a\u0000b\u001Bc\u007Fd\u0085e.py",
    "a" + "\u0301".repeat(60) + ".py",
    "a\uD800b.py",
    "a&#64;octocat.py",
];

describe("a file name in the summary", () => {
    it("adds no line, no active markup and no hidden character, whatever it holds", () => {
        const hunks: UnreviewedHunk[] = HOSTILE_FILES.slice(0, 10).map(
            (file, i) => ({ file, line: i, status: "refused" as const }),
        );

        const summary = buildIncompleteSummary(0, 0, record(hunks));

        assertOnlyKnownLines(summary, 10);
        expect(activeMarkupProblem(summary)).toBeNull();
    });

    it.each(HOSTILE_FILES.map((file, i) => [i, file] as const))(
        "stays inside one code span: hostile name %i",
        (_i, file) => {
            const summary = buildIncompleteSummary(
                0,
                0,
                record([{ file, line: 3, status: "error" }]),
            );

            const [span] = assertOnlyKnownLines(summary, 1);
            expect(span.endsWith(":3")).toBe(true);
            expect(activeMarkupProblem(summary)).toBeNull();
        },
    );
});

// Inputs that make a parser or a backtracking expression slow: long runs of
// backticks, tildes, spaces and newlines, nested brackets, unterminated comment
// openers, and zero-width or combining characters.
const repeated = (unit: string) =>
    unit.repeat(Math.ceil(HOSTILE_SIZE / unit.length)).slice(0, HOSTILE_SIZE);
const SLOW_SHAPES: Array<[string, string]> = [
    ["backticks", repeated("`")],
    ["tildes", repeated("~")],
    ["spaces", repeated(" ")],
    ["newlines", repeated("\n")],
    ["carriage returns and line feeds", repeated("\r\n")],
    ["open brackets", repeated("[")],
    [
        "nested brackets",
        "[".repeat(HOSTILE_SIZE / 2) + "]".repeat(HOSTILE_SIZE / 2),
    ],
    ["image and link openers", repeated("![](")],
    ["unterminated comment openers", repeated("<!--")],
    ["one comment opener and dashes", "<!--" + "-".repeat(HOSTILE_SIZE)],
    ["zero-width characters", repeated("\u200B")],
    ["combining marks", "a" + "\u0301".repeat(HOSTILE_SIZE)],
    [
        "combining marks between letters",
        repeated("a\u0301\u0302\u0303\u0304\u0305"),
    ],
    ["backslashes", repeated("\\")],
    ["character references", repeated("&a;")],
    ["at signs", repeated("@")],
    ["scheme separators", repeated("://")],
];

/** CPU milliseconds spent by `work`. */
function cpuMs(work: () => void): number {
    const before = process.cpuUsage();
    work();
    const used = process.cpuUsage(before);
    return (used.user + used.system) / 1000;
}

describe("hostile input is linear", () => {
    it.each(SLOW_SHAPES)(
        "a 50000-character name of %s is read and shown in under two seconds",
        (_shape, file) => {
            expect(file.length).toBeGreaterThanOrEqual(HOSTILE_SIZE);
            const text = JSON.stringify({
                ...VALID,
                hunks_total: 6,
                hunks_reviewed: 1,
                hunks_unreviewed: 5,
                hunks: Array.from({ length: 5 }, (_, i) => ({
                    file,
                    line: i + 1,
                    status: "error",
                    detail: "x",
                })),
                hunks_omitted: 0,
            });
            let summary = "";

            const elapsed = cpuMs(() => {
                const parsed = parseOutcomeRecord(text);
                summary = buildIncompleteSummary(1, 1, parsed);
            });

            expect(elapsed).toBeLessThan(TIME_LIMIT_MS);
            // The record was read (five names were listed), not rejected as too large.
            assertOnlyKnownLines(summary, 5);
        },
    );

    it("a record of ten thousand entries is read and summarised in under two seconds", () => {
        const text = JSON.stringify({
            ...VALID,
            hunks_total: 10_001,
            hunks_reviewed: 1,
            hunks_unreviewed: 10_000,
            hunks: Array.from({ length: 10_000 }, (_, i) => ({
                file: `src/dir${i % 50}/file${i}.py`,
                line: i,
                status: "truncated",
                detail: "x",
            })),
            hunks_omitted: 0,
        });
        let summary = "";

        const elapsed = cpuMs(() => {
            summary = buildIncompleteSummary(0, 0, parseOutcomeRecord(text));
        });

        expect(elapsed).toBeLessThan(TIME_LIMIT_MS);
        assertOnlyKnownLines(summary, 10);
    });
});
