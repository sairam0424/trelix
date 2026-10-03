import { generateKeyPairSync } from "node:crypto";
import {
    chmodSync,
    mkdtempSync,
    readdirSync,
    readFileSync,
    rmSync,
    writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join, relative } from "node:path";
import { fileURLToPath } from "node:url";
import { Octokit } from "@octokit/rest";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
    postCheckRun,
    postReviewFailureCheckRun,
    ReviewFinding,
    runReview,
    toAnnotations,
} from "../src/review-runner.js";
import { sanitizeAnnotation, sanitizeCheckOutput } from "../src/sanitize.js";
import { AppConfig } from "../src/config.js";
import { activeMarkupProblem } from "./support/active-markup.js";

// The real sanitiser, wrapped so that a test can see what it was called with
// and, for one call, make it return something recognisable.
vi.mock("../src/sanitize.js", async (importOriginal) => {
    const actual = await importOriginal<typeof import("../src/sanitize.js")>();
    return {
        ...actual,
        sanitizeCheckOutput: vi.fn(actual.sanitizeCheckOutput),
        sanitizeAnnotation: vi.fn(actual.sanitizeAnnotation),
    };
});

const AT = "\uFF20"; // fullwidth commercial at
const LT = "\uFF1C"; // fullwidth less-than sign
const GT = "\uFF1E"; // fullwidth greater-than sign

/** Every field a finding can fill carries a word that must come out readable, next to something hostile. */
const HOSTILE_FINDING: ReviewFinding = {
    file: "src/\u202Ecanary-path.ts",
    lines: "7-9",
    severity: "WARN",
    comment:
        "canary-message ![](https://attacker.example/canary.png)[c](https://attacker.example/canary) " +
        "@canary-user <img src=x onerror=1> \u200B<!-- canary-hidden -->tail",
};

const SANITISED_HOSTILE_ANNOTATION = {
    path: "src/canary-path.ts",
    start_line: 7,
    end_line: 9,
    annotation_level: "warning",
    message: `canary-message [c] (https[:]//attacker.example/canary) ${AT}canary-user ${LT}img src=x onerror=1${GT} tail`,
    title: "trelix review",
};

function fakeOctokit() {
    const calls: Array<Record<string, unknown>> = [];
    const octokit = new Octokit({});
    octokit.hook.wrap("request", async (_request, options) => {
        if (
            options.method === "POST" &&
            options.url === "/repos/{owner}/{repo}/check-runs"
        ) {
            calls.push(options);
            return { status: 201, url: "", headers: {}, data: {} };
        }
        if (
            options.method === "GET" &&
            options.url === "/repos/{owner}/{repo}/pulls/{pull_number}"
        ) {
            return {
                status: 200,
                url: "",
                headers: {},
                data: { head: { sha: "deadbeef" } },
            };
        }
        throw new Error(
            `unexpected octokit request in test: ${options.method} ${options.url}`,
        );
    });
    return { octokit, calls };
}

beforeEach(() => {
    vi.mocked(sanitizeCheckOutput).mockClear();
    vi.mocked(sanitizeAnnotation).mockClear();
});

describe("toAnnotations", () => {
    it("returns sanitised annotations", () => {
        expect(toAnnotations([HOSTILE_FINDING])).toEqual([
            SANITISED_HOSTILE_ANNOTATION,
        ]);
    });

    it("builds every annotation through the sanitiser, and posts what it returns", () => {
        vi.mocked(sanitizeAnnotation).mockImplementationOnce((a) => ({
            ...a,
            path: `S[${a.path}]`,
            message: `S[${a.message}]`,
            title: `S[${a.title}]`,
        }));

        expect(
            toAnnotations([{ ...HOSTILE_FINDING, comment: "plain" }]),
        ).toEqual([
            {
                path: "S[src/\u202Ecanary-path.ts]",
                start_line: 7,
                end_line: 9,
                annotation_level: "warning",
                message: "S[plain]",
                title: "S[trelix review]",
            },
        ]);
    });

    it("skips a finding with an unusable path and still fills the limit", () => {
        const bad: ReviewFinding = { ...HOSTILE_FINDING, file: "../x" };
        const good: ReviewFinding = { ...HOSTILE_FINDING, file: "ok.py" };

        const annotations = toAnnotations([bad, bad, good, good, good], 2);

        expect(annotations.map((a) => a.path)).toEqual(["ok.py", "ok.py"]);
    });
});

describe("postCheckRun", () => {
    it("posts the sanitised title, summary and annotations", async () => {
        const { octokit, calls } = fakeOctokit();

        await postCheckRun(octokit, "o", "r", "deadbeef", [HOSTILE_FINDING]);

        expect(calls).toHaveLength(1);
        expect(calls[0]).toMatchObject({
            owner: "o",
            repo: "r",
            name: "trelix Code Review",
            head_sha: "deadbeef",
            status: "completed",
            conclusion: "success",
        });
        expect(calls[0].output).toEqual({
            title: "trelix found 1 issue(s)",
            summary: "trelix reviewed the PR and found 1 issue(s).",
            annotations: [SANITISED_HOSTILE_ANNOTATION],
        });
        const posted = calls[0].output as {
            annotations: Array<{ message: string }>;
        };
        expect(activeMarkupProblem(posted.annotations[0].message)).toBeNull();
    });

    it("posts whatever the sanitiser returns for the whole output", async () => {
        const { octokit, calls } = fakeOctokit();
        vi.mocked(sanitizeCheckOutput).mockImplementationOnce((output) => ({
            title: `S[${output.title}]`,
            summary: `S[${output.summary}]`,
            annotations: [],
        }));

        await postCheckRun(octokit, "o", "r", "deadbeef", []);

        expect(calls[0].output).toEqual({
            title: "S[trelix found 0 issue(s)]",
            summary: "S[trelix reviewed the PR and found 0 issue(s).]",
            annotations: [],
        });
        expect(vi.mocked(sanitizeCheckOutput)).toHaveBeenCalledTimes(1);
    });

    it("judges the verdict and the count by the findings, not by the annotations", async () => {
        const findings: ReviewFinding[] = [
            ...Array.from({ length: 59 }, (_, i) => ({
                file: `f${i}.py`,
                lines: "1-1",
                severity: "INFO" as const,
                comment: "note",
            })),
            {
                file: "last.py",
                lines: "1-1",
                severity: "ERROR",
                comment: "bug",
            },
        ];
        const { octokit, calls } = fakeOctokit();

        await postCheckRun(octokit, "o", "r", "deadbeef", findings);

        const output = calls[0].output as {
            title: string;
            summary: string;
            annotations: unknown[];
        };
        expect(calls[0].conclusion).toBe("failure");
        expect(output.title).toBe("trelix found 60 issue(s)");
        expect(output.annotations).toHaveLength(50);
        expect(output.summary).toBe(
            "trelix reviewed the PR and found 60 issue(s). 10 of them could not be shown as inline annotations (GitHub allows 50 per check, or the file path was not usable).",
        );
    });

    it("does not turn an error with an unusable path into a success", async () => {
        const { octokit, calls } = fakeOctokit();

        await postCheckRun(octokit, "o", "r", "deadbeef", [
            { file: "../x", lines: "1-1", severity: "ERROR", comment: "bug" },
            { file: "ok.py", lines: "2-2", severity: "INFO", comment: "note" },
        ]);

        const output = calls[0].output as {
            title: string;
            summary: string;
            annotations: Array<{ path: string }>;
        };
        expect(calls[0].conclusion).toBe("failure");
        expect(output.title).toBe("trelix found 2 issue(s)");
        expect(output.annotations.map((a) => a.path)).toEqual(["ok.py"]);
        expect(output.summary).toContain("1 of them could not be shown");
    });

    it("says nothing extra when every finding has an annotation", async () => {
        const { octokit, calls } = fakeOctokit();

        await postCheckRun(octokit, "o", "r", "deadbeef", []);

        expect(calls[0].conclusion).toBe("success");
        expect(calls[0].output).toEqual({
            title: "trelix found 0 issue(s)",
            summary: "trelix reviewed the PR and found 0 issue(s).",
            annotations: [],
        });
    });
});

describe("postReviewFailureCheckRun", () => {
    it("posts the sanitiser's result for the title and the summary of every failure kind", async () => {
        const errors: Array<[unknown, string, string]> = [
            [new Error("boom"), "neutral", "trelix review did not complete"],
            [
                Object.assign(new Error("t"), {
                    killed: true,
                    signal: "SIGTERM",
                }),
                "timed_out",
                "trelix review timed out",
            ],
            [
                Object.assign(new Error("x"), { code: 3 }),
                "neutral",
                "trelix review did not run",
            ],
        ];

        for (const [error, conclusion, title] of errors) {
            const { octokit, calls } = fakeOctokit();
            vi.mocked(sanitizeCheckOutput).mockImplementationOnce((output) => ({
                title: `S[${output.title}]`,
                summary: `S[${output.summary}]`,
            }));

            await postReviewFailureCheckRun(
                octokit,
                "o",
                "r",
                "deadbeef",
                error,
            );

            expect(calls).toHaveLength(1);
            expect(calls[0].conclusion).toBe(conclusion);
            const output = calls[0].output as {
                title: string;
                summary: string;
            };
            expect(output.title).toBe(`S[${title}]`);
            expect(output.summary).toMatch(/^S\[trelix .+\.\]$/);
            expect(Object.keys(output).sort()).toEqual(["summary", "title"]);
        }
    });

    it("posts the fixed wording unchanged, since it is already clean", async () => {
        const { octokit, calls } = fakeOctokit();

        await postReviewFailureCheckRun(
            octokit,
            "o",
            "r",
            "deadbeef",
            Object.assign(new Error("x"), { code: 3 }),
        );

        expect(calls[0].output).toEqual({
            title: "trelix review did not run",
            summary:
                "trelix could not review this PR: no usable LLM is configured for this trelix instance, or every LLM call failed. No code was reviewed, so this is not a clean result.",
        });
    });
});

describe("runReview, end to end", () => {
    let binDir: string;
    let originalPath: string | undefined;

    beforeEach(() => {
        binDir = mkdtempSync(join(tmpdir(), "trelix-wiring-bin-"));
        originalPath = process.env.PATH;
        process.env.PATH = `${binDir}:${originalPath}`;
    });

    afterEach(() => {
        process.env.PATH = originalPath;
        rmSync(binDir, { recursive: true, force: true });
    });

    /** A `trelix` whose `review` prints `reviewOutput` (a file's content) and exits `reviewExit`. */
    function installFakeTrelix(reviewOutput: string, reviewExit = 0): void {
        const outputFile = join(binDir, "review.json");
        writeFileSync(outputFile, reviewOutput);
        const shim = join(binDir, "trelix");
        writeFileSync(
            shim,
            [
                "#!/bin/sh",
                'if [ "$1" = "index" ]; then exit 0; fi',
                `cat "${outputFile}"`,
                `exit ${reviewExit}`,
                "",
            ].join("\n"),
        );
        chmodSync(shim, 0o755);
    }

    function config(): AppConfig {
        const { privateKey } = generateKeyPairSync("rsa", {
            modulusLength: 2048,
            publicKeyEncoding: { type: "spki", format: "pem" },
            privateKeyEncoding: { type: "pkcs1", format: "pem" },
        });
        return { appId: "1", privateKey, webhookSecret: "fake", port: 0 };
    }

    async function review(octokit: Octokit): Promise<void> {
        await runReview(
            config(),
            { owner: "o", repo: "r", prNumber: 1, installationId: 9 },
            {
                checkoutPullRequest: vi.fn(async () => ({
                    path: ".",
                    cleanup: async () => {},
                })),
                request: vi.fn(async () => ({
                    data: {
                        token: "test-k",
                        expires_at: "2099-01-01T00:00:00Z",
                        permissions: {},
                        repository_selection: "all",
                    },
                })) as never,
                octokit,
            },
        );
    }

    it("sanitises what a hostile model reply puts in the Check", async () => {
        installFakeTrelix(JSON.stringify([HOSTILE_FINDING]));
        const { octokit, calls } = fakeOctokit();

        await review(octokit);

        expect(calls).toHaveLength(1);
        expect(calls[0].output).toEqual({
            title: "trelix found 1 issue(s)",
            summary: "trelix reviewed the PR and found 1 issue(s).",
            annotations: [SANITISED_HOSTILE_ANNOTATION],
        });
    });

    it("posts the failure Check through the sanitiser too", async () => {
        installFakeTrelix("[]", 3);
        const { octokit, calls } = fakeOctokit();
        vi.mocked(sanitizeCheckOutput).mockImplementationOnce((output) => ({
            title: `S[${output.title}]`,
            summary: `S[${output.summary}]`,
        }));

        await expect(review(octokit)).rejects.toBeDefined();

        expect(calls).toHaveLength(1);
        expect(calls[0].conclusion).toBe("neutral");
        expect((calls[0].output as { title: string }).title).toBe(
            "S[trelix review did not run]",
        );
    });
});

describe("no way around the sanitiser", () => {
    const srcDir = fileURLToPath(new URL("../src", import.meta.url));

    function sourceFiles(dir: string): string[] {
        return readdirSync(dir, { withFileTypes: true }).flatMap((entry) =>
            entry.isDirectory()
                ? sourceFiles(join(dir, entry.name))
                : entry.name.endsWith(".ts")
                  ? [join(dir, entry.name)]
                  : [],
        );
    }

    it("creates Check runs in exactly one place, and that place sanitises", () => {
        const sources = sourceFiles(srcDir).map((file) => ({
            name: relative(srcDir, file),
            text: readFileSync(file, "utf8"),
        }));

        const posting = sources.filter(({ text }) =>
            /\bchecks\.[a-z]+\(|check-runs|\.request\(/.test(
                text.replace(/^\s*(\/\/|\*|\/\*).*$/gm, ""),
            ),
        );

        // auth.ts mints tokens with a request of its own; it never posts a Check.
        const checkPosters = posting.filter(({ name }) => name !== "auth.ts");
        expect(checkPosters.map(({ name }) => name)).toEqual([
            "review-runner.ts",
        ]);
        const runner = checkPosters[0].text;
        expect(runner.match(/\bchecks\.[a-z]+\(/g)).toEqual(["checks.create("]);
        expect(runner).toMatch(
            /checks\.create\(\{[^}]*\boutput: sanitizeCheckOutput\(output\),/,
        );
    });
});
