import { generateKeyPairSync } from "node:crypto";
import {
    writeFileSync,
    mkdtempSync,
    chmodSync,
    readdirSync,
    readFileSync,
    rmSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Octokit } from "@octokit/rest";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
    toAnnotations,
    indexRepository,
    runReviewCli,
    postReviewFailureCheckRun,
    ReviewFinding,
    RunReviewOptions,
} from "../src/review-runner.js";
import { AppConfig } from "../src/config.js";
import { Workspace } from "../src/repo-checkout.js";
import {
    loadConclusionCases,
    outcomeTextFor,
    stdoutTextFor,
} from "./support/conclusion-cases.js";
import { runReviewForTest } from "./support/run-review.js";

describe("toAnnotations", () => {
    it("maps ERROR/WARN/INFO severities to failure/warning/notice", () => {
        const findings: ReviewFinding[] = [
            { file: "a.py", lines: "1-1", severity: "ERROR", comment: "bug" },
            { file: "b.py", lines: "2-2", severity: "WARN", comment: "smell" },
            { file: "c.py", lines: "3-3", severity: "INFO", comment: "note" },
        ];

        const annotations = toAnnotations(findings);

        expect(annotations[0].annotation_level).toBe("failure");
        expect(annotations[1].annotation_level).toBe("warning");
        expect(annotations[2].annotation_level).toBe("notice");
    });

    it("parses the 'start-end' lines field into start_line/end_line", () => {
        const findings: ReviewFinding[] = [
            { file: "a.py", lines: "10-25", severity: "INFO", comment: "x" },
        ];

        const [annotation] = toAnnotations(findings);

        expect(annotation.start_line).toBe(10);
        expect(annotation.end_line).toBe(25);
    });

    it("caps annotations at the given limit (GitHub's per-check-run annotation limit)", () => {
        const findings: ReviewFinding[] = Array.from(
            { length: 60 },
            (_, i) => ({
                file: `f${i}.py`,
                lines: "1-1",
                severity: "INFO" as const,
                comment: "x",
            }),
        );

        expect(toAnnotations(findings, 50)).toHaveLength(50);
        expect(toAnnotations(findings)).toHaveLength(50); // default limit
    });

    it("carries the comment through as the annotation message and a fixed title", () => {
        const findings: ReviewFinding[] = [
            {
                file: "a.py",
                lines: "1-1",
                severity: "ERROR",
                comment: "specific finding text",
            },
        ];

        const [annotation] = toAnnotations(findings);

        expect(annotation.message).toBe("specific finding text");
        expect(annotation.title).toBe("trelix review");
        expect(annotation.path).toBe("a.py");
    });

    it("returns an empty list for an empty findings array", () => {
        expect(toAnnotations([])).toEqual([]);
    });
});

describe("postReviewFailureCheckRun", () => {
    /** Same octokit.hook.wrap request-interceptor pattern used throughout this file. */
    function fakeOctokitCapturingChecksCreate() {
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
            throw new Error(
                `unexpected octokit request in test: ${options.method} ${options.url}`,
            );
        });
        return { octokit, calls };
    }

    it("posts conclusion 'timed_out' for a Node timeout-kill error (killed + SIGTERM)", async () => {
        const { octokit, calls } = fakeOctokitCapturingChecksCreate();
        const timeoutErr = Object.assign(new Error("command timed out"), {
            killed: true,
            signal: "SIGTERM",
        });

        await postReviewFailureCheckRun(
            octokit,
            "o",
            "r",
            "deadbeef",
            timeoutErr,
        );

        expect(calls).toHaveLength(1);
        expect(calls[0]).toMatchObject({ conclusion: "timed_out" });
    });

    it("posts conclusion 'neutral' for a non-timeout error (CLI crash, bad JSON, etc.)", async () => {
        const { octokit, calls } = fakeOctokitCapturingChecksCreate();

        await postReviewFailureCheckRun(
            octokit,
            "o",
            "r",
            "deadbeef",
            new Error("exit code 1"),
        );

        expect(calls).toHaveLength(1);
        expect(calls[0]).toMatchObject({ conclusion: "neutral" });
    });

    // `trelix review` exits 3 (REVIEW_NOT_RUN_EXIT_CODE in src/trelix/cli/main.py)
    // when no LLM is usable or no hunk got a usable review and none kept a finding.
    // The check must say so instead of the generic "failed to run to completion",
    // and must never read as a pass.
    it("explains the 'review did not run' exit code (3) instead of a generic failure", async () => {
        const { octokit, calls } = fakeOctokitCapturingChecksCreate();
        const notRunErr = Object.assign(new Error("Command failed"), { code: 3 });

        await postReviewFailureCheckRun(
            octokit,
            "o",
            "r",
            "deadbeef",
            notRunErr,
        );

        expect(calls).toHaveLength(1);
        expect(calls[0]).toMatchObject({ conclusion: "neutral" });
        const output = calls[0].output as { title: string; summary: string };
        expect(output.title).toMatch(/did not run/i);
        expect(output.summary).toMatch(/LLM/);
        expect(output.title).not.toMatch(/0 issue/);
    });

    it("keeps the generic wording for other non-zero exit codes", async () => {
        const { octokit, calls } = fakeOctokitCapturingChecksCreate();
        const crashErr = Object.assign(new Error("Command failed"), { code: 1 });

        await postReviewFailureCheckRun(
            octokit,
            "o",
            "r",
            "deadbeef",
            crashErr,
        );

        const output = calls[0].output as { title: string; summary: string };
        expect(output.title).toBe("trelix review did not complete");
    });

    it("still posts conclusion 'neutral' for a non-Error thrown value", async () => {
        const { octokit, calls } = fakeOctokitCapturingChecksCreate();

        await postReviewFailureCheckRun(
            octokit,
            "o",
            "r",
            "deadbeef",
            "a string, not an Error",
        );

        expect(calls).toHaveLength(1);
        expect(calls[0]).toMatchObject({ conclusion: "neutral" });
    });
});

describe("runReviewCli timeout", () => {
    let binDir: string;
    let originalPath: string | undefined;

    beforeEach(() => {
        binDir = mkdtempSync(join(tmpdir(), "trelix-fake-bin-"));
        // A real slow "trelix" binary — not a mock of execFile's timeout
        // mechanism, an actual subprocess that actually sleeps, so this test
        // exercises the real kill-on-timeout path end to end.
        const shim = join(binDir, "trelix");
        writeFileSync(shim, "#!/bin/sh\nsleep 5\necho '[]'\n");
        chmodSync(shim, 0o755);
        originalPath = process.env.PATH;
        process.env.PATH = `${binDir}:${originalPath}`;
    });

    afterEach(() => {
        process.env.PATH = originalPath;
        rmSync(binDir, { recursive: true, force: true });
    });

    it("kills a hung `trelix review` subprocess once the timeout elapses", async () => {
        const request = { owner: "o", repo: "r", prNumber: 1 };

        await expect(
            runReviewCli(request, ".", "fake-token", 200),
        ).rejects.toMatchObject({
            killed: true,
            signal: "SIGTERM",
        });
    }, 10_000);

    it("does not time out a fast-returning subprocess", async () => {
        const shim = join(binDir, "trelix");
        writeFileSync(shim, "#!/bin/sh\necho '[]'\n");
        chmodSync(shim, 0o755);
        const request = { owner: "o", repo: "r", prNumber: 1 };

        await expect(
            runReviewCli(request, ".", "fake-token", 5000),
        ).resolves.toEqual([]);
    });

    // Regression test — found live: `trelix review --pr` fetches the PR
    // diff from GitHub's own API (src/trelix/cli/main.py's review()
    // command) and hard-requires a GITHUB_TOKEN env var to do so, exactly
    // like .github/workflows/trelix-review.yml's own
    // `GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}` step does. Every real
    // webhook failed with "GITHUB_TOKEN environment variable is required
    // for --pr." because runReviewCli never passed the installation token
    // it's given through to the subprocess's environment at all.
    it("passes the installation token through as GITHUB_TOKEN in the subprocess environment", async () => {
        const shim = join(binDir, "trelix");
        writeFileSync(
            shim,
            [
                "#!/bin/sh",
                'if [ "$GITHUB_TOKEN" = "canary-installation-token" ]; then',
                "  echo '[]'",
                "else",
                "  echo 'GITHUB_TOKEN missing or wrong' >&2",
                "  exit 1",
                "fi",
                "",
            ].join("\n"),
        );
        chmodSync(shim, 0o755);
        const request = { owner: "o", repo: "r", prNumber: 1 };

        await expect(
            runReviewCli(request, ".", "canary-installation-token", 5000),
        ).resolves.toEqual([]);
    });
});

describe("runReview orchestration", () => {
    let binDir: string;
    let originalPath: string | undefined;

    beforeEach(() => {
        binDir = mkdtempSync(join(tmpdir(), "trelix-orchestration-bin-"));
        // A real "trelix" binary faking both subcommands runReview shells
        // out to: `index` (deliberately fails, to exercise the tolerant
        // fallback) and `review --json` (succeeds with one finding).
        const shim = join(binDir, "trelix");
        writeFileSync(
            shim,
            [
                "#!/bin/sh",
                'if [ "$1" = "index" ]; then',
                '  echo "simulated index failure" >&2',
                "  exit 1",
                "fi",
                'echo \'[{"file":"a.py","lines":"1-1","severity":"INFO","comment":"note"}]\'',
                "",
            ].join("\n"),
        );
        chmodSync(shim, 0o755);
        originalPath = process.env.PATH;
        process.env.PATH = `${binDir}:${originalPath}`;
    });

    afterEach(() => {
        process.env.PATH = originalPath;
        rmSync(binDir, { recursive: true, force: true });
    });

    function makeConfig(): AppConfig {
        const { privateKey } = generateKeyPairSync("rsa", {
            modulusLength: 2048,
            publicKeyEncoding: { type: "spki", format: "pem" },
            privateKeyEncoding: { type: "pkcs1", format: "pem" },
        });
        return { appId: "1", privateKey, webhookSecret: "fake", port: 0 };
    }

    /** Same RequestInterface-fake pattern as auth.test.ts's fakeRequest, for the installation-token mint. */
    function fakeAuthRequest(token: string) {
        return vi.fn(async () => ({
            data: {
                token,
                expires_at: "2099-01-01T00:00:00Z",
                permissions: {},
                repository_selection: "all",
            },
        })) as never;
    }

    /** Fakes @octokit/rest's own HTTP transport via its documented `hook.wrap("request", ...)` extension point. */
    function fakeOctokit(
        headSha: string,
        opts: {
            checksCreateShouldThrow?: boolean;
            checksCreateCalls?: Array<Record<string, unknown>>;
        } = {},
    ) {
        const octokit = new Octokit({});
        octokit.hook.wrap("request", async (_request, options) => {
            if (
                options.method === "GET" &&
                options.url === "/repos/{owner}/{repo}/pulls/{pull_number}"
            ) {
                return {
                    status: 200,
                    url: "",
                    headers: {},
                    data: { head: { sha: headSha } },
                };
            }
            if (
                options.method === "POST" &&
                options.url === "/repos/{owner}/{repo}/check-runs"
            ) {
                opts.checksCreateCalls?.push(options);
                if (opts.checksCreateShouldThrow) {
                    throw new Error("checks.create failed (simulated)");
                }
                return { status: 201, url: "", headers: {}, data: {} };
            }
            throw new Error(
                `unexpected octokit request in test: ${options.method} ${options.url}`,
            );
        });
        return octokit;
    }

    function fakeWorkspace(): {
        workspace: Workspace;
        cleanup: ReturnType<typeof vi.fn>;
    } {
        const cleanup = vi.fn(async () => {});
        return { workspace: { path: ".", cleanup }, cleanup };
    }

    it("does not block the review when `trelix index` fails (tolerant-failure fallback)", async () => {
        const config = makeConfig();
        const { workspace, cleanup } = fakeWorkspace();
        const checkoutPullRequest: RunReviewOptions["checkoutPullRequest"] =
            vi.fn(async () => workspace);
        const octokit = fakeOctokit("deadbeef");

        const findings = await runReviewForTest(
            config,
            { owner: "o", repo: "r", prNumber: 1, installationId: 999 },
            {
                checkoutPullRequest,
                request: fakeAuthRequest("ghs_faketoken"),
                octokit,
            },
        );

        expect(findings).toEqual([
            { file: "a.py", lines: "1-1", severity: "INFO", comment: "note" },
        ]);
        expect(cleanup).toHaveBeenCalledTimes(1);
    });

    it("runs workspace.cleanup() exactly once even when posting the Check run throws", async () => {
        const config = makeConfig();
        const { workspace, cleanup } = fakeWorkspace();
        const checkoutPullRequest: RunReviewOptions["checkoutPullRequest"] =
            vi.fn(async () => workspace);
        const octokit = fakeOctokit("deadbeef", {
            checksCreateShouldThrow: true,
        });

        await expect(
            runReviewForTest(
                config,
                { owner: "o", repo: "r", prNumber: 1, installationId: 999 },
                {
                    checkoutPullRequest,
                    request: fakeAuthRequest("ghs_faketoken"),
                    octokit,
                },
            ),
        ).rejects.toThrow("checks.create failed (simulated)");

        expect(cleanup).toHaveBeenCalledTimes(1);
    });

    // Regression test — found live by the E2E production-dry-run workflow:
    // when `trelix review --pr` itself fails (timeout, crash, bad JSON),
    // the exception used to propagate straight past postCheckRun, so
    // webhook.ts's caller only console.error'd it -- the PR was left with
    // NO Check run at all, not even a failure one.
    it("posts a failure Check run (not silence) when the trelix review subprocess itself fails", async () => {
        const config = makeConfig();
        const { workspace, cleanup } = fakeWorkspace();
        const checkoutPullRequest: RunReviewOptions["checkoutPullRequest"] =
            vi.fn(async () => workspace);
        const checksCreateCalls: Array<Record<string, unknown>> = [];
        const octokit = fakeOctokit("deadbeef", { checksCreateCalls });

        const shim = join(binDir, "trelix");
        writeFileSync(
            shim,
            [
                "#!/bin/sh",
                'if [ "$1" = "index" ]; then',
                '  echo "simulated index failure" >&2',
                "  exit 1",
                "fi",
                'echo "simulated trelix review crash" >&2',
                "exit 1",
                "",
            ].join("\n"),
        );
        chmodSync(shim, 0o755);

        await expect(
            runReviewForTest(
                config,
                { owner: "o", repo: "r", prNumber: 1, installationId: 999 },
                {
                    checkoutPullRequest,
                    request: fakeAuthRequest("ghs_faketoken"),
                    octokit,
                },
            ),
        ).rejects.toThrow();

        expect(checksCreateCalls).toHaveLength(1);
        expect(checksCreateCalls[0]).toMatchObject({
            name: "trelix Code Review",
            status: "completed",
            conclusion: "neutral",
        });
        expect(cleanup).toHaveBeenCalledTimes(1);
    });

    it("posts a neutral 'did not run' Check run (never success) when the CLI exits 3", async () => {
        const config = makeConfig();
        const { workspace, cleanup } = fakeWorkspace();
        const checkoutPullRequest: RunReviewOptions["checkoutPullRequest"] =
            vi.fn(async () => workspace);
        const checksCreateCalls: Array<Record<string, unknown>> = [];
        const octokit = fakeOctokit("deadbeef", { checksCreateCalls });

        const shim = join(binDir, "trelix");
        writeFileSync(
            shim,
            [
                "#!/bin/sh",
                'if [ "$1" = "index" ]; then',
                "  exit 0",
                "fi",
                'echo "[]"',
                'echo "LLM is not configured" >&2',
                "exit 3",
                "",
            ].join("\n"),
        );
        chmodSync(shim, 0o755);

        await expect(
            runReviewForTest(
                config,
                { owner: "o", repo: "r", prNumber: 1, installationId: 999 },
                {
                    checkoutPullRequest,
                    request: fakeAuthRequest("ghs_faketoken"),
                    octokit,
                },
            ),
        ).rejects.toThrow();

        expect(checksCreateCalls).toHaveLength(1);
        expect(checksCreateCalls[0]).toMatchObject({
            conclusion: "neutral",
        });
        const output = checksCreateCalls[0].output as { title: string };
        expect(output.title).toMatch(/did not run/i);
        expect(cleanup).toHaveBeenCalledTimes(1);
    });

    it("passes the minted installation token through to checkoutPullRequest", async () => {
        const config = makeConfig();
        const { workspace } = fakeWorkspace();
        const checkoutPullRequest: RunReviewOptions["checkoutPullRequest"] =
            vi.fn(async () => workspace);
        const octokit = fakeOctokit("deadbeef");

        await runReviewForTest(
            config,
            { owner: "o", repo: "r", prNumber: 1, installationId: 999 },
            {
                checkoutPullRequest,
                request: fakeAuthRequest("ghs_faketoken"),
                octokit,
            },
        );

        expect(checkoutPullRequest).toHaveBeenCalledWith(
            "ghs_faketoken",
            expect.objectContaining({ owner: "o", repo: "r", prNumber: 1 }),
        );
    });

    // Regression test — found live: this is the exact end-to-end path that
    // failed against real GitHub with "GITHUB_TOKEN environment variable is
    // required for --pr." runReview mints the installation token and passes
    // it to checkoutPullRequest, but was never forwarding that same token to
    // the `trelix review --pr ...` subprocess's environment, which requires
    // GITHUB_TOKEN to fetch the PR diff via GitHub's API (see
    // src/trelix/cli/main.py's review() command, and the identical
    // GITHUB_TOKEN pattern .github/workflows/trelix-review.yml already uses).
    it("forwards the minted installation token to the trelix review subprocess as GITHUB_TOKEN", async () => {
        const config = makeConfig();
        const { workspace, cleanup } = fakeWorkspace();
        const checkoutPullRequest: RunReviewOptions["checkoutPullRequest"] =
            vi.fn(async () => workspace);
        const octokit = fakeOctokit("deadbeef");
        const installationTokenForCli = "installation-token-for-cli-env";

        const shim = join(binDir, "trelix");
        writeFileSync(
            shim,
            [
                "#!/bin/sh",
                'if [ "$1" = "index" ]; then',
                '  echo "simulated index failure" >&2',
                "  exit 1",
                "fi",
                `if [ "$GITHUB_TOKEN" != "${installationTokenForCli}" ]; then`,
                '  echo "GITHUB_TOKEN missing or wrong in orchestration" >&2',
                "  exit 1",
                "fi",
                "echo '[]'",
                "",
            ].join("\n"),
        );
        chmodSync(shim, 0o755);

        const findings = await runReviewForTest(
            config,
            { owner: "o", repo: "r", prNumber: 1, installationId: 999 },
            {
                checkoutPullRequest,
                request: fakeAuthRequest(installationTokenForCli),
                octokit,
            },
        );

        expect(findings).toEqual([]);
        expect(cleanup).toHaveBeenCalledTimes(1);
    });
});

// How a review ends decides what the Check says. Each case runs the whole of runReview
// against a real `trelix` stand-in (a shell script that exits with the row's status, prints
// the row's findings and, like the real one, writes the outcome record to the path in
// TRELIX_REVIEW_OUTCOME_FILE). The rows are the ones the workflow is tested with.
describe("runReview, by how the review ended", () => {
    const CASES = loadConclusionCases();
    const APP_ROWS = CASES.cases.filter((row) =>
        (row.applies_to ?? ["workflow", "app"]).includes("app"),
    );

    let binDir: string;
    let dumpDir: string;
    let outcomeBase: string;
    let workspaceDir: string;
    let originalPath: string | undefined;

    beforeEach(() => {
        binDir = mkdtempSync(join(tmpdir(), "trelix-ended-bin-"));
        dumpDir = mkdtempSync(join(tmpdir(), "trelix-ended-dump-"));
        outcomeBase = mkdtempSync(join(tmpdir(), "trelix-ended-outcomes-"));
        workspaceDir = mkdtempSync(join(tmpdir(), "trelix-ended-workspace-"));
        originalPath = process.env.PATH;
        process.env.PATH = `${binDir}:${originalPath}`;
    });

    afterEach(() => {
        process.env.PATH = originalPath;
        for (const dir of [binDir, dumpDir, outcomeBase, workspaceDir]) {
            rmSync(dir, { recursive: true, force: true });
        }
        vi.restoreAllMocks();
    });

    /** A `trelix` that exits `exitCode` after printing `stdout` and writing `outcomeText` (none: no file) where it was told to. */
    function installTrelix(
        exitCode: number,
        stdout: string,
        outcomeText: string | null = null,
    ): void {
        writeFileSync(join(binDir, "stdout.txt"), stdout);
        if (outcomeText !== null) {
            writeFileSync(join(binDir, "outcome-source.json"), outcomeText);
        }
        const shim = join(binDir, "trelix");
        writeFileSync(
            shim,
            [
                "#!/bin/sh",
                'if [ "$1" = "index" ]; then',
                `  echo "\${TRELIX_REVIEW_OUTCOME_FILE-unset}" > "${dumpDir}/index.outcome-path"`,
                "  exit 0",
                "fi",
                `echo "\${TRELIX_REVIEW_OUTCOME_FILE-unset}" > "${dumpDir}/review.outcome-path"`,
                // An empty destination would make BSD cp write into the current directory.
                `if [ -n "$TRELIX_REVIEW_OUTCOME_FILE" ] && [ -f "${binDir}/outcome-source.json" ]; then cp "${binDir}/outcome-source.json" "$TRELIX_REVIEW_OUTCOME_FILE"; fi`,
                `cat "${binDir}/stdout.txt"`,
                `exit ${exitCode}`,
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

    function fakeOctokit(checksCreateShouldThrow = false) {
        const calls: Array<Record<string, unknown>> = [];
        const octokit = new Octokit({});
        octokit.hook.wrap("request", async (_request, options) => {
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
            if (
                options.method === "POST" &&
                options.url === "/repos/{owner}/{repo}/check-runs"
            ) {
                calls.push(options);
                if (checksCreateShouldThrow) {
                    throw new Error("checks.create failed (simulated)");
                }
                return { status: 201, url: "", headers: {}, data: {} };
            }
            throw new Error(
                `unexpected octokit request in test: ${options.method} ${options.url}`,
            );
        });
        return { octokit, calls };
    }

    /** Runs the whole review; resolves to what runReview returned or the error it threw. */
    async function review(octokit: Octokit, cleanupError?: Error) {
        const cleanup = vi.fn(async () => {
            if (cleanupError !== undefined) {
                throw cleanupError;
            }
        });
        const checkoutPullRequest: RunReviewOptions["checkoutPullRequest"] =
            vi.fn(async () => ({ path: workspaceDir, cleanup }));
        const outcome = await runReviewForTest(
            config(),
            { owner: "o", repo: "r", prNumber: 1, installationId: 999 },
            {
                checkoutPullRequest,
                request: vi.fn(async () => ({
                    data: {
                        token: "test-k",
                        expires_at: "2099-01-01T00:00:00Z",
                        permissions: {},
                        repository_selection: "all",
                    },
                })) as never,
                octokit,
                outcomeBaseDir: outcomeBase,
            },
        ).then(
            (findings) => ({ findings }),
            (error: unknown) => ({ error }),
        );
        return { ...outcome, cleanup };
    }

    const output = (call: Record<string, unknown>) =>
        call.output as {
            title: string;
            summary: string;
            annotations?: Array<Record<string, unknown>>;
        };

    it("has a row for every way a review ends", () => {
        expect(APP_ROWS.length).toBeGreaterThanOrEqual(14);
    });

    it.each(APP_ROWS.map((row) => [row.id, row] as const))(
        "the shared row %s gets its conclusion and title",
        async (_id, row) => {
            installTrelix(
                Number(row.exit_code),
                stdoutTextFor(row),
                outcomeTextFor(row, CASES),
            );
            const { octokit, calls } = fakeOctokit();

            const result = await review(octokit);

            expect(calls).toHaveLength(1);
            expect(calls[0]).toMatchObject({
                head_sha: "deadbeef",
                status: "completed",
                conclusion: row.conclusion,
            });
            expect(output(calls[0]).title).toBe(row.title);
            // Exit 4 is a review that ran and returns its findings, readable or not. Exit 0
            // returns them only if stdout is a findings array; when it is not, the error is
            // thrown after the Check is posted, as it is for any other status.
            const returned =
                row.exit_code === "4" ||
                (row.exit_code === "0" && row.stdout_text === undefined);
            expect("findings" in result).toBe(returned);
            expect("error" in result).toBe(!returned);
            expect(result.cleanup).toHaveBeenCalledTimes(1);
            expect(readdirSync(outcomeBase)).toEqual([]);
        },
    );

    describe("a review that covered only part of the diff (exit 4)", () => {
        const WARN = {
            file: "src/a.py",
            lines: "3-4",
            severity: "WARN",
            comment: "smell",
        };

        it("posts the findings as annotations and what was left unreviewed, and returns the findings", async () => {
            installTrelix(
                4,
                JSON.stringify([WARN]),
                JSON.stringify(CASES.valid_outcome),
            );
            const { octokit, calls } = fakeOctokit();

            const result = await review(octokit);

            expect(result).toMatchObject({ findings: [WARN] });
            expect(calls).toHaveLength(1);
            expect(calls[0].conclusion).toBe("neutral");
            expect(output(calls[0])).toEqual({
                title: "trelix review incomplete",
                summary: [
                    "trelix reviewed only part of this PR: 3 of 5 hunks were reviewed and 2 were not. This is not a clean result.",
                    "",
                    "1 issue(s) found in the hunks that were reviewed.",
                    "",
                    "Hunks that were not reviewed:",
                    "- `src/a.py:10` (truncated)",
                    "- `src/b.py:20` (refused)",
                ].join("\n"),
                annotations: [
                    {
                        path: "src/a.py",
                        start_line: 3,
                        end_line: 4,
                        annotation_level: "warning",
                        message: "smell",
                        title: "trelix review",
                    },
                ],
            });
        });

        it("fails the check for an ERROR finding", async () => {
            installTrelix(
                4,
                JSON.stringify([{ ...WARN, severity: "ERROR" }]),
                JSON.stringify(CASES.valid_outcome),
            );
            const { octokit, calls } = fakeOctokit();

            await review(octokit);

            expect(calls[0].conclusion).toBe("failure");
            expect(output(calls[0]).title).toBe("trelix review incomplete");
        });

        it("says the extent is unknown when the record is missing, and does not call the review clean", async () => {
            vi.spyOn(console, "warn").mockImplementation(() => {});
            installTrelix(4, JSON.stringify([WARN]));
            const { octokit, calls } = fakeOctokit();

            await review(octokit);

            expect(calls[0].conclusion).toBe("neutral");
            const { summary } = output(calls[0]);
            expect(summary).toContain(
                "how much was left unreviewed is unknown",
            );
            expect(summary).not.toContain("Hunks that were not reviewed");
            expect(output(calls[0]).annotations).toHaveLength(1);
        });

        it("says the findings could not be read when stdout is not a JSON array, and still posts", async () => {
            vi.spyOn(console, "warn").mockImplementation(() => {});
            installTrelix(4, "not json", JSON.stringify(CASES.valid_outcome));
            const { octokit, calls } = fakeOctokit();

            const result = await review(octokit);

            expect(result).toMatchObject({ findings: [] });
            expect(calls[0].conclusion).toBe("neutral");
            expect(output(calls[0]).summary).toContain(
                "The list of findings could not be read, so none are shown.",
            );
            expect(output(calls[0]).annotations).toEqual([]);
        });

        it("says the findings could not be read when stdout is a JSON object", async () => {
            vi.spyOn(console, "warn").mockImplementation(() => {});
            installTrelix(
                4,
                '{"severity":"ERROR"}',
                JSON.stringify(CASES.valid_outcome),
            );
            const { octokit, calls } = fakeOctokit();

            await review(octokit);

            expect(calls[0].conclusion).toBe("neutral");
            expect(output(calls[0]).summary).toContain("could not be read");
        });

        it("says the findings could not be read when stdout is an array with something that is not an object in it", async () => {
            vi.spyOn(console, "warn").mockImplementation(() => {});
            // toAnnotations would throw on the null, and no Check would be posted at all.
            installTrelix(
                4,
                '[{"file":"a.py","lines":"1-1","severity":"ERROR","comment":"x"},null]',
                JSON.stringify(CASES.valid_outcome),
            );
            const { octokit, calls } = fakeOctokit();

            const result = await review(octokit);

            expect(result).toMatchObject({ findings: [] });
            expect(calls).toHaveLength(1);
            expect(calls[0].conclusion).toBe("neutral");
            expect(output(calls[0]).title).toBe("trelix review incomplete");
            expect(output(calls[0]).summary).toContain(
                "The list of findings could not be read, so none are shown.",
            );
            expect(output(calls[0]).annotations).toEqual([]);
        });

        it("returns normally instead of throwing, and does not use the did-not-run wording", async () => {
            installTrelix(4, "[]", JSON.stringify(CASES.valid_outcome));
            const { octokit, calls } = fakeOctokit();

            const result = await review(octokit);

            expect("error" in result).toBe(false);
            expect(output(calls[0]).title).not.toMatch(
                /did not (run|complete)/,
            );
        });

        it("still removes the outcome directory and the workspace when posting the Check fails", async () => {
            installTrelix(4, "[]", JSON.stringify(CASES.valid_outcome));
            const { octokit } = fakeOctokit(true);

            const result = await review(octokit);

            expect(result).toMatchObject({ error: expect.any(Error) });
            expect(result.cleanup).toHaveBeenCalledTimes(1);
            expect(readdirSync(outcomeBase)).toEqual([]);
        });
    });

    it("still removes the outcome directory when the workspace cannot be cleaned up", async () => {
        installTrelix(4, "[]", JSON.stringify(CASES.valid_outcome));
        const { octokit } = fakeOctokit();

        const result = await review(octokit, new Error("workspace stuck"));

        expect(result).toMatchObject({ error: new Error("workspace stuck") });
        expect(readdirSync(outcomeBase)).toEqual([]);
    });

    describe("the findings on stdout are read for exit 0 and exit 4 only", () => {
        const ERROR = JSON.stringify([
            {
                file: "src/a.py",
                lines: "1-1",
                severity: "ERROR",
                comment: "bug",
            },
        ]);

        it.each([1, 2, 3, 5, 124, 255])(
            "exit %i: a neutral check with no annotations, whatever stdout holds",
            async (exitCode) => {
                installTrelix(
                    exitCode,
                    ERROR,
                    JSON.stringify(CASES.valid_outcome),
                );
                const { octokit, calls } = fakeOctokit();

                const result = await review(octokit);

                expect(result).toMatchObject({ error: expect.anything() });
                expect(calls[0].conclusion).toBe("neutral");
                expect(output(calls[0]).annotations).toBeUndefined();
                expect(output(calls[0]).summary).not.toContain(
                    "Hunks that were not reviewed",
                );
                expect(output(calls[0]).title).toMatch(
                    /^trelix review did not (run|complete)$/,
                );
            },
        );
    });

    describe("the outcome record's path", () => {
        it("goes to the review child alone, in a private directory outside the checkout", async () => {
            installTrelix(0, "[]");
            const { octokit } = fakeOctokit();

            await review(octokit);

            const given = readFileSync(
                join(dumpDir, "review.outcome-path"),
                "utf8",
            ).trim();
            expect(
                given.startsWith(`${outcomeBase}/trelix-review-outcome-`),
            ).toBe(true);
            expect(given.endsWith("/outcome.json")).toBe(true);
            expect(given.startsWith(workspaceDir)).toBe(false);
            expect(
                readFileSync(
                    join(dumpDir, "index.outcome-path"),
                    "utf8",
                ).trim(),
            ).toBe("unset");
        });

        it("is a different fresh directory for each review, removed when it ends", async () => {
            installTrelix(0, "[]");
            const first = fakeOctokit();
            const second = fakeOctokit();

            await review(first.octokit);
            const pathOne = readFileSync(
                join(dumpDir, "review.outcome-path"),
                "utf8",
            );
            await review(second.octokit);
            const pathTwo = readFileSync(
                join(dumpDir, "review.outcome-path"),
                "utf8",
            );

            expect(pathOne).not.toBe(pathTwo);
            expect(readdirSync(outcomeBase)).toEqual([]);
        });

        it("ignores a record left in the checkout: only the path it chose is read", async () => {
            vi.spyOn(console, "warn").mockImplementation(() => {});
            installTrelix(4, "[]");
            writeFileSync(
                join(workspaceDir, "outcome.json"),
                JSON.stringify(CASES.valid_outcome),
            );
            const { octokit, calls } = fakeOctokit();

            await review(octokit);

            expect(output(calls[0]).summary).toContain("is unknown");
        });
    });
});

// Secrets the App's own process holds (its credentials, the platform's tokens) and
// a name nobody listed. Neither `trelix index` nor `trelix review` may see any of
// them: both children run over content an outside PR author controls.
// child-env.test.ts covers the allow-list itself; the tests below prove the two
// spawn sites really use it.
const APP_CREDENTIAL_NAMES = [
    "GITHUB_APP_PRIVATE_KEY",
    "GITHUB_WEBHOOK_SECRET",
    "RAILWAY_TOKEN",
    "SOME_UNLISTED_SECRET",
] as const;

function fakeAppCredentials(): Record<string, string> {
    return Object.fromEntries(
        APP_CREDENTIAL_NAMES.map((name) => [name, `fake-${name}-for-tests`]),
    );
}

// What an operator configures so the LLM works: these must still reach the child.
const PROVIDER_SETTINGS: Record<string, string> = {
    AZURE_ENDPOINT: "https://llm.example.invalid",
    TRELIX_LLM_PROVIDER: "azure",
};

describe("trelix child process environment", () => {
    const MANAGED_KEYS = [
        ...APP_CREDENTIAL_NAMES,
        ...Object.keys(PROVIDER_SETTINGS),
        "GITHUB_TOKEN",
        "TRELIX_WALKER_FOLLOW_SYMLINKS",
        "TRELIX_TEST_ENV_DUMP_DIR",
        "PATH",
    ];
    let binDir: string;
    let dumpDir: string;
    let savedEnv: Record<string, string | undefined>;

    beforeEach(() => {
        savedEnv = Object.fromEntries(
            MANAGED_KEYS.map((key) => [key, process.env[key]]),
        );
        binDir = mkdtempSync(join(tmpdir(), "trelix-env-bin-"));
        dumpDir = mkdtempSync(join(tmpdir(), "trelix-env-dump-"));
        // A real `trelix` stand-in that records the environment it was
        // actually spawned with, keyed by subcommand ($1).
        const shim = join(binDir, "trelix");
        writeFileSync(
            shim,
            "#!/bin/sh\nenv > \"$TRELIX_TEST_ENV_DUMP_DIR/$1.env\"\necho '[]'\n",
        );
        chmodSync(shim, 0o755);

        Object.assign(process.env, fakeAppCredentials(), PROVIDER_SETTINGS);
        process.env.PATH = `${binDir}:${savedEnv.PATH}`;
        process.env.TRELIX_TEST_ENV_DUMP_DIR = dumpDir;
        process.env.TRELIX_WALKER_FOLLOW_SYMLINKS = "true";
        delete process.env.GITHUB_TOKEN;
    });

    afterEach(() => {
        for (const key of MANAGED_KEYS) {
            if (savedEnv[key] === undefined) {
                delete process.env[key];
            } else {
                process.env[key] = savedEnv[key];
            }
        }
        rmSync(binDir, { recursive: true, force: true });
        rmSync(dumpDir, { recursive: true, force: true });
    });

    /** Parses the `env` dump a child wrote into a key -> value map. */
    function readChildEnv(subcommand: string): Record<string, string> {
        const lines = readFileSync(
            join(dumpDir, `${subcommand}.env`),
            "utf8",
        ).split("\n");
        const entries: Array<[string, string]> = [];
        for (const line of lines) {
            const at = line.indexOf("=");
            if (at > 0) entries.push([line.slice(0, at), line.slice(at + 1)]);
        }
        return Object.fromEntries(entries);
    }

    const request = { owner: "o", repo: "r", prNumber: 1 };

    it("hides the App's secrets from `trelix index`, keeps the provider settings and forces the symlink flag off", async () => {
        await indexRepository(".", 5000);

        const env = readChildEnv("index");
        for (const name of APP_CREDENTIAL_NAMES) {
            expect(env).not.toHaveProperty(name);
        }
        expect(env).toMatchObject(PROVIDER_SETTINGS);
        expect(env.TRELIX_WALKER_FOLLOW_SYMLINKS).toBe("false");
    });

    it("hides the App's secrets from `trelix review`, keeps the provider settings and forces the symlink flag off", async () => {
        await runReviewCli(request, ".", "fake-installation-token", 5000);

        const env = readChildEnv("review");
        for (const name of APP_CREDENTIAL_NAMES) {
            expect(env).not.toHaveProperty(name);
        }
        expect(env).toMatchObject(PROVIDER_SETTINGS);
        expect(env.TRELIX_WALKER_FOLLOW_SYMLINKS).toBe("false");
    });

    it("gives the installation token to `trelix review` only, never to `trelix index`", async () => {
        await indexRepository(".", 5000);
        await runReviewCli(request, ".", "fake-installation-token", 5000);

        expect(readChildEnv("index")).not.toHaveProperty("GITHUB_TOKEN");
        expect(readChildEnv("review").GITHUB_TOKEN).toBe(
            "fake-installation-token",
        );
    });

    it("leaves the server's own process.env unchanged", async () => {
        const before = { ...process.env };

        await indexRepository(".", 5000);
        await runReviewCli(request, ".", "fake-installation-token", 5000);

        expect({ ...process.env }).toEqual(before);
        expect(process.env.TRELIX_WALKER_FOLLOW_SYMLINKS).toBe("true");
        for (const name of APP_CREDENTIAL_NAMES) {
            expect(process.env[name]).toBe(`fake-${name}-for-tests`);
        }
    });
});
