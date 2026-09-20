import { generateKeyPairSync } from "node:crypto";
import { writeFileSync, mkdtempSync, chmodSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Octokit } from "@octokit/rest";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
    toAnnotations,
    runReviewCli,
    runReview,
    ReviewFinding,
    RunReviewOptions,
} from "../src/review-runner.js";
import { AppConfig } from "../src/config.js";
import { Workspace } from "../src/repo-checkout.js";

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
        opts: { checksCreateShouldThrow?: boolean } = {},
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

        const findings = await runReview(
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
            runReview(
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

    it("passes the minted installation token through to checkoutPullRequest", async () => {
        const config = makeConfig();
        const { workspace } = fakeWorkspace();
        const checkoutPullRequest: RunReviewOptions["checkoutPullRequest"] =
            vi.fn(async () => workspace);
        const octokit = fakeOctokit("deadbeef");

        await runReview(
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

        const findings = await runReview(
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
