/**
 * runReview with the three purpose-scoped installation tokens, over a fake GitHub.
 *
 * Nothing here replaces @octokit/auth-app or the default Octokit: `fetch` is the only
 * fake, so the requests are the ones the real libraries would send. Each placeholder
 * answer is chosen from the permissions the request for it asked for, so a token
 * minted with the wrong permissions is a different value from the one its consumer
 * should have got.
 */
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
import { join } from "node:path";
import { inspect } from "node:util";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { AppConfig } from "../src/config.js";
import type { RunReviewOptions } from "../src/review-runner.js";
import {
    HEAD_SHA,
    NEWER_SHA,
    reviewRequest,
} from "./support/review-fixtures.js";
import { runReviewForTest } from "./support/run-review.js";

// Placeholders, one per purpose. A log-leak test looks for these exact strings.
const CHECKOUT_CANARY = "canary-for-checkout";
const REVIEW_CANARY = "canary-for-review";
const POSTER_CANARY = "canary-for-poster";
const ALL_CANARIES = [CHECKOUT_CANARY, REVIEW_CANARY, POSTER_CANARY];

const { privateKey } = generateKeyPairSync("rsa", {
    modulusLength: 2048,
    publicKeyEncoding: { type: "spki", format: "pem" },
    privateKeyEncoding: { type: "pkcs1", format: "pem" },
});

/** A new object per call: @octokit/auth-app's token cache lives per AppConfig. */
function makeConfig(): AppConfig {
    return { appId: "1", privateKey, webhookSecret: "fake", port: 0 };
}

interface SeenRequest {
    method: string;
    url: string;
    authorization: string | null;
    body: Record<string, unknown> | undefined;
}

function answerFor(permissions: Record<string, string>): string {
    if ("contents" in permissions) return CHECKOUT_CANARY;
    if ("pull_requests" in permissions) return REVIEW_CANARY;
    if ("checks" in permissions) return POSTER_CANARY;
    return "canary-for-nothing-known";
}

function json(status: number, data: unknown): Response {
    return new Response(JSON.stringify(data), {
        status,
        headers: { "content-type": "application/json" },
    });
}

/** A fake GitHub behind `fetch`: it answers the token exchange and the Check, and records everything. */
function fakeGitHub(checkRunStatus = 201): SeenRequest[] {
    const seen: SeenRequest[] = [];
    vi.stubGlobal(
        "fetch",
        vi.fn(async (url: string, init: RequestInit = {}) => {
            const body =
                init.body === undefined
                    ? undefined
                    : (JSON.parse(String(init.body)) as Record<
                          string,
                          unknown
                      >);
            seen.push({
                method: init.method ?? "GET",
                url,
                authorization: new Headers(init.headers).get("authorization"),
                body,
            });
            if (url.endsWith("/access_tokens")) {
                const permissions = body?.permissions as Record<string, string>;
                return json(201, {
                    token: answerFor(permissions),
                    expires_at: "2099-01-01T00:00:00Z",
                    permissions,
                    repository_selection: "selected",
                });
            }
            if (url.endsWith("/repos/o/r/check-runs")) {
                return json(
                    checkRunStatus,
                    checkRunStatus < 300 ? {} : { message: "refused" },
                );
            }
            return json(404, { message: `unexpected request: ${url}` });
        }),
    );
    return seen;
}

const mintRequests = (seen: SeenRequest[]) =>
    seen.filter((request) => request.url.endsWith("/access_tokens"));
const checkRuns = (seen: SeenRequest[]) =>
    seen.filter((request) => request.url.endsWith("/check-runs"));

/** The permissions of a mint request, as one sortable string. */
const permissionNames = (request: SeenRequest) =>
    Object.keys(request.body?.permissions as object)
        .sort()
        .join();

describe("runReview with purpose-scoped installation tokens", () => {
    let binDir: string;
    let dumpDir: string;
    let outcomeBase: string;
    let workspaceDir: string;
    let originalPath: string | undefined;

    beforeEach(() => {
        binDir = mkdtempSync(join(tmpdir(), "trelix-scoped-bin-"));
        dumpDir = mkdtempSync(join(tmpdir(), "trelix-scoped-dump-"));
        outcomeBase = mkdtempSync(join(tmpdir(), "trelix-scoped-outcomes-"));
        workspaceDir = mkdtempSync(join(tmpdir(), "trelix-scoped-workspace-"));
        originalPath = process.env.PATH;
        process.env.PATH = `${binDir}:${originalPath}`;
    });

    afterEach(() => {
        process.env.PATH = originalPath;
        vi.unstubAllGlobals();
        vi.restoreAllMocks();
        for (const dir of [binDir, dumpDir, outcomeBase, workspaceDir]) {
            rmSync(dir, { recursive: true, force: true });
        }
    });

    /** A `trelix` that notes it ran, and (for `review`) the GITHUB_TOKEN it was given. */
    function installTrelix(reviewExit = 0): void {
        const shim = join(binDir, "trelix");
        writeFileSync(
            shim,
            [
                "#!/bin/sh",
                'if [ "$1" = "index" ]; then',
                `  echo ran > "${dumpDir}/index-ran"`,
                "  exit 0",
                "fi",
                `printf '%s' "$GITHUB_TOKEN" > "${dumpDir}/review-token"`,
                `if [ ${reviewExit} -ne 0 ]; then echo "review broke" >&2; exit ${reviewExit}; fi`,
                "echo '[]'",
                "",
            ].join("\n"),
        );
        chmodSync(shim, 0o755);
    }

    /** A checkout double at `headSha` that records what it was called with. */
    function fakeCheckout(headSha: string) {
        const cleanup = vi.fn(async () => {});
        const checkoutPullRequest = vi.fn(
            async (_credential: string, _target: unknown) => ({
                path: workspaceDir,
                headSha,
                cleanup,
            }),
        );
        return { checkoutPullRequest, cleanup };
    }

    const review = (
        checkoutPullRequest: ReturnType<
            typeof fakeCheckout
        >["checkoutPullRequest"],
        onNoVerdict?: () => void,
    ) =>
        runReviewForTest(makeConfig(), reviewRequest(), {
            checkoutPullRequest:
                checkoutPullRequest as RunReviewOptions["checkoutPullRequest"],
            outcomeBaseDir: outcomeBase,
            onNoVerdict,
        });

    it("hands the checkout, the review child and the Check call each the token minted for it, and nothing else is requested", async () => {
        installTrelix();
        const seen = fakeGitHub();
        const { checkoutPullRequest, cleanup } = fakeCheckout(HEAD_SHA);
        const onNoVerdict = vi.fn();

        const findings = await review(checkoutPullRequest, onNoVerdict);

        expect(findings).toEqual([]);
        // A review that reached a verdict Check has nothing to hand back.
        expect(onNoVerdict).not.toHaveBeenCalled();
        expect(checkoutPullRequest).toHaveBeenCalledTimes(1);
        expect(checkoutPullRequest.mock.calls[0][0]).toBe(CHECKOUT_CANARY);
        expect(readFileSync(join(dumpDir, "review-token"), "utf8")).toBe(
            REVIEW_CANARY,
        );
        expect(checkRuns(seen)).toHaveLength(1);
        expect(checkRuns(seen)[0].authorization).toBe(`token ${POSTER_CANARY}`);
        expect(checkRuns(seen)[0].body).toMatchObject({
            head_sha: "0123456789abcdef0123456789abcdef01234567",
            name: "trelix Code Review",
            conclusion: "success",
        });
        // Three token exchanges and one Check: no `pulls.get`, no second write.
        expect(seen.map((r) => `${r.method} ${r.url}`).sort()).toEqual([
            "POST https://api.github.com/app/installations/999/access_tokens",
            "POST https://api.github.com/app/installations/999/access_tokens",
            "POST https://api.github.com/app/installations/999/access_tokens",
            "POST https://api.github.com/repos/o/r/check-runs",
        ]);
        expect(cleanup).toHaveBeenCalledTimes(1);
    });

    it("asks for each token with the repository of the pull request and only its permissions", async () => {
        installTrelix();
        const seen = fakeGitHub();

        await review(fakeCheckout(HEAD_SHA).checkoutPullRequest);

        const asked = mintRequests(seen)
            .sort((a, b) =>
                permissionNames(a).localeCompare(permissionNames(b)),
            )
            .map((request) => request.body);
        expect(asked).toStrictEqual([
            {
                repository_ids: [4242],
                permissions: { checks: "write", metadata: "read" },
            },
            {
                repository_ids: [4242],
                permissions: { contents: "read", metadata: "read" },
            },
            {
                repository_ids: [4242],
                permissions: { pull_requests: "read", metadata: "read" },
            },
        ]);
    });

    it.each([
        ["a newer push moved the ref", NEWER_SHA],
        [
            "the ref is behind the delivery",
            "0000000000000000000000000000000000000001",
        ],
    ])(
        "skips the review, posts nothing and leaves nothing behind when %s",
        async (_name, checkoutSha) => {
            installTrelix();
            const seen = fakeGitHub();
            const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
            const { checkoutPullRequest, cleanup } = fakeCheckout(checkoutSha);
            const onNoVerdict = vi.fn();

            const findings = await review(checkoutPullRequest, onNoVerdict);

            expect(findings).toEqual([]);
            // Nothing was posted, so the commit has no verdict: it may be reviewed again.
            expect(onNoVerdict).toHaveBeenCalledTimes(1);
            expect(checkRuns(seen)).toHaveLength(0);
            // Neither `trelix index` nor `trelix review` ran, and no outcome directory was made.
            expect(readdirSync(dumpDir)).toEqual([]);
            expect(readdirSync(outcomeBase)).toEqual([]);
            expect(cleanup).toHaveBeenCalledTimes(1);
            expect(warn).toHaveBeenCalledTimes(1);
            expect(warn).toHaveBeenCalledWith(
                `[review-runner] skipping o/r#1: checkout is at ${checkoutSha}, delivery is for 0123456789abcdef0123456789abcdef01234567; nothing posted`,
            );
        },
    );

    it("does not start a review for a repository id GitHub could not scope a token to", async () => {
        installTrelix();
        const seen = fakeGitHub();
        const { checkoutPullRequest } = fakeCheckout(HEAD_SHA);

        await expect(
            runReviewForTest(makeConfig(), reviewRequest({ repositoryId: 0 }), {
                checkoutPullRequest:
                    checkoutPullRequest as RunReviewOptions["checkoutPullRequest"],
                outcomeBaseDir: outcomeBase,
            }),
        ).rejects.toThrow(
            "an installation token needs the numeric id of one repository",
        );

        expect(seen).toHaveLength(0);
        expect(checkoutPullRequest).not.toHaveBeenCalled();
    });

    describe("no installation token or App JWT reaches a log line", () => {
        /** Everything console printed, formatted the way console formats it. */
        function captureConsole(): string[] {
            const lines: string[] = [];
            const sink = (...args: unknown[]) => {
                lines.push(
                    args
                        .map((arg) =>
                            typeof arg === "string"
                                ? arg
                                : inspect(arg, { depth: 8 }),
                        )
                        .join(" "),
                );
            };
            for (const method of [
                "log",
                "info",
                "warn",
                "error",
                "debug",
            ] as const) {
                vi.spyOn(console, method).mockImplementation(sink);
            }
            return lines;
        }

        const appJwtsIn = (seen: SeenRequest[]) =>
            mintRequests(seen).map((request) =>
                String(request.authorization).replace(/^bearer /, ""),
            );

        it("the capture sees a line that holds a canary (so an empty result means something)", () => {
            const lines = captureConsole();

            console.warn(`leaked ${REVIEW_CANARY}`);
            console.error(new Error(`leaked ${POSTER_CANARY}`));

            expect(lines.join("\n")).toContain(REVIEW_CANARY);
            expect(lines.join("\n")).toContain(POSTER_CANARY);
        });

        it.each([
            ["a clean review", HEAD_SHA, 0, 201],
            ["a checkout at another commit", NEWER_SHA, 0, 201],
            ["a review that fails", HEAD_SHA, 1, 201],
            ["a Check the API refuses", HEAD_SHA, 0, 403],
        ])(
            "after %s",
            async (_name, checkoutSha, reviewExit, checkRunStatus) => {
                installTrelix(reviewExit);
                const seen = fakeGitHub(checkRunStatus);
                const lines = captureConsole();

                // What the webhook handler prints for a rejection.
                await review(
                    fakeCheckout(checkoutSha).checkoutPullRequest,
                ).catch((err: unknown) =>
                    console.error("[webhook] review failed:", err),
                );

                const printed = lines.join("\n");
                expect(mintRequests(seen)).toHaveLength(3);
                for (const secret of [...ALL_CANARIES, ...appJwtsIn(seen)]) {
                    expect(secret.length).toBeGreaterThan(10);
                    expect(printed).not.toContain(secret);
                }
            },
        );
    });
});
