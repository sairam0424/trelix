import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { Octokit } from "@octokit/rest";
import type { RequestInterface } from "@octokit/types";
import { AppConfig } from "./config.js";
import { getInstallationToken } from "./auth.js";
import { checkoutPullRequest as defaultCheckoutPullRequest } from "./repo-checkout.js";

const execFileAsync = promisify(execFile);

// A slow/hung `trelix review` (LLM synthesis latency, a huge diff, a stuck
// index) would otherwise tie up this process indefinitely per webhook
// delivery — Node kills the child and execFileAsync rejects once this
// elapses.
const REVIEW_TIMEOUT_MS = 5 * 60 * 1000;

// Indexing a huge/pathological repo shouldn't hang the whole review either
// — same ceiling as the review step itself.
const INDEX_TIMEOUT_MS = 5 * 60 * 1000;

export interface ReviewRequest {
    owner: string;
    repo: string;
    prNumber: number;
    installationId?: number;
}

/** Matches trelix review --pr ... --json's real output shape exactly — see src/trelix/cli/main.py. */
export interface ReviewFinding {
    file: string;
    lines: string; // "start-end", 1-indexed
    severity: "ERROR" | "WARN" | "INFO";
    comment: string;
}

export interface CheckAnnotation {
    path: string;
    start_line: number;
    end_line: number;
    annotation_level: "failure" | "warning" | "notice";
    message: string;
    title: string;
}

export function toAnnotations(
    findings: ReviewFinding[],
    limit = 50,
): CheckAnnotation[] {
    return findings.slice(0, limit).map((f) => {
        const [startLine, endLine] = f.lines.split("-").map(Number);
        return {
            path: f.file,
            start_line: startLine || 1,
            end_line: endLine || startLine || 1,
            annotation_level:
                f.severity === "ERROR"
                    ? "failure"
                    : f.severity === "WARN"
                      ? "warning"
                      : "notice",
            message: f.comment,
            title: "trelix review",
        };
    });
}

/**
 * Runs `trelix review --pr owner/repo#N --json` and returns the parsed
 * findings. Requires PR #83's fix (stdout carries ONLY the JSON array;
 * status/progress messages go to stderr) — this function reads stdout
 * exclusively and would break against the pre-#83 CLI.
 *
 * `--pr` mode fetches the PR diff from GitHub's own API rather than the
 * local clone (see src/trelix/cli/main.py's review() command) and hard-
 * requires a GITHUB_TOKEN env var to do so — exactly like
 * .github/workflows/trelix-review.yml's own
 * `GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}` step. `token` here is the
 * same installation token runReview already minted for checkoutPullRequest
 * and the Checks API — found live: every real webhook failed with
 * "GITHUB_TOKEN environment variable is required for --pr." until this
 * was forwarded through.
 */
export async function runReviewCli(
    request: ReviewRequest,
    repoPath: string,
    token: string,
    timeoutMs: number = REVIEW_TIMEOUT_MS,
): Promise<ReviewFinding[]> {
    const prRef = `${request.owner}/${request.repo}#${request.prNumber}`;
    const { stdout } = await execFileAsync(
        "trelix",
        ["review", repoPath, "--pr", prRef, "--json"],
        { timeout: timeoutMs, env: { ...process.env, GITHUB_TOKEN: token } },
    );
    return JSON.parse(stdout) as ReviewFinding[];
}

/**
 * Posts findings as a completed Check run with inline annotations,
 * mirroring trelix-review.yml's github-script step's github.rest.checks.create
 * call exactly (same conclusion logic: any 'failure'-level annotation ->
 * overall 'failure', else 'success').
 */
export async function postCheckRun(
    octokit: Octokit,
    owner: string,
    repo: string,
    headSha: string,
    findings: ReviewFinding[],
): Promise<void> {
    const annotations = toAnnotations(findings);
    await octokit.rest.checks.create({
        owner,
        repo,
        name: "trelix Code Review",
        head_sha: headSha,
        status: "completed",
        conclusion: annotations.some((a) => a.annotation_level === "failure")
            ? "failure"
            : "success",
        output: {
            title: `trelix found ${annotations.length} issue(s)`,
            summary: `trelix reviewed the PR and found ${annotations.length} issue(s).`,
            annotations,
        },
    });
}

/**
 * Posts a completed Check run recording that the review itself never ran
 * to completion — distinct from postCheckRun, which posts real findings.
 * Without this, a `runReviewCli` failure (timeout, CLI crash, bad JSON)
 * left the PR with NO Check run at all: the caller only saw a server-side
 * console.error, invisible to anyone looking at the PR. `conclusion` is
 * "timed_out" for Node's timeout-triggered kill (matches
 * runReviewCli timeout's `{ killed: true, signal: 'SIGTERM' }` shape) and
 * "neutral" otherwise — a broken reviewer isn't the same claim as "this
 * code has failures".
 */
export async function postReviewFailureCheckRun(
    octokit: Octokit,
    owner: string,
    repo: string,
    headSha: string,
    err: unknown,
): Promise<void> {
    const timedOut =
        typeof err === "object" &&
        err !== null &&
        (err as { killed?: boolean }).killed === true &&
        (err as { signal?: string }).signal === "SIGTERM";

    await octokit.rest.checks.create({
        owner,
        repo,
        name: "trelix Code Review",
        head_sha: headSha,
        status: "completed",
        conclusion: timedOut ? "timed_out" : "neutral",
        output: {
            title: timedOut
                ? "trelix review timed out"
                : "trelix review did not complete",
            summary: timedOut
                ? "trelix review did not finish within the time limit and was stopped. No findings were produced for this PR."
                : "trelix review failed to run to completion. No findings were produced for this PR.",
        },
    });
}

/**
 * Runs `trelix index <repoPath>`, mirroring trelix-review.yml's own
 * tolerant `if ! trelix index .; then ::warning ...; fi` pattern: an
 * indexing failure (network-restricted host, OOM) degrades findings to
 * structural-only per the CLI's own documented fallback rather than
 * blocking the review outright.
 */
export async function indexRepository(
    repoPath: string,
    timeoutMs: number = INDEX_TIMEOUT_MS,
): Promise<void> {
    try {
        await execFileAsync("trelix", ["index", repoPath], {
            timeout: timeoutMs,
        });
    } catch (err) {
        console.warn(
            `[review-runner] trelix index failed for ${repoPath} — review findings will be structural-only:`,
            err,
        );
    }
}

export interface RunReviewOptions {
    /** Injectable — tests substitute a fake to avoid a real git clone. */
    checkoutPullRequest?: typeof defaultCheckoutPullRequest;
    /**
     * Injectable fake HTTP transport for the installation-token mint —
     * same RequestInterface-fake pattern getInstallationToken's own
     * optional third param already uses (see auth.test.ts).
     */
    request?: RequestInterface;
    /**
     * Injectable Octokit instance for the PR-fetch/Check-run calls — tests
     * substitute one with an `octokit.hook.wrap("request", ...)`
     * interceptor registered so those calls never hit the real GitHub
     * API. Defaults to a real Octokit authenticated with the minted
     * installation token.
     */
    octokit?: Octokit;
}

/**
 * End-to-end: mint an installation token, clone the PR head into a fresh
 * workspace, index it, run the CLI review against it, and post the
 * findings as a Check run — cleaning up the workspace unconditionally.
 * Requires request.installationId (the webhook payload's
 * `installation.id` — always present for App-installed webhook
 * deliveries).
 */
export async function runReview(
    config: AppConfig,
    request: ReviewRequest,
    options: RunReviewOptions = {},
): Promise<ReviewFinding[]> {
    if (request.installationId === undefined) {
        throw new Error(
            "runReview requires an installationId to authenticate the Checks API call",
        );
    }

    const checkoutPullRequest =
        options.checkoutPullRequest ?? defaultCheckoutPullRequest;

    const token = await getInstallationToken(
        config,
        request.installationId,
        options.request,
    );
    const octokit = options.octokit ?? new Octokit({ auth: token });

    const { data: pull } = await octokit.rest.pulls.get({
        owner: request.owner,
        repo: request.repo,
        pull_number: request.prNumber,
    });

    const workspace = await checkoutPullRequest(token, request);
    try {
        await indexRepository(workspace.path);
        let findings: ReviewFinding[];
        try {
            findings = await runReviewCli(request, workspace.path, token);
        } catch (err) {
            // A timed-out or crashed CLI must still leave a visible signal
            // on the PR -- without this, the exception below propagated
            // straight past postCheckRun, and webhook.ts's caller only
            // console.error'd it, leaving the PR with no Check run at all.
            await postReviewFailureCheckRun(
                octokit,
                request.owner,
                request.repo,
                pull.head.sha,
                err,
            );
            throw err;
        }
        await postCheckRun(
            octokit,
            request.owner,
            request.repo,
            pull.head.sha,
            findings,
        );
        return findings;
    } finally {
        await workspace.cleanup();
    }
}
