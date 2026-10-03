import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { Octokit } from "@octokit/rest";
import type { RequestInterface } from "@octokit/types";
import { AppConfig } from "./config.js";
import { getInstallationToken } from "./auth.js";
import { buildIndexChildEnv, buildReviewChildEnv } from "./child-env.js";
import { checkoutPullRequest as defaultCheckoutPullRequest } from "./repo-checkout.js";
import {
    CheckAnnotation,
    CheckOutput,
    MAX_ANNOTATIONS,
    sanitizeAnnotation,
    sanitizeCheckOutput,
} from "./sanitize.js";

// Defined next to the sanitiser that produces it, so sanitize.ts does not
// import from this module; re-exported because callers know it from here.
export type { CheckAnnotation };

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

/**
 * Maps findings to Check annotations, sanitised (see sanitize.ts). A finding
 * whose file path is not acceptable (absolute, a `..` segment, markup) gets no
 * annotation, so the result can be shorter than `findings`; it holds at most
 * `limit` annotations, counted after the unacceptable ones are skipped.
 */
export function toAnnotations(
    findings: ReviewFinding[],
    limit = MAX_ANNOTATIONS,
): CheckAnnotation[] {
    const annotations: CheckAnnotation[] = [];
    for (const f of findings) {
        if (annotations.length >= limit) {
            break;
        }
        const [startLine, endLine] = f.lines.split("-").map(Number);
        const annotation = sanitizeAnnotation({
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
        });
        if (annotation !== null) {
            annotations.push(annotation);
        }
    }
    return annotations;
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
        {
            timeout: timeoutMs,
            env: buildReviewChildEnv(token),
        },
    );
    return JSON.parse(stdout) as ReviewFinding[];
}

const CHECK_RUN_NAME = "trelix Code Review";

type CheckConclusion = "success" | "failure" | "neutral" | "timed_out";

/**
 * The only place this service creates a Check run. The text comes from an
 * LLM reading attacker-written pull requests, so `output` always goes
 * through sanitizeCheckOutput on the way out: a poster cannot forget it.
 */
async function createCheckRun(
    octokit: Octokit,
    owner: string,
    repo: string,
    headSha: string,
    conclusion: CheckConclusion,
    output: CheckOutput,
): Promise<void> {
    await octokit.rest.checks.create({
        owner,
        repo,
        name: CHECK_RUN_NAME,
        head_sha: headSha,
        status: "completed",
        conclusion,
        output: sanitizeCheckOutput(output),
    });
}

/**
 * Posts findings as a completed Check run with inline annotations,
 * mirroring trelix-review.yml's github-script step's github.rest.checks.create
 * call (any 'ERROR' finding -> overall 'failure', else 'success').
 *
 * The verdict and the count come from the findings, not from the
 * annotations: a finding can go without an annotation (over GitHub's limit of
 * 50 per request, or an unacceptable file path) and must not turn a failure
 * into a success or vanish from the count. The summary says how many were
 * left out.
 */
export async function postCheckRun(
    octokit: Octokit,
    owner: string,
    repo: string,
    headSha: string,
    findings: ReviewFinding[],
): Promise<void> {
    const annotations = toAnnotations(findings);
    const withoutAnnotation = findings.length - annotations.length;
    let summary = `trelix reviewed the PR and found ${findings.length} issue(s).`;
    if (withoutAnnotation > 0) {
        summary += ` ${withoutAnnotation} of them could not be shown as inline annotations (GitHub allows 50 per check, or the file path was not usable).`;
    }
    await createCheckRun(
        octokit,
        owner,
        repo,
        headSha,
        findings.some((f) => f.severity === "ERROR") ? "failure" : "success",
        {
            title: `trelix found ${findings.length} issue(s)`,
            summary,
            annotations,
        },
    );
}

/**
 * Exit status of `trelix review` when it could not review at all (no usable
 * LLM, or every hunk's LLM call failed) — REVIEW_NOT_RUN_EXIT_CODE in
 * src/trelix/cli/main.py. Keep the two in step.
 */
const REVIEW_NOT_RUN_EXIT_CODE = 3;

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

    // execFile rejects with the child's numeric exit status in `code`.
    const notRun =
        typeof err === "object" &&
        err !== null &&
        (err as { code?: unknown }).code === REVIEW_NOT_RUN_EXIT_CODE;

    let title = "trelix review did not complete";
    let summary =
        "trelix review failed to run to completion. No findings were produced for this PR.";
    if (timedOut) {
        title = "trelix review timed out";
        summary =
            "trelix review did not finish within the time limit and was stopped. No findings were produced for this PR.";
    } else if (notRun) {
        title = "trelix review did not run";
        summary =
            "trelix could not review this PR: no usable LLM is configured for this trelix instance, or every LLM call failed. No code was reviewed, so this is not a clean result.";
    }

    await createCheckRun(
        octokit,
        owner,
        repo,
        headSha,
        timedOut ? "timed_out" : "neutral",
        { title, summary },
    );
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
            env: buildIndexChildEnv(),
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
