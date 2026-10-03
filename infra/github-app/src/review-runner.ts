import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { Octokit } from "@octokit/rest";
import type { RequestInterface } from "@octokit/types";
import { AppConfig } from "./config.js";
import { getInstallationToken } from "./auth.js";
import { buildIndexChildEnv, buildReviewChildEnv } from "./child-env.js";
import { checkoutPullRequest as defaultCheckoutPullRequest } from "./repo-checkout.js";
import {
    buildIncompleteSummary,
    createOutcomeLocation,
    isRecord,
    OutcomeLocation,
    readOutcomeRecord,
    REVIEW_INCOMPLETE_EXIT_CODE,
    REVIEW_NOT_RUN_EXIT_CODE,
    reviewConclusion,
    ReviewOutcomeRecord,
} from "./review-outcome.js";
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
 * The findings in what `trelix review --json` printed. The same parse for a
 * review that finished (exit 0) and for one that covered only part of the diff
 * (exit 4, whose stdout is still the findings array). Throws when it is not a
 * JSON array of objects: a `null` or a number among the findings would make
 * toAnnotations throw later, after the point where a Check can still be posted
 * about it. The workflow's readFindings applies the same checks.
 */
export function parseFindings(stdout: string): ReviewFinding[] {
    const parsed: unknown = JSON.parse(stdout);
    if (!Array.isArray(parsed)) {
        throw new Error("trelix review did not print a JSON array");
    }
    if (!parsed.every(isRecord)) {
        throw new Error(
            "trelix review printed a finding that is not an object",
        );
    }
    // Each is an object; the fields of a finding are not checked here.
    return parsed as unknown as ReviewFinding[];
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
 *
 * `outcomeFile`, when given, is handed to the child as
 * TRELIX_REVIEW_OUTCOME_FILE: where it writes the record of which hunks it
 * did not review. Resolves with the findings on exit 0 and rejects on any
 * other exit status, including 4 (a review that covered only part of the
 * diff), where the findings are in the error's `stdout`: see runReview.
 */
export async function runReviewCli(
    request: ReviewRequest,
    repoPath: string,
    token: string,
    timeoutMs: number = REVIEW_TIMEOUT_MS,
    outcomeFile?: string,
): Promise<ReviewFinding[]> {
    const prRef = `${request.owner}/${request.repo}#${request.prNumber}`;
    const { stdout } = await execFileAsync(
        "trelix",
        ["review", repoPath, "--pr", prRef, "--json"],
        {
            timeout: timeoutMs,
            env: buildReviewChildEnv(token, process.env, outcomeFile),
        },
    );
    return parseFindings(stdout);
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
        reviewConclusion(0, findings),
        {
            title: `trelix found ${findings.length} issue(s)`,
            summary,
            annotations,
        },
    );
}

/** The numeric exit status an `execFile` rejection carries in `code`, else null. */
function exitCodeOf(err: unknown): number | null {
    if (typeof err !== "object" || err === null) {
        return null;
    }
    const { code } = err as { code?: unknown };
    return typeof code === "number" ? code : null;
}

/**
 * Posts the Check run for a review that covered only part of the diff
 * (`trelix review` exit 4): the findings it did produce as annotations, a
 * summary with the counts and the first unreviewed hunks, and a conclusion of
 * neutral, or failure if any finding is an ERROR. `findings` null means the
 * findings could not be read; `outcome` null means the record of what was
 * covered is missing or not valid. Either way the summary says so: neither is
 * read as "nothing was left out".
 */
export async function postIncompleteCheckRun(
    octokit: Octokit,
    owner: string,
    repo: string,
    headSha: string,
    findings: ReviewFinding[] | null,
    outcome: ReviewOutcomeRecord | null,
): Promise<void> {
    const shown = findings ?? [];
    const annotations = toAnnotations(shown);
    await createCheckRun(
        octokit,
        owner,
        repo,
        headSha,
        reviewConclusion(REVIEW_INCOMPLETE_EXIT_CODE, findings),
        {
            title: "trelix review incomplete",
            summary: buildIncompleteSummary(
                findings === null ? null : findings.length,
                annotations.length,
                outcome,
            ),
            annotations,
        },
    );
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

    // execFile rejects with the child's numeric exit status in `code`.
    const exitCode = exitCodeOf(err);
    const notRun = exitCode === REVIEW_NOT_RUN_EXIT_CODE;

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
            "trelix could not review this PR: no usable LLM is configured for this trelix instance, or no hunk got a usable review and none kept a finding. No code was reviewed, so this is not a clean result.";
    }

    await createCheckRun(
        octokit,
        owner,
        repo,
        headSha,
        timedOut ? "timed_out" : reviewConclusion(exitCode, []),
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
    /**
     * Tests only: the directory the per-review outcome directory is created
     * in, so a test can see that it is created and removed. Defaults to the
     * OS temp directory.
     */
    outcomeBaseDir?: string;
}

/** The findings `trelix review` printed before it exited 4, or null when its stdout is missing or not a JSON array. */
function findingsOfIncompleteReview(err: unknown): ReviewFinding[] | null {
    const { stdout } = err as { stdout?: unknown };
    if (typeof stdout !== "string") {
        return null;
    }
    try {
        return parseFindings(stdout);
    } catch (parseErr) {
        console.warn(
            "[review-runner] the findings of an incomplete review could not be parsed:",
            parseErr,
        );
        return null;
    }
}

/**
 * Runs the review in an already checked-out workspace and posts the Check run
 * that matches how it ended: the findings on exit 0, the findings with what was
 * left unreviewed on exit 4 (REVIEW_INCOMPLETE_EXIT_CODE; returns normally,
 * because the review did run), and a neutral or timed-out Check on anything
 * else, after which the error is rethrown.
 */
async function reviewAndPost(
    octokit: Octokit,
    request: ReviewRequest,
    headSha: string,
    workspacePath: string,
    token: string,
    outcomeFile: string,
): Promise<ReviewFinding[]> {
    let findings: ReviewFinding[];
    try {
        findings = await runReviewCli(
            request,
            workspacePath,
            token,
            undefined,
            outcomeFile,
        );
    } catch (err) {
        if (exitCodeOf(err) === REVIEW_INCOMPLETE_EXIT_CODE) {
            // Only exit 4 has findings in `stdout`: a crash, a timeout or exit 3
            // prints nothing that is a review.
            const partial = findingsOfIncompleteReview(err);
            await postIncompleteCheckRun(
                octokit,
                request.owner,
                request.repo,
                headSha,
                partial,
                await readOutcomeRecord(outcomeFile),
            );
            return partial ?? [];
        }
        // A timed-out or crashed CLI must still leave a visible signal
        // on the PR -- without this, the exception below propagated
        // straight past postCheckRun, and webhook.ts's caller only
        // console.error'd it, leaving the PR with no Check run at all.
        await postReviewFailureCheckRun(
            octokit,
            request.owner,
            request.repo,
            headSha,
            err,
        );
        throw err;
    }
    await postCheckRun(octokit, request.owner, request.repo, headSha, findings);
    return findings;
}

/**
 * End-to-end: mint an installation token, clone the PR head into a fresh
 * workspace, index it, run the CLI review against it, and post the
 * findings as a Check run — cleaning up the workspace (and the private
 * directory the review writes its outcome record to) unconditionally.
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
    let outcomeLocation: OutcomeLocation | undefined;
    try {
        // Outside the checkout, so the pull request cannot touch the file.
        outcomeLocation = await createOutcomeLocation(options.outcomeBaseDir);
        await indexRepository(workspace.path);
        return await reviewAndPost(
            octokit,
            request,
            pull.head.sha,
            workspace.path,
            token,
            outcomeLocation.file,
        );
    } finally {
        try {
            await workspace.cleanup();
        } finally {
            await outcomeLocation?.cleanup();
        }
    }
}
