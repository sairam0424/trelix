import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { Octokit } from "@octokit/rest";
import type { RequestInterface } from "@octokit/types";
import { AppConfig } from "./config.js";
import { getPurposeTokens } from "./auth.js";
import { buildIndexChildEnv, buildReviewChildEnv } from "./child-env.js";
import {
    exitCodeOf,
    postCheckRun,
    postIncompleteCheckRun,
    postReviewFailureCheckRun,
    ReviewFinding,
    toAnnotations,
} from "./check-posting.js";
import { checkoutPullRequest as defaultCheckoutPullRequest } from "./repo-checkout.js";
import {
    createOutcomeLocation,
    isRecord,
    OutcomeLocation,
    readOutcomeRecord,
    REVIEW_INCOMPLETE_EXIT_CODE,
} from "./review-outcome.js";
import type { CheckAnnotation } from "./sanitize.js";

// The Check-posting helpers live in check-posting.ts; callers (and the tests)
// know them from here, so they are re-exported.
export {
    postCheckRun,
    postIncompleteCheckRun,
    postReviewFailureCheckRun,
    toAnnotations,
};
export type { ReviewFinding };

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
    /** `repository.id`: every installation token is limited to this repository. */
    repositoryId: number;
    /** The delivery's `pull_request.head.sha`: a full lowercase hex commit id. */
    headSha: string;
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
 * review-purpose installation token (pull_requests:read on one repository,
 * see auth.ts) — found live: every real webhook failed with "GITHUB_TOKEN
 * environment variable is required for --pr." until it was forwarded.
 *
 * `outcomeFile`, when given, is handed to the child as
 * TRELIX_REVIEW_OUTCOME_FILE: where it writes the record of which hunks it
 * did not review. Resolves with the findings on exit 0 and rejects on any
 * other exit status, including 4 (a review that covered only part of the
 * diff), where the findings are in the error's `stdout`: see runReview.
 */
export async function runReviewCli(
    request: Pick<ReviewRequest, "owner" | "repo" | "prNumber">,
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
    /** Injectable fake HTTP transport for the installation-token mints (see auth.test.ts). */
    request?: RequestInterface;
    /**
     * Injectable Octokit instance for the Check-run call — tests substitute
     * one with an `octokit.hook.wrap("request", ...)` interceptor so the call
     * never hits the real GitHub API. Defaults to a real Octokit
     * authenticated with the `poster` installation token.
     */
    octokit?: Octokit;
    /**
     * Tests only: the directory the per-review outcome directory is created
     * in, so a test can see that it is created and removed. Defaults to the
     * OS temp directory.
     */
    outcomeBaseDir?: string;
    /**
     * Called when the run ends without a verdict Check for `request.headSha`:
     * the review covered only part of the diff (a neutral or failing
     * "incomplete" Check was posted) or the checkout was no longer at that
     * commit (nothing was posted). The webhook queue uses it to release its
     * claim on the commit, so the same commit can be reviewed again. Not
     * called when the run throws: the caller sees the error instead.
     */
    onNoVerdict?: () => void;
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
 * because the review did run, after calling `onNoVerdict`: that Check is not
 * a verdict), and a neutral or timed-out Check on anything else, after which
 * the error is rethrown.
 */
async function reviewAndPost(
    octokit: Octokit,
    request: ReviewRequest,
    headSha: string,
    workspacePath: string,
    token: string,
    outcomeFile: string,
    onNoVerdict?: () => void,
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
            onNoVerdict?.();
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

/** `request.installationId`; throws, before any token is minted, when the delivery had none. */
function requireInstallationId(request: ReviewRequest): number {
    if (request.installationId === undefined) {
        throw new Error(
            "runReview requires an installationId to authenticate the Checks API call",
        );
    }
    return request.installationId;
}

/**
 * End-to-end: mint the three purpose-scoped installation tokens (auth.ts: `checkout`
 * for git, `review` for the `trelix review` child, `poster` for the Octokit that creates
 * the Check), clone the PR head into a fresh workspace, index it, review it and post the
 * findings as a Check run on `request.headSha`, cleaning up the workspace and the outcome
 * directory unconditionally. Requires request.installationId (`installation.id`). If the
 * checkout is not at `request.headSha` (a newer push moved refs/pull/<n>/head; it has its
 * own delivery) it logs, reviews nothing and returns `[]`.
 */
export async function runReview(
    config: AppConfig,
    request: ReviewRequest,
    options: RunReviewOptions = {},
): Promise<ReviewFinding[]> {
    const checkoutPullRequest =
        options.checkoutPullRequest ?? defaultCheckoutPullRequest;

    const tokens = await getPurposeTokens(
        config,
        requireInstallationId(request),
        request.repositoryId,
        options.request,
    );
    const octokit = options.octokit ?? new Octokit({ auth: tokens.poster });

    const workspace = await checkoutPullRequest(tokens.checkout, request);
    let outcomeLocation: OutcomeLocation | undefined;
    try {
        if (workspace.headSha !== request.headSha) {
            console.warn(
                `[review-runner] skipping ${request.owner}/${request.repo}#${request.prNumber}: checkout is at ${workspace.headSha}, delivery is for ${request.headSha}; nothing posted`,
            );
            options.onNoVerdict?.();
            return [];
        }
        // Outside the checkout, so the pull request cannot touch the file.
        outcomeLocation = await createOutcomeLocation(options.outcomeBaseDir);
        await indexRepository(workspace.path);
        return await reviewAndPost(
            octokit,
            request,
            request.headSha,
            workspace.path,
            tokens.review,
            outcomeLocation.file,
            options.onNoVerdict,
        );
    } finally {
        try {
            await workspace.cleanup();
        } finally {
            await outcomeLocation?.cleanup();
        }
    }
}
