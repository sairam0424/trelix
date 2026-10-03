import type { Octokit } from "@octokit/rest";
import {
    buildIncompleteSummary,
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

/**
 * Everything this service posts to the Checks API, and the mapping from the
 * findings of `trelix review` to what is posted. Split out of review-runner.ts,
 * which runs the review; nothing here starts a process or reads the environment.
 */

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
export function exitCodeOf(err: unknown): number | null {
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
