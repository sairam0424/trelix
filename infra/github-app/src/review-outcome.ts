/**
 * What the App does with a `trelix review` that covered only part of the diff.
 *
 * `trelix review` exits 4 (REVIEW_INCOMPLETE_EXIT_CODE in src/trelix/cli/main.py)
 * after printing the findings it has, and with TRELIX_REVIEW_OUTCOME_FILE set it
 * writes a small JSON record of what it did and did not cover. This module holds
 * the three pieces of that path that have no I/O to a GitHub API: the conclusion
 * rule, the strict reader of the record, and the summary text. The workflow
 * .github/workflows/trelix-review.yml carries its own copy of each (it is inline
 * JavaScript and cannot import this file); both are tested against the same table,
 * infra/github-app/tests/fixtures/review-conclusion-cases.json.
 *
 * The record is read as untrusted input. Its path is a file name from the pull
 * request, so it is shown only inside a code span, on one line, after the
 * sanitiser has been through it; its statuses are matched against the fixed
 * vocabulary and never echoed; and nothing in it that is not a number, a path or
 * one of those statuses is ever posted. Any doubt about the record makes the whole
 * record "unknown", which the summary says in words: it is never read as clean.
 */

import { lstat, mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { TEMP_DIR_PREFIX } from "./repo-checkout.js";
import { sanitizeText } from "./sanitize.js";

/**
 * Exit status of `trelix review` when it could not review at all (no usable
 * LLM, or no hunk got a usable review and none kept a finding) —
 * REVIEW_NOT_RUN_EXIT_CODE in src/trelix/cli/main.py. Keep the two in step:
 * tests/unit/test_review_exit_code_contract.py fails if they drift.
 */
export const REVIEW_NOT_RUN_EXIT_CODE = 3;

/**
 * Exit status of `trelix review` when it reviewed part of the diff and left more
 * of it unreviewed than TRELIX_REVIEW_MAX_UNREVIEWED_FRACTION allows; stdout still
 * has the findings — REVIEW_INCOMPLETE_EXIT_CODE in src/trelix/cli/main.py. Kept
 * in step with it by the same contract test.
 */
export const REVIEW_INCOMPLETE_EXIT_CODE = 4;

export type ReviewConclusion = "success" | "failure" | "neutral";

/**
 * The conclusion of the Check for a `trelix review` run that ended with
 * `exitCode` (null: no exit status is known) and printed `findings`:
 *
 * - exit 0: success, or failure if any finding is an ERROR;
 * - exit 4 (incomplete): neutral, or failure if any finding is an ERROR;
 * - exit 3 (did not run), any other status, or none: neutral;
 * - `findings` null (they could not be read): neutral whatever the status,
 *   because nothing can be said about them. Never read as "no findings".
 *
 * The verdict comes from all the findings, not from the ones that got an
 * annotation. The workflow's copy is tested against the same table of cases.
 */
export function reviewConclusion(
    exitCode: number | null,
    findings: readonly { severity: unknown }[] | null,
): ReviewConclusion {
    if (findings === null) {
        return "neutral";
    }
    const hasError = findings.some((finding) => finding.severity === "ERROR");
    if (exitCode === 0) {
        return hasError ? "failure" : "success";
    }
    if (exitCode === REVIEW_INCOMPLETE_EXIT_CODE) {
        return hasError ? "failure" : "neutral";
    }
    return "neutral";
}

/** The statuses `trelix review` gives an unreviewed hunk (src/trelix/review/hunk_status.py). "reviewed" never appears in the list. */
const UNREVIEWED_STATUSES = [
    "truncated",
    "refused",
    "parse_failed",
    "error",
] as const;

export type UnreviewedStatus = (typeof UNREVIEWED_STATUSES)[number];

export interface UnreviewedHunk {
    file: string;
    line: number;
    status: UnreviewedStatus;
}

/** A record that passed every check in parseOutcomeRecord. */
export interface ReviewOutcomeRecord {
    hunksTotal: number;
    hunksReviewed: number;
    hunksUnreviewed: number;
    /** Unreviewed hunks the record counts but does not list. */
    hunksOmitted: number;
    hunks: UnreviewedHunk[];
}

/** schema_version of src/trelix/review/outcome_file.py. */
const OUTCOME_SCHEMA_VERSION = 1;
/** The record has a hundred hunks at most, each with a file name; this is generous and bounds what is read. */
const MAX_OUTCOME_BYTES = 1024 * 1024;

function isCount(value: unknown): value is number {
    return (
        typeof value === "number" && Number.isSafeInteger(value) && value >= 0
    );
}

/** A JSON object: not null, not an array. */
export function isRecord(value: unknown): value is Record<string, unknown> {
    return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isUnreviewedStatus(value: unknown): value is UnreviewedStatus {
    return (UNREVIEWED_STATUSES as readonly unknown[]).includes(value);
}

function parseHunk(value: unknown): UnreviewedHunk | null {
    if (!isRecord(value)) {
        return null;
    }
    const { file, line, status } = value;
    if (typeof file !== "string" || !isCount(line)) {
        return null;
    }
    return isUnreviewedStatus(status) ? { file, line, status } : null;
}

/**
 * The record in `raw`, or null when it is not valid JSON or does not hold
 * together: wrong schema version, counts that are not whole numbers or do not
 * add up (reviewed + unreviewed = total, listed + omitted = unreviewed), no
 * unreviewed hunk at all, an exit code other than 4, or a hunk with a status
 * outside the vocabulary. Nothing is repaired: a record that is partly wrong is
 * wholly unknown.
 */
export function parseOutcomeRecord(raw: string): ReviewOutcomeRecord | null {
    let document: unknown;
    try {
        document = JSON.parse(raw);
    } catch {
        return null;
    }
    if (!isRecord(document)) {
        return null;
    }
    const total = document.hunks_total;
    const reviewed = document.hunks_reviewed;
    const unreviewed = document.hunks_unreviewed;
    const omitted = document.hunks_omitted;
    if (
        document.schema_version !== OUTCOME_SCHEMA_VERSION ||
        document.exit_code !== REVIEW_INCOMPLETE_EXIT_CODE ||
        !isCount(total) ||
        !isCount(reviewed) ||
        !isCount(unreviewed) ||
        !isCount(omitted) ||
        unreviewed < 1 ||
        reviewed + unreviewed !== total ||
        !Array.isArray(document.hunks)
    ) {
        return null;
    }
    const hunks: UnreviewedHunk[] = [];
    for (const entry of document.hunks) {
        const hunk = parseHunk(entry);
        if (hunk === null) {
            return null;
        }
        hunks.push(hunk);
    }
    if (hunks.length + omitted !== unreviewed) {
        return null;
    }
    return {
        hunksTotal: total,
        hunksReviewed: reviewed,
        hunksUnreviewed: unreviewed,
        hunksOmitted: omitted,
        hunks,
    };
}

/**
 * Reads and checks the record at `path`; null when it is missing, is not a
 * regular file, is too large, cannot be read or does not pass
 * parseOutcomeRecord. The reason is logged, never posted.
 */
export async function readOutcomeRecord(
    path: string,
): Promise<ReviewOutcomeRecord | null> {
    try {
        const info = await lstat(path);
        if (!info.isFile() || info.size > MAX_OUTCOME_BYTES) {
            console.warn(
                "[review-outcome] the outcome record is not a regular file of a sane size; treating it as unknown",
            );
            return null;
        }
        const record = parseOutcomeRecord(await readFile(path, "utf8"));
        if (record === null) {
            console.warn(
                "[review-outcome] the outcome record is not valid or not consistent; treating it as unknown",
            );
        }
        return record;
    } catch (err) {
        console.warn(
            "[review-outcome] the outcome record could not be read; treating it as unknown:",
            err,
        );
        return null;
    }
}

const OUTCOME_DIR_PREFIX = `${TEMP_DIR_PREFIX}outcome-`;
const OUTCOME_FILE_NAME = "outcome.json";

export interface OutcomeLocation {
    /** Where `trelix review` is told to write its record (TRELIX_REVIEW_OUTCOME_FILE). */
    file: string;
    /** Removes the directory; never rejects (a failure is logged). */
    cleanup(): Promise<void>;
}

/**
 * A fresh private directory for the record, outside the pull request's
 * checkout: a pull request cannot pre-create, replace or symlink the file.
 * `mkdtemp` creates it with mode 0700. The name starts with the checkout
 * prefix, so `sweepStaleWorkspaces` removes one left behind by a killed process.
 */
export async function createOutcomeLocation(
    baseDir: string = tmpdir(),
): Promise<OutcomeLocation> {
    const dir = await mkdtemp(join(baseDir, OUTCOME_DIR_PREFIX));
    return {
        file: join(dir, OUTCOME_FILE_NAME),
        cleanup: async () => {
            try {
                await rm(dir, { recursive: true, force: true });
            } catch (err) {
                console.warn(`[review-outcome] could not remove ${dir}:`, err);
            }
        },
    };
}

/** How many unreviewed hunks the summary lists, and how much of a file name it shows. Sized so the longest summary fits MAX_SUMMARY_LENGTH after sanitising. */
export const MAX_LISTED_HUNKS = 10;
export const MAX_DISPLAY_PATH_LENGTH = 100;
/** Only this much of a file name is looked at, so the cost of showing one does not depend on how long it is. Generous: the sanitiser removes some of it before the cap applies. */
const MAX_PATH_INPUT_LENGTH = 4 * MAX_DISPLAY_PATH_LENGTH;

const PATH_PLACEHOLDER = "(unprintable file name)";
/** Modifier letter grave accent: looks like a backtick, cannot close a code span. */
const LOOKALIKE_BACKTICK = "ˋ";
const BACKTICK = /`/g;

/**
 * A file name from the pull request made safe to show: one line, no hidden
 * characters, capped, and without a backtick, so that inside a code span it
 * cannot do anything but be read.
 */
export function displayPath(file: string): string {
    const text = sanitizeText(
        file.slice(0, MAX_PATH_INPUT_LENGTH),
        MAX_DISPLAY_PATH_LENGTH,
        {
            singleLine: true,
        },
    );
    return text === ""
        ? PATH_PLACEHOLDER
        : text.replace(BACKTICK, LOOKALIKE_BACKTICK);
}

function hunkLine(hunk: UnreviewedHunk): string {
    return `- \`${displayPath(hunk.file)}:${hunk.line}\` (${hunk.status})`;
}

function findingsLine(
    findingCount: number | null,
    annotationCount: number,
): string {
    if (findingCount === null) {
        return "The list of findings could not be read, so none are shown.";
    }
    if (findingCount === 0) {
        return "No findings in the hunks that were reviewed.";
    }
    let line = `${findingCount} issue(s) found in the hunks that were reviewed.`;
    if (annotationCount < findingCount) {
        line += ` ${findingCount - annotationCount} of them could not be shown as inline annotations (GitHub allows 50 per check, or the file path was not usable).`;
    }
    return line;
}

/**
 * The summary of the Check for a review that covered only part of the diff.
 * `outcome` null means the record is unknown; `findingCount` null means the
 * findings could not be read. The only text that reaches it from outside is
 * the file names in `outcome`, through displayPath; the statuses are the fixed
 * vocabulary and the rest are numbers. No model prose is ever included.
 */
export function buildIncompleteSummary(
    findingCount: number | null,
    annotationCount: number,
    outcome: ReviewOutcomeRecord | null,
): string {
    const lines: string[] = [];
    if (outcome === null) {
        lines.push(
            "trelix reviewed only part of this PR, but the record of what it covered is missing or unreadable, so how much was left unreviewed is unknown. This is not a clean result.",
        );
    } else {
        lines.push(
            `trelix reviewed only part of this PR: ${outcome.hunksReviewed} of ${outcome.hunksTotal} hunks were reviewed and ${outcome.hunksUnreviewed} were not. This is not a clean result.`,
        );
    }
    lines.push("", findingsLine(findingCount, annotationCount));
    if (outcome !== null) {
        const listed = outcome.hunks.slice(0, MAX_LISTED_HUNKS);
        lines.push("", "Hunks that were not reviewed:");
        lines.push(...listed.map(hunkLine));
        const unlisted = outcome.hunksUnreviewed - listed.length;
        if (unlisted > 0) {
            lines.push(`- and ${unlisted} more.`);
        }
    }
    return lines.join("\n");
}
