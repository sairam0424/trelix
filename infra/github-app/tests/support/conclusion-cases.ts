/**
 * The table of cases in tests/fixtures/review-conclusion-cases.json, which the
 * Actions workflow's publishing script is tested against as well
 * (tests/unit/test_review_not_run_exit.py and test_review_publish_incomplete.py).
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

export interface ConclusionCase {
    id: string;
    /** What the workflow step records; null = nothing. */
    exit_code: string | null;
    severities: string[];
    /** What `trelix review --json` printed when that is not a findings array; the findings cannot be read. */
    stdout_text?: string;
    outcome?: "valid" | "missing" | "corrupt";
    conclusion: "success" | "failure" | "neutral";
    title: string;
    applies_to?: Array<"workflow" | "app">;
}

export interface ConclusionCases {
    valid_outcome: {
        schema_version: number;
        hunks_total: number;
        hunks_reviewed: number;
        hunks_unreviewed: number;
        exit_code: number;
        hunks: Array<{
            file: string;
            line: number;
            status: string;
            detail: string;
        }>;
        hunks_omitted: number;
    };
    corrupt_outcome_text: string;
    invalid_outcomes: Array<{ id: string; text: string }>;
    cases: ConclusionCase[];
}

export function loadConclusionCases(): ConclusionCases {
    const path = fileURLToPath(
        new URL("../fixtures/review-conclusion-cases.json", import.meta.url),
    );
    return JSON.parse(readFileSync(path, "utf8")) as ConclusionCases;
}

/** The findings `trelix review --json` prints for these severities, one file each. */
export function findingsText(severities: readonly string[]): string {
    return JSON.stringify(
        severities.map((severity, i) => ({
            file: `src/f${i}.py`,
            lines: `${i + 1}-${i + 1}`,
            severity,
            comment: "c",
        })),
    );
}

/** What `trelix review --json` prints in a table row: its `stdout_text`, or the findings its severities describe. */
export function stdoutTextFor(row: ConclusionCase): string {
    return row.stdout_text ?? findingsText(row.severities);
}

/** The outcome record a table row asks for: the valid one, none, or one that is not JSON. */
export function outcomeTextFor(
    row: ConclusionCase,
    cases: ConclusionCases,
): string | null {
    switch (row.outcome) {
        case "valid":
            return JSON.stringify(cases.valid_outcome);
        case "corrupt":
            return cases.corrupt_outcome_text;
        default:
            return null;
    }
}
