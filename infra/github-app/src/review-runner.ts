import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { Octokit } from "@octokit/rest";
import { AppConfig } from "./config.js";
import { getInstallationToken } from "./auth.js";

const execFileAsync = promisify(execFile);

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

export function toAnnotations(findings: ReviewFinding[], limit = 50): CheckAnnotation[] {
  return findings.slice(0, limit).map((f) => {
    const [startLine, endLine] = f.lines.split("-").map(Number);
    return {
      path: f.file,
      start_line: startLine || 1,
      end_line: endLine || startLine || 1,
      annotation_level:
        f.severity === "ERROR" ? "failure" : f.severity === "WARN" ? "warning" : "notice",
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
 */
export async function runReviewCli(request: ReviewRequest, repoPath: string): Promise<ReviewFinding[]> {
  const prRef = `${request.owner}/${request.repo}#${request.prNumber}`;
  const { stdout } = await execFileAsync("trelix", ["review", repoPath, "--pr", prRef, "--json"]);
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
    conclusion: annotations.some((a) => a.annotation_level === "failure") ? "failure" : "success",
    output: {
      title: `trelix found ${annotations.length} issue(s)`,
      summary: `trelix reviewed the PR and found ${annotations.length} issue(s).`,
      annotations,
    },
  });
}

/**
 * End-to-end: mint an installation token, run the CLI review, and post
 * the findings as a Check run. Requires request.installationId (the
 * webhook payload's `installation.id` — always present for App-installed
 * webhook deliveries).
 */
export async function runReview(
  config: AppConfig,
  request: ReviewRequest,
): Promise<ReviewFinding[]> {
  if (request.installationId === undefined) {
    throw new Error("runReview requires an installationId to authenticate the Checks API call");
  }

  const token = await getInstallationToken(config, request.installationId);
  const octokit = new Octokit({ auth: token });

  const { data: pull } = await octokit.rest.pulls.get({
    owner: request.owner,
    repo: request.repo,
    pull_number: request.prNumber,
  });

  const findings = await runReviewCli(request, config.reviewRepoPath);
  await postCheckRun(octokit, request.owner, request.repo, pull.head.sha, findings);
  return findings;
}
