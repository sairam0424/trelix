import { execFile } from "node:child_process";
import { promisify } from "node:util";
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
 *
 * Posting Check annotations via Octokit (using the installation token from
 * getInstallationToken) is not yet wired — see item 6b, which also adds
 * the auth this depends on.
 */
export async function runReview(
  config: AppConfig,
  request: ReviewRequest,
): Promise<ReviewFinding[]> {
  if (request.installationId !== undefined) {
    // Not yet used to authenticate the review call itself — see item 6b.
    // Calling it here only so its wiring point is visible in this skeleton.
    await getInstallationToken(config, request.installationId).catch(() => undefined);
  }

  const prRef = `${request.owner}/${request.repo}#${request.prNumber}`;
  const { stdout } = await execFileAsync("trelix", [
    "review",
    config.reviewRepoPath,
    "--pr",
    prRef,
    "--json",
  ]);

  return JSON.parse(stdout) as ReviewFinding[];
}
