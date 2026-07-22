import { writeFileSync, mkdtempSync, chmodSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { toAnnotations, runReviewCli, ReviewFinding } from "../src/review-runner.js";

describe("toAnnotations", () => {
  it("maps ERROR/WARN/INFO severities to failure/warning/notice", () => {
    const findings: ReviewFinding[] = [
      { file: "a.py", lines: "1-1", severity: "ERROR", comment: "bug" },
      { file: "b.py", lines: "2-2", severity: "WARN", comment: "smell" },
      { file: "c.py", lines: "3-3", severity: "INFO", comment: "note" },
    ];

    const annotations = toAnnotations(findings);

    expect(annotations[0].annotation_level).toBe("failure");
    expect(annotations[1].annotation_level).toBe("warning");
    expect(annotations[2].annotation_level).toBe("notice");
  });

  it("parses the 'start-end' lines field into start_line/end_line", () => {
    const findings: ReviewFinding[] = [
      { file: "a.py", lines: "10-25", severity: "INFO", comment: "x" },
    ];

    const [annotation] = toAnnotations(findings);

    expect(annotation.start_line).toBe(10);
    expect(annotation.end_line).toBe(25);
  });

  it("caps annotations at the given limit (GitHub's per-check-run annotation limit)", () => {
    const findings: ReviewFinding[] = Array.from({ length: 60 }, (_, i) => ({
      file: `f${i}.py`,
      lines: "1-1",
      severity: "INFO" as const,
      comment: "x",
    }));

    expect(toAnnotations(findings, 50)).toHaveLength(50);
    expect(toAnnotations(findings)).toHaveLength(50); // default limit
  });

  it("carries the comment through as the annotation message and a fixed title", () => {
    const findings: ReviewFinding[] = [
      { file: "a.py", lines: "1-1", severity: "ERROR", comment: "specific finding text" },
    ];

    const [annotation] = toAnnotations(findings);

    expect(annotation.message).toBe("specific finding text");
    expect(annotation.title).toBe("trelix review");
    expect(annotation.path).toBe("a.py");
  });

  it("returns an empty list for an empty findings array", () => {
    expect(toAnnotations([])).toEqual([]);
  });
});

describe("runReviewCli timeout", () => {
  let binDir: string;
  let originalPath: string | undefined;

  beforeEach(() => {
    binDir = mkdtempSync(join(tmpdir(), "trelix-fake-bin-"));
    // A real slow "trelix" binary — not a mock of execFile's timeout
    // mechanism, an actual subprocess that actually sleeps, so this test
    // exercises the real kill-on-timeout path end to end.
    const shim = join(binDir, "trelix");
    writeFileSync(shim, "#!/bin/sh\nsleep 5\necho '[]'\n");
    chmodSync(shim, 0o755);
    originalPath = process.env.PATH;
    process.env.PATH = `${binDir}:${originalPath}`;
  });

  afterEach(() => {
    process.env.PATH = originalPath;
    rmSync(binDir, { recursive: true, force: true });
  });

  it("kills a hung `trelix review` subprocess once the timeout elapses", async () => {
    const request = { owner: "o", repo: "r", prNumber: 1 };

    await expect(runReviewCli(request, ".", 200)).rejects.toMatchObject({
      killed: true,
      signal: "SIGTERM",
    });
  }, 10_000);

  it("does not time out a fast-returning subprocess", async () => {
    const shim = join(binDir, "trelix");
    writeFileSync(shim, "#!/bin/sh\necho '[]'\n");
    chmodSync(shim, 0o755);
    const request = { owner: "o", repo: "r", prNumber: 1 };

    await expect(runReviewCli(request, ".", 5000)).resolves.toEqual([]);
  });
});
