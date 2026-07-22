import { describe, expect, it } from "vitest";
import { toAnnotations, ReviewFinding } from "../src/review-runner.js";

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
