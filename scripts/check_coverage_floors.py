#!/usr/bin/env python3
"""Per-package coverage floors, checked after a run against its JSON report.

WHY THIS IS A SCRIPT AND NOT A TEST. Two reasons, both of which make the pytest version
wrong rather than merely inconvenient:

  1. A test cannot read its own run's coverage. pytest-cov writes the report at session
     finish, after every test has already passed or failed.
  2. A test that reads a report file must skip when the file is absent -- and in CI that
     file is produced by the very command running the test. A check that can only skip is
     a green identical to a check that never ran, which is the exact defect class this
     repo has spent a month removing.

So: run the suite with `--cov-report=json:<path>`, then run this. Exit 1 names the package.

WHY PER-PACKAGE AT ALL. One global `fail_under` cannot say WHERE coverage fell, and it
lets a rise in a well-covered package mask a collapse elsewhere. The extractors are this
repo's core value (per-language symbol extraction) and simultaneously its weakest area --
11 of the 15 worst-covered modules, ~727 missing statements -- so they get a floor that
cannot be offset by `retrieval/` improving.

Floors are seeded from measured values MINUS a small margin, so this starts green and
ratchets. It is a regression detector, not an aspiration.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

# package prefix (relative to src/trelix/) -> minimum combined line+branch percent.
# Seeded 2026-08-23 from a measured `--cov-branch` run of tests/unit, each floor set a
# few points below actual so normal drift does not trip it. Raise a floor only with the
# measurement that justifies it in the same commit.
FLOORS: dict[str, float] = {
    "indexing/parser/extractors": 65.0,
    "indexing": 70.0,
    "retrieval": 75.0,
    "store": 72.0,
    "embedder": 70.0,
    "cli": 60.0,
}


def package_percent(report: dict[str, Any], package: str) -> tuple[float | None, int]:
    """Combined line+branch percent for one package prefix, and the file count.

    Returns (None, 0) when the prefix matches nothing -- a dead floor, which this script
    treats as an error rather than a pass. A floor guarding a package that has been
    renamed or removed is worse than no floor: it reports success forever.
    """
    covered = total = files = 0
    for path, entry in report["files"].items():
        rel = path.replace("src/trelix/", "", 1)
        if not rel.startswith(package):
            continue
        files += 1
        s = entry["summary"]
        # Mirror coverage.py's own combined denominator: statements + branch destinations.
        total += s["num_statements"] + s.get("num_branches", 0)
        covered += (s["num_statements"] - s["missing_lines"]) + s.get("covered_branches", 0)
    if total == 0:
        return None, files
    return 100.0 * covered / total, files


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("report", type=pathlib.Path, help="coverage JSON report path")
    args = ap.parse_args()

    if not args.report.is_file():
        print(f"ERROR: no coverage report at {args.report}", file=sys.stderr)
        print("Run pytest with --cov-report=json:<path> first.", file=sys.stderr)
        return 2

    report = json.loads(args.report.read_text())
    overall = report["totals"]["percent_covered"]
    print(f"overall combined coverage: {overall:.2f}%")

    failures: list[str] = []
    for package in sorted(FLOORS):
        pct, files = package_percent(report, package)
        floor = FLOORS[package]
        if pct is None:
            failures.append(
                f"{package}: matched 0 files -- the floor is DEAD. Either the package "
                f"moved or the prefix is wrong; a floor that matches nothing passes forever."
            )
            print(f"  DEAD  {package:32} matched no files")
            continue
        ok = pct >= floor
        if not ok:
            failures.append(f"{package}: {pct:.2f}% is below its recorded floor {floor}%")
        status = "ok  " if ok else "FAIL"
        print(f"  {status}  {package:32} {pct:6.2f}%  (floor {floor}, {files} files)")

    if failures:
        print("\ncoverage floor violations:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1

    print("\nall package floors met")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
