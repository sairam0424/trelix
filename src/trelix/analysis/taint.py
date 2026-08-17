"""
Taint analysis integration via Semgrep.

Semgrep's taint mode propagates taint from sources (user input, env vars,
DB reads) to sinks (SQL queries, shell commands, HTTP responses). Three tiers:
  - Default (intraprocedural): fast, within-function only
  - --pro-intrafile: cross-function within one file
  - --pro --interfile: full inter-procedural, most powerful

Research basis: Semgrep taint-mode docs (2-1 adversarial vote).

This module wraps the semgrep CLI via subprocess. Semgrep is optional
(pip install trelix[taint]). Returns [] if semgrep is not installed.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

logger = logging.getLogger("trelix.analysis.taint")


class ScanOutcome(StrEnum):
    """Why a scan produced the flows it did — or produced none.

    `run()` collapses all four of these into an empty list, which is why callers could
    not tell "your code is clean" from "the tool never ran". For a security check those
    are opposite conclusions.
    """

    OK = "ok"  # semgrep examined files and reported whatever it found
    SEMGREP_MISSING = "semgrep-missing"  # the binary is not on PATH
    SEMGREP_FAILED = "semgrep-failed"  # it ran and errored
    SCANNED_NOTHING = "scanned-nothing"  # it succeeded but examined zero files


@dataclass
class ScanResult:
    """Flows plus enough context to describe them honestly."""

    flows: list[TaintFlow] = field(default_factory=list)
    outcome: ScanOutcome = ScanOutcome.OK
    detail: str = ""
    files_scanned: int = 0

    @property
    def is_trustworthy(self) -> bool:
        """True only when an empty `flows` genuinely means "nothing found"."""
        return self.outcome is ScanOutcome.OK


def _extract_location(node: object) -> dict[str, Any] | None:
    """Return the `{path, start: {line}}` location out of a dataflow-trace entry.

    semgrep encodes `taint_source` / `taint_sink` as a two-element tagged LIST —
    `["CliLoc", [<location>, "<matched code>"]]` — not as a mapping. Calling `.get()`
    on that list is what previously raised `AttributeError` and, via a bare `except`,
    discarded every genuine taint flow.

    Both shapes are accepted so the parser does not depend on one semgrep version:
    the tagged list above, and the `{"location": {...}}` (or bare location) mapping
    that older builds emit. Anything else yields None so the caller can fall back to
    the match location rather than invent one.
    """
    if isinstance(node, list):
        # ["CliLoc", [location, code]] — the payload is itself a list.
        if len(node) < 2 or not isinstance(node[1], list) or not node[1]:
            return None
        location = node[1][0]
        return location if isinstance(location, dict) else None

    if isinstance(node, dict):
        nested = node.get("location")
        if isinstance(nested, dict):
            return nested
        # A bare location mapping, already at the right level.
        return node or None

    return None


@dataclass
class TaintFlow:
    """A single taint propagation path from source to sink."""

    source_file: str
    source_line: int
    sink_file: str
    sink_line: int
    rule_id: str
    severity: str  # "ERROR" | "WARNING" | "INFO"


class TaintAnalyzer:
    """
    Run Semgrep taint analysis against a repository and return TaintFlow results.

    Requires: pip install trelix[taint]  (installs semgrep)
    Safe to instantiate without semgrep installed -- run() returns [] gracefully.

    Usage:
        analyzer = TaintAnalyzer("/path/to/repo")
        flows = analyzer.run()
        # flows: list[TaintFlow] with source->sink propagation paths
    """

    def __init__(self, repo_path: str, tier: str = "default") -> None:
        """
        Args:
            repo_path: absolute path to the repository root
            tier: "default" (intraprocedural), "intrafile", or "interfile"
        """
        self._repo_path = repo_path
        self._tier = tier

    def run(self, rules_path: str | None = None) -> list[TaintFlow]:
        """
        Run semgrep taint analysis. Returns [] on any failure.

        Kept as a thin wrapper over `scan()` so its documented never-raises,
        always-a-list contract is unchanged for existing callers. Anything that needs to
        distinguish "clean" from "did not run" must use `scan()` — this signature cannot
        express the difference, which is exactly how a failed scan came to be reportable
        as a clean one.
        """
        return self.scan(rules_path).flows

    def scan(self, rules_path: str | None = None) -> ScanResult:
        """Run semgrep and report both the flows and WHY there were that many.

        Three independent signals are needed, because no one of them covers every case:

          - the exit code catches a hard failure, but semgrep exits 1 when it HAS
            findings, so a non-zero code is not itself an error;
          - the `errors[]` array in semgrep's own JSON catches a rule or target problem
            that still produced parseable output — invisible to an exit-code check that
            also requires empty stdout;
          - `paths.scanned` catches the vacuous success where semgrep exits 0 having
            examined nothing at all, which reports identically to a clean scan.

        Never raises, matching `run()`.
        """
        if shutil.which("semgrep") is None:
            return ScanResult(
                outcome=ScanOutcome.SEMGREP_MISSING,
                detail="the semgrep binary is not on PATH",
            )

        try:
            completed = self._invoke_semgrep(rules_path)
        except Exception as exc:
            logger.warning("Taint scan could not start: %s", exc)
            return ScanResult(outcome=ScanOutcome.SEMGREP_FAILED, detail=str(exc))

        stdout = completed.stdout or ""
        try:
            payload = json.loads(stdout) if stdout.strip() else {}
        except json.JSONDecodeError as exc:
            logger.warning(
                "Taint scan produced unparseable output (exit %s): %s",
                completed.returncode,
                (completed.stderr or str(exc))[:400].strip(),
            )
            return ScanResult(
                outcome=ScanOutcome.SEMGREP_FAILED,
                detail=(completed.stderr or str(exc))[:400].strip(),
            )

        # Every accessor below is shape-guarded. `run()` documents that it never raises,
        # and a payload that parses to `null`, a list, or a dict whose "paths" is a
        # string made these `.get()` calls throw straight out through `run()` — three
        # shapes the pre-fix code returned [] for.
        if not isinstance(payload, dict):
            logger.warning(
                "Taint scan output was valid JSON but not an object (%s) — treating as a "
                "failed scan",
                type(payload).__name__,
            )
            return ScanResult(
                outcome=ScanOutcome.SEMGREP_FAILED,
                detail=f"unexpected JSON payload of type {type(payload).__name__}",
            )

        raw_errors = payload.get("errors")
        errors = raw_errors if isinstance(raw_errors, list) else []
        raw_paths = payload.get("paths")
        raw_scanned = raw_paths.get("scanned") if isinstance(raw_paths, dict) else None
        scanned = raw_scanned if isinstance(raw_scanned, list) else []
        flows = self._parse_semgrep_output(stdout) if stdout.strip() else []

        # Only errors semgrep itself considers errors. It uses `level` to distinguish a
        # genuine failure from a note — a rule skipped by min-version reports
        # {"code": 0, "level": "info"} on an otherwise successful exit-0 scan, and
        # treating that as a hard failure made the command cry wolf on a healthy run.
        # An entry with no `level` is treated as an error, because that is what semgrep
        # emitted before the field existed.
        fatal_errors = [
            e
            for e in errors
            if not isinstance(e, dict) or str(e.get("level", "error")).lower() != "info"
        ]

        # A non-zero code with no usable payload, or any error semgrep called an error.
        if fatal_errors or (completed.returncode != 0 and not payload):
            detail = (completed.stderr or "").strip() or "; ".join(
                str(e.get("message", e) if isinstance(e, dict) else e)[:200]
                for e in fatal_errors[:3]
            )
            logger.warning(
                "Taint scan FAILED (exit %s) — results are not a clean bill of health: %s",
                completed.returncode,
                detail[:400],
            )
            return ScanResult(
                flows=flows,
                outcome=ScanOutcome.SEMGREP_FAILED,
                detail=detail[:400],
                files_scanned=len(scanned),
            )

        if not scanned:
            logger.warning("Taint scan examined 0 files — check the target path and rule languages")
            return ScanResult(
                flows=flows,
                outcome=ScanOutcome.SCANNED_NOTHING,
                detail="semgrep succeeded but examined no files",
            )

        return ScanResult(flows=flows, outcome=ScanOutcome.OK, files_scanned=len(scanned))

    def _run_semgrep(self, rules_path: str | None) -> str:
        """Invoke semgrep and return its stdout.

        Retained for callers that patch it. It discards the exit code and stderr, which
        is why `scan()` uses `_invoke_semgrep` instead: a failed run returns "" here and
        is indistinguishable from a clean one.
        """
        return self._invoke_semgrep(rules_path).stdout

    def _invoke_semgrep(self, rules_path: str | None) -> subprocess.CompletedProcess[str]:
        """Invoke the semgrep CLI and return the whole CompletedProcess."""
        cmd = ["semgrep", "--json", "--no-rewrite-rule-ids"]

        if self._tier == "intrafile":
            cmd.extend(["--pro-intrafile"])
        elif self._tier == "interfile":
            cmd.extend(["--pro", "--interfile"])

        if rules_path:
            cmd.extend(["--config", rules_path])
        else:
            # Use auto-detect taint rules from semgrep registry
            cmd.extend(["--config", "p/default"])

        cmd.append(self._repo_path)

        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )

    def _parse_semgrep_output(self, output: str) -> list[TaintFlow]:
        """Parse semgrep JSON output into TaintFlow objects."""
        if not output.strip():
            return []
        try:
            data = json.loads(output)
        except json.JSONDecodeError:
            return []

        flows: list[TaintFlow] = []
        for item in data.get("results", []):
            try:
                rule_id = item.get("check_id", "unknown")

                # semgrep reports the match AT THE SINK, so the top-level location is
                # the sink's — not the source's. It is the right fallback for both ends
                # when a rule carries no dataflow trace at all.
                match_file = item.get("path", "")
                match_line = item.get("start", {}).get("line", 0)

                extra = item.get("extra") or {}
                # Severity lives under `extra`. The top-level read is kept as a fallback
                # for the dict-shaped payloads older semgrep builds produced.
                severity = extra.get("severity") or item.get("severity") or "INFO"

                trace = extra.get("dataflow_trace") or {}
                source_loc = _extract_location(trace.get("taint_source")) or {}
                sink_loc = _extract_location(trace.get("taint_sink")) or {}

                flows.append(
                    TaintFlow(
                        source_file=source_loc.get("path") or match_file,
                        source_line=int(source_loc.get("start", {}).get("line", match_line)),
                        sink_file=sink_loc.get("path") or match_file,
                        sink_line=int(sink_loc.get("start", {}).get("line", match_line)),
                        rule_id=rule_id,
                        severity=severity,
                    )
                )
            except Exception as exc:
                # Skipping one malformed result is right; doing it silently is what let a
                # parser that could not read a single real taint flow ship unnoticed.
                logger.warning(
                    "Skipping unparseable semgrep result %r: %s",
                    item.get("check_id", "<unknown>"),
                    exc,
                )
                continue

        return flows
