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
import subprocess
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("trelix.analysis.taint")


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

        Args:
            rules_path: path to custom semgrep rules directory/file.
                       If None, uses the built-in taint rules registry.
        """
        try:
            output = self._run_semgrep(rules_path)
            return self._parse_semgrep_output(output)
        except Exception as exc:
            logger.debug("TaintAnalyzer.run() failed (non-fatal): %s", exc)
            return []

    def _run_semgrep(self, rules_path: str | None) -> str:
        """Invoke semgrep CLI and return JSON output string."""
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

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
        return result.stdout

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
