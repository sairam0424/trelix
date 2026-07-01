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

logger = logging.getLogger("trelix.analysis.taint")


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
                source_file = item.get("path", "")
                source_line = item.get("start", {}).get("line", 0)
                rule_id = item.get("check_id", "unknown")
                severity = item.get("severity", "INFO")

                # Try to extract sink from dataflow_trace
                trace = item.get("extra", {}).get("dataflow_trace", {})
                sink_loc = trace.get("taint_sink", {}).get("location", {})
                sink_file = sink_loc.get("path", source_file)
                sink_line = sink_loc.get("start", {}).get("line", source_line)

                flows.append(TaintFlow(
                    source_file=source_file,
                    source_line=int(source_line),
                    sink_file=sink_file,
                    sink_line=int(sink_line),
                    rule_id=rule_id,
                    severity=severity,
                ))
            except Exception:
                continue

        return flows
