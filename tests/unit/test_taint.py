"""Tests for TaintFlow model and TaintAnalyzer (semgrep integration)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from trelix.analysis.taint import TaintAnalyzer, TaintFlow


class TestTaintFlow:
    def test_dataclass_fields(self) -> None:
        flow = TaintFlow(
            source_file="src/auth.py",
            source_line=10,
            sink_file="src/db.py",
            sink_line=25,
            rule_id="taint.sql-injection",
            severity="ERROR",
        )
        assert flow.rule_id == "taint.sql-injection"
        assert flow.severity == "ERROR"


class TestTaintAnalyzer:
    def test_init_with_repo_path(self, tmp_path: Path) -> None:
        analyzer = TaintAnalyzer(str(tmp_path))
        assert analyzer is not None

    def test_run_returns_empty_when_semgrep_not_installed(self, tmp_path: Path) -> None:
        with patch.dict("sys.modules", {"semgrep": None}):
            analyzer = TaintAnalyzer(str(tmp_path))
            result = analyzer.run()
        assert isinstance(result, list)

    def test_run_parses_semgrep_json_output(self, tmp_path: Path) -> None:
        semgrep_output = {
            "results": [
                {
                    "check_id": "taint.sql-injection",
                    "severity": "ERROR",
                    "path": "src/auth.py",
                    "start": {"line": 10},
                    "extra": {
                        "dataflow_trace": {
                            "taint_sink": {"location": {"path": "src/db.py", "start": {"line": 25}}}
                        }
                    },
                }
            ]
        }
        analyzer = TaintAnalyzer(str(tmp_path))
        flows = analyzer._parse_semgrep_output(json.dumps(semgrep_output))
        assert len(flows) == 1
        assert flows[0].rule_id == "taint.sql-injection"
        assert flows[0].source_file == "src/auth.py"

    def test_run_never_raises(self, tmp_path: Path) -> None:
        analyzer = TaintAnalyzer(str(tmp_path))
        with patch.object(analyzer, "_run_semgrep", side_effect=RuntimeError("semgrep not found")):
            result = analyzer.run()
        assert isinstance(result, list)


class TestTaintDB:
    def test_insert_and_retrieve(self, tmp_path: Path) -> None:
        from trelix.store.db import Database

        db = Database(tmp_path / "index.db")
        flows = [TaintFlow("src/a.py", 1, "src/b.py", 5, "sql-inj", "ERROR")]
        db.insert_taint_flows(flows)
        result = db.get_taint_flows()
        assert len(result) == 1
        assert result[0].rule_id == "sql-inj"

    def test_get_taint_flows_by_severity(self, tmp_path: Path) -> None:
        from trelix.store.db import Database

        db = Database(tmp_path / "index.db")
        flows = [
            TaintFlow("a.py", 1, "b.py", 2, "rule-a", "ERROR"),
            TaintFlow("c.py", 3, "d.py", 4, "rule-b", "WARNING"),
        ]
        db.insert_taint_flows(flows)
        errors = db.get_taint_flows(severity="ERROR")
        assert len(errors) == 1
        assert errors[0].severity == "ERROR"


# ---------------------------------------------------------------------------
# Real semgrep output
# ---------------------------------------------------------------------------

# Captured verbatim from `semgrep --json --config <taint-rule> ` (semgrep 1.x) against:
#
#     1  import sqlite3
#     2  def handler(conn):
#     3      user = input("name: ")          <- taint SOURCE
#     4      cur = conn.cursor()
#     5      cur.execute("SELECT ... " + user)  <- taint SINK
#
# Only `path` values were rewritten to a repo-relative stub. Nothing else is
# reshaped, because the point of this fixture is that the shapes are semgrep's
# and not ours: `severity` lives under `extra`, and `taint_source` / `taint_sink`
# are two-element ["CliLoc", [location, code]] LISTS rather than dicts.
#
# The previous version of this file hand-wrote a fixture with `severity` at the
# top level and `taint_sink` as a {"location": ...} dict. Both are wrong, so the
# parser was verified against its own misreading and shipped unable to parse a
# single real taint flow.
REAL_SEMGREP_TAINT_OUTPUT = {
    "results": [
        {
            "check_id": "trelix-test-sql-injection",
            "end": {
                "col": 60,
                "line": 5,
                "offset": 144
            },
            "extra": {
                "dataflow_trace": {
                    "intermediate_vars": [
                        {
                            "content": "user",
                            "location": {
                                "end": {
                                    "col": 9,
                                    "line": 3,
                                    "offset": 42
                                },
                                "path": "src/vuln.py",
                                "start": {
                                    "col": 5,
                                    "line": 3,
                                    "offset": 38
                                }
                            }
                        }
                    ],
                    "taint_sink": [
                        "CliLoc",
                        [
                            {
                                "end": {
                                    "col": 60,
                                    "line": 5,
                                    "offset": 144
                                },
                                "path": "src/vuln.py",
                                "start": {
                                    "col": 5,
                                    "line": 5,
                                    "offset": 89
                                }
                            },
                            "cur.execute(\"SELECT * FROM t WHERE n = '\" + user + \"'\")"
                        ]
                    ],
                    "taint_source": [
                        "CliLoc",
                        [
                            {
                                "end": {
                                    "col": 27,
                                    "line": 3,
                                    "offset": 60
                                },
                                "path": "src/vuln.py",
                                "start": {
                                    "col": 12,
                                    "line": 3,
                                    "offset": 45
                                }
                            },
                            "input(\"name: \")"
                        ]
                    ]
                },
                "message": "Untrusted input reaches a SQL execute call",
                "severity": "ERROR"
            },
            "path": "src/vuln.py",
            "start": {
                "col": 5,
                "line": 5,
                "offset": 89
            }
        }
    ]
}


class TestParseRealSemgrepOutput:
    """The parser must handle the JSON semgrep actually emits.

    Each test below fails against the pre-v3.1.2 parser, and each failure is a
    different consequence of the same root cause: the fixture it was written
    against was invented rather than captured.
    """

    def _flows(self, tmp_path: Path) -> list[TaintFlow]:
        analyzer = TaintAnalyzer(str(tmp_path))
        return analyzer._parse_semgrep_output(json.dumps(REAL_SEMGREP_TAINT_OUTPUT))

    def test_a_real_taint_flow_is_not_discarded(self, tmp_path: Path) -> None:
        """`taint_sink` is a list, so `.get("location")` raised AttributeError.

        The bare `except Exception: continue` swallowed it and dropped the whole
        finding. Because only genuine taint flows carry a `dataflow_trace`, that
        meant taint analysis reported nothing, ever.
        """
        assert len(self._flows(tmp_path)) == 1, (
            "a real semgrep taint result was dropped by the parser"
        )

    def test_source_is_the_taint_source_not_the_match_location(self, tmp_path: Path) -> None:
        """semgrep reports the match AT THE SINK; the source comes from the trace.

        Reading the top-level `path`/`start` as the source inverted the flow —
        it reported the sink line as both source and sink.
        """
        flow = self._flows(tmp_path)[0]
        assert flow.source_line == 3, (
            f"source should be the input() call on line 3, got {flow.source_line}"
        )

    def test_sink_is_read_from_the_taint_sink_entry(self, tmp_path: Path) -> None:
        flow = self._flows(tmp_path)[0]
        assert flow.sink_line == 5, (
            f"sink should be the execute() call on line 5, got {flow.sink_line}"
        )

    def test_severity_comes_from_extra(self, tmp_path: Path) -> None:
        """semgrep puts severity under `extra`, so a top-level read always
        fell through to the "INFO" default and every flow looked harmless."""
        flow = self._flows(tmp_path)[0]
        assert flow.severity == "ERROR", (
            f"severity should be ERROR from extra.severity, got {flow.severity!r}"
        )

    def test_paths_are_populated(self, tmp_path: Path) -> None:
        flow = self._flows(tmp_path)[0]
        assert flow.source_file == "src/vuln.py"
        assert flow.sink_file == "src/vuln.py"

    def test_rule_id_is_preserved(self, tmp_path: Path) -> None:
        assert self._flows(tmp_path)[0].rule_id == "trelix-test-sql-injection"


class TestParserRobustness:
    """Malformed or unfamiliar shapes must degrade, never crash or fabricate."""

    def _parse(self, tmp_path: Path, payload: object) -> list[TaintFlow]:
        return TaintAnalyzer(str(tmp_path))._parse_semgrep_output(json.dumps(payload))

    def test_empty_output_returns_empty(self, tmp_path: Path) -> None:
        assert TaintAnalyzer(str(tmp_path))._parse_semgrep_output("") == []

    def test_invalid_json_returns_empty(self, tmp_path: Path) -> None:
        assert TaintAnalyzer(str(tmp_path))._parse_semgrep_output("{not json") == []

    def test_result_without_dataflow_trace_still_parses(self, tmp_path: Path) -> None:
        """Non-taint rules have no trace; the match location is the only location."""
        flows = self._parse(tmp_path, {"results": [{
            "check_id": "no-trace-rule",
            "path": "src/x.py",
            "start": {"line": 7},
            "extra": {"severity": "WARNING"},
        }]})
        assert len(flows) == 1
        assert flows[0].source_line == 7 and flows[0].sink_line == 7
        assert flows[0].severity == "WARNING"

    def test_truncated_cliloc_does_not_crash(self, tmp_path: Path) -> None:
        """A ["CliLoc"] with no payload must fall back, not raise."""
        flows = self._parse(tmp_path, {"results": [{
            "check_id": "truncated",
            "path": "src/y.py",
            "start": {"line": 2},
            "extra": {"severity": "ERROR", "dataflow_trace": {"taint_sink": ["CliLoc"]}},
        }]})
        assert len(flows) == 1
        assert flows[0].sink_line == 2

    def test_legacy_dict_shaped_sink_is_still_understood(self, tmp_path: Path) -> None:
        """Older semgrep builds (and the previous fixture) used a location dict.

        Accepting both shapes means the fix cannot regress an environment whose
        semgrep emits the dict form.
        """
        flows = self._parse(tmp_path, {"results": [{
            "check_id": "legacy",
            "path": "src/a.py",
            "start": {"line": 10},
            "extra": {"severity": "ERROR", "dataflow_trace": {
                "taint_sink": {"location": {"path": "src/b.py", "start": {"line": 25}}}
            }},
        }]})
        assert len(flows) == 1
        assert flows[0].sink_file == "src/b.py" and flows[0].sink_line == 25


class TestScanOutcome:
    """A failed scan must never be reportable as a clean scan.

    `run()` returns `[]` for every one of: semgrep not installed, semgrep exited
    non-zero, semgrep reported errors, and semgrep genuinely found nothing. The CLI
    therefore could not distinguish them, and an earlier attempt at fixing the message
    made this worse rather than better: branching on `shutil.which("semgrep")` alone
    turned a FAILED scan into a confident green "semgrep ran and reported nothing" at
    exit 0.

    Reproduced: `trelix taint . --tier intrafile` needs the Semgrep Pro Engine, which
    `pip install trelix[taint]` does not provide. semgrep exits 2 with zero bytes on
    stdout, `_run_semgrep` returns that empty string without checking `returncode`, and
    the parse yields []. A security tool claiming "no findings" for a scan that never
    ran is the worst failure mode available to it.

    `scan()` classifies the outcome from three independent signals, because no single
    one covers every case: the exit code, the `errors[]` array in semgrep's own JSON,
    and `paths.scanned`. An rc-only guard misses the case where semgrep exits non-zero
    but still emits valid JSON, and an errors-only guard misses a hard crash with no
    JSON at all.
    """

    @staticmethod
    def _analyzer(tmp_path):  # type: ignore[no-untyped-def]
        return TaintAnalyzer(str(tmp_path))

    def test_semgrep_missing_is_its_own_outcome(self, tmp_path: Path) -> None:
        from unittest.mock import patch

        from trelix.analysis.taint import ScanOutcome

        with patch("shutil.which", return_value=None):
            result = self._analyzer(tmp_path).scan()

        assert result.outcome is ScanOutcome.SEMGREP_MISSING
        assert result.flows == []

    def test_nonzero_exit_with_empty_stdout_is_a_failed_scan(self, tmp_path: Path) -> None:
        """The reproduced case: --tier intrafile without the Pro engine."""
        from unittest.mock import MagicMock, patch

        from trelix.analysis.taint import ScanOutcome

        completed = MagicMock(returncode=2, stdout="", stderr="Semgrep Pro is uninstalled")
        with patch("shutil.which", return_value="/usr/bin/semgrep"), patch(
            "subprocess.run", return_value=completed
        ):
            result = self._analyzer(tmp_path).scan()

        assert result.outcome is ScanOutcome.SEMGREP_FAILED
        assert "Pro" in result.detail

    def test_json_errors_array_marks_the_scan_failed(self, tmp_path: Path) -> None:
        """Non-zero exit WITH valid JSON — an rc-plus-empty-stdout guard misses this."""
        import json as _json
        from unittest.mock import MagicMock, patch

        from trelix.analysis.taint import ScanOutcome

        payload = _json.dumps({
            "results": [],
            "errors": [{"message": "rule parse error"}],
            "paths": {"scanned": []},
        })
        completed = MagicMock(returncode=8, stdout=payload, stderr="")
        with patch("shutil.which", return_value="/usr/bin/semgrep"), patch(
            "subprocess.run", return_value=completed
        ):
            result = self._analyzer(tmp_path).scan()

        assert result.outcome is ScanOutcome.SEMGREP_FAILED

    def test_zero_files_scanned_is_not_a_clean_scan(self, tmp_path: Path) -> None:
        """Exit 0 and no errors, but nothing was actually examined."""
        import json as _json
        from unittest.mock import MagicMock, patch

        from trelix.analysis.taint import ScanOutcome

        payload = _json.dumps({"results": [], "errors": [], "paths": {"scanned": []}})
        completed = MagicMock(returncode=0, stdout=payload, stderr="")
        with patch("shutil.which", return_value="/usr/bin/semgrep"), patch(
            "subprocess.run", return_value=completed
        ):
            result = self._analyzer(tmp_path).scan()

        assert result.outcome is ScanOutcome.SCANNED_NOTHING

    def test_a_genuine_clean_scan_is_reported_as_clean(self, tmp_path: Path) -> None:
        import json as _json
        from unittest.mock import MagicMock, patch

        from trelix.analysis.taint import ScanOutcome

        payload = _json.dumps({
            "results": [], "errors": [], "paths": {"scanned": ["a.py", "b.py"]},
        })
        completed = MagicMock(returncode=0, stdout=payload, stderr="")
        with patch("shutil.which", return_value="/usr/bin/semgrep"), patch(
            "subprocess.run", return_value=completed
        ):
            result = self._analyzer(tmp_path).scan()

        assert result.outcome is ScanOutcome.OK
        assert result.files_scanned == 2

    def test_findings_are_returned_with_an_ok_outcome(self, tmp_path: Path) -> None:
        import json as _json
        from unittest.mock import MagicMock, patch

        from trelix.analysis.taint import ScanOutcome

        payload = dict(REAL_SEMGREP_TAINT_OUTPUT)
        payload["errors"] = []
        payload["paths"] = {"scanned": ["src/vuln.py"]}
        completed = MagicMock(returncode=1, stdout=_json.dumps(payload), stderr="")
        with patch("shutil.which", return_value="/usr/bin/semgrep"), patch(
            "subprocess.run", return_value=completed
        ):
            result = self._analyzer(tmp_path).scan()

        # semgrep exits 1 when it HAS findings, which must not read as a failure.
        assert result.outcome is ScanOutcome.OK
        assert len(result.flows) == 1
        assert result.flows[0].severity == "ERROR"

    def test_run_still_returns_a_list_and_never_raises(self, tmp_path: Path) -> None:
        """run()'s documented contract must survive scan() being introduced."""
        from unittest.mock import patch

        with patch.object(TaintAnalyzer, "_run_semgrep", side_effect=RuntimeError("boom")):
            assert TaintAnalyzer(str(tmp_path)).run() == []

    def test_failed_scan_is_logged_at_warning(self, tmp_path: Path, caplog) -> None:  # type: ignore[no-untyped-def]
        """A DEBUG-level record is invisible: the CLI configures WARNING."""
        import logging
        from unittest.mock import MagicMock, patch

        completed = MagicMock(returncode=2, stdout="", stderr="Semgrep Pro is uninstalled")
        with caplog.at_level(logging.WARNING, logger="trelix.analysis.taint"):
            with patch("shutil.which", return_value="/usr/bin/semgrep"), patch(
                "subprocess.run", return_value=completed
            ):
                TaintAnalyzer(str(tmp_path)).scan()

        assert any("Pro" in r.message or "Pro" in str(r.args) for r in caplog.records), (
            f"semgrep's stderr was not surfaced at WARNING: {caplog.records}"
        )
