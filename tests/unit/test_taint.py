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
                            "taint_sink": {
                                "location": {"path": "src/db.py", "start": {"line": 25}}
                            }
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
