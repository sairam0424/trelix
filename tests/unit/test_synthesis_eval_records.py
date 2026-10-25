"""The per-query records of `trelix eval-synthesis`: `aggregate_synthesis_metrics` over
hand-built records, the `--per-query-out` document and the command that prints the table and
writes the file. Records are built only through `make_record` (tests/unit/
synthesis_eval_fixtures.py), so the next change in this series, which inserts seven fields,
changes that helper's defaults and nothing here.

MUTATIONS THAT MUST MAKE THIS FILE FAIL (each applied alone, sha256-proved restore; results in
the PR body)
---------------------------------------------------------------------------------------------
1. Exclude errored records from the four means ->
   `test_the_means_run_over_answerable_records_and_the_counts_over_all` reads 0.0.
2. `json.dump` instead of `write_outcome_file` -> `test_the_file_is_private` (mode) and
   `test_the_write_is_write_outcome_files_and_its_error_is_returned` (refuse) fail.
3. Drop `"harness": "synthesis"` -> `test_the_document_has_the_four_keys_and_names_its_harness`.
4. Exit 0 on a write error -> `test_an_unwritable_file_exits_1_after_the_table`.
5. Call `run` instead of `run_detailed` in the CLI -> every TestTheCommand test fails (the
   patched `run_detailed` is never reached).
6. Print `unscoreable` in the `Unanswerable queries` row (swapped count keys), 7. drop that
   row's Direction text, 8. print `n_queries + 1`, 9. swap the two count rows -> each fails
   `test_the_table_prints_every_row_with_its_value_and_direction_in_order` (distinct counts).
10. Print `""` instead of the write error -> `test_an_unwritable_file_exits_1_after_the_table`.
"""

from __future__ import annotations

import dataclasses
import io
import json
import re
import stat
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from rich.console import Console
from typer.testing import CliRunner, Result

from tests.unit.synthesis_eval_fixtures import AGGREGATE_KEYS, make_record
from trelix.cli.main import _synthesis_table, app
from trelix.eval import synthesis_records
from trelix.eval.synthesis import SynthesisEvalHarness
from trelix.eval.synthesis_records import (
    SynthesisRecord,
    aggregate_synthesis_metrics,
    write_synthesis_per_query_file,
)
from trelix.store.dimension_guard import DimensionMismatchError

runner = CliRunner()

_ZEROS = {"hallucination": 0.0, "completeness": 0.0, "faithfulness": 0.0, "overall": 0.0}
R = [
    make_record(id="e1", query="a"),
    make_record(id="e2", query="b", error="boom", **{**_ZEROS, "hallucination": 1.0}),
    make_record(id="e3", query="c"),
    make_record(id="e4", query="d", answerable=False, **_ZEROS),
]
# An input to the file writer, never an expected value.
_AGGREGATE = aggregate_synthesis_metrics(R)


def test_the_means_run_over_answerable_records_and_the_counts_over_all() -> None:
    assert aggregate_synthesis_metrics(R) == pytest.approx(
        {
            "hallucination_rate": 1 / 3,
            "completeness": 2 / 3,
            "faithfulness": 2 / 3,
            "overall": 2 / 3,
            "n_queries": 4.0,
            "unscoreable": 1.0,
            "n_unanswerable": 1.0,
        }
    )


def _write(path: Path, records: list[SynthesisRecord] = R) -> dict[str, Any]:
    assert write_synthesis_per_query_file(str(path), records, _AGGREGATE) is None
    return json.loads(path.read_text(encoding="utf-8"))


class TestTheDocument:
    def test_the_document_has_the_four_keys_and_names_its_harness(self, tmp_path: Path) -> None:
        document = _write(tmp_path / "out.json")
        assert set(document) == {"schema_version", "harness", "records", "aggregate"}
        assert document["schema_version"] == 1
        assert document["harness"] == "synthesis"
        assert set(document["aggregate"]) == AGGREGATE_KEYS

    def test_every_record_has_exactly_the_dataclass_fields(self, tmp_path: Path) -> None:
        records = _write(tmp_path / "out.json")["records"]
        names = {f.name for f in dataclasses.fields(SynthesisRecord)}
        assert len(records) == 4
        assert all(set(record) == names for record in records)
        assert records[1]["error"] == "boom"
        assert records[0]["error"] is None

    def test_keys_are_sorted_at_every_level(self, tmp_path: Path) -> None:
        path = tmp_path / "out.json"
        _write(path)
        key_orders: list[list[str]] = []

        def record_order(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            key_orders.append([key for key, _ in pairs])
            return dict(pairs)

        json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=record_order)
        assert len(key_orders) == 6  # the document, its four records and the aggregate
        assert all(order == sorted(order) for order in key_orders)

    def test_non_ascii_text_round_trips_through_ascii_with_one_trailing_newline(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "out.json"
        document = _write(path, [make_record(query="ünïcode")])
        raw = path.read_bytes()
        assert raw.isascii()
        assert raw.endswith(b"}\n") and not raw.endswith(b"\n\n")
        assert document["records"][0]["query"] == "ünïcode"

    def test_the_file_is_private(self, tmp_path: Path) -> None:
        path = tmp_path / "out.json"
        _write(path)
        assert stat.S_IMODE(path.stat().st_mode) == 0o600

    def test_the_write_is_write_outcome_files_and_its_error_is_returned(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[str] = []

        def refuse(path: str, payload: dict[str, Any]) -> str | None:
            calls.append(path)
            return "OSError: disk full"

        monkeypatch.setattr(synthesis_records, "write_outcome_file", refuse)
        target = str(tmp_path / "out.json")
        assert write_synthesis_per_query_file(target, R, _AGGREGATE) == "OSError: disk full"
        assert calls == [target]


def _invoke(tmp_path: Path, *extra: str, run_detailed: Any = None) -> Result:
    """`eval-synthesis REPO --golden G ...` with the harness built as a no-op and `run_detailed`
    returning `R` (or raising what `run_detailed` names)."""
    outcome = {"side_effect": run_detailed} if run_detailed else {"return_value": R}
    args = ["eval-synthesis", str(tmp_path), "--golden", str(tmp_path / "g.jsonl"), *extra]
    with (
        patch.object(SynthesisEvalHarness, "__init__", lambda self, config: None),
        patch.object(SynthesisEvalHarness, "run_detailed", **outcome),
    ):
        return runner.invoke(app, args)


# Seven distinct values, so a row printing another row's key or an off-by-one count cannot pass.
_TABLE_METRICS = {"hallucination_rate": 0.25, "completeness": 0.5, "faithfulness": 0.75}
_TABLE_METRICS |= {"overall": 0.6, "n_queries": 7.0, "unscoreable": 2.0, "n_unanswerable": 3.0}
_TABLE_ROWS = [
    r"Hallucination rate\s*│\s*0\.2500\s*│\s*lower = better",
    r"Completeness\s*│\s*0\.5000\s*│\s*higher = better",
    r"Faithfulness\s*│\s*0\.7500\s*│\s*higher = better",
    r"Overall\s*│\s*0\.6000\s*│\s*higher = better",
    r"Queries evaluated\s*│\s*7\s*│\s*│",
    r"Unanswerable queries\s*│\s*3\s*│\s*not in the four scores above",
    r"Unscoreable queries\s*│\s*2\s*│\s*placeholder scores; see the log",
]


def test_the_table_prints_every_row_with_its_value_and_direction_in_order() -> None:
    buffer = io.StringIO()
    Console(file=buffer, width=120, force_terminal=False).print(_synthesis_table(_TABLE_METRICS))
    text = buffer.getvalue()
    assert "Synthesis Quality Results (GroUSE-style)" in text
    matches = [re.search(row, text) for row in _TABLE_ROWS]
    assert all(matches), text
    starts = [match.start() for match in matches if match]
    assert starts == sorted(starts), text


class TestTheCommand:
    def test_the_table_is_the_same_with_and_without_the_file(self, tmp_path: Path) -> None:
        out = tmp_path / "out.json"
        plain = _invoke(tmp_path)
        with_file = _invoke(tmp_path, "--per-query-out", str(out))
        assert (plain.exit_code, with_file.exit_code) == (0, 0)
        assert plain.output == with_file.output
        for needle in ("Hallucination rate", "Completeness", "Faithfulness", "Overall"):
            assert needle in plain.output
        for needle in ("Queries evaluated", "Unanswerable queries", "Unscoreable queries"):
            assert needle in plain.output
        assert "0.3333" in plain.output and "0.6667" in plain.output
        document = json.loads(out.read_text(encoding="utf-8"))
        assert set(document) == {"schema_version", "harness", "records", "aggregate"}
        assert [record["id"] for record in document["records"]] == ["e1", "e2", "e3", "e4"]

    def test_an_unwritable_file_exits_1_after_the_table(self, tmp_path: Path) -> None:
        result = _invoke(tmp_path, "--per-query-out", "/nonexistent/dir/out.json")
        assert result.exit_code == 1
        assert "Queries evaluated" in result.output
        assert "Could not write per-query results:" in result.output
        assert "FileNotFoundError" in result.output

    def test_help_lists_the_option(self) -> None:
        result = runner.invoke(app, ["eval-synthesis", "--help"])
        assert result.exit_code == 0
        assert "--per-query-out" in re.sub(r"\x1b\[[0-9;]*m", "", result.output)

    def test_a_failure_in_run_detailed_is_a_clean_error_not_a_traceback(
        self, tmp_path: Path
    ) -> None:
        exc = DimensionMismatchError(stored=768, current=1536, provider="openai")
        result = _invoke(tmp_path, run_detailed=exc)
        assert result.exit_code == 1
        assert "Evaluation failed" in result.output
        assert "Traceback" not in result.output
