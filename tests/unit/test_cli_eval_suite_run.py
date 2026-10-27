"""`trelix eval-suite --arm --out`: the exit codes and the lines of a run through the command.

The run itself (`run_suite`) is tested in `test_eval_suite_run.py` and
`test_eval_suite_run_refusals.py`; `--prepare-only`, the refusals before a run and the cache
directory are in `test_cli_eval_suite.py`. This file drives the command through Typer with a
real suite and a real local repository, nothing built and a stub ranking
(`eval_suite_harness.local_run`).

MUTATIONS THAT MUST MAKE THIS FILE FAIL
---------------------------------------
1. a line of the success output changed or dropped        (test_a_run_prints_what_it_...)
   or the `index:` line not reading the run's dimension   (test_the_dimension_line_...)
2. `_safe_text` dropped from the `results:` line           (test_the_out_path_is_printed_...)
3. the `--out` check moved after the clone                 (test_a_missing_out_directory_...)
4. a run with a query that raised exiting 0                (test_a_query_that_raised_exits_1_...)
5. a missing frozen plan or an existing run directory ending in a traceback or a file, or the
   refusal after a missing plan not naming the run directory it leaves behind
                                                           (test_a_missing_frozen_plan_...,
                                                            test_an_existing_run_directory_...)
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from tests.unit.eval_suite_harness import GOLDEN_SHA256, Remote, invoke
from tests.unit.eval_suite_harness import local_run as local_run
from tests.unit.eval_suite_harness import remote as remote
from tests.unit.eval_suite_harness import suite_json as suite_json
from trelix.eval.results import load_results
from trelix.eval.suite_run import IndexOutcome
from trelix.retrieval.planner.agent import PlanCacheMissError

_ESC = "\x1b"


def _args(tmp_path: Path, suite_json: Path, out: Path | None = None) -> tuple[str, ...]:
    target = out if out is not None else tmp_path / "out" / "results.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    return (
        str(suite_json),
        "--cache-dir",
        str(tmp_path / "cache"),
        "--arm",
        "baseline",
        "--out",
        str(target),
    )


@pytest.mark.usefixtures("local_run")
class TestRun:
    def test_a_run_prints_what_it_verified_built_and_wrote_and_exits_0(
        self, tmp_path: Path, remote: Remote, suite_json: Path
    ) -> None:
        plans_sha256 = hashlib.sha256((tmp_path / "suite" / "plans.jsonl").read_bytes()).hexdigest()
        cache = (tmp_path / "cache").resolve()
        out = tmp_path / "out" / "results.json"
        result = invoke(*_args(tmp_path, suite_json))
        assert result.exit_code == 0
        assert result.stderr == ""
        assert result.stdout == (
            "suite: demo (golden_version v1, license MIT)\n"
            f"repository: {remote.path}\n"
            f"sha: {remote.first}\n"
            f"golden: sha256 {GOLDEN_SHA256}, 2 queries, 2 gold files\n"
            f"plans: sha256 {plans_sha256}, a recorded plan for every golden query\n"
            f"clone: {cache}/clones/{remote.first}/demo\n"
            "arm: baseline\n"
            f"run directory: {cache}/arms/{remote.first}/demo/baseline\n"
            "index: built, embedding dimension 8\n"
            "ndcg@10 0.8155, recall@10 1.0000, mrr 0.7500 over 2 queries; rerank disabled\n"
            f"results: {out}\n"
        )
        assert load_results(out).arm == "baseline"
        assert (cache / "arms" / remote.first / "demo" / "baseline" / "plans.jsonl").is_file()

    def test_the_dimension_line_is_the_one_the_index_build_reported(
        self, tmp_path: Path, suite_json: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`local_run` builds with the harness's `no_index` (looked up when the command runs),
        which reports 8 like every other stub; this one reports 384."""
        monkeypatch.setattr(
            "tests.unit.eval_suite_harness.no_index",
            lambda config: IndexOutcome(errors=0, dimension=384),
        )
        result = invoke(*_args(tmp_path, suite_json))
        assert result.exit_code == 0
        assert "index: built, embedding dimension 384\n" in result.stdout
        assert load_results(tmp_path / "out" / "results.json").embedder.dimension == 384

    def test_the_out_path_is_printed_literally(self, tmp_path: Path, suite_json: Path) -> None:
        out = tmp_path / "out[red]x[/red]" / "results.json"
        result = invoke(*_args(tmp_path, suite_json, out))
        assert result.exit_code == 0
        assert f"results: {out}\n" in result.stdout
        assert "[red]x[/red]" in result.stdout
        assert _ESC not in result.stdout

    def test_a_missing_out_directory_is_refused_before_the_cache_is_touched(
        self, tmp_path: Path, suite_json: Path
    ) -> None:
        out = tmp_path / "nonexistent" / "results.json"
        args = _args(tmp_path, suite_json, tmp_path / "out" / "results.json")
        result = invoke(*args[:-1], str(out))
        assert result.exit_code == 1
        assert result.stdout == ""
        assert result.stderr == (
            f"refused: --out: the directory {out.parent} does not exist; create it before the run\n"
        )
        assert not (tmp_path / "cache").exists()

    def test_an_existing_run_directory_is_refused_and_the_clone_reused(
        self, tmp_path: Path, remote: Remote, suite_json: Path
    ) -> None:
        args = _args(tmp_path, suite_json)
        assert invoke(*args).exit_code == 0
        (tmp_path / "out" / "results.json").unlink()
        result = invoke(*args)
        run_dir = (tmp_path / "cache").resolve() / "arms" / remote.first / "demo" / "baseline"
        assert result.exit_code == 1
        assert result.stdout == ""
        assert result.stderr == (
            f"refused: arm 'baseline' already has a run directory at {run_dir}: choose another "
            "--arm or delete it\n"
        )
        assert not (tmp_path / "out" / "results.json").exists()

    def test_a_query_that_raised_exits_1_after_the_file_is_written(
        self, tmp_path: Path, suite_json: Path, local_run: dict[str, Any]
    ) -> None:
        local_run["what is in the readme"] = RuntimeError("boom")
        result = invoke(*_args(tmp_path, suite_json))
        assert result.exit_code == 1
        assert f"results: {tmp_path / 'out' / 'results.json'}\n" in result.stdout
        assert "1 of 2 queries raised during retrieval" in result.stderr
        assert "q0002: boom" in result.stderr
        assert load_results(tmp_path / "out" / "results.json").records[1].error == "boom"

    def test_a_missing_frozen_plan_is_a_refusal_naming_the_run_directory_and_no_file(
        self, tmp_path: Path, remote: Remote, suite_json: Path, local_run: dict[str, Any]
    ) -> None:
        """The operator fixes plans.jsonl and re-runs the same `--arm`: without the second line
        the next run would be refused for a run directory this one never mentioned."""
        local_run["what is in the readme"] = PlanCacheMissError("no frozen plan for it")
        result = invoke(*_args(tmp_path, suite_json))
        run_dir = (tmp_path / "cache").resolve() / "arms" / remote.first / "demo" / "baseline"
        assert result.exit_code == 1
        assert isinstance(result.exception, SystemExit)
        assert result.stdout == ""
        assert result.stderr == (
            "refused: a golden query had no frozen plan at run time, so no results.json was "
            "written: no frozen plan for it\n"
            f"refused: the run directory {run_dir} is left as it is; delete it before running "
            "this arm again\n"
        )
        assert not (tmp_path / "out" / "results.json").exists()
        assert (run_dir / "plans.jsonl").is_file()
