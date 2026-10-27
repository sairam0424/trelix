"""`run_suite`: what refuses a run, and the git isolation of the whole run.

Nothing is built here (`index_fn` is injected) and the ranking comes from a stub retriever
under the real `EvalHarness.run_detailed`; see `eval_suite_harness.stub_run`. The results file
itself is in `test_eval_suite_run.py`, the real Indexer in `test_eval_suite_run_indexer.py`.

MUTATIONS THAT MUST MAKE THIS FILE FAIL
---------------------------------------
1. `exist_ok=True` on the run directory            (test_an_existing_run_directory_is_refused_...)
2. `errors > 0` tolerated                          (test_an_index_error_is_refused_and_no_file_...)
3. a `PlanCacheMissError` scored as a miss, or a file written for it, or let out as itself
                                                  (test_a_missing_frozen_plan_stops_the_run_...)
4. the `--out` or `--arm` check moved after the clone, the `--arm` match relaxed to a prefix
   match, or the writability check of `--out` dropped
                                                  (test_out_is_checked_before_the_cache_is_touched,
                                                   test_an_out_that_is_a_directory_...,
                                                   test_an_out_whose_directory_is_not_writable_...,
                                                   test_an_arm_that_is_not_a_label_...)
5. the input bytes not re-hashed before the claim  (test_a_committed_file_that_changed_...)
6. a refusal after the claim not ending with the run-directory line, or an `OSError` of the
   input copies or an `OSError`/`ImportError` of the index build ending in a traceback, or the
   I/O error `write_results` returns for the results file ignored
                                                  (TestAfterTheClaim)
7. one of the seven isolation variables omitted, or one of the five repository variables not
   removed, or the restore removed, or the restore skipped when the run raises
                                                  (TestGitIsolation)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from tests.unit.eval_suite_harness import (
    RANKED,
    jsonl,
    plan_record,
    stub_harness_factory,
    stub_run,
)
from tests.unit.eval_suite_harness import remote as remote
from tests.unit.eval_suite_harness import spec as spec
from trelix.core.config import IndexConfig
from trelix.eval.suite import SuiteError, SuiteSpec
from trelix.eval.suite_run import IndexOutcome
from trelix.retrieval.planner.agent import PlanCacheMissError

_NULL = os.devnull


def _left_behind(tmp_path: Path, spec: SuiteSpec) -> str:
    run_dir = tmp_path / "cache" / "arms" / spec.repo_sha / "demo" / "baseline"
    return f"the run directory {run_dir} is left as it is; delete it before running this arm again"


class TestRefusals:
    def test_an_existing_run_directory_is_refused_with_its_path_and_nothing_is_written(
        self, tmp_path: Path, spec: SuiteSpec
    ) -> None:
        stub_run(tmp_path, spec)
        run_dir = tmp_path / "cache" / "arms" / spec.repo_sha / "demo" / "baseline"
        (tmp_path / "out" / "results.json").unlink()
        with pytest.raises(SuiteError) as caught:
            stub_run(tmp_path, spec)
        assert caught.value.problems == (
            f"arm 'baseline' already has a run directory at {run_dir}: choose another --arm "
            "or delete it",
        )
        assert not (tmp_path / "out" / "results.json").exists()
        assert (run_dir / "plans.jsonl").exists()
        assert stub_run(tmp_path, spec, arm="flag-on").out.is_file()

    def test_an_index_error_is_refused_and_no_file_is_written(
        self, tmp_path: Path, spec: SuiteSpec
    ) -> None:
        clone = tmp_path / "cache" / "clones" / spec.repo_sha / "demo"
        with pytest.raises(SuiteError) as caught:
            stub_run(tmp_path, spec, index_fn=lambda config: IndexOutcome(errors=1, dimension=8))
        first, second = caught.value.problems
        assert first == (
            f"indexing {clone} reported 1 error(s): a run with a parse or write error is not a "
            "measurement, so no results were written"
        )
        assert second == _left_behind(tmp_path, spec)
        assert not (tmp_path / "out" / "results.json").exists()

    def test_a_missing_frozen_plan_stops_the_run_and_writes_no_file(
        self, tmp_path: Path, spec: SuiteSpec
    ) -> None:
        ranked = {**RANKED, "what is in the readme": PlanCacheMissError("no frozen plan")}
        with pytest.raises(SuiteError) as caught:
            stub_run(tmp_path, spec, harness_factory=stub_harness_factory(ranked))
        assert caught.value.problems == (
            "a golden query had no frozen plan at run time, so no results.json was written: "
            "no frozen plan",
            _left_behind(tmp_path, spec),
        )
        assert isinstance(caught.value.__cause__, PlanCacheMissError)
        assert not (tmp_path / "out" / "results.json").exists()

    def test_out_is_checked_before_the_cache_is_touched(
        self, tmp_path: Path, spec: SuiteSpec
    ) -> None:
        with pytest.raises(SuiteError) as caught:
            stub_run(tmp_path, spec, out=tmp_path / "nonexistent" / "x.json")
        assert caught.value.problems == (
            f"--out: the directory {tmp_path / 'nonexistent'} does not exist; create it before "
            "the run",
        )
        assert not (tmp_path / "cache").exists()

    def test_an_out_that_already_exists_is_refused_before_the_cache_is_touched(
        self, tmp_path: Path, spec: SuiteSpec
    ) -> None:
        """MUTATION that must make this fail: drop the `out.exists()` check (file replaced)."""
        existing = tmp_path / "old.json"
        existing.write_text("{}", encoding="utf-8")
        with pytest.raises(SuiteError) as caught:
            stub_run(tmp_path, spec, out=existing)
        assert caught.value.problems == (
            f"--out {existing} already exists: each arm needs its own results file",
        )
        assert existing.read_text(encoding="utf-8") == "{}"
        assert not (tmp_path / "cache").exists()

    def test_an_out_that_is_a_directory_is_refused_before_the_cache_is_touched(
        self, tmp_path: Path, spec: SuiteSpec
    ) -> None:
        with pytest.raises(SuiteError) as caught:
            stub_run(tmp_path, spec, out=tmp_path / "out")
        assert caught.value.problems == (
            f"--out {tmp_path / 'out'} is a directory: name the results file itself",
        )
        assert not (tmp_path / "cache").exists()

    @pytest.mark.parametrize("arm", ["Bad Arm", "baseline x", "a/../b"])
    def test_an_arm_that_is_not_a_label_is_refused_before_the_cache_is_touched(
        self, tmp_path: Path, spec: SuiteSpec, arm: str
    ) -> None:
        """`baseline x` and `a/../b` begin with a valid label: a prefix match would accept them
        and claim the run directory `arms/<sha>/<name>/a/../b` after the clone."""
        with pytest.raises(SuiteError) as caught:
            stub_run(tmp_path, spec, arm=arm)
        assert caught.value.problems == (
            f"--arm must match [a-z0-9][a-z0-9_-]{{0,62}} (got {arm!r})",
        )
        assert not (tmp_path / "cache").exists()

    def test_an_out_whose_directory_is_not_writable_is_refused_before_the_cache_is_touched(
        self, tmp_path: Path, spec: SuiteSpec
    ) -> None:
        parent = tmp_path / "readonly"
        parent.mkdir()
        parent.chmod(0o500)
        try:
            if os.access(parent, os.W_OK):
                pytest.skip("running as root: every directory is writable")
            with pytest.raises(SuiteError) as caught:
                stub_run(tmp_path, spec, out=parent / "results.json")
        finally:
            parent.chmod(0o700)
        assert caught.value.problems == (f"--out: cannot write in {parent}",)
        assert not (tmp_path / "cache").exists()

    def test_a_committed_file_that_changed_after_loading_is_refused_before_the_claim(
        self, tmp_path: Path, spec: SuiteSpec
    ) -> None:
        """Still covering every query, so the pre-flight passes and only the hash can refuse."""
        queries = ["how does login work", "what is in the readme", "one more"]
        spec.plans_file.write_bytes(jsonl([plan_record(q) for q in queries]))
        with pytest.raises(SuiteError) as caught:
            stub_run(tmp_path, spec)
        [problem] = caught.value.problems
        assert problem.startswith("plans: sha256 of plans.jsonl is now ")
        assert not (tmp_path / "cache" / "arms").exists()


class TestAfterTheClaim:
    """Every failure once the run directory exists says so; nothing before the claim does."""

    def test_an_os_error_while_the_index_is_built_is_a_refusal_naming_the_run_directory(
        self, tmp_path: Path, spec: SuiteSpec
    ) -> None:
        def full(config: IndexConfig) -> IndexOutcome:
            raise OSError(28, "No space left on device")

        with pytest.raises(SuiteError) as caught:
            stub_run(tmp_path, spec, index_fn=full)
        assert caught.value.problems == (
            "cannot build or open the index: [Errno 28] No space left on device",
            _left_behind(tmp_path, spec),
        )
        assert not (tmp_path / "out" / "results.json").exists()

    def test_a_missing_local_extra_is_a_refusal_not_a_traceback(
        self, tmp_path: Path, spec: SuiteSpec
    ) -> None:
        """`make_embedder` raises ImportError when sentence-transformers is not installed."""

        def no_extra(config: IndexConfig) -> IndexOutcome:
            raise ImportError("No module named 'sentence_transformers'")

        with pytest.raises(SuiteError) as caught:
            stub_run(tmp_path, spec, index_fn=no_extra)
        assert caught.value.problems == (
            "cannot build or open the index: No module named 'sentence_transformers'",
            _left_behind(tmp_path, spec),
        )

    def test_a_copy_that_cannot_be_written_is_a_refusal_naming_the_file(
        self, tmp_path: Path, spec: SuiteSpec, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_dir = tmp_path / "cache" / "arms" / spec.repo_sha / "demo" / "baseline"
        built: list[IndexConfig] = []

        def write_bytes(path: Path, data: bytes) -> int:
            raise OSError(28, f"No space left on device: '{path}'")

        def index_fn(config: IndexConfig) -> IndexOutcome:
            built.append(config)
            return IndexOutcome(errors=0, dimension=8)

        monkeypatch.setattr(Path, "write_bytes", write_bytes)
        with pytest.raises(SuiteError) as caught:
            stub_run(tmp_path, spec, index_fn=index_fn)
        assert caught.value.problems == (
            f"cannot write {run_dir / 'golden.jsonl'}: [Errno 28] No space left on device: "
            f"'{run_dir / 'golden.jsonl'}'",
            _left_behind(tmp_path, spec),
        )
        assert built == []
        assert not (tmp_path / "out" / "results.json").exists()

    def test_a_results_file_that_cannot_be_written_is_a_refusal_naming_the_file(
        self, tmp_path: Path, spec: SuiteSpec, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`write_results` returns its I/O error instead of raising (a full disk, an `--out`
        directory removed while the index was built); the run must not end as a success that
        names a file which is not there."""
        monkeypatch.setattr(
            "trelix.eval.suite_run.write_results",
            lambda path, doc: "[Errno 28] No space left on device",
        )
        out = tmp_path / "out" / "results.json"
        with pytest.raises(SuiteError) as caught:
            stub_run(tmp_path, spec)
        assert caught.value.problems == (
            f"cannot write {out}: [Errno 28] No space left on device",
            _left_behind(tmp_path, spec),
        )
        assert not out.exists()

    def test_a_refusal_before_the_claim_has_no_run_directory_line(
        self, tmp_path: Path, spec: SuiteSpec
    ) -> None:
        with pytest.raises(SuiteError) as caught:
            stub_run(tmp_path, spec, arm="Bad Arm")
        assert len(caught.value.problems) == 1
        assert "run directory" not in caught.value.problems[0]


_ISOLATION = [
    ("GIT_CONFIG_GLOBAL", _NULL),
    ("GIT_CONFIG_SYSTEM", _NULL),
    ("GIT_CONFIG_NOSYSTEM", "1"),
    ("GIT_ALLOW_PROTOCOL", "https"),
    ("GIT_TERMINAL_PROMPT", "0"),
    ("GIT_LFS_SKIP_SMUDGE", "1"),
    ("GIT_OPTIONAL_LOCKS", "0"),
]
_REPOSITORY = [
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_COMMON_DIR",
    "GIT_OBJECT_DIRECTORY",
]


class TestGitIsolation:
    @pytest.mark.parametrize(("name", "value"), _ISOLATION)
    def test_the_variable_is_set_while_the_index_is_built_and_the_queries_run_then_removed(
        self,
        tmp_path: Path,
        spec: SuiteSpec,
        monkeypatch: pytest.MonkeyPatch,
        name: str,
        value: str,
    ) -> None:
        monkeypatch.delenv(name, raising=False)
        seen: dict[str, dict[str, str]] = {}

        def index_fn(config: IndexConfig) -> IndexOutcome:
            seen["index"] = dict(os.environ)
            return IndexOutcome(errors=0, dimension=8)

        class Capturing(stub_harness_factory()):  # type: ignore[misc]
            def run_detailed(self, golden_path: str, **kwargs: Any) -> Any:
                seen["retrieve"] = dict(os.environ)
                return super().run_detailed(golden_path, **kwargs)

        stub_run(tmp_path, spec, index_fn=index_fn, harness_factory=Capturing)
        assert seen["index"][name] == value
        assert seen["retrieve"][name] == value
        assert name not in os.environ

    def test_a_variable_that_was_set_before_gets_its_value_back(
        self, tmp_path: Path, spec: SuiteSpec, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GIT_TERMINAL_PROMPT", "canary")
        monkeypatch.setenv("GIT_CONFIG_GLOBAL", "canary-global")
        seen: dict[str, str] = {}

        def index_fn(config: IndexConfig) -> IndexOutcome:
            seen.update(os.environ)
            return IndexOutcome(errors=0, dimension=8)

        stub_run(tmp_path, spec, index_fn=index_fn)
        assert seen["GIT_TERMINAL_PROMPT"] == "0"
        assert seen["GIT_CONFIG_GLOBAL"] == _NULL
        assert os.environ["GIT_TERMINAL_PROMPT"] == "canary"
        assert os.environ["GIT_CONFIG_GLOBAL"] == "canary-global"

    def test_the_environment_is_restored_when_the_run_raises(
        self, tmp_path: Path, spec: SuiteSpec, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("GIT_CONFIG_NOSYSTEM", raising=False)
        monkeypatch.setenv("GIT_DIR", "canary-dir")
        with pytest.raises(SuiteError):
            stub_run(tmp_path, spec, index_fn=lambda config: IndexOutcome(errors=1, dimension=8))
        assert "GIT_CONFIG_NOSYSTEM" not in os.environ
        assert os.environ["GIT_DIR"] == "canary-dir"

    @pytest.mark.parametrize("name", _REPOSITORY)
    def test_an_exported_repository_variable_is_removed_for_the_run_and_put_back(
        self, tmp_path: Path, spec: SuiteSpec, monkeypatch: pytest.MonkeyPatch, name: str
    ) -> None:
        """`trelix.store.provenance` runs `git` with `cwd=` and no `-C`, so an exported `GIT_DIR`
        would make the index's provenance rows describe another repository."""
        monkeypatch.setenv(name, "canary")
        seen: dict[str, dict[str, str]] = {}

        def index_fn(config: IndexConfig) -> IndexOutcome:
            seen["index"] = dict(os.environ)
            return IndexOutcome(errors=0, dimension=8)

        class Capturing(stub_harness_factory()):  # type: ignore[misc]
            def run_detailed(self, golden_path: str, **kwargs: Any) -> Any:
                seen["retrieve"] = dict(os.environ)
                return super().run_detailed(golden_path, **kwargs)

        stub_run(tmp_path, spec, index_fn=index_fn, harness_factory=Capturing)
        assert name not in seen["index"]
        assert name not in seen["retrieve"]
        assert os.environ[name] == "canary"
