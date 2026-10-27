"""`run_suite` with the real `Indexer`, `Retriever` and `EvalHarness` (over `FakeEmbedder`).

The other `run_suite` tests (`test_eval_suite_run.py`, `test_eval_suite_run_refusals.py`)
inject `index_fn` and a stub retriever. The tests here drive the real wiring of a configuration
built by `run_config`, so it is exercised by something other than a stub: two independent
builds of the same index agree record for record; nothing from the clone runs, nothing is
written into the clone, and a tracked symlink that leaves the clone is not indexed; and the
default `index_fn` (`index_repo`) reports what the Indexer built, not a constant.

The real `Chunker` loads tiktoken's `cl100k_base` encoding, which tiktoken downloads unless it
is cached: offline (the hermetic wrapper, pytest-socket) the cache must already hold it, under
`TIKTOKEN_CACHE_DIR` or `<TMPDIR>/data-gym-cache/`.

MUTATIONS THAT MUST MAKE THIS FILE FAIL
---------------------------------------
1. parsed files inserted in completion order, if that ever changes a file-level top 10
                                                 (test_two_independent_builds_give_identical_records)
2. the index written inside the clone (`store.db_path` under `repo_path`), or
   `walker.follow_symlinks` not forced, or `GIT_CONFIG_GLOBAL`/`GIT_CONFIG_NOSYSTEM` missing from
   the environment of the Indexer's own git call, or an exported `GIT_DIR` reaching it
                                                 (test_nothing_from_the_clone_runs_...)
3. `index_repo` reporting a constant dimension instead of the Indexer's embedder's (8 is pinned
   by test_nothing_from_the_clone_runs_..., 16 here)
                                                 (test_the_dimension_is_the_indexers_embedders_...)
4. `index_repo` not reading `stats["errors"]` from what `Indexer.index` returned
                                                 (test_an_error_count_the_indexer_reports_...)
"""

from __future__ import annotations

import os
import runpy
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from tests.unit.eval_suite_harness import (
    FIRST_FILES,
    clock,
    fake_embedders,
    git,
    load_local,
    make_remote,
    out_file,
    record_provenance_git,
    run_local,
    write_suite,
)
from tests.unit.eval_suite_harness import remote as remote
from tests.unit.eval_suite_harness import spec as spec
from trelix.core.config import IndexConfig
from trelix.eval.harness import EvalHarness
from trelix.eval.results import load_results
from trelix.eval.suite import SuiteError, SuiteSpec
from trelix.indexing.indexer import Indexer

_NULL = os.devnull
# The canary the symlinked file holds; the word a leaked chunk would contain.
_CANARY_WORD = "CANARY_" + "SECRET"
_CANARY_FILE = f'def canary():\n    return "{_CANARY_WORD}"\n'
_LOGIN = "def login(user):\n    return user\n"
_TWINS = {
    **{f"src/other{n}.py": f"def other{n}():\n    return {n}\n" for n in range(1, 9)},
    "src/a.py": _LOGIN,
    "src/b.py": _LOGIN,
    "README.md": "demo\n",
}
_TWIN_GOLDEN = [
    {"query": "how does login work", "relevant_files": ["src/a.py"]},
    {"query": "what is in the readme", "relevant_files": ["README.md"]},
]


class TestTheRealIndexer:
    def test_two_independent_builds_give_identical_records(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`src/a.py` and `src/b.py` are byte-identical, so their BM25 and vector scores tie and
        only the insertion order (parsed with four workers, inserted as they complete) could
        separate them. Five pairs of arms, each pair with its own index builds."""
        fake_embedders(monkeypatch)
        twins = make_remote(tmp_path, _TWINS, {"src/extra.py": "X = 1\n"}, name="twins")
        spec = load_local(
            write_suite(tmp_path / "suite", str(twins.path), twins.first, golden=_TWIN_GOLDEN)
        )
        seen: list[IndexConfig] = []

        def factory(config: IndexConfig) -> EvalHarness:
            seen.append(config)
            return EvalHarness(config)

        for repetition in range(5):
            cache = tmp_path / f"cache{repetition}"
            one, two = (
                run_local(
                    spec,
                    arm=arm,
                    cache_root=cache,
                    out=out_file(tmp_path, f"{repetition}-{arm}.json"),
                    harness_factory=factory,
                )
                for arm in ("one", "two")
            )
            assert one.records == two.records, repetition
            assert [r.error for r in one.records] == [None, None]
            assert all(r.top10 for r in one.records)
        assert all(config.parse_workers == 4 for config in seen)
        assert len(seen) == 10

    def test_nothing_from_the_clone_runs_the_clone_stays_pristine_and_the_canary_is_not_indexed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The pinned commit holds `conftest.py`, `setup.py` and `sitecustomize.py`, each writing
        `EXECUTED` beside itself, and `src/leak.py`, a symlink to a file outside the clone that
        holds the canary word. The index goes into the run directory; the clone is untouched.
        `GIT_DIR` is exported, as a git hook's shell has it; the Indexer's git must not see it."""
        monkeypatch.setenv("GIT_DIR", str(tmp_path / "elsewhere" / ".git"))
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "canary.py").write_text(_CANARY_FILE, encoding="utf-8")
        planted = make_remote(
            tmp_path,
            FIRST_FILES,
            {"src/extra.py": "X = 1\n"},
            links={"src/leak.py": outside / "canary.py"},
        )
        control = tmp_path / "control"
        control.mkdir()
        (control / "setup.py").write_text(FIRST_FILES["setup.py"], encoding="utf-8")
        runpy.run_path(str(control / "setup.py"))
        assert (control / "EXECUTED").exists(), "control: the canary did not fire when run"
        (control / "EXECUTED").unlink()

        fake_embedders(monkeypatch)
        calls = record_provenance_git(monkeypatch)
        spec = load_local(write_suite(tmp_path / "suite", str(planted.path), planted.first))
        run = run_local(
            spec,
            arm="baseline",
            cache_root=tmp_path / "cache",
            out=out_file(tmp_path),
            now=clock("2026-10-05T12:00:00+00:00"),
        )

        clone = run.prepared.clone
        assert not (clone / "EXECUTED").exists()
        assert (clone / "src" / "leak.py").is_symlink()
        assert git(clone, "status", "--porcelain", "--untracked-files=all", "--ignored") == ""
        assert not (clone / ".trelix").exists()
        assert (run.run_dir / "index.db").is_file()
        with sqlite3.connect(run.run_dir / "index.db") as db:
            paths = {row[0] for row in db.execute("SELECT rel_path FROM files")}
            leaked = db.execute(
                "SELECT count(*) FROM chunks WHERE chunk_text LIKE ?", (f"%{_CANARY_WORD}%",)
            ).fetchone()[0]
        assert "src/app.py" in paths
        assert "src/leak.py" not in paths
        assert leaked == 0
        assert [r.repo for r in run.records] == ["demo", "demo"]
        assert [r.error for r in run.records] == [None, None]
        assert run.dimension == 8
        assert calls, "the Indexer made no git call: nothing to check"
        for call in calls:
            assert call.env["GIT_CONFIG_GLOBAL"] == _NULL
            assert call.env["GIT_CONFIG_NOSYSTEM"] == "1"
            assert "GIT_DIR" not in call.env
        assert os.environ["GIT_DIR"] == str(tmp_path / "elsewhere" / ".git")
        assert load_results(run.out).arm == "baseline"


class TestWhatIndexRepoReads:
    """`index_repo`, the default `index_fn`, reads two things off the real Indexer: the width of
    its embedder and the error count of the build. Every other test here embeds in 8 dimensions
    and builds without an error, so a constant in either place would pass them."""

    def test_the_dimension_is_the_indexers_embedders_not_a_constant(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spec: SuiteSpec
    ) -> None:
        fake_embedders(monkeypatch, dimension=16)
        run = run_local(spec, arm="wide", cache_root=tmp_path / "cache", out=out_file(tmp_path))
        assert run.dimension == 16
        assert load_results(run.out).embedder.dimension == 16
        assert [r.error for r in run.records] == [None, None]

    def test_an_error_count_the_indexer_reports_refuses_the_run_and_writes_no_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spec: SuiteSpec
    ) -> None:
        """The real build runs to the end (the index is there) and its `stats` come back with
        `errors` raised to 2, as a parse or write failure would leave them."""
        fake_embedders(monkeypatch)
        real_index = Indexer.index

        def two_errors(self: Indexer) -> dict[str, Any]:
            return {**real_index(self), "errors": 2}

        monkeypatch.setattr(Indexer, "index", two_errors)
        clone = tmp_path / "cache" / "clones" / spec.repo_sha / "demo"
        run_dir = tmp_path / "cache" / "arms" / spec.repo_sha / "demo" / "baseline"
        with pytest.raises(SuiteError) as caught:
            run_local(spec, arm="baseline", cache_root=tmp_path / "cache", out=out_file(tmp_path))
        assert caught.value.problems == (
            f"indexing {clone} reported 2 error(s): a run with a parse or write error is not a "
            "measurement, so no results were written",
            f"the run directory {run_dir} is left as it is; delete it before running this arm "
            "again",
        )
        assert (run_dir / "index.db").is_file()
        assert not (tmp_path / "out" / "results.json").exists()
