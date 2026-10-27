"""`trelix eval-suite`: the exit codes and the lines of `--prepare-only` and of the refusals
before a run.

The checks themselves are in `test_eval_suite_spec.py`, `test_eval_suite_clone.py`,
`test_eval_suite_git_isolation.py`, `test_eval_suite_gold.py` and `test_eval_suite_run.py`, and
a run through the command is in `test_cli_eval_suite_run.py`; this file drives the command
through Typer with a real suite and a real local repository.

MUTATIONS THAT MUST MAKE THIS FILE FAIL
---------------------------------------
1. the command hidden from `--help` again, or `--arm`/`--out` missing from its own
                                                             (test_the_command_is_listed_...)
2. a run accepted without `--arm` or without `--out`, or that message changed; `--prepare-only`
   accepted together with `--arm` or `--out`
                                                             (TestWithoutArmAndOut,
                                                              test_prepare_only_with_arm_or_out_...)
3. a refusal exiting 0, or a reason printed to stdout        (TestRefusals)
4. only the first reason printed                              (test_every_reason_is_its_own_line)
5. `_safe_text` dropped from the lines                        (test_a_path_is_printed_literally)
6. `--cache-dir` ignored, not resolved, or an empty one taken as the current directory; the
   default root not used or its rules changed; a cache directory that cannot be created ending
   in a traceback                              (TestCacheDirectory, TestDefaultCacheRoot)
7. a line of the success output changed or dropped            (test_prepare_only_prints_what_...)
8. the command passing `allow_local=True`, or the host check skipped before a clone
                                                          (TestAnyOtherRepositoryIsRefused)
9. an error the walker raises on the clone's own files ending in a traceback
                                                       (test_a_gitignore_the_walker_cannot_...)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tests.unit.eval_suite_harness import (
    FIRST_FILES,
    GOLDEN_SHA256,
    Remote,
    invoke,
    make_remote,
    record_git,
    write_suite,
)
from tests.unit.eval_suite_harness import remote as remote
from tests.unit.eval_suite_harness import suite_json as suite_json
from tests.unit.eval_validate_harness import plain
from trelix.cli.main import app
from trelix.eval import suite_prepare
from trelix.eval.suite import SuiteError
from trelix.eval.suite_git import default_cache_root
from trelix.eval.suite_prepare import PreparedSuite

_ESC = "\x1b"
_REQUIRED = (
    "refused: --arm and --out are required to run a suite; pass --prepare-only to verify it "
    "without running\n"
)


@pytest.fixture
def local_repositories(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let the command prepare a suite whose repository is a local path, as only the tests may.

    The command never sets `allow_local`; this stands in for the Python API call that does.
    """
    real = suite_prepare.prepare_suite

    def prepare_locally(
        suite_file: Path | str, cache_dir: Path | str | None = None
    ) -> PreparedSuite:
        return real(suite_file, cache_dir, allow_local=True)

    monkeypatch.setattr("trelix.eval.suite_prepare.prepare_suite", prepare_locally)


@pytest.mark.usefixtures("local_repositories")
class TestPrepareOnly:
    def test_prepare_only_prints_what_it_verified_and_exits_0(
        self, tmp_path: Path, remote: Remote, suite_json: Path
    ) -> None:
        plans_sha256 = hashlib.sha256((tmp_path / "suite" / "plans.jsonl").read_bytes()).hexdigest()
        result = invoke(str(suite_json), "--cache-dir", str(tmp_path / "cache"), "--prepare-only")
        assert result.exit_code == 0
        assert result.stderr == ""
        assert result.stdout == (
            "suite: demo (golden_version v1, license MIT)\n"
            f"repository: {remote.path}\n"
            f"sha: {remote.first}\n"
            f"golden: sha256 {GOLDEN_SHA256}, 2 queries, 2 gold files\n"
            f"plans: sha256 {plans_sha256}, a recorded plan for every golden query\n"
            f"clone: {(tmp_path / 'cache').resolve()}/clones/{remote.first}/demo\n"
            "prepared: every check passed; nothing was indexed or run\n"
        )

    def test_a_second_run_reuses_the_clone_and_exits_0(
        self, tmp_path: Path, suite_json: Path
    ) -> None:
        args = (str(suite_json), "--cache-dir", str(tmp_path / "cache"), "--prepare-only")
        assert invoke(*args).exit_code == 0
        assert invoke(*args).exit_code == 0

    def test_a_path_is_printed_literally(self, tmp_path: Path, suite_json: Path) -> None:
        """Markup in a path is escaped and control bytes are dropped, as in `eval-compare`."""
        cache = tmp_path / "cache[red]x"
        result = invoke(str(suite_json), "--cache-dir", str(cache), "--prepare-only")
        assert result.exit_code == 0
        assert f"clone: {cache.resolve()}/clones/" in result.stdout
        assert _ESC not in result.stdout

    @pytest.mark.parametrize(
        "given",
        [
            ("--arm", "baseline"),
            ("--out", "results.json"),
            ("--arm", "baseline", "--out", "r.json"),
        ],
        ids=["arm", "out", "both"],
    )
    def test_prepare_only_with_arm_or_out_is_refused_and_nothing_is_cloned(
        self, tmp_path: Path, suite_json: Path, given: tuple[str, ...]
    ) -> None:
        """A user who typed all three meant one of two things; neither is guessed."""
        args = (str(suite_json), "--cache-dir", str(tmp_path / "cache"), "--prepare-only", *given)
        result = invoke(*args)
        assert result.exit_code == 1
        assert result.stdout == ""
        assert result.stderr == (
            "refused: --prepare-only runs nothing and takes no --arm or --out: drop it to run "
            "the arm, or drop them to verify the suite\n"
        )
        assert not (tmp_path / "cache").exists()


class TestWithoutArmAndOut:
    @pytest.mark.parametrize(
        "given",
        [(), ("--arm", "baseline"), ("--out", "results.json")],
        ids=["neither", "arm-only", "out-only"],
    )
    def test_a_run_needs_both_and_nothing_is_cloned_without_them(
        self, tmp_path: Path, suite_json: Path, given: tuple[str, ...]
    ) -> None:
        result = invoke(str(suite_json), "--cache-dir", str(tmp_path / "cache"), *given)
        assert result.exit_code == 1
        assert result.stdout == ""
        assert result.stderr == _REQUIRED
        assert not (tmp_path / "cache").exists()


class TestAnyOtherRepositoryIsRefused:
    """No fixture here: the command as a user runs it, which never allows a local repository."""

    _MESSAGE = (
        "refused: repo.url must be https://<host>/<owner>/<repo>[.git] with <host> one of "
        "github.com, and no credentials, port, query or fragment (got "
    )

    def test_a_local_repository_is_refused_and_nothing_is_cloned(
        self, tmp_path: Path, suite_json: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = record_git(monkeypatch)
        result = invoke(str(suite_json), "--cache-dir", str(tmp_path / "cache"), "--prepare-only")
        assert result.exit_code == 1
        assert result.stdout == ""
        assert result.stderr.startswith(self._MESSAGE)
        assert calls == []
        assert not (tmp_path / "cache").exists()

    @pytest.mark.parametrize(
        "url",
        [
            "https://169.254.169.254/latest/meta-data",
            "https://localhost/o/r",
            "http://github.com/o/r",
            "https://github.com.evil.example/o/r",
        ],
    )
    def test_a_url_on_another_host_is_refused_before_any_git_command(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, url: str
    ) -> None:
        calls = record_git(monkeypatch)
        path = write_suite(tmp_path / "suite", url, "1" * 40)
        result = invoke(str(path), "--cache-dir", str(tmp_path / "cache"), "--prepare-only")
        assert result.exit_code == 1
        assert result.stdout == ""
        assert result.stderr.startswith(self._MESSAGE)
        assert calls == []
        assert not (tmp_path / "cache").exists()


class TestRefusals:
    def test_a_missing_suite_file_is_refused(self, tmp_path: Path) -> None:
        result = invoke(str(tmp_path / "absent.json"), "--prepare-only")
        assert result.exit_code == 1
        assert result.stdout == ""
        assert result.stderr.startswith("refused: cannot read suite.json: ")

    @pytest.mark.usefixtures("local_repositories")
    def test_a_hash_mismatch_is_refused_with_both_digests_and_clones_nothing(
        self, tmp_path: Path, suite_json: Path
    ) -> None:
        (tmp_path / "suite" / "golden.jsonl").write_bytes(b"x\n")
        other = hashlib.sha256(b"x\n").hexdigest()
        result = invoke(str(suite_json), "--cache-dir", str(tmp_path / "cache"), "--prepare-only")
        assert result.exit_code == 1
        assert result.stdout == ""
        assert result.stderr == (
            f"refused: golden: sha256 of golden.jsonl is {other} but suite.json says "
            f"{GOLDEN_SHA256}\n"
        )
        assert not (tmp_path / "cache").exists()

    @pytest.mark.usefixtures("local_repositories")
    def test_every_reason_is_its_own_line(self, tmp_path: Path, suite_json: Path) -> None:
        doc = json.loads(suite_json.read_text())
        doc["name"] = "Bad Name"
        doc["repo"]["sha"] = "abc"
        suite_json.write_text(json.dumps(doc), encoding="utf-8")
        result = invoke(str(suite_json), "--cache-dir", str(tmp_path / "cache"), "--prepare-only")
        assert result.exit_code == 1
        lines = result.stderr.splitlines()
        assert len(lines) == 2
        assert all(line.startswith("refused: ") for line in lines)
        assert "name must match" in lines[0]
        assert "repo.sha" in lines[1]

    @pytest.mark.usefixtures("local_repositories")
    def test_a_clone_that_is_not_pristine_is_refused_and_left_alone(
        self, tmp_path: Path, remote: Remote, suite_json: Path
    ) -> None:
        args = (str(suite_json), "--cache-dir", str(tmp_path / "cache"), "--prepare-only")
        assert invoke(*args).exit_code == 0
        clone = (tmp_path / "cache").resolve() / "clones" / remote.first / "demo"
        (clone / "stray.txt").write_text("x", encoding="utf-8")
        result = invoke(*args)
        assert result.exit_code == 1
        assert result.stdout == ""
        assert "the worktree is not pristine (1 entries: ?? stray.txt)" in result.stderr
        assert f"eval-suite never repairs or deletes a clone: remove {clone} and run again" in (
            result.stderr
        )
        assert (clone / "stray.txt").exists()

    @pytest.mark.usefixtures("local_repositories")
    def test_a_gold_path_missing_at_the_pin_is_refused(
        self, tmp_path: Path, remote: Remote
    ) -> None:
        """`src/extra.py` is in the second commit only; the pin is the first."""
        golden = [{"query": "q", "relevant_files": ["src/extra.py"]}]
        path = write_suite(tmp_path / "s", str(remote.path), remote.first, golden=golden)
        result = invoke(str(path), "--cache-dir", str(tmp_path / "cache"), "--prepare-only")
        assert result.exit_code == 1
        assert result.stdout == ""
        assert result.stderr == (
            "refused: golden: line 1: \"relevant_files\" path 'src/extra.py' does not exist "
            f"at {remote.first}\n"
        )

    @pytest.mark.usefixtures("local_repositories")
    def test_a_gitignore_the_walker_cannot_parse_is_one_refusal_on_every_run(
        self, tmp_path: Path
    ) -> None:
        """The clone is kept once it is verified, so a traceback here would come back on every
        run. The lone `!` is a line pathspec rejects, in a repository that is otherwise fine."""
        hostile = make_remote(tmp_path, {**FIRST_FILES, ".gitignore": "build/\n!\n"})
        path = write_suite(tmp_path / "suite", str(hostile.path), hostile.first)
        args = (str(path), "--cache-dir", str(tmp_path / "cache"), "--prepare-only")
        for _ in range(2):
            result = invoke(*args)
            assert result.exit_code == 1
            assert isinstance(result.exception, SystemExit)
            assert result.stdout == ""
            [line] = result.stderr.splitlines()
            assert line.startswith("refused: the walker failed on the clone: ")

    def test_a_usage_error_exits_2_and_prints_no_result(self) -> None:
        result = invoke()
        assert result.exit_code == 2
        assert "prepared" not in result.stdout


@pytest.mark.usefixtures("local_repositories")
class TestCacheDirectory:
    def test_the_default_cache_root_is_under_xdg_cache_home(
        self, tmp_path: Path, remote: Remote, suite_json: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
        result = invoke(str(suite_json), "--prepare-only")
        assert result.exit_code == 0
        clone = tmp_path / "xdg" / "trelix" / "eval-suites" / "clones" / remote.first / "demo"
        assert f"clone: {clone.resolve()}\n" in result.stdout
        assert (clone / "src" / "app.py").is_file()

    def test_a_relative_cache_dir_is_printed_as_an_absolute_path(
        self, tmp_path: Path, remote: Remote, suite_json: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        result = invoke(str(suite_json), "--cache-dir", "cache", "--prepare-only")
        assert result.exit_code == 0
        clone = (tmp_path / "cache").resolve() / "clones" / remote.first / "demo"
        assert f"clone: {clone}\n" in result.stdout

    def test_an_empty_cache_dir_is_the_default_and_not_the_current_directory(
        self, tmp_path: Path, suite_json: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        work = tmp_path / "work"
        work.mkdir()
        monkeypatch.chdir(work)
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
        result = invoke(str(suite_json), "--cache-dir", "", "--prepare-only")
        assert result.exit_code == 0
        assert f"clone: {(tmp_path / 'xdg').resolve()}/trelix/eval-suites/clones/" in result.stdout
        assert list(work.iterdir()) == []

    def test_a_cache_dir_that_cannot_be_created_is_one_refusal_and_not_a_traceback(
        self, tmp_path: Path, suite_json: Path
    ) -> None:
        in_the_way = tmp_path / "a-file"
        in_the_way.write_text("not a directory", encoding="utf-8")
        result = invoke(str(suite_json), "--cache-dir", str(in_the_way), "--prepare-only")
        assert result.exit_code == 1
        assert isinstance(result.exception, SystemExit)
        assert result.stdout == ""
        assert len(result.stderr.splitlines()) == 1
        assert result.stderr.startswith(f"refused: cannot create {in_the_way.resolve()}/clones/")
        assert in_the_way.read_text(encoding="utf-8") == "not a directory"


class TestDefaultCacheRoot:
    def test_xdg_cache_home_is_used_when_absolute(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
        assert default_cache_root() == tmp_path / "xdg" / "trelix" / "eval-suites"

    @pytest.mark.parametrize("value", ["", "relative/dir"])
    def test_an_empty_or_relative_xdg_cache_home_falls_back_to_the_home_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        monkeypatch.setenv("XDG_CACHE_HOME", value)
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        assert default_cache_root() == tmp_path / "home" / ".cache" / "trelix" / "eval-suites"

    def test_no_home_directory_asks_for_cache_dir(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def no_home() -> Path:
            raise RuntimeError("Could not determine home directory.")

        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
        monkeypatch.setattr(Path, "home", staticmethod(no_home))
        with pytest.raises(SuiteError) as caught:
            default_cache_root()
        assert caught.value.problems == (
            "cannot find a home directory for the suite cache: pass --cache-dir",
        )


class TestListed:
    def test_the_command_is_listed_in_help_with_its_four_options(self) -> None:
        listing = CliRunner().invoke(app, ["--help"])
        assert listing.exit_code == 0
        assert "eval-suite" in plain(listing.output)
        assert "eval-compare" in plain(listing.output)

        own = CliRunner().invoke(app, ["eval-suite", "--help"])
        assert own.exit_code == 0
        for option in ("--arm", "--out", "--cache-dir", "--prepare-only"):
            assert option in plain(own.output)
