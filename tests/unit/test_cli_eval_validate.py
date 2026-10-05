"""`trelix eval-validate`: schema, exit codes, duplicates and paths at a git revision.

The strata, the validated share and the v1-only handling are in
`test_cli_eval_validate_v2_checks.py`; the helpers and the git fixture are in
`eval_validate_harness.py`.

MUTATIONS THAT MUST MAKE THIS FILE FAIL
---------------------------------------
1. `.casefold()` removed or swapped for `.lower()`     (test_a_duplicate_query_differing_...)
2. `.strip()` removed from `_query_key`                (test_a_duplicate_query_differing_...)
3. the duplicate-id or duplicate-query check removed   (test_a_duplicate_id_is_reported_...)
4. `rev` ignored in `git ls-tree` (always HEAD)        (test_an_older_rev_sees_the_file_...)
5. the exit code 1 on a violation removed              (test_a_violation_exits_1_and_is_...)
6. `markup=False` removed from the print               (test_output_is_printed_verbatim_...)
   `emoji=False` removed from the print                (test_output_is_printed_byte_for_byte_...)
   `soft_wrap=True` removed from the print             (test_a_long_violation_stays_on_one_line)
7. the default `--rev HEAD` changed                    (test_paths_are_checked_at_head_by_default)
8. the path check skipped, `--full-tree`, no `-z`, a malformed path also reported as missing,
   the directory check or the git exit status ignored (the TestPathsAtARevision cases)
9. a non-UTF-8 file, an empty file, a non-object line, an empty `relevant_files` or a blank
   `query` accepted                                   (the TestSchemaAndExitCodes cases)
10. an `id` case-folded or stripped before comparing  (test_ids_are_compared_exactly)
    a blank `id` accepted or reported as a duplicate  (test_a_blank_id_is_a_schema_violation_...)
11. `TimeoutExpired` / `UnicodeDecodeError` / `OSError` dropped from the `except` of
    `read_repo_tree`, or the whole `except` removed     (test_a_git_that_times_out_...)
12. the leading-`-` check on `--rev` removed or narrowed to `--`  (test_a_rev_that_looks_like_...)
13. `OSError` dropped from the `except` of `eval_validate`     (test_a_golden_file_that_cannot_...)
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tests.unit.eval_validate_harness import (
    NOTE_NO_REPO,
    NOTE_V1,
    golden_entry,
    invoke_eval_validate,
    run_git,
    write_golden,
)
from tests.unit.eval_validate_harness import repo as repo
from trelix.eval.harness import _parse_golden

_REPO_ROOT = Path(__file__).resolve().parents[2]


class TestSchemaAndExitCodes:
    def test_a_valid_v1_file_exits_0_and_says_what_it_skipped(self, tmp_path: Path) -> None:
        golden = write_golden(
            tmp_path, [golden_entry("first"), golden_entry("second", "a.py", "b/c.py")]
        )

        result = invoke_eval_validate(golden)

        assert result.exit_code == 0
        assert result.stdout.splitlines() == [
            NOTE_NO_REPO,
            NOTE_V1,
            "valid: entries 2, violations 0",
        ]

    def test_the_shipped_golden_file_is_valid(self) -> None:
        result = invoke_eval_validate(str(_REPO_ROOT / "eval" / "golden.jsonl"))

        assert result.exit_code == 0
        assert result.stdout.splitlines()[-1] == "valid: entries 54, violations 0"

    def test_a_violation_exits_1_and_is_printed_on_stdout_as_line_n(self, tmp_path: Path) -> None:
        golden = write_golden(
            tmp_path, [golden_entry("fine"), {"query": "no files", "relevant_files": []}]
        )

        result = invoke_eval_validate(golden)

        assert result.exit_code == 1
        assert result.stdout.splitlines() == [
            'line 2: "relevant_files" must be a non-empty list of repo-relative POSIX paths',
            NOTE_NO_REPO,
            NOTE_V1,
            "invalid: entries 2, violations 1",
        ]
        assert result.stderr == ""

    def test_every_violation_is_on_its_own_line(self, tmp_path: Path) -> None:
        path = tmp_path / "golden.jsonl"
        path.write_text(
            '{"query": "ok", "relevant_files": ["a.py"]}\n'
            '{"query": oops}\n'
            "[1, 2]\n"
            '{"query": "  ", "relevant_files": ["./a.py"], "kind": "code"}\n',
            encoding="utf-8",
        )

        result = invoke_eval_validate(str(path))

        assert result.exit_code == 1
        # A `kind` key, even a wrong one, makes line 4 a v2 entry, so the file is checked as v2.
        assert result.stdout.splitlines() == [
            "line 2: not valid JSON (Expecting value at column 11)",
            "line 3: expected a JSON object, got list",
            'line 4: "query" must be a non-empty string',
            "line 4: \"relevant_files\" './a.py' is not normalised (did you mean 'a.py'?)",
            "line 4: \"kind\" must be one of nl, keyword, commit, issue (got 'code')",
            "file: 0 of 2 entries have a gold_status of validated or pooled (0.0000); "
            "--min-validated requires 0.95",
            NOTE_NO_REPO,
            "invalid: entries 2, violations 6",
        ]

    def test_an_empty_file_is_a_violation_not_a_pass(self, tmp_path: Path) -> None:
        path = tmp_path / "golden.jsonl"
        path.write_text("\n   \n", encoding="utf-8")

        result = invoke_eval_validate(str(path))

        assert result.exit_code == 1
        assert result.stdout.splitlines() == [
            "file: no golden entries",
            "invalid: entries 0, violations 1",
        ]

    def test_a_file_that_is_not_utf8_is_a_violation(self, tmp_path: Path) -> None:
        path = tmp_path / "golden.jsonl"
        path.write_bytes(b'\xff{"query": "q", "relevant_files": ["a.py"]}\n')

        result = invoke_eval_validate(str(path))

        assert result.exit_code == 1
        assert (
            result.stdout.splitlines()[0] == "file: not valid UTF-8 (invalid start byte at byte 0)"
        )

    def test_a_missing_golden_file_is_an_error_on_stderr(self, tmp_path: Path) -> None:
        result = invoke_eval_validate(str(tmp_path / "nope.jsonl"))

        assert result.exit_code == 1
        assert "Golden file not found" in result.stderr
        assert result.stdout == ""

    def test_a_golden_file_that_cannot_be_opened_is_a_one_line_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def deny(self: Path, *_args: object, **_kwargs: object) -> str:
            raise PermissionError(13, "Permission denied", str(self))

        golden = write_golden(tmp_path, [golden_entry("code")])
        monkeypatch.setattr(Path, "read_text", deny)
        result = invoke_eval_validate(golden)

        assert result.exit_code == 1
        stderr = " ".join(result.stderr.split())
        assert "Golden validation failed: [Errno 13] Permission denied" in stderr
        assert isinstance(result.exception, SystemExit)  # a handled exit, not a crash
        assert result.stdout == ""

    @pytest.mark.parametrize(
        "line",
        [
            '{"query": "q", "relevant_files": []}',
            '{"query": "q"}',
            '{"relevant_files": ["a.py"]}',
            '{"query": "   ", "relevant_files": ["a.py"]}',
            '{"query": "q", "relevant_files": "a.py"}',
            '{"query": "q", "relevant_files": ["./a.py"]}',
            '{"query": "q", "relevant_files": ["/abs/a.py"]}',
            '{"query": "q", "relevant_files": ["src\\\\a.py"]}',
            '{"query": "q", "relevant_files": ["src/../a.py"]}',
            '{"query": "q", "relevant_files": ["a.py", ""]}',
            '{"query": "q", "relevant_files": ["a.py"], "kind": "code"}',
            '{"query": "q", "relevant_files": ["a.py"], "split": null}',
        ],
    )
    def test_it_refuses_exactly_the_lines_the_loader_refuses(
        self, tmp_path: Path, line: str
    ) -> None:
        path = tmp_path / "golden.jsonl"
        path.write_text(line + "\n", encoding="utf-8")

        with pytest.raises(ValueError, match="line 1"):
            _parse_golden(path)
        result = invoke_eval_validate(str(path))

        assert result.exit_code == 1
        assert result.stdout.splitlines()[0].startswith("line 1: ")

    @pytest.mark.parametrize(
        "line",
        [
            '{"query": "q", "relevant_files": ["a.py"]}',
            '{"query": "q", "relevant_files": ["a.py", "b/c.py"], "area": 5, "note": [1]}',
            '{"query": "q", "relevant_files": ["a.py"], "kind": "nl", "split": "dev"}',
        ],
    )
    def test_it_accepts_exactly_the_lines_the_loader_accepts(
        self, tmp_path: Path, line: str
    ) -> None:
        path = tmp_path / "golden.jsonl"
        path.write_text(line + "\n", encoding="utf-8")

        assert len(_parse_golden(path)) == 1
        assert (
            invoke_eval_validate(
                str(path), "--min-per-stratum", "0", "--min-validated", "0"
            ).exit_code
            == 0
        )


class TestDuplicates:
    @pytest.mark.parametrize(
        ("first", "second"),
        [
            ("How does auth work", "how does AUTH work"),
            ("How does auth work", "  How does auth work\t"),
            ("How does auth work", "  HOW DOES auth WORK "),
            ("the Straße rule", "THE STRASSE RULE"),
        ],
    )
    def test_a_duplicate_query_differing_only_in_case_and_whitespace(
        self, tmp_path: Path, first: str, second: str
    ) -> None:
        golden = write_golden(
            tmp_path, [golden_entry(first), golden_entry("something else"), golden_entry(second)]
        )

        result = invoke_eval_validate(golden)

        assert result.exit_code == 1
        assert result.stdout.splitlines()[0] == (
            "line 3: duplicate query (same as line 1 once stripped and case-folded)"
        )
        assert result.stdout.splitlines()[-1] == "invalid: entries 3, violations 1"

    def test_a_duplicate_id_is_reported_on_the_later_line(self, tmp_path: Path) -> None:
        golden = write_golden(
            tmp_path,
            [
                golden_entry("one", id="a"),
                golden_entry("two", id="b"),
                golden_entry("three", id="a"),
            ],
        )

        result = invoke_eval_validate(golden)

        assert result.exit_code == 1
        assert result.stdout.splitlines()[0] == "line 3: duplicate id 'a' (first used on line 1)"
        assert result.stdout.splitlines()[-1] == "invalid: entries 3, violations 1"

    def test_distinct_queries_and_ids_are_not_duplicates(self, tmp_path: Path) -> None:
        golden = write_golden(tmp_path, [golden_entry("one", id="a"), golden_entry("two", id="b")])

        assert invoke_eval_validate(golden).exit_code == 0

    def test_output_is_printed_verbatim_not_as_rich_markup(self, tmp_path: Path) -> None:
        golden = write_golden(
            tmp_path,
            [golden_entry("one", id="[bold]x[/bold]"), golden_entry("two", id="[bold]x[/bold]")],
        )

        result = invoke_eval_validate(golden)

        assert (
            "line 2: duplicate id '[bold]x[/bold]' (first used on line 1)"
            in result.stdout.splitlines()
        )

    def test_output_is_printed_byte_for_byte_not_as_emoji_shortcodes(
        self, tmp_path: Path, repo: Path
    ) -> None:
        # Rich turns ":new:" and ":warning:" into emoji even with markup off; a user who greps
        # the golden file for what was printed must find it.
        golden = write_golden(
            tmp_path,
            [
                golden_entry("one", "src/:warning:.py", id="nl:new:7"),
                golden_entry("two", "README.md", id="nl:new:7"),
            ],
        )

        result = invoke_eval_validate(golden, "--repo", str(repo))

        assert result.stdout.splitlines()[:2] == [
            "line 1: \"relevant_files\" path 'src/:warning:.py' does not exist at HEAD",
            "line 2: duplicate id 'nl:new:7' (first used on line 1)",
        ]

    def test_ids_are_compared_exactly(self, tmp_path: Path) -> None:
        golden = write_golden(
            tmp_path,
            [
                golden_entry("one", id="A"),
                golden_entry("two", id="a"),
                golden_entry("three", id=" a"),
            ],
        )

        assert invoke_eval_validate(golden).exit_code == 0

    def test_a_blank_id_is_a_schema_violation_and_never_a_duplicate(self, tmp_path: Path) -> None:
        golden = write_golden(tmp_path, [golden_entry("one", id=""), golden_entry("two", id="")])

        result = invoke_eval_validate(golden)

        assert result.exit_code == 1
        assert result.stdout.splitlines()[:2] == [
            'line 1: "id" must be a non-empty string',
            'line 2: "id" must be a non-empty string',
        ]
        assert result.stdout.splitlines()[-1] == "invalid: entries 2, violations 2"


class TestPathsAtARevision:
    def test_paths_are_checked_at_head_by_default(self, tmp_path: Path, repo: Path) -> None:
        golden = write_golden(
            tmp_path, [golden_entry("renamed", "src/new.py"), golden_entry("stale", "src/old.py")]
        )

        result = invoke_eval_validate(golden, "--repo", str(repo))

        assert result.exit_code == 1
        assert result.stdout.splitlines() == [
            "line 2: \"relevant_files\" path 'src/old.py' does not exist at HEAD",
            NOTE_V1,
            "invalid: entries 2, violations 1",
        ]

    def test_an_older_rev_sees_the_file_as_it_was_before_the_rename(
        self, tmp_path: Path, repo: Path
    ) -> None:
        golden = write_golden(
            tmp_path, [golden_entry("renamed", "src/new.py"), golden_entry("stale", "src/old.py")]
        )

        result = invoke_eval_validate(golden, "--repo", str(repo), "--rev", "HEAD~1")

        assert result.exit_code == 1
        assert result.stdout.splitlines() == [
            "line 1: \"relevant_files\" path 'src/new.py' does not exist at HEAD~1",
            NOTE_V1,
            "invalid: entries 2, violations 1",
        ]

    def test_only_the_missing_path_of_a_multi_path_entry_is_named(
        self, tmp_path: Path, repo: Path
    ) -> None:
        golden = write_golden(
            tmp_path, [golden_entry("two files", "README.md", "src/new.py", "gone.py")]
        )

        result = invoke_eval_validate(golden, "--repo", str(repo))

        assert result.exit_code == 1
        assert result.stdout.splitlines()[0] == (
            "line 1: \"relevant_files\" path 'gone.py' does not exist at HEAD"
        )
        assert result.stdout.splitlines()[-1] == "invalid: entries 1, violations 1"

    def test_with_a_repo_the_skipped_note_goes_away_and_a_good_file_passes(
        self, tmp_path: Path, repo: Path
    ) -> None:
        golden = write_golden(
            tmp_path, [golden_entry("readme", "README.md"), golden_entry("code", "src/new.py")]
        )

        result = invoke_eval_validate(golden, "--repo", str(repo))

        assert result.exit_code == 0
        assert result.stdout.splitlines() == [NOTE_V1, "valid: entries 2, violations 0"]

    def test_a_malformed_path_is_reported_once_not_also_as_missing(
        self, tmp_path: Path, repo: Path
    ) -> None:
        golden = write_golden(tmp_path, [golden_entry("dotted", "./src/new.py")])

        result = invoke_eval_validate(golden, "--repo", str(repo))

        assert result.stdout.splitlines()[0] == (
            "line 1: \"relevant_files\" './src/new.py' is not normalised "
            "(did you mean 'src/new.py'?)"
        )
        assert result.stdout.splitlines()[-1] == "invalid: entries 1, violations 1"

    def test_a_repo_that_is_a_subdirectory_lists_only_its_own_subtree(
        self, tmp_path: Path, repo: Path
    ) -> None:
        inside = write_golden(tmp_path, [golden_entry("in", "new.py")], name="inside.jsonl")
        outside = write_golden(tmp_path, [golden_entry("out", "src/new.py")], name="outside.jsonl")

        assert invoke_eval_validate(inside, "--repo", str(repo / "src")).exit_code == 0
        assert invoke_eval_validate(outside, "--repo", str(repo / "src")).exit_code == 1

    def test_a_file_name_git_would_quote_is_found(self, tmp_path: Path, repo: Path) -> None:
        name = "docs/héllo wörld.md"
        (repo / "docs").mkdir()
        (repo / name).write_text("hi\n", encoding="utf-8")
        run_git(repo, "add", name)
        run_git(repo, "commit", "-q", "-m", "unicode name")
        golden = write_golden(tmp_path, [golden_entry("unicode", name)])

        result = invoke_eval_validate(golden, "--repo", str(repo))

        assert result.exit_code == 0
        assert result.stdout.splitlines()[-1] == "valid: entries 1, violations 0"

    def test_a_long_violation_stays_on_one_line(self, tmp_path: Path, repo: Path) -> None:
        long_path = "src/" + "deep/" * 30 + "file.py"
        golden = write_golden(tmp_path, [golden_entry("long", long_path)])

        result = invoke_eval_validate(golden, "--repo", str(repo))

        assert result.stdout.splitlines()[0] == (
            f"line 1: \"relevant_files\" path '{long_path}' does not exist at HEAD"
        )
        assert len(long_path) == 161

    def test_an_unknown_rev_is_an_error_and_nothing_else_is_printed(
        self, tmp_path: Path, repo: Path
    ) -> None:
        golden = write_golden(tmp_path, [golden_entry("code", "src/new.py")])

        result = invoke_eval_validate(golden, "--repo", str(repo), "--rev", "no-such-rev")

        assert result.exit_code == 1
        assert "Golden validation failed" in result.stderr
        assert "no-such-rev" in result.stderr
        assert result.stdout == ""

    def test_a_missing_repo_is_an_error(self, tmp_path: Path) -> None:
        golden = write_golden(tmp_path, [golden_entry("code")])

        result = invoke_eval_validate(golden, "--repo", str(tmp_path / "absent"))

        assert result.exit_code == 1
        assert "isnotadirectory" in "".join(result.stderr.split())  # Rich folds a long path
        assert result.stdout == ""

    def test_a_directory_that_is_no_git_repository_is_an_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        plain = tmp_path / "plain"
        plain.mkdir()
        monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
        golden = write_golden(tmp_path, [golden_entry("code")])

        result = invoke_eval_validate(golden, "--repo", str(plain))

        assert result.exit_code == 1
        assert "not a git repository" in result.stderr
        assert result.stdout == ""

    @pytest.mark.parametrize(
        ("raised", "named"),
        [
            (subprocess.TimeoutExpired(cmd="git", timeout=30), "TimeoutExpired"),
            (
                UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte"),
                "UnicodeDecodeError",
            ),
            (FileNotFoundError(2, "No such file or directory"), "FileNotFoundError"),
        ],
    )
    def test_a_git_that_times_out_or_is_missing_or_unreadable_is_a_one_line_error(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        raised: Exception,
        named: str,
    ) -> None:
        def fail(*_args: str) -> subprocess.CompletedProcess[str]:
            raise raised

        monkeypatch.setattr("trelix.eval.golden_validate._run_git", fail)
        golden = write_golden(tmp_path, [golden_entry("code")])

        result = invoke_eval_validate(golden, "--repo", str(tmp_path))

        assert result.exit_code == 1
        assert f"Golden validation failed: git ls-tree failed: {named}:" in " ".join(
            result.stderr.split()
        )
        assert "Traceback" not in result.stderr
        assert result.stdout == ""

    @pytest.mark.parametrize("rev", ["--full-tree", "-l"])
    def test_a_rev_that_looks_like_a_git_option_is_refused_before_git_reads_it(
        self, tmp_path: Path, repo: Path, rev: str
    ) -> None:
        golden = write_golden(tmp_path, [golden_entry("code", "src/new.py")])

        result = invoke_eval_validate(golden, "--repo", str(repo), f"--rev={rev}")

        assert result.exit_code == 1
        assert f"revision '{rev}' starts with '-'" in " ".join(result.stderr.split())
        assert result.stdout == ""
