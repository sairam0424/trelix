"""`trelix eval-validate` on a v2 file: strata, the validated share, and v1-only handling.

The helpers and the git fixture are in `eval_validate_harness.py`.

MUTATIONS THAT MUST MAKE THIS FILE FAIL
---------------------------------------
1. `if 0 < counts[kind] < min_per_stratum` -> `<=`     (test_a_stratum_of_19_fails_and_20_passes)
   or without the `0 <`                                (test_a_kind_that_is_absent_...)
2. `if share >= min_validated` -> `>`                  (test_the_validated_share_boundary)
3. "pooled" dropped from REVIEWED_GOLD_STATUSES        (test_pooled_counts_as_reviewed)
4. a missing gold_status counted as reviewed           (test_a_missing_gold_status_counts_...)
5. the CLI defaults 20 / 0.95 changed                  (the 19-versus-20 and 94-versus-95 tests)
6. `kind` not type-checked                             (test_an_unknown_kind_is_one_schema_...)
7. a file without a `kind` taken as v2                 (test_a_v1_file_is_not_held_to_...)
   or any extra key making it v2                       (test_v2_fields_without_a_kind_...)
8. the skipped-path-check note printed the wrong way round (test_a_v1_file_still_gets_...)
9. the share divided by the entries that have a `kind`, not all (test_the_share_is_over_all_...)
10. the `isinstance(..., str)` guard on `kind` in `_stratum_problems` replaced by `"kind" in e.item`
                                                       (test_a_kind_that_is_a_list_or_object_...)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests.unit.eval_validate_harness import (
    NOTE_NO_REPO,
    NOTE_V1,
    golden_entry,
    invoke_eval_validate,
    v2_entries,
    write_golden,
)
from tests.unit.eval_validate_harness import repo as repo


class TestStrataAndValidatedShare:
    def test_a_stratum_of_19_fails_and_20_passes(self, tmp_path: Path) -> None:
        enough = write_golden(
            tmp_path, [*v2_entries("nl", 20), *v2_entries("keyword", 20)], "a.jsonl"
        )
        short = write_golden(
            tmp_path, [*v2_entries("nl", 19), *v2_entries("keyword", 20)], "b.jsonl"
        )

        passing = invoke_eval_validate(enough)
        failing = invoke_eval_validate(short)

        assert passing.exit_code == 0
        assert passing.stdout.splitlines() == [NOTE_NO_REPO, "valid: entries 40, violations 0"]
        assert failing.exit_code == 1
        assert failing.stdout.splitlines() == [
            "file: kind 'nl' has 19 queries, fewer than the 20 required per kind "
            "(--min-per-stratum)",
            NOTE_NO_REPO,
            "invalid: entries 39, violations 1",
        ]

    def test_min_per_stratum_moves_the_boundary(self, tmp_path: Path) -> None:
        golden = write_golden(tmp_path, v2_entries("nl", 19))

        assert invoke_eval_validate(golden).exit_code == 1
        assert invoke_eval_validate(golden, "--min-per-stratum", "19").exit_code == 0
        assert invoke_eval_validate(golden, "--min-per-stratum", "20").exit_code == 1

    def test_every_short_kind_is_reported(self, tmp_path: Path) -> None:
        golden = write_golden(
            tmp_path, [*v2_entries("issue", 3), *v2_entries("nl", 25), *v2_entries("commit", 1)]
        )

        result = invoke_eval_validate(golden)

        assert result.stdout.splitlines()[:2] == [
            "file: kind 'commit' has 1 queries, fewer than the 20 required per kind "
            "(--min-per-stratum)",
            "file: kind 'issue' has 3 queries, fewer than the 20 required per kind "
            "(--min-per-stratum)",
        ]

    def test_a_kind_that_is_absent_is_not_a_short_stratum(self, tmp_path: Path) -> None:
        golden = write_golden(tmp_path, v2_entries("commit", 20))

        assert invoke_eval_validate(golden).exit_code == 0

    def test_entries_without_a_kind_belong_to_no_stratum(self, tmp_path: Path) -> None:
        unlabelled = [golden_entry(f"plain {i}", gold_status="validated") for i in range(5)]
        golden = write_golden(tmp_path, [*v2_entries("nl", 20), *unlabelled])

        result = invoke_eval_validate(golden)

        assert result.exit_code == 0
        assert result.stdout.splitlines()[-1] == "valid: entries 25, violations 0"

    def test_an_unknown_kind_is_one_schema_violation_and_no_stratum(self, tmp_path: Path) -> None:
        golden = write_golden(tmp_path, [*v2_entries("nl", 20), *v2_entries("code", 1)])

        result = invoke_eval_validate(golden)

        assert result.exit_code == 1
        assert result.stdout.splitlines()[0] == (
            "line 21: \"kind\" must be one of nl, keyword, commit, issue (got 'code')"
        )
        assert result.stdout.splitlines()[-1] == "invalid: entries 21, violations 1"

    @pytest.mark.parametrize(("kind", "shown"), [(["nl"], "['nl']"), ({"a": 1}, "{'a': 1}")])
    def test_a_kind_that_is_a_list_or_object_is_one_schema_violation_not_a_crash(
        self, tmp_path: Path, kind: object, shown: str
    ) -> None:
        golden = write_golden(tmp_path, [golden_entry("one", kind=kind)])

        result = invoke_eval_validate(golden, "--min-validated", "0")

        assert result.exit_code == 1
        assert result.stdout.splitlines() == [
            f'line 1: "kind" must be one of nl, keyword, commit, issue (got {shown})',
            NOTE_NO_REPO,
            "invalid: entries 1, violations 1",
        ]

    @staticmethod
    def _shares(reviewed: int, status: str = "validated", missing: bool = False) -> list[Any]:
        """100 entries, 50 per kind: `reviewed` carry `status`, the rest are unreviewed/absent."""
        entries = []
        for i in range(100):
            kind = "nl" if i < 50 else "keyword"
            entry = golden_entry(f"q {i}", id=f"id-{i}", kind=kind)
            if i < reviewed:
                entry["gold_status"] = status
            elif not missing:
                entry["gold_status"] = "unreviewed"
            entries.append(entry)
        return entries

    def test_the_validated_share_boundary(self, tmp_path: Path) -> None:
        at_boundary = write_golden(tmp_path, self._shares(95), "a.jsonl")
        below = write_golden(tmp_path, self._shares(94), "b.jsonl")

        passing = invoke_eval_validate(at_boundary)
        failing = invoke_eval_validate(below)

        assert passing.exit_code == 0
        assert failing.exit_code == 1
        assert failing.stdout.splitlines()[0] == (
            "file: 94 of 100 entries have a gold_status of validated or pooled (0.9400); "
            "--min-validated requires 0.95"
        )
        assert failing.stdout.splitlines()[-1] == "invalid: entries 100, violations 1"

    def test_the_share_is_over_all_entries_including_those_with_no_kind(
        self, tmp_path: Path
    ) -> None:
        unlabelled = [golden_entry(f"plain {i}", id=f"plain-{i}") for i in range(2)]
        golden = write_golden(tmp_path, [*v2_entries("nl", 19), *unlabelled])

        result = invoke_eval_validate(golden, "--min-per-stratum", "0")

        assert result.exit_code == 1
        assert result.stdout.splitlines()[0] == (
            "file: 19 of 21 entries have a gold_status of validated or pooled (0.9048); "
            "--min-validated requires 0.95"
        )
        assert result.stdout.splitlines()[-1] == "invalid: entries 21, violations 1"

    def test_min_validated_moves_the_boundary(self, tmp_path: Path) -> None:
        golden = write_golden(tmp_path, self._shares(94))

        assert invoke_eval_validate(golden, "--min-validated", "0.94").exit_code == 0
        assert invoke_eval_validate(golden, "--min-validated", "0.95").exit_code == 1
        assert invoke_eval_validate(golden, "--min-validated", "0").exit_code == 0
        assert (
            invoke_eval_validate(
                write_golden(tmp_path, self._shares(99), "c.jsonl"), "--min-validated", "1"
            ).exit_code
            == 1
        )
        assert (
            invoke_eval_validate(
                write_golden(tmp_path, self._shares(100), "d.jsonl"), "--min-validated", "1"
            ).exit_code
            == 0
        )

    def test_pooled_counts_as_reviewed(self, tmp_path: Path) -> None:
        golden = write_golden(tmp_path, self._shares(95, status="pooled"))

        assert invoke_eval_validate(golden).exit_code == 0

    def test_a_missing_gold_status_counts_as_unreviewed(self, tmp_path: Path) -> None:
        passing = write_golden(tmp_path, self._shares(95, missing=True), "a.jsonl")
        failing = write_golden(tmp_path, self._shares(94, missing=True), "b.jsonl")

        assert invoke_eval_validate(passing).exit_code == 0
        assert invoke_eval_validate(failing).exit_code == 1

    def test_a_bad_gold_status_is_a_violation_and_is_not_reviewed(self, tmp_path: Path) -> None:
        entries = self._shares(95)
        entries[0]["gold_status"] = "done"
        golden = write_golden(tmp_path, entries)

        result = invoke_eval_validate(golden)

        assert result.stdout.splitlines()[0] == (
            "line 1: \"gold_status\" must be one of validated, pooled, unreviewed (got 'done')"
        )
        assert result.stdout.splitlines()[1].startswith("file: 94 of 100 entries")

    @pytest.mark.parametrize(
        ("option", "value"),
        [("--min-per-stratum", "-1"), ("--min-validated", "1.01"), ("--min-validated", "-0.1")],
    )
    def test_an_out_of_range_threshold_is_a_usage_error(
        self, tmp_path: Path, option: str, value: str
    ) -> None:
        golden = write_golden(tmp_path, v2_entries("nl", 20))

        assert invoke_eval_validate(golden, option, value).exit_code == 2


class TestAV1FileGetsOnlyTheV1Checks:
    def test_a_v1_file_is_not_held_to_strata_or_the_validated_share(self, tmp_path: Path) -> None:
        golden = write_golden(tmp_path, [golden_entry("one"), golden_entry("two")])

        result = invoke_eval_validate(golden)

        assert result.exit_code == 0
        assert NOTE_V1 in result.stdout.splitlines()

    def test_v2_fields_without_a_kind_do_not_make_it_a_v2_file(self, tmp_path: Path) -> None:
        golden = write_golden(
            tmp_path,
            [
                golden_entry("one", id="a", gold_status="unreviewed"),
                golden_entry("two", split="dev"),
            ],
        )

        result = invoke_eval_validate(golden)

        assert result.exit_code == 0
        assert NOTE_V1 in result.stdout.splitlines()

    def test_one_entry_with_a_kind_makes_it_a_v2_file(self, tmp_path: Path) -> None:
        golden = write_golden(tmp_path, [golden_entry("one", kind="nl"), golden_entry("two")])

        result = invoke_eval_validate(golden)

        assert result.exit_code == 1
        assert NOTE_V1 not in result.stdout.splitlines()
        assert result.stdout.splitlines()[0].startswith("file: kind 'nl' has 1 queries")

    def test_a_v1_file_still_gets_the_duplicate_and_path_checks(
        self, tmp_path: Path, repo: Path
    ) -> None:
        golden = write_golden(
            tmp_path, [golden_entry("same", "src/old.py"), golden_entry("SAME", "README.md")]
        )

        result = invoke_eval_validate(golden, "--repo", str(repo))

        assert result.stdout.splitlines()[:2] == [
            "line 1: \"relevant_files\" path 'src/old.py' does not exist at HEAD",
            "line 2: duplicate query (same as line 1 once stripped and case-folded)",
        ]
