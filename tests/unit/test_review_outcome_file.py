"""The JSON record `trelix review` writes when TRELIX_REVIEW_OUTCOME_FILE names a path.

It is read by other programs (a workflow step, the GitHub App) and may be published in a Check,
so the shape is pinned here as literals, it must never carry anything but the fixed vocabulary,
and the write must be atomic and must not follow a symlink.

Mutations each test was checked against (every one fails a test below): the cap of 100 changed,
`hunks_omitted` dropped, reviewed hunks listed too, the temporary file left behind, the mode
widened, the write made non-atomic.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from trelix.review import outcome_file
from trelix.review.hunk_status import HunkResult, HunkStatus
from trelix.review.outcome_file import (
    MAX_LISTED_HUNKS,
    SCHEMA_VERSION,
    build_outcome_payload,
    write_outcome_file,
)
from trelix.review.reviewer import ReviewOutcome


def _outcome(*results: HunkResult) -> ReviewOutcome:
    return ReviewOutcome(
        llm_available=True,
        hunks_total=len(results),
        hunks_failed=sum(1 for r in results if not r.reviewed),
        hunk_results=tuple(results),
    )


class TestPayload:
    def test_a_partial_review_lists_only_the_unreviewed_hunks(self) -> None:
        outcome = _outcome(
            HunkResult("a.py", 1, HunkStatus.REVIEWED, "", 2),
            HunkResult("b.py", 5, HunkStatus.TRUNCATED, "length_after_retry", 1),
            HunkResult("c.py", 9, HunkStatus.REFUSED, "refusal"),
        )

        assert build_outcome_payload(outcome, exit_code=4) == {
            "schema_version": 1,
            "hunks_total": 3,
            "hunks_reviewed": 1,
            "hunks_unreviewed": 2,
            "exit_code": 4,
            "hunks": [
                {"file": "b.py", "line": 5, "status": "truncated", "detail": "length_after_retry"},
                {"file": "c.py", "line": 9, "status": "refused", "detail": "refusal"},
            ],
            "hunks_omitted": 0,
        }

    def test_a_complete_review_lists_nothing(self) -> None:
        payload = build_outcome_payload(
            _outcome(HunkResult("a.py", 1, HunkStatus.REVIEWED)), exit_code=0
        )

        assert payload["hunks"] == []
        assert payload["hunks_unreviewed"] == 0
        assert payload["exit_code"] == 0

    def test_the_list_is_capped_and_the_rest_is_counted(self) -> None:
        results = [
            HunkResult(f"f{i}.py", i, HunkStatus.PARSE_FAILED, "no_review_array")
            for i in range(130)
        ]

        payload = build_outcome_payload(_outcome(*results), exit_code=4)

        assert len(payload["hunks"]) == 100
        assert payload["hunks"][0]["file"] == "f0.py"
        assert payload["hunks"][-1]["file"] == "f99.py"
        assert payload["hunks_omitted"] == 30
        assert payload["hunks_unreviewed"] == 130

    def test_exactly_at_the_cap_omits_nothing(self) -> None:
        results = [HunkResult(f"f{i}.py", i, HunkStatus.ERROR, "x") for i in range(100)]

        payload = build_outcome_payload(_outcome(*results), exit_code=4)

        assert len(payload["hunks"]) == 100
        assert payload["hunks_omitted"] == 0

    def test_more_failures_than_hunks_never_reports_negative_reviewed(self) -> None:
        outcome = ReviewOutcome(llm_available=True, hunks_total=2, hunks_failed=5)

        payload = build_outcome_payload(outcome, exit_code=3)

        assert payload["hunks_reviewed"] == 0
        assert payload["hunks_unreviewed"] == 5

    def test_the_constants_are_the_documented_values(self) -> None:
        assert SCHEMA_VERSION == 1
        assert MAX_LISTED_HUNKS == 100

    def test_the_payload_has_exactly_these_keys(self) -> None:
        payload = build_outcome_payload(ReviewOutcome(), exit_code=0)

        assert sorted(payload) == [
            "exit_code",
            "hunks",
            "hunks_omitted",
            "hunks_reviewed",
            "hunks_total",
            "hunks_unreviewed",
            "schema_version",
        ]

    def test_a_review_with_no_llm_lists_every_hunk_as_an_error(self) -> None:
        results = (
            HunkResult("a.py", 1, HunkStatus.ERROR, "not_configured"),
            HunkResult("b.py", 2, HunkStatus.ERROR, "not_configured"),
        )
        outcome = ReviewOutcome(
            llm_available=False, hunks_total=2, hunks_failed=2, hunk_results=results
        )

        payload = build_outcome_payload(outcome, exit_code=3)

        assert [h["detail"] for h in payload["hunks"]] == ["not_configured", "not_configured"]
        assert payload["exit_code"] == 3


class TestWrite:
    def test_the_document_round_trips(self, tmp_path: Path) -> None:
        target = tmp_path / "outcome.json"
        payload = build_outcome_payload(
            _outcome(HunkResult("a.py", 1, HunkStatus.TRUNCATED, "length")), exit_code=4
        )

        assert write_outcome_file(str(target), payload) is None

        assert json.loads(target.read_text(encoding="utf-8")) == payload
        assert target.read_text(encoding="utf-8").endswith("\n")

    def test_no_temporary_file_is_left_behind(self, tmp_path: Path) -> None:
        write_outcome_file(str(tmp_path / "outcome.json"), {"a": 1})

        assert [p.name for p in tmp_path.iterdir()] == ["outcome.json"]

    def test_the_file_is_private_to_the_owner(self, tmp_path: Path) -> None:
        target = tmp_path / "outcome.json"

        write_outcome_file(str(target), {"a": 1})

        assert stat.S_IMODE(target.stat().st_mode) == 0o600

    def test_an_existing_file_is_replaced(self, tmp_path: Path) -> None:
        target = tmp_path / "outcome.json"
        target.write_text("old contents", encoding="utf-8")

        assert write_outcome_file(str(target), {"new": True}) is None

        assert json.loads(target.read_text(encoding="utf-8")) == {"new": True}

    def test_a_symlink_at_the_path_is_replaced_not_followed(self, tmp_path: Path) -> None:
        victim = tmp_path / "victim.txt"
        victim.write_text("do not touch", encoding="utf-8")
        target = tmp_path / "outcome.json"
        os.symlink(victim, target)

        assert write_outcome_file(str(target), {"a": 1}) is None

        assert victim.read_text(encoding="utf-8") == "do not touch"
        assert not target.is_symlink()
        assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1}

    def test_a_missing_directory_is_reported_not_raised_and_not_created(
        self, tmp_path: Path
    ) -> None:
        target = tmp_path / "missing" / "outcome.json"

        error = write_outcome_file(str(target), {"a": 1})

        assert error is not None
        assert error.startswith("FileNotFoundError")
        assert not (tmp_path / "missing").exists()

    def test_a_path_that_is_a_directory_is_reported_and_leaves_no_temporary(
        self, tmp_path: Path
    ) -> None:
        box = tmp_path / "box"
        box.mkdir()
        target = box / "dir"
        target.mkdir()

        error = write_outcome_file(str(target), {"a": 1})

        assert error is not None
        assert [p.name for p in box.iterdir()] == ["dir"]  # the temporary is beside the target

    @pytest.mark.parametrize("path", [".", "/", "./", ""])
    def test_a_path_with_no_file_name_is_reported_not_raised(self, path: str) -> None:
        error = write_outcome_file(path, {"a": 1})

        assert error is not None
        assert error.startswith("ValueError")

    def test_an_existing_temporary_name_is_refused_and_left_alone(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The temporary name is random; force a collision to show O_EXCL refuses to reuse one.
        monkeypatch.setattr(outcome_file.secrets, "token_hex", lambda _n: "fixed")
        planted = tmp_path / ".outcome.json.fixed.tmp"
        planted.write_text("planted", encoding="utf-8")

        error = write_outcome_file(str(tmp_path / "outcome.json"), {"a": 1})

        assert error is not None
        assert error.startswith("FileExistsError")
        assert planted.read_text(encoding="utf-8") == "planted"
        assert not (tmp_path / "outcome.json").exists()

    def test_the_temporary_name_differs_between_writes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        names: list[str] = []
        original = os.open

        def _recording_open(path: object, *args: object, **kwargs: object) -> int:
            names.append(Path(str(path)).name)
            return original(path, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(os, "open", _recording_open)

        write_outcome_file(str(tmp_path / "outcome.json"), {"a": 1})
        write_outcome_file(str(tmp_path / "outcome.json"), {"a": 2})

        assert len(names) == 2
        assert names[0] != names[1]
        assert all(n.startswith(".outcome.json.") and n.endswith(".tmp") for n in names)

    def test_a_failure_while_writing_leaves_no_temporary_and_no_target(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _boom(src: object, dst: object) -> None:
            raise PermissionError("denied")

        monkeypatch.setattr(os, "replace", _boom)

        error = write_outcome_file(str(tmp_path / "outcome.json"), {"a": 1})

        assert error is not None
        assert error.startswith("PermissionError")
        assert list(tmp_path.iterdir()) == []
