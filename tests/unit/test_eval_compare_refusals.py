"""The eval-compare judge: what is refused, and the three-file entry point.

MUTATIONS THAT MUST MAKE THIS FILE FAIL
---------------------------------------
1. one of the seven suite fields left out of the identity check  (test_a_suite_field_that_differs)
2. the labels of the two runs not compared          (TestLabels)
3. the comparison_id check removed                 (test_swapped_arguments_are_refused)
4. a frozen flag not checked (any of the four), or plans not checked  (TestFrozenRuns)
5. one of the refusal rows dropped from `refusals()`  (TestRefusals, test_several_reasons_...)
6. `compare_files` letting an exception out         (test_an_unexpected_exception_is_a_refusal)
7. `--prereg` absent not refused                    (test_no_prereg_is_a_refusal)
8. the mixed-split test moved by one record either way, or the message counting the unlabelled
   records                                           (test_f14_..., the 1 and 19 cases)
9. the different-queries refusal looking at the base-only ids alone
                                                     (test_a_candidate_with_an_extra_query_...)
"""

from __future__ import annotations

from pathlib import Path

import pytest

import trelix.eval.compare as compare_module
from tests.unit.eval_compare_fixtures import (
    SUITE,
    base_doc,
    cand_doc,
    pipeline_doc,
    query_ids,
    results_doc,
    verdict_of,
    write_json,
    write_prereg,
)
from trelix.eval.compare import Verdict, compare_files


class TestRefusals:
    @pytest.mark.parametrize(
        ("field", "other"),
        [
            ("name", "other"),
            ("golden_version", "v2"),
            ("repo_url", "/srv/other.git"),
            ("repo_sha", "d" * 40),
            ("license", "Apache-2.0"),
            ("golden_sha256", "e" * 64),
            ("plans_sha256", "f" * 64),
        ],
    )
    def test_a_suite_field_that_differs(self, field: str, other: str) -> None:
        cand = cand_doc(0.5625, suite={**SUITE, field: other})

        verdict = verdict_of(base_doc(), cand)

        assert verdict.outcome == "REFUSED"
        assert verdict.reasons == (f"suite identity differs in: {field}",)

    def test_every_differing_suite_field_is_named(self) -> None:
        suite = {**SUITE, "golden_sha256": "e" * 64, "repo_sha": "d" * 40}

        verdict = verdict_of(base_doc(), cand_doc(0.5625, suite=suite))

        assert verdict.reasons == ("suite identity differs in: repo_sha, golden_sha256",)

    def test_swapped_arguments_are_refused(self) -> None:
        verdict = verdict_of(cand_doc(0.5625), base_doc())

        assert verdict.outcome == "REFUSED"
        assert verdict.reasons == (
            "comparison_id baseline..flag-on does not match the arms of the files given as "
            "base and cand (flag-on..baseline): check the order of the arguments",
        )

    def test_a_comparison_id_naming_other_arms_is_refused(self) -> None:
        verdict = verdict_of(base_doc(), cand_doc(0.5625), comparison_id="baseline..other")

        assert verdict.outcome == "REFUSED"
        assert verdict.reasons[0].startswith("comparison_id baseline..other does not match")

    def test_two_files_of_one_arm_are_refused(self) -> None:
        verdict = verdict_of(base_doc(), results_doc("baseline", 0.5625))

        assert verdict.outcome == "REFUSED"
        assert verdict.reasons[0].startswith("comparison_id baseline..flag-on does not match")

    def test_queries_only_in_one_run_are_listed_both_ways(self) -> None:
        base = base_doc(ids=query_ids(20))
        cand = cand_doc(0.5625, ids=[*query_ids(19), "q99"])

        verdict = verdict_of(base, cand)

        assert verdict.reasons == (
            "the runs cover different queries: only in base: q20; only in cand: q99",
        )

    def test_only_the_first_five_ids_are_listed(self) -> None:
        base = base_doc(ids=query_ids(27))
        cand = cand_doc(0.5625, ids=query_ids(20))

        verdict = verdict_of(base, cand)

        assert verdict.reasons == (
            "the runs cover different queries: only in base: q21, q22, q23, q24, q25 "
            "(and 2 more); only in cand: none",
        )

    @pytest.mark.parametrize(
        ("labelled", "counted"), [(1, "1 of 20"), (10, "10 of 20"), (19, "19 of 20")]
    )
    def test_f14_split_labels_on_some_base_records_only_are_refused(
        self, labelled: int, counted: str
    ) -> None:
        # One labelled record is already "mixed", and so is one unlabelled record, and the
        # message counts the labelled ones (the 10-of-20 case alone cannot tell which it counts).
        splits = ["test"] * labelled + [None] * (20 - labelled)
        verdict = verdict_of(base_doc(splits=splits), cand_doc(0.5625, splits=splits))

        assert verdict.outcome == "REFUSED"
        assert verdict.reasons == (
            f"split labels are mixed in the base run: {counted} records have one, "
            "so there is no decision set",
        )

    def test_a_candidate_with_an_extra_query_only_is_refused(self) -> None:
        verdict = verdict_of(base_doc(), cand_doc(0.5625, ids=query_ids(21)))

        assert verdict.outcome == "REFUSED"
        assert verdict.reasons == (
            "the runs cover different queries: only in base: none; only in cand: q21",
        )

    def test_several_reasons_are_all_listed_in_the_order_of_the_rule(self) -> None:
        suite = {**SUITE, "repo_sha": "d" * 40}
        cand = cand_doc(0.5625, ids=query_ids(19), suite=suite)

        verdict = verdict_of(base_doc(errors={"q01": "boom"}), cand)

        assert [reason.split(":")[0] for reason in verdict.reasons] == [
            "suite identity differs in",
            "the runs cover different queries",
            "base run has 1 record(s) with an error (first",
        ]


class TestLabels:
    @pytest.mark.parametrize("label", ["split", "kind", "lang"])
    def test_the_runs_must_label_every_query_alike(self, label: str) -> None:
        values = {"split": ["dev", "test"], "kind": ["nl", "keyword"], "lang": ["python", "go"]}
        first, second = values[label]
        base_labels = {f"{label}s": [first] * 20}
        cand_labels = {f"{label}s": [first] * 2 + [second] + [first] * 17}

        verdict = verdict_of(base_doc(**base_labels), cand_doc(0.5625, **cand_labels))

        assert verdict.outcome == "REFUSED"
        assert verdict.reasons == ("the runs label queries differently: q03",)


class TestFrozenRuns:
    @pytest.mark.parametrize(
        "flag", ["rerank", "hyde_fallback_enabled", "multi_query_enabled", "flare_enabled"]
    )
    @pytest.mark.parametrize("side", ["base", "candidate"])
    def test_a_run_with_a_frozen_flag_on_is_refused(self, flag: str, side: str) -> None:
        pipeline = {**pipeline_doc(), flag: True}
        base = base_doc(pipeline=pipeline if side == "base" else pipeline_doc())
        cand = cand_doc(0.5625, pipeline=pipeline if side == "candidate" else pipeline_doc())

        verdict = verdict_of(base, cand)

        assert verdict.outcome == "REFUSED"
        assert verdict.reasons == (f"{side} is not a frozen-plan, rerank-off run: {flag} is true",)

    def test_a_run_whose_plans_were_not_replayed_is_refused(self) -> None:
        pipeline = {**pipeline_doc(), "plans": "recorded"}

        verdict = verdict_of(base_doc(pipeline=pipeline), cand_doc(0.5625, pipeline=pipeline))

        assert verdict.reasons == (
            "base is not a frozen-plan, rerank-off run: plans is 'recorded', not 'replayed'",
            "candidate is not a frozen-plan, rerank-off run: plans is 'recorded', not 'replayed'",
        )


class TestFromFiles:
    def test_three_good_files_are_judged(self, tmp_path: Path) -> None:
        base = write_json(tmp_path, "base.json", base_doc())
        cand = write_json(tmp_path, "cand.json", cand_doc(0.5625))

        verdict = compare_files(base, cand, write_prereg(tmp_path))

        assert verdict.outcome == "PASS"

    def test_every_unreadable_file_is_reported_at_once(self, tmp_path: Path) -> None:
        missing = str(tmp_path / "absent.json")

        verdict = compare_files(missing, missing, str(tmp_path / "absent.yaml"))

        assert verdict.outcome == "REFUSED"
        assert [reason.split(" ")[0] for reason in verdict.reasons] == ["base", "cand", "prereg"]
        assert verdict.reasons[0] == (
            f"base {missing}: not a results.json: cannot read the file: "
            "[Errno 2] No such file or directory: " + repr(missing)
        )
        assert verdict.lines == ("verdict: REFUSED",)

    def test_a_malformed_results_file_names_its_role_and_path(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("{}", encoding="utf-8")
        cand = write_json(tmp_path, "cand.json", cand_doc(0.5625))

        verdict = compare_files(str(bad), cand, write_prereg(tmp_path))

        assert verdict.outcome == "REFUSED"
        assert len(verdict.reasons) == 1
        assert verdict.reasons[0].startswith(f"base {bad}: not a results.json: ")

    def test_no_prereg_is_a_refusal(self, tmp_path: Path) -> None:
        base = write_json(tmp_path, "base.json", base_doc())
        cand = write_json(tmp_path, "cand.json", cand_doc(0.5625))

        verdict = compare_files(base, cand, None)

        assert verdict.outcome == "REFUSED"
        assert verdict.reasons == (
            "prereg: --prereg is required: the hypothesis is fixed before the run",
        )

    def test_a_bad_prereg_lists_its_problems_with_its_path(self, tmp_path: Path) -> None:
        base = write_json(tmp_path, "base.json", base_doc())
        cand = write_json(tmp_path, "cand.json", cand_doc(0.5625))
        prereg = write_prereg(tmp_path, alpha="0.5", min_queries="3")

        verdict = compare_files(base, cand, prereg)

        assert len(verdict.reasons) == 2
        assert verdict.reasons[0].startswith(f"prereg {prereg}: alpha must be")
        assert verdict.reasons[1].startswith(f"prereg {prereg}: min_queries must be")

    def test_an_unexpected_exception_is_a_refusal(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(*_args: object) -> Verdict:
            raise RuntimeError("boom")

        monkeypatch.setattr(compare_module, "compare", boom)
        base = write_json(tmp_path, "base.json", base_doc())
        cand = write_json(tmp_path, "cand.json", cand_doc(0.5625))

        verdict = compare_files(base, cand, write_prereg(tmp_path))

        assert verdict.outcome == "REFUSED"
        assert verdict.exit_code == 3
        assert verdict.reasons == ("internal error: RuntimeError: boom",)
        assert verdict.lines == ("verdict: REFUSED",)
