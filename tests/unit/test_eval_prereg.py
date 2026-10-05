"""The pre-registration loader (trelix.eval.prereg): every key required, every range closed.

Expected values are literals. The sample file is `tests.unit.eval_compare_fixtures.prereg_text()`.

MUTATIONS THAT MUST MAKE THIS FILE FAIL
---------------------------------------
1. `0 < value <= high` -> `0 < value < high` in `_real`      (test_alpha_boundaries)
2. the alpha ceiling `ALPHA_MAX` 0.05 -> 0.5                  (test_alpha_boundaries)
3. the floor `MIN_QUERIES_FLOOR` 20 -> 1                      (test_min_queries_boundaries)
4. a bool accepted as a number or an integer                  (test_a_bool_is_not_a_number)
5. the duplicate-key check removed                            (test_a_duplicate_key_is_refused)
6. the anchor/alias check removed                             (test_anchors_and_aliases_are_refused)
7. `alpha / family < MIN_ALPHA_EFFECTIVE` -> `<=`             (test_the_effective_alpha_floor)
   or the whole floor removed                                 (test_the_effective_alpha_floor)
8. the `MAX_FAMILY_SIZE` upper bound removed                 (test_a_huge_family_size_is_refused...)
9. an unknown key accepted; a missing key accepted            (test_unknown_and_missing_keys)
10. the `cost_class` lookup changed for one class             (test_the_hurdle_table)
11. the size cap removed                                     (test_an_oversize_file_...)
12. `safe_load` swapped for an unsafe loader                  (test_a_python_tag_is_refused_not_run)
13. the problem list no longer capped                         (test_the_problem_list_is_capped)
14. the `isinstance(cost_class, str)` guard removed           (test_a_list_or_mapping_...)
15. `clip` no longer collapsing line breaks                   (test_broken_yaml_is_refused)
16. `ValueError` no longer caught around the YAML load         (test_a_date_pyyaml_cannot_build_...)
17. the size cap 64 KiB or the 60-character clip changed       (test_the_cap_is_64_kib,
                                                                test_a_long_value_is_quoted_...)
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import trelix.eval.prereg as prereg_module
from tests.unit.eval_compare_fixtures import PREREG_DEFAULTS, prereg_text
from trelix.eval.prereg import Prereg, PreregError, load_prereg


def _write(tmp_path: Path, text: str, name: str = "exp.yaml") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def load_text(tmp_path: Path, text: str) -> Prereg:
    return load_prereg(_write(tmp_path, text))


def problems_of(tmp_path: Path, text: str) -> tuple[str, ...]:
    with pytest.raises(PreregError) as caught:
        load_text(tmp_path, text)
    return caught.value.problems


class TestTheSampleFile:
    def test_loads_to_the_written_values(self, tmp_path: Path) -> None:
        prereg = load_prereg(_write(tmp_path, prereg_text()))

        assert prereg.experiment_id == "EXP-example"
        assert prereg.comparison_id == "baseline..flag-on"
        assert prereg.primary_metric == "ndcg@10"
        assert prereg.direction == "increase"
        assert prereg.expected_effect == 0.03
        assert prereg.alpha == 0.05
        assert prereg.family_size == 1
        assert prereg.min_queries == 20
        assert prereg.cost_class == "flag"

    def test_the_sample_in_the_readme_loads(self, tmp_path: Path) -> None:
        readme = Path(__file__).resolve().parents[2] / "eval" / "README.md"
        text = readme.read_text(encoding="utf-8")
        blocks = re.findall(r"```yaml\n(.*?)```", text, re.DOTALL)
        block = next(item for item in blocks if "experiment_id:" in item)

        prereg = load_prereg(_write(tmp_path, block))

        assert prereg.min_queries == 30
        assert prereg.cost_class == "flag"

    @pytest.mark.parametrize(
        ("cost_class", "hurdle"), [("flag", 0.01), ("index", 0.02), ("heavy", 0.03)]
    )
    def test_the_hurdle_table(self, tmp_path: Path, cost_class: str, hurdle: float) -> None:
        prereg = load_prereg(_write(tmp_path, prereg_text(cost_class=cost_class)))

        assert prereg.hurdle == hurdle

    def test_alpha_is_shared_out_over_the_family(self, tmp_path: Path) -> None:
        prereg = load_prereg(_write(tmp_path, prereg_text(alpha=0.04, family_size=4)))

        assert prereg.alpha_effective == 0.01

    def test_an_integer_effect_is_a_number(self, tmp_path: Path) -> None:
        prereg = load_prereg(_write(tmp_path, prereg_text(expected_effect=1)))

        assert prereg.expected_effect == 1.0
        assert isinstance(prereg.expected_effect, float)


class TestEveryRangeHasItsBoundary:
    @pytest.mark.parametrize(
        ("alpha", "accepted"),
        [
            ("0.05", True),
            ("0.0500001", False),
            ("0.001", True),
            ("0", False),
            ("-0.01", False),
            (".nan", False),
            (".inf", False),
            ("'0.05'", False),
            ("true", False),
        ],
    )
    def test_alpha_boundaries(self, tmp_path: Path, alpha: str, accepted: bool) -> None:
        text = prereg_text(alpha=alpha)
        if accepted:
            assert load_text(tmp_path, text).alpha == float(alpha)
        else:
            assert any(p.startswith("alpha must be") for p in problems_of(tmp_path, text))

    @pytest.mark.parametrize(
        ("minimum", "accepted"),
        [
            ("20", True),
            ("19", False),
            ("0", False),
            ("20.0", False),
            ("true", False),
            ("'30'", False),
        ],
    )
    def test_min_queries_boundaries(self, tmp_path: Path, minimum: str, accepted: bool) -> None:
        text = prereg_text(min_queries=minimum)
        if accepted:
            assert load_text(tmp_path, text).min_queries == 20
        else:
            assert any(p.startswith("min_queries must be") for p in problems_of(tmp_path, text))

    @pytest.mark.parametrize(
        ("family", "accepted"),
        [("1", True), ("0", False), ("1.0", False), ("yes", False), ("-2", False)],
    )
    def test_family_size_boundaries(self, tmp_path: Path, family: str, accepted: bool) -> None:
        text = prereg_text(family_size=family)
        if accepted:
            assert load_text(tmp_path, text).family_size == 1
        else:
            assert any(p.startswith("family_size must be") for p in problems_of(tmp_path, text))

    @pytest.mark.parametrize(
        ("effect", "accepted"),
        [
            ("1", True),
            ("0.001", True),
            ("1.0000001", False),
            ("0", False),
            ("-0.1", False),
            ("true", False),
            (".nan", False),
            (".inf", False),
            ("'0.03'", False),
        ],
    )
    def test_expected_effect_boundaries(self, tmp_path: Path, effect: str, accepted: bool) -> None:
        text = prereg_text(expected_effect=effect)
        if accepted:
            assert load_text(tmp_path, text).expected_effect > 0
        else:
            assert any(p.startswith("expected_effect must be") for p in problems_of(tmp_path, text))

    @pytest.mark.parametrize("field", ["alpha", "family_size", "min_queries", "expected_effect"])
    def test_a_bool_is_not_a_number(self, tmp_path: Path, field: str) -> None:
        # `true` is an int in Python, so 1 <= True <= 50 would pass without the type test.
        problems = problems_of(tmp_path, prereg_text(**{field: "true"}))

        assert any(p.startswith(f"{field} must be") for p in problems)


class TestTheEffectiveAlphaFloor:
    def test_the_effective_alpha_floor(self, tmp_path: Path) -> None:
        # 0.05 / 50 is exactly 0.001: on the floor, so accepted; one more comparison is not.
        on_the_floor = load_text(tmp_path, prereg_text(alpha="0.05", family_size="50"))
        assert on_the_floor.family_size == 50

        below = problems_of(tmp_path, prereg_text(alpha="0.04", family_size="50"))
        assert below == (
            "alpha / family_size must be at least 0.001 (got 0.0008): "
            "a tail of the bootstrap would hold fewer than 5 replicates",
        )

    def test_a_small_alpha_alone_can_sit_below_the_floor(self, tmp_path: Path) -> None:
        problems = problems_of(tmp_path, prereg_text(alpha="0.0009", family_size="1"))

        assert len(problems) == 1
        assert problems[0].startswith("alpha / family_size must be at least 0.001")

    def test_a_huge_family_size_is_refused_not_an_overflow(self, tmp_path: Path) -> None:
        huge = "1" + "0" * 400  # dividing 0.05 by it would raise OverflowError

        problems = problems_of(tmp_path, prereg_text(family_size=huge))

        assert len(problems) == 1
        assert problems[0].startswith("family_size must be an integer from 1 to 50")

    def test_fifty_one_comparisons_are_too_many(self, tmp_path: Path) -> None:
        problems = problems_of(tmp_path, prereg_text(family_size="51"))

        assert any(p.startswith("family_size must be an integer from 1 to 50") for p in problems)


class TestTheIdentityFields:
    @pytest.mark.parametrize(
        ("field", "value", "fragment"),
        [
            ("experiment_id", "d", "experiment_id must look like EXP-<id>"),
            ("experiment_id", "EXP-", "experiment_id must look like EXP-<id>"),
            ("experiment_id", '"EXP-x\\n"', "experiment_id must look like EXP-<id>"),
            ("comparison_id", "a..a", "comparison_id must be <baseline arm>..<candidate arm>"),
            ("comparison_id", "baseline-flag-on", "comparison_id must be"),
            ("comparison_id", "Base..cand", "comparison_id must be"),
            ("comparison_id", "a..b..c", "comparison_id must be"),
            ("primary_metric", "recall@10", "primary_metric must be 'ndcg@10'"),
            ("direction", "decrease", "direction must be 'increase'"),
            ("cost_class", "heavyweight", "cost_class must be one of flag, index, heavy"),
            ("cost_class", "Flag", "cost_class must be one of flag, index, heavy"),
        ],
    )
    def test_a_bad_value_is_refused(
        self, tmp_path: Path, field: str, value: str, fragment: str
    ) -> None:
        problems = problems_of(tmp_path, prereg_text(**{field: value}))

        assert len(problems) == 1
        assert fragment in problems[0]

    def test_a_long_value_is_quoted_cut_to_60_characters(self, tmp_path: Path) -> None:
        problems = problems_of(tmp_path, prereg_text(cost_class="x" * 100))

        assert problems == (
            "cost_class must be one of flag, index, heavy (got '" + "x" * 56 + "...)",
        )

    @pytest.mark.parametrize(
        ("value", "shown"), [("[flag]", "['flag']"), ("{a: 1}", "{'a': 1}"), ("[]", "[]")]
    )
    def test_a_list_or_mapping_cost_class_is_a_problem_not_a_crash(
        self, tmp_path: Path, value: str, shown: str
    ) -> None:
        # Unhashable: a bare `in HURDLES` lookup would raise TypeError and the bad alpha below
        # would never be reported.
        problems = problems_of(tmp_path, prereg_text(cost_class=value, alpha="0.5"))

        assert problems == (
            f"cost_class must be one of flag, index, heavy (got {shown})",
            "alpha must be a number above 0 and at most 0.05 (got 0.5)",
        )

    @pytest.mark.parametrize("version", ["2", "'1'", "true", "1.0", "null"])
    def test_another_schema_version_is_unsupported(self, tmp_path: Path, version: str) -> None:
        problems = problems_of(tmp_path, prereg_text(schema_version=version))

        assert problems[0].startswith("unsupported schema_version ")
        assert problems[0].endswith("this reader understands 1")


class TestUnknownAndMissingKeys:
    def test_unknown_and_missing_keys(self, tmp_path: Path) -> None:
        text = prereg_text(alpa="0.05", alpha=None)

        assert problems_of(tmp_path, text) == ("unsupported key 'alpa'", "missing key 'alpha'")

    @pytest.mark.parametrize("key", sorted(PREREG_DEFAULTS))
    def test_every_key_is_required(self, tmp_path: Path, key: str) -> None:
        problems = problems_of(tmp_path, prereg_text(**{key: None}))

        assert problems == (f"missing key {key!r}",)

    def test_a_file_with_three_problems_reports_all_three(self, tmp_path: Path) -> None:
        text = prereg_text(alpha="0.5", min_queries="3", cost_class="big")

        problems = problems_of(tmp_path, text)

        assert [p.split(" ")[0] for p in problems] == ["cost_class", "alpha", "min_queries"]

    def test_the_problem_list_is_capped(self, tmp_path: Path) -> None:
        extra = "".join(f"k{number}: 1\n" for number in range(30))

        problems = problems_of(tmp_path, prereg_text() + extra)

        assert len(problems) == 21
        assert problems[-1] == "... and 10 more problems"


class TestTheYamlIsStrict:
    def test_a_duplicate_key_is_refused(self, tmp_path: Path) -> None:
        problems = problems_of(tmp_path, prereg_text() + "alpha: 0.01\n")

        assert len(problems) == 1
        assert problems[0].startswith("not valid YAML: ")
        assert "duplicate key 'alpha'" in problems[0]

    def test_anchors_and_aliases_are_refused(self, tmp_path: Path) -> None:
        anchored = problems_of(tmp_path, prereg_text(alpha="&a 0.05"))
        aliased = problems_of(tmp_path, prereg_text(alpha="&a 0.05", expected_effect="*a"))

        assert "anchors and aliases are not allowed" in anchored[0]
        assert "anchors and aliases are not allowed" in aliased[0]

    def test_a_python_tag_is_refused_not_run(self, tmp_path: Path) -> None:
        problems = problems_of(tmp_path, prereg_text(alpha="!!python/name:os.getcwd"))

        assert problems[0].startswith("not valid YAML: ")

    def test_an_unhashable_key_is_a_yaml_error_not_a_crash(self, tmp_path: Path) -> None:
        problems = problems_of(tmp_path, "? [a, b]\n: 1\n")

        assert problems[0].startswith("not valid YAML: ")

    @pytest.mark.parametrize("text", ["", "- a\n- b\n", "just a string\n", "42\n", "null\n"])
    def test_the_top_level_must_be_a_mapping(self, tmp_path: Path, text: str) -> None:
        problems = problems_of(tmp_path, text)

        assert problems == ("the top level must be a mapping with the ten pre-registration keys",)

    def test_two_documents_are_refused(self, tmp_path: Path) -> None:
        problems = problems_of(tmp_path, prereg_text() + "---\n" + prereg_text())

        assert problems[0].startswith("not valid YAML: ")

    def test_a_date_pyyaml_cannot_build_is_a_listed_problem_not_a_crash(
        self, tmp_path: Path
    ) -> None:
        problems = problems_of(tmp_path, prereg_text(alpha="2026-13-45"))

        assert len(problems) == 1
        assert problems[0].startswith("not valid YAML: ")
        assert "month must be in 1..12" in problems[0]

    def test_broken_yaml_is_refused(self, tmp_path: Path) -> None:
        problems = problems_of(tmp_path, "alpha: [0.05\n")

        assert problems[0].startswith("not valid YAML: ")
        assert "\n" not in problems[0]  # PyYAML's message spans five lines; a problem is one


class TestTheFileItself:
    def test_a_missing_file_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(PreregError) as caught:
            load_prereg(tmp_path / "absent.yaml")

        assert caught.value.problems[0].startswith("cannot read the pre-registration file: ")

    def test_a_non_utf8_file_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "exp.yaml"
        path.write_bytes(b"alpha: \xff\xfe\n")

        with pytest.raises(PreregError) as caught:
            load_prereg(path)

        assert caught.value.problems[0].startswith("cannot read the pre-registration file: ")

    def test_an_oversize_file_is_refused_unread(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(prereg_module, "MAX_PREREG_BYTES", 100)
        path = _write(tmp_path, prereg_text() + "# " + "x" * 200 + "\n")

        with pytest.raises(PreregError) as caught:
            load_prereg(path)

        assert caught.value.problems == (
            "cannot read the pre-registration file: is larger than 100 bytes",
        )

    def test_the_cap_is_64_kib(self) -> None:
        assert prereg_module.MAX_PREREG_BYTES == 64 * 1024

    def test_a_file_exactly_at_the_cap_is_read(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        text = prereg_text()
        monkeypatch.setattr(prereg_module, "MAX_PREREG_BYTES", len(text.encode("utf-8")))

        assert load_prereg(_write(tmp_path, text)).experiment_id == "EXP-example"
