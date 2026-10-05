"""Golden v2: the optional entry fields, and the v1 loader that must not change for a v1 file.

v2 adds `id`, `lang`, `kind`, `source`, `gold_status` and `split` to a golden line. All are
optional. The loader (`harness._parse_golden`) refuses one that is present and wrong-typed,
with the line number, and otherwise behaves exactly as it did before v2.

MUTATIONS THAT MUST MAKE THIS FILE FAIL
---------------------------------------
1. Remove the `validate_entry(item)` call from `_parse_golden`
   -> `test_the_loader_refuses_a_wrong_field_naming_its_line` fails.
2. Make that call stop the line's other checks (a `continue` after a v2 problem)
   -> `test_a_line_with_a_v2_problem_and_a_v1_problem_reports_both` fails.
3. Let `_CHOICE_FIELDS` skip `kind`, `gold_status` or `split`
   -> the matching rows of `test_a_wrong_value_is_a_problem_with_a_literal_message` fail.
4. Accept a `null` v2 field (`item.get(name) is not None`), or a blank `id` / `lang` / `source`
   (drop `not value.strip()`)
   -> the `kind`-`None` and the blank-string rows of the same test fail.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from trelix.eval.golden import GOLD_STATUSES, KINDS, SPLITS, validate_entry
from trelix.eval.harness import _parse_golden

_REPO_ROOT = Path(__file__).resolve().parents[2]
_V1_ENTRY = {"query": "how does auth work", "relevant_files": ["src/auth.py"]}


def _write(tmp_path: Path, entries: list[dict[str, object]]) -> Path:
    path = tmp_path / "golden.jsonl"
    path.write_text("".join(json.dumps(e) + "\n" for e in entries), encoding="utf-8")
    return path


class TestTheSchemaConstants:
    def test_the_allowed_values_are_these_and_no_others(self) -> None:
        assert KINDS == ("nl", "keyword", "commit", "issue")
        assert GOLD_STATUSES == ("validated", "pooled", "unreviewed")
        assert SPLITS == ("dev", "test")


class TestValidateEntry:
    def test_a_v1_entry_has_no_problems(self) -> None:
        assert validate_entry(_V1_ENTRY) == []

    def test_keys_it_does_not_know_are_allowed(self) -> None:
        assert validate_entry({**_V1_ENTRY, "difficulty": "hard", "area": "indexing"}) == []

    @pytest.mark.parametrize(
        ("name", "value"),
        [
            ("kind", "nl"),
            ("kind", "keyword"),
            ("kind", "commit"),
            ("kind", "issue"),
            ("gold_status", "validated"),
            ("gold_status", "pooled"),
            ("gold_status", "unreviewed"),
            ("split", "dev"),
            ("split", "test"),
            ("id", "q-0001"),
            ("lang", "python"),
            ("source", "issue #12"),
        ],
    )
    def test_an_allowed_value_is_accepted(self, name: str, value: str) -> None:
        assert validate_entry({**_V1_ENTRY, name: value}) == []

    def test_every_field_together_is_accepted(self) -> None:
        full = {
            **_V1_ENTRY,
            "id": "a1",
            "lang": "go",
            "kind": "commit",
            "source": "git log",
            "gold_status": "pooled",
            "split": "test",
        }
        assert validate_entry(full) == []

    @pytest.mark.parametrize(
        ("name", "value", "message"),
        [
            ("kind", "code", "\"kind\" must be one of nl, keyword, commit, issue (got 'code')"),
            ("kind", "NL", "\"kind\" must be one of nl, keyword, commit, issue (got 'NL')"),
            ("kind", "", "\"kind\" must be one of nl, keyword, commit, issue (got '')"),
            ("kind", 5, '"kind" must be one of nl, keyword, commit, issue (got 5)'),
            ("kind", None, '"kind" must be one of nl, keyword, commit, issue (got None)'),
            ("kind", ["nl"], "\"kind\" must be one of nl, keyword, commit, issue (got ['nl'])"),
            (
                "gold_status",
                "done",
                "\"gold_status\" must be one of validated, pooled, unreviewed (got 'done')",
            ),
            (
                "gold_status",
                True,
                '"gold_status" must be one of validated, pooled, unreviewed (got True)',
            ),
            ("split", "train", "\"split\" must be one of dev, test (got 'train')"),
            ("split", 1, '"split" must be one of dev, test (got 1)'),
            ("id", "", '"id" must be a non-empty string'),
            ("id", 7, '"id" must be a non-empty string'),
            ("lang", "   ", '"lang" must be a non-empty string'),
            ("lang", None, '"lang" must be a non-empty string'),
            ("source", 1.5, '"source" must be a non-empty string'),
            ("source", ["x"], '"source" must be a non-empty string'),
        ],
    )
    def test_a_wrong_value_is_a_problem_with_a_literal_message(
        self, name: str, value: object, message: str
    ) -> None:
        assert validate_entry({**_V1_ENTRY, name: value}) == [message]

    def test_every_bad_field_is_reported_not_just_the_first(self) -> None:
        bad = {**_V1_ENTRY, "kind": "code", "split": "train", "id": ""}
        assert validate_entry(bad) == [
            '"id" must be a non-empty string',
            "\"kind\" must be one of nl, keyword, commit, issue (got 'code')",
            "\"split\" must be one of dev, test (got 'train')",
        ]


class TestTheLoaderIsUnchangedForV1:
    def test_the_shipped_golden_file_loads_as_before(self) -> None:
        entries = _parse_golden(_REPO_ROOT / "eval" / "golden.jsonl")

        assert len(entries) == 54
        assert [e.line_no for e in entries] == list(range(1, 55))
        first = entries[0]
        assert first.query.startswith("Does a .gitignore sitting in a subdirectory")
        assert first.relevant_files == frozenset(
            {"src/trelix/indexing/walker.py", "docs/CONFIGURATION.md"}
        )
        assert first.area == "indexing"
        assert Counter(e.area for e in entries) == {
            "indexing": 10,
            "retrieval": 10,
            "storage": 10,
            "llm-embed": 10,
            "cli-config-graph": 10,
            "ops": 4,
        }

    def test_hand_made_v1_lines_load_with_their_line_numbers(self, tmp_path: Path) -> None:
        path = tmp_path / "golden.jsonl"
        path.write_text(
            '{"query": "q one", "relevant_files": ["a.py", "b/c.py"]}\n'
            "\n"
            '{"query": "q two", "relevant_files": ["d.py"], "area": "x", "difficulty": "hard"}\n',
            encoding="utf-8",
        )

        entries = _parse_golden(path)

        assert [(e.line_no, e.query, e.area) for e in entries] == [
            (1, "q one", None),
            (3, "q two", "x"),
        ]
        assert entries[0].relevant_files == frozenset({"a.py", "b/c.py"})

    def test_an_unknown_key_is_still_allowed(self, tmp_path: Path) -> None:
        path = _write(tmp_path, [{**_V1_ENTRY, "reviewer": "someone", "nested": {"a": [1]}}])

        assert [e.query for e in _parse_golden(path)] == ["how does auth work"]


class TestTheLoaderAcceptsV2Fields:
    def test_a_fully_labelled_line_loads(self, tmp_path: Path) -> None:
        full = {
            **_V1_ENTRY,
            "id": "a1",
            "lang": "python",
            "kind": "nl",
            "source": "docs",
            "gold_status": "validated",
            "split": "dev",
        }
        path = _write(tmp_path, [full, _V1_ENTRY | {"query": "another"}])

        entries = _parse_golden(path)

        assert [e.query for e in entries] == ["how does auth work", "another"]

    @pytest.mark.parametrize(
        ("name", "value"),
        [("kind", "issue"), ("gold_status", "pooled"), ("split", "test"), ("id", "x")],
    )
    def test_a_single_valid_field_loads(self, tmp_path: Path, name: str, value: str) -> None:
        path = _write(tmp_path, [{**_V1_ENTRY, name: value}])

        assert len(_parse_golden(path)) == 1


class TestTheLoaderRefusesWrongV2Fields:
    @pytest.mark.parametrize(
        ("name", "value"),
        [
            ("kind", "code"),
            ("kind", "NL"),
            ("kind", 3),
            ("kind", None),
            ("gold_status", "done"),
            ("gold_status", ["validated"]),
            ("split", "train"),
            ("split", 0),
            ("id", ""),
            ("id", 12),
            ("lang", " "),
            ("source", False),
        ],
    )
    def test_the_loader_refuses_a_wrong_field_naming_its_line(
        self, tmp_path: Path, name: str, value: object
    ) -> None:
        path = _write(tmp_path, [_V1_ENTRY, {**_V1_ENTRY, "query": "second", name: value}])

        with pytest.raises(ValueError) as refused:
            _parse_golden(path)

        message = str(refused.value)
        assert f'line 2: "{name}" must be ' in message
        assert "line 1" not in message
        assert "1 unusable golden entry" in message

    def test_the_loader_refuses_the_exact_message_for_a_wrong_kind(self, tmp_path: Path) -> None:
        path = _write(tmp_path, [{**_V1_ENTRY, "kind": "code"}])

        with pytest.raises(ValueError) as refused:
            _parse_golden(path)

        assert "line 1: \"kind\" must be one of nl, keyword, commit, issue (got 'code')" in str(
            refused.value
        )

    def test_the_loader_reports_every_bad_line_at_once(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            [
                {**_V1_ENTRY, "kind": "code"},
                _V1_ENTRY | {"query": "fine"},
                {**_V1_ENTRY, "query": "bad split", "split": "train"},
            ],
        )

        with pytest.raises(ValueError) as refused:
            _parse_golden(path)

        message = str(refused.value)
        assert "2 unusable golden entries" in message
        assert 'line 1: "kind"' in message
        assert 'line 3: "split"' in message
        assert "line 2" not in message

    def test_a_line_with_a_v2_problem_and_a_v1_problem_reports_both(self, tmp_path: Path) -> None:
        path = _write(tmp_path, [{"query": "q", "relevant_files": [], "kind": "code"}])

        with pytest.raises(ValueError) as refused:
            _parse_golden(path)

        message = str(refused.value)
        assert "2 unusable golden entries" in message
        assert 'line 1: "kind" must be one of' in message
        assert 'line 1: "relevant_files" must be a non-empty list' in message

    def test_a_v1_refusal_is_still_a_refusal_next_to_valid_v2_fields(self, tmp_path: Path) -> None:
        path = _write(tmp_path, [{"query": "q", "relevant_files": [], "kind": "nl"}])

        with pytest.raises(ValueError, match="line 1"):
            _parse_golden(path)
