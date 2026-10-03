"""How the review workflow publishes a review that covered only part of the diff (exit 4).

``trelix review`` exits 4 after printing the findings it has, and writes a JSON record of which
hunks it left unreviewed to ``TRELIX_REVIEW_OUTCOME_FILE``. The publishing script of
``.github/workflows/trelix-review.yml`` reads both. Two things about that record are the subject
here, on top of the conclusion table in ``test_review_not_run_exit.py``:

* it is read as untrusted input. The runner's ``/tmp`` is shared with steps that run the pull
  request's content, so a record that is missing, is not a regular file, is oversized, is not JSON
  or does not add up is "unknown", and the check says so in words. It is never read as "nothing
  was left out".
* its file names are pull-request text. Each is shown only inside one code span, on one line, so
  that nothing in a name can start a heading, close the span, link, mention or hide text. The
  checks below do not look at how the script does it: they look at every line of the summary.

Both read the record, so both must stay linear on hostile input: the time bounds here are CPU time
spent by the script, for 50,000-character names of the shapes that make a parser or a
backtracking regular expression slow.
"""

from __future__ import annotations

import copy
import json
import re
import unicodedata
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from tests.unit.review_workflow_harness import (
    NODE,
    Published,
    findings_json,
    load_cases,
    run_publish_script,
)

_needs_node = pytest.mark.skipif(NODE is None, reason="node is required to run the script")

_VALID: dict[str, Any] = load_cases()["valid_outcome"]
_INVALID: list[dict[str, str]] = load_cases()["invalid_outcomes"]
_MAX_RECORD_BYTES = 1024 * 1024
_TIME_LIMIT_MS = 2000
_HOSTILE_SIZE = 50_000

_UNKNOWN = (
    "trelix reviewed only part of PR #7, but the record of what it covered is missing or "
    "unreadable, so how much was left unreviewed is unknown. This is not a clean result."
)
_FIRST_LINE = re.compile(
    r"^trelix reviewed only part of PR #7: \d+ of \d+ hunks were reviewed and \d+ were not\. "
    r"This is not a clean result\.$"
)
_ENTRY = re.compile(r"^- `([^`\n]+)` \((truncated|refused|parse_failed|error)\)$")
_MORE = re.compile(r"^- and \d+ more\.$")
_FIXED_LINES = {
    "",
    "No findings in the hunks that were reviewed.",
    "Hunks that were not reviewed:",
    'See the "Run trelix review" step log for the reason.',
}
_HIDDEN = {"Cc", "Cf", "Cs", "Zl", "Zp"}


def _record(**changes: Any) -> dict[str, Any]:
    record = copy.deepcopy(_VALID)
    record.update(changes)
    return record


def _one_hunk(file: str, status: str = "error", line: int = 3) -> dict[str, Any]:
    return _record(
        hunks_total=2,
        hunks_reviewed=1,
        hunks_unreviewed=1,
        hunks=[{"file": file, "line": line, "status": status, "detail": "x"}],
        hunks_omitted=0,
    )


def _publish(
    tmp_path: Path,
    record: str | dict[str, Any] | None,
    severities: list[str] | None = None,
    *,
    wide: bool = False,
) -> Published:
    """Publish exit 4 with this record. `wide` writes characters as they are, not as \\u escapes:
    the only way five 50,000-character names stay under the size limit of the record."""
    text = json.dumps(record, ensure_ascii=not wide) if isinstance(record, dict) else record
    return run_publish_script(tmp_path, "4", findings_json(severities or []), outcome=text)


def _assert_only_known_lines(summary: str, *, entries: int) -> list[str]:
    """Every line of the summary is one the script writes itself; returns the code-span texts."""
    lines = summary.split("\n")
    spans: list[str] = []
    for position, line in enumerate(lines):
        if position == 0:
            assert _FIRST_LINE.match(line), line
        elif line in _FIXED_LINES or _MORE.match(line):
            continue
        elif re.match(r"^\d+ issue\(s\) found in the hunks that were reviewed\.$", line):
            continue
        elif match := _ENTRY.match(line):
            spans.append(match.group(1))
        else:
            raise AssertionError(f"line {position} is not one the script writes: {line!r}")
    assert len(spans) == entries
    # No code span but the ones around the entries: a backtick in a name did not survive.
    assert summary.count("`") == 2 * entries
    return spans


@_needs_node
class TestARecordThatIsNotTrustworthyIsUnknown:
    def test_the_table_of_bad_records_is_not_vacuous(self) -> None:
        assert len(_INVALID) >= 30

    @pytest.mark.parametrize("record", _INVALID, ids=[r["id"] for r in _INVALID])
    def test_the_check_says_it_is_unknown_and_uses_nothing_from_it(
        self, tmp_path: Path, record: dict[str, str]
    ) -> None:
        """The same records the GitHub App's parseOutcomeRecord rejects (review-outcome.test.ts)."""
        published = _publish(tmp_path, record["text"], ["WARN"])

        summary = published.call["output"]["summary"]
        assert published.call["output"]["title"] == "trelix review incomplete"
        assert published.call["conclusion"] == "neutral"
        assert summary.split("\n")[0] == _UNKNOWN
        assert "Hunks that were not reviewed" not in summary
        assert "3 of 5" not in summary
        assert "src/a.py" not in summary

    def test_no_file_at_all_is_unknown(self, tmp_path: Path) -> None:
        published = _publish(tmp_path, None, ["WARN"])

        assert published.call["output"]["summary"].split("\n")[0] == _UNKNOWN

    def test_a_directory_at_the_path_is_unknown(self, tmp_path: Path) -> None:
        published = run_publish_script(
            tmp_path, "4", findings_json([]), outcome=None, before_run=lambda path: path.mkdir()
        )

        assert published.call["output"]["summary"].split("\n")[0] == _UNKNOWN

    def test_a_symlink_to_a_valid_record_is_unknown(self, tmp_path: Path) -> None:
        elsewhere = tmp_path / "elsewhere.json"
        elsewhere.write_text(json.dumps(_VALID), encoding="utf-8")

        published = run_publish_script(
            tmp_path,
            "4",
            findings_json([]),
            outcome=None,
            before_run=lambda path: path.symlink_to(elsewhere),
        )

        assert published.call["output"]["summary"].split("\n")[0] == _UNKNOWN

    def test_a_record_of_exactly_the_size_limit_is_read(self, tmp_path: Path) -> None:
        document = json.dumps(_VALID)
        padded = document + " " * (_MAX_RECORD_BYTES - len(document.encode("utf-8")))
        assert len(padded.encode("utf-8")) == _MAX_RECORD_BYTES

        published = _publish(tmp_path, padded)

        assert "3 of 5 hunks were reviewed and 2 were not" in published.call["output"]["summary"]

    def test_a_record_one_byte_over_the_size_limit_is_unknown(self, tmp_path: Path) -> None:
        document = json.dumps(_VALID)
        padded = document + " " * (_MAX_RECORD_BYTES + 1 - len(document.encode("utf-8")))
        assert len(padded.encode("utf-8")) == _MAX_RECORD_BYTES + 1

        published = _publish(tmp_path, padded)

        assert published.call["output"]["summary"].split("\n")[0] == _UNKNOWN

    def test_an_unknown_record_still_fails_the_check_for_an_error_finding(
        self, tmp_path: Path
    ) -> None:
        published = _publish(tmp_path, "{not json", ["INFO", "ERROR"])

        assert published.call["conclusion"] == "failure"
        assert published.call["output"]["summary"].split("\n")[0] == _UNKNOWN
        assert len(published.call["output"]["annotations"]) == 2

    @pytest.mark.parametrize("exit_code", ["0", "3", "1", "2", "5"])
    def test_the_record_is_only_ever_used_for_exit_4(self, tmp_path: Path, exit_code: str) -> None:
        published = run_publish_script(
            tmp_path, exit_code, findings_json([]), outcome=json.dumps(_VALID)
        )

        assert "Hunks that were not reviewed" not in published.call["output"]["summary"]
        assert "src/a.py" not in published.call["output"]["summary"]


@_needs_node
class TestTheUnreviewedList:
    def test_it_lists_ten_and_counts_the_rest(self, tmp_path: Path) -> None:
        hunks = [
            {"file": f"src/f{i}.py", "line": i + 1, "status": "truncated", "detail": "x"}
            for i in range(100)
        ]
        record = _record(
            hunks_total=200,
            hunks_reviewed=70,
            hunks_unreviewed=130,
            hunks=hunks,
            hunks_omitted=30,
        )

        summary = _publish(tmp_path, record).call["output"]["summary"]

        lines = summary.split("\n")
        entries = [line for line in lines if line.startswith("- `")]
        assert entries[0] == "- `src/f0.py:1` (truncated)"
        assert entries[-1] == "- `src/f9.py:10` (truncated)"
        assert len(entries) == 10
        # 130 unreviewed in all, 10 shown: the other 120 are counted, not just the 20 the
        # record happens to hold.
        assert "- and 120 more." in lines
        assert "70 of 200 hunks were reviewed and 130 were not." in lines[0]

    @pytest.mark.parametrize(("unreviewed", "more"), [(10, None), (11, "- and 1 more.")])
    def test_the_more_line_appears_only_when_something_is_not_listed(
        self, tmp_path: Path, unreviewed: int, more: str | None
    ) -> None:
        hunks = [
            {"file": f"f{i}.py", "line": 1, "status": "error", "detail": "x"}
            for i in range(unreviewed)
        ]
        record = _record(
            hunks_total=unreviewed + 1,
            hunks_reviewed=1,
            hunks_unreviewed=unreviewed,
            hunks=hunks,
            hunks_omitted=0,
        )

        lines = _publish(tmp_path, record).call["output"]["summary"].split("\n")

        assert len([line for line in lines if line.startswith("- `")]) == 10
        assert [line for line in lines if line.startswith("- and")] == ([more] if more else [])

    @pytest.mark.parametrize("status", ["truncated", "refused", "parse_failed", "error"])
    def test_each_status_of_the_vocabulary_is_shown_as_it_is(
        self, tmp_path: Path, status: str
    ) -> None:
        summary = _publish(tmp_path, _one_hunk("src/a.py", status)).call["output"]["summary"]

        assert f"- `src/a.py:3` ({status})" in summary.split("\n")


_HOSTILE_NAMES: dict[str, tuple[str, str]] = {
    # name: (file name in the record, what is left of it in the code span)
    "a heading and a link on new lines": (
        "src/a.py\n# Heading\n- [x](https://attacker.example/canary)",
        "src/a.py # Heading - [x](https://attacker.example/canary)",
    ),
    "carriage returns and line separators": ("a\rb\u2028c\u2029d", "a b c d"),
    "backticks that would close the span": ("a`b`, **bold**", "a\u02cbb\u02cb, **bold**"),
    "a long run of backticks": ("`" * 200, "\u02cb" * 99 + "\u2026"),
    "html": (
        "<img src=x onerror=alert(1)><script>canary</script>",
        "<img src=x onerror=alert(1)><script>canary</script>",
    ),
    "an html comment opener": ("src/<!-- canary", "src/<!-- canary"),
    "a mention and a link": (
        "@octo-org/team [x](https://attacker.example/canary)",
        "@octo-org/team [x](https://attacker.example/canary)",
    ),
    "a bidi override": ("src/\u202eexe.py", "src/exe.py"),
    "zero-width characters": ("sr\u200bc/\u2060a.py", "src/a.py"),
    "tag characters": ("a\U000e0041\U000e0042.py", "a.py"),
    "control characters": ("a\x00b\x1bc\x7fd\x85e.py", "a b c d e.py"),
    "a run of combining marks": ("a" + "\u0301" * 60 + ".py", "a" + "\u0301" * 4 + ".py"),
    "a lone surrogate": ("a\ud800b.py", "ab.py"),
    "nothing but hidden characters": ("\u200b\u200c\u202e", "(unprintable file name)"),
    "an empty name": ("", "(unprintable file name)"),
    "a very long name": ("x" * _HOSTILE_SIZE, "x" * 99 + "\u2026"),
}


@_needs_node
class TestAFileNameIsPullRequestText:
    @pytest.mark.parametrize("name", list(_HOSTILE_NAMES))
    def test_it_stays_inside_one_code_span_on_one_line(self, tmp_path: Path, name: str) -> None:
        file, shown = _HOSTILE_NAMES[name]

        summary = _publish(tmp_path, _one_hunk(file), ["WARN"]).call["output"]["summary"]

        [span] = _assert_only_known_lines(summary, entries=1)
        assert span == f"{shown}:3"
        hidden = [c for c in summary if c != "\n" and unicodedata.category(c) in _HIDDEN]
        assert hidden == []
        assert not re.search("[\u0300-\u036f]{5,}", summary)

    def test_a_name_longer_than_the_limit_is_cut_at_one_hundred_characters(
        self, tmp_path: Path
    ) -> None:
        summary = _publish(tmp_path, _one_hunk("y" * 101)).call["output"]["summary"]

        [span] = _assert_only_known_lines(summary, entries=1)
        assert span == "y" * 99 + "\u2026:3"

    @pytest.mark.parametrize(
        ("file", "shown"),
        [
            # Only the first 400 characters of a name are looked at (four times the 100
            # shown), so what it costs to show a name does not depend on how long it is. The
            # hidden characters are removed before the cap, so a name that is nothing but
            # hidden characters in its first 400 reads as empty, whatever follows.
            ("\u200b" * 400 + "visible.py", "(unprintable file name)"),
            ("\u200b" * 500 + "visible.py", "(unprintable file name)"),
            ("\u200b" * 399 + "visible.py", "v"),
        ],
        ids=["400-hidden", "500-hidden", "399-hidden"],
    )
    def test_only_the_first_400_characters_of_a_name_are_read(
        self, tmp_path: Path, file: str, shown: str
    ) -> None:
        summary = _publish(tmp_path, _one_hunk(file)).call["output"]["summary"]

        [span] = _assert_only_known_lines(summary, entries=1)
        assert span == f"{shown}:3"

    def test_a_name_of_exactly_the_limit_is_shown_whole(self, tmp_path: Path) -> None:
        summary = _publish(tmp_path, _one_hunk("y" * 100)).call["output"]["summary"]

        [span] = _assert_only_known_lines(summary, entries=1)
        assert span == "y" * 100 + ":3"

    def test_no_name_can_add_a_line(self, tmp_path: Path) -> None:
        names = [file for file, _ in _HOSTILE_NAMES.values()]
        hunks = [
            {"file": file, "line": i, "status": "refused", "detail": "x"}
            for i, file in enumerate(names[:10])
        ]
        record = _record(
            hunks_total=11, hunks_reviewed=1, hunks_unreviewed=10, hunks=hunks, hunks_omitted=0
        )

        summary = _publish(tmp_path, record).call["output"]["summary"]

        _assert_only_known_lines(summary, entries=10)


def _repeated(unit: str) -> str:
    return (unit * (_HOSTILE_SIZE // len(unit) + 1))[:_HOSTILE_SIZE]


# Inputs that make a parser or a backtracking expression slow: long runs of backticks, tildes,
# spaces and newlines, nested brackets, unterminated comment openers, and zero-width or
# combining characters. Each is one name of 50,000 characters.
_SLOW_SHAPES: dict[str, str] = {
    "backticks": _repeated("`"),
    "tildes": _repeated("~"),
    "spaces": _repeated(" "),
    "newlines": _repeated("\n"),
    "carriage returns and line feeds": _repeated("\r\n"),
    "open brackets": _repeated("["),
    "nested brackets": "[" * (_HOSTILE_SIZE // 2) + "]" * (_HOSTILE_SIZE // 2),
    "image and link openers": _repeated("![]("),
    "unterminated comment openers": _repeated("<!--"),
    "one comment opener and dashes": "<!--" + "-" * _HOSTILE_SIZE,
    "zero-width characters": _repeated("\u200b"),
    "combining marks": "a" + "\u0301" * _HOSTILE_SIZE,
    "combining marks between letters": _repeated("a\u0301\u0302\u0303\u0304\u0305"),
    "backslashes": _repeated("\\"),
    "character references": _repeated("&a;"),
}


_GROWTH_ROUNDS = 3
_MAX_GROWTH = 30
_NOISE_FLOOR_MS = 2


def _many_objects(size: int, last: str = "{}") -> str:
    return "[" + "{}," * (size // 3) + last + "]"


# What the review step's stdout can look like, for the shapes that make a parser or a
# backtracking expression slow (see _SLOW_SHAPES), built to `size` characters; the flag says
# whether the script can read it as findings. The last entries are arrays with as many
# elements as fit, which the check of every element has to walk.
_FINDINGS_SHAPES: dict[str, Callable[[int], tuple[str, bool]]] = {
    "backticks": lambda n: ("`" * n, False),
    "tildes": lambda n: ("~" * n, False),
    "open brackets": lambda n: ("[" * n, False),
    "nested brackets": lambda n: ("[" * (n // 2) + "]" * (n // 2), False),
    "spaces in an empty array": lambda n: ("[" + " " * n + "]", True),
    "newlines in an empty array": lambda n: ("[" + "\n" * n + "]", True),
    "unterminated comment openers": lambda n: ("<!--" * (n // 4 + 1), False),
    "zero-width characters": lambda n: ("\u200b" * n, False),
    "combining marks": lambda n: ("a" + "\u0301" * n, False),
    "a finding with a long comment of backticks": lambda n: (
        json.dumps([{"file": "a.py", "lines": "1-1", "severity": "INFO", "comment": "`" * n}]),
        True,
    ),
    "as many empty objects as fit": lambda n: (_many_objects(n), True),
    "as many empty objects as fit, then a null": lambda n: (_many_objects(n, "null"), False),
    "as many nulls as fit": lambda n: ("[" + "null," * (n // 5) + "null]", False),
}


@_needs_node
class TestHostileInputIsLinear:
    @pytest.mark.parametrize("shape", list(_SLOW_SHAPES))
    def test_a_50000_character_name_is_handled_in_under_two_seconds(
        self, tmp_path: Path, shape: str
    ) -> None:
        file = _SLOW_SHAPES[shape]
        assert len(file) >= _HOSTILE_SIZE
        # Five names, so the record is as large as the limit allows (4 bytes per character at most).
        hunks = [{"file": file, "line": i + 1, "status": "error", "detail": "x"} for i in range(5)]
        record = _record(
            hunks_total=6, hunks_reviewed=1, hunks_unreviewed=5, hunks=hunks, hunks_omitted=0
        )

        published = _publish(tmp_path, record, ["WARN"], wide=True)

        summary = published.call["output"]["summary"]
        # Not "unknown" through being too large: the five names really were read and shown.
        _assert_only_known_lines(summary, entries=5)
        assert published.cpu_ms < _TIME_LIMIT_MS

    def test_a_record_full_of_entries_is_handled_in_under_two_seconds(self, tmp_path: Path) -> None:
        hunks = [
            {"file": f"src/dir{i % 50}/file{i}.py", "line": i, "status": "truncated", "detail": "x"}
            for i in range(10_000)
        ]
        record = _record(
            hunks_total=10_001,
            hunks_reviewed=1,
            hunks_unreviewed=10_000,
            hunks=hunks,
            hunks_omitted=0,
        )

        published = _publish(tmp_path, record)

        _assert_only_known_lines(published.call["output"]["summary"], entries=10)
        assert published.cpu_ms < _TIME_LIMIT_MS

    @pytest.mark.parametrize("exit_code", ["0", "4"])
    @pytest.mark.parametrize("shape", list(_FINDINGS_SHAPES))
    def test_a_50000_character_findings_file_is_read_in_under_two_seconds(
        self, tmp_path: Path, shape: str, exit_code: str
    ) -> None:
        text, readable = _FINDINGS_SHAPES[shape](_HOSTILE_SIZE)
        assert len(text) >= _HOSTILE_SIZE

        published = run_publish_script(tmp_path, exit_code, text, outcome=json.dumps(_VALID))

        assert published.cpu_ms < _TIME_LIMIT_MS
        # The file was really read the way the shape says: not rejected for some other reason.
        output = published.call["output"]
        if exit_code == "4":
            assert output["title"] == "trelix review incomplete"
            assert ("could not be read" in output["summary"]) is (not readable)
        else:
            assert output["title"].startswith("trelix found") is readable
            assert (output["title"] == "trelix review did not complete") is (not readable)

    @pytest.mark.parametrize("shape", [n for n in _FINDINGS_SHAPES if "empty objects" in n])
    def test_the_time_to_read_the_findings_grows_with_the_file_not_with_its_square(
        self, tmp_path: Path, shape: str
    ) -> None:
        """A quadratic check can still finish 50,000 characters inside the limit on a fast
        machine, so the time is also compared across sizes: 8 times the file must cost about 8
        times the time, not 64."""
        small, _ = _FINDINGS_SHAPES[shape](25_000)
        large, _ = _FINDINGS_SHAPES[shape](200_000)

        def best(text: str) -> float:
            return min(
                run_publish_script(tmp_path, "0", text).cpu_ms for _ in range(_GROWTH_ROUNDS)
            )

        best(small)  # warm up
        assert best(large) / max(best(small), _NOISE_FLOOR_MS) < _MAX_GROWTH

    def test_a_50000_character_findings_file_is_handled_in_under_two_seconds(
        self, tmp_path: Path
    ) -> None:
        findings = [{"file": "a.py", "lines": "1-1", "severity": "INFO", "comment": "`" * 50_000}]

        published = run_publish_script(
            tmp_path, "4", json.dumps(findings), outcome=json.dumps(_VALID)
        )

        assert published.cpu_ms < _TIME_LIMIT_MS
        # The model's 50,000 backticks are in the annotation, never in the summary.
        _assert_only_known_lines(published.call["output"]["summary"], entries=2)
