"""Boundary and ceiling pins for LineWindowParser, written against measured mutants.

`line_window.py` sits at 100% line and 100% branch coverage from
`test_line_window_parser.py`, and a hand sweep of 36 single semantic edits still left 19 of
them alive there. Coverage counts execution; none of those tests asserted the two numbers
the module exists to produce — the char ceiling and the exact line span. With this file
added, 35 of the 36 die.

Each test below names in its docstring the mutation it was written to fail on, and each
was verified to fail under that mutation and pass without it. Survivors this file closes:

    _DEFAULT_WINDOW_LINES = 60      -> 61                   (line budget unpinned)
    _MAX_WINDOW_CHARS = 1_600       -> 16_000               (ceiling disabled)
    max(1, window_lines)            -> window_lines         (floor deleted; spins forever)
    max(1, window_lines)            -> max(2, window_lines) (floor off by one)
    max(80, max_window_chars)       -> min(80, ...)         (floor inverted)
    max(80, max_window_chars)       -> max_window_chars     (floor deleted)
    max(80, max_window_chars)       -> max(81, ...)         (floor off by one)
    len(body) > self._max...        -> >=                   (shrinks an exact-fit window)
    window_number += 1              -> += 2                 (names skip)
    window_number += 1              hoisted above the emit guard
    f"section_{n}"                  -> f"section{n}"
    qualified_name=name             -> ""
    signature=lines[start]          -> lines[end - 1]       (last line, not first)
    signature=...strip()[:200]      -> ...[:200]            (strip deleted)
    signature=...strip()[:200]      -> ...strip()[:1]       (truncation boundary)
    source.splitlines()             -> source.split("\\n")   (invents a trailing line)
    lines[start:end] in the shrink  -> lines[start + 1:end] (drops the window's own head)

One measured survivor is deliberately NOT tested here, because it is an equivalent
mutant rather than a gap: deleting the `if not lines: return ParseResult(...)` early
return changes nothing observable — an empty `lines` makes `while start < len(lines)`
false immediately and the function returns the identical empty `ParseResult`. No test
can distinguish the two, and the guard is dead code.
"""

from __future__ import annotations

import threading
from typing import Any

from trelix.indexing.parser.extractors.line_window import LineWindowParser

# The two module constants are deliberately re-declared as literals here rather than
# imported. A test that imports the constant it is pinning passes for any value the
# module happens to hold, which is exactly how the ceiling mutant survived.
EXPECTED_DEFAULT_WINDOW_LINES = 60
EXPECTED_MAX_WINDOW_CHARS = 1_600
EXPECTED_MIN_MAX_WINDOW_CHARS = 80
EXPECTED_SIGNATURE_LIMIT = 200


def _spans(result: Any) -> list[tuple[int, int]]:
    return [(s.line_start, s.line_end) for s in result.symbols]


def _parse_within(parser: LineWindowParser, source: str, seconds: float = 5.0) -> Any:
    """Call `parse` on a worker thread and fail if it has not returned in `seconds`.

    Needed because one mutation (deleting the `max(1, ...)` floor on `window_lines`) makes
    `end` never exceed `start`, so the loop in `parse` cannot advance and spins forever.
    Asserting on a return value alone would hang the suite instead of failing it.
    """
    box: dict[str, Any] = {}

    def target() -> None:
        # Capture rather than let it die on the worker thread. Without this, an exception
        # inside parse() leaves box["result"] unset, is_alive() is False so the assert
        # below passes, and the test dies with a bare `KeyError: 'result'` naming neither
        # the exception nor the line that raised it. Re-raised on the calling thread below
        # so pytest reports the real traceback.
        try:
            box["result"] = parser.parse(source, file_id=1)
        except BaseException as exc:  # noqa: BLE001 - re-raised verbatim below
            box["error"] = exc

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    worker.join(seconds)
    assert not worker.is_alive(), (
        f"parse() did not return within {seconds}s — the window never advances. "
        "A non-positive window_lines must be clamped up to 1, or `end` can never "
        "exceed `start` and the while loop in parse() never terminates."
    )
    if "error" in box:
        raise box["error"]
    return box["result"]


class TestTheCharCeilingIsEnforced:
    """The ceiling is the whole point of the module and nothing asserted it.

    `Chunker` truncates an over-budget chunk rather than splitting it, so a body over the
    ceiling loses its tail silently — the exact failure this parser exists to avoid.
    """

    def test_the_default_ceiling_bounds_every_body_at_1600_chars(self) -> None:
        """MUTATION: `_MAX_WINDOW_CHARS = 1_600` -> `16_000` (ceiling effectively removed).

        Also fails on `1_600` -> `800`, and on deleting the shrink loop.
        """
        source = "\n".join("x" * 100 for _ in range(60))

        # Precondition: the 60-line budget alone would emit this whole file as ONE
        # 6,059-char window. If this fixture ever stops exceeding the ceiling, the
        # assertions below become true by construction instead of by enforcement.
        assert len(source) == 6059
        assert len(source) > EXPECTED_MAX_WINDOW_CHARS

        result = LineWindowParser().parse(source, file_id=1)

        # 100-char lines: n lines cost 101n - 1 chars, so 15 lines (1,514) is the largest
        # window that fits under 1,600 and 16 (1,615) is the first that does not.
        assert _spans(result) == [(1, 15), (16, 30), (31, 45), (46, 60)]
        assert [len(s.body) for s in result.symbols] == [1514, 1514, 1514, 1514]
        assert all(len(s.body) <= EXPECTED_MAX_WINDOW_CHARS for s in result.symbols)

    def test_the_default_line_budget_is_60_lines(self) -> None:
        """MUTATION: `_DEFAULT_WINDOW_LINES = 60` -> `61` (also caught: `59`)."""
        source = "\n".join("a" for _ in range(61))

        # Precondition: 121 chars total, so the char ceiling cannot bind and the split
        # point below is decided by the line budget alone.
        assert len(source) == 121
        assert len(source) < EXPECTED_MAX_WINDOW_CHARS

        result = LineWindowParser().parse(source, file_id=1)

        assert _spans(result) == [(1, EXPECTED_DEFAULT_WINDOW_LINES), (61, 61)]

    def test_the_ceiling_has_a_floor_of_80_chars(self) -> None:
        """MUTATION: `max(80, max_window_chars)` -> `max(81, ...)`, `min(80, ...)`, or
        `max_window_chars` (floor deleted).

        Also fails on `lines[start:end]` -> `lines[start + 1:end]` inside the shrink loop,
        which drops the window's own first line while keeping its line_start.
        """
        source = "\n".join("x" for _ in range(41))

        # Precondition: exactly one char over the 80-char floor, so a floor of 81 (or no
        # floor at all, given the caller asked for 1) produces a different split.
        assert len(source) == 81
        assert len(source) == EXPECTED_MIN_MAX_WINDOW_CHARS + 1

        result = LineWindowParser(window_lines=60, max_window_chars=1).parse(source, file_id=1)

        assert _spans(result) == [(1, 40), (41, 41)]
        assert [s.body for s in result.symbols] == ["\n".join("x" for _ in range(40)), "x"]
        assert [len(s.body) for s in result.symbols] == [79, 1]

    def test_a_body_exactly_at_the_ceiling_is_not_shrunk(self) -> None:
        """MUTATION: `while len(body) > self._max_window_chars` -> `>=`.

        An exact fit must be emitted whole; shrinking it wastes a window and a chunk.
        """
        source = "\n".join("y" * 9 for _ in range(10))

        # Precondition: exactly at the ceiling, which is the only input that separates
        # `>` from `>=`. 99 is above the 80-char floor, so no clamp interferes.
        assert len(source) == 99

        result = LineWindowParser(window_lines=10, max_window_chars=99).parse(source, file_id=1)

        assert _spans(result) == [(1, 10)]
        assert len(result.symbols[0].body) == 99


class TestTheLineBudgetFloor:
    """`window_lines` is clamped up to 1; below that the loop cannot advance."""

    def test_a_one_line_window_emits_one_symbol_per_line(self) -> None:
        """MUTATION: `max(1, window_lines)` -> `max(2, window_lines)`."""
        result = LineWindowParser(window_lines=1).parse("alpha\nbeta\ngamma", file_id=1)

        assert _spans(result) == [(1, 1), (2, 2), (3, 3)]
        assert [s.body for s in result.symbols] == ["alpha", "beta", "gamma"]

    def test_a_nonpositive_window_is_clamped_and_parse_terminates(self) -> None:
        """MUTATION: `max(1, window_lines)` -> `window_lines` (floor deleted).

        Without the floor, `end = min(start + 0, len(lines))` equals `start` forever, so
        `parse` never terminates. Run under a deadline so that shows up as a failure.
        """
        result = _parse_within(LineWindowParser(window_lines=0), "alpha\nbeta\ngamma")

        assert _spans(result) == [(1, 1), (2, 2), (3, 3)]


class TestWindowNamesAreConsecutive:
    """Names are the citation handle; a gap or a format change breaks a stored reference."""

    def test_window_numbers_run_1_2_3_in_both_name_fields(self) -> None:
        """MUTATION: `window_number += 1` -> `+= 2`; `f"section_{n}"` -> `f"section{n}"`;
        `qualified_name=name` -> `""`.

        The expected names are written out rather than derived from the symbols, so
        deleting or renumbering one makes this check fail instead of check less.
        """
        source = "\n".join(f"line {i}" for i in range(1, 10))

        result = LineWindowParser(window_lines=3).parse(source, file_id=1)

        assert [s.name for s in result.symbols] == ["section_1", "section_2", "section_3"]
        assert [s.qualified_name for s in result.symbols] == [
            "section_1",
            "section_2",
            "section_3",
        ]

    def test_a_dropped_blank_window_does_not_consume_a_number(self) -> None:
        """MUTATION: `window_number += 1` hoisted above the `if body.strip():` guard.

        Numbering counts emitted windows, not attempted ones, so a file with blank runs
        must still be numbered densely from 1.
        """
        result = LineWindowParser(window_lines=1).parse("alpha\n\n   \ngamma", file_id=1)

        # Precondition: two of the four windows really are blank and dropped, otherwise
        # hoisting the increment would be invisible here.
        assert _spans(result) == [(1, 1), (4, 4)]
        assert [s.name for s in result.symbols] == ["section_1", "section_2"]


class TestTheSignatureIsTheWindowsFirstLine:
    """The signature is what a ranked result shows, so it must be the window's own head."""

    def test_the_signature_is_the_first_line_not_the_last(self) -> None:
        """MUTATION: `signature=lines[start]` -> `signature=lines[end - 1]`."""
        result = LineWindowParser(window_lines=2).parse("alpha\nbeta\ngamma\ndelta", file_id=1)

        # Precondition: every window's first and last line differ, so first-vs-last is
        # observable rather than true by construction.
        assert _spans(result) == [(1, 2), (3, 4)]
        assert [s.signature for s in result.symbols] == ["alpha", "gamma"]

    def test_the_signature_is_stripped(self) -> None:
        """MUTATION: `lines[start].strip()[:200]` -> `lines[start][:200]`."""
        result = LineWindowParser().parse("\t  echo hello  \nnext line", file_id=1)

        assert [s.signature for s in result.symbols] == ["echo hello"]

    def test_the_signature_is_truncated_at_200_chars(self) -> None:
        """MUTATION: `[:200]` -> `[:1]` (also caught: `[:199]`, `[:201]`, truncation removed)."""
        result = LineWindowParser().parse(("z" * 250) + "\nsecond line", file_id=1)

        assert len(result.symbols) == 1
        assert len(result.symbols[0].signature) == EXPECTED_SIGNATURE_LIMIT
        assert result.symbols[0].signature == "z" * 200


class TestLineNumbersMatchTheSourceExactly:
    """A citation with an invented line number sends a reader to the wrong place."""

    def test_a_trailing_newline_does_not_invent_a_line(self) -> None:
        """MUTATION: `source.splitlines()` -> `source.split("\\n")`.

        `"alpha\\nbeta\\n".split("\\n")` yields a third, empty element, which would report
        a two-line file as ending on line 3.
        """
        result = LineWindowParser().parse("alpha\nbeta\n", file_id=1)

        assert _spans(result) == [(1, 2)]
        assert [s.body for s in result.symbols] == ["alpha\nbeta"]

    def test_crlf_line_endings_do_not_invent_a_line(self) -> None:
        """MUTATION: `source.splitlines()` -> `source.split("\\n")`.

        A CRLF file split on "\\n" keeps a trailing "\\r" on every line and adds an empty
        final element.
        """
        result = LineWindowParser().parse("alpha\r\nbeta\r\n", file_id=1)

        assert _spans(result) == [(1, 2)]
        assert [s.body for s in result.symbols] == ["alpha\nbeta"]

    def test_every_body_holds_exactly_as_many_lines_as_its_span_claims(self) -> None:
        """MUTATION: any off-by-one between `line_start`/`line_end` and the body slice,
        including `lines[start:end]` -> `lines[start + 1:end]` in the shrink loop.
        """
        source = "\n".join("q" * 30 for _ in range(9))
        result = LineWindowParser(window_lines=4, max_window_chars=95).parse(source, file_id=1)

        # Precondition: the shrink loop must actually have fired, or this only re-checks
        # the unshrunk path the existing tests already cover.
        assert _spans(result) == [(1, 3), (4, 6), (7, 9)]

        assert [s.body.count("\n") + 1 for s in result.symbols] == [3, 3, 3]
        assert [s.line_end - s.line_start + 1 for s in result.symbols] == [3, 3, 3]
