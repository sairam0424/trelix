"""Unit tests for the comment-text helpers in extractors/_comment_text.py.

The helpers replace three regular expressions that backtracked quadratically on a long run of
whitespace, blank lines or ``<``. Every golden below is what the old expression returned, so
the helpers stay exact; the hostile cases pin the linear running time.
"""

from __future__ import annotations

import time

import pytest

from trelix.indexing.parser.extractors._comment_text import (
    strip_block_comment_end,
    strip_star_line_prefixes,
    strip_xml_tags,
)

# The old patterns needed minutes on 200,000 characters; the helpers need milliseconds.
HOSTILE_LENGTH = 200_000
MAX_SECONDS = 2.0


def _assert_fast_and_unchanged(helper, text: str) -> None:
    started = time.perf_counter()
    result = helper(text)
    elapsed = time.perf_counter() - started
    assert result == text
    assert elapsed < MAX_SECONDS


class TestStripBlockCommentEnd:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("a */", "a"),
            ("a   */", "a"),
            ("a **/", "a"),
            ("a * */", "a *"),
            ("a\n */", "a"),
            ("a\n\n*/", "a"),
            ("a\t\xa0 */", "a"),
            ("a\x1c*/", "a"),
            ("a */ */", "a */"),
            ("*/", ""),
            ("**/", ""),
            (" */", ""),
        ],
    )
    def test_strips_the_closing_delimiter_and_the_blanks_before_it(self, raw, expected):
        assert strip_block_comment_end(raw) == expected

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("a */\n", "a\n"),
            ("a */\n */", "a */"),
            ("a */\n\n", "a */\n\n"),
        ],
    )
    def test_treats_one_final_newline_like_the_end_of_the_text(self, raw, expected):
        assert strip_block_comment_end(raw) == expected

    @pytest.mark.parametrize(
        "raw",
        ["", "a", "/", "a /", "a/", "a */ ", "a */ b", "a\u200b */x", "a*"],
    )
    def test_leaves_text_without_a_closing_delimiter_alone(self, raw):
        assert strip_block_comment_end(raw) == raw

    def test_zero_width_space_is_not_whitespace(self):
        assert strip_block_comment_end("a\u200b */") == "a\u200b"

    @pytest.mark.timeout(30)
    @pytest.mark.parametrize(
        "text",
        [
            "a" + " " * HOSTILE_LENGTH + "b",
            "a" + "\t" * HOSTILE_LENGTH + "b",
            "a" + " \n" * (HOSTILE_LENGTH // 2) + "b",
            "a" + "*" * HOSTILE_LENGTH + "b",
            "a" + " *" * (HOSTILE_LENGTH // 2) + "b",
        ],
        ids=["spaces", "tabs", "blank-lines", "stars", "spaced-stars"],
    )
    def test_long_runs_before_a_non_delimiter_take_linear_time(self, text):
        _assert_fast_and_unchanged(strip_block_comment_end, text)

    @pytest.mark.timeout(30)
    def test_long_run_before_the_delimiter_is_removed_in_linear_time(self):
        started = time.perf_counter()
        result = strip_block_comment_end("a" + " " * HOSTILE_LENGTH + "*" * 50 + "/")
        elapsed = time.perf_counter() - started
        assert result == "a"
        assert elapsed < MAX_SECONDS


class TestStripStarLinePrefixes:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (" * a\n * b", "a\nb"),
            ("* a", "a"),
            ("*a", "a"),
            ("a\n*b", "a\nb"),
            ("  *   x", "  x"),
            ("*  two", " two"),
            ("\t* x\n\t* y", "x\ny"),
            ("a\r\n * b\r\n * c", "a\r\nb\r\nc"),
            (" *", ""),
            ("*\n*\n*", ""),
            ("\n* x", "x"),
            ("  \xa0* z", "z"),
        ],
    )
    def test_drops_the_leading_star_column(self, raw, expected):
        assert strip_star_line_prefixes(raw) == expected

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("a\n\n * b", "a\nb"),
            ("   \n\n  * x", "x"),
        ],
    )
    def test_blank_lines_before_a_star_line_are_swallowed_with_it(self, raw, expected):
        assert strip_star_line_prefixes(raw) == expected

    @pytest.mark.parametrize(
        "raw",
        ["", "a", "a\n  b", "a\n   \n b", "x * y", "a\n\n\nb"],
    )
    def test_lines_without_a_leading_star_are_kept_with_their_indentation(self, raw):
        assert strip_star_line_prefixes(raw) == raw

    @pytest.mark.timeout(30)
    @pytest.mark.parametrize(
        "text",
        [
            "a" + "\n" * HOSTILE_LENGTH + "b",
            "a" + " \n" * (HOSTILE_LENGTH // 2) + "b",
            "\n" * HOSTILE_LENGTH + "b",
            "a\n" * (HOSTILE_LENGTH // 2),
            "a" + " " * HOSTILE_LENGTH + "b",
        ],
        ids=["blank-lines", "spaced-blank-lines", "leading-blank-lines", "short-lines", "spaces"],
    )
    def test_long_runs_without_a_star_take_linear_time(self, text):
        _assert_fast_and_unchanged(strip_star_line_prefixes, text)

    @pytest.mark.timeout(30)
    def test_long_run_of_blank_lines_before_a_star_is_removed_in_linear_time(self):
        started = time.perf_counter()
        result = strip_star_line_prefixes("a\n" + "\n" * HOSTILE_LENGTH + "* b")
        elapsed = time.perf_counter() - started
        assert result == "a\nb"
        assert elapsed < MAX_SECONDS


class TestStripXmlTags:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("<summary>Hi</summary>", "Hi"),
            ("a <b> c", "a  c"),
            ("<a>b<c>d", "bd"),
            ("<a\n>b", "b"),
            ("List<int> x", "List x"),
            ("<<a>", ""),
            ("<<>>", ">"),
            ("<a>>", ">"),
            ("<><a>", "<>"),
            ("a<>b<c>d", "a<>bd"),
        ],
    )
    def test_removes_each_span_from_a_less_than_to_the_next_greater_than(self, text, expected):
        assert strip_xml_tags(text) == expected

    def test_a_nested_less_than_ends_the_span_at_the_first_greater_than(self):
        assert strip_xml_tags('Multi <see cref="List<int>"/> line.') == 'Multi "/> line.'

    @pytest.mark.parametrize("text", ["", "<>", "<>>", "<a", "a > b", "x<y", "plain"])
    def test_text_without_a_complete_span_is_kept(self, text):
        assert strip_xml_tags(text) == text

    @pytest.mark.timeout(30)
    @pytest.mark.parametrize(
        "text",
        [
            "<" * HOSTILE_LENGTH,
            ">" + "<" * HOSTILE_LENGTH,
            "<>" * (HOSTILE_LENGTH // 2),
            "<a" * (HOSTILE_LENGTH // 2),
        ],
        ids=["less-than-run", "greater-than-then-run", "empty-spans", "unterminated-tags"],
    )
    def test_long_runs_without_a_closing_bracket_take_linear_time(self, text):
        _assert_fast_and_unchanged(strip_xml_tags, text)

    @pytest.mark.timeout(30)
    def test_long_run_before_a_closing_bracket_is_removed_in_linear_time(self):
        started = time.perf_counter()
        result = strip_xml_tags("x" + "<" * HOSTILE_LENGTH + ">y")
        elapsed = time.perf_counter() - started
        assert result == "xy"
        assert elapsed < MAX_SECONDS
