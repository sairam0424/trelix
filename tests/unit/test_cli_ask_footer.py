"""The verified `Sources:` footer of `trelix ask` (C-5 design, R3b, R4b).

Human mode prints a blank line and `footer_lines()` after the streamed answer when the context
carried `citation_sources` and the model answered; with the flag off stdout is what it was
before the footer existed, and after an abstention there is no footer. `footer_lines` and
`citation_as_json` are the two renderings of one `Citation`, both built in
`trelix.retrieval.citations` so the footer and `--json` agree. Fixtures:
tests/unit/ask_cli_harness.py; `--json` itself is in tests/unit/test_cli_ask_json.py.

Every expected value is a literal in this file or the harness. MUTATIONS, each named by the
test it breaks:
print the footer whenever `--json` is absent -> test_no_footer_without_citation_sources;
print `valid` rows with the `unverified` template -> test_footer_lines_for_the_design_result;
print the footer after an abstention -> test_no_footer_after_an_abstention;
skip the footer when `citations` is empty (`if cited and citations:`) ->
    test_footer_says_none_cited_for_an_answer_without_markers;
drop the blank line before `Sources:` -> test_footer_follows_the_streamed_answer_*;
drop `_safe_text` from the footer -> test_footer_path_with_markup_* (MarkupError, exit 1);
drop `soft_wrap=True` from the footer rows -> test_footer_row_longer_than_the_console_width_*
    (Rich hard-wraps the 105-character row at the 80 columns a pipe gets);
emit `lines` as two ints, or from `line_start` alone -> test_citation_as_json_*;
`None` lines for every non-valid row -> test_citation_as_json_has_exactly_the_six_keys (the
    `line_out_of_range` row keeps `"10-30"`).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.unit.ask_cli_harness import (
    ABSTENTION,
    ABSTENTION_TOKENS,
    ANSWER,
    ANSWER_TOKENS,
    JWT,
    MIDDLEWARE,
    ScriptedChatClient,
    invoke_ask,
    make_context,
    make_repo,
)
from trelix.retrieval.citations import Citation, CitationSource, citation_as_json, footer_lines

TWO_VALID_FOOTER = (
    "Sources:\n"
    "  [C2] src/auth/middleware.py:70-80 AuthMiddleware.bearer\n"
    "  [C1] src/auth/middleware.py:42-67 AuthMiddleware.verify\n"
)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return make_repo(tmp_path)


# ---------------------------------------------------------------------------
# R4b: the human footer
# ---------------------------------------------------------------------------


class TestHumanFooter:
    def test_footer_follows_the_streamed_answer_after_a_blank_line(self, repo: Path) -> None:
        result = invoke_ask(repo, ScriptedChatClient(ANSWER_TOKENS), make_context())

        assert result.exit_code == 0, result.stderr
        assert result.stdout == ANSWER + "\n\n" + TWO_VALID_FOOTER
        assert result.stderr == ""

    def test_no_footer_without_citation_sources(self, repo: Path) -> None:
        """Flag off: stdout is exactly what it was before the footer existed."""
        result = invoke_ask(repo, ScriptedChatClient(("The ", "answer.")), make_context(sources=()))

        assert result.exit_code == 0, result.stderr
        assert result.stdout == "The answer.\n"

    def test_no_footer_after_an_abstention(self, repo: Path) -> None:
        result = invoke_ask(repo, ScriptedChatClient(ABSTENTION_TOKENS), make_context())

        assert result.exit_code == 0, result.stderr
        assert result.stdout == ABSTENTION + "\n"

    def test_footer_says_none_cited_for_an_answer_without_markers(self, repo: Path) -> None:
        """Sources present, no marker in the answer: the footer still prints, as `none cited`."""
        result = invoke_ask(repo, ScriptedChatClient(("No markers here.",)), make_context())

        assert result.exit_code == 0, result.stderr
        assert result.stdout == "No markers here.\n\nSources: none cited.\n"

    def test_footer_path_with_markup_is_printed_intact(self, repo: Path) -> None:
        """`[/!]` is a Rich closing tag with no opener; `_safe_text` keeps it literal."""
        odd = repo / "src" / "auth" / "["
        odd.mkdir()
        (odd / "!]mw.py").write_bytes(b"a\nb\nc\n")
        sources = (CitationSource(1, 11, "src/auth/[/!]mw.py", 1, 3, "mw"),)

        result = invoke_ask(
            repo, ScriptedChatClient(("Read it [C1].",)), make_context(sources=sources)
        )

        assert result.exit_code == 0, result.stderr
        assert result.stdout == "Read it [C1].\n\nSources:\n  [C1] src/auth/[/!]mw.py:1-3 mw\n"

    def test_footer_row_longer_than_the_console_width_stays_on_one_line(self, repo: Path) -> None:
        """The design's [C3] row is 105 characters; stdout in a pipe is 80 columns wide to Rich,
        which would otherwise break the row (and a long path) mid-word onto a second line."""
        result = invoke_ask(
            repo, ScriptedChatClient(("Decoding happens in [C3].",)), make_context()
        )

        assert result.exit_code == 0, result.stderr
        assert result.stdout == (
            "Decoding happens in [C3].\n\nSources:\n"
            "  [C3] unverified (line_out_of_range): src/auth/jwt.py has 25 lines, the cited chunk "
            "ends at line 30; re-index\n"
        )

    def test_flare_human_mode_still_streams_once_and_adds_the_footer(self, repo: Path) -> None:
        result = invoke_ask(repo, ScriptedChatClient(ANSWER_TOKENS), make_context(), flare=True)

        assert result.exit_code == 0, result.stderr
        assert result.stdout.count(ANSWER) == 1
        assert result.stdout.endswith("\n" + TWO_VALID_FOOTER)


# ---------------------------------------------------------------------------
# R3b: footer_lines and citation_as_json
# ---------------------------------------------------------------------------

DESIGN_RESULT = [
    Citation("[C2]", 2, "valid", MIDDLEWARE, 70, 80, "AuthMiddleware.bearer", ""),
    Citation("[C1]", 1, "valid", MIDDLEWARE, 42, 67, "AuthMiddleware.verify", ""),
    Citation(
        "[C3]",
        3,
        "line_out_of_range",
        JWT,
        10,
        30,
        "decode_token",
        "src/auth/jwt.py has 25 lines, the cited chunk ends at line 30; re-index",
    ),
    Citation(
        "[C4]",
        4,
        "file_missing",
        "src/auth/old.py",
        1,
        20,
        "OldAuth.check",
        "src/auth/old.py is not in the repository; re-index",
    ),
    Citation("[C9]", 9, "unknown", None, None, None, None, "no retrieved chunk has this tag"),
]


def test_footer_lines_for_the_design_result() -> None:
    assert footer_lines(DESIGN_RESULT) == [
        "Sources:",
        "  [C2] src/auth/middleware.py:70-80 AuthMiddleware.bearer",
        "  [C1] src/auth/middleware.py:42-67 AuthMiddleware.verify",
        "  [C3] unverified (line_out_of_range): src/auth/jwt.py has 25 lines, the cited chunk "
        "ends at line 30; re-index",
        "  [C4] unverified (file_missing): src/auth/old.py is not in the repository; re-index",
        "  [C9] unverified (unknown): no retrieved chunk has this tag",
    ]


def test_footer_lines_for_no_citations() -> None:
    assert footer_lines([]) == ["Sources: none cited."]


def test_citation_as_json_has_exactly_the_six_keys() -> None:
    assert [citation_as_json(c) for c in DESIGN_RESULT[1:3]] == [
        {
            "marker": "[C1]",
            "status": "valid",
            "path": MIDDLEWARE,
            "lines": "42-67",
            "symbol": "AuthMiddleware.verify",
            "detail": "",
        },
        {
            "marker": "[C3]",
            "status": "line_out_of_range",
            "path": JWT,
            "lines": "10-30",
            "symbol": "decode_token",
            "detail": "src/auth/jwt.py has 25 lines, the cited chunk ends at line 30; re-index",
        },
    ]


def test_citation_as_json_of_an_unknown_marker_has_null_path_lines_and_symbol() -> None:
    assert citation_as_json(DESIGN_RESULT[4]) == {
        "marker": "[C9]",
        "status": "unknown",
        "path": None,
        "lines": None,
        "symbol": None,
        "detail": "no retrieved chunk has this tag",
    }
