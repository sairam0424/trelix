"""Crafted markup and style files must parse in linear time.

The Razor, CSHTML, csproj, CSS, HTML and Markdown extractors read whole repository files, and a
contributor chooses the contents of a pull request. Several of their regexes used to retry from
every character of a long run of whitespace, ``<`` or ``[`` before failing: about four times the
time for each doubling of the input, a few minutes for a file at the 500,000 byte walker cap, with
the GIL held (the standard library ``re`` does not release it).

Each case goes through the parser entry point the finding used and has two checks: the parse
RESULT is still the extracted one (so a pattern that stops matching fails too), and the call
uses less than ``BOUND_SECONDS`` of CPU time. The bound is on CPU time, not wall-clock time, so
a loaded machine cannot fail it. Every input is large enough that the old patterns needed several
seconds or more of CPU; the new ones need milliseconds, so the bound is generous.
"""

from __future__ import annotations

import time

import pytest

from trelix.indexing.parser.extractors import markdown
from trelix.indexing.parser.extractors.cshtml import CshtmlParser
from trelix.indexing.parser.extractors.csproj import CsprojParser
from trelix.indexing.parser.extractors.css import CssParser
from trelix.indexing.parser.extractors.html import HtmlParser
from trelix.indexing.parser.extractors.markdown import MarkdownParser
from trelix.indexing.parser.extractors.razor import RazorParser

FILE_ID = 1
BOUND_SECONDS = 2.0

_PROJECT_SDK = '<Project Sdk="Microsoft.NET.Sdk"'
_PROJECT_ITEMS = (
    '><ItemGroup><PackageReference Include="Pkg.A" Version="1.2" /></ItemGroup></Project>'
)


def _project(result) -> dict:
    return {
        "imports": [(e.imported_from, e.imported_names) for e in result.import_edges],
        "types": [(t.to_type_name, t.edge_kind) for t in result.type_edges],
        "symbols": [(s.name, s.docstring) for s in result.symbols],
        "qualified": [s.qualified_name for s in result.symbols],
        "parse_errors": result.parse_errors,
    }


HOSTILE_CASES = [
    # --- Razor / CSHTML directive regexes -------------------------------------------------
    pytest.param(
        RazorParser,
        "@inject IFoo foo\n@inject A" + " " * 35000 + "!\n",
        {"imports": [("IFoo", ["foo"])]},
        id="razor-inject-space-run",
    ),
    pytest.param(
        RazorParser,
        "@using A.B\n@inject IFoo foo\n" + "\n" * 100_000 + "x",
        {"imports": [("A.B", []), ("IFoo", ["foo"])]},
        id="razor-blank-lines",
    ),
    pytest.param(
        CshtmlParser,
        "@using A.B\n@inject IFoo foo\n" + "\n" * 100_000 + "x",
        {"imports": [("A.B", []), ("IFoo", ["foo"])]},
        id="cshtml-blank-lines",
    ),
    pytest.param(
        RazorParser,
        "@inherits A" + "<" * 180000 + "\n",
        {"types": [("A" + "<" * 180000, "extends")]},
        id="razor-inherits-open-angles",
    ),
    pytest.param(
        RazorParser,
        "@implements IFoo<T>\n@implements A" + "<" * 167000 + "\n",
        {"types": [("IFoo", "implements"), ("A" + "<" * 167000, "implements")]},
        id="razor-implements-open-angles",
    ),
    pytest.param(
        RazorParser,
        "@inject IFoo<T> a\n@inject A" + "<" * 146000 + " svc\n",
        {"imports": [("IFoo", ["a"]), ("A" + "<" * 146000, ["svc"])]},
        id="razor-inject-open-angles",
    ),
    pytest.param(
        CshtmlParser,
        "@model Foo" + "<x" * 120000 + "\n",
        {"imports": [("Foo" + "<x" * 120000, [])]},
        id="cshtml-model-open-angles",
    ),
    pytest.param(
        CshtmlParser,
        "@inherits A" + "<" * 180000 + "\n",
        {"types": [("A" + "<" * 180000, "extends")]},
        id="cshtml-inherits-open-angles",
    ),
    pytest.param(
        CshtmlParser,
        "@inject IFoo<T> a\n@inject A" + "<" * 168000 + " svc\n",
        {"imports": [("IFoo", ["a"]), ("A" + "<" * 168000, ["svc"])]},
        id="cshtml-inject-open-angles",
    ),
    # --- csproj namespace strip -----------------------------------------------------------
    pytest.param(
        CsprojParser,
        _PROJECT_SDK + "\t" * 101000 + _PROJECT_ITEMS,
        {"imports": [("Pkg.A", ["1.2"])], "parse_errors": 0},
        id="csproj-whitespace-run",
    ),
    pytest.param(
        CsprojParser,
        "<Project" + " xmlns:a" * 12500 + ">",
        {"imports": [], "symbols": [("<project>", None)], "parse_errors": 1},
        id="csproj-xmlns-prefix-run",
    ),
    # --- CSS ------------------------------------------------------------------------------
    pytest.param(
        CssParser,
        "@media screen" + " \t" * 43000 + "and (min-width:1px) { .a { color: red } }",
        {
            "symbols": [
                ("@media screen and (min-width:1px)", None),
                (".a", None),
            ]
        },
        id="css-media-whitespace-run",
    ),
    pytest.param(
        CssParser,
        '@import "a.css";\n@use url(' + "url(!" * 15500 + ";",
        {"imports": [("a.css", [])]},
        id="css-use-url-run",
    ),
    pytest.param(
        CssParser,
        "/* a" + " " * 36000 + "b */\n.x { color: red }",
        {"symbols": [(".x", "a" + " " * 36000 + "b")]},
        id="css-comment-space-run",
    ),
    pytest.param(
        CssParser,
        "/* a" + "\n" * 45000 + "b */\n.x { color: red }",
        {"symbols": [(".x", "a" + "\n" * 45000 + "b")]},
        id="css-comment-blank-lines",
    ),
    # --- HTML -----------------------------------------------------------------------------
    pytest.param(
        HtmlParser,
        "<!-- a" + " " * 100000 + "b --><form id=x></form>",
        {"symbols": [("document", None), ("x", "a" + " " * 100000 + "b")]},
        id="html-comment-space-run",
    ),
    # --- Markdown -------------------------------------------------------------------------
    pytest.param(
        MarkdownParser,
        "# " + "[" * 59000,
        {"qualified": ["[" * 59000]},
        id="markdown-open-brackets",
    ),
    pytest.param(
        MarkdownParser,
        "# " + "[" * 59000 + "]",
        {"qualified": ["[" * 59000 + "]"]},
        id="markdown-open-brackets-one-close",
    ),
    pytest.param(
        MarkdownParser,
        "# " + "[a](" * 89000,
        {"qualified": ["[a](" * 89000]},
        id="markdown-links-without-close-paren",
    ),
]


@pytest.mark.timeout(30)
@pytest.mark.parametrize(("parser_cls", "source", "expected"), HOSTILE_CASES)
def test_crafted_file_parses_in_linear_time(parser_cls, source, expected) -> None:
    parser = parser_cls()

    start = time.process_time()
    result = parser.parse(source, FILE_ID)
    elapsed = time.process_time() - start

    projected = _project(result)
    assert {key: projected[key] for key in expected} == expected
    assert elapsed < BOUND_SECONDS


@pytest.mark.timeout(30)
def test_markdown_heading_with_no_parent_level_parses_in_linear_time() -> None:
    """Every `##` heading used to walk back through all the earlier ones to find a parent."""
    count = 20700
    source = "# Top\n" + "## s\n" * count
    parser = MarkdownParser()

    start = time.process_time()
    result = parser.parse(source, FILE_ID)
    elapsed = time.process_time() - start

    assert len(result.symbols) == count + 1
    assert result.symbols[0].qualified_name == "Top"
    assert result.symbols[1].qualified_name == "Top > s"
    assert result.symbols[-1].qualified_name == "Top > s"
    assert elapsed < BOUND_SECONDS


class _FindCounter(str):
    """A str that counts its find() calls, to pin that the link scan never rescans text."""

    finds = 0

    def find(self, *args):
        self.finds += 1
        return super().find(*args)


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("[" * 1000, id="no-close-bracket"),
        pytest.param("[" * 1000 + "]", id="one-close-bracket"),
        pytest.param("[a](" * 1000, id="no-close-paren"),
    ],
)
def test_link_scan_does_not_rescan_after_it_has_failed(text) -> None:
    """Counts instead of timing: one search per piece of text, however many `[` there are."""
    counted = _FindCounter(text)

    assert markdown._strip_md_links(counted) == text
    assert counted.finds <= 4


class _FlatNode:
    """Just enough of a tree-sitter node for CssParser._handle_media to read its text."""

    children: list = []
    start_byte = 0
    start_point = (0, 0)
    end_point = (0, 0)

    def __init__(self, size: int) -> None:
        self.end_byte = size


@pytest.mark.timeout(30)
def test_media_prelude_without_a_brace_is_read_in_linear_time() -> None:
    """A media_statement always has a `{`, but the pattern must not rely on that.

    The grammar never produces this input, so the test builds the node by hand (`_FlatNode`) and
    calls `_handle_media` directly; it is the only test that reaches the no-brace path.
    """
    text = "@media" + " " * 60000 + "screen"
    symbols: list = []

    start = time.process_time()
    CssParser()._handle_media(
        _FlatNode(len(text)), text.encode(), FILE_ID, symbols, [], set(), set()
    )
    elapsed = time.process_time() - start

    assert [s.name for s in symbols] == ["@media query"]
    assert elapsed < BOUND_SECONDS
