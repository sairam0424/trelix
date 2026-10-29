"""Invalid input that the linear Razor, CSS and csproj regexes read differently from the old ones.

Each case is a file that is not well formed: a directive split over two lines, an empty `@media`
query, a `url(` whose unquoted path holds a `(`, an `xmlns:` name that holds whitespace, an
`xmlns` after two spaces. Well-formed files read the same as before (test_parser_markup_golden.py
pins that); these are the differences that making the patterns linear brought, and the CHANGELOG
entry lists them.
"""

from __future__ import annotations

from trelix.indexing.parser.extractors.csproj import CsprojParser
from trelix.indexing.parser.extractors.css import CssParser
from trelix.indexing.parser.extractors.razor import RazorParser

FILE_ID = 1


def test_inject_with_type_and_name_on_different_lines_is_not_a_directive() -> None:
    """A Razor directive is one line: the old regex joined `@inject IFoo` and the next line."""
    source = "@inject IFoo\nfoo\n@inject\nIBar bar\n@inject IBaz baz\n"

    result = RazorParser().parse(source, FILE_ID)
    module = result.symbols[0]

    assert [(e.imported_from, e.imported_names) for e in result.import_edges] == [("IBaz", ["baz"])]
    assert result.type_edges == []
    assert (module.signature, module.body, module.decorators) == (
        "@component",
        "@inject IBaz baz",
        [],
    )


def test_empty_media_prelude_gives_an_empty_query() -> None:
    """The old regex read past the first `{` and named the section after the rule inside."""
    result = CssParser().parse("@media {\n  .a { x: y }\n}\n", FILE_ID)

    assert [(s.name, s.kind.value) for s in result.symbols] == [
        ("@media ", "section"),
        (".a", "variable"),
    ]


def test_url_with_an_open_paren_in_an_unquoted_path_is_not_an_import() -> None:
    result = CssParser().parse("@forward url(x(y)z);\n", FILE_ID)

    assert result.import_edges == []


def test_csproj_xmlns_prefix_with_whitespace_is_not_stripped() -> None:
    """`xmlns:a b="x"` is not an attribute; the old strip swallowed it and parsed the file."""
    source = (
        '<Project xmlns:a b="x"><PropertyGroup><TargetFramework>net8.0</TargetFramework>'
        "</PropertyGroup></Project>"
    )

    result = CsprojParser().parse(source, FILE_ID)

    assert [(s.name, s.kind.value, s.signature, s.body) for s in result.symbols] == [
        ("<project>", "module", "<project> (parse error)", source)
    ]
    assert result.import_edges == []
    assert result.parse_errors == 1


def test_csproj_xmlns_touching_the_next_attribute_is_stripped() -> None:
    """Only one whitespace character goes with the xmlns, so the next attribute stays separate."""
    source = (
        '<Project Sdk="X"  xmlns="http://schemas.microsoft.com/developer/msbuild/2003"'
        'ToolsVersion="4.0"></Project>'
    )

    result = CsprojParser().parse(source, FILE_ID)

    assert [(s.name, s.kind.value, s.signature, s.body) for s in result.symbols] == [
        ("<project>", "module", "<project>", "Sdk: X")
    ]
    assert result.import_edges == []
    assert result.parse_errors == 0
