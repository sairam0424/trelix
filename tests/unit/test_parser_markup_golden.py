"""What the Razor, CSHTML, csproj, CSS, HTML and Markdown extractors read from ordinary files.

The expected values were taken from the extractors BEFORE their regexes were made linear
(see test_parser_markup_redos.py), so they pin that the rewrite changed nothing for well-formed
input. The differences on invalid input are pinned in test_parser_markup_invalid_input.py.
"""

from __future__ import annotations

import pytest

from trelix.indexing.parser.extractors import markdown, razor
from trelix.indexing.parser.extractors.cshtml import CshtmlParser
from trelix.indexing.parser.extractors.csproj import CsprojParser
from trelix.indexing.parser.extractors.css import CssParser
from trelix.indexing.parser.extractors.html import HtmlParser
from trelix.indexing.parser.extractors.markdown import MarkdownParser
from trelix.indexing.parser.extractors.razor import RazorParser

FILE_ID = 1


def _razor_view(result) -> dict:
    module = result.symbols[0]
    return {
        "imports": [(e.imported_from, e.imported_names) for e in result.import_edges],
        "types": [(t.to_type_name, t.edge_kind) for t in result.type_edges],
        "signature": module.signature,
        "body": module.body,
        "decorators": module.decorators,
    }


def _css_view(result) -> dict:
    return {
        "symbols": [(s.name, s.kind.value, s.docstring) for s in result.symbols],
        "imports": [e.imported_from for e in result.import_edges],
    }


def _csproj_view(result) -> dict:
    return {
        "symbols": [(s.name, s.kind.value, s.signature, s.body) for s in result.symbols],
        "imports": [e.imported_from for e in result.import_edges],
        "parse_errors": result.parse_errors,
    }


# ---------------------------------------------------------------------------
# Razor and CSHTML directives
# ---------------------------------------------------------------------------

RAZOR_PAGE = (
    '@page "/counter"\n@namespace MyApp.Pages\n@layout MainLayout\n@rendermode InteractiveServer\n'
    "@using System.Linq\n@using MyApp.Models\n@inject ILogger<Counter> logger\n"
    "@inject Dictionary<string, int> cache\n@inject NavigationManager Nav\n@inject IJSRuntime\tJS\n"
    "@inherits Base<T>\n@implements IDisposable\n@implements IAsyncDisposable<T>\n"
    "@typeparam T where T : IBase\n"
    '@attribute [Authorize(Roles="admin")]\n<h1>Counter</h1>\n<MyComponent Id="1" />\n'
    "@code {\n    private int count;\n    public void Increment() { count++; Log(); }\n}\n"
)


def test_razor_page_directives_are_read() -> None:
    result = RazorParser().parse(RAZOR_PAGE, FILE_ID)

    assert _razor_view(result) == {
        "imports": [
            ("MainLayout", []),
            ("System.Linq", []),
            ("MyApp.Models", []),
            ("ILogger", ["logger"]),
            ("Dictionary", ["cache"]),
            ("NavigationManager", ["Nav"]),
            ("IJSRuntime", ["JS"]),
        ],
        "types": [
            ("Base", "extends"),
            ("IDisposable", "implements"),
            ("IAsyncDisposable", "implements"),
        ],
        "signature": '@component  @page "/counter"  @namespace MyApp.Pages'
        "  @typeparam T where T : IBase  @layout MainLayout",
        "body": '@page "/counter"\n@layout MainLayout\n@typeparam T where T : IBase\n'
        "@using System.Linq\n@using MyApp.Models\n@inject ILogger<Counter> logger\n"
        "@inject Dictionary<string, int> cache\n@inject NavigationManager Nav\n"
        '@inject IJSRuntime JS\n@attribute [Authorize(Roles="admin")]\n'
        "@rendermode InteractiveServer",
        "decorators": ['[Authorize(Roles="admin")]', "@rendermode InteractiveServer"],
    }


def test_razor_directives_may_be_indented_and_follow_blank_lines() -> None:
    source = (
        "\n   \n\t@using A.B\r\n  @inject   IFoo[]?  \t foo  \r\n\n\n  @inject Map<K,  V> map\r\n"
        "\t@inherits Page<Model<T>>\r\n  \n@layout Shell\n"
    )

    assert _razor_view(RazorParser().parse(source, FILE_ID)) == {
        "imports": [
            ("Shell", []),
            ("A.B", []),
            ("IFoo[]?", ["foo"]),
            ("Map", ["map"]),
        ],
        "types": [("Page", "extends")],
        "signature": "@component  @layout Shell",
        "body": "@layout Shell\n@using A.B\n@inject IFoo[]? foo\n@inject Map<K,  V> map",
        "decorators": [],
    }


def test_every_razor_directive_may_be_indented_with_spaces_or_tabs() -> None:
    source = (
        '  @page\t"/p"\n\t@namespace\tA.B\n \t@layout  Shell\n  @implements\tIDisposable\n'
        "\t@typeparam\tT\n  @rendermode\tInteractiveServer\n\t@attribute\t[Authorize]\n"
        "\t@inject\tIFoo\tfoo\n  @inject Func<int, string, bool> fn\n"
    )

    assert _razor_view(RazorParser().parse(source, FILE_ID)) == {
        "imports": [("Shell", []), ("IFoo", ["foo"]), ("Func", ["fn"])],
        "types": [("IDisposable", "implements")],
        "signature": '@component  @page "/p"  @namespace A.B  @typeparam T  @layout Shell',
        "body": '@page "/p"\n@layout Shell\n@typeparam T\n@inject IFoo foo\n'
        "@inject Func<int, string, bool> fn\n@attribute [Authorize]\n"
        "@rendermode InteractiveServer",
        "decorators": ["[Authorize]", "@rendermode InteractiveServer"],
    }


def test_razor_directives_may_be_indented_with_any_whitespace_but_a_newline() -> None:
    # Form feed, vertical tab and a no-break space are whitespace to `\s` and to the
    # `[^\S\n]` class that replaced it, so they indent a directive as before.
    source = (
        '\x0c@page "/p"\n\xa0@layout Shell\n\x0b@inject IFoo foo\n'
        " @using A.B\n\x0c @implements IDisposable\n"
    )

    assert _razor_view(RazorParser().parse(source, FILE_ID)) == {
        "imports": [("Shell", []), ("A.B", []), ("IFoo", ["foo"])],
        "types": [("IDisposable", "implements")],
        "signature": '@component  @page "/p"  @layout Shell',
        "body": '@page "/p"\n@layout Shell\n@using A.B\n@inject IFoo foo',
        "decorators": [],
    }


def test_razor_plain_type_names_have_nothing_to_strip() -> None:
    source = "@inherits Base\n@inject IFoo foo\n@implements IDisposable\n"

    assert _razor_view(RazorParser().parse(source, FILE_ID)) == {
        "imports": [("IFoo", ["foo"])],
        "types": [("Base", "extends"), ("IDisposable", "implements")],
        "signature": "@component",
        "body": "@inject IFoo foo",
        "decorators": [],
    }


@pytest.mark.parametrize(
    "junk",
    ["@inject IFoo foo;", "@inject A b!", "@inject A\tB\tC"],
    ids=["text-after-the-name", "punctuation-after-the-name", "tab-inside-the-type"],
)
def test_inject_that_is_not_type_then_name_alone_on_its_line_is_not_a_directive(junk) -> None:
    source = f"{junk}\n@inject IOk ok\n"

    assert _razor_view(RazorParser().parse(source, FILE_ID)) == {
        "imports": [("IOk", ["ok"])],
        "types": [],
        "signature": "@component",
        "body": "@inject IOk ok",
        "decorators": [],
    }


def test_cshtml_directives_are_read() -> None:
    source = (
        '@page "/orders/{id}"\n@model OrderViewModel<Item>\n@using System.Linq\n'
        "@inject IOptions<Settings> opts\n@inject  ILogger<HomeController>\tlogger\n"
        "@inherits RazorPage<OrderViewModel<Item>>\n@{ var total = Sum(); }\n"
        "@section Scripts {\n  <script>go();</script>\n}\n"
        "@functions {\n  string Fmt(int x) { return Format(x); }\n}\n"
    )

    assert _razor_view(CshtmlParser().parse(source, FILE_ID)) == {
        "imports": [
            ("System.Linq", []),
            ("OrderViewModel", []),
            ("IOptions", ["opts"]),
            ("ILogger", ["logger"]),
        ],
        "types": [("RazorPage", "extends")],
        "signature": '@razor-page  @model OrderViewModel<Item>  @page "/orders/{id}"',
        "body": '@model OrderViewModel<Item>\n@page "/orders/{id}"\n@using System.Linq\n'
        "@inject IOptions<Settings> opts\n@inject ILogger<HomeController> logger",
        "decorators": [],
    }


def test_cshtml_directives_may_be_indented_and_follow_blank_lines() -> None:
    source = "\n  \n\t@model Plain\r\n   @page\r\n  \n  @inject IFoo  foo\n"

    assert _razor_view(CshtmlParser().parse(source, FILE_ID)) == {
        "imports": [("Plain", []), ("IFoo", ["foo"])],
        "types": [],
        "signature": "@razor-page  @model Plain",
        "body": "@model Plain\n@inject IFoo foo",
        "decorators": [],
    }


@pytest.mark.parametrize(
    ("type_name", "expected"),
    [
        ("Foo", "Foo"),
        ("List<int>", "List"),
        ("A<B<C>>", "A"),
        ("A<B>C<D>", "A"),
        (">a<b", ">a<b"),
        ("A<B", "A<B"),
        ("A>B<C>", "A>B"),
        ("<>", ""),
        ("A<>B", "AB"),
        ("Dictionary<string, int>", "Dictionary"),
        ("A<B>?", "A?"),
        ("", ""),
        ("x<y>z>", "x"),
        ("a>bc", "a>bc"),
    ],
)
def test_strip_generic_args_removes_first_open_to_last_close(type_name, expected) -> None:
    assert razor._strip_generic_args(type_name) == expected


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

CSS_SHEET = (
    "/* plain */\n.a { color: red; }\n\n/**\n * Doc block\n *\n * second paragraph\n */\n"
    "#main, .b { margin: 0; }\n/*\n\n * blank line before star\n */\n.c { x: y }\n"
    "/***   stars   ***/\n.d { x: y }\n/* a   b */\n.e { x: y }\n"
    "/* x *//* y */\n.g { x: y }\n/*\n   indented\n   lines\n*/\n"
    "@keyframes spin { from { a: b } }\n"
    "@media screen and (min-width:1px), print { .h { x: y } }\n"
    "@media  (max-width: 600px)\n  and (orientation: landscape)\n{ .i { x: y } }\n"
    "@import \"one.css\";\n@import 'two.css' screen;\n@import url(three.css);\n"
    '@import url("four.css");\n@import url( five.css );\n@use "sass:math";\n'
    '@forward "src/list" hide x;\n@use url(six.css);\n'
)


def test_css_comments_media_queries_and_imports_are_read() -> None:
    assert _css_view(CssParser().parse(CSS_SHEET, FILE_ID)) == {
        "symbols": [
            (".a", "variable", "plain"),
            ("#main", "variable", "Doc block\nsecond paragraph"),
            (".b", "variable", "Doc block\nsecond paragraph"),
            (".c", "variable", "blank line before star"),
            (".d", "variable", "stars"),
            (".e", "variable", "a   b"),
            (".g", "variable", "y"),
            ("spin", "function", "indented\n   lines"),
            ("@media screen and (min-width:1px), print", "section", None),
            (".h", "variable", None),
            ("@media (max-width: 600px) and (orientation: landscape)", "section", None),
            (".i", "variable", None),
        ],
        "imports": ["one.css", "two.css", "three.css", "four.css", "five.css"],
    }


def test_css_comment_above_a_rule_is_cleaned_once_for_all_its_selectors(monkeypatch) -> None:
    calls: list[str] = []
    clean = CssParser._clean_comment

    def counting_clean(raw: str) -> str:
        calls.append(raw)
        return clean(raw)

    monkeypatch.setattr(CssParser, "_clean_comment", staticmethod(counting_clean))

    result = CssParser().parse("/* shared */\n.a, .b, .c { x: y }\n", FILE_ID)

    assert [(s.name, s.docstring) for s in result.symbols] == [
        (".a", "shared"),
        (".b", "shared"),
        (".c", "shared"),
    ]
    assert calls == ["/* shared */"]


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------


def test_html_comment_above_an_element_becomes_its_docstring() -> None:
    source = (
        "<!-- login box --><form id=login></form>\n<!--tight--><form id=a></form>\n"
        "<!--\n  multi\n  line\n--><section id=intro></section>\n"
        "<!--   spaced   -->\n<nav aria-label=Main></nav>\n<!-- --><form id=b></form>\n"
        "<!-- a -- b --><form id=d></form>\n"
    )

    result = HtmlParser().parse(source, FILE_ID)

    assert [(s.name, s.kind.value, s.docstring) for s in result.symbols] == [
        ("document", "module", None),
        ("login", "section", "login box"),
        ("a", "section", "tight"),
        ("intro", "section", "multi\n  line"),
        ("Main", "section", "spaced"),
        ("b", "section", None),
        ("d", "section", "a -- b"),
    ]


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

MARKDOWN_DOC = (
    "---\ntitle: Guide\ntags: [a, b]\n---\n# Install\n## **Bold** step\n### Use `code` here\n"
    "## A [link](http://x.y) and [two](u2) more\n# [Top](t) heading\n## [a [b](c)](d)\n"
    "### [x]() empty url\n### [](y) empty label\n## ![img](u.png) alt\n"
    "## [unclosed and (paren)\n### a ]( b\nSetext Title\n============\nSub [s](t)\n---\n"
    "#### Deep\n## Back up\n# Again\n### Skip two\n## Level two\n"
)


def test_markdown_heading_names_and_parent_chains_are_read() -> None:
    result = MarkdownParser().parse(MARKDOWN_DOC, FILE_ID)

    assert [(s.name, s.qualified_name, s.signature) for s in result.symbols] == [
        ("Guide", "Guide", "title: Guide | tags: a, b"),
        ("Install", "Install", "# Install"),
        ("Bold step", "Install > Bold step", "## Bold step"),
        ("Use code here", "Install > Bold step > Use code here", "### Use code here"),
        ("A link and two more", "Install > A link and two more", "## A link and two more"),
        ("Top heading", "Top heading", "# Top heading"),
        ("a [b](d)", "Top heading > a [b](d)", "## a [b](d)"),
        ("[x]() empty url", "Top heading > a [b](d) > [x]() empty url", "### [x]() empty url"),
        (
            "[](y) empty label",
            "Top heading > a [b](d) > [](y) empty label",
            "### [](y) empty label",
        ),
        ("!img alt", "Top heading > !img alt", "## !img alt"),
        (
            "[unclosed and (paren)",
            "Top heading > [unclosed and (paren)",
            "## [unclosed and (paren)",
        ),
        ("a ]( b", "Top heading > [unclosed and (paren) > a ]( b", "### a ]( b"),
        ("Setext Title", "Setext Title", "# Setext Title"),
        ("Sub s", "Setext Title > Sub s", "## Sub s"),
        ("Deep", "Setext Title > Sub s > Deep", "#### Deep"),
        ("Back up", "Setext Title > Back up", "## Back up"),
        ("Again", "Again", "# Again"),
        ("Skip two", "Again > Skip two", "### Skip two"),
        ("Level two", "Again > Level two", "## Level two"),
    ]


def test_markdown_levels_without_a_top_heading_chain_to_the_nearest_lower_level() -> None:
    result = MarkdownParser().parse("## one\n## two\n### three\n## four\n", FILE_ID)

    assert [(s.name, s.qualified_name) for s in result.symbols] == [
        ("one", "one"),
        ("two", "two"),
        ("three", "two > three"),
        ("four", "four"),
    ]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("[a](b)", "a"),
        ("x [a](b) y [c](d) z", "x a y c z"),
        ("[a [b](c)](d)", "a [b](d)"),
        ("[](y)", "[](y)"),
        ("[x]()", "[x]()"),
        ("[unclosed", "[unclosed"),
        ("a ]( b", "a ]( b"),
        ("[a](b(c)d)", "ad)"),
        ("[[x]](y)", "[[x]](y)"),
        ("[a] (b)", "[a] (b)"),
        ("![img](u)", "!img"),
        ("[a](b", "[a](b"),
        ("[a][b](c)", "[a]b"),
        ("[a](b) [c](", "a [c]("),
        ("no brackets", "no brackets"),
        ("[a]](b)", "[a]](b)"),
        ("[a](b)(c)", "a(c)"),
        ("[[a](b)", "[a"),
        ("x [[a](b) y", "x [a y"),
        ("[[[a](b)", "[[a"),
        ("[a [b](c)", "a [b"),
    ],
)
def test_strip_md_links_keeps_the_label_of_each_inline_link(text, expected) -> None:
    assert markdown._strip_md_links(text) == expected


# ---------------------------------------------------------------------------
# csproj
# ---------------------------------------------------------------------------


def test_csproj_namespaces_are_stripped_before_the_xml_is_read() -> None:
    source = (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<Project Sdk="Microsoft.NET.Sdk"\n    xmlns="http://schemas.microsoft.com/developer/msbuild/2003"\n'
        '\txmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">\n  <PropertyGroup>\n'
        "    <TargetFramework>net8.0</TargetFramework>\n    <OutputType>Exe</OutputType>\n"
        "    <RootNamespace>My.App</RootNamespace>\n  </PropertyGroup>\n  <ItemGroup>\n"
        '    <PackageReference Include="Newtonsoft.Json" Version="13.0.3" />\n'
        '    <ProjectReference Include="..\\Lib\\Lib.csproj" />\n  </ItemGroup>\n</Project>\n'
    )

    assert _csproj_view(CsprojParser().parse(source, FILE_ID)) == {
        "symbols": [
            (
                "<project>",
                "module",
                "<project>  net8.0  Exe",
                "Sdk: Microsoft.NET.Sdk\nTargetFramework: net8.0\nOutputType: Exe\n"
                "RootNamespace: My.App",
            ),
            (
                "Newtonsoft.Json",
                "constant",
                '<PackageReference Include="Newtonsoft.Json" Version="13.0.3" />',
                "Newtonsoft.Json 13.0.3",
            ),
        ],
        "imports": ["Newtonsoft.Json", "Lib/Lib.csproj"],
        "parse_errors": 0,
    }


@pytest.mark.parametrize("separator", ["\t", "\n"], ids=["tab", "newline"])
def test_csproj_default_xmlns_right_after_a_tab_or_newline_is_stripped(separator) -> None:
    """The strip takes the one whitespace character before `xmlns`, not only a space."""
    source = (
        f'<Project Sdk="X"{separator}xmlns="http://schemas.microsoft.com/developer/msbuild/2003">'
        '<ItemGroup><PackageReference Include="Pkg.A" Version="1.0" /></ItemGroup></Project>'
    )

    assert _csproj_view(CsprojParser().parse(source, FILE_ID)) == {
        "symbols": [
            ("<project>", "module", "<project>", "Sdk: X"),
            (
                "Pkg.A",
                "constant",
                '<PackageReference Include="Pkg.A" Version="1.0" />',
                "Pkg.A 1.0",
            ),
        ],
        "imports": ["Pkg.A"],
        "parse_errors": 0,
    }


def test_csproj_xmlns_with_spaces_around_the_equals_sign_is_stripped() -> None:
    source = (
        '<Project xmlns = "http://schemas.microsoft.com/developer/msbuild/2003">'
        '<ItemGroup><PackageReference Include="P" Version="1" /></ItemGroup></Project>'
    )

    assert _csproj_view(CsprojParser().parse(source, FILE_ID)) == {
        "symbols": [
            ("<project>", "module", "<project>", "<project>"),
            ("P", "constant", '<PackageReference Include="P" Version="1" />', "P 1"),
        ],
        "imports": ["P"],
        "parse_errors": 0,
    }


def test_csproj_prefixed_xmlns_is_stripped_so_a_use_of_its_prefix_is_a_parse_error() -> None:
    source = (
        '<Project xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:nil="true"></Project>'
    )

    assert _csproj_view(CsprojParser().parse(source, FILE_ID)) == {
        "symbols": [("<project>", "module", "<project> (parse error)", source)],
        "imports": [],
        "parse_errors": 1,
    }
