"""
Unit tests for the six zero-coverage parser extractors.

Covers: CppParser, CssParser, HtmlParser, RazorParser, CshtmlParser, CsprojParser.

Each parser gets 3-5 focused tests:
  - instantiation without error
  - parse() on a minimal snippet returns a ParseResult with a list for symbols
  - parse() on a realistic snippet yields at least one symbol
  - edge cases: empty input, comment-only, nested/complex structures
"""

from __future__ import annotations

import pytest

from trelix.core.models import SymbolKind
from trelix.indexing.parser.extractors.cpp import CppParser
from trelix.indexing.parser.extractors.css import CssParser
from trelix.indexing.parser.extractors.cshtml import CshtmlParser
from trelix.indexing.parser.extractors.csproj import CsprojParser
from trelix.indexing.parser.extractors.html import HtmlParser
from trelix.indexing.parser.extractors.razor import RazorParser

FILE_ID = 1

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def symbol_names(result) -> list[str]:
    return [s.name for s in result.symbols]


def symbol_kinds(result) -> list[SymbolKind]:
    return [s.kind for s in result.symbols]


# ===========================================================================
# CppParser
# ===========================================================================


@pytest.fixture()
def cpp() -> CppParser:
    return CppParser()


class TestCppParser:
    def test_language_name(self, cpp: CppParser) -> None:
        assert cpp.language_name == "cpp"

    def test_instantiation(self, cpp: CppParser) -> None:
        assert cpp is not None

    def test_empty_source_returns_list(self, cpp: CppParser) -> None:
        result = cpp.parse("", file_id=FILE_ID)
        assert isinstance(result.symbols, list)
        assert result.symbols == []

    def test_comment_only_no_symbols(self, cpp: CppParser) -> None:
        source = "// just a comment\n/* block comment */"
        result = cpp.parse(source, file_id=FILE_ID)
        assert result.symbols == []

    def test_simple_function_extracted(self, cpp: CppParser) -> None:
        source = """\
int add(int a, int b) {
    return a + b;
}
"""
        result = cpp.parse(source, file_id=FILE_ID)
        names = symbol_names(result)
        assert "add" in names

    def test_class_and_method_extracted(self, cpp: CppParser) -> None:
        source = """\
#include <string>

class Greeter {
public:
    std::string greet(const std::string& name) {
        return "Hello, " + name;
    }
    std::string greeting;
};
"""
        result = cpp.parse(source, file_id=FILE_ID)
        names = symbol_names(result)
        assert "Greeter" in names
        kinds = symbol_kinds(result)
        assert SymbolKind.CLASS in kinds

    def test_include_produces_import_edge(self, cpp: CppParser) -> None:
        source = '#include <vector>\n#include "mylib.h"\n'
        result = cpp.parse(source, file_id=FILE_ID)
        imported = [e.imported_from for e in result.import_edges]
        assert "vector" in imported
        assert "mylib.h" in imported

    def test_enum_extracted(self, cpp: CppParser) -> None:
        source = """\
enum Color {
    RED,
    GREEN,
    BLUE
};
"""
        result = cpp.parse(source, file_id=FILE_ID)
        names = symbol_names(result)
        assert "Color" in names
        kinds = symbol_kinds(result)
        assert SymbolKind.ENUM in kinds


# ===========================================================================
# CssParser
# ===========================================================================


@pytest.fixture()
def css() -> CssParser:
    return CssParser()


class TestCssParser:
    def test_language_name(self, css: CssParser) -> None:
        assert css.language_name == "css"

    def test_empty_source_returns_list(self, css: CssParser) -> None:
        result = css.parse("", file_id=FILE_ID)
        assert isinstance(result.symbols, list)

    def test_class_selector_extracted(self, css: CssParser) -> None:
        source = """\
.btn {
    color: red;
    padding: 8px 16px;
}
"""
        result = css.parse(source, file_id=FILE_ID)
        names = symbol_names(result)
        assert ".btn" in names

    def test_id_selector_extracted(self, css: CssParser) -> None:
        source = """\
#header {
    background: #fff;
    height: 60px;
}
"""
        result = css.parse(source, file_id=FILE_ID)
        names = symbol_names(result)
        assert "#header" in names

    def test_keyframes_extracted(self, css: CssParser) -> None:
        source = """\
@keyframes fade-in {
    from { opacity: 0; }
    to   { opacity: 1; }
}
"""
        result = css.parse(source, file_id=FILE_ID)
        names = symbol_names(result)
        assert "fade-in" in names
        kinds = symbol_kinds(result)
        assert SymbolKind.FUNCTION in kinds

    def test_import_edge_extracted(self, css: CssParser) -> None:
        source = '@import "variables.css";\n.x { color: blue; }\n'
        result = css.parse(source, file_id=FILE_ID)
        imported = [e.imported_from for e in result.import_edges]
        assert "variables.css" in imported

    def test_media_query_extracted(self, css: CssParser) -> None:
        source = """\
@media (max-width: 768px) {
    .container {
        width: 100%;
    }
}
"""
        result = css.parse(source, file_id=FILE_ID)
        kinds = symbol_kinds(result)
        assert SymbolKind.SECTION in kinds


# ===========================================================================
# HtmlParser
# ===========================================================================


@pytest.fixture()
def html() -> HtmlParser:
    return HtmlParser()


class TestHtmlParser:
    def test_language_name(self, html: HtmlParser) -> None:
        assert html.language_name == "html"

    def test_empty_source_returns_list(self, html: HtmlParser) -> None:
        result = html.parse("", file_id=FILE_ID)
        assert isinstance(result.symbols, list)

    def test_minimal_html_returns_document_symbol(self, html: HtmlParser) -> None:
        source = "<html><head><title>Hello</title></head><body></body></html>"
        result = html.parse(source, file_id=FILE_ID)
        assert len(result.symbols) >= 1
        assert result.symbols[0].kind == SymbolKind.MODULE

    def test_title_captured_in_document_symbol(self, html: HtmlParser) -> None:
        source = """\
<!DOCTYPE html>
<html>
<head>
  <title>My Page</title>
</head>
<body></body>
</html>
"""
        result = html.parse(source, file_id=FILE_ID)
        doc = result.symbols[0]
        assert doc.name == "My Page"

    def test_section_with_id_extracted(self, html: HtmlParser) -> None:
        source = """\
<html><body>
  <section id="about">
    <p>About us</p>
  </section>
</body></html>
"""
        result = html.parse(source, file_id=FILE_ID)
        names = symbol_names(result)
        assert "about" in names

    def test_script_src_produces_import_edge(self, html: HtmlParser) -> None:
        source = """\
<html><head>
  <script src="/static/app.js"></script>
</head><body></body></html>
"""
        result = html.parse(source, file_id=FILE_ID)
        imported = [e.imported_from for e in result.import_edges]
        assert "/static/app.js" in imported

    def test_custom_element_extracted(self, html: HtmlParser) -> None:
        source = """\
<html><body>
  <app-root></app-root>
</body></html>
"""
        result = html.parse(source, file_id=FILE_ID)
        names = symbol_names(result)
        assert "app-root" in names


# ===========================================================================
# RazorParser
# ===========================================================================


@pytest.fixture()
def razor() -> RazorParser:
    return RazorParser()


class TestRazorParser:
    def test_language_name(self, razor: RazorParser) -> None:
        assert razor.language_name == "razor"

    def test_empty_source_returns_list(self, razor: RazorParser) -> None:
        result = razor.parse("", file_id=FILE_ID)
        assert isinstance(result.symbols, list)

    def test_minimal_razor_returns_component_symbol(self, razor: RazorParser) -> None:
        source = "<h1>Hello</h1>\n"
        result = razor.parse(source, file_id=FILE_ID)
        assert len(result.symbols) >= 1
        assert result.symbols[0].kind == SymbolKind.MODULE

    def test_page_directive_captured(self, razor: RazorParser) -> None:
        source = '@page "/counter"\n\n<h1>Counter</h1>\n'
        result = razor.parse(source, file_id=FILE_ID)
        assert len(result.symbols) >= 1
        sig = result.symbols[0].signature
        assert "/counter" in sig

    def test_using_produces_import_edge(self, razor: RazorParser) -> None:
        source = "@using MyApp.Services\n\n<p>Hello</p>\n"
        result = razor.parse(source, file_id=FILE_ID)
        imported = [e.imported_from for e in result.import_edges]
        assert "MyApp.Services" in imported

    def test_code_block_method_extracted(self, razor: RazorParser) -> None:
        source = """\
@page "/counter"

<h1>Count: @count</h1>
<button @onclick="Increment">+1</button>

@code {
    private int count = 0;

    private void Increment() {
        count++;
    }
}
"""
        result = razor.parse(source, file_id=FILE_ID)
        names = symbol_names(result)
        assert "Increment" in names

    def test_inject_directive_import_edge(self, razor: RazorParser) -> None:
        source = "@inject NavigationManager Nav\n\n<p>Page</p>\n"
        result = razor.parse(source, file_id=FILE_ID)
        imported = [e.imported_from for e in result.import_edges]
        assert "NavigationManager" in imported


# ===========================================================================
# CshtmlParser
# ===========================================================================


@pytest.fixture()
def cshtml() -> CshtmlParser:
    return CshtmlParser()


class TestCshtmlParser:
    def test_language_name(self, cshtml: CshtmlParser) -> None:
        assert cshtml.language_name == "cshtml"

    def test_empty_source_returns_list(self, cshtml: CshtmlParser) -> None:
        result = cshtml.parse("", file_id=FILE_ID)
        assert isinstance(result.symbols, list)

    def test_minimal_cshtml_returns_view_symbol(self, cshtml: CshtmlParser) -> None:
        source = "<p>Hello from view</p>\n"
        result = cshtml.parse(source, file_id=FILE_ID)
        assert len(result.symbols) >= 1
        assert result.symbols[0].kind == SymbolKind.MODULE

    def test_model_directive_import_edge(self, cshtml: CshtmlParser) -> None:
        source = "@model MyApp.Models.ProductViewModel\n\n<h1>@Model.Name</h1>\n"
        result = cshtml.parse(source, file_id=FILE_ID)
        imported = [e.imported_from for e in result.import_edges]
        assert "MyApp.Models.ProductViewModel" in imported

    def test_section_directive_extracted(self, cshtml: CshtmlParser) -> None:
        source = """\
@model MyApp.Models.HomeViewModel

<h1>Home</h1>

@section Scripts {
    <script>console.log("hello");</script>
}
"""
        result = cshtml.parse(source, file_id=FILE_ID)
        names = symbol_names(result)
        assert "Scripts" in names
        kinds = symbol_kinds(result)
        assert SymbolKind.SECTION in kinds

    def test_using_directive_import_edge(self, cshtml: CshtmlParser) -> None:
        source = "@using MyApp.Helpers\n\n<p>View</p>\n"
        result = cshtml.parse(source, file_id=FILE_ID)
        imported = [e.imported_from for e in result.import_edges]
        assert "MyApp.Helpers" in imported

    def test_functions_block_method_extracted(self, cshtml: CshtmlParser) -> None:
        source = """\
@model string

<p>@Model</p>

@functions {
    public string FormatName(string name) {
        return name.ToUpper();
    }
}
"""
        result = cshtml.parse(source, file_id=FILE_ID)
        names = symbol_names(result)
        assert "FormatName" in names


# ===========================================================================
# CsprojParser
# ===========================================================================


@pytest.fixture()
def csproj() -> CsprojParser:
    return CsprojParser()


_MINIMAL_CSPROJ = """\
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <TargetFramework>net8.0</TargetFramework>
    <OutputType>Exe</OutputType>
  </PropertyGroup>
</Project>
"""

_REALISTIC_CSPROJ = """\
<Project Sdk="Microsoft.NET.Sdk.Web">
  <PropertyGroup>
    <TargetFramework>net8.0</TargetFramework>
    <OutputType>Exe</OutputType>
    <RootNamespace>MyWebApp</RootNamespace>
    <Nullable>enable</Nullable>
    <LangVersion>latest</LangVersion>
  </PropertyGroup>
  <ItemGroup>
    <PackageReference Include="Microsoft.EntityFrameworkCore" Version="8.0.0" />
    <PackageReference Include="Serilog" Version="3.1.1" />
  </ItemGroup>
  <ItemGroup>
    <ProjectReference Include="../MyApp.Core/MyApp.Core.csproj" />
  </ItemGroup>
</Project>
"""


class TestCsprojParser:
    def test_language_name(self, csproj: CsprojParser) -> None:
        assert csproj.language_name == "csproj"

    def test_empty_source_returns_fallback_symbol(self, csproj: CsprojParser) -> None:
        # Empty string is invalid XML → parse error path → still returns a symbol list
        result = csproj.parse("", file_id=FILE_ID)
        assert isinstance(result.symbols, list)

    def test_minimal_csproj_returns_project_module(self, csproj: CsprojParser) -> None:
        result = csproj.parse(_MINIMAL_CSPROJ, file_id=FILE_ID)
        assert len(result.symbols) >= 1
        assert result.symbols[0].kind == SymbolKind.MODULE

    def test_target_framework_in_signature(self, csproj: CsprojParser) -> None:
        result = csproj.parse(_MINIMAL_CSPROJ, file_id=FILE_ID)
        sig = result.symbols[0].signature
        assert "net8.0" in sig

    def test_package_references_extracted(self, csproj: CsprojParser) -> None:
        result = csproj.parse(_REALISTIC_CSPROJ, file_id=FILE_ID)
        names = symbol_names(result)
        assert "Microsoft.EntityFrameworkCore" in names
        assert "Serilog" in names

    def test_package_references_produce_import_edges(self, csproj: CsprojParser) -> None:
        result = csproj.parse(_REALISTIC_CSPROJ, file_id=FILE_ID)
        imported = [e.imported_from for e in result.import_edges]
        assert "Microsoft.EntityFrameworkCore" in imported
        assert "Serilog" in imported

    def test_project_reference_import_edge(self, csproj: CsprojParser) -> None:
        result = csproj.parse(_REALISTIC_CSPROJ, file_id=FILE_ID)
        imported = [e.imported_from for e in result.import_edges]
        # ProjectReference path is normalised and leading ./ stripped
        assert any("MyApp.Core.csproj" in p for p in imported)

    def test_malformed_xml_returns_parse_error(self, csproj: CsprojParser) -> None:
        result = csproj.parse("<Project><Broken>", file_id=FILE_ID)
        assert result.parse_errors >= 1
        assert len(result.symbols) >= 1
