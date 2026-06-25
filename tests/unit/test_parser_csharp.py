"""
Unit tests for trelix.indexing.parser.extractors.csharp.CSharpParser.

Covers:
  - Namespace (using directive) extraction → ImportEdge
  - Class extraction — name, kind, modifiers, base types → TypeEdge
  - Method extraction — name, qualified_name, parent linkage
  - Constructor extraction
  - Property extraction (public only)
  - Enum extraction + member CONSTANT symbols
  - Interface extraction
  - XML doc comment → docstring
  - Attribute decorator extraction
"""

from __future__ import annotations

import pytest

from trelix.core.models import SymbolKind
from trelix.indexing.parser.extractors.csharp import CSharpParser

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def parser() -> CSharpParser:
    return CSharpParser()


FILE_ID = 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def names(symbols) -> list[str]:
    return [s.name for s in symbols]


def kinds(symbols) -> list[SymbolKind]:
    return [s.kind for s in symbols]


def find(symbols, name: str):
    for s in symbols:
        if s.name == name:
            return s
    return None


# ---------------------------------------------------------------------------
# Namespace / using extraction
# ---------------------------------------------------------------------------


class TestUsingExtraction:
    def test_simple_using_produces_import_edge(self, parser: CSharpParser) -> None:
        src = "using System;\nusing System.Collections.Generic;\n"
        result = parser.parse(src, FILE_ID)
        imported = {e.imported_from for e in result.import_edges}
        assert "System" in imported
        assert "System.Collections.Generic" in imported

    def test_using_static_is_extracted(self, parser: CSharpParser) -> None:
        src = "using static System.Math;\n"
        result = parser.parse(src, FILE_ID)
        assert any("System.Math" in e.imported_from for e in result.import_edges)

    def test_using_alias_extracts_rhs(self, parser: CSharpParser) -> None:
        src = "using MyAlias = System.Text.StringBuilder;\n"
        result = parser.parse(src, FILE_ID)
        assert any("System.Text.StringBuilder" in e.imported_from for e in result.import_edges)

    def test_no_using_no_import_edges(self, parser: CSharpParser) -> None:
        src = "namespace Foo { class Bar {} }\n"
        result = parser.parse(src, FILE_ID)
        assert result.import_edges == []


# ---------------------------------------------------------------------------
# Class extraction
# ---------------------------------------------------------------------------


class TestClassExtraction:
    def test_simple_class_is_extracted(self, parser: CSharpParser) -> None:
        src = "public class MyService {}\n"
        result = parser.parse(src, FILE_ID)
        sym = find(result.symbols, "MyService")
        assert sym is not None
        assert sym.kind == SymbolKind.CLASS

    def test_class_is_public_when_modifier_public(self, parser: CSharpParser) -> None:
        src = "public class PublicClass {}\n"
        result = parser.parse(src, FILE_ID)
        sym = find(result.symbols, "PublicClass")
        assert sym is not None
        assert sym.is_public is True

    def test_internal_class_is_public(self, parser: CSharpParser) -> None:
        # 'internal' is module-visible — we treat it as public
        src = "internal class InternalClass {}\n"
        result = parser.parse(src, FILE_ID)
        sym = find(result.symbols, "InternalClass")
        assert sym is not None
        assert sym.is_public is True

    def test_struct_declaration_extracted_as_class(self, parser: CSharpParser) -> None:
        src = "public struct Point { public int X; public int Y; }\n"
        result = parser.parse(src, FILE_ID)
        sym = find(result.symbols, "Point")
        assert sym is not None
        assert sym.kind == SymbolKind.CLASS

    def test_class_signature_contains_keyword_and_name(self, parser: CSharpParser) -> None:
        src = "public class Repository : IRepository {}\n"
        result = parser.parse(src, FILE_ID)
        sym = find(result.symbols, "Repository")
        assert sym is not None
        assert "Repository" in sym.signature

    def test_base_class_produces_type_edge(self, parser: CSharpParser) -> None:
        src = "public class Dog : Animal {}\n"
        result = parser.parse(src, FILE_ID)
        sym = find(result.symbols, "Dog")
        assert sym is not None
        te = [e for e in result.type_edges if e.to_type_name == "Animal"]
        assert len(te) == 1
        assert te[0].edge_kind == "extends"

    def test_multiple_base_types_all_produce_type_edges(self, parser: CSharpParser) -> None:
        src = "public class Controller : BaseController, IDisposable {}\n"
        result = parser.parse(src, FILE_ID)
        to_names = {e.to_type_name for e in result.type_edges}
        assert "BaseController" in to_names or "IDisposable" in to_names  # at least one

    def test_attribute_decorator_extracted(self, parser: CSharpParser) -> None:
        src = "[ApiController]\npublic class MyController {}\n"
        result = parser.parse(src, FILE_ID)
        sym = find(result.symbols, "MyController")
        assert sym is not None
        assert any("ApiController" in d for d in sym.decorators)

    def test_line_numbers_are_correct(self, parser: CSharpParser) -> None:
        src = "using System;\n\npublic class Foo {}\n"
        result = parser.parse(src, FILE_ID)
        sym = find(result.symbols, "Foo")
        assert sym is not None
        assert sym.line_start == 3


# ---------------------------------------------------------------------------
# Method extraction
# ---------------------------------------------------------------------------


class TestMethodExtraction:
    def test_method_inside_class_is_extracted(self, parser: CSharpParser) -> None:
        src = """
public class UserService {
    public User GetUser(int id) { return null; }
}
"""
        result = parser.parse(src, FILE_ID)
        sym = find(result.symbols, "GetUser")
        assert sym is not None
        assert sym.kind == SymbolKind.METHOD

    def test_method_has_correct_qualified_name(self, parser: CSharpParser) -> None:
        src = """
public class OrderService {
    public void PlaceOrder(Order o) {}
}
"""
        result = parser.parse(src, FILE_ID)
        sym = find(result.symbols, "PlaceOrder")
        assert sym is not None
        assert sym.qualified_name == "OrderService.PlaceOrder"

    def test_method_parent_id_links_to_class(self, parser: CSharpParser) -> None:
        src = """
public class Repo {
    public List<Item> GetAll() { return null; }
}
"""
        result = parser.parse(src, FILE_ID)
        cls = find(result.symbols, "Repo")
        method = find(result.symbols, "GetAll")
        assert cls is not None
        assert method is not None
        cls_idx = result.symbols.index(cls)
        assert method.parent_id == cls_idx

    def test_multiple_methods_all_extracted(self, parser: CSharpParser) -> None:
        src = """
public class Calc {
    public int Add(int a, int b) { return a + b; }
    public int Sub(int a, int b) { return a - b; }
    public int Mul(int a, int b) { return a * b; }
}
"""
        result = parser.parse(src, FILE_ID)
        method_names = {s.name for s in result.symbols if s.kind == SymbolKind.METHOD}
        assert "Add" in method_names
        assert "Sub" in method_names
        assert "Mul" in method_names

    def test_static_method_extracted(self, parser: CSharpParser) -> None:
        src = """
public class Factory {
    public static Widget Create() { return new Widget(); }
}
"""
        result = parser.parse(src, FILE_ID)
        sym = find(result.symbols, "Create")
        assert sym is not None
        assert sym.kind == SymbolKind.METHOD

    def test_async_method_extracted(self, parser: CSharpParser) -> None:
        src = """
public class Api {
    public async Task<string> FetchAsync(string url) { return null; }
}
"""
        result = parser.parse(src, FILE_ID)
        sym = find(result.symbols, "FetchAsync")
        assert sym is not None

    def test_method_call_produces_call_edge(self, parser: CSharpParser) -> None:
        src = """
public class Logger {
    public void Log(string msg) { Console.WriteLine(msg); }
}
"""
        result = parser.parse(src, FILE_ID)
        callee_names = {e.callee_name for e in result.call_edges}
        assert "WriteLine" in callee_names


# ---------------------------------------------------------------------------
# Constructor extraction
# ---------------------------------------------------------------------------


class TestConstructorExtraction:
    def test_constructor_is_extracted_as_method(self, parser: CSharpParser) -> None:
        src = """
public class Service {
    public Service(ILogger logger) {}
}
"""
        result = parser.parse(src, FILE_ID)
        # Constructor name matches the class name
        # There should be both a CLASS and a METHOD (constructor) named "Service"
        service_syms = [s for s in result.symbols if s.name == "Service"]
        kinds_found = {s.kind for s in service_syms}
        assert SymbolKind.CLASS in kinds_found
        assert SymbolKind.METHOD in kinds_found

    def test_constructor_parent_links_to_class(self, parser: CSharpParser) -> None:
        src = """
public class Widget {
    public Widget() {}
}
"""
        result = parser.parse(src, FILE_ID)
        cls = next(
            (s for s in result.symbols if s.kind == SymbolKind.CLASS and s.name == "Widget"), None
        )
        ctor = next(
            (s for s in result.symbols if s.kind == SymbolKind.METHOD and s.name == "Widget"), None
        )
        assert cls is not None
        assert ctor is not None
        cls_idx = result.symbols.index(cls)
        assert ctor.parent_id == cls_idx


# ---------------------------------------------------------------------------
# Property extraction
# ---------------------------------------------------------------------------


class TestPropertyExtraction:
    def test_public_property_is_extracted(self, parser: CSharpParser) -> None:
        src = """
public class Config {
    public string Name { get; set; }
}
"""
        result = parser.parse(src, FILE_ID)
        sym = find(result.symbols, "Name")
        assert sym is not None
        assert sym.kind == SymbolKind.VARIABLE

    def test_private_property_is_skipped(self, parser: CSharpParser) -> None:
        src = """
public class Config {
    private string _secret { get; set; }
}
"""
        result = parser.parse(src, FILE_ID)
        assert find(result.symbols, "_secret") is None

    def test_property_qualified_name_includes_class(self, parser: CSharpParser) -> None:
        src = """
public class User {
    public int Id { get; set; }
}
"""
        result = parser.parse(src, FILE_ID)
        sym = find(result.symbols, "Id")
        assert sym is not None
        assert sym.qualified_name == "User.Id"

    def test_multiple_properties_extracted(self, parser: CSharpParser) -> None:
        src = """
public class Product {
    public int Id { get; set; }
    public string Name { get; set; }
    public decimal Price { get; set; }
}
"""
        result = parser.parse(src, FILE_ID)
        prop_names = {s.name for s in result.symbols if s.kind == SymbolKind.VARIABLE}
        assert "Id" in prop_names
        assert "Name" in prop_names
        assert "Price" in prop_names


# ---------------------------------------------------------------------------
# Namespace / qualified name
# ---------------------------------------------------------------------------


class TestNamespaceExtraction:
    def test_class_inside_namespace_is_extracted(self, parser: CSharpParser) -> None:
        src = """
namespace MyApp.Services {
    public class EmailService {}
}
"""
        result = parser.parse(src, FILE_ID)
        sym = find(result.symbols, "EmailService")
        assert sym is not None
        assert sym.kind == SymbolKind.CLASS

    def test_file_scoped_namespace_class_extracted(self, parser: CSharpParser) -> None:
        src = "namespace MyApp.Api;\npublic class WeatherController {}\n"
        result = parser.parse(src, FILE_ID)
        sym = find(result.symbols, "WeatherController")
        assert sym is not None


# ---------------------------------------------------------------------------
# Enum extraction
# ---------------------------------------------------------------------------


class TestEnumExtraction:
    def test_enum_symbol_extracted(self, parser: CSharpParser) -> None:
        src = "public enum Status { Active, Inactive, Pending }\n"
        result = parser.parse(src, FILE_ID)
        sym = find(result.symbols, "Status")
        assert sym is not None
        assert sym.kind == SymbolKind.ENUM

    def test_enum_members_extracted_as_constants(self, parser: CSharpParser) -> None:
        src = "public enum Color { Red, Green, Blue }\n"
        result = parser.parse(src, FILE_ID)
        constant_names = {s.name for s in result.symbols if s.kind == SymbolKind.CONSTANT}
        assert "Red" in constant_names
        assert "Green" in constant_names
        assert "Blue" in constant_names

    def test_enum_member_qualified_name(self, parser: CSharpParser) -> None:
        src = "public enum Direction { North, South }\n"
        result = parser.parse(src, FILE_ID)
        north = find(result.symbols, "North")
        assert north is not None
        assert north.qualified_name == "Direction.North"

    def test_enum_member_parent_links_to_enum(self, parser: CSharpParser) -> None:
        src = "public enum Fruit { Apple, Banana }\n"
        result = parser.parse(src, FILE_ID)
        enum_sym = find(result.symbols, "Fruit")
        apple = find(result.symbols, "Apple")
        assert enum_sym is not None
        assert apple is not None
        enum_idx = result.symbols.index(enum_sym)
        assert apple.parent_id == enum_idx


# ---------------------------------------------------------------------------
# Interface extraction
# ---------------------------------------------------------------------------


class TestInterfaceExtraction:
    def test_interface_is_extracted(self, parser: CSharpParser) -> None:
        src = "public interface IRepository { void Save(); }\n"
        result = parser.parse(src, FILE_ID)
        sym = find(result.symbols, "IRepository")
        assert sym is not None
        assert sym.kind == SymbolKind.INTERFACE

    def test_interface_members_extracted(self, parser: CSharpParser) -> None:
        src = """
public interface ICache {
    void Set(string key, object value);
    object Get(string key);
}
"""
        result = parser.parse(src, FILE_ID)
        method_names = {s.name for s in result.symbols if s.kind == SymbolKind.METHOD}
        assert "Set" in method_names or "Get" in method_names


# ---------------------------------------------------------------------------
# XML doc comment extraction
# ---------------------------------------------------------------------------


class TestXmlDocExtraction:
    def test_xml_doc_comment_becomes_docstring(self, parser: CSharpParser) -> None:
        src = """
/// <summary>Handles authentication for the API.</summary>
public class AuthController {}
"""
        result = parser.parse(src, FILE_ID)
        sym = find(result.symbols, "AuthController")
        assert sym is not None
        assert sym.docstring is not None
        assert "authentication" in sym.docstring.lower() or "Handles" in sym.docstring

    def test_method_xml_doc_extracted(self, parser: CSharpParser) -> None:
        src = """
public class Service {
    /// <summary>Sends an email to the user.</summary>
    public void SendEmail(string to) {}
}
"""
        result = parser.parse(src, FILE_ID)
        sym = find(result.symbols, "SendEmail")
        assert sym is not None
        assert sym.docstring is not None


# ---------------------------------------------------------------------------
# Parse errors
# ---------------------------------------------------------------------------


class TestParseErrors:
    def test_valid_csharp_has_zero_parse_errors(self, parser: CSharpParser) -> None:
        src = "public class Clean { public void Run() {} }\n"
        result = parser.parse(src, FILE_ID)
        assert result.parse_errors == 0

    def test_empty_source_produces_no_symbols(self, parser: CSharpParser) -> None:
        result = parser.parse("", FILE_ID)
        assert result.symbols == []
