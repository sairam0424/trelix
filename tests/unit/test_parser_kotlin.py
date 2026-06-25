"""
Unit tests for trelix.indexing.parser.extractors.kotlin.KotlinParser.

Covers:
  - Regular class extraction
  - Data class extraction
  - Object declaration (singleton) extraction
  - fun (function/method) extraction
  - Enum class extraction + CONSTANT members
  - Interface extraction
  - Import edge extraction
  - Type edge (supertype) extraction
  - Annotation decorator extraction
  - Companion object — methods attributed to parent class
  - Extension function
  - Top-level const val and ALL_CAPS property → CONSTANT
"""

from __future__ import annotations

import pytest

from trelix.core.models import SymbolKind
from trelix.indexing.parser.extractors.kotlin import KotlinParser

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def parser() -> KotlinParser:
    return KotlinParser()


FILE_ID = 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def names(symbols) -> list[str]:
    return [s.name for s in symbols]


def find(symbols, name: str):
    for s in symbols:
        if s.name == name:
            return s
    return None


def find_all(symbols, name: str) -> list:
    return [s for s in symbols if s.name == name]


# ---------------------------------------------------------------------------
# Regular class
# ---------------------------------------------------------------------------


class TestClassExtraction:
    def test_simple_class_extracted(self, parser: KotlinParser) -> None:
        src = "class MyService\n"
        result = parser.parse(src, FILE_ID)
        sym = find(result.symbols, "MyService")
        assert sym is not None
        assert sym.kind == SymbolKind.CLASS

    def test_class_with_body_extracted(self, parser: KotlinParser) -> None:
        src = "class Repository {\n    fun findAll(): List<Item> = listOf()\n}\n"
        result = parser.parse(src, FILE_ID)
        sym = find(result.symbols, "Repository")
        assert sym is not None
        assert sym.kind == SymbolKind.CLASS

    def test_class_is_public_by_default(self, parser: KotlinParser) -> None:
        src = "class PublicClass\n"
        result = parser.parse(src, FILE_ID)
        sym = find(result.symbols, "PublicClass")
        assert sym is not None
        assert sym.is_public is True

    def test_private_class_is_not_public(self, parser: KotlinParser) -> None:
        src = "private class HiddenClass\n"
        result = parser.parse(src, FILE_ID)
        sym = find(result.symbols, "HiddenClass")
        assert sym is not None
        assert sym.is_public is False

    def test_class_line_numbers(self, parser: KotlinParser) -> None:
        src = "\n\nclass Foo\n"
        result = parser.parse(src, FILE_ID)
        sym = find(result.symbols, "Foo")
        assert sym is not None
        assert sym.line_start == 3


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


class TestDataClassExtraction:
    def test_data_class_extracted_as_class(self, parser: KotlinParser) -> None:
        src = "data class User(val id: Int, val name: String)\n"
        result = parser.parse(src, FILE_ID)
        sym = find(result.symbols, "User")
        assert sym is not None
        assert sym.kind == SymbolKind.CLASS

    def test_data_class_signature_contains_data(self, parser: KotlinParser) -> None:
        src = "data class Point(val x: Double, val y: Double)\n"
        result = parser.parse(src, FILE_ID)
        sym = find(result.symbols, "Point")
        assert sym is not None
        assert "data" in sym.signature

    def test_sealed_class_extracted(self, parser: KotlinParser) -> None:
        src = "sealed class Result\n"
        result = parser.parse(src, FILE_ID)
        sym = find(result.symbols, "Result")
        assert sym is not None
        assert sym.kind == SymbolKind.CLASS
        assert "sealed" in sym.signature

    def test_abstract_class_extracted(self, parser: KotlinParser) -> None:
        src = "abstract class BasePresenter\n"
        result = parser.parse(src, FILE_ID)
        sym = find(result.symbols, "BasePresenter")
        assert sym is not None
        assert sym.kind == SymbolKind.CLASS


# ---------------------------------------------------------------------------
# Object declaration (singleton)
# ---------------------------------------------------------------------------


class TestObjectDeclaration:
    def test_object_extracted_as_class(self, parser: KotlinParser) -> None:
        src = 'object AppConfig {\n    const val VERSION = "1.0"\n}\n'
        result = parser.parse(src, FILE_ID)
        sym = find(result.symbols, "AppConfig")
        assert sym is not None
        assert sym.kind == SymbolKind.CLASS

    def test_object_signature_contains_object_keyword(self, parser: KotlinParser) -> None:
        src = "object Singleton\n"
        result = parser.parse(src, FILE_ID)
        sym = find(result.symbols, "Singleton")
        assert sym is not None
        assert "object" in sym.signature

    def test_object_is_public_by_default(self, parser: KotlinParser) -> None:
        src = "object Registry\n"
        result = parser.parse(src, FILE_ID)
        sym = find(result.symbols, "Registry")
        assert sym is not None
        assert sym.is_public is True


# ---------------------------------------------------------------------------
# fun (function / method) extraction
# ---------------------------------------------------------------------------


class TestFunExtraction:
    def test_top_level_fun_extracted(self, parser: KotlinParser) -> None:
        src = 'fun greet(name: String): String = "Hello $name"\n'
        result = parser.parse(src, FILE_ID)
        sym = find(result.symbols, "greet")
        assert sym is not None
        assert sym.kind == SymbolKind.FUNCTION

    def test_method_inside_class_extracted(self, parser: KotlinParser) -> None:
        src = """
class Calculator {
    fun add(a: Int, b: Int): Int = a + b
}
"""
        result = parser.parse(src, FILE_ID)
        sym = find(result.symbols, "add")
        assert sym is not None
        assert sym.kind == SymbolKind.METHOD

    def test_method_qualified_name_includes_class(self, parser: KotlinParser) -> None:
        src = """
class OrderService {
    fun placeOrder(order: Order) {}
}
"""
        result = parser.parse(src, FILE_ID)
        sym = find(result.symbols, "placeOrder")
        assert sym is not None
        assert sym.qualified_name == "OrderService.placeOrder"

    def test_method_parent_id_links_to_class(self, parser: KotlinParser) -> None:
        src = """
class Repo {
    fun getAll(): List<Item> = listOf()
}
"""
        result = parser.parse(src, FILE_ID)
        cls = find(result.symbols, "Repo")
        method = find(result.symbols, "getAll")
        assert cls is not None
        assert method is not None
        cls_idx = result.symbols.index(cls)
        assert method.parent_id == cls_idx

    def test_multiple_methods_extracted(self, parser: KotlinParser) -> None:
        src = """
class MathUtils {
    fun square(n: Int): Int = n * n
    fun cube(n: Int): Int = n * n * n
    fun sqrt(n: Double): Double = kotlin.math.sqrt(n)
}
"""
        result = parser.parse(src, FILE_ID)
        method_names = {s.name for s in result.symbols if s.kind == SymbolKind.METHOD}
        assert "square" in method_names
        assert "cube" in method_names
        assert "sqrt" in method_names

    def test_extension_function_extracted(self, parser: KotlinParser) -> None:
        src = "fun String.reverse(): String = this.reversed()\n"
        result = parser.parse(src, FILE_ID)
        sym = find(result.symbols, "reverse")
        assert sym is not None
        assert sym.kind == SymbolKind.FUNCTION
        # Extension functions have qualified_name like "String.reverse"
        assert "reverse" in sym.qualified_name

    def test_fun_signature_contains_fun_keyword(self, parser: KotlinParser) -> None:
        src = "fun doWork(x: Int): Boolean = x > 0\n"
        result = parser.parse(src, FILE_ID)
        sym = find(result.symbols, "doWork")
        assert sym is not None
        assert "fun" in sym.signature


# ---------------------------------------------------------------------------
# Enum class
# ---------------------------------------------------------------------------


class TestEnumClassExtraction:
    def test_enum_class_extracted_as_enum(self, parser: KotlinParser) -> None:
        src = "enum class Direction { NORTH, SOUTH, EAST, WEST }\n"
        result = parser.parse(src, FILE_ID)
        sym = find(result.symbols, "Direction")
        assert sym is not None
        assert sym.kind == SymbolKind.ENUM

    def test_enum_entries_extracted_as_constants(self, parser: KotlinParser) -> None:
        src = "enum class Color { RED, GREEN, BLUE }\n"
        result = parser.parse(src, FILE_ID)
        constant_names = {s.name for s in result.symbols if s.kind == SymbolKind.CONSTANT}
        assert "RED" in constant_names
        assert "GREEN" in constant_names
        assert "BLUE" in constant_names

    def test_enum_entry_parent_links_to_enum(self, parser: KotlinParser) -> None:
        src = "enum class Status { ACTIVE, INACTIVE }\n"
        result = parser.parse(src, FILE_ID)
        enum_sym = find(result.symbols, "Status")
        active = find(result.symbols, "ACTIVE")
        assert enum_sym is not None
        assert active is not None
        enum_idx = result.symbols.index(enum_sym)
        assert active.parent_id == enum_idx

    def test_enum_entry_qualified_name(self, parser: KotlinParser) -> None:
        src = "enum class Fruit { APPLE, BANANA }\n"
        result = parser.parse(src, FILE_ID)
        apple = find(result.symbols, "APPLE")
        assert apple is not None
        assert apple.qualified_name == "Fruit.APPLE"


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------


class TestInterfaceExtraction:
    def test_interface_extracted(self, parser: KotlinParser) -> None:
        src = "interface Repository {\n    fun findById(id: Int): Item?\n}\n"
        result = parser.parse(src, FILE_ID)
        sym = find(result.symbols, "Repository")
        assert sym is not None
        assert sym.kind == SymbolKind.INTERFACE

    def test_interface_method_extracted(self, parser: KotlinParser) -> None:
        src = """
interface Cache {
    fun get(key: String): Any?
    fun set(key: String, value: Any)
}
"""
        result = parser.parse(src, FILE_ID)
        method_names = {s.name for s in result.symbols if s.kind == SymbolKind.METHOD}
        assert "get" in method_names or "set" in method_names


# ---------------------------------------------------------------------------
# Import extraction
# ---------------------------------------------------------------------------


class TestImportExtraction:
    def test_import_produces_import_edge(self, parser: KotlinParser) -> None:
        src = "import kotlin.math.sqrt\nfun f(): Double = sqrt(2.0)\n"
        result = parser.parse(src, FILE_ID)
        imported = {e.imported_from for e in result.import_edges}
        assert any("kotlin.math" in s for s in imported)

    def test_wildcard_import_extracted(self, parser: KotlinParser) -> None:
        src = "import java.util.*\n"
        result = parser.parse(src, FILE_ID)
        assert result.import_edges


# ---------------------------------------------------------------------------
# Type edge extraction (supertype delegation)
# ---------------------------------------------------------------------------


class TestTypeEdges:
    def test_class_extending_base_produces_type_edge(self, parser: KotlinParser) -> None:
        src = "class Dog : Animal()\n"
        result = parser.parse(src, FILE_ID)
        te = [e for e in result.type_edges if e.to_type_name == "Animal"]
        assert len(te) >= 1

    def test_class_implementing_interface_produces_type_edge(self, parser: KotlinParser) -> None:
        src = "class SqlRepo : Repository\n"
        result = parser.parse(src, FILE_ID)
        te = [e for e in result.type_edges if e.to_type_name == "Repository"]
        assert len(te) >= 1


# ---------------------------------------------------------------------------
# Annotation decorator extraction
# ---------------------------------------------------------------------------


class TestAnnotationExtraction:
    def test_annotation_on_class_extracted_as_decorator(self, parser: KotlinParser) -> None:
        src = "@Component\nclass MyBean\n"
        result = parser.parse(src, FILE_ID)
        sym = find(result.symbols, "MyBean")
        assert sym is not None
        assert any("Component" in d for d in sym.decorators)

    def test_annotation_on_function_extracted(self, parser: KotlinParser) -> None:
        src = """
class Controller {
    @GetMapping("/items")
    fun getItems(): List<Item> = listOf()
}
"""
        result = parser.parse(src, FILE_ID)
        sym = find(result.symbols, "getItems")
        assert sym is not None
        assert any("GetMapping" in d for d in sym.decorators)


# ---------------------------------------------------------------------------
# Companion object
# ---------------------------------------------------------------------------


class TestCompanionObject:
    def test_companion_object_methods_attributed_to_class(self, parser: KotlinParser) -> None:
        src = """
class Config {
    companion object {
        fun create(): Config = Config()
    }
}
"""
        result = parser.parse(src, FILE_ID)
        sym = find(result.symbols, "create")
        assert sym is not None
        cls = find(result.symbols, "Config")
        assert cls is not None
        cls_idx = result.symbols.index(cls)
        assert sym.parent_id == cls_idx


# ---------------------------------------------------------------------------
# Top-level constants
# ---------------------------------------------------------------------------


class TestTopLevelConstants:
    def test_const_val_extracted_as_constant(self, parser: KotlinParser) -> None:
        src = "const val MAX_RETRIES = 3\n"
        result = parser.parse(src, FILE_ID)
        sym = find(result.symbols, "MAX_RETRIES")
        assert sym is not None
        assert sym.kind == SymbolKind.CONSTANT

    def test_all_caps_val_extracted_as_constant(self, parser: KotlinParser) -> None:
        src = "val DEFAULT_TIMEOUT = 30\n"
        result = parser.parse(src, FILE_ID)
        sym = find(result.symbols, "DEFAULT_TIMEOUT")
        assert sym is not None
        assert sym.kind == SymbolKind.CONSTANT


# ---------------------------------------------------------------------------
# Parse quality
# ---------------------------------------------------------------------------


class TestParseQuality:
    def test_valid_kotlin_has_zero_errors(self, parser: KotlinParser) -> None:
        src = "class Clean {\n    fun run(): Unit {}\n}\n"
        result = parser.parse(src, FILE_ID)
        assert result.parse_errors == 0

    def test_empty_source_produces_no_symbols(self, parser: KotlinParser) -> None:
        result = parser.parse("", FILE_ID)
        assert result.symbols == []
