"""
Unit tests for trelix.indexing.parser.extractors.ruby.RubyParser.

Covers:
  - Class extraction (name, kind=CLASS, line numbers)
  - Module extraction (kind=MODULE)
  - Method extraction with parent_id linkage
  - Singleton method extraction
  - Private / public visibility handling
  - Class inheritance → TypeEdge (extends)
  - include/prepend/extend → TypeEdge (implements)
  - require/require_relative → ImportEdge
  - Qualified name format: "MyClass#my_method" for instance methods
  - Qualified name format: "MyClass.my_method" for singleton methods
  - Top-level constants extraction
  - Call edge extraction
  - Nested module::class qualified names
  - Singleton class (class << self)
"""

from __future__ import annotations

import pytest

from trelix.core.models import SymbolKind
from trelix.indexing.parser.extractors.ruby import RubyParser

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def parser() -> RubyParser:
    return RubyParser()


FILE_ID = 99


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
# Class extraction
# ---------------------------------------------------------------------------


class TestClassExtraction:
    def test_simple_class_extracted(self, parser: RubyParser) -> None:
        src = "class MyService\nend\n"
        result = parser.parse(src, FILE_ID)
        sym = find(result.symbols, "MyService")
        assert sym is not None
        assert sym.kind == SymbolKind.CLASS

    def test_class_kind_is_class(self, parser: RubyParser) -> None:
        src = "class Foo\nend\n"
        result = parser.parse(src, FILE_ID)
        sym = find(result.symbols, "Foo")
        assert sym is not None
        assert sym.kind == SymbolKind.CLASS

    def test_class_line_numbers(self, parser: RubyParser) -> None:
        src = "# comment\nclass Widget\n  def draw\n  end\nend\n"
        result = parser.parse(src, FILE_ID)
        sym = find(result.symbols, "Widget")
        assert sym is not None
        assert sym.line_start == 2
        assert sym.line_end == 5

    def test_class_is_public(self, parser: RubyParser) -> None:
        src = "class Public\nend\n"
        result = parser.parse(src, FILE_ID)
        sym = find(result.symbols, "Public")
        assert sym is not None
        assert sym.is_public is True

    def test_class_with_superclass(self, parser: RubyParser) -> None:
        src = "class Dog < Animal\nend\n"
        result = parser.parse(src, FILE_ID)
        sym = find(result.symbols, "Dog")
        assert sym is not None
        assert "Animal" in sym.signature

    def test_class_file_id(self, parser: RubyParser) -> None:
        src = "class A\nend\n"
        result = parser.parse(src, FILE_ID)
        sym = find(result.symbols, "A")
        assert sym is not None
        assert sym.file_id == FILE_ID

    def test_multiple_classes_extracted(self, parser: RubyParser) -> None:
        src = "class Foo\nend\nclass Bar\nend\n"
        result = parser.parse(src, FILE_ID)
        assert "Foo" in names(result.symbols)
        assert "Bar" in names(result.symbols)


# ---------------------------------------------------------------------------
# Module extraction
# ---------------------------------------------------------------------------


class TestModuleExtraction:
    def test_module_kind_is_module(self, parser: RubyParser) -> None:
        src = "module Greeter\nend\n"
        result = parser.parse(src, FILE_ID)
        sym = find(result.symbols, "Greeter")
        assert sym is not None
        assert sym.kind == SymbolKind.MODULE

    def test_module_line_numbers(self, parser: RubyParser) -> None:
        src = "module Utils\n  def helper\n  end\nend\n"
        result = parser.parse(src, FILE_ID)
        sym = find(result.symbols, "Utils")
        assert sym is not None
        assert sym.line_start == 1
        assert sym.line_end == 4

    def test_nested_module_class_extracted(self, parser: RubyParser) -> None:
        src = "module Payments\n  class Invoice\n  end\nend\n"
        result = parser.parse(src, FILE_ID)
        assert find(result.symbols, "Payments") is not None
        assert find(result.symbols, "Invoice") is not None

    def test_module_qualified_name(self, parser: RubyParser) -> None:
        src = "module Payments\n  class Invoice\n  end\nend\n"
        result = parser.parse(src, FILE_ID)
        inv = find(result.symbols, "Invoice")
        assert inv is not None
        assert "Payments" in inv.qualified_name


# ---------------------------------------------------------------------------
# Method extraction and parent_id linkage
# ---------------------------------------------------------------------------


class TestMethodExtraction:
    def test_method_inside_class_is_kind_method(self, parser: RubyParser) -> None:
        src = "class Foo\n  def bar\n  end\nend\n"
        result = parser.parse(src, FILE_ID)
        sym = find(result.symbols, "bar")
        assert sym is not None
        assert sym.kind == SymbolKind.METHOD

    def test_top_level_method_is_kind_function(self, parser: RubyParser) -> None:
        src = "def standalone\nend\n"
        result = parser.parse(src, FILE_ID)
        sym = find(result.symbols, "standalone")
        assert sym is not None
        assert sym.kind == SymbolKind.FUNCTION

    def test_method_parent_id_links_to_class(self, parser: RubyParser) -> None:
        src = "class MyClass\n  def my_method\n  end\nend\n"
        result = parser.parse(src, FILE_ID)
        cls = find(result.symbols, "MyClass")
        meth = find(result.symbols, "my_method")
        assert cls is not None
        assert meth is not None
        cls_idx = result.symbols.index(cls)
        assert meth.parent_id == cls_idx

    def test_top_level_method_has_no_parent(self, parser: RubyParser) -> None:
        src = "def global_func\nend\n"
        result = parser.parse(src, FILE_ID)
        sym = find(result.symbols, "global_func")
        assert sym is not None
        assert sym.parent_id is None

    def test_instance_method_qualified_name(self, parser: RubyParser) -> None:
        src = "class Order\n  def process\n  end\nend\n"
        result = parser.parse(src, FILE_ID)
        meth = find(result.symbols, "process")
        assert meth is not None
        assert meth.qualified_name == "Order#process"

    def test_singleton_method_qualified_name(self, parser: RubyParser) -> None:
        src = "class User\n  def self.find(id)\n  end\nend\n"
        result = parser.parse(src, FILE_ID)
        meth = find(result.symbols, "find")
        assert meth is not None
        assert meth.qualified_name == "User.find"

    def test_singleton_method_is_method_kind(self, parser: RubyParser) -> None:
        src = "class User\n  def self.create(attrs)\n  end\nend\n"
        result = parser.parse(src, FILE_ID)
        meth = find(result.symbols, "create")
        assert meth is not None
        assert meth.kind == SymbolKind.METHOD

    def test_method_line_numbers(self, parser: RubyParser) -> None:
        src = "class A\n  def foo\n    puts 'x'\n  end\nend\n"
        result = parser.parse(src, FILE_ID)
        meth = find(result.symbols, "foo")
        assert meth is not None
        assert meth.line_start == 2
        assert meth.line_end == 4

    def test_multiple_methods_in_class(self, parser: RubyParser) -> None:
        src = (
            "class Calculator\n"
            "  def add(a, b)\n    a + b\n  end\n"
            "  def subtract(a, b)\n    a - b\n  end\n"
            "end\n"
        )
        result = parser.parse(src, FILE_ID)
        assert find(result.symbols, "add") is not None
        assert find(result.symbols, "subtract") is not None


# ---------------------------------------------------------------------------
# Visibility (private / public)
# ---------------------------------------------------------------------------


class TestVisibility:
    def test_methods_are_public_by_default(self, parser: RubyParser) -> None:
        src = "class Foo\n  def bar\n  end\nend\n"
        result = parser.parse(src, FILE_ID)
        meth = find(result.symbols, "bar")
        assert meth is not None
        assert meth.is_public is True

    def test_private_method_is_not_public(self, parser: RubyParser) -> None:
        src = "class Foo\n  private\n  def secret\n  end\nend\n"
        result = parser.parse(src, FILE_ID)
        meth = find(result.symbols, "secret")
        assert meth is not None
        assert meth.is_public is False

    def test_public_after_private_restores_visibility(self, parser: RubyParser) -> None:
        src = "class Foo\n  private\n  def hidden\n  end\n  public\n  def visible\n  end\nend\n"
        result = parser.parse(src, FILE_ID)
        hidden = find(result.symbols, "hidden")
        visible = find(result.symbols, "visible")
        assert hidden is not None and hidden.is_public is False
        assert visible is not None and visible.is_public is True


# ---------------------------------------------------------------------------
# TypeEdge — inheritance
# ---------------------------------------------------------------------------


class TestTypeEdgeInheritance:
    def test_class_inheritance_produces_type_edge(self, parser: RubyParser) -> None:
        src = "class Cat < Animal\nend\n"
        result = parser.parse(src, FILE_ID)
        assert len(result.type_edges) >= 1
        edge = result.type_edges[0]
        assert edge.to_type_name == "Animal"
        assert edge.edge_kind == "extends"

    def test_type_edge_from_symbol_id_is_class_index(self, parser: RubyParser) -> None:
        src = "class Cat < Animal\nend\n"
        result = parser.parse(src, FILE_ID)
        cls = find(result.symbols, "Cat")
        assert cls is not None
        cls_idx = result.symbols.index(cls)
        edge = result.type_edges[0]
        assert edge.from_symbol_id == cls_idx

    def test_no_type_edge_without_superclass(self, parser: RubyParser) -> None:
        src = "class Plain\nend\n"
        result = parser.parse(src, FILE_ID)
        extends_edges = [e for e in result.type_edges if e.edge_kind == "extends"]
        assert len(extends_edges) == 0


# ---------------------------------------------------------------------------
# TypeEdge — mixins (include/prepend/extend)
# ---------------------------------------------------------------------------


class TestTypeEdgeMixins:
    def test_include_produces_implements_edge(self, parser: RubyParser) -> None:
        src = "class Foo\n  include Comparable\nend\n"
        result = parser.parse(src, FILE_ID)
        impl_edges = [e for e in result.type_edges if e.edge_kind == "implements"]
        assert any(e.to_type_name == "Comparable" for e in impl_edges)

    def test_prepend_produces_implements_edge(self, parser: RubyParser) -> None:
        src = "class Foo\n  prepend Loggable\nend\n"
        result = parser.parse(src, FILE_ID)
        impl_edges = [e for e in result.type_edges if e.edge_kind == "implements"]
        assert any(e.to_type_name == "Loggable" for e in impl_edges)

    def test_extend_produces_implements_edge(self, parser: RubyParser) -> None:
        src = "class Foo\n  extend Serializable\nend\n"
        result = parser.parse(src, FILE_ID)
        impl_edges = [e for e in result.type_edges if e.edge_kind == "implements"]
        assert any(e.to_type_name == "Serializable" for e in impl_edges)

    def test_both_extends_and_implements_edges(self, parser: RubyParser) -> None:
        src = "class Person < BaseEntity\n  include Comparable\n  extend Serializable\nend\n"
        result = parser.parse(src, FILE_ID)
        extends = [e for e in result.type_edges if e.edge_kind == "extends"]
        impls = [e for e in result.type_edges if e.edge_kind == "implements"]
        assert any(e.to_type_name == "BaseEntity" for e in extends)
        assert any(e.to_type_name == "Comparable" for e in impls)
        assert any(e.to_type_name == "Serializable" for e in impls)


# ---------------------------------------------------------------------------
# ImportEdge — require / require_relative
# ---------------------------------------------------------------------------


class TestImportEdge:
    def test_require_produces_import_edge(self, parser: RubyParser) -> None:
        src = 'require "json"\n'
        result = parser.parse(src, FILE_ID)
        assert len(result.import_edges) >= 1
        edge = result.import_edges[0]
        assert edge.imported_from == "json"
        assert edge.file_id == FILE_ID

    def test_require_relative_produces_import_edge(self, parser: RubyParser) -> None:
        src = 'require_relative "./models/user"\n'
        result = parser.parse(src, FILE_ID)
        assert len(result.import_edges) >= 1
        edge = result.import_edges[0]
        assert edge.imported_from == "./models/user"

    def test_multiple_requires(self, parser: RubyParser) -> None:
        src = 'require "json"\nrequire "net/http"\n'
        result = parser.parse(src, FILE_ID)
        paths = [e.imported_from for e in result.import_edges]
        assert "json" in paths
        assert "net/http" in paths

    def test_require_import_names_is_empty_list(self, parser: RubyParser) -> None:
        src = 'require "csv"\n'
        result = parser.parse(src, FILE_ID)
        assert result.import_edges[0].imported_names == []


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_allcaps_constant_extracted(self, parser: RubyParser) -> None:
        src = "MAX_SIZE = 100\n"
        result = parser.parse(src, FILE_ID)
        sym = find(result.symbols, "MAX_SIZE")
        assert sym is not None
        assert sym.kind == SymbolKind.CONSTANT

    def test_camelcase_constant_extracted(self, parser: RubyParser) -> None:
        src = "DefaultConfig = {timeout: 30}\n"
        result = parser.parse(src, FILE_ID)
        sym = find(result.symbols, "DefaultConfig")
        assert sym is not None
        assert sym.kind == SymbolKind.CONSTANT

    def test_class_level_constant_extracted(self, parser: RubyParser) -> None:
        src = 'class Foo\n  GREETING = "Hello"\nend\n'
        result = parser.parse(src, FILE_ID)
        sym = find(result.symbols, "GREETING")
        assert sym is not None
        assert sym.kind == SymbolKind.CONSTANT

    def test_class_level_constant_parent_id(self, parser: RubyParser) -> None:
        src = 'class Foo\n  GREETING = "Hello"\nend\n'
        result = parser.parse(src, FILE_ID)
        cls = find(result.symbols, "Foo")
        const = find(result.symbols, "GREETING")
        assert cls is not None and const is not None
        cls_idx = result.symbols.index(cls)
        assert const.parent_id == cls_idx


# ---------------------------------------------------------------------------
# Call edges
# ---------------------------------------------------------------------------


class TestCallEdges:
    def test_call_inside_method_produces_call_edge(self, parser: RubyParser) -> None:
        src = "class Foo\n  def do_work\n    helper()\n  end\nend\n"
        result = parser.parse(src, FILE_ID)
        callee_names = [e.callee_name for e in result.call_edges]
        assert "helper" in callee_names

    def test_method_call_receiver_name_extracted(self, parser: RubyParser) -> None:
        src = "class Foo\n  def run\n    client.connect\n  end\nend\n"
        result = parser.parse(src, FILE_ID)
        callee_names = [e.callee_name for e in result.call_edges]
        assert "connect" in callee_names

    def test_call_edge_caller_id_is_method_index(self, parser: RubyParser) -> None:
        src = "class Foo\n  def run\n    helper()\n  end\nend\n"
        result = parser.parse(src, FILE_ID)
        run_method = find(result.symbols, "run")
        assert run_method is not None
        run_idx = result.symbols.index(run_method)
        helper_edges = [e for e in result.call_edges if e.callee_name == "helper"]
        assert len(helper_edges) >= 1
        assert helper_edges[0].caller_id == run_idx


# ---------------------------------------------------------------------------
# Singleton class (class << self)
# ---------------------------------------------------------------------------


class TestSingletonClass:
    def test_singleton_class_extracted(self, parser: RubyParser) -> None:
        src = "class Foo\n  class << self\n    def configure\n    end\n  end\nend\n"
        result = parser.parse(src, FILE_ID)
        singleton = next(
            (s for s in result.symbols if "singleton" in s.name.lower()),
            None,
        )
        assert singleton is not None
        assert singleton.kind == SymbolKind.CLASS

    def test_singleton_class_method_extracted(self, parser: RubyParser) -> None:
        src = "class Foo\n  class << self\n    def configure\n    end\n  end\nend\n"
        result = parser.parse(src, FILE_ID)
        meth = find(result.symbols, "configure")
        assert meth is not None
        assert meth.kind == SymbolKind.METHOD


# ---------------------------------------------------------------------------
# Parse result integrity
# ---------------------------------------------------------------------------


class TestParseResultIntegrity:
    def test_parse_returns_parse_result(self, parser: RubyParser) -> None:
        from trelix.indexing.parser.base import ParseResult

        result = parser.parse("class A\nend\n", FILE_ID)
        assert isinstance(result, ParseResult)

    def test_empty_file_no_symbols(self, parser: RubyParser) -> None:
        result = parser.parse("", FILE_ID)
        assert result.symbols == []
        assert result.call_edges == []
        assert result.import_edges == []

    def test_comment_only_file_no_symbols(self, parser: RubyParser) -> None:
        result = parser.parse("# This is a comment\n# Another comment\n", FILE_ID)
        assert result.symbols == []

    def test_language_name(self, parser: RubyParser) -> None:
        assert parser.language_name == "ruby"

    def test_parse_errors_zero_for_valid_code(self, parser: RubyParser) -> None:
        src = "class Foo\n  def bar\n  end\nend\n"
        result = parser.parse(src, FILE_ID)
        assert result.parse_errors == 0

    def test_all_symbols_have_correct_file_id(self, parser: RubyParser) -> None:
        src = "module M\n  class C\n    def m\n    end\n  end\nend\n"
        result = parser.parse(src, FILE_ID)
        for sym in result.symbols:
            assert sym.file_id == FILE_ID


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_registry_returns_ruby_parser(self) -> None:
        from trelix.core.models import Language
        from trelix.indexing.parser.registry import get_parser

        # Clear lru_cache so previous runs don't interfere
        get_parser.cache_clear()
        p = get_parser(Language.RUBY)
        assert p is not None
        assert isinstance(p, RubyParser)
