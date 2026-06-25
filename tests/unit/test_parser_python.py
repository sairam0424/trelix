"""
Unit tests for the Python parser (PythonParser).

All fixtures are inline source strings — no files on disk are read.
"""

from __future__ import annotations

import pytest

from trelix.core.models import SymbolKind
from trelix.indexing.parser.extractors.python import PythonParser


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def parser() -> PythonParser:
    return PythonParser()


def _parse(parser: PythonParser, source: str):
    return parser.parse(source, file_id=1)


# ---------------------------------------------------------------------------
# language_name
# ---------------------------------------------------------------------------

def test_language_name(parser: PythonParser) -> None:
    assert parser.language_name == "python"


# ---------------------------------------------------------------------------
# Class extraction
# ---------------------------------------------------------------------------

CLASS_SOURCE = """\
class MyService:
    \"\"\"A simple service.\"\"\"

    def do_work(self) -> None:
        pass
"""


def test_class_name(parser: PythonParser) -> None:
    result = _parse(parser, CLASS_SOURCE)
    names = [s.name for s in result.symbols]
    assert "MyService" in names


def test_class_kind(parser: PythonParser) -> None:
    result = _parse(parser, CLASS_SOURCE)
    cls = next(s for s in result.symbols if s.name == "MyService")
    assert cls.kind == SymbolKind.CLASS


def test_class_line_numbers(parser: PythonParser) -> None:
    result = _parse(parser, CLASS_SOURCE)
    cls = next(s for s in result.symbols if s.name == "MyService")
    assert cls.line_start == 1
    assert cls.line_end >= 1
    # class ends at or after the method
    assert cls.line_end >= 4


def test_class_is_public(parser: PythonParser) -> None:
    result = _parse(parser, CLASS_SOURCE)
    cls = next(s for s in result.symbols if s.name == "MyService")
    assert cls.is_public is True


def test_private_class_is_not_public(parser: PythonParser) -> None:
    source = "class _Internal:\n    pass\n"
    result = _parse(parser, source)
    cls = next(s for s in result.symbols if s.name == "_Internal")
    assert cls.is_public is False


def test_class_docstring(parser: PythonParser) -> None:
    result = _parse(parser, CLASS_SOURCE)
    cls = next(s for s in result.symbols if s.name == "MyService")
    assert cls.docstring is not None
    assert "simple service" in cls.docstring


# ---------------------------------------------------------------------------
# Enum and Protocol detection
# ---------------------------------------------------------------------------

ENUM_SOURCE = """\
from enum import Enum

class Status(Enum):
    PENDING = 1
    ACTIVE = 2
    DONE = "done"
"""


def test_enum_kind(parser: PythonParser) -> None:
    result = _parse(parser, ENUM_SOURCE)
    cls = next(s for s in result.symbols if s.name == "Status")
    assert cls.kind == SymbolKind.ENUM


def test_enum_members_extracted_as_constants(parser: PythonParser) -> None:
    result = _parse(parser, ENUM_SOURCE)
    member_names = {s.name for s in result.symbols if s.kind == SymbolKind.CONSTANT}
    assert {"PENDING", "ACTIVE", "DONE"}.issubset(member_names)


PROTOCOL_SOURCE = """\
from typing import Protocol

class Drawable(Protocol):
    def draw(self) -> None: ...
"""


def test_protocol_kind(parser: PythonParser) -> None:
    result = _parse(parser, PROTOCOL_SOURCE)
    cls = next(s for s in result.symbols if s.name == "Drawable")
    assert cls.kind == SymbolKind.INTERFACE


# ---------------------------------------------------------------------------
# Function extraction
# ---------------------------------------------------------------------------

FUNCTION_SOURCE = """\
def add(a: int, b: int) -> int:
    \"\"\"Return a + b.\"\"\"
    return a + b


def _helper(x):
    return x * 2
"""


def test_function_name(parser: PythonParser) -> None:
    result = _parse(parser, FUNCTION_SOURCE)
    names = [s.name for s in result.symbols]
    assert "add" in names


def test_function_kind(parser: PythonParser) -> None:
    result = _parse(parser, FUNCTION_SOURCE)
    fn = next(s for s in result.symbols if s.name == "add")
    assert fn.kind == SymbolKind.FUNCTION


def test_function_signature(parser: PythonParser) -> None:
    result = _parse(parser, FUNCTION_SOURCE)
    fn = next(s for s in result.symbols if s.name == "add")
    assert "def add" in fn.signature
    assert "a: int" in fn.signature
    assert "b: int" in fn.signature
    assert "-> int" in fn.signature


def test_function_line_numbers(parser: PythonParser) -> None:
    result = _parse(parser, FUNCTION_SOURCE)
    fn = next(s for s in result.symbols if s.name == "add")
    assert fn.line_start == 1
    assert fn.line_end >= 3


def test_private_function_is_not_public(parser: PythonParser) -> None:
    result = _parse(parser, FUNCTION_SOURCE)
    fn = next(s for s in result.symbols if s.name == "_helper")
    assert fn.is_public is False


def test_function_docstring(parser: PythonParser) -> None:
    result = _parse(parser, FUNCTION_SOURCE)
    fn = next(s for s in result.symbols if s.name == "add")
    assert fn.docstring == "Return a + b."


# ---------------------------------------------------------------------------
# Method extraction with parent_id linkage
# ---------------------------------------------------------------------------

METHOD_SOURCE = """\
class Calculator:
    def add(self, a: int, b: int) -> int:
        return a + b

    def subtract(self, a: int, b: int) -> int:
        return a - b

    def _internal(self):
        pass
"""


def test_method_kind(parser: PythonParser) -> None:
    result = _parse(parser, METHOD_SOURCE)
    methods = [s for s in result.symbols if s.kind == SymbolKind.METHOD]
    method_names = {m.name for m in methods}
    assert {"add", "subtract", "_internal"}.issubset(method_names)


def test_method_parent_id_links_to_class(parser: PythonParser) -> None:
    result = _parse(parser, METHOD_SOURCE)
    cls = next(s for s in result.symbols if s.name == "Calculator")
    cls_idx = result.symbols.index(cls)

    for method in result.symbols:
        if method.kind == SymbolKind.METHOD and method.name in ("add", "subtract"):
            assert method.parent_id == cls_idx, (
                f"Method {method.name}.parent_id={method.parent_id} "
                f"expected {cls_idx}"
            )


def test_method_qualified_name(parser: PythonParser) -> None:
    result = _parse(parser, METHOD_SOURCE)
    add_method = next(
        s for s in result.symbols
        if s.kind == SymbolKind.METHOD and s.name == "add"
    )
    assert add_method.qualified_name == "Calculator.add"


def test_dunder_method_is_public(parser: PythonParser) -> None:
    source = "class Foo:\n    def __init__(self): pass\n"
    result = _parse(parser, source)
    init = next(s for s in result.symbols if s.name == "__init__")
    assert init.is_public is True


# ---------------------------------------------------------------------------
# Import edge extraction
# ---------------------------------------------------------------------------

IMPORT_SOURCE = """\
import os
import sys
from pathlib import Path
from typing import Optional, List
from . import utils
from ..core import models
import collections.abc
"""


def test_simple_import_edge(parser: PythonParser) -> None:
    result = _parse(parser, IMPORT_SOURCE)
    modules = {e.imported_from for e in result.import_edges}
    assert "os" in modules
    assert "sys" in modules


def test_from_import_edge(parser: PythonParser) -> None:
    result = _parse(parser, IMPORT_SOURCE)
    edge = next(e for e in result.import_edges if e.imported_from == "pathlib")
    assert "Path" in edge.imported_names


def test_from_import_multiple_names(parser: PythonParser) -> None:
    result = _parse(parser, IMPORT_SOURCE)
    edge = next(e for e in result.import_edges if e.imported_from == "typing")
    assert "Optional" in edge.imported_names
    assert "List" in edge.imported_names


def test_relative_import_edge(parser: PythonParser) -> None:
    result = _parse(parser, IMPORT_SOURCE)
    modules = {e.imported_from for e in result.import_edges}
    # ". import utils" or "..core import models" should be captured
    assert any("." in m for m in modules)


def test_import_edges_have_correct_file_id(parser: PythonParser) -> None:
    result = _parse(parser, IMPORT_SOURCE)
    for edge in result.import_edges:
        assert edge.file_id == 1


# ---------------------------------------------------------------------------
# Call edge extraction
# ---------------------------------------------------------------------------

CALL_SOURCE = """\
import os

def process(path: str) -> str:
    data = open(path)
    result = str(data)
    os.path.join("a", "b")
    return result

def another():
    process("hello")
    print("done")
"""


def test_call_edges_extracted(parser: PythonParser) -> None:
    result = _parse(parser, CALL_SOURCE)
    callee_names = {e.callee_name for e in result.call_edges}
    # direct function calls
    assert "open" in callee_names or "str" in callee_names or "process" in callee_names


def test_call_edge_caller_id_set(parser: PythonParser) -> None:
    result = _parse(parser, CALL_SOURCE)
    for edge in result.call_edges:
        # caller_id must be an int (local symbol index), not None
        assert isinstance(edge.caller_id, int)


def test_call_edge_line_number(parser: PythonParser) -> None:
    result = _parse(parser, CALL_SOURCE)
    for edge in result.call_edges:
        assert edge.line >= 1


def test_method_call_extracted(parser: PythonParser) -> None:
    result = _parse(parser, CALL_SOURCE)
    callee_names = {e.callee_name for e in result.call_edges}
    # os.path.join → attribute call, attribute name is "join"
    assert "join" in callee_names


# ---------------------------------------------------------------------------
# Decorator extraction
# ---------------------------------------------------------------------------

DECORATOR_SOURCE = """\
import functools
from dataclasses import dataclass

def my_decorator(fn):
    return fn

@my_decorator
def standalone():
    pass

@dataclass
class Config:
    name: str = "default"

class Router:
    @staticmethod
    def route():
        pass

    @classmethod
    def from_env(cls):
        pass

    @property
    def path(self):
        return "/"
"""


def test_function_decorator_captured(parser: PythonParser) -> None:
    result = _parse(parser, DECORATOR_SOURCE)
    fn = next(s for s in result.symbols if s.name == "standalone")
    assert any("my_decorator" in d for d in fn.decorators)


def test_class_decorator_captured(parser: PythonParser) -> None:
    result = _parse(parser, DECORATOR_SOURCE)
    cls = next(s for s in result.symbols if s.name == "Config")
    assert any("dataclass" in d for d in cls.decorators)


def test_method_decorators_captured(parser: PythonParser) -> None:
    result = _parse(parser, DECORATOR_SOURCE)
    route_method = next(
        s for s in result.symbols
        if s.kind == SymbolKind.METHOD and s.name == "route"
    )
    assert any("staticmethod" in d for d in route_method.decorators)


# ---------------------------------------------------------------------------
# Module-level constant extraction
# ---------------------------------------------------------------------------

CONSTANT_SOURCE = """\
VERSION = "1.0.0"
__version__ = "1.0.0"
MAX_SIZE = 100
_INTERNAL_MAP = {}
__all__ = ["Foo", "Bar"]
"""


def test_all_caps_constant(parser: PythonParser) -> None:
    result = _parse(parser, CONSTANT_SOURCE)
    names = {s.name for s in result.symbols if s.kind == SymbolKind.CONSTANT}
    assert "MAX_SIZE" in names
    assert "VERSION" in names


def test_dunder_constant(parser: PythonParser) -> None:
    result = _parse(parser, CONSTANT_SOURCE)
    names = {s.name for s in result.symbols if s.kind == SymbolKind.CONSTANT}
    assert "__version__" in names
    assert "__all__" in names


def test_all_body_has_exports_prefix(parser: PythonParser) -> None:
    result = _parse(parser, CONSTANT_SOURCE)
    all_sym = next(s for s in result.symbols if s.name == "__all__")
    assert "Exports:" in all_sym.body
    assert "Foo" in all_sym.body
    assert "Bar" in all_sym.body


# ---------------------------------------------------------------------------
# Type edge (inheritance)
# ---------------------------------------------------------------------------

INHERITANCE_SOURCE = """\
class Base:
    pass

class Child(Base):
    pass

class Multi(Base, dict):
    pass
"""


def test_type_edges_extracted(parser: PythonParser) -> None:
    result = _parse(parser, INHERITANCE_SOURCE)
    assert len(result.type_edges) > 0


def test_type_edge_kind_is_extends(parser: PythonParser) -> None:
    result = _parse(parser, INHERITANCE_SOURCE)
    for edge in result.type_edges:
        assert edge.edge_kind == "extends"


def test_type_edge_base_name(parser: PythonParser) -> None:
    result = _parse(parser, INHERITANCE_SOURCE)
    base_names = {e.to_type_name for e in result.type_edges}
    assert "Base" in base_names


# ---------------------------------------------------------------------------
# Module docstring → MODULE symbol
# ---------------------------------------------------------------------------

MODULE_DOC_SOURCE = '''\
"""
This module does important things.
"""


def foo():
    pass
'''


def test_module_symbol_extracted_when_docstring_present(parser: PythonParser) -> None:
    result = _parse(parser, MODULE_DOC_SOURCE)
    module_syms = [s for s in result.symbols if s.kind == SymbolKind.MODULE]
    assert len(module_syms) == 1
    assert module_syms[0].name == "<module>"


def test_no_module_symbol_without_docstring(parser: PythonParser) -> None:
    result = _parse(parser, "def foo(): pass\n")
    module_syms = [s for s in result.symbols if s.kind == SymbolKind.MODULE]
    assert len(module_syms) == 0


# ---------------------------------------------------------------------------
# parse_errors field
# ---------------------------------------------------------------------------

def test_clean_source_has_zero_errors(parser: PythonParser) -> None:
    result = _parse(parser, "def foo():\n    return 42\n")
    assert result.parse_errors == 0


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------

def test_get_parser_returns_python_parser() -> None:
    from trelix.core.models import Language
    from trelix.indexing.parser.registry import get_parser

    p = get_parser(Language.PYTHON)
    assert p is not None
    assert isinstance(p, PythonParser)


def test_get_parser_returns_none_for_unknown() -> None:
    from trelix.core.models import Language
    from trelix.indexing.parser.registry import get_parser

    p = get_parser(Language.UNKNOWN)
    assert p is None


def test_get_parser_is_cached() -> None:
    from trelix.core.models import Language
    from trelix.indexing.parser.registry import get_parser

    p1 = get_parser(Language.PYTHON)
    p2 = get_parser(Language.PYTHON)
    assert p1 is p2
