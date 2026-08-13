"""
Regression tests: the synthetic "<module>" symbol must not shift local indices.

PythonParser records three families of LOCAL INDEX into its `symbols` list
while walking the AST:

    Symbol.parent_id        — method / field  -> enclosing class
    CallEdge.caller_id      — call site       -> enclosing function
    TypeEdge.from_symbol_id — subclass        -> its base

The parser also synthesises a file-level "<module>" symbol at symbols[0] when
the module has a docstring. That symbol used to be `symbols.insert(0, ...)`ed
*after* the walk, which shifted every recorded index by one and silently
corrupted all three families at once. The Indexer resolves those indices
against the final list, so the off-by-one propagated straight into the DB and
from there into the call graph, blast radius, PageRank and symbol hierarchy.

The invariant pinned here: **a module docstring is documentation, not
structure.** Adding one may add a symbol, but it must not change what any
index means. Every test below therefore parses the same source twice — with
and without a leading docstring — and compares the RESOLVED symbol names.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import trelix
from trelix.core.models import SymbolKind
from trelix.indexing.parser.base import ParseResult
from trelix.indexing.parser.extractors.python import PythonParser

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

MODULE_DOCSTRING = '"""Module docstring."""\n'

# Two classes and two functions, so an off-by-one lands on a real neighbouring
# symbol rather than falling off the list: the shifted parent of Service.run is
# Base (a plausible-looking wrong answer), and the shifted caller of beta's
# call to alpha is alpha itself (fabricated self-recursion).
SOURCE = """
def alpha():
    return 1


def beta():
    return alpha()


class Base:
    pass


class Service(Base):
    def run(self):
        return beta()
"""

# A single class as the first symbol: shifting by one makes its method's parent
# land exactly on the synthetic "<module>" symbol at index 0.
SINGLE_CLASS_SOURCE = """
class Service:
    def run(self):
        return 1
"""

# class-inside-function and method-calling-method, to cover indices recorded at
# a nesting depth greater than one.
NESTED_SOURCE = """
class Repo:
    def fetch(self):
        return 1

    def load(self):
        return self.fetch()


def make():
    class Inner:
        def go(self):
            return 2

    return Inner
"""


@pytest.fixture
def parser() -> PythonParser:
    return PythonParser()


def _name_of(result: ParseResult, idx: int | None) -> str:
    """Resolve a local index to a qualified name, loudly flagging bad indices."""
    if idx is None:
        return "<none>"
    if not 0 <= idx < len(result.symbols):
        return f"<out-of-range:{idx}>"
    return result.symbols[idx].qualified_name


def _hierarchy(result: ParseResult) -> dict[str, str]:
    """{child qualified_name: resolved parent qualified_name} for parented symbols."""
    return {
        s.qualified_name: _name_of(result, s.parent_id)
        for s in result.symbols
        if s.parent_id is not None
    }


def _calls(result: ParseResult) -> list[tuple[str, str]]:
    """[(resolved caller qualified_name, callee_name)] for every call edge."""
    return [(_name_of(result, e.caller_id), e.callee_name) for e in result.call_edges]


def _type_edges(result: ParseResult) -> list[tuple[str, str]]:
    """[(resolved subclass qualified_name, base type name)] for every type edge."""
    return [(_name_of(result, e.from_symbol_id), e.to_type_name) for e in result.type_edges]


def _both(parser: PythonParser, source: str) -> tuple[ParseResult, ParseResult]:
    """Parse `source` bare and with a leading module docstring."""
    bare = parser.parse(source, file_id=1)
    documented = parser.parse(MODULE_DOCSTRING + source, file_id=1)
    return bare, documented


# ---------------------------------------------------------------------------
# Guard: prove which source tree is under test
# ---------------------------------------------------------------------------


def test_guard_source_tree_under_test() -> None:
    """
    Pin down which `trelix` package these tests actually imported.

    pyproject.toml sets [tool.pytest.ini_options] pythonpath = ["src", "."],
    which is inserted at sys.path[0] and BEATS the PYTHONPATH env var. A
    fail-before proof driven by PYTHONPATH alone therefore silently imports the
    FIXED source and reports a bogus pass. The pre-fix run must override the
    setting with `-o pythonpath=<tree>` and export TRELIX_EXPECT_ROOT=<tree>;
    this test fails loudly if that override did not take effect.
    """
    actual = Path(trelix.__file__).resolve().parent
    expected_root = os.environ.get("TRELIX_EXPECT_ROOT")
    if expected_root:
        expected = Path(expected_root).resolve() / "trelix"
    else:
        expected = Path(__file__).resolve().parents[2] / "src" / "trelix"
    assert actual == expected, f"imported trelix from {actual}, expected {expected}"


# ---------------------------------------------------------------------------
# Symbol.parent_id
# ---------------------------------------------------------------------------


def test_method_parent_is_its_class_with_module_docstring(parser: PythonParser) -> None:
    """A method's parent_id must resolve to its class, not to a shifted neighbour."""
    result = parser.parse(MODULE_DOCSTRING + SOURCE, file_id=1)
    assert _hierarchy(result)["Service.run"] == "Service"


def test_method_parent_is_its_class_without_module_docstring(parser: PythonParser) -> None:
    """Over-shift guard: the docstring-free case must stay correct too."""
    result = parser.parse(SOURCE, file_id=1)
    assert _hierarchy(result)["Service.run"] == "Service"


def test_method_parent_is_never_the_module_symbol(parser: PythonParser) -> None:
    """
    No real symbol may be parented to the synthetic "<module>" symbol.

    Uses the single-class fixture so the off-by-one lands squarely on index 0:
    this is the exact reported symptom, Service.run parent -> "<module>".
    """
    result = parser.parse(MODULE_DOCSTRING + SINGLE_CLASS_SOURCE, file_id=1)
    assert _hierarchy(result) == {"Service.run": "Service"}
    assert "<module>" not in _hierarchy(result).values()


def test_hierarchy_is_identical_with_and_without_docstring(parser: PythonParser) -> None:
    bare, documented = _both(parser, SOURCE)
    assert _hierarchy(documented) == _hierarchy(bare)


def test_nested_hierarchy_is_identical_with_and_without_docstring(parser: PythonParser) -> None:
    """Covers a class declared inside a function (nesting depth > 1)."""
    bare, documented = _both(parser, NESTED_SOURCE)
    assert _hierarchy(documented) == _hierarchy(bare)
    assert _hierarchy(documented) == {
        "Repo.fetch": "Repo",
        "Repo.load": "Repo",
        "Inner.go": "Inner",
    }


# ---------------------------------------------------------------------------
# CallEdge.caller_id
# ---------------------------------------------------------------------------


def test_call_caller_is_lexically_enclosing_function_with_docstring(
    parser: PythonParser,
) -> None:
    """Each call must be attributed to the function that lexically contains it."""
    result = parser.parse(MODULE_DOCSTRING + SOURCE, file_id=1)
    assert sorted(_calls(result)) == [("Service.run", "beta"), ("beta", "alpha")]


def test_call_caller_is_lexically_enclosing_function_without_docstring(
    parser: PythonParser,
) -> None:
    """Over-shift guard for the call graph."""
    result = parser.parse(SOURCE, file_id=1)
    assert sorted(_calls(result)) == [("Service.run", "beta"), ("beta", "alpha")]


def test_no_fabricated_self_recursion_from_module_docstring(parser: PythonParser) -> None:
    """
    The shifted caller index used to make beta's call to alpha look like
    alpha calling itself — a recursive edge that does not exist in the source.
    """
    result = parser.parse(MODULE_DOCSTRING + SOURCE, file_id=1)
    for caller, callee in _calls(result):
        assert caller.split(".")[-1] != callee, f"fabricated self-call {caller} -> {callee}"


def test_calls_are_identical_with_and_without_docstring(parser: PythonParser) -> None:
    bare, documented = _both(parser, SOURCE)
    assert sorted(_calls(documented)) == sorted(_calls(bare))


def test_nested_method_call_attribution_is_identical(parser: PythonParser) -> None:
    """Method-calling-method: Repo.load -> fetch, at nesting depth > 1."""
    bare, documented = _both(parser, NESTED_SOURCE)
    assert sorted(_calls(documented)) == sorted(_calls(bare))
    assert _calls(documented) == [("Repo.load", "fetch")]


def test_no_call_edge_index_is_out_of_range(parser: PythonParser) -> None:
    result = parser.parse(MODULE_DOCSTRING + SOURCE, file_id=1)
    for edge in result.call_edges:
        assert 0 <= edge.caller_id < len(result.symbols)


# ---------------------------------------------------------------------------
# TypeEdge.from_symbol_id
# ---------------------------------------------------------------------------


def test_type_edge_from_symbol_is_the_subclass_with_docstring(parser: PythonParser) -> None:
    """`class Service(Base)` must produce Service -> Base, not Base -> Base."""
    result = parser.parse(MODULE_DOCSTRING + SOURCE, file_id=1)
    assert _type_edges(result) == [("Service", "Base")]


def test_type_edge_from_symbol_is_the_subclass_without_docstring(parser: PythonParser) -> None:
    """Over-shift guard for the type hierarchy."""
    result = parser.parse(SOURCE, file_id=1)
    assert _type_edges(result) == [("Service", "Base")]


def test_no_fabricated_self_inheritance_from_module_docstring(parser: PythonParser) -> None:
    """The shifted index used to make Service(Base) look like Base(Base)."""
    result = parser.parse(MODULE_DOCSTRING + SOURCE, file_id=1)
    for subclass, base in _type_edges(result):
        assert subclass != base, f"fabricated self-inheritance {subclass} -> {base}"


def test_type_edges_are_identical_with_and_without_docstring(parser: PythonParser) -> None:
    bare, documented = _both(parser, SOURCE)
    assert _type_edges(documented) == _type_edges(bare)


# ---------------------------------------------------------------------------
# Contract preserved by the fix: "<module>" is still symbols[0], unchanged
# ---------------------------------------------------------------------------


def test_module_symbol_is_still_first(parser: PythonParser) -> None:
    """Reserving the slot must keep "<module>" at index 0, as consumers expect."""
    result = parser.parse(MODULE_DOCSTRING + SOURCE, file_id=1)
    assert result.symbols[0].kind == SymbolKind.MODULE
    assert result.symbols[0].qualified_name == "<module>"


def test_module_symbol_body_lists_top_level_symbols_but_not_itself(
    parser: PythonParser,
) -> None:
    """
    The body still carries the top-level signature index, and must not list the
    module symbol's own "module" signature — the reserved slot is skipped when
    the index is built.
    """
    body = parser.parse(MODULE_DOCSTRING + SOURCE, file_id=1).symbols[0].body
    assert body.startswith("Module docstring.")
    assert "# Symbols:" in body
    listed = body.split("# Symbols:\n", 1)[1].splitlines()
    # Signatures are asserted by prefix: _class_signature renders bases as
    # "class Service((Base))" (a pre-existing quirk unrelated to this fix).
    assert [s.split("(")[0] for s in listed] == [
        "def alpha",
        "def beta",
        "class Base",
        "class Service",
    ]
    assert "module" not in listed  # the reserved slot must not list itself


def test_docstring_adds_exactly_one_symbol(parser: PythonParser) -> None:
    """The fix must not duplicate or drop the reserved slot."""
    bare, documented = _both(parser, SOURCE)
    assert len(documented.symbols) == len(bare.symbols) + 1
    assert sum(1 for s in documented.symbols if s.kind == SymbolKind.MODULE) == 1
