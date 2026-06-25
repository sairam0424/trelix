"""
Unit tests for trelix.indexing.parser.extractors.java.JavaParser.

Covers:
- Class extraction (kind=CLASS) with extends/implements type edges
- Interface extraction (kind=INTERFACE) with method members
- Method extraction (kind=METHOD) with parent_id linkage
- Constructor extraction (kind=METHOD) with parent_id linkage
- Enum extraction (kind=ENUM) with constants
- Static final field → CONSTANT
- Annotated/public field → VARIABLE
- Import edges
- Call edges
- extends/implements TypeEdge generation
"""

from __future__ import annotations

import pytest

from trelix.core.models import SymbolKind
from trelix.indexing.parser.extractors.java import JavaParser

FILE_ID = 99


@pytest.fixture
def parser() -> JavaParser:
    return JavaParser()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def syms_by_kind(result, kind: SymbolKind):
    return [s for s in result.symbols if s.kind == kind]


def sym_named(result, name: str):
    for s in result.symbols:
        if s.name == name:
            return s
    return None


# ---------------------------------------------------------------------------
# Class extraction
# ---------------------------------------------------------------------------

CLASS_SOURCE = """\
/**
 * User service.
 */
public class UserService {
    private String name;

    public UserService(String name) {
        this.name = name;
    }

    public String getName() {
        return this.name;
    }
}
"""


def test_class_extracted(parser):
    result = parser.parse(CLASS_SOURCE, FILE_ID)
    classes = syms_by_kind(result, SymbolKind.CLASS)
    assert len(classes) == 1
    assert classes[0].name == "UserService"
    assert classes[0].file_id == FILE_ID


def test_class_is_public(parser):
    result = parser.parse(CLASS_SOURCE, FILE_ID)
    classes = syms_by_kind(result, SymbolKind.CLASS)
    assert classes[0].is_public is True


def test_class_docstring(parser):
    result = parser.parse(CLASS_SOURCE, FILE_ID)
    classes = syms_by_kind(result, SymbolKind.CLASS)
    assert classes[0].docstring is not None
    assert "service" in classes[0].docstring.lower()


# ---------------------------------------------------------------------------
# Interface extraction
# ---------------------------------------------------------------------------

INTERFACE_SOURCE = """\
public interface Repository {
    void save(Object entity);
    Object findById(long id);
}
"""


def test_interface_extracted(parser):
    result = parser.parse(INTERFACE_SOURCE, FILE_ID)
    ifaces = syms_by_kind(result, SymbolKind.INTERFACE)
    assert any(i.name == "Repository" for i in ifaces)


def test_interface_methods_extracted(parser):
    result = parser.parse(INTERFACE_SOURCE, FILE_ID)
    ifaces = syms_by_kind(result, SymbolKind.INTERFACE)
    repo = next(i for i in ifaces if i.name == "Repository")
    repo_idx = result.symbols.index(repo)

    methods = [s for s in result.symbols if s.kind == SymbolKind.METHOD and s.parent_id == repo_idx]
    method_names = {m.name for m in methods}
    assert "save" in method_names
    assert "findById" in method_names


# ---------------------------------------------------------------------------
# Method extraction and parent linkage
# ---------------------------------------------------------------------------


def test_method_extracted(parser):
    result = parser.parse(CLASS_SOURCE, FILE_ID)
    methods = syms_by_kind(result, SymbolKind.METHOD)
    names = {m.name for m in methods}
    assert "getName" in names


def test_method_parent_id(parser):
    result = parser.parse(CLASS_SOURCE, FILE_ID)
    classes = syms_by_kind(result, SymbolKind.CLASS)
    class_idx = result.symbols.index(classes[0])

    methods = syms_by_kind(result, SymbolKind.METHOD)
    for m in methods:
        assert m.parent_id == class_idx, (
            f"Method {m.name}.parent_id={m.parent_id}, expected {class_idx}"
        )


def test_method_qualified_name(parser):
    result = parser.parse(CLASS_SOURCE, FILE_ID)
    methods = syms_by_kind(result, SymbolKind.METHOD)
    get_name = next(m for m in methods if m.name == "getName")
    assert get_name.qualified_name == "UserService.getName"


def test_method_signature(parser):
    result = parser.parse(CLASS_SOURCE, FILE_ID)
    methods = syms_by_kind(result, SymbolKind.METHOD)
    get_name = next(m for m in methods if m.name == "getName")
    assert "getName" in get_name.signature
    assert "UserService" in get_name.signature


# ---------------------------------------------------------------------------
# Constructor extraction
# ---------------------------------------------------------------------------


def test_constructor_extracted(parser):
    result = parser.parse(CLASS_SOURCE, FILE_ID)
    methods = syms_by_kind(result, SymbolKind.METHOD)
    ctors = [m for m in methods if m.name == "UserService"]
    assert len(ctors) == 1


def test_constructor_parent_id(parser):
    result = parser.parse(CLASS_SOURCE, FILE_ID)
    classes = syms_by_kind(result, SymbolKind.CLASS)
    class_idx = result.symbols.index(classes[0])

    methods = syms_by_kind(result, SymbolKind.METHOD)
    ctor = next(m for m in methods if m.name == "UserService")
    assert ctor.parent_id == class_idx


# ---------------------------------------------------------------------------
# Extends / implements type edges
# ---------------------------------------------------------------------------

EXTENDS_SOURCE = """\
public class AdminUser extends BaseUser implements Serializable, Auditable {
    public void doAdmin() {}
}
"""


def test_extends_type_edge(parser):
    result = parser.parse(EXTENDS_SOURCE, FILE_ID)
    edge_kinds = {e.edge_kind for e in result.type_edges}
    assert "extends" in edge_kinds
    extends_edges = [e for e in result.type_edges if e.edge_kind == "extends"]
    to_names = {e.to_type_name for e in extends_edges}
    assert "BaseUser" in to_names


def test_implements_type_edges(parser):
    result = parser.parse(EXTENDS_SOURCE, FILE_ID)
    impl_edges = [e for e in result.type_edges if e.edge_kind == "implements"]
    to_names = {e.to_type_name for e in impl_edges}
    assert "Serializable" in to_names
    assert "Auditable" in to_names


# ---------------------------------------------------------------------------
# Enum extraction
# ---------------------------------------------------------------------------

ENUM_SOURCE = """\
public enum Status {
    PENDING,
    ACTIVE,
    DISABLED
}
"""


def test_enum_extracted(parser):
    result = parser.parse(ENUM_SOURCE, FILE_ID)
    enums = syms_by_kind(result, SymbolKind.ENUM)
    assert len(enums) == 1
    assert enums[0].name == "Status"


def test_enum_constants_extracted(parser):
    result = parser.parse(ENUM_SOURCE, FILE_ID)
    consts = syms_by_kind(result, SymbolKind.CONSTANT)
    names = {c.name for c in consts}
    assert "PENDING" in names
    assert "ACTIVE" in names
    assert "DISABLED" in names


def test_enum_constant_qualified_name(parser):
    result = parser.parse(ENUM_SOURCE, FILE_ID)
    consts = syms_by_kind(result, SymbolKind.CONSTANT)
    pending = next(c for c in consts if c.name == "PENDING")
    assert pending.qualified_name == "Status.PENDING"


def test_enum_constant_parent_id(parser):
    result = parser.parse(ENUM_SOURCE, FILE_ID)
    enums = syms_by_kind(result, SymbolKind.ENUM)
    enum_idx = result.symbols.index(enums[0])
    for const in syms_by_kind(result, SymbolKind.CONSTANT):
        assert const.parent_id == enum_idx


# ---------------------------------------------------------------------------
# Static final field → CONSTANT
# ---------------------------------------------------------------------------

CONST_FIELD_SOURCE = """\
public class Config {
    public static final String VERSION = "1.0.0";
    public static final int MAX_CONNECTIONS = 100;
    private String secret;
}
"""


def test_static_final_field_as_constant(parser):
    result = parser.parse(CONST_FIELD_SOURCE, FILE_ID)
    consts = syms_by_kind(result, SymbolKind.CONSTANT)
    names = {c.name for c in consts}
    assert "VERSION" in names
    assert "MAX_CONNECTIONS" in names
    assert "secret" not in names


# ---------------------------------------------------------------------------
# Import edges
# ---------------------------------------------------------------------------

IMPORT_SOURCE = """\
import java.util.List;
import java.util.HashMap;

public class App {
    public void run() {}
}
"""


def test_import_edges_extracted(parser):
    result = parser.parse(IMPORT_SOURCE, FILE_ID)
    assert len(result.import_edges) >= 2
    paths = {e.imported_from for e in result.import_edges}
    assert any("java.util" in p or "util" in p for p in paths)


# ---------------------------------------------------------------------------
# Call edges
# ---------------------------------------------------------------------------

CALL_SOURCE = """\
import java.util.ArrayList;

public class Processor {
    public void process() {
        ArrayList<String> list = new ArrayList<>();
        list.add("item");
        helper();
    }

    private void helper() {}
}
"""


def test_call_edges_extracted(parser):
    result = parser.parse(CALL_SOURCE, FILE_ID)
    callee_names = {e.callee_name for e in result.call_edges}
    assert "add" in callee_names or "helper" in callee_names or "ArrayList" in callee_names


# ---------------------------------------------------------------------------
# Parse errors
# ---------------------------------------------------------------------------


def test_clean_parse_no_errors(parser):
    result = parser.parse(CLASS_SOURCE, FILE_ID)
    assert result.parse_errors == 0


# ---------------------------------------------------------------------------
# Language name
# ---------------------------------------------------------------------------


def test_language_name(parser):
    assert parser.language_name == "java"
