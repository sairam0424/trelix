"""
Unit tests for trelix.indexing.parser.extractors.rust.RustParser.

Covers:
- Struct extraction (kind=CLASS) with pub fields
- Enum extraction (kind=ENUM) with variants as CONSTANT
- Trait extraction (kind=INTERFACE) with method signatures
- Impl method extraction (kind=METHOD) linked to struct via parent_id
- Top-level function extraction (kind=FUNCTION)
- Import edges (use declarations)
- Call edges
- Type alias
- Constants and statics
"""

from __future__ import annotations

import pytest

from trelix.core.models import SymbolKind
from trelix.indexing.parser.extractors.rust import RustParser

FILE_ID = 7


@pytest.fixture
def parser() -> RustParser:
    return RustParser()


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
# Struct extraction
# ---------------------------------------------------------------------------

STRUCT_SOURCE = """\
/// A network connection.
pub struct Connection {
    pub host: String,
    pub port: u16,
    secret: Vec<u8>,
}
"""


def test_struct_extracted(parser):
    result = parser.parse(STRUCT_SOURCE, FILE_ID)
    structs = syms_by_kind(result, SymbolKind.CLASS)
    assert len(structs) == 1
    assert structs[0].name == "Connection"
    assert structs[0].file_id == FILE_ID


def test_struct_is_public(parser):
    result = parser.parse(STRUCT_SOURCE, FILE_ID)
    structs = syms_by_kind(result, SymbolKind.CLASS)
    assert structs[0].is_public is True


def test_struct_pub_fields_extracted(parser):
    result = parser.parse(STRUCT_SOURCE, FILE_ID)
    fields = syms_by_kind(result, SymbolKind.VARIABLE)
    field_names = {f.name for f in fields}
    assert "host" in field_names
    assert "port" in field_names
    # private field must NOT appear
    assert "secret" not in field_names


def test_struct_field_parent_id(parser):
    result = parser.parse(STRUCT_SOURCE, FILE_ID)
    structs = syms_by_kind(result, SymbolKind.CLASS)
    struct_idx = result.symbols.index(structs[0])
    for field in syms_by_kind(result, SymbolKind.VARIABLE):
        assert field.parent_id == struct_idx


def test_struct_signature(parser):
    result = parser.parse(STRUCT_SOURCE, FILE_ID)
    structs = syms_by_kind(result, SymbolKind.CLASS)
    assert "struct" in structs[0].signature
    assert "Connection" in structs[0].signature


def test_struct_docstring(parser):
    result = parser.parse(STRUCT_SOURCE, FILE_ID)
    structs = syms_by_kind(result, SymbolKind.CLASS)
    # The preceding /// comment is extracted as a docstring
    assert structs[0].docstring is not None
    assert "network connection" in structs[0].docstring.lower()


# ---------------------------------------------------------------------------
# Enum extraction
# ---------------------------------------------------------------------------

ENUM_SOURCE = """\
pub enum Color {
    Red,
    Green,
    Blue,
}
"""


def test_enum_extracted(parser):
    result = parser.parse(ENUM_SOURCE, FILE_ID)
    enums = syms_by_kind(result, SymbolKind.ENUM)
    assert len(enums) == 1
    assert enums[0].name == "Color"


def test_enum_variants_as_constants(parser):
    result = parser.parse(ENUM_SOURCE, FILE_ID)
    consts = syms_by_kind(result, SymbolKind.CONSTANT)
    variant_names = {c.name for c in consts}
    assert "Red" in variant_names
    assert "Green" in variant_names
    assert "Blue" in variant_names


def test_enum_variant_qualified_name(parser):
    result = parser.parse(ENUM_SOURCE, FILE_ID)
    consts = syms_by_kind(result, SymbolKind.CONSTANT)
    red = next(c for c in consts if c.name == "Red")
    assert red.qualified_name == "Color::Red"


def test_enum_variant_parent_id(parser):
    result = parser.parse(ENUM_SOURCE, FILE_ID)
    enums = syms_by_kind(result, SymbolKind.ENUM)
    enum_idx = result.symbols.index(enums[0])
    for variant in syms_by_kind(result, SymbolKind.CONSTANT):
        assert variant.parent_id == enum_idx


# ---------------------------------------------------------------------------
# Trait extraction
# ---------------------------------------------------------------------------

TRAIT_SOURCE = """\
pub trait Drawable {
    fn draw(&self);
    fn resize(&mut self, factor: f64);
}
"""


def test_trait_extracted(parser):
    result = parser.parse(TRAIT_SOURCE, FILE_ID)
    ifaces = syms_by_kind(result, SymbolKind.INTERFACE)
    assert any(i.name == "Drawable" for i in ifaces)


def test_trait_method_signatures(parser):
    result = parser.parse(TRAIT_SOURCE, FILE_ID)
    ifaces = syms_by_kind(result, SymbolKind.INTERFACE)
    drawable = next(i for i in ifaces if i.name == "Drawable")
    drawable_idx = result.symbols.index(drawable)

    # Trait methods are METHOD symbols with parent_id = trait idx
    methods = [
        s for s in result.symbols if s.kind == SymbolKind.METHOD and s.parent_id == drawable_idx
    ]
    method_names = {m.name for m in methods}
    assert "draw" in method_names
    assert "resize" in method_names


def test_trait_method_qualified_name(parser):
    result = parser.parse(TRAIT_SOURCE, FILE_ID)
    methods = syms_by_kind(result, SymbolKind.METHOD)
    draw = next(m for m in methods if m.name == "draw")
    assert draw.qualified_name == "Drawable::draw"


# ---------------------------------------------------------------------------
# Impl methods — parent_id linkage to struct
# ---------------------------------------------------------------------------

IMPL_SOURCE = """\
pub struct Rectangle {
    pub width: f64,
    pub height: f64,
}

impl Rectangle {
    pub fn area(&self) -> f64 {
        self.width * self.height
    }

    pub fn new(width: f64, height: f64) -> Self {
        Self { width, height }
    }
}
"""


def test_impl_method_extracted(parser):
    result = parser.parse(IMPL_SOURCE, FILE_ID)
    methods = syms_by_kind(result, SymbolKind.METHOD)
    assert any(m.name == "area" for m in methods)


def test_impl_method_parent_id(parser):
    result = parser.parse(IMPL_SOURCE, FILE_ID)
    structs = syms_by_kind(result, SymbolKind.CLASS)
    rect = next(s for s in structs if s.name == "Rectangle")
    rect_idx = result.symbols.index(rect)

    methods = syms_by_kind(result, SymbolKind.METHOD)
    area = next(m for m in methods if m.name == "area")
    assert area.parent_id == rect_idx


def test_impl_associated_function_kind(parser):
    """new() has no self param → FUNCTION not METHOD."""
    result = parser.parse(IMPL_SOURCE, FILE_ID)
    funcs = syms_by_kind(result, SymbolKind.FUNCTION)
    assert any(f.name == "new" for f in funcs)


def test_impl_method_qualified_name(parser):
    result = parser.parse(IMPL_SOURCE, FILE_ID)
    methods = syms_by_kind(result, SymbolKind.METHOD)
    area = next(m for m in methods if m.name == "area")
    assert area.qualified_name == "Rectangle::area"


# ---------------------------------------------------------------------------
# Top-level function
# ---------------------------------------------------------------------------

FUNC_SOURCE = """\
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}

fn private_helper() {}
"""


def test_toplevel_function_extracted(parser):
    result = parser.parse(FUNC_SOURCE, FILE_ID)
    funcs = syms_by_kind(result, SymbolKind.FUNCTION)
    assert any(f.name == "add" for f in funcs)


def test_toplevel_function_is_public(parser):
    result = parser.parse(FUNC_SOURCE, FILE_ID)
    funcs = syms_by_kind(result, SymbolKind.FUNCTION)
    add = next(f for f in funcs if f.name == "add")
    assert add.is_public is True


def test_private_function_is_not_public(parser):
    result = parser.parse(FUNC_SOURCE, FILE_ID)
    funcs = syms_by_kind(result, SymbolKind.FUNCTION)
    helper = next(f for f in funcs if f.name == "private_helper")
    assert helper.is_public is False


# ---------------------------------------------------------------------------
# Import edges
# ---------------------------------------------------------------------------

IMPORT_SOURCE = """\
use std::collections::HashMap;
use std::io::{self, Write};

fn main() {}
"""


def test_import_edges_extracted(parser):
    result = parser.parse(IMPORT_SOURCE, FILE_ID)
    assert len(result.import_edges) > 0
    all_paths = " ".join(e.imported_from for e in result.import_edges)
    assert "std" in all_paths or "collections" in all_paths


# ---------------------------------------------------------------------------
# Call edges
# ---------------------------------------------------------------------------

CALL_SOURCE = """\
fn process(items: Vec<String>) -> usize {
    let count = items.len();
    println!("{}", count);
    helper(count)
}

fn helper(n: usize) -> usize { n }
"""


def test_call_edges_extracted(parser):
    result = parser.parse(CALL_SOURCE, FILE_ID)
    callee_names = {e.callee_name for e in result.call_edges}
    assert "helper" in callee_names or "len" in callee_names


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

TYPE_ALIAS_SOURCE = """\
pub type Result<T> = std::result::Result<T, String>;
"""


def test_type_alias_extracted(parser):
    result = parser.parse(TYPE_ALIAS_SOURCE, FILE_ID)
    ifaces = syms_by_kind(result, SymbolKind.INTERFACE)
    assert any(i.name == "Result" for i in ifaces)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONST_SOURCE = """\
pub const MAX_SIZE: usize = 1024;
const PRIVATE_LIMIT: u32 = 10;
static GREETING: &str = "hello";
"""


def test_const_extracted(parser):
    result = parser.parse(CONST_SOURCE, FILE_ID)
    consts = syms_by_kind(result, SymbolKind.CONSTANT)
    names = {c.name for c in consts}
    assert "MAX_SIZE" in names
    assert "PRIVATE_LIMIT" in names
    assert "GREETING" in names


# ---------------------------------------------------------------------------
# Parse errors
# ---------------------------------------------------------------------------


def test_clean_parse_no_errors(parser):
    result = parser.parse(STRUCT_SOURCE, FILE_ID)
    assert result.parse_errors == 0


# ---------------------------------------------------------------------------
# Language name
# ---------------------------------------------------------------------------


def test_language_name(parser):
    assert parser.language_name == "rust"
