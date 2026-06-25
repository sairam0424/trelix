"""
Unit tests for trelix.indexing.parser.extractors.go.GoParser.

Covers:
- Struct extraction (kind=CLASS)
- Interface extraction (kind=INTERFACE) with method specs
- Function extraction (kind=FUNCTION)
- Method extraction (kind=METHOD) with receiver linkage to struct parent_id
- Import edges
- Call edges
- Type alias and named type extraction
- Package-level constants
"""
from __future__ import annotations

import pytest

from trelix.core.models import SymbolKind
from trelix.indexing.parser.extractors.go import GoParser


FILE_ID = 42


@pytest.fixture
def parser() -> GoParser:
    return GoParser()


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
package server

// Server is the main HTTP server.
type Server struct {
    Host    string
    Port    int
    Timeout int
    secret  string  // unexported — should be skipped
}
"""


def test_struct_extracted(parser):
    result = parser.parse(STRUCT_SOURCE, FILE_ID)
    structs = syms_by_kind(result, SymbolKind.CLASS)
    assert len(structs) == 1
    assert structs[0].name == "Server"
    assert structs[0].qualified_name == "Server"
    assert structs[0].file_id == FILE_ID
    assert structs[0].is_public is True


def test_struct_signature(parser):
    result = parser.parse(STRUCT_SOURCE, FILE_ID)
    structs = syms_by_kind(result, SymbolKind.CLASS)
    assert "struct" in structs[0].signature
    assert "Server" in structs[0].signature


def test_struct_exported_fields_extracted(parser):
    result = parser.parse(STRUCT_SOURCE, FILE_ID)
    fields = syms_by_kind(result, SymbolKind.VARIABLE)
    field_names = {f.name for f in fields}
    assert "Host" in field_names
    assert "Port" in field_names
    assert "Timeout" in field_names
    # unexported field must NOT be extracted
    assert "secret" not in field_names


def test_struct_field_parent_id(parser):
    result = parser.parse(STRUCT_SOURCE, FILE_ID)
    structs = syms_by_kind(result, SymbolKind.CLASS)
    struct_idx = result.symbols.index(structs[0])
    for field in syms_by_kind(result, SymbolKind.VARIABLE):
        assert field.parent_id == struct_idx, (
            f"Field {field.name} has parent_id={field.parent_id}, expected {struct_idx}"
        )


# ---------------------------------------------------------------------------
# Interface extraction
# ---------------------------------------------------------------------------

INTERFACE_SOURCE = """\
package handler

type Handler interface {
    ServeHTTP(w ResponseWriter, r *Request)
    Close() error
}
"""


def test_interface_extracted(parser):
    result = parser.parse(INTERFACE_SOURCE, FILE_ID)
    ifaces = syms_by_kind(result, SymbolKind.INTERFACE)
    assert any(i.name == "Handler" for i in ifaces)


def test_interface_method_specs(parser):
    result = parser.parse(INTERFACE_SOURCE, FILE_ID)
    ifaces = syms_by_kind(result, SymbolKind.INTERFACE)
    handler = next(i for i in ifaces if i.name == "Handler")
    handler_idx = result.symbols.index(handler)

    # Interface method specs are emitted as FUNCTION symbols with parent_id = interface idx
    iface_funcs = [
        s for s in result.symbols
        if s.kind == SymbolKind.FUNCTION and s.parent_id == handler_idx
    ]
    method_names = {f.name for f in iface_funcs}
    assert "ServeHTTP" in method_names
    assert "Close" in method_names


# ---------------------------------------------------------------------------
# Function extraction
# ---------------------------------------------------------------------------

FUNCTION_SOURCE = """\
package main

func NewServer(host string, port int) *Server {
    return &Server{Host: host, Port: port}
}

func unexported() {}
"""


def test_exported_function_extracted(parser):
    result = parser.parse(FUNCTION_SOURCE, FILE_ID)
    funcs = syms_by_kind(result, SymbolKind.FUNCTION)
    assert any(f.name == "NewServer" for f in funcs)
    new_server = next(f for f in funcs if f.name == "NewServer")
    assert new_server.is_public is True


def test_unexported_function_extracted(parser):
    # Unexported functions are still extracted — is_public=False
    result = parser.parse(FUNCTION_SOURCE, FILE_ID)
    funcs = syms_by_kind(result, SymbolKind.FUNCTION)
    assert any(f.name == "unexported" for f in funcs)
    unexp = next(f for f in funcs if f.name == "unexported")
    assert unexp.is_public is False


def test_function_signature(parser):
    result = parser.parse(FUNCTION_SOURCE, FILE_ID)
    funcs = syms_by_kind(result, SymbolKind.FUNCTION)
    new_server = next(f for f in funcs if f.name == "NewServer")
    assert "func" in new_server.signature
    assert "NewServer" in new_server.signature


# ---------------------------------------------------------------------------
# Method extraction and receiver linkage
# ---------------------------------------------------------------------------

METHOD_SOURCE = """\
package server

type Server struct {
    Host string
}

// Start starts the server.
func (s *Server) Start() error {
    return nil
}

func (s *Server) unexported() {}
"""


def test_method_extracted(parser):
    result = parser.parse(METHOD_SOURCE, FILE_ID)
    methods = syms_by_kind(result, SymbolKind.METHOD)
    assert any(m.name == "Start" for m in methods)


def test_method_receiver_linkage(parser):
    """Method parent_id must point to the Server struct's local index."""
    result = parser.parse(METHOD_SOURCE, FILE_ID)
    structs = syms_by_kind(result, SymbolKind.CLASS)
    server_struct = next(s for s in structs if s.name == "Server")
    server_idx = result.symbols.index(server_struct)

    methods = syms_by_kind(result, SymbolKind.METHOD)
    for m in methods:
        assert m.parent_id == server_idx, (
            f"Method {m.name} parent_id={m.parent_id}, expected {server_idx}"
        )


def test_method_qualified_name(parser):
    result = parser.parse(METHOD_SOURCE, FILE_ID)
    methods = syms_by_kind(result, SymbolKind.METHOD)
    start = next(m for m in methods if m.name == "Start")
    assert start.qualified_name == "Server.Start"


def test_method_signature_contains_receiver(parser):
    result = parser.parse(METHOD_SOURCE, FILE_ID)
    methods = syms_by_kind(result, SymbolKind.METHOD)
    start = next(m for m in methods if m.name == "Start")
    assert "Server" in start.signature


def test_method_docstring(parser):
    result = parser.parse(METHOD_SOURCE, FILE_ID)
    methods = syms_by_kind(result, SymbolKind.METHOD)
    start = next(m for m in methods if m.name == "Start")
    assert start.docstring is not None
    assert "Start" in start.docstring


# ---------------------------------------------------------------------------
# Import edges
# ---------------------------------------------------------------------------

IMPORT_SOURCE = """\
package main

import (
    "fmt"
    "net/http"
)
"""


def test_import_edges(parser):
    result = parser.parse(IMPORT_SOURCE, FILE_ID)
    paths = {e.imported_from for e in result.import_edges}
    assert "fmt" in paths
    assert "net/http" in paths


# ---------------------------------------------------------------------------
# Call edges
# ---------------------------------------------------------------------------

CALL_SOURCE = """\
package main

import "fmt"

func greet(name string) {
    fmt.Println(name)
    helper()
}

func helper() {}
"""


def test_call_edges_extracted(parser):
    result = parser.parse(CALL_SOURCE, FILE_ID)
    callee_names = {e.callee_name for e in result.call_edges}
    assert "Println" in callee_names
    assert "helper" in callee_names


# ---------------------------------------------------------------------------
# Type alias / named type
# ---------------------------------------------------------------------------

TYPE_ALIAS_SOURCE = """\
package types

type MyString = string
type MyInt int
"""


def test_type_alias_extracted(parser):
    result = parser.parse(TYPE_ALIAS_SOURCE, FILE_ID)
    ifaces = syms_by_kind(result, SymbolKind.INTERFACE)
    names = {i.name for i in ifaces}
    assert "MyString" in names
    assert "MyInt" in names


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONST_SOURCE = """\
package config

const MaxRetries = 3
const DefaultTimeout = 30
const privateConst = "secret"
"""


def test_exported_const_extracted(parser):
    result = parser.parse(CONST_SOURCE, FILE_ID)
    consts = syms_by_kind(result, SymbolKind.CONSTANT)
    names = {c.name for c in consts}
    assert "MaxRetries" in names
    assert "DefaultTimeout" in names
    assert "privateConst" not in names


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
    assert parser.language_name == "go"


# ---------------------------------------------------------------------------
# Empty source
# ---------------------------------------------------------------------------

def test_empty_source(parser):
    result = parser.parse("package main\n", FILE_ID)
    # Should not raise — may have zero symbols
    assert isinstance(result.symbols, list)
    assert result.parse_errors == 0
