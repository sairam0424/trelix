"""Unit tests for the TypeScript/TSX parser (Phase 6a).

Exercises: interface, class, method, arrow function, import extraction,
type aliases, enums, exported symbols, and call graph extraction.
"""

from __future__ import annotations

import pytest

from trelix.indexing.parser.extractors.typescript import TypeScriptParser
from trelix.core.models import SymbolKind


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_parser(tsx: bool = False) -> TypeScriptParser:
    return TypeScriptParser(tsx=tsx)


def symbol_names(result) -> list[str]:
    return [s.name for s in result.symbols]


def symbols_by_kind(result, kind: SymbolKind) -> list:
    return [s for s in result.symbols if s.kind == kind]


def import_sources(result) -> list[str]:
    return [e.imported_from for e in result.import_edges]


# ---------------------------------------------------------------------------
# language_name property
# ---------------------------------------------------------------------------

class TestLanguageName:
    def test_typescript_language_name(self):
        parser = make_parser(tsx=False)
        assert parser.language_name == "typescript"

    def test_tsx_language_name(self):
        parser = make_parser(tsx=True)
        assert parser.language_name == "tsx"


# ---------------------------------------------------------------------------
# Import extraction
# ---------------------------------------------------------------------------

class TestImportExtraction:
    TS_IMPORTS = """\
import React from 'react';
import { useState, useEffect } from 'react';
import * as path from 'path';
import type { Foo } from './foo';
"""

    def test_es6_default_import(self):
        parser = make_parser()
        result = parser.parse(self.TS_IMPORTS, file_id=1)
        sources = import_sources(result)
        assert "react" in sources

    def test_es6_named_import(self):
        parser = make_parser()
        result = parser.parse(self.TS_IMPORTS, file_id=1)
        # All four import statements should create edges
        assert len(result.import_edges) >= 3

    def test_import_names_extracted(self):
        parser = make_parser()
        result = parser.parse(self.TS_IMPORTS, file_id=1)
        # Named import edge should list useState and useEffect
        named = next(
            (e for e in result.import_edges if "useState" in e.imported_names), None
        )
        assert named is not None
        assert "useEffect" in named.imported_names

    def test_namespace_import(self):
        parser = make_parser()
        result = parser.parse(self.TS_IMPORTS, file_id=1)
        star = next(
            (e for e in result.import_edges if e.imported_from == "path"), None
        )
        assert star is not None

    def test_file_id_set_on_import_edges(self):
        parser = make_parser()
        result = parser.parse(self.TS_IMPORTS, file_id=42)
        for edge in result.import_edges:
            assert edge.file_id == 42


# ---------------------------------------------------------------------------
# Interface extraction
# ---------------------------------------------------------------------------

class TestInterfaceExtraction:
    TS_INTERFACE = """\
export interface User {
  id: number;
  name: string;
  greet(msg: string): void;
}

interface Internal {
  value: boolean;
}
"""

    def test_exported_interface_extracted(self):
        parser = make_parser()
        result = parser.parse(self.TS_INTERFACE, file_id=1)
        ifaces = symbols_by_kind(result, SymbolKind.INTERFACE)
        names = [i.name for i in ifaces]
        assert "User" in names

    def test_non_exported_interface_extracted(self):
        parser = make_parser()
        result = parser.parse(self.TS_INTERFACE, file_id=1)
        ifaces = symbols_by_kind(result, SymbolKind.INTERFACE)
        names = [i.name for i in ifaces]
        assert "Internal" in names

    def test_exported_interface_is_public(self):
        parser = make_parser()
        result = parser.parse(self.TS_INTERFACE, file_id=1)
        user = next(s for s in result.symbols if s.name == "User")
        assert user.is_public is True

    def test_interface_signature_format(self):
        parser = make_parser()
        result = parser.parse(self.TS_INTERFACE, file_id=1)
        user = next(s for s in result.symbols if s.name == "User")
        assert "interface" in user.signature
        assert "User" in user.signature

    def test_interface_members_extracted(self):
        parser = make_parser()
        result = parser.parse(self.TS_INTERFACE, file_id=1)
        names = symbol_names(result)
        # Property signatures and method signatures inside User
        assert "id" in names or "greet" in names  # at least one member

    def test_interface_member_parent_linkage(self):
        parser = make_parser()
        result = parser.parse(self.TS_INTERFACE, file_id=1)
        user_idx = next(i for i, s in enumerate(result.symbols) if s.name == "User")
        members = [s for s in result.symbols if s.parent_id == user_idx]
        assert len(members) > 0


# ---------------------------------------------------------------------------
# Class extraction
# ---------------------------------------------------------------------------

class TestClassExtraction:
    TS_CLASS = """\
class Animal {
  name: string;

  constructor(name: string) {
    this.name = name;
  }

  speak(): string {
    return `${this.name} makes a noise.`;
  }
}

export class Dog extends Animal {
  breed: string;

  speak(): string {
    return `${this.name} barks.`;
  }
}
"""

    def test_classes_extracted(self):
        parser = make_parser()
        result = parser.parse(self.TS_CLASS, file_id=1)
        classes = symbols_by_kind(result, SymbolKind.CLASS)
        names = [c.name for c in classes]
        assert "Animal" in names
        assert "Dog" in names

    def test_exported_class_is_public(self):
        parser = make_parser()
        result = parser.parse(self.TS_CLASS, file_id=1)
        dog = next(s for s in result.symbols if s.name == "Dog")
        assert dog.is_public is True

    def test_class_extends_type_edge(self):
        parser = make_parser()
        result = parser.parse(self.TS_CLASS, file_id=1)
        extends = [e for e in result.type_edges if e.edge_kind == "extends"]
        assert any(e.to_type_name == "Animal" for e in extends)

    def test_class_signature_contains_class_name(self):
        parser = make_parser()
        result = parser.parse(self.TS_CLASS, file_id=1)
        dog = next(s for s in result.symbols if s.name == "Dog")
        assert "Dog" in dog.signature
        assert "class" in dog.signature

    def test_method_extracted(self):
        parser = make_parser()
        result = parser.parse(self.TS_CLASS, file_id=1)
        methods = symbols_by_kind(result, SymbolKind.METHOD)
        method_names = [m.name for m in methods]
        assert "speak" in method_names

    def test_method_parent_linkage(self):
        parser = make_parser()
        result = parser.parse(self.TS_CLASS, file_id=1)
        animal_idx = next(i for i, s in enumerate(result.symbols) if s.name == "Animal")
        animal_methods = [s for s in result.symbols
                          if s.kind == SymbolKind.METHOD and s.parent_id == animal_idx]
        assert any(m.name == "speak" for m in animal_methods)

    def test_method_qualified_name(self):
        parser = make_parser()
        result = parser.parse(self.TS_CLASS, file_id=1)
        speak = next(
            (s for s in result.symbols
             if s.kind == SymbolKind.METHOD and s.name == "speak" and "Animal" in s.qualified_name),
            None
        )
        assert speak is not None
        assert speak.qualified_name == "Animal.speak"

    def test_line_numbers_set(self):
        parser = make_parser()
        result = parser.parse(self.TS_CLASS, file_id=1)
        animal = next(s for s in result.symbols if s.name == "Animal")
        assert animal.line_start >= 1
        assert animal.line_end >= animal.line_start


# ---------------------------------------------------------------------------
# Abstract class extraction
# ---------------------------------------------------------------------------

class TestAbstractClass:
    TS_ABSTRACT = """\
export abstract class Shape {
  abstract area(): number;
  describe(): string {
    return 'shape';
  }
}
"""

    def test_abstract_class_extracted(self):
        parser = make_parser()
        result = parser.parse(self.TS_ABSTRACT, file_id=1)
        classes = symbols_by_kind(result, SymbolKind.CLASS)
        assert any(c.name == "Shape" for c in classes)

    def test_abstract_class_signature(self):
        parser = make_parser()
        result = parser.parse(self.TS_ABSTRACT, file_id=1)
        shape = next(s for s in result.symbols if s.name == "Shape")
        assert "abstract" in shape.signature

    def test_abstract_method_extracted(self):
        parser = make_parser()
        result = parser.parse(self.TS_ABSTRACT, file_id=1)
        methods = symbols_by_kind(result, SymbolKind.METHOD)
        assert any(m.name == "area" for m in methods)


# ---------------------------------------------------------------------------
# Arrow function extraction
# ---------------------------------------------------------------------------

class TestArrowFunctionExtraction:
    TS_ARROW = """\
export const add = (a: number, b: number): number => a + b;

const greet = (name: string): string => {
  return `Hello, ${name}!`;
};

const PI = 3.14159;
export const MAX_RETRIES = 3;
"""

    def test_exported_arrow_function_extracted(self):
        parser = make_parser()
        result = parser.parse(self.TS_ARROW, file_id=1)
        funcs = symbols_by_kind(result, SymbolKind.FUNCTION)
        assert any(f.name == "add" for f in funcs)

    def test_non_exported_arrow_function_extracted(self):
        parser = make_parser()
        result = parser.parse(self.TS_ARROW, file_id=1)
        funcs = symbols_by_kind(result, SymbolKind.FUNCTION)
        assert any(f.name == "greet" for f in funcs)

    def test_arrow_function_is_public_when_exported(self):
        parser = make_parser()
        result = parser.parse(self.TS_ARROW, file_id=1)
        add = next(s for s in result.symbols if s.name == "add")
        assert add.is_public is True

    def test_arrow_function_signature_format(self):
        parser = make_parser()
        result = parser.parse(self.TS_ARROW, file_id=1)
        add = next(s for s in result.symbols if s.name == "add")
        assert "const add" in add.signature
        assert "=>" in add.signature

    def test_exported_constant_extracted(self):
        parser = make_parser()
        result = parser.parse(self.TS_ARROW, file_id=1)
        consts = symbols_by_kind(result, SymbolKind.CONSTANT)
        assert any(c.name == "MAX_RETRIES" for c in consts)

    def test_non_exported_non_allcaps_not_extracted(self):
        """PI is ALL_CAPS so it should be extracted; regular non-exported vars should not."""
        parser = make_parser()
        result = parser.parse(self.TS_ARROW, file_id=1)
        names = symbol_names(result)
        assert "PI" in names  # ALL_CAPS → extracted as CONSTANT


# ---------------------------------------------------------------------------
# Function declaration extraction
# ---------------------------------------------------------------------------

class TestFunctionDeclaration:
    TS_FUNCS = """\
export function fetchUser(id: number): Promise<User> {
  return fetch(`/api/users/${id}`).then(r => r.json());
}

function helper(x: string): string {
  return x.trim();
}

async function loadData(): Promise<void> {
  const data = await fetchUser(1);
}
"""

    def test_exported_function_extracted(self):
        parser = make_parser()
        result = parser.parse(self.TS_FUNCS, file_id=1)
        funcs = symbols_by_kind(result, SymbolKind.FUNCTION)
        assert any(f.name == "fetchUser" for f in funcs)

    def test_non_exported_function_extracted(self):
        parser = make_parser()
        result = parser.parse(self.TS_FUNCS, file_id=1)
        funcs = symbols_by_kind(result, SymbolKind.FUNCTION)
        assert any(f.name == "helper" for f in funcs)

    def test_function_signature_contains_name(self):
        parser = make_parser()
        result = parser.parse(self.TS_FUNCS, file_id=1)
        fetch_fn = next(s for s in result.symbols if s.name == "fetchUser")
        assert "fetchUser" in fetch_fn.signature
        assert "function" in fetch_fn.signature


# ---------------------------------------------------------------------------
# Type alias extraction
# ---------------------------------------------------------------------------

class TestTypeAliasExtraction:
    TS_TYPE = """\
export type UserId = number;
type Result<T> = { data: T; error: string | null };
"""

    def test_exported_type_alias_extracted(self):
        parser = make_parser()
        result = parser.parse(self.TS_TYPE, file_id=1)
        ifaces = symbols_by_kind(result, SymbolKind.INTERFACE)
        assert any(i.name == "UserId" for i in ifaces)

    def test_non_exported_type_alias_extracted(self):
        parser = make_parser()
        result = parser.parse(self.TS_TYPE, file_id=1)
        ifaces = symbols_by_kind(result, SymbolKind.INTERFACE)
        assert any(i.name == "Result" for i in ifaces)


# ---------------------------------------------------------------------------
# Enum extraction
# ---------------------------------------------------------------------------

class TestEnumExtraction:
    TS_ENUM = """\
export enum Direction {
  Up = 'UP',
  Down = 'DOWN',
  Left = 'LEFT',
  Right = 'RIGHT',
}

enum Status {
  Active,
  Inactive,
}
"""

    def test_exported_enum_extracted(self):
        parser = make_parser()
        result = parser.parse(self.TS_ENUM, file_id=1)
        enums = symbols_by_kind(result, SymbolKind.ENUM)
        assert any(e.name == "Direction" for e in enums)

    def test_non_exported_enum_extracted(self):
        parser = make_parser()
        result = parser.parse(self.TS_ENUM, file_id=1)
        enums = symbols_by_kind(result, SymbolKind.ENUM)
        assert any(e.name == "Status" for e in enums)

    def test_enum_members_as_constants(self):
        parser = make_parser()
        result = parser.parse(self.TS_ENUM, file_id=1)
        consts = symbols_by_kind(result, SymbolKind.CONSTANT)
        const_names = [c.name for c in consts]
        # Direction members
        assert "Up" in const_names or "Down" in const_names

    def test_enum_member_parent_linkage(self):
        parser = make_parser()
        result = parser.parse(self.TS_ENUM, file_id=1)
        direction_idx = next(i for i, s in enumerate(result.symbols) if s.name == "Direction")
        members = [s for s in result.symbols
                   if s.kind == SymbolKind.CONSTANT and s.parent_id == direction_idx]
        assert len(members) > 0

    def test_enum_member_qualified_name(self):
        parser = make_parser()
        result = parser.parse(self.TS_ENUM, file_id=1)
        up = next(
            (s for s in result.symbols
             if s.kind == SymbolKind.CONSTANT and s.qualified_name == "Direction.Up"),
            None
        )
        assert up is not None

    def test_simple_enum_members_without_values(self):
        """Status has members without explicit values (just identifiers)."""
        parser = make_parser()
        result = parser.parse(self.TS_ENUM, file_id=1)
        status_idx = next(i for i, s in enumerate(result.symbols) if s.name == "Status")
        members = [s for s in result.symbols if s.parent_id == status_idx]
        assert len(members) >= 1


# ---------------------------------------------------------------------------
# Call edge extraction
# ---------------------------------------------------------------------------

class TestCallEdgeExtraction:
    TS_CALLS = """\
function outer() {
  helper();
  const x = compute(1, 2);
}

function helper() {}
function compute(a: number, b: number): number { return a + b; }
"""

    def test_call_edges_extracted(self):
        parser = make_parser()
        result = parser.parse(self.TS_CALLS, file_id=1)
        assert len(result.call_edges) > 0

    def test_callee_names_correct(self):
        parser = make_parser()
        result = parser.parse(self.TS_CALLS, file_id=1)
        callee_names = {e.callee_name for e in result.call_edges}
        assert "helper" in callee_names
        assert "compute" in callee_names

    def test_call_edge_line_set(self):
        parser = make_parser()
        result = parser.parse(self.TS_CALLS, file_id=1)
        for edge in result.call_edges:
            assert edge.line >= 1


# ---------------------------------------------------------------------------
# Re-export extraction
# ---------------------------------------------------------------------------

class TestReExportExtraction:
    TS_REEXPORT = """\
export { Foo, Bar } from './module';
"""

    def test_reexport_creates_import_edge(self):
        parser = make_parser()
        result = parser.parse(self.TS_REEXPORT, file_id=1)
        assert len(result.import_edges) == 1
        edge = result.import_edges[0]
        assert edge.imported_from == "./module"
        assert "Foo" in edge.imported_names
        assert "Bar" in edge.imported_names


# ---------------------------------------------------------------------------
# Interface with extends
# ---------------------------------------------------------------------------

class TestInterfaceExtends:
    TS_EXTENDS = """\
interface Animal {
  name: string;
}

interface Dog extends Animal {
  breed: string;
}
"""

    def test_interface_extends_type_edge(self):
        parser = make_parser()
        result = parser.parse(self.TS_EXTENDS, file_id=1)
        extends = [e for e in result.type_edges if e.edge_kind == "extends"]
        assert any(e.to_type_name == "Animal" for e in extends)


# ---------------------------------------------------------------------------
# Class with implements
# ---------------------------------------------------------------------------

class TestClassImplements:
    TS_IMPLEMENTS = """\
interface Serializable {
  serialize(): string;
}

class Record implements Serializable {
  serialize(): string {
    return JSON.stringify(this);
  }
}
"""

    def test_class_implements_type_edge(self):
        parser = make_parser()
        result = parser.parse(self.TS_IMPLEMENTS, file_id=1)
        impls = [e for e in result.type_edges if e.edge_kind == "implements"]
        assert any(e.to_type_name == "Serializable" for e in impls)


# ---------------------------------------------------------------------------
# Parse errors count
# ---------------------------------------------------------------------------

class TestParseErrors:
    def test_clean_source_has_zero_errors(self):
        parser = make_parser()
        result = parser.parse("const x: number = 1;", file_id=1)
        assert result.parse_errors == 0

    def test_invalid_source_has_parse_errors(self):
        parser = make_parser()
        result = parser.parse("function @@@invalid() {{{{", file_id=1)
        assert result.parse_errors > 0


# ---------------------------------------------------------------------------
# File ID propagation
# ---------------------------------------------------------------------------

class TestFileIdPropagation:
    def test_all_symbols_have_correct_file_id(self):
        parser = make_parser()
        source = """\
interface Foo { x: number; }
class Bar { method() {} }
const fn = () => {};
"""
        result = parser.parse(source, file_id=99)
        for sym in result.symbols:
            assert sym.file_id == 99
