"""Unit tests for the JavaScript parser (Phase 6a).

Exercises: class, function, CommonJS require, ES6 import,
arrow functions, module.exports, call graph extraction.
"""

from __future__ import annotations

from trelix.core.models import SymbolKind
from trelix.indexing.parser.extractors.javascript import JavaScriptParser

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_parser() -> JavaScriptParser:
    return JavaScriptParser()


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
    def test_javascript_language_name(self):
        assert make_parser().language_name == "javascript"


# ---------------------------------------------------------------------------
# ES6 import extraction
# ---------------------------------------------------------------------------


class TestES6ImportExtraction:
    JS_IMPORTS = """\
import React from 'react';
import { useState, useEffect } from 'react';
import * as path from 'path';
"""

    def test_default_import_source(self):
        parser = make_parser()
        result = parser.parse(self.JS_IMPORTS, file_id=1)
        sources = import_sources(result)
        assert "react" in sources

    def test_named_import_names(self):
        parser = make_parser()
        result = parser.parse(self.JS_IMPORTS, file_id=1)
        named = next((e for e in result.import_edges if "useState" in e.imported_names), None)
        assert named is not None
        assert "useEffect" in named.imported_names

    def test_namespace_import(self):
        parser = make_parser()
        result = parser.parse(self.JS_IMPORTS, file_id=1)
        assert "path" in import_sources(result)

    def test_file_id_on_import_edges(self):
        parser = make_parser()
        result = parser.parse(self.JS_IMPORTS, file_id=7)
        for edge in result.import_edges:
            assert edge.file_id == 7


# ---------------------------------------------------------------------------
# CommonJS require() extraction
# ---------------------------------------------------------------------------


class TestCommonJSRequire:
    JS_CJS = """\
const fs = require('fs');
const path = require('path');
const { EventEmitter } = require('events');
"""

    def test_require_from_var_decl(self):
        parser = make_parser()
        result = parser.parse(self.JS_CJS, file_id=1)
        sources = import_sources(result)
        assert "fs" in sources
        assert "path" in sources

    def test_destructured_require(self):
        parser = make_parser()
        result = parser.parse(self.JS_CJS, file_id=1)
        sources = import_sources(result)
        assert "events" in sources

    def test_standalone_require_expression(self):
        """Bare require() in expression_statement position."""
        src = "require('./config');"
        parser = make_parser()
        result = parser.parse(src, file_id=1)
        sources = import_sources(result)
        assert "./config" in sources


# ---------------------------------------------------------------------------
# Class extraction
# ---------------------------------------------------------------------------


class TestClassExtraction:
    JS_CLASS = """\
class Animal {
  constructor(name) {
    this.name = name;
  }

  speak() {
    return this.name + ' makes a noise.';
  }
}

export class Dog extends Animal {
  constructor(name) {
    super(name);
  }

  speak() {
    return this.name + ' barks.';
  }
}
"""

    def test_classes_extracted(self):
        parser = make_parser()
        result = parser.parse(self.JS_CLASS, file_id=1)
        classes = symbols_by_kind(result, SymbolKind.CLASS)
        names = [c.name for c in classes]
        assert "Animal" in names
        assert "Dog" in names

    def test_exported_class_is_public(self):
        parser = make_parser()
        result = parser.parse(self.JS_CLASS, file_id=1)
        dog = next(s for s in result.symbols if s.name == "Dog")
        assert dog.is_public is True

    def test_non_exported_class_is_not_public(self):
        parser = make_parser()
        result = parser.parse(self.JS_CLASS, file_id=1)
        animal = next(s for s in result.symbols if s.name == "Animal")
        assert animal.is_public is False

    def test_class_extends_type_edge(self):
        parser = make_parser()
        result = parser.parse(self.JS_CLASS, file_id=1)
        extends = [e for e in result.type_edges if e.edge_kind == "extends"]
        assert any(e.to_type_name == "Animal" for e in extends)

    def test_class_signature(self):
        parser = make_parser()
        result = parser.parse(self.JS_CLASS, file_id=1)
        dog = next(s for s in result.symbols if s.name == "Dog")
        assert "class Dog" in dog.signature

    def test_method_extracted(self):
        parser = make_parser()
        result = parser.parse(self.JS_CLASS, file_id=1)
        methods = symbols_by_kind(result, SymbolKind.METHOD)
        names = [m.name for m in methods]
        assert "speak" in names

    def test_method_parent_linkage(self):
        parser = make_parser()
        result = parser.parse(self.JS_CLASS, file_id=1)
        animal_idx = next(i for i, s in enumerate(result.symbols) if s.name == "Animal")
        animal_methods = [
            s for s in result.symbols if s.kind == SymbolKind.METHOD and s.parent_id == animal_idx
        ]
        assert any(m.name == "speak" for m in animal_methods)

    def test_method_qualified_name(self):
        parser = make_parser()
        result = parser.parse(self.JS_CLASS, file_id=1)
        speak = next(
            (
                s
                for s in result.symbols
                if s.kind == SymbolKind.METHOD
                and s.name == "speak"
                and s.qualified_name == "Animal.speak"
            ),
            None,
        )
        assert speak is not None

    def test_constructor_extracted(self):
        parser = make_parser()
        result = parser.parse(self.JS_CLASS, file_id=1)
        methods = symbols_by_kind(result, SymbolKind.METHOD)
        assert any(m.name == "constructor" for m in methods)

    def test_line_numbers_set(self):
        parser = make_parser()
        result = parser.parse(self.JS_CLASS, file_id=1)
        animal = next(s for s in result.symbols if s.name == "Animal")
        assert animal.line_start >= 1
        assert animal.line_end > animal.line_start


# ---------------------------------------------------------------------------
# Function extraction
# ---------------------------------------------------------------------------


class TestFunctionExtraction:
    JS_FUNCS = """\
export function greet(name) {
  return 'Hello, ' + name + '!';
}

function helper(x) {
  return x * 2;
}

function* generator() {
  yield 1;
  yield 2;
}
"""

    def test_exported_function_extracted(self):
        parser = make_parser()
        result = parser.parse(self.JS_FUNCS, file_id=1)
        funcs = symbols_by_kind(result, SymbolKind.FUNCTION)
        assert any(f.name == "greet" for f in funcs)

    def test_non_exported_function_extracted(self):
        parser = make_parser()
        result = parser.parse(self.JS_FUNCS, file_id=1)
        funcs = symbols_by_kind(result, SymbolKind.FUNCTION)
        assert any(f.name == "helper" for f in funcs)

    def test_exported_function_is_public(self):
        parser = make_parser()
        result = parser.parse(self.JS_FUNCS, file_id=1)
        greet = next(s for s in result.symbols if s.name == "greet")
        assert greet.is_public is True

    def test_function_signature_format(self):
        parser = make_parser()
        result = parser.parse(self.JS_FUNCS, file_id=1)
        greet = next(s for s in result.symbols if s.name == "greet")
        assert "function greet" in greet.signature

    def test_generator_function_extracted(self):
        parser = make_parser()
        result = parser.parse(self.JS_FUNCS, file_id=1)
        funcs = symbols_by_kind(result, SymbolKind.FUNCTION)
        assert any(f.name == "generator" for f in funcs)

    def test_generator_function_signature(self):
        parser = make_parser()
        result = parser.parse(self.JS_FUNCS, file_id=1)
        gen = next(s for s in result.symbols if s.name == "generator")
        assert "function*" in gen.signature or "generator" in gen.signature


# ---------------------------------------------------------------------------
# Arrow function extraction
# ---------------------------------------------------------------------------


class TestArrowFunctionExtraction:
    JS_ARROW = """\
export const double = (x) => x * 2;

const greet = (name) => {
  return 'Hello, ' + name + '!';
};

const MAX_SIZE = 100;
export const DEFAULT_TIMEOUT = 5000;
"""

    def test_exported_arrow_function_extracted(self):
        parser = make_parser()
        result = parser.parse(self.JS_ARROW, file_id=1)
        funcs = symbols_by_kind(result, SymbolKind.FUNCTION)
        assert any(f.name == "double" for f in funcs)

    def test_non_exported_arrow_function_extracted(self):
        parser = make_parser()
        result = parser.parse(self.JS_ARROW, file_id=1)
        funcs = symbols_by_kind(result, SymbolKind.FUNCTION)
        assert any(f.name == "greet" for f in funcs)

    def test_arrow_function_is_public_when_exported(self):
        parser = make_parser()
        result = parser.parse(self.JS_ARROW, file_id=1)
        double = next(s for s in result.symbols if s.name == "double")
        assert double.is_public is True

    def test_arrow_signature_format(self):
        parser = make_parser()
        result = parser.parse(self.JS_ARROW, file_id=1)
        double = next(s for s in result.symbols if s.name == "double")
        assert "const double" in double.signature
        assert "=>" in double.signature

    def test_allcaps_constant_extracted(self):
        parser = make_parser()
        result = parser.parse(self.JS_ARROW, file_id=1)
        consts = symbols_by_kind(result, SymbolKind.CONSTANT)
        assert any(c.name == "MAX_SIZE" for c in consts)

    def test_exported_constant_extracted(self):
        parser = make_parser()
        result = parser.parse(self.JS_ARROW, file_id=1)
        consts = symbols_by_kind(result, SymbolKind.CONSTANT)
        assert any(c.name == "DEFAULT_TIMEOUT" for c in consts)


# ---------------------------------------------------------------------------
# module.exports extraction
# ---------------------------------------------------------------------------


class TestModuleExports:
    JS_CJS_EXPORTS = """\
const value = 42;
module.exports = { value, greet };
"""

    def test_module_exports_extracted_as_constant(self):
        parser = make_parser()
        result = parser.parse(self.JS_CJS_EXPORTS, file_id=1)
        consts = symbols_by_kind(result, SymbolKind.CONSTANT)
        assert any(c.name == "module.exports" for c in consts)

    def test_module_exports_is_public(self):
        parser = make_parser()
        result = parser.parse(self.JS_CJS_EXPORTS, file_id=1)
        exports_sym = next(s for s in result.symbols if s.name == "module.exports")
        assert exports_sym.is_public is True

    def test_module_exports_signature(self):
        parser = make_parser()
        result = parser.parse(self.JS_CJS_EXPORTS, file_id=1)
        exports_sym = next(s for s in result.symbols if s.name == "module.exports")
        assert "module.exports" in exports_sym.signature


# ---------------------------------------------------------------------------
# Re-export extraction
# ---------------------------------------------------------------------------


class TestReExportExtraction:
    JS_REEXPORT = """\
export { foo, bar } from './utils';
"""

    def test_reexport_creates_import_edge(self):
        parser = make_parser()
        result = parser.parse(self.JS_REEXPORT, file_id=1)
        assert len(result.import_edges) == 1
        edge = result.import_edges[0]
        assert edge.imported_from == "./utils"
        assert "foo" in edge.imported_names
        assert "bar" in edge.imported_names


# ---------------------------------------------------------------------------
# Call edge extraction
# ---------------------------------------------------------------------------


class TestCallEdgeExtraction:
    JS_CALLS = """\
function outer() {
  helper();
  const x = compute(1, 2);
  console.log(x);
}

function helper() {}
function compute(a, b) { return a + b; }
"""

    def test_call_edges_extracted(self):
        parser = make_parser()
        result = parser.parse(self.JS_CALLS, file_id=1)
        assert len(result.call_edges) > 0

    def test_callee_names_correct(self):
        parser = make_parser()
        result = parser.parse(self.JS_CALLS, file_id=1)
        callee_names = {e.callee_name for e in result.call_edges}
        assert "helper" in callee_names
        assert "compute" in callee_names

    def test_method_call_extracted(self):
        """console.log() should appear as callee 'log'."""
        parser = make_parser()
        result = parser.parse(self.JS_CALLS, file_id=1)
        callee_names = {e.callee_name for e in result.call_edges}
        assert "log" in callee_names

    def test_call_edge_line_set(self):
        parser = make_parser()
        result = parser.parse(self.JS_CALLS, file_id=1)
        for edge in result.call_edges:
            assert edge.line >= 1


# ---------------------------------------------------------------------------
# Class field extraction
# ---------------------------------------------------------------------------


class TestClassFieldExtraction:
    JS_FIELDS = """\
class Component {
  state = { count: 0 };
  title = 'My Component';
  handleClick = () => { this.setState({}); };
}
"""

    def test_public_field_extracted(self):
        parser = make_parser()
        result = parser.parse(self.JS_FIELDS, file_id=1)
        vars_ = symbols_by_kind(result, SymbolKind.VARIABLE)
        names = [v.name for v in vars_]
        assert "state" in names or "title" in names

    def test_field_parent_linkage(self):
        parser = make_parser()
        result = parser.parse(self.JS_FIELDS, file_id=1)
        comp_idx = next(i for i, s in enumerate(result.symbols) if s.name == "Component")
        child_vars = [
            s for s in result.symbols if s.kind == SymbolKind.VARIABLE and s.parent_id == comp_idx
        ]
        assert len(child_vars) > 0


# ---------------------------------------------------------------------------
# Parse errors
# ---------------------------------------------------------------------------


class TestParseErrors:
    def test_clean_source_has_zero_errors(self):
        parser = make_parser()
        result = parser.parse("const x = 1;", file_id=1)
        assert result.parse_errors == 0

    def test_invalid_source_has_parse_errors(self):
        parser = make_parser()
        result = parser.parse("function @@@bad() {{{{", file_id=1)
        assert result.parse_errors > 0


# ---------------------------------------------------------------------------
# File ID propagation
# ---------------------------------------------------------------------------


class TestFileIdPropagation:
    def test_all_symbols_have_correct_file_id(self):
        parser = make_parser()
        source = """\
import React from 'react';
class Foo { bar() {} }
const fn = () => {};
"""
        result = parser.parse(source, file_id=55)
        for sym in result.symbols:
            assert sym.file_id == 55


# ---------------------------------------------------------------------------
# Module docstring
# ---------------------------------------------------------------------------


class TestModuleDocstring:
    def test_leading_jsdoc_extracted_as_module_symbol(self):
        src = """\
/**
 * Utility module for string operations.
 * @module string-utils
 */

export function trim(s) { return s.trim(); }
"""
        parser = make_parser()
        result = parser.parse(src, file_id=1)
        modules = symbols_by_kind(result, SymbolKind.MODULE)
        assert any(m.name == "<module>" for m in modules)

    def test_non_leading_comment_not_module_symbol(self):
        src = """\
export function foo() {}
/**
 * This JSDoc is not at the top.
 */
export function bar() {}
"""
        parser = make_parser()
        result = parser.parse(src, file_id=1)
        modules = symbols_by_kind(result, SymbolKind.MODULE)
        assert len(modules) == 0
