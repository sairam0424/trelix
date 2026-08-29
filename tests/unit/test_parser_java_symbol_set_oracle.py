"""Mutation-verified exact-set oracles for the Java extractor.

Same shape as tests/unit/test_parser_ts_symbol_set_oracle.py and
tests/unit/test_parser_module_symbol_set_oracle.py: every expectation is an
EXPLICIT LITERAL table hand-read off the fixture in this file. Nothing is
imported from ``trelix.indexing.parser.extractors.java`` except the parser class,
and no expected value is built by iterating the parser's own output — the
pre-existing tests/unit/test_parser_java.py asks "is my symbol in the list?",
which cannot see a deleted ``symbols.append``.

Set comparisons are two-way (``==`` on sets plus an explicit ``len`` check), so a
SPURIOUS extra symbol fails exactly as loudly as a missing one.

LINE SPANS: ``line_start``/``line_end`` are 1-based and inclusive, so every row
is checkable by eye against the numbered fixture comment. An audit found a
``line_end`` off-by-one surviving in 9 of 15 extractors, so there is also a
dedicated test (``test_java_no_symbol_claims_a_line_past_end_of_file``) that
pins the outermost declaration's span against the fixture's real line count.
The Java extractor emits NO synthetic MODULE symbol, so the widest span in the
file belongs to the top-level class.

FIXTURES CARRY NO TRAILING NEWLINE, matching the sibling oracles: with one,
tree-sitter's root ``end_point`` reports a line that does not exist, and pinning
that number would lock in an artifact.

``Symbol.body`` is NOT asserted to be verbatim source except at the documented
500-character truncation boundary for fields and interface constants — that
width is the behaviour under test.

CURRENT-BUT-WRONG BEHAVIOUR PINNED HERE (each marked at its assertion):
  * ``import java.util.*;`` is recorded as ``imported_from="java"`` /
    ``imported_names=["util"]``. The ``*`` is a sibling of the
    ``scoped_identifier``, not part of it, so the ``if parts[-1] != "*"`` branch
    is unreachable;
  * a NESTED type keeps a bare ``qualified_name`` (``Nested``, not
    ``Svc.Nested``), so two same-named inner classes in one file collide;

MUTANTS REPORTED, NOT TESTED:
  * ``JavaParser.__init__``: ``self._ts_language = load_language("java")``
    -> ``self._ts_language = None``. EQUIVALENT. ``_ts_language`` is assigned and
    never read; the live path is ``self._parser = make_parser("java")``. Killing
    it would require poking a private attribute production code never touches,
    which would assert a config value and lock in dead code.
  * the ``if not name_node: return`` guards in ``_handle_class``,
    ``_handle_interface``, ``_handle_enum`` etc. UNREACHABLE through ``parse()``:
    measured, ``class {}`` (and the interface/enum equivalents) is recovered by
    tree-sitter-java as ``ERROR`` + ``block``, never as a ``class_declaration``
    with a missing identifier, so ``_walk_top_level`` never calls the handler.
    Deleting a guard therefore changes nothing observable and no test here claims
    to kill it.
"""

from __future__ import annotations

import pytest

from trelix.core.models import TypeEdge
from trelix.indexing.parser.extractors.java import JavaParser

# ---------------------------------------------------------------------------
# Fixture 1 — one file covering every symbol kind the extractor emits.
# The line number of each source line is in the comment column below so the
# expected table is hand-checkable.
# ---------------------------------------------------------------------------
#  1 package com.example;
#  2
#  3 import java.util.List;
#  4 import java.util.*;
#  5
#  6 /** Service docs. */
#  7 @Service
#  8 public abstract class Svc extends Base implements Runnable, AutoCloseable {
#  9     public static final String VERSION = "1.0";
# 10     @Autowired
# 11     private Repo repo;
# 12     private int hidden;
# 13     protected String label;
# 14
# 15     public Svc(Repo repo) {
# 16         this.repo = repo;
# 17         register(new Helper());
# 18     }
# 19
# 20     public int run(int n) {
# 21         return repo.find(n);
# 22     }
# 23
# 24     private void quiet() {}
# 25
# 26     static class Nested {
# 27         public int v;
# 28     }
# 29 }
# 30
# 31 interface Shape {
# 32     int SIDES = 3;
# 33     int area();
# 34     default int twice() {
# 35         return area() * 2;
# 36     }
# 37 }
# 38
# 39 enum Status implements Runnable {
# 40     OK,
# 41     BAD;
# 42
# 43     public void run() {}
# 44 }
# 45
# 46 record Point(int x, int y) implements Cloneable {
# 47     public int sum() {
# 48         return x + y;
# 49     }
# 50 }
# 51
# 52 @interface Marker {}
KIND_SINK_JAVA = """\
package com.example;

import java.util.List;
import java.util.*;

/** Service docs. */
@Service
public abstract class Svc extends Base implements Runnable, AutoCloseable {
    public static final String VERSION = "1.0";
    @Autowired
    private Repo repo;
    private int hidden;
    protected String label;

    public Svc(Repo repo) {
        this.repo = repo;
        register(new Helper());
    }

    public int run(int n) {
        return repo.find(n);
    }

    private void quiet() {}

    static class Nested {
        public int v;
    }
}

interface Shape {
    int SIDES = 3;
    int area();
    default int twice() {
        return area() * 2;
    }
}

enum Status implements Runnable {
    OK,
    BAD;

    public void run() {}
}

record Point(int x, int y) implements Cloneable {
    public int sum() {
        return x + y;
    }
}

@interface Marker {}"""

# The fixture's real line count, written as a literal so a fixture edit that
# changes it is forced through the assertion in
# test_java_no_symbol_claims_a_line_past_end_of_file.
KIND_SINK_JAVA_LINES = 52

# (qualified_name, kind name, line_start, line_end, is_public, parent qualified_name)
KIND_SINK_EXPECTED: set[tuple[str, str, int, int, bool, str | None]] = {
    # line_start 7, not 8: the `modifiers` node holds @Service, so the class node
    # starts on the annotation line.
    ("Svc", "CLASS", 7, 29, True, None),
    ("Svc.VERSION", "CONSTANT", 9, 9, True, "Svc"),
    # 10-11: the field_declaration spans its @Autowired modifier line too.
    # is_public False: annotated-but-private fields are kept, not promoted.
    ("Svc.repo", "VARIABLE", 10, 11, False, "Svc"),
    # line 12's unannotated private `hidden` is absent: _handle_field_decl skips
    # non-static-final, non-annotated, non-public/protected fields.
    ("Svc.label", "VARIABLE", 13, 13, True, "Svc"),
    # A constructor is a METHOD whose name repeats the class name.
    ("Svc.Svc", "METHOD", 15, 18, True, "Svc"),
    ("Svc.run", "METHOD", 20, 22, True, "Svc"),
    ("Svc.quiet", "METHOD", 24, 24, False, "Svc"),
    # CURRENT-BUT-WRONG: a nested type is NOT qualified by its outer class.
    ("Nested", "CLASS", 26, 28, False, None),
    ("Nested.v", "VARIABLE", 27, 27, True, "Nested"),
    ("Shape", "INTERFACE", 31, 37, False, None),
    # Interface constants are hardcoded is_public=True even though `interface
    # Shape` is package-private, so `is_public=True` -> `is_public=is_public`
    # in _handle_interface_constant fails here.
    ("Shape.SIDES", "CONSTANT", 32, 32, True, "Shape"),
    ("Shape.area", "METHOD", 33, 33, False, "Shape"),
    ("Shape.twice", "METHOD", 34, 36, False, "Shape"),
    ("Status", "ENUM", 39, 44, False, None),
    # Enum constants INHERIT the enum's is_public (False here), unlike interface
    # constants above — so `is_public=is_public` -> `is_public=True` fails.
    ("Status.OK", "CONSTANT", 40, 40, False, "Status"),
    ("Status.BAD", "CONSTANT", 41, 41, False, "Status"),
    ("Status.run", "METHOD", 43, 43, True, "Status"),
    ("Point", "CLASS", 46, 50, False, None),
    # Record components are VARIABLE symbols, always public, parented to the
    # record, and spanned to the record header's own line.
    ("Point.x", "VARIABLE", 46, 46, True, "Point"),
    ("Point.y", "VARIABLE", 46, 46, True, "Point"),
    ("Point.sum", "METHOD", 47, 49, True, "Point"),
    ("Marker", "INTERFACE", 52, 52, False, None),
}

# (caller qualified_name, callee_name, line)
KIND_SINK_CALLS_EXPECTED: set[tuple[str, str, int]] = {
    ("Svc.Svc", "register", 17),
    # `new Helper()` is recorded as a call to the type being instantiated.
    ("Svc.Svc", "Helper", 17),
    ("Svc.run", "find", 21),
    ("Shape.twice", "area", 35),
}

# (imported_from, tuple(imported_names))
KIND_SINK_IMPORTS_EXPECTED: set[tuple[str, tuple[str, ...]]] = {
    ("java.util", ("List",)),
    # CURRENT-BUT-WRONG: the wildcard is lost — see the module docstring.
    ("java", ("util",)),
}

# (from_symbol qualified_name, to_type_name, edge_kind)
KIND_SINK_TYPE_EDGES_EXPECTED: set[tuple[str, str, str]] = {
    ("Svc", "Base", "extends"),
    ("Svc", "Runnable", "implements"),
    ("Svc", "AutoCloseable", "implements"),
    ("Status", "Runnable", "implements"),
    ("Point", "Cloneable", "implements"),
}


def _rows(result) -> set[tuple[str, str, int, int, bool, str | None]]:
    """Project ParseResult.symbols onto the tuple shape of the expected table."""
    out = set()
    for sym in result.symbols:
        parent = result.symbols[sym.parent_id].qualified_name if sym.parent_id is not None else None
        out.add(
            (
                sym.qualified_name,
                sym.kind.name,
                sym.line_start,
                sym.line_end,
                sym.is_public,
                parent,
            )
        )
    return out


def test_java_symbol_set_and_line_spans_are_exact():
    """Kills: any `line_end=node.end_point[0] + 1` -> `+ 0`/`+ 2` in
    _handle_class, _handle_interface, _handle_interface_constant, _handle_record,
    _handle_annotation_type, _handle_method, _handle_constructor,
    _handle_field_decl and _handle_enum; deletion of any of those
    `symbols.append(...)` calls; the `if not is_static_final and not
    has_annotations and not is_public: return` skip in _handle_field_decl;
    `is_static_final = "static" in mods_text and "final" in mods_text` -> `or`
    (which would make `protected String label` a CONSTANT); `is_public = "public"
    in mods_text or "protected" in mods_text` -> dropping the `protected` arm;
    `is_public=is_public` -> `is_public=True` for enum constants; `is_public=True`
    -> the interface's own visibility for interface constants;
    `SymbolKind.CLASS`/`INTERFACE`/`ENUM`/`METHOD`/`CONSTANT`/`VARIABLE` swaps;
    and the nested-declaration recursion inside _handle_class.
    """
    # Preconditions: the fixture must keep the features that make this oracle
    # discriminating. If KIND_SINK_JAVA is edited to drop them the test would
    # silently stop testing them.
    assert "    private int hidden;" in KIND_SINK_JAVA, (
        "KIND_SINK_JAVA lost its unannotated private (skipped) field"
    )
    assert "    protected String label;" in KIND_SINK_JAVA, (
        "KIND_SINK_JAVA lost its `protected` field (the only test of that arm)"
    )
    assert "    @Autowired\n    private Repo repo;" in KIND_SINK_JAVA, (
        "KIND_SINK_JAVA lost its annotated private field"
    )
    assert "interface Shape {" in KIND_SINK_JAVA, (
        "KIND_SINK_JAVA's interface must stay package-private so its constants' "
        "hardcoded is_public=True shows"
    )
    assert "enum Status implements Runnable {" in KIND_SINK_JAVA, (
        "KIND_SINK_JAVA's enum must stay package-private so its constants' "
        "inherited is_public=False shows"
    )
    assert "    static class Nested {" in KIND_SINK_JAVA, "KIND_SINK_JAVA lost its nested class"
    assert not KIND_SINK_JAVA.endswith("\n"), (
        "KIND_SINK_JAVA must NOT end with a newline, to match the sibling oracles"
    )

    result = JavaParser().parse(KIND_SINK_JAVA, file_id=8)

    assert _rows(result) == KIND_SINK_EXPECTED
    # Set equality cannot see duplicates; pin the count too.
    assert len(result.symbols) == 22


def test_java_no_symbol_claims_a_line_past_end_of_file():
    """Kills: `line_end=node.end_point[0] + 1` -> `+ 2` in _handle_class,
    _handle_annotation_type and _handle_record (the line_end off-by-one an audit
    found surviving in 9 of 15 extractors).

    `@interface Marker {}` is the LAST line of the fixture, so a superfluous
    `+ 1` there names a line the file does not have.
    """
    # Precondition: the literal below must still match the fixture, otherwise
    # this test degenerates into comparing two numbers that both drifted.
    assert len(KIND_SINK_JAVA.split("\n")) == KIND_SINK_JAVA_LINES, (
        "KIND_SINK_JAVA_LINES no longer matches KIND_SINK_JAVA"
    )
    assert KIND_SINK_JAVA.split("\n")[-1] == "@interface Marker {}", (
        "KIND_SINK_JAVA's last line must be the @interface, which is what pins the boundary"
    )

    result = JavaParser().parse(KIND_SINK_JAVA, file_id=8)
    marker = next(s for s in result.symbols if s.qualified_name == "Marker")

    assert marker.line_start == KIND_SINK_JAVA_LINES
    assert marker.line_end == KIND_SINK_JAVA_LINES
    assert max(s.line_end for s in result.symbols) <= KIND_SINK_JAVA_LINES


def test_java_call_import_and_type_edges_are_exact():
    """Kills: any `child.start_point[0] + 1` -> `+ 0`/`+ 2` in _walk_body's
    method_invocation and object_creation_expression branches; deletion of the
    object_creation_expression branch (`new Helper()` would vanish); deletion of
    the `self._walk_body(child, ...)` recursion in the else arm (every call
    nested inside a `return` statement would vanish); adding
    `"method_invocation"` to the do-not-recurse tuple; the
    `.split("<")[0].strip()` generic strip; the `type_list` arm of
    _extract_type_list_edges (all three of Svc's edges live there); and the
    `edge_kind` literals "extends"/"implements".
    """
    # Preconditions: each edge kind can only be observed because the fixture
    # still contains its trigger.
    assert "register(new Helper());" in KIND_SINK_JAVA, (
        "KIND_SINK_JAVA lost its object_creation_expression nested in a call argument"
    )
    assert "        return repo.find(n);" in KIND_SINK_JAVA, (
        "KIND_SINK_JAVA lost its call nested inside a return statement"
    )
    assert "extends Base implements Runnable, AutoCloseable" in KIND_SINK_JAVA, (
        "KIND_SINK_JAVA lost its extends/implements clause"
    )
    assert "import java.util.*;" in KIND_SINK_JAVA, "KIND_SINK_JAVA lost its wildcard import"

    result = JavaParser().parse(KIND_SINK_JAVA, file_id=8)

    calls = {
        (result.symbols[e.caller_id].qualified_name, e.callee_name, e.line)
        for e in result.call_edges
    }
    assert calls == KIND_SINK_CALLS_EXPECTED
    assert len(result.call_edges) == 4

    imports = {(e.imported_from, tuple(e.imported_names)) for e in result.import_edges}
    assert imports == KIND_SINK_IMPORTS_EXPECTED
    assert len(result.import_edges) == 2

    type_edges = {
        (result.symbols[e.from_symbol_id].qualified_name, e.to_type_name, e.edge_kind)
        for e in result.type_edges
    }
    assert type_edges == KIND_SINK_TYPE_EDGES_EXPECTED
    assert len(result.type_edges) == 5

    assert result.parse_errors == 0


RECORD_ONLY_JAVA = "public record Pt(int x, int y) implements Cloneable {}"


def test_java_record_implements_edge_is_present():
    """J1/J1b (record components) and J2 (record implements edge) are both
    fixed: a record's components are extracted and its `implements` clause
    produces a real TypeEdge via `_extract_type_list_edges`, exactly like
    `_handle_class` already does for ordinary classes.
    """
    # Precondition: the record must actually declare components and an
    # implements clause, or "everything was extracted" would be trivially true.
    assert "(int x, int y)" in RECORD_ONLY_JAVA, "RECORD_ONLY_JAVA lost its two components"
    assert "implements Cloneable" in RECORD_ONLY_JAVA, "RECORD_ONLY_JAVA lost its implements clause"

    result = JavaParser().parse(RECORD_ONLY_JAVA, file_id=1)

    assert {s.qualified_name for s in result.symbols} == {"Pt", "Pt.x", "Pt.y"}
    assert len(result.symbols) == 3
    assert [s.kind.name for s in result.symbols] == ["CLASS", "VARIABLE", "VARIABLE"]
    assert result.type_edges == [
        TypeEdge(
            from_symbol_id=0, to_type_name="Cloneable", edge_kind="implements", to_symbol_id=None
        )
    ]
    assert result.symbols[0].signature == "public record Pt(int x, int y)"


def test_java_signatures_are_exact():
    """Kills: the `f"interface {name}"` / `f"@interface {name}"` / `f"enum {name}"`
    / `f"{name}.{cname}"` signature templates; the
    `f"{mods}{ret}{class_name}.{name}{params}"` assembly in _method_signature
    (including dropping the return type or the `class_name.` prefix); the
    `f"{class_name}{params}"` constructor template; `_get_return_type`'s
    type-node set; and `decl_text.split("\\n")[0][:200].strip()` for fields,
    where `decl_text` starts after the field's modifiers node.
    """
    result = JavaParser().parse(KIND_SINK_JAVA, file_id=8)
    sigs = {s.qualified_name: s.signature for s in result.symbols}

    assert sigs["Svc"] == (
        "@Service\npublic abstract class Svc extends Base implements Runnable, AutoCloseable"
    )
    assert sigs["Shape"] == "interface Shape"
    assert sigs["Status"] == "enum Status"
    assert sigs["Marker"] == "@interface Marker"
    assert sigs["Status.OK"] == "Status.OK"
    assert sigs["Svc.Svc"] == "Svc(Repo repo)"
    assert sigs["Svc.run"] == "public int Svc.run(int n)"
    assert sigs["Svc.quiet"] == "private void Svc.quiet()"
    assert sigs["Shape.twice"] == "default int Shape.twice()"
    # The signature starts after the field's modifiers node, so `public static
    # final` / `protected` / the `@Autowired` annotation are excluded.
    assert sigs["Svc.VERSION"] == 'String VERSION = "1.0";'
    assert sigs["Svc.label"] == "String label;"
    assert sigs["Svc.repo"] == "Repo repo;"

    decorators = {s.qualified_name: s.decorators for s in result.symbols}
    assert decorators["Svc"] == ["@Service"]
    assert decorators["Svc.repo"] == ["@Autowired"]
    assert decorators["Svc.label"] == []


# ---------------------------------------------------------------------------
# Fixture 2 — doc comments: adjacency and the blank-line gap.
# ---------------------------------------------------------------------------
#  1 /** Attached to A. */
#  2 class A {}
#  3
#  4 /** Detached from B. */
#  5
#  6 class B {}
#  7
#  8 // Line comment doc.
#  9 class C {}
DOC_JAVA = """\
/** Attached to A. */
class A {}

/** Detached from B. */

class B {}

// Line comment doc.
class C {}"""

MULTILINE_DOC_JAVA = """\
/**
 * First line.
 * Second line.
 */
class D {}"""


def test_java_doc_comment_attachment_is_exact():
    """Kills: `if prev.end_point[0] + 1 < next_start_line: break` -> `<=` in
    _get_preceding_comment (which drops EVERY adjacent Javadoc) and the reverse
    `>` form (which would attach a comment separated by a blank line); and the
    `^/\\*+\\s*`, `\\s*\\*+/$`, `^\\s*\\*\\s?` and `^//\\s?` substitutions in
    _clean_comment.
    """
    # Preconditions: exactly one blank line must separate the detached comment
    # from `class B`, and the multi-line Javadoc must keep its ` * ` prefixes.
    assert "/** Attached to A. */\nclass A {}" in DOC_JAVA, "DOC_JAVA lost its adjacent Javadoc"
    assert "/** Detached from B. */\n\nclass B {}" in DOC_JAVA, (
        "DOC_JAVA no longer has EXACTLY one blank line before `class B`"
    )
    assert "// Line comment doc.\nclass C {}" in DOC_JAVA, "DOC_JAVA lost its `//` comment case"
    assert "\n * First line.\n * Second line.\n" in MULTILINE_DOC_JAVA, (
        "MULTILINE_DOC_JAVA lost its ` * ` continuation prefixes"
    )

    parser = JavaParser()
    docs = {s.qualified_name: s.docstring for s in parser.parse(DOC_JAVA, file_id=1).symbols}
    assert docs["A"] == "Attached to A."
    assert docs["B"] is None
    assert docs["C"] == "Line comment doc."

    multi = parser.parse(MULTILINE_DOC_JAVA, file_id=1)
    assert next(s for s in multi.symbols if s.name == "D").docstring == "First line.\nSecond line."


# ---------------------------------------------------------------------------
# Fixture 3 — the MAX_FIELDS cap, including its multi-declarator overshoot.
# ---------------------------------------------------------------------------

WIDE_CLASS_JAVA = (
    "class Wide {\n" + "".join(f"    public int f{i:02d};\n" for i in range(1, 32)) + "}"
)

# 28 single-declarator fields, then ONE declaration holding three declarators,
# then one more. The cap is checked BEFORE a declaration is processed and
# advanced by the NUMBER OF SYMBOLS the declaration produced, so the multi
# pushes the counter from 28 to 31 and `zz` is refused.
MULTI_DECLARATOR_JAVA = (
    "class Multi {\n"
    + "".join(f"    public int f{i:02d};\n" for i in range(1, 29))
    + "    public int aa, bb, cc;\n"
    + "    public int zz;\n"
    + "}"
)

WIDE_INTERFACE_JAVA = (
    "interface WideI {\n" + "".join(f"    int c{i:02d} = {i};\n" for i in range(1, 32)) + "}"
)

# fmt: off
# 30 field names — written out, NOT generated from the parser's output.
FIRST_30_FIELDS = {
    "f01", "f02", "f03", "f04", "f05", "f06", "f07", "f08", "f09", "f10",
    "f11", "f12", "f13", "f14", "f15", "f16", "f17", "f18", "f19", "f20",
    "f21", "f22", "f23", "f24", "f25", "f26", "f27", "f28", "f29", "f30",
}
# 28 singles plus the three declarators of the one multi declaration — 31 names.
MULTI_EXPECTED_FIELDS = {
    "f01", "f02", "f03", "f04", "f05", "f06", "f07", "f08", "f09", "f10",
    "f11", "f12", "f13", "f14", "f15", "f16", "f17", "f18", "f19", "f20",
    "f21", "f22", "f23", "f24", "f25", "f26", "f27", "f28",
    "aa", "bb", "cc",
}
# 30 interface constant names — written out.
FIRST_30_CONSTANTS = {
    "c01", "c02", "c03", "c04", "c05", "c06", "c07", "c08", "c09", "c10",
    "c11", "c12", "c13", "c14", "c15", "c16", "c17", "c18", "c19", "c20",
    "c21", "c22", "c23", "c24", "c25", "c26", "c27", "c28", "c29", "c30",
}
# fmt: on


def test_java_field_caps_are_exact():
    """Kills: MAX_FIELDS 30 -> any other value; `field_count < self.MAX_FIELDS`
    -> `<=`; and `field_count += len(symbols) - before` -> `field_count += 1`
    (which the MULTI_DECLARATOR_JAVA case is built specifically to catch: with
    `+= 1` the counter reaches only 29 after the three-declarator line, so `zz`
    would be admitted and a 32nd field appear).
    """
    # Preconditions: the cap can only be observed if the fixtures overshoot it,
    # and the multi-declarator case only discriminates at exactly 28 + 3 + 1.
    assert WIDE_CLASS_JAVA.count("public int f") == 31, (
        "WIDE_CLASS_JAVA no longer declares 31 fields"
    )
    assert MULTI_DECLARATOR_JAVA.count("public int f") == 28, (
        "MULTI_DECLARATOR_JAVA no longer has exactly 28 single-declarator fields, "
        "so it no longer discriminates `+= len(symbols) - before` from `+= 1`"
    )
    assert "    public int aa, bb, cc;\n    public int zz;\n" in MULTI_DECLARATOR_JAVA, (
        "MULTI_DECLARATOR_JAVA lost its three-declarator line or its trailing `zz`"
    )
    assert WIDE_INTERFACE_JAVA.count("    int c") == 31, (
        "WIDE_INTERFACE_JAVA no longer declares 31 constants"
    )

    parser = JavaParser()

    wide = parser.parse(WIDE_CLASS_JAVA, file_id=1)
    field_names = {s.name for s in wide.symbols if s.kind.name == "VARIABLE"}
    assert field_names == FIRST_30_FIELDS
    assert len([s for s in wide.symbols if s.kind.name == "VARIABLE"]) == 30

    multi = parser.parse(MULTI_DECLARATOR_JAVA, file_id=1)
    multi_names = {s.name for s in multi.symbols if s.kind.name == "VARIABLE"}
    assert multi_names == MULTI_EXPECTED_FIELDS
    assert len([s for s in multi.symbols if s.kind.name == "VARIABLE"]) == 31
    assert "zz" not in multi_names

    wide_i = parser.parse(WIDE_INTERFACE_JAVA, file_id=1)
    const_names = {s.name for s in wide_i.symbols if s.kind.name == "CONSTANT"}
    assert const_names == FIRST_30_CONSTANTS
    assert len([s for s in wide_i.symbols if s.kind.name == "CONSTANT"]) == 30


_FIELD_MODS = "public static final "
_FIELD_HEAD = 'public static final String V = "'
_CONSTANT_HEAD = 'String V = "'
_FIELD_TAIL = '";'

assert _FIELD_HEAD == _FIELD_MODS + _CONSTANT_HEAD, "_FIELD_MODS/_CONSTANT_HEAD arithmetic is wrong"


def _field_of_length(n: int) -> str:
    """A field declaration whose tree-sitter node text is exactly n characters."""
    text = _FIELD_HEAD + "x" * (n - len(_FIELD_HEAD) - len(_FIELD_TAIL)) + _FIELD_TAIL
    assert len(text) == n, "_field_of_length arithmetic is wrong"
    return text


def _constant_of_length(n: int) -> str:
    """An interface constant_declaration whose node text is exactly n characters."""
    text = _CONSTANT_HEAD + "x" * (n - len(_CONSTANT_HEAD) - len(_FIELD_TAIL)) + _FIELD_TAIL
    assert len(text) == n, "_constant_of_length arithmetic is wrong"
    return text


def test_java_field_body_truncation_width_is_exact():
    """Kills: `if len(body) > 500` -> `>= 500` or `> 5` in _handle_field_decl and
    _handle_interface_constant; `body[:500]` -> another width; the `+ "..."`
    suffix; and `signature=decl_text.split("\\n")[0][:200].strip()` -> another
    width or a dropped `.strip()`.

    500 is the documented contract, which is why body is asserted verbatim here
    and nowhere else in this file.
    """
    parser = JavaParser()

    exact = _field_of_length(500)
    over = _field_of_length(501)
    r_exact = parser.parse("class C {\n    " + exact + "\n}", file_id=1)
    r_over = parser.parse("class C {\n    " + over + "\n}", file_id=1)
    exact_sym = next(s for s in r_exact.symbols if s.name == "V")
    over_sym = next(s for s in r_over.symbols if s.name == "V")
    assert exact_sym.body == exact
    assert over_sym.body == over[:500] + "..."
    assert len(over_sym.body) == 503
    # One line, so the signature is the truncated declaration (the body minus
    # the `public static final` modifiers) capped at 200.
    assert over_sym.signature == over[len(_FIELD_MODS) :][:200]
    assert len(over_sym.signature) == 200

    # Same boundary in the interface-constant path, which has its own copy of
    # the `if len(body) > 500` branch.
    iface_exact = _constant_of_length(500)
    iface_over = _constant_of_length(501)
    i_exact = parser.parse("interface I {\n    " + iface_exact + "\n}", file_id=1)
    i_over = parser.parse("interface I {\n    " + iface_over + "\n}", file_id=1)
    assert next(s for s in i_exact.symbols if s.name == "V").body == iface_exact
    over_const = next(s for s in i_over.symbols if s.name == "V")
    assert over_const.body == iface_over[:500] + "..."
    assert len(over_const.body) == 503
    assert over_const.signature == iface_over[:200]


@pytest.mark.parametrize(
    "source",
    [
        "class {}",
        "interface {}",
        "enum {}",
    ],
)
def test_java_malformed_source_is_reported_via_parse_errors(source):
    """Kills: `count = 1 if node.type == "ERROR" else 0` -> `count = 0` in
    _count_errors; deletion of the `for child in node.children` recursion in
    _count_errors (the ERROR node here is NOT the root); and
    `parse_errors=self._count_errors(root)` -> `parse_errors=0` in parse().

    This deliberately does NOT claim to test the `if not name_node: return`
    guards. Measured: tree-sitter-java's error recovery turns each source above
    into an `ERROR` + `block` pair, so _walk_top_level never sees a
    `class_declaration`/`interface_declaration`/`enum_declaration` and the guards
    are never reached. See the module docstring for that finding.
    """
    result = JavaParser().parse(source, file_id=1)
    assert result.symbols == []
    assert result.parse_errors == 1


# ---------------------------------------------------------------------------
# Fixture 4 — the static/final split that decides CONSTANT vs VARIABLE.
# A field needs BOTH modifiers to become a CONSTANT.
# ---------------------------------------------------------------------------
#  1 class Kinds {
#  2     public static final int BOTH = 1;
#  3     public final int onlyFinal = 2;
#  4     public static int onlyStatic = 3;
#  5 }
STATIC_FINAL_JAVA = """\
class Kinds {
    public static final int BOTH = 1;
    public final int onlyFinal = 2;
    public static int onlyStatic = 3;
}"""

# (name, kind name)
STATIC_FINAL_EXPECTED: set[tuple[str, str]] = {
    ("Kinds", "CLASS"),
    ("BOTH", "CONSTANT"),
    ("onlyFinal", "VARIABLE"),
    ("onlyStatic", "VARIABLE"),
}


def test_java_constant_requires_both_static_and_final():
    """Kills: `is_static_final = "static" in mods_text and "final" in mods_text`
    -> `or` (which promotes both single-modifier fields to CONSTANT) and -> just
    one of the two conjuncts.

    The main fixture cannot catch this: every field there has BOTH modifiers or
    NEITHER, so `and` and `or` agree on it. This fixture exists solely to supply
    the two mixed cases.
    """
    # Preconditions: exactly one field with both, one with only final, one with
    # only static — the whole point of this fixture.
    assert "public static final int BOTH" in STATIC_FINAL_JAVA, (
        "STATIC_FINAL_JAVA lost its static+final field"
    )
    assert "public final int onlyFinal" in STATIC_FINAL_JAVA, (
        "STATIC_FINAL_JAVA lost its final-but-not-static field, so `and` -> `or` survives"
    )
    assert "public static int onlyStatic" in STATIC_FINAL_JAVA, (
        "STATIC_FINAL_JAVA lost its static-but-not-final field, so `and` -> `or` survives"
    )

    result = JavaParser().parse(STATIC_FINAL_JAVA, file_id=1)
    assert {(s.name, s.kind.name) for s in result.symbols} == STATIC_FINAL_EXPECTED
    assert len(result.symbols) == 4
