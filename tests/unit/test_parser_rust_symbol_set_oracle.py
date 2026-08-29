"""Mutation-verified exact-set oracles for the Rust extractor.

Same shape as tests/unit/test_parser_ts_symbol_set_oracle.py and
tests/unit/test_parser_module_symbol_set_oracle.py: every expectation is an
EXPLICIT LITERAL table hand-read off the fixture in this file. Nothing is
imported from ``trelix.indexing.parser.extractors.rust`` except the parser class,
and no expected value is built by iterating the parser's own output — that is the
failure mode that let whole ``symbols.append`` calls be deleted with a green
suite.

Set comparisons are two-way (``==`` on sets plus an explicit ``len`` check), so a
SPURIOUS extra symbol fails exactly as loudly as a missing one.

LINE SPANS: ``line_start``/``line_end`` are 1-based and inclusive, so every row
is checkable by eye against the numbered fixture comment. An audit found a
``line_end`` off-by-one surviving in 9 of 15 extractors, so there is also a
dedicated test (``test_rust_no_symbol_claims_a_line_past_end_of_file``) that
pins the module symbol's span against the fixture's real line count.

FIXTURES CARRY NO TRAILING NEWLINE. With one, tree-sitter's root ``end_point``
lands on the position after it and the MODULE symbol claims line N+1 of an
N-line file; pinning that number would lock in an artifact instead of testing
``line_end=root.end_point[0] + 1``. Without it, that ``+ 1`` is the correct
formula and deleting it is caught.

``Symbol.body`` is NOT asserted to be verbatim source except at the two
documented truncation boundaries (const/static at 500 chars, macro_rules at
800) — those widths are the behaviour under test.

CURRENT-BUT-WRONG BEHAVIOUR PINNED HERE (each marked at its assertion):
  * a ``///`` doc comment is LOST when an ``#[attr]`` sits between it and the
    item, because ``_get_preceding_comment`` walks ``prev_named_sibling`` and
    stops at the non-comment ``attribute_item``  (row ``Boxy``, and
    ``test_rust_doc_comment_attachment_is_exact``);
  * the blank-line gap guard in ``_get_preceding_comment`` is itself off by one
    for Rust: a ``///`` comment separated from its item by ONE blank line is
    still attached, because tree-sitter-rust's ``line_comment`` node text
    includes the trailing newline so ``end_point`` already names the next line.
    Java's identical guard is correct because ``*/`` closes on its own line;
  * an associated type inside ``impl Trait for Type`` is emitted with
    ``parent_id=None`` (row ``Out``), unlike the same declaration inside the
    trait itself (row ``Draw::Out``);
  * ``fn`` items inside a ``mod`` block get no module prefix in their
    ``qualified_name`` (row ``nested``);
  * trait method signatures report ``is_public=False`` even though trait items
    are public in Rust (rows ``Draw::draw``, ``Draw::helper``).

MUTANTS REPORTED, NOT TESTED:
  * ``RustParser.__init__``: ``self._ts_language = load_language("rust")``
    -> ``self._ts_language = None``. EQUIVALENT. ``_ts_language`` is assigned and
    never read; the live path is ``self._parser = make_parser("rust")``. Killing
    it would require poking a private attribute production code never touches,
    which would assert a config value and lock in dead code.
  * the ``if not name_node: return`` guards in ``_handle_struct``,
    ``_handle_enum``, ``_handle_trait`` etc. UNREACHABLE through ``parse()``:
    measured, ``pub struct {}`` (and the enum/trait equivalents) is recovered by
    tree-sitter-rust as ``ERROR`` + ``expression_statement``, never as a
    ``struct_item`` with a missing name field, so ``_walk_top_level`` never calls
    the handler. Deleting a guard therefore changes nothing observable and no
    test here claims to kill it.
"""

from __future__ import annotations

import pytest

from trelix.indexing.parser.extractors.rust import RustParser

# ---------------------------------------------------------------------------
# Fixture 1 — one file covering every symbol kind the extractor emits.
# The line number of each source line is in the comment column below so the
# expected table is hand-checkable.
# ---------------------------------------------------------------------------
#  1 //! Widget crate.
#  2 //! Second line.
#  3
#  4 use std::collections::HashMap;
#  5 use crate::util::{Alpha, Beta};
#  6 extern crate serde;
#  7
#  8 pub const MAX: usize = 10;
#  9 static NAMES: &[&str] = &["a"];
# 10 static mut COUNTER: u32 = 0;
# 11
# 12 /// A box.
# 13 #[derive(Debug)]
# 14 pub struct Boxy {
# 15     pub width: u32,
# 16     height: u32,
# 17 }
# 18
# 19 pub union Raw {
# 20     pub i: i32,
# 21 }
# 22
# 23 enum Mode {
# 24     Fast,
# 25     Slow(u8),
# 26 }
# 27
# 28 pub trait Draw: Clone {
# 29     type Out;
# 30     fn draw(&self) -> Self::Out;
# 31     fn helper(&self) -> u8 {
# 32         self.draw();
# 33         1
# 34     }
# 35 }
# 36
# 37 pub type Alias = HashMap<String, u32>;
# 38
# 39 impl Boxy {
# 40     pub const ZERO: u32 = 0;
# 41     pub fn new() -> Self {
# 42         Boxy { width: 0, height: 0 }
# 43     }
# 44     fn area(&self) -> u32 {
# 45         self.width
# 46     }
# 47 }
# 48
# 49 impl Draw for Boxy {
# 50     type Out = u32;
# 51     fn draw(&self) -> u32 {
# 52         helper_fn();
# 53         0
# 54     }
# 55 }
# 56
# 57 /// Returns a helper.
# 58 pub fn helper_fn() -> u32 {
# 59     let m = HashMap::new();
# 60     std::mem::swap(&mut 1, &mut 2);
# 61     m.len();
# 62     items.iter().count();
# 63     0
# 64 }
# 65
# 66 macro_rules! shout {
# 67     () => {};
# 68 }
# 69
# 70 mod inner {
# 71     pub fn nested() {}
# 72 }
KIND_SINK_RS = """\
//! Widget crate.
//! Second line.

use std::collections::HashMap;
use crate::util::{Alpha, Beta};
extern crate serde;

pub const MAX: usize = 10;
static NAMES: &[&str] = &["a"];
static mut COUNTER: u32 = 0;

/// A box.
#[derive(Debug)]
pub struct Boxy {
    pub width: u32,
    height: u32,
}

pub union Raw {
    pub i: i32,
}

enum Mode {
    Fast,
    Slow(u8),
}

pub trait Draw: Clone {
    type Out;
    fn draw(&self) -> Self::Out;
    fn helper(&self) -> u8 {
        self.draw();
        1
    }
}

pub type Alias = HashMap<String, u32>;

impl Boxy {
    pub const ZERO: u32 = 0;
    pub fn new() -> Self {
        Boxy { width: 0, height: 0 }
    }
    fn area(&self) -> u32 {
        self.width
    }
}

impl Draw for Boxy {
    type Out = u32;
    fn draw(&self) -> u32 {
        helper_fn();
        0
    }
}

/// Returns a helper.
pub fn helper_fn() -> u32 {
    let m = HashMap::new();
    std::mem::swap(&mut 1, &mut 2);
    m.len();
    items.iter().count();
    0
}

macro_rules! shout {
    () => {};
}

mod inner {
    pub fn nested() {}
}"""

# The fixture's real line count, written as a literal so a fixture edit that
# changes it is forced through the assertion in
# test_rust_no_symbol_claims_a_line_past_end_of_file.
KIND_SINK_RS_LINES = 72

# (qualified_name, kind name, line_start, line_end, is_public, parent qualified_name)
KIND_SINK_EXPECTED: set[tuple[str, str, int, int, bool, str | None]] = {
    # line_end 72, not 73: the fixture deliberately has NO trailing newline, so
    # `line_end=root.end_point[0] + 1` is the CORRECT inclusive last line and
    # deleting the `+ 1` is caught here rather than pinning an artifact.
    ("crate", "MODULE", 1, 72, True, None),
    ("MAX", "CONSTANT", 8, 8, True, None),
    ("NAMES", "CONSTANT", 9, 9, False, None),
    # line 10's `static mut COUNTER` is absent: _handle_static_item skips
    # mutable statics. A spurious COUNTER row fails on set equality.
    ("Boxy", "CLASS", 14, 17, True, None),
    ("width", "VARIABLE", 15, 15, True, "Boxy"),
    # line 16's private `height` is absent: _extract_struct_fields keeps only
    # fields carrying a visibility_modifier.
    ("Raw", "CLASS", 19, 21, True, None),
    ("i", "VARIABLE", 20, 20, True, "Raw"),
    ("Mode", "ENUM", 23, 26, False, None),
    # Variants are hardcoded is_public=True even though `enum Mode` is private —
    # so `is_public=True` -> `is_public=is_pub` in _extract_enum_variants fails.
    ("Mode::Fast", "CONSTANT", 24, 24, True, "Mode"),
    ("Mode::Slow", "CONSTANT", 25, 25, True, "Mode"),
    ("Draw", "INTERFACE", 28, 35, True, None),
    ("Draw::Out", "INTERFACE", 29, 29, True, "Draw"),
    # CURRENT-BUT-WRONG: trait items are public in Rust, but _handle_trait_fn
    # looks for a visibility_modifier that trait fns never carry.
    ("Draw::draw", "METHOD", 30, 30, False, "Draw"),
    ("Draw::helper", "METHOD", 31, 34, False, "Draw"),
    ("Alias", "INTERFACE", 37, 37, True, None),
    ("ZERO", "CONSTANT", 40, 40, True, "Boxy"),
    # `new()` has no self param -> FUNCTION, `area(&self)` does -> METHOD.
    ("Boxy::new", "FUNCTION", 41, 43, True, "Boxy"),
    ("Boxy::area", "METHOD", 44, 46, False, "Boxy"),
    # CURRENT-BUT-WRONG: `type Out = u32;` inside `impl Draw for Boxy` routes
    # through _handle_type_alias, which takes no parent, so this associated type
    # is emitted as a top-level INTERFACE named `Out` with no parent — compare
    # the `Draw::Out` row above.
    ("Out", "INTERFACE", 50, 50, False, None),
    ("Boxy::draw", "METHOD", 51, 54, False, "Boxy"),
    ("helper_fn", "FUNCTION", 58, 64, True, None),
    # macro_rules! is emitted as a FUNCTION and is not `pub`.
    ("shout", "FUNCTION", 66, 68, False, None),
    # CURRENT-BUT-WRONG: `mod inner` is flattened, so `nested` gets no
    # `inner::` prefix in its qualified_name.
    ("nested", "FUNCTION", 71, 71, True, None),
}

# (caller qualified_name, callee_name, line)
KIND_SINK_CALLS_EXPECTED: set[tuple[str, str, int]] = {
    ("Draw::helper", "draw", 32),
    ("Boxy::draw", "helper_fn", 52),
    # scoped_identifier `HashMap::new` -> last segment.
    ("helper_fn", "new", 59),
    # scoped_identifier `std::mem::swap` -> last segment.
    ("helper_fn", "swap", 60),
    ("helper_fn", "len", 61),
    # `items.iter().count()` yields BOTH links from the one line.
    ("helper_fn", "count", 62),
    ("helper_fn", "iter", 62),
}

# (imported_from, tuple(imported_names))
KIND_SINK_IMPORTS_EXPECTED: set[tuple[str, tuple[str, ...]]] = {
    ("std.collections", ("HashMap",)),
    # `use crate::util::{Alpha, Beta};` -> one edge per brace member. The
    # module string keeps the `::` separator (unlike the generic "split on
    # ::" fallback path above, which normalises to `.`) because it is read
    # straight off the `scoped_use_list`'s `path` field text.
    ("crate::util", ("Alpha",)),
    ("crate::util", ("Beta",)),
    # `extern crate serde;` -> no imported names.
    ("serde", ()),
}

# (from_symbol qualified_name, to_type_name, edge_kind)
KIND_SINK_TYPE_EDGES_EXPECTED: set[tuple[str, str, str]] = {
    ("Draw", "Clone", "extends"),
    ("Boxy", "Draw", "trait_impl"),
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


def test_rust_symbol_set_and_line_spans_are_exact():
    """Kills: any `line_end=node.end_point[0] + 1` -> `+ 0`/`+ 2` in
    _get_module_symbol, _handle_struct, _extract_struct_fields, _handle_union,
    _handle_macro_def, _handle_enum, _extract_enum_variants, _handle_trait,
    _handle_trait_fn, _handle_type_alias, _handle_impl_fn, _handle_function,
    _handle_const_item and _handle_static_item; deletion of any of those
    `symbols.append(...)` calls; the `if not is_pub: continue` skip in
    _extract_struct_fields; the `static mut` early return in _handle_static_item;
    `is_public=True` -> `is_public=is_pub` in _extract_enum_variants; the
    `_has_self_param` METHOD/FUNCTION split in _handle_impl_fn; `type_idx[name] =
    local_idx` deletion in _handle_struct (which orphans every impl method);
    `SymbolKind.CLASS` -> `STRUCT`-ish swaps for struct/union and
    `SymbolKind.INTERFACE` for trait/type-alias; the `mod_item` recursion in
    _walk_top_level.
    """
    # Preconditions: the fixture must keep the features that make this oracle
    # discriminating. If KIND_SINK_RS is edited to drop them the test would
    # silently stop testing them.
    assert "    height: u32," in KIND_SINK_RS, (
        "KIND_SINK_RS lost its private (skipped) struct field"
    )
    assert "static mut COUNTER" in KIND_SINK_RS, "KIND_SINK_RS lost its skipped `static mut`"
    assert "enum Mode {" in KIND_SINK_RS, (
        "KIND_SINK_RS's enum must stay NON-pub so its variants' hardcoded is_public=True shows"
    )
    assert "    pub fn new() -> Self {" in KIND_SINK_RS, (
        "KIND_SINK_RS lost its self-less associated fn (the FUNCTION vs METHOD split)"
    )
    assert "    fn area(&self) -> u32 {" in KIND_SINK_RS, (
        "KIND_SINK_RS lost its &self method (the FUNCTION vs METHOD split)"
    )
    assert "mod inner {" in KIND_SINK_RS, "KIND_SINK_RS lost its inline module"
    assert not KIND_SINK_RS.endswith("\n"), (
        "KIND_SINK_RS must NOT end with a newline, or the MODULE row's line_end "
        "becomes a line that does not exist in the fixture"
    )

    result = RustParser().parse(KIND_SINK_RS, file_id=7)

    assert _rows(result) == KIND_SINK_EXPECTED
    # Set equality cannot see duplicates; pin the count too.
    assert len(result.symbols) == 23


def test_rust_no_symbol_claims_a_line_past_end_of_file():
    """Kills: `line_end=root.end_point[0] + 1` -> `+ 2` in _get_module_symbol
    (the line_end off-by-one an audit found surviving in 9 of 15 extractors).

    The MODULE symbol spans the whole file, so it is the one row where a
    superfluous `+ 1` produces a line number the file does not have.
    """
    # Precondition: the literal below must still match the fixture, otherwise
    # this test degenerates into comparing two numbers that both drifted.
    assert len(KIND_SINK_RS.split("\n")) == KIND_SINK_RS_LINES, (
        "KIND_SINK_RS_LINES no longer matches KIND_SINK_RS"
    )

    result = RustParser().parse(KIND_SINK_RS, file_id=7)
    module_sym = next(s for s in result.symbols if s.kind.name == "MODULE")

    assert module_sym.line_start == 1
    assert module_sym.line_end == KIND_SINK_RS_LINES
    assert max(s.line_end for s in result.symbols) <= KIND_SINK_RS_LINES


def test_rust_call_import_and_type_edges_are_exact():
    """Kills: any `child.start_point[0] + 1` -> `+ 0`/`+ 2` in _walk_body's
    call_expression / scoped_identifier / field_expression / method_call_expression
    branches; deletion of the terminal `self._walk_body(child, ...)` recursion
    (which loses every nested call); the scoped_identifier last-segment lookup
    (`std::mem::swap` would vanish); the `field_expression` branch (`.iter()`
    would vanish); the `edge_kind="extends"` / `"trait_impl"` literals in
    _handle_trait and _handle_impl; the `if trait_name and parent_local_idx is
    not None` guard; and _handle_extern_crate returning None.
    """
    # Preconditions: each edge kind can only be observed because the fixture
    # still contains its trigger.
    assert "std::mem::swap" in KIND_SINK_RS, "KIND_SINK_RS lost its scoped_identifier call"
    assert "items.iter().count();" in KIND_SINK_RS, "KIND_SINK_RS lost its chained field-expr call"
    assert "extern crate serde;" in KIND_SINK_RS, "KIND_SINK_RS lost its extern crate declaration"
    assert "pub trait Draw: Clone {" in KIND_SINK_RS, "KIND_SINK_RS lost its supertrait bound"
    assert "impl Draw for Boxy {" in KIND_SINK_RS, "KIND_SINK_RS lost its trait impl block"

    result = RustParser().parse(KIND_SINK_RS, file_id=7)

    calls = {
        (result.symbols[e.caller_id].qualified_name, e.callee_name, e.line)
        for e in result.call_edges
    }
    assert calls == KIND_SINK_CALLS_EXPECTED
    assert len(result.call_edges) == 7

    imports = {(e.imported_from, tuple(e.imported_names)) for e in result.import_edges}
    assert imports == KIND_SINK_IMPORTS_EXPECTED
    assert len(result.import_edges) == 4

    type_edges = {
        (result.symbols[e.from_symbol_id].qualified_name, e.to_type_name, e.edge_kind)
        for e in result.type_edges
    }
    assert type_edges == KIND_SINK_TYPE_EDGES_EXPECTED
    assert len(result.type_edges) == 2

    assert result.parse_errors == 0


def test_rust_signatures_and_attributes_are_exact():
    """Kills: the `f"struct {name}{tp_str}"` / `f"union ..."` / `f"enum ..."` /
    `f"trait ..."` / `f"type {name}{tp_str} = {type_str}"` / `f"macro_rules! {name}"`
    signature templates; the `fn {prefix}{name}{tp_str}{params}{ret}` assembly in
    _fn_signature (including dropping the `-> ` return-type join or the
    `{type_name}::` prefix); and `result.insert(0, ...)` -> `result.append(...)`
    ordering in _get_rust_attributes.
    """
    result = RustParser().parse(KIND_SINK_RS, file_id=7)
    sigs = {s.qualified_name: s.signature for s in result.symbols}

    assert sigs["Boxy"] == "struct Boxy"
    assert sigs["Raw"] == "union Raw"
    assert sigs["Mode"] == "enum Mode"
    assert sigs["Draw"] == "trait Draw"
    assert sigs["Alias"] == "type Alias = HashMap<String, u32>"
    assert sigs["shout"] == "macro_rules! shout"
    assert sigs["helper_fn"] == "fn helper_fn() -> u32"
    assert sigs["Boxy::new"] == "fn Boxy::new() -> Self"
    assert sigs["Boxy::area"] == "fn Boxy::area(&self) -> u32"
    assert sigs["Draw::draw"] == "fn Draw::draw(&self) -> Self::Out"

    decorators = {s.qualified_name: s.decorators for s in result.symbols}
    assert decorators["Boxy"] == ["#[derive(Debug)]"]
    assert decorators["Raw"] == []


# ---------------------------------------------------------------------------
# Fixture 2 — doc comments: adjacency, the blank-line gap, and the
# attribute-in-between case.
# ---------------------------------------------------------------------------
#  1 /// Attached to a.
#  2 pub fn a() {}
#  3
#  4 /// One blank line before b.
#  5
#  6 pub fn b() {}
#  7
#  8 /// Two blank lines before c.
#  9
# 10
# 11 pub fn c() {}
# 12
# 13 /// Two
# 14 /// lines.
# 15 pub fn d() {}
DOC_RS = """\
/// Attached to a.
pub fn a() {}

/// One blank line before b.

pub fn b() {}

/// Two blank lines before c.


pub fn c() {}

/// Two
/// lines.
pub fn d() {}"""

# `#[attr]` sits between the doc comment and the item.
ATTR_DOC_RS = """\
/// Blocked by the attribute.
#[inline]
pub fn d() {}"""

# The module docstring path: two `//!` lines, then real code.
MODULE_DOC_RS = """\
//! Crate line one.
//! Crate line two.

pub fn e() {}"""

# No `//!` header at all -> no MODULE symbol.
NO_MODULE_DOC_RS = """\
// Just an ordinary comment.
pub fn f() {}"""


def test_rust_doc_comment_attachment_is_exact():
    """Kills: `if prev.end_point[0] + 1 < next_start_line: break` -> `<=` in
    _get_preceding_comment (which detaches the one-blank-line case pinned below)
    and the reverse `>` form (which detaches the adjacent case); `next_start_line
    = prev.start_point[0]` deletion (which drops the second line of a two-line
    doc comment); and the `^//[/!]?\\s?` substitution in _clean_comment.
    """
    # Preconditions: the three gap widths (0, 1 and 2 blank lines) must all still
    # be present, otherwise the gap rule below stops discriminating.
    assert "/// Attached to a.\npub fn a()" in DOC_RS, "DOC_RS lost its zero-gap doc comment"
    assert "/// One blank line before b.\n\npub fn b()" in DOC_RS, (
        "DOC_RS no longer has EXACTLY one blank line before `pub fn b`"
    )
    assert "/// Two blank lines before c.\n\n\npub fn c()" in DOC_RS, (
        "DOC_RS no longer has EXACTLY two blank lines before `pub fn c`"
    )
    assert "/// Two\n/// lines.\npub fn d()" in DOC_RS, "DOC_RS lost its two-line doc comment"
    assert "#[inline]\npub fn d()" in ATTR_DOC_RS, (
        "ATTR_DOC_RS lost the attribute that sits between the doc comment and the fn"
    )

    parser = RustParser()
    docs = {s.qualified_name: s.docstring for s in parser.parse(DOC_RS, file_id=1).symbols}
    assert docs["a"] == "Attached to a."
    # CURRENT-BUT-WRONG, pinned deliberately: a comment separated by ONE blank
    # line is still attached. tree-sitter-rust's line_comment node text includes
    # its trailing newline, so `prev.end_point[0]` already names the line AFTER
    # the comment and the `+ 1` in `prev.end_point[0] + 1 < next_start_line` is
    # one too many. The identical guard in the Java extractor is correct because
    # a Java `/** */` block comment ends at `*/`, on its own line. Fixing this
    # means dropping the `+ 1` for Rust; when that happens this row becomes None
    # and this assertion is the thing to update.
    assert docs["b"] == "One blank line before b."
    assert docs["c"] is None
    # tree-sitter-rust's line_comment node text INCLUDES its trailing newline, so
    # joining two of them with "\n" yields a blank line between them. Pinned as
    # observed behaviour, not endorsed: the semantic value is "Two\nlines.".
    assert docs["d"] == "Two\n\nlines."

    # CURRENT-BUT-WRONG: the `///` doc is silently lost because
    # _get_preceding_comment walks prev_named_sibling and the attribute_item is
    # not a comment, so the loop never reaches the comment. Rust puts attributes
    # AFTER doc comments idiomatically, so this loses docs on most real items.
    attr_result = parser.parse(ATTR_DOC_RS, file_id=1)
    attr_docs = {s.qualified_name: s.docstring for s in attr_result.symbols}
    assert attr_docs["d"] is None
    assert attr_result.symbols[0].decorators == ["#[inline]"]


def test_rust_module_symbol_requires_inner_doc_comment():
    """Kills: `if text.startswith("//!")` -> `startswith("//")` in
    _get_module_symbol (which would turn any leading `//` comment into a crate
    docstring), and `if not inner_doc_lines: return None` deletion.
    """
    # Preconditions: one fixture has `//!`, the sibling has only `//`.
    assert MODULE_DOC_RS.startswith("//! "), "MODULE_DOC_RS lost its inner doc comment"
    assert NO_MODULE_DOC_RS.startswith("// "), (
        "NO_MODULE_DOC_RS must start with a plain `//` comment, not `//!`"
    )

    parser = RustParser()

    with_doc = parser.parse(MODULE_DOC_RS, file_id=1)
    assert {s.qualified_name for s in with_doc.symbols} == {"crate", "e"}
    crate = next(s for s in with_doc.symbols if s.qualified_name == "crate")
    # Same trailing-newline-in-node-text quirk as above.
    assert crate.docstring == "Crate line one.\n\nCrate line two."
    assert crate.body == crate.docstring
    assert crate.signature == "crate"

    without = parser.parse(NO_MODULE_DOC_RS, file_id=1)
    assert {s.qualified_name for s in without.symbols} == {"f"}


# ---------------------------------------------------------------------------
# Fixture 3 — the extractor's hard caps and truncation widths.
# ---------------------------------------------------------------------------

WIDE_STRUCT_RS = (
    "pub struct Wide {\n" + "".join(f"    pub f{i:02d}: u32,\n" for i in range(1, 32)) + "}"
)

BIG_ENUM_RS = "pub enum Big {\n" + "".join(f"    V{i:02d},\n" for i in range(1, 52)) + "}"

# Five private fields FIRST, then thirty pub ones: the private fields must not
# eat into the MAX_STRUCT_FIELDS budget.
PRIVATE_FIRST_RS = (
    "pub struct Mixed {\n"
    + "".join(f"    p{i:02d}: u32,\n" for i in range(1, 6))
    + "".join(f"    pub q{i:02d}: u32,\n" for i in range(1, 31))
    + "}"
)

# fmt: off
# 30 field names — written out, NOT generated from the parser's output.
FIRST_30_FIELDS = {
    "f01", "f02", "f03", "f04", "f05", "f06", "f07", "f08", "f09", "f10",
    "f11", "f12", "f13", "f14", "f15", "f16", "f17", "f18", "f19", "f20",
    "f21", "f22", "f23", "f24", "f25", "f26", "f27", "f28", "f29", "f30",
}
# 50 variant names — written out.
FIRST_50_VARIANTS = {
    "Big::V01", "Big::V02", "Big::V03", "Big::V04", "Big::V05",
    "Big::V06", "Big::V07", "Big::V08", "Big::V09", "Big::V10",
    "Big::V11", "Big::V12", "Big::V13", "Big::V14", "Big::V15",
    "Big::V16", "Big::V17", "Big::V18", "Big::V19", "Big::V20",
    "Big::V21", "Big::V22", "Big::V23", "Big::V24", "Big::V25",
    "Big::V26", "Big::V27", "Big::V28", "Big::V29", "Big::V30",
    "Big::V31", "Big::V32", "Big::V33", "Big::V34", "Big::V35",
    "Big::V36", "Big::V37", "Big::V38", "Big::V39", "Big::V40",
    "Big::V41", "Big::V42", "Big::V43", "Big::V44", "Big::V45",
    "Big::V46", "Big::V47", "Big::V48", "Big::V49", "Big::V50",
}
# All 30 pub fields of PRIVATE_FIRST_RS — written out.
ALL_30_Q_FIELDS = {
    "q01", "q02", "q03", "q04", "q05", "q06", "q07", "q08", "q09", "q10",
    "q11", "q12", "q13", "q14", "q15", "q16", "q17", "q18", "q19", "q20",
    "q21", "q22", "q23", "q24", "q25", "q26", "q27", "q28", "q29", "q30",
}
# fmt: on


def test_rust_field_and_variant_caps_are_exact():
    """Kills: MAX_STRUCT_FIELDS 30 -> any other value; `field_count >=
    MAX_STRUCT_FIELDS` -> `>`; MAX_ENUM_VARIANTS 50 -> any other value;
    `variant_count >= MAX_ENUM_VARIANTS` -> `>`; and moving `field_count += 1`
    above the `if not is_pub: continue` skip (which would let private fields
    consume the budget).
    """
    # Preconditions: the caps can only be observed if the fixtures overshoot them.
    assert WIDE_STRUCT_RS.count(": u32,") == 31, "WIDE_STRUCT_RS no longer declares 31 pub fields"
    assert BIG_ENUM_RS.count("    V") == 51, "BIG_ENUM_RS no longer declares 51 variants"
    assert PRIVATE_FIRST_RS.count("    p0") == 5, (
        "PRIVATE_FIRST_RS lost its 5 leading private fields"
    )
    assert PRIVATE_FIRST_RS.count("    pub q") == 30, (
        "PRIVATE_FIRST_RS no longer declares exactly 30 pub fields after the private ones"
    )

    parser = RustParser()

    wide = parser.parse(WIDE_STRUCT_RS, file_id=1)
    field_names = {s.name for s in wide.symbols if s.kind.name == "VARIABLE"}
    assert field_names == FIRST_30_FIELDS
    assert len([s for s in wide.symbols if s.kind.name == "VARIABLE"]) == 30

    big = parser.parse(BIG_ENUM_RS, file_id=1)
    variant_names = {s.qualified_name for s in big.symbols if s.kind.name == "CONSTANT"}
    assert variant_names == FIRST_50_VARIANTS
    assert len([s for s in big.symbols if s.kind.name == "CONSTANT"]) == 50

    mixed = parser.parse(PRIVATE_FIRST_RS, file_id=1)
    mixed_fields = {s.name for s in mixed.symbols if s.kind.name == "VARIABLE"}
    assert mixed_fields == ALL_30_Q_FIELDS
    assert len([s for s in mixed.symbols if s.kind.name == "VARIABLE"]) == 30


_CONST_HEAD = 'pub const BIG: &str = "'
_CONST_TAIL = '";'
_MACRO_HEAD = 'macro_rules! m { () => { let s = "'
_MACRO_TAIL = '"; }; }'


def _const_of_length(n: int) -> str:
    """A whole-file `pub const` item whose node text is exactly n characters."""
    text = _CONST_HEAD + "x" * (n - len(_CONST_HEAD) - len(_CONST_TAIL)) + _CONST_TAIL
    assert len(text) == n, "_const_of_length arithmetic is wrong"
    return text


def _static_of_length(n: int) -> str:
    """A whole-file `pub static` item whose node text is exactly n characters.

    Needed because the existing static fixtures were DERIVED from the const ones by
    substituting "pub static " for "pub const " -- and that string is one character
    longer, so `static_exact` came out at 501 and NEITHER static fixture sat on the
    500 boundary. Adversarial review measured the consequence: mutating only
    `_handle_static_item`'s `> 500` to `>= 500` SURVIVED all 12 tests, because the kill
    attributed to that mutation came entirely from the const half of a coupled
    two-site edit. This builds a static that is exactly n characters so the boundary
    is actually reachable.
    """
    head = _CONST_HEAD.replace("pub const ", "pub static ")
    text = head + "x" * (n - len(head) - len(_CONST_TAIL)) + _CONST_TAIL
    assert len(text) == n, "_static_of_length arithmetic is wrong"
    return text


def _macro_of_length(n: int) -> str:
    """A whole-file `macro_rules!` item whose node text is exactly n characters."""
    text = _MACRO_HEAD + "y" * (n - len(_MACRO_HEAD) - len(_MACRO_TAIL)) + _MACRO_TAIL
    assert len(text) == n, "_macro_of_length arithmetic is wrong"
    return text


def test_rust_body_truncation_widths_are_exact():
    """Kills: `if len(body) > 500` -> `>= 500` or `> 5` in _handle_const_item and
    _handle_static_item; `body[:500]` -> another width; the `+ "..."` suffix;
    `signature=body.split("\\n")[0][:200]` -> another width; and `if len(body) >
    800` / `body[:800]` in _handle_macro_def.

    These widths ARE the documented contract, which is why body is asserted
    verbatim here and nowhere else in this file.
    """
    parser = RustParser()

    # Each source below is exactly one item with no trailing newline, so the
    # tree-sitter node text equals the source string.
    const_exact = _const_of_length(500)
    const_over = _const_of_length(501)
    exact_sym = next(s for s in parser.parse(const_exact, file_id=1).symbols if s.name == "BIG")
    over_sym = next(s for s in parser.parse(const_over, file_id=1).symbols if s.name == "BIG")
    assert exact_sym.body == const_exact
    assert over_sym.body == const_over[:500] + "..."
    assert len(over_sym.body) == 503
    assert over_sym.signature == const_over[:200]
    assert len(over_sym.signature) == 200

    static_exact = "pub static " + const_exact[len("pub const ") :]
    static_over = "pub static " + const_over[len("pub const ") :]
    assert len(static_exact) == 501, "static fixture arithmetic is wrong"
    s_exact = next(s for s in parser.parse(static_exact, file_id=1).symbols if s.name == "BIG")
    s_over = next(s for s in parser.parse(static_over, file_id=1).symbols if s.name == "BIG")
    # 501 chars: `static_exact` is already one OVER the boundary, so it truncates.
    assert s_exact.body == static_exact[:500] + "..."
    assert s_over.body == static_over[:500] + "..."

    # THE BOUNDARY ITSELF, for the static branch. Without this the static-only mutant
    # (`_handle_static_item` `> 500` -> `>= 500`) survives: both fixtures above are 501
    # chars, so neither distinguishes `>` from `>=`. A genuinely 500-char static must come
    # back VERBATIM, which is false the moment the comparison becomes inclusive.
    static_500 = _static_of_length(500)
    assert len(static_500) == 500, "fixture must sit exactly on the boundary or this is vacuous"
    s_500 = next(s for s in parser.parse(static_500, file_id=1).symbols if s.name == "BIG")
    assert s_500.body == static_500, (
        "a 500-char static must not be truncated; `>= 500` in _handle_static_item would "
        "truncate it and this is the only assertion that sees it"
    )

    macro_exact = _macro_of_length(800)
    macro_over = _macro_of_length(801)
    m_exact = next(s for s in parser.parse(macro_exact, file_id=1).symbols if s.name == "m")
    m_over = next(s for s in parser.parse(macro_over, file_id=1).symbols if s.name == "m")
    assert m_exact.body == macro_exact
    assert m_over.body == macro_over[:800] + "..."
    assert len(m_over.body) == 803


LONG_ATTR_RS = '#[doc = "' + "z" * 400 + '"]\npub fn attributed() {}'


def test_rust_attribute_text_is_truncated_at_200_chars():
    """Kills: `text[:200] if len(text) <= 200 else text[:200] + "..."` -> a
    different width or a dropped ellipsis in _get_rust_attributes.
    """
    # Precondition: the attribute must actually exceed 200 characters.
    assert len(LONG_ATTR_RS.split("\n")[0]) > 200, "LONG_ATTR_RS's attribute is no longer over 200"

    result = RustParser().parse(LONG_ATTR_RS, file_id=1)
    sym = next(s for s in result.symbols if s.name == "attributed")
    assert len(sym.decorators) == 1
    assert sym.decorators[0] == LONG_ATTR_RS.split("\n")[0][:200] + "..."
    assert len(sym.decorators[0]) == 203


@pytest.mark.parametrize(
    "source",
    [
        "pub struct {}",
        "pub enum {}",
        "pub trait {}",
    ],
)
def test_rust_malformed_source_is_reported_via_parse_errors(source):
    """Kills: `count = 1 if node.type == "ERROR" else 0` -> `count = 0` in
    _count_errors; deletion of the `for child in node.children` recursion in
    _count_errors (the ERROR node here is NOT the root); and
    `parse_errors=self._count_errors(root)` -> `parse_errors=0` in parse().

    This deliberately does NOT claim to test the `if not name_node: return`
    guards. Measured: tree-sitter-rust's error recovery turns each source above
    into an `ERROR` + `expression_statement` pair, so _walk_top_level never sees
    a `struct_item`/`enum_item`/`trait_item` and the guards are never reached.
    See the module docstring for that finding.
    """
    result = RustParser().parse(source, file_id=1)
    assert result.symbols == []
    assert result.parse_errors == 1
