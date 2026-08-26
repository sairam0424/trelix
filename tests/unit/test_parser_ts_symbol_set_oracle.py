"""Mutation-verified oracles for the TypeScript/TSX extractor.

Every expectation below is an EXPLICIT LITERAL table written from the fixture
source in this file.  Nothing is imported from
``trelix.indexing.parser.extractors.typescript`` except the parser class itself,
and no expectation is built by iterating the parser's own output — that is the
failure mode that let 14 ``walker.EXTENSION_MAP`` entries be deleted with a
green suite.

Set comparisons are two-way (``==`` on sets plus an explicit length check), so a
SPURIOUS extra symbol fails just as loudly as a MISSING one.

Bodies are deliberately not asserted to be verbatim source except where the
module's documented contract is a truncation boundary (class fields at 300
chars, type aliases at 500) — those two are the behaviour under test.
"""

from __future__ import annotations

from trelix.indexing.parser.extractors.typescript import TypeScriptParser

# ---------------------------------------------------------------------------
# Fixture 1 — one file covering every symbol kind the extractor emits.
# Line numbers in the expected table below are 1-based and hand-counted from
# this literal.  The blank first-column comment marks the line number.
# ---------------------------------------------------------------------------
#  1 /**
#  2  * Widget module.
#  3  */
#  4 export interface Shape {
#  5   id: number;
#  6   area(): number;
#  7 }
#  8
#  9 interface Hidden {
# 10   flag: boolean;
# 11 }
# 12
# 13 export enum Mode {
# 14   Fast = 'FAST',
# 15   Slow,
# 16 }
# 17
# 18 export type Id = string;
# 19
# 20 export class Box implements Shape {
# 21   id: number = 0;
# 22   protected size: number = 1;
# 23   private _cache: string = '';
# 24
# 25   private hide(): void {
# 26     return;
# 27   }
# 28
# 29   public area(): number {
# 30     return this.id;
# 31   }
# 32 }
# 33
# 34 export const build = (n: number): Box => new Box();
# 35
# 36 function local(): void {
# 37   return;
# 38 }
# 39
# 40 const helper = 1;
# 41 const LIMIT = 5;
KIND_SINK_TS = """\
/**
 * Widget module.
 */
export interface Shape {
  id: number;
  area(): number;
}

interface Hidden {
  flag: boolean;
}

export enum Mode {
  Fast = 'FAST',
  Slow,
}

export type Id = string;

export class Box implements Shape {
  id: number = 0;
  protected size: number = 1;
  private _cache: string = '';

  private hide(): void {
    return;
  }

  public area(): number {
    return this.id;
  }
}

export const build = (n: number): Box => new Box();

function local(): void {
  return;
}

const helper = 1;
const LIMIT = 5;"""

# (qualified_name, kind name, line_start, line_end, is_public, parent qualified_name)
KIND_SINK_EXPECTED: set[tuple[str, str, int, int, bool, str | None]] = {
    # line_end 41, not 42, and the fixture above deliberately has NO trailing newline.
    #
    # With a trailing newline tree-sitter's root `end_point` lands on the position AFTER
    # it, so the MODULE symbol claims line 42 of a 41-line file — a line that does not
    # exist, while every other row here is an inclusive 1-based span. Pinning 42 would
    # silently lock that artifact in as expected behaviour and block the fix.
    #
    # The sibling oracle tests/unit/test_parser_module_symbol_set_oracle.py already
    # established this convention for exactly this reason, and says so in its docstring.
    # Verified: every non-module row is byte-identical with and without the newline, and
    # without it the `+ 1` in `line_end=root.end_point[0] + 1` becomes the CORRECT formula
    # — so deleting that `+ 1` is now itself caught, i.e. this row tests the arithmetic
    # instead of pinning an artifact.
    ("<module>", "MODULE", 1, 41, True, None),
    ("Shape", "INTERFACE", 4, 7, True, None),
    ("Shape.id", "VARIABLE", 5, 5, True, "Shape"),
    ("Shape.area", "FUNCTION", 6, 6, True, "Shape"),
    ("Hidden", "INTERFACE", 9, 11, False, None),
    ("Hidden.flag", "VARIABLE", 10, 10, False, "Hidden"),
    ("Mode", "ENUM", 13, 16, True, None),
    ("Mode.Fast", "CONSTANT", 14, 14, True, "Mode"),
    ("Mode.Slow", "CONSTANT", 15, 15, True, "Mode"),
    ("Id", "INTERFACE", 18, 18, True, None),
    ("Box", "CLASS", 20, 32, True, None),
    ("Box.id", "VARIABLE", 21, 21, True, "Box"),
    ("Box.size", "VARIABLE", 22, 22, False, "Box"),
    ("Box.hide", "METHOD", 25, 27, False, "Box"),
    ("Box.area", "METHOD", 29, 31, True, "Box"),
    ("build", "FUNCTION", 34, 34, True, None),
    ("local", "FUNCTION", 36, 38, False, None),
    ("LIMIT", "CONSTANT", 41, 41, True, None),
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


def test_typescript_symbol_set_and_line_spans_are_exact():
    """Kills: `line_end=node.end_point[0] + 1` -> `+ 0` (or `+ 2`) in _handle_class,
    _handle_method, _handle_function, _handle_interface, _handle_enum, the arrow-function
    branch of _handle_var_decl, _extract_interface_members and the enum-member loop;
    `_is_constant_name` -> `return True`; deletion of the leading-underscore field skip
    in _handle_class_field; `is_public = self._txt(vis_node, src) == "public"` -> `!=`
    in _handle_method; `is_public=exported` -> `is_public=True` in _handle_interface;
    swapping SymbolKind.FUNCTION/VARIABLE for interface method_signature vs
    property_signature.
    """
    # Preconditions: the fixture must keep the four features that make this
    # oracle discriminating. If KIND_SINK_TS is edited to drop them the test
    # would silently stop testing them.
    assert "  private _cache" in KIND_SINK_TS, "KIND_SINK_TS lost its skipped _underscore field"
    assert "const helper = 1;" in KIND_SINK_TS, "KIND_SINK_TS lost its non-constant local const"
    assert "  protected size" in KIND_SINK_TS, "KIND_SINK_TS lost its non-public class field"
    assert "  private hide" in KIND_SINK_TS, "KIND_SINK_TS lost its non-public method"
    assert "  public area" in KIND_SINK_TS, "KIND_SINK_TS lost its explicitly public method"

    result = TypeScriptParser(tsx=False).parse(KIND_SINK_TS, file_id=3)

    assert _rows(result) == KIND_SINK_EXPECTED
    # Set equality cannot see duplicates; pin the count too.
    assert len(result.symbols) == 18


# ---------------------------------------------------------------------------
# Fixture 2 — the extractor's four hard limits.
# ---------------------------------------------------------------------------

WIDE_CLASS_TS = (
    "class Wide {\n" + "".join(f"  f{i:02d}: number = {i};\n" for i in range(1, 32)) + "}\n"
)

BIG_INTERFACE_TS = (
    "interface Big {\n" + "".join(f"  m{i:02d}: number;\n" for i in range(1, 22)) + "}\n"
)

# fmt: off
# 30 field names — written out, NOT generated from the parser's output.
FIRST_30_FIELDS = {
    "f01", "f02", "f03", "f04", "f05", "f06", "f07", "f08", "f09", "f10",
    "f11", "f12", "f13", "f14", "f15", "f16", "f17", "f18", "f19", "f20",
    "f21", "f22", "f23", "f24", "f25", "f26", "f27", "f28", "f29", "f30"
}
# 20 interface member names — written out.
FIRST_20_MEMBERS = {
    "m01", "m02", "m03", "m04", "m05", "m06", "m07", "m08", "m09", "m10",
    "m11", "m12", "m13", "m14", "m15", "m16", "m17", "m18", "m19", "m20"
}
# fmt: on

_FIELD_HEAD = 'wide: string = "'
_FIELD_TAIL = '"'


def _field_of_length(n: int) -> str:
    """Return a class-field declaration whose node text is exactly n characters.

    Tree-sitter's public_field_definition excludes the trailing semicolon, so the
    node text equals this string.
    """
    pad = "x" * (n - len(_FIELD_HEAD) - len(_FIELD_TAIL))
    text = f"{_FIELD_HEAD}{pad}{_FIELD_TAIL}"
    assert len(text) == n, "_field_of_length arithmetic is wrong"
    return text


def _alias_of_length(n: int) -> str:
    """Return a non-exported `type ... = "..."` declaration of exactly n characters."""
    head, tail = 'type Long = "', '";'
    text = head + "y" * (n - len(head) - len(tail)) + tail
    assert len(text) == n, "_alias_of_length arithmetic is wrong"
    return text


def test_typescript_extraction_limits_are_exact():
    """Kills: MAX_CLASS_FIELDS 30 -> 0 (or any other value); `field_count <
    self.MAX_CLASS_FIELDS` -> `<=`; MAX_INTERFACE_MEMBERS 20 -> 0; `count >=
    self.MAX_INTERFACE_MEMBERS` -> `>`; the class-field body truncation `if len(body) >
    300` -> `>= 300` or `> 3000`; `body[:300]` -> another width; the class-field
    `signature=body.split("\\n")[0][:200]` -> another width; the type-alias truncation
    `if len(body) > 500` -> `>= 500` or `> 5`.
    """
    # Preconditions: the caps can only be observed if the fixtures overshoot them.
    assert WIDE_CLASS_TS.count(": number = ") == 31, "WIDE_CLASS_TS no longer declares 31 fields"
    assert BIG_INTERFACE_TS.count(": number;") == 21, (
        "BIG_INTERFACE_TS no longer declares 21 members"
    )

    parser = TypeScriptParser(tsx=False)

    # --- MAX_CLASS_FIELDS: 31 declared, exactly the first 30 kept.
    wide = parser.parse(WIDE_CLASS_TS, file_id=1)
    field_names = {s.name for s in wide.symbols if s.kind.name == "VARIABLE"}
    assert field_names == FIRST_30_FIELDS
    assert len([s for s in wide.symbols if s.kind.name == "VARIABLE"]) == 30

    # --- MAX_INTERFACE_MEMBERS: 21 declared, exactly the first 20 kept.
    big = parser.parse(BIG_INTERFACE_TS, file_id=1)
    member_names = {s.name for s in big.symbols if s.parent_id is not None}
    assert member_names == FIRST_20_MEMBERS
    assert len([s for s in big.symbols if s.parent_id is not None]) == 20

    # --- class-field body truncation boundary: 300 kept verbatim, 301 truncated to 303.
    exact = _field_of_length(300)
    over = _field_of_length(301)
    r_exact = parser.parse("class C {\n  " + exact + ";\n}\n", file_id=1)
    r_over = parser.parse("class C {\n  " + over + ";\n}\n", file_id=1)
    exact_sym = next(s for s in r_exact.symbols if s.name == "wide")
    over_sym = next(s for s in r_over.symbols if s.name == "wide")
    assert exact_sym.body == exact
    assert over_sym.body == over[:300] + "..."
    assert len(over_sym.body) == 303
    # signature is the first line capped at 200 characters.
    assert over_sym.signature == over[:200]
    assert len(over_sym.signature) == 200

    # --- type-alias body truncation boundary: 500 kept verbatim, 501 truncated to 503.
    alias_exact = _alias_of_length(500)
    alias_over = _alias_of_length(501)
    a_exact = parser.parse(alias_exact + "\n", file_id=1)
    a_over = parser.parse(alias_over + "\n", file_id=1)
    assert next(s for s in a_exact.symbols if s.name == "Long").body == alias_exact
    over_alias = next(s for s in a_over.symbols if s.name == "Long")
    assert over_alias.body == alias_over[:500] + "..."
    assert len(over_alias.body) == 503


# ---------------------------------------------------------------------------
# Fixture 3 — doc-comment attachment.
# ---------------------------------------------------------------------------
#  1 /**
#  2  * Module header.
#  3  */
#  4 import { x } from './x';
#  5
#  6 /** Adds numbers. */
#  7 export function add(a: number, b: number): number {
#  8   return a + b;
#  9 }
# 10
# 11 /** Detached comment. */
# 12
# 13 export function detached(): void {}
DOC_TS = """\
/**
 * Module header.
 */
import { x } from './x';

/** Adds numbers. */
export function add(a: number, b: number): number {
  return a + b;
}

/** Detached comment. */

export function detached(): void {}
"""

# Same file, but the first block comment starts on line 2, so it is NOT a
# module header.
LATE_COMMENT_TS = """\

/** Not a module header. */
export function only(): void {}
"""


def test_typescript_doc_comment_attachment_is_exact():
    """Kills: `if prev.end_point[0] + 1 < next_start_line` -> `<=` in
    _get_preceding_comment (which silently drops EVERY adjacent doc comment) and the
    reverse `>` form (which would attach a comment separated by a blank line);
    `child.start_point[0] == 0` -> `<= 1` in _get_module_docstring.
    """
    # Precondition: exactly one blank line must separate the detached comment
    # from `export function detached`, otherwise the gap rule is untested.
    assert "/** Detached comment. */\n\nexport function detached" in DOC_TS, (
        "DOC_TS no longer has a blank line between the detached comment and its function"
    )
    assert LATE_COMMENT_TS.startswith("\n/**"), (
        "LATE_COMMENT_TS must start with a blank line so its comment is not at line 0"
    )

    parser = TypeScriptParser(tsx=False)
    result = parser.parse(DOC_TS, file_id=5)
    docs = {s.qualified_name: s.docstring for s in result.symbols}

    assert docs["<module>"] == "Module header."
    assert docs["add"] == "Adds numbers."
    assert docs["detached"] is None

    late = parser.parse(LATE_COMMENT_TS, file_id=5)
    late_names = {s.qualified_name for s in late.symbols}
    assert late_names == {"only"}
    assert next(s for s in late.symbols if s.qualified_name == "only").docstring == (
        "Not a module header."
    )


# ---------------------------------------------------------------------------
# Fixture 4 — call edges: callee name, line, and parameter-derived type hint.
# ---------------------------------------------------------------------------
#  1 class Ctl {
#  2   constructor(private svc: AuthService) {}
#  3
#  4   run(db: Database, n: number): void {
#  5     db.query();
#  6     this.svc.check();
#  7     helper(db.escape());
#  8     new Widget();
#  9   }
# 10 }
CALLS_TS = """\
class Ctl {
  constructor(private svc: AuthService) {}

  run(db: Database, n: number): void {
    db.query();
    this.svc.check();
    helper(db.escape());
    new Widget();
  }
}
"""

# (caller qualified_name, callee_name, line, callee_type_hint)
CALLS_EXPECTED: set[tuple[str, str, int, str | None]] = {
    ("Ctl.run", "query", 5, "Database"),
    ("Ctl.run", "check", 6, None),
    ("Ctl.run", "helper", 7, None),
    ("Ctl.run", "escape", 7, "Database"),
    ("Ctl.run", "Widget", 8, None),
}


def test_typescript_call_edges_carry_exact_line_and_type_hint():
    """Kills: `type_hint = pt.get(receiver_name)` -> `type_hint = None` in _walk_node
    (which silently kills the priority-2 type-hint call-resolution path); any
    `child.start_point[0] + 1` -> `+ 0`/`+ 2` in the call_expression, member_expression
    or new_expression branches; dropping the `self._walk_body(...)` recursion that finds
    calls nested inside call arguments.
    """
    # Preconditions: the type hint can only be observed because `db` is a typed
    # parameter and one nested call goes through the same receiver.
    assert "run(db: Database" in CALLS_TS, "CALLS_TS lost the typed `db: Database` parameter"
    assert "helper(db.escape())" in CALLS_TS, "CALLS_TS lost its nested call argument"

    result = TypeScriptParser(tsx=False).parse(CALLS_TS, file_id=9)
    actual = {
        (result.symbols[e.caller_id].qualified_name, e.callee_name, e.line, e.callee_type_hint)
        for e in result.call_edges
    }
    assert actual == CALLS_EXPECTED
    assert len(result.call_edges) == 5
