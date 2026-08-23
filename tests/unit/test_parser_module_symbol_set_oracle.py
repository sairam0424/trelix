"""
Exact-set oracle for the parser layer's synthetic MODULE symbol.

Mutation testing showed the parser can silently lose a whole symbol without a
single test noticing, because the existing parser tests ask "is my symbol in the
list?" and never "is the list exactly this?".  A `len(symbols) > 0` or
`any(s.name == "Foo")` assertion cannot see a deleted append; an exact, ordered
table of (name, kind, line_start, line_end, resolved-parent-name) can.

MUTANTS KILLED HERE
-------------------
M1  src/trelix/indexing/parser/extractors/python.py :: PythonParser.parse
        symbols.append(self._module_symbol(file_id, root, module_doc))
    -> deleted / replaced with `pass`
    Killed by: test_python_symbol_set_is_exact,
               test_python_first_walked_class_is_not_clobbered_by_module_symbol
    That append RESERVES symbols[0] for the "<module>" symbol before the AST
    walk.  Delete it and the walk starts writing at index 0, so the later
    `symbols[0] = self._module_symbol(...)` rebuild OVERWRITES the first real
    symbol.  One symbol vanishes from the file's index, and when the first
    walked symbol is a class its methods' parent_id then resolves to
    "<module>" — a method whose parent is a file.

M2  src/trelix/indexing/parser/extractors/python.py :: PythonParser.__init__
        self._ts_lang = load_language("python")
    -> self._ts_lang = None
    NOT KILLED, and deliberately not tested: this is an EQUIVALENT mutant.
    Measured: `_ts_lang` / `_ts_language` is assigned in 13 extractors and READ
    IN NONE — the live parse path is `self._parser = make_parser(...)`.  So the
    mutation changes no observable behaviour, and the only way to "kill" it is
    to reach into a private attribute production code never touches.  A draft of
    this file did exactly that; it was removed, because such a test asserts a
    config value rather than behaviour AND locks in dead code — the correct
    cleanup is deleting those 13 assignments, which the test would then block.
    See the note where that test used to live, below the Python section.

M3  src/trelix/indexing/parser/extractors/go.py :: GoParser.parse
        symbols.append(module_sym)
    -> deleted / replaced with `pass`
    Killed by: test_go_symbol_set_is_exact
    Measured before this file existed: the Python one (M1) is already killed by
    tests/unit/test_parser_module_insert_indices.py, but the identical mutation
    in the Go extractor survived the whole parser suite — no test anywhere
    asserted that a Go file's package comment produces a MODULE symbol.

DELIBERATELY NOT ASSERTED: Symbol.body.  It is documented as verbatim source
text but is synthesised for several extractors (flattened key paths for
json/toml/yaml, docstring-only bodies for Go), so a universal verbatim
assertion is wrong by design.  Names, kinds, line spans and parent linkage are
the fields the indexer, the graph and the line-window retriever depend on.

LINE SPANS: line_start / line_end are 1-indexed and inclusive, so every row
below can be checked by eye against the fixture text.  Because the whole table
is compared for equality, a one-line drift in ANY symbol's start or end fails
these tests — including the `line_end` off-by-one class of bug.

Every fixture is written to disk WITHOUT a trailing newline, so the MODULE
symbol's `line_end` (tree-sitter's root `end_point`) equals the fixture's real
line count.  With a trailing newline both extractors report line_count + 1 for
the module symbol, i.e. a line that does not exist in the file; that is a
separate defect and is not what these tests pin.
"""

from __future__ import annotations

from pathlib import Path

from trelix.indexing.parser.base import ParseResult
from trelix.indexing.parser.extractors.go import GoParser
from trelix.indexing.parser.extractors.python import PythonParser

# Rows are (name, kind, line_start, line_end, resolved parent name).
Row = tuple[str, str, int, int, str | None]

# ---------------------------------------------------------------------------
# Python fixture A — the first symbol the walk appends is a module constant.
# The line number of each source line is in the comment column so the expected
# table is hand-checkable.
# ---------------------------------------------------------------------------

PY_FIXTURE_A = (
    '"""Widget helpers for the fixture module."""\n'  # 1
    "\n"  # 2
    "DEFAULT_TIMEOUT = 30\n"  # 3
    "\n"  # 4
    "\n"  # 5
    "class Alpha:\n"  # 6
    '    """Alpha docstring."""\n'  # 7
    "\n"  # 8
    "    def run(self):\n"  # 9
    "        return DEFAULT_TIMEOUT\n"  # 10
    "\n"  # 11
    "\n"  # 12
    "class Beta:\n"  # 13
    "    def go(self):\n"  # 14
    "        return 2"  # 15  (no trailing newline — see module docstring)
)

# Written out literally, in parse order, one row per symbol.  Never generated
# from the parser's own output, and nothing here is iterated to build itself:
# deleting a row would make this test fail, not make it check less.
PY_EXPECTED_A: list[Row] = [
    ("<module>", "module", 1, 15, None),
    ("DEFAULT_TIMEOUT", "constant", 3, 3, None),
    ("Alpha", "class", 6, 10, None),
    ("run", "method", 9, 10, "Alpha"),
    ("Beta", "class", 13, 15, None),
    ("go", "method", 14, 15, "Beta"),
]

# ---------------------------------------------------------------------------
# Python fixture B — the first symbol the walk appends is a CLASS, so losing
# the reserved slot also re-points that class's method at the module symbol.
# ---------------------------------------------------------------------------

PY_FIXTURE_B = (
    '"""Two classes, no module constant."""\n'  # 1
    "\n"  # 2
    "\n"  # 3
    "class Alpha:\n"  # 4
    "    def run(self):\n"  # 5
    "        return 1\n"  # 6
    "\n"  # 7
    "\n"  # 8
    "class Beta:\n"  # 9
    "    def go(self):\n"  # 10
    "        return 2"  # 11  (no trailing newline)
)

PY_EXPECTED_B: list[Row] = [
    ("<module>", "module", 1, 11, None),
    ("Alpha", "class", 4, 6, None),
    ("run", "method", 5, 6, "Alpha"),
    ("Beta", "class", 9, 11, None),
    ("go", "method", 10, 11, "Beta"),
]

# ---------------------------------------------------------------------------
# Go fixture — a package doc comment plus a struct, a field, a method and a
# function, so a lost module symbol shows up as a missing row rather than an
# empty result.
# ---------------------------------------------------------------------------

GO_FIXTURE = (
    "// Package widget builds widgets.\n"  # 1
    "package widget\n"  # 2
    "\n"  # 3
    "// Store holds widgets.\n"  # 4
    "type Store struct {\n"  # 5
    "\tName string\n"  # 6
    "}\n"  # 7
    "\n"  # 8
    "// Get returns the name.\n"  # 9
    "func (s *Store) Get() string {\n"  # 10
    "\treturn s.Name\n"  # 11
    "}\n"  # 12
    "\n"  # 13
    "func New() *Store {\n"  # 14
    "\treturn &Store{}\n"  # 15
    "}"  # 16  (no trailing newline)
)

# NOTE on the first row's name: GoParser._get_module_symbol looks the package
# name up with `pkg_node.child_by_field_name("name")`, which is None for this
# grammar's `package_clause`, so the name falls back to the literal "package"
# instead of "widget".  That is a real defect, pinned here as CURRENT
# behaviour so it is visible rather than invisible — when it is fixed this row
# becomes ("widget", "module", 1, 16, None) and this test will say so.
GO_EXPECTED: list[Row] = [
    ("package", "module", 1, 16, None),
    ("Store", "class", 5, 7, None),
    ("Name", "variable", 6, 6, "Store"),
    ("Get", "method", 10, 12, "Store"),
    ("New", "function", 14, 16, None),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_and_read(tmp_path: Path, filename: str, text: str, line_count: int) -> str:
    """Write *text* to a real file, read it back, and assert it still discriminates.

    The preconditions matter: if a fixture's leading doc comment is ever
    dropped, the extractor's module-symbol branch never runs and the deletion
    mutants become invisible — the tests would then pass for the wrong reason.
    If the line count or the trailing newline changes, the hand-written line
    spans are stale.
    """
    path = tmp_path / filename
    path.write_text(text, encoding="utf-8")
    source = path.read_text(encoding="utf-8")

    assert not source.endswith("\n"), "fixture must not end with a newline"
    assert len(source.splitlines()) == line_count, "fixture line count changed"

    return source


def _as_rows(result: ParseResult) -> list[Row]:
    """Project the PARSED result (never the expected table) into comparable rows.

    parent_id is a local index into this same symbols list at parse time, so it
    is resolved to the parent's name: an index that shifts by one still points
    at a real, plausible-looking symbol, and only the name makes that visible.
    """
    rows: list[Row] = []
    for sym in result.symbols:
        parent_name = None if sym.parent_id is None else result.symbols[sym.parent_id].name
        rows.append((sym.name, str(sym.kind), sym.line_start, sym.line_end, parent_name))
    return rows


# ---------------------------------------------------------------------------
# Python
# ---------------------------------------------------------------------------


def test_python_symbol_set_is_exact(tmp_path: Path) -> None:
    """Kills M1: deleting `symbols.append(self._module_symbol(...))` in
    PythonParser.parse (the reserved symbols[0] slot).

    Without the reserved slot the walk's first symbol (DEFAULT_TIMEOUT) is
    overwritten by the "<module>" rebuild, so the extracted set is 5 rows and
    not these 6.  Also kills any off-by-one in line_start / line_end for any of
    the six symbols.
    """
    source = _write_and_read(tmp_path, "fixture_a.py", PY_FIXTURE_A, line_count=15)
    assert source.startswith('"""'), "fixture must open with a module docstring"

    result = PythonParser().parse(source, file_id=7)

    assert _as_rows(result) == PY_EXPECTED_A
    # The module symbol is not merely present, it occupies index 0 — the slot
    # the indexer's local-index remapping is built around.
    assert result.symbols[0].name == "<module>"
    assert result.parse_errors == 0


def test_python_first_walked_class_is_not_clobbered_by_module_symbol(tmp_path: Path) -> None:
    """Kills M1 a second way: via parent linkage rather than a missing row.

    Here the first walked symbol is `class Alpha`.  Without the reserved slot,
    Alpha is overwritten by "<module>" and Alpha.run's parent_id (0) resolves to
    the module symbol.  Both the vanished class and the bogus parent break the
    row comparison.
    """
    source = _write_and_read(tmp_path, "fixture_b.py", PY_FIXTURE_B, line_count=11)
    assert source.startswith('"""'), "fixture must open with a module docstring"

    result = PythonParser().parse(source, file_id=7)

    assert _as_rows(result) == PY_EXPECTED_B
    # Stated again positively, because "a method's parent is a class, never the
    # file" is the invariant the graph layer relies on.
    assert result.symbols[1].name == "Alpha"
    assert result.symbols[2].name == "run"
    assert result.symbols[2].parent_id == 1
    assert result.parse_errors == 0


# DELIBERATELY NOT TESTED: `self._ts_lang = load_language("python")`.
#
# An earlier draft of this file asserted that attribute holds a usable Python grammar, to
# kill the mutant `self._ts_lang = None`. Adversarial review measured that `_ts_lang` /
# `_ts_language` is ASSIGNED in 13 extractors and READ IN NONE — the live parse path is
# `self._parser = make_parser(...)`. So `= None` is an EQUIVALENT mutant: it changes no
# observable behaviour, and the only way to "kill" it is to reach into a private attribute
# production code never touches and re-derive a grammar from it.
#
# That test would have been a config-value assertion dressed as behaviour, and worse, it
# would LOCK IN DEAD CODE: the correct cleanup here is deleting those 13 assignments, and
# the test would then fail with AttributeError and block it. A test that prevents a
# legitimate refactor while asserting nothing observable is negative value.
#
# The honest disposition is to classify the mutant as equivalent, record why, and delete
# the assignments — not to test them. Recorded here so the mutant is not re-reported as a
# survivor and someone does not re-add the test.


# ---------------------------------------------------------------------------
# Go
# ---------------------------------------------------------------------------


def test_go_symbol_set_is_exact(tmp_path: Path) -> None:
    """Kills M3: deleting `symbols.append(module_sym)` in GoParser.parse.

    A Go file whose package clause carries a doc comment must yield a MODULE
    symbol as its first symbol.  Nothing else in the suite asserted this, so the
    deletion left the package comment unindexed and every Go parser test still
    passed.  The exact table also pins the struct/field/method/function spans.
    """
    source = _write_and_read(tmp_path, "widget.go", GO_FIXTURE, line_count=16)
    assert source.startswith("// Package widget"), "fixture must open with a package doc comment"

    result = GoParser().parse(source, file_id=7)

    assert _as_rows(result) == GO_EXPECTED
    assert str(result.symbols[0].kind) == "module"
    assert result.symbols[0].docstring == "Package widget builds widgets."
    assert result.parse_errors == 0
