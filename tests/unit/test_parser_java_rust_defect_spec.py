"""Executable SPECIFICATION for the live defects in the Java and Rust extractors.

This file is the counterpart to tests/unit/test_parser_java_symbol_set_oracle.py and
tests/unit/test_parser_rust_symbol_set_oracle.py. Those two files pin the extractors'
CURRENT behaviour with passing assertions, including the behaviour that is WRONG --
which documents each defect but does not make anyone fix it, and actively punishes the
person who does (their correct fix reads as a test failure).

Every defect below is therefore pinned the other way round: as a STRICT xfail whose
assertion states the CORRECT behaviour. The marker is a boomerang. While the defect is
live the assertion fails and the test xfails quietly. The moment the defect is fixed the
assertion passes, ``strict=True`` turns that XPASS into a FAILURE, and the fixer is
forced to delete the marker -- which is the release note. This is the "xfail now, fix in
its own release" pattern the project already uses.

MECHANICS, each earned from a shipped defect
--------------------------------------------
* Every xfail carries ``raises=AssertionError``. Without it the marker absorbs ANY
  exception -- an ImportError, a TypeError from a refactor, a grammar upgrade that makes
  the fixture unparseable -- and the boomerang silently stops working. That failure was
  found in four separate places across earlier rounds.
* Preconditions raise ``FixturePreconditionError``, NOT ``AssertionError``. This is the
  control that reads differently when the setup is wrong. An ``assert`` in a
  ``raises=AssertionError`` xfail is indistinguishable from the defect it is guarding,
  so a fixture that stopped discriminating would report a tidy ``xfailed`` forever. A
  ``FixturePreconditionError`` escapes the marker and ERRORS the test loudly.
* The GRAMMAR FACT tests at the top are plain passing tests, not xfails. Each defect
  note below names a tree-sitter node type; a pinned-but-wrong mechanism is worse than
  no pin, so the node names are asserted against the INSTALLED grammar. If a grammar
  upgrade renames anything, those tests fail and every diagnosis below is re-opened
  before anyone acts on it.
* Fixtures carry NO trailing newline. With one, tree-sitter's root ``end_point`` reports
  a line the file does not have, and pinning that number locks in an artifact.
* ``Symbol.body`` is never asserted to be verbatim source -- that is false for >= 10 of
  26 languages by design.
* One defect per test. Where a single root cause has two observable consequences at two
  different code sites (the Rust comment-newline quirk shows up in
  ``_get_preceding_comment``'s join AND in ``_get_module_symbol``'s join) they get
  SEPARATE tests with SEPARATE fixtures, because a coupled measurement reported as an
  individual one is how an earlier round mis-attributed a kill.

DEFECTS, RANKED BY USER IMPACT
------------------------------
Tier 1 -- an entire language feature, or the first symbol of every file, is wrong.

  J1  ``java._handle_record`` IS DEAD CODE END TO END. It looks for grammar nodes
      ``record_parameters`` / ``record_component``; the installed tree-sitter-java emits
      ``formal_parameters`` / ``formal_parameter`` (VERIFIED -- see
      ``test_grammar_fact_java_record_header_is_formal_parameters``, and note the field
      name on ``record_declaration`` is ``parameters``). Consequence: Java RECORDS are
      entirely invisible to the indexer. Not one component field is extracted, even
      though java.py's own module docstring calls them "the primary query surface", and
      ``_record_signature`` renders every record as ``record Name()`` -- so the record's
      own embedded header omits its entire shape. Records are the standard value type
      since JDK 16. THIS ONE DESERVES ITS OWN RELEASE, AHEAD OF EVERYTHING ELSE HERE.
  R14 A ``//!`` CRATE-LEVEL doc comment is ALSO attached as the docstring of the first
      following item. ``_get_preceding_comment`` does not distinguish ``//!`` (documents
      the enclosing scope) from ``///`` (documents the next item), and the gap guard is
      one out (R7), so a gap of zero OR one blank line both leak. Opening a Rust file
      with ``//!`` is idiomatic, so the first symbol of most real Rust files carries the
      whole crate header as its embedded doc text. NEW -- neither oracle sees this,
      because in their fixture the ``//!`` block is followed by a ``use`` and an
      ``extern crate``, so no symbol-bearing item is the comment's next named sibling.

Tier 2 -- corrupts data that gets EMBEDDED or graph-joined. Silent retrieval decay.

  J3  ``java._extract_import`` CANNOT REPRESENT A WILDCARD. For ``import java.util.*;``
      the ``*`` is an ``asterisk`` node that is a SIBLING of the ``scoped_identifier``
      (VERIFIED), so the recorded edge is ``imported_from="java"`` /
      ``imported_names=["util"]``: a bogus dependency on package ``java`` AND a lost
      dependency on ``java.util``. ``ImportEdge``'s own docstring in core/models.py says
      ``imported_names`` is ``["*"]`` for wildcards, and python.py honours it (pinned as
      a control below), so this is a CONTRACT VIOLATION, not a preference.
  J6  An ANNOTATED FIELD'S SIGNATURE IS THE ANNOTATION LINE. ``signature`` is
      ``body.split("\\n")[0]`` and the ``field_declaration`` node starts at its
      ``modifiers``, so ``Svc.repo`` gets ``"@Autowired"`` -- no type, no name. The
      annotated fields are exactly the ones java.py says it surfaces on purpose
      (``@Autowired``, ``@Column``, ``@Id``, ``@Value``), and the signature is embedded
      text, so the dependency-injection surface is indexed as a bare annotation.
  R10 ``use crate::util::{Alpha, Beta};`` STORES THE LITERAL ``"{Alpha, Beta}"`` AS AN
      IMPORTED NAME. ``_flatten_use_tree`` branches on ``use_tree_list`` / ``use_tree``;
      the installed tree-sitter-rust emits ``scoped_use_list`` / ``use_list`` (VERIFIED),
      so both branches are unreachable and the generic "split on ``::``" fallback runs.
      Brace imports are the dominant Rust import form, so the import graph loses every
      symbol in them.
  R9  EVERY MULTI-LINE ``///`` DOC COMMENT GAINS A BLANK LINE BETWEEN EVERY PAIR OF
      LINES. tree-sitter-rust's ``line_comment`` node text INCLUDES its trailing newline
      (VERIFIED), and ``_get_preceding_comment`` joins the pieces with another ``"\\n"``.
      ``"Two\\nlines."`` is indexed as ``"Two\\n\\nlines."``.
  R9b The SAME quirk at a DIFFERENT code site: ``_get_module_symbol``'s
      ``"\\n".join(inner_doc_lines)`` does it to the crate docstring too. Pinned
      separately because it is a separate join that a fix to R9 need not touch.
  J4  NESTED JAVA TYPES GET A BARE ``qualified_name`` AND NO PARENT. ``_handle_class``
      recurses with no outer-name prefix and does not pass itself as the parent, so
      ``A.Inner`` and ``B.Inner`` BOTH become ``Inner``, with ``parent_id=None``.
      *** This is the SAME COLLISION CLASS as the ``symbols.qualified_name`` finding in
      tests/unit/test_db_scoping_and_boundaries.py -- and it is strictly worse. That
      finding needs TWO FILES to collide; this one collides WITHIN A SINGLE FILE.
      ``Indexer._insert_one`` keys ``existing_hashes`` on ``qualified_name``, so the
      second ``Inner`` with an equal body is declared unchanged, never inserted, never
      chunked, never embedded -- silently unsearchable while ``trelix stats`` still
      counts it. Fixing db.py's scoping does NOT fix this; the collision is upstream of
      the query. These two should ship together. ***
  R12 ``fn`` items inside a ``mod`` block get NO module prefix (``nested``, not
      ``inner::nested``). Same collision class as J4: two ``mod``s in one file each with
      a ``fn new`` collide.
  R8  A ``///`` DOC COMMENT IS LOST WHENEVER AN ``#[attr]`` SITS BETWEEN IT AND THE ITEM,
      because ``_get_preceding_comment`` walks ``prev_named_sibling`` and stops at the
      ``attribute_item`` (VERIFIED: the attribute really is the item's immediately
      preceding named sibling). Rust idiom puts attributes AFTER doc comments, so this
      loses the docs on most ``#[derive]``d, ``#[inline]``d and ``#[serde]``d items.

Tier 3 -- wrong graph edges or wrong metadata, narrower blast radius.

  J2  A RECORD'S ``implements`` TypeEdge IS LOST. ``_handle_record`` scans the
      ``super_interfaces`` node's DIRECT children for ``type_identifier``, but the
      identifiers live one level down inside a ``type_list`` (VERIFIED). Affects only
      records that implement an interface, and only the type graph -- not searchable
      text -- which is why it ranks below J1 despite sharing a function with it.
  R11 An ASSOCIATED TYPE inside ``impl Trait for Type`` is emitted as a TOP-LEVEL
      ``INTERFACE`` named ``Out`` with ``parent_id=None``, because ``_handle_impl``
      routes ``type_item`` through ``_handle_type_alias``, which takes no parent.
      ``Out`` / ``Item`` / ``Error`` are near-universal associated-type names, so this
      is also a collision source.
  R7  The BLANK-LINE GAP GUARD IN ``_get_preceding_comment`` IS OFF BY ONE FOR RUST. A
      ``///`` comment separated from its item by ONE blank line is still attached,
      because the ``line_comment`` node's ``end_point`` ALREADY names the line after the
      comment (its text includes the newline), so the ``+ 1`` in
      ``prev.end_point[0] + 1 < next_start_line`` is one too many (VERIFIED, with the
      arithmetic pinned). Java's identical guard is CORRECT there, because a Java
      ``/** */`` block comment ends at ``*/`` on its own line -- so this must be fixed in
      rust.py only, and a "consistent" fix applied to both extractors would break the
      healthy one.
  R13 TRAIT METHOD SIGNATURES REPORT ``is_public=False`` on a ``pub trait``.
      ``_handle_trait_fn`` looks for a ``visibility_modifier`` that trait fns never
      carry. Metadata only, but it is metadata that ``is_public`` filters act on.

Tier 4 -- cosmetic in isolation, but the signature IS embedded text.

  J5  ``_class_signature`` DUPLICATES THE KEYWORDS:
      ``class Svc extends extends Base implements implements Runnable, AutoCloseable``.
      The ``superclass`` and ``super_interfaces`` nodes already contain their keyword
      (VERIFIED). Asked directly: DOES THIS CHANGE RETRIEVAL? Yes, mildly and
      systematically. The signature is not a UI string -- it is part of the text that
      gets embedded and lexically indexed for the class header. Duplicating ``extends``
      and ``implements`` doubles those terms' frequency in EVERY Java class chunk that
      has a superclass or an interface list, which biases BM25 term statistics across
      the whole Java corpus. Nothing becomes unfindable, which is why it is last, but
      "cosmetic" would be the wrong word for a token that reaches the index.

RELEASE ORDER
-------------
  1. J1 (+J1b) alone -- Java records. An entire language feature.
  2. R14 + R9 + R8 + R7 as one "Rust doc comments" release; they share
     ``_get_preceding_comment`` and fixing them separately means touching it four times.
     R7's fix must be rust-only.
  3. J3 + J6 + R10 -- "import and signature fidelity". All three corrupt embedded text.
  4. J4 + R12 + R11 -- "qualified-name scoping", shipped WITH the db.py scoping fix.
  5. J2, R13, J5.

MUTANTS AND DEAD CODE REPORTED, NOT TESTED (rule: a test pinning dead code blocks cleanup)
------------------------------------------------------------------------------------------
* ``java._extract_import``: ``imported_name = parts[-1] if parts[-1] != "*" else "*"``.
  The ``else "*"`` arm is UNREACHABLE -- ``parts`` comes from the ``scoped_identifier``,
  which the ``asterisk`` is not part of, so ``parts[-1]`` can never be ``"*"``. Deleting
  the conditional is an EQUIVALENT mutant today. It becomes live only as part of J3's fix.
* ``rust._flatten_use_tree``: the ``use_tree_list`` and ``use_tree`` branches are DEAD
  against the installed grammar (node types are ``scoped_use_list`` / ``use_list``).
  Deleting both branches is an EQUIVALENT mutant. Pinned only negatively, by
  ``test_grammar_fact_rust_brace_import_is_a_scoped_use_list``, which asserts those node
  types are ABSENT rather than asserting the dead branches run.
* ``java._handle_record``'s ``record_parameters`` / ``record_component`` lookups are
  likewise dead; the xfails below specify the ABSENCE of their output rather than
  exercising them.
* ``JavaParser.__init__`` / ``RustParser.__init__``: ``self._ts_language = ...`` is
  assigned and never read (already reported by the two oracles). EQUIVALENT.
"""

from __future__ import annotations

import pytest
from tree_sitter import Node

from trelix.indexing.parser._grammar import make_parser
from trelix.indexing.parser.extractors.java import JavaParser
from trelix.indexing.parser.extractors.python import PythonParser
from trelix.indexing.parser.extractors.rust import RustParser


class FixturePreconditionError(RuntimeError):
    """A pinning fixture has stopped discriminating.

    Deliberately NOT an AssertionError. Every pin in this file is
    ``xfail(strict=True, raises=AssertionError)``, so an AssertionError raised by a
    precondition would be absorbed by the marker and the pin would report a tidy
    ``xfailed`` while testing nothing -- the green-when-vacuous failure mode. This type
    escapes ``raises=`` and errors the test loudly instead.
    """


def require(condition: object, message: str) -> None:
    """Precondition that survives an xfail marker. See FixturePreconditionError."""
    if not condition:
        raise FixturePreconditionError(message)


def _named_child_types(node: Node) -> list[str]:
    return [c.type for c in node.children if c.is_named]


def _all_descendant_types(node: Node) -> set[str]:
    seen = {node.type}
    for child in node.children:
        seen |= _all_descendant_types(child)
    return seen


# ===========================================================================
# GRAMMAR FACTS -- the control for every diagnosis below.
#
# These are plain PASSING tests. They assert the node names the installed
# tree-sitter grammars actually emit, written out as literals. A grammar upgrade
# that renames any of them fails here FIRST, which re-opens the corresponding
# defect note instead of letting a stale mechanism ride.
# ===========================================================================

RECORD_JAVA = "public record Pt(int x, int y) implements Cloneable {}"
WILDCARD_IMPORT_JAVA = "import java.util.*;"


def _java_record_node(source: str) -> Node:
    root = make_parser("java").parse(source.encode("utf-8")).root_node
    node = root.children[0]
    require(
        node.type == "record_declaration",
        f"fixture no longer parses to a record_declaration (got {node.type!r}); "
        "the record diagnoses J1/J1b/J2 depend on this",
    )
    return node


def test_grammar_fact_java_record_header_is_formal_parameters() -> None:
    """MECHANISM CONTROL for J1/J1b. Fails if a tree-sitter-java upgrade renames the
    record header nodes -- at which point "``_handle_record`` looks for the wrong node
    names" must be re-verified before anyone acts on it.

    Kills the claim, not a mutant: if this file said ``record_parameters`` and the
    grammar said so too, ``_handle_record`` would work and J1 would be fiction.
    """
    node = _java_record_node(RECORD_JAVA)
    types = _named_child_types(node)

    # Written out as literals; NOT read back from the parser.
    assert types == [
        "modifiers",
        "identifier",
        "formal_parameters",
        "super_interfaces",
        "class_body",
    ]
    # The field name, which is what a fix would use.
    params = node.child_by_field_name("parameters")
    assert params is not None
    assert params.type == "formal_parameters"
    assert _named_child_types(params) == ["formal_parameter", "formal_parameter"]

    # The two names java.py actually looks for do not exist ANYWHERE in this tree.
    descendants = _all_descendant_types(node)
    assert "record_parameters" not in descendants
    assert "record_component" not in descendants


def test_grammar_fact_java_super_interfaces_wraps_the_types_in_a_type_list() -> None:
    """MECHANISM CONTROL for J2. ``_handle_record`` scans the ``interfaces`` node's
    DIRECT children for ``type_identifier``; this pins that the only direct named child
    is a ``type_list``, one level above where the identifier lives.
    """
    node = _java_record_node(RECORD_JAVA)
    interfaces = node.child_by_field_name("interfaces")
    assert interfaces is not None
    assert interfaces.type == "super_interfaces"
    assert _named_child_types(interfaces) == ["type_list"]
    type_list = interfaces.children[-1]
    assert type_list.type == "type_list"
    assert _named_child_types(type_list) == ["type_identifier"]


def test_grammar_fact_java_class_superclass_and_interfaces_nodes_carry_their_keyword() -> None:
    """MECHANISM CONTROL for J5. ``_class_signature`` prepends " extends " and
    " implements " to text that already begins with those words.
    """
    source = "class Svc extends Base implements Runnable, AutoCloseable {}"
    node = make_parser("java").parse(source.encode("utf-8")).root_node.children[0]
    require(
        node.type == "class_declaration",
        f"fixture no longer parses to a class_declaration (got {node.type!r})",
    )
    superclass = node.child_by_field_name("superclass")
    interfaces = node.child_by_field_name("interfaces")
    assert superclass is not None
    assert interfaces is not None
    assert source[superclass.start_byte : superclass.end_byte] == "extends Base"
    assert (
        source[interfaces.start_byte : interfaces.end_byte] == "implements Runnable, AutoCloseable"
    )


def test_grammar_fact_java_wildcard_asterisk_is_a_sibling_of_the_scoped_identifier() -> None:
    """MECHANISM CONTROL for J3. The ``*`` is its own node next to the path, so no
    amount of string-splitting the ``scoped_identifier`` can ever see it.
    """
    root = make_parser("java").parse(WILDCARD_IMPORT_JAVA.encode("utf-8")).root_node
    node = root.children[0]
    require(
        node.type == "import_declaration",
        f"fixture no longer parses to an import_declaration (got {node.type!r})",
    )
    assert _named_child_types(node) == ["scoped_identifier", "asterisk"]
    scoped = node.children[1]
    assert WILDCARD_IMPORT_JAVA[scoped.start_byte : scoped.end_byte] == "java.util"


GAP_ZERO_RS = "/// Attached to a.\npub fn a() {}"
GAP_ONE_RS = "/// One blank line before b.\n\npub fn b() {}"
ATTR_DOC_RS = "/// Blocked by the attribute.\n#[inline]\npub fn d() {}"
BRACE_USE_RS = "use crate::util::{Alpha, Beta};"


def test_grammar_fact_rust_line_comment_text_includes_its_trailing_newline() -> None:
    """MECHANISM CONTROL for R7 and R9. Pins the exact arithmetic that makes the gap
    guard one out: a ``line_comment``'s ``end_point`` row is the row AFTER the comment,
    at column 0, because the newline is inside the node.

    ``_get_preceding_comment`` breaks on ``prev.end_point[0] + 1 < next_start_line``.
    With the numbers below, the ONE-blank-line case computes ``1 + 1 < 2`` -> False, so
    the comment is (wrongly) kept. Dropping the ``+ 1`` gives ``1 < 2`` -> True.
    """
    parser = make_parser("rust")

    zero = parser.parse(GAP_ZERO_RS.encode("utf-8")).root_node
    comment, item = zero.children[0], zero.children[1]
    assert comment.type == "line_comment"
    assert GAP_ZERO_RS[comment.start_byte : comment.end_byte] == "/// Attached to a.\n"
    assert comment.start_point[0] == 0
    assert comment.end_point[0] == 1
    assert comment.end_point[1] == 0
    assert item.start_point[0] == 1

    one = parser.parse(GAP_ONE_RS.encode("utf-8")).root_node
    gap_comment, gap_item = one.children[0], one.children[1]
    assert gap_comment.type == "line_comment"
    assert gap_comment.end_point[0] == 1
    assert gap_item.type == "function_item"
    assert gap_item.start_point[0] == 2
    # The guard's own expression, spelled out: it does not fire, so a detached
    # comment is attached. This is R7.
    assert (gap_comment.end_point[0] + 1 < gap_item.start_point[0]) is False


def test_grammar_fact_rust_attribute_item_sits_between_the_doc_comment_and_the_item() -> None:
    """MECHANISM CONTROL for R8. The item's immediately preceding NAMED sibling is the
    ``attribute_item``, not the comment, so a ``prev_named_sibling`` walk that only
    accepts comment nodes stops before it ever reaches the doc.
    """
    root = make_parser("rust").parse(ATTR_DOC_RS.encode("utf-8")).root_node
    assert _named_child_types(root) == ["line_comment", "attribute_item", "function_item"]
    item = root.children[2]
    assert item.prev_named_sibling is not None
    assert item.prev_named_sibling.type == "attribute_item"


def test_grammar_fact_rust_brace_import_is_a_scoped_use_list() -> None:
    """MECHANISM CONTROL for R10. Pins that ``use_tree`` / ``use_tree_list`` -- the two
    node types ``_flatten_use_tree`` branches on -- are ABSENT from a brace import, so
    both branches are dead code and the generic fallback is what actually runs.

    Asserting their ABSENCE rather than exercising them is deliberate: a test that
    drove those branches would pin dead code and block its removal.
    """
    root = make_parser("rust").parse(BRACE_USE_RS.encode("utf-8")).root_node
    decl = root.children[0]
    require(
        decl.type == "use_declaration",
        f"fixture no longer parses to a use_declaration (got {decl.type!r})",
    )
    assert _named_child_types(decl) == ["scoped_use_list"]
    scoped_use_list = decl.children[1]
    assert _named_child_types(scoped_use_list) == ["scoped_identifier", "use_list"]

    descendants = _all_descendant_types(decl)
    assert "use_tree" not in descendants
    assert "use_tree_list" not in descendants


def test_python_extractor_honours_the_wildcard_import_contract() -> None:
    """CROSS-LANGUAGE CONTROL for J3. ``ImportEdge``'s docstring in core/models.py says
    ``imported_names`` is ``["*"]`` for a wildcard. python.py does exactly that, which
    is what makes J3 a CONTRACT VIOLATION rather than a matter of taste.

    Kills: ``imported_names = ["*"]`` -> anything else in python.py's
    ``wildcard_import`` branch.
    """
    source = "from os.path import *"
    require("import *" in source, "the control fixture lost its wildcard")

    result = PythonParser().parse(source, file_id=1)
    edges = {(e.imported_from, tuple(e.imported_names)) for e in result.import_edges}
    assert edges == {("os.path", ("*",))}
    assert len(result.import_edges) == 1


# ===========================================================================
# JAVA PINS
# ===========================================================================


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="DEFECT J1 (pinned deliberately): java._handle_record looks for the grammar "
    "nodes 'record_parameters'/'record_component'; tree-sitter-java emits "
    "'formal_parameters'/'formal_parameter', so record components are NEVER extracted "
    "and Java records are invisible to the indexer. Tier 1 -- ship first.",
)
def test_java_record_components_are_extracted_as_fields() -> None:
    """SPEC: a record's components ARE symbols -- java.py's module docstring calls them
    "the primary query surface".

    BOOMERANG: teaching ``_handle_record`` the real node names makes this XPASS, and
    ``strict=True`` turns that into a failure, forcing this marker's removal.

    Errors loudly (FixturePreconditionError, which ``raises=AssertionError`` does not
    absorb) if RECORD_JAVA stops declaring components.
    """
    require(
        "(int x, int y)" in RECORD_JAVA,
        "RECORD_JAVA lost its two components, so 'nothing was extracted' would be "
        "trivially true and this pin would xfail vacuously",
    )
    require(
        not RECORD_JAVA.endswith("\n"),
        "RECORD_JAVA must not end with a newline (tree-sitter reports a line that does "
        "not exist and any pinned span becomes an artifact)",
    )

    result = JavaParser().parse(RECORD_JAVA, file_id=1)
    rows = {(s.qualified_name, s.kind.name) for s in result.symbols}

    # Explicit literal table, set equality BOTH ways, plus the count that set
    # equality cannot see.
    assert rows == {("Pt", "CLASS"), ("Pt.x", "VARIABLE"), ("Pt.y", "VARIABLE")}
    assert len(result.symbols) == 3


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="DEFECT J1b (pinned deliberately): the same wrong node name in "
    "java._record_signature empties every record's parameter list, so the record's "
    "embedded header omits its entire shape. Separate code site from J1.",
)
def test_java_record_signature_shows_its_components() -> None:
    """SPEC: the signature of ``record Pt(int x, int y)`` names its components.

    Pinned separately from J1 because ``_record_signature`` is a DIFFERENT call site
    with the same wrong literal; reporting one measurement for two sites is the coupled
    measurement that mis-attributed a kill in an earlier round.

    BOOMERANG: the signature becoming ``public record Pt(int x, int y)`` makes this
    XPASS -> strict failure -> marker removed.
    """
    require(
        "(int x, int y)" in RECORD_JAVA,
        "RECORD_JAVA lost its components, so the signature assertion is vacuous",
    )

    result = JavaParser().parse(RECORD_JAVA, file_id=1)
    record = next(s for s in result.symbols if s.qualified_name == "Pt")

    assert record.signature == "public record Pt(int x, int y)"


def test_java_record_implements_clause_produces_a_type_edge() -> None:
    """SPEC: ``record Pt(...) implements Cloneable`` yields one ``implements`` TypeEdge.

    BOOMERANG: routing ``_handle_record`` through ``_extract_type_list_edges`` (which
    already has the ``type_list`` arm the class path uses) makes this XPASS.
    """
    require(
        "implements Cloneable" in RECORD_JAVA,
        "RECORD_JAVA lost its implements clause, so 'no edge was produced' would be "
        "correct rather than a defect",
    )

    result = JavaParser().parse(RECORD_JAVA, file_id=1)
    edges = {
        (result.symbols[e.from_symbol_id].qualified_name, e.to_type_name, e.edge_kind)
        for e in result.type_edges
    }

    assert edges == {("Pt", "Cloneable", "implements")}
    assert len(result.type_edges) == 1


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="DEFECT J3 (pinned deliberately): java._extract_import cannot see the "
    "'asterisk' sibling node, so 'import java.util.*;' is recorded as "
    "imported_from='java' / imported_names=['util'] -- a bogus edge to package 'java' "
    "and a lost edge to 'java.util'. Violates ImportEdge's documented ['*'] contract, "
    "which python.py honours (see the control test above).",
)
def test_java_wildcard_import_records_the_wildcard() -> None:
    """SPEC: ``import java.util.*;`` is ``imported_from="java.util"``,
    ``imported_names=["*"]`` -- the shape ``ImportEdge``'s own docstring specifies and
    ``PythonParser`` produces.

    BOOMERANG: reading the ``asterisk`` sibling makes this XPASS.
    """
    require(
        WILDCARD_IMPORT_JAVA == "import java.util.*;",
        "WILDCARD_IMPORT_JAVA is no longer a single wildcard import, so the exact edge "
        "table below no longer describes it",
    )

    result = JavaParser().parse(WILDCARD_IMPORT_JAVA, file_id=1)
    edges = {(e.imported_from, tuple(e.imported_names)) for e in result.import_edges}

    assert edges == {("java.util", ("*",))}
    assert len(result.import_edges) == 1


NESTED_COLLISION_JAVA = """\
class A {
    static class Inner {
        public int a;
    }
}

class B {
    static class Inner {
        public int b;
    }
}"""


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="DEFECT J4 (pinned deliberately): java._handle_class recurses into nested "
    "types with no outer-name prefix, so A.Inner and B.Inner both get "
    "qualified_name='Inner'. SAME COLLISION CLASS as the symbols.qualified_name "
    "finding in tests/unit/test_db_scoping_and_boundaries.py, but worse: that one "
    "needs two files, this one collides inside ONE file, and Indexer._insert_one keys "
    "existing_hashes on qualified_name.",
)
def test_java_nested_types_are_qualified_by_their_outer_type() -> None:
    """SPEC: a nested type's ``qualified_name`` is ``Outer.Nested``, and no two symbols
    in one file share a ``qualified_name``.

    The second assertion is the HARM, stated directly rather than as a proxy: the
    incremental indexer treats an equal ``qualified_name`` + equal body as "unchanged"
    and skips insert, chunk and embed, so the colliding member is silently unsearchable
    while ``trelix stats`` still counts it. Cross-reference
    tests/unit/test_db_scoping_and_boundaries.py -- fixing db.py's scoping does NOT fix
    this, because the collision happens upstream of the query.

    BOOMERANG: prefixing the recursion with the outer name makes this XPASS.
    """
    require(
        NESTED_COLLISION_JAVA.count("static class Inner {") == 2,
        "NESTED_COLLISION_JAVA no longer declares TWO same-named nested classes, which "
        "is the only fixture shape in which the collision is observable",
    )
    require(
        not NESTED_COLLISION_JAVA.endswith("\n"),
        "NESTED_COLLISION_JAVA must not end with a newline",
    )

    result = JavaParser().parse(NESTED_COLLISION_JAVA, file_id=1)

    class_names = sorted(s.qualified_name for s in result.symbols if s.kind.name == "CLASS")
    assert class_names == ["A", "A.Inner", "B", "B.Inner"]

    all_names = [s.qualified_name for s in result.symbols]
    assert len(all_names) == len(set(all_names))


SIGNATURE_JAVA = "public abstract class Svc extends Base implements Runnable, AutoCloseable {}"


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="DEFECT J5 (pinned deliberately): java._class_signature prepends ' extends ' "
    "and ' implements ' to node text that already starts with those keywords, emitting "
    "'class Svc extends extends Base implements implements Runnable, AutoCloseable'. "
    "The signature is EMBEDDED and lexically indexed text, so this doubles those terms' "
    "frequency in every Java class chunk with a superclass or interface list.",
)
def test_java_class_signature_does_not_duplicate_its_keywords() -> None:
    """SPEC: ``extends`` and ``implements`` appear ONCE each in a class signature.

    Asserted as exact counts rather than as one full expected string on purpose: the
    defect is the duplication, and a fix that also changed (say) whether the annotation
    block is included must still make this pin fire. A full-string assertion could fail
    for the wrong reason, stay swallowed by ``raises=AssertionError``, and leave the
    boomerang broken -- which is exactly how a strict xfail goes quiet.

    BOOMERANG: emitting each keyword once makes this XPASS.
    """
    require(
        "extends Base implements Runnable" in SIGNATURE_JAVA,
        "SIGNATURE_JAVA lost its extends and/or implements clause, so a duplicated "
        "keyword could not appear either way",
    )

    result = JavaParser().parse(SIGNATURE_JAVA, file_id=1)
    signature = next(s for s in result.symbols if s.qualified_name == "Svc").signature

    assert signature.count("extends") == 1
    assert signature.count("implements") == 1


ANNOTATED_FIELD_JAVA = """\
class Svc {
    @Autowired
    private Repo repo;
}"""


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="DEFECT J6 (pinned deliberately): an annotated field's signature is "
    "body.split('\\n')[0] and the field_declaration node starts at its modifiers, so "
    "Svc.repo is indexed with signature '@Autowired' -- no declared type, no name. "
    "These are exactly the @Autowired/@Column/@Id/@Value fields java.py surfaces on "
    "purpose, and the signature is embedded text.",
)
def test_java_annotated_field_signature_describes_the_declaration() -> None:
    """SPEC: the signature of an annotated field names its declared TYPE and NAME. The
    natural fix emits ``private Repo repo;``.

    Asserted as ``"Repo repo" in signature`` rather than as an exact string so that ANY
    acceptable fix (with or without the annotation kept as a prefix) fires the
    boomerang. An over-specified expectation here would keep failing after the fix,
    stay absorbed by ``raises=AssertionError``, and silently never fire.

    BOOMERANG: a signature that describes the declaration makes this XPASS.
    """
    require(
        "    @Autowired\n    private Repo repo;" in ANNOTATED_FIELD_JAVA,
        "ANNOTATED_FIELD_JAVA lost its annotation-then-declaration shape, which is the "
        "only shape in which body.split()[0] picks the wrong line",
    )

    result = JavaParser().parse(ANNOTATED_FIELD_JAVA, file_id=1)
    field = next(s for s in result.symbols if s.qualified_name == "Svc.repo")

    assert "Repo repo" in field.signature


# ===========================================================================
# RUST PINS
# ===========================================================================

# ZERO GAP between the `//!` and the item, deliberately. With one blank line, R7 (the
# off-by-one blank-line gap guard) ALSO detaches this comment, so fixing R7 alone would
# make this test XPASS and strict-xfail would demand the marker's removal -- spending
# R14's boomerang on a different defect's fix and losing the pin for the one it exists
# for. Adjacent, the gap guard is not involved at all, so only a fix that actually
# distinguishes `//!` from `///` can turn this green.
INNER_DOC_LEAK_RS = "//! Crate docs.\npub fn e() {}"


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="DEFECT R14 (pinned deliberately, NEW -- not in either oracle): a '//!' "
    "crate-level inner doc comment is also attached as the docstring of the first "
    "following item, because rust._get_preceding_comment accepts any line_comment and "
    "does not distinguish '//!' (documents the enclosing scope) from '///' (documents "
    "the next item). Opening a file with '//!' is idiomatic Rust, so the first symbol "
    "of most real Rust files carries the whole crate header as its embedded doc text.",
)
def test_rust_crate_inner_doc_is_not_attached_to_the_following_item() -> None:
    """SPEC: a ``//!`` comment documents the crate, never the next item.

    Neither oracle can see this: in their fixture the ``//!`` block is followed by a
    ``use`` and an ``extern crate``, so no symbol-bearing item is the comment's next
    named sibling. Hence a dedicated minimal fixture.

    BOOMERANG: ONLY skipping ``//!`` in ``_get_preceding_comment`` makes this XPASS. The
    fixture is adjacent (no blank line) precisely so that fixing R7's gap guard cannot.
    """
    require(
        INNER_DOC_LEAK_RS.startswith("//! "),
        "INNER_DOC_LEAK_RS lost its '//!' inner doc comment, so there is nothing to leak",
    )
    require(
        "docs.\npub fn e() {}" in INNER_DOC_LEAK_RS,
        "INNER_DOC_LEAK_RS must keep the `//!` line IMMEDIATELY adjacent to `pub fn e` "
        "with no blank line: a gap makes R7's fix able to satisfy this pin, which would "
        "spend R14's boomerang on the wrong defect",
    )

    result = RustParser().parse(INNER_DOC_LEAK_RS, file_id=1)
    docs = {s.qualified_name: s.docstring for s in result.symbols}

    require(
        "crate" in docs,
        "no MODULE symbol was produced, so this fixture is no longer exercising the "
        "'//!' path at all",
    )
    # The crate symbol keeps the docs; the function must not.
    assert docs["e"] is None


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="DEFECT R7 (pinned deliberately): rust._get_preceding_comment's gap guard "
    "'prev.end_point[0] + 1 < next_start_line' is one too many for Rust, because a "
    "line_comment's text includes its trailing newline so end_point already names the "
    "next line. A '///' comment separated from its item by ONE blank line is wrongly "
    "attached. The IDENTICAL guard in java.py is CORRECT there (a '/** */' block "
    "closes at '*/' on its own line), so this must be fixed in rust.py ONLY -- a "
    "uniform fix across both extractors would break the healthy one.",
)
def test_rust_doc_comment_separated_by_a_blank_line_is_detached() -> None:
    """SPEC: one blank line between a ``///`` comment and an item detaches it.

    BOOMERANG: dropping the ``+ 1`` in rust.py's guard makes this XPASS.
    """
    require(
        "/// One blank line before b.\n\npub fn b() {}" == GAP_ONE_RS,
        "GAP_ONE_RS no longer has EXACTLY one blank line before `pub fn b`, which is the "
        "only gap width at which the off-by-one is visible",
    )

    result = RustParser().parse(GAP_ONE_RS, file_id=1)
    docs = {s.qualified_name: s.docstring for s in result.symbols}

    require("b" in docs, "GAP_ONE_RS no longer yields a symbol named `b`")
    assert docs["b"] is None


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="DEFECT R8 (pinned deliberately): rust._get_preceding_comment walks "
    "prev_named_sibling and stops at the non-comment attribute_item, so a '///' doc is "
    "silently lost whenever an #[attr] sits between it and the item. Rust idiom puts "
    "attributes AFTER doc comments, so this loses the docs on most #[derive]d, "
    "#[inline]d and #[serde]d items.",
)
def test_rust_doc_comment_survives_an_attribute_between_it_and_the_item() -> None:
    """SPEC: ``/// doc`` then ``#[inline]`` then ``fn`` keeps the doc on the fn.

    BOOMERANG: skipping ``attribute_item`` while walking back makes this XPASS.
    """
    require(
        "#[inline]\npub fn d() {}" in ATTR_DOC_RS,
        "ATTR_DOC_RS lost the attribute that sits between the doc comment and the fn, "
        "which is the entire mechanism under test",
    )
    require(
        ATTR_DOC_RS.startswith("/// "),
        "ATTR_DOC_RS lost its doc comment, so 'the doc is None' would be correct",
    )

    result = RustParser().parse(ATTR_DOC_RS, file_id=1)
    symbol = next(s for s in result.symbols if s.qualified_name == "d")

    # The attribute is still collected -- so the walk DOES reach the attribute; it is
    # the comment behind it that is dropped. Asserting this rules out "the fixture has
    # no attribute" as an explanation for the failure below.
    require(
        symbol.decorators == ["#[inline]"],
        f"the attribute was not collected either (decorators={symbol.decorators!r}); "
        "this fixture is no longer isolating the doc-comment walk",
    )
    assert symbol.docstring == "Blocked by the attribute."


MULTILINE_DOC_RS = "/// Two\n/// lines.\npub fn d() {}"


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="DEFECT R9 (pinned deliberately): tree-sitter-rust's line_comment text "
    "includes its trailing newline, and rust._get_preceding_comment joins the pieces "
    "with another '\\n', so every multi-line '///' doc gains a blank line between every "
    "pair of lines. 'Two\\nlines.' is indexed as 'Two\\n\\nlines.'.",
)
def test_rust_multiline_doc_comment_has_no_spurious_blank_lines() -> None:
    """SPEC: two consecutive ``///`` lines join to ``"Two\\nlines."``.

    BOOMERANG: joining on ``""`` (or stripping the newline first) makes this XPASS.
    """
    require(
        MULTILINE_DOC_RS == "/// Two\n/// lines.\npub fn d() {}",
        "MULTILINE_DOC_RS is no longer exactly two adjacent '///' lines followed by the "
        "item, so the join under test is not exercised",
    )

    result = RustParser().parse(MULTILINE_DOC_RS, file_id=1)
    symbol = next(s for s in result.symbols if s.qualified_name == "d")

    assert symbol.docstring == "Two\nlines."


MODULE_DOC_RS = "//! Crate line one.\n//! Crate line two.\n\npub fn e() {}"


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="DEFECT R9b (pinned deliberately): the same trailing-newline quirk at a "
    "DIFFERENT code site -- rust._get_module_symbol's own '\\n'.join(inner_doc_lines) "
    "-- so the crate docstring gains a blank line between every pair of '//!' lines. "
    "Pinned separately from R9 because it is a separate join that a fix to "
    "_get_preceding_comment need not touch.",
)
def test_rust_crate_docstring_has_no_spurious_blank_lines() -> None:
    """SPEC: two consecutive ``//!`` lines join to
    ``"Crate line one.\\nCrate line two."``.

    Deliberately a SEPARATE test with a SEPARATE fixture from R9: one mutation at a
    time, and two code sites measured together is the coupled measurement that
    mis-attributed a kill in an earlier round.

    BOOMERANG: fixing ``_get_module_symbol``'s join makes this XPASS.
    """
    require(
        MODULE_DOC_RS.startswith("//! Crate line one.\n//! Crate line two.\n"),
        "MODULE_DOC_RS is no longer exactly two adjacent '//!' lines, so the join under "
        "test is not exercised",
    )

    result = RustParser().parse(MODULE_DOC_RS, file_id=1)
    crate = next(s for s in result.symbols if s.kind.name == "MODULE")

    assert crate.docstring == "Crate line one.\nCrate line two."


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="DEFECT R10 (pinned deliberately): rust._flatten_use_tree branches on "
    "'use_tree_list'/'use_tree', but the installed tree-sitter-rust emits "
    "'scoped_use_list'/'use_list', so both branches are dead and the generic 'split on "
    "::' fallback stores the literal text '{Alpha, Beta}' as an imported NAME. Brace "
    "imports are the dominant Rust import form, so the import graph loses every symbol "
    "in them.",
)
def test_rust_brace_import_is_expanded_into_its_members() -> None:
    """SPEC: ``use crate::util::{Alpha, Beta};`` yields the imported names
    ``{"Alpha", "Beta"}``.

    Asserted on the flattened NAME SET rather than on the edge tuples, so that either
    correct shape -- one edge carrying both names, or one edge per member -- fires the
    boomerang. Over-specifying the shape would leave a fix failing for the wrong
    reason, absorbed by ``raises=AssertionError``, with the pin silently dead.

    BOOMERANG: handling ``use_list`` makes this XPASS.
    """
    require(
        BRACE_USE_RS == "use crate::util::{Alpha, Beta};",
        "BRACE_USE_RS is no longer a single brace import with two members",
    )

    result = RustParser().parse(BRACE_USE_RS, file_id=1)
    names = {n for e in result.import_edges for n in e.imported_names}

    assert names == {"Alpha", "Beta"}


IMPL_ASSOC_TYPE_RS = """\
pub struct Boxy;

pub trait Draw {
    type Out;
}

impl Draw for Boxy {
    type Out = u32;
}"""


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="DEFECT R11 (pinned deliberately): rust._handle_impl routes a type_item "
    "inside an impl block through _handle_type_alias, which takes no parent, so "
    "'type Out = u32;' inside 'impl Draw for Boxy' becomes a TOP-LEVEL INTERFACE named "
    "'Out' with parent_id=None -- both a bogus top-level symbol and a lost parent link. "
    "Out/Item/Error are near-universal associated-type names, so this is also a "
    "collision source (see J4).",
)
def test_rust_associated_type_in_an_impl_block_is_scoped_to_its_type() -> None:
    """SPEC: ``type Out = u32;`` inside ``impl Draw for Boxy`` is ``Boxy::Out`` with
    ``Boxy`` as its parent -- the same shape the identical declaration gets inside the
    trait body (``Draw::Out``, which the extractor already produces correctly).

    BOOMERANG: passing the impl's parent index through makes this XPASS.
    """
    require(
        "impl Draw for Boxy {\n    type Out = u32;\n}" in IMPL_ASSOC_TYPE_RS,
        "IMPL_ASSOC_TYPE_RS lost the associated type inside its impl block",
    )
    require(
        "pub trait Draw {\n    type Out;\n}" in IMPL_ASSOC_TYPE_RS,
        "IMPL_ASSOC_TYPE_RS lost the trait-body associated type that shows the correct "
        "shape already exists, so the comparison this pin rests on is gone",
    )
    require(
        not IMPL_ASSOC_TYPE_RS.endswith("\n"),
        "IMPL_ASSOC_TYPE_RS must not end with a newline",
    )

    result = RustParser().parse(IMPL_ASSOC_TYPE_RS, file_id=1)
    rows = {
        (
            s.qualified_name,
            result.symbols[s.parent_id].qualified_name if s.parent_id is not None else None,
        )
        for s in result.symbols
        if s.name == "Out"
    }

    assert rows == {("Draw::Out", "Draw"), ("Boxy::Out", "Boxy")}


MOD_FN_RS = """\
mod inner {
    pub fn nested() {}
}"""


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="DEFECT R12 (pinned deliberately): rust._walk_top_level recurses into a "
    "mod_item's declaration_list without carrying the module name, so 'fn nested' "
    "inside 'mod inner' gets qualified_name='nested'. SAME COLLISION CLASS as J4 and "
    "as the symbols.qualified_name finding in "
    "tests/unit/test_db_scoping_and_boundaries.py: two mods in one file each with a "
    "'fn new' collide, and Indexer._insert_one keys existing_hashes on qualified_name.",
)
def test_rust_function_inside_a_module_is_qualified_by_the_module() -> None:
    """SPEC: ``fn nested`` inside ``mod inner`` is ``inner::nested`` -- the ``::``
    separator the extractor already uses for ``Type::method`` and ``Enum::Variant``.

    BOOMERANG: threading the module name through the recursion makes this XPASS.
    """
    require(
        "mod inner {" in MOD_FN_RS and "pub fn nested() {}" in MOD_FN_RS,
        "MOD_FN_RS lost its inline module or the fn inside it",
    )
    require(not MOD_FN_RS.endswith("\n"), "MOD_FN_RS must not end with a newline")

    result = RustParser().parse(MOD_FN_RS, file_id=1)
    names = {s.qualified_name for s in result.symbols}

    assert names == {"inner::nested"}


PUB_TRAIT_RS = """\
pub trait Draw {
    fn draw(&self);
}"""


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="DEFECT R13 (pinned deliberately): rust._handle_trait_fn derives is_public "
    "from a visibility_modifier that trait fns never carry (a `pub fn` inside a trait "
    "is a compile error in Rust), so every method of a `pub trait` is recorded "
    "is_public=False. Metadata only -- but it is metadata that is_public filters and "
    "ranking act on.",
)
def test_rust_method_of_a_public_trait_is_public() -> None:
    """SPEC: the methods of a ``pub trait`` are part of the public API. The correct rule
    is to INHERIT the trait's visibility, not to look for a modifier the grammar
    forbids on the item.

    BOOMERANG: inheriting the trait's visibility makes this XPASS.
    """
    require(
        PUB_TRAIT_RS.startswith("pub trait Draw {"),
        "PUB_TRAIT_RS's trait must stay `pub`, otherwise is_public=False would be the "
        "CORRECT answer and this pin would assert a falsehood",
    )
    require(
        "    fn draw(&self);" in PUB_TRAIT_RS,
        "PUB_TRAIT_RS lost its trait method",
    )

    result = RustParser().parse(PUB_TRAIT_RS, file_id=1)
    trait = next(s for s in result.symbols if s.qualified_name == "Draw")
    method = next(s for s in result.symbols if s.qualified_name == "Draw::draw")

    require(
        trait.is_public is True,
        "the trait itself was not recorded public, so `inherit the trait's visibility` "
        "would not produce True either and this pin no longer discriminates",
    )
    assert method.is_public is True
