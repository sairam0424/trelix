"""Test-quality rule 1, enforced: never let the module under test SUPPLY the
expected value.

WHY. ``assert db.schema_version() == SCHEMA_VERSION`` cannot fail when
``SCHEMA_VERSION`` changes, because both sides move together. The literal is the
whole point: a test that writes ``== 4`` fails the moment the stamp becomes 5 and
forces a human to decide whether that was intended. Rule 1 has been enforced by
review alone; this file enforces it mechanically. The regression it generalises is
recorded in tests/unit/test_provider_prompt_provenance.py's docstring -- a wrong
prompt string survived 28 tagged releases because the tests asserted
``encoded_text.startswith(_QUERY_INSTRUCTION)``, "deriving the expected value FROM
the constant under test, so they passed for any value at all".

THE DISTINCTION THIS FILE EXISTS TO DRAW, because the pattern is NOT always a defect
=================================================================================
tests/unit/test_audit_hash_chain_columns.py imports ``_CONTENT_COLUMNS`` from the
module under test ON PURPOSE, and review accepted the argument. A blanket ban would
forbid it; a blanket allowlist of every current site would make this file a rubber
stamp. So the rule is neither. It separates two uses of an imported constant:

  SELECTS (legitimate) -- the import enumerates WHAT to check. Its *membership* is
      the operand; the asserted VALUE is still a literal or an independently
      obtained observation. Adding a column to the DDL then adds a test case
      automatically instead of going silently untested. Three shapes:
        @pytest.mark.parametrize("column", _CONTENT_COLUMNS)   -- not in an assert
        assert hashable == set(_CONTENT_COLUMNS)               -- membership view
        assert intent in INTENT_STRATEGIES                     -- container side
  SUPPLIES (the defect) -- the import IS the expected value, and nothing else in
      the assertion pins it. Both sides move together:
        assert db.schema_version() == SCHEMA_VERSION
        assert plan.strategy is INTENT_STRATEGIES[intent]
        assert digest.startswith(f"{_DIGEST_SCHEME}:")

Encoded as: a constant reference is exempt when its syntactic role is a MEMBERSHIP
VIEW (``set``/``frozenset``/``sorted``/``list``/``tuple``/``dict``/``len`` of it, or
``.keys()``/``.values()``/``.items()``), a LITERAL INDEX (``EXTENSION_MAP[".py"]`` --
the test wrote down which entry), an INPUT ARGUMENT (passed *into* the code under
test, e.g. ``AuditEvent(action=ACTION_ADMIN)``), or the CONTAINER side of
``in``/``not in``. It is a finding when the role is a bare value operand with no
literal anchor, or a DYNAMIC INDEX (``INTENT_STRATEGIES[intent]`` -- the test looked
up whatever the module has and compared it to whatever the module produced), or an
argument to a string predicate (``startswith``/``endswith``/...).

That rule alone takes the tree from 95 sites / 50 (file, constant) pairs down to 37
/ 20 without one allowlist entry, and it clears test_audit_hash_chain_columns.py to
ZERO -- see ``test_the_reviewed_counter_example_is_clean_by_rule``, which is the
proof that this is a rule and not a stamp. Removing one exemption (the membership
view) makes that test fail with 2 sites in the counter-example, which is how the
clean result was verified to be caused rather than coincidental.

WHAT A LITERAL ANCHOR IS, precisely, because the positional detail is what makes the
rule work. A ``Constant`` in VALUE position anywhere in the comparison, or a name
bound to a literal at module/class scope IN THE TEST FILE (the
``BGE_CODE_COSQA_INSTRUCTION = "..."`` idiom, which is the sanctioned form and must
not be flagged). Deliberately NOT anchors, each because it pins something other than
the value:
  * a literal used as a subscript index -- ``ev["outcome"] == OUTCOME_SUCCESS`` pins
    the key;
  * a literal passed to a non-builtin call -- ``Database(tmp_path / "index.db")``
    names a file, it does not pin a schema version;
  * a literal in arithmetic ON the constant -- ``[_MAX_JWKS_BYTES + 1]`` pins "one
    past the cap", and holds for any cap.
That third one is not a guess: this repo already reached it by hand. See the
``SCHEMA_VERSION`` entry in ``_ALLOWED``.

NOT A COUNTER, same shape as tests/unit/test_no_raw_env_mutation_in_tests.py. The
sites are compared against an EXPLICIT table keyed ``(path, constant, kind)`` with a
count, equality BOTH ways: a new site is an undeclared-site failure, and an entry for
a site that no longer exists is a stale-entry failure, so the table ratchets down and
cannot rot into permission for something already fixed. Line numbers are not part of
the key -- they churn on unrelated edits above the site.

THE NUMBER, RE-DERIVED RATHER THAN INHERITED. This work was handed "35 (test-file,
constant) pairs across 68 sites". Measured on this tree: **20 pairs / 37 sites across
13 files** under the definition above (236 .py files walked). Thirteen other
definitions were measured and none produces 35/68 either; the count is dominated by
the definition, not by the tree:
    any trelix constant anywhere in an assert                57 pairs / 110 sites
    ... in an assert comparison (any position)               50 / 95
    ... comparison with an equality or identity operator     43 / 79
    ... constant is a bare top-level operand                 25 / 38
    THIS FILE'S RULE                                         20 / 37
    ... bare top-level operand and no literal anywhere         8 / 12
So 35/68 is bracketed by these but reproduced by none of them, and the inherited
figure is not used anywhere below. Anyone re-deriving it must state which of these
they mean; "68 sites" alone does not identify a measurement.

DECLARED LIMITS, rather than left for the next reader to assume.
  * CONSTANTS ONLY. A name is a candidate only if it is UPPER_SNAKE (leading
    underscores ignored) and imported by ``from trelix... import NAME``. So
    ``assert first["entry_hash"] == _canonical_hash(GENESIS_HASH, content)``
    (tests/unit/test_audit_store.py:88) is NOT caught: the expected value there is
    RECOMPUTED by calling the module's own function, and no syntactic rule can tell
    a recompute of the expected side from an ordinary read of the actual side --
    both are just calls. It is reported in the round summary instead of pinned here.
  * ``import trelix.x`` then ``trelix.x.CONST`` escapes it, as does
    ``from trelix.y import CONST as c`` where the alias is not UPPER_SNAKE. Neither
    occurs under tests/ today (checked); resolving them needs the import-graph
    tracking this guard deliberately does not attempt.
  * NO DATAFLOW. The rule reads one ``assert`` statement plus the test file's own
    module/class-level literal constants. A literal that reaches the comparison
    through a local variable is not seen, so such a site must be declared.

WHY A MATCHER SELF-CHECK. This file reduces to "the scan found nothing undeclared",
which is also what it reports if the matcher silently stops matching or the glob
returns nothing. ``TestTheScannerActuallyDetects`` runs the same scanner over inline
source with known answers -- including the audit-chain shape, which must come back
EMPTY -- so a broken matcher fails loudly there instead of turning this file green.
"""

from __future__ import annotations

import ast
import builtins
from functools import lru_cache
from pathlib import Path

import pytest

_TESTS_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _TESTS_ROOT.parent

# A broken glob reports "no undeclared sites" just as loudly as a clean tree. 236
# .py files live under tests/ today (this one included); the floor is well below that
# because it must hold on every CI leg. Safe to assert at all: tests/ ships in the
# checkout on all four legs, and this whole file imports nothing but the stdlib and
# pytest, so it cannot pass locally for a reason CI's leaner install lacks.
_MIN_FILES_SCANNED = 200

_BUILTIN_NAMES = frozenset(dir(builtins))

# Wrapping a constant in one of these yields its MEMBERSHIP, not its values: the
# import is enumerating cases. `len` is here for the same reason -- a count is a
# property of the collection, not of what it maps to.
_MEMBERSHIP_VIEWS = frozenset({"set", "frozenset", "sorted", "list", "tuple", "dict", "len"})
_VIEW_METHODS = frozenset({"keys", "values", "items"})

# `assert x.startswith(CONST)` is the exact shape that hid the bge-code prompt bug.
_STR_PREDICATES = frozenset(
    {"startswith", "endswith", "removeprefix", "removesuffix", "count", "index", "find", "split"}
)

_KIND_UNANCHORED = "unanchored-value"
_KIND_DYNAMIC = "dynamic-lookup"
_KIND_PREDICATE = "predicate-arg"

# ---------------------------------------------------------------------------
# The declared sites. Each needs a REASON, not just an entry, and each is a debt
# this table is meant to shrink -- not a permission slip.
# ---------------------------------------------------------------------------
_ALLOWED: dict[tuple[str, str, str], int] = {
    # --- wire strings the tests never pin ---------------------------------
    # `assert ev["outcome"] == OUTCOME_SUCCESS`. The API writes the event using
    # this same constant, so renaming the stored value "success" -> "ok" keeps
    # every one of these green while breaking every consumer of the audit log.
    # What they DO discriminate is which branch ran, which is why they are debt
    # rather than nonsense: the fix is to write the wire string as a literal.
    ("tests/unit/test_api_audit.py", "OUTCOME_SUCCESS", _KIND_UNANCHORED): 3,
    ("tests/unit/test_api_audit.py", "OUTCOME_DENIED", _KIND_UNANCHORED): 2,
    ("tests/unit/test_api_audit.py", "OUTCOME_ERROR", _KIND_UNANCHORED): 1,
    ("tests/unit/test_api_audit.py", "ACTION_SEARCH", _KIND_UNANCHORED): 2,
    ("tests/unit/test_cli_audit.py", "OUTCOME_DENIED", _KIND_UNANCHORED): 1,
    # Same shape for the verify-failure reasons. Note the CONTRAST inside these
    # files, which is why the rule's literal-anchor test matters: the sibling
    # assertions `_verify(db) == (4, REASON_COUNT_MISMATCH)` carry the row id as a
    # literal and are correctly NOT flagged; these bare `reason == REASON_...`
    # forms carry nothing.
    ("tests/unit/test_audit_anchor_presence.py", "REASON_COUNT_MISMATCH", _KIND_UNANCHORED): 1,
    ("tests/unit/test_audit_anchor_presence.py", "REASON_ANCHOR_MISSING", _KIND_UNANCHORED): 1,
    ("tests/unit/test_audit_anchor_presence.py", "REASON_ANCHOR_CORRUPT", _KIND_UNANCHORED): 2,
    # `assert {_META_COUNT, _META_HEAD} <= keys` -- a presence check on the two
    # anchor key NAMES. Closest thing here to a legitimate cross-check, but the
    # key names are the module's own and the row is written by the module, so a
    # rename moves both sides. Two literals would fix it.
    ("tests/unit/test_audit_anchor_presence.py", "_META_COUNT", _KIND_UNANCHORED): 1,
    ("tests/unit/test_audit_anchor_presence.py", "_META_HEAD", _KIND_UNANCHORED): 1,
    # --- values whose whole contract is the number/string -----------------
    # `assert db.schema_version() == SCHEMA_VERSION` (x2) and
    # `assert str(SCHEMA_VERSION + 1) in str(exc.value)`. INDEPENDENTLY CONFIRMED,
    # which is why this entry is the calibration point for the whole rule: an
    # earlier round reached the same verdict about this same file by hand and wrote
    # tests/unit/test_db_schema_version_pinned.py to compensate. That file's
    # docstring says so outright -- "every assertion there is written in terms of
    # SCHEMA_VERSION imported from the module under test (`SCHEMA_VERSION + 1`,
    # `== SCHEMA_VERSION`), so changing the constant to any other value" leaves them
    # green -- and it pins the stamp against the literal 1 read back with a raw
    # PRAGMA. So the CONTRACT is covered elsewhere and these three are residual
    # debt, not an open hole. Their listing of `SCHEMA_VERSION + 1` as defective is
    # also what set this rule's treatment of offset literals.
    ("tests/unit/test_db_structural.py", "SCHEMA_VERSION", _KIND_UNANCHORED): 3,
    # `assert seen.get("pattern") == TICKET_PATTERN_DEFAULT`. The most legible
    # instance: the line two above it writes `== r"#\d+"` and the line below
    # writes `== 5_000`, so this one test pins two of its three defaults with
    # literals and the third with the import.
    ("tests/unit/test_git_linker.py", "TICKET_PATTERN_DEFAULT", _KIND_UNANCHORED): 1,
    # `assert kwargs.get("prompt_name") == _QUERY_PROMPT_NAME`. The published
    # prompt name IS pinned as the literal "query" in the same file, so the
    # contract is covered; this site is a pass-through check that happens to
    # spell it with the import.
    ("tests/unit/test_embedder_nomic.py", "_QUERY_PROMPT_NAME", _KIND_UNANCHORED): 1,
    # `assert str(_MAX_JWKS_BYTES) in message` -- the claim is "the error names
    # the cap", which the import cannot break; the cap's VALUE is unpinned here.
    # Plus two `read_sizes == [_MAX_JWKS_BYTES + 1]`: "read one byte past the cap"
    # is the right claim, and it survives any change to the cap itself.
    ("tests/unit/test_oidc.py", "_MAX_JWKS_BYTES", _KIND_UNANCHORED): 3,
    # `assert len(warnings) == _MAX_FAILURE_DETAIL_LOGS + 1` -- same offset form:
    # the "+ 1" is the summary line, and the log budget is unpinned.
    ("tests/unit/test_connector_error_binding.py", "_MAX_FAILURE_DETAIL_LOGS", _KIND_UNANCHORED): 1,
    # `walk_config_history == (_UNKNOWN_DIGEST,)` and `_UNKNOWN_DIGEST in
    # history`. The module writes the sentinel and the test reads it back, so the
    # sentinel's own text is untested; the structural claim (a history entry
    # appears at all) does hold.
    ("tests/unit/test_provenance.py", "_UNKNOWN_DIGEST", _KIND_UNANCHORED): 3,
    ("tests/unit/test_reviewer_context_fallback.py", "_NO_CONTEXT_LABEL", _KIND_UNANCHORED): 2,
    # --- dynamic lookup: the tautology in its clearest form ---------------
    # `assert plan.strategy is INTENT_STRATEGIES[intent]`. The planner PRODUCES
    # plan.strategy by looking up that very mapping, so the assertion reduces to
    # `INTENT_STRATEGIES[intent] is INTENT_STRATEGIES[intent]`. Change any
    # intent's strategy and all three stay green. The file's own
    # `test_all_intent_types_covered` is the legitimate use of the same import
    # and is correctly not flagged.
    ("tests/unit/test_planner.py", "INTENT_STRATEGIES", _KIND_DYNAMIC): 2,
    ("tests/unit/test_planner_coverage.py", "INTENT_STRATEGIES", _KIND_DYNAMIC): 1,
    # --- string predicates: the shape that hid the bge-code bug ------------
    # `assert digest.startswith(f"{_DIGEST_SCHEME}:")` -- the literal parts of
    # the f-string pin the FORMAT; the scheme name itself ("sha256" -> "md5")
    # could change under all three and none would notice.
    ("tests/unit/test_drift_honesty.py", "_DIGEST_SCHEME", _KIND_PREDICATE): 3,
    # `assert c.comment.endswith(_NO_CONTEXT_LABEL)` -- the label the reviewer
    # appends when it had no context; its text is the user-visible contract and
    # is nowhere written down.
    ("tests/unit/test_reviewer_context_fallback.py", "_NO_CONTEXT_LABEL", _KIND_PREDICATE): 2,
}

# The rule must keep finding all three kinds. If a future refactor made one of
# them unreachable, the scan would still "agree with _ALLOWED" for the other two
# while silently enforcing nothing for the third.
_KINDS_THAT_MUST_STILL_FIRE = frozenset({_KIND_UNANCHORED, _KIND_DYNAMIC, _KIND_PREDICATE})

# The reviewed counter-example. Zero findings here is a RULE outcome, not a table
# entry -- if it ever needs an entry, the rule stopped drawing the distinction.
_COUNTER_EXAMPLE = "tests/unit/test_audit_hash_chain_columns.py"

# The file that documents the historical defect and demonstrates the correct form
# (imported constant compared against a snapshot literal declared in the test).
_CORRECT_FORM_EXEMPLAR = "tests/unit/test_provider_prompt_provenance.py"


# ---------------------------------------------------------------------------
# The scanner
# ---------------------------------------------------------------------------
def _is_constant_name(name: str) -> bool:
    core = name.lstrip("_")
    return bool(core) and core.upper() == core and any(char.isalpha() for char in core)


def _imported_trelix_constants(tree: ast.AST) -> set[str]:
    """Names bound by ``from trelix... import NAME`` that look like constants.

    ANY SCOPE, and that is load-bearing rather than incidental. Measured: 10 files
    under tests/ import a trelix constant ONLY inside a function or class body
    (test_bm25, test_cli_serve_exposure_warning, test_conditional_ignore_dirs,
    test_drift_honesty, test_embedder_bge, test_embedder_nomic, test_git_linker,
    test_provenance, test_provider_prompt_provenance, test_retriever_core). Four of
    the thirteen files this rule flags are in that list, so a module-scope-only scan
    would silently under-report by a third and still look like a clean tree.
    """
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".")[0] != "trelix":
                continue
            for alias in node.names:
                local = alias.asname or alias.name
                if _is_constant_name(local):
                    found.add(local)
    return found


def _is_literal_expression(node: ast.AST | None) -> bool:
    """A value written down in the source, including containers and arithmetic."""
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.Tuple | ast.List | ast.Set):
        return all(_is_literal_expression(element) for element in node.elts)
    if isinstance(node, ast.Dict):
        return all(_is_literal_expression(key) for key in node.keys) and all(
            _is_literal_expression(value) for value in node.values
        )
    if isinstance(node, ast.BinOp):
        return _is_literal_expression(node.left) and _is_literal_expression(node.right)
    if isinstance(node, ast.UnaryOp):
        return _is_literal_expression(node.operand)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        return node.func.id in _MEMBERSHIP_VIEWS and all(
            _is_literal_expression(arg) for arg in node.args
        )
    return False


def _test_file_literal_constants(tree: ast.AST) -> set[str]:
    """``NAME = <literal>`` at module or class scope IN THE TEST FILE.

    This is the sanctioned form -- a snapshot value written down under a name and
    a comment saying where it came from -- so it must count as a literal anchor.
    Without it, test_provider_prompt_provenance.py, the file that DOCUMENTS this
    defect class, would be reported as an instance of it.
    """
    found: set[str] = set()
    scopes: list[ast.AST] = [tree, *(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef))]
    for scope in scopes:
        for statement in getattr(scope, "body", []):
            if isinstance(statement, ast.Assign):
                targets: list[ast.expr] = list(statement.targets)
                value: ast.expr | None = statement.value
            elif isinstance(statement, ast.AnnAssign):
                targets, value = [statement.target], statement.value
            else:
                continue
            if not _is_literal_expression(value):
                continue
            for target in targets:
                if isinstance(target, ast.Name) and _is_constant_name(target.id):
                    found.add(target.id)
    return found


def _parent_map(root: ast.AST) -> dict[int, ast.AST]:
    parents: dict[int, ast.AST] = {}
    for node in ast.walk(root):
        for child in ast.iter_child_nodes(node):
            parents[id(child)] = node
    return parents


def _reference_role(name_node: ast.Name, parents: dict[int, ast.AST]) -> str:
    """What the assertion DOES with this constant reference.

    ``view``/``index-literal``/``input-arg`` mean the import is selecting or
    feeding; ``index-dynamic`` and ``whole`` mean it is supplying.
    """
    node: ast.AST = name_node
    while True:
        parent = parents.get(id(node))
        if parent is None:
            return "whole"
        if isinstance(parent, ast.Subscript):
            if parent.value is node:
                return "index-literal" if _is_literal_expression(parent.slice) else "index-dynamic"
            return "input-arg"  # the constant is an index INTO something else
        if isinstance(parent, ast.Attribute):
            if parent.attr in _VIEW_METHODS:
                return "view"
            if parent.attr == "get":
                call = parents.get(id(parent))
                if isinstance(call, ast.Call) and call.func is parent:
                    first = call.args[0] if call.args else None
                    return "index-literal" if _is_literal_expression(first) else "index-dynamic"
            return "whole"
        if isinstance(parent, ast.Call):
            is_argument = node in parent.args or any(kw.value is node for kw in parent.keywords)
            if not is_argument:
                return "whole"
            func = parent.func
            if isinstance(func, ast.Name) and func.id in _MEMBERSHIP_VIEWS:
                return "view"
            if isinstance(func, ast.Name) and func.id in _BUILTIN_NAMES:
                node = parent  # a transparent builtin like str(); keep looking outward
                continue
            return "input-arg"  # passed INTO the code under test
        if isinstance(parent, ast.BinOp | ast.UnaryOp | ast.Tuple | ast.List | ast.Set | ast.Dict):
            node = parent
            continue
        return "whole"


def _has_literal_anchor(compare: ast.Compare, file_literals: set[str], constants: set[str]) -> bool:
    parents = _parent_map(compare)

    def is_in_value_position(node: ast.AST) -> bool:
        current: ast.AST = node
        while True:
            parent = parents.get(id(current))
            if parent is None:
                return True
            if isinstance(parent, ast.Subscript) and parent.slice is current:
                return False  # pins a key, never a value
            if isinstance(parent, ast.BinOp):
                # `_MAX_JWKS_BYTES + 1` -- the literal is an OFFSET FROM the constant,
                # so it pins the relationship ("one past the cap") and not the cap.
                # Change the cap and the assertion still holds. This repo reached the
                # same conclusion by hand: test_db_schema_version_pinned.py exists
                # because test_db_structural.py writes `SCHEMA_VERSION + 1`, and its
                # docstring lists that form alongside `== SCHEMA_VERSION` as the reason
                # the stamp needed pinning against a literal `1` somewhere else.
                other = parent.right if parent.left is current else parent.left
                if any(isinstance(n, ast.Name) and n.id in constants for n in ast.walk(other)):
                    return False
            if isinstance(parent, ast.Call) and (
                current in parent.args or any(kw.value is current for kw in parent.keywords)
            ):
                func = parent.func
                if not (isinstance(func, ast.Name) and func.id in _BUILTIN_NAMES):
                    return False  # an argument to the code under test, not an expectation
            current = parent

    for node in ast.walk(compare):
        if isinstance(node, ast.Constant) and node.value is not None:
            if is_in_value_position(node):
                return True
        elif isinstance(node, ast.Name) and node.id in file_literals:
            if is_in_value_position(node):
                return True
    return False


def _container_side_constants(compare: ast.Compare, constants: set[str]) -> set[str]:
    """Constants on the CONTAINER side of ``in``/``not in``: pure enumerators."""
    operands = [compare.left, *compare.comparators]
    found: set[str] = set()
    for index, operator in enumerate(compare.ops):
        if isinstance(operator, ast.In | ast.NotIn):
            right = operands[index + 1]
            found |= {
                n.id for n in ast.walk(right) if isinstance(n, ast.Name) and n.id in constants
            }
    return found


def findings_in(source: str, label: str) -> list[tuple[str, str, str, int]]:
    """``(label, constant, kind, lineno)`` for every SUPPLIES site in *source*."""
    tree = ast.parse(source, filename=label)
    constants = _imported_trelix_constants(tree)
    if not constants:
        return []
    file_literals = _test_file_literal_constants(tree)
    found: list[tuple[str, str, str, int]] = []

    for assertion in [n for n in ast.walk(tree) if isinstance(n, ast.Assert)]:
        for compare in [n for n in ast.walk(assertion.test) if isinstance(n, ast.Compare)]:
            names = [n for n in ast.walk(compare) if isinstance(n, ast.Name)]
            hits = [n for n in names if n.id in constants]
            if not hits:
                continue
            # No independent operand at all means there is no expected value to
            # get from the wrong place: `len(_C) == len(set(_C))` asserts a
            # structural invariant OF the constant, which is a different claim.
            if not ({n.id for n in names} - constants - _BUILTIN_NAMES - file_literals):
                continue
            anchored = _has_literal_anchor(compare, file_literals, constants)
            enumerators = _container_side_constants(compare, constants)
            parents = _parent_map(compare)
            for hit in hits:
                role = _reference_role(hit, parents)
                if role in {"input-arg", "index-literal", "view"}:
                    continue
                if role == "index-dynamic":
                    found.append((label, hit.id, _KIND_DYNAMIC, assertion.lineno))
                elif hit.id not in enumerators and not anchored:
                    found.append((label, hit.id, _KIND_UNANCHORED, assertion.lineno))

        for call in [n for n in ast.walk(assertion.test) if isinstance(n, ast.Call)]:
            func = call.func
            if not (isinstance(func, ast.Attribute) and func.attr in _STR_PREDICATES):
                continue
            receiver = {n.id for n in ast.walk(func.value) if isinstance(n, ast.Name)}
            for argument in call.args:
                for node in ast.walk(argument):
                    if isinstance(node, ast.Name) and node.id in constants:
                        if node.id not in receiver:
                            found.append((label, node.id, _KIND_PREDICATE, assertion.lineno))

    deduplicated = sorted(set(found), key=lambda site: (site[3], site[1], site[2]))
    return deduplicated


@lru_cache(maxsize=1)
def _scan_tests_tree() -> tuple[
    dict[tuple[str, str, str], int], int, tuple[tuple[str, str, str, int], ...]
]:
    counts: dict[tuple[str, str, str], int] = {}
    sites: list[tuple[str, str, str, int]] = []
    scanned = 0
    for path in sorted(_TESTS_ROOT.rglob("*.py")):
        relative = path.relative_to(_REPO_ROOT).as_posix()
        scanned += 1
        for label, constant, kind, lineno in findings_in(path.read_text(), relative):
            key = (label, constant, kind)
            counts[key] = counts.get(key, 0) + 1
            sites.append((label, constant, kind, lineno))
    return counts, scanned, tuple(sites)


# ---------------------------------------------------------------------------
# Matcher self-check: preconditions, not claims of their own
# ---------------------------------------------------------------------------
class TestTheScannerActuallyDetects:
    """Each of these fails if ``findings_in`` stops drawing the SELECTS/SUPPLIES
    line -- the failure mode that would otherwise turn the real assertion below
    green forever, in either direction."""

    def test_the_canonical_defect_is_reported(self) -> None:
        source = (
            "from trelix.store.database import SCHEMA_VERSION\n"
            "def test_stamp(db):\n"
            "    assert db.schema_version() == SCHEMA_VERSION\n"
        )
        assert findings_in(source, "inline") == [("inline", "SCHEMA_VERSION", _KIND_UNANCHORED, 3)]

    def test_a_dynamic_lookup_into_the_pinned_mapping_is_reported(self) -> None:
        source = (
            "from trelix.retrieval.planner import INTENT_STRATEGIES\n"
            "def test_strategy(plan, intent):\n"
            "    assert plan.strategy is INTENT_STRATEGIES[intent]\n"
        )
        assert findings_in(source, "inline") == [("inline", "INTENT_STRATEGIES", _KIND_DYNAMIC, 3)]

    def test_the_startswith_form_that_hid_the_prompt_bug_is_reported(self) -> None:
        source = (
            "from trelix.embedder.bge_code import _QUERY_INSTRUCTION\n"
            "def test_prefix(encoded_text):\n"
            "    assert encoded_text.startswith(_QUERY_INSTRUCTION)\n"
        )
        assert findings_in(source, "inline") == [
            ("inline", "_QUERY_INSTRUCTION", _KIND_PREDICATE, 3)
        ]

    def test_a_literal_key_does_not_launder_an_unpinned_value(self) -> None:
        """``ev["outcome"]`` carries a string literal, but it pins the KEY. A
        matcher that counted any literal at all would clear this whole class.
        """
        source = (
            "from trelix.audit.events import OUTCOME_SUCCESS\n"
            "def test_outcome(ev):\n"
            '    assert ev["outcome"] == OUTCOME_SUCCESS\n'
        )
        assert findings_in(source, "inline") == [("inline", "OUTCOME_SUCCESS", _KIND_UNANCHORED, 3)]

    def test_an_offset_from_the_constant_does_not_launder_it(self) -> None:
        """``[_MAX_JWKS_BYTES + 1]`` pins "one past the cap", never the cap. This
        repo reached the same verdict by hand -- see the ``SCHEMA_VERSION`` entry in
        ``_ALLOWED`` -- and this case is what keeps the rule matching it.
        """
        source = (
            "from trelix.auth.oidc import _MAX_JWKS_BYTES\n"
            "def test_cap(response):\n"
            "    assert response.read_sizes == [_MAX_JWKS_BYTES + 1]\n"
        )
        assert findings_in(source, "inline") == [("inline", "_MAX_JWKS_BYTES", _KIND_UNANCHORED, 3)]

    def test_a_literal_beside_an_unrelated_name_is_still_an_anchor(self) -> None:
        """The other direction for the offset rule: only arithmetic ON the constant
        is disqualified. A matcher that dropped every literal inside any BinOp would
        re-flag the sanctioned form.
        """
        source = (
            "from trelix.llm.prompt import MIN_FENCE_LENGTH\n"
            "def test_fence(observed):\n"
            "    assert MIN_FENCE_LENGTH == observed + 3\n"
        )
        assert findings_in(source, "inline") == []

    def test_a_constructor_argument_does_not_launder_an_unpinned_value(self) -> None:
        """``Database(tmp_path / "index.db")`` names a file. If a literal in a
        call argument counted as an anchor, the canonical defect would go clean
        the moment the fixture was inlined.
        """
        source = (
            "from trelix.store.database import SCHEMA_VERSION, Database\n"
            "def test_stamp(tmp_path):\n"
            '    db = Database(tmp_path / "index.db")\n'
            "    assert db.schema_version() == SCHEMA_VERSION\n"
        )
        assert findings_in(source, "inline") == [("inline", "SCHEMA_VERSION", _KIND_UNANCHORED, 4)]

    # --- the other direction: legitimate uses must come back EMPTY ---------
    def test_the_literal_form_the_rule_wants_is_not_reported(self) -> None:
        source = (
            "from trelix.llm.prompt import MIN_FENCE_LENGTH\n"
            "def test_fence():\n"
            "    assert MIN_FENCE_LENGTH == 3\n"
        )
        assert findings_in(source, "inline") == []

    def test_a_snapshot_literal_declared_in_the_test_file_is_an_anchor(self) -> None:
        """The sanctioned form: the published value written down under a name.
        Without this, test_provider_prompt_provenance.py -- the file that
        documents this defect class -- would be reported as an instance of it.
        """
        source = (
            "from trelix.embedder.bge_code import _QUERY_INSTRUCTION\n"
            'BGE_PUBLISHED = "Given a web search query, retrieve relevant code."\n'
            "def test_matches_published():\n"
            "    assert _QUERY_INSTRUCTION == BGE_PUBLISHED\n"
        )
        assert findings_in(source, "inline") == []

    def test_an_input_argument_is_not_an_expected_value(self) -> None:
        source = (
            "from trelix.audit.events import ACTION_ADMIN, AuditEvent\n"
            "def test_append(store):\n"
            "    assert store.append(AuditEvent(action=ACTION_ADMIN)) is True\n"
        )
        assert findings_in(source, "inline") == []

    def test_a_structural_invariant_of_the_constant_is_not_reported(self) -> None:
        source = (
            "from trelix.audit.store import _CONTENT_COLUMNS\n"
            "def test_no_duplicates():\n"
            "    assert len(_CONTENT_COLUMNS) == len(set(_CONTENT_COLUMNS))\n"
        )
        assert findings_in(source, "inline") == []

    def test_the_audit_chain_shape_is_not_reported(self) -> None:
        """THE COUNTER-EXAMPLE, as source. All three of its uses of the imported
        constant are SELECTS: parametrize enumerates the cases, the comprehension
        enumerates the keys to compare, and the set-equality compares its
        MEMBERSHIP against a set obtained from the live DDL. Every asserted VALUE
        is a literal or an independent observation, so the whole shape is clean by
        rule -- no allowlist entry anywhere.
        """
        source = (
            "import pytest\n"
            "from trelix.audit.store import _CONTENT_COLUMNS\n"
            '_UNHASHED = frozenset({"id", "prev_hash", "entry_hash"})\n'
            '@pytest.mark.parametrize("column", _CONTENT_COLUMNS)\n'
            "def test_each_column(db, column):\n"
            '    assert _verify(db) == (3, "row_mutated")\n'
            "def test_columns_cover_the_schema(conn):\n"
            '    schema = {r[1] for r in conn.execute("PRAGMA table_info(audit_log)")}\n'
            "    hashable = schema - _UNHASHED\n"
            "    assert hashable == set(_CONTENT_COLUMNS)\n"
            "    assert set(_CONTENT_COLUMNS) == hashable\n"
            "def test_predecessor_is_committed_to(first, second):\n"
            "    content_first = {c: first[c] for c in _CONTENT_COLUMNS}\n"
            "    content_second = {c: second[c] for c in _CONTENT_COLUMNS}\n"
            "    assert content_first == content_second\n"
        )
        assert findings_in(source, "inline") == []

    def test_container_side_membership_is_not_reported(self) -> None:
        source = (
            "from trelix.retrieval.planner import INTENT_STRATEGIES\n"
            "def test_covered(IntentType):\n"
            "    for intent in IntentType:\n"
            "        assert intent in INTENT_STRATEGIES\n"
        )
        assert findings_in(source, "inline") == []

    def test_a_literal_index_into_the_pinned_table_is_not_reported(self) -> None:
        """``EXTENSION_MAP[".py"] == Language.PYTHON`` restates one row of the
        table with the key written down, so remapping ``.py`` fails it. That is a
        different claim from ``MAP[runtime_value] == module_output``.
        """
        source = (
            "from trelix.indexing.walker import EXTENSION_MAP\n"
            "def test_python(Language):\n"
            '    assert EXTENSION_MAP[".py"] == Language.PYTHON\n'
        )
        assert findings_in(source, "inline") == []

    def test_a_non_trelix_constant_is_out_of_scope(self) -> None:
        """A matcher keyed on "any UPPER_SNAKE import" would flag stdlib names and
        make the check unmaintainable.
        """
        source = (
            "from http import HTTPStatus\n"
            "from logging import WARNING\n"
            "def test_status(response, record):\n"
            "    assert response.status_code == HTTPStatus.OK\n"
            "    assert record.levelno == WARNING\n"
        )
        assert findings_in(source, "inline") == []


# ---------------------------------------------------------------------------
# The real assertions
# ---------------------------------------------------------------------------
class TestNoUndeclaredImportedExpectedValue:
    """Adding ``assert <something from the module> == <CONSTANT imported from it>``
    to any file under tests/ is the mutation that must make this fail."""

    def test_the_walk_reached_the_whole_tests_tree(self) -> None:
        """PRECONDITION: an empty walk reports a clean tree just as loudly."""
        _, scanned, _ = _scan_tests_tree()
        assert scanned >= _MIN_FILES_SCANNED, (
            f"only {scanned} python files were scanned under {_TESTS_ROOT}; the walk is "
            "broken, so the assertions below would be vacuous"
        )

    def test_every_imported_expected_value_is_declared(self) -> None:
        counts, _, sites = _scan_tests_tree()
        undeclared = {key: n for key, n in counts.items() if key not in _ALLOWED}
        assert undeclared == {}, (
            "assertion(s) whose expected value comes from the module under test, with no "
            "entry in _ALLOWED:\n"
            + "\n".join(
                f"  {path}:{line} [{kind}] {constant}"
                for path, constant, kind, line in sites
                if (path, constant, kind) in undeclared
            )
            + "\nWrite the expected value as a literal. If the import is genuinely there to "
            "ENUMERATE cases rather than to supply the value, use its membership "
            "(set(CONST), CONST.keys(), `x in CONST`) or index it with a literal, and the "
            "rule will clear it without an entry -- see "
            f"{_COUNTER_EXAMPLE} for the reviewed precedent."
        )

    def test_no_declared_site_has_gone_stale_or_grown(self) -> None:
        """Equality the other way, so the table ratchets DOWN.

        Made to fail by fixing a declared site without deleting its entry, or by
        adding another site of an already-declared (file, constant, kind).
        """
        counts, _, _ = _scan_tests_tree()
        mismatched = {key: n for key, n in _ALLOWED.items() if counts.get(key, 0) != n}
        assert mismatched == {}, (
            "_ALLOWED disagrees with the tree; entries are (path, constant, kind) -> "
            "expected count, actual counts are "
            f"{ ({key: counts.get(key, 0) for key in mismatched}) }. Remove the stale entry "
            "if the site was fixed, or justify the new one."
        )

    def test_the_table_and_the_scan_agree_on_the_total(self) -> None:
        """A site counted by neither direction above would mean the keying is wrong."""
        counts, _, sites = _scan_tests_tree()
        assert sum(counts.values()) == len(sites)
        assert sum(_ALLOWED.values()) == len(sites), (
            f"_ALLOWED declares {sum(_ALLOWED.values())} sites, the scan found {len(sites)}"
        )


class TestTheRuleAndNotAnAllowlistDrawsTheLine:
    """The distinction is load-bearing, so it is pinned against the real files.

    If these ever need an ``_ALLOWED`` entry, the rule stopped separating "the
    import enumerates the cases" from "the import supplies the expected value",
    and this file became the rubber stamp it is meant not to be.
    """

    def test_the_reviewed_counter_example_is_clean_by_rule(self) -> None:
        counts, _, _ = _scan_tests_tree()
        offending = {key: n for key, n in counts.items() if key[0] == _COUNTER_EXAMPLE}
        assert offending == {}, (
            f"{_COUNTER_EXAMPLE} imports _CONTENT_COLUMNS from the module under test "
            "DELIBERATELY, and review accepted the argument: the import enumerates the "
            "columns to check while every asserted value stays a literal or an "
            f"independent observation. The rule now reports {offending} there, so it is "
            "forbidding the reviewed pattern instead of the defect."
        )
        assert (_TESTS_ROOT.parent / _COUNTER_EXAMPLE).is_file(), (
            f"{_COUNTER_EXAMPLE} is gone, so the assertion above passes vacuously"
        )

    def test_the_counter_example_really_does_import_from_the_module_under_test(self) -> None:
        """PRECONDITION for the test above, naming the fixture it depends on.

        Without this, deleting the import (or the whole file) would make "clean by
        rule" true by construction rather than by the rule working.
        """
        source = (_TESTS_ROOT.parent / _COUNTER_EXAMPLE).read_text()
        constants = _imported_trelix_constants(ast.parse(source, filename=_COUNTER_EXAMPLE))
        assert "_CONTENT_COLUMNS" in constants, (
            f"{_COUNTER_EXAMPLE} no longer imports _CONTENT_COLUMNS from "
            "trelix.audit.store, so it is no longer the counter-example this rule was "
            f"designed against; it imports {sorted(constants)}"
        )

    def test_the_correct_form_exemplar_is_clean_by_rule(self) -> None:
        """The file that DOCUMENTS this defect class compares imported constants
        against snapshot literals declared at its own module scope. Flagging it
        would mean the rule cannot tell the fix from the bug.
        """
        counts, _, _ = _scan_tests_tree()
        offending = {key: n for key, n in counts.items() if key[0] == _CORRECT_FORM_EXEMPLAR}
        assert offending == {}, (
            f"{_CORRECT_FORM_EXEMPLAR} pins each provider prompt against a literal "
            f"snapshot of the model's published config; the rule reports {offending}, so "
            "the literal-anchor test has stopped recognising the sanctioned form."
        )
        exemplar = _TESTS_ROOT.parent / _CORRECT_FORM_EXEMPLAR
        assert exemplar.is_file(), f"{_CORRECT_FORM_EXEMPLAR} is gone; assertion was vacuous"
        constants = _imported_trelix_constants(ast.parse(exemplar.read_text()))
        assert "_QUERY_INSTRUCTION" in constants, (
            "the exemplar no longer imports _QUERY_INSTRUCTION, so 'clean by rule' above "
            "holds by construction rather than by the anchor test working"
        )

    def test_all_three_kinds_are_still_reachable_on_this_tree(self) -> None:
        """PRECONDITION: a kind with no sites enforces nothing, and the two
        equality assertions above would still agree.
        """
        counts, _, _ = _scan_tests_tree()
        live = {kind for _, _, kind in counts}
        assert live == _KINDS_THAT_MUST_STILL_FIRE, (
            f"kinds firing on this tree: {sorted(live)}; expected "
            f"{sorted(_KINDS_THAT_MUST_STILL_FIRE)}. A kind that reaches zero sites is "
            "either fully paid off (delete it from _KINDS_THAT_MUST_STILL_FIRE and say so) "
            "or silently broken."
        )


if __name__ == "__main__":  # pragma: no cover - developer convenience
    raise SystemExit(pytest.main([__file__]))
