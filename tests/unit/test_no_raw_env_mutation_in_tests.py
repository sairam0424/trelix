"""No test may mutate ``os.environ`` outside ``monkeypatch`` / ``patch.dict``.

WHY. A raw ``os.environ[...] = ...`` is not scoped to the test that writes it: the
value is still there for every test that runs afterwards, in whatever order the
runner picked. ``try: ... finally: os.environ.pop(...)`` does not fix it -- ``pop``
DELETES the key, so if the operator's environment (or tests/_env_isolation.py) had
already put a value there, the "cleanup" destroys it instead of restoring it.
``monkeypatch`` and ``patch.dict`` record the prior state, including "was absent".

tests/unit/test_model_aware_budget.py::test_env_var_parsing was the raw assignment
in tests/unit; it is now ``monkeypatch.setenv``. This file finds the NEXT one.

NOT A COUNTER. It compares the sites found under tests/ against an EXPLICIT table
keyed by ``(path, kind)`` with a count, with equality BOTH ways: a new mutation is
an undeclared-site failure, and an entry for a site that no longer exists is a
stale-entry failure, so the table cannot rot into permission for something already
cleaned up. Line numbers are deliberately not part of the key -- they churn on
unrelated edits above the site, and a guard that cries wolf gets deleted.

LIMIT, DECLARED RATHER THAN LEFT FOR THE NEXT READER TO ASSUME. The receiver match is
NAME-EXACT: it looks for ``os.environ`` written that way. ``import os as _os`` then
``_os.environ[...] = ...``, or ``from os import environ as env`` then ``env[...] = ...``,
both escape it, as does reaching the mapping through an alias assigned at runtime. No
occurrence of any of those exists under tests/ today (checked), and resolving aliases
properly needs import-graph tracking this guard deliberately does not attempt. So this
finds the ordinary form, which is the form that has actually appeared here twice -- it is
not proof that no raw mutation exists.

WHY A MATCHER SELF-CHECK. The file reduces to "the scan found nothing
unexpected", which is also what it reports if the AST matcher silently stops
matching or the glob returns nothing. ``TestTheScannerActuallyDetects`` runs the
same scanner over inline source with known answers, so a broken matcher fails
loudly there instead of turning this file into a rubber stamp.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_TESTS_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _TESTS_ROOT.parent

# A broken glob reports "no unexpected mutations" just as loudly as a clean tree.
_MIN_FILES_SCANNED = 80

# `from os import environ` is the obvious way past a check that only knows
# "os.environ".
_ENVIRON_RECEIVERS = frozenset({"os.environ", "environ"})

# Mutating methods. get/keys/items/copy are reads and are deliberately absent.
_MUTATING_METHODS = frozenset({"__setitem__", "update", "setdefault", "pop", "popitem", "clear"})

# Functions that write the environment behind os.environ's back.
_MUTATING_OS_FUNCTIONS = frozenset({"os.putenv", "os.unsetenv"})

# ---------------------------------------------------------------------------
# The deliberate sites. Each one needs a reason, not just an entry.
# ---------------------------------------------------------------------------
_ALLOWED: dict[tuple[str, str], int] = {
    # The isolation layer itself: sets LITELLM_MODE at IMPORT time because litellm
    # reads that flag while IT imports, so a fixture would run far too late. There
    # is nothing to restore it to -- it must hold for the whole process.
    ("tests/_env_isolation.py", "subscript-assign"): 1,
    # HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE, also at import time and deliberately
    # `setdefault`, so a developer debugging a real Hub-fetch problem can export
    # HF_HUB_OFFLINE=0 and be obeyed.
    ("tests/unit/conftest.py", "setdefault-call"): 2,
    # TOKENIZERS_PARALLELISM for the real-model sparse test; setdefault, and the
    # value only silences a warning.
    ("tests/unit/test_sparse_padding_contamination.py", "setdefault-call"): 1,
    # Inside `with patch.dict(os.environ, ...)`, which restores the whole mapping
    # on exit -- scoped by the context manager, not raw.
    ("tests/unit/test_retrieval_breadth_floor.py", "pop-call"): 1,
    # KNOWN and deliberately NOT converted: two raw assignments with `finally:
    # pop`, in the credential-gated integration suite, which cannot be executed in
    # an offline environment -- converting them would ship an unverified edit to a
    # test nobody here can run. Counted exactly, so a THIRD one still fails.
    ("tests/integration/test_indexer.py", "subscript-assign"): 2,
    ("tests/integration/test_indexer.py", "pop-call"): 2,
}


def _mutations_in(source: str, label: str) -> list[tuple[str, str, int]]:
    """Return ``(label, kind, lineno)`` for every os.environ mutation in *source*."""
    found: list[tuple[str, str, int]] = []
    tree = ast.parse(source, filename=label)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign | ast.AugAssign | ast.AnnAssign):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if (
                    isinstance(target, ast.Subscript)
                    and ast.unparse(target.value) in _ENVIRON_RECEIVERS
                ):
                    found.append((label, "subscript-assign", node.lineno))
        elif isinstance(node, ast.Delete):
            for target in node.targets:
                if (
                    isinstance(target, ast.Subscript)
                    and ast.unparse(target.value) in _ENVIRON_RECEIVERS
                ):
                    found.append((label, "del-subscript", node.lineno))
        elif isinstance(node, ast.Call):
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr in _MUTATING_METHODS
                and ast.unparse(func.value) in _ENVIRON_RECEIVERS
            ):
                found.append((label, f"{func.attr.strip('_')}-call", node.lineno))
            elif ast.unparse(func) in _MUTATING_OS_FUNCTIONS:
                found.append((label, f"{ast.unparse(func).split('.')[-1]}-call", node.lineno))
    # ast.walk is breadth-first, so `found` is in tree order, not source order.
    return sorted(found, key=lambda site: (site[2], site[1]))


def _scan_tests_tree() -> tuple[dict[tuple[str, str], int], int, list[tuple[str, str, int]]]:
    counts: dict[tuple[str, str], int] = {}
    sites: list[tuple[str, str, int]] = []
    scanned = 0
    for path in sorted(_TESTS_ROOT.rglob("*.py")):
        rel = path.relative_to(_REPO_ROOT).as_posix()
        scanned += 1
        for label, kind, lineno in _mutations_in(path.read_text(), rel):
            counts[label, kind] = counts.get((label, kind), 0) + 1
            sites.append((label, kind, lineno))
    return counts, scanned, sites


class TestTheScannerActuallyDetects:
    """PRECONDITIONS for TestNoUndeclaredEnvMutation, not claims of their own.

    Each fails if ``_mutations_in`` stops matching -- the failure mode that would
    otherwise turn the real assertion green forever.
    """

    def test_a_raw_subscript_assignment_is_reported(self) -> None:
        source = "import os\ndef t():\n    os.environ['X'] = '1'\n"
        assert _mutations_in(source, "inline") == [("inline", "subscript-assign", 3)]

    def test_the_bare_environ_import_form_is_reported_too(self) -> None:
        source = "from os import environ\ndef t():\n    environ['X'] = '1'\n"
        assert _mutations_in(source, "inline") == [("inline", "subscript-assign", 3)]

    def test_setitem_update_pop_del_and_putenv_are_reported(self) -> None:
        source = (
            "import os\n"
            "def t():\n"
            "    os.environ.__setitem__('A', '1')\n"
            "    os.environ.update({'B': '2'})\n"
            "    os.environ.pop('C', None)\n"
            "    del os.environ['D']\n"
            "    os.putenv('E', '3')\n"
        )
        assert [(kind, line) for _, kind, line in _mutations_in(source, "inline")] == [
            ("setitem-call", 3),
            ("update-call", 4),
            ("pop-call", 5),
            ("del-subscript", 6),
            ("putenv-call", 7),
        ]

    def test_the_sanctioned_forms_are_not_reported(self) -> None:
        """monkeypatch/patch.dict restore, and reads are not mutations.

        Without this, a matcher that flagged EVERYTHING would still pass the three
        tests above while making the real assertion unmaintainable.
        """
        source = (
            "import os\n"
            "from unittest.mock import patch\n"
            "def t(monkeypatch):\n"
            "    monkeypatch.setenv('A', '1')\n"
            "    monkeypatch.delenv('B', raising=False)\n"
            "    with patch.dict(os.environ, {'C': '2'}):\n"
            "        pass\n"
            "    value = os.environ.get('D')\n"
            "    other = os.environ['E']\n"
            "    return value, other\n"
        )
        assert _mutations_in(source, "inline") == []


class TestNoUndeclaredEnvMutation:
    """Adding a raw ``os.environ['X'] = ...`` to any file under tests/ is the
    mutation that must make this fail."""

    def test_the_walk_reached_the_whole_tests_tree(self) -> None:
        """PRECONDITION: an empty walk reports a clean tree."""
        _, scanned, _ = _scan_tests_tree()
        assert scanned >= _MIN_FILES_SCANNED, (
            f"only {scanned} python files were scanned under {_TESTS_ROOT}; the walk is "
            "broken, so the assertion below would be vacuous"
        )

    def test_every_environ_mutation_under_tests_is_declared(self) -> None:
        counts, _, sites = _scan_tests_tree()
        undeclared = {k: v for k, v in counts.items() if k not in _ALLOWED}
        assert undeclared == {}, (
            "raw os.environ mutation(s) with no entry in _ALLOWED:\n"
            + "\n".join(
                f"  {path}:{line} [{kind}]"
                for path, kind, line in sites
                if (path, kind) in undeclared
            )
            + "\nUse monkeypatch.setenv/delenv (or patch.dict) instead. If the site is "
            "genuinely process-wide, add it to _ALLOWED with the reason."
        )

    def test_no_declared_site_has_gone_stale_or_grown(self) -> None:
        """Equality the other way: _ALLOWED must not outlive its sites.

        Made to fail by deleting an allowed mutation without deleting its entry, or
        by adding a second mutation of an already-allowed kind to that file.
        """
        counts, _, _ = _scan_tests_tree()
        mismatched = {k: n for k, n in _ALLOWED.items() if counts.get(k, 0) != n}
        assert mismatched == {}, (
            "_ALLOWED disagrees with the tree; entries are (path, kind) -> expected "
            f"count, actual counts are { ({k: counts.get(k, 0) for k in mismatched}) }. "
            "Remove the stale entry, or justify the new site."
        )


def test_the_allowed_table_and_the_scan_agree_on_the_total() -> None:
    """A site counted by neither direction above would mean the keying is wrong."""
    counts, _, sites = _scan_tests_tree()
    assert sum(counts.values()) == len(sites)
    assert sum(_ALLOWED.values()) == len(sites), (
        f"_ALLOWED declares {sum(_ALLOWED.values())} mutations, the scan found {len(sites)}: "
        + ", ".join(f"{p}:{ln}[{k}]" for p, k, ln in sites)
    )


if __name__ == "__main__":  # pragma: no cover - developer convenience
    raise SystemExit(pytest.main([__file__]))
