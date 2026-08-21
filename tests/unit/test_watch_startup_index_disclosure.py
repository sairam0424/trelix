"""`trelix watch` runs a full index before it watches, and two places said it did not.

WHY THIS EXISTS. `watch`'s startup pass calls `Indexer.index()`, which reaches the
repo-wide reconcile that diffs the vector store against the `chunks` table. Measured on a
7-file fixture with the local embedder:

  * up to date, no missing vectors — "Nothing to index", **0 chunks embedded**
  * up to date, 3 vectors deleted — "Repairing 3 chunk(s)...", **3 chunks embedded**
  * never indexed                 — embeds the whole repository

So starting `watch` on a partial index repairs it and bills for it. Two places claimed
the opposite:

1. `docs/PROVIDERS.md`, in the partial-index recovery section, told readers
   "`trelix watch` will not do it — it only re-indexes files as they change, and never
   scans the store for holes." That is the section a user reads *while holding a partial
   index*, and it sent them to `trelix index` on the grounds that `watch` could not help.
2. `src/trelix/indexing/indexer.py` carried the same claim as a source comment —
   "`trelix watch`, which deliberately never reconciles" — which is presumably where the
   doc sentence came from.

Both are true of the **watching phase** (`FileWatcher` calls `index_file()` per changed
file, with no repo-wide diff) and false of the **startup pass**. That distinction is the
whole content of the fix, so it is what these tests pin: the disclosure must stay in the
help text, the false sentences must not come back, and — the load-bearing one — the
startup call must keep existing, because if it is ever removed the corrected docs become
wrong in the other direction.
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _watch_docstring() -> str:
    from trelix.cli.main import watch

    return watch.__doc__ or ""


def test_the_help_text_says_it_indexes_before_watching() -> None:
    """An operator deciding whether to run this must be told at the point of decision."""
    doc = _watch_docstring().lower()

    assert "index" in doc and ("first" in doc or "startup" in doc), (
        "watch's help no longer discloses that it indexes before watching: " + doc[:200]
    )


def test_the_help_text_distinguishes_the_free_case_from_the_paid_ones() -> None:
    """ "It indexes first" alone would read as "always expensive", which is also wrong.

    On an up-to-date index with no holes the pass embeds nothing, and saying so is what
    stops the disclosure from being scary rather than accurate.
    """
    doc = _watch_docstring().lower()

    assert "up to date" in doc, "the zero-cost case is not described"
    assert "never indexed" in doc or "whole repository" in doc, "the full-cost case is missing"
    assert "repair" in doc, "the holed-index case — the surprising one — is missing"


def test_providers_md_no_longer_claims_watch_never_scans_for_holes() -> None:
    """The exact false sentence, pinned so it cannot be reinstated."""
    text = (_REPO_ROOT / "docs" / "PROVIDERS.md").read_text(encoding="utf-8")

    assert "will not do it" not in text, (
        "docs/PROVIDERS.md again tells readers `trelix watch` will not repair a partial "
        "index; its startup pass does, and bills for it"
    )
    # Scoped to the claim about `watch`, so an accurate sentence about the WATCHING phase
    # ("never scans the store for holes again") is still allowed.
    bad = re.search(r"`trelix watch`[^.]*never scans the store for holes(?! again)", text)
    assert bad is None, f"the unqualified claim is back: {bad.group(0) if bad else ''!r}"


def test_the_indexer_comment_no_longer_claims_watch_never_reconciles() -> None:
    """Same claim, in the source comment the doc sentence appears to have come from."""
    text = (_REPO_ROOT / "src" / "trelix" / "indexing" / "indexer.py").read_text(encoding="utf-8")

    assert "deliberately never reconciles" not in text, (
        "indexer.py again asserts that `trelix watch` never reconciles; its startup "
        "index() call reaches _chunks_missing_vectors like any other index run"
    )


def test_watch_really_does_index_before_it_starts_watching() -> None:
    """The behaviour the corrected docs now promise.

    Pinned because the docs are only right while this call exists. Remove it and the
    freshly-corrected PROVIDERS.md paragraph becomes wrong in the opposite direction —
    which is exactly how the original false sentence survived: nothing tied it to code.

    Read statically rather than by invoking the command. `watch` blocks forever by design,
    and a first attempt that drove it through CliRunner hung the suite until it was killed:
    patching the watcher is not enough, because the command's own wait loop is what blocks.
    A test whose subject is "does this call precede that one" does not need to run either.
    """
    import ast
    import inspect
    import textwrap

    from trelix.cli.main import watch

    tree = ast.parse(textwrap.dedent(inspect.getsource(watch)))

    index_calls: list[int] = []
    watcher_uses: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "index":
            index_calls.append(node.lineno)
        if isinstance(func, ast.Name) and func.id == "FileWatcher":
            watcher_uses.append(node.lineno)
        if isinstance(func, ast.Attribute) and func.attr in {"start", "watch"}:
            watcher_uses.append(node.lineno)

    assert index_calls, (
        "watch() no longer calls index() at all, so its help text and the "
        "docs/PROVIDERS.md paragraph about startup repair are both now wrong"
    )
    assert watcher_uses, "watch() no longer constructs or starts a watcher — fixture is stale"
    assert min(index_calls) < min(watcher_uses), (
        "watch() now starts the watcher BEFORE indexing, so the documented "
        f"'indexes first, then watches' order is wrong: index at {index_calls}, "
        f"watcher at {watcher_uses}"
    )
