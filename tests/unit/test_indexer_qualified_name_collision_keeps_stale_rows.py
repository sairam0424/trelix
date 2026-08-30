"""END-TO-END pin for the `qualified_name` COLLISION CLASS: editing one member of a
colliding pair leaves the PRE-EDIT version of that symbol permanently in the index.

WHAT THIS FILE IS. Three separately-pinned findings all say "`symbols.qualified_name` is
not unique enough to key on":

  * ``src/trelix/store/db.py``     -- `symbols.qualified_name` is not file-qualified
                                     (``src/trelix/core/models.py:103``, ``:110``)
  * ``java.py`` defect J4         -- nested types get a BARE qualified_name
  * ``rust.py`` defect R12        -- a ``fn`` inside a ``mod`` is not module-qualified

Each of those is pinned at the level it lives at: a db helper's scoping, or one
extractor's output. NONE of them shows a user what it costs. This file is the missing
link: it drives the REAL ``Indexer`` over a REAL file with the REAL tree-sitter parser,
edits ONE symbol, and shows the other version still sitting in ``symbols`` with a live
``chunks`` row attached -- i.e. still retrievable.

THREE CORRECTIONS TO THE HANDED-DOWN DESCRIPTION, each found by re-deriving the
mechanism from the source rather than by re-running an earlier round's command.

1. THE CROSS-FILE HALF OF THE db.py FINDING DOES NOT REACH THE INCREMENTAL PATH.
   The description says the incremental path "cannot distinguish two same-named symbols
   in DIFFERENT files". Measured: it can. Every helper ``Indexer._insert_one`` uses is
   already ``WHERE file_id = ?`` scoped --
   ``get_symbol_hashes_for_file`` (db.py:917), ``delete_symbols_by_qualified_names``
   (db.py:928, ``WHERE file_id = ? AND qualified_name IN (...)``),
   ``get_chunk_ids_for_symbols`` (db.py:1529), and ``_insert_one``'s own inline
   ``SELECT qualified_name, id FROM symbols WHERE file_id = ?``. Two files each holding a
   ``main`` do NOT interfere; ``tests/unit/test_db_scoping_and_boundaries.py`` already
   pins exactly that. The collision that BITES is WITHIN ONE FILE, which is what J4 and
   R12 produce -- so this file's fixtures are single-file, and the db.py finding is
   restated below as what it actually is.

2. "NEVER INSERTED, NEVER CHUNKED, NEVER EMBEDDED" IS WRONG FOR THE FIRST INDEX.
   J4's note predicts the second colliding symbol is "declared unchanged, never
   inserted". Measured on a fresh index: ``existing_hashes`` is EMPTY, so BOTH colliding
   symbols take the ``else`` branch and BOTH are inserted. Verified below by
   ``test_first_index_stores_both_members_of_the_colliding_pair``, which is a plain
   PASSING test precisely because it contradicts the prediction. The damage is a
   RE-INDEX phenomenon, not a first-index one.

3. IT IS NOT UNBOUNDED GROWTH. A repeated no-op re-index does NOT keep adding rows:
   ``index_file`` short-circuits on the unchanged FILE hash before ``_insert_one`` runs,
   so passes 3 and 4 over an untouched file changed nothing (measured: 8 rows, 8 rows).
   The leak is one-shot per edit, not per pass.

THE MECHANISM, derived from ``indexer.py:1287-1306``
---------------------------------------------------
``existing_hashes = self.db.get_symbol_hashes_for_file(file_id)`` builds
``{qualified_name: content_hash}`` from a row list, so for a colliding pair THE LAST ROW
WINS and the first row's hash is not in the map at all. Then:

    if existing_hashes.get(symbol.qualified_name) == new_hash:
        unchanged_qualified_names.add(symbol.qualified_name)     # <-- per NAME
    ...
    qualified_names_to_delete = [
        qn for qn in existing_hashes if qn not in unchanged_qualified_names
    ]

``unchanged_qualified_names`` is a set of NAMES, not of symbols. One parsed symbol
matching is enough to mark the whole NAME unchanged, and
``delete_symbols_by_qualified_names`` is then never called for it -- so the OTHER row
under that name is never deleted, while its changed twin is inserted as an ADDITIONAL
row. The result is a symbols table with three rows for a name that has two symbols, one
of them holding source text the file no longer contains.

THE ASYMMETRY IS THE PROOF THAT "LAST ROW WINS" IS THE MECHANISM, and it is pinned
below in both directions:
  * edit the FIRST-declared member  -> its stored hash was NOT the map winner, so its
                                       own old row survives.  MEASURED 6 -> 8 rows,
                                       2 rows still carrying the pre-edit body.
  * edit the LAST-declared member   -> its stored hash WAS the map winner, both rows
                                       land in the delete list, and the pass is CLEAN.
                                       MEASURED 6 -> 6 rows, 0 stale.
An explanation that did not name "last row wins" could not predict that asymmetry, which
is why it is asserted rather than described.

MECHANICS
* The pins are ``xfail(strict=True, raises=AssertionError)``. ``raises=`` is not
  optional: without it the marker absorbs ANY exception -- a grammar upgrade that stops
  producing the collision, an ``Indexer`` refactor, a missing extractor -- and the
  boomerang silently stops working, which has already happened in four places on this
  project.
* Preconditions raise ``FixturePreconditionError``, NOT ``AssertionError``, so a fixture
  that stopped producing a collision ERRORS loudly instead of reporting a tidy
  ``xfailed`` forever. Same reason as
  ``tests/unit/test_parser_java_rust_defect_spec.py``.
* THE BOOMERANG WAS MEASURED, NOT ASSUMED, and it fires on BOTH kinds of fix -- but
  differently, so it is worth writing down which signal means what.
    - Fix an EXTRACTOR (measured: ``_handle_class`` given an ``outer`` prefix so nested
      types become ``Alpha.Config`` / ``Beta.Config``). The collision is gone, so
      ``_require`` raises ``FixturePreconditionError``, which is deliberately NOT an
      ``AssertionError`` and therefore ESCAPES ``raises=``. Measured result:
      "4 failed, 3 passed, 1 xfailed", with the error message printing the new,
      correctly-qualified name list. The Java pin, the Java grammar-fact test, the
      first-index test and the asymmetry test all go red together; that cluster is the
      release note.
    - Fix the INDEXER's keying instead, leaving the extractors alone. The collision
      still exists so the preconditions still hold, the assertions start passing, and
      ``strict=True`` turns the XPASS into a FAILURE.
    - AND THE RUST PIN STAYED ``xfailed`` THROUGH THE JAVA FIX (measured). That is the
      point of pinning the two extractors separately: a single combined test would have
      reported the Java fix as though it had also fixed Rust.
* The COLLISION ITSELF is asserted against the INSTALLED grammar as a plain passing
  test before any pin runs. A pinned-but-wrong mechanism is worse than no pin.
* A NON-COLLIDING CONTROL runs the identical edit and is asserted CLEAN as a passing
  test. Without it, "editing the first symbol leaves stale rows" would be an equally
  good explanation of the pins, and the fix would be aimed at the wrong code.
* No env claim: no optional extra, no credential, no socket, no model weight. The
  embedder and vector store are plain fakes; the tree-sitter grammars for Java and Rust
  are core dependencies (``tests/unit/test_parser_java_rust_defect_spec.py`` relies on
  them with no ``importorskip`` and is not in ``REQUIRES_EXTRA_FILES``). There is
  therefore no leaner CI install under which an assertion here is stronger than the
  environment guarantees, and nothing to skip on.
* No module-scope ``pytest.importorskip``, so this file needs no entry in
  ``tests/conftest.py``'s ``REQUIRES_EXTRA_FILES``.

DO NOT FIX FROM HERE. The fix is three source edits (see the round report's
findings_left); this file only makes the cost visible.
"""

from __future__ import annotations

import pathlib
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock, patch

from trelix.core.config import EmbedderConfig, IndexConfig, StoreConfig

_DIM = 4


class FixturePreconditionError(RuntimeError):
    """A pinning fixture has stopped discriminating.

    Deliberately NOT an AssertionError: every pin here is
    ``xfail(strict=True, raises=AssertionError)``, so an AssertionError from a
    precondition would be absorbed by the marker and the pin would report a tidy
    ``xfailed`` while measuring nothing.
    """


def _require(condition: bool, message: str) -> None:
    """Precondition that survives an xfail marker. See FixturePreconditionError."""
    if not condition:
        raise FixturePreconditionError(message)


# ---------------------------------------------------------------------------
# Fixtures. Distinct, greppable body tokens per member so a surviving row can be
# attributed to the symbol it came from -- identical bodies would make "stale
# Alpha row" and "legitimate Beta row" indistinguishable.
# ---------------------------------------------------------------------------

#: Two nested Java types that BOTH parse to the bare qualified_name "Config"
#: (defect J4), and whose methods BOTH parse to "Config.tag".
JAVA_COLLIDING_V1 = """class Alpha {
    static class Config {
        String tag() { return "ALPHA_V1"; }
    }
}
class Beta {
    static class Config {
        String tag() { return "BETA_V1"; }
    }
}
"""

#: The control: the SAME shape with the nesting removed, so every qualified_name
#: is unique. The edit applied to it is byte-for-byte the same edit.
JAVA_UNIQUE_V1 = """class AlphaConfig {
    String tag() { return "ALPHA_V1"; }
}
class BetaConfig {
    String tag() { return "BETA_V1"; }
}
"""

#: Two Rust `mod`s whose `fn`s BOTH parse to the bare qualified_name "tag"
#: (defect R12).
RUST_COLLIDING_V1 = """pub mod alpha {
    pub fn tag() -> u32 { 1001 }
}
pub mod beta {
    pub fn tag() -> u32 { 2001 }
}
"""


class _FakeEmbedder:
    """Plain fake -- no model, no network. Records nothing this file asserts on."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * _DIM for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.1] * _DIM

    async def embed_async(self, texts: list[str]) -> list[list[float]]:
        return self.embed(texts)

    @property
    def dimension(self) -> int:
        return _DIM


class _FakeVectorStore:
    """Accepts and discards. The pin is on the SQLite rows, not on vectors."""

    def upsert_batch(self, pairs: list[tuple[int, list[float]]]) -> None:
        pass

    def delete_batch(self, ids: list[int]) -> None:
        pass

    def search(self, vector: list[float], k: int) -> list[Any]:
        return []


@contextmanager
def _quiet_progress():
    bar = MagicMock()
    bar.__enter__ = MagicMock(return_value=bar)
    bar.__exit__ = MagicMock(return_value=False)
    bar.add_task = MagicMock(return_value=0)
    bar.advance = MagicMock()
    with patch("trelix.indexing.indexer.Progress", return_value=bar):
        yield bar


def _make_indexer(repo: pathlib.Path) -> Any:
    """A real Indexer with a real SQLite Database and a real parser registry.

    `Progress`, the embedder and the vector store are the only things replaced, and
    none of them is the interface under test.
    """
    from trelix.indexing.indexer import Indexer

    cfg = IndexConfig(
        repo_path=str(repo),
        incremental=False,
        store=StoreConfig(db_path=str(repo / ".trelix" / "index.db")),
        embedder=EmbedderConfig.model_construct(provider="local"),
    )
    with (
        patch("trelix.indexing.indexer.make_embedder", return_value=_FakeEmbedder()),
        patch("trelix.indexing.indexer.make_vector_store", return_value=_FakeVectorStore()),
    ):
        return Indexer(cfg, quiet=True)


class _IndexRun:
    """Result of index -> edit -> re-index, as plain counted facts."""

    def __init__(
        self,
        qualified_names_after_first_pass: list[str],
        rows_after_first_pass: int,
        rows_after_second_pass: int,
        rows_still_holding_pre_edit_text: int,
        chunks_on_those_rows: int,
    ) -> None:
        self.qualified_names_after_first_pass = qualified_names_after_first_pass
        self.rows_after_first_pass = rows_after_first_pass
        self.rows_after_second_pass = rows_after_second_pass
        self.rows_still_holding_pre_edit_text = rows_still_holding_pre_edit_text
        self.chunks_on_those_rows = chunks_on_those_rows


def _index_edit_reindex(
    tmp_path: pathlib.Path,
    filename: str,
    source_v1: str,
    edit_from: str,
    edit_to: str,
) -> _IndexRun:
    """Index *source_v1*, replace *edit_from* with *edit_to*, re-index, report counts.

    The edit is a single token substitution, so exactly ONE symbol's body changes --
    "one mutation at a time" applied to the fixture rather than to the source.
    """
    source_path = tmp_path / filename
    source_path.write_text(source_v1)
    indexer = _make_indexer(tmp_path)

    with _quiet_progress():
        indexer.index_file(str(source_path))
    first_pass_names = [
        row[0]
        for row in indexer.db._conn.execute(
            "SELECT qualified_name FROM symbols ORDER BY id"
        ).fetchall()
    ]
    rows_before = len(first_pass_names)

    _require(
        edit_from in source_v1 and source_v1.count(edit_from) == 1,
        f"the edit token {edit_from!r} must occur exactly once in the fixture",
    )
    source_path.write_text(source_v1.replace(edit_from, edit_to))
    with _quiet_progress():
        indexer.index_file(str(source_path))

    rows_after = indexer.db._conn.execute("SELECT COUNT(*) FROM symbols").fetchone()[0]
    stale_rows = indexer.db._conn.execute(
        "SELECT COUNT(*) FROM symbols WHERE body LIKE ?", (f"%{edit_from}%",)
    ).fetchone()[0]
    stale_chunks = indexer.db._conn.execute(
        "SELECT COUNT(*) FROM chunks c JOIN symbols s ON c.symbol_id = s.id WHERE s.body LIKE ?",
        (f"%{edit_from}%",),
    ).fetchone()[0]

    return _IndexRun(
        first_pass_names, int(rows_before), int(rows_after), int(stale_rows), int(stale_chunks)
    )


def _duplicated(names: list[str]) -> set[str]:
    return {n for n in names if names.count(n) > 1}


# ---------------------------------------------------------------------------
# 1. The collision exists, against the INSTALLED grammars. Plain passing tests.
# ---------------------------------------------------------------------------


class TestTheCollisionExistsInTheInstalledExtractors:
    """Asserted, not assumed. Everything below is meaningless if these fail.

    NON-DISCRIMINATING COMPANIONS by design: no source mutation is claimed to make
    these fail, because their job is to make the DIAGNOSIS falsifiable rather than to
    kill a mutant. If a grammar or extractor change removes the collision, these go red
    and every pin below is re-opened before anyone acts on it -- which is the opposite
    of the pins, whose XPASS means the defect is FIXED.

    J4 AND R12 ARE BOTH FIXED: this went red exactly as the docstring above
    predicted, for both extractors independently, which is how each half was caught
    and rewritten to assert its new, non-colliding reality (see
    ``test_java_nested_types_no_longer_collide_after_j4`` and
    ``test_rust_module_scoped_fns_no_longer_collide_after_r12``).
    """

    def test_java_nested_types_no_longer_collide_after_j4(self) -> None:
        from trelix.indexing.parser.extractors.java import JavaParser

        names = [s.qualified_name for s in JavaParser().parse(JAVA_COLLIDING_V1, 1).symbols]

        assert names == [
            "Alpha",
            "Alpha.Config",
            "Alpha.Config.tag",
            "Beta",
            "Beta.Config",
            "Beta.Config.tag",
        ]
        assert _duplicated(names) == set()

    def test_rust_module_scoped_fns_no_longer_collide_after_r12(self) -> None:
        """R12 IS FIXED: mirrors test_java_nested_types_no_longer_collide_after_j4."""
        from trelix.indexing.parser.extractors.rust import RustParser

        names = [s.qualified_name for s in RustParser().parse(RUST_COLLIDING_V1, 1).symbols]

        assert names == ["alpha::tag", "beta::tag"]
        assert _duplicated(names) == set()

    def test_the_control_fixture_has_no_colliding_qualified_name(self) -> None:
        """The control must actually be a control."""
        from trelix.indexing.parser.extractors.java import JavaParser

        names = [s.qualified_name for s in JavaParser().parse(JAVA_UNIQUE_V1, 1).symbols]

        assert names == ["AlphaConfig", "AlphaConfig.tag", "BetaConfig", "BetaConfig.tag"]
        assert _duplicated(names) == set()


# ---------------------------------------------------------------------------
# 2. Correction 2: the FIRST index stores both. Plain passing test.
# ---------------------------------------------------------------------------


class TestFirstIndexIsNotWhereTheDamageHappens:
    """Contradicted J4's stated consequence back when the collision was live.

    J4's original note predicted the second colliding symbol would be "declared
    unchanged, never inserted, never chunked, never embedded -- silently
    unsearchable". Re-derived from ``indexer.py:1287``: on a fresh index
    ``existing_hashes`` is ``{}``, so ``existing_hashes.get(qn)`` is ``None`` for BOTH
    members and both take the ``else: changed_local_indices.add(...)`` branch. Both
    were inserted even under the collision.

    J4 IS FIXED, so ``Alpha.Config``/``Beta.Config`` no longer share a
    ``qualified_name`` at all -- there is no longer a collision for a first index to
    be "not the site of the damage" for. This now just confirms the first-index path
    still stores both members correctly under their (now distinct) qualified names.
    """

    def test_first_index_stores_both_members_under_their_distinct_names(
        self, tmp_path: pathlib.Path
    ) -> None:
        source_path = tmp_path / "Svc.java"
        source_path.write_text(JAVA_COLLIDING_V1)
        indexer = _make_indexer(tmp_path)
        with _quiet_progress():
            indexer.index_file(str(source_path))

        rows = indexer.db._conn.execute(
            "SELECT qualified_name, body FROM symbols ORDER BY id"
        ).fetchall()
        names = [r[0] for r in rows]

        assert names.count("Alpha.Config") == 1
        assert names.count("Beta.Config") == 1
        assert names.count("Alpha.Config.tag") == 1
        assert names.count("Beta.Config.tag") == 1
        # Both bodies are present, so neither member was skipped as "unchanged".
        bodies = "\n".join(r[1] for r in rows)
        assert "ALPHA_V1" in bodies
        assert "BETA_V1" in bodies


# ---------------------------------------------------------------------------
# 3. The control: the identical edit on a NON-colliding file is clean.
# ---------------------------------------------------------------------------


class TestTheSameEditOnANonCollidingFileIsClean:
    """Rules out "editing the first-declared symbol leaves stale rows" as the cause.

    Without this, the pins below have two candidate explanations and a fixer could
    aim at ``index_file``'s change detection instead of at qualified-name scoping.
    MEASURED: 4 rows -> 4 rows, 0 stale.
    """

    def test_no_stale_row_survives_when_every_qualified_name_is_unique(
        self, tmp_path: pathlib.Path
    ) -> None:
        run = _index_edit_reindex(tmp_path, "Svc.java", JAVA_UNIQUE_V1, "ALPHA_V1", "ALPHA_V2")

        _require(
            _duplicated(run.qualified_names_after_first_pass) == set(),
            "the control fixture must contain NO colliding qualified_name; it now "
            f"contains {_duplicated(run.qualified_names_after_first_pass)}",
        )

        assert run.rows_after_first_pass == 4
        assert run.rows_after_second_pass == 4
        assert run.rows_still_holding_pre_edit_text == 0
        assert run.chunks_on_those_rows == 0


# ---------------------------------------------------------------------------
# 4. THE PINS. Strict xfail, raises=AssertionError, correct behaviour asserted.
# ---------------------------------------------------------------------------


class TestEditingOneMemberOfACollidingPairLeavesTheOldVersionIndexed:
    """DEFECT (pinned deliberately) -- the user-visible cost of the collision class.

    HOW THIS GOES RED WHEN THE DEFECT IS FIXED, stated as measured rather than as
    predicted -- the two routes give DIFFERENT signals and both are correct:
      * fix an EXTRACTOR -> the collision disappears, ``_require`` raises
        ``FixturePreconditionError``, which escapes ``raises=AssertionError`` and ERRORS
        the test, alongside a red grammar-fact test above.
      * fix the INDEXER's keying -> the collision remains, so the preconditions still
        hold, the assertions below start passing, and ``strict=True`` turns that XPASS
        into a FAILURE.
    Either way the marker must be deleted, and that deletion is the release note. See
    the module docstring for the measured output of both.
    """

    def test_java_editing_the_first_declared_member_removes_its_old_body(
        self, tmp_path: pathlib.Path
    ) -> None:
        """J4 IS FIXED via the extractor route: the collision disappeared, so the
        collision-existence precondition below was removed (it correctly ERRORed
        with FixturePreconditionError when this test still had it, per the class
        docstring). The remaining assertions were already stated as the CORRECT
        behaviour and now pass because Alpha.Config/Beta.Config are genuinely
        distinct rows -- editing one can no longer touch the other's data at all.
        """
        run = _index_edit_reindex(tmp_path, "Svc.java", JAVA_COLLIDING_V1, "ALPHA_V1", "ALPHA_V2")

        _require(
            run.rows_after_first_pass == 6,
            f"expected 6 symbols on the first pass, got {run.rows_after_first_pass}",
        )

        # CORRECT behaviour, stated as the assertion. Today: 8, 2 and 2.
        assert run.rows_still_holding_pre_edit_text == 0, (
            "the pre-edit body ALPHA_V1 is still stored after the edit"
        )
        assert run.chunks_on_those_rows == 0, (
            "the pre-edit body still has a chunks row, so it is still retrievable"
        )
        assert run.rows_after_second_pass == 6, "editing one symbol must not grow the symbols table"

    def test_rust_editing_the_first_declared_member_removes_its_old_body(
        self, tmp_path: pathlib.Path
    ) -> None:
        """R12 IS FIXED via the extractor route: alpha::tag/beta::tag no longer
        collide, so the collision-existence precondition below was removed. The
        remaining assertions were already stated as the CORRECT behaviour and now
        pass because the two functions are genuinely distinct rows.
        """
        run = _index_edit_reindex(tmp_path, "lib.rs", RUST_COLLIDING_V1, "1001", "1002")

        _require(
            run.rows_after_first_pass == 2,
            f"expected 2 symbols on the first pass, got {run.rows_after_first_pass}",
        )

        # CORRECT behaviour. Today: 3, 1 and 1.
        assert run.rows_still_holding_pre_edit_text == 0, (
            "the pre-edit body `1001` is still stored after the edit"
        )
        assert run.chunks_on_those_rows == 0
        assert run.rows_after_second_pass == 2


class TestTheDamageIsAsymmetricWhichNamesTheMechanism:
    """ "Last row wins" in ``get_symbol_hashes_for_file``'s dict comprehension.

    ``{row[0]: row[1] for row in rows}`` (db.py:926) keeps the LAST row's hash for a
    duplicated name. So editing the LAST-declared member makes its stored hash the one
    that no longer matches, BOTH rows land in ``qualified_names_to_delete``, and the
    pass is clean. Editing the FIRST-declared member does not. That asymmetry was the
    signature of this mechanism and no other -- while the Java collision was live.

    J4 IS FIXED: Alpha.Config/Beta.Config no longer collide at all, so there is no
    more asymmetry to demonstrate on the Java fixture (editing either member now
    trivially leaves nothing stale, since they were never the same qualified_name to
    begin with). This is kept as a plain regression check on the fixed behaviour
    rather than removed. The still-live counterpart on the Rust side is
    ``test_rust_editing_the_first_declared_member_removes_its_old_body`` above.
    """

    def test_editing_the_last_declared_member_leaves_nothing_stale(
        self, tmp_path: pathlib.Path
    ) -> None:
        run = _index_edit_reindex(tmp_path, "Svc.java", JAVA_COLLIDING_V1, "BETA_V1", "BETA_V2")

        _require(
            _duplicated(run.qualified_names_after_first_pass) == set(),
            "the Java fixture is expected to be collision-free after the J4 fix; it "
            f"still produced duplicate qualified_names in {run.qualified_names_after_first_pass}",
        )

        assert run.rows_after_first_pass == 6
        assert run.rows_after_second_pass == 6
        assert run.rows_still_holding_pre_edit_text == 0
        assert run.chunks_on_those_rows == 0
