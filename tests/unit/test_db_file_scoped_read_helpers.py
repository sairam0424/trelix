"""Structural guard: EVERY read helper in store/db.py that takes a scoping key must scope.

Round 4 killed three mutants in `db.py` (`get_symbol_hashes_for_file`,
`get_chunk_ids_for_symbols`, `get_symbol_ids_for_file_id` each losing their
`WHERE file_id = ?`) and then observed that the three are not three bugs — they are
three instances of ONE latent defect class that nothing guarded structurally:

    a read helper that accepts a file_id and then does not constrain its query by it.

Writing a fourth point test would leave the class open. This file closes it by shape:

  1. `TestEveryFileIdHelperIsAccountedFor` DISCOVERS, by reflection over
     `Database.__dict__`, every method that takes a `file_id` parameter, and asserts
     set-equality (both directions) against the hand-written tables below. A helper
     added tomorrow is not silently skipped — the discovery test FAILS and whoever
     added it must either give it a scoping probe or justify it as a non-reader.
     `TestEveryPathKeyedHelperIsAccountedFor` does the same for the path-keyed readers.

  2. `TestEveryFileIdReaderScopesToItsFile` runs each probe in `_FILE_ID_READER_PROBES`
     against a two-file index in which BOTH files contain symbols with the SAME
     qualified_names and the same shape, differing only in body text. Every probe
     collapses the helper's return value to the set of OWNER TAGS it observed, and the
     assertion is `probe(ix, "A") == {"A"}` and `probe(ix, "B") == {"B"}`, both
     directions of set equality. A helper that ignores its file_id returns both owners
     (or, for the `fetchone()` helpers, always the first file) and fails.

Note on rule 2 (never iterate the collection you pin): what is enumerated here are
FUNCTIONS, discovered from the class. The EXPECTATIONS are hand-written — the probe
tables are literal dicts typed out below, the expected owner sets are the literals
`{"A"}` and `{"B"}`, and nothing is derived from `db.py`'s own answers.

Why the collision fixture is the only one that discriminates. `symbols.qualified_name`
is `ClassName.method_name` (`Symbol`, core/models.py) — it is NOT file-qualified. So
`main`, `__init__`, `handler` and `Widget.render` collide across files in every real
repo. `Indexer._insert_one` decides what to re-embed from
`db.get_symbol_hashes_for_file(file_id)`; the moment that map leaks another file's
rows, a symbol in the file being indexed is declared unchanged because a DIFFERENT
file holds the same qualified_name — never inserted, never chunked, never embedded,
silently unsearchable while `trelix stats` still counts it. Two files with disjoint
symbol names would make every unfiltered query look correct.

That non-file-qualified `qualified_name` is itself pinned, as a defect, by
`TestQualifiedNameIsNotFileQualified` at the bottom of this file.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest

from trelix.core.models import ImportEdge, IndexedFile, Language, Symbol, SymbolKind
from trelix.store.db import Database

# Symbol names deliberately shared by BOTH files in the fixture.
_SHARED_QUALIFIED_NAMES = ["Widget", "main"]


@dataclass(frozen=True)
class TwoFileIndex:
    """A real (on-disk, sqlite) index with two colliding files, plus ground-truth
    ownership maps recorded by the fixture's OWN queries — never by the helpers
    under test."""

    db: Database
    file_of: dict[str, int]  # "A" -> file id of the A-side file, "B" -> B-side
    symbol_owner: dict[int, str]  # symbol id -> "A" | "B"
    chunk_owner: dict[int, str]  # chunk id  -> "A" | "B"
    file_owner: dict[int, str]  # file id   -> "A" | "B"  (imported files included)
    rel_path_owner: dict[str, str]  # rel_path  -> "A" | "B"
    hash_owner: dict[str, str]  # files.hash -> "A" | "B"
    content_hash_owner: dict[str, str]  # symbols.content_hash -> "A" | "B"
    summary_owner: dict[str, str]  # file_summaries.summary -> "A" | "B"
    import_target_of: dict[str, int]  # "A" -> file id that ONLY A imports


def _insert_file(db: Database, rel_path: str, file_hash: str) -> int:
    return db.upsert_file(
        IndexedFile(
            path=f"/repo/{rel_path}",
            rel_path=rel_path,
            language=Language.PYTHON,
            hash=file_hash,
            size_bytes=17,
        )
    )


def _insert_symbol(db: Database, file_id: int, qualified_name: str, body: str) -> int:
    kind = SymbolKind.CLASS if qualified_name == "Widget" else SymbolKind.FUNCTION
    symbol_id = db.insert_symbol(
        Symbol(
            file_id=file_id,
            name=qualified_name.split(".")[-1],
            qualified_name=qualified_name,
            kind=kind,
            line_start=1,
            line_end=4,
            signature=f"def {qualified_name}()",
            body=body,
        )
    )
    db._conn.commit()
    return int(symbol_id)


@pytest.fixture()
def two_file_index(tmp_path: Path) -> TwoFileIndex:
    """Two files, identical symbol names, distinct bodies; A imports only c.py and
    B imports only d.py; each file has its own summary and its own file hash."""
    db = Database(tmp_path / "index.db")

    file_a = _insert_file(db, "pkg/a.py", "filehash-A")
    file_b = _insert_file(db, "pkg/b.py", "filehash-B")
    file_c = _insert_file(db, "pkg/c.py", "filehash-C")
    file_d = _insert_file(db, "pkg/d.py", "filehash-D")

    symbol_owner: dict[int, str] = {}
    chunk_owner: dict[int, str] = {}
    for tag, file_id in (("A", file_a), ("B", file_b)):
        for qualified_name in _SHARED_QUALIFIED_NAMES:
            symbol_id = _insert_symbol(
                db, file_id, qualified_name, body=f"def {qualified_name}(): return {tag!r}"
            )
            symbol_owner[symbol_id] = tag
            chunk_id = db.insert_chunk_for_symbol(symbol_id, f"chunk text owner={tag}", 4)
            chunk_owner[int(chunk_id)] = tag

    # Import edges: A -> c.py, B -> d.py. imported_file_id is written by the
    # fixture's own SQL so this file never depends on resolve_import_file_ids().
    db.insert_imports([ImportEdge(file_id=file_a, imported_from="pkg.c", imported_names=["x"])])
    db.insert_imports([ImportEdge(file_id=file_b, imported_from="pkg.d", imported_names=["y"])])
    db._conn.execute("UPDATE imports SET imported_file_id = ? WHERE file_id = ?", (file_c, file_a))
    db._conn.execute("UPDATE imports SET imported_file_id = ? WHERE file_id = ?", (file_d, file_b))
    db._conn.commit()

    db.upsert_file_summary(file_a, "file summary owner=A")
    db.upsert_file_summary(file_b, "file summary owner=B")

    # Ground truth read back with the fixture's own SELECTs.
    content_hash_owner: dict[str, str] = {}
    for row in db._conn.execute("SELECT id, content_hash FROM symbols").fetchall():
        content_hash_owner[str(row[1])] = symbol_owner[int(row[0])]

    return TwoFileIndex(
        db=db,
        file_of={"A": file_a, "B": file_b},
        symbol_owner=symbol_owner,
        chunk_owner=chunk_owner,
        # c.py belongs to A's world (only A imports it); d.py to B's.
        file_owner={file_a: "A", file_c: "A", file_b: "B", file_d: "B"},
        rel_path_owner={"pkg/a.py": "A", "pkg/c.py": "A", "pkg/b.py": "B", "pkg/d.py": "B"},
        hash_owner={"filehash-A": "A", "filehash-C": "A", "filehash-B": "B", "filehash-D": "B"},
        content_hash_owner=content_hash_owner,
        summary_owner={"file summary owner=A": "A", "file summary owner=B": "B"},
        import_target_of={"A": file_c, "B": file_d},
    )


# ---------------------------------------------------------------------------
# Probes. Each returns the set of OWNER TAGS the helper's answer exposes when
# asked for `want`'s data. Correct answer is always exactly {want}.
# ---------------------------------------------------------------------------

_Probe = Callable[[TwoFileIndex, str], set[str]]


def _probe_symbol_ids_for_file_id(ix: TwoFileIndex, want: str) -> set[str]:
    ids = ix.db.get_symbol_ids_for_file_id(ix.file_of[want])
    return {ix.symbol_owner[int(i)] for i in ids}


def _probe_symbols_for_file(ix: TwoFileIndex, want: str) -> set[str]:
    symbols = ix.db.get_symbols_for_file(ix.file_of[want])
    return {ix.symbol_owner[int(s.id)] for s in symbols if s.id is not None}


def _probe_all_symbols_for_file(ix: TwoFileIndex, want: str) -> set[str]:
    ids = ix.db.get_all_symbols_for_file(ix.file_of[want])
    return {ix.symbol_owner[int(i)] for i in ids}


def _probe_top_symbols_for_file(ix: TwoFileIndex, want: str) -> set[str]:
    # limit far above the fixture's symbol count, so truncation cannot hide a leak.
    ids = ix.db.get_top_symbols_for_file(ix.file_of[want], limit=99)
    return {ix.symbol_owner[int(i)] for i in ids}


def _probe_symbol_hashes_for_file(ix: TwoFileIndex, want: str) -> set[str]:
    # Returns {qualified_name: content_hash}. The two files share every
    # qualified_name, so an unfiltered query does NOT change the key set — it
    # changes the VALUES (the later rowid wins the dict comprehension). Owners are
    # therefore read off the hashes, which is what Indexer._insert_one compares.
    hashes = ix.db.get_symbol_hashes_for_file(ix.file_of[want])
    return {ix.content_hash_owner[str(h)] for h in hashes.values()}


def _probe_chunk_ids_for_file(ix: TwoFileIndex, want: str) -> set[str]:
    ids = ix.db.get_chunk_ids_for_file(ix.file_of[want])
    return {ix.chunk_owner[int(i)] for i in ids}


def _probe_chunk_ids_for_symbols(ix: TwoFileIndex, want: str) -> set[str]:
    ids = ix.db.get_chunk_ids_for_symbols(ix.file_of[want], list(_SHARED_QUALIFIED_NAMES))
    return {ix.chunk_owner[int(i)] for i in ids}


def _probe_imports_for_file(ix: TwoFileIndex, want: str) -> set[str]:
    edges = ix.db.get_imports_for_file(ix.file_of[want])
    return {ix.file_owner[int(e.file_id)] for e in edges}


def _probe_file_imports_resolved(ix: TwoFileIndex, want: str) -> set[str]:
    ids = ix.db.get_file_imports_resolved(ix.file_of[want])
    return {ix.file_owner[int(i)] for i in ids}


def _probe_files_importing(ix: TwoFileIndex, want: str) -> set[str]:
    # Reverse direction: c.py is imported by A alone, d.py by B alone.
    ids = ix.db.get_files_importing(ix.import_target_of[want])
    return {ix.file_owner[int(i)] for i in ids}


def _probe_file_summary(ix: TwoFileIndex, want: str) -> set[str]:
    summary = ix.db.get_file_summary(ix.file_of[want])
    return set() if summary is None else {ix.summary_owner[str(summary)]}


def _probe_file_by_id(ix: TwoFileIndex, want: str) -> set[str]:
    record = ix.db.get_file_by_id(ix.file_of[want])
    return set() if record is None else {ix.rel_path_owner[str(record.rel_path)]}


# Path-keyed readers: the scoping key is an exact rel_path rather than a file_id.
def _probe_file_hash(ix: TwoFileIndex, want: str) -> set[str]:
    rel_path = "pkg/a.py" if want == "A" else "pkg/b.py"
    stored = ix.db.get_file_hash(rel_path)
    return set() if stored is None else {ix.hash_owner[str(stored)]}


def _probe_symbol_ids_for_file(ix: TwoFileIndex, want: str) -> set[str]:
    rel_path = "pkg/a.py" if want == "A" else "pkg/b.py"
    ids = ix.db.get_symbol_ids_for_file(rel_path)
    return {ix.symbol_owner[int(i)] for i in ids}


# ---------------------------------------------------------------------------
# The hand-written tables. Typed out, not generated.
# ---------------------------------------------------------------------------

_FILE_ID_READER_PROBES: dict[str, _Probe] = {
    "get_all_symbols_for_file": _probe_all_symbols_for_file,
    "get_chunk_ids_for_file": _probe_chunk_ids_for_file,
    "get_chunk_ids_for_symbols": _probe_chunk_ids_for_symbols,
    "get_file_by_id": _probe_file_by_id,
    "get_file_imports_resolved": _probe_file_imports_resolved,
    "get_file_summary": _probe_file_summary,
    "get_files_importing": _probe_files_importing,
    "get_imports_for_file": _probe_imports_for_file,
    "get_symbol_hashes_for_file": _probe_symbol_hashes_for_file,
    "get_symbol_ids_for_file_id": _probe_symbol_ids_for_file_id,
    "get_symbols_for_file": _probe_symbols_for_file,
    "get_top_symbols_for_file": _probe_top_symbols_for_file,
}

# Methods that take a file_id but are NOT read helpers, each with the reason it
# needs no scoping probe. Listing them here is what lets the discovery test below
# demand set-equality instead of a subset.
_FILE_ID_NON_READERS: dict[str, str] = {
    "delete_file_symbols": "writer (DELETE); its scoping is covered by delete/purge tests",
    "delete_symbols_by_qualified_names": "writer (DELETE), partial re-index path",
    "upsert_file_summary": "writer (INSERT ... ON CONFLICT), keyed by file_id UNIQUE",
}

_PATH_KEYED_READER_PROBES: dict[str, _Probe] = {
    "get_file_hash": _probe_file_hash,
    "get_symbol_ids_for_file": _probe_symbol_ids_for_file,
}

_PATH_KEYED_NOT_PROBED: dict[str, str] = {
    "bm25_search": "path_filter is an optional LIKE narrowing, not an identity key",
    "delete_file_by_path": "writer; covered by TestDeleteFileByPathMatchesEitherKey",
    "find_file_by_path_fragment": "substring LIKE by design — returns many files on purpose",
    "get_file_by_rel_path_suffix": "suffix LIKE by design; returns None when ambiguous",
    "get_files_by_import_path": "LIKE pattern by design — returns many files on purpose",
}

# Parameter names that mean 'a key that is supposed to narrow the result'.
_PATH_KEY_PARAM_NAMES = frozenset(
    {"rel_path", "path", "abs_path", "suffix", "fragment", "pattern", "path_filter"}
)


def _discover_methods_with_param(param_names: frozenset[str]) -> set[str]:
    """Enumerate Database methods taking any of param_names. Enumerating FUNCTIONS
    is the point of this file; the EXPECTED sets are the literal tables above."""
    found: set[str] = set()
    for name, attr in vars(Database).items():
        if name.startswith("__") or not callable(attr):
            continue
        try:
            signature = inspect.signature(attr)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            continue
        if param_names & set(signature.parameters):
            found.add(name)
    return found


class TestEveryFileIdHelperIsAccountedFor:
    """Nothing may take a file_id without either a scoping probe or a written reason.

    Mutation that must make this fail: adding any new `def get_..._for_file(self,
    file_id: int)` to `Database` without adding it to `_FILE_ID_READER_PROBES`
    (or to `_FILE_ID_NON_READERS`). Equally, deleting an entry from either table.

    Consequence if it survives: the file_id-ignored read-helper class re-opens
    silently the next time somebody adds a helper, which is exactly how it got to
    three instances.
    """

    def test_discovered_set_equals_the_hand_written_tables(self) -> None:
        discovered = _discover_methods_with_param(frozenset({"file_id"}))
        tabled = set(_FILE_ID_READER_PROBES) | set(_FILE_ID_NON_READERS)

        assert discovered == tabled, (
            "Database methods taking file_id have drifted from the tables in this "
            "file. Add each new READ helper to _FILE_ID_READER_PROBES with a probe "
            "that proves it scopes, or to _FILE_ID_NON_READERS with a reason.\n"
            f"  undeclared (in db.py, not in the tables): {sorted(discovered - tabled)}\n"
            f"  stale (in the tables, not in db.py):      {sorted(tabled - discovered)}"
        )
        assert tabled == discovered

    def test_the_two_tables_do_not_overlap(self) -> None:
        assert set(_FILE_ID_READER_PROBES) & set(_FILE_ID_NON_READERS) == set()

    def test_twelve_readers_are_probed(self) -> None:
        """Written as a literal so silently emptying the probe table cannot make
        TestEveryFileIdReaderScopesToItsFile pass by having nothing to run."""
        assert len(_FILE_ID_READER_PROBES) == 12


class TestEveryPathKeyedHelperIsAccountedFor:
    """Same accounting for the readers whose scoping key is a path rather than an id.

    Mutation that must make this fail: adding a new rel_path-keyed helper to
    `Database` without tabling it below.
    """

    def test_discovered_set_equals_the_hand_written_tables(self) -> None:
        discovered = _discover_methods_with_param(_PATH_KEY_PARAM_NAMES)
        tabled = set(_PATH_KEYED_READER_PROBES) | set(_PATH_KEYED_NOT_PROBED)

        assert discovered == tabled, (
            "Database methods taking a path-shaped key have drifted from the "
            "tables in this file.\n"
            f"  undeclared: {sorted(discovered - tabled)}\n"
            f"  stale:      {sorted(tabled - discovered)}"
        )
        assert tabled == discovered

    def test_two_path_keyed_readers_are_probed(self) -> None:
        assert len(_PATH_KEYED_READER_PROBES) == 2


class TestTheCollisionFixtureStillDiscriminates:
    """Preconditions on `two_file_index`. If any of these stops holding, every
    scoping assertion below becomes true by construction and proves nothing.

    Mutation that must make these fail: changing `two_file_index` to give the two
    files different symbol names, identical bodies, one shared import target, or
    one shared summary.
    """

    def test_both_files_hold_every_shared_qualified_name(
        self, two_file_index: TwoFileIndex
    ) -> None:
        ix = two_file_index
        rows = ix.db._conn.execute("SELECT file_id, qualified_name FROM symbols").fetchall()
        per_file: dict[int, set[str]] = {}
        for file_id, qualified_name in rows:
            per_file.setdefault(int(file_id), set()).add(str(qualified_name))
        assert per_file[ix.file_of["A"]] == {"Widget", "main"}
        assert per_file[ix.file_of["B"]] == {"Widget", "main"}

    def test_the_two_files_content_hashes_are_disjoint(self, two_file_index: TwoFileIndex) -> None:
        ix = two_file_index
        a_hashes = {h for h, owner in ix.content_hash_owner.items() if owner == "A"}
        b_hashes = {h for h, owner in ix.content_hash_owner.items() if owner == "B"}
        assert len(a_hashes) == 2
        assert len(b_hashes) == 2
        assert a_hashes & b_hashes == set()

    def test_each_file_has_its_own_import_target(self, two_file_index: TwoFileIndex) -> None:
        ix = two_file_index
        assert ix.import_target_of["A"] != ix.import_target_of["B"]
        resolved = ix.db._conn.execute(
            "SELECT file_id, imported_file_id FROM imports ORDER BY file_id"
        ).fetchall()
        assert {(int(r[0]), int(r[1])) for r in resolved} == {
            (ix.file_of["A"], ix.import_target_of["A"]),
            (ix.file_of["B"], ix.import_target_of["B"]),
        }

    def test_each_file_has_its_own_summary_and_hash(self, two_file_index: TwoFileIndex) -> None:
        ix = two_file_index
        summaries = ix.db._conn.execute(
            "SELECT summary FROM file_summaries ORDER BY file_id"
        ).fetchall()
        assert [str(r[0]) for r in summaries] == [
            "file summary owner=A",
            "file summary owner=B",
        ]
        stored = ix.db._conn.execute(
            "SELECT hash FROM files WHERE rel_path IN ('pkg/a.py', 'pkg/b.py') ORDER BY rel_path"
        ).fetchall()
        assert [str(r[0]) for r in stored] == ["filehash-A", "filehash-B"]


class TestEveryFileIdReaderScopesToItsFile:
    """The behavioural oracle for the whole class: ask each helper for one file's
    data in an index where the OTHER file holds identically-named symbols, and the
    other file must be absent from the answer.

    Mutation that must make one of these fail: appending ` OR 1=1` to (or deleting)
    the `WHERE file_id = ?` / `WHERE s.file_id = ?` / `WHERE id = ?` clause of ANY
    single helper named in `_FILE_ID_READER_PROBES` — one site at a time, never two.

    Measured, one mutation at a time (14 separate runs), against the pre-existing
    db/store/indexer/retrieval/graph unit scope with THIS FILE EXCLUDED:

        helper                        was caught before?   params this file kills
        get_all_symbols_for_file      no (SURVIVED)         A and B
        get_top_symbols_for_file      no (SURVIVED)         A and B
        get_chunk_ids_for_file        no (SURVIVED)         A and B
        get_imports_for_file          no (SURVIVED)         A and B
        get_file_imports_resolved     no (SURVIVED)         A and B
        get_files_importing           no (SURVIVED)         A and B
        get_symbol_ids_for_file       no (SURVIVED)         A and B   (path-keyed)
        get_file_by_id                no (SURVIVED)         B only
        get_file_summary              no (SURVIVED)         B only
        get_chunk_ids_for_symbols     yes                   A and B
        get_symbol_ids_for_file_id    yes                   A and B
        get_symbols_for_file          yes                   A and B
        get_symbol_hashes_for_file    yes                   A only
        get_file_hash                 yes                   B only   (path-keyed)

    Why some kill only one direction, and why BOTH are parametrised anyway. The
    `fetchone()` helpers (`get_file_by_id`, `get_file_summary`, `get_file_hash`)
    return the LOWEST-rowid row once the predicate is neutered, which is A's — so
    the leak is visible only when B was asked for. A mutant that instead returned
    the LAST row would be visible only when A was asked for. Running both
    directions is what makes the pair exhaustive rather than lucky.
    `get_symbol_hashes_for_file` returns a dict keyed by qualified_name, and the
    fixture's two files share every key, so an unfiltered read changes the values
    (the later rowid wins) and not the key set — again visible from A's side only.

    Consequence if a mutant survives: `Indexer._insert_one` declares a changed
    symbol unchanged because another file holds the same qualified_name, so it is
    never re-chunked and never re-embedded (see the module docstring); and on the
    retrieval side `file_overview` renders one file's symbols under another's path.
    """

    @pytest.mark.parametrize("helper", sorted(_FILE_ID_READER_PROBES))
    @pytest.mark.parametrize("want", ["A", "B"])
    def test_the_answer_names_only_the_requested_file(
        self, two_file_index: TwoFileIndex, helper: str, want: str
    ) -> None:
        observed = _FILE_ID_READER_PROBES[helper](two_file_index, want)
        expected = {want}
        # Set equality both ways; the empty set fails it, so a helper that returns
        # nothing at all cannot pass vacuously.
        assert observed == expected, f"{helper}({want!r}) exposed owners {sorted(observed)}"
        assert expected == observed

    @pytest.mark.parametrize("helper", sorted(_PATH_KEYED_READER_PROBES))
    @pytest.mark.parametrize("want", ["A", "B"])
    def test_path_keyed_answer_names_only_the_requested_file(
        self, two_file_index: TwoFileIndex, helper: str, want: str
    ) -> None:
        observed = _PATH_KEYED_READER_PROBES[helper](two_file_index, want)
        expected = {want}
        assert observed == expected, f"{helper}({want!r}) exposed owners {sorted(observed)}"
        assert expected == observed


class TestQualifiedNameIsNotFileQualified:
    """DEFECT (pinned deliberately): `symbols.qualified_name` carries no file identity.

    This is the UPSTREAM cause of the blast radius above, not a property of db.py.
    `Symbol.qualified_name` is documented as `ClassName.method_name`
    (core/models.py) and `Parser`/`Indexer` store it verbatim, so two files each
    defining `main` are stored under the byte-identical qualified_name. Every
    incremental-path decision that is keyed on qualified_name alone
    (`Indexer._insert_one`'s unchanged-set, `delete_symbols_by_qualified_names`,
    `get_chunk_ids_for_symbols`) is therefore only correct because a file_id is
    carried alongside — the key itself cannot distinguish them.

    Pinned rather than fixed: this file may not touch src/. Suggested src fix is in
    the round report. strict + raises=AssertionError, so the day qualified_name
    becomes file-qualified this XPASSes, strict turns that into a failure, and
    whoever landed the fix must delete this marker.

    Preconditions below use pytest.fail (NOT assert) on purpose: `Failed` is not an
    AssertionError, so `raises=AssertionError` cannot absorb a broken fixture and
    report it as a tidy xfail.
    """

    @pytest.mark.xfail(
        reason=(
            "DEFECT: symbols.qualified_name is not file-qualified, so two same-named "
            "symbols in different files are indistinguishable by key alone."
        ),
        raises=AssertionError,
        strict=True,
    )
    def test_two_same_named_symbols_in_different_files_get_different_keys(
        self, two_file_index: TwoFileIndex
    ) -> None:
        ix = two_file_index
        rows = ix.db._conn.execute(
            "SELECT file_id, qualified_name FROM symbols WHERE name = 'main' ORDER BY file_id"
        ).fetchall()

        if len(rows) != 2:
            pytest.fail(f"fixture two_file_index no longer stores exactly two 'main' rows: {rows}")
        if int(rows[0][0]) == int(rows[1][0]):
            pytest.fail("fixture two_file_index no longer puts the two 'main' rows in two files")

        assert str(rows[0][1]) != str(rows[1][1])
