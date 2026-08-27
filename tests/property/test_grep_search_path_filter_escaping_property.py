"""grep_search.py's three `path_filter` LIKE sites, fixed the same way as
db.bm25_search's (tests/property/test_path_filter_escaping_property.py).

Three call sites share the identical unescaped-LIKE defect shape before this
fix: `_name_search`'s `f.rel_path LIKE ?`, and `_body_search`'s two branches
(the FTS5 pre-filter and the LIMIT-2000 fallback scan). All three now go
through the same `escape_like_pattern()` helper the bm25 fix introduced, so
what these tests need to prove is narrower than a fresh property sweep of the
escaping logic itself (that's already covered exhaustively by Hypothesis in
the sibling file) — only that each call site actually WIRES the escape in.
Hand-picked concrete examples are proportionate to that claim; a Hypothesis
sweep here would mostly be re-testing the shared helper a second time.

MUTATION: revert any one of the three `escape_like_pattern(path_filter)` call
sites back to a bare `path_filter` and the corresponding test below fails.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from trelix.core.models import Chunk, IndexedFile, Language, Symbol, SymbolKind
from trelix.retrieval.grep_search import grep_search
from trelix.store.db import Database

_SEGMENT_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789"
_DECOY_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _make_file(db: Database, rel_path: str) -> int:
    f = IndexedFile(
        path=f"/repo/{rel_path}",
        rel_path=rel_path,
        language=Language.PYTHON,
        hash="deadbeef",
        size_bytes=64,
    )
    return db.upsert_file(f)


def _insert_symbol(db: Database, file_id: int, name: str, body: str) -> None:
    sym = Symbol(
        file_id=file_id,
        name=name,
        qualified_name=name,
        kind=SymbolKind.FUNCTION,
        line_start=1,
        line_end=1,
        signature=f"def {name}():",
        body=body,
    )
    sym_id = db.insert_symbol(sym)
    db._conn.commit()
    chunk = Chunk(symbol_id=sym_id, chunk_text=body, token_count=1)
    db.insert_chunk(chunk)
    db._conn.commit()


class TestNameSearchPathFilterWildcardLeak:
    """`_name_search`'s `f.rel_path LIKE ?` — reached by every `grep_search()`
    call via its exact-name branch, so this is the path any caller hits first.
    """

    @settings(
        derandomize=True,
        max_examples=15,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    @example(("src", "_", "auth", "X"))
    @example(("src", "%", "auth", "Z"))
    @given(
        fixture=st.tuples(
            st.text(alphabet=_SEGMENT_ALPHABET, min_size=1, max_size=8),
            st.sampled_from(("_", "%")),
            st.text(alphabet=_SEGMENT_ALPHABET, min_size=1, max_size=8),
            st.sampled_from(_DECOY_ALPHABET),
        )
    )
    def test_wildcard_char_in_path_filter_is_treated_literally(
        self, fixture: tuple[str, str, str, str]
    ) -> None:
        prefix, wildcard, suffix, decoy_char = fixture
        tmp_dir = Path(tempfile.mkdtemp())
        try:
            db = Database(tmp_dir / "index.db")
            target_dir = f"{prefix}{wildcard}{suffix}"
            decoy_dir = f"{prefix}{decoy_char}{suffix}"
            assert decoy_dir != target_dir, (
                f"fixture built an identical decoy and target path ({target_dir!r})"
            )

            target_file = _make_file(db, f"{target_dir}/mod.py")
            decoy_file = _make_file(db, f"{decoy_dir}/mod.py")
            _insert_symbol(db, target_file, "zzn_target_fn", "def zzn_target_fn(): pass")
            _insert_symbol(db, decoy_file, "zzn_target_fn", "def zzn_target_fn(): pass")

            results = grep_search(db, "zzn_target_fn", k=10, path_filter=f"{prefix}{wildcard}")
            matched_paths = {r.file.rel_path for r in results}

            assert matched_paths == {f"{target_dir}/mod.py"}
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


class TestBodySearchPathFilterWildcardLeak:
    """`_body_search`'s two branches — reached only when `_name_search` finds
    nothing, so the query below is deliberately NOT any symbol's name/prefix.
    """

    def test_fts5_branch_treats_the_wildcard_literally(self) -> None:
        """`use_regex=False`: the FTS5 MATCH pre-filter branch."""
        tmp_dir = Path(tempfile.mkdtemp())
        try:
            db = Database(tmp_dir / "index.db")
            target_file = _make_file(db, "src_fts/mod.py")
            decoy_file = _make_file(db, "srcXfts/mod.py")
            # "helper" appears in the body but is not this symbol's name/prefix,
            # so _name_search finds nothing and _body_search's FTS5 branch runs.
            _insert_symbol(db, target_file, "zzf_one", "def zzf_one(): return helper_marker()")
            _insert_symbol(db, decoy_file, "zzf_two", "def zzf_two(): return helper_marker()")

            results = grep_search(db, "helper_marker", k=10, path_filter="src_fts")
            matched_paths = {r.file.rel_path for r in results}

            assert matched_paths == {"src_fts/mod.py"}
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_fallback_scan_branch_treats_the_wildcard_literally(self) -> None:
        """`use_regex=True` with a pattern FTS5's tokenizer cannot MATCH (a bare
        regex quantifier is not a valid FTS5 query token), forcing the
        LIMIT-2000 fallback scan branch to run instead of the FTS5 pre-filter."""
        tmp_dir = Path(tempfile.mkdtemp())
        try:
            db = Database(tmp_dir / "index.db")
            target_file = _make_file(db, "src_scan/mod.py")
            decoy_file = _make_file(db, "srcXscan/mod.py")
            _insert_symbol(db, target_file, "zzs_one", "def zzs_one(): return scan_marker_9()")
            _insert_symbol(db, decoy_file, "zzs_two", "def zzs_two(): return scan_marker_9()")

            results = grep_search(
                db, r"scan_marker_\d", k=10, path_filter="src_scan", use_regex=True
            )
            matched_paths = {r.file.rel_path for r in results}

            assert matched_paths == {"src_scan/mod.py"}
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
