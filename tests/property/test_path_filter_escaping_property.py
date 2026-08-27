"""FIXED: `bm25_search`'s `path_filter` was a raw SQL LIKE pattern, unescaped --
`%` and `_` in a `path_filter` value were interpreted as LIKE wildcards instead
of literal characters. This file was originally written as a pinned `xfail`
before the fix landed; both classes below now assert the real, fixed contract.

The sibling, already-fixed function `Database.get_index_metadata_with_prefix`
(src/trelix/store/db.py) had the identical defect shape and was fixed the same
way first — this fix (`escape_like_pattern()` + explicit `ESCAPE '\\'`, both in
src/trelix/store/db.py) is that same helper, now shared by both call sites
rather than a second inline copy. `bm25_search`'s SQL now reads
`fi.rel_path LIKE ? ESCAPE '\\'` with the parameter built from
`escape_like_pattern(path_filter) + "%"`. The retriever's OWN vector-search
path_filter (`retriever.py::_vector_search`) was never affected: it filters
with plain Python `str.startswith`, which never treats any character as a
wildcard. `grep_search.py` had the identical unescaped shape in three call
sites (`_name_search`, `_body_search` x2) and is fixed the same way, covered by
tests/property/test_grep_search_path_filter_escaping_property.py.

FALSIFYING INPUT THAT PROVED THE ORIGINAL DEFECT: two files "src_auth/login.py"
and "srcXauth/login.py", both with an identically-named, identically-matching
symbol. Querying `bm25_search(db, query, path_filter="src_auth")` returned BOTH
files before this fix -- the literal "_" in "src_auth" was read as a LIKE
single-char wildcard and matched "srcXauth" too. The boundary: a `path_filter`
with no `%`/`_` in it was never affected -- see
TestPlainPrefixIsUnaffectedControl below, which must keep passing (it is the
discriminating control: if it ever fails, the assertion above stops proving
anything about the wildcard characters specifically).

DERANDOMIZED: both `@settings` below pin `derandomize=True`, so a regression
here is reproducible rather than depending on which random input Hypothesis
happens to draw on a given run.

MUTATION: revert `escape_like_pattern()` to `return value` (or drop the
`ESCAPE '\\'` clause from bm25_search's SQL) and
test_wildcard_char_in_path_filter_is_treated_literally fails again.
"""

from __future__ import annotations

import shutil
import string
import tempfile
from pathlib import Path

from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from trelix.core.models import Chunk, IndexedFile, Language, Symbol, SymbolKind
from trelix.retrieval.bm25 import bm25_search
from trelix.store.db import Database

# Decoy replacement characters for the wildcard position: anything that is NOT
# the wildcard itself and not a path separator (so the decoy path is still a
# plausible single-segment sibling directory name).
_DECOY_ALPHABET = "".join(c for c in string.ascii_uppercase + string.digits if c not in ("_", "%"))


def _make_file(db: Database, rel_path: str) -> int:
    f = IndexedFile(
        path=f"/repo/{rel_path}",
        rel_path=rel_path,
        language=Language.PYTHON,
        hash="deadbeef",
        size_bytes=64,
    )
    return db.upsert_file(f)


def _insert_matching_symbol(db: Database, file_id: int, name: str) -> None:
    sym = Symbol(
        file_id=file_id,
        name=name,
        qualified_name=name,
        kind=SymbolKind.FUNCTION,
        line_start=1,
        line_end=1,
        signature=f"def {name}():",
        body=f"def {name}(): pass",
    )
    sym_id = db.insert_symbol(sym)
    db._conn.commit()
    chunk = Chunk(symbol_id=sym_id, chunk_text=name, token_count=1)
    db.insert_chunk(chunk)
    db._conn.commit()


_SEGMENT_ALPHABET = string.ascii_lowercase + string.digits

# (prefix, wildcard_char) pairs: `prefix + wildcard_char` is the path_filter under
# test. `_` and `%` are the only two SQL LIKE metacharacters; ESCAPE would need to
# handle both, so both are swept.
_WILDCARD_CHARS = ("_", "%")


@st.composite
def _prefix_wildcard_suffix_decoy(draw: st.DrawFn) -> tuple[str, str, str, str]:
    prefix = draw(st.text(alphabet=_SEGMENT_ALPHABET, min_size=1, max_size=8))
    wildcard = draw(st.sampled_from(_WILDCARD_CHARS))
    suffix = draw(st.text(alphabet=_SEGMENT_ALPHABET, min_size=1, max_size=8))
    decoy_char = draw(st.sampled_from(_DECOY_ALPHABET))
    return prefix, wildcard, suffix, decoy_char


class TestPathFilterWildcardLeak:
    """A `path_filter` containing a literal `_` or `%` must match ONLY that literal
    prefix, not every string with an arbitrary character substituted at the
    wildcard's position.
    """

    @settings(
        derandomize=True,
        max_examples=20,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    @example(("src", "_", "auth", "X"))  # the hand-verified case
    @example(("src", "%", "auth", "Z"))
    @given(fixture=_prefix_wildcard_suffix_decoy())
    def test_wildcard_char_in_path_filter_is_treated_literally(
        self, fixture: tuple[str, str, str, str]
    ) -> None:
        prefix, wildcard, suffix, decoy_char = fixture
        tmp_dir = Path(tempfile.mkdtemp())
        try:
            db = Database(tmp_dir / "index.db")

            target_dir = f"{prefix}{wildcard}{suffix}"
            decoy_dir = f"{prefix}{decoy_char}{suffix}"
            # Precondition: the decoy must actually differ from the target as a string,
            # or this fixture cannot discriminate a leak from a correct match.
            assert decoy_dir != target_dir, (
                f"fixture built an identical decoy and target path ({target_dir!r}); "
                "the decoy_char draw must differ from the wildcard char at this position"
            )

            target_file = _make_file(db, f"{target_dir}/mod.py")
            decoy_file = _make_file(db, f"{decoy_dir}/mod.py")
            _insert_matching_symbol(db, target_file, "zzq_target_fn")
            _insert_matching_symbol(db, decoy_file, "zzq_target_fn")

            results = bm25_search(db, "zzq_target_fn", k=10, path_filter=f"{prefix}{wildcard}")
            matched_paths = {r.file.rel_path for r in results}

            # The wildcard character in path_filter must be treated literally.
            assert matched_paths == {f"{target_dir}/mod.py"}
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


class TestPlainPrefixIsUnaffectedControl:
    """Discriminating control for the xfail above: a path_filter with NO SQL LIKE
    metacharacter must behave correctly TODAY (this is not a general "path_filter
    is broken" claim). If this ever fails, the xfail above stops proving anything
    about the wildcard characters specifically.
    """

    @settings(
        derandomize=True,
        max_examples=15,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    @example(prefix="clean", suffix="prefix")
    @given(
        prefix=st.text(alphabet=_SEGMENT_ALPHABET, min_size=1, max_size=8),
        suffix=st.text(alphabet=_SEGMENT_ALPHABET, min_size=1, max_size=8),
    )
    def test_plain_alnum_prefix_excludes_its_sibling(self, prefix: str, suffix: str) -> None:
        tmp_dir = Path(tempfile.mkdtemp())
        try:
            db = Database(tmp_dir / "index.db")

            target_dir = f"{prefix}{suffix}"
            decoy_dir = f"other{prefix}{suffix}"  # shares the suffix, does not start with prefix
            target_file = _make_file(db, f"{target_dir}/mod.py")
            decoy_file = _make_file(db, f"{decoy_dir}/mod.py")
            _insert_matching_symbol(db, target_file, "zzp_target_fn")
            _insert_matching_symbol(db, decoy_file, "zzp_target_fn")

            results = bm25_search(db, "zzp_target_fn", k=10, path_filter=target_dir)
            matched_paths = {r.file.rel_path for r in results}

            assert matched_paths == {f"{target_dir}/mod.py"}
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
