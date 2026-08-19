"""
Import → file resolution (``Database.resolve_import_file_ids``).

Regression suite for EXE-03: **Python relative imports never resolved.**

``_resolve_module_to_file`` had exactly one relative-import branch and it split the
specifier on ``"/"``. Python relative imports are DOT-separated, so ``".cache"``
became the single path component ``".cache"`` — which matches neither ``".."`` nor
``"."`` and was therefore appended verbatim, producing the nonexistent candidate
``"pkg/.cache"``. Every Python relative import fell through to ``None``.

Measured on trelix's own live index before the fix:

    js-style  ("./x", "../x")   34 rows, 34 resolved
    py-style  (".x",  "..x")    40 rows,  0 resolved

The rows were present and named the target correctly; they just carried
``imported_file_id = NULL`` forever, which is why importer queries reported ``(0)``
for symbols that are demonstrably imported, and why the ``imports`` table that
call-resolution wants to consult is empty for trelix's own primary language.

Everything here is a synthetic in-DB fixture: rows are inserted directly, so no
indexing pass, no parser and no embedder are involved.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from trelix.core.models import ImportEdge, IndexedFile, Language
from trelix.store.db import Database

# ---------------------------------------------------------------------------
# Synthetic file tree
#
# Mirrors the shape that actually broke: a Python package under a "src/" root
# with a sub-package, plus a TypeScript corner used as the control. Extensions
# matter (the resolver strips them), directory depth matters (relative ascent
# is counted against it), and "__init__.py" matters (a Python relative
# specifier is allowed to name a *package*, not just a module).
#
# "src/pkg/mod.py" is deliberate, not incidental: the resolver's "/mod" strip
# is Rust's mod.rs convention, and applying it language-blind registered this
# ordinary Python module under the directory key "src/pkg" — so "from .. import
# x" in sub/leaf.py resolved to mod.py instead of pkg/__init__.py. That wrong
# edge is asserted against below.
# ---------------------------------------------------------------------------

_TREE: tuple[tuple[str, Language], ...] = (
    ("src/pkg/__init__.py", Language.PYTHON),
    ("src/pkg/cache.py", Language.PYTHON),
    ("src/pkg/mod.py", Language.PYTHON),
    ("src/pkg/store/__init__.py", Language.PYTHON),
    ("src/pkg/sub/__init__.py", Language.PYTHON),
    ("src/pkg/sub/leaf.py", Language.PYTHON),
    ("web/app.ts", Language.TYPESCRIPT),
    ("web/util.ts", Language.TYPESCRIPT),
    ("web/nested/deep.ts", Language.TYPESCRIPT),
)

# (importer rel_path, imported_from, expected target rel_path or None)
#
# The Python rows are the defect. The two "./util" / "../util" rows are the
# control: the slash-separated convention already worked (34/34 live) and a fix
# that treats the two conventions identically would break it, because N leading
# dots do not mean the same thing in each:
#
#     JS    "./x"  = this directory        Python  ".x"   = this package   (up 0)
#     JS    "../x" = parent directory      Python  "..x"  = parent package (up 1)
#
# i.e. a Python specifier ascends (dots - 1) levels; a slash specifier ascends
# once per ".." segment.
_CASES: tuple[tuple[str, str, str | None], ...] = (
    # single dot → the importer's own package
    ("src/pkg/mod.py", ".cache", "src/pkg/cache.py"),
    # double dot → one package up, module target
    ("src/pkg/sub/leaf.py", "..cache", "src/pkg/cache.py"),
    # double dot → one package up, *package* target (resolves to its __init__.py)
    ("src/pkg/sub/leaf.py", "..store", "src/pkg/store/__init__.py"),
    # dotted package path after the leading dot
    ("src/pkg/mod.py", ".sub.leaf", "src/pkg/sub/leaf.py"),
    # bare "." — "from . import x" names the importer's own package
    ("src/pkg/mod.py", ".", "src/pkg/__init__.py"),
    # bare ".." — "from .. import x" names the parent package
    ("src/pkg/sub/leaf.py", "..", "src/pkg/__init__.py"),
    # CONTROL: slash-separated relative imports must keep resolving
    ("web/app.ts", "./util", "web/util.ts"),
    ("web/nested/deep.ts", "../util", "web/util.ts"),
    # A relative specifier naming nothing must stay NULL — an invented edge is
    # worse than a missing one, so there is no last-component fallback here.
    ("src/pkg/mod.py", ".missing", None),
    # More leading dots than there are directories to ascend: escapes the repo
    # root, so there is nothing to point at.
    ("src/pkg/mod.py", "....escape", None),
)


@pytest.fixture()
def resolved_index(tmp_path: Path) -> tuple[Database, dict[str, int], dict[int, str]]:
    """
    Build the synthetic tree + import rows, run one resolution pass, and hand
    back the DB plus rel_path↔file_id maps so assertions can talk in paths.
    """
    db = Database(tmp_path / "index.db")

    fid_by_path: dict[str, int] = {}
    for rel, lang in _TREE:
        fid_by_path[rel] = db.upsert_file(
            IndexedFile(
                path=f"/repo/{rel}",
                rel_path=rel,
                language=lang,
                hash=f"hash-{rel}",
                size_bytes=100,
            )
        )
    path_by_fid = {fid: rel for rel, fid in fid_by_path.items()}

    db.insert_imports(
        [
            ImportEdge(
                file_id=fid_by_path[importer],
                imported_from=spec,
                imported_names=["Thing"],
            )
            for importer, spec, _expected in _CASES
        ]
    )

    db.resolve_import_file_ids()
    return db, fid_by_path, path_by_fid


def _resolved_target(
    db: Database, fid_by_path: dict[str, int], path_by_fid: dict[int, str], importer: str, spec: str
) -> str | None:
    row = db._conn.execute(
        "SELECT imported_file_id FROM imports WHERE file_id = ? AND imported_from = ?",
        (fid_by_path[importer], spec),
    ).fetchone()
    assert row is not None, f"import row for {importer} -> {spec} disappeared"
    return None if row[0] is None else path_by_fid[row[0]]


class TestRelativeImportResolution:
    @pytest.mark.parametrize(("importer", "spec", "expected"), _CASES)
    def test_relative_specifier_resolves_to_expected_file(
        self,
        resolved_index: tuple[Database, dict[str, int], dict[int, str]],
        importer: str,
        spec: str,
        expected: str | None,
    ) -> None:
        db, fid_by_path, path_by_fid = resolved_index
        assert _resolved_target(db, fid_by_path, path_by_fid, importer, spec) == expected

    def test_relative_resolution_count_matches_resolvable_cases(
        self,
        resolved_index: tuple[Database, dict[str, int], dict[int, str]],
    ) -> None:
        """
        Count, not just non-crash: the whole point of EXE-03 is that the count
        was 0 for Python. Eight of the ten fixture specifiers are resolvable;
        the other two must stay NULL.
        """
        db, _fid_by_path, _path_by_fid = resolved_index
        expected_resolvable = sum(1 for *_rest, target in _CASES if target is not None)
        (count,) = db._conn.execute(
            "SELECT COUNT(*) FROM imports WHERE imported_file_id IS NOT NULL"
        ).fetchone()
        assert count == expected_resolvable == 8

    def test_relative_dot_separated_specifiers_all_resolve(
        self,
        resolved_index: tuple[Database, dict[str, int], dict[int, str]],
    ) -> None:
        """
        The measurement that named the defect, reproduced in miniature: on the
        live index dot-separated relative imports resolved 0/40 while
        slash-separated ones resolved 34/34. Assert the dot-separated family
        specifically, so a regression cannot hide behind the passing controls.
        """
        db, fid_by_path, path_by_fid = resolved_index
        dotted = [
            (importer, spec, target)
            for importer, spec, target in _CASES
            if "/" not in spec and target is not None
        ]
        assert len(dotted) == 6, "fixture no longer covers the dot-separated family"
        resolved = [
            (importer, spec)
            for importer, spec, _t in dotted
            if _resolved_target(db, fid_by_path, path_by_fid, importer, spec) is not None
        ]
        assert len(resolved) == len(dotted), (
            f"dot-separated relative imports resolved {len(resolved)}/{len(dotted)}"
        )


class TestAbsoluteImportResolutionUnaffected:
    """
    ``resolve_import_file_ids`` builds one shared path lookup for every
    language, so a relative-import fix reaches the absolute paths too. These
    pin that blast radius.
    """

    def test_dotted_absolute_module_still_resolves(self, tmp_path: Path) -> None:
        db = Database(tmp_path / "index.db")
        fids = {
            rel: db.upsert_file(
                IndexedFile(
                    path=f"/repo/{rel}",
                    rel_path=rel,
                    language=lang,
                    hash=f"hash-{rel}",
                    size_bytes=100,
                )
            )
            for rel, lang in _TREE
        }
        db.insert_imports(
            [
                # module target — worked before the fix, must keep working
                ImportEdge(
                    file_id=fids["src/pkg/mod.py"], imported_from="pkg.cache", imported_names=[]
                ),
                # package target — "import pkg.store" is the package's __init__.py
                ImportEdge(
                    file_id=fids["src/pkg/mod.py"], imported_from="pkg.store", imported_names=[]
                ),
            ]
        )
        db.resolve_import_file_ids()

        rows = dict(
            db._conn.execute(
                "SELECT imported_from, imported_file_id FROM imports WHERE file_id = ?",
                (fids["src/pkg/mod.py"],),
            ).fetchall()
        )
        assert rows["pkg.cache"] == fids["src/pkg/cache.py"]
        assert rows["pkg.store"] == fids["src/pkg/store/__init__.py"]
