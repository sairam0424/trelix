"""Tests confirming Indexer skips re-embedding symbols whose content is unchanged
on a partial re-index (Item 5 of the v2.6.0 scale backlog).

Strategy mirrors tests/unit/test_indexer_core.py: mock make_embedder and
make_vector_store so no ML models are loaded, use a real SQLite Database and
the real tree-sitter Python parser so symbol content_hash values are genuine.
"""

from __future__ import annotations

import pathlib
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock, patch

from trelix.core.config import EmbedderConfig, IndexConfig, StoreConfig

_DIM = 4


class _FakeEmbedder:
    """Records every text passed to embed() so tests can assert on exactly
    what was (or wasn't) sent for re-embedding."""

    def __init__(self) -> None:
        self.embed_call_texts: list[str] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.embed_call_texts.extend(texts)
        return [[0.1] * _DIM for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.1] * _DIM

    async def embed_async(self, texts: list[str]) -> list[list[float]]:
        return self.embed(texts)

    @property
    def dimension(self) -> int:
        return _DIM


class _FakeVectorStore:
    """In-memory vector store stub — stores nothing, raises nothing."""

    def upsert_batch(self, pairs: list[tuple[int, list[float]]]) -> None:
        pass

    def delete_batch(self, ids: list[int]) -> None:
        pass

    def search(self, vector: list[float], k: int) -> list[Any]:
        return []


@contextmanager
def _patch_rich_progress():
    """Suppress rich terminal output during tests."""
    mock_progress = MagicMock()
    mock_progress.__enter__ = MagicMock(return_value=mock_progress)
    mock_progress.__exit__ = MagicMock(return_value=False)
    mock_progress.add_task = MagicMock(return_value=0)
    mock_progress.advance = MagicMock()
    with patch("trelix.indexing.indexer.Progress", return_value=mock_progress):
        yield mock_progress


def _make_indexer(tmp_dir: str, fake_embedder: _FakeEmbedder) -> Any:
    from trelix.indexing.indexer import Indexer

    cfg = IndexConfig(
        repo_path=tmp_dir,
        incremental=False,
        store=StoreConfig(db_path=str(pathlib.Path(tmp_dir) / ".trelix" / "index.db")),
        embedder=EmbedderConfig.model_construct(provider="local"),
    )

    with (
        patch("trelix.indexing.indexer.make_embedder", return_value=fake_embedder),
        patch("trelix.indexing.indexer.make_vector_store", return_value=_FakeVectorStore()),
    ):
        indexer = Indexer(cfg, quiet=True)

    return indexer


class TestIncrementalSymbolReEmbedding:
    def test_unchanged_symbol_is_not_re_embedded_on_second_index_pass(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Index a file, then re-index it with ONE function body changed and
        ONE unchanged. Only the changed function's chunk must be re-embedded."""
        repo = tmp_path
        source_file = repo / "mod.py"
        source_file.write_text(
            "def unchanged_fn():\n    return 1\n\n\ndef changed_fn():\n    return 2\n"
        )

        fake_embedder = _FakeEmbedder()
        indexer = _make_indexer(str(repo), fake_embedder)

        with _patch_rich_progress():
            indexer.index_file(str(source_file))

        fake_embedder.embed_call_texts.clear()

        # Second pass: change only changed_fn's body.
        source_file.write_text(
            "def unchanged_fn():\n    return 1\n\n\ndef changed_fn():\n    return 999\n"
        )
        with _patch_rich_progress():
            indexer.index_file(str(source_file))

        embed_call_texts = fake_embedder.embed_call_texts
        changed_fn_reembedded = any("999" in text for text in embed_call_texts)
        unchanged_fn_reembedded = any(
            "return 1" in text and "999" not in text for text in embed_call_texts
        )
        assert changed_fn_reembedded, "changed_fn's new body must be re-embedded"
        assert not unchanged_fn_reembedded, (
            "unchanged_fn must NOT be re-embedded — its content_hash didn't change. "
            f"Embed was called with: {embed_call_texts}"
        )

    def test_removed_symbol_is_deleted_from_db(self, tmp_path: pathlib.Path) -> None:
        """A symbol present in the first pass but absent from the second parse
        (function deleted from the file) must be removed from the symbols table."""
        repo = tmp_path
        source_file = repo / "mod.py"
        source_file.write_text("def stays():\n    return 1\n\n\ndef goes():\n    return 2\n")

        fake_embedder = _FakeEmbedder()
        indexer = _make_indexer(str(repo), fake_embedder)

        with _patch_rich_progress():
            indexer.index_file(str(source_file))

        source_file.write_text("def stays():\n    return 1\n")
        with _patch_rich_progress():
            indexer.index_file(str(source_file))

        rows = indexer.db._conn.execute("SELECT qualified_name FROM symbols").fetchall()
        names = {r[0] for r in rows}
        assert "stays" in names
        assert "goes" not in names

    def test_unchanged_symbol_keeps_its_row_id(self, tmp_path: pathlib.Path) -> None:
        """Unchanged symbols must not be deleted+re-inserted — their DB row
        (and hence chunk_id/embedding) is left untouched, so the symbol id
        must be stable across the second pass."""
        repo = tmp_path
        source_file = repo / "mod.py"
        source_file.write_text(
            "def unchanged_fn():\n    return 1\n\n\ndef changed_fn():\n    return 2\n"
        )

        fake_embedder = _FakeEmbedder()
        indexer = _make_indexer(str(repo), fake_embedder)

        with _patch_rich_progress():
            indexer.index_file(str(source_file))

        row = indexer.db._conn.execute(
            "SELECT id FROM symbols WHERE qualified_name = 'unchanged_fn'"
        ).fetchone()
        first_pass_id = row[0]

        source_file.write_text(
            "def unchanged_fn():\n    return 1\n\n\ndef changed_fn():\n    return 999\n"
        )
        with _patch_rich_progress():
            indexer.index_file(str(source_file))

        row = indexer.db._conn.execute(
            "SELECT id FROM symbols WHERE qualified_name = 'unchanged_fn'"
        ).fetchone()
        second_pass_id = row[0]

        assert first_pass_id == second_pass_id, (
            "unchanged_fn's symbol row must be preserved (same id), not deleted and re-inserted"
        )

    def test_unchanged_child_parent_id_survives_parent_content_change(
        self, tmp_path: pathlib.Path
    ) -> None:
        """When a class (parent) changes but one of its methods (child) does
        not, the child's parent_id must be repointed at the parent's new
        row id — not silently left NULL by the FK cascade that fires when
        the old parent row is deleted."""
        repo = tmp_path
        source_file = repo / "mod.py"
        source_file.write_text('class Foo:\n    """v1"""\n\n    def bar(self):\n        return 1\n')

        fake_embedder = _FakeEmbedder()
        indexer = _make_indexer(str(repo), fake_embedder)

        with _patch_rich_progress():
            indexer.index_file(str(source_file))

        # Change only the class docstring — Foo.bar's body/signature is untouched.
        source_file.write_text('class Foo:\n    """v2"""\n\n    def bar(self):\n        return 1\n')
        with _patch_rich_progress():
            indexer.index_file(str(source_file))

        new_foo_id = indexer.db._conn.execute(
            "SELECT id FROM symbols WHERE qualified_name = 'Foo'"
        ).fetchone()[0]
        bar_parent_id = indexer.db._conn.execute(
            "SELECT parent_id FROM symbols WHERE qualified_name = 'Foo.bar'"
        ).fetchone()[0]

        assert bar_parent_id == new_foo_id, (
            "Foo.bar's parent_id must point at Foo's new row id after Foo's "
            "content changed — it must not be left NULL by the ON DELETE "
            "SET NULL cascade that fires when the old Foo row is deleted."
        )

    def test_unchanged_caller_callee_id_survives_callee_content_change(
        self, tmp_path: pathlib.Path
    ) -> None:
        """When callee() changes but caller() (which calls it) does not, the
        pre-existing calls row's callee_id must be repointed at callee()'s
        new row id — not silently left NULL by the FK cascade that fires
        when the old callee row is deleted."""
        repo = tmp_path
        source_file = repo / "mod.py"
        source_file.write_text(
            "def caller():\n    return callee()\n\n\ndef callee():\n    return 1\n"
        )

        fake_embedder = _FakeEmbedder()
        indexer = _make_indexer(str(repo), fake_embedder)

        with _patch_rich_progress():
            indexer.index_file(str(source_file))

        # Change only callee()'s body — caller()'s body/signature is untouched.
        source_file.write_text(
            "def caller():\n    return callee()\n\n\ndef callee():\n    return 999\n"
        )
        with _patch_rich_progress():
            indexer.index_file(str(source_file))

        new_callee_id = indexer.db._conn.execute(
            "SELECT id FROM symbols WHERE qualified_name = 'callee'"
        ).fetchone()[0]
        caller_id = indexer.db._conn.execute(
            "SELECT id FROM symbols WHERE qualified_name = 'caller'"
        ).fetchone()[0]
        callee_id_in_calls_row = indexer.db._conn.execute(
            "SELECT callee_id FROM calls WHERE caller_id = ?", (caller_id,)
        ).fetchone()[0]

        assert callee_id_in_calls_row == new_callee_id, (
            "The calls row's callee_id must point at callee()'s new row id "
            "after callee()'s content changed — it must not be left NULL by "
            "the ON DELETE SET NULL cascade that fires when the old callee "
            "row is deleted."
        )


class TestSameNameMethodCallResolution:
    """Regression tests for _store_call_edges()/_store_type_edges() taking
    matches[0] unconditionally when Database.get_symbol_by_name() returns
    multiple rows — e.g. two classes each defining an identically-named
    method. index_file() is used deliberately (not index()): it's the file
    watcher's hot path, and never runs the batch-mode
    resolve_cross_file_calls()/resolve_cross_file_type_edges() cascade
    (gated on files_in_batch >= _FULL_RESOLVE_THRESHOLD = 5), so
    _store_call_edges()/_store_type_edges()'s own resolution is the ONLY
    resolution this path ever gets."""

    def test_ambiguous_same_named_method_call_leaves_callee_id_null(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Two classes each define retrieve(). A third file calls retrieve()
        on a variable with no static type annotation (no callee_type_hint),
        so the call is genuinely ambiguous — callee_id must stay NULL, not
        silently resolve to whichever class was indexed first."""
        repo = tmp_path
        (repo / "a.py").write_text("class Retriever:\n    def retrieve(self):\n        return 1\n")
        (repo / "b.py").write_text(
            "class FederatedRetriever:\n    def retrieve(self):\n        return 2\n"
        )
        (repo / "c.py").write_text("def use_it(r):\n    return r.retrieve()\n")

        fake_embedder = _FakeEmbedder()
        indexer = _make_indexer(str(repo), fake_embedder)

        with _patch_rich_progress():
            indexer.index_file(str(repo / "a.py"))
            indexer.index_file(str(repo / "b.py"))
            indexer.index_file(str(repo / "c.py"))

        caller_id = indexer.db._conn.execute(
            "SELECT id FROM symbols WHERE qualified_name = 'use_it'"
        ).fetchone()[0]
        callee_id = indexer.db._conn.execute(
            "SELECT callee_id FROM calls WHERE caller_id = ? AND callee_name = 'retrieve'",
            (caller_id,),
        ).fetchone()[0]

        assert callee_id is None, (
            "An ambiguous call to a bare method name shared by two classes "
            "must leave callee_id NULL — resolving it to whichever class "
            "happened to be indexed first (the old matches[0] behavior) "
            "wires the call-graph/blast-radius/PageRank to the wrong symbol."
        )

    def test_type_hinted_same_named_method_call_resolves_correctly(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Same two-class setup, but the caller has a type annotation
        (callee_type_hint) disambiguating which retrieve() is meant —
        must resolve to the CORRECT one, not just any match."""
        repo = tmp_path
        (repo / "a.py").write_text("class Retriever:\n    def retrieve(self):\n        return 1\n")
        (repo / "b.py").write_text(
            "class FederatedRetriever:\n    def retrieve(self):\n        return 2\n"
        )
        (repo / "c.py").write_text(
            "from a import Retriever\n\n\ndef use_it(r: Retriever):\n    return r.retrieve()\n"
        )

        fake_embedder = _FakeEmbedder()
        indexer = _make_indexer(str(repo), fake_embedder)

        with _patch_rich_progress():
            indexer.index_file(str(repo / "a.py"))
            indexer.index_file(str(repo / "b.py"))
            indexer.index_file(str(repo / "c.py"))

        retriever_retrieve_id = indexer.db._conn.execute(
            "SELECT id FROM symbols WHERE qualified_name = 'Retriever.retrieve'"
        ).fetchone()[0]
        caller_id = indexer.db._conn.execute(
            "SELECT id FROM symbols WHERE qualified_name = 'use_it'"
        ).fetchone()[0]
        callee_id = indexer.db._conn.execute(
            "SELECT callee_id FROM calls WHERE caller_id = ? AND callee_name = 'retrieve'",
            (caller_id,),
        ).fetchone()[0]

        assert callee_id == retriever_retrieve_id, (
            "A type-hinted call to r.retrieve() where r: Retriever must "
            "resolve to Retriever.retrieve, not FederatedRetriever.retrieve "
            "or an arbitrary matches[0] pick."
        )

    def test_unambiguous_bare_name_call_still_resolves(self, tmp_path: pathlib.Path) -> None:
        """Positive case: when only ONE symbol in the whole index has the
        called name, resolution must still succeed exactly as before —
        the fix must not turn every bare-name call into an unresolved edge."""
        repo = tmp_path
        (repo / "a.py").write_text("def helper():\n    return 1\n")
        (repo / "b.py").write_text("def caller():\n    return helper()\n")

        fake_embedder = _FakeEmbedder()
        indexer = _make_indexer(str(repo), fake_embedder)

        with _patch_rich_progress():
            indexer.index_file(str(repo / "a.py"))
            indexer.index_file(str(repo / "b.py"))

        helper_id = indexer.db._conn.execute(
            "SELECT id FROM symbols WHERE qualified_name = 'helper'"
        ).fetchone()[0]
        caller_id = indexer.db._conn.execute(
            "SELECT id FROM symbols WHERE qualified_name = 'caller'"
        ).fetchone()[0]
        callee_id = indexer.db._conn.execute(
            "SELECT callee_id FROM calls WHERE caller_id = ?", (caller_id,)
        ).fetchone()[0]

        assert callee_id == helper_id, (
            "An unambiguous bare-name call (only one matching symbol in the "
            "whole index) must still resolve correctly."
        )

    def test_ambiguous_same_named_type_extends_leaves_to_symbol_id_null(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Same ambiguity class for type edges: two same-named base classes
        in different files, a third class extends the bare name with no
        way to disambiguate — to_symbol_id must stay NULL."""
        repo = tmp_path
        (repo / "a.py").write_text("class Base:\n    def a_only(self):\n        return 1\n")
        (repo / "b.py").write_text("class Base:\n    def b_only(self):\n        return 2\n")
        (repo / "c.py").write_text("class Child(Base):\n    pass\n")

        fake_embedder = _FakeEmbedder()
        indexer = _make_indexer(str(repo), fake_embedder)

        with _patch_rich_progress():
            indexer.index_file(str(repo / "a.py"))
            indexer.index_file(str(repo / "b.py"))
            indexer.index_file(str(repo / "c.py"))

        child_id = indexer.db._conn.execute(
            "SELECT id FROM symbols WHERE qualified_name = 'Child'"
        ).fetchone()[0]
        to_symbol_id = indexer.db._conn.execute(
            "SELECT to_symbol_id FROM type_edges "
            "WHERE from_symbol_id = ? AND to_type_name = 'Base'",
            (child_id,),
        ).fetchone()[0]

        assert to_symbol_id is None, (
            "An ambiguous 'extends Base' edge, where two unrelated classes "
            "named Base exist in different files, must leave to_symbol_id "
            "NULL rather than resolving to whichever Base was indexed first."
        )


def _sym(qualified_name: str, symbol_id: int) -> Any:
    from trelix.core.models import Symbol, SymbolKind

    return Symbol(
        file_id=1,
        name=qualified_name.rsplit(".", 1)[-1],
        qualified_name=qualified_name,
        kind=SymbolKind.METHOD,
        line_start=1,
        line_end=2,
        signature="def x(self)",
        body="",
        id=symbol_id,
    )


class TestResolveSymbolMatch:
    """Direct tests for Indexer._resolve_symbol_match() — the pure-function
    priority cascade _store_call_edges()/_store_type_edges() now use instead
    of matches[0]. Mirrors tests/unit/test_store.py::TestResolveCallsPriority
    at the same granularity (qualified-name priority, type-hint priority,
    unique-name-only, ambiguous -> None), so the two cascades' behavior stays
    testable at matching levels and any future divergence is caught here
    too, not just at the SQL-cascade level."""

    def test_unique_qualified_name_match_wins(self) -> None:
        from trelix.indexing.indexer import Indexer

        matches = [_sym("AuthService.login", 1), _sym("OtherService.login", 2)]
        result = Indexer._resolve_symbol_match(matches, "AuthService.login", None)
        assert result == 1

    def test_ambiguous_qualified_name_match_returns_none(self) -> None:
        """Two top-level symbols (qualified_name == bare name) sharing the
        exact same qualified_name are still ambiguous — priority 1 must
        require uniqueness, not just presence."""
        from trelix.indexing.indexer import Indexer

        matches = [_sym("Base", 1), _sym("Base", 2)]
        result = Indexer._resolve_symbol_match(matches, "Base", None)
        assert result is None

    def test_type_hint_resolves_correct_match(self) -> None:
        from trelix.indexing.indexer import Indexer

        matches = [_sym("UserService.login", 1), _sym("AdminService.login", 2)]
        result = Indexer._resolve_symbol_match(matches, "login", "UserService")
        assert result == 1

    def test_ambiguous_type_hint_returns_none(self) -> None:
        from trelix.indexing.indexer import Indexer

        matches = [_sym("UserService.login", 1), _sym("UserService.login", 2)]
        result = Indexer._resolve_symbol_match(matches, "login", "UserService")
        assert result is None

    def test_unique_bare_name_match_resolves(self) -> None:
        from trelix.indexing.indexer import Indexer

        matches = [_sym("helper", 1)]
        result = Indexer._resolve_symbol_match(matches, "helper", None)
        assert result == 1

    def test_ambiguous_bare_name_match_returns_none(self) -> None:
        from trelix.indexing.indexer import Indexer

        matches = [_sym("ServiceA.process", 1), _sym("ServiceB.process", 2)]
        result = Indexer._resolve_symbol_match(matches, "process", None)
        assert result is None

    def test_empty_matches_returns_none(self) -> None:
        from trelix.indexing.indexer import Indexer

        assert Indexer._resolve_symbol_match([], "anything", None) is None
