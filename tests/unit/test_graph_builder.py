"""Tests for GraphBuilder — full graph construction pipeline."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from trelix.core.config import IndexConfig
from trelix.core.models import CallEdge, IndexedFile, Language, Symbol, SymbolKind
from trelix.graph.builder import GraphBuildResult, GraphBuilder
from trelix.store.db import Database


def _populated_repo(tmp_path: Path) -> Path:
    """Create a minimal indexed repo at tmp_path."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".trelix").mkdir()
    db = Database(repo / ".trelix" / "index.db")

    fid = db.upsert_file(IndexedFile(
        path=str(repo / "auth.py"), rel_path="auth.py",
        language=Language.PYTHON, hash="x", size_bytes=100,
    ))
    sid1 = db.insert_symbol(Symbol(
        file_id=fid, name="login", qualified_name="login",
        kind=SymbolKind.FUNCTION, line_start=1, line_end=10,
        signature="def login()", body="def login(): pass",
    ))
    sid2 = db.insert_symbol(Symbol(
        file_id=fid, name="hash_password", qualified_name="hash_password",
        kind=SymbolKind.FUNCTION, line_start=12, line_end=20,
        signature="def hash_password()", body="def hash_password(): pass",
    ))
    db.insert_call_edges([
        CallEdge(caller_id=sid1, callee_name="hash_password", callee_id=sid2, line=5)
    ])
    db._conn.commit()
    db.close()
    return repo


class TestGraphBuilder:
    def test_build_returns_result(self, tmp_path: Path) -> None:
        repo = _populated_repo(tmp_path)
        config = IndexConfig(repo_path=str(repo))
        builder = GraphBuilder(config)
        result = builder.build(extract_concepts=False)
        assert isinstance(result, GraphBuildResult)
        assert result.node_count >= 2
        assert result.edge_count >= 1
        assert result.community_count >= 1
        assert result.concept_count == 0  # no concept extraction

    def test_build_with_concepts_disabled_does_not_call_llm(self, tmp_path: Path) -> None:
        repo = _populated_repo(tmp_path)
        config = IndexConfig(repo_path=str(repo))
        builder = GraphBuilder(config)
        with patch("trelix.graph.builder.ConceptExtractor") as MockCE:
            result = builder.build(extract_concepts=False)
        MockCE.assert_not_called()
        assert result.concept_count == 0

    def test_build_assigns_communities(self, tmp_path: Path) -> None:
        repo = _populated_repo(tmp_path)
        config = IndexConfig(repo_path=str(repo))
        builder = GraphBuilder(config)
        result = builder.build(extract_concepts=False)
        # All nodes should have community set
        for _, attrs in result.code_graph.nx.nodes(data=True):
            assert attrs.get("community") is not None

    def test_elapsed_seconds_positive(self, tmp_path: Path) -> None:
        repo = _populated_repo(tmp_path)
        config = IndexConfig(repo_path=str(repo))
        result = GraphBuilder(config).build(extract_concepts=False)
        assert result.elapsed_seconds >= 0.0
