"""Tests for ConceptExtractor — LLM semantic concept extraction."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from trelix.core.config import LLMConfig
from trelix.core.models import IndexedFile, Language, Symbol, SymbolKind
from trelix.graph.concepts import ConceptExtractor, SemanticConcept, save_concepts, load_concepts
from trelix.store.db import Database


def _make_symbols() -> list[Symbol]:
    return [
        Symbol(
            id=1, file_id=1, name="authenticate_user",
            qualified_name="AuthService.authenticate_user",
            kind=SymbolKind.METHOD, line_start=10, line_end=30,
            signature="def authenticate_user(self, token: str) -> User",
            body="def authenticate_user(self, token: str) -> User:\n    ...",
        ),
        Symbol(
            id=2, file_id=1, name="refresh_token",
            qualified_name="AuthService.refresh_token",
            kind=SymbolKind.METHOD, line_start=35, line_end=55,
            signature="def refresh_token(self, token: str) -> str",
            body="def refresh_token(self, token: str) -> str:\n    ...",
        ),
    ]


class TestConceptExtractor:
    def test_extract_returns_list_of_semantic_concepts(self) -> None:
        mock_client = MagicMock()
        mock_client.complete.return_value = MagicMock(
            content=(
                '[{"entity": "JWT authentication", "importance": 5, "category": "security"}, '
                '{"entity": "token refresh", "importance": 4, "category": "concept"}]'
            )
        )
        cfg = LLMConfig()
        with patch("trelix.graph.concepts.build_chat_client", return_value=mock_client):
            extractor = ConceptExtractor(cfg)
            concepts = extractor.extract_from_symbols(_make_symbols())

        assert isinstance(concepts, list)
        assert len(concepts) == 2
        assert all(isinstance(c, SemanticConcept) for c in concepts)
        assert concepts[0].name == "jwt authentication"  # lowercased
        assert concepts[0].importance == 5
        assert concepts[0].category == "security"

    def test_extract_tolerates_malformed_json(self) -> None:
        mock_client = MagicMock()
        mock_client.complete.return_value = MagicMock(content="not valid json at all")
        cfg = LLMConfig()
        with patch("trelix.graph.concepts.build_chat_client", return_value=mock_client):
            extractor = ConceptExtractor(cfg)
            concepts = extractor.extract_from_symbols(_make_symbols())
        # Should return empty list, not crash
        assert concepts == []

    def test_extract_tolerates_llm_exception(self) -> None:
        mock_client = MagicMock()
        mock_client.complete.side_effect = RuntimeError("LLM unavailable")
        cfg = LLMConfig()
        with patch("trelix.graph.concepts.build_chat_client", return_value=mock_client):
            extractor = ConceptExtractor(cfg)
            concepts = extractor.extract_from_symbols(_make_symbols())
        assert concepts == []

    def test_save_and_load_concepts(self, tmp_path: Path) -> None:
        db = Database(tmp_path / "index.db")
        # Insert a dummy file and symbol so DB is valid
        from trelix.core.models import IndexedFile, Language
        fid = db.upsert_file(IndexedFile(path="/r/a.py", rel_path="a.py",
                                         language=Language.PYTHON, hash="x", size_bytes=10))
        concepts = [
            SemanticConcept(name="jwt auth", category="security", importance=5, source_symbol_ids=[1, 2]),
            SemanticConcept(name="token refresh", category="concept", importance=3, source_symbol_ids=[2]),
        ]
        save_concepts(db, concepts)
        loaded = load_concepts(db)
        assert len(loaded) == 2
        names = {c.name for c in loaded}
        assert "jwt auth" in names
