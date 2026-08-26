"""Tests for ConceptExtractor — LLM semantic concept extraction."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from trelix.core.config import LLMConfig
from trelix.core.models import Symbol, SymbolKind
from trelix.graph.concepts import ConceptExtractor, SemanticConcept, load_concepts, save_concepts
from trelix.store.db import Database


def _make_symbols() -> list[Symbol]:
    return [
        Symbol(
            id=1,
            file_id=1,
            name="authenticate_user",
            qualified_name="AuthService.authenticate_user",
            kind=SymbolKind.METHOD,
            line_start=10,
            line_end=30,
            signature="def authenticate_user(self, token: str) -> User",
            body="def authenticate_user(self, token: str) -> User:\n    ...",
        ),
        Symbol(
            id=2,
            file_id=1,
            name="refresh_token",
            qualified_name="AuthService.refresh_token",
            kind=SymbolKind.METHOD,
            line_start=35,
            line_end=55,
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

    def test_extract_returns_empty_when_llm_returns_a_json_object_not_a_list(self) -> None:
        """Kills: `if not isinstance(parsed, list):` -> `if isinstance(parsed, list):`
        (or the check deleted) in `extract_from_symbols`.

        The extraction contract is a JSON *array* of concept objects. A well-formed
        JSON *object* (valid `json.loads`, wrong shape) must not be iterated as if it
        were a list of concepts -- doing so would either raise or silently coerce the
        object's keys into bogus concept names.
        """
        mock_client = MagicMock()
        mock_client.complete.return_value = MagicMock(
            content='{"entity": "not an array", "importance": 5}'
        )
        cfg = LLMConfig()
        with patch("trelix.graph.concepts.build_chat_client", return_value=mock_client):
            extractor = ConceptExtractor(cfg)
            concepts = extractor.extract_from_symbols(_make_symbols())
        assert concepts == []

    def test_extract_skips_items_that_are_not_dicts_or_lack_an_entity_key(self) -> None:
        """Kills: `if not isinstance(item, dict) or "entity" not in item:` ->
        `and` (or either half deleted) in `extract_from_symbols`'s filter loop.

        A batch with one well-formed item, one dict missing "entity", and one
        non-dict element must keep only the well-formed item -- proving the filter
        actually discriminates rather than passing everything (or nothing) through.
        """
        mock_client = MagicMock()
        mock_client.complete.return_value = MagicMock(
            content=(
                '[{"entity": "kept concept", "importance": 2, "category": "misc"}, '
                '{"importance": 5}, '
                '"a bare string"]'
            )
        )
        cfg = LLMConfig()
        with patch("trelix.graph.concepts.build_chat_client", return_value=mock_client):
            extractor = ConceptExtractor(cfg)
            concepts = extractor.extract_from_symbols(_make_symbols())
        assert len(concepts) == 1, f"expected exactly the one well-formed item, got {concepts!r}"
        assert concepts[0].name == "kept concept"

    def test_save_and_load_concepts(self, tmp_db: Database) -> None:
        # Insert a dummy file and symbol so DB is valid
        from trelix.core.models import IndexedFile, Language

        tmp_db.upsert_file(
            IndexedFile(
                path="/r/a.py", rel_path="a.py", language=Language.PYTHON, hash="x", size_bytes=10
            )
        )
        concepts = [
            SemanticConcept(
                name="jwt auth", category="security", importance=5, source_symbol_ids=[1, 2]
            ),
            SemanticConcept(
                name="token refresh", category="concept", importance=3, source_symbol_ids=[2]
            ),
        ]
        save_concepts(tmp_db, concepts)
        loaded = load_concepts(tmp_db)
        assert len(loaded) == 2
        names = {c.name for c in loaded}
        assert "jwt auth" in names


class TestConceptExtractorFileSummary:
    """`extract_from_file_summary` -- the RAPTOR file-summary sibling of
    `extract_from_symbols`, never exercised by any test before this class (measured:
    graph's mutation baseline reported 48 `no_tests` mutants in concepts.py, and this
    method's entire body was one of the two reachable-but-never-run code paths, the
    other being the two branches covered above)."""

    def test_extract_from_file_summary_returns_empty_for_blank_summary_without_calling_the_llm(
        self,
    ) -> None:
        """Kills: `if not summary.strip():` -> `if summary.strip():` (condition
        inverted) in `extract_from_file_summary`.

        A whitespace-only summary must short-circuit BEFORE any LLM call -- asserting
        only the return value would still pass if the inverted condition happened to
        route through `try/except` and land on the same `[]` some other way, so the
        mock's call count is the real oracle here.
        """
        mock_client = MagicMock()
        cfg = LLMConfig()
        with patch("trelix.graph.concepts.build_chat_client", return_value=mock_client):
            extractor = ConceptExtractor(cfg)
            concepts = extractor.extract_from_file_summary("   \n\t  ", file_id=999)
        assert concepts == []
        mock_client.complete.assert_not_called()

    def test_extract_from_file_summary_parses_concepts_and_stamps_the_given_file_id(
        self,
    ) -> None:
        """Kills: `source_symbol_ids=[file_id]` -> `source_symbol_ids=[]` (or any
        other id) in `extract_from_file_summary`'s list comprehension.

        `file_id` is a summary-level id, not a per-symbol one like
        `extract_from_symbols` uses -- a wrong id here would silently attribute a
        concept to the wrong file with no error anywhere.
        """
        mock_client = MagicMock()
        mock_client.complete.return_value = MagicMock(
            content='[{"entity": "Token Bucket", "importance": 4, "category": "pattern"}]'
        )
        cfg = LLMConfig()
        with patch("trelix.graph.concepts.build_chat_client", return_value=mock_client):
            extractor = ConceptExtractor(cfg)
            concepts = extractor.extract_from_file_summary(
                "This module implements rate limiting.", file_id=777
            )
        assert len(concepts) == 1
        assert concepts[0].name == "token bucket"  # lowercased + stripped
        assert concepts[0].importance == 4
        assert concepts[0].category == "pattern"
        assert concepts[0].source_symbol_ids == [777]

    def test_extract_from_file_summary_returns_empty_when_llm_returns_a_json_object(
        self,
    ) -> None:
        """Kills: `if not isinstance(parsed, list):` -> `if isinstance(parsed, list):`
        (or deleted) in `extract_from_file_summary`. Mirrors the symbols-path test
        above, for this method's own independent copy of the same check."""
        mock_client = MagicMock()
        mock_client.complete.return_value = MagicMock(content='{"entity": "not a list"}')
        cfg = LLMConfig()
        with patch("trelix.graph.concepts.build_chat_client", return_value=mock_client):
            extractor = ConceptExtractor(cfg)
            concepts = extractor.extract_from_file_summary("some summary text", file_id=1)
        assert concepts == []

    def test_extract_from_file_summary_tolerates_malformed_json(self) -> None:
        """Kills: removing the `try`/`except Exception` around the `json.loads` +
        list-comprehension in `extract_from_file_summary` (unguarded, this raises
        `json.JSONDecodeError` instead of returning `[]`)."""
        mock_client = MagicMock()
        mock_client.complete.return_value = MagicMock(content="not valid json at all")
        cfg = LLMConfig()
        with patch("trelix.graph.concepts.build_chat_client", return_value=mock_client):
            extractor = ConceptExtractor(cfg)
            concepts = extractor.extract_from_file_summary("some summary text", file_id=1)
        assert concepts == []

    def test_extract_from_file_summary_tolerates_llm_exception(self) -> None:
        """Kills: removing the `try`/`except Exception` in `extract_from_file_summary`
        around the `self._client.complete(...)` call itself."""
        mock_client = MagicMock()
        mock_client.complete.side_effect = RuntimeError("LLM unavailable")
        cfg = LLMConfig()
        with patch("trelix.graph.concepts.build_chat_client", return_value=mock_client):
            extractor = ConceptExtractor(cfg)
            concepts = extractor.extract_from_file_summary("some summary text", file_id=1)
        assert concepts == []

    def test_extract_from_file_summary_skips_items_that_are_not_dicts_or_lack_entity(
        self,
    ) -> None:
        """Kills: `if isinstance(item, dict) and "entity" in item` -> `or` (or either
        half deleted) in `extract_from_file_summary`'s comprehension filter.

        Mirrors the symbols-path filter test above, for this method's own independent
        copy of the same guard -- a bare string and a dict missing "entity" must both
        be dropped, keeping only the one well-formed item.
        """
        mock_client = MagicMock()
        mock_client.complete.return_value = MagicMock(
            content=(
                '[{"entity": "kept", "importance": 1, "category": "misc"}, '
                '{"importance": 9}, '
                '"not a dict"]'
            )
        )
        cfg = LLMConfig()
        with patch("trelix.graph.concepts.build_chat_client", return_value=mock_client):
            extractor = ConceptExtractor(cfg)
            concepts = extractor.extract_from_file_summary("some summary text", file_id=1)
        assert len(concepts) == 1, f"expected exactly the one well-formed item, got {concepts!r}"
        assert concepts[0].name == "kept"
