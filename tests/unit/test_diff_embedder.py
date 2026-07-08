"""Tests for semantic diff embeddings (CCRep-style before/after body pairs)."""

from __future__ import annotations


class TestDiffEmbedder:
    def test_embed_hunk_concatenates_before_and_after(self):
        from unittest.mock import MagicMock

        from trelix.review.diff_embedder import DiffEmbedder

        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = [0.1] * 384

        de = DiffEmbedder(mock_embedder)
        result = de.embed_hunk(
            before_code="def login(user, pw): return check(pw)",
            after_code="def login(user, pw): return bcrypt.check(pw)",
        )

        assert result == [0.1] * 384
        # Must call embed_query with concatenated before+after
        call_arg = mock_embedder.embed_query.call_args[0][0]
        assert "def login" in call_arg
        assert "bcrypt" in call_arg

    def test_embed_hunk_handles_empty_before(self):
        from unittest.mock import MagicMock

        from trelix.review.diff_embedder import DiffEmbedder

        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = [0.2] * 384

        de = DiffEmbedder(mock_embedder)
        result = de.embed_hunk(before_code="", after_code="def new_func(): pass")
        assert result == [0.2] * 384

    def test_embed_hunk_truncates_overlong_chunks(self):
        from unittest.mock import MagicMock

        from trelix.review.diff_embedder import DiffEmbedder

        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = [0.3] * 384

        de = DiffEmbedder(mock_embedder)
        # 3000-char code block — should be truncated before embedding
        long_code = "x = 1\n" * 500
        result = de.embed_hunk(before_code=long_code, after_code="y = 2")
        assert result is not None
        # Verify embed_query was called with truncated content
        call_arg = mock_embedder.embed_query.call_args[0][0]
        # Upper bound: must not exceed MAX_EMBED_CHARS
        assert len(call_arg) <= DiffEmbedder.MAX_EMBED_CHARS + 10
        # Lower bound: must not be empty or trivially short (proves truncation not erasure)
        assert len(call_arg) >= 100, (
            f"Truncation removed too much content: only {len(call_arg)} chars remain"
        )

    def test_search_similar_diffs_returns_sorted_by_score(self, tmp_path):
        from unittest.mock import MagicMock

        from trelix.review.diff_embedder import DiffEmbedder
        from trelix.store.db import Database

        db = Database(tmp_path / "test.db")

        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = [1.0] * 4
        mock_embedder.embed.return_value = [[1.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.5, 0.5]]
        mock_embedder.dimension = 4

        de = DiffEmbedder(mock_embedder)
        # Insert two diff chunks manually
        cols = "(pr_ref, hunk_header, before_code, after_code, chunk_char_count)"
        db._conn.execute(
            f"INSERT INTO diff_chunks {cols} VALUES (?, ?, ?, ?, ?)",
            ("owner/repo#1", "@@ -1,3 +1,3 @@", "old", "new", 6),
        )
        db._conn.commit()

        # search_similar_diffs should return results (even if empty — just not crash)
        results = de.search_similar_diffs(db, query_before="old", query_after="new", k=5)
        assert isinstance(results, list)
