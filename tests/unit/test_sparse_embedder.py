"""Tests for SparseEmbedder (SPLADE-Code)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestSparseEmbedder:
    def test_import_without_torch(self) -> None:
        """SparseEmbedder must be importable even without torch installed."""
        from trelix.embedder.sparse import SparseEmbedder
        assert SparseEmbedder is not None

    def test_embed_returns_sparse_dicts_when_model_mocked(self) -> None:
        import torch
        from trelix.embedder.sparse import SparseEmbedder

        mock_model = MagicMock()
        # Simulate SPLADE output: logsparsity activations
        mock_output = MagicMock()
        mock_output.logits = torch.zeros(2, 30522)  # batch=2, vocab_size=30522
        mock_output.logits[0, 100] = 2.5
        mock_output.logits[0, 200] = 1.8
        mock_output.logits[1, 150] = 3.0
        mock_model.return_value = mock_output

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.zeros(2, 10, dtype=torch.long),
            "attention_mask": torch.ones(2, 10, dtype=torch.long),
        }

        with patch("trelix.embedder.sparse.AutoModelForMaskedLM", return_value=mock_model):
            with patch("trelix.embedder.sparse.AutoTokenizer", return_value=mock_tokenizer):
                embedder = SparseEmbedder("test-model", top_k=128)
                embedder._model = mock_model
                embedder._tokenizer = mock_tokenizer
                result = embedder.embed(["def login(user): ...", "class AuthService: ..."])

        assert len(result) == 2
        assert isinstance(result[0], dict)
        assert all(isinstance(k, int) and isinstance(v, float) for k, v in result[0].items())

    def test_embed_returns_empty_when_not_installed(self) -> None:
        from trelix.embedder.sparse import SparseEmbedder
        embedder = SparseEmbedder("test-model", top_k=128)
        # Without torch/transformers mocked as installed, should return empty dicts
        with patch("trelix.embedder.sparse._TORCH_AVAILABLE", False):
            result = embedder.embed(["test"])
        assert result == [{}]

    def test_embed_query_returns_dict(self) -> None:
        import torch
        from trelix.embedder.sparse import SparseEmbedder

        mock_model = MagicMock()
        mock_output = MagicMock()
        mock_output.logits = torch.zeros(1, 30522)
        mock_output.logits[0, 42] = 1.5
        mock_model.return_value = mock_output

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.zeros(1, 8, dtype=torch.long),
            "attention_mask": torch.ones(1, 8, dtype=torch.long),
        }

        with patch("trelix.embedder.sparse.AutoModelForMaskedLM", return_value=mock_model):
            with patch("trelix.embedder.sparse.AutoTokenizer", return_value=mock_tokenizer):
                embedder = SparseEmbedder("test-model", top_k=128)
                embedder._model = mock_model
                embedder._tokenizer = mock_tokenizer
                result = embedder.embed_query("how does auth work")

        assert isinstance(result, dict)
