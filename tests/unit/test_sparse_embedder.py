"""Tests for SparseEmbedder (SPLADE-Code)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


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

        # AutoModelForMaskedLM/AutoTokenizer are only in the sparse module's namespace
        # when transformers is installed and importable.  When not available the module
        # falls back to _TORCH_AVAILABLE=False; patch _TORCH_AVAILABLE and inject the
        # fake model/tokenizer directly to test the embedding path.
        embedder = SparseEmbedder("test-model", top_k=128)
        embedder._model = mock_model
        embedder._tokenizer = mock_tokenizer

        with patch("trelix.embedder.sparse._TORCH_AVAILABLE", True):
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

        # Inject mock model/tokenizer directly; patch _TORCH_AVAILABLE=True so
        # the embed path runs without needing transformers installed.
        embedder = SparseEmbedder("test-model", top_k=128)
        embedder._model = mock_model
        embedder._tokenizer = mock_tokenizer

        with patch("trelix.embedder.sparse._TORCH_AVAILABLE", True):
            result = embedder.embed_query("how does auth work")

        assert isinstance(result, dict)


class TestSparseEmbedderThreadSafety:
    def test_load_is_thread_safe_under_concurrent_calls(self):
        """Multiple threads calling _load() simultaneously must only load the model once."""
        import threading
        import time
        from unittest.mock import MagicMock, patch

        from trelix.embedder.sparse import SparseEmbedder

        embedder = SparseEmbedder(model_name="fake-model")
        call_count = {"tokenizer": 0, "model": 0}

        def slow_tokenizer_from_pretrained(*args, **kwargs):
            call_count["tokenizer"] += 1
            time.sleep(0.05)  # widen the race window
            return MagicMock()

        def slow_model_from_pretrained(*args, **kwargs):
            call_count["model"] += 1
            time.sleep(0.05)
            mock_model = MagicMock()
            mock_model.eval = MagicMock()
            return mock_model

        with (
            patch("trelix.embedder.sparse._TORCH_AVAILABLE", True),
            patch(
                "transformers.AutoTokenizer.from_pretrained",
                side_effect=slow_tokenizer_from_pretrained,
            ),
            patch(
                "transformers.AutoModelForMaskedLM.from_pretrained",
                side_effect=slow_model_from_pretrained,
            ),
        ):
            threads = [threading.Thread(target=embedder._load) for _ in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        assert call_count["tokenizer"] == 1, (
            f"Expected exactly 1 tokenizer load, got {call_count['tokenizer']}"
        )
        assert call_count["model"] == 1, f"Expected exactly 1 model load, got {call_count['model']}"


class TestSparseEmbedderBatches:
    """`embed()` must not put the whole corpus through one forward pass.

    It tokenized the entire `texts` list into a single tensor and ran one
    `model(**inputs)`, producing logits of shape (batch, seq_len, vocab_size). With
    max_length=512 and DistilBERT's 30522-token vocabulary that is

        10,700 chunks x 512 x 30522 x 4 bytes = 668.8 GB

    for this repository's own index — and 12.5 GB at just 200 chunks. The phase could
    not run on any real corpus regardless of which model was configured.

    `SparseConfig.batch_size` existed for this and was referenced NOWHERE in src/; the
    embedder's constructor did not even accept it, although docs/architecture.md
    documented `batch_size=16` as part of the signature.
    """

    @staticmethod
    def _embedder_with_counting_model(batch_size: int):  # type: ignore[no-untyped-def]
        """A SparseEmbedder whose model records the batch size of every call."""
        from unittest.mock import MagicMock

        import torch

        from trelix.embedder.sparse import SparseEmbedder

        emb = SparseEmbedder(model_name="stub", top_k=4, batch_size=batch_size)
        calls: list[int] = []

        def tokenizer(texts, **kwargs):  # type: ignore[no-untyped-def]
            n = len(texts)
            calls.append(n)
            return {"input_ids": torch.ones((n, 6), dtype=torch.long),
                    "attention_mask": torch.ones((n, 6), dtype=torch.long)}

        def model(**inputs):  # type: ignore[no-untyped-def]
            n = inputs["input_ids"].shape[0]
            return MagicMock(logits=torch.rand((n, 6, 50)))

        emb._tokenizer = tokenizer
        emb._model = model
        return emb, calls

    def test_a_large_batch_is_split(self) -> None:
        emb, calls = self._embedder_with_counting_model(batch_size=4)
        texts = [f"def f{i}(): pass" for i in range(10)]

        vecs = emb.embed(texts)

        assert len(vecs) == 10, "every input must still get a vector back"
        assert calls == [4, 4, 2], f"expected three batches of 4/4/2, got {calls}"

    def test_batch_size_of_one_still_works(self) -> None:
        emb, calls = self._embedder_with_counting_model(batch_size=1)
        vecs = emb.embed(["a", "b", "c"])

        assert len(vecs) == 3
        assert calls == [1, 1, 1]

    def test_a_small_input_is_a_single_batch(self) -> None:
        emb, calls = self._embedder_with_counting_model(batch_size=16)
        emb.embed(["only one"])
        assert calls == [1]

    def test_results_stay_in_input_order(self) -> None:
        """Batching must not permute the results relative to their chunk ids.

        The indexer zips these vectors against `pending` positionally, so a reordering
        would attach every sparse vector to the wrong chunk.
        """
        from unittest.mock import MagicMock

        import torch

        from trelix.embedder.sparse import SparseEmbedder

        emb = SparseEmbedder(model_name="stub", top_k=1, batch_size=2)

        def tokenizer(texts, **kwargs):  # type: ignore[no-untyped-def]
            # Encode each text's index in the input_ids so the model can echo it back.
            ids = torch.tensor([[int(t)] * 3 for t in texts], dtype=torch.long)
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

        def model(**inputs):  # type: ignore[no-untyped-def]
            n = inputs["input_ids"].shape[0]
            logits = torch.zeros((n, 3, 50))
            for row in range(n):
                # Make the argmax token equal to that text's index.
                logits[row, :, int(inputs["input_ids"][row][0])] = 5.0
            return MagicMock(logits=logits)

        emb._tokenizer = tokenizer
        emb._model = model

        vecs = emb.embed([str(i) for i in range(6)])

        top_tokens = [max(v, key=v.get) for v in vecs]
        assert top_tokens == list(range(6)), f"results were reordered: {top_tokens}"

    def test_default_batch_size_is_bounded(self) -> None:
        """The default must be small enough that one batch cannot exhaust memory."""
        from trelix.embedder.sparse import SparseEmbedder

        emb = SparseEmbedder(model_name="stub")
        assert 1 <= emb._batch_size <= 64
