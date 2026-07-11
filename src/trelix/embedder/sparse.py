"""
SPLADE-Code sparse embedder — learned sparse retrieval for code.

Produces {token_id: weight} sparse vectors using a SPLADE-variant model.
High weights indicate which vocabulary tokens are most important for a text.
At search time, dot-product over overlapping tokens provides relevance scores.

Research basis: SPLADE-Code (Lupart et al., NAVER Labs Europe, arXiv:2603.22008).
Addresses BM25's failures on code: identifier subword fragmentation and
NL/code vocabulary mismatch.

Requires: pip install trelix[sparse]  (installs transformers + torch)
"""

from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger("trelix.embedder.sparse")

# Flag checked before any torch/transformers import to allow graceful degradation
try:
    import torch  # noqa: F401
    from transformers import AutoModelForMaskedLM, AutoTokenizer  # noqa: F401

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


class SparseEmbedder:
    """
    SPLADE-style sparse embedder for code retrieval.

    Returns sparse {token_id: weight} vectors. Only the top_k highest-weight
    tokens are kept to control index size.

    Usage::

        embedder = SparseEmbedder("naver-splab/splade-code-distil", top_k=128)
        sparse_vecs = embedder.embed(["def login(user, pw): ...", "..."])
        # Returns: [{token_id: weight, ...}, ...]
    """

    def __init__(
        self,
        model_name: str = "naver-splab/splade-code-distil",
        top_k: int = 128,
    ) -> None:
        self._model_name = model_name
        self._top_k = top_k
        self._model: Any = None
        self._tokenizer: Any = None
        self._lock = threading.Lock()

    def _load(self) -> bool:
        """Lazy-load model and tokenizer. Returns True if successful.

        Thread-safe via double-checked locking: the outer check avoids lock
        contention on the common already-loaded path; the inner re-check
        (held under self._lock) closes the TOCTOU race where two threads
        could otherwise both observe self._model is None and both call
        from_pretrained concurrently.
        """
        if self._model is not None:
            return True
        if not _TORCH_AVAILABLE:
            logger.debug("SparseEmbedder: torch/transformers not installed")
            return False
        with self._lock:
            if self._model is not None:  # re-check inside the lock
                return True
            try:
                from transformers import AutoModelForMaskedLM, AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
                self._model = AutoModelForMaskedLM.from_pretrained(self._model_name)
                self._model.eval()
                logger.info("SparseEmbedder loaded: %s", self._model_name)
                return True
            except Exception as exc:
                logger.warning("SparseEmbedder failed to load %s: %s", self._model_name, exc)
                return False

    def embed(self, texts: list[str]) -> list[dict[int, float]]:
        """
        Embed a batch of texts as sparse vectors.

        Returns list of {token_id: weight} dicts.
        Returns [{} * len(texts)] on failure.
        """
        if not texts:
            return []
        if not self._load():
            return [{} for _ in texts]

        try:
            import torch

            results: list[dict[int, float]] = []
            inputs = self._tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            with torch.no_grad():
                outputs = self._model(**inputs)

            # SPLADE aggregation: max over sequence, then log(1 + ReLU(logits))
            logits = outputs.logits  # (batch, seq_len, vocab_size)
            agg = torch.log(1 + torch.relu(logits)).max(dim=1).values  # (batch, vocab_size)

            for i in range(len(texts)):
                scores = agg[i]  # (vocab_size,)
                topk_vals, topk_ids = torch.topk(scores, k=min(self._top_k, scores.shape[0]))
                vec: dict[int, float] = {}
                for tok_id, weight in zip(topk_ids.tolist(), topk_vals.tolist()):
                    if weight > 0.0:
                        vec[int(tok_id)] = float(weight)
                results.append(vec)
            return results
        except Exception as exc:
            logger.debug("SparseEmbedder.embed() failed: %s", exc)
            return [{} for _ in texts]

    def embed_query(self, text: str) -> dict[int, float]:
        """Embed a single query string as a sparse vector."""
        results = self.embed([text])
        return results[0] if results else {}
