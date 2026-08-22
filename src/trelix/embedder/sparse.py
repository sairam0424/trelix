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

        embedder = SparseEmbedder("naver/splade-v3-distilbert", top_k=128)
        sparse_vecs = embedder.embed(["def login(user, pw): ...", "..."])
        # Returns: [{token_id: weight, ...}, ...]
    """

    def __init__(
        self,
        model_name: str = "naver/splade-v3-distilbert",
        top_k: int = 128,
        batch_size: int = 16,
    ) -> None:
        self._model_name = model_name
        self._top_k = top_k
        # A MaskedLM emits logits of shape (batch, seq_len, vocab_size), so memory grows
        # linearly with the batch. Unbatched, this repository's own 10,700 chunks at
        # max_length=512 against a 30,522-token vocabulary would need a 668 GB tensor;
        # 200 chunks already needs 12.5 GB. The batch is the only thing bounding it.
        self._batch_size = max(1, batch_size)
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
                if self._model_name.startswith("naver/splade"):
                    # Stated once, on load, because the obligation attaches to whoever
                    # downloads the weights and nothing else in the pipeline would
                    # mention it. trelix is MIT and does not redistribute them.
                    logger.warning(
                        "%s is published under CC BY-NC-SA-4.0 (non-commercial). "
                        "Set TRELIX_SPARSE_MODEL to a permissively-licensed model, or "
                        "leave TRELIX_RETRIEVAL_SPARSE off, for commercial use.",
                        self._model_name,
                    )
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
            # Chunked, because a MaskedLM's logits are (batch, seq_len, vocab_size) and
            # one pass over a whole corpus does not fit in memory — see __init__.
            # Batches are appended in order, so `results` stays positionally aligned
            # with `texts`; the indexer zips these against its pending chunks by
            # position, and a reordering would attach every vector to the wrong chunk.
            for start in range(0, len(texts), self._batch_size):
                batch = texts[start : start + self._batch_size]
                inputs = self._tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                )
                with torch.no_grad():
                    outputs = self._model(**inputs)

                # SPLADE aggregation: log(1 + ReLU(logits)), masked, then max over sequence.
                #
                # The mask is load-bearing, not defensive. `padding=True` above pads every
                # batch to its own longest member, and a MaskedLM emits real logits at PAD
                # positions too — it predicts a distribution for every position, including
                # the ones that only exist because a longer chunk shares the batch. Maxing
                # over them mixes those predictions into the stored vector, so a chunk's
                # sparse representation came to depend on WHICH OTHER CHUNKS happened to be
                # batched with it. Measured on naver/splade-v3-distilbert with top_k=128: a
                # 28-token chunk batched with a 185-token one gained 28 phantom terms and
                # lost 28 real ones at the top_k cut — 22% of the vector — with 0.478 max
                # weight drift on the terms that survived. The phantom tokens are ordinary
                # English the chunk never contained ('where', 'gene', 'sequence', 'phrase').
                # The positive control is what pins the cause: the same chunk batched with an
                # EQUAL-length chunk, needing no padding, is bit-identical to it embedded
                # alone. So batching was never the problem; padding was.
                #
                # Two consequences made this worth fixing rather than documenting. The
                # vectors are PERSISTED (store/db.py `sparse_embeddings`), so an index was
                # not reproducible — re-indexing the same repository with a different file
                # order or a different TRELIX_SPARSE_BATCH_SIZE rewrote every row. And
                # `embed_query` routes through `embed([text])`, a batch of one that needs no
                # padding, so queries were always clean and were scored against contaminated
                # documents — an asymmetry no amount of query tuning could correct.
                #
                # Multiplying is sound because these weights are non-negative by
                # construction (log1p of a ReLU), so zeroing the padded positions can never
                # win a max against a real one. This is also NAVER's own formulation.
                # `torch.log(1 + ...)` is kept verbatim rather than switched to `log1p`: the
                # mask is the only intended change to the numbers.
                logits = outputs.logits  # (batch, seq_len, vocab_size)
                weighted = torch.log(1 + torch.relu(logits))
                mask = inputs["attention_mask"].unsqueeze(-1)  # (batch, seq_len, 1)
                agg = (weighted * mask).max(dim=1).values  # (batch, vocab)

                for i in range(len(batch)):
                    scores = agg[i]  # (vocab_size,)
                    topk_vals, topk_ids = torch.topk(scores, k=min(self._top_k, scores.shape[0]))
                    vec: dict[int, float] = {}
                    for tok_id, weight in zip(topk_ids.tolist(), topk_vals.tolist()):
                        if weight > 0.0:
                            vec[int(tok_id)] = float(weight)
                    results.append(vec)
            return results
        except Exception as exc:
            # WARNING, not DEBUG. The degradation is deliberate — a failing sparse leg must
            # not fail the whole index — but the failure mode is indistinguishable from
            # success at a glance: every chunk gets `{}`, the dot product over an empty
            # vector is 0, and the leg contributes nothing while reporting nothing. At DEBUG
            # the CLI's default level hid it entirely, so the sparse index could be
            # uniformly empty with no signal anywhere. It also made tests over this function
            # vacuous: any assertion comparing two embeddings passes as `{} == {}` when both
            # sides crashed, which is why the tests below assert the vectors are non-empty
            # before comparing them.
            logger.warning(
                "SparseEmbedder.embed() failed on a batch of %d; those chunks get empty "
                "sparse vectors and contribute nothing to the sparse leg: %s",
                len(texts),
                exc,
            )
            return [{} for _ in texts]

    def embed_query(self, text: str) -> dict[int, float]:
        """Embed a single query string as a sparse vector."""
        results = self.embed([text])
        return results[0] if results else {}
