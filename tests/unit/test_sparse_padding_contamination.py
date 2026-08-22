"""A chunk's stored sparse vector must not depend on which chunks share its batch.

WHY THIS EXISTS. `SparseEmbedder.embed` tokenizes with `padding=True`, so every batch is
padded to its own longest member. A MaskedLM emits real logits at PAD positions — it
predicts a distribution for every position, including ones that exist only because a
longer chunk shares the batch — and the SPLADE aggregation used to max over all of them.
So the vector written to the `sparse_embeddings` table depended on batch composition.

Measured on the real naver/splade-v3-distilbert with top_k=128, before the fix: a
28-token chunk embedded alone versus batched with a 185-token chunk gained 28 phantom
terms and lost 28 real ones at the top_k cut — 22% of the vector — with 0.478 max weight
drift on the terms that survived. The phantom tokens were ordinary English the chunk never
contained: 'where', 'store', 'numbers', 'text', 'gene', 'sequence', 'phrase', 'messages'.

The positive control is what pinned the cause: the same chunk batched with an EQUAL-length
chunk, needing no padding, was bit-identical to it embedded alone. Batching was never the
problem; padding was.

WHY IT MATTERED, twice over. The vectors are persisted, so an index was not reproducible —
re-indexing the same repository with a different file order or a different
TRELIX_SPARSE_BATCH_SIZE rewrote every row. And `embed_query` routes through
`embed([text])`, a batch of one that needs no padding, so queries were always clean and
were scored against contaminated documents. No amount of query tuning corrects that
asymmetry.

WHY THE EXISTING TESTS COULD NOT CATCH IT. Every fake tokenizer in
tests/unit/test_sparse_embedder.py returns `attention_mask=torch.ones(...)`. An all-ones
mask makes the masked and unmasked aggregations identical by construction, so all ten of
those tests pass either way. The fixture below pads for real, and one of the tests asserts
that it does — because a fixture that quietly stopped padding would make every other
assertion here vacuous.

The fakes are plain classes, not MagicMock, deliberately. A MagicMock answers to any
attribute asked of it, which is exactly how this provider family shipped a call to a
method FlagEmbedding has never had; a fake that cannot be wrong about its own interface is
not a fake worth having.
"""

from __future__ import annotations

import os
import pathlib

import pytest

# Vocabulary is tiny so the top_k cut is easy to reason about.
_VOCAB = 12
_PAD_ID = 0
# The id the fake model predicts at PAD positions. Nothing else ever emits it, so its
# presence in an output vector is unambiguous evidence that padding leaked into the max.
_PHANTOM_ID = 7
_PAD_LOGIT = 5.0
_REAL_LOGIT = 3.0


class _PaddingTokenizer:
    """Pads a batch to its longest member, exactly as `padding=True` does.

    Returns a real `attention_mask` with real zeros — the whole point of the fixture.
    """

    def __call__(self, texts: list[str], **kwargs: object) -> dict[str, object]:
        import torch

        # One token per word, +2 for the [CLS]/[SEP] a BERT-family tokenizer adds. Content
        # ids are derived from the word so two different texts differ, and are kept clear
        # of _PAD_ID and _PHANTOM_ID so neither can be produced by accident.
        rows: list[list[int]] = []
        for text in texts:
            ids = [1] + [2 + (sum(map(ord, w)) % 4) for w in text.split()] + [3]
            rows.append(ids)
        width = max(len(r) for r in rows)
        input_ids = torch.full((len(rows), width), _PAD_ID, dtype=torch.long)
        attention_mask = torch.zeros((len(rows), width), dtype=torch.long)
        for i, r in enumerate(rows):
            input_ids[i, : len(r)] = torch.tensor(r, dtype=torch.long)
            attention_mask[i, : len(r)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class _MaskedLMOutput:
    def __init__(self, logits: object) -> None:
        self.logits = logits


class _PhantomAtPadModel:
    """A MaskedLM that predicts _PHANTOM_ID wherever the input token is PAD.

    This is not a contrivance: PAD is a token like any other, and a real MaskedLM does
    produce a distribution from it. Keying off `input_ids == _PAD_ID` rather than off the
    attention mask keeps the fake honest — it does not know which positions are "supposed"
    to be ignored, any more than the real model does.
    """

    def __call__(self, **inputs: object) -> _MaskedLMOutput:
        import torch

        input_ids = inputs["input_ids"]
        batch, width = input_ids.shape  # type: ignore[union-attr]
        logits = torch.zeros((batch, width, _VOCAB))
        is_pad = input_ids == _PAD_ID  # type: ignore[operator]
        for b in range(batch):
            for p in range(width):
                if is_pad[b, p]:
                    logits[b, p, _PHANTOM_ID] = _PAD_LOGIT
                else:
                    logits[b, p, int(input_ids[b, p]) % _VOCAB] = _REAL_LOGIT  # type: ignore[index]
        return _MaskedLMOutput(logits)


def _embedder(batch_size: int = 16):  # type: ignore[no-untyped-def]
    from trelix.embedder.sparse import SparseEmbedder

    emb = SparseEmbedder(model_name="stub", top_k=_VOCAB, batch_size=batch_size)
    emb._tokenizer = _PaddingTokenizer()
    emb._model = _PhantomAtPadModel()
    return emb


_SHORT = "alpha beta"
_LONG = "gamma delta epsilon zeta eta theta iota kappa lambda mu"
_EQUAL = "sigma tau"


class TestTheFixtureItself:
    """If these fail, every assertion in the next class is vacuous."""

    def test_the_fixture_really_pads(self) -> None:
        """A fixture that stopped padding would make the contamination untestable."""
        pytest.importorskip("torch")
        inputs = _PaddingTokenizer()([_SHORT, _LONG])
        mask = inputs["attention_mask"]

        assert mask.shape[0] == 2  # type: ignore[union-attr]
        assert int(mask.min()) == 0, (
            "the fixture produced no padding at all, so masking and not masking are the "
            "same operation here and the tests below prove nothing"
        )
        assert int(mask[0].sum()) < mask.shape[1], "the short row was not padded"  # type: ignore[union-attr]
        assert int(mask[1].sum()) == mask.shape[1], "the long row should need no padding"  # type: ignore[union-attr]

    def test_the_model_really_emits_the_phantom_at_pad_positions(self) -> None:
        """Otherwise a mask-less aggregation would look clean for the wrong reason."""
        pytest.importorskip("torch")
        inputs = _PaddingTokenizer()([_SHORT, _LONG])
        logits = _PhantomAtPadModel()(**inputs).logits

        pad_positions = (inputs["input_ids"] == _PAD_ID).nonzero()  # type: ignore[operator]
        assert len(pad_positions) > 0, "no PAD positions — fixture is not exercising the path"
        b, p = int(pad_positions[0][0]), int(pad_positions[0][1])
        assert float(logits[b, p, _PHANTOM_ID]) == _PAD_LOGIT

    def test_vectors_are_non_empty(self) -> None:
        """`embed` swallows exceptions and returns `{}` per chunk.

        Without this, a crash inside embed() turns every comparison below into
        `{} == {}` and the suite reports success over code that never ran.
        """
        emb = _embedder()
        vecs = emb.embed([_SHORT, _LONG])

        assert len(vecs) == 2
        assert all(v for v in vecs), (
            "embed() returned an empty vector, which means it raised and logged rather "
            "than aggregating — the comparisons in this file would be meaningless"
        )


class TestPaddingDoesNotLeakIntoTheStoredVector:
    def test_the_phantom_token_never_appears(self) -> None:
        """The direct assertion: a padded row must not carry PAD-position predictions.

        MUTATION: drop `* mask` from the aggregation in sparse.py and this fails —
        _PHANTOM_ID enters with weight log(1 + 5.0), the largest in the row.
        """
        emb = _embedder()

        alone = emb.embed([_SHORT])[0]
        batched = emb.embed([_SHORT, _LONG])[0]

        assert _PHANTOM_ID not in alone, "unpadded row should never see the phantom"
        assert _PHANTOM_ID not in batched, (
            f"token {_PHANTOM_ID} is only ever predicted at PAD positions, so its presence "
            "means the aggregation maxed over padding: a chunk's stored vector depends on "
            "which other chunks share its batch"
        )

    def test_a_chunk_embeds_identically_alone_and_beside_a_longer_chunk(self) -> None:
        """The property that actually matters: batch composition must not change a vector.

        MUTATION: drop `* mask` and the two dicts differ.
        """
        emb = _embedder()

        alone = emb.embed([_SHORT])[0]
        batched = emb.embed([_SHORT, _LONG])[0]

        assert alone, "baseline is empty — see TestTheFixtureItself"
        assert batched == alone, (
            "the same text produced different sparse vectors depending on its batch; "
            f"gained {sorted(set(batched) - set(alone))}, lost {sorted(set(alone) - set(batched))}"
        )

    def test_an_equal_length_batch_was_never_affected(self) -> None:
        """The control that localises the cause to padding rather than to batching.

        This one passed BEFORE the fix too, and that is the point: it is what proves the
        failure above was caused by padding specifically. A test suite where every
        assertion changes with the fix cannot distinguish the cause from a coincidence.
        """
        emb = _embedder()

        alone = emb.embed([_SHORT])[0]
        with_equal = emb.embed([_SHORT, _EQUAL])[0]

        assert alone, "baseline is empty — see TestTheFixtureItself"
        assert with_equal == alone

    def test_batch_size_does_not_change_any_vector(self) -> None:
        """Re-indexing with a different TRELIX_SPARSE_BATCH_SIZE must be a no-op.

        Before the fix this was false for every chunk shorter than its batch's longest,
        which is what made a persisted sparse index non-reproducible.
        """
        texts = [_SHORT, _LONG, _EQUAL, "nu xi omicron pi rho"]

        one_at_a_time = _embedder(batch_size=1).embed(texts)
        all_together = _embedder(batch_size=16).embed(texts)

        assert all(one_at_a_time), "per-chunk baseline is empty"
        assert all_together == one_at_a_time, (
            "changing only the batch size changed the stored vectors, so the sparse index "
            "is not reproducible across runs"
        )

    def test_the_query_path_agrees_with_the_document_path(self) -> None:
        """`embed_query` is a batch of one, so it was always clean.

        That asymmetry — clean queries scored against contaminated documents — is the part
        no query-side tuning could correct, so it is worth pinning that the two agree.
        """
        emb = _embedder()

        as_query = emb.embed_query(_SHORT)
        as_document = emb.embed([_SHORT, _LONG])[0]

        assert as_query, "query vector is empty"
        assert as_document == as_query


_SPLADE_SNAPSHOT = (
    pathlib.Path.home() / ".cache/huggingface/hub/models--naver--splade-v3-distilbert"
)


@pytest.mark.skipif(
    not _SPLADE_SNAPSHOT.is_dir(),
    reason="real splade weights not cached; the fake-model tests above are the portable proof",
)
def test_real_weights_agree_alone_and_batched(monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end on the published model, strictly offline — never downloads.

    The fake-model tests are the ones that must always run; this exists because the
    original measurement was made here, and a fake that drifted from the real model's
    behaviour would otherwise go unnoticed.
    """
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    from trelix.embedder.sparse import SparseEmbedder

    emb = SparseEmbedder(top_k=128, batch_size=16)
    if not emb._load():
        pytest.skip("cached splade snapshot could not be loaded offline")

    short = "def hash_bytes(data): return hashlib.sha256(data).hexdigest()"
    long_chunk = "\n".join(f"    self.field_{i} = compute_{i}(self.state, {i})" for i in range(40))

    alone = emb.embed([short])[0]
    batched = emb.embed([short, long_chunk])[0]

    assert alone, "real-weight baseline is empty"
    assert len(emb._tokenizer(short)["input_ids"]) < len(emb._tokenizer(long_chunk)["input_ids"]), (
        "fixture chunks are not different lengths, so no padding occurs and this proves nothing"
    )
    assert batched == alone, (
        f"real weights still contaminate: gained {len(set(batched) - set(alone))} phantom "
        f"terms, lost {len(set(alone) - set(batched))}"
    )
