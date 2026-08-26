"""L2 normalisation of NomicCodeEmbedder output, on BOTH encode paths.

CodeRankEmbed's `modules.json` is [Transformer, Pooling] with no `2_Normalize` stage, so
nothing normalises unless the caller asks. trelix asks, via `normalize_embeddings=True`,
because the sqlite vec0 table declares no distance metric and sqlite-vec defaults to L2 —
which is monotone in cosine only on unit vectors, and cosine is what the model's
`similarity_fn_name` declares. Deleting that kwarg from either path therefore corrupts
ranking silently, and until this file nothing in the suite noticed.

The assertion is on the returned VECTOR's L2 norm rather than on the kwarg dict, because the
unit norm is the property the vector store depends on and the kwarg is only the mechanism.

BE HONEST ABOUT WHAT THAT BUYS, THOUGH. The encoder below is hand-written, and it decides
the norm FROM the kwarg — so this is not an independent measurement of normalisation; it is
isomorphic to `assert kwargs["normalize_embeddings"]` with extra steps. It discriminates,
and it kills every plausible mutation of that kwarg (None / False / deleted / applied to
only one of the two paths), which is what it is for. But it cannot detect a world where the
kwarg is passed and sentence-transformers ignores it. Only real weights could, and they are
not loadable offline. Do not read the norm assertion as stronger evidence than that.

Real weights are not loadable here (CodeRankEmbed needs einops + remote code + a download,
and the unit suite is offline), so the encoder is a plain hand-written class — never
MagicMock, which would answer to any kwarg name and could not distinguish an absent
`normalize_embeddings` from a misspelled one. The fake reproduces the one behaviour under
test and nothing else: with `normalize_embeddings` truthy it returns a unit vector, and
without it the raw un-normalised vector, exactly as sentence-transformers' `encode` does.

`test_the_fake_encoder_actually_discriminates` is the precondition guard for that fake. If
someone ever "simplifies" it into returning the same vector both ways, every other test in
this file would pass BY CONSTRUCTION — so that test fails first and says why.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from trelix.core.config import EmbedderConfig
from trelix.embedder.base import REMOTE_MODEL_CODE_ENV_VAR

if TYPE_CHECKING:
    from trelix.embedder.nomic_code import NomicCodeEmbedder

# Two vectors chosen so the norms are exact in binary floating point and different by a
# wide margin: (0.6, 0.8, 0.0) has L2 norm 1.0, (3.0, 4.0, 0.0) has L2 norm 5.0. Written
# as literals here rather than derived from anything in the module under test.
_UNIT_VECTOR = [0.6, 0.8, 0.0]
_UNNORMALISED_VECTOR = [3.0, 4.0, 0.0]
_UNNORMALISED_L2_NORM = 5.0


def _l2_norm(vector: list[float]) -> float:
    return math.sqrt(sum(component * component for component in vector))


class _NormalizeAwareEncoder:
    """Hand-written stand-in for a loaded SentenceTransformer.

    Mirrors the single real behaviour under test: `encode(..., normalize_embeddings=True)`
    returns unit-norm vectors, and any other state of that kwarg (absent, None, False)
    returns the raw pooled vector, which for CodeRankEmbed's [Transformer, Pooling] module
    stack is un-normalised. Every call's kwargs are recorded so a test can show which
    branch it took.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def encode(self, texts: list[str], **kwargs: object) -> list[list[float]]:
        self.calls.append(dict(kwargs))
        normalize = bool(kwargs.get("normalize_embeddings", False))
        row = list(_UNIT_VECTOR) if normalize else list(_UNNORMALISED_VECTOR)
        return [list(row) for _ in texts]

    def get_sentence_embedding_dimension(self) -> int:
        return len(_UNIT_VECTOR)


class TestNomicCodeEmbedderNormalisesOutput:
    @pytest.fixture(autouse=True)
    def _remote_model_code_opt_in(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Grant SEC-03c's opt-in for this class only.

        nomic-code loads CodeRankEmbed with remote code trusted, which the gate in
        embedder/base.py refuses unless the operator set the variable in the real process
        environment. Scoped to this class deliberately: in conftest.py it would blind the
        whole suite to the gate regressing.
        """
        monkeypatch.setenv(REMOTE_MODEL_CODE_ENV_VAR, "1")

    def _build(self, encoder: _NormalizeAwareEncoder) -> NomicCodeEmbedder:
        from trelix.embedder.nomic_code import NomicCodeEmbedder as _NomicCodeEmbedder

        with patch("trelix.embedder.nomic_code.SentenceTransformer", return_value=encoder):
            cfg = EmbedderConfig(provider="nomic-code", _env_file=None)
            return _NomicCodeEmbedder(cfg)

    # ── Precondition: the fixture must be able to tell the two worlds apart ───────────

    def test_the_fake_encoder_actually_discriminates(self) -> None:
        """Guard for rule 4 — no MUTATION of src/ makes this fail; a weakened fake does.

        If `_NormalizeAwareEncoder.encode` is ever changed to return the same vector
        whether or not `normalize_embeddings` is passed, the two tests below would pass BY
        CONSTRUCTION and could no longer kill
        `normalize_embeddings=True` -> `None` / `False`. This fails first, with a message
        pointing at the fake rather than at the provider.
        """
        encoder = _NormalizeAwareEncoder()

        normalised = encoder.encode(["x"], normalize_embeddings=True)[0]
        absent = encoder.encode(["x"])[0]
        explicit_false = encoder.encode(["x"], normalize_embeddings=False)[0]
        explicit_none = encoder.encode(["x"], normalize_embeddings=None)[0]

        assert _l2_norm(normalised) == pytest.approx(1.0, abs=1e-9)
        for label, vector in (
            ("absent", absent),
            ("False", explicit_false),
            ("None", explicit_none),
        ):
            assert _l2_norm(vector) == pytest.approx(_UNNORMALISED_L2_NORM, abs=1e-9), (
                f"the fake encoder stopped discriminating for normalize_embeddings={label}: "
                f"it returned {vector!r} (norm {_l2_norm(vector)}). While that is true, the "
                "normalisation tests in this file prove nothing."
            )

    # ── The property the vector store depends on ──────────────────────────────────────

    def test_embed_returns_unit_norm_document_vectors(self) -> None:
        """MUTATION: in `NomicCodeEmbedder.embed`, change `normalize_embeddings=True` to
        `None` (or `False`, or delete the kwarg) and this fails — the returned document
        vectors come back with L2 norm 5.0 instead of 1.0.
        """
        encoder = _NormalizeAwareEncoder()
        embedder = self._build(encoder)

        vectors = embedder.embed(["def login(user): ...", "class Session: ..."])

        assert len(vectors) == 2
        for index, vector in enumerate(vectors):
            norm = _l2_norm(vector)
            assert norm == pytest.approx(1.0, abs=1e-6), (
                f"document vector {index} is not L2-normalised: norm={norm}, "
                f"vector={vector!r}. The sqlite vec0 table declares no metric and "
                "sqlite-vec defaults to L2, which only ranks like cosine on unit vectors."
            )

    def test_embed_query_returns_a_unit_norm_vector(self) -> None:
        """MUTATION: in `NomicCodeEmbedder.embed_query`, change `normalize_embeddings=True`
        to `None` (or `False`, or delete the kwarg) and this fails — the query vector comes
        back with L2 norm 5.0 instead of 1.0.
        """
        encoder = _NormalizeAwareEncoder()
        embedder = self._build(encoder)

        vector = embedder.embed_query("where is authentication handled")

        norm = _l2_norm(vector)
        assert norm == pytest.approx(1.0, abs=1e-6), (
            f"the query vector is not L2-normalised: norm={norm}, vector={vector!r}. An "
            "un-normalised query against normalised documents scores every candidate on a "
            "different scale than cosine."
        )

    def test_documents_and_queries_are_normalised_on_the_same_scale(self) -> None:
        """MUTATION: normalise on only ONE path — e.g. leave `embed` at
        `normalize_embeddings=True` but set `embed_query` to `None` — and this fails even
        though each path is individually self-consistent.

        Asymmetric normalisation is the silent failure this provider's own docstring warns
        about ("Both encode paths must pass it or queries and documents would live on
        different scales"), and it is invisible to any test that checks one path alone.
        """
        encoder = _NormalizeAwareEncoder()
        embedder = self._build(encoder)

        document_norm = _l2_norm(embedder.embed(["def login(user): ..."])[0])
        query_norm = _l2_norm(embedder.embed_query("where is authentication handled"))

        assert document_norm == pytest.approx(query_norm, abs=1e-6), (
            f"documents and queries live on different scales: document norm={document_norm}, "
            f"query norm={query_norm}. Distances between them are then meaningless."
        )
        assert document_norm == pytest.approx(1.0, abs=1e-6)
