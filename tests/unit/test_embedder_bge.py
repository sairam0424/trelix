"""Tests for BGECodeEmbedder."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from trelix.core.config import EmbedderConfig


class _FakeHFConfig:
    hidden_size = 1536  # BAAI/bge-code-v1 config.json hidden_size


class _FakeHFModel:
    config = _FakeHFConfig()


class _FakeFlagModel:
    """FlagModel's real attribute surface — and nothing more.

    Checked against the FlagEmbedding 1.3.5 sdist: `inference.embedder.encoder_only.base.
    BaseEmbedder(AbsEmbedder)` defines only __init__ / encode / encode_corpus /
    encode_queries / encode_single_device / pooling, on top of AbsEmbedder's __del__,
    encode*, get_detailed_instruct, get_target_devices and multi-process helpers.
    `get_sentence_embedding_dimension` is in neither class, in no release (1.2.11, 1.3.5,
    1.4.0), and neither defines `__getattr__`. Both bases assign `self.model =
    AutoModel.from_pretrained(...)`, so `.model.config.hidden_size` is where the width is.

    A plain class, deliberately: `MagicMock` grows whatever attribute it is asked for,
    which is how a property that raised `AttributeError` in production passed its test.
    """

    def __init__(
        self, model_name_or_path: str, truncate_dim: int | None = None, **_: object
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.truncate_dim = truncate_dim
        self.model = _FakeHFModel()
        self.tokenizer = object()

    def _vecs(self, n: int) -> list[list[float]]:
        return [[0.01] * (self.truncate_dim or _FakeHFConfig.hidden_size) for _ in range(n)]

    def encode(self, sentences: list[str], batch_size: int = 256, **_: object) -> list[list[float]]:
        return self._vecs(len(sentences))

    encode_corpus = encode

    def encode_queries(self, queries: list[str], **_: object) -> list[list[float]]:
        return self._vecs(len(queries))


class TestBGECodeEmbedder:
    def test_importable(self) -> None:
        from trelix.embedder.bge_code import BGECodeEmbedder

        assert BGECodeEmbedder is not None

    def test_is_base_embedder(self) -> None:
        from trelix.embedder.base import BaseEmbedder
        from trelix.embedder.bge_code import BGECodeEmbedder

        assert issubclass(BGECodeEmbedder, BaseEmbedder)

    def test_dimension_is_the_width_that_actually_comes_back(self) -> None:
        """1536, and read off the model rather than from a method FlagEmbedding lacks."""
        from trelix.embedder.bge_code import BGECodeEmbedder

        with patch("trelix.embedder.bge_code.FlagModel", _FakeFlagModel):
            cfg = EmbedderConfig(provider="bge-code", _env_file=None)
            emb = BGECodeEmbedder(cfg)
            assert emb.dimension == 1536
            assert len(emb.embed(["def foo(): pass"])[0]) == emb.dimension
            assert len(emb.embed_query("how does auth work")) == emb.dimension

    def test_dimension_honours_matryoshka_truncation(self) -> None:
        """`truncate_dim` narrows the output, so it must win over `hidden_size`."""
        from trelix.embedder.bge_code import BGECodeEmbedder

        def _truncated(name: str, **kwargs: object) -> _FakeFlagModel:
            return _FakeFlagModel(name, truncate_dim=512, **kwargs)

        with patch("trelix.embedder.bge_code.FlagModel", _truncated):
            cfg = EmbedderConfig(provider="bge-code", _env_file=None)
            emb = BGECodeEmbedder(cfg)
            assert emb.dimension == 512
            assert len(emb.embed(["def foo(): pass"])[0]) == 512

    def test_dimension_falls_back_to_config_and_never_raises(self) -> None:
        """No readable HF config — return the configured width, do not raise."""
        from trelix.embedder.bge_code import BGECodeEmbedder

        class _Opaque(_FakeFlagModel):
            def __init__(self, name: str, **kwargs: object) -> None:
                super().__init__(name, **kwargs)
                self.model = None

        with patch("trelix.embedder.bge_code.FlagModel", _Opaque):
            cfg = EmbedderConfig(provider="bge-code", bge_code_dimensions=4096, _env_file=None)
            emb = BGECodeEmbedder(cfg)
            assert emb.dimension == 4096

    def test_embed_returns_correct_shape(self) -> None:
        from trelix.embedder.bge_code import BGECodeEmbedder

        mock_model = MagicMock()
        mock_model.encode.return_value = [[0.1] * 768, [0.2] * 768]
        with patch("trelix.embedder.bge_code.FlagModel", return_value=mock_model):
            cfg = EmbedderConfig(provider="bge-code", _env_file=None)
            emb = BGECodeEmbedder(cfg)
            result = emb.embed(["def foo(): pass", "class Bar: pass"])
            assert len(result) == 2
            assert len(result[0]) == 768

    def test_embed_query_uses_query_instruction(self) -> None:
        from trelix.embedder.bge_code import BGECodeEmbedder

        mock_model = MagicMock()
        mock_model.encode_queries.return_value = [[0.1] * 768]
        with patch("trelix.embedder.bge_code.FlagModel", return_value=mock_model):
            cfg = EmbedderConfig(provider="bge-code", _env_file=None)
            emb = BGECodeEmbedder(cfg)
            emb.embed_query("how does auth work")
            assert mock_model.encode_queries.called
            called_texts = mock_model.encode_queries.call_args[0][0]
            # no manual prefix — encode_queries adds it internally
            assert called_texts == ["how does auth work"]

    def test_make_embedder_returns_bge(self) -> None:
        from trelix.embedder.base import make_embedder

        mock_model = MagicMock()
        with patch("trelix.embedder.bge_code.FlagModel", return_value=mock_model):
            cfg = EmbedderConfig(provider="bge-code", _env_file=None)
            from trelix.embedder.bge_code import BGECodeEmbedder

            emb = make_embedder(cfg)
            assert isinstance(emb, BGECodeEmbedder)

    def test_config_effective_dimension(self) -> None:
        """The default, not an argument echoed back — the default is what was wrong."""
        cfg = EmbedderConfig(provider="bge-code", _env_file=None)
        assert cfg.bge_code_dimensions == 1536
        assert cfg.effective_dimension == 1536


# ---------------------------------------------------------------------------
# Pooling.
#
# 3.1.6 fixed `dimension` so `bge-code` could reach Phase 1 at all. Nothing in
# this file pinned WHICH FlagEmbedding class `bge_code.py` constructs, or how
# that class pools -- `_FakeFlagModel` above is substituted for the class and
# manufactures its return value from a constant (`_vecs`), so
# `DEFAULT_POOLING_METHOD`, `pooling()` and `last_hidden_state[:, 0]` are never
# reached, and `_vecs(2)` hands back two IDENTICAL vectors, i.e. the fake models
# the very degeneracy a pooling test must detect. The tests below use the real
# pooling functions out of the installed FlagEmbedding and a real causal forward
# pass, never a stand-in for either.
# ---------------------------------------------------------------------------

# Verbatim bytes of BAAI/bge-code-v1's published pooling config. Fetched once:
#   curl https://huggingface.co/BAAI/bge-code-v1/raw/main/1_Pooling/config.json
# Checked in, so nothing below touches the network at run time.
_PUBLISHED_1_POOLING_CONFIG = """{
  "word_embedding_dimension": 1536,
  "pooling_mode_cls_token": false,
  "pooling_mode_mean_tokens": false,
  "pooling_mode_max_tokens": false,
  "pooling_mode_mean_sqrt_len_tokens": false,
  "pooling_mode_weightedmean_tokens": false,
  "pooling_mode_lasttoken": true,
  "include_prompt": true
}"""

# Verbatim bytes of the same revision's top-level config.json. Fetched once:
#   curl https://huggingface.co/BAAI/bge-code-v1/raw/main/config.json
# Note also tokenizer_config.json's `"add_bos_token": false` at that revision:
# no BOS is injected, so token 0 of every query is the first token of the
# constant `_QUERY_INSTRUCTION` prefix.
_PUBLISHED_MODEL_CONFIG = """{
  "_name_or_path": "bge-code-v1",
  "architectures": [
    "Qwen2Model"
  ],
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "eos_token_id": 151643,
  "hidden_act": "silu",
  "hidden_size": 1536,
  "initializer_range": 0.02,
  "intermediate_size": 8960,
  "max_position_embeddings": 32768,
  "max_window_layers": 28,
  "model_type": "qwen2",
  "num_attention_heads": 12,
  "num_key_value_heads": 2,
  "rms_norm_eps": 1e-06,
  "rope_scaling": null,
  "rope_theta": 1000000.0,
  "sliding_window": null,
  "tie_word_embeddings": true,
  "torch_dtype": "float32",
  "transformers_version": "4.49.0",
  "use_cache": false,
  "use_sliding_window": false,
  "vocab_size": 151667
}"""

# sentence-transformers spells the mode as one boolean per method; FlagEmbedding
# spells it as a single string. This is the only translation between them.
_ST_FLAG_TO_FLAGEMBEDDING_METHOD = {
    "pooling_mode_cls_token": "cls",
    "pooling_mode_mean_tokens": "mean",
    "pooling_mode_lasttoken": "last_token",
}


def _published_pooling_method() -> str:
    """The one pooling mode BAAI published for bge-code-v1, in FlagEmbedding's vocabulary."""
    published = json.loads(_PUBLISHED_1_POOLING_CONFIG)
    enabled = [method for key, method in _ST_FLAG_TO_FLAGEMBEDDING_METHOD.items() if published[key]]
    assert len(enabled) == 1, (
        f"published 1_Pooling/config.json enables {enabled}; expected exactly one"
    )
    return enabled[0]


def _pool_the_way_flagembedding_does(model_class: object, hidden: object, mask: object) -> object:
    """Pool `hidden` with the REAL function `model_class` uses -- no imitation.

    `encoder_only.base.BaseEmbedder` implements cls and mean in its own
    `pooling()` (`base.py:284-308`, `cls` being `return last_hidden_state[:, 0]`).
    `decoder_only.base.BaseLLMEmbedder` has NO `pooling()` attribute at all; it
    calls the module-level `last_token_pool` inline (`base.py:261` and `:280`).
    Dispatching on the class's declared method mirrors FlagEmbedding's own
    structure, so one test body measures `FlagModel` today and `FlagLLMModel`
    after a fix -- without which this test could never stop being an xfail,
    because `FlagLLMModel.pooling` does not exist.
    """
    method = model_class.DEFAULT_POOLING_METHOD  # type: ignore[attr-defined]
    if method == "last_token":
        from FlagEmbedding.inference.embedder.decoder_only.base import last_token_pool

        return last_token_pool(hidden, mask)
    from FlagEmbedding.inference.embedder.encoder_only.base import BaseEmbedder as _EncBase

    return _EncBase.pooling(SimpleNamespace(pooling_method=method), hidden, attention_mask=mask)


class TestBGECodePooling:
    """The pooling method bge-code actually runs, versus the one BAAI published."""

    def test_published_snapshot_describes_the_model_this_provider_configures(self) -> None:
        """The snapshots are tied to live config, so they cannot rot silently.

        No FlagEmbedding and no torch, so this one always executes: it is the
        tripwire that says the block below ran rather than being collected away.
        """
        from trelix.embedder.bge_code import _QUERY_INSTRUCTION

        cfg = EmbedderConfig(provider="bge-code", _env_file=None)
        published = json.loads(_PUBLISHED_MODEL_CONFIG)

        assert cfg.bge_code_model == "BAAI/bge-code-v1", (
            "snapshots below are BAAI/bge-code-v1's; refresh them if the model changes"
        )
        assert published["hidden_size"] == cfg.bge_code_dimensions
        assert published["architectures"] == ["Qwen2Model"]
        assert published["model_type"] == "qwen2"
        assert _published_pooling_method() == "last_token"

        # Why token 0 is shared: the prefix is a constant, prepended to every
        # query by query_instruction_format "{}{}" (encoder_only/base.py:48).
        assert isinstance(_QUERY_INSTRUCTION, str) and _QUERY_INSTRUCTION != ""

    @pytest.mark.xfail(
        strict=True,
        # raises=AssertionError is load-bearing. Without it a strict xfail absorbs ANY
        # exception, so the boomerang silently breaks: the marker stays satisfied while
        # the assertion it exists for is never reached. Round 3 found exactly that in
        # test_cli_failure_exit_codes.py, where a JSONDecodeError stood in for the real
        # check. These two importorskip FlagEmbedding/torch/transformers and then build a
        # Qwen2Model, so any upstream TypeError would have satisfied the marker and gone
        # green. Verified under --runxfail that they die on the intended AssertionError.
        raises=AssertionError,
        reason=(
            "bge-code constructs FlagModel, which is encoder_only.base.BaseEmbedder "
            "with DEFAULT_POOLING_METHOD='cls'; BAAI published bge-code-v1 with "
            "pooling_mode_lasttoken. Fixing this means FlagLLMModel + "
            "trust_remote_code + real-weight validation, deliberately deferred past "
            "3.1.7. When it lands this XPASSes and the marker must be removed."
        ),
    )
    def test_constructed_class_pools_the_way_the_model_was_published(self) -> None:
        pytest.importorskip("FlagEmbedding", reason="bge-code extra not installed")
        from trelix.embedder import bge_code

        assert bge_code.FlagModel is not None
        assert bge_code.FlagModel.DEFAULT_POOLING_METHOD == _published_pooling_method()

    @pytest.mark.xfail(
        strict=True,
        # raises=AssertionError is load-bearing. Without it a strict xfail absorbs ANY
        # exception, so the boomerang silently breaks: the marker stays satisfied while
        # the assertion it exists for is never reached. Round 3 found exactly that in
        # test_cli_failure_exit_codes.py, where a JSONDecodeError stood in for the real
        # check. These two importorskip FlagEmbedding/torch/transformers and then build a
        # Qwen2Model, so any upstream TypeError would have satisfied the marker and gone
        # green. Verified under --runxfail that they die on the intended AssertionError.
        raises=AssertionError,
        reason=(
            "cls pooling on a causal decoder reads position 0 only, and every query "
            "shares token 0 via the constant instruction prefix, so every query "
            "embedding is byte-identical. XPASSes the moment the provider pools the "
            "published way; remove the marker then."
        ),
    )
    def test_two_queries_differing_after_token_0_get_different_embeddings(self) -> None:
        """A degeneracy detector: real pooling fn, real causal forward, no weights."""
        pytest.importorskip("FlagEmbedding", reason="bge-code extra not installed")
        torch = pytest.importorskip("torch")
        pytest.importorskip("transformers")
        from FlagEmbedding.inference.embedder.decoder_only.base import last_token_pool
        from transformers import Qwen2Config, Qwen2Model

        from trelix.embedder import bge_code

        published = json.loads(_PUBLISHED_MODEL_CONFIG)
        assert published["architectures"] == ["Qwen2Model"]

        # Randomly initialised and tiny: no weights are downloaded, and the
        # degeneracy is a property of the causal mask, not of the parameters.
        torch.manual_seed(0)
        model = Qwen2Model(
            Qwen2Config(
                vocab_size=64,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=published["num_key_value_heads"],
                hidden_act=published["hidden_act"],
                max_position_embeddings=64,
            )
        ).eval()

        # Two instruction-prefixed queries: identical prefix (ids 5,6,7),
        # different tails. Asserted rather than assumed -- picking inputs that
        # differ at token 0 would make this pass for the wrong reason.
        ids = torch.tensor([[5, 6, 7, 11, 12, 13], [5, 6, 7, 41, 42, 43]])
        mask = torch.ones_like(ids)
        assert ids[0][0].item() == ids[1][0].item(), "inputs must share token 0"
        assert ids[0].tolist() != ids[1].tolist(), "inputs must differ somewhere"

        with torch.no_grad():
            hidden = model(input_ids=ids, attention_mask=mask).last_hidden_state

        # POSITIVE CONTROL, evaluated first: the pooling BAAI published must
        # separate these two inputs. If it does not, the forward pass carried no
        # signal and a zero below would be an artefact, not the defect.
        control = last_token_pool(hidden, mask)
        control_delta = (control[0] - control[1]).abs().max().item()
        assert control_delta > 1e-3, (
            f"control dead: last_token pooling gave max|delta|={control_delta:.3e}; "
            "the model produced no signal, so this run proves nothing either way"
        )

        pooled = _pool_the_way_flagembedding_does(bge_code.FlagModel, hidden, mask)
        delta = (pooled[0] - pooled[1]).abs().max().item()
        assert delta > 1e-3, (
            f"every query embeds identically: {bge_code.FlagModel.__name__} pools with "
            f"{bge_code.FlagModel.DEFAULT_POOLING_METHOD!r} and gave "
            f"max|delta|={delta:.3e} on inputs sharing only token 0, while the "
            f"published last_token pooling separated the same hidden states by "
            f"{control_delta:.3e}"
        )
