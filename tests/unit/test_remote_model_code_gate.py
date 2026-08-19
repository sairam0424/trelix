"""SEC-03c — ``trust_remote_code=True`` is gated by an opt-in a ``.env`` cannot set.

The ``local-code`` and ``nomic-code`` providers load their model with
``trust_remote_code=True``, which runs the model repository's own Python inside
this process. That is required by the SFR / CodeRank architectures (see
``embedder/base.py``), so the gate is an **opt-in, not a removal** — both
providers still load, and ``reranker.py`` still pins ``trust_remote_code=False``.

The containment is the asymmetry between the two ways trelix learns a setting:

* a ``BaseSettings`` field can be supplied by a ``.env`` file, and until SEC-03a
  that file could be one committed in the repository being indexed;
* ``os.environ`` cannot — nothing in trelix calls ``load_dotenv()``, so a dotenv
  key never becomes a process environment variable.

So the opt-in is read from ``os.environ`` only. Half A below plants a ``.env``
that tries to grant it and asserts the load is refused *and that the model
constructor is never reached*; Half B sets it in the real process environment and
asserts the kwarg arrives. Half B is monkeypatched deliberately — the real models
need ~8 GB, so the decidable assertion is the kwarg, not a download.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from trelix.core.config import EmbedderConfig
from trelix.embedder.base import (
    REMOTE_MODEL_CODE_ENV_VAR,
    LocalCodeEmbedder,
    RemoteModelCodeNotAllowedError,
)
from trelix.embedder.nomic_code import NomicCodeEmbedder


@pytest.fixture
def hostile_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """cwd inside an untrusted repo whose .env tries to switch the gate on."""
    monkeypatch.delenv(REMOTE_MODEL_CODE_ENV_VAR, raising=False)
    (tmp_path / ".env").write_text(f"{REMOTE_MODEL_CODE_ENV_VAR}=1\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def recording_sentence_transformer() -> MagicMock:
    """A stand-in constructor that records its kwargs and never downloads a model."""
    factory = MagicMock()
    factory.return_value = MagicMock()
    factory.return_value.get_embedding_dimension.return_value = 4096
    factory.return_value.get_sentence_embedding_dimension.return_value = 768
    return factory


class TestRefusedWithoutTheProcessEnvOptIn:
    def test_local_code_embedder_refuses(
        self, hostile_repo: Path, recording_sentence_transformer: MagicMock
    ) -> None:
        config = EmbedderConfig(provider="local-code", _env_file=None)  # type: ignore[call-arg]
        module = MagicMock()
        module.SentenceTransformer = recording_sentence_transformer

        with patch.dict(sys.modules, {"sentence_transformers": module}):
            with pytest.raises(RemoteModelCodeNotAllowedError, match=REMOTE_MODEL_CODE_ENV_VAR):
                LocalCodeEmbedder(config)

        assert recording_sentence_transformer.call_count == 0

    def test_nomic_code_embedder_refuses(
        self, hostile_repo: Path, recording_sentence_transformer: MagicMock
    ) -> None:
        config = EmbedderConfig(provider="nomic-code", _env_file=None)  # type: ignore[call-arg]

        with patch(
            "trelix.embedder.nomic_code.SentenceTransformer", recording_sentence_transformer
        ):
            with pytest.raises(RemoteModelCodeNotAllowedError, match=REMOTE_MODEL_CODE_ENV_VAR):
                NomicCodeEmbedder(config)

        assert recording_sentence_transformer.call_count == 0


class TestLoadsWithTheProcessEnvOptIn:
    def test_local_code_embedder_passes_trust_remote_code(
        self,
        monkeypatch: pytest.MonkeyPatch,
        recording_sentence_transformer: MagicMock,
    ) -> None:
        monkeypatch.setenv(REMOTE_MODEL_CODE_ENV_VAR, "1")
        config = EmbedderConfig(provider="local-code", _env_file=None)  # type: ignore[call-arg]
        module = MagicMock()
        module.SentenceTransformer = recording_sentence_transformer

        with patch.dict(sys.modules, {"sentence_transformers": module}):
            embedder = LocalCodeEmbedder(config)

        assert embedder.dimension == 4096
        assert recording_sentence_transformer.call_args.kwargs["trust_remote_code"] is True
        assert recording_sentence_transformer.call_args.args[0] == config.local_code_model

    def test_nomic_code_embedder_passes_trust_remote_code(
        self,
        monkeypatch: pytest.MonkeyPatch,
        recording_sentence_transformer: MagicMock,
    ) -> None:
        monkeypatch.setenv(REMOTE_MODEL_CODE_ENV_VAR, "1")
        config = EmbedderConfig(provider="nomic-code", _env_file=None)  # type: ignore[call-arg]

        with patch(
            "trelix.embedder.nomic_code.SentenceTransformer", recording_sentence_transformer
        ):
            embedder = NomicCodeEmbedder(config)

        assert embedder.dimension == 768
        assert recording_sentence_transformer.call_args.kwargs["trust_remote_code"] is True
        assert recording_sentence_transformer.call_args.args[0] == config.nomic_code_model
