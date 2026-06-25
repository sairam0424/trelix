"""
Embedder abstraction — four providers, same interface.

  local  → LocalEmbedder         (sentence-transformers, no API key needed)
  openai → OpenAIEmbedder        (standard OpenAI API)
  azure  → AzureOpenAIEmbedder   (Azure OpenAI, uses AZURE_* env vars)
  aava   → AavaPlatformEmbedder  (Aava platform service, credentials injected by VS Code plugin)

The rest of the pipeline only ever calls embed() / embed_query().
Switching provider = change one line in config (TRELIX_EMBEDDER_PROVIDER).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from trelix.core.config import EmbedderConfig


class BaseEmbedder(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns list of embedding vectors."""
        ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding vector dimension."""
        ...


class AzureOpenAIEmbedder(BaseEmbedder):
    """
    Azure OpenAI embeddings via the openai SDK's AzureOpenAI client.

    Uses: text-embedding-3-large (3072 dims) — best quality available.
    Reads credentials from config which loads them from .env automatically.
    """

    def __init__(self, config: EmbedderConfig) -> None:
        from openai import AzureOpenAI
        self._client = AzureOpenAI(
            api_key=config.azure_api_key,
            azure_endpoint=config.azure_endpoint or "",
            api_version=config.azure_api_version,
        )
        self._deployment = config.azure_embeddings_deployment
        self._dimensions = config.azure_dimensions
        self._batch_size = config.batch_size

    def embed(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            response = self._client.embeddings.create(
                model=self._deployment,  # Azure uses deployment name, not model name
                input=batch,
                dimensions=self._dimensions,
            )
            results.extend([item.embedding for item in response.data])
        return results

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]

    @property
    def dimension(self) -> int:
        return self._dimensions


class OpenAIEmbedder(BaseEmbedder):
    """Standard OpenAI text-embedding-3-large."""

    def __init__(self, config: EmbedderConfig) -> None:
        from openai import OpenAI
        self._client = OpenAI(api_key=config.openai_api_key)
        self._model = config.openai_model
        self._dimensions = config.openai_dimensions
        self._batch_size = config.batch_size

    def embed(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            response = self._client.embeddings.create(
                model=self._model,
                input=batch,
                dimensions=self._dimensions,
            )
            results.extend([item.embedding for item in response.data])
        return results

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]

    @property
    def dimension(self) -> int:
        return self._dimensions


class LocalEmbedder(BaseEmbedder):
    """
    sentence-transformers local model — no API key, runs on CPU/GPU.

    Default model: all-MiniLM-L6-v2 (384 dimensions).
    Requires the optional 'local' extra:
        pip install 'trelix[local]'
    """

    def __init__(self, config: EmbedderConfig) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for the local embedder. "
                "Install it with: pip install 'trelix[local]'"
            ) from exc
        self._model = SentenceTransformer(config.local_model)
        self._batch_size = config.batch_size

    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings.tolist()  # type: ignore[return-value]

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]

    @property
    def dimension(self) -> int:
        # get_embedding_dimension is the new name; fall back to legacy for older versions
        getter = getattr(self._model, "get_embedding_dimension", None) or \
                 getattr(self._model, "get_sentence_embedding_dimension", None)
        return getter()  # type: ignore[return-value]


class AavaPlatformEmbedder(BaseEmbedder):
    """
    Aava platform embedding service — POST /embedding/knowledge/v2/chunks/embed.

    Credentials are injected by the VS Code plugin at indexing time:
      EMBEDDING_BEARER_TOKEN  — SSO access token (rotated automatically by the plugin)
      EMBEDDING_BASE_URL      — service base URL (e.g. https://aava-dev.avateam.io)

    No API keys are stored in config files or VS Code settings.
    Embedding dimension: 3072 (gemini-embedding-001).
    """

    _API_PATH = "/embedding/knowledge/v2/chunks/embed"
    _DIMENSION = 3072  # gemini-embedding-001

    def __init__(self, config: EmbedderConfig) -> None:
        self._bearer_token = config.embedding_bearer_token or ""
        self._base_url = config.embedding_base_url.rstrip("/")
        self._embedding_service = config.embedding_service
        self._model_ref = config.embedding_model_ref
        self._batch_size = config.batch_size

    def embed(self, texts: list[str]) -> list[list[float]]:
        import json
        import time
        import urllib.error
        import urllib.request

        results: list[list[float]] = []
        url = f"{self._base_url}{self._API_PATH}"

        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            payload = json.dumps({
                "embedding_service": self._embedding_service,
                "model_ref": self._model_ref,
                "chunks": [{"text": t, "metadata": {}} for t in batch],
            }).encode("utf-8")

            req = urllib.request.Request(
                url=url,
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self._bearer_token}",
                },
                method="POST",
            )

            last_exc: Exception | None = None
            data: dict = {}
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(req) as resp:
                        data = json.loads(resp.read().decode("utf-8"))
                    last_exc = None
                    break
                except urllib.error.HTTPError as exc:
                    body = exc.read().decode("utf-8", errors="replace")
                    last_exc = RuntimeError(
                        f"Embedding service returned {exc.code}: {body}"
                    )
                    if exc.code < 500:
                        raise last_exc from exc
                    time.sleep(2 ** attempt)

            if last_exc is not None:
                raise last_exc

            results.extend([e["vector"] for e in data["data"]["embeddings"]])

        return results

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]

    @property
    def dimension(self) -> int:
        return self._DIMENSION


def make_embedder(config: EmbedderConfig) -> BaseEmbedder:
    """Factory — instantiate the right embedder from config.provider.

    Args:
        config: EmbedderConfig with provider set to "local", "openai", "azure", or "aava".

    Returns:
        The appropriate BaseEmbedder subclass instance.

    Raises:
        ValueError: If config.provider is not a recognised value.
        ImportError: If provider is "local" and sentence-transformers is not installed.
    """
    match config.provider:
        case "aava":
            return AavaPlatformEmbedder(config)
        case "azure":
            return AzureOpenAIEmbedder(config)
        case "openai":
            return OpenAIEmbedder(config)
        case "local":
            return LocalEmbedder(config)
        case _:
            raise ValueError(
                f"Unknown embedder provider: {config.provider!r}. "
                "Expected one of: 'local', 'openai', 'azure', 'aava'."
            )
