"""
Embedder abstraction — five providers, same interface.

  local       → LocalEmbedder         (sentence-transformers, no API key needed)
  openai      → OpenAIEmbedder        (standard OpenAI API)
  azure       → AzureOpenAIEmbedder   (Azure OpenAI, uses AZURE_* env vars)
  voyage      → VoyageEmbedder        (Voyage AI voyage-code-3, 1024 dims, 56.26 CoIR)
  local-code  → LocalCodeEmbedder     (SFR-Embedding-Code-2B_R, 4096 dims, 67.41 CoIR)

The rest of the pipeline only ever calls embed() / embed_query().
Switching provider = change one line in config (TRELIX_EMBEDDER_PROVIDER).

Async support (U5):
  embed_async(texts) is available on all providers for concurrent batch API calls.
  OpenAI / Azure: true async via AsyncOpenAI / AsyncAzureOpenAI clients.
  Local / VoyageEmbedder (sync library): run_in_executor (CPU-bound or sync SDK).
  BaseEmbedder provides a default fallback via run_in_executor for any subclass
  that does not override embed_async.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor

from trelix.core.config import EmbedderConfig

# Module-level thread pool for sync embedders that need to run in an executor.
# Modest pool: each task is either CPU-bound (local) or a blocking sync SDK call.
_SYNC_EXECUTOR: ThreadPoolExecutor | None = None


def _get_sync_executor() -> ThreadPoolExecutor:
    global _SYNC_EXECUTOR
    if _SYNC_EXECUTOR is None:
        _SYNC_EXECUTOR = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="trelix-embed-sync"
        )
    return _SYNC_EXECUTOR


class BaseEmbedder(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns list of embedding vectors."""
        ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        ...

    async def embed_async(self, texts: list[str]) -> list[list[float]]:
        """
        Async embed — default implementation runs self.embed() in a thread executor.

        Subclasses that support true async (OpenAI, Azure) override this to use
        the async SDK clients directly for lower overhead and true concurrency.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_get_sync_executor(), self.embed, texts)

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

    Async: uses AsyncAzureOpenAI client for true async without blocking the
    event loop (U5 concurrent batch embedding).
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
        self._async_client_config = config  # stored for lazy async client init

    def _get_async_client(self):  # type: ignore[return]
        """Lazily create AsyncAzureOpenAI client (avoids import at module level)."""
        from openai import AsyncAzureOpenAI
        return AsyncAzureOpenAI(
            api_key=self._async_client_config.azure_api_key,
            azure_endpoint=self._async_client_config.azure_endpoint or "",
            api_version=self._async_client_config.azure_api_version,
        )

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

    async def embed_async(self, texts: list[str]) -> list[list[float]]:
        """True async via AsyncAzureOpenAI — does not block the event loop."""
        async_client = self._get_async_client()
        results: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            response = await async_client.embeddings.create(
                model=self._deployment,
                input=batch,
                dimensions=self._dimensions,
            )
            results.extend([item.embedding for item in response.data])
        await async_client.close()
        return results

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]

    @property
    def dimension(self) -> int:
        return self._dimensions


class OpenAIEmbedder(BaseEmbedder):
    """
    Standard OpenAI text-embedding-3-large.

    Async: uses AsyncOpenAI client for true async without blocking the
    event loop (U5 concurrent batch embedding).
    """

    def __init__(self, config: EmbedderConfig) -> None:
        from openai import OpenAI
        self._client = OpenAI(api_key=config.openai_api_key)
        self._model = config.openai_model
        self._dimensions = config.openai_dimensions
        self._batch_size = config.batch_size
        self._async_client_config = config  # stored for lazy async client init

    def _get_async_client(self):  # type: ignore[return]
        """Lazily create AsyncOpenAI client."""
        from openai import AsyncOpenAI
        return AsyncOpenAI(api_key=self._async_client_config.openai_api_key)

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

    async def embed_async(self, texts: list[str]) -> list[list[float]]:
        """True async via AsyncOpenAI — does not block the event loop."""
        async_client = self._get_async_client()
        results: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            response = await async_client.embeddings.create(
                model=self._model,
                input=batch,
                dimensions=self._dimensions,
            )
            results.extend([item.embedding for item in response.data])
        await async_client.close()
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

    Async: CPU-bound — uses run_in_executor (BaseEmbedder default).
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
        getter = (
            getattr(self._model, "get_embedding_dimension", None)
            or getattr(self._model, "get_sentence_embedding_dimension", None)
        )
        return getter()  # type: ignore[return-value]

    # embed_async: inherited BaseEmbedder default (run_in_executor) — CPU-bound,
    # running in a thread keeps the event loop free.


class VoyageEmbedder(BaseEmbedder):
    """
    Voyage AI code-optimised embedder — voyage-code-3 (1024 dims, 56.26 CoIR).

    Distinguishes document vs query embeddings (input_type parameter).
    Batches requests in groups of 128 (Voyage API per-call limit).

    Requires the optional 'voyage' extra:
        pip install 'trelix[voyage]'
    """

    _BATCH_LIMIT = 128

    def __init__(self, config: EmbedderConfig) -> None:
        try:
            import voyageai
        except ImportError as exc:
            raise ImportError(
                "voyageai is required for the voyage embedder. "
                "Install it with: pip install 'trelix[voyage]'"
            ) from exc
        self._client = voyageai.Client(api_key=config.voyage_api_key)
        self._model = config.voyage_model
        self._dimensions = config.voyage_dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float]] = []
        for i in range(0, len(texts), self._BATCH_LIMIT):
            batch = texts[i : i + self._BATCH_LIMIT]
            response = self._client.embed(batch, model=self._model, input_type="document")
            results.extend(response.embeddings)
        return results

    def embed_query(self, text: str) -> list[float]:
        response = self._client.embed([text], model=self._model, input_type="query")
        return response.embeddings[0]

    @property
    def dimension(self) -> int:
        return self._dimensions


class LocalCodeEmbedder(BaseEmbedder):
    """
    SFR-Embedding-Code-2B_R — best open-source code embedder.

    Performance: 67.41 avg on CoIR (vs Ada-002's 45.59 = 49% gap).
    Dimensions: 4096 by default (model.get_embedding_dimension()).

    NOTE: This model requires approximately 8 GB RAM / GPU memory (2B parameters).
    trust_remote_code=True is required for the SFR model architecture.

    Requires the optional 'local' extra:
        pip install 'trelix[local]'
    """

    def __init__(self, config: EmbedderConfig) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for the local-code embedder. "
                "Install it with: pip install 'trelix[local]'"
            ) from exc
        self._model = SentenceTransformer(config.local_code_model, trust_remote_code=True)
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
        getter = getattr(self._model, "get_embedding_dimension", None) or \
                 getattr(self._model, "get_sentence_embedding_dimension", None)
        if getter is not None:
            return getter()  # type: ignore[return-value]
        # Fallback: SFR-Embedding-Code-2B_R native output dimension
        return 4096


def make_embedder(config: EmbedderConfig) -> BaseEmbedder:
    """Factory — instantiate the right embedder from config.provider.

    Args:
        config: EmbedderConfig with provider set to one of:
            "local", "openai", "azure", "voyage", "local-code".

    Returns:
        The appropriate BaseEmbedder subclass instance.

    Raises:
        ValueError: If config.provider is not a recognised value.
        ImportError: If the required optional dependency is not installed.
    """
    match config.provider:
        case "azure":
            return AzureOpenAIEmbedder(config)
        case "openai":
            return OpenAIEmbedder(config)
        case "local":
            return LocalEmbedder(config)
        case "voyage":
            return VoyageEmbedder(config)
        case "local-code":
            return LocalCodeEmbedder(config)
        case _:
            raise ValueError(
                f"Unknown embedder provider: {config.provider!r}. "
                "Expected one of: 'local', 'openai', 'azure', 'voyage', 'local-code'."
            )
