"""
Embedder abstraction — seven providers, same interface.

  local          → LocalEmbedder           (sentence-transformers, no API key)
  openai         → OpenAIEmbedder          (text-embedding-3-large, 3072 dims)
  azure          → AzureOpenAIEmbedder     (Azure OpenAI, AZURE_* env vars)
  voyage         → VoyageEmbedder          (voyage-code-3, 1024 dims, 56.26 CoIR)
  local-code     → LocalCodeEmbedder       (SFR-Embedding-Code-2B_R, 4096 dims, 67.41 CoIR)
  bedrock-titan  → BedrockTitanEmbedder    (amazon.titan-embed-text-v2, 256/512/1024 dims)
  bedrock-cohere → BedrockCohereEmbedder   (cohere.embed-english-v3, 1024 dims)

The rest of the pipeline only ever calls embed() / embed_query().
Switching provider = set TRELIX_EMBEDDER_PROVIDER in .env — zero code changes.

AWS credentials (bedrock-titan / bedrock-cohere):
  Reuses AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY from .env (same as LLMConfig).
  Base64-encoded credentials are decoded transparently.

Titan dimension trade-off:
  1024 → matches Voyage CoIR quality, 4× storage vs 256
  512  → balanced quality/storage sweet spot for most repos
  256  → minimum — good for large repos where storage matters

Async support (U5):
  embed_async(texts) is available on all providers for concurrent batch API calls.
  OpenAI / Azure: true async via AsyncOpenAI / AsyncAzureOpenAI clients.
  Bedrock: uses run_in_executor (boto3 is sync-only; one thread per batch chunk).
  Local / VoyageEmbedder (sync library): run_in_executor (CPU-bound or sync SDK).
  BaseEmbedder provides a default fallback via run_in_executor for any subclass
  that does not override embed_async.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from trelix.core.config import EmbedderConfig

# Module-level thread pool for sync embedders that need to run in an executor.
# Modest pool: each task is either CPU-bound (local) or a blocking sync SDK call.
_SYNC_EXECUTOR: ThreadPoolExecutor | None = None


def _get_sync_executor() -> ThreadPoolExecutor:
    global _SYNC_EXECUTOR
    if _SYNC_EXECUTOR is None:
        _SYNC_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="trelix-embed-sync")
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

    def _get_async_client(self) -> Any:
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

    def _get_async_client(self) -> Any:
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
        return embeddings.tolist()  # type: ignore[no-any-return]

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]

    @property
    def dimension(self) -> int:
        # get_embedding_dimension is the new name; fall back to legacy for older versions
        getter = getattr(self._model, "get_embedding_dimension", None) or getattr(
            self._model, "get_sentence_embedding_dimension", None
        )
        return getter()  # type: ignore[no-any-return, misc]

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
        return response.embeddings[0]  # type: ignore[no-any-return]

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
        return embeddings.tolist()  # type: ignore[no-any-return]

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]

    @property
    def dimension(self) -> int:
        getter = getattr(self._model, "get_embedding_dimension", None) or getattr(
            self._model, "get_sentence_embedding_dimension", None
        )
        if getter is not None:
            return getter()  # type: ignore[no-any-return]
        # Fallback: SFR-Embedding-Code-2B_R native output dimension
        return 4096


class _BedrockEmbedderBase(BaseEmbedder):
    """
    Shared boto3 client setup and credential decode for both Bedrock embedders.

    Bedrock embedding uses invoke_model (not Converse) — completely different
    endpoint from the chat API. Credentials reuse AWS_* env vars already in .env.
    """

    @staticmethod
    def _decode_credential(value: str) -> str:
        """Transparently decode base64-encoded credentials stored in .env."""
        import base64

        try:
            decoded = base64.b64decode(value).decode("utf-8")
            if decoded.isprintable() and "\n" not in decoded:
                return decoded
        except Exception:  # noqa: BLE001
            pass
        return value

    def _make_boto3_client(self, config: EmbedderConfig) -> Any:
        try:
            import boto3
        except ImportError as exc:
            raise ImportError(
                "Bedrock embedders require boto3. Install it with: pip install 'trelix[bedrock]'"
            ) from exc
        session_kwargs: dict[str, Any] = {}
        if config.bedrock_aws_profile:
            session_kwargs["profile_name"] = config.bedrock_aws_profile
        session = boto3.Session(**session_kwargs)
        client_kwargs: dict[str, Any] = {"region_name": config.bedrock_aws_region}
        if config.bedrock_aws_access_key_id:
            client_kwargs["aws_access_key_id"] = self._decode_credential(
                config.bedrock_aws_access_key_id
            )
        if config.bedrock_aws_secret_access_key:
            client_kwargs["aws_secret_access_key"] = self._decode_credential(
                config.bedrock_aws_secret_access_key
            )
        return session.client("bedrock-runtime", **client_kwargs)


class BedrockTitanEmbedder(_BedrockEmbedderBase):
    """
    AWS Bedrock Titan Embed Text v2 embedder.

    Model: amazon.titan-embed-text-v2:0
    Dimensions: 256 | 512 | 1024 (configurable — default 1024)
    Normalize: True (unit vectors — better cosine similarity)

    Trade-offs vs other providers:
      - No extra API key needed beyond AWS creds already in .env
      - 1024 dims matches Voyage quality for general-purpose retrieval
      - 256 dims: 4× lower storage, good for very large repos
      - Batch limit: 1 document per invoke_model call (no batching in Titan)
        → each text in the batch is a separate boto3 call, parallelised
          in embed_async via asyncio.gather(run_in_executor) per text
    """

    # Titan API: one text per call — no native batching
    _BATCH_SIZE = 1

    def __init__(self, config: EmbedderConfig) -> None:
        self._client = self._make_boto3_client(config)
        self._model = config.bedrock_titan_model
        self._dims = config.bedrock_titan_dimensions
        self._normalize = config.bedrock_titan_normalize

    def _embed_one(self, text: str) -> list[float]:
        import json

        body = json.dumps(
            {
                "inputText": text,
                "dimensions": self._dims,
                "normalize": self._normalize,
            }
        )
        response = self._client.invoke_model(
            modelId=self._model,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        return json.loads(response["body"].read())["embedding"]  # type: ignore[no-any-return]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text)

    async def embed_async(self, texts: list[str]) -> list[list[float]]:
        """Parallelise per-text boto3 calls in a thread pool."""
        loop = asyncio.get_event_loop()
        tasks = [loop.run_in_executor(_get_sync_executor(), self._embed_one, t) for t in texts]
        return list(await asyncio.gather(*tasks))

    @property
    def dimension(self) -> int:
        return self._dims


class BedrockCohereEmbedder(_BedrockEmbedderBase):
    """
    AWS Bedrock Cohere Embed English v3 embedder.

    Model: cohere.embed-english-v3
    Dimensions: 1024 (fixed)
    Batch limit: 96 texts per invoke_model call (Cohere API limit)

    Distinguishes document vs query embeddings via input_type — same pattern
    as VoyageEmbedder.  document embeddings use "search_document",
    query embeddings use "search_query".

    Why Cohere over Titan for code retrieval:
      - Cohere embed-english-v3 is trained on diverse code/text datasets
      - Asymmetric retrieval (doc vs query input_type) improves precision
      - Fixed 1024 dims — predictable storage, no tuning needed
    """

    _BATCH_LIMIT = 96  # Cohere Bedrock API: max 96 texts per call
    _MAX_CHARS = 2048  # Bedrock validates length BEFORE truncation — must pre-truncate

    def __init__(self, config: EmbedderConfig) -> None:
        self._client = self._make_boto3_client(config)
        self._model = config.bedrock_cohere_model

    def _embed_batch(self, texts: list[str], input_type: str) -> list[list[float]]:
        import json

        # Pre-truncate: Bedrock rejects texts >2048 chars with ValidationException
        # even when truncate="END" is set — the validation fires before truncation.
        safe = [t[: self._MAX_CHARS] for t in texts]
        body = json.dumps(
            {
                "texts": safe,
                "input_type": input_type,
                "truncate": "END",
            }
        )
        response = self._client.invoke_model(
            modelId=self._model,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        return json.loads(response["body"].read())["embeddings"]  # type: ignore[no-any-return]

    def embed(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float]] = []
        for i in range(0, len(texts), self._BATCH_LIMIT):
            batch = texts[i : i + self._BATCH_LIMIT]
            results.extend(self._embed_batch(batch, "search_document"))
        return results

    def embed_query(self, text: str) -> list[float]:
        # Cohere distinguishes query from document — use search_query for queries
        return self._embed_batch([text], "search_query")[0]

    @property
    def dimension(self) -> int:
        return 1024


def make_embedder(config: EmbedderConfig) -> BaseEmbedder:
    """Factory — instantiate the right embedder from config.provider.

    Args:
        config: EmbedderConfig with provider set to one of:
            "local", "openai", "azure", "voyage", "local-code",
            "bedrock-titan", "bedrock-cohere".

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
        case "bedrock-titan":
            return BedrockTitanEmbedder(config)
        case "bedrock-cohere":
            return BedrockCohereEmbedder(config)
        case "bge-code":
            from trelix.embedder.bge_code import BGECodeEmbedder

            return BGECodeEmbedder(config)
        case _:
            raise ValueError(
                f"Unknown embedder provider: {config.provider!r}. "
                "Expected one of: 'local', 'openai', 'azure', 'voyage', "
                "'local-code', 'bedrock-titan', 'bedrock-cohere', 'bge-code', 'nomic-code'."
            )
