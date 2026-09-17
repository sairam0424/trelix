"""
Embedder abstraction — ten providers, same interface.

  local          → LocalEmbedder           (sentence-transformers, no API key)
  openai         → OpenAIEmbedder          (text-embedding-3-large, 3072 dims)
  azure          → AzureOpenAIEmbedder     (Azure OpenAI, AZURE_* env vars)
  voyage         → VoyageEmbedder          (voyage-code-3, 1024 dims, 56.26 CoIR)
  local-code     → LocalCodeEmbedder       (SFR-Embedding-Code-2B_R, 2304 dims) EXPERIMENTAL
  bedrock-titan  → BedrockTitanEmbedder    (amazon.titan-embed-text-v2, 256/512/1024 dims)
  bedrock-cohere → BedrockCohereEmbedder   (cohere.embed-english-v3, 1024 dims, via Bedrock)
  bge-code       → BGECodeEmbedder         (bge_code.py; bge-code-v1, 1536 dims) EXPERIMENTAL
  nomic-code     → NomicCodeEmbedder       (nomic_code.py; CodeRankEmbed, 768 dims) EXPERIMENTAL
  cohere         → CohereEmbedder          (cohere.py; embed-english-v3.0, 1024 dims, direct API)

This docstring said "seven" and listed seven, while make_embedder() below had
dispatched nine since v2 — bge-code and nomic-code were never added here.

EXPERIMENTAL means: the wrapper's encoding protocol has been checked against the
model repository and does NOT match it. bge-code pools with `cls` on a causal
decoder (see bge_code.py). local-code omits the query instruction its model card
requires. nomic-code sends a different Nomic model's task prefixes. None of the
three has been validated against real weights; all three are retained, documented,
and deliberately unfixed rather than silently advertised as working.

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

Cost metrics (opt-in, TRELIX_OTEL_ENABLED=true):
  Every successful provider call is counted via _count_embed_call() —
  requests/texts/characters always, provider-reported tokens where the response
  carries them. Embedding is the only per-call billed operation in trelix, so
  this is the one place spend is countable rather than merely traceable.
"""

from __future__ import annotations

import asyncio
import os
import sys
from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from trelix.core.config import EmbedderConfig
from trelix.core.retry import with_retry
from trelix.retrieval import otel_tracing

# Module-level thread pool for sync embedders that need to run in an executor.
# Modest pool: each task is either CPU-bound (local) or a blocking sync SDK call.
_SYNC_EXECUTOR: ThreadPoolExecutor | None = None


def _get_sync_executor() -> ThreadPoolExecutor:
    global _SYNC_EXECUTOR
    if _SYNC_EXECUTOR is None:
        _SYNC_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="trelix-embed-sync")
    return _SYNC_EXECUTOR


def _count_embed_call(
    provider: str, model: str, texts: list[str], tokens: int | None = None
) -> None:
    """Record one successful provider call on the OTel embedding counters.

    Called after the response lands, so retried-then-failed attempts are not
    counted as requests. *tokens* is whatever the provider reported, or None —
    local models and Cohere-on-Bedrock report nothing, and a chars/4 guess in
    a cost counter is worse than a visibly missing series.

    The metrics_enabled() guard is what keeps the character sum off the hot
    path when the flag is off (a memoized tuple read; see otel_tracing).
    """
    if not otel_tracing.metrics_enabled():
        return
    otel_tracing.record_embedding_call(
        provider=provider,
        model=model,
        texts=len(texts),
        characters=sum(len(t) for t in texts),
        tokens=tokens,
    )


def _usage_tokens(response: Any) -> int | None:
    """Provider-reported total token count, or None when it reports none.

    OpenAI/Azure put it on `response.usage.total_tokens`; Voyage puts it flat
    on `response.total_tokens`. Anything else yields None rather than an
    estimate. Bedrock returns JSON, not an object, and is read at its call site.
    """
    usage = getattr(response, "usage", None)
    total = (
        getattr(usage, "total_tokens", None)
        if usage is not None
        else getattr(response, "total_tokens", None)
    )
    return int(total) if isinstance(total, int | float) and not isinstance(total, bool) else None


# ---------------------------------------------------------------------------
# Remote model code gate
# ---------------------------------------------------------------------------

# Opt-in that unlocks trusted remote model code. Read from os.environ ONLY —
# deliberately not a pydantic-settings field on EmbedderConfig, and never read
# through any class carrying `env_file`. A dotenv key never becomes a process
# environment variable (nothing in trelix calls load_dotenv), so this lookup is
# the one input a `.env` inside an indexed repository cannot reach. That
# asymmetry is the whole containment: without it, a committed `.env` selecting
# `local-code` and naming its own model turned `trelix index` into arbitrary
# code execution, because a trusted-remote-code load runs the model repository's
# Python in this process.
REMOTE_MODEL_CODE_ENV_VAR = "TRELIX_ALLOW_REMOTE_MODEL_CODE"
_REMOTE_MODEL_CODE_TRUTHY = frozenset({"1", "true", "yes", "on"})


class RemoteModelCodeNotAllowedError(ValueError):
    """A provider that executes model-repository Python was selected without the opt-in.

    A ValueError, not a RuntimeError: this is a misconfiguration, and every command
    in cli/main.py already pairs its handler with `except (ValueError, ...)`. A
    RuntimeError would reach the user as a traceback instead of the message above.
    """


def remote_model_code_allowed() -> bool:
    """True when the operator has opted in via the real process environment."""
    return (
        os.environ.get(REMOTE_MODEL_CODE_ENV_VAR, "").strip().lower() in _REMOTE_MODEL_CODE_TRUTHY
    )


def load_remote_code_model(model_name: str, *, provider: str, factory: Callable[..., Any]) -> Any:
    """Load *model_name* trusting its repository's code, or refuse.

    The package's only trusted-remote-code load — both providers that need one
    route through here, so the gate cannot be present at one site and missing at
    the other, which is how this class of fix usually fails.
    `retrieval/reranker.py` keeps its own `trust_remote_code=False` instead of
    calling this: the cross-encoder needs no remote code, and routing it through
    here would hand it a capability it never asked for.

    Raises RemoteModelCodeNotAllowedError *before* touching *factory*, so a
    refusal never reaches the model hub.
    """
    if not remote_model_code_allowed():
        raise RemoteModelCodeNotAllowedError(
            f"embedder provider {provider!r} loads {model_name!r} with remote code "
            "trusted, which executes that model repository's own Python inside this "
            "process. Refusing, because the model name can come from configuration "
            f"trelix does not own. To allow it, set {REMOTE_MODEL_CODE_ENV_VAR}=1 in "
            "the process environment — a .env file cannot enable it, by design."
        )
    return factory(model_name, trust_remote_code=True)


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

        # max_retries=0: @with_retry below is the sole retry layer — the
        # SDK's own default (2 attempts) would otherwise stack underneath
        # tenacity's 5-attempt loop, multiplying worst-case wall-clock time
        # on a persistent outage far beyond what max_attempts=5 implies.
        self._client = AzureOpenAI(
            api_key=config.azure_api_key,
            azure_endpoint=config.azure_endpoint or "",
            api_version=config.azure_api_version,
            max_retries=0,
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
            max_retries=0,
        )

    @with_retry(max_attempts=5)
    def _create(self, **kwargs: Any) -> Any:
        return self._client.embeddings.create(**kwargs)

    @with_retry(max_attempts=5)
    async def _create_async(self, async_client: Any, **kwargs: Any) -> Any:
        return await async_client.embeddings.create(**kwargs)

    def embed(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            response = self._create(
                model=self._deployment,  # Azure uses deployment name, not model name
                input=batch,
                dimensions=self._dimensions,
            )
            _count_embed_call("azure", self._deployment, batch, _usage_tokens(response))
            results.extend([item.embedding for item in response.data])
        return results

    async def embed_async(self, texts: list[str]) -> list[list[float]]:
        """True async via AsyncAzureOpenAI — does not block the event loop."""
        async_client = self._get_async_client()
        results: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            response = await self._create_async(
                async_client,
                model=self._deployment,
                input=batch,
                dimensions=self._dimensions,
            )
            _count_embed_call("azure", self._deployment, batch, _usage_tokens(response))
            results.extend([item.embedding for item in response.data])
        await async_client.close()
        return results

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]

    @property
    def dimension(self) -> int:
        return self._dimensions


class BatchJobTerminalError(RuntimeError):
    """OpenAI itself reported the batch job as dead (failed/expired/cancelled).

    Deliberately its own type, not a bare RuntimeError: callers must be able
    to distinguish "OpenAI says this job will never complete" from any other
    exception poll_batch() can raise (a network error surviving retries, a
    malformed/partial response) -- catching Exception broadly to mark a job
    'failed' would also abandon jobs that only failed to be *fetched*, not
    jobs that actually failed on OpenAI's side, and OpenAI still bills for
    the abandoned one.
    """


class BatchJobIncompleteError(RuntimeError):
    """poll_batch() got fewer result records than requests were submitted.

    OpenAI can report overall batch status "completed" while individual
    requests inside it failed (content filter, per-request error, etc.) --
    failed rows have no usable embedding and are not silently droppable,
    because a positional zip(chunk_ids, vectors) with a short vectors list
    re-pairs every later chunk_id with the wrong vector instead of just the
    missing one.
    """


class OpenAIEmbedder(BaseEmbedder):
    """
    Standard OpenAI text-embedding-3-large.

    Async: uses AsyncOpenAI client for true async without blocking the
    event loop (U5 concurrent batch embedding).
    """

    def __init__(self, config: EmbedderConfig) -> None:
        from openai import OpenAI

        # max_retries=0: @with_retry below is the sole retry layer — see
        # AzureOpenAIEmbedder.__init__'s comment for why.
        self._client = OpenAI(api_key=config.openai_api_key, max_retries=0)
        self._model = config.openai_model
        self._dimensions = config.openai_dimensions
        self._batch_size = config.batch_size
        self._async_client_config = config  # stored for lazy async client init

    def _get_async_client(self) -> Any:
        """Lazily create AsyncOpenAI client."""
        from openai import AsyncOpenAI

        return AsyncOpenAI(api_key=self._async_client_config.openai_api_key, max_retries=0)

    @with_retry(max_attempts=5)
    def _create(self, **kwargs: Any) -> Any:
        return self._client.embeddings.create(**kwargs)

    @with_retry(max_attempts=5)
    async def _create_async(self, async_client: Any, **kwargs: Any) -> Any:
        return await async_client.embeddings.create(**kwargs)

    def embed(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            response = self._create(
                model=self._model,
                input=batch,
                dimensions=self._dimensions,
            )
            _count_embed_call("openai", self._model, batch, _usage_tokens(response))
            results.extend([item.embedding for item in response.data])
        return results

    async def embed_async(self, texts: list[str]) -> list[list[float]]:
        """True async via AsyncOpenAI — does not block the event loop."""
        async_client = self._get_async_client()
        results: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            response = await self._create_async(
                async_client,
                model=self._model,
                input=batch,
                dimensions=self._dimensions,
            )
            _count_embed_call("openai", self._model, batch, _usage_tokens(response))
            results.extend([item.embedding for item in response.data])
        await async_client.close()
        return results

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]

    # -- Batch API (submit_batch / poll_batch) -----------------------------
    #
    # OpenAI's Batch API: a 50%-cheaper, async, 24h-completion-window job —
    # architecturally distinct from the embed()/_create() request batching
    # above (which just chunks synchronous calls to self._batch_size). This
    # is trelix's first long-running external job pattern: submit a job now,
    # poll it later, pull results from an output file once it completes.
    # Deliberately OpenAIEmbedder-only, not on BaseEmbedder: most providers
    # (including Voyage — voyageai==0.5.0's Client has no batch-job method at
    # all) have no equivalent, and forcing every embedder to implement an
    # abstract method it cannot support is the wrong shape for that gap.
    #
    # Batch input-file JSONL shape (one line per text, each a full
    # /v1/embeddings request keyed by index) follows OpenAI's documented,
    # stable batch request format; it was not independently re-verified
    # against live docs in this session (no network access) — flagged here
    # per the task's own instruction to do so if unverified.

    _BATCH_TERMINAL_FAILURE_STATUSES = frozenset({"failed", "expired", "cancelled"})

    @with_retry(max_attempts=5)
    def _files_create(self, **kwargs: Any) -> Any:
        return self._client.files.create(**kwargs)

    @with_retry(max_attempts=5)
    def _batches_create(self, **kwargs: Any) -> Any:
        return self._client.batches.create(**kwargs)

    @with_retry(max_attempts=5)
    def _batches_retrieve(self, batch_id: str) -> Any:
        return self._client.batches.retrieve(batch_id)

    @with_retry(max_attempts=5)
    def _files_content(self, file_id: str) -> Any:
        return self._client.files.content(file_id)

    def submit_batch(self, texts: list[str]) -> str:
        """Submit *texts* as an OpenAI Batch API embeddings job.

        Uploads a JSONL input file (one /v1/embeddings request per line,
        custom_id = str(index) into *texts*) and creates a batch job against
        it with a 24h completion window. Returns immediately with the batch
        job id — this does not wait for completion; pair with poll_batch().
        """
        import io
        import json

        lines = [
            json.dumps(
                {
                    "custom_id": str(index),
                    "method": "POST",
                    "url": "/v1/embeddings",
                    "body": {
                        "model": self._model,
                        "input": text,
                        "dimensions": self._dimensions,
                    },
                }
            )
            for index, text in enumerate(texts)
        ]
        jsonl_file = io.BytesIO(("\n".join(lines) + "\n").encode("utf-8"))
        uploaded = self._files_create(file=jsonl_file, purpose="batch")
        batch = self._batches_create(
            completion_window="24h",
            endpoint="/v1/embeddings",
            input_file_id=uploaded.id,
        )
        return batch.id  # type: ignore[no-any-return]

    def poll_batch(
        self, job_id: str, expected_count: int | None = None
    ) -> list[list[float]] | None:
        """Poll an OpenAI Batch API job by id.

        Returns None while the job is still in flight (any non-terminal
        status: "validating", "in_progress", "finalizing", ...). Once the
        job's status is "completed", downloads its output file and returns
        the embedding vectors re-ordered to match the original *texts*
        passed to submit_batch() — the output file's line order is not
        guaranteed to match input order, only custom_id is.

        Raises BatchJobTerminalError if the job reaches a terminal failure
        status (failed/expired/cancelled): there are no vectors to retrieve,
        and the caller needs to know to resubmit rather than poll forever.
        This is a dedicated exception, not a bare RuntimeError, precisely so
        callers can catch only this and let every other exception (network,
        parsing) propagate without marking the job dead.

        Raises BatchJobIncompleteError if the batch reports "completed" but
        one or more individual requests inside it failed (OpenAI allows
        this: overall status can be "completed" with request_counts.failed
        > 0, or even output_file_id=None if every request failed). Passing
        `expected_count` (the number of texts originally submitted) lets
        this be detected even when *no* per-line data is missing but the
        count is still short — e.g. a line whose "response" is null rather
        than absent entirely.
        """
        import json

        batch = self._batches_retrieve(job_id)
        if batch.status in self._BATCH_TERMINAL_FAILURE_STATUSES:
            raise BatchJobTerminalError(
                f"OpenAI batch job {job_id!r} ended with status {batch.status!r}"
            )
        if batch.status != "completed":
            return None

        if batch.output_file_id is None:
            raise BatchJobIncompleteError(
                f"OpenAI batch job {job_id!r} is 'completed' but has no output_file_id "
                "-- every request in the batch failed. See the job's error_file_id "
                "on OpenAI's side for details."
            )

        output = self._files_content(batch.output_file_id)
        by_custom_id: dict[int, list[float]] = {}
        failed_ids: list[str] = []
        for line in output.text.splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            response = record.get("response")
            if response is None:
                failed_ids.append(record.get("custom_id", "<unknown>"))
                continue
            by_custom_id[int(record["custom_id"])] = response["body"]["data"][0]["embedding"]

        total_expected = expected_count if expected_count is not None else len(by_custom_id)
        if failed_ids or len(by_custom_id) < total_expected:
            raise BatchJobIncompleteError(
                f"OpenAI batch job {job_id!r} is 'completed' but only "
                f"{len(by_custom_id)}/{total_expected} requests succeeded "
                f"(failed custom_ids: {failed_ids or 'unknown -- count mismatch only'})."
            )

        return [by_custom_id[i] for i in sorted(by_custom_id)]

    @property
    def dimension(self) -> int:
        return self._dimensions


class LocalEmbedder(BaseEmbedder):
    """
    sentence-transformers local model — no API key, always runs on CPU.

    Default model: all-MiniLM-L6-v2 (384 dimensions).
    Requires the optional 'local' extra:
        pip install 'trelix[local]'

    Device is pinned to CPU rather than left to sentence-transformers'
    auto-detection (which picks MPS on Apple Silicon): PyTorch's Metal
    backend initializing inside a child process spawned deep inside a
    sandboxed host (e.g. an Electron-based editor's extension host
    spawning trelix-mcp over stdio) is a known native-crash class on
    macOS, and it fails silently from this process's perspective — the
    parent only sees the stdio pipe close. A 384-dim model's per-call
    cost is small enough that CPU inference costs nothing that matters
    for interactive use; large bulk re-index jobs are the one case where
    GPU throughput would help, and those already batch requests, so the
    per-call CPU cost amortizes fine there too.

    Async: CPU-bound — uses run_in_executor (BaseEmbedder default).
    """

    def __init__(self, config: EmbedderConfig) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            if getattr(sys, "frozen", False):
                # PyInstaller's standalone binary (trelix.spec) deliberately
                # excludes sentence-transformers/torch to stay ~30-40 MB instead
                # of ~500 MB. The frozen process never consults the host's
                # Python/pip environment, so telling the user to `pip install`
                # here would be actively misleading — running it has zero
                # effect on this binary.
                raise ImportError(
                    "sentence-transformers is required for the local embedder, "
                    "and the standalone trelix binary does not bundle it — "
                    "installing it on this host has no effect on this binary. "
                    "Use an API-backed provider instead (e.g. --provider "
                    "openai/azure/voyage/bedrock-titan), or use the trelix "
                    "Python package (trelix[local]) instead of the standalone "
                    "binary for local/offline embedding."
                ) from exc
            raise ImportError(
                "sentence-transformers is required for the local embedder. "
                "Install it with: pip install 'trelix[local]'"
            ) from exc
        self._model = SentenceTransformer(config.local_model, device="cpu")
        self._model_name = config.local_model  # self._model is the loaded model, not its id
        self._batch_size = config.batch_size

    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        # No tokens: sentence-transformers reports no usage and nothing is
        # billed — the characters counter is the only volume signal here.
        _count_embed_call("local", self._model_name, texts)
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
        self._output_dimensions = config.voyage_output_dimensions

    @with_retry(max_attempts=5)
    def _embed(self, texts: list[str], **kwargs: object) -> Any:
        response = self._client.embed(texts, **kwargs)
        # Single call site for both embed() and embed_query(), so counting here
        # covers document and query embeddings alike.
        _count_embed_call("voyage", self._model, texts, _usage_tokens(response))
        return response

    def embed(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float]] = []
        for i in range(0, len(texts), self._BATCH_LIMIT):
            batch = texts[i : i + self._BATCH_LIMIT]
            kwargs: dict[str, object] = {"model": self._model, "input_type": "document"}
            if self._output_dimensions is not None:
                kwargs["output_dimension"] = self._output_dimensions
            response = self._embed(batch, **kwargs)
            results.extend(response.embeddings)
        return results

    def embed_query(self, text: str) -> list[float]:
        kwargs: dict[str, object] = {"model": self._model, "input_type": "query"}
        if self._output_dimensions is not None:
            kwargs["output_dimension"] = self._output_dimensions
        response = self._embed([text], **kwargs)
        return response.embeddings[0]  # type: ignore[no-any-return]

    @property
    def dimension(self) -> int:
        return self._output_dimensions or self._dimensions


class LocalCodeEmbedder(BaseEmbedder):
    """
    SFR-Embedding-Code-2B_R — best open-source code embedder.

    Performance: 67.41 avg on CoIR (vs Ada-002's 45.59 = 49% gap) — the model card's
    self-report, never reproduced here.
    Dimensions: 2304 (model.get_embedding_dimension()). config.json `hidden_size` and
    1_Pooling/config.json `word_embedding_dimension` both say 2304, and modules.json
    has no Dense module to widen it. The 4096 this line claimed is the model's
    sentence_bert_config.json `max_seq_length`.

    Pooling is CORRECT here, unlike bge-code, and for a reason worth writing down:
    this is also a causal decoder (Gemma-2-2B fine-tune), but modules.json routes it
    through sentence_transformers Pooling, whose load() reads 1_Pooling/config.json,
    and this model publishes `pooling_mode_lasttoken: true`. sentence-transformers
    honours that file; FlagEmbedding never reads it, which is the whole difference
    between this provider and bge-code.

    KNOWN GAP (deferred — needs real-weight validation): the model card requires a
    query instruction, `encode(queries, prompt="Instruct: Given Code or Text,
    retrieval relevant content" + newline + "Query: ")`, with passages unprefixed.
    embed_query() below sends the document path instead, so queries are encoded as
    passages. config_sentence_transformers.json ships `prompts: {}`, so
    sentence-transformers cannot supply the instruction on our behalf either.

    LICENCE: these weights are CC-BY-NC-4.0, research use only (the model card says
    so explicitly). Nothing in the docs says this, though sparse.py already warns on
    load for the same reason about naver/splade.

    NOTE: This model requires approximately 8 GB RAM / GPU memory (2B parameters).
    trust_remote_code=True is required for the SFR model architecture — it is not
    optional here, so it is GATED rather than removed: the load goes through
    load_remote_code_model(), which refuses unless the operator set
    TRELIX_ALLOW_REMOTE_MODEL_CODE in the process environment.

    Requires the optional 'local' extra:
        pip install 'trelix[local]'
    """

    def __init__(self, config: EmbedderConfig) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            if getattr(sys, "frozen", False):
                # See LocalEmbedder.__init__'s comment — same PyInstaller
                # exclusion, same reason `pip install` is the wrong message here.
                raise ImportError(
                    "sentence-transformers is required for the local-code "
                    "embedder, and the standalone trelix binary does not "
                    "bundle it — installing it on this host has no effect on "
                    "this binary. Use an API-backed provider instead (e.g. "
                    "--provider openai/azure/voyage/bedrock-titan), or use the "
                    "trelix Python package (trelix[local]) instead of the "
                    "standalone binary for local/offline embedding."
                ) from exc
            raise ImportError(
                "sentence-transformers is required for the local-code embedder. "
                "Install it with: pip install 'trelix[local]'"
            ) from exc
        self._model = load_remote_code_model(
            config.local_code_model, provider="local-code", factory=SentenceTransformer
        )
        self._model_name = config.local_code_model
        self._batch_size = config.batch_size

    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        _count_embed_call("local-code", self._model_name, texts)
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
        # Fallback: SFR-Embedding-Code-2B_R native output dimension — 2304, from
        # config.json `hidden_size` and 1_Pooling/config.json
        # `word_embedding_dimension`, with no Dense module in modules.json to widen
        # it. The previous 4096 was that model's max_seq_length.
        return 2304


def _split_s3_uri(uri: str) -> tuple[str, str]:
    """Split "s3://bucket/key/prefix" into ("bucket", "key/prefix")."""
    without_scheme = uri.removeprefix("s3://")
    bucket, _, key_prefix = without_scheme.partition("/")
    return bucket, key_prefix


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

    def _make_boto3_client(
        self, config: EmbedderConfig, service_name: str = "bedrock-runtime"
    ) -> Any:
        """Build a boto3 client for *service_name* using the same session,
        region, profile, credentials, and zero-native-retries config every
        Bedrock embedder client already uses.

        *service_name* defaults to "bedrock-runtime" (both Titan's and
        Cohere's real-time invoke_model client, the only client either needed
        before batch support existed). BedrockTitanEmbedder's batch methods
        reuse this same construction for two more AWS services — "bedrock"
        (the control-plane client: create_model_invocation_job /
        get_model_invocation_job) and "s3" (uploading the input JSONL,
        discovering and downloading the output JSONL) — rather than
        duplicating the region/profile/credential-decode logic a second and
        third time.
        """
        try:
            import boto3
            from botocore.config import Config as BotoConfig
        except ImportError as exc:
            raise ImportError(
                "Bedrock embedders require boto3. Install it with: pip install 'trelix[bedrock]'"
            ) from exc
        if not config.bedrock_aws_region:
            raise ValueError(
                "AWS_REGION must be set explicitly for Bedrock embedders as of "
                "trelix v3.3.0 (previously silently defaulted to 'us-east-1', "
                "matching anthropic-sdk-python v1.0.0's AnthropicBedrock region "
                "enforcement). Set the AWS_REGION environment variable."
            )
        session_kwargs: dict[str, Any] = {}
        if config.bedrock_aws_profile:
            session_kwargs["profile_name"] = config.bedrock_aws_profile
        session = boto3.Session(**session_kwargs)
        client_kwargs: dict[str, Any] = {
            "region_name": config.bedrock_aws_region,
            # max_attempts=0: @with_retry (via _invoke_model() on both
            # Titan/Cohere embedders, and via the batch-API methods on
            # Titan) is meant to be the sole retry layer — see
            # BedrockBackend._build_client's comment for why.
            "config": BotoConfig(retries={"max_attempts": 0, "mode": "standard"}),
        }
        if config.bedrock_aws_access_key_id:
            client_kwargs["aws_access_key_id"] = self._decode_credential(
                config.bedrock_aws_access_key_id
            )
        if config.bedrock_aws_secret_access_key:
            client_kwargs["aws_secret_access_key"] = self._decode_credential(
                config.bedrock_aws_secret_access_key
            )
        return session.client(service_name, **client_kwargs)


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

    # -- Batch API (submit_batch / poll_batch) ------------------------------
    #
    # Amazon Bedrock's batch inference: a BYO-S3/BYO-IAM-role async job with no
    # fixed completion window — trelix's second long-running external job
    # pattern after OpenAIEmbedder's (see that class's own Batch API comment
    # above for the shared design). Deliberately BedrockTitanEmbedder-only, not
    # on _BedrockEmbedderBase or BedrockCohereEmbedder: Cohere Embed is not in
    # AWS's supported-models list for Bedrock batch inference at all, only
    # Titan Text Embeddings V2 supports it — the same reasoning that keeps
    # submit_batch/poll_batch off BaseEmbedder entirely (VoyageEmbedder has no
    # batch mechanism either).
    #
    # Verified against real AWS docs/boto3 reference during this session (not
    # assumed from memory): CreateModelInvocationJob's request shape,
    # GetModelInvocationJob's response shape (confirms outputDataConfig IS
    # returned on every describe call, so poll_batch never needs to be handed
    # the output URI by a caller), and a real AWS ML blog's worked example of
    # locating a job's actual output object under its output prefix.

    _BATCH_TERMINAL_FAILURE_STATUSES = frozenset({"Failed", "Expired", "Stopped"})
    _BATCH_NON_TERMINAL_STATUSES = frozenset(
        {"Submitted", "Validating", "Scheduled", "InProgress", "Stopping"}
    )
    # Bedrock writes the JSONL result next to a manifest.json.out summary file
    # under the same output prefix — this suffix is what distinguishes it.
    _BATCH_OUTPUT_SUFFIX = ".jsonl.out"

    # Class-level default (not just an __init__ assignment) so that
    # MagicMock(spec=BedrockTitanEmbedder) — the indexer test suite's mocking
    # convention — exposes this attribute at all; a spec mock only sees
    # attributes dir() finds on the class, not ones assigned solely inside
    # __init__ on a real instance.
    last_batch_s3_uris: tuple[str, str] | None = None

    def __init__(self, config: EmbedderConfig) -> None:
        self._client = self._make_boto3_client(config)
        self._model = config.bedrock_titan_model
        self._dims = config.bedrock_titan_dimensions
        self._normalize = config.bedrock_titan_normalize
        self._config = config  # kept for lazy control-plane/S3 client construction
        self._bedrock_control_client: Any | None = None
        self._s3_client: Any | None = None
        # Set by submit_batch(), read by the indexer immediately afterwards to
        # persist via db.insert_batch_job() — submit_batch's return type is
        # pinned to `str` (matching OpenAIEmbedder.submit_batch's contract), so
        # the S3 URIs it generated ride along on the instance instead.
        self.last_batch_s3_uris: tuple[str, str] | None = None

    def _get_bedrock_control_client(self) -> Any:
        """Lazily create the 'bedrock' control-plane client (create/get
        model invocation job) — a different client from 'bedrock-runtime'
        (self._client, used for real-time invoke_model)."""
        if self._bedrock_control_client is None:
            self._bedrock_control_client = self._make_boto3_client(self._config, "bedrock")
        return self._bedrock_control_client

    def _get_s3_client(self) -> Any:
        """Lazily create the S3 client used to upload the batch input JSONL
        and discover/download its output JSONL."""
        if self._s3_client is None:
            self._s3_client = self._make_boto3_client(self._config, "s3")
        return self._s3_client

    @with_retry(max_attempts=5)
    def _invoke_model(self, **kwargs: Any) -> Any:
        return self._client.invoke_model(**kwargs)

    @with_retry(max_attempts=5)
    def _create_model_invocation_job(self, **kwargs: Any) -> Any:
        return self._get_bedrock_control_client().create_model_invocation_job(**kwargs)

    @with_retry(max_attempts=5)
    def _get_model_invocation_job(self, **kwargs: Any) -> Any:
        return self._get_bedrock_control_client().get_model_invocation_job(**kwargs)

    @with_retry(max_attempts=5)
    def _s3_put_object(self, **kwargs: Any) -> Any:
        return self._get_s3_client().put_object(**kwargs)

    @with_retry(max_attempts=5)
    def _s3_list_objects_v2(self, **kwargs: Any) -> Any:
        return self._get_s3_client().list_objects_v2(**kwargs)

    @with_retry(max_attempts=5)
    def _s3_get_object(self, **kwargs: Any) -> Any:
        return self._get_s3_client().get_object(**kwargs)

    def _invoke_body(self, text: str) -> dict[str, Any]:
        """The real-time invoke_model request body for *text* — shared by
        _embed_one() (the synchronous path) and submit_batch() (each batch
        record's "modelInput", which Bedrock passes through to InvokeModel
        verbatim for an InvokeModel-type batch job) so the two never drift
        into two different shapes for the same model."""
        return {
            "inputText": text,
            "dimensions": self._dims,
            "normalize": self._normalize,
        }

    def _embed_one(self, text: str) -> list[float]:
        import json

        body = json.dumps(self._invoke_body(text))
        response = self._invoke_model(
            modelId=self._model,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        payload = json.loads(response["body"].read())
        # Titan is the one Bedrock embedder that reports usage, as
        # inputTextTokenCount on the response body.
        _count_embed_call("bedrock-titan", self._model, [text], payload.get("inputTextTokenCount"))
        return payload["embedding"]  # type: ignore[no-any-return]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text)

    async def embed_async(self, texts: list[str]) -> list[list[float]]:
        """Parallelise per-text boto3 calls in a thread pool."""
        loop = asyncio.get_event_loop()
        tasks = [loop.run_in_executor(_get_sync_executor(), self._embed_one, t) for t in texts]
        return list(await asyncio.gather(*tasks))

    def submit_batch(self, texts: list[str]) -> str:
        """Submit *texts* as a Bedrock batch inference job on Titan Text
        Embeddings V2. Returns the job's jobArn immediately — this does not
        wait for completion; pair with poll_batch()/poll_batch_records().

        BYO-infrastructure: requires an S3 bucket and IAM role the operator
        has already provisioned (TRELIX_BEDROCK_BATCH_S3_BUCKET /
        TRELIX_BEDROCK_BATCH_ROLE_ARN). trelix never calls CreateBucket,
        PutBucketPolicy, CreateRole, or PutRolePolicy — raises a clear
        ValueError naming both required env vars up front rather than letting
        an unset one surface as a confusing boto3 error three calls deep.

        Each text becomes one JSONL record, {"recordId": str(index),
        "modelInput": {...}} — recordId is the stringified position of *text*
        in the *texts* list, which poll_batch()/poll_batch_records() map back
        to when reading the output.
        """
        import json
        import uuid

        bucket = self._config.bedrock_batch_s3_bucket
        role_arn = self._config.bedrock_batch_role_arn
        if not bucket or not role_arn:
            raise ValueError(
                "Bedrock Titan Batch API requires both TRELIX_BEDROCK_BATCH_S3_BUCKET "
                "and TRELIX_BEDROCK_BATCH_ROLE_ARN to be set. Both must name an S3 "
                "bucket and IAM role the operator has already provisioned -- trelix "
                "never creates either itself. Set both environment variables, or "
                "leave --use-batch-api off for this provider."
            )

        job_uuid = str(uuid.uuid4())
        prefix = f"trelix-batch/{job_uuid}/"
        lines = [
            json.dumps({"recordId": str(index), "modelInput": self._invoke_body(text)})
            for index, text in enumerate(texts)
        ]
        input_key = f"{prefix}input.jsonl"
        self._s3_put_object(
            Bucket=bucket,
            Key=input_key,
            Body=("\n".join(lines) + "\n").encode("utf-8"),
        )

        input_uri = f"s3://{bucket}/{input_key}"
        output_uri = f"s3://{bucket}/{prefix}output/"
        response = self._create_model_invocation_job(
            jobName=f"trelix-batch-{job_uuid}",
            roleArn=role_arn,
            modelId=self._model,
            inputDataConfig={"s3InputDataConfig": {"s3Uri": input_uri}},
            outputDataConfig={"s3OutputDataConfig": {"s3Uri": output_uri}},
        )
        self.last_batch_s3_uris = (input_uri, output_uri)
        return response["jobArn"]  # type: ignore[no-any-return]

    def _read_batch_output(self, job_id: str, s3_output_uri: str) -> dict[int, list[float]]:
        """Discover and parse the JSONL Bedrock generated for *job_id* under
        *s3_output_uri*, keyed by original input position (not by output-file
        line order, and not by any positional zip/index assumption).

        Bedrock generates the exact output object key itself — verified
        against a real AWS ML blog's worked example
        (job_details["outputDataConfig"]["s3OutputDataConfig"]["s3Uri"], with
        the actual filename one level deeper than the prefix) — it is not
        simply "output.jsonl". Rather than hardcode that exact nested path
        (which could drift across model/API versions), this discovers the
        real key(s) via ListObjectsV2 under the known output prefix and reads
        every JSONL result file, excluding the manifest.json.out Bedrock also
        writes there.

        Bedrock's own docs state it "generates an output JSONL file for each
        input JSONL file" — submit_batch() currently only ever uploads one
        input.jsonl per job, so today there is exactly one matching object,
        but this reads and merges ALL of them rather than assuming index [0]
        is the only one: a future change that splits a large batch across
        multiple input files (e.g. to respect a records-per-file quota) would
        otherwise silently lose every record in every shard but the first,
        with no error to signal it.
        """
        import json

        bucket, key_prefix = _split_s3_uri(s3_output_uri)
        listing = self._s3_list_objects_v2(Bucket=bucket, Prefix=key_prefix)
        candidate_keys = [
            obj["Key"]
            for obj in listing.get("Contents", [])
            if obj["Key"].endswith(self._BATCH_OUTPUT_SUFFIX)
        ]
        if not candidate_keys:
            raise RuntimeError(
                f"Bedrock batch inference job {job_id!r} is reported complete, but no "
                f"{self._BATCH_OUTPUT_SUFFIX!r} output object was found under "
                f"s3://{bucket}/{key_prefix} -- cannot recover any vectors from this job."
            )

        records: dict[int, list[float]] = {}
        for key in candidate_keys:
            output_object = self._s3_get_object(Bucket=bucket, Key=key)
            body = output_object["Body"].read().decode("utf-8")
            for line in body.splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                record_id = record.get("recordId")
                if record_id is None:
                    continue
                model_output = record.get("modelOutput")
                if not model_output or "embedding" not in model_output:
                    # Per-record failure ({"recordId": ..., "error": {...}}) --
                    # correctly omitted, never filled with a placeholder that
                    # would shift every later position out of alignment.
                    continue
                records[int(record_id)] = model_output["embedding"]
        return records

    def poll_batch_records(self, job_id: str) -> dict[int, list[float]] | None:
        """Poll a Bedrock batch inference job by jobArn, returning vectors
        keyed by their ORIGINAL input position (the int form of the recordId
        submit_batch assigned) rather than a bare ordered list.

        This is the position-preserving variant poll_batch() itself is built
        from, and the one the indexer uses to pair vectors back to the
        correct chunk_ids without a positional zip/index assumption: a
        PartiallyCompleted job's dict simply has gaps (missing keys) where a
        record failed, instead of every later vector shifting down to fill
        the gap the way a plain list would.

        Returns None while non-terminal (Submitted/Validating/Scheduled/
        InProgress/Stopping). Raises BatchJobTerminalError for a terminal
        failure (Failed/Expired/Stopped) -- there are no vectors to recover.
        Completed and PartiallyCompleted both return a dict; PartiallyCompleted
        is a SUCCESS path here (see poll_batch()'s docstring), not an error.
        """
        response = self._get_model_invocation_job(jobIdentifier=job_id)
        status = response["status"]
        if status in self._BATCH_NON_TERMINAL_STATUSES:
            return None
        if status in self._BATCH_TERMINAL_FAILURE_STATUSES:
            raise BatchJobTerminalError(
                f"Bedrock batch inference job {job_id!r} ended with status {status!r}"
            )
        # Completed or PartiallyCompleted: GetModelInvocationJob always
        # returns outputDataConfig (verified against the real API response
        # shape), so no caller ever needs to pass the output URI in.
        output_uri = response["outputDataConfig"]["s3OutputDataConfig"]["s3Uri"]
        return self._read_batch_output(job_id, output_uri)

    def poll_batch(
        self, job_id: str, expected_count: int | None = None
    ) -> list[list[float]] | None:
        """Poll a Bedrock batch inference job by jobArn.

        Returns None while the job is still in flight. Once terminal, returns
        the embedding vectors re-ordered by original input position (never a
        positional zip/list-index assumption over the output file's own line
        order). Raises BatchJobTerminalError if the job reaches
        Failed/Expired/Stopped.

        Unlike OpenAIEmbedder.poll_batch, a returned-count short of
        *expected_count* never raises BatchJobIncompleteError here:
        PartiallyCompleted is Bedrock's own status telling you upfront that
        some records failed, which is a strictly better signal than inferring
        it from a count mismatch after the fact -- OpenAI has no
        PartiallyCompleted-equivalent status, which is why *it* needs the
        count check and this does not. *expected_count* is accepted only for
        interface parity with OpenAIEmbedder.poll_batch (the indexer calls
        both polymorphically) and is intentionally unused for any raise
        decision here; the returned list is simply shorter, and whatever
        chunks it omits are picked up by trelix's own next-run reconciliation
        exactly like a partial real-time embedding failure already degrades
        today. Prefer poll_batch_records() when you need to know WHICH
        original positions succeeded (the indexer does).
        """
        del expected_count  # status is the source of truth here, not the count
        records = self.poll_batch_records(job_id)
        if records is None:
            return None
        return [records[i] for i in sorted(records)]

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

    @with_retry(max_attempts=5)
    def _invoke_model(self, **kwargs: Any) -> Any:
        return self._client.invoke_model(**kwargs)

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
        response = self._invoke_model(
            modelId=self._model,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        # `safe`, not `texts`: the pre-truncated strings are what was actually
        # sent and billed. Cohere on Bedrock reports no token count → None.
        _count_embed_call("bedrock-cohere", self._model, safe)
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
            "bedrock-titan", "bedrock-cohere", "bge-code", "nomic-code",
            "cohere".

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
        case "nomic-code":
            from trelix.embedder.nomic_code import NomicCodeEmbedder

            return NomicCodeEmbedder(config)
        case "cohere":
            from trelix.embedder.cohere import CohereEmbedder

            return CohereEmbedder(config)
        case _:
            raise ValueError(
                f"Unknown embedder provider: {config.provider!r}. "
                "Expected one of: 'local', 'openai', 'azure', 'voyage', "
                "'local-code', 'bedrock-titan', 'bedrock-cohere', 'bge-code', "
                "'nomic-code', 'cohere'."
            )
