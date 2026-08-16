"""
Validated configuration for every stage of the pipeline.
Uses pydantic-settings so values are overridable via environment variables
or a .env file — no hardcoded secrets.

Default embedding provider is `local` (sentence-transformers, no API key).
Set TRELIX_EMBEDDER_PROVIDER=openai and OPENAI_API_KEY for higher quality.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .models import Language

# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------


class WalkerConfig(BaseSettings):
    """Controls which files get indexed."""

    model_config = SettingsConfigDict(env_prefix="TRELIX_WALKER_")

    languages: list[Language] = [
        Language.PYTHON,
        Language.JAVASCRIPT,
        Language.TYPESCRIPT,
        Language.TSX,
        Language.GO,
        Language.RUST,
        Language.JAVA,
        Language.KOTLIN,
        Language.RUBY,
        Language.CPP,
        Language.C,
        Language.CSHARP,
        Language.RAZOR,
        Language.CSHTML,
        Language.CSPROJ,
        Language.MARKDOWN,
        Language.JSON,
        Language.YAML,
        Language.TOML,
        Language.HTML,
        Language.CSS,
    ]
    max_file_size_bytes: int = 500_000
    respect_gitignore: bool = True
    # Default True preserves the behaviour every existing index was built with:
    # symlinks are traversed, and a symlinked file outside repo_path is indexed
    # under a rel_path that looks like it is inside (`relative_to` is computed on
    # the UNRESOLVED path). Set False to confine the walk to repo_path by
    # resolved path — opt-in because flipping it would silently drop files from
    # any repo that symlinks to shared or vendored directories.
    follow_symlinks: bool = True
    extra_ignore_dirs: list[str] = [
        ".git",
        ".hg",
        ".svn",
        "node_modules",
        "__pycache__",
        ".mypy_cache",
        ".ruff_cache",
        "venv",
        ".venv",
        "env",
        "dist",
        "build",
        "target",
        "out",
        ".next",
        ".nuxt",
        "coverage",
        ".coverage",
        "vendor",
        "Pods",
        ".gradle",
        ".idea",
        ".vscode",
        ".angular",
        # .NET build output
        "bin",
        "obj",
        "packages",
        ".vs",
        ".rider",
        # trelix own index data — never index the index
        ".trelix",
    ]
    extra_ignore_filenames: list[str] = [
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "bun.lockb",
        "angular.json",
    ]
    extra_ignore_extensions: list[str] = [
        ".pyc",
        ".pyo",
        ".so",
        ".dylib",
        ".dll",
        ".exe",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".svg",
        ".ico",
        ".webp",
        ".pdf",
        ".zip",
        ".tar",
        ".gz",
        ".bz2",
        ".min.js",
        ".min.css",
        ".lock",
        ".nupkg",
        ".snupkg",
        ".pdb",
        ".ilk",
        ".exp",
        ".lib",
    ]


class ParserConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TRELIX_PARSER_")

    extract_calls: bool = True
    extract_imports: bool = True
    max_symbol_lines: int = 500
    # Data-flow analysis -- def-use chain extraction (no new deps)
    dataflow_enabled: bool = Field(
        default=False,
        alias="TRELIX_PARSER_DATAFLOW",
    )
    # Taint analysis -- requires pip install trelix[taint]
    taint_enabled: bool = Field(
        default=False,
        alias="TRELIX_PARSER_TAINT",
    )


class ChunkerConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TRELIX_CHUNKER_")

    max_tokens_per_chunk: int = 512
    include_imports_in_header: bool = True
    max_imports_in_header: int = 8
    include_parent_signature: bool = True
    include_diagram_tags: bool = True

    # Contextual chunking (off by default — requires LLM API access)
    contextual: bool = False
    contextual_model: str = "gpt-4o-mini"
    contextual_max_tokens: int = 100

    # Multi-granularity sub-symbol indexing (MGS3, off by default)
    multi_granularity_enabled: bool = Field(
        default=False,
        alias="TRELIX_CHUNKER_MULTI_GRANULARITY",
    )
    multi_granularity_levels: list[str] = Field(
        default_factory=lambda: ["block", "statement"],
        alias="TRELIX_CHUNKER_GRANULARITY_LEVELS",
    )


class SparseConfig(BaseSettings):
    """Configuration for learned sparse embeddings (SPLADE-Code)."""

    model_config = SettingsConfigDict(env_prefix="TRELIX_SPARSE_")

    # The previous default, "naver-splab/splade-code-distil", does not exist on the
    # HuggingFace Hub (model_info raises RepositoryNotFoundError), so SparseEmbedder
    # never loaded and sparse_embeddings stayed at 0 rows however the flag was set.
    #
    # Correcting the id alone would not have been enough. Both real SPLADE-Code
    # releases — naver/splade-code-8B and naver/splade-code-06B — are
    # `model_type=qwen3`, and qwen3 is absent from transformers'
    # MODEL_FOR_MASKED_LM_MAPPING_NAMES, so the AutoModelForMaskedLM call in
    # embedder/sparse.py cannot load them. SPLADE-Code is causal-LM based while this
    # loader assumes the BERT-family MaskedLM shape.
    #
    # So the default is a real BERT-family SPLADE the existing loader can actually
    # load: 269 MB, verified to return non-empty token weights for code snippets. It is
    # NL-trained rather than code-specialised, which is the trade-off — a working NL
    # sparse leg is worth more than a code-specialised one that cannot be loaded at all.
    # Using a SPLADE-Code model needs causal-LM support in sparse.py first.
    #
    # LICENCE — read before enabling this in a commercial product. The SPLADE weights
    # are published under CC BY-NC-SA-4.0 (NON-COMMERCIAL, share-alike), verified via
    # the HuggingFace API: cardData.license == "cc-by-nc-sa-4.0". trelix itself is MIT
    # and does not redistribute them — they are downloaded at runtime, and only if you
    # opt into the sparse leg, which is off by default. But the obligations attach to
    # whoever downloads and uses them, so a commercial deployment should either pick a
    # permissively-licensed model via TRELIX_SPARSE_MODEL or leave the leg disabled.
    # Every naver/splade* checkpoint checked carries the same licence.
    model: str = Field(
        default="naver/splade-v3-distilbert",
        alias="TRELIX_SPARSE_MODEL",
    )
    top_k_tokens: int = Field(
        default=128,
        ge=16,
        le=512,
        alias="TRELIX_SPARSE_TOP_K_TOKENS",
    )
    batch_size: int = Field(
        default=16,
        ge=1,
        alias="TRELIX_SPARSE_BATCH_SIZE",
    )


class EmbedderConfig(BaseSettings):
    """
    Embedding provider config.

    Default provider: "local" — uses sentence-transformers, no API key needed.
    Set TRELIX_EMBEDDER_PROVIDER=openai + OPENAI_API_KEY for higher quality.
    Set TRELIX_EMBEDDER_PROVIDER=azure + AZURE_API_KEY + AZURE_ENDPOINT for Azure.
    """

    model_config = SettingsConfigDict(
        env_prefix="TRELIX_EMBEDDER_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    provider: Literal[
        "openai",
        "azure",
        "local",
        "voyage",
        "local-code",
        "bedrock-titan",
        "bedrock-cohere",
        "bge-code",
        "nomic-code",
    ] = "local"

    # ── OpenAI ───────────────────────────────────────────────────────────────
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = "text-embedding-3-large"
    openai_dimensions: int = 3072

    # ── Azure OpenAI ─────────────────────────────────────────────────────────
    azure_api_key: str | None = Field(default=None, alias="AZURE_API_KEY")
    azure_endpoint: str | None = Field(default=None, alias="AZURE_ENDPOINT")
    azure_api_version: str = Field(default="2025-04-01-preview", alias="AZURE_API_VERSION")
    azure_embeddings_deployment: str = Field(
        default="text-embedding-3-large", alias="AZURE_EMBEDDINGS_MODEL"
    )
    azure_chat_deployment: str = Field(default="gpt-4o", alias="AZURE_CHAT_MODEL")
    azure_dimensions: int = 3072

    # ── OpenAI chat model (for planner + synthesizer) ─────────────────────────
    openai_chat_model: str = Field(default="gpt-4o", alias="OPENAI_MODEL")

    # ── Local (sentence-transformers) ────────────────────────────────────────
    local_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ── Voyage (code-optimised API embedder) ─────────────────────────────────
    voyage_api_key: str | None = Field(default=None, alias="VOYAGE_API_KEY")
    voyage_model: str = Field(default="voyage-code-3", alias="TRELIX_EMBEDDER_VOYAGE_MODEL")
    voyage_dimensions: int = 1024
    # Matryoshka output dimension (voyage-code-3 supports 256/512/1024/2048).
    # None = use full voyage_dimensions. Set smaller for faster HNSW search.
    voyage_output_dimensions: int | None = None

    # ── Local-code (SFR-Embedding-Code-2B_R) ─────────────────────────────────
    local_code_model: str = "Salesforce/SFR-Embedding-Code-2B_R"
    local_code_dimensions: int = 4096

    # ── BGE-Code-v1 (BAAI, CoIR SOTA 2025) ────────────────────────────────────
    # Uses FlagEmbedding library. pip install trelix[bge-code]
    bge_code_model: str = "BAAI/bge-code-v1"
    bge_code_dimensions: int = 768  # BGE-Code-v1 default embedding dim

    # ── Nomic CodeRankEmbed ────────────────────────────────────────────────────
    # Uses sentence-transformers. pip install trelix[local]
    nomic_code_model: str = "nomic-ai/CodeRankEmbed"
    nomic_code_dimensions: int = 768  # CodeRankEmbed default embedding dim

    # ── AWS Bedrock (Titan v2 + Cohere) ──────────────────────────────────────
    # Reuses AWS_* env vars — same credentials as BedrockBackend in LLMConfig.
    # bedrock-titan: amazon.titan-embed-text-v2:0 — 256/512/1024 configurable dims
    # bedrock-cohere: cohere.embed-english-v3 — 1024 dims, strong code retrieval
    bedrock_aws_region: str = Field(default="us-east-1", alias="AWS_REGION")
    bedrock_aws_access_key_id: str | None = Field(default=None, alias="AWS_ACCESS_KEY_ID")
    bedrock_aws_secret_access_key: str | None = Field(default=None, alias="AWS_SECRET_ACCESS_KEY")
    bedrock_aws_profile: str | None = Field(default=None, alias="AWS_PROFILE")
    # Titan: configurable dims — 1024 matches voyage quality, 256 cuts storage 4×
    bedrock_titan_model: str = "amazon.titan-embed-text-v2:0"
    bedrock_titan_dimensions: int = 1024  # 256 | 512 | 1024
    bedrock_titan_normalize: bool = True
    # Cohere: fixed 1024 dims, input_type controls doc vs query embedding
    bedrock_cohere_model: str = "cohere.embed-english-v3"
    bedrock_cohere_dimensions: int = 1024

    batch_size: int = 64

    # ── Indexing performance / rate limiting ─────────────────────────────────
    embed_max_tokens_per_batch: int = 100_000
    tpm_limit: int = 0  # 0 = unlimited (local provider has no rate limit)

    @property
    def effective_dimension(self) -> int:
        """Fallback dimension — prefer embedder.dimension after instantiation."""
        if self.provider == "azure":
            return self.azure_dimensions
        if self.provider == "openai":
            return self.openai_dimensions
        if self.provider == "voyage":
            return self.voyage_output_dimensions or self.voyage_dimensions
        if self.provider == "local-code":
            return self.local_code_dimensions
        if self.provider == "bedrock-titan":
            return self.bedrock_titan_dimensions
        if self.provider == "bedrock-cohere":
            return self.bedrock_cohere_dimensions
        if self.provider == "bge-code":
            return self.bge_code_dimensions
        if self.provider == "nomic-code":
            return self.nomic_code_dimensions
        return 384  # all-MiniLM-L6-v2


class StoreConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="TRELIX_STORE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    db_path: str = ".trelix/index.db"

    # HNSW index settings (sqlite-vec ≥ 0.1.6)
    hnsw: bool = Field(default=True, alias="TRELIX_STORE_HNSW")
    hnsw_m: int = Field(default=16, alias="TRELIX_STORE_HNSW_M")
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = Field(default=50, alias="TRELIX_STORE_HNSW_EF_SEARCH")

    # ── Backend selection ────────────────────────────────────────────────────
    backend: Literal["sqlite", "qdrant", "lance"] = Field(
        default="sqlite",
        validation_alias="TRELIX_STORE_BACKEND",
    )

    # ── Qdrant connection ────────────────────────────────────────────────────
    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    qdrant_api_key: str | None = Field(default=None, alias="QDRANT_API_KEY")
    qdrant_collection: str = Field(default="trelix", alias="QDRANT_COLLECTION")
    qdrant_prefer_grpc: bool = Field(default=False, alias="QDRANT_PREFER_GRPC")
    qdrant_timeout: float = Field(default=10.0, alias="QDRANT_TIMEOUT")

    # ── LanceDB connection ───────────────────────────────────────────────────
    lance_uri: str = Field(default=".trelix/lance", alias="LANCE_URI")
    lance_table: str = Field(default="chunks", alias="LANCE_TABLE")

    # ── Parallel BM25 reads ──────────────────────────────────────────────────
    # 0 = disabled (default) — bm25_search() uses the single shared writer
    # connection exactly as before. >0 opts into a pool of that many
    # read-only connections for parallel FTS5 reads.
    bm25_read_pool_size: int = Field(default=0, alias="TRELIX_STORE_BM25_READ_POOL_SIZE")


class RetrievalConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="TRELIX_RETRIEVAL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    top_k_vector: int = 20
    top_k_bm25: int = 20
    top_k_grep: int = 10

    # Vector leg has no SQL predicate to push a path_filter into (unlike
    # BM25/grep), so it over-fetches by this factor and post-filters by
    # rel_path prefix before truncating back to top_k_vector — protects
    # recall against the filter discarding some of the raw ANN results.
    path_filter_oversample: int = Field(
        default=3,
        ge=1,
        alias="TRELIX_RETRIEVAL_PATH_FILTER_OVERSAMPLE",
    )

    graph_expansion_depth: int = 1
    graph_expansion_max_symbols: int = 10
    graph_import_max_extra: int = 3

    rrf_k: int = 60

    rerank: bool = True
    rerank_provider: Literal["cohere", "cross_encoder", "plaid", "xtr"] = "cohere"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_n: int = 15

    # PLAID late-interaction reranker (ColBERT via RAGatouille)
    # pip install trelix[plaid]
    plaid_model: str = Field(
        default="colbert-ir/colbertv2.0",
        alias="TRELIX_RETRIEVAL_PLAID_MODEL",
    )

    # Cohere reranker
    cohere_api_key: str | None = Field(default=None, alias="COHERE_API_KEY")
    cohere_endpoint: str | None = Field(default=None, alias="COHERE_ENDPOINT")
    cohere_rerank_model: str = Field(default="Cohere-rerank-v4.0-pro", alias="COHERE_MODEL_RERANK")

    context_token_budget: int | None = 12_000
    """
    Token budget for context assembly.

    When set to an explicit int (default 12000), uses that fixed budget —
    preserves exact v2.12.0 behavior.

    When set to None, derives budget automatically from the LLM's context window:
      effective_budget = window_size * context_window_fraction
    For example:
      - gpt-4o (128k window) × 0.5 = 64,000 tokens
      - claude-sonnet-4 (200k window) × 0.5 = 100,000 tokens
      - gemini-2.5-pro (1M window) × 0.5 = 500,000 tokens

    Env: TRELIX_RETRIEVAL_CONTEXT_TOKEN_BUDGET=null (or omit for default 12000)
    """

    @field_validator("context_token_budget", mode="before")
    @classmethod
    def _blank_budget_means_auto(cls, v: object) -> object:
        """Make the documented ``=null`` env route actually work.

        Env vars arrive as strings and an ``int | None`` field cannot coerce
        ``"null"``/``"none"``/``""`` to ``None``, so the documented setting above
        raised ValidationError. Since the whole model-aware budget path is gated
        on ``context_token_budget is None``, that made v3.0's auto-derived budget
        reachable only from the Python API — never from env or the CLI.
        """
        if isinstance(v, str) and v.strip().lower() in {"", "null", "none", "auto", "~"}:
            return None
        return v

    context_window_fraction: float = Field(
        default=0.5,
        ge=0.1,
        le=0.9,
        alias="TRELIX_RETRIEVAL_CONTEXT_WINDOW_FRACTION",
    )
    """
    Fraction of model context window to use when context_token_budget=None.

    Default 0.5 leaves half the window for the LLM's response. Conservative
    values (0.3-0.4) prevent context overflow; aggressive values (0.7-0.8)
    maximize retrieval recall at the risk of hitting the window ceiling.

    Only applies when context_token_budget is None — ignored when an explicit
    int budget is set.

    Env: TRELIX_RETRIEVAL_CONTEXT_WINDOW_FRACTION=0.5
    """

    scale_top_k_to_budget: bool = Field(
        default=False,
        alias="TRELIX_RETRIEVAL_SCALE_TOP_K_TO_BUDGET",
    )
    """
    Scale retrieval ceilings (top_k_vector, rerank_top_n) when auto-deriving budget.

    When True and context_token_budget=None, top_k_vector and rerank_top_n are
    scaled proportionally to the effective budget:
      scale_factor = effective_budget / 12000
      top_k_vector = top_k_vector * scale_factor
      rerank_top_n = rerank_top_n * scale_factor

    When False (default), top_k values remain unchanged — preserves v2.12.0
    behavior where rerank_top_n=15 caps candidates BEFORE budget is applied.

    Example: gpt-4o (64k effective budget) with top_k_vector=20:
      scale_factor = 64000 / 12000 = 5.33
      scaled top_k_vector = 20 * 5.33 = ~107

    Env: TRELIX_RETRIEVAL_SCALE_TOP_K_TO_BUDGET=false
    """

    synthesis_max_tokens: int = 12_000

    # Split context_token_budget across source legs (vector/bm25/grep/...)
    # proportionally to each leg's result count, instead of one shared pool a
    # single noisy leg could crowd out. Off by default — False reproduces
    # today's exact single-pool greedy-pack behavior byte-for-byte.
    context_budget_per_source: bool = Field(
        default=False,
        alias="TRELIX_RETRIEVAL_CONTEXT_BUDGET_PER_SOURCE",
    )

    # CodeGraph BFS retrieval (4th leg — off by default)
    graph_search_enabled: bool = False  # Enable CodeGraph as 4th retrieval leg
    graph_search_depth: int = 2  # BFS depth for graph expansion
    graph_search_max_results: int = 15  # Max results from graph search leg

    # File-summary retrieval leg (5th leg — RAPTOR-style, off by default)
    # Requires file_summaries_enabled=True at index time to have any summaries stored.
    file_summary_leg_enabled: bool = Field(
        default=False,
        alias="TRELIX_RETRIEVAL_FILE_SUMMARY_LEG",
    )
    top_k_file_summary: int = Field(
        default=5,
        alias="TRELIX_RETRIEVAL_FILE_SUMMARY_TOP_K",
    )

    # Sparse+dense hybrid retrieval leg (SPLADE-Code, 6th leg — off by default)
    sparse_enabled: bool = Field(
        default=False,
        alias="TRELIX_RETRIEVAL_SPARSE",
    )
    top_k_sparse: int = Field(
        default=20,
        ge=1,
        alias="TRELIX_RETRIEVAL_SPARSE_TOP_K",
    )

    # HyDE fallback — for no-LLM Tier 1 queries, generate a synthetic snippet
    # using the LLM before embedding (requires LLM config).
    # When the planner already set hyde_snippet, this is skipped (no double-call).
    hyde_fallback_enabled: bool = Field(
        default=False,
        alias="TRELIX_RETRIEVAL_HYDE_FALLBACK",
    )
    # Multi-query expansion — generate N query variants, run each as a sub-query
    multi_query_enabled: bool = Field(
        default=False,
        alias="TRELIX_RETRIEVAL_MULTI_QUERY",
    )
    multi_query_count: int = Field(
        default=2,
        ge=1,
        le=4,
        alias="TRELIX_RETRIEVAL_MULTI_QUERY_COUNT",
    )

    # Short-query lexical fallback (Plan B, v2.6.0)
    # Queries with <= short_query_token_threshold meaningful tokens (after stop-word
    # removal) are routed to BM25+grep only, skipping vector ANN embedding.
    # Research: CoREB (arXiv:2605.04615) shows all embedding models score near-zero
    # nDCG@10 on short keyword queries — lexical search wins at this query length.
    short_query_lexical_enabled: bool = Field(
        default=False,
        alias="TRELIX_RETRIEVAL_SHORT_QUERY_LEXICAL",
    )
    short_query_token_threshold: int = Field(
        default=5,
        ge=1,
        le=10,
        alias="TRELIX_RETRIEVAL_SHORT_QUERY_TOKENS",
    )

    # OpenTelemetry tracing — off by default (pip install trelix[otel]).
    # Emits one span per retrieval leg using the gen_ai.* semantic conventions
    # (status: Development, not yet Stable — attribute names may still shift
    # upstream). Zero import/runtime cost when disabled.
    otel_enabled: bool = Field(
        default=False,
        alias="TRELIX_OTEL_ENABLED",
    )
    otel_service_name: str = Field(
        default="trelix",
        alias="OTEL_SERVICE_NAME",
    )
    otel_exporter_endpoint: str | None = Field(
        default=None,
        alias="OTEL_EXPORTER_OTLP_ENDPOINT",
    )

    # FLARE-style confidence-gated re-retrieval
    flare_enabled: bool = Field(
        default=False,
        alias="TRELIX_RETRIEVAL_FLARE",
    )
    flare_max_retries: int = Field(
        default=1,
        ge=1,
        le=3,
        validation_alias=AliasChoices(
            "TRELIX_RETRIEVAL_FLARE_MAX_RETRIES",
            "TRELIX_RETRIEVAL_FLARE_MAX_ITER",  # legacy — deprecated in v2.4
        ),
    )

    @model_validator(mode="after")
    def _warn_deprecated_flare_iter_env(self) -> RetrievalConfig:
        import os
        import warnings

        if "TRELIX_RETRIEVAL_FLARE_MAX_ITER" in os.environ:
            warnings.warn(
                "TRELIX_RETRIEVAL_FLARE_MAX_ITER is deprecated as of trelix v2.4.0. "
                "Use TRELIX_RETRIEVAL_FLARE_MAX_RETRIES instead. "
                "The old name will be removed in v4.0.0.",
                DeprecationWarning,
                stacklevel=2,
            )
        return self

    # PageRank-based symbol importance boost
    pagerank_boost_enabled: bool = Field(
        default=False,
        alias="TRELIX_RETRIEVAL_PAGERANK_BOOST",
    )
    pagerank_boost_factor: float = Field(
        default=1.3,
        ge=1.0,
        le=3.0,
        alias="TRELIX_RETRIEVAL_PAGERANK_BOOST_FACTOR",
    )

    # FTS5 declaration-boost ranking — boosts symbols whose name/qualified_name
    # contain the query match (a declaration-like hit) over ones where the
    # match only appears in docstring/body/context_summary (an incidental
    # mention). Threaded into Database.bm25_search()'s explicit bm25()
    # call; default weight=1.0 is a guaranteed no-op, verified byte-identical
    # to today's unweighted `rank` column ordering.
    declaration_boost_enabled: bool = Field(
        default=False,
        alias="TRELIX_RETRIEVAL_DECLARATION_BOOST",
    )
    declaration_boost_weight: float = Field(
        default=1.0,
        ge=1.0,
        le=10.0,
        alias="TRELIX_RETRIEVAL_DECLARATION_BOOST_WEIGHT",
    )

    # Personalized PageRank — teleport mass concentrated on symbols with a
    # cross-source generic_edge (ticket/artifact reference) instead of the
    # uniform 1/n default. Off by default: nx.pagerank() is called exactly
    # as before when this is False, so there's zero behavior change for
    # anyone not opting in. See rank_by_pagerank() (retrieval/graph.py) and
    # compute_pagerank() (graph/community.py) — both independently gated by
    # this same flag, since they don't share a PageRank implementation.
    #
    # Interaction risk with pagerank_boost_enabled: compute_pagerank()'s
    # teleport mass is uniform across every ticket/artifact-linked symbol,
    # with no weighting by call-graph importance. On a repo where only a
    # small fraction of symbols have ever been referenced by a ticket,
    # enabling both flags together can invert get_top_central_symbols()'s
    # ranking — a single ticket-touched leaf can outscore genuinely central
    # hub symbols that pagerank_boost_enabled is meant to surface. If both
    # are enabled and boost results look off, try disabling personalization
    # first to isolate which one is driving the change.
    pagerank_personalization_enabled: bool = Field(
        default=False,
        alias="TRELIX_RETRIEVAL_PAGERANK_PERSONALIZATION",
    )

    # XTR late-interaction reranker — candidate token count (experimental, v2.6.0)
    xtr_candidate_tokens: int = Field(
        default=100,
        ge=10,
        le=1000,
        alias="TRELIX_RETRIEVAL_XTR_TOKENS",
    )

    # Agentic ReAct loop — multi-turn retrieve+observe+synthesize
    agentic_enabled: bool = Field(
        default=False,
        alias="TRELIX_RETRIEVAL_AGENTIC",
    )
    agent_max_turns: int = Field(
        default=8,
        ge=1,
        le=20,
        alias="TRELIX_RETRIEVAL_AGENT_MAX_TURNS",
    )
    agent_token_budget: int = Field(
        default=6000,
        ge=1000,
        alias="TRELIX_RETRIEVAL_AGENT_TOKEN_BUDGET",
    )
    agent_session_max_age_seconds: float = Field(
        default=604_800.0,  # 7 days. 0 disables eviction entirely.
        ge=0,
        alias="TRELIX_RETRIEVAL_AGENT_SESSION_MAX_AGE_SECONDS",
    )

    # GraphRAG map-reduce synthesis
    graph_rag_enabled: bool = Field(default=True, alias="TRELIX_RETRIEVAL_GRAPH_RAG")
    graph_rag_threshold_tokens: int = 8000
    graph_rag_threshold_results: int = 20

    # Sub-chunk search leg (MGS3 block/statement granularity, off by default)
    sub_chunk_search_enabled: bool = Field(
        default=False,
        alias="TRELIX_RETRIEVAL_SUB_CHUNK",
    )
    top_k_sub_chunk: int = Field(
        default=10,
        ge=1,
        alias="TRELIX_RETRIEVAL_SUB_CHUNK_TOP_K",
    )

    # ── Context compression (SeleCom, off by default) ─────────────────────────
    # Lets a result that does NOT fit the packing budget be INCLUDED in a
    # shrunk, citation-faithful form instead of dropped. Strictly additive:
    # False reproduces today's assembled context byte-for-byte, because the
    # assembler is only handed a compressor when this is True.
    compression_enabled: bool = Field(
        default=False,
        alias="TRELIX_RETRIEVAL_COMPRESSION",
    )
    compression_provider: Literal["extractive"] = Field(
        default="extractive",
        alias="TRELIX_RETRIEVAL_COMPRESSION_PROVIDER",
    )
    """
    Compression backend. "extractive" is zero-inference (it reuses already-stored
    sub-chunk vectors, or a lexical splitter) — it never makes an embedding, API,
    or network call. Abstractive/LLM providers are reserved for v3.4.
    """

    compression_target_ratio: float = Field(
        default=0.45,
        ge=0.1,
        le=1.0,
        alias="TRELIX_RETRIEVAL_COMPRESSION_RATIO",
    )
    """
    Fallback target fraction of a body's original tokens to keep.

    Used only when the intent is unknown/absent — a known intent takes its ratio
    from RetrievalStrategy.compression_ratio (see retrieval/planner/models.py),
    which is per-intent and individually overridable via
    TRELIX_RETRIEVAL_COMPRESSION_RATIO_<INTENT>. 1.0 means "no compression".
    """

    compression_min_tokens: int = Field(
        default=120,
        ge=0,
        alias="TRELIX_RETRIEVAL_COMPRESSION_MIN_TOKENS",
    )
    """
    Bodies below this token count are never compressed — the elision markers and
    per-span headers would cost more than the shrink saves. Such a result is
    handled exactly as today (kept if it fits, skipped if it doesn't).
    """

    # ── Multi-repo federated search ───────────────────────────────────────────
    federation_enabled: bool = Field(
        default=False,
        alias="TRELIX_FEDERATION_ENABLED",
    )
    federation_max_workers: int = Field(
        default=4,
        ge=1,
        le=16,
        alias="TRELIX_FEDERATION_MAX_WORKERS",
    )
    federation_max_repos: int = Field(
        default=50,
        ge=1,
        le=500,
        alias="TRELIX_FEDERATION_MAX_REPOS",
    )

    # ── Query embedding cache ─────────────────────────────────────────────────
    # Caches embed_query() results in-memory (LRU, per-Retriever session).
    # 0 = disabled. Default 256 covers a typical interactive session.
    query_cache_size: int = Field(
        default=256,
        ge=0,
        alias="TRELIX_RETRIEVAL_QUERY_CACHE_SIZE",
    )

    # ── QueryPlan LLM call cache ──────────────────────────────────────────────
    # Caches QueryPlan objects in-memory (LRU, per-Retriever session).
    # 0 = disabled. Default 128: query diversity in a session is lower than
    # embedding diversity, so 128 covers all realistic interactive workloads.
    plan_cache_size: int = Field(
        default=128,
        ge=0,
        alias="TRELIX_RETRIEVAL_PLAN_CACHE_SIZE",
    )

    # ── File-type weighting ──────────────────────────────────────────────────
    # Applies a per-language multiplier to RRF scores after fusion.
    # Env: TRELIX_RETRIEVAL_FILE_TYPE_WEIGHTING=false to disable entirely.
    file_type_weighting_enabled: bool = Field(
        default=True,
        alias="TRELIX_RETRIEVAL_FILE_TYPE_WEIGHTING",
    )
    """
    Master switch. False → no weight multiplier, identical to current behaviour.
    Env: TRELIX_RETRIEVAL_FILE_TYPE_WEIGHTING=false
    """

    file_type_weights: dict[str, float] = Field(
        default_factory=lambda: {
            # Source code — full weight
            "python": 1.0,
            "javascript": 1.0,
            "typescript": 1.0,
            "tsx": 1.0,
            "go": 1.0,
            "rust": 1.0,
            "java": 1.0,
            "kotlin": 1.0,
            "ruby": 1.0,
            "cpp": 1.0,
            "c": 1.0,
            "csharp": 1.0,
            "razor": 1.0,
            "cshtml": 1.0,
            "csproj": 1.0,
            # Style / markup
            "html": 0.4,
            "css": 0.4,
            # Config / data
            "json": 0.5,
            "yaml": 0.5,
            "toml": 0.5,
            # Documentation
            "markdown": 0.3,
            # Unknown — conservative default, do not penalise unknown files
            "unknown": 0.8,
        },
    )
    """
    Per-language RRF score multiplier applied after fusion.
    Keys are Language enum values (lowercase strings).
    Missing key → multiplier = 1.0 (safe fallback, does not downrank unknown types).

    Individual overrides via env (one var per language):
      TRELIX_RETRIEVAL_FILE_TYPE_WEIGHT_MARKDOWN=0.1
      TRELIX_RETRIEVAL_FILE_TYPE_WEIGHT_YAML=0.6
      ...

    Note: Pydantic BaseSettings does not natively merge individual env keys into a
    dict field. The model_post_init hook reads
    TRELIX_RETRIEVAL_FILE_TYPE_WEIGHT_{LANG} vars and merges them on top of
    the default dict at construction time.
    """

    # ── Per-leg RRF weighting ────────────────────────────────────────────────
    # Multiplies each retrieval leg's RRF contribution (1/(k+rank)) before
    # summing, via reciprocal_rank_fusion()'s existing list_weights= param —
    # already implemented, tested, and used in production by
    # FederatedRetriever for per-repo weighting; this just threads it through
    # the main single-repo Retriever._retrieve_standard() call site too.
    # All-1.0 (the default) is a no-op — byte-for-byte identical to today's
    # unweighted fusion. No master enable/disable flag: an all-1.0 dict is
    # already inert, so a separate bool would add config surface without
    # adding capability.
    leg_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "vector": 1.0,
            "bm25": 1.0,
            "grep": 1.0,
            "summary": 1.0,
            "sub_chunk": 1.0,
            "sparse": 1.0,
        },
    )
    """
    Per-leg RRF score multiplier applied during fusion (before summing, not
    after like file_type_weights). Keys match the leg names used in
    Retriever's own leg-size telemetry logging: vector, bm25, grep, summary,
    sub_chunk, sparse. Missing key -> multiplier = 1.0.

    Individual overrides via env (one var per leg):
      TRELIX_RETRIEVAL_LEG_WEIGHT_VECTOR=1.2
      TRELIX_RETRIEVAL_LEG_WEIGHT_BM25=0.8
      ...
    """

    def model_post_init(self, __context: Any) -> None:
        import json
        import os

        # Build the canonical defaults (same dict as default_factory).
        # When pydantic-settings reads TRELIX_RETRIEVAL_FILE_TYPE_WEIGHTS from the
        # environment, it replaces the default_factory result entirely. We merge the
        # env-provided dict ON TOP of defaults so that unspecified keys retain their
        # default values.
        _defaults: dict[str, float] = {
            "python": 1.0,
            "javascript": 1.0,
            "typescript": 1.0,
            "tsx": 1.0,
            "go": 1.0,
            "rust": 1.0,
            "java": 1.0,
            "kotlin": 1.0,
            "ruby": 1.0,
            "cpp": 1.0,
            "c": 1.0,
            "csharp": 1.0,
            "razor": 1.0,
            "cshtml": 1.0,
            "csproj": 1.0,
            "html": 0.4,
            "css": 0.4,
            "json": 0.5,
            "yaml": 0.5,
            "toml": 0.5,
            "markdown": 0.3,
            "unknown": 0.8,
        }

        # If pydantic-settings read TRELIX_RETRIEVAL_FILE_TYPE_WEIGHTS as a JSON
        # string or as a partial dict, self.file_type_weights may only contain the
        # keys from the env var. Merge: defaults ← env-dict, so env wins for
        # specified keys and defaults supply the rest.
        env_weights = os.environ.get("TRELIX_RETRIEVAL_FILE_TYPE_WEIGHTS")
        if env_weights:
            partial = json.loads(env_weights)
            self.file_type_weights = {**_defaults, **partial}
        else:
            # No JSON env var — current value is either the default_factory result or
            # whatever pydantic injected. Ensure all default keys are present.
            self.file_type_weights = {**_defaults, **self.file_type_weights}

        # Per-language overrides (highest priority — applied last).
        # These are NOT picked up by pydantic-settings since they do not match
        # any field name (the field is file_type_weights, not file_type_weight_*).
        prefix = "TRELIX_RETRIEVAL_FILE_TYPE_WEIGHT_"
        for key, val in os.environ.items():
            if key.startswith(prefix):
                lang = key[len(prefix) :].lower()
                self.file_type_weights[lang] = float(val)

        # Per-leg RRF weighting — same defaults-then-env-merge pattern as
        # file_type_weights above.
        _leg_defaults: dict[str, float] = {
            "vector": 1.0,
            "bm25": 1.0,
            "grep": 1.0,
            "summary": 1.0,
            "sub_chunk": 1.0,
            "sparse": 1.0,
        }
        self.leg_weights = {**_leg_defaults, **self.leg_weights}

        leg_prefix = "TRELIX_RETRIEVAL_LEG_WEIGHT_"
        for key, val in os.environ.items():
            if key.startswith(leg_prefix):
                leg = key[len(leg_prefix) :].lower()
                self.leg_weights[leg] = float(val)


class LLMConfig(BaseSettings):
    """
    Chat/synthesis LLM provider config.
    Separate from EmbedderConfig — you can embed with Azure and synthesize
    with Anthropic, for example.
    """

    model_config = SettingsConfigDict(
        env_prefix="TRELIX_LLM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    provider: Literal["openai", "azure", "anthropic", "bedrock", "vertex", "litellm"] = "openai"
    model: str = "gpt-4o"

    # ── OpenAI ──────────────────────────────────────────────────────────────
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")

    # ── Azure OpenAI ─────────────────────────────────────────────────────────
    azure_api_key: str | None = Field(default=None, alias="AZURE_API_KEY")
    azure_endpoint: str | None = Field(default=None, alias="AZURE_ENDPOINT")
    azure_api_version: str = Field(default="2025-04-01-preview", alias="AZURE_API_VERSION")
    azure_chat_deployment: str = Field(default="gpt-4o", alias="AZURE_CHAT_MODEL")

    # ── Anthropic ────────────────────────────────────────────────────────────
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")

    # ── AWS Bedrock ───────────────────────────────────────────────────────────
    aws_region: str = Field(default="us-east-1", alias="AWS_REGION")
    aws_access_key_id: str | None = Field(default=None, alias="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: str | None = Field(default=None, alias="AWS_SECRET_ACCESS_KEY")
    aws_profile: str | None = Field(default=None, alias="AWS_PROFILE")
    # Inference profile IDs (us.* prefix required for on-demand throughput).
    # Primary is tried first; if Bedrock returns a ValidationException (model not
    # available in the region or throughput tier), the backend retries with fallback.
    bedrock_primary_model: str = Field(
        default="us.anthropic.claude-sonnet-4-6",
        alias="TRELIX_LLM_BEDROCK_PRIMARY_MODEL",
    )
    bedrock_fallback_model: str = Field(
        default="us.anthropic.claude-haiku-4-5-20251001-v1:0",
        alias="TRELIX_LLM_BEDROCK_FALLBACK_MODEL",
    )

    # ── Vertex AI / Gemini ────────────────────────────────────────────────────
    google_project_id: str | None = Field(default=None, alias="GOOGLE_CLOUD_PROJECT")
    google_location: str = Field(default="us-central1", alias="GOOGLE_CLOUD_LOCATION")
    google_api_key: str | None = Field(default=None, alias="GOOGLE_API_KEY")

    # ── LiteLLM passthrough ───────────────────────────────────────────────────
    litellm_model: str | None = Field(default=None, alias="TRELIX_LLM_LITELLM_MODEL")
    litellm_drop_params: bool = True

    # ── Common ────────────────────────────────────────────────────────────────
    max_tokens: int = 2048
    temperature: float = 0.0
    timeout: float = 30.0

    # ── Extended thinking (Anthropic only) ────────────────────────────────────
    thinking_enabled: bool = Field(default=False, alias="TRELIX_LLM_THINKING_ENABLED")
    thinking_budget_tokens: int = Field(default=4096, alias="TRELIX_LLM_THINKING_BUDGET_TOKENS")


# ---------------------------------------------------------------------------
# Indexer pipeline config
# ---------------------------------------------------------------------------


class IndexerConfig(BaseSettings):
    """Controls low-level pipeline behaviour inside Indexer."""

    model_config = SettingsConfigDict(
        env_prefix="TRELIX_INDEXER_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # Streaming pipeline: yield files one at a time through a bounded queue
    # instead of collecting all files in memory before parsing begins.
    # Reduces peak memory from O(repo_size) to O(queue_size=64).
    # Default off — opt-in via TRELIX_INDEXER_STREAMING=true.
    streaming_enabled: bool = Field(
        default=False,
        alias="TRELIX_INDEXER_STREAMING",
    )


# Prefixes that form ticket-SHAPED strings but never name a ticket. A ticket key and a
# technical constant are structurally identical — UTF-8 against ENG-8, SHA-256 against
# PROJ-256 — so no amount of regex anchoring separates them and a vocabulary is the only
# thing that can. Measured on this repository: the previous r"[A-Z]+-\d+" matched 12
# strings across 830 commits and every one was a false positive of this kind.
_TICKET_NOISE_PREFIXES: tuple[str, ...] = (
    "UTF", "SHA", "MD", "HTTP", "HTTPS", "BASE", "ISO", "RFC", "AES", "RSA",
    "SSL", "TLS", "IPV", "CRC", "HMAC", "PBKDF", "ARGON", "SIMD", "X", "EC",
    "GB", "KB", "MB", "TB", "PY",
    # Identifier schemes with the same PREFIX-digits shape as a ticket key. CVE matters
    # most: its PREFIX-digits-digits form ("CVE-2021-44228") is truncated to "CVE-2021"
    # by the deliberate trailing-hyphen allowance, so a security advisory in a commit
    # message became a ticket reference — in a tool that ships taint analysis.
    "CVE", "PEP", "RFE", "ADR",
)

# Built from the list above rather than written out, so adding a prefix is a one-word
# edit. Three deliberate choices in the surrounding anchors:
#
#   (?<![A-Za-z0-9])    the key must start at a boundary, so "xPROJ-123" is not a ticket.
#                       A preceding HYPHEN is allowed on purpose: branch and tag names
#                       routinely put a key after one ("feature-PROJ-123",
#                       "release-2024-ENG-45"), and excluding it silently dropped every
#                       such reference. Technical constants are excluded by the
#                       vocabulary below, not by this anchor.
#   [A-Z][A-Z0-9]{1,9}  2-10 characters, upper-case first. Rules out single letters
#                       ("B-1") and lower-case prose while allowing real keys like AB-9.
#   -\d+                greedy, NOT \d{1,6}. A bounded run plus the trailing guard below
#                       made a 7-digit ticket match NOTHING AT ALL: every truncation the
#                       regex tried was followed by another digit, so the lookahead
#                       rejected each one in turn. Greedy consumes the whole number and
#                       the guard then only has a non-digit to inspect.
#   (?![A-Za-z0-9])     rejects a trailing letter, so "PROJ-123x" is not a ticket, but
#                       deliberately ALLOWS a trailing hyphen. That is load-bearing:
#                       branch names are the main place ticket keys appear in merge
#                       subjects ("Merge pull request #12 from feature/PROJ-456-thing"),
#                       and a stricter guard silently dropped every one of them.
TICKET_PATTERN_DEFAULT: str = (
    r"(?<![A-Za-z0-9])"
    # [0-9]* after the alternation, because the key prefix below admits digits
    # ([A-Z][A-Z0-9]{1,9}) while this vocabulary lists only the digit-free spellings.
    # Without it, SHA3-256, X86-64, IPV6-1, UTF8-1 and MD5-1 all read as ticket keys —
    # the same constants the vocabulary exists to exclude, merely spelled with a version
    # number.
    r"(?!(?:" + "|".join(_TICKET_NOISE_PREFIXES) + r")[0-9]*-\d)"
    r"[A-Z][A-Z0-9]{1,9}-\d+"
    r"(?![A-Za-z0-9])"
)


class GitLinkerConfig(BaseSettings):
    """
    Walks `git log` to link code symbols to external ticket references found
    in commit messages (e.g. Jira "PROJ-123", GitHub "#456") — feeds
    generic_edges for cross-source PageRank. Off by default: requires the
    repo to actually be a git checkout, and is a separate, slower pass from
    the main index pipeline (run via `trelix link-tickets`, not auto-invoked
    from Indexer.index()).
    """

    model_config = SettingsConfigDict(
        env_prefix="TRELIX_GIT_LINKER_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    enabled: bool = False
    # Matches Jira-style ticket IDs by default (e.g. "PROJ-123"). Different
    # orgs use different conventions (GitHub "#123", Linear "ENG-123") —
    # override via TRELIX_GIT_LINKER_TICKET_PATTERN.
    ticket_pattern: str = TICKET_PATTERN_DEFAULT
    # Bounds on how much history to walk — required from day one, not an
    # afterthought, since large repos can have 100k+ commits.
    max_commits: int = Field(default=5_000, ge=1)
    since: str | None = None  # e.g. "90 days ago" — passed straight to `git log --since`


class ArtifactLinkerConfig(BaseSettings):
    """
    Links connector-fetched artifacts (Jira tickets, TestRail cases, ...)
    to code symbols by scanning each artifact's title/body for symbol name
    or qualified_name mentions — feeds generic_edges the same way GitLinker
    does for git-commit-message ticket references, but for artifact content
    fetched via `trelix connector sync` (which never touches generic_edges
    on its own).

    Regex reference-extraction runs unconditionally (free, deterministic).
    Embedding-similarity fallback is opt-in and only runs for artifacts
    where the regex pass found zero matches — costs one embed call per
    unmatched artifact, and lower-confidence matches (weight=0.5 vs. a
    regex hit's weight=1.0) so they don't dominate PageRank mass.
    """

    model_config = SettingsConfigDict(
        env_prefix="TRELIX_ARTIFACT_LINKER_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    embedding_fallback_enabled: bool = False
    similarity_threshold: float = Field(default=0.75, ge=0.0, le=1.0)


class JiraConnectorConfig(BaseSettings):
    """
    Jira Cloud REST API connector. HTTP Basic auth (email + API token) —
    no OAuth needed for a read-only, single-project-scope connector.
    Credentials env-only, never in a chart's plaintext values by default,
    matching every other credential in this file.
    """

    model_config = SettingsConfigDict(
        env_prefix="TRELIX_JIRA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    base_url: str | None = Field(default=None, alias="TRELIX_JIRA_BASE_URL")
    email: str | None = Field(default=None, alias="TRELIX_JIRA_EMAIL")
    api_token: str | None = Field(default=None, alias="TRELIX_JIRA_API_TOKEN")
    project_key: str | None = Field(default=None, alias="TRELIX_JIRA_PROJECT_KEY")
    page_size: int = Field(default=100, ge=1, le=100)


class TestRailConnectorConfig(BaseSettings):
    """
    TestRail REST API connector. HTTP Basic auth (username + API key).
    """

    model_config = SettingsConfigDict(
        env_prefix="TRELIX_TESTRAIL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    base_url: str | None = Field(default=None, alias="TRELIX_TESTRAIL_BASE_URL")
    username: str | None = Field(default=None, alias="TRELIX_TESTRAIL_USERNAME")
    api_key: str | None = Field(default=None, alias="TRELIX_TESTRAIL_API_KEY")
    project_id: int | None = Field(default=None, alias="TRELIX_TESTRAIL_PROJECT_ID")
    # TestRail's own API max is 250/page.
    page_size: int = Field(default=250, ge=1, le=250)


class XrayConnectorConfig(BaseSettings):
    """
    Xray Cloud connector (Cloud only — Server/DC has a completely different
    REST-only API surface with PAT/Basic/OAuth1.0a auth, not worth doubling
    this connector's scope for the lowest-priority item in this plan).

    Auth: client_id/client_secret issued by a Jira admin in Xray's global
    settings (distinct from a user's own Jira API token) — exchanged for a
    short-lived bearer JWT via POST /api/v2/authenticate. Xray Cloud tests
    are Jira issues under the hood, so project_key mirrors Jira's shape.
    """

    model_config = SettingsConfigDict(
        env_prefix="TRELIX_XRAY_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    client_id: str | None = Field(default=None, alias="TRELIX_XRAY_CLIENT_ID")
    client_secret: str | None = Field(default=None, alias="TRELIX_XRAY_CLIENT_SECRET")
    project_key: str | None = Field(default=None, alias="TRELIX_XRAY_PROJECT_KEY")
    # Jira base URL is reused for the Jira-issue-fields half of each test
    # (title/url) — Xray Cloud tests are Jira issues, fetched via Jira's own
    # REST v3 API, not Xray's GraphQL endpoint.
    jira_base_url: str | None = Field(default=None, alias="TRELIX_XRAY_JIRA_BASE_URL")
    page_size: int = Field(default=100, ge=1, le=100)


class LinearConnectorConfig(BaseSettings):
    """
    Linear GraphQL API connector. Personal API key auth via
    `Authorization: <API_KEY>` (no Bearer prefix — Linear's own documented
    scheme, distinct from every other connector in this file). Scoped to a
    single team via its key (e.g. "ENG"), mirroring Jira's project_key /
    TestRail's project_id precedent. No base_url field: unlike Jira/
    TestRail/Xray, Linear's GraphQL endpoint is one fixed URL for every
    user (see linear.py's _LINEAR_GRAPHQL_URL), not per-org configurable.

    v1 always does a full resync (no updatedAt filter) — see linear.py's
    module docstring for why incremental sync was explicitly deferred.
    """

    model_config = SettingsConfigDict(
        env_prefix="TRELIX_LINEAR_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    api_key: str | None = Field(default=None, alias="TRELIX_LINEAR_API_KEY")
    team_key: str | None = Field(default=None, alias="TRELIX_LINEAR_TEAM_KEY")
    # No documented max page size found for Linear's issues(first: N)
    # connection — 100 mirrors Jira/Xray's own ceiling and stays well under
    # the 10,000-point per-query complexity cap (~771 points at first=100
    # with this connector's field selection; see linear.py). Not a
    # confirmed platform ceiling — an assumption, flagged as such.
    page_size: int = Field(default=100, ge=1, le=100)


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------


class IndexConfig(BaseSettings):
    """
    Top-level config. Instantiate once and pass through the whole pipeline.

        config = IndexConfig(repo_path="/path/to/repo")
        indexer = Indexer(config)
    """

    model_config = SettingsConfigDict(
        env_prefix="TRELIX_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    repo_path: str
    incremental: bool = True
    parse_workers: int = 4

    walker: WalkerConfig = Field(default_factory=WalkerConfig)
    parser: ParserConfig = Field(default_factory=ParserConfig)
    chunker: ChunkerConfig = Field(default_factory=ChunkerConfig)
    embedder: EmbedderConfig = Field(default_factory=EmbedderConfig)
    store: StoreConfig = Field(default_factory=StoreConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    sparse: SparseConfig = Field(default_factory=SparseConfig)
    indexer: IndexerConfig = Field(default_factory=IndexerConfig)
    git_linker: GitLinkerConfig = Field(default_factory=GitLinkerConfig)

    # Multi-granularity indexing: generate LLM file-level summaries (RAPTOR-style).
    # Requires LLM API access. Off by default — zero cost when disabled.
    file_summaries_enabled: bool = Field(
        default=False,
        alias="TRELIX_FILE_SUMMARIES_ENABLED",
    )

    # Query telemetry: record every retrieve() call to query_telemetry table.
    # Off by default — zero overhead when disabled.
    telemetry_enabled: bool = Field(
        default=False,
        alias="TRELIX_TELEMETRY_ENABLED",
    )

    @field_validator("repo_path")
    @classmethod
    def repo_must_exist(cls, v: str) -> str:
        if not Path(v).exists():
            raise ValueError(f"repo_path does not exist: {v}")
        return str(Path(v).resolve())

    @property
    def db_path_absolute(self) -> Path:
        p = Path(self.store.db_path)
        if not p.is_absolute():
            p = Path(self.repo_path) / p
        p.parent.mkdir(parents=True, exist_ok=True)
        # Prevent Git / IDE watchers from tracking the index files
        gitignore = p.parent / ".gitignore"
        if not gitignore.exists():
            gitignore.write_text("*\n", encoding="utf-8")
        return p


# ---------------------------------------------------------------------------
# Audit config (API-layer, standalone — like _ApiAuthSettings in api/app.py)
# ---------------------------------------------------------------------------


class AuditConfig(BaseSettings):
    """Tamper-evident request auditing — additive and OFF by default.

    Standalone BaseSettings (not nested on IndexConfig) because auditing is an
    API-layer concern loaded once in ``create_app()``, mirroring the
    ``_ApiAuthSettings`` precedent. ``enabled=False`` (the default) means the
    audit middleware is never registered and no ``audit.db`` is ever created —
    byte-identical to pre-audit behavior.

    The audit trail lives in its OWN ``audit.db``, deliberately separate from
    the disposable index DB (which is rebuilt at will) so the trail survives
    re-indexing. When ``db_path`` is unset it defaults to
    ``<cwd>/.trelix/audit.db``.
    """

    model_config = SettingsConfigDict(
        env_prefix="TRELIX_AUDIT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    enabled: bool = False
    db_path: str | None = None
    log_queries: bool = False
    fail_closed: bool = False
    retention_days: int = 365

    @property
    def resolved_db_path(self) -> Path:
        """Absolute-or-relative path to the audit DB, defaulting under .trelix/."""
        if self.db_path:
            return Path(self.db_path)
        return Path.cwd() / ".trelix" / "audit.db"


# ---------------------------------------------------------------------------
# SSO / OIDC config (API-layer, standalone — like _ApiAuthSettings in app.py)
# ---------------------------------------------------------------------------


class SSOConfig(BaseSettings):
    """OIDC single-sign-on — additive and OFF by default.

    Standalone BaseSettings (not nested on IndexConfig) because SSO is an
    API-layer concern loaded once in ``create_app()``, mirroring the
    ``_ApiAuthSettings`` / ``AuditConfig`` precedent. ``enabled=False`` (the
    default) means no :class:`~trelix.auth.oidc.OidcVerifier` is ever built and
    the API behaves byte-identically to today (static-token / open modes).

    ``algorithms`` is an asymmetric-only allowlist — the verifier rejects
    ``alg: none`` and every ``HS*`` variant, so a symmetric algorithm can never
    be configured here in practice.
    """

    model_config = SettingsConfigDict(
        env_prefix="TRELIX_OIDC_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    enabled: bool = False
    issuer: str = ""
    audience: str = ""
    algorithms: list[str] = Field(default_factory=lambda: ["RS256", "ES256"])
    jwks_uri: str = ""
    jwks_ttl_seconds: int = 3600
