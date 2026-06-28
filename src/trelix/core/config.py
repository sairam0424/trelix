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

from pydantic import Field, field_validator
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


class ChunkerConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TRELIX_CHUNKER_")

    max_tokens_per_chunk: int = 512
    include_imports_in_header: bool = True
    max_imports_in_header: int = 8
    include_parent_signature: bool = True

    # Contextual chunking (off by default — requires LLM API access)
    contextual: bool = False
    contextual_model: str = "gpt-4o-mini"
    contextual_max_tokens: int = 100


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
            return self.voyage_dimensions
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
    backend: Literal["sqlite", "qdrant"] = Field(
        default="sqlite",
        validation_alias="TRELIX_STORE_BACKEND",
    )

    # ── Qdrant connection ────────────────────────────────────────────────────
    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    qdrant_api_key: str | None = Field(default=None, alias="QDRANT_API_KEY")
    qdrant_collection: str = Field(default="trelix", alias="QDRANT_COLLECTION")


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
    graph_expansion_depth: int = 1
    graph_expansion_max_symbols: int = 10
    graph_import_max_extra: int = 3

    rrf_k: int = 60

    rerank: bool = True
    rerank_provider: Literal["cohere", "cross_encoder"] = "cohere"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_n: int = 15

    # Cohere reranker
    cohere_api_key: str | None = Field(default=None, alias="COHERE_API_KEY")
    cohere_endpoint: str | None = Field(default=None, alias="COHERE_ENDPOINT")
    cohere_rerank_model: str = Field(default="Cohere-rerank-v4.0-pro", alias="COHERE_MODEL_RERANK")

    context_token_budget: int = 12_000
    synthesis_max_tokens: int = 12_000

    # GraphRAG map-reduce synthesis
    graph_rag_enabled: bool = Field(default=True, alias="TRELIX_RETRIEVAL_GRAPH_RAG")
    graph_rag_threshold_tokens: int = 8000
    graph_rag_threshold_results: int = 20

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
