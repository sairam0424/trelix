"""
Validated configuration for every stage of the pipeline.
Uses pydantic-settings so values are overridable via environment variables
or a .env file — no hardcoded secrets.

Default embedding provider is `local` (sentence-transformers, no API key).
Set TRELIX_EMBEDDER_PROVIDER=openai and OPENAI_API_KEY for higher quality.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

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
        Language.PYTHON, Language.JAVASCRIPT, Language.TYPESCRIPT,
        Language.TSX, Language.GO, Language.RUST, Language.JAVA,
        Language.KOTLIN, Language.RUBY,
        Language.CPP, Language.C,
        Language.CSHARP, Language.RAZOR, Language.CSHTML, Language.CSPROJ,
        Language.MARKDOWN, Language.JSON, Language.YAML, Language.TOML,
        Language.HTML, Language.CSS,
    ]
    max_file_size_bytes: int = 500_000
    respect_gitignore: bool = True
    extra_ignore_dirs: list[str] = [
        ".git", ".hg", ".svn",
        "node_modules", "__pycache__", ".mypy_cache", ".ruff_cache",
        "venv", ".venv", "env",
        "dist", "build", "target", "out", ".next", ".nuxt",
        "coverage", ".coverage", "vendor", "Pods", ".gradle", ".idea", ".vscode",
        ".angular", "19.2.17",
        # .NET build output
        "bin", "obj", "packages", ".vs", ".rider",
        # trelix own index data — never index the index
        ".trelix",
    ]
    extra_ignore_filenames: list[str] = [
        "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "bun.lockb",
        "angular.json",
    ]
    extra_ignore_extensions: list[str] = [
        ".pyc", ".pyo", ".so", ".dylib", ".dll", ".exe",
        ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".webp",
        ".pdf", ".zip", ".tar", ".gz", ".bz2",
        ".min.js", ".min.css", ".lock",
        ".nupkg", ".snupkg", ".pdb", ".ilk", ".exp", ".lib",
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

    provider: Literal["openai", "azure", "local", "voyage", "local-code"] = "local"

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

    batch_size: int = 64

    # ── Indexing performance / rate limiting ─────────────────────────────────
    embed_max_tokens_per_batch: int = 100_000
    tpm_limit: int = 0   # 0 = unlimited (local provider has no rate limit)

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
        return 384   # all-MiniLM-L6-v2


class StoreConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="TRELIX_STORE_",
        populate_by_name=True,
    )

    db_path: str = ".trelix/index.db"

    # HNSW index settings (sqlite-vec ≥ 0.1.6)
    hnsw: bool = Field(default=True, alias="TRELIX_STORE_HNSW")
    hnsw_m: int = Field(default=16, alias="TRELIX_STORE_HNSW_M")
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = Field(default=50, alias="TRELIX_STORE_HNSW_EF_SEARCH")


class RetrievalConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="TRELIX_RETRIEVAL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
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
    cohere_rerank_model: str = Field(
        default="Cohere-rerank-v4.0-pro", alias="COHERE_MODEL_RERANK"
    )

    context_token_budget: int = 12_000
    synthesis_max_tokens: int = 12_000


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

    walker:    WalkerConfig    = Field(default_factory=WalkerConfig)
    parser:    ParserConfig    = Field(default_factory=ParserConfig)
    chunker:   ChunkerConfig   = Field(default_factory=ChunkerConfig)
    embedder:  EmbedderConfig  = Field(default_factory=EmbedderConfig)
    store:     StoreConfig     = Field(default_factory=StoreConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)

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
