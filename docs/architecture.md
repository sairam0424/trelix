# trelix Architecture

> **Version:** 2.7.1 | **Python:** 3.11+ | **110+ source modules**

This document describes the complete architecture of trelix — every layer, every data flow, every design decision, and every class that matters. It is the definitive reference for contributors and anyone integrating trelix at a deep level.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Core Data Models](#2-core-data-models)
3. [Configuration System](#3-configuration-system)
4. [Storage Layer](#4-storage-layer)
5. [Indexing Pipeline](#5-indexing-pipeline)
6. [Embedder Layer](#6-embedder-layer)
7. [Retrieval Pipeline](#7-retrieval-pipeline)
8. [Query Planner](#8-query-planner)
9. [Fusion and Ranking](#9-fusion-and-ranking)
10. [Synthesis and Agent Loop](#10-synthesis-and-agent-loop)
11. [Graph Layer](#11-graph-layer)
12. [Analysis Layer](#12-analysis-layer)
13. [Federation Layer (v2.4.0)](#13-federation-layer-v240)
14. [Review Layer (v2.4.0)](#14-review-layer-v240)
15. [MCP Server](#15-mcp-server)
16. [REST API](#16-rest-api)
17. [CLI Layer](#17-cli-layer)
18. [File Watching](#18-file-watching)
19. [Telemetry and Observability (v2.4.0)](#19-telemetry-and-observability-v240)
20. [LLM Client Abstraction](#20-llm-client-abstraction)
21. [Key Design Invariants](#21-key-design-invariants)
22. [Data Flow Diagrams](#22-data-flow-diagrams)
23. [Extension Points](#23-extension-points)

---

## 1. System Overview

trelix is a **code intelligence engine** built on two orthogonal pipelines:

### Offline Pipeline (index time)

```
Repo files
  └─ FileWalker          (gitignore-aware, SHA-256 incremental hashing)
       └─ Language Router (20+ languages via per-extractor registry)
            └─ tree-sitter Parser (extracts Symbol, CallEdge, TypeEdge, ImportEdge)
                 └─ Chunker (context-headed Chunk per symbol, tiktoken-counted)
                      └─ Embedder (dense vectors; optionally sparse SPLADE)
                           └─ SQLite Store (symbols + FTS5 BM25 + sqlite-vec HNSW)
                                └─ Cross-file Resolution Pass
                                     └─ DimensionGuard (dimension mismatch detection)
```

### Online Pipeline (query time)

```
User Query
  └─ AdaptiveRouter        (Tier 1/2/3 detection — direct/single-step/multi-step)
       └─ QueryPlanner     (LLM intent classification → RetrievalStrategy)
            └─ 7 Retrieval Legs (parallel: vector, BM25, grep, sparse, file-summary, sub-chunk, graph BFS)
                 └─ RRF Fusion  (k=60, per-language weight multipliers)
                      └─ Graph Expansion (call-graph, import-graph, type-edges)
                           └─ Reranker (cross-encoder / Cohere / PLAID ColBERT)
                                └─ PageRank Boost (optional, top-200 central symbols)
                                     └─ ContextAssembler (greedy token budget)
                                          └─ Synthesizer / AgentLoop
                                               └─ RetrievedContext / LLM answer
```

### Architecture Principles

| Principle | Implementation |
|-----------|---------------|
| **Zero-infra default** | All state in `.trelix/index.db` (SQLite + sqlite-vec). No external services required. |
| **Graceful degradation** | Every optional feature (sparse, graph, FLARE, PLAID, taint) catches all exceptions and continues. |
| **Provider agnosticism** | `BaseEmbedder` and `TrelixChatClient` ABCs abstract all providers behind identical interfaces. |
| **Incremental correctness** | SHA-256 file hashes skip unchanged files. `DimensionGuard` prevents silent embedding mismatches. |
| **Data-driven behavior** | `INTENT_STRATEGIES` dict drives all retrieval behavior — no intent-switching logic in `Retriever`. |
| **No mutation** | All dataclasses are created fresh (not mutated) when scores or ranks change in the pipeline. |
| **Lazy imports** | FastAPI, torch, boto3, semgrep imported inside functions only — core install stays minimal. |

---

## 2. Core Data Models

**Module:** `src/trelix/core/models.py`

All models are Python dataclasses. They are the single source of truth that flows from walker → parser → chunker → store → retrieval → synthesis.

### Enumerations

```python
class SymbolKind(StrEnum):
    FUNCTION = "function"
    METHOD = "method"
    CLASS = "class"
    INTERFACE = "interface"
    STRUCT = "struct"
    ENUM = "enum"
    CONSTANT = "constant"
    VARIABLE = "variable"
    MODULE = "module"        # file-level module symbol
    SECTION = "section"      # markdown heading
    UNKNOWN = "unknown"

class Language(StrEnum):
    PYTHON | JAVASCRIPT | TYPESCRIPT | TSX | GO | RUST | JAVA
    CPP | C | CSHARP | RAZOR | CSHTML | CSPROJ | KOTLIN | RUBY
    MARKDOWN | JSON | YAML | TOML | HTML | CSS | UNKNOWN  # 21 total
```

### Data Classes (pipeline order)

**`IndexedFile`** — output of `FileWalker`
```python
@dataclass
class IndexedFile:
    path: str            # absolute path
    rel_path: str        # stable key for incremental hashing, used as dedup key
    language: Language
    hash: str            # SHA-256 of file content
    size_bytes: int
    id: int | None = None          # populated after DB upsert
    indexed_at: datetime | None = None
```

**`Symbol`** — output of tree-sitter extractors
```python
@dataclass
class Symbol:
    file_id: int
    name: str                    # bare identifier (e.g. "login")
    qualified_name: str          # dot-separated (e.g. "AuthService.login")
    kind: SymbolKind
    line_start: int              # 1-indexed, inclusive
    line_end: int
    signature: str               # e.g. "def login(self, user: str) -> bool"
    body: str                    # verbatim source text of the symbol
    docstring: str | None
    decorators: list[str]
    is_public: bool
    parent_id: int | None        # links methods to enclosing class symbol
    id: int | None = None
    context_summary: str | None = None  # LLM-generated (contextual chunking only)
```

**`CallEdge`** — directed call graph edge
```python
@dataclass
class CallEdge:
    caller_id: int
    callee_name: str
    line: int
    callee_id: int | None = None      # resolved in 2nd pass (None = external/stdlib)
    callee_type_hint: str | None = None  # e.g. "UserService" for typed method calls
                                          # used in 4-priority cascade resolution
```

Design note on `callee_type_hint`: when the parser sees `user_service.login()` where `user_service: UserService` is the declared type, it stores `"UserService"` in `callee_type_hint`. During cross-file resolution, this narrows the match to symbols with `qualified_name LIKE "UserService.%"`, dramatically reducing false-positive edges.

**`TypeEdge`** — type hierarchy
```python
@dataclass
class TypeEdge:
    from_symbol_id: int
    to_type_name: str
    edge_kind: str    # "extends" | "implements" | "trait_impl" | "embedded"
    to_symbol_id: int | None = None  # resolved post-store
```

**`ImportEdge`** — file-level import
```python
@dataclass
class ImportEdge:
    file_id: int
    imported_from: str      # module path
    imported_names: list[str]  # ["*"] for wildcard
    imported_file_id: int | None = None  # resolved post-store
```

**`Chunk`** — the unit that gets embedded and indexed for BM25
```python
@dataclass
class Chunk:
    symbol_id: int
    chunk_text: str     # context header + symbol body (NOT just raw code)
    token_count: int    # tiktoken cl100k_base, pre-computed at index time
    embedding: list[float] | None = None
    id: int | None = None
```

The context header prepended to every `chunk_text` (example):
```
# File: src/auth/service.py | Language: python
# Imports: from .models import User, Session; from .crypto import hash_password
# Class: AuthService
def login(self, username: str, password: str) -> Session | None:
    ...
```

This header lets the embedding model understand where the symbol lives in the codebase without seeing the whole file.

**`SearchResult`** — output of each retrieval leg, preserved through fusion
```python
@dataclass
class SearchResult:
    chunk: Chunk
    symbol: Symbol
    file: IndexedFile
    score: float    # raw cosine/BM25/RRF score
    rank: int       # 1-indexed position
    source: str     # "vector" | "bm25" | "grep" | "graph_expansion" |
                    # "file_summary" | "sub_chunk" | "sparse" | "file_direct" |
                    # "graph_callers" | "graph_callees" | "graph_importers"
```

**`RetrievedContext`** — final payload passed to LLM
```python
@dataclass
class RetrievedContext:
    query: str
    results: list[SearchResult]
    context_text: str            # assembled, token-budgeted text for LLM
    total_tokens: int            # how much of the 12k budget was used
    retrieval_sources: dict[str, int]  # {"vector": 8, "bm25": 4, ...}
    elapsed_seconds: float
    intent: str                  # from planner (drives per-intent synthesis prompts)
```

---

## 3. Configuration System

**Module:** `src/trelix/core/config.py`

All configuration classes inherit `pydantic_settings.BaseSettings`. Environment variables are loaded from `.env` file in the repo root, then from shell environment (shell wins). The root object `IndexConfig` is the single configuration instance passed through the entire pipeline.

### Configuration Hierarchy

```
IndexConfig
├── WalkerConfig         TRELIX_WALKER_*
├── ParserConfig         TRELIX_PARSER_*
├── ChunkerConfig        TRELIX_CHUNKER_*
├── SparseConfig         TRELIX_SPARSE_*
├── EmbedderConfig       TRELIX_EMBEDDER_*
├── StoreConfig          TRELIX_STORE_*
├── RetrievalConfig      TRELIX_RETRIEVAL_*
└── LLMConfig            TRELIX_LLM_*
```

### `WalkerConfig`

```python
class WalkerConfig(BaseSettings):
    languages: list[str] = [all 20 languages]     # which to index
    max_file_size_bytes: int = 500_000
    respect_gitignore: bool = True
    extra_ignore_dirs: list[str]  # node_modules, __pycache__, .venv, dist, .trelix, etc.
    extra_ignore_filenames: list[str]  # package-lock.json, yarn.lock, angular.json, etc.
    extra_ignore_extensions: list[str]  # .pyc, .min.js, .lock, .dll, etc.
```

Design note: `.trelix` is in `extra_ignore_dirs` — the walker never indexes the index itself. This prevents recursive indexing when `trelix index .` is run from inside an already-indexed repo.

### `ParserConfig`

```python
class ParserConfig(BaseSettings):
    extract_calls: bool = True
    extract_imports: bool = True
    max_symbol_lines: int = 500        # symbols longer than this are truncated
    dataflow_enabled: bool = False     # TRELIX_PARSER_DATAFLOW — def-use extraction
    taint_enabled: bool = False        # TRELIX_PARSER_TAINT — requires trelix[taint]
```

### `ChunkerConfig`

```python
class ChunkerConfig(BaseSettings):
    max_tokens_per_chunk: int = 512         # tiktoken cl100k_base
    include_imports_in_header: bool = True
    max_imports_in_header: int = 8
    include_parent_signature: bool = True   # method gets enclosing class in header
    contextual: bool = False                # LLM context summaries (TRELIX_CHUNKER_CONTEXTUAL)
    contextual_model: str = "gpt-4o-mini"
    contextual_max_tokens: int = 100
    multi_granularity_enabled: bool = False # TRELIX_CHUNKER_MULTI_GRANULARITY (MGS3)
    multi_granularity_levels: list[str] = ["block", "statement"]
```

### `EmbedderConfig`

```python
class EmbedderConfig(BaseSettings):
    provider: Literal["local","openai","azure","voyage","local-code",
                      "bedrock-titan","bedrock-cohere","bge-code","nomic-code"] = "local"
    batch_size: int = 64
    embed_max_tokens_per_batch: int = 100_000
    tpm_limit: int = 0  # 0 = unlimited
```

Provider dimensions (used by `DimensionGuard`):

| Provider | Model | Dimensions |
|----------|-------|-----------|
| local | all-MiniLM-L6-v2 | 384 |
| openai | text-embedding-3-large | 3072 |
| azure | text-embedding-3-large | 3072 |
| voyage | voyage-code-3 | 1024 (Matryoshka: 256/512/1024/2048) |
| local-code | SFR-Embedding-Code-2B_R | 4096 |
| bge-code | BAAI/bge-code-v1 | 768 |
| nomic-code | nomic-ai/CodeRankEmbed | 768 |
| bedrock-titan | amazon.titan-embed-text-v2:0 | 1024 (configurable: 256/512/1024) |
| bedrock-cohere | cohere.embed-english-v3 | 1024 |

### `RetrievalConfig` (complete)

```python
class RetrievalConfig(BaseSettings):
    # Retrieval leg sizes
    top_k_vector: int = 20
    top_k_bm25: int = 20
    top_k_grep: int = 10
    rrf_k: int = 60              # Cormack 2009 RRF constant

    # Reranking
    rerank: bool = True
    rerank_provider: str = "cohere"   # "cohere" | "cross_encoder" | "plaid"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_n: int = 15            # candidates sent to reranker
    cohere_rerank_model: str = "Cohere-rerank-v4.0-pro"
    plaid_model: str = "colbert-ir/colbertv2.0"

    # Assembly
    context_token_budget: int = 12_000
    synthesis_max_tokens: int = 12_000

    # Graph expansion
    graph_expansion_depth: int = 1       # call-graph hops
    graph_expansion_max_symbols: int = 10

    # Feature flags (all False unless noted)
    graph_search_enabled: bool = False
    file_summary_leg_enabled: bool = False
    sparse_enabled: bool = False
    hyde_fallback_enabled: bool = False
    multi_query_enabled: bool = False
    multi_query_count: int = 2           # range: 1–4
    flare_enabled: bool = False
    flare_max_retries: int = 1           # ge=1, le=3 — v2.4.0 rename
    pagerank_boost_enabled: bool = False
    pagerank_boost_factor: float = 1.3   # 1.0–3.0
    agentic_enabled: bool = False        # ReAct AgentLoop
    agent_max_turns: int = 8             # 1–20
    agent_token_budget: int = 6_000      # history compression budget
    graph_rag_enabled: bool = True       # ON by default
    graph_rag_threshold_tokens: int = 8_000   # trigger threshold
    sub_chunk_search_enabled: bool = False
    federation_enabled: bool = False
    federation_max_workers: int = 4      # 1–16
    telemetry_enabled: bool = False      # v2.4.0

    # LRU caches
    query_cache_size: int = 256   # embed_query() results
    plan_cache_size: int = 128    # QueryPlanner.plan() results

    # File-type RRF weights
    file_type_weighting_enabled: bool = True
    # Defaults: source code=1.0, HTML/CSS=0.4, JSON/YAML/TOML=0.5, Markdown=0.3, Unknown=0.8
    # Override per-language: TRELIX_RETRIEVAL_FILE_TYPE_WEIGHT_PYTHON=1.2
```

### `IndexConfig` (root)

```python
class IndexConfig(BaseSettings):
    repo_path: str             # required; resolved to absolute path and validated
    incremental: bool = True
    parse_workers: int = 4
    file_summaries_enabled: bool = False  # TRELIX_FILE_SUMMARIES_ENABLED
    telemetry_enabled: bool = False       # TRELIX_TELEMETRY_ENABLED

    # Sub-configs (auto-populated from env/file)
    walker: WalkerConfig = ...
    parser: ParserConfig = ...
    chunker: ChunkerConfig = ...
    embedder: EmbedderConfig = ...
    store: StoreConfig = ...
    retrieval: RetrievalConfig = ...
    llm: LLMConfig = ...
    sparse: SparseConfig = ...

    @property
    def db_path_absolute(self) -> Path:
        # Resolves .trelix/index.db relative to repo_path
        # Auto-creates parent dirs
        # Writes .trelix/.gitignore = "*" to prevent accidental index commits
```

---

## 4. Storage Layer

**Module:** `src/trelix/store/db.py`

### SQLite Schema

Pragmas applied at every connection: `WAL`, `foreign_keys = ON`, `synchronous = NORMAL`, `busy_timeout = 5000`, `temp_store = MEMORY`.

```sql
-- Core tables (always present)
files (id PK, path UNIQUE, rel_path, language, hash, size_bytes, indexed_at)

symbols (id PK, file_id FK→files CASCADE,
         name, qualified_name, kind, line_start, line_end,
         signature, docstring, context_summary,
         decorators TEXT,  -- JSON array
         is_public BOOL, parent_id FK→symbols SET_NULL, body,
         content_hash TEXT NOT NULL DEFAULT '')  -- sha256(signature+body); skips re-embed on partial re-index

calls (id PK, caller_id FK→symbols CASCADE,
       callee_name, callee_id FK→symbols SET_NULL,
       line, callee_type_hint TEXT)

imports (id PK, file_id FK→files CASCADE,
         imported_from, imported_names JSON,
         imported_file_id FK→files)  -- nullable, resolved post-store

chunks (id PK, symbol_id FK→symbols CASCADE,
        chunk_text, token_count)

type_edges (id PK, from_symbol_id FK→symbols CASCADE,
            to_type_name, edge_kind, to_symbol_id FK→symbols SET_NULL)

-- FTS5 virtual table (BM25 index)
symbols_fts (name, qualified_name, docstring, body, context_summary)
  CONTENT='symbols' CONTENT_ROWID='id' TOKENIZE='porter ascii'
  -- Maintained by 3 triggers: AFTER INSERT/DELETE/UPDATE on symbols

-- Extension tables (all added via idempotent migrations)
sub_chunks (id PK, parent_symbol_id FK→symbols CASCADE,
            granularity CHECK('function','block','statement'),
            chunk_text, line_start, line_end, token_count)

file_summaries (id PK, file_id UNIQUE FK→files CASCADE,
                summary, chunk_id FK→chunks SET_NULL, created_at)

index_metadata (key PK, value TEXT)   -- dimension guard storage

query_telemetry (id PK, ts, query, intent, elapsed_ms, result_count,
                 leg_sizes JSON, thumbs_up,
                 -- v2.4.0 expansion observability:
                 expansion_used INT, expansion_variants INT, expansion_elapsed_ms REAL)

def_use_edges (id PK, symbol_id, var_name, def_line, use_line,
               edge_type CHECK('def','use'))

taint_flows (id PK, source_file, source_line, sink_file, sink_line,
             rule_id, severity)

sparse_embeddings (chunk_id PK, token_id PK, weight REAL)  -- SPLADE inverted index
```

### Indexes

```sql
idx_symbols_file_id, idx_symbols_name, idx_symbols_kind
idx_calls_caller, idx_calls_callee
idx_chunks_symbol_id
idx_sub_chunks_symbol
idx_file_summaries_file_id
idx_type_edges_from, idx_type_edges_to
idx_def_use_symbol, idx_taint_severity, idx_sparse_token
```

### Migration Pattern

`_apply_migrations()` runs in `Database.__init__` every time. Each migration:
- Uses `PRAGMA table_info()` to detect missing columns
- Uses `CREATE TABLE IF NOT EXISTS` for new tables
- Is fully idempotent — safe to run on any DB version

`content_hash` on `symbols` follows this same idempotent `PRAGMA table_info()` detection pattern — `ALTER TABLE symbols ADD COLUMN content_hash TEXT NOT NULL DEFAULT ''` only runs if the column isn't already present.

**Migration history** (embedded in code, applied automatically):
1. `imported_file_id` on imports table (initial)
2. `decorators`, `is_public`, `context_summary` on symbols
3. `callee_type_hint` on calls (type-hint-guided resolution)
4. `file_summaries` table
5. `sub_chunks` table (MGS3)
6. `query_telemetry` table (v2.3 observability)
7. `expansion_used`, `expansion_variants`, `expansion_elapsed_ms` on `query_telemetry` (v2.4.0)
8. `index_metadata` table (v2.3 DimensionGuard)
9. `def_use_edges`, `taint_flows`, `sparse_embeddings` (v2.2)

### Key `Database` Methods

```python
class Database:
    def __init__(self, db_path: Path) -> None
    
    # File management
    def upsert_file(file: IndexedFile) -> int          # INSERT … ON CONFLICT RETURNING id
    def get_file_hash(rel_path: str) -> str | None     # incremental hash check
    def delete_file_symbols(file_id: int) -> None      # clean before re-index
    def delete_file_by_path(abs_path, rel_path, vector_store) -> bool  # full delete

    # Symbol operations
    def insert_symbol(symbol: Symbol) -> int           # JSON-encodes decorators
    def get_symbol_by_name(name: str) -> list[Symbol]  # all symbols matching bare name
    def get_symbols_for_file(file_id: int) -> list[Symbol]
    def get_chunk_with_context(chunk_id: int) -> tuple[Chunk, Symbol, IndexedFile] | None
    def get_symbol_with_file(symbol_id: int) -> tuple[Symbol, IndexedFile] | None

    # BM25 search (FTS5)
    def bm25_search(query: str, limit: int) -> list[tuple[int, float]]
    # Returns (symbol_id, rank) — lower rank = more relevant (FTS5 returns negative)
    # score = 1.0 / (1.0 + abs(bm25_rank)) for pipeline compatibility

    # Graph queries
    def get_callers(symbol_id: int) -> list[int]       # 1-hop incoming call edges
    def get_callees(symbol_id: int) -> list[int]       # 1-hop resolved outgoing
    def get_imports_for_file(file_id: int) -> list[ImportEdge]
    def get_file_imports_resolved(file_id: int) -> list[int]  # imported file_ids

    # Cross-file resolution passes (called after all files indexed)
    def resolve_cross_file_calls() -> int      # 4-priority cascade
    def resolve_import_file_ids() -> int       # language-aware path normalization
    def resolve_cross_file_type_edges() -> int # class/interface name matching
    def resolve_angular_selectors() -> int     # Angular component selector edges

    # Telemetry (v2.4.0)
    def insert_query_telemetry(query, intent, elapsed_ms, result_count,
                                leg_sizes=None, *, expansion_used=None,
                                expansion_variants=None, expansion_elapsed_ms=None) -> int
    def get_recent_telemetry(limit: int) -> list[dict]

    # Dimension guard
    def get_embedding_dimension() -> int | None
    def set_embedding_dimension(dim: int) -> None
```

### Call Resolution: 4-Priority Cascade

The `resolve_cross_file_calls()` method resolves `callee_id` using this priority order:

1. **Exact qualified_name match** — most precise; e.g. `"UserService.login"` → unique DB match
2. **Type-hint + name match** — when `callee_type_hint = "UserService"`, look for symbols with `qualified_name LIKE "UserService.%"` AND `name = callee_name`
3. **Unique name match** — if only one symbol in the DB has `name = callee_name`
4. **Leave NULL** — ambiguous call; no wrong edge is better than a false-positive edge

This cascade prioritizes precision over recall — correct graph traversal is more valuable than a complete graph with incorrect edges.

### Vector Store Backends

**Module:** `src/trelix/store/vector.py`, `vector_qdrant.py`, `vector_lance.py`

```python
class BaseVectorStore(ABC):
    @abstractmethod
    def upsert_batch(self, chunk_ids, embeddings, symbol_ids, file_ids) -> None
    @abstractmethod
    def search(self, query_embedding, k) -> list[tuple[int, float]]
    @abstractmethod
    def delete_batch(self, chunk_ids) -> None
    @abstractmethod
    def count(self) -> int
    def upsert_file_summary_embedding(self, file_id, embedding) -> None
    def search_file_summaries(self, query_embedding, k) -> list[tuple[int, float]]
    def upsert_sub_chunk_embedding(self, sub_chunk_id, embedding) -> None
    def search_sub_chunks(self, query_embedding, k) -> list[tuple[int, float]]
```

**`SQLiteVecStore`** (default):
- `vec0` virtual table with optional HNSW index (`TRELIX_STORE_HNSW=true`, default)
- HNSW params: M=16, ef_construction=200, ef_search=50
- Separate virtual tables for file summaries and sub-chunks
- Fallback: flat scan when sqlite-vec < 0.1.6 (no HNSW support)

**`QdrantVectorStore`**: cloud-ready, TRELIX_STORE_BACKEND=qdrant, supports named collections, IVF+HNSW, metadata filtering.

**`LanceVectorStore`**: LanceDB, ARM-native HNSW, 3–5× faster insert at 100k+ chunks, memory-mapped storage, Apache Arrow format.

### `DimensionGuard`

**Module:** `src/trelix/store/dimension_guard.py`

```python
class DimensionMismatchError(Exception):
    def __init__(self, stored: int, current: int, provider: str)
    # Message includes: stored dim, current dim, exact migration command

class DimensionGuard:
    @staticmethod
    def check(db, current_dimension, provider) -> None
    # Raises DimensionMismatchError if stored != current
    # No-op if: no dimension stored yet (first index run), or any DB error

    @staticmethod
    def record(db, dimension) -> None
    # Called after successful embed phase; stores in index_metadata

    @staticmethod
    def reset(db) -> None
    # Clears stored dimension; called by trelix migrate-vectors --reset
```

Called in both `Indexer.__init__` and `Retriever.__init__` — catches provider switches before they produce silent wrong-results.

### Parallel BM25 reads (opt-in)

**Module:** `src/trelix/store/read_pool.py`

```python
class ReadOnlyConnectionPool:
    def __init__(self, db_path: Path, pool_size: int = 4) -> None
    # Opens `pool_size` separate mode=ro SQLite connections, PRAGMA query_only=ON,
    # each check_same_thread=False, held in a queue.Queue

    def acquire(self) -> Iterator[sqlite3.Connection]  # @contextmanager, blocks until a conn is free
    def close_all(self) -> None
```

Wired via `Database.enable_bm25_read_pool(pool_size)` — no-op if `pool_size <= 0`. `Retriever.__init__` calls it automatically when `config.store.bm25_read_pool_size > 0`.

- **Disabled (default, `bm25_read_pool_size=0`)**: `bm25_search()` uses the single shared writer connection, guarded by `Database._conn_lock`.
- **Enabled**: `bm25_search()` draws a dedicated read-only connection from the pool instead, allowing true parallel FTS5 reads across the sub-query `ThreadPoolExecutor` legs.

Env var: `TRELIX_STORE_BM25_READ_POOL_SIZE` (default `0`, disabled).

---

## 5. Indexing Pipeline

**Module:** `src/trelix/indexing/indexer.py`

### `Indexer` Construction

```python
class Indexer:
    def __init__(self, config: IndexConfig, quiet: bool = False,
                 progress_callback: Callable | None = None) -> None
```

- Creates `Database`, `BaseEmbedder`, `BaseVectorStore`
- Runs `DimensionGuard.check()` — raises `DimensionMismatchError` if provider changed
- Builds `Chunker` or `ContextualChunker` (lazy LLM client, falls back to plain `Chunker` on API failure)
- Builds optional `FileSummarizer` (only when `file_summaries_enabled=True`)

Phase weight constants used for progress reporting:
- Discovery: 0–5%, Parse: 5–30%, Insert/Chunk: 30–50%, Embed: 50–95%, Resolve: 95–100%

`_FULL_RESOLVE_THRESHOLD = 5` — skip expensive `O(N)` cross-file resolve passes for small batches (e.g. single-file watch events).

### `index() → dict[str, Any]`

Returns stats: `files_found`, `files_indexed`, `files_skipped`, `symbols_extracted`, `chunks_total`, `chunks_embedded`, `errors`, `elapsed_seconds`.

**Phase 0 — Discovery:**
```python
files: list[IndexedFile] = FileWalker(config.walker).walk(repo_path)
```
`FileWalker` respects `.gitignore` via `pathspec`, applies walker config filters.

**Incremental filter:**
```python
to_index = [f for f in files if db.get_file_hash(f.rel_path) != f.hash]
```
Files with unchanged SHA-256 are skipped entirely — zero embedding cost.

**Phase 1 — Parallel Parse:**
```python
with ThreadPoolExecutor(max_workers=config.parse_workers) as pool:
    parsed_files = list(pool.map(_parse_one, to_index))
```
`_parse_one(file)` does: `get_parser(language)` → `Path.read_text()` → `parser.parse(source, file_id=0)`. Placeholder `file_id=0` is used because the real DB id is not yet known. No DB access in Phase 1 — pure CPU (tree-sitter + file I/O, GIL-releasing).

**Phase 2 — Sequential DB Write + Chunk:**

Per file (`_insert_one(parsed_file, stats)`):
1. `db.upsert_file()` → real `file_id`, replaces all placeholder `file_id=0` references
2. Stale vector cleanup: `vector_store.delete_batch(old_chunk_ids)` + `db.delete_file_symbols(file_id)`
3. Insert symbols — **parent_id remapping**: `local_index → real_db_id` via `local_to_db: dict` (methods reference their class by local index before insertion)
4. `db.transaction()` context wraps the entire symbol batch insert
5. `_store_call_edges()` and `_store_type_edges()` (best-effort intra-file resolution)
6. Optional `DataFlowExtractor` (def-use chains, off by default)
7. `chunker.build_chunks(symbols, imports, file_rel_path, language, parent_symbols)`
8. Persist `context_summary` to DB if `ContextualChunker` generated it

Optional Phase 2.5 (`file_summaries_enabled`):
- `FileSummarizer.summarize(file)` → `db.upsert_file_summary()` + `vector_store.upsert_file_summary_embedding()`

Optional Phase 2.6 (MGS3, `multi_granularity_enabled`):
- `MultiGranularityChunker.extract_sub_chunks(symbols)` → `db.insert_sub_chunks()` → embed immediately

Returns `list[_PendingChunk]` — chunk IDs known, embeddings missing.

**Phase 3 — Async Concurrent Batch Embed:**
```python
asyncio.run(_batch_embed_and_store_async(pending_chunks, stats))
```
- `_make_token_batches()`: greedy grouping by `token_count ≤ embed_max_tokens_per_batch`. Single oversized chunk gets its own batch.
- `asyncio.Semaphore(4)`: max 4 concurrent `embedder.embed_async()` calls
- `asyncio.gather()` fans out all batches
- `vector_store.upsert_batch()` is synchronous → `loop.run_in_executor(ThreadPoolExecutor(2))` to avoid blocking event loop
- `_AsyncTpmRateLimiter`: asyncio-native token-per-minute rate limiter; `asyncio.Lock` prevents races in the limit check
- `DimensionGuard.record()` called after successful embed phase

Optional sparse step (SPLADE): `SparseEmbedder.embed(texts)` → `SparseStore.upsert_batch()`

**Phase 4 — Cross-File Resolution:**
```python
db.resolve_cross_file_calls()     # 4-priority cascade (see §4)
db.resolve_import_file_ids()      # language-aware path normalization
db.resolve_cross_file_type_edges()
db.resolve_angular_selectors()    # Angular template → component edges
```

Skipped when `files_in_batch < _FULL_RESOLVE_THRESHOLD (5)`.

### `index_file(file_path, files_in_batch=1)` — Hot Path

Used by file watcher for single-file incremental updates.
- Hash check → skip if unchanged
- `_parse_one()` + `_insert_one()` + `_batch_embed_and_store()` (sync variant, not async)
- Skips Phase 4 when `files_in_batch < 5` (default watch event = 1 file)
- Returns `{"status": "ok", "symbols_updated": N, "chunks_updated": N, "ms": N}`

### Language Parser Registry

**Module:** `src/trelix/indexing/parser/registry.py`

```python
_PARSER_REGISTRY: dict[Language, type[BaseParser]] = {
    Language.PYTHON: PythonParser,
    Language.JAVASCRIPT: JavaScriptParser,
    Language.TYPESCRIPT: TypeScriptParser,
    Language.TSX: TypeScriptParser,   # TSX uses TS parser
    Language.GO: GoParser,
    Language.RUST: RustParser,
    Language.JAVA: JavaParser,
    Language.CPP: CppParser,
    Language.C: CParser,
    Language.CSHARP: CSharpParser,
    Language.RAZOR: RazorParser,
    Language.CSHTML: CshtmlParser,    # inherits RazorParser
    Language.CSPROJ: CsprojParser,
    Language.KOTLIN: KotlinParser,
    Language.RUBY: RubyParser,
    Language.MARKDOWN: MarkdownParser,
    Language.JSON: JsonParser,
    Language.YAML: YamlParser,
    Language.TOML: TomlParser,
    Language.HTML: HtmlParser,
    Language.CSS: CssParser,
}
```

Each parser inherits `BaseParser(ABC)` and implements:
```python
def parse(self, source: str, file_id: int) -> ParseResult
# Returns: ParseResult(symbols, call_edges, type_edges, import_edges)
```

Parsers use tree-sitter WASM bindings via the `tree-sitter` Python package. They walk the CST emitting `Symbol`, `CallEdge`, `TypeEdge`, `ImportEdge` objects without regex or line-based heuristics.

### Chunker

**Module:** `src/trelix/indexing/chunker.py`

```python
class Chunker:
    def build_chunks(self, symbols, imports, file_rel_path,
                     language, parent_symbols) -> list[Chunk]
    # One Chunk per Symbol (whole-symbol chunking)
    # Context header prepended: file path, language, up to 8 imports, parent class
    # Body truncated at max_tokens_per_chunk (512 by default)

class ContextualChunker(Chunker):
    # Extends: calls LLM for 2-3 sentence context summary per symbol
    # Summary prepended to chunk_text, stored in symbols.context_summary
    # Falls back to base Chunker on any LLM failure (per-symbol, not all-or-nothing)
```

### Multi-Granularity Sub-Chunk Indexing (MGS3)

**Module:** `src/trelix/indexing/multi_granularity.py`

Enables block-level and statement-level indexing in addition to symbol-level.

```python
class Granularity(StrEnum):
    FUNCTION = "function"
    BLOCK = "block"
    STATEMENT = "statement"

@dataclass
class SubSymbolChunk:
    parent_symbol_id: int
    granularity: Granularity
    chunk_text: str
    line_start: int
    line_end: int
    token_count: int

class MultiGranularityChunker:
    def extract_sub_chunks(self, symbols: list[Symbol]) -> list[SubSymbolChunk]
    # Uses tree-sitter to extract code blocks and individual statements
    # Stored in sub_chunks table, embedded into a separate vector namespace
```

When `sub_chunk_search_enabled=True`, the retriever queries the sub-chunk vector namespace in addition to the symbol-level namespace, providing finer-grained results for long function bodies.

---

## 6. Embedder Layer

**Module:** `src/trelix/embedder/base.py`, `bge_code.py`, `nomic_code.py`, `sparse.py`

### `BaseEmbedder` (ABC)

```python
class BaseEmbedder(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]: ...
    
    @abstractmethod
    def embed_query(self, text: str) -> list[float]: ...
    
    async def embed_async(self, texts: list[str]) -> list[list[float]]:
        # Default: runs self.embed in thread executor (sync → async bridge)
        # Concrete classes override with true async (OpenAI, Azure)
    
    @property
    @abstractmethod
    def dimension(self) -> int: ...
```

Module-level shared executor: `_SYNC_EXECUTOR = ThreadPoolExecutor(max_workers=4)` — used by all sync embedders needing async bridging.

### Concrete Embedders

**`OpenAIEmbedder`** / **`AzureOpenAIEmbedder`**
- True async via `AsyncOpenAI` / `AsyncAzureOpenAI` clients (lazy-created)
- Batch splits by `batch_size` (default 64)
- Dimension from config (default 3072)

**`LocalEmbedder`** (sentence-transformers)
- `SentenceTransformer(config.local_model).encode(texts, batch_size, convert_to_numpy=True)`
- Async via executor (CPU-bound, GIL-releasing for numpy)
- First run downloads ~80MB model to `~/.cache/`

**`VoyageEmbedder`**
- Asymmetric: documents → `input_type="document"`, queries → `input_type="query"`
- Batch limit: 128 per API call (`_BATCH_LIMIT = 128`)
- Matryoshka support via `voyage_output_dimensions`

**`BedrockTitanEmbedder`**
- `_BATCH_SIZE = 1` — Titan API accepts one text per invoke
- Async: `asyncio.gather` over parallel executor calls (not sequential)
- `_decode_credential(value)` — transparent base64 decode for secrets passed base64-encoded

**`BedrockCohereEmbedder`**
- Batch limit 96; pre-truncates at 2048 chars to avoid Bedrock `ValidationException`
- `_BATCH_LIMIT = 96`, `_MAX_CHARS = 2048`

**`BGECodeEmbedder`** — BAAI/bge-code-v1 (768 dims, CoIR SOTA 2025)

**`NomicCodeEmbedder`** — nomic-ai/CodeRankEmbed (768 dims)

**`LocalCodeEmbedder`** — Salesforce/SFR-Embedding-Code-2B_R (4096 dims, ~8GB RAM)

### `CachingEmbedder`

**Module:** `src/trelix/embedder/cache.py`

```python
class CachingEmbedder(BaseEmbedder):
    def __init__(self, embedder: BaseEmbedder, max_size: int = 256)
    # LRU cache over embed_query() only (not embed())
    # Cache key: str(text) — no normalization (query is always a string)
    # embed() bypass: batch indexing should never use cached results
```

Wrapped around the underlying embedder in `Retriever.__init__` when `query_cache_size > 0`.

### `SparseEmbedder` (SPLADE)

**Module:** `src/trelix/embedder/sparse.py`

```python
class SparseEmbedder:
    def __init__(self, model_name="naver-splab/splade-code-distil",
                 top_k=128, batch_size=16)
    def embed(self, texts: list[str]) -> list[dict[int, float]]
    def embed_query(self, text: str) -> dict[int, float]
    # Returns sparse dict: {token_id: weight}
    # SPLADE aggregation: log(1 + ReLU(logits)).max(dim=1)
    # Keeps only top_k highest-weight tokens
```

`_TORCH_AVAILABLE` module flag — returns empty `[{}]` gracefully when torch/transformers absent.

Research basis: SPLADE-Code (naver-splab) — learned sparse retrieval via transformer vocabulary space, trained specifically on code data.

---

## 7. Retrieval Pipeline

**Module:** `src/trelix/retrieval/retriever.py`

### `Retriever` Construction

```python
class Retriever:
    def __init__(self, config: IndexConfig) -> None:
        db = Database(config.db_path_absolute)
        embedder = CachingEmbedder(make_embedder(config.embedder), query_cache_size)
        vector_store = make_vector_store(config, embedder.dimension)
        _planner = CachingPlanner(QueryPlanner(config.embedder), plan_cache_size)
        DimensionGuard.check(db, embedder.dimension, config.embedder.provider)
        _debug_dir = <repo>/.trelix/debug/    # per-query JSON traces
        _sparse_embedder = None               # memoized slot (SPLADE model load ~seconds)
```

### `retrieve(query, plan=None) → RetrievedContext`

Full pipeline (11 stages):

```
1.  Timer start
2.  QueryPlanner.plan(query) → QueryPlan  [or use provided plan]
3.  Intent routing (_execute_plan)
4.  Sub-query parallel execution [ThreadPoolExecutor when execution_mode="parallel"]
5.  Optional multi-query expansion [MultiQueryExpander, N LLM variants, parallel]
6.  Per-leg aggregation [vector, bm25, grep, sparse across all sub-queries]
7.  Optional file-summary leg [5th leg, negative-ID Chunk sentinels]
8.  Optional sub-chunk leg [6th leg, MGS3]
9.  RRF fusion [reciprocal_rank_fusion, k=60, file-type weights]
10. Graph expansion [call-graph + import-graph + type-edges on top fused results]
11. Optional CodeGraph BFS [4th leg when graph_search_enabled]
12. Dedup [by chunk.symbol_id, keep highest score]
13. Rerank [cross-encoder/Cohere/PLAID, controlled by strategy.skip_reranker]
14. Optional PageRank boost [×1.3 for top-200 central symbols, re-sort]
15. ContextAssembler [greedy token packing within 12k budget]
16. Timer stop, telemetry write, debug trace flush
```

### Intent Routing

```
RoutingTier.TIER_1_DIRECT   → _retrieve_project_overview (skip all legs)
IntentType.FILE_OVERVIEW    → _retrieve_file_overview
IntentType.PROJECT_OVERVIEW → _retrieve_project_overview
IntentType.CONFIG_LOOKUP    → _retrieve_config
All others                  → _retrieve_standard (full 11-stage pipeline)
```

### `_run_subquery_legs(sq, strategy)`

```python
# Vector leg
embed_text = sq.hyde_snippet or sq.semantic_query
if hyde_fallback_enabled and not sq.hyde_snippet:
    embed_text = HyDEExpander(llm).expand(sq.semantic_query)  # LLM call
embedding = embedder.embed_query(embed_text)
out["vector"] = vector_store.search(embedding, top_k_vector)

# BM25 leg
bm25_query = " ".join(sq.bm25_tokens) or sq.semantic_query
out["bm25"] = db.bm25_search(bm25_query, top_k_bm25)

# Grep leg
hints = sq.grep_hints or [sq.semantic_query]
out["grep"] = [grep_search(db, hint, top_k_grep) for hint in hints]

# Sparse leg (memoized SPLADE)
if sparse_enabled:
    query_sparse = self._get_sparse_embedder().embed_query(sq.semantic_query)
    out["sparse"] = sparse_search(sparse_store, db, query_sparse, k=top_k_sparse)
```

### BLAST_RADIUS Special Handling

```python
# Seed from import path patterns (@alias/... patterns in grep_hints)
if strategy.import_direction == "reverse":
    seed_from_import_paths(db, patterns=sq.grep_hints, max_extra=...)
# Then: expand_with_imports(db, top, direction="reverse")
# reverse = find files that IMPORT the target (dependents)
# assembly_mode = "breadth_first" = 1-2 symbols from many files (blast radius spread)
```

### Direct-Path Retrieval Methods

```python
def _retrieve_file_overview(plan) -> RetrievedContext:
    # db.find_file_by_path_fragment(hint) → db.get_all_symbols_for_file()
    # Structural order (by line number), score = 1.0 - rank × 0.001
    # Skips reranker; falls back to _retrieve_standard if no file match

def _retrieve_project_overview(plan) -> RetrievedContext:
    # db.get_module_and_readme_symbols(limit=40)
    # Priority: README → markdown → manifests → module-level symbols
    # db.get_top_level_directories() for monorepo heuristic

def _retrieve_config(plan) -> RetrievedContext:
    # File-direct for hints matching config extensions
    # Falls back to _retrieve_standard
```

### Public Graph Query API

```python
def get_callers(symbol_name: str) -> list[SearchResult]
# Matches all symbols by name, union of db.get_callers(sym.id)
# Deduped by symbol_id, sorted by file + line_start

def get_callees(symbol_name: str) -> list[SearchResult]
# Symmetric

def get_importers(module_path: str) -> list[SearchResult]
# Suffix-match on files.rel_path; returns first symbol per importing file
```

### Debug Tracing

Every `retrieve()` call writes:
```
<repo>/.trelix/debug/<ISO-timestamp>_<query-slug>.json
```

Sections: `planner` (intent, routing tier, sub-queries), `retrieval_legs` (per-leg counts and top results), `post_fusion` (RRF scores), `expansion` (graph expansion results), `post_rerank` (final scores), `assembly` (token budget usage).

Thread-local storage (`_trace_local = threading.local()`) ensures parallel eval workers never cross-contaminate traces.

---

## 8. Query Planner

**Modules:** `src/trelix/retrieval/planner/models.py`, `planner/agent.py`

### `RoutingTier`

```python
class RoutingTier(int, Enum):
    TIER_1_DIRECT = 1   # trivial factual: "what is X?", "list all Y"
    TIER_2_SINGLE = 2   # default: ~90% of queries
    TIER_3_MULTI = 3    # complex: "from X to Y", "walk me through", len>80 with 2+ "and"
```

### `IntentType` and `INTENT_STRATEGIES`

The complete `INTENT_STRATEGIES` dict (single source of truth):

| Intent | Legs | expand_depth | import_depth | import_direction | import_max_extra | rerank_top_n | assembly_mode |
|--------|------|:---:|:---:|:---:|:---:|:---:|:---:|
| SYMBOL_LOOKUP | grep+bm25+vector | 1 | 1 | both | 3 | 20 | greedy |
| FILE_OVERVIEW | file_direct | 0 | 0 | both | 0 | 20 | greedy |
| FEATURE_FLOW | vector+bm25 | 2 | 2 | both | 15 | 30 | greedy |
| PROJECT_OVERVIEW | file_direct | 0 | 0 | both | 0 | 20 | greedy |
| COMPARISON | vector+bm25+grep | 1 | 1 | both | 8 | 35 | greedy |
| CONFIG_LOOKUP | file_direct+grep | 0 | 0 | both | 0 | 20 | greedy |
| DEPENDENCY_MAP | vector+bm25 | 1 | 2 | forward | 20 | 30 | breadth_first |
| BLAST_RADIUS | grep+vector+bm25 | 0 | 1 | reverse | 30 | 40 | breadth_first |

Design note: SYMBOL_LOOKUP leads with `grep` (exact match), not vector. BLAST_RADIUS leads with `grep` (exact symbol name) then uses `reverse` import walk + `breadth_first` assembly to surface 1-2 symbols from many affected files (blast radius spread).

### `SubQuery`

```python
@dataclass
class SubQuery:
    semantic_query: str     # rephrased as technical description (NOT a question)
    hyde_snippet: str       # hypothetical code snippet for vector embedding
    bm25_tokens: list[str]  # clean keywords, no stop words
    grep_hints: list[str]   # exact symbol names, filename fragments
    file_hints: list[str]   # filename fragments to bias retrieval
    depends_on: list[int]   # 0-based indices of prerequisite sub-queries
```

### `AdaptiveRouter`

```python
class AdaptiveRouter:
    def route(self, query: str, project_context: str = "") -> QueryPlan
    # Never raises: all exceptions → default_plan()
    
    # Tier 1 patterns (regex): "what is X?", "list all Y", "define Z"
    # Tier 3 phrases: "from X to Y", "end-to-end", "step by step",
    #                 "walk me through", "full flow"
    # Tier 3 also: len(query) > 80 AND count(" and ") >= 2
    # Tier 2: everything else
    
    def _tier1_plan(self, query) -> QueryPlan
    # PROJECT_OVERVIEW, routing_tier=TIER_1_DIRECT
    # Retriever skips all legs — directly calls _retrieve_project_overview
    
    def _multi_step_plan(self, query, context) -> QueryPlan
    # LLM decomposes into 2-3 sub-questions
    # Each sub-question → SubQuery; FEATURE_FLOW intent; execution_mode="parallel"
    # Falls back to single-step on any parse error; clamps to 2-3 sub-questions
```

### `QueryPlanner`

```python
class QueryPlanner:
    def __init__(self, config: EmbedderConfig) -> None
    # Builds TrelixChatClient; _client=None means no LLM available → use default_plan()
    
    def plan(self, query: str, project_context: str = "") -> QueryPlan
    # Delegates to AdaptiveRouter.route()
    
    def _plan_direct(self, query, context) -> QueryPlan
    # LLM tool call with PLANNER_TOOL_SCHEMA (forces structured output)
    # model: gpt-4o-mini (OpenAI) or gpt-4o (Azure)
    # Falls back to default_plan() on any exception
    
    def default_plan(query) -> QueryPlan
    # FEATURE_FLOW intent, execution_mode="parallel", raw query as semantic_query
    # Used when: no API key, LLM call fails, or empty query
```

### `CachingPlanner`

**Module:** `src/trelix/retrieval/plan_cache.py`

```python
class CachingPlanner:
    def __init__(self, planner: QueryPlanner, max_size: int = 128)
    def plan(self, query: str, project_context: str = "") -> QueryPlan
    # LRU cache keyed by (query, project_context)
    # Avoids repeated gpt-4o-mini calls for the same query in eval loops
```

---

## 9. Fusion and Ranking

### Reciprocal Rank Fusion

**Module:** `src/trelix/retrieval/fusion.py`

```python
def reciprocal_rank_fusion(
    ranked_lists: list[list[SearchResult]],
    k: int = 60,
    weights: dict[str, float] | None = None,
) -> list[SearchResult]
```

**Algorithm** (Cormack et al. 2009):
```python
rrf_score[chunk_id] += 1.0 / (k + rank)  # for each (list, rank) pair
```

**Key design decisions:**
- Dedup key: `result.chunk.symbol_id` (not `chunk.id`) — symbol-level dedup even when different legs produce different chunk IDs for the same symbol
- `best_result` keeps first-seen result: `source` reflects which leg first found the symbol. RRF scores are NOT comparable to raw cosine/BM25 scores, so first-seen source is correct.
- File-type weighting: post-RRF multiplicative step on `rrf_score` — missing weights fall back to 1.0
- Accepts empty lists safely (no-op)
- Production call in `_retrieve_standard` passes up to 6 lists: vector, bm25, grep, summary, sub_chunk, sparse

**File-type weights** (applied post-RRF):
```
source code (Python, Go, Rust, etc.) = 1.0
HTML, CSS                            = 0.4
JSON, YAML, TOML                     = 0.5
Markdown                             = 0.3
Unknown                              = 0.8
```

### Graph Expansion

**Module:** `src/trelix/retrieval/graph.py`

```python
def expand_with_call_graph(db, results, depth=1, max_extra=10) -> list[SearchResult]
# Follows CALLS edges in both directions (callers + callees) for top fused symbols
# depth=1: 1 hop, depth=2: 2 hops (expensive — avoid for TIER_2_SINGLE)

def expand_with_imports(db, results, max_extra=8, depth=1, direction="both") -> list[SearchResult]
# direction="both"   → callers of and imports by the symbol
# direction="forward" → files this symbol imports (dependency analysis)
# direction="reverse" → files that import this symbol (blast radius / dependents)

def expand_with_type_edges(db, results, max_extra=15) -> list[SearchResult]
# Follows EXTENDS/IMPLEMENTS/TRAIT_IMPL edges
# Fixed max_extra=15 (type hierarchies tend to be shallow)

def seed_from_import_paths(db, patterns, max_extra) -> list[SearchResult]
# BLAST_RADIUS specific: @-prefixed grep_hints are treated as import path patterns
# Used to surface files importing via barrel exports / path aliases
```

### Rerankers

**Module:** `src/trelix/retrieval/reranker.py`, `reranker_plaid.py`

```python
def rerank(query, candidates, config, top_n) -> list[SearchResult]
# Routes to:
# - CrossEncoderReranker when rerank_provider="cross_encoder" (sentence-transformers)
# - CohereReranker when rerank_provider="cohere" (requires COHERE_API_KEY)
# - PlaidReranker when rerank_provider="plaid" (requires trelix[plaid])

class PlaidReranker:
    # RAGatouille ColBERT late-interaction reranker
    # 7–45× faster than exact ColBERT with equivalent quality
    # Lazy-loads colbert-ir/colbertv2.0 on first call
```

### `ContextAssembler`

**Module:** `src/trelix/retrieval/assembler.py`

```python
class ContextAssembler:
    def __init__(self, token_budget: int = 12_000) -> None
    def assemble(self, query, results, intent, assembly_mode) -> RetrievedContext
```

**Greedy mode** (default): packs results in score order until token budget exhausted. `chunk.token_count` (pre-computed) for fast budget accounting.

**Breadth-first mode** (BLAST_RADIUS, DEPENDENCY_MAP): round-robin across files — 1-2 symbols per file so many files are represented rather than many symbols from one file.

---

## 10. Synthesis and Agent Loop

### Synthesizer

**Module:** `src/trelix/retrieval/synthesizer.py`

```python
class Synthesizer:
    def synthesize(self, ctx: RetrievedContext, config: EmbedderConfig) -> str
    def stream(self, ctx: RetrievedContext, config: RetrievalConfig) -> Iterator[str]
    # stream() yields tokens for SSE in the REST API
```

When `graph_rag_enabled=True` and `ctx.total_tokens > graph_rag_threshold_tokens` (8000):
```
GraphRAGSynthesizer.synthesize(ctx)
  ├─ Map phase: LLM summarizes each result chunk individually
  └─ Reduce phase: LLM synthesizes all summaries into final answer
```

GraphRAG prevents exceeding the LLM context window on large codebases.

### FLARE Loop

**Module:** `src/trelix/retrieval/flare.py`

```python
class FLARELoop:
    def __init__(self, retriever, synthesizer, config: IndexConfig) -> None
    def run(self, query: str) -> str
```

```
1. ctx = retriever.retrieve(query)
2. answer = synthesizer.synthesize(ctx)
3. if not flare_enabled: return answer
4. Repeat up to flare_max_retries - 1 additional times:
   a. if not _contains_uncertainty(answer): return answer
   b. Enrich query with uncertainty context
   c. ctx = retriever.retrieve(enriched_query)
   d. answer = synthesizer.synthesize(ctx)
5. return answer
```

`_contains_uncertainty()`: regex over phrases like "I don't know", "not sure", "could not find", "unable to determine".

`flare_max_retries` semantics: total budget including initial synthesis (not just retries). `flare_max_retries=1` means 0 extra iterations (initial only). `ge=1, le=3`.

### Agent Loop

**Module:** `src/trelix/agent/loop.py`

```python
class AgentLoop:
    def run(self, query: str) -> str  # never raises
```

ReAct (Reason+Act) multi-turn orchestrator with three available actions:

| ActionType | Tool | Arguments | Returns |
|---|---|---|---|
| RETRIEVE | retrieve | `{"query": str}` | Top 8 results, body[:300] each |
| GREP | grep | `{"pattern": str, "max_results": 10}` | file:line — name list |
| GET_SYMBOL | get_symbol | `{"qualified_name": str}` | Full body in code fence |
| DONE | done | `{"answer": str}` | Final answer |

System prompt mandate: "Never call done until you've done at least one retrieval."

**`TurnHistory` + `HistoryCompressor`:**
```python
class TurnHistory:
    def add(self, turn: Turn) -> None
    def get_messages(self) -> list[dict]  # role/content LLM message format
    
class HistoryCompressor:
    def __init__(self, token_budget: int = 6_000)
    def compress(self, history: TurnHistory) -> list[dict]
    # Keeps last N turns within token budget; LLM-summarizes older turns
```

`_fallback_answer()`: returns first 3 successful observation contents when max_turns reached.

---

## 11. Graph Layer

### `CodeGraph`

**Module:** `src/trelix/graph/code_graph.py`

```python
class CodeGraph:
    def __init__(self, db: Database) -> None  # builds immediately
    @property def nx(self) -> nx.MultiDiGraph
    @property def node_count(self) -> int
    @property def edge_count(self) -> int
    def neighbors(self, symbol_id) -> list[int]        # callers + callees, deduped
    def shortest_path(self, src, dst) -> list[int] | None  # undirected for traversal
    def subgraph(self, symbol_ids) -> nx.MultiDiGraph
    def get_node_attrs(self, symbol_id) -> dict[str, Any]
```

**Construction sequence:**
1. Nodes: all symbols from `db.iter_all_symbols_with_files()` — attrs: type, name, qualified_name, kind, file, language, community=None
2. CALLS edges: resolved call edges only (callee_id is not None)
3. IMPORTS edges: file nodes added on-demand if not already symbol nodes
4. TYPE edges: EXTENDS, IMPLEMENTS, TRAIT_IMPL, EMBEDDED, ANGULAR_SELECTOR

**Design:** `MultiDiGraph` supports multiple parallel edges between the same node pair (e.g. a class both implements and extends from the same base) and directed edges preserve call/import directionality.

### Community Detection

**Module:** `src/trelix/graph/community.py`

```python
def detect_communities(cg: CodeGraph, algorithm: str = "louvain") -> dict[int, int]
def compute_pagerank(cg: CodeGraph, alpha: float = 0.85) -> dict[int, float]
def get_community_summary(cg: CodeGraph) -> list[dict[str, Any]]
```

Algorithms:
- **louvain** (default): `nx.community.louvain_communities(G_connected, seed=42)`, O(n log n), deterministic via seed
- **girvan_newman**: betweenness-based, O(n³), max 3 iterations, for small graphs
- **label_prop**: very fast, approximate, for >10k nodes

PageRank: `nx.pagerank(g, alpha=0.85, max_iter=100)`. Retries with `max_iter=500, tol=1e-4` on convergence failure. Normalized to [0,1].

### `GraphBuilder`

**Module:** `src/trelix/graph/builder.py`

```python
@dataclass
class GraphBuildResult:
    code_graph: CodeGraph
    community_count: int
    node_count: int
    edge_count: int
    concept_count: int
    elapsed_seconds: float
    community_summary: list[dict[str, Any]]

class GraphBuilder:
    def build(self, extract_concepts: bool = False) -> GraphBuildResult
```

Build pipeline: CodeGraph → detect_communities → assign_communities → save_metadata → compute_pagerank → optional ConceptExtractor.

### `GraphUpdater`

**Module:** `src/trelix/graph/updater.py`

```python
class GraphUpdater:
    def update_file(self, rel_path: str) -> None
```

Called by `FileWatcher` after a file is re-indexed. Full graph rebuild (not incremental). Rationale: simpler than partial updates and avoids stale-edge bugs when call targets change. Target rebuild time: ~50ms for typical repos over SQLite reads.

### Semantic Concept Extraction

**Module:** `src/trelix/graph/concepts.py`

```python
@dataclass
class SemanticConcept:
    symbol_id: int
    concept: str       # LLM-extracted architectural concept label
    confidence: float
    
class ConceptExtractor:
    def extract(self, symbols: list[Symbol], limit: int = 200) -> list[SemanticConcept]
    # Batches symbols in groups of 20; LLM call per batch
    # Extracts high-level concepts: "Authentication", "Database connection pooling", etc.
```

---

## 12. Analysis Layer

### `DataFlowExtractor`

**Module:** `src/trelix/analysis/defuse.py`

Intra-procedural def-use chains extracted via tree-sitter AST walk.

```python
@dataclass
class DefUseEdge:
    symbol_id: int
    var_name: str
    def_line: int
    use_line: int
    edge_type: str  # "def" | "use"

class DataFlowExtractor:
    def extract(self, symbol: Symbol) -> list[DefUseEdge]  # never raises
```

Assignment nodes: `"assignment"`, `"augmented_assignment"`, `"named_expression"` (walrus :=), `"for_statement"`, `"with_statement"`, `"import_statement"`, `"import_from_statement"`, `"parameters"`.

Fallback to regex when tree-sitter unavailable: `r"\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*="` (excludes def/class/return/import keywords).

### `TaintAnalyzer`

**Module:** `src/trelix/analysis/taint.py`

```python
@dataclass
class TaintFlow:
    source_file: str
    source_line: int
    sink_file: str
    sink_line: int
    rule_id: str
    severity: str  # "ERROR" | "WARNING" | "INFO"

class TaintAnalyzer:
    def __init__(self, repo_path: str, tier: str = "default")
    def run(self, rules_path: str | None = None) -> list[TaintFlow]  # [] on failure
```

Tiers:
- `"default"` — intra-procedural, fast
- `"intrafile"` — `--pro-intrafile`, cross-function within file
- `"interfile"` — `--pro --interfile`, full inter-procedural

CLI: `semgrep --json --no-rewrite-rule-ids [flags] --config <rules> <repo>`. Timeout: 120s. Returns `[]` when semgrep absent.

---

## 13. Federation Layer (v2.4.0)

### `RepoRegistry`

**Module:** `src/trelix/federation/registry.py`

```python
@dataclass
class RepoEntry:
    alias: str
    path: str
    weight: float = 1.0   # RRF score multiplier

class RepoRegistry:
    @classmethod
    def load(cls, config_path: str | None = None) -> RepoRegistry
    # Default: ~/.config/trelix/repos.json
    # Returns empty registry (not error) if file missing/invalid
    
    def add(self, alias, path, weight=1.0) -> None  # ValueError on duplicate alias
    def remove(self, alias) -> None                  # no-op if not found
    def list(self) -> list[RepoEntry]
    def save(self) -> None                           # writes JSON
```

### `FederatedRetriever`

**Module:** `src/trelix/federation/retriever.py`

```python
class FederatedRetriever:
    def __init__(self, registry: RepoRegistry,
                 max_workers: int = 4,
                 cache_ttl: float = 120.0) -> None
    
    def retrieve(self, query: str, k: int = 10) -> list[SearchResult]
    def _query_repos(self, query, k) -> list[SearchResult]  # fan-out, no cache
    def _make_cache_key(self, query, k) -> str              # SHA-256
    def _get_cached(self, key) -> list[SearchResult] | None
    def _set_cached(self, key, results) -> None
    def cache_stats(self) -> dict[str, int]   # {hits, misses, size}
    def clear_cache(self) -> None
```

**`_query_repos()` parallel fan-out:**
```python
with ThreadPoolExecutor(max_workers=min(max_workers, len(entries))) as pool:
    futures = {pool.submit(_query_one, entry.path): entry for entry in entries}
    for future in as_completed(futures):
        results = future.result(timeout=30)
        per_repo_results.append(results)
```

**Merging:** `reciprocal_rank_fusion(per_repo_results)` → dedup by `f"{r.file.rel_path}:{r.chunk.symbol_id}"`.

**TTL Cache (v2.4.0):**
- Cache key: `SHA-256(f"{query}|{sorted_repo_paths}|{k}")`
- Thread-safe: `threading.Lock` guards `_cache`, `_hits`, `_misses`
- TTL check: `time.monotonic() > expiry` (not wall clock — monotonic avoids daylight saving jumps)
- `cache_ttl=0` disables entirely (no cache dict population)
- Expected hit rate: ~90% for typical debugging-session query patterns (repeated same-question queries)

---

## 14. Review Layer (v2.4.0)

### `DiffParser`

**Module:** `src/trelix/review/diff_parser.py`

```python
@dataclass
class DiffHunk:
    file_path: str
    old_start: int; new_start: int      # from @@ header, 1-indexed
    old_lines: int; new_lines: int
    added: list[str]                     # lines starting with +
    removed: list[str]                   # lines starting with -
    context: list[str]                   # unchanged surrounding lines
    
    def to_search_query(self) -> str     # extracts identifiers for retrieval
    
class DiffParser:
    def parse(self, diff_text: str) -> list[DiffHunk]     # instance method
    def from_git(self, repo_path, base="HEAD~1", head="HEAD") -> list[DiffHunk]
    # Runs: git diff --unified=3; returns [] on any failure
```

### `DiffReviewer`

**Module:** `src/trelix/review/reviewer.py`

```python
class ReviewComment:
    file_path: str; line_start: int; line_end: int
    severity: str   # "ERROR" | "WARN" | "INFO"
    comment: str

class DiffReviewer:
    def __init__(self, config: IndexConfig) -> None
    def review(self, hunks: list[DiffHunk] | None = None,
               diff_text: str | None = None) -> list[ReviewComment]
    # Returns [] on any failure (never raises)
    # When diff_text provided: DiffParser().parse(diff_text) → hunks
    # Per-hunk: retrieve(hunk.to_search_query()) → LLM generate ReviewComment[]
```

### GitHub PR Integration

**Module:** `src/trelix/review/github.py`

```python
class GitHubAPIError(Exception): ...

@dataclass(frozen=True)
class PRFile:
    filename: str
    status: str    # all 7 GitHub statuses: added/removed/modified/renamed/copied/changed/unchanged
    additions: int; deletions: int
    patch: str | None   # None for binary files and oversized diffs
    previous_filename: str | None  # for renamed/copied files

@dataclass
class ReviewComment:
    path: str; line: int; body: str
    side: str = "RIGHT"  # RIGHT=addition, LEFT=deletion context

class GitHubPRClient:
    def __init__(self, token: str, base_url: str = "https://api.github.com") -> None
    
    def get_pr_files(self, owner, repo, pr_number) -> list[PRFile]
    # Paginates: GET /repos/{owner}/{repo}/pulls/{number}/files, 100/page
    # Warns when >=3000 files (GitHub silent truncation)
    
    def post_review(self, owner, repo, pr_number, commit_sha, body,
                    comments, event="COMMENT") -> dict
    # Single batched API call (80 write requests/min rate limit safe)
    # POST /repos/{owner}/{repo}/pulls/{number}/reviews
    
    def get_pr_head_sha(self, owner, repo, pr_number) -> str
    # Guards: isinstance(data, dict) → GitHubAPIError if list returned

def parse_pr_ref(pr_ref: str) -> tuple[str, str, int]
# Parses "owner/repo#number" → (owner, repo, number)
# ValueError with usage hint on malformed input
```

---

## 15. MCP Server

**Package:** `packages/trelix-mcp/src/trelix_mcp/server.py`

Transport: `stdio` (subprocess model — IDE spawns `trelix-mcp` as child process).

**Critical:** All logging directed to `stderr`. `stdout` is the exclusive MCP JSON protocol pipe.

```python
logging.basicConfig(stream=sys.stderr)  # FIRST LINE — before any imports
mcp = FastMCP("trelix")
```

### Resources (application-controlled, URI-addressable)

| URI | Handler | Returns |
|-----|---------|---------|
| `trelix://index/stats` | `resource_index_stats` | JSON usage hint |
| `trelix://repo/{repo_path}/manifest` | `resource_repo_manifest` | `{file_count, files[]}` |
| `trelix://repo/{repo_path}/symbols/{qualified_name}` | `resource_symbol_source` | `{qualified_name, kind, signature, body}` |

### Prompts (user-controlled LLM templates)

| Name | Parameters | Purpose |
|------|-----------|---------|
| `trelix-search` | query, repo_path | Semantic search prompt |
| `trelix-explain` | qualified_name, repo_path | Symbol explanation |
| `trelix-blast-radius` | symbol_name, repo_path | Impact analysis before refactoring |

### Tools (callable by MCP clients)

```python
def search_code(query: str, repo_path: str, k: int = 10, cursor: int = 0) -> dict:
    # v2.4.0: returns pagination envelope
    # {"results": [...], "next_cursor": int|null, "total_available": int}
    # cursor=0 default preserves backward compatibility

def index_codebase(repo_path: str, provider: str = "local",
                   ctx: Context | None = None) -> dict:
    # Emits ctx.report_progress(current, total) via asyncio.get_running_loop().create_task()
    # Best-effort: silently skips when no running event loop

def get_symbol(qualified_name: str, repo_path: str) -> dict | None:
    # Exact qualified_name match; fallback to bare name if not found

def blast_radius(symbol_name: str, repo_path: str) -> list[dict]:
    # Query: f"blast radius dependencies of {symbol_name}"
    # Deduplicates by file.rel_path

def build_knowledge_graph(repo_path: str, extract_concepts: bool = False) -> dict:
    # Full GraphBuilder.build() pipeline

def graph_search_mcp(query: str, repo_path: str, k: int = 10) -> list[dict]:
    # Two-phase: standard retrieval (top-5 seed) + CodeGraph BFS depth=2
```

---

## 16. REST API

**Module:** `src/trelix/api/app.py`

```python
def create_app() -> FastAPI:
    app = FastAPI(title="trelix API", version=__version__)
    # Routes registered here
    return app
```

Design note: `IndexConfig` and `Retriever` imported at module scope (not inside `create_app()`) specifically to enable `patch("trelix.api.app.Retriever")` in tests before the app is created.

### Routes

| Method | Path | Returns | Notes |
|--------|------|---------|-------|
| GET | `/health` | `{status, version}` | |
| GET | `/search` | `list[result_dict]` | query, repo, k=10 |
| GET | `/ask` | `StreamingResponse` | SSE token stream |
| POST | `/index` | `index_stats_dict` | body: `{repo_path}` |
| GET | `/stats` | `{files, symbols, chunks}` | repo param |
| GET | `/graph` | `GraphBuildResult` | repo param |
| GET | `/graph/communities` | community_summary | repo param |
| GET | `/graph/visualize` | `{path, node_count}` | Pyvis HTML output |
| GET | `/graph/search` | `list[result_dict]` | repo, symbol_id, depth=2 |

**`/ask` SSE generator:**
```python
# Yields: "data: {token}\n\n"
# Terminates: "data: [DONE]\n\n"
# Error: "data: [ERROR: {exc}]\n\n"
```

**Security guards:**
- `/graph/visualize`: output path must start with `<repo_root>/.trelix/` — directory traversal protection
- `/graph/search`: depth clamped to `max(1, min(depth, 10))`

---

## 17. CLI Layer

**Module:** `src/trelix/cli/main.py`

Single entry point: `trelix = "trelix.cli.main:app"` (Typer application).

### Command Summary

| Command | Key flags | Description |
|---------|-----------|-------------|
| `index <repo>` | --provider, --workers, --json | Build search index |
| `search <repo> <query>` | --k, --json, --provider | Hybrid search |
| `ask <repo> <question>` | --provider, --agentic, --json | LLM synthesis |
| `query <repo> <question>` | --k, --json | Search without LLM |
| `call-graph <repo> <symbol>` | --depth, --json | Call/import graph |
| `stats <repo>` | --json | Index statistics |
| `update-index <repo> <file>` | | Re-index single file |
| `migrate-vectors <repo>` | --reset, --to | Vector migration |
| `watch <repo>` | --provider | Single-repo watch |
| `watch-all` | --config | All federated repos (v2.4.0) |
| `serve <repo>` | --port, --host | REST API server |
| `graph <repo>` | --concepts, --export-html | Build knowledge graph |
| `telemetry <repo>` | --limit, --json | Query telemetry |
| `eval <repo>` | --golden, --k | nDCG@10/Recall/MRR |
| `taint <repo>` | --json | Semgrep taint analysis |
| `review <repo>` | --diff, --base, --head, --json, --pr, --post-comments | Diff review (v2.4.0) |
| `search-all <query>` | --k, --json, --config | Federated search |
| `federation add` | alias, path, --weight | Register repo |
| `federation list` | --config | List registered repos |

---

## 18. File Watching

### `FileWatcher` (single repo)

**Module:** `src/trelix/indexing/watcher.py`

Uses `watchdog` library (synchronous, per-path event emitter threads).

```python
class FileWatcher:
    def __init__(self, indexer: Indexer, walker: FileWalker,
                 debounce_ms: int = 500) -> None
    def start(self) -> None
    def stop(self) -> None
```

**Debounce:** rapid edits to the same file collapse into a single re-index (default 500ms window). Each file has its own debounce timer; saving mid-edit doesn't trigger partial re-index.

**Event handling:**
- `on_modified` / `on_created` → `indexer.index_file(path)` after debounce
- `on_deleted` → `db.delete_file_by_path(abs_path, rel_path, vector_store)`
- Directory events: ignored
- Unknown event type: ignored

Requires `pip install trelix[watch]` (`watchdog>=4.0.0`).

### `MultiRepoWatcher` (v2.4.0, all federated repos)

**Module:** `src/trelix/indexing/multi_watcher.py`

Uses `watchfiles` library (Rust-based, async, single call over all paths).

```python
class MultiRepoWatcher:
    def __init__(self, registry: RepoRegistry, debounce_ms: int = 1600) -> None
    async def run(self, stop_event: asyncio.Event) -> None  # blocks until set
    def stats(self) -> dict[str, int]   # {repos_watched, files_reindexed, files_skipped_unchanged}
    def _get_repo_for_path(self, file_path: str) -> str | None
    def _is_unchanged(self, path: str) -> bool   # MD5 hash guard
```

**Single call:** `watchfiles.awatch(*all_repo_paths, stop_event=stop_event, debounce=1600)` — watchfiles' Rust layer handles debounce for all repos simultaneously.

**Hash guard:** MD5 of file bytes cached in `_file_hashes`. If content unchanged, skip re-index (prevents cascade loops when indexer writes `.trelix/` files inside watched tree).

**Deletion handling:**
- `Change.deleted` → `db.delete_file_by_path(abs_path, rel_path, vector_store)` — removes from SQLite and vector store
- Evicts path from `_file_hashes`

**Path boundary guard:** uses trailing `/` check to prevent `/myrepo` matching `/myrepo2/file.py`:
```python
repo_dir = entry.path.rstrip("/") + "/"
if file_path.startswith(repo_dir) or file_path == entry.path.rstrip("/"): ...
```

Requires `pip install trelix[watch]` (`watchfiles>=0.21`).

---

## 19. Telemetry and Observability (v2.4.0)

### `TelemetryWriter`

**Module:** `src/trelix/retrieval/telemetry.py`

```python
class TelemetryWriter:
    def __init__(self, db: Database, enabled: bool = True) -> None
    def record(self, context: RetrievedContext, elapsed_ms: float,
               expansion_result: "ExpandResult | None" = None) -> None
    # No-op when enabled=False
    # Never raises — all exceptions caught and logged as debug
```

Written to `query_telemetry` table: `query`, `intent`, `elapsed_ms`, `result_count`, `leg_sizes` (JSON), `thumbs_up`.

**v2.4.0 expansion columns** (nullable, stored as SQL NULL when expansion not used):
- `expansion_used: bool` — True when LLM expansion ran
- `expansion_variants: int` — number of query variants generated (including original)
- `expansion_elapsed_ms: float` — LLM call latency

### `ExpandResult` (v2.4.0)

**Module:** `src/trelix/retrieval/query_expansion.py`

```python
@dataclass(frozen=True)
class ExpandResult:
    queries: list[str]     # [original] + up to N variants
    llm_used: bool         # False when LLM unavailable or expansion failed
    elapsed_ms: float      # LLM call latency (0.0 on fallback)
```

Replaces bare `list[str]` return type of `MultiQueryExpander.expand()`. Immutable (frozen=True) — cannot be mutated after creation.

### `MultiQueryExpander`

```python
class MultiQueryExpander:
    def __init__(self, llm_config: LLMConfig | None, n: int = 2) -> None
    def expand(self, query: str) -> ExpandResult
    # Returns ExpandResult(queries=[original], llm_used=False, elapsed_ms=0.0) on failure
    # Never raises
```

### `HyDEExpander`

```python
class HyDEExpander:
    def expand(self, query: str) -> str
    # Returns synthetic code snippet, or "" on failure
    # Used as vector query instead of NL query
    # Research: Gao et al. 2022, arXiv:2212.10496
```

---

## 20. LLM Client Abstraction

**Module:** `src/trelix/llm/client.py`, `llm/providers/`

```python
@dataclass
class ChatMessage:
    role: str   # "system" | "user" | "assistant"
    content: str

@dataclass
class ChatResponse:
    content: str
    input_tokens: int
    output_tokens: int
    model: str

class TrelixChatClient(ABC):
    @abstractmethod
    def complete(self, messages, max_tokens, temperature, system) -> ChatResponse: ...
    def tool_call(self, messages, tools, force_tool, max_tokens) -> ToolCallResponse: ...
    def stream(self, messages, max_tokens, temperature, system) -> Iterator[str]: ...
```

Concrete backends:
- `OpenAIBackend` — `openai.OpenAI`, sync + streaming
- `AnthropicBackend` — `anthropic.Anthropic`, with prompt caching support
- `BedrockBackend` — boto3 `invoke_model` / `invoke_model_with_response_stream`; primary → fallback chain on `ValidationException`
- `VertexBackend` — `google.generativeai`
- `LiteLLMBackend` — 100+ providers via single interface

Factory: `build_chat_client(config: LLMConfig) -> TrelixChatClient`

---

## 21. Key Design Invariants

These invariants must not be violated without understanding the downstream effects:

1. **Zero-mutation immutability** — `SearchResult` and all pipeline dataclasses are re-created (not mutated) when scores/ranks change. This prevents subtle bugs in parallel evaluation pipelines.

2. **SHA-256 for incremental indexing, MD5 for watch guard** — SHA-256 (`IndexedFile.hash`) drives skip-unchanged-files in batch indexing. MD5 (`MultiRepoWatcher._is_unchanged`) drives the watch-event hash guard — MD5 is faster and correctness matters less (false positive = extra re-index, not data loss).

3. **`DimensionGuard` at both index AND query time** — catching provider switches at Indexer startup prevents corrupted index. Catching at Retriever startup prevents silent wrong-results on existing corrupt index.

4. **FTS5 triggers maintain BM25 sync** — `AFTER INSERT`, `AFTER DELETE`, `AFTER UPDATE` triggers on `symbols` table keep `symbols_fts` in sync. Never directly `INSERT` into `symbols_fts` — it will desync.

5. **`callee_id=None` is correct for external calls** — unresolved calls to stdlib/external libraries intentionally have `callee_id=None`. Wrong edge is worse than missing edge. The 4-priority cascade resolves internal calls; external remains NULL.

6. **Debug traces are thread-local** — `_trace_local = threading.local()` ensures parallel eval workers (e.g. batch retrieval benchmarks) never cross-contaminate JSON traces.

7. **No intent-switch logic in `Retriever`** — all behavior is data-driven through `RetrievalStrategy`. `INTENT_STRATEGIES` is the single source of truth. Adding new intent = one dict entry only.

8. **`graph_rag_enabled=True` is the only feature flag on by default** — all other optional features (FLARE, HyDE, multi-query, sparse, graph search, file summaries, MGS3, agentic) default to False.

9. **MCP server `stdout` is exclusive to JSON protocol** — `logging.basicConfig(stream=sys.stderr)` is the first line of `server.py`, before any imports. Any `print()` to stdout breaks the MCP pipe silently.

10. **`FederatedRetriever` uses monotonic time for TTL** — `time.monotonic()` not `time.time()` — avoids cache expiry bugs during daylight saving transitions.

11. **`trelix migrate-vectors --reset` must be followed by `trelix index`** — `--reset` deletes all stored embeddings but does NOT re-create them. The DimensionGuard record is also cleared. Running queries against an empty vector store returns zero results.

12. **`GraphUpdater` always does full rebuild** — never partial. Rationale: avoids stale-edge bugs when call targets change across files. A partial update would need to know which other symbols reference the changed one — exactly the problem the graph is solving.

---

## 22. Data Flow Diagrams

### Full Indexing Pipeline

```
repo/
├── src/auth.py
├── src/models.py
└── tests/test_auth.py

     FileWalker (gitignore + extension filter + size limit)
          │
          ▼
     [IndexedFile...] ──── SHA-256 hash check ────► SKIP (unchanged)
          │                                          │
          │ (changed or new)                         ▼
          ▼                                       [continue]
     Phase 1: ThreadPoolExecutor (parse_workers=4)
          │
          ├── PythonParser (tree-sitter) ──► ParseResult(symbols, calls, types, imports)
          ├── TypeScriptParser
          └── ...
          
     Phase 2: Sequential DB write (main thread)
          │
          ├── db.upsert_file() ─────────────► files table
          ├── db.insert_symbol() ──────────► symbols table + FTS5 trigger
          ├── db.transaction() ────────────► commit batch
          ├── _store_call_edges() ─────────► calls table
          ├── _store_type_edges() ─────────► type_edges table
          ├── chunker.build_chunks() ──────► [Chunk(symbol_id, chunk_text, token_count)]
          └── [Optional] FileSummarizer ──► file_summaries table + summary embedding

     Phase 3: asyncio.gather (embed + store, Semaphore(4))
          │
          ├── embedder.embed_async(batch) ─► [embedding vectors]
          └── vector_store.upsert_batch() ─► sqlite-vec HNSW (or Qdrant/LanceDB)
          │
          └── [Optional] SparseEmbedder ──► sparse_embeddings table

     Phase 4: Cross-file resolution
          │
          ├── resolve_cross_file_calls() ──► callee_id filled in calls table
          ├── resolve_import_file_ids() ───► imported_file_id in imports table
          ├── resolve_cross_file_type_edges()
          └── resolve_angular_selectors()

     DimensionGuard.record(db, dimension) ─► index_metadata table
```

### Full Retrieval Pipeline

```
User: "how does the authentication middleware work?"
          │
          ▼
     AdaptiveRouter.route()
          │ TIER_2_SINGLE (default — 90% of queries)
          ▼
     QueryPlanner._plan_direct() ──── LLM (gpt-4o-mini) ───► QueryPlan
          │                                                    intent: FEATURE_FLOW
          │                                                    legs: vector+bm25
          │                                                    expand_depth: 2
          │                                                    import_depth: 2
          ▼
     _retrieve_standard(plan)
          │
          ├── Sub-query leg execution (parallel)
          │    ├── VECTOR: embed_query("auth middleware how") → HNSW → [SearchResult×20]
          │    ├── BM25:   FTS5("authentication middleware") → [SearchResult×20]
          │    └── GREP:   grep("authenticate") → [SearchResult×10]
          │
          ├── [Optional] Multi-query expansion
          │    ├── LLM variant 1: "JWT validation middleware chain"
          │    ├── LLM variant 2: "request auth filter pipeline"
          │    └── Each variant runs all 3 legs in parallel
          │
          ├── RRF fusion (k=60, file-type weights)
          │    └── Merged, deduped by symbol_id → [SearchResult×40]
          │
          ├── Graph expansion (on top 10 fused)
          │    ├── expand_with_call_graph(depth=2)
          │    ├── expand_with_imports(depth=2, direction="both")
          │    └── expand_with_type_edges(max_extra=15)
          │
          ├── [Optional] Reranker (cross-encoder / Cohere / PLAID)
          │    └── → top 15 reranked
          │
          ├── [Optional] PageRank boost
          │    └── × 1.3 for top-200 central symbols
          │
          └── ContextAssembler (greedy, 12k tokens)
               └── RetrievedContext
                    └── Synthesizer → streamed answer
```

### Multi-Repo Federation (v2.4.0)

```
trelix search-all "how does caching work"
          │
          ▼
     FederatedRetriever.retrieve(query, k=10)
          │
          ├── Cache check [SHA-256(query + sorted_paths + k)] ─► HIT: return cached
          │                                                        │
          │ MISS                                                   │
          ▼                                                       ▼
     _query_repos() [ThreadPoolExecutor, max_workers=4]
          │
          ├── Retriever(IndexConfig(repo_path="/myapp")).retrieve(query)
          ├── Retriever(IndexConfig(repo_path="/infra")).retrieve(query)
          └── Retriever(IndexConfig(repo_path="/shared")).retrieve(query)
          │
          ▼
     reciprocal_rank_fusion([results_a, results_b, results_c])
          │
          ▼
     dedup by (file.rel_path, chunk.symbol_id)
          │
          ▼
     Cache set (TTL=120s) + return list[SearchResult]
```

---

## 23. Extension Points

### Adding a New Language Parser

1. Create `src/trelix/indexing/parser/extractors/<language>.py`
2. Inherit `BaseParser(ABC)` and implement `parse(source, file_id) → ParseResult`
3. Add entry to `Language` enum in `core/models.py`
4. Register in `_PARSER_REGISTRY` in `parser/registry.py`
5. Add to default `WalkerConfig.languages` list

### Adding a New Embedding Provider

1. Implement `BaseEmbedder` ABC in `embedder/base.py` or a new module
2. Add to `provider` Literal in `EmbedderConfig`
3. Add case to `make_embedder()` factory in `embedder/base.py`
4. Add dimension constant to `EmbedderConfig.effective_dimension` property

### Adding a New LLM Backend

1. Implement `TrelixChatClient` ABC in `llm/providers/<name>_backend.py`
2. Add to `provider` Literal in `LLMConfig`
3. Add case to `build_chat_client()` factory in `llm/factory.py`

### Adding a New Retrieval Intent

1. Add value to `IntentType(StrEnum)`
2. Add entry to `INTENT_STRATEGIES` dict in `retrieval/planner/models.py`
3. Add training example to the LLM planner prompt in `planner/agent.py`

That's it — no changes to `Retriever` needed.

### Adding a New Vector Backend

1. Implement `BaseVectorStore` ABC in `store/vector_<name>.py`
2. Add case to `make_vector_store()` factory in `store/vector.py`
3. Add to `StoreConfig.backend` Literal

---

*trelix v2.7.1 — last updated 2026-07-10*
