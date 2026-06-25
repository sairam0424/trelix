"""
50 eval cases for trelix self-evaluation.

Each case is a (query, expected_file) pair where expected_file is a
rel_path fragment (substring match) against the indexed trelix source.

Categories:
  - Symbol lookups   (15 cases): specific classes, functions, or modules
  - Feature flows    (15 cases): end-to-end pipeline questions
  - Blast radius     (10 cases): dependency / import lookup questions
  - Config/overview  (10 cases): configuration and project overview questions
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Symbol lookups (15 cases)
# ---------------------------------------------------------------------------
#
# These queries name a specific class/function and expect the retriever to
# surface the file that defines it in the top results.

SYMBOL_CASES: list[tuple[str, str]] = [
    ("where is RRF fusion implemented", "src/trelix/retrieval/fusion.py"),
    ("QueryPlanner class definition", "src/trelix/retrieval/planner/agent.py"),
    ("BM25 search implementation", "src/trelix/retrieval/bm25.py"),
    ("where is FileWalker defined", "src/trelix/indexing/walker.py"),
    ("symbol chunking logic and Chunker class", "src/trelix/indexing/chunker.py"),
    ("SearchResult dataclass", "src/trelix/core/models.py"),
    ("Retriever class definition", "src/trelix/retrieval/retriever.py"),
    ("ContextAssembler implementation", "src/trelix/retrieval/assembler.py"),
    ("grep_search function", "src/trelix/retrieval/grep_search.py"),
    ("Database class SQLite store", "src/trelix/store/db.py"),
    ("BaseVectorStore and make_vector_store", "src/trelix/store/vector.py"),
    ("BaseEmbedder interface and make_embedder", "src/trelix/embedder/base.py"),
    ("Indexer class index method", "src/trelix/indexing/indexer.py"),
    ("graph expansion expand_with_call_graph", "src/trelix/retrieval/graph.py"),
    ("rerank function implementation", "src/trelix/retrieval/reranker.py"),
]

# ---------------------------------------------------------------------------
# Feature flows (15 cases)
# ---------------------------------------------------------------------------
#
# These queries describe a pipeline behaviour and expect the primary
# orchestrating file to appear in the top results.

FEATURE_FLOW_CASES: list[tuple[str, str]] = [
    ("how does the indexing pipeline work end to end", "src/trelix/indexing/indexer.py"),
    ("how does retrieval work end to end", "src/trelix/retrieval/retriever.py"),
    ("how are files discovered and walked", "src/trelix/indexing/walker.py"),
    ("how does query planning and intent routing work", "src/trelix/retrieval/planner/agent.py"),
    ("how are symbols parsed from source files", "src/trelix/indexing/parser/base.py"),
    (
        "how does Python symbol extraction work",
        "src/trelix/indexing/parser/extractors/python.py",
    ),
    (
        "how does TypeScript extraction work",
        "src/trelix/indexing/parser/extractors/typescript.py",
    ),
    ("how does vector similarity search work", "src/trelix/store/vector.py"),
    ("how is context assembled for LLM injection", "src/trelix/retrieval/assembler.py"),
    ("how is the CLI entry point structured", "src/trelix/cli/main.py"),
    (
        "how does Go parser extract symbols",
        "src/trelix/indexing/parser/extractors/go.py",
    ),
    (
        "how does Rust parser extract symbols",
        "src/trelix/indexing/parser/extractors/rust.py",
    ),
    ("how does graph expansion use call edges", "src/trelix/retrieval/graph.py"),
    (
        "how are query plan models and intent types defined",
        "src/trelix/retrieval/planner/models.py",
    ),
    ("how is the synthesizer prompt built", "src/trelix/retrieval/synthesizer.py"),
]

# ---------------------------------------------------------------------------
# Blast radius (10 cases)
# ---------------------------------------------------------------------------
#
# These queries ask which file defines the thing that is imported everywhere.

BLAST_RADIUS_CASES: list[tuple[str, str]] = [
    ("what imports the Chunk dataclass", "src/trelix/core/models.py"),
    ("what defines IndexedFile used across retrieval", "src/trelix/core/models.py"),
    ("where is Symbol dataclass defined", "src/trelix/core/models.py"),
    ("what file defines RetrievedContext", "src/trelix/core/models.py"),
    ("what module defines IndexConfig", "src/trelix/core/config.py"),
    ("where is EmbedderConfig defined", "src/trelix/core/config.py"),
    ("where is RetrievalConfig defined", "src/trelix/core/config.py"),
    ("what file exports the parser registry", "src/trelix/indexing/parser/registry.py"),
    ("where is reciprocal_rank_fusion exported from", "src/trelix/retrieval/fusion.py"),
    ("what file defines CallEdge dataclass", "src/trelix/core/models.py"),
]

# ---------------------------------------------------------------------------
# Config / overview (10 cases)
# ---------------------------------------------------------------------------
#
# These queries ask about configuration values or project-level concepts.

CONFIG_OVERVIEW_CASES: list[tuple[str, str]] = [
    ("what is the default token budget for context", "src/trelix/core/config.py"),
    ("what are the default walker ignore directories", "src/trelix/core/config.py"),
    ("what is the default embedding provider", "src/trelix/core/config.py"),
    ("what are the supported languages", "src/trelix/core/config.py"),
    ("RRF k constant configuration", "src/trelix/retrieval/fusion.py"),
    ("chunker max tokens per chunk setting", "src/trelix/core/config.py"),
    ("what does the trelix package export", "src/trelix/__init__.py"),
    ("parser configuration max symbol lines", "src/trelix/core/config.py"),
    ("top k vector bm25 retrieval config defaults", "src/trelix/core/config.py"),
    ("reranker provider configuration", "src/trelix/core/config.py"),
]

# ---------------------------------------------------------------------------
# Combined: all 50 cases
# ---------------------------------------------------------------------------

TRELIX_SELF_CASES: list[tuple[str, str]] = (
    SYMBOL_CASES + FEATURE_FLOW_CASES + BLAST_RADIUS_CASES + CONFIG_OVERVIEW_CASES
)

assert len(TRELIX_SELF_CASES) == 50, f"Expected 50 eval cases, got {len(TRELIX_SELF_CASES)}"
