"""
trelix — fast, reliable code indexing and retrieval.

Tree-sitter AST parsing → contextual hybrid search (vector + BM25 + grep)
→ adaptive 3-tier query planning → call-graph expansion
→ GraphRAG synthesis.

Quick start:
    from trelix.core.config import IndexConfig
    from trelix.indexing.indexer import Indexer
    from trelix.retrieval.retriever import Retriever

    config = IndexConfig(repo_path="/path/to/repo")
    Indexer(config).index()
    ctx = Retriever(config).retrieve("how does authentication work?")
    print(ctx.context_text)
"""

__version__ = "0.5.0"
__all__ = ["__version__"]
