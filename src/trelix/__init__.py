"""
trelix — fast, reliable code indexing and retrieval.

Tree-sitter AST parsing → contextual hybrid search (vector + BM25 + grep)
→ adaptive 3-tier query planning → call-graph expansion
→ GraphRAG synthesis.

Quick start::

    from trelix import IndexConfig, Indexer, Retriever

    config = IndexConfig(repo_path="/path/to/repo")
    Indexer(config).index()
    ctx = Retriever(config).retrieve("how does authentication work?")
    print(ctx.context_text)

LLM synthesis::

    from trelix import IndexConfig, LLMConfig
    # Set TRELIX_LLM_PROVIDER=bedrock in .env for AWS Bedrock
    config = IndexConfig(repo_path="/path/to/repo")
    # trelix ask ./repo "how does auth work?"
"""

from __future__ import annotations

from trelix.core.config import EmbedderConfig, IndexConfig, LLMConfig
from trelix.embedder.base import BaseEmbedder, make_embedder
from trelix.indexing.indexer import Indexer
from trelix.llm.client import ChatMessage, ChatResponse, ToolCallResponse, TrelixChatClient
from trelix.llm.factory import build_chat_client
from trelix.retrieval.retriever import Retriever

__version__ = "2.7.1"

__all__ = [
    "__version__",
    # Config
    "IndexConfig",
    "EmbedderConfig",
    "LLMConfig",
    # Core pipeline
    "Indexer",
    "Retriever",
    # Embedder
    "BaseEmbedder",
    "make_embedder",
    # LLM client
    "TrelixChatClient",
    "ChatMessage",
    "ChatResponse",
    "ToolCallResponse",
    "build_chat_client",
]
