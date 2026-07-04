# trelix-llama-index

LlamaIndex retriever for [trelix](https://github.com/sairam0424/trelix) — semantic code search using Tree-sitter AST parsing, hybrid BM25+vector search, call-graph expansion, and streaming synthesis support.

## Install

```bash
pip install trelix-llama-index
```

## Usage

```python
from trelix_llama_index import TrelixIndexRetriever

# First index your repo (one-time)
# trelix index /path/to/repo

retriever = TrelixIndexRetriever(repo_path="/path/to/repo", k=10)
nodes = retriever.retrieve("how does authentication work?")

for node in nodes:
    print(node.node.metadata["file"], node.score)
    print(node.node.text[:200])
```

## With LlamaIndex query engine

```python
from llama_index.core import VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from trelix_llama_index import TrelixIndexRetriever

retriever = TrelixIndexRetriever(repo_path="/path/to/repo", k=10)
query_engine = RetrieverQueryEngine.from_args(retriever)
response = query_engine.query("How does the authentication middleware work?")
print(response)
```

## Streaming synthesis (v2.0.0+, enhanced v2.4.0)

```python
from trelix_llama_index import TrelixIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

retriever = TrelixIndexRetriever(repo_path="/path/to/repo", k=10)
query_engine = RetrieverQueryEngine.from_args(retriever)

# Stream response token-by-token
response = query_engine.query_stream("Explain the payment flow")
for text_chunk in response:
    print(text_chunk, end="", flush=True)
```

## Configuration

| Env var | Default | Description |
|---------|---------|-------------|
| `TRELIX_EMBEDDER_PROVIDER` | `local` | Embedding provider: `local` \| `openai` \| `azure` \| `bedrock-cohere` \| `bedrock-titan` \| `huggingface` \| `voyage` |
| `OPENAI_API_KEY` | — | Required for `openai` provider |
| `AZURE_API_KEY` | — | Required for `azure` provider |
| `AWS_ACCESS_KEY_ID` | — | Required for Bedrock providers |
| `HUGGINGFACE_API_KEY` | — | Required for `huggingface` provider |
| `VOYAGE_API_KEY` | — | Required for `voyage` provider |

## Provider switching (v2.0.0+)

```bash
# Use Bedrock Cohere embeddings (best retrieval, reuses AWS creds)
TRELIX_EMBEDDER_PROVIDER=bedrock-cohere trelix index /path/to/repo

# Use HuggingFace embeddings (open-source alternatives)
TRELIX_EMBEDDER_PROVIDER=huggingface HUGGINGFACE_API_KEY=hf_... trelix index /path/to/repo

# Use Voyage embeddings (specialized for code search)
TRELIX_EMBEDDER_PROVIDER=voyage VOYAGE_API_KEY=pa-... trelix index /path/to/repo

# Use local embeddings (no API key needed)
TRELIX_EMBEDDER_PROVIDER=local trelix index /path/to/repo
```

## Graph-Enhanced Retrieval

Enable the knowledge graph as a 4th retrieval leg for architecture-aware queries:

```python
from trelix_llama_index import TrelixIndexRetriever

# With graph-aware BFS (requires trelix[knowledge-graph])
retriever = TrelixIndexRetriever(
    repo_path="/path/to/repo",
    k=10,
    graph_search_enabled=True,   # enables 4th BFS retrieval leg
    graph_search_depth=2,
)

nodes = retriever.retrieve("how does the auth module interact with the DB layer?")
for node in nodes:
    print(node.node.metadata.get("source"))  # file path
    print(node.score)                         # combined RRF + graph score
```

Install with graph support:

```bash
pip install trelix-llama-index 'trelix[knowledge-graph]'
```

### How it works

When `graph_search_enabled=True`, trelix builds (or loads) a NetworkX MultiDiGraph over the
indexed repository and runs a BFS expansion from the highest-degree nodes relevant to the query.
Results from all four legs are fused via Reciprocal Rank Fusion (RRF):

| Retrieval leg | Technique |
|---------------|-----------|
| Vector | Semantic embedding similarity |
| BM25 | Keyword / TF-IDF |
| Call-graph expansion | Symbol → caller/callee traversal |
| **Graph BFS** *(new)* | **Knowledge-graph breadth-first search** |

### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `graph_search_enabled` | `False` | Enable the graph BFS retrieval leg (opt-in, zero impact when off) |
| `graph_search_depth` | `2` | BFS depth from seed nodes |
| `graph_search_max_results` | `15` | Maximum nodes returned from graph leg before RRF |

Environment variable alternative:

```bash
TRELIX_GRAPH_SEARCH_ENABLED=true trelix index /path/to/repo
```

### Benchmarks (trelix repo, 4,599 nodes / 4,945 edges)

- Graph build time: **0.34 s**
- Communities detected: **2,409** (Louvain algorithm)
- `graph_search` depth=2: **10 results** from top node (degree 438)
- Full retrieval with graph enabled: **30 results** (5 graph + 19 vector + 4 BM25 + 2 graph_expansion)

### Breaking change (v2.0.0)

The old `trelix graph <repo> <symbol>` call-graph display command was renamed:

```bash
# Before (v1.x)
trelix graph ./repo MyClass

# After (v2.0.0+)
trelix call-graph ./repo MyClass
```

`trelix graph` now builds and queries the knowledge graph:

```bash
trelix graph ./repo                          # build graph, print summary
trelix graph ./repo --visualize              # open Pyvis HTML in browser
trelix graph ./repo --concepts               # run LLM concept extraction
trelix graph ./repo --json                   # emit graph stats as JSON
```

## What's new in v2.4.0

### ⚠️ Breaking change — `search_code` MCP tool response envelope

`search_code` now returns a pagination envelope instead of a bare list:

```json
{"results": [...], "next_cursor": 10, "total_available": 25}
```

Update any MCP client code that iterates `search_code(...)` directly:

```python
# Before (v2.3.0)
for result in search_code(query="auth", repo_path="/repo"):
    ...

# After (v2.4.0)
response = search_code(query="auth", repo_path="/repo")
for result in response["results"]:
    ...
# Paginate: pass response["next_cursor"] as cursor= for the next page
```

### FederatedRetriever TTL cache

```python
from trelix_llama_index import TrelixIndexRetriever

# cache_ttl=120 (seconds) — SHA-256-keyed, thread-safe
retriever = TrelixIndexRetriever(repo_path="/path/to/repo", k=10, cache_ttl=120.0)

# Inspect cache stats
print(retriever.cache_stats())  # {"hits": 3, "misses": 1, "size": 1}

# Force eviction
retriever.clear_cache()
```

Set `cache_ttl=0` to disable caching entirely. Expected ~90% hit rate for typical debugging-session query patterns.

### Multi-Query Expansion observability

When `multi_query_enabled=True` (requires `trelix>=2.3.0`), each retrieval now records expansion metadata:

```python
nodes = retriever.retrieve("how does auth work?")
# expansion_used, expansion_variants, expansion_elapsed_ms written to query_telemetry table
```

### GitHub PR review integration

```bash
# Review a PR diff locally
trelix review --pr owner/repo#42

# Review and post findings back as a GitHub review comment
trelix review --pr owner/repo#42 --post-comments
```

Requires `GITHUB_TOKEN` env var. The `TrelixIndexRetriever` can be used as the retrieval backend inside `DiffReviewer`.

### Multi-repo file watching

```bash
# Watch all indexed repos simultaneously; updates index on file changes
trelix watch-all
```

Deleted files are removed from the SQLite index and vector store automatically.

### Config field rename

`flare_max_retries` replaces `flare_max_iterations` in `RetrievalConfig`. Both the new env var `TRELIX_RETRIEVAL_FLARE_MAX_RETRIES` and the old `TRELIX_RETRIEVAL_FLARE_MAX_ITER` are accepted (old name emits `DeprecationWarning` and will be removed in v3.0.0).

## Links

- [trelix on GitHub](https://github.com/sairam0424/trelix)
- [trelix on PyPI](https://pypi.org/project/trelix/)
- [trelix-mcp](https://pypi.org/project/trelix-mcp/) — MCP server for Claude Code, Cursor, Windsurf
- [trelix-langchain](https://pypi.org/project/trelix-langchain/) — LangChain retriever
