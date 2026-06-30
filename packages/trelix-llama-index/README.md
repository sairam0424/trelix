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

## Streaming synthesis (v2.0.0+)

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

## Beast-mode retrieval (v2.1.0+)

v2.1.0 activates enhanced retrieval features via environment variables. Enable HyDE synthetic snippet embedding and PageRank-based symbol boosting for architecturally central symbols:

```python
from trelix_llama_index import TrelixIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

# v2.1.0: Beast-mode features active via env vars
# TRELIX_RETRIEVAL_HYDE_FALLBACK=true — HyDE synthetic snippet embedding
# TRELIX_RETRIEVAL_PAGERANK_BOOST=true — boost architecturally central symbols

retriever = TrelixIndexRetriever(
    repo_path="/path/to/repo",
    k=10,
)
nodes = retriever.retrieve("how does the payment processing work?")
```

Enable beast-mode via environment:

```bash
# HyDE fallback: generate synthetic query augmentation if real embeddings underperform
TRELIX_RETRIEVAL_HYDE_FALLBACK=true trelix index /path/to/repo

# PageRank boost: weight results by architectural centrality
TRELIX_RETRIEVAL_PAGERANK_BOOST=true trelix index /path/to/repo

# Both enabled (recommended for complex monorepos)
TRELIX_RETRIEVAL_HYDE_FALLBACK=true TRELIX_RETRIEVAL_PAGERANK_BOOST=true trelix index /path/to/repo
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

## Graph-Enhanced Retrieval (v2.1.0+)

Enable the knowledge graph as a 4th retrieval leg for architecture-aware queries. v2.1.0 integrates beast-mode features for optimal performance:

```python
from trelix_llama_index import TrelixIndexRetriever

# With graph-aware BFS (requires trelix[knowledge-graph])
# v2.1.0: HyDE + PageRank boost activate automatically in this mode
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

## Evaluation

Use `trelix eval --golden golden.jsonl` to measure nDCG@10, Recall@10, and MRR on a golden query set. This harness is recommended before and after enabling beast-mode features (HyDE, PageRank boost) to quantify retrieval quality improvements:

```bash
# Create golden queries (JSONL: {"query": "...", "relevant_files": ["file1.py", "file2.py"]})
trelix eval --golden golden.jsonl --config index.db

# Output: nDCG@10, Recall@10, MRR, per-query breakdowns
```

Example golden.jsonl:
```json
{"query": "how does authentication middleware work?", "relevant_files": ["src/auth/middleware.py", "src/auth/decorators.py"]}
{"query": "payment processing flow", "relevant_files": ["src/payment/processor.py", "src/payment/handler.py"]}
```

## Links

- [trelix on GitHub](https://github.com/sairam0424/trelix)
- [trelix on PyPI](https://pypi.org/project/trelix/)
- [trelix-mcp](https://pypi.org/project/trelix-mcp/) — MCP server for Claude Code, Cursor, Windsurf
- [trelix-langchain](https://pypi.org/project/trelix-langchain/) — LangChain retriever
