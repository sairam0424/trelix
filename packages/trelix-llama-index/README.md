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

## Links

- [trelix on GitHub](https://github.com/sairam0424/trelix)
- [trelix on PyPI](https://pypi.org/project/trelix/)
- [trelix-mcp](https://pypi.org/project/trelix-mcp/) — MCP server for Claude Code, Cursor, Windsurf
- [trelix-langchain](https://pypi.org/project/trelix-langchain/) — LangChain retriever
