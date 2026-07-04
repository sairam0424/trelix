# trelix-langchain

LangChain retriever for [trelix](https://github.com/sairam0424/trelix) — semantic code search using Tree-sitter AST parsing, hybrid BM25+vector search, call-graph expansion, and streaming synthesis support.

## Install

```bash
pip install trelix-langchain
```

For AWS Bedrock embeddings (Cohere or Titan):

```bash
pip install "trelix-langchain[bedrock]"
```

For code-optimized embeddings (BGE-Code, Nomic-Code, or Lance backend):

```bash
pip install "trelix-langchain[code-embeddings]"
```

With knowledge graph support (NetworkX BFS retrieval leg):

```bash
pip install trelix-langchain 'trelix[knowledge-graph]'
```

## Basic Usage

```python
from trelix_langchain import TrelixRetriever

# First index your repo (one-time)
# trelix index /path/to/repo

retriever = TrelixRetriever(repo_path="/path/to/repo", k=10)
docs = retriever.invoke("how does authentication work?")

for doc in docs:
    print(doc.metadata["source"], doc.metadata["score"])
    print(doc.page_content[:200])
```

Each returned `Document` carries rich metadata:

| Metadata key | Example value |
|---|---|
| `source` | `"src/auth/middleware.py"` |
| `symbol` | `"auth.middleware.require_login"` |
| `language` | `"python"` |
| `kind` | `"function"` |
| `lines` | `"42-78"` |
| `score` | `0.91` |
| `retrieval_source` | `"hybrid"` |

## Graph-Enhanced Retrieval

Enable the knowledge graph as a 4th retrieval leg for architecture-aware queries:

```python
from trelix_langchain import TrelixRetriever

# Standard hybrid retrieval
retriever = TrelixRetriever(repo_path="/path/to/repo", k=10)

# With graph-aware BFS (requires trelix[knowledge-graph])
retriever = TrelixRetriever(
    repo_path="/path/to/repo",
    k=10,
    graph_search_enabled=True,   # enables 4th BFS retrieval leg
    graph_search_depth=2,
)

# Each Document.metadata includes graph source info
docs = retriever.invoke("how does auth relate to the data layer?")
for doc in docs:
    print(doc.metadata["retrieval_source"])  # "graph_search", "vector", "bm25"
```

When `graph_search_enabled=True`, the retriever merges results from four legs:

| Leg | Source | Typical share |
|---|---|---|
| vector | semantic embedding similarity | majority |
| bm25 | keyword / BM25 full-text | secondary |
| graph_expansion | call-graph neighbourhood | supplementary |
| graph_search | BFS over NetworkX knowledge graph | up to `k//2` |

Graph BFS surfaces structurally related symbols even when semantic similarity is low — useful for cross-cutting concerns like auth, logging, and rate-limiting that touch many modules.

### Graph config options

| Parameter | Default | Description |
|---|---|---|
| `graph_search_enabled` | `False` | Opt-in — zero overhead when off |
| `graph_search_depth` | `2` | BFS depth from seed nodes |
| `graph_search_max_results` | `15` | Cap on graph leg results |

You can also set these via environment variables:

```bash
TRELIX_GRAPH_SEARCH_ENABLED=true trelix index /path/to/repo
```

> **Prerequisite**: build the knowledge graph before querying — `trelix graph /path/to/repo`.
> The graph is persisted in `<repo>/.trelix/` and reused across retriever calls.

## LangChain RAG Chain (LCEL)

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from trelix_langchain import TrelixRetriever

retriever = TrelixRetriever(repo_path="/path/to/repo", k=8)

prompt = ChatPromptTemplate.from_template(
    "Answer the question using only the code context below.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}"
)

def format_docs(docs):
    return "\n\n".join(
        f"# {d.metadata['source']} ({d.metadata['symbol']})\n{d.page_content}"
        for d in docs
    )

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | ChatOpenAI(model="gpt-4o")
    | StrOutputParser()
)

answer = chain.invoke("How does the authentication middleware work?")
print(answer)
```

## RetrievalQA (classic interface)

```python
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from trelix_langchain import TrelixRetriever

retriever = TrelixRetriever(repo_path="/path/to/repo", k=10)
llm = ChatOpenAI(model="gpt-4o")

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
)

result = qa.invoke({"query": "Where is rate limiting applied?"})
print(result["result"])
for doc in result["source_documents"]:
    print(" -", doc.metadata["source"])
```

## Configuration

| Env var | Default | Description |
|---|---|---|
| `TRELIX_EMBEDDER_PROVIDER` | `local` | Embedding provider: `local` \| `local-code` \| `bge-code` \| `nomic-code` \| `lance` \| `openai` \| `azure` \| `voyage` \| `bedrock-cohere` \| `bedrock-titan` |
| `OPENAI_API_KEY` | — | Required for `openai` provider |
| `AZURE_API_KEY` | — | Required for `azure` provider |
| `AWS_ACCESS_KEY_ID` | — | Required for Bedrock providers |
| `AWS_SECRET_ACCESS_KEY` | — | Required for Bedrock providers |
| `AWS_DEFAULT_REGION` | `us-east-1` | AWS region for Bedrock |

You can also set the provider directly on the retriever instance:

```python
retriever = TrelixRetriever(repo_path="/path/to/repo", provider="openai", k=10)
```

## Provider Switching (v2.0.0+, updated v2.4.0)

```bash
# Use code-optimized BGE-Code embeddings (best for code semantics)
TRELIX_EMBEDDER_PROVIDER=bge-code trelix index /path/to/repo

# Use Nomic-Code embeddings
TRELIX_EMBEDDER_PROVIDER=nomic-code trelix index /path/to/repo

# Use Bedrock Cohere embeddings (reuses AWS credentials)
TRELIX_EMBEDDER_PROVIDER=bedrock-cohere trelix index /path/to/repo

# Use Azure OpenAI embeddings
TRELIX_EMBEDDER_PROVIDER=azure trelix index /path/to/repo

# Use local sentence-transformers (no API key needed, works offline)
TRELIX_EMBEDDER_PROVIDER=local trelix index /path/to/repo
```

The index and the retriever must use the same provider — re-index whenever you switch.

## Streaming Synthesis (v2.0.0+)

Streaming synthesis support for real-time code context generation:

```python
from trelix_langchain import TrelixRetriever, StreamingSynthesizer
from langchain_openai import ChatOpenAI

retriever = TrelixRetriever(repo_path="/path/to/repo", k=8)
synthesizer = StreamingSynthesizer(
    llm=ChatOpenAI(model="gpt-4o"),
    retriever=retriever
)

# Streamed synthesis output
for chunk in synthesizer.synthesize_stream("How does the auth flow work?"):
    print(chunk, end="", flush=True)
```

## GitHub PR Review (v2.4.0+)

Fetch a pull request diff from GitHub and run `DiffReviewer` directly through the retriever:

```python
from trelix_langchain import TrelixRetriever

retriever = TrelixRetriever(repo_path="/path/to/repo", k=8)

# Retrieve context relevant to a PR diff
# Use the trelix CLI: trelix review --pr owner/repo#42
# Or post review comments: trelix review --pr owner/repo#42 --post-comments
# Requires GITHUB_TOKEN env var
```

Set `GITHUB_TOKEN` in your environment. The integration fetches all changed files in
the PR, retrieves relevant code context for each diff hunk, and can optionally post
a single batched review back to GitHub.

## MCP Pagination (v2.4.0+)

The `search_code` MCP tool now returns a pagination envelope instead of a raw list.
If you call trelix-mcp from LangChain tool wrappers, update your iteration:

```python
# v2.4.0+ response shape from search_code MCP tool
response = search_code_tool.run({"query": "auth", "repo_path": "/repo"})
# response = {"results": [...], "next_cursor": 10, "total_available": 25}

for result in response["results"]:
    print(result)

# Paginate: pass next_cursor as cursor= in the next call
```

## Multi-Query Expansion Observability (v2.4.0+)

When `multi_query_enabled=True` in your `IndexConfig`, the retriever now surfaces
expansion telemetry via the `ExpandResult` dataclass:

```python
from trelix_langchain import TrelixRetriever
from trelix.retrieval import MultiQueryExpander

expander = MultiQueryExpander(llm=your_llm)
expand_result = expander.expand("how does auth work?")
# expand_result.queries       — list of sub-queries generated
# expand_result.llm_used      — model name
# expand_result.elapsed_ms    — wall-clock time for expansion
```

Expansion metadata (`expansion_used`, `expansion_variants`, `expansion_elapsed_ms`) is
persisted automatically to the `query_telemetry` table. Existing databases are upgraded
automatically via an idempotent `ALTER TABLE ADD COLUMN` migration.

## FederatedRetriever Cache (v2.4.0+)

When using `FederatedRetriever` across multiple repos, enable the TTL cache to avoid
redundant retrievals within a debugging session:

```python
from trelix.retrieval import FederatedRetriever

retriever = FederatedRetriever(registry=my_registry, cache_ttl=120.0)
# cache_ttl=0 disables caching
stats = retriever.cache_stats()   # {"hits": 42, "misses": 5, "size": 18}
retriever.clear_cache()           # force eviction
```

The cache is SHA-256-keyed, thread-safe, and scoped to the process lifetime.
Expected ~90% hit rate for typical debugging-session query patterns.

## Multi-Repo Watching (v2.4.0+)

Watch multiple repos simultaneously and keep their indexes live:

```bash
# CLI
trelix watch-all

# Watches all registered repos; shows per-repo stats on exit; Ctrl+C to stop
```

```python
from trelix.watchers import MultiRepoWatcher

watcher = MultiRepoWatcher(repo_paths=["/repo/a", "/repo/b"])
await watcher.watch()  # uses watchfiles under the hood; hash guard prevents cascade re-index
```

## Configuration (v2.4.0+)

In addition to the env vars above, v2.4.0 adds:

| Env var | Default | Description |
|---|---|---|
| `TRELIX_RETRIEVAL_FLARE_MAX_RETRIES` | `3` | Max FLARE re-retrieval iterations (replaces `TRELIX_RETRIEVAL_FLARE_MAX_ITER`) |
| `TRELIX_GRAPH_SEARCH_ENABLED` | `false` | Enable graph BFS retrieval leg |
| `GITHUB_TOKEN` | — | Required for `trelix review --pr` GitHub integration |

> `TRELIX_RETRIEVAL_FLARE_MAX_ITER` is still accepted but emits a `DeprecationWarning`. It will be removed in v3.0.0.

## Links

- [trelix on GitHub](https://github.com/sairam0424/trelix)
- [trelix on PyPI](https://pypi.org/project/trelix/)
- [trelix-mcp](https://pypi.org/project/trelix-mcp/) — MCP server for Claude Code, Cursor, Windsurf
- [trelix-llama-index](https://pypi.org/project/trelix-llama-index/) — LlamaIndex retriever
