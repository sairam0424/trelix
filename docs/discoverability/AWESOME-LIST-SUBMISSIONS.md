# Awesome List Submissions

PR bodies for submitting trelix to major awesome lists.

---

## awesome-mcp-servers

**Repo**: https://github.com/punkpeye/awesome-mcp-servers

### PR Title

`Add trelix — code intelligence MCP server for semantic search & retrieval`

### PR Body

```
## Description

Adding [trelix](https://github.com/sairam0424/trelix) to the code intelligence / developer tools section.

**One-line description**: trelix is a code intelligence MCP server that gives AI assistants fast semantic search, call-graph traversal, and LLM synthesis over any local repository — zero external infrastructure required.

### Install

```bash
pip install trelix-mcp
claude mcp add trelix -- trelix-mcp
```

### Key differentiators

1. **Hybrid search by default** — contextual embeddings + BM25 + grep fused via Reciprocal Rank Fusion; not just vector similarity, so keyword-precise queries (symbol names, error codes) work out of the box.
2. **Call-graph expansion** — results are automatically enriched with callers, callees, and import edges so the LLM sees the full context of any code path, not just the nearest chunk.
3. **Zero-infra** — ships with a single SQLite file (`.trelix/index.db`) using sqlite-vec HNSW; scales to Qdrant for >500k chunks with one env-var flip.

### Checklist

- [x] Link points to a public GitHub repository
- [x] Package is on PyPI (`pip install trelix-mcp`)
- [x] README includes install instructions
- [x] MIT licensed
```

---

## awesome-llm-apps

**Repo**: https://github.com/Shubhamsaboo/awesome-llm-apps

### PR Title

`Add trelix — code retrieval engine for LLM-powered developer tools`

### PR Body

```
## Description

Adding [trelix](https://github.com/sairam0424/trelix) to the code intelligence / RAG section.

**One-line description**: trelix is a production-grade code retrieval engine that combines tree-sitter AST parsing, contextual hybrid search, and call-graph expansion to give LLMs accurate, context-rich answers about any codebase.

### Install

```bash
pip install trelix
# or with Voyage AI code embeddings (best quality)
pip install "trelix[voyage]"
```

### Key differentiators

1. **3-tier adaptive query planner** — routes queries through direct answer / single-step retrieval / multi-step decomposition automatically, so LLM calls are never wasted on retrieval it doesn't need.
2. **GraphRAG map-reduce synthesis** — for large result sets (>8k tokens), trelix runs a map-reduce summarisation pass before LLM synthesis, handling arbitrarily large codebases without context-window overflows.
3. **Eval harness built in** — ships with MRR, Recall@1/5/10, and NDCG@10 metrics on 50 real queries (`make eval-full`), making it easy to benchmark retrieval quality before shipping.

### Checklist

- [x] Link points to a public GitHub repository
- [x] Package is on PyPI
- [x] README includes a working code example
- [x] MIT licensed
```

---

## awesome-langchain

**Repo**: https://github.com/kyrolabs/awesome-langchain

### PR Title

`Add trelix-langchain — code intelligence retriever for LangChain`

### PR Body

```
## Description

Adding [trelix](https://github.com/sairam0424/trelix) to the retrievers / integrations section.

**One-line description**: `trelix-langchain` is a drop-in LangChain retriever that indexes any local repository with tree-sitter AST parsing and serves semantically-relevant code chunks via the standard `BaseRetriever` interface.

### Install

```bash
pip install trelix-langchain
```

### Usage

```python
from trelix_langchain import TrelixRetriever

retriever = TrelixRetriever(repo_path="/path/to/repo")
docs = retriever.invoke("how does authentication work?")
```

### Key differentiators

1. **LangChain-native interface** — implements `BaseRetriever` so it slots into any existing LCEL chain, RAG pipeline, or agent tool without adapter code.
2. **Code-aware chunking** — unlike generic text splitters, trelix chunks at AST boundaries (function / class / method level) and prepends LLM-generated context summaries to each chunk, reducing retrieval failures by ~67%.
3. **Offline-first** — `TrelixRetriever(provider="local")` runs fully offline using sentence-transformers; no API key required for indexing or retrieval.

### Checklist

- [x] Link points to a public GitHub repository
- [x] Package is on PyPI (`pip install trelix-langchain`)
- [x] README includes a working code example
- [x] MIT licensed
```
