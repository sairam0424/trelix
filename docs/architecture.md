# Trelix Architecture

## Indexing Pipeline (offline)

```
Repository
  └─ FileWalker           (.gitignore-aware, SHA-256 change detection)
       └─ Tree-sitter Parser  (per-language extractors → symbols + call/import/type edges)
            ├─ Chunker     (context-header breadcrumb chunks → tiktoken token count)
            │    └─ Embedder  (azure | openai | local sentence-transformers)
            │         └─ sqlite-vec  (ANN vector store)
            └─ SQLite DB   (files, symbols, call_graph, imports, FTS5 BM25 index)
```

### Four phases

| Phase | What | Parallelism |
|-------|------|-------------|
| 1 — Parse | Tree-sitter AST traversal per file | ThreadPoolExecutor (parse_workers=4) |
| 2 — Write | Symbol + chunk insertion, parent_id remapping | Sequential (DB consistency) |
| 3 — Embed | Token-aware batch embedding + TPM rate limiting | Batch API calls |
| 4 — Resolve | Cross-file call edges, import paths, type edges, Angular selectors | Sequential |

---

## Retrieval Pipeline (per query)

```
User Query
  └─ QueryPlanner (optional LLM: 8 intents → RetrievalStrategy)
       ├─ Vector Search   (HyDE snippet → ANN)
       ├─ BM25 Search     (FTS5, pre-cleaned tokens)
       └─ Grep Search     (exact / regex symbol names)
            └─ RRF Fusion (Reciprocal Rank Fusion, k=60)
                 └─ Graph Expansion
                      ├─ call_graph (callers + callees, PageRank)
                      ├─ import_graph (forward/reverse, configurable depth)
                      └─ type_edges (extends/implements/trait_impl/embedded)
                           └─ Reranker (Cohere | cross-encoder)
                                └─ Context Assembler (greedy | breadth_first, token budget)
                                     └─ LLM Synthesis (trelix ask)  ← optional
```

### 8 intent types

| Intent | Legs | Graph | Rerank top-n | Assembly |
|--------|------|-------|--------------|----------|
| `symbol_lookup` | grep + BM25 + vector | call (depth 1) | 20 | greedy |
| `file_overview` | file-direct | none | — | greedy |
| `feature_flow` | vector + BM25 | call+import (depth 2) | 30 | greedy |
| `project_overview` | file-direct | none | — | greedy |
| `comparison` | all 3 | call+import (depth 1) | 35 | greedy |
| `config_lookup` | file-direct + grep | none | — | greedy |
| `dependency_map` | vector + BM25 | import forward (depth 2) | 30 | breadth_first |
| `blast_radius` | grep + vector + BM25 | import reverse (depth 1) | 40 | breadth_first |

---

## Store

Single SQLite file (`.trelix/index.db`) — zero external infrastructure.

| Table | Purpose |
|-------|---------|
| `files` | Indexed files with SHA-256 hash for incremental updates |
| `symbols` | Extracted symbols (function, class, method, …) with line spans |
| `call_graph` | Directed call edges (caller_id → callee_id) |
| `imports` | File-level import edges |
| `type_edges` | Inheritance / implements / trait edges |
| `chunks` | Embeddable text (context header + symbol body) |
| `symbols_fts` | FTS5 virtual table for BM25 full-text search |
| `vec_chunks` | sqlite-vec vector table for ANN search |

---

## Embedding providers

| Provider | Key required | Model | Dimensions |
|----------|-------------|-------|------------|
| `local` | None | `all-MiniLM-L6-v2` | 384 |
| `openai` | `OPENAI_API_KEY` | `text-embedding-3-large` | 3072 |
| `azure` | `AZURE_API_KEY` + endpoint | `text-embedding-3-large` | 3072 |
