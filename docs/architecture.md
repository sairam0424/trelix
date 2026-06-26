# Trelix Architecture (v0.5.0)

## Indexing Pipeline (offline — `trelix index`)

```
Repository
  └─ FileWalker           (.gitignore-aware, SHA-256 change detection)
       └─ Tree-sitter Parser  (20 languages → symbols + call/import/type edges)
            ├─ ContextualChunker  (LLM context summary + breadcrumb header)
            │    └─ Embedder  (voyage | local-code | azure | openai | local)
            │         └─ sqlite-vec HNSW  (O(log n) ANN — or Qdrant for >500k)
            └─ SQLite DB   (files, symbols, call_graph, imports, FTS5 BM25)
```

### Four phases

| Phase | What | Parallelism |
|-------|------|-------------|
| 1 — Parse | Tree-sitter AST traversal per file | ThreadPoolExecutor (parse_workers=4) |
| 2 — Write | Symbol + chunk insertion, parent_id remapping | Sequential (DB consistency) |
| 3 — Embed | Token-aware batch embedding (v0.4.0: 4 concurrent async API calls) | `asyncio.gather` + `Semaphore(4)` |
| 4 — Resolve | Cross-file call edges (qualified-name priority, v0.4.0), imports, type edges | Sequential |

### Contextual Chunking (v0.4.0)

When `TRELIX_CHUNKER_CONTEXTUAL=true`, each symbol gets an LLM-generated 2-3 sentence summary prepended to its chunk text before embedding and BM25 indexing. The summary is stored in `symbols.context_summary` and indexed in `symbols_fts`.

```
Symbol body
  → LLM call (gpt-4o-mini): "Describe what this code does in 2-3 sentences"
  → context_summary: "This function validates a username/password pair..."
  → chunk_text: "{context_summary}\n\n# File: ...\n{symbol_body}"
  → embedded + stored in FTS5
```

Research basis: Anthropic contextual retrieval (2024) — 67% retrieval failure reduction.

---

## Retrieval Pipeline (per query — `trelix search` / `trelix ask`)

```
User Query
  └─ AdaptiveRouter (v0.4.0)
       ├─ Tier 1: Direct — trivial factual → skip retrieval
       ├─ Tier 2: Single-step — 8-intent classification → RetrievalStrategy
       └─ Tier 3: Multi-step — LLM decomposes → 2-3 sub-queries in parallel
            └─ Per sub-query:
                 ├─ Vector Search   (HyDE snippet → sqlite-vec HNSW ANN)
                 ├─ Contextual BM25 (FTS5, includes context_summary, v0.4.0)
                 └─ Grep Search     (exact / regex symbol names)
                      └─ RRF Fusion (Reciprocal Rank Fusion, k=60)
                           └─ Graph Expansion
                                ├─ call_graph (qualified-name + type-hint precision, v0.4.0)
                                ├─ import_graph (forward/reverse, depth 1-2)
                                └─ type_edges (extends/implements/trait_impl)
                                     └─ Reranker (Cohere | cross-encoder)
                                          └─ Context Assembler (greedy | breadth_first)
                                               └─ Synthesis
                                                    ├─ ≤8k tokens: Direct LLM call
                                                    └─ >8k tokens: GraphRAG map-reduce (v0.4.0)
```

### Adaptive Query Router (v0.4.0)

Replaces the fixed single-LLM-call planner with a 3-tier router:

| Tier | Detection | Behavior |
|------|-----------|---------|
| 1 — Direct | Regex: `what is X`, `define X`, `list all` | Skip retrieval entirely; LLM answers directly |
| 2 — Single-step | Default | 8-intent classification → pre-baked RetrievalStrategy |
| 3 — Multi-step | Long queries with `walk me through`, `end-to-end`, `step by step` | LLM decomposes into 2-3 focused sub-queries; each retrieved independently; merged before rerank |

### 8 intent types (Tier 2)

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

### GraphRAG Synthesis (v0.4.0)

Activated when `len(results) > 20` OR `total_tokens > 8000`:

```
results (N > 20)
  MAP:    split into groups of ~10 results (~3k tokens each)
          → LLM answers each group: "Partially answer: {query}\n{group_context}"
  REDUCE: merge partial answers
          → LLM synthesizes final: "Synthesize these partial answers: {partial_answers}"
```

---

## Store

### SQLite (default, zero-infra)

Single file (`.trelix/index.db`) with WAL mode + FTS5 + sqlite-vec HNSW.

| Table | Columns | Purpose |
|-------|---------|---------|
| `files` | id, path, rel_path, language, hash, size_bytes | File tracking; SHA-256 for incremental |
| `symbols` | id, file_id, name, qualified_name, kind, line_start, line_end, signature, body, docstring, **context_summary** (v0.4.0), decorators, is_public, parent_id | All code symbols |
| `calls` | id, caller_id, callee_name, callee_id, line, **callee_type_hint** (v0.4.0) | Call graph edges with precision |
| `imports` | id, file_id, imported_from, imported_names, imported_file_id | Import edges |
| `type_edges` | id, from_symbol_id, to_type_name, edge_kind, to_symbol_id | Inheritance / trait / embed |
| `chunks` | id, symbol_id, chunk_text, token_count | Embeddable text units |
| `symbols_fts` | FTS5 virtual table over name, qualified_name, docstring, body, **context_summary** (v0.4.0) | BM25 keyword search |
| `vec_chunks` | sqlite-vec HNSW virtual table | ANN vector search |

### Qdrant (optional, v0.4.0 — for >500k chunks)

Drop-in `QdrantVectorStore` via `BaseVectorStore` ABC. Set `TRELIX_STORE_BACKEND=qdrant`.

Collection config: HNSW m=16, ef_construct=200, filterable by file_id.

Migration: `trelix migrate-vectors --to qdrant --url http://localhost:6333`

---

## Embedding Providers

| Provider | Model | Dim | CoIR Score | Install |
|----------|-------|-----|-----------|---------|
| `local` | all-MiniLM-L6-v2 | 384 | baseline | `trelix[local]` |
| `local-code` | SFR-Embedding-Code-2B_R | 4096 | **67.41** | `trelix[local-code]` |
| `openai` | text-embedding-3-large | 3072 | ~45 | base package |
| `azure` | text-embedding-3-large | 3072 | ~45 | base package |
| `voyage` | voyage-code-3 | 1024 | **56.26** | `trelix[voyage]` |

CoIR (Code Information Retrieval) benchmark — ACL 2025. Higher is better.

---

## Call Graph Precision (v0.4.0)

Callee resolution uses 3-priority matching:

```
1. Exact qualified_name match     → callee_id set (highest confidence)
2. name + callee_type_hint match  → callee_id set (receiver annotation extracted at parse time)
3. name match, unique             → callee_id set (existing behavior)
4. name match, ambiguous          → callee_id = NULL (better than wrong)
```

`callee_type_hint` is extracted from receiver type annotations at parse time:
- Python: `user_service: UserService` → calls to `user_service.login()` get `callee_type_hint="UserService"`
- TypeScript: `const auth: AuthService` → `auth.verify()` gets `callee_type_hint="AuthService"`

Expected impact: ~40% reduction in false-positive cross-file call edges.

---

## File Watcher (v0.4.0 — `trelix watch`)

```
trelix watch <repo> [--provider local|openai|azure|voyage]
  → full index on startup
  → watchdog Observer monitors all files
  → on_modified/on_created: debounce 500ms → indexer.index_file(path)
  → on_deleted: remove file + symbols + chunks + vectors from DB
  → respects .gitignore (reuses FileWalker.should_ignore())
  → Ctrl+C to stop
```

Requires `pip install trelix[watch]` (watchdog).

---

## Test Coverage (v0.5.0)

| Suite | Count | What's covered |
|-------|-------|---------------|
| Unit tests (core) | **860** | All modules, all parsers, all new features |
| Integration tests | **39** | Full pipeline, recall eval, CLI |
| Eval harness | 50 queries | MRR, Recall@1/5/10, NDCG@10 on trelix-self |
| trelix-mcp tests | **9** | 4 tools, stdout-clean MCP protocol test |
| trelix-langchain tests | **19** | BaseRetriever, Document structure, metadata keys |
| trelix-llama-index tests | **10** | BaseRetriever, NodeWithScore structure |
| **Total** | **987** | |

---

## Ecosystem Packages (v0.5.0)

| Package | PyPI | Purpose |
|---------|------|---------|
| `trelix` | [pypi.org/project/trelix](https://pypi.org/project/trelix/) | Core library + CLI |
| `trelix-mcp` | [pypi.org/project/trelix-mcp](https://pypi.org/project/trelix-mcp/) | MCP server — Claude Code, Cursor, Windsurf |
| `trelix-langchain` | [pypi.org/project/trelix-langchain](https://pypi.org/project/trelix-langchain/) | LangChain `BaseRetriever` |
| `trelix-llama-index` | [pypi.org/project/trelix-llama-index](https://pypi.org/project/trelix-llama-index/) | LlamaIndex `BaseRetriever` |

### MCP Server Tools

```
trelix-mcp (stdio transport)
  ├── search_code(query, repo_path, k=10)      → list[dict]
  ├── index_codebase(repo_path, provider)       → dict (stats)
  ├── get_symbol(qualified_name, repo_path)     → dict | None
  └── blast_radius(symbol_name, repo_path)      → list[dict]
```

### Integration Surface

```
Claude Code / Cursor / Windsurf / Continue.dev
  └── pip install trelix-mcp
      └── claude mcp add trelix -- trelix-mcp

LangChain RAG pipeline
  └── pip install trelix-langchain
      └── TrelixRetriever(repo_path=".").invoke(query)

LlamaIndex RAG pipeline
  └── pip install trelix-llama-index
      └── TrelixIndexRetriever(repo_path=".").retrieve(QueryBundle(query))

GitHub Actions CI
  └── uses: sairam0424/trelix-index-action@v1

Homebrew (macOS)
  └── brew tap sairam0424/trelix && brew install trelix
```
