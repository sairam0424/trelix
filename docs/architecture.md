# Trelix Architecture (v2.4.0)

## v2.4.0 Additions

### FederatedRetriever TTL Cache (Plan C)

```
FederatedRetriever.retrieve(query, k=10)
  └─ _make_cache_key(query, sorted_repos, k)  → SHA-256
       ├─ cache HIT  → return cached list[SearchResult] (avg <1ms)
       └─ cache MISS → _query_repos() fan-out → TTLCache(120s)
```

**Constructor:** `FederatedRetriever(registry, cache_ttl=120.0)`

- Thread-safe via `threading.Lock`; safe for concurrent async callers
- `cache_ttl=0` disables caching entirely (useful for tests / real-time sessions)
- `cache_stats()` returns `{hits, misses, size}` for observability
- `clear_cache()` flushes all entries immediately
- Typical cache hit rate: ~90% for debugging sessions (repeated queries over the same repos)

### Multi-Query Expansion Observability (Plan B)

`MultiQueryExpander.expand()` now returns a frozen `ExpandResult` dataclass:

```python
@dataclass(frozen=True)
class ExpandResult:
    queries: list[str]       # the expanded query variants
    llm_used: str            # which LLM backend produced the expansion
    elapsed_ms: float        # wall-clock time for the LLM call
```

Three new nullable columns in `query_telemetry`:

| Column | Type | Description |
|--------|------|-------------|
| `expansion_used` | BOOLEAN | Whether multi-query expansion was applied |
| `expansion_variants` | INTEGER | Number of variant queries generated |
| `expansion_elapsed_ms` | REAL | LLM call duration for expansion |

`TelemetryWriter.record()` accepts `expansion_result=` kwarg; `NULL` stored when not provided (fully backward-compatible).

### `flare_max_retries` rename (Plan A)

`RetrievalConfig` field renamed for clarity:

| Old field | New field | Status |
|-----------|-----------|--------|
| `flare_max_iterations` | `flare_max_retries` | Old env var deprecated until v3.0.0 |

Env var migration:

| Env var | Status |
|---------|--------|
| `TRELIX_RETRIEVAL_FLARE_MAX_RETRIES` | **New (canonical)** |
| `TRELIX_RETRIEVAL_FLARE_MAX_ITER` | Deprecated — still accepted; emits `DeprecationWarning` |

### GitHub PR Review (Plan D)

New module: `src/trelix/review/github.py`

```
trelix review --pr owner/repo#N
  └─ GitHubPRClient.get_pr_files()  → list[PRFile] (7 status values)
       └─ DiffReviewer.review(diff_text=...)
            └─ [--post-comments] GitHubPRClient.post_review() → single batched API call
```

**Key details:**
- `GITHUB_TOKEN` env var only — no OAuth flow
- All 7 GitHub file status values handled: `added`, `modified`, `removed`, `renamed`, `copied`, `changed`, `unchanged`
- PRs with >3000 files emit a truncation warning; only the first 3000 files are reviewed
- `parse_pr_ref("owner/repo#N")` helper for CLI arg parsing
- `--post-comments` posts findings as a single batched GitHub review (one API call)

### MCP Pagination — BREAKING (Plan F)

`search_code()` return type changed:

**Before (v2.3.x and earlier):**
```python
search_code(query, repo_path, k=10) → list[dict]
```

**After (v2.4.0+):**
```python
search_code(query, repo_path, k=10, cursor=0) → {
    "results": list[dict],
    "next_cursor": int | None,   # null on last page
    "total_available": int
}
```

**Migration:** replace `for item in response` with `for item in response["results"]`.

`index_codebase()` now emits `ctx.report_progress()` notifications via `asyncio` during indexing (no interface change for callers).

---

## Indexing Pipeline (offline — `trelix index`)

```
Repository
  └─ FileWalker           (.gitignore-aware, SHA-256 change detection)
       └─ Tree-sitter Parser  (20 languages → symbols + call/import/type edges)
            ├─ ContextualChunker  (LLM context summary + breadcrumb header)
            │    └─ Embedder  (voyage | local-code | azure | openai | local |
            │    │             bedrock-titan | bedrock-cohere | bge-code | nomic-code)
            │         └─ sqlite-vec HNSW  (O(log n) ANN — or Qdrant / LanceDB for >500k)
            └─ SQLite DB   (files, symbols, call_graph, imports, FTS5 BM25)
```

### Four phases

| Phase | What | Parallelism |
|-------|------|-------------|
| 1 — Parse | Tree-sitter AST traversal per file | ThreadPoolExecutor (parse_workers=4) |
| 2 — Write | Symbol + chunk insertion, parent_id remapping | Sequential (DB consistency) |
| 3 — Embed | Token-aware batch embedding (4 concurrent async API calls) | `asyncio.gather` + `Semaphore(4)` |
| 4 — Resolve | Cross-file call edges (qualified-name priority), imports, type edges | Sequential |

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

`ContextualChunker` accepts `TrelixChatClient` (v0.7.0) or a raw OpenAI client (backward compat).

---

## Retrieval Pipeline (per query — `trelix search` / `trelix ask`)

```
User Query
  └─ FederatedRetriever (v2.4.0: TTL cache, SHA-256 key, threading.Lock)
       ├─ cache HIT  → return cached list[SearchResult] (avg <1ms)
       └─ cache MISS →
            └─ AdaptiveRouter
                 ├─ Tier 1: Direct — trivial factual → skip retrieval
                 ├─ Tier 2: Single-step — 8-intent classification → RetrievalStrategy
                 └─ Tier 3: Multi-step — LLM decomposes → 2-3 sub-queries in parallel
                      └─ Per sub-query:
                           ├─ Vector Search   (HyDE snippet → sqlite-vec HNSW ANN)
                           ├─ Contextual BM25 (FTS5, includes context_summary)
                           └─ Grep Search     (exact / regex symbol names)
                                └─ RRF Fusion (Reciprocal Rank Fusion, k=60)
                                     └─ Graph Expansion
                                          ├─ call_graph (qualified-name + type-hint precision)
                                          ├─ import_graph (forward/reverse, depth 1-2)
                                          └─ type_edges (extends/implements/trait_impl)
                                               └─ Reranker (Cohere | cross-encoder | PLAID)
                                                    └─ Context Assembler (greedy | breadth_first)
                                                         └─ Synthesis (via TrelixChatClient)
                                                              ├─ ≤8k tokens: Direct LLM call
                                                              └─ >8k tokens: GraphRAG map-reduce
                                                                   └─ TTLCache(120s) stored
```

### Adaptive Query Router

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

### GraphRAG Synthesis

Activated when `len(results) > 20` OR `total_tokens > 8000`:

```
results (N > 20)
  MAP:    split into groups of ~10 results (~3k tokens each)
          → LLM answers each group: "Partially answer: {query}\n{group_context}"
  REDUCE: merge partial answers
          → LLM synthesizes final: "Synthesize these partial answers: {partial_answers}"
```

### Streaming Synthesis

`Synthesizer.stream(query, context)` returns an `Iterator[str]` — each yielded token
is flushed immediately to the caller. Used by the `/ask` SSE endpoint and the
`trelix ask --stream` CLI flag. GraphRAG map-reduce switches to non-streaming for
the MAP phase (parallel LLM calls) and streams only the final REDUCE step.

---

## LLM Client Factory (v0.7.0)

All chat/synthesis LLM call sites use a provider-agnostic `TrelixChatClient` ABC.
No business logic file imports a provider SDK directly.

```
LLMConfig  ──▶  build_chat_client()  ──▶  TrelixChatClient (ABC)
                                              ├── OpenAIBackend    (OpenAI / Azure)
                                              ├── AnthropicBackend (Claude direct)
                                              ├── BedrockBackend   (AWS Bedrock Converse)
                                              ├── VertexBackend    (Google Vertex AI)
                                              └── LiteLLMBackend   (100+ providers)
```

**Provider selection:** `TRELIX_LLM_PROVIDER` env var (default: `openai`).

### TrelixChatClient interface

```python
class TrelixChatClient(ABC):
    def complete(messages: list[ChatMessage], **kwargs) -> ChatResponse
    def stream(messages: list[ChatMessage], **kwargs) -> Iterator[str]
    def tool_call(messages: list[ChatMessage], tools: list[dict], **kwargs) -> ToolCallResponse
```

### Backend details

| Backend | Provider | Key behaviours | Install |
|---------|----------|---------------|---------|
| `OpenAIBackend` | OpenAI + Azure | Auto-detects `max_completion_tokens` vs `max_tokens` by model family (gpt-4o → `max_completion_tokens`; gpt-4/gpt-3.5 → `max_tokens`) | base package |
| `AnthropicBackend` | Anthropic Claude | `max_tokens=`, `system=` separate param, `input_schema` tool format, `end_turn` → `stop` normalization | `trelix[anthropic]` |
| `BedrockBackend` | AWS Bedrock Converse | `inferenceConfig.maxTokens` (nested camelCase), `system=[{"text":...}]` top-level, content always list-of-dicts, `{"auto":{}}` tool choice. Base64-encoded credentials decoded transparently. Bare model IDs rejected — uses `us.*` inference profile IDs. | `trelix[bedrock]` |
| `VertexBackend` | Google Vertex AI / Gemini | `max_output_tokens` in `GenerateContentConfig`, `system_instruction=` param | `trelix[vertex]` |
| `LiteLLMBackend` | 100+ providers | `drop_params=True` suppresses UnsupportedParamsError. Model strings: `"bedrock/claude-3-5-sonnet"`, `"gemini/gemini-2.0-flash"` | `trelix[litellm]` |

### Bedrock model selection

Primary model: `us.anthropic.claude-sonnet-4-6` (override: `TRELIX_LLM_BEDROCK_PRIMARY_MODEL`)

Automatic fallback to `us.anthropic.claude-haiku-4-5-20251001-v1:0` on `ValidationException`
(override: `TRELIX_LLM_BEDROCK_FALLBACK_MODEL`). Fallback is transparent — no caller change needed.

### LLM call sites (all migrated in v0.7.0, extended in v2.0.0)

All call sites use `TrelixChatClient` via `build_chat_client()`:

1. `ContextualChunker` — per-symbol context summary generation
2. `Indexer` — coordinating chunker during indexing
3. `Synthesizer` — final answer synthesis
4. `QueryPlanner` / `AdaptiveAgent` — query decomposition and intent classification
5. `GraphRAGSynthesizer` — map-reduce partial answer generation

### LLM config env vars (v0.7.0)

| Variable | Default | Description |
|----------|---------|-------------|
| `TRELIX_LLM_PROVIDER` | `openai` | `openai` \| `azure` \| `anthropic` \| `bedrock` \| `vertex` \| `litellm` |
| `TRELIX_LLM_MODEL` | `gpt-4o` | Model override for the selected provider |
| `TRELIX_LLM_BEDROCK_PRIMARY_MODEL` | `us.anthropic.claude-sonnet-4-6` | Bedrock primary model |
| `TRELIX_LLM_BEDROCK_FALLBACK_MODEL` | `us.anthropic.claude-haiku-4-5-20251001-v1:0` | Bedrock auto-fallback model |
| `ANTHROPIC_API_KEY` | — | Anthropic direct API key |
| `GOOGLE_CLOUD_PROJECT` | — | Google Cloud project for Vertex AI |
| `GOOGLE_API_KEY` | — | Google AI Studio key (alternative to service account) |
| `AWS_ACCESS_KEY_ID` | — | AWS credentials for Bedrock |
| `AWS_SECRET_ACCESS_KEY` | — | AWS credentials for Bedrock |
| `AWS_REGION` | `us-east-1` | AWS region for Bedrock |

---

## Store

### SQLite (default, zero-infra)

Single file (`.trelix/index.db`) with WAL mode + FTS5 + sqlite-vec HNSW.

| Table | Columns | Purpose |
|-------|---------|---------|
| `files` | id, path, rel_path, language, hash, size_bytes | File tracking; SHA-256 for incremental |
| `symbols` | id, file_id, name, qualified_name, kind, line_start, line_end, signature, body, docstring, **context_summary**, decorators, is_public, parent_id | All code symbols |
| `calls` | id, caller_id, callee_name, callee_id, line, **callee_type_hint** | Call graph edges with precision |
| `imports` | id, file_id, imported_from, imported_names, imported_file_id | Import edges |
| `type_edges` | id, from_symbol_id, to_type_name, edge_kind, to_symbol_id | Inheritance / trait / embed |
| `chunks` | id, symbol_id, chunk_text, token_count | Embeddable text units |
| `file_summaries` | id, file_id, summary, created_at | LLM-generated 2–4 sentence file descriptions (RAPTOR) |
| `symbols_fts` | FTS5 virtual table over name, qualified_name, docstring, body, **context_summary** | BM25 keyword search |
| `vec_chunks` | sqlite-vec HNSW virtual table | ANN vector search |
| `graph_metadata` | symbol_id INTEGER PRIMARY KEY, community INTEGER, centrality REAL, node_type TEXT | Knowledge-graph community assignments and degree centrality (v2.0.0) |
| `graph_concepts` | name TEXT, category TEXT, importance REAL, source_symbol_ids TEXT | LLM-extracted architectural concepts (v2.0.0, optional) |

### Qdrant (optional — for >500k chunks)

Drop-in `QdrantVectorStore` via `BaseVectorStore` ABC. Set `TRELIX_STORE_BACKEND=qdrant`.

Collection config: HNSW m=16, ef_construct=200, filterable by file_id.

Migration: `trelix migrate-vectors --to qdrant --url http://localhost:6333`

### LanceDB (optional — embedded columnar store)

Drop-in `LanceDBVectorStore` via `BaseVectorStore` ABC. Set `TRELIX_STORE_BACKEND=lancedb`.

LanceDB stores all chunk vectors in a Lance columnar format under `.trelix/lancedb/`.
File-level summary vectors use a **sentinel chunk_id** of `-(file_id)` — a negative file
ID that cannot collide with any real chunk row — so summary entries and code-chunk entries
share the same table without a separate column.

```python
# sentinel convention
SENTINEL_CHUNK_ID = -(file_id)   # e.g. file_id=42 → chunk_id=-42
```

Install: `pip install trelix[lancedb]`  
Migration: `trelix migrate-vectors --to lancedb --path .trelix/lancedb`

---

## Embedding Providers

| Provider | Model | Dim | CoIR Score | Install |
|----------|-------|-----|-----------|---------|
| `local` | all-MiniLM-L6-v2 | 384 | baseline | `trelix[local]` |
| `local-code` | SFR-Embedding-Code-2B_R | 4096 | **67.41** | `trelix[local-code]` |
| `openai` | text-embedding-3-large | 3072 | ~45 | base package |
| `azure` | text-embedding-3-large | 3072 | ~45 | base package |
| `voyage` | voyage-code-3 | 1024 | **56.26** | `trelix[voyage]` |
| `bedrock-titan` | amazon.titan-embed-text-v2:0 | 256/512/1024 | — | `trelix[bedrock]` |
| `bedrock-cohere` | cohere.embed-english-v3 | 1024 | — | `trelix[bedrock]` |
| `bge-code` | BGE-Code-v1 | 1536 | **63.10** | `trelix[bge-code]` |
| `nomic-code` | nomic-embed-code (CodeRankEmbed) | 768 | **58.40** | `trelix[nomic-code]` |

CoIR (Code Information Retrieval) benchmark — ACL 2025. Higher is better.

### BGE-Code-v1 (v2.0.0)

**BGECodeEmbedder** (`bge-code`): FlagEmbedding-based code embedder producing 1536-dim vectors.
Instruction-tuned for code retrieval — prepend `"Represent this code for retrieval: "` at
query time (handled internally). Lazy-loaded on first call.

```bash
pip install trelix[bge-code]   # installs FlagEmbedding + torch
```

### Nomic CodeRankEmbed (v2.0.0)

**NomicCodeEmbedder** (`nomic-code`): sentence-transformers wrapper around
`nomic-ai/nomic-embed-code` (CodeRankEmbed). 768-dim, ~137M params, Apache 2.0 license.
Asymmetric: query prefix `"search_query: "` / document prefix `"search_document: "` applied
automatically. Runs fully offline — no API key required.

```bash
pip install trelix[nomic-code]   # installs sentence-transformers + torch
```

### Bedrock embedders (v0.7.0)

**BedrockTitanEmbedder** (`bedrock-titan`): configurable dimensions (256/512/1024) via
`TRELIX_EMBEDDER_BEDROCK_TITAN_DIMENSIONS`. Default 1024 matches Voyage quality; 256 cuts
storage 4x for large repos. `normalize=True` by default.

**BedrockCohereEmbedder** (`bedrock-cohere`): asymmetric doc/query retrieval — uses
`search_document` input_type at index time and `search_query` at query time for maximum
retrieval precision.

Both reuse `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_REGION` from `.env` —
no extra credentials beyond what `BedrockBackend` already requires.

---

## Reranking

### Cohere / cross-encoder (existing)

`CohereReranker` and `CrossEncoderReranker` are the default reranking backends,
selected via `TRELIX_RERANKER` env var (`cohere` | `cross-encoder`).

### PLAID Reranker (v2.0.0)

**PlaidReranker** uses [RAGatouille](https://github.com/bclavie/RAGatouille) to run a
ColBERT late-interaction reranker (PLAID index engine) locally.

Key behaviours:
- **Lazy model loading**: the ColBERT model is loaded on the first `rerank()` call, not at
  import time — cold start is ~2s on first use, instant thereafter.
- **Graceful fallback**: if `ragatouille` is not installed or model loading fails, the
  reranker falls back to score-passthrough (results returned in their original order) and
  logs a warning rather than raising.
- **from_pretrained() API**: model is loaded via
  `RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")` (override:
  `TRELIX_RERANKER_PLAID_MODEL`).

```bash
pip install trelix[plaid]   # installs ragatouille + faiss-cpu
```

```python
# env var selection
TRELIX_RERANKER=plaid
TRELIX_RERANKER_PLAID_MODEL=colbert-ir/colbertv2.0   # optional override
```

---

## Call Graph Precision

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

## File-Level Summaries — RAPTOR (v2.0.0)

`FileSummarizer` generates a concise 2–4 sentence LLM description of each file during
indexing, inspired by the RAPTOR hierarchical retrieval approach (abstractive summarisation
at multiple granularities).

### Storage

- **DB table**: `file_summaries` (columns: `id`, `file_id`, `summary`, `created_at`)
- **Vector index**: each summary is embedded and stored alongside chunk vectors.  
  In the LanceDB backend the sentinel `chunk_id = -(file_id)` is used so no separate
  table or column is needed. In the SQLite backend a dedicated `vec_file_summaries`
  virtual table is used.

### Retrieval integration

File-summary vectors participate in the `file_overview` and `project_overview` retrieval
strategies — they are retrieved first, then individual chunk results are merged in.

### Feature flag

Gated by `TRELIX_FILE_SUMMARIES_ENABLED` (default: `false`). Set to `true` to enable
during indexing. Re-indexing is required when the flag is first enabled on an existing
index.

```bash
TRELIX_FILE_SUMMARIES_ENABLED=true trelix index .
```

---

## File Watcher (`trelix watch` / `trelix watch-all`)

```
trelix watch <repo> [--provider local|openai|azure|voyage|bedrock-titan|bedrock-cohere]
  → full index on startup
  → watchdog Observer monitors all files
  → on_modified/on_created: debounce 500ms → indexer.index_file(path)
  → on_deleted: remove file + symbols + chunks + vectors from DB
  → respects .gitignore (reuses FileWalker.should_ignore())
  → Ctrl+C to stop
```

Requires `pip install trelix[watch]` (watchdog).

### Multi-Repo Watcher (`trelix watch-all`) — v2.4.0

```
RepoRegistry (registered repos)
  └─ MultiRepoWatcher
       └─ watchfiles.awatch(*all_repo_paths, debounce=1600ms)
            ├─ Change.modified/added → MD5 hash guard → Indexer.index_file()
            └─ Change.deleted        → db.delete_file_by_path() + vector_store cleanup
```

**Key behaviours:**
- Single `watchfiles.awatch()` call monitors all registered repositories simultaneously (vs. one-per-repo watchdog observers)
- MD5 hash guard prevents re-index cascades on no-op saves
- Deleted files are removed from both SQLite and the vector store atomically
- Per-repo stats displayed in the terminal; graceful `Ctrl+C` shutdown
- Source: `src/trelix/indexing/multi_watcher.py`

```bash
trelix watch-all   # watches all repos registered in the active RepoRegistry
```

---

## Test Coverage (v2.4.0)

| Suite | Count | What's covered |
|-------|-------|---------------|
| Unit tests (core) | **1,467** | All modules, all parsers, all providers, LLM factory, cache, multi-watcher, GitHub PR client, MCP pagination |
| Integration tests (live) | **16** | Azure + Bedrock chat (complete/stream/tool_call) + Bedrock embeddings; skip gracefully when creds absent |
| Eval harness | 50 queries | MRR, Recall@1/5/10, NDCG@10 on trelix-self; LLM-as-judge score per result |
| trelix-mcp tests | **41** | All tools including paginated search_code envelope, index_codebase progress notifications |
| trelix-langchain tests | **19** | BaseRetriever, Document structure, metadata keys |
| trelix-llama-index tests | **10** | BaseRetriever, NodeWithScore structure |

**Total passing: 1,508**

### LLM-as-Judge Eval (v2.0.0)

`LLMJudge` in `tests/eval/llm_judge.py` scores each retrieval result on a 0–1 relevance
scale using a secondary LLM call (default: `gpt-4o-mini`). Scores are integrated into
the eval harness pipeline:

- `EvalResult.judge_score` — per-query float (0.0–1.0)
- `EvalReport.mean_judge_score` — aggregate mean across all 50 eval queries

`LLMJudge` is optional — the eval harness runs without it if `TRELIX_EVAL_LLM_JUDGE=false`
(default: `false`). When enabled, it adds ~$0.02 per full eval run at gpt-4o-mini pricing.

Python 3.11 and 3.12 supported.

---

## REST API (v2.0.0)

`trelix.api.app.create_app()` returns a FastAPI application that exposes the full
retrieval + indexing surface over HTTP. Start it with:

```bash
pip install trelix[api]   # installs fastapi + uvicorn + sse-starlette
uvicorn trelix.api.app:app --host 0.0.0.0 --port 8080
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/search` | Keyword + vector hybrid search; body: `{query, repo_path, k}` |
| `POST` | `/ask` | SSE streaming answer; body: `{query, repo_path}`; response: `text/event-stream` |
| `POST` | `/index` | Trigger (re-)indexing of a repo; body: `{repo_path, provider}` |
| `GET`  | `/health` | Liveness check; returns `{"status": "ok"}` |
| `GET`  | `/stats` | Index statistics (file count, symbol count, chunk count, last_indexed) |

The `/ask` endpoint uses `Synthesizer.stream()` internally — each SSE `data:` event is a
single yielded token, terminated by a `data: [DONE]` sentinel.

---

## Ecosystem Packages (v2.0.0)

| Package | PyPI | Purpose |
|---------|------|---------|
| `trelix` | [pypi.org/project/trelix](https://pypi.org/project/trelix/) | Core library + CLI |
| `trelix-mcp` | [pypi.org/project/trelix-mcp](https://pypi.org/project/trelix-mcp/) | MCP server — Claude Code, Cursor, Windsurf |
| `trelix-langchain` | [pypi.org/project/trelix-langchain](https://pypi.org/project/trelix-langchain/) | LangChain `BaseRetriever` |
| `trelix-llama-index` | [pypi.org/project/trelix-llama-index](https://pypi.org/project/trelix-llama-index/) | LlamaIndex `BaseRetriever` |

### Install options

```bash
pip install trelix               # OpenAI + Azure (default)
pip install trelix[anthropic]    # + Anthropic direct
pip install trelix[bedrock]      # + AWS Bedrock (chat + both Bedrock embedders)
pip install trelix[vertex]       # + Google Vertex AI
pip install trelix[litellm]      # + LiteLLM (100+ providers)
pip install trelix[llm-all]      # all LLM providers
pip install trelix[local]        # + local sentence-transformers (no API key)
pip install trelix[local-code]   # + SFR-Embedding-Code-2B_R (no API key, ~8GB RAM)
pip install trelix[bge-code]     # + BGE-Code-v1 embedder (FlagEmbedding, no API key)
pip install trelix[nomic-code]   # + Nomic CodeRankEmbed embedder (no API key)
pip install trelix[voyage]       # + Voyage AI code embeddings
pip install trelix[rerank]       # + Cohere reranker
pip install trelix[plaid]        # + PLAID/ColBERT reranker (RAGatouille)
pip install trelix[qdrant]       # + Qdrant vector backend
pip install trelix[lancedb]      # + LanceDB columnar vector backend
pip install trelix[api]          # + FastAPI REST server (uvicorn + sse-starlette)
pip install trelix[watch]        # + file watcher (watchdog)
pip install trelix[knowledge-graph]  # + knowledge graph (pyvis, networkx)
pip install trelix[graph-viz]        # alias for trelix[knowledge-graph]
pip install trelix[all]              # everything
```

### MCP Server Tools

```
trelix-mcp (stdio transport)
  ├── search_code(query, repo_path, k=10, cursor=0)                           → {results, next_cursor, total_available}  [BREAKING v2.4.0]
  ├── index_codebase(repo_path, provider)                                      → dict (stats) + progress notifications   [v2.4.0]
  ├── get_symbol(qualified_name, repo_path)                                    → dict | None
  ├── blast_radius(symbol_name, repo_path)                                     → list[dict]
  ├── build_knowledge_graph(repo_path, detect_communities, extract_concepts)   → dict (stats)  [v2.0.0]
  └── graph_search_mcp(query, repo_path, depth, max_results)                  → list[dict]    [v2.0.0]
```

**Migration for `search_code` callers:** replace `for item in response` with `for item in response["results"]`. Use `cursor=response["next_cursor"]` to fetch the next page; `next_cursor=null` indicates the last page.

---

## Knowledge Graph Layer

Added in **v2.0.0**. A new `trelix/graph/` module wraps the existing SQLite edge tables
into a unified Code Property Graph backed by NetworkX, with community detection, LLM
concept extraction, graph-aware BFS retrieval, and an interactive Pyvis visualizer.

**Breaking change**: the old `trelix graph <repo> <symbol>` call-graph display command is
renamed to `trelix call-graph <repo> <symbol>`. The `trelix graph` subcommand now builds
and queries the full knowledge graph.

### Graph Structure — CodeGraph (`trelix/graph/code_graph.py`)

`CodeGraph` holds a `networkx.MultiDiGraph`. Every symbol (function, class, method) and
file in the index is a node; every static relationship is a typed directed edge.

**Node attributes**

| Attribute | Type | Description |
|-----------|------|-------------|
| `name` | str | Simple symbol or file name |
| `qualified_name` | str | Fully-qualified identifier |
| `kind` | str | `function` \| `class` \| `method` \| `file` \| … |
| `file` | str | Source file path |
| `language` | str | Language detected by Tree-sitter |
| `community` | int | Community ID set after detection (default -1) |

**Edge types**

| Label | Source table | Direction |
|-------|-------------|-----------|
| `CALLS` | `calls` | caller → callee |
| `IMPORTS` | `imports` | file → imported\_file |
| `EXTENDS` | `type_edges` (extends) | child → parent |
| `IMPLEMENTS` | `type_edges` (implements) | implementor → interface |
| `TRAIT_IMPL` | `type_edges` (trait\_impl) | struct → trait |
| `EMBEDDED` | `type_edges` (embedded) | struct → embedded |

**Live stats on the trelix codebase (dry run)**

| Metric | Value |
|--------|-------|
| Nodes | 4,599 |
| Edges | 4,945 |
| Communities (Louvain) | 2,409 |
| Build time | 0.34 s |
| Top node degree | 438 (`parse`) |

### Community Detection (`trelix/graph/community.py`)

Three algorithms selectable at runtime:

| Algorithm | Complexity | Best for |
|-----------|-----------|---------|
| `louvain` (default) | O(n log n) | General use; fast on large graphs |
| `girvan_newman` | O(n³) | Quality-oriented; small/medium graphs |
| `label_prop` | O(n) | Very large graphs (>100k nodes) |

Output: a `{symbol_id: community_id}` mapping. After detection, each node's `community`
attribute is set in-memory and written to the `graph_metadata` SQLite table.

Communities represent logical module groupings (auth layer, data layer, API layer) inferred
purely from structural connectivity — no human labeling required.

### Graph Persistence (`trelix/graph/persistence.py`)

**Table: `graph_metadata`**

| Column | Type | Description |
|--------|------|-------------|
| `symbol_id` | INTEGER PRIMARY KEY | Foreign key to `symbols.id` |
| `community` | INTEGER | Community assignment from detection |
| `centrality` | REAL | Degree centrality computed at save time |
| `node_type` | TEXT | Node kind (mirrors `symbols.kind`) |

`GraphPersistence.save()` upserts all rows in a single transaction. Degree centrality is
pre-computed so graph-adjacent queries do not require loading the full graph.

### Semantic Concepts — optional (`trelix/graph/concepts.py`)

`ConceptExtractor` sends batches of symbol names and signatures to the configured LLM and
extracts architectural concepts (e.g. "JWT authentication", "event sourcing", "CQRS").

**Table: `graph_concepts`**

| Column | Type |
|--------|------|
| `name` | TEXT |
| `category` | TEXT |
| `importance` | REAL |
| `source_symbol_ids` | TEXT (JSON array) |

Enabled via `--concepts` flag on `trelix graph` or `ConceptExtractorConfig.enabled=True`.
**Crash-safe**: any LLM failure returns `[]` without aborting the graph build.

### Graph-Aware Search — 4th retrieval leg (`trelix/graph/search.py`)

BFS from the top results of the existing RRF fusion step, traversing the `CodeGraph` to
surface structurally adjacent symbols that plain vector/BM25/grep search misses.

**Score decay**: `score = 0.5^hop` — nodes one hop away score 0.5, two hops 0.25, etc.

**Config** (`RetrievalConfig`)

| Field | Default | Description |
|-------|---------|-------------|
| `graph_search_enabled` | `False` | Opt-in flag — zero impact when off |
| `graph_search_depth` | `2` | BFS depth from seed nodes |
| `graph_search_max_results` | `15` | Cap on graph-sourced results |

**Env var**: `TRELIX_GRAPH_SEARCH_ENABLED=true`

**Live retrieval mix** with `graph_search_enabled=True`: 30 results total
(5 graph, 19 vector, 4 BM25, 2 graph\_expansion).

When disabled, the retrieval pipeline is identical to v1.x — no performance regression.

### Visualization (`trelix/graph/visualizer.py`)

`GraphVisualizer.export_html()` writes a Pyvis interactive HTML file to
`<repo>/.trelix/graph.html`.

**Visual encoding**

| Attribute | Encoding |
|-----------|---------|
| Node color | Community % 10 → 10-color pastel palette |
| Node size | Proportional to degree (more connections = larger) |
| `CALLS` edges | Blue |
| `IMPORTS` edges | Purple |
| `EXTENDS` edges | Green |
| `IMPLEMENTS` edges | Teal |
| Physics layout | ForceAtlas2 |

**Security**: the output path is constrained to `<repo>/.trelix/` — path traversal
attempts (e.g. `../../etc/passwd`) are rejected with HTTP 400.

### REST Endpoints (added in v2.0.0)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/graph` | Return graph stats (node count, edge count, top-degree nodes) |
| `GET` | `/graph/communities` | Community listing with member counts |
| `GET` | `/graph/visualize` | Stream the Pyvis HTML file |
| `GET` | `/graph/search` | BFS graph search; params: `query`, `repo_path`, `depth` |

### MCP Tools (added in v2.0.0)

```
trelix-mcp (stdio transport)
  ├── build_knowledge_graph(repo_path, detect_communities, extract_concepts)  → dict (stats)
  └── graph_search_mcp(query, repo_path, depth, max_results)                  → list[dict]
```

### CLI

```bash
# Build graph (+ optional community detection and concept extraction)
trelix graph ./repo
trelix graph ./repo --visualize          # also write Pyvis HTML
trelix graph ./repo --concepts           # also run LLM concept extraction
trelix graph ./repo --json               # machine-readable stats to stdout

# Old call-graph display (RENAMED — breaking change)
trelix call-graph ./repo MyClass.method
```

### Install

```bash
pip install 'trelix[knowledge-graph]'   # pyvis>=0.3.2, networkx>=3.3.0
pip install 'trelix[graph-viz]'         # alias for the same extras
```

---

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
