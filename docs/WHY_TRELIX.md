# Why Trelix

> **The thesis:** Every competitor asks "how do I autocomplete the next token?" Trelix asks "which code across your entire codebase is actually relevant to answering this question — and how does it connect to everything else?"

Code search is a retrieval problem, not an autocomplete problem. When you ask "how does authentication work?" you do not need the next 20 tokens. You need the five files, three call paths, and two import chains that together constitute your authentication layer — retrieved accurately, ranked precisely, and synthesized into an answer that cites real code, not hallucinated stubs.

Trelix is the retrieval engine that makes that possible: a 7-leg hybrid pipeline built on Tree-sitter AST parsing, call-graph traversal, and FLARE-guided re-retrieval, packaged as a single SQLite file that runs completely offline.

---

## The Core Problem Trelix Solves

**Existing tools treat code search as text search. Trelix treats it as graph traversal over a structured program representation.**

When you have a 50-file project, grepping works. When you have 50,000 files, 20 languages, and a question that spans 3 layers of abstraction, you need:

1. **Structural awareness** — understand what a function is, who calls it, what it imports, what type hierarchy it belongs to
2. **Multi-signal fusion** — no single retrieval method is best; the right answer comes from combining semantic vectors, BM25 keyword frequency, call-graph traversal, sparse SPLADE embeddings, and exact-match grep
3. **Zero-infrastructure deployment** — a local indexer that runs offline, stores everything in one file, and does not require a cloud account or a Kubernetes cluster
4. **Provider independence** — your retrieval engine should not be hostage to a single LLM vendor

None of this exists in the existing code-search tooling available today.

---

## What Trelix Provides (Unique Capabilities)

### 1. Zero-Infrastructure Deployment

**What it does:** The entire index — vectors, BM25, call graph, import graph, file summaries — lives in a single SQLite file (`.trelix/index.db`) using WAL mode, FTS5, and sqlite-vec HNSW. No Docker, no Redis, no Postgres, no cloud account required.

**Implementation:**
- sqlite-vec HNSW provides O(log n) approximate nearest neighbor search inside a stock SQLite file
- FTS5 provides BM25 over symbol names, qualified names, docstrings, bodies, and LLM-generated context summaries
- Call graph, import graph, and type edges stored as normalized SQLite tables — no graph database needed
- Everything under `.trelix/` — gitignore it or commit it, your choice

**Why it matters:** Cursor and Copilot require a running IDE extension connected to a cloud backend. Sourcegraph Cody requires a Sourcegraph instance. Trelix runs `trelix index ./repo` and produces a portable file you can put anywhere, query from scripts, or check into CI.

**Who else has it (OSS):** Nobody provides this combination of HNSW vector search + BM25 + call graph in a single embedded file. pgvector DIY requires Postgres. Sourcegraph requires a full server stack.

---

### 2. 7-Leg Hybrid Retrieval

**What it does:** Every query fans out across up to 7 independent retrieval legs simultaneously, fuses results via Reciprocal Rank Fusion (RRF, k=60), applies graph expansion, and reranks with Cohere, cross-encoder, or PLAID ColBERT.

**The 7 legs:**

| Leg | Source tag | What it finds |
|-----|-----------|---------------|
| 1 | `vector` | Semantically similar chunks (HNSW ANN over embeddings) |
| 2 | `bm25` | Keyword-frequency matches (FTS5 over name, body, context\_summary) |
| 3 | `grep` | Exact or regex symbol name matches (zero-latency SQL LIKE) |
| 4 | `file_summary` | Files matching at the 2–4 sentence LLM summary level (RAPTOR-style) |
| 5 | `graph_search` | Structurally adjacent symbols (CodeGraph BFS, score decay 0.5^hop) |
| 6 | `sparse` | SPLADE-Code learned sparse vectors (handles BM25 subword-fragmentation) |
| 7 | `sub_chunk` | Block- and statement-level granularity (multi-granularity indexing) |

**Why it matters:** GitHub Copilot uses dense vector search. Sourcegraph Cody uses BM25 + vector. Cursor uses embedding search with some call-context. No tool combines all 7 legs. The difference is not marginal — each leg surfaces different results. RRF fusion rewards results that appear across multiple legs, giving the final ranking far higher precision than any single leg alone.

**Who else has it (OSS):** Nobody. This is the most comprehensive multi-leg code retrieval pipeline available in any open-source tool.

---

### 3. Tree-sitter AST Parsing — 20+ Languages

**What it does:** Trelix parses every source file with Tree-sitter, extracting qualified names, call edges, import edges, type hierarchy edges (extends/implements/trait_impl/embedded), line spans, signatures, docstrings, and receiver type annotations — not just text.

**Implementation:**
- 20 languages: Python, TypeScript/TSX, JavaScript/JSX, Go, Java, Rust, C, C++, C#, Kotlin, Ruby, plus Razor, MSBuild, JSON/TOML/YAML config, Markdown, HTML, CSS/SCSS
- Qualified-name extraction: every symbol stored as `module.Class.method`, not just `method`
- Callee resolution uses 3-priority matching: exact qualified\_name → name + type hint annotation → name-unique fallback; ambiguous matches stored as NULL rather than wrong
- `callee_type_hint` extracted from receiver annotations at parse time: `user_service: UserService` → `auth.verify()` call gets `callee_type_hint="UserService"`, enabling ~40% reduction in false-positive cross-file call edges

**Why it matters:** Text-based retrieval finds files that mention "authentication." AST-based retrieval finds the exact `verify_token()` method, all 23 callers, the 4 files it imports, and the 2 classes it inherits from — because those are first-class database entries, not pattern matches over text.

**Who else has it (OSS):** Sourcegraph has deep language parsing for Go and TypeScript. No tool provides this breadth (20 languages) with call-graph precision (qualified-name + type-hint callee resolution) in a standalone offline library.

---

### 4. v2.4.0 — FederatedRetriever, MultiRepoWatcher, GitHub PR Review, MCP Pagination

**v2.4.0 shipped four independent capability upgrades:**

**FederatedRetriever TTL Cache (Plan C):**
- SHA-256 cache key over (query, sorted repo list, k)
- Thread-safe via `threading.Lock`; safe for concurrent async callers
- `cache_ttl=120s` default; `cache_ttl=0` disables entirely for tests
- Typical hit rate ~90% in debugging sessions (repeated queries over the same repos)
- `cache_stats()` returns `{hits, misses, size}` for observability
- Average hit latency <1ms vs. 50–500ms for a full retrieval fan-out

**Multi-Repo Watcher (Plan E):**
- Single `watchfiles.awatch()` call monitors all registered repositories simultaneously
- MD5 hash guard prevents re-index cascades on no-op saves
- Deleted files removed from SQLite and vector store atomically
- `trelix watch-all` — one command keeps every repo in sync

**GitHub PR Review (Plan D):**
- `trelix review --pr owner/repo#N` — retrieves diff, runs DiffReviewer over every changed file
- All 7 GitHub file status values handled correctly
- `--post-comments` posts findings as a single batched GitHub review API call
- PRs with >3000 files emit a truncation warning; 3000-file cap applied gracefully

**MCP Cursor Pagination (Plan F):**
- `search_code()` now returns `{results, next_cursor, total_available}` envelope
- Allows MCP clients (Claude Code, Cursor, Windsurf) to page through large result sets
- `index_codebase()` emits `ctx.report_progress()` notifications during indexing

**Why it matters:** The federated cache alone delivers a 50–500ms retrieval win on every repeated query in a session. The multi-repo watcher eliminates manual re-indexing across monorepos. GitHub PR review turns Trelix into a CI-native code reviewer. MCP pagination makes large repository search tractable from any MCP client.

---

### 5. LLM-Provider Agnostic — 6 Backends, 100+ Providers

**What it does:** Every LLM call site in Trelix — contextual chunking, query planning, synthesis, GraphRAG map-reduce — uses a single `TrelixChatClient` ABC. No business logic file imports a provider SDK directly. Switching providers is a single environment variable.

**The backends:**

| Backend | Provider | Key behaviour |
|---------|----------|---------------|
| `OpenAIBackend` | OpenAI + Azure | Auto-detects `max_completion_tokens` vs `max_tokens` per model family |
| `AnthropicBackend` | Anthropic Claude | `input_schema` tool format, `end_turn` normalization |
| `BedrockBackend` | AWS Bedrock Converse | `inferenceConfig.maxTokens` camelCase; base64 credential decoding; inference profile IDs enforced |
| `VertexBackend` | Google Vertex AI / Gemini | `max_output_tokens` in `GenerateContentConfig` |
| `LiteLLMBackend` | 100+ providers | `drop_params=True`; model strings `"bedrock/claude"`, `"gemini/gemini-2.0-flash"` |

**Bedrock auto-fallback:** Primary model (`us.anthropic.claude-sonnet-4-6`) falls back transparently to Haiku on `ValidationException`. No caller change required.

**DimensionGuard:** Prevents silent wrong-result bugs from provider switches that change embedding dimensions. If index dimension ≠ query dimension, Trelix fails loudly instead of returning garbage results ranked by cosine distance between incompatible vectors.

**Why it matters:** Cursor and Copilot are locked to a single provider. Switching from OpenAI to Bedrock in Trelix is `TRELIX_LLM_PROVIDER=bedrock` — one variable, no code changes, no new integration work.

**Who else has it (OSS):** LiteLLM provides the abstraction layer. Trelix is the first code-search tool to apply provider-agnosticism systematically across all LLM call sites including embedders, rerankers, chunkers, and synthesizers.

---

### 6. Python-Native Library — LangChain and LlamaIndex Retrievers

**What it does:** Trelix ships as a Python library, not just a CLI. It exposes a `LangChain BaseRetriever` and a `LlamaIndex BaseRetriever` as first-class packages, making it a drop-in retrieval backend for any LangChain RAG pipeline or LlamaIndex agent.

**Implementation:**

```python
# LangChain
from trelix_langchain import TrelixRetriever
retriever = TrelixRetriever(repo_path="/path/to/repo")
docs = retriever.invoke("how does authentication work?")

# LlamaIndex
from trelix_llama_index import TrelixIndexRetriever
from llama_index.core import QueryBundle
retriever = TrelixIndexRetriever(repo_path="/path/to/repo")
nodes = retriever.retrieve(QueryBundle("how does authentication work?"))
```

**Why it matters:** Copilot, Cursor, and Cody are IDE-native — you cannot call them from a Python script. Trelix retrieval can be embedded into agents, CI pipelines, evaluation harnesses, and data workflows because it is a library, not a service.

**Who else has it (OSS):** Some vector store libraries expose LangChain retrievers, but they are text-only. No tool provides a code-aware retriever (call graph, AST, BM25 + vector + grep) as a pip-installable LangChain component.

---

### 7. Fully Local Option — No API Key Required

**What it does:** `pip install "trelix[local]"` + `trelix index ./repo` works completely offline. The local embedder (`all-MiniLM-L6-v2`) runs via sentence-transformers with no API call. The local-code embedder (`SFR-Embedding-Code-2B_R`) is a 2B-parameter code-specialized model that also runs fully offline with no API key.

**Additional local-first embedding options:**
- `nomic-code`: Nomic CodeRankEmbed (768-dim, Apache 2.0, sentence-transformers, CoIR score 58.40)
- `bge-code`: BGE-Code-v1 (1536-dim, FlagEmbedding, CoIR score 63.10)
- `local-code`: SFR-Embedding-Code-2B_R (4096-dim, CoIR score 67.41 — highest quality local option)

**Why it matters:** Copilot requires a GitHub account. Cursor requires a Cursor account. Sourcegraph Cody requires a Sourcegraph account. Trelix requires nothing — it works in an air-gapped environment, behind a corporate firewall, or on a plane with no Wi-Fi. LLM synthesis (`trelix ask`) still needs an API key, but indexing and search (`trelix index`, `trelix search`) are fully local.

**Who else has it (OSS):** chroma-db and other vector stores allow local embedding. No code-specific tool combines local embedding with BM25 + call-graph retrieval in a zero-dependency local mode.

---

### 8. MCP-Native — 6 Tools, 3 Resources, 3 Prompts

**What it does:** `trelix-mcp` is a first-class MCP server that exposes Trelix retrieval to any MCP client (Claude Code, Cursor, Windsurf, Continue.dev). It speaks stdio transport and integrates in two commands.

**MCP tools:**

```
trelix-mcp (stdio transport)
  ├── search_code(query, repo_path, k=10, cursor=0)                              → {results, next_cursor, total_available}
  ├── index_codebase(repo_path, provider)                                         → dict (stats) + progress notifications
  ├── get_symbol(qualified_name, repo_path)                                       → dict | None
  ├── blast_radius(symbol_name, repo_path)                                        → list[dict]
  ├── build_knowledge_graph(repo_path, detect_communities, extract_concepts)      → dict (stats)
  └── graph_search_mcp(query, repo_path, depth, max_results)                     → list[dict]
```

**Setup:**
```bash
pip install trelix-mcp
claude mcp add trelix -- trelix-mcp
```

**Why it matters:** MCP is becoming the standard interface for AI tool use. Trelix's 7-leg retrieval is available inside Claude Code, Cursor, and Windsurf without any modification to those clients. The `blast_radius` tool — which finds all symbols that import or call a given symbol — is particularly powerful inside an IDE where you are about to refactor something.

**Who else has it (OSS):** Some tools expose ad-hoc MCP integrations. Trelix's MCP server exposes the full retrieval pipeline including call-graph, blast-radius analysis, and knowledge graph construction as first-class tools — not just a search wrapper.

---

### 9. Call-Graph Expansion — Callers and Callees of Any Symbol

**What it does:** After RRF fusion produces a ranked result list, Trelix expands each result by traversing the call graph in both directions — finding all functions that call a given symbol (callers) and all functions a given symbol calls (callees), up to configurable depth.

**Implementation:**
- Call graph stored as `calls(caller_id, callee_name, callee_id, callee_type_hint, line)` rows
- 3-priority callee resolution: qualified name exact match → name + type hint → name-unique → NULL for ambiguous
- 8 intent-specific expansion profiles (e.g. `symbol_lookup` uses depth 1 call expansion; `feature_flow` uses call+import depth 2; `blast_radius` uses import reverse depth 1)
- PageRank boost: symbols with high degree centrality (stored in `graph_metadata.centrality`) receive a 1.3× score multiplier at query time — no graph traversal needed, centrality pre-computed at graph-build time

**Why it matters:** "Find the authentication middleware" returns the middleware function. Call-graph expansion additionally surfaces every route handler that invokes the middleware, every helper the middleware calls, and every import the middleware file pulls in — the complete relevant context, automatically.

**Who else has it (OSS):** Sourcegraph has cross-reference navigation for some languages. No code search tool exposes call-graph traversal as a retrieval leg inside a ranked hybrid search pipeline.

---

### 10. FLARE Re-Retrieval Loop — Confidence-Gated Iterative Retrieval

**What it does:** During synthesis, Trelix monitors token-level generation probability. When the LLM emits a low-confidence span (probability below `TRELIX_FLARE_THRESHOLD`, default 0.3), synthesis pauses, uses the uncertain span as a new retrieval query, injects the supplemental results into context, and resumes generation.

**Implementation:**
```
Synthesis in progress
  → LLM token probability < FLARE_THRESHOLD?
       YES → extract uncertain span
             → re-retrieve (all enabled legs)
             → inject new results into context
             → resume synthesis
       NO  → continue normally
```

**Why it matters:** Standard RAG retrieves once before generation begins. If the initial retrieval missed relevant context, the LLM either hallucinates or produces a confident-sounding wrong answer. FLARE catches the uncertainty at generation time and retrieves targeted supplemental context before it can become a hallucination. This is the difference between "I don't have enough context" and "let me go look for more."

**Who else has it (OSS):** The FLARE technique is from a 2023 academic paper. Trelix is the first code-search tool to implement it as a production retrieval enhancement in a shipped open-source library.

---

### 11. DimensionGuard — Prevents Silent Wrong Results on Provider Switch

**What it does:** When you switch embedding providers (e.g., from OpenAI 3072-dim to local 384-dim), the existing index vectors have a different dimension than the new query embeddings. Without a guard, cosine distance comparisons silently return garbage results ranked by garbage scores. DimensionGuard detects the dimension mismatch at query time and fails loudly with a clear error message instead.

**Why it matters:** This is an easy mistake to make. You try a new provider, results seem worse, you spend hours debugging retrieval quality before realizing all your cosine similarity scores are mathematically meaningless because the vectors are from different spaces. DimensionGuard makes the failure mode obvious instead of invisible.

**Who else has it (OSS):** Nobody. This is defensive engineering — checking a precondition that the rest of the field skips.

---

## Competitive Comparison

| Capability | Trelix | GitHub Copilot Workspace | Cursor | Sourcegraph Cody | pgvector DIY | OpenAI File Search |
|------------|--------|--------------------------|--------|------------------|--------------|-------------------|
| **Vector search** | ✅ HNSW sqlite-vec | ✅ cloud | ✅ cloud | ✅ cloud | ✅ Postgres | ✅ cloud |
| **BM25 keyword search** | ✅ FTS5 | ❌ | partial | ✅ | manual | ❌ |
| **Exact/regex grep** | ✅ | ❌ | partial | ✅ | ❌ | ❌ |
| **Call-graph expansion** | ✅ 3-priority resolution | ❌ | partial | partial | ❌ | ❌ |
| **Import-graph expansion** | ✅ forward + reverse | ❌ | ❌ | partial | ❌ | ❌ |
| **File-summary retrieval leg** | ✅ RAPTOR-style | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Sparse (SPLADE) retrieval leg** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Tree-sitter AST parsing** | ✅ 20 languages | ❌ | partial | ✅ | ❌ | ❌ |
| **Qualified-name extraction** | ✅ | ❌ | partial | ✅ | ❌ | ❌ |
| **Zero-infra (single file)** | ✅ SQLite | ❌ cloud-only | ❌ cloud-only | ❌ server required | ❌ Postgres required | ❌ cloud-only |
| **Fully offline** | ✅ | ❌ | ❌ | ❌ | partial | ❌ |
| **LLM-provider agnostic** | ✅ 6 backends + LiteLLM | ❌ locked to OpenAI | ❌ locked to Cursor | ❌ locked to Anthropic | ❌ manual | ❌ locked to OpenAI |
| **Python library (pip install)** | ✅ | ❌ | ❌ | ❌ | partial | ❌ |
| **LangChain retriever** | ✅ | ❌ | ❌ | ❌ | partial | ❌ |
| **LlamaIndex retriever** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **MCP server (6 tools)** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **GitHub PR review** | ✅ v2.4.0 | ✅ cloud | partial | ❌ | ❌ | ❌ |
| **Multi-repo federated search** | ✅ FederatedRetriever | partial | ❌ | ✅ enterprise | ❌ | ❌ |
| **FLARE re-retrieval** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **DimensionGuard** | ✅ | N/A | N/A | N/A | ❌ | N/A |
| **Real-time file watching** | ✅ trelix watch | ❌ | ✅ | ✅ | ❌ | ❌ |
| **Knowledge graph (Louvain)** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **CoIR eval harness** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **GraphRAG map-reduce** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Self-hosted, open-source** | ✅ MIT | ❌ | ❌ closed-source | partial | ✅ | ❌ |

**Legend:** ✅ = implemented · partial = limited implementation · ❌ = not available · cloud-only = requires vendor cloud account

---

## Use Cases

### "I want to ask questions about an unfamiliar codebase without sending it to a cloud service"

**Fit: High.**

`pip install "trelix[local]" && trelix index ./repo && trelix ask ./repo "how does the request lifecycle work?"` — no API key, no cloud, no data egress. The local embedder produces competitive retrieval quality for most codebase exploration tasks. Add `--provider bge-code` or `--provider nomic-code` for code-specialized embeddings, still fully offline.

---

### "I'm building a RAG pipeline over my company's internal codebase and need a retriever"

**Fit: High.**

```python
from trelix_langchain import TrelixRetriever
retriever = TrelixRetriever(repo_path="/path/to/repo")
# Drop into any LangChain chain or LlamaIndex pipeline
docs = retriever.invoke("explain the payment processing flow")
```

Trelix handles indexing, hybrid retrieval, reranking, and result formatting. Your pipeline handles everything else. No custom retrieval code required.

---

### "I want AI-assisted GitHub PR review that understands the codebase context, not just the diff"

**Fit: High.**

```bash
trelix review --pr owner/repo#42 --post-comments
```

The reviewer retrieves codebase context related to each changed file, runs the diff through the DiffReviewer, and posts findings as a single batched GitHub review. Unlike generic LLM PR reviewers, Trelix retrieves relevant context from the indexed codebase before generating each comment — so it knows that the function being modified is called in 23 other places.

---

### "I need code search inside Claude Code / Cursor / Windsurf without leaving my editor"

**Fit: High.**

```bash
pip install trelix-mcp
claude mcp add trelix -- trelix-mcp
```

Trelix's 6 MCP tools are available inside any MCP-compatible client. The `blast_radius` tool is especially useful when you select a function and want to know everything that depends on it before refactoring.

---

### "I'm maintaining a monorepo with 15 services and want a single search interface"

**Fit: High.**

`FederatedRetriever` fans queries out to all registered repositories simultaneously, merges results, and caches the combined result set with a 120-second TTL. `trelix watch-all` keeps all repos indexed in real time. The ~90% cache hit rate means repeated queries in a debugging session are sub-millisecond.

---

### "I want to understand the architecture of a codebase before making changes"

**Fit: High.**

```bash
trelix graph ./repo --visualize   # Pyvis interactive HTML
trelix ask ./repo "explain the auth architecture"
```

The Knowledge Graph layer runs Louvain community detection on the call+import+type graph and clusters symbols into architectural modules (auth layer, data layer, API layer) purely from structural connectivity — no human labeling required. The interactive HTML visualization makes the module structure visually navigable.

---

### "I want the simplest possible code search that just works"

**Fit: Medium-High.**

`pip install "trelix[local]" && trelix index . && trelix search . "database pooling"` — three commands, returns a Rich table of results. The advanced capabilities (FLARE, GraphRAG, knowledge graph, multi-repo federation) are available but do not get in the way. If you do not need them, they are off by default.

If you just want IDE autocomplete, Copilot or Cursor is the right tool. Trelix earns its depth on retrieval-heavy tasks: answering architectural questions, building RAG pipelines, reviewing PRs, and navigating unfamiliar codebases.

---

## Scope and Boundaries

### What Trelix is

- **Retrieval engine** — finds the most relevant code for a given question or query
- **Index layer** — offline, incremental, AST-aware, stores everything in a single SQLite file
- **Synthesis tool** — `trelix ask` turns retrieved context into a cited answer
- **RAG component** — drop-in LangChain and LlamaIndex retriever for any pipeline
- **MCP server** — 6 tools available in Claude Code, Cursor, Windsurf
- **Library and CLI** — both surfaces first-class, both tested

### What Trelix is not

- An IDE extension with real-time autocomplete (Copilot and Cursor own that space)
- A code generation tool (Trelix retrieves; you or your LLM generates)
- A static analysis tool (Tree-sitter parsing is for retrieval, not linting or type-checking)
- A code review platform with UI (GitHub PR review is CLI-based and MCP-based, not a hosted product)
- A real-time streaming completions service (retrieval happens at query time, not keystroke time)

### Current version scope (v2.4.0)

| In v2.4.0 | Planned (v3.0+) |
|-----------|-----------------|
| 7-leg hybrid retrieval pipeline | Streaming retrieval with incremental result delivery |
| FederatedRetriever with TTL cache (~90% hit rate) | Native IDE plugin (VS Code extension) |
| 20-language Tree-sitter AST parsing | Cross-repo symbol resolution |
| 6 LLM backends + LiteLLM 100+ providers | Advanced taint analysis (Semgrep integration GA) |
| MCP server (6 tools, stdio transport) | Hosted cloud index option |
| GitHub PR review + batch comment posting | Multi-granularity indexing GA |
| Multi-repo watching (watchfiles.awatch) | Sub-chunk statement-level search GA |
| Knowledge graph (Louvain, Pyvis, BFS leg) | LLM-as-judge eval integration |
| FLARE confidence-gated re-retrieval | |
| DimensionGuard provider-switch protection | |
| CoIR eval harness (MRR, Recall, NDCG) | |
| LangChain + LlamaIndex retrievers | |
| REST API with SSE streaming | |

---

## Honest Caveats

**Trelix is not a real-time IDE plugin.** It is a CLI tool and MCP server. Queries go through `trelix search` or the MCP interface — there is no keystroke-level autocomplete. If you want token-by-token completion in your editor, Copilot or Cursor is the right tool. Trelix is built for retrieval-heavy tasks: answering questions, reviewing PRs, building RAG pipelines.

**Indexing is required upfront.** Before you can search, you must run `trelix index ./repo`. For a 50,000-file repository with OpenAI embeddings, this can take 5–20 minutes depending on API rate limits and concurrency. Incremental updates (`trelix watch`) keep the index current after the initial build, but there is no zero-setup instant-search mode.

**LLM synthesis (`trelix ask`) needs an API key.** `trelix index` and `trelix search` work fully offline with local embedders. But `trelix ask` — which synthesizes a natural-language answer from retrieved context — calls an LLM. You need at least one provider configured (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, AWS Bedrock credentials, etc.). If you only want search without synthesis, `trelix search` is completely offline.

**Beast-mode features compose but each has a cost.** FLARE adds 0–2 LLM calls per query. HyDE adds 1 LLM call per unique query (cached on repeat). Multi-query expansion adds N LLM calls per query (default N=3). GraphRAG map-reduce adds M LLM calls for large result sets. Enabling all features simultaneously on a slow or expensive LLM backend will increase latency and cost. The safe approach is to enable flags one at a time and validate with `trelix eval --golden`.

**The knowledge graph requires a separate build step.** `trelix graph ./repo` must be run (or re-run) to populate the `graph_metadata` table and enable graph-aware retrieval. `trelix watch` automatically patches the graph on file changes once built, but the initial graph build is not automatic.

---

## The Name

Trelix comes from **tree** (Tree-sitter AST parsing — the structural foundation of everything) and **relix** (retrieval + indexing). The core insight behind the name is the same insight behind the tool: code is not text. Code is a tree — a call tree, an import tree, a type hierarchy. Retrieving code well means traversing that tree, not scanning text.

The other tools search your codebase. Trelix understands it.
