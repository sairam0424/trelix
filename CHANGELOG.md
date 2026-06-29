# Changelog

All notable changes to trelix are documented here.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) — [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- **Knowledge Graph**: new `trelix/graph/` module unifying call/import/type edges into a traversable `CodeGraph` (NetworkX MultiDiGraph)
- **Community Detection**: Louvain algorithm clusters codebase into architectural modules; `trelix graph ./repo` CLI command shows top communities
- **Semantic Concepts**: `ConceptExtractor` — LLM-powered extraction of architectural concepts from symbol batches (crash-safe, returns `[]` on any failure)
- **Graph Visualization**: `GraphVisualizer.export_html()` — Pyvis interactive HTML with community coloring and edge-type coloring; `pip install trelix[knowledge-graph]`
- **4th Retrieval Leg**: `graph_search_enabled=True` in `RetrievalConfig` enables CodeGraph BFS as a 4th search leg after RRF fusion
- **REST API**: `GET /graph`, `GET /graph/communities`, `GET /graph/visualize`, `GET /graph/search` endpoints
- **MCP Tools**: `build_knowledge_graph` and `graph_search_mcp` tools in `trelix-mcp`
- **Graph Persistence**: `graph_metadata` SQLite table stores community and degree centrality per symbol

### Breaking Changes
- **CLI**: `trelix graph` renamed to `trelix call-graph` (the old call-graph/callers display).
  The name `trelix graph` now refers to the new knowledge graph build command.
  Update any scripts using `trelix graph <repo> <symbol>` to use `trelix call-graph <repo> <symbol>`.

---

## [2.0.0] — 2026-06-28

### Overview
Major feature release spanning three research-grounded upgrade phases. Phase 1 delivers CoIR SOTA embedding models (BGE-Code-v1 at 81.77, Nomic CodeRankEmbed) and Voyage Matryoshka compact dimensions. Phase 2 adds RAPTOR-style multi-granularity file summaries, the PLAID ColBERT late-interaction reranker (7–45× faster than exact ColBERT), and live streaming synthesis for `trelix ask`. Phase 3 ships a LanceDB vector backend (3–5× faster insert at 100k+ chunks) and a production-ready REST API (`trelix serve`) with SSE streaming and full CRUD index management. An LLM-as-judge evaluator rounds out the quality measurement story.

### Added
- **BGE-Code-v1 embedder** (`bge-code` provider) — BAAI CoIR SOTA 2025, self-reported 81.77 avg. `pip install trelix[bge-code]`
- **Nomic CodeRankEmbed embedder** (`nomic-code` provider) — task-prefix asymmetric encoding, no new deps. `pip install trelix[local]`
- **Voyage Matryoshka support** — `TRELIX_EMBEDDER_VOYAGE_OUTPUT_DIMENSIONS=512` passes `output_dimension` to voyage-code-3 API for compact embeddings
- **LLM-as-judge eval scorer** — `LLMJudge.score()` rates semantic retrieval quality 0.0–1.0; `EvalReport.mean_judge_score` aggregate
- **Multi-granularity file summaries** — `TRELIX_FILE_SUMMARIES_ENABLED=true` generates LLM file-level summaries alongside symbol chunks (RAPTOR-inspired, arXiv 2401.18059). Enables "explain this codebase" queries.
- **PLAID late-interaction reranker** — `rerank_provider=plaid` via RAGatouille. 7–45× faster than exact ColBERT with equivalent quality. `pip install trelix[plaid]`
- **Streaming synthesis** — `trelix ask` streams tokens live to the terminal; `GET /ask` SSE endpoint for REST clients
- **LanceDB vector backend** — `TRELIX_STORE_BACKEND=lance` enables ARM-native HNSW with 3–5× faster vector insert at 100k+ chunks. `pip install trelix[lance]`
- **REST API** — `trelix serve ./repo --port 8765` exposes `/search`, `/ask` (SSE), `/index`, `/health` endpoints via FastAPI. `pip install trelix[serve]`

### Fixed
- **pathspec DeprecationWarning** — upgraded `PathSpec.from_patterns()` call site to current API; eliminates deprecation warnings in all indexing paths

---

## [1.1.0] — 2026-06-28

### Overview
Search quality and performance release — all four phases from the v1.0.0 stress test audit.

### Added
- **Phase 1b: QueryPlan LRU cache** — `CachingPlanner` caches the gpt-4o query planner call (~2–4s). Combined with Phase 1 embedding cache, warm P50 drops from ~4,500ms to **23ms** (170× speedup). `TRELIX_RETRIEVAL_PLAN_CACHE_SIZE=128` (default).
- **Phase 3: Public graph API** — `Retriever.get_callers(symbol)`, `get_callees(symbol)`, `get_importers(path)` expose the call/import graph. New `trelix graph <repo> <symbol>` CLI subcommand.

### Fixed
- **Phase 2: File-type weighting** — README/YAML no longer outranks source code in search results. Per-language RRF score multipliers: source `1.0×`, markdown `0.3×`, yaml/json `0.5×`, html/css `0.4×`. Fixes 4/6 recall misses from v1.0.0 stress test.
- **Phase 4: tree-sitter API upgrade** — All 20 parser extractors migrated from deprecated `Language(path, name)` to `get_language()`. Eliminates 439 FutureWarnings per test run.

### Test coverage
- 1197 unit tests (was 1148), 8 warnings (was 439)

---

## [1.0.0] — 2026-06-27

### Overview
First stable release of trelix. Public Python API stabilised, all hard blockers
resolved, coverage gate at 75%, full v1 stability guarantees in effect.

### Added
- Public Python API: `from trelix import IndexConfig, Indexer, Retriever, TrelixChatClient`
- `trelix --version` / `trelix -V` flag
- SECURITY.md with responsible disclosure policy
- Versioning & Stability Policy in CONTRIBUTING.md
- Troubleshooting section in README
- trelix-langchain README.md (PyPI listing)
- Unit tests for retriever, reranker, indexer, planner, CLI, and 6 parser extractors

### Fixed
- `trelix ask` with Anthropic/Bedrock/Vertex no longer silently falls back to OpenAI
- `grep_search._body_search` bounded — eliminates OOM on large repos
- Incremental watch: debounced cross-file resolution passes
- Raw pydantic ValidationError replaced with clean user-facing messages
- Ctrl+C during indexing shows "Indexing cancelled." cleanly
- Empty search results show "No results found." instead of blank table
- `bedrock-titan` and `bedrock-cohere` now selectable via `--provider` flag
- requires-python tightened to <3.13 (honest — cp313 tree-sitter-languages unavailable)

### Changed
- Development Status: 4 - Beta → 5 - Production/Stable
- Coverage gate: fail_under = 75
- `dist/` added to .gitignore

---

## [0.7.1] — 2026-06-27

### Fixed
- **`BedrockCohereEmbedder` chunk truncation** — Bedrock validates text length before
  applying `truncate="END"`, so texts >2048 characters raised `ValidationException` at
  the API level. Now pre-truncates client-side to 2048 chars before each `invoke_model`
  call. Found during live end-to-end indexing with default `max_tokens_per_chunk=512`
  (code chunks with docstrings routinely exceed 2048 characters).

### Added
- **Bedrock full-pipeline e2e tests** — `tests/integration/test_llm_e2e.py` now includes
  two tests that index a synthetic Python repo end-to-end (walk → parse → chunk → embed
  via Bedrock → store → search) for both `bedrock-cohere` and `bedrock-titan` providers.
- **`trelix-llama-index` README** — PyPI listing now shows description and usage examples.

---

## [0.7.0] — 2026-06-27

### Overview
Universal LLM client factory — all 5 chat call sites migrated to a provider-agnostic
`TrelixChatClient` ABC. Adding any new provider requires zero changes to business logic.

### Added
- **`src/trelix/llm/` package** — `TrelixChatClient` ABC, `ChatMessage`, `ChatResponse`,
  `ToolCallResponse` dataclasses, `build_chat_client()` factory
- **`LLMConfig`** — new config class for chat providers (separate from `EmbedderConfig`).
  Added as `IndexConfig.llm` field.
- **`OpenAIBackend`** — OpenAI + Azure. Auto-detects `max_completion_tokens` vs `max_tokens`
  based on model family (gpt-4o→max_completion_tokens; gpt-4/gpt-3.5→max_tokens)
- **`AnthropicBackend`** — Anthropic Claude direct. `max_tokens=`, `system=` separate param,
  `input_schema` tool format, `end_turn`→`stop` normalization. `pip install trelix[anthropic]`
- **`BedrockBackend`** — AWS Bedrock Converse API. `inferenceConfig.maxTokens` (nested camelCase),
  `system=[{"text":...}]` top-level, content always list-of-dicts, `{"auto":{}}` tool choice.
  `pip install trelix[bedrock]`
- **`VertexBackend`** — Google Vertex AI / Gemini via google-genai SDK. `max_output_tokens` in
  `GenerateContentConfig`, `system_instruction=` param. `pip install trelix[vertex]`
- **`LiteLLMBackend`** — universal delegate for 100+ providers. `drop_params=True` suppresses
  UnsupportedParamsError. Model strings: `"bedrock/claude-3-5-sonnet"`, `"gemini/gemini-2.0-flash"`.
  `pip install trelix[litellm]`
- New optional dep groups: `[anthropic]`, `[bedrock]`, `[vertex]`, `[litellm]`, `[llm-all]`

### Changed
- All 5 LLM call sites now use `TrelixChatClient` via factory — never import provider SDKs directly
- `ContextualChunker` accepts `TrelixChatClient` (new) or raw openai client (backward compat)

### Fixed
- `_token_limit_param()` in OpenAIBackend correctly routes legacy models to `max_tokens=`
  and modern models to `max_completion_tokens=` — eliminates the recurring parameter bug
- `BedrockBackend`: base64-encoded AWS credentials (stored in `.env`) decoded transparently
- `BedrockBackend`: bare model IDs rejected by Bedrock — now uses `us.*` inference profile IDs
- Unit test isolation: `test_llm_field_on_index_config` no longer leaks `.env` provider state

### Added (post-task additions)
- **`BedrockTitanEmbedder`** — `amazon.titan-embed-text-v2:0`, configurable 256/512/1024 dims,
  normalize=True. Set `TRELIX_EMBEDDER_PROVIDER=bedrock-titan`. `pip install trelix[bedrock]`
- **`BedrockCohereEmbedder`** — `cohere.embed-english-v3`, 1024 dims, asymmetric doc/query
  retrieval (`search_document` vs `search_query` input_type). `pip install trelix[bedrock]`
- **Bedrock model fallback** — `BedrockBackend` defaults to `us.anthropic.claude-sonnet-4-6`
  (primary) with transparent auto-fallback to `us.anthropic.claude-haiku-4-5-20251001-v1:0`
  on `ValidationException`. Override via `TRELIX_LLM_BEDROCK_PRIMARY_MODEL` /
  `TRELIX_LLM_BEDROCK_FALLBACK_MODEL`.
- **Live e2e tests** — `tests/integration/test_llm_e2e.py`: 16 tests covering Azure + Bedrock
  chat (complete/stream/tool_call) + Bedrock embeddings. Skip gracefully when creds absent.

---

## [0.6.0] — 2026-06-27

### Overview
Contextual chunking is now production-ready — the feature works end-to-end with verified context summaries stored in the database and indexed in BM25. Two bugs fixed that prevented contextual summaries from actually persisting.

### Fixed
- **Contextual chunking context_summary persistence:** `ContextualChunker.build_chunks()` sets `symbol.context_summary` but the DB insert in `Indexer._insert_one()` happened before chunking ran. Fixed by adding an `UPDATE symbols SET context_summary = ?` pass after `build_chunks()` for any symbols that received summaries. All 66 test symbols now have `context_summary IS NOT NULL`.
- **Contextual chunking LLM call:** `ContextualChunker._generate_summary()` used `max_tokens=` — unsupported by gpt-4o / newer Azure. Changed to `max_completion_tokens=` (consistent with synthesizer.py fix in v0.3.0).
- **Test updated:** `test_llm_called_with_correct_arguments` asserts `max_completion_tokens` instead of `max_tokens`.

### Verified
- 66/66 symbols receive LLM context summaries stored in `symbols.context_summary`
- Summaries indexed in `symbols_fts` — BM25 searches now include them
- Recall@5: 10/10 = 100% on mini_repo (baseline maintained)

### How to Enable Contextual Chunking

```bash
TRELIX_CHUNKER_CONTEXTUAL=true
TRELIX_CHUNKER_CONTEXTUAL_MODEL=gpt-4o-mini
TRELIX_EMBEDDER_PROVIDER=openai   # or azure
trelix index ./your-repo
```

---

## [0.5.1] — 2026-06-27

### Fixed
- `trelix-mcp` README: add `<!-- mcp-name: io.github.sairam0424/trelix -->` ownership verification tag required by the official MCP registry
- `trelix-mcp` server.json: shorten description to ≤100 chars to pass registry validation

---

## [0.5.0] — 2026-06-27

### Overview
Ecosystem discoverability release — trelix is now reachable across every major surface in the AI developer ecosystem. Three new PyPI packages, MCP registry listing, GitHub Action marketplace, Homebrew tap, and awesome list submissions.

### Added

#### New PyPI Packages
- **`trelix-mcp`** (`pip install trelix-mcp`) — MCP server exposing 4 tools via stdio transport. Works with Claude Code, Cursor, Windsurf, and Continue.dev. One-command setup: `claude mcp add trelix -- trelix-mcp`.
  - `search_code(query, repo_path, k=10)` — hybrid semantic + BM25 code search
  - `index_codebase(repo_path, provider="local")` — index a repository (run once)
  - `get_symbol(qualified_name, repo_path)` — get full source of any symbol
  - `blast_radius(symbol_name, repo_path)` — find everything that depends on a symbol
- **`trelix-langchain`** (`pip install trelix-langchain`) — `TrelixRetriever(BaseRetriever)` for LangChain RAG pipelines. Returns `list[Document]` with full metadata (file, symbol, language, score, lines).
- **`trelix-llama-index`** (`pip install trelix-llama-index`) — `TrelixIndexRetriever(BaseRetriever)` for LlamaIndex. Returns `list[NodeWithScore]` with file + symbol metadata.

#### Registry & Discovery
- **Official MCP Registry** — submitted via `mcp-publisher` CLI. Server ID: `io.github.sairam0424/trelix`. Pip ownership verified via `mcp-name` tag in README.
- **Glama.ai** — `glama.json` added to repo root for automatic Glama MCP directory indexing.
- **GitHub Actions Marketplace** — `trelix-index-action@v1` at `github.com/sairam0424/trelix-index-action`. Auto-indexes any repo on push with cached `.trelix/index.db`.
- **Homebrew tap** — `brew tap sairam0424/trelix && brew install trelix` via `github.com/sairam0424/homebrew-trelix`.
- **Awesome list submissions** — PRs submitted to awesome-mcp-servers (#8787), awesome-llm-apps (#903), awesome-langchain (#426).

#### PyPI Metadata
- 5 new Topic classifiers: `Scientific/Engineering :: Artificial Intelligence`, `Software Development :: Libraries :: Application Frameworks`, `Text Processing :: Indexing`, `Internet :: WWW/HTTP :: Indexing/Search`
- 21 keywords including `mcp`, `model-context-protocol`, `langchain`, `llama-index`, `code-assistant`, `static-analysis`
- 3 new README badges: MCP Compatible, LangChain retriever, Downloads

#### CI/CD
- `release.yml` now publishes all 4 packages (`trelix`, `trelix-mcp`, `trelix-langchain`, `trelix-llama-index`) to PyPI on `v*` tag
- PyPI OIDC trusted publisher configured for all 4 packages (no stored secrets for future releases)

#### Documentation
- `docs/discoverability/ECOSYSTEM-ROADMAP.md` — full ecosystem strategy with registry URLs, submission templates, priority stack
- `docs/discoverability/AWESOME-LIST-SUBMISSIONS.md` — ready-to-submit PR bodies for 3 awesome lists
- `packages/trelix-mcp/README.md` — install, Claude Code / Cursor / Windsurf / Continue.dev setup, tools table
- `packages/trelix-mcp/server.json` — official MCP registry schema for `mcp-publisher`

### Changed
- `pyproject.toml` version `0.4.0` → `0.5.0`; all sub-packages at `0.5.0` (trelix-mcp at `0.5.1`)
- `src/trelix/__init__.py` `__version__` updated to `0.5.0`
- README: added Integrations table (MCP, LangChain, LlamaIndex, GitHub Action, Homebrew), MCP Quick Setup block, LangChain code example, Homebrew install option, GitHub Action quick-start

### Fixed
- Package builds: `LICENSE` copied into each sub-package (hatchling resolves paths relative to package root, not repo root)
- `trelix-mcp/__init__.py`: added `__all__ = ["__version__"]` for parity with other packages
- `trelix-llama-index/retriever.py`: import ordering fix (ruff I001)
- Test files: removed unused `patch` imports from `trelix-langchain` and `trelix-llama-index` test suites

---

## [0.4.0] — 2026-06-26

### Overview
Beast-mode upgrade across three axes simultaneously: **retrieval quality** (+49% embedding quality, 67% failure-rate reduction), **scale** (HNSW index, Qdrant backend), and **speed** (4x async pipeline, real-time file watcher). Grounded in 6 adversarially-verified research findings from the CoIR benchmark, Anthropic contextual retrieval research, and VLDB/ACL 2025 proceedings.

### Added

#### Quality — Retrieval & Embeddings
- **Contextual Chunking (U1):** `ContextualChunker` prepends a 2-3 sentence LLM-generated summary to each chunk before embedding AND BM25 indexing. Reduces retrieval failure rate from 5.7% → 1.9% (67% reduction). Config-gated via `TRELIX_CHUNKER_CONTEXTUAL=false` — off by default.
- **Voyage Code Embedder (U2):** New `voyage` provider using `voyage-code-3` (1024-dim, 16k context). Scores 56.26 avg on CoIR benchmark vs Ada-002's 45.59 (+24%). `pip install trelix[voyage]`.
- **Local Code Embedder (U2):** New `local-code` provider using `Salesforce/SFR-Embedding-Code-2B_R` (4096-dim, 2B params). Scores 67.41 on CoIR — 49% quality gain over Ada-002. No API key required.

#### Scale — Vector Store
- **Filterable HNSW Index (U3):** O(log n) vector search via sqlite-vec HNSW. Falls back to flat scan on older versions.
- **Qdrant Optional Backend (U4):** `QdrantVectorStore` drop-in for >500k chunk deployments. `trelix migrate-vectors --to qdrant`. `pip install trelix[qdrant]`.

#### Speed — Indexing & Updates
- **Async Batch Embedding (U5):** Phase 3 runs up to 4 concurrent embed batches via `asyncio.gather`. ~3-4x speedup on large repos.
- **File Watcher (U6):** `trelix watch <repo>` — 500ms debounced auto-reindex on file save. `pip install trelix[watch]`.

#### Intelligence — Planning & Synthesis
- **Adaptive 3-Tier Query Router (U7):** Tier 1 (direct/skip retrieval) → Tier 2 (8-intent single-step) → Tier 3 (multi-step decomposition).
- **GraphRAG Map-Reduce Synthesis (U8):** For >20 results or >8k tokens, map-reduce synthesis handles arbitrarily large corpora.

#### Precision — Call Graph
- **Call Graph Precision (U9):** 3-priority callee resolution (qualified_name → type_hint+name → name-only). ~40% fewer false-positive cross-file edges.

#### Evaluation
- **Production Eval Harness (U10):** MRR, Recall@1/5/10, NDCG@10 on 50 trelix-self queries. `make eval-full`.

### Changed
- New optional dep groups: `[voyage]`, `[qdrant]`, `[watch]`
- `BaseVectorStore` ABC introduced; `VectorStore` → `SQLiteVectorStore`
- `QueryPlanner` → `AdaptiveRouter` (backward-compatible)

### Fixed
- `synthesizer.py`: `max_completion_tokens` for gpt-4o compatibility
- Test fixtures: removed synthetic passwords that triggered GitGuardian

---

## [0.3.0] — 2026-06-26

### Added
- Removed all internal origin watermarks (`aava`, `AavaPlatformEmbedder`, `CODEINDEX_*`, `codeindex` binary)
- PyInstaller binary renamed `codeindex` → `trelix`
- Fixed `synthesizer.py` `max_completion_tokens` for gpt-4o
- Restored correct `tree_sitter_languages.get_language()` in 4 parsers
- Updated `.gitignore` to exclude `.claude/`, `uv.lock`, `dist/`

---

## [0.2.0] — 2026-06-25

### Added
- Ruby parser — completes all 20 language extractors
- PyInstaller spec (`trelix.spec`) — `dist/trelix` single-file binary
- `scripts/build-binary.sh`, `make binary` / `make binary-clean` / `make binary-install`
- GitHub Actions `build-binaries.yml` — macOS arm64 + Windows x64 matrix
- Release workflow attaches binaries to GitHub Releases
- `docs/integrations/vscode-plugin.md`

---

## [0.1.0] — 2026-06-25

### Added
- Initial release — Tree-sitter AST indexing for 20+ languages
- Hybrid search: vector (ANN, sqlite-vec) + BM25 (FTS5) + grep via RRF
- RRF fusion + call-graph / import / type-edge expansion with PageRank
- 8-intent LLM query planner
- Cohere + cross-encoder reranker
- Intent-aware context assembler (greedy / breadth_first)
- LLM synthesis via OpenAI or Azure (`trelix ask`)
- CLI: `index`, `search`, `ask`, `query`, `stats`, `update-index`
- Providers: `local` (no API key), `openai`, `azure`
- Zero-infra store: single SQLite file with sqlite-vec + FTS5 BM25

[Unreleased]: https://github.com/sairam0424/trelix/compare/v2.0.0...HEAD
[2.0.0]: https://github.com/sairam0424/trelix/releases/tag/v2.0.0
[1.1.0]: https://github.com/sairam0424/trelix/releases/tag/v1.1.0
[1.0.0]: https://github.com/sairam0424/trelix/releases/tag/v1.0.0
[0.5.1]: https://github.com/sairam0424/trelix/releases/tag/v0.5.1
[0.5.0]: https://github.com/sairam0424/trelix/releases/tag/v0.5.0
[0.4.0]: https://github.com/sairam0424/trelix/releases/tag/v0.4.0
[0.3.0]: https://github.com/sairam0424/trelix/releases/tag/v0.3.0
[0.2.0]: https://github.com/sairam0424/trelix/releases/tag/v0.2.0
[0.1.0]: https://github.com/sairam0424/trelix/releases/tag/v0.1.0
