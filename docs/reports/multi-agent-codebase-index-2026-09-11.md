# Trelix — End-to-End Codebase Index

> Generated 2026-09-11 via a 24-agent Claude Code workflow: 20 independent domain specialists (Read/Glob/Grep against the live repo, not docs), a completeness critic + docs-vs-code accuracy auditor, a dedicated LLM call-path tracer commissioned to close a critic-flagged gap, and one synthesis pass. Residual open gaps the critic flagged after this pass are listed in §6 — treat §5/§6 as living findings, not a closed audit.


## 1. System Overview

Trelix is a Python code-intelligence engine that turns a source repository into a queryable, LLM-augmented knowledge base. It walks a repo (respecting nested `.gitignore`s), parses every file with Tree-sitter into symbols/calls/imports/type-edges, chunks those symbols into embeddable units, embeds them with any of 9 pluggable providers, and stores everything — vectors, BM25 full-text, and the structured code graph — in a single SQLite file (`.trelix/index.db`, sqlite-vec `vec0` + FTS5). At query time a hybrid retrieval pipeline fuses up to 7 legs (vector, BM25, grep, sparse/SPLADE, file-summary, sub-chunk, graph-search) via Reciprocal Rank Fusion, expands across the call/import/type graph, reranks, and packs a token-budgeted context that an LLM (via one of 5 provider backends: OpenAI/Azure, Anthropic, Bedrock, Vertex, LiteLLM) synthesizes into an answer — optionally through a multi-turn ReAct agent loop, GraphRAG map-reduce, or a FLARE confidence-gated re-retrieval loop.

Around this core sit two front doors (a FastAPI REST server and a ~3,970-line Typer CLI, `trelix`), optional additive security layers (OIDC auth + hash-chained audit trail, both HTTP-perimeter-only), multi-repo federation, automated PR review, a retrieval/synthesis eval harness, and five satellite integration packages (MCP server, LangChain adapter, LlamaIndex adapter, TypeScript SDK, and a GitHub App) plus a VS Code extension — all thin wrappers delegating to the same core engine, none of which share auth/audit coverage with the REST API.

## 2. Architecture Map

### core-config (`src/trelix/core/`)
**Purpose**: Foundation package — config tree, shared dataclasses, cross-cutting utilities. No business logic; every other subsystem imports from it, never the reverse.
**Key files**: `config.py` (IndexConfig root + 10 nested sub-configs + standalone AuditConfig/SSOConfig + 4 connector configs; `resolve_operator_env_file()`/`OPERATOR_ENV_FILE`); `models.py` (IndexedFile, Symbol, CallEdge, TypeEdge, GenericEdge, Chunk, SearchResult, RerankOutcome, RetrievedContext); `console_safety.py` (`safe_text()`); `logging_setup.py` (structlog wiring); `retry.py` (`with_retry()`, the tenacity contract every network call in the codebase shares).
**Entry points**: `IndexConfig(repo_path=...)`; `with_retry(...)`; `safe_text(value)`; `setup_console_logging`/`setup_json_logging`.
**Connects to**: everything downstream (83 files import from it). It is the root of the dependency graph.

### indexing-pipeline (`src/trelix/indexing/`)
**Purpose**: "Repo on disk → chunks ready for embedding." `FileWalker` discovers files; `Indexer` runs a 5-phase pipeline (parse → insert+chunk → embed → resolve cross-file edges); `FileWatcher`/`MultiRepoWatcher` provide live re-indexing; connectors (Jira/Linear/TestRail/Xray) + `GitLinker`/`ArtifactLinker` pull in external tickets/commits.
**Key files**: `walker.py` (`detect_language()` single chokepoint), `indexer.py` (spine), `parser/registry.py` + `parser/_grammar.py` + `parser/extractors/*.py` (Tree-sitter per-language, `line_window.py` fallback), `chunker.py` (+ContextualChunker), `file_summarizer.py`, `multi_granularity.py`, `git_linker.py`, `artifact_linker.py`, `connectors/{base,registry,jira,linear,testrail,xray}.py`, `watcher.py`, `multi_watcher.py`.
**Entry points**: `Indexer(config).index()` / `.index_file(path)`; `FileWalker(config).walk()`; `FileWatcher.start()`; `MultiRepoWatcher.run()`; `GitLinker.link()`; `ArtifactLinker.link()`.
**Connects to**: reads `core.config`/`core.models`; writes through `store.db.Database`; hands chunks to `store.vector`; calls `llm.factory.build_chat_client` for contextual chunking/summaries; feeds `graph.updater.GraphUpdater` on file change; feeds `analysis.defuse.DataFlowExtractor`.

### embedding-store (`src/trelix/embedder/` + `src/trelix/store/`)
**Purpose**: Turns chunk text into vectors (9 providers) and persists vectors + all structured metadata in one SQLite file.
**Key files**: `embedder/base.py` (BaseEmbedder ABC, `make_embedder()`, 6 stable providers), `embedder/bge_code.py`/`nomic_code.py` (EXPERIMENTAL/broken), `embedder/cache.py` (CachingEmbedder LRU), `embedder/sparse.py` (SPLADE-Code); `store/db.py` (`Database` — 14+ tables, FTS5 `symbols_fts`, schema versioning via `PRAGMA user_version`), `store/vector.py` (`SQLiteVectorStore`, sqlite-vec `vec0` + optional HNSW, sentinel-id multiplexing for file-summary/sub-chunk vectors), `store/vector_lance.py`/`vector_qdrant.py` (optional backends), `store/dimension_guard.py`, `store/sparse_store.py`, `store/provenance.py`, `store/read_pool.py`.
**Entry points**: `make_embedder(EmbedderConfig)`; `Database(db_path)`; `make_vector_store(config, dimension)`; `DimensionGuard.check/record`.
**Connects to**: consumed by indexing, retrieval, graph, federation, review, agent, cli, api.

### retrieval-ranking (`src/trelix/retrieval/`)
**Purpose**: Turns a natural-language question into a token-budgeted context string. Planner classifies intent → `Retriever` fans out legs → RRF fusion → graph expansion → rerank → `ContextAssembler` packs → `Synthesizer` streams an answer.
**Key files**: `planner/models.py` (`INTENT_STRATEGIES` table), `planner/agent.py` (`QueryPlanner`, `AdaptiveRouter` 3-tier), `fusion.py` (`reciprocal_rank_fusion`, the single cross-repo/cross-leg dedupe authority keyed on `(abs path, symbol_id)`), `retriever.py` (`Retriever`, spine), `graph.py` (post-fusion call/import/type-edge expansion + PageRank boost), `reranker.py` (+`_plaid`/`_xtr`), `assembler.py` (`ContextAssembler`), `bm25.py`, `grep_search.py`, `sparse_search.py`, `query_expansion.py` (HyDE/MultiQuery), `plan_cache.py`, `graph_rag.py` (map-reduce), `flare.py` (confidence-gated re-retrieval), `synthesizer.py`, `context_compression.py`, `otel_tracing.py`, `telemetry.py`.
**Entry points**: `Retriever(config).retrieve(query)`; `QueryPlanner.plan()`/`AdaptiveRouter.route()`; `reciprocal_rank_fusion(...)`; `Synthesizer.synthesize()`/`.stream()`; `FLARELoop.run()`.
**Connects to**: `store.db`/`store.vector`/`store.sparse_store` for hydration; `embedder.*` for query embedding; `llm.factory.build_chat_client` for planner/HyDE/GraphRAG/Synthesizer; `graph.code_graph`/`graph.search`/`graph.persistence` for graph leg + PageRank boost; consumed by `federation.retriever` (shares `fusion.py`'s dedupe identity), `agent.loop`, `api.app`, `cli.main`, `eval.harness`.

### graph-analysis (`src/trelix/graph/` + `src/trelix/analysis/`)
**Purpose**: Builds/queries a NetworkX code-property graph over already-resolved DB edges (Louvain communities, PageRank, optional LLM concept extraction, Pyvis visualization); separately runs two best-effort static-analysis utilities (def-use chains, Semgrep taint).
**Key files**: `graph/code_graph.py` (`CodeGraph`, dangling-reference-guarded edge insertion), `graph/builder.py` (`GraphBuilder.build()`), `graph/community.py` (Louvain + `PartitionQuality`), `graph/persistence.py` (`graph_metadata` table, `get_top_central_symbols`), `graph/updater.py` (`GraphUpdater`, incremental), `graph/search.py` (`graph_search` BFS leg), `graph/concepts.py`, `graph/visualizer.py`; `analysis/taint.py` (`TaintAnalyzer`, Semgrep subprocess, explicit `ScanOutcome`), `analysis/defuse.py` (`DataFlowExtractor`, hardcoded to Python).
**Entry points**: `GraphBuilder(config).build()`; `graph_search(...)`; `get_top_central_symbols(...)`; `GraphUpdater.update_file()`; `TaintAnalyzer(repo_path, tier).scan()`; `DataFlowExtractor().extract(symbol)`.
**Connects to**: sole data source is `store.db.Database`'s `iter_resolved_*` methods (never resolves references itself); consumed by `retrieval.retriever` (graph leg + PageRank boost), `indexing.watcher` (GraphUpdater), `indexing.indexer` (DataFlowExtractor), `cli.main`/`api.app` (GraphBuilder/Visualizer/TaintAnalyzer).

### agent-orchestration (`src/trelix/agent/`)
**Purpose**: Multi-turn ReAct (Thought→Action→Observation) query mode as an alternative to single-pass retrieve-then-synthesize. Small, self-contained (~450 LOC), one call site.
**Key files**: `loop.py` (`AgentLoop`), `tools.py` (`AGENT_TOOLS` — 4-tool fixed action space), `actions.py` (dataclasses), `history.py` (`TurnHistory`/`HistoryCompressor`, fence-safety helpers).
**Entry points**: `AgentLoop(config).run(query, session_id=None)`.
**Connects to**: lazy-imports `retrieval.retriever.Retriever` and `retrieval.grep_search.grep_search`; persists sessions via `store.db.Database`; builds its LLM client via `llm.factory.build_chat_client`; sole consumer is `cli.main`'s `ask --agentic`/`--session`.

### llm-providers (`src/trelix/llm/`)
**Purpose**: The single provider-abstraction seam for every chat/completion call in trelix. See §3 for the full call-path trace.
**Key files**: `client.py` (`TrelixChatClient` ABC, `ChatMessage`/`ChatResponse`/`ToolCallResponse`, `seed_kwargs`), `factory.py` (`build_chat_client`), `context_windows.py` (`resolve_window`, hand-maintained prefix table), `prompt.py` (`fenced_block`/`fence_for`), `providers/{anthropic,bedrock,openai,vertex,litellm}_backend.py`.
**Entry points**: `build_chat_client(LLMConfig)`; `TrelixChatClient.complete/stream/tool_call`.
**Connects to**: consumed by `agent.loop`, `review.reviewer`, `retrieval.{planner.agent, query_expansion, graph_rag, synthesizer}`, `indexing.{indexer, chunker, file_summarizer}`, `graph.concepts`; every backend funnels through `core.retry.with_retry`.

### api-cli (`src/trelix/api/` + `src/trelix/cli/`)
**Purpose**: The two external front doors. Both are thin orchestration: build `IndexConfig`/`RetrievalConfig`, delegate to core engines, render as Pydantic JSON (API) or Rich/`--json` (CLI).
**Key files**: `api/app.py` (`create_app()`, 10 routes, `authenticate`/`confine_repo` dependencies — one 1,022-line module in mcp-terms but here it's the FastAPI app); `cli/main.py` (~3,970 lines, 20 top-level commands + 4 sub-Typer groups).
**Entry points**: `GET/POST` routes (`/health,/search,/ask,/index,/parse,/stats,/graph*`); `trelix {index,search,ask,query,call-graph,stats,link-tickets,link-artifacts,update-index,migrate-vectors,watch,watch-all,serve,graph,telemetry,eval,eval-synthesis,taint,review,search-all,federation *,agent sessions *,connector sync,audit *}`.
**Connects to**: everything — `retrieval.*`, `indexing.*`, `graph.*`, `store.db`, `auth.oidc`/`audit.*`, `agent.AgentLoop`, `review.*`, `analysis.taint`, `eval.*`, `federation.*`. `trelix serve` is the bridge: calls `api.app.create_app()` in-process.

### auth-audit (`src/trelix/auth/` + `src/trelix/audit/`)
**Purpose**: Two small, additive, default-OFF subsystems wired together only in `api.app.create_app()` — optional OIDC bearer auth and optional tamper-evident (not tamper-proof) HTTP audit logging. **Guards exactly one ingress**: the FastAPI app. MCP server, agentic loop, CLI, direct library use are unaudited/unauthenticated.
**Key files**: `auth/oidc.py` (`OidcVerifier`, 3x algorithm-allowlist check, host-pinned JWKS fetch), `auth/principal.py` (`Principal`, keyed on `(sub, iss)` — never email), `auth/store.py` (`PrincipalStore`, JIT upsert into `principals` table inside `audit.db`), `audit/events.py`, `audit/middleware.py` (`AuditMiddleware`, outermost), `audit/store.py` (`AuditStore`, SHA-256 hash chain).
**Entry points**: `OidcVerifier.authenticate(token)`; `PrincipalStore.jit_upsert`; `AuditStore.append`/`verify`/`verify_chain`; the inline `authenticate()` FastAPI dependency in `api/app.py`.
**Connects to**: `api.app.create_app()` is the sole wiring point; `cli.main`'s `trelix audit list|verify|export` reads `AuditStore` read-only.

### federation-compression (`src/trelix/federation/` + `src/trelix/compression/`)
**Purpose**: Multi-repo fan-out search (RRF-merged across repos) and one context-compression provider (extractive, shrinks oversized symbol bodies toward a token ratio without ever raising).
**Key files**: `federation/registry.py` (`RepoRegistry`, JSON-backed), `federation/retriever.py` (`FederatedRetriever`, thread-pool fan-out, TTL cache, in-memory `federation_symbols` sqlite table); `compression/base.py` (`Compressor` ABC, `make_compressor()`), `compression/extractive.py` (`ExtractiveCompressor`, sub-chunk-cosine or lexical-overlap scoring).
**Entry points**: `RepoRegistry.load/add/remove/list`; `FederatedRetriever(registry).retrieve(query, k)`; `make_compressor(config, db, embedder)`.
**Connects to**: federation reuses `retrieval.fusion.reciprocal_rank_fusion` and constructs one `retrieval.retriever.Retriever` per repo; compression is wired into `retrieval.assembler`/`context_compression`/`retriever`; `cli.main` (`federation *`, `search-all`, `watch-all`) and `packages/trelix-mcp` are the only consumers.

### review-eval (`src/trelix/review/` + `src/trelix/eval/`)
**Purpose**: Retrieval-augmented automated PR review (git diff or GitHub PR → LLM comments) and two quality harnesses (retrieval nDCG/Recall/MRR against a 54-query golden set; synthesis hallucination/completeness/faithfulness scoring).
**Key files**: `review/reviewer.py` (`DiffReviewer`), `review/github.py` (`GitHubPRClient`), `review/diff_parser.py` (`DiffParser`/`DiffHunk`), `review/diff_embedder.py` (built, **unwired** — no CLI call site); `eval/harness.py` (`EvalHarness`), `eval/ndcg.py` (single shipped metric impl), `eval/synthesis.py` (`SynthesisEvalHarness`).
**Entry points**: `trelix review [--diff|--pr]`; `trelix eval`/`eval-synthesis`; `DiffReviewer.review()`; `EvalHarness.run()`.
**Connects to**: both retrieve via `retrieval.retriever.Retriever`; both call LLMs via `llm.factory.build_chat_client`; reviewer uses `llm.prompt.fenced_block`; sole CLI consumer.

### mcp-package (`packages/trelix-mcp/`)
**Purpose**: Official MCP server — wraps core trelix as 15 tools, 3 resources, 3 prompts over stdio (FastMCP).
**Key files**: `server.py` (1,022 lines, all tools/resources/prompts registered), `resources.py` (pure, transport-agnostic), `prompts.py` (pure), `subscriptions.py` (`SubscriptionRegistry`, raw stdout JSON-RPC notifications).
**Entry points**: console script `trelix-mcp`; tools `search_code, index_codebase, get_symbol, blast_radius, build_knowledge_graph, graph_search_mcp, subscribe/unsubscribe_resource, federation_*, ask_agent, agent_list_sessions, agent_clear_session`.
**Connects to**: imports `trelix.core.config`, `trelix.retrieval.retriever`, `trelix.indexing.indexer`, `trelix.store.db`, `trelix.graph.builder`, `trelix.federation.*`, `trelix.agent.loop` directly (in-process, same Python env); `trelix`'s own `FileWatcher` calls back into `subscriptions.notify_file_changed`. Consumed by `workspace-vscode` and any MCP client.

### langchain-sdk / llamaindex-sdk (`packages/trelix-langchain/`, `packages/trelix-llama-index/`)
**Purpose**: Thin single-class adapters (`TrelixRetriever` / `TrelixIndexRetriever`) exposing trelix as a native `BaseRetriever` for each framework. No tool/chain wrappers despite README claims.
**Key files**: `retriever.py` in each package (~55 lines).
**Entry points**: `TrelixRetriever(repo_path, provider="local", k=10)`; `TrelixIndexRetriever(repo_path, provider="local", k=10)`.
**Connects to**: both lazily import `trelix.core.config.{IndexConfig,EmbedderConfig}` and `trelix.retrieval.retriever.Retriever`, constructing a fresh `Retriever` per call — same core engine, independent PyPI release cadence, independently pinned `trelix>=3.0.0` floors.

### typescript-sdk (`packages/trelix-typescript/`)
**Purpose**: `@trelix/sdk` — zero-runtime-dependency TS HTTP client for the REST API (`trelix serve`), hand-glued over `openapi-typescript`-generated types.
**Key files**: `src/client.ts` (`TrelixClient`), `src/sse.ts` (`askStream`, manual SSE frame parser), `src/generated/schema.ts` (checked-in, CI-drift-gated).
**Entry points**: `new TrelixClient(baseUrl)`; `askStream(baseUrl, {query, repo})`.
**Connects to**: talks REST/SSE only to `src/trelix/api/app.py`; regenerated via `scripts/codegen.mjs` shelling out to `python3 -c "from trelix.api.app import create_app; ..."`; **no auth header support** despite the server's gated routes.

### vscode-extension (`workspace-vscode/`)
**Purpose**: VS Code extension exposing search/ask/hover/code-lens/chat-participant, all funneled through one `TrelixMcpClient` that spawns `trelix-mcp` as a stdio child process.
**Key files**: `src/extension.ts`, `src/mcp-client.ts`, `src/search-controller.ts`, `src/chat-handler.ts`, `src/hover-provider.ts`, `src/code-lens-provider.ts`.
**Entry points**: commands `trelix.search/ask/findSimilar/blastRadius`; chat participant `@trelix`.
**Connects to**: exclusively `packages/trelix-mcp`'s `search_code`/`get_symbol`/`ask_agent`/`blast_radius` tools over JSON-RPC/stdio — never talks to the REST API.

### github-app (`infra/github-app/`)
**Purpose**: Standalone Node/Express GitHub App — alternative, zero-workflow-YAML delivery for automated PR review, duplicating `.github/workflows/trelix-review.yml`'s capability via an installable App.
**Key files**: `src/webhook.ts` (signature verification, event filtering), `src/auth.ts` (installation-token minting/caching), `src/review-runner.ts` (shells out to the `trelix` CLI: `trelix review <repo> --pr owner/repo#N --json`).
**Entry points**: `POST /webhooks/github`; `runReview(config, request)`.
**Connects to**: the **only** coupling to Python trelix is a subprocess call to the `trelix` CLI binary — no in-process import; consumes `src/trelix/cli/main.py`'s `review --pr --json` contract.

### testing-qa (`tests/` + `scripts/`)
**Purpose**: The quality-gate system — pytest suite (unit/integration/e2e/eval/property/regressions/perf) plus standalone scripts (`mutation.py`, `check_coverage_floors.py`, `verify_release.py`, `self-index.sh`, `verify-index.sh`). Distinctive trait: nearly every mechanism closes a named, measured past incident.
**Key files**: `tests/conftest.py` (path-based marker taxonomy), `tests/_env_isolation.py` (single shared env-scrub table), `tests/regressions/harness.py` (git-worktree patch-restoration harness, catches test-suite-itself weakening that mutation testing structurally can't), `tests/integration/test_eval.py` (per-query rank ledger), `scripts/mutation.py` (scoped mutmut, per-module survivor ceilings, never a ratio), `scripts/check_coverage_floors.py`, `scripts/verify_release.py`.
**Connects to**: exercises the real `Indexer`/`Retriever`/`Database`/embedder/vector-store stack; CI (`ci.yml`) wires unit→coverage-floors, `release.yml` wires test→build→`verify_release.py`.

### infra-deployment (root `Dockerfile`, `helm/`, `Makefile`, `.github/workflows/`)
**Purpose**: Build/package/deploy/release. Two Docker image variants (slim / `-local` with torch) to GHCR; a Helm chart deploying `trelix serve` as a stateless multi-repo Deployment; PyInstaller binaries for 4 platforms; 11 GitHub Actions workflows gating PRs and driving tag-triggered releases (4 PyPI packages via OIDC trusted publishing).
**Key files**: `Dockerfile`, `helm/trelix/templates/deployment.yaml`, `.github/workflows/{ci,release,docker-publish,security-scan,helm-lint,verify-release,build-binaries,schema-drift,trelix-review}.yml`.
**Entry points**: Docker `ENTRYPOINT ["trelix"] CMD ["serve","/repo",...]`; Helm Deployment args `["serve", mountPath, "--host","0.0.0.0","--port",port]`.
**Connects to**: ships `src/trelix` + `packages/trelix-mcp` only into the Docker image (langchain/llamaindex/typescript adapters are PyPI/npm-published but never enter the image); release pipeline gates on 12 version-stamp checks across `pyproject.toml`, 3 package `__init__.py`s, Helm `Chart.yaml`/`values.yaml`, trelix-mcp `server.json`.

### docs-architecture (`docs/`)
**Purpose**: Hand-maintained documentation set (v3.2.5, ~14,200 lines) — onboarding, deep reference (`architecture.md` is the cited source-of-truth other docs defer to), integrations, security/ops, versioning policy, provenance artifacts. Unusually self-auditing: several files narrate their own prior errors with measured evidence.
**Key files**: `docs/README.md` (nav index), `docs/architecture.md` (2,289 lines, the reference other docs cite by module path), `docs/CONFIGURATION.md`, `docs/CLI_REFERENCE.md`, `docs/GLOSSARY.md`, `docs/MCP_GUIDE.md`, `docs/BACKWARDS_COMPATIBILITY.md`, `docs/SSO.md`, `docs/AUDIT.md`.
**Connects to**: documents (without importing) every subsystem above; see §5 for confirmed inaccuracies.

## 3. End-to-End Data Flow

**Phase A — Indexing a repo** (`trelix index <repo>` → `Indexer(config).index()`, `src/trelix/indexing/indexer.py`):
1. `FileWalker.walk()` (`walker.py`) recursively lists the repo, applying nested-`.gitignore` chains, a two-tier ignore-dir policy, symlink-loop guards, computes a SHA-256 hash per file, and yields `IndexedFile` via `detect_language()`.
2. Phase 1 (thread-pooled): `parser.registry.get_parser(language)` dispatches to a cached `BaseParser` (Tree-sitter, `parser/extractors/*.py`) or falls back to `LineWindowParser` (`parser/extractors/line_window.py`) for symbol-empty files or unsupported languages (shell/dockerfile/make/sql/proto), producing `Symbol`/`CallEdge`/`ImportEdge`/`TypeEdge` with local-index parent/caller links.
3. Phase 2 (single-threaded, for FK-remap correctness): diffs against `sha256(signature+body)` hashes already in `Database` (`store/db.py`), inserts changed symbols to get real DB ids, remaps local→DB indices, optionally runs `DataFlowExtractor` (`analysis/defuse.py`), then `Chunker.build_chunks()` (`chunker.py`) turns symbols into `Chunk` objects with context headers, token-budgeted via `tiktoken`. Optional Phase 2.5 (`file_summarizer.py`) and 2.6 (`multi_granularity.py`) add file-level and sub-symbol chunks.
4. Phase 3 (async, TPM-rate-limited): `_batch_embed_and_store_async` calls `embedder.embed_async()` (from `make_embedder(config.embedder)`, `embedder/base.py`) in batches, writes vectors via `vector_store.upsert_batch()` (`store/vector.py`'s `SQLiteVectorStore`, sqlite-vec `vec0`). Any batch failure raises `PartialIndexError` (chunk rows are already committed — reconciled later by `_chunks_missing_vectors()`).
5. Phase 4: `db.resolve_cross_file_calls/import_file_ids/cross_file_type_edges/angular_selectors()` wire up edges that couldn't resolve within a single file.
6. Separately: `GitLinker.link()` shells out to `git log`, and `ArtifactLinker.link()` regex/embedding-matches connector-fetched `Artifact` records — both write into the shared `generic_edges` table.
7. (If graph is built): `GraphBuilder(config).build()` (`graph/builder.py`) reads the now-resolved `calls`/`imports`/`type_edges`/`generic_edges` tables via `CodeGraph._build()` (`graph/code_graph.py`), runs Louvain + PageRank, persists to `graph_metadata` (`graph/persistence.py`).

**Phase B — Answering a query** (`trelix ask <query>` → `cli/main.py`'s `ask()` command, ~line 1242):
1. CLI builds `IndexConfig(repo_path=...)`. If `--agentic`/`--session` or `config.retrieval.agentic_enabled`, dispatches to `AgentLoop(config).run(query, session_id)` (`agent/loop.py`) instead of the path below.
2. Default path: `Retriever(config).retrieve(query)` (`retrieval/retriever.py`). `QueryPlanner.plan()`/`AdaptiveRouter.route()` (`retrieval/planner/agent.py`) does one LLM tool-call (or a Tier-1 regex shortcut) to produce a `QueryPlan` (intent + `RetrievalStrategy` from `INTENT_STRATEGIES` + 1-3 `SubQuery` objects).
3. `_execute_plan()` routes: direct-lookup intents (file/project/config overview) hit `Database` directly with a "breadth floor" escalation; everything else goes to `_retrieve_standard()`, which runs each leg (vector via `embedder.embed_query()` → `vector_store.search()`; BM25 via `Database.bm25_search()` FTS5; grep; optionally sparse/file-summary/sub-chunk) in a `ThreadPoolExecutor`.
4. `reciprocal_rank_fusion()` (`retrieval/fusion.py`) merges leg lists, deduped on `(abs file path, symbol_id)`.
5. `graph.py`'s `expand_with_call_graph`/`expand_with_imports`/`expand_with_type_edges` (using `graph/code_graph.CodeGraph` + `graph/persistence.get_top_central_symbols` for the PageRank boost) add structurally-related symbols.
6. `rerank_with_outcome()` (`retrieval/reranker.py`) reranks to `top_n` (cross-encoder/Cohere/PLAID/XTR, or skipped).
7. `ContextAssembler.assemble()` (`retrieval/assembler.py`) packs into the token budget (greedy/proportional/breadth-first), optionally invoking `compression/extractive.py`'s `ExtractiveCompressor` for overflow, producing a `RetrievedContext`.
8. `Synthesizer.synthesize()`/`.stream()` (`retrieval/synthesizer.py`) — or `GraphRAGSynthesizer` for oversized contexts, or `FLARELoop` for confidence-gated re-retrieval — turns the context into an LLM answer. This is where the LLM call-path (below) kicks in.
9. Streamed tokens flow back through the CLI's Rich console (each token passed through `core.console_safety.safe_text`/`_safe_text` before printing) or, for the API's `GET /ask`, as a `text/event-stream` SSE response terminated by `data: [DONE]`.

**Phase C — The actual model call** (client construction → provider backend → SDK → streaming → retry → back to caller):

All three call sites (`Retriever`'s `Synthesizer`, `AgentLoop`, `DiffReviewer`) obtain a client via `trelix.llm.factory.build_chat_client(LLMConfig)`:
- `Synthesizer.__init__` (`synthesizer.py:113-155`) builds it eagerly at construction (`build_chat_client(llm_config)`, imported at module level specifically so tests can patch it).
- `AgentLoop._get_client()` (`loop.py:64-69`) and `DiffReviewer._get_client()` (`reviewer.py:84-93`) build it lazily and memoize on the instance; `DiffReviewer` additionally swallows construction failure into `None` and short-circuits `review()` to `[]`.

`build_chat_client` (`factory.py`) is a single `match config.provider` dispatch — `openai`/`azure`→`OpenAIBackend`, `anthropic`→`AnthropicBackend`, `bedrock`→`BedrockBackend`, `vertex`→`VertexBackend`, `litellm`→`LiteLLMBackend` — each backend imported inside its own `match` arm so only the selected SDK loads.

Each backend funnels its **one** (or two, for streaming) network chokepoint through `@with_retry(max_attempts=5)` (`core/retry.py`), with the underlying SDK's own retry explicitly disabled (`max_retries=0` / `botocore max_attempts=0`) to avoid double-retrying:
- **OpenAI/Azure** (`openai_backend.py`): `_create()` → `self._client.chat.completions.create(**kwargs)`. `stream()` retries only the connection-opening call (synchronous under `stream=True`), then `yield chunk.choices[0].delta.content` per chunk — plain strings back to `Synthesizer.stream`'s generator.
- **Anthropic** (`anthropic_backend.py`): `_create()` for non-streaming; `_open_stream()` retries `manager.__enter__()` only (not the generator), then `yield from stream.text_stream` inside `try/finally: manager.__exit__(...)`.
- **Bedrock**: `_call_with_retry()` around the Converse API, with a secondary (not itself retried) fallback swap to `bedrock_fallback_model` on `ValidationException`.
- **Vertex**: `_generate_content()`/`_open_content_stream()`, pulling the first stream chunk inside the retried call to keep retries pre-first-byte.
- **LiteLLM**: single `_completion()` shared by `complete`/`stream`/`tool_call`, since `litellm.completion()` opens its connection synchronously even under `stream=True`.

`with_retry()` (`core/retry.py`) wraps `tenacity.retry(retry_if_exception(is_retryable_http_error), wait=_wait_retry_after_or_exponential(1.0, 60.0), stop=stop_after_attempt(5))`. `is_retryable_http_error()` classifies 429/5xx/connection errors across httpx/requests/botocore/openai/anthropic/google-genai/voyageai by checking `sys.modules` first (never force-importing an SDK just to classify an exception).

**Credential resolution** (per provider, all via `LLMConfig`, `core/config.py:1265-1328`, `SettingsConfigDict(env_prefix="TRELIX_LLM_", env_file=OPERATOR_ENV_FILE, ...)`): each provider field is aliased to bypass the `TRELIX_LLM_` prefix and read the SDK's own conventional env var name — `openai_api_key`→`OPENAI_API_KEY`, `anthropic_api_key`→`ANTHROPIC_API_KEY`, `azure_api_key`/`azure_endpoint`→`AZURE_API_KEY`/`AZURE_ENDPOINT`, `aws_access_key_id`/`aws_secret_access_key`→standard AWS names (Bedrock also runs these through `_decode_credential()`, a base64-heuristic that risks silently corrupting a plaintext secret that happens to decode cleanly), `google_api_key`/`google_project_id`→`GOOGLE_API_KEY`/`GOOGLE_CLOUD_PROJECT` (Vertex). All are gated identically by `OPERATOR_ENV_FILE` (resolved once from `TRELIX_CONFIG_FILE` or `~/.config/trelix/env`, **never the process cwd** — a deliberate fix for a repo-committed-`.env`-hijack defect). No backend re-implements or bypasses this; it's enforced once, centrally.

**Files/functions load-bearing to Phase C**: `src/trelix/llm/factory.py::build_chat_client`; `src/trelix/llm/client.py::TrelixChatClient/ChatMessage/ChatResponse/ToolCallResponse`; `src/trelix/llm/providers/{openai,anthropic,bedrock,vertex,litellm}_backend.py`; `src/trelix/core/retry.py::with_retry/is_retryable_http_error`; `src/trelix/core/config.py::LLMConfig,resolve_operator_env_file,OPERATOR_ENV_FILE`; `src/trelix/retrieval/synthesizer.py::Synthesizer.{__init__,synthesize,stream}`; `src/trelix/agent/loop.py::AgentLoop._get_client/_next_action`; `src/trelix/review/reviewer.py::DiffReviewer._get_client/_review_hunk`.

## 4. Cross-Cutting Concerns

**Auth/audit**: Two additive, default-OFF subsystems (`auth/`, `audit/`) wired together in exactly one place, `api/app.py`'s `create_app()`. OIDC verifies bearer JWTs (asymmetric-algorithm-only, checked three separate times; host-pinned JWKS fetch that silently disables itself if `TRELIX_OIDC_ISSUER` isn't a full URL — a documented, self-disabling security control) into an immutable `Principal` keyed on `(subject, issuer)`, JIT-provisioned into a `principals` table living inside the same `audit.db` the hash-chained `AuditStore` writes to. Coverage is HTTP-perimeter-only: MCP tool calls, the agentic loop, CLI commands, and direct library use are **never** authenticated or audited — a gap both `docs/SSO.md` and `docs/AUDIT.md` disclose explicitly. Static-token auth remains a fully co-equal trust path even with OIDC enabled (both must be disabled to raise the bar), and there is no authorization layer — any authenticated principal reaches every allowed repo.

**Observability**: Fragmented across four places by design, not by oversight: `core/logging_setup.py` gives structlog-based console/JSON logging; `retrieval/otel_tracing.py` wraps every pipeline stage (planner, legs, fusion, expansion, rerank, assembly) in OTel spans, propagated across `ThreadPoolExecutor` boundaries via `with_current_context`; `embedder/base.py` records 4 monotonic embedding-cost OTel counters per call (explicitly **not** recording LLM token spend, reranker cost, or retrieval latency at the metrics layer — though `retrieval/telemetry.py`'s `TelemetryWriter` *does* separately persist per-query timing/result-count to the DB when `telemetry_enabled`, and `llm/client.py`'s `ChatResponse` *does* carry token counts end-to-end — so the docs' "metrics don't cover token spend" claim is narrowly true for the OTel-metrics surface specifically, not for the codebase overall); `audit/store.py` is the separate tamper-evident hash chain. Two providers (bge-code, nomic-code) are completely uninstrumented for even the 4 cost counters. No single module or doc synthesizes what an operator can see end-to-end — this remains a real navigational gap.

**Config/credentials**: Every sub-config is a `pydantic-settings` `BaseSettings` with its own `env_prefix`, all anchored to the same `OPERATOR_ENV_FILE` (resolved once, at import time, from `TRELIX_CONFIG_FILE` or `~/.config/trelix/env` — **never the process cwd**, closing a defect where a repo-committed `.env` could silently repoint providers when trelix runs inside someone else's checkout, e.g. via `trelix review`/`trelix index .` on a PR branch). Each of the 6 LLM providers (openai, azure, anthropic, bedrock, vertex, litellm) resolves credentials via a field-level `alias` on `LLMConfig` that bypasses the `TRELIX_LLM_` prefix to read the SDK's own conventional env var name (see §3's credential-resolution paragraph) — this pattern is now fully traced and closes the completeness critique's "secrets lifecycle" gap for the load path specifically. What remains unaddressed anywhere in the codebase: rotation, revocation, or an audit-of-use trail for any of these keys once loaded — no report or trace found evidence that LLM/embedder/connector secrets ever leak into `AuditEvent.detail` or OTel span attributes, but nothing actively proves they don't either (this was checked in the trace only for `AuditMiddleware`, which never records header/query content and always has `detail=None` at its call site — narrower reassurance than a full secrets-never-logged guarantee across the whole codebase).

**Versioning**: Four PyPI packages (`trelix`, `trelix-mcp`, `trelix-langchain`, `trelix-llama-index`) are lockstep-versioned off one git tag (`v*`), enforced by a 12-check `verify-version` CI job spanning `pyproject.toml`, three `__init__.py`s, Helm `Chart.yaml`/`values.yaml`, and `trelix-mcp/server.json` (2 fields) — but `packages/trelix-typescript` and the Helm chart's own `chart: version` (0.2.4, independent of `appVersion`) are **not** covered by that check, and `infra/github-app`/`workspace-vscode` are gated only by their own separate CI workflows, entirely outside the release pipeline. Deprecation policy requires the greater of 2 minor versions or 3 months, with a documented historical conflict against `CONTRIBUTING.md`'s wording that was never independently re-verified (see §6).

## 5. Documentation Accuracy Notes

| Claim | Doc source | Reality | Severity |
|---|---|---|---|
| Operator config `.env` resolves relative to the process's cwd | `docs/CONFIGURATION.md`, `docs/architecture.md §3` | `OPERATOR_ENV_FILE` is anchored to `TRELIX_CONFIG_FILE` or `~/.config/trelix/env` — deliberately **never** cwd, per `core/config.py`'s `resolve_operator_env_file()` | High |
| `TRELIX_FEDERATION_MAX_REPOS` has no effect on `trelix search-all`, only on MCP tools | `docs/GLOSSARY.md`, `docs/architecture.md §13` | The CLI's `search-all` explicitly reads and passes `RetrievalConfig().federation_max_repos` into `FederatedRetriever` — the claim is backwards | High |
| Bedrock Titan has a "fixed dimension" of 1024, same rigidity as the other 8 providers | `docs/architecture.md §3-4`, `docs/PROVIDERS.md` | Titan actually supports selectable 256/512/1024-dim; 1024 is just `EmbedderConfig`'s default | Medium |
| "Trelix's 6 MCP tools" | `docs/WHY_TRELIX.md` | 15 tools, confirmed by `mcp-package` report and `docs/MCP_GUIDE.md` itself — stale, era-of-v2.4.0 leftover | Medium (internal doc-set inconsistency) |
| FAQ.md header stamps v3.1.5 | `docs/FAQ.md` | Rest of doc set (README, architecture.md, CLI_REFERENCE.md, ROADMAP.md, pyproject.toml) stamps v3.2.5 as of the same claimed update date | Low-Medium |
| GLOSSARY.md: SQLite schema has "eight main tables" | `docs/GLOSSARY.md` | `store/db.py` DDL lists 14+ tables plus `graph_metadata` (outside db.py) — significant undercount | Medium |
| `.trelix/federation.json` per-repo registry override exists | `docs/GLOSSARY.md` | `federation/registry.py`'s actual default path is `~/.config/trelix/repos.json`; no code-side counterpart found for the claimed override | Medium |
| Watch-All automatically invalidates FederatedRetriever's cache | `docs/FEDERATION_GUIDE.md` | `MultiRepoWatcher` never imports or calls `FederatedRetriever.clear_cache()` — every CLI/MCP invocation constructs a fresh (empty-cache) `FederatedRetriever` anyway, so the described behavior has no code path that would ever exercise it | Medium |
| RRF dedup "keeps the highest-scoring occurrence" | `docs/FEDERATION_GUIDE.md` | `fusion.py` keeps the **first-seen** result object (per an explicit code comment forbidding score-based replacement, since cross-leg scores aren't comparable) and only overwrites `.score` afterward | Low |
| SECURITY.md claims `trelix serve` binds to 127.0.0.1 by default | `.github/SECURITY.md` (per infra-deployment's risk note) | Docker image's own default CMD is `--host 0.0.0.0` — flagged as a contradiction but **SECURITY.md itself was never read directly** by any report; treat as unconfirmed | Unconfirmed — verify before trusting either claim |
| Helm chart's `deployment.yaml`/README call the served `repo_path`/mountPath argument "vestigial"/"decorative" | `helm/trelix/README.md`, `helm/trelix/templates/deployment.yaml` comments | `cli/main.py`'s `serve()` passes `served_root=repo_path` into `create_app()`, and `api/app.py`'s `_resolve_allowed_roots` uses it as the first entry of the per-request containment allow-list — the mount path **is** load-bearing, contradicting the chart's own comment | High (security-relevant self-contradiction) |

## 6. Coverage Gaps & Follow-Ups

Items from the completeness critique that the LLM call-path trace **resolved**: the end-to-end CLI→planner→retriever→synthesizer→LLM-backend→streamed/escaped-output flow is now traced concretely (§3, Phase B/C); credential resolution for all 6 providers is now traced per-provider (§3, §4); the `OPERATOR_ENV_FILE`/cwd doc discrepancy is now confirmed rather than merely flagged.

Still genuinely open after the trace:

- **`config/semgrep-taint.yaml`** (the rule set backing both `trelix taint` and the CI-blocking `security-scan.yml` taint job) was never opened by any report or the trace — its actual vulnerability-class coverage remains an unverified black box despite gating merges.
- **Root-level governance files** (`CHANGELOG.md`, `CONTRIBUTING.md`, `SECURITY.md`, `SUPPORT.md`, `LICENSE`, root `README.md`) were never read directly by any report. Two specific flagged contradictions remain unconfirmed: SECURITY.md's actual bind-address claim (table row above), and CONTRIBUTING.md's actual grace-period wording versus `BACKWARDS_COMPATIBILITY.md`'s claimed conflict.
- **`.github/` non-workflow files** (issue/PR templates, CODEOWNERS) — never examined.
- **`src/trelix/__init__.py`'s actual re-export surface** — only ever inferred from consumer imports across reports, never read directly, despite being cited as "the module docstring documents the canonical Indexer/Retriever pattern."
- **Observability synthesis** — still no single module or doc gives an operator one coherent answer to "what can I see in production." The trace clarified the OTel-metrics-vs-telemetry-vs-audit split (§4) but did not unify it into tooling; this is a documentation/tooling gap, not just a coverage gap.
- **Secrets-in-logs/audit guarantee** — the trace confirmed `AuditEvent.detail` is `None` at its one call site (so HTTP audit rows can't leak secrets), but no report or trace checked OTel span attributes, structlog output, or the CLI's Rich console rendering for accidental credential echoing (e.g., a misconfigured provider error message containing a key). This is narrower than "secrets never leak" and should not be read as a clean bill of health.
- **Cost/token accounting**: `ChatResponse` carries token counts and Anthropic-specific cache-read/cache-write counts per call, and `embedder/base.py` records embedding-cost OTel counters, but nothing in the trace or reports identified a place these are aggregated into a per-query or per-session cost total — `retrieval/telemetry.py`'s `TelemetryWriter` records timing/result-counts, not $-cost. Treat "what did this query cost" as unanswered by the current codebase.
- **DiffEmbedder** (`review/diff_embedder.py`) remains fully implemented but wired into nothing — confirmed dead code, not a documentation issue, but worth flagging to anyone navigating `review/`.