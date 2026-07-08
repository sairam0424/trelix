# trelix Roadmap

> **Status:** Living document — updated with each release.
> **Version:** 2.5.0 (current) | **Next:** 2.6.0

This roadmap describes planned features, research directions, and long-term vision for trelix. Items are organized by phase; specific timelines are intentionally loose to reflect research-driven development.

---

## ✅ Shipped (v2.0 – v2.5)

| Version | Feature |
|---------|---------|
| v2.0.0 | BGE-Code-v1 embedder, PLAID reranker, LanceDB backend, REST API, Knowledge Graph |
| v2.1.0 | Beast-mode retrieval: FLARE loop, HyDE fallback, multi-query expansion, query telemetry, CoIR eval |
| v2.2.0 | Agentic ReAct loop, data-flow analysis, SPLADE sparse retrieval, multi-granularity indexing |
| v2.3.0 | DimensionGuard, MultiQueryExpander wiring, MCP Resources + Prompts, DiffReviewer, FederatedRetriever |
| v2.4.0 | flare_max_retries rename, expansion observability, federation cache, GitHub PR review, watch-all, MCP pagination |
| v2.5.0 | Multi-query expansion wiring (TRELIX_RETRIEVAL_MULTI_QUERY), DimensionGuard at watch startup, MCP resource subscriptions, v3.0.0 deprecation audit |

---

## 🐛 v2.5.1 — Backlog (bugs / hardening from v2.5.0)

- [ ] **SparseEmbedder TOCTOU under parallel multi-query** — add `threading.Lock` around lazy-init path hit by `ThreadPoolExecutor` workers
- [ ] **`send_resource_notification` stdout isolation** — fix asyncio transport conflict when FastMCP writes notifications to stdout concurrently
- [ ] **`SubscriptionRegistry` max-subscriber cap / TTL eviction** — unbounded subscription growth; add configurable cap and TTL-based cleanup
- [ ] **Watch bridge: wire `notify_file_changed` into `FileWatcher._do_reindex` callback** — MCP subscribers not notified after file-change re-index completes

---

## 🏗️ v2.6.0 — Scale & Performance (Q4 2026)

**Goal:** Handle 1M+ symbol codebases without degradation.

### Plan A: Incremental Louvain Community Detection — ✅ Shipped (v2.6.0)
- [x] **DF Louvain frontier heuristic** — Maintains prior partition, reprocesses only affected-vertex frontier
- [x] `compute_affected_frontier(G, seed_nodes, partition)` — Computes affected nodes
- [x] `detect_communities_incremental()` — Incremental Louvain with >50% frontier fallback
- [x] **GraphUpdater** — Stores `_prev_partition`, uses incremental detection on file changes

### Remaining backlog
- [ ] **Cross-repo symbol resolution** — Sourcegraph-style SCIP symbol IDs for cross-repository lookup
- [ ] **Semantic diff embeddings** — CCRep-style before/after body pair embeddings for diff-aware retrieval
- [ ] **Streaming indexing** — yield symbols as parsed (no in-memory buffer for large repos)
- [ ] **Qdrant Cloud integration** — first-class remote vector store with auto-migration
- [ ] **Incremental embedding** — only re-embed changed symbols on partial re-index
- [ ] **Parallel BM25 shard** — FTS5 read-only shards for read-heavy deployments
- [ ] **Binary releases** — single-file executables for Linux ARM64 + Windows ARM64

---

## 🌐 v3.0.0 — Breaking Changes & Ecosystem (H1 2027)

**Goal:** Clean API surface + first-class cloud deployment.

- [ ] **Remove deprecated** — `flare_max_iterations` removed (deprecated in v2.4)
- [ ] **MCP streaming** — true streaming tool responses once MCP spec supports it
- [ ] **Python 3.13 support** — test matrix expansion
- [ ] **OpenTelemetry integration** — spans for every retrieval leg
- [ ] **Helm chart** — production Kubernetes deployment for `trelix serve`
- [ ] **TypeScript SDK** — native SDK matching Python API surface

---

## 💡 Research Backlog (no timeline)

Ideas being researched but not yet committed to a release:

- **CodeBERT fine-tuning** — domain-adapted embedding model trained on trelix's own telemetry data
- **Semantic diff** — diff-aware retrieval (weight recently-changed symbols higher)
- **IDE plugins** — VS Code extension with inline search, JetBrains plugin
- **GitHub App** — PR review comments posted automatically via GitHub App auth
- **Multi-modal** — index diagrams, comments referencing architecture docs
- **Agent memory** — persist AgentLoop history across sessions

---

## How We Decide What to Build

1. **CoIR benchmark** — does it move nDCG@10?
2. **User telemetry** — what queries fail today?
3. **Integration requests** — LangChain/LlamaIndex ecosystem needs
4. **Security requirements** — supply chain, audit trail

File issues or start Discussions to influence the roadmap.
