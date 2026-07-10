# trelix Roadmap

> **Status:** Living document — updated with each release.
> **Version:** 2.7.1 (current) | **Next:** 3.0.0

This roadmap describes planned features, research directions, and long-term vision for trelix. Items are organized by phase; specific timelines are intentionally loose to reflect research-driven development.

---

## ✅ Shipped (v2.0 – v2.7.1)

| Version | Feature |
|---------|---------|
| v2.0.0 | BGE-Code-v1 embedder, PLAID reranker, LanceDB backend, REST API, Knowledge Graph |
| v2.1.0 | Beast-mode retrieval: FLARE loop, HyDE fallback, multi-query expansion, query telemetry, CoIR eval |
| v2.2.0 | Agentic ReAct loop, data-flow analysis, SPLADE sparse retrieval, multi-granularity indexing |
| v2.3.0 | DimensionGuard, MultiQueryExpander wiring, MCP Resources + Prompts, DiffReviewer, FederatedRetriever |
| v2.4.0 | flare_max_retries rename, expansion observability, federation cache, GitHub PR review, watch-all, MCP pagination |
| v2.5.0 | Query telemetry, Eval harness, CoIR benchmark integration |
| v2.6.0 | Incremental Louvain community detection, DF frontier heuristic, GraphUpdater wiring |
| v2.7.0 | Watch bridge wired, DB index, AdaptiveRouter fix, cross-repo symbols, diff embeddings, streaming indexing, VS Code extension, GitHub App PR review |
| v2.7.1 | Release pipeline asset-collision fix, Linux binary in CI, reverted unjustified dependency floors, trelix-mcp tests wired into CI, doc/env-var corrections |

---

## 🔬 v2.5.0 — Retrieval Quality (Q3 2026) ✅ Shipped

**Goal:** Measurably better search results on real codebases.

- [x] **Cross-encoder reranker improvements** — fine-tuned model specifically for code retrieval (research: CoIR benchmark SOTA)
- [x] **Symbol-level PageRank** — use commit frequency + test coverage as centrality signals (beyond import graph)
- [ ] **Multi-language call graph** — resolve cross-language calls (Python → TypeScript via REST/gRPC boundaries)
- [x] **FLARE with uncertainty estimation** — replace heuristic phrase matching with model-calibrated confidence
- [x] **Eval harness improvements** — golden set generation from GitHub starred repos + nDCG@10 public leaderboard

---

## 🏗️ v2.6.0 — Scale & Performance (Q4 2026)

**Goal:** Handle 1M+ symbol codebases without degradation.

### Plan A: Incremental Louvain Community Detection — ✅ In Progress
- [x] **DF Louvain frontier heuristic** — Maintains prior partition, reprocesses only affected-vertex frontier
- [x] `compute_affected_frontier(G, seed_nodes, partition)` — Computes affected nodes
- [x] `detect_communities_incremental()` — Incremental Louvain with >50% frontier fallback
- [x] **GraphUpdater** — Stores `_prev_partition`, uses incremental detection on file changes

### Remaining backlog
- [x] **Streaming indexing** — yield symbols as parsed (no in-memory buffer for large repos) ✅ (shipped v2.7.0)
- [x] **Cross-repo symbol resolution** — SCIP-style IDs ✅ (shipped v2.7.0)
- [x] **Semantic diff embeddings** — CCRep-style before/after encoding ✅ (shipped v2.7.0)
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
- [ ] **VS Code extension improvements** — inline search refinement, snippet preview
- [ ] **GitHub App GA** — public marketplace listing, production hardening

---

## 💡 Research Backlog (no timeline)

Ideas being researched but not yet committed to a release:

- **CodeBERT fine-tuning** — domain-adapted embedding model trained on trelix's own telemetry data
- ~~**Semantic diff** — diff-aware retrieval (weight recently-changed symbols higher)~~ ✅ shipped in Phase 2 Plan B
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
