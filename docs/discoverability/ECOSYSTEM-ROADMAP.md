# trelix — Ecosystem Discoverability Roadmap

Research basis: 108-agent deep research, 730 tool uses, adversarial verification (v2.4.0 — July 2026).  
Target audience: AI agent developers + IDE users + DevOps/CI engineers (all three simultaneously).

**Latest updates (v2.4.0 — July 2026):**
- ✅ FederatedRetriever TTL cache shipped (v2.4.0) — SHA-256 keyed, thread-safe, cache_ttl=0 to disable, cache_stats()
- ✅ Multi-repo file watching (watch-all) shipped (v2.4.0) — MultiRepoWatcher, single awatch() over all repos, hash guard, `trelix watch-all`
- ✅ GitHub PR API integration shipped (v2.4.0) — `trelix review --pr owner/repo#N`, GitHubPRClient, --post-comments, GITHUB_TOKEN
- ✅ MCP search_code pagination shipped (v2.4.0) — **BREAKING**: returns `{results, next_cursor, total_available}` envelope; cursor= param added
- ✅ Multi-query expansion observability shipped (v2.4.0) — ExpandResult dataclass, 3 new query_telemetry columns, expansion_result= kwarg
- ✅ flare_max_retries backward-compat rename shipped (v2.4.0) — TRELIX_RETRIEVAL_FLARE_MAX_RETRIES (new) + TRELIX_RETRIEVAL_FLARE_MAX_ITER (deprecated until v3.0.0)

**Previous updates (v2.0.0 — June 2026):**
- ✅ LanceDB backend shipped
- ✅ PLAID reranker shipped
- ✅ REST API shipped
- ✅ Streaming synthesis shipped
- ✅ BGE-Code-v1 embedder shipped
- ✅ Nomic CodeRankEmbed shipped
- ✅ Knowledge Graph shipped (v2.0.0) — CodeGraph, Louvain communities, Pyvis viz, 4th retrieval leg, MCP tools

---

## Priority Stack (highest → lowest discovery ROI)

```
Tier 1 — Multiplier (one artifact, many ecosystems)
  L1.1  trelix-mcp server         → Claude Code, Cursor, Windsurf, Continue.dev, Copilot
  L1.2  PyPI metadata upgrade     → LangChain/LlamaIndex/Haystack discoverability

Tier 2 — Ecosystem-specific integrations  
  L2.1  LangChain retriever       → pip install trelix-langchain
  L2.2  LlamaIndex retriever      → pip install trelix-llama-index
  L2.3  GitHub Action             → trelix-index marketplace action

Tier 3 — Platform listings
  L3.1  VS Code extension         → Marketplace under Machine Learning + Programming Languages
  L3.2  Homebrew tap              → brew install sairam0424/trelix/trelix
  L3.3  Docker Hub                → docker pull sairam0424/trelix

Tier 4 — Community & discovery
  L4.1  Awesome lists             → awesome-mcp-servers, awesome-llm-apps, awesome-langchain
  L4.2  Product Hunt launch       → scheduled Show HN post
  L4.3  OpenAI Custom GPT action  → OpenAPI spec for ChatGPT tool use
```

---

## Tier 1.1 — trelix MCP Server (HIGHEST PRIORITY)

### Why it's #1
A single MCP server artifact covers: **Claude Code**, **Cursor**, **Windsurf**, **Continue.dev** (now recommends MCP as preferred context extension), and any future MCP-compatible agent. Research finding [1] confirms this is 3-0 verified.

### What to build: `trelix-mcp`
A separate Python package published to PyPI as `trelix-mcp`.

**Tools to expose** (research finding [2] — 3-0 verified):
| Tool | Type | Maps to |
|---|---|---|
| `search_code` | Tool | `Retriever.retrieve(query)` |
| `index_repo` | Tool | `Indexer.index()` |
| `blast_radius` | Tool | `Retriever.retrieve(query, intent=blast_radius)` |
| `get_symbol` | Resource | DB lookup by qualified_name |
| `list_languages` | Resource | Walker language list |
| `repo_stats` | Resource | `trelix stats` output |
| `build_knowledge_graph` | Tool | `GraphBuilder.build()` — ✅ shipped v2.0.0 |
| `graph_search_mcp` | Tool | `graph_search(query, depth)` — ✅ shipped v2.0.0 |

**Implementation pattern** (FastMCP + Streamable HTTP — research finding [1]):
```python
# trelix_mcp/server.py
from fastmcp import FastMCP
from trelix.core.config import IndexConfig
from trelix.retrieval.retriever import Retriever

mcp = FastMCP("trelix", stateless_http=True, json_response=True)

@mcp.tool()
def search_code(query: str, repo_path: str, k: int = 10, cursor: int = 0) -> dict:
    """Search code in an indexed repository using hybrid semantic + keyword search.
    
    Returns {results: list, next_cursor: int|null, total_available: int}.
    Use cursor= for pagination. Migrate: response["results"] instead of iterating response directly.
    """
    config = IndexConfig(repo_path=repo_path)
    ctx = Retriever(config).retrieve(query)
    page = ctx.results[cursor:cursor + k]
    next_cursor = cursor + k if cursor + k < len(ctx.results) else None
    return {
        "results": [{"file": r.file.rel_path, "symbol": r.symbol.qualified_name,
                     "lines": f"{r.symbol.line_start}-{r.symbol.line_end}",
                     "score": round(r.score, 4), "body": r.symbol.body[:500]}
                    for r in page],
        "next_cursor": next_cursor,
        "total_available": len(ctx.results),
    }

@mcp.tool()
def index_repo(repo_path: str, provider: str = "local") -> dict:
    """Index a repository for code search. Run once before searching."""
    from trelix.indexing.indexer import Indexer
    from trelix.core.config import EmbedderConfig
    config = IndexConfig(repo_path=repo_path, 
                         embedder=EmbedderConfig(provider=provider))
    return Indexer(config).index()

@mcp.tool()
def blast_radius(symbol_name: str, repo_path: str) -> list[dict]:
    """Find all files that would break if a symbol changes."""
    config = IndexConfig(repo_path=repo_path)
    ctx = Retriever(config).retrieve(
        f"what imports or calls {symbol_name}"
    )
    return [{"file": r.file.rel_path, "symbol": r.symbol.qualified_name}
            for r in ctx.results]

if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8000)
```

**pyproject.toml additions:**
```toml
[project]
name = "trelix-mcp"
dependencies = ["trelix>=0.4.0", "fastmcp>=2.0.0", "mcp>=1.0.0"]

[project.scripts]
trelix-mcp = "trelix_mcp.server:main"

[project.optional-dependencies]
local = ["trelix[local]"]
```

### Installation by users
```bash
# Claude Code
claude mcp add --transport http trelix http://localhost:8000/mcp

# Cursor / Windsurf — add to .cursor/mcp.json or mcp_config.json:
{ "trelix": { "url": "http://localhost:8000/mcp" } }

# Continue.dev — add to ~/.continue/config.json:
{ "mcpServers": [{ "name": "trelix", "url": "http://localhost:8000/mcp" }] }
```

### Registry submissions (research finding [3])
1. **Official MCP Registry**: `registry.modelcontextprotocol.io` — CLI-driven, self-reported metadata. Submit at: `github.com/modelcontextprotocol/registry`
2. **Smithery**: `smithery.ai` — URL-based HTTP registration (Smithery Gateway proxies it). Submit at: `smithery.ai/publish`
3. **mcp.so**: Community directory, submit via PR.

---

## Tier 1.2 — PyPI Metadata Upgrade

### Current state → Target state

**Classifier upgrade** (research finding [4] — 3-0 verified, mirrors llama-index/langchain):
```toml
# pyproject.toml — add these classifiers
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",          # ← ADD
    "Topic :: Software Development :: Libraries :: Application Frameworks",# ← ADD
    "Topic :: Software Development :: Libraries :: Python Modules",        # ← KEEP
    "Topic :: Text Processing :: Indexing",                               # ← ADD
    "Topic :: Internet :: WWW/HTTP :: Indexing/Search",                   # ← ADD
]

keywords = [
    "code-search", "code-intelligence", "tree-sitter", "semantic-search",
    "hybrid-search", "llm", "rag", "developer-tools", "code-indexing",
    "mcp", "model-context-protocol", "ast", "vector-search", "bm25",
    "call-graph", "embeddings", "openai", "langchain", "llama-index",
    "code-assistant", "code-retrieval", "static-analysis",               # ← NEW
    "knowledge-graph", "graph-search", "community-detection",            # ← NEW (v2.0.0)
]
```

**README badges to add:**
```markdown
[![PyPI](https://img.shields.io/pypi/v/trelix)](https://pypi.org/project/trelix/)
[![Downloads](https://img.shields.io/pypi/dm/trelix)](https://pypi.org/project/trelix/)
[![MCP Compatible](https://img.shields.io/badge/MCP-compatible-blue)](https://github.com/sairam0424/trelix-mcp)
[![LangChain](https://img.shields.io/badge/LangChain-retriever-green)](https://github.com/sairam0424/trelix-langchain)
```

---

## Tier 2.1 — LangChain Retriever

**Package**: `trelix-langchain` — `pip install trelix-langchain`

```python
# trelix_langchain/retriever.py
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from trelix.core.config import IndexConfig
from trelix.retrieval.retriever import Retriever as TrelixRetriever

class TrelixRetriever(BaseRetriever):
    repo_path: str
    provider: str = "local"
    k: int = 10
    
    def _get_relevant_documents(self, query: str) -> list[Document]:
        config = IndexConfig(repo_path=self.repo_path)
        ctx = TrelixRetriever(config).retrieve(query)
        return [
            Document(
                page_content=r.symbol.body,
                metadata={
                    "source": r.file.rel_path,
                    "symbol": r.symbol.qualified_name,
                    "language": r.file.language.value,
                    "lines": f"{r.symbol.line_start}-{r.symbol.line_end}",
                    "score": r.score,
                    "kind": r.symbol.kind.value,
                }
            ) for r in ctx.results[:self.k]
        ]
```

**Submission**: PR to `langchain-ai/langchain-community` integrations directory.

---

## Tier 2.2 — LlamaIndex Retriever

**Package**: `trelix-llama-index` — `pip install trelix-llama-index`

```python
# trelix_llama_index/retriever.py
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, TextNode

class TrelixIndexRetriever(BaseRetriever):
    def __init__(self, repo_path: str, provider: str = "local"):
        self._repo_path = repo_path
        self._provider = provider

    def _retrieve(self, query_bundle) -> list[NodeWithScore]:
        from trelix.core.config import IndexConfig
        from trelix.retrieval.retriever import Retriever
        config = IndexConfig(repo_path=self._repo_path)
        ctx = Retriever(config).retrieve(query_bundle.query_str)
        return [
            NodeWithScore(
                node=TextNode(
                    text=r.symbol.body,
                    metadata={"file": r.file.rel_path, 
                               "symbol": r.symbol.qualified_name}
                ),
                score=r.score
            ) for r in ctx.results
        ]
```

---

## Tier 2.3 — GitHub Action

**Repository**: `sairam0424/trelix-index-action`

**action.yml:**
```yaml
name: 'trelix index'
description: 'Index a repository with trelix for code intelligence'
branding:
  icon: 'search'
  color: 'green'
inputs:
  repo-path:
    description: 'Path to the repository to index'
    default: '.'
  provider:
    description: 'Embedding provider: local | openai | azure | voyage'
    default: 'local'
  cache-key:
    description: 'Cache key for the trelix index'
    default: 'trelix-index-${{ github.sha }}'
runs:
  using: 'composite'
  steps:
    - uses: actions/cache@v4
      with:
        path: .trelix/
        key: ${{ inputs.cache-key }}
    - run: pip install trelix && trelix index ${{ inputs.repo-path }} --provider ${{ inputs.provider }}
      shell: bash
```

**Usage in any repo:**
```yaml
- uses: sairam0424/trelix-index-action@v1
  with:
    provider: local
```

**GitHub Actions Marketplace listing**: requires `action.yml` with `branding.icon` + `branding.color` + `description`. No approval gate — automatically listed when action is public.

---

## Tier 3.1 — VS Code Extension

**Categories** (research finding [5] — 2-1 verified): `Machine Learning` + `Programming Languages`

**Minimal extension** wrapping the trelix binary already in `src/assets/bin/`:
- Command: `trelix.search` — opens quick pick, calls `trelix search`
- Command: `trelix.index` — runs `trelix index` on workspace
- Status bar: shows index status
- Publisher: `sairam0424` on marketplace.visualstudio.com

**Submission**: `vsce publish` — auto-reviewed, typically live within hours.

---

## Tier 3.2 — Homebrew Tap

**Fastest path**: Custom tap (not homebrew-core, which requires 75+ stars + 30-day wait).

```bash
# Create repo: github.com/sairam0424/homebrew-trelix
# File: Formula/trelix.rb

class Trelix < Formula
  desc "Fast code intelligence — Tree-sitter parsing, hybrid search, LLM synthesis"
  homepage "https://github.com/sairam0424/trelix"
  version "0.4.0"

  on_macos do
    if Hardware::CPU.arm?
      url "https://github.com/sairam0424/trelix/releases/download/v0.4.0/trelix"
      sha256 "<arm64-sha256>"
    end
  end

  def install
    bin.install "trelix"
  end
end
```

**User install:**
```bash
brew tap sairam0424/trelix
brew install trelix
```

---

## Tier 3.3 — Docker Hub

**Image**: `sairam0424/trelix`

```dockerfile
# Dockerfile.server
FROM python:3.11-slim
RUN pip install "trelix[local]" fastapi uvicorn
COPY trelix_server.py .
EXPOSE 8000
CMD ["uvicorn", "trelix_server:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Usage:**
```bash
docker run -v /your/repo:/repo -p 8000:8000 sairam0424/trelix
```

---

## Tier 4.1 — Awesome Lists (PR targets)

| List | URL | Status |
|---|---|---|
| awesome-mcp-servers | github.com/punkpeye/awesome-mcp-servers | Open PRs accepted |
| awesome-llm-apps | github.com/Shubhamsaboo/awesome-llm-apps | Open PRs accepted |
| awesome-langchain | github.com/kyrolabs/awesome-langchain | Open PRs accepted |
| awesome-llm-agents | github.com/e2b-dev/awesome-ai-agents | Open PRs accepted |
| awesome-code-search | Multiple — search for "awesome code search" | Various |

**PR message template:**
```
## trelix — Tree-sitter code intelligence with hybrid search + LLM synthesis

[trelix](https://github.com/sairam0424/trelix) is a Python library and CLI for 
code intelligence: indexes 20+ languages via Tree-sitter AST, hybrid 
BM25+vector search, call-graph expansion, knowledge graph with Louvain community 
detection, adaptive query planning, and LLM synthesis. Zero infrastructure 
(SQLite). MCP-compatible.

`pip install trelix` | MIT License
```

---

## Tier 4.2 — Product Hunt + Show HN

**Product Hunt**: Launch on a Tuesday/Wednesday. Key sections: tagline (≤60 chars), description, gallery screenshots of `trelix search`, `trelix ask` output, and the Pyvis knowledge-graph visualization. Request maker comment first.

**Tagline options:**
- "Search any codebase like you talk to a developer"
- "Code intelligence that understands calls, imports, and types"
- "From code search to code intelligence — with knowledge graphs"

**Show HN title**: `Show HN: trelix – hybrid code search with Tree-sitter, BM25+vectors, call graphs, and knowledge graph`

---

## Tier 4.3 — OpenAI Custom GPT Action

trelix can be called by ChatGPT custom GPTs via an OpenAPI 3.1 spec pointed at a deployed `trelix-mcp` server. The same FastAPI wrapper used for Docker Hub doubles as the GPT action endpoint.

```yaml
# openapi.yaml
openapi: "3.1.0"
info:
  title: trelix Code Search
  version: "0.4.0"
paths:
  /search:
    post:
      operationId: searchCode
      summary: Search code in an indexed repository
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                query: { type: string }
                repo_path: { type: string }
```

---

## Knowledge Graph Ecosystem

The v2.0.0 knowledge graph feature opens new discoverability and positioning angles that go beyond standard code search.

### Live dry-run numbers (trelix indexing itself)

| Metric | Value |
|---|---|
| Graph nodes | 4,599 |
| Graph edges | 4,945 |
| Louvain communities | 2,409 |
| Graph build time | 0.34s |
| Top node degree | 438 (`parse`) |
| Retriever results with graph enabled | 30 (5 graph + 19 vector + 4 BM25 + 2 graph_expansion) |

### Graph visualization as a demo asset

The `trelix graph ./repo --visualize` command produces an interactive Pyvis HTML file. This is screenshot-friendly for:

- **Product Hunt gallery**: an interactive graph of a well-known OSS repo (e.g., FastAPI or Pydantic) is a compelling visual that communicates "code intelligence" immediately.
- **Show HN post**: embedding or linking to a hosted Pyvis HTML is a low-friction live demo — viewers can pan/zoom a real call graph without installing anything.
- **Social media (Twitter/X, LinkedIn)**: a graph screenshot of a large repo (thousands of nodes, visible cluster colors) is attention-grabbing and shareable with no context required.
- **README hero image**: replace or supplement the current ASCII output with a graph screenshot to signal sophistication at first glance.

### "Architecture understanding" as a distinct use-case angle

Most code-search tools are positioned around "find the relevant snippet." The knowledge graph enables a second, orthogonal positioning:

- **Onboarding**: "Understand a new codebase's architecture in 30 seconds" — generate a community-clustered graph, each cluster labeled with its dominant concept. This targets staff engineers, new-hire onboarding workflows, and open-source contributors.
- **Blast radius analysis**: graph traversal makes it possible to answer "what else does this module touch?" structurally, not just lexically.
- **Refactoring planning**: community boundaries are natural refactoring seams. Messaging this to senior engineers and architects is a differentiated angle no BM25+vector tool can match.

This framing lets trelix appeal simultaneously to:
- Juniors / new hires (onboarding)
- Staff engineers (architecture review)
- AI agent developers (graph_search MCP tool for richer context)

### MCP tools usage patterns (Claude Code / Cursor)

The two shipped MCP tools create concrete agent workflows that can be featured in docs and demos:

```
# Workflow 1 — Understand a new repo
build_knowledge_graph(repo_path="./my-repo", detect_communities=True)
→ Returns: node count, edge count, top communities with descriptions

# Workflow 2 — Graph-aware code search
graph_search_mcp(query="authentication flow", repo_path="./my-repo", depth=2)
→ Returns: 10 structurally adjacent results from degree-438 hub nodes
```

These patterns are worth publishing as Claude Code usage examples and adding to the Smithery/mcp.so listing descriptions — they differentiate trelix from simple retrieval MCP servers.

### Install path for graph features

```bash
# Full knowledge-graph install
pip install 'trelix[knowledge-graph]'   # pyvis>=0.3.2, networkx>=3.3.0

# Alias
pip install 'trelix[graph-viz]'         # same deps

# Opt-in config (zero impact when off)
graph_search_enabled: bool = False      # set True to activate 4th retrieval leg
graph_search_depth: int = 2
graph_search_max_results: int = 15

# Or via env var
TRELIX_GRAPH_SEARCH_ENABLED=true
```

### Breaking change note (v2.0.0)

`trelix graph <repo> <symbol>` (old call-graph display) is now `trelix call-graph <repo> <symbol>`. The `trelix graph` command now invokes the knowledge graph builder. Document this prominently in migration guides and PyPI changelog.

### Blog post: "From code search to code intelligence: trelix's knowledge graph"

Proposed outline for a technical blog post targeting senior engineers and AI agent builders:

1. **The problem**: vector search finds semantically similar symbols but misses structural relationships — a function called by 438 other symbols is not inherently more "relevant" by embedding alone.
2. **The approach**: build a NetworkX MultiDiGraph from Tree-sitter AST call/import/type edges, run Louvain community detection, persist to SQLite.
3. **The numbers**: 4,599 nodes, 4,945 edges, 2,409 communities, 0.34s build on trelix itself.
4. **The retrieval impact**: with `graph_search_enabled=True`, the retriever fuses graph BFS results with vector and BM25 hits — 30 results across 4 legs vs. the baseline.
5. **MCP integration**: two new tools (`build_knowledge_graph`, `graph_search_mcp`) let Claude Code and Cursor agents query the graph directly.
6. **Visualization**: Pyvis HTML — paste the generated file into a browser for an interactive exploration of your codebase.

Publish on: personal site (canonical) → Medium import → Dev.to with `canonical_url` → LinkedIn carousel.

---

## Implementation Order

```
Week 1:  trelix-mcp (Tier 1.1) — FastMCP server + PyPI publish + registry submissions
Week 1:  PyPI metadata upgrade (Tier 1.2) — classifiers + keywords + badges
Week 2:  trelix-langchain (Tier 2.1) + trelix-llama-index (Tier 2.2)
Week 2:  GitHub Action (Tier 2.3) — sairam0424/trelix-index-action
Week 3:  VS Code extension (Tier 3.1) + Homebrew tap (Tier 3.2)
Week 3:  Docker Hub (Tier 3.3)
Week 4:  Awesome list PRs (Tier 4.1) + Product Hunt launch (Tier 4.2)
Week 4:  Knowledge graph blog post + Pyvis demo page (Knowledge Graph Ecosystem section)
```

---

## v2.x Backlog (post-v2.4.0)

**Planned research & integration work:**

| Item | Priority | Notes |
|---|---|---|
| Wire `multi_query_enabled` into retriever | Medium | Allow callers to toggle HyDE expansion per query instead of global env flag |
| Rename `flare_max_retries` for clarity | Low | Consider `flare_adaptive_depth` or `flare_loop_threshold` for semantics |
| LanceDB/Qdrant `search_file_summaries` score normalization | High | Normalize file-summary scores to 0–1 range; fuse with BM25/vector consistently |
| BGE double-prefix investigation | 📋 Backlog | Evaluate BGE double-prefix strategy for improved code semantics retrieval |
| Knowledge graph — LLM-powered concept labeling per community | 📋 Backlog | Use ConceptExtractor to auto-label Louvain clusters for richer onboarding output |
| Multi-language semantic matching | 📋 Backlog | Extend query expansion to polyglot repos (e.g., TypeScript + Python calls) |

---

## v2.3.0 Backlog

**Planned features & improvements:**

| Item | Status | Priority | Notes |
|---|---|---|---|
| Wire multi_query_enabled into retriever | 📋 Backlog | P1 | Expose `multi_query_enabled` config flag to retriever for multi-query expansion routing |
| flare_max_iterations → flare_max_retries rename | 📋 Backlog | P2 | Semantic rename for consistency with agentic ReAct retry semantics |
| MCP resources exposure | 📋 Backlog | P1 | Expose symbol metadata as MCP resources (not just tools) for richer Cursor/Claude Code context |
| Dimension guard for embedding provider switches | 📋 Backlog | P2 | Runtime validation to prevent mismatched embedding dimensions when switching providers mid-lifecycle |
| PR/diff semantic review | 📋 Backlog | P1 | Semantic diff embeddings for LLM-aware code review suggestions in GitHub integration |
| Multi-repo federated search | 📋 Backlog | P2 | Search across multiple indexed repositories with rank fusion and repo-scoped filtering |

**Shipped in v2.4.0 (removed from backlog):**

| Item | Shipped | Notes |
|---|---|---|
| FederatedRetriever TTL cache | ✅ v2.4.0 | SHA-256 keyed, thread-safe, ~90% hit rate for debugging sessions |
| Multi-repo file watching | ✅ v2.4.0 | MultiRepoWatcher + `trelix watch-all` CLI |
| GitHub PR API integration | ✅ v2.4.0 | `trelix review --pr owner/repo#N`, optional --post-comments |
| MCP search_code pagination | ✅ v2.4.0 | BREAKING: envelope return type; cursor= param |
| Multi-query expansion observability | ✅ v2.4.0 | ExpandResult dataclass + 3 new telemetry columns |
| flare_max_retries rename | ✅ v2.4.0 | Old env var deprecated until v3.0.0 with DeprecationWarning |

**Shipped in v2.4.0 (removed from backlog):**

| Item | Shipped | Notes |
|---|---|---|
| FederatedRetriever TTL cache | ✅ v2.4.0 | SHA-256 keyed, thread-safe, ~90% hit rate for debugging sessions |
| Multi-repo file watching | ✅ v2.4.0 | MultiRepoWatcher + `trelix watch-all` CLI |
| GitHub PR API integration | ✅ v2.4.0 | `trelix review --pr owner/repo#N`, optional --post-comments |
| MCP search_code pagination | ✅ v2.4.0 | BREAKING: envelope return type; cursor= param |
| Multi-query expansion observability | ✅ v2.4.0 | ExpandResult dataclass + 3 new telemetry columns |
| flare_max_retries rename | ✅ v2.4.0 | Old env var deprecated until v3.0.0 with DeprecationWarning |

---

## Expected Discovery Surface After Full Rollout

| Channel | Audience | Monthly reach |
|---|---|---|
| MCP registries (official + Smithery) | AI agent developers | 50k+ |
| PyPI organic search | Python developers | 100k+ |
| LangChain integrations page | RAG builders | 500k+ |
| VS Code Marketplace | IDE users | 1M+ |
| GitHub Actions Marketplace | DevOps engineers | 100k+ |
| Homebrew | macOS developers | 50k+ |
| Awesome lists | All developers | 200k+ |
| Knowledge graph blog post + demo | Engineers & AI builders | 10k–50k (launch) |
| **Total** | | **~2M+ potential touchpoints** |
