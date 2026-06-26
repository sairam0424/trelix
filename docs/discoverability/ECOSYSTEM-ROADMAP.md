# trelix — Ecosystem Discoverability Roadmap

Research basis: 108-agent deep research, 730 tool uses, adversarial verification (June 2026).  
Target audience: AI agent developers + IDE users + DevOps/CI engineers (all three simultaneously).

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

**Implementation pattern** (FastMCP + Streamable HTTP — research finding [1]):
```python
# trelix_mcp/server.py
from fastmcp import FastMCP
from trelix.core.config import IndexConfig
from trelix.retrieval.retriever import Retriever

mcp = FastMCP("trelix", stateless_http=True, json_response=True)

@mcp.tool()
def search_code(query: str, repo_path: str, k: int = 10) -> list[dict]:
    """Search code in an indexed repository using hybrid semantic + keyword search."""
    config = IndexConfig(repo_path=repo_path)
    ctx = Retriever(config).retrieve(query)
    return [{"file": r.file.rel_path, "symbol": r.symbol.qualified_name,
             "lines": f"{r.symbol.line_start}-{r.symbol.line_end}",
             "score": round(r.score, 4), "body": r.symbol.body[:500]} 
            for r in ctx.results[:k]]

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
BM25+vector search, call-graph expansion, adaptive query planning, and LLM 
synthesis. Zero infrastructure (SQLite). MCP-compatible.

`pip install trelix` | MIT License
```

---

## Tier 4.2 — Product Hunt + Show HN

**Product Hunt**: Launch on a Tuesday/Wednesday. Key sections: tagline (≤60 chars), description, gallery screenshots of `trelix search` and `trelix ask` output. Request maker comment first.

**Tagline options:**
- "Search any codebase like you talk to a developer"
- "Code intelligence that understands calls, imports, and types"

**Show HN title**: `Show HN: trelix – hybrid code search with Tree-sitter, BM25+vectors, and call graphs`

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

## Implementation Order

```
Week 1:  trelix-mcp (Tier 1.1) — FastMCP server + PyPI publish + registry submissions
Week 1:  PyPI metadata upgrade (Tier 1.2) — classifiers + keywords + badges
Week 2:  trelix-langchain (Tier 2.1) + trelix-llama-index (Tier 2.2)
Week 2:  GitHub Action (Tier 2.3) — sairam0424/trelix-index-action
Week 3:  VS Code extension (Tier 3.1) + Homebrew tap (Tier 3.2)
Week 3:  Docker Hub (Tier 3.3)
Week 4:  Awesome list PRs (Tier 4.1) + Product Hunt launch (Tier 4.2)
```

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
| **Total** | | **~2M potential touchpoints** |
