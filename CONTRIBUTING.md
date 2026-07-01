# Contributing to trelix

Thank you for your interest in contributing! This guide covers dev setup, testing, and how to add a new language parser or LLM provider.

## Development Setup

```bash
git clone https://github.com/sairam0424/trelix
cd trelix

# Create virtualenv
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install in editable mode with all dev + optional deps (v2.1.0)
make install-dev
# equivalent: pip install -e ".[bge-code,plaid,lance,serve,dev]"
# Optional extras:
#   [bge-code]        — BGE-Code embeddings (requires torch, transformers)
#   [plaid]           — Plaid financial data integration
#   [lance]           — LanceDB vector store for large-scale deployments
#   [serve]           — REST API server (FastAPI + Uvicorn)
#   [knowledge-graph] — Knowledge graph + visualization (pyvis>=0.3.2, networkx>=3.3.0)
#   [graph-viz]       — Alias for [knowledge-graph]
#   [dev]             — testing, linting, type-checking (always included)

# Standard dev setup (no graph visualization)
pip install -e ".[local,dev]"

# Include graph visualization (pyvis + networkx)
pip install -e ".[local,dev,knowledge-graph]"

# Copy environment template
cp .env.example .env
# Edit .env — at minimum set TRELIX_EMBEDDER_PROVIDER=local
```

## Running Tests

```bash
make test           # full suite with coverage (1325 unit + 16 integration tests)

make test           # full suite with coverage (1370+ unit + 16 integration tests)
make test-fast      # unit tests only (no API calls, fast)
make lint           # ruff check + ruff format (auto-formats before diff-check, cross-platform safe)
make format         # ruff format
make typecheck      # mypy
```

**Note on CI checks (v2.1.0):** The ruff format step now runs as part of linting — files are auto-formatted before the diff-check, ensuring cross-platform consistency (Windows CRLF vs Unix LF).

### Running specific test subsets

```bash
# Unit tests only — no credentials needed
pytest -m "not integration"

# Live integration tests — require Azure or AWS credentials
pytest tests/integration/
# tests/integration/test_llm_e2e.py covers Azure + Bedrock chat and embeddings;
# individual tests skip gracefully when the relevant credentials are absent
```

## Branch Strategy

```
main          ← stable releases only (do not push directly)
  └─ develop  ← integration branch — open PRs here
       └─ feature/<name>  ← your work
```

1. Fork the repo and create a branch from `develop`: `git checkout -b feature/my-feature develop`
2. Make your changes with tests
3. Open a PR targeting `develop` (not `main`)

## Extension Points

### trelix/graph/ — Knowledge Graph module

The graph module lives at `src/trelix/graph/` and is organized as:

| File | Responsibility |
|------|----------------|
| `code_graph.py` | CodeGraph — NetworkX MultiDiGraph over SQLite edge tables |
| `community.py` | Community detection (Louvain/Girvan-Newman) |
| `persistence.py` | Save/load community + centrality to graph_metadata table |
| `concepts.py` | LLM semantic concept extraction (crash-safe) |
| `builder.py` | GraphBuilder — orchestrates the full pipeline |
| `visualizer.py` | Pyvis HTML export (requires `trelix[knowledge-graph]`) |
| `search.py` | BFS graph_search function (4th retrieval leg) |

Tests live in `tests/unit/test_graph_*.py`. All graph tests can run without pyvis or any LLM configured.

**Opt-in config keys** (all default to off — zero impact when disabled):

| Key | Default | Env var |
|-----|---------|---------|
| `graph_search_enabled` | `False` | `TRELIX_GRAPH_SEARCH_ENABLED=true` |
| `graph_search_depth` | `2` | — |
| `graph_search_max_results` | `15` | — |

**Adding a new graph algorithm:**

1. Add the implementation to the most appropriate existing file or create a new file under `src/trelix/graph/`
2. Expose it through `GraphBuilder` in `builder.py` so the pipeline can call it
3. If it requires a new optional dependency, add an extras group to `pyproject.toml` and document it here
4. Write tests in `tests/unit/test_graph_<name>.py` — mock any LLM calls; do not require pyvis

### trelix/retrieval/ — Query Enhancement Modules

The retrieval enhancement modules live at `src/trelix/retrieval/` and are organized as:

| File | Responsibility |
|------|----------------|
| `query_expansion.py` | HyDEExpander (synthetic snippet embedding), MultiQueryExpander (N-variant recall) |
| `flare.py` | FLARELoop — confidence-gated re-retrieval, _contains_uncertainty phrase check |
| `telemetry.py` | TelemetryWriter — crash-safe per-query latency/intent recorder |

All three modules are crash-safe (return empty/original on any failure) and gated by config flags.

**Opt-in config keys** (all default to off — zero impact when disabled):

| Key | Default | Env var |
|-----|---------|---------|
| `query_expansion_enabled` | `False` | `TRELIX_QUERY_EXPANSION_ENABLED=true` |
| `flare_enabled` | `False` | `TRELIX_FLARE_ENABLED=true` |
| `telemetry_enabled` | `False` | `TRELIX_TELEMETRY_ENABLED=true` |

**Adding a new query enhancement:**

1. Add the implementation under `src/trelix/retrieval/`
2. Ensure any failure path returns the original query or empty results — never raises
3. Gate the feature with a config flag defaulting to `False`
4. Write tests in `tests/unit/test_retrieval_<name>.py` — mock any LLM calls

### trelix/eval/ — Evaluation Harness

The evaluation harness lives at `src/trelix/eval/` and is organized as:

| File | Responsibility |
|------|----------------|
| `ndcg.py` | Pure-Python ndcg_at_k, recall_at_k, mrr — no pandas dependency |
| `harness.py` | EvalHarness.run(golden_path) — reads JSONL, retrieves, returns aggregate metrics |

**Usage:**

```bash
trelix eval --golden .trelix/golden.jsonl
```

**Golden file format** (one line per query):

```json
{"query": "how does auth work", "relevant_files": ["src/auth.py"]}
```

**Adding new metrics:**

1. Add the pure-Python metric function to `src/trelix/eval/ndcg.py`
2. Wire it into `EvalHarness.run()` in `src/trelix/eval/harness.py`
3. Write tests in `tests/unit/test_eval_<name>.py` — no LLM calls required for metric functions

### trelix/agent/ — ReAct Agentic Loop

The agent module lives at `src/trelix/agent/` and implements a ReAct (Reason + Act) loop over the trelix retrieval stack:

| File | Responsibility |
|------|----------------|
| `actions.py` | ActionType enum, AgentAction, Observation, Turn dataclasses |
| `history.py` | TurnHistory, HistoryCompressor (token-budget context trimming) |
| `tools.py` | OpenAI function-calling tool schemas for 4 actions |
| `loop.py` | AgentLoop orchestrator — ReAct Thought→Action→Observation cycle |

All agent tests live in `tests/unit/test_agent_*.py`. No LLM calls are needed — `TrelixChatClient` is mocked throughout the test suite.

### trelix/analysis/ — Program Analysis

The analysis module lives at `src/trelix/analysis/` and provides static program analysis on top of the indexed codebase:

| File | Responsibility |
|------|----------------|
| `defuse.py` | DataFlowExtractor — tree-sitter def-use chain extraction (crash-safe) |
| `taint.py` | TaintAnalyzer — Semgrep CLI wrapper (requires `trelix[taint]`) |

To use taint analysis, install the optional extra:

```bash
pip install -e ".[taint]"
```

Tests that exercise `TaintAnalyzer` mock the subprocess call, so the full test suite runs without Semgrep installed.

### trelix/embedder/sparse.py and trelix/store/sparse_store.py — Sparse Embeddings

`SparseEmbedder` produces `{token_id: weight}` SPLADE-Code vectors and requires the optional extra:

```bash
pip install -e ".[sparse]"
```

`SparseStore` is a SQLite inverted index — no external service or vector database is needed. Tests run without torch: `SparseEmbedder` returns `{}` automatically when `_TORCH_AVAILABLE=False`, so the sparse test suite passes in any environment.

### Adding a New Language Parser

1. Create `src/trelix/indexing/parser/extractors/<language>.py`
2. Subclass `BaseParser` from `src/trelix/indexing/parser/base.py`
3. Implement `parse(source: str, file_id: int) -> ParseResult`
4. Register in `src/trelix/indexing/parser/registry.py`: add `Language.YOURLANG: YourParser()`
5. Add file extensions to `EXTENSION_MAP` in `src/trelix/indexing/walker.py`
6. Add `Language.YOURLANG` to `WalkerConfig.languages` default list in `src/trelix/core/config.py`
7. Write tests in `tests/unit/test_parser_<language>.py` with fixture files

### Embedder Providers

trelix v2.1.0 ships with built-in support for multiple embedding backends:

- **Local embeddings** (`local`) — Uses transformers library (default, no API keys needed)
- **BGE-Code-v1** (`bge-code`) — BAAI General Embedding for code, optimized for semantic code search
- **Nomic CodeRankEmbed** (`nomic-code`) — Open-source embeddings specialized for code ranking
- **Azure OpenAI Embeddings** — Enterprise deployment via Azure; set `TRELIX_EMBEDDER_PROVIDER=azure`
- **Bedrock** — AWS-hosted embeddings via Bedrock

To use BGE-Code or CodeRank embeddings, install the optional extra:

```bash
pip install -e ".[bge-code]"
# Then set TRELIX_EMBEDDER_PROVIDER=bge-code in .env
```

### Adding a New LLM Provider (v2.1.0)

trelix uses a provider-agnostic `TrelixChatClient` ABC (`src/trelix/llm/client.py`). All five built-in backends (`OpenAIBackend`, `AnthropicBackend`, `BedrockBackend`, `VertexBackend`, `LiteLLMBackend`) implement the same three methods: `complete()`, `stream()`, and `tool_call()`. Adding a new provider requires zero changes to business logic (chunker, synthesizer, planner, graph_rag).

1. Create `src/trelix/llm/providers/<name>_backend.py`
2. Subclass `TrelixChatClient` and implement `complete()`, `stream()`, `tool_call()`
3. Add a `case "<name>":` branch to `src/trelix/llm/factory.py` (`build_chat_client()`)
4. Add credential fields to `LLMConfig` in `src/trelix/core/config.py`
5. Add `"<name>"` to the `Literal` type of `LLMConfig.provider`
6. Add an optional dep group to `pyproject.toml` if the provider SDK is not already a dependency
7. Write unit tests in `tests/unit/test_llm_<name>_backend.py` — mock the provider SDK, no real API calls

No changes are needed in `chunker.py`, `synthesizer.py`, `planner/agent.py`, or `graph_rag.py`.

## Coding Standards

- Python 3.11+ type hints everywhere
- Line length: 100 chars (ruff enforced)
- No mutable default arguments
- New objects, never mutate in-place
- Functions > 50 lines should be split

## Reporting Issues

Use the GitHub issue templates:
- **Bug report** — include Python version, OS, trelix version, minimal reproduction
- **Feature request** — describe the use case, not just the solution

## Questions

Open a [GitHub Discussion](https://github.com/sairam0424/trelix/discussions) for questions.

## Versioning & Stability Policy

trelix follows [Semantic Versioning 2.0.0](https://semver.org/). Current version: **2.1.0**.

trelix follows [Semantic Versioning 2.0.0](https://semver.org/). Current version: **2.2.0**.

### Stable public API (guaranteed not to change without a major version bump)

- **CLI commands and flags**: `trelix index`, `trelix search`, `trelix ask`, `trelix query`, `trelix stats`, `trelix watch`, `trelix update-index`, `trelix migrate-vectors` and all documented flags
- **Python API**: `IndexConfig`, `EmbedderConfig`, `LLMConfig`, `Indexer`, `Retriever`, `TrelixChatClient`, `ChatMessage`, `ChatResponse`, `ToolCallResponse`, `build_chat_client`, `BaseEmbedder`, `make_embedder`
- **Sub-package interfaces**: `TrelixRetriever` (trelix-langchain), `TrelixIndexRetriever` (trelix-llama-index), MCP tool signatures (trelix-mcp)
- **Environment variable names**: all `TRELIX_*` env vars documented in `.env.example`

### What counts as a breaking change

- Removing or renaming a public class, method, or CLI flag
- Changing a method signature in an incompatible way
- Changing the SQLite schema in a way that requires re-indexing
- Removing a previously supported Python version

**CLI command renames** (e.g. `trelix graph` → `trelix call-graph` in v2.0.0) are breaking changes and must be documented under a `### Breaking Changes` heading in `CHANGELOG.md` for the relevant release, alongside a migration note showing the old and new invocation.

### Deprecation policy

- Deprecated features are marked with `DeprecationWarning` and noted in the CHANGELOG
- Deprecated features are maintained for at least **one minor version** before removal in a major version
- The CLI will print a deprecation notice on first use of deprecated flags

### Python version support

- Supported: Python 3.11, 3.12
- Dropped versions are announced one minor release in advance

---

## Working on Sub-packages

trelix ships three integration packages. To work on them:

```bash
# Install a sub-package in editable mode
pip install -e packages/trelix-mcp/
pip install -e packages/trelix-langchain/
pip install -e packages/trelix-llama-index/

# Run tests for a specific package
python -m pytest packages/trelix-mcp/tests/ --override-ini="testpaths=packages" -v
python -m pytest packages/trelix-langchain/tests/ --override-ini="testpaths=packages" -v
python -m pytest packages/trelix-llama-index/tests/ --override-ini="testpaths=packages" -v
```

Each package has its own `pyproject.toml` and `tests/` directory. The `src/` layout mirrors the main package.
