# Contributing to trelix

Thank you for your interest in contributing! This guide covers dev setup, testing, and how to add a new language parser or LLM provider.

## Development Setup

```bash
git clone https://github.com/sairam0424/trelix
cd trelix

# Create virtualenv
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install in editable mode with all dev + optional deps (v2.0.0)
make install-dev
# equivalent: pip install -e ".[bge-code,plaid,lance,serve,dev]"
# Optional extras:
#   [bge-code]  — BGE-Code embeddings (requires torch, transformers)
#   [plaid]     — Plaid financial data integration
#   [lance]     — LanceDB vector store for large-scale deployments
#   [serve]     — REST API server (FastAPI + Uvicorn)
#   [dev]       — testing, linting, type-checking (always included)

# Copy environment template
cp .env.example .env
# Edit .env — at minimum set TRELIX_EMBEDDER_PROVIDER=local
```

## Running Tests

```bash
make test           # full suite with coverage (929 unit + 16 integration tests)
make test-fast      # unit tests only (no API calls, fast)
make lint           # ruff check + ruff format (auto-formats before diff-check, cross-platform safe)
make format         # ruff format
make typecheck      # mypy
```

**Note on CI checks (v2.0.0):** The ruff format step now runs as part of linting — files are auto-formatted before the diff-check, ensuring cross-platform consistency (Windows CRLF vs Unix LF).

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

### Adding a New Language Parser

1. Create `src/trelix/indexing/parser/extractors/<language>.py`
2. Subclass `BaseParser` from `src/trelix/indexing/parser/base.py`
3. Implement `parse(source: str, file_id: int) -> ParseResult`
4. Register in `src/trelix/indexing/parser/registry.py`: add `Language.YOURLANG: YourParser()`
5. Add file extensions to `EXTENSION_MAP` in `src/trelix/indexing/walker.py`
6. Add `Language.YOURLANG` to `WalkerConfig.languages` default list in `src/trelix/core/config.py`
7. Write tests in `tests/unit/test_parser_<language>.py` with fixture files

### Embedder Providers

trelix v2.0.0 ships with built-in support for multiple embedding backends:

- **Local embeddings** (`local`) — Uses transformers library (default, no API keys needed)
- **BGE-Code-v1** (`bge-code`) — BAAI General Embedding for code, optimized for semantic code search
- **Nomic CodeRankEmbed** (`code-rank`) — Open-source embeddings specialized for code ranking
- **Azure OpenAI Embeddings** — Enterprise deployment via Azure; set `TRELIX_EMBEDDER_PROVIDER=azure`
- **Bedrock** — AWS-hosted embeddings via Bedrock

To use BGE-Code or CodeRank embeddings, install the optional extra:

```bash
pip install -e ".[bge-code]"
# Then set TRELIX_EMBEDDER_PROVIDER=bge-code in .env
```

### Adding a New LLM Provider (v2.0.0)

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

trelix follows [Semantic Versioning 2.0.0](https://semver.org/). Current version: **2.0.0**.

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
