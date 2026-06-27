# Contributing to trelix

Thank you for your interest in contributing! This guide covers dev setup, testing, and how to add a new language parser or LLM provider.

## Development Setup

```bash
git clone https://github.com/sairam0424/trelix
cd trelix

# Create virtualenv
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install in editable mode with all dev + optional deps
make install-dev
# equivalent: pip install -e ".[local,rerank,dev]"

# Copy environment template
cp .env.example .env
# Edit .env — at minimum set TRELIX_EMBEDDER_PROVIDER=local
```

## Running Tests

```bash
make test           # full suite with coverage (929 unit + 16 integration tests)
make test-fast      # unit tests only (no API calls, fast)
make lint           # ruff check
make format         # ruff format
make typecheck      # mypy
```

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

### Adding a New LLM Provider (v0.7.0)

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
