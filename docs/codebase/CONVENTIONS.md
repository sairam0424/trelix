# Coding Conventions

## 1) Naming Rules

| Item | Rule | Example | Evidence |
|------|------|---------|----------|
| Python files | `snake_case` | `indexer.py`, `test_retriever_core.py` | `src/trelix/`, `tests/unit/` |
| Functions and variables | `snake_case` | `make_embedder`, `repo_path` | `src/trelix/core/config.py` |
| Classes and exceptions | `PascalCase` | `Indexer`, `PartialIndexError` | `src/trelix/indexing/indexer.py` |
| Constants and environment keys | `UPPER_SNAKE_CASE`; environment settings generally use `TRELIX_` prefix | `SCHEMA_VERSION`, `TRELIX_CONFIG_FILE` | `src/trelix/store/db.py`, `src/trelix/core/config.py` |
| Python tests | `test_*.py`; test classes commonly use `Test*` | `test_config.py`, `TestLanguage` | `tests/unit/test_config.py` |

## 2) Formatting and Linting

- Formatter: Ruff (`ruff format`), configured in `pyproject.toml`.
- Linter: Ruff (`ruff check`), configured in `pyproject.toml`.
- Enforced Ruff rule families include `E`, `F`, `I`, `UP`, `S`, and `B`, with documented existing exclusions/ignores in the same config.
- Python line length is 100; Ruff targets Python 3.12.
- mypy uses `strict = true` with `ignore_missing_imports = true`; CI also checks selected optional-dependency combinations.
- Run commands: `make lint`, `make format`, `make typecheck`.

## 3) Import and Module Conventions

- Imports are grouped and sorted by Ruff's `I` rules; core imports use absolute `trelix.*` module paths.
- The root package defines an explicit `__all__` for its public convenience API in `src/trelix/__init__.py`.
- No root path alias configuration was found in the inspected Python manifests.

## 4) Error and Logging Conventions

- Configuration uses Pydantic validation; indexing and retrieval define explicit exceptions for incomplete or invalid operations (`src/trelix/core/config.py`, `src/trelix/indexing/indexer.py`).
- CLI code maps failures to user-facing messages and command exit codes (`src/trelix/cli/main.py`); REST and MCP code validate requests at their boundaries (`src/trelix/api/app.py`, `packages/trelix-mcp/src/trelix_mcp/arguments.py`).
- Python modules use named standard-library loggers; `structlog` is a core dependency for structured logging (`pyproject.toml`, `src/trelix/core/logging_setup.py`).
- Secrets are expected from environment or operator-owned configuration, not committed dotenv files (`.env.example`, `src/trelix/core/config.py`).
- A complete error/logging/redaction matrix across all adapters is `[TODO]`.

## 5) Testing Conventions

- Python tests use pytest and are grouped under `tests/unit/`, `tests/integration/`, and `tests/e2e/`; adapter packages have their own tests.
- Unit tests use shared fixtures such as `tmp_db` and `index_config` and isolate environment-dependent settings (`tests/unit/conftest.py`, `tests/fixtures/`).
- Network access is disabled by default for the root pytest suite; explicitly marked integration tests and subprocess end-to-end tests cover external boundaries (`pyproject.toml`, `tests/integration/conftest.py`, `tests/e2e/conftest.py`).
- A point-in-time current coverage percentage is `[TODO]`; the configured branch coverage threshold is 79% (`pyproject.toml`).

## 6) Evidence

- `pyproject.toml`
- `.pre-commit-config.yaml`
- `.github/workflows/ci.yml`
- `src/trelix/core/config.py`
- `tests/unit/conftest.py`
