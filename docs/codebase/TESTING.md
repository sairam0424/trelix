# Testing Patterns

## 1) Test Stack and Commands

- Primary test framework: pytest; the root config enables asyncio auto mode, strict configuration/markers, socket restrictions, and a 60-second timeout.
- Assertion and mocking tools: pytest assertions and fixtures; test files use standard mocking patterns where needed. GitHub App tests use Vitest and Supertest.

```bash
make test         # Python unit suite plus trelix-mcp tests
make test-fast    # Core Python unit tests
make test-mcp     # trelix-mcp package tests
make test-e2e     # Real subprocess / HTTP / package-install E2E tests
make test-cov     # Core unit tests with coverage
cd infra/github-app && npm test
```

`make test-e2e` intentionally runs separately from the default unit/MCP test target.

## 2) Test Layout

- `tests/unit/`: core unit tests, with shared settings and database fixtures.
- `tests/integration/`: API/provider and pipeline integration tests; some require external credentials or network access.
- `tests/e2e/`: real subprocess, MCP HTTP/stdio, and distribution-install scenarios.
- `packages/trelix-mcp/tests/`, `packages/trelix-langchain/tests/`, and `packages/trelix-llama-index/tests/`: adapter-level coverage.
- `infra/github-app/tests/`: TypeScript service tests.
- Python test names generally use `test_*.py`; the GitHub App uses Vitest.

## 3) Test Scope Matrix

| Scope | Covered? | Typical target | Notes |
|-------|----------|----------------|-------|
| Unit | Yes | Parser, config, retrieval, storage, CLI, security controls | Most core tests are in `tests/unit/`; network is disabled by default |
| Integration | Yes | Providers, connectors, API, end-to-end pipeline pieces | `tests/integration/`; network/credentials are explicitly enabled where needed |
| E2E | Yes | CLI subprocesses, MCP transports, installed distributions | `tests/e2e/`; deliberately separate from `make test` |
| Adapter | Yes | MCP, LangChain, LlamaIndex, GitHub App | Separate package/service test suites |

## 4) Mocking and Isolation Strategy

- Unit tests use temporary SQLite databases and `IndexConfig` fixtures (`tests/fixtures/`).
- `tests/unit/conftest.py` isolates environment variables and operator dotenv configuration to reduce test dependence on the developer's machine.
- The root pytest config disables network sockets by default, with explicit markers/configuration for tests that need real external services.
- E2E tests use real processes and install paths rather than mocks for those boundaries (`tests/e2e/`).
- A comprehensive per-provider mock inventory is `[TODO]`.

## 5) Coverage and Quality Signals

- Coverage tool: pytest-cov with branch coverage enabled.
- Configured threshold: 79% (`pyproject.toml`).
- Current measured coverage for this checkout: `[TODO]` (not run as part of this repository map).
- CI runs Python tests on 3.12, 3.13, and 3.14 and has separate lint/typecheck and optional-extra typecheck jobs (`.github/workflows/ci.yml`).
- The scan found no dedicated performance-testing configuration; benchmark test files and measured reports are present (`tests/unit/test_bench_vector_search.py`, `docs/reports/`).

## 6) Evidence

- `pyproject.toml`
- `Makefile`
- `.github/workflows/ci.yml`
- `tests/unit/conftest.py`
- `tests/integration/conftest.py`
- `tests/e2e/`
- `infra/github-app/package.json`
