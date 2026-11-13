# External Integrations

## 1) Integration Inventory

| System | Type (API/DB/Queue/etc) | Purpose | Auth model | Criticality | Evidence |
|--------|---------------------------|---------|------------|-------------|----------|
| OpenAI-compatible APIs, Azure OpenAI, Anthropic, AWS Bedrock, Vertex AI, LiteLLM | Model APIs | Chat planning/synthesis and, depending on provider, embeddings | Provider credentials or configured local-compatible endpoint | Optional; required for selected hosted model features | `src/trelix/llm/providers/`, `src/trelix/embedder/`, `.env.example` |
| Sentence Transformers and Hugging Face model files | Local model runtime | Local embeddings and optional cross-encoder reranking | Local model/cache access; remote model code requires explicit opt-in where applicable | Optional; supports local use | `pyproject.toml`, `src/trelix/embedder/base.py` |
| SQLite, sqlite-vec, SQLite FTS5 | Local data store | File/symbol metadata, vectors, and BM25 search | Local filesystem permissions | Core | `src/trelix/store/db.py`, `src/trelix/store/vector.py` |
| Qdrant and LanceDB | Vector stores | Optional alternate vector storage | Qdrant endpoint configuration; LanceDB local storage | Optional | `src/trelix/store/vector_qdrant.py`, `src/trelix/store/vector_lance.py` |
| Jira, TestRail, Xray Cloud, Linear | External artifact APIs | Sync ticket/test artifacts and link them to indexed code | Per-service credentials in configuration | Optional | `src/trelix/indexing/connectors/` |
| GitHub REST API and webhooks | SCM API / webhook | Fetch and review pull requests; publish Check results | Actions `GITHUB_TOKEN` or GitHub App credentials and verified webhook delivery | Optional | `src/trelix/review/github.py`, `infra/github-app/src/webhook.ts`, `infra/github-app/README.md` |
| Model Context Protocol clients | Tool protocol | Expose search/indexing tools via stdio or Streamable HTTP | HTTP bearer token when configured for hosted/non-loopback transport; stdio can use configured root confinement | Optional adapter | `packages/trelix-mcp/src/trelix_mcp/server.py`, `packages/trelix-mcp/pyproject.toml` |
| OpenTelemetry collectors/exporters | Observability | Export traces and metrics when enabled | OTLP endpoint configuration | Optional | `src/trelix/retrieval/otel_tracing.py`, `docs/OBSERVABILITY.md` |

## 2) Data Stores

| Store | Role | Access layer | Key risk | Evidence |
|-------|------|--------------|----------|----------|
| `.trelix/index.db` | Default local metadata, symbols, and FTS index | `Database` in `src/trelix/store/db.py` | Local disk capacity and schema compatibility | `src/trelix/store/db.py`, `src/trelix/core/config.py` |
| sqlite-vec | Default vector index | `src/trelix/store/vector.py` | Flat-scan performance can become limiting at scale; advisory is implemented | `src/trelix/store/scale_advisory.py`, `docs/reports/vector-search-bench-2026-10-07-apple-m4-shared.json` |
| Qdrant / LanceDB | Optional vector backends | `src/trelix/store/vector_qdrant.py`, `src/trelix/store/vector_lance.py` | External service configuration or additional local dependencies | `pyproject.toml`, store modules |
| Audit SQLite database | Optional append-only API audit trail | `src/trelix/audit/store.py` | Separate audit-store retention and filesystem management | `src/trelix/audit/`, `docs/AUDIT.md` |

## 3) Secrets and Credentials Handling

- Core provider credentials are supplied through environment variables or the operator-owned settings file resolved by `TRELIX_CONFIG_FILE` / XDG config. `.env.example` documents variable names and examples, not live credentials.
- GitHub App configuration is read by `infra/github-app/src/config.ts`; webhook signing and GitHub App credentials are required for that hosted integration.
- Rotation procedures are provider-specific and not centralized in the codebase: `[TODO]`.

## 4) Reliability and Failure Behavior

- Shared retry/backoff behavior is implemented in `src/trelix/core/retry.py` with `tenacity`.
- Indexing batches are bounded and report a `PartialIndexError` rather than silently presenting a failed embedding write as a complete index (`src/trelix/indexing/indexer.py`).
- REST and MCP entry points apply request, result-size, transport, and repository-confinement checks (`src/trelix/api/app.py`, `packages/trelix-mcp/src/trelix_mcp/`).
- The GitHub App has a bounded in-process review queue; no durable external queue was found in its runtime configuration (`infra/github-app/src/review-intake.ts`, `infra/github-app/src/server.ts`).
- Provider-specific timeout and retry coverage varies; a single cross-provider timeout policy is `[TODO]`.

## 5) Observability for Integrations

- OpenTelemetry tracing and metrics are optional and controlled by configuration; OTLP export is configured through standard `OTEL_*` variables (`src/trelix/retrieval/otel_tracing.py`, `docs/OBSERVABILITY.md`).
- Server-side structured logging uses `structlog` and standard logging (`src/trelix/core/logging_setup.py`).
- A centralized integration health dashboard or SLO definition was not found: `[TODO]`.

## 6) Evidence

- `pyproject.toml`
- `.env.example`
- `src/trelix/core/config.py`
- `src/trelix/store/`
- `src/trelix/indexing/connectors/`
- `packages/trelix-mcp/src/trelix_mcp/server.py`
- `infra/github-app/src/`
