# Codebase Concerns

## 1) Top Risks (Prioritized)

| Severity | Concern | Evidence | Impact | Suggested action |
|----------|---------|----------|--------|------------------|
| Medium | Large, high-churn CLI module | `src/trelix/cli/main.py` is the largest Python source file and had 70 changes in the last 90 days | CLI behavior and output changes are concentrated in a difficult-to-review file | Keep changes command-scoped; consider extracting cohesive command groups when touching related code |
| Medium | Strict type checking has a recorded optional-dependency error baseline | `.github/workflows/ci.yml` documents seven mypy errors across four files in its optional-extra job | New errors outside the baseline are gated, but known errors remain | Retire baseline entries as their underlying typing issues are fixed |
| Medium | Ruff security/bugbear rule exclusions record existing debt | `pyproject.toml` lists scoped/global ignores including S110, S608, B905, B904, and others with explanatory counts | Those specific checks are not uniformly enforced until the debt is addressed | Reduce each ignore with focused fixes and tests; retain documented rationale for any intentional exception |
| Low | Documentation architecture count appears stale | `docs/architecture.md` says 140 source modules; the current `src/trelix/` Python file listing contains 175 files | Readers may get an outdated sense of the core's size | Update the header/count when that architecture document is next maintained |

## 2) Technical Debt

| Debt item | Why it exists | Where | Risk if ignored | Suggested fix |
|-----------|---------------|-------|-----------------|---------------|
| Ruff ignored-rule backlog | Rules were enabled while existing violations were retained as explicit ignores | `pyproject.toml` | Relevant classes of mistakes can reappear in ignored areas | Pay down categories incrementally and remove ignores |
| Optional-extra mypy baseline | Some third-party types and integrations currently yield known type errors | `.github/workflows/ci.yml` | Baseline can conceal additional same-code errors in baseline files | Improve annotations or dependency stubs and shrink the baseline |
| Large CLI module | CLI commands and presentation accumulated in a single module | `src/trelix/cli/main.py` | Increased merge conflicts and regression risk | Extract command groups with behavior-preserving tests |
| No single integration reliability policy | Providers and connectors have different timeout and retry behavior | `src/trelix/core/retry.py`, provider and connector modules | Operators must understand service-specific failure behavior | Document/test per-integration timeout and retry contracts; current comprehensive matrix is `[TODO]` |

The scan detected no actionable production `TODO` or `FIXME` item: its one production-code match is explicitly documented as not being a TODO (`scripts/mutation.py`).

## 3) Security Concerns

This section records trust boundaries and existing safeguards from inspected files; it is not a security audit or an assertion that a vulnerability exists.

| Risk | OWASP category (if applicable) | Evidence | Current mitigation | Gap |
|------|--------------------------------|----------|--------------------|-----|
| Optional hosted providers and connectors can transmit queries, code context, or repository artifacts outside the local machine | N/A | `.env.example`, `src/trelix/llm/`, `src/trelix/indexing/connectors/` | Local provider and no-credential modes are supported; provider choice is configurable | Operators must select providers appropriate to their data policy; project-wide data-egress matrix is `[TODO]` |
| REST and MCP surfaces accept repository/path inputs and therefore rely on boundary enforcement | N/A | `src/trelix/api/app.py`, `src/trelix/core/confinement.py`, `packages/trelix-mcp/src/trelix_mcp/confinement.py` | Repository-root confinement and symlink-store checks have dedicated tests | Deployment-specific root and authentication settings still need operator review |
| GitHub App handles untrusted webhook and PR data | N/A | `infra/github-app/src/webhook.ts`, `infra/github-app/src/` | Webhook verification, bounded review queue, isolated child environment, and policy modules are present | Ongoing dependency and deployment configuration review remains necessary |

## 4) Performance and Scaling Concerns

| Concern | Evidence | Current symptom | Scaling risk | Suggested improvement |
|---------|----------|-----------------|-------------|-----------------------|
| Default SQLite vector search may become slower as indexes grow | `src/trelix/store/scale_advisory.py`, `docs/reports/vector-search-bench-2026-10-07-apple-m4-shared.json` | Code emits a flat-scan advisory based on vector count and width | Large indexes may need a different vector backend | Measure on the target workload and consider the optional Qdrant/LanceDB stores |
| Local embeddings and optional local reranking use model resources | `pyproject.toml` local extra, `src/trelix/embedder/` | Model loading and local inference consume disk, memory, and CPU/GPU | Large models or repositories can make local indexing/query latency high | Select provider and worker settings to match host capacity; benchmarks are workload-specific |
| Dedicated performance CI configuration was not detected | `.github/workflows/ci.yml`, `tests/unit/test_bench_vector_search.py`, `docs/reports/vector-search-bench-2026-10-07-apple-m4-shared.json` | Benchmark tests and reports exist, but no dedicated performance gate was found | Regressions may not be caught automatically | `[TODO]` Determine whether performance thresholds should run in CI |

## 5) Fragile/High-Churn Areas

| Area | Why fragile | Churn signal | Safe change strategy |
|------|-------------|--------------|----------------------|
| `CHANGELOG.md` | Release facts and extensive per-release history | 189 changes in the last 90 days | Use the release checklist and verify against shipped behavior |
| `src/trelix/cli/main.py` | Large command module and user-facing behavior | 70 changes in the last 90 days; 225.3 KB in scan | Keep changes narrow and add command-level regression coverage |
| `pyproject.toml` | Core dependencies, extras, tooling, and coverage gates are centralized | 56 changes in the last 90 days | Update dependency floor/extra tests and CI together |
| `src/trelix/core/config.py` | Shared configuration controls many pipeline modules | 40 changes in the last 90 days | Preserve defaults and environment-resolution tests |
| `packages/trelix-mcp/pyproject.toml` | Adapter compatibility depends on core API floors | 41 changes in the last 90 days | Keep release floor checks and MCP package tests aligned |

Churn counts are from the repository scan's 90-day history snapshot and are not a prediction of future change.

## 6) `[ASK USER]` Questions

No unresolved team-intent decisions were identified in the inspected intent and configuration documents. The documented architecture-count mismatch is recorded above without changing the existing architecture document.

## 7) Evidence

- Repository scan and recent Git history were inspected during this map; the temporary scan output was removed after verification.
- `docs/architecture.md`
- `pyproject.toml`
- `.github/workflows/ci.yml`
- `src/trelix/cli/main.py`
- `src/trelix/store/scale_advisory.py`
- `tests/unit/test_bench_vector_search.py`
- `infra/github-app/src/`
