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
# equivalent: pip install -e ".[bge-code,plaid,lance,serve,dev]"
# Optional extras:
#   [bge-code]        — BGE-Code embeddings (requires torch, transformers)
#   [plaid]           — Plaid financial data integration
#   [lance]           — LanceDB vector store for large-scale deployments
#   [serve]           — REST API server (FastAPI + Uvicorn)
#   [knowledge-graph] — Knowledge graph + visualization (pyvis>=0.3.2, networkx>=3.3.0)
#   [graph-viz]       — Alias for [knowledge-graph]
#   [watch]           — Multi-repo file watching (watchfiles)
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
make test           # unit + MCP: 2,861 unit + 102 MCP = 2,963 tests (no coverage)
make test-fast      # unit tests only, 2,861 (no API calls, fast)
make test-cov       # unit tests with the coverage report
make lint           # ruff check + ruff format (auto-formats before diff-check, cross-platform safe)
make format         # ruff format
make typecheck      # mypy
```

**Note on CI checks:** The ruff format step runs as part of linting — files are auto-formatted before the diff-check, ensuring cross-platform consistency (Windows CRLF vs Unix LF).

### Running specific test subsets

```bash
# Unit tests only — no credentials needed
# Collects 2,861 of 2,965 and deselects the 104 integration tests.
pytest -m "not integration"

# Live integration tests — require Azure or AWS credentials (104 tests)
pytest tests/integration/
# tests/integration/test_llm_e2e.py covers Azure + Bedrock chat and embeddings;
# individual tests skip gracefully when the relevant credentials are absent
```

You do not apply the `integration` marker by hand: `tests/integration/conftest.py`
applies it to every test in that directory, so a new file there is credential-gated on
arrival. The marker is registered in `pyproject.toml` under `[tool.pytest.ini_options]
markers`, and `addopts` carries `--strict-markers` so a typo becomes a collection error.
Both matter — while the marker was unregistered, `-m "not integration"` matched nothing
and quietly ran the entire suite including the live Azure/Bedrock tests.

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
| `graph_search_enabled` | `False` | `TRELIX_RETRIEVAL_GRAPH_SEARCH_ENABLED=true` |
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

trelix ships with built-in support for multiple embedding backends:

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

### Adding a New LLM Provider

trelix uses a provider-agnostic `TrelixChatClient` ABC (`src/trelix/llm/client.py`). All five built-in backends (`OpenAIBackend`, `AnthropicBackend`, `BedrockBackend`, `VertexBackend`, `LiteLLMBackend`) implement the same three methods: `complete()`, `stream()`, and `tool_call()`. Adding a new provider requires zero changes to business logic (chunker, synthesizer, planner, graph_rag).

1. Create `src/trelix/llm/providers/<name>_backend.py`
2. Subclass `TrelixChatClient` and implement `complete()`, `stream()`, `tool_call()`
3. Add a `case "<name>":` branch to `src/trelix/llm/factory.py` (`build_chat_client()`)
4. Add credential fields to `LLMConfig` in `src/trelix/core/config.py`
5. Add `"<name>"` to the `Literal` type of `LLMConfig.provider`
6. Add an optional dep group to `pyproject.toml` if the provider SDK is not already a dependency
7. Write unit tests in `tests/unit/test_llm_<name>_backend.py` — mock the provider SDK, no real API calls

No changes are needed in `chunker.py`, `synthesizer.py`, `planner/agent.py`, or `graph_rag.py`.

### Adding federation cache configuration

`FederatedRetriever` (`src/trelix/federation/retriever.py`) ships with a SHA-256 keyed, thread-safe TTL cache:

```python
from trelix.federation.retriever import FederatedRetriever

# Default: 120-second TTL
fed = FederatedRetriever(registry)

# Custom TTL
fed = FederatedRetriever(registry, cache_ttl=300.0)

# Disable caching entirely
fed = FederatedRetriever(registry, cache_ttl=0)

# Inspect cache performance
stats = fed.cache_stats()   # -> {"hits": int, "misses": int, "size": int}

# Invalidate all cached results
fed.clear_cache()
```

The cache achieves ~90% hit rate for typical debugging sessions. Set `cache_ttl=0` in tests to avoid stale results across test cases.

### Adding multi-repo watchers

`MultiRepoWatcher` (`src/trelix/indexing/multi_watcher.py`) drives a single `watchfiles.awatch()` call over all repos registered in `RepoRegistry`. A SHA-256 hash guard prevents re-indexing unchanged files; deleted files are removed from both SQLite and the vector store.

```bash
# Install the watch extra
pip install -e ".[watch]"

# Watch all registered repos
trelix watch-all
```

To integrate programmatically:

```python
from trelix.indexing.multi_watcher import MultiRepoWatcher

watcher = MultiRepoWatcher(registry=repo_registry, indexer=indexer)
await watcher.run()   # streams per-repo stats; graceful on KeyboardInterrupt
```

### GitHub PR review integration

`GitHubPRClient` (`src/trelix/review/github.py`) fetches PR diffs via the GitHub REST API and posts batched review comments back. Requires the `GITHUB_TOKEN` environment variable.

```bash
# Review a PR diff locally
trelix review --pr owner/repo#42

# Review and post findings as a GitHub review
trelix review --pr owner/repo#42 --post-comments
```

All seven GitHub file status values (`added`, `modified`, `removed`, `renamed`, `copied`, `changed`, `unchanged`) are handled. PRs with more than 3,000 files emit a truncation warning.

To call `DiffReviewer` directly with a raw unified diff string (skipping file I/O):

```python
from trelix.review.diff import DiffReviewer

reviewer = DiffReviewer(llm_client=client)
findings = await reviewer.review(diff_text=raw_unified_diff)
```

`parse_pr_ref("owner/repo#42")` is the canonical helper for parsing `--pr` argument values.

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

trelix follows [Semantic Versioning 2.0.0](https://semver.org/). Read the current
version from the tree rather than from this sentence:

```bash
python -c "import trelix; print(trelix.__version__)"
```

This section deliberately no longer carries a copy of the number. It said "Current
version: **2.12.0**" while v3.0.0, v3.0.1, v3.1.0, v3.1.1 and v3.1.2 shipped — five
releases of rot inside the very section that tells you to keep versions in sync. The
doc-stamp grep in the release checklist below would now catch it, but a prose stamp no
release step has to touch is better deleted than monitored.

### Stable public API (guaranteed not to change without a major version bump)

- **CLI commands and flags**: `trelix index`, `trelix search`, `trelix ask`, `trelix query`, `trelix stats`, `trelix watch`, `trelix update-index`, `trelix migrate-vectors` and all documented flags
- **Python API**: `IndexConfig`, `EmbedderConfig`, `LLMConfig`, `Indexer`, `Retriever`, `TrelixChatClient`, `ChatMessage`, `ChatResponse`, `ToolCallResponse`, `build_chat_client`, `BaseEmbedder`, `make_embedder`
- **Sub-package interfaces**: `TrelixRetriever` (trelix-langchain), `TrelixIndexRetriever` (trelix-llama-index), MCP tool signatures (trelix-mcp) — note: `search_code` return type changed to `{results, next_cursor, total_available}` envelope in v2.4.0 (see Breaking Changes in CHANGELOG)
- **Environment variable names**: all `TRELIX_*` env vars documented in `.env.example`

### What counts as a breaking change

- Removing or renaming a public class, method, or CLI flag
- Changing a method signature in an incompatible way
- Changing the SQLite schema in a way that requires re-indexing
- Removing a previously supported Python version

**CLI command renames** (e.g. `trelix graph` → `trelix call-graph` in v2.0.0) are breaking changes and must be documented under a `### Breaking Changes` heading in `CHANGELOG.md` for the relevant release, alongside a migration note showing the old and new invocation.

### Deprecation policy

- Deprecated features are marked with `DeprecationWarning` and noted in the CHANGELOG
- The grace period is **minimum 2 minor versions and minimum 3 months, whichever lands
  later**, with removal only on a major bump. [docs/BACKWARDS_COMPATIBILITY.md](docs/BACKWARDS_COMPATIBILITY.md#deprecation-policy)
  is authoritative — go there for the reasoning and the current deprecation table
- The CLI will print a deprecation notice on first use of deprecated flags

This file previously said "at least one minor version", contradicting the policy doc's
"2 minor versions". The stricter number won: trelix shipped eight minor releases in the
30 days from v2.4.0 to v2.12.0, so a one-minor grace period can be over in days.

### Python version support

- Supported: Python 3.11, 3.12, 3.13
- Dropped versions are announced one minor release in advance

### Release checklist — the twelve version stamps

**Eleven files carry the release version, in twelve stamps** — `server.json` carries it
twice. All twelve are gated. Bump them together. Missing one ships a package whose `--version` disagrees with
its metadata, or a Helm chart that advertises one version and deploys another; neither is
hypothetical. `helm/trelix/values.yaml`'s `image.tag` sat on `2.12.0` while `Chart.yaml`
advertised `appVersion: 3.1.2`, and both adapters sat at `2.4.0` while core reached
`3.1.2` — which `skip-existing: true` turned into a silent 2-of-4 publish on every tag.

| # | Site | Key | Gated by `verify-version` |
|---|------|-----|---------------------------|
| 1 | `pyproject.toml` | `[project] version` | yes |
| 2 | `src/trelix/__init__.py` | `__version__` | yes |
| 3 | `packages/trelix-mcp/pyproject.toml` | `[project] version` | yes |
| 4 | `packages/trelix-mcp/src/trelix_mcp/__init__.py` | `__version__` | yes |
| 5 | `helm/trelix/Chart.yaml` | `appVersion:` | yes |
| 6 | `helm/trelix/values.yaml` | `image.tag` | yes |
| 7 | `packages/trelix-mcp/server.json` | `version` **and** `packages[0].version` | yes (both) |
| 8 | `packages/trelix-langchain/pyproject.toml` | `[project] version` | yes |
| 9 | `packages/trelix-langchain/src/trelix_langchain/__init__.py` | `__version__` | yes |
| 10 | `packages/trelix-llama-index/pyproject.toml` | `[project] version` | yes |
| 11 | `packages/trelix-llama-index/src/trelix_llama_index/__init__.py` | `__version__` | yes |

`.github/workflows/release.yml`'s `verify-version` job now fails the release if a `v*`
tag disagrees with any of those stamps, which is what turns a missed bump from a silent
mis-publish into a red build. It runs twelve `check` calls — one per stamp above, with no
exceptions. Site 4 used to be one: it was documented as "verify it by hand until that check
is added", which is the same silent-mis-publish risk as the adapters had, so it is now
checked like the rest.

Sites 4 and 8–11 are newly gated.
[docs/BACKWARDS_COMPATIBILITY.md](docs/BACKWARDS_COMPATIBILITY.md#integration-package-policy)
has always put all three integration packages on the core version; both adapters sat at
`2.4.0` anyway, across the seventeen releases that shipped after it, and nothing in CI
noticed. Read the jump as a
re-alignment to that line rather than as adapter change:
`git diff v2.4.0..HEAD -- packages/trelix-*/src` is 13 insertions and 3 deletions, all
type annotations, so `3.1.2` adds no feature and breaks nothing the adapters exposed at
`2.4.0`.

Two more files hold the number without being sites of their own:
`packages/trelix-langchain/tests/test_retriever.py` and
`packages/trelix-llama-index/tests/test_retriever.py` each assert their own package's
`__version__`. Bump those literals too — but they are assertions *on* sites 9 and 11, not
independent stamps, so they stay out of the table and out of the count. Skipping them
turns a suite red rather than shipping anything wrong, and `release.yml`'s `test` job now
runs both adapter suites, so that red arrives on the tag and not only in `ci.yml` (which
never fires on one).

Two things that are **not** version sites, and must not be bumped with them:

- `helm/trelix/Chart.yaml` has both `version:` (the *chart's* own version, currently
  `0.2.0` and independent of trelix) and `appVersion:` (which tracks trelix). Only
  `appVersion` moves.
- Both adapters' `dependencies = ["trelix>=3.0.0", ...]`. Lockstep governs the version
  *stamp* — identity — not the dependency *floor*, which is a compatibility contract that
  moves only when an import demands it. Each `pyproject.toml:35` records why it reads
  `3.0.0`: "the lowest published core verified to expose every name used here." An
  adapter stamped `3.1.2` that declares `trelix>=3.0.0` is saying something true. Raising
  the floor to match a release is the mistake CHANGELOG v2.7.1 already reverted
  ("Unjustified dependency-floor bumps reverted") on these same two packages, where it had
  been raised on an unverified assumption about API usage.

#### Verify by printing what each site says — never by grepping for a version string

Both directions of that grep fail silently, which is how `image.tag` stayed on `2.12.0`
across five releases:

- Grepping for the **new** version lists only the sites already bumped. A stale site
  produces no line, and a missing line is not a signal you will notice. Reproduced on a
  tree with `values.yaml` and `server.json` left at `2.12.0`: a grep for the new version
  over the five previously-listed files printed five clean hits and exited `0`.
- Grepping for the **previous** version cannot see a site that skipped a release. The
  same tree, grepped for the previous version, returned *zero* hits — the stale sites
  read `2.12.0`, not the previous version. A site stranded on 2.x is structurally
  invisible to this check — which is why `verify-version`, not any grep in this
  checklist, is the actual gate. Every site that had really drifted was of that class:
  `image.tag`, both of `server.json`'s fields, and both adapters at `2.4.0`.

So print the value each site actually holds and collapse them:

```bash
python - <<'PY'
import json, re, tomllib
from pathlib import Path
def toml(p): return tomllib.load(open(p, "rb"))["project"]["version"]
def rx(p, pat): return re.search(pat, Path(p).read_text(), re.M).group(1)
sj = json.load(open("packages/trelix-mcp/server.json"))
for label, got in [
    ("pyproject.toml", toml("pyproject.toml")),
    ("src/trelix/__init__.py", rx("src/trelix/__init__.py", r'__version__ = "([^"]+)"')),
    ("trelix-mcp/pyproject.toml", toml("packages/trelix-mcp/pyproject.toml")),
    ("trelix_mcp/__init__.py", rx("packages/trelix-mcp/src/trelix_mcp/__init__.py", r'__version__ = "([^"]+)"')),
    ("Chart.yaml appVersion", rx("helm/trelix/Chart.yaml", r'^appVersion:\s*"?([^"\s]+)')),
    ("values.yaml image.tag", rx("helm/trelix/values.yaml", r'^\s+tag:\s*"?([^"\s]+)')),
    ("server.json version", sj["version"]),
    ("server.json packages[0]", sj["packages"][0]["version"]),
    ("trelix-langchain/pyproject.toml", toml("packages/trelix-langchain/pyproject.toml")),
    ("trelix_langchain/__init__.py", rx("packages/trelix-langchain/src/trelix_langchain/__init__.py", r'__version__ = "([^"]+)"')),
    ("trelix-llama-index/pyproject.toml", toml("packages/trelix-llama-index/pyproject.toml")),
    ("trelix_llama_index/__init__.py", rx("packages/trelix-llama-index/src/trelix_llama_index/__init__.py", r'__version__ = "([^"]+)"')),
]:
    print(f"{got:<10} {label}")
PY
```

Twelve lines out, all the same version. Pipe it through `| awk '{print $1}' | sort -u`
and you should get exactly one line — more than one means a stale site, named rather
than merely absent. This mirrors `verify-version`'s own extraction site for site, so a
clean local run predicts a green tag. `tests/unit/test_release_version_gate.py` asserts the
same agreement in CI, which is the part that catches drift *before* a tag exists at all —
the gate itself can only fail once someone has cut one.

#### Doc version stamps

Doc stamps rot every release, and they rot to *arbitrary* old versions — this file's own
"Current version" line reached `2.12.0`-vs-`3.1.2`, five releases behind — so grepping
for the previous version misses exactly the worst cases. Grep for the *assertions* that
must name the current version, whatever number they currently hold:

```bash
grep -rnE '[Cc]urrent version|trelix(-mcp|-langchain|-llama-index)?==|^\*\*Version|image: .*trelix:' \
    docs/*.md *.md packages/*/README.md \
  | grep -v '^CHANGELOG.md' \
  | grep -viE "new in|fixed in|added in|since v|what's new|removed in|deprecated in"
```

That returns 15 readable lines — several of which are this section quoting its own
pattern — rather than the 251 that "every semver-shaped token that isn't the current
version" produces over the same files. It catches the stamps a previous-version grep
cannot: `docs/CLI_REFERENCE.md`'s `**Version:**` header, the adapter `==` install pins in
`docs/FAQ.md`'s pin-your-requirements answer, a Helm `image: …trelix:` tag. The pattern
finds those pins by package name, not by number, so it keeps working across bumps — naming
the number here would only rot this sentence. `docs/LANGCHAIN_LLAMAINDEX_GUIDE.md` no
longer has any: its install lines are deliberately unpinned, since a version hardcoded into
a doc's own install command is what rotted them last time.
`CHANGELOG.md` is excluded on purpose — it is an append-only historical record, never a
stamp to bump.

`packages/*/README.md` is in scope because each distribution's `pyproject.toml` sets
`readme = "README.md"`, making those files the **PyPI long description** — the project page
readers copy commands from. The glob was `docs/*.md *.md` until this release, which never
descended into `packages/`, and that blind spot is exactly where the worst rot was:
`packages/trelix-mcp/README.md` hard-pinned `trelix-mcp==2.12.0` in seven places, including
the primary command under its own `## Install` heading, while the package shipped `3.1.2`.
A grep with a blind spot reads as coverage. `tests/unit/test_readme_install_commands.py`
now asserts the same properties in CI, so this does not depend on anyone remembering to run
the grep.

Read every hit before editing. A blind `sed` over `docs/` will silently rewrite
"New in v3.0.0" and the shipped-version table in `ROADMAP.md`, turning accurate
history into a false claim.

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
