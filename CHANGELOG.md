# Changelog

All notable changes to trelix are documented here.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) — [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- **OpenTelemetry tracing for the retrieval pipeline** — opt-in via
  `pip install trelix[otel]` + `TRELIX_OTEL_ENABLED=true` (off by default,
  zero import cost and zero behavior change when disabled). Emits one
  `gen_ai.*`-conventions retrieval span per leg (vector, BM25, grep, sparse,
  sub-chunk, file-summary) via `opentelemetry-util-genai`'s
  `TelemetryHandler.retrieval()`, plus `trelix.*`-namespaced pipeline-stage
  spans (planner, fusion, expansion, rerank, pagerank boost, assembly).
  Correctly nests leg spans under the query's root span across the
  `ThreadPoolExecutor` boundary used for parallel sub-query execution (OTel's
  context is contextvars-based and does not cross thread pools on its own).
  New optional `OTEL_EXPORTER_OTLP_ENDPOINT` exports to any OTLP collector.
  See `docs/OBSERVABILITY.md` for the full span reference and a stability
  caveat (the `gen_ai.*` conventions are officially adopted but still
  "Development," not yet "Stable," upstream).

### Security
- **GitHub App: payload size limit and subprocess timeout** — the webhook
  route now caps request bodies at 25MB (GitHub's own documented webhook
  payload cap), rejecting oversized bodies with `413` during parsing
  rather than buffering an arbitrarily large request into memory; this
  matters because signature verification happens *after* body parsing, so
  the size limit is the only defense against a sender who doesn't know
  the webhook secret sending a deliberately huge payload.
  `runReviewCli`'s `trelix review` shell-out now passes a 5-minute
  `timeout`, so a hung/slow review (LLM synthesis latency, a huge diff, a
  stuck index) no longer ties up the process indefinitely — Node kills
  the child process (`SIGTERM`) and the call rejects. New tests exercise
  both with real subprocesses/payloads rather than mocks: a genuinely
  slow shell shim proves the timeout actually kills the process, and a
  real 26MB request body proves the size limit actually rejects.
- **GitHub App: webhook signature verification** — `infra/github-app/src/webhook.ts`
  now verifies `X-Hub-Signature-256` (HMAC-SHA256 over the raw request
  body, keyed by the webhook secret) via `@octokit/webhooks-methods`'s
  `verify()`, which compares using `crypto.timingSafeEqual` rather than a
  naive string compare (avoids leaking timing information about how many
  leading bytes matched). Requests with a missing, wrong-secret, or
  tampered-after-signing body are rejected with `401` before the route
  handler — and therefore `runReview`/the trelix CLI shell-out — ever sees
  the payload. New tests cover both the accept-valid and reject-tampered
  paths explicitly (the common real bug is only testing the happy path):
  no-header, wrong-secret, and tampered-body all assert `401` +
  `runReview` never called; a correctly-signed control case asserts `202`.

### Added
- **GitHub App: GA-readiness docs polish** — `infra/github-app/README.md`
  finalized (production deployment notes: HTTPS requirement, secret-manager
  guidance, runtime prerequisites) and its status upgraded from
  "skeleton"/"auth wired" to "installable and hardened" now that items
  6a-6c are complete. `docs/ROADMAP.md`'s "GitHub App GA" line explicitly
  states this App is installable and hardened, **not** Marketplace-listed
  — Marketplace paid-app listing has its own separate business/adoption
  requirements out of scope for this engineering work.
- **GitHub App: installation-token minting + Check-annotation posting**
  (`infra/github-app/src/auth.ts`, `src/review-runner.ts`) — completes the
  auth/posting work stubbed in item 6a. `getInstallationToken` uses
  `@octokit/auth-app` (App-ID+private-key JWT signing -> installation-token
  exchange), with one `AuthInterface` reused per `AppConfig` (a `WeakMap`)
  so the library's own expiry-aware cache actually has a chance to hit
  across calls — verified with mocked-transport tests proving a second
  call for the same config+installation makes zero additional HTTP
  requests, while distinct installations/configs never share a cached
  token. `runReview` now mints a token, fetches the PR's head SHA via
  `octokit.rest.pulls.get`, runs `trelix review --pr ... --json`, and
  posts a completed Check run with inline annotations via
  `octokit.rest.checks.create` (same conclusion logic as the existing
  `trelix-review.yml` workflow: any `failure`-level annotation ->
  `failure`, else `success`).
- **GitHub App skeleton** (`infra/github-app/`, `@trelix/github-app`) — the
  start of a standalone, webhook-driven GitHub App for zero-setup PR
  review (install the App, no workflow YAML needed in the installing
  repo), per the explicit architecture decision to build a standalone
  webhook-to-direct-execution service rather than a thin bridge to the
  existing Actions workflow. Ships `manifest.yml` (same
  `pull_requests`/`checks`/`contents` permissions and `pull_request`
  event the existing `trelix-review.yml` workflow already uses), an
  Express server with `/health` and `/webhooks/github`, webhook routing
  for `pull_request` `opened`/`synchronize`/`reopened` (mirroring the
  existing workflow's trigger), and a review-runner that shells out to
  `trelix review --pr ... --json` and maps findings to GitHub Check
  annotations (`toAnnotations` — a TypeScript port of the mapping logic
  fixed in the `trelix-review.yml` workflow). **Not yet wired for
  production use**: signature verification, installation-token minting,
  and Check-annotation posting are explicitly stubbed/unimplemented —
  land in item 6b. `infra/github-app/README.md` rewritten to cover both
  integration paths (the existing Actions workflow and this new App) so
  its previous "no App registration required" framing doesn't read as
  false now that a real App skeleton exists. New
  `.github/workflows/github-app-ci.yml` runs
  `npm ci && npm run typecheck && npm run build && npm test`, gated on
  `infra/github-app/**`.
- **VS Code extension: live-narrowing search + snippet preview** —
  `trelix.search` is now a debounced (250ms) search-as-you-type
  `QuickPick` instead of a one-shot `showInputBox` → static-list flow.
  Highlighting a result (arrow keys, not just accepting) shows a real
  snippet preview via `showTextDocument({preview: true})` against a new
  virtual-document `TextDocumentContentProvider`
  (`trelix-preview:` scheme, `src/preview.ts`) — this gets genuine VS Code
  syntax highlighting for free, since the virtual URI keeps the real
  file's extension. A `"Load more results…"` pseudo-item appears whenever
  `search_code`'s `next_cursor` is non-null (using the pagination fields
  PR #81/item 5a fixed), fetching and appending the next page without
  losing the current results or query. The debounce/cursor-pagination/
  stale-response-rejection state machine lives in a new, Extension-Host-
  independent `SearchController` class (`src/search-controller.ts`) —
  `search-controller.test.ts` uses an injectable fake-timer harness to
  simulate rapid keystrokes and prove `search()` fires exactly once per
  debounce window (not once per keystroke), that a stale in-flight
  response is discarded once a newer query has superseded it, and that
  `loadMore()` correctly appends via `next_cursor` and no-ops when there
  isn't one.
- **`docs/ROADMAP.md`**: logged the original Phase 3 plan's `@trelix` chat
  participant + hover providers (never actually delivered — only the 2
  QuickPick/Webview commands shipped) as an explicit v3.1.0 candidate,
  rather than silently dropping it again.

### Security
- **VS Code extension: XSS in the `trelix.ask` Webview** — `panel.webview.html`
  interpolated the raw, unescaped LLM answer string directly, with the
  Webview's `options` an empty `{}` (no CSP, no `enableScripts: false`, no
  `localResourceRoots` restriction at all). A crafted or adversarial answer
  could execute arbitrary script in the Webview's context. Now HTML-escapes
  the answer before interpolation and sets `enableScripts: false` plus an
  explicit `default-src 'none'` CSP meta tag.

### Fixed
- **VS Code extension: `search_code` results were silently mis-parsed** —
  `mcp-client.ts`'s `search()` read `symbol_name`/`file_path` off each
  result, but the real MCP `search_code` tool's response keys are
  `symbol`/`file` (confirmed against `packages/trelix-mcp/src/trelix_mcp/
  server.py`) — those two fields were always empty strings, and clicking a
  search result opened a broken/empty file URI. Also fixed: `next_cursor`/
  `total_available` were parsed off the response but discarded entirely
  (`search()` returned only `parsed.results`), and `kind`/`lines`/`source`/
  `language` were dropped from the parsed shape though the server already
  returns them. `search()` now returns the full `{results, nextCursor,
  totalAvailable}` shape with every field; `extension.ts` uses the newly
  available `lines` field ("start-end", 1-indexed) to jump to and highlight
  the matched symbol's line range on open, instead of just opening the file
  with no selection.
- **`trelix review --pr ... --json`'s stdout was never valid JSON** —
  `console.print(...)` status/progress messages (e.g. "Fetching PR diff
  from GitHub...") ran unconditionally to stdout even in `--json` mode,
  and `"No issues found."`/`"No textual changes..."` styled messages ran
  *instead of* an empty `[]` when there were zero comments. Combined with
  `.github/workflows/trelix-review.yml`'s `> file 2>&1` redirect, the
  review-posting Check's `JSON.parse()` has always thrown and been
  silently swallowed by a `try/catch` — meaning **the "trelix Code
  Review" Check has never posted a single real annotation** since this
  workflow shipped. All `--pr --json` status/progress messages now go to
  `err_console` (stderr); the workflow now redirects only stdout, keeping
  stderr in a separate log for debugging.
- **The same workflow's annotation-posting logic never matched trelix's
  real output shape even when parsing succeeded** — it read
  `data.findings || data.reviews || []` against `trelix review --json`'s
  real bare-array output (never matches, so `findings` was always `[]`
  regardless), and compared `f.severity === 'error'`/`'warning'`
  (lowercase) against the real values `"ERROR"`/`"WARN"`/`"INFO"`
  (uppercase — `'WARN' !== 'warning'` either way). Every annotation would
  have posted as `notice` severity even if the JSON had parsed. Now reads
  the real `{file, lines, severity, comment}` shape directly and maps
  `ERROR`→`failure`, `WARN`→`warning`, `INFO`→`notice`.
- New `tests/unit/test_review_pr_json.py` (4 tests) — regression-tests
  `--json` stdout purity for the has-comments, zero-comments, and
  no-textual-changes paths, plus confirms non-`--json` mode still prints
  status messages to stdout (the fix is `--json`-gated, not a blanket
  behavior change). Verified these tests actually fail against the
  pre-fix code (3/4 failed with the exact `JSONDecodeError` this bug
  produces) before confirming they pass against the fix.

### Changed
- **VS Code extension build/test infrastructure** — added `esbuild`
  (bundles `dist/extension.js`, `external: ["vscode"]`) instead of plain
  `tsc` emit, so the packaged `.vsix` no longer risks shipping unbundled
  `node_modules` (the extension's only runtime dependency,
  `@modelcontextprotocol/sdk`). `tsc --noEmit` remains a separate
  `typecheck` script since esbuild doesn't type-check. Added a
  `.vscodeignore` (previously absent) and a `@vscode/test-electron`+Mocha
  test harness (`src/test/runTest.ts`, `src/test/suite/`) — new
  `extension.test.ts` verifies activation and command registration;
  `mcp-client.test.ts` verifies the `search()`/`ask()` parsing fixes above
  against a mocked MCP transport. New
  `.github/workflows/vscode-extension-ci.yml` runs
  `npm ci && npm run typecheck && npm run build && xvfb-run -a npm test`,
  gated on `workspace-vscode/**` changes. Version bumped `0.1.0` → `0.2.0`
  (unchanged since the v2.7.0 scaffold).
- **`docs/integrations/vscode-plugin.md` full rewrite** — the previous
  version described a PyInstaller-binary-bundling architecture that was
  never actually built, and never once mentioned the real MCP-stdio-client
  architecture the extension actually ships with. Rewritten to describe
  the real `dist/extension.js` (esbuild bundle) → `trelix-mcp` (stdio
  child process) → trelix core data flow, the real `search_code` response
  shape, the security notes above, and the real build/test/package
  commands.

### Added
- **Helm chart** (`helm/trelix/`) for deploying `trelix serve` to Kubernetes —
  `Deployment`/`Service`/`PVC`/`Secret`/`Ingress` templates, `values.yaml`
  covering the full `StoreConfig` surface (`store.backend`: sqlite/qdrant/
  lance, HNSW tuning, BM25 read-pool size) plus embedder-provider
  credentials (OpenAI/Voyage/Cohere, either plaintext `apiKey` for dev or
  `existingSecretName`/`existingSecretKey` for shared clusters). Models
  `trelix serve`'s actual behavior directly: since `create_app()` takes zero
  arguments and every route re-derives its config from the request's own
  `repo` param, one Deployment is already multi-repo-capable — the chart's
  PVC (mounted at `/data` by default) is a *shared* data directory across
  every repo you index/serve, documented loudly in `NOTES.txt`/`README.md`
  since it's non-obvious. `ingress.enabled` defaults to `false`: `trelix
  serve` has zero auth middleware, so `NOTES.txt` warns explicitly about
  exposing `/index`/`/ask`/`/search` before enabling a public Ingress.
  Qdrant is treated strictly as an external, user-managed dependency — this
  chart only points `QDRANT_URL`/`QDRANT_API_KEY` at one, never deploys or
  operates Qdrant itself (its own chart states support is
  community-limited; self-hosted lacks zero-downtime upgrades and
  backup/DR). New `.github/workflows/helm-lint.yml` runs `helm lint` +
  `helm template` across all three `store.backend` values plus an
  ingress-enabled render, on every push/PR touching `helm/**`.
- **Official Docker image** — a multi-stage `Dockerfile` (root) publishes
  `ghcr.io/sairam0424/trelix` for `linux/amd64`+`linux/arm64` on every
  release tag, in two variants sharing one build (`EXTRAS` build arg):
  `:X.Y.Z` (slim, API-embedder-only — OpenAI/Voyage/Cohere/Azure) and
  `:X.Y.Z-local` (bundles `sentence-transformers`/`torch` for the
  local/offline embedder and cross-encoder reranker). Runs as a non-root
  `trelix` user, `ENTRYPOINT ["trelix"]` with `CMD ["serve", "/repo",
  "--host", "0.0.0.0", "--port", "8765"]` (overrides the CLI's
  `127.0.0.1` default, which isn't reachable from outside a container's
  network namespace), and a `HEALTHCHECK` hitting `/health`. New
  `docker-compose.yml` at the repo root is a runnable version of
  `docs/INSTALLATION_GUIDE.md`'s Docker Compose snippet. New
  `.github/workflows/docker-publish.yml` builds/pushes both variants on
  `v*` tags; CI gained a `docker-build` job that builds the slim image and
  runs `--help` against it on every push/PR, mirroring the existing
  per-OS binary `--help` smoke tests in `release.yml`.
- New Makefile targets: `docker-build`, `docker-build-local`, `docker-run`.

### Fixed
- **`TRELIX_EMBEDDER` was a silent no-op env var** — `docs/
  INSTALLATION_GUIDE.md` and `docker-compose.yml` both referenced
  `TRELIX_EMBEDDER`, but `EmbedderConfig`'s real env var is
  `TRELIX_EMBEDDER_PROVIDER` (confirmed empirically: setting
  `TRELIX_EMBEDDER=openai` in a clean environment left `provider` at its
  default `"local"`). On the slim Docker image this silently falls back to
  a provider that isn't installed and crashes, rather than erroring at the
  variable name. Also fixed a `--embedder` CLI flag reference in the same
  section — the real flag is `--provider`. Found while writing this
  chart's `values.yaml` example and wanting to confirm the var name against
  source before using it.
- **`docs/INSTALLATION_GUIDE.md`'s Docker Compose/serve examples used the
  wrong port** (8080) and a nonexistent `serve --repo` flag (`repo_path`
  is positional) — same class of bug already fixed for the `docker run`
  examples in PR #77, now fixed here too since this PR touches the same
  section.

## [2.8.1] — 2026-07-20

### Security
- **MCP federation `config_path` path confinement** — `federation_list_repos`/
  `federation_add_repo`/`federation_remove_repo`/`federation_search_all`
  previously passed a caller-supplied `config_path` straight into
  `RepoRegistry.load()`/`.save()` with no validation, letting an MCP client
  (including a prompt-injected agent) point registry I/O at an arbitrary
  path. Now confined to `~/.config/trelix/` or `<mcp-server-cwd>/.trelix/`
  via `Path.is_relative_to()` (not a naive string-prefix check, which would
  incorrectly also match a sibling directory like `~/.config/trelixevil/`).
  Found in the pre-push audit of v2.8.0 (issue #69).

### Added
- **Federation repo-count and fan-out caps** — `RepoRegistry.add()` gained an
  optional `max_repos` parameter (CLI callers remain unbounded by default;
  MCP's `federation_add_repo` now passes `TRELIX_FEDERATION_MAX_REPOS`,
  default 50). `FederatedRetriever` gained a `max_repos` constructor param
  capping how many registered repos are actually queried per call;
  `federation_search_all`'s response gained a `repos_skipped` field.
  Prevents a runaway/adversarial `federation_add_repo` loop from making
  every subsequent search scale linearly with an unbounded repo count.

### Fixed
- **`federation_search_all` pagination wasn't a stable slice** — previously
  requested `fed.retrieve(query, k=max(k+cursor, k))`, so the per-repo
  candidate pool feeding RRF fusion widened as `cursor` grew, meaning page 2
  could be fused from a differently-shaped pool than page 1 (items could
  shift rank, get deduped differently, or disappear between pages). Now
  fetches a fixed, cursor-independent width once and slices pages from the
  final fused list — mirrors `search_code`'s existing single-fetch-then-slice
  pattern.

### Changed
- All 4 federation MCP tools now consistently return an `"error": str|None`
  key on every response path (previously only present on failure paths for
  `federation_add_repo`), matching the convention already used by
  `ask_agent`/`agent_list_sessions`/`agent_clear_session`.

## [2.8.0] — 2026-07-20

### Added
- **Multi-repo support in MCP** — 4 new MCP tools (`federation_list_repos`,
  `federation_add_repo`, `federation_remove_repo`, `federation_search_all`)
  expose the existing `RepoRegistry`/`FederatedRetriever` CLI infrastructure
  (`trelix federation add/list`, `trelix search-all`) to MCP clients (Claude
  Desktop, Cursor, any IDE). Also added the missing `trelix federation remove`
  CLI command (the registry method existed but had no CLI entry point).
- **Persistent agent (ReAct loop) memory** — the agentic loop
  (`trelix ask --agentic`, `TRELIX_RETRIEVAL_AGENTIC=true`) now persists turn
  history to new `agent_sessions`/`agent_turns` tables in the per-repo
  `.trelix/index.db`, keyed by a client-supplied or auto-generated UUID4
  `session_id`. `AgentLoop.run()` now returns `(answer, session_id)` — pass
  the session_id back on a follow-up call to resume with full prior context.
  New CLI: `trelix ask --session <id>`, `trelix agent sessions list/show/clear`.
  New MCP tools: `ask_agent`, `agent_list_sessions`, `agent_clear_session`.
  Sessions auto-evict after `TRELIX_RETRIEVAL_AGENT_SESSION_MAX_AGE_SECONDS`
  of inactivity (default 7 days; `0` disables eviction).

### Fixed
- **Federated search lost repo provenance** — `FederatedRetriever` used to tag
  each result's `source` with `"{alias}:{leg}"` so callers could tell which
  repo a result came from, but this was silently dropped in a prior refactor.
  `trelix search-all`'s "Repo" column and `--json` output had been blank ever
  since, with no test catching it. Restored the tagging and added a
  regression test.
- **`RepoEntry.weight` was never applied** — settable via
  `trelix federation add --weight`, stored, and documented, but the fan-out
  fusion path never forwarded it into RRF, so per-repo weighting silently did
  nothing. `reciprocal_rank_fusion()` gained a new `list_weights` parameter
  (orthogonal to the existing per-language `weights` parameter; `None` is
  backward-compatible) and `FederatedRetriever` now passes each repo's weight
  through.
- **`agent_turns.turn_index` could silently collide on session resume** — found
  in pre-push audit. `AgentLoop.run()` used to compute the resume anchor from
  `len(prior_rows)` (a row-count snapshot), which drifts from reality after
  any persistence gap (a dropped turn) or a concurrent resume of the same
  `session_id`, silently producing duplicate `turn_index` rows with no error.
  `Database.insert_agent_turn()` now assigns `turn_index` atomically via
  `MAX(turn_index)+1` under the same lock as the insert, and `agent_turns`
  gained a `UNIQUE(session_id, turn_index)` index as defense-in-depth — any
  residual race now raises `IntegrityError` (caught and logged) instead of
  silently duplicating a row.

## [2.7.3] — 2026-07-13

### Changed
- **README.md end-to-end audit and rewrite** — fixed 15+ factual bugs (wrong
  env var names, fabricated pip extras, a broken Homebrew tap, a crash-causing
  `TRELIX_RETRIEVAL_RERANK_PROVIDER` value, wrong REST method/table names),
  rewrote the "How it works" diagram to show all 7 retrieval legs (was 3) plus
  the agentic/FLARE alternate synthesis modes, and consolidated duplicated
  content (3x REST API sections, Installation/Knowledge-Graph/Embedding-Providers
  duplicating `docs/`) into short pointers. 867 → 634 lines.
- **"What's New" and "Troubleshooting" moved out of README** — backfilled
  CHANGELOG.md's empty `[2.2.0]` entry with its 5 features (agentic ReAct loop,
  data-flow analysis, taint analysis, sparse+dense hybrid, multi-granularity
  indexing — previously undocumented anywhere else) and added README's 5
  Troubleshooting entries to `docs/TROUBLESHOOTING.md`'s existing sections,
  then trimmed both README sections to short pointers.

## [2.7.2] — 2026-07-12

### Added
- **Qdrant Cloud readiness** — `QdrantVectorStore` now accepts `prefer_grpc` and
  `timeout` options, wired through `StoreConfig.qdrant_prefer_grpc`
  (`QDRANT_PREFER_GRPC`, default `false`) and `StoreConfig.qdrant_timeout`
  (`QDRANT_TIMEOUT`, default `10.0`). Enables gRPC transport (port 6334) and
  longer request timeouts for Qdrant Cloud's higher network latency.
- **Incremental per-symbol embedding on partial re-index** — new
  `symbols.content_hash` column (`sha256(signature + body)`, backfilled via an
  `ALTER TABLE ... ADD COLUMN` migration guard). `Indexer._insert_one` now diffs
  each parsed symbol's `(qualified_name, content_hash)` against the stored row;
  unchanged symbols skip delete/re-chunk/re-embed entirely and keep their
  existing chunk rows and vectors. Only changed or new symbols flow through the
  delete → re-insert → chunk → embed path.
- **Opt-in parallel BM25 read pool** — new `ReadOnlyConnectionPool`
  (`src/trelix/store/read_pool.py`) opens N read-only SQLite connections
  (`mode=ro`, `PRAGMA query_only = ON`) for concurrent FTS5 reads.
  `TRELIX_STORE_BM25_READ_POOL_SIZE` (default `0`, disabled) — when set > 0,
  `Retriever.__init__` calls `Database.enable_bm25_read_pool()` automatically.
- **Linux ARM64 binary releases** — `build-binaries.yml` and `release.yml`
  matrices add `ubuntu-24.04-arm` (artifact `trelix-linux-arm64`);
  `docs/INSTALLATION_GUIDE.md` gained a "Linux ARM64" install section.

### Fixed
- **`SparseEmbedder` TOCTOU race** — `_load()` checked `self._model is not None`
  before acquiring the lock, so two threads could both pass the check and
  double-load the model concurrently. Fixed with double-checked locking:
  `self._model is not None` is re-checked again inside `self._lock`.
- **MCP stdout notification write race** — concurrent `send_resource_notification()`
  calls from different threads could interleave partial JSON-RPC lines on
  stdout. Added a module-level `_stdout_lock` guarding the `sys.stdout.write()` +
  `flush()` pair.
- **`SubscriptionRegistry` unbounded growth** — subscriptions were never capped
  or expired, so a misbehaving client could grow the registry indefinitely.
  Added `max_subscribers` (`TRELIX_MCP_MAX_SUBSCRIBERS`, default `1000`)
  enforced via a new `SubscriptionLimitExceeded` exception, and `ttl_seconds`
  (`TRELIX_MCP_SUBSCRIPTION_TTL_SECONDS`, default `3600`) swept by
  `_evict_expired_locked()` before every registry operation. The
  `subscribe_resource` tool now catches the limit error and returns a soft
  `{"subscribed": false, ...}` payload instead of raising.
- **Silent `parent_id`/`callee_id`/`type_edges` corruption on partial re-index** —
  `symbols.parent_id`, `calls.callee_id`, and `type_edges.to_symbol_id` are all
  `ON DELETE SET NULL`, so deleting a changed symbol's old row silently nulled
  these links on any other row (including unchanged ones) that pointed at it.
  Added `Database.get_children_with_stale_parent`/`repoint_parent_ids`,
  `get_calls_referencing_symbols`/`repoint_call_callee_ids`, and
  `get_type_edges_referencing_symbols`/`repoint_type_edge_targets`; the indexer
  snapshots stale links before the cascading delete and repoints them to the
  replacement symbol's new id afterward.
- **Incomplete BM25 concurrency lock** — `Database._conn`
  (`check_same_thread=False`) is not safe for concurrent statement execution
  from multiple threads despite that flag; the grep, sparse, and vector
  retrieval legs all hydrate through the same shared connection from sibling
  `ThreadPoolExecutor` threads. Added `self._conn_lock` and applied it to
  `bm25_search()`'s non-pool fallback, `get_symbol_with_file()`,
  `get_first_chunk_for_symbol()`, `get_chunk_with_context()`,
  `grep_search.py`'s `_name_search`/`_body_search`, and a new locked
  `Database.get_chunk_by_id()` helper for `sparse_search.py`'s raw chunk
  lookup. Verified via a 60-thread x 10-iteration x 3-leg stress test with
  zero errors.
- **`qdrant-client` 1.18 API migration** — `QdrantVectorStore` used the
  deprecated `search()` method; migrated to `query_points()`. Pinned
  `qdrant-client>=1.9.0,<2.0.0` in `pyproject.toml` to prevent an unguarded
  2.x upgrade from breaking the client again.

### Changed
- **Windows ARM64 binary intentionally not shipped** — `windows-11-arm` was
  briefly added to both binary-build matrices, then reverted:
  `tree-sitter-languages` and `sqlite-vec` publish no `win_arm64` wheel or
  sdist, so `pip install` fails before the build ever runs. Linux ARM64 ships;
  Windows ARM64 does not.

## [2.7.1] — 2026-07-10

### Fixed
- **Release pipeline asset collision** — `release.yml` referenced the macOS and Linux
  PyInstaller binaries by bare basename (both built as `dist/trelix`).
  `softprops/action-gh-release` uploads by basename, so the two identically-named
  binaries collided into a single GitHub Release asset. The published v2.7.0 release
  had only 2 binary assets instead of 3, and it was unknowable whether the surviving
  `trelix` asset was macOS or Linux. Each binary is now renamed to a unique filename
  (`trelix-macos-arm64` / `trelix-windows-x64.exe` / `trelix-linux-x64`) before upload.
- **No Linux binary in PR-time CI** — `build-binaries.yml` only built and verified
  macOS + Windows even though `release.yml` already builds Linux at tag time. Added
  the `ubuntu-latest` matrix entry and a Linux verify step.
- **Unjustified dependency-floor bumps reverted** — `trelix-mcp`, `trelix-langchain`,
  and `trelix-llama-index` had their `trelix>=X.Y.Z` floors raised to `>=2.7.0`/
  `>=2.4.0` in v2.7.0 based on an unverified assumption about API usage. Re-checked
  every import in all three packages — none use any Phase 1–3 v2.7.0 API. Reverted
  to `trelix>=0.4.0`.
- **`trelix-mcp` tests never ran in CI** — `ci.yml`'s test job never installed or
  executed `packages/trelix-mcp/tests/`. This let a real regression sit undetected:
  `test_four_tools_registered` asserted "exactly 6 tools" when the server has
  registered 8 since `subscribe_resource`/`unsubscribe_resource` shipped in v2.5.0.
  Fixed the test's expected set and wired `packages/trelix-mcp/tests/` into `ci.yml`.
- **Wrong env var name in docs** — `TRELIX_GRAPH_SEARCH_ENABLED` was incorrect in
  7 places across `docs/FAQ.md`, `docs/USER_GUIDE.md`, `CONTRIBUTING.md`. The real
  variable is `TRELIX_RETRIEVAL_GRAPH_SEARCH_ENABLED` (`graph_search_enabled` has
  no explicit alias override, so it inherits `RetrievalConfig`'s
  `env_prefix="TRELIX_RETRIEVAL_"`).
- **CHANGELOG footer link collision** — `[2.2.0]` was defined twice with conflicting
  URLs; markdown silently resolves to the last definition, making the first dead.
  `[2.3.0]`, `[1.1.0]`, `[0.7.1]`, `[0.7.0]`, `[0.6.0]` had no comparison link at all
  despite existing as dated release headers. Rebuilt the footer from scratch,
  cross-checked against `git tag -l`.

## [2.7.0] — 2026-07-09

### Added — Phase 1: Watch Bridge, DB Index, AdaptiveRouter Config Fix
- `FileWatcher._do_reindex` now fires `notify_file_changed()` after a successful
  re-index (not on hash-identical skips). MCP subscribers receive live
  `notifications/resources/updated` pushes when watched files change.
  Non-fatal when `trelix-mcp` is not installed.
- `idx_files_rel_path` index added to `files.rel_path` — eliminates full table
  scan on every `GraphUpdater.update_file()` call (`WHERE rel_path = ?`).
  `CREATE INDEX IF NOT EXISTS` — safe on existing databases.
- `AdaptiveRouter.__init__` now accepts `retrieval_config: RetrievalConfig | None = None`.
  When provided, it is used directly instead of constructing a new instance from env
  vars — fixes silent-ignore of programmatic config overrides.
- `Retriever` passes `config.retrieval` through `QueryPlanner → AdaptiveRouter`.

### Added — Phase 2: Cross-Repo Symbol Resolution, Semantic Diff Embeddings, Streaming Indexing
- `make_scip_symbol_id(package, version, qualified_name)` — stable SCIP-style
  cross-repo symbol ID using `||`-separated sha256[:16]. Unambiguous for scoped
  npm packages (`@scope/pkg`).
- `FederatedRetriever.record_exports(alias, repo_path)` — indexes all symbols from
  a trelix-indexed repo into an in-memory `federation_symbols` table.
- `FederatedRetriever.resolve_symbol(qualified_name)` — returns all repos that
  define a symbol. Supports exact match and suffix-LIKE (`%.verify`). Thread-safe
  via `threading.Lock` + `check_same_thread=False`.
- `DiffEmbedder` — CCRep-style before/after body pair embeddings for PR diff hunks
  (arXiv:2302.03924). `store_pr_diff()` caps at 500 hunks/PR; `search_similar_diffs()`
  finds historically similar changes by cosine similarity with NaN guard and
  dimension mismatch protection.
- `diff_chunks` SQLite table + `idx_diff_chunks_pr_ref` index added to schema.
- `TRELIX_INDEXER_STREAMING=true` — generator-based file processing pipeline.
  `_iter_files()` yields files lazily; `_index_streaming()` uses bounded
  `Queue(maxsize=64)` with `try/finally` producer sentinel guarantee.
  Default off — zero behavior change on default path.

### Added — Phase 3: VS Code Extension, GitHub App PR Review
- `workspace-vscode/` — VS Code extension scaffold (`trelix.search` and `trelix.ask`
  commands) using `TrelixMcpClient` over MCP stdio transport. Piggybacks on existing
  `trelix-mcp` package — no new Python backend.
- `.github/workflows/trelix-review.yml` — GitHub Actions workflow that runs
  `trelix review --pr N --json` on every pull request and posts findings as
  GitHub Check annotations with file+line references.
  Permissions: `checks: write`, `pull-requests: write`, `contents: read`.
  Index step has `continue-on-error: true` for CI environments without local models.
- `infra/github-app/README.md` — GitHub App integration setup guide.

## [2.6.0] — 2026-07-08

### Added — XTR Late-Interaction Reranker (Plan C, EXPERIMENTAL)
- `TRELIX_RETRIEVAL_RERANK_PROVIDER=xtr` — XTR reranker (NeurIPS 2023,
  arXiv:2304.01982). Scoring stage is 100–1000x cheaper than ColBERT/PLAID
  by reusing already-retrieved tokens instead of loading all document tokens.
- `TRELIX_RETRIEVAL_XTR_TOKENS=100` — candidate token count for XTR retrieval
  (range 10–1000).
- `trelix.retrieval.reranker_xtr` — pure-Python XTR scoring module
  (`xtr_score_documents`, `warn_experimental`).
- **EXPERIMENTAL:** XTR has not been benchmarked on code-specific retrieval
  (CoIR/CoREB evaluation pending). Emits `UserWarning` on first use. PLAID
  remains the production-validated late-interaction option.

### Added — GroUSE-Style Synthesis Quality Harness (Plan D)
- `trelix.eval.synthesis` — `SynthesisEvalHarness`, `evaluate_synthesis`,
  `score_hallucination`, `score_completeness`, `score_faithfulness`, `SynthesisResult`.
- `trelix eval-synthesis --golden <path>` — CLI command for synthesis quality evaluation.
- `eval/golden_synthesis_sample.jsonl` — sample golden file for getting started.
- Golden file format extends the existing eval harness with optional
  `expected_answer_fragments` and `expected_symbols` fields.
- Research basis: GroUSE (arXiv:2409.06595, COLING 2025) — 7 failure modes,
  144 unit tests. GPT-4 correlation is insufficient as a quality proxy.

### Added — Short-Query Lexical Fallback (Plan B)
- `TRELIX_RETRIEVAL_SHORT_QUERY_LEXICAL=true` — enables BM25+grep-only routing
  for queries with ≤ threshold meaningful tokens (default off).
- `TRELIX_RETRIEVAL_SHORT_QUERY_TOKENS` — sets the meaningful-token threshold
  (default 5, range 1–10).
- `is_short_query(query, threshold)` and `count_meaningful_tokens(query)` helpers
  in `trelix.retrieval.bm25`.
- `SubQuery.lexical_only: bool` — new field; when True, `_run_subquery_legs` skips
  vector ANN embedding entirely.
- Research basis: CoREB benchmark (arXiv:2605.04615) confirms all embedding models
  score 0.000–0.015 nDCG@10 on short keyword queries vs 0.45–0.58 on long queries.

### Added — Incremental Louvain Community Detection (Plan A)
- `detect_communities_incremental(cg, seed_nodes, prev_partition)` — DF Louvain
  frontier heuristic (arXiv:2404.19634). Reprocesses only the affected-vertex
  frontier instead of the full graph on file-change events.
- `compute_affected_frontier(G, seed_nodes, partition)` — computes the DF Louvain
  frontier: seed nodes + their neighbors + their community members.
- `GraphUpdater` now maintains `_prev_partition` across calls and uses incremental
  detection for subsequent updates. First run and large-frontier (>50% of nodes)
  fall back to full Louvain.
- `Database.get_symbol_ids_for_file(rel_path)` — returns symbol IDs for a file
  (used to seed the incremental frontier from a file-change event).

## [2.5.0] — 2026-07-06

### Overview
Phase A–C of the v2.5.0 backlog. Three independent subsystems shipped:
multi-query expansion wired into `_retrieve_standard`, DimensionGuard at
`FileWatcher.__init__`, and MCP resource subscriptions (capability declaration
+ subscription registry + file-change notification bridge). v3.0.0 deprecation
schedule documented and regression-tested.

### Added — Multi-Query Expansion Wiring (Phase A)
- `MultiQueryExpander` is now wired into `_retrieve_standard` via `ThreadPoolExecutor`
- Enable with `TRELIX_RETRIEVAL_MULTI_QUERY=true`, tune with `TRELIX_RETRIEVAL_MULTI_QUERY_COUNT=3`
- Variant queries run in parallel; results RRF-merged with k=60 before dedup
- `ExpandResult.llm_used` indicates whether LLM expansion ran or fell back to original

### Added — DimensionGuard at Watch Startup (Phase A)
- `FileWatcher.__init__` now calls `DimensionGuard.check()` at startup
- Raises `DimensionMismatchError` immediately if provider was changed since last index run
- Prevents silent embedding corruption from mismatched providers during watch

### Added — MCP Resource Subscriptions (Phase B)
- `trelix-mcp` now advertises `resources.subscribe=True` in server capabilities
- `SubscriptionRegistry` tracks subscription IDs per resource URI (thread-safe)
- `notify_file_changed()` fires `notifications/resources/updated` (URI-only, per MCP spec)
  over stdio for all active subscribers when watchfiles detects a change
- Wire protocol: `resources/subscribe` -> `notifications/resources/updated` -> `resources/read`

### Documentation
- `docs/BACKWARDS_COMPATIBILITY.md` — v3.0.0 breaking changes table with file:line refs
- Deprecation warning for `TRELIX_RETRIEVAL_FLARE_MAX_ITER` regression-tested

### Breaking Changes
None — all changes are additive or fail-fast safety improvements.

## [2.4.0] — 2026-07-04

### Overview
Six backlog items shipped across Plans A–F. 1,467 unit tests passing, all features default-ON or backward-compatible.

### ⚠️ BREAKING CHANGE — `search_code` MCP tool response envelope

**Before (v2.3.0):** `search_code` returned `list[dict]` directly.

**After (v2.4.0):** `search_code` returns a pagination envelope:
```json
{"results": [...], "next_cursor": 10, "total_available": 25}
```

**Migration:** Update any MCP client code that iterates `search_code(...)` directly:
```python
# Before
for result in search_code(query="auth", repo_path="/repo"):
    ...

# After
response = search_code(query="auth", repo_path="/repo")
for result in response["results"]:
    ...
# Paginate: pass response["next_cursor"] as cursor= for the next page
```

### Added — Config field rename: `flare_max_retries` (Plan A)
- **`flare_max_retries`** replaces `flare_max_iterations` in `RetrievalConfig`
- Both `TRELIX_RETRIEVAL_FLARE_MAX_RETRIES` (new) and `TRELIX_RETRIEVAL_FLARE_MAX_ITER` (old) accepted via `AliasChoices`
- Using the old env var emits `DeprecationWarning`; old name removed in v3.0.0
- **⚠️ Range constraint:** field enforces `ge=1, le=3`. If you previously set `TRELIX_RETRIEVAL_FLARE_MAX_ITER` to a value >3, lower it before upgrading or pydantic raises `ValidationError` at startup.

### Added — Multi-Query Expansion Observability (Plan B)
- **`ExpandResult`** dataclass — `(queries, llm_used, elapsed_ms)` returned by `MultiQueryExpander.expand()`
- Three new nullable columns in `query_telemetry`: `expansion_used`, `expansion_variants`, `expansion_elapsed_ms`
- `TelemetryWriter.record()` accepts optional `expansion_result=` to persist expansion metadata
- Migration: idempotent `ALTER TABLE ADD COLUMN` — existing DBs upgraded automatically

### Added — FederatedRetriever TTL Cache (Plan C)
- **`FederatedRetriever(registry, cache_ttl=120.0)`** — SHA-256-keyed in-memory cache, thread-safe via `threading.Lock`
- `cache_ttl=0` disables caching; `cache_stats()` returns hit/miss/size; `clear_cache()` for forced eviction
- Expected ~90% hit rate for typical debugging-session query patterns

### Added — GitHub PR API integration (Plan D)
- **`GitHubPRClient`** — fetch PR file diffs and post review comments via GitHub REST API
- **`trelix review --pr owner/repo#N`** — fetches PR diff from GitHub and runs `DiffReviewer`
- **`trelix review --pr owner/repo#N --post-comments`** — posts findings back as a single batched GitHub review
- Token from `GITHUB_TOKEN` env var only; handles all 7 file status values; 3,000-file truncation warning

### Added — Multi-repo file watching (Plan E)
- **`MultiRepoWatcher`** — single `watchfiles.awatch(*all_paths)` call watching all registered repos simultaneously
- Hash guard prevents re-index cascade loops; deleted files are removed from the SQLite index + vector store
- **`trelix watch-all`** — new CLI command; shows per-repo stats on exit; graceful Ctrl+C shutdown

### Added — MCP pagination + progress notifications (Plan F)
- **`search_code` pagination** — `cursor=` (offset) + `next_cursor` in response; MCP-spec-approved pattern for large payloads
- **`index_codebase` progress** — `ctx.report_progress()` sends `notifications/progress` during indexing stages (best-effort)

## [2.3.0] — 2026-07-02

### Overview
Five research-grounded intelligence and infrastructure upgrades. All features default **OFF** — zero regression when disabled. 42/42 e2e checks pass, 1458 unit tests, zero blockers.

### Added — Embedding Dimension Guard (Plan E)
- **`DimensionGuard`** — detects provider/dimension mismatch at `Retriever.__init__` startup; raises `DimensionMismatchError` with exact `trelix migrate-vectors --reset` recovery instruction
- **`index_metadata` SQLite table** — records embedding dimension after each successful index run
- **`trelix migrate-vectors --reset`** — clears `chunk_embeddings` + dimension metadata for fresh re-index after provider switch
- Prevents silent wrong-results bug when switching e.g. Azure (3072-dim) → local (384-dim)

### Added — Multi-Query Retrieval Wiring (Plan A)
- **`MultiQueryExpander` wired** into `_retrieve_standard` — the class already existed; this commit connects it to the live retrieval pipeline
- When `TRELIX_RETRIEVAL_MULTI_QUERY=true`, primary query expands to N variants, each runs all retrieval legs in parallel via `ThreadPoolExecutor`, results merge into `leg_results_list` before RRF fusion
- `variants[1:]` used (not `variants[:]`) — original query never runs twice
- Falls back gracefully (non-fatal `logger.warning`) when LLM unavailable

### Added — MCP Resources + Prompts (Plan B)
- **MCP Resources** (application-controlled URI-addressable data):
  - `trelix://index/stats` — aggregate index statistics
  - `trelix://repo/{repo_path}/manifest` — indexed file list
  - `trelix://repo/{repo_path}/symbols/{qualified_name}` — symbol source code
- **MCP Prompts** (reusable LLM interaction templates):
  - `trelix-search` — structured code search prompt
  - `trelix-explain` — symbol explanation prompt
  - `trelix-blast-radius` — impact analysis prompt
- All resource handlers return JSON even on error; stdout stays clean for MCP stdio protocol
- Research basis: MCP spec (5× 3-0 adversarial votes on Resources/Templates/Prompts primitives)

### Added — Semantic PR/Diff Review (Plan C)
- **`DiffParser`** — parses unified git diff into `DiffHunk` objects; `from_git(repo, base, head)` uses `subprocess.run` with `shell=False` (no injection risk); `to_search_query()` extracts identifiers for hybrid retrieval
- **`DiffReviewer(config).review(hunks)`** — retrieval-augmented review: each hunk → search query → retrieve context → LLM generates `ReviewComment` objects; crash-safe, never raises
- **`trelix review <repo> [--diff <file>] [--base] [--head] [--json]`** CLI command with Rich table output

### Added — Multi-Repo Federated Search (Plan D)
- **`RepoRegistry`** — load/save/manage `~/.config/trelix/repos.json`; `add(alias, path, weight)`, `remove`, `list`; raises `ValueError` on duplicate alias
- **`FederatedRetriever(registry, max_workers=4).retrieve(query, k)`** — parallel fan-out across registered repos via `ThreadPoolExecutor`; RRF merge; deduplicates by `(file_path, symbol_id)`; crash-safe (returns `[]` when all repos fail)
- **`trelix search-all <query>`** — federated search CLI
- **`trelix federation add/list`** — registry management CLI
- Config: `federation_enabled=False` (`TRELIX_FEDERATION_ENABLED`), `federation_max_workers=4`

### Breaking Changes
None — all new features are opt-in via config flags.

### v2.4.0 Backlog
- Multi-query expansion observability (log which mode: LLM-assisted vs fallback)
- MCP subscription/streaming (server-push on index changes)
- FederatedRetriever caching layer for repeated queries
- `trelix review` integration with GitHub PR API
- Real-time multi-repo watch (`trelix watch-all`)

---

## [2.2.0] — 2026-07-01

### Overview
Intelligence upgrades: an agentic ReAct retrieval loop, static analysis (data-flow
and taint), and two new hybrid-search legs (sparse SPLADE-Code, multi-granularity
chunking). All opt-in via config flags that default to `False` — zero regression
when disabled.

### Added
- **Agentic ReAct loop** (`agentic_enabled`, `TRELIX_RETRIEVAL_AGENTIC=true`) —
  multi-turn retrieve → observe → re-retrieve loop with self-correction, replacing
  the single-shot Retriever → Synthesizer chain when enabled.
- **Data-flow analysis** (`dataflow_enabled`, `TRELIX_PARSER_DATAFLOW=true`) —
  per-function def-use chains extracted via a tree-sitter AST walk, stored in the
  `def_use_edges` table.
- **Taint analysis** (`taint_enabled`; `pip install trelix[taint]` then
  `trelix taint .`) — Semgrep-backed source→sink flow detection, findings stored
  in `taint_flows`.
- **Multi-granularity indexing** (`multi_granularity_enabled`,
  `TRELIX_CHUNKER_MULTI_GRANULARITY=true`) — block- and statement-level
  sub-chunks indexed as a 6th RRF leg alongside symbol-level chunks.
- **Sparse+dense hybrid retrieval** (`sparse_enabled`, `TRELIX_RETRIEVAL_SPARSE=true`)
  — SPLADE-Code sparse embeddings as a 7th RRF leg alongside BM25, with a
  memoized, thread-safe `SparseEmbedder`.

## [2.1.0] — 2026-06-30

### Overview
Two major feature sets landing together. Phase A ships the Knowledge Graph layer (v2.0.0 development).
Phase B is the Beast-Mode Upgrade: seven research-grounded retrieval improvements, all opt-in via
config flags that default to `False` — zero regression when disabled.

### Added — Knowledge Graph (Phase A)
- **Knowledge Graph**: new `trelix/graph/` module unifying call/import/type edges into a traversable `CodeGraph` (NetworkX MultiDiGraph)
- **Community Detection**: Louvain algorithm clusters codebase into architectural modules; `trelix graph ./repo` CLI command shows top communities
- **Semantic Concepts**: `ConceptExtractor` — LLM-powered extraction of architectural concepts from symbol batches (crash-safe, returns `[]` on any failure)
- **Graph Visualization**: `GraphVisualizer.export_html()` — Pyvis interactive HTML with community coloring and edge-type coloring; `pip install trelix[knowledge-graph]`
- **4th Retrieval Leg**: `graph_search_enabled=True` in `RetrievalConfig` enables CodeGraph BFS as a 4th search leg after RRF fusion
- **REST API**: `GET /graph`, `GET /graph/communities`, `GET /graph/visualize`, `GET /graph/search` endpoints
- **MCP Tools**: `build_knowledge_graph` and `graph_search_mcp` tools in `trelix-mcp`
- **Graph Persistence**: `graph_metadata` SQLite table stores community and degree centrality per symbol
- **PageRank symbol boosting** (`pagerank_boost_enabled`) — scores symbols by import-graph centrality; boosts high-centrality symbols post-rerank
- **Incremental graph updater** — `GraphUpdater.update_file()` refreshes community + PageRank for a changed file; wired into `trelix watch`

### Added — Beast-Mode Retrieval (Phase B)
- **File-summary 5th retrieval leg** (`file_summary_leg_enabled`) — RAPTOR-style file-level embeddings used as a 5th RRF leg (arXiv:2401.18059); requires `TRELIX_FILE_SUMMARIES_ENABLED=true` at index time
- **HyDE fallback** (`hyde_fallback_enabled`) — Hypothetical Document Embeddings (arXiv:2212.10496): generates a synthetic code snippet, embeds it instead of the raw NL query
- **Multi-query expansion** (`multi_query_enabled`) — decomposes a query into N variants, retrieves independently, RRF-fuses for broader recall
- **FLARE re-retrieval loop** (`flare_enabled`) — confidence-gated iterative retrieval (arXiv:2305.06983): re-retrieves when synthesis output contains uncertainty phrases
- **Query telemetry** (`telemetry_enabled`) — `TelemetryWriter` writes per-query rows (latency, intent, result count) to `query_telemetry` SQLite table; `trelix telemetry` CLI shows recent queries
- **CoIR evaluation harness** — `trelix eval --golden <file>` reports nDCG@10, Recall@10, MRR (CoIR format, ACL 2025 arXiv:2407.02883); pure-Python `trelix.eval.ndcg` with no pandas dependency

### Breaking Changes
- **CLI**: `trelix graph` renamed to `trelix call-graph` (the old call-graph/callers display).
  The name `trelix graph` now refers to the knowledge graph build command.
  Update any scripts using `trelix graph <repo> <symbol>` to `trelix call-graph <repo> <symbol>`.

---

## [2.0.0] — 2026-06-28

### Overview
Major feature release spanning three research-grounded upgrade phases. Phase 1 delivers CoIR SOTA embedding models (BGE-Code-v1 at 81.77, Nomic CodeRankEmbed) and Voyage Matryoshka compact dimensions. Phase 2 adds RAPTOR-style multi-granularity file summaries, the PLAID ColBERT late-interaction reranker (7–45× faster than exact ColBERT), and live streaming synthesis for `trelix ask`. Phase 3 ships a LanceDB vector backend (3–5× faster insert at 100k+ chunks) and a production-ready REST API (`trelix serve`) with SSE streaming and full CRUD index management. An LLM-as-judge evaluator rounds out the quality measurement story.

### Added
- **BGE-Code-v1 embedder** (`bge-code` provider) — BAAI CoIR SOTA 2025, self-reported 81.77 avg. `pip install trelix[bge-code]`
- **Nomic CodeRankEmbed embedder** (`nomic-code` provider) — task-prefix asymmetric encoding, no new deps. `pip install trelix[local]`
- **Voyage Matryoshka support** — `TRELIX_EMBEDDER_VOYAGE_OUTPUT_DIMENSIONS=512` passes `output_dimension` to voyage-code-3 API for compact embeddings
- **LLM-as-judge eval scorer** — `LLMJudge.score()` rates semantic retrieval quality 0.0–1.0; `EvalReport.mean_judge_score` aggregate
- **Multi-granularity file summaries** — `TRELIX_FILE_SUMMARIES_ENABLED=true` generates LLM file-level summaries alongside symbol chunks (RAPTOR-inspired, arXiv 2401.18059). Enables "explain this codebase" queries.
- **PLAID late-interaction reranker** — `rerank_provider=plaid` via RAGatouille. 7–45× faster than exact ColBERT with equivalent quality. `pip install trelix[plaid]`
- **Streaming synthesis** — `trelix ask` streams tokens live to the terminal; `GET /ask` SSE endpoint for REST clients
- **LanceDB vector backend** — `TRELIX_STORE_BACKEND=lance` enables ARM-native HNSW with 3–5× faster vector insert at 100k+ chunks. `pip install trelix[lance]`
- **REST API** — `trelix serve ./repo --port 8765` exposes `/search`, `/ask` (SSE), `/index`, `/health` endpoints via FastAPI. `pip install trelix[serve]`

### Fixed
- **pathspec DeprecationWarning** — upgraded `PathSpec.from_patterns()` call site to current API; eliminates deprecation warnings in all indexing paths

---

## [1.1.0] — 2026-06-28

### Overview
Search quality and performance release — all four phases from the v1.0.0 stress test audit.

### Added
- **Phase 1b: QueryPlan LRU cache** — `CachingPlanner` caches the gpt-4o query planner call (~2–4s). Combined with Phase 1 embedding cache, warm P50 drops from ~4,500ms to **23ms** (170× speedup). `TRELIX_RETRIEVAL_PLAN_CACHE_SIZE=128` (default).
- **Phase 3: Public graph API** — `Retriever.get_callers(symbol)`, `get_callees(symbol)`, `get_importers(path)` expose the call/import graph. New `trelix graph <repo> <symbol>` CLI subcommand.

### Fixed
- **Phase 2: File-type weighting** — README/YAML no longer outranks source code in search results. Per-language RRF score multipliers: source `1.0×`, markdown `0.3×`, yaml/json `0.5×`, html/css `0.4×`. Fixes 4/6 recall misses from v1.0.0 stress test.
- **Phase 4: tree-sitter API upgrade** — All 20 parser extractors migrated from deprecated `Language(path, name)` to `get_language()`. Eliminates 439 FutureWarnings per test run.

### Test coverage
- 1197 unit tests (was 1148), 8 warnings (was 439)

---

## [1.0.0] — 2026-06-27

### Overview
First stable release of trelix. Public Python API stabilised, all hard blockers
resolved, coverage gate at 75%, full v1 stability guarantees in effect.

### Added
- Public Python API: `from trelix import IndexConfig, Indexer, Retriever, TrelixChatClient`
- `trelix --version` / `trelix -V` flag
- SECURITY.md with responsible disclosure policy
- Versioning & Stability Policy in CONTRIBUTING.md
- Troubleshooting section in README
- trelix-langchain README.md (PyPI listing)
- Unit tests for retriever, reranker, indexer, planner, CLI, and 6 parser extractors

### Fixed
- `trelix ask` with Anthropic/Bedrock/Vertex no longer silently falls back to OpenAI
- `grep_search._body_search` bounded — eliminates OOM on large repos
- Incremental watch: debounced cross-file resolution passes
- Raw pydantic ValidationError replaced with clean user-facing messages
- Ctrl+C during indexing shows "Indexing cancelled." cleanly
- Empty search results show "No results found." instead of blank table
- `bedrock-titan` and `bedrock-cohere` now selectable via `--provider` flag
- requires-python tightened to <3.13 (honest — cp313 tree-sitter-languages unavailable)

### Changed
- Development Status: 4 - Beta → 5 - Production/Stable
- Coverage gate: fail_under = 75
- `dist/` added to .gitignore

---

## [0.7.1] — 2026-06-27

### Fixed
- **`BedrockCohereEmbedder` chunk truncation** — Bedrock validates text length before
  applying `truncate="END"`, so texts >2048 characters raised `ValidationException` at
  the API level. Now pre-truncates client-side to 2048 chars before each `invoke_model`
  call. Found during live end-to-end indexing with default `max_tokens_per_chunk=512`
  (code chunks with docstrings routinely exceed 2048 characters).

### Added
- **Bedrock full-pipeline e2e tests** — `tests/integration/test_llm_e2e.py` now includes
  two tests that index a synthetic Python repo end-to-end (walk → parse → chunk → embed
  via Bedrock → store → search) for both `bedrock-cohere` and `bedrock-titan` providers.
- **`trelix-llama-index` README** — PyPI listing now shows description and usage examples.

---

## [0.7.0] — 2026-06-27

### Overview
Universal LLM client factory — all 5 chat call sites migrated to a provider-agnostic
`TrelixChatClient` ABC. Adding any new provider requires zero changes to business logic.

### Added
- **`src/trelix/llm/` package** — `TrelixChatClient` ABC, `ChatMessage`, `ChatResponse`,
  `ToolCallResponse` dataclasses, `build_chat_client()` factory
- **`LLMConfig`** — new config class for chat providers (separate from `EmbedderConfig`).
  Added as `IndexConfig.llm` field.
- **`OpenAIBackend`** — OpenAI + Azure. Auto-detects `max_completion_tokens` vs `max_tokens`
  based on model family (gpt-4o→max_completion_tokens; gpt-4/gpt-3.5→max_tokens)
- **`AnthropicBackend`** — Anthropic Claude direct. `max_tokens=`, `system=` separate param,
  `input_schema` tool format, `end_turn`→`stop` normalization. `pip install trelix[anthropic]`
- **`BedrockBackend`** — AWS Bedrock Converse API. `inferenceConfig.maxTokens` (nested camelCase),
  `system=[{"text":...}]` top-level, content always list-of-dicts, `{"auto":{}}` tool choice.
  `pip install trelix[bedrock]`
- **`VertexBackend`** — Google Vertex AI / Gemini via google-genai SDK. `max_output_tokens` in
  `GenerateContentConfig`, `system_instruction=` param. `pip install trelix[vertex]`
- **`LiteLLMBackend`** — universal delegate for 100+ providers. `drop_params=True` suppresses
  UnsupportedParamsError. Model strings: `"bedrock/claude-3-5-sonnet"`, `"gemini/gemini-2.0-flash"`.
  `pip install trelix[litellm]`
- New optional dep groups: `[anthropic]`, `[bedrock]`, `[vertex]`, `[litellm]`, `[llm-all]`

### Changed
- All 5 LLM call sites now use `TrelixChatClient` via factory — never import provider SDKs directly
- `ContextualChunker` accepts `TrelixChatClient` (new) or raw openai client (backward compat)

### Fixed
- `_token_limit_param()` in OpenAIBackend correctly routes legacy models to `max_tokens=`
  and modern models to `max_completion_tokens=` — eliminates the recurring parameter bug
- `BedrockBackend`: base64-encoded AWS credentials (stored in `.env`) decoded transparently
- `BedrockBackend`: bare model IDs rejected by Bedrock — now uses `us.*` inference profile IDs
- Unit test isolation: `test_llm_field_on_index_config` no longer leaks `.env` provider state

### Added (post-task additions)
- **`BedrockTitanEmbedder`** — `amazon.titan-embed-text-v2:0`, configurable 256/512/1024 dims,
  normalize=True. Set `TRELIX_EMBEDDER_PROVIDER=bedrock-titan`. `pip install trelix[bedrock]`
- **`BedrockCohereEmbedder`** — `cohere.embed-english-v3`, 1024 dims, asymmetric doc/query
  retrieval (`search_document` vs `search_query` input_type). `pip install trelix[bedrock]`
- **Bedrock model fallback** — `BedrockBackend` defaults to `us.anthropic.claude-sonnet-4-6`
  (primary) with transparent auto-fallback to `us.anthropic.claude-haiku-4-5-20251001-v1:0`
  on `ValidationException`. Override via `TRELIX_LLM_BEDROCK_PRIMARY_MODEL` /
  `TRELIX_LLM_BEDROCK_FALLBACK_MODEL`.
- **Live e2e tests** — `tests/integration/test_llm_e2e.py`: 16 tests covering Azure + Bedrock
  chat (complete/stream/tool_call) + Bedrock embeddings. Skip gracefully when creds absent.

---

## [0.6.0] — 2026-06-27

### Overview
Contextual chunking is now production-ready — the feature works end-to-end with verified context summaries stored in the database and indexed in BM25. Two bugs fixed that prevented contextual summaries from actually persisting.

### Fixed
- **Contextual chunking context_summary persistence:** `ContextualChunker.build_chunks()` sets `symbol.context_summary` but the DB insert in `Indexer._insert_one()` happened before chunking ran. Fixed by adding an `UPDATE symbols SET context_summary = ?` pass after `build_chunks()` for any symbols that received summaries. All 66 test symbols now have `context_summary IS NOT NULL`.
- **Contextual chunking LLM call:** `ContextualChunker._generate_summary()` used `max_tokens=` — unsupported by gpt-4o / newer Azure. Changed to `max_completion_tokens=` (consistent with synthesizer.py fix in v0.3.0).
- **Test updated:** `test_llm_called_with_correct_arguments` asserts `max_completion_tokens` instead of `max_tokens`.

### Verified
- 66/66 symbols receive LLM context summaries stored in `symbols.context_summary`
- Summaries indexed in `symbols_fts` — BM25 searches now include them
- Recall@5: 10/10 = 100% on mini_repo (baseline maintained)

### How to Enable Contextual Chunking

```bash
TRELIX_CHUNKER_CONTEXTUAL=true
TRELIX_CHUNKER_CONTEXTUAL_MODEL=gpt-4o-mini
TRELIX_EMBEDDER_PROVIDER=openai   # or azure
trelix index ./your-repo
```

---

## [0.5.1] — 2026-06-27

### Fixed
- `trelix-mcp` README: add `<!-- mcp-name: io.github.sairam0424/trelix -->` ownership verification tag required by the official MCP registry
- `trelix-mcp` server.json: shorten description to ≤100 chars to pass registry validation

---

## [0.5.0] — 2026-06-27

### Overview
Ecosystem discoverability release — trelix is now reachable across every major surface in the AI developer ecosystem. Three new PyPI packages, MCP registry listing, GitHub Action marketplace, Homebrew tap, and awesome list submissions.

### Added

#### New PyPI Packages
- **`trelix-mcp`** (`pip install trelix-mcp`) — MCP server exposing 4 tools via stdio transport. Works with Claude Code, Cursor, Windsurf, and Continue.dev. One-command setup: `claude mcp add trelix -- trelix-mcp`.
  - `search_code(query, repo_path, k=10)` — hybrid semantic + BM25 code search
  - `index_codebase(repo_path, provider="local")` — index a repository (run once)
  - `get_symbol(qualified_name, repo_path)` — get full source of any symbol
  - `blast_radius(symbol_name, repo_path)` — find everything that depends on a symbol
- **`trelix-langchain`** (`pip install trelix-langchain`) — `TrelixRetriever(BaseRetriever)` for LangChain RAG pipelines. Returns `list[Document]` with full metadata (file, symbol, language, score, lines).
- **`trelix-llama-index`** (`pip install trelix-llama-index`) — `TrelixIndexRetriever(BaseRetriever)` for LlamaIndex. Returns `list[NodeWithScore]` with file + symbol metadata.

#### Registry & Discovery
- **Official MCP Registry** — submitted via `mcp-publisher` CLI. Server ID: `io.github.sairam0424/trelix`. Pip ownership verified via `mcp-name` tag in README.
- **Glama.ai** — `glama.json` added to repo root for automatic Glama MCP directory indexing.
- **GitHub Actions Marketplace** — `trelix-index-action@v1` at `github.com/sairam0424/trelix-index-action`. Auto-indexes any repo on push with cached `.trelix/index.db`.
- **Homebrew tap** — `brew tap sairam0424/trelix && brew install trelix` via `github.com/sairam0424/homebrew-trelix`.
- **Awesome list submissions** — PRs submitted to awesome-mcp-servers (#8787), awesome-llm-apps (#903), awesome-langchain (#426).

#### PyPI Metadata
- 5 new Topic classifiers: `Scientific/Engineering :: Artificial Intelligence`, `Software Development :: Libraries :: Application Frameworks`, `Text Processing :: Indexing`, `Internet :: WWW/HTTP :: Indexing/Search`
- 21 keywords including `mcp`, `model-context-protocol`, `langchain`, `llama-index`, `code-assistant`, `static-analysis`
- 3 new README badges: MCP Compatible, LangChain retriever, Downloads

#### CI/CD
- `release.yml` now publishes all 4 packages (`trelix`, `trelix-mcp`, `trelix-langchain`, `trelix-llama-index`) to PyPI on `v*` tag
- PyPI OIDC trusted publisher configured for all 4 packages (no stored secrets for future releases)

#### Documentation
- `docs/discoverability/ECOSYSTEM-ROADMAP.md` — full ecosystem strategy with registry URLs, submission templates, priority stack
- `docs/discoverability/AWESOME-LIST-SUBMISSIONS.md` — ready-to-submit PR bodies for 3 awesome lists
- `packages/trelix-mcp/README.md` — install, Claude Code / Cursor / Windsurf / Continue.dev setup, tools table
- `packages/trelix-mcp/server.json` — official MCP registry schema for `mcp-publisher`

### Changed
- `pyproject.toml` version `0.4.0` → `0.5.0`; all sub-packages at `0.5.0` (trelix-mcp at `0.5.1`)
- `src/trelix/__init__.py` `__version__` updated to `0.5.0`
- README: added Integrations table (MCP, LangChain, LlamaIndex, GitHub Action, Homebrew), MCP Quick Setup block, LangChain code example, Homebrew install option, GitHub Action quick-start

### Fixed
- Package builds: `LICENSE` copied into each sub-package (hatchling resolves paths relative to package root, not repo root)
- `trelix-mcp/__init__.py`: added `__all__ = ["__version__"]` for parity with other packages
- `trelix-llama-index/retriever.py`: import ordering fix (ruff I001)
- Test files: removed unused `patch` imports from `trelix-langchain` and `trelix-llama-index` test suites

---

## [0.4.0] — 2026-06-26

### Overview
Beast-mode upgrade across three axes simultaneously: **retrieval quality** (+49% embedding quality, 67% failure-rate reduction), **scale** (HNSW index, Qdrant backend), and **speed** (4x async pipeline, real-time file watcher). Grounded in 6 adversarially-verified research findings from the CoIR benchmark, Anthropic contextual retrieval research, and VLDB/ACL 2025 proceedings.

### Added

#### Quality — Retrieval & Embeddings
- **Contextual Chunking (U1):** `ContextualChunker` prepends a 2-3 sentence LLM-generated summary to each chunk before embedding AND BM25 indexing. Reduces retrieval failure rate from 5.7% → 1.9% (67% reduction). Config-gated via `TRELIX_CHUNKER_CONTEXTUAL=false` — off by default.
- **Voyage Code Embedder (U2):** New `voyage` provider using `voyage-code-3` (1024-dim, 16k context). Scores 56.26 avg on CoIR benchmark vs Ada-002's 45.59 (+24%). `pip install trelix[voyage]`.
- **Local Code Embedder (U2):** New `local-code` provider using `Salesforce/SFR-Embedding-Code-2B_R` (4096-dim, 2B params). Scores 67.41 on CoIR — 49% quality gain over Ada-002. No API key required.

#### Scale — Vector Store
- **Filterable HNSW Index (U3):** O(log n) vector search via sqlite-vec HNSW. Falls back to flat scan on older versions.
- **Qdrant Optional Backend (U4):** `QdrantVectorStore` drop-in for >500k chunk deployments. `trelix migrate-vectors --to qdrant`. `pip install trelix[qdrant]`.

#### Speed — Indexing & Updates
- **Async Batch Embedding (U5):** Phase 3 runs up to 4 concurrent embed batches via `asyncio.gather`. ~3-4x speedup on large repos.
- **File Watcher (U6):** `trelix watch <repo>` — 500ms debounced auto-reindex on file save. `pip install trelix[watch]`.

#### Intelligence — Planning & Synthesis
- **Adaptive 3-Tier Query Router (U7):** Tier 1 (direct/skip retrieval) → Tier 2 (8-intent single-step) → Tier 3 (multi-step decomposition).
- **GraphRAG Map-Reduce Synthesis (U8):** For >20 results or >8k tokens, map-reduce synthesis handles arbitrarily large corpora.

#### Precision — Call Graph
- **Call Graph Precision (U9):** 3-priority callee resolution (qualified_name → type_hint+name → name-only). ~40% fewer false-positive cross-file edges.

#### Evaluation
- **Production Eval Harness (U10):** MRR, Recall@1/5/10, NDCG@10 on 50 trelix-self queries. `make eval-full`.

### Changed
- New optional dep groups: `[voyage]`, `[qdrant]`, `[watch]`
- `BaseVectorStore` ABC introduced; `VectorStore` → `SQLiteVectorStore`
- `QueryPlanner` → `AdaptiveRouter` (backward-compatible)

### Fixed
- `synthesizer.py`: `max_completion_tokens` for gpt-4o compatibility
- Test fixtures: removed synthetic passwords that triggered GitGuardian

---

## [0.3.0] — 2026-06-26

### Added
- Removed all internal origin watermarks (`aava`, `AavaPlatformEmbedder`, `CODEINDEX_*`, `codeindex` binary)
- PyInstaller binary renamed `codeindex` → `trelix`
- Fixed `synthesizer.py` `max_completion_tokens` for gpt-4o
- Restored correct `tree_sitter_languages.get_language()` in 4 parsers
- Updated `.gitignore` to exclude `.claude/`, `uv.lock`, `dist/`

---

## [0.2.0] — 2026-06-25

### Added
- Ruby parser — completes all 20 language extractors
- PyInstaller spec (`trelix.spec`) — `dist/trelix` single-file binary
- `scripts/build-binary.sh`, `make binary` / `make binary-clean` / `make binary-install`
- GitHub Actions `build-binaries.yml` — macOS arm64 + Windows x64 matrix
- Release workflow attaches binaries to GitHub Releases
- `docs/integrations/vscode-plugin.md`

---

## [0.1.0] — 2026-06-25

### Added
- Initial release — Tree-sitter AST indexing for 20+ languages
- Hybrid search: vector (ANN, sqlite-vec) + BM25 (FTS5) + grep via RRF
- RRF fusion + call-graph / import / type-edge expansion with PageRank
- 8-intent LLM query planner
- Cohere + cross-encoder reranker
- Intent-aware context assembler (greedy / breadth_first)
- LLM synthesis via OpenAI or Azure (`trelix ask`)
- CLI: `index`, `search`, `ask`, `query`, `stats`, `update-index`
- Providers: `local` (no API key), `openai`, `azure`
- Zero-infra store: single SQLite file with sqlite-vec + FTS5 BM25

[Unreleased]: https://github.com/sairam0424/trelix/compare/v2.8.1...HEAD
[2.8.1]: https://github.com/sairam0424/trelix/compare/v2.8.0...v2.8.1
[2.8.0]: https://github.com/sairam0424/trelix/compare/v2.7.3...v2.8.0
[2.7.3]: https://github.com/sairam0424/trelix/compare/v2.7.2...v2.7.3
[2.7.2]: https://github.com/sairam0424/trelix/compare/v2.7.1...v2.7.2
[2.7.1]: https://github.com/sairam0424/trelix/compare/v2.7.0...v2.7.1
[2.7.0]: https://github.com/sairam0424/trelix/compare/v2.6.0...v2.7.0
[2.6.0]: https://github.com/sairam0424/trelix/compare/v2.5.0...v2.6.0
[2.5.0]: https://github.com/sairam0424/trelix/compare/v2.4.0...v2.5.0
[2.4.0]: https://github.com/sairam0424/trelix/compare/v2.3.0...v2.4.0
[2.3.0]: https://github.com/sairam0424/trelix/compare/v2.2.0...v2.3.0
[2.2.0]: https://github.com/sairam0424/trelix/compare/v2.1.0...v2.2.0
[2.1.0]: https://github.com/sairam0424/trelix/compare/v2.0.0...v2.1.0
[2.0.0]: https://github.com/sairam0424/trelix/compare/v1.1.0...v2.0.0
[1.1.0]: https://github.com/sairam0424/trelix/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/sairam0424/trelix/compare/v0.7.1...v1.0.0
[0.7.1]: https://github.com/sairam0424/trelix/compare/v0.7.0...v0.7.1
[0.7.0]: https://github.com/sairam0424/trelix/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/sairam0424/trelix/compare/v0.5.1...v0.6.0
[0.5.1]: https://github.com/sairam0424/trelix/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/sairam0424/trelix/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/sairam0424/trelix/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/sairam0424/trelix/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/sairam0424/trelix/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/sairam0424/trelix/releases/tag/v0.1.0
