# E2E / Release-Verification Gap Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the 4 confirmed, evidence-backed gaps in trelix's existing (already comprehensive) E2E/release-verification infrastructure, without duplicating or restructuring what already works.

**Architecture:** trelix already has a well-built release-verification layer: `scripts/verify_release.py` (5-category post-release check), `tests/e2e/` (real-subprocess MCP + real fresh-venv PyPI install tests), and `release.yml`'s `smoke-test-built-artifacts` job (runs `tests/e2e/` against actually-built wheels pre-publish). This plan adds 4 narrow, independent pieces on top of that existing structure — it does NOT rebuild any of it.

**Tech Stack:** Python 3.11 (pytest, asyncio), GitHub Actions (workflow_run, matrix builds), Docker, Helm, the `mcp` Python SDK (stdio client), LangChain (`BaseRetriever`), llama_index (`BaseRetriever`).

**Spec:** No separate spec doc — this plan's context is the corrected picture from this session's research + 2 rounds of live-verified exploration (see conversation; all facts below are cited to exact files/lines, not inferred).

## Global Constraints

- Do not modify `scripts/verify_release.py`'s 5 existing check functions — they are correct and already caught 2 real production defects. Only change how/when it's *invoked*.
- Do not add new runtime dependencies. `mcp`, `langchain_core`, `llama_index.core`, `pytest` are already installed where needed.
- Every new test must be a genuine round-trip against real code (real index, real subprocess, real retriever construction) — no mocking `_get_trelix_retriever`, matching this repo's own `pyproject.toml` comment flagging the existing wholesale-mock tests as a known gap.
- Mutation-verify every new test: before committing, temporarily break the thing under test and confirm the new test fails for the documented reason, then restore.
- One task per PR/commit, consistent with this repo's established defect-fix discipline.

---

### Task 1: Automate `scripts/verify_release.py` via a new `verify-release.yml` workflow

**Files:**
- Create: `.github/workflows/verify-release.yml`
- Modify: `CONTRIBUTING.md` (~line 543-548 — the manual-step instructions)

**Context:** `scripts/verify_release.py`'s own docstring says "A REPORT, run once per release, by hand no more" — but nothing runs it automatically. `release.yml` (trigger: `push: tags: ["v*"]`) and `docker-publish.yml` (same trigger, plus `workflow_dispatch`) fire **independently** off the same tag push with no `needs`/`workflow_run` link between them. `release.yml`'s sink job is `publish` (needs: `[build-binaries, build-distributions, smoke-test-built-artifacts]`) — nothing needs it today. `verify_release.py` itself makes no network calls needing secrets beyond default `GITHUB_TOKEN` for `gh release download`; PyPI installs are anonymous/public; Docker pulls have no login step in the script today (implying the GHCR packages are public — **confirm this at Step 1 below**, don't assume).

Because the two workflows are unlinked, a single `workflow_run` trigger firing after just one of them finishing isn't enough — this task must explicitly wait for **both**.

- [ ] Step 1: Confirm `ghcr.io/sairam0424/trelix` package visibility (Settings → Packages, or `docker pull ghcr.io/sairam0424/trelix:3.2.3` from a machine with no `docker login` performed). If private, this workflow's job needs `permissions: packages: read` plus a `docker/login-action` step using `GITHUB_TOKEN` before the pull — add that step only if this check shows it's needed.
- [ ] Step 2: Create `.github/workflows/verify-release.yml`:
```yaml
name: Verify Release
on:
  workflow_run:
    workflows: ["Release", "Docker Publish"]
    types: [completed]
  workflow_dispatch:
    inputs:
      version:
        description: "Version to verify, e.g. 3.2.3 (no leading v)"
        required: true

permissions:
  contents: read
  packages: read

jobs:
  verify:
    name: Verify release artifacts
    runs-on: ubuntu-latest
    timeout-minutes: 30
    if: >
      github.event_name == 'workflow_dispatch' ||
      github.event.workflow_run.conclusion == 'success'
    steps:
      - uses: actions/checkout@11d5960a326750d5838078e36cf38b85af677262 # v4
        with:
          fetch-depth: 0
          fetch-tags: true

      - name: Resolve version and tag to verify
        id: resolve
        run: |
          if [ -n "${{ github.event.inputs.version }}" ]; then
            echo "version=${{ github.event.inputs.version }}" >> "$GITHUB_OUTPUT"
          else
            tag="$(git tag --points-at "${{ github.event.workflow_run.head_sha }}" | grep '^v' | head -1)"
            if [ -z "$tag" ]; then
              echo "no v* tag found at ${{ github.event.workflow_run.head_sha }}; nothing to verify"
              echo "version=" >> "$GITHUB_OUTPUT"
            else
              echo "version=${tag#v}" >> "$GITHUB_OUTPUT"
            fi
          fi

      # This workflow_run trigger fires once per named workflow's completion, but
      # both must be green before verify_release.py can succeed (Docker checks need
      # docker-publish.yml done, PyPI checks need release.yml done). Rather than
      # restructure the two independent, already-working pipelines, check the
      # SIBLING workflow's own latest run for this same tag here — the run that
      # finishes SECOND is the one that actually proceeds past this step.
      - name: Wait for the sibling release workflow to also be green
        id: sibling
        if: steps.resolve.outputs.version != ''
        env:
          GH_TOKEN: ${{ github.token }}
        run: |
          tag="v${{ steps.resolve.outputs.version }}"
          for wf in "Release" "Docker Publish"; do
            conclusion="$(gh run list --workflow "$wf" --branch "$tag" --limit 1 --json conclusion -q '.[0].conclusion')"
            echo "$wf: $conclusion"
            if [ "$conclusion" != "success" ]; then
              echo "ready=false" >> "$GITHUB_OUTPUT"
              echo "$wf has not succeeded yet for $tag — skipping this run; the other workflow's completion will re-trigger this check."
              exit 0
            fi
          done
          echo "ready=true" >> "$GITHUB_OUTPUT"

      - uses: actions/setup-python@0b93645e9fea7318ecaed2b359559ac225c90a2 # v5
        if: steps.sibling.outputs.ready == 'true'
        with:
          python-version: "3.11"

      - name: Install pip-audit + verify_release.py's own runtime needs
        if: steps.sibling.outputs.ready == 'true'
        run: python -m pip install --quiet pip-audit

      - uses: azure/setup-helm@b9e51907a09c216f16ebe8536097933489208112 # v5
        if: steps.sibling.outputs.ready == 'true'

      - name: Run scripts/verify_release.py
        if: steps.sibling.outputs.ready == 'true'
        env:
          GH_TOKEN: ${{ github.token }}
        run: python scripts/verify_release.py --version "${{ steps.resolve.outputs.version }}"
```
Pin the `actions/setup-python` and `azure/setup-helm` SHAs to whatever this repo's other workflows already use (grep `helm-lint.yml` and `ci.yml` for the exact pinned SHAs already in use — reuse them verbatim rather than inventing new pins).
- [ ] Step 3: Test the new workflow without waiting for a real release: push this file to a branch, then manually trigger it via `gh workflow run "Verify Release" -f version=3.2.3` (the current released version) against a PR branch, and confirm the job runs `scripts/verify_release.py --version 3.2.3` and reports the same PASS summary a manual run already produces.
- [ ] Step 4: Update `CONTRIBUTING.md` (~line 543-548): change the instruction from "run this by hand" to state that `verify-release.yml` now runs this automatically after both release workflows finish, and that the manual command remains available for ad-hoc re-verification (e.g. after fixing a bug in the script itself, or re-checking an older release) — keep the exact command documented, don't remove it.
- [ ] Step 5: Commit as `feat(ci): automate scripts/verify_release.py via a new verify-release workflow`.

---

### Task 2: Real `search_code` functional round-trip in `tests/e2e/test_mcp_stdio_e2e.py`

**Files:**
- Modify: `tests/e2e/test_mcp_stdio_e2e.py`

**Context:** The existing tests only check the server boots and lists `search_code` as an available tool — nothing calls it. `search_code(query, repo_path, k=10, cursor=0, intent_hint=None, hyde_snippet_hint=None) -> dict` (server.py:276-347) delegates to `Retriever(IndexConfig(repo_path=repo_path)).retrieve(query)` and returns `{"results": [{"file", "symbol", "kind", "lines", "score", "source", "body", "language"}, ...], "next_cursor", "total_available"}`. `tests/fixtures/mini_repo/` (7 real files: `auth.py`, `user.py`, `api.py`, `utils.py`, `main.py`, plus 2 non-Python files) is the repo's own reusable fixture tree, already indexed the same way in `tests/integration/test_eval.py` via `Indexer(IndexConfig(repo_path=..., incremental=False, parse_workers=2, embedder=EmbedderConfig(provider="local")), quiet=True).index()`.

- [ ] Step 1: Add a module-scoped fixture to `tests/e2e/test_mcp_stdio_e2e.py` that copies `tests/fixtures/mini_repo/` into a fresh `tmp_path_factory` dir and indexes it with the local embedder, mirroring `tests/integration/test_eval.py`'s `mini_repo_dir`/`mini_repo_config` pattern exactly (same copy-then-`Indexer(...).index()` shape):
```python
import shutil
from pathlib import Path

import pytest

from trelix.core.config import EmbedderConfig, IndexConfig
from trelix.indexing.indexer import Indexer

_MINI_REPO = Path(__file__).parent.parent / "fixtures" / "mini_repo"


@pytest.fixture(scope="module")
def indexed_mini_repo(tmp_path_factory) -> Path:
    dest = tmp_path_factory.mktemp("mcp_search_code_e2e")
    for f in _MINI_REPO.iterdir():
        if f.is_file():
            shutil.copy2(f, dest / f.name)
    config = IndexConfig(
        repo_path=str(dest),
        incremental=False,
        parse_workers=2,
        embedder=EmbedderConfig(provider="local"),
    )
    Indexer(config, quiet=True).index()
    return dest
```
- [ ] Step 2: Before writing the assertion, run `python -c "..."` (or a throwaway script) that starts `trelix-mcp` and calls `session.call_tool("search_code", {...})` against this fixture to inspect the REAL shape of `CallToolResult.content` (the ground-truth research could not confirm this without running it — `fastmcp` marshals a returned `dict` into either a single `TextContent` with JSON-encoded text, or `structuredContent`; check both `result.content` and `result.structuredContent` on the real object). Do not guess this — it determines exactly how Step 3's assertion parses the result.
- [ ] Step 3: Add the new test, using whichever content-access pattern Step 2 confirmed:
```python
async def test_search_code_finds_real_results_in_a_real_index(indexed_mini_repo: Path) -> None:
    async def run() -> None:
        async with _connected_session() as session:
            await session.initialize()
            result = await session.call_tool(
                "search_code",
                {"query": "authenticate user", "repo_path": str(indexed_mini_repo)},
            )
            assert result.isError is False
            payload = ...  # parsed per Step 2's finding
            assert payload["total_available"] > 0
            files = {r["file"] for r in payload["results"]}
            assert "auth.py" in files

    await asyncio.wait_for(run(), timeout=_HANDSHAKE_TIMEOUT)
```
- [ ] Step 4: Mutation-verify: temporarily rename `search_code`'s `@mcp.tool()` registration (or make it return `{"results": [], "next_cursor": None, "total_available": 0}` unconditionally) and confirm this new test fails with a clear assertion error (not a timeout/hang) before restoring.
- [ ] Step 5: Run `python -m pytest tests/e2e/test_mcp_stdio_e2e.py -v` and confirm all tests (existing + new) pass.
- [ ] Step 6: Commit as `test(mcp-e2e): add a real search_code round-trip against a real indexed repo`.

---

### Task 3: Functional round-trip tests for `TrelixRetriever` and `TrelixIndexRetriever` against a real index

**Files:**
- Create: `packages/trelix-langchain/tests/e2e/test_retriever_e2e.py`
- Create: `packages/trelix-llama-index/tests/e2e/test_retriever_e2e.py`
- Modify: `.github/workflows/release.yml` (`smoke-test-built-artifacts` job's pytest invocation)

**Context:** Both packages' existing `tests/test_retriever.py` mock `_get_trelix_retriever` wholesale — their own `pyproject.toml` comments admit "the suite passes against a core that isn't installed." `TrelixRetriever(repo_path: str, provider: str = "local", k: int = 10)` (LangChain, Pydantic fields, call via `.invoke(query)` → `list[Document]` with `page_content=r.symbol.body`, rich `metadata`). `TrelixIndexRetriever(repo_path: str, provider: str = "local", k: int = 10)` (llama_index, plain `__init__`, call via `.retrieve(query)` → `list[NodeWithScore]` wrapping `TextNode(text=r.symbol.body, metadata={"file", "symbol"})`). Both internally build a real `trelix.retrieval.retriever.Retriever` when NOT mocked. `release.yml`'s `smoke-test-built-artifacts` job already installs all 4 built wheels and runs `pytest tests/e2e/ tests/integration/test_cli.py` — this is the natural place to also run these two new suites, since the real wheels (including `trelix` core) are already installed there.

- [ ] Step 1: Create the new directories (`packages/trelix-langchain/tests/e2e/`, `packages/trelix-llama-index/tests/e2e/`) — keeping these physically separate from each package's existing `tests/test_retriever.py` (which stays as the fast hermetic unit test; this is additive, not a replacement).
- [ ] Step 2: `packages/trelix-langchain/tests/e2e/test_retriever_e2e.py`:
```python
from pathlib import Path

from trelix.core.config import EmbedderConfig, IndexConfig
from trelix.indexing.indexer import Indexer
from trelix_langchain import TrelixRetriever

_AUTH_PY = '''
def authenticate_user(username: str, password: str) -> bool:
    """Verify a user's credentials against the stored hash."""
    return _check_password(username, password)


def _check_password(username: str, password: str) -> bool:
    return True
'''


def _index_real_repo(tmp_path: Path) -> Path:
    (tmp_path / "auth.py").write_text(_AUTH_PY)
    config = IndexConfig(
        repo_path=str(tmp_path),
        incremental=False,
        parse_workers=2,
        embedder=EmbedderConfig(provider="local"),
    )
    Indexer(config, quiet=True).index()
    return tmp_path


def test_trelix_retriever_returns_real_documents_from_a_real_index(tmp_path: Path) -> None:
    repo = _index_real_repo(tmp_path)
    retriever = TrelixRetriever(repo_path=str(repo))  # no mocking of _get_trelix_retriever
    docs = retriever.invoke("authenticate a user")
    assert len(docs) > 0
    assert any("authenticate_user" in d.page_content for d in docs)
    assert any(d.metadata["source"] == "auth.py" for d in docs)
```
- [ ] Step 3: `packages/trelix-llama-index/tests/e2e/test_retriever_e2e.py` — same fixture pattern, using `TrelixIndexRetriever(repo_path=str(repo)).retrieve("authenticate a user")` and asserting on `NodeWithScore.node.text` / `.node.metadata["file"]` instead of `Document.page_content`/`.metadata["source"]`.
- [ ] Step 4: Mutation-verify each: temporarily change `provider="local"`'s default to point at a nonexistent/empty repo path (or monkeypatch `Retriever.retrieve` to return an empty context) and confirm the new test fails with a clear assertion, then restore.
- [ ] Step 5: In `.github/workflows/release.yml`, extend the `smoke-test-built-artifacts` job's pytest line from `pytest tests/e2e/ tests/integration/test_cli.py` to also include `packages/trelix-langchain/tests/e2e/ packages/trelix-llama-index/tests/e2e/`. Before doing so, confirm this job's steps already prefetch the local embedder model (grep the job for a "Prefetch the local embedder model" step matching `release.yml`'s separate `test` job) — if absent, add the equivalent prefetch step here too, since these new tests construct a real local-provider embedder.
- [ ] Step 6: Run `python -m pytest packages/trelix-langchain/tests/ packages/trelix-llama-index/tests/ -v` locally (with `trelix` importable from `src/`) and confirm both existing (mocked) and new (real) tests pass.
- [ ] Step 7: Commit as `test(retrievers-e2e): add real-index round-trip tests for TrelixRetriever and TrelixIndexRetriever`.

---

### Task 4: Smoke-test the `-local` Docker variant in `ci.yml`'s `docker-build` job

**Files:**
- Modify: `.github/workflows/ci.yml` (`docker-build` job, lines 446-466)

**Context:** `docker-build` currently only builds/verifies the slim (`EXTRAS=serve`) image on every push/PR — the `-local` variant (`EXTRAS=serve,local`, bundles torch/sentence-transformers) is first checked only in `docker-publish.yml` (post-merge, at tag time) or by `scripts/verify_release.py` (manual, post-publish). The slim job currently runs in ~1.5-2 minutes thanks to `cache-from/cache-to: type=gha`; the `-local` build is materially heavier (multi-GB image per `docker-publish.yml`'s own comments) and would add real minutes to every push/PR if run unconditionally — this repo's Dockerfile comments already document a past disk-space incident from this exact build target.

**Decision point (flagged, not silently assumed):** given the cost tradeoff, this task gates the new `-local` steps to only run when Docker-relevant files actually change, rather than on every push/PR. If you'd rather accept the ~5-10 min cost on every push for tighter coverage, drop the `if:` conditions in Step 2 and let it run unconditionally instead.

- [ ] Step 1: Add a `paths-filter`-style condition. Since `ci.yml`'s existing jobs don't already use `dorny/paths-filter` or similar, the simplest addition consistent with this repo's style is a workflow-level `on.push.paths`/`on.pull_request.paths` HAS to apply to the whole workflow, not one job — so instead gate the new steps with a job-level `if:` checking `github.event_name == 'schedule'` OR add a manual `workflow_dispatch` escape hatch, whichever the team prefers. Simplest concrete choice for this plan: run the `-local` build steps only when the PR/push touches `Dockerfile` or `pyproject.toml`, using `tj-actions/changed-files` (or, if this repo avoids third-party actions for this, a plain `git diff --name-only` step) to set a step output, then `if:` that output on the 3 new steps below.
- [ ] Step 2: Add, after the existing line 466, mirroring the slim steps exactly per the established `EXTRAS=serve,local` ↔ `-local` convention already used in `docker-publish.yml`:
```yaml
      - name: Build local image
        if: steps.docker-relevant-changes.outputs.changed == 'true'
        uses: docker/build-push-action@53b7df96c91f9c12dcc8a07bcb9ccacbed38856a # v7
        with:
          context: .
          build-args: EXTRAS=serve,local
          tags: trelix:ci-test-local
          push: false
          load: true
          cache-from: type=gha,scope=local
          cache-to: type=gha,mode=max,scope=local
      - name: Verify local image runs
        if: steps.docker-relevant-changes.outputs.changed == 'true'
        run: docker run --rm trelix:ci-test-local --help
      - name: Verify trelix-mcp is present in the local image
        if: steps.docker-relevant-changes.outputs.changed == 'true'
        run: docker run --rm --entrypoint trelix-mcp trelix:ci-test-local --version
      - name: Verify torch/sentence-transformers actually import in the local image
        if: steps.docker-relevant-changes.outputs.changed == 'true'
        run: docker run --rm --entrypoint python trelix:ci-test-local -c "import torch, sentence_transformers"
```
Use a **distinct** `gha` cache scope (`scope=local`) so the slim and local builds' layer caches don't collide, per the ground-truth finding that they'd otherwise overwrite each other.
- [ ] Step 3: Confirm on a real PR that touches `Dockerfile` that the 4 new steps actually run (check the Actions run log), and on a PR that doesn't touch `Dockerfile`/`pyproject.toml` that they're skipped (job still passes, just without the extra ~5-10 min).
- [ ] Step 4: Commit as `ci(docker): smoke-test the -local image variant when Docker-relevant files change`.

---

## Self-Review

**Spec coverage** — all 4 gaps from the corrected research map to exactly one task: verify_release.py automation → Task 1; search_code functional round-trip → Task 2; retriever functional round-trips → Task 3; -local Docker pre-merge coverage → Task 4.

**Placeholder scan** — Task 1's workflow YAML is fully formed except the two action SHA pins deliberately deferred to "reuse whatever's already pinned elsewhere" (a real repo-consistency requirement, not a placeholder). Task 2 Step 2 deliberately requires a live investigation (the exact `CallToolResult.content` shape) before the assertion can be finalized — flagged explicitly rather than guessed, consistent with this repo's verify-before-assert culture. Task 3's fixture code and assertions are concrete and complete. Task 4's gating mechanism names a specific tool (`tj-actions/changed-files`) but flags the plain-`git diff` alternative and leaves the unconditional-cost alternative explicit as a real choice.

**Type consistency** — `TrelixRetriever`/`TrelixIndexRetriever` constructor signatures and return shapes (Task 3) match exactly what the ground-truth exploration read from the real source files, not guessed. `search_code`'s return dict shape (Task 2) matches server.py:331-347 verbatim.
