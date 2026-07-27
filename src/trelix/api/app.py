"""
trelix REST API.

Provides HTTP endpoints for trelix search, indexing, and LLM synthesis.
The /ask endpoint uses Server-Sent Events (SSE) for streaming output.

Install:
    pip install 'trelix[serve]'

Run:
    trelix serve ./my-repo --port 8765

Endpoints:
    GET  /health                         — liveness check (never requires auth)
    GET  /search?query=&repo=&k=&cursor= — hybrid search, paginated JSON envelope
    GET  /ask?query=&repo=               — LLM synthesis, SSE stream
    POST /index                          — index a repository (body: {"repo_path": "..."})
    GET  /stats?repo=                    — index statistics

Auth
----
Every route except /health requires an ``X-Trelix-Api-Key`` header matching
``TRELIX_API_AUTH_TOKEN`` — but only when that env var is actually set. Unset
(the default) means every route stays open, matching every other opt-in flag
in this config surface (``otel_enabled``, ``telemetry_enabled``). See
``_ApiAuthSettings`` below.

Tracing
-------
Each route wraps its work in a ``pipeline_stage_span`` (see
``retrieval/otel_tracing.py``) so REST-level latency shows up in the same
trace tree as the internal retrieval spans, gated by the same
``TRELIX_OTEL_ENABLED`` flag — zero cost when disabled.

Import contract
---------------
``IndexConfig`` and ``Retriever`` are imported at *module level* (not inside
``create_app``) as an intentional design decision: it allows test code to patch
them via ``patch("trelix.api.app.Retriever")`` before ``create_app()`` is
called, which is the standard ``unittest.mock`` patching idiom.  These two
modules have no dependency on FastAPI, so the module remains importable without
``trelix[serve]`` installed.  FastAPI itself **is** imported lazily inside
``create_app()`` — that is the only optional dependency gated by this module.

Response models
----------------
Every route declares a Pydantic ``response_model`` so FastAPI's auto-generated
OpenAPI schema (``/openapi.json``) carries real per-field types instead of an
untyped ``object``/``array`` — this is what makes the schema useful as input to
an OpenAPI-codegen pass (e.g. for a generated TypeScript client). ``pydantic``
is already a core trelix dependency (via ``pydantic-settings``/``IndexConfig``),
so this adds no new dependency.
"""

from __future__ import annotations

import hmac
import logging
from collections.abc import Generator
from typing import Any

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Intentionally at module scope — see "Import contract" in the module docstring.
# Patching these via patch("trelix.api.app.Retriever") / patch("trelix.api.app.IndexConfig")
# only works when they are resolved at import time, not inside the function body.
# Neither module requires fastapi, so this file stays importable without trelix[serve].
from trelix import __version__
from trelix.core.config import IndexConfig
from trelix.retrieval.otel_tracing import pipeline_stage_span
from trelix.retrieval.retriever import Retriever

logger = logging.getLogger("trelix.api")


class _ApiAuthSettings(BaseSettings):
    """REST-API auth gate — independent of IndexConfig (which is per-request,
    per-repo). Auth is opt-in: unset TRELIX_API_AUTH_TOKEN means every route
    stays open, matching today's behavior and every other "off by default"
    flag in this config surface (otel_enabled, telemetry_enabled)."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    api_auth_token: str | None = Field(default=None, alias="TRELIX_API_AUTH_TOKEN")


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str
    version: str


class SearchResultModel(BaseModel):
    file: str
    symbol: str
    kind: str
    lines: str
    score: float
    source: str
    body: str
    language: str


class SearchResponse(BaseModel):
    """Matches the MCP `search_code` tool's pagination envelope exactly."""

    results: list[SearchResultModel]
    next_cursor: int | None
    total_available: int


class IndexResponse(BaseModel):
    """Matches Indexer.index()'s return dict exactly (both the batch and
    streaming code paths return this same shape — see indexer.py)."""

    files_found: int
    files_indexed: int
    files_skipped: int
    symbols_extracted: int
    chunks_total: int
    chunks_embedded: int
    errors: int
    elapsed_seconds: float


class StatsResponse(BaseModel):
    files: int
    symbols: int
    chunks: int


class GraphStatsResponse(BaseModel):
    node_count: int
    edge_count: int
    community_count: int
    elapsed_seconds: float


class CommunitySummaryModel(BaseModel):
    """Matches get_community_summary()'s return dict exactly (graph/community.py)."""

    community_id: int
    size: int
    top_files: list[str]
    top_symbols: list[str]
    label: str


class GraphVisualizeResponse(BaseModel):
    path: str
    node_count: int


class GraphSearchResultModel(BaseModel):
    symbol: str
    file: str
    kind: str
    score: float
    source: str


def create_app() -> Any:  # noqa: ANN201
    """Create and return the FastAPI application.

    FastAPI is imported lazily inside this function so the module is importable
    even without fastapi installed (``trelix[serve]`` is the optional extra that
    provides it).  The trelix core imports (``IndexConfig``, ``Retriever``) are
    at module scope intentionally — see the module-level docstring for details.
    """
    try:
        from fastapi import Depends, FastAPI, Header, HTTPException
        from fastapi.responses import StreamingResponse  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "FastAPI is required for trelix serve. Install with: pip install 'trelix[serve]'"
        ) from e

    app = FastAPI(title="trelix API", version=__version__)

    # Read once at app construction, not per-request — TRELIX_API_AUTH_TOKEN
    # unset means every route stays open (today's behavior, unchanged).
    auth_settings = _ApiAuthSettings()

    def verify_api_key(
        x_trelix_api_key: str | None = Header(default=None, alias="X-Trelix-Api-Key"),
    ) -> None:
        token = auth_settings.api_auth_token
        if token is None:
            return
        if x_trelix_api_key is None or not hmac.compare_digest(x_trelix_api_key, token):
            logger.warning("Rejected request: missing or invalid X-Trelix-Api-Key")
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

    auth = [Depends(verify_api_key)]

    @app.get("/health")
    def health() -> HealthResponse:
        # Intentionally NOT gated by `auth` — liveness probes (k8s, load
        # balancers) must reach this without a token.
        return HealthResponse(status="ok", version=__version__)

    @app.get("/search", dependencies=auth)
    def search(
        query: str,
        repo: str,
        k: int = 10,
        cursor: int = 0,
        intent_hint: str | None = None,
        hyde_snippet_hint: str | None = None,
    ) -> SearchResponse:
        """
        Paginated hybrid search. Matches the MCP `search_code` tool's envelope
        exactly: use cursor=0 for the first page; if next_cursor is not null,
        pass it as cursor for the next page.

        `intent_hint` (optional): one of the IntentType values
        (symbol_lookup/file_overview/feature_flow/project_overview/comparison/
        config_lookup/dependency_map/blast_radius). When set and valid, skips
        trelix's internal LLM intent classification and routes directly using
        that intent's strategy — for callers (e.g. an agent) that already
        classified the query themselves. An invalid/unrecognized value is
        never rejected — it silently falls through to normal classification.
        `hyde_snippet_hint` is only used when `intent_hint` is also valid.
        """
        from trelix.retrieval.planner.models import plan_from_intent_hint

        config = IndexConfig(repo_path=repo)
        with pipeline_stage_span(config.retrieval, "http_search", {"k": k, "cursor": cursor}):
            plan = (
                plan_from_intent_hint(query, intent_hint, hyde_snippet_hint)
                if intent_hint is not None
                else None
            )
            ctx = Retriever(config).retrieve(query, plan=plan)
            all_results = ctx.results

            page = all_results[cursor : cursor + k]
            next_cursor = cursor + k if cursor + k < len(all_results) else None

            return SearchResponse(
                results=[
                    SearchResultModel(
                        file=r.file.rel_path,
                        symbol=r.symbol.qualified_name,
                        kind=r.symbol.kind.value,
                        lines=f"{r.symbol.line_start}-{r.symbol.line_end}",
                        score=round(r.score, 4),
                        source=r.source,
                        body=r.symbol.body[:800],
                        language=r.file.language.value,
                    )
                    for r in page
                ],
                next_cursor=next_cursor,
                total_available=len(all_results),
            )

    @app.get("/ask", dependencies=auth)
    def ask(query: str, repo: str) -> Any:  # noqa: ANN201
        from fastapi.responses import StreamingResponse

        def _generate() -> Generator[str, None, None]:
            try:
                from trelix.retrieval.synthesizer import Synthesizer

                config = IndexConfig(repo_path=repo)
                with pipeline_stage_span(config.retrieval, "http_ask"):
                    ctx = Retriever(config).retrieve(query)
                    synth = Synthesizer(config.embedder)
                    for token in synth.stream(ctx, config.retrieval):
                        yield f"data: {token}\n\n"
                    yield "data: [DONE]\n\n"
            except Exception as exc:  # noqa: BLE001
                yield f"data: [ERROR: {exc}]\n\n"

        return StreamingResponse(_generate(), media_type="text/event-stream")

    @app.post("/index", dependencies=auth)
    def index_repo(body: dict[str, str]) -> IndexResponse:
        from trelix.indexing.indexer import Indexer

        config = IndexConfig(repo_path=body["repo_path"])
        with pipeline_stage_span(config.retrieval, "http_index"):
            return IndexResponse(**Indexer(config).index())

    @app.get("/stats", dependencies=auth)
    def stats(repo: str) -> StatsResponse:
        from trelix.store.db import Database

        config = IndexConfig(repo_path=repo)
        with pipeline_stage_span(config.retrieval, "http_stats"):
            db = Database(config.db_path_absolute)
            return StatsResponse(
                files=db.count_files(),
                symbols=db.count_symbols(),
                chunks=db.count_chunks(),
            )

    @app.get("/graph", dependencies=auth)
    def graph_stats(repo: str) -> GraphStatsResponse:
        """Build CodeGraph and return stats."""
        from trelix.graph.builder import GraphBuilder

        config = IndexConfig(repo_path=repo)
        with pipeline_stage_span(config.retrieval, "http_graph"):
            result = GraphBuilder(config).build(extract_concepts=False)
            return GraphStatsResponse(
                node_count=result.node_count,
                edge_count=result.edge_count,
                community_count=result.community_count,
                elapsed_seconds=round(result.elapsed_seconds, 3),
            )

    @app.get("/graph/communities", dependencies=auth)
    def graph_communities(repo: str) -> list[CommunitySummaryModel]:
        """Return community summary list."""
        from trelix.graph.builder import GraphBuilder

        config = IndexConfig(repo_path=repo)
        with pipeline_stage_span(config.retrieval, "http_graph_communities"):
            result = GraphBuilder(config).build(extract_concepts=False)
            return [CommunitySummaryModel(**c) for c in result.community_summary]

    @app.get("/graph/visualize", dependencies=auth)
    def graph_visualize(repo: str, output: str = "") -> GraphVisualizeResponse:
        """Build graph and export Pyvis HTML. Returns path and node count."""
        from pathlib import Path as _Path

        from trelix.graph.builder import GraphBuilder
        from trelix.graph.visualizer import GraphVisualizer

        config = IndexConfig(repo_path=repo)
        with pipeline_stage_span(config.retrieval, "http_graph_visualize"):
            result = GraphBuilder(config).build(extract_concepts=False)
            repo_root = _Path(repo).resolve()
            if output:
                requested = _Path(output).resolve()
                allowed = repo_root / ".trelix"
                if not str(requested).startswith(str(allowed)):
                    raise HTTPException(
                        status_code=400,
                        detail="output path must be inside <repo>/.trelix/",
                    )
                out = str(requested)
            else:
                out = str(repo_root / ".trelix" / "graph.html")
            viz = GraphVisualizer()
            path = viz.export_html(result.code_graph, out)
            return GraphVisualizeResponse(path=path, node_count=result.node_count)

    @app.get("/graph/search", dependencies=auth)
    def graph_search_endpoint(
        repo: str, symbol_id: int, depth: int = 2
    ) -> list[GraphSearchResultModel]:
        """BFS graph search starting from a symbol ID."""
        from trelix.graph.builder import GraphBuilder
        from trelix.graph.search import graph_search
        from trelix.store.db import Database

        depth = max(1, min(depth, 10))
        config = IndexConfig(repo_path=repo)
        with pipeline_stage_span(config.retrieval, "http_graph_search"):
            result = GraphBuilder(config).build(extract_concepts=False)
            db = Database(config.db_path_absolute)
            results = graph_search(
                db, result.code_graph, [symbol_id], depth=depth, max_results=20
            )
            return [
                GraphSearchResultModel(
                    symbol=r.symbol.qualified_name,
                    file=r.file.rel_path,
                    kind=r.symbol.kind.value,
                    score=round(r.score, 4),
                    source=r.source,
                )
                for r in results
            ]

    return app
