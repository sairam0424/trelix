"""
trelix REST API.

Provides HTTP endpoints for trelix search, indexing, and LLM synthesis.
The /ask endpoint uses Server-Sent Events (SSE) for streaming output.

Install:
    pip install 'trelix[serve]'

Run:
    trelix serve ./my-repo --port 8765

Endpoints:
    GET  /health                    — liveness check
    GET  /search?query=&repo=&k=   — hybrid search, returns JSON
    GET  /ask?query=&repo=          — LLM synthesis, SSE stream
    POST /index                    — index a repository (body: {"repo_path": "..."})
    GET  /stats?repo=               — index statistics
"""
from __future__ import annotations

import logging
from typing import Any

from trelix.core.config import IndexConfig
from trelix.retrieval.retriever import Retriever

logger = logging.getLogger("trelix.api")


def create_app():  # noqa: ANN201
    """Create and return the FastAPI application.

    FastAPI is imported lazily so this module is importable even without
    fastapi installed (``trelix[serve]`` is the optional extra that adds it).
    """
    try:
        from fastapi import FastAPI
        from fastapi.responses import StreamingResponse  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "FastAPI is required for trelix serve. "
            "Install with: pip install 'trelix[serve]'"
        ) from e

    app = FastAPI(title="trelix API", version="1.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": "1.1.0"}

    @app.get("/search")
    def search(query: str, repo: str, k: int = 10) -> list[dict[str, Any]]:
        config = IndexConfig(repo_path=repo)
        ctx = Retriever(config).retrieve(query)
        return [
            {
                "file": r.file.rel_path,
                "symbol": r.symbol.qualified_name,
                "kind": r.symbol.kind.value,
                "lines": f"{r.symbol.line_start}-{r.symbol.line_end}",
                "score": round(r.score, 4),
                "source": r.source,
                "body": r.symbol.body[:800],
                "language": r.file.language.value,
            }
            for r in ctx.results[:k]
        ]

    @app.get("/ask")
    def ask(query: str, repo: str):  # noqa: ANN201
        from fastapi.responses import StreamingResponse

        def _generate():
            try:
                from trelix.retrieval.synthesizer import Synthesizer

                config = IndexConfig(repo_path=repo)
                ctx = Retriever(config).retrieve(query)
                synth = Synthesizer(config.embedder)
                for token in synth.stream(ctx, config.retrieval):
                    yield f"data: {token}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as exc:  # noqa: BLE001
                yield f"data: [ERROR: {exc}]\n\n"

        return StreamingResponse(_generate(), media_type="text/event-stream")

    @app.post("/index")
    def index_repo(body: dict[str, str]) -> dict[str, Any]:
        from trelix.indexing.indexer import Indexer

        config = IndexConfig(repo_path=body["repo_path"])
        return Indexer(config).index()

    @app.get("/stats")
    def stats(repo: str) -> dict[str, Any]:
        from trelix.store.db import Database

        config = IndexConfig(repo_path=repo)
        db = Database(config.db_path_absolute)
        return {
            "files": db.count_files(),
            "symbols": db.count_symbols(),
            "chunks": db.count_chunks(),
        }

    return app
