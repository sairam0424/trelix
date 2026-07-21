"""Tests for trelix REST API."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestTrelixAPI:
    def test_app_importable(self) -> None:
        from trelix.api.app import create_app

        assert create_app is not None

    def test_search_endpoint_exists(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """GET /search returns the paginated envelope, not a bare list."""
        from fastapi.testclient import TestClient

        from trelix.api.app import create_app

        mock_ctx = MagicMock()
        mock_ctx.results = []

        with patch("trelix.api.app.Retriever") as MockRetriever:
            MockRetriever.return_value.retrieve.return_value = mock_ctx
            app = create_app()
            client = TestClient(app)
            resp = client.get(f"/search?query=auth&repo={tmp_path}&k=5")
            assert resp.status_code == 200
            data = resp.json()
            assert isinstance(data, dict)
            assert data["results"] == []
            assert data["next_cursor"] is None
            assert data["total_available"] == 0

    def test_health_endpoint(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        from fastapi.testclient import TestClient

        from trelix.api.app import create_app

        app = create_app()
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_search_returns_result_dicts(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        from fastapi.testclient import TestClient

        from trelix.api.app import create_app

        mock_result = MagicMock()
        mock_result.file.rel_path = "src/auth.py"
        mock_result.symbol.qualified_name = "AuthService.login"
        mock_result.symbol.kind.value = "method"
        mock_result.symbol.line_start = 10
        mock_result.symbol.line_end = 20
        mock_result.symbol.body = "def login(): pass"
        mock_result.file.language.value = "python"
        mock_result.score = 0.9
        mock_result.source = "vector"

        mock_ctx = MagicMock()
        mock_ctx.results = [mock_result]

        with patch("trelix.api.app.Retriever") as MockRetriever:
            MockRetriever.return_value.retrieve.return_value = mock_ctx
            app = create_app()
            client = TestClient(app)
            resp = client.get(f"/search?query=auth&repo={tmp_path}&k=5")
            data = resp.json()
            assert len(data["results"]) == 1
            assert data["results"][0]["file"] == "src/auth.py"
            assert data["results"][0]["score"] == pytest.approx(0.9)
            assert data["total_available"] == 1
            assert data["next_cursor"] is None

    def test_ask_endpoint_returns_sse_stream(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """GET /ask must return 200 with content-type text/event-stream (SSE)."""
        from fastapi.testclient import TestClient

        from trelix.api.app import create_app

        mock_ctx = MagicMock()
        mock_ctx.results = []

        with (
            patch("trelix.api.app.Retriever") as MockRetriever,
            patch("trelix.retrieval.synthesizer.Synthesizer") as MockSynth,
        ):
            MockRetriever.return_value.retrieve.return_value = mock_ctx
            MockSynth.return_value.stream.return_value = iter(["Hello", " world"])
            app = create_app()
            # stream_response=False so TestClient consumes the full SSE body
            client = TestClient(app)
            resp = client.get(f"/ask?query=hello&repo={tmp_path}")
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers["content-type"]


class TestSearchPagination:
    """GET /search's cursor/next_cursor/total_available contract, matching
    the MCP search_code tool's envelope exactly (server.py's search_code)."""

    def _mock_results(self, n: int) -> list[MagicMock]:
        results = []
        for i in range(n):
            r = MagicMock()
            r.file.rel_path = f"src/file_{i}.py"
            r.symbol.qualified_name = f"func_{i}"
            r.symbol.kind.value = "function"
            r.symbol.line_start = 1
            r.symbol.line_end = 5
            r.symbol.body = f"def func_{i}(): pass"
            r.file.language.value = "python"
            r.score = 1.0 - (i * 0.01)
            r.source = "vector"
            results.append(r)
        return results

    def test_first_page_returns_next_cursor_when_more_remain(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        from fastapi.testclient import TestClient

        from trelix.api.app import create_app

        mock_ctx = MagicMock()
        mock_ctx.results = self._mock_results(25)

        with patch("trelix.api.app.Retriever") as MockRetriever:
            MockRetriever.return_value.retrieve.return_value = mock_ctx
            app = create_app()
            client = TestClient(app)
            resp = client.get(f"/search?query=x&repo={tmp_path}&k=10&cursor=0")
            data = resp.json()
            assert len(data["results"]) == 10
            assert data["next_cursor"] == 10
            assert data["total_available"] == 25
            assert data["results"][0]["symbol"] == "func_0"

    def test_second_page_uses_next_cursor_from_first_page(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        from fastapi.testclient import TestClient

        from trelix.api.app import create_app

        mock_ctx = MagicMock()
        mock_ctx.results = self._mock_results(25)

        with patch("trelix.api.app.Retriever") as MockRetriever:
            MockRetriever.return_value.retrieve.return_value = mock_ctx
            app = create_app()
            client = TestClient(app)
            resp = client.get(f"/search?query=x&repo={tmp_path}&k=10&cursor=10")
            data = resp.json()
            assert len(data["results"]) == 10
            assert data["results"][0]["symbol"] == "func_10"
            assert data["next_cursor"] == 20
            assert data["total_available"] == 25

    def test_last_page_has_null_next_cursor(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        from fastapi.testclient import TestClient

        from trelix.api.app import create_app

        mock_ctx = MagicMock()
        mock_ctx.results = self._mock_results(25)

        with patch("trelix.api.app.Retriever") as MockRetriever:
            MockRetriever.return_value.retrieve.return_value = mock_ctx
            app = create_app()
            client = TestClient(app)
            resp = client.get(f"/search?query=x&repo={tmp_path}&k=10&cursor=20")
            data = resp.json()
            assert len(data["results"]) == 5
            assert data["next_cursor"] is None
            assert data["total_available"] == 25

    def test_cursor_defaults_to_zero(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """Omitting cursor entirely must behave identically to cursor=0."""
        from fastapi.testclient import TestClient

        from trelix.api.app import create_app

        mock_ctx = MagicMock()
        mock_ctx.results = self._mock_results(3)

        with patch("trelix.api.app.Retriever") as MockRetriever:
            MockRetriever.return_value.retrieve.return_value = mock_ctx
            app = create_app()
            client = TestClient(app)
            resp = client.get(f"/search?query=x&repo={tmp_path}&k=10")
            data = resp.json()
            assert data["results"][0]["symbol"] == "func_0"
            assert data["total_available"] == 3


class TestOpenApiSchema:
    """Regression guard: every route must keep a real Pydantic response_model
    (inferred from its return-type annotation) so /openapi.json carries actual
    per-field types instead of an untyped object/array — this is what makes
    the schema usable as input to an OpenAPI-codegen pass for a TS client."""

    def test_search_response_schema_has_typed_fields(self) -> None:
        from trelix.api.app import create_app

        app = create_app()
        schema = app.openapi()
        search_response_ref = schema["paths"]["/search"]["get"]["responses"]["200"]["content"][
            "application/json"
        ]["schema"]
        # FastAPI emits either a direct object schema or a $ref into components;
        # resolve either shape to the real SearchResponse model definition.
        if "$ref" in search_response_ref:
            model_name = search_response_ref["$ref"].rsplit("/", 1)[-1]
            model_schema = schema["components"]["schemas"][model_name]
        else:
            model_schema = search_response_ref
        properties = model_schema["properties"]
        assert set(properties) == {"results", "next_cursor", "total_available"}
        assert properties["total_available"]["type"] == "integer"

    def test_every_route_has_a_non_empty_response_schema(self) -> None:
        """No route should fall back to FastAPI's untyped default schema —
        every route must declare real response properties, EXCEPT /ask, which
        is a raw SSE token stream (not a single JSON body) and therefore has
        no meaningful static Pydantic schema — see its docstring/route body
        for the documented `data: <token>` / `data: [DONE]` event contract."""
        from trelix.api.app import create_app

        app = create_app()
        schema = app.openapi()
        for path, methods in schema["paths"].items():
            if path == "/ask":
                continue
            for method, spec in methods.items():
                response_schema = spec["responses"]["200"]["content"]["application/json"]["schema"]
                if "$ref" in response_schema:
                    continue  # resolves to a named model — has real fields by definition
                if response_schema.get("type") == "array":
                    items = response_schema["items"]
                    assert "$ref" in items or items.get("properties"), (
                        f"{method.upper()} {path} has an untyped array response"
                    )
                else:
                    assert response_schema.get("properties"), (
                        f"{method.upper()} {path} has an untyped response schema"
                    )
