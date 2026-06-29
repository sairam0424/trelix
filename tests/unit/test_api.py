"""Tests for trelix REST API."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestTrelixAPI:
    def test_app_importable(self) -> None:
        from trelix.api.app import create_app

        assert create_app is not None

    def test_search_endpoint_exists(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
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
            assert isinstance(resp.json(), list)

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
            assert len(data) == 1
            assert data[0]["file"] == "src/auth.py"
            assert data[0]["score"] == pytest.approx(0.9)

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
