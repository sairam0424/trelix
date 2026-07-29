"""
Unit tests for the Jira/TestRail source connectors (ArtifactSource
implementations). Mirrors test_github_pr.py's HTTP-mocking convention
(patch the module's `httpx.get`, not respx) since that's the established
pattern for an external API client in this codebase.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from trelix.core.config import JiraConnectorConfig, TestRailConnectorConfig
from trelix.core.models import Artifact
from trelix.indexing.connectors.base import ArtifactSource, ConnectorSyncResult
from trelix.indexing.connectors.jira import JiraConnector, JiraConnectorError
from trelix.indexing.connectors.registry import get_artifact_source
from trelix.indexing.connectors.testrail import TestRailConnector, TestRailConnectorError

_JIRA_CONFIG = JiraConnectorConfig(
    base_url="https://example.atlassian.net",
    email="me@example.com",
    api_token="tok",
    project_key="PROJ",
)

_TESTRAIL_CONFIG = TestRailConnectorConfig(
    base_url="https://example.testrail.io",
    username="me",
    api_key="key",
    project_id=1,
)


def _mock_response(
    status_code: int = 200, json_data: dict | None = None, headers=None
) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.text = str(json_data or {})
    resp.headers = headers or {}
    return resp


# ---------------------------------------------------------------------------
# JiraConnector
# ---------------------------------------------------------------------------


class TestJiraConnectorValidateConfig:
    def test_missing_all_fields_raises(self) -> None:
        connector = JiraConnector(JiraConnectorConfig())
        with pytest.raises(ValueError, match="TRELIX_JIRA_BASE_URL"):
            connector.validate_config()

    def test_missing_one_field_lists_only_that_field(self) -> None:
        config = JiraConnectorConfig(
            base_url="https://x.atlassian.net", email="e@e.com", api_token="t", project_key=None
        )
        connector = JiraConnector(config)
        with pytest.raises(ValueError, match="TRELIX_JIRA_PROJECT_KEY"):
            connector.validate_config()

    def test_all_fields_present_does_not_raise(self) -> None:
        JiraConnector(_JIRA_CONFIG).validate_config()


class TestJiraConnectorFetch:
    def test_fetch_returns_artifacts_from_issues(self) -> None:
        resp = _mock_response(
            200,
            {
                "issues": [
                    {
                        "key": "PROJ-1",
                        "fields": {
                            "summary": "Fix login bug",
                            "description": "Users cannot log in",
                            "status": {"name": "Open"},
                        },
                    }
                ],
                "nextPageToken": None,
            },
        )
        with patch("trelix.indexing.connectors.jira.httpx.get", return_value=resp):
            artifacts = JiraConnector(_JIRA_CONFIG).fetch()

        assert len(artifacts) == 1
        a = artifacts[0]
        assert a.source_ref == "ticket:PROJ-1"
        assert a.artifact_kind == "ticket"
        assert a.title == "Fix login bug"
        assert a.url == "https://example.atlassian.net/browse/PROJ-1"
        assert a.metadata["status"] == "Open"

    def test_fetch_paginates_via_next_page_token(self) -> None:
        page1 = _mock_response(
            200,
            {
                "issues": [{"key": "PROJ-1", "fields": {"summary": "One"}}],
                "nextPageToken": "token-2",
            },
        )
        page2 = _mock_response(
            200,
            {
                "issues": [{"key": "PROJ-2", "fields": {"summary": "Two"}}],
                "nextPageToken": None,
            },
        )
        with patch(
            "trelix.indexing.connectors.jira.httpx.get", side_effect=[page1, page2]
        ) as mock_get:
            artifacts = JiraConnector(_JIRA_CONFIG).fetch()

        assert {a.source_ref for a in artifacts} == {"ticket:PROJ-1", "ticket:PROJ-2"}
        assert mock_get.call_count == 2

    def test_fetch_ignores_atlassian_document_format_description(self) -> None:
        """Jira v3's description is ADF (nested JSON), not plain text — the
        connector falls back to empty body rather than crashing on it."""
        resp = _mock_response(
            200,
            {
                "issues": [
                    {
                        "key": "PROJ-1",
                        "fields": {
                            "summary": "Fix bug",
                            "description": {"type": "doc", "content": []},
                        },
                    }
                ],
                "nextPageToken": None,
            },
        )
        with patch("trelix.indexing.connectors.jira.httpx.get", return_value=resp):
            artifacts = JiraConnector(_JIRA_CONFIG).fetch()

        assert artifacts[0].body == ""

    def test_fetch_401_raises_jira_connector_error(self) -> None:
        resp = _mock_response(401, {"message": "Unauthorized"})
        with patch("trelix.indexing.connectors.jira.httpx.get", return_value=resp):
            with pytest.raises(JiraConnectorError, match="401"):
                JiraConnector(_JIRA_CONFIG).fetch()

    def test_fetch_retries_on_429_then_succeeds(self) -> None:
        rate_limited = _mock_response(429, headers={"Retry-After": "0"})
        success = _mock_response(200, {"issues": [], "nextPageToken": None})
        with (
            patch("trelix.indexing.connectors.jira.httpx.get", side_effect=[rate_limited, success]),
            patch("tenacity.nap.time.sleep"),
        ):
            artifacts = JiraConnector(_JIRA_CONFIG).fetch()

        assert artifacts == []

    def test_fetch_exhausts_retries_and_raises(self) -> None:
        rate_limited = _mock_response(429, headers={})
        with (
            patch("trelix.indexing.connectors.jira.httpx.get", return_value=rate_limited),
            patch("tenacity.nap.time.sleep"),
        ):
            with pytest.raises(JiraConnectorError, match="429"):
                JiraConnector(_JIRA_CONFIG).fetch()

    def test_fetch_network_error_never_raises_raw_httpx_exception(self) -> None:
        """A raw httpx exception must surface as JiraConnectorError, not
        leak the underlying exception type to callers."""
        import httpx

        with (
            patch(
                "trelix.indexing.connectors.jira.httpx.get",
                side_effect=httpx.ConnectError("connection refused"),
            ),
            patch("tenacity.nap.time.sleep"),
        ):
            with pytest.raises(JiraConnectorError):
                JiraConnector(_JIRA_CONFIG).fetch()


# ---------------------------------------------------------------------------
# TestRailConnector
# ---------------------------------------------------------------------------


class TestTestRailConnectorValidateConfig:
    def test_missing_all_fields_raises(self) -> None:
        connector = TestRailConnector(TestRailConnectorConfig())
        with pytest.raises(ValueError, match="TRELIX_TESTRAIL_BASE_URL"):
            connector.validate_config()

    def test_all_fields_present_does_not_raise(self) -> None:
        TestRailConnector(_TESTRAIL_CONFIG).validate_config()


class TestTestRailConnectorFetch:
    def test_fetch_returns_artifacts_from_cases(self) -> None:
        resp = _mock_response(
            200,
            {
                "cases": [
                    {
                        "id": 101,
                        "title": "Login with valid credentials",
                        "custom_preconds": "User has an account",
                        "custom_steps": "1. Enter username\n2. Enter password",
                        "priority_id": 3,
                    }
                ]
            },
        )
        with patch("trelix.indexing.connectors.testrail.httpx.get", return_value=resp):
            artifacts = TestRailConnector(_TESTRAIL_CONFIG).fetch()

        assert len(artifacts) == 1
        a = artifacts[0]
        assert a.source_ref == "test_case:101"
        assert a.artifact_kind == "test_case"
        assert "User has an account" in a.body
        assert "Enter username" in a.body
        assert a.url == "https://example.testrail.io/index.php?/cases/view/101"

    def test_fetch_paginates_via_offset(self) -> None:
        config = TestRailConnectorConfig(
            base_url="https://example.testrail.io",
            username="me",
            api_key="key",
            project_id=1,
            page_size=2,
        )
        page1 = _mock_response(200, {"cases": [{"id": 1, "title": "A"}, {"id": 2, "title": "B"}]})
        page2 = _mock_response(200, {"cases": [{"id": 3, "title": "C"}]})
        with patch(
            "trelix.indexing.connectors.testrail.httpx.get", side_effect=[page1, page2]
        ) as mock_get:
            artifacts = TestRailConnector(config).fetch()

        assert {a.source_ref for a in artifacts} == {
            "test_case:1",
            "test_case:2",
            "test_case:3",
        }
        assert mock_get.call_count == 2

    def test_fetch_401_raises_testrail_connector_error(self) -> None:
        resp = _mock_response(401)
        with patch("trelix.indexing.connectors.testrail.httpx.get", return_value=resp):
            with pytest.raises(TestRailConnectorError, match="401"):
                TestRailConnector(_TESTRAIL_CONFIG).fetch()

    def test_fetch_handles_missing_custom_fields_gracefully(self) -> None:
        """A case with no preconditions/steps set must not raise — body
        just ends up empty rather than containing 'None'."""
        resp = _mock_response(200, {"cases": [{"id": 1, "title": "Bare case"}]})
        with patch("trelix.indexing.connectors.testrail.httpx.get", return_value=resp):
            artifacts = TestRailConnector(_TESTRAIL_CONFIG).fetch()

        assert artifacts[0].body == ""

    def test_fetch_retries_on_429_then_succeeds(self) -> None:
        rate_limited = _mock_response(429, headers={"Retry-After": "0"})
        success = _mock_response(200, {"cases": []})
        with (
            patch(
                "trelix.indexing.connectors.testrail.httpx.get",
                side_effect=[rate_limited, success],
            ),
            patch("tenacity.nap.time.sleep"),
        ):
            artifacts = TestRailConnector(_TESTRAIL_CONFIG).fetch()

        assert artifacts == []

    def test_fetch_exhausts_retries_and_raises(self) -> None:
        rate_limited = _mock_response(429, headers={})
        with (
            patch("trelix.indexing.connectors.testrail.httpx.get", return_value=rate_limited),
            patch("tenacity.nap.time.sleep"),
        ):
            with pytest.raises(TestRailConnectorError, match="429"):
                TestRailConnector(_TESTRAIL_CONFIG).fetch()

    def test_fetch_network_error_never_raises_raw_httpx_exception(self) -> None:
        """A raw httpx exception must surface as TestRailConnectorError, not
        leak the underlying exception type to callers."""
        import httpx

        with (
            patch(
                "trelix.indexing.connectors.testrail.httpx.get",
                side_effect=httpx.ConnectError("connection refused"),
            ),
            patch("tenacity.nap.time.sleep"),
        ):
            with pytest.raises(TestRailConnectorError):
                TestRailConnector(_TESTRAIL_CONFIG).fetch()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestConnectorRegistry:
    def test_jira_resolves_to_jira_connector(self) -> None:
        assert isinstance(get_artifact_source("jira"), JiraConnector)

    def test_testrail_resolves_to_testrail_connector(self) -> None:
        assert isinstance(get_artifact_source("testrail"), TestRailConnector)

    def test_unknown_name_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown connector"):
            get_artifact_source("bogus")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# ArtifactSource.sync() — shared base-class behavior
# ---------------------------------------------------------------------------


class _FakeConnector(ArtifactSource):
    """Minimal ArtifactSource for testing sync()'s persistence logic in
    isolation from any real HTTP client."""

    def __init__(self, artifacts: list[Artifact], valid: bool = True) -> None:
        self._artifacts = artifacts
        self._valid = valid

    def validate_config(self) -> None:
        if not self._valid:
            raise ValueError("fake connector misconfigured")

    def fetch(self) -> list[Artifact]:
        return self._artifacts


class _FakeWriter:
    """Structurally satisfies ArtifactWriter without needing a real Database."""

    def __init__(self, fail_on: set[str] | None = None) -> None:
        self.written: list[Artifact] = []
        self._fail_on = fail_on or set()

    def upsert_artifact(self, artifact: Artifact) -> int:
        if artifact.source_ref in self._fail_on:
            raise RuntimeError("simulated write failure")
        self.written.append(artifact)
        return len(self.written)


class TestArtifactSourceSync:
    def test_sync_writes_every_fetched_artifact(self) -> None:
        artifacts = [
            Artifact(source_ref="ticket:A", artifact_kind="ticket", title="A", body=""),
            Artifact(source_ref="ticket:B", artifact_kind="ticket", title="B", body=""),
        ]
        connector = _FakeConnector(artifacts)
        writer = _FakeWriter()

        result = connector.sync(writer)  # type: ignore[arg-type]

        assert result == ConnectorSyncResult(artifacts_fetched=2, artifacts_written=2, errors=0)
        assert len(writer.written) == 2

    def test_sync_counts_write_failures_without_stopping(self) -> None:
        artifacts = [
            Artifact(source_ref="ticket:A", artifact_kind="ticket", title="A", body=""),
            Artifact(source_ref="ticket:B", artifact_kind="ticket", title="B", body=""),
        ]
        connector = _FakeConnector(artifacts)
        writer = _FakeWriter(fail_on={"ticket:A"})

        result = connector.sync(writer)  # type: ignore[arg-type]

        assert result.artifacts_fetched == 2
        assert result.artifacts_written == 1
        assert result.errors == 1

    def test_sync_calls_validate_config_before_fetch(self) -> None:
        connector = _FakeConnector([], valid=False)
        writer = _FakeWriter()

        with pytest.raises(ValueError, match="misconfigured"):
            connector.sync(writer)  # type: ignore[arg-type]
