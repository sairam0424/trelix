"""
Unit tests for the Jira/TestRail source connectors (ArtifactSource
implementations). Mirrors test_github_pr.py's HTTP-mocking convention
(patch the module's `httpx.get`, not respx) since that's the established
pattern for an external API client in this codebase.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from trelix.core.config import (
    JiraConnectorConfig,
    LinearConnectorConfig,
    TestRailConnectorConfig,
    XrayConnectorConfig,
)
from trelix.core.models import Artifact
from trelix.indexing.connectors.base import ArtifactSource, ConnectorSyncResult
from trelix.indexing.connectors.jira import JiraConnector, JiraConnectorError
from trelix.indexing.connectors.linear import LinearConnector, LinearConnectorError
from trelix.indexing.connectors.registry import get_artifact_source
from trelix.indexing.connectors.testrail import TestRailConnector, TestRailConnectorError
from trelix.indexing.connectors.xray import XrayConnector, XrayConnectorError

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

_XRAY_CONFIG = XrayConnectorConfig(
    client_id="cid",
    client_secret="sek",
    project_key="PROJ",
    jira_base_url="https://example.atlassian.net",
)

_LINEAR_CONFIG = LinearConnectorConfig(api_key="key123", team_key="ENG")


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

    def test_fetch_renders_atlassian_document_format_description_to_plain_text(
        self,
    ) -> None:
        """Jira v3's description is ADF (nested JSON), not plain text — the
        connector must render real, readable content out of it rather than
        dropping it to an empty body (the pre-fix behavior)."""
        resp = _mock_response(
            200,
            {
                "issues": [
                    {
                        "key": "PROJ-1",
                        "fields": {
                            "summary": "Fix bug",
                            "description": {
                                "type": "doc",
                                "content": [
                                    {
                                        "type": "paragraph",
                                        "content": [
                                            {"type": "text", "text": "Users cannot log in. See "},
                                            {
                                                "type": "text",
                                                "text": "the runbook",
                                                "marks": [
                                                    {
                                                        "type": "link",
                                                        "attrs": {
                                                            "href": "https://example.com/runbook"
                                                        },
                                                    }
                                                ],
                                            },
                                            {"type": "text", "text": " for details."},
                                        ],
                                    },
                                    {
                                        "type": "bulletList",
                                        "content": [
                                            {
                                                "type": "listItem",
                                                "content": [
                                                    {
                                                        "type": "paragraph",
                                                        "content": [
                                                            {"type": "text", "text": "Step one"}
                                                        ],
                                                    }
                                                ],
                                            },
                                            {
                                                "type": "listItem",
                                                "content": [
                                                    {
                                                        "type": "paragraph",
                                                        "content": [
                                                            {"type": "text", "text": "Step two"}
                                                        ],
                                                    }
                                                ],
                                            },
                                        ],
                                    },
                                ],
                            },
                        },
                    }
                ],
                "nextPageToken": None,
            },
        )
        with patch("trelix.indexing.connectors.jira.httpx.get", return_value=resp):
            artifacts = JiraConnector(_JIRA_CONFIG).fetch()

        body = artifacts[0].body
        assert "Users cannot log in" in body
        assert "the runbook (https://example.com/runbook)" in body
        assert "- Step one" in body
        assert "- Step two" in body

    def test_fetch_empty_adf_document_produces_empty_body(self) -> None:
        """An ADF doc with genuinely no content (not a malformed one) must
        still produce an empty body, not a crash or placeholder text."""
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


class TestAdfToText:
    """Direct tests of the ADF-to-plain-text renderer against node shapes
    confirmed by fetching real descriptions from a live Jira Cloud site —
    not just the paragraph/link/bulletList shapes already covered by
    TestJiraConnectorFetch's end-to-end test."""

    def test_heading(self) -> None:
        from trelix.indexing.connectors.jira import _adf_to_text

        doc = {
            "type": "doc",
            "content": [
                {
                    "type": "heading",
                    "attrs": {"level": 2},
                    "content": [{"type": "text", "text": "Prerequisites"}],
                }
            ],
        }
        assert _adf_to_text(doc) == "Prerequisites"

    def test_ordered_list_numbers_items(self) -> None:
        from trelix.indexing.connectors.jira import _adf_to_text

        doc = {
            "type": "doc",
            "content": [
                {
                    "type": "orderedList",
                    "content": [
                        {
                            "type": "listItem",
                            "content": [
                                {
                                    "type": "paragraph",
                                    "content": [{"type": "text", "text": "First"}],
                                }
                            ],
                        },
                        {
                            "type": "listItem",
                            "content": [
                                {
                                    "type": "paragraph",
                                    "content": [{"type": "text", "text": "Second"}],
                                }
                            ],
                        },
                    ],
                }
            ],
        }
        text = _adf_to_text(doc)
        assert "1. First" in text
        assert "2. Second" in text

    def test_code_block_preserves_code_text(self) -> None:
        from trelix.indexing.connectors.jira import _adf_to_text

        doc = {
            "type": "doc",
            "content": [
                {
                    "type": "codeBlock",
                    "attrs": {"language": "json"},
                    "content": [{"type": "text", "text": '{"key": "value"}'}],
                }
            ],
        }
        assert '{"key": "value"}' in _adf_to_text(doc)

    def test_panel_and_expand_render_nested_content(self) -> None:
        from trelix.indexing.connectors.jira import _adf_to_text

        doc = {
            "type": "doc",
            "content": [
                {
                    "type": "panel",
                    "attrs": {"panelType": "info"},
                    "content": [
                        {"type": "paragraph", "content": [{"type": "text", "text": "Need help?"}]}
                    ],
                },
                {
                    "type": "expand",
                    "attrs": {"title": "Advanced setup"},
                    "content": [
                        {
                            "type": "paragraph",
                            "content": [{"type": "text", "text": "Extra steps here."}],
                        }
                    ],
                },
            ],
        }
        text = _adf_to_text(doc)
        assert "Need help?" in text
        assert "Advanced setup:" in text
        assert "Extra steps here." in text

    def test_rule_produces_no_stray_text(self) -> None:
        from trelix.indexing.connectors.jira import _adf_to_text

        doc = {
            "type": "doc",
            "content": [
                {"type": "paragraph", "content": [{"type": "text", "text": "Above"}]},
                {"type": "rule"},
                {"type": "paragraph", "content": [{"type": "text", "text": "Below"}]},
            ],
        }
        text = _adf_to_text(doc)
        assert text == "Above\nBelow"

    def test_embed_card_renders_url(self) -> None:
        from trelix.indexing.connectors.jira import _adf_to_text

        doc = {
            "type": "doc",
            "content": [
                {
                    "type": "embedCard",
                    "attrs": {"url": "https://www.loom.com/share/abc123"},
                }
            ],
        }
        assert _adf_to_text(doc) == "https://www.loom.com/share/abc123"

    def test_unknown_node_type_falls_through_to_children(self) -> None:
        """A future/undocumented ADF node type must not crash or drop its
        text — it degrades to block-level rendering of its children."""
        from trelix.indexing.connectors.jira import _adf_to_text

        doc = {
            "type": "doc",
            "content": [
                {
                    "type": "someBrandNewNodeType",
                    "content": [
                        {"type": "paragraph", "content": [{"type": "text", "text": "Still here"}]}
                    ],
                }
            ],
        }
        assert "Still here" in _adf_to_text(doc)


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
# XrayConnector
# ---------------------------------------------------------------------------


def _mock_auth_response(status_code: int = 200, token: str = "jwt-token") -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = token
    resp.text = token
    return resp


def _mock_graphql_response(status_code: int = 200, results: list[dict] | None = None) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = {"data": {"getTests": {"results": results or []}}}
    resp.text = str(results or [])
    return resp


class TestXrayConnectorValidateConfig:
    def test_missing_all_fields_raises(self) -> None:
        connector = XrayConnector(XrayConnectorConfig())
        with pytest.raises(ValueError, match="TRELIX_XRAY_CLIENT_ID"):
            connector.validate_config()

    def test_all_fields_present_does_not_raise(self) -> None:
        XrayConnector(_XRAY_CONFIG).validate_config()


class TestXrayConnectorFetch:
    def test_fetch_authenticates_then_returns_artifacts_from_tests(self) -> None:
        auth_resp = _mock_auth_response(token="jwt-abc")
        graphql_resp = _mock_graphql_response(
            results=[
                {
                    "issueId": "10001",
                    "jira": {"key": "PROJ-1", "summary": "Login test", "status": {"name": "Open"}},
                    "testType": {"name": "Manual"},
                    "steps": [{"action": "Open login page", "data": "", "result": "Page loads"}],
                    "unstructured": "",
                }
            ]
        )
        with (
            patch(
                "trelix.indexing.connectors.xray.httpx.post", side_effect=[auth_resp, graphql_resp]
            ),
        ):
            artifacts = XrayConnector(_XRAY_CONFIG).fetch()

        assert len(artifacts) == 1
        a = artifacts[0]
        assert a.source_ref == "xray-test:PROJ-1"
        assert a.artifact_kind == "test_case"
        assert a.title == "Login test"
        assert a.url == "https://example.atlassian.net/browse/PROJ-1"
        assert a.metadata["status"] == "Open"
        assert a.metadata["test_type"] == "Manual"
        assert "Open login page" in a.body

    def test_fetch_paginates_via_limit_start(self) -> None:
        config = XrayConnectorConfig(
            client_id="cid",
            client_secret="sek",
            project_key="PROJ",
            jira_base_url="https://example.atlassian.net",
            page_size=2,
        )
        auth_resp = _mock_auth_response()
        page1 = _mock_graphql_response(
            results=[
                {"issueId": "1", "jira": {"key": "PROJ-1", "summary": "One"}},
                {"issueId": "2", "jira": {"key": "PROJ-2", "summary": "Two"}},
            ]
        )
        page2 = _mock_graphql_response(
            results=[{"issueId": "3", "jira": {"key": "PROJ-3", "summary": "Three"}}]
        )
        with patch(
            "trelix.indexing.connectors.xray.httpx.post",
            side_effect=[auth_resp, page1, page2],
        ) as mock_post:
            artifacts = XrayConnector(config).fetch()

        assert {a.source_ref for a in artifacts} == {
            "xray-test:PROJ-1",
            "xray-test:PROJ-2",
            "xray-test:PROJ-3",
        }
        assert mock_post.call_count == 3  # 1 auth + 2 graphql pages

    def test_authenticate_401_raises_xray_connector_error(self) -> None:
        resp = _mock_auth_response(status_code=401)
        with patch("trelix.indexing.connectors.xray.httpx.post", return_value=resp):
            with pytest.raises(XrayConnectorError, match="401"):
                XrayConnector(_XRAY_CONFIG).fetch()

    def test_graphql_401_raises_xray_connector_error(self) -> None:
        auth_resp = _mock_auth_response()
        graphql_resp = _mock_graphql_response(status_code=401)
        with patch(
            "trelix.indexing.connectors.xray.httpx.post",
            side_effect=[auth_resp, graphql_resp],
        ):
            with pytest.raises(XrayConnectorError, match="401"):
                XrayConnector(_XRAY_CONFIG).fetch()

    def test_graphql_error_response_raises_xray_connector_error_not_attributeerror(
        self,
    ) -> None:
        """Per the GraphQL spec, a query-level error (invalid JQL, no Xray
        license on the project, query complexity limit, ...) comes back as
        HTTP 200 with `data: null` and an `errors` array. The outer
        `.get("data", {})` doesn't help here — the "data" key IS present
        with value None, so the {} default never applies, and calling
        .get() on that None used to raise AttributeError instead of the
        documented XrayConnectorError."""
        auth_resp = _mock_auth_response()
        graphql_error_resp = MagicMock()
        graphql_error_resp.status_code = 200
        graphql_error_resp.json.return_value = {
            "data": None,
            "errors": [{"message": "Invalid JQL syntax"}],
        }
        with patch(
            "trelix.indexing.connectors.xray.httpx.post",
            side_effect=[auth_resp, graphql_error_resp],
        ):
            with pytest.raises(XrayConnectorError, match="Invalid JQL syntax"):
                XrayConnector(_XRAY_CONFIG).fetch()

    def test_authenticate_retry_exhaustion_raises_xray_connector_error(self) -> None:
        """A persistent 5xx on /authenticate must surface as
        XrayConnectorError once retries are exhausted, not leak the raw
        httpx.HTTPStatusError — matching Jira/TestRail's own contract
        where every real API failure funnels through their connector's
        own error type."""
        always_fails = MagicMock(status_code=500, text="Internal Server Error")
        with (
            patch("trelix.indexing.connectors.xray.httpx.post", return_value=always_fails),
            patch("tenacity.nap.time.sleep"),
        ):
            with pytest.raises(XrayConnectorError):
                XrayConnector(_XRAY_CONFIG).fetch()

    def test_authenticate_connection_error_raises_xray_connector_error(self) -> None:
        """A connection-level failure (DNS/timeout/refused) on
        /authenticate must surface as XrayConnectorError, not leak the raw
        httpx exception type to callers."""
        import httpx

        with (
            patch(
                "trelix.indexing.connectors.xray.httpx.post",
                side_effect=httpx.ConnectError("connection refused"),
            ),
            patch("tenacity.nap.time.sleep"),
        ):
            with pytest.raises(XrayConnectorError):
                XrayConnector(_XRAY_CONFIG).fetch()

    def test_fetch_retries_on_429_then_succeeds(self) -> None:
        auth_resp = _mock_auth_response()
        rate_limited = MagicMock(status_code=429, text="")
        success = _mock_graphql_response(results=[])
        with (
            patch(
                "trelix.indexing.connectors.xray.httpx.post",
                side_effect=[auth_resp, rate_limited, success],
            ),
            patch("tenacity.nap.time.sleep"),
        ):
            artifacts = XrayConnector(_XRAY_CONFIG).fetch()

        assert artifacts == []

    def test_fetch_network_error_never_raises_raw_httpx_exception(self) -> None:
        """A raw httpx exception must surface as XrayConnectorError, not
        leak the underlying exception type to callers."""
        import httpx

        auth_resp = _mock_auth_response()
        call_count = {"n": 0}

        def _post_side_effect(*_args: object, **_kwargs: object) -> MagicMock:
            call_count["n"] += 1
            if call_count["n"] == 1:
                return auth_resp
            raise httpx.ConnectError("connection refused")

        with (
            patch(
                "trelix.indexing.connectors.xray.httpx.post",
                side_effect=_post_side_effect,
            ),
            patch("tenacity.nap.time.sleep"),
        ):
            with pytest.raises(XrayConnectorError):
                XrayConnector(_XRAY_CONFIG).fetch()

    def test_fetch_handles_missing_steps_and_unstructured_gracefully(self) -> None:
        """A test with no steps/unstructured content must not raise — body
        just ends up empty."""
        auth_resp = _mock_auth_response()
        graphql_resp = _mock_graphql_response(
            results=[{"issueId": "1", "jira": {"key": "PROJ-1", "summary": "Bare test"}}]
        )
        with patch(
            "trelix.indexing.connectors.xray.httpx.post",
            side_effect=[auth_resp, graphql_resp],
        ):
            artifacts = XrayConnector(_XRAY_CONFIG).fetch()

        assert artifacts[0].body == ""


# ---------------------------------------------------------------------------
# LinearConnector
# ---------------------------------------------------------------------------


def _mock_linear_page(
    status_code: int = 200,
    nodes: list[dict] | None = None,
    has_next_page: bool = False,
    end_cursor: str | None = None,
    errors: list[dict] | None = None,
    headers: dict | None = None,
) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    body: dict = {}
    if errors is not None:
        body["errors"] = errors
    else:
        body["data"] = {
            "issues": {
                "nodes": nodes or [],
                "pageInfo": {"hasNextPage": has_next_page, "endCursor": end_cursor},
            }
        }
    resp.json.return_value = body
    resp.text = str(body)
    resp.headers = headers or {}
    return resp


class TestLinearConnectorValidateConfig:
    def test_missing_all_fields_raises(self) -> None:
        connector = LinearConnector(LinearConnectorConfig())
        with pytest.raises(ValueError, match="TRELIX_LINEAR_API_KEY"):
            connector.validate_config()

    def test_missing_one_field_lists_only_that_field(self) -> None:
        config = LinearConnectorConfig(api_key="key123", team_key=None)
        connector = LinearConnector(config)
        with pytest.raises(ValueError, match="TRELIX_LINEAR_TEAM_KEY"):
            connector.validate_config()

    def test_all_fields_present_does_not_raise(self) -> None:
        LinearConnector(_LINEAR_CONFIG).validate_config()


class TestLinearConnectorFetch:
    def test_fetch_returns_artifacts_from_issues(self) -> None:
        resp = _mock_linear_page(
            nodes=[
                {
                    "identifier": "ENG-1",
                    "title": "Fix login bug",
                    "description": "Users cannot log in",
                    "url": "https://linear.app/acme/issue/ENG-1",
                    "priority": 2.0,
                    "state": {"id": "s1", "name": "In Progress", "type": "started"},
                    "team": {"id": "t1", "key": "ENG", "name": "Engineering"},
                    "assignee": {"id": "u1", "name": "Ada Lovelace", "email": "ada@example.com"},
                    "labels": {"nodes": [{"id": "l1", "name": "bug"}]},
                }
            ]
        )
        with patch("trelix.indexing.connectors.linear.httpx.post", return_value=resp):
            artifacts = LinearConnector(_LINEAR_CONFIG).fetch()

        assert len(artifacts) == 1
        a = artifacts[0]
        assert a.source_ref == "linear-issue:ENG-1"
        assert a.artifact_kind == "ticket"
        assert a.title == "Fix login bug"
        assert a.body == "Users cannot log in"
        assert a.url == "https://linear.app/acme/issue/ENG-1"
        assert a.metadata["status"] == "In Progress"
        assert a.metadata["team_key"] == "ENG"
        assert a.metadata["assignee"] == "Ada Lovelace"
        assert a.metadata["priority"] == "2.0"
        assert a.metadata["labels"] == "bug"

    def test_fetch_paginates_via_cursor(self) -> None:
        page1 = _mock_linear_page(
            nodes=[{"identifier": "ENG-1", "title": "One"}],
            has_next_page=True,
            end_cursor="cursor-2",
        )
        page2 = _mock_linear_page(nodes=[{"identifier": "ENG-2", "title": "Two"}])
        with patch(
            "trelix.indexing.connectors.linear.httpx.post", side_effect=[page1, page2]
        ) as mock_post:
            artifacts = LinearConnector(_LINEAR_CONFIG).fetch()

        assert {a.source_ref for a in artifacts} == {"linear-issue:ENG-1", "linear-issue:ENG-2"}
        assert mock_post.call_count == 2
        second_call_body = mock_post.call_args_list[1].kwargs["json"]
        assert second_call_body["variables"]["after"] == "cursor-2"

    def test_fetch_sends_explicit_team_filter_and_page_size(self) -> None:
        resp = _mock_linear_page()
        with patch("trelix.indexing.connectors.linear.httpx.post", return_value=resp) as mock_post:
            LinearConnector(_LINEAR_CONFIG).fetch()

        first_call_body = mock_post.call_args_list[0].kwargs["json"]
        assert first_call_body["variables"] == {"teamKey": "ENG", "first": 100, "after": None}

    def test_fetch_401_raises_linear_connector_error(self) -> None:
        resp = _mock_linear_page(status_code=401, errors=[{"extensions": {"code": "AUTH"}}])
        with patch("trelix.indexing.connectors.linear.httpx.post", return_value=resp):
            with pytest.raises(LinearConnectorError, match="401"):
                LinearConnector(_LINEAR_CONFIG).fetch()

    def test_fetch_ratelimited_400_retries_then_succeeds(self) -> None:
        rate_limited = _mock_linear_page(
            status_code=400,
            errors=[{"extensions": {"code": "RATELIMITED"}}],
            headers={"X-RateLimit-Requests-Reset": "0"},
        )
        success = _mock_linear_page()
        with (
            patch(
                "trelix.indexing.connectors.linear.httpx.post",
                side_effect=[rate_limited, success],
            ) as mock_post,
            patch("trelix.indexing.connectors.linear.time.sleep"),
        ):
            artifacts = LinearConnector(_LINEAR_CONFIG).fetch()

        assert artifacts == []
        assert mock_post.call_count == 2

    def test_fetch_ratelimited_400_exhausts_retries_and_raises(self) -> None:
        always_rate_limited = _mock_linear_page(
            status_code=400,
            errors=[{"extensions": {"code": "RATELIMITED"}}],
            headers={"X-RateLimit-Requests-Reset": "0"},
        )
        with (
            patch(
                "trelix.indexing.connectors.linear.httpx.post",
                return_value=always_rate_limited,
            ) as mock_post,
            patch("trelix.indexing.connectors.linear.time.sleep"),
        ):
            with pytest.raises(LinearConnectorError, match="RATELIMITED"):
                LinearConnector(_LINEAR_CONFIG).fetch()

        assert mock_post.call_count == 3

    def test_fetch_ratelimited_400_computes_wait_from_reset_headers(self) -> None:
        future_reset_ms = (time.time() + 42) * 1000
        rate_limited = _mock_linear_page(
            status_code=400,
            errors=[{"extensions": {"code": "RATELIMITED"}}],
            headers={"X-RateLimit-Requests-Reset": str(future_reset_ms)},
        )
        success = _mock_linear_page()
        with (
            patch(
                "trelix.indexing.connectors.linear.httpx.post",
                side_effect=[rate_limited, success],
            ),
            patch("trelix.indexing.connectors.linear.time.sleep") as mock_sleep,
        ):
            LinearConnector(_LINEAR_CONFIG).fetch()

        mock_sleep.assert_called_once()
        waited = mock_sleep.call_args[0][0]
        assert 35 <= waited <= 42

    def test_fetch_400_non_ratelimited_error_raises_immediately_without_retry(self) -> None:
        input_error = _mock_linear_page(
            status_code=400,
            errors=[{"extensions": {"code": "INPUT_ERROR"}, "message": "too complex"}],
        )
        with (
            patch(
                "trelix.indexing.connectors.linear.httpx.post", return_value=input_error
            ) as mock_post,
            patch("trelix.indexing.connectors.linear.time.sleep") as mock_sleep,
        ):
            with pytest.raises(LinearConnectorError, match="too complex"):
                LinearConnector(_LINEAR_CONFIG).fetch()

        assert mock_post.call_count == 1
        mock_sleep.assert_not_called()

    def test_fetch_retries_on_5xx_via_shared_with_retry_then_succeeds(self) -> None:
        server_error = MagicMock(status_code=500, text="Internal Server Error")
        success = _mock_linear_page()
        with (
            patch(
                "trelix.indexing.connectors.linear.httpx.post",
                side_effect=[server_error, success],
            ),
            patch("tenacity.nap.time.sleep"),
        ):
            artifacts = LinearConnector(_LINEAR_CONFIG).fetch()

        assert artifacts == []

    def test_fetch_exhausts_shared_retry_on_persistent_5xx_and_raises(self) -> None:
        server_error = MagicMock(status_code=500, text="Internal Server Error")
        with (
            patch("trelix.indexing.connectors.linear.httpx.post", return_value=server_error),
            patch("tenacity.nap.time.sleep"),
        ):
            with pytest.raises(LinearConnectorError):
                LinearConnector(_LINEAR_CONFIG).fetch()

    def test_fetch_network_error_never_raises_raw_httpx_exception(self) -> None:
        """A raw httpx exception must surface as LinearConnectorError, not
        leak the underlying exception type to callers."""
        import httpx

        with (
            patch(
                "trelix.indexing.connectors.linear.httpx.post",
                side_effect=httpx.ConnectError("connection refused"),
            ),
            patch("tenacity.nap.time.sleep"),
        ):
            with pytest.raises(LinearConnectorError):
                LinearConnector(_LINEAR_CONFIG).fetch()

    def test_fetch_handles_missing_optional_fields_gracefully(self) -> None:
        """An issue with no description/state/team/assignee/labels keys at
        all must not raise — body/metadata just end up empty."""
        resp = _mock_linear_page(nodes=[{"identifier": "ENG-5", "title": "Bare issue"}])
        with patch("trelix.indexing.connectors.linear.httpx.post", return_value=resp):
            artifacts = LinearConnector(_LINEAR_CONFIG).fetch()

        a = artifacts[0]
        assert a.body == ""
        assert a.metadata["status"] == ""
        assert a.metadata["team_key"] == ""
        assert a.metadata["assignee"] == ""
        assert a.metadata["priority"] == ""
        assert a.metadata["labels"] == ""

    def test_fetch_maps_zero_priority_correctly(self) -> None:
        """priority=0.0 ('No priority' in Linear's scale) must be preserved
        as "0.0", not blanked to "" by a falsy check."""
        resp = _mock_linear_page(
            nodes=[{"identifier": "ENG-6", "title": "No priority", "priority": 0.0}]
        )
        with patch("trelix.indexing.connectors.linear.httpx.post", return_value=resp):
            artifacts = LinearConnector(_LINEAR_CONFIG).fetch()

        assert artifacts[0].metadata["priority"] == "0.0"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestConnectorRegistry:
    def test_jira_resolves_to_jira_connector(self) -> None:
        assert isinstance(get_artifact_source("jira"), JiraConnector)

    def test_testrail_resolves_to_testrail_connector(self) -> None:
        assert isinstance(get_artifact_source("testrail"), TestRailConnector)

    def test_xray_resolves_to_xray_connector(self) -> None:
        assert isinstance(get_artifact_source("xray"), XrayConnector)

    def test_linear_resolves_to_linear_connector(self) -> None:
        assert isinstance(get_artifact_source("linear"), LinearConnector)

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
