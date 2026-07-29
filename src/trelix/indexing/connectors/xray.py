"""
Xray Cloud connector — fetches tests for a configured Jira project as
Artifacts, for cross-source linking via generic_edges.

Cloud only (see XrayConnectorConfig's docstring for why Server/DC is out of
scope). Xray Cloud tests are Jira issues under the hood, but test-specific
content (steps/definition) has no Jira-native equivalent — this connector
combines two API calls per page: a plain Jira REST v3 search (title/url,
same pattern JiraConnector already uses) and Xray's own GraphQL `getTests`
query (the one genuinely new API surface, for step/definition content).

Auth: Xray Cloud's own API-key flow — POST /api/v2/authenticate with
{client_id, client_secret} (issued by a Jira admin in Xray's global
settings, distinct from a user's Jira API token) returns a short-lived
bearer JWT. Every subsequent Xray REST/GraphQL call carries
Authorization: Bearer <token>. The token request itself is wrapped in the
shared retry contract — a 429 on /authenticate should retry, not fail the
whole sync.

Pagination: Xray Cloud's GraphQL uses a limit/start cursor (confirmed
different from TestRail's offset-based one, though numerically equivalent —
kept as a distinct field name to match Xray's own API vocabulary).
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from trelix.core.config import XrayConnectorConfig
from trelix.core.models import Artifact
from trelix.core.retry import with_retry
from trelix.indexing.connectors.base import ArtifactSource

logger = logging.getLogger("trelix.indexing.connectors.xray")

_MAX_RETRIES = 5
_AUTHENTICATE_URL = "https://xray.cloud.getxray.app/api/v2/authenticate"
_GRAPHQL_URL = "https://xray.cloud.getxray.app/api/v2/graphql"

_GET_TESTS_QUERY = """
query GetTests($jql: String!, $limit: Int!, $start: Int!) {
  getTests(jql: $jql, limit: $limit, start: $start) {
    total
    start
    limit
    results {
      issueId
      jira(fields: ["key", "summary", "status"])
      testType { name }
      steps { action data result }
      unstructured
    }
  }
}
"""


class XrayConnectorError(Exception):
    """Raised on a real, non-retryable Xray API failure."""


class XrayConnector(ArtifactSource):
    def __init__(self, config: XrayConnectorConfig | None = None) -> None:
        self._config = config or XrayConnectorConfig()
        self._token: str | None = None

    def validate_config(self) -> None:
        missing = [
            name
            for name, val in (
                ("TRELIX_XRAY_CLIENT_ID", self._config.client_id),
                ("TRELIX_XRAY_CLIENT_SECRET", self._config.client_secret),
                ("TRELIX_XRAY_PROJECT_KEY", self._config.project_key),
                ("TRELIX_XRAY_JIRA_BASE_URL", self._config.jira_base_url),
            )
            if not val
        ]
        if missing:
            raise ValueError(f"XrayConnector is missing required config: {', '.join(missing)}")

    def fetch(self) -> list[Artifact]:
        self.validate_config()
        assert self._config.client_id and self._config.client_secret
        assert self._config.project_key and self._config.jira_base_url

        self._token = self._authenticate()
        jira_base_url = self._config.jira_base_url.rstrip("/")
        jql = f"project = {self._config.project_key}"
        artifacts: list[Artifact] = []
        start = 0

        while True:
            try:
                data = self._get_tests(jql, self._config.page_size, start)
            except httpx.HTTPError as exc:
                raise XrayConnectorError(f"Xray API request failed: {exc}") from exc
            page = data.get("data", {}).get("getTests", {})
            results = page.get("results", [])
            for test in results:
                artifacts.append(self._test_to_artifact(test, jira_base_url))

            if len(results) < self._config.page_size:
                break
            start += self._config.page_size

        return artifacts

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _test_to_artifact(self, test: dict[str, Any], jira_base_url: str) -> Artifact:
        jira_fields = test.get("jira") or {}
        key = jira_fields.get("key", "")
        title = jira_fields.get("summary", "")
        status = jira_fields.get("status") or {}
        status_name = status.get("name", "") if isinstance(status, dict) else str(status)

        steps = test.get("steps") or []
        step_lines = [
            "\n".join(part for part in (s.get("action"), s.get("data"), s.get("result")) if part)
            for s in steps
        ]
        unstructured = test.get("unstructured") or ""
        body = "\n\n".join(part for part in (unstructured, *step_lines) if part)

        return Artifact(
            source_ref=f"xray-test:{key}",
            artifact_kind="test_case",
            title=title,
            body=body,
            url=f"{jira_base_url}/browse/{key}" if key else None,
            metadata={
                "status": status_name,
                "test_type": (test.get("testType") or {}).get("name", ""),
            },
        )

    @with_retry(max_attempts=_MAX_RETRIES)
    def _authenticate(self) -> str:
        """POST client_id/client_secret, get back a short-lived bearer JWT.
        Wrapped in the shared retry contract — a 429 here should retry, not
        fail the whole sync before it even starts."""
        response = httpx.post(
            _AUTHENTICATE_URL,
            json={"client_id": self._config.client_id, "client_secret": self._config.client_secret},
            timeout=30,
        )
        if response.status_code == 401:
            raise XrayConnectorError(
                "401 Unauthorized — check TRELIX_XRAY_CLIENT_ID/TRELIX_XRAY_CLIENT_SECRET"
            )
        if response.status_code not in (200, 201):
            raise httpx.HTTPStatusError(
                f"Xray authenticate error {response.status_code}: {str(response.text)[:200]}",
                request=httpx.Request("POST", _AUTHENTICATE_URL),
                response=response,
            )
        # Xray's authenticate endpoint returns the JWT as a raw JSON string
        # (e.g. '"eyJhbGci..."'), not a {"token": ...} object.
        token = response.json()
        return str(token)

    @with_retry(max_attempts=_MAX_RETRIES)
    def _get_tests(self, jql: str, limit: int, start: int) -> dict[str, Any]:
        """GraphQL getTests query with the shared retry contract's
        full-jitter exponential backoff on 429/5xx and connection-level
        errors. Same posture as Jira/TestRail — a connector failure means
        real content is missing, so it raises rather than being silently
        swallowed."""
        response = httpx.post(
            _GRAPHQL_URL,
            json={
                "query": _GET_TESTS_QUERY,
                "variables": {"jql": jql, "limit": limit, "start": start},
            },
            headers={"Authorization": f"Bearer {self._token}"},
            timeout=30,
        )
        if response.status_code == 401:
            raise XrayConnectorError("401 Unauthorized — Xray bearer token expired or invalid")
        if response.status_code not in (200, 201):
            raise httpx.HTTPStatusError(
                f"Xray GraphQL error {response.status_code}: {str(response.text)[:200]}",
                request=httpx.Request("POST", _GRAPHQL_URL),
                response=response,
            )
        return dict(response.json())
