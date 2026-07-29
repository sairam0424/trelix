"""
TestRail REST API connector — fetches test cases for a configured project
as Artifacts, for cross-source linking via generic_edges.

Auth: HTTP Basic (username + API key).
Pagination: offset/limit, TestRail's own max page size is 250.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from trelix.core.config import TestRailConnectorConfig
from trelix.core.models import Artifact
from trelix.core.retry import with_retry
from trelix.indexing.connectors.base import ArtifactSource

logger = logging.getLogger("trelix.indexing.connectors.testrail")

_MAX_RETRIES = 5


class TestRailConnectorError(Exception):
    """Raised on a real, non-retryable TestRail API failure."""


class TestRailConnector(ArtifactSource):
    def __init__(self, config: TestRailConnectorConfig | None = None) -> None:
        self._config = config or TestRailConnectorConfig()

    def validate_config(self) -> None:
        missing = [
            name
            for name, val in (
                ("TRELIX_TESTRAIL_BASE_URL", self._config.base_url),
                ("TRELIX_TESTRAIL_USERNAME", self._config.username),
                ("TRELIX_TESTRAIL_API_KEY", self._config.api_key),
                ("TRELIX_TESTRAIL_PROJECT_ID", self._config.project_id),
            )
            if not val
        ]
        if missing:
            raise ValueError(f"TestRailConnector is missing required config: {', '.join(missing)}")

    def fetch(self) -> list[Artifact]:
        self.validate_config()
        assert self._config.base_url and self._config.username
        assert self._config.api_key and self._config.project_id is not None

        base_url = self._config.base_url.rstrip("/")
        auth = (self._config.username, self._config.api_key)
        artifacts: list[Artifact] = []
        offset = 0

        while True:
            try:
                data = self._get(
                    f"{base_url}/index.php?/api/v2/get_cases/{self._config.project_id}",
                    params={"limit": self._config.page_size, "offset": offset},
                    auth=auth,
                )
            except httpx.HTTPError as exc:
                raise TestRailConnectorError(f"TestRail API request failed: {exc}") from exc
            cases = data.get("cases", [])
            for case in cases:
                artifacts.append(self._case_to_artifact(case, base_url))

            if len(cases) < self._config.page_size:
                break
            offset += self._config.page_size

        return artifacts

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _case_to_artifact(self, case: dict[str, Any], base_url: str) -> Artifact:
        case_id = case["id"]
        preconditions = case.get("custom_preconds") or ""
        steps = case.get("custom_steps") or ""
        body = "\n\n".join(part for part in (preconditions, steps) if part)
        return Artifact(
            source_ref=f"test_case:{case_id}",
            artifact_kind="test_case",
            title=case.get("title", ""),
            body=body,
            url=f"{base_url}/index.php?/cases/view/{case_id}",
            metadata={"priority_id": str(case.get("priority_id", ""))},
        )

    @with_retry(max_attempts=_MAX_RETRIES)
    def _get(self, url: str, *, params: dict[str, Any], auth: tuple[str, str]) -> dict[str, Any]:
        """GET with the shared retry contract's full-jitter exponential
        backoff on 429/5xx and connection-level errors. Same posture as
        JiraConnector — a connector failure means real content is missing,
        so it raises rather than being silently swallowed."""
        response = httpx.get(url, params=params, auth=auth, timeout=30)
        if response.status_code == 401:
            raise TestRailConnectorError(
                "401 Unauthorized — check TRELIX_TESTRAIL_USERNAME/TRELIX_TESTRAIL_API_KEY"
            )
        if response.status_code not in (200, 201):
            raise httpx.HTTPStatusError(
                f"TestRail API error {response.status_code}: {str(response.text)[:200]}",
                request=httpx.Request("GET", url),
                response=response,
            )
        return dict(response.json())
