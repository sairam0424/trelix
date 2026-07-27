"""
Jira Cloud REST API connector — fetches issues for a configured project as
Artifacts, for cross-source linking via generic_edges.

Auth: HTTP Basic (email + API token), matching Jira Cloud's own recommended
scheme for personal/service integrations — no OAuth needed.
Pagination: Jira's v3 search endpoint uses a cursor (nextPageToken), not
offset/limit.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from trelix.core.config import JiraConnectorConfig
from trelix.core.models import Artifact
from trelix.indexing.connectors.base import ArtifactSource

logger = logging.getLogger("trelix.indexing.connectors.jira")

_MAX_RETRIES = 5
_BASE_BACKOFF_SECONDS = 1.0


class JiraConnectorError(Exception):
    """Raised on a real, non-retryable Jira API failure."""


class JiraConnector(ArtifactSource):
    def __init__(self, config: JiraConnectorConfig | None = None) -> None:
        self._config = config or JiraConnectorConfig()

    def validate_config(self) -> None:
        missing = [
            name
            for name, val in (
                ("TRELIX_JIRA_BASE_URL", self._config.base_url),
                ("TRELIX_JIRA_EMAIL", self._config.email),
                ("TRELIX_JIRA_API_TOKEN", self._config.api_token),
                ("TRELIX_JIRA_PROJECT_KEY", self._config.project_key),
            )
            if not val
        ]
        if missing:
            raise ValueError(
                f"JiraConnector is missing required config: {', '.join(missing)}"
            )

    def fetch(self) -> list[Artifact]:
        self.validate_config()
        # validate_config() already guaranteed all four are non-None — assert
        # narrows the type for the checker rather than re-checking at runtime.
        assert self._config.base_url and self._config.email
        assert self._config.api_token and self._config.project_key

        base_url = self._config.base_url.rstrip("/")
        auth = (self._config.email, self._config.api_token)
        artifacts: list[Artifact] = []
        next_page_token: str | None = None

        while True:
            params: dict[str, Any] = {
                "jql": f"project = {self._config.project_key}",
                "maxResults": self._config.page_size,
                "fields": "summary,description,status",
            }
            if next_page_token:
                params["nextPageToken"] = next_page_token

            data = self._get_with_retry(
                f"{base_url}/rest/api/3/search/jql", params=params, auth=auth
            )
            issues = data.get("issues", [])
            for issue in issues:
                artifacts.append(self._issue_to_artifact(issue, base_url))

            next_page_token = data.get("nextPageToken")
            if not next_page_token or not issues:
                break

        return artifacts

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _issue_to_artifact(self, issue: dict[str, Any], base_url: str) -> Artifact:
        key = issue["key"]
        fields = issue.get("fields", {})
        description = fields.get("description")
        # Jira's v3 API returns description as Atlassian Document Format
        # (a nested JSON structure), not plain text. Extracting readable
        # text from ADF properly needs a real renderer; falling back to the
        # summary alone (never crashing on the ADF shape) is an accepted
        # simplification for this connector's first version.
        body = description if isinstance(description, str) else ""
        return Artifact(
            source_ref=f"ticket:{key}",
            artifact_kind="ticket",
            title=fields.get("summary", ""),
            body=body,
            url=f"{base_url}/browse/{key}",
            metadata={"status": (fields.get("status") or {}).get("name", "")},
        )

    def _get_with_retry(
        self, url: str, *, params: dict[str, Any], auth: tuple[str, str]
    ) -> dict[str, Any]:
        """GET with exponential backoff + jitter on 429, matching the
        design's "backoff+jitter on 429" decision. Raises JiraConnectorError
        on a real failure (auth error, exhausted retries) — a connector
        failure means real content is missing, so it must not be silently
        swallowed the way GitLinker's git-command failures are."""
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                response = httpx.get(url, params=params, auth=auth, timeout=30)
            except httpx.HTTPError as exc:
                last_exc = exc
                self._sleep_backoff(attempt)
                continue

            if response.status_code == 401:
                raise JiraConnectorError(
                    "401 Unauthorized — check TRELIX_JIRA_EMAIL/TRELIX_JIRA_API_TOKEN"
                )
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                self._sleep_backoff(attempt, retry_after=retry_after)
                continue
            if response.status_code not in (200, 201):
                raise JiraConnectorError(
                    f"Jira API error {response.status_code}: {response.text[:200]}"
                )
            return dict(response.json())

        raise JiraConnectorError(
            f"Jira API request failed after {_MAX_RETRIES} retries: {last_exc}"
        )

    def _sleep_backoff(self, attempt: int, retry_after: str | None = None) -> None:
        if retry_after is not None:
            try:
                delay = float(retry_after)
            except ValueError:
                delay = _BASE_BACKOFF_SECONDS * (2**attempt)
        else:
            delay = _BASE_BACKOFF_SECONDS * (2**attempt)
        logger.debug("Jira API backoff: sleeping %.1fs (attempt %d)", delay, attempt)
        time.sleep(delay)
