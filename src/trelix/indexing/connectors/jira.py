"""
Jira Cloud REST API connector — fetches issues for a configured project as
Artifacts, for cross-source linking via generic_edges.

Auth: HTTP Basic (email + API token), matching Jira Cloud's own recommended
scheme for personal/service integrations — no OAuth needed.
Pagination: Jira's v3 search endpoint uses a cursor (nextPageToken), not
offset/limit.

Description text: Jira Cloud's v3 API always returns `description` as
Atlassian Document Format (ADF) — a nested JSON tree, never a plain string
(confirmed live against a real site: every issue, unconditionally). The
original version of this connector fell back to an empty body whenever
description wasn't a plain `str`, which meant every ticket's description
was silently dropped in practice — not a rare edge case. `_adf_to_text()`
below walks the tree and renders a plain-text approximation good enough for
ArtifactLinker's regex matching and for readable ticket bodies elsewhere;
it is not a full ADF renderer (no tables, no user/date mentions resolved
to names) but covers every node type observed on a real Jira Cloud site:
paragraphs, headings, bullet/ordered lists, code blocks, inline code,
blockquotes, panels, expand sections, rules, links, and embedded cards.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from trelix.core.config import JiraConnectorConfig
from trelix.core.models import Artifact
from trelix.core.retry import with_retry
from trelix.indexing.connectors.base import ArtifactSource

logger = logging.getLogger("trelix.indexing.connectors.jira")

_MAX_RETRIES = 5


class JiraConnectorError(Exception):
    """Raised on a real, non-retryable Jira API failure."""


# ADF node types whose children are inline text that flows together with no
# separator (concatenated directly) — as opposed to block-level children
# (paragraphs, list items, panels, ...) which are newline-separated.
_INLINE_CONTAINER_TYPES = frozenset({"paragraph", "heading", "codeBlock"})


def _adf_node_to_text(node: Any) -> str:
    """Render one ADF node (and its descendants) to a plain-text
    approximation. Unknown/unhandled node types fall through to rendering
    their `content` children as block-level (newline-joined) — safer than
    dropping an unrecognized node outright, since a future/undocumented
    node type is far more likely to still carry meaningful nested text
    than to be purely structural."""
    if not isinstance(node, dict):
        return ""
    node_type = node.get("type")
    content = node.get("content") or []

    if node_type == "text":
        text = str(node.get("text", ""))
        for mark in node.get("marks") or []:
            if mark.get("type") == "link":
                href = (mark.get("attrs") or {}).get("href")
                if href:
                    text = f"{text} ({href})"
        return text

    if node_type == "hardBreak":
        return "\n"

    if node_type in ("embedCard", "inlineCard"):
        return str((node.get("attrs") or {}).get("url", ""))

    if node_type == "mention":
        return str((node.get("attrs") or {}).get("text", ""))

    if node_type == "status":
        return str((node.get("attrs") or {}).get("text", ""))

    if node_type == "emoji":
        attrs = node.get("attrs") or {}
        return str(attrs.get("text") or attrs.get("shortName", ""))

    if node_type == "rule":
        return ""

    if node_type in ("bulletList", "orderedList", "taskList", "decisionList"):
        lines = []
        for i, item in enumerate(content, start=1):
            item_text = _adf_node_to_text(item).strip()
            if not item_text:
                continue
            prefix = f"{i}. " if node_type == "orderedList" else "- "
            lines.append(f"{prefix}{item_text}")
        return "\n".join(lines)

    if node_type == "expand":
        title = (node.get("attrs") or {}).get("title", "")
        body = _join_block_children(content)
        return f"{title}:\n{body}" if title else body

    if node_type in _INLINE_CONTAINER_TYPES:
        return "".join(_adf_node_to_text(child) for child in content)

    # doc, blockquote, panel, listItem, table/tableRow/tableCell, or any
    # unrecognized future node type: treat children as block-level.
    return _join_block_children(content)


def _join_block_children(content: list[Any]) -> str:
    parts = (_adf_node_to_text(child).strip() for child in content)
    return "\n".join(part for part in parts if part)


def _adf_to_text(doc: dict[str, Any]) -> str:
    """Entry point: render a full ADF document (the `description` field's
    value) to plain text."""
    return _adf_node_to_text(doc).strip()


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
            raise ValueError(f"JiraConnector is missing required config: {', '.join(missing)}")

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

            try:
                data = self._get(f"{base_url}/rest/api/3/search/jql", params=params, auth=auth)
            except httpx.HTTPError as exc:
                raise JiraConnectorError(f"Jira API request failed: {exc}") from exc
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
        if isinstance(description, str):
            body = description
        elif isinstance(description, dict):
            body = _adf_to_text(description)
        else:
            body = ""
        return Artifact(
            source_ref=f"ticket:{key}",
            artifact_kind="ticket",
            title=fields.get("summary", ""),
            body=body,
            url=f"{base_url}/browse/{key}",
            metadata={"status": (fields.get("status") or {}).get("name", "")},
        )

    @with_retry(max_attempts=_MAX_RETRIES)
    def _get(self, url: str, *, params: dict[str, Any], auth: tuple[str, str]) -> dict[str, Any]:
        """GET with the shared retry contract's full-jitter exponential
        backoff on 429/5xx and connection-level errors. 401 is raised
        immediately (not retryable — a bad credential never becomes valid
        by waiting). Raises on a real failure (auth error, exhausted
        retries) — a connector failure means real content is missing, so it
        must not be silently swallowed the way GitLinker's git-command
        failures are."""
        response = httpx.get(url, params=params, auth=auth, timeout=30)
        if response.status_code == 401:
            raise JiraConnectorError(
                "401 Unauthorized — check TRELIX_JIRA_EMAIL/TRELIX_JIRA_API_TOKEN"
            )
        if response.status_code not in (200, 201):
            raise httpx.HTTPStatusError(
                f"Jira API error {response.status_code}: {str(response.text)[:200]}",
                request=httpx.Request("GET", url),
                response=response,
            )
        return dict(response.json())
