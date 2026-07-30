"""
Linear connector — fetches issues for a configured team as Artifacts, for
cross-source linking via generic_edges.

Transport: Linear's API is GraphQL-only, a single POST endpoint
(https://api.linear.app/graphql). No official Python SDK exists (only a
TypeScript one, @linear/sdk) — hand-rolled httpx-over-GraphQL matches the
Xray connector's own precedent and is validated by prior art (e.g.
SurfSense's OSS connector does exactly this).

Auth: **personal API key sent as `Authorization: <API_KEY>` — no "Bearer"
prefix.** This is the one detail most likely to get "corrected" back to
`Bearer <API_KEY>` by someone skimming Xray's connector for comparison —
don't. Linear's OAuth2 tokens use `Bearer`; personal API keys explicitly do
not (per Linear's own docs, verified live against the API: an unauthenticated
request returns a 401 with extensions.code == "AUTHENTICATION_ERROR").

Pagination: Relay-style cursor pagination (`first`/`after` args,
`pageInfo { hasNextPage, endCursor }`). Unlike Xray's limit/start or Jira/
TestRail's page-size-based termination, Linear's loop must terminate on
`hasNextPage`, not on `len(results) < page_size` — a page can legitimately
come back short of `first` while more pages remain.

Query complexity: Linear enforces a 10,000-point-per-query hard cap (points:
0.1/scalar property, 1/object, a connection multiplies its children's cost
by its own `first` argument, or 50 if omitted). This connector's field
selection costs ~7.7 points/issue; at page_size=100 that's ~1 + 770 ≈ 771
points — comfortably under the cap, with headroom for schema drift. Always
pass `first` explicitly (never rely on the default of 50) — this applies to
both the top-level `issues` connection and the nested `labels` connection,
which has the same default-50 risk.

Rate limiting is a two-layer concern here, deliberately NOT folded into the
shared core/retry.with_retry() contract:
  - Generic 5xx/connection failures still go through @with_retry exactly
    like every other connector (Jira/TestRail/Xray) and LLM backend.
  - Linear signals rate-limiting differently from everything else in this
    codebase: HTTP 400 (not 429), a GraphQL body error with
    extensions.code == "RATELIMITED", and no Retry-After header — instead,
    reset timing comes from X-RateLimit-Requests-Reset /
    X-RateLimit-Complexity-Reset response headers (epoch milliseconds).
    core/retry.py's is_retryable_http_error() only recognizes status codes
    {429,500,502,503,504} and a Retry-After header, so it cannot and must
    not be taught Linear-specific GraphQL-body semantics — that would
    couple a shared contract used by every connector/LLM backend/embedder
    to one connector's response shape. Instead, _fetch_issues_page() below
    implements a small, connector-local bounded retry loop that inspects
    the body directly. A non-RATELIMITED 400 (e.g. extensions.code ==
    "INPUT_ERROR" from exceeding the complexity cap) is NOT retried — that
    indicates a real query/config problem, not a transient condition.

Two research uncertainties, left as open risk rather than silently resolved:
  - Linear's own rate-limiting docs give two different numbers for the
    personal-API-key request-count limit (2,500 in a table, 5,000 in prose)
    — this connector doesn't hard-code either figure, so the discrepancy is
    harmless here, but it's why _fetch_issues_page() derives its wait time
    from the response headers rather than a hardcoded budget.
  - The general claim "Linear's GraphQL errors always arrive as HTTP 200 +
    an errors array" did NOT survive adversarial verification during
    research — only the RATELIMITED-specific 400 case is confirmed. This
    connector defensively checks for a body-level `errors` array on
    nominally-2xx responses too (mirroring Xray's own `data.get("errors")`
    check), in case a real error surfaces on a 200.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from trelix.core.config import LinearConnectorConfig
from trelix.core.models import Artifact
from trelix.core.retry import with_retry
from trelix.indexing.connectors.base import ArtifactSource

logger = logging.getLogger("trelix.indexing.connectors.linear")

_MAX_RETRIES = 5
_LINEAR_GRAPHQL_URL = "https://api.linear.app/graphql"

# Connector-local rate-limit retry loop — deliberately separate from
# @with_retry's shared contract (see module docstring).
_RATELIMIT_MAX_ATTEMPTS = 3
_RATELIMIT_FALLBACK_WAIT_SECONDS = 5.0
_RATELIMIT_MAX_WAIT_SECONDS = 60.0

# Nested labels connection has the same "50 if first is omitted" default
# risk as the top-level issues connection — always pass first explicitly.
_LABELS_PAGE_SIZE = 10

_GET_ISSUES_QUERY = f"""
query GetIssues($teamKey: String!, $first: Int!, $after: String) {{
  issues(first: $first, after: $after, filter: {{ team: {{ key: {{ eq: $teamKey }} }} }}) {{
    nodes {{
      id
      identifier
      title
      description
      url
      updatedAt
      createdAt
      priority
      state {{ id name type }}
      team {{ id key name }}
      assignee {{ id name email }}
      labels(first: {_LABELS_PAGE_SIZE}) {{
        nodes {{ id name }}
      }}
    }}
    pageInfo {{ hasNextPage endCursor }}
  }}
}}
"""


class LinearConnectorError(Exception):
    """Raised on a real, non-retryable Linear API failure."""


class LinearConnector(ArtifactSource):
    def __init__(self, config: LinearConnectorConfig | None = None) -> None:
        self._config = config or LinearConnectorConfig()

    def validate_config(self) -> None:
        missing = [
            name
            for name, val in (
                ("TRELIX_LINEAR_API_KEY", self._config.api_key),
                ("TRELIX_LINEAR_TEAM_KEY", self._config.team_key),
            )
            if not val
        ]
        if missing:
            raise ValueError(f"LinearConnector is missing required config: {', '.join(missing)}")

    def fetch(self) -> list[Artifact]:
        self.validate_config()
        assert self._config.api_key and self._config.team_key

        artifacts: list[Artifact] = []
        after: str | None = None

        while True:
            try:
                data = self._fetch_issues_page(after)
            except httpx.HTTPError as exc:
                raise LinearConnectorError(f"Linear API request failed: {exc}") from exc
            page = (data.get("data") or {}).get("issues") or {}
            nodes = page.get("nodes", [])
            for issue in nodes:
                artifacts.append(self._issue_to_artifact(issue))

            page_info = page.get("pageInfo") or {}
            if not page_info.get("hasNextPage"):
                break
            after = page_info.get("endCursor")
            if not after:
                # Defensive: hasNextPage=True with no cursor would loop
                # forever re-requesting the same page.
                break

        return artifacts

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _issue_to_artifact(self, issue: dict[str, Any]) -> Artifact:
        identifier = issue.get("identifier", "")
        state = issue.get("state") or {}
        team = issue.get("team") or {}
        assignee = issue.get("assignee") or {}
        labels = [
            label.get("name", "")
            for label in (issue.get("labels") or {}).get("nodes", [])
            if label.get("name")
        ]
        priority = issue.get("priority")

        return Artifact(
            # "linear-issue:" rather than reusing Jira's "ticket:" prefix —
            # a Jira project key and a Linear team key could coincide (both
            # "ENG"), and upsert_artifact() keys on source_ref UNIQUE, so
            # sharing a prefix risks a silent cross-source collision.
            source_ref=f"linear-issue:{identifier}",
            artifact_kind="ticket",
            title=issue.get("title", ""),
            # Linear's description is plain markdown (confirmed via schema
            # introspection), unlike Jira's ADF — no stripping needed.
            body=issue.get("description") or "",
            url=issue.get("url"),
            metadata={
                "status": state.get("name", ""),
                "team_key": team.get("key", ""),
                "assignee": assignee.get("name", "") if assignee else "",
                # priority's scale includes 0 == "No priority" as a real
                # value — `is not None`, not a falsy check, or a genuine
                # 0.0 would wrongly become "".
                "priority": str(priority) if priority is not None else "",
                "labels": ",".join(labels),
            },
        )

    @with_retry(max_attempts=_MAX_RETRIES)
    def _post_graphql(self, variables: dict[str, Any]) -> httpx.Response:
        """POST the issues query. 401 fails fast; 5xx/connection errors
        raise httpx.HTTPStatusError/httpx.HTTPError for the shared
        @with_retry contract to classify and retry. HTTP 400 is returned
        as-is (not raised) — the caller must inspect the body to tell a
        retryable RATELIMITED error apart from a fatal one (e.g.
        INPUT_ERROR from exceeding the complexity cap)."""
        response = httpx.post(
            _LINEAR_GRAPHQL_URL,
            json={"query": _GET_ISSUES_QUERY, "variables": variables},
            headers={"Authorization": self._config.api_key or ""},
            timeout=30,
        )
        if response.status_code == 401:
            raise LinearConnectorError("401 Unauthorized — check TRELIX_LINEAR_API_KEY")
        if response.status_code == 400:
            return response
        if response.status_code not in (200, 201):
            raise httpx.HTTPStatusError(
                f"Linear GraphQL error {response.status_code}: {str(response.text)[:200]}",
                request=httpx.Request("POST", _LINEAR_GRAPHQL_URL),
                response=response,
            )
        return response

    def _fetch_issues_page(self, after: str | None) -> dict[str, Any]:
        """Connector-local bounded retry loop for Linear's RATELIMITED
        signal (HTTP 400 + GraphQL error code) — see module docstring for
        why this is intentionally not part of the shared retry contract.
        Not @with_retry-decorated itself; it wraps a call that already is.
        """
        variables = {
            "teamKey": self._config.team_key,
            "first": self._config.page_size,
            "after": after,
        }
        for attempt in range(1, _RATELIMIT_MAX_ATTEMPTS + 1):
            response = self._post_graphql(variables)
            body = response.json()

            if response.status_code == 400:
                if _is_rate_limited_error(body):
                    if attempt == _RATELIMIT_MAX_ATTEMPTS:
                        raise LinearConnectorError(
                            f"Linear GraphQL rate limit exceeded after {attempt} attempts "
                            "(RATELIMITED)"
                        )
                    wait_seconds = _compute_rate_limit_wait_seconds(response.headers)
                    logger.warning(
                        "Linear rate-limited, sleeping %.1fs (attempt %d/%d)",
                        wait_seconds,
                        attempt,
                        _RATELIMIT_MAX_ATTEMPTS,
                    )
                    time.sleep(wait_seconds)
                    continue
                raise LinearConnectorError(f"Linear GraphQL error: {body.get('errors')}")

            # Defensive: the "errors always come with HTTP 200" claim was
            # adversarially refuted during research — this branch exists
            # in case a real error surfaces on an otherwise-2xx response.
            if body.get("errors"):
                raise LinearConnectorError(f"Linear GraphQL error: {body['errors']}")
            return dict(body)

        raise LinearConnectorError("Linear GraphQL request failed: exhausted rate-limit retries")


def _is_rate_limited_error(body: dict[str, Any]) -> bool:
    errors = body.get("errors") or []
    return any(
        isinstance(err, dict) and (err.get("extensions") or {}).get("code") == "RATELIMITED"
        for err in errors
    )


def _compute_rate_limit_wait_seconds(headers: httpx.Headers) -> float:
    """RATELIMITED doesn't say which of the two leaky buckets (request-count
    vs. complexity-budget) tripped, so take the max of whichever
    X-RateLimit-*-Reset headers parse — waiting for the later of the two is
    always safe. Falls back to a fixed wait if neither header is present or
    parseable (defensive — docs say they're always present, but that's
    exactly the kind of claim this module doesn't trust unconditionally)."""
    candidates: list[float] = []
    for header_name in ("X-RateLimit-Requests-Reset", "X-RateLimit-Complexity-Reset"):
        raw = headers.get(header_name)
        if raw is None:
            continue
        try:
            reset_ms = float(raw)
        except ValueError:
            continue
        wait = (reset_ms - time.time() * 1000) / 1000
        candidates.append(wait)

    if not candidates:
        return _RATELIMIT_FALLBACK_WAIT_SECONDS

    wait_seconds = max(candidates)
    return max(0.0, min(wait_seconds, _RATELIMIT_MAX_WAIT_SECONDS))
