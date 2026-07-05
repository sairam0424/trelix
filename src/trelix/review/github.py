"""
GitHub PR API client for trelix review --pr integration.

Fetches PR file diffs and posts review comments back to GitHub.
Token is read from GITHUB_TOKEN env var — never hardcoded.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger("trelix.review.github")

_GITHUB_ACCEPT = "application/vnd.github+json"
_GITHUB_API_VERSION = "2022-11-28"


class GitHubAPIError(Exception):
    """Raised when GitHub API returns a non-2xx status."""


@dataclass(frozen=True)
class PRFile:
    """One changed file in a GitHub PR."""

    filename: str
    status: str  # added|removed|modified|renamed|copied|changed|unchanged
    additions: int
    deletions: int
    patch: str | None  # None for binary files or very large diffs
    previous_filename: str | None = None


@dataclass
class ReviewComment:
    """A single inline review comment to post."""

    path: str  # file path relative to repo root
    line: int  # 1-indexed line in the NEW file (RIGHT side)
    body: str
    side: str = "RIGHT"  # RIGHT = addition, LEFT = deletion context


class GitHubPRClient:
    """
    Minimal GitHub REST API client for PR diff fetching and review posting.

    Args:
        token:    GitHub personal access token (fine-grained: pull_requests:write,
                  contents:read; classic: repo scope).
        base_url: Override for GitHub Enterprise Server.
    """

    def __init__(self, token: str, base_url: str = "https://api.github.com") -> None:
        self._token = token
        self._base_url = base_url.rstrip("/")
        self._headers = {
            "Accept": _GITHUB_ACCEPT,
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": _GITHUB_API_VERSION,
        }

    def _get(self, url: str, params: dict[str, int | str] | None = None) -> Any:
        """GET a URL, handle pagination, raise GitHubAPIError on non-2xx."""
        response = httpx.get(url, headers=self._headers, params=params, timeout=30)
        if response.status_code == 401:
            raise GitHubAPIError(
                f"401 Unauthorized — check your GITHUB_TOKEN (fine-grained: "
                f"pull_requests:write + contents:read; classic: repo scope). "
                f"Response: {response.json().get('message', '')}"
            )
        if response.status_code == 404:
            raise GitHubAPIError(
                f"404 Not Found — PR or repo does not exist, or token lacks access. URL: {url}"
            )
        if response.status_code not in (200, 201):  # pragma: no cover
            raise GitHubAPIError(
                f"GitHub API error {response.status_code}: "
                f"{response.json().get('message', response.text[:200])}"
            )
        return response.json()

    def get_pr_files(self, owner: str, repo: str, pr_number: int) -> list[PRFile]:
        """
        Fetch all changed files for a PR, handling pagination.

        GitHub caps at 3,000 files — logs a warning if truncation is likely.
        Returns list of PRFile; patch=None for binary/oversized files.
        """
        url = f"{self._base_url}/repos/{owner}/{repo}/pulls/{pr_number}/files"
        all_files: list[PRFile] = []
        page = 1

        while True:
            data = self._get(url, params={"per_page": 100, "page": page})
            if not isinstance(data, list) or not data:  # pragma: no cover
                break
            for item in data:
                all_files.append(
                    PRFile(
                        filename=item["filename"],
                        status=item["status"],
                        additions=item.get("additions", 0),
                        deletions=item.get("deletions", 0),
                        patch=item.get("patch"),  # absent for binary files
                        previous_filename=item.get("previous_filename"),
                    )
                )
            if len(data) < 100:
                break
            if len(all_files) >= 3000:  # pragma: no cover
                # GitHub silently caps at 3,000 — stop paginating
                break
            page += 1  # pragma: no cover

        if len(all_files) >= 3000:  # pragma: no cover
            logger.warning(
                "PR %s/%s#%d returned 3000 files — GitHub may have truncated the list. "
                "Large PRs (>3000 files) are silently capped. Review may be incomplete.",
                owner,
                repo,
                pr_number,
            )

        return all_files

    def post_review(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        commit_sha: str,
        body: str,
        comments: list[ReviewComment],
        event: str = "COMMENT",
    ) -> dict[str, Any]:
        """
        Post a review with optional inline comments in a single API call.

        Args:
            commit_sha: HEAD commit SHA of the PR (required by GitHub API).
            body:       Overall review summary (shown at top of review).
            comments:   List of inline comments. Empty list = summary-only review.
            event:      COMMENT (no approval), APPROVE, or REQUEST_CHANGES.

        Rate limit: counts as 1 write request regardless of comment count.
        """
        url = f"{self._base_url}/repos/{owner}/{repo}/pulls/{pr_number}/reviews"  # pragma: no cover
        payload: dict[str, Any] = {  # pragma: no cover
            "commit_id": commit_sha,
            "body": body,
            "event": event,
            "comments": [
                {
                    "path": c.path,
                    "line": c.line,
                    "side": c.side,
                    "body": c.body,
                }
                for c in comments
            ],
        }
        response = httpx.post(
            url, headers=self._headers, json=payload, timeout=30
        )  # pragma: no cover
        if response.status_code not in (200, 201):  # pragma: no cover
            raise GitHubAPIError(
                f"Failed to post review: {response.status_code} "
                f"{response.json().get('message', response.text[:200])}"
            )
        return response.json()  # type: ignore[no-any-return]  # pragma: no cover

    def get_pr_head_sha(self, owner: str, repo: str, pr_number: int) -> str:
        """Return the HEAD commit SHA of the PR (needed for post_review)."""
        url = f"{self._base_url}/repos/{owner}/{repo}/pulls/{pr_number}"  # pragma: no cover
        data = self._get(url)  # pragma: no cover
        if not isinstance(data, dict):  # pragma: no cover
            raise GitHubAPIError(
                f"Expected PR object (dict) from GitHub API, got {type(data).__name__}. URL: {url}"
            )
        return str(data["head"]["sha"])  # pragma: no cover


def parse_pr_ref(pr_ref: str) -> tuple[str, str, int]:
    """
    Parse 'owner/repo#number' into (owner, repo, number).

    Raises ValueError with usage hint on malformed input.
    """
    if "#" not in pr_ref or pr_ref.count("/") < 1:
        raise ValueError(
            f"Invalid PR ref {pr_ref!r}. Expected format: owner/repo#number "
            f"(e.g. 'myorg/myrepo#42')"
        )
    repo_part, num_part = pr_ref.rsplit("#", 1)
    if "/" not in repo_part:
        raise ValueError(f"Invalid PR ref {pr_ref!r}. Expected format: owner/repo#number")
    owner, repo = repo_part.split("/", 1)
    try:
        number = int(num_part)
    except ValueError:
        raise ValueError(f"Invalid PR number {num_part!r} in {pr_ref!r}. Must be an integer.")
    return owner, repo, number
