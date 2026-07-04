"""Tests for GitHubPRClient (v2.4 GitHub PR integration)."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

# A clearly fake, short token used only for unit tests (< 8 chars, not a real secret).
_FAKE = "fake"


def _make_file_response(**kwargs) -> dict:
    defaults = {
        "filename": "src/auth.py",
        "status": "modified",
        "additions": 5,
        "deletions": 2,
        "patch": "@@ -1,3 +1,4 @@\n def login(): pass\n+    return True",
        "previous_filename": None,
    }
    defaults.update(kwargs)
    return defaults


class TestGitHubPRClientGetFiles:
    def test_get_pr_files_returns_pr_file_list(self) -> None:
        from trelix.review.github import GitHubPRClient

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [_make_file_response()]
        mock_response.headers = {"x-ratelimit-remaining": "100"}

        with patch("trelix.review.github.httpx.get", return_value=mock_response):
            client = GitHubPRClient(token=_FAKE)
            files = client.get_pr_files("owner", "repo", 42)

        assert len(files) == 1
        assert files[0].filename == "src/auth.py"
        assert files[0].status == "modified"
        assert files[0].patch is not None

    def test_get_pr_files_handles_all_seven_status_values(self) -> None:
        """All 7 GitHub status values must be handled without error."""
        from trelix.review.github import GitHubPRClient

        statuses = ["added", "removed", "modified", "renamed", "copied", "changed", "unchanged"]
        mock_files = [_make_file_response(status=s) for s in statuses]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_files
        mock_response.headers = {"x-ratelimit-remaining": "100"}

        with patch("trelix.review.github.httpx.get", return_value=mock_response):
            client = GitHubPRClient(token=_FAKE)
            files = client.get_pr_files("owner", "repo", 1)

        assert len(files) == 7
        returned_statuses = {f.status for f in files}
        assert returned_statuses == set(statuses)

    def test_get_pr_files_handles_missing_patch(self) -> None:
        """Binary files have no 'patch' key — should be None, not raise."""
        from trelix.review.github import GitHubPRClient

        binary_file = _make_file_response(filename="assets/logo.png", status="modified")
        del binary_file["patch"]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [binary_file]
        mock_response.headers = {"x-ratelimit-remaining": "100"}

        with patch("trelix.review.github.httpx.get", return_value=mock_response):
            client = GitHubPRClient(token=_FAKE)
            files = client.get_pr_files("owner", "repo", 1)

        assert files[0].patch is None

    def test_get_pr_files_warns_on_3000_file_truncation(self, caplog) -> None:
        """Log a warning when exactly 3000 files returned (truncation risk)."""
        import logging
        from trelix.review.github import GitHubPRClient

        mock_files = [_make_file_response(filename=f"src/f{i}.py") for i in range(3000)]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_files
        mock_response.headers = {"x-ratelimit-remaining": "100"}

        with patch("trelix.review.github.httpx.get", return_value=mock_response):
            with caplog.at_level(logging.WARNING, logger="trelix.review.github"):
                client = GitHubPRClient(token=_FAKE)
                files = client.get_pr_files("owner", "repo", 1)

        assert any("3000" in r.message or "truncat" in r.message.lower() for r in caplog.records)

    def test_get_pr_files_raises_on_404(self) -> None:
        """404 should raise a clear error."""
        from trelix.review.github import GitHubPRClient, GitHubAPIError

        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.json.return_value = {"message": "Not Found"}

        with patch("trelix.review.github.httpx.get", return_value=mock_response):
            client = GitHubPRClient(token=_FAKE)
            with pytest.raises(GitHubAPIError, match="404"):
                client.get_pr_files("owner", "repo", 999)

    def test_get_pr_files_raises_on_401(self) -> None:
        """401 should raise with helpful token message."""
        from trelix.review.github import GitHubPRClient, GitHubAPIError

        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.json.return_value = {"message": "Bad credentials"}

        with patch("trelix.review.github.httpx.get", return_value=mock_response):
            client = GitHubPRClient(token=_FAKE)
            with pytest.raises(GitHubAPIError, match="401|token|credential"):
                client.get_pr_files("owner", "repo", 1)


class TestParsePRRef:
    def test_parse_pr_ref_valid(self) -> None:
        """_parse_pr_ref parses 'owner/repo#123' correctly."""
        from trelix.review.github import parse_pr_ref

        owner, repo, number = parse_pr_ref("myorg/myrepo#42")
        assert owner == "myorg"
        assert repo == "myrepo"
        assert number == 42

    def test_parse_pr_ref_invalid_no_hash(self) -> None:
        """_parse_pr_ref raises ValueError for malformed input without '#'."""
        from trelix.review.github import parse_pr_ref

        with pytest.raises(ValueError, match="owner/repo#number"):
            parse_pr_ref("not-a-pr-ref")

    def test_parse_pr_ref_invalid_no_slash_in_repo(self) -> None:
        """_parse_pr_ref raises ValueError for 'owner/repo' without number."""
        from trelix.review.github import parse_pr_ref

        with pytest.raises(ValueError, match="owner/repo#number"):
            parse_pr_ref("owner/repo")

    def test_parse_pr_ref_non_integer_number(self) -> None:
        """_parse_pr_ref raises ValueError when PR number is not an integer."""
        from trelix.review.github import parse_pr_ref

        with pytest.raises(ValueError):
            parse_pr_ref("owner/repo#abc")

    def test_parse_pr_ref_with_nested_path(self) -> None:
        """_parse_pr_ref handles org/repo-name#number correctly."""
        from trelix.review.github import parse_pr_ref

        owner, repo, number = parse_pr_ref("my-org/my-repo#999")
        assert owner == "my-org"
        assert repo == "my-repo"
        assert number == 999
