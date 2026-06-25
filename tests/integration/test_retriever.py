"""
Integration test for the Retriever orchestrator (Phase 13).

Flow:
  1. Create a minimal repo with 2 Python files that contain auth-related code.
  2. Index with provider=local (no API key required).
  3. Instantiate Retriever and call retrieve("how does authentication work").
  4. Assert the key invariants on the returned RetrievedContext.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from trelix.core.config import EmbedderConfig, IndexConfig
from trelix.indexing.indexer import Indexer
from trelix.retrieval.retriever import Retriever


# ---------------------------------------------------------------------------
# Fixture: tiny auth-focused repo
# ---------------------------------------------------------------------------

@pytest.fixture()
def auth_mini_repo(tmp_path: Path) -> Path:
    """
    Create a minimal repo with two Python files that contain
    authentication-related symbols.  No external API keys needed.
    """

    # auth.py — core authentication logic
    auth_src = textwrap.dedent("""\
        import hashlib
        import secrets


        def hash_password(password: str) -> str:
            \"\"\"Hash a plain-text password using SHA-256 + a random salt.\"\"\"
            salt = secrets.token_hex(16)
            digest = hashlib.sha256(f"{salt}{password}".encode()).hexdigest()
            return f"{salt}:{digest}"


        def verify_password(password: str, stored_hash: str) -> bool:
            \"\"\"Verify a plain-text password against a stored hash.\"\"\"
            salt, digest = stored_hash.split(":", 1)
            expected = hashlib.sha256(f"{salt}{password}".encode()).hexdigest()
            return secrets.compare_digest(expected, digest)


        class AuthManager:
            \"\"\"Manages user authentication state and sessions.\"\"\"

            def __init__(self) -> None:
                self._users: dict[str, str] = {}
                self._sessions: dict[str, str] = {}

            def register(self, username: str, password: str) -> None:
                \"\"\"Register a new user with a hashed password.\"\"\"
                if username in self._users:
                    raise ValueError(f"User {username!r} already exists.")
                self._users[username] = hash_password(password)

            def login(self, username: str, password: str) -> str:
                \"\"\"Authenticate the user and return a session token.\"\"\"
                stored = self._users.get(username)
                if stored is None or not verify_password(password, stored):
                    raise PermissionError("Invalid credentials.")
                token = secrets.token_urlsafe(32)
                self._sessions[token] = username
                return token

            def logout(self, token: str) -> None:
                \"\"\"Invalidate an existing session token.\"\"\"
                self._sessions.pop(token, None)

            def get_user(self, token: str) -> str | None:
                \"\"\"Return the username for a valid token, or None.\"\"\"
                return self._sessions.get(token)
    """)
    (tmp_path / "auth.py").write_text(auth_src, encoding="utf-8")

    # middleware.py — request-level auth check
    middleware_src = textwrap.dedent("""\
        from auth import AuthManager


        class AuthMiddleware:
            \"\"\"Request middleware that enforces token-based authentication.\"\"\"

            def __init__(self, auth_manager: AuthManager) -> None:
                self._auth = auth_manager

            def authenticate_request(self, headers: dict) -> str | None:
                \"\"\"
                Extract Bearer token from Authorization header and
                return the associated username, or None if unauthenticated.
                \"\"\"
                header = headers.get("Authorization", "")
                if not header.startswith("Bearer "):
                    return None
                token = header[len("Bearer "):]
                return self._auth.get_user(token)

            def require_auth(self, headers: dict) -> str:
                \"\"\"Like authenticate_request, but raises on failure.\"\"\"
                user = self.authenticate_request(headers)
                if user is None:
                    raise PermissionError("Authentication required.")
                return user
    """)
    (tmp_path / "middleware.py").write_text(middleware_src, encoding="utf-8")

    return tmp_path


@pytest.fixture()
def indexed_auth_repo(auth_mini_repo: Path) -> Path:
    """Index the auth_mini_repo using the local embedder."""
    config = IndexConfig(
        repo_path=str(auth_mini_repo),
        incremental=False,
        parse_workers=2,
        embedder=EmbedderConfig(provider="local"),
    )
    Indexer(config, quiet=True).index()
    return auth_mini_repo


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _make_config(repo_path: Path) -> IndexConfig:
    """Build an IndexConfig using the local (no-API-key) embedder."""
    return IndexConfig(
        repo_path=str(repo_path),
        incremental=False,
        parse_workers=2,
        embedder=EmbedderConfig(provider="local"),
    )


class TestRetriever:
    """Integration tests for the Retriever orchestrator."""

    def test_retrieve_returns_non_empty_results(self, indexed_auth_repo: Path) -> None:
        """Retriever must return at least one result for an auth-related query."""
        retriever = Retriever(_make_config(indexed_auth_repo))
        context = retriever.retrieve("how does authentication work")

        assert context.results, (
            "Expected at least one SearchResult for an auth query against an auth repo, got none."
        )

    def test_context_text_contains_code(self, indexed_auth_repo: Path) -> None:
        """context_text must contain actual code (def or class keyword)."""
        retriever = Retriever(_make_config(indexed_auth_repo))
        context = retriever.retrieve("how does authentication work")

        has_code = "def " in context.context_text or "class " in context.context_text
        assert has_code, (
            "context_text should contain Python source code (def/class), "
            f"but got:\n{context.context_text[:500]}"
        )

    def test_total_tokens_positive(self, indexed_auth_repo: Path) -> None:
        """total_tokens must be > 0 when results are returned."""
        retriever = Retriever(_make_config(indexed_auth_repo))
        context = retriever.retrieve("how does authentication work")

        assert context.total_tokens > 0, (
            f"Expected total_tokens > 0, got {context.total_tokens}"
        )

    def test_intent_is_set(self, indexed_auth_repo: Path) -> None:
        """intent field must be a non-empty string on the returned context."""
        retriever = Retriever(_make_config(indexed_auth_repo))
        context = retriever.retrieve("how does authentication work")

        assert context.intent, (
            f"Expected intent to be set, got {context.intent!r}"
        )

    def test_retrieve_with_explicit_plan(self, indexed_auth_repo: Path) -> None:
        """
        Passing an explicit plan bypasses the internal planner.
        The retriever must still return valid results.
        """
        from trelix.retrieval.planner.models import default_plan

        retriever = Retriever(_make_config(indexed_auth_repo))
        plan = default_plan("authentication login session")
        context = retriever.retrieve("authentication login session", plan=plan)

        assert context.results, "Expected results even when an explicit plan is passed."

    def test_debug_trace_written(self, indexed_auth_repo: Path) -> None:
        """After a query, at least one debug trace JSON should exist in .trelix/debug/."""
        retriever = Retriever(_make_config(indexed_auth_repo))
        retriever.retrieve("how does authentication work")

        debug_dir = Path(indexed_auth_repo) / ".trelix" / "debug"
        trace_files = list(debug_dir.glob("*.json"))
        assert trace_files, (
            f"Expected at least one debug trace JSON in {debug_dir}, found none."
        )

    def test_elapsed_seconds_set(self, indexed_auth_repo: Path) -> None:
        """elapsed_seconds must be a positive float."""
        retriever = Retriever(_make_config(indexed_auth_repo))
        context = retriever.retrieve("hash_password implementation")

        assert context.elapsed_seconds > 0, (
            f"Expected elapsed_seconds > 0, got {context.elapsed_seconds}"
        )
