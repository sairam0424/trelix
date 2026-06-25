"""Authentication service for managing user login, logout, and token validation."""

import hashlib
import secrets


class AuthService:
    """Manages user authentication, sessions, and token validation."""

    def __init__(self) -> None:
        self._users: dict[str, str] = {}
        self._sessions: dict[str, str] = {}

    def login(self, username: str, password: str) -> str:
        """Authenticate a user and return a session token.

        Args:
            username: The user's login name.
            password: The user's plain-text password.

        Returns:
            A session token string on successful authentication.

        Raises:
            PermissionError: If the credentials are invalid.
        """
        stored_hash = self._users.get(username)
        if stored_hash is None:
            raise PermissionError("Invalid credentials.")
        if not self._verify(password, stored_hash):
            raise PermissionError("Invalid credentials.")
        token = secrets.token_urlsafe(32)
        self._sessions[token] = username
        return token

    def logout(self, token: str) -> None:
        """Invalidate an active session token.

        Args:
            token: The session token to revoke.
        """
        self._sessions.pop(token, None)

    def validate_token(self, token: str) -> bool:
        """Check whether a session token is currently active.

        Args:
            token: The session token to validate.

        Returns:
            True if the token is valid, False otherwise.
        """
        return token in self._sessions

    def register(self, username: str, password: str) -> None:
        """Register a new user with a hashed password.

        Args:
            username: The desired username.
            password: The plain-text password to hash and store.

        Raises:
            ValueError: If the username is already taken.
        """
        if username in self._users:
            raise ValueError(f"User {username!r} already exists.")
        self._users[username] = self._hash(password)

    def _hash(self, password: str) -> str:
        salt = secrets.token_hex(16)
        digest = hashlib.sha256(f"{salt}{password}".encode()).hexdigest()
        return f"{salt}:{digest}"

    def _verify(self, password: str, stored_hash: str) -> bool:
        salt, digest = stored_hash.split(":", 1)
        expected = hashlib.sha256(f"{salt}{password}".encode()).hexdigest()
        return secrets.compare_digest(expected, digest)
