"""Utility functions for password hashing and verification."""

import hashlib
import secrets


def hash_password(pwd: str) -> str:
    """Hash a plain-text password with a random salt using SHA-256.

    Uses a 16-byte random salt prepended to the password before hashing.
    The returned value is ``<hex_salt>:<hex_digest>`` and is safe to store.

    Args:
        pwd: The plain-text password to hash.

    Returns:
        A string in the format ``<salt>:<digest>`` suitable for storage.

    Example::

        stored = hash_password("example-password")
        assert verify_password("example-password", stored) is True
    """
    salt = secrets.token_hex(16)
    digest = hashlib.sha256(f"{salt}{pwd}".encode()).hexdigest()
    return f"{salt}:{digest}"


def verify_password(pwd: str, hashed: str) -> bool:
    """Verify a plain-text password against a previously hashed value.

    Constant-time comparison is used to prevent timing attacks.

    Args:
        pwd: The candidate plain-text password.
        hashed: The stored hash in ``<salt>:<digest>`` format, as returned
                by :func:`hash_password`.

    Returns:
        True if the password matches, False otherwise.

    Raises:
        ValueError: If ``hashed`` is not in the expected format.

    Example::

        stored = hash_password("example-password")
        assert verify_password("example-password", stored) is True
        assert verify_password("wrong-password", stored) is False
    """
    if ":" not in hashed:
        raise ValueError(f"Invalid hash format: {hashed!r}")
    salt, digest = hashed.split(":", 1)
    expected = hashlib.sha256(f"{salt}{pwd}".encode()).hexdigest()
    return secrets.compare_digest(expected, digest)


def generate_token(length: int = 32) -> str:
    """Generate a cryptographically secure random URL-safe token.

    Args:
        length: Number of random bytes to use (default: 32).

    Returns:
        A URL-safe base64 encoded string of approximately ``length * 4/3`` characters.
    """
    return secrets.token_urlsafe(length)


def constant_time_compare(a: str, b: str) -> bool:
    """Compare two strings in constant time to prevent timing attacks.

    Args:
        a: First string.
        b: Second string.

    Returns:
        True if both strings are identical.
    """
    return secrets.compare_digest(a.encode("utf-8"), b.encode("utf-8"))
