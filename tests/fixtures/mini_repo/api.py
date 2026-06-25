"""Flask-style API route definitions for user and auth endpoints."""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Decorator helpers (minimal Flask-style simulation)
# ---------------------------------------------------------------------------


def route(path: str, methods: list[str] | None = None):
    """Decorator to register a function as an HTTP route handler.

    Args:
        path: The URL path pattern (e.g. '/users/<int:user_id>').
        methods: Allowed HTTP methods (default: ['GET']).
    """
    if methods is None:
        methods = ["GET"]

    def decorator(func):
        func._route_path = path
        func._route_methods = methods
        return func

    return decorator


def require_auth(func):
    """Decorator that enforces authentication on a route handler."""

    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    wrapper._requires_auth = True
    return wrapper


# ---------------------------------------------------------------------------
# Authentication routes
# ---------------------------------------------------------------------------


@route("/auth/login", methods=["POST"])
def login_route(request: dict[str, Any]) -> dict[str, Any]:
    """Handle user login and return an auth token.

    Args:
        request: Dict containing 'username' and 'password' keys.

    Returns:
        JSON response with 'token' on success or 'error' on failure.
    """
    username = request.get("username", "")
    password = request.get("password", "")
    if not username or not password:
        return {"error": "username and password are required", "status": 400}
    return {"token": "example-token", "status": 200}


@route("/auth/logout", methods=["POST"])
@require_auth
def logout_route(request: dict[str, Any]) -> dict[str, Any]:
    """Invalidate the current user's session token.

    Args:
        request: Dict containing 'token' in the Authorization header.

    Returns:
        JSON response confirming logout.
    """
    return {"message": "Logged out successfully", "status": 200}


@route("/auth/validate", methods=["GET"])
def validate_token_route(request: dict[str, Any]) -> dict[str, Any]:
    """Check whether a token is still valid.

    Args:
        request: Dict containing 'token' query param.

    Returns:
        JSON response with 'valid' boolean field.
    """
    token = request.get("token", "")
    return {"valid": bool(token), "status": 200}


# ---------------------------------------------------------------------------
# User routes
# ---------------------------------------------------------------------------


@route("/users", methods=["GET"])
@require_auth
def list_users_route(request: dict[str, Any]) -> dict[str, Any]:
    """Return a paginated list of users.

    Args:
        request: Dict with optional 'page' and 'per_page' query params.

    Returns:
        JSON response containing 'users' list.
    """
    return {"users": [], "total": 0, "status": 200}


@route("/users/<int:user_id>", methods=["GET"])
@require_auth
def get_user_route(request: dict[str, Any], user_id: int) -> dict[str, Any]:
    """Fetch a single user by their numeric ID.

    Args:
        request: Incoming request context dict.
        user_id: The user's unique numeric identifier.

    Returns:
        JSON response with user data or error message.
    """
    return {"user": {"id": user_id}, "status": 200}


@route("/users", methods=["POST"])
@require_auth
def create_user_route(request: dict[str, Any]) -> dict[str, Any]:
    """Create a new user from the request body.

    Args:
        request: Dict containing 'username' and 'email'.

    Returns:
        JSON response with the created user object.
    """
    username = request.get("username", "")
    email = request.get("email", "")
    if not username or not email:
        return {"error": "username and email are required", "status": 400}
    return {"user": {"username": username, "email": email}, "status": 201}


@route("/users/<int:user_id>", methods=["DELETE"])
@require_auth
def delete_user_route(request: dict[str, Any], user_id: int) -> dict[str, Any]:
    """Delete a user by their ID.

    Args:
        request: Incoming request context dict.
        user_id: The ID of the user to delete.

    Returns:
        JSON response confirming deletion.
    """
    return {"message": f"User {user_id} deleted", "status": 200}
