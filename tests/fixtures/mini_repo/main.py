"""Application entry point — wires together auth, user, and API layers."""

from __future__ import annotations

from auth import AuthService
from user import UserRepository
from utils import hash_password, verify_password


def setup_demo_data(
    auth_svc: AuthService,
    user_repo: UserRepository,
) -> None:
    """Populate the services with demo users for local development.

    Creates two demo accounts:
    - admin / admin123
    - alice  / password1

    Args:
        auth_svc: The authentication service to register users in.
        user_repo: The user repository to persist User records in.
    """
    demo_users = [
        ("admin", "admin123", "admin@example.com"),
        ("alice", "password1", "alice@example.com"),
    ]
    for username, password, email in demo_users:
        try:
            auth_svc.register(username, password)
            user_repo.create(username, email)
        except ValueError:
            pass  # already registered — skip


def run_auth_flow(auth_svc: AuthService) -> None:
    """Demonstrate the full login → validate → logout flow.

    Args:
        auth_svc: The authentication service to exercise.
    """
    token = auth_svc.login("admin", "admin123")
    print(f"Logged in — token: {token[:16]}…")

    is_valid = auth_svc.validate_token(token)
    print(f"Token valid: {is_valid}")

    auth_svc.logout(token)
    is_valid_after = auth_svc.validate_token(token)
    print(f"Token valid after logout: {is_valid_after}")


def main() -> None:
    """Application entry point.

    Initialises all services, loads demo data, and runs a quick
    auth-flow smoke-test to verify the wiring is correct.
    """
    auth_svc = AuthService()
    user_repo = UserRepository()

    setup_demo_data(auth_svc, user_repo)

    print(f"Users in repository: {user_repo.count()}")

    # Smoke-test password utilities independently
    stored = hash_password("test_password")
    assert verify_password("test_password", stored), "Password verification failed!"
    print("Password hash/verify: OK")

    run_auth_flow(auth_svc)

    # Show all users
    for user in user_repo.list_all():
        print(f"  User({user.id}): {user.username} <{user.email}>")

    print("Done.")


if __name__ == "__main__":
    main()
