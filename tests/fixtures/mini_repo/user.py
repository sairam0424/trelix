"""User domain models and repository for managing user persistence."""

from dataclasses import dataclass, field


@dataclass
class User:
    """Represents a registered user account.

    Attributes:
        id: Unique numeric identifier.
        username: Login name (must be unique).
        email: User's email address.
        is_active: Whether the account is active.
    """

    id: int
    username: str
    email: str
    is_active: bool = True
    roles: list[str] = field(default_factory=list)


class UserRepository:
    """In-memory repository for CRUD operations on User objects."""

    def __init__(self) -> None:
        self._store: dict[int, User] = {}
        self._next_id: int = 1

    def get_by_id(self, user_id: int) -> User | None:
        """Retrieve a user by their numeric ID.

        Args:
            user_id: The unique user identifier.

        Returns:
            The User object if found, None otherwise.
        """
        return self._store.get(user_id)

    def get_by_username(self, username: str) -> User | None:
        """Retrieve a user by their username.

        Args:
            username: The login name to search for.

        Returns:
            The first User with matching username, or None.
        """
        for user in self._store.values():
            if user.username == username:
                return user
        return None

    def create(self, username: str, email: str) -> User:
        """Create a new user and persist it in the repository.

        Args:
            username: The login name for the new user.
            email: The email address for the new user.

        Returns:
            The newly created User object with an assigned id.

        Raises:
            ValueError: If username already exists.
        """
        if self.get_by_username(username) is not None:
            raise ValueError(f"Username {username!r} already taken.")
        user = User(id=self._next_id, username=username, email=email)
        self._store[self._next_id] = user
        self._next_id += 1
        return user

    def delete(self, user_id: int) -> bool:
        """Delete a user from the repository.

        Args:
            user_id: The ID of the user to remove.

        Returns:
            True if the user was found and removed, False otherwise.
        """
        if user_id not in self._store:
            return False
        del self._store[user_id]
        return True

    def list_all(self) -> list[User]:
        """Return all users in the repository.

        Returns:
            A list of all User objects, sorted by id.
        """
        return sorted(self._store.values(), key=lambda u: u.id)

    def count(self) -> int:
        """Return the total number of users in the repository."""
        return len(self._store)
