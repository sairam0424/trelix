"""
RepoRegistry — manage the list of repos for federated search.

Config file format (JSON):
    {
        "repos": [
            {"alias": "trelix", "path": "/Users/me/trelix", "weight": 1.0},
            {"alias": "myapp",  "path": "/Users/me/myapp",  "weight": 0.8}
        ]
    }

Default config path: ~/.config/trelix/repos.json
Can be overridden by .trelix/federation.json in any repo.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("trelix.federation.registry")

_DEFAULT_CONFIG = Path.home() / ".config" / "trelix" / "repos.json"


@dataclass
class RepoEntry:
    """A single repo registered for federated search."""

    alias: str
    path: str
    weight: float = 1.0


class RepoRegistry:
    """Load, manage, and persist the federation repo list."""

    def __init__(self, config_path: str, entries: list[RepoEntry]) -> None:
        self._config_path = Path(config_path)
        self._entries: list[RepoEntry] = entries

    @classmethod
    def load(cls, config_path: str | None = None) -> RepoRegistry:
        """Load registry from JSON file. Returns empty registry if file missing or invalid."""
        path = Path(config_path) if config_path else _DEFAULT_CONFIG
        if not path.exists():
            return cls(str(path), [])
        try:
            data = json.loads(path.read_text())
            entries = [
                RepoEntry(
                    alias=r["alias"],
                    path=r["path"],
                    weight=float(r.get("weight", 1.0)),
                )
                for r in data.get("repos", [])
            ]
            return cls(str(path), entries)
        except Exception as exc:
            logger.debug("RepoRegistry.load failed for %s: %s", path, exc)
            return cls(str(path), [])

    def add(self, alias: str, path: str, weight: float = 1.0) -> None:
        """Add a repo. Raises ValueError if alias already registered."""
        if any(e.alias == alias for e in self._entries):
            raise ValueError(f"Alias '{alias}' already registered")
        self._entries.append(RepoEntry(alias=alias, path=path, weight=weight))

    def remove(self, alias: str) -> None:
        """Remove a repo by alias. No-op if not found."""
        self._entries = [e for e in self._entries if e.alias != alias]

    def list(self) -> list[RepoEntry]:
        """Return all registered repos."""
        return list(self._entries)

    def save(self) -> None:
        """Persist registry to JSON file."""
        self._config_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "repos": [
                {"alias": e.alias, "path": e.path, "weight": e.weight}
                for e in self._entries
            ]
        }
        self._config_path.write_text(json.dumps(data, indent=2))
