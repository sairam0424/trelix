"""Stand-ins shared by the REST confinement tests.

``tree`` snapshots a directory so a refusal can be shown to have written nothing;
``CreatingIndexer`` is an ``Indexer`` that, like the real one, creates the store the moment it is
built; ``pyvis_without_asserts`` is what pyvis becomes under ``python -O``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from trelix.store.db import Database

INDEX_BODY = {
    "files_found": 1,
    "files_indexed": 1,
    "files_skipped": 0,
    "symbols_extracted": 1,
    "chunks_total": 1,
    "chunks_embedded": 1,
    "errors": 0,
    "elapsed_seconds": 0.5,
}


def tree(path: Path) -> dict[str, bytes | None]:
    """Every entry below ``path`` with its bytes (``None`` for a directory)."""
    return {
        str(entry.relative_to(path)): entry.read_bytes() if entry.is_file() else None
        for entry in sorted(path.rglob("*"))
    }


class CreatingIndexer:
    """Stands in for ``Indexer``: like the real one it opens (and so creates) the store."""

    def __init__(self, config: Any) -> None:  # noqa: ANN401 - mirrors Indexer(config)
        self._config = config

    def index(self) -> dict[str, Any]:
        Database(self._config.db_path_absolute).close()
        return dict(INDEX_BODY)


def pyvis_without_asserts(monkeypatch: pytest.MonkeyPatch) -> None:
    """A pyvis whose ``save_graph`` writes whatever path it is given: ``python -O`` removes
    the ``assert`` that real pyvis uses to insist on ``.html``."""

    def _save_graph(path: str) -> None:
        Path(path).write_text("<html><body>graph</body></html>", encoding="utf-8")

    network = MagicMock()
    network.Network.return_value.save_graph.side_effect = _save_graph
    package = MagicMock()
    package.network = network
    monkeypatch.setitem(sys.modules, "pyvis", package)
    monkeypatch.setitem(sys.modules, "pyvis.network", network)
