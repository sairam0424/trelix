"""``mark_indexed``: give a tmp repo the (empty) index the read surfaces now require.

The CLI read commands, MCP tools, REST read routes, the LangChain and LlamaIndex retrievers,
``search-all`` and ``review`` all refuse, or skip, a repository that has no
``.trelix/index.db`` (see ``trelix.core.index_check``). A test that patches the Retriever
or GraphBuilder behind one of them and points it at a bare ``tmp_path`` therefore has to say
"this repository is indexed" first; the index it gets is a real, schema-only database, the
same file ``trelix index`` would have started from.
"""

from __future__ import annotations

from pathlib import Path

from trelix.core.config import IndexConfig
from trelix.store.db import Database


def mark_indexed(repo: Path) -> Path:
    """Create an empty but real index under ``repo``; returns ``repo`` for chaining.

    ``repo`` is created first when it does not exist yet.
    """
    repo.mkdir(parents=True, exist_ok=True)
    config = IndexConfig(repo_path=str(repo.resolve()))
    Database(config.db_path_absolute).close()
    return repo
