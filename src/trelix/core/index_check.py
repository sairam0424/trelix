"""Does a repository have an index yet? Answered without creating one.

Opening the index database creates it (its schema and an empty vec0 table), and reading
`IndexConfig.db_path_absolute` creates `.trelix/` and its `.gitignore`. A read surface that
does either on a repository that was never indexed leaves an empty index behind: a later
`trelix search` then answers "no results" instead of "not indexed", and the repository looks
indexed. The read surfaces (the CLI read commands that build a Retriever or GraphBuilder, MCP
tools, REST routes, the LangChain and LlamaIndex retrievers, `search-all`, and `review`, which
uses it only to skip retrieval) call `require_index` BEFORE they build a Retriever,
GraphBuilder, AgentLoop or Database. `eval`, `eval-synthesis`, `telemetry`, `agent sessions` and
`taint` do not yet (see the CHANGELOG). Writers (`index`, `update-index`, `watch`,
`POST /index`) do not: creating the index is their job.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trelix.core.config import IndexConfig


class IndexNotFoundError(FileNotFoundError):
    """The repository has no index database to read.

    A `FileNotFoundError` (so an `OSError`) because that is what it is: the file a read
    needs is missing. A caller that already handles a missing file handles this too.
    """

    def __init__(self, db_path: Path, repo_path: str) -> None:
        super().__init__(f"No index found at {db_path}. Run trelix index {repo_path} first.")
        self.db_path = db_path
        self.repo_path = repo_path

    def __reduce__(self) -> tuple[type[IndexNotFoundError], tuple[Path, str]]:
        """Rebuild from the two constructor arguments, so pickle and `copy` work.

        `args` holds only the message, and the default reconstruction calls
        `IndexNotFoundError(message)`, which raises a `TypeError`. A process-pool or Celery-style
        worker that hits an unindexed repository would then break the pool instead of handing its
        parent this error.
        """
        return (type(self), (self.db_path, self.repo_path))


def require_index(config: IndexConfig) -> None:
    """Raise `IndexNotFoundError` unless `config`'s index database already exists.

    Uses `db_path_resolved`, not `db_path_absolute`: the latter creates `.trelix/` and its
    `.gitignore` as a side effect of being read.
    """
    db_path = config.db_path_resolved
    if not db_path.exists():
        raise IndexNotFoundError(db_path, config.repo_path)
