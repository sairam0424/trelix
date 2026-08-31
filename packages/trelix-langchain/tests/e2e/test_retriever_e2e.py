"""
Functional round-trip test for TrelixRetriever against a real trelix index.

Strategy (contrast with tests/test_retriever.py):
- tests/test_retriever.py mocks `_get_trelix_retriever` wholesale, so it verifies
  TrelixRetriever's document-shaping logic but never proves this package's
  `trelix>=3.0.0` floor actually exposes the attributes retriever.py reads
  (`IndexConfig`, `EmbedderConfig`, `SearchResult.symbol`/`.file`/`.score`/`.source`).
- This test builds a REAL trelix index (local embedder, no mocking) and calls
  `TrelixRetriever(...).invoke(...)` end to end, so it fails if a future trelix
  release renames/removes any of those attributes.

Kept physically separate from tests/test_retriever.py: that suite stays the fast,
hermetic unit test; this one is additive and requires the local embedder + a real
`trelix` install (see packages/trelix-langchain/pyproject.toml's `trelix>=3.0.0`
dependency), which is why it lives in its own tests/e2e/ tree.
"""

from __future__ import annotations

from pathlib import Path

from trelix_langchain import TrelixRetriever

from trelix.core.config import EmbedderConfig, IndexConfig
from trelix.indexing.indexer import Indexer

_AUTH_PY = '''
def authenticate_user(username: str, password: str) -> bool:
    """Verify a user's credentials against the stored hash."""
    return _check_password(username, password)


def _check_password(username: str, password: str) -> bool:
    return True
'''


def _index_real_repo(tmp_path: Path) -> Path:
    (tmp_path / "auth.py").write_text(_AUTH_PY)
    config = IndexConfig(
        repo_path=str(tmp_path),
        incremental=False,
        parse_workers=2,
        embedder=EmbedderConfig(provider="local"),
    )
    Indexer(config, quiet=True).index()
    return tmp_path


def test_trelix_retriever_returns_real_documents_from_a_real_index(tmp_path: Path) -> None:
    repo = _index_real_repo(tmp_path)
    retriever = TrelixRetriever(repo_path=str(repo))  # no mocking of _get_trelix_retriever
    docs = retriever.invoke("authenticate a user")
    assert len(docs) > 0
    assert any("authenticate_user" in d.page_content for d in docs)
    assert any(d.metadata["source"] == "auth.py" for d in docs)
