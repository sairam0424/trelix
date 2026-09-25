"""
Tests for TrelixRetriever.

Strategy:
- Import / subclass checks run without any trelix index on disk.
- Functional tests mock out trelix internals so they stay fast and hermetic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from trelix_langchain import TrelixRetriever

# ---------------------------------------------------------------------------
# Helpers — minimal stubs that mimic trelix data-model shapes
# ---------------------------------------------------------------------------


@dataclass
class _Language:
    value: str = "python"


@dataclass
class _SymbolKind:
    value: str = "function"


@dataclass
class _Symbol:
    body: str = "def hello(): pass"
    qualified_name: str = "mymodule.hello"
    kind: _SymbolKind = field(default_factory=_SymbolKind)
    line_start: int = 1
    line_end: int = 3


@dataclass
class _File:
    rel_path: str = "src/mymodule.py"
    language: _Language = field(default_factory=_Language)


@dataclass
class _SearchResult:
    symbol: _Symbol = field(default_factory=_Symbol)
    file: _File = field(default_factory=_File)
    score: float = 0.95
    source: str = "vector"


@dataclass
class _RetrievedContext:
    results: list


def _make_context(n: int = 3) -> _RetrievedContext:
    """Return a fake RetrievedContext with *n* results."""
    return _RetrievedContext(results=[_SearchResult() for _ in range(n)])


# ---------------------------------------------------------------------------
# 1. Importability
# ---------------------------------------------------------------------------


def test_import_trelix_retriever():
    """TrelixRetriever must be importable from the package root."""
    from trelix_langchain import TrelixRetriever as TR  # noqa: F401

    assert TR is TrelixRetriever


def test_version_exposed():
    """The dunder must agree with this package's own pyproject.toml.

    This asserted the literal "3.1.2", which quietly made it a version stamp the
    release gate does not police: bumping the eleven stamps it *does* check left
    this one behind, so CI would have gone red only after the bump was committed.
    The file is read rather than queried through importlib.metadata because an
    editable install freezes its METADATA at install time -- that would fail for
    anyone who bumps the version without reinstalling.
    """
    import re
    from pathlib import Path

    import trelix_langchain

    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    packaged = re.search(r'^version = "(.+?)"', pyproject.read_text(), re.M)
    assert packaged is not None, f"no version field found in {pyproject}"
    assert hasattr(trelix_langchain, "__version__")
    assert trelix_langchain.__version__ == packaged.group(1)


# ---------------------------------------------------------------------------
# 2. BaseRetriever subclass
# ---------------------------------------------------------------------------


def test_is_base_retriever_subclass():
    assert issubclass(TrelixRetriever, BaseRetriever)


def test_instantiation_sets_fields():
    r = TrelixRetriever(repo_path="/tmp/repo", provider="openai", k=5)
    assert r.repo_path == "/tmp/repo"
    assert r.provider == "openai"
    assert r.k == 5


def test_default_field_values():
    r = TrelixRetriever(repo_path="/tmp/repo")
    assert r.provider == "local"
    assert r.k == 10


# ---------------------------------------------------------------------------
# 3. invoke() / _get_relevant_documents() returns Documents with correct metadata
# ---------------------------------------------------------------------------


def _make_retriever_with_mock(k: int = 10, n_results: int = 3) -> TrelixRetriever:
    """Return a TrelixRetriever whose internal trelix retriever is mocked."""
    tr = TrelixRetriever(repo_path="/tmp/fake-repo", k=k)
    mock_inner = MagicMock()
    mock_inner.retrieve.return_value = _make_context(n_results)
    tr._get_trelix_retriever = lambda: mock_inner  # type: ignore[method-assign]
    return tr


def test_invoke_returns_list_of_documents():
    tr = _make_retriever_with_mock()
    docs = tr.invoke("find hello function")
    assert isinstance(docs, list)
    assert len(docs) == 3
    assert all(isinstance(d, Document) for d in docs)


def test_document_page_content_is_symbol_body():
    tr = _make_retriever_with_mock()
    docs = tr.invoke("hello")
    assert docs[0].page_content == "def hello(): pass"


def test_document_metadata_source():
    tr = _make_retriever_with_mock()
    docs = tr.invoke("hello")
    assert docs[0].metadata["source"] == "src/mymodule.py"


def test_document_metadata_symbol():
    tr = _make_retriever_with_mock()
    docs = tr.invoke("hello")
    assert docs[0].metadata["symbol"] == "mymodule.hello"


def test_document_metadata_language():
    tr = _make_retriever_with_mock()
    docs = tr.invoke("hello")
    assert docs[0].metadata["language"] == "python"


def test_document_metadata_kind():
    tr = _make_retriever_with_mock()
    docs = tr.invoke("hello")
    assert docs[0].metadata["kind"] == "function"


def test_document_metadata_lines():
    tr = _make_retriever_with_mock()
    docs = tr.invoke("hello")
    assert docs[0].metadata["lines"] == "1-3"


def test_document_metadata_score():
    tr = _make_retriever_with_mock()
    docs = tr.invoke("hello")
    assert docs[0].metadata["score"] == pytest.approx(0.95)


def test_document_metadata_retrieval_source():
    tr = _make_retriever_with_mock()
    docs = tr.invoke("hello")
    assert docs[0].metadata["retrieval_source"] == "vector"


# ---------------------------------------------------------------------------
# 4. k limits number of returned documents
# ---------------------------------------------------------------------------


def test_k_limits_results_when_fewer_available():
    """k=5, but only 3 results available — should return 3."""
    tr = _make_retriever_with_mock(k=5, n_results=3)
    docs = tr.invoke("query")
    assert len(docs) == 3


def test_k_limits_results_when_more_available():
    """k=2, 5 results available — should return exactly 2."""
    tr = _make_retriever_with_mock(k=2, n_results=5)
    docs = tr.invoke("query")
    assert len(docs) == 2


def test_k_equals_zero_returns_empty():
    tr = _make_retriever_with_mock(k=0, n_results=5)
    docs = tr.invoke("query")
    assert docs == []


def test_k_default_ten_limits_large_result_set():
    """Default k=10: 15 results available — should cap at 10."""
    tr = _make_retriever_with_mock(k=10, n_results=15)
    docs = tr.invoke("query")
    assert len(docs) == 10


# ---------------------------------------------------------------------------
# 5. Empty results
# ---------------------------------------------------------------------------


def test_empty_results_returns_empty_list():
    tr = _make_retriever_with_mock(k=10, n_results=0)
    docs = tr.invoke("nothing here")
    assert docs == []


# ---------------------------------------------------------------------------
# 6. provider cast() stays in sync with core's real Literal
# ---------------------------------------------------------------------------


def test_provider_cast_covers_every_value_core_actually_accepts():
    """_get_trelix_retriever() casts self.provider to a hardcoded Literal before
    handing it to EmbedderConfig. cast() performs no runtime validation, so a
    stale, narrower list here doesn't break anything at runtime -- it just
    lies to IDEs/mypy about which providers are valid. Deriving the expected
    set from EmbedderConfig.provider's own annotation means this test fails
    the moment core adds a provider the adapter's cast doesn't yet list,
    instead of silently drifting again."""
    import inspect
    import re
    import typing

    from trelix.core.config import EmbedderConfig

    core_values = set(typing.get_args(typing.get_type_hints(EmbedderConfig)["provider"]))

    source = inspect.getsource(TrelixRetriever._get_trelix_retriever)
    match = re.search(r"Literal\[(.*?)\]", source, re.DOTALL)
    assert match, "expected a Literal[...] cast target in _get_trelix_retriever"
    adapter_values = {v.strip().strip('"') for v in match.group(1).split(",") if v.strip()}

    assert adapter_values == core_values, (
        f"adapter cast() Literal is stale: missing {core_values - adapter_values}, "
        f"has-extra {adapter_values - core_values}"
    )


# ---------------------------------------------------------------------------
# 6. _get_trelix_retriever caches the underlying Retriever on the instance
# ---------------------------------------------------------------------------


def test_get_trelix_retriever_constructs_the_underlying_retriever_only_once(tmp_path):
    """Retriever construction is expensive (loads the embedding model from
    disk for the local provider) -- repeated calls on the same
    TrelixRetriever instance must reuse it, not rebuild every time."""
    from unittest.mock import patch

    tr = TrelixRetriever(repo_path=str(tmp_path))

    with patch("trelix.retrieval.retriever.Retriever") as MockRetriever:
        first = tr._get_trelix_retriever()
        second = tr._get_trelix_retriever()
        third = tr._get_trelix_retriever()

    assert first is second is third
    assert MockRetriever.call_count == 1


def test_different_trelix_retriever_instances_do_not_share_a_cached_retriever(tmp_path):
    """Caching is per-instance (a pydantic PrivateAttr), not global -- two
    separate TrelixRetriever objects must each build their own."""
    from unittest.mock import patch

    repo_a = tmp_path / "a"
    repo_b = tmp_path / "b"
    repo_a.mkdir()
    repo_b.mkdir()
    tr_a = TrelixRetriever(repo_path=str(repo_a))
    tr_b = TrelixRetriever(repo_path=str(repo_b))

    with patch("trelix.retrieval.retriever.Retriever") as MockRetriever:
        tr_a._get_trelix_retriever()
        tr_b._get_trelix_retriever()

    assert MockRetriever.call_count == 2
