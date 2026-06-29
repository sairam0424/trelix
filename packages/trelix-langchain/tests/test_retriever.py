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
    import trelix_langchain

    assert hasattr(trelix_langchain, "__version__")
    assert trelix_langchain.__version__ == "1.1.0"


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
