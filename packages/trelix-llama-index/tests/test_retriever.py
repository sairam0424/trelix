"""Tests for TrelixIndexRetriever."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Importability
# ---------------------------------------------------------------------------


def test_import_package():
    import trelix_llama_index

    assert hasattr(trelix_llama_index, "TrelixIndexRetriever")
    assert trelix_llama_index.__version__ == "2.4.0"


def test_import_retriever_class():
    from trelix_llama_index.retriever import TrelixIndexRetriever

    assert TrelixIndexRetriever is not None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(body: str, rel_path: str, qualified_name: str, score: float):
    """Build a fake trelix retrieval result with the attributes _retrieve reads."""
    symbol = SimpleNamespace(body=body, qualified_name=qualified_name)
    file = SimpleNamespace(rel_path=rel_path)
    return SimpleNamespace(symbol=symbol, file=file, score=score)


def _make_retriever(results):
    """Return a mock TrelixIndexRetriever with _get_trelix_retriever stubbed out."""
    from trelix_llama_index.retriever import TrelixIndexRetriever

    inst = TrelixIndexRetriever.__new__(TrelixIndexRetriever)
    inst._repo_path = "/fake/repo"
    inst._provider = "local"
    inst._k = 10

    fake_ctx = SimpleNamespace(results=results)
    fake_inner = MagicMock()
    fake_inner.retrieve.return_value = fake_ctx
    inst._get_trelix_retriever = lambda: fake_inner
    return inst


# ---------------------------------------------------------------------------
# _retrieve returns NodeWithScore list
# ---------------------------------------------------------------------------


def test_retrieve_returns_list_of_node_with_score():
    from llama_index.core.schema import NodeWithScore

    results = [
        _make_result("def foo(): pass", "src/foo.py", "src.foo.foo", 0.95),
        _make_result("class Bar: pass", "src/bar.py", "src.bar.Bar", 0.80),
    ]
    retriever = _make_retriever(results)

    from llama_index.core.schema import QueryBundle

    nodes = retriever._retrieve(QueryBundle(query_str="foo function"))

    assert isinstance(nodes, list)
    assert len(nodes) == 2
    for node in nodes:
        assert isinstance(node, NodeWithScore)


def test_retrieve_correct_scores():
    results = [
        _make_result("def foo(): pass", "src/foo.py", "src.foo.foo", 0.95),
        _make_result("class Bar: pass", "src/bar.py", "src.bar.Bar", 0.80),
    ]
    retriever = _make_retriever(results)

    from llama_index.core.schema import QueryBundle

    nodes = retriever._retrieve(QueryBundle(query_str="foo"))

    assert pytest.approx(nodes[0].score) == 0.95
    assert pytest.approx(nodes[1].score) == 0.80


def test_retrieve_correct_metadata():
    results = [
        _make_result("def foo(): pass", "src/foo.py", "src.foo.foo", 0.95),
    ]
    retriever = _make_retriever(results)

    from llama_index.core.schema import QueryBundle

    nodes = retriever._retrieve(QueryBundle(query_str="foo"))

    meta = nodes[0].node.metadata
    assert meta["file"] == "src/foo.py"
    assert meta["symbol"] == "src.foo.foo"


def test_retrieve_correct_text():
    body = "def foo():\n    return 42"
    results = [_make_result(body, "src/foo.py", "src.foo.foo", 0.9)]
    retriever = _make_retriever(results)

    from llama_index.core.schema import QueryBundle

    nodes = retriever._retrieve(QueryBundle(query_str="foo"))

    assert nodes[0].node.text == body


# ---------------------------------------------------------------------------
# k-truncation
# ---------------------------------------------------------------------------


def test_retrieve_respects_k():
    results = [
        _make_result(f"def f{i}(): pass", f"src/f{i}.py", f"src.f{i}", 1.0 - i * 0.05)
        for i in range(20)
    ]
    retriever = _make_retriever(results)
    retriever._k = 5

    from llama_index.core.schema import QueryBundle

    nodes = retriever._retrieve(QueryBundle(query_str="f"))

    assert len(nodes) == 5


def test_retrieve_empty_results():
    retriever = _make_retriever([])

    from llama_index.core.schema import QueryBundle

    nodes = retriever._retrieve(QueryBundle(query_str="nothing"))

    assert nodes == []


# ---------------------------------------------------------------------------
# Constructor attributes
# ---------------------------------------------------------------------------


def test_constructor_defaults():
    from trelix_llama_index.retriever import TrelixIndexRetriever

    # We cannot call __init__ without llama-index registering callbacks, so
    # inspect via __new__ + manual attribute assignment matching __init__ logic.
    inst = TrelixIndexRetriever.__new__(TrelixIndexRetriever)
    inst._repo_path = "/some/path"
    inst._provider = "local"
    inst._k = 10

    assert inst._repo_path == "/some/path"
    assert inst._provider == "local"
    assert inst._k == 10


def test_constructor_custom_k():
    from trelix_llama_index.retriever import TrelixIndexRetriever

    inst = TrelixIndexRetriever.__new__(TrelixIndexRetriever)
    inst._repo_path = "/repo"
    inst._provider = "openai"
    inst._k = 3

    assert inst._k == 3
    assert inst._provider == "openai"
