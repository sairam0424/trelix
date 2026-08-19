"""Outcome tests for the caller-supplied `intent_hint` (EXE-01).

WHY THIS FILE EXISTS
--------------------
`plan_from_intent_hint()` used to stamp `routing_tier=TIER_1_DIRECT` on EVERY
intent while also setting `strategy=INTENT_STRATEGIES[intent]` — and
`Retriever._execute_plan()` tests the tier BEFORE the intent, so it returned
`_retrieve_project_overview(plan)` and the strategy was never read. The right
plan was computed and thrown away one function later. Measured on the trelix
repo itself: all 8 `IntentType` values returned byte-identical output — 40
README sections, 0 code files, 0/17 overlap with the correct result set, in
0.02s because no retrieval ran. Reproduced at fixture scale below: every hint
returned the same 5 `file_direct` symbols and DELETED `verify_token`, which the
no-hint baseline finds.

Two tests in `tests/unit/test_planner.py` locked that in and were removed, not
amended — they are the reason a green suite said nothing about this path:

  * `test_valid_intent_stamps_tier1_direct` asserted the stamp was correct
    because it "Skips the LLM entirely, same as AdaptiveRouter._tier1_plan()",
    conflating skipping the LLM CLASSIFIER with skipping ALL RETRIEVAL;
  * `test_valid_intent_returns_matching_plan` asserted
    `plan.strategy is INTENT_STRATEGIES[intent]` — true, and worthless, because
    the tier it was stamped beside guaranteed nothing would ever read it.

Both are replaced by `test_tier1_direct_is_paired_only_with_project_overview`
there and by the outcome assertions here, which look at what retrieval RETURNED
rather than at a field on the plan object.

WHY THE FIXTURE HAS BOTH A README AND A MODULE DOCSTRING
--------------------------------------------------------
`_assemble_direct`'s breadth floor widens a thin direct lookup to standard
retrieval when it matched fewer than 2 files AND fewer than 10 symbols. A
README (4 markdown `section` symbols) plus one Python file carrying a module
docstring (a `<module>` symbol) is 2 files, so the floor does NOT fire and the
project-overview short-circuit is observable rather than silently rescued —
which is exactly the shape of the real repo, where `limit=40` is exhausted by
the root README alone.

NO NETWORK, NO PAID CALLS
-------------------------
`make_embedder`/`make_vector_store` are replaced with fakes in both the indexer
and the retriever, and `QueryPlanner` with a stub that returns `default_plan()`
— the documented no-API-key fallback. The no-hint baseline must NOT reach a
live planner: with a real key in the environment it would be a billed LLM call
on every run.
"""

from __future__ import annotations

import pathlib
import textwrap
from typing import Any
from unittest.mock import patch

import pytest

from trelix.core.config import EmbedderConfig, IndexConfig, StoreConfig
from trelix.core.models import RetrievedContext
from trelix.retrieval.planner.models import (
    INTENT_STRATEGIES,
    IntentType,
    QueryPlan,
    default_plan,
    plan_from_intent_hint,
)

# The query names the target symbol so the lexical legs (bm25/grep) can find it
# without a real embedding model: the fake vector store below returns nothing,
# so a hint whose legs are honoured has to earn its hit lexically.
_QUERY = "what does verify_token do"
_TARGET_FUNCTION = "verify_token"
# project_overview's correct answer IS the module/README summary set, so its
# target is the module symbol rather than the function. That destination being
# query-independent is a separate, still-open defect (DOG-09); this file only
# asserts that the hint routes to the destination its strategy names.
_MODULE_SYMBOL = "<module>"

# The three intents whose strategy declares the `file_direct` leg each own a
# dedicated direct-lookup method in `_execute_plan`. Derived from
# INTENT_STRATEGIES by the equivalence asserted in the test body, so this table
# cannot drift away from the strategies it mirrors.
_DIRECT_PATHS: dict[IntentType, str] = {
    IntentType.FILE_OVERVIEW: "_retrieve_file_overview",
    IntentType.PROJECT_OVERVIEW: "_retrieve_project_overview",
    IntentType.CONFIG_LOOKUP: "_retrieve_config",
}

_DIM = 4


class _FakeEmbedder:
    """Deterministic stand-in — no model download, no API key, no cost."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * _DIM for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.1] * _DIM

    async def embed_async(self, texts: list[str]) -> list[list[float]]:
        return self.embed(texts)

    @property
    def dimension(self) -> int:
        return _DIM


class _FakeVectorStore:
    """Stores nothing and matches nothing: every hit here is lexical."""

    def upsert_batch(self, pairs: list[tuple[int, list[float]]]) -> None:
        pass

    def delete_batch(self, ids: list[int]) -> None:
        pass

    def search(self, vector: list[float], k: int) -> list[Any]:
        return []


class _StubPlanner:
    """The no-API-key QueryPlanner behaviour, made unconditional.

    A real QueryPlanner builds a chat client from EmbedderConfig credentials,
    which pydantic-settings will read out of a developer's `.env` even when the
    process environment is clean. The baseline leg of every test below calls
    `retrieve()` with `plan=None`, so without this stub the suite would bill an
    LLM call per test on any machine that has a key.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def plan(self, query: str, project_context: dict[str, Any] | None = None) -> QueryPlan:
        return default_plan(query)


def _write_fixture(root: pathlib.Path) -> None:
    (root / "README.md").write_text(
        textwrap.dedent("""\
        # Demo Project

        ## Overview
        A tiny demo repository used to exercise retrieval routing.

        ## Installation
        Nothing to install.

        ## Usage
        Import the auth module.
        """),
        encoding="utf-8",
    )
    (root / "auth.py").write_text(
        textwrap.dedent('''\
        """Token verification helpers for the demo project."""


        def verify_token(token: str) -> bool:
            """Return True when the supplied token is a well-formed session token."""
            return token.startswith("sess_") and len(token) > 12


        def issue_token(user: str) -> str:
            """Mint a session token for the named user."""
            return f"sess_{user}_abcdefghijkl"
        '''),
        encoding="utf-8",
    )


def _make_config(repo: pathlib.Path) -> IndexConfig:
    return IndexConfig(
        repo_path=str(repo),
        incremental=False,
        store=StoreConfig(db_path=str(repo / ".trelix" / "index.db")),
        embedder=EmbedderConfig.model_construct(provider="local"),
    )


# Indexing is the expensive half and the fixture is read-only, so it runs once
# per session. The fixture is function-scoped on purpose even so: conftest's
# autouse env isolation is function-scoped, and a module-scoped fixture would be
# built BEFORE it — indexing a repo with a developer's `.env` in force, where
# TRELIX_FILE_SUMMARIES_ENABLED=true means the indexer summarises files through
# an LLM. Memoising keeps one index without moving the work outside isolation.
_INDEXED: pathlib.Path | None = None


@pytest.fixture()
def indexed_repo(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    global _INDEXED
    if _INDEXED is not None:
        return _INDEXED

    repo = tmp_path_factory.mktemp("intent_hint_outcome")
    _write_fixture(repo)
    from trelix.indexing.indexer import Indexer

    with (
        patch("trelix.indexing.indexer.make_embedder", return_value=_FakeEmbedder()),
        patch("trelix.indexing.indexer.make_vector_store", return_value=_FakeVectorStore()),
    ):
        Indexer(_make_config(repo), quiet=True).index()

    _INDEXED = repo
    return repo


def _build_retriever(repo: pathlib.Path) -> Any:
    from trelix.retrieval.retriever import Retriever

    with (
        patch("trelix.retrieval.retriever.make_embedder", return_value=_FakeEmbedder()),
        patch("trelix.retrieval.retriever.make_vector_store", return_value=_FakeVectorStore()),
        patch("trelix.retrieval.retriever.QueryPlanner", _StubPlanner),
    ):
        return Retriever(_make_config(repo))


@pytest.mark.parametrize("intent", list(IntentType), ids=lambda i: i.value)
def test_intent_hint_reaches_the_destination_its_strategy_names(
    indexed_repo: pathlib.Path, intent: IntentType
) -> None:
    """A hint must execute its own strategy, not the project-overview shortcut.

    Three assertions, all on the retrieval OUTCOME:
      1. the symbol a correct answer for this intent must contain is present;
      2. `total_available` (which both the REST and MCP envelopes define as
         `len(ctx.results)`) is no lower with the hint than without it — a hint
         is a narrowing instruction, never a reason to lose the answer;
      3. the router reached this intent's destination, and the legs actually
         handed to `_run_subquery_legs` are this intent's INTENT_STRATEGIES
         legs.
    """
    legs = INTENT_STRATEGIES[intent].legs
    # Keeps _DIRECT_PATHS honest: the direct-lookup intents are exactly the ones
    # whose strategy declares the file_direct leg.
    assert ("file_direct" in legs) is (intent in _DIRECT_PATHS)

    retriever = _build_retriever(indexed_repo)
    baseline: RetrievedContext = retriever.retrieve(_QUERY)

    plan = plan_from_intent_hint(_QUERY, intent.value)
    assert plan is not None

    leg_calls: list[list[str]] = []
    real_run_legs = retriever._run_subquery_legs

    def _record_legs(sq: Any, strategy: Any) -> Any:
        leg_calls.append(list(strategy.legs))
        return real_run_legs(sq, strategy)

    with (
        patch.object(retriever, "_run_subquery_legs", side_effect=_record_legs),
        patch.object(
            retriever, "_retrieve_project_overview", wraps=retriever._retrieve_project_overview
        ) as overview_spy,
        patch.object(
            retriever, "_retrieve_file_overview", wraps=retriever._retrieve_file_overview
        ) as file_overview_spy,
        patch.object(retriever, "_retrieve_config", wraps=retriever._retrieve_config) as config_spy,
        patch.object(
            retriever, "_retrieve_standard", wraps=retriever._retrieve_standard
        ) as standard_spy,
    ):
        hinted: RetrievedContext = retriever.retrieve(_QUERY, plan=plan)

    names = [r.symbol.name for r in hinted.results]
    expected = _MODULE_SYMBOL if intent is IntentType.PROJECT_OVERVIEW else _TARGET_FUNCTION
    assert expected in names, (
        f"intent_hint={intent.value!r} lost {expected!r}; got {names}. "
        f"The no-hint baseline returned {[r.symbol.name for r in baseline.results]}."
    )

    assert len(hinted.results) >= len(baseline.results), (
        f"intent_hint={intent.value!r} returned fewer results ({len(hinted.results)}) "
        f"than no hint at all ({len(baseline.results)})."
    )

    # The project-overview short-circuit is reachable for exactly one intent.
    assert overview_spy.called is (intent is IntentType.PROJECT_OVERVIEW), (
        f"intent_hint={intent.value!r} routed through _retrieve_project_overview="
        f"{overview_spy.called}"
    )

    spies = {
        "_retrieve_project_overview": overview_spy,
        "_retrieve_file_overview": file_overview_spy,
        "_retrieve_config": config_spy,
    }
    if intent in _DIRECT_PATHS:
        assert spies[_DIRECT_PATHS[intent]].called, (
            f"intent_hint={intent.value!r} declares the file_direct leg but "
            f"{_DIRECT_PATHS[intent]} never ran"
        )
        # file_overview and config_lookup then find no hints to look up — a hint
        # carries no file_hints/grep_hints — and widen to standard retrieval on
        # their own `if not results:` fallback, which runs default_plan()'s legs
        # rather than the intent's. That is why only the standard intents below
        # can assert leg equality.
    else:
        assert standard_spy.called, (
            f"intent_hint={intent.value!r} never reached the standard pipeline"
        )
        assert leg_calls == [legs], (
            f"intent_hint={intent.value!r} executed legs {leg_calls}, "
            f"expected exactly one call with INTENT_STRATEGIES legs {legs}"
        )
