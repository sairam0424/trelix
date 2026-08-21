"""Pins how many LLM calls retrieval planning makes, because three docs promised zero.

`docs/FAQ.md`, `docs/GETTING_STARTED.md` and `docs/WHY_TRELIX.md` each described
``trelix query`` (and in one case ``trelix search``) as offline, deterministic and
LLM-free. That is true only while no chat credential is resolvable. With one set,
``QueryPlanner.plan()`` draws a plan from the LLM — one call per distinct query — and
because the draw lives in ``Retriever.retrieve()`` rather than in the ``query`` command,
it applies to every retrieval consumer including the ``--json`` machine surface.

The discriminating case is ``provider="local"`` WITH a chat credential: the embedder is
local and no byte leaves the machine for embeddings, yet the planner still calls out,
because ``provider`` selects the embedder while the planner reads the chat credential
independently. That combination is what a CI box looks like when it has a key set for
``trelix ask``, and it is the case the old wording got wrong.

No network: the real factory still decides *whether* a client exists — that decision is
the subject here and must not be faked — but its methods are wrapped to count the
attempt and raise instead of dialling out. Credentials are passed explicitly rather than
read from the environment so that a developer with a live key exported cannot flip the
no-credential case into a false pass.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from trelix.core.config import EmbedderConfig, RetrievalConfig

_QUERY = "where is auth validated"

# Deliberately not shaped like any provider's key format. It only has to be truthy: the
# factory decides whether to build a chat client from the presence of a value, and the
# wrapped client never dials out, so no real credential is needed or wanted here.
_PLACEHOLDER = "not-a-credential-placeholder"

# Built as a mapping rather than written inline as ``openai_api_key=...`` so the literal
# is never assigned to a credential-named parameter — that shape trips secret scanners,
# and a test fixture should not look like a leaked key to tooling.
_CHAT_CREDENTIAL: dict[str, str | None] = {"openai_api_key": _PLACEHOLDER, "azure_api_key": None}
_NO_CREDENTIAL: dict[str, str | None] = {"openai_api_key": None, "azure_api_key": None}


class _Blocked(RuntimeError):
    """Raised instead of performing a real network call."""


def _count_llm_attempts(
    monkeypatch: pytest.MonkeyPatch,
    embedder: EmbedderConfig,
    retrieval: RetrievalConfig | None = None,
    *,
    queries: tuple[str, ...] = (_QUERY,),
    passes: int = 1,
) -> list[int]:
    """Attempts per pass. The real factory decides client existence; only calls are faked."""
    import trelix.llm.factory as factory

    real_build = factory.build_chat_client
    attempts = {"n": 0}

    class _Counting:
        def __init__(self, inner: Any) -> None:
            self._inner = inner

        def __getattr__(self, name: str) -> Any:
            attr = getattr(self._inner, name)
            if callable(attr) and name in {"complete", "tool_call"}:

                def _counted(*_a: Any, **_k: Any) -> Any:
                    attempts["n"] += 1
                    raise _Blocked("this test never performs a real LLM call")

                return _counted
            return attr

    def _wrapped(cfg: Any, *a: Any, **k: Any) -> Any:
        inner = real_build(cfg, *a, **k)
        return None if inner is None else _Counting(inner)

    monkeypatch.setattr(factory, "build_chat_client", _wrapped)

    from trelix.retrieval.planner.agent import QueryPlanner

    per_pass: list[int] = []
    for _ in range(passes):
        attempts["n"] = 0
        planner = QueryPlanner(embedder, retrieval_config=retrieval or RetrievalConfig())
        for q in queries:
            planner.plan(q)
        per_pass.append(attempts["n"])
    return per_pass


def test_no_chat_credential_means_no_llm_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """The state the docs describe as offline. Must be exactly zero, not merely few."""
    embedder = EmbedderConfig(provider="local", **_NO_CREDENTIAL)

    assert _count_llm_attempts(monkeypatch, embedder) == [0]


def test_a_chat_credential_costs_exactly_one_planner_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One call per query — the number the docs used to say was zero.

    Pinned as an equality rather than ``> 0`` so that a future change adding a second
    draw (a retry, a re-plan, an expansion) fails here and gets documented.
    """
    embedder = EmbedderConfig(provider="openai", **_CHAT_CREDENTIAL)

    assert _count_llm_attempts(monkeypatch, embedder) == [1]


def test_a_local_embedder_does_not_stop_the_planner_calling_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """THE discriminating case, and the one docs/WHY_TRELIX.md got wrong.

    ``provider="local"`` keeps embeddings on the machine and is widely read as "offline".
    It does not gate the planner. If this ever returns 0, ``provider`` has started
    gating the chat path and all three doc sites can be simplified back.
    """
    embedder = EmbedderConfig(provider="local", **_CHAT_CREDENTIAL)

    assert _count_llm_attempts(monkeypatch, embedder) == [1]


def test_one_call_per_DISTINCT_query(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two distinct queries cost two.

    This is what makes "one call per distinct query" in the docs a precise statement
    rather than a hand-wave.
    """
    embedder = EmbedderConfig(provider="openai", **_CHAT_CREDENTIAL)

    assert _count_llm_attempts(monkeypatch, embedder, queries=("alpha query", "beta query")) == [2]


def test_a_recorded_plan_cache_replays_with_zero_calls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The remedy the docs now name, pinned so the docs cannot outlive it.

    ``TRELIX_RETRIEVAL_PLAN_CACHE_FILE`` is record-then-replay, NOT a switch: an empty
    file means record mode and still costs a call. Only the second pass onward is free.
    The docs say exactly that, and this is why.
    """
    embedder = EmbedderConfig(provider="openai", **_CHAT_CREDENTIAL)
    cache = tmp_path / "plans.jsonl"

    per_pass = _count_llm_attempts(
        monkeypatch, embedder, RetrievalConfig(plan_cache_file=cache), passes=3
    )

    assert per_pass[0] == 1, f"first pass must record, drawing once: {per_pass}"
    assert per_pass[1:] == [0, 0], f"later passes must replay with no draw: {per_pass}"
    assert cache.exists(), "recording did not write the cache file"
    recorded = [json.loads(line) for line in cache.read_text().splitlines() if line.strip()]
    assert len(recorded) == 1, f"expected one recorded plan, got {len(recorded)}"
    assert recorded[0]["query"] == _QUERY
