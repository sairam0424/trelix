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


_OUTBOUND = frozenset({"complete", "tool_call", "stream", "create"})
_RAW_CLIENT_ATTRS = frozenset({"_client", "client"})


class _NoNetwork:
    """Stands in for the raw provider client. Any traversal chains; any call is blocked.

    `_plan_direct` has a legacy branch (agent.py:494-505) that reaches straight through to
    `planner._client.chat.completions.create(...)`. An earlier version of this wrapper let
    `_client` through untouched — its `__getattr__` only intercepted `complete` and
    `tool_call` and returned everything else verbatim — so that branch would have made a REAL
    request with whatever key happened to be exported, while the counter stayed at zero and
    this module's docstring still promised "no network". No test reached it (every query here
    is Tier 2), which is why it went unnoticed until an audit probed a Tier-3 query and got a
    live 401 back from the provider.

    Chaining rather than raising on attribute access is deliberate: raising on `.chat` would
    make the branch fail for the wrong reason and hide whether the call site was reached.

    Module-level rather than nested inside the helper so the guard itself is testable — the
    regression it exists to prevent was invisible precisely because nothing could reach it.
    """

    def __init__(self, counter: dict[str, int]) -> None:
        self._counter = counter

    def __getattr__(self, _name: str) -> Any:
        return self

    def __call__(self, *_a: Any, **_k: Any) -> Any:
        self._counter["n"] += 1
        raise _Blocked("this test never performs a real LLM call")


class _Counting:
    """Wraps a real chat client: counts outbound calls, never lets one leave the machine."""

    def __init__(self, inner: Any, counter: dict[str, int]) -> None:
        self._inner = inner
        self._counter = counter

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._inner, name)
        # Neutralise the raw provider client without changing whether one EXISTS. The real
        # factory's decision is the subject of these tests, so a None stays None —
        # substituting a stand-in unconditionally made the no-credential case report 1 call
        # instead of 0, i.e. the wrapper lying about the thing under test.
        if name in _RAW_CLIENT_ATTRS:
            return None if attr is None else _NoNetwork(self._counter)
        if callable(attr) and name in _OUTBOUND:

            def _counted(*_a: Any, **_k: Any) -> Any:
                self._counter["n"] += 1
                raise _Blocked("this test never performs a real LLM call")

            return _counted
        return attr


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

    def _wrapped(cfg: Any, *a: Any, **k: Any) -> Any:
        inner = real_build(cfg, *a, **k)
        return None if inner is None else _Counting(inner, attempts)

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


# ---------------------------------------------------------------------------
# The harness's own no-network guarantee
# ---------------------------------------------------------------------------


class TestTheHarnessCannotDialOut:
    """Pins the wrapper's own promise, because it was false once and nothing noticed.

    This module's docstring claims "no network". That held only for the two methods the
    wrapper intercepted. `AdaptiveRouter._decompose_via_llm` reaches the RAW provider client
    at `planner._client.chat.completions.create(...)`, and the wrapper handed `_client` back
    untouched — so a Tier-3-shaped query would have made a real request with whatever key was
    exported, and still reported zero calls. An audit found it by getting a live 401.

    A guard whose failure mode is "spends money silently and lies about it" needs a test.
    """

    def test_the_raw_client_chain_is_counted_instead_of_dialling_out(self) -> None:
        """`_client.chat.completions.create(...)` must raise, and must be counted."""
        counter = {"n": 0}
        client = _NoNetwork(counter)

        with pytest.raises(_Blocked):
            client.chat.completions.create(model="x", messages=[])

        assert counter["n"] == 1, "the raw-client path was not counted"

    def test_an_arbitrarily_deep_attribute_chain_still_blocks(self) -> None:
        """Chaining must not run out — a provider SDK may nest differently than we expect."""
        counter = {"n": 0}

        with pytest.raises(_Blocked):
            _NoNetwork(counter).a.b.c.d.e(1, kw=2)

        assert counter["n"] == 1

    def test_a_present_raw_client_is_replaced_but_an_absent_one_stays_absent(self) -> None:
        """The wrapper must not change WHETHER a client exists — only what a call does.

        Substituting a stand-in unconditionally made the no-credential case report one call
        instead of zero, since `_plan_direct` gates on `planner._client is not None`. That is
        the wrapper lying about the very quantity these tests measure.
        """
        counter = {"n": 0}

        class _InnerWithClient:
            _client = object()

        class _InnerWithout:
            _client = None

        assert isinstance(_Counting(_InnerWithClient(), counter)._client, _NoNetwork)
        assert _Counting(_InnerWithout(), counter)._client is None

    # Written out as a literal, NOT read from `_OUTBOUND`. The test below used to iterate the
    # constant itself, so deleting a name from `_OUTBOUND` just made it check fewer names and
    # still pass — it validated against the thing it was meant to pin. Mutation caught that:
    # removing "create" was invisible across all nine tests. If a provider gains another
    # send-shaped method, add it in BOTH places; the test above will say so if you don't.
    EXPECTED_OUTBOUND = frozenset({"complete", "tool_call", "stream", "create"})

    def test_the_outbound_set_has_not_silently_shrunk(self) -> None:
        """Guards the constant, independently of the constant."""
        assert _OUTBOUND == self.EXPECTED_OUTBOUND, (
            "the set of intercepted send-shaped methods changed; if that is deliberate, update "
            f"EXPECTED_OUTBOUND too: {sorted(_OUTBOUND)} != {sorted(self.EXPECTED_OUTBOUND)}"
        )

    def test_every_outbound_method_name_is_intercepted(self) -> None:
        """Each send-shaped method must raise, be counted, and never reach the inner object."""
        counter = {"n": 0}

        class _Inner:
            def complete(self, *a: Any, **k: Any) -> str:
                return "sent"

            def tool_call(self, *a: Any, **k: Any) -> str:
                return "sent"

            def stream(self, *a: Any, **k: Any) -> str:
                return "sent"

            def create(self, *a: Any, **k: Any) -> str:
                return "sent"

            def harmless(self) -> str:
                return "local"

        wrapped = _Counting(_Inner(), counter)
        for name in sorted(self.EXPECTED_OUTBOUND):
            with pytest.raises(_Blocked):
                getattr(wrapped, name)()
        assert counter["n"] == len(self.EXPECTED_OUTBOUND)
        # A non-sending attribute must pass through untouched, or the wrapper would break
        # unrelated behaviour and the counts would stop meaning anything.
        assert wrapped.harmless() == "local"
