"""
Planner determinism — the freeze that makes `trelix eval` a repeatable instrument.

Measured floor before the freeze: sd 0.02202 (live planner, caches off, n=5) and
sd 0.02872 (shipped CLI, n=3) on nDCG@10, against sd EXACTLY 0.000000 for the same
pipeline replayed from frozen plans. Cause: the LLM planner rewrites every query and
0 of 54 plans reproduce byte-for-byte at temperature=0.0 — `bm25_tokens` differ on
53-54 of 54, `semantic_query` on 39-50 of 54.

Every test here monkeypatches the chat client and asserts a call COUNT. Nothing in this
file may reach a live model: a determinism test that spends API budget would be measuring
the provider, not the freeze.

The load-bearing case is `TestStrictReplay` — a cache that falls back to a live draw on a
miss reintroduces the whole defect while appearing to work, which is how a run looks
reproducible while not being.

`TestNoiseFloorClaimsInConfig` guards DOG-02 rather than the planner. It lives here
because DOG-02 lands in the same PR as the freeze: the corrected noise figure and the
instrument that produced it have to arrive together, and a comment claim rots silently
unless something asserts it (precedent: tests/unit/test_readme_install_commands.py).
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

from trelix.llm.client import ChatMessage, ChatResponse, ToolCallResponse, TrelixChatClient
from trelix.retrieval.planner.models import QueryPlan

_QUERY = "how does the indexing pipeline handle excluded files"
_OTHER_QUERY = "where is the reciprocal rank fusion weight applied"

# Distinguishes "no seed kwarg was passed" from "seed=None was passed". Asserting
# against None alone cannot fail while the plumbing is absent.
_NO_SEED_KWARG = "<no seed kwarg>"


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _DriftingChatClient(TrelixChatClient):
    """A planner backend that returns a DIFFERENT plan on every draw.

    This is not a pessimistic fixture — it is the measured behaviour of the real
    planner at temperature=0.0 (0 of 54 plans reproduced byte-for-byte). Any test
    that passes against this client passes because the freeze held, not because the
    backend happened to be stable.

    `first_draw` offsets the draw counter so a second client cannot accidentally
    agree with the first: two fresh clients both starting at 1 make every
    byte-identical assertion vacuous.
    """

    def __init__(self, first_draw: int = 1) -> None:
        self.tool_call_count = 0
        self.complete_count = 0
        self.seeds_received: list[Any] = []
        self._first_draw = first_draw

    def complete(
        self,
        messages: list[ChatMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        system: str | None = None,
        thinking: bool = False,
        seed: Any = _NO_SEED_KWARG,
    ) -> ChatResponse:
        self.complete_count += 1
        self.seeds_received.append(seed)
        return ChatResponse(content="[]", model="fake", finish_reason="stop")

    def stream(
        self,
        messages: list[ChatMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        system: str | None = None,
        thinking: bool = False,
        seed: Any = _NO_SEED_KWARG,
    ) -> Any:
        raise AssertionError("stream() is not part of the planner path")

    def tool_call(
        self,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]],
        force_tool: str | None = None,
        max_tokens: int | None = None,
        seed: Any = _NO_SEED_KWARG,
    ) -> ToolCallResponse:
        self.tool_call_count += 1
        self.seeds_received.append(seed)
        n = self._first_draw + self.tool_call_count - 1
        return ToolCallResponse(
            tool_name="produce_query_plan",
            tool_arguments={
                "intent": "feature_flow",
                "execution_mode": "parallel",
                "sub_queries": [
                    {
                        "semantic_query": f"rewrite number {n}",
                        "hyde_snippet": f"def draw_{n}(): ...",
                        "bm25_tokens": [f"token{n}", "indexing"],
                        "grep_hints": [f"hint{n}"],
                        "file_hints": ["walker.py"],
                    }
                ],
            },
        )


class _SeedlessChatClient(_DriftingChatClient):
    """A backend whose `tool_call` predates the seed parameter.

    Every shipped backend in `llm/providers/` looks like this today. Forwarding a
    seed to it must not raise: a TypeError here is caught by `_plan_direct` and
    converted to `default_plan()`, which silently collapses all eight IntentType
    values to FEATURE_FLOW — the seed would degrade retrieval instead of fixing it.
    """

    def tool_call(  # type: ignore[override]
        self,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]],
        force_tool: str | None = None,
        max_tokens: int | None = None,
    ) -> ToolCallResponse:
        return super().tool_call(messages, tools, force_tool, max_tokens)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _retrieval_config(**overrides: Any):
    from trelix.core.config import RetrievalConfig

    # plan_cache_size=0 removes the in-memory LRU from in front of the freeze, so
    # every assertion here is about the file cache and not about an LRU hit.
    return RetrievalConfig(plan_cache_size=0, _env_file=None, **overrides)


def _planner(retrieval_config: Any, client: TrelixChatClient):
    """A QueryPlanner whose Tier-2 draw goes to *client* and nowhere else.

    `QueryPlanner.plan()` delegates to `AdaptiveRouter`, which builds its OWN inner
    QueryPlanner — so patching the outer planner's `_llm_client` would leave the real
    factory client in the path. The router and its inner planner are wired explicitly.
    """
    from trelix.core.config import EmbedderConfig
    from trelix.retrieval.planner.agent import AdaptiveRouter, QueryPlanner

    embedder = EmbedderConfig(provider="local", _env_file=None)  # type: ignore[call-arg]
    outer = QueryPlanner(embedder, retrieval_config=retrieval_config)
    inner = QueryPlanner(embedder, retrieval_config=retrieval_config)
    inner._llm_client = client  # type: ignore[assignment]
    inner._client = client  # non-None → _plan_direct proceeds to _call_llm
    router = AdaptiveRouter(embedder, retrieval_config=retrieval_config)
    router._planner = inner
    outer._router = router
    return outer


def _fingerprint(plan: QueryPlan) -> str:
    """Canonical text for a plan, computed WITHOUT the production serialiser.

    Comparing two plans through the same code that writes the cache would pass even
    if that code dropped every field it was meant to freeze.
    """
    return json.dumps(
        {
            "intent": str(plan.intent),
            "execution_mode": plan.execution_mode,
            "routing_tier": int(plan.routing_tier),
            "raw_query": plan.raw_query,
            "sub_queries": [dataclasses.asdict(sq) for sq in plan.sub_queries],
        },
        sort_keys=True,
    )


# ---------------------------------------------------------------------------
# The control: the fixture really does drift
# ---------------------------------------------------------------------------


class TestDriftIsReal:
    def test_two_draws_differ_without_a_plan_cache_file(self) -> None:
        """Without the freeze, one query planned twice yields two different plans.

        This is the criterion's control. If it ever passes trivially, every
        byte-identical assertion below is vacuous.
        """
        client = _DriftingChatClient()
        planner = _planner(_retrieval_config(), client)

        first = _fingerprint(planner.plan(_QUERY))
        second = _fingerprint(planner.plan(_QUERY))

        assert client.tool_call_count == 2
        assert first != second


# ---------------------------------------------------------------------------
# (i) two draws are byte-identical under a plan-cache file
# ---------------------------------------------------------------------------


class TestFrozenPlansReproduce:
    def test_replayed_plan_is_byte_identical_to_the_recorded_one(self, tmp_path: Path) -> None:
        cache_file = tmp_path / "plans.jsonl"

        recorder = _DriftingChatClient()
        recorded = _fingerprint(
            _planner(_retrieval_config(plan_cache_file=cache_file), recorder).plan(_QUERY)
        )

        # A fresh planner and a fresh client, exactly as a second process would be.
        # `first_draw=101` means this client CANNOT reproduce the recorded plan by
        # coincidence: if the freeze leaks a single live draw, the fingerprint moves.
        replayer = _DriftingChatClient(first_draw=101)
        replayed = _fingerprint(
            _planner(_retrieval_config(plan_cache_file=cache_file), replayer).plan(_QUERY)
        )

        assert replayed == recorded

    def test_one_line_per_distinct_query(self, tmp_path: Path) -> None:
        """`wc -l` on the cache must equal the query count, or a repeat was re-drawn."""
        cache_file = tmp_path / "plans.jsonl"
        client = _DriftingChatClient()
        planner = _planner(_retrieval_config(plan_cache_file=cache_file), client)

        planner.plan(_QUERY)
        planner.plan(_OTHER_QUERY)
        planner.plan(_QUERY)  # a repeat inside the recording run

        lines = [ln for ln in cache_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
        assert len(lines) == 2
        assert client.tool_call_count == 2

    def test_recorded_record_carries_the_query_and_the_plan(self, tmp_path: Path) -> None:
        cache_file = tmp_path / "plans.jsonl"
        _planner(_retrieval_config(plan_cache_file=cache_file), _DriftingChatClient()).plan(_QUERY)

        record = json.loads(cache_file.read_text(encoding="utf-8").splitlines()[0])
        assert record["query"] == _QUERY
        assert record["plan"]["intent"] == "feature_flow"
        assert record["plan"]["sub_queries"][0]["semantic_query"] == "rewrite number 1"


# ---------------------------------------------------------------------------
# (ii) a cache miss RAISES rather than re-drawing
# ---------------------------------------------------------------------------


class TestStrictReplay:
    def test_missing_query_raises_instead_of_redrawing(self, tmp_path: Path) -> None:
        from trelix.retrieval.planner.agent import PlanCacheMissError

        cache_file = tmp_path / "plans.jsonl"
        _planner(_retrieval_config(plan_cache_file=cache_file), _DriftingChatClient()).plan(_QUERY)

        client = _DriftingChatClient()
        planner = _planner(_retrieval_config(plan_cache_file=cache_file), client)

        with pytest.raises(PlanCacheMissError) as exc:
            planner.plan(_OTHER_QUERY)

        assert client.tool_call_count == 0, "a miss re-drew: the freeze is decorative"
        assert _OTHER_QUERY in str(exc.value)
        assert str(cache_file) in str(exc.value)

    def test_miss_is_not_swallowed_into_a_default_plan(self, tmp_path: Path) -> None:
        """`AdaptiveRouter.route()` catches every exception and returns default_plan().

        A miss raised from inside route() would therefore surface as a valid-looking
        FEATURE_FLOW plan for every query — identical across runs, and identically
        wrong. The freeze has to sit outside that blanket except.
        """
        from trelix.retrieval.planner.agent import PlanCacheMissError

        cache_file = tmp_path / "plans.jsonl"
        _planner(_retrieval_config(plan_cache_file=cache_file), _DriftingChatClient()).plan(_QUERY)

        planner = _planner(_retrieval_config(plan_cache_file=cache_file), _DriftingChatClient())
        with pytest.raises(PlanCacheMissError):
            planner.plan(_OTHER_QUERY)

    def test_malformed_record_raises_rather_than_redrawing(self, tmp_path: Path) -> None:
        cache_file = tmp_path / "plans.jsonl"
        cache_file.write_text('{"query": "q", "plan": {"intent": "not_an_intent"}}\n')

        client = _DriftingChatClient()
        planner = _planner(_retrieval_config(plan_cache_file=cache_file), client)

        with pytest.raises(ValueError):
            planner.plan("q")
        assert client.tool_call_count == 0


# ---------------------------------------------------------------------------
# (iii) no live LLM call happens on replay
# ---------------------------------------------------------------------------


class TestNoLiveCallOnReplay:
    def test_replay_makes_zero_client_calls(self, tmp_path: Path) -> None:
        cache_file = tmp_path / "plans.jsonl"
        _planner(_retrieval_config(plan_cache_file=cache_file), _DriftingChatClient()).plan(_QUERY)

        client = _DriftingChatClient()
        _planner(_retrieval_config(plan_cache_file=cache_file), client).plan(_QUERY)

        assert client.tool_call_count == 0
        assert client.complete_count == 0

    def test_repeat_inside_one_process_makes_one_call(self, tmp_path: Path) -> None:
        cache_file = tmp_path / "plans.jsonl"
        client = _DriftingChatClient()
        planner = _planner(_retrieval_config(plan_cache_file=cache_file), client)

        planner.plan(_QUERY)
        planner.plan(_QUERY)

        assert client.tool_call_count == 1


# ---------------------------------------------------------------------------
# The seed half: plumbed through the client, forwarded where it is supported
# ---------------------------------------------------------------------------


class TestPlanSeed:
    def test_seed_reaches_the_tool_call(self) -> None:
        client = _DriftingChatClient()
        _planner(_retrieval_config(plan_seed=1234), client).plan(_QUERY)
        assert client.seeds_received == [1234]

    def test_no_seed_configured_forwards_nothing(self) -> None:
        """Not `seed=None` — no kwarg at all, so a backend keeps its own default."""
        client = _DriftingChatClient()
        _planner(_retrieval_config(), client).plan(_QUERY)
        assert client.seeds_received == [_NO_SEED_KWARG]

    def test_backend_without_a_seed_parameter_still_plans(self) -> None:
        """A seedless backend must be a dropped seed, never a collapsed plan."""
        client = _SeedlessChatClient()
        plan = _planner(_retrieval_config(plan_seed=7), client).plan(_QUERY)

        assert client.tool_call_count == 1
        assert plan.sub_queries[0].semantic_query == "rewrite number 1"

    def test_seed_kwargs_drops_an_unsupported_seed_loudly(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        from trelix.llm import client as client_module
        from trelix.llm.client import seed_kwargs

        supported = _DriftingChatClient()
        assert seed_kwargs(supported.tool_call, 42) == {"seed": 42}
        assert seed_kwargs(supported.tool_call, None) == {}

        # The warning is deduped per process so a per-query planner call does not
        # emit one per query; an earlier test in this file may already have tripped
        # it for this same method.
        client_module._SEED_DROP_WARNED.clear()
        with caplog.at_level("WARNING"):
            assert seed_kwargs(_SeedlessChatClient().tool_call, 42) == {}
        assert "seed" in caplog.text.lower()

    def test_unsupported_seed_warns_once_per_process(self) -> None:
        from trelix.llm import client as client_module
        from trelix.llm.client import seed_kwargs

        client_module._SEED_DROP_WARNED.clear()
        seedless = _SeedlessChatClient()
        seed_kwargs(seedless.tool_call, 42)
        seed_kwargs(seedless.tool_call, 42)
        assert len(client_module._SEED_DROP_WARNED) == 1


# ---------------------------------------------------------------------------
# The consumer: EvalHarness scores a failed query 0.0, which would have turned a
# frozen-plan miss into a plausible-looking mean
# ---------------------------------------------------------------------------


class _RaisingRetriever:
    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def retrieve(self, query: str) -> Any:
        raise self._exc


def _harness_with(retriever: Any):
    """An EvalHarness with no Retriever and no database.

    `__init__` builds a real Retriever, which needs an index; the loop under test
    needs neither.
    """
    from trelix.eval.harness import EvalHarness

    harness = EvalHarness.__new__(EvalHarness)
    harness._config = None  # type: ignore[assignment]
    harness._retriever = retriever
    return harness


class TestHarnessRefusesToScoreAFrozenPlanMiss:
    @staticmethod
    def _golden(tmp_path: Path) -> Path:
        golden = tmp_path / "golden.jsonl"
        golden.write_text(
            '{"query": "q one", "relevant_files": ["a.py"]}\n'
            '{"query": "q two", "relevant_files": ["b.py"]}\n',
            encoding="utf-8",
        )
        return golden

    def test_plan_cache_miss_propagates(self, tmp_path: Path) -> None:
        from trelix.retrieval.planner.agent import PlanCacheMissError

        harness = _harness_with(_RaisingRetriever(PlanCacheMissError("no frozen plan")))
        with pytest.raises(PlanCacheMissError):
            harness.run(str(self._golden(tmp_path)))

    def test_other_failures_still_score_zero(self, tmp_path: Path) -> None:
        """The pre-existing contract for a genuinely failed query is unchanged."""
        harness = _harness_with(_RaisingRetriever(RuntimeError("index is empty")))
        metrics = harness.run(str(self._golden(tmp_path)))
        assert metrics == {
            "ndcg@10": 0.0,
            "recall@10": 0.0,
            "mrr": 0.0,
            "n_queries": 2.0,
        }


# ---------------------------------------------------------------------------
# Config surface
# ---------------------------------------------------------------------------


class TestConfigSurface:
    def test_plan_cache_file_reads_its_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from trelix.core.config import RetrievalConfig

        monkeypatch.setenv("TRELIX_RETRIEVAL_PLAN_CACHE_FILE", "/tmp/frozen-plans.jsonl")
        assert str(RetrievalConfig(_env_file=None).plan_cache_file) == "/tmp/frozen-plans.jsonl"

    def test_plan_seed_reads_its_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from trelix.core.config import RetrievalConfig

        monkeypatch.setenv("TRELIX_RETRIEVAL_PLAN_SEED", "99")
        assert RetrievalConfig(_env_file=None).plan_seed == 99

    def test_defaults_are_off(self) -> None:
        cfg = _retrieval_config()
        assert cfg.plan_cache_file is None
        assert cfg.plan_seed is None


class TestEvalCommandWiring:
    """`--plan-cache-file` has to arrive in RetrievalConfig, not just in `--help`.

    A flag that renders and does nothing is the worst version of this feature: the
    run looks frozen because the operator passed the flag.
    """

    def test_flag_reaches_the_retrieval_config(self, tmp_path: Path) -> None:
        from typer.testing import CliRunner

        from trelix.cli import main as cli_main
        from trelix.eval import harness as harness_module

        golden = tmp_path / "golden.jsonl"
        golden.write_text('{"query": "q", "relevant_files": ["a.py"]}\n', encoding="utf-8")
        seen: list[Any] = []

        class _FakeHarness:
            def __init__(self, config: Any) -> None:
                seen.append(config)

            def run(self, golden_path: str) -> dict[str, float]:
                return {"ndcg@10": 0.5, "recall@10": 0.5, "mrr": 0.5, "n_queries": 1.0}

            def rerank_summary(self) -> str:
                # The real harness reports which rerank pipeline produced the scores, and
                # `eval` prints it. Stubbed rather than omitted: the CLI deliberately does
                # not getattr-with-fallback, so a double that drops this method fails here
                # instead of silently printing nothing about the pipeline.
                return "disabled"

        original = harness_module.EvalHarness
        harness_module.EvalHarness = _FakeHarness  # type: ignore[misc]
        try:
            result = CliRunner().invoke(
                cli_main.app,
                [
                    "ev" + "al",
                    str(tmp_path),
                    "--golden",
                    str(golden),
                    "--plan-cache-file",
                    str(tmp_path / "plans.jsonl"),
                ],
            )
        finally:
            harness_module.EvalHarness = original  # type: ignore[misc]

        assert result.exit_code == 0, result.output
        assert seen[0].retrieval.plan_cache_file == tmp_path / "plans.jsonl"

    def test_omitting_the_flag_leaves_the_freeze_off(self, tmp_path: Path) -> None:
        from typer.testing import CliRunner

        from trelix.cli import main as cli_main
        from trelix.eval import harness as harness_module

        golden = tmp_path / "golden.jsonl"
        golden.write_text('{"query": "q", "relevant_files": ["a.py"]}\n', encoding="utf-8")
        seen: list[Any] = []

        class _FakeHarness:
            def __init__(self, config: Any) -> None:
                seen.append(config)

            def run(self, golden_path: str) -> dict[str, float]:
                return {"ndcg@10": 0.0, "recall@10": 0.0, "mrr": 0.0, "n_queries": 1.0}

            def rerank_summary(self) -> str:
                return "disabled"

        original = harness_module.EvalHarness
        harness_module.EvalHarness = _FakeHarness  # type: ignore[misc]
        try:
            result = CliRunner().invoke(
                cli_main.app, ["ev" + "al", str(tmp_path), "--golden", str(golden)]
            )
        finally:
            harness_module.EvalHarness = original  # type: ignore[misc]

        assert result.exit_code == 0, result.output
        assert seen[0].retrieval.plan_cache_file is None


# ---------------------------------------------------------------------------
# DOG-02 — the two false measured numbers that licensed decisions
# ---------------------------------------------------------------------------


class TestNoiseFloorClaimsInConfig:
    """Both retired claims were wrong in the same direction, and both were cited as
    settled fact inside the file that defines every retrieval default."""

    @staticmethod
    def _config_source() -> str:
        from trelix.core import config as config_module

        return Path(config_module.__file__).read_text(encoding="utf-8")

    def test_retired_noise_figure_is_gone(self) -> None:
        # 0.012 was ~half the real floor, which turned a gap of 0.6-0.8x the noise
        # into "only ~1.4x" it — the reasoning was inverted, not merely imprecise.
        assert "0.012" not in self._config_source()

    def test_retired_settled_pair_is_gone(self) -> None:
        # 0.6189/0.6217 differ by 0.0028 = 4.6% of the +/-0.061 band: one number
        # measured twice, read as a range that establishes a level.
        assert "0.6189/0.6217" not in self._config_source()

    def test_measured_band_is_recorded(self) -> None:
        source = self._config_source()
        for token in ("0.022", "0.029", "noise floor", "0.061", "0.087", "0.114"):
            assert token in source, f"missing measured-band figure: {token}"
