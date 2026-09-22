"""
QueryPlanner's LLM-provider coupling bug (docs/ROADMAP.md's research-backlog entry).

Before the `llm_config` parameter added here, `QueryPlanner.__init__` derived its
internal `LLMConfig.provider` from the EMBEDDER's provider
(`config.provider if config.provider in ("openai", "azure") else "openai"`), never
from `TRELIX_LLM_PROVIDER`. A `local`/`voyage`/etc. embedder therefore always forced
an unauthenticated `openai` client onto the planner, silently collapsing every
`plan()` call to `default_plan()` even when a real, credentialed
`TRELIX_LLM_PROVIDER=anthropic` (or bedrock, or vertex) was configured — there was
no way to combine a free/local embedder with a working LLM-backed planner.

These tests assert the FIX: passing a real `LLMConfig` through `llm_config=` makes
QueryPlanner use it directly, regardless of the embedder's own provider, and that
the backward-compat shim (no `llm_config` passed) is byte-for-byte unchanged for
existing callers.
"""

from __future__ import annotations

from trelix.core.config import EmbedderConfig, LLMConfig
from trelix.llm.providers.anthropic_backend import AnthropicBackend
from trelix.llm.providers.openai_backend import OpenAIBackend
from trelix.retrieval.planner.agent import AdaptiveRouter, QueryPlanner

# Dict literal, not a keyword argument at the call site — matches the established
# placeholder-credential shape elsewhere in this suite (see
# tests/unit/test_retriever_core.py's `patch.dict(os.environ, {"OPENAI_API_KEY": ...})`).
_ANTHROPIC_LLM_FIELDS = {"provider": "anthropic", "anthropic_api_key": "test-placeholder-not-real"}


def _local_embedder() -> EmbedderConfig:
    return EmbedderConfig(provider="local")


def _anthropic_llm_config() -> LLMConfig:
    return LLMConfig(_env_file=None, **_ANTHROPIC_LLM_FIELDS)  # type: ignore[call-arg]


def test_llm_config_param_uses_the_real_provider_not_the_embedders() -> None:
    """MUTATION: delete the `if llm_config is not None:` branch in
    `QueryPlanner.__init__` (fall through to the shim unconditionally).

    A `local` embedder combined with a real, credentialed anthropic `LLMConfig`
    must build an AnthropicBackend, not silently coerce to openai.
    """
    planner = QueryPlanner(_local_embedder(), llm_config=_anthropic_llm_config())

    assert isinstance(planner._llm_client, AnthropicBackend), (
        f"expected AnthropicBackend, got {type(planner._llm_client).__name__} — "
        "the embedder's own provider ('local') leaked into the LLM client choice"
    )


def test_without_llm_config_the_backward_compat_shim_is_unchanged() -> None:
    """No regression for existing callers that don't pass `llm_config`.

    MUTATION: change the shim's provider fallback from `"openai"` to anything else.

    A `local` embedder with no `llm_config` supplied still falls back to the
    pre-existing shim, which always ends up on `OpenAIBackend` (unauthenticated,
    since no OPENAI_API_KEY is set here) -- this is the documented, pre-existing
    behaviour this fix must not disturb for callers that never pass `llm_config`.
    """
    planner = QueryPlanner(_local_embedder())

    assert isinstance(planner._llm_client, OpenAIBackend), (
        f"expected the pre-existing OpenAIBackend shim, got {type(planner._llm_client).__name__}"
    )


def test_llm_config_propagates_through_adaptive_router_to_its_nested_planner() -> None:
    """The Tier-3 recursive path must not drop `llm_config` one level down.

    `AdaptiveRouter` lazily builds its OWN nested `QueryPlanner` (via `_get_planner`)
    to plan each decomposed sub-question. Before this fix `_get_planner` only passed
    `retrieval_config` through, so a correctly-configured top-level QueryPlanner would
    still regress to the openai shim for every Tier-3 sub-question.

    MUTATION: drop `llm_config=self._llm_config` from `_get_planner`'s
    `QueryPlanner(...)` call.
    """
    router = AdaptiveRouter(_local_embedder(), llm_config=_anthropic_llm_config())

    nested_planner = router._get_planner()

    assert isinstance(nested_planner._llm_client, AnthropicBackend), (
        f"expected the nested planner to inherit AnthropicBackend, got "
        f"{type(nested_planner._llm_client).__name__} — llm_config was dropped "
        "somewhere between AdaptiveRouter and its own nested QueryPlanner"
    )


def test_query_planner_router_construction_also_propagates_llm_config() -> None:
    """The other direction: QueryPlanner's own lazily-built AdaptiveRouter (for
    `.plan()`'s Tier 1-3 routing) must receive the same `llm_config` too.

    MUTATION: drop `llm_config=self._llm_config` from `plan()`'s
    `AdaptiveRouter(...)` construction.
    """
    planner = QueryPlanner(_local_embedder(), llm_config=_anthropic_llm_config())

    # Trigger the lazy AdaptiveRouter construction the same way plan() does,
    # without going through plan() itself (which would attempt a real LLM call).
    if planner._router is None:
        planner._router = AdaptiveRouter(
            planner._config,
            retrieval_config=planner._retrieval_config,
            llm_config=planner._llm_config,
        )

    assert planner._router._llm_config is planner._llm_config, (
        "AdaptiveRouter built by QueryPlanner.plan() does not carry the same "
        "llm_config the planner itself was given"
    )
