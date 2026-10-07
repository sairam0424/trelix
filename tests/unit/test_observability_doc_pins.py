"""docs/OBSERVABILITY.md's load-bearing claims, pinned to code and to literals.

Two groups. The first is the claim about WHEN trelix installs its ``MeterProvider``. The second
(``TestLlmChatSpansSection``) pins the "LLM chat spans" section the C-8 chat spans added: the
provider map, the util-genai floor, the WARNING text and the error-text rule, each as a literal
here and, independently, as a literal in tests/unit/test_otel_llm_wrapper.py or
test_otel_llm_spans.py against the code, so the document and the code are two statements that
have to agree rather than one derived from the other. The last test pins the ``.env.example``
comment on ``TRELIX_OTEL_CAPTURE_CONTENT`` to docs/CONFIGURATION.md's wording now that the chat
spans exist.

The "Not instrumented" section says ``opentelemetry-util-genai`` records a
``gen_ai.client.operation.duration`` histogram per retrieval leg span once a ``MeterProvider``
exists, and that trelix installs one itself. A review of the spike PR found the first wording
("on the first embedding call") was an over-claim: the only path that installs a provider is
``otel_tracing.record_embedding_call()``, reached from ``_count_embed_call()`` in
``embedder/base.py`` and ``embedder/cohere.py``; ``bge_code.py`` and ``nomic_code.py`` never call
it (the document's own coverage table lists both as "not counted at all"), and a
``CachingEmbedder`` hit returns before any provider call. So a ``bge-code`` deployment, or a
process whose queries all hit the cache, exports the leg spans and records no duration unless the
host installs a provider.

Two pins. The first holds the sentence to its qualified form. The second holds the code fact the
sentence rests on: if either module gains the counter call, the sentence becomes wrong and this
test says so in the same commit.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_DOC = _ROOT / "docs" / "OBSERVABILITY.md"
_EMBEDDER = _ROOT / "src" / "trelix" / "embedder"

# A call STATEMENT, not any mention: cohere.py also names the function in a comment, and a
# probe that matched the comment let a mutant with the real call renamed survive.
_COUNTER_CALL = re.compile(r"^\s*_count_embed_call\(", re.MULTILINE)


def _bullet(marker: str) -> str:
    """One bullet of the 'Not instrumented' list, from *marker* to the start of the NEXT bullet,
    so an edit to an unrelated bullet cannot break a pin; falls back to the end of the paragraph
    when it is the last bullet."""
    text = _DOC.read_text(encoding="utf-8")
    start = text.index(marker)
    end = text.find("\n- **", start + 1)
    if end == -1:
        end = text.find("\n\n", start)
    return text[start:end]


def _histogram_bullet() -> str:
    """The 'Retrieval latency or throughput' bullet of the 'Not instrumented' list."""
    return _bullet("- **Retrieval latency or throughput**")


def _llm_chat_spans_section() -> str:
    """The '## LLM chat spans' section, up to the next '## ' heading."""
    text = _DOC.read_text(encoding="utf-8")
    start = text.index("\n## LLM chat spans\n")
    end = text.index("\n## ", start + 1)
    return text[start:end]


def _has_counter_call(name: str) -> bool:
    return _COUNTER_CALL.search((_EMBEDDER / name).read_text(encoding="utf-8")) is not None


def test_the_doc_conditions_the_meter_provider_install_on_a_counted_provider_call() -> None:
    """The sentence must name the condition and both exceptions, not 'the first embedding call'."""
    bullet = _histogram_bullet()
    assert "first counted embedding provider call" in bullet
    assert "every provider except `bge-code` and `nomic-code`, which" in bullet
    assert "never install one" in bullet
    assert "a `CachingEmbedder` hit is not a provider call" in bullet
    assert "on the first embedding call" not in bullet


def test_the_two_named_embedders_really_have_no_counter_call() -> None:
    """The exceptions the doc names are exactly the provider modules without the counter call."""
    for name in ("bge_code.py", "nomic_code.py"):
        assert not _has_counter_call(name), name
    # Control: the probe is the call statement the counted providers really make.
    for name in ("base.py", "cohere.py"):
        assert _has_counter_call(name), name


class TestLlmChatSpansSection:
    """The C-8 chat-span claims a reader acts on, pinned as literals (the code side of each is a
    literal in test_otel_llm_wrapper.py / test_otel_llm_spans.py)."""

    def test_the_provider_map_rows_are_the_wrappers(self) -> None:
        section = _llm_chat_spans_section()
        for row in (
            "| `openai` | `openai` |",
            "| `azure` | `azure.ai.openai` |",
            "| `anthropic` | `anthropic` |",
            "| `bedrock` | `aws.bedrock` |",
            "| `vertex` with `GOOGLE_API_KEY` | `gcp.gemini` |",
            "| `vertex` with `GOOGLE_CLOUD_PROJECT` | `gcp.vertex_ai` |",
            "| `litellm` | `litellm` (a custom value",
        ):
            assert row in section, row

    def test_the_doc_and_pyproject_name_the_same_util_genai_floor(self) -> None:
        """The floor is a decision readers act on (`pip install`); both sites say 1.2b0."""
        assert "`opentelemetry-util-genai>=1.2b0`" in _llm_chat_spans_section()
        with (_ROOT / "pyproject.toml").open("rb") as fh:
            otel_extra = tomllib.load(fh)["project"]["optional-dependencies"]["otel"]
        assert "opentelemetry-util-genai>=1.2b0" in otel_extra

    def test_the_unavailable_warning_is_quoted_verbatim(self) -> None:
        """Same literal the factory path logs (pinned against the code in
        test_otel_llm_wrapper.py), with `<reason>` where the cause goes."""
        assert (
            "TRELIX_OTEL_ENABLED is set but OpenTelemetry GenAI spans are unavailable (<reason>) — "
            "LLM calls will NOT be traced. Install: pip install 'trelix[otel]'"
        ) in _llm_chat_spans_section()

    def test_the_error_text_rule_and_the_retrieval_leg_gap_are_stated(self) -> None:
        section = _llm_chat_spans_section()
        assert "never `str(exc)`" in section
        assert "The retrieval leg spans still record `str(exc)`" in section

    def test_the_llm_tokens_bullet_points_at_the_chat_spans(self) -> None:
        """The old bullet said no counter exists and nothing carries usage; now the spans do."""
        bullet = _bullet("- **LLM tokens")
        assert "[chat span](#llm-chat-spans)" in bullet
        assert "`gen_ai.usage.*`" in bullet
        assert "`gen_ai.client.token.usage`" in bullet
        assert "none increments a counter" not in bullet


def test_env_example_says_the_capture_flag_covers_prompts_and_replies() -> None:
    """PR 2 wrote the sample config's comment forward ("prompts/replies once chat spans ship");
    the spans shipped, so it says what docs/CONFIGURATION.md says. An operator reading the stale
    line would conclude the flag does not yet export prompts, while `SPAN_ONLY` puts repository
    code on every `complete()` span."""
    text = (_ROOT / ".env.example").read_text(encoding="utf-8")
    assert (
        "# Hand prompts, replies and retrieval query text to the GenAI instrumentation (stream() "
        "replies\n# and images never); it includes repository code."
    ) in text
    assert "once chat spans ship" not in text
