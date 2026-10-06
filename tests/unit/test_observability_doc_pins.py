"""docs/OBSERVABILITY.md's claim about WHEN trelix installs its ``MeterProvider``, pinned to code.

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
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_DOC = _ROOT / "docs" / "OBSERVABILITY.md"
_EMBEDDER = _ROOT / "src" / "trelix" / "embedder"

# A call STATEMENT, not any mention: cohere.py also names the function in a comment, and a
# probe that matched the comment let a mutant with the real call renamed survive.
_COUNTER_CALL = re.compile(r"^\s*_count_embed_call\(", re.MULTILINE)


def _histogram_bullet() -> str:
    """The 'Retrieval latency or throughput' bullet of the 'Not instrumented' list."""
    text = _DOC.read_text(encoding="utf-8")
    start = text.index("- **Retrieval latency or throughput**")
    # Slice to the start of the NEXT bullet, so an edit to an unrelated bullet cannot break
    # this pin; fall back to the end of the paragraph when it is the last bullet.
    end = text.find("\n- **", start + 1)
    if end == -1:
        end = text.find("\n\n", start)
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
