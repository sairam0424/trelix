"""
Prompt-construction primitives shared by every trelix LLM call site.

Lives under ``trelix.llm`` rather than in ``agent/`` or ``review/`` because the
two consumers (``trelix.agent.loop`` and ``trelix.review.reviewer``) are peers:
neither imports the other, so a helper in either one would be a layering
inversion. Both already depend on ``trelix.llm.client`` for ``ChatMessage``, so
``trelix.llm`` is the nearest common ancestor. This module imports nothing from
trelix, so it cannot participate in an import cycle.
"""

from __future__ import annotations

import re

# Every maximal run of backticks anywhere in the payload. We deliberately do
# NOT restrict this to line-initial runs (which is all CommonMark treats as a
# closing fence) because the consumer is an LLM, not a spec-compliant parser:
# models routinely treat an inline ``` as a block boundary. Over-counting costs
# one extra backtick; under-counting silently truncates the payload.
_BACKTICK_RUN_RE = re.compile(r"`+")

# CommonMark requires an opening code fence of at least three backticks.
MIN_FENCE_LENGTH = 3


def fence_for(payload: str) -> str:
    """
    Return the shortest code fence that can safely wrap ``payload``.

    CommonMark ends a fenced block at the first fence that is *at least as long
    as* the opening one, so a hard-coded three-backtick fence is broken by any
    payload that itself contains ``` — the fence closes early and everything
    after it stops being quoted code. That is not a hypothetical: trelix indexes
    markdown (``MarkdownParser`` is in the parser registry) and 72 of the 79
    tracked markdown files in this repo contain a three-backtick run, so a
    markdown symbol body pasted into a prompt terminates its own fence on
    ordinary use, not just under attack.

    Returns ``max(3, longest_backtick_run + 1)`` backticks.
    """
    longest = max((len(m.group()) for m in _BACKTICK_RUN_RE.finditer(payload)), default=0)
    return "`" * max(MIN_FENCE_LENGTH, longest + 1)


def fenced_block(payload: str) -> str:
    """
    Wrap ``payload`` in a fence long enough to survive its own backticks.

    For any payload whose longest backtick run is under three (the overwhelming
    majority — source code), the result is byte-identical to the historical
    ``f"```\\n{payload}\\n```"`` this replaces, which is what makes it safe to
    drop in at existing call sites without changing rendered prompts.
    """
    fence = fence_for(payload)
    return f"{fence}\n{payload}\n{fence}"
