"""
LLM-as-judge scorer for trelix retrieval quality evaluation.

Given a query, the retrieved code snippets, and the expected file, asks
an LLM to rate how well the retrieval answered the query.

The judge prompt asks for a JSON response:
    {"score": <float 0.0–1.0>, "reason": "<brief explanation>"}

score = 1.0 → retrieved result perfectly answers the query
score = 0.0 → retrieved result is completely irrelevant

This is distinct from recall@k (which checks exact file match) — it measures
semantic relevance, which can be high even when the exact file is missed.

Usage::

    from trelix.llm.factory import build_chat_client
    from trelix.core.config import LLMConfig
    from tests.eval.llm_judge import LLMJudge

    client = build_chat_client(LLMConfig())
    judge = LLMJudge(client)
    score = judge.score(query, snippets, expected_file)
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trelix.llm.client import TrelixChatClient

logger = logging.getLogger("trelix.eval.judge")

_JUDGE_SYSTEM = (
    "You are a code retrieval quality evaluator. "
    "Given a developer's query and retrieved code snippets, rate how well "
    "the retrieval answered the query. Return ONLY valid JSON."
)

_JUDGE_TEMPLATE = """\
Query: {query}

Expected file: {expected_file}

Retrieved code snippets:
{snippets}

Rate the quality of this retrieval on a scale of 0.0 to 1.0:
- 1.0: The retrieved code directly and completely answers the query
- 0.7: The retrieved code is highly relevant but incomplete
- 0.4: The retrieved code is somewhat relevant
- 0.1: The retrieved code is tangentially related
- 0.0: The retrieved code is irrelevant to the query

Respond with ONLY this JSON:
{{"score": <float>, "reason": "<one sentence>"}}"""


class LLMJudge:
    """LLM-as-judge retrieval quality scorer."""

    def __init__(self, client: TrelixChatClient) -> None:
        self._client = client

    def score(
        self,
        query: str,
        retrieved_snippets: list[str],
        expected_file: str,
        max_snippet_chars: int = 2000,
    ) -> float:
        """
        Score retrieval quality for a single query.

        Returns float in [0.0, 1.0]. Returns 0.0 on any LLM failure
        so eval runs never crash due to API errors.
        """
        snippets_text = "\n---\n".join(s[:max_snippet_chars] for s in retrieved_snippets[:5])
        prompt = _JUDGE_TEMPLATE.format(
            query=query,
            expected_file=expected_file,
            snippets=snippets_text,
        )
        try:
            from trelix.llm.client import ChatMessage

            response = self._client.complete(
                messages=[ChatMessage(role="user", content=prompt)],
                max_tokens=150,
                temperature=0.0,
                system=_JUDGE_SYSTEM,
            )
            data = json.loads(response.content)
            raw_score = float(data.get("score", 0.0))
            return max(0.0, min(1.0, raw_score))
        except Exception as exc:
            logger.warning("LLM judge failed for query %r: %s", query, exc)
            return 0.0
