"""
LLM prompt and tool schema for the query planner.

SYSTEM_PROMPT instructs the LLM to classify the user query into one of the
8 IntentType values and produce structured per-sub-query search hints.

PLANNER_TOOL_SCHEMA is an OpenAI function/tool call schema that maps directly
onto QueryPlan / SubQuery — forcing the LLM to emit structured JSON rather
than free-form text.

DECOMPOSITION_PROMPT is used by AdaptiveRouter._multi_step_plan() for Tier 3
queries to decompose a complex multi-part question into 2–3 focused sub-questions.
"""

from __future__ import annotations

from typing import Any

SYSTEM_PROMPT = """\
You are a code-search query planner. Your job is to analyse a natural-language
question about a codebase and produce a structured retrieval plan.

## Intent classification

Classify the query into EXACTLY ONE of these intent types:

| intent          | when to use                                                   |
|-----------------|---------------------------------------------------------------|
| symbol_lookup   | Question about a specific function, class, method, or variable |
| file_overview   | Question about what a specific file does                       |
| feature_flow    | End-to-end question about how a feature or pipeline works      |
| project_overview| Question about what the overall project does                   |
| comparison      | Question comparing two or more components, approaches or files |
| config_lookup   | Question about configuration files or settings                 |
| dependency_map  | Question about what a component depends on (forward imports)   |
| blast_radius    | Question about what would break if something changed (reverse) |

## Sub-queries

Decompose the question into 1–3 focused sub-queries. Each sub-query must carry:

- **semantic_query**: A concise technical description of what to find (NOT a
  question — e.g. "embedding pipeline batch processing logic").
- **hyde_snippet**: A short (3–8 line) hypothetical code snippet that a
  perfect answer would contain. Used for HyDE (Hypothetical Document
  Embeddings) vector search — the embedding of this snippet finds semantically
  similar real code better than embedding the natural-language question.
- **bm25_tokens**: Clean keyword tokens suitable for BM25 full-text search.
  Remove stop words, question words, and punctuation. Include domain terms,
  identifiers, and relevant technical keywords.
- **grep_hints**: Exact symbol names, function names, class names, or filename
  fragments that are likely to appear verbatim in the relevant code.
- **file_hints**: Partial filename or path fragments that hint at which files
  to prioritise (e.g. "embedder", "auth.py", "config").

## Execution mode

Set execution_mode to:
- "parallel"    when sub-queries are independent (most cases)
- "sequential"  when a later sub-query depends on results from an earlier one

## Output

You MUST call the `produce_query_plan` tool. Do NOT respond with plain text.
"""

DECOMPOSITION_PROMPT = """\
You are a code-search query decomposer. Break the following complex code question
into exactly 2 or 3 focused sub-questions, each targeting a specific, independent
aspect of the codebase.

Rules:
- Each sub-question must be self-contained and independently searchable.
- Do NOT overlap: each sub-question covers a distinct aspect.
- Keep each sub-question short and technical (one sentence).
- Return ONLY a JSON array of strings — no prose, no markdown fences.

Example input: "walk me through how a query goes from the CLI to the final LLM answer"
Example output: [
  "how does the CLI parse and dispatch a user query",
  "how does the retrieval pipeline process a query into context",
  "how does the synthesizer produce the final LLM answer from retrieved context"
]

Query: {query}
"""

PLANNER_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "produce_query_plan",
        "description": (
            "Produce a structured retrieval plan for the given code-search query. "
            "Classify the intent and decompose into focused sub-queries with "
            "per-leg search hints."
        ),
        "parameters": {
            "type": "object",
            "required": ["intent", "execution_mode", "sub_queries"],
            "additionalProperties": False,
            "properties": {
                "intent": {
                    "type": "string",
                    "enum": [
                        "symbol_lookup",
                        "file_overview",
                        "feature_flow",
                        "project_overview",
                        "comparison",
                        "config_lookup",
                        "dependency_map",
                        "blast_radius",
                    ],
                    "description": "Classified intent type for the query.",
                },
                "execution_mode": {
                    "type": "string",
                    "enum": ["parallel", "sequential"],
                    "description": (
                        "'parallel' when sub-queries are independent; "
                        "'sequential' when later sub-queries depend on earlier results."
                    ),
                },
                "sub_queries": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 3,
                    "description": "Decomposed sub-queries — 1 to 3 focused retrieval units.",
                    "items": {
                        "type": "object",
                        "required": [
                            "semantic_query",
                            "hyde_snippet",
                            "bm25_tokens",
                            "grep_hints",
                            "file_hints",
                        ],
                        "additionalProperties": False,
                        "properties": {
                            "semantic_query": {
                                "type": "string",
                                "description": (
                                    "Technical description of what to find — NOT a question. "
                                    "E.g. 'batch embedding logic inside the indexing pipeline'."
                                ),
                            },
                            "hyde_snippet": {
                                "type": "string",
                                "description": (
                                    "Hypothetical 3–8 line code snippet that a perfect answer "
                                    "would contain. Used for HyDE vector search."
                                ),
                            },
                            "bm25_tokens": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": (
                                    "Clean keyword tokens for BM25 search. "
                                    "No stop words, no question words, no punctuation."
                                ),
                            },
                            "grep_hints": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": (
                                    "Exact symbol names, function names, class names, or "
                                    "filename fragments likely to appear verbatim in the code."
                                ),
                            },
                            "file_hints": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": (
                                    "Partial filename or path fragments hinting at "
                                    "which files to prioritise."
                                ),
                            },
                            "depends_on": {
                                "type": "array",
                                "items": {"type": "integer"},
                                "description": (
                                    "0-based indices of sub-queries that must complete "
                                    "before this one runs. Empty list means no dependency."
                                ),
                                "default": [],
                            },
                        },
                    },
                },
            },
        },
    },
}
