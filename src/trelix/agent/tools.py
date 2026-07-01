"""
OpenAI-format tool schemas for the trelix ReAct agent.

These four tools are the complete action space. The LLM must call exactly
one per turn. 'done' terminates the loop with the final answer.
"""

from __future__ import annotations

from typing import Any

AGENT_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "retrieve",
            "description": (
                "Retrieve code symbols relevant to a natural-language query. "
                "Use this to explore unfamiliar parts of the codebase or find "
                "implementations. Returns up to 10 results with file, symbol name, and body."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Natural-language or keyword query describing the code to find."
                        ),
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": (
                "Search the codebase for an exact string or identifier. "
                "Use for finding specific variable names, function calls, or error messages. "
                "Returns file paths and matching lines."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Exact string or simple glob pattern to search for.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum results to return (default 10, max 50).",
                        "default": 10,
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_symbol",
            "description": (
                "Get the full source code of a specific symbol by its qualified name. "
                "Use when you know the exact function or class name. "
                "Example: 'AuthService.login' or 'hash_password'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "qualified_name": {
                        "type": "string",
                        "description": "Fully-qualified symbol name, e.g. 'MyClass.my_method'.",
                    }
                },
                "required": ["qualified_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "done",
            "description": (
                "Signal that you have sufficient context and provide the final answer. "
                "Call this when you are confident in your response. "
                "The answer will be returned directly to the user."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "The complete, final answer to the user's question.",
                    }
                },
                "required": ["answer"],
            },
        },
    },
]

TOOL_NAMES: frozenset[str] = frozenset(t["function"]["name"] for t in AGENT_TOOLS)
