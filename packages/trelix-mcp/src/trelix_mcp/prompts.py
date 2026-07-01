"""
trelix-mcp MCP Prompts — reusable LLM interaction templates.

MCP Prompts are the third server-side primitive (alongside Tools and Resources).
They are user-controlled templates that structure interactions with language models,
guiding the LLM to use trelix Tools and Resources in a predictable, productive way.

Prompts implemented:
  trelix-search       — structured prompt for semantic code search
  trelix-explain      — structured prompt for explaining a specific code symbol
  trelix-blast-radius — structured prompt for impact analysis before refactoring

All builder functions return a list of message dicts compatible with the MCP
``PromptMessage`` schema (``{"role": str, "content": str}``).
All log output goes to stderr only — no print() calls.
"""
from __future__ import annotations


def build_search_prompt(query: str, repo_path: str) -> list[dict[str, str]]:
    """Build a structured prompt for semantic code search using trelix.

    Args:
        query: Natural-language or keyword search query.
        repo_path: Absolute path to the repository root.

    Returns:
        List of message dicts with ``role`` and ``content`` keys.
    """
    return [
        {
            "role": "user",
            "content": (
                f"Search the codebase at `{repo_path}` for: {query}\n\n"
                "Use the search_code tool with this query. "
                "Show the most relevant results with file paths and relevant code snippets. "
                "Group results by file and annotate each with its symbol name and line numbers."
            ),
        }
    ]


def build_explain_prompt(qualified_name: str, repo_path: str) -> list[dict[str, str]]:
    """Build a structured prompt for explaining a specific code symbol.

    Args:
        qualified_name: Fully-qualified symbol name, e.g. ``AuthService.login``.
        repo_path: Absolute path to the repository root.

    Returns:
        List of message dicts with ``role`` and ``content`` keys.
    """
    return [
        {
            "role": "user",
            "content": (
                f"Explain the symbol `{qualified_name}` in the codebase at `{repo_path}`.\n\n"
                f"1. Use get_symbol(qualified_name='{qualified_name}', repo_path='{repo_path}') "
                "to fetch its source code.\n"
                "2. Explain what it does, its parameters, return value, and when to use it.\n"
                "3. Show any callers or dependents using blast_radius if relevant.\n"
                "4. Highlight any edge cases or failure modes evident in the implementation."
            ),
        }
    ]


def build_blast_radius_prompt(symbol_name: str, repo_path: str) -> list[dict[str, str]]:
    """Build a structured prompt for impact analysis before refactoring a symbol.

    Args:
        symbol_name: Name or qualified name of the symbol to analyse.
        repo_path: Absolute path to the repository root.

    Returns:
        List of message dicts with ``role`` and ``content`` keys.
    """
    return [
        {
            "role": "user",
            "content": (
                f"Analyze the blast radius of changing `{symbol_name}` in `{repo_path}`.\n\n"
                f"1. Use blast_radius(symbol_name='{symbol_name}', repo_path='{repo_path}') "
                "to find all dependents.\n"
                "2. List files that would be affected and estimate the change surface.\n"
                "3. Identify the highest-risk callers that might break.\n"
                "4. Suggest a safe refactoring approach with the smallest possible diff."
            ),
        }
    ]
