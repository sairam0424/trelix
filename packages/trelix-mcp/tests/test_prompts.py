"""Tests for trelix-mcp MCP Prompts."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))


class TestPromptsRegistered:
    async def test_three_prompts_registered(self) -> None:
        from trelix_mcp.server import mcp

        prompts = await mcp.list_prompts()
        prompt_names = {p.name for p in prompts}
        assert "trelix-search" in prompt_names, f"Prompts: {prompt_names}"
        assert "trelix-explain" in prompt_names, f"Prompts: {prompt_names}"
        assert "trelix-blast-radius" in prompt_names, f"Prompts: {prompt_names}"

    async def test_exactly_three_prompts(self) -> None:
        from trelix_mcp.server import mcp

        prompts = await mcp.list_prompts()
        names = [p.name for p in prompts]
        assert len(prompts) == 3, f"Expected 3 prompts, got {len(prompts)}: {names}"


class TestSearchPrompt:
    def test_build_search_prompt_returns_messages(self) -> None:
        from trelix_mcp.prompts import build_search_prompt

        messages = build_search_prompt(query="how does JWT auth work", repo_path="/my/repo")
        assert len(messages) >= 1

    def test_build_search_prompt_contains_query(self) -> None:
        from trelix_mcp.prompts import build_search_prompt

        messages = build_search_prompt(query="how does JWT auth work", repo_path="/my/repo")
        assert any("JWT" in m["content"] for m in messages)

    def test_build_search_prompt_contains_repo_path(self) -> None:
        from trelix_mcp.prompts import build_search_prompt

        messages = build_search_prompt(query="token refresh", repo_path="/my/repo")
        assert any("/my/repo" in m["content"] for m in messages)

    def test_build_search_prompt_has_user_role(self) -> None:
        from trelix_mcp.prompts import build_search_prompt

        messages = build_search_prompt(query="auth", repo_path="/any")
        assert messages[0]["role"] == "user"

    def test_build_search_prompt_message_has_role_and_content(self) -> None:
        from trelix_mcp.prompts import build_search_prompt

        messages = build_search_prompt(query="auth", repo_path="/any")
        for msg in messages:
            assert "role" in msg
            assert "content" in msg


class TestExplainPrompt:
    def test_build_explain_prompt_returns_messages(self) -> None:
        from trelix_mcp.prompts import build_explain_prompt

        messages = build_explain_prompt(qualified_name="AuthService.login", repo_path="/my/repo")
        assert len(messages) >= 1

    def test_build_explain_prompt_contains_qualified_name(self) -> None:
        from trelix_mcp.prompts import build_explain_prompt

        messages = build_explain_prompt(qualified_name="AuthService.login", repo_path="/my/repo")
        assert any("AuthService.login" in m["content"] for m in messages)

    def test_build_explain_prompt_contains_repo_path(self) -> None:
        from trelix_mcp.prompts import build_explain_prompt

        messages = build_explain_prompt(qualified_name="AuthService.login", repo_path="/my/repo")
        assert any("/my/repo" in m["content"] for m in messages)

    def test_build_explain_prompt_has_user_role(self) -> None:
        from trelix_mcp.prompts import build_explain_prompt

        messages = build_explain_prompt(qualified_name="Foo.bar", repo_path="/x")
        assert messages[0]["role"] == "user"


class TestBlastRadiusPrompt:
    def test_build_blast_radius_prompt_returns_messages(self) -> None:
        from trelix_mcp.prompts import build_blast_radius_prompt

        messages = build_blast_radius_prompt(symbol_name="login", repo_path="/my/repo")
        assert len(messages) >= 1

    def test_build_blast_radius_prompt_contains_symbol(self) -> None:
        from trelix_mcp.prompts import build_blast_radius_prompt

        messages = build_blast_radius_prompt(symbol_name="login", repo_path="/my/repo")
        assert any("login" in m["content"] for m in messages)

    def test_build_blast_radius_prompt_contains_repo_path(self) -> None:
        from trelix_mcp.prompts import build_blast_radius_prompt

        messages = build_blast_radius_prompt(symbol_name="login", repo_path="/my/repo")
        assert any("/my/repo" in m["content"] for m in messages)

    def test_build_blast_radius_prompt_has_user_role(self) -> None:
        from trelix_mcp.prompts import build_blast_radius_prompt

        messages = build_blast_radius_prompt(symbol_name="foo", repo_path="/x")
        assert messages[0]["role"] == "user"
