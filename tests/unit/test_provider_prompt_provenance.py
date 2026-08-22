"""No provider may carry a prompt string that belongs to a different model.

WHY THIS EXISTS. `src/trelix/embedder/bge_code.py` shipped, from its first release through
3.1.7, with `_QUERY_INSTRUCTION = "Represent this query for searching relevant code: "` —
which is, character for character, `nomic-ai/CodeRankEmbed`'s published `prompts.query`,
read from its own config_sentence_transformers.json. `BAAI/bge-code-v1` publishes something
entirely different: the format `<instruct>{}\\n<query>{}` with a task instruction, and it
ships `<instruct>` / `<query>` as `additional_special_tokens` in its tokenizer.

Two things let it survive 28 tagged releases. The provider's pooling degeneracy meant every
query embedding was byte-identical, so the prompt text could not change any observable — no
measurement could see it. And the tests that existed asserted
`encoded_text.startswith(_QUERY_INSTRUCTION)`, deriving the expected value FROM the constant
under test, so they passed for any value at all.

So this file does not test one provider. It tests the INVARIANT the bug violated: a prompt
constant must match what its own model publishes, and must appear in exactly one provider.
That is checkable offline against recorded snapshots of each model's published config, and it
is what makes a copy-paste between providers fail loudly instead of silently.

SNAPSHOT PROVENANCE. Every string below was fetched from huggingface.co on 2026-08-22 from
the named file. They are data, not behaviour: nothing in src/ reads this module, so a drift
here cannot change a vector — it can only fail a test.
"""

from __future__ import annotations

import inspect

# ── nomic-ai/CodeRankEmbed, config_sentence_transformers.json ──────────────────────────
#   {"prompts": {"query": "Represent this query for searching relevant code: "}}
# Documents take no prefix. Note the trailing space; it is part of the published string.
CODERANKEMBED_QUERY_PROMPT = "Represent this query for searching relevant code: "

# ── BAAI/bge-code-v1 ───────────────────────────────────────────────────────────────────
#   config_sentence_transformers.json: "prompts": {}  -- EMPTY, so the instruction cannot
#   come from the model config and must be supplied by the caller.
#   Model card: query_instruction_format="<instruct>{}\n<query>{}" and a per-task
#   instruction table; line 181 is the CosQA entry, the published task closest to trelix's
#   (a natural-language query retrieving code).
BGE_CODE_QUERY_FORMAT = "<instruct>{}\n<query>{}"
BGE_CODE_COSQA_INSTRUCTION = (
    "Given a web search query, retrieve relevant code that can help answer the query."
)
# tokenizer_config.json additional_special_tokens — why the format's markers are single
# tokens rather than literal text.
BGE_CODE_SPECIAL_TOKENS = ["<instruct>", "<query>"]


class TestBgeCodeCarriesBaaisOwnProtocol:
    def test_the_query_instruction_is_not_another_models_prompt(self) -> None:
        """The exact regression, pinned.

        MUTATION: set `_QUERY_INSTRUCTION` back to CODERANKEMBED_QUERY_PROMPT and this
        fails. That is the state trelix shipped in for 28 releases.
        """
        from trelix.embedder.bge_code import _QUERY_INSTRUCTION

        assert _QUERY_INSTRUCTION != CODERANKEMBED_QUERY_PROMPT, (
            "bge-code is again using nomic-ai/CodeRankEmbed's published query prompt. "
            "BAAI/bge-code-v1 publishes prompts:{} and the model card supplies the "
            "instruction and format instead."
        )

    def test_the_query_instruction_is_one_baai_publishes(self) -> None:
        from trelix.embedder.bge_code import _QUERY_INSTRUCTION

        assert _QUERY_INSTRUCTION == BGE_CODE_COSQA_INSTRUCTION

    def test_the_format_is_baais_instruct_query_form(self) -> None:
        """MUTATION: revert to FlagEmbedding's `"{}{}"` default and this fails."""
        from trelix.embedder.bge_code import _QUERY_INSTRUCTION_FORMAT

        assert _QUERY_INSTRUCTION_FORMAT == BGE_CODE_QUERY_FORMAT
        for marker in BGE_CODE_SPECIAL_TOKENS:
            assert marker in _QUERY_INSTRUCTION_FORMAT, (
                f"{marker} is one of bge-code-v1's additional_special_tokens and must appear "
                "in the format, or the model never sees the token it was trained on"
            )

    def test_the_format_is_actually_passed_to_the_model(self) -> None:
        """A correct constant that nothing uses is the failure mode this guards.

        FlagEmbedding's `query_instruction_format` defaults to `"{}{}"`, so omitting the
        kwarg silently reduces BAAI's format to a plain prefix.

        MUTATION: delete the `query_instruction_format=` argument in `BGECodeEmbedder.
        __init__` and this fails.
        """
        from trelix.embedder.bge_code import BGECodeEmbedder

        src = inspect.getsource(BGECodeEmbedder.__init__)

        assert "query_instruction_format=" in src, (
            "the format constant is defined but never handed to the model, so "
            "FlagEmbedding's plain-concatenation default applies instead"
        )
        assert "query_instruction_for_retrieval=" in src


class TestNoPromptIsSharedBetweenProviders:
    """The invariant, rather than the instance."""

    def test_coderankembeds_prompt_appears_in_exactly_one_provider(self) -> None:
        """It is nomic-code's, and only nomic-code's.

        Detected by parsing the AST, not by substring search, and the distinction is not
        academic: a plain `in source` check flagged bge_code.py on its first run, because
        the comment there EXPLAINS the bug and quotes the offending string verbatim. That
        explanation is load-bearing documentation and must stay. Comments do not exist in
        an AST, so parsing sees only strings that are actually code. Module and class/function
        docstrings are skipped for the same reason — a docstring naming the old value is
        prose, not protocol.

        MUTATION: paste CODERANKEMBED_QUERY_PROMPT into any other embedder module as a real
        string literal (an assignment, a call argument, a default) and this fails. Adding it
        to a comment or docstring correctly does NOT fail.
        """
        import ast

        import trelix.embedder.base as base_mod
        import trelix.embedder.bge_code as bge_mod
        import trelix.embedder.nomic_code as nomic_mod
        import trelix.embedder.sparse as sparse_mod

        modules = {
            "nomic_code": nomic_mod,
            "bge_code": bge_mod,
            "base": base_mod,
            "sparse": sparse_mod,
        }

        def _code_string_literals(source: str) -> set[str]:
            tree = ast.parse(source)
            docstrings = {
                id(node.body[0].value)
                for node in ast.walk(tree)
                if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef)
                and node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            }
            return {
                n.value
                for n in ast.walk(tree)
                if isinstance(n, ast.Constant)
                and isinstance(n.value, str)
                and id(n) not in docstrings
            }

        carriers = [
            name
            for name, mod in modules.items()
            if CODERANKEMBED_QUERY_PROMPT in _code_string_literals(inspect.getsource(mod))
        ]

        assert carriers == ["nomic_code"], (
            f"CodeRankEmbed's published query prompt appears as a code literal in {carriers}; "
            "it belongs to nomic-code alone. A prompt shared between providers means at "
            "least one of them is applying a protocol its model never published."
        )

    def test_the_ast_detector_ignores_comments_but_not_code(self) -> None:
        """The detector above is only trustworthy if this holds.

        Without it, an AST walk that silently returned nothing would make the previous test
        pass for every possible input — the exact 'reads the same on success and on nothing
        ran' failure this project keeps hitting.
        """
        import ast

        def _code_string_literals(source: str) -> set[str]:
            tree = ast.parse(source)
            docstrings = {
                id(node.body[0].value)
                for node in ast.walk(tree)
                if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef)
                and node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            }
            return {
                n.value
                for n in ast.walk(tree)
                if isinstance(n, ast.Constant)
                and isinstance(n.value, str)
                and id(n) not in docstrings
            }

        assert _code_string_literals('# "needle"\nx = 1\n') == set(), "comment was counted"
        assert _code_string_literals('"""A docstring saying needle."""\nx = 1\n') == set(), (
            "module docstring was counted"
        )
        assert _code_string_literals('x = "needle"\n') == {"needle"}, "code literal was MISSED"
        assert _code_string_literals('f(kw="needle")\n') == {"needle"}, "kwarg literal was MISSED"

    def test_every_provider_prompt_constant_is_traceable_to_its_own_model(self) -> None:
        """A directory of which constant belongs to which model, asserted not just written.

        This is the part that makes the invariant maintainable: adding a provider with a
        prompt means adding a row here, which forces someone to name the published source.
        """
        from trelix.embedder import bge_code, nomic_code

        owned = {
            # module constant                              -> the model that publishes it
            nomic_code._EXPECTED_PUBLISHED_QUERY_PROMPT: "nomic-ai/CodeRankEmbed",
            bge_code._QUERY_INSTRUCTION: "BAAI/bge-code-v1",
            bge_code._QUERY_INSTRUCTION_FORMAT: "BAAI/bge-code-v1",
        }

        assert len(owned) == 3, (
            f"two providers are using the identical prompt string, so one of them is wrong: {owned}"
        )
        assert owned[CODERANKEMBED_QUERY_PROMPT] == "nomic-ai/CodeRankEmbed"
        assert owned[BGE_CODE_COSQA_INSTRUCTION] == "BAAI/bge-code-v1"
