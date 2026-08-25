"""Closes real coverage gaps in ContextualChunker's LLM integration.

Round 10 measured indexing.chunker at 187 mutants / 147 killed / 40 survived,
with EVERY ONE of the 40 survivors inside ContextualChunker (zero in the base
Chunker). This round re-ran the same scoped mutmut measurement on this tree
(`scripts/mutation.py --report --modules indexing.chunker`) and reproduced the
identical 147/40 split, then read every one of the 40 survivors' own diffs via
`mutmut show` rather than re-deriving them from prose. All 40 fall into exactly
two functions:

  * ContextualChunker.build_chunks   -- 12 survivors
  * ContextualChunker._generate_summary -- 28 survivors

The 28 in `_generate_summary` cluster into two symmetric groups that this file
names explicitly:

  * the "New path" (`isinstance(self._llm_client, TrelixChatClient)`) branch's
    prompt construction and its `complete(messages=..., max_tokens=...,
    temperature=...)` call -- NEVER exercised by any pre-existing test, because
    every ContextualChunker test elsewhere in this repo builds its stub around
    `.chat.completions.create(...)` (the "Legacy path"), so the New path's own
    isinstance check always resolves False against them.
  * the "Legacy path"'s own `messages=[{"role": "user", "content": prompt}]`
    dict construction -- exercised for `model`/`max_completion_tokens`/
    `temperature` by test_chunker.py's `test_llm_called_with_correct_arguments`,
    but that test never inspects the `messages` kwarg itself.

Every fake below is a real subclass/hand-built stand-in for the interface it
imitates (never a Mock), matching the existing `_DriftingChatClient` pattern in
test_planner_determinism.py and the existing `_StubLLMClient` pattern in
test_chunker_token_budget_boundary.py.
"""

from __future__ import annotations

from typing import Any

from trelix.core.config import ChunkerConfig
from trelix.core.models import ImportEdge, Symbol, SymbolKind
from trelix.indexing.chunker import ContextualChunker
from trelix.llm.client import ChatMessage, ChatResponse, ToolCallResponse, TrelixChatClient

_REL_PATH = "src/pkg/math_ops.py"
_LANGUAGE = "python"

_EXPECTED_PROMPT = (
    "In 2-3 sentences, describe what this code does. "
    "File: src/pkg/math_ops.py, Language: python. Code:\n"
    "def add(a, b):\n    return a + b"
)


def _symbol(
    body: str,
    *,
    id: int | None = 1,
    name: str = "add",
    kind: SymbolKind = SymbolKind.FUNCTION,
    parent_id: int | None = None,
) -> Symbol:
    return Symbol(
        id=id,
        file_id=1,
        name=name,
        qualified_name=name,
        kind=kind,
        line_start=1,
        line_end=2,
        signature=f"def {name}()",
        body=body,
        docstring=None,
        parent_id=parent_id,
    )


# ---------------------------------------------------------------------------
# Fakes -- real ABC subclasses / hand-built stand-ins, never Mock
# ---------------------------------------------------------------------------


class _StubTrelixChatClient(TrelixChatClient):
    """Hand-written fake of the `TrelixChatClient` ABC.

    Deliberately NOT a MagicMock: a mock answers to any attribute/isinstance
    check, so it could never discriminate `isinstance(x, TrelixChatClient)`
    the way a real subclass does. `.calls` records every `complete()`
    invocation's own arguments so tests can assert on exactly what
    ContextualChunker sent, not on a value recomputed from the module under
    test.
    """

    def __init__(self, content: str) -> None:
        self._content = content
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        messages: list[ChatMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        system: str | None = None,
        thinking: bool = False,
    ) -> ChatResponse:
        self.calls.append(
            {"messages": messages, "max_tokens": max_tokens, "temperature": temperature}
        )
        return ChatResponse(content=self._content, model="stub-model", finish_reason="stop")

    def stream(self, *args: object, **kwargs: object) -> Any:
        raise AssertionError("ContextualChunker never calls stream()")

    def tool_call(self, *args: object, **kwargs: object) -> ToolCallResponse:
        raise AssertionError("ContextualChunker never calls tool_call()")


class _StubLegacyCompletions:
    """Hand-written stand-in for openai's `client.chat.completions`.

    Same shape as test_chunker_token_budget_boundary.py's `_StubCompletions`,
    extended to record the FULL kwargs of each call (that file only records a
    call count) so a test here can assert on the exact `messages` list built
    by the "Legacy path" -- the thing test_chunker.py's own
    `test_llm_called_with_correct_arguments` never inspects.
    """

    def __init__(self, content: str) -> None:
        self._content = content
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: object) -> _StubLegacyResponse:
        self.calls.append(kwargs)
        return _StubLegacyResponse(self._content)


class _StubLegacyChat:
    def __init__(self, completions: _StubLegacyCompletions) -> None:
        self.completions = completions


class _StubLegacyMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _StubLegacyChoice:
    def __init__(self, content: str) -> None:
        self.message = _StubLegacyMessage(content)


class _StubLegacyResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_StubLegacyChoice(content)]


class _StubLegacyClient:
    def __init__(self, content: str) -> None:
        self.completions = _StubLegacyCompletions(content)
        self.chat = _StubLegacyChat(self.completions)


# ---------------------------------------------------------------------------
# The TrelixChatClient branch itself (chunker.py:289) is exercised at all
# ---------------------------------------------------------------------------


class TestTrelixChatClientPathIsExercised:
    def test_summary_from_trelix_chat_client_is_prepended(self) -> None:
        """Fails under: `isinstance(self._llm_client, TrelixChatClient)` ->
        `not isinstance(...)`, deleting the branch's body, or
        `response = self._llm_client.complete(...)` -> `response = None`.

        A MagicMock legacy-shaped client would satisfy this test trivially by
        falling through to the Legacy path instead -- the point of this stub
        is that it is a REAL TrelixChatClient subclass, so only the New path
        can produce a result. If the isinstance check is inverted, removed,
        or the call itself is deleted, the code either falls into the legacy
        `self._llm_client.chat...` access (AttributeError: no `.chat` on this
        stub) or crashes on `None.content`, both swallowed by the outer
        `except Exception`, and no summary is prepended.
        """
        client = _StubTrelixChatClient("Adds two numbers and returns the sum.")
        chunker = ContextualChunker(ChunkerConfig(contextual=True), llm_client=client)
        symbol = _symbol("def add(a, b):\n    return a + b")

        chunks = chunker.build_chunks([symbol], [], _REL_PATH, _LANGUAGE)

        assert len(client.calls) == 1
        assert chunks[0].chunk_text.startswith("Adds two numbers and returns the sum.\n\n")


class TestTrelixChatClientPromptContent:
    def test_prompt_message_is_exact_role_and_text(self) -> None:
        """Fails under: any mutation of `_CONTEXT_PROMPT`'s literal text, the
        `rel_path`/`language`/`body` arguments fed into `.format(...)`, the
        `role="user"` literal, or the `messages=[ChatMessage(...)]` structure
        in `_generate_summary`'s New path.

        The expected string is hand-typed at module scope, not built by
        calling `ContextualChunker._CONTEXT_PROMPT.format(...)` -- recomputing
        the expected value from the module under test would make this
        vacuous.
        """
        client = _StubTrelixChatClient("A summary.")
        chunker = ContextualChunker(ChunkerConfig(contextual=True), llm_client=client)
        symbol = _symbol("def add(a, b):\n    return a + b")

        chunker.build_chunks([symbol], [], _REL_PATH, _LANGUAGE)

        assert len(client.calls) == 1
        messages = client.calls[0]["messages"]
        assert len(messages) == 1
        assert messages[0].role == "user"
        assert messages[0].content == _EXPECTED_PROMPT

    def test_max_tokens_and_temperature_forwarded(self) -> None:
        """Fails under: `max_tokens=self.config.contextual_max_tokens` -> a
        different value or removed entirely, or `temperature=0` -> a nonzero
        literal or removed entirely.
        """
        client = _StubTrelixChatClient("A summary.")
        config = ChunkerConfig(contextual=True, contextual_max_tokens=77)
        chunker = ContextualChunker(config, llm_client=client)
        symbol = _symbol("def add(a, b):\n    return a + b")

        chunker.build_chunks([symbol], [], _REL_PATH, _LANGUAGE)

        assert client.calls[0]["max_tokens"] == 77
        assert client.calls[0]["temperature"] == 0


class TestPromptBodyTruncatedAt800Chars:
    def test_body_sliced_at_exactly_800_characters(self) -> None:
        """Fails under: `symbol.body[:800]` -> `symbol.body[:801]` (or any
        other boundary shift) in `_generate_summary`.

        Body is built so the 800th character (index 799) is a unique marker
        immediately followed by 200 characters that must NEVER reach the
        prompt if the slice is exactly 800. A `[:801]` slice would leak one
        tail character through; a `[:799]` slice would drop the marker
        itself -- either is observable here.
        """
        body = ("v" * 799) + "Q" + ("k" * 200)
        assert len(body) == 1000  # precondition: fixture is actually 1000 chars

        client = _StubTrelixChatClient("A summary.")
        chunker = ContextualChunker(ChunkerConfig(contextual=True), llm_client=client)
        symbol = _symbol(body)

        chunker.build_chunks([symbol], [], _REL_PATH, _LANGUAGE)

        # The template's own text ("Code:\n" is its final, fixed fragment)
        # is stripped off first, so this compares ONLY the body-derived tail
        # of the prompt -- not the whole string, which would otherwise let a
        # stray "v"/"k"/"Q" inside the template's boilerplate (e.g. the "k"
        # in rel_path's "pkg") confuse the count.
        prompt = client.calls[0]["messages"][0].content
        code_marker = "Code:\n"
        prompt_body = prompt[prompt.index(code_marker) + len(code_marker) :]
        assert prompt_body == ("v" * 799) + "Q"


class TestContextualEnabledPropertyDirect:
    """Direct tests of the `_contextual_enabled` property, isolated from
    `build_chunks` so each boolean-logic mutation is pinned to its own
    unambiguous assertion rather than an outcome that more than one code path
    could produce.
    """

    def test_false_when_client_is_none_even_if_contextual_flag_is_true(self) -> None:
        """Fails under: `self._llm_client is not None` -> `... is None` in
        `_contextual_enabled`.

        Observable ONLY at the property itself: through `build_chunks`, this
        exact mutation is equivalent, because `_generate_summary`'s own
        `assert self._llm_client is not None` re-raises into the same
        `except Exception` that a missing client already falls into, so the
        externally visible chunk_text is identical either way.
        """
        chunker = ContextualChunker(ChunkerConfig(contextual=True), llm_client=None)
        assert chunker._contextual_enabled is False

    def test_false_when_contextual_flag_is_false_even_with_a_real_client(self) -> None:
        """Fails under: `and` -> `or` in `_contextual_enabled`.

        With `contextual=False` and a real (non-None) client, `and` yields
        False (correct: disabled) while `or` yields True (wrong: would try to
        call the LLM). Needs a REAL client object, not None, or both branches
        of the boolean agree and the mutation survives -- which is exactly
        why the pre-existing `test_contextual_false_identical_to_base_chunker`
        (llm_client=None) cannot catch this mutation.
        """
        client = _StubTrelixChatClient("should never be read")
        chunker = ContextualChunker(ChunkerConfig(contextual=False), llm_client=client)
        assert chunker._contextual_enabled is False


class TestContextualFalseWithRealClientNeverCallsLLM:
    def test_build_chunks_does_not_invoke_a_present_client_when_disabled(self) -> None:
        """Behavioural companion to the `and`/`or` property test above: proves
        `build_chunks` actually consults `_contextual_enabled` rather than
        some other gate, by checking the stub was never called at all.

        Fails under: `and` -> `or` in `_contextual_enabled` (the LLM would be
        called and `client.calls` would be non-empty).
        """
        client = _StubTrelixChatClient("should never be read")
        chunker = ContextualChunker(ChunkerConfig(contextual=False), llm_client=client)
        symbol = _symbol("def add(a, b):\n    return a + b")

        chunker.build_chunks([symbol], [], _REL_PATH, _LANGUAGE)

        assert client.calls == []


class TestGenerateSummaryEmptyResponseHandling:
    def test_whitespace_only_content_returns_none_not_empty_string(self) -> None:
        """Fails under: `response.content.strip() or None` -> `... and None`.

        `_generate_summary` is called directly (it is the unit actually
        responsible for this decision) rather than observed through
        `build_chunks`, because `build_chunks` treats "" and None identically
        (`if context_summary:` is falsy for both), which would make this
        mutation equivalent at that level. `"" and None` evaluates to `""`
        (short-circuits on the first falsy operand), so an `is None` check
        tells the mutant apart from the correct `or`, where `"" or None`
        evaluates to `None`.
        """
        client = _StubTrelixChatClient("   \n  ")
        chunker = ContextualChunker(ChunkerConfig(contextual=True), llm_client=client)
        symbol = _symbol("def add(a, b):\n    return a + b")

        result = chunker._generate_summary(symbol, _REL_PATH, _LANGUAGE)

        assert result is None

    def test_content_is_stripped_of_surrounding_whitespace(self) -> None:
        """Fails under: deleting `.strip()` in `_generate_summary`'s New
        path.
        """
        client = _StubTrelixChatClient("  Adds two numbers.  \n")
        chunker = ContextualChunker(ChunkerConfig(contextual=True), llm_client=client)
        symbol = _symbol("def add(a, b):\n    return a + b")

        result = chunker._generate_summary(symbol, _REL_PATH, _LANGUAGE)

        assert result == "Adds two numbers."


class TestLegacyPathPromptMessages:
    """The "Legacy path" (chunker.py:296-303) builds its own, separately
    literal `messages=[{"role": "user", "content": prompt}]` -- a raw dict,
    not a `ChatMessage`. test_chunker.py's `test_llm_called_with_correct_
    arguments` checks `model`/`max_completion_tokens`/`temperature` on this
    same call but never looks at `messages` itself, so its 4 keys/values
    (`messages=None`/omitted, `"role"`/`"user"`/`"content"` each renamed) are
    unguarded.
    """

    def test_legacy_messages_list_is_exact_role_and_content(self) -> None:
        """Fails under: `messages=[{"role": "user", "content": prompt}]` ->
        `messages=None`, the kwarg dropped entirely, or any of the two dict
        keys or the `"user"` value renamed/mutated.

        Whole-structure equality (one list containing one dict), not a
        key-by-key walk -- this is the "explicit table" form, not an
        iteration over the collection being pinned.
        """
        client = _StubLegacyClient("A summary.")
        chunker = ContextualChunker(ChunkerConfig(contextual=True), llm_client=client)
        symbol = _symbol("def add(a, b):\n    return a + b")

        chunker.build_chunks([symbol], [], _REL_PATH, _LANGUAGE)

        assert len(client.completions.calls) == 1
        assert client.completions.calls[0]["messages"] == [
            {"role": "user", "content": _EXPECTED_PROMPT}
        ]


class TestBuildChunksDelegationForwardsRealArguments:
    """ContextualChunker.build_chunks's disabled branch
    (`return super().build_chunks(symbols, imports, file_rel_path, language,
    parent_symbols)`) forwards 5 arguments verbatim. The pre-existing
    `test_contextual_false_identical_to_base_chunker` and
    `test_contextual_true_no_client_falls_back_to_base_output` both call with
    `imports=[]` and no `parent_symbols`, so a mutation replacing either
    forwarded argument with `None` is unobservable there -- `[]` and a
    delegate-side `None` both end up producing an empty import header, and no
    `parent_symbols` was going to produce a parent header either way.
    """

    def test_disabled_contextual_forwards_real_imports_to_base_chunker(self) -> None:
        """Fails under: the delegating `super().build_chunks(...)` call's
        `imports` argument -> `None`.
        """
        chunker = ContextualChunker(ChunkerConfig(contextual=False), llm_client=None)
        symbol = _symbol("def add(a, b):\n    return a + b")
        imp = ImportEdge(file_id=1, imported_from="os.path", imported_names=["join"])

        chunks = chunker.build_chunks([symbol], [imp], _REL_PATH, _LANGUAGE)

        assert "os.path" in chunks[0].chunk_text

    def test_disabled_contextual_forwards_real_parent_symbols_to_base_chunker(self) -> None:
        """Fails under: the delegating `super().build_chunks(...)` call's
        `parent_symbols` argument -> `None`, or dropped entirely (both fall
        back to the base class's own default of `{}`, silently losing the
        real parent map).
        """
        config = ChunkerConfig(contextual=False, include_parent_signature=True)
        chunker = ContextualChunker(config, llm_client=None)
        parent = _symbol("class Foo: ...", id=10, name="Foo", kind=SymbolKind.CLASS)
        method = _symbol("def bar(self): pass", id=11, name="bar", parent_id=10)

        chunks = chunker.build_chunks(
            [method], [], _REL_PATH, _LANGUAGE, parent_symbols={10: parent}
        )

        assert any(line == "# Class: Foo" for line in chunks[0].chunk_text.splitlines())


class TestEnabledContextualChunkerHeaderForwarding:
    """The SAME two arguments, forwarded a second time inside
    ContextualChunker's OWN `build_chunks` loop (the `_build_chunk_text(...,
    import_header=import_header, parent_symbols=parent_symbols)` call reached
    only when `_contextual_enabled` is True) -- a separate code site from the
    delegation branch above, and from the base Chunker's own identical-
    looking call.
    """

    def test_enabled_contextual_chunker_includes_real_imports_in_header(self) -> None:
        """Fails under: `import_header = self._build_import_header(imports)`
        -> `import_header = None` or `self._build_import_header(None)`, or
        the `import_header=import_header` kwarg at the `_build_chunk_text`
        call site -> `import_header=None`.
        """
        client = _StubTrelixChatClient("A summary.")
        chunker = ContextualChunker(ChunkerConfig(contextual=True), llm_client=client)
        symbol = _symbol("def add(a, b):\n    return a + b")
        imp = ImportEdge(file_id=1, imported_from="collections.abc", imported_names=["Mapping"])

        chunks = chunker.build_chunks([symbol], [imp], _REL_PATH, _LANGUAGE)

        assert "collections.abc" in chunks[0].chunk_text

    def test_enabled_contextual_chunker_includes_real_parent_class_in_header(self) -> None:
        """Fails under: the `parent_symbols=parent_symbols` kwarg at the
        `_build_chunk_text` call site inside ContextualChunker's own
        `build_chunks` -> `parent_symbols=None`.
        """
        client = _StubTrelixChatClient("A summary.")
        config = ChunkerConfig(contextual=True, include_parent_signature=True)
        chunker = ContextualChunker(config, llm_client=client)
        parent = _symbol("class Widget: ...", id=20, name="Widget", kind=SymbolKind.CLASS)
        method = _symbol("def render(self): pass", id=21, name="render", parent_id=20)

        chunks = chunker.build_chunks(
            [method], [], _REL_PATH, _LANGUAGE, parent_symbols={20: parent}
        )

        assert any(line == "# Class: Widget" for line in chunks[0].chunk_text.splitlines())


class TestParentSymbolsDefaultInContextualChunkerOwnBuildChunks:
    def test_omitted_parent_symbols_defaults_to_empty_dict_not_none(self) -> None:
        """Fails under: `if parent_symbols is None:` -> `if parent_symbols is
        not None:`, or `parent_symbols = {}` -> `parent_symbols = None`, in
        ContextualChunker.build_chunks's OWN copy of this default
        (chunker.py:228-229) -- a separate code site from the base Chunker's
        identical-looking line, only reachable when `_contextual_enabled` is
        True.

        With either mutation, `parent_symbols` stays `None`, and
        `_build_chunk_text`'s `symbol.parent_id and symbol.parent_id in
        parent_symbols` then evaluates `... in None` for any symbol with a
        truthy `parent_id`, raising `TypeError: argument of type 'NoneType'
        is not iterable`. This symbol's `parent_id=42` is deliberately not
        present in any real parent map, so the correct behaviour is "no
        crash, no parent header" rather than a parent line actually
        appearing.
        """
        client = _StubTrelixChatClient("A summary.")
        chunker = ContextualChunker(
            ChunkerConfig(contextual=True, include_parent_signature=True),
            llm_client=client,
        )
        symbol = _symbol("def method(self): pass", parent_id=42)

        chunks = chunker.build_chunks([symbol], [], _REL_PATH, _LANGUAGE)

        assert not any(line.startswith("# Class:") for line in chunks[0].chunk_text.splitlines())


class TestSymbolIdFallback:
    def test_symbol_id_none_falls_back_to_zero(self) -> None:
        """Fails under: `symbol_id=symbol.id or 0` -> `symbol_id=symbol.id or
        1` in ContextualChunker's own `Chunk(...)` construction.

        Every other ContextualChunker test in this repo uses a truthy
        `id` (1, 42, 77, ...), so `symbol.id or X` evaluates to `symbol.id`
        regardless of X and cannot discriminate the fallback literal. `id`
        must be falsy (`None`) here for the two candidates to disagree at
        all.
        """
        client = _StubTrelixChatClient("A summary.")
        chunker = ContextualChunker(ChunkerConfig(contextual=True), llm_client=client)
        symbol = _symbol("def add(a, b):\n    return a + b", id=None)

        chunks = chunker.build_chunks([symbol], [], _REL_PATH, _LANGUAGE)

        assert chunks[0].symbol_id == 0
