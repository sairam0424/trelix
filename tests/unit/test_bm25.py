"""
Unit tests for trelix.retrieval.bm25.bm25_search.

Uses a real (file-based tmp_path) SQLite DB with FTS5 triggers — no mocks.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from trelix.core.models import (
    Chunk,
    IndexedFile,
    Language,
    Symbol,
    SymbolKind,
)
from trelix.retrieval.bm25 import _escape_fts5, bm25_search
from trelix.store.db import Database

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_file(db: Database, rel_path: str = "src/auth/login.py") -> int:
    f = IndexedFile(
        path=f"/repo/{rel_path}",
        rel_path=rel_path,
        language=Language.PYTHON,
        hash="deadbeef",
        size_bytes=512,
    )
    return db.upsert_file(f)


def _insert_symbol(
    db: Database,
    file_id: int,
    name: str,
    body: str,
    docstring: str | None = None,
) -> int:
    sym = Symbol(
        file_id=file_id,
        name=name,
        qualified_name=name,
        kind=SymbolKind.FUNCTION,
        line_start=1,
        line_end=10,
        signature=f"def {name}():",
        body=body,
        docstring=docstring,
    )
    sym_id = db.insert_symbol(sym)
    db._conn.commit()
    return sym_id


def _insert_chunk(db: Database, symbol_id: int, text: str) -> int:
    chunk = Chunk(symbol_id=symbol_id, chunk_text=text, token_count=len(text.split()))
    chunk_id = db.insert_chunk(chunk)
    db._conn.commit()
    return chunk_id


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db(tmp_path: Path) -> Database:
    """Fresh SQLite DB per test (real file so FTS5 triggers fire correctly)."""
    return Database(tmp_path / "index.db")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBm25Search:
    def test_returns_empty_for_empty_query(self, db: Database) -> None:
        """An empty query should return an empty list without error."""
        file_id = _make_file(db)
        sym_id = _insert_symbol(db, file_id, "authenticate_user", "def authenticate_user(): pass")
        _insert_chunk(db, sym_id, "authenticate_user")

        results = bm25_search(db, "", k=10)
        assert results == []

    def test_finds_matching_symbol(self, db: Database) -> None:
        """bm25_search finds a symbol whose name matches the query."""
        file_id = _make_file(db)
        sym_id = _insert_symbol(
            db,
            file_id,
            "authenticate_user",
            "def authenticate_user(username, password):\n"
            "    return check_password(username, password)",
            docstring="Authenticate a user by username and password.",
        )
        _insert_chunk(db, sym_id, "def authenticate_user(): ...")

        results = bm25_search(db, "authenticate_user", k=10)
        assert len(results) >= 1
        assert results[0].symbol.name == "authenticate_user"
        assert results[0].source == "bm25"

    def test_relevance_ordering(self, db: Database) -> None:
        """
        Symbol with stronger BM25 signal (query term in name + body + docstring)
        should rank above a symbol with weaker signal (term only in body).
        """
        file_id = _make_file(db)

        # Strong match: "tokenize" appears in name, body, and docstring
        strong_id = _insert_symbol(
            db,
            file_id,
            "tokenize_source",
            "def tokenize_source(text):\n    return tokenize(text)",
            docstring="Tokenize a source string into tokens.",
        )
        _insert_chunk(db, strong_id, "tokenize source tokens")

        # Weak match: "tokenize" appears only once in body
        weak_id = _insert_symbol(
            db,
            file_id,
            "parse_config",
            "def parse_config(path):\n    # tokenize before parsing\n    return {}",
        )
        _insert_chunk(db, weak_id, "parse config")

        results = bm25_search(db, "tokenize", k=10)
        assert len(results) >= 2

        names = [r.symbol.name for r in results]
        assert "tokenize_source" in names
        assert "parse_config" in names
        # Strong match must appear before weak match
        assert names.index("tokenize_source") < names.index("parse_config")

    def test_score_positive_and_in_range(self, db: Database) -> None:
        """SearchResult scores must be > 0 and <= 1."""
        file_id = _make_file(db)
        sym_id = _insert_symbol(
            db,
            file_id,
            "process_request",
            "def process_request(req): return req",
        )
        _insert_chunk(db, sym_id, "process request handler")

        results = bm25_search(db, "process_request", k=5)
        assert results, "Expected at least one result"
        for r in results:
            assert 0 < r.score <= 1.0

    def test_result_has_correct_source(self, db: Database) -> None:
        """source field must always be 'bm25'."""
        file_id = _make_file(db)
        sym_id = _insert_symbol(db, file_id, "handle_login", "def handle_login(): pass")
        _insert_chunk(db, sym_id, "handle login")

        results = bm25_search(db, "handle_login", k=5)
        for r in results:
            assert r.source == "bm25"

    def test_respects_k_limit(self, db: Database) -> None:
        """bm25_search must return at most k results."""
        file_id = _make_file(db)
        for i in range(10):
            sym_id = _insert_symbol(
                db,
                file_id,
                f"validate_field_{i}",
                f"def validate_field_{i}(v): return validate(v)",
            )
            _insert_chunk(db, sym_id, f"validate field {i}")

        results = bm25_search(db, "validate", k=3)
        assert len(results) <= 3

    def test_fallback_chunk_created_when_no_chunk_stored(self, db: Database) -> None:
        """
        When a symbol has no chunk row, bm25_search creates a synthetic Chunk
        from the symbol body so the SearchResult is still complete.
        """
        file_id = _make_file(db)
        body = "def orphan_function():\n    pass"
        # Insert symbol but NO chunk
        _insert_symbol(db, file_id, "orphan_function", body)

        results = bm25_search(db, "orphan_function", k=5)
        assert len(results) >= 1
        r = results[0]
        assert r.chunk is not None
        # Synthetic chunk is sliced from body
        assert body[:2000] in r.chunk.chunk_text or r.chunk.chunk_text in body[:2000]


class TestPathFilter:
    def test_path_filter_excludes_symbols_outside_prefix(self, db: Database) -> None:
        included_file = _make_file(db, "src/auth/login.py")
        excluded_file = _make_file(db, "src/billing/invoice.py")
        included_sym = _insert_symbol(
            db, included_file, "authenticate_user", "def authenticate_user(): pass"
        )
        excluded_sym = _insert_symbol(
            db, excluded_file, "authenticate_charge", "def authenticate_charge(): pass"
        )
        _insert_chunk(db, included_sym, "authenticate_user")
        _insert_chunk(db, excluded_sym, "authenticate_charge")

        results = bm25_search(db, "authenticate", k=10, path_filter="src/auth")
        names = {r.symbol.name for r in results}
        assert "authenticate_user" in names
        assert "authenticate_charge" not in names

    def test_no_path_filter_returns_symbols_from_every_file(self, db: Database) -> None:
        file_a = _make_file(db, "src/auth/login.py")
        file_b = _make_file(db, "src/billing/invoice.py")
        sym_a = _insert_symbol(db, file_a, "authenticate_user", "def authenticate_user(): pass")
        sym_b = _insert_symbol(db, file_b, "authenticate_charge", "def authenticate_charge(): pass")
        _insert_chunk(db, sym_a, "authenticate_user")
        _insert_chunk(db, sym_b, "authenticate_charge")

        results = bm25_search(db, "authenticate", k=10)
        names = {r.symbol.name for r in results}
        assert {"authenticate_user", "authenticate_charge"}.issubset(names)


# ---------------------------------------------------------------------------
# _escape_fts5 unit tests
# ---------------------------------------------------------------------------


class TestEscapeFts5:
    def test_single_identifier_becomes_prefix_search(self) -> None:
        result = _escape_fts5("authenticate_user")
        assert result == '"authenticate_user"*'

    def test_multi_word_strips_stop_words(self) -> None:
        result = _escape_fts5("what is the authenticate function")
        # Stop words: what, is, the, function → only "authenticate" should remain.
        # Check whole-word presence via token split rather than substring (avoids
        # "the" falsely matching inside "authenticate").
        tokens = result.split()
        assert "authenticate" in tokens
        assert "what" not in tokens
        assert "is" not in tokens
        # "the" would be a standalone token if not stripped; must not appear alone
        assert "the" not in tokens

    def test_empty_query_returns_empty_matcher(self) -> None:
        result = _escape_fts5("")
        # Empty/whitespace → single identifier branch fires → '""*' (no-match prefix)
        # OR multi-word branch returns '""'. Either is acceptable as long as it
        # produces no real FTS5 matches.
        assert result in ('""', '""*')


# ---------------------------------------------------------------------------
# is_short_query unit tests
# ---------------------------------------------------------------------------


class TestIsShortQuery:
    def test_single_word_is_short(self) -> None:
        from trelix.retrieval.bm25 import is_short_query

        assert is_short_query("login") is True

    def test_five_meaningful_words_at_threshold(self) -> None:
        from trelix.retrieval.bm25 import is_short_query

        # exactly 5 meaningful tokens → short at default threshold=5
        assert is_short_query("JWT token validation auth middleware") is True

    def test_six_meaningful_words_not_short(self) -> None:
        from trelix.retrieval.bm25 import is_short_query

        assert is_short_query("JWT token validation auth middleware handler") is False

    def test_stop_words_not_counted(self) -> None:
        from trelix.retrieval.bm25 import is_short_query

        # "how does the auth work" — only "auth" and "work" are meaningful (len>2, not stop)
        assert is_short_query("how does the auth work") is True

    def test_custom_threshold(self) -> None:
        from trelix.retrieval.bm25 import is_short_query

        assert is_short_query("auth token user session", threshold=3) is False
        assert is_short_query("auth token", threshold=3) is True

    def test_count_meaningful_tokens(self) -> None:
        from trelix.retrieval.bm25 import count_meaningful_tokens

        assert count_meaningful_tokens("JWT auth middleware") == 3
        assert count_meaningful_tokens("how does it work") == 1  # only "work" passes

    def test_empty_query_is_short(self) -> None:
        from trelix.retrieval.bm25 import is_short_query

        assert is_short_query("") is True


class TestBM25ReadPoolOptIn:
    """Opt-in read-only connection pool for parallel bm25_search() calls
    (v2.6.x scale backlog, Plan C Task C-1). Default (pool disabled) must
    be byte-for-byte identical to pre-existing behavior."""

    def test_bm25_search_unaffected_when_pool_not_enabled(self, tmp_path):
        """Default behavior (no enable_bm25_read_pool call) must produce
        identical results to before this feature existed."""
        db = Database(tmp_path / "test.db")
        file_id = _make_file(db, rel_path="foo.py")
        _insert_symbol(db, file_id, "authenticate_user", "def authenticate_user(): ...")

        results = bm25_search(db, "authenticate", k=10)
        assert len(results) == 1
        assert db._bm25_read_pool is None

    def test_bm25_search_works_identically_with_pool_enabled(self, tmp_path):
        db_path = tmp_path / "test.db"
        db = Database(db_path)
        file_id = _make_file(db, rel_path="foo.py")
        _insert_symbol(db, file_id, "authenticate_user", "def authenticate_user(): ...")
        db.enable_bm25_read_pool(pool_size=2)

        results = bm25_search(db, "authenticate", k=10)
        assert len(results) == 1

        db._bm25_read_pool.close_all()

    def test_concurrent_bm25_search_with_pool_enabled(self, tmp_path):
        """The whole point of this feature: N threads querying bm25_search
        concurrently must all succeed without 'database is locked' errors,
        when the read pool is enabled."""
        import threading

        db_path = tmp_path / "test.db"
        db = Database(db_path)
        file_id = _make_file(db, rel_path="foo.py")
        for i in range(20):
            _insert_symbol(db, file_id, f"fn_{i}", f"def fn_{i}(): return {i}")
        db.enable_bm25_read_pool(pool_size=4)

        errors = []

        def worker():
            try:
                bm25_search(db, "fn", k=10)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"concurrent bm25_search under the read pool raised: {errors}"
        db._bm25_read_pool.close_all()


# Explicit expected table. Deliberately written out as literals rather than
# derived from bm25._STOP_WORDS: an expectation built by iterating the module's
# own collection shrinks with the collection, which is precisely how 14
# walker.EXTENSION_MAP entries became deletable with a green suite.
#
# Only members of length >= 3 are pinned. Both `count_meaningful_tokens` and
# `_escape_fts5` also require len(token) >= 3 / > 2, so the 22 one- and
# two-character entries ("a", "is", "it", "of", ...) are unreachable: deleting
# one changes no behaviour. Pinning them would lock down dead data.
_EXPECTED_LIVE_STOP_WORDS = {
    "about",
    "after",
    "all",
    "also",
    "and",
    "any",
    "are",
    "been",
    "before",
    "between",
    "but",
    "call",
    "calls",
    "can",
    "class",
    "code",
    "could",
    "did",
    "does",
    "each",
    "file",
    "find",
    "for",
    "from",
    "function",
    "get",
    "give",
    "had",
    "has",
    "have",
    "her",
    "his",
    "how",
    "into",
    "just",
    "know",
    "like",
    "list",
    "make",
    "method",
    "more",
    "new",
    "not",
    "our",
    "out",
    "over",
    "return",
    "returns",
    "she",
    "should",
    "show",
    "some",
    "such",
    "tell",
    "than",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "they",
    "this",
    "type",
    "use",
    "used",
    "using",
    "value",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "will",
    "with",
    "would",
    "you",
    "your",
}

# Words that must never become stop words: they are ordinary code identifiers a
# developer searches for. Guards the "someone pastes a big NLTK list in"
# direction of the same defect.
_MUST_STAY_SEARCHABLE = (
    "auth",
    "token",
    "login",
    "parse",
    "index",
    "query",
    "search",
    "config",
    "schema",
    "session",
)


class TestStopWordListIsPinned:
    def test_every_live_stop_word_is_still_dropped(self) -> None:
        """
        MUTATION: deleting any length>=3 member of bm25._STOP_WORDS (e.g. the
        line '"show",'). The walker.EXTENSION_MAP failure mode: deleting "show"
        let "show me the auth code" carry the term `show` into the FTS5 MATCH
        expression and raised its meaningful-token count, and the whole suite
        stayed green.
        """
        from trelix.retrieval.bm25 import _STOP_WORDS, _escape_fts5, count_meaningful_tokens

        # Direction 1: nothing was deleted. Direction 2: nothing was added.
        live_members = {w for w in _STOP_WORDS if len(w) >= 3}
        assert live_members == _EXPECTED_LIVE_STOP_WORDS
        assert _EXPECTED_LIVE_STOP_WORDS == live_members
        assert len(_EXPECTED_LIVE_STOP_WORDS) == 80

        # Behavioural half: each pinned word is actually removed from both the
        # token count and the FTS5 MATCH expression, so the set above is not
        # just decoration.
        for word in sorted(_EXPECTED_LIVE_STOP_WORDS):
            phrase = f"{word} authenticate"
            assert count_meaningful_tokens(phrase) == 1, (
                f"stop word {word!r} was counted as a meaningful token in {phrase!r}"
            )
            assert _escape_fts5(phrase).split() == ["authenticate"], (
                f"stop word {word!r} leaked into the FTS5 MATCH expression for "
                f"{phrase!r}: {_escape_fts5(phrase)!r}"
            )

        for word in _MUST_STAY_SEARCHABLE:
            phrase = f"{word} authenticate"
            assert count_meaningful_tokens(phrase) == 2, (
                f"{word!r} is an ordinary code identifier and must not be a stop word"
            )
            assert word in _escape_fts5(phrase).split(), (
                f"{word!r} was stripped from the FTS5 MATCH expression for {phrase!r}"
            )


class TestMultiWordQueryPrecision:
    def test_multiword_query_requires_every_token_including_three_char_ones(
        self, db: Database
    ) -> None:
        """
        MUTATIONS (two, both survived the pre-existing suite):
          1. `return " ".join(tokens)` -> `return " OR ".join(tokens)` in
             _escape_fts5 — FTS5's implicit AND becomes OR, so every multi-word
             query floods the top-k with symbols that matched only one term.
          2. `len(t) > 2` -> `len(t) > 3` in _escape_fts5 — three-character code
             terms ("jwt", "sql", "api", "url") are silently dropped from the
             MATCH expression, which has the same flooding effect.
        Both are invisible to a `len(results) >= 1` style assertion; only an
        exact result set catches them.
        """
        file_id = _make_file(db, "src/auth/tokens.py")

        both_terms_body = "def decode_jwt_token(raw):\n    return jwt.decode(raw)"
        one_term_body = "def decode_base64_blob(raw):\n    return base64.b64decode(raw)"

        # Precondition: the fixture only discriminates while the second symbol
        # contains "decode" but no occurrence of "jwt" anywhere FTS5 indexes it
        # (name, qualified_name, docstring, body, context_summary).
        assert "decode" in one_term_body and "jwt" not in one_term_body.lower(), (
            "fixture 'decode_base64_blob' must contain 'decode' but never 'jwt'; "
            "otherwise an OR-joined MATCH would look identical to an AND-joined one"
        )

        both_id = _insert_symbol(db, file_id, "decode_jwt_token", both_terms_body)
        one_id = _insert_symbol(db, file_id, "decode_base64_blob", one_term_body)
        _insert_chunk(db, both_id, both_terms_body)
        _insert_chunk(db, one_id, one_term_body)

        results = bm25_search(db, "jwt decode", k=10)

        assert {r.symbol.name for r in results} == {"decode_jwt_token"}


class TestFallbackChunkBody:
    def test_fallback_chunk_carries_the_whole_body_up_to_2000_chars(self, db: Database) -> None:
        """
        MUTATIONS: `symbol.body[:2000]` -> `symbol.body[:0]` (empty) or
        `symbol.body[:200]` (truncated) in the synthetic-Chunk fallback. The
        pre-existing test asserted
        `body[:2000] in chunk_text or chunk_text in body[:2000]`, and the second
        half of that `or` is true for ANY prefix — including "" — so a
        chunk-less symbol could reach the assembler carrying no code text at
        all and the suite stayed green.
        """
        file_id = _make_file(db, "src/walk/orphan.py")

        short_body = "def walk_orphan_tree(node):\n" + "\n".join(
            f"    step_{i} = compute(node, {i})" for i in range(30)
        )
        long_body = "def walk_giant_tree(node):\n" + "\n".join(
            f"    stage_{i} = accumulate(node, {i})" for i in range(120)
        )
        assert 200 < len(short_body) < 2000, (
            "fixture 'short_body' must be longer than 200 chars and shorter than 2000 "
            f"to discriminate a truncating slice; it is {len(short_body)}"
        )
        assert len(long_body) > 2000, (
            f"fixture 'long_body' must exceed 2000 chars to pin the cap; it is {len(long_body)}"
        )

        # Symbols inserted with NO chunk row -> fallback path.
        short_id = _insert_symbol(db, file_id, "walk_orphan_tree", short_body)
        long_id = _insert_symbol(db, file_id, "walk_giant_tree", long_body)
        # Precondition: both symbols must reach bm25_search with NO chunk row, or the
        # synthetic-Chunk fallback under test is never entered and a truncating
        # `symbol.body[:N]` slice becomes invisible.
        #
        # This guard is not hypothetical. Every OTHER test in this file inserts chunks, so
        # omitting `_insert_chunk` here is exactly the kind of asymmetry a future edit
        # "tidies up". Adversarial review simulated that drift — added the two
        # `_insert_chunk` calls, re-applied the `body[:0]` mutation — and the file reported
        # `26 passed`: the mutant survived and this test silently stopped testing anything.
        for sym_id, label in ((short_id, "walk_orphan_tree"), (long_id, "walk_giant_tree")):
            assert db.get_first_chunk_for_symbol(sym_id) is None, (
                f"fixture {label!r} now has a chunk row, so bm25_search never builds the "
                "synthetic fallback Chunk and this test no longer pins symbol.body[:2000]"
            )

        short_results = bm25_search(db, "walk_orphan_tree", k=5)
        assert [r.symbol.name for r in short_results] == ["walk_orphan_tree"]
        assert short_results[0].chunk.chunk_text == short_body

        long_results = bm25_search(db, "walk_giant_tree", k=5)
        assert [r.symbol.name for r in long_results] == ["walk_giant_tree"]
        assert len(long_results[0].chunk.chunk_text) == 2000
        assert long_results[0].chunk.chunk_text == long_body[:2000]


class TestDeclarationBoostIsWired:
    def test_declaration_boost_weight_reaches_the_sql_layer(self, db: Database) -> None:
        """
        MUTATION: `declaration_boost_weight=declaration_boost_weight` ->
        `declaration_boost_weight=1.0` in the db.bm25_search() call. The
        argument is accepted, documented, and threaded in from
        RetrievalConfig.declaration_boost_weight — but dropping it on the floor
        made every caller's configured boost a silent no-op with a green suite.
        """
        file_id = _make_file(db, "src/pay/charge.py")

        declaration_body = "def process_payment(amount): pass"
        incidental_body = "def batch_orchestrator(amounts):\n" + "\n".join(
            f"    process_payment(amount_{i})  # calls process_payment" for i in range(15)
        )
        declaration_id = _insert_symbol(db, file_id, "process_payment", declaration_body)
        incidental_id = _insert_symbol(
            db,
            file_id,
            "batch_orchestrator",
            incidental_body,
            docstring=" ".join(["process_payment"] * 15),
        )
        _insert_chunk(db, declaration_id, declaration_body)
        _insert_chunk(db, incidental_id, incidental_body)
        for i in range(10):
            noise_id = _insert_symbol(
                db, file_id, f"unrelated_{i}", f"def unrelated_{i}(): return {i}"
            )
            _insert_chunk(db, noise_id, f"unrelated {i}")

        unboosted = [r.symbol.name for r in bm25_search(db, "process_payment", k=10)]
        # Precondition: without the boost this fixture reproduces the ranking
        # defect the boost exists to fix. If it stops doing so, the assertion
        # below proves nothing.
        assert unboosted.index("batch_orchestrator") < unboosted.index("process_payment"), (
            "fixture 'batch_orchestrator' no longer outranks the 'process_payment' "
            "declaration at weight 1.0, so a boost of 5.0 would change nothing"
        )

        boosted = [
            r.symbol.name
            for r in bm25_search(db, "process_payment", k=10, declaration_boost_weight=5.0)
        ]
        assert boosted.index("process_payment") < boosted.index("batch_orchestrator")


# ---------------------------------------------------------------------------
# Round 3: the two CONFIRMED open survivors in trelix.retrieval.bm25
#
# Both mutants below survived a 311-test selection
# (test_bm25 + test_retriever_core + test_store + test_federation + test_fusion)
# before these tests existed.
# ---------------------------------------------------------------------------


class TestDefaultResultBudgetIsTwenty:
    """
    Pins the DEFAULT value of `k` in bm25_search's signature.

    The pre-existing `test_respects_k_limit` passes `k=3` explicitly and asserts
    `len(results) <= 3`. An upper bound on an explicitly-passed k can never
    observe the default, which is why the mutant below survived.
    """

    def test_default_k_returns_exactly_twenty_results(self, db: Database) -> None:
        """
        MUTATION THIS MUST KILL: `k: int = 20` -> `k: int = 3` in the
        bm25_search signature (src/trelix/retrieval/bm25.py).

        Oracle is the OBSERVED result count from a corpus with strictly more
        than 20 matching symbols, with 20 written as a literal. Nothing is
        imported from the module under test.
        """
        file_id = _make_file(db)
        # 30 symbols that all match the query token "validate".
        for i in range(30):
            sym_id = _insert_symbol(
                db,
                file_id,
                f"validate_field_{i:02d}",
                f"def validate_field_{i:02d}(v): return validate(v)",
            )
            _insert_chunk(db, sym_id, f"validate field {i:02d}")

        # PRECONDITION: the corpus built above must hold MORE than 20 matches,
        # otherwise `== 20` below would be satisfied by corpus exhaustion rather
        # than by the default budget clamping. If this fires, the 30-symbol loop
        # in this test's `db` fixture setup stopped discriminating and the
        # assertion beneath it proves nothing.
        generous = bm25_search(db, "validate", k=30)
        assert len(generous) == 30, (
            "precondition failed: the 30-symbol corpus built in this test's `db` "
            f"fixture yielded only {len(generous)} matches for 'validate' at k=30, "
            "so a default-budget assertion of 20 would not discriminate"
        )

        # THE ASSERTION: call with NO k argument, so the signature default runs.
        defaulted = bm25_search(db, "validate")
        assert len(defaulted) == 20

    def test_default_k_budget_matches_explicit_twenty(self, db: Database) -> None:
        """
        MUTATION THIS MUST KILL: `k: int = 20` -> `k: int = 3`.

        Second, independent angle on the same default: omitting `k` must be
        indistinguishable from passing the literal 20 -- same count AND same
        ordered symbol names. Pins the default as a value, not just a count.
        """
        file_id = _make_file(db)
        for i in range(30):
            sym_id = _insert_symbol(
                db,
                file_id,
                f"resolve_import_{i:02d}",
                f"def resolve_import_{i:02d}(m): return resolve(m)",
            )
            _insert_chunk(db, sym_id, f"resolve import {i:02d}")

        # PRECONDITION: 20 is a real clamp here, not the whole corpus. Names the
        # fixture: the 30-symbol loop in this test's `db` setup.
        assert len(bm25_search(db, "resolve", k=30)) == 30, (
            "precondition failed: this test's `db` fixture corpus no longer holds "
            "30 'resolve' matches, so clamping to 20 is not observable"
        )

        explicit_twenty = [r.symbol.name for r in bm25_search(db, "resolve", k=20)]
        defaulted = [r.symbol.name for r in bm25_search(db, "resolve")]

        assert len(explicit_twenty) == 20
        assert defaulted == explicit_twenty


class TestFts5EmptyQuerySentinel:
    """
    Pins the empty-query sentinel returned by `_escape_fts5`.

    The pre-existing `test_empty_query_returns_empty_matcher` asserts
    `result in ('\"\"', '\"\"*')` -- a disjunction that accepts either branch -- and
    it probes `""`, which takes the *single-identifier prefix* branch and never
    reaches the sentinel at all. Hence the mutant below survived.

    NOTE, verified by running `_escape_fts5` rather than assumed: `""` and
    `"   "` do NOT reach the sentinel; both return '""*' from the prefix branch.
    Nor does every all-stop-word query: `"a an the"` returns 'the' and
    `"he she"` returns 'she', because the len(t) > 2 fallback keeps 3-char stop
    words. Only queries whose tokens are ALL punctuation, or all stop words of
    length <= 2, reach `return '""'`.
    """

    def test_sentinel_is_a_quoted_empty_string_for_every_triggering_input(self) -> None:
        """
        MUTATION THIS MUST KILL: `return '\"\"'` -> `return '\"a\"'` in
        _escape_fts5 (src/trelix/retrieval/bm25.py).

        Explicit expected table, compared as whole mappings in BOTH directions,
        so a changed return value AND a newly-diverging input both fail. The
        expected strings are literals -- nothing is imported from the module
        under test, and the inputs are not derived by iterating _STOP_WORDS.
        """
        # Every input here was confirmed by execution to reach the sentinel.
        expected = {
            "!!!": '""',
            "!!! ???": '""',
            "... ---": '""',
            "?? !!": '""',
            "@#$ %^&": '""',
            "-": '""',
            "+++ ***": '""',
            "is it": '""',
            "of to do be": '""',
            "it is so": '""',
            "no my we": '""',
            "a b c": '""',
        }
        actual = {q: _escape_fts5(q) for q in expected}

        # Set equality both ways over (input, output) pairs.
        assert set(actual.items()) == set(expected.items())
        assert set(expected.items()) == set(actual.items())

    def test_non_triggering_inputs_do_not_reach_the_sentinel(self) -> None:
        """
        MUTATION THIS MUST KILL: none on its own -- this is the discriminating
        control for the test above. It pins the CURRENT boundary of the sentinel
        so that `return '\"\"'` cannot be trivially satisfied by every input.

        Documents CURRENT behaviour that is arguably a DEFECT: `""` and `"   "`
        fall into the single-identifier *prefix* branch and yield '""*', and
        `"a an the"` / `"he she"` leak a bare 3-char stop word as the FTS5
        expression. Neither looks intentional, but both are what ships today.
        """
        assert _escape_fts5("") == '""*'
        assert _escape_fts5("   ") == '""*'
        # Stop words are excluded from the fallback too, so an all-stop-word
        # query hits the FTS5 sentinel instead of leaking a raw stop word.
        assert _escape_fts5("a an the") == '""'
        assert _escape_fts5("he she") == '""'

    def test_sentinel_query_matches_no_document_containing_the_token_a(self, db: Database) -> None:
        """
        MUTATION THIS MUST KILL: `return '\"\"'` -> `return '\"a\"'`.

        The search-behaviour half of the oracle: a test on the return string
        alone pins an implementation detail, so this asserts the observable
        consequence. The corpus deliberately contains the standalone token "a",
        which the mutant's '"a"' expression would match.
        """
        file_id = _make_file(db)
        sym_id = _insert_symbol(
            db,
            file_id,
            "alpha_handler",
            "def alpha_handler(x):\n    # a is a standalone token a here\n    return a",
            docstring="a a a standalone token a",
        )
        _insert_chunk(db, sym_id, "a a a standalone token a")

        # PRECONDITION: the corpus above must actually be matchable by the
        # mutant's expression, otherwise the empty-result assertions below would
        # hold for the mutant too and prove nothing. If this fires, the
        # "alpha_handler" symbol inserted in this test's `db` setup no longer
        # indexes a standalone "a" token and this test has stopped discriminating.
        assert db.bm25_search('"a"', limit=10), (
            "precondition failed: the `alpha_handler` symbol built in this test's "
            "`db` fixture is not matched by the FTS5 expression '\"a\"', so this "
            "test can no longer distinguish the '\"\"' sentinel from '\"a\"'"
        )

        # Every sentinel-triggering query must return NOTHING, exactly.
        for query in ("!!!", "!!! ???", "is it", "of to do be", "a b c", "@#$ %^&"):
            assert bm25_search(db, query, k=10) == [], (
                f"sentinel query {query!r} matched documents; the FTS5 "
                "empty-query sentinel is no longer a quoted empty string"
            )
