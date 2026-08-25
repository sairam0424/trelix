"""Property: `_escape_fts5`'s multi-word AND-join round-trips its own meaningful
tokens, and `count_meaningful_tokens` agrees with it, across ARBITRARY synthetic
tokens -- not just the hand-picked "jwt token"-style cases in
tests/unit/test_bm25.py (TestMultiWordQueryPrecision, TestStopWordListIsPinned).

Read first: src/trelix/retrieval/bm25.py's `_escape_fts5` and
`count_meaningful_tokens`, and tests/unit/test_bm25.py in full (849 lines). That
file already hand-pins: the empty-query sentinel, the single-identifier prefix
branch, the AND-vs-OR join, the `len(t) > 2` boundary, and all 80 pinned
length>=3 stop words. What it does NOT sweep is the join/count agreement over
MANY combinations of token count, length, and content -- which is exactly the
shape of defect that survived until TestMultiWordQueryPrecision was written
(two independent single-line mutations, both invisible to `len(results) >= 1`).

Token construction avoids the existing suite's territory entirely: every
generated token is forced to contain at least one digit (`_DIGIT_TOKEN`
below), which makes it structurally impossible to collide with any of
`_STOP_WORDS` -- every one of the 102 entries there is pure alphabetic. So this
property never needs to reason about the stop-word table at all; it is a
disjoint discriminator from TestStopWordListIsPinned by construction, verified
below as its own precondition.

FALSIFYING INPUT CONFIRMED BY HAND (see PROOF PROTOCOL / round notes): tokens
["ab1", "cd2"] joined "ab1 cd2". Unmutated: `_escape_fts5("ab1 cd2").split()`
== ["ab1", "cd2"]. Mutating `" ".join(tokens)` -> `" OR ".join(tokens)` in
_escape_fts5 (the exact mutation TestMultiWordQueryPrecision's docstring
documents as having survived the pre-existing suite) turns this into
"ab1 OR cd2", whose .split() is ["ab1", "OR", "cd2"] -- length 3, not 2, so the
round-trip assertion fails. Verified by actually mutating and reverting the
source, output pasted in the round report.

DERANDOMIZED: all three `@settings` below pin `derandomize=True` -- fixed,
hash-of-test-function example sequence, not a fresh random seed per run.
`max_examples` is unchanged for all three (25, 40, 40): re-running the
AND-join mutation above under the pinned seed still catches it on every one
of three consecutive runs (see round report).
"""

from __future__ import annotations

import string

from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from trelix.retrieval.bm25 import _STOP_WORDS, _escape_fts5, count_meaningful_tokens

# Every token: 1-6 lowercase letters, then exactly one digit, then 1-4 more
# lowercase-letter-or-digit characters. Min length is 1+1+1=3 (satisfies both
# functions' `len(t) > 2` / `len(t) >= 3` floor), first char is always a letter
# (satisfies the `[a-zA-Z_][a-zA-Z0-9_]*` token regex both functions use), and
# the mandatory digit makes every token structurally unable to equal any of
# _STOP_WORDS' 102 pure-alphabetic entries.
_DIGIT_TOKEN = st.builds(
    lambda letters, digit, tail: letters + digit + tail,
    letters=st.text(alphabet=string.ascii_lowercase, min_size=1, max_size=6),
    digit=st.text(alphabet=string.digits, min_size=1, max_size=1),
    tail=st.text(alphabet=string.ascii_lowercase + string.digits, min_size=1, max_size=4),
)

_TOKEN_LIST = st.lists(_DIGIT_TOKEN, min_size=2, max_size=6)


def _query_from_tokens(tokens: list[str]) -> str:
    return " ".join(tokens)


class TestTokenGenerationIsDisjointFromStopWords:
    """Precondition control: if this ever fails, every property below is testing
    a query that could hit the *existing* stop-word-filtering code path instead
    of the AND-join/count path this file targets, and would stop discriminating.
    """

    @given(tokens=_TOKEN_LIST)
    @settings(derandomize=True, max_examples=25, deadline=None)
    def test_generated_tokens_never_collide_with_a_stop_word(self, tokens: list[str]) -> None:
        for token in tokens:
            assert token.lower() not in _STOP_WORDS, (
                f"generated token {token!r} collides with _STOP_WORDS; the digit-forcing "
                "strategy in this file no longer guarantees disjointness from the stop-word "
                "table, so TestEscapeFts5RoundTrip below may be exercising the wrong branch"
            )


class TestEscapeFts5RoundTrip:
    """Fails under: `\" \".join(tokens)` -> `\" OR \".join(tokens)` in _escape_fts5
    (turns AND into OR, injecting literal "OR" tokens into the split-back list);
    or `len(t) > 2` -> `len(t) > 3` (drops any generated token of length exactly 3).
    """

    @settings(
        derandomize=True,
        max_examples=40,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    @example(tokens=["ab1", "cd2"])  # the hand-verified case
    @example(tokens=["a1b", "c2d", "e3f"])  # three length-3 tokens, boundary-sensitive
    @given(tokens=_TOKEN_LIST)
    def test_multiword_and_join_round_trips_every_token_in_order(self, tokens: list[str]) -> None:
        query = _query_from_tokens(tokens)
        result = _escape_fts5(query)

        assert result.split() == tokens, (
            f"_escape_fts5({query!r}) = {result!r}; expected the AND-joined tokens back "
            f"in order, got {result.split()!r}"
        )


class TestCountMeaningfulTokensAgreesWithEscape:
    """Fails under: any drift between the two independent `re.findall` + filter
    implementations in count_meaningful_tokens and _escape_fts5 -- e.g. changing
    one function's length floor (`>= 3` / `> 2`) without the other, which a
    hand-written test asserting only one function's output would never catch.
    """

    @settings(
        derandomize=True,
        max_examples=40,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    @example(tokens=["ab1", "cd2"])
    @given(tokens=_TOKEN_LIST)
    def test_count_matches_the_number_of_tokens_escape_fts5_keeps(self, tokens: list[str]) -> None:
        query = _query_from_tokens(tokens)

        assert count_meaningful_tokens(query) == len(_escape_fts5(query).split()), (
            f"count_meaningful_tokens({query!r}) = {count_meaningful_tokens(query)} but "
            f"_escape_fts5({query!r}).split() has {len(_escape_fts5(query).split())} tokens "
            f"({_escape_fts5(query).split()!r}); the two tokenizers disagree"
        )
