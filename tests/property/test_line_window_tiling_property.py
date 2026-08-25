"""Property (round's 6th target, own choice): `LineWindowParser.parse()`'s
emitted SECTION windows exactly TILE the source -- no gaps, no overlaps,
1-indexed-inclusive line ranges that reconstruct the original lines verbatim
-- across ARBITRARY line counts and window sizes, not just the 36
hand-constructed mutants tests/unit/test_line_window_parser_bounds.py already
pins (read in full first; its own docstring table lists all 36).

That file's docstring states the ONE deliberately-unkillable survivor
(deleting the `if not lines: return ...` early return is an equivalent
mutant) and names, per test, the single-line change it kills. What none of
those 36 do is vary the LINE COUNT and WINDOW SIZE together across many
combinations while checking a GLOBAL structural invariant across the whole
returned symbol list -- each hand test targets one specific boundary
(exact-fit windows, the char ceiling, the floor clamps) in isolation.

This file avoids that file's fixtures entirely: every generated line here is
short, single-line, and non-blank by construction (so no window is ever
dropped by the `if body.strip()` filter and every candidate window becomes an
observable symbol) -- deliberately staying OUT of the char-ceiling-shrink
territory that test file already exhaustively covers by hand.

FALSIFYING INPUT CONFIRMED BY HAND (see PROOF PROTOCOL below): 5 lines
["l0","l1","l2","l3","l4"], window_lines=2. Unmutated: 3 windows, line ranges
(1,2),(3,4),(5,5), and reconstructing each symbol's body and concatenating
gives back the original 5 lines in order. Mutating `line_start=start + 1` ->
`line_start=start` (drops the +1) breaks BOTH the "first window starts at
line 1" assertion (it would start at 0) AND the "consecutive windows tile
with no gap" arithmetic; confirmed by actually mutating and reverting the
source, output pasted in the round report.

DERANDOMIZED: the `@settings` below pins `derandomize=True` -- fixed,
hash-of-test-function example sequence, not a fresh random seed per run.
`max_examples=40` is unchanged: re-running the `line_start=start + 1` ->
`line_start=start` mutation above under the pinned seed still catches it on
every one of three consecutive runs (see round report).
"""

from __future__ import annotations

import string

from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from trelix.indexing.parser.extractors.line_window import LineWindowParser

# Non-blank, single-line, no-newline content. Deliberately narrow (max 30
# chars) so no generated example ever approaches _MAX_WINDOW_CHARS (1_600) or
# even a shrunk `max_window_chars` floor (80) -- keeping this file entirely
# out of the char-ceiling-shrink code path that
# tests/unit/test_line_window_parser_bounds.py already covers by hand.
_LINE_ALPHABET = string.ascii_lowercase + string.digits + " _"
_NONBLANK_LINE = st.text(alphabet=_LINE_ALPHABET, min_size=1, max_size=30).filter(
    lambda s: s.strip() != ""
)


class TestWindowsTileTheSourceExactly:
    """Fails under: `line_start=start + 1` -> `line_start=start` (breaks both the
    line-1 start and the consecutive-tiling arithmetic below); `window_number
    += 1` moved above the emit guard (creates a skip visible only via names, not
    checked here -- see docstring); `lines[start:end]` -> `lines[start + 1:end]`
    in the char-ceiling shrink loop (not reachable by these short lines, so this
    property does not claim to catch it -- that mutation is
    test_line_window_parser_bounds.py's territory).
    """

    @settings(
        derandomize=True,
        max_examples=40,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    @example(lines=["l0", "l1", "l2", "l3", "l4"], window_lines=2)  # hand-verified case
    @given(
        lines=st.lists(_NONBLANK_LINE, min_size=1, max_size=40),
        window_lines=st.integers(min_value=1, max_value=15),
    )
    def test_symbols_reconstruct_the_source_with_no_gap_or_overlap(
        self, lines: list[str], window_lines: int
    ) -> None:
        source = "\n".join(lines)
        parser = LineWindowParser(window_lines=window_lines)
        result = parser.parse(source, file_id=1)

        assert result.symbols, "every line here is non-blank, so at least one window must emit"

        # 1) First window starts at line 1 (1-indexed).
        assert result.symbols[0].line_start == 1

        # 2) Last window ends at the last line.
        assert result.symbols[-1].line_end == len(lines)

        # 3) Consecutive windows tile with NO gap and NO overlap.
        for prev, nxt in zip(result.symbols, result.symbols[1:]):
            assert nxt.line_start == prev.line_end + 1, (
                f"gap or overlap between windows: {prev.line_start}-{prev.line_end} "
                f"then {nxt.line_start}-{nxt.line_end}"
            )

        # 4) Every window's own range is non-degenerate.
        for sym in result.symbols:
            assert sym.line_start <= sym.line_end

        # 5) Reconstruction: concatenating every window's body (split on the
        # same "\n" it was joined with) gives back the exact original lines,
        # in order -- the strongest, single round-trip check.
        reconstructed: list[str] = []
        for sym in result.symbols:
            reconstructed.extend(sym.body.split("\n"))
        assert reconstructed == lines

        # 6) The signature is the first line of the window, contract-checked
        # against the SAME `lines` list this test built (not re-derived from
        # the module under test).
        for sym in result.symbols:
            expected_signature = lines[sym.line_start - 1].strip()[:200]
            assert sym.signature == expected_signature
