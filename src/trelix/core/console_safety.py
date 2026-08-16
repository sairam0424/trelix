"""Sanitiser for untrusted content rendered to a Rich console.

Lives here rather than in `cli/main.py` because `indexing/indexer.py` needs it too and
cannot import from the CLI — `cli.main` imports the indexer, so that direction is a cycle.
Both modules construct a markup-enabled `Console`, and they are the only two in `src/`
that do (`grep -rln "Console(" src/trelix/`), which makes this the complete set of callers.

That second caller is why this module exists. The sanitiser originally shipped in
`cli/main.py` alone, and the release notes then claimed it was "applied to every dynamic
value rendered to the terminal" — which was false for `trelix index`, where four sites
rendered repo-controlled filenames through bare `escape()`. An absent fix is better than a
fix the notes over-claim, because only the second one stops people looking.
"""

from __future__ import annotations

import re

from rich.markup import escape

# C0 and C1 control characters, minus the two that legitimately structure text.
# \n and \t are preserved because retrieved source is multi-line and indented.
# \r is in the stripped range, but credit where due: Rich already normalises a
# lone carriage return on its own (measured — `escape("real\rFAKE")` renders
# "realFAKE" with no CR), so removing it here is belt-and-braces rather than the
# thing standing between you and a cursor-return overwrite.
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def safe_text(value: object) -> str:
    """Escape Rich markup AND strip terminal control bytes. For UNTRUSTED content.

    `rich.markup.escape()` neutralises "[tag]" brackets and nothing else, so a raw
    ESC byte in an indexed file travels through `console.print()` straight to the
    terminal, which then executes it. Verified against a live Rich console — these
    all reached the terminal intact before this helper existed:

      * `\\x1b]52;c;<base64>\\x07` — OSC 52. **Writes the user's clipboard** on
        iTerm2, kitty and WezTerm. Content in a repository should not be able to
        put data on your clipboard.
      * `\\x1b[1A\\x1b[2K` — cursor-up plus erase-line. Scrubs trelix's own output
        above it, so injected content can hide the fact it was ever displayed.
      * `\\x1b[?25l`, BEL, and backspace-overwrite tricks.

    Stripping rather than escaping is deliberate: there is no rendering of a raw
    control byte that is useful to a human reading source code, and no tracked file
    in this repository contains one, so the change is invisible for real code. It
    is visible for terminal recordings and ANSI-parser fixtures — those render as
    their printable remainder.

    Apply this ONLY to indexed, LLM or third-party content. Two things must not go
    through it: trelix's own literal markup (which has to stay interpretable), and
    anything machine-readable — `_print_json()` keeps its payload byte-exact, and
    that invariant is pinned by test_search_json_output_is_not_escaped.

    DO NOT rely on Rich's highlighter for this. With the default `highlight=True`,
    Rich's reprhighlighter happens to insert SGR codes between the ESC and its
    payload — `\\x1b]52;…` renders as `\\x1b\\x1b[1m]\\x1b[0m\\x1b[1;36m52…` — so
    multi-token sequences are incidentally broken up. That is an accident, not a
    control: it is undone by adding `highlight=False` anywhere, which is an
    otherwise innocuous change, and it does not help for short sequences with
    nothing to tokenise. Measured through a default-highlighting console, these
    survive intact and need this helper:

        \\x1bc    RIS — full terminal reset
        \\x1b#8   DECALN — fills the screen with 'E'
        \\x1b=    application keypad mode

    ORDER IS LOAD-BEARING: strip FIRST, then escape. The reverse composes into a
    worse bug than either fix prevents. `rich.markup.escape()` matches
    ``(\\\\*)(\\[[a-z#/@][^[]*?])`` — a byte that is not in ``[a-z#/@]`` right after
    the ``[`` means no tag is recognised and the bracket is left UNESCAPED. Strip
    that byte afterwards and you have SYNTHESISED a live markup tag that escape()
    never got the chance to neutralise. Demonstrated with a real pty against a real
    `trelix search`: a symbol name of

        pre[\\x1blink=http://evil.example]CLICK[\\x1b/link]post

    escaped to ``pre[\\x1blink=…]`` (bracket untouched, since \\x1b is not a tag
    start), then stripped to ``pre[link=http://evil.example]…`` — which Rich
    rendered as a genuine OSC 8 hyperlink to the attacker's URL, with the
    ``[link=…]`` markers invisible so the reader could not tell. Any of the 33
    stripped codepoints works, not just ESC. Stripping first removes the byte while
    the bracket is still bare, so escape() then sees ``[link=…]`` and neutralises
    it.
    """
    return escape(_CONTROL_RE.sub("", str(value)))
