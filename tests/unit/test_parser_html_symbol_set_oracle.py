"""Exact-set oracles for the HTML extractor.

Mutation-verified. The pre-existing HTML tests in
``tests/unit/test_parser_zero_coverage.py`` assert membership only
(``"about" in names``, ``len(result.symbols) >= 1``), so 36 of 40 hand mutations
to ``indexing/parser/extractors/html.py`` were green. Every expected value below
is written as a LITERAL — nothing is imported from the module under test, and no
collection inside the module is iterated to build an expectation, because that is
what let members be deleted from ``walker.EXTENSION_MAP`` in silence.

Each test asserts set/list equality in BOTH directions, so an added symbol or a
duplicated edge fails just as loudly as a missing one.
"""

from __future__ import annotations

import pytest

from trelix.core.models import SymbolKind
from trelix.indexing.parser.extractors.html import HtmlParser

FILE_ID = 7


@pytest.fixture()
def html() -> HtmlParser:
    return HtmlParser()


# ===========================================================================
# 1. <link> / <script src> dependency edges
# ===========================================================================

LINK_DOC = """\
<!DOCTYPE html>
<html>
<head>
  <link rel="stylesheet" href="/css/site.css">
  <link rel="preload" href="/fonts/inter.woff2">
  <link rel="modulepreload" href="/js/app.mjs">
  <link rel="import" href="/components/card.html">
  <link rel="STYLESHEET" href="/css/upper.css">
  <link rel="icon" href="/favicon.ico">
  <link rel="stylesheet">
  <script src="/static/app.js?v=2"></script>
</head>
<body></body>
</html>
"""

# Written out by hand, one entry per rel value the extractor is supposed to
# treat as a dependency, plus the external script. /favicon.ico (rel="icon")
# and the href-less stylesheet must NOT appear.
EXPECTED_LINK_IMPORTS = [
    "/components/card.html",
    "/css/site.css",
    "/css/upper.css",
    "/fonts/inter.woff2",
    "/js/app.mjs",
    "/static/app.js?v=2",
]


def test_link_and_script_import_edges_are_an_exact_set(html: HtmlParser) -> None:
    """Kills: deleting any member of the ``rel in ("stylesheet", "preload",
    "modulepreload", "import")`` tuple in ``_handle_element``; changing
    ``if href and rel in (...)`` to ``or``; dropping ``.lower()`` from
    ``rel = attrs.get("rel", "").lower()``.
    """
    # Precondition: LINK_DOC must keep the cases that discriminate. If these
    # are edited away, the assertion below stops proving anything.
    assert 'rel="icon" href="/favicon.ico"' in LINK_DOC, (
        "LINK_DOC lost its non-dependency rel case; the exact-set assertion "
        "can no longer detect an over-broad rel check"
    )
    assert '<link rel="stylesheet">' in LINK_DOC, (
        "LINK_DOC lost its href-less <link>; the exact-set assertion can no "
        "longer detect 'href and' being weakened to 'href or'"
    )
    assert 'rel="STYLESHEET"' in LINK_DOC, (
        "LINK_DOC lost its uppercase rel; the exact-set assertion can no "
        "longer detect a dropped .lower() on the rel attribute"
    )

    result = html.parse(LINK_DOC, file_id=FILE_ID)

    # sorted list, not a set: a duplicated edge must fail too.
    assert sorted(e.imported_from for e in result.import_edges) == EXPECTED_LINK_IMPORTS
    assert [list(e.imported_names) for e in result.import_edges] == [[]] * 6
    assert [e.file_id for e in result.import_edges] == [FILE_ID] * 6


# ===========================================================================
# 2. inline <script> / <style> blocks
# ===========================================================================

SCRIPT_STYLE_DOC = """\
<html>
<head>
  <script src="/static/app.js"></script>
  <script type="module">import "./m.js";
console.log("boot");</script>
  <script></script>
  <style>body { color: red; }</style>
  <style scoped></style>
</head>
<body></body>
</html>
"""

# (name, line_start, line_end, signature, body) — hand-written literals.
# The empty <script></script> and the empty <style scoped></style> are absent
# on purpose: empty blocks carry nothing to index.
EXPECTED_SCRIPT_STYLE_SECTIONS = {
    ("app.js", 3, 3, '<script src="/static/app.js">', '<script src="/static/app.js">'),
    (
        "inline-script",
        4,
        5,
        '<script type="module">',
        'import "./m.js";\nconsole.log("boot");',
    ),
    ("inline-style", 7, 7, "<style>", "body { color: red; }"),
}


def test_script_and_style_sections_are_an_exact_set(html: HtmlParser) -> None:
    """Kills: ``if not raw and not script_src: return`` -> ``or`` (every inline
    script becomes invisible); ``if not raw: return`` -> ``if raw: return`` in
    ``_handle_style`` (every style block becomes invisible);
    ``script_src.split("/")[-1]`` -> ``[0]``; ``line_start=node.start_point[0]
    + 1`` -> ``+ 0`` in ``_handle_script``; ``body=raw[:1000] if raw else sig``
    -> ``body=sig``.
    """
    # Precondition: the empty script and empty style are what prove the
    # emptiness guards are still guards and not inverted no-ops.
    assert "<script></script>" in SCRIPT_STYLE_DOC, (
        "SCRIPT_STYLE_DOC lost its empty <script>; the exact-set assertion can "
        "no longer detect an inverted emptiness guard"
    )
    assert "<style scoped></style>" in SCRIPT_STYLE_DOC, (
        "SCRIPT_STYLE_DOC lost its empty <style>; the exact-set assertion can "
        "no longer detect an inverted emptiness guard"
    )

    result = html.parse(SCRIPT_STYLE_DOC, file_id=FILE_ID)

    sections = {
        (s.name, s.line_start, s.line_end, s.signature, s.body)
        for s in result.symbols
        if s.kind is SymbolKind.SECTION
    }
    assert sections == EXPECTED_SCRIPT_STYLE_SECTIONS
    # One MODULE page symbol plus exactly those three SECTIONs — nothing else,
    # so an extra symbol for an empty block fails here.
    assert len(result.symbols) == 4
    assert [s.kind for s in result.symbols].count(SymbolKind.MODULE) == 1


# ===========================================================================
# 3. id / aria-label element extraction policy
# ===========================================================================

ID_DOC = """\
<html>
<body>
  <div id="content">main body</div>
  <span id="badge">3</span>
  <table id="grid"></table>
  <li id="row1">r</li>
  <section id="about">a</section>
  <nav aria-label="Primary">n</nav>
  <main id="app-main">m</main>
  <!-- the login form -->
  <form id="login" action="/login"></form>
  <form [formGroup]="profileForm" id="profile"></form>
</body>
</html>
"""

# (name, kind, line_start, signature, docstring) — hand-written literals.
# <span id="badge"> and <li id="row1"> are absent on purpose: inline tags are
# skipped. <nav> and <main> must be SECTION, not VARIABLE.
EXPECTED_ID_SYMBOLS = {
    ("content", SymbolKind.VARIABLE, 3, '<div id="content">', None),
    ("grid", SymbolKind.VARIABLE, 5, '<table id="grid">', None),
    ("about", SymbolKind.SECTION, 7, '<section id="about">', None),
    ("Primary", SymbolKind.SECTION, 8, '<nav aria-label="Primary">', None),
    ("app-main", SymbolKind.SECTION, 9, '<main id="app-main">', None),
    ("login", SymbolKind.SECTION, 11, '<form id="login" action="/login">', "the login form"),
    ("profileForm", SymbolKind.SECTION, 12, '<form [formGroup]="profileForm">', None),
}


def test_id_and_aria_element_symbols_are_an_exact_set(html: HtmlParser) -> None:
    """Kills: ``elif tag not in _SKIP_ID_TAGS`` -> ``tag in _SKIP_ID_TAGS``
    (every <div id> disappears and every <span id> appears); deleting "main" or
    "nav" from ``_SECTION_TAGS``; ``label = form_group or form_id`` ->
    ``form_id or form_group``; ``line_start = node.start_point[0] + 1`` ->
    ``+ 0``; ``prev.type == "comment"`` -> ``!=`` in
    ``_get_preceding_comment``.
    """
    # Precondition: the skipped inline tags must stay in the fixture, otherwise
    # this test cannot tell an inverted skip-list from a correct one.
    assert '<span id="badge">' in ID_DOC, (
        "ID_DOC lost its <span id>; the exact-set assertion can no longer "
        "detect an inverted _SKIP_ID_TAGS membership test"
    )
    assert '<li id="row1">' in ID_DOC, (
        "ID_DOC lost its <li id>; the exact-set assertion can no longer "
        "detect an inverted _SKIP_ID_TAGS membership test"
    )
    assert "<!-- the login form -->" in ID_DOC, (
        'ID_DOC lost the comment above <form id="login">; the docstring '
        "assertion can no longer detect a broken preceding-comment lookup"
    )

    result = html.parse(ID_DOC, file_id=FILE_ID)

    extracted = {
        (s.name, s.kind, s.line_start, s.signature, s.docstring)
        for s in result.symbols
        if s.kind is not SymbolKind.MODULE
    }
    assert extracted == EXPECTED_ID_SYMBOLS
    assert len(result.symbols) == 8  # 7 above + the page MODULE symbol


# ===========================================================================
# 4. custom elements: dedup, self-closing tags, framework bindings
# ===========================================================================

COMPONENT_DOC = """\
<html>
<body>
  <app-root></app-root>
  <app-root></app-root>
  <user-card *ngFor="let u of users" [routerLink]="/u" formControlName="name" \
(select)="pick(u)" [disabled]="busy"></user-card>
  <vue-list v-for="i in items" v-model="sel" :size="n" @tap="go"></vue-list>
  <my-icon />
</body>
</html>
"""

# Document order, one entry per UNIQUE hyphenated tag. app-root appears twice
# in the source and must yield exactly one symbol; <my-icon /> is self-closing
# and must still yield one.
EXPECTED_COMPONENT_NAMES = ["app-root", "user-card", "vue-list", "my-icon"]

EXPECTED_USER_CARD_DECORATORS = [
    '*ngFor="let u of users"',
    '[routerLink]="/u"',
    'formControlName="name"',
]
EXPECTED_USER_CARD_SIGNATURE = (
    '<user-card *ngFor="let u of users" [routerLink]="/u" '
    'formControlName="name" (select)="pick(u)">'
)
EXPECTED_VUE_LIST_DECORATORS = ['v-for="i in items"', 'v-model="sel"']
EXPECTED_VUE_LIST_SIGNATURE = '<vue-list v-for="i in items" v-model="sel" :size="n" @tap="go">'


def test_custom_element_symbols_dedup_and_carry_framework_decorators(
    html: HtmlParser,
) -> None:
    """Kills: ``MAX_CUSTOM_ELEMS = 150`` -> ``1``; deleting
    ``seen_custom_tags.add(tag)`` (repeated tags emit duplicate symbols);
    dropping "self_closing_tag" from ``_get_start_or_self_closing_tag``;
    deleting ``decorators.append(part)`` from the ``_NG_STRUCTURAL`` branch;
    deleting "v-for" from ``_VUE_STRUCTURAL``; ``" ".join(key_attrs[:4])``
    -> ``[:1]``.
    """
    # Precondition: the duplicate tag and the self-closing tag are the whole
    # point of this fixture.
    assert COMPONENT_DOC.count("<app-root>") == 2, (
        "COMPONENT_DOC no longer repeats <app-root>; the dedup assertion "
        "would pass even with deduplication removed"
    )
    assert "<my-icon />" in COMPONENT_DOC, (
        "COMPONENT_DOC lost its self-closing element; this test can no longer "
        "detect self_closing_tag being dropped from the tag lookup"
    )

    result = html.parse(COMPONENT_DOC, file_id=FILE_ID)

    components = [s for s in result.symbols if s.kind is SymbolKind.VARIABLE]
    assert [s.name for s in components] == EXPECTED_COMPONENT_NAMES
    assert len(result.symbols) == 5  # 4 components + the page MODULE symbol

    by_name = {s.name: s for s in components}
    assert list(by_name["user-card"].decorators) == EXPECTED_USER_CARD_DECORATORS
    assert by_name["user-card"].signature == EXPECTED_USER_CARD_SIGNATURE
    assert list(by_name["vue-list"].decorators) == EXPECTED_VUE_LIST_DECORATORS
    assert by_name["vue-list"].signature == EXPECTED_VUE_LIST_SIGNATURE
    assert list(by_name["app-root"].decorators) == []
    assert list(by_name["my-icon"].decorators) == []
