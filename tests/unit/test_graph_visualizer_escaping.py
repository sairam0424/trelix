"""SEC-04 — repo-controlled text must not become executable HTML in graph.html.

Attacker position. The adversary is any repository trelix indexes and the attacker can
write to: a vendored SDK's README, a `third_party/` subtree, a PR branch, a monorepo
directory owned by another team. No write access to trelix itself is needed. The victim
is whoever opens the generated `.trelix/graph.html` — a file the CLI prints the path of
as an invitation, and which carries up to 500 nodes of symbol names, qualified names and
file paths of a private repository.

Three carriers, all driven here through the REAL pipeline rather than a hand-built graph:

  1. a markdown HEADING becomes `Symbol.qualified_name` (MarkdownParser) and is
     interpolated into the node tooltip;
  2. a FILENAME becomes `IndexedFile.rel_path` (FileWalker) and is interpolated into the
     same tooltip;
  3. `generic_edges.source_ref` becomes a *string* node id (CodeGraph), which pyvis emits
     as `<option value="{{node.id}}">{{node.id}}</option>` — a second sink that needs no
     hover, that carries no JSON encoding at all, and that escaping the tooltip alone
     leaves wide open. The connectors take that ref verbatim from a Jira/Linear JSON
     response.

Two controls that look like a fix and are not, both asserted around here:

  - pyvis's JSON encoder. `{{nodes|tojson}}` turns `<` into `\\u003c`, so grepping the raw
     file for `<img` "passes" while the payload is fully intact. Every tooltip assertion
     below therefore json.loads the `vis.DataSet` payload and asserts on the DECODED
     value — the string pyvis's custom popup hands to `innerHTML`.
  - `pyvis.utils.check_html`, which validates only the output filename's extension.

No network, no browser and no executable payload is written anywhere: the assertions are
on the escaping of the generated text.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

# The REAL exporter, or nothing. Two reasons this is `importorskip` at module scope and
# not a lazy import inside the tests: the fix only exists on the pyvis path, and
# tests/unit/test_graph_visualizer.py installs a MagicMock into `sys.modules["pyvis"]`
# and `["pyvis.network"]` with setdefault(). Importing the real package (and the real
# submodule — `import pyvis` alone does not bind `pyvis.network`) during collection makes
# that setdefault a no-op, so these assertions can never be met by a mock that writes
# "<html><body>graph</body></html>" and would satisfy nothing.
pytest.importorskip("pyvis.network", reason="SEC-04 lives in the pyvis HTML export")

from trelix.core.config import IndexConfig  # noqa: E402
from trelix.core.models import GenericEdge  # noqa: E402
from trelix.graph.code_graph import CodeGraph  # noqa: E402
from trelix.graph.visualizer import GraphVisualizer  # noqa: E402
from trelix.indexing.parser.extractors.markdown import MarkdownParser  # noqa: E402
from trelix.indexing.walker import FileWalker  # noqa: E402
from trelix.store.db import Database  # noqa: E402

# `href` is load-bearing, not decoration: pyvis switches its whole template to an
# innerHTML popup when ANY node title contains that substring (network.py:449-460), so
# the payload turns on the sink it then uses.
_HEADING_PAYLOAD = '<a href="#" onclick="alert(1)">pwn</a>'
# Legal on APFS and on ext4 — only "/" and NUL are not.
_FILENAME_PAYLOAD = 'readme<img src=x onerror="alert(2)">.md'
# Breaks out of the attribute in `<option value="...">`, then out of the element.
_SOURCE_REF_PAYLOAD = 'ticket:"><img src=x onerror="alert(3)">'

# The exporter authors exactly these three tags itself. Everything else in a tooltip is
# repo text, so this is the whitelist a "no unescaped `<`" assertion subtracts.
_OWN_TOOLTIP_TAGS = ("<b>", "</b>", "<br>")

# A well-formed option: the id can contain neither the quote that ends the attribute nor
# the angle brackets that would end the element. Written as a whole-line match so a
# breakout cannot hide in the part of the line the assertion forgot to look at.
_OPTION_RE = re.compile(r'<option value="([^"<>]*)">([^<>]*)</option>')


def _export(tmp_path: Path) -> str:
    """Index a hostile repo for real, build the real graph, run the real exporter."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / _FILENAME_PAYLOAD).write_text(f"# {_HEADING_PAYLOAD}\n\nsome prose\n")

    config = IndexConfig(repo_path=str(repo))
    db = Database(tmp_path / "index.db")

    symbol_ids: list[int] = []
    for indexed in FileWalker(config).walk():
        file_id = db.upsert_file(indexed)
        parsed = MarkdownParser().parse(Path(indexed.path).read_text(), file_id=file_id)
        for symbol in parsed.symbols:
            symbol_ids.append(db.insert_symbol(symbol))

    assert symbol_ids, "the walker/parser produced no symbols — the payloads never reached a node"

    # Path B: the connector-supplied artifact ref, which becomes a string node id.
    db.insert_generic_edges(
        [
            GenericEdge(
                from_symbol_id=symbol_ids[0],
                source_ref=_SOURCE_REF_PAYLOAD,
                edge_kind="references_ticket",
            )
        ]
    )

    out = str(tmp_path / "graph.html")
    return GraphVisualizer().export_html(CodeGraph(db), out)


def _dataset(text: str, name: str) -> list[dict]:
    """Decode one `new vis.DataSet([...])` payload out of the generated page.

    Asserting on the raw file would credit pyvis's `\\u003c` JSON escaping as a fix. The
    decoded value is what `popup.innerHTML = nodeData[0].title` receives.
    """
    match = re.search(rf"{name} = new vis\.DataSet\((.*)\);", text)
    assert match is not None, f"no vis.DataSet payload for {name} in the generated page"
    payload = json.loads(match.group(1))
    assert isinstance(payload, list)
    return payload


def _strip_own_tags(title: str) -> str:
    for tag in _OWN_TOOLTIP_TAGS:
        title = title.replace(tag, "")
    return title


class TestGeneratedGraphEscapesRepoText:
    def test_markdown_heading_cannot_inject_a_tag_into_the_tooltip(self, tmp_path: Path) -> None:
        """Carrier 1: heading -> qualified_name -> node title -> popup.innerHTML."""
        text = Path(_export(tmp_path)).read_text()
        titles = [str(node.get("title", "")) for node in _dataset(text, "nodes")]

        assert any("pwn" in t for t in titles), (
            "the heading never reached a tooltip, so this test proves nothing"
        )
        for title in titles:
            assert "<" not in _strip_own_tags(title), (
                f"repo text opened a tag inside a node tooltip: {title!r}"
            )

    def test_filename_cannot_inject_a_tag_into_the_tooltip(self, tmp_path: Path) -> None:
        """Carrier 2: filename -> rel_path -> the tooltip's File: line."""
        text = Path(_export(tmp_path)).read_text()
        titles = [str(node.get("title", "")) for node in _dataset(text, "nodes")]

        assert any("onerror" in t for t in titles), (
            "the filename never reached a tooltip, so this test proves nothing"
        )
        for title in titles:
            assert "<img" not in _strip_own_tags(title), (
                f"a filename opened an <img> tag inside a node tooltip: {title!r}"
            )

    def test_string_node_id_cannot_break_out_of_the_select_menu(self, tmp_path: Path) -> None:
        """Carrier 3: the second sink. No hover, no JSON encoding, no user interaction.

        `<option value="{{node.id}}">{{node.id}}</option>` is plain Jinja in a template
        whose Environment has no autoescape, so a `">` in a node id closes the attribute
        and then the element.
        """
        text = Path(_export(tmp_path)).read_text()
        options = [line for line in text.splitlines() if "<option value=" in line]

        assert options, "select_menu is off — the sink under test is not in the output"
        assert any("ticket:" in line for line in options), (
            "the artifact ref never reached the option list, so this test proves nothing"
        )
        for line in options:
            assert _OPTION_RE.fullmatch(line.strip()), (
                f"a node id broke out of the option list: {line.strip()!r}"
            )

    def test_no_attacker_authored_tag_reaches_any_dom_sink(self, tmp_path: Path) -> None:
        """Whole-document sweep, because a fix aimed at one sink is how SEC-04 got shipped.

        The pyvis 0.3.2 template authors no `<img` and no `<a ` of its own, so either one
        appearing in the output came from the repo. Swept over the raw page AND over every
        DECODED value that reaches the DOM — the tooltip pyvis assigns to `innerHTML`, the
        node ids it writes into the select menu, and the edge endpoints that follow them.

        `label` is excluded on purpose, and the exclusion is the point rather than a gap:
        vis-network draws labels with canvas text and never inserts them into the
        document, so the only escaping a label needs is the one that keeps it from ending
        the `<script>` element it is embedded in — which is Jinja's `tojson`, asserted
        below. HTML-escaping it instead would print `Vec&lt;T&gt;` and `don&#x27;t.md` on
        the graph for content that was never dangerous.
        """
        text = Path(_export(tmp_path)).read_text()
        nodes = _dataset(text, "nodes")
        edges = _dataset(text, "edges")

        dom_values = [str(node.get("title", "")) for node in nodes]
        dom_values += [str(node.get("id", "")) for node in nodes]
        dom_values += [str(edge.get(end, "")) for edge in edges for end in ("from", "to")]
        dom_values += [str(edge.get("title", "")) for edge in edges]

        for haystack in [text, *dom_values]:
            lowered = haystack.lower()
            assert "<img" not in lowered, f"an <img> tag reached a DOM sink: {haystack[:200]!r}"
            assert "<a href" not in lowered, f"an <a> tag reached a DOM sink: {haystack[:200]!r}"

        # The label is the one repo value that is deliberately not HTML-escaped, so pin
        # that the exclusion is genuinely exercised. What protects it is asserted in the
        # loop above: `<a href` appears nowhere in the RAW page, because Jinja's `tojson`
        # encodes the label's `<` as a \\u003c escape inside `<script>`. Stated this way
        # round so that HTML-escaping labels later — a hardening change — cannot read as a
        # regression here.
        assert any("href" in str(node.get("label", "")) for node in nodes), (
            "the payload never reached a node label, so the label boundary is untested"
        )

    def test_escaping_does_not_delete_the_graph(self, tmp_path: Path) -> None:
        """Gate, do not remove: every node still ships, and the ids still line up.

        An escape applied to node ids has to be applied to edge endpoints too, or the
        edges silently detach from their nodes and the visualization degrades into an
        unconnected dust cloud that still passes every escaping assertion above.
        """
        text = Path(_export(tmp_path)).read_text()
        nodes = _dataset(text, "nodes")
        edges = _dataset(text, "edges")
        ids = {json.dumps(node["id"]) for node in nodes}

        assert len(nodes) >= 2, f"expected the md section plus the artifact node, got {nodes}"
        assert edges, "the generic edges vanished from the export"
        for edge in edges:
            assert json.dumps(edge["from"]) in ids, f"edge endpoint is not a node: {edge}"
            assert json.dumps(edge["to"]) in ids, f"edge endpoint is not a node: {edge}"
