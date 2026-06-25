"""HTML parser — tree-sitter AST traversal for structural HTML analysis.

Extracts:
  - Page document (kind=MODULE) — <title> content + meta description
  - Custom / web-component elements (kind=VARIABLE) — any hyphenated tag like
    <app-root>, <router-outlet>, <my-button> (Angular, LitElement, etc.)
  - Angular template references (kind=VARIABLE) — #refName attributes
  - <form> elements (kind=SECTION) — with formGroup / action / id attributes
  - <script> blocks (kind=SECTION) — inline JS or external src
  - <style> blocks (kind=SECTION) — inline CSS
  - Structural semantic elements with id/aria-label (kind=SECTION) —
    <section>, <main>, <nav>, <header>, <footer>, <article>, <aside>
  - Block elements with id (kind=VARIABLE) — <div id="content">, <table id="…">

Angular-aware:
  Detects Angular structural directives (*ngIf, *ngFor, *ngSwitch),
  router bindings ([routerLink]), form bindings (formGroup, formControlName),
  and input/output bindings ([prop], (event)) — recorded in signature and
  decorators list on custom-element symbols.

Tree-sitter note:
  HTML grammar has NO child_by_field_name fields — every node is accessed
  via child type iteration (same pattern as _get_child_by_type in other parsers).
"""

from __future__ import annotations

import re

import tree_sitter_languages
from tree_sitter import Node, Parser

from trelix.core.models import ImportEdge, Symbol, SymbolKind
from trelix.indexing.parser.base import BaseParser, ParseResult

# Vue.js directive attribute prefixes / names
_VUE_STRUCTURAL = frozenset({"v-if", "v-else-if", "v-else", "v-for", "v-show", "v-html", "v-text"})
_VUE_FORM_ATTRS = frozenset({"v-model", "v-model.lazy", "v-model.trim", "v-model.number"})

# Semantic block tags worth extracting when they have id or aria-label
_SECTION_TAGS = frozenset({"section", "main", "nav", "header", "footer", "article", "aside"})

# Angular structural directive attribute names (lowercased for comparison)
_NG_STRUCTURAL = frozenset({"*ngif", "*ngfor", "*ngswitch", "*ngswichcase", "*ngswitchdefault"})

# Angular form attribute names (lowercased)
_NG_FORM_ATTRS = frozenset(
    {
        "formgroup",
        "[formgroup]",
        "formcontrolname",
        "[formcontrolname]",
        "formarrayname",
        "[formarrayname]",
        "ngmodel",
        "[(ngmodel)]",
    }
)

# Tags too small/inline to bother extracting by id
_SKIP_ID_TAGS = frozenset(
    {
        "span",
        "a",
        "button",
        "input",
        "label",
        "img",
        "br",
        "hr",
        "li",
        "td",
        "th",
        "tr",
        "i",
        "b",
        "em",
        "strong",
        "option",
        "textarea",
        "select",
        "link",
        "meta",
        "script",
        "style",
    }
)

MAX_DEPTH = 40  # guard against pathological nesting
MAX_CUSTOM_ELEMS = 150  # max unique custom element symbols per file


class HtmlParser(BaseParser):
    """Tree-sitter based HTML parser for structural and Angular template analysis."""

    def __init__(self) -> None:
        self._ts_language = tree_sitter_languages.get_language("html")
        self._parser = Parser()
        self._parser.set_language(self._ts_language)

    @property
    def language_name(self) -> str:
        return "html"

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def parse(self, source: str, file_id: int) -> ParseResult:
        source_bytes = source.encode("utf-8")
        tree = self._parser.parse(source_bytes)
        root = tree.root_node

        symbols: list[Symbol] = []
        import_edges: list[ImportEdge] = []
        # Deduplicate custom element symbols: one symbol per unique tag name per file
        seen_custom_tags: set[str] = set()

        # 1. Page-level document symbol (title + meta description)
        doc_sym = self._build_document_symbol(root, source_bytes, file_id)
        if doc_sym:
            symbols.append(doc_sym)

        # 2. Walk all elements
        self._walk_elements(
            root, source_bytes, file_id, symbols, import_edges, seen_custom_tags, depth=0
        )

        return ParseResult(
            symbols=symbols,
            call_edges=[],
            import_edges=import_edges,
            parse_errors=self._count_errors(root),
        )

    # ------------------------------------------------------------------
    # Document symbol
    # ------------------------------------------------------------------

    def _build_document_symbol(self, root: Node, src: bytes, file_id: int) -> Symbol | None:
        """Build a MODULE symbol for the whole page using <title> + meta tags."""
        meta: dict[str, str] = {}  # keys: title, description, keywords, og_title, og_description
        self._collect_page_meta(root, src, meta, depth=0)

        title = meta.get("title", "")
        description = meta.get("description", "")
        name = title or "document"
        sig = f"<title>{title}</title>" if title else "<html>"

        body_lines = []
        if title:
            body_lines.append(f"Title: {title}")
        if description:
            body_lines.append(f"Description: {description}")
        if meta.get("keywords"):
            body_lines.append(f"Keywords: {meta['keywords']}")
        if meta.get("og_title") and meta.get("og_title") != title:
            body_lines.append(f"OG Title: {meta['og_title']}")
        if meta.get("og_description") and meta.get("og_description") != description:
            body_lines.append(f"OG Description: {meta['og_description']}")

        return Symbol(
            file_id=file_id,
            name=name,
            qualified_name=name,
            kind=SymbolKind.MODULE,
            line_start=1,
            line_end=root.end_point[0] + 1,
            signature=sig,
            body="\n".join(body_lines)
            if body_lines
            else src.decode("utf-8", errors="replace")[:500],
            docstring=description or None,
            is_public=True,
        )

    def _collect_page_meta(
        self,
        node: Node,
        src: bytes,
        meta: dict[str, str],
        depth: int,
    ) -> None:
        """Recursively collect <title>, <meta name/property> tags from the document."""
        if depth > 10:
            return
        for child in node.children:
            if child.type not in ("element", "script_element", "style_element"):
                continue
            if child.type == "element":
                tag_node = self._get_start_or_self_closing_tag(child)
                if tag_node:
                    tag = self._get_tag_name(tag_node, src)
                    if tag == "title" and "title" not in meta:
                        for sub in child.children:
                            if sub.type == "text":
                                text = self._txt(sub, src).strip()
                                if text:
                                    meta["title"] = text
                                break
                    elif tag == "meta":
                        attrs = self._get_attributes(tag_node, src)
                        name_attr = attrs.get("name", "").lower()
                        prop_attr = attrs.get("property", "").lower()
                        content = attrs.get("content", "").strip()
                        if content:
                            if name_attr == "description" and "description" not in meta:
                                meta["description"] = content
                            elif name_attr == "keywords" and "keywords" not in meta:
                                meta["keywords"] = content
                            elif prop_attr == "og:title" and "og_title" not in meta:
                                meta["og_title"] = content
                            elif prop_attr == "og:description" and "og_description" not in meta:
                                meta["og_description"] = content
                # Recurse into <head> and <html> etc.
                self._collect_page_meta(child, src, meta, depth + 1)

    # ------------------------------------------------------------------
    # Element walker
    # ------------------------------------------------------------------

    def _walk_elements(
        self,
        node: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
        import_edges: list[ImportEdge],
        seen_custom_tags: set[str],
        depth: int,
    ) -> None:
        """Recursively walk all nodes, emitting symbols for interesting elements."""
        if depth > MAX_DEPTH:
            return

        for child in node.children:
            ntype = child.type

            if ntype == "element":
                tag_node = self._get_start_or_self_closing_tag(child)
                if tag_node:
                    tag = self._get_tag_name(tag_node, src)
                    attrs = self._get_attributes(tag_node, src)
                    self._handle_element(
                        child,
                        tag_node,
                        tag,
                        attrs,
                        src,
                        file_id,
                        symbols,
                        import_edges,
                        seen_custom_tags,
                        depth,
                    )
                self._walk_elements(
                    child, src, file_id, symbols, import_edges, seen_custom_tags, depth + 1
                )

            elif ntype == "script_element":
                self._handle_script(child, src, file_id, symbols, import_edges)

            elif ntype == "style_element":
                self._handle_style(child, src, file_id, symbols)

            else:
                self._walk_elements(
                    child, src, file_id, symbols, import_edges, seen_custom_tags, depth + 1
                )

    def _handle_element(
        self,
        node: Node,
        tag_node: Node,
        tag: str,
        attrs: dict[str, str],
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
        import_edges: list[ImportEdge],
        seen_custom_tags: set[str],
        depth: int,
    ) -> None:
        line_start = node.start_point[0] + 1
        line_end = node.end_point[0] + 1
        body_text = self._txt(node, src)[:1000]

        # ── 0. <link> — CSS/font/module imports ───────────────────────────
        if tag == "link":
            href = attrs.get("href", "")
            rel = attrs.get("rel", "").lower()
            if href and rel in ("stylesheet", "preload", "modulepreload", "import"):
                import_edges.append(
                    ImportEdge(
                        file_id=file_id,
                        imported_from=href,
                        imported_names=[],
                    )
                )
            return  # link is self-closing, nothing else to extract

        # ── 1. <template> and <ng-template> — Angular/Vue template blocks ──
        if tag in ("template", "ng-template"):
            ref_name = ""
            for attr_name, attr_val in attrs.items():
                if attr_name.startswith("#"):
                    ref_name = attr_name[1:] or attr_val
                    break
            name = ref_name or f"{tag}-{line_start}"
            symbols.append(
                Symbol(
                    file_id=file_id,
                    name=name,
                    qualified_name=name,
                    kind=SymbolKind.SECTION,
                    line_start=line_start,
                    line_end=line_end,
                    signature=f"<{tag}" + (f" #{ref_name}" if ref_name else "") + ">",
                    body=body_text,
                )
            )

        # ── 1b. Custom / web-component elements (hyphenated tag names) ─────
        elif "-" in tag and len(seen_custom_tags) < MAX_CUSTOM_ELEMS:
            if tag not in seen_custom_tags:
                seen_custom_tags.add(tag)
                sig, decorators = self._framework_element_signature(tag, attrs)
                symbols.append(
                    Symbol(
                        file_id=file_id,
                        name=tag,
                        qualified_name=tag,
                        kind=SymbolKind.VARIABLE,
                        line_start=line_start,
                        line_end=line_end,
                        signature=sig,
                        body=body_text,
                        decorators=decorators,
                        is_public=True,
                    )
                )

        # ── 2. Angular / Vue template references (#refName) ──────────────
        for attr_name, attr_val in attrs.items():
            if attr_name.startswith("#") and tag not in ("template", "ng-template"):
                ref_name = attr_name[1:] or attr_val
                if ref_name:
                    symbols.append(
                        Symbol(
                            file_id=file_id,
                            name=ref_name,
                            qualified_name=ref_name,
                            kind=SymbolKind.VARIABLE,
                            line_start=line_start,
                            line_end=line_end,
                            signature=f"#{ref_name} on <{tag}>",
                            body=self._txt(tag_node, src),
                        )
                    )

        # ── 3. <form> elements ────────────────────────────────────────────
        if tag == "form":
            form_group = (
                attrs.get("[formGroup]", "")
                or attrs.get("formGroup", "")
                or attrs.get("[formgroup]", "")
                or attrs.get("formgroup", "")
            )
            form_id = attrs.get("id", "")
            action = attrs.get("action", "")
            label = form_group or form_id or "form"
            sig = "<form"
            if form_group:
                sig += f' [formGroup]="{form_group}"'
            elif form_id:
                sig += f' id="{form_id}"'
            if action:
                sig += f' action="{action}"'
            sig += ">"
            symbols.append(
                Symbol(
                    file_id=file_id,
                    name=label,
                    qualified_name=label,
                    kind=SymbolKind.SECTION,
                    line_start=line_start,
                    line_end=line_end,
                    signature=sig,
                    body=body_text,
                    docstring=self._get_preceding_comment(node, src),
                )
            )

        # ── 4. Structural semantic elements with id / aria-label ──────────
        elif tag in _SECTION_TAGS:
            section_id = attrs.get("id", "")
            aria_label = attrs.get("aria-label", "")
            label = section_id or aria_label
            if label:
                sig = f'<{tag} id="{label}">' if section_id else f'<{tag} aria-label="{label}">'
                symbols.append(
                    Symbol(
                        file_id=file_id,
                        name=label,
                        qualified_name=label,
                        kind=SymbolKind.SECTION,
                        line_start=line_start,
                        line_end=line_end,
                        signature=sig,
                        body=body_text,
                        docstring=self._get_preceding_comment(node, src),
                    )
                )

        # ── 5. Other block elements with id attribute ─────────────────────
        elif tag not in _SKIP_ID_TAGS and tag not in ("html", "head", "body"):
            elem_id = attrs.get("id", "")
            if elem_id:
                symbols.append(
                    Symbol(
                        file_id=file_id,
                        name=elem_id,
                        qualified_name=elem_id,
                        kind=SymbolKind.VARIABLE,
                        line_start=line_start,
                        line_end=line_end,
                        signature=f'<{tag} id="{elem_id}">',
                        body=body_text,
                    )
                )

    # ------------------------------------------------------------------
    # Script / style
    # ------------------------------------------------------------------

    def _handle_script(
        self,
        node: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
        import_edges: list[ImportEdge],
    ) -> None:
        """Extract <script> blocks as SECTION symbols and ImportEdges for external scripts."""
        tag_node = self._get_start_or_self_closing_tag(node)
        attrs = self._get_attributes(tag_node, src) if tag_node else {}

        script_src = attrs.get("src", "")
        script_type = attrs.get("type", "")

        # External script → ImportEdge (JS/module dependency)
        if script_src:
            import_edges.append(
                ImportEdge(
                    file_id=file_id,
                    imported_from=script_src,
                    imported_names=[],
                )
            )

        # Find inline content
        raw = ""
        for child in node.children:
            if child.type == "raw_text":
                raw = self._txt(child, src).strip()
                break

        # Skip empty tags with no informative attributes
        if not raw and not script_src:
            return

        name = script_src.split("/")[-1].split("?")[0] if script_src else "inline-script"
        sig = "<script"
        if script_src:
            sig += f' src="{script_src}"'
        if script_type:
            sig += f' type="{script_type}"'
        sig += ">"

        symbols.append(
            Symbol(
                file_id=file_id,
                name=name,
                qualified_name=name,
                kind=SymbolKind.SECTION,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                signature=sig,
                body=raw[:1000] if raw else sig,
            )
        )

    def _handle_style(self, node: Node, src: bytes, file_id: int, symbols: list[Symbol]) -> None:
        """Extract non-empty <style> blocks as SECTION symbols."""
        raw = ""
        for child in node.children:
            if child.type == "raw_text":
                raw = self._txt(child, src).strip()
                break
        if not raw:
            return

        tag_node = self._get_start_or_self_closing_tag(node)
        attrs = self._get_attributes(tag_node, src) if tag_node else {}
        scoped = " scoped" if "scoped" in attrs else ""

        symbols.append(
            Symbol(
                file_id=file_id,
                name="inline-style",
                qualified_name="inline-style",
                kind=SymbolKind.SECTION,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                signature=f"<style{scoped}>",
                body=raw[:1000],
            )
        )

    # ------------------------------------------------------------------
    # Angular element signature builder
    # ------------------------------------------------------------------

    def _framework_element_signature(
        self, tag: str, attrs: dict[str, str]
    ) -> tuple[str, list[str]]:
        """Build a rich signature and decorators list for custom/Angular/Vue elements.

        Recognises Angular bindings ([prop], (event), *ngIf, routerLink) and
        Vue directives (v-if, v-for, v-model, :prop, @event). Returns at most
        4 key attributes in the signature to keep it readable.
        """
        key_attrs: list[str] = []
        decorators: list[str] = []

        for attr_name, attr_val in attrs.items():
            lower = attr_name.lower()

            # ── Angular structural directives (*ngIf, *ngFor, etc.) ──────
            if lower in _NG_STRUCTURAL:
                part = f'{attr_name}="{attr_val}"' if attr_val else attr_name
                key_attrs.append(part)
                decorators.append(part)

            elif lower in ("[routerlink]", "routerlink"):
                part = f'[routerLink]="{attr_val}"'
                key_attrs.append(part)
                decorators.append(part)

            elif lower in _NG_FORM_ATTRS:
                part = f'{attr_name}="{attr_val}"'
                key_attrs.append(part)
                decorators.append(part)

            elif lower.startswith("[("):
                # Angular two-way binding: [(ngModel)]
                key_attrs.append(f'{attr_name}="{attr_val}"')
                decorators.append(f'{attr_name}="{attr_val}"')

            elif lower.startswith("[") and not lower.startswith("[("):
                # Angular input property binding: [disabled], [class.active]
                key_attrs.append(f'{attr_name}="{attr_val}"')

            elif lower.startswith("("):
                # Angular event binding: (click), (submit)
                key_attrs.append(f'{attr_name}="{attr_val}"')

            # ── Vue.js directives ─────────────────────────────────────────
            elif lower in _VUE_STRUCTURAL:
                part = f'{attr_name}="{attr_val}"' if attr_val else attr_name
                key_attrs.append(part)
                decorators.append(part)

            elif lower in _VUE_FORM_ATTRS:
                part = f'{attr_name}="{attr_val}"'
                key_attrs.append(part)
                decorators.append(part)

            elif lower.startswith(":") and not lower.startswith("::"):
                # Vue shorthand prop binding: :disabled, :class
                key_attrs.append(f'{attr_name}="{attr_val}"')

            elif lower.startswith("@"):
                # Vue shorthand event: @click, @submit
                key_attrs.append(f'{attr_name}="{attr_val}"')

        sig = f"<{tag}"
        if key_attrs:
            sig += " " + " ".join(key_attrs[:4])  # cap to keep sig short
        sig += ">"

        return sig, decorators

    # ------------------------------------------------------------------
    # AST helpers
    # ------------------------------------------------------------------

    def _get_start_or_self_closing_tag(self, element_node: Node) -> Node | None:
        """Return the start_tag or self_closing_tag child of an element node."""
        for child in element_node.children:
            if child.type in ("start_tag", "self_closing_tag"):
                return child
        return None

    def _get_tag_name(self, tag_node: Node, src: bytes) -> str:
        """Return the lowercased tag name from a start_tag or self_closing_tag."""
        for child in tag_node.children:
            if child.type == "tag_name":
                return self._txt(child, src).lower()
        return ""

    def _get_attributes(self, tag_node: Node, src: bytes) -> dict[str, str]:
        """Return {attribute_name: attribute_value} for all attributes in a tag."""
        attrs: dict[str, str] = {}
        for child in tag_node.children:
            if child.type != "attribute":
                continue
            attr_name = ""
            attr_val = ""
            for sub in child.children:
                if sub.type == "attribute_name":
                    attr_name = self._txt(sub, src)
                elif sub.type == "quoted_attribute_value":
                    # quoted_attribute_value wraps attribute_value between quotes
                    val_node = self._get_child_by_type(sub, "attribute_value")
                    attr_val = (
                        self._txt(val_node, src)
                        if val_node
                        else re.sub(r'^["\']|["\']$', "", self._txt(sub, src))
                    )
                elif sub.type == "attribute_value":
                    attr_val = self._txt(sub, src)
            if attr_name:
                attrs[attr_name] = attr_val
        return attrs

    def _get_preceding_comment(self, node: Node, src: bytes) -> str | None:
        """Return the text of an HTML comment immediately before this node."""
        prev = node.prev_named_sibling
        if prev is not None and prev.type == "comment":
            raw = self._txt(prev, src).strip()
            # Strip <!-- ... --> delimiters
            raw = re.sub(r"^<!--\s*", "", raw)
            raw = re.sub(r"\s*-->$", "", raw)
            return raw.strip() or None
        return None

    def _count_errors(self, node: Node) -> int:
        count = 1 if node.type == "ERROR" else 0
        for child in node.children:
            count += self._count_errors(child)
        return count

    def _txt(self, node: Node, src: bytes) -> str:
        return src[node.start_byte : node.end_byte].decode("utf-8", errors="replace")

    def _get_child_by_type(self, node: Node, type_name: str) -> Node | None:
        for child in node.children:
            if child.type == type_name:
                return child
        return None
