"""YAML config file parser using PyYAML's compose_all() API.

Extracts configuration keys and their values as indexed symbols:
  - File-level MODULE symbol summarising top-level key names per document
  - Mapping keys with mapping/sequence values → SymbolKind.SECTION
  - Mapping keys with scalar values → SymbolKind.CONSTANT
  - Nested keys up to MAX_DEPTH → SECTION or CONSTANT
  - Sequence-of-mappings items disambiguated by name/id/type field or index
  - Multi-document YAML (--- separators) — all documents indexed
  - YAML merge keys (<<: *anchor) skipped to avoid noise symbols

Covers:
  - GitHub Actions workflows (.github/workflows/*.yml)
  - Docker Compose (docker-compose.yml)
  - Kubernetes manifests (multi-doc, kind/apiVersion fields)
  - Helm values.yaml, kustomization.yaml
  - Any other .yaml/.yml config file

Uses yaml.compose_all() (PyYAML, already a dependency) which provides a
node tree with exact start_mark / end_mark per key and value — giving us
accurate line_start / line_end without text-search approximation.
"""

from __future__ import annotations

from typing import Optional

import yaml

from trelix.core.models import Symbol, SymbolKind
from trelix.indexing.parser.base import BaseParser, ParseResult

# Scalar field names used to disambiguate sequence items (checked in order)
_ITEM_NAME_KEYS = ("name", "id", "type", "key", "title", "step")

# YAML merge key tag — skip <<: *anchor entries to avoid noise
_MERGE_TAG = "tag:yaml.org,2002:merge"


class YamlParser(BaseParser):
    """
    Parser for YAML configuration files.

    Extracts key-path symbols so config values become searchable.
    Caps total symbols at 80 per document to avoid flooding from large files.
    """

    MAX_SYMBOLS = 80
    MAX_DEPTH = 3
    MAX_BODY_LEN = 800

    @property
    def language_name(self) -> str:
        return "yaml"

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def parse(self, source: str, file_id: int) -> ParseResult:
        try:
            docs = list(yaml.compose_all(source))
        except yaml.YAMLError:
            return ParseResult(symbols=[], call_edges=[], import_edges=[],
                               parse_errors=1, type_edges=[])

        if not docs:
            return ParseResult(symbols=[], call_edges=[], import_edges=[],
                               parse_errors=0, type_edges=[])

        lines = source.splitlines()
        symbols: list[Symbol] = []

        for doc_idx, root in enumerate(docs):
            if not isinstance(root, yaml.MappingNode):
                continue

            doc_prefix = self._doc_prefix(root, doc_idx, len(docs))
            self._emit_module_symbol(root, doc_prefix, file_id, symbols, lines)
            self._walk_mapping(root, doc_prefix, file_id, symbols, lines, depth=0)

        return ParseResult(symbols=symbols, call_edges=[], import_edges=[],
                           parse_errors=0, type_edges=[])

    # ------------------------------------------------------------------
    # Module symbol (file/document summary)
    # ------------------------------------------------------------------

    def _doc_prefix(self, root: yaml.MappingNode, doc_idx: int, total_docs: int) -> str:
        """Return the top-level prefix for this document.

        Single-doc files use '' so key paths stay clean (e.g. 'metadata.name').
        Multi-doc files prefix with the kind/name of the doc or 'doc{n}'.
        """
        if total_docs == 1:
            return ""
        # Try kind, then name, then doc index
        for field in ("kind", "name"):
            val = self._get_scalar_field(root, field)
            if val:
                return val
        return f"doc{doc_idx + 1}"

    def _emit_module_symbol(
        self,
        root: yaml.MappingNode,
        prefix: str,
        file_id: int,
        symbols: list[Symbol],
        lines: list[str],
    ) -> None:
        """Emit one MODULE symbol summarising the top-level keys of this document."""
        top_keys = [
            k.value
            for k, _ in root.value
            if isinstance(k, yaml.ScalarNode) and k.tag != _MERGE_TAG
        ]
        if not top_keys:
            return
        line_start = root.start_mark.line + 1
        line_end = root.end_mark.line + 1
        body = self._slice_source(lines, root.start_mark.line, root.end_mark.line)
        name = prefix or "config"
        sig_keys = " ".join(f"[{k}]" for k in top_keys[:12])
        symbols.append(Symbol(
            file_id=file_id,
            name=name,
            qualified_name=name,
            kind=SymbolKind.MODULE,
            line_start=line_start,
            line_end=line_end,
            signature=sig_keys,
            body=body,
            is_public=True,
        ))

    # ------------------------------------------------------------------
    # Recursive mapping walker
    # ------------------------------------------------------------------

    def _walk_mapping(
        self,
        node: yaml.MappingNode,
        prefix: str,
        file_id: int,
        symbols: list[Symbol],
        lines: list[str],
        depth: int,
    ) -> None:
        for key_node, val_node in node.value:
            if len(symbols) >= self.MAX_SYMBOLS:
                return

            # Key must be a scalar (string key)
            if not isinstance(key_node, yaml.ScalarNode):
                continue

            # Skip YAML merge keys (<<: *anchor) — they inject noise paths
            if key_node.tag == _MERGE_TAG:
                continue

            key = key_node.value
            path = f"{prefix}.{key}" if prefix else key
            line_start = key_node.start_mark.line + 1
            line_end = val_node.end_mark.line + 1
            body = self._slice_source(lines, key_node.start_mark.line, val_node.end_mark.line)

            if isinstance(val_node, yaml.MappingNode):
                symbols.append(Symbol(
                    file_id=file_id,
                    name=key,
                    qualified_name=path,
                    kind=SymbolKind.SECTION,
                    line_start=line_start,
                    line_end=line_end,
                    signature=f"{path}:",
                    body=body,
                    is_public=True,
                ))
                if depth < self.MAX_DEPTH:
                    self._walk_mapping(val_node, path, file_id, symbols, lines, depth + 1)

            elif isinstance(val_node, yaml.SequenceNode):
                item_count = len(val_node.value)
                symbols.append(Symbol(
                    file_id=file_id,
                    name=key,
                    qualified_name=path,
                    kind=SymbolKind.SECTION,
                    line_start=line_start,
                    line_end=line_end,
                    signature=f"{path}: [{item_count} items]",
                    body=body,
                    is_public=True,
                ))
                # Recurse into sequence-of-mappings (e.g. GitHub Actions steps)
                if depth < self.MAX_DEPTH:
                    for item_idx, item in enumerate(val_node.value):
                        if not isinstance(item, yaml.MappingNode):
                            continue
                        if len(symbols) >= self.MAX_SYMBOLS:
                            break
                        # Disambiguate path by name/id/type field, else by index
                        item_key = self._get_scalar_field(item, *_ITEM_NAME_KEYS)
                        item_path = f"{path}[{item_key or item_idx}]"
                        self._walk_mapping(item, item_path, file_id, symbols, lines, depth + 1)

            else:
                # Scalar value (string, int, bool, null, alias anchor name, etc.)
                val_text = val_node.value if val_node.value is not None else ""
                body = f"{path}: {val_text}"
                symbols.append(Symbol(
                    file_id=file_id,
                    name=key,
                    qualified_name=path,
                    kind=SymbolKind.CONSTANT,
                    line_start=line_start,
                    line_end=line_end,
                    signature=body[:200],
                    body=body,
                    is_public=True,
                ))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_scalar_field(self, node: yaml.MappingNode, *keys: str) -> Optional[str]:
        """Return the value of the first matching scalar key in a mapping node."""
        for k, v in node.value:
            if (
                isinstance(k, yaml.ScalarNode)
                and k.value in keys
                and isinstance(v, yaml.ScalarNode)
                and v.value
            ):
                return v.value
        return None

    def _slice_source(self, lines: list[str], start_line: int, end_line: int) -> str:
        """Extract source lines [start_line, end_line] (0-indexed), capped at MAX_BODY_LEN."""
        sliced = "\n".join(lines[start_line: end_line + 1])
        if len(sliced) > self.MAX_BODY_LEN:
            sliced = sliced[: self.MAX_BODY_LEN] + "\n  ..."
        return sliced
