"""
Unit tests for Phase 6d config + markup parsers:
  JsonParser, YamlParser, TomlParser, MarkdownParser

These tests exercise the parsers in isolation — no DB, no embedder,
no file walker. All inputs are inline strings.
"""

from __future__ import annotations

from trelix.core.models import SymbolKind
from trelix.indexing.parser.extractors.json_config import JsonParser
from trelix.indexing.parser.extractors.markdown import MarkdownParser
from trelix.indexing.parser.extractors.toml_config import TomlParser
from trelix.indexing.parser.extractors.yaml_config import YamlParser

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _symbols_by_name(result, name: str):
    return [s for s in result.symbols if s.name == name]


def _qnames(result) -> list[str]:
    return [s.qualified_name for s in result.symbols]


def _kinds(result) -> list[str]:
    return [s.kind for s in result.symbols]


# ===========================================================================
# JSON / JSONC Tests
# ===========================================================================


class TestJsonParser:
    """JsonParser — object key extraction, JSONC comment handling."""

    def setup_method(self):
        self.parser = JsonParser()

    # ------------------------------------------------------------------
    # MODULE symbol
    # ------------------------------------------------------------------

    def test_module_symbol_emitted_for_object_root(self):
        src = '{"name": "my-app", "version": "1.0.0"}'
        result = self.parser.parse(src, file_id=1)
        modules = [s for s in result.symbols if s.kind == SymbolKind.MODULE]
        assert len(modules) == 1
        assert modules[0].name == "config"
        assert modules[0].qualified_name == "config"

    def test_module_symbol_signature_lists_top_keys(self):
        src = '{"name": "x", "version": "1", "scripts": {}}'
        result = self.parser.parse(src, file_id=1)
        mod = next(s for s in result.symbols if s.kind == SymbolKind.MODULE)
        # signature should contain the top-level key names
        assert "name" in mod.signature
        assert "version" in mod.signature
        assert "scripts" in mod.signature

    # ------------------------------------------------------------------
    # Scalar keys → CONSTANT
    # ------------------------------------------------------------------

    def test_scalar_string_key_is_constant(self):
        src = '{"name": "my-app"}'
        result = self.parser.parse(src, file_id=1)
        constants = [s for s in result.symbols if s.kind == SymbolKind.CONSTANT]
        names = [s.name for s in constants]
        assert "name" in names

    def test_scalar_bool_key_is_constant(self):
        src = '{"private": true}'
        result = self.parser.parse(src, file_id=1)
        const = next(s for s in result.symbols if s.name == "private")
        assert const.kind == SymbolKind.CONSTANT

    def test_scalar_number_key_is_constant(self):
        src = '{"port": 3000}'
        result = self.parser.parse(src, file_id=1)
        const = next(s for s in result.symbols if s.name == "port")
        assert const.kind == SymbolKind.CONSTANT

    def test_null_value_key_is_constant(self):
        src = '{"deprecated": null}'
        result = self.parser.parse(src, file_id=1)
        const = next(s for s in result.symbols if s.name == "deprecated")
        assert const.kind == SymbolKind.CONSTANT

    # ------------------------------------------------------------------
    # Object keys → SECTION
    # ------------------------------------------------------------------

    def test_object_value_key_is_section(self):
        src = '{"scripts": {"build": "tsc", "test": "jest"}}'
        result = self.parser.parse(src, file_id=1)
        sections = [s for s in result.symbols if s.kind == SymbolKind.SECTION]
        names = [s.name for s in sections]
        assert "scripts" in names

    def test_array_value_key_is_section(self):
        src = '{"include": ["src/**", "tests/**"]}'
        result = self.parser.parse(src, file_id=1)
        sections = [s for s in result.symbols if s.kind == SymbolKind.SECTION]
        names = [s.name for s in sections]
        assert "include" in names

    # ------------------------------------------------------------------
    # Nested keys
    # ------------------------------------------------------------------

    def test_nested_key_has_dotted_qualified_name(self):
        src = '{"compilerOptions": {"target": "ES2020"}}'
        result = self.parser.parse(src, file_id=1)
        qnames = _qnames(result)
        # nested scalar key should be "compilerOptions.target"
        assert "compilerOptions.target" in qnames

    def test_deeply_nested_key_capped_at_max_depth(self):
        # MAX_DEPTH=3: walk recurses at depth 0,1,2,3 which covers a, a.b, a.b.c, a.b.c.d
        # A 5-level deep key (a.b.c.d.e) should be excluded since recursion
        # stops at depth=MAX_DEPTH (recurse only when depth < MAX_DEPTH)
        src = '{"a": {"b": {"c": {"d": {"e": "too-deep"}}}}}'
        result = self.parser.parse(src, file_id=1)
        qnames = _qnames(result)
        # a.b.c.d.e is depth 5 — beyond MAX_DEPTH=3 recursion boundary
        assert "a.b.c.d.e" not in qnames
        # but a.b.c is still present at depth 2
        assert "a.b.c" in qnames

    # ------------------------------------------------------------------
    # Dependency ImportEdges
    # ------------------------------------------------------------------

    def test_dependencies_emit_import_edges(self):
        src = '{"dependencies": {"react": "^18.0.0", "typescript": "^5.0.0"}}'
        result = self.parser.parse(src, file_id=1)
        froms = [e.imported_from for e in result.import_edges]
        assert "react" in froms
        assert "typescript" in froms

    def test_dev_dependencies_emit_import_edges(self):
        src = '{"devDependencies": {"jest": "^29.0.0"}}'
        result = self.parser.parse(src, file_id=1)
        froms = [e.imported_from for e in result.import_edges]
        assert "jest" in froms

    # ------------------------------------------------------------------
    # JSONC comment handling
    # ------------------------------------------------------------------

    def test_jsonc_line_comment_becomes_docstring(self):
        src = '{\n  // Build target\n  "target": "ES2020"\n}'
        result = self.parser.parse(src, file_id=1)
        target_syms = _symbols_by_name(result, "target")
        assert target_syms, "Expected 'target' symbol"
        # The // comment should be captured as docstring
        assert target_syms[0].docstring == "Build target"

    def test_jsonc_block_comment_becomes_docstring(self):
        src = '{\n  /* Module resolution mode */\n  "moduleResolution": "node"\n}'
        result = self.parser.parse(src, file_id=1)
        sym = next((s for s in result.symbols if s.name == "moduleResolution"), None)
        assert sym is not None
        assert sym.docstring is not None
        assert "Module resolution mode" in sym.docstring

    # ------------------------------------------------------------------
    # Array root
    # ------------------------------------------------------------------

    def test_array_root_emits_module_symbol(self):
        src = '[{"name": "a"}, {"name": "b"}]'
        result = self.parser.parse(src, file_id=1)
        modules = [s for s in result.symbols if s.kind == SymbolKind.MODULE]
        assert len(modules) == 1
        assert "2 items" in modules[0].signature

    # ------------------------------------------------------------------
    # Line numbers
    # ------------------------------------------------------------------

    def test_line_numbers_are_one_indexed(self):
        src = '{\n  "key": "value"\n}'
        result = self.parser.parse(src, file_id=1)
        sym = next((s for s in result.symbols if s.name == "key"), None)
        assert sym is not None
        assert sym.line_start >= 1

    # ------------------------------------------------------------------
    # Empty / malformed
    # ------------------------------------------------------------------

    def test_empty_object_returns_no_symbols(self):
        result = self.parser.parse("{}", file_id=1)
        non_mod = [s for s in result.symbols if s.kind != SymbolKind.MODULE]
        assert len(non_mod) == 0

    def test_parse_errors_counted_on_malformed_json(self):
        result = self.parser.parse("{bad json}", file_id=1)
        assert result.parse_errors >= 1


# ===========================================================================
# YAML Tests
# ===========================================================================


class TestYamlParser:
    """YamlParser — single doc, multi-doc, kubernetes manifest shape."""

    def setup_method(self):
        self.parser = YamlParser()

    # ------------------------------------------------------------------
    # Single document
    # ------------------------------------------------------------------

    def test_single_doc_module_symbol(self):
        src = "name: my-service\nversion: 1.0.0\n"
        result = self.parser.parse(src, file_id=1)
        modules = [s for s in result.symbols if s.kind == SymbolKind.MODULE]
        assert len(modules) == 1
        assert modules[0].name == "config"
        assert modules[0].qualified_name == "config"

    def test_scalar_key_is_constant(self):
        src = "name: my-service\nport: 8080\n"
        result = self.parser.parse(src, file_id=1)
        const = next((s for s in result.symbols if s.name == "name"), None)
        assert const is not None
        assert const.kind == SymbolKind.CONSTANT

    def test_mapping_value_is_section(self):
        src = "server:\n  host: localhost\n  port: 8080\n"
        result = self.parser.parse(src, file_id=1)
        section = next((s for s in result.symbols if s.name == "server"), None)
        assert section is not None
        assert section.kind == SymbolKind.SECTION

    def test_sequence_value_is_section(self):
        src = "ports:\n  - 80\n  - 443\n"
        result = self.parser.parse(src, file_id=1)
        section = next((s for s in result.symbols if s.name == "ports"), None)
        assert section is not None
        assert section.kind == SymbolKind.SECTION
        assert "2 items" in section.signature

    def test_nested_key_has_dotted_qualified_name(self):
        src = "database:\n  host: localhost\n  port: 5432\n"
        result = self.parser.parse(src, file_id=1)
        qnames = _qnames(result)
        assert "database.host" in qnames
        assert "database.port" in qnames

    def test_single_doc_qualified_names_have_no_doc_prefix(self):
        """Single-doc files should NOT prefix paths with 'config.'"""
        src = "name: x\nversion: 1\n"
        result = self.parser.parse(src, file_id=1)
        # top-level scalar should be just "name", not "config.name"
        assert "name" in _qnames(result)
        assert "config.name" not in _qnames(result)

    # ------------------------------------------------------------------
    # Multi-document YAML
    # ------------------------------------------------------------------

    def test_multi_doc_emits_module_per_document(self):
        src = "name: first\n---\nname: second\n"
        result = self.parser.parse(src, file_id=1)
        modules = [s for s in result.symbols if s.kind == SymbolKind.MODULE]
        assert len(modules) == 2

    def test_multi_doc_prefixes_use_kind_or_name(self):
        src = "kind: Deployment\nname: web\n---\nkind: Service\nname: web-svc\n"
        result = self.parser.parse(src, file_id=1)
        modules = [s for s in result.symbols if s.kind == SymbolKind.MODULE]
        mod_names = {m.name for m in modules}
        # Should pick up "Deployment" and "Service" from kind field
        assert "Deployment" in mod_names or "Service" in mod_names

    # ------------------------------------------------------------------
    # Kubernetes manifest shape
    # ------------------------------------------------------------------

    def test_kubernetes_manifest_top_keys(self):
        src = textwrap_dedent("""
            apiVersion: apps/v1
            kind: Deployment
            metadata:
              name: my-deploy
              namespace: default
            spec:
              replicas: 3
        """).lstrip()
        result = self.parser.parse(src, file_id=1)
        qnames = _qnames(result)
        assert "apiVersion" in qnames
        assert "kind" in qnames
        assert "metadata" in qnames
        assert "spec" in qnames

    def test_kubernetes_metadata_nested_keys(self):
        src = "metadata:\n  name: my-deploy\n  namespace: default\n"
        result = self.parser.parse(src, file_id=1)
        qnames = _qnames(result)
        assert "metadata.name" in qnames
        assert "metadata.namespace" in qnames

    def test_kubernetes_sequence_of_mappings_steps(self):
        src = textwrap_dedent("""
            steps:
              - name: checkout
                uses: actions/checkout@v3
              - name: build
                run: npm run build
        """).lstrip()
        result = self.parser.parse(src, file_id=1)
        # Should have a "steps" SECTION
        assert "steps" in _qnames(result)

    # ------------------------------------------------------------------
    # YAML merge keys skipped
    # ------------------------------------------------------------------

    def test_merge_keys_are_skipped(self):
        src = "defaults: &defaults\n  timeout: 30\nserver:\n  <<: *defaults\n  port: 8080\n"
        result = self.parser.parse(src, file_id=1)
        # <<  (merge key) should not produce a symbol
        merge_syms = [s for s in result.symbols if s.name == "<<"]
        assert len(merge_syms) == 0

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    def test_invalid_yaml_returns_parse_error(self):
        result = self.parser.parse("{unclosed: [}", file_id=1)
        assert result.parse_errors >= 1
        assert result.symbols == []

    def test_empty_yaml_returns_no_symbols(self):
        result = self.parser.parse("", file_id=1)
        assert result.symbols == []

    # ------------------------------------------------------------------
    # Line numbers
    # ------------------------------------------------------------------

    def test_line_numbers_are_one_indexed(self):
        src = "name: x\nversion: 1\n"
        result = self.parser.parse(src, file_id=1)
        for sym in result.symbols:
            assert sym.line_start >= 1
            assert sym.line_end >= sym.line_start


# ===========================================================================
# TOML Tests
# ===========================================================================


class TestTomlParser:
    """TomlParser — dotted headers, array-of-tables."""

    def setup_method(self):
        self.parser = TomlParser()

    # ------------------------------------------------------------------
    # MODULE symbol
    # ------------------------------------------------------------------

    def test_module_symbol_lists_section_headers(self):
        src = '[package]\nname = "myapp"\n\n[dependencies]\nrequests = "2.28"\n'
        result = self.parser.parse(src, file_id=1)
        modules = [s for s in result.symbols if s.kind == SymbolKind.MODULE]
        assert len(modules) == 1
        sig = modules[0].signature
        assert "package" in sig
        assert "dependencies" in sig

    # ------------------------------------------------------------------
    # [section] headers → SECTION
    # ------------------------------------------------------------------

    def test_table_header_is_section(self):
        src = '[build-system]\nrequires = ["hatchling"]\n'
        result = self.parser.parse(src, file_id=1)
        sections = [s for s in result.symbols if s.kind == SymbolKind.SECTION]
        names = [s.name for s in sections]
        assert "build-system" in names

    def test_table_header_signature_uses_bracket_notation(self):
        src = '[package]\nname = "myapp"\n'
        result = self.parser.parse(src, file_id=1)
        sec = next((s for s in result.symbols if s.name == "package"), None)
        assert sec is not None
        assert "[package]" in sec.signature

    # ------------------------------------------------------------------
    # Dotted headers
    # ------------------------------------------------------------------

    def test_dotted_header_qualified_name_preserved(self):
        src = '[tool.pytest.ini_options]\ntestpaths = ["tests"]\n'
        result = self.parser.parse(src, file_id=1)
        sections = [s for s in result.symbols if s.kind == SymbolKind.SECTION]
        qnames = [s.qualified_name for s in sections]
        assert "tool.pytest.ini_options" in qnames

    def test_dotted_header_name_is_last_part(self):
        src = '[tool.ruff.lint]\nselect = ["E", "F"]\n'
        result = self.parser.parse(src, file_id=1)
        sec = next((s for s in result.symbols if s.qualified_name == "tool.ruff.lint"), None)
        assert sec is not None
        assert sec.name == "lint"

    # ------------------------------------------------------------------
    # [[array-of-tables]]
    # ------------------------------------------------------------------

    def test_array_of_tables_uses_double_bracket_signature(self):
        src = '[[bin]]\nname = "mycli"\npath = "src/main.rs"\n'
        result = self.parser.parse(src, file_id=1)
        sec = next(
            (
                s
                for s in result.symbols
                if s.kind == SymbolKind.SECTION and "bin" in s.qualified_name
            ),
            None,
        )
        assert sec is not None
        assert "[[bin]]" in sec.signature

    def test_array_of_tables_disambiguated_by_name(self):
        src = (
            '[[bin]]\nname = "tool1"\npath = "src/tool1.rs"\n\n'
            '[[bin]]\nname = "tool2"\npath = "src/tool2.rs"\n'
        )
        result = self.parser.parse(src, file_id=1)
        # Should get two bin sections, disambiguated by name
        bin_sections = [
            s for s in result.symbols if s.kind == SymbolKind.SECTION and "bin" in s.qualified_name
        ]
        assert len(bin_sections) >= 2
        qnames = [s.qualified_name for s in bin_sections]
        assert any("tool1" in qn for qn in qnames)
        assert any("tool2" in qn for qn in qnames)

    # ------------------------------------------------------------------
    # Key-value pairs inside sections
    # ------------------------------------------------------------------

    def test_scalar_pair_is_constant(self):
        src = '[package]\nname = "myapp"\nversion = "0.1.0"\n'
        result = self.parser.parse(src, file_id=1)
        const = next((s for s in result.symbols if s.name == "name"), None)
        assert const is not None
        assert const.kind == SymbolKind.CONSTANT

    def test_array_pair_is_section(self):
        src = '[build-system]\nrequires = ["hatchling", "hatch-vcs"]\n'
        result = self.parser.parse(src, file_id=1)
        sec = next((s for s in result.symbols if s.name == "requires"), None)
        assert sec is not None
        assert sec.kind == SymbolKind.SECTION

    def test_inline_table_pair_is_section(self):
        src = '[package]\nauthors = [{name = "Alice", email = "a@b.com"}]\n'
        # authors is an array (of inline tables); we just need it not to crash
        result = self.parser.parse(src, file_id=1)
        assert result.parse_errors == 0

    # ------------------------------------------------------------------
    # Dependency ImportEdges
    # ------------------------------------------------------------------

    def test_dependencies_table_emits_import_edges(self):
        src = '[dependencies]\nrequests = "2.28"\nnumpy = ">=1.26"\n'
        result = self.parser.parse(src, file_id=1)
        froms = [e.imported_from for e in result.import_edges]
        assert "requests" in froms
        assert "numpy" in froms

    def test_dev_dependencies_table_emits_import_edges(self):
        src = '[dev-dependencies]\npytest = ">=8.0"\n'
        result = self.parser.parse(src, file_id=1)
        froms = [e.imported_from for e in result.import_edges]
        assert "pytest" in froms

    # ------------------------------------------------------------------
    # Comment docstrings
    # ------------------------------------------------------------------

    def test_preceding_comment_becomes_docstring(self):
        src = '# The main package table\n[package]\nname = "x"\n'
        result = self.parser.parse(src, file_id=1)
        sec = next((s for s in result.symbols if s.name == "package"), None)
        assert sec is not None
        assert sec.docstring is not None
        assert "main package" in sec.docstring

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    def test_empty_toml_returns_no_symbols(self):
        result = self.parser.parse("", file_id=1)
        # empty file: no sections → no MODULE, no symbols
        assert len([s for s in result.symbols if s.kind != SymbolKind.MODULE]) == 0

    # ------------------------------------------------------------------
    # Line numbers
    # ------------------------------------------------------------------

    def test_line_numbers_one_indexed(self):
        src = '[package]\nname = "x"\n'
        result = self.parser.parse(src, file_id=1)
        for sym in result.symbols:
            assert sym.line_start >= 1


# ===========================================================================
# Markdown Tests
# ===========================================================================


class TestMarkdownParser:
    """MarkdownParser — heading hierarchy, breadcrumb qualified names."""

    def setup_method(self):
        self.parser = MarkdownParser()

    # ------------------------------------------------------------------
    # Basic heading extraction
    # ------------------------------------------------------------------

    def test_single_h1_becomes_section(self):
        src = "# Installation\n\nSome text here.\n"
        result = self.parser.parse(src, file_id=1)
        sections = [s for s in result.symbols if s.kind == SymbolKind.SECTION]
        assert len(sections) == 1
        assert sections[0].name == "Installation"
        assert sections[0].qualified_name == "Installation"

    def test_multiple_h1_headings(self):
        src = "# First\n\nText.\n\n# Second\n\nMore text.\n"
        result = self.parser.parse(src, file_id=1)
        names = [s.name for s in result.symbols if s.kind == SymbolKind.SECTION]
        assert "First" in names
        assert "Second" in names

    def test_h2_under_h1_has_breadcrumb_qualified_name(self):
        src = "# Installation\n\n## Quick Start\n\nInstructions.\n"
        result = self.parser.parse(src, file_id=1)
        h2 = next((s for s in result.symbols if s.name == "Quick Start"), None)
        assert h2 is not None
        assert h2.qualified_name == "Installation > Quick Start"

    def test_h3_under_h2_under_h1_three_level_breadcrumb(self):
        src = "# Installation\n\n## Quick Start\n\n### With Docker\n\nDocker steps.\n"
        result = self.parser.parse(src, file_id=1)
        h3 = next((s for s in result.symbols if s.name == "With Docker"), None)
        assert h3 is not None
        assert h3.qualified_name == "Installation > Quick Start > With Docker"

    # ------------------------------------------------------------------
    # Heading levels
    # ------------------------------------------------------------------

    def test_h1_signature_uses_single_hash(self):
        src = "# Title\n\nContent.\n"
        result = self.parser.parse(src, file_id=1)
        sym = next(s for s in result.symbols if s.name == "Title")
        assert sym.signature == "# Title"

    def test_h2_signature_uses_double_hash(self):
        src = "## Subtitle\n\nContent.\n"
        result = self.parser.parse(src, file_id=1)
        sym = next(s for s in result.symbols if s.name == "Subtitle")
        assert sym.signature == "## Subtitle"

    def test_h6_is_supported(self):
        src = "###### Deep Heading\n\nContent.\n"
        result = self.parser.parse(src, file_id=1)
        sym = next((s for s in result.symbols if s.name == "Deep Heading"), None)
        assert sym is not None

    # ------------------------------------------------------------------
    # Setext headings
    # ------------------------------------------------------------------

    def test_setext_h1_detected(self):
        src = "My Title\n========\n\nContent.\n"
        result = self.parser.parse(src, file_id=1)
        names = [s.name for s in result.symbols if s.kind == SymbolKind.SECTION]
        assert "My Title" in names

    def test_setext_h2_detected(self):
        src = "My Subtitle\n-----------\n\nContent.\n"
        result = self.parser.parse(src, file_id=1)
        names = [s.name for s in result.symbols if s.kind == SymbolKind.SECTION]
        assert "My Subtitle" in names

    # ------------------------------------------------------------------
    # Inline markdown stripped from heading text
    # ------------------------------------------------------------------

    def test_bold_stripped_from_heading(self):
        src = "# **Bold** Title\n\nContent.\n"
        result = self.parser.parse(src, file_id=1)
        names = [s.name for s in result.symbols if s.kind == SymbolKind.SECTION]
        assert "Bold Title" in names

    def test_code_stripped_from_heading(self):
        src = "# The `run()` method\n\nContent.\n"
        result = self.parser.parse(src, file_id=1)
        names = [s.name for s in result.symbols if s.kind == SymbolKind.SECTION]
        assert "The run() method" in names

    def test_link_stripped_from_heading(self):
        src = "# [Installation](https://example.com)\n\nContent.\n"
        result = self.parser.parse(src, file_id=1)
        names = [s.name for s in result.symbols if s.kind == SymbolKind.SECTION]
        assert "Installation" in names

    # ------------------------------------------------------------------
    # No-heading fallback
    # ------------------------------------------------------------------

    def test_prose_only_file_gets_document_symbol(self):
        src = "This is just plain prose.\n\nNo headings here.\n"
        result = self.parser.parse(src, file_id=1)
        assert len(result.symbols) == 1
        assert result.symbols[0].name == "Document"
        assert result.symbols[0].qualified_name == "Document"
        assert result.symbols[0].kind == SymbolKind.SECTION

    def test_empty_file_returns_no_symbols(self):
        result = self.parser.parse("", file_id=1)
        assert result.symbols == []

    # ------------------------------------------------------------------
    # Front matter
    # ------------------------------------------------------------------

    def test_yaml_front_matter_becomes_module(self):
        src = (
            "---\ntitle: My Guide\ndescription: A guide\n"
            "tags: [python, async]\n---\n\n# Introduction\n"
        )
        result = self.parser.parse(src, file_id=1)
        modules = [s for s in result.symbols if s.kind == SymbolKind.MODULE]
        assert len(modules) == 1
        assert modules[0].name == "My Guide"

    def test_yaml_front_matter_tags_in_signature(self):
        src = "---\ntitle: Tutorial\ntags: [python, async]\n---\n\nContent.\n"
        result = self.parser.parse(src, file_id=1)
        mod = next((s for s in result.symbols if s.kind == SymbolKind.MODULE), None)
        assert mod is not None
        assert "python" in mod.signature or "async" in mod.signature

    def test_toml_front_matter_becomes_module(self):
        src = '+++\ntitle = "My Article"\n+++\n\n# Overview\n'
        result = self.parser.parse(src, file_id=1)
        modules = [s for s in result.symbols if s.kind == SymbolKind.MODULE]
        assert len(modules) == 1
        assert modules[0].name == "My Article"

    def test_unclosed_front_matter_ignored(self):
        src = "---\ntitle: Bad FM\n\n# Heading\n\nContent.\n"
        result = self.parser.parse(src, file_id=1)
        # Should not crash; the heading should still be found
        sections = [s for s in result.symbols if s.kind == SymbolKind.SECTION]
        assert len(sections) >= 1

    # ------------------------------------------------------------------
    # Line numbers
    # ------------------------------------------------------------------

    def test_line_numbers_are_one_indexed(self):
        src = "# First\n\n## Sub\n\nContent.\n"
        result = self.parser.parse(src, file_id=1)
        for sym in result.symbols:
            assert sym.line_start >= 1
            assert sym.line_end >= sym.line_start

    def test_h1_starts_at_line_one(self):
        src = "# Title\n\nContent.\n"
        result = self.parser.parse(src, file_id=1)
        sym = next(s for s in result.symbols if s.name == "Title")
        assert sym.line_start == 1

    # ------------------------------------------------------------------
    # Section body boundaries
    # ------------------------------------------------------------------

    def test_section_body_does_not_include_next_h1_content(self):
        src = "# First\n\nFirst content.\n\n# Second\n\nSecond content.\n"
        result = self.parser.parse(src, file_id=1)
        first = next(s for s in result.symbols if s.name == "First")
        # body of First should not contain "Second content"
        assert "Second content" not in first.body

    def test_h2_section_ends_at_next_h2(self):
        src = "# Top\n\n## Alpha\n\nAlpha text.\n\n## Beta\n\nBeta text.\n"
        result = self.parser.parse(src, file_id=1)
        alpha = next(s for s in result.symbols if s.name == "Alpha")
        assert "Beta text" not in alpha.body

    # ------------------------------------------------------------------
    # parse_errors always 0 (pure regex, no tree-sitter)
    # ------------------------------------------------------------------

    def test_parse_errors_always_zero(self):
        for src in [
            "# Title\nContent",
            "Plain prose",
            "",
            "---\ntitle: x\n---\n# Head\n",
        ]:
            result = self.parser.parse(src, file_id=1)
            assert result.parse_errors == 0


# ---------------------------------------------------------------------------
# Helper imported at test time to avoid top-level import issues
# ---------------------------------------------------------------------------


def textwrap_dedent(s: str) -> str:
    import textwrap

    return textwrap.dedent(s)
