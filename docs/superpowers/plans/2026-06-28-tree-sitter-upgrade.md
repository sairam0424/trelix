# Phase 4: tree-sitter Deprecated API Upgrade — Implementation Plan

**Date:** 2026-06-28
**Spec:** `docs/superpowers/specs/2026-06-28-tree-sitter-upgrade-design.md`
**Branch:** `feat/phase4-tree-sitter-warning-suppress`
**Estimated wall-clock time:** ~30 min

---

## Context

Every `tree_sitter_languages.get_language()` call fires:
```
FutureWarning: Language(path, name) is deprecated. Use Language(ptr, name) instead.
```

This produces 439 warnings per test run (699 tests). The chosen fix (Option D from
the spec) is a two-part suppression:

1. A new `_grammar.py` helper that wraps every `get_language()` call in
   `warnings.catch_warnings()`.
2. A `filterwarnings` entry in `pyproject.toml` to silence residual warnings in
   test output.

No grammar-loading logic changes. No new runtime dependencies. Parse output is identical.

---

## Pre-flight

```bash
cd /Users/sairamugge/Desktop/Not-Humans-World/trelix
git checkout develop
git pull
git checkout -b feat/phase4-tree-sitter-warning-suppress

# Verify baseline: warnings exist, all tests pass
python -m pytest tests/ -x -q 2>&1 | tail -5
python -m pytest --no-header -q 2>&1 | grep -c FutureWarning   # expect >0
```

---

## Task 1 — Create `_grammar.py` + add pytest filterwarnings

**Goal:** Introduce the helper module and silence warnings in the test runner.
No extractor is changed yet; existing tests must still pass.

### 1a. Create `src/trelix/indexing/parser/_grammar.py`

Create the file with exactly this content:

```python
"""
Grammar loading helpers for tree-sitter 0.21.x.

tree_sitter_languages.get_language() calls Language(path, name) internally,
which fires a FutureWarning on tree-sitter 0.21.3. We suppress it at the call
site so neither library callers nor the test suite see the noise.

When upgrading to tree-sitter 0.22 this is the single place to update:
  1. Delete this file.
  2. Add per-language grammar packages (tree-sitter-python, etc.).
  3. Update each extractor to: Language(tree_sitter_<lang>.language()).
  4. Remove the filterwarnings entry from pyproject.toml.
"""
from __future__ import annotations

import warnings

from tree_sitter import Language, Parser


def load_language(name: str) -> Language:
    """Return a tree-sitter Language for *name*, suppressing 0.21.x FutureWarning."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            module="tree_sitter",
        )
        from tree_sitter_languages import get_language

        return get_language(name)  # type: ignore[no-any-return]


def make_parser(name: str) -> Parser:
    """Return a Parser pre-loaded with the named grammar."""
    lang = load_language(name)
    parser = Parser()
    parser.set_language(lang)
    return parser
```

### 1b. Add `filterwarnings` to `pyproject.toml`

Find the `[tool.pytest.ini_options]` block. It currently reads:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
addopts = "--tb=short"
pythonpath = ["src", "."]
```

Add the `filterwarnings` line so it becomes:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
addopts = "--tb=short"
pythonpath = ["src", "."]
filterwarnings = [
    "ignore::FutureWarning:tree_sitter",
]
```

### 1c. Verify

```bash
# Module imports cleanly
python -c "from trelix.indexing.parser._grammar import load_language, make_parser; print('ok')"

# All tests still pass (extractors unchanged; this is a no-regression gate)
python -m pytest tests/ -x -q 2>&1 | tail -5

# FutureWarnings already gone from test output (pytest filter active)
python -m pytest --no-header -q 2>&1 | grep -c FutureWarning   # expect 0
```

### 1d. Commit

```
git add src/trelix/indexing/parser/_grammar.py pyproject.toml
git commit -m "feat(parser): add _grammar.py load_language/make_parser helpers

Introduces a central grammar-loading module that wraps every
tree_sitter_languages.get_language() call in warnings.catch_warnings() to
suppress the FutureWarning fired by tree-sitter 0.21.3.

Also adds filterwarnings to [tool.pytest.ini_options] so the 439
test-run warnings are eliminated from pytest output.

Extractors are unchanged in this commit; no behaviour change."
```

---

## Task 2 — Migrate all extractor files

**Goal:** Replace every raw `tree_sitter_languages.get_language()` call with
`load_language()` / `make_parser()` from `_grammar.py`. Apply the pattern
mechanically across all 15 affected files.

### The change pattern (apply once to each file below)

**Standard extractors** (12 files — `__init__` sets `_ts_language` + `_parser`):

Before:
```python
import tree_sitter_languages
from tree_sitter import Node, Parser

class XxxParser(BaseParser):
    def __init__(self) -> None:
        self._ts_language = tree_sitter_languages.get_language("<lang>")
        self._parser = Parser()
        self._parser.set_language(self._ts_language)
```

After:
```python
from tree_sitter import Node, Parser

from .._grammar import load_language, make_parser

class XxxParser(BaseParser):
    def __init__(self) -> None:
        self._ts_language = load_language("<lang>")
        self._parser = make_parser("<lang>")
```

**Note:** Some extractors use `_ts_lang` instead of `_ts_language` as the attribute
name — preserve whichever name the file already uses.

**Lazy-loaded extractors** (`json_config.py` and `toml_config.py`):

The `get_language()` call lives inside a method body, not `__init__`. Pattern:

Before:
```python
    def _parse_json_treesitter(self, source: str, file_id: int) -> ParseResult:
        try:
            import tree_sitter_languages
            from tree_sitter import Parser as TSParser

            lang = tree_sitter_languages.get_language("json")
            parser = TSParser()
            parser.set_language(lang)
        except Exception:
            ...
```

After:
```python
    def _parse_json_treesitter(self, source: str, file_id: int) -> ParseResult:
        try:
            from trelix.indexing.parser._grammar import load_language, make_parser

            lang = load_language("json")
            parser = make_parser("json")
        except Exception:
            ...
```

Same pattern for `toml_config.py` (method `parse`, language `"toml"`).

**`python.py` — special case** (remove the two helper functions):

`python.py` has two module-level helper functions `_get_python_language()` and
`_make_parser()` that implement 0.21/0.22 compatibility shims. The `try/except
ImportError` fallback is dead code given `tree-sitter-languages>=1.10.2`.
Replace both with direct use of `_grammar.py`.

Before (lines 40-80 of `python.py`):
```python
from tree_sitter import Language, Node, Parser

...

def _get_python_language() -> Language:
    try:
        import tree_sitter_languages  # noqa: F401
        return tree_sitter_languages.get_language("python")  # type: ignore[no-any-return]
    except ImportError:
        import tree_sitter_python  # noqa: F401
        return Language(tree_sitter_python.language())  # type: ignore[call-arg]


def _make_parser(language: Language) -> Parser:
    try:
        p = Parser()
        p.set_language(language)
        return p
    except (TypeError, AttributeError):
        return Parser(language)  # type: ignore[call-arg]
```

After:
```python
from tree_sitter import Node, Parser

from .._grammar import load_language, make_parser
```

Then update `PythonParser.__init__` to call:
```python
self._ts_lang = load_language("python")
self._parser = make_parser("python")
```

(Find the exact attribute names by reading the `__init__` body before editing.
The helpers `_get_python_language()` and `_make_parser()` are deleted entirely.)

### File-by-file checklist

Apply the standard pattern to each file. The grammar name string for each:

| File | Grammar string | Attribute name |
|------|---------------|----------------|
| `javascript.py` | `"javascript"` | `_ts_language` |
| `typescript.py` | variable `lang_name` (already computed) | `_ts_language` |
| `go.py` | `"go"` | `_ts_language` |
| `rust.py` | `"rust"` | `_ts_language` |
| `java.py` | `"java"` | `_ts_language` |
| `kotlin.py` | `"kotlin"` | `_ts_lang` |
| `cpp.py` | `"cpp"` | `_ts_lang` |
| `c.py` | `"c"` | `_ts_lang` |
| `csharp.py` | `"c_sharp"` | `_ts_lang` |
| `ruby.py` | `self._GRAMMAR` (class constant) | `_ts_lang` |
| `html.py` | `"html"` | `_ts_language` |
| `css.py` | `"css"` | `_ts_language` |
| `json_config.py` | `"json"` | lazy (in method) |
| `toml_config.py` | `"toml"` | lazy (in method) |
| `python.py` | `"python"` | `_ts_lang` (check) |

**`typescript.py` note:** The extractor computes `lang_name` before calling
`get_language()`. Preserve that logic; just replace the call:
```python
self._ts_language = load_language(lang_name)
self._parser = make_parser(lang_name)
```

**`ruby.py` note:** Uses `self._GRAMMAR` as the language name constant. Preserve:
```python
self._ts_lang = load_language(self._GRAMMAR)
self._parser = make_parser(self._GRAMMAR)
```

**Not changed:** `markdown.py`, `razor.py`, `cshtml.py`, `csproj.py`,
`yaml_config.py` — these contain no tree-sitter grammar loading.

### 2c. Verify after all 15 files are updated

```bash
# No raw get_language() or old Language() wrapper calls remain in extractors
grep -rn "tree_sitter_languages.get_language\|Language(tree_sitter" \
    src/trelix/indexing/parser/extractors/ | grep -v "__pycache__"
# Expected: no output

# Spot-check: PythonParser works
python -c "
from trelix.indexing.parser.extractors.python import PythonParser
p = PythonParser()
result = p.parse('def foo(): pass', 'test.py')
print('symbols:', len(result.symbols))
"

# Full parser test suite
python -m pytest tests/ -x -q 2>&1 | tail -5
# Expected: 699 passed
```

### 2d. Commit

```
git add src/trelix/indexing/parser/extractors/
git commit -m "refactor(parser): migrate all extractors to load_language/make_parser

Replaces every bare tree_sitter_languages.get_language() call across 15
extractor files with load_language() / make_parser() from _grammar.py.

Removes the _get_python_language() / _make_parser() compat shims from
python.py — they are dead code given tree-sitter-languages>=1.10.2.

No grammar-loading logic changed; parse output is identical."
```

---

## Task 3 — Full validation

**Goal:** Confirm zero FutureWarnings in test output, all 699 tests pass,
and the grep gates from the spec are clean.

### 3a. Run the four verification criteria from the spec

```bash
cd /Users/sairamugge/Desktop/Not-Humans-World/trelix

# Criterion 1: FutureWarning count is zero
python -m pytest --no-header -q 2>&1 | grep -c FutureWarning
# Must print: 0

# Criterion 2: Full suite passes
python -m pytest tests/ -x -q
# Must exit 0, 699 passed

# Criterion 3: Direct instantiation emits no warning
python -W error::FutureWarning -c "
from trelix.indexing.parser.extractors.python import PythonParser
p = PythonParser()
result = p.parse('def foo(): pass', 'test.py')
print('ok:', len(result.symbols), 'symbols')
"
# Must print: ok: 1 symbols  (no FutureWarning raised as error)

# Criterion 4: No legacy call sites remain
grep -rn "tree_sitter_languages.get_language\|Language(tree_sitter" \
    src/trelix/indexing/parser/extractors/ | grep -v "__pycache__"
# Must produce no output
```

### 3b. Spot-check three more extractors

```bash
python -W error::FutureWarning -c "
from trelix.indexing.parser.extractors.javascript import JavaScriptParser
from trelix.indexing.parser.extractors.typescript import TypeScriptParser
from trelix.indexing.parser.extractors.rust import RustParser
for cls in [JavaScriptParser, TypeScriptParser, RustParser]:
    p = cls()
    print(cls.__name__, 'ok')
"
```

### 3c. Open PR

```bash
git push -u origin feat/phase4-tree-sitter-warning-suppress
gh pr create \
  --base develop \
  --title "feat(parser): suppress tree-sitter 0.21.x FutureWarning (Phase 4)" \
  --body "## Summary
- Adds \`src/trelix/indexing/parser/_grammar.py\` with \`load_language()\` / \`make_parser()\` helpers that wrap \`get_language()\` in \`warnings.catch_warnings()\`.
- Migrates all 15 affected extractors to use the new helpers.
- Removes dead \`_get_python_language()\` / \`_make_parser()\` shims from \`python.py\`.
- Adds \`filterwarnings\` to \`[tool.pytest.ini_options]\` — eliminates 439 warnings from test output.

## Test plan
- [ ] \`pytest --no-header -q 2>&1 | grep -c FutureWarning\` returns \`0\`
- [ ] \`pytest tests/ -x -q\` exits 0 with 699 tests passing
- [ ] \`python -W error::FutureWarning -c 'from trelix.indexing.parser.extractors.python import PythonParser; PythonParser()'\` exits 0
- [ ] grep for \`tree_sitter_languages.get_language\` in extractors returns no matches"
```

---

## Rollback

If any test regresses, `git revert HEAD~2..HEAD` and open a bug report with the
exact failure. The `_grammar.py` helper is strictly additive; the only breaking
change possible is a wrong grammar name string in an extractor, which tests catch
immediately.

---

## Forward path (tree-sitter 0.22)

When upgrading to 0.22:
1. Delete `src/trelix/indexing/parser/_grammar.py`.
2. Add per-language grammar packages to `pyproject.toml` (e.g. `tree-sitter-python`).
3. Update each extractor to: `Language(tree_sitter_<lang>.language())` directly.
4. Remove the `filterwarnings` entry from `pyproject.toml`.

The 0.22 migration is entirely contained in `_grammar.py` + 15 extractor files.
Phase 4 makes that future migration no harder than it already is.
