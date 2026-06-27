# Phase 4: tree-sitter Deprecated API Upgrade — Design Spec

**Date:** 2026-06-28
**Status:** Ready for implementation
**Author:** Design spec generated from live codebase audit

---

## 1. Problem Statement

Every call to `tree_sitter_languages.get_language(name)` fires a `FutureWarning`:

```
FutureWarning: Language(path, name) is deprecated. Use Language(ptr, name) instead.
```

This fires **once per `get_language()` call** — no in-process deduplication. In the test
suite (699 tests, parsers constructed per test in many fixtures) this produces **439
warnings per test run**. In a future tree-sitter release the warning becomes an error
and trelix stops parsing entirely.

Root cause (confirmed by reading `tree_sitter==0.21.3` source):

```python
# tree_sitter/__init__.py:123-138
def __init__(self, path_or_ptr: Union[PathLike, str, int], name: str):
    if isinstance(path_or_ptr, (str, PathLike)):
        _deprecate("Language(path, name)", "Language(ptr, name)")   # <-- fires here
        self.lib = cdll.LoadLibrary(fspath(path_or_ptr))
        ...
    elif isinstance(path_or_ptr, int):
        self.language_id = path_or_ptr   # ptr path — no warning
```

`tree_sitter_languages.get_language()` calls `Language(path, name)` internally
(the `.so` bundle path), so it always hits the deprecated branch on this version.
There is **no caching** inside `get_language()` that would deduplicate.

---

## 2. Current State

### 2.1 Installed versions

| Package | Version |
|---|---|
| `tree-sitter` | 0.21.3 |
| `tree-sitter-languages` | 1.10.2 |

Constraint in `pyproject.toml`:
```
"tree-sitter>=0.21,<0.22",
"tree-sitter-languages>=1.10.2",
```

### 2.2 Two distinct grammar-loading patterns in use

**Pattern A — `tree_sitter_languages.get_language()` + manual `Parser()`**

Used by 14 of 15 active extractors. The warning fires on every
`get_language()` call.

```python
# extractors/javascript.py (representative)
import tree_sitter_languages
from tree_sitter import Node, Parser

class JavaScriptParser(BaseParser):
    def __init__(self) -> None:
        self._ts_language = tree_sitter_languages.get_language("javascript")  # warning here
        self._parser = Parser()
        self._parser.set_language(self._ts_language)
```

Extractors using this pattern:
- `javascript.py` — `"javascript"`
- `typescript.py` — `"typescript"` / `"tsx"`
- `go.py` — `"go"`
- `rust.py` — `"rust"`
- `java.py` — `"java"`
- `kotlin.py` — `"kotlin"`
- `cpp.py` — `"cpp"`
- `c.py` — `"c"`
- `csharp.py` — `"c_sharp"`
- `ruby.py` — `"ruby"`
- `html.py` — `"html"`
- `css.py` — `"css"`
- `json_config.py` — `"json"` (lazy-loaded inside method)
- `toml_config.py` — `"toml"` (lazy-loaded inside method)

**Pattern B — try/except fallback with redundant `Language()` wrap (python.py only)**

```python
# extractors/python.py
def _get_python_language() -> Language:
    try:
        import tree_sitter_languages
        return tree_sitter_languages.get_language("python")   # warning here (path A)
    except ImportError:
        import tree_sitter_python
        return Language(tree_sitter_python.language())        # warning here (path B)
```

Path A fires the warning. Path B additionally wraps the ptr in a second `Language(ptr, name)`
call — which is the _new_ API form shown in the deprecation message, so it does not warn.
But path A (the one that actually runs) does warn.

### 2.3 Existing warning suppression

`src/trelix/cli/main.py` suppresses the warning for CLI users:

```python
warnings.filterwarnings("ignore", category=FutureWarning, module="tree_sitter")
```

This silences it in production but **does nothing in tests** (conftest.py has no
equivalent filter, and `pyproject.toml [tool.pytest.ini_options]` has no
`filterwarnings` entry). The 439 warnings appear in every test run.

---

## 3. Options Analysis

### Option A — Pin tree-sitter to a non-warning version

Keep `"tree-sitter>=0.21,<0.21.3"` or `==0.21.2`. The FutureWarning was introduced
in 0.21.3.

- No code changes required.
- Closes off the upgrade path permanently. The moment a transitive dependency needs
  0.21.3+ or >=0.22, the pin breaks.
- Does not fix the underlying API use.
- **Rejected.** Pinning to a known-deprecated version is not a migration; it is
  avoidance.

### Option B — Migrate to the new Language(ptr, name) API

tree-sitter 0.22 removed `Language(path, name)` entirely and requires grammar packages
(`tree-sitter-python`, `tree-sitter-javascript`, etc.) to be installed individually.
That means replacing `tree-sitter-languages` (which bundles 150+ grammars into one `.so`)
with ~15 separate grammar packages — a significant dependency change and a known source
of ABI compatibility issues between the grammar packages and `tree-sitter` core.

- Requires adding 14+ new `tree-sitter-*` package dependencies.
- Grammar packages have their own release cadence; some lag behind core.
- `python.py` already has the fallback code for this path (the `except ImportError` branch).
- **Not recommended** for this change. The cost-to-risk ratio is poor. Revisit when
  tree-sitter 0.22 is actually required by a dependency.

### Option C — Replace `get_language()` + `Parser()` + `set_language()` with `get_parser()`

`tree_sitter_languages.get_parser(name)` returns a fully configured `Parser` in one call
and is already present in the installed `tree-sitter-languages==1.10.2`:

```python
from tree_sitter_languages import get_parser
p = get_parser('python')  # <-- still fires FutureWarning (calls get_language internally)
```

**Important:** `get_parser()` calls `get_language()` internally, so it also fires
the FutureWarning. This is not a solution to the warning. It is a code simplification
only, and it removes the ability to hold the `Language` object separately (needed by
some extractors for query compilation). **Not chosen.**

### Option D — Suppress the warning at the call site + add pytest filterwarnings

The warning is cosmetic noise from a transitive call inside `tree_sitter_languages`
that we do not control. The actual behaviour (Language object returned, parsing
correct) is unchanged. The correct fix for a FutureWarning in a third-party library
call is to suppress it at the call site — not to restructure our code.

Two-part fix:
1. Wrap every `get_language()` call in `warnings.catch_warnings()` to suppress the
   `FutureWarning` from `tree_sitter`.
2. Add `filterwarnings = ["ignore::FutureWarning:tree_sitter"]` to
   `[tool.pytest.ini_options]` so the 439 test-run warnings are gone.

This is clean, surgical, and does not change any grammar-loading logic. When
tree-sitter 0.22 support is eventually needed, the suppressors are removed as part
of that larger migration.

**Chosen approach.** See Section 4.

---

## 4. Recommended Approach: Targeted Warning Suppression

### 4.1 Rationale

- Zero grammar-loading logic changes — impossible to introduce parse regressions.
- Minimal diff: a helper function + one pytest.ini change.
- Reversible: delete the helper and the filterwarnings entry when upgrading to 0.22.
- Consistent with `cli/main.py`'s existing pattern, extended to the library layer.

### 4.2 New helper: `src/trelix/indexing/parser/_grammar.py`

Create a new module (not an extractor) that owns grammar loading:

```python
"""
Grammar loading helpers for tree-sitter 0.21.x.

tree_sitter_languages.get_language() calls Language(path, name) internally, which
fires a FutureWarning on tree-sitter 0.21.3. We suppress it at the call site.
When upgrading to tree-sitter 0.22 this module is the single place to update.
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

### 4.3 Extractor migration pattern

Before:
```python
import tree_sitter_languages
from tree_sitter import Node, Parser

class JavaScriptParser(BaseParser):
    def __init__(self) -> None:
        self._ts_language = tree_sitter_languages.get_language("javascript")
        self._parser = Parser()
        self._parser.set_language(self._ts_language)
```

After:
```python
from tree_sitter import Node, Parser

from .._grammar import load_language, make_parser

class JavaScriptParser(BaseParser):
    def __init__(self) -> None:
        self._ts_language = load_language("javascript")
        self._parser = make_parser("javascript")
```

Or more concisely where `_ts_language` is not used for query compilation:
```python
        self._parser = make_parser("javascript")
```

### 4.4 python.py migration

The `_get_python_language()` / `_make_parser()` functions in `python.py` already solve
the 0.21 vs 0.22 compatibility. Simplify to use `load_language`:

```python
# python.py — replace the two helper functions with:
from .._grammar import load_language, make_parser

class PythonParser(BaseParser):
    def __init__(self) -> None:
        self._ts_lang = load_language("python")
        self._parser = make_parser("python")
```

The `try/except ImportError` fallback for `tree_sitter_python` can be deleted; we are
pinned to `tree-sitter-languages>=1.10.2` which always provides `"python"`.

### 4.5 pytest filterwarnings

Add to `pyproject.toml`:

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

This eliminates the 439 warnings from the test run output. The library-level suppression
in `_grammar.py` covers runtime/library callers; this pytest entry covers any test that
constructs a parser directly or indirectly via fixtures.

---

## 5. Files Affected

| File | Change |
|---|---|
| `src/trelix/indexing/parser/_grammar.py` | **CREATE** — new `load_language()` / `make_parser()` helpers |
| `src/trelix/indexing/parser/__init__.py` | Verify `_grammar` is importable (no change if `__init__` uses `*`) |
| `src/trelix/indexing/parser/extractors/javascript.py` | Replace `get_language()` → `load_language()`, `Parser()` + `set_language()` → `make_parser()` |
| `src/trelix/indexing/parser/extractors/typescript.py` | Same |
| `src/trelix/indexing/parser/extractors/go.py` | Same |
| `src/trelix/indexing/parser/extractors/rust.py` | Same |
| `src/trelix/indexing/parser/extractors/java.py` | Same |
| `src/trelix/indexing/parser/extractors/kotlin.py` | Same |
| `src/trelix/indexing/parser/extractors/cpp.py` | Same |
| `src/trelix/indexing/parser/extractors/c.py` | Same |
| `src/trelix/indexing/parser/extractors/csharp.py` | Same |
| `src/trelix/indexing/parser/extractors/ruby.py` | Same |
| `src/trelix/indexing/parser/extractors/html.py` | Same |
| `src/trelix/indexing/parser/extractors/css.py` | Same |
| `src/trelix/indexing/parser/extractors/json_config.py` | Replace inline lazy `get_language()` → `load_language()` |
| `src/trelix/indexing/parser/extractors/toml_config.py` | Replace inline lazy `get_language()` → `load_language()` |
| `src/trelix/indexing/parser/extractors/python.py` | Remove `_get_python_language()` / `_make_parser()` functions; use `load_language()` / `make_parser()` |
| `pyproject.toml` | Add `filterwarnings` entry to `[tool.pytest.ini_options]` |

**Total: 16 extractor files + 1 new file + pyproject.toml = 18 files.**

The `Language()` call count being suppressed at source: **1 `Language()` call** in
`python.py` fallback path (dead code once `_grammar.py` is in place, but removed
cleanly). All 14 other extractors call `tree_sitter_languages.get_language()` which
fires the warning inside the library — these are suppressed by `load_language()`'s
`catch_warnings` context.

---

## 6. Backward Compatibility

- **No change to public API.** `BaseParser`, `ParseResult`, all `Symbol`/`CallEdge`/
  `ImportEdge` types, the registry, and the CLI are unaffected.
- **Parse output identical.** We are calling the same grammar via the same
  `tree_sitter_languages` package; only the warning is suppressed.
- **Python version.** Tested on Python 3.11 (project's only supported version).
- **tree-sitter 0.22 forward path.** When upgrading to 0.22 the migration is:
  1. Delete `_grammar.py`.
  2. Add per-language grammar packages (`tree-sitter-python`, etc.).
  3. Update extractors to `from tree_sitter_python import language as py_lang; Language(py_lang())`.
  4. Remove `filterwarnings` from `pyproject.toml`.
  The 0.22 migration is a single-file-per-extractor change with a clear mechanical
  pattern. Nothing in Phase 4 makes that harder.

---

## 7. Verification Criteria

After implementation:

1. `pytest --no-header -q 2>&1 | grep -c FutureWarning` returns `0`.
2. `pytest tests/ -x -q` exits 0 with all 699 tests passing.
3. `python -c "from trelix.indexing.parser.extractors.python import PythonParser; p = PythonParser(); print(p.parse('def foo(): pass', 'test.py'))"` exits 0 without any warning.
4. `grep -rn "tree_sitter_languages.get_language\|Language(tree_sitter" src/trelix/indexing/parser/extractors/` returns no matches.

---

## 8. Out of Scope

- Upgrading to tree-sitter 0.22+ (separate phase).
- Adding grammar packages for languages not currently in `tree-sitter-languages`.
- Changing the `parser.set_language()` → `Parser(language)` constructor call pattern
  (that is a 0.22 API change, not needed here).
- `markdown.py`, `razor.py`, `cshtml.py`, `csproj.py`, `yaml_config.py` — these use
  no tree-sitter grammar loading at all (regex or custom parsers) and require no changes.
