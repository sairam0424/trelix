"""
Back-compat tripwire: compression OFF must be BYTE-IDENTICAL to the pre-change
implementation.

Unlike ``test_assembler_compression.py`` (which compares the new assembler
against *itself* with ``compression_ratio=1.0``), this file diffs the new
assembler against the ACTUAL previous implementation, loaded verbatim out of
``git show v2.12.0:src/trelix/retrieval/assembler.py``. That is the only way to
catch a regression introduced by the ``_pack_breadth_first`` refactor or by the
new ``eligible``/``compressed`` plumbing in ``assemble()``.

The baseline is pinned to a RELEASE TAG rather than ``HEAD`` on purpose: a
``HEAD`` baseline self-destructs the instant the change under test is committed
(HEAD then contains compression, so there is nothing to diff and the whole
module skips). A tag is immutable, so these assertions keep running forever.

Zero network, zero embedding, zero DB: pure in-memory dataclasses + tiktoken.
"""

from __future__ import annotations

import hashlib
import importlib.util
import pathlib
import subprocess
import sys
import tempfile
from datetime import datetime

import pytest
import tiktoken

from trelix.core.models import (
    Chunk,
    IndexedFile,
    Language,
    SearchResult,
    Symbol,
    SymbolKind,
)
from trelix.retrieval.assembler import ContextAssembler

_ENC = tiktoken.get_encoding("cl100k_base")
_REPO = pathlib.Path(__file__).resolve().parents[2]
QUERY = "how does dispatch build the payload for stage 3"

INTENTS = [
    None,
    "file_overview",
    "project_overview",
    "comparison",
    "symbol_lookup",
    "dependency_map",
    "blast_radius",
    "feature_flow",
]


# ---------------------------------------------------------------------------
# Load the PRE-CHANGE assembler straight out of a pinned git tag
# ---------------------------------------------------------------------------

#: The baseline to diff against. v2.12.0 is the LAST RELEASE BEFORE SeleCom
#: context compression landed (it shipped in v3.0.0), so this tree is the
#: genuine pre-compression assembler — verified two ways: the file contains no
#: "compress" token at all, and it is byte-identical to the parent commit of the
#: change that introduced compression.
#:
#: Do NOT bump this to a newer tag. It is not "the previous release", it is "the
#: last compression-free release"; every tag from v3.0.0 on contains the change
#: under test, which would make the comparison vacuous (and is caught below).
_BASELINE_REF = "v2.12.0"
_BASELINE_PATH = "src/trelix/retrieval/assembler.py"


def _load_legacy_assembler() -> type:
    proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
        ["git", "-C", str(_REPO), "show", f"{_BASELINE_REF}:{_BASELINE_PATH}"],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
    )
    # allow_module_level: this runs at import time (the legacy class is built once
    # for the module), so a bare pytest.skip() here is a COLLECTION ERROR rather
    # than a skip. This is the FALLBACK path, not the normal one: it fires only
    # where git is missing or the tag was never fetched (e.g. actions/checkout
    # without fetch-tags — .github/workflows/ci.yml sets that so CI takes the
    # real path). If you see this skip, fix the checkout; do not accept it.
    if proc.returncode != 0:
        pytest.skip(
            f"cannot read {_BASELINE_REF}:{_BASELINE_PATH} ({proc.stderr.strip()}) — "
            "is the tag fetched? (actions/checkout needs fetch-tags: true)",
            allow_module_level=True,
        )
    source = proc.stdout
    if "compressor" in source:
        # A hard error, deliberately NOT a skip: a baseline that already contains
        # compression makes all 177 assertions below compare the change to
        # itself. Silently skipping is how this suite went dark the first time.
        raise RuntimeError(
            f"baseline {_BASELINE_REF}:{_BASELINE_PATH} already contains the compression "
            "change, so it is not a valid pre-change baseline. _BASELINE_REF must point at "
            "the last compression-free release, not simply the previous release."
        )

    slug = "".join(c if c.isalnum() else "_" for c in _BASELINE_REF)
    path = pathlib.Path(tempfile.gettempdir()) / f"trelix_legacy_assembler_golden_{slug}.py"
    path.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("trelix_legacy_assembler_golden", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.ContextAssembler  # type: ignore[no-any-return]


LegacyAssembler = _load_legacy_assembler()


# ---------------------------------------------------------------------------
# Fixtures — several files, mixed sources, a wide spread of body sizes
# ---------------------------------------------------------------------------


def _file(idx: int) -> IndexedFile:
    rel = f"src/pack/mod_{idx}.py"
    return IndexedFile(
        path=f"/repo/{rel}",
        rel_path=rel,
        language=Language.PYTHON,
        hash=f"sha-{idx}",
        size_bytes=4096,
        id=idx,
        indexed_at=datetime(2024, 1, 1),
    )


def _body(name: str, blocks: int) -> str:
    lines = [f"def {name}(request, session, retries):", f'    """Handle {name}."""']
    for b in range(blocks):
        lines.append("")
        lines.append(f"    # step {b}: prepare the payload for stage {b}")
        lines.append(f"    payload_{b} = build_payload(request, stage={b})")
        lines.append(f"    dispatched_{b} = dispatch(payload_{b}, timeout={b + 1})")
    lines.append("")
    lines.append("    return aggregate(dispatched_0)")
    return "\n".join(lines)


_KINDS = [
    SymbolKind.FUNCTION,
    SymbolKind.CLASS,
    SymbolKind.METHOD,
    SymbolKind.MODULE,
    SymbolKind.CONSTANT,
]
_SOURCES = ["vector", "bm25", "grep", "summary", "sparse"]


def _corpus() -> list[SearchResult]:
    """15 results over 4 files, deterministic, mixed sizes/kinds/sources."""
    results: list[SearchResult] = []
    sym_id = 0
    for i in range(15):
        sym_id += 1
        file = _file(i % 4)
        blocks = 1 + (i * 3) % 9  # 1..9 -> token counts spread widely
        name = f"handler_{i}"
        body = _body(name, blocks)
        line_start = 10 + i * 200
        symbol = Symbol(
            file_id=file.id or 1,
            name=name,
            qualified_name=f"Mod{i % 4}.{name}",
            kind=_KINDS[i % len(_KINDS)],
            line_start=line_start,
            line_end=line_start + len(body.splitlines()) - 1,
            signature=f"def {name}(request, session, retries)",
            body=body,
            docstring=f"Handle {name}.",
            id=sym_id,
        )
        results.append(
            SearchResult(
                chunk=Chunk(
                    symbol_id=sym_id,
                    chunk_text=body,
                    token_count=len(_ENC.encode(body)),
                    id=sym_id,
                ),
                symbol=symbol,
                file=file,
                score=1.0 - i * 0.05,
                rank=i + 1,
                source=_SOURCES[i % len(_SOURCES)],
            )
        )
    return results


CORPUS = _corpus()
TOTAL = sum(r.chunk.token_count for r in CORPUS)
SMALLEST = min(r.chunk.token_count for r in CORPUS)
# Budgets that exercise: nothing fits, one fits, partial packs, everything fits.
BUDGETS = [
    0,
    1,
    SMALLEST - 1,
    SMALLEST,
    TOTAL // 8,
    TOTAL // 4,
    TOTAL // 3,
    TOTAL // 2,
    (TOTAL * 3) // 4,
    TOTAL - 1,
    TOTAL,
    TOTAL + 500,
]


class _NeverCallMeCompressor:
    """Any consultation while 'disabled' is a hard failure."""

    last_path = None

    def compress(self, *args: object, **kwargs: object) -> object:  # noqa: ANN401
        raise AssertionError("compressor consulted while compression is disabled")


def _assert_identical(new_ctx: object, old_ctx: object, label: str) -> None:
    assert new_ctx.context_text == old_ctx.context_text, f"context_text differs: {label}"
    assert new_ctx.total_tokens == old_ctx.total_tokens, f"total_tokens differs: {label}"
    assert new_ctx.retrieval_sources == old_ctx.retrieval_sources, f"sources differ: {label}"
    assert new_ctx.intent == old_ctx.intent, f"intent differs: {label}"
    assert [id(r) for r in new_ctx.results] == [id(r) for r in old_ctx.results], (
        f"selection identity differs: {label}"
    )


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("budget", BUDGETS)
@pytest.mark.parametrize("mode", ["greedy", "breadth_first"])
@pytest.mark.parametrize("per_source", [False, True])
def test_default_ctor_is_byte_identical_to_head(budget: int, mode: str, per_source: bool) -> None:
    """No compressor supplied at all — the DEFAULT constructor."""
    new = ContextAssembler(token_budget=budget, per_source_budget=per_source).assemble(
        QUERY, CORPUS, assembly_mode=mode
    )
    old = LegacyAssembler(token_budget=budget, per_source_budget=per_source).assemble(
        QUERY, CORPUS, assembly_mode=mode
    )
    _assert_identical(new, old, f"budget={budget} mode={mode} per_source={per_source}")


@pytest.mark.parametrize("intent", INTENTS)
@pytest.mark.parametrize("mode", ["greedy", "breadth_first"])
def test_every_intent_preamble_is_byte_identical_to_head(intent: str | None, mode: str) -> None:
    budget = TOTAL // 3
    new = ContextAssembler(token_budget=budget).assemble(
        QUERY, CORPUS, intent=intent, assembly_mode=mode
    )
    old = LegacyAssembler(token_budget=budget).assemble(
        QUERY, CORPUS, intent=intent, assembly_mode=mode
    )
    _assert_identical(new, old, f"intent={intent} mode={mode}")


@pytest.mark.parametrize("ratio", [1.0, 1.5, 0.0, -0.5])
@pytest.mark.parametrize("mode", ["greedy", "breadth_first"])
@pytest.mark.parametrize("per_source", [False, True])
def test_inactive_ratio_with_compressor_is_byte_identical_to_head(
    ratio: float, mode: str, per_source: bool
) -> None:
    """A compressor may be attached; a non-shrinking ratio must be a no-op."""
    budget = TOTAL // 3
    new = ContextAssembler(
        token_budget=budget,
        per_source_budget=per_source,
        compressor=_NeverCallMeCompressor(),  # type: ignore[arg-type]
        compression_ratio=ratio,
    ).assemble(QUERY, CORPUS, assembly_mode=mode)
    old = LegacyAssembler(token_budget=budget, per_source_budget=per_source).assemble(
        QUERY, CORPUS, assembly_mode=mode
    )
    _assert_identical(new, old, f"ratio={ratio} mode={mode} per_source={per_source}")


def test_empty_results_is_byte_identical_to_head() -> None:
    new = ContextAssembler(token_budget=8_000).assemble(QUERY, [])
    old = LegacyAssembler(token_budget=8_000).assemble(QUERY, [])
    _assert_identical(new, old, "empty")


@pytest.mark.parametrize("max_per_file", [1, 2, 3, 5])
@pytest.mark.parametrize("budget", [0, SMALLEST, TOTAL // 4, TOTAL // 2, TOTAL + 500])
def test_pack_breadth_first_refactor_is_equivalent(max_per_file: int, budget: int) -> None:
    """The refactor split this into candidates+greedy — prove the split is exact."""
    new = ContextAssembler(token_budget=budget)._pack_breadth_first(CORPUS, max_per_file)
    old = LegacyAssembler(token_budget=budget)._pack_breadth_first(CORPUS, max_per_file)
    assert [id(r) for r in new] == [id(r) for r in old]


def test_query_embedding_kwarg_is_ignored_when_disabled() -> None:
    """Passing an embedding with no compressor must not change a single byte."""
    budget = TOTAL // 3
    new = ContextAssembler(token_budget=budget).assemble(QUERY, CORPUS, query_embedding=[0.1] * 8)
    old = LegacyAssembler(token_budget=budget).assemble(QUERY, CORPUS)
    _assert_identical(new, old, "query_embedding with no compressor")


# ---------------------------------------------------------------------------
# Retriever wiring: compression_enabled=False writes NO trace section
# ---------------------------------------------------------------------------


class _NoSubChunkDB:
    _conn = None

    def get_sub_chunks_for_symbol(self, symbol_id: int) -> list[object]:  # noqa: ARG002
        return []


def _stub_retriever(**retrieval_kwargs: object):  # noqa: ANN202
    from trelix.core.config import RetrievalConfig
    from trelix.retrieval.retriever import Retriever

    class _Cfg:
        def __init__(self) -> None:
            self.retrieval = RetrievalConfig(**retrieval_kwargs)  # type: ignore[arg-type]

    class _Stub:
        _assemble = Retriever._assemble
        _make_compressor = Retriever._make_compressor
        _cached_query_embedding = Retriever._cached_query_embedding
        _trace = Retriever._trace

        def __init__(self) -> None:
            self.config = _Cfg()
            self.db = _NoSubChunkDB()
            self.embedder = None
            self._effective_budget = TOTAL // 3

    return _Stub()


def test_retriever_assemble_disabled_matches_head_and_writes_no_trace() -> None:
    from trelix.retrieval import retriever as retriever_mod

    retriever_mod._trace_local.data = {}
    stub = _stub_retriever(compression_enabled=False)
    new = stub._assemble(QUERY, CORPUS, intent="feature_flow")
    old = LegacyAssembler(
        token_budget=TOTAL // 3,
        per_source_budget=stub.config.retrieval.context_budget_per_source,
    ).assemble(QUERY, CORPUS, intent="feature_flow")
    _assert_identical(new, old, "retriever._assemble disabled")
    assert "compression" not in retriever_mod._trace_local.data


def test_retriever_assemble_enabled_does_write_a_trace_section() -> None:
    """Counterpart: the tripwire above must be able to fail."""
    from trelix.retrieval import retriever as retriever_mod

    retriever_mod._trace_local.data = {}
    stub = _stub_retriever(compression_enabled=True)
    stub._assemble(QUERY, CORPUS, intent="feature_flow")
    assert "compression" in retriever_mod._trace_local.data


# ---------------------------------------------------------------------------
# Frozen digests — a second, independent tripwire
#
# These digests were captured from the implementation AFTER it was proven
# identical to the v2.12.0 baseline by the tests above. They pin the exact
# disabled-path text without consulting the baseline at all, so they also catch a
# drift that hits BOTH implementations (e.g. a change in _format_context shared
# via trelix.core.models). Note they still share this module's baseline-load
# skip; making them survive it would mean loading the baseline lazily per test.
# A change here means the default-off assembled context moved: either fix the
# change or (deliberately, with a reason) re-freeze.
# ---------------------------------------------------------------------------

_FROZEN_TOTAL = 2835  # sum of CORPUS chunk token_counts — guards the fixture itself

_FROZEN_DIGESTS: dict[str, str] = {
    "greedy|False|708|None": "1d504cea55459fb7",
    "greedy|False|708|file_overview": "beb47032f5164a58",
    "greedy|False|708|project_overview": "f6e121266071cdd7",
    "greedy|False|708|comparison": "52847f99bbd67ab9",
    "greedy|False|708|symbol_lookup": "a43a7447f6952ca7",
    "greedy|False|708|dependency_map": "1d504cea55459fb7",
    "greedy|False|1417|None": "4a23e0f14161602d",
    "greedy|False|1417|file_overview": "33583ebe3afeacbb",
    "greedy|False|1417|project_overview": "ed4998c1c6ccf5a6",
    "greedy|False|1417|comparison": "2595e5c6f3af17a7",
    "greedy|False|1417|symbol_lookup": "169d6ce9df345b13",
    "greedy|False|1417|dependency_map": "4a23e0f14161602d",
    "greedy|False|3335|None": "a93e167ebe232dc3",
    "greedy|False|3335|file_overview": "da7b6a4c3bbc72e9",
    "greedy|False|3335|project_overview": "786a5520619d5888",
    "greedy|False|3335|comparison": "6585ca09bc8d3e6c",
    "greedy|False|3335|symbol_lookup": "4382b825e02c18d3",
    "greedy|False|3335|dependency_map": "a93e167ebe232dc3",
    "greedy|True|708|None": "fd5c43528a8df364",
    "greedy|True|708|file_overview": "61b4430c6256474b",
    "greedy|True|708|project_overview": "219fe1079204a2ca",
    "greedy|True|708|comparison": "65c7835424d9ebaf",
    "greedy|True|708|symbol_lookup": "54fb26c569e731dc",
    "greedy|True|708|dependency_map": "fd5c43528a8df364",
    "greedy|True|1417|None": "9fba250c8c2ca411",
    "greedy|True|1417|file_overview": "bc995d1c27ef925d",
    "greedy|True|1417|project_overview": "3bbe16fbf83f576d",
    "greedy|True|1417|comparison": "21de1f14c0a98da1",
    "greedy|True|1417|symbol_lookup": "77e287651b78940c",
    "greedy|True|1417|dependency_map": "9fba250c8c2ca411",
    "greedy|True|3335|None": "a93e167ebe232dc3",
    "greedy|True|3335|file_overview": "da7b6a4c3bbc72e9",
    "greedy|True|3335|project_overview": "786a5520619d5888",
    "greedy|True|3335|comparison": "6585ca09bc8d3e6c",
    "greedy|True|3335|symbol_lookup": "4382b825e02c18d3",
    "greedy|True|3335|dependency_map": "a93e167ebe232dc3",
    "breadth_first|False|708|None": "acca94fd04e919de",
    "breadth_first|False|708|file_overview": "87d52daa22ea336d",
    "breadth_first|False|708|project_overview": "c6708f1c3995fc02",
    "breadth_first|False|708|comparison": "16052efcaae2f4da",
    "breadth_first|False|708|symbol_lookup": "adc1774af2b55786",
    "breadth_first|False|708|dependency_map": "acca94fd04e919de",
    "breadth_first|False|1417|None": "4a23e0f14161602d",
    "breadth_first|False|1417|file_overview": "3d77a65d82903eb1",
    "breadth_first|False|1417|project_overview": "ed4998c1c6ccf5a6",
    "breadth_first|False|1417|comparison": "2595e5c6f3af17a7",
    "breadth_first|False|1417|symbol_lookup": "169d6ce9df345b13",
    "breadth_first|False|1417|dependency_map": "4a23e0f14161602d",
    "breadth_first|False|3335|None": "4a23e0f14161602d",
    "breadth_first|False|3335|file_overview": "3d77a65d82903eb1",
    "breadth_first|False|3335|project_overview": "ed4998c1c6ccf5a6",
    "breadth_first|False|3335|comparison": "2595e5c6f3af17a7",
    "breadth_first|False|3335|symbol_lookup": "169d6ce9df345b13",
    "breadth_first|False|3335|dependency_map": "4a23e0f14161602d",
    "breadth_first|True|708|None": "acca94fd04e919de",
    "breadth_first|True|708|file_overview": "87d52daa22ea336d",
    "breadth_first|True|708|project_overview": "c6708f1c3995fc02",
    "breadth_first|True|708|comparison": "16052efcaae2f4da",
    "breadth_first|True|708|symbol_lookup": "adc1774af2b55786",
    "breadth_first|True|708|dependency_map": "acca94fd04e919de",
    "breadth_first|True|1417|None": "4a23e0f14161602d",
    "breadth_first|True|1417|file_overview": "3d77a65d82903eb1",
    "breadth_first|True|1417|project_overview": "ed4998c1c6ccf5a6",
    "breadth_first|True|1417|comparison": "2595e5c6f3af17a7",
    "breadth_first|True|1417|symbol_lookup": "169d6ce9df345b13",
    "breadth_first|True|1417|dependency_map": "4a23e0f14161602d",
    "breadth_first|True|3335|None": "4a23e0f14161602d",
    "breadth_first|True|3335|file_overview": "3d77a65d82903eb1",
    "breadth_first|True|3335|project_overview": "ed4998c1c6ccf5a6",
    "breadth_first|True|3335|comparison": "2595e5c6f3af17a7",
    "breadth_first|True|3335|symbol_lookup": "169d6ce9df345b13",
    "breadth_first|True|3335|dependency_map": "4a23e0f14161602d",
}


def test_fixture_token_total_is_stable() -> None:
    """If this fails the digests below are meaningless — fix the fixture first."""
    assert TOTAL == _FROZEN_TOTAL


@pytest.mark.parametrize("key", sorted(_FROZEN_DIGESTS))
def test_disabled_output_matches_frozen_digest(key: str) -> None:
    mode, per_source, budget, intent = key.split("|")
    context = ContextAssembler(
        token_budget=int(budget), per_source_budget=per_source == "True"
    ).assemble(
        QUERY,
        CORPUS,
        intent=None if intent == "None" else intent,
        assembly_mode=mode,
    )
    digest = hashlib.sha256(context.context_text.encode()).hexdigest()[:16]
    assert digest == _FROZEN_DIGESTS[key], f"default-off assembled context moved for {key}"
