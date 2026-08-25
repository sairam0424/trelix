"""One metric implementation, and it is the shipped one.

``tests/eval/metrics.py`` used to be a second implementation of recall@k / MRR /
nDCG@k, and the CI retrieval-quality gate computed its verdict with THAT one while
``trelix eval`` shipped ``src/trelix/eval/ndcg.py``. Measured on this tree, on the
identical ranking with the identical ground truth:

    ranking (chunk order)                        ground truth   deleted  shipped
    ["src/other.py"] * 10 + ["src/auth.py"]      src/auth.py    0.000000 0.630930
    ["src/other.py"] *  9 + ["src/auth.py"]      src/auth.py    0.289065 0.630930
    [superuser.py, a, b, c, src/user.py]         src/user.py    1.000000 0.386853

Worst divergence 0.6309297536 nDCG@10 — on the first row the gate's metric reported a
total retrieval failure for a ranking whose real nDCG@10 is 0.63, and on the third it
reported a perfect hit on the WRONG file. Two causes:

1. ``@k`` applied to CHUNK positions instead of distinct FILES (the shipped metric
   collapses repeats to their best rank first, so ``@10`` means ten files).
2. Ground truth matched by SUBSTRING, so ``"user.py"`` matched ``src/superuser.py``.

This file pins that the second implementation is gone, that the boundary the gate now
scores through (``tests.eval.harness.score_ranking``) reproduces the shipped numbers
on exactly those rankings, and that the shipped numbers agree with an independent
implementation.

The metric ARITHMETIC is not re-pinned here — ``tests/unit/test_ndcg_known_answers.py``
already does that with hand-derived literals, and duplicating it would be the same
mistake in test form.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests.eval.harness import EvalHarness, score_ranking
from trelix.eval import ndcg as shipped

_REL = 1e-12
_TESTS_DIR = Path(__file__).parent.parent
_SHIPPED_SOURCE = Path(shipped.__file__)

# The names that constitute a retrieval-metric implementation. Explicit table, not a
# pattern: a pattern would silently under-match a renamed copy.
METRIC_DEFINITION_NAMES = frozenset({"ndcg_at_k", "recall_at_k", "mrr", "reciprocal_rank", "dcg"})


def _is_property(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    return any(isinstance(d, ast.Name) and d.id == "property" for d in node.decorator_list)


def _defined_function_names(path: Path) -> set[str]:
    """Every function name defined in `path` at any nesting depth, minus ``@property``.

    Properties are excluded because ``EvalReport.mrr`` is a zero-argument accessor that
    averages the per-query numbers the shipped ``mrr`` already produced — an aggregate
    over results, not an implementation of a metric. Anything that could actually score
    a ranking takes arguments and so is not a property.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and not _is_property(node)
    }


class TestNoSecondMetricImplementation:
    def test_the_scanner_finds_metric_definitions_where_they_do_live(self) -> None:
        """Precondition (rule 4) for the two tests below, which assert an ABSENCE.

        Named fixture: ``src/trelix/eval/ndcg.py``. If ``_defined_function_names``
        stopped finding definitions — a broken parse, a moved file — the absence
        assertions would pass vacuously against every file in the tree.

        NON-DISCRIMINATING COMPANION: no mutation of the shipped metric bodies fails
        this test; it exists solely to prove the scanner is not blind.
        """
        found = _defined_function_names(_SHIPPED_SOURCE) & METRIC_DEFINITION_NAMES
        assert found == {"ndcg_at_k", "recall_at_k", "mrr", "dcg"}

    def test_no_metric_is_implemented_anywhere_under_tests_eval(self) -> None:
        """MUTATION THAT MUST FAIL THIS TEST:
        re-add a ``def ndcg_at_k(...)`` (or recall_at_k / mrr / reciprocal_rank) to any
        module under ``tests/eval/`` — i.e. reintroduce the deleted duplicate.
        """
        offenders: dict[str, set[str]] = {}
        for path in sorted((_TESTS_DIR / "eval").rglob("*.py")):
            clash = _defined_function_names(path) & METRIC_DEFINITION_NAMES
            if clash:
                offenders[path.relative_to(_TESTS_DIR).as_posix()] = clash
        assert offenders == {}, (
            "a second retrieval-metric implementation is back under tests/eval/: "
            f"{offenders} — the gate must score with trelix.eval.ndcg, the module "
            "trelix eval reports from"
        )

    def test_tests_eval_metrics_module_is_gone(self) -> None:
        """MUTATION THAT MUST FAIL THIS TEST:
        recreate ``tests/eval/metrics.py`` (even empty).
        """
        assert not (_TESTS_DIR / "eval" / "metrics.py").exists()
        with pytest.raises(ModuleNotFoundError):
            __import__("tests.eval.metrics")

    def test_the_harness_imports_its_metrics_from_the_shipped_module(self) -> None:
        """MUTATION THAT MUST FAIL THIS TEST:
        in ``tests/eval/harness.py``, change ``from trelix.eval.ndcg import ...`` to
        import from anywhere else (or drop it and hand-roll the arithmetic).
        """
        tree = ast.parse((_TESTS_DIR / "eval" / "harness.py").read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "trelix.eval.ndcg":
                imported.update(alias.name for alias in node.names)
        assert imported == {"mrr", "ndcg_at_k", "recall_at_k"}


class TestTheTwoDivergencesAreGone:
    """Regression tests for the exact rankings on which the two implementations split.

    Every expected value is a decimal literal, hand-derived in the docstring from
    log2(3) = 1.584962500721156 and log2(6) = 2.584962500721156. Nothing here recomputes
    an expectation with the code under test.
    """

    def test_chunk_repeats_do_not_push_a_relevant_file_past_the_cutoff(self) -> None:
        """MUTATION THAT MUST FAIL THIS TEST:
        src/trelix/eval/ndcg.py: ``actual = dcg(_dedupe(ranked_ids), ...)`` ->
        ``actual = dcg(ranked_ids, ...)`` — i.e. score chunk positions again, which is
        precisely what the deleted duplicate did.

        Hand arithmetic. Ten chunks of one irrelevant file, then the relevant file:
        after collapsing repeats the ranking is [other, auth], so the single relevant
        document sits at rank 2 and IDCG is 1.0 (one relevant document at rank 1):

            nDCG@10 = 1 / log2(3) = 1 / 1.584962500721156 = 0.6309297535714575
            MRR     = 1 / 2       = 0.5
            recall@10 = 1 of 1 relevant found = 1.0

        The deleted implementation scored 0.0 here: it counted raw chunk positions, put
        the relevant file at position 11, and applied k = 10.
        """
        ranked = ["src/other.py"] * 10 + ["src/auth.py"]
        # Precondition (rule 4): the fixture only discriminates while the relevant file
        # sits beyond raw position k=10 yet inside the deduplicated top 10.
        assert ranked.index("src/auth.py") == 10
        assert len(dict.fromkeys(ranked)) == 2

        r1, r5, r10, rr, ndcg, rank = score_ranking(ranked, ["src/auth.py"])
        assert rank == 2
        assert rr == pytest.approx(0.5, rel=_REL)
        assert ndcg == pytest.approx(0.6309297535714575, rel=_REL)
        assert r10 == pytest.approx(1.0, rel=_REL)
        assert r5 == pytest.approx(1.0, rel=_REL)
        assert r1 == pytest.approx(0.0, rel=_REL)

    def test_a_bare_stem_matches_nothing_rather_than_the_wrong_file(self) -> None:
        """MUTATION THAT MUST FAIL THIS TEST:
        tests/eval/harness.py, ``score_ranking``:
            ``relevant = set(relevant_files)``
            -> ``relevant = {f for f in ranked_files for rel in relevant_files
                             if rel in f} or set(relevant_files)``
        i.e. the deleted duplicate's substring rule.

        The ground truth here is the bare stem ``user.py``. No retrieved path IS that
        string, so exact matching scores nothing: rank -1, nDCG@10 0.0. Under substring
        matching the stem matches ``src/superuser.py`` at rank 1 and the run scores a
        PERFECT 1.0 on a file that is not the answer.

        A correction worth recording, because the first version of this test could not
        see the mutation at all: it used the full path ``src/user.py`` as ground truth,
        and ``"src/superuser.py"`` does not contain ``"src/user.py"`` — only the stem
        ``"user.py"``. Substring and exact matching agree on that input, so the mutant
        SURVIVED. The stem has to be the ground truth for the two rules to diverge.

        Scoring 0.0 is not the harness's answer to a stem, only ``score_ranking``'s:
        ``EvalHarness._validate_cases`` refuses such a case before any query runs, which
        the test below pins.
        """
        ranked = ["src/superuser.py", "a.py", "b.py", "c.py", "src/user.py"]
        # Precondition (rule 4): the decoy must contain the stem while not being the
        # answer, or substring and exact matching agree and the case proves nothing.
        assert "user.py" in ranked[0]
        assert ranked[0] != "src/user.py"

        r1, r5, r10, rr, ndcg, rank = score_ranking(ranked, ["user.py"])
        assert rank == -1
        assert rr == 0.0
        assert ndcg == 0.0
        assert r1 == 0.0
        assert r5 == 0.0
        assert r10 == 0.0

    def test_a_relevant_file_behind_a_superstring_decoy_scores_its_own_rank(self) -> None:
        """MUTATION THAT MUST FAIL THIS TEST:
        src/trelix/eval/ndcg.py, ``mrr``: ``enumerate(_dedupe(ranked_ids), start=1)`` ->
        ``enumerate(_dedupe(ranked_ids), start=2)`` (or any off-by-one in the rank), which
        would move every ledger budget by one.

        Hand arithmetic. Ground truth ``src/user.py`` at rank 5 behind four other files:

            nDCG@10 = 1 / log2(6) = 1 / 2.584962500721156 = 0.38685280723454163
            MRR     = 1 / 5 = 0.2
            recall@5 = 1.0 (rank 5 is inside the top 5), recall@1 = 0.0

        Stated plainly: this case is INVARIANT to the substring/exact rule (the decoy
        does not contain the full path), so it is not the substring pin — the test above
        is. It pins the rank arithmetic the ledger budgets are expressed in.
        """
        ranked = ["src/superuser.py", "a.py", "b.py", "c.py", "src/user.py"]
        r1, r5, r10, rr, ndcg, rank = score_ranking(ranked, ["src/user.py"])
        assert rank == 5
        assert rr == pytest.approx(0.2, rel=_REL)
        assert ndcg == pytest.approx(0.38685280723454163, rel=_REL)
        assert r5 == pytest.approx(1.0, rel=_REL)
        assert r1 == pytest.approx(0.0, rel=_REL)


class TestTheHarnessRefusesUnusableGroundTruth:
    """Exact matching turns a stale fixture into a silent 0.0 unless it is refused."""

    def test_a_bare_stem_is_refused_before_any_query_runs(self, tmp_path: Path) -> None:
        """MUTATION THAT MUST FAIL THIS TEST:
        tests/eval/harness.py, ``_validate_cases``: delete the
        ``if not (root / expected).exists():`` branch. The bare stem would then be
        accepted, score 0.0 on every query, and read as total retrieval failure — the
        exact misreading that made the substring rule look reasonable in the first place.

        No indexing happens here: ``run()`` validates before it indexes, so this raises
        without touching an embedder, a model or a socket.
        """
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "user.py").write_text("x = 1\n", encoding="utf-8")
        harness = EvalHarness(repo_path=str(tmp_path))

        with pytest.raises(ValueError, match="does not exist under"):
            harness.run([("who is the user", "user.py")])

        # Precondition (rule 4): the check must ACCEPT the corrected path, or it would
        # be refusing everything and the assertion above would pass for the wrong reason.
        # Validation is called directly rather than through run() because run() would go
        # on to index a repo, which this hermetic suite must not do.
        harness._validate_cases([("who is the user", "src/user.py")])


class TestRankIsTheMetricsOwnRank:
    """The ledger's rank must be the rank the shipped MRR scored, not a second opinion."""

    def test_rank_round_trips_through_the_shipped_reciprocal_rank(self) -> None:
        """MUTATION THAT MUST FAIL THIS TEST:
        tests/eval/harness.py, ``_rank_from_mrr``: ``round(1.0 / reciprocal)`` ->
        ``int(1.0 / reciprocal)``.

        Correction to an earlier draft of this docstring, which claimed rank 3 arrives
        as 3.0000000000000004 and truncates to 2. It does arrive as
        3.0000000000000004, and ``int()`` returns 3 — truncation is harmless there, and
        the mutation SURVIVED the small-rank table. Measured over ranks 1..100000,
        ``int(1.0 / (1.0 / r)) != r`` for 5,850 ranks, the smallest being **93**
        (``1.0 / (1.0 / 93) == 92.99999999999999``), then 99, 105, 117, 123. So the
        first three assertions below are a NON-DISCRIMINATING companion kept for
        readability, and the rank-93 case is what kills the mutation.

        This is not a hypothetical: ``score_ranking`` takes whatever ranked list the
        retriever produced, and one file can occupy many chunk slots, so a
        distinct-file rank in the nineties is reachable on a large repo. Under the
        mutation that rank reports 92 and would clear a budget of 92.

        Explicit table, one flat assertion per rank (rule 2).
        """
        assert score_ranking(["a"], ["a"])[5] == 1
        assert score_ranking(["x", "a"], ["a"])[5] == 2
        assert score_ranking(["x", "y", "a"], ["a"])[5] == 3
        assert score_ranking(["x", "y", "z", "a"], ["a"])[5] == 4
        assert score_ranking(["1", "2", "3", "4", "5", "6", "a"], ["a"])[5] == 7
        assert score_ranking(["1", "2", "3", "4", "5", "6", "7", "8", "9", "a"], ["a"])[5] == 10

        # rank 93: 92 distinct decoys, then the relevant file.
        deep = [f"d{i}.py" for i in range(92)] + ["a"]
        assert len(dict.fromkeys(deep)) == 93  # precondition: no accidental duplicates
        assert score_ranking(deep, ["a"])[5] == 93

    def test_a_relevant_file_that_was_never_retrieved_ranks_minus_one(self) -> None:
        """MUTATION THAT MUST FAIL THIS TEST:
        tests/eval/harness.py, ``_rank_from_mrr``: ``if reciprocal <= 0.0: return -1``
        -> ``return 1``. That would report the best possible rank for a file that was
        never retrieved, and the ledger would pass on total failure.
        """
        r1, r5, r10, rr, ndcg, rank = score_ranking(["x", "y"], ["src/missing.py"])
        assert rank == -1
        assert rr == 0.0
        assert ndcg == 0.0
        assert r10 == 0.0


class TestShippedMetricAgreesWithAnIndependentImplementation:
    def test_ndcg_matches_sklearn_on_the_divergence_cases(self) -> None:
        """Cross-check against a metric implementation nobody here wrote.

        MUTATION THAT MUST FAIL THIS TEST:
        src/trelix/eval/ndcg.py, ``dcg``: ``1.0 / math.log2(rank + 2)`` ->
        ``1.0 / math.log2(rank + 3)``.

        scikit-learn is only present transitively (``sentence-transformers`` requires
        it, via the ``local`` extra that both CI test jobs install). It is NOT a
        declared trelix dependency, so a leaner install can legitimately lack it —
        skip loudly there rather than assert something the environment does not
        guarantee.

        The comparison is fed the DEDUPLICATED distinct-document ranking, because
        sklearn has no notion of a document appearing twice; deduplication is the
        shipped contract (``@k`` counts distinct files), not a concession made to make
        this test pass.

        Agreement is asserted at 1e-12, not 0.0: measured |shipped - sklearn| is
        2.22e-16 (one double ULP) on three of these four cases, from a different but
        algebraically identical summation order.
        """
        np = pytest.importorskip("numpy", reason="scikit-learn oracle needs numpy")
        skmetrics = pytest.importorskip(
            "sklearn.metrics",
            reason="scikit-learn absent (transitive dep of the 'local' extra); the "
            "independent nDCG oracle is out of reach in this environment",
        )

        def oracle(ranked: list[str], relevant: set[str], k: int) -> float:
            docs = list(dict.fromkeys(ranked))
            docs += sorted(relevant - set(docs))
            y_true = np.array([[1.0 if d in relevant else 0.0 for d in docs]])
            y_score = np.array([[float(len(docs) - i) for i in range(len(docs))]])
            return float(skmetrics.ndcg_score(y_true, y_score, k=k))

        def shipped_ndcg(ranked: list[str], relevant: set[str], k: int) -> float:
            universe = {f: i for i, f in enumerate(set(ranked) | relevant)}
            return shipped.ndcg_at_k(
                [universe[f] for f in ranked], {universe[f] for f in relevant}, k=k
            )

        # Explicit cases, one flat pair of assertions each (rule 2).
        case_a = (["src/other.py"] * 10 + ["src/auth.py"], {"src/auth.py"})
        assert shipped_ndcg(*case_a, 10) == pytest.approx(oracle(*case_a, 10), abs=1e-12)
        assert shipped_ndcg(*case_a, 10) == pytest.approx(0.6309297535714575, rel=_REL)

        case_b = (["src/superuser.py", "a.py", "b.py", "c.py", "src/user.py"], {"src/user.py"})
        assert shipped_ndcg(*case_b, 10) == pytest.approx(oracle(*case_b, 10), abs=1e-12)
        assert shipped_ndcg(*case_b, 10) == pytest.approx(0.38685280723454163, rel=_REL)

        case_c = (["x", "src/a.py", "y", "src/b.py"], {"src/a.py", "src/b.py"})
        assert shipped_ndcg(*case_c, 10) == pytest.approx(oracle(*case_c, 10), abs=1e-12)
        assert shipped_ndcg(*case_c, 10) == pytest.approx(0.6509209298071326, rel=_REL)

        # A relevant file that was never retrieved. The 12 irrelevant fillers are
        # load-bearing, not padding for its own sake: sklearn scores a dense vector, so
        # the unretrieved relevant document has to be REPRESENTED somewhere, and the
        # oracle can only agree with "absent from the ranking" if that somewhere is
        # beyond the k=10 cutoff. Measured with only 3 fillers, sklearn credits it at
        # the bottom of the list and returns 0.8772153153380493 against the shipped
        # 0.6131471927654584 — the oracle's own artefact, not a metric disagreement.
        fillers = [f"f{i}.py" for i in range(12)]
        case_d = (["src/a.py", *fillers], {"src/a.py", "src/missing.py"})
        assert len(dict.fromkeys(case_d[0])) + 1 > 10
        assert shipped_ndcg(*case_d, 10) == pytest.approx(oracle(*case_d, 10), abs=1e-12)
        assert shipped_ndcg(*case_d, 10) == pytest.approx(0.6131471927654584, rel=_REL)
