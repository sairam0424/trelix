"""Breadth floor on the three direct-lookup retrieval paths.

Before this floor existed, `_retrieve_file_overview`, `_retrieve_project_overview` and
`_retrieve_config` widened to standard retrieval only on a total miss (`if not results:`).
One matched file yielding a handful of symbols returned exclusively those symbols and
suppressed the vector, BM25 and grep legs entirely. Reproduced on this tree with a mocked
DB (1 file, 2 symbol ids): `_retrieve_standard` was never called and assembly saw 1
result.

These tests pin, per path:
  - thin direct result -> standard candidates ALSO run and are merged
  - direct hits come FIRST in the merged list (high precision, order not score)
  - broad direct result -> no widening (the cheap path stays cheap)
  - the `breadth_floor` trace section records the decision either way
  - TRELIX_RETRIEVAL_BREADTH_FLOOR=false restores the old short-circuit exactly
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from trelix.retrieval.planner.models import (
    INTENT_STRATEGIES,
    IntentType,
    QueryPlan,
    SubQuery,
)
from trelix.retrieval.retriever import (
    _breadth_floor_thresholds,
)

from .test_retriever_core import (
    _build_retriever,
    _make_chunk,
    _make_file,
    _make_retrieved_context,
    _make_search_result,
    _make_symbol,
)

# Every env var this module reads, cleared per test so a developer's shell cannot
# silently flip the floor off and turn these assertions green for the wrong reason.
_FLOOR_ENV = (
    "TRELIX_RETRIEVAL_BREADTH_FLOOR",
    "TRELIX_RETRIEVAL_BREADTH_FLOOR_MIN_FILES",
    "TRELIX_RETRIEVAL_BREADTH_FLOOR_MIN_SYMBOLS",
)


@pytest.fixture(autouse=True)
def _clean_floor_env() -> None:
    with patch.dict(os.environ, {k: "" for k in _FLOOR_ENV}, clear=False):
        for k in _FLOOR_ENV:
            os.environ.pop(k, None)
        yield


def _mock_direct_lookup(retriever, *, file_ids: list[int], symbols_per_file: int) -> None:
    """Point the DB mocks at N files with M distinct symbols each.

    Symbol ids are globally unique so `_dedup` (keyed on symbol_id) cannot collapse
    them — otherwise a "10 symbols" case would silently become a 1-symbol case.
    """
    files = {fid: _make_file(fid, rel_path=f"cfg/file_{fid}.yml") for fid in file_ids}
    symbols: dict[int, tuple] = {}
    per_file: dict[int, list[int]] = {}
    next_sid = 1000
    for fid in file_ids:
        ids = []
        for _ in range(symbols_per_file):
            next_sid += 1
            symbols[next_sid] = (_make_symbol(next_sid, fid, name=f"sym_{next_sid}"), files[fid])
            ids.append(next_sid)
        per_file[fid] = ids

    retriever.db.find_file_by_path_fragment.side_effect = lambda hint: list(file_ids)
    retriever.db.get_all_symbols_for_file.side_effect = lambda fid: per_file[fid]
    retriever.db.get_symbol_with_file.side_effect = lambda sid: symbols[sid]
    retriever.db.get_first_chunk_for_symbol.side_effect = lambda sid: _make_chunk(sid)


def _config_plan(query: str = "what port does the container expose") -> QueryPlan:
    return QueryPlan(
        intent=IntentType.CONFIG_LOOKUP,
        execution_mode="sequential",
        strategy=INTENT_STRATEGIES[IntentType.CONFIG_LOOKUP],
        sub_queries=[
            SubQuery(
                semantic_query=query,
                hyde_snippet="",
                bm25_tokens=["EXPOSE"],
                grep_hints=["Dockerfile"],
                file_hints=[],
            )
        ],
        raw_query=query,
    )


def _file_overview_plan(hint: str, query: str = "what does this file do") -> QueryPlan:
    return QueryPlan(
        intent=IntentType.FILE_OVERVIEW,
        execution_mode="sequential",
        strategy=INTENT_STRATEGIES[IntentType.FILE_OVERVIEW],
        sub_queries=[
            SubQuery(
                semantic_query=query,
                hyde_snippet="",
                bm25_tokens=[],
                grep_hints=[hint],
                file_hints=[],
            )
        ],
        raw_query=query,
    )


# ---------------------------------------------------------------------------
# Threshold resolution
# ---------------------------------------------------------------------------


# Sourced from the config rather than restated: these were module constants in retriever.py
# before the knobs were promoted to RetrievalConfig, and a hardcoded copy here would silently
# stop matching the code the day a default changes.
def _default(field: str) -> int:
    from trelix.core.config import RetrievalConfig

    return int(RetrievalConfig.model_fields[field].default)


_BREADTH_FLOOR_MIN_FILES = _default("breadth_floor_min_files")
_BREADTH_FLOOR_MIN_SYMBOLS = _default("breadth_floor_min_symbols")


class TestBreadthFloorThresholds:
    """The knobs resolve through RetrievalConfig, not through raw os.environ.

    They started as `os.environ.get` reads, with the stated intent to "promote to
    RetrievalConfig once the numbers are settled". They are settled — the floor restored
    nDCG@10 to 0.6189/0.6217 on the 50-query set — and the raw reads had a defect worth
    naming: `os.environ` does not see `.env`, so the kill switch documented in the release
    notes was silently inert in the one place a user would set it. `CONTRIBUTING.md` also
    makes every `TRELIX_*` name in `.env.example` stable public API, which an os.environ
    read cannot participate in.
    """

    @staticmethod
    def _resolve() -> tuple[bool, int, int]:
        from trelix.core.config import RetrievalConfig

        return _breadth_floor_thresholds(RetrievalConfig())

    def test_defaults_when_env_unset(self) -> None:
        assert self._resolve() == (True, 2, 10)

    @pytest.mark.parametrize("value", ["false", "FALSE", "0", "no", "off"])
    def test_kill_switch_recognised(self, value: str) -> None:
        """pydantic's bool parsing must accept the same spellings the old reader did."""
        with patch.dict(os.environ, {"TRELIX_RETRIEVAL_BREADTH_FLOOR": value}):
            assert self._resolve()[0] is False

    def test_thresholds_overridable(self) -> None:
        with patch.dict(
            os.environ,
            {
                "TRELIX_RETRIEVAL_BREADTH_FLOOR_MIN_FILES": "5",
                "TRELIX_RETRIEVAL_BREADTH_FLOOR_MIN_SYMBOLS": "42",
            },
        ):
            assert self._resolve() == (True, 5, 42)

    def test_a_malformed_override_now_fails_fast_naming_the_field(self) -> None:
        """DELIBERATE BEHAVIOUR CHANGE, for consistency inside this release.

        The os.environ version fell back to the default and logged a warning, reasoning that
        "a malformed env var must not take retrieval down". The retrieval-weight parser in
        this same release made the opposite call, and gave the reason: a silently ignored
        override leaves you measuring the default while believing you measured your value —
        which is the exact failure class this whole release is about. Two knobs in one config
        should not disagree about it.

        pydantic reports the field and the bad value, which is strictly better than the
        warning was: the old message named the env var but the run continued, so a CI job
        tuning the floor would have recorded numbers for the default.
        """
        from pydantic import ValidationError

        from trelix.core.config import RetrievalConfig

        with patch.dict(os.environ, {"TRELIX_RETRIEVAL_BREADTH_FLOOR_MIN_FILES": "lots"}):
            with pytest.raises(ValidationError) as excinfo:
                RetrievalConfig()

        rendered = str(excinfo.value)
        assert "breadth_floor_min_files" in rendered or "MIN_FILES" in rendered.upper(), rendered

    def test_a_negative_threshold_is_rejected(self) -> None:
        """`ge=0` — a negative floor would make the condition unsatisfiable, silently."""
        from pydantic import ValidationError

        from trelix.core.config import RetrievalConfig

        with patch.dict(os.environ, {"TRELIX_RETRIEVAL_BREADTH_FLOOR_MIN_SYMBOLS": "-1"}):
            with pytest.raises(ValidationError):
                RetrievalConfig()


# ---------------------------------------------------------------------------
# config_lookup — the measured Dockerfile case
# ---------------------------------------------------------------------------


class TestConfigLookupBreadthFloor:
    def test_thin_direct_hit_also_runs_standard_and_merges(self, tmp_path: Path) -> None:
        """1 file / 2 symbols must NOT suppress the vector, BM25 and grep legs."""
        retriever = _build_retriever(str(tmp_path))
        _mock_direct_lookup(retriever, file_ids=[1], symbols_per_file=2)

        standard = [_make_search_result(idx=i, score=0.5, source="vector") for i in (7, 8, 9)]

        with (
            patch.object(retriever, "_standard_candidates", return_value=standard) as mock_std,
            patch.object(
                retriever, "_assemble", return_value=_make_retrieved_context()
            ) as mock_assemble,
        ):
            retriever._retrieve_config(_config_plan())

        mock_std.assert_called_once()
        merged = mock_assemble.call_args[0][1]
        assert len(merged) == 5, "2 direct + 3 standard candidates should reach assembly"

    def test_direct_hits_rank_first_in_merged_list(self, tmp_path: Path) -> None:
        """Direct hits are high precision; the greedy pack consumes list order."""
        retriever = _build_retriever(str(tmp_path))
        _mock_direct_lookup(retriever, file_ids=[1], symbols_per_file=2)

        # Standard candidates deliberately outscore the direct ones (a Cohere rerank
        # score can be ~1.0), so a score sort would put them first.
        standard = [_make_search_result(idx=i, score=0.99, source="vector") for i in (7, 8)]

        with (
            patch.object(retriever, "_standard_candidates", return_value=standard),
            patch.object(retriever, "_assemble", return_value=_make_retrieved_context()) as asm,
        ):
            retriever._retrieve_config(_config_plan())

        merged = asm.call_args[0][1]
        assert [r.source for r in merged[:2]] == ["file_direct", "file_direct"]
        assert all(r.source == "vector" for r in merged[2:])

    def test_broad_direct_hit_does_not_widen(self, tmp_path: Path) -> None:
        """The cheap path stays cheap when the direct lookup already covers the answer."""
        retriever = _build_retriever(str(tmp_path))
        _mock_direct_lookup(
            retriever, file_ids=[1], symbols_per_file=_BREADTH_FLOOR_MIN_SYMBOLS + 1
        )

        with (
            patch.object(retriever, "_standard_candidates") as mock_std,
            patch.object(retriever, "_assemble", return_value=_make_retrieved_context()) as asm,
        ):
            retriever._retrieve_config(_config_plan())

        mock_std.assert_not_called()
        assert all(r.source == "file_direct" for r in asm.call_args[0][1])

    def test_many_files_does_not_widen(self, tmp_path: Path) -> None:
        """The file count alone clears the floor — both counts must be short to fire."""
        retriever = _build_retriever(str(tmp_path))
        _mock_direct_lookup(
            retriever, file_ids=list(range(1, _BREADTH_FLOOR_MIN_FILES + 1)), symbols_per_file=1
        )

        with (
            patch.object(retriever, "_standard_candidates") as mock_std,
            patch.object(retriever, "_assemble", return_value=_make_retrieved_context()),
        ):
            retriever._retrieve_config(_config_plan())

        mock_std.assert_not_called()

    def test_kill_switch_restores_short_circuit(self, tmp_path: Path) -> None:
        """TRELIX_RETRIEVAL_BREADTH_FLOOR=false reproduces the pre-3.1.2 behaviour.

        The env var is set BEFORE the Retriever is built, which is now load-bearing. The
        knob moved from a per-call `os.environ.get` to a RetrievalConfig field, so it is
        resolved once when the config is constructed — the same as every other
        `TRELIX_RETRIEVAL_*` setting. Patching the environment around the call alone has no
        effect, which is exactly what this test caught when the promotion landed.
        """
        with patch.dict(os.environ, {"TRELIX_RETRIEVAL_BREADTH_FLOOR": "false"}):
            retriever = _build_retriever(str(tmp_path))
            _mock_direct_lookup(retriever, file_ids=[1], symbols_per_file=2)

            with (
                patch.object(retriever, "_standard_candidates") as mock_std,
                patch.object(retriever, "_assemble", return_value=_make_retrieved_context()) as asm,
            ):
                retriever._retrieve_config(_config_plan())

        mock_std.assert_not_called()
        assert len(asm.call_args[0][1]) == 2

    def test_standard_leg_failure_degrades_to_direct_and_warns(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A query that answered before the floor existed must not start failing."""
        retriever = _build_retriever(str(tmp_path))
        _mock_direct_lookup(retriever, file_ids=[1], symbols_per_file=2)

        with (
            patch.object(
                retriever, "_standard_candidates", side_effect=RuntimeError("embedder down")
            ),
            patch.object(retriever, "_assemble", return_value=_make_retrieved_context()) as asm,
            caplog.at_level("WARNING", logger="trelix.retrieval"),
        ):
            retriever._retrieve_config(_config_plan())

        assert len(asm.call_args[0][1]) == 2
        assert "breadth-floor standard leg failed" in caplog.text


# ---------------------------------------------------------------------------
# file_overview — the looser hint-promotion path
# ---------------------------------------------------------------------------


class TestFileOverviewBreadthFloor:
    def test_dotted_grep_hint_thin_match_widens(self, tmp_path: Path) -> None:
        """`self.db` is promoted to a file hint by the dot test; 2 symbols is not an answer."""
        retriever = _build_retriever(str(tmp_path))
        _mock_direct_lookup(retriever, file_ids=[1], symbols_per_file=2)

        standard = [_make_search_result(idx=i, score=0.4, source="bm25") for i in (7, 8, 9, 10)]

        with (
            patch.object(retriever, "_standard_candidates", return_value=standard) as mock_std,
            patch.object(retriever, "_assemble", return_value=_make_retrieved_context()) as asm,
        ):
            retriever._retrieve_file_overview(_file_overview_plan("self.db"))

        mock_std.assert_called_once()
        assert len(asm.call_args[0][1]) == 6

    def test_real_single_file_overview_does_not_widen(self, tmp_path: Path) -> None:
        """A genuine 1-file / many-symbol overview must not pay for a rerank round-trip."""
        retriever = _build_retriever(str(tmp_path))
        _mock_direct_lookup(retriever, file_ids=[1], symbols_per_file=40)

        with (
            patch.object(retriever, "_standard_candidates") as mock_std,
            patch.object(retriever, "_assemble", return_value=_make_retrieved_context()),
        ):
            retriever._retrieve_file_overview(_file_overview_plan("retriever.py"))

        mock_std.assert_not_called()


# ---------------------------------------------------------------------------
# project_overview
# ---------------------------------------------------------------------------


class TestProjectOverviewBreadthFloor:
    def test_thin_overview_widens(self, tmp_path: Path) -> None:
        retriever = _build_retriever(str(tmp_path))
        sym, file, chunk = _make_symbol(1, 1), _make_file(1, "README.md"), _make_chunk(1)
        retriever.db.get_module_and_readme_symbols.return_value = [1]
        retriever.db.get_symbol_with_file.return_value = (sym, file)
        retriever.db.get_first_chunk_for_symbol.return_value = chunk

        standard = [_make_search_result(idx=i, score=0.3, source="vector") for i in (7, 8)]
        plan = _file_overview_plan("x.py", query="what does this project do")
        plan = QueryPlan(
            intent=IntentType.PROJECT_OVERVIEW,
            execution_mode="sequential",
            strategy=INTENT_STRATEGIES[IntentType.PROJECT_OVERVIEW],
            sub_queries=plan.sub_queries,
            raw_query="what does this project do",
        )

        with (
            patch.object(retriever, "_standard_candidates", return_value=standard) as mock_std,
            patch.object(retriever, "_assemble", return_value=_make_retrieved_context()) as asm,
        ):
            retriever._retrieve_project_overview(plan)

        mock_std.assert_called_once()
        assert len(asm.call_args[0][1]) == 3


# ---------------------------------------------------------------------------
# Trace visibility — the lead measures the floor from this section alone
# ---------------------------------------------------------------------------


class TestBreadthFloorTrace:
    def _trace_of(self, retriever, plan, *, standard_candidates) -> dict:
        captured: dict[str, dict] = {}

        def _capture(section: str, data: dict) -> None:
            captured[section] = data

        with (
            patch.object(retriever, "_trace", side_effect=_capture),
            patch.object(retriever, "_standard_candidates", return_value=standard_candidates),
            patch.object(retriever, "_assemble", return_value=_make_retrieved_context()),
        ):
            retriever._retrieve_config(plan)
        return captured

    def test_trace_records_fired_decision_with_counts(self, tmp_path: Path) -> None:
        retriever = _build_retriever(str(tmp_path))
        _mock_direct_lookup(retriever, file_ids=[1], symbols_per_file=2)
        standard = [_make_search_result(idx=i, score=0.5) for i in (7, 8, 9)]

        floor = self._trace_of(retriever, _config_plan(), standard_candidates=standard)[
            "breadth_floor"
        ]

        assert floor["fired"] is True
        assert floor["path"] == "config_lookup"
        assert floor["direct_files"] == 1
        assert floor["direct_symbols"] == 2
        assert floor["min_files"] == _BREADTH_FLOOR_MIN_FILES
        assert floor["min_symbols"] == _BREADTH_FLOOR_MIN_SYMBOLS
        assert floor["standard_candidates"] == 3
        assert floor["merged_symbols"] == 5
        # _make_search_result reuses one rel_path for every idx, so the three standard
        # candidates contribute a single distinct file on top of the direct one.
        assert floor["merged_files"] == 2

    def test_trace_records_not_fired_decision(self, tmp_path: Path) -> None:
        """A trace must distinguish "floor passed" from "floor never ran"."""
        retriever = _build_retriever(str(tmp_path))
        _mock_direct_lookup(
            retriever, file_ids=[1], symbols_per_file=_BREADTH_FLOOR_MIN_SYMBOLS + 1
        )

        floor = self._trace_of(retriever, _config_plan(), standard_candidates=[])["breadth_floor"]

        assert floor["fired"] is False
        assert floor["enabled"] is True
        assert "standard_candidates" not in floor

    def test_file_overview_trace_names_promoted_grep_hints(self, tmp_path: Path) -> None:
        """A thin file_overview must be traceable to the loose dot promotion rule."""
        retriever = _build_retriever(str(tmp_path))
        _mock_direct_lookup(retriever, file_ids=[1], symbols_per_file=2)
        captured: dict[str, dict] = {}

        with (
            patch.object(retriever, "_trace", side_effect=lambda s, d: captured.setdefault(s, d)),
            patch.object(retriever, "_standard_candidates", return_value=[]),
            patch.object(retriever, "_assemble", return_value=_make_retrieved_context()),
        ):
            retriever._retrieve_file_overview(_file_overview_plan("self.db"))

        assert captured["file_overview"]["promoted_grep_hints"] == ["self.db"]
