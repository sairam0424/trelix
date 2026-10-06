"""The eval-compare judge: the `note:` lines for differences that are reported, not refused.

MUTATIONS THAT MUST MAKE THIS FILE FAIL
---------------------------------------
1. the `trelix_version`, `embedder` or `pipeline.config` note dropped  (the TestNotes cases)
2. a config leaf compared after clipping, or by Python equality (`True == 1`)  (the clip and
   the kind-of-leaf tests)
3. the 20-leaf cap removed                           (test_at_most_twenty_leaves_are_listed)
4. an empty root config read as a leaf               (the empty nested object test)
5. a note about `created_at`                         (test_the_timestamps_are_shown_and_never_noted)
6. `sort_keys=True` dropped from the rendering of a leaf  (test_a_list_leaf_is_compared_...)
"""

from __future__ import annotations

from tests.unit.eval_compare_fixtures import (
    base_doc,
    cand_doc,
    pipeline_doc,
    verdict_of,
)


class TestNotes:
    def test_a_different_trelix_version_is_a_note_not_a_refusal(self) -> None:
        verdict = verdict_of(base_doc(), cand_doc(0.5625, trelix_version="3.4.4"))

        assert verdict.outcome == "PASS"
        assert "note: trelix_version differs: base 3.4.3, cand 3.4.4" in verdict.lines

    def test_a_different_embedder_is_a_note_per_field(self) -> None:
        embedder = {
            "provider": "local",
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "dimension": 768,
            "library_version": "5.1.0",
        }

        verdict = verdict_of(base_doc(), cand_doc(0.5625, embedder=embedder))

        assert verdict.outcome == "PASS"
        assert "note: embedder differs: dimension (base 384, cand 768)" in verdict.lines
        assert 'note: embedder differs: library_version (base null, cand "5.1.0")' in verdict.lines
        assert sum(1 for line in verdict.lines if line.startswith("note: embedder")) == 2

    def test_a_different_config_is_a_note_per_leaf_in_path_order(self) -> None:
        config = {"retrieval": {"top_k": 10, "declaration_boost_enabled": True, "extra": 5}}
        pipeline = {**pipeline_doc(), "config": config}

        verdict = verdict_of(base_doc(), cand_doc(0.5625, pipeline=pipeline))

        notes = [line for line in verdict.lines if line.startswith("note:")]
        assert notes == [
            "note: pipeline.config differs: retrieval.declaration_boost_enabled "
            "(base false, cand true)",
            "note: pipeline.config differs: retrieval.extra (base absent, cand 5)",
        ]

    def test_an_empty_nested_object_is_a_leaf_and_an_empty_config_has_none(self) -> None:
        base_pipeline = {**pipeline_doc(), "config": {"a": {}}}
        cand_pipeline = {**pipeline_doc(), "config": {"a": {"b": 1}}}

        verdict = verdict_of(
            base_doc(pipeline=base_pipeline), cand_doc(0.5625, pipeline=cand_pipeline)
        )

        notes = [line for line in verdict.lines if line.startswith("note:")]
        assert notes == [
            "note: pipeline.config differs: a (base {}, cand absent)",
            "note: pipeline.config differs: a.b (base absent, cand 1)",
        ]

    def test_a_value_that_is_the_word_absent_is_not_an_absent_value(self) -> None:
        base_pipeline = {**pipeline_doc(), "config": {"a": "absent"}}
        cand_pipeline = {**pipeline_doc(), "config": {}}

        verdict = verdict_of(
            base_doc(pipeline=base_pipeline), cand_doc(0.5625, pipeline=cand_pipeline)
        )

        assert 'note: pipeline.config differs: a (base "absent", cand absent)' in verdict.lines

    def test_identical_configs_say_so_once(self) -> None:
        verdict = verdict_of(base_doc(), cand_doc(0.5625))

        notes = [line for line in verdict.lines if line.startswith("note:")]
        assert notes == ["note: pipeline.config identical"]

    def test_a_value_differing_only_past_the_clip_is_still_a_difference(self) -> None:
        long_value = "x" * 80
        base_pipeline = {**pipeline_doc(), "config": {"a": long_value + "1"}}
        cand_pipeline = {**pipeline_doc(), "config": {"a": long_value + "2"}}

        verdict = verdict_of(
            base_doc(pipeline=base_pipeline), cand_doc(0.5625, pipeline=cand_pipeline)
        )

        notes = [line for line in verdict.lines if line.startswith("note:")]
        assert len(notes) == 1
        assert notes[0].startswith("note: pipeline.config differs: a (base ")

    def test_a_config_leaf_of_one_kind_differs_from_another(self) -> None:
        # json renders true and 1 differently; python says True == 1.
        base_pipeline = {**pipeline_doc(), "config": {"a": True}}
        cand_pipeline = {**pipeline_doc(), "config": {"a": 1}}

        verdict = verdict_of(
            base_doc(pipeline=base_pipeline), cand_doc(0.5625, pipeline=cand_pipeline)
        )

        assert "note: pipeline.config differs: a (base true, cand 1)" in verdict.lines

    def test_a_list_leaf_is_compared_and_shown_with_its_keys_in_sorted_order(self) -> None:
        # A list is a leaf. Its mappings are the same whatever order their keys were written in.
        base_config = {"rules": [{"b": 2, "a": 1}], "same": [{"x": 1, "y": 2}]}
        cand_config = {"rules": [{"b": 3, "a": 1}], "same": [{"y": 2, "x": 1}]}

        verdict = verdict_of(
            base_doc(pipeline={**pipeline_doc(), "config": base_config}),
            cand_doc(0.5625, pipeline={**pipeline_doc(), "config": cand_config}),
        )

        notes = [line for line in verdict.lines if line.startswith("note:")]
        assert notes == [
            "note: pipeline.config differs: "
            'rules (base [{"a": 1, "b": 2}], cand [{"a": 1, "b": 3}])'
        ]

    def test_at_most_twenty_leaves_are_listed(self) -> None:
        many = {f"k{number:02d}": number for number in range(25)}
        pipeline = {**pipeline_doc(), "config": many}

        verdict = verdict_of(
            base_doc(pipeline={**pipeline_doc(), "config": {}}), cand_doc(0.5625, pipeline=pipeline)
        )

        notes = [line for line in verdict.lines if line.startswith("note:")]
        assert len(notes) == 21
        assert notes[0] == "note: pipeline.config differs: k00 (base absent, cand 0)"
        assert notes[-1] == "note: pipeline.config differs in 5 more values"

    def test_the_timestamps_are_shown_and_never_noted(self) -> None:
        verdict = verdict_of(
            base_doc(created_at="2026-10-05T12:00:00+00:00"),
            cand_doc(0.5625, created_at="2026-10-05T12:30:00+00:00"),
        )

        assert (
            "runs: base 2026-10-05T12:00:00+00:00 cand 2026-10-05T12:30:00+00:00" in verdict.lines
        )
        assert not any("created_at" in line for line in verdict.lines)
        assert [line for line in verdict.lines if line.startswith("note:")] == [
            "note: pipeline.config identical"
        ]

    def test_a_different_timestamp_does_not_change_the_verdict(self) -> None:
        first = verdict_of(base_doc(), cand_doc(0.5625))
        later = verdict_of(base_doc(), cand_doc(0.5625, created_at="2027-01-01T00:00:00+00:00"))

        assert first.outcome == later.outcome == "PASS"
        assert first.reasons == later.reasons
