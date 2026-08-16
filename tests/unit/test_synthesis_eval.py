"""Tests for the synthesis quality evaluation harness."""

from __future__ import annotations


class TestSynthesisScoring:
    def test_hallucination_zero_when_all_symbols_present(self):
        from trelix.eval.synthesis import score_hallucination

        answer = "The AuthMiddleware.verify method calls UserRepository.get_by_token."
        retrieved = ["AuthMiddleware.verify", "UserRepository.get_by_token"]
        expected = ["AuthMiddleware.verify", "UserRepository.get_by_token"]
        assert score_hallucination(answer, retrieved, expected) == 0.0

    def test_hallucination_one_when_symbol_not_in_retrieved(self):
        from trelix.eval.synthesis import score_hallucination

        # answer mentions FakeClass.method which was NOT retrieved
        answer = "The FakeClass.method handles this."
        retrieved = ["AuthMiddleware.verify"]
        expected = ["FakeClass.method"]
        assert score_hallucination(answer, retrieved, expected) == 1.0

    def test_hallucination_partial(self):
        from trelix.eval.synthesis import score_hallucination

        answer = "AuthMiddleware.verify calls FakeClass.method."
        retrieved = ["AuthMiddleware.verify"]  # FakeClass.method not retrieved
        expected = ["AuthMiddleware.verify", "FakeClass.method"]
        score = score_hallucination(answer, retrieved, expected)
        assert 0.0 < score < 1.0

    def test_completeness_all_fragments_present(self):
        from trelix.eval.synthesis import score_completeness

        answer = "The jwt token is validated by the middleware layer."
        fragments = ["jwt", "validated", "middleware"]
        assert score_completeness(answer, fragments) == 1.0

    def test_completeness_no_fragments_present(self):
        from trelix.eval.synthesis import score_completeness

        answer = "This handles authentication."
        fragments = ["jwt", "token", "bearer"]
        assert score_completeness(answer, fragments) == 0.0

    def test_completeness_partial(self):
        from trelix.eval.synthesis import score_completeness

        answer = "The jwt middleware validates the request."
        fragments = ["jwt", "missing_fragment"]
        score = score_completeness(answer, fragments)
        assert score == 0.5  # 1 of 2 present

    def test_completeness_empty_fragments_returns_one(self):
        from trelix.eval.synthesis import score_completeness

        assert score_completeness("any answer", []) == 1.0

    def test_faithfulness_answer_references_context(self):
        from trelix.eval.synthesis import score_faithfulness

        context = "def verify(token): return jwt.decode(token, SECRET)"
        answer = "The verify function decodes the jwt token using SECRET."
        score = score_faithfulness(answer, context)
        assert score > 0.0

    def test_faithfulness_empty_answer(self):
        from trelix.eval.synthesis import score_faithfulness

        assert score_faithfulness("", "some context") == 0.0


class TestSynthesisResult:
    def test_synthesis_result_construction(self):
        from trelix.eval.synthesis import SynthesisResult

        r = SynthesisResult(
            query="how does auth work",
            answer="The AuthMiddleware.verify validates tokens.",
            retrieved_symbols=["AuthMiddleware.verify"],
            expected_symbols=["AuthMiddleware.verify"],
            expected_fragments=["verify", "token"],
            hallucinated_symbols=[],
            missing_fragments=[],
            scores={"hallucination": 0.0, "completeness": 1.0, "faithfulness": 0.8},
        )
        assert r.scores["hallucination"] == 0.0
        assert r.scores["completeness"] == 1.0

    def test_synthesis_result_defaults(self):
        from trelix.eval.synthesis import SynthesisResult

        r = SynthesisResult(
            query="test query",
            answer="some answer",
            retrieved_symbols=[],
            expected_symbols=[],
            expected_fragments=[],
            hallucinated_symbols=[],
            missing_fragments=[],
        )
        assert r.scores == {}
        assert r.query == "test query"

    def test_evaluate_synthesis_populates_all_scores(self):
        from trelix.eval.synthesis import evaluate_synthesis

        result = evaluate_synthesis(
            query="how does auth work",
            answer="The AuthMiddleware.verify decodes jwt tokens.",
            retrieved_context="def verify(token): return jwt.decode(token, SECRET)",
            retrieved_symbols=["AuthMiddleware.verify"],
            expected_symbols=["AuthMiddleware.verify"],
            expected_fragments=["jwt", "verify"],
        )
        assert "hallucination" in result.scores
        assert "completeness" in result.scores
        assert "faithfulness" in result.scores
        assert "overall" in result.scores
        assert result.scores["hallucination"] == 0.0
        assert result.scores["completeness"] == 1.0


class TestSynthesisEvalHarness:
    def _make_golden(self, tmp_path, entries):
        import json

        golden = tmp_path / "golden_synthesis.jsonl"
        with open(golden, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")
        return str(golden)

    def test_harness_returns_required_keys(self, tmp_path):
        from unittest.mock import MagicMock, patch

        from trelix.core.config import IndexConfig
        from trelix.eval.synthesis import SynthesisEvalHarness

        golden = self._make_golden(
            tmp_path,
            [
                {
                    "query": "how does auth work",
                    "relevant_files": ["src/auth.py"],
                    "expected_answer_fragments": ["jwt"],
                    "expected_symbols": ["AuthMiddleware.verify"],
                }
            ],
        )

        config = IndexConfig(repo_path=str(tmp_path))
        harness = SynthesisEvalHarness.__new__(SynthesisEvalHarness)
        harness._config = config

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = MagicMock(
            context_text="def verify(token): return jwt.decode(token)",
            results=[MagicMock(symbol=MagicMock(qualified_name="AuthMiddleware.verify"))],
        )
        harness._retriever = mock_retriever

        with patch("trelix.eval.synthesis.Synthesizer") as MockSynth:
            answer = "The AuthMiddleware.verify decodes jwt tokens."
            MockSynth.return_value.synthesize.return_value = answer
            metrics = harness.run(golden)

        assert "hallucination_rate" in metrics
        assert "completeness" in metrics
        assert "faithfulness" in metrics
        assert "overall" in metrics
        assert "n_queries" in metrics
        assert metrics["n_queries"] == 1.0

    def test_empty_golden_returns_zeros(self, tmp_path):
        from trelix.core.config import IndexConfig
        from trelix.eval.synthesis import SynthesisEvalHarness

        golden = self._make_golden(tmp_path, [])
        config = IndexConfig(repo_path=str(tmp_path))
        harness = SynthesisEvalHarness.__new__(SynthesisEvalHarness)
        harness._config = config
        harness._retriever = None

        metrics = harness.run(golden)
        assert metrics["n_queries"] == 0.0
        assert metrics["overall"] == 0.0


class TestHarnessConstructsSynthesizerCorrectly:
    """The harness must build a Synthesizer the way every other call site does.

    `SynthesisEvalHarness.run()` called `Synthesizer(self._config)` while `self._config`
    is the `IndexConfig` the CLI passes. `Synthesizer.__init__` takes an
    `EmbedderConfig` first and, with `llm_config=None`, falls into a shim that reads
    `config.provider` — so it raised `AttributeError: 'IndexConfig' object has no
    attribute 'provider'`, which a bare `except Exception` turned into `answer = ""`.

    Every query therefore scored against an empty answer. Measured before the fix on
    eval/golden_synthesis_sample.jsonl: hallucination 0.0000, completeness 0.0000,
    faithfulness 0.0000, overall a constant 0.4000. `trelix eval-synthesis` could not
    produce a non-empty answer for any input.

    Two safety nets should have caught it and both were defeated:
      - `__init__(self, config: Any)` — the `Any` annotation stopped mypy --strict from
        type-checking this call.
      - the pre-existing test patches `Synthesizer` wholesale, and a MagicMock accepts
        any constructor arguments, so the wrong type never raised under test.
    That is why these tests assert on the constructor ARGUMENTS rather than on the shape
    of the returned metrics dict.
    """

    @staticmethod
    def _golden(tmp_path, n=1):  # type: ignore[no-untyped-def]
        import json

        path = tmp_path / "golden.jsonl"
        with open(path, "w") as f:
            for i in range(n):
                f.write(
                    json.dumps(
                        {
                            "query": f"query number {i}",
                            "relevant_files": ["src/x.py"],
                            "expected_answer_fragments": ["alpha"],
                            "expected_symbols": ["Thing.method"],
                        }
                    )
                    + "\n"
                )
        return str(path)

    @staticmethod
    def _harness(tmp_path):  # type: ignore[no-untyped-def]
        from unittest.mock import MagicMock

        from trelix.core.config import IndexConfig
        from trelix.eval.synthesis import SynthesisEvalHarness

        harness = SynthesisEvalHarness.__new__(SynthesisEvalHarness)
        harness._config = IndexConfig(repo_path=str(tmp_path))
        retriever = MagicMock()
        retriever.retrieve.return_value = MagicMock(
            context_text="def method(): return 'alpha'",
            results=[MagicMock(symbol=MagicMock(qualified_name="Thing.method"))],
        )
        harness._retriever = retriever
        return harness

    def test_synthesizer_receives_an_embedder_config(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """The first positional argument must be the EmbedderConfig, not the IndexConfig."""
        from unittest.mock import patch

        from trelix.core.config import EmbedderConfig

        harness = self._harness(tmp_path)
        with patch("trelix.eval.synthesis.Synthesizer") as MockSynth:
            MockSynth.return_value.synthesize.return_value = "alpha appears here"
            harness.run(self._golden(tmp_path))

        assert MockSynth.call_args is not None, "Synthesizer was never constructed"
        first_arg = MockSynth.call_args.args[0]
        assert isinstance(first_arg, EmbedderConfig), (
            f"Synthesizer was given a {type(first_arg).__name__}; it takes an "
            "EmbedderConfig, and anything else raises AttributeError inside the shim"
        )

    def test_synthesizer_receives_the_llm_config(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """Passing llm_config avoids the shim entirely, as `trelix ask` does."""
        from unittest.mock import patch

        from trelix.core.config import LLMConfig

        harness = self._harness(tmp_path)
        with patch("trelix.eval.synthesis.Synthesizer") as MockSynth:
            MockSynth.return_value.synthesize.return_value = "alpha"
            harness.run(self._golden(tmp_path))

        assert isinstance(MockSynth.call_args.kwargs.get("llm_config"), LLMConfig), (
            "llm_config was not passed, so the Synthesizer falls back to rebuilding one "
            "from the embedder config"
        )

    def test_synthesizer_is_built_once_not_per_query(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """Construction was inside the per-entry loop, re-initialising an LLM client
        for every query in the golden file."""
        from unittest.mock import patch

        harness = self._harness(tmp_path)
        with patch("trelix.eval.synthesis.Synthesizer") as MockSynth:
            MockSynth.return_value.synthesize.return_value = "alpha"
            harness.run(self._golden(tmp_path, n=5))

        assert MockSynth.call_count == 1, (
            f"Synthesizer was constructed {MockSynth.call_count} times for 5 queries"
        )

    def test_the_real_constructor_accepts_what_the_harness_passes(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """Integration guard: build a real Synthesizer the way the harness now does.

        The mocked tests above cannot catch a type error, because a MagicMock accepts
        anything. This one uses the real class, with the local provider so no network or
        credentials are involved.
        """
        from trelix.core.config import IndexConfig
        from trelix.retrieval.synthesizer import Synthesizer

        config = IndexConfig(repo_path=str(tmp_path))
        config.embedder.provider = "local"

        synth = Synthesizer(
            config.embedder,
            retrieval_config=config.retrieval,
            llm_config=config.llm,
        )
        assert synth is not None

    def test_answers_are_not_silently_empty(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """A working synthesizer must produce scores that reflect the answer.

        With `answer = ""` for every query, completeness and faithfulness are both 0.0
        and overall collapses to a constant. Non-zero completeness is the signal that a
        real answer reached the scorer.
        """
        from unittest.mock import patch

        harness = self._harness(tmp_path)
        with patch("trelix.eval.synthesis.Synthesizer") as MockSynth:
            MockSynth.return_value.synthesize.return_value = (
                "Thing.method returns alpha, as defined in the retrieved context."
            )
            metrics = harness.run(self._golden(tmp_path))

        assert metrics["completeness"] > 0.0, (
            f"expected the 'alpha' fragment to be found in the answer; metrics={metrics}"
        )
