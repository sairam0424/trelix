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
