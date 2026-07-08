"""trelix evaluation harness — CoIR-compatible nDCG@10 + Recall@10 + MRR."""

from trelix.eval.synthesis import (
    SynthesisEvalHarness,
    SynthesisResult,
    evaluate_synthesis,
)

__all__ = ["SynthesisEvalHarness", "SynthesisResult", "evaluate_synthesis"]
