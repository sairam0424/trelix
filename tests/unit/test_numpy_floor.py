"""numpy floor-bump regression guard (numpy 1.26 -> 2.0+).

WHY THIS EXISTS. NumPy 2.0.0 broke C-API/ABI compatibility and changed NEP 50's
type-promotion rules for mixed-dtype arithmetic. The only direct numpy usage in
src/trelix is src/trelix/compression/extractive.py, and every array-constructing call
there passes an explicit `dtype=np.float64` (never left to type inference), so NEP 50's
mixed-dtype changes don't apply. This test pins the actual numeric output of that file's
cosine-similarity helper against an independently-computed (non-numpy) expected value, so
a future numpy upgrade that somehow changed float64 arithmetic semantics would be caught
here rather than silently shifting retrieval/compression scores.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from trelix.compression.extractive import _cosine


def test_cosine_matches_independently_computed_value() -> None:
    """_cosine's float64 dot-product/norm path matches hand-computed cosine similarity."""
    q = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    vec = np.asarray([1.0, 1.0, 0.0], dtype=np.float64)
    q_norm = float(np.linalg.norm(q))

    result = _cosine(q, q_norm, vec)

    expected = 1.0 / math.sqrt(2.0)  # dot=1.0, |q|=1.0, |vec|=sqrt(2) -- computed without numpy
    assert result == pytest.approx(expected, rel=1e-12)


def test_cosine_returns_zero_for_mismatched_shapes() -> None:
    """Shape-mismatched vectors return 0.0 rather than raising a numpy broadcasting error."""
    q = np.asarray([1.0, 0.0], dtype=np.float64)
    vec = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    assert _cosine(q, 1.0, vec) == 0.0


def test_cosine_returns_zero_for_zero_vector() -> None:
    """A zero-norm candidate vector returns 0.0 rather than a division-by-zero NaN/inf."""
    q = np.asarray([1.0, 0.0], dtype=np.float64)
    vec = np.asarray([0.0, 0.0], dtype=np.float64)
    assert _cosine(q, 1.0, vec) == 0.0
