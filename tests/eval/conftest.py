"""
Eval test configuration.

The env-isolation table lives in ``tests/_env_isolation.py`` and is shared with
the unit and integration suites — see that module for why each variable is
pinned the way it is.

This is load-bearing for context-compression A/B evals: the harness scores
faithfulness/completeness/hallucination on the assembled context, so the "off"
baseline must be byte-identical to today. If a dev's ``.env`` flips
``TRELIX_RETRIEVAL_COMPRESSION`` on — or merely retunes
``TRELIX_RETRIEVAL_COMPRESSION_RATIO``, which this file used to miss because it
carried its own partial copy of the table — the baseline silently compresses
differently and the A/B delta is meaningless.
"""

import pytest

from tests._env_isolation import apply_env_isolation


@pytest.fixture(autouse=True)
def _isolate_beast_mode_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    """Override beast-mode feature flags to false so eval runs see code defaults."""
    apply_env_isolation(monkeypatch)
