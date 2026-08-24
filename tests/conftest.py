"""Marker taxonomy for the whole suite, applied BY PATH.

Why this file exists rather than 3,969 hand-written decorators
-------------------------------------------------------------
This repo has already lost the "registered marker" bet once. ``integration`` was
registered in ``pyproject.toml``, documented in CONTRIBUTING.md as *the*
credential-free run, and carried by NO TEST -- so ``pytest -m "not integration"``
deselected nothing while driving live Azure/Bedrock calls. The fix was to apply
it by directory in ``tests/integration/conftest.py``. This file is that same
mechanism generalised: one hook, one table per marker, and a new file in a
covered path is tagged on arrival instead of the day someone remembers to
decorate it.

``--strict-markers`` (see ``addopts``) only validates ``@pytest.mark.<name>``.
It does NOT validate ``-m`` expressions: on this tree
``pytest tests/unit/test_multi_watcher.py -m "not integrationn"`` collected all
8 tests and exited 0. Nothing pytest ships can catch that typo, so the guard is
``tests/unit/test_marker_taxonomy.py``, which pins that every marker below both
selects everything in a known carrier and deselects it -- and pins the typo
behaviour as its own control, so the assertions cannot pass vacuously.

Granularity, stated plainly
---------------------------
Every rule below is FILE-level except ``requires_weights``, which is node-level
because the two files involved also contain the portable fake-model tests that
must always run. A file-level marker therefore tags fast tests that happen to
live in a heavy file; that is the price of not hand-decorating, and it is why
``slow`` means "this FILE costs >= 4.0s" rather than "this test is slow".

No counts are written into this file. They live in
``tests/unit/test_marker_taxonomy.py`` where drift FAILS a run instead of merely
misinforming a reader -- the mistake an in-code "104 deselected" comment already
made once (the number measures 89 today).
"""

from __future__ import annotations

from pathlib import Path

import pytest

_TESTS_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# slow -- measured file-level cost, not a guess.
#
# Measured on this tree with `pytest tests/unit --durations=0 --durations-min=0`,
# summing call+setup+teardown per file: 199.3s over 191 files. These are every file
# at or above 4.0s, ~113s of that total. The threshold is a judgement, the
# memberships are measurements, and both are re-derivable with the command above.
# They are NOT re-measured at collection time: timings are machine-dependent and a
# self-measuring marker would make selection nondeterministic across the four
# Python legs.
#
# test_marker_taxonomy.py is in here by that same rule, applied to itself: it spawns
# ~30 child `pytest --collect-only` processes and measured 22.5s. Consequence stated
# plainly -- the taxonomy's own guard does NOT run in the fast inner loop. It is a
# config guard, so CI's full run is the right place for it; the alternative is
# a table that violates its own documented threshold.
#
# regressions/test_regressions.py is here by the same rule and the same reasoning:
# measured 32.1s and 61.9s over the same 22 tests on two runs of this tree (the spread is
# machine load; both are quoted rather than the flattering one), of which the four child
# runs cost 8.74s, 7.36s, 6.83s and 6.35s on the faster one. It creates one throwaway
# `git worktree` and spawns one child pytest per manifest entry, so its cost is structural
# rather than incidental, and no plausible machine puts it under 4.0s. It is the only file
# in this table outside tests/unit, which is why the key is relative to tests/ rather than a
# basename -- see _relative_key below.
# ---------------------------------------------------------------------------
SLOW_FILES = frozenset(
    {
        "regressions/test_regressions.py",
        "unit/test_audit_wipe_detection.py",
        "unit/test_cli_closed_stdout.py",
        "unit/test_cli_smoke.py",
        "unit/test_cli_watch_all_signals.py",
        "unit/test_connectors.py",
        "unit/test_dimension_guard.py",
        "unit/test_dotenv_anchoring.py",
        "unit/test_dry_run.py",
        "unit/test_embedder.py",
        "unit/test_eval_harness.py",
        "unit/test_indexer_vector_repair.py",
        "unit/test_marker_taxonomy.py",
        "unit/test_otel_metrics.py",
        "unit/test_otel_metrics_reentry.py",
        "unit/test_phase25_concurrency.py",
        "unit/test_retriever_core.py",
        "unit/test_vector_coverage.py",
    }
)

# ---------------------------------------------------------------------------
# requires_extra -- a MECHANICAL criterion, not taste: these are exactly the
# files carrying a MODULE-SCOPE `pytest.importorskip(...)`, i.e. files that skip
# in their entirety when an optional extra is absent. The meta-test re-derives
# this set from the sources and compares both ways, so adding a module-scope
# importorskip without updating this table fails the suite.
#
# Extras involved, measured: pyvis (escaping, 1 file), watchfiles (2), litellm
# (1), lancedb (2). Per-TEST importorskip calls -- test_vector_coverage.py,
# test_embedder.py, test_retry.py and friends -- deliberately do NOT land here:
# those files still have work to do without the extra.
# ---------------------------------------------------------------------------
REQUIRES_EXTRA_FILES = frozenset(
    {
        "unit/test_graph_visualizer_escaping.py",
        "unit/test_multi_watcher.py",
        "unit/test_multi_watcher_filtering.py",
        "unit/test_operator_env_leak.py",
        # Landed in the same round as this taxonomy, by a different agent that never saw
        # it: module-scope pytest.importorskip("litellm"). The guard below caught the
        # omission the moment both files met in one tree, which is the drift alarm working
        # rather than a false positive -- but it means a new module-scope importorskip has
        # to be declared here in the SAME commit that introduces it.
        "unit/test_subprocess_operator_env_inheritance.py",
        "unit/test_vector_lance_concurrency.py",
        "unit/test_vector_lance_upsert.py",
    }
)

# ---------------------------------------------------------------------------
# requires_weights -- node-level. These are the only tests that construct a real
# published model from cached weights; everything else around them uses fakes and
# MUST keep running on a runner with no HF cache. Marking their files would
# deselect the portable proofs, which is the opposite of the point.
#
# Each already guards itself (skipif on the snapshot dir / importorskip on
# FlagEmbedding), so the marker adds selectability, not protection.
# ---------------------------------------------------------------------------
REQUIRES_WEIGHTS_NODES = frozenset(
    {
        (
            "unit/test_embedder_bge.py",
            "test_constructed_class_pools_the_way_the_model_was_published",
        ),
        (
            "unit/test_embedder_bge.py",
            "test_two_queries_differing_after_token_0_get_different_embeddings",
        ),
        (
            "unit/test_sparse_padding_contamination.py",
            "test_real_weights_agree_alone_and_batched",
        ),
    }
)

# ---------------------------------------------------------------------------
# security -- files whose SUBJECT is a security control: authn/authz, path
# containment, output escaping, secret redaction, audit-log tamper evidence,
# read-only enforcement, remote-code gating, and the socket ban's own control.
# An explicit table rather than a name pattern, because there is no naming
# convention to lean on here and a pattern would silently under-tag.
# ---------------------------------------------------------------------------
SECURITY_FILES = frozenset(
    {
        "unit/test_api_audit.py",
        "unit/test_api_containment.py",
        "unit/test_audit_hash_chain_columns.py",
        "unit/test_audit_read_hardening.py",
        "unit/test_audit_undetectable.py",
        "unit/test_audit_verify_snapshot.py",
        "unit/test_audit_wipe_detection.py",
        "unit/test_cli_audit_read_only.py",
        "unit/test_cli_markup_safety.py",
        "unit/test_cli_serve_exposure_warning.py",
        "unit/test_config_error_redaction.py",
        "unit/test_defuse.py",
        "unit/test_dotenv_anchoring.py",
        "unit/test_graph_visualizer_escaping.py",
        "unit/test_network_is_blocked.py",
        "unit/test_oidc.py",
        "unit/test_oidc_jwks_size_cap.py",
        "unit/test_operator_env_leak.py",
        "unit/test_remote_model_code_gate.py",
        "unit/test_store_read_only.py",
        "unit/test_taint.py",
        "unit/test_walker_containment.py",
    }
)

# ---------------------------------------------------------------------------
# Prefix rules, used only where the repo already has a naming convention, so a
# new file is tagged on arrival. `cli` also picks up tests/integration/test_cli.py
# by the same prefix, which is correct: it drives the installed console script.
# ---------------------------------------------------------------------------
CLI_PREFIX = "test_cli"
API_PREFIX = "test_api"
PARSER_PREFIX = "test_parser_"

# Members of the same layer that the prefix cannot reach.
API_EXTRA_FILES = frozenset(
    {
        "unit/test_graph_api.py",
        "integration/test_graph_api_integration.py",
    }
)
PARSER_EXTRA_FILES = frozenset(
    {
        "unit/test_detect_language_contract.py",
        "unit/test_diff_parser.py",
        "unit/test_line_window_parser.py",
        "unit/test_line_window_parser_bounds.py",
    }
)


def _relative_key(item: pytest.Item) -> str:
    """``tests/unit/test_x.py`` -> ``unit/test_x.py``; unrelated paths -> ``""``.

    Keyed on the path RELATIVE TO ``tests/``, never on the bare basename: four
    basenames are duplicated across ``tests/unit`` and ``tests/integration``
    (``test_connectors.py`` among them, and it is in ``SLOW_FILES``), so a
    basename key would tag the integration twin with a cost never measured there.
    """
    try:
        return item.path.relative_to(_TESTS_DIR).as_posix()
    except ValueError:
        return ""


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Apply the taxonomy above to every collected item.

    ``pytest_collection_modifyitems`` in this conftest is handed the whole
    session's items, including ``tests/integration``; ``integration``,
    ``enable_socket`` and ``requires_network`` are applied by
    ``tests/integration/conftest.py`` and are deliberately NOT repeated here.
    """
    for item in items:
        key = _relative_key(item)
        if not key:
            continue
        basename = key.rsplit("/", 1)[-1]

        if key in SLOW_FILES:
            item.add_marker(pytest.mark.slow)
        if key in REQUIRES_EXTRA_FILES:
            item.add_marker(pytest.mark.requires_extra)
        if (key, item.name) in REQUIRES_WEIGHTS_NODES:
            item.add_marker(pytest.mark.requires_weights)
        if key in SECURITY_FILES:
            item.add_marker(pytest.mark.security)
        if basename.startswith(CLI_PREFIX):
            item.add_marker(pytest.mark.cli)
        if basename.startswith(API_PREFIX) or key in API_EXTRA_FILES:
            item.add_marker(pytest.mark.api)
        if basename.startswith(PARSER_PREFIX) or key in PARSER_EXTRA_FILES:
            item.add_marker(pytest.mark.parser)
