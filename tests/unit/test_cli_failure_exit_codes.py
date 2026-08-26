"""Failure paths of `trelix` commands must exit NON-ZERO.

A CLI that prints an error and exits 0 is silently broken: `trelix ... && next-step`
runs `next-step`, and a CI job goes green on a step that did nothing. `src/trelix/cli/
main.py` raises `typer.Exit(1)` in ~90 places, so the intent is unambiguous — what is
missing is coverage that would notice a `raise` turning into a `return`.

Three commands are pinned here because no unit test drove any of them before, so a
mutation on their exit is invisible to the suite:

  * `update-index` — its two `except ... raise typer.Exit(1)` config handlers.
  * `link-artifacts` — the "No index found" guard.
  * `connector sync` — the post-sync `if result.errors: raise typer.Exit(1)`.

`tests/integration/test_cli.py` covers only the SUCCESS side of `update-index`
(`test_update_index_exits_zero`, `test_update_index_returns_json`) and needs a real
index; everything here runs through `CliRunner` against a tmp fixture and never
indexes a real repository.

The fourth test documents a defect rather than pinning behaviour — see its docstring.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from trelix.cli.main import app

runner = CliRunner()


def _flat(output: str) -> str:
    """Rich hard-wraps at console width, so a message can be split mid-sentence.

    Collapsing all whitespace is what makes a substring assertion about a message
    stable; asserting on the wrapped form pins the terminal width instead.
    """
    return re.sub(r"\s+", " ", output)


def _repo_with_empty_index(tmp_path: Path) -> Path:
    """A repo whose `.trelix/index.db` exists but holds no files.

    Built with `Database` directly rather than by running `trelix index`: indexing
    needs an embedder, and none of the paths under test read a single row. The
    per-test assertions below therefore state which side of the guard the fixture
    is meant to be on, so the fixture drifting stops the test instead of the test
    quietly passing for a new reason.
    """
    from trelix.core.config import IndexConfig
    from trelix.store.db import Database

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "calc.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    config = IndexConfig(repo_path=str(repo.resolve()))
    Database(config.db_path_absolute).close()
    return repo


class _PartiallyFailingSource:
    """A connector whose sync half-succeeded: 3 fetched, 1 written, 2 errors.

    A plain class, not a Mock: a Mock would answer `validate_config()` and `sync()`
    whatever their real signatures are, which is how a provider that called a method
    no release has ever shipped passed its tests. This implements exactly the two
    members `connector_sync` calls, and returns the real `ConnectorSyncResult`.
    """

    def validate_config(self) -> None:
        return None

    def sync(self, db: object, linker: object = None) -> object:
        from trelix.indexing.connectors.base import ConnectorSyncResult

        return ConnectorSyncResult(
            artifacts_fetched=3,
            artifacts_written=1,
            errors=2,
            edges_linked=0,
        )


def test_update_index_rejects_an_unknown_provider_with_a_nonzero_exit(tmp_path: Path) -> None:
    """MUTATION: `raise typer.Exit(1) from exc` -> `return` in `update_index`'s
    `except _PydanticValidationError` handler (main.py's "Configuration error" branch).

    Under that mutation the same message is still printed, so only the exit code
    catches it — which is the point. `--provider nope` is not an embedder provider,
    so `IndexConfig` validation rejects it before anything is read or written.
    """
    repo = _repo_with_empty_index(tmp_path)

    result = runner.invoke(app, ["update-index", str(repo), "calc.py", "--provider", "nope"])

    # Precondition: this must be the config-validation branch, not some later
    # failure that happens to exit 1 too (a `return` mutation leaves `config`
    # unbound, and the resulting NameError also exits 1).
    assert "Configuration error" in _flat(result.output), (
        "expected the IndexConfig validation branch; got:\n" + result.output
    )
    assert result.exit_code == 1, (
        f"a rejected --provider must exit 1, got {result.exit_code}\n{result.output}"
    )


def test_link_artifacts_without_an_index_exits_nonzero(tmp_path: Path) -> None:
    """MUTATION: `raise typer.Exit(1)` -> `return` in `link_artifacts`'s
    `if not db_path.exists():` guard.

    Under that mutation the command prints "No index found ... run `trelix index`
    first" and exits 0, so a script that chains a sync-then-link pipeline treats an
    entirely absent index as a completed link pass.
    """
    repo = tmp_path / "never-indexed"
    repo.mkdir()

    # Precondition naming the fixture: the guard under test only fires when the DB
    # is genuinely absent, so assert that rather than trusting mkdir().
    from trelix.core.config import IndexConfig

    db_path = IndexConfig(repo_path=str(repo.resolve())).db_path_absolute
    assert not db_path.exists(), (
        f"the `never-indexed` fixture must have no index at {db_path}; "
        "with one present this test stops exercising the guard"
    )

    result = runner.invoke(app, ["link-artifacts", str(repo)])

    assert "No index found" in _flat(result.output), result.output
    assert result.exit_code == 1, (
        f"link-artifacts on an unindexed repo must exit 1, got {result.exit_code}\n{result.output}"
    )


def test_connector_sync_exits_nonzero_when_the_sync_reported_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MUTATION: delete `if result.errors: raise typer.Exit(1)` at the end of
    `connector_sync` (or flip it to `if not result.errors:`).

    The existing `test_connector_sync_jira_end_to_end` in
    `tests/unit/test_cli_smoke.py` asserts exit 0 on a sync with ZERO errors, so it
    passes either way; nothing covered the errors>0 side. A Jira sync that failed
    on 2 of 3 tickets exiting 0 means a nightly refresh job reports success while
    the index silently drifts.
    """
    repo = _repo_with_empty_index(tmp_path)
    monkeypatch.setattr(
        "trelix.indexing.connectors.registry.get_artifact_source",
        lambda name: _PartiallyFailingSource(),
    )

    result = runner.invoke(app, ["connector", "sync", str(repo), "jira"])

    # Precondition: the substituted source really ran and its counts reached the
    # summary. Without this the test would also pass if the command had bailed at
    # the "No index found" guard or on a misconfigured real Jira connector —
    # neither of which is the branch under test.
    assert "fetched 3, wrote 1, errors 2" in _flat(result.output), (
        "_PartiallyFailingSource did not reach the summary line; this test is no "
        "longer exercising the post-sync errors check:\n" + result.output
    )
    assert result.exit_code == 1, (
        f"a sync with 2 errors must exit 1, got {result.exit_code}\n{result.output}"
    )


@pytest.mark.xfail(
    strict=True,
    # `raises=AssertionError` is load-bearing, and it was missing.
    #
    # Without it, ANY exception satisfies the xfail. Adversarial review proved that is not
    # theoretical: `result.output` begins with Hub warnings and a `Loading weights: 100%|...`
    # progress bar, so `json.loads(result.output)` raised JSONDecodeError and the test never
    # reached `assert result.exit_code != 0` -- the assertion its docstring exists for. The
    # reviewer then FIXED the underlying defect and re-ran: still XFAIL. The boomerang was
    # broken, i.e. fixing the bug would not have forced the marker's removal, which is the
    # entire point of a strict xfail.
    #
    # Pinning the type means a JSONDecodeError is now a hard failure, and only the intended
    # assertion can satisfy the marker.
    raises=AssertionError,
    reason=(
        "DEFECT, not a pinned behaviour: `Indexer.index_file()` catches every "
        "exception and RETURNS {'status': 'error', ...} — it never raises. So "
        "`update_index`'s `except Exception: raise typer.Exit(1)` is dead for every "
        "failure inside index_file, and the command prints an error payload on "
        "stdout with exit 0. `indexing/watcher.py` checks `result.get('status') == "
        "'ok'`; the CLI does not. Fix: exit non-zero when status != 'ok', then "
        "delete this xfail."
    ),
)
def test_update_index_exits_nonzero_when_the_file_could_not_be_indexed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MUTATION: none — this asserts the CORRECT behaviour, which does not hold yet.

    Kept as a strict xfail with `raises=AssertionError` so that fixing the defect turns it
    into an XPASS failure, forcing the marker's removal rather than leaving a stale skip.

    NO REAL WEIGHTS. `Indexer.index_file` is stubbed to return the error payload the real
    method returns on failure. Two reasons, both measured: this was the only test in either
    CLI file that loaded a real model (5.59s on a warm cache), and on a COLD cache under
    `--disable-socket` it would raise, satisfy a `raises`-less xfail, and report green
    either way — silently passing while testing nothing. The payload shape is what this
    test is about, not whether an embedder can be built.
    """
    repo = _repo_with_empty_index(tmp_path)

    # No real weights. `Indexer.__init__` calls `make_embedder(config.embedder)` at
    # indexer.py:385, so stubbing `index_file` is NOT enough -- the load happens during
    # construction. Patch the factory instead. Measured: 5.59s -> 0.4s, and the test stops
    # depending on a warm ~/.cache/huggingface.
    class _NoopEmbedder:
        dimension = 8

        def embed(self, texts: list[str]) -> list[list[float]]:
            return [[0.0] * 8 for _ in texts]

        def embed_query(self, text: str) -> list[float]:
            return [0.0] * 8

    monkeypatch.setattr("trelix.indexing.indexer.make_embedder", lambda _config: _NoopEmbedder())

    result = runner.invoke(
        app, ["update-index", str(repo), "does_not_exist.py", "--provider", "local"]
    )

    # Select the JSON payload LINE, not the whole stream and not "the last line".
    #
    # CliRunner folds the log stream into `output`, and the progress bar emits `\r`, which
    # `splitlines()` also splits on -- so both `json.loads(result.output)` and
    # `splitlines()[-1]` die on non-JSON. Exactly one line starts with "{", and the
    # precondition below asserts that, so a future CLI change emitting two JSON objects
    # fails loudly instead of silently parsing the wrong one.
    json_lines = [ln for ln in result.output.splitlines() if ln.lstrip().startswith("{")]
    assert len(json_lines) == 1, (
        f"expected exactly one JSON payload line, found {len(json_lines)}; this selector "
        f"can no longer identify the payload:\n{result.output}"
    )
    payload = json.loads(json_lines[0])
    assert payload["status"] == "error", result.output

    assert result.exit_code != 0, (
        "update-index printed a status=error payload and exited 0:\n" + result.output
    )
