"""CLI smoke tests — exercises every public command via CliRunner.

These tests verify that:
- The app starts, responds to --version and --help.
- Each subcommand exposes its --help without error.
- Error paths (missing args, bad paths) exit non-zero.
"""

from __future__ import annotations

import subprocess
import sys

from typer.testing import CliRunner

from trelix import __version__
from trelix.cli.main import _build_embedder_config, _print_error, app

runner = CliRunner()


def test_main_module_reconfigures_stdout_to_utf8_on_legacy_codepage():
    """Regression test: importing trelix.cli.main must upgrade a legacy-codepage stream.

    Windows' default console codepage (cp1252 etc.) can't encode the
    Unicode braille glyphs Rich's spinner renders (e.g. U+280B), crashing
    with "'charmap' codec can't encode character ...". Reproduced via a
    real PyInstaller-built binary in CI. The module-level reconfigure()
    call must run before any rich.Console is constructed, so this test
    simulates a legacy stdout in a fresh subprocess (importing in-process
    would be a no-op, since trelix.cli.main is already imported above).
    """
    code = (
        "import io, sys\n"
        "sys.stdout = io.TextIOWrapper(io.BytesIO(), encoding='cp1252', "
        "errors='strict', write_through=True)\n"
        "import trelix.cli.main\n"
        "print(sys.stdout.encoding, sys.stdout.errors, file=sys.stderr)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    assert "utf-8 replace" in result.stderr


def test_print_error_preserves_literal_brackets(capsys):
    """Regression test: exception text with brackets must render intact.

    Rich interprets bracketed substrings as markup tags, so an exception
    like ImportError("... pip install 'trelix[local]'") rendered via
    f"[red]...{exc}[/red]" silently stripped the literal "[local]" —
    telling the user to run the exact broken command they'd already run.
    _print_error must escape() the dynamic text so it survives verbatim.
    """
    exc = ImportError("Install it with: pip install 'trelix[local]'")
    _print_error("Indexing failed", exc)
    captured = capsys.readouterr()
    assert "trelix[local]" in captured.err


def test_build_embedder_config_honors_env_var_when_no_flag(monkeypatch):
    """Regression test: --provider must default to None, not "local".

    EmbedderConfig is a pydantic-settings model — an explicit constructor
    kwarg always wins over TRELIX_EMBEDDER_PROVIDER. Every CLI command used
    to pass provider="local" unconditionally (Typer's declared default),
    which silently overrode the env var even when the user never touched
    --provider, contradicting the documented "env var, overridden by
    --provider" contract.
    """
    monkeypatch.setenv("TRELIX_EMBEDDER_PROVIDER", "openai")
    assert _build_embedder_config(None).provider == "openai"


def test_build_embedder_config_explicit_flag_overrides_env_var(monkeypatch):
    monkeypatch.setenv("TRELIX_EMBEDDER_PROVIDER", "openai")
    assert _build_embedder_config("local").provider == "local"


def test_build_embedder_config_defaults_to_local_when_nothing_set(monkeypatch):
    # pydantic-settings reads .env directly, so an empty shell env isn't
    # enough to isolate this — monkeypatch.setenv (not delenv) is required
    # to actually override whatever a developer's local .env has set,
    # matching conftest.py's _isolate_beast_mode_flags convention.
    monkeypatch.setenv("TRELIX_EMBEDDER_PROVIDER", "local")
    assert _build_embedder_config(None).provider == "local"


def test_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "trelix" in result.output
    assert __version__ in result.output


def test_version_short_flag():
    result = runner.invoke(app, ["-V"])
    assert result.exit_code == 0
    assert "trelix" in result.output
    assert __version__ in result.output


def test_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "index" in result.output


def test_index_help():
    result = runner.invoke(app, ["index", "--help"])
    assert result.exit_code == 0
    # Strip ANSI codes before asserting — CliRunner with color enabled wraps flags
    import re

    plain = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
    assert "--provider" in plain


def test_search_help():
    result = runner.invoke(app, ["search", "--help"])
    assert result.exit_code == 0


def test_ask_help():
    result = runner.invoke(app, ["ask", "--help"])
    assert result.exit_code == 0


def test_stats_help():
    result = runner.invoke(app, ["stats", "--help"])
    assert result.exit_code == 0


def test_stats_configures_logging():
    """Regression test: `stats` previously never called _setup_logging()
    at all, despite doing real DB I/O with its own logger.* call sites —
    those log records went to Python's bare logging.lastResort fallback
    instead of trelix's structured console formatter, an inconsistency
    every other command (index/search/graph/review/...) doesn't have."""
    import tempfile
    from pathlib import Path
    from unittest.mock import patch

    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "repo"
        repo.mkdir()
        with patch("trelix.cli.main._setup_logging") as mock_setup:
            runner.invoke(app, ["stats", str(repo)])
        mock_setup.assert_called_once()


def test_query_help():
    result = runner.invoke(app, ["query", "--help"])
    assert result.exit_code == 0


def test_watch_help():
    result = runner.invoke(app, ["watch", "--help"])
    assert result.exit_code == 0


def test_update_index_help():
    result = runner.invoke(app, ["update-index", "--help"])
    assert result.exit_code == 0


def test_migrate_vectors_help():
    result = runner.invoke(app, ["migrate-vectors", "--help"])
    assert result.exit_code == 0


def test_link_tickets_help():
    result = runner.invoke(app, ["link-tickets", "--help"])
    assert result.exit_code == 0
    import re

    plain = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
    assert "--max-commits" in plain
    assert "--since" in plain
    assert "--ticket-pattern" in plain


def test_link_tickets_requires_path():
    result = runner.invoke(app, ["link-tickets"])
    assert result.exit_code != 0


def test_link_tickets_no_index_found():
    """trelix link-tickets on a repo that was never indexed exits 1 with a
    helpful message, same as trelix stats on an unindexed repo."""
    import subprocess
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "repo"
        repo.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
        result = runner.invoke(app, ["link-tickets", str(repo)])
        assert result.exit_code == 1


def test_link_tickets_links_real_ticket_reference():
    """End-to-end: a real git repo with a ticket-referencing commit, and a
    manually-seeded index DB (bypassing the embedder dependency, which this
    command never touches), produces a real GenericEdge."""
    import subprocess
    import tempfile
    from pathlib import Path

    from trelix.core.config import IndexConfig
    from trelix.core.models import IndexedFile, Language, Symbol, SymbolKind
    from trelix.store.db import Database

    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "repo"
        repo.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
        subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=repo, check=True)
        subprocess.run(["git", "config", "user.name", "T"], cwd=repo, check=True)
        (repo / "auth.py").write_text("def login():\n    pass\n")
        subprocess.run(["git", "add", "auth.py"], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "PROJ-1: add login"], cwd=repo, check=True)

        config = IndexConfig(repo_path=str(repo))
        db = Database(config.db_path_absolute)
        file_id = db.upsert_file(
            IndexedFile(
                path=str(repo / "auth.py"),
                rel_path="auth.py",
                language=Language.PYTHON,
                hash="h",
                size_bytes=10,
            )
        )
        sym_id = db.insert_symbol(
            Symbol(
                file_id=file_id,
                name="login",
                qualified_name="login",
                kind=SymbolKind.FUNCTION,
                line_start=1,
                line_end=2,
                signature="def login()",
                body="def login(): pass",
            )
        )
        db._conn.commit()

        result = runner.invoke(app, ["link-tickets", str(repo)])
        assert result.exit_code == 0
        assert "Linked 1 symbol-ticket edge" in result.output
        assert db.get_generic_edge_targets(sym_id) == ["ticket:PROJ-1"]


def test_index_nonexistent_path():
    result = runner.invoke(app, ["index", "/nonexistent/path/xyz"])
    assert result.exit_code != 0


def test_search_requires_path():
    # Missing both repo and query positional args
    result = runner.invoke(app, ["search"])
    assert result.exit_code != 0


def test_search_requires_query():
    # Repo provided but query missing
    result = runner.invoke(app, ["search", "/some/repo"])
    assert result.exit_code != 0


def test_ask_requires_args():
    result = runner.invoke(app, ["ask"])
    assert result.exit_code != 0


def test_stats_requires_path():
    result = runner.invoke(app, ["stats"])
    assert result.exit_code != 0


def test_watch_all_help() -> None:
    """trelix watch-all --help exits 0 and shows expected options."""
    from typer.testing import CliRunner

    from trelix.cli.main import app

    runner = CliRunner()
    result = runner.invoke(app, ["watch-all", "--help"])
    assert result.exit_code == 0
    assert "watch-all" in result.output.lower() or "registry" in result.output.lower()


def test_eval_help():
    result = runner.invoke(app, ["eval", "--help"])
    assert result.exit_code == 0


def test_eval_synthesis_help():
    result = runner.invoke(app, ["eval-synthesis", "--help"])
    assert result.exit_code == 0
    assert "--golden" in result.output or "golden" in result.output.lower()


def test_watch_all_no_repos_exits_gracefully() -> None:
    """trelix watch-all with empty registry shows helpful message."""
    from unittest.mock import patch

    from typer.testing import CliRunner

    from trelix.cli.main import app
    from trelix.federation.registry import RepoRegistry

    runner = CliRunner()
    empty_reg = RepoRegistry.__new__(RepoRegistry)
    from pathlib import Path

    empty_reg._config_path = Path("/tmp/fake.json")
    empty_reg._entries = []

    with patch("trelix.cli.main.RepoRegistry.load", return_value=empty_reg):
        result = runner.invoke(app, ["watch-all"])
    assert result.exit_code == 0
    assert "no repos" in result.output.lower() or "register" in result.output.lower()


def test_federation_remove_help() -> None:
    result = runner.invoke(app, ["federation", "remove", "--help"])
    assert result.exit_code == 0


def test_federation_remove_missing_alias_exits_gracefully() -> None:
    """trelix federation remove <alias> with an empty registry is a graceful no-op."""
    from pathlib import Path
    from unittest.mock import patch

    from trelix.federation.registry import RepoRegistry

    empty_reg = RepoRegistry.__new__(RepoRegistry)
    empty_reg._config_path = Path("/tmp/fake.json")
    empty_reg._entries = []

    with patch("trelix.cli.main.RepoRegistry.load", return_value=empty_reg):
        result = runner.invoke(app, ["federation", "remove", "nonexistent"])
    assert result.exit_code == 0
    assert "No repo registered" in result.output


def test_ask_session_flag_help() -> None:
    result = runner.invoke(app, ["ask", "--help"])
    assert result.exit_code == 0
    # Strip ANSI codes before asserting — CliRunner with color enabled wraps
    # flags in styling spans that split literal substring matches.
    import re

    plain = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
    assert "--session" in plain


def test_agent_sessions_list_help() -> None:
    result = runner.invoke(app, ["agent", "sessions", "list", "--help"])
    assert result.exit_code == 0


def test_agent_sessions_show_help() -> None:
    result = runner.invoke(app, ["agent", "sessions", "show", "--help"])
    assert result.exit_code == 0


def test_agent_sessions_clear_help() -> None:
    result = runner.invoke(app, ["agent", "sessions", "clear", "--help"])
    assert result.exit_code == 0


def test_connector_sync_help():
    result = runner.invoke(app, ["connector", "sync", "--help"])
    assert result.exit_code == 0


def test_connector_sync_unknown_name_exits_nonzero(tmp_path):  # type: ignore[no-untyped-def]
    from trelix.store.db import Database

    Database(tmp_path / ".trelix" / "index.db")
    result = runner.invoke(app, ["connector", "sync", str(tmp_path), "bogus"])
    assert result.exit_code != 0


def test_connector_sync_no_index_found(tmp_path):  # type: ignore[no-untyped-def]
    result = runner.invoke(app, ["connector", "sync", str(tmp_path), "jira"])
    assert result.exit_code != 0


def test_connector_sync_missing_config_exits_nonzero(tmp_path, monkeypatch):  # type: ignore[no-untyped-def]
    """No TRELIX_JIRA_* env vars set — validate_config() must fail fast
    with a clear message, before ever making an HTTP call.

    setenv("", ...) rather than delenv(): a developer's .env may have real
    Jira credentials (e.g. for live connector testing), and
    pydantic-settings reads that file directly regardless of the process
    environment — delenv() only clears the latter, so the .env value would
    still surface here. Overriding to "" (falsy, same as unset for
    validate_config()'s `if not val` checks) is what actually neutralizes
    it, same fix as tests/unit/conftest.py's _EMPTY_STRING_BY_DEFAULT."""
    from trelix.core.config import IndexConfig
    from trelix.store.db import Database

    for var in (
        "TRELIX_JIRA_BASE_URL",
        "TRELIX_JIRA_EMAIL",
        "TRELIX_JIRA_API_TOKEN",
        "TRELIX_JIRA_PROJECT_KEY",
    ):
        monkeypatch.setenv(var, "")

    config = IndexConfig(repo_path=str(tmp_path))
    Database(config.db_path_absolute)

    result = runner.invoke(app, ["connector", "sync", str(tmp_path), "jira"])
    assert result.exit_code != 0


def test_connector_sync_jira_end_to_end(tmp_path, monkeypatch):  # type: ignore[no-untyped-def]
    """Real CLI invocation, mocked Jira HTTP response, real DB round-trip."""
    from unittest.mock import MagicMock, patch

    from trelix.core.config import IndexConfig
    from trelix.store.db import Database

    config = IndexConfig(repo_path=str(tmp_path))
    db = Database(config.db_path_absolute)

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "issues": [
            {
                "key": "PROJ-1",
                "fields": {
                    "summary": "Fix login",
                    "description": "bug",
                    "status": {"name": "Open"},
                },
            }
        ],
        "nextPageToken": None,
    }

    monkeypatch.setenv("TRELIX_JIRA_BASE_URL", "https://example.atlassian.net")
    monkeypatch.setenv("TRELIX_JIRA_EMAIL", "me@example.com")
    monkeypatch.setenv("TRELIX_JIRA_API_TOKEN", "tok")
    monkeypatch.setenv("TRELIX_JIRA_PROJECT_KEY", "PROJ")

    with patch("trelix.indexing.connectors.jira.httpx.get", return_value=mock_resp):
        result = runner.invoke(app, ["connector", "sync", str(tmp_path), "jira"])

    assert result.exit_code == 0
    assert "fetched 1" in result.output
    fetched = db.get_artifact_by_source_ref("ticket:PROJ-1")
    assert fetched is not None
    assert fetched.title == "Fix login"


class TestTaintRulesOption:
    """`trelix taint --rules` exposes TaintAnalyzer's existing rules_path.

    Without it the CLI was pinned to `--config p/default`, the Semgrep registry pack:
    it needs outbound network access, and its contents can change between runs, so a
    taint result was neither reproducible offline nor pinnable in CI. The library
    already accepted a rules path; only the CLI did not pass one.
    """

    def test_rules_path_is_forwarded_to_the_analyzer(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        from unittest.mock import patch

        rules = tmp_path / "rules.yaml"
        rules.write_text("rules: []\n", encoding="utf-8")

        with patch("trelix.analysis.taint.TaintAnalyzer.run", return_value=[]) as mock_run:
            result = runner.invoke(app, ["taint", str(tmp_path), "--rules", str(rules)])

        assert result.exit_code == 0, result.output
        assert mock_run.call_args.kwargs["rules_path"] == str(rules.resolve())

    def test_omitting_rules_keeps_the_registry_default(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """None must be passed through so TaintAnalyzer picks its own default."""
        from unittest.mock import patch

        with patch("trelix.analysis.taint.TaintAnalyzer.run", return_value=[]) as mock_run:
            result = runner.invoke(app, ["taint", str(tmp_path)])

        assert result.exit_code == 0, result.output
        assert mock_run.call_args.kwargs["rules_path"] is None

    def test_missing_rules_file_fails_fast(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """A typo in the path must not silently fall back to the network pack.

        Falling back would report registry findings under the reader's assumption that
        their own rules produced them.
        """
        result = runner.invoke(
            app, ["taint", str(tmp_path), "--rules", str(tmp_path / "nope.yaml")]
        )

        assert result.exit_code == 1
        assert "Rules not found" in result.output


class TestGraphVisualizeWithJson:
    """`--visualize` must not be silently ignored when `--json` is also passed.

    The graph command's `if json_output:` branch printed the payload and RETURNED
    before the `if visualize:` block that exports the pyvis HTML. So the flag was
    accepted, no file was written, and nothing said so.

    Confirmed on this repository: `trelix graph . --visualize --json` left a stale
    31-byte graph.html from a previous run untouched, while the same command without
    `--json` wrote 326 KB.

    The pyvis export is now performed before the early return, and its path is
    reported in the JSON payload so a machine consumer learns where the file went.
    Adding a key is additive — the four documented keys keep their names and types,
    and the new one appears only when the flag is passed.
    """

    @staticmethod
    def _patched_builder():  # type: ignore[no-untyped-def]
        from unittest.mock import MagicMock, patch

        result = MagicMock(
            node_count=10, edge_count=20, community_count=3, concept_count=0,
            elapsed_seconds=0.5, community_summary=[], code_graph=MagicMock(),
        )
        builder = patch("trelix.graph.builder.GraphBuilder")
        return builder, result

    def test_html_is_written_when_json_is_also_requested(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        from unittest.mock import patch

        out = tmp_path / "graph.html"
        builder_patch, result = self._patched_builder()

        with builder_patch as MockBuilder, patch(
            "trelix.graph.visualizer.GraphVisualizer"
        ) as MockViz:
            MockBuilder.return_value.build.return_value = result
            MockViz.return_value.export_html.return_value = str(out)
            res = runner.invoke(
                app, ["graph", str(tmp_path), "--visualize", "--json", "--output", str(out)]
            )

        assert res.exit_code == 0, res.output
        assert MockViz.return_value.export_html.called, (
            "--visualize was ignored because --json returned first"
        )

    def test_json_payload_reports_the_written_path(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        import json as _json
        from unittest.mock import patch

        out = tmp_path / "graph.html"
        builder_patch, result = self._patched_builder()

        with builder_patch as MockBuilder, patch(
            "trelix.graph.visualizer.GraphVisualizer"
        ) as MockViz:
            MockBuilder.return_value.build.return_value = result
            MockViz.return_value.export_html.return_value = str(out)
            res = runner.invoke(
                app, ["graph", str(tmp_path), "--visualize", "--json", "--output", str(out)]
            )

        payload = _json.loads(res.stdout)
        assert payload.get("visualization_path") == str(out)

    def test_the_four_documented_keys_are_unchanged(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """The documented JSON contract must survive; the new key is additive only."""
        import json as _json
        from unittest.mock import patch

        builder_patch, result = self._patched_builder()
        with builder_patch as MockBuilder:
            MockBuilder.return_value.build.return_value = result
            res = runner.invoke(app, ["graph", str(tmp_path), "--json"])

        payload = _json.loads(res.stdout)
        assert set(payload) == {"node_count", "edge_count", "community_count", "concept_count"}, (
            "plain --json output gained or lost a key"
        )
        assert payload["node_count"] == 10


class TestTaintCleanScanMessage:
    """A clean scan must not be reported as a possible missing install.

    `TaintAnalyzer.run()` returns `[]` both when semgrep is absent and when semgrep
    ran fine and found nothing, so the CLI could not tell the two apart and printed
    one message for both: "No taint flows found. Ensure semgrep is installed: pip
    install trelix[taint]".

    Verified on this repository: semgrep IS installed, it scanned all 140 source files
    with `config/semgrep-taint.yaml` and reported 0 findings and 0 errors, and the CLI
    still suggested installing it. A security tool that reports a clean result as a
    possible misconfiguration teaches people to distrust its output.

    The comment above that message already records two previously-fixed defects in it
    — Rich swallowing "[taint]" as a markup tag, and prose going to stdout under
    `--json`. This is a third, so both of those behaviours are pinned here too.
    """

    def test_clean_scan_does_not_suggest_installing_semgrep(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        from unittest.mock import patch

        with patch("trelix.analysis.taint.TaintAnalyzer.run", return_value=[]), patch(
            "shutil.which", return_value="/usr/local/bin/semgrep"
        ):
            res = runner.invoke(app, ["taint", str(tmp_path)])

        assert res.exit_code == 0, res.output
        assert "pip install" not in res.output, (
            f"a clean scan told the user to install semgrep: {res.output!r}"
        )

    def test_missing_semgrep_still_says_how_to_install_it(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """The install hint must survive for the case it was written for."""
        from unittest.mock import patch

        with patch("trelix.analysis.taint.TaintAnalyzer.run", return_value=[]), patch(
            "shutil.which", return_value=None
        ):
            res = runner.invoke(app, ["taint", str(tmp_path)])

        assert res.exit_code == 0, res.output
        assert "pip install" in res.output

    def test_install_hint_is_not_swallowed_as_markup(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """Regression on an already-fixed defect: Rich reads "[taint]" as a tag.

        Unescaped, the hint renders as "pip install trelix" — telling the reader to
        install the package they already have.
        """
        from unittest.mock import patch

        with patch("trelix.analysis.taint.TaintAnalyzer.run", return_value=[]), patch(
            "shutil.which", return_value=None
        ):
            res = runner.invoke(app, ["taint", str(tmp_path)])

        assert "trelix[taint]" in res.output, (
            f"the extras marker was swallowed by Rich markup: {res.output!r}"
        )

    def test_json_mode_stays_machine_readable_on_the_empty_path(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """Regression on an already-fixed defect: prose on stdout broke `| jq`."""
        import json as _json
        from unittest.mock import patch

        for which in ("/usr/local/bin/semgrep", None):
            with patch("trelix.analysis.taint.TaintAnalyzer.run", return_value=[]), patch(
                "shutil.which", return_value=which
            ):
                res = runner.invoke(app, ["taint", str(tmp_path), "--json"])

            assert _json.loads(res.stdout) == [], (
                f"--json stdout was not parseable with which={which!r}: {res.stdout!r}"
            )
