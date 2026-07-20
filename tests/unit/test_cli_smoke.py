"""CLI smoke tests — exercises every public command via CliRunner.

These tests verify that:
- The app starts, responds to --version and --help.
- Each subcommand exposes its --help without error.
- Error paths (missing args, bad paths) exit non-zero.
"""

from __future__ import annotations

from typer.testing import CliRunner

from trelix.cli.main import app

runner = CliRunner()


def test_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "trelix" in result.output
    assert "2.8.1" in result.output


def test_version_short_flag():
    result = runner.invoke(app, ["-V"])
    assert result.exit_code == 0
    assert "trelix" in result.output
    assert "2.8.1" in result.output


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
