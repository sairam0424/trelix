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
    assert "2.2.0" in result.output


def test_version_short_flag():
    result = runner.invoke(app, ["-V"])
    assert result.exit_code == 0
    assert "trelix" in result.output
    assert "2.2.0" in result.output


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
