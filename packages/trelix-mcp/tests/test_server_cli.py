"""Tests for trelix_mcp.server's console-script argv handling.

Before argparse was wired in, main() never touched sys.argv at all, so `trelix-mcp
--help`, `-h`, `--version`, and any unrecognized flag all silently launched the real
MCP server on stdio instead of doing what a CLI user expects — confirmed live on the
published PyPI package. These tests pin the CLI contract at the boundary that matters:
whichever branch runs, does it call mcp.run(transport="stdio"), and does argv make it
there unparsed. The no-args path (how every real MCP client launches this) must be
untouched by the new parser.

Patches sys.argv directly rather than any server_module attribute, since argparse
reads the real sys.argv — patching a module-local copy would not exercise the same
code path a real `trelix-mcp ...` invocation does.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import trelix_mcp.server as server_module


def test_no_args_starts_the_server(monkeypatch: pytest.MonkeyPatch) -> None:
    """The normal MCP-client launch path (no argv) must fall straight through."""
    monkeypatch.setattr("sys.argv", ["trelix-mcp"])
    mock_run = MagicMock()
    monkeypatch.setattr(server_module.mcp, "run", mock_run)

    server_module.main()

    mock_run.assert_called_once_with(transport="stdio")


@pytest.mark.parametrize("flag", ["--help", "-h"])
def test_help_prints_usage_and_does_not_start_the_server(
    flag: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """--help/-h must print usage and exit 0 — not silently launch the server."""
    monkeypatch.setattr("sys.argv", ["trelix-mcp", flag])
    mock_run = MagicMock()
    monkeypatch.setattr(server_module.mcp, "run", mock_run)

    with pytest.raises(SystemExit) as exc_info:
        server_module.main()

    assert exc_info.value.code == 0
    mock_run.assert_not_called()
    captured = capsys.readouterr()
    assert "trelix-mcp" in captured.out


def test_version_prints_the_real_version_and_does_not_start_the_server(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """--version must print the actual package version and exit 0, never starting the server."""
    monkeypatch.setattr("sys.argv", ["trelix-mcp", "--version"])
    mock_run = MagicMock()
    monkeypatch.setattr(server_module.mcp, "run", mock_run)

    with pytest.raises(SystemExit) as exc_info:
        server_module.main()

    assert exc_info.value.code == 0
    mock_run.assert_not_called()
    captured = capsys.readouterr()
    assert server_module.__version__ in captured.out


def test_unknown_flag_is_rejected_and_does_not_start_the_server(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An unrecognized flag must error out (non-zero exit), never launching the server.

    This is the exact silent-launch bug: previously any typo'd flag fell straight
    through to mcp.run(transport="stdio") because main() never looked at sys.argv.
    """
    monkeypatch.setattr("sys.argv", ["trelix-mcp", "--bogus-flag-xyz"])
    mock_run = MagicMock()
    monkeypatch.setattr(server_module.mcp, "run", mock_run)

    with pytest.raises(SystemExit) as exc_info:
        server_module.main()

    assert exc_info.value.code != 0
    mock_run.assert_not_called()
    captured = capsys.readouterr()
    assert "trelix-mcp" in captured.err
